import gzip
import json
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_flaccs_experiment as runner
from src.consensus import AdaptiveSeedPolicy
from src.data.loader import CANONICAL_COLUMNS, STUDY_LABELS
from src.inference import flaccs
from src.utils.journal import Journal

ROOT = Path(__file__).resolve().parents[1]
FLACCS_LABELS = tuple(l for l in STUDY_LABELS if l != "others")


def _experiment(assignments_path: Path) -> dict:
    return {
        "version": 1, "id": "rq4_artifact_flaccs_fixture", "research_question": "rq4_2",
        "shared": {
            "dataset": "unused", "representation": "message_and_diff",
            "initial_seeds": [10, 20, 30], "adaptive_policy": {"start": 40, "step": 10},
            "consensus": {"method": "first_to_two", "threshold": 2},
        },
        "subset": {
            "assignments": str(assignments_path), "label_space": list(FLACCS_LABELS),
            "context_window_tokens": 1024,
        },
        "conditions": {
            "flaccs": {
                "model": "0x404/ccs-code-llama-7b", "prompt_id": "p",
                "inference": {"do_sample": False, "max_new_tokens": 10,
                              "message_max_tokens": 64, "context_window_tokens": 1024},
            },
            "gpt_oss_20b": {"model": "gpt_oss_20b", "prompt_id": "other", "temperature": 0.6},
        },
    }


@pytest.fixture
def execution(tmp_path, monkeypatch):
    rows = []
    for key in ("a", "b"):
        row = dict.fromkeys(CANONICAL_COLUMNS, "")
        row.update(project="p", commit_hash=key, target="fix", partition="pre_cutoff",
                   processed_message=key, diff="diff", git_show=key)
        rows.append(row)
    dataset = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(dataset, index=False)

    assignments = tmp_path / "assignments.csv"
    assignments.write_text(
        "protocol,project,commit_hash,canonical_target,protocol_target\n"
        "flaccs_compatible,p,a,fix,fix\n"
        "flaccs_compatible,p,b,fix,fix\n"
    )

    prompt = tmp_path / "prompt.yaml"
    prompt.write_text(
        '- role: user\n  activate: true\n  content: |\n'
        '    head {{COMMIT_MESSAGE}} mid {{COMMIT_DIFF}} tail\n'
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("prompts:\n  p:\n    file: prompt.yaml\n")

    experiment = _experiment(assignments)
    experiment["shared"]["dataset"] = str(dataset)
    config = tmp_path / "config.yaml"
    config.write_text("version: 1\n")

    calls = []
    fail = set()
    votes = {}

    class FakeBackend:
        def __init__(self, config): self.config = config

        def build_prompt(self, *, template, message, diff, message_max_tokens, context_window_tokens):
            return f"{template}::{message}::{diff}"

        def infer(self, *, prompt, seed, task_labels):
            key = (prompt.split("::")[1], seed)
            calls.append(key)
            if key in fail:
                raise RuntimeError("mock interrupted call")
            label = votes.get(key, "fix")
            parsed = types.SimpleNamespace(prediction=label, parse_status="mock", parser_label=label,
                                            parser_raw_label=label, normalization_rule="")
            return flaccs.FlaccsCallResult(prompt, parsed, "ok", "", 0.0)

    monkeypatch.setattr(flaccs, "FlaccsBackend", FakeBackend)

    evaluation = {"taxonomy": {"labels": STUDY_LABELS}}
    jobs = runner.expand_jobs(experiment, evaluation=evaluation)
    assert len(jobs) == 1
    job = jobs[0]

    kw = dict(repo_root=tmp_path, job=job, output_base=tmp_path / "runs",
              experiment_path=config, evaluation_path=config, prompt_manifest=manifest,
              resume=False, dry_run=False, limit=None)
    return types.SimpleNamespace(kw=kw, job=job, calls=calls, fail=fail, votes=votes,
                                  directory=tmp_path / "runs/rq4_artifact_flaccs_fixture/flaccs",
                                  repo_root=tmp_path, config=config)


def run(e, **overrides):
    runner._run_job(**(e.kw | overrides))


def saved(e):
    return Journal(e.directory / "raw_predictions.journal.jsonl").read()[0]


def test_expand_jobs_skips_non_inference_conditions(execution):
    e = execution
    assert e.job.condition_id == "flaccs"
    assert e.job.task_labels == FLACCS_LABELS
    assert e.job.do_sample is False
    assert e.job.message_max_tokens == 64 and e.job.context_window_tokens == 1024


def test_dry_run_is_offline_and_does_not_write(execution):
    run(execution, dry_run=True)
    assert not execution.calls and not execution.directory.exists()


def test_basic_run_and_resume(execution):
    e = execution
    e.fail.add(("b", 10))
    with pytest.raises(RuntimeError, match="mock interrupted"):
        run(e)
    prior = saved(e)
    assert [(r["commit_hash"], r["seed"]) for r in prior] == [("a", 10)]
    assert json.loads((e.directory / "run_manifest.json").read_text())["status"] == "incomplete"
    e.calls.clear(); e.fail.clear()
    run(e, resume=True)
    assert e.calls == [("b", 10), ("a", 20), ("b", 20), ("a", 30), ("b", 30)]
    assert saved(e)[0] == prior[0]
    manifest = json.loads((e.directory / "run_manifest.json").read_text())
    assert manifest["status"] == "completed" and len(manifest["resume_events"]) == 1
    final = pd.read_csv(e.directory / "predictions.csv.gz")
    assert sorted(final.commit_hash.tolist()) == ["a", "b"]
    assert final.prediction.tolist() == ["fix", "fix"]
    e.calls.clear()
    run(e, resume=True)
    assert not e.calls


def test_adaptive_seed_progression(execution):
    e = execution
    # Commit "a" disagrees across all three initial seeds, forcing an adaptive
    # seed 40 that matches seed 10's vote to reach the first-to-two threshold.
    e.votes["a", 10] = "feat"; e.votes["a", 20] = "docs"; e.votes["a", 30] = "test"
    e.votes["a", 40] = "feat"
    run(e)
    assert ("a", 40) in e.calls and ("b", 40) not in e.calls
    final = pd.read_csv(e.directory / "predictions.csv.gz").set_index("commit_hash")
    assert final.loc["a", "prediction"] == "feat" and final.loc["a", "decisive_seed"] == 40
    assert final.loc["b", "prediction"] == "fix" and final.loc["b", "decisive_seed"] == 30


def test_invalid_terminality(execution):
    e = execution
    e.votes["a", 10] = e.votes["a", 20] = "<invalid_output>"
    run(e)
    assert ("a", 30) in e.calls and all(seed <= 30 for _, seed in e.calls)
    final = pd.read_csv(e.directory / "predictions.csv.gz").set_index("commit_hash")
    assert final.loc["a", "prediction"] == "<invalid_output>"
    assert final.loc["b", "prediction"] == "fix"


def test_results_protected(execution):
    with pytest.raises(runner.ConfigError):
        run(execution, output_base=execution.repo_root / "results")


def test_existing_directory_without_resume(execution):
    e = execution
    e.directory.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Existing job directory"):
        run(e)


def test_identity_rejection_preserves_manifest(execution):
    e = execution
    run(e)
    before = (e.directory / "run_manifest.json").read_bytes()
    e.config.write_text("changed")
    with pytest.raises(RuntimeError, match="already contains a different run"):
        run(e, resume=True)
    assert (e.directory / "run_manifest.json").read_bytes() == before


def test_missing_placeholder_prompt_rejected(execution, tmp_path):
    e = execution
    bad_prompt = e.repo_root / "prompt.yaml"
    bad_prompt.write_text('- role: user\n  activate: true\n  content: "no placeholders here"\n')
    with pytest.raises(runner.ConfigError, match="missing"):
        run(e, dry_run=True)
