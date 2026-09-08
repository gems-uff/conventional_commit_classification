import asyncio
import gzip
import json
import sys
import subprocess
import types
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_ollama_experiment as runner
from src.config import load_evaluation_config, load_experiment, load_inference_config, load_models
from src.consensus import AdaptiveSeedPolicy
from src.data.loader import CANONICAL_COLUMNS, STUDY_LABELS
from src.inference import ollama
from src.prompts.loader import PromptRegistry
from src.utils.journal import Journal

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def execution(tmp_path, monkeypatch):
    # This is an orchestration double, never a substitute canonical parser file.
    monkeypatch.setitem(sys.modules, "src.parsing.strict", types.SimpleNamespace(PARSER_VERSION="4.1.1"))
    rows = []
    for key in ("a", "b"):
        row = dict.fromkeys(CANONICAL_COLUMNS, "")
        row.update(project="p", commit_hash=key, target="fix", partition="pre_cutoff",
                   processed_message=key, diff="diff", git_show=key)
        rows.append(row)
    dataset = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(dataset, index=False)
    config = tmp_path / "config.yaml"
    config.write_text("version: 1\n")
    (tmp_path / "src/parsing").mkdir(parents=True)
    (tmp_path / "src/parsing/strict_response_parser.py").write_text("# identity fixture only\n")
    prompt = tmp_path / "prompt.yaml"
    prompt.write_text('- role: user\n  activate: true\n  content: "{{COMMIT_INPUT}}"\n')
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("prompts:\n  p:\n    file: prompt.yaml\n")
    calls = []
    fail = set()
    votes = {}

    class Backend:
        def __init__(self, config): self.config = config
        def prepare_content(self, text): return text, len(text)
        async def infer(self, **kw):
            key = (kw["prepared_input"].split("\n")[0], kw["seed"])
            calls.append(key)
            if key in fail:
                raise RuntimeError("mock interrupted call")
            label = votes.get(key, "fix")
            kw["attempt_sink"]({"attempt": 1, "raw_response": label, "technical_status": "ok",
                                "technical_detail": "", "processing_time_seconds": 0})
            parsed = types.SimpleNamespace(prediction=label, parse_status="mock", parser_label=label,
                                           parser_raw_label=label, normalization_rule="")
            return ollama.OllamaCallResult(label, parsed, "ok", "", 0, 1, 4096)

    monkeypatch.setattr(ollama, "OllamaBackend", Backend)
    monkeypatch.setattr(ollama, "installed_model_identity", lambda *a: {"tag": "mock", "digest": "d"})
    job = runner.OllamaJob("e", "rq", "c", "m", str(dataset), "p", "message_and_diff", .6,
                           (10, 20, 30), AdaptiveSeedPolicy(), 2, STUDY_LABELS, 300)
    cfg = load_inference_config(ROOT / "config/inference.yaml")
    cfg["execution"]["batch_size"] = 2
    kw = dict(repo_root=tmp_path, job=job, output_base=tmp_path / "runs", prompt_manifest=manifest,
              inference_cfg=cfg, model_spec={"ollama_tag": "mock", "digest": "d"},
              experiment_path=config, inference_path=config, evaluation_path=config, models_path=config,
              resume=False, dry_run=False, project_filter=None, limit=None, max_concurrent_override=2)
    return types.SimpleNamespace(kw=kw, calls=calls, fail=fail, votes=votes,
                                 directory=tmp_path / "runs/e/c/m", config=config)


def run(e, **overrides):
    asyncio.run(runner._run_job(**(e.kw | overrides)))


def saved(e):
    return Journal(e.directory / "raw_predictions.journal.jsonl").read()[0]


def test_base_resume_mid_batch_and_snapshot_recovery(execution):
    e = execution
    e.fail.add(("b", 10))
    with pytest.raises(RuntimeError, match="mock interrupted"):
        run(e)
    prior = saved(e)
    assert [(r["commit_hash"], r["seed"]) for r in prior] == [("a", 10)]
    assert json.loads((e.directory / "run_manifest.json").read_text())["status"] == "incomplete"
    original_start = json.loads((e.directory / "run_manifest.json").read_text())["started_at_utc"]
    (e.directory / "raw_predictions.jsonl.gz").write_bytes(b"truncated gzip")
    e.calls.clear(); e.fail.clear()
    run(e, resume=True)
    assert e.calls == [("b", 10), ("a", 20), ("b", 20), ("a", 30), ("b", 30)]
    assert saved(e)[0] == prior[0]
    manifest = json.loads((e.directory / "run_manifest.json").read_text())
    assert manifest["started_at_utc"] == original_start and len(manifest["resume_events"]) == 1
    assert manifest["temperature"] == .6 and manifest["num_predict"] == 1024
    assert len(gzip.decompress((e.directory / "raw_predictions.jsonl.gz").read_bytes()).splitlines()) == 6
    e.calls.clear()
    run(e, resume=True)
    assert not e.calls


def test_resume_before_first_completed_call(execution):
    e = execution
    e.fail.update({("a", 10), ("b", 10)})
    with pytest.raises(RuntimeError): run(e)
    assert saved(e) == []
    e.fail.clear(); e.calls.clear()
    run(e, resume=True)
    assert len(e.calls) == 6 and len(saved(e)) == 6


def test_adaptive_resume_different_stages_and_late_records(execution):
    e = execution
    for key in ("a", "b"):
        for seed, vote in ((10, "fix"), (20, "feat"), (30, "docs")):
            e.votes[key, seed] = vote
    e.votes["b", 40] = "test"
    e.fail.add(("b", 50))
    with pytest.raises(RuntimeError): run(e)
    prior = saved(e)
    assert len(prior) == 8
    # Preserve a later response without letting it override a's terminal seed 40.
    late = dict(next(r for r in prior if r["commit_hash"] == "a"))
    late.update(seed=100, prediction="feat")
    Journal(e.directory / "raw_predictions.journal.jsonl").append(late)
    e.calls.clear(); e.fail.clear()
    run(e, resume=True)
    assert e.calls == [("b", 50)]
    assert saved(e)[:8] == prior and saved(e)[8] == late
    final = pd.read_csv(e.directory / "predictions.csv.gz")
    assert final.decisive_seed.tolist() == [40, 50]


@pytest.mark.parametrize("mutation", ["config", "parser", "prompt", "data", "model", "selection", "assignment"])
def test_identity_rejection_preserves_manifest(execution, mutation, monkeypatch):
    e = execution
    run(e)
    before = (e.directory / "run_manifest.json").read_bytes()
    overrides = {"resume": True}
    if mutation == "config": e.config.write_text("changed")
    elif mutation == "parser": (e.kw["repo_root"] / "src/parsing/strict_response_parser.py").write_text("changed")
    elif mutation == "prompt": (e.kw["repo_root"] / "prompt.yaml").write_text("changed")
    elif mutation == "data":
        path = Path(e.kw["job"].dataset)
        path.write_text(path.read_text() + "\n")
    elif mutation == "model":
        monkeypatch.setattr(ollama, "installed_model_identity", lambda *a: {"tag": "mock", "digest": "changed"})
    elif mutation == "selection": overrides["limit"] = 1
    else:
        assignment = e.kw["repo_root"] / "assign.csv"
        assignment.write_text("protocol,fold,split,project,commit_hash,target\nholdout,0,test,p,a,fix\nholdout,0,test,p,b,fix\n")
        overrides["job"] = replace(e.kw["job"], assignment_path=str(assignment), assignment_split="test")
    with pytest.raises(RuntimeError): run(e, **overrides)
    assert (e.directory / "run_manifest.json").read_bytes() == before


@pytest.mark.parametrize("kind", ["empty", "manifest"])
def test_existing_directory(execution, kind):
    e = execution
    e.directory.mkdir(parents=True)
    if kind == "manifest": (e.directory / "run_manifest.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Existing job directory"): run(e)


@pytest.mark.parametrize("kind", ["malformed", "conflict"])
def test_bad_record_rejected_before_manifest_update(execution, kind):
    e = execution
    run(e)
    before = (e.directory / "run_manifest.json").read_bytes()
    bad = {} if kind == "malformed" else saved(e)[0] | {"raw_response": "changed"}
    Journal(e.directory / "raw_predictions.journal.jsonl").append(bad)
    with pytest.raises(RuntimeError): run(e, resume=True)
    assert (e.directory / "run_manifest.json").read_bytes() == before


def test_torn_journal_tail(execution):
    e = execution
    run(e)
    path = e.directory / "raw_predictions.journal.jsonl"
    prior = path.read_bytes()
    with path.open("ab") as handle: handle.write(b'{"unfinished')
    run(e, resume=True)
    assert path.read_bytes() == prior
    assert list(e.directory.glob("*.torn-*"))


@pytest.mark.parametrize("suffix", ["results", "results/new", "runs/../results/x"])
def test_results_protected(execution, suffix):
    with pytest.raises(runner.ConfigError):
        run(execution, output_base=execution.kw["repo_root"] / suffix)


@pytest.mark.parametrize("path", sorted((ROOT / "config/experiments").rglob("*.yaml")))
def test_all_conditions_expand_and_render(path):
    jobs = runner.expand_jobs(load_experiment(path), model_registry=load_models(ROOT / "config/models.yaml"),
                              evaluation=load_evaluation_config(ROOT / "config/evaluation.yaml"))
    registry = PromptRegistry(ROOT / "prompts/manifest.yaml")
    row = types.SimpleNamespace(processed_message="literal {{UNDEFINED}}", diff="diff", git_show="show")
    assert jobs
    for job in jobs:
        messages = registry.render_messages(job.prompt_id, **runner._prompt_variables(
            row, representation=job.representation, prepared_input="literal {{UNDEFINED}}"))
        assert messages and "{{UNDEFINED}}" in messages[-1]["content"]


def test_installed_model_check(monkeypatch):
    monkeypatch.setattr(ollama, "list_models", lambda: types.SimpleNamespace(models=[types.SimpleNamespace(model="m", digest="d")]))
    assert ollama.installed_model_identity("m", "d")["digest"] == "d"
    with pytest.raises(RuntimeError): ollama.installed_model_identity("m", "other")
    with pytest.raises(RuntimeError): ollama.installed_model_identity("missing", "d")


def test_runner_progresses_beyond_80(execution):
    e = execution
    for seed, label in enumerate(STUDY_LABELS, 1):
        e.votes["a", seed * 10] = label
    run(e)
    assert [seed for key, seed in e.calls if key == "a"] == list(range(10, 121, 10))
    final = pd.read_csv(e.directory / "predictions.csv.gz")
    assert final.decisive_seed.tolist() == [120, 30]


def test_runner_invalid_terminality(execution):
    e = execution
    e.votes["a", 10] = e.votes["a", 20] = "<invalid_output>"
    run(e)
    assert ("a", 30) in e.calls and all(seed <= 30 for _, seed in e.calls)
    assert pd.read_csv(e.directory / "predictions.csv.gz").prediction.tolist() == ["<invalid_output>", "fix"]


def test_complete_call_saved_while_sibling_waits(execution, monkeypatch):
    e = execution
    original = ollama.OllamaBackend.infer
    async def infer(self, **kwargs):
        if kwargs["prepared_input"].startswith("b"):
            assert any(r["commit_hash"] == "a" for r in saved(e))
            raise RuntimeError("sibling interrupted after durable completion")
        return await original(self, **kwargs)
    monkeypatch.setattr(ollama.OllamaBackend, "infer", infer)
    with pytest.raises(RuntimeError, match="sibling interrupted"): run(e)
    assert len(saved(e)) == 1


def test_malformed_complete_journal_line(execution):
    e = execution
    run(e)
    path = e.directory / "raw_predictions.journal.jsonl"
    with path.open("ab") as handle: handle.write(b"not-json\n")
    with pytest.raises(RuntimeError, match="Malformed journal record"): run(e, resume=True)


def test_exclusive_run_lock(tmp_path):
    from src.utils.journal import exclusive_run
    path = tmp_path / "run.lock"
    with exclusive_run(path):
        with pytest.raises(RuntimeError, match="Another process"):
            with exclusive_run(path): pass
    with exclusive_run(path): pass


@pytest.mark.parametrize("policy", [{"start": 40, "step": 10, "max_seed": 100}, {"start": 40}, {"start": 50, "step": 10}])
def test_invalid_config_policy(policy):
    with pytest.raises(runner.ConfigError):
        runner._consensus_from_scope({"initial_seeds": [10, 20, 30], "adaptive_policy": policy})


def test_finite_policy_rejected():
    with pytest.raises(runner.ConfigError, match="Finite"):
        runner._consensus_from_scope({"initial_seeds": [10, 20, 30], "adaptive_seeds": [40, 50]})


def test_conflicting_attempt_rejected(execution):
    e = execution
    run(e)
    attempts = Journal(e.directory / "attempts.journal.jsonl")
    rows, _ = attempts.read()
    attempts.append(rows[0] | {"raw_response": "conflict"})
    with pytest.raises(RuntimeError, match="Conflicting duplicate attempt"): run(e, resume=True)


def test_atomic_manifest_failure_preserves_old_file(tmp_path, monkeypatch):
    from src.utils import journal
    path = tmp_path / "manifest.json"
    journal.atomic_json(path, {"old": True})
    def fail(*args): raise OSError("interrupted replacement")
    monkeypatch.setattr(journal.os, "replace", fail)
    with pytest.raises(OSError): journal.atomic_json(path, {"new": True})
    assert json.loads(path.read_text()) == {"old": True}


def test_link_into_results_rejected(execution):
    e = execution
    frozen = e.kw["repo_root"] / "results"
    frozen.mkdir()
    alias = e.kw["repo_root"] / "alias"
    try:
        alias.symlink_to(frozen, target_is_directory=True)
    except OSError:
        if sys.platform != "win32":
            pytest.skip("Creating symlinks is unavailable")
        linked = subprocess.run(["cmd", "/c", "mklink", "/J", str(alias), str(frozen)], capture_output=True)
        assert linked.returncode == 0, linked.stderr
    with pytest.raises(runner.ConfigError): run(e, output_base=alias)


def test_prompt_bytes_frozen_with_fingerprint(execution):
    e = execution
    registry = PromptRegistry(e.kw["prompt_manifest"])
    before = registry.sha256("p")
    (e.kw["repo_root"] / "prompt.yaml").write_text('- role: user\n  activate: true\n  content: changed\n')
    assert registry.sha256("p") == before
    assert registry.render_messages("p", COMMIT_INPUT="original")[0]["content"] == "original"
    fresh = PromptRegistry(e.kw["prompt_manifest"])
    assert fresh.sha256("p") != before


def test_clipped_message_aliases():
    row = types.SimpleNamespace(processed_message="long message", diff="original diff", git_show="")
    values = runner._prompt_variables(row, representation="message_and_diff", prepared_input="long")
    assert values["COMMIT_INPUT"] == values["COMMIT_MESSAGE"] == "long"
    assert values["COMMIT_DIFF"] == ""


def test_lost_durable_records_fail_instead_of_repeating(execution):
    e = execution
    run(e)
    path = e.directory / "raw_predictions.journal.jsonl"
    path.write_bytes(b"".join(path.read_bytes().splitlines(keepends=True)[:-1]))
    e.calls.clear()
    with pytest.raises(RuntimeError, match="fewer completed calls"): run(e, resume=True)
    assert not e.calls


def test_dry_run_is_offline_and_does_not_write(execution, monkeypatch):
    def forbidden(*args): pytest.fail("Dry-run queried the installed model")
    monkeypatch.setattr(ollama, "installed_model_identity", forbidden)
    run(execution, dry_run=True)
    assert not execution.calls and not execution.directory.exists()


@pytest.mark.parametrize("expectation", [{"expected_rows": 1}, {"expected_rows": 2, "expected_cohort_sha256": "wrong"}])
def test_frozen_cohort_preflight_cannot_be_bypassed_by_limit(execution, expectation):
    e = execution
    with pytest.raises(runner.ConfigError, match="Frozen cohort mismatch"):
        run(e, job=replace(e.kw["job"], **expectation), dry_run=True, limit=1)
    assert not e.calls and not e.directory.exists()


def test_artifact_native_ids_retained(execution):
    e = execution
    path = e.kw["repo_root"] / "artifact.csv"
    path.write_text("protocol,project,commit_hash,canonical_target,protocol_target,source_index,protocol_record_id\n"
                    "hqcm_compatible,p,a,fix,fix,0,native_00000\n"
                    "hqcm_compatible,p,b,fix,fix,1,native_00001\n")
    run(e, job=replace(e.kw["job"], assignment_path=str(path)))
    assert saved(e)[0]["protocol_record_id"] == "native_00000"
    final = pd.read_csv(e.directory / "predictions.csv.gz")
    assert final.commit_hash.tolist() == ["a", "b"]
    assert final.protocol_record_id.tolist() == ["native_00000", "native_00001"]
