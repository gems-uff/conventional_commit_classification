"""Direct tests for src/inference/hqcm.py.

HqcmBackend only wraps the upstream ``hqcm.gen`` CLI, so ``subprocess.run`` and
``time.sleep`` are replaced by fakes: no HQCM checkout, adapter or GPU needed.
"""
import subprocess
from pathlib import Path

import pytest

from src.inference import hqcm
from src.inference.hqcm import HqcmBackend, HqcmConfig
from src.parsing.strict import INVALID_OUTPUT

HQCM_LABELS = {"build", "ci", "docs", "feat", "fix", "refactor", "style", "test"}


class FakeRun:
    """Stands in for subprocess.run.

    ``outcomes`` scripts each hqcm.gen call: "ok" writes the output JSON,
    "error" raises CalledProcessError, "no_output" exits cleanly without writing.
    """

    def __init__(self, outcomes, *, help_text="usage: hqcm.gen [--seed SEED] adapter"):
        self.outcomes = list(outcomes)
        self.help_text = help_text
        self.calls = []

    def __call__(self, cmd, **kwargs):
        if "--help" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=self.help_text, stderr="")
        self.calls.append((list(cmd), kwargs))
        outcome = self.outcomes.pop(0)
        if outcome == "error":
            raise subprocess.CalledProcessError(1, cmd)
        if outcome == "ok":
            Path(cmd[cmd.index("--output") + 1]).write_text("[]")
        return subprocess.CompletedProcess(cmd, 0)


@pytest.fixture
def backend(tmp_path):
    return HqcmBackend(HqcmConfig(
        hqcm_root=tmp_path / "codefuse-hqcm",
        adapter=tmp_path / "chcl_llama2_7b_len1024",
        python=tmp_path / "venv" / "python",
        max_attempts=3,
        retry_delay_seconds=5.0,
    ))


@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(hqcm.time, "sleep", calls.append)
    return calls


def run_chunk(backend, tmp_path, fake, monkeypatch, seed=20):
    monkeypatch.setattr(hqcm.subprocess, "run", fake)
    output = tmp_path / "out.json"
    backend.run_chunk(dataset_dir=tmp_path / "chunk", output_json=output, seed=seed)
    return output


def test_run_chunk_builds_native_command_env_and_cwd(backend, tmp_path, monkeypatch, sleeps):
    fake = FakeRun(["ok"])
    output = run_chunk(backend, tmp_path, fake, monkeypatch, seed=20)

    (cmd, kwargs), = fake.calls
    cfg = backend.config
    assert cmd == [
        str(cfg.python), "-m", "hqcm.gen",
        "--dataset", str(tmp_path / "chunk"), "--split", "test",
        "--output", str(output), "--temperature", "0",
        "--seed", "20",
        str(cfg.adapter),
    ]
    assert kwargs["cwd"] == str(cfg.hqcm_root)
    assert kwargs["check"] is True
    assert kwargs["env"]["PYTHONHASHSEED"] == "20"
    assert kwargs["env"]["PYTHONPATH"] == str(cfg.hqcm_root)
    assert output.exists()
    assert sleeps == []


def test_run_chunk_omits_seed_when_hqcm_gen_does_not_support_it(backend, tmp_path, monkeypatch, sleeps):
    fake = FakeRun(["ok"], help_text="usage: hqcm.gen [--temperature T] adapter")
    run_chunk(backend, tmp_path, fake, monkeypatch)
    (cmd, _), = fake.calls
    assert "--seed" not in cmd
    assert cmd[-1] == str(backend.config.adapter)


def test_run_chunk_retries_after_failure_then_succeeds(backend, tmp_path, monkeypatch, sleeps):
    fake = FakeRun(["error", "ok"])
    output = run_chunk(backend, tmp_path, fake, monkeypatch)
    assert len(fake.calls) == 2
    assert sleeps == [5.0]
    assert output.exists()


def test_run_chunk_aborts_after_max_attempts_without_synthetic_rows(backend, tmp_path, monkeypatch, sleeps):
    fake = FakeRun(["error", "error", "error"])
    with pytest.raises(RuntimeError, match="No synthetic invalid rows") as excinfo:
        run_chunk(backend, tmp_path, fake, monkeypatch)
    assert len(fake.calls) == 3
    assert sleeps == [5.0, 5.0]
    assert isinstance(excinfo.value.__cause__, subprocess.CalledProcessError)
    assert not (tmp_path / "out.json").exists()


def test_run_chunk_treats_missing_output_as_failure(backend, tmp_path, monkeypatch, sleeps):
    fake = FakeRun(["no_output", "no_output", "no_output"])
    with pytest.raises(RuntimeError, match="failed after 3 attempts") as excinfo:
        run_chunk(backend, tmp_path, fake, monkeypatch)
    assert "did not create" in str(excinfo.value.__cause__)


def test_run_chunk_removes_stale_output_before_each_attempt(backend, tmp_path, monkeypatch, sleeps):
    (tmp_path / "out.json").write_text("stale")
    fake = FakeRun(["no_output", "no_output", "no_output"])
    with pytest.raises(RuntimeError):
        run_chunk(backend, tmp_path, fake, monkeypatch)
    assert not (tmp_path / "out.json").exists()


# parse() -------------------------------------------------------------------------

def test_parse_maps_native_cicd_to_ci():
    parsed = HqcmBackend.parse("cicd", task_labels=HQCM_LABELS)
    assert parsed.prediction == "ci"
    assert parsed.normalization_rule == "cicd_to_ci"


def test_parse_cicd_is_invalid_when_ci_is_not_a_task_label():
    assert HqcmBackend.parse("cicd", task_labels=HQCM_LABELS - {"ci"}).prediction == INVALID_OUTPUT


def test_parse_delegates_other_labels_to_strict_parser():
    assert HqcmBackend.parse("fix", task_labels=HQCM_LABELS).prediction == "fix"
    assert HqcmBackend.parse("perf", task_labels=HQCM_LABELS).prediction == INVALID_OUTPUT


def test_parse_technical_failure_is_invalid_even_for_cicd():
    parsed = HqcmBackend.parse("cicd", task_labels=HQCM_LABELS, technical_status="failure")
    assert parsed.prediction == INVALID_OUTPUT
    assert parsed.technical_status == "failure"
