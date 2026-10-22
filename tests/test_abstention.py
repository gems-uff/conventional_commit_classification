"""Confidence-based abstention (RQ3.2 Factor 4)."""
import types
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.consensus import ABSTAINED, INVALID_OUTPUT, resolve_first_to_two
from src.data.loader import STUDY_LABELS
from src.inference import ollama
from src.parsing.confidence import INVALID_CONFIDENCE, abstention_vote, parse_confidence
from test_ollama_execution import execution, run  # noqa: F401  (fixture reuse)

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {"p_exclude_low": ("low",), "p_high_only": ("low", "medium")}


def resolve(votes, abstention=True):
    by_seed = dict(zip([10, 20, 30, 40, 50, 60], votes))
    return resolve_first_to_two(by_seed, initial_seeds=(10, 20, 30), task_labels=STUDY_LABELS,
                                abstention=abstention)


# consensus ---------------------------------------------------------------------

def test_two_abstentions_end_as_abstained():
    result = resolve(["fix", ABSTAINED, ABSTAINED])
    assert (result.classification, result.status, result.decisive_seed) == (ABSTAINED, "terminal_abstained", 30)


def test_abstentions_do_not_count_as_class_votes():
    result = resolve(["fix", ABSTAINED, INVALID_OUTPUT])
    assert not result.complete and result.missing_seed == 40
    assert resolve(["fix", ABSTAINED, INVALID_OUTPUT, "fix"]).classification == "fix"


def test_abstained_is_invalid_without_abstention_policy():
    assert resolve(["fix", ABSTAINED, ABSTAINED], abstention=False).classification == INVALID_OUTPUT


# confidence parsing and seed votes -----------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("classification: fix\nconfidence: High\nreason: ...", "high"),
    ("classification: discard\nconfidence: Medium", "medium"),
    ("**Confidence:** low", "low"),
    ('{"classification": "fix", "confidence": "HIGH"}', "high"),
    ("classification: fix\nreason: no confidence field", INVALID_CONFIDENCE),
    ("confidence: high\nconfidence: low", INVALID_CONFIDENCE),
    ("", INVALID_CONFIDENCE),
])
def test_parse_confidence(raw, expected):
    assert parse_confidence(raw) == expected


@pytest.mark.parametrize("prediction,confidence,policy,expected", [
    ("fix", "high", "p_high_only", "fix"),
    ("fix", "medium", "p_high_only", ABSTAINED),
    ("fix", "medium", "p_exclude_low", "fix"),
    (INVALID_OUTPUT, "low", "p_exclude_low", ABSTAINED),
    ("fix", INVALID_CONFIDENCE, "p_high_only", INVALID_OUTPUT),
])
def test_abstention_vote(prediction, confidence, policy, expected):
    assert abstention_vote(prediction, confidence, POLICIES[policy]) == expected


def test_policy_reproduces_published_factor_4_predictions():
    """Seed predictions + seed confidences in results/ -> the published consensus."""
    confidences = pd.read_csv(ROOT / "results/rq3_2/predictions/factor4_confidence.csv.gz", keep_default_na=False)
    predictions = pd.read_csv(ROOT / "results/rq3_2/predictions/predictions.csv.gz", keep_default_na=False)
    merged = predictions.rename(columns={"treatment": "prompt"}).merge(
        confidences, on=["prompt", "project", "commit_hash"])
    assert len(merged) == 2 * 7937
    for row in merged.itertuples(index=False):
        row = row._asdict()
        by_seed = {}
        for seed in (10, 20, 30, 40, 50, 60, 70, 80):
            if row[f"seed_{seed}"] == "" and row[f"confidence_{seed}"] == "":
                break
            by_seed[seed] = abstention_vote(row[f"seed_{seed}"], row[f"confidence_{seed}"], POLICIES[row["prompt"]])
        result = resolve_first_to_two(by_seed, initial_seeds=(10, 20, 30), task_labels=STUDY_LABELS, abstention=True)
        assert result.classification == row["prediction"], (row["prompt"], row["commit_hash"])


# runner ------------------------------------------------------------------------

def test_runner_applies_abstention_policy(execution, monkeypatch):  # noqa: F811
    responses = {  # (commit, seed) -> (classification, confidence)
        ("a", 10): ("fix", "high"), ("a", 20): ("discard", "medium"), ("a", 30): ("discard", "low"),
        ("b", 10): ("fix", "high"), ("b", 20): ("discard", "medium"), ("b", 30): ("<invalid_output>", "high"),
        ("b", 40): ("fix", "high"),
    }
    calls = []

    class Backend:
        def __init__(self, config): self.config = config
        def prepare_content(self, text): return text, len(text)
        async def infer(self, **kw):
            key = (kw["prepared_input"].split("\n")[0], kw["seed"])
            calls.append(key)
            label, confidence = responses[key]
            raw = f"classification: {label}\nconfidence: {confidence}"
            prediction = label if label in STUDY_LABELS else INVALID_OUTPUT
            kw["attempt_sink"]({"attempt": 1, "raw_response": raw, "technical_status": "ok",
                                "technical_detail": "", "processing_time_seconds": 0})
            parsed = types.SimpleNamespace(prediction=prediction, parse_status="mock", parser_label=prediction,
                                           parser_raw_label=label, normalization_rule="")
            return ollama.OllamaCallResult(raw, parsed, "ok", "", 0, 1, 4096)

    monkeypatch.setattr(ollama, "OllamaBackend", Backend)
    job = replace(execution.kw["job"], abstain_on=("low", "medium"))
    run(execution, job=job)

    assert ("b", 40) in calls and ("a", 40) not in calls
    final = pd.read_csv(execution.directory / "predictions.csv.gz", keep_default_na=False)
    assert final.prediction.tolist() == [ABSTAINED, "fix"]
    assert final.loc[0, ["confidence_10", "confidence_20", "confidence_30"]].tolist() == ["high", "medium", "low"]
