from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class GenerativePredictionRecord:
    """Compact per-seed result emitted by generative backends.

    Inputs are not stored per row: ``prompt_id`` + ``representation`` + the
    commit key identify the request, and the run manifest records the
    prompt/parser/model/config hashes.
    """

    experiment_id: str
    condition_id: str
    run_id: str
    backend: str
    model_id: str
    prompt_id: str
    representation: str
    partition: str
    platform: str
    project: str
    commit_hash: str
    target: str
    seed: int
    prediction: str
    correct: bool
    raw_response: str
    parse_status: str
    parser_label: str = ""
    parser_raw_label: str = ""
    normalization_rule: str = ""
    technical_status: str = "ok"
    processing_time_seconds: float = 0.0
    input_tokens: int | None = None
    num_ctx: int | None = None
    input_sha256: str = ""
    protocol: str = ""
    fold: int | None = None
    split: str = ""
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClosedSetPredictionRecord:
    """Compact per-seed prediction for SVM, BiLSTM, and CodeBERT."""

    experiment_id: str
    condition_id: str
    run_id: str
    backend: str
    model_id: str
    representation: str
    partition: str
    platform: str
    project: str
    commit_hash: str
    target: str
    seed: int
    prediction: str
    correct: bool
    protocol: str = ""
    fold: int | None = None
    split: str = ""
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
