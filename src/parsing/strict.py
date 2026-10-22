from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Collection

try:
    from .strict_response_parser import (
        INVALID_OUTPUT, LABELS, LABEL_SET, PARSER_VERSION, ParseResult, parse_raw_response,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - setup guard
    raise RuntimeError(
        "src/parsing/strict_response_parser.py is missing; restore it from the repository."
    ) from exc

EXPECTED_PARSER_VERSION = "4.1.1"
if PARSER_VERSION != EXPECTED_PARSER_VERSION:
    raise RuntimeError(f"Expected parser {EXPECTED_PARSER_VERSION}, found {PARSER_VERSION}")


@dataclass(frozen=True)
class PredictionParse:
    prediction: str
    parse_status: str
    parser_label: str
    parser_raw_label: str
    normalization_rule: str
    technical_status: str


def strict_prediction_from_parse(
    parsed: ParseResult,
    *,
    task_labels: Collection[str] | None = None,
    technical_status: str = "ok",
) -> str:
    """Apply the final strict-taxonomy interpretation used by the package.

    Only the strict taxonomy is applied: an explicit external label is invalid
    even though the parser records its normalized semantic destination for
    audit purposes.
    """
    allowed = set(LABEL_SET if task_labels is None else task_labels)
    unknown = allowed - LABEL_SET
    if unknown:
        raise ValueError(f"Task labels outside study taxonomy: {sorted(unknown)}")
    if technical_status != "ok":
        return INVALID_OUTPUT
    if str(parsed.status).startswith("valid_") and parsed.label in allowed:
        return str(parsed.label)
    return INVALID_OUTPUT


def parse_strict(
    raw_response: Any,
    *,
    task_labels: Collection[str] | None = None,
    technical_status: str = "ok",
) -> PredictionParse:
    parsed = parse_raw_response(raw_response)
    prediction = strict_prediction_from_parse(
        parsed, task_labels=task_labels, technical_status=technical_status
    )
    return PredictionParse(
        prediction=prediction,
        parse_status=str(parsed.status),
        parser_label="" if parsed.label is None else str(parsed.label),
        parser_raw_label="" if parsed.raw_label is None else str(parsed.raw_label),
        normalization_rule="" if parsed.normalization_rule is None else str(parsed.normalization_rule),
        technical_status=technical_status,
    )


def _clean_complete_scalar(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n").strip()
    text = text.replace("</s>", "").replace("<s>", "").strip()
    wrappers = (
        ("***", "***"), ("___", "___"), ("**", "**"), ("__", "__"),
        ("`", "`"), ('"', '"'), ("'", "'"), ("*", "*"), ("_", "_"),
    )
    for _ in range(12):
        before = text
        for left, right in wrappers:
            if len(text) > len(left) + len(right) and text.startswith(left) and text.endswith(right):
                text = text[len(left):-len(right)].strip()
                break
        if text == before:
            break
    return re.sub(r"\s+", " ", text.rstrip(".,;").strip()).casefold()


def parse_hqcm(
    raw_response: Any,
    *,
    task_labels: Collection[str],
    technical_status: str = "ok",
) -> PredictionParse:
    """HQCM native output contract plus strict parser v4.1.1.

    A complete bare `cicd` answer is mapped to `ci`; every other response goes
    through the strict parser.
    """
    allowed = set(task_labels)
    if technical_status == "ok" and _clean_complete_scalar(raw_response) == "cicd":
        return PredictionParse(
            prediction="ci" if "ci" in allowed else INVALID_OUTPUT,
            parse_status="valid_hqcm_native_bare_cicd",
            parser_label="ci",
            parser_raw_label="cicd",
            normalization_rule="cicd_to_ci",
            technical_status=technical_status,
        )
    return parse_strict(
        raw_response,
        task_labels=allowed,
        technical_status=technical_status,
    )
