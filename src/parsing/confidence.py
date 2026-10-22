from __future__ import annotations

import re
from typing import Collection

from ..consensus import ABSTAINED, INVALID_OUTPUT

CONFIDENCE_LEVELS = ("low", "medium", "high")
INVALID_CONFIDENCE = "<invalid_confidence>"

_CONFIDENCE_RE = re.compile(
    r"""confidence["'*_`\s]*[:=]["'*_`\s]*(low|medium|high)\b""",
    re.IGNORECASE,
)


def parse_confidence(raw_response: str | None) -> str:
    """Return ``low``, ``medium``, or ``high`` from a ``confidence: <level>`` field.

    Returns ``INVALID_CONFIDENCE`` when the field is missing or reports more than
    one level.
    """
    levels = {m.group(1).casefold() for m in _CONFIDENCE_RE.finditer(raw_response or "")}
    return levels.pop() if len(levels) == 1 else INVALID_CONFIDENCE


def abstention_vote(prediction: str, confidence: str, abstain_on: Collection[str]) -> str:
    """Seed-level vote under a confidence-based abstention policy.

    A confidence level in ``abstain_on`` is an abstention; a missing or invalid
    confidence is an invalid response; otherwise the seed's prediction is the vote.
    """
    if confidence in abstain_on:
        return ABSTAINED
    if confidence not in CONFIDENCE_LEVELS:
        return INVALID_OUTPUT
    return prediction
