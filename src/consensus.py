from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .parsing.strict import INVALID_OUTPUT, LABEL_SET


@dataclass(frozen=True)
class ConsensusResult:
    classification: str | None
    status: str
    complete: bool
    missing_seed: int | None
    decisive_seed: int | None
    used_seeds: tuple[int, ...]
    votes: tuple[str, ...]


def resolve_first_to_two(
    by_seed: Mapping[int, str],
    *,
    initial_seeds: Sequence[int],
    adaptive_seeds: Sequence[int],
    task_labels: Iterable[str],
    threshold: int = 2,
    invalid_terminal: bool = True,
) -> ConsensusResult:
    """Resolve the frozen sequential first-to-two protocol.

    All initial seeds must exist before a terminal decision is evaluated. Extra
    seeds are consumed in the supplied order only while the item is unresolved.
    With no adaptive seeds, an unresolved initial trio exhausts to invalid.
    """
    if threshold != 2:
        raise ValueError("This study's frozen consensus threshold is 2")
    allowed = set(task_labels)
    if not allowed.issubset(LABEL_SET):
        raise ValueError(f"Task labels outside study taxonomy: {sorted(allowed - LABEL_SET)}")

    missing_initial = [s for s in initial_seeds if s not in by_seed]
    if missing_initial:
        return ConsensusResult(None, "missing_initial_seed", False, missing_initial[0], None, (), ())

    used: list[int] = []
    votes: list[str] = []
    sequence = list(initial_seeds) + list(adaptive_seeds)
    initial_last = initial_seeds[-1] if initial_seeds else None

    for seed in sequence:
        if seed not in by_seed:
            if seed in adaptive_seeds:
                return ConsensusResult(None, "needs_adaptive_seed", False, seed, None, tuple(used), tuple(votes))
            raise RuntimeError(f"Unexpected missing initial seed {seed}")
        vote = str(by_seed[seed])
        if vote not in allowed and vote != INVALID_OUTPUT:
            vote = INVALID_OUTPUT
        used.append(seed); votes.append(vote)

        if seed != initial_last and seed in initial_seeds:
            continue
        semantic = Counter(v for v in votes if v in allowed)
        winner = next((v for v in votes if v in allowed and semantic[v] >= threshold), None)
        if winner is not None:
            return ConsensusResult(winner, "semantic", True, None, seed, tuple(used), tuple(votes))
        if invalid_terminal and sum(v == INVALID_OUTPUT for v in votes) >= threshold:
            return ConsensusResult(INVALID_OUTPUT, "terminal_invalid", True, None, seed, tuple(used), tuple(votes))

    return ConsensusResult(
        INVALID_OUTPUT, "exhausted_invalid", True, None,
        sequence[-1] if sequence else None, tuple(used), tuple(votes),
    )
