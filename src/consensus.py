from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .data.loader import STUDY_LABELS

INVALID_OUTPUT = "<invalid_output>"
ABSTAINED = "<abstained>"
LABEL_SET = set(STUDY_LABELS)


@dataclass(frozen=True)
class AdaptiveSeedPolicy:
    start: int = 40
    step: int = 10

    def __post_init__(self):
        if type(self.start) is not int or type(self.step) is not int or (self.start, self.step) != (40, 10):
            raise ValueError("Adaptive policy must start at 40 with step 10; no maximum")

    def contains(self, seed: int) -> bool:
        return seed >= self.start and (seed - self.start) % self.step == 0


def validate_base_seeds(seeds: Sequence[int]) -> None:
    if any(type(seed) is not int for seed in seeds) or tuple(seeds) != (10, 20, 30):
        raise ValueError("Mandatory base seeds must be exactly 10, 20, 30 in order")


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
    task_labels: Iterable[str],
    adaptive_policy: AdaptiveSeedPolicy | None = None,
    threshold: int = 2,
    invalid_terminal: bool = True,
    abstention: bool = False,
) -> ConsensusResult:
    """Mandatory base block followed by sequential votes, without exhaustion.

    Missing evidence remains incomplete; stored votes after terminality are ignored.
    With ``abstention=True``, ``ABSTAINED`` votes are kept and two of them end the
    commit as ``ABSTAINED``.
    """
    validate_base_seeds(initial_seeds)
    policy = adaptive_policy or AdaptiveSeedPolicy()
    if threshold != 2:
        raise ValueError("This study's frozen consensus threshold is 2")
    allowed = set(task_labels)
    if not allowed or not allowed.issubset(LABEL_SET):
        raise ValueError(f"Task labels outside study taxonomy: {sorted(allowed - LABEL_SET)}")

    missing_initial = [s for s in initial_seeds if s not in by_seed]
    if missing_initial:
        return ConsensusResult(None, "missing_initial_seed", False, missing_initial[0], None, (), ())

    used: list[int] = []
    votes: list[str] = []
    def seeds():
        yield from initial_seeds
        seed = policy.start
        while True:
            yield seed
            seed += policy.step

    sequence = seeds()
    initial_last = initial_seeds[-1] if initial_seeds else None

    for seed in sequence:
        if seed not in by_seed:
            if policy.contains(seed):
                return ConsensusResult(None, "needs_adaptive_seed", False, seed, None, tuple(used), tuple(votes))
            raise RuntimeError(f"Unexpected missing initial seed {seed}")
        vote = str(by_seed[seed])
        if vote not in allowed and vote != INVALID_OUTPUT and not (abstention and vote == ABSTAINED):
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
        if abstention and sum(v == ABSTAINED for v in votes) >= threshold:
            return ConsensusResult(ABSTAINED, "terminal_abstained", True, None, seed, tuple(used), tuple(votes))
