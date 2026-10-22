import pytest
from src.consensus import AdaptiveSeedPolicy, resolve_first_to_two, INVALID_OUTPUT
from src.data.loader import STUDY_LABELS


def resolve(votes, **kwargs):
    return resolve_first_to_two(votes, initial_seeds=[10, 20, 30], task_labels=STUDY_LABELS, **kwargs)


def test_mandatory_base_block():
    result = resolve({10: "fix", 20: "fix"})
    assert not result.complete and result.missing_seed == 30
    assert resolve({10: "fix", 20: "fix", 30: "feat"}).decisive_seed == 30


def test_adaptive_beyond_80_and_terminal():
    votes = {10 * (i + 1): label for i, label in enumerate(STUDY_LABELS)}
    for seed in range(40, 120, 10):
        result = resolve({s: v for s, v in votes.items() if s < seed})
        assert not result.complete and result.missing_seed == seed
    assert resolve(votes).missing_seed == 120
    votes[120] = "fix"
    result = resolve(votes)
    assert result.complete and result.classification == "fix" and result.decisive_seed == 120


def test_two_invalid_and_ignore_later():
    votes = {10: INVALID_OUTPUT, 20: INVALID_OUTPUT, 30: "fix", 40: "feat", 50: "feat"}
    result = resolve(votes)
    assert result.classification == INVALID_OUTPUT and result.used_seeds == (10, 20, 30)


def test_first_missing_sequential_seed():
    result = resolve({10: "fix", 20: "feat", 30: "docs", 50: "fix"})
    assert result.missing_seed == 40
    result = resolve({10: "fix", 20: "feat", 30: "docs", 40: "test", 50: "fix", 60: "docs"})
    assert result.decisive_seed == 50 and 60 not in result.used_seeds


@pytest.mark.parametrize("seeds", [[10, 10, 30], [10, 20], [30, 20, 10], [10, 20, 30.0]])
def test_invalid_base(seeds):
    with pytest.raises(ValueError):
        resolve_first_to_two({}, initial_seeds=seeds, task_labels=STUDY_LABELS)


@pytest.mark.parametrize("kwargs", [{"start": 30}, {"step": 0}, {"step": -10}, {"start": 40.0}])
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        AdaptiveSeedPolicy(**kwargs)
