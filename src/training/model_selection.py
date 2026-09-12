from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ValidationSplitConfig:
    """How much of a training partition to carve off for internal model selection.

    This only subdivides an already-existing training partition; it never touches
    the frozen test partition of a fold or of the within-corpus holdout.
    """
    fraction: float = 0.15
    stratify: bool = True


def carve_validation_split(
    train: pd.DataFrame,
    *,
    seed: int,
    config: ValidationSplitConfig | None = None,
    group_col: str | None = None,
    stratify_col: str = "target",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `train` into (train', dev) for internal hyperparameter/epoch selection.

    `group_col`, when given (e.g. "project"), keeps every group entirely on one
    side of the split, mirroring the outer cross-project protocol's own
    no-project-leakage guarantee, so the selection signal reflects unseen-project
    generalization rather than an easier within-project split. When omitted, a
    stratified per-commit split is used, matching the within-corpus protocol's
    own commit-level design. Selection never sees the test partition of the fold
    or holdout it runs inside.
    """
    c = config or ValidationSplitConfig()
    if not 0 < c.fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    if train.empty:
        raise RuntimeError("Cannot carve a validation split from an empty training partition")

    if group_col is None:
        stratify = train[stratify_col] if c.stratify else None
        idx_train, idx_dev = train_test_split(
            train.index, test_size=c.fraction, random_state=seed, stratify=stratify,
        )
        train_prime, dev = train.loc[idx_train].copy(), train.loc[idx_dev].copy()
    else:
        rng = np.random.RandomState(seed)
        groups = train[group_col].drop_duplicates().sort_values().tolist()
        if len(groups) < 2:
            raise RuntimeError(f"Need at least 2 distinct {group_col!r} groups to carve a grouped validation split")
        order = rng.permutation(len(groups))
        groups = [groups[i] for i in order]
        target_dev_rows = int(round(len(train) * c.fraction))
        group_sizes = train.groupby(group_col).size()
        dev_groups: list = []
        dev_rows = 0
        for g in groups:
            if dev_rows >= target_dev_rows or len(dev_groups) >= len(groups) - 1:
                break
            dev_groups.append(g)
            dev_rows += int(group_sizes[g])
        dev_mask = train[group_col].isin(dev_groups)
        train_prime, dev = train.loc[~dev_mask].copy(), train.loc[dev_mask].copy()

    if train_prime.empty or dev.empty:
        raise RuntimeError("Validation split produced an empty side; adjust fraction or grouping")
    missing_classes = sorted(set(train_prime[stratify_col]) - set(train[stratify_col]))
    if missing_classes:
        raise RuntimeError(f"train' is missing classes present in train: {missing_classes}")
    return train_prime, dev


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(list(y_true), list(y_pred), average="macro", zero_division=0))
