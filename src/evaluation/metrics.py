from __future__ import annotations

from typing import Sequence, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..parsing.strict import INVALID_OUTPUT, LABELS, LABEL_SET


def labels_present(y_true: Sequence[str], labels: Iterable[str] = LABELS) -> list[str]:
    present = set(map(str, y_true))
    return [label for label in labels if label in present]


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str], *, task_labels: Iterable[str] = LABELS) -> dict:
    y_true = list(map(str, y_true)); y_pred = list(map(str, y_pred))
    allowed = tuple(task_labels)
    present = labels_present(y_true, allowed)
    if not present:
        raise RuntimeError("Ground truth has no task labels")
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present, average=None, zero_division=0
    )
    weights = support / support.sum() if support.sum() else np.zeros_like(support, dtype=float)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.sum(f1 * weights)),
        "labels_evaluated": present,
        "n_labels_evaluated": len(present),
        "invalid_predictions": int(sum(v == INVALID_OUTPUT or v not in allowed for v in y_pred)),
    }


def macro_f1_present_classes(y_true: Sequence[str], y_pred: Sequence[str], *, task_labels: Iterable[str] = LABELS) -> float:
    return compute_metrics(y_true, y_pred, task_labels=task_labels)["macro_f1"]


def metrics_by_project(
    predictions: pd.DataFrame,
    *,
    project_col: str = "project",
    target_col: str = "target",
    prediction_col: str = "classification",
    task_labels: Iterable[str] = LABELS,
) -> pd.DataFrame:
    required = {project_col, target_col, prediction_col}
    missing = required - set(predictions.columns)
    if missing:
        raise RuntimeError(f"Missing metric columns {sorted(missing)}")
    rows = []
    for project, group in predictions.groupby(project_col, sort=True):
        m = compute_metrics(group[target_col].tolist(), group[prediction_col].tolist(), task_labels=task_labels)
        rows.append({
            "project": project,
            "n_commits": int(len(group)),
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"],
            "n_invalid": m["invalid_predictions"],
            "labels_evaluated": ",".join(m["labels_evaluated"]),
            "n_classes_present": m["n_labels_evaluated"],
        })
    return pd.DataFrame(rows)
