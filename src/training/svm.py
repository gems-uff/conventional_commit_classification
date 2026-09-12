from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .model_selection import ValidationSplitConfig, carve_validation_split, macro_f1


@dataclass(frozen=True)
class SvmConfig:
    max_features: int = 50000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True
    lowercase: bool = True
    C: float = 1.0
    penalty: str = "l2"
    loss: str = "squared_hinge"
    dual: bool = True
    tol: float = 1e-4
    max_iter: int = 5000
    class_weight: str = "balanced"


def build_model(seed: int, config: SvmConfig | None = None) -> Pipeline:
    c = config or SvmConfig()
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=c.max_features, ngram_range=c.ngram_range,
                                  min_df=c.min_df, max_df=c.max_df, sublinear_tf=c.sublinear_tf,
                                  lowercase=c.lowercase)),
        ("svm", LinearSVC(C=c.C, penalty=c.penalty, loss=c.loss, dual=c.dual, tol=c.tol,
                          max_iter=c.max_iter, class_weight=c.class_weight, random_state=seed)),
    ])


def train_predict(train: pd.DataFrame, test: pd.DataFrame, *, seed: int, config: SvmConfig | None = None) -> pd.DataFrame:
    for name, frame in (("train", train), ("test", test)):
        missing = {"input_text", "target", "project", "commit_hash"} - set(frame.columns)
        if missing: raise RuntimeError(f"{name}: missing columns {sorted(missing)}")
    model = build_model(seed, config)
    model.fit(train["input_text"], train["target"])
    out = test.copy(); out["prediction"] = model.predict(test["input_text"]); out["seed"] = seed
    out["correct"] = out["target"] == out["prediction"]
    return out


def select_c(
    train: pd.DataFrame,
    *,
    seed: int,
    grid: Sequence[float],
    validation: ValidationSplitConfig | None = None,
    group_col: str | None = None,
    base_config: SvmConfig | None = None,
) -> tuple[SvmConfig, dict[str, Any]]:
    """Pick `C` via one held-out split carved from `train` (never touches test).

    Runs once per fold/holdout; the returned config is then reused, fixed, across
    every seed's from-scratch fit in `train_predict` for that fold/holdout -- the
    repeated-seed protocol varies training stochasticity, not the hyperparameter.
    """
    if not grid:
        raise ValueError("grid must be non-empty")
    base = base_config or SvmConfig()
    train_prime, dev = carve_validation_split(train, seed=seed, config=validation, group_col=group_col)
    scores: dict[float, float] = {}
    for c in grid:
        candidate = replace(base, C=float(c))
        model = build_model(seed, candidate)
        model.fit(train_prime["input_text"], train_prime["target"])
        scores[float(c)] = macro_f1(dev["target"], model.predict(dev["input_text"]))
    best_c = max(scores, key=scores.get)
    diagnostics = {
        "grid_scores": scores, "selected_C": best_c,
        "train_prime_rows": len(train_prime), "dev_rows": len(dev),
    }
    return replace(base, C=best_c), diagnostics
