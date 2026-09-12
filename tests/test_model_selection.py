import numpy as np
import pandas as pd
import pytest

from src.training.model_selection import ValidationSplitConfig, carve_validation_split
from src.training.svm import select_c


def _synthetic_frame(n_per_class=10, classes=("fix", "feat", "docs", "chore"), projects=("p1", "p2", "p3", "p4")):
    rows = []
    for cls_idx, cls in enumerate(classes):
        for i in range(n_per_class):
            project = projects[(cls_idx * n_per_class + i) % len(projects)]
            rows.append({
                "project": project,
                "commit_hash": f"{cls}-{i}",
                "target": cls,
                "input_text": f"{cls} change number {i} " * (i % 3 + 1),
            })
    return pd.DataFrame(rows)


def test_carve_validation_split_ungrouped_is_stratified_and_disjoint():
    df = _synthetic_frame()
    train_prime, dev = carve_validation_split(df, seed=1, config=ValidationSplitConfig(fraction=0.25))
    assert len(train_prime) + len(dev) == len(df)
    assert set(train_prime["target"]) == set(df["target"])
    assert not set(train_prime.index) & set(dev.index)


def test_carve_validation_split_grouped_keeps_every_project_on_one_side():
    df = _synthetic_frame()
    train_prime, dev = carve_validation_split(
        df, seed=1, config=ValidationSplitConfig(fraction=0.3), group_col="project")
    assert not set(train_prime["project"]) & set(dev["project"])
    assert len(train_prime) + len(dev) == len(df)


def test_carve_validation_split_rejects_bad_fraction():
    df = _synthetic_frame()
    with pytest.raises(ValueError):
        carve_validation_split(df, seed=1, config=ValidationSplitConfig(fraction=0.0))


def test_carve_validation_split_rejects_empty_train():
    with pytest.raises(RuntimeError):
        carve_validation_split(pd.DataFrame(columns=["target", "project"]), seed=1)


def test_svm_select_c_picks_from_grid_and_reports_diagnostics():
    df = _synthetic_frame(n_per_class=15)
    grid = [0.01, 1.0, 100.0]
    config, diagnostics = select_c(df, seed=1, grid=grid, validation=ValidationSplitConfig(fraction=0.2))
    assert config.C in grid
    assert diagnostics["selected_C"] == config.C
    assert set(diagnostics["grid_scores"]) == set(grid)
    assert diagnostics["train_prime_rows"] + diagnostics["dev_rows"] == len(df)


def test_svm_select_c_requires_nonempty_grid():
    df = _synthetic_frame()
    with pytest.raises(ValueError):
        select_c(df, seed=1, grid=[], validation=ValidationSplitConfig(fraction=0.2))


def test_bilstm_select_num_epochs_returns_positive_epoch_count_within_budget():
    pytest.importorskip("torch")
    from src.training.bilstm import BiLstmConfig, select_num_epochs

    df = _synthetic_frame(n_per_class=12)
    tiny = BiLstmConfig(max_vocab_size=200, max_seq_len=16, embedding_dim=8, hidden_dim=8, batch_size=8)
    config, diagnostics = select_num_epochs(
        df, seed=1, max_epochs=2, patience=1,
        validation=ValidationSplitConfig(fraction=0.25), base_config=tiny)
    assert 1 <= config.num_epochs <= 2
    assert diagnostics["selected_num_epochs"] == config.num_epochs
    assert len(diagnostics["dev_scores"]) >= 1
    assert diagnostics["train_prime_rows"] + diagnostics["dev_rows"] == len(df)
