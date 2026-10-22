"""Direct tests for the trainable RQ4 baselines in src/training/.

Covers the three ``train_predict()`` functions and ``codebert.select_num_epochs``
(``svm.select_c``/``bilstm.select_num_epochs`` live in test_model_selection.py).
BiLSTM and CodeBERT tests need ``torch`` (and ``transformers``/``datasets`` for
CodeBERT) and are skipped otherwise. CodeBERT runs against a tiny, randomly
initialized BERT written to a temp dir -- no download of microsoft/codebert-base --
so these tests check the training/prediction wiring, not model quality.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import STUDY_LABELS
from src.training import svm
from src.training.model_selection import ValidationSplitConfig

OUTPUT_COLUMNS = {"prediction", "seed", "correct"}


def _frame(n_per_class=6, classes=STUDY_LABELS, projects=("p1", "p2", "p3"), prefix="c"):
    """Synthetic commits whose text carries a class-specific token, so the task is learnable."""
    rows = []
    for cls_idx, cls in enumerate(classes):
        for i in range(n_per_class):
            rows.append({
                "project": projects[(cls_idx * n_per_class + i) % len(projects)],
                "commit_hash": f"{prefix}-{cls}-{i}",
                "target": cls,
                "input_text": f"{cls}token {cls}token change {i % 3}",
            })
    return pd.DataFrame(rows)


def _assert_output_contract(out, test, seed, labels=STUDY_LABELS):
    assert list(out.index) == list(test.index)
    assert OUTPUT_COLUMNS <= set(out.columns)
    assert OUTPUT_COLUMNS.isdisjoint(test.columns), "train_predict must not mutate its test frame"
    pd.testing.assert_frame_equal(out[list(test.columns)], test)
    assert set(out["prediction"]) <= set(labels)
    assert (out["seed"] == seed).all()
    assert (out["correct"] == (out["target"] == out["prediction"])).all()


# TF-IDF + LinearSVC --------------------------------------------------------------

def test_svm_train_predict_output_contract_and_learns_separable_task():
    train, test = _frame(), _frame(n_per_class=2, prefix="t")
    out = svm.train_predict(train, test, seed=10)
    _assert_output_contract(out, test, seed=10)
    assert out["correct"].all()


def test_svm_train_predict_is_deterministic_for_a_seed():
    train, test = _frame(), _frame(n_per_class=2, prefix="t")
    a = svm.train_predict(train, test, seed=10)
    b = svm.train_predict(train, test, seed=10)
    assert a["prediction"].tolist() == b["prediction"].tolist()


@pytest.mark.parametrize("side", ["train", "test"])
def test_svm_train_predict_rejects_missing_columns(side):
    train, test = _frame(), _frame(n_per_class=2, prefix="t")
    frames = {"train": train, "test": test}
    frames[side] = frames[side].drop(columns=["commit_hash"])
    with pytest.raises(RuntimeError, match=f"{side}: missing columns"):
        svm.train_predict(frames["train"], frames["test"], seed=10)


def test_svm_build_model_uses_config_and_seed():
    model = svm.build_model(7, svm.SvmConfig(C=0.5, max_features=123))
    assert model.named_steps["svm"].C == 0.5
    assert model.named_steps["svm"].random_state == 7
    assert model.named_steps["tfidf"].max_features == 123


# BiLSTM --------------------------------------------------------------------------

def _tiny_bilstm(**overrides):
    from src.training.bilstm import BiLstmConfig
    base = dict(max_vocab_size=200, max_seq_len=16, embedding_dim=8, hidden_dim=8, batch_size=8, num_epochs=1)
    return BiLstmConfig(**{**base, **overrides})


def test_bilstm_train_predict_output_contract():
    pytest.importorskip("torch")
    from src.training import bilstm
    train, test = _frame(), _frame(n_per_class=2, prefix="t")
    out = bilstm.train_predict(train, test, seed=10, config=_tiny_bilstm())
    _assert_output_contract(out, test, seed=10)


def test_bilstm_train_predict_is_deterministic_for_a_seed():
    pytest.importorskip("torch")
    from src.training import bilstm
    train, test = _frame(), _frame(n_per_class=2, prefix="t")
    a = bilstm.train_predict(train, test, seed=10, config=_tiny_bilstm(num_epochs=2))
    b = bilstm.train_predict(train, test, seed=10, config=_tiny_bilstm(num_epochs=2))
    assert a["prediction"].tolist() == b["prediction"].tolist()


def test_bilstm_train_predict_rejects_training_split_missing_a_class():
    pytest.importorskip("torch")
    from src.training import bilstm
    train = _frame(classes=[l for l in STUDY_LABELS if l != "perf"])
    with pytest.raises(RuntimeError, match="missing one or more classes"):
        bilstm.train_predict(train, _frame(n_per_class=1, prefix="t"), seed=10, config=_tiny_bilstm())


def test_bilstm_train_predict_truncates_to_max_seq_len():
    pytest.importorskip("torch")
    from src.training import bilstm
    train = _frame()
    test = _frame(n_per_class=1, prefix="t")
    test["input_text"] = test["input_text"] + " filler" * 500
    out = bilstm.train_predict(train, test, seed=10, config=_tiny_bilstm(max_seq_len=4))
    _assert_output_contract(out, test, seed=10)


# CodeBERT ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_bert(tmp_path_factory) -> Path:
    """A tiny random BERT + WordPiece tokenizer saved locally, standing in for codebert-base."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    from src.training.codebert import ID2LABEL, LABEL2ID, LABELS

    path = tmp_path_factory.mktemp("tiny_bert")
    words = sorted({w for text in _frame()["input_text"] for w in text.split()})
    vocab = path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]) + "\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab))
    tokenizer.save_pretrained(path)
    config = transformers.BertConfig(
        vocab_size=tokenizer.vocab_size, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64,
        num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    transformers.BertForSequenceClassification(config).save_pretrained(path)
    return path


def _tiny_codebert(model_path: Path, **overrides):
    from src.training.codebert import CodeBertConfig
    base = dict(model_name=str(model_path), max_length=32, num_train_epochs=1,
                learning_rate=1e-3, train_batch_size=16, eval_batch_size=16)
    return CodeBertConfig(**{**base, **overrides})


def test_codebert_train_predict_output_contract(tiny_bert, tmp_path):
    from src.training import codebert
    train, test = _frame(), _frame(n_per_class=2, prefix="t")
    out = codebert.train_predict(train, test, seed=10, output_dir=str(tmp_path / "run"),
                                 config=_tiny_codebert(tiny_bert))
    _assert_output_contract(out, test, seed=10)


def test_codebert_train_predict_truncates_long_inputs_to_max_length(tiny_bert, tmp_path):
    from src.training import codebert
    train = _frame()
    test = _frame(n_per_class=1, prefix="t")
    # Far beyond max_position_embeddings=64: only works if truncation honors max_length.
    test["input_text"] = test["input_text"] + " change" * 500
    out = codebert.train_predict(train, test, seed=10, output_dir=str(tmp_path / "run"),
                                 config=_tiny_codebert(tiny_bert, max_length=16))
    _assert_output_contract(out, test, seed=10)


def test_codebert_select_num_epochs_picks_logged_epoch_and_cleans_up(tiny_bert, tmp_path):
    from src.training import codebert
    output_dir = tmp_path / "selection"
    base = _tiny_codebert(tiny_bert, num_train_epochs=99)
    config, diagnostics = codebert.select_num_epochs(
        _frame(n_per_class=8), seed=10, max_epochs=3, patience=1, output_dir=str(output_dir),
        validation=ValidationSplitConfig(fraction=0.25), base_config=base)

    assert 1 <= config.num_train_epochs <= 3
    assert diagnostics["selected_num_epochs"] == config.num_train_epochs
    epochs = [r["epoch"] for r in diagnostics["dev_scores"]]
    assert 1 <= len(epochs) <= 3 and epochs == sorted(epochs)
    best = max(diagnostics["dev_scores"], key=lambda r: r["macro_f1"])
    assert diagnostics["best_dev_macro_f1"] == best["macro_f1"]
    assert config.num_train_epochs == round(best["epoch"])
    assert diagnostics["train_prime_rows"] + diagnostics["dev_rows"] == len(_frame(n_per_class=8))
    # Only the epoch count changes; the rest of the base config is carried over.
    assert config.model_name == base.model_name and config.max_length == base.max_length
    # Selection scratch space (HF checkpoints) is removed: no selection model is kept.
    assert not output_dir.exists()


def test_codebert_select_num_epochs_respects_grouped_validation(tiny_bert, tmp_path):
    from src.training import codebert
    df = _frame(n_per_class=9, projects=("p1", "p2", "p3"))
    _, diagnostics = codebert.select_num_epochs(
        df, seed=10, max_epochs=1, patience=1, output_dir=str(tmp_path / "selection"),
        validation=ValidationSplitConfig(fraction=0.3), group_col="project",
        base_config=_tiny_codebert(tiny_bert))
    # Target is round(99 * 0.3) = 30 rows; whole projects (33 rows each) are
    # added until it is reached, so exactly one project goes to dev. A
    # row-level split would have produced 30.
    assert diagnostics["dev_rows"] == len(df) // 3


@pytest.mark.parametrize("max_epochs,patience", [(0, 1), (3, 0), (-1, 2)])
def test_codebert_select_num_epochs_rejects_non_positive_budget(max_epochs, patience, tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    from src.training import codebert
    with pytest.raises(ValueError, match="must be positive"):
        codebert.select_num_epochs(_frame(), seed=10, max_epochs=max_epochs, patience=patience,
                                   output_dir=str(tmp_path / "selection"))


# run_baselines_experiment.py --limit ---------------------------------------------

def test_smoke_train_subset_is_class_balanced_across_projects_and_feeds_selection():
    from scripts.run_baselines_experiment import _smoke_train_subset
    from src.training.model_selection import carve_validation_split

    # Rows sorted by project, as in the fold files: a plain head() would hold one project.
    train = _frame(n_per_class=20, projects=tuple(f"p{i}" for i in range(10))).sort_values("project")
    subset = _smoke_train_subset(train, per_class=7)

    assert subset["target"].value_counts().max() <= 7
    assert set(subset["target"]) == set(train["target"])
    assert subset.groupby("target")["project"].nunique().min() >= 2
    assert list(subset.index) == sorted(subset.index, key=list(train.index).index)
    for group_col in ("project", None):
        carve_validation_split(subset, seed=0, config=ValidationSplitConfig(fraction=0.15), group_col=group_col)
