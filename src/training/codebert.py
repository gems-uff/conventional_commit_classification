from __future__ import annotations

import random
import shutil
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from .model_selection import ValidationSplitConfig, carve_validation_split, macro_f1

LABELS = ("fix", "feat", "build", "ci", "docs", "style", "refactor", "perf", "test", "chore", "others")
LABEL2ID = {x: i for i, x in enumerate(LABELS)}; ID2LABEL = {i: x for x, i in LABEL2ID.items()}


@dataclass(frozen=True)
class CodeBertConfig:
    model_name: str = "microsoft/codebert-base"
    max_length: int = 512
    num_train_epochs: int = 3
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 8
    weight_decay: float = 0.01


def _tokenizer_kwarg(tokenizer, trainer_cls=None) -> dict[str, Any]:
    """Pass the tokenizer as `processing_class` (transformers >= 4.46) or `tokenizer`."""
    import inspect
    if trainer_cls is None:
        from transformers import Trainer as trainer_cls
    name = "processing_class" if "processing_class" in inspect.signature(trainer_cls.__init__).parameters else "tokenizer"
    return {name: tokenizer}


def set_all(seed: int) -> None:
    import torch
    from transformers import set_seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    set_seed(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def train_predict(train: pd.DataFrame, test: pd.DataFrame, *, seed: int, output_dir: str, config: CodeBertConfig | None = None) -> pd.DataFrame:
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
    c = config or CodeBertConfig(); set_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(c.model_name)
    def ds(frame):
        tmp = frame[["input_text", "target"]].copy(); tmp["label"] = tmp["target"].map(LABEL2ID)
        d = Dataset.from_pandas(tmp[["input_text", "label"]], preserve_index=False)
        return d.map(lambda b: tokenizer(b["input_text"], truncation=True, max_length=c.max_length), batched=True)
    train_ds, test_ds = ds(train), ds(test)
    model = AutoModelForSequenceClassification.from_pretrained(c.model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    args = TrainingArguments(output_dir=output_dir, seed=seed, data_seed=seed,
        num_train_epochs=c.num_train_epochs, learning_rate=c.learning_rate,
        per_device_train_batch_size=c.train_batch_size, per_device_eval_batch_size=c.eval_batch_size,
        weight_decay=c.weight_decay, eval_strategy="no", save_strategy="no", load_best_model_at_end=False,
        report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, **_tokenizer_kwarg(tokenizer),
                      data_collator=DataCollatorWithPadding(tokenizer=tokenizer))
    trainer.train(); pred = trainer.predict(test_ds); ids = np.argmax(pred.predictions, axis=-1)
    out = test.copy(); out["prediction"] = [ID2LABEL[int(i)] for i in ids]; out["seed"] = seed
    out["correct"] = out["target"] == out["prediction"]
    return out


def select_num_epochs(
    train: pd.DataFrame,
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    output_dir: str,
    validation: ValidationSplitConfig | None = None,
    group_col: str | None = None,
    base_config: CodeBertConfig | None = None,
) -> tuple[CodeBertConfig, dict[str, Any]]:
    """Pick a fixed epoch budget via one held-out split carved from `train`.

    Runs once per fold/holdout with early stopping on dev Macro-F1. The resulting
    epoch count is then reused, fixed, for every seed's from-scratch fit in
    `train_predict` for that fold/holdout. `output_dir` is used only as HF Trainer
    scratch space for this selection run and is removed before returning: no
    model checkpoint from selection is kept, only the chosen epoch count.
    """
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                               EarlyStoppingCallback, Trainer, TrainingArguments)

    if max_epochs <= 0 or patience <= 0:
        raise ValueError("max_epochs and patience must be positive")
    c = base_config or CodeBertConfig()
    train_prime, dev = carve_validation_split(train, seed=seed, config=validation, group_col=group_col)
    set_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(c.model_name)

    def ds(frame):
        tmp = frame[["input_text", "target"]].copy(); tmp["label"] = tmp["target"].map(LABEL2ID)
        d = Dataset.from_pandas(tmp[["input_text", "label"]], preserve_index=False)
        return d.map(lambda b: tokenizer(b["input_text"], truncation=True, max_length=c.max_length), batched=True)

    train_ds, dev_ds = ds(train_prime), ds(dev)
    model = AutoModelForSequenceClassification.from_pretrained(
        c.model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"macro_f1": macro_f1(labels, preds)}

    try:
        args = TrainingArguments(
            output_dir=output_dir, seed=seed, data_seed=seed,
            num_train_epochs=max_epochs, learning_rate=c.learning_rate,
            per_device_train_batch_size=c.train_batch_size, per_device_eval_batch_size=c.eval_batch_size,
            weight_decay=c.weight_decay, eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
            save_total_limit=1, report_to="none")
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds,
                          **_tokenizer_kwarg(tokenizer), data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                          compute_metrics=compute_metrics,
                          callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)])
        trainer.train()
        dev_scores = [
            {"epoch": r["epoch"], "macro_f1": r["eval_macro_f1"]}
            for r in trainer.state.log_history if "eval_macro_f1" in r
        ]
        if not dev_scores:
            raise RuntimeError("No evaluation rounds were logged during epoch selection")
        best = max(dev_scores, key=lambda r: r["macro_f1"])
        best_epoch = max(1, round(best["epoch"]))
        diagnostics = {
            "dev_scores": dev_scores, "selected_num_epochs": best_epoch, "best_dev_macro_f1": best["macro_f1"],
            "train_prime_rows": len(train_prime), "dev_rows": len(dev),
        }
        return replace(c, num_train_epochs=best_epoch), diagnostics
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
