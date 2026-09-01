from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

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
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, tokenizer=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer=tokenizer))
    trainer.train(); pred = trainer.predict(test_ds); ids = np.argmax(pred.predictions, axis=-1)
    out = test.copy(); out["prediction"] = [ID2LABEL[int(i)] for i in ids]; out["seed"] = seed
    out["correct"] = out["target"] == out["prediction"]
    return out
