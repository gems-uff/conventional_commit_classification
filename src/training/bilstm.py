from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

LABELS = ("fix", "feat", "build", "ci", "docs", "style", "refactor", "perf", "test", "chore", "others")
LABEL2ID = {x: i for i, x in enumerate(LABELS)}; ID2LABEL = {i: x for x, i in LABEL2ID.items()}
PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"


@dataclass(frozen=True)
class BiLstmConfig:
    max_vocab_size: int = 50000
    max_seq_len: int = 512
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*|[0-9]+|==|!=|<=|>=|->|=>|[{}()\[\].,;:+\-*/=<>]", str(text).lower())


def set_all_seeds(seed: int) -> None:
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def train_predict(train: pd.DataFrame, test: pd.DataFrame, *, seed: int, config: BiLstmConfig | None = None) -> pd.DataFrame:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    c = config or BiLstmConfig(); set_all_seeds(seed)
    counts = Counter()
    for text in train["input_text"]: counts.update(tokenize(text))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in counts.most_common(c.max_vocab_size - 2): vocab[token] = len(vocab)

    def encode(text: str):
        ids = [vocab.get(t, 1) for t in tokenize(text)[:c.max_seq_len]]
        return ids + [0] * (c.max_seq_len - len(ids))

    class DS(Dataset):
        def __init__(self, frame): self.frame = frame.reset_index(drop=True)
        def __len__(self): return len(self.frame)
        def __getitem__(self, i):
            r = self.frame.iloc[i]
            return torch.tensor(encode(r.input_text), dtype=torch.long), torch.tensor(LABEL2ID[r.target], dtype=torch.long)

    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.emb = nn.Embedding(len(vocab), c.embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(c.embedding_dim, c.hidden_dim, num_layers=c.num_layers, batch_first=True,
                                bidirectional=True, dropout=c.dropout if c.num_layers > 1 else 0.0)
            self.drop = nn.Dropout(c.dropout); self.fc = nn.Linear(c.hidden_dim * 2, len(LABELS))
        def forward(self, x):
            _, (h, _) = self.lstm(self.emb(x)); return self.fc(self.drop(torch.cat([h[-2], h[-1]], dim=1)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    class_counts = train["target"].value_counts().reindex(LABELS).fillna(0).to_numpy(float)
    if np.any(class_counts == 0): raise RuntimeError("Training split is missing one or more classes")
    weights = torch.tensor(len(train) / (len(LABELS) * class_counts), dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(DS(train), batch_size=c.batch_size, shuffle=True, generator=gen)
    for _ in range(c.num_epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device); optimizer.zero_grad(); loss = criterion(model(x), y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), c.gradient_clip_norm); optimizer.step()
    model.eval(); preds = []
    with torch.no_grad():
        for x, _ in DataLoader(DS(test), batch_size=c.batch_size, shuffle=False):
            preds.extend(torch.argmax(model(x.to(device)), dim=1).cpu().tolist())
    out = test.copy(); out["prediction"] = [ID2LABEL[i] for i in preds]; out["seed"] = seed
    out["correct"] = out["target"] == out["prediction"]
    return out
