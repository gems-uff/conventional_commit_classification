from __future__ import annotations

import gc
import random
import time
from dataclasses import dataclass
from typing import Collection

import numpy as np

from ..parsing.strict import PredictionParse, parse_strict


@dataclass(frozen=True)
class FlaccsConfig:
    model: str = "0x404/ccs-code-llama-7b"
    revision: str | None = None
    max_new_tokens: int = 10
    do_sample: bool = False


@dataclass(frozen=True)
class FlaccsCallResult:
    raw_response: str
    parsed: PredictionParse
    technical_status: str
    technical_detail: str
    processing_time_seconds: float


class FlaccsBackend:
    """FLACCS text-generation adapter using the artifact's native greedy decoding."""

    def __init__(self, config: FlaccsConfig | None = None):
        self.config = config or FlaccsConfig()
        self._pipe = None

    def load(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from transformers import pipeline
        kwargs = {
            "model": self.config.model,
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
        if self.config.revision:
            kwargs["revision"] = self.config.revision
        self._pipe = pipeline("text-generation", **kwargs)
        self._pipe.tokenizer.pad_token = self._pipe.tokenizer.eos_token
        return self._pipe

    @staticmethod
    def set_seed(seed: int) -> None:
        import torch
        from transformers import set_seed
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        set_seed(seed)

    def infer(self, *, prompt: str, seed: int, task_labels: Collection[str]) -> FlaccsCallResult:
        import torch
        pipe = self.load()
        self.set_seed(seed)
        start = time.monotonic()
        try:
            # Native FLACCS configuration: deterministic greedy decoding. Do not
            # pass temperature/top_p when sampling is disabled.
            raw = pipe(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=pipe.tokenizer.eos_token_id,
                return_full_text=False,
            )[0]["generated_text"]
            technical = "ok"
            detail = ""
        except torch.cuda.OutOfMemoryError as exc:
            raw = "<failure>"; technical = "cuda_oom"; detail = str(exc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as exc:
            raw = "<failure>"; technical = "failure"; detail = f"{type(exc).__name__}: {exc}"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        parsed = parse_strict(raw, task_labels=task_labels, technical_status=technical)
        return FlaccsCallResult(raw, parsed, technical, detail, time.monotonic() - start)
