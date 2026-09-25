from __future__ import annotations

import gc
import random
import time
from dataclasses import dataclass
from typing import Collection

import numpy as np

from ..parsing.strict import PredictionParse, parse_strict

MESSAGE_PLACEHOLDER = "{{COMMIT_MESSAGE}}"
DIFF_PLACEHOLDER = "{{COMMIT_DIFF}}"


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

    def build_prompt(
        self, *, template: str, message: str, diff: str,
        message_max_tokens: int, context_window_tokens: int,
    ) -> str:
        """Render an artifact-compatible FLACCS prompt with tokenizer-level truncation.

        Used by the artifact-compatible RQ4 condition, whose ``message_max_tokens``
        and ``context_window_tokens`` come from ``config/experiments/rq4/artifact_flaccs.yaml``.
        The template's literal text outside the two placeholders (the system
        preamble and the short "given commit message/diff" labels) is treated as
        fixed, unconstrained overhead. Only the message and diff *content* are
        capped: the message to ``message_max_tokens`` tokens, and the diff to
        whatever remains of ``context_window_tokens`` after that overhead and the
        (possibly truncated) message.

        This differs slightly from the FLACCS artifact's own historical
        preparation script, which additionally charged a handful of tokens of
        wrapper prose ("- given commit message:\\n") against the message cap.
        The difference is a handful of tokens out of a 1,024-token budget and
        does not change which commits require truncation --
        ``data/rq4/artifact_compatible/flaccs/assignments.csv`` already restricts
        this condition to commits whose original, untruncated prompt fits the
        budget (``fits_1024``).
        """
        if MESSAGE_PLACEHOLDER not in template or DIFF_PLACEHOLDER not in template:
            raise RuntimeError(f"FLACCS prompt template is missing {MESSAGE_PLACEHOLDER}/{DIFF_PLACEHOLDER}")
        head, rest = template.split(MESSAGE_PLACEHOLDER, 1)
        mid, tail = rest.split(DIFF_PLACEHOLDER, 1)

        tokenizer = self.load().tokenizer

        def enc(text: str, max_len: int | None = None) -> list[int]:
            kwargs = {"add_special_tokens": False}
            if max_len is not None:
                kwargs["max_length"] = max_len
                kwargs["truncation"] = True
            return tokenizer.encode(text, **kwargs)

        head_ids = enc(head)
        mid_ids = enc(mid)
        tail_ids = enc(tail)
        message_ids = enc(message or "", message_max_tokens)
        overhead = len(head_ids) + len(mid_ids) + len(tail_ids) + len(message_ids)
        remaining = max(1, context_window_tokens - overhead)
        diff_ids = enc(diff or "", remaining)
        return tokenizer.decode(head_ids + message_ids + mid_ids + diff_ids + tail_ids)

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
