from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Mapping, Any, Callable

import tiktoken
from ollama import chat, list as list_models

def parse_strict(*args, **kwargs):
    from ..parsing.strict import parse_strict as canonical
    return canonical(*args, **kwargs)


def installed_model_identity(tag: str, expected_digest: str) -> dict:
    response = list_models()
    matches = [m for m in response.models if m.model == tag]
    if len(matches) != 1:
        raise RuntimeError(f"Installed Ollama model not found uniquely: {tag}")
    digest = matches[0].digest
    if not expected_digest or digest != expected_digest:
        raise RuntimeError(f"Installed model digest mismatch for {tag}: {digest}")
    return {"tag": tag, "digest": digest}



@dataclass(frozen=True)
class OllamaRuntimeConfig:
    max_content_tokens: int = 30000
    truncation_marker: str = "\n\n[TRUNCATED...]"
    min_num_ctx: int = 4096
    generation_buffer_tokens: int = 1024
    max_num_ctx: int = 32048
    num_predict: int = 1024
    keep_alive: str = "30m"
    max_attempts: int = 3
    retry_delay_seconds: float = 5.0
    degenerate_retry_delay_seconds: float = 2.0
    restart_after_consecutive_failures: int = 3
    restart_delay_seconds: float = 3.0

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "OllamaRuntimeConfig":
        """Build from the current flat config/inference.yaml schema."""
        inp = cfg["input"]
        ctx = cfg["context"]
        gen = cfg["generation"]
        exe = cfg["execution"]
        return cls(
            max_content_tokens=int(inp["max_content_tokens"]),
            truncation_marker=str(inp["truncation"]["marker"]),
            min_num_ctx=int(ctx["min_num_ctx"]),
            generation_buffer_tokens=int(ctx["generation_buffer_tokens"]),
            max_num_ctx=int(ctx["max_num_ctx"]),
            num_predict=int(gen["num_predict"]),
            keep_alive=str(gen["keep_alive"]),
            max_attempts=int(exe["max_attempts"]),
            retry_delay_seconds=float(exe["retry_delay_seconds"]),
            degenerate_retry_delay_seconds=float(exe["degenerate_response_retry_delay_seconds"]),
            restart_after_consecutive_failures=int(exe["model_health"]["restart_after_consecutive_failures"]),
            restart_delay_seconds=float(exe["model_health"]["restart_delay_seconds"]),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OllamaRuntimeConfig":
        from ..config import load_inference_config
        return cls.from_mapping(load_inference_config(path))


@dataclass(frozen=True)
class OllamaCallResult:
    raw_response: str
    parsed: Any
    technical_status: str
    technical_detail: str
    processing_time_seconds: float
    input_tokens: int
    num_ctx: int


class ModelHealthMonitor:
    def __init__(self, model: str, *, threshold: int = 3, restart_delay_seconds: float = 3.0):
        self.model = model
        self.threshold = threshold
        self.restart_delay_seconds = restart_delay_seconds
        self.consecutive_failures = 0
        self.consecutive_timeouts = 0
        self.total_restarts = 0

    def success(self) -> None:
        self.consecutive_failures = 0
        self.consecutive_timeouts = 0

    async def failure(self, *, timeout: bool) -> None:
        if timeout:
            self.consecutive_timeouts += 1
            reached = self.consecutive_timeouts >= self.threshold
        else:
            self.consecutive_failures += 1
            reached = self.consecutive_failures >= self.threshold
        if reached:
            await self.restart()

    async def restart(self) -> None:
        subprocess.run(
            ["ollama", "stop", self.model],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        await asyncio.sleep(self.restart_delay_seconds)
        await asyncio.to_thread(
            chat, model=self.model, messages=[{"role": "user", "content": "test"}]
        )
        self.consecutive_failures = 0
        self.consecutive_timeouts = 0
        self.total_restarts += 1


class OllamaBackend:
    """Shared Ollama mechanics.

    Every returned model response is evaluated immediately with
    strict_response_parser v4.1.1.
    """

    def __init__(self, config: OllamaRuntimeConfig | None = None):
        self.config = config or OllamaRuntimeConfig()
        self._enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(str(text)))

    def prepare_content(self, text: str) -> tuple[str, int]:
        tokens = self._enc.encode(str(text))
        if len(tokens) <= self.config.max_content_tokens:
            return str(text), len(tokens)
        clipped = self._enc.decode(tokens[: self.config.max_content_tokens]) + self.config.truncation_marker
        return clipped, self.config.max_content_tokens

    def num_ctx(self, input_tokens: int) -> int:
        return max(
            self.config.min_num_ctx,
            min(input_tokens + self.config.generation_buffer_tokens, self.config.max_num_ctx),
        )

    @staticmethod
    def is_degenerate(text: str) -> bool:
        if not text or len(text) < 10:
            return False
        t = text.strip()
        unique_chars = len(set(t))
        if unique_chars <= 3 and len(t) > 20:
            return True
        if len(t) > 30:
            for plen in range(2, 11):
                pattern = t[:plen]
                if pattern.strip() and (t.count(pattern) * plen) / len(t) > 0.7:
                    return True
        if len(t) > 100 and unique_chars < 15:
            low = t.casefold()
            if "classification:" not in low and "classification :" not in low:
                return True
        return False

    async def infer(
        self,
        *,
        model: str,
        messages: list[dict],
        prepared_input: str,
        seed: int,
        temperature: float,
        request_timeout_seconds: float,
        task_labels: Collection[str],
        health_monitor: ModelHealthMonitor | None = None,
        attempt_sink: Callable[[dict], None] | None = None,
    ) -> OllamaCallResult:
        input_tokens = self.count_tokens(prepared_input)
        ctx = self.num_ctx(input_tokens)
        monitor = health_monitor or ModelHealthMonitor(
            model,
            threshold=self.config.restart_after_consecutive_failures,
            restart_delay_seconds=self.config.restart_delay_seconds,
        )
        start = time.monotonic()
        last_raw = ""
        last_detail = ""

        if self.config.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        for attempt in range(1, self.config.max_attempts + 1):
            attempt_start = time.monotonic()
            raw, technical, detail = "", "ok", ""
            task = asyncio.create_task(asyncio.to_thread(
                chat, model=model, messages=messages,
                options={"temperature": float(temperature), "seed": int(seed),
                         "num_ctx": int(ctx), "num_predict": int(self.config.num_predict)},
                keep_alive=self.config.keep_alive,
            ))
            cancelled = False
            try:
                response = await asyncio.wait_for(asyncio.shield(task), request_timeout_seconds)
                raw = str(response["message"]["content"])
                if self.is_degenerate(raw):
                    technical, detail = "degenerate_response", "degenerate_response"
            except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
                cancelled = isinstance(exc, asyncio.CancelledError)
                technical = "cancelled" if cancelled else "timeout"
                detail = f"request_timeout={request_timeout_seconds}; blocking request drained before releasing slot"
                # A Python thread cannot be cancelled. Retain the caller's slot
                # until it exits, even under repeated cancellation. No restart
                # or replacement request is launched while this call is live.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        cancelled = True
                    except Exception:
                        break
                try:
                    raw = str(task.result()["message"]["content"])
                except Exception as drained:
                    detail += f"; {type(drained).__name__}: {drained}"
            except Exception as exc:
                technical, detail = "failure", f"{type(exc).__name__}: {exc}"
            evidence = {
                "attempt": attempt, "raw_response": raw,
                "technical_status": technical, "technical_detail": detail,
                "processing_time_seconds": time.monotonic() - attempt_start,
            }
            if attempt_sink is not None:
                attempt_sink(evidence)
            if cancelled:
                raise asyncio.CancelledError
            if technical == "ok":
                monitor.success()
            # Automatic model restart is unsafe with other requests in flight.
            # Technical failures are recorded; orchestration can be resumed.
            final = technical in {"ok", "timeout"} or attempt == self.config.max_attempts
            if final:
                parsed = parse_strict(raw, task_labels=task_labels, technical_status=technical)
                return OllamaCallResult(raw, parsed, technical, detail,
                                        time.monotonic() - start, input_tokens, ctx)
            delay = (self.config.degenerate_retry_delay_seconds if technical == "degenerate_response"
                     else self.config.retry_delay_seconds)
            await asyncio.sleep(delay)
        raise RuntimeError("Unreachable inference state")
