import asyncio
import threading
import time
import types
from dataclasses import replace

import pytest

from src.inference import ollama


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(ollama.tiktoken, "get_encoding", lambda name: types.SimpleNamespace(
        encode=lambda text: list(text), decode=lambda tokens: "".join(tokens)))
    return ollama.OllamaBackend(replace(ollama.OllamaRuntimeConfig(), retry_delay_seconds=0,
                                       degenerate_retry_delay_seconds=0))


def arguments():
    return dict(model="mock", messages=[], prepared_input="input", seed=10,
                temperature=.6, request_timeout_seconds=1, task_labels={"fix", "feat"})


def test_every_retry_preserved_and_final_invalid(backend, monkeypatch):
    responses = iter(["ab" * 50, RuntimeError("transport failed"), RuntimeError("again")])
    def chat(**kwargs):
        response = next(responses)
        if isinstance(response, Exception): raise response
        return {"message": {"content": response}}
    monkeypatch.setattr(ollama, "chat", chat)
    attempts = []
    result = asyncio.run(backend.infer(**arguments(), attempt_sink=attempts.append))
    assert [a["technical_status"] for a in attempts] == ["degenerate_response", "failure", "failure"]
    assert attempts[0]["raw_response"] == "ab" * 50
    assert "transport failed" in attempts[1]["technical_detail"]
    assert result.parsed.prediction == "<invalid_output>"


def test_timeout_drains_before_replacement(backend, monkeypatch):
    active = maximum = calls = 0
    lock = threading.Lock()
    def chat(**kwargs):
        nonlocal active, maximum, calls
        with lock:
            active += 1; calls += 1; maximum = max(maximum, active)
        time.sleep(.05)
        with lock: active -= 1
        return {"message": {"content": "fix"}}
    monkeypatch.setattr(ollama, "chat", chat)
    attempts = []
    async def execute():
        semaphore = asyncio.Semaphore(1)
        async def one():
            async with semaphore:
                return await backend.infer(**(arguments() | {"request_timeout_seconds": .001}),
                                           attempt_sink=attempts.append)
        return await asyncio.gather(one(), one())
    results = asyncio.run(execute())
    assert active == 0 and maximum == 1 and calls == 2
    assert all(r.technical_status == "timeout" and r.parsed.prediction == "<invalid_output>" for r in results)
    assert all(a["raw_response"] == "fix" for a in attempts)


def test_cancellation_drains_request(backend, monkeypatch):
    done = threading.Event()
    started = threading.Event()
    def chat(**kwargs):
        started.set()
        time.sleep(.04)
        done.set()
        return {"message": {"content": "fix"}}
    monkeypatch.setattr(ollama, "chat", chat)
    attempts = []
    async def execute():
        task = asyncio.create_task(backend.infer(**arguments(), attempt_sink=attempts.append))
        while not started.is_set(): await asyncio.sleep(.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert done.is_set()
    asyncio.run(execute())
    assert attempts[0]["technical_status"] == "cancelled"


def test_success_after_retry(backend, monkeypatch):
    responses = iter(["xy" * 50, "fix"])
    monkeypatch.setattr(ollama, "chat", lambda **kwargs: {"message": {"content": next(responses)}})
    attempts = []
    result = asyncio.run(backend.infer(**arguments(), attempt_sink=attempts.append))
    assert len(attempts) == 2 and result.parsed.prediction == "fix"
