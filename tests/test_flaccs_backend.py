"""Direct tests for src/inference/flaccs.py.

Run without the real model, torch or transformers: the tokenizer and the
text-generation pipeline are replaced by fakes (``FlaccsBackend.load()`` returns
``_pipe`` when it is already set), and ``torch`` is stubbed in ``sys.modules``
for ``infer()``. One optional test uses the real FLACCS tokenizer when it is
already in the local Hugging Face cache.
"""
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data.loader import STUDY_LABELS
from src.inference import flaccs
from src.inference.flaccs import DIFF_PLACEHOLDER, MESSAGE_PLACEHOLDER, FlaccsBackend
from src.parsing.strict import INVALID_OUTPUT

ROOT = Path(__file__).resolve().parents[1]
FLACCS_LABELS = tuple(l for l in STUDY_LABELS if l != "others")
TEMPLATE = f"H{MESSAGE_PLACEHOLDER}M{DIFF_PLACEHOLDER}T"


class CharTokenizer:
    """One token per character, so token budgets map exactly to string lengths."""

    eos_token_id = 0

    def __init__(self):
        self.encode_kwargs = []

    def encode(self, text, add_special_tokens=True, max_length=None, truncation=False):
        self.encode_kwargs.append({"add_special_tokens": add_special_tokens,
                                   "max_length": max_length, "truncation": truncation})
        ids = [ord(c) for c in text]
        return ids[:max_length] if truncation and max_length is not None else ids

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakePipe:
    def __init__(self, response="fix", error=None):
        self.tokenizer = CharTokenizer()
        self.response = response
        self.error = error
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.error is not None:
            raise self.error
        return [{"generated_text": self.response}]


class FakeOutOfMemoryError(RuntimeError):
    pass


def backend_with(pipe) -> FlaccsBackend:
    backend = FlaccsBackend()
    backend._pipe = pipe
    return backend


def build(backend, *, message, diff, message_max_tokens=64, context_window_tokens=1024, template=TEMPLATE):
    return backend.build_prompt(template=template, message=message, diff=diff,
                                message_max_tokens=message_max_tokens,
                                context_window_tokens=context_window_tokens)


# build_prompt() ----------------------------------------------------------------

def test_build_prompt_keeps_short_inputs_whole_and_in_order():
    assert build(backend_with(FakePipe()), message="msg", diff="diff") == "HmsgMdiffT"


def test_build_prompt_truncates_message_to_message_max_tokens():
    out = build(backend_with(FakePipe()), message="a" * 100, diff="d", message_max_tokens=10)
    assert out == "H" + "a" * 10 + "MdT"


def test_build_prompt_truncates_diff_to_remaining_window():
    out = build(backend_with(FakePipe()), message="m" * 5, diff="d" * 500,
                message_max_tokens=10, context_window_tokens=50)
    # overhead = H + M + T (3) + message (5) -> 42 tokens left for the diff
    assert out == "H" + "m" * 5 + "M" + "d" * 42 + "T"
    assert len(out) == 50


def test_build_prompt_fills_window_exactly_when_both_inputs_are_long():
    out = build(backend_with(FakePipe()), message="m" * 500, diff="d" * 5000,
                message_max_tokens=64, context_window_tokens=1024)
    assert len(out) == 1024
    assert out.count("m") == 64


@pytest.mark.parametrize("message,diff", [(None, None), ("", "")])
def test_build_prompt_accepts_empty_message_and_diff(message, diff):
    assert build(backend_with(FakePipe()), message=message, diff=diff) == "HMT"


def test_build_prompt_encodes_without_special_tokens():
    pipe = FakePipe()
    build(backend_with(pipe), message="msg", diff="diff")
    assert pipe.tokenizer.encode_kwargs
    assert all(kw["add_special_tokens"] is False for kw in pipe.tokenizer.encode_kwargs)


def test_build_prompt_keeps_one_diff_token_when_overhead_exceeds_window():
    # Pins current behavior: remaining = max(1, ...), so the prompt can exceed
    # the window when the fixed overhead alone does. Not reachable in the study:
    # data/rq4_2/artifact_compatible/flaccs/assignments.csv only keeps fits_1024 commits.
    out = build(backend_with(FakePipe()), message="m" * 20, diff="d" * 20,
                message_max_tokens=20, context_window_tokens=10)
    assert out == "H" + "m" * 20 + "MdT"
    assert len(out) > 10


@pytest.mark.parametrize("template", [
    f"H{DIFF_PLACEHOLDER}T",
    f"H{MESSAGE_PLACEHOLDER}T",
    "no placeholders",
])
def test_build_prompt_rejects_template_without_both_placeholders(template):
    with pytest.raises(RuntimeError, match="missing"):
        build(backend_with(FakePipe()), message="m", diff="d", template=template)


def _native_template() -> str:
    from src.prompts.loader import PromptRegistry
    messages = PromptRegistry(ROOT / "prompts/manifest.yaml").load_messages("p_flaccs_native_10class")
    assert len(messages) == 1 and messages[0]["role"] == "user"
    return messages[0]["content"]


def test_native_flaccs_template_has_each_placeholder_once_in_order():
    template = _native_template()
    assert template.count(MESSAGE_PLACEHOLDER) == 1
    assert template.count(DIFF_PLACEHOLDER) == 1
    assert template.index(MESSAGE_PLACEHOLDER) < template.index(DIFF_PLACEHOLDER)


def test_build_prompt_with_native_template_fits_window_and_keeps_wrapper():
    template = _native_template()
    # With one token per character the wrapper text alone exceeds 1,024, so size
    # the window from the template: fixed overhead + 64 message + 100 diff tokens.
    overhead = len(template) - len(MESSAGE_PLACEHOLDER) - len(DIFF_PLACEHOLDER)
    window = overhead + 64 + 100
    out = build(backend_with(FakePipe()), message="m" * 500, diff="d" * 5000,
                message_max_tokens=64, context_window_tokens=window, template=template)
    assert len(out) == window
    assert out.count("d" * 100) == 1 and "d" * 101 not in out
    assert out.startswith("<s>[INST] <<SYS>>")
    assert out.rstrip().endswith("[/INST]")
    assert out.count("m" * 64) == 1 and "m" * 65 not in out


def test_build_prompt_with_real_flaccs_tokenizer_fits_window():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            FlaccsBackend().config.model, local_files_only=True)
    except Exception:
        pytest.skip("FLACCS tokenizer not in the local Hugging Face cache")
    backend = backend_with(SimpleNamespace(tokenizer=tokenizer))
    out = build(backend, message="fix the parser " * 100, diff="+ x = 1\n" * 2000,
                message_max_tokens=64, context_window_tokens=1024, template=_native_template())
    assert len(tokenizer.encode(out, add_special_tokens=False)) <= 1024


# infer() -------------------------------------------------------------------------

@pytest.fixture
def fake_torch(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(OutOfMemoryError=FakeOutOfMemoryError,
                                 is_available=lambda: False, empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    seeds = []
    monkeypatch.setattr(flaccs.FlaccsBackend, "set_seed", staticmethod(seeds.append))
    return seeds


def test_infer_uses_native_greedy_decoding_and_parses_response(fake_torch):
    pipe = FakePipe(response="fix")
    result = backend_with(pipe).infer(prompt="PROMPT", seed=20, task_labels=FLACCS_LABELS)

    assert fake_torch == [20]
    (prompt, kwargs), = pipe.calls
    assert prompt == "PROMPT"
    assert kwargs == {"max_new_tokens": 10, "do_sample": False,
                      "pad_token_id": pipe.tokenizer.eos_token_id, "return_full_text": False}
    assert "temperature" not in kwargs and "top_p" not in kwargs
    assert result.raw_response == "fix"
    assert result.technical_status == "ok" and result.technical_detail == ""
    assert result.parsed.prediction == "fix"
    assert result.processing_time_seconds >= 0


def test_infer_rejects_label_outside_task_labels(fake_torch):
    result = backend_with(FakePipe(response="others")).infer(
        prompt="p", seed=10, task_labels=FLACCS_LABELS)
    assert result.technical_status == "ok"
    assert result.parsed.prediction == INVALID_OUTPUT


def test_infer_turns_generic_exception_into_failure(fake_torch):
    pipe = FakePipe(error=ValueError("boom"))
    result = backend_with(pipe).infer(prompt="p", seed=10, task_labels=FLACCS_LABELS)
    assert result.raw_response == "<failure>"
    assert result.technical_status == "failure"
    assert result.technical_detail == "ValueError: boom"
    assert result.parsed.prediction == INVALID_OUTPUT
    assert result.parsed.technical_status == "failure"


def test_infer_turns_cuda_oom_into_cuda_oom_status(fake_torch):
    pipe = FakePipe(error=FakeOutOfMemoryError("out of memory"))
    result = backend_with(pipe).infer(prompt="p", seed=10, task_labels=FLACCS_LABELS)
    assert result.raw_response == "<failure>"
    assert result.technical_status == "cuda_oom"
    assert result.technical_detail == "out of memory"
    assert result.parsed.prediction == INVALID_OUTPUT
