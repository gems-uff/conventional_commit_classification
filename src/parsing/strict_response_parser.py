#!/usr/bin/env python3
"""Precision-first deterministic parser for LLM commit classifications.

The parser receives only ``raw_response``.  It recognizes a deliberately
restricted grammar of explicit classification declarations and prioritizes
precision over extraction coverage.  It never infers a label from explanatory
prose, lists, code changes, or commit semantics.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

PARSER_VERSION = "4.1.1"
INVALID_OUTPUT = "<invalid_output>"

LABELS: tuple[str, ...] = (
    "fix", "feat", "build", "ci", "docs", "style",
    "refactor", "perf", "test", "chore", "others",
)
LABEL_SET = set(LABELS)

# Closed, auditable whitelist.  Aliases are applied only after an explicit
# classification construction has been recognized.  Bare responses must use a
# canonical label.
LABEL_ALIASES: dict[str, set[str]] = {
    "feat": {"feature", "features"},
    "fix": {"bugfix", "bug-fix", "bug fix", "fixing"},
    "docs": {"doc", "documentation"},
    "style": {"formatting"},
    "refactor": {"refactoring"},
    "perf": {"performance"},
    "test": {"tests", "testing"},
    "build": {"build-system", "build system"},
    "ci": {"cicd", "ci/cd", "ci-cd", "ci cd"},
    "chore": {"chores"},
    "others": {"other", "none of the above", "none-of-the-above"},
}
ALIAS_TO_LABEL = {
    alias.casefold(): label
    for label, aliases in LABEL_ALIASES.items()
    for alias in aliases
}

# Only these field names are treated as classification fields.
FIELD_ALIASES: dict[str, str] = {
    "classification": "classification",
    "classification_type": "classification_type",
    "commit_classification": "commit_classification",
    "commit_type": "commit_type",
    "type": "type",
}
STRONG_FIELDS = {
    "classification", "classification_type", "commit_classification", "commit_type"
}
SUPPORT_FIELDS = {
    "breaking_change", "breakingchange", "is_breaking_change",
    "reason", "explanation", "confidence", "classification_reason",
}
CONTENT_FIELDS = {
    "content", "children", "properties", "items", "enum", "description",
    "title", "text", "src", "alt", "oneof", "anyof", "allof", "$schema",
    "scripts", "dependencies", "devdependencies", "module", "exports",
}

TECHNICAL_EXACT = {
    "<failure>", "failure", "timeout", "<timeout>",
    "timeout_error", "<timeout_error>", "parse_exception",
}

FINAL_MARKERS = {
    "final answer", "final response", "final classification", "answer final",
}
FINAL_INLINE_KEYS = {
    "final answer", "final response", "final classification",
}

PROSE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("the final classification is", "final_classification_prose"),
    ("the commit classification is", "classification_prose"),
    ("the commit type is", "classification_prose"),
    ("the classification is", "classification_prose"),
    ("the commit is classified as", "classified_as_prose"),
    ("this commit is classified as", "classified_as_prose"),
    ("i classify this commit as", "classify_as_prose"),
    ("i classify it as", "classify_as_prose"),
)

EXTERNAL_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "it", "its",
    "since", "because", "determined", "likely", "probably", "possibly",
    "maybe", "perhaps", "either", "for", "from", "by", "with", "without",
    "according", "based", "which", "where", "when", "whether", "what",
    "how", "why", "therefore", "thus", "hence", "main", "just", "simply",
    "should", "would", "could", "might", "may", "can", "will", "is", "are",
    "was", "were", "be", "being", "been", "to", "of", "as", "and", "or",
    "yes", "no", "true", "false", "none", "unknown", "unclear", "uncertain",
}
AMBIGUITY_WORDS = {
    "or", "either", "versus", "vs", "maybe", "perhaps", "possibly",
    "likely", "probably", "could", "might", "may", "not", "uncertain", "unclear",
}
# These are values from auxiliary output fields or specification markers, not
# commit-intent labels.  They must never be converted into an external label.
MALFORMED_LABEL_VALUES = {
    "breaking change", "breaking changes", "true", "false", "yes", "no", "none",
    "classification", "commit classification", "commit_classification",
    "classification label", "classification_label", "commit type", "commit_type",
    "type", "label", "class",
}
LABEL_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.+/-]*$")
_CANONICAL_ALT = "|".join(re.escape(label) for label in LABELS)
CONVENTIONAL_HEADER_RE = re.compile(
    rf"^(?P<label>{_CANONICAL_ALT})(?P<scope>\([^()\r\n]+\))?(?P<bang>!)?:\s+(?P<subject>\S.*)$",
    re.IGNORECASE,
)
LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+>]\s+|\d+[.)]\s+)")
HEADING_PREFIX_RE = re.compile(r"^\s*#{1,6}\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Recover only clearly delimited auxiliary fields appended to an otherwise
# explicit classification declaration, e.g.
# ``classification: chore breaking_change: false``.  The marker must have its
# own separator; arbitrary trailing prose is never truncated.
INLINE_SUPPORT_FIELD_RE = re.compile(
    r"\s+(?=(?:breaking(?:[ _-]+change)|is[ _-]+breaking[ _-]+change|"
    r"reason|explanation|confidence|classification[ _-]+reason)"
    r"\s*(?::|=|\bis\b))",
    re.IGNORECASE,
)

# Some models concatenate a final field to a preceding sentence without a
# newline (``...format.classification: chore``).  Only strong classification
# fields preceded by sentence punctuation are recoverable through this rule.
EMBEDDED_STRONG_FIELD_RE = re.compile(
    r"(?:^|[.!?;])\s*(?P<field>classification(?:[ _]type)?|"
    r"commit[ _](?:classification|type))\s*"
    r"(?P<sep>:|=|\bis\b)\s*(?P<value>.+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Region:
    name: str
    text: str
    index: int


@dataclass(frozen=True)
class LineInfo:
    region: str
    region_index: int
    line_index: int
    raw: str
    in_code: bool
    in_diff: bool
    final_context: bool
    first_substantive: bool
    near_terminal: bool
    support_nearby: bool


@dataclass(frozen=True)
class Candidate:
    label: str
    raw_label: str
    semantic_kind: str  # canonical | alias | external
    normalization_rule: str
    source: str
    field: str
    region: str
    region_index: int
    line_index: int
    evidence: str
    final_context: bool
    canonical: bool
    extraction_form: str
    reasoning_safe: bool


@dataclass(frozen=True)
class ParseResult:
    label: str
    status: str
    raw_label: str | None
    normalized_label: str | None
    normalization_rule: str | None
    source: str | None
    field: str | None
    region: str
    evidence: str
    canonical: bool
    extraction_form: str | None
    note: str
    candidate_count: int
    candidates_json: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")


def _normalize_key(value: str) -> str:
    text = value.strip().casefold()
    text = text.replace("**", "").replace("__", "")
    text = text.strip("`\"' ")
    text = re.sub(r"[^a-z0-9$]+", "_", text)
    return text.strip("_")


def _strip_outer_markdown(value: str) -> str:
    text = value.strip()
    text = text.replace("**", "").replace("__", "")
    return text.strip()


def _strip_matching_outer_quote(value: str) -> str:
    """Remove a quote wrapping an entire answer line, but not a JSON key."""
    text = value.strip()
    if len(text) < 2 or text[0] not in {'"', "'", "`"} or text[-1] != text[0]:
        return text
    # Preserve JSON-like lines such as ``"classification": "fix"``.
    if text[0] in {'"', "'"} and re.match(r'^(["\'])[^"\']+\1\s*[:=]', text):
        return text
    return text[1:-1].strip()


def _strip_line_prefix_for_field(value: str) -> tuple[str, bool, bool]:
    """Strip presentation-only prefixes and report list/heading origin."""
    text = value.rstrip()
    list_like = bool(LIST_PREFIX_RE.match(text))
    heading_like = bool(HEADING_PREFIX_RE.match(text))
    text = LIST_PREFIX_RE.sub("", text, count=1)
    text = HEADING_PREFIX_RE.sub("", text, count=1)
    text = _strip_outer_markdown(text)
    text = _strip_matching_outer_quote(text)
    return text.strip(), list_like, heading_like


def _technical_status(text: str) -> str | None:
    cleaned = text.strip().casefold()
    if not cleaned:
        return "invalid_empty_response"
    if cleaned in TECHNICAL_EXACT or cleaned.startswith("<timeout"):
        return "invalid_technical_failure"
    return None


def split_think_regions(text: str) -> list[Region]:
    lower = text.casefold()
    regions: list[Region] = []
    pos = 0
    inside = False
    seen_closed = False
    index = 0
    while pos < len(text):
        if not inside:
            start = lower.find("<think>", pos)
            if start < 0:
                name = "post_think" if seen_closed else "outside_think"
                if text[pos:]:
                    regions.append(Region(name, text[pos:], index))
                break
            if start > pos:
                name = "post_think" if seen_closed else "outside_think"
                regions.append(Region(name, text[pos:start], index))
                index += 1
            pos = start + len("<think>")
            inside = True
        else:
            end = lower.find("</think>", pos)
            if end < 0:
                regions.append(Region("unclosed_think", text[pos:], index))
                break
            regions.append(Region("closed_think", text[pos:end], index))
            index += 1
            pos = end + len("</think>")
            inside = False
            seen_closed = True
    if not regions and text:
        regions.append(Region("outside_think", text, 0))
    return regions


def _is_final_marker(line: str) -> bool:
    clean, _, _ = _strip_line_prefix_for_field(line)
    clean = clean.rstrip(":=").strip().casefold()
    return clean in FINAL_MARKERS


def _field_parts(line: str) -> tuple[str, str, str, bool, bool] | None:
    """Return (key, separator, value, list_like, heading_like).

    Only explicit separators ``:``, ``=``, and the word ``is`` are accepted.
    Whitespace alone is never a separator.
    """
    clean, list_like, heading_like = _strip_line_prefix_for_field(line)
    if not clean:
        return None
    # Complete JSON objects are handled only by the answer-shaped JSON parser;
    # do not reinterpret their punctuation as a line-oriented field.
    if clean.startswith("{") and clean.endswith("}"):
        return None

    # Colon/equal separators are checked before the word connector.
    positions: list[tuple[int, str, int]] = []
    for sep in (":", "="):
        pos = clean.find(sep)
        if pos > 0:
            positions.append((pos, sep, 1))
    is_match = re.search(r"\s+is\s+", clean, flags=re.IGNORECASE)
    if is_match:
        positions.append((is_match.start(), "is", is_match.end() - is_match.start()))
    if not positions:
        return None

    pos, sep, width = sorted(positions, key=lambda item: item[0])[0]
    key_text = clean[:pos].strip()
    value = clean[pos + width:].strip()
    key = FIELD_ALIASES.get(_normalize_key(key_text))
    if key is None or not value:
        return None
    return key, sep, value, list_like, heading_like


def _support_key(line: str) -> str | None:
    clean, _, _ = _strip_line_prefix_for_field(line)
    for sep in (":", "="):
        if sep in clean:
            left = clean.split(sep, 1)[0]
            key = _normalize_key(left)
            return key if key in SUPPORT_FIELDS else None
    match = re.search(r"\s+is\s+", clean, flags=re.IGNORECASE)
    if match:
        key = _normalize_key(clean[:match.start()])
        return key if key in SUPPORT_FIELDS else None
    return None


def _looks_like_answer_start(line: str) -> bool:
    if _is_final_marker(line):
        return True
    parts = _field_parts(line)
    if parts is not None:
        return True
    clean, _, _ = _strip_line_prefix_for_field(line)
    lowered = clean.casefold()
    return any(lowered.startswith(prefix + " ") for prefix, _ in PROSE_PREFIXES)


def _analyze_region(region: Region) -> list[LineInfo]:
    raw_lines = region.text.splitlines()
    substantive = [i for i, line in enumerate(raw_lines) if line.strip()]
    first_index = substantive[0] if substantive else -1
    last_index = substantive[-1] if substantive else -1
    support_indices = {i for i, line in enumerate(raw_lines) if _support_key(line)}

    infos: list[LineInfo] = []
    in_code = False
    fence: str | None = None
    in_diff = False
    blank_seen = False
    final_pending = False

    for i, raw in enumerate(raw_lines):
        stripped = raw.strip()
        fence_match = re.match(r"^\s*(```+|~~~+)", raw)
        if fence_match:
            marker = fence_match.group(1)[0]
            if not in_code:
                in_code = True
                fence = marker
            elif fence == marker:
                in_code = False
                fence = None
            # Fence delimiter itself is never content.
            infos.append(LineInfo(
                region.name, region.index, i, raw, True, in_diff, final_pending,
                i == first_index, last_index >= 0 and i >= max(first_index, last_index - 4),
                any(abs(i - j) <= 4 for j in support_indices),
            ))
            continue

        if not in_code:
            if (
                raw.startswith("diff --git ")
                or raw.startswith("@@ ")
                or raw.startswith("--- a/")
                or raw.startswith("+++ b/")
            ):
                in_diff = True
                blank_seen = False
            unprefixed = bool(raw) and raw[0] not in {" ", "+", "-"}
            if in_diff and unprefixed and blank_seen and _looks_like_answer_start(raw):
                in_diff = False

        marker = (not in_code and not in_diff and _is_final_marker(raw))
        line_final = final_pending or marker
        infos.append(LineInfo(
            region.name, region.index, i, raw, in_code, in_diff, line_final,
            i == first_index, last_index >= 0 and i >= max(first_index, last_index - 4),
            any(abs(i - j) <= 4 for j in support_indices),
        ))

        if marker:
            final_pending = True
        elif stripped:
            final_pending = False

        if in_diff:
            blank_seen = not stripped
        else:
            blank_seen = False

    return infos


def _unwrap_declared_scalar(value: str) -> tuple[str, bool]:
    """Remove only wrappers enclosing the complete declared scalar.

    Wrappers may be nested in either order, for example ``**`chore`**`` or
    ```**chore**```.  The routine is deliberately local: it never removes
    markup from the middle of a multiword phrase and therefore cannot turn
    explanatory prose into a label.
    """
    text = value.strip()
    fully_quoted = False
    wrappers = (
        ("***", "***", False),
        ("___", "___", False),
        ("**", "**", False),
        ("__", "__", False),
        ("`", "`", True),
        ('"', '"', True),
        ("'", "'", True),
        ("*", "*", False),
        ("_", "_", False),
    )

    for _ in range(12):
        previous = text
        text = text.strip()
        for left, right, quoted in wrappers:
            if len(text) <= len(left) + len(right):
                continue
            if text.startswith(left) and text.endswith(right):
                text = text[len(left):-len(right)].strip()
                fully_quoted = fully_quoted or quoted
                break
        if text == previous:
            break
    return text, fully_quoted


def _clean_declared_scalar(
    value: str,
    *,
    strip_article: bool,
) -> tuple[str, bool, list[str]]:
    """Normalize only local wrappers and explicit output syntax.

    The function may remove an appended auxiliary field only when that field
    has an explicit key and separator.  It also removes a single trailing
    Conventional-Commits breaking marker (``!``) from the declared label.
    """
    text = _strip_outer_markdown(value).strip()
    transformations: list[str] = []

    support_match = INLINE_SUPPORT_FIELD_RE.search(text)
    if support_match is not None:
        prefix = text[:support_match.start()].strip()
        if prefix:
            text = prefix
            transformations.append("strip_inline_support_field")

    text = text.rstrip().rstrip(",")
    # Sentence punctuation may trail a scalar declaration.
    text = text.rstrip().rstrip(".;")
    text = re.sub(r"\s+", " ", text).strip()

    # In constructions such as ``is a **`chore`**``, the article precedes the
    # wrappers.  Remove it before unwrapping the scalar itself.
    if strip_article:
        lowered = text.casefold()
        if lowered.startswith("a "):
            text = text[2:].strip()
            transformations.append("strip_article")
        elif lowered.startswith("an "):
            text = text[3:].strip()
            transformations.append("strip_article")

    text, fully_quoted = _unwrap_declared_scalar(text)
    text = re.sub(r"\s+", " ", text).strip()

    # ``feat!`` and ``refactor!`` are unambiguous Conventional Commit forms
    # when they are already the value of an explicit classification field.
    if text.endswith("!") and text[:-1].strip():
        text = text[:-1].strip()
        transformations.append("strip_breaking_bang")
        text, nested_quoted = _unwrap_declared_scalar(text)
        fully_quoted = fully_quoted or nested_quoted

    return text, fully_quoted, transformations


def _parse_declared_value(
    value: str,
    *,
    allow_alias: bool,
    allow_external: bool,
    strip_article: bool = True,
) -> tuple[str, str, str, str] | None:
    """Return (normalized label, raw label, semantic kind, rule)."""
    cleaned, fully_quoted, transformations = _clean_declared_scalar(
        value, strip_article=strip_article
    )
    if not cleaned:
        return None

    # Angle brackets are often prompt placeholders (for example,
    # ``<commit_classification>``).  Treat them as a wrapper only when the
    # enclosed value is itself a canonical label or a whitelisted alias.
    # Angle-wrapped unknown values are never external labels.
    angle_wrapped = False
    if len(cleaned) > 2 and cleaned.startswith("<") and cleaned.endswith(">"):
        angle_wrapped = True
        inner = cleaned[1:-1].strip()
        inner, nested_quoted = _unwrap_declared_scalar(inner)
        fully_quoted = fully_quoted or nested_quoted
        cleaned = re.sub(r"\s+", " ", inner).strip()
        transformations.append("strip_angle_wrapper")

    lowered = cleaned.casefold()
    if lowered in MALFORMED_LABEL_VALUES:
        return None
    words = set(re.findall(r"[A-Za-z]+", lowered))
    if words & AMBIGUITY_WORDS:
        return None

    modifier = "+".join(transformations)

    if lowered in LABEL_SET:
        rule = "identity" if not modifier else modifier
        return lowered, cleaned, "canonical", rule

    if allow_alias and lowered in ALIAS_TO_LABEL:
        label = ALIAS_TO_LABEL[lowered]
        rule = f"{lowered}_to_{label}"
        if modifier:
            rule += "+" + modifier
        return label, cleaned, "alias", rule

    if not allow_external or angle_wrapped:
        return None

    # External labels are deliberately narrow: one label-like token, or an
    # explicitly quoted phrase of at most three label-like tokens.
    tokens = cleaned.split()
    if len(tokens) > 1 and not fully_quoted:
        return None
    if not (1 <= len(tokens) <= 3):
        return None
    normalized_tokens: list[str] = []
    for token in tokens:
        token = token.strip("`*_\"'.,()[]{}<>")
        if not LABEL_TOKEN_RE.fullmatch(token):
            return None
        if token.casefold() in EXTERNAL_STOPWORDS:
            return None
        normalized_tokens.append(token)
    raw_label = " ".join(normalized_tokens)
    rule = "external_explicit"
    if modifier:
        rule += "+" + modifier
    return "others", raw_label, "external", rule


def _reasoning_safe(info: LineInfo, source: str) -> bool:
    if info.region == "closed_think":
        return True
    if info.region != "unclosed_think":
        return True
    return info.final_context or info.near_terminal or source == "final_answer"


def _candidate(
    parsed: tuple[str, str, str, str],
    *,
    source: str,
    field: str,
    info: LineInfo,
    evidence: str,
    extraction_form: str,
) -> Candidate | None:
    reasoning_safe = _reasoning_safe(info, source)
    if (
        not reasoning_safe
        and info.region == "unclosed_think"
        and source in {"field_line", "embedded_field_line"}
        and field in STRONG_FIELDS
        and info.support_nearby
    ):
        reasoning_safe = True
    if not reasoning_safe:
        return None
    label, raw_label, kind, rule = parsed
    return Candidate(
        label=label,
        raw_label=raw_label,
        semantic_kind=kind,
        normalization_rule=rule,
        source=source,
        field=field,
        region=info.region,
        region_index=info.region_index,
        line_index=info.line_index,
        evidence=evidence.strip()[:2000],
        final_context=info.final_context,
        canonical=(kind == "canonical"),
        extraction_form=extraction_form,
        reasoning_safe=True,
    )


def _eligible_explicit_line(info: LineInfo, field: str, list_like: bool, heading_like: bool) -> bool:
    if info.in_diff:
        return False
    if info.in_code:
        return False
    if info.region == "closed_think":
        return True
    if info.region == "unclosed_think":
        # A complete answer-shaped block may be emitted before an unclosed
        # reasoning trace ends. Recover a strong classification field when a
        # supporting output field appears nearby; otherwise retain the
        # terminal/final-context safeguard.
        if field in STRONG_FIELDS and info.support_nearby:
            return True
        return _reasoning_safe(info, "field_line")
    if heading_like and not info.final_context:
        return False
    if list_like and not (info.first_substantive or info.final_context or info.support_nearby):
        return False
    if field == "type":
        return info.first_substantive or info.final_context or info.near_terminal or info.support_nearby
    return info.first_substantive or info.final_context or info.near_terminal or info.support_nearby


def _field_candidate(info: LineInfo) -> Candidate | None:
    parts = _field_parts(info.raw)
    if parts is None:
        return None
    field, separator, value, list_like, heading_like = parts
    if not _eligible_explicit_line(info, field, list_like, heading_like):
        return None
    parsed = _parse_declared_value(
        value,
        allow_alias=True,
        allow_external=True,
        strip_article=(separator == "is"),
    )
    if parsed is None:
        return None
    return _candidate(
        parsed,
        source="field_line",
        field=field,
        info=info,
        evidence=info.raw,
        extraction_form=f"field_{separator}",
    )


def _embedded_field_candidate(info: LineInfo) -> Candidate | None:
    """Recover a strong field concatenated after sentence punctuation.

    This deliberately excludes the generic ``type`` field and requires the
    same positional and region safeguards used by ordinary explicit fields.
    """
    clean, list_like, heading_like = _strip_line_prefix_for_field(info.raw)
    if info.in_code or info.in_diff or list_like or heading_like:
        return None
    match = EMBEDDED_STRONG_FIELD_RE.search(clean)
    if match is None or match.start("field") == 0:
        return None
    field = FIELD_ALIASES.get(_normalize_key(match.group("field")))
    if field not in STRONG_FIELDS:
        return None
    if not _eligible_explicit_line(info, field, False, False):
        return None
    separator = match.group("sep").casefold()
    parsed = _parse_declared_value(
        match.group("value"),
        allow_alias=True,
        allow_external=True,
        strip_article=(separator == "is"),
    )
    if parsed is None:
        return None
    return _candidate(
        parsed,
        source="embedded_field_line",
        field=field,
        info=info,
        evidence=info.raw,
        extraction_form=f"embedded_field_{separator}",
    )


def _final_inline_candidate(info: LineInfo) -> Candidate | None:
    clean, list_like, heading_like = _strip_line_prefix_for_field(info.raw)
    if info.in_code or info.in_diff or heading_like:
        return None
    for sep in (":", "="):
        if sep not in clean:
            continue
        left, right = clean.split(sep, 1)
        if left.strip().casefold() not in FINAL_INLINE_KEYS:
            continue
        parsed = _parse_declared_value(right, allow_alias=True, allow_external=True)
        if parsed is None:
            return None
        return _candidate(
            parsed,
            source="final_answer",
            field="final_answer",
            info=info,
            evidence=info.raw,
            extraction_form=f"final_answer_{sep}",
        )

    # A marker on the preceding substantive line (for example, ``Final answer:``)
    # turns the current scalar line into an explicit final-answer value.
    if info.final_context and clean and not list_like:
        parsed = _parse_declared_value(clean, allow_alias=True, allow_external=True)
        if parsed is not None:
            return _candidate(
                parsed,
                source="final_answer",
                field="final_answer",
                info=info,
                evidence=info.raw,
                extraction_form="final_answer_next_line",
            )
    return None


def _prose_candidates(info: LineInfo) -> list[Candidate]:
    if info.in_code or info.in_diff:
        return []
    if info.region not in {"closed_think", "unclosed_think"} and not (
        info.first_substantive or info.near_terminal or info.final_context or info.region == "post_think"
    ):
        return []

    clean, list_like, heading_like = _strip_line_prefix_for_field(info.raw)
    if list_like or heading_like:
        return []
    sentences = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(clean) if segment.strip()]
    rows: list[Candidate] = []
    for sentence in sentences:
        lowered = sentence.casefold()
        for prefix, source in PROSE_PREFIXES:
            if not lowered.startswith(prefix + " "):
                continue
            value = sentence[len(prefix):].strip()
            parsed = _parse_declared_value(value, allow_alias=True, allow_external=True, strip_article=True)
            if parsed is None:
                break
            candidate = _candidate(
                parsed,
                source=source,
                field="prose",
                info=info,
                evidence=sentence,
                extraction_form="restricted_declarative_sentence",
            )
            if candidate:
                rows.append(candidate)
            break
    return rows


def _conventional_header_candidate(info: LineInfo) -> Candidate | None:
    """Recognize a canonical Conventional Commit header used as the answer.

    The header must occupy the first substantive line of an outside/post-think
    answer region (or an explicitly marked final-answer line), start with an
    exact taxonomy label, contain the required colon, and have a non-empty
    subject.  Aliases are deliberately not accepted in this form.
    """
    if info.in_code or info.in_diff:
        return None
    if info.region not in {"outside_think", "post_think"}:
        return None
    if not (info.first_substantive or info.final_context):
        return None
    raw = info.raw
    if LIST_PREFIX_RE.match(raw) or HEADING_PREFIX_RE.match(raw):
        return None
    clean = _strip_outer_markdown(raw).strip()
    clean = _strip_matching_outer_quote(clean)
    clean = clean.strip("`*_ ")
    match = CONVENTIONAL_HEADER_RE.fullmatch(clean)
    if not match:
        return None
    label = match.group("label").casefold()
    parsed = (label, match.group("label"), "canonical", "identity")
    return _candidate(
        parsed,
        source="conventional_commit_header",
        field="conventional_commit_type",
        info=info,
        evidence=info.raw,
        extraction_form="canonical_conventional_commit_header",
    )


def _bare_candidate(info: LineInfo) -> Candidate | None:
    """Accept only a canonical label as the first substantive answer token."""
    if info.in_code or info.in_diff or not info.first_substantive:
        return None
    if info.region in {"closed_think", "unclosed_think"}:
        return None
    raw = info.raw
    if LIST_PREFIX_RE.match(raw) or HEADING_PREFIX_RE.match(raw):
        return None
    clean = _strip_outer_markdown(raw).strip()
    clean = clean.lstrip("`*_ ")
    match = re.match(r"^(fix|feat|build|ci|docs|style|refactor|perf|test|chore|others)\b(?P<rest>.*)$", clean, re.IGNORECASE)
    if not match:
        return None
    label = match.group(1).casefold()
    rest = match.group("rest")
    # Colon/scope syntax belongs to the dedicated Conventional Commit header
    # rule and must include a non-empty subject.  Bare labels may otherwise be
    # followed only by sentence punctuation or an explanation dash/semicolon.
    if rest and not re.match(r"^\s*(?:[.;]|[-–—])", rest):
        return None
    parsed = (label, match.group(1), "canonical", "identity")
    return _candidate(
        parsed,
        source="bare_label",
        field="bare",
        info=info,
        evidence=info.raw,
        extraction_form="first_substantive_canonical_label",
    )


def _json_candidates(region: Region, infos: Sequence[LineInfo]) -> list[Candidate]:
    """Parse answer-shaped JSON objects without scanning arbitrary code values."""
    rows: list[Candidate] = []
    text = region.text.strip()
    objects: list[tuple[str, int, bool]] = []

    # Whole-region object.
    if text.startswith("{") and text.endswith("}"):
        objects.append((text, 0, False))

    # Fenced blocks that contain a complete JSON object.
    lines = region.text.splitlines()
    start: int | None = None
    fence_char: str | None = None
    for i, line in enumerate(lines):
        m = re.match(r"^\s*(```+|~~~+)", line)
        if not m:
            continue
        char = m.group(1)[0]
        if start is None:
            start = i + 1
            fence_char = char
        elif fence_char == char:
            body = "\n".join(lines[start:i]).strip()
            if body.startswith("{") and body.endswith("}"):
                objects.append((body, start, True))
            start = None
            fence_char = None

    seen: set[str] = set()
    for raw_json, line_index, fenced in objects:
        if raw_json in seen:
            continue
        seen.add(raw_json)
        try:
            obj = json.loads(raw_json)
        except Exception:
            continue
        if not isinstance(obj, Mapping):
            continue

        candidates_objects: list[Mapping[str, Any]] = [obj]
        for key, child in obj.items():
            if _normalize_key(str(key)) in {"answer", "final_answer", "response", "result"} and isinstance(child, Mapping):
                candidates_objects.append(child)

        for current in candidates_objects:
            keys = {_normalize_key(str(k)) for k in current}
            support = bool(keys & SUPPORT_FIELDS)
            content_like = bool(keys & CONTENT_FIELDS)
            info = infos[min(line_index, len(infos) - 1)] if infos else LineInfo(
                region.name, region.index, 0, raw_json, fenced, False, False, True, True, support
            )
            for raw_key, raw_value in current.items():
                field = FIELD_ALIASES.get(_normalize_key(str(raw_key)))
                if field is None or not isinstance(raw_value, (str, int, float)):
                    continue
                if field == "type" and not (support or info.final_context):
                    continue
                if field != "classification" and content_like and not support:
                    continue
                if fenced and field != "classification" and not support and not info.final_context:
                    continue
                parsed = _parse_declared_value(
                    str(raw_value), allow_alias=True, allow_external=True, strip_article=False
                )
                if parsed is None:
                    continue
                candidate = _candidate(
                    parsed,
                    source="json_field",
                    field=field,
                    info=info,
                    evidence=f"{raw_key}: {raw_value}",
                    extraction_form="json_scalar_field",
                )
                if candidate:
                    rows.append(candidate)
    return rows


def extract_candidates(raw_response: Any) -> tuple[list[Candidate], list[Region]]:
    text = _norm_text(raw_response)
    regions = split_think_regions(text)
    candidates: list[Candidate] = []

    for region in regions:
        infos = _analyze_region(region)
        candidates.extend(_json_candidates(region, infos))
        for info in infos:
            if not info.raw.strip() or _is_final_marker(info.raw):
                continue
            final_inline = _final_inline_candidate(info)
            if final_inline:
                candidates.append(final_inline)
            field = _field_candidate(info)
            if field:
                candidates.append(field)
            else:
                embedded_field = _embedded_field_candidate(info)
                if embedded_field:
                    candidates.append(embedded_field)
            candidates.extend(_prose_candidates(info))
            header = _conventional_header_candidate(info)
            if header:
                candidates.append(header)
            else:
                bare = _bare_candidate(info)
                if bare:
                    candidates.append(bare)

    # Deduplicate declarations recognized through overlapping structural paths.
    unique: dict[tuple[str, str, int, str, str], Candidate] = {}
    for candidate in candidates:
        key = (
            candidate.region, candidate.source, candidate.line_index,
            candidate.raw_label.casefold(), candidate.evidence,
        )
        previous = unique.get(key)
        if previous is None or (candidate.canonical and not previous.canonical):
            unique[key] = candidate
    ordered = sorted(
        unique.values(),
        key=lambda c: (c.region_index, c.line_index, c.final_context, c.source),
    )
    return ordered, regions


def _candidate_json(candidates: Sequence[Candidate]) -> str:
    return json.dumps([asdict(c) for c in candidates], ensure_ascii=False, separators=(",", ":"))


def _invalid(status: str, text: str, note: str, candidates: Sequence[Candidate] = ()) -> ParseResult:
    return ParseResult(
        label=INVALID_OUTPUT,
        status=status,
        raw_label=None,
        normalized_label=None,
        normalization_rule=None,
        source=None,
        field=None,
        region="whole_response",
        evidence=text[:2000],
        canonical=False,
        extraction_form=None,
        note=note,
        candidate_count=len(candidates),
        candidates_json=_candidate_json(candidates),
    )


def _active_candidates(candidates: Sequence[Candidate]) -> tuple[list[Candidate], bool]:
    post = [c for c in candidates if c.region == "post_think"]
    if post:
        finals = [c for c in post if c.final_context or c.source == "final_answer"]
        return finals or post, False

    outside = [c for c in candidates if c.region == "outside_think"]
    if outside:
        finals = [c for c in outside if c.final_context or c.source == "final_answer"]
        return finals or outside, False

    reasoning = [
        c for c in candidates
        if c.region in {"closed_think", "unclosed_think"} and c.reasoning_safe
    ]
    finals = [c for c in reasoning if c.final_context or c.source == "final_answer"]
    return finals or reasoning, bool(reasoning)


def _decision_key(candidate: Candidate) -> tuple[str, str]:
    if candidate.semantic_kind == "external":
        return "external", candidate.raw_label.casefold()
    return "taxonomy", candidate.label


def parse_raw_response(raw_response: Any) -> ParseResult:
    text = _norm_text(raw_response)
    technical = _technical_status(text)
    if technical:
        return _invalid(technical, text, "No semantic classification was produced.")

    candidates, _ = extract_candidates(text)
    active, reasoning_fallback = _active_candidates(candidates)
    if not active:
        return _invalid(
            "invalid_no_decision",
            text,
            "No explicit classification matching the restricted grammar was recoverable.",
            candidates,
        )

    decisions = {_decision_key(candidate) for candidate in active}
    if len(decisions) > 1:
        status = "invalid_multiple_reasoning_decisions" if reasoning_fallback else "invalid_conflicting_decisions"
        return _invalid(
            status,
            " || ".join(candidate.evidence for candidate in active),
            "More than one distinct explicit classification remained in the active answer region.",
            candidates,
        )

    # Prefer structured canonical evidence after agreement is established.
    preference = {
        "json_field": 6,
        "field_line": 5,
        "embedded_field_line": 5,
        "final_answer": 4,
        "final_classification_prose": 3,
        "classification_prose": 3,
        "classified_as_prose": 3,
        "classify_as_prose": 3,
        "conventional_commit_header": 2,
        "bare_label": 1,
    }
    selected = sorted(
        active,
        key=lambda c: (
            c.canonical,
            c.semantic_kind != "external",
            preference.get(c.source, 0),
            c.region_index,
            c.line_index,
        ),
    )[-1]

    if selected.semantic_kind == "canonical":
        if selected.source == "bare_label":
            status = "valid_canonical_bare"
        elif selected.source == "conventional_commit_header":
            status = "valid_canonical_header"
        else:
            status = "valid_canonical_structured"
    elif selected.semantic_kind == "alias":
        status = "valid_alias_structured"
    else:
        status = "explicit_external_label"

    return ParseResult(
        label=selected.label,
        status=status,
        raw_label=selected.raw_label,
        normalized_label=selected.label,
        normalization_rule=selected.normalization_rule,
        source=selected.source,
        field=selected.field,
        region=selected.region,
        evidence=selected.evidence,
        canonical=selected.canonical,
        extraction_form=selected.extraction_form,
        note="",
        candidate_count=len(candidates),
        candidates_json=_candidate_json(candidates),
    )


__all__ = [
    "PARSER_VERSION", "INVALID_OUTPUT", "LABELS", "LABEL_SET", "LABEL_ALIASES",
    "Candidate", "ParseResult", "extract_candidates", "parse_raw_response",
]
