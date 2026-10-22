from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping, Any

import yaml

PLACEHOLDER_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


class PromptRegistry:
    """Resolve stable prompt IDs through prompts/manifest.yaml."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        payload = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict) or not isinstance(payload.get("prompts"), dict):
            raise RuntimeError(f"{self.manifest_path}: expected top-level 'prompts' mapping")
        self._prompts = payload["prompts"]
        self.prompts_root = self.manifest_path.parent
        self._messages: dict[str, list[dict]] = {}
        self._source_hashes: dict[str, str] = {}

    def ids(self) -> tuple[str, ...]:
        return tuple(self._prompts)

    def entry(self, prompt_id: str) -> dict:
        try:
            entry = self._prompts[prompt_id]
        except KeyError as exc:
            raise KeyError(f"Unknown prompt_id {prompt_id!r}") from exc
        if not isinstance(entry, dict) or "file" not in entry:
            raise RuntimeError(f"Manifest entry {prompt_id!r} has no file")
        return dict(entry)

    def path(self, prompt_id: str) -> Path:
        path = (self.prompts_root / self.entry(prompt_id)["file"]).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Prompt {prompt_id!r}: {path}")
        return path

    def sha256(self, prompt_id: str) -> str:
        self.load_messages(prompt_id)
        return self._source_hashes[prompt_id]

    def load_messages(self, prompt_id: str) -> list[dict]:
        if prompt_id in self._messages:
            return [dict(message) for message in self._messages[prompt_id]]
        path = self.path(prompt_id)
        source = path.read_bytes()
        payload = yaml.safe_load(source) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(source)
        if not isinstance(payload, list):
            raise RuntimeError(f"{path}: prompt must be a list of message objects")
        active = []
        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                raise RuntimeError(f"{path}: message #{idx} is not a mapping")
            if item.get("activate", False):
                role = str(item.get("role", "")).strip()
                if role not in {"system", "user", "assistant"}:
                    raise RuntimeError(f"{path}: invalid role {role!r}")
                active.append({"role": role, "content": str(item.get("content", ""))})
        if not active:
            raise RuntimeError(f"{path}: no active prompt messages")
        # Freeze the exact bytes used for this registry/run. Re-reading the YAML
        # per commit could silently change a prompt under the same fingerprint.
        self._messages[prompt_id] = active
        self._source_hashes[prompt_id] = hashlib.sha256(source).hexdigest()
        return [dict(message) for message in active]

    @staticmethod
    def _render(template: str, variables: Mapping[str, Any]) -> str:
        unresolved = sorted(set(PLACEHOLDER_RE.findall(template)) - set(variables))
        if unresolved:
            raise RuntimeError(f"Unresolved prompt placeholders: {unresolved}")
        # Substitute template placeholders once; commit text is literal data.
        return PLACEHOLDER_RE.sub(lambda match: "" if variables[match[1]] is None else str(variables[match[1]]), template)

    def render_messages(self, prompt_id: str, **variables: Any) -> list[dict]:
        return [
            {"role": m["role"], "content": self._render(m["content"], variables)}
            for m in self.load_messages(prompt_id)
        ]

    def render_text(self, prompt_id: str, **variables: Any) -> str:
        """Render a text-generation prompt such as native FLACCS/HQCM.

        The prompt file must have exactly one active user message.
        """
        messages = self.render_messages(prompt_id, **variables)
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise RuntimeError(
                f"Prompt {prompt_id!r} is not a single-user text prompt; use render_messages()"
            )
        return messages[0]["content"]
