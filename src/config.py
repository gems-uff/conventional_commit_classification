from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected a YAML mapping")
    return payload


def load_inference_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    if cfg.get("version") != 1:
        raise RuntimeError("Unsupported inference config version")
    if cfg.get("runtime", {}).get("provider") != "ollama":
        raise RuntimeError("Shared inference config must use runtime.provider=ollama")
    required = {"token_estimation", "input", "context", "generation", "execution", "parsing", "output"}
    missing = sorted(required - set(cfg))
    if missing:
        raise RuntimeError(f"Inference config missing sections: {missing}")
    return cfg


def load_evaluation_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    if cfg.get("version") != 1:
        raise RuntimeError("Unsupported evaluation config version")
    labels = tuple(cfg.get("taxonomy", {}).get("labels", []))
    expected = ("fix", "feat", "build", "ci", "docs", "style", "refactor", "perf", "test", "chore", "others")
    if labels != expected:
        raise RuntimeError(f"Unexpected evaluation taxonomy: {labels}")
    return cfg


def load_models(path: str | Path) -> dict[str, dict[str, Any]]:
    cfg = load_yaml(path)
    if cfg.get("version") != 1 or cfg.get("runtime", {}).get("provider") != "ollama":
        raise RuntimeError("Unexpected models registry schema")
    models = cfg.get("models")
    if not isinstance(models, dict) or not models:
        raise RuntimeError("models.yaml contains no models")
    return models


def resolve_ollama_model(path: str | Path, model_id: str) -> dict[str, Any]:
    models = load_models(path)
    if model_id not in models:
        raise KeyError(f"Unknown model_id {model_id!r}")
    spec = dict(models[model_id])
    if "ollama_tag" not in spec:
        raise RuntimeError(f"Model {model_id!r} has no ollama_tag")
    return spec


def load_experiment(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    for key in ("version", "id", "research_question"):
        if key not in cfg:
            raise RuntimeError(f"Experiment config missing {key!r}")
    if cfg["version"] != 1:
        raise RuntimeError("Unsupported experiment config version")
    return cfg
