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
    if cfg["parsing"].get("parser_version") != "4.1.1":
        raise RuntimeError("Inference requires canonical parser v4.1.1")
    if cfg["token_estimation"].get("tokenizer") != "cl100k_base":
        raise RuntimeError("Unsupported token estimator")
    if cfg["input"]["truncation"].get("strategy") != "prefix" or not cfg["input"]["truncation"].get("enabled"):
        raise RuntimeError("Only enabled prefix truncation is supported")
    if cfg["execution"]["timeout"].get("retry_same_request") is not False:
        raise RuntimeError("Timed-out requests must not be retried")
    for section, key in (("generation", "num_predict"), ("execution", "max_attempts"),
                         ("execution", "batch_size"), ("execution", "max_concurrent_requests"),
                         ("input", "max_content_tokens")):
        value = cfg[section].get(key)
        if type(value) is not int or value <= 0:
            raise RuntimeError(f"{section}.{key} must be a positive integer")
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


def load_training_config(path: str | Path) -> dict[str, Any]:
    """Load a config/training/*.yaml for a trainable RQ4 baseline (SVM/BiLSTM/CodeBERT).
    """
    cfg = load_yaml(path)
    if cfg.get("version") != 1:
        raise RuntimeError("Unsupported training config version")
    if "model" not in cfg:
        raise RuntimeError("Training config missing 'model'")
    selection = cfg.get("model_selection")
    if not isinstance(selection, dict):
        raise RuntimeError("Training config missing 'model_selection'")
    if selection.get("method") not in {"grid_search", "early_stopping"}:
        raise RuntimeError(f"Unsupported model_selection.method: {selection.get('method')!r}")
    if selection.get("method") == "grid_search" and not selection.get("grid"):
        raise RuntimeError("model_selection.method=grid_search requires a non-empty 'grid'")
    if selection.get("method") == "early_stopping":
        for key in ("max_epochs", "patience"):
            value = selection.get(key)
            if type(value) is not int or value <= 0:
                raise RuntimeError(f"model_selection.{key} must be a positive integer")
    validation = selection.get("validation", {})
    fraction = validation.get("fraction")
    if type(fraction) not in (int, float) or not 0 < fraction < 1:
        raise RuntimeError("model_selection.validation.fraction must be in (0, 1)")
    return cfg
