#!/usr/bin/env python3
"""Run the package's Ollama-backed experiments from declarative YAML configs.

This is the single generative runner for RQ1, RQ2, RQ3.1, RQ3.2, and the
GPT-OSS conditions in RQ4. It deliberately does not execute FLACCS, HQCM, or
closed-set trainable baselines; those backends have separate runners.

Key properties
--------------
* canonical datasets and authoritative RQ4 assignment manifests are reused;
* the prompt registry resolves stable prompt IDs;
* all mandatory initial seeds are executed before consensus is evaluated;
* adaptive seeds are executed sequentially only for unresolved commits;
* strict_response_parser v4.1.1 is applied during inference;
* raw responses and parser audit fields are preserved;
* --resume never repeats an already completed commit/seed request;
* published results under results/ are never modified (default output: runs/).

Examples
--------
python scripts/run_ollama_experiment.py \
  --experiment config/experiments/rq3_1.yaml --dry-run

python scripts/run_ollama_experiment.py \
  --experiment config/experiments/rq2.yaml \
  --model gpt_oss_20b --condition temperature_0p6

python scripts/run_ollama_experiment.py \
  --experiment config/experiments/rq4/cross_project.yaml \
  --model gpt_oss_20b --resume
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: Any) -> str:
    text = str(value).strip().casefold()
    text = text.replace("+", "plus")
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    return text.strip("_.-") or "condition"


def _temperature_slug(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return f"temperature_{text}"


def _resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_fingerprint(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return _sha256_text(text)


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"{path}: invalid JSONL at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise RuntimeError(f"{path}: JSONL line {line_no} is not an object")
            rows.append(obj)
    return rows


def _append_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "at", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
        handle.flush()


@dataclass(frozen=True)
class OllamaJob:
    experiment_id: str
    research_question: str
    condition_id: str
    model_id: str
    dataset: str
    prompt_id: str
    representation: str
    temperature: float
    initial_seeds: tuple[int, ...]
    adaptive_seeds: tuple[int, ...]
    threshold: int
    task_labels: tuple[str, ...]
    request_timeout_seconds: float
    max_content_tokens: int | None = None
    assignment_path: str | None = None
    assignment_split: str | None = None
    assignment_protocol: str | None = None
    stage_memberships: tuple[str, ...] = ()

    @property
    def all_seeds(self) -> tuple[int, ...]:
        return self.initial_seeds + self.adaptive_seeds


class ConfigError(RuntimeError):
    pass


def _consensus_from_scope(scope: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    consensus = scope.get("consensus", {}) or {}
    if not isinstance(consensus, Mapping):
        raise ConfigError("consensus must be a mapping")

    initial = consensus.get("initial_seeds", scope.get("initial_seeds"))
    adaptive = consensus.get("adaptive_seeds", scope.get("adaptive_seeds"))
    threshold = int(consensus.get("threshold", 2))
    method = str(consensus.get("method", "adaptive_first_to_two")).casefold()
    if method not in {"first_to_two", "adaptive_first_to_two"}:
        raise ConfigError(f"Unsupported consensus method: {method!r}")
    if initial is None or adaptive is None:
        raise ConfigError("Consensus initial_seeds/adaptive_seeds are not defined")
    initial_t = tuple(int(x) for x in initial)
    adaptive_t = tuple(int(x) for x in adaptive)
    if not initial_t:
        raise ConfigError("initial_seeds cannot be empty")
    if len(set(initial_t + adaptive_t)) != len(initial_t + adaptive_t):
        raise ConfigError("Duplicate consensus seed")
    if threshold != 2:
        raise ConfigError("The frozen study consensus threshold must be 2")
    return initial_t, adaptive_t, threshold


def _timeout_from_overrides(overrides: Mapping[str, Any] | None) -> float:
    overrides = overrides or {}
    value = overrides.get("request_timeout_seconds")
    if value is None:
        raise ConfigError("Ollama job has no request_timeout_seconds override")
    value = float(value)
    if value <= 0:
        raise ConfigError("request_timeout_seconds must be > 0")
    return value


def _task_labels(experiment: Mapping[str, Any], default_labels: Sequence[str]) -> tuple[str, ...]:
    subset = experiment.get("subset")
    if isinstance(subset, Mapping) and subset.get("label_space"):
        labels = tuple(str(x).casefold() for x in subset["label_space"])
    else:
        labels = tuple(default_labels)
    unknown = sorted(set(labels) - set(default_labels))
    if unknown:
        raise ConfigError(f"Task label space contains unknown labels: {unknown}")
    return labels


def _validate_consensus_against_evaluation(
    initial: tuple[int, ...], adaptive: tuple[int, ...], evaluation: Mapping[str, Any]
) -> None:
    ref = evaluation.get("consensus", {}) or {}
    ref_initial = tuple(int(x) for x in ref.get("initial_seeds", []))
    ref_adaptive = tuple(int(x) for x in ref.get("adaptive_seeds", []))
    if initial != ref_initial or adaptive != ref_adaptive:
        raise ConfigError(
            "Experiment consensus seeds differ from config/evaluation.yaml. "
            f"experiment initial={initial}, adaptive={adaptive}; "
            f"evaluation initial={ref_initial}, adaptive={ref_adaptive}"
        )


def expand_jobs(
    experiment: Mapping[str, Any],
    *,
    model_registry: Mapping[str, Mapping[str, Any]],
    evaluation: Mapping[str, Any],
) -> list[OllamaJob]:
    """Expand the package's four experiment-config shapes into Ollama jobs."""
    exp_id = str(experiment["id"])
    rq = str(experiment["research_question"])
    default_labels = tuple(evaluation["taxonomy"]["labels"])
    task_labels = _task_labels(experiment, default_labels)
    jobs: list[OllamaJob] = []

    # RQ1: shared execution settings + two temporal conditions + model list.
    if "shared" in experiment and "models" in experiment and "conditions" in experiment:
        shared = experiment["shared"]
        initial, adaptive, threshold = _consensus_from_scope(shared)
        _validate_consensus_against_evaluation(initial, adaptive, evaluation)
        for condition_id, cond in experiment["conditions"].items():
            for model_id in experiment["models"]:
                if model_id not in model_registry:
                    continue
                per_model = (cond.get("runtime_overrides", {}) or {}).get(model_id, {})
                jobs.append(OllamaJob(
                    experiment_id=exp_id,
                    research_question=rq,
                    condition_id=str(condition_id),
                    model_id=str(model_id),
                    dataset=str(cond["dataset"]),
                    prompt_id=str(shared["prompt_id"]),
                    representation=str(shared["representation"]),
                    temperature=float(shared["temperature"]),
                    initial_seeds=initial,
                    adaptive_seeds=adaptive,
                    threshold=threshold,
                    task_labels=task_labels,
                    request_timeout_seconds=_timeout_from_overrides(per_model),
                ))
        return jobs

    # RQ2: model comparison and temperature-sensitivity sub-experiments.
    if "model_comparison" in experiment and "temperature_sensitivity" in experiment:
        initial, adaptive, threshold = _consensus_from_scope(experiment)
        _validate_consensus_against_evaluation(initial, adaptive, evaluation)
        comparison = experiment["model_comparison"]
        for model_id in comparison["models"]:
            if model_id not in model_registry:
                continue
            overrides = (comparison.get("runtime_overrides", {}) or {}).get(model_id, {})
            jobs.append(OllamaJob(
                exp_id, rq, "model_comparison", str(model_id), str(experiment["dataset"]),
                str(experiment["prompt_id"]), str(experiment["representation"]),
                float(comparison["temperature"]), initial, adaptive, threshold, task_labels,
                _timeout_from_overrides(overrides),
            ))
        sensitivity = experiment["temperature_sensitivity"]
        model_id = str(sensitivity["model"])
        if model_id in model_registry:
            for temperature in sensitivity["temperatures"]:
                jobs.append(OllamaJob(
                    exp_id, rq, _temperature_slug(float(temperature)), model_id,
                    str(experiment["dataset"]), str(experiment["prompt_id"]),
                    str(experiment["representation"]), float(temperature),
                    initial, adaptive, threshold, task_labels,
                    float(sensitivity["request_timeout_seconds"]),
                ))
        return jobs

    # RQ3.1: one model, condition-specific prompt/representation.
    if "conditions" in experiment and "model" in experiment and "dataset" in experiment:
        initial, adaptive, threshold = _consensus_from_scope(experiment)
        _validate_consensus_against_evaluation(initial, adaptive, evaluation)
        model_id = str(experiment["model"])
        if model_id in model_registry:
            for condition_id, cond in experiment["conditions"].items():
                jobs.append(OllamaJob(
                    exp_id, rq, str(condition_id), model_id, str(experiment["dataset"]),
                    str(cond["prompt_id"]), str(cond["representation"]),
                    float(experiment["temperature"]), initial, adaptive, threshold, task_labels,
                    _timeout_from_overrides(experiment.get("runtime_overrides")),
                ))
        return jobs

    # RQ3.2: each unique prompt is an execution condition; repeated reference
    # prompts across stages are run only once and keep stage membership metadata.
    if "stages" in experiment and "model" in experiment and "dataset" in experiment:
        initial, adaptive, threshold = _consensus_from_scope(experiment)
        _validate_consensus_against_evaluation(initial, adaptive, evaluation)
        memberships: dict[str, list[str]] = {}
        order: list[str] = []
        for stage_id, stage in experiment["stages"].items():
            for prompt_id in stage.get("prompts", []):
                prompt_id = str(prompt_id)
                if prompt_id not in memberships:
                    memberships[prompt_id] = []
                    order.append(prompt_id)
                memberships[prompt_id].append(str(stage_id))
        model_id = str(experiment["model"])
        if model_id in model_registry:
            for prompt_id in order:
                jobs.append(OllamaJob(
                    exp_id, rq, prompt_id, model_id, str(experiment["dataset"]), prompt_id,
                    str(experiment["representation"]), float(experiment["temperature"]),
                    initial, adaptive, threshold, task_labels,
                    _timeout_from_overrides(experiment.get("runtime_overrides")),
                    stage_memberships=tuple(memberships[prompt_id]),
                ))
        return jobs

    # RQ4: shared settings plus heterogeneous conditions. Only conditions whose
    # model field resolves through models.yaml belong to this Ollama runner.
    if "shared" in experiment and "conditions" in experiment:
        shared = experiment["shared"]
        initial, adaptive, threshold = _consensus_from_scope(shared)
        _validate_consensus_against_evaluation(initial, adaptive, evaluation)
        assignment_path = None
        assignment_split = None
        assignment_protocol = None
        if isinstance(experiment.get("split"), Mapping):
            split_cfg = experiment["split"]
            assignment_path = str(split_cfg["assignments"])
            assignment_split = "test"
            # The assignment file is already protocol-specific. Do not filter by
            # the human-readable protocol name from the experiment YAML; preserve
            # the manifest's own protocol field and select only held-out rows.
            assignment_protocol = None
        elif isinstance(experiment.get("subset"), Mapping):
            assignment_path = str(experiment["subset"]["assignments"])

        for condition_id, cond in experiment["conditions"].items():
            model_id = cond.get("model")
            if model_id not in model_registry:
                continue
            jobs.append(OllamaJob(
                exp_id, rq, str(condition_id), str(model_id), str(shared["dataset"]),
                str(cond["prompt_id"]), str(shared["representation"]),
                float(cond["temperature"]), initial, adaptive, threshold, task_labels,
                _timeout_from_overrides(cond.get("runtime_overrides")),
                max_content_tokens=(
                    int((cond.get("input_constraint", {}) or {}).get("context_window_tokens"))
                    if (cond.get("input_constraint", {}) or {}).get("context_window_tokens") is not None
                    else (
                        int((experiment.get("subset", {}) or {}).get("context_window_tokens"))
                        if isinstance(experiment.get("subset"), Mapping)
                        and (experiment.get("subset", {}) or {}).get("context_window_tokens") is not None
                        else None
                    )
                ),
                assignment_path=assignment_path,
                assignment_split=assignment_split,
                assignment_protocol=assignment_protocol,
            ))
        return jobs

    raise ConfigError(f"Unsupported experiment config shape for {exp_id!r}")


def _prompt_variables(row: Any, *, representation: str, prepared_input: str) -> dict[str, Any]:
    """Provide stable aliases used by the package's prompt templates.

    INPUT/INPUT_TEXT/COMMIT always refer to the exact prepared representation.
    Representation-specific aliases are also populated to support historical
    prompt files without leaking the target label.
    """
    def get(name: str) -> str:
        value = getattr(row, name, "")
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        return str(value)

    processed_message = get("processed_message")
    original_message = get("original_message")
    diff = get("diff")
    git_show = get("git_show")

    # Preserve the 30k-token cap when a historical prompt uses separate MESSAGE
    # and DIFF placeholders for the message+diff representation. Message text is
    # the prefix of that canonical representation, so the clipped suffix is the
    # effective diff when the message itself fits.
    prepared_message = processed_message
    prepared_diff = diff
    prepared_git_show = git_show
    separator = "\n\n"
    if representation == "message_only":
        prepared_message = prepared_input
    elif representation == "diff_only":
        prepared_diff = prepared_input
    elif representation == "git_show":
        prepared_git_show = prepared_input
    elif representation == "message_and_diff":
        prefix = processed_message + separator
        if prepared_input.startswith(prefix):
            prepared_diff = prepared_input[len(prefix):]
        elif prepared_input == processed_message:
            prepared_diff = ""

    return {
        "INPUT": prepared_input,
        "INPUT_TEXT": prepared_input,
        "CONTENT": prepared_input,
        "COMMIT": prepared_input,
        "COMMIT_DATA": prepared_input,
        "MESSAGE": prepared_message,
        "COMMIT_MESSAGE": prepared_message,
        "PROCESSED_MESSAGE": prepared_message,
        "ORIGINAL_MESSAGE": original_message,
        "DIFF": prepared_diff,
        "GIT_DIFF": prepared_diff,
        "GIT_SHOW": prepared_git_show,
        "BREAKING_CHANGE": str(bool(getattr(row, "breaking_change", False))).lower(),
    }


def _existing_index(rows: Sequence[Mapping[str, Any]], run_id: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("run_id", "")) != run_id:
            raise RuntimeError(
                "Existing raw prediction file contains a different run_id; "
                "use a different --output-dir instead of mixing runs."
            )
        key = (str(row["project"]), str(row["commit_hash"]), int(row["seed"]))
        previous = index.get(key)
        if previous is not None:
            if previous != dict(row):
                raise RuntimeError(f"Conflicting duplicate raw prediction for {key}")
            continue
        index[key] = dict(row)
    return index


def _consensus_for_row(row: Any, existing: Mapping[tuple[str, str, int], Mapping[str, Any]], job: OllamaJob, resolve_first_to_two):
    by_seed = {
        seed: str(existing[(str(row.project), str(row.commit_hash), seed)]["prediction"])
        for seed in job.all_seeds
        if (str(row.project), str(row.commit_hash), seed) in existing
    }
    return resolve_first_to_two(
        by_seed,
        initial_seeds=job.initial_seeds,
        adaptive_seeds=job.adaptive_seeds,
        task_labels=job.task_labels,
        threshold=job.threshold,
        invalid_terminal=True,
    )


def _final_prediction_frame(df: pd.DataFrame, existing: Mapping[tuple[str, str, int], Mapping[str, Any]], job: OllamaJob, resolve_first_to_two) -> pd.DataFrame:
    rows = []
    for item in df.itertuples(index=False):
        result = _consensus_for_row(item, existing, job, resolve_first_to_two)
        if not result.complete:
            raise RuntimeError(
                f"Incomplete consensus after execution for {item.project}/{item.commit_hash}: "
                f"status={result.status}, missing_seed={result.missing_seed}"
            )
        out = {
            "experiment_id": job.experiment_id,
            "condition_id": job.condition_id,
            "model_id": job.model_id,
            "partition": str(item.partition),
            "platform": str(item.platform),
            "project": str(item.project),
            "commit_hash": str(item.commit_hash),
            "target": str(item.target),
            "prediction": result.classification,
            "consensus_status": result.status,
            "decisive_seed": result.decisive_seed,
            "used_seeds": ",".join(str(x) for x in result.used_seeds),
            "protocol": str(getattr(item, "protocol", "")),
            "fold": getattr(item, "fold", None),
            "split": str(getattr(item, "split", "")),
        }
        for seed in job.all_seeds:
            rec = existing.get((str(item.project), str(item.commit_hash), seed))
            out[f"seed_{seed}"] = None if rec is None else rec["prediction"]
        rows.append(out)
    return pd.DataFrame(rows)


def _job_payload(job: OllamaJob) -> dict[str, Any]:
    return {
        "experiment_id": job.experiment_id,
        "research_question": job.research_question,
        "condition_id": job.condition_id,
        "model_id": job.model_id,
        "dataset": job.dataset,
        "prompt_id": job.prompt_id,
        "representation": job.representation,
        "temperature": job.temperature,
        "initial_seeds": list(job.initial_seeds),
        "adaptive_seeds": list(job.adaptive_seeds),
        "threshold": job.threshold,
        "task_labels": list(job.task_labels),
        "request_timeout_seconds": job.request_timeout_seconds,
        "max_content_tokens": job.max_content_tokens,
        "assignment_path": job.assignment_path,
        "assignment_split": job.assignment_split,
        "assignment_protocol": job.assignment_protocol,
        "stage_memberships": list(job.stage_memberships),
    }


def _job_output_dir(base: Path, job: OllamaJob) -> Path:
    return base / _slug(job.experiment_id) / _slug(job.condition_id) / _slug(job.model_id)


def _print_job(job: OllamaJob, index: int | None = None) -> None:
    prefix = f"[{index}] " if index is not None else ""
    assignment = f", assignments={job.assignment_path}" if job.assignment_path else ""
    print(
        f"{prefix}{job.condition_id}: model={job.model_id}, dataset={job.dataset}, "
        f"prompt={job.prompt_id}, representation={job.representation}, "
        f"temperature={job.temperature:g}, seeds={list(job.initial_seeds)}+{list(job.adaptive_seeds)}, "
        f"timeout={job.request_timeout_seconds:g}s{assignment}"
    )


async def _run_job(
    *,
    repo_root: Path,
    job: OllamaJob,
    output_base: Path,
    prompt_manifest: Path,
    inference_cfg: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    experiment_path: Path,
    inference_path: Path,
    evaluation_path: Path,
    models_path: Path,
    resume: bool,
    dry_run: bool,
    project_filter: set[str] | None,
    limit: int | None,
    max_concurrent_override: int | None,
) -> None:
    # Imports are delayed so --help/--list-jobs work before the canonical parser
    # has been installed into src/parsing/.
    from src.consensus import resolve_first_to_two
    from src.data.loader import (
        cohort_sha256,
        load_assignments,
        load_canonical_dataset,
        materialize_assignments,
    )
    from src.inference.ollama import ModelHealthMonitor, OllamaBackend, OllamaRuntimeConfig
    from src.inference.schema import GenerativePredictionRecord
    from src.parsing.strict import PARSER_VERSION
    from src.prompts.loader import PromptRegistry
    from src.representations import add_input_text
    from src.utils.provenance import git_state, sha256_file

    dataset_path = _resolve_repo_path(repo_root, job.dataset)
    dataset = load_canonical_dataset(dataset_path)

    if job.assignment_path:
        assignment_path = _resolve_repo_path(repo_root, job.assignment_path)
        assignments = load_assignments(assignment_path)
        dataset = materialize_assignments(
            dataset,
            assignments,
            protocol=job.assignment_protocol,
            split=job.assignment_split,
        )
    else:
        assignment_path = None
        dataset = dataset.copy()
        dataset["protocol"] = ""
        dataset["fold"] = None
        dataset["split"] = ""

    dataset = add_input_text(dataset, job.representation)
    if project_filter:
        dataset = dataset[dataset["project"].isin(project_filter)].copy()
        if dataset.empty:
            raise RuntimeError(f"Project filter selected no rows: {sorted(project_filter)}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be > 0")
        dataset = dataset.head(limit).copy()
    dataset = dataset.reset_index(drop=True)
    if dataset.empty:
        raise RuntimeError("Selected dataset is empty")

    registry = PromptRegistry(prompt_manifest)
    prompt_path = registry.path(job.prompt_id)

    runtime = OllamaRuntimeConfig.from_mapping(inference_cfg)
    if job.max_content_tokens is not None:
        runtime = replace(runtime, max_content_tokens=int(job.max_content_tokens))
    backend = OllamaBackend(runtime)
    ollama_tag = str(model_spec["ollama_tag"])
    monitor = ModelHealthMonitor(
        ollama_tag,
        threshold=runtime.restart_after_consecutive_failures,
        restart_delay_seconds=runtime.restart_delay_seconds,
    )

    output_dir = _job_output_dir(output_base, job)
    raw_path = output_dir / "raw_predictions.jsonl.gz"
    final_path = output_dir / "predictions.csv.gz"
    manifest_path = output_dir / "run_manifest.json"

    selection_keys = dataset[["project", "commit_hash", "target"]].astype(str).to_dict(orient="records")
    selection_sha = _json_fingerprint({"rows": selection_keys})
    core = _job_payload(job)
    core.update({
        "parser_version": PARSER_VERSION,
        "model_ollama_tag": ollama_tag,
        "model_digest": model_spec.get("digest"),
        "dataset_sha256": sha256_file(dataset_path),
        "assignment_sha256": sha256_file(assignment_path) if assignment_path else None,
        "prompt_sha256": registry.sha256(job.prompt_id),
        "selection_sha256": selection_sha,
        "experiment_config_sha256": sha256_file(experiment_path),
        "inference_config_sha256": sha256_file(inference_path),
        "evaluation_config_sha256": sha256_file(evaluation_path),
        "models_config_sha256": sha256_file(models_path),
        "prompt_manifest_sha256": sha256_file(prompt_manifest),
    })
    run_id = _json_fingerprint(core)[:24]

    if manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("run_id") != run_id:
            raise RuntimeError(
                f"{output_dir} already contains a different run. "
                "Use a different --output-dir or restore the original selection/config."
            )
    elif raw_path.exists() or final_path.exists():
        raise RuntimeError(f"Output files exist without run_manifest.json: {output_dir}")

    if raw_path.exists() and not resume and not dry_run:
        raise RuntimeError(f"Existing run found at {output_dir}; pass --resume to continue it")

    print("\n" + "=" * 100)
    print(f"JOB: {job.experiment_id} / {job.condition_id} / {job.model_id}")
    print("=" * 100)
    print(f"Dataset rows:     {len(dataset):,}")
    print(f"Projects:         {dataset['project'].nunique():,}")
    print(f"Cohort SHA-256:   {cohort_sha256(dataset)}")
    print(f"Ollama tag:       {ollama_tag}")
    print(f"Prompt:           {job.prompt_id} ({prompt_path})")
    print(f"Representation:   {job.representation}")
    print(f"Temperature:      {job.temperature:g}")
    print(f"Initial seeds:    {list(job.initial_seeds)}")
    print(f"Adaptive seeds:   {list(job.adaptive_seeds)}")
    print(f"Input token cap:  {runtime.max_content_tokens:,}")
    print(f"Output:           {output_dir}")
    print(f"Run ID:           {run_id}")

    # Validate prompt rendering on the first selected commit even in dry-run.
    first = dataset.iloc[0]
    prepared_first, _ = backend.prepare_content(str(first["input_text"]))
    registry.render_messages(
        job.prompt_id,
        **_prompt_variables(first, representation=job.representation, prepared_input=prepared_first),
    )

    if dry_run:
        print("DRY RUN: configuration, dataset, assignment selection, model, and prompt resolved successfully.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    manifest = {
        "version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at_utc": started,
        "completed_at_utc": None,
        "job": _job_payload(job),
        "model": dict(model_spec),
        "ollama_tag": ollama_tag,
        "parser_version": PARSER_VERSION,
        "cohort_sha256": cohort_sha256(dataset),
        "selected_rows": len(dataset),
        "selected_projects": int(dataset["project"].nunique()),
        "hashes": {
            "experiment_config": sha256_file(experiment_path),
            "inference_config": sha256_file(inference_path),
            "evaluation_config": sha256_file(evaluation_path),
            "models_config": sha256_file(models_path),
            "dataset": sha256_file(dataset_path),
            "assignment": sha256_file(assignment_path) if assignment_path else None,
            "prompt_manifest": sha256_file(prompt_manifest),
            "prompt": registry.sha256(job.prompt_id),
        },
        "selection_sha256": selection_sha,
        "git": git_state(repo_root),
        "output_files": {
            "raw_predictions": str(raw_path.relative_to(repo_root)) if raw_path.is_relative_to(repo_root) else str(raw_path),
            "predictions": str(final_path.relative_to(repo_root)) if final_path.is_relative_to(repo_root) else str(final_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    existing_rows = _read_jsonl_gz(raw_path)
    existing = _existing_index(existing_rows, run_id)
    if existing:
        print(f"Resume state:     {len(existing):,} completed commit/seed calls")

    execution_cfg = inference_cfg["execution"]
    max_concurrent = int(max_concurrent_override or execution_cfg["max_concurrent_requests"])
    batch_size = int(execution_cfg["batch_size"])
    semaphore = asyncio.Semaphore(max_concurrent)

    async def infer_one(item: Any, seed: int) -> dict[str, Any]:
        prepared_input, _ = backend.prepare_content(str(item.input_text))
        messages = registry.render_messages(
            job.prompt_id,
            **_prompt_variables(item, representation=job.representation, prepared_input=prepared_input),
        )
        async with semaphore:
            result = await backend.infer(
                model=ollama_tag,
                messages=messages,
                prepared_input=prepared_input,
                seed=seed,
                temperature=job.temperature,
                request_timeout_seconds=job.request_timeout_seconds,
                task_labels=job.task_labels,
                health_monitor=monitor,
            )
        record = GenerativePredictionRecord(
            experiment_id=job.experiment_id,
            condition_id=job.condition_id,
            run_id=run_id,
            backend="ollama",
            model_id=job.model_id,
            prompt_id=job.prompt_id,
            representation=job.representation,
            partition=str(item.partition),
            platform=str(item.platform),
            project=str(item.project),
            commit_hash=str(item.commit_hash),
            target=str(item.target),
            seed=int(seed),
            prediction=result.parsed.prediction,
            correct=result.parsed.prediction == str(item.target),
            raw_response=result.raw_response,
            parse_status=result.parsed.parse_status,
            parser_label=result.parsed.parser_label,
            parser_raw_label=result.parsed.parser_raw_label,
            normalization_rule=result.parsed.normalization_rule,
            technical_status=result.technical_status,
            processing_time_seconds=result.processing_time_seconds,
            input_tokens=result.input_tokens,
            num_ctx=result.num_ctx,
            input_sha256=hashlib.sha256(prepared_input.encode("utf-8")).hexdigest(),
            protocol=str(getattr(item, "protocol", "")),
            fold=None if pd.isna(getattr(item, "fold", None)) else int(getattr(item, "fold")),
            split=str(getattr(item, "split", "")),
        ).as_dict()
        output_cfg = inference_cfg.get("output", {}) or {}
        if bool(output_cfg.get("preserve_input_text", False)):
            record["input_text"] = prepared_input
        if bool(output_cfg.get("preserve_full_prompt", False)):
            record["full_prompt"] = messages
        return record

    async def execute_seed(seed: int, selected: pd.DataFrame) -> None:
        if selected.empty:
            return
        total = len(selected)
        print(f"seed {seed}: executing {total:,} commit(s)")
        records = list(selected.itertuples(index=False))
        for start in range(0, total, batch_size):
            batch = records[start:start + batch_size]
            produced = await asyncio.gather(*(infer_one(item, seed) for item in batch))
            _append_jsonl_gz(raw_path, produced)
            for rec in produced:
                key = (str(rec["project"]), str(rec["commit_hash"]), int(rec["seed"]))
                existing[key] = rec
            done = min(start + len(batch), total)
            print(f"  seed {seed}: {done:,}/{total:,}", flush=True)

    # Initial block is mandatory: execute every missing base seed for every commit.
    for seed in job.initial_seeds:
        missing_mask = [
            (str(r.project), str(r.commit_hash), seed) not in existing
            for r in dataset.itertuples(index=False)
        ]
        selected = dataset.loc[missing_mask].copy()
        await execute_seed(seed, selected)

    # Adaptive execution: each seed is run only for commits whose consensus says
    # that exact next seed is required.
    for seed in job.adaptive_seeds:
        needed_indices = []
        for idx, item in enumerate(dataset.itertuples(index=False)):
            result = _consensus_for_row(item, existing, job, resolve_first_to_two)
            if not result.complete and result.missing_seed == seed:
                needed_indices.append(idx)
        selected = dataset.iloc[needed_indices].copy() if needed_indices else dataset.iloc[0:0].copy()
        await execute_seed(seed, selected)

    final = _final_prediction_frame(dataset, existing, job, resolve_first_to_two)
    final.to_csv(final_path, index=False, compression="gzip")

    seed_counts = {
        str(seed): int(sum(k[2] == seed for k in existing))
        for seed in job.all_seeds
    }
    invalid_count = int((final["prediction"] == "<invalid_output>").sum())
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = _utc_now()
    manifest["completed_seed_calls"] = len(existing)
    manifest["seed_counts"] = seed_counts
    manifest["final_invalid_predictions"] = invalid_count
    manifest["model_restarts"] = monitor.total_restarts
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\nCompleted")
    print(f"Final rows:       {len(final):,}")
    print(f"Invalid final:    {invalid_count:,}")
    print(f"Seed calls:       {seed_counts}")
    print(f"Model restarts:   {monitor.total_restarts}")


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, help="Experiment YAML path, relative to repository root or absolute.")
    p.add_argument("--repo-root", default=str(default_root), help="Replication-package repository root.")
    p.add_argument("--output-dir", default=None, help="Base output directory. Default: <repo>/runs")
    p.add_argument("--prompt-manifest", default=None, help="Default: prompts/manifest.yaml (fallback: manifest.yaml).")
    p.add_argument("--model", action="append", default=[], help="Restrict to one or more model IDs. Repeatable.")
    p.add_argument("--condition", action="append", default=[], help="Restrict to condition IDs. Repeatable.")
    p.add_argument("--prompt", action="append", default=[], help="Restrict to prompt IDs. Repeatable.")
    p.add_argument("--project", action="append", default=[], help="Restrict selected dataset to project(s). Repeatable.")
    p.add_argument("--limit", type=int, default=None, help="Stable first-N commit smoke test; use a separate output directory.")
    p.add_argument("--max-concurrent", type=int, default=None, help="Override inference.yaml max_concurrent_requests.")
    p.add_argument("--resume", action="store_true", help="Continue an existing matching run without repeating completed calls.")
    p.add_argument("--dry-run", action="store_true", help="Resolve and validate jobs/datasets/prompts but do not call Ollama.")
    p.add_argument("--list-jobs", action="store_true", help="Print expanded Ollama jobs and exit.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not repo_root.exists():
        raise FileNotFoundError(repo_root)
    sys.path.insert(0, str(repo_root))

    # These imports do not require the canonical parser, so job listing remains
    # useful before tools/install_canonical_parser.py is run.
    from src.config import load_evaluation_config, load_experiment, load_inference_config, load_models

    experiment_path = _resolve_repo_path(repo_root, args.experiment)
    inference_path = repo_root / "config/inference.yaml"
    evaluation_path = repo_root / "config/evaluation.yaml"
    models_path = repo_root / "config/models.yaml"

    experiment = load_experiment(experiment_path)
    inference_cfg = load_inference_config(inference_path)
    evaluation = load_evaluation_config(evaluation_path)
    models = load_models(models_path)

    jobs = expand_jobs(experiment, model_registry=models, evaluation=evaluation)
    if args.model:
        wanted = set(args.model)
        jobs = [j for j in jobs if j.model_id in wanted]
    if args.condition:
        wanted = set(args.condition)
        jobs = [j for j in jobs if j.condition_id in wanted]
    if args.prompt:
        wanted = set(args.prompt)
        jobs = [j for j in jobs if j.prompt_id in wanted]
    if not jobs:
        raise RuntimeError(
            "No Ollama-backed jobs matched the experiment/filters. FLACCS, HQCM, "
            "SVM, BiLSTM, and CodeBERT require their dedicated runners."
        )

    print(f"Expanded {len(jobs)} Ollama job(s) from {experiment_path}")
    for i, job in enumerate(jobs, 1):
        _print_job(job, i)
    if args.list_jobs:
        return 0

    prompt_manifest = (
        _resolve_repo_path(repo_root, args.prompt_manifest)
        if args.prompt_manifest
        else (repo_root / "prompts/manifest.yaml")
    )
    if not prompt_manifest.exists():
        fallback = repo_root / "manifest.yaml"
        if fallback.exists():
            prompt_manifest = fallback
        else:
            raise FileNotFoundError(
                f"Prompt manifest not found at {prompt_manifest} or {fallback}"
            )

    output_base = (
        _resolve_repo_path(repo_root, args.output_dir)
        if args.output_dir
        else (repo_root / "runs")
    )
    project_filter = set(args.project) if args.project else None

    async def run_all():
        for job in jobs:
            await _run_job(
                repo_root=repo_root,
                job=job,
                output_base=output_base,
                prompt_manifest=prompt_manifest,
                inference_cfg=inference_cfg,
                model_spec=models[job.model_id],
                experiment_path=experiment_path,
                inference_path=inference_path,
                evaluation_path=evaluation_path,
                models_path=models_path,
                resume=args.resume,
                dry_run=args.dry_run,
                project_filter=project_filter,
                limit=args.limit,
                max_concurrent_override=args.max_concurrent,
            )

    asyncio.run(run_all())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
