#!/usr/bin/env python3
"""Run the FLACCS artifact-compatible RQ4.2 condition (``config/experiments/rq4_2/artifact_flaccs.yaml``).

Examples
--------
python scripts/run_flaccs_experiment.py --list-jobs

python scripts/run_flaccs_experiment.py --dry-run

python scripts/run_flaccs_experiment.py --resume
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from src.consensus import AdaptiveSeedPolicy

import pandas as pd

DEFAULT_EXPERIMENT = "config/experiments/rq4_2/artifact_flaccs.yaml"


class ConfigError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: Any) -> str:
    import re
    text = str(value).strip().casefold().replace("+", "plus")
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    return text.strip("_.-") or "condition"


def _resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fingerprint(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return _sha256_text(text)


def protect_output(repo_root: Path, path: Path) -> None:
    for root in {repo_root.resolve(), Path(__file__).resolve().parents[1]}:
        if path.resolve().is_relative_to((root / "results").resolve()):
            raise ConfigError(f"Output must be outside frozen results: {path}")


# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlaccsJob:
    experiment_id: str
    research_question: str
    condition_id: str
    model: str
    dataset: str
    representation: str
    prompt_id: str
    initial_seeds: tuple[int, ...]
    adaptive_policy: "AdaptiveSeedPolicy"
    threshold: int
    task_labels: tuple[str, ...]
    assignment_path: str
    do_sample: bool
    max_new_tokens: int
    message_max_tokens: int
    context_window_tokens: int


def _task_labels(experiment: Mapping[str, Any], default_labels: tuple[str, ...]) -> tuple[str, ...]:
    subset = experiment.get("subset")
    if not isinstance(subset, Mapping) or not subset.get("label_space"):
        raise ConfigError("Experiment is missing subset.label_space")
    labels = tuple(str(x).casefold() for x in subset["label_space"])
    unknown = sorted(set(labels) - set(default_labels))
    if unknown:
        raise ConfigError(f"Task label space contains unknown labels: {unknown}")
    return labels


def expand_jobs(experiment: Mapping[str, Any], *, evaluation: Mapping[str, Any]) -> list[FlaccsJob]:
    """Expand an artifact_flaccs.yaml-shaped experiment into FLACCS jobs.

    Only conditions with an ``inference`` key are run here; ``gpt_oss_20b`` runs
    with ``run_ollama_experiment.py``.
    """
    from src.consensus import AdaptiveSeedPolicy, validate_base_seeds

    exp_id = str(experiment["id"])
    rq = str(experiment["research_question"])
    default_labels = tuple(evaluation["taxonomy"]["labels"])
    task_labels = _task_labels(experiment, default_labels)

    shared = experiment["shared"]
    consensus = shared.get("consensus", {}) or {}
    if consensus.get("method") not in {"first_to_two", "adaptive_first_to_two"}:
        raise ConfigError("Unsupported consensus method")
    if consensus.get("threshold", 2) != 2:
        raise ConfigError("Consensus threshold must be 2")
    initial = tuple(shared["initial_seeds"])
    validate_base_seeds(initial)
    adaptive = AdaptiveSeedPolicy(**shared["adaptive_policy"])

    subset = experiment["subset"]
    assignment_path = str(subset["assignments"])

    jobs: list[FlaccsJob] = []
    for condition_id, cond in experiment["conditions"].items():
        inference = cond.get("inference")
        if inference is None:
            continue
        jobs.append(FlaccsJob(
            experiment_id=exp_id, research_question=rq, condition_id=str(condition_id),
            model=str(cond["model"]), dataset=str(shared["dataset"]), representation=str(shared["representation"]),
            prompt_id=str(cond["prompt_id"]), initial_seeds=initial, adaptive_policy=adaptive, threshold=2,
            task_labels=task_labels, assignment_path=assignment_path,
            do_sample=bool(inference["do_sample"]), max_new_tokens=int(inference["max_new_tokens"]),
            message_max_tokens=int(inference["message_max_tokens"]),
            context_window_tokens=int(inference["context_window_tokens"]),
        ))
    return jobs


def _job_output_dir(base: Path, job: FlaccsJob) -> Path:
    return base / _slug(job.experiment_id) / _slug(job.condition_id)


def _print_job(job: FlaccsJob, index: int | None = None) -> None:
    prefix = f"[{index}] " if index is not None else ""
    print(
        f"{prefix}{job.condition_id}: model={job.model}, dataset={job.dataset}, prompt={job.prompt_id}, "
        f"labels={list(job.task_labels)}, do_sample={job.do_sample}, "
        f"message_max_tokens={job.message_max_tokens}, context_window_tokens={job.context_window_tokens}, "
        f"seeds={list(job.initial_seeds)}+{asdict(job.adaptive_policy)}"
    )


# ---------------------------------------------------------------------------
# Consensus (verbatim reuse of the study-wide policy)
# ---------------------------------------------------------------------------

def _consensus_for_row(row: Any, existing: Mapping[tuple[str, str, int], Mapping[str, Any]], job: FlaccsJob, resolve_first_to_two):
    prefix = (str(row.project), str(row.commit_hash))
    by_seed = {seed: str(existing[(*prefix, seed)]["prediction"])
               for seed in job.initial_seeds if (*prefix, seed) in existing}
    seed = job.adaptive_policy.start
    while (*prefix, seed) in existing:
        by_seed[seed] = str(existing[(*prefix, seed)]["prediction"])
        seed += job.adaptive_policy.step
    return resolve_first_to_two(
        by_seed, initial_seeds=job.initial_seeds, adaptive_policy=job.adaptive_policy,
        task_labels=job.task_labels, threshold=job.threshold, invalid_terminal=True,
    )


def _existing_index(rows: list[dict], run_id: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        required = ("project", "commit_hash", "target", "prediction", "raw_response",
                    "parse_status", "technical_status", "run_id")
        if any(not isinstance(row.get(f), str) for f in required) or type(row.get("seed")) is not int:
            raise RuntimeError("Malformed raw prediction record")
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


def _final_prediction_frame(df: pd.DataFrame, existing: Mapping[tuple[str, str, int], Mapping[str, Any]],
                             job: FlaccsJob, resolve_first_to_two) -> pd.DataFrame:
    rows = []
    output_seeds = sorted(set(job.initial_seeds) | {key[2] for key in existing})
    for item in df.itertuples(index=False):
        result = _consensus_for_row(item, existing, job, resolve_first_to_two)
        if not result.complete:
            raise RuntimeError(
                f"Incomplete consensus after execution for {item.project}/{item.commit_hash}: "
                f"status={result.status}, missing_seed={result.missing_seed}"
            )
        out = {
            "experiment_id": job.experiment_id, "condition_id": job.condition_id, "model_id": job.model,
            "partition": str(item.partition), "platform": str(item.platform),
            "project": str(item.project), "commit_hash": str(item.commit_hash), "target": str(item.target),
            "prediction": result.classification, "consensus_status": result.status,
            "decisive_seed": result.decisive_seed, "used_seeds": ",".join(str(x) for x in result.used_seeds),
            "protocol": str(getattr(item, "protocol", "")), "fold": getattr(item, "fold", None),
            "split": str(getattr(item, "split", "")),
        }
        for seed in output_seeds:
            rec = existing.get((str(item.project), str(item.commit_hash), seed))
            out[f"seed_{seed}"] = None if rec is None else rec["prediction"]
        rows.append(out)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _run_job(*, repo_root: Path, job: FlaccsJob, output_base: Path, **kwargs) -> None:
    from src.utils.journal import exclusive_run
    protect_output(repo_root, output_base)
    directory = _job_output_dir(output_base, job)
    protect_output(repo_root, directory)
    if kwargs["dry_run"]:
        _run_job_locked(repo_root=repo_root, job=job, output_base=output_base, **kwargs)
        return
    lock_path = output_base / ".locks" / (_sha256_text(str(directory.resolve())) + ".lock")
    protect_output(repo_root, lock_path)
    with exclusive_run(lock_path):
        _run_job_locked(repo_root=repo_root, job=job, output_base=output_base, **kwargs)


def _run_job_locked(
    *, repo_root: Path, job: FlaccsJob, output_base: Path, experiment_path: Path, evaluation_path: Path,
    prompt_manifest: Path, resume: bool, dry_run: bool, limit: int | None,
) -> None:
    from src.data.loader import (
        cohort_sha256, load_assignments, load_canonical_dataset, materialize_assignments,
        validate_assignment_selection,
    )
    from src.prompts.loader import PromptRegistry
    from src.utils.provenance import git_state, sha256_file

    output_dir = _job_output_dir(output_base, job)
    protect_output(repo_root, output_dir)
    if output_dir.exists() and not resume and not dry_run:
        raise RuntimeError(f"Existing job directory: {output_dir}; use --resume")

    dataset_path = _resolve_repo_path(repo_root, job.dataset)
    dataset = load_canonical_dataset(dataset_path)
    assignment_path = _resolve_repo_path(repo_root, job.assignment_path)
    assignments = load_assignments(assignment_path)
    # The FLACCS-compatible assignment manifest has no train/test partition or
    # fold; load_assignments() already synthesizes split="test"/fold=0 for the
    # whole membership, so no protocol/fold/split filter is applied here.
    selected = materialize_assignments(dataset, assignments)
    validate_assignment_selection(selected, assignments)
    if not set(selected["target"]).issubset(job.task_labels):
        raise ConfigError("Selected targets outside task taxonomy")

    registry = PromptRegistry(prompt_manifest)
    messages = registry.load_messages(job.prompt_id)
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ConfigError(f"Prompt {job.prompt_id!r} is not a single-user text prompt")
    template = messages[0]["content"]
    from src.inference.flaccs import MESSAGE_PLACEHOLDER, DIFF_PLACEHOLDER
    if MESSAGE_PLACEHOLDER not in template or DIFF_PLACEHOLDER not in template:
        raise ConfigError(f"Prompt {job.prompt_id!r} is missing {MESSAGE_PLACEHOLDER}/{DIFF_PLACEHOLDER}")

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be > 0")
        selected = selected.head(limit).copy()
    selected = selected.reset_index(drop=True)
    if selected.empty:
        raise RuntimeError("Selected dataset is empty")

    core = {
        "experiment_id": job.experiment_id, "condition_id": job.condition_id, "model": job.model,
        "prompt_id": job.prompt_id, "initial_seeds": list(job.initial_seeds),
        "adaptive_policy": asdict(job.adaptive_policy), "task_labels": list(job.task_labels),
        "do_sample": job.do_sample, "max_new_tokens": job.max_new_tokens,
        "message_max_tokens": job.message_max_tokens, "context_window_tokens": job.context_window_tokens,
        "dataset_sha256": sha256_file(dataset_path), "assignment_sha256": sha256_file(assignment_path),
        "prompt_sha256": registry.sha256(job.prompt_id),
        "experiment_config_sha256": sha256_file(experiment_path), "evaluation_config_sha256": sha256_file(evaluation_path),
        "prompt_manifest_sha256": sha256_file(prompt_manifest),
    }
    run_id = _fingerprint(core)[:24]

    manifest_path = output_dir / "run_manifest.json"
    journal_path = output_dir / "raw_predictions.journal.jsonl"
    final_path = output_dir / "predictions.csv.gz"

    old_manifest = None
    if manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("run_id") != run_id or old_manifest.get("identity") != core:
            raise RuntimeError(f"{output_dir} already contains a different run. Use a different --output-dir.")
    elif not dry_run and output_dir.exists() and (journal_path.exists() or final_path.exists()):
        raise RuntimeError(f"Output files exist without run_manifest.json: {output_dir}")

    print("\n" + "=" * 100)
    print(f"JOB: {job.experiment_id} / {job.condition_id}")
    print("=" * 100)
    print(f"Model:            {job.model}")
    print(f"Rows:             {len(selected):,}")
    print(f"Cohort SHA-256:   {cohort_sha256(selected)}")
    print(f"Prompt:           {job.prompt_id}")
    print(f"do_sample:        {job.do_sample}")
    print(f"Message/context:  {job.message_max_tokens}/{job.context_window_tokens} tokens")
    print(f"Initial seeds:    {list(job.initial_seeds)}")
    print(f"Adaptive seeds:   {asdict(job.adaptive_policy)}")
    print(f"Output:           {output_dir}")
    print(f"Run ID:           {run_id}")

    if dry_run:
        print("DRY RUN OK: config, assignments, and prompt template validated; model not loaded, no inference performed.")
        return

    from src.consensus import resolve_first_to_two
    from src.inference.flaccs import FlaccsBackend, FlaccsConfig
    from src.inference.schema import GenerativePredictionRecord
    from src.utils.journal import Journal, atomic_bytes, atomic_json

    backend = FlaccsBackend(FlaccsConfig(
        model=job.model, max_new_tokens=job.max_new_tokens, do_sample=job.do_sample,
    ))

    output_dir.mkdir(parents=True, exist_ok=True)
    journal = Journal(journal_path)
    existing_rows, torn = journal.read()
    if not journal.path.exists() and final_path.exists():
        raise RuntimeError("Snapshot without authoritative journal; cannot safely resume")
    existing = _existing_index(existing_rows, run_id)
    journal.recover_tail(torn)
    journal.initialize()

    started = old_manifest["started_at_utc"] if old_manifest else _utc_now()
    manifest = {
        "version": 1, "run_id": run_id, "identity": core, "status": "running",
        "started_at_utc": started, "completed_at_utc": None,
        "job": {**asdict(job), "adaptive_policy": asdict(job.adaptive_policy)},
        "git": git_state(repo_root),
        "resume_events": (list(old_manifest.get("resume_events", [])) + [{"at_utc": _utc_now(), "completed_calls": len(existing)}]) if old_manifest else [],
    }
    atomic_json(manifest_path, manifest)
    if existing:
        print(f"Resume state:     {len(existing):,} completed (commit, seed) records")

    def execute_seed(seed: int, rows: pd.DataFrame) -> None:
        pending = [r for r in rows.itertuples(index=False) if (str(r.project), str(r.commit_hash), seed) not in existing]
        if not pending:
            if not rows.empty:
                print(f"seed {seed}: already complete for {len(rows):,} commit(s) -- skipping")
            return
        total = len(pending)
        print(f"seed {seed}: scoring {total:,} commit(s)")
        for position, item in enumerate(pending, start=1):
            prompt = backend.build_prompt(
                template=template, message=str(item.processed_message), diff=str(item.diff),
                message_max_tokens=job.message_max_tokens, context_window_tokens=job.context_window_tokens,
            )
            result = backend.infer(prompt=prompt, seed=seed, task_labels=job.task_labels)
            record = GenerativePredictionRecord(
                experiment_id=job.experiment_id, condition_id=job.condition_id, run_id=run_id,
                backend="flaccs", model_id=job.model, prompt_id=job.prompt_id, representation=job.representation,
                partition=str(item.partition), platform=str(item.platform), project=str(item.project),
                commit_hash=str(item.commit_hash), target=str(item.target), seed=int(seed),
                prediction=result.parsed.prediction, correct=result.parsed.prediction == str(item.target),
                raw_response=result.raw_response, parse_status=result.parsed.parse_status,
                parser_label=result.parsed.parser_label, parser_raw_label=result.parsed.parser_raw_label,
                normalization_rule=result.parsed.normalization_rule, technical_status=result.technical_status,
                processing_time_seconds=result.processing_time_seconds,
                input_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                protocol=str(getattr(item, "protocol", "")),
                fold=None if pd.isna(getattr(item, "fold", None)) else int(getattr(item, "fold")),
                split=str(getattr(item, "split", "")),
            ).as_dict()
            record["technical_detail"] = result.technical_detail
            key = (str(item.project), str(item.commit_hash), seed)
            journal.append(record)
            existing[key] = record
            if position % 25 == 0 or position == total:
                print(f"  seed {seed}: {position:,}/{total:,}")

    try:
        for seed in job.initial_seeds:
            execute_seed(seed, selected)

        while True:
            needed: dict[int, list[int]] = {}
            for idx, item in enumerate(selected.itertuples(index=False)):
                result = _consensus_for_row(item, existing, job, resolve_first_to_two)
                if not result.complete:
                    needed.setdefault(result.missing_seed, []).append(idx)
            if not needed:
                break
            for seed, indices in sorted(needed.items()):
                execute_seed(seed, selected.iloc[indices])

        final = _final_prediction_frame(selected, existing, job, resolve_first_to_two)
        atomic_bytes(final_path, gzip.compress(final.to_csv(index=False).encode("utf-8"), mtime=0))
        journal.snapshot(output_dir / "raw_predictions.jsonl.gz")

        seed_counts = {str(seed): int(sum(k[2] == seed for k in existing))
                       for seed in sorted(set(job.initial_seeds) | {key[2] for key in existing})}
        invalid_count = int((final["prediction"] == "<invalid_output>").sum())
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = _utc_now()
        manifest["completed_records"] = len(existing)
        manifest["seed_counts"] = seed_counts
        manifest["final_invalid_predictions"] = invalid_count
        atomic_json(manifest_path, manifest)

        print("\nCompleted")
        print(f"Final rows:       {len(final):,}")
        print(f"Invalid final:    {invalid_count:,}")
        print(f"Seed calls:       {seed_counts}")
    except BaseException as exc:
        manifest["status"] = "incomplete"
        manifest["interrupted_at_utc"] = _utc_now()
        manifest["interruption"] = f"{type(exc).__name__}: {exc}"
        manifest["completed_records"] = len(existing)
        atomic_json(manifest_path, manifest)
        journal.snapshot(output_dir / "raw_predictions.jsonl.gz")
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="Experiment YAML path, relative to repository root or absolute.")
    p.add_argument("--repo-root", default=str(default_root))
    p.add_argument("--output-dir", default=None, help="Base output directory. Default: <repo>/runs")
    p.add_argument("--prompt-manifest", default=None, help="Default: prompts/manifest.yaml")
    p.add_argument("--limit", type=int, default=None,
                   help="Truncate the selected commits to a small smoke test; use a separate --output-dir.")
    p.add_argument("--resume", action="store_true", help="Continue an existing matching run without repeating completed (commit, seed) records.")
    p.add_argument("--dry-run", action="store_true", help="Resolve and validate config/data/prompt but do not load the model or run inference.")
    p.add_argument("--list-jobs", action="store_true", help="Print expanded jobs and exit.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not repo_root.exists():
        raise FileNotFoundError(repo_root)
    sys.path.insert(0, str(repo_root))

    from src.config import load_evaluation_config, load_experiment

    experiment_path = _resolve_repo_path(repo_root, args.experiment)
    evaluation_path = repo_root / "config/evaluation.yaml"
    prompt_manifest = (
        _resolve_repo_path(repo_root, args.prompt_manifest)
        if args.prompt_manifest
        else (repo_root / "prompts/manifest.yaml")
    )
    if not prompt_manifest.exists():
        raise FileNotFoundError(prompt_manifest)

    experiment = load_experiment(experiment_path)
    evaluation = load_evaluation_config(evaluation_path)
    jobs = expand_jobs(experiment, evaluation=evaluation)
    if not jobs:
        raise RuntimeError("No FLACCS jobs matched the experiment (no condition carries an 'inference' key).")

    print(f"Expanded {len(jobs)} FLACCS job(s) from {experiment_path}")
    for i, job in enumerate(jobs, 1):
        _print_job(job, i)
    if args.list_jobs:
        return 0

    output_base = _resolve_repo_path(repo_root, args.output_dir) if args.output_dir else (repo_root / "runs")

    for job in jobs:
        _run_job(
            repo_root=repo_root, job=job, output_base=output_base,
            experiment_path=experiment_path, evaluation_path=evaluation_path, prompt_manifest=prompt_manifest,
            resume=args.resume, dry_run=args.dry_run, limit=args.limit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
