#!/usr/bin/env python3
"""Run the RQ4.1 trainable baselines: TF-IDF + SVM, BiLSTM, CodeBERT.

* ``--stage cross_project``: ``config/experiments/rq4_1/cross_project.yaml`` (folds 1-5);
* ``--stage within_corpus``: ``config/experiments/rq4_1/within_corpus.yaml`` (fold 0).

GPT-OSS runs with ``run_ollama_experiment.py``.

Output: <output-dir>/<experiment_id>/<condition>/fold_<N>/ (default output-dir:
runs/). Writing inside results/ is refused. --resume skips seeds already
completed for a fold. BiLSTM and CodeBERT require ``torch``.

Examples
--------
python scripts/run_baselines_experiment.py --stage cross_project --list-jobs

python scripts/run_baselines_experiment.py \
  --stage cross_project --model svm --dry-run

python scripts/run_baselines_experiment.py \
  --stage cross_project --model bilstm --model codebert --resume

Within-corpus preflight
-----------------------
``config/experiments/rq4_1/within_corpus.yaml`` sets ``expected_rows`` and
``expected_cohort_sha256`` for the test cohort. ``--stage within_corpus`` aborts
before training if ``assignments.csv`` does not match them; the split is never
regenerated. ``--stage cross_project`` has no such check.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from src.consensus import AdaptiveSeedPolicy

import pandas as pd

# Seed for the per-fold hyperparameter/epoch selection (not a prediction seed).
SELECTION_SEED = 0

MODEL_KEY_BY_CONDITION = {"tfidf_linear_svm": "svm", "bilstm": "bilstm", "codebert": "codebert"}
STAGE_FILES = {
    "cross_project": "config/experiments/rq4_1/cross_project.yaml",
    "within_corpus": "config/experiments/rq4_1/within_corpus.yaml",
}


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
class BaselineJob:
    experiment_id: str
    research_question: str
    condition_id: str
    model_key: str  # "svm" | "bilstm" | "codebert"
    dataset: str
    representation: str
    initial_seeds: tuple[int, ...]
    adaptive_policy: "AdaptiveSeedPolicy"
    threshold: int
    task_labels: tuple[str, ...]
    assignment_path: str
    protocol: str
    fold: int
    group_col: str | None
    training_config_path: str
    expected_rows: int | None = None
    expected_cohort_sha256: str | None = None


def expand_jobs(experiment: Mapping[str, Any], *, evaluation: Mapping[str, Any]) -> list[BaselineJob]:
    """Expand a cross_project.yaml/within_corpus.yaml into trainable-baseline jobs.

    Only conditions carrying a ``training_config`` key are trainable baselines
    (``tfidf_linear_svm``, ``bilstm``, ``codebert``); ``gpt_oss_20b``,
    ``flaccs``, and ``hqcm`` are skipped -- they belong to other runners.
    """
    from src.consensus import AdaptiveSeedPolicy, validate_base_seeds

    exp_id = str(experiment["id"])
    rq = str(experiment["research_question"])
    task_labels = tuple(evaluation["taxonomy"]["labels"])

    shared = experiment["shared"]
    consensus = shared.get("consensus", {}) or {}
    if consensus.get("method") not in {"first_to_two", "adaptive_first_to_two"}:
        raise ConfigError("Unsupported consensus method")
    if consensus.get("threshold", 2) != 2:
        raise ConfigError("Consensus threshold must be 2")
    initial = tuple(shared["initial_seeds"])
    validate_base_seeds(initial)
    adaptive = AdaptiveSeedPolicy(**shared["adaptive_policy"])

    split = experiment["split"]
    protocol = str(split["protocol"])
    assignment_path = str(split["assignments"])
    if protocol == "cross_project_5_fold":
        folds = list(range(1, int(split["folds"]) + 1))
        group_col = "project"
    elif protocol == "within_corpus_holdout":
        folds = [0]
        group_col = None
    else:
        raise ConfigError(f"Unsupported split protocol for the baselines runner: {protocol!r}")

    jobs: list[BaselineJob] = []
    for condition_id, cond in experiment["conditions"].items():
        training_config_path = cond.get("training_config")
        if training_config_path is None:
            continue
        model_key = MODEL_KEY_BY_CONDITION.get(condition_id)
        if model_key is None:
            raise ConfigError(f"Unknown trainable condition {condition_id!r}")
        for fold in folds:
            jobs.append(BaselineJob(
                experiment_id=exp_id, research_question=rq, condition_id=str(condition_id),
                model_key=model_key, dataset=str(shared["dataset"]), representation=str(shared["representation"]),
                initial_seeds=initial, adaptive_policy=adaptive, threshold=2, task_labels=task_labels,
                assignment_path=assignment_path, protocol=protocol, fold=fold, group_col=group_col,
                training_config_path=str(training_config_path),
                expected_rows=shared.get("expected_rows"),
                expected_cohort_sha256=shared.get("expected_cohort_sha256"),
            ))
    return jobs


SMOKE_TRAIN_PER_CLASS = 7


def _smoke_train_subset(train_df: pd.DataFrame, per_class: int = SMOKE_TRAIN_PER_CLASS) -> pd.DataFrame:
    """Training subset for --limit: up to `per_class` commits per class, taken from
    as many different projects as possible, in the original row order."""
    rank_in_project = train_df.groupby(["target", "project"], sort=False).cumcount()
    order = train_df.assign(_rank=rank_in_project.values, _pos=range(len(train_df)))
    picked = (order.sort_values(["target", "_rank", "_pos"], kind="stable")
                   .groupby("target", sort=False).head(per_class))
    return train_df.loc[picked.sort_values("_pos").index].copy()


def _job_output_dir(base: Path, job: BaselineJob) -> Path:
    return base / _slug(job.experiment_id) / _slug(job.condition_id) / f"fold_{job.fold}"


def _print_job(job: BaselineJob, index: int | None = None) -> None:
    prefix = f"[{index}] " if index is not None else ""
    print(
        f"{prefix}{job.condition_id} fold={job.fold}: model={job.model_key}, protocol={job.protocol}, "
        f"seeds={list(job.initial_seeds)}+{asdict(job.adaptive_policy)}, "
        f"training_config={job.training_config_path}"
    )


# ---------------------------------------------------------------------------
# Model adapters -- a uniform interface over svm/bilstm/codebert
# ---------------------------------------------------------------------------

def _base_config_from_yaml(model_key: str, cfg: Mapping[str, Any]):
    if model_key == "svm":
        from src.training.svm import SvmConfig
        t, s = cfg["tfidf"], cfg["svm"]
        return SvmConfig(
            max_features=t["max_features"], ngram_range=tuple(t["ngram_range"]), min_df=t["min_df"],
            max_df=t["max_df"], sublinear_tf=t["sublinear_tf"], lowercase=t["lowercase"],
            penalty=s["penalty"], loss=s["loss"], dual=s["dual"], tol=s["tol"],
            max_iter=s["max_iter"], class_weight=s["class_weight"],
        )
    if model_key == "bilstm":
        from src.training.bilstm import BiLstmConfig
        a = cfg["architecture"]
        return BiLstmConfig(
            max_vocab_size=a["max_vocab_size"], max_seq_len=a["max_seq_len"],
            embedding_dim=a["embedding_dim"], hidden_dim=a["hidden_dim"], num_layers=a["num_layers"],
            dropout=a["dropout"], batch_size=a["batch_size"], learning_rate=a["learning_rate"],
            weight_decay=a["weight_decay"], gradient_clip_norm=a["gradient_clip_norm"],
        )
    if model_key == "codebert":
        from src.training.codebert import CodeBertConfig
        a = cfg["architecture"]
        return CodeBertConfig(
            model_name=a["model_name"], max_length=a["max_length"], learning_rate=a["learning_rate"],
            train_batch_size=a["train_batch_size"], eval_batch_size=a["eval_batch_size"],
            weight_decay=a["weight_decay"],
        )
    raise ConfigError(f"Unknown model_key {model_key!r}")


def _select(model_key: str, train: pd.DataFrame, *, seed: int, selection_cfg: Mapping[str, Any],
            validation, group_col: str | None, base_config, scratch_dir: str):
    if model_key == "svm":
        from src.training.svm import select_c
        return select_c(train, seed=seed, grid=selection_cfg["grid"], validation=validation,
                         group_col=group_col, base_config=base_config)
    if model_key == "bilstm":
        from src.training.bilstm import select_num_epochs
        return select_num_epochs(train, seed=seed, max_epochs=selection_cfg["max_epochs"],
                                  patience=selection_cfg["patience"], validation=validation,
                                  group_col=group_col, base_config=base_config)
    if model_key == "codebert":
        from src.training.codebert import select_num_epochs
        return select_num_epochs(train, seed=seed, max_epochs=selection_cfg["max_epochs"],
                                  patience=selection_cfg["patience"], validation=validation,
                                  group_col=group_col, base_config=base_config, output_dir=scratch_dir)
    raise ConfigError(f"Unknown model_key {model_key!r}")


def _fit_predict(model_key: str, train: pd.DataFrame, test: pd.DataFrame, *, seed: int, config, scratch_dir: str) -> pd.DataFrame:
    if model_key == "svm":
        from src.training.svm import train_predict
        return train_predict(train, test, seed=seed, config=config)
    if model_key == "bilstm":
        from src.training.bilstm import train_predict
        return train_predict(train, test, seed=seed, config=config)
    if model_key == "codebert":
        from src.training.codebert import train_predict
        return train_predict(train, test, seed=seed, config=config, output_dir=scratch_dir)
    raise ConfigError(f"Unknown model_key {model_key!r}")


def _selected_value(model_key: str, config) -> Any:
    return config.C if model_key == "svm" else (config.num_epochs if model_key == "bilstm" else config.num_train_epochs)


def _config_with_selected_value(model_key: str, base_config, value: Any):
    from dataclasses import replace
    if model_key == "svm":
        return replace(base_config, C=float(value))
    if model_key == "bilstm":
        return replace(base_config, num_epochs=int(value))
    return replace(base_config, num_train_epochs=int(value))


# ---------------------------------------------------------------------------
# Consensus (verbatim reuse of the study-wide policy)
# ---------------------------------------------------------------------------

def _consensus_for_row(row: Any, existing: Mapping[tuple[str, str, int], Mapping[str, Any]], job: BaselineJob, resolve_first_to_two):
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


def _existing_index(rows: Sequence[Mapping[str, Any]], run_id: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        required = ("project", "commit_hash", "target", "prediction", "run_id")
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


def _final_prediction_frame(test_df: pd.DataFrame, existing: Mapping[tuple[str, str, int], Mapping[str, Any]],
                             job: BaselineJob, resolve_first_to_two) -> pd.DataFrame:
    rows = []
    output_seeds = sorted(set(job.initial_seeds) | {key[2] for key in existing})
    for item in test_df.itertuples(index=False):
        result = _consensus_for_row(item, existing, job, resolve_first_to_two)
        if not result.complete:
            raise RuntimeError(
                f"Incomplete consensus after execution for {item.project}/{item.commit_hash}: "
                f"status={result.status}, missing_seed={result.missing_seed}"
            )
        out = {
            "partition": str(item.partition), "model": job.model_key, "project": str(item.project),
            "commit_hash": str(item.commit_hash), "target": str(item.target), "prediction": result.classification,
        }
        for seed in output_seeds:
            rec = existing.get((str(item.project), str(item.commit_hash), seed))
            out[f"seed_{seed}"] = None if rec is None else rec["prediction"]
        rows.append(out)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _run_job(*, repo_root: Path, job: BaselineJob, output_base: Path, **kwargs) -> None:
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
    *, repo_root: Path, job: BaselineJob, output_base: Path, experiment_path: Path, evaluation_path: Path,
    resume: bool, dry_run: bool, limit: int | None,
) -> None:
    from src.consensus import resolve_first_to_two
    from src.data.loader import (
        cohort_sha256, load_assignments, load_canonical_dataset, materialize_assignments,
        validate_assignment_selection,
    )
    from src.config import load_training_config
    from src.representations import add_input_text
    from src.training.model_selection import ValidationSplitConfig
    from src.utils.journal import Journal, atomic_bytes, atomic_json
    from src.utils.provenance import git_state, sha256_file

    output_dir = _job_output_dir(output_base, job)
    protect_output(repo_root, output_dir)
    if output_dir.exists() and not resume and not dry_run:
        raise RuntimeError(f"Existing job directory: {output_dir}; use --resume")

    dataset_path = _resolve_repo_path(repo_root, job.dataset)
    dataset = load_canonical_dataset(dataset_path)
    assignment_path = _resolve_repo_path(repo_root, job.assignment_path)
    assignments = load_assignments(assignment_path)
    # The assignments file is already protocol-specific (its "protocol" column
    # holds a short value such as "cross_project", not the YAML's descriptive
    # split.protocol name); do not filter on it, mirroring
    # run_ollama_experiment.py's own handling of the same RQ4 manifests.
    full_fold = materialize_assignments(dataset, assignments, fold=job.fold)
    validate_assignment_selection(full_fold, assignments)

    train_df = full_fold[full_fold["split"] == "train"].reset_index(drop=True)
    test_df = full_fold[full_fold["split"] == "test"].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise RuntimeError(f"Fold {job.fold} has an empty train or test side")

    if job.expected_rows is not None and len(test_df) != job.expected_rows:
        raise ConfigError(
            f"Frozen cohort mismatch: assignments select {len(test_df)} test rows; expected {job.expected_rows}. "
            "The cohort must match expected_rows/expected_cohort_sha256 in the experiment YAML; "
            "no new split or training was performed."
        )
    if job.expected_cohort_sha256 is not None and cohort_sha256(test_df) != job.expected_cohort_sha256:
        raise ConfigError(
            "Frozen cohort mismatch: test cohort hash does not match expected_cohort_sha256 in the "
            "experiment YAML; no new split or training was performed."
        )
    if not set(full_fold["target"]).issubset(job.task_labels):
        raise ConfigError("Selected targets outside task taxonomy")

    train_df = add_input_text(train_df, job.representation)
    test_df = add_input_text(test_df, job.representation)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be > 0")
        train_df = _smoke_train_subset(train_df)
        test_df = test_df.head(limit).copy()

    training_config_path = _resolve_repo_path(repo_root, job.training_config_path)
    training_cfg = load_training_config(training_config_path)
    base_config = _base_config_from_yaml(job.model_key, training_cfg)
    selection_cfg = training_cfg["model_selection"]
    validation_cfg = ValidationSplitConfig(
        fraction=selection_cfg["validation"]["fraction"],
        stratify=selection_cfg["validation"].get("stratify", True),
    )

    core = {
        "experiment_id": job.experiment_id, "condition_id": job.condition_id, "model_key": job.model_key,
        "fold": job.fold, "protocol": job.protocol, "initial_seeds": list(job.initial_seeds),
        "adaptive_policy": asdict(job.adaptive_policy), "task_labels": list(job.task_labels),
        "selection_seed": SELECTION_SEED,
        "dataset_sha256": sha256_file(dataset_path), "assignment_sha256": sha256_file(assignment_path),
        "training_config_sha256": sha256_file(training_config_path),
        "experiment_config_sha256": sha256_file(experiment_path), "evaluation_config_sha256": sha256_file(evaluation_path),
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
    print(f"JOB: {job.experiment_id} / {job.condition_id} / fold={job.fold}")
    print("=" * 100)
    print(f"Model:            {job.model_key}")
    print(f"Train rows:       {len(train_df):,}")
    print(f"Test rows:        {len(test_df):,}")
    print(f"Test cohort SHA:  {cohort_sha256(test_df)}")
    print(f"Initial seeds:    {list(job.initial_seeds)}")
    print(f"Adaptive seeds:   {asdict(job.adaptive_policy)}")
    print(f"Output:           {output_dir}")
    print(f"Run ID:           {run_id}")

    if dry_run:
        print("DRY RUN OK: config, assignments, and training config validated; no training performed.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Model selection: once per fold, cached in the manifest across resumes ---
    if old_manifest and "model_selection" in old_manifest:
        selected_value = old_manifest["model_selection"]["selected_value"]
        selection_diagnostics = old_manifest["model_selection"]["diagnostics"]
        print(f"Model selection:  reused from manifest ({selected_value})")
    else:
        print(f"Model selection:  running (seed={SELECTION_SEED}, group_col={job.group_col!r})...")
        with tempfile.TemporaryDirectory(prefix="baseline_selection_") as scratch:
            selected_config, selection_diagnostics = _select(
                job.model_key, train_df, seed=SELECTION_SEED, selection_cfg=selection_cfg,
                validation=validation_cfg, group_col=job.group_col, base_config=base_config, scratch_dir=scratch,
            )
        selected_value = _selected_value(job.model_key, selected_config)
        print(f"Model selection:  selected_value={selected_value}, diagnostics={selection_diagnostics}")

    final_config = _config_with_selected_value(job.model_key, base_config, selected_value)

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
        "model_selection": {"selected_value": selected_value, "diagnostics": selection_diagnostics},
        "job": {**asdict(job), "adaptive_policy": asdict(job.adaptive_policy)},
        "git": git_state(repo_root),
        "resume_events": (list(old_manifest.get("resume_events", [])) + [{"at_utc": _utc_now(), "completed_calls": len(existing)}]) if old_manifest else [],
    }
    atomic_json(manifest_path, manifest)
    if existing:
        print(f"Resume state:     {len(existing):,} completed (commit, seed) records")

    def execute_seed(seed: int, selected: pd.DataFrame) -> None:
        if selected.empty:
            return
        missing_mask = [(str(r.project), str(r.commit_hash), seed) not in existing for r in selected.itertuples(index=False)]
        if not any(missing_mask):
            print(f"seed {seed}: already complete for this fold ({len(selected):,} commits) -- skipping retrain")
            return
        print(f"seed {seed}: training on {len(train_df):,} rows, scoring {len(selected):,} commit(s)")
        with tempfile.TemporaryDirectory(prefix=f"{job.model_key}_seed{seed}_") as scratch:
            preds = _fit_predict(job.model_key, train_df, selected, seed=seed, config=final_config, scratch_dir=scratch)
        for item in preds.itertuples(index=False):
            key = (str(item.project), str(item.commit_hash), seed)
            record = {
                "run_id": run_id, "project": str(item.project), "commit_hash": str(item.commit_hash),
                "target": str(item.target), "prediction": str(item.prediction), "seed": int(seed),
                "recorded_at_utc": _utc_now(),
            }
            if key in existing:
                if {k: v for k, v in existing[key].items() if k != "recorded_at_utc"} != \
                   {k: v for k, v in record.items() if k != "recorded_at_utc"}:
                    raise RuntimeError(f"Non-deterministic retrain produced a conflicting prediction for {key}")
                continue
            journal.append(record)
            existing[key] = record
        print(f"  seed {seed}: {len(selected):,} commit(s) scored")

    try:
        for seed in job.initial_seeds:
            execute_seed(seed, test_df)

        while True:
            needed: dict[int, list[int]] = {}
            for idx, item in enumerate(test_df.itertuples(index=False)):
                result = _consensus_for_row(item, existing, job, resolve_first_to_two)
                if not result.complete:
                    needed.setdefault(result.missing_seed, []).append(idx)
            if not needed:
                break
            for seed, indices in sorted(needed.items()):
                execute_seed(seed, test_df.iloc[indices])

        final = _final_prediction_frame(test_df, existing, job, resolve_first_to_two)
        atomic_bytes(final_path, gzip.compress(final.to_csv(index=False).encode("utf-8"), mtime=0))
        journal.snapshot(output_dir / "raw_predictions.jsonl.gz")

        seed_counts = {str(seed): int(sum(k[2] == seed for k in existing))
                       for seed in sorted(set(job.initial_seeds) | {key[2] for key in existing})}
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = _utc_now()
        manifest["completed_records"] = len(existing)
        manifest["seed_counts"] = seed_counts
        atomic_json(manifest_path, manifest)

        print("\nCompleted")
        print(f"Final rows:       {len(final):,}")
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
    p.add_argument("--stage", choices=sorted(STAGE_FILES), default=None,
                   help="Select config/experiments/rq4_1/{cross_project,within_corpus}.yaml. Alternative to --experiment.")
    p.add_argument("--experiment", default=None, help="Experiment YAML path. Alternative to --stage.")
    p.add_argument("--repo-root", default=str(default_root))
    p.add_argument("--output-dir", default=None, help="Base output directory. Default: <repo>/runs")
    p.add_argument("--model", action="append", default=[], choices=sorted(set(MODEL_KEY_BY_CONDITION.values())),
                   help="Restrict to one or more of svm/bilstm/codebert. Repeatable.")
    p.add_argument("--fold", action="append", type=int, default=[], help="Restrict to specific fold(s). Repeatable.")
    p.add_argument("--limit", type=int, default=None,
                   help="Smoke test: first N test commits and a small class-balanced training subset; use a separate --output-dir.")
    p.add_argument("--resume", action="store_true", help="Continue an existing matching run without repeating completed (commit, seed) records.")
    p.add_argument("--dry-run", action="store_true", help="Resolve and validate jobs/data/training config but do not train.")
    p.add_argument("--list-jobs", action="store_true", help="Print expanded jobs and exit.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not repo_root.exists():
        raise FileNotFoundError(repo_root)
    sys.path.insert(0, str(repo_root))

    from src.config import load_evaluation_config, load_experiment

    if bool(args.stage) == bool(args.experiment):
        raise ConfigError(f"Pass exactly one of --stage {{{','.join(sorted(STAGE_FILES))}}} or --experiment <path>")
    experiment_path = (repo_root / STAGE_FILES[args.stage]) if args.stage else _resolve_repo_path(repo_root, args.experiment)
    evaluation_path = repo_root / "config/evaluation.yaml"

    experiment = load_experiment(experiment_path)
    evaluation = load_evaluation_config(evaluation_path)
    jobs = expand_jobs(experiment, evaluation=evaluation)

    if args.model:
        wanted = set(args.model)
        jobs = [j for j in jobs if j.model_key in wanted]
    if args.fold:
        wanted_folds = set(args.fold)
        jobs = [j for j in jobs if j.fold in wanted_folds]
    if not jobs:
        raise RuntimeError("No trainable-baseline jobs matched the experiment/filters.")

    print(f"Expanded {len(jobs)} baseline job(s) from {experiment_path}")
    for i, job in enumerate(jobs, 1):
        _print_job(job, i)
    if args.list_jobs:
        return 0

    output_base = _resolve_repo_path(repo_root, args.output_dir) if args.output_dir else (repo_root / "runs")

    for job in jobs:
        _run_job(
            repo_root=repo_root, job=job, output_base=output_base,
            experiment_path=experiment_path, evaluation_path=evaluation_path,
            resume=args.resume, dry_run=args.dry_run, limit=args.limit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
