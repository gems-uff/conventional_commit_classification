#!/usr/bin/env python3
"""Assemble results/rq4_1/{cross_project,within_corpus}/metrics/evaluation_macro_f1.csv.

Combines the trainable-baselines runner's output (scripts/run_baselines_experiment.py:
SVM/BiLSTM/CodeBERT, freshly retrained with internal-validation hyperparameter/epoch
selection) with the existing frozen GPT-OSS/FLACCS/HQCM predictions already committed
under results/rq4_1/, and writes a project x approach Macro-F1 matrix in the same shape
as the corresponding frozen evaluation_macro_f1.csv.

Never overwrites anything under results/: --output must point elsewhere. Promoting a
recomputed file into results/ is a deliberate, separate step for a human to take after
reviewing it -- this script only assembles and validates.

Before computing anything, it verifies that all six approaches share the *exact same*
commit cohort (same (project, commit_hash) set) -- it refuses to combine mismatched
cohorts rather than silently producing an invalid comparison.

The within-corpus cohort was historically a point of failure here: the frozen
our_gpt_model/flaccs/hqcm rows were once pinned to a stale 1,630-commit cohort while
the trainable baselines ran on the corrected 1,632 (see TODO.md item 2). That has
since been reconciled -- results/rq4_1/within_corpus/predictions/predictions.csv.gz
already holds the 1,632-commit cherry-picked data -- so --other-models-source=frozen
(the default) now works directly for both evaluations. The --other-models-source
cross_project_audit path below is kept only for provenance: it reproduces exactly how
that reconciliation was performed (cherry-picking within-corpus test commits out of
the full cross-project audit), but it reads rq4_benchmark_audit/... raw audit CSVs
that only ever existed on the original execution machine ("griffin") and are not part
of this repository -- it cannot be exercised here.

Usage
-----
python scripts/rq4_1/build_evaluation_macro_f1.py --evaluation cross_project \
  --output runs/rq4_cross_project/evaluation_macro_f1.csv

python scripts/rq4_1/build_evaluation_macro_f1.py --evaluation within_corpus \
  --output runs/rq4_within_corpus/evaluation_macro_f1.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

# Directory name (as written by run_baselines_experiment.py, one per condition_id)
# -> model_key value stored inside that runner's own "model" column.
TRAINABLE_DIRS = {
    "tfidf_linear_svm": "svm",
    "bilstm": "bilstm",
    "codebert": "codebert",
}

# Alternative source for GPT-OSS/FLACCS/HQCM on within-corpus: these three
# approaches do not train on anything, so a commit's prediction depends only on
# (commit content, model, prompt, seed) -- never on which split/protocol it
# happens to belong to. cross_project.yaml and within_corpus.yaml configure
# gpt_oss_20b/flaccs/hqcm identically (same prompt_id, temperature, seeds,
# representation), so the already-computed cross-project consensus for a
# within-corpus test commit *is* its within-corpus prediction; no new inference
# is needed, just filtering the full 8,159-commit cross-project audit down to
# the within-corpus test cohort. Kept for provenance only -- see module
# docstring: the raw audit CSVs this reads are griffin-only, not in this repo.
CROSS_PROJECT_AUDIT_FILES = {
    "our_gpt_model": {
        "path": "rq4_benchmark_audit/gpt_oss_v2/gpt_oss_reconstructed_strict_first_to_two.csv",
        "prediction_col": "prediction",
    },
    "hqcm": {
        "path": "rq4_benchmark_audit/cross_project_hqcm_v2/reconstructed_true_first_to_two.csv",
        "prediction_col": "classification_true_first_to_two",
    },
    "flaccs": {
        "path": "rq4_benchmark_audit/cross_project_flaccs_v2/reconstructed_true_first_to_two.csv",
        "prediction_col": "classification_true_first_to_two",
    },
}

EVALUATIONS = {
    "cross_project": {
        "runs_root": "rq4_cross_project",
        "folds": list(range(1, 6)),
        "frozen_predictions": "results/rq4_1/cross_project/predictions/predictions.csv.gz",
        "frozen_other_models": ["our_gpt_model", "flaccs", "hqcm"],
        # model_key/frozen-model-name -> output column name
        "rename": {
            "svm": "tfidf_linear_svm", "bilstm": "bilstm", "codebert": "codebert",
            "our_gpt_model": "our_gpt_model", "flaccs": "flaccs", "hqcm": "hqcm",
        },
        "column_order": ["project", "tfidf_linear_svm", "codebert", "bilstm",
                          "our_gpt_model", "flaccs", "hqcm"],
    },
    "within_corpus": {
        "runs_root": "rq4_within_corpus",
        "folds": [0],
        "frozen_predictions": "results/rq4_1/within_corpus/predictions/predictions.csv.gz",
        "frozen_other_models": ["our_gpt_model", "flaccs", "hqcm"],
        "rename": {
            "svm": "TF-IDF + Linear SVM", "bilstm": "BiLSTM", "codebert": "CodeBERT",
            "our_gpt_model": "GPT-OSS 20B", "flaccs": "FLACCS", "hqcm": "CodeFuse HQCM",
        },
        "column_order": ["project", "GPT-OSS 20B", "CodeBERT", "FLACCS", "BiLSTM",
                          "TF-IDF + Linear SVM", "CodeFuse HQCM"],
    },
}


def load_new_baseline_predictions(runs_dir: Path, runs_root: str, folds: list[int]) -> pd.DataFrame:
    frames = []
    for dir_name in TRAINABLE_DIRS:
        for fold in folds:
            job_dir = runs_dir / runs_root / dir_name / f"fold_{fold}"
            pred_path = job_dir / "predictions.csv.gz"
            manifest_path = job_dir / "run_manifest.json"
            if not pred_path.exists() or not manifest_path.exists():
                raise FileNotFoundError(f"Missing run output: {job_dir}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "completed":
                raise RuntimeError(f"{manifest_path}: status={manifest.get('status')!r}, expected 'completed'")
            frames.append(pd.read_csv(pred_path, dtype=str, keep_default_na=False))
    combined = pd.concat(frames, ignore_index=True)
    dup = combined.duplicated(["model", "project", "commit_hash"])
    if dup.any():
        raise RuntimeError(f"Duplicate (model, project, commit_hash) rows in new predictions: {int(dup.sum())}")
    return combined


def load_frozen_other_models(frozen_path: Path, other_models: list[str]) -> pd.DataFrame:
    if not frozen_path.exists():
        raise FileNotFoundError(frozen_path)
    frozen = pd.read_csv(frozen_path, dtype=str, keep_default_na=False)
    frozen = frozen[frozen["model"].isin(other_models)].copy()
    missing = set(other_models) - set(frozen["model"].unique())
    if missing:
        raise RuntimeError(f"{frozen_path}: missing expected models {sorted(missing)}")
    return frozen


def load_within_corpus_test_keys(repo_root: Path) -> set[tuple[str, str]]:
    path = repo_root / "data/rq4_1/within_corpus/assignments.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["project"] = df["project"].str.strip()
    df["commit_hash"] = df["commit_hash"].str.strip().str.casefold()
    df["split"] = df["split"].str.strip().str.casefold()
    test = df[df["split"] == "test"]
    return set(zip(test["project"], test["commit_hash"]))


def load_cross_project_audit_other_models(audit_root: Path, test_keys: set[tuple[str, str]]) -> pd.DataFrame:
    """Cherry-pick the within-corpus test commits out of the full cross-project audit.

    See CROSS_PROJECT_AUDIT_FILES docstring above for why this is valid: these
    three approaches don't train, so a commit's prediction is the same
    regardless of which evaluation/protocol it is nominally evaluated under.
    """
    frames = []
    for model, spec in CROSS_PROJECT_AUDIT_FILES.items():
        path = audit_root / spec["path"]
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        required = {"project", "commit_hash", "target", spec["prediction_col"]}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
        df["project"] = df["project"].str.strip()
        df["commit_hash"] = df["commit_hash"].str.strip().str.casefold()
        df["target"] = df["target"].str.strip().str.casefold()
        df["prediction"] = df[spec["prediction_col"]].str.strip().str.casefold()
        keys = pd.Series(list(zip(df["project"], df["commit_hash"])))
        available = set(keys)
        missing_keys = test_keys - available
        if missing_keys:
            raise RuntimeError(
                f"{path}: {len(missing_keys)} within-corpus test commit(s) are absent from "
                f"the full cross-project audit -- cannot cherry-pick a complete cohort for {model!r}"
            )
        sub = df.loc[keys.isin(test_keys).values, ["project", "commit_hash", "target", "prediction"]].copy()
        sub["model"] = model
        dup = sub.duplicated(["project", "commit_hash"])
        if dup.any():
            raise RuntimeError(f"{path}: {int(dup.sum())} duplicate (project, commit_hash) rows after filtering")
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def validate_matching_cohorts(df: pd.DataFrame) -> None:
    """Refuse to combine approaches that were not evaluated on the same commits."""
    keys_by_model = {m: set(zip(g["project"], g["commit_hash"])) for m, g in df.groupby("model")}
    reference_model = next(iter(keys_by_model))
    reference_keys = keys_by_model[reference_model]
    problems = []
    for model, keys in keys_by_model.items():
        if keys == reference_keys:
            continue
        only_model = len(keys - reference_keys)
        only_reference = len(reference_keys - keys)
        problems.append(
            f"  {model}: {len(keys):,} commits vs. {reference_model}'s {len(reference_keys):,} "
            f"({only_model:,} only in {model}, {only_reference:,} only in {reference_model})"
        )
    if problems:
        raise RuntimeError(
            "Cohort mismatch across approaches -- refusing to combine into one evaluation table "
            "(these are not the same test set and cannot be compared):\n" + "\n".join(problems)
        )


def compute_macro_f1_by_project(df: pd.DataFrame) -> pd.DataFrame:
    """Macro-F1 restricted to the classes present in that project's ground truth.

    Matches main.tex (Conclusion Validity): "project-level Macro-F1 was computed
    over the classes present in each project's ground truth, so projects may
    contribute scores based on different subsets of the taxonomy. This avoids
    penalizing absent classes." Using sklearn's default (union of y_true/y_pred
    labels) would instead dilute the score with classes the model predicted but
    that never occur in that project's ground truth.
    """
    rows = []
    for (model, project), group in df.groupby(["model", "project"]):
        labels = sorted(group["target"].unique())
        score = f1_score(group["target"], group["prediction"], labels=labels, average="macro", zero_division=0)
        rows.append({"model": model, "project": project, "macro_f1": score})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evaluation", choices=sorted(EVALUATIONS), required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--runs-dir", default=None, help="Default: <repo-root>/runs")
    ap.add_argument("--other-models-source", choices=["frozen", "cross_project_audit"], default="frozen",
                     help="'frozen' reads results/rq4_1/.../predictions/predictions.csv.gz (the reconciled, "
                          "valid cohort for both --evaluation cross_project and within_corpus). "
                          "'cross_project_audit' cherry-picks the within-corpus test commits out of the full "
                          "cross-project audit CSVs instead -- kept for provenance only; those raw audit CSVs "
                          "are griffin-only and not part of this repository (requires --audit-root).")
    ap.add_argument("--audit-root", default=None,
                     help="Root containing rq4_benchmark_audit/... (e.g. ~/nrn/replication_package). "
                          "Required when --other-models-source=cross_project_audit.")
    ap.add_argument("--output", required=True, help="Output CSV path. Must not be inside results/.")
    args = ap.parse_args()

    if args.other_models_source == "cross_project_audit" and args.evaluation != "within_corpus":
        raise RuntimeError("--other-models-source=cross_project_audit only makes sense with --evaluation within_corpus "
                            "(cross_project's own test set already is the full cohort, no cherry-picking needed)")
    if args.other_models_source == "cross_project_audit" and not args.audit_root:
        raise RuntimeError("--audit-root is required when --other-models-source=cross_project_audit")

    repo_root = Path(args.repo_root).resolve()
    runs_dir = Path(args.runs_dir).resolve() if args.runs_dir else repo_root / "runs"
    output_path = Path(args.output).resolve()
    if output_path.is_relative_to((repo_root / "results").resolve()):
        raise RuntimeError(f"Refusing to write inside results/: {output_path}. Pass a different --output.")

    cfg = EVALUATIONS[args.evaluation]

    new_df = load_new_baseline_predictions(runs_dir, cfg["runs_root"], cfg["folds"])
    new_df["model"] = new_df["model"].map(cfg["rename"])
    if new_df["model"].isna().any():
        raise RuntimeError("Unmapped model key found in new baseline predictions")

    if args.other_models_source == "frozen":
        frozen_df = load_frozen_other_models(repo_root / cfg["frozen_predictions"], cfg["frozen_other_models"])
    else:
        test_keys = load_within_corpus_test_keys(repo_root)
        frozen_df = load_cross_project_audit_other_models(Path(args.audit_root).expanduser(), test_keys)
        print(f"Cherry-picked {len(frozen_df):,} rows from the cross-project audit "
              f"({len(test_keys):,} within-corpus test commits x 3 approaches)")
    frozen_df["model"] = frozen_df["model"].map(cfg["rename"])
    if frozen_df["model"].isna().any():
        raise RuntimeError("Unmapped model key found in other-models predictions")

    combined = pd.concat([new_df, frozen_df], ignore_index=True)
    print(f"Combined rows: {len(combined):,} across models: "
          f"{sorted(combined['model'].unique())}")

    validate_matching_cohorts(combined)
    print("Cohort check OK: all approaches share the exact same (project, commit_hash) set.")

    scores = compute_macro_f1_by_project(combined)
    wide = scores.pivot(index="project", columns="model", values="macro_f1").reset_index()
    missing_cols = set(cfg["column_order"]) - set(wide.columns)
    if missing_cols:
        raise RuntimeError(f"Missing expected columns in pivoted result: {sorted(missing_cols)}")
    wide = wide[cfg["column_order"]].sort_values("project").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(output_path, index=False)
    print(f"\nWrote {output_path} ({len(wide)} projects)")
    print(wide.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
