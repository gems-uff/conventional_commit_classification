#!/usr/bin/env python3
"""Build the per-project Macro-F1 tables read by the analysis notebooks.

Reads consensus predictions either from results/ (--from results) or from a
run directory written by the runners (--from runs), and writes the tables in
the format of results/<rq>/metrics/. --check compares the built tables with
the ones in results/.

Usage
-----
python scripts/build_metrics.py --rq rq1 --from runs --output-dir runs/metrics
python scripts/build_metrics.py --rq all --from results --output-dir runs/metrics --check

Tables (one --rq builds all tables of that RQ):

  rq1    results/rq1/metrics/{pre_cutoff,post_cutoff}.csv
  rq2    results/rq2/metrics/{model_macro_f1,temperature_macro_f1}.csv
  rq3_1  results/rq3_1/metrics/macro_f1.csv
  rq3_2  results/rq3_2/metrics/factor_{1,2,3,4}_macro_f1.csv
         results/rq3_2/temporal_robustness/metrics/temporal_robustness.csv
  rq4_1  results/rq4_1/{cross_project,within_corpus}/metrics/evaluation_macro_f1.csv
  rq4_2  results/rq4_2/related_work_artifacts/{flaccs,hqcm}/metrics/evaluation_macro_f1.csv

With --from runs, predictions are read from <runs-dir>/<experiment_id>/... as
written by run_ollama_experiment.py, run_baselines_experiment.py, and
run_flaccs_experiment.py. RQ4 approaches that were not run are taken from
results/ (GPT-OSS, FLACCS, HQCM, or the trainable baselines).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import macro_f1_present_classes  # noqa: E402

ABSTAINED = "<abstained>"

# Column names used in results/<rq>/metrics/, keyed by every model/prompt
# identifier that can appear in results/ predictions or in runs/ directories.
RQ1_COLUMNS = {
    "Gemma 3": ("gemma3_27b", "gemma_3_27b"),
    "GPT-OSS 20B": ("gpt-oss_20b", "gpt_oss_20b"),
    "Granite 3.3 8B": ("granite3.3_8b", "granite_3_3_8b"),
    "Llama 3.2 11B": ("llama3.2-vision_11b", "llama_3_2_vision_11b"),
    "Magistral 24B": ("magistral", "magistral_24b"),
    "Mistral Small 3.2 24B": ("mistral-small3.2_24b", "mistral_small_3_2_24b"),
    "Phi-4 RP 14B": ("phi4-reasoning_plus", "phi4_reasoning_plus_14b"),
    "QwQ 32B": ("qwq_latest", "qwq_32b"),
}
RQ2_COLUMNS = {
    "deepSeek r1 8b": ("deepseek_r1_8b", "deepseek_r1_8b_0528_qwen3"),
    "gemma 3 27b": ("gemma_3_27b",),
    "granite 3.3 8b": ("granite_3_3_8b",),
    "llama vision 3.2 11b": ("llama_3_2_vision_11b",),
    "magistral 24b": ("magistral_24b",),
    "mistral small 3.2 24b": ("mistral_small_3_2_24b",),
    "phi-4 rp 14b": ("phi_4_rp_14b", "phi4_reasoning_plus_14b"),
    "qwq 32b": ("qwq_32b",),
    "aya expanse 32b": ("aya_expanse_32b",),
    "devstral 24b": ("devstral_24b", "devstral_small_24b"),
    "gpt-oss 20b": ("gpt_oss_20b",),
    "qwen3 32b": ("qwen3_32b",),
    "qwen3 coder 30b": ("qwen3_coder_30b",),
}
RQ2_TEMPERATURES = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
RQ3_1_COLUMNS = ["message_only", "diff_only", "message_and_diff", "git_show"]
# Factor key in config/experiments/rq3_2.yaml -> (table file, factor value in the table).
RQ3_2_FACTORS = {
    "factor_1_definitions": ("factor_1_macro_f1.csv", "factor_1_definitions"),
    "factor_2_heuristics": ("factor_2_macro_f1.csv", "factor_2_heuristics"),
    "factor_3_cot_and_few_shot": ("factor_3_macro_f1.csv", "factor_3_few_shot_cot"),
    "factor_4_abstention": ("factor_4_macro_f1.csv", "factor_4_abstention"),
}
# Prompt IDs written with hyphens in the tables.
RQ3_2_PROMPT_COLUMNS = {
    "p_zero_shot_cot": "p_zero-shot_cot",
    "p_few_shot": "p_few-shot",
    "p_cot_few_shot": "p_cot_few-shot",
}
TEMPORAL_COLUMNS = ["baseline_pre_cutoff", "refined_pre_cutoff", "baseline_post_cutoff", "refined_post_cutoff"]
RQ4_1_COLUMNS = {
    "cross_project": ["svm", "codebert", "bilstm", "gpt-oss", "flaccs", "hqcm"],
    "within_corpus": ["gpt-oss", "codebert", "flaccs", "bilstm", "svm", "hqcm"],
}
RQ4_COLUMN_BY_ID = {
    "svm": "svm", "tfidf_linear_svm": "svm", "bilstm": "bilstm", "codebert": "codebert",
    "our_gpt_model": "gpt-oss", "gpt_oss_20b": "gpt-oss", "flaccs": "flaccs", "hqcm": "hqcm",
}


@dataclass
class Table:
    path: str                       # path relative to results/ and to --output-dir
    frame: pd.DataFrame
    decimals: int | None = None     # rounding used by the table in results/
    csv_kwargs: dict = field(default_factory=dict)


def _alias(columns: dict[str, tuple[str, ...]]) -> dict[str, str]:
    return {identifier: name for name, ids in columns.items() for identifier in ids}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, low_memory=False)


def _project_key(df: pd.DataFrame, with_platform: bool) -> pd.Series:
    if with_platform and "platform" in df.columns:
        return df["platform"].astype(str) + "/" + df["project"].astype(str)
    return df["project"].astype(str)


def _matrix(df: pd.DataFrame, key: str, columns: list[str], *, drop_abstained: bool = False) -> pd.DataFrame:
    """Per-project Macro-F1 (classes present in the project's ground truth), one column per key value."""
    missing = sorted(set(columns) - set(df[key]))
    if missing:
        raise RuntimeError(f"No predictions for: {missing}")
    rows = []
    for (column, project), group in df[df[key].isin(columns)].groupby([key, "project_key"]):
        if drop_abstained:
            group = group[group["prediction"] != ABSTAINED]
        rows.append({"project": project, key: column,
                     "f1": macro_f1_present_classes(group["target"], group["prediction"])})
    wide = pd.DataFrame(rows).pivot(index="project", columns=key, values="f1")
    return wide[columns].reset_index().rename_axis(columns=None)


def _runs(runs_dir: Path, experiment_id: str) -> pd.DataFrame:
    """Concatenate <runs_dir>/<experiment_id>/**/predictions.csv.gz with the job path as columns."""
    root = runs_dir / experiment_id
    frames = []
    for path in sorted(root.rglob("predictions.csv.gz")):
        parts = path.parent.relative_to(root).parts
        df = _read(path)
        df["run_condition"] = parts[0]
        df["run_model"] = parts[1] if len(parts) > 1 else parts[0]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No predictions.csv.gz under {root}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-RQ builders: (source, runs_dir) -> list[Table]
# ---------------------------------------------------------------------------

def build_rq1(source: str, runs_dir: Path) -> list[Table]:
    if source == "results":
        df = _read(REPO_ROOT / "results/rq1/predictions/predictions.csv.gz")
        df["column"] = df["model"].map(_alias(RQ1_COLUMNS))
    else:
        df = _runs(runs_dir, "rq1_temporal_alignment")
        df["partition"] = df["run_condition"]
        df["column"] = df["run_model"].map(_alias(RQ1_COLUMNS))
    df["project_key"] = _project_key(df, with_platform=True)
    columns = list(RQ1_COLUMNS)
    return [Table(f"rq1/metrics/{part}.csv", _matrix(df[df["partition"] == part], "column", columns), 6)
            for part in ("pre_cutoff", "post_cutoff")]


def build_rq2(source: str, runs_dir: Path) -> list[Table]:
    if source == "results":
        models = _read(REPO_ROOT / "results/rq2/predictions/model_predictions.csv.gz")
        models["column"] = models["model"].map(_alias(RQ2_COLUMNS))
        temps = _read(REPO_ROOT / "results/rq2/predictions/temperature_predictions.csv.gz")
        temps["column"] = "temp_" + temps["temperature"].astype(float).map("{:.1f}".format)
    else:
        df = _runs(runs_dir, "rq2_model_and_temperature")
        models = df[df["run_condition"] == "model_comparison"].copy()
        models["column"] = models["run_model"].map(_alias(RQ2_COLUMNS))
        temps = df[df["run_condition"].str.startswith("temperature_")].copy()
        temps["column"] = "temp_" + (temps["run_condition"].str.removeprefix("temperature_")
                                     .str.replace("p", ".").astype(float).map("{:.1f}".format))
    for frame in (models, temps):
        frame["project_key"] = _project_key(frame, with_platform=True)
    return [
        Table("rq2/metrics/model_macro_f1.csv", _matrix(models, "column", list(RQ2_COLUMNS)), 4),
        Table("rq2/metrics/temperature_macro_f1.csv",
              _matrix(temps, "column", [f"temp_{t}" for t in RQ2_TEMPERATURES]), 4),
    ]


def build_rq3_1(source: str, runs_dir: Path) -> list[Table]:
    if source == "results":
        df = _read(REPO_ROOT / "results/rq3_1/predictions/predictions.csv.gz")
        df["column"] = df["representation"]
    else:
        df = _runs(runs_dir, "rq3_1_input_representation")
        df["column"] = df["run_condition"]
    df["project_key"] = _project_key(df, with_platform=True)
    return [Table("rq3_1/metrics/macro_f1.csv", _matrix(df, "column", RQ3_1_COLUMNS))]


def build_rq3_2(source: str, runs_dir: Path) -> list[Table]:
    config = yaml.safe_load((REPO_ROOT / "config/experiments/rq3_2.yaml").read_text(encoding="utf-8"))
    if source == "results":
        df = _read(REPO_ROOT / "results/rq3_2/predictions/predictions.csv.gz")
        df["column"] = df["treatment"]
    else:
        try:
            df = _runs(runs_dir, "rq3_2_prompt_refinement")
            df["column"] = df["run_condition"].map(lambda p: RQ3_2_PROMPT_COLUMNS.get(p, p))
        except FileNotFoundError:
            print("  rq3_2_prompt_refinement: no runs found; factor tables skipped")
            df = pd.DataFrame(columns=["column", "project", "target", "prediction"])
    df["project_key"] = _project_key(df, with_platform=True)
    tables = []
    for factor_key, (filename, factor_value) in RQ3_2_FACTORS.items():
        if df.empty:
            break
        columns = [RQ3_2_PROMPT_COLUMNS.get(p, p) for p in config["factors"][factor_key]["treatments"]]
        wide = _matrix(df, "column", columns, drop_abstained=True)
        wide.insert(0, "factor", factor_value)
        tables.append(Table(f"rq3_2/metrics/{filename}", wide, 6))
    temporal = _temporal_frame(source, runs_dir, df)
    if temporal is not None:
        wide = _matrix(temporal, "condition", TEMPORAL_COLUMNS).rename(columns={"project": "Project"})
        tables.append(Table("rq3_2/temporal_robustness/metrics/temporal_robustness.csv", wide,
                            csv_kwargs={"sep": ";", "decimal": ",", "float_format": "%.17g"}))
    return tables


def _temporal_frame(source: str, runs_dir: Path, rq3_2: pd.DataFrame) -> pd.DataFrame | None:
    """Pre-cutoff conditions from rq3_2_temporal_robustness; post-cutoff conditions are the
    RQ3.2 runs of p0_baseline and p_code_disambiguation."""
    if source == "results":
        temporal = _read(REPO_ROOT / "results/rq3_2/temporal_robustness/predictions/predictions.csv.gz")
        temporal["project_key"] = _project_key(temporal, with_platform=False)
        return temporal
    try:
        pre = _runs(runs_dir, "rq3_2_temporal_robustness")
    except FileNotFoundError:
        print("  rq3_2_temporal_robustness: no runs found; temporal_robustness.csv skipped")
        return None
    pre["condition"] = pre["run_condition"]
    pre["project_key"] = _project_key(pre, with_platform=True)
    post = rq3_2[rq3_2["column"].isin(["p0_baseline", "p_code_disambiguation"])].copy()
    if post.empty:
        print("  rq3_2_prompt_refinement: post-cutoff conditions taken from results/rq3_2/predictions/")
        post = _read(REPO_ROOT / "results/rq3_2/predictions/predictions.csv.gz")
        post["column"] = post["treatment"]
        post = post[post["column"].isin(["p0_baseline", "p_code_disambiguation"])].copy()
        post["project_key"] = _project_key(post, with_platform=True)
    post["condition"] = post["column"].map({"p0_baseline": "baseline_post_cutoff",
                                            "p_code_disambiguation": "refined_post_cutoff"})
    return pd.concat([pre, post], ignore_index=True)


def _rq4_frame(source: str, runs_dir: Path, results_path: str, experiment_id: str) -> pd.DataFrame:
    published = _read(REPO_ROOT / results_path)
    published["column"] = published["model"].map(RQ4_COLUMN_BY_ID)
    if source == "results":
        return published
    try:
        run = _runs(runs_dir, experiment_id)
    except FileNotFoundError:
        print(f"  {experiment_id}: no runs found, all approaches taken from {results_path}")
        return published
    run["column"] = run["run_condition"].map(RQ4_COLUMN_BY_ID)
    ran = set(run["column"].dropna())
    missing = sorted(set(published["column"]) - ran)
    if missing:
        print(f"  {experiment_id}: not found in runs, taken from {results_path}: {missing}")
    return pd.concat([run, published[published["column"].isin(missing)]], ignore_index=True)


def build_rq4_1(source: str, runs_dir: Path) -> list[Table]:
    tables = []
    for evaluation, columns in RQ4_1_COLUMNS.items():
        df = _rq4_frame(source, runs_dir, f"results/rq4_1/{evaluation}/predictions/predictions.csv.gz",
                        f"rq4_{evaluation}")
        df["project_key"] = _project_key(df, with_platform=False)
        tables.append(Table(f"rq4_1/{evaluation}/metrics/evaluation_macro_f1.csv",
                            _matrix(df, "column", columns)))
    return tables


def build_rq4_2(source: str, runs_dir: Path) -> list[Table]:
    tables = []
    for artifact in ("flaccs", "hqcm"):
        results_path = f"results/rq4_2/related_work_artifacts/{artifact}/predictions/predictions.csv.gz"
        df = _rq4_frame(source, runs_dir, results_path, f"rq4_artifact_{artifact}")
        if artifact not in set(df["column"]):
            print(f"  rq4_2/{artifact}: no {artifact} predictions available; table skipped")
            continue
        df["project_key"] = _project_key(df, with_platform=False)
        wide = _matrix(df, "column", ["gpt-oss", artifact])
        wide[f"delta_gpt_minus_{artifact}"] = wide["gpt-oss"] - wide[artifact]
        tables.append(Table(f"rq4_2/related_work_artifacts/{artifact}/metrics/evaluation_macro_f1.csv", wide))
    return tables


BUILDERS: dict[str, Callable[[str, Path], list[Table]]] = {
    "rq1": build_rq1, "rq2": build_rq2, "rq3_1": build_rq3_1,
    "rq3_2": build_rq3_2, "rq4_1": build_rq4_1, "rq4_2": build_rq4_2,
}


def write(table: Table, output_dir: Path) -> Path:
    path = output_dir / table.path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = table.frame.round(table.decimals) if table.decimals is not None else table.frame
    frame.to_csv(path, index=False, **table.csv_kwargs)
    return path


def check(table: Table) -> bool:
    reference_path = REPO_ROOT / "results" / table.path
    read_kwargs = {k: v for k, v in table.csv_kwargs.items() if k in ("sep", "decimal")}
    reference = pd.read_csv(reference_path, **read_kwargs)
    built = table.frame
    key = built.columns[0] if built.columns[0] != "factor" else "project"
    if list(reference.columns) != list(built.columns):
        print(f"  DIFF {table.path}: columns {list(built.columns)} != {list(reference.columns)}")
        return False
    reference = reference.set_index(key).sort_index()
    built = built.set_index(key).sort_index()
    if list(reference.index) != list(built.index):
        print(f"  DIFF {table.path}: projects differ")
        return False
    numeric = [c for c in built.columns if pd.api.types.is_numeric_dtype(built[c])]
    tolerance = 0.5 * 10 ** -table.decimals + 1e-12 if table.decimals is not None else 1e-9
    diff = (built[numeric] - reference[numeric]).abs().stack()
    bad = diff[diff > tolerance]
    if bad.empty:
        print(f"  OK   {table.path}")
        return True
    print(f"  DIFF {table.path}: {len(bad)} cell(s) differ")
    for (project, column), value in bad.items():
        print(f"         {project} / {column}: built={built.at[project, column]:.6f} "
              f"results={reference.at[project, column]:.6f} (|diff|={value:.6f})")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq", choices=[*BUILDERS, "all"], required=True)
    ap.add_argument("--from", dest="source", choices=["results", "runs"], required=True,
                    help="Read predictions from results/ or from a runs directory.")
    ap.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"), help="Default: <repo>/runs")
    ap.add_argument("--output-dir", required=True, help="Output directory. Must not be inside results/.")
    ap.add_argument("--check", action="store_true", help="Compare the built tables with results/.")
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if output_dir.is_relative_to((REPO_ROOT / "results").resolve()):
        raise SystemExit(f"Refusing to write inside results/: {output_dir}")
    rqs = list(BUILDERS) if args.rq == "all" else [args.rq]

    all_ok = True
    for rq in rqs:
        print(f"{rq}:")
        for table in BUILDERS[rq](args.source, Path(args.runs_dir).resolve()):
            print(f"  wrote {write(table, output_dir)}")
            if args.check:
                all_ok &= check(table)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
