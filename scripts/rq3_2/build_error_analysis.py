#!/usr/bin/env python3
"""Class-confusion analysis of one RQ3.2 treatment (optional).

Writes the files of results/rq3_2/error_analysis/ from consensus predictions:

  confusion_counts.csv            target x prediction counts
  confusion_row_normalized.csv    the same, divided by the row total
  top_systematic_confusions.csv   target -> prediction pairs with at least
                                  --min-count errors in at least
                                  --min-project-coverage of the projects
  top_error_examples.csv          up to 5 commits per pair, from different projects
  report.md                       summary

Usage
-----
# from the predictions in results/, comparing with results/rq3_2/error_analysis/
python scripts/rq3_2/build_error_analysis.py --output-dir runs/error_analysis --check

# from a new run
python scripts/rq3_2/build_error_analysis.py \\
  --predictions runs/rq3_2_prompt_refinement/p_header_structured/gpt_oss_20b/predictions.csv.gz \\
  --output-dir runs/error_analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.consensus import INVALID_OUTPUT  # noqa: E402
from src.data.loader import STUDY_LABELS  # noqa: E402

DEFAULT_PREDICTIONS = REPO_ROOT / "results/rq3_2/predictions/predictions.csv.gz"
REFERENCE_DIR = REPO_ROOT / "results/rq3_2/error_analysis"
LABELS = list(STUDY_LABELS)
EXAMPLES_PER_PAIR = 5
REPORT_TOP_PAIRS = 15
FILES = ["confusion_counts.csv", "confusion_row_normalized.csv", "top_systematic_confusions.csv",
         "top_error_examples.csv", "report.md"]


def load(path: Path, treatment: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, low_memory=False)
    if "treatment" in df.columns:
        df = df[df["treatment"] == treatment]
    if "platform" in df.columns:
        df = df.assign(project=df["platform"].astype(str) + "/" + df["project"].astype(str))
    if df.empty:
        raise SystemExit(f"No predictions for {treatment} in {path}")
    return df[["project", "commit_hash", "target", "prediction"]].reset_index(drop=True)


def confusion(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = LABELS + [INVALID_OUTPUT]
    counts = pd.crosstab(df["target"], df["prediction"]).reindex(index=LABELS, columns=columns, fill_value=0)
    normalized = counts.div(counts.sum(axis=1), axis=0)
    return counts.rename_axis("target").rename_axis(None, axis=1), normalized.rename_axis("target").rename_axis(None, axis=1)


def directional_pairs(df: pd.DataFrame) -> pd.DataFrame:
    n_projects = df["project"].nunique()
    errors = df[df["target"] != df["prediction"]]
    pairs = (errors.groupby(["target", "prediction"])
             .agg(count=("commit_hash", "size"), projects_with_confusion=("project", "nunique"))
             .reset_index())
    pairs = pairs[pairs["prediction"].isin(LABELS)].copy()
    support = df.groupby("target").size()
    target_errors = errors.groupby("target").size()
    reverse = pairs.set_index(["target", "prediction"])["count"]
    pairs["target_support"] = pairs["target"].map(support)
    pairs["target_error_count"] = pairs["target"].map(target_errors)
    pairs["target_rate"] = pairs["count"] / pairs["target_support"]
    pairs["share_of_target_errors"] = pairs["count"] / pairs["target_error_count"]
    pairs["project_coverage"] = pairs["projects_with_confusion"] / n_projects
    pairs["prediction_kind"] = "semantic"
    pairs["reverse_count"] = [int(reverse.get((p, t), 0)) for t, p in zip(pairs["target"], pairs["prediction"])]
    pairs["pair_total_bidirectional"] = pairs["count"] + pairs["reverse_count"]
    pairs["direction_share"] = pairs["count"] / pairs["pair_total_bidirectional"]
    pairs["net_direction_count"] = pairs["count"] - pairs["reverse_count"]
    pairs["smoothed_direction_ratio"] = (pairs["count"] + 0.5) / (pairs["reverse_count"] + 0.5)
    return pairs.sort_values(["count", "projects_with_confusion", "target", "prediction"],
                             ascending=[False, False, True, True]).reset_index(drop=True)


def examples(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, prediction in zip(pairs["target"], pairs["prediction"]):
        hits = df[(df["target"] == target) & (df["prediction"] == prediction)]
        hits = hits.sort_values(["project", "commit_hash"]).drop_duplicates("project")
        rows.append(hits.head(EXAMPLES_PER_PAIR))
    return pd.concat(rows, ignore_index=True)


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def report(df: pd.DataFrame, pairs: pd.DataFrame, systematic: pd.DataFrame, treatment: str) -> str:
    n = len(df)
    correct = int((df["target"] == df["prediction"]).sum())
    lines = [
        f"# RQ3.2 Systematic Error Pattern Analysis — `{treatment}`",
        "",
        "This is a descriptive analysis of the **final strict consensus predictions** for the structured prompt "
        "condition. It does not reparse model outputs and does not alter the RQ3.2 consensus.",
        "",
        "## Scope",
        "",
        f"- Commits analyzed: **{n:,}**",
        f"- Projects represented: **{df['project'].nunique()}**",
        f"- Correct predictions: **{correct:,}**",
        f"- Errors: **{n - correct:,}** ({pct((n - correct) / n)})",
        "",
        "## Top semantic directional confusions",
        "",
        "| target | prediction | count | target rate | share of target errors | projects | project coverage "
        "| reverse count | direction share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in pairs.head(REPORT_TOP_PAIRS).itertuples(index=False):
        lines.append(f"| {r.target} | {r.prediction} | {r.count} | {pct(r.target_rate)} | "
                     f"{pct(r.share_of_target_errors)} | {r.projects_with_confusion} | {pct(r.project_coverage)} | "
                     f"{r.reverse_count} | {pct(r.direction_share)} |")
    invalid = df[(df["prediction"] == INVALID_OUTPUT)]
    lines += [
        "",
        "## Systematic-confusion filter",
        "",
        "`top_systematic_confusions.csv` contains semantic directional pairs that satisfy both the configured "
        "minimum occurrence count and minimum project coverage thresholds.",
        "",
        f"Pairs satisfying the thresholds: **{len(systematic)}**.",
        "",
        "## Non-semantic outcomes",
        "",
        "| prediction | count | projects |",
        "|---|---:|---:|",
        f"| {INVALID_OUTPUT} | {len(invalid)} | {invalid['project'].nunique()} |",
        "",
        "## Column definitions",
        "",
        "- `target_rate` = occurrences of target→prediction divided by all commits whose ground-truth target is "
        "that target class.",
        "- `share_of_target_errors` = occurrences of target→prediction divided by all errors made on that target "
        "class.",
        "- `project_coverage` = number of projects exhibiting the directional confusion divided by all projects "
        "represented in the structured-prompt cohort.",
        "- `reverse_count` compares A→B with B→A. `direction_share` is A→B / (A→B + B→A), so values near 1 "
        "indicate a strongly directional confusion.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS),
                    help="Consensus predictions (default: results/rq3_2/predictions/predictions.csv.gz).")
    ap.add_argument("--treatment", default="p_header_structured")
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--min-project-coverage", type=float, default=0.2)
    ap.add_argument("--output-dir", required=True, help="Must not be inside results/.")
    ap.add_argument("--check", action="store_true", help="Compare the output with results/rq3_2/error_analysis/.")
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()
    if out.is_relative_to((REPO_ROOT / "results").resolve()):
        raise SystemExit(f"Refusing to write inside results/: {out}")
    out.mkdir(parents=True, exist_ok=True)

    df = load(Path(args.predictions), args.treatment)
    counts, normalized = confusion(df)
    pairs = directional_pairs(df)
    systematic = pairs[(pairs["count"] >= args.min_count)
                       & (pairs["project_coverage"] >= args.min_project_coverage - 1e-12)]

    counts.to_csv(out / "confusion_counts.csv")
    normalized.to_csv(out / "confusion_row_normalized.csv", float_format="%.2f")
    systematic.to_csv(out / "top_systematic_confusions.csv", index=False, float_format="%.2f")
    examples(df, systematic).to_csv(out / "top_error_examples.csv", index=False)
    (out / "report.md").write_text(report(df, pairs, systematic, args.treatment), encoding="utf-8")
    print(f"Wrote {len(FILES)} files to {out}")

    if not args.check:
        return 0
    ok = True
    for name in FILES:
        built = (out / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        reference = (REFERENCE_DIR / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        same = built.strip() == reference.strip()
        ok &= same
        print(f"  {'OK  ' if same else 'DIFF'} {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
