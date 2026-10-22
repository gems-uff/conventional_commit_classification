#!/usr/bin/env python3
"""GPT-OSS vs HQCM on the original HQCM test set (518 commits, 8 classes).

Writes the files of results/rq4_2/related_work_artifacts/hqcm_original_dataset/:

  predictions/predictions.gz            (--from runs only)
  metrics/overall_metrics.csv           accuracy, macro P/R/F1, bootstrap CI of Macro-F1
  metrics/per_class_comparison.csv      per-class P/R/F1, delta F1 with bootstrap CI,
                                        paired randomization p, Holm-adjusted p
  metrics/disagreement_summary.csv      paired correctness counts
  metrics/confusion_matrix_gpt_oss.csv  GPT-OSS confusion matrix

The test set is one paired sample (no projects). Bootstrap: stratified by true
class; randomization: per-commit swap of the two predictions; both with
--reps repetitions and --seed.

Usage
-----
# from the predictions in results/, comparing with results/
python scripts/rq4_2/build_hqcm_original_comparison.py --from results --output-dir runs/hqcm_original --check

# from a new run
python scripts/rq4_2/build_hqcm_original_comparison.py --from runs \\
  --gpt-predictions runs/rq4_hqcm_original/gpt_oss_20b/gpt_oss_20b/predictions.csv.gz \\
  --hqcm-results data/external/hqcm/codefuse-hqcm/outputs/chcl_results_len1024.json \\
  --output-dir runs/hqcm_original --check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results/rq4_2/related_work_artifacts/hqcm_original_dataset"
EXPECTED_COMMITS = 518
LABELS = ("feat", "fix", "docs", "style", "refactor", "test", "build", "ci")
LABEL_SET = set(LABELS)
INVALID_OUTPUT = "<invalid_output>"
METRIC_FILES = ("overall_metrics.csv", "per_class_comparison.csv", "disagreement_summary.csv",
                "confusion_matrix_gpt_oss.csv")


def norm_label(value, *, target: bool = False) -> str:
    s = "" if value is None else str(value).strip().casefold().strip("`'\" \t\r\n.,;:")
    s = {"cicd": "ci", "ci/cd": "ci", "ci-cd": "ci", "ci cd": "ci"}.get(s, s)
    if s in LABEL_SET:
        return s
    if target:
        raise SystemExit(f"ERROR: target outside the HQCM 8-class taxonomy: {value!r}")
    return INVALID_OUTPUT


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def from_results() -> pd.DataFrame:
    p = pd.read_csv(RESULTS_DIR / "predictions/predictions.gz", keep_default_na=False)
    h = p[p["model"] == "hqcm"][["commit_hash", "target", "prediction"]].rename(columns={"prediction": "hqcm_prediction"})
    g = p[p["model"] == "our_gpt_model"][["commit_hash", "target", "prediction"]].rename(
        columns={"prediction": "gpt_prediction", "target": "target_gpt"})
    return h.merge(g, on="commit_hash", how="inner")


def from_runs(gpt_predictions: Path, hqcm_results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """HQCM JSON row i is commit hqcm_test_<i> (hqcm.gen keeps the test-split order)."""
    payload = json.loads(hqcm_results.read_text(encoding="utf-8"))
    hqcm = pd.DataFrame([{
        "commit_hash": f"hqcm_test_{i:05d}",
        "target": norm_label(item.get("expected_answer"), target=True),
        "hqcm_prediction": norm_label(item.get("actual_answer")),
    } for i, item in enumerate(payload)])
    gpt_rows = pd.read_csv(gpt_predictions, keep_default_na=False, dtype=str)
    gpt = gpt_rows[["commit_hash", "target", "prediction"]].rename(
        columns={"prediction": "gpt_prediction", "target": "target_gpt"})
    paired = hqcm.merge(gpt, on="commit_hash", how="inner")

    seeds = sorted((c for c in gpt_rows.columns if c.startswith("seed_")), key=lambda c: int(c.split("_")[1]))
    predictions = pd.concat([
        hqcm.rename(columns={"hqcm_prediction": "prediction"}).assign(model="hqcm"),
        gpt_rows.assign(model="our_gpt_model")[["model", "commit_hash", "target", "prediction", *seeds]],
    ], ignore_index=True).reindex(columns=["model", "commit_hash", "target", "prediction", *seeds]).fillna("")
    return paired, predictions


def validate(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) != EXPECTED_COMMITS:
        raise SystemExit(f"ERROR: {len(df)} paired commits, expected {EXPECTED_COMMITS}")
    if (df["target"] != df["target_gpt"]).any():
        raise SystemExit("ERROR: HQCM and GPT-OSS targets differ for the same commit")
    if set(df["target"]) != LABEL_SET:
        raise SystemExit(f"ERROR: expected the eight classes, found {sorted(set(df['target']))}")
    df = df.drop(columns=["target_gpt"])
    df["hqcm_correct"] = df["hqcm_prediction"] == df["target"]
    df["gpt_correct"] = df["gpt_prediction"] == df["target"]
    return df.sort_values("commit_hash").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def metric_vector(y_true, y_pred):
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(LABELS), average=None, zero_division=0)
    return (float(accuracy_score(y_true, y_pred)), float(np.mean(p)), float(np.mean(r)), float(np.mean(f1)),
            p.astype(float), r.astype(float), f1.astype(float), support.astype(int))


def macro_f1_and_class_f1(y_true, y_pred):
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(LABELS), average=None, zero_division=0)
    return float(np.mean(f1)), f1.astype(float)


def stratified_bootstrap_indices(y_true, rng):
    return np.concatenate([rng.choice(np.flatnonzero(y_true == label), size=int(np.sum(y_true == label)), replace=True)
                           for label in LABELS])


def percentile_ci(values, alpha=0.05):
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def holm_adjust(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def run_statistics(df: pd.DataFrame, reps: int, seed: int):
    y = df["target"].to_numpy(dtype=str)
    pred_h = df["hqcm_prediction"].to_numpy(dtype=str)
    pred_g = df["gpt_prediction"].to_numpy(dtype=str)
    h_metrics, g_metrics = metric_vector(y, pred_h), metric_vector(y, pred_g)
    observed_delta_class_f1 = g_metrics[6] - h_metrics[6]
    rng = np.random.default_rng(seed)

    boot_h, boot_g = np.empty(reps), np.empty(reps)
    boot_delta_class = np.empty((reps, len(LABELS)))
    for i in range(reps):
        idx = stratified_bootstrap_indices(y, rng)
        hm, hf = macro_f1_and_class_f1(y[idx], pred_h[idx])
        gm, gf = macro_f1_and_class_f1(y[idx], pred_g[idx])
        boot_h[i], boot_g[i] = hm, gm
        boot_delta_class[i, :] = gf - hf

    perm_delta_class = np.empty((reps, len(LABELS)))
    for i in range(reps):
        swap = rng.random(len(y)) < 0.5
        h_perm, g_perm = pred_h.copy(), pred_g.copy()
        h_perm[swap], g_perm[swap] = pred_g[swap], pred_h[swap]
        _, hf = macro_f1_and_class_f1(y, h_perm)
        _, gf = macro_f1_and_class_f1(y, g_perm)
        perm_delta_class[i, :] = gf - hf

    class_p = [float((1 + np.sum(np.abs(perm_delta_class[:, j]) >= abs(observed_delta_class_f1[j]))) / (reps + 1))
               for j in range(len(LABELS))]
    class_holm = holm_adjust(class_p)

    overall = pd.DataFrame([{
        "model": key, "display_name": display, "n_commits": len(df), "n_classes": len(LABELS),
        "accuracy": m[0], "macro_precision": m[1], "macro_recall": m[2], "macro_f1": m[3],
        "macro_f1_bootstrap_ci95_low": ci[0], "macro_f1_bootstrap_ci95_high": ci[1],
        "invalid_predictions": int((df[col] == INVALID_OUTPUT).sum()),
    } for key, display, m, ci, col in [
        ("hqcm_reproduction", "HQCM reproduced model", h_metrics, percentile_ci(boot_h), "hqcm_prediction"),
        ("gpt_oss_20b", "GPT-OSS 20B", g_metrics, percentile_ci(boot_g), "gpt_prediction"),
    ]])

    per_class = pd.DataFrame([{
        "class": label, "support": int(h_metrics[7][j]),
        "hqcm_precision": h_metrics[4][j], "hqcm_recall": h_metrics[5][j], "hqcm_f1": h_metrics[6][j],
        "gpt_precision": g_metrics[4][j], "gpt_recall": g_metrics[5][j], "gpt_f1": g_metrics[6][j],
        "delta_f1_gpt_minus_hqcm": observed_delta_class_f1[j],
        "delta_f1_bootstrap_ci95_low": percentile_ci(boot_delta_class[:, j])[0],
        "delta_f1_bootstrap_ci95_high": percentile_ci(boot_delta_class[:, j])[1],
        "paired_randomization_p_two_sided": class_p[j], "holm_adjusted_p": class_holm[j],
    } for j, label in enumerate(LABELS)])

    h_ok, g_ok = df["hqcm_correct"].to_numpy(bool), df["gpt_correct"].to_numpy(bool)
    disagreement = pd.DataFrame([{
        "n_commits": len(df),
        "same_prediction": int((pred_h == pred_g).sum()), "different_prediction": int((pred_h != pred_g).sum()),
        "both_correct": int((h_ok & g_ok).sum()), "gpt_correct_hqcm_wrong": int((g_ok & ~h_ok).sum()),
        "hqcm_correct_gpt_wrong": int((h_ok & ~g_ok).sum()), "both_wrong": int((~h_ok & ~g_ok).sum()),
    }])

    cm_labels = list(LABELS) + [INVALID_OUTPUT]
    confusion = pd.DataFrame(confusion_matrix(df["target"], df["gpt_prediction"], labels=cm_labels),
                             index=[f"true_{x}" for x in cm_labels], columns=[f"pred_{x}" for x in cm_labels])
    return {"overall_metrics.csv": (overall, False), "per_class_comparison.csv": (per_class, False),
            "disagreement_summary.csv": (disagreement, False), "confusion_matrix_gpt_oss.csv": (confusion, True)}


# ---------------------------------------------------------------------------
# Output and check
# ---------------------------------------------------------------------------

def check(name: str, built_path: Path, index: bool) -> bool:
    ref = pd.read_csv(RESULTS_DIR / "metrics" / name, index_col=0 if index else None)
    built = pd.read_csv(built_path, index_col=0 if index else None)
    if list(ref.columns) != list(built.columns) or list(ref.index) != list(built.index):
        print(f"  DIFF {name}: columns or rows differ")
        return False
    numeric = [c for c in ref.columns if pd.api.types.is_numeric_dtype(ref[c])]
    other = [c for c in ref.columns if c not in numeric]
    diff = np.abs(ref[numeric].to_numpy(float) - built[numeric].to_numpy(float))
    ok = (not other or ref[other].equals(built[other])) and float(np.nanmax(diff, initial=0)) <= 1e-9
    print(f"  {'OK  ' if ok else 'DIFF'} {name}" + ("" if ok else f" (max |diff| {np.nanmax(diff):.3g})"))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="source", choices=["results", "runs"], required=True)
    ap.add_argument("--gpt-predictions", help="--from runs: predictions.csv.gz of the rq4_hqcm_original run.")
    ap.add_argument("--hqcm-results", help="--from runs: JSON written by hqcm.gen on the chcl test split.")
    ap.add_argument("--output-dir", required=True, help="Must not be inside results/.")
    ap.add_argument("--reps", type=int, default=20000, help="Bootstrap and randomization repetitions.")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--check", action="store_true", help="Compare the output with results/.")
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()
    if out.is_relative_to((REPO_ROOT / "results").resolve()):
        raise SystemExit(f"Refusing to write inside results/: {out}")
    (out / "metrics").mkdir(parents=True, exist_ok=True)

    if args.source == "results":
        paired, predictions = from_results(), None
    else:
        if not args.gpt_predictions or not args.hqcm_results:
            raise SystemExit("--from runs requires --gpt-predictions and --hqcm-results")
        paired, predictions = from_runs(Path(args.gpt_predictions), Path(args.hqcm_results))
        (out / "predictions").mkdir(parents=True, exist_ok=True)
        predictions.to_csv(out / "predictions/predictions.gz", index=False, compression="gzip")

    tables = run_statistics(validate(paired), args.reps, args.seed)
    ok = True
    for name, (frame, index) in tables.items():
        path = out / "metrics" / name
        frame.to_csv(path, index=index)
        print(f"wrote {path}")
        if args.check:
            ok &= check(name, path, index)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
