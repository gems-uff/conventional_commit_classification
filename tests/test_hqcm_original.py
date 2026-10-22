"""Original HQCM test set (RQ4.2): input format, materialization, comparison."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import load_canonical_dataset
from src.representations import add_input_text, build_representation
from scripts.rq4_2 import build_hqcm_original_comparison as comparison

ROOT = Path(__file__).resolve().parents[1]
HQCM_TYPES = ["feat", "fix", "docs", "style", "refactor", "test", "build", "cicd"]


def test_summary_and_diff_representation():
    row = {"processed_message": "S", "diff": "D"}
    assert build_representation(row, "summary_and_diff") == "Commit summary:\nS\n\nGit diff:\nD"


def test_materialized_test_set_reproduces_the_original_runner_input(tmp_path):
    items = [{"type": HQCM_TYPES[i % 8], "change": f" diff {i} ", "summaries": {"en": f" summary {i} "}, "subtype": ""}
             for i in range(518)]
    test_json, output = tmp_path / "test.json", tmp_path / "hqcm.csv"
    test_json.write_text(json.dumps(items), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts/rq4_2/materialize_hqcm_original_testset.py"),
                    "--test-json", str(test_json), "--output", str(output)], check=True, capture_output=True)

    df = add_input_text(load_canonical_dataset(output), "summary_and_diff")
    assert df["commit_hash"].tolist()[:2] == ["hqcm_test_00000", "hqcm_test_00001"]
    assert df.loc[7, "target"] == "ci"  # cicd -> ci
    # Same text as the original runner: f"Commit summary:\n{summary}\n\nGit diff:\n{diff}" (stripped)
    assert df.loc[3, "input_text"] == "Commit summary:\nsummary 3\n\nGit diff:\ndiff 3"


def test_comparison_tables_from_published_predictions():
    df = comparison.validate(comparison.from_results())
    tables = comparison.run_statistics(df, reps=20, seed=1)
    overall = tables["overall_metrics.csv"][0]
    published = pd.read_csv(comparison.RESULTS_DIR / "metrics/overall_metrics.csv")
    # Point estimates do not depend on the number of repetitions.
    cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "invalid_predictions"]
    pd.testing.assert_frame_equal(overall[cols], published[cols], check_dtype=False)
    assert tables["disagreement_summary.csv"][0].iloc[0].tolist() == [518, 345, 173, 286, 34, 120, 78]


def test_comparison_rejects_target_mismatch():
    df = comparison.from_results()
    df.loc[0, "target_gpt"] = "docs" if df.loc[0, "target"] != "docs" else "fix"
    with pytest.raises(SystemExit, match="targets differ"):
        comparison.validate(df)
