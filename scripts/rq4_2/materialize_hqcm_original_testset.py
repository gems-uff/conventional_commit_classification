#!/usr/bin/env python3
"""Convert the original HQCM test set into the canonical dataset format.

Reads dataset/test.json of the HQCM repository (see data/external/hqcm/README.md)
and writes a CSV that config/experiments/rq4_2/hqcm_original.yaml uses as its
dataset. Each item becomes one row:

  commit_hash        hqcm_test_<index>, index = position in test.json
  target             item["type"] (cicd -> ci)
  processed_message  item["summaries"]["en"]
  diff               item["change"]

Items with a type outside the eight HQCM classes or an empty change are skipped.

Usage
-----
python scripts/rq4_2/materialize_hqcm_original_testset.py \\
  --test-json data/external/hqcm/codefuse-hqcm/dataset/test.json \\
  --output data/external/hqcm/materialized/hqcm_original_test.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
HQCM_TYPES = {"fix", "feat", "build", "cicd", "docs", "style", "refactor", "test"}
EXPECTED_ROWS = 518
COLUMNS = ("partition", "platform", "project", "commit_hash", "commit_date", "target", "original_type",
           "breaking_change", "original_message", "processed_message", "has_diff", "diff", "git_show")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-json", default=str(REPO_ROOT / "data/external/hqcm/codefuse-hqcm/dataset/test.json"))
    ap.add_argument("--output", default=str(REPO_ROOT / "data/external/hqcm/materialized/hqcm_original_test.csv"))
    args = ap.parse_args()

    items = json.loads(Path(args.test_json).read_text(encoding="utf-8"))
    rows = []
    for index, item in enumerate(items):
        hqcm_type = str(item.get("type", "")).strip().lower()
        diff = str(item.get("change", "")).strip()
        summary = str((item.get("summaries") or {}).get("en", "")).strip()
        if hqcm_type not in HQCM_TYPES or not diff:
            continue
        rows.append({
            "partition": "hqcm_original", "platform": "hqcm", "project": "hqcm",
            "commit_hash": f"hqcm_test_{index:05d}", "commit_date": "",
            "target": "ci" if hqcm_type == "cicd" else hqcm_type, "original_type": hqcm_type,
            "breaking_change": False, "original_message": summary, "processed_message": summary,
            "has_diff": True, "diff": diff, "git_show": "",
        })
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(f"ERROR: {len(rows)} items kept, expected {EXPECTED_ROWS}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(output, index=False)
    print(f"Wrote {output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
