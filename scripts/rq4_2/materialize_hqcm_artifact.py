#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


REQUIRED_DATASET_COLUMNS = {
    "partition",
    "platform",
    "project",
    "commit_hash",
    "commit_date",
    "target",
    "breaking_change",
    "original_message",
    "processed_message",
    "diff",
}

REQUIRED_ASSIGNMENT_COLUMNS = {
    "protocol",
    "project",
    "commit_hash",
    "canonical_target",
    "protocol_target",
    "source_index",
    "protocol_record_id",
    "message_tokens",
    "diff_tokens",
    "total_tokens_original",
    "prompt_tokens",
    "total_tokens",
}

OUTPUT_COLUMNS = [
    "index",
    "hash",
    "input_text",
    "target",
    "breaking_change",
    "project",
    "repository",
    "commit_hash_original",
    "commit_date",
    "original_label",
    "hqcm_label",
    "commit_message",
    "git_diff",
    "message_tokens",
    "diff_tokens",
    "total_tokens_original",
    "prompt_tokens_llama",
    "total_tokens_llama",
]


def normalize_string(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def normalize_hash(series: pd.Series) -> pd.Series:
    return normalize_string(series).str.lower()


def to_int(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="raise")
    if values.isna().any():
        raise ValueError(f"Missing numeric value in {name}.")
    if (values % 1 != 0).any():
        raise ValueError(f"Non-integer value found in {name}.")
    return values.astype(int)


def normalize_bool_value(value) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False

    raise ValueError(f"Invalid boolean value: {value!r}")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Canonical dataset not found: {path}")

    df = pd.read_csv(
        path,
        compression="infer",
        keep_default_na=False,
        low_memory=False,
        dtype={
            "partition": str,
            "platform": str,
            "project": str,
            "commit_hash": str,
            "target": str,
        },
    )

    missing = REQUIRED_DATASET_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Canonical dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    for column in ("partition", "platform", "project", "target"):
        df[column] = normalize_string(df[column])

    df["commit_hash"] = normalize_hash(df["commit_hash"])

    if set(df["partition"]) != {"pre_cutoff"}:
        raise RuntimeError(
            "Expected the canonical pre-cutoff dataset only."
        )

    duplicates = df.duplicated(
        subset=["project", "commit_hash"],
        keep=False,
    )

    if duplicates.any():
        sample = df.loc[
            duplicates,
            ["project", "commit_hash"],
        ].head(20)
        raise RuntimeError(
            "Duplicate commits in canonical dataset:\n"
            f"{sample.to_string(index=False)}"
        )

    return df


def load_assignments(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"HQCM assignments not found: {path}")

    df = pd.read_csv(
        path,
        keep_default_na=False,
        low_memory=False,
        dtype={
            "protocol": str,
            "project": str,
            "commit_hash": str,
            "canonical_target": str,
            "protocol_target": str,
            "protocol_record_id": str,
        },
    )

    missing = REQUIRED_ASSIGNMENT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "HQCM assignments are missing required columns: "
            f"{sorted(missing)}"
        )

    for column in (
        "protocol",
        "project",
        "canonical_target",
        "protocol_target",
        "protocol_record_id",
    ):
        df[column] = normalize_string(df[column])

    df["commit_hash"] = normalize_hash(df["commit_hash"])

    for column in (
        "source_index",
        "message_tokens",
        "diff_tokens",
        "total_tokens_original",
        "prompt_tokens",
        "total_tokens",
    ):
        df[column] = to_int(df[column], column)

    protocols = set(df["protocol"])
    if protocols != {"hqcm_compatible"}:
        raise RuntimeError(
            "Expected protocol='hqcm_compatible', "
            f"found {sorted(protocols)}."
        )

    duplicates = df.duplicated(
        subset=["project", "commit_hash"],
        keep=False,
    )
    if duplicates.any():
        sample = df.loc[
            duplicates,
            ["project", "commit_hash"],
        ].head(20)
        raise RuntimeError(
            "Duplicate commits in HQCM assignments:\n"
            f"{sample.to_string(index=False)}"
        )

    if df["protocol_record_id"].duplicated().any():
        raise RuntimeError("Duplicate protocol_record_id values.")

    if df["source_index"].duplicated().any():
        raise RuntimeError("Duplicate source_index values.")

    ordered_indexes = sorted(df["source_index"].tolist())
    expected_indexes = list(range(len(df)))
    if ordered_indexes != expected_indexes:
        raise RuntimeError(
            "source_index is expected to be contiguous from 0 "
            f"to {len(df) - 1}."
        )

    over_limit = df[df["total_tokens"] > 1024]
    if not over_limit.empty:
        raise RuntimeError(
            "HQCM assignments contain records above the 1024-token "
            "artifact limit."
        )

    return df.sort_values(
        "source_index",
        kind="stable",
    ).reset_index(drop=True)


def build_input_text(commit_message: str, git_diff: str) -> str:
    """
    Build the HQCM-compatible input text.
    """
    return (
        f"Commit summary:\n{commit_message}"
        f"\n\nGit diff:\n{git_diff}"
    )


def materialize(
    dataset: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    selected = assignments.merge(
        dataset,
        on=["project", "commit_hash"],
        how="left",
        validate="one_to_one",
        suffixes=("_assignment", "_dataset"),
        indicator=True,
    )

    missing = selected[selected["_merge"] != "both"]
    if not missing.empty:
        raise RuntimeError(
            "HQCM assignments reference commits absent from the "
            "canonical dataset:\n"
            f"{missing[['project', 'commit_hash']].head(20).to_string(index=False)}"
        )

    canonical_mismatch = selected[
        selected["canonical_target"]
        != selected["target"]
    ]
    if not canonical_mismatch.empty:
        raise RuntimeError(
            "canonical_target does not match the canonical dataset:\n"
            f"{canonical_mismatch[['project', 'commit_hash', 'canonical_target', 'target']].head(20).to_string(index=False)}"
        )

    records = []

    for _, row in selected.sort_values(
        "source_index",
        kind="stable",
    ).iterrows():

        repository = f"{row['platform']}/{row['project']}"
        commit_message = str(row["processed_message"])
        git_diff = str(row["diff"])

        records.append(
            {
                "index": int(row["source_index"]),
                "hash": row["protocol_record_id"],
                "input_text": build_input_text(
                    commit_message,
                    git_diff,
                ),
                "target": row["protocol_target"],
                "breaking_change": normalize_bool_value(
                    row["breaking_change"]
                ),
                "project": row["project"],
                "repository": repository,
                "commit_hash_original": row["commit_hash"],
                "commit_date": row["commit_date"],
                "original_label": row["canonical_target"],
                "hqcm_label": row["protocol_target"],
                "commit_message": commit_message,
                "git_diff": git_diff,
                "message_tokens": int(row["message_tokens"]),
                "diff_tokens": int(row["diff_tokens"]),
                "total_tokens_original": int(
                    row["total_tokens_original"]
                ),
                "prompt_tokens_llama": int(row["prompt_tokens"]),
                "total_tokens_llama": int(row["total_tokens"]),
            }
        )

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def validate_reference(
    generated: pd.DataFrame,
    reference_path: Path,
) -> None:
    """
    Optional audit against a reference HQCM-compatible CSV.

    This compares logical CSV values, not raw file bytes.
    """
    reference = pd.read_csv(
        reference_path,
        keep_default_na=False,
        low_memory=False,
    )

    missing = set(OUTPUT_COLUMNS) - set(reference.columns)
    if missing:
        raise RuntimeError(
            "Reference HQCM CSV is missing columns: "
            f"{sorted(missing)}"
        )

    reference = reference[OUTPUT_COLUMNS].copy()

    if len(reference) != len(generated):
        raise RuntimeError(
            "Generated/reference row-count mismatch: "
            f"{len(generated)} != {len(reference)}"
        )

    # Normalize booleans before comparison.
    reference["breaking_change"] = reference[
        "breaking_change"
    ].map(normalize_bool_value)

    generated_cmp = generated.reset_index(drop=True)
    reference_cmp = reference.reset_index(drop=True)

    differing_columns = []

    for column in OUTPUT_COLUMNS:
        left = generated_cmp[column].astype(str)
        right = reference_cmp[column].astype(str)

        if not left.equals(right):
            differing_columns.append(column)

    if differing_columns:
        raise RuntimeError(
            "Generated HQCM artifact differs from the reference "
            f"reference in columns: {differing_columns}"
        )

    print("Reference audit:     OK")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the HQCM-compatible RQ4 artifact from the "
            "canonical pre-cutoff dataset and its assignment manifest."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Canonical data/pre_cutoff/commits.csv.gz.",
    )

    parser.add_argument(
        "--assignments",
        required=True,
        type=Path,
        help="HQCM artifact-compatible assignments.csv.",
    )

    parser.add_argument(
        "--csv-output",
        required=True,
        type=Path,
        help="Output path for our_hqcm_len1024.csv.",
    )

    parser.add_argument(
        "--json-output",
        required=True,
        type=Path,
        help="Output path for our_hqcm_len1024.json.",
    )

    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=None,
        help=(
            "Optional reference HQCM-compatible CSV used to audit the "
            "reconstructed artifact."
        ),
    )

    parser.add_argument(
        "--expected-total",
        type=int,
        default=2132,
    )

    args = parser.parse_args()

    print("=" * 72)
    print("HQCM ARTIFACT MATERIALIZATION")
    print("=" * 72)

    dataset = load_dataset(args.dataset)
    assignments = load_assignments(args.assignments)

    if len(assignments) != args.expected_total:
        raise RuntimeError(
            f"Expected {args.expected_total} HQCM-compatible commits, "
            f"found {len(assignments)}."
        )

    generated = materialize(
        dataset,
        assignments,
    )

    args.csv_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated.to_csv(
        args.csv_output,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )

    records = generated.to_dict(
        orient="records"
    )

    with args.json_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    if args.reference_csv is not None:
        validate_reference(
            generated,
            args.reference_csv,
        )

    print(f"Commits:             {len(generated)}")
    print(f"Projects:            {generated['project'].nunique()}")
    print(f"Max total tokens:    {generated['total_tokens_llama'].max()}")
    print(f"CSV output:          {args.csv_output}")
    print(f"JSON output:         {args.json_output}")
    print("HQCM artifact materialized successfully.")


if __name__ == "__main__":
    main()
