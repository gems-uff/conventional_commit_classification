#!/usr/bin/env python3

import argparse
import csv
import shlex
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
    "message_tokens",
    "diff_tokens",
    "total_tokens_original",
    "prompt_tokens",
    "label_tokens",
    "total_tokens_with_label",
    "fits_1024",
}

OUTPUT_COLUMNS = [
    "project",
    "platform",
    "repository",
    "commit_hash",
    "commit_date",
    "annotated_type",
    "breaking_change",
    "message_original",
    "commit_message",
    "git_diff",
    "message_tokens",
    "diff_tokens",
    "total_tokens_original",
    "flaccs_prompt_tokens",
    "flaccs_label_tokens",
    "flaccs_total_tokens_with_label",
    "flaccs_fits_1024",
    "files_changed",
    "insertions",
    "deletions",
    "files_formatted",
    "stats_summary",
]


def normalize_string(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def normalize_hash(series: pd.Series) -> pd.Series:
    return normalize_string(series).str.lower()


def normalize_bool_value(value) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False

    raise ValueError(f"Invalid boolean value: {value!r}")


def to_int(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="raise")
    if values.isna().any():
        raise ValueError(f"Missing numeric value in {name}.")
    if (values % 1 != 0).any():
        raise ValueError(f"Non-integer value found in {name}.")
    return values.astype(int)


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
        raise FileNotFoundError(f"FLACCS assignments not found: {path}")

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
        },
    )

    missing = REQUIRED_ASSIGNMENT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "FLACCS assignments are missing required columns: "
            f"{sorted(missing)}"
        )

    for column in (
        "protocol",
        "project",
        "canonical_target",
        "protocol_target",
    ):
        df[column] = normalize_string(df[column])

    df["commit_hash"] = normalize_hash(df["commit_hash"])

    for column in (
        "message_tokens",
        "diff_tokens",
        "total_tokens_original",
        "prompt_tokens",
        "label_tokens",
        "total_tokens_with_label",
    ):
        df[column] = to_int(df[column], column)

    df["fits_1024"] = df["fits_1024"].map(
        normalize_bool_value
    )

    protocols = set(df["protocol"])
    if protocols != {"flaccs_compatible"}:
        raise RuntimeError(
            "Expected protocol='flaccs_compatible', "
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
            "Duplicate commits in FLACCS assignments:\n"
            f"{sample.to_string(index=False)}"
        )

    invalid_fit = df[
        (~df["fits_1024"])
        | (df["total_tokens_with_label"] > 1024)
    ]
    if not invalid_fit.empty:
        raise RuntimeError(
            "FLACCS assignments contain rows that do not satisfy "
            "the stored <=1024 compatibility condition."
        )

    # Preserve manifest order, which comes from the historical source CSV.
    df["_assignment_order"] = range(len(df))

    return df


def parse_diff_stats(diff: str):
    """
    Reconstruct the per-file statistics used in the historical FLACCS
    artifact from the canonical unified Git diff.

    Insertions/deletions are counted only inside hunks, so metadata lines
    such as +++ and --- are not counted.
    """
    files = []
    current = None
    in_hunk = False

    for line in str(diff).splitlines():
        if line.startswith("diff --git "):
            if current is not None:
                files.append(current)

            raw = line[len("diff --git "):]

            try:
                parts = shlex.split(raw)
            except ValueError:
                parts = raw.split()

            if len(parts) >= 2:
                path = parts[1]
            elif parts:
                path = parts[-1]
            else:
                path = ""

            if path.startswith("b/"):
                path = path[2:]

            current = {
                "path": path,
                "insertions": 0,
                "deletions": 0,
            }
            in_hunk = False
            continue

        if current is None:
            continue

        if line.startswith("@@"):
            in_hunk = True
            continue

        if not in_hunk:
            continue

        if line.startswith("+") and not line.startswith("+++"):
            current["insertions"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            current["deletions"] += 1

    if current is not None:
        files.append(current)

    files_changed = len(files)
    insertions = sum(item["insertions"] for item in files)
    deletions = sum(item["deletions"] for item in files)

    files_formatted = "\n".join(
        f"- `{item['path']}` "
        f"(+{item['insertions']}, -{item['deletions']})"
        for item in files
    )

    stats_summary = (
        f"{files_changed} files changed, "
        f"{insertions} insertions(+), "
        f"{deletions} deletions(-)"
    )

    return {
        "files_changed": files_changed,
        "insertions": insertions,
        "deletions": deletions,
        "files_formatted": files_formatted,
        "stats_summary": stats_summary,
    }


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
            "FLACCS assignments reference commits absent from the "
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

    selected = selected.sort_values(
        "_assignment_order",
        kind="stable",
    )

    for _, row in selected.iterrows():
        stats = parse_diff_stats(
            row["diff"]
        )

        records.append(
            {
                "project": row["project"],
                "platform": row["platform"],
                "repository": f"{row['platform']}/{row['project']}",
                "commit_hash": row["commit_hash"],
                "commit_date": row["commit_date"],
                "annotated_type": row["protocol_target"],
                "breaking_change": normalize_bool_value(
                    row["breaking_change"]
                ),
                "message_original": row["original_message"],
                "commit_message": row["processed_message"],
                "git_diff": row["diff"],
                "message_tokens": int(row["message_tokens"]),
                "diff_tokens": int(row["diff_tokens"]),
                "total_tokens_original": int(
                    row["total_tokens_original"]
                ),
                "flaccs_prompt_tokens": int(row["prompt_tokens"]),
                "flaccs_label_tokens": int(row["label_tokens"]),
                "flaccs_total_tokens_with_label": int(
                    row["total_tokens_with_label"]
                ),
                "flaccs_fits_1024": bool(row["fits_1024"]),
                "files_changed": stats["files_changed"],
                "insertions": stats["insertions"],
                "deletions": stats["deletions"],
                "files_formatted": stats["files_formatted"],
                "stats_summary": stats["stats_summary"],
            }
        )

    return pd.DataFrame(
        records,
        columns=OUTPUT_COLUMNS,
    )


def validate_reference(
    generated: pd.DataFrame,
    reference_path: Path,
) -> None:
    """
    Optional development-time audit against the historical FLACCS CSV.

    This is strongly recommended once before publishing the package,
    because file-stat formatting is reconstructed from the Git diff.
    """
    reference = pd.read_csv(
        reference_path,
        keep_default_na=False,
        low_memory=False,
    )

    missing = set(OUTPUT_COLUMNS) - set(reference.columns)
    if missing:
        raise RuntimeError(
            "Reference FLACCS CSV is missing columns: "
            f"{sorted(missing)}"
        )

    reference = reference[OUTPUT_COLUMNS].copy()

    if len(reference) != len(generated):
        raise RuntimeError(
            "Generated/reference row-count mismatch: "
            f"{len(generated)} != {len(reference)}"
        )

    for frame in (generated, reference):
        frame["breaking_change"] = frame[
            "breaking_change"
        ].map(normalize_bool_value)

        frame["flaccs_fits_1024"] = frame[
            "flaccs_fits_1024"
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
        print(
            "Columns differing from historical reference: "
            + ", ".join(differing_columns)
        )

        for column in differing_columns:
            mask = (
                generated_cmp[column].astype(str)
                != reference_cmp[column].astype(str)
            )
            first = mask[mask].index[0]

            print()
            print(f"First mismatch in {column}: row {first}")
            print(f"  generated: {generated_cmp.loc[first, column]!r}")
            print(f"  reference: {reference_cmp.loc[first, column]!r}")

        raise RuntimeError(
            "Generated FLACCS artifact does not exactly reproduce "
            "the historical reference."
        )

    print("Reference audit:     OK")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the FLACCS-compatible RQ4 artifact from the "
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
        help="FLACCS artifact-compatible assignments.csv.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for our_flaccs_10_len1024.csv.",
    )

    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=None,
        help=(
            "Optional historical FLACCS CSV used to audit the "
            "reconstructed artifact. Recommended before publication."
        ),
    )

    parser.add_argument(
        "--expected-total",
        type=int,
        default=2573,
    )

    args = parser.parse_args()

    print("=" * 72)
    print("FLACCS ARTIFACT MATERIALIZATION")
    print("=" * 72)

    dataset = load_dataset(args.dataset)
    assignments = load_assignments(args.assignments)

    if len(assignments) != args.expected_total:
        raise RuntimeError(
            f"Expected {args.expected_total} FLACCS-compatible commits, "
            f"found {len(assignments)}."
        )

    generated = materialize(
        dataset,
        assignments,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated.to_csv(
        args.output,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )

    if args.reference_csv is not None:
        validate_reference(
            generated,
            args.reference_csv,
        )

    print(f"Commits:             {len(generated)}")
    print(f"Projects:            {generated['project'].nunique()}")
    print(
        "Max total tokens:    "
        f"{generated['flaccs_total_tokens_with_label'].max()}"
    )
    print(f"Output:              {args.output}")
    print("FLACCS artifact materialized successfully.")


if __name__ == "__main__":
    main()
