#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import pandas as pd


REQUIRED_DATASET_COLUMNS = {
    "partition",
    "platform",
    "project",
    "commit_hash",
    "commit_date",
    "target",
    "original_type",
    "breaking_change",
    "original_message",
    "processed_message",
    "has_diff",
    "diff",
    "git_show",
}

REQUIRED_ASSIGNMENT_COLUMNS = {
    "protocol",
    "fold",
    "split",
    "project",
    "commit_hash",
    "target",
}


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the canonical pre-cutoff dataset.

    pandas detects gzip compression automatically from the .gz suffix.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

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

    for column in ("partition", "platform", "project", "commit_hash", "target"):
        df[column] = df[column].astype(str).str.strip()

    df["commit_hash"] = df["commit_hash"].str.lower()

    return df


def load_assignments(path: Path) -> pd.DataFrame:
    """
    Load the historical within-corpus split assignments.
    """
    if not path.exists():
        raise FileNotFoundError(f"Assignments file not found: {path}")

    df = pd.read_csv(
        path,
        keep_default_na=False,
        dtype={
            "protocol": str,
            "split": str,
            "project": str,
            "commit_hash": str,
            "target": str,
        },
    )

    missing = REQUIRED_ASSIGNMENT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Assignments file is missing required columns: "
            f"{sorted(missing)}"
        )

    for column in ("protocol", "split", "project", "commit_hash", "target"):
        df[column] = df[column].astype(str).str.strip()

    df["protocol"] = df["protocol"].str.lower()
    df["split"] = df["split"].str.lower()
    df["commit_hash"] = df["commit_hash"].str.lower()
    df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(int)

    # Preserve the historical assignment order so the materialized
    # train/test files follow assignments.csv exactly.
    df["_assignment_order"] = range(len(df))

    return df


def validate_dataset(dataset: pd.DataFrame, expected_total: int) -> None:
    print()
    print("Validating canonical pre-cutoff dataset...")

    if len(dataset) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} commits, found {len(dataset)}."
        )

    partitions = set(dataset["partition"])
    if partitions != {"pre_cutoff"}:
        raise RuntimeError(
            "Expected only partition='pre_cutoff', "
            f"found {sorted(partitions)}."
        )

    duplicates = dataset.duplicated(
        subset=["project", "commit_hash"],
        keep=False,
    )

    if duplicates.any():
        sample = dataset.loc[
            duplicates,
            ["project", "commit_hash"],
        ].head(20)

        raise RuntimeError(
            "Duplicate commits in canonical dataset:\n"
            f"{sample.to_string(index=False)}"
        )

    print(f"  commits:  {len(dataset)}")
    print(f"  projects: {dataset['project'].nunique()}")
    print("  OK")


def validate_assignments(
    dataset: pd.DataFrame,
    assignments: pd.DataFrame,
    expected_train: int,
    expected_test: int,
) -> None:
    print()
    print("Validating within-corpus assignments...")

    protocols = set(assignments["protocol"])
    if protocols != {"within_corpus"}:
        raise RuntimeError(
            "Unexpected protocol values: "
            f"{sorted(protocols)}"
        )

    folds = sorted(assignments["fold"].unique())
    if folds != [0]:
        raise RuntimeError(
            "Expected fold=0 for the within-corpus holdout split, "
            f"found {folds}."
        )

    splits = set(assignments["split"])
    if splits != {"train", "test"}:
        raise RuntimeError(
            "Expected split values {'train', 'test'}, "
            f"found {sorted(splits)}."
        )

    # In the within-corpus protocol each canonical commit must occur
    # exactly once: either train or test.
    duplicates = assignments.duplicated(
        subset=["project", "commit_hash"],
        keep=False,
    )

    if duplicates.any():
        sample = assignments.loc[
            duplicates,
            ["split", "project", "commit_hash"],
        ].head(20)

        raise RuntimeError(
            "A commit is assigned more than once:\n"
            f"{sample.to_string(index=False)}"
        )

    canonical = dataset[
        ["project", "commit_hash", "target"]
    ].rename(
        columns={"target": "dataset_target"}
    )

    checked = assignments.merge(
        canonical,
        on=["project", "commit_hash"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing = checked[checked["_merge"] != "both"]
    if not missing.empty:
        raise RuntimeError(
            "Assignments reference commits absent from the "
            "canonical dataset:\n"
            f"{missing[['split', 'project', 'commit_hash']].head(20).to_string(index=False)}"
        )

    target_mismatch = checked[
        checked["target"] != checked["dataset_target"]
    ]

    if not target_mismatch.empty:
        raise RuntimeError(
            "Target mismatch between assignments.csv and the "
            "canonical dataset:\n"
            f"{target_mismatch[['project', 'commit_hash', 'target', 'dataset_target']].head(20).to_string(index=False)}"
        )

    assignment_keys = assignments[
        ["project", "commit_hash"]
    ]

    coverage = dataset[
        ["project", "commit_hash"]
    ].merge(
        assignment_keys,
        on=["project", "commit_hash"],
        how="left",
        indicator=True,
    )

    uncovered = coverage[coverage["_merge"] != "both"]
    if not uncovered.empty:
        raise RuntimeError(
            "Canonical commits missing from assignments.csv:\n"
            f"{uncovered[['project', 'commit_hash']].head(20).to_string(index=False)}"
        )

    if len(assignments) != len(dataset):
        raise RuntimeError(
            "Expected one assignment row per canonical commit "
            f"({len(dataset)}), found {len(assignments)}."
        )

    train = assignments[assignments["split"] == "train"]
    test = assignments[assignments["split"] == "test"]

    if len(train) != expected_train:
        raise RuntimeError(
            f"Expected {expected_train} training commits, "
            f"found {len(train)}."
        )

    if len(test) != expected_test:
        raise RuntimeError(
            f"Expected {expected_test} test commits, "
            f"found {len(test)}."
        )

    # A within-corpus split is expected to contain all projects in
    # both train and test, unlike the cross-project protocol.
    dataset_projects = set(dataset["project"])
    train_projects = set(train["project"])
    test_projects = set(test["project"])

    if train_projects != dataset_projects:
        missing_projects = sorted(dataset_projects - train_projects)
        extra_projects = sorted(train_projects - dataset_projects)
        raise RuntimeError(
            "Training split project coverage differs from the "
            "canonical dataset. "
            f"Missing={missing_projects}, extra={extra_projects}"
        )

    if test_projects != dataset_projects:
        missing_projects = sorted(dataset_projects - test_projects)
        extra_projects = sorted(test_projects - dataset_projects)
        raise RuntimeError(
            "Test split project coverage differs from the "
            "canonical dataset. "
            f"Missing={missing_projects}, extra={extra_projects}"
        )

    # No commit overlap is allowed, but project overlap is expected.
    train_keys = set(
        zip(train["project"], train["commit_hash"])
    )
    test_keys = set(
        zip(test["project"], test["commit_hash"])
    )

    commit_overlap = train_keys & test_keys
    if commit_overlap:
        raise RuntimeError(
            "Commit overlap between train and test:\n"
            f"{sorted(commit_overlap)[:20]}"
        )

    project_overlap = train_projects & test_projects

    print(f"  assignment rows: {len(assignments)}")
    print(f"  train commits:   {len(train)}")
    print(f"  test commits:    {len(test)}")
    print(f"  train projects:  {len(train_projects)}")
    print(f"  test projects:   {len(test_projects)}")
    print(f"  commit overlap:  {len(commit_overlap)}")
    print(f"  project overlap: {len(project_overlap)} (expected)")
    print("  OK")


def materialize_split(
    dataset: pd.DataFrame,
    assignments: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    print()
    print("Materializing within-corpus train/test files...")

    output_dir.mkdir(parents=True, exist_ok=True)

    canonical = dataset.copy()

    generated = []

    for split in ("train", "test"):
        selected = assignments.loc[
            assignments["split"] == split,
            [
                "project",
                "commit_hash",
                "_assignment_order",
            ],
        ]

        materialized = selected.merge(
            canonical,
            on=["project", "commit_hash"],
            how="left",
            validate="one_to_one",
        )

        # Preserve exact historical assignment order.
        materialized = materialized.sort_values(
            "_assignment_order",
            kind="stable",
        ).drop(
            columns=["_assignment_order"]
        )

        output_path = output_dir / f"{split}.csv"

        materialized.to_csv(
            output_path,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )

        generated.append(output_path)

        print(
            f"  {split:<5}: {len(materialized):>5} commits "
            f"-> {output_path}"
        )

    return generated


def print_project_distribution(assignments: pd.DataFrame) -> None:
    print()
    print("=" * 72)
    print("WITHIN-CORPUS SPLIT SUMMARY")
    print("=" * 72)
    print()

    table = (
        assignments
        .groupby(["project", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for _, row in table.iterrows():
        print(
            f"{row['project']:<24} "
            f"train={int(row.get('train', 0)):>4} "
            f"test={int(row.get('test', 0)):>4}"
        )

    print()
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the RQ4 within-corpus holdout split "
            "from the canonical pre-cutoff dataset and "
            "assignments.csv."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Canonical pre-cutoff commits.csv.gz.",
    )

    parser.add_argument(
        "--assignments",
        required=True,
        type=Path,
        help="Historical within-corpus assignments.csv.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where train.csv and test.csv will be generated.",
    )

    parser.add_argument(
        "--expected-total",
        type=int,
        default=8159,
        help="Expected number of canonical commits.",
    )

    parser.add_argument(
        "--expected-train",
        type=int,
        default=6527,
        help="Expected number of training commits.",
    )

    parser.add_argument(
        "--expected-test",
        type=int,
        default=1632,
        help="Expected number of test commits.",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("RQ4 WITHIN-CORPUS SPLIT MATERIALIZATION")
    print("=" * 72)

    print()
    print(f"Dataset:     {args.dataset}")
    print(f"Assignments: {args.assignments}")
    print(f"Output:      {args.output_dir}")

    dataset = load_dataset(args.dataset)
    assignments = load_assignments(args.assignments)

    validate_dataset(
        dataset,
        expected_total=args.expected_total,
    )

    validate_assignments(
        dataset,
        assignments,
        expected_train=args.expected_train,
        expected_test=args.expected_test,
    )

    generated = materialize_split(
        dataset,
        assignments,
        args.output_dir,
    )

    print_project_distribution(assignments)

    print()
    print(f"Generated files: {len(generated)}")
    print("Within-corpus split materialized successfully.")


if __name__ == "__main__":
    main()
