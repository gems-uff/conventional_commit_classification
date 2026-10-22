#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import pandas as pd


# ============================================================
# Expected schemas
# ============================================================

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

REQUIRED_SUMMARY_COLUMNS = {
    "fold",
    "splitter",
    "train_commits",
    "test_commits",
    "train_projects",
    "test_projects",
    "project_overlap",
}


# ============================================================
# Loading
# ============================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the canonical pre-cutoff dataset.

    pandas automatically detects .gz compression from the
    filename extension.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(
        path,
        compression="infer",
        keep_default_na=False,
        low_memory=False,
        dtype={
            "platform": str,
            "project": str,
            "commit_hash": str,
            "target": str,
        },
    )

    missing = (
        REQUIRED_DATASET_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Canonical dataset is missing columns: "
            f"{sorted(missing)}"
        )

    df["platform"] = (
        df["platform"]
        .astype(str)
        .str.strip()
    )

    df["project"] = (
        df["project"]
        .astype(str)
        .str.strip()
    )

    df["commit_hash"] = (
        df["commit_hash"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["target"] = (
        df["target"]
        .astype(str)
        .str.strip()
    )

    return df


def load_assignments(
    path: Path,
) -> pd.DataFrame:
    """
    Load the commit-level cross-project assignments.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Assignments file not found: {path}"
        )

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

    missing = (
        REQUIRED_ASSIGNMENT_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Assignments file is missing columns: "
            f"{sorted(missing)}"
        )

    df["protocol"] = (
        df["protocol"]
        .astype(str)
        .str.strip()
    )

    df["split"] = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["project"] = (
        df["project"]
        .astype(str)
        .str.strip()
    )

    df["commit_hash"] = (
        df["commit_hash"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["target"] = (
        df["target"]
        .astype(str)
        .str.strip()
    )

    df["fold"] = pd.to_numeric(
        df["fold"],
        errors="raise",
    ).astype(int)

    return df


def load_fold_summary(
    path: Path,
) -> pd.DataFrame:
    """
    Load the fold summary for optional validation.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Fold summary not found: {path}"
        )

    df = pd.read_csv(
        path,
        keep_default_na=False,
    )

    missing = (
        REQUIRED_SUMMARY_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Fold summary is missing columns: "
            f"{sorted(missing)}"
        )

    df["fold"] = pd.to_numeric(
        df["fold"],
        errors="raise",
    ).astype(int)

    return df


# ============================================================
# Dataset validation
# ============================================================

def validate_dataset(
    dataset: pd.DataFrame,
    expected_total: int,
):
    print()
    print("Validating canonical dataset...")

    if len(dataset) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} commits "
            f"in canonical dataset, found "
            f"{len(dataset)}."
        )

    partitions = set(
        dataset["partition"]
        .astype(str)
        .str.strip()
    )

    if partitions != {"pre_cutoff"}:
        raise RuntimeError(
            "Expected only partition='pre_cutoff', "
            f"found: {sorted(partitions)}"
        )

    duplicate_mask = dataset.duplicated(
        subset=[
            "project",
            "commit_hash",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicated = dataset.loc[
            duplicate_mask,
            [
                "project",
                "commit_hash",
            ],
        ]

        raise RuntimeError(
            "Duplicate commits in canonical dataset:\n"
            f"{duplicated.head(20).to_string(index=False)}"
        )

    print(
        f"  commits:  {len(dataset)}"
    )

    print(
        f"  projects: "
        f"{dataset['project'].nunique()}"
    )

    print("  OK")


# ============================================================
# Assignment validation
# ============================================================

def validate_assignments(
    dataset: pd.DataFrame,
    assignments: pd.DataFrame,
):
    print()
    print("Validating assignments...")

    # --------------------------------------------------------
    # Protocol
    # --------------------------------------------------------

    protocols = set(
        assignments["protocol"]
    )

    if protocols != {"cross_project"}:
        raise RuntimeError(
            "Unexpected protocol values: "
            f"{sorted(protocols)}"
        )

    # --------------------------------------------------------
    # Five folds
    # --------------------------------------------------------

    folds = sorted(
        assignments["fold"].unique()
    )

    if folds != [1, 2, 3, 4, 5]:
        raise RuntimeError(
            "Expected folds [1, 2, 3, 4, 5], "
            f"found {folds}."
        )

    # --------------------------------------------------------
    # train/test only
    # --------------------------------------------------------

    splits = set(
        assignments["split"]
    )

    if splits != {"train", "test"}:
        raise RuntimeError(
            "Expected split values "
            "{'train', 'test'}, "
            f"found {sorted(splits)}."
        )

    # --------------------------------------------------------
    # No duplicate assignment for same commit/fold
    # --------------------------------------------------------

    duplicated = assignments.duplicated(
        subset=[
            "fold",
            "project",
            "commit_hash",
        ],
        keep=False,
    )

    if duplicated.any():
        sample = assignments.loc[
            duplicated,
            [
                "fold",
                "split",
                "project",
                "commit_hash",
            ],
        ]

        raise RuntimeError(
            "A commit appears more than once "
            "inside the same fold:\n"
            f"{sample.head(20).to_string(index=False)}"
        )

    # --------------------------------------------------------
    # Every assignment must exist in canonical dataset
    # --------------------------------------------------------

    canonical_keys = dataset[
        [
            "project",
            "commit_hash",
            "target",
        ]
    ].rename(
        columns={
            "target": "dataset_target",
        }
    )

    checked = assignments.merge(
        canonical_keys,
        on=[
            "project",
            "commit_hash",
        ],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    missing = checked[
        checked["_merge"]
        != "both"
    ]

    if not missing.empty:
        raise RuntimeError(
            "Assignments reference commits absent "
            "from the canonical dataset:\n"
            f"{missing[['fold', 'split', 'project', 'commit_hash']].head(20).to_string(index=False)}"
        )

    # --------------------------------------------------------
    # Target must agree
    # --------------------------------------------------------

    target_mismatch = checked[
        checked["target"]
        != checked["dataset_target"]
    ]

    if not target_mismatch.empty:
        raise RuntimeError(
            "Target mismatch between assignments "
            "and canonical dataset:\n"
            f"{target_mismatch[['project', 'commit_hash', 'target', 'dataset_target']].head(20).to_string(index=False)}"
        )

    # --------------------------------------------------------
    # Every canonical commit must appear in assignments
    # --------------------------------------------------------

    assignment_keys = (
        assignments[
            [
                "project",
                "commit_hash",
            ]
        ]
        .drop_duplicates()
    )

    coverage = dataset[
        [
            "project",
            "commit_hash",
        ]
    ].merge(
        assignment_keys,
        on=[
            "project",
            "commit_hash",
        ],
        how="left",
        indicator=True,
    )

    uncovered = coverage[
        coverage["_merge"]
        != "both"
    ]

    if not uncovered.empty:
        raise RuntimeError(
            "Canonical commits missing from "
            "assignments.csv:\n"
            f"{uncovered[['project', 'commit_hash']].head(20).to_string(index=False)}"
        )

    # --------------------------------------------------------
    # Each commit must appear in all five folds:
    #
    #   1 test
    #   4 train
    # --------------------------------------------------------

    commit_counts = (
        assignments
        .groupby(
            [
                "project",
                "commit_hash",
            ],
            sort=False,
        )
        .agg(
            rows=("fold", "size"),
            unique_folds=(
                "fold",
                "nunique",
            ),
            test_count=(
                "split",
                lambda s: (
                    s == "test"
                ).sum(),
            ),
            train_count=(
                "split",
                lambda s: (
                    s == "train"
                ).sum(),
            ),
        )
        .reset_index()
    )

    invalid_commit_counts = (
        commit_counts[
            (
                commit_counts["rows"]
                != 5
            )
            |
            (
                commit_counts[
                    "unique_folds"
                ]
                != 5
            )
            |
            (
                commit_counts[
                    "test_count"
                ]
                != 1
            )
            |
            (
                commit_counts[
                    "train_count"
                ]
                != 4
            )
        ]
    )

    if not invalid_commit_counts.empty:
        raise RuntimeError(
            "Invalid per-commit fold assignment:\n"
            f"{invalid_commit_counts.head(20).to_string(index=False)}"
        )

    # --------------------------------------------------------
    # Every project must belong to one test fold only
    # --------------------------------------------------------

    test_assignments = assignments[
        assignments["split"]
        == "test"
    ]

    project_test_folds = (
        test_assignments
        .groupby(
            "project"
        )["fold"]
        .nunique()
    )

    invalid_projects = (
        project_test_folds[
            project_test_folds != 1
        ]
    )

    if not invalid_projects.empty:
        raise RuntimeError(
            "Some projects occur in more than one "
            "test fold:\n"
            f"{invalid_projects.to_string()}"
        )

    # --------------------------------------------------------
    # No project overlap between train/test in same fold
    # --------------------------------------------------------

    for fold in folds:

        fold_rows = assignments[
            assignments["fold"]
            == fold
        ]

        train_projects = set(
            fold_rows.loc[
                fold_rows["split"]
                == "train",
                "project",
            ]
        )

        test_projects = set(
            fold_rows.loc[
                fold_rows["split"]
                == "test",
                "project",
            ]
        )

        overlap = (
            train_projects
            & test_projects
        )

        if overlap:
            raise RuntimeError(
                f"Fold {fold} has train/test "
                "project overlap: "
                f"{sorted(overlap)}"
            )

    expected_assignment_rows = (
        len(dataset) * 5
    )

    if (
        len(assignments)
        != expected_assignment_rows
    ):
        raise RuntimeError(
            f"Expected "
            f"{expected_assignment_rows} "
            f"assignment rows, found "
            f"{len(assignments)}."
        )

    print(
        f"  rows:     "
        f"{len(assignments)}"
    )

    print(
        f"  commits:  "
        f"{len(commit_counts)}"
    )

    print(
        f"  folds:    "
        f"{len(folds)}"
    )

    print(
        "  each commit: "
        "4 train + 1 test"
    )

    print(
        "  project overlap: 0"
    )

    print("  OK")


# ============================================================
# Fold summary validation
# ============================================================

def validate_fold_summary(
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
):
    print()
    print(
        "Validating fold_summary.csv..."
    )

    summary_by_fold = (
        summary
        .set_index("fold")
    )

    for fold in range(
        1,
        6,
    ):
        if fold not in summary_by_fold.index:
            raise RuntimeError(
                f"Fold {fold} missing from "
                "fold_summary.csv."
            )

        fold_rows = assignments[
            assignments["fold"]
            == fold
        ]

        train = fold_rows[
            fold_rows["split"]
            == "train"
        ]

        test = fold_rows[
            fold_rows["split"]
            == "test"
        ]

        train_projects = (
            train["project"]
            .nunique()
        )

        test_projects = (
            test["project"]
            .nunique()
        )

        overlap = len(
            set(train["project"])
            & set(test["project"])
        )

        expected = summary_by_fold.loc[
            fold
        ]

        checks = {
            "train_commits": (
                len(train),
                int(
                    expected[
                        "train_commits"
                    ]
                ),
            ),
            "test_commits": (
                len(test),
                int(
                    expected[
                        "test_commits"
                    ]
                ),
            ),
            "train_projects": (
                train_projects,
                int(
                    expected[
                        "train_projects"
                    ]
                ),
            ),
            "test_projects": (
                test_projects,
                int(
                    expected[
                        "test_projects"
                    ]
                ),
            ),
            "project_overlap": (
                overlap,
                int(
                    expected[
                        "project_overlap"
                    ]
                ),
            ),
        }

        for name, (
            actual,
            expected,
        ) in checks.items():

            if actual != expected:
                raise RuntimeError(
                    f"Fold {fold}: "
                    f"{name} differs from "
                    f"fold_summary.csv "
                    f"(actual={actual}, "
                    f"expected={expected})."
                )

    print("  OK")


# ============================================================
# Fold materialization
# ============================================================

def materialize_folds(
    dataset: pd.DataFrame,
    assignments: pd.DataFrame,
    output_dir: Path,
):
    print()
    print(
        "Materializing fold datasets..."
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Keep the canonical dataset order inside every
    # materialized train/test file.
    dataset = dataset.copy()

    dataset[
        "_canonical_order"
    ] = range(
        len(dataset)
    )

    output_columns = [
        column
        for column in dataset.columns
        if column
        != "_canonical_order"
    ]

    generated = []

    for fold in range(
        1,
        6,
    ):

        for split in (
            "train",
            "test",
        ):

            selected = assignments[
                (
                    assignments["fold"]
                    == fold
                )
                &
                (
                    assignments["split"]
                    == split
                )
            ][
                [
                    "project",
                    "commit_hash",
                ]
            ]

            materialized = (
                selected.merge(
                    dataset,
                    on=[
                        "project",
                        "commit_hash",
                    ],
                    how="left",
                    validate="one_to_one",
                )
            )

            materialized = (
                materialized
                .sort_values(
                    "_canonical_order",
                    kind="stable",
                )
            )

            materialized = (
                materialized[
                    output_columns
                ]
            )

            output_path = (
                output_dir
                / (
                    f"fold_{fold}_"
                    f"{split}.csv"
                )
            )

            materialized.to_csv(
                output_path,
                index=False,
                encoding="utf-8",
                quoting=csv.QUOTE_MINIMAL,
                lineterminator="\n",
            )

            generated.append(
                output_path
            )

            print(
                f"  fold {fold} "
                f"{split:<5}: "
                f"{len(materialized):>5} commits "
                f"-> {output_path.name}"
            )

    return generated


# ============================================================
# Final report
# ============================================================

def print_fold_report(
    assignments: pd.DataFrame,
):
    print()
    print("=" * 72)
    print(
        "CROSS-PROJECT FOLD SUMMARY"
    )
    print("=" * 72)

    print()

    for fold in range(
        1,
        6,
    ):

        rows = assignments[
            assignments["fold"]
            == fold
        ]

        train = rows[
            rows["split"]
            == "train"
        ]

        test = rows[
            rows["split"]
            == "test"
        ]

        print(
            f"Fold {fold}: "
            f"train={len(train):>4} "
            f"({train['project'].nunique()} projects), "
            f"test={len(test):>4} "
            f"({test['project'].nunique()} projects)"
        )

    print()
    print("=" * 72)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Materialize the RQ4 cross-project "
            "five-fold datasets from the canonical "
            "pre-cutoff dataset and assignments.csv."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help=(
            "Canonical pre-cutoff commits.csv.gz."
        ),
    )

    parser.add_argument(
        "--assignments",
        required=True,
        type=Path,
        help=(
            "Cross-project "
            "assignments.csv."
        ),
    )

    parser.add_argument(
        "--fold-summary",
        type=Path,
        default=None,
        help=(
            "Optional fold_summary.csv "
            "used for additional validation."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Directory where fold_N_train.csv and "
            "fold_N_test.csv will be generated."
        ),
    )

    parser.add_argument(
        "--expected-total",
        type=int,
        default=8159,
        help=(
            "Expected number of canonical commits."
        ),
    )

    args = parser.parse_args()

    print("=" * 72)
    print(
        "RQ4 CROSS-PROJECT FOLD MATERIALIZATION"
    )
    print("=" * 72)

    print()
    print(
        f"Dataset:     {args.dataset}"
    )

    print(
        f"Assignments: {args.assignments}"
    )

    print(
        f"Output:      {args.output_dir}"
    )

    dataset = load_dataset(
        args.dataset
    )

    assignments = load_assignments(
        args.assignments
    )

    validate_dataset(
        dataset,
        expected_total=args.expected_total,
    )

    validate_assignments(
        dataset,
        assignments,
    )

    if (
        args.fold_summary
        is not None
    ):
        summary = load_fold_summary(
            args.fold_summary
        )

        validate_fold_summary(
            assignments,
            summary,
        )

    generated = materialize_folds(
        dataset,
        assignments,
        args.output_dir,
    )

    print_fold_report(
        assignments
    )

    print()
    print(
        f"Generated files: "
        f"{len(generated)}"
    )

    print()
    print(
        "Cross-project folds materialized "
        "successfully."
    )


if __name__ == "__main__":
    main()