# RQ4 Cross-Project 5-Fold Split

This directory contains the split definition used for the cross-project evaluation in RQ4.

The protocol is based on the canonical pre-cutoff dataset available at:

`data/pre_cutoff/commits.csv.gz`

The canonical dataset contains 8,159 commits from 30 repositories and preserves the 11-class taxonomy used by the study:

`build`, `chore`, `ci`, `docs`, `feat`, `fix`, `others`, `perf`, `refactor`, `style`, and `test`.

## Files

### `assignments.csv`

This is the authoritative commit-level definition of the five cross-project folds.

Columns:

| Column | Description |
|---|---|
| `protocol` | Experimental protocol (`cross_project`). |
| `fold` | Fold identifier, from 1 to 5. |
| `split` | Whether the commit belongs to the training or test partition for that fold. |
| `project` | Source project. |
| `commit_hash` | Git commit SHA-1. |
| `target` | Ground-truth commit class. |

Each commit occurs five times in this file: once in the test split and four times in training splits.

Projects are used as grouping units. Consequently, a project assigned to the test partition of a fold does not occur in the training partition of that same fold.

The assignments were generated using `StratifiedGroupKFold`.

### `fold_summary.csv`

Provides a compact audit of the five folds, including the numbers of training and test commits, numbers of projects, project overlap, and class coverage.

The historical split sizes are:

| Fold | Train commits | Test commits | Train projects | Test projects |
|---:|---:|---:|---:|---:|
| 1 | 6,498 | 1,661 | 24 | 6 |
| 2 | 6,696 | 1,463 | 24 | 6 |
| 3 | 6,681 | 1,478 | 25 | 5 |
| 4 | 6,650 | 1,509 | 24 | 6 |
| 5 | 6,111 | 2,048 | 23 | 7 |

There is no project overlap between the training and test partitions of any fold.

## Materializing the folds

The complete `fold_*_train.csv` and `fold_*_test.csv` files are not stored in the replication package because they duplicate the commit contents already available in the canonical pre-cutoff dataset.

They can be reconstructed exactly from `commits.csv.gz` and `assignments.csv` using:

```bash
python3 scripts/rq4_1/materialize_cross_project_folds.py \
  --dataset data/pre_cutoff/commits.csv.gz \
  --assignments data/rq4_1/cross_project/assignments.csv \
  --fold-summary data/rq4_1/cross_project/fold_summary.csv \
  --output-dir data/rq4_1/cross_project/folds
```

This produces:

```text
fold_1_train.csv
fold_1_test.csv
fold_2_train.csv
fold_2_test.csv
fold_3_train.csv
fold_3_test.csv
fold_4_train.csv
fold_4_test.csv
fold_5_train.csv
fold_5_test.csv
```

Each materialized file contains the same canonical fields available in `data/pre_cutoff/commits.csv.gz`.

## Validation

Before materializing the folds, the script verifies that:

- the canonical dataset contains 8,159 unique pre-cutoff commits;
- all assignment records refer to commits present in the canonical dataset;
- the class stored in `assignments.csv` matches the canonical target;
- every canonical commit is represented in the assignments;
- every commit occurs in all five folds;
- every commit is assigned to the test split exactly once and to the training split exactly four times;
- every project belongs to exactly one test fold;
- there is no project overlap between training and test data within a fold; and
- the reconstructed fold sizes match `fold_summary.csv`.

The `target` field in `assignments.csv` is redundant with the canonical dataset and is retained as an integrity check of the historical split definition.

## Derived diagnostic files

Class-distribution, project-distribution, and project-by-class distribution files used during development are not included because they are fully derivable from the canonical dataset and the assignments.

The authoritative artifacts for reproducing this protocol are therefore:

```text
data/pre_cutoff/commits.csv.gz
data/rq4_1/cross_project/assignments.csv
data/rq4_1/cross_project/fold_summary.csv
scripts/rq4_1/materialize_cross_project_folds.py
```
