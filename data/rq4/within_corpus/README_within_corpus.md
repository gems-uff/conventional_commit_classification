# RQ4 Within-Corpus Holdout Split

This directory contains the split definition used for the supplementary within-corpus evaluation in RQ4.

The protocol is based on the canonical pre-cutoff dataset:

`data/pre_cutoff/commits.csv.gz`

The canonical dataset contains 8,159 commits from 30 repositories and preserves the 11-class taxonomy used by the study:

`build`, `chore`, `ci`, `docs`, `feat`, `fix`, `others`, `perf`, `refactor`, `style`, and `test`.

## Protocol

Unlike the cross-project protocol, the within-corpus protocol uses a single train/test holdout split in which commits from every project may occur in both partitions.

The historical split contains:

| Split | Commits | Projects |
|---|---:|---:|
| Train | 6,527 | 30 |
| Test | 1,632 | 30 |
| Total | 8,159 | 30 |

The split therefore uses approximately 80% of the commits for training and 20% for testing.

Project overlap between train and test is intentional in this protocol. Commit overlap is not allowed.

## Files

### `assignments.csv`

This is the authoritative commit-level definition of the within-corpus split.

Columns:

| Column | Description |
|---|---|
| `protocol` | Experimental protocol (`within_corpus`). |
| `fold` | Fixed to `0` because this protocol uses a single holdout split rather than cross-validation folds. |
| `split` | Whether the commit belongs to the `train` or `test` partition. |
| `project` | Source project. |
| `commit_hash` | Git commit SHA-1. |
| `target` | Ground-truth commit class. |

Each canonical commit appears exactly once in this file, either in the training split or in the test split.

The `target` field is redundant with the canonical dataset and is retained as an integrity check of the historical split definition.

## Materializing the split

The complete historical `train.csv` and `test.csv` files are not stored in the replication package because they duplicate commit contents already available in `data/pre_cutoff/commits.csv.gz`.

They can be reconstructed from the canonical dataset and `assignments.csv` using:

```bash
python3 scripts/rq4/materialize_within_corpus_split.py \
  --dataset data/pre_cutoff/commits.csv.gz \
  --assignments data/rq4/within_corpus/assignments.csv \
  --output-dir data/rq4/within_corpus/split
```

This produces:

```text
data/rq4/within_corpus/split/
├── train.csv
└── test.csv
```

The generated files contain the canonical fields from `data/pre_cutoff/commits.csv.gz`.

The materialization script preserves the row order recorded in `assignments.csv`.

## Validation

Before materializing the split, the script verifies that:

- the canonical dataset contains 8,159 unique pre-cutoff commits;
- `assignments.csv` uses the `within_corpus` protocol and `fold=0`;
- every assignment refers to a commit present in the canonical dataset;
- the target stored in `assignments.csv` matches the canonical target;
- every canonical commit appears in the assignments exactly once;
- the training split contains 6,527 commits;
- the test split contains 1,632 commits;
- all 30 projects occur in the training split;
- all 30 projects occur in the test split; and
- there is no commit overlap between training and test.

Project overlap between train and test is expected because this is a within-corpus protocol.

## Derived diagnostic files

The following development-time diagnostic files are not included in the replication package because they are fully derivable from the canonical dataset and the assignments:

- `train_class_distribution.csv`
- `test_class_distribution.csv`
- `train_project_distribution.csv`
- `test_project_distribution.csv`
- `train_project_by_class_distribution.csv`
- `test_project_by_class_distribution.csv`

The authoritative artifacts for reproducing this protocol are therefore:

```text
data/pre_cutoff/commits.csv.gz
data/rq4/within_corpus/assignments.csv
scripts/rq4/materialize_within_corpus_split.py
```

## Notes on CSV line counts

The canonical and materialized CSV files contain multiline fields such as Git patches and Git Show output. As a result, shell utilities such as `wc -l` do not report the number of commit records. Use a CSV-aware reader such as pandas to count rows.
