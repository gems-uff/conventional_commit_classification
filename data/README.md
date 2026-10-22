# Data

```text
data/
├── repositories.csv
├── pre_cutoff/
│   ├── commits.csv.gz
│   └── commits.csv.gz.sha256
├── post_cutoff/
│   ├── commits.csv.gz
│   └── commits.csv.gz.sha256
├── rq4_1/
│   ├── cross_project/   assignments.csv, fold_summary.csv
│   └── within_corpus/   assignments.csv
├── rq4_2/
│   └── artifact_compatible/
│       ├── flaccs/      assignments.csv
│       └── hqcm/        assignments.csv
└── external/
    └── hqcm/            retrieval script for the upstream HQCM artifact
```

## Data used by each RQ

| RQ | Data |
|---|---|
| RQ1 | `pre_cutoff/commits.csv.gz`, `post_cutoff/commits.csv.gz` |
| RQ2, RQ3.1, RQ3.2 | `post_cutoff/commits.csv.gz` |
| RQ4.1 cross-project | `pre_cutoff/commits.csv.gz` + `rq4_1/cross_project/assignments.csv` |
| RQ4.1 within-corpus | `pre_cutoff/commits.csv.gz` + `rq4_1/within_corpus/assignments.csv` |
| RQ4.2 FLACCS | `pre_cutoff/commits.csv.gz` + `rq4_2/artifact_compatible/flaccs/assignments.csv` |
| RQ4.2 HQCM | `pre_cutoff/commits.csv.gz` + `rq4_2/artifact_compatible/hqcm/assignments.csv` |
| RQ4.2 HQCM original dataset | upstream HQCM artifact (`external/hqcm/README.md`) |

## Canonical datasets

| File | Commits | Projects |
|---|---:|---:|
| `pre_cutoff/commits.csv.gz` | 8,159 | 30 |
| `post_cutoff/commits.csv.gz` | 7,937 | 30 |

Verify the files:

```bash
(cd data/pre_cutoff && sha256sum -c commits.csv.gz.sha256)
(cd data/post_cutoff && sha256sum -c commits.csv.gz.sha256)
```

Columns:

| Column | Content |
|---|---|
| `partition` | `pre_cutoff` or `post_cutoff`. |
| `platform` | Ecosystem identifier. |
| `project` | Project identifier. |
| `commit_hash` | Git commit SHA-1. |
| `commit_date` | Commit date. |
| `target` | Ground-truth class. |
| `original_type` | Type declared in the original commit message. |
| `breaking_change` | Whether the commit is marked as a breaking change. |
| `original_message` | Original commit message. |
| `processed_message` | Commit message given to the models. |
| `has_diff` | Whether the commit has a Git patch. |
| `diff` | Git diff (empty when `has_diff` is false). |
| `git_show` | Git Show text given to the models. |

Labels: `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `others`, `perf`,
`refactor`, `style`, `test`.

Read the files with a CSV parser (`diff` and `git_show` are multiline):

```python
import pandas as pd

pre = pd.read_csv("data/pre_cutoff/commits.csv.gz", keep_default_na=False, low_memory=False)
```

`repositories.csv` lists the 30 projects (`source_order`, `platform`,
`project`, `repository`, `repository_url`).

## RQ4 splits and subsets

The runners read the `assignments.csv` files directly. The commands below are
optional: they write the splits and subsets as separate CSV files, after
validating them against the canonical dataset. See the README in each
directory for the file format.

```bash
# RQ4.1 cross-project: fold_<N>_train.csv / fold_<N>_test.csv
python3 scripts/rq4_1/materialize_cross_project_folds.py \
  --dataset data/pre_cutoff/commits.csv.gz \
  --assignments data/rq4_1/cross_project/assignments.csv \
  --fold-summary data/rq4_1/cross_project/fold_summary.csv \
  --output-dir data/rq4_1/cross_project/folds

# RQ4.1 within-corpus: train.csv / test.csv
python3 scripts/rq4_1/materialize_within_corpus_split.py \
  --dataset data/pre_cutoff/commits.csv.gz \
  --assignments data/rq4_1/within_corpus/assignments.csv \
  --output-dir data/rq4_1/within_corpus/split

# RQ4.2 FLACCS-compatible subset
python3 scripts/rq4_2/materialize_flaccs_artifact.py \
  --dataset data/pre_cutoff/commits.csv.gz \
  --assignments data/rq4_2/artifact_compatible/flaccs/assignments.csv \
  --output data/rq4_2/artifact_compatible/flaccs/materialized/our_flaccs_10_len1024.csv

# RQ4.2 HQCM-compatible subset
python3 scripts/rq4_2/materialize_hqcm_artifact.py \
  --dataset data/pre_cutoff/commits.csv.gz \
  --assignments data/rq4_2/artifact_compatible/hqcm/assignments.csv \
  --csv-output data/rq4_2/artifact_compatible/hqcm/materialized/our_hqcm_len1024.csv \
  --json-output data/rq4_2/artifact_compatible/hqcm/materialized/our_hqcm_len1024.json
```
