# Data

This directory contains the canonical datasets and dataset-derived definitions required to reproduce the experiments reported in the study.

The replication package separates **canonical commit datasets** from **experiment-specific split and artifact-compatibility definitions**. Canonical datasets store each selected commit once; derived RQ4 datasets are represented through compact assignment manifests and materialized on demand.

Third-party artifacts are not stored here. External upstream artifacts, such as CodeFuse-HQCM, are documented under `external/`.

## Directory structure

```text
data/
├── README.md
├── repositories.csv
├── pre_cutoff/
│   ├── commits.csv.gz
│   └── commits.csv.gz.sha256
├── post_cutoff/
│   ├── commits.csv.gz
│   └── commits.csv.gz.sha256
└── rq4/
    ├── cross_project/
    │   ├── README.md
    │   ├── assignments.csv
    │   └── fold_summary.csv
    ├── within_corpus/
    │   ├── README.md
    │   └── assignments.csv
    └── artifact_compatible/
        ├── hqcm/
        │   ├── README.md
        │   └── assignments.csv
        └── flaccs/
            ├── README.md
            └── assignments.csv
```

## Repository manifest

`repositories.csv` lists the 30 corpus entries used by the study and preserves their original corpus order.

Columns:

| Column | Description |
|---|---|
| `source_order` | Position in the original corpus definition. |
| `platform` | Ecosystem identifier used by the corpus. |
| `project` | Project identifier used throughout the datasets. |
| `repository` | Combined corpus identifier (`platform/project`). |

The `repository` field is the identifier used by the dataset-generation pipeline; it should not be interpreted as a GitHub `owner/repository` URL.

## Canonical datasets

### `pre_cutoff/commits.csv.gz`

The canonical pre-cutoff dataset contains **8,159 commits from 30 projects**.

Of these, 8,149 contain a Git diff and 10 do not.

Class distribution:

| Class | Commits | Percentage |
|---|---:|---:|
| `chore` | 2,077 | 25.46% |
| `fix` | 1,914 | 23.46% |
| `docs` | 1,363 | 16.71% |
| `feat` | 986 | 12.08% |
| `refactor` | 568 | 6.96% |
| `others` | 461 | 5.65% |
| `test` | 340 | 4.17% |
| `ci` | 212 | 2.60% |
| `build` | 157 | 1.92% |
| `perf` | 66 | 0.81% |
| `style` | 15 | 0.18% |

This dataset is the basis for RQ4 and its derived split and artifact-compatible definitions.

### `post_cutoff/commits.csv.gz`

The canonical post-cutoff dataset contains **7,937 commits from the same 30 projects**.

It is used by the post-cutoff experiments, including the model, temperature, input-representation, and prompt analyses.

## Canonical schema

Both canonical datasets use the same schema:

| Column | Description |
|---|---|
| `partition` | Temporal partition (`pre_cutoff` or `post_cutoff`). |
| `platform` | Repository ecosystem identifier used by the corpus. |
| `project` | Project identifier. |
| `commit_hash` | Git commit SHA-1. |
| `commit_date` | Commit date used by the dataset-construction pipeline. |
| `target` | Ground-truth class in the study taxonomy. |
| `original_type` | Conventional Commit type extracted from the original developer-authored message before unsupported valid types are mapped to `others`. |
| `breaking_change` | Whether the commit was marked as a breaking change. |
| `original_message` | Original developer-authored commit message. |
| `processed_message` | Commit message after preprocessing used by the experimental pipeline. |
| `has_diff` | Whether the commit contains a Git patch. |
| `diff` | Pure Git diff representation. Empty for commits without a patch. |
| `git_show` | Processed Git Show representation used by the study. |

The canonical taxonomy contains 11 classes:

```text
build
chore
ci
docs
feat
fix
others
perf
refactor
style
test
```

`others` is a valid semantic class. Invalid or unparseable model outputs are evaluation failures and are not represented as a semantic class.

## Loading the canonical datasets

The datasets are gzip-compressed CSV files and can be read directly by common CSV libraries.

Example:

```python
import pandas as pd

pre = pd.read_csv(
    "data/pre_cutoff/commits.csv.gz",
    compression="infer",
    keep_default_na=False,
    low_memory=False,
)

post = pd.read_csv(
    "data/post_cutoff/commits.csv.gz",
    compression="infer",
    keep_default_na=False,
    low_memory=False,
)
```

Fields such as `diff` and `git_show` are multiline. Therefore, line-oriented tools such as `wc -l` do not provide valid commit-record counts; use a CSV-aware parser.

## Dataset use by research question

| Research question / protocol | Data source |
|---|---|
| RQ1 temporal comparison | `pre_cutoff/commits.csv.gz` and `post_cutoff/commits.csv.gz` |
| RQ2 model comparison and temperature analysis | `post_cutoff/commits.csv.gz` |
| RQ3 input and prompt analyses | `post_cutoff/commits.csv.gz` |
| RQ4 cross-project benchmark | `pre_cutoff/commits.csv.gz` + `rq4/cross_project/assignments.csv` |
| RQ4 within-corpus supplementary benchmark | `pre_cutoff/commits.csv.gz` + `rq4/within_corpus/assignments.csv` |
| RQ4 HQCM artifact-compatible evaluation | `pre_cutoff/commits.csv.gz` + `rq4/artifact_compatible/hqcm/assignments.csv` |
| RQ4 FLACCS artifact-compatible evaluation | `pre_cutoff/commits.csv.gz` + `rq4/artifact_compatible/flaccs/assignments.csv` |

The separate evaluation on the **original HQCM artifact and data** is not represented as a dataset under `data/`. Its upstream provenance and retrieval instructions are available at:

`external/hqcm/README.md`

## RQ4 derived data

### Cross-project evaluation

`rq4/cross_project/assignments.csv` is the authoritative commit-level definition of the five cross-project folds.

Materialize the train/test files with:

```bash
python3 scripts/rq4/materialize_cross_project_folds.py \
  --dataset data/pre_cutoff/commits.csv.gz \
  --assignments data/rq4/cross_project/assignments.csv \
  --fold-summary data/rq4/cross_project/fold_summary.csv \
  --output-dir data/rq4/cross_project/folds
```

### Within-corpus evaluation

`rq4/within_corpus/assignments.csv` defines the historical within-corpus holdout.

Materialize it with:

```bash
python3 scripts/rq4/materialize_within_corpus_split.py \
  --dataset data/pre_cutoff/commits.csv.gz \
  --assignments data/rq4/within_corpus/assignments.csv \
  --output-dir data/rq4/within_corpus/split
```

### HQCM artifact-compatible subset

The HQCM-compatible subset contains **2,132 commits from 30 projects**.

Materialize it with:

```bash
python3 scripts/rq4/materialize_hqcm_artifact.py \
  --dataset data/pre_cutoff/commits.csv.gz \
  --assignments data/rq4/artifact_compatible/hqcm/assignments.csv \
  --csv-output data/rq4/artifact_compatible/hqcm/materialized/our_hqcm_len1024.csv \
  --json-output data/rq4/artifact_compatible/hqcm/materialized/our_hqcm_len1024.json
```

The original HQCM artifact is external and is documented under `external/hqcm/`.

### FLACCS artifact-compatible subset

The FLACCS-compatible subset contains **2,573 commits from 30 projects**.

Materialize it with:

```bash
python3 scripts/rq4/materialize_flaccs_artifact.py \
  --dataset data/pre_cutoff/commits.csv.gz \
  --assignments data/rq4/artifact_compatible/flaccs/assignments.csv \
  --output data/rq4/artifact_compatible/flaccs/materialized/our_flaccs_10_len1024.csv
```

## Checksums

Each canonical dataset is accompanied by a SHA-256 checksum:

```text
pre_cutoff/commits.csv.gz.sha256
post_cutoff/commits.csv.gz.sha256
```

Verify them with:

```bash
sha256sum -c data/pre_cutoff/commits.csv.gz.sha256
sha256sum -c data/post_cutoff/commits.csv.gz.sha256
```

## Provenance

The canonical CSV files are the primary data artifacts for reproducing our experiments. They are published directly so that reproduction does not require rebuilding the historical corpus from the source repositories.

Dataset-construction scripts under `scripts/` document and audit how the canonical artifacts were assembled from experiment-time source files.

RQ4 assignment manifests preserve the exact historical splits and artifact-compatible subsets used in the experiments. The larger derived files are deliberately reconstructed rather than duplicated in version control.

Third-party datasets and source code remain external to this directory. Their exact upstream provenance should be documented and pinned under `external/`.

## Generated files

Materialized RQ4 files are derivative artifacts and do not need to be versioned.

Recommended `.gitignore` entries:

```gitignore
/data/rq4/cross_project/folds/
/data/rq4/within_corpus/split/
/data/rq4/artifact_compatible/hqcm/materialized/
/data/rq4/artifact_compatible/flaccs/materialized/
/external/hqcm/codefuse-hqcm/
```
