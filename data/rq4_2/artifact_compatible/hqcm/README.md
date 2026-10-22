# HQCM Artifact-Compatible Dataset

This directory contains the compact manifest required to reconstruct the HQCM-compatible dataset used in RQ4.

The artifact is derived from the canonical pre-cutoff dataset:

`data/pre_cutoff/commits.csv.gz`

It is **not** the original HQCM dataset. The original third-party HQCM artifact is documented separately under:

`data/external/hqcm/`

Official upstream repository:

https://github.com/codefuse-ai/codefuse-hqcm

Pinned upstream revision used in this study:

```text
ea7e7fab1bca176d9afd77350496649d7ca4743f
```

## Purpose

This subset was prepared from our own corpus so that HQCM could be evaluated under artifact-compatible input, taxonomy, and token-budget constraints.

Instead of storing another full copy of commit messages and diffs, this directory stores only the historical subset definition and protocol-specific metadata required to recreate the HQCM-compatible input artifact.

## Artifact size

The HQCM-compatible subset contains:

- 2,132 commits;
- 30 projects; and
- 8 protocol-specific labels.

The labels represented in the historical artifact are:

`build`, `cicd`, `docs`, `feat`, `fix`, `refactor`, `style`, and `test`.

The manifest explicitly preserves the mapping between the canonical study label and the HQCM-specific label. In particular, the canonical `ci` label is represented as `cicd` in the HQCM artifact.

## Files

### `assignments.csv`

This file is the authoritative definition of the HQCM-compatible subset and its protocol-specific metadata.

| Column | Description |
|---|---|
| `protocol` | Protocol identifier (`hqcm_compatible`). |
| `project` | Source project. |
| `commit_hash` | Original Git commit SHA-1. |
| `canonical_target` | Ground-truth label in the canonical study taxonomy. |
| `protocol_target` | Label used by the HQCM-compatible artifact. |
| `source_index` | Historical row index in the HQCM artifact. |
| `protocol_record_id` | Synthetic record identifier used by the HQCM artifact. |
| `message_tokens` | Historical token count for the processed commit message. |
| `diff_tokens` | Historical token count for the Git diff. |
| `total_tokens_original` | Historical combined token count before HQCM prompt formatting. |
| `prompt_tokens` | Historical token count after HQCM prompt formatting. |
| `total_tokens` | Historical final token count recorded for the HQCM-compatible item. |

`protocol_record_id` is not a Git hash. The canonical identity of each record is `project` + `commit_hash`.

## Materializing the HQCM-compatible artifact

The full historical CSV and JSON can be reconstructed with:

```bash
python3 scripts/rq4_2/materialize_hqcm_artifact.py \
  --dataset data/pre_cutoff/commits.csv.gz \
  --assignments data/rq4_2/artifact_compatible/hqcm/assignments.csv \
  --csv-output data/rq4_2/artifact_compatible/hqcm/materialized/our_hqcm_len1024.csv \
  --json-output data/rq4_2/artifact_compatible/hqcm/materialized/our_hqcm_len1024.json
```

The materializer reconstructs the HQCM input representation as:

```text
Commit summary:
<processed commit message>

Git diff:
<git diff>
```

It also restores the historical schema, including the synthetic record ID, label mapping, token counts, repository identifier, commit date, breaking-change flag, processed message, and Git diff.

## Validation

Before generating the artifact, the materialization script verifies that:

- every assignment references a commit in `data/pre_cutoff/commits.csv.gz`;
- `canonical_target` agrees with the canonical dataset;
- commit identifiers are unique;
- synthetic HQCM record identifiers are unique;
- `source_index` values are unique and contiguous;
- the manifest contains the expected 2,132 records; and
- the stored HQCM token total does not exceed 1,024.

The script can optionally compare the reconstructed CSV against the historical artifact used during experiment development through `--reference-csv`.

## Relationship to the original HQCM dataset

This directory contains **our data adapted to the HQCM artifact**, not a copy of the original HQCM dataset.

For the original HQCM source code and dataset used in the separate original-artifact evaluation, see:

`data/external/hqcm/README.md`

The upstream artifact can be retrieved at its pinned revision with:

```bash
python3 data/external/hqcm/download_hqcm.py
```

## Authoritative inputs

The HQCM-compatible artifact is reproducible from:

```text
data/pre_cutoff/commits.csv.gz
data/rq4_2/artifact_compatible/hqcm/assignments.csv
scripts/rq4_2/materialize_hqcm_artifact.py
```

Generated files under `materialized/` are derivative artifacts and do not need to be versioned.
