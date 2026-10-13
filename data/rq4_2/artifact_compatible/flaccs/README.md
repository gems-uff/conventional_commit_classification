# FLACCS Artifact-Compatible Dataset

This directory contains the compact manifest required to reconstruct the FLACCS-compatible dataset used in RQ4.

The artifact is derived from the canonical pre-cutoff dataset:

`data/pre_cutoff/commits.csv.gz`

The historical FLACCS-compatible file contains a protocol-specific subset of the canonical commits together with token-budget metadata and diff statistics. The replication package stores the subset definition and protocol-specific metadata rather than duplicating all commit contents.

## Artifact size

The FLACCS-compatible subset contains:

- 2,573 commits;
- 30 projects; and
- only records retained by the historical FLACCS artifact preparation procedure.

The manifest records the exact selected commits. Membership should therefore be reproduced from `assignments.csv` rather than inferred again solely from the 1,024-token threshold.

## Files

### `assignments.csv`

This is the authoritative definition of the FLACCS-compatible subset.

Columns:

| Column | Description |
|---|---|
| `protocol` | Protocol identifier (`flaccs_compatible`). |
| `project` | Source project. |
| `commit_hash` | Git commit SHA-1. |
| `canonical_target` | Ground-truth label in the canonical study taxonomy. |
| `protocol_target` | Label stored in the FLACCS-compatible artifact. |
| `message_tokens` | Historical token count for the processed commit message. |
| `diff_tokens` | Historical token count for the Git diff. |
| `total_tokens_original` | Historical combined token count before FLACCS formatting. |
| `prompt_tokens` | Historical FLACCS prompt token count. |
| `label_tokens` | Historical token count of the target label. |
| `total_tokens_with_label` | Historical total token count including the label. |
| `fits_1024` | Whether the historical FLACCS-formatted record fits the 1,024-token constraint. |

## Materializing the FLACCS artifact

The full historical CSV is not stored as a primary data file because the commit message and diff contents already exist in the canonical pre-cutoff dataset.

It can be reconstructed with:

```bash
python3 scripts/rq4_2/materialize_flaccs_artifact.py \
  --dataset data/pre_cutoff/commits.csv.gz \
  --assignments data/rq4_2/artifact_compatible/flaccs/assignments.csv \
  --output data/rq4_2/artifact_compatible/flaccs/materialized/our_flaccs_10_len1024.csv
```

The materializer restores the historical schema, including:

- project and platform;
- repository identifier;
- commit hash and date;
- target label;
- breaking-change flag;
- original and processed commit messages;
- Git diff;
- historical token counts;
- the 1,024-token compatibility flag; and
- file/change statistics derived from the Git diff.

The following fields are deterministically reconstructed from the canonical diff:

`files_changed`, `insertions`, `deletions`, `files_formatted`, and `stats_summary`.

## Validation

Before generating the artifact, the script verifies that:

- every assignment references a commit in `data/pre_cutoff/commits.csv.gz`;
- `canonical_target` agrees with the canonical dataset;
- commit identifiers are unique;
- the manifest contains the expected 2,573 records;
- all selected records have `fits_1024=True`; and
- the stored total token count with label does not exceed 1,024.

Because the file-stat fields are reconstructed from the Git diff, one development-time audit against the historical FLACCS CSV is recommended before publishing the replication package:

```bash
python3 scripts/rq4_2/materialize_flaccs_artifact.py \
  --dataset data/pre_cutoff/commits.csv.gz \
  --assignments data/rq4_2/artifact_compatible/flaccs/assignments.csv \
  --output /tmp/our_flaccs_10_len1024.csv \
  --reference-csv /path/to/historical/our_flaccs_10_len1024.csv
```

When `--reference-csv` is supplied, the script compares the reconstructed logical CSV values with the historical artifact and aborts if any output column differs.

The reference file is not required for normal replication.

## Authoritative inputs

The FLACCS-compatible artifact is reproducible from:

```text
data/pre_cutoff/commits.csv.gz
data/rq4_2/artifact_compatible/flaccs/assignments.csv
scripts/rq4_2/materialize_flaccs_artifact.py
```

Generated files under `materialized/` are derivative artifacts and do not need to be versioned.
