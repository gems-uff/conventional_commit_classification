# RQ4.2 FLACCS-compatible subset

Subset of `data/pre_cutoff/commits.csv.gz` (2,573 commits, 30 projects) used by
`config/experiments/rq4_2/artifact_flaccs.yaml`.

## Files

`assignments.csv`: one row per commit.

| Column | Content |
|---|---|
| `protocol` | `flaccs_compatible`. |
| `project` | Project identifier. |
| `commit_hash` | Git commit SHA-1. |
| `canonical_target` | Ground-truth class in the canonical dataset. |
| `protocol_target` | Label in the FLACCS label space. |
| `message_tokens` | Token count of the processed commit message. |
| `diff_tokens` | Token count of the Git diff. |
| `total_tokens_original` | Combined token count before FLACCS formatting. |
| `prompt_tokens` | Token count of the FLACCS prompt. |
| `label_tokens` | Token count of the target label. |
| `total_tokens_with_label` | Total token count including the label. |
| `fits_1024` | Whether the formatted record fits in 1,024 tokens. |

## Materialize the subset (optional)

```bash
python3 scripts/rq4_2/materialize_flaccs_artifact.py \
  --dataset data/pre_cutoff/commits.csv.gz \
  --assignments data/rq4_2/artifact_compatible/flaccs/assignments.csv \
  --output data/rq4_2/artifact_compatible/flaccs/materialized/our_flaccs_10_len1024.csv
```

Writes one CSV with the commit fields, token counts, and per-file diff
statistics. The script stops with an error if the assignments do not match the
canonical dataset.
