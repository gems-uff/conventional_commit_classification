# RQ4.2 HQCM-compatible subset

Subset of `data/pre_cutoff/commits.csv.gz` (2,132 commits, 30 projects) used by
`config/experiments/rq4_2/artifact_hqcm.yaml`. This is not the original HQCM
dataset; for that, see `data/external/hqcm/README.md`.

Labels: `build`, `cicd`, `docs`, `feat`, `fix`, `refactor`, `style`, `test`
(`cicd` corresponds to `ci` in the canonical dataset).

## Files

`assignments.csv`: one row per commit.

| Column | Content |
|---|---|
| `protocol` | `hqcm_compatible`. |
| `project` | Project identifier. |
| `commit_hash` | Git commit SHA-1. |
| `canonical_target` | Ground-truth class in the canonical dataset. |
| `protocol_target` | Label in the HQCM label space. |
| `source_index` | Row index in the HQCM-compatible file. |
| `protocol_record_id` | Record identifier in the HQCM-compatible file (not a Git hash). |
| `message_tokens` | Token count of the processed commit message. |
| `diff_tokens` | Token count of the Git diff. |
| `total_tokens_original` | Combined token count before HQCM formatting. |
| `prompt_tokens` | Token count after HQCM formatting. |
| `total_tokens` | Final token count. |

## Materialize the subset (optional)

```bash
python3 scripts/rq4_2/materialize_hqcm_artifact.py \
  --dataset data/pre_cutoff/commits.csv.gz \
  --assignments data/rq4_2/artifact_compatible/hqcm/assignments.csv \
  --csv-output data/rq4_2/artifact_compatible/hqcm/materialized/our_hqcm_len1024.csv \
  --json-output data/rq4_2/artifact_compatible/hqcm/materialized/our_hqcm_len1024.json
```

Writes a CSV and a JSON file in the HQCM input format:

```text
Commit summary:
<processed commit message>

Git diff:
<git diff>
```

The script stops with an error if the assignments do not match the canonical
dataset.
