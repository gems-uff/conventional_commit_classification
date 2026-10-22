# RQ4.1 cross-project folds

Five folds over `data/pre_cutoff/commits.csv.gz`, used by
`config/experiments/rq4_1/cross_project.yaml`.

## Files

`assignments.csv`: one row per (commit, fold).

| Column | Content |
|---|---|
| `protocol` | `cross_project`. |
| `fold` | 1–5. |
| `split` | `train` or `test`. |
| `project` | Project identifier. |
| `commit_hash` | Git commit SHA-1. |
| `target` | Ground-truth class. |

`fold_summary.csv`: number of train/test commits and projects per fold.

## Materialize the folds (optional)

```bash
python3 scripts/rq4_1/materialize_cross_project_folds.py \
  --dataset data/pre_cutoff/commits.csv.gz \
  --assignments data/rq4_1/cross_project/assignments.csv \
  --fold-summary data/rq4_1/cross_project/fold_summary.csv \
  --output-dir data/rq4_1/cross_project/folds
```

Writes `fold_<N>_train.csv` and `fold_<N>_test.csv` (N = 1–5) with the columns
of the canonical dataset. The script stops with an error if the assignments do
not match the canonical dataset or `fold_summary.csv`.
