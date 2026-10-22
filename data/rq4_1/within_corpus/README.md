# RQ4.1 within-corpus split

Train/test split of `data/pre_cutoff/commits.csv.gz` (6,527 train and 1,632
test commits), used by `config/experiments/rq4_1/within_corpus.yaml`.

## Files

`assignments.csv`: one row per commit.

| Column | Content |
|---|---|
| `protocol` | `within_corpus`. |
| `fold` | `0`. |
| `split` | `train` or `test`. |
| `project` | Project identifier. |
| `commit_hash` | Git commit SHA-1. |
| `target` | Ground-truth class. |

## Materialize the split (optional)

```bash
python3 scripts/rq4_1/materialize_within_corpus_split.py \
  --dataset data/pre_cutoff/commits.csv.gz \
  --assignments data/rq4_1/within_corpus/assignments.csv \
  --output-dir data/rq4_1/within_corpus/split
```

Writes `train.csv` and `test.csv` with the columns of the canonical dataset, in
the order of `assignments.csv`. The script stops with an error if the
assignments do not match the canonical dataset.
