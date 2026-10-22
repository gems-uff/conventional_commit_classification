# CodeFuse HQCM (external)

The HQCM code, fine-tuning scripts, and original dataset are not included in
this package. They come from the HQCM authors' repository and are subject to its
license. The repository does not include a fine-tuned model; the model is
obtained by running its fine-tuning scripts.

- repository: https://github.com/codefuse-ai/codefuse-hqcm
- revision used: `ea7e7fab1bca176d9afd77350496649d7ca4743f`
- paper: *Understanding Code Changes Practically with Small-Scale Language Models* (ASE 2024)

To reproduce HQCM, follow the instructions in that repository.

## Download the pinned revision

From the repository root:

```bash
python3 data/external/hqcm/download_hqcm.py
```

This clones the repository into `data/external/hqcm/codefuse-hqcm/` (ignored by
git) and checks out the revision above. Manual equivalent:

```bash
git clone https://github.com/codefuse-ai/codefuse-hqcm.git
cd codefuse-hqcm
git checkout ea7e7fab1bca176d9afd77350496649d7ca4743f
```

## Related files in this package

- `data/rq4_2/artifact_compatible/hqcm/`: subset of our dataset in the HQCM
  input format.
- `results/rq4_2/related_work_artifacts/hqcm/` and
  `results/rq4_2/related_work_artifacts/hqcm_original_dataset/`: HQCM
  predictions and metrics.
