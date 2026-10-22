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

## Changes applied to the pinned revision

Apply `changes.patch` in the checkout before running the HQCM scripts:

```bash
cd data/external/hqcm/codefuse-hqcm
git apply ../changes.patch
```

The patch changes `hqcm/xdata.py` and `hqcm/gen.py`. Training (`sft.py`), the
model architecture, the class definitions, and the model weights are not
changed. Do not edit `changes.patch` with an editor that strips trailing
whitespace.

### `hqcm/xdata.py`

`xdata.py` reads `args.fewshot` (chcl and coderef tasks), but argparse exposes
the `--few-shot` option as `args.few_shot`.

### `hqcm/gen.py`

Inference limits set to a 1,024-token budget: up to 1,016 prompt tokens and up
to 8 response tokens.

| Change | Original | Used in this study |
|---|---|---|
| Response limit (`max_new_tokens`) | 512 tokens | 8 tokens |
| `SFTModel` default `max_prompt_length` | 4,096 tokens | 1,016 tokens |
| `max_prompt_length` passed when instantiating `SFTModel` | 512 tokens | 1,016 tokens |
| Prompt tokenization in `SFTModel.query` | no truncation | `truncation=True`, `max_length=1016` |
| Limits stored in the model object | — | `self.max_prompt_length`, `self.max_response_length` |

The 1,024-token budget matches the `--max-length 1024` used for fine-tuning
(below). In the test split of the chcl variant (518 examples), 10 prompts exceed
1,016 tokens and are truncated (indices 3, 92, 116, 174, 191, 222, 425, 471,
481, 508).

## Fine-tuning and inference used in this study

Run from the patched checkout. The fine-tuning arguments are those recorded
for the adapter used in the study (`outputs/chcl_llama2_7b_len1024`).

```bash
# Change-classification (chcl) variant of the HQCM dataset
python -m hqcm.xdata --task chcl --output ./dataset/chcl ./dataset/

# LoRA fine-tuning of Llama 2 7B
python -m hqcm.sft                  \
    --seed 0                        \
    --learning-rate '2e-4'          \
    --num-epochs 5                  \
    --batch-size 1                  \
    --micro-batch-size 1            \
    --lora-rank 64                  \
    --lora-alpha 16                 \
    --lora-dropout 0.1              \
    --quantization '-1'             \
    --percentage 100                \
    --dataset ./dataset/chcl        \
    --split 'train'                 \
    --max-length 1024               \
    --output ./outputs/chcl_llama2_7b_len1024 \
    meta-llama/Llama-2-7b-hf

# Classification of the original test set (518 examples)
PYTHONPATH="$PWD" python -m hqcm.gen  \
    --dataset ./dataset/chcl        \
    --split test                    \
    --output ./outputs/chcl_results_len1024.json \
    --temperature 0                 \
    ./outputs/chcl_llama2_7b_len1024
```

`--max-length 1024` differs from the example in the HQCM README (512).
`meta-llama/Llama-2-7b-hf` requires accepting Meta's license on Hugging Face.

## Comparison on the original HQCM test set (RQ4.2)

From the repository root, after the classification above:

```bash
# 1. Test set in the canonical format (written to data/external/hqcm/materialized/, ignored by git)
python scripts/rq4_2/materialize_hqcm_original_testset.py

# 2. GPT-OSS on the 518 examples
python scripts/run_ollama_experiment.py --rq rq4 --stage hqcm_original

# 3. Predictions and metrics of both models
python scripts/rq4_2/build_hqcm_original_comparison.py --from runs \
  --gpt-predictions runs/rq4_hqcm_original/gpt_oss_20b/gpt_oss_20b/predictions.csv.gz \
  --hqcm-results data/external/hqcm/codefuse-hqcm/outputs/chcl_results_len1024.json \
  --output-dir runs/hqcm_original --check
```

Step 3 writes the files of `results/rq4_2/related_work_artifacts/hqcm_original_dataset/`
(bootstrap and randomization with 20,000 repetitions, seed 20260807). With
`--from results` instead of `--from runs`, it recomputes the metrics from the
predictions in `results/`.

## Related files in this package

- `data/rq4_2/artifact_compatible/hqcm/`: subset of our dataset in the HQCM
  input format.
- `results/rq4_2/related_work_artifacts/hqcm/` and
  `results/rq4_2/related_work_artifacts/hqcm_original_dataset/`: HQCM
  predictions and metrics.
