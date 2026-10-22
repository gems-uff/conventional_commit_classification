# Replication package

Code, data, configuration, and results needed to reproduce RQ1–RQ4.

## Repository layout

```text
config/       experiment, inference, evaluation, model, and training configuration
data/         canonical datasets and RQ4 split/subset definitions (see data/README.md)
prompts/      prompt files, indexed by prompts/manifest.yaml
scripts/      runners and data materialization scripts
src/          shared implementation used by the runners
results/      predictions, metrics, and analysis notebooks for each RQ
tests/        test suite
environment.txt   hardware and Ollama version of the execution machine
```

## 1. Install

```bash
python -m pip install -r requirements.txt
```

The LLM experiments run on Ollama: see [`OLLAMA_SETUP.md`](OLLAMA_SETUP.md)
to install it and pull the models. FLACCS, BiLSTM, and CodeBERT require a
GPU-capable machine with `torch`.

Check the installation:

```bash
python -m pip install pytest
python -m pytest
```

## 2. Data

The canonical datasets are in `data/pre_cutoff/` and `data/post_cutoff/`.
Verify them with:

```bash
(cd data/pre_cutoff && sha256sum -c commits.csv.gz.sha256)
(cd data/post_cutoff && sha256sum -c commits.csv.gz.sha256)
```

The runners read the RQ4 `assignments.csv` files directly; materializing the
RQ4 splits as separate files is optional (see `data/README.md`).

## 3. Run the experiments

All runners write to `runs/` by default (`--output-dir` to change it) and refuse
to write inside `results/`. Each runner accepts:

- `--list-jobs`: print the jobs and exit;
- `--dry-run`: validate configuration, data, and prompts without running models;
- `--limit N`: run only the first N commits (use a separate `--output-dir`);
- `--resume`: continue an interrupted run without repeating completed calls.

Each job directory contains `predictions.csv.gz` (one consensus prediction per
commit) and `run_manifest.json`. Keep the `*.journal.jsonl` files: `--resume`
depends on them.

### RQ1–RQ3 and the GPT-OSS conditions of RQ4 (Ollama)

```bash
python scripts/run_ollama_experiment.py --rq rq1
python scripts/run_ollama_experiment.py --rq rq2
python scripts/run_ollama_experiment.py --rq rq3_1
python scripts/run_ollama_experiment.py --rq rq3_2
python scripts/run_ollama_experiment.py --rq rq3_2_temporal
python scripts/run_ollama_experiment.py --rq rq4
```

Options:

- `--rq rq3_2 --stage <factor>` runs one factor: `factor_1_definitions`,
  `factor_2_heuristics`, `factor_3_cot_and_few_shot`, or `factor_4_abstention`.
- In `--rq rq3_2`, the Factor 4 prompts (`p_exclude_low`, `p_high_only`) apply
  the abstention policies set by `abstain_on` in `config/experiments/rq3_2.yaml`:
  a seed whose confidence is in `abstain_on` abstains, and two abstentions give
  `<abstained>`. Their `predictions.csv.gz` also has `confidence_<seed>` columns.
- `--rq rq3_2_temporal` runs the baseline and final prompts on the pre-cutoff
  dataset (temporal robustness); the post-cutoff conditions are the `--rq rq3_2`
  runs of `p0_baseline` and `p_code_disambiguation`.
- `--rq rq4 --stage <name>` runs one RQ4 experiment: `cross_project`,
  `within_corpus`, `artifact_flaccs`, or `artifact_hqcm`. Without `--stage`,
  all four run in sequence.
- `--model`, `--condition`, `--prompt`, and `--project` restrict the jobs.
- `--experiment <yaml>` runs any experiment file under `config/experiments/`.

Before a real run, the runner checks that the installed model digest matches
`config/models.yaml` (see `OLLAMA_SETUP.md`, step 3).

Output: `runs/<experiment_id>/<condition>/<model>/`.

### RQ4.1 trainable baselines (TF-IDF + SVM, BiLSTM, CodeBERT)

```bash
python scripts/run_baselines_experiment.py --stage cross_project
python scripts/run_baselines_experiment.py --stage within_corpus
```

`--model {svm,bilstm,codebert}` and `--fold N` restrict the jobs.

Output: `runs/<experiment_id>/<condition>/fold_<N>/`.

### RQ4.2 FLACCS artifact-compatible condition

```bash
python scripts/run_flaccs_experiment.py
```

The model `0x404/ccs-code-llama-7b` is downloaded from the Hugging Face Hub
on the first run.

Output: `runs/<experiment_id>/flaccs/`.

### RQ4.2 HQCM

The CodeFuse HQCM code, fine-tuning scripts, and original dataset are not
redistributed in this package; they are obtained from the HQCM authors'
repository, which does not include a fine-tuned model. The model is obtained by
running the HQCM fine-tuning scripts. See `data/external/hqcm/README.md`. The HQCM
predictions used in the study are in `results/rq4_1/` and
`results/rq4_2/related_work_artifacts/hqcm_original_dataset/`.

## 4. Analyses

Each `results/<rq>/` directory contains:

- `predictions/`: consensus predictions per commit;
- `metrics/`: per-project Macro-F1 tables read by the notebooks;
- `analysis/*.ipynb`: statistical analyses and figures.

Build the `metrics/` tables from predictions:

```bash
# from a new run (default runs directory: runs/)
python scripts/build_metrics.py --rq rq1 --from runs --output-dir runs/metrics
# from the predictions in results/, comparing with the tables in results/
python scripts/build_metrics.py --rq all --from results --output-dir runs/metrics --check
```

`--rq` is one of `rq1`, `rq2`, `rq3_1`, `rq3_2`, `rq4_1`, `rq4_2`, or `all`. The
tables are written under `--output-dir` with the same paths as in `results/`.
For RQ4, approaches not found in the runs directory are taken from `results/`.

Optional: the class-confusion analysis of `p_header_structured` in
`results/rq3_2/error_analysis/` is built with

```bash
python scripts/rq3_2/build_error_analysis.py --output-dir runs/error_analysis --check
```

(`--predictions <file>` to use a new run of that prompt).

Run a notebook with Jupyter (`python -m pip install jupyter`), using
*Run All*. The notebooks read the CSV files in the `metrics/` directory next
to them; to analyze a new run, replace those files with the tables built by
`build_metrics.py`.

## Configuration

See `config/README.md`. Experiment files are under `config/experiments/`;
prompts are referenced by ID through `prompts/manifest.yaml` (see
`prompts/README.md`).

## License and citation

Code is under the MIT License and data under CC BY 4.0; commit content in
`data/` remains under the license of each source repository (see `LICENSE`).
To cite this package, see `CITATION.cff`.
