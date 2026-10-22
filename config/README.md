# Configuration

| File | Content |
|---|---|
| `models.yaml` | LLMs run on Ollama: tag, digest, size, quantization, reported knowledge cutoff. |
| `inference.yaml` | Ollama execution settings shared by all experiments: input truncation, context size, generation length, retries, concurrency, parser. |
| `evaluation.yaml` | Label taxonomy, consensus, and metric settings. |
| `experiments/` | One file per experiment: dataset, models, prompts, representation, temperature, seeds, request timeouts. |
| `training/` | Settings of the trainable RQ4.1 baselines (TF-IDF + SVM, BiLSTM, CodeBERT). |

Experiment files:

| RQ | File | Runner |
|---|---|---|
| RQ1 | `experiments/rq1.yaml` | `run_ollama_experiment.py --rq rq1` |
| RQ2 | `experiments/rq2.yaml` | `run_ollama_experiment.py --rq rq2` |
| RQ3.1 | `experiments/rq3_1.yaml` | `run_ollama_experiment.py --rq rq3_1` |
| RQ3.2 | `experiments/rq3_2.yaml` | `run_ollama_experiment.py --rq rq3_2` |
| RQ3.2 temporal robustness | `experiments/rq3_2_temporal_robustness.yaml` | `run_ollama_experiment.py --rq rq3_2_temporal` |
| RQ4.1 | `experiments/rq4_1/cross_project.yaml`, `experiments/rq4_1/within_corpus.yaml` | `run_ollama_experiment.py --rq rq4 --stage {cross_project,within_corpus}` (GPT-OSS), `run_baselines_experiment.py --stage {cross_project,within_corpus}` (SVM/BiLSTM/CodeBERT) |
| RQ4.2 | `experiments/rq4_2/artifact_flaccs.yaml`, `experiments/rq4_2/artifact_hqcm.yaml` | `run_ollama_experiment.py --rq rq4 --stage {artifact_flaccs,artifact_hqcm}` (GPT-OSS), `run_flaccs_experiment.py` (FLACCS) |
| RQ4.2 original HQCM test set | `experiments/rq4_2/hqcm_original.yaml` | `run_ollama_experiment.py --rq rq4 --stage hqcm_original` (GPT-OSS; see `data/external/hqcm/README.md`) |

Prompts are referenced by `prompt_id` and resolved through `prompts/manifest.yaml`.
