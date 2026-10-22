# Ollama setup

Steps to install Ollama and the models used by `scripts/run_ollama_experiment.py`.

## Models per RQ

| RQ | Models (IDs in `config/models.yaml`) |
|---|---|
| RQ1 | `gemma_3_27b`, `gpt_oss_20b`, `granite_3_3_8b`, `llama_3_2_vision_11b`, `magistral_24b`, `mistral_small_3_2_24b`, `phi4_reasoning_plus_14b`, `qwq_32b` |
| RQ2 | all 13 models in `config/models.yaml` |
| RQ3.1, RQ3.2, RQ4 | `gpt_oss_20b` |

FLACCS and HQCM do not run on Ollama (see the end of this file).

## 1. Install Ollama

- **Windows**: `winget install --id Ollama.Ollama --accept-package-agreements --accept-source-agreements`
- **macOS / Linux**: follow <https://ollama.com/download>.

Check that the service is running:

```bash
curl http://localhost:11434/api/tags
```

A JSON response (e.g. `{"models":[]}`) means it is running. Otherwise, start
it with `ollama serve`.

## 2. Pull the models

Pull each model by its `ollama_tag` in `config/models.yaml`:

```bash
ollama pull gpt-oss:20b                       # gpt_oss_20b             (RQ1–RQ4)
ollama pull granite3.3:8b                     # granite_3_3_8b          (RQ1, RQ2)
ollama pull gemma3:27b                        # gemma_3_27b             (RQ1, RQ2)
ollama pull llama3.2-vision:11b               # llama_3_2_vision_11b    (RQ1, RQ2)
ollama pull magistral:latest                  # magistral_24b           (RQ1, RQ2)
ollama pull mistral-small3.2:24b              # mistral_small_3_2_24b   (RQ1, RQ2)
ollama pull phi4-reasoning:plus               # phi4_reasoning_plus_14b (RQ1, RQ2)
ollama pull qwq:latest                        # qwq_32b                 (RQ1, RQ2)
ollama pull deepseek-r1:8b-0528-qwen3-q4_K_M  # deepseek_r1_8b_0528_qwen3 (RQ2)
ollama pull devstral:latest                   # devstral_small_24b      (RQ2)
ollama pull aya-expanse:32b                   # aya_expanse_32b         (RQ2)
ollama pull qwen3-coder:30b                   # qwen3_coder_30b         (RQ2)
ollama pull qwen3:32b                         # qwen3_32b               (RQ2)
```

## 3. Check the model digests

Before a real run, `scripts/run_ollama_experiment.py` compares the installed
model digest with the `digest` field in `config/models.yaml` and stops if they
differ:

```
RuntimeError: Installed model digest mismatch for gpt-oss:20b: <installed digest>
```

An Ollama tag can point to a different build over time. If the digests differ:

- try to pull the recorded build by digest: `ollama pull <tag>@sha256:<digest>`; or
- update `digest` in `config/models.yaml` to run with the installed build.
  Results obtained with a different build may differ from those in `results/`.

## 4. Memory

Models in the 20B+ class need enough free RAM when they are loaded, not only
enough VRAM. An out-of-memory failure at load time is recorded in
`attempts.journal.jsonl` as:

```
ResponseError: llama-server process has terminated: exit status 1:
ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size ...
```

Free memory and run again with `--resume`.

## 5. Run

Continue with section 3 of `README.md`. Start with `--dry-run`.

## FLACCS and HQCM

- **FLACCS** (`0x404/ccs-code-llama-7b`) is downloaded from the Hugging Face Hub
  by `scripts/run_flaccs_experiment.py` on its first run. It requires `torch`
  and `transformers` and a GPU.
- **CodeFuse HQCM** is not redistributed here; its model is obtained with the HQCM
  fine-tuning scripts. See `data/external/hqcm/README.md`.
