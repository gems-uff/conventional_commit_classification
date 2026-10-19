# Ollama setup — from installation to first run

This is the step the rest of the package assumes is already done: installing
Ollama itself and pulling the exact models the study used. `README.md`
documents how to *run* `scripts/run_ollama_experiment.py` once this is in
place; this file documents how to get to that point.

## Which models you actually need

Not every research question needs every model. Pull only what the RQ you are
reproducing requires:

| RQ | Models needed |
|---|---|
| RQ1 (temporal alignment) | 8 models: `gemma_3_27b`, `gpt_oss_20b`, `granite_3_3_8b`, `llama_3_2_vision_11b`, `magistral_24b`, `mistral_small_3_2_24b`, `phi4_reasoning_plus_14b`, `qwq_32b` |
| RQ2 (model/temperature comparison) | all 13 models in `config/models.yaml` |
| RQ3.1, RQ3.2, RQ4 (cross-project, within-corpus, artifact-compatible GPT-OSS side) | `gpt_oss_20b` only — the model RQ2 selected |

FLACCS and CodeFuse HQCM are **not** Ollama models — see
[Non-Ollama models](#non-ollama-models-flaccs-and-hqcm) below.

## 1. Install Ollama

- **Windows**: `winget install --id Ollama.Ollama --accept-package-agreements --accept-source-agreements`
  (the installer also registers Ollama to start automatically; you should not
  need to start it by hand — `ollama list` or `ollama --version` working in a
  new shell confirms it did).
- **macOS / Linux**: follow the official instructions at
  <https://ollama.com/download>. On Linux, the installer script sets up a
  `systemd` service; `sudo systemctl status ollama` confirms it is running.
- Confirm the HTTP API is up before continuing:

  ```bash
  curl http://localhost:11434/api/tags
  ```

  An empty `{"models":[]}` (or a non-empty list) means the service is up. A
  connection error/timeout means Ollama is not running — start it
  (`ollama serve`, or restart the installed service) before pulling anything.

## 2. Pull the models

Each model is pulled by its `ollama_tag` from `config/models.yaml`, not by the
internal `model_id` key used elsewhere in this package:

```bash
ollama pull gpt-oss:20b                # gpt_oss_20b     (RQ1/RQ2/RQ3/RQ4)
ollama pull granite3.3:8b              # granite_3_3_8b  (RQ1/RQ2)
ollama pull gemma3:27b                 # gemma_3_27b     (RQ1/RQ2)
ollama pull llama3.2-vision:11b        # llama_3_2_vision_11b (RQ1/RQ2)
ollama pull magistral:latest           # magistral_24b   (RQ1/RQ2)
ollama pull mistral-small3.2:24b       # mistral_small_3_2_24b (RQ1/RQ2)
ollama pull phi4-reasoning:plus        # phi4_reasoning_plus_14b (RQ1/RQ2)
ollama pull qwq:latest                 # qwq_32b         (RQ1/RQ2)
# RQ2-only (not used by RQ1):
ollama pull deepseek-r1:8b-0528-qwen3-q4_K_M  # deepseek_r1_8b_0528_qwen3
ollama pull devstral:latest             # devstral_small_24b
ollama pull aya-expanse:32b             # aya_expanse_32b
ollama pull qwen3-coder:30b             # qwen3_coder_30b
ollama pull qwen3:32b                   # qwen3_32b
```

Downloads range from ~5 GB (8B-class models, Q4_K_M) to ~20 GB (the 30B+
models). Pull only the subset your RQ needs — see the table above.

## 3. Verify the installed model matches the frozen digest

`scripts/run_ollama_experiment.py` checks the installed model's digest against
`config/models.yaml` before every real (non-`--dry-run`) execution, and
**refuses to run** if they differ — this is deliberate, not a bug:

```
RuntimeError: Installed model digest mismatch for gpt-oss:20b: <installed digest>
```

### Why this can happen: Ollama tags are not immutable

Confirmed hands-on while validating this package (2026-10-13): pulling
`gpt-oss:20b` fresh produced digest `f38aa0c53da5...`, not the
`aa4295ac10c3...` recorded in `config/models.yaml` from the study's original
execution. The Ollama registry had republished different content under the
same tag sometime between the study's execution and this pull. The same risk
applies to every model in the registry, especially ones tagged `:latest`
(`devstral`, `magistral`, `qwq`) — those are the most likely to drift.

**If you hit this mismatch**, you have two options, and the choice matters —
pick deliberately, don't just silence the check:

1. **Accept today's build and say so.** Update the `digest:` field in
   `config/models.yaml` for that model, and record in your own notes (or this
   package's `TODO.md`) that you ran against a different build than the one
   backing the published results. Treat any numbers you get as *not*
   byte-for-byte reproducing the paper — a soft replication, not a strict one.
2. **Try to get the original build back.** Some registries allow pulling by
   digest instead of tag (`ollama pull <tag>@sha256:<digest>`); whether the
   *original* digest is still resolvable depends on whether Ollama's registry
   kept it after republishing the tag. This is not guaranteed to work.

Do not edit `config/models.yaml`'s digest as a permanent fix without one of
the two choices above — that field is the audit trail of which exact model
build produced the published results.

## 4. Hardware sizing (what actually fits)

Confirmed hands-on on a 16 GB RAM / RTX 3060 Ti (8 GB VRAM) machine:

- `granite3.3:8b` (~5 GB download): ran without issue.
- `gpt-oss:20b` (~13 GB download, MXFP4): **failed with an out-of-memory error
  from `llama-server`** when only ~1.4 GB of system RAM was free (other
  processes were using the rest), even though the GPU itself had headroom
  (6+ GB VRAM free). The same model **succeeded** once enough RAM was freed
  (~7 GB free was enough). The failure was about free RAM at load time, not
  an inherent incompatibility with the hardware.

Takeaway: before running a model in the 20B+ class, check free RAM (not just
total RAM), and close other memory-heavy applications if it's tight. A failed
pull/run due to OOM looks like this in the attempt log
(`technical_detail` in `attempts.journal.jsonl`):

```
ResponseError: llama-server process has terminated: exit status 1:
ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size ...
```

That specific error is a memory problem, not a configuration problem — the
pipeline and canonical parser are not involved yet when it happens (the call
to Ollama itself failed).

## 5. Run

Once Ollama is installed, the model(s) are pulled, and digests match (or you
made a deliberate decision in step 3), continue with the "Ollama execution
and resume" section of `README.md` (`scripts/run_ollama_experiment.py --rq ...
--dry-run` first, then without `--dry-run`).

## Non-Ollama models: FLACCS and HQCM

- **FLACCS** (`0x404/ccs-code-llama-7b`) is downloaded automatically from the
  Hugging Face Hub the first time `scripts/run_flaccs_experiment.py` runs
  (needs `torch`/`transformers` from `requirements.txt`, and a GPU is strongly
  recommended — it is not an Ollama model and is not in `config/models.yaml`).
- **CodeFuse HQCM** has no runner in this package by design — see `README.md`
  ("Accepted limitation, not a gap") and `TODO.md` item 1 for why.
