# Shared `src/` package for the replication repository

This package is the proposed shared implementation layer behind RQ1–RQ4. It was
constructed from the final protocol/configuration and the corrected runners, not
from the legacy parsed prediction fields.

## Design rule

**Runners orchestrate; `src/` owns scientific behavior.** Dataset loading,
representation construction, prompt resolution, strict parsing, consensus,
project Macro-F1, statistical helpers, and model backends live here once.

The historical `classifica_dataset_raw.py` is used only as evidence for the
Ollama execution mechanics (timeouts/retries/model restart/context/truncation).
Its old response parser and its fallback-to-`others` behavior are intentionally
not present.

## Execution environment

`environment.txt` (repository root) records the OS, kernel, CPU, GPU/driver/CUDA,
RAM, and Ollama version of the machine used to run the LLM experiments. It was
captured after the study's original execution window, so its OS/driver/CUDA
versions reflect routine machine updates since then and may differ in patch
level from the exact figures reported in the paper (Section 3.1: Ubuntu 24.04.3,
kernel `6.8.0-87-generic`, NVIDIA driver 570.211.01, CUDA 12.8, "up to 5.7 GHz"
CPU boost clock vs. the 6.0 GHz reported here, likely a BIOS/turbo setting
change); the underlying hardware (CPU/GPU model, core count, RAM) is unchanged.

## Tree

```text
src/
├── config.py
├── data/loader.py
├── representations.py
├── prompts/loader.py
├── parsing/
│   ├── strict.py
│   └── strict_response_parser.py   # installed byte-for-byte from canonical v4.1.1
├── inference/
│   ├── schema.py
│   ├── ollama.py
│   ├── flaccs.py
│   └── hqcm.py
├── consensus.py
├── evaluation/
│   ├── metrics.py
│   └── statistics.py
├── training/
│   ├── svm.py
│   ├── bilstm.py
│   └── codebert.py
└── utils/
    ├── io.py
    └── provenance.py
```

## 1. Install the canonical parser

The parser is deliberately **not reimplemented** in this archive. Copy the exact
frozen file already used by the study:

```bash
python3 tools/install_canonical_parser.py \
  --source /absolute/path/to/strict_response_parser.py
```

The installer validates parser version `4.1.1`, the expected 11 labels, and the
required API before copying it byte-for-byte to `src/parsing/`.

For the final GitHub replication package, commit that installed file directly as
`src/parsing/strict_response_parser.py`.

## 2. Canonical dataset

`src.data.loader.load_canonical_dataset()` expects exactly the canonical schema:

```text
partition, platform, project, commit_hash, commit_date, target,
original_type, breaking_change, original_message, processed_message,
has_diff, diff, git_show
```

It performs no delimiter guessing and no reconstruction from historical files.
The canonical datasets are already the reproduction starting point.

Representations are derived at runtime:

```text
message_only     = processed_message
diff_only        = diff
message_and_diff = processed_message + "\n\n" + diff
git_show         = git_show
```

RQ4 folds/holdout are materialized from the authoritative assignment manifests;
`materialize_assignments()` never re-splits or shuffles the data.

## 3. Parsing and `others`

There is no public `policy` switch. `src.parsing.strict` exposes only the final
strict-taxonomy decision:

- parser-valid task label -> semantic prediction;
- explicit external label -> `<invalid_output>`;
- malformed/ambiguous/unparseable output -> `<invalid_output>`;
- technical failure -> `<invalid_output>`;
- `others` -> semantic class only when the parser validly returns `others` and
  `others` is in the active task label space.

Task-specific artifact protocols can pass a smaller allowed label set. A valid
study-wide label outside that task's label space becomes invalid for that task.

HQCM has one narrow native-contract exception: a complete bare `cicd` answer is
normalized to `ci`. Other responses go through the canonical parser.

## 4. Ollama

**New to this machine?** See [`OLLAMA_SETUP.md`](OLLAMA_SETUP.md) first —
installing Ollama itself, which `ollama pull` tag to use for each model in
`config/models.yaml`, the digest-mismatch check this package enforces (and
why Ollama tags can drift), and rough RAM/VRAM sizing per model class. This
section and "Ollama execution and resume" below assume that part is done.

`src.inference.ollama` combines the useful runtime behavior from the historical
classifier with the final protocol:

- `cl100k_base` token estimation;
- 30,000 content-token prefix truncation;
- dynamic context `max(4096, min(input_tokens + 1024, 32048))`;
- `num_predict=1024`, `keep_alive=30m`;
- three attempts for retryable non-timeout errors;
- two-second retry delay for degenerate responses;
- timeout is not retried for the same request;
- model-health restart after three consecutive failures/timeouts;
- raw response retained;
- strict parser applied during inference.

The runtime dataclass can be constructed directly from the current flat `config/inference.yaml` through `OllamaRuntimeConfig.from_yaml()`. Temperature and request timeout remain experiment-specific.

## 5. FLACCS and HQCM

FLACCS uses its native greedy inference (`do_sample=False`, max 10 new tokens) and
does not pass sampling temperature/top-p. HQCM calls the native `hqcm.gen` pipeline
with temperature 0. HQCM chunk failure retries the chunk and then aborts; it never
manufactures invalid predictions for every row of a failed chunk.

Prompt text is not embedded in these backends: both consume the prompt files indexed
by `prompts/manifest.yaml`, keeping prompt provenance separate from execution code.

## 6. Training backends

The modules preserve the final runner settings supplied for the study, updated
2026-09-12 in response to reviewer M7 (fixed hyperparameters/epochs without
validation, and the 512-token BiLSTM/CodeBERT context asymmetry):

- TF-IDF + LinearSVC: 50k features, 1–2 grams, balanced LinearSVC;
- BiLSTM: vocab 50k, embedding/hidden 128, weighted CE, AdamW 1e-3, weight
  decay 1e-5, grad clip 1.0, max sequence length 10,000 tokens (raised from
  512: the empirical p95 of the pre-cutoff corpus — 512 truncated 57.6% of it);
- CodeBERT: `microsoft/codebert-base`, max length 512, LR 2e-5, batch 8,
  weight decay .01. 512 remains an architectural ceiling of the checkpoint
  (`max_position_embeddings=514`), not a config choice.

`C` (SVM) and the number of epochs (BiLSTM/CodeBERT) are no longer fixed in
advance. `src/training/model_selection.py` carves an internal train/dev split
per fold (`carve_validation_split()` — grouped by repository in cross-project,
class-stratified in within-corpus, never touching the test partition) and
`select_c()`/`select_num_epochs()` pick the hyperparameter once per fold from
that split before every seed retrains from scratch on the full outer-training
partition. Defaults for the search live in `config/training/{svm,bilstm,codebert}.yaml`.

These functions produce per-seed predictions only. Consensus is centralized.

## 7. Consensus

`resolve_first_to_two()` is parameterized by the experiment YAML. It requires all
initial seeds before evaluating a terminal result, then consumes adaptive seeds in
order. Two equal semantic votes are terminal. For generative models, two invalid
votes may be terminal. Exhaustion produces `<invalid_output>`.

This avoids legacy majority/tie-first-seed logic in older SVM/RQ4 runners.

## 8. Output contract

Prediction tables are intentionally compact: canonical
inputs and rendered prompts are reconstructable from the dataset, representation, and
prompt ID, so they are not duplicated in every row. A run-level manifest freezes their
checksums and the model/parser/runtime provenance. Development-only fields such as
`policy`, `mapped_vote`, `old_prediction`, and `changed_from_old` do not belong to new
reproduction outputs.

## Tests

After installing the canonical parser:

```bash
PYTHONPATH=. python3 tests/test_strict_parser.py
PYTHONPATH=. python3 tests/test_consensus.py
PYTHONPATH=. python3 tests/test_data_and_representations.py
```

## Integration status

Three orchestrators now read `config/experiments/*.yaml` and write the
standardized output schema:

- `scripts/run_ollama_experiment.py` — GPT-OSS and every other generative
  condition run through Ollama (RQ1–RQ3 and the GPT-OSS/FLACCS/HQCM columns
  of the RQ4 cross-project/within-corpus evaluations, plus the GPT-OSS side of
  the artifact-compatible comparisons);
- `scripts/run_baselines_experiment.py` — the three trainable RQ4 baselines
  (TF-IDF+SVM, BiLSTM, CodeBERT);
- `scripts/run_flaccs_experiment.py` — the FLACCS side of the
  artifact-compatible comparison (`config/experiments/rq4_2/artifact_flaccs.yaml`),
  running the released `0x404/ccs-code-llama-7b` artifact directly from the
  Hugging Face Hub against its native 10-class prompt. `FlaccsBackend` (in
  `src/inference/flaccs.py`) owns the model call and the tokenizer-level
  prompt truncation (`build_prompt()`); the runner only orchestrates
  manifest/journal/consensus, the same shape as the other two runners.

**Accepted limitation, not a gap**: there is no `scripts/run_hqcm_experiment.py`.
Unlike FLACCS, CodeFuse HQCM has no ready-to-use released weights — reproducing
it means reconstructing the original artifact's LoRA fine-tuning environment
(a dedicated Python venv under a `--hqcm-root`, plus the already-trained
`chcl_llama2_7b_len1024` adapter), not just running inference. That was judged
out of scope for this replication package. `results/rq4_2/related_work_artifacts/hqcm*/`
and the `hqcm`/`CodeFuse HQCM` columns of the cross-project/within-corpus
evaluations stay frozen from the original execution; see `TODO.md` item 1 for
the reasoning and item 4 of the reviewer-response letter for how this is
disclosed.


### Ollama execution and resume

Use `python scripts/run_ollama_experiment.py --experiment config/experiments/rq2.yaml --list-jobs`
to inspect jobs, or replace `--list-jobs` with `--dry-run --limit 1` for a configuration smoke check.
Dry-run never contacts Ollama. Real execution checks the installed model digest
against `config/models.yaml`. Install the canonical parser with
`python tools/install_canonical_parser.py --source /path/to/strict_response_parser.py`.

`--experiment <path>` can be replaced with `--rq {rq1,rq2,rq3_1,rq3_2,rq4}` to select an
experiment by research question instead of a YAML path:

```bash
python scripts/run_ollama_experiment.py --rq rq1 --list-jobs
python scripts/run_ollama_experiment.py --rq rq3_2 --stage factor_1_definitions --dry-run
python scripts/run_ollama_experiment.py --rq rq4 --stage cross_project --resume
python scripts/run_ollama_experiment.py --rq rq4 --resume   # all four rq4 sub-experiments in sequence
```

`--stage` selects one of the four sub-experiments (`cross_project`, `within_corpus`
under `config/experiments/rq4_1/*.yaml`; `artifact_flaccs`, `artifact_hqcm` under
`config/experiments/rq4_2/*.yaml`) when `--rq rq4`
is used; without `--stage`, all four run one after another, each keeping its own
run directory, manifest, and config-hash provenance. For `--rq rq3_2` (or `--experiment`
pointed directly at `config/experiments/rq3_2.yaml`), `--stage` instead filters the
already-expanded jobs down to one of the YAML's `factor_*` keys (e.g.
`factor_1_definitions`, `factor_2_heuristics`, `factor_3_cot_and_few_shot`,
`factor_4_abstention`); treatments (prompts) shared across factors still run once.
`--stage` is not accepted for `rq1`, `rq2`, or `rq3_1`, which have no sub-experiments
or factors.

New runs default to `runs/`; output inside `results/` is rejected. Use `--resume`
only for an existing compatible run. The append-only call and attempt journals
are authoritative and must be retained. Compressed raw snapshots are rebuilt
atomically; torn journal tails are archived before repair. Each completed call
is fsynced independently. Resume preserves the original start time and records
separate events. Snapshot-only older runs require an explicit migration, which
this runner does not perform.

The mandatory base block is 10/20/30. Adaptive seeds start at 40 and increase by
10 until two semantic or two invalid votes agree. No finite seed list determines
exhaustion. Operational interruptions remain incomplete. Historical runs resolved
before seed 80 was needed; this is an observation rather than a stopping rule.

A timed-out blocking request retains its concurrency slot until it finishes.
This prioritizes the configured in-flight limit over an immediate timeout return;
no automatic model restart runs alongside live requests. Every returned retry
attempt retains raw output and technical details.

Install test support with `python -m pip install pytest`, then run `python -m pytest`.
Tests use mocked inference and never need a live model. Supervised aggregation,
training configuration, and HQCM-original statistics await the original research
machine and are outside this execution layer.


### Baselines execution and resume

Use `python scripts/run_baselines_experiment.py --stage cross_project --list-jobs`
(or `--stage within_corpus`) to inspect jobs for the three trainable baselines
(TF-IDF+SVM, BiLSTM, CodeBERT), and `--dry-run --limit 1` for a configuration
smoke check — same manifest/journal/`--resume` conventions as the Ollama
runner, resumable at the granularity of "seed already complete for this fold"
(one seed = one from-scratch retrain + batch prediction, not per-commit).
`torch` is required on the execution machine for BiLSTM/CodeBERT; the
development checkout this README ships from does not have it installed, so
those two backends are validated on the machine that actually runs the study
(see `tests/test_model_selection.py` for what does run without `torch`).

### FLACCS artifact-compatible execution

Use `python scripts/run_flaccs_experiment.py --list-jobs` (or `--dry-run`) to
validate config/data/prompt without loading the model — this works on a
machine without `torch`/`transformers` installed, same as `--list-jobs`/
`--dry-run` on the other two runners. Real execution downloads
`0x404/ccs-code-llama-7b` from the Hugging Face Hub on first run and needs a
GPU-capable machine; `--resume` continues an interrupted run without
repeating completed (commit, seed) calls, and `--limit N` truncates to a
smoke test (use a separate `--output-dir`).

### within-corpus cohort preflight (resolved 2026-09-12)

The within-corpus condition previously failed cohort preflight: the supplied
assignments selected 1,632 commits while a stale config pinned the frozen
1,630-commit cohort/hash from an older corpus snapshot. This was reconciled
against original evidence — `config/experiments/rq4_1/within_corpus.yaml` now
pins `expected_rows: 1632` and the corresponding `expected_cohort_sha256`,
confirmed by hashing the `split=="test"` side of
`data/rq4_1/within_corpus/assignments.csv` and matching it against the legacy
package's own `within_corpus/test.csv` byte-for-byte. Both
`--rq rq4 --stage within_corpus` (Ollama) and
`run_baselines_experiment.py --stage within_corpus` now pass preflight.

The GPT-OSS/FLACCS/HQCM predictions frozen against the old 1,630-commit cohort
were superseded without re-running inference: because those three approaches
don't train, a commit's prediction depends only on the commit + model +
prompt + seed, not on which protocol (cross-project vs. within-corpus) it was
sampled into — and `cross_project.yaml`/`within_corpus.yaml` configure those
three conditions identically. `results/rq4_1/within_corpus/` now
holds predictions/metrics cherry-picked from the cross-project execution
(which covers the full 8,159-commit pre-cutoff partition) filtered down to
the 1,632 commits of the current within-corpus cohort. See `TODO.md` §2 for
the full reconciliation trail.
