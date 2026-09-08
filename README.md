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

The modules preserve the final runner settings supplied for the study:

- TF-IDF + LinearSVC: 50k features, 1–2 grams, balanced LinearSVC;
- BiLSTM: vocab 50k, seq 512, embedding/hidden 128, 10 epochs, weighted CE,
  AdamW 1e-3, weight decay 1e-5, grad clip 1.0;
- CodeBERT: `microsoft/codebert-base`, max length 512, fixed 3 epochs, LR 2e-5,
  batch 8, weight decay .01, no test-time model selection.

These functions produce per-seed predictions only. Consensus is centralized.

## 7. Consensus

`resolve_first_to_two()` is parameterized by the experiment YAML. It requires all
initial seeds before evaluating a terminal result, then consumes adaptive seeds in
order. Two equal semantic votes are terminal. For generative models, two invalid
votes may be terminal. Exhaustion produces `<invalid_output>`.

This avoids legacy majority/tie-first-seed logic in older SVM/RQ4 runners.

## 8. Output contract

See `OUTPUT_SCHEMA.md`. Prediction tables are intentionally compact: canonical
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

This archive is the shared library layer. The next package should add the thin
`scripts/run_experiment.py` orchestrator that reads `config/experiments/*.yaml`,
selects the backend from the condition, and writes the standardized output schema.


### Ollama execution and resume

Use `python scripts/run_ollama_experiment.py --experiment config/experiments/rq2.yaml --list-jobs`
to inspect jobs, or replace `--list-jobs` with `--dry-run --limit 1` for a configuration smoke check.
Dry-run never contacts Ollama. Real execution checks the installed model digest
against `config/models.yaml`. Install the canonical parser with
`python tools/install_canonical_parser.py --source /path/to/strict_response_parser.py`.

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


The within-corpus Ollama condition currently fails cohort preflight: the supplied
assignments select 1,632 commits while frozen predictions contain 1,630, with
substantial membership differences. Its config pins the frozen count and hash;
reconcile the assignments from original evidence before running it. The runner
will not silently regenerate the split or bypass this check with `--limit`.
