# Configuration

This directory contains the declarative configuration used by the replication
package.

The replication package exposes the **final corrected implementation** of the
experimental pipeline. Historical intermediate parser behavior is not
reproduced.

## `models.yaml`

Canonical registry of the evaluated LLMs, including Ollama tags, parameter
sizes, quantization, complete model digests, and reported knowledge cutoffs.

## `inference.yaml`

Shared Ollama execution mechanics:

- token counting and input truncation;
- dynamic context sizing;
- generation length;
- retries and model-health behavior;
- concurrency;
- output fields; and
- the final strict parser applied during inference.

Experiment-specific variables such as temperature, model selection, dataset,
prompt, representation, and request timeout belong in `experiments/`.

## `evaluation.yaml`

Defines the final corrected interpretation and evaluation policy.

The resulting per-seed predictions are then combined using the adaptive
first-to-two consensus protocol.

Raw model responses are still preserved in the output artifacts for
auditability and debugging, but post-hoc reparsing is not required by the
published replication workflow.

## `experiments/`

Each experiment file binds together the concrete conditions used by one
research question or protocol:

- dataset;
- model/model set;
- prompt;
- input representation;
- temperature;
- seeds;
- model-specific runtime overrides; and
- output location.

## `training/`

Contains hyperparameters for the trainable RQ4 baselines. These models do not
use the Ollama inference configuration.

