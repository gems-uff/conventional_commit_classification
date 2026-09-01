# Prompts

This directory contains the prompt specifications used in the experiments.

Prompt files are stored separately from experiment configurations and execution code so that the exact instructions sent to the models can be inspected, versioned, and reused without duplicating prompt text inside runners.

## Directory structure

```text
prompts/
├── README.md
├── manifest.yaml
├── baseline/
│   └── p0_baseline.yaml
├── rq3_1/
│   ├── p_diff_only.yaml
│   ├── p_message_and_diff.yaml
│   └── p_message_only.yaml
├── rq3_2/
│   ├── abstention/
│   │   ├── p_exclude_low.yaml
│   │   └── p_high_only.yaml
│   ├── cot_and_few_shot/
│   │   ├── p_cot_few-shot.yaml
│   │   ├── p_few-shot.yaml
│   │   └── p_zero-shot_cot.yaml
│   ├── definitions/
│   │   ├── p1_definition_augmented.yaml
│   │   ├── p2_angular_definition.yaml
│   │   ├── p3_extensive_definition.yaml
│   │   └── p4_mix_definition.yaml
│   └── heuristics/
│       ├── p_chore_negative_and_dominant.yaml
│       ├── p_chore_rule.yaml
│       ├── p_chore_rule_and_negative.yaml
│       ├── p_code_disambiguation.yaml
│       ├── p_disambiguation.yaml
│       ├── p_file_pattern.yaml
│       ├── p_infra_disambiguation.yaml
│       ├── p_negative_constraints.yaml
│       ├── p_priority_rules.yaml
│       └── p_header_structured.yaml
└── rq4/
    ├── p_winner_adapted_to_flaccs.yaml
    └── p_winner_adapted_to_hqcm.yaml
```

## Prompt manifest

`manifest.yaml` is the index of the prompt collection.

It provides a stable prompt identifier for each prompt and records:

- the corresponding file path;
- the experimental category;
- the research question in which the prompt was used;
- the input representation, when applicable; and
- the external artifact targeted by an adapted prompt, when applicable.

The prompt YAML files themselves remain the authoritative source for the actual prompt content.

Experiment configurations should refer to prompt IDs from `manifest.yaml` rather than duplicating the prompt text.

## Baseline

`baseline/p0_baseline.yaml` contains the common baseline prompt.

It is reused by experiments that require the baseline classification setting, including RQ1, RQ2, and the Git Show baseline condition used in RQ3.1.

The baseline prompt is stored only once to avoid having multiple copies that could diverge.

## RQ3.1: input representation

RQ3.1 evaluates how the information supplied to the model affects commit classification.

The representation-specific prompt files are:

- `p_message_only.yaml`
- `p_diff_only.yaml`
- `p_message_and_diff.yaml`

The Git Show condition reuses `p0_baseline.yaml`.

The corresponding model inputs are derived from the canonical post-cutoff dataset:

```text
message_only
    = processed_message

diff_only
    = diff

message_and_diff
    = processed_message + "\n\n" + diff

git_show
    = git_show
```

These representations are generated at execution time and are not stored as separate datasets.

## RQ3.2: prompt refinement

RQ3.2 contains the staged prompt-refinement experiments.

### Definitions

`rq3_2/definitions/` contains variants that change or expand the definitions of the commit classes.

### Heuristics

`rq3_2/heuristics/` contains variants that introduce explicit rules or disambiguation guidance for ambiguous classes and code-change patterns.

### Chain-of-thought and few-shot variants

`rq3_2/cot_and_few_shot/` contains the reasoning- and example-based variants evaluated after the definition and heuristic stages.

The directory name identifies the experimental family; the individual YAML files contain the exact instructions and examples used in each condition.

### Abstention variants

`rq3_2/abstention/` contains confidence-aware prompt variants used to evaluate abstention or prediction filtering.

## RQ4: artifact-specific adaptations

`rq4/` contains adaptations of the selected prompt for external comparison artifacts.

These files are not independent prompt-refinement candidates. They adapt the selected classification strategy to constraints imposed by the corresponding external artifacts, including differences in input format and taxonomy.

The current artifact-specific prompts are:

- `p_winner_adapted_to_flaccs.yaml`
- `p_winner_adapted_to_hqcm.yaml`

The datasets associated with these evaluations are documented under:

```text
data/rq4/artifact_compatible/flaccs/
data/rq4/artifact_compatible/hqcm/
```

## Prompt IDs and filenames

Prompt IDs are intended to remain stable across experiment configurations, analysis scripts, and processed outputs.

For example:

```yaml
prompt_id: p_code_disambiguation
```

resolves through `manifest.yaml` to:

```text
rq3_2/heuristics/p_code_disambiguation.yaml
```

This separation makes it possible to reorganize implementation code without changing the identifier associated with a published experimental condition.

## Reproducibility

When reproducing an experiment:

1. load the experiment configuration;
2. resolve its `prompt_id` through `prompts/manifest.yaml`;
3. load the referenced YAML prompt file;
4. construct the specified input representation from the canonical dataset; and
5. execute the model using the inference parameters recorded by the experiment configuration.

Prompt content should not be copied or reimplemented inside experiment runners.
