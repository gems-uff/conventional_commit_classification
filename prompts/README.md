# Prompts

Experiment files refer to prompts by `prompt_id`. `manifest.yaml` maps each ID
to its file, for example:

```yaml
prompt_id: p_code_disambiguation   # -> rq3_2/heuristics/p_code_disambiguation.yaml
```

```text
prompts/
├── manifest.yaml
├── baseline/p0_baseline.yaml                RQ1, RQ2, RQ3.1 (git_show), RQ3.2
├── rq3_1/                                   RQ3.1
│   ├── p_message_only.yaml
│   ├── p_diff_only.yaml
│   └── p_message_and_diff.yaml
├── rq3_2/                                   RQ3.2, one directory per factor
│   ├── definitions/
│   ├── heuristics/
│   ├── cot_and_few_shot/
│   └── abstention/
└── rq4/                                     RQ4
    ├── p_flaccs_native_10class.yaml         FLACCS, RQ4.2
    ├── p_flaccs_extended_11class.yaml       FLACCS, RQ4.1
    ├── p_hqcm_native_8class.yaml            HQCM, RQ4.2
    ├── p_hqcm_extended_11class.yaml         HQCM, RQ4.1
    ├── p_winner_adapted_to_flaccs.yaml      GPT-OSS, RQ4.2 FLACCS subset
    └── p_winner_adapted_to_hqcm.yaml        GPT-OSS, RQ4.2 HQCM subset
```

RQ4.1 GPT-OSS uses `p_code_disambiguation` (`rq3_2/heuristics/`).

## File format

Each prompt file is a list of messages. Only messages with `activate: true`
are sent:

```yaml
- name: <message name>
  role: system | user | assistant
  activate: true
  content: |
    ...
```

Placeholders filled at run time from the commit:

| Placeholder | Value |
|---|---|
| `{{COMMIT_INPUT}}` | Input representation set in the experiment file (`message_only`, `diff_only`, `message_and_diff`, or `git_show`). |
| `{{COMMIT_MESSAGE}}` | `processed_message`. |
| `{{COMMIT_DIFF}}` | `diff`. |

Ollama runs apply the input-length limit of `config/inference.yaml`
(`input.max_content_tokens`) before filling the placeholders. FLACCS prompts are
truncated as set in `config/experiments/rq4_2/artifact_flaccs.yaml`.

Representations:

```text
message_only     = processed_message
diff_only        = diff
message_and_diff = processed_message + "\n\n" + diff
git_show         = git_show
```
