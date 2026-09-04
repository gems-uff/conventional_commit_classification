# RQ3.2 Systematic Error Pattern Analysis — `p_header_structured`

This is a descriptive analysis of the **final strict consensus predictions** for the structured prompt condition. It does not reparse model outputs and does not alter the RQ3.2 consensus. Its purpose is to identify recurring directional class confusions that can motivate subsequent prompt-disambiguation rules; it is not an independent validation of those rules.

## Scope

- Commits analyzed: **7,937**
- Projects represented: **30**
- Correct predictions: **4,314**
- Errors: **3,623** (45.65%)

## Top semantic directional confusions

| target | prediction | count | target rate | share of target errors | projects | project coverage | reverse count | direction share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chore | build | 500 | 24.04% | 37.37% | 28 | 93.33% | 88 | 85.03% |
| fix | feat | 367 | 17.44% | 39.93% | 30 | 100.00% | 36 | 91.07% |
| fix | refactor | 264 | 12.55% | 28.73% | 30 | 100.00% | 22 | 92.31% |
| others | chore | 171 | 45.60% | 45.72% | 14 | 46.67% | 6 | 96.61% |
| chore | refactor | 157 | 7.55% | 11.73% | 24 | 80.00% | 18 | 89.71% |
| chore | ci | 149 | 7.16% | 11.14% | 27 | 90.00% | 34 | 81.42% |
| chore | docs | 137 | 6.59% | 10.24% | 23 | 76.67% | 58 | 70.26% |
| chore | fix | 119 | 5.72% | 8.89% | 22 | 73.33% | 91 | 56.67% |
| chore | test | 117 | 5.62% | 8.74% | 17 | 56.67% | 17 | 87.31% |
| chore | feat | 112 | 5.38% | 8.37% | 20 | 66.67% | 44 | 71.79% |
| fix | chore | 91 | 4.33% | 9.90% | 25 | 83.33% | 119 | 43.33% |
| build | chore | 88 | 26.67% | 46.07% | 6 | 20.00% | 500 | 14.97% |
| fix | build | 65 | 3.09% | 7.07% | 18 | 60.00% | 19 | 77.38% |
| feat | refactor | 60 | 7.52% | 29.56% | 21 | 70.00% | 46 | 56.60% |
| docs | chore | 58 | 5.19% | 28.29% | 14 | 46.67% | 137 | 29.74% |

## Systematic-confusion filter

`top_systematic_confusions.csv` contains semantic directional pairs that satisfy both the configured minimum occurrence count and minimum project coverage thresholds. The complete unfiltered results remain in `directional_errors_semantic.csv`.

Pairs satisfying the thresholds: **47**.

## Non-semantic outcomes

| prediction | count | projects |
|---|---:|---:|
| <invalid_output> | 36 | 18 |

## Interpretation notes

- `target_rate` = occurrences of target→prediction divided by all commits whose ground-truth target is that target class.
- `share_of_target_errors` = occurrences of target→prediction divided by all errors made on that target class.
- `project_coverage` = number of projects exhibiting the directional confusion divided by all projects represented in the structured-prompt cohort.
- `reverse_count` compares A→B with B→A. `direction_share` is A→B / (A→B + B→A), so values near 1 indicate a strongly directional confusion.
- The analysis is exploratory/developmental because the prompt heuristics and their evaluation are based on the same post-cutoff partition. It can justify why a heuristic was introduced, but should not be presented as independent evidence that the heuristic generalizes.
