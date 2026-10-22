# CodeFuse-HQCM External Artifact

This directory documents the external CodeFuse-HQCM artifact used in the RQ4 comparison.

## Upstream project

HQCM was released by the CodeFuse team with the paper:

**Understanding Code Changes Practically with Small-Scale Language Models**  
ASE 2024

Official repository:

https://github.com/codefuse-ai/codefuse-hqcm

The exact upstream revision used in this study is:

```text
ea7e7fab1bca176d9afd77350496649d7ca4743f
```

The upstream repository is the authoritative source for the original HQCM implementation, model artifacts, and dataset.

## Use in this study

RQ4 uses HQCM in two distinct settings:

1. **Original HQCM evaluation** — evaluation using the original HQCM artifact and its original test data.
2. **Artifact-compatible evaluation on our corpus** — evaluation using a subset of our canonical pre-cutoff corpus prepared to satisfy the HQCM artifact requirements.

These two settings are independent and should not be confused.

The HQCM-compatible subset derived from our corpus is documented under:

`data/rq4_2/artifact_compatible/hqcm/`

## Retrieving the upstream artifact

The original HQCM repository and dataset are not redistributed in this replication package.

To retrieve the exact upstream revision used in the study, run from the replication-package root:

```bash
python3 data/external/hqcm/download_hqcm.py
```

The script clones:

```text
https://github.com/codefuse-ai/codefuse-hqcm.git
```

and checks out:

```text
ea7e7fab1bca176d9afd77350496649d7ca4743f
```

By default, the checkout is created at:

```text
data/external/hqcm/codefuse-hqcm/
```

The downloaded upstream repository should not be committed to this replication package.

Recommended `.gitignore` entry:

```gitignore
/data/external/hqcm/codefuse-hqcm/
```

The same operation can be performed manually:

```bash
git clone https://github.com/codefuse-ai/codefuse-hqcm.git
cd codefuse-hqcm
git checkout ea7e7fab1bca176d9afd77350496649d7ca4743f
```

Using the pinned revision is preferred over relying on the moving default branch.

## Provenance

This replication package does not claim ownership of HQCM code, models, or original data. Third-party files remain governed by the licensing and attribution terms provided by the HQCM authors in the upstream repository.

Our package contains only:

- a reference to the upstream HQCM artifact and the revision used;
- a retrieval script;
- our experiment configuration and outputs;
- our canonical commit datasets; and
- the HQCM artifact-compatible subset derived from our own corpus.
