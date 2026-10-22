from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

CANONICAL_COLUMNS = (
    "partition", "platform", "project", "commit_hash", "commit_date",
    "target", "original_type", "breaking_change", "original_message",
    "processed_message", "has_diff", "diff", "git_show",
)
ASSIGNMENT_COLUMNS = ("protocol", "fold", "split", "project", "commit_hash", "target")
STUDY_LABELS = (
    "fix", "feat", "build", "ci", "docs", "style",
    "refactor", "perf", "test", "chore", "others",
)
LABEL_SET = set(STUDY_LABELS)


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def load_canonical_dataset(path: str | Path, *, expected_partition: str | None = None) -> pd.DataFrame:
    """Load one canonical pre/post dataset with no schema guessing.

    The canonical CSV is the authoritative experimental data source. Extra
    columns are allowed for forward-compatible metadata, but all canonical
    fields must be present.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(
        path,
        compression="infer",
        keep_default_na=False,
        low_memory=False,
        dtype={
            "partition": str,
            "platform": str,
            "project": str,
            "commit_hash": str,
            "target": str,
        },
    )
    missing = sorted(set(CANONICAL_COLUMNS) - set(df.columns))
    if missing:
        raise RuntimeError(f"{path}: missing canonical columns {missing}")

    for col in ("partition", "platform", "project", "commit_hash", "target"):
        df[col] = df[col].astype(str).str.strip()
    df["partition"] = df["partition"].str.casefold()
    df["commit_hash"] = df["commit_hash"].str.casefold()
    df["target"] = df["target"].str.casefold()
    df["breaking_change"] = df["breaking_change"].map(_bool)
    df["has_diff"] = df["has_diff"].map(_bool)

    invalid_targets = sorted(set(df["target"]) - LABEL_SET)
    if invalid_targets:
        raise RuntimeError(f"{path}: targets outside study taxonomy: {invalid_targets}")
    if expected_partition is not None:
        expected = expected_partition.casefold()
        found = set(df["partition"])
        if found != {expected}:
            raise RuntimeError(f"{path}: expected partition={expected!r}, found {sorted(found)}")
    if df["project"].eq("").any() or df["commit_hash"].eq("").any():
        raise RuntimeError(f"{path}: empty project or commit_hash")
    if df.duplicated(["project", "commit_hash"]).any():
        sample = df.loc[df.duplicated(["project", "commit_hash"], keep=False), ["project", "commit_hash"]].head(20)
        raise RuntimeError(f"{path}: duplicate commit keys\n{sample.to_string(index=False)}")

    return df.reset_index(drop=True)


def load_assignments(path: str | Path) -> pd.DataFrame:
    """Load an RQ4 assignment manifest without reconstructing the split."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, keep_default_na=False, low_memory=False)
    artifact_columns = {"protocol", "project", "commit_hash", "canonical_target", "protocol_target"}
    if artifact_columns.issubset(df.columns) and "target" not in df:
        protocols = set(df["protocol"])
        if protocols not in ({"flaccs_compatible"}, {"hqcm_compatible"}):
            raise RuntimeError("Unknown artifact assignment protocol")
        native = df["protocol_target"].astype(str).str.strip().str.casefold()
        if protocols == {"hqcm_compatible"}:
            native = native.replace({"cicd": "ci"})
        canonical = df["canonical_target"].astype(str).str.strip().str.casefold()
        if not native.equals(canonical):
            raise RuntimeError("Artifact protocol target differs from canonical target")
        df["target"] = canonical
        # Membership manifests have no train/test partition or CV fold. These
        # execution aliases select all members; original metadata is retained.
        df["split"] = "test"
        df["fold"] = 0
    missing = sorted(set(ASSIGNMENT_COLUMNS) - set(df.columns))
    if missing:
        raise RuntimeError(f"{path}: missing assignment columns {missing}")
    for col in ("protocol", "split", "project", "commit_hash", "target"):
        df[col] = df[col].astype(str).str.strip()
    df["protocol"] = df["protocol"].str.casefold()
    df["split"] = df["split"].str.casefold()
    df["commit_hash"] = df["commit_hash"].str.casefold()
    df["target"] = df["target"].str.casefold()
    df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(int)
    if not set(df["split"]).issubset({"train", "test"}):
        raise RuntimeError(f"{path}: unsupported split values {sorted(set(df['split']))}")
    df["_assignment_order"] = range(len(df))
    if df.duplicated(list(ASSIGNMENT_COLUMNS[:-1])).any():
        raise RuntimeError("Duplicate assignment rows")
    validate_assignment_selection(None, df)
    return df


def validate_assignment_selection(selected: pd.DataFrame | None, assignments: pd.DataFrame) -> None:
    """Check frozen split invariants without regenerating assignments."""
    for (_, fold), group in assignments.groupby(["protocol", "fold"]):
        train = group[group["split"] == "train"]
        test = group[group["split"] == "test"]
        train_keys = set(zip(train.project, train.commit_hash))
        if train_keys & set(zip(test.project, test.commit_hash)):
            raise RuntimeError(f"Train/test commit leakage in fold {fold}")
        if group["protocol"].str.contains("cross_project").any():
            if set(train.project) & set(test.project):
                raise RuntimeError(f"Cross-project repository leakage in fold {fold}")
    cross_test = assignments[(assignments["split"] == "test") & assignments["protocol"].str.contains("cross_project")]
    if not cross_test.empty and (cross_test.groupby(["protocol", "project"])["fold"].nunique() > 1).any():
        raise RuntimeError("Cross-project repository assigned to multiple test folds")
    if selected is not None and selected.duplicated(["project", "commit_hash"]).any():
        raise RuntimeError("Duplicate selected commit keys")


def materialize_assignments(
    dataset: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    protocol: str | None = None,
    fold: int | None = None,
    split: str | None = None,
) -> pd.DataFrame:
    """Join an authoritative assignment manifest to canonical commit content.

    No re-splitting, shuffling, or sampling is performed. Assignment order is
    preserved exactly. The redundant assignment target is verified against the
    canonical dataset.
    """
    a = assignments.copy()
    if protocol is not None:
        a = a[a["protocol"] == protocol.casefold()].copy()
    if fold is not None:
        a = a[a["fold"] == int(fold)].copy()
    if split is not None:
        a = a[a["split"] == split.casefold()].copy()
    if a.empty:
        raise RuntimeError("Assignment selection is empty")

    canonical = dataset.copy()
    canonical["commit_hash"] = canonical["commit_hash"].astype(str).str.casefold()
    joined = a.merge(
        canonical,
        on=["project", "commit_hash"],
        how="left",
        validate="many_to_one",
        suffixes=("_assignment", ""),
        indicator=True,
    )
    missing = joined[joined["_merge"] != "both"]
    if not missing.empty:
        raise RuntimeError(
            "Assignments reference commits absent from canonical dataset:\n"
            + missing[["project", "commit_hash"]].head(20).to_string(index=False)
        )
    mismatch = joined[joined["target_assignment"] != joined["target"]]
    if not mismatch.empty:
        raise RuntimeError(
            "Target mismatch between assignments and canonical dataset:\n"
            + mismatch[["project", "commit_hash", "target_assignment", "target"]].head(20).to_string(index=False)
        )
    joined = joined.drop(columns=["target_assignment", "_merge"])
    return joined.sort_values("_assignment_order", kind="stable").reset_index(drop=True)


def cohort_sha256(df: pd.DataFrame) -> str:
    rows = (
        df[["project", "commit_hash", "target"]]
        .astype(str)
        .sort_values(["project", "commit_hash", "target"], kind="stable")
        .to_dict(orient="records")
    )
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
