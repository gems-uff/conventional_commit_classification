from __future__ import annotations

from typing import Any
import pandas as pd

REPRESENTATIONS = ("message_only", "diff_only", "message_and_diff", "summary_and_diff", "git_show")


def build_representation(row: Any, representation: str) -> str:
    """Build an experimental input from the canonical dataset fields."""
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation {representation!r}; expected one of {REPRESENTATIONS}")

    def get(name: str) -> str:
        if isinstance(row, dict):
            value = row.get(name, "")
        elif hasattr(row, name):
            value = getattr(row, name)
        else:
            value = row[name]
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        return str(value)

    if representation == "message_only":
        return get("processed_message")
    if representation == "diff_only":
        return get("diff")
    if representation == "message_and_diff":
        return f"{get('processed_message')}\n\n{get('diff')}"
    if representation == "summary_and_diff":
        return f"Commit summary:\n{get('processed_message')}\n\nGit diff:\n{get('diff')}"
    return get("git_show")


def add_input_text(df: pd.DataFrame, representation: str) -> pd.DataFrame:
    out = df.copy()
    out["input_text"] = [build_representation(row, representation) for row in out.itertuples(index=False)]
    out["representation"] = representation
    return out
