#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

EXPECTED_VERSION = "4.1.1"
EXPECTED_LABELS = (
    "fix", "feat", "build", "ci", "docs", "style",
    "refactor", "perf", "test", "chore", "others",
)


def load(path: Path):
    spec = importlib.util.spec_from_file_location("canonical_parser_validation", path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # required by @dataclass on Python 3.12+
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--destination", default=str(Path(__file__).resolve().parents[1] / "src/parsing/strict_response_parser.py"))
    args = ap.parse_args(); src = Path(args.source).expanduser().resolve(); dst = Path(args.destination).expanduser().resolve()
    if not src.exists(): raise FileNotFoundError(src)
    m = load(src)
    if getattr(m, "PARSER_VERSION", None) != EXPECTED_VERSION: raise RuntimeError(f"Expected parser {EXPECTED_VERSION}")
    if tuple(getattr(m, "LABELS", ())) != EXPECTED_LABELS: raise RuntimeError(f"Unexpected parser labels {getattr(m, 'LABELS', None)}")
    if not hasattr(m, "parse_raw_response") or not hasattr(m, "ParseResult"): raise RuntimeError("Parser API incomplete")
    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    print(f"Installed canonical parser: {dst}")
    print(f"Version: {m.PARSER_VERSION}")

if __name__ == "__main__": main()
