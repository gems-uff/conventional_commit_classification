"""Durable append-only call evidence; compressed files are replaceable snapshots."""
from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def exclusive_run(path: Path):
    """OS-owned lock released on process death, without stale lock removal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another process owns this run") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_json(path: Path, value: dict) -> None:
    atomic_bytes(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8"))


class Journal:
    def __init__(self, path: Path):
        self.path = path

    def initialize(self) -> None:
        with self.path.open("ab") as handle:
            handle.flush()
            os.fsync(handle.fileno())

    def read(self) -> tuple[list[dict], bytes]:
        if not self.path.exists():
            return [], b""
        data = self.path.read_bytes()
        end = data.rfind(b"\n") + 1
        rows = []
        for number, line in enumerate(data[:end].splitlines(), 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected object")
            except (ValueError, UnicodeError) as exc:
                raise RuntimeError(f"Malformed journal record {number} in {self.path}") from exc
            rows.append(row)
        return rows, data[end:]

    def recover_tail(self, tail: bytes) -> None:
        # Only after identity and complete records have been validated.
        if tail:
            import hashlib
            backup = self.path.with_name(self.path.name + ".torn-" + hashlib.sha256(tail).hexdigest())
            atomic_bytes(backup, tail)
            with self.path.open("r+b") as handle:
                handle.truncate(self.path.stat().st_size - len(tail))
                handle.flush()
                os.fsync(handle.fileno())

    def append(self, row: dict) -> None:
        payload = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        with self.path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def snapshot(self, path: Path) -> None:
        fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as target:
                with gzip.GzipFile(fileobj=target, mode="wb", filename="", mtime=0) as compressed:
                    if self.path.exists():
                        with self.path.open("rb") as source:
                            shutil.copyfileobj(source, compressed)
                target.flush()
                os.fsync(target.fileno())
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)
