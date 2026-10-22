from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Collection

from ..parsing.strict import PredictionParse, parse_hqcm


@dataclass(frozen=True)
class HqcmConfig:
    hqcm_root: Path
    adapter: Path
    python: Path
    temperature: float = 0.0
    chunk_size: int = 250
    max_attempts: int = 3
    retry_delay_seconds: float = 5.0


class HqcmBackend:
    """CodeFuse HQCM native-pipeline adapter.

    Infrastructure failure of a chunk aborts after retries; it is never expanded
    into synthetic invalid predictions for every commit in that chunk.
    """

    def __init__(self, config: HqcmConfig):
        self.config = config

    def supports_seed_arg(self) -> bool:
        result = subprocess.run(
            [str(self.config.python), "-m", "hqcm.gen", "--help"],
            cwd=str(self.config.hqcm_root), capture_output=True, text=True, check=False,
        )
        return "--seed" in ((result.stdout or "") + (result.stderr or ""))

    def run_chunk(
        self,
        *,
        dataset_dir: Path,
        output_json: Path,
        seed: int,
    ) -> None:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        env["PYTHONPATH"] = str(self.config.hqcm_root)
        cmd = [
            str(self.config.python), "-m", "hqcm.gen",
            "--dataset", str(dataset_dir), "--split", "test",
            "--output", str(output_json), "--temperature", "0",
        ]
        if self.supports_seed_arg():
            cmd += ["--seed", str(seed)]
        cmd += [str(self.config.adapter)]

        last_error: Exception | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                output_json.unlink(missing_ok=True)
                subprocess.run(cmd, cwd=str(self.config.hqcm_root), env=env, check=True)
                if not output_json.exists():
                    raise RuntimeError(f"hqcm.gen did not create {output_json}")
                return
            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_attempts:
                    time.sleep(self.config.retry_delay_seconds)
        raise RuntimeError(
            f"HQCM chunk failed after {self.config.max_attempts} attempts. "
            "No synthetic invalid rows were created; rerun to resume completed chunks."
        ) from last_error

    @staticmethod
    def parse(raw_response: str, *, task_labels: Collection[str], technical_status: str = "ok") -> PredictionParse:
        return parse_hqcm(raw_response, task_labels=task_labels, technical_status=technical_status)
