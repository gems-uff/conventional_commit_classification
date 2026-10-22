#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


HQCM_REPOSITORY = "https://github.com/codefuse-ai/codefuse-hqcm.git"
HQCM_REVISION = "ea7e7fab1bca176d9afd77350496649d7ca4743f"


def run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        location = f" (cwd={cwd})" if cwd else ""
        raise RuntimeError(
            "Command failed"
            f"{location}:\n"
            f"  {' '.join(command)}\n\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return result.stdout.strip()


def current_revision(repo_dir: Path) -> str | None:
    try:
        return run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
        )
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve the exact CodeFuse-HQCM upstream revision used "
            "by this replication package."
        )
    )

    default_output = (
        Path(__file__).resolve().parent / "codefuse-hqcm"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=(
            "Directory where CodeFuse-HQCM will be cloned. "
            "Default: data/external/hqcm/codefuse-hqcm"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Remove an existing output directory before cloning. "
            "Use with care."
        ),
    )

    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    if shutil.which("git") is None:
        raise RuntimeError(
            "Git was not found in PATH. Install Git before running "
            "this script."
        )

    print("=" * 72)
    print("CODEFUSE-HQCM UPSTREAM RETRIEVAL")
    print("=" * 72)
    print(f"Repository: {HQCM_REPOSITORY}")
    print(f"Revision:   {HQCM_REVISION}")
    print(f"Output:     {output_dir}")
    print()

    if output_dir.exists():
        existing_revision = current_revision(output_dir)

        if existing_revision == HQCM_REVISION:
            print(
                "The requested HQCM revision is already present."
            )
            print(f"HEAD: {existing_revision}")
            return

        if not args.force:
            raise RuntimeError(
                f"Output directory already exists: {output_dir}\n"
                "It does not contain the expected checked-out revision.\n"
                "Remove it manually, choose another --output-dir, or "
                "rerun with --force."
            )

        print("Removing existing output directory...")
        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Cloning upstream repository...")
    run(
        [
            "git",
            "clone",
            "--no-checkout",
            HQCM_REPOSITORY,
            str(output_dir),
        ]
    )

    print("Checking out pinned revision...")
    run(
        [
            "git",
            "checkout",
            "--detach",
            HQCM_REVISION,
        ],
        cwd=output_dir,
    )

    checked_out = run(
        ["git", "rev-parse", "HEAD"],
        cwd=output_dir,
    )

    if checked_out != HQCM_REVISION:
        raise RuntimeError(
            "Revision verification failed:\n"
            f"expected: {HQCM_REVISION}\n"
            f"found:    {checked_out}"
        )

    remote = run(
        ["git", "remote", "get-url", "origin"],
        cwd=output_dir,
    )

    print()
    print("HQCM retrieved successfully.")
    print(f"Origin: {remote}")
    print(f"HEAD:   {checked_out}")
    print(f"Path:   {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
