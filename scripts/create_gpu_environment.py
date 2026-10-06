"""Create the declared Playpen GPU retrieval environment, reproducibly and locally.

The enhanced retrieval recipes (the Qwen3 encoder and reranker) run on one
supported local GPU backend: PyTorch with CUDA from the PyTorch wheel index
and the transformers loader. Neither belongs in the default `.venv` or in the
CPU retrieval environment, so this builds a third worktree-local environment
the same way `create_retrieval_environment.py` does: the committed `uv.lock`
first, then the pinned retrieval additions, then the hash-locked GPU
additions from `config/requirements-gpu.lock`. Reproducible from committed
files; no parent project, lock or interpreter is consulted; a missing lock is
a stop.

    .venv/Scripts/python.exe scripts/create_gpu_environment.py [--offline]

Building it is setup for the enhanced recipes, not a prerequisite for the
CPU recipes: the product runs without it, and a missing GPU runtime is an
explicit capability refusal, never a silent fallback to another model.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PLAYPEN = Path(__file__).resolve().parents[1]
if str(PLAYPEN) not in sys.path:
    sys.path.insert(0, str(PLAYPEN))
from scripts.create_retrieval_environment import refusal  # noqa: E402

LOCK = PLAYPEN / "uv.lock"
ENVIRONMENT = PLAYPEN / ".venv-gpu"
RETRIEVAL_ADDITIONS = PLAYPEN / "config" / "requirements-retrieval.lock"
GPU_ADDITIONS = PLAYPEN / "config" / "requirements-gpu.lock"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def _uv() -> str:
    return os.environ.get("UV", "uv")


def interpreter_path() -> Path:
    return ENVIRONMENT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def create(*, offline: bool) -> Path:
    """Build the environment; return the interpreter that owns it."""

    for required in (LOCK, RETRIEVAL_ADDITIONS, GPU_ADDITIONS):
        if not required.is_file():
            raise FileNotFoundError(required)
    interpreter = interpreter_path()
    environment = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(ENVIRONMENT)}
    offline_flag = ("--offline",) if offline else ()
    subprocess.run(
        (
            _uv(),
            "sync",
            "--project",
            str(PLAYPEN),
            "--locked",
            "--all-extras",
            "--group",
            "dev",
            *offline_flag,
        ),
        check=True,
        cwd=PLAYPEN,
        env=environment,
    )
    subprocess.run(
        (
            _uv(),
            "pip",
            "install",
            "--python",
            str(interpreter),
            "-r",
            str(RETRIEVAL_ADDITIONS),
            *offline_flag,
        ),
        check=True,
        cwd=PLAYPEN,
        env=environment,
    )
    subprocess.run(
        (
            _uv(),
            "pip",
            "install",
            "--python",
            str(interpreter),
            "--require-hashes",
            "--index-url",
            TORCH_INDEX,
            "--extra-index-url",
            "https://pypi.org/simple",
            "--index-strategy",
            "unsafe-best-match",
            "-r",
            str(GPU_ADDITIONS),
            *offline_flag,
        ),
        check=True,
        cwd=PLAYPEN,
        env=environment,
    )
    return interpreter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="resolve the locks and the additions from the local uv cache only",
    )
    arguments = parser.parse_args()
    try:
        interpreter = create(offline=arguments.offline)
    except Exception as error:
        # The retrieval environment's refusal, for this environment: its failure in words (V539).
        answer = refusal(
            error,
            offline=arguments.offline,
            kind="gpu_environment",
            environment=ENVIRONMENT,
            locks=(LOCK, RETRIEVAL_ADDITIONS, GPU_ADDITIONS),
        )
        print(json.dumps(answer, indent=1))
        return 2
    print(interpreter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
