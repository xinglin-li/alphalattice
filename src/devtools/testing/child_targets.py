"""Child-process targets that tests start by name.

A child imports its target by module name in a fresh interpreter. A source module resolves to
this tree's own code there; a test module (`tests.*`) resolves through whatever `tests` package
the child's import path finds first, which may be another checkout's.
"""

from __future__ import annotations

import os
from pathlib import Path


def exit_at_start(*, cancelled: object) -> None:
    """End the child at once, with code 7, as a worker that dies before its body runs."""
    os._exit(7)


def file_size(path: str, cancelled: object) -> int:
    """A worker's data read: the size in bytes of the file at `path`."""
    return len(Path(path).read_bytes())
