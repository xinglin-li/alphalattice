"""A checkout whose first-use setup is done, as a test or a golden build states it."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from alphalattice.interface.local_application import retrieval_environment


@contextmanager
def setup_done(root: Path) -> Iterator[dict[str, str]]:
    """Its `alphalattice` command on PATH and its retrieval runtime readable: `load` itself in this
    process, and for a client in a subprocess the returned environment's stub module."""
    stub = root / "setup-done"
    (stub / "fastembed").mkdir(parents=True, exist_ok=True)
    (stub / "fastembed" / "__init__.py").write_text("", encoding="utf-8")
    env = {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        "PYTHONPATH": str(stub) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    with patch.object(retrieval_environment, "load", lambda: True):
        yield env
