"""A checkout whose first-use setup is done, as a test or a golden build states it."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch


@contextmanager
def setup_done(root: Path) -> Iterator[dict[str, str]]:
    """Provide the CLI, in-process retrieval loaders and a subprocess stub environment."""
    stub = root / "setup-done"
    (stub / "fastembed").mkdir(parents=True, exist_ok=True)
    (stub / "fastembed" / "__init__.py").write_text("", encoding="utf-8")
    env = {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        "PYTHONPATH": str(stub) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    with ExitStack() as stack:
        for target in (
            "alphalattice.interface.local_application.native_bridge.load",
            "alphalattice.control.product_host.composition.evidence_authority_setup.load",
            "alphalattice.interface.local_application.retrieval_environment.load",
        ):
            stack.enter_context(patch(target, lambda: True))
        yield env
