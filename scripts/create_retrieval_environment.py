"""Checkout shim for the declared retrieval environment setup."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alphalattice.interface.local_application import retrieval_environment as _owner  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(_owner.main())
else:
    sys.modules[__name__] = _owner
