"""Checkout shim for the product install_retrieval_pack setup owner."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alphalattice.control.product_host.composition import (  # noqa: E402
    retrieval_pack_setup as _owner,
)

if __name__ == "__main__":
    raise SystemExit(_owner.main())
else:
    sys.modules[__name__] = _owner
