"""Checkout shim for the AlphaLattice root command."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alphalattice.control.product_host.composition.entry import (  # noqa: E402, F401
    main,
    restore,
    sandbox,
    serve,
)

if __name__ == "__main__":
    raise SystemExit(main())
