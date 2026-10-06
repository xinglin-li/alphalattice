"""Checkout shim for the Local Web launcher."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alphalattice.control.product_host.composition.web_launcher import (  # noqa: E402, F401
    LocalPortfolioWebSession,
    _wait_for_stop_on_stdin,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
