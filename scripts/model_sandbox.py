"""Checkout import for the product model sandbox."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alphalattice.control.product_host.composition.model_sandbox import (  # noqa: E402, F401
    run_sandbox,
)
