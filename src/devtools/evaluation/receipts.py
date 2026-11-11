"""Shared numeric and atomic receipt operations for evaluation."""

import json
import math
import os
import time
from pathlib import Path


def number(value: object) -> bool:
    """Only finite nonnegative recorded numbers are additive measurements."""
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def write(path: Path, value: object) -> None:
    temporary = path.with_suffix(".pending")
    temporary.write_text(
        json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    # Shared readers and scanners briefly deny a Windows atomic replacement.
    for delay in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.3, 0.0):
        try:
            os.replace(temporary, path)
            return
        except OSError as exc:
            if os.name != "nt" or getattr(exc, "winerror", None) not in {5, 32, 33} or not delay:
                raise
            time.sleep(delay)
