"""The scenario person's ASK as a bounded process: the native runner's only poster.

Read by the native runner; the person itself (scenario_person) starts no process, so its unit
tests stay process-free.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any


def poster(argv: list[str] | None) -> Callable[[Path, str], bool] | None:
    """The ASK as one bounded process per question, or None when no ASK command is set."""
    if argv is None:
        return None

    def post(record: Path, question: str) -> bool:
        command = [
            a.replace("{ask}", str(record)).replace("{question}", question[:120]) for a in argv
        ]
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
            said: dict[str, Any] = dict(exit=done.returncode, said=done.stderr[-300:])
        except (OSError, subprocess.TimeoutExpired) as error:
            said = dict(error=str(error))
        record.with_suffix(".posted.json").write_text(json.dumps(said), encoding="utf-8")
        return said.get("exit") == 0

    return post
