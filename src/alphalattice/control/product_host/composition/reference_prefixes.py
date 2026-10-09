"""The hashes and ids the Host has answered a client with, kept for short references.

The client's compact display shows each hash or UUID by its first twelve
characters and reads such a beginning back as the whole value before a request leaves it
(`client.whole_references`), from this ledger: the Host keeps every whole hash and id it has
answered a client with, the newest `LIMIT`, in the workspace's runtime, so a restart keeps
them. The Host itself reads only whole values, so no request and no identity changes.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from alphalattice.interface.local_application.client import (
    REFERENCE_LEDGER_NAME,
    WHOLE_REFERENCE,
)

LIMIT = 200_000
"""The most values kept, the newest; an older one is sent whole."""


class ReferenceLedger:
    """The hashes and ids this workspace's Host has answered a client with."""

    def __init__(self, workspace: Path) -> None:
        """Keep the workspace's ledger, read once when first needed.

        Args:
            workspace: The workspace whose runtime holds the ledger.
        """
        self._path = workspace / "runtime" / REFERENCE_LEDGER_NAME
        self._lock = threading.Lock()
        self._values: dict[str, None] | None = None

    def _loaded(self) -> dict[str, None]:
        if self._values is None:
            values: dict[str, None] = {}
            try:
                for line in self._path.read_text(encoding="utf-8").splitlines():
                    if WHOLE_REFERENCE.fullmatch(line):
                        values[line] = None
            except OSError:
                pass
            self._values = values
        return self._values

    def record(self, answer: object) -> None:
        """Keep every whole hash and id the answer holds, in its strings at any depth.

        Args:
            answer: An owner's answer, as sent to the client.
        """
        found: list[str] = []
        _strings(answer, lambda text: found.extend(WHOLE_REFERENCE.findall(text)))
        if not found:
            return
        with self._lock:
            values = self._loaded()
            new = [value for value in dict.fromkeys(found) if value not in values]
            if not new:
                return
            values.update(dict.fromkeys(new))
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if len(values) > LIMIT:
                kept = list(values)[-LIMIT:]
                self._values = dict.fromkeys(kept)
                self._path.write_text(
                    "".join(f"{value}\n" for value in kept), encoding="utf-8", newline="\n"
                )
                return
            with self._path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write("".join(f"{value}\n" for value in new))


def _strings(value: object, visit: Callable[[str], None]) -> None:
    if isinstance(value, str):
        visit(value)
    elif isinstance(value, dict):
        for item in value.values():
            _strings(item, visit)
    elif isinstance(value, list | tuple):
        for item in value:
            _strings(item, visit)
