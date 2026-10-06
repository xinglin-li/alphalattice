"""The Evidence binding tuple moves only with its successors recorded or its predecessor
listed (binding plan, E2; LAWS.md ID1).

Six bindings seal every Alternative Evidence analysis, each an identity role. A change that
keeps what they mean records each moved binding's successor
(`scripts/identity_readout.py --record`), and the analyses sealed under the previous tuple
stay current. A change that moves a meaning makes them history: they read back when the
tuple is listed in ``SUPPORTED_HISTORICAL_BINDINGS``, with the change that retired it.
``config/evidence-binding-tuples.json`` records the current tuple and every previous one
listed as history; this check refuses a tree whose computed tuple the recorded current one
neither is nor leads to by recorded moves, or whose recorded previous tuples are not all
listed.

    python -m devtools.architecture.evidence_bindings [--record]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

RECORD_PATH = Path("config") / "evidence-binding-tuples.json"
_SCHEMA = "evidence-binding-tuples"


def problems(root: Path) -> list[str]:
    """What is wrong with the recorded tuples at ``root``; empty when all holds."""

    from alphalattice.evidence.alternative_evidence.publication.contracts import (
        bindings_current,
    )
    from alphalattice.evidence.alternative_evidence.runtime.identity import (
        SUPPORTED_HISTORICAL_BINDINGS,
        current_evidence_binding_tuple,
    )

    record = json.loads((root / RECORD_PATH).read_text(encoding="utf-8"))
    if record.get("schema") != _SCHEMA or record.get("version") != 1:
        return [f"{RECORD_PATH} is not an {_SCHEMA} record of version 1"]
    current = current_evidence_binding_tuple(root)
    listed = {value.bindings for value in SUPPORTED_HISTORICAL_BINDINGS}
    found = []
    if tuple(record["current"]) != current and not bindings_current(record["current"], current):
        found.append(
            "the Evidence binding tuple moved: record each moved binding's successor when "
            "the change keeps its meaning (scripts/identity_readout.py --record), or list the "
            "recorded current tuple in SUPPORTED_HISTORICAL_BINDINGS (runtime/identity.py) "
            f"with the change that moved it and move it to 'previous' in {RECORD_PATH}; then "
            "record the new one (python -m devtools.architecture.evidence_bindings --record)"
        )
    for previous in record.get("previous", ()):
        if tuple(previous) not in listed:
            found.append(
                f"a previous Evidence tuple is not listed as history: {[v[:8] for v in previous]}"
            )
    return found


def record_current(root: Path) -> None:
    """Record the computed tuple as current; the one it replaces goes to 'previous' unless
    recorded moves lead from it to the computed one (then its analyses stay current)."""

    from alphalattice.evidence.alternative_evidence.publication.contracts import (
        bindings_current,
    )
    from alphalattice.evidence.alternative_evidence.runtime.identity import (
        current_evidence_binding_tuple,
    )

    path = root / RECORD_PATH
    record: dict[str, Any] = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {"schema": _SCHEMA, "version": 1, "current": None, "previous": []}
    )
    current = list(current_evidence_binding_tuple(root))
    if record["current"] not in (None, current) and not bindings_current(
        record["current"], current
    ):
        record["previous"] = [*record["previous"], record["current"]]
    record["current"] = current
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--record", action="store_true", help="Record the computed tuple.")
    args = parser.parse_args(argv)
    if args.record:
        record_current(args.root)
    found = problems(args.root)
    for line in found:
        print(line)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
