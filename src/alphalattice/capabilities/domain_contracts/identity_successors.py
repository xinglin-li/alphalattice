"""Recorded identity moves: an old value that still names the current authority.

An identity built from source moves whenever a file its closure tracks changes. Most moves
change meaning, and a record made before one is historical. Some do not: a build file leaves
a closure while every library stays bound by its installed version, and the policy the old
value named is the policy the new value names. Such a move is recorded once, with its
reason and the change that made it, under ``config/identity-successors/``. The frozen
``config/identity-successors.json`` is the first set. No closure tracks the sets.

``is_current`` is the one check an owner uses where it would compare a recorded value with
the installed one for current use (reuse, handoff, a new plan's inputs): the record is
current when its value is the installed one, or when recorded moves lead from it to the
installed one. Nothing is inferred: an unrecorded value is not current, and a move is
recorded only for a named role, predecessor and successor. A record's own bytes are still
verified by its owner; this answers only which authority the recorded value names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

from alphalattice.capabilities.domain_contracts.project_layout import resolve_playpen_root
from alphalattice.kernel.identity import is_blake3

IDENTITY_SUCCESSORS_PATH = Path("config") / "identity-successors"
FROZEN_SUCCESSORS_PATH = IDENTITY_SUCCESSORS_PATH.with_suffix(".json")
_SCHEMA = "identity-successors"
_VERSION = 1


@lru_cache(maxsize=4096)
def _audit_digest(value: str) -> str | None:
    """Classify a digest when its first raw spelling enters the successor owner."""
    if is_blake3(value):
        return "blake3"
    if len(value) == 64 and all(character in "0123456789abcdef" for character in value):
        return "historical"
    return None


class IdentitySuccessorError(ValueError):
    """The recorded moves cannot be read as this build reads them."""

    def __init__(self, code: str, detail: str | Path = "") -> None:
        """Keep the registered failure code and identify the refused input."""
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True, slots=True)
class IdentityMove:
    """One recorded move of one identity: the old value names what the new one names."""

    role: str
    predecessor: str
    successor: str
    change: str
    reason: str


def _read_set(path: Path) -> list[tuple[IdentityMove, bool]]:
    """Decode one committed set or name the file the reader refused."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise IdentitySuccessorError(
            "shared_kernel.identity_successors_unreadable", path
        ) from error
    if (
        not isinstance(document, dict)
        or document.get("schema") != _SCHEMA
        or document.get("version") != _VERSION
    ):
        raise IdentitySuccessorError("shared_kernel.identity_successors_from_another_build", path)
    entries = document.get("moves")
    if not isinstance(entries, list):
        raise IdentitySuccessorError("shared_kernel.identity_successor_invalid", path)
    moves = []
    for entry in entries:
        try:
            move = IdentityMove(
                role=str(entry["role"]),
                predecessor=str(entry["predecessor"]),
                successor=str(entry["successor"]),
                change=str(entry["change"]),
                reason=str(entry["reason"]),
            )
        except (KeyError, TypeError) as error:
            raise IdentitySuccessorError(
                "shared_kernel.identity_successor_invalid", path
            ) from error
        predecessor_kind = _audit_digest(move.predecessor)
        successor_kind = _audit_digest(move.successor)
        if move.predecessor == move.successor or predecessor_kind is None or successor_kind is None:
            raise IdentitySuccessorError("shared_kernel.identity_successor_invalid", path)
        moves.append((move, predecessor_kind == successor_kind == "blake3"))
    return moves


@dataclass(frozen=True, slots=True)
class _SuccessorLedger:
    recorded: tuple[IdentityMove, ...]
    effective: tuple[IdentityMove, ...]
    forward: dict[tuple[str, str], str]
    back: dict[tuple[str, str], list[str]]
    origins: dict[tuple[str, str], str]


@cache
def _ledger(root: Path | None) -> _SuccessorLedger:
    """Read and index the frozen ledger first, then sets in filename and row order.

    A role's value can move back to one it held before; its recorded moves then lead from that
    value around to itself. A later move from such a revisited value starts a new link: it
    supersedes the earlier move from it, so every earlier value in the chain leads to the newest
    (``latest`` and ``is_current`` read the later move). A repeated predecessor whose earlier
    moves do not lead back to it is refused by name.
    """
    base = resolve_playpen_root(Path(__file__)) if root is None else root
    directory = base / IDENTITY_SUCCESSORS_PATH
    try:
        paths = sorted(path for path in directory.iterdir() if path.suffix == ".json")
    except FileNotFoundError:
        paths = []
    except OSError as error:
        raise IdentitySuccessorError(
            "shared_kernel.identity_successors_unreadable", directory
        ) from error
    frozen = base / FROZEN_SUCCESSORS_PATH
    try:
        frozen.stat()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise IdentitySuccessorError(
            "shared_kernel.identity_successors_unreadable", frozen
        ) from error
    else:
        paths.insert(0, frozen)
    moves: list[IdentityMove] = []
    current: dict[tuple[str, str], str] = {}  # includes historical audit moves
    effective: dict[tuple[str, str], IdentityMove] = {}
    origins: dict[tuple[str, str], str] = {}
    for path in paths:
        for move, current_format in _read_set(path):
            key = (move.role, move.predecessor)
            if key in current and not _returns(current, move.role, current[key], move.predecessor):
                raise IdentitySuccessorError(
                    "shared_kernel.identity_successor_invalid",
                    f"duplicate {move.role} {move.predecessor} in {path}",
                )
            current[key] = move.successor
            moves.append(move)
            if current_format:
                # Reinsert in recorded order when a later move supersedes this one.
                effective.pop(key, None)
                effective[key] = move
                origins.setdefault((move.role, move.successor), move.predecessor)
    back: dict[tuple[str, str], list[str]] = {}
    for move in effective.values():
        back.setdefault((move.role, move.successor), []).append(move.predecessor)
    return _SuccessorLedger(
        recorded=tuple(moves),
        effective=tuple(effective.values()),
        forward={key: move.successor for key, move in effective.items()},
        back=back,
        origins=origins,
    )


def recorded_moves(root: Path | None = None) -> tuple[IdentityMove, ...]:
    """The admitted moves in recorded order, including historic audit identities."""
    return _ledger(root).recorded


def _returns(current: dict[tuple[str, str], str], role: str, start: str, target: str) -> bool:
    """Whether the moves recorded so far lead from ``start`` back to ``target`` for ``role``."""
    value, seen = start, {start}
    while value != target:
        value = current.get((role, value), "")
        if not value or value in seen:
            return False
        seen.add(value)
    return True


def effective_moves(root: Path | None = None) -> tuple[IdentityMove, ...]:
    """The BLAKE3 moves in force, in the order of their last admitted records."""
    return _ledger(root).effective


def _last(links: dict[tuple[str, str], str], role: str, value: str) -> str:
    seen = {value}
    while (next_value := links.get((role, value))) is not None:
        value = next_value
        if value in seen:
            break
        seen.add(value)
    return value


def latest(role: str, value: str, *, root: Path | None = None) -> str:
    """The value recorded moves lead to from ``value`` for ``role``: itself when none do.

    A key built from ``latest`` on both sides matches exactly when ``is_current`` would,
    because nothing succeeds the installed value; an index keyed by a recorded value uses it
    so that a recorded move keeps its entries found.
    """
    return _last(_ledger(root).forward, role, value)


def predecessors(role: str, value: str, *, root: Path | None = None) -> tuple[str, ...]:
    """``value`` and every recorded value whose moves lead to it for ``role``, nearest first.

    A store keyed by a recorded value finds, through these, the entries sealed before a
    recorded move: each is current exactly when ``value`` is the installed one.
    """
    back = _ledger(root).back
    found = [value]
    for current in found:
        found.extend(v for v in back.get((role, current), ()) if v not in found)
    return tuple(found)


def recorded_origin(role: str, value: str, *, root: Path | None = None) -> str:
    """The value ``value`` was reached from by recorded moves for ``role``: itself when none.

    A value bound where it is compared by equality (a Program's authorities, a captured
    source) is held at its BLAKE3 origin: a recorded move leaves it where it was, so everything
    sealed under BLAKE3 before the move stays current, and an
    unrecorded move is a new value (LAWS.md ID1). Moves are recorded forward, from the value
    the last readout held; where two recorded moves lead to one value, the first recorded
    is followed.

    Args:
        role: The identity's role.
        value: Its installed value.
        root: The checkout whose successor records are read.

    Returns:
        The origin of ``value``.
    """
    return _last(_ledger(root).origins, role, value)


def is_current(role: str, recorded: str, installed: str, *, root: Path | None = None) -> bool:
    """Whether a recorded value names the installed authority for ``role``."""
    if not isinstance(recorded, str) or not isinstance(installed, str):
        return False
    if _audit_digest(recorded) != "blake3" or _audit_digest(installed) != "blake3":
        return False
    if recorded == installed:
        return True
    # The admitted links and comparison values need no format scan here.
    successors = _ledger(root).forward
    seen = {recorded}
    value = recorded
    while (next_value := successors.get((role, value))) is not None:
        value = next_value
        if value == installed:
            return True
        if value in seen:
            return False
        seen.add(value)
    return False
