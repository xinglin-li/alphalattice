"""Recorded identity moves: an old value that still names the current authority.

An identity built from source moves whenever a file its closure tracks changes. Most moves
change meaning, and a record made before one is historical. Some do not: a build file leaves
a closure while every library stays bound by its installed version, and the policy the old
value named is the policy the new value names. Such a move is recorded once, with its
reason and the change that made it, in ``config/identity-successors.json`` -- a committed
file no closure tracks, so recording a move never moves an identity itself.

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
from functools import cache
from pathlib import Path

from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root

IDENTITY_SUCCESSORS_PATH = Path("config") / "identity-successors.json"
_SCHEMA = "identity-successors"
_VERSION = 1


class IdentitySuccessorError(ValueError):
    """The recorded moves cannot be read as this build reads them."""


@dataclass(frozen=True, slots=True)
class IdentityMove:
    """One recorded move of one identity: the old value names what the new one names."""

    role: str
    predecessor: str
    successor: str
    change: str
    reason: str


@cache
def recorded_moves(root: Path | None = None) -> tuple[IdentityMove, ...]:
    """Every recorded move, as committed beside this build."""
    base = resolve_playpen_root(Path(__file__)) if root is None else root
    path = base / IDENTITY_SUCCESSORS_PATH
    if not path.is_file():
        return ()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise IdentitySuccessorError("shared_kernel.identity_successors_unreadable") from error
    if document.get("schema") != _SCHEMA or document.get("version") != _VERSION:
        raise IdentitySuccessorError("shared_kernel.identity_successors_from_another_build")
    moves = []
    for entry in document.get("moves", ()):
        try:
            move = IdentityMove(
                role=str(entry["role"]),
                predecessor=str(entry["predecessor"]),
                successor=str(entry["successor"]),
                change=str(entry["change"]),
                reason=str(entry["reason"]),
            )
        except (KeyError, TypeError) as error:
            raise IdentitySuccessorError("shared_kernel.identity_successor_invalid") from error
        if move.predecessor == move.successor or not all(
            len(value) == 64 and all(c in "0123456789abcdef" for c in value)
            for value in (move.predecessor, move.successor)
        ):
            raise IdentitySuccessorError("shared_kernel.identity_successor_invalid")
        moves.append(move)
    return tuple(moves)


def latest(role: str, value: str, *, root: Path | None = None) -> str:
    """The value recorded moves lead to from ``value`` for ``role``: itself when none do.

    A key built from ``latest`` on both sides matches exactly when ``is_current`` would,
    because nothing succeeds the installed value; an index keyed by a recorded value uses it
    so that a recorded move keeps its entries found.
    """
    successors = {m.predecessor: m.successor for m in recorded_moves(root) if m.role == role}
    seen = {value}
    while value in successors:
        value = successors[value]
        if value in seen:
            break
        seen.add(value)
    return value


def predecessors(role: str, value: str, *, root: Path | None = None) -> tuple[str, ...]:
    """``value`` and every recorded value whose moves lead to it for ``role``, nearest first.

    A store keyed by a recorded value finds, through these, the entries sealed before a
    recorded move: each is current exactly when ``value`` is the installed one.
    """
    back: dict[str, list[str]] = {}
    for move in recorded_moves(root):
        if move.role == role:
            back.setdefault(move.successor, []).append(move.predecessor)
    found = [value]
    for current in found:
        found.extend(v for v in back.get(current, ()) if v not in found)
    return tuple(found)


def recorded_origin(role: str, value: str, *, root: Path | None = None) -> str:
    """The value ``value`` was reached from by recorded moves for ``role``: itself when none.

    A value bound where it is compared by equality (a Program's authorities, a captured
    source) is held at its origin, as the switch table holds a byte value: a recorded move
    leaves it where it was, so everything sealed before the move stays current, and an
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
    back: dict[str, str] = {}
    for move in recorded_moves(root):
        if move.role == role:
            back.setdefault(move.successor, move.predecessor)
    seen = {value}
    while value in back:
        value = back[value]
        if value in seen:
            break
        seen.add(value)
    return value


def is_current(role: str, recorded: str, installed: str, *, root: Path | None = None) -> bool:
    """Whether a recorded value names the installed authority for ``role``."""
    if recorded == installed:
        return True
    successors = {m.predecessor: m.successor for m in recorded_moves(root) if m.role == role}
    seen = {recorded}
    value = recorded
    while value in successors:
        value = successors[value]
        if value == installed:
            return True
        if value in seen:
            return False
        seen.add(value)
    return False
