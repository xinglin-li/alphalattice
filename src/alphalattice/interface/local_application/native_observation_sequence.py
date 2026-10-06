"""Durable indexed producer retries, not an observation store or delivery queue."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from alphalattice.kernel.shared_kernel.persistence import fsync_directory

LEGACY_MAX_SEQUENCE_ENTRIES = 1024
LEGACY_MAX_SEQUENCE_BYTES = 300_000
LEGACY_SEQUENCE_NAME = "native-research-sequences.local.json"
SEQUENCE_NAME = "native-research-sequences.local.sqlite3"
LOCK_NAME = "native-research-sequences.local.lock"
_HASH = re.compile(r"[0-9a-f]{64}")
_TABLE_SQL = """CREATE TABLE native_observation_sequence (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    scope BLOB NOT NULL CHECK(length(scope) = 32),
    event_id BLOB NOT NULL CHECK(length(event_id) = 32),
    fingerprint BLOB NOT NULL CHECK(length(fingerprint) = 32),
    occurred_at TEXT NOT NULL,
    UNIQUE(scope, event_id)
)"""


class NativeSequenceError(ValueError):
    """Safe metadata failure, never authority to repeat research."""


def metadata_path(project: Path, name: str) -> Path:
    """Locate a sequence metadata file inside the project's Codex directory.

    Args:
        project: The admitted project root.
        name: The metadata filename.

    Returns:
        The contained metadata path.

    Raises:
        NativeSequenceError: If the path is a symlink or escapes the project.
    """
    path = project / ".codex" / name
    if path.is_symlink() or not path.resolve().is_relative_to(project.resolve()):
        raise NativeSequenceError("native_bridge.sequence_path_invalid")
    return path


def _legacy_rows(project: Path) -> list[list[str]]:
    """Read the previous bounded format only when initializing its successor."""
    path = metadata_path(project, LEGACY_SEQUENCE_NAME)
    try:
        with path.open("rb") as stream:
            raw = stream.read(LEGACY_MAX_SEQUENCE_BYTES + 1)
    except FileNotFoundError:
        return []
    if len(raw) > LEGACY_MAX_SEQUENCE_BYTES:
        raise NativeSequenceError("native_bridge.sequence_metadata_invalid")
    try:
        rows = json.loads(raw)
        if not isinstance(rows, list) or len(rows) > LEGACY_MAX_SEQUENCE_ENTRIES:
            raise ValueError
        keys = set()
        for row in rows:
            if not isinstance(row, list) or len(row) != 4:
                raise ValueError
            if any(not isinstance(value, str) for value in row):
                raise ValueError
            if any(_HASH.fullmatch(value) is None for value in row[:3]):
                raise ValueError
            stamp = datetime.fromisoformat(row[3])
            key = tuple(row[:2])
            if stamp.utcoffset() is None or key in keys:
                raise ValueError
            keys.add(key)
    except (ValueError, TypeError, RecursionError):
        raise NativeSequenceError("native_bridge.sequence_metadata_invalid") from None
    return rows


def reserve_sequence(
    project: Path, *, scope: str, event_id: str, fingerprint: str
) -> tuple[int, str]:
    """Caller holds the OS lock through reservation and its one delivery attempt.

    The indexed rows keep only hashes and the first receipt time. AUTOINCREMENT
    keeps a durable high-water mark even if a later retention owner prunes rows;
    this owner retains every retry and never evicts or renumbers one. SQLite's
    one-based key is returned zero-based to preserve the Host's existing sequences.
    The previous JSON, if present, migrates atomically with every sequence and time
    unchanged and remains untouched. This is not proof of native authorship.
    """
    if any(_HASH.fullmatch(value) is None for value in (scope, event_id, fingerprint)):
        raise NativeSequenceError("native_bridge.sequence_identity_invalid")
    path = metadata_path(project, SEQUENCE_NAME)
    metadata_path(project, SEQUENCE_NAME + "-journal")
    identity = (bytes.fromhex(scope), bytes.fromhex(event_id))
    content = bytes.fromhex(fingerprint)
    try:
        with closing(sqlite3.connect(path, timeout=2.0)) as store, store:
            store.execute("PRAGMA synchronous=FULL")
            store.execute("BEGIN IMMEDIATE")
            version = store.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                if store.execute("SELECT name FROM sqlite_master").fetchone() is not None:
                    raise NativeSequenceError("native_bridge.sequence_metadata_invalid")
                rows = _legacy_rows(project)
                store.execute(_TABLE_SQL)
                store.executemany(
                    "INSERT INTO native_observation_sequence VALUES (?, ?, ?, ?, ?)",
                    (
                        (index + 1, *(bytes.fromhex(value) for value in row[:3]), row[3])
                        for index, row in enumerate(rows)
                    ),
                )
                store.execute("PRAGMA user_version=1")
            elif version != 1:
                raise NativeSequenceError("native_bridge.sequence_metadata_invalid")
            schema = store.execute(
                "SELECT sql FROM sqlite_master WHERE name='native_observation_sequence'"
            ).fetchone()
            if schema != (_TABLE_SQL,):
                raise NativeSequenceError("native_bridge.sequence_metadata_invalid")
            row = store.execute(
                "SELECT sequence, fingerprint, occurred_at FROM native_observation_sequence "
                "WHERE scope=? AND event_id=?",
                identity,
            ).fetchone()
            if row is not None:
                if row[1] != content:
                    raise NativeSequenceError("native_bridge.event_identity_collision")
                stamp = datetime.fromisoformat(row[2])
                if row[0] < 1 or stamp.utcoffset() is None:
                    raise NativeSequenceError("native_bridge.sequence_metadata_invalid")
                result = (row[0] - 1, row[2])
            else:
                occurred_at = datetime.now(UTC).isoformat()
                inserted = store.execute(
                    "INSERT INTO native_observation_sequence "
                    "(scope, event_id, fingerprint, occurred_at) VALUES (?, ?, ?, ?)",
                    (*identity, content, occurred_at),
                )
                if inserted.lastrowid is None:
                    raise NativeSequenceError("native_bridge.sequence_metadata_invalid")
                result = (inserted.lastrowid - 1, occurred_at)
        fsync_directory(path.parent)
        return result
    except (sqlite3.Error, OSError, ValueError, TypeError) as error:
        if isinstance(error, NativeSequenceError):
            raise
        raise NativeSequenceError("native_bridge.sequence_metadata_invalid") from None
