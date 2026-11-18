"""Derived files: a derived partition held as one immutable file rather than as catalog rows.

A Panel chunk is one. Its body is the chunk's own Parquet, byte for byte, at
blobs/<table>/<digest><suffix> under the data root; the log's cache_entry keys it (recipe, table,
part, digest) and names the file by `body`, its handle, and a workspace that reads it roots it in
cache_root. A body is placed once per digest: workspaces holding the same partition share one
file.

Write order: the entry and its root are committed, then the body is placed: a hard link to the
source where the volume allows (the same bytes, no copy), else a copy written beside its place and
renamed. A crash between leaves an entry whose body is absent, which a read takes as not held and
the next keep places; a copy a crash left half written keeps its staged name, which the next keep
overwrites.

A read finds a body by its digest alone, on cache_entry's digest index (`found`), so it reads no
file to find it: in the Host on its log, elsewhere through a read-only attach of it.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from alphalattice.kernel.data_layer.log import HostLog

TABLES: Final = {"feature_panel": ".parquet"}
"""The derived tables held as files, each partition one file, by its file suffix; StorePort keeps
no other."""
FOLDER: Final = "blobs"
STAGED: Final = ".staging"
"""A copied body's suffix while it is written beside its place."""

type Read = Callable[[str, tuple[Any, ...]], list[Any]]
"""A query on the log: the Host's `HostLog.read`, or an attached reader's."""


def handle(table: str, digest: str) -> str:
    """A file partition's handle under the data root, as cache_entry's `body` names it: its
    table's folder, its digest and suffix."""
    return f"{FOLDER}/{table}/{digest}{TABLES[table]}"


def keep(
    root: Path,
    log: HostLog,
    table: str,
    *,
    recipe_key: str,
    part: str,
    digest: str,
    source: Path,
    rows: int,
    workspace: str | None,
) -> Path:
    """Hold the file `source` as the derived partition (`table`, `recipe_key`, `part`, `digest`),
    rooted by `workspace`; returns the body's path. The body is placed once: a digest already held
    adds only its entry and root."""
    name = handle(table, digest)
    target = root / name
    log.cached(
        table, recipe_key, [(part, digest, rows, source.stat().st_size)], compute_ms=0, body=name
    )
    if workspace is not None:
        log.rooted(workspace, table, [(recipe_key, part, digest)])
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}{STAGED}")
        staged.unlink(missing_ok=True)
        try:
            os.link(source, staged)
        except OSError:  # another volume, or links refused: the same bytes by a copy
            shutil.copyfile(source, staged)
        os.replace(staged, target)
    return target


def found(read: Read, root: Path, table: str, digest: str) -> Path | None:
    """The body `table`'s log names for the partition with `digest`, or None while it holds none.
    The file is not touched here: the reader opens it, and takes one that is gone as not held."""
    held = read(
        "SELECT body FROM cache_entry WHERE derived_table = ? AND input_digest = ?"
        " AND body IS NOT NULL LIMIT 1",
        (table, digest),
    )
    return root / str(held[0][0]) if held else None


def unrooted(read: Read, table: str) -> list[str]:
    """The digests of `table`'s partitions no workspace roots, which collection drops."""
    held = read(
        "SELECT DISTINCT input_digest FROM cache_entry e WHERE derived_table = ? AND NOT EXISTS"
        " (SELECT 1 FROM cache_root r WHERE r.derived_table = e.derived_table"
        " AND r.input_digest = e.input_digest)",
        (table,),
    )
    return [str(row[0]) for row in held]


def unread(read: Read, table: str, workspace: str, before_us: int) -> list[str]:
    """The digests of `table`'s partitions `workspace` alone roots whose body is unread since
    `before_us`: those eviction may drop, once each one's source still verifies."""
    held = read(
        "SELECT DISTINCT e.input_digest FROM cache_entry e JOIN cache_root r"
        " ON r.derived_table = e.derived_table AND r.input_digest = e.input_digest"
        " WHERE e.derived_table = ? AND r.workspace_id = ? AND e.body IS NOT NULL"
        " AND e.last_read_us < ? AND NOT EXISTS (SELECT 1 FROM cache_root o"
        " WHERE o.derived_table = e.derived_table AND o.input_digest = e.input_digest"
        " AND o.workspace_id <> r.workspace_id)",
        (table, workspace, before_us),
    )
    return [str(row[0]) for row in held]


def drop(root: Path, log: HostLog, table: str, digests: list[str]) -> list[str]:
    """Delete these partitions' bodies, then their entries: a crash between leaves an entry
    whose body is gone, which a read takes as not held. A body another process holds open (a
    delete Windows refuses) keeps its entry for a later drop, and is logged with the count of
    such bodies. The digests dropped."""
    gone: list[str] = []
    held: list[tuple[str, OSError]] = []
    for digest in digests:
        try:
            (root / handle(table, digest)).unlink(missing_ok=True)
        except OSError as error:
            held.append((digest, error))
        else:
            gone.append(digest)
    if held:
        logging.getLogger(__name__).warning(
            "%d %s bodies are held open and stay for a later drop; the first: %s (%s)",
            len(held),
            table,
            held[0][0],
            held[0][1],
        )
    if gone:
        log.uncached(table, gone)
    return gone


__all__ = ["FOLDER", "STAGED", "TABLES", "drop", "found", "handle", "keep", "unread", "unrooted"]
