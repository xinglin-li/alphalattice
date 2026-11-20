"""Keep and read derived-file partitions through the Host log."""

from __future__ import annotations

from collections.abc import Callable, Collection
from datetime import timedelta
from pathlib import Path

import alphalattice.kernel.data_layer.derived_files as derived_files
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.history_runtime import (
    _attached,
    _mark_reads,
    _open,
    _read_lock,
    _reads,
    recording,
    store,
)
from alphalattice.kernel.data_layer.log import attach
from alphalattice.kernel.data_layer.root import (
    data_root,
)


def derived_file_body(table: str, digest: str) -> Path | None:
    """The body the log names for `table`'s file partition with `digest`, or None
    (derived_files.found): read on the Host's log, or outside a Host on its data root's, attached
    read-only once per process and root."""
    log, root = _open.get("log"), _open.get("root")
    if log is not None and root is not None:
        held = derived_files.found(log.read, Path(root), table, digest)
        if held is not None:
            with _read_lock:  # its last read, marked in one batch (_mark_reads)
                _reads.setdefault(table, set()).add(digest)
        return held
    where = data_root()
    with _read_lock:
        held = _attached.get("log")
        if held is None or held[0] != where:
            if held is not None:
                held[1].close()
            held = _attached["log"] = (where, attach(where, shared=True))
        reader = held[1]
        return derived_files.found(
            lambda sql, args: reader.execute(sql, args).fetchall(), where, table, digest
        )


def root_derived_files(
    table: str,
    digests: Collection[str],
    *,
    unread_for: timedelta,
    verified: Callable[[str], bool],
) -> dict[str, int] | None:
    """In the Host, make the file partitions with these digests its workspace's only roots of
    `table`, evict what has gone unread for `unread_for` where `verified` holds its source, and
    collect what no workspace roots (StorePort.root_files), the reads noted marked first. None
    outside a Host, whose roots wait for the Host's next call; nothing it keeps is lost by that."""
    port = store()
    if port is None or not recording():
        return None
    _mark_reads(_open["log"])
    return port.root_files(
        table, str(_open["workspace"]), digests, unread_for=unread_for, verified=verified
    )


def keep_derived_file(table: str, *, recipe_key: str, part: str, digest: str, source: Path) -> None:
    """Hand the file `source` to the Host's StorePort to keep as a derived partition, rooted by
    the Host's workspace, on its own worker (StorePort.keep_file_later): the caller never waits
    on the keep. Refused by name outside a Host, which alone writes the data root."""
    port = store()
    if port is None or not recording():
        raise DataError(
            f"only the Host keeps {table}'s partitions in the data root; open this workspace in"
            " the Host, which keeps it, or read it from the workspace's own file",
            code="data_layer.derived_files_need_host",
        )
    port.keep_file_later(
        table,
        recipe_key=recipe_key,
        part=part,
        digest=digest,
        source=source,
        rooted_by=str(_open["workspace"]),
    )
