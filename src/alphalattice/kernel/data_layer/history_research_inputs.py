"""Keep and carry the research-input bindings a workspace holds."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from alphalattice.kernel.data_layer.history_runtime import (
    _array_reader,
    _held_log,
    _once,
    _open,
    _query,
    held,
    holding,
)
from alphalattice.kernel.data_layer.log import attach
from alphalattice.kernel.data_layer.root import (
    data_root,
)


def research_input_root(binding_hash: str, workspace: Path) -> Path:
    """The `research-inputs/` folder the door reads a binding from for `workspace`: its kept view
    on this data root (data_layer.inputs). A binding this root does not keep is refused by name:
    no read falls back to another store. A workspace the log does not know shares the view `-`,
    as it has no selection."""
    import alphalattice.kernel.data_layer.inputs as inputs  # lazy: BLAKE3

    with _array_reader() as (root, read):
        known = read(
            "SELECT workspace_id FROM workspace WHERE folder = ? AND status = 'active'",
            (str(workspace.resolve()),),
        )
        return inputs.view(root, read, binding_hash, str(known[0][0]) if known else "-")


def research_input_workspace(holder: Path) -> Path | None:
    """The workspace folder a view's `views/<workspace_id>` stands for, as the log records it."""
    with _array_reader() as (_root, read):
        rows = read("SELECT folder FROM workspace WHERE workspace_id = ?", (holder.name,))
        return Path(rows[0][0]) if rows and rows[0][0] else None


def research_input_kept(binding_hash: str) -> bool:
    """Whether this data root keeps the binding (data_layer.inputs)."""
    import alphalattice.kernel.data_layer.inputs as inputs  # lazy: BLAKE3

    with _array_reader() as (root, read):
        return inputs.kept(root, read, binding_hash)


def keep_research_input(binding_hash: str, files: Iterable[tuple[str, Path]]) -> None:
    """Keep a bound input's files at their paths under `research-inputs/` through the Host's log
    (data_layer.inputs.keep_binding), held by the Host's workspace."""
    import alphalattice.kernel.data_layer.inputs as inputs  # lazy: BLAKE3

    with _once:
        with _held_log() as found:
            current = found
        log, workspace, root = current if current is not None else (held(), None, data_root())
        inputs.keep_binding(root, log, binding_hash, files, workspace=workspace)


def hold_research_input(binding_hash: str) -> None:
    """The Host's workspace's hold on a binding this root already keeps (data_layer.inputs), once,
    so a second workspace that binds it keeps it too; nothing while no workspace is open."""
    import alphalattice.kernel.data_layer.inputs as inputs  # lazy: BLAKE3

    with _once:
        with _held_log() as found:
            current = found
        if current is None or current[1] is None:
            return
        log, workspace, root = current
        inputs.hold(root, log, binding_hash, workspace)
        log.flush()


def carry_research_inputs(workspace: Path, copy: Path, *, root: Path | None = None) -> int:
    """Write the bindings the log (`root`'s when named) holds for `workspace` into its copy `copy`
    (data_layer.inputs.carry); the Host that admits the copy (`move_carried`) keeps them on the
    log its process holds. Returns the bindings carried."""
    import alphalattice.kernel.data_layer.inputs as inputs  # lazy: BLAKE3

    where = data_root() if root is None else root

    def carried(read: Callable[..., Any]) -> int:
        known = read(
            "SELECT workspace_id FROM workspace WHERE folder = ? AND status = 'active'",
            (str(workspace.resolve()),),
        )
        return int(inputs.carry(read, where, str(known[0][0]), copy)) if known else 0

    log = holding()
    if log is not None and Path(_open["root"]) == Path(where):
        return carried(log.read)
    if not (where / "machine" / "log.sqlite").is_file():
        return 0
    with closing(attach(where)) as con:
        return carried(_query(con))
