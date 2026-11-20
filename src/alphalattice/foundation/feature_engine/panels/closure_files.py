"""Where a Panel closure root's files are: its folder, or its workspace's kept files on the log.

A workspace's own closure root (`artifacts/feature-panel/closure`) moves onto the log at the
upgrade (`closure_upgrade_family`): every file, byte for byte, under its name in the workspace
(`kept_files`), in the folder's write order, and last the root's own kept entry (`MOVED`), its
durable cutover. From then on the closure store, the Feature ledger and the logical identity read
and write it there. A closure root anywhere else (a published output's, or one outside any
workspace) keeps its folder.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from alphalattice.kernel.data_layer import history
from alphalattice.kernel.data_layer.upgrade import PROGRESS, Family, workspace_files
from alphalattice.kernel.shared_kernel.persistence import (
    DURABLE_REPLACE_DELAYS,
    replace_with_retry,
)

CLOSURE_STORE_ENV: Final = "ALPHALATTICE_CLOSURE_STORE"
"""The workspace closure's reader: `log` reads its kept files; `legacy` reads its folder, for
rollback; unset, a workspace reads its folder only while that holds a closure the upgrade has not
moved (`MOVED`), and the log otherwise. Any other value is refused."""
CLOSURE: Final = Path("artifacts", "feature-panel", "closure")
"""A workspace's own closure root, under its folder."""
MANIFEST: Final = "research-workspace.json"
"""The file that marks a workspace's folder (the Host's RESEARCH_WORKSPACE_MANIFEST_NAME)."""
MOVED: Final = CLOSURE.as_posix()
"""The closure root's own kept entry, kept after its last file: the durable cutover. No closure file
has this name, since each lies under the root; a copy carries it with the files, and its event on
the copy's log precedes any upgrade run there."""


class ClosureFiles(Protocol):
    """A closure root's files, by their names under it (`category/<hash>.json`)."""

    def read(self, name: str) -> bytes | None:
        """The file's bytes, or None when it does not exist."""
        ...

    def exists(self, name: str) -> bool:
        """Whether the file exists, its bytes unread."""
        ...

    def write(self, name: str, data: bytes) -> None:
        """Write the file whole, replacing what it held."""
        ...

    def names(self, folder: str) -> list[str]:
        """The names of the files directly in `folder`, sorted."""
        ...

    def written(self, folder: str) -> list[str]:
        """The names of the files directly in `folder`, the one written last at the end."""
        ...

    def remove(self, name: str) -> None:
        """Remove the file; nothing when it does not exist."""
        ...


class FolderFiles:
    """A closure root's files in its folder."""

    def __init__(self, root: Path) -> None:
        """Read and write under `root`."""
        self.root = root

    def read(self, name: str) -> bytes | None:
        """The file's bytes, or None when it does not exist."""
        path = self.root / name
        return path.read_bytes() if path.is_file() else None

    def exists(self, name: str) -> bool:
        """Whether the file exists."""
        return (self.root / name).is_file()

    def write(self, name: str, data: bytes) -> None:
        """Stage the bytes beside the file and replace it, outliving a momentary Windows sharing
        violation within a bound; the staged file is removed even when the replace is refused."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        staged.write_bytes(data)
        try:
            replace_with_retry(staged, path, delays=DURABLE_REPLACE_DELAYS)
        finally:
            staged.unlink(missing_ok=True)

    def names(self, folder: str) -> list[str]:
        """The names of the files directly in `folder`, sorted; a staged file is none."""
        where = self.root / folder
        if not where.is_dir():
            return []
        return sorted(p.name for p in where.iterdir() if p.is_file() and not _staged(p.name))

    def written(self, folder: str) -> list[str]:
        """The names of the files directly in `folder`, by their last write's time."""
        return sorted(
            self.names(folder), key=lambda name: (self.root / folder / name).stat().st_mtime_ns
        )

    def remove(self, name: str) -> None:
        """Remove the file; nothing when it does not exist."""
        (self.root / name).unlink(missing_ok=True)


class KeptFiles:
    """A workspace closure root's files kept on the log, under their names in the workspace."""

    def __init__(self, workspace_id: str, prefix: str) -> None:
        """The workspace's kept files under `prefix`, the closure root's place in its folder."""
        self.workspace_id, self.prefix = workspace_id, prefix

    def read(self, name: str) -> bytes | None:
        """The file's bytes, proved, or None when it is not kept."""
        found: bytes | None = history.read_kept_file(self.workspace_id, f"{self.prefix}/{name}")
        return found

    def exists(self, name: str) -> bool:
        """Whether the file is kept, its body unread."""
        return history.kept_file_digest(self.workspace_id, f"{self.prefix}/{name}") is not None

    def write(self, name: str, data: bytes) -> None:
        """Keep the bytes as the file on the log this process holds."""
        history.keep_workspace_file(self.workspace_id, f"{self.prefix}/{name}", data)

    def names(self, folder: str) -> list[str]:
        """The names of the kept files directly in `folder`, sorted."""
        found: list[str] = history.kept_file_names(self.workspace_id, f"{self.prefix}/{folder}")
        return found

    def written(self, folder: str) -> list[str]:
        """The names of the kept files directly in `folder`, by their newest write."""
        name = f"{self.prefix}/{folder}"
        found: list[str] = history.kept_file_names(self.workspace_id, name, by_write=True)
        return found

    def remove(self, name: str) -> None:
        """Remove the file on the log this process holds."""
        history.remove_workspace_file(self.workspace_id, f"{self.prefix}/{name}")


def closure_store_mode() -> str | None:
    """The switch's value (`CLOSURE_STORE_ENV`), None when it is unset; any other value refuses."""
    mode = os.environ.get(CLOSURE_STORE_ENV)
    if mode in {None, "log", "legacy"}:
        return mode
    raise ValueError("feature_closure.closure_store_invalid")


def closure_files(root: Path) -> ClosureFiles:
    """The files of the closure root `root`: a workspace's own root reads its kept files, unless
    its folder holds a closure the upgrade has not moved yet (`MOVED`, `CLOSURE_STORE_ENV`); every
    other root reads its folder. A new workspace, its folder empty, starts on the log."""
    mode = closure_store_mode()
    workspace = _workspace_of(root)
    if mode == "legacy" or workspace is None:
        return FolderFiles(root)
    held = history.kept_file_workspace(workspace)
    if held is None:
        return FolderFiles(root)
    if mode is None and not _moved(held) and _unmoved(workspace):
        return FolderFiles(root)  # a 0.2.7 closure keeps its folder until the upgrade moves it
    return KeptFiles(held, CLOSURE.as_posix())


def _workspace_of(root: Path) -> Path | None:
    """The workspace whose own closure root `root` is, or None for any other root."""
    if root.parts[-len(CLOSURE.parts) :] != CLOSURE.parts:
        return None
    workspace = root.parents[len(CLOSURE.parts) - 1]
    return workspace if (workspace / MANIFEST).is_file() else None


def _moved(held: str) -> bool:
    """Whether the workspace's closure was moved onto the log: its cutover is kept."""
    return history.kept_file_digest(held, MOVED) is not None


def _moved_by(runner: Any, held: str) -> bool:
    """Whether `runner`'s run moved the workspace's closure: its cutover's event on this log
    follows the run's durable start (`upgrade.progress` `started`), which a resumed run keeps.
    A cutover carried by a copy, or moved by an earlier run, precedes it. A run with no start
    recorded (a family driven outside the runner) owns every cutover."""
    import alphalattice.kernel.data_layer.kept_files as kept_files  # lazy: the array door

    started = runner.log.read(
        "SELECT min(seq) FROM event WHERE type = ? AND subject_id = ?"
        " AND json_extract(attrs, '$.kind') = 'started'",
        (PROGRESS, runner.subject),
    )
    moved = runner.log.read(
        "SELECT max(seq) FROM event WHERE type = ? AND subject_id = ?",
        (kept_files.EVENT, kept_files.subject(held, MOVED)),
    )
    return started[0][0] is None or (moved[0][0] is not None and moved[0][0] > started[0][0])


def _root(workspace: Path) -> Path | None:
    """The workspace's closure root, None when it has none. The root and each folder from the
    workspace down to it must be the workspace's own: one that is a link, or resolves outside the
    workspace, is refused by name, as `workspace_files` refuses a linked directory."""
    where = workspace
    for part in CLOSURE.parts:
        where = where / part
        if where.is_symlink() or where.is_junction():
            raise ValueError(f"workspace closure root is an alias: {where}")
        if not where.is_dir():
            return None
    if not where.resolve().is_relative_to(workspace.resolve()):
        raise ValueError(f"workspace closure root is an alias: {where}")
    return where


def _members(workspace: Path) -> Iterator[tuple[Path, Path]]:
    """The closure root's files, each with the root, a staged file none. A file that is a link, or
    resolves outside the root, is refused by name: its bytes are not the workspace's."""
    root = _root(workspace)
    if root is None:
        return
    inside = root.resolve()
    for path in workspace_files(root):
        if _staged(path.name):
            continue
        if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(inside):
            raise ValueError(f"workspace closure file is an alias: {path}")
        yield root, path


def _unmoved(workspace: Path) -> bool:
    """Whether the workspace's folder holds any closure file, read up to the first one."""
    return next(_members(workspace), None) is not None


def _folder(workspace: Path) -> dict[str, Path]:
    """The workspace closure folder's files, by their names in the workspace, in their write order
    (`FolderFiles.written`'s)."""
    found = [
        (path.stat().st_mtime_ns, (CLOSURE / path.relative_to(root)).as_posix(), path)
        for root, path in _members(workspace)
    ]
    return {name: path for _written, name, path in sorted(found)}


def _kept(held: str) -> dict[str, str]:
    """The workspace's kept closure files, by their names in the workspace: their digests, in the
    order of their newest writes."""
    inside = f"{CLOSURE.as_posix()}/"
    return {n: d for n, d in history.kept_files_held(held).items() if n.startswith(inside)}


def _staged(name: str) -> bool:
    return name.startswith(".") and name.endswith(".tmp")


def _in_order(folder: dict[str, Path], kept: dict[str, str]) -> bool:
    """Whether each closure folder lists its files in the same write order on the log as in the
    workspace (`ClosureFiles.written`), so the newest file of a folder stays its newest."""

    def by_parent(names: list[str]) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for name in names:
            found.setdefault(name.rpartition("/")[0], []).append(name)
        return found

    return by_parent(list(folder)) == by_parent(list(kept))


def closure_upgrade_family(holder: Callable[[Path], str | None]) -> Family:
    """The upgrade's move of each workspace's closure folder onto the log: every file kept with
    its workspace (`holder` names its id) in its write order, then its cutover (`MOVED`), the
    folder left as it is. A workspace already moved (by an earlier run, or carried by a copy), or
    whose empty folder holds nothing to move, is not touched, so a later run that includes it
    keeps what was written since. Its ALL EQUAL reads every file this run moved (`_moved_by`)
    back from the log against its bytes, in the folder's write order, and the log holds no other;
    a resumed run checks its own moves again. Its switch picks the closure's reader."""

    def step(runner: Any) -> None:
        for workspace in runner.workspaces:
            held = history.workspace_on_log(
                workspace, workspace.name, holder(workspace), runner.root
            )
            folder = _folder(workspace)
            if _moved(held) or not folder:
                continue
            for name, path in folder.items():
                history.keep_workspace_file(
                    held, name, path.read_bytes(), durable=False, rewrite=True
                )
            for name in _kept(held).keys() - folder.keys():  # an earlier step's, gone since
                history.remove_workspace_file(held, name)
            cutover = {"files": len(folder), "store": "panel-closure"}
            history.keep_workspace_file(held, MOVED, json.dumps(cutover).encode(), durable=False)
            runner.log.flush()

    def verify(runner: Any) -> bool:
        for workspace in runner.workspaces:
            held = history.kept_file_workspace(workspace)
            folder = _folder(workspace)
            if held is None or not _moved(held):
                if folder:
                    return False
                continue
            if not _moved_by(runner, held):  # carried, or an earlier run's: current since
                continue
            kept = _kept(held)
            if sorted(kept) != sorted(folder) or not _in_order(folder, kept):
                return False
            for name, path in folder.items():
                if history.read_kept_file(held, name) != path.read_bytes():
                    return False
        return True

    def switch(mode: Literal["old", "new"]) -> None:
        # contract: remove the legacy choice when row U retires the workspace closure folders.
        if mode == "old":
            os.environ[CLOSURE_STORE_ENV] = "legacy"
        else:  # unset: a workspace whose closure the upgrade moved reads the log
            os.environ.pop(CLOSURE_STORE_ENV, None)

    def holds(workspace: Path) -> bool:
        if not _folder(workspace):  # every member, so a linked one is refused here
            return False
        held = history.kept_file_workspace(workspace)
        return held is None or not _moved(held)

    return Family("panel-closure", step, verify, switch, holds)
