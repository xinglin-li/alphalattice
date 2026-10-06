"""Stored paths stay inside the workspace: every link-mediated escape is refused.

`resolve_confined` used to re-resolve every component's real path after
proving it not link-like; a fifty-issuer evidence repeat paid six seconds for
those lookups on Windows. The proof now rests on the `lstat` of each existing
component and on the root's single resolution, which is what the refusals
below hold to: a junction or symlink anywhere on the chain, a link-like or
hard-linked target, a link-like root, an immutable target that already
exists. Nothing here weakens: a dangling link component is refused too,
which the old existence check let through.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from alphalattice.control.workspace_runtime.paths import resolve_confined, resolve_workspace_root
from alphalattice.kernel.shared_kernel.domain.errors import WorkspaceSecurityError


def _link_directory(link: Path, target: Path) -> None:
    """A directory link this host can make without privilege: a junction on
    Windows (a reparse point), a symlink elsewhere."""

    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
        return
    link.symlink_to(target, target_is_directory=True)


def _link_file(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("this host does not permit an unprivileged file symlink")


def test_a_plain_stored_path_resolves_under_the_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    (root / "knowledge" / "documents").mkdir(parents=True)
    resolved = resolve_confined(root, "knowledge/documents/a.json")
    assert resolved == resolve_workspace_root(root) / "knowledge" / "documents" / "a.json"
    # A path whose parents do not exist yet resolves the same way.
    assert resolve_confined(root, "knowledge/new/b.json").parent.name == "new"


def test_a_link_like_component_on_the_chain_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    (root / "knowledge").mkdir(parents=True)
    outside.mkdir()
    (outside / "secret.json").write_text("{}", encoding="utf-8")
    _link_directory(root / "knowledge" / "documents", outside)
    with pytest.raises(WorkspaceSecurityError, match="link-like"):
        resolve_confined(root, "knowledge/documents/secret.json")
    with pytest.raises(WorkspaceSecurityError, match="link-like"):
        resolve_confined(root, "knowledge/documents/missing/deeper.json")


def test_a_dangling_link_component_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    (root / "knowledge").mkdir(parents=True)
    gone = tmp_path / "gone"
    gone.mkdir()
    _link_directory(root / "knowledge" / "documents", gone)
    gone.rmdir()
    with pytest.raises(WorkspaceSecurityError, match="link-like"):
        resolve_confined(root, "knowledge/documents/a.json")


def test_a_link_like_target_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    (root / "knowledge").mkdir(parents=True)
    outside = tmp_path / "secret.json"
    outside.write_text("{}", encoding="utf-8")
    _link_file(root / "knowledge" / "a.json", outside)
    with pytest.raises(WorkspaceSecurityError, match="target is link-like"):
        resolve_confined(root, "knowledge/a.json")


def test_a_hard_linked_target_is_refused_unless_allowed(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    (root / "knowledge").mkdir(parents=True)
    original = root / "knowledge" / "a.json"
    original.write_text("{}", encoding="utf-8")
    os.link(original, tmp_path / "alias.json")
    with pytest.raises(WorkspaceSecurityError, match="multiple hard links"):
        resolve_confined(root, "knowledge/a.json")
    assert resolve_confined(root, "knowledge/a.json", reject_existing_hardlink=False) == (
        resolve_workspace_root(root) / "knowledge" / "a.json"
    )


def test_an_immutable_target_that_exists_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    (root / "knowledge").mkdir(parents=True)
    (root / "knowledge" / "a.json").write_text("{}", encoding="utf-8")
    with pytest.raises(WorkspaceSecurityError, match="already exists"):
        resolve_confined(root, "knowledge/a.json", allow_existing=False)


def test_a_root_swapped_for_a_link_is_refused_after_it_was_resolved(tmp_path: Path) -> None:
    """The root's real path is reused only while the root is the same directory
    object; a link put in its place is a new identity and is refused."""

    root = tmp_path / "workspace"
    root.mkdir()
    assert resolve_workspace_root(root) == root.resolve()
    root.rmdir()
    real = tmp_path / "elsewhere"
    real.mkdir()
    _link_directory(root, real)
    with pytest.raises(WorkspaceSecurityError, match="workspace root must not be a link"):
        resolve_workspace_root(root)


def test_a_link_like_root_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    _link_directory(tmp_path / "alias", real)
    with pytest.raises(WorkspaceSecurityError, match="workspace root must not be a link"):
        resolve_confined(tmp_path / "alias", "knowledge/a.json")


@pytest.mark.parametrize(
    "stored", ["../a.json", "knowledge/../a.json", "/knowledge/a.json", "knowledge\\a.json"]
)
def test_an_unsafe_stored_path_never_reaches_the_filesystem(tmp_path: Path, stored: str) -> None:
    with pytest.raises(WorkspaceSecurityError):
        resolve_confined(tmp_path, stored)
