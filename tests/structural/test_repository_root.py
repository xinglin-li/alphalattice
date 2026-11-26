"""The repository root is the product's tracked set at HEAD: its required public files, nothing
loose, and no local file of the checkout it was started from."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


# The project's license is Apache-2.0, and its NOTICE joins it.
# CLAUDE.md imports the agent guide AGENTS.md for Claude Code.
ROOT_REQUIRED = {
    "AGENTS.md",
    "CLA.md",
    "CLAUDE.md",
    "GOVERNANCE.md",
    "LAWS.md",
    "LICENSE",
    "README.md",
    "README.zh-CN.md",
    "SECURITY.md",
    "pyproject.toml",
    "uv.lock",
}
# The contribution agreement, governance and private reporting policy are required
# public documents; the build configuration and contribution guide may join them.
ROOT_ALLOWED = ROOT_REQUIRED | {"mkdocs.yml", "NOTICE", "CONTRIBUTING.md"}


def _assert_root_complete(root: Path) -> None:
    """The root as git tracks it at HEAD, never the working tree: a checkout's ignored or untracked
    local files (CLAUDE.local.md, AGENTS.override.md) are never the product's."""
    tracked = set(
        subprocess.run(
            ("git", "-c", "core.quotepath=false", "ls-tree", "-r", "--name-only", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            encoding="utf-8",
        ).stdout.splitlines()
    )
    assert any("/" in path for path in tracked)  # the whole tree, never the root alone
    assert not any(path.startswith("product-desgin/") for path in tracked)
    root_files = {path for path in tracked if "/" not in path and not path.startswith(".")}
    assert not {path for path in tracked if "/" not in path and path.endswith(".py")}
    assert ROOT_REQUIRED <= root_files <= ROOT_ALLOWED, sorted(root_files - ROOT_ALLOWED)


def _root_fixture(root: Path, case: str) -> Path:
    """A committed root with every required file, then the case's local or tracked addition."""
    git = ("git", "-c", "user.name=t", "-c", "user.email=t@localhost")
    subprocess.run(("git", "init", "-q", str(root)), check=True)
    (root / "docs").mkdir()
    for name in (*ROOT_REQUIRED, "docs/README.md"):
        (root / name).write_text("x\n", encoding="utf-8")
    (root / ".gitignore").write_text("CLAUDE.local.md\nAGENTS.override.md\n", encoding="utf-8")
    if case == "tracked unexpected file":
        (root / "NOTES.md").write_text("a stray note\n", encoding="utf-8")
    if case == "tracked hidden python":
        (root / ".hidden.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run((*git, "add", "-A"), cwd=root, check=True)
    subprocess.run((*git, "commit", "-q", "-m", "root"), cwd=root, check=True)
    for local in ("CLAUDE.local.md", "AGENTS.override.md", "scratch.md"):  # ignored, and untracked
        (root / local).write_text("a maintainer's own\n", encoding="utf-8")
    return root


@pytest.mark.proves("architecture.repository_root_is_its_tracked_public_set")
@pytest.mark.parametrize(
    "case",
    ["this checkout", "ignored local files", "tracked unexpected file", "tracked hidden python"],
)
def test_canonical_product_design_and_repository_root_are_complete(tmp_path, case) -> None:
    """The root's tracked files at HEAD are the required and allowed set, whatever lies beside."""
    root = ROOT if case == "this checkout" else _root_fixture(tmp_path / "repo", case)
    if case == "tracked unexpected file":  # a file the product tracks is still refused by name
        with pytest.raises(AssertionError, match=r"NOTES\.md"):
            _assert_root_complete(root)
        return
    if case == "tracked hidden python":
        with pytest.raises(AssertionError):
            _assert_root_complete(root)
        return
    _assert_root_complete(root)
    if case != "this checkout":
        return
    tracked_markdown = subprocess.run(
        ("git", "-c", "core.quotepath=false", "ls-files", "*.md"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        encoding="utf-8",
    ).stdout.splitlines()
    stale_links = tuple(
        path
        for value in tracked_markdown
        if re.search(r"\]\([^)]*product-desgin", (path := ROOT / value).read_text(encoding="utf-8"))
    )
    assert stale_links == ()
