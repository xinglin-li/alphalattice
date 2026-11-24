import shutil
import subprocess
from pathlib import Path

import pytest

from alphalattice.capabilities.domain_contracts import spawn
from devtools.testing.collection import CHANGED
from tests._support.project_discovery import project_discovery as project_discovery
from tests.structural import source_shape_samples as samples


def _history(root):
    run = spawn.run  # fixture Git stays real when a test replaces its subject's runner
    config = (
        "user.name=Synthetic Author",
        "user.email=synthetic@example.test",
        "core.hooksPath=",
        "core.autocrlf=false",
        "maintenance.auto=false",
        "gc.auto=0",
    )

    def git(*args):
        options = [part for setting in config for part in ("-c", setting)]
        return run(
            ["git", *options, "-C", str(root), *args],
            text=True,
            stdout=subprocess.PIPE,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip()

    def commit(message, *, advance=True):
        git("add", "--all")
        if not advance:
            return git("commit-tree", git("write-tree"), "-p", "HEAD", "-m", message)
        git("commit", "-q", "-m", message)

    return git, commit


@pytest.fixture(scope="session")
def _history_start(tmp_path_factory):
    """The first commit, built once a session: a test copies it, never replays its Git."""
    root = tmp_path_factory.mktemp("synthetic-history")
    git, commit = _history(root)
    git("init", "-q")
    (root / "README.md").write_text("Synthetic tree\n", encoding="utf-8", newline="\n")
    commit("Start synthetic history")
    return root


@pytest.fixture
def synthetic_history(tmp_path, _history_start):
    shutil.copytree(
        _history_start, tmp_path, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.lock")
    )
    return _history(tmp_path)


@pytest.fixture(autouse=True)
def per_file_scope(request, monkeypatch):
    """A scoped per_file node reads repository Python only through scanned(), only changed."""
    changed = request.config.stash.get(CHANGED, None)
    if changed is None or not request.node.get_closest_marker("per_file"):
        yield
        return
    opened, root = Path.open, samples.ROOT.resolve()

    def guarded(self, *args, **kwargs):
        path = Path(self).resolve()
        if path.suffix == ".py" and path.is_relative_to(root):
            name = path.relative_to(root).as_posix()
            assert name in changed, f"a per_file node read {name} outside its scope"
        return opened(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    monkeypatch.setattr(samples, "SCOPE", changed)
    yield
