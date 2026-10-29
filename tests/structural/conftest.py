import subprocess
from pathlib import Path

import pytest

from devtools.architecture.registries import CHANGED
from tests.structural import source_shape_samples as samples


@pytest.fixture
def synthetic_history(tmp_path):
    config = (
        "user.name=Synthetic Author",
        "user.email=synthetic@example.test",
        "core.hooksPath=",
        "core.autocrlf=false",
    )

    def git(*args):
        options = [part for setting in config for part in ("-c", setting)]
        return subprocess.check_output(
            ["git", *options, "-C", str(tmp_path), *args], text=True
        ).strip()

    def commit(message, *, advance=True):
        git("add", "--all")
        if not advance:
            return git("commit-tree", git("write-tree"), "-p", "HEAD", "-m", message)
        git("commit", "-q", "-m", message)

    git("init", "-q")
    (tmp_path / "README.md").write_text("Synthetic tree\n", encoding="utf-8", newline="\n")
    commit("Start synthetic history")
    return git, commit


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
