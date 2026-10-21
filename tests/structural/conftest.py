import subprocess

import pytest


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
