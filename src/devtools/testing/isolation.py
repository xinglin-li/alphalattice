"""What a finished test left changed in its interpreter, the import path or the working
directory: named for that test, and put back so no later test inherits it."""

from __future__ import annotations

import os
import sys

State = tuple[list[str], str]


def interpreter_state() -> State:
    """The import path and the working directory, as a test starts."""
    return list(sys.path), os.getcwd()


def _paths(entries: list[str]) -> set[str]:
    return {os.path.normcase(os.path.abspath(entry)) for entry in entries if entry}


def interpreter_leak(nodeid: str, before: State) -> str:
    """The test's leak by name, or "" when it left nothing; the state is put back either way."""
    path, cwd = before
    added = sorted(_paths(sys.path) - _paths(path))
    removed = sorted(_paths(path) - _paths(sys.path))
    moved = os.getcwd() != cwd
    sys.path[:] = path
    os.chdir(cwd)
    changes = [f"sys.path gained {added}"] if added else []
    changes += [f"sys.path lost {removed}"] if removed else []
    changes += [f"the working directory moved from {cwd}"] if moved else []
    if not changes:
        return ""
    return (
        f"tests.interpreter_state_leaked: {nodeid} left {'; '.join(changes)}; change them "
        "through monkeypatch (syspath_prepend, chdir), which puts them back"
    )
