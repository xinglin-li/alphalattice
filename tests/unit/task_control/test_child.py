from types import SimpleNamespace

import pytest

from alphalattice.control.task_control import child

TEST_SOURCE = "src/alphalattice/control/task_control/child.py"


class _Ended(Exception):
    """Stands in for the worker's exit, so the test process itself goes on."""


@pytest.mark.proves("task_control.a_worker_ends_with_its_parent")
@pytest.mark.parametrize(
    "parent",
    [
        pytest.param(SimpleNamespace(sentinel=7), id="blocks-on-the-parents-sentinel"),
        pytest.param(None, id="no-parent-ends-at-once"),
    ],
)
def test_a_kept_worker_ends_when_its_parent_ends(monkeypatch, parent):
    """A kept worker blocks on its parent's sentinel, with no loop or timer, and ends with
    code 3 when it returns; a worker with no parent ends at once."""
    waited: list[list[int]] = []

    def ended(code: int) -> None:
        raise _Ended(code)

    monkeypatch.setattr(child.multiprocessing, "parent_process", lambda: parent)
    monkeypatch.setattr(child.multiprocessing.connection, "wait", waited.append)
    monkeypatch.setattr(child, "os", SimpleNamespace(_exit=ended))  # its exit, recorded
    with pytest.raises(_Ended, match="3"):
        child._watch_parent()
    assert waited == ([] if parent is None else [[7]])
