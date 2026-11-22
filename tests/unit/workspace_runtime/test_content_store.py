from concurrent.futures import Future

import pytest

from alphalattice.control.workspace_runtime import content_store

TEST_SOURCE = "src/alphalattice/control/workspace_runtime/content_store.py"


class _Interrupted(Future):
    """A check still running when the waiting thread is interrupted (Ctrl-C)."""

    def result(self, timeout=None):
        raise KeyboardInterrupt


@pytest.mark.proves("workspace_runtime.an_interrupt_ends_a_source_check_wait")
def test_an_interrupt_while_waiting_on_checks_is_raised_at_once():
    """An interrupt raised while a check still runs ends the wait at once; a finished check's own
    failure is kept, the first in input order."""
    failed: Future = Future()
    failed.set_exception(ValueError("first input"))
    later: Future = Future()
    later.set_exception(ValueError("second input"))
    assert str(content_store._joined([failed, later])) == "first input"
    with pytest.raises(KeyboardInterrupt):
        content_store._joined([failed, _Interrupted()])
