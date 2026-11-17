"""The reading views' reader: a process that does not answer is ended and refused by name."""

import io
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import reading

TEST_SOURCE = "src/alphalattice/kernel/data_layer/reading.py"


class _Silent:
    """A reader's process that takes each request and never answers: its output ends only when it
    is killed."""

    def __init__(self) -> None:
        self.stdin, self.stdout = io.BytesIO(), self
        self.code: int | None = None
        self._killed = threading.Event()

    def read(self, _size: int) -> bytes:
        self._killed.wait(10)
        return b""

    def poll(self) -> int | None:
        return self.code

    def kill(self) -> None:
        self.code = -9
        self._killed.set()


@pytest.mark.proves("data_layer.a_stuck_log_reader_is_ended")
def test_a_reader_that_does_not_answer_is_ended_and_refused_by_name(tmp_path, monkeypatch):
    """A reader's process that never answers is refused past READ_SECONDS, and is ended."""
    monkeypatch.setattr(reading, "READ_SECONDS", 0.2)
    silent = _Silent()
    with ThreadPoolExecutor(max_workers=1) as answers, pytest.raises(DataError) as refused:
        reading._ask(silent, ("experience", {"limit": 1}), answers, tmp_path)
    assert refused.value.failure.code == "data_layer.history_unavailable"
    assert silent.poll() == -9
