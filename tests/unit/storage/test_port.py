"""The storage port opens nothing itself: an open in a process with no driver is refused."""

import pytest

from alphalattice.kernel import storage

TEST_SOURCE = "src/alphalattice/kernel/storage/port.py"


@pytest.mark.proves("storage.an_open_without_a_driver_is_refused_by_name")
def test_an_open_without_a_driver_is_refused_by_name() -> None:
    """With no driver installed, an open is refused by name with its recovery, and counted as no
    instance; the kernel opens no engine of its own."""
    previous = storage.install_driver(None)
    before = storage.opens()
    try:
        with pytest.raises(storage.DriverMissing, match=r"^storage\.driver_missing: .*DRIVER"):
            storage.connect(storage.MEMORY)
    finally:
        storage.install_driver(previous)
    assert storage.opens() == before
