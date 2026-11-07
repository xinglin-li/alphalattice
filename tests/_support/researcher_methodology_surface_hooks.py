"""What methodology tests get without asking: kept apart from the fixtures they ask for by name."""

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session", autouse=True)
def offline_execution_environment() -> Iterator[None]:
    """The operator's offline switch, as the product's research entry points set it.

    The authored-execution policy reads no switch: the workflow holds every run offline
    so a case passes whatever the operator's shell carries. The switch stays on here
    because the data readers these cases reach outside a run keep a process offline by it.
    """

    previous = os.environ.get("ALPHALATTICE_NETWORK_DISABLED")
    os.environ["ALPHALATTICE_NETWORK_DISABLED"] = "1"
    try:
        yield
    finally:
        if previous is None:
            del os.environ["ALPHALATTICE_NETWORK_DISABLED"]
        else:
            os.environ["ALPHALATTICE_NETWORK_DISABLED"] = previous


# They tamper with their seed's market data on purpose, so each gets a private copy.
PRIVATE_DATA = {
    "test_binding_refuses_a_corrupt_copy_a_changed_source_and_a_tampered_pool",
    "test_input_and_result_tamper_refuse_without_losing_historical_readback",
}


def pytest_collection_modifyitems(items):
    for item in items:
        if getattr(item, "originalname", item.name) in PRIVATE_DATA:
            item.add_marker(pytest.mark.private_data)
