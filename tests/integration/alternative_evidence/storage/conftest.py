from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _review_store() -> Iterator[None]:
    """World storage tests open a Host history of their own, so no review store wraps them."""
    yield
