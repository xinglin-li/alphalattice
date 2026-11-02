"""Legacy and non-unit layers opt into the run cache; isolation stays at the root."""

import pytest


@pytest.fixture(autouse=True)
def project_discovery(golden_cache):
    """Keep cache setup for opted-in layers after the universal root isolation."""
    yield
