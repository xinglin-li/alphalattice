"""The Host's history for tests that build a workspace's activity without a Host."""

from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest

from alphalattice.kernel.data_layer import history


@pytest.fixture
def recording_history(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Open this test process's history as a Host opens it, with a workspace of the test's own:
    the process's Hosts share one data root, and each reads only its own workspace. A history a
    Host of this process holds stays its own; the test's workspace is only recorded in it."""
    workspace_id = f"test-{uuid4().hex[:12]}"
    folder, opened_here = tmp_path_factory.mktemp(workspace_id), not history.recording()
    if opened_here:
        history.open_history(workspace_id, "test", folder)
    else:
        history.workspace_on_log(folder, "test", workspace_id)
    try:
        yield workspace_id
    finally:
        if opened_here:
            history.release_history()
