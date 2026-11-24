"""The Host's history for tests that build a workspace's activity without a Host."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest

from alphalattice.foundation.data_layer import history


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


NATIVE_FIRST_USE: list[tuple[str, float, int | None, str | None, str | None]] = [
    # (event, second, duration_ms, decision source, agent id): one Claude Code conversation's
    # records in the order a captured session exported them (names, subagent id and keys as
    # captured; times, durations and decision sources planted; no other field kept), between
    # its calls ending at second 2 and starting at second 7.
    ("claude_code.managed_settings_resolved", 2.1, None, None, None),
    ("claude_code.user_prompt", 2.4, None, None, None),
    ("claude_code.tool_decision", 2.9, None, "user_temporary", None),
    ("claude_code.tool_result", 3.2, 400, None, None),
    ("claude_code.api_request", 3.9, 600, None, None),
    ("claude_code.tool_decision", 4.0, None, "config", None),
    ("claude_code.api_request", 4.6, 500, None, None),
    ("claude_code.tool_result", 4.8, 150, None, None),
    ("claude_code.tool_decision", 4.9, None, "config", None),
    ("claude_code.api_request", 5.4, 400, None, None),
    ("claude_code.api_request", 5.9, 400, None, "a1"),
    ("claude_code.assistant_response", 5.95, None, None, "a1"),
    ("claude_code.subagent_completed", 6.0, None, None, "a1"),
    ("claude_code.tool_result", 6.3, 250, None, None),
    ("claude_code.api_request", 6.8, 400, None, None),
    ("claude_code.api_request", 8.0, 900, None, None),  # inside the next call: not the gap's
]
"""Split as planted: the person 0.9 s, tools 0.7 s, the model 2.3 s, unattributed 1.1 s of 5 s."""


def native_first_use(conversation: str, lo_us: int, hi_us: int) -> list[dict[str, Any]]:
    """The spike-shaped records as the breakdown reads them, their planted gap (seconds 2 to 7)
    laid over the gap from `lo_us` to `hi_us`, times and durations scaled alike."""
    scale = (hi_us - lo_us) / 5_000_000
    return [
        {
            "conversation": conversation,
            "name": name,
            "at_us": lo_us + round((second - 2) * 1_000_000 * scale),
            "duration_ms": None if duration is None else duration * scale,
            "source": source,
            "agent_id": agent,
        }
        for name, second, duration, source, agent in NATIVE_FIRST_USE
    ]
