"""Owner timing reaches the actual Task collection and selected Task readers."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.control.task_control.timing import task_timing
from tests.portfolio_strategy_lab.local_web_support import run_node


@pytest.mark.parametrize("case", ["collection-owner-spans", "selected-task-status"])
def test_task_duration_reads_owner_spans_and_exact_selected_status(case):
    """BEHAVIOUR F31/: public Task timing producer and retired-reader controls."""
    started = datetime(2026, 8, 1, 9, tzinfo=UTC)
    timing: dict[str, object] = {}
    for lifecycle in (TaskLifecycle.RUNNING, TaskLifecycle.SUCCEEDED):
        record = SimpleNamespace(
            lifecycle=lifecycle,
            admitted_at=started - timedelta(seconds=5),
            started_at=started,
            updated_at=started + timedelta(seconds=45),
        )
        timing[lifecycle.value] = task_timing(record, (), now=started + timedelta(seconds=125))  # type: ignore[arg-type]
    timing["last_activity_at"] = record.updated_at.isoformat()
    root = Path(__file__).resolve().parents[2]
    source = root / "src/alphalattice/interface/local_application/assets/workbench-source/js/app"
    run_node(
        [
            str(Path(__file__).with_name("workbench_task_duration.cjs")),
            str(source),
            case,
            json.dumps(timing),
        ],
        required=True,
        cwd=root,
        check=True,
        timeout=30,
    )
