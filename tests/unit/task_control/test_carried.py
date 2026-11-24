"""What an adapter carries from one stage of a Task's execution to its later stages."""

from uuid import UUID, uuid4

import pytest

from alphalattice.control.task_control.carried import carried_by
from alphalattice.control.task_control.contracts import TaskExecution

TEST_SOURCE = "src/alphalattice/control/task_control/carried.py"


def _execution(task: UUID) -> TaskExecution:
    return TaskExecution.model_construct(task_id=task, execution_id=uuid4())


@pytest.mark.proves("task_control.a_recovery_execution_carries_nothing_from_the_interrupted_one")
def test_a_recovery_execution_carries_nothing_from_the_interrupted_one() -> None:
    """An execution's later stages take what it carried; a new execution of the same Task, as a
    recovery starts after an interruption, takes nothing of the previous one's."""
    task = uuid4()
    interrupted, recovery = _execution(task), _execution(task)
    previous: dict[UUID, tuple[UUID, list[str]]] = {}
    carried_by(previous, interrupted, list).append("verified")

    carried = carried_by(previous, interrupted, list)
    recovered = carried_by(previous, recovery, list)
    other_task = carried_by(previous, _execution(uuid4()), list)

    assert (carried, recovered, other_task) == (["verified"], [], [])
