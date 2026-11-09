import pytest

TEST_SOURCE = "src/alphalattice/control/product_host/composition/goals.py"


@pytest.mark.proves("composition.only_a_finished_first_use_owns_its_last_delegated_network_state")
@pytest.mark.parametrize(
    ("kind", "state", "changes", "expected"),
    [
        ("FIRST_USE", "CLOSED", [True], True),
        ("FIRST_USE", "OPEN", [True], False),
        ("FIRST_USE", "CLOSED", [True, False], False),
        ("FIRST_USE", "CLOSED", [False, True], True),
        ("RESEARCH", "CLOSED", [True], False),
    ],
)
def test_network_cleanup_uses_only_a_finished_first_uses_last_delegated_choice(
    tmp_path, kind, state, changes, expected
):
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from alphalattice.control.product_host.composition.goals import GoalApplication

    entries = [
        {
            "operation": "NETWORK_ACCESS_SET",
            "delegation": "first-use-goal:one",
            "network_enabled": enabled,
        }
        for enabled in changes
    ]
    store = SimpleNamespace(attributed=lambda _: entries)
    app = GoalApplication(
        store,
        lambda: datetime(2026, 1, 1, tzinfo=UTC),
        lambda *_: {},
        lambda *_: None,
        lambda *_: None,
        workspace=tmp_path,
    )
    goal = SimpleNamespace(
        declaration=SimpleNamespace(kind=kind),
        state=state,
        goal_id="one",
        intent_registered_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert app.network_left_open(goal) is expected
