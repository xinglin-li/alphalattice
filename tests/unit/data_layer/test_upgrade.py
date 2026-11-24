"""Family admission and verification are decisions; only crash durability needs a process."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer.upgrade import Family, UpgradeRunner

TEST_SOURCE = "src/alphalattice/foundation/data_layer/upgrade.py"


@pytest.mark.proves("data_layer.upgrade_stops_before_unverified_readers")
@pytest.mark.parametrize(
    "case", ["complete", "failed-verify", "missing-backup", "unknown-backup", "unrecorded-backup"]
)
def test_upgrade_completes_only_an_equal_family_after_backup_verification(case):
    """Unknown backups do no work; unequal values stop on old, and done steps do not repeat."""
    log = Mock(spec=["read", "record", "flush"])
    log.read.side_effect = lambda sql, args: (
        ([('{"kind":"started","families":["fixture"]}',)] if case == "unrecorded-backup" else [])
        if "FROM event" in sql
        else [(1,)]
    )
    events, steps = [], []
    log.record.side_effect = lambda kind, subject, attrs, **kw: events.append(attrs)
    old, new = [1, 2, 3], []

    def step(runner):
        steps.append("fixture")
        new[:] = old if case != "failed-verify" else [1, 2, 4]
        runner.map_id("fixture", "old", "new")

    def backup(root, workspaces, recorded):
        if case == "unknown-backup":
            raise OSError("backup unreadable")
        return {"fixture": "verified backup"} if case != "missing-backup" else {}

    runner = UpgradeRunner(
        Path("data"),
        log,
        (Path("workspace"),),
        (Family("fixture", step, lambda _: old == new),),
        backup,
    )
    if case == "complete":
        assert runner.run() is True
        before = list(events)
        assert runner.run() is False
        assert events == before
        assert steps == ["fixture"]
        assert runner.id_map == {("fixture", "old"): "new"}
        assert events[-2:] == [
            {"kind": "family_done", "family": "fixture", "verdict": "ALL EQUAL"},
            {"kind": "completed"},
        ]
    else:
        with pytest.raises(DataError, match=r"fixture|pre-upgrade backup") as stopped:
            runner.run()
        assert stopped.value.failure.code == "upgrade.refused"
        assert steps == (["fixture"] if case == "failed-verify" else [])
        assert not any(event["kind"] == "family_done" for event in events)
    assert old == [1, 2, 3]


@pytest.mark.proves("data_layer.family_declares_old_data_by_holds_alone")
def test_a_family_declares_its_old_data_by_its_holds_predicate_alone(tmp_path):
    """The fourth positional argument is the workspace predicate; with none given a family holds
    no old data, so no workspace is upgraded for it and no deferred predicate is set."""

    def step(_runner) -> None:
        """A family step that does nothing."""

    def verify(_runner) -> bool:
        """A family verification that always passes."""
        return True

    def holds(workspace):
        return workspace.name == "held"

    family = Family("x", step, verify, holds)
    assert family.holds is holds
    assert family.deferred is None
    assert Family("y", step, verify).holds(tmp_path) is False
