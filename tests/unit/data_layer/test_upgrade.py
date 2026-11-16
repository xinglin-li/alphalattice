"""Family admission and switching are decisions; only crash durability needs a process."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.upgrade import Family, UpgradeRunner

TEST_SOURCE = "src/alphalattice/kernel/data_layer/upgrade.py"


@pytest.mark.proves("data_layer.upgrade_stops_before_unverified_readers")
@pytest.mark.parametrize(
    "case", ["complete", "failed-verify", "missing-backup", "unknown-backup", "unrecorded-backup"]
)
def test_upgrade_switches_only_an_equal_family_after_backup_verification(case):
    """Unknown backups do no work; unequal values stop on old, and done steps do not repeat."""
    log = Mock(spec=["read", "record", "flush"])
    log.read.side_effect = lambda sql, args: (
        ([('{"kind":"started","families":["fixture"]}',)] if case == "unrecorded-backup" else [])
        if "FROM event" in sql
        else [(1,)]
    )
    events, steps, modes = [], [], []
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
        (Family("fixture", step, lambda _: old == new, modes.append),),
        backup,
    )
    if case == "complete":
        assert runner.run() is True
        before = list(events)
        assert runner.run() is False
        assert events == before
        assert steps == ["fixture"]
        assert modes == ["old", "new", "new"]
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
        assert modes == (["old", "old"] if case == "failed-verify" else [])
        assert not any(event["kind"] == "family_done" for event in events)
    assert old == [1, 2, 3]
