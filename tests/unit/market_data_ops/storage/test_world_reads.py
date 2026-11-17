"""The market reads setting: world by default, the workspace as the rollback, and nothing else."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from alphalattice.foundation.market_data_ops.storage import world_reads
from alphalattice.kernel.data.errors import DataError

TEST_SOURCE = "src/alphalattice/foundation/market_data_ops/storage/world_reads.py"


@pytest.mark.proves("market_data_ops.market_reads_setting_is_world_or_workspace")
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(None, "world", id="unset"),
        pytest.param("world", "world", id="world"),
        pytest.param("workspace", "workspace", id="workspace"),
        pytest.param("lake", None, id="anything else"),
    ],
)
def test_the_setting_is_world_unless_it_names_the_workspace(monkeypatch, value, expected):
    """Unset, the reads come from world; set, it names world or the workspace, and any other value
    is refused by name before a read."""
    if value is None:
        monkeypatch.delenv(world_reads.SETTING, raising=False)
    else:
        monkeypatch.setenv(world_reads.SETTING, value)
    if expected is None:
        with pytest.raises(DataError) as refused:
            world_reads.chosen()
        assert refused.value.failure.code == "market_data.market_reads_unknown"
    else:
        assert world_reads.chosen() == expected


@pytest.mark.proves("market_data_ops.world_relations_read_as_of_their_cutoff")
def test_each_workspace_table_has_a_world_relation_read_as_of_the_cutoff():
    """Every workspace table a value read names has a world relation under the same name, and
    each one reads world through its as-of macros at the cutoff it is given, never later."""
    assert (
        world_reads.cutoff(datetime.fromisoformat("2023-11-14T22:13:20.000001"))
        == 1_700_000_000_000_001
    )
    assert (
        world_reads.cutoff(datetime.fromisoformat("2023-11-14T17:13:20-05:00"))
        == 1_700_000_000_000_000
    )
    named = world_reads.relations(1_700_000_000_000_000, workspace_id="w1")
    held = world_reads.relations(1_700_000_000_000_000, 1_600_000_000_000_000, workspace_id="w1")
    assert set(named) == set(world_reads.WORKSPACE)
    for name, relation in named.items():
        if name in {"bar_revisions", "action_revisions"}:
            assert "occurred_us <= 1700000000000000" in relation
            assert "recorded_us <= 1600000000000000" in held[name]
            assert "workspace_id = 'w1'" in relation
            assert "workspace_id = NULL" in world_reads.relations()[name]
            continue
        cutoffs = {part.split(")")[0] for part in relation.split("_asof(")[1:]}
        assert cutoffs == {"1700000000000000"}
        assert "known_us <= 1600000000000000" in held[name]
        if name in {"bars", "adjusted", "actions", "action_evidence"}:
            assert "workspace_id = 'w1'" in relation
    assert "c.known_asof <= make_timestamptz(1700000000000000)" in named["action_evidence"]
    assert "c.visible_to > make_timestamptz(1700000000000000)" in named["action_evidence"]


BINDING = "0123456789abcdef" * 4


@pytest.mark.proves("market_data_ops.a_research_copy_is_named_by_its_place")
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        pytest.param(f"w/research-inputs/{BINDING}/source", BINDING, id="a copy"),
        pytest.param("w", None, id="the workspace's own database"),
        pytest.param(f"w/research-inputs/{BINDING}/other", None, id="not its source folder"),
        pytest.param("w/research-inputs/databases/source", None, id="the pool, not a binding"),
        pytest.param(f"w/research-inputs/{BINDING.upper()}/source", None, id="not a binding hash"),
    ],
)
def test_a_research_copy_is_named_by_its_binding_folder(path, expected):
    """Only a database in a binding's source folder is a research-input copy, which reads world
    through its selection; every other database reads as before."""
    assert world_reads.binding(Path(path) / "market-data.duckdb") == expected


@pytest.mark.proves("market_data_ops.world_serves_only_an_admission_that_took_every_row")
def test_an_admission_that_dropped_an_equal_time_row_does_not_hold_its_marks():
    """An admission's marks count only when it dropped no workspace row as written at the same time
    as world's version of its key; an older row's drop leaves world the newer value."""
    admissions = [('{"bars": [2, 7]}', 1), ('{"bars": [3, 8]}', None), (None, None)]
    assert world_reads.taken(admissions) == [{"bars": [3, 8]}]
