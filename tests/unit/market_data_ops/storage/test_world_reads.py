"""The market reads setting: world by default, the workspace as the rollback, and nothing else."""

from __future__ import annotations

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
    named = world_reads.relations(1_700_000_000_000_000)
    assert set(named) == set(world_reads.WORKSPACE)
    for relation in named.values():
        cutoffs = {part.split(")")[0] for part in relation.split("_asof(")[1:]}
        assert cutoffs == {"1700000000000000"}
