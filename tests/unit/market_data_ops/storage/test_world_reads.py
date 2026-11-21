"""World's read relations and research-copy binding."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from alphalattice.foundation.market_data_ops.storage import world_reads

TEST_SOURCE = "src/alphalattice/foundation/market_data_ops/storage/world_reads.py"


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
        alias = f"provider_symbol_asof({world_reads.LATEST}) WHERE provider_id"
        cutoffs = {part.split(")")[0] for part in relation.replace(alias, "").split("_asof(")[1:]}
        assert cutoffs == (set() if name == "action_evidence" else {"1700000000000000"})
        assert "known_us <= 1600000000000000" in held[name]
        if name in {"bars", "adjusted", "actions", "action_evidence"}:
            assert "workspace_id = 'w1'" in relation
    assert "c.known_asof <= make_timestamptz(1700000000000000)" in named["action_evidence"]
    assert "c.visible_to > make_timestamptz(1700000000000000)" in named["action_evidence"]


@pytest.mark.proves("market_data_ops.world_relations_read_as_of_their_cutoff")
def test_a_relation_names_its_listings_by_their_0_2_7_alias_whatever_the_cutoff():
    """A relation reads its rows at the cutoff and names their listings by the 0.2.7 alias at
    LATEST, an identity of a key world already holds."""
    relation = world_reads.relations(1_700_000_000_000_000, workspace_id="w1")["bars"]
    assert "raw_bar_asof(1700000000000000)" in relation
    assert f"provider_symbol_asof({world_reads.LATEST}) WHERE provider_id" in relation


@pytest.mark.proves("market_data_ops.world_relations_read_as_of_their_cutoff")
def test_a_held_relation_replaces_a_version_only_by_an_admission_held_by_then():
    """Read as world held it at a selection, a version is replaced only by an admission committed
    by then: a row is known at its writer's observed time, which a later publish may precede."""
    held = world_reads.relations(1_700_000_000_000_000, 1_600_000_000_000_000, workspace_id="w1")
    named = world_reads.relations(1_700_000_000_000_000, workspace_id="w1")
    replaced = (
        "by_admission_id IN (SELECT admission_id FROM admission WHERE known_us <= 1600000000000000)"
    )
    for name in ("bars", "adjusted", "actions", "action_evidence", "mappings"):
        assert replaced in held[name]
        assert "by_admission_id" not in named[name]
    for macro in ("raw_bar_asof(", "corporate_action_asof(", "corporate_action_versions"):
        assert not any(macro in held[name] for name in ("bars", "actions", "action_evidence"))


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


@pytest.mark.proves("market_data_ops.binding_names_a_copy_by_its_research_input_place")
def test_only_a_database_in_a_research_input_source_is_a_binding_copy(tmp_path):
    """A copy's database sits at research-inputs/<64-hex binding>/source/market-data.duckdb; any
    other place, or a name that is not a binding's hash, is no copy and reads its own tables."""
    key = "a" * 64
    copy = tmp_path / "ws" / "research-inputs" / key / "source" / "market-data.duckdb"
    assert world_reads.binding(copy) == key
    assert world_reads.binding(tmp_path / "ws" / "other" / key / "source" / "x.duckdb") is None
    short = tmp_path / "ws" / "research-inputs" / "not-a-hash" / "source" / "market-data.duckdb"
    assert world_reads.binding(short) is None
