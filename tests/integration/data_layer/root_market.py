"""Market's historical seed boundary."""

from alphalattice.foundation.data_layer.root import seed_rows


def assert_seed_history(named: dict[str, set[str]]) -> None:
    """The Market table joins the seed at count 120."""
    assert len(seed_rows(historical_count=120)) == 120
    assert "market" in named["world_table"]
