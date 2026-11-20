"""Alpha's historical seed boundaries."""

from alphalattice.kernel.data_layer.root import seed_rows


def assert_seed_history(_named: dict[str, set[str]]) -> None:
    """The eleven Alpha development rows arrived at count 119."""
    assert [len(seed_rows(historical_count=n)) for n in (118, 119)] == [108, 119]
