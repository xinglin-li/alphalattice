from collections.abc import Iterator
from pathlib import Path

import pytest

from tests._support.alternative_evidence_desk_fixtures import built_http_book as built_http_book
from tests._support.alternative_evidence_desk_fixtures import http_book as http_book
from tests._support.portfolio_strategy_lab_fixtures import completed_host as completed_host
from tests._support.portfolio_strategy_lab_fixtures import fake_host as fake_host
from tests._support.portfolio_strategy_lab_fixtures import live as live
from tests._support.portfolio_strategy_lab_fixtures import read_only_live as read_only_live


@pytest.fixture
def portfolio_result_history(tmp_path: Path) -> Iterator[None]:
    """Open the machine catalog for standalone Portfolio publication fixtures."""
    from alphalattice.kernel.data_layer import history  # lazy: only fixtures that publish need it

    history.open_history("portfolio-test-history", "portfolio", tmp_path / "portfolio-history")
    try:
        yield
    finally:
        history.release_history()
