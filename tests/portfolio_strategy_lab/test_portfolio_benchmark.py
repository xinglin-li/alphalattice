"""The Portfolio benchmark value surface and the source authority it admits."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphalattice.capabilities.portfolio_inputs import benchmark as benchmark_owner

PLAYPEN_ROOT = Path(__file__).resolve().parents[2]
PLAYPEN_SRC = PLAYPEN_ROOT / "src"


def test_benchmark_value_surface_refuses_a_source_authority_substitution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A current Program cannot reuse SPY values from another raw/action lineage."""

    sessions = (date(2024, 1, 2),)
    parent = SimpleNamespace(revision_sha256="1" * 64)
    reference = SimpleNamespace(manifest=parent, listing_id="SPY")
    rows = [
        {"session_date": date(2024, 1, 2), "open_split_adjusted": 100.0, "cash_dividend": 0.0},
        {"session_date": date(2024, 1, 3), "open_split_adjusted": 101.0, "cash_dividend": 0.0},
        {"session_date": date(2024, 1, 4), "open_split_adjusted": 102.0, "cash_dividend": 0.0},
    ]
    market = SimpleNamespace(
        current_quality_filtered_research_manifest=lambda **kwargs: parent,
    )
    feature = SimpleNamespace(
        projected_feature_frame=lambda *args, **kwargs: (rows, "2" * 64, "3" * 64),
    )
    monkeypatch.setattr(benchmark_owner, "MarketDataRepository", lambda workspace: market)
    monkeypatch.setattr(
        benchmark_owner, "FeatureStateRepository", lambda workspace, market_data: feature
    )
    monkeypatch.setattr(benchmark_owner.MarketReference, "spy", lambda manifest: reference)
    expected = benchmark_owner.PortfolioBenchmarkSourceAuthority.create(
        formation_sessions=sessions,
        universe_manifest_revision="1" * 64,
        raw_input_hash="2" * 64,
        action_set_hash="4" * 64,
    )
    with pytest.raises(
        benchmark_owner.PortfolioBenchmarkBoundaryError,
        match="source_authority_not_admitted",
    ):
        benchmark_owner.build_portfolio_benchmark_surface(
            workspace=tmp_path,
            formation_sessions=sessions,
            expected_source_authority=expected,
        )
