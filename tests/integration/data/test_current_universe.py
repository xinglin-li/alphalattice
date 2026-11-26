"""Focused offline checks for the current-universe bootstrap helper."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

import pytest

from alphalattice.foundation.market_data_ops.sources import universe as current_universe
from alphalattice.kernel.data.enums import UniverseIndex
from alphalattice.kernel.data.errors import DataProviderError
from alphalattice.kernel.data.universe import (
    FROZEN_UNIVERSE_SOURCES,
    FetchedUniverseSource,
    parse_membership,
)


def _source(index: UniverseIndex, members: tuple[tuple[str, str], ...]) -> FetchedUniverseSource:
    config = FROZEN_UNIVERSE_SOURCES[index]
    header = "".join(f"<th>{item}</th>" for item in config.header_prefix)
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{value}</td>"
            for value in (
                (symbol, company)
                if config.symbol_column == 0
                else (company, "NYSE", symbol, "Technology", "", "", "")
            )
        )
        + "</tr>"
        for symbol, company in members
    )
    content = f"<table><tr>{header}</tr>{rows}</table>".encode()
    return FetchedUniverseSource(
        config=config,
        content=content,
        retrieved_at=datetime(2026, 8, 2, tzinfo=UTC),
        response_hash=sha256(content).hexdigest(),
    )


def test_current_universe_helper_reuses_all_frozen_sources(monkeypatch) -> None:
    fixtures = {
        UniverseIndex.SP500: _source(UniverseIndex.SP500, (("BRK/B", "Berkshire Hathaway"),)),
        UniverseIndex.NASDAQ100: _source(UniverseIndex.NASDAQ100, (("MSFT", "Microsoft"),)),
        UniverseIndex.DJIA: _source(UniverseIndex.DJIA, (("MSFT", "Microsoft Corporation"),)),
    }
    requested: list[UniverseIndex] = []

    def fetch(config):
        requested.append(config.index)
        return fixtures[config.index]

    monkeypatch.setattr(current_universe, "fetch_universe_source", fetch)

    result = current_universe.discover_current_universe_candidates(
        observed_at=datetime(2026, 8, 2, tzinfo=UTC)
    )

    assert requested == list(FROZEN_UNIVERSE_SOURCES)
    assert result.candidate_symbols == ("BRK.B", "MSFT")
    assert result.source_manifest.candidates[1].source_memberships == (
        current_universe.CandidateMembershipEvidence(
            index="DJIA", company_name="Microsoft Corporation"
        ),
        current_universe.CandidateMembershipEvidence(index="NASDAQ100", company_name="Microsoft"),
    )
    assert result.standard.minimum_history_calendar_years == 10
    assert result.standard.maximum_missing_ratio == 0.02
    assert result.standard.maximum_consecutive_missing_sessions == 20
    assert result.source_manifest.data_validity_class == "CURRENT_UNIVERSE_RESEARCH_ONLY"


def test_moved_constituent_lists_bind_consumed_columns_not_incidental_decoration():
    nasdaq = FROZEN_UNIVERSE_SOURCES[UniverseIndex.NASDAQ100]
    djia = FROZEN_UNIVERSE_SOURCES[UniverseIndex.DJIA]
    assert nasdaq.uri.endswith("/List_of_NASDAQ-100_companies")
    assert djia.uri.endswith("/List_of_Dow_Jones_Industrial_Average_companies")
    assert parse_membership(
        nasdaq,
        b"<table><tr><th>Ticker</th><th>Company</th><th>ICB Industry[1]</th></tr>"
        b"<tr><td>MSFT</td><td>Microsoft</td><td>Technology</td></tr></table>",
    ) == (("MSFT", "Microsoft"),)
    assert parse_membership(
        djia,
        b"<table><tr><th>Company</th><th>Exchange</th><th>Symbol</th></tr>"
        b"<tr><td>Microsoft</td><td>NASDAQ</td><td>MSFT</td></tr></table>",
    ) == (("MSFT", "Microsoft"),)
    with pytest.raises(DataProviderError):
        parse_membership(
            djia,
            b"<table><tr><th>Company</th><th>Exchange</th><th>Symbol</th></tr>"
            b"<tr><td>Microsoft</td><td>NASDAQ</td></tr></table>",
        )
