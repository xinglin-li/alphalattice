"""A recorded SEC transport for Evidence fixtures: two issuers, one filing, no network."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from alphalattice.evidence.alternative_evidence.sources.sec_edgar import SecOfficialResponse

NOW = datetime(2026, 8, 12, 14, 0, tzinfo=UTC)


class SecFixtureTransport:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, url: str, *, maximum_bytes: int) -> SecOfficialResponse:
        self.calls.append(url)
        if url.endswith("company_tickers.json"):
            content = json.dumps(
                {
                    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
                    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft"},
                }
            ).encode()
        elif "/submissions/" in url:
            content = json.dumps(
                {
                    "filings": {
                        "recent": {
                            "accessionNumber": ["0000320193-26-000001"],
                            "filingDate": ["2026-08-01"],
                            "acceptanceDateTime": ["2026-08-01T12:00:00Z"],
                            "form": ["8-K"],
                            "primaryDocument": ["issuer-8k.htm"],
                        }
                    }
                }
            ).encode()
        elif "/companyfacts/" in url:
            content = json.dumps({"facts": {"us-gaap": {}}}).encode()
        else:
            content = b"<html><body>Official filing says revenue declined.</body></html>"
        return SecOfficialResponse(
            status_code=200,
            content_type="application/json" if content.startswith(b"{") else "text/html",
            content=content[:maximum_bytes],
            retrieved_at=NOW,
        )
