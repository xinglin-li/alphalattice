"""The wire replay's scenarios: per symbol a seeded walk on the exchange calendar, the days a test
runs, named cases of provider events, and the index lists. Each event names the day the
provider first shows it (`known`) or the day it strikes (`on`); a case's injection is removed
by deleting its rows."""

from __future__ import annotations

from typing import Any

SCENARIOS: dict[str, dict[str, Any]] = {
    "realistic": {
        "calendar": "XNAS",
        "start": "2025-10-01",
        "symbols": {"AAPL": 180.0, "MSFT": 410.0},
        "days": ["2026-07-29", "2026-07-30", "2026-07-31"],
        "cases": {
            "dividend-stored": [
                {
                    "kind": "dividend",
                    "symbol": "AAPL",
                    "ex": "2026-05-12",
                    "amount": 0.26,
                    "known": "2026-05-01",
                },
                {
                    "kind": "redirect",
                    "symbol": "AAPL",
                    "on": "2026-07-30",
                    "count": 1,
                    "to": "query1.finance.yahoo.com",
                },
            ],
            "dividend-new": [
                {
                    "kind": "dividend",
                    "symbol": "AAPL",
                    "ex": "2026-07-30",
                    "amount": 0.26,
                    "known": "2026-07-30",
                }
            ],
            "dividend-late": [
                {
                    "kind": "dividend",
                    "symbol": "AAPL",
                    "ex": "2026-07-28",
                    "amount": 0.26,
                    "known": "2026-07-30",
                }
            ],
            "dividend-corrected": [
                {
                    "kind": "dividend",
                    "symbol": "AAPL",
                    "ex": "2026-07-21",
                    "amount": 0.26,
                    "known": "2026-07-10",
                    "restated": {"known": "2026-07-30", "amount": 0.31},
                }
            ],
            "split-forward": [
                {
                    "kind": "split",
                    "symbol": "AAPL",
                    "ex": "2026-07-30",
                    "ratio": 4.0,
                    "known": "2026-07-30",
                }
            ],
            "split-reverse": [
                {
                    "kind": "split",
                    "symbol": "AAPL",
                    "ex": "2026-07-30",
                    "ratio": 0.5,
                    "known": "2026-07-30",
                }
            ],
            "split-late": [
                {
                    "kind": "split",
                    "symbol": "AAPL",
                    "ex": "2026-07-28",
                    "ratio": 2.0,
                    "known": "2026-07-30",
                }
            ],
            "actions-in-one-history": [
                {
                    "kind": "dividend",
                    "symbol": "AAPL",
                    "ex": "2026-02-10",
                    "amount": 0.25,
                    "known": "2026-02-01",
                },
                {
                    "kind": "split",
                    "symbol": "AAPL",
                    "ex": "2026-03-16",
                    "ratio": 3.0,
                    "known": "2026-03-01",
                },
                {
                    "kind": "dividend",
                    "symbol": "AAPL",
                    "ex": "2026-05-12",
                    "amount": 0.09,
                    "known": "2026-05-01",
                },
            ],
            "rerounding": [
                {"kind": "reround", "symbol": "AAPL", "scale": 4e-07, "known": "2026-07-30"}
            ],
            "outage-rate-limited": [
                {
                    "kind": "outage",
                    "symbol": "AAPL",
                    "on": "2026-07-30",
                    "status": 429,
                    "retry_after": 1,
                    "count": 2,
                }
            ],
            "outage-server-error": [
                {"kind": "outage", "symbol": "AAPL", "on": "2026-07-30", "status": 503, "count": 1}
            ],
            "outage-timeout": [
                {
                    "kind": "outage",
                    "symbol": "AAPL",
                    "on": "2026-07-30",
                    "status": "timeout",
                    "count": 1,
                }
            ],
            "membership-change": [
                {"kind": "membership", "index": "SP500", "on": "2026-07-30", "leave": ["MSFT"]},
                {"kind": "membership", "index": "NASDAQ100", "on": "2026-07-30", "join": ["NVDA"]},
                {
                    "kind": "outage",
                    "index": "SP500",
                    "on": "2026-07-30",
                    "status": 429,
                    "retry_after": 7,
                    "count": 1,
                },
            ],
        },
        "membership": {
            "SP500": [["AAPL", "Apple Inc."], ["MSFT", "Microsoft"]],
            "NASDAQ100": [["AAPL", "Apple Inc."]],
            "DJIA": [["AAPL", "Apple Inc."]],
        },
    },
    "restatement": {
        "calendar": "XNAS",
        "start": "2025-10-01",
        "symbols": {"AAPL": 100.0},
        "days": ["2026-01-30", "2026-08-03", "2026-08-04"],
        "cases": {
            "catchup": [
                {
                    "kind": "dividend",
                    "symbol": "AAPL",
                    "ex": "2026-01-15",
                    "amount": 0.5,
                    "known": "2025-10-01",
                    "restated": {"known": "2026-08-04", "amount": 1.0},
                }
            ],
            "daily": [
                {
                    "kind": "dividend",
                    "symbol": "AAPL",
                    "ex": "2026-07-27",
                    "amount": 0.5,
                    "known": "2025-10-01",
                    "restated": {"known": "2026-08-04", "amount": 1.0},
                }
            ],
        },
        "membership": {"SP500": [], "NASDAQ100": [], "DJIA": []},
    },
}
