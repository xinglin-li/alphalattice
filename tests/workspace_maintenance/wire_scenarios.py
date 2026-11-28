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
            "chart-empty-200": [{"kind": "chart-empty", "symbol": "AAPL", "on": "2026-07-30"}],
            "chart-nan-close": [{"kind": "chart-null-close", "symbol": "AAPL", "on": "2026-07-30"}],
            "chart-zero-close": [
                {"kind": "chart-zero-close", "symbol": "AAPL", "on": "2026-07-30"}
            ],
            "chart-unfinished-next-session": [
                {"kind": "chart-partial", "symbol": "AAPL", "on": "2026-07-31"}
            ],
            "cash-acquisition-delists": [
                {
                    "kind": "cash-acquisition",
                    "symbol": "AAPL",
                    "ex": "2026-07-30",
                    "on": "2026-07-31",
                    "amount": 205.0,
                }
            ],
            "ticker-change": [
                {"kind": "ticker-change", "symbol": "AAPL", "to": "AAPX", "on": "2026-07-30"}
            ],
            "special-dividend": [
                {
                    "kind": "dividend",
                    "symbol": "AAPL",
                    "ex": "2026-07-30",
                    "known": "2026-07-30",
                    "amount": 28.0,
                }
            ],
            "split-reverse-fifth": [
                {
                    "kind": "split",
                    "symbol": "AAPL",
                    "ex": "2026-07-30",
                    "known": "2026-07-30",
                    "ratio": 0.2,
                }
            ],
            "spin-off-price-adjustment": [
                {
                    "kind": "split",
                    "symbol": "AAPL",
                    "ex": "2026-07-30",
                    "known": "2026-07-30",
                    "ratio": 1.589,
                }
            ],
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
    "five-days": {
        "calendar": "XNYS",
        "start": "2023-11-01",
        "symbols": {
            **{symbol: 60.0 + 9 * index for index, symbol in enumerate(("WTA", "WTB", "WTC"))},
            **{symbol: 85.0 + 7 * index for index, symbol in enumerate(("WTD", "WTE", "WTF"))},
            **{symbol: 40.0 + 11 * index for index, symbol in enumerate(("WHA", "WHB", "WHC"))},
            **{symbol: 70.0 + 5 * index for index, symbol in enumerate(("WHD", "WHE", "WHF"))},
            "WTN": 55.0,
            "SPY": 450.0,
        },
        "sectors": {
            **dict.fromkeys(("WTA", "WTB", "WTC", "WTD", "WTE", "WTF", "WTN"), "Technology"),
            **dict.fromkeys(("WHA", "WHB", "WHC", "WHD", "WHE", "WHF"), "Healthcare"),
        },
        "days": [
            "2025-11-21",
            "2025-11-24",
            "2025-11-25",
            "2025-11-26",
            "2025-11-27",
            "2025-11-28",
        ],
        "cases": {
            "five-days": [
                # Monday: a new dividend, and a re-rounding of one history's adjusted closes.
                {
                    "kind": "dividend",
                    "symbol": "WTA",
                    "ex": "2025-11-24",
                    "amount": 0.24,
                    "known": "2025-11-24",
                    "restated": {"known": "2025-11-27", "amount": 0.27},
                },
                {"kind": "reround", "symbol": "WHB", "scale": 4e-07, "known": "2025-11-24"},
                # Tuesday: a leaver and a joiner whose source is down all day, and a split.
                {"kind": "membership", "index": "SP500", "on": "2025-11-25", "leave": ["WHF"]},
                {"kind": "membership", "index": "NASDAQ100", "on": "2025-11-25", "join": ["WTN"]},
                {
                    "kind": "outage",
                    "symbol": "WTN",
                    "on": "2025-11-25",
                    "status": 503,
                    "count": 1000,
                },
                {
                    "kind": "split",
                    "symbol": "WTB",
                    "ex": "2025-11-25",
                    "ratio": 2.0,
                    "known": "2025-11-25",
                },
                # Wednesday: a dividend reported late, and a reverse split.
                {
                    "kind": "dividend",
                    "symbol": "WHA",
                    "ex": "2025-11-24",
                    "amount": 0.3,
                    "known": "2025-11-26",
                },
                {
                    "kind": "split",
                    "symbol": "WHC",
                    "ex": "2025-11-26",
                    "ratio": 0.5,
                    "known": "2025-11-26",
                },
                # Thursday, the holiday: Monday's dividend is corrected (the `restated` above).
                # Friday is the half day.
            ]
        },
        "membership": {
            "SP500": [
                [symbol, symbol.title()] for symbol in ("WHA", "WHB", "WHC", "WHD", "WHE", "WHF")
            ],
            "NASDAQ100": [
                [symbol, symbol.title()] for symbol in ("WTA", "WTB", "WTC", "WTD", "WTE", "WTF")
            ],
            "DJIA": [],
        },
    },
}
