"""The local-file implementation of the daily market-data port."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from .contracts import CorporateActionEvent, ProviderAdjustedClosePoint
from .providers import ProviderFetchError

_FILE_BAR_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "split_ratio",
    "cash_dividend",
    "capital_gain",
)
_FILE_ACTIONS: tuple[tuple[str, Literal["SPLIT", "CASH_DIVIDEND", "CAPITAL_GAIN"]], ...] = (
    ("split_ratio", "SPLIT"),
    ("cash_dividend", "CASH_DIVIDEND"),
    ("capital_gain", "CAPITAL_GAIN"),
)


@dataclass(frozen=True)
class LocalFileMarketDataProvider:
    """Replay a named source's recorded bars from an already resolved capture directory."""

    root: Path
    name: str

    def _window(self, symbol: str, start: date, end: date) -> dict[date, dict[str, float | None]]:
        if start > end:
            raise ValueError("daily provider range start is after end")
        if not re.fullmatch(r"[A-Za-z0-9^][A-Za-z0-9.^=_-]*", symbol):
            raise ProviderFetchError("data.local_bars_invalid", "Invalid symbol", retryable=False)
        path = self.root / f"{symbol}.json"
        try:
            path = path.resolve()
            if path.parent != self.root:
                raise ValueError("bar file leaves the source folder")
            document = json.loads(path.read_text(encoding="utf-8"))
            if (
                document["schema"] != "alphalattice.market-bars.v1"
                or document["provider"] != self.name
                or not isinstance(document["bars"], list)
            ):
                raise ValueError("bar file has a different format or source")
            seen = set()
            rows = {}
            for raw in document["bars"]:
                session = date.fromisoformat(raw["session_date"])
                if session in seen:
                    raise ValueError("bar file repeats a session")
                seen.add(session)
                if start <= session <= end:
                    rows[session] = {key: self._number(raw[key]) for key in _FILE_BAR_FIELDS}
                    rows[session]["adjusted_close"] = self._number(raw.get("adjusted_close"))
        except FileNotFoundError as error:
            raise ProviderFetchError(
                "data.empty_payload", f"No local bars for {symbol}", retryable=False
            ) from error
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            raise ProviderFetchError(
                "data.local_bars_invalid", f"Cannot read local bars for {symbol}", retryable=False
            ) from error
        return dict(sorted(rows.items()))

    @staticmethod
    def _number(value: object) -> float | None:
        if value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError("a local bar value must be finite or null")
        return float(value)

    def fetch_daily(
        self, symbols: Sequence[str], *, start: date, end: date
    ) -> Mapping[str, Sequence[Mapping[str, object]]]:
        """Return the bounded daily port rows and keep adjusted close on its own surface."""
        if start > end:
            raise ValueError("daily provider range start is after end")
        return {
            symbol: tuple(
                {"session_date": session.isoformat(), **{key: row[key] for key in _FILE_BAR_FIELDS}}
                for session, row in self._window(symbol, start, end).items()
            )
            for symbol in symbols
        }

    def fetch_action_history(
        self, *, listing_id: str, provider_symbol: str, start: date, end: date
    ) -> tuple[CorporateActionEvent, ...]:
        """Read action observations from the same sessions as the daily prices."""
        events = []
        for session, row in self._window(provider_symbol, start, end).items():
            for column, kind in _FILE_ACTIONS:
                amount = row[column]
                if amount is None or amount == 0 or (kind == "SPLIT" and amount == 1):
                    continue
                if amount < 0:
                    raise ProviderFetchError(
                        "data.invalid_action_payload",
                        f"Negative local action amount for {provider_symbol}",
                        retryable=False,
                    )
                events.append(
                    CorporateActionEvent(
                        listing_id=listing_id,
                        provider=self.name,
                        effective_date=session,
                        action_kind=kind,
                        new_shares_per_old_share=amount if kind == "SPLIT" else None,
                        cash_amount=amount if kind != "SPLIT" else None,
                    )
                )
        return tuple(events)

    def fetch_adjusted_close_history(
        self, *, listing_id: str, provider_symbol: str, start: date, end: date
    ) -> tuple[ProviderAdjustedClosePoint, ...]:
        """Read optional diagnostic adjusted-close values supplied by the source."""
        points = []
        for session, row in self._window(provider_symbol, start, end).items():
            value = row["adjusted_close"]
            if value is None:
                continue
            if value <= 0:
                raise ProviderFetchError(
                    "data.invalid_adjusted_close_payload",
                    f"Invalid local adjusted close for {provider_symbol}",
                    retryable=False,
                )
            points.append(ProviderAdjustedClosePoint(listing_id, self.name, session, value))
        return tuple(points)
