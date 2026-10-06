"""A Yahoo provider for the maintenance suite: every range recorded, unplanned history refused."""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from threading import Lock

from alphalattice.foundation.market_data_ops.sources.providers import YFinanceMarketDataProvider


class RecordingBoundedProvider:
    """Delegate to Yahoo while rejecting any unplanned historical expansion."""

    name = "yfinance"

    def __init__(
        self,
        cache_root: Path,
        *,
        earliest_allowed: date,
        target: date,
        bounded_listing_starts: dict[str, date] | None = None,
        bounded_subject_starts: dict[str, date] | None = None,
        authorized_full_history_starts: dict[str, date] | None = None,
    ) -> None:
        self.delegate = YFinanceMarketDataProvider(cache_root)
        self.earliest_allowed = earliest_allowed
        self.target = target
        self.bounded_listing_starts = dict(bounded_listing_starts or {})
        self.bounded_subject_starts = dict(bounded_subject_starts or {})
        self.authorized_full_history_starts = dict(authorized_full_history_starts or {})
        self._lock = Lock()
        self.calls: list[dict[str, object]] = []

    def _record(
        self,
        *,
        kind: str,
        subject: str,
        start: date,
        end: date,
        listing_id: str | None = None,
    ) -> tuple[float, str]:
        authorization_scope = "ROLLING_BOUND"
        rolling_start = self.bounded_listing_starts.get(
            listing_id or "",
            self.bounded_subject_starts.get(subject, self.earliest_allowed),
        )
        if start < rolling_start:
            authorized_start = self.authorized_full_history_starts.get(listing_id or "")
            if authorized_start is None or start < authorized_start:
                raise RuntimeError(
                    f"BOUNDED_PROVIDER_PLAN_VIOLATION: {kind} {subject} requested {start}..{end}"
                )
            authorization_scope = "LISTING_FULL_HISTORY"
        if end > self.target:
            raise RuntimeError(
                f"BOUNDED_PROVIDER_PLAN_VIOLATION: {kind} {subject} requested {start}..{end}"
            )
        return time.perf_counter(), authorization_scope

    def _finish(
        self,
        *,
        kind: str,
        subject: str,
        start: date,
        end: date,
        started: float,
        authorization_scope: str,
        listing_id: str | None = None,
    ) -> None:
        with self._lock:
            self.calls.append(
                {
                    "kind": kind,
                    "subject": subject,
                    "listing_id": listing_id,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "authorization_scope": authorization_scope,
                    "wall_seconds": round(time.perf_counter() - started, 6),
                }
            )

    def fetch_hydration(self, *, listing_id, provider_symbol, start, end):
        started, authorization_scope = self._record(
            kind="hydration",
            subject=provider_symbol,
            start=start,
            end=end,
            listing_id=listing_id,
        )
        try:
            return self.delegate.fetch_hydration(
                listing_id=listing_id,
                provider_symbol=provider_symbol,
                start=start,
                end=end,
            )
        finally:
            self._finish(
                kind="hydration",
                subject=provider_symbol,
                start=start,
                end=end,
                started=started,
                authorization_scope=authorization_scope,
                listing_id=listing_id,
            )

    def fetch_daily(self, symbols, *, start, end):
        subject = ",".join(symbols)
        started, authorization_scope = self._record(
            kind="daily", subject=subject, start=start, end=end
        )
        try:
            return self.delegate.fetch_daily(symbols, start=start, end=end)
        finally:
            self._finish(
                kind="daily",
                subject=subject,
                start=start,
                end=end,
                started=started,
                authorization_scope=authorization_scope,
            )

    def __getattr__(self, name: str):
        return getattr(self.delegate, name)
