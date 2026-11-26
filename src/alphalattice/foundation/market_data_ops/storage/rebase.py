"""A split-adjusting provider's rebase of a listing's stored bars, kept as factors.

CRSP keeps prices as they traded and a cumulative adjustment factor (CFACPR, CFACSHR) that its
readers apply. A provider that reports a split rescales every earlier bar of the listing, each
re-rounded; the store instead keeps each stored bar times the split's factors and records one
row a split. The kept history and the frozen observations, whose bytes the rescale would
otherwise rewrite, keep theirs and apply the same factors, in the same order, when read: a
rebased value is the same float wherever it is read.
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from hashlib import sha256
from typing import Any

import numpy as np
import numpy.typing as npt

from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    RawDailyBar,
)

PRICES = ("open", "high", "low", "close")
"""The bar fields a rebase scales by its price factor; volume takes its volume factor."""
HALF = 0.5
"""A share count is rounded half away from zero, as the provider rounds it: floor(x + HALF)."""
RAW_REBASES = "raw_rebases"
"""The rebases a refresh kept, as its change document and the Goal disclose them."""


_RECEIPT = (
    "listing_id", "provider", "ex_date", "first_session", "last_session", "price_factor",
    "volume_factor", "largest_difference", "observed_at",
)  # fmt: skip
"""What a rebase's receipt binds."""


@dataclass(frozen=True)
class RawBarRebase:
    """One split's rebase of a listing's stored bars before its ex-date."""

    listing_id: str
    provider: str
    ex_date: date
    first_session: date
    last_session: date
    price_factor: float
    """One over the new shares per old share: a stored price times it."""
    volume_factor: float
    """The new shares per old share: a stored volume times it, rounded."""
    largest_difference: float
    """The largest relative difference from the provider's re-sent bars."""
    observed_at: datetime

    @property
    def receipt_hash(self) -> str:
        """Its identity, of each field it records.

        The fields are declared by name, so a new field is a new identity by intent.
        """
        declared = {name: getattr(self, name) for name in _RECEIPT}
        payload = json.dumps(declared, sort_keys=True, separators=(",", ":"), default=str)
        return sha256(payload.encode("utf-8")).hexdigest()

    def price(self, value: float) -> float:
        """A stored price after it."""
        return value * self.price_factor

    def volume(self, value: float) -> float:
        """A stored volume after it, rounded half away from zero as the provider rounds."""
        return math.floor(value * self.volume_factor + HALF)


def disclosure(rebase: RawBarRebase) -> dict[str, Any]:
    """A rebase as it is disclosed: its split, the sessions it scaled and its factors."""
    return {
        "ex_date": rebase.ex_date.isoformat(),
        "first_session": rebase.first_session.isoformat(),
        "last_session": rebase.last_session.isoformat(),
        "price_factor": rebase.price_factor,
        "volume_factor": rebase.volume_factor,
        "largest_difference": rebase.largest_difference,
        "receipt_hash": rebase.receipt_hash,
    }


def _rescaled(bar: RawDailyBar, rebases: Iterable[RawBarRebase]) -> RawDailyBar:
    for rebase in rebases:
        bar = replace(
            bar,
            **{name: rebase.price(getattr(bar, name)) for name in PRICES},
            volume=int(rebase.volume(bar.volume)),
        )
    return bar


def _difference(observed: RawDailyBar, expected: RawDailyBar) -> float:
    prices = [abs(getattr(observed, n) / getattr(expected, n) - 1.0) for n in PRICES]
    return float(max(*prices, abs(observed.volume - expected.volume) / max(expected.volume, 1)))


def _agrees(observed: RawDailyBar, expected: RawDailyBar, agreement: tuple[float, float]) -> bool:
    """Whether each field is within the rule's agreement: relative, then absolute near zero."""
    return all(
        abs(getattr(observed, n) - getattr(expected, n))
        <= agreement[1] + agreement[0] * abs(getattr(expected, n))
        for n in (*PRICES, "volume")
    )


def standing(
    stored: Sequence[RawDailyBar],
    observed: Sequence[RawDailyBar],
    ex_dates: Collection[date],
    agreement: tuple[float, float],
) -> tuple[RawDailyBar, ...]:
    """The observed bars, a stored one before a known split's ex-date standing for its resend.

    A provider re-rounds the rescale it reported before, so a resent bar within the rule's
    agreement of the stored one is that bar, not a correction; one past it is as observed.
    """
    if not ex_dates or agreement[0] <= 0.0:
        return tuple(observed)
    last, prior = max(ex_dates), {bar.session_date: bar for bar in stored}
    return tuple(
        before
        if bar.session_date < last
        and (before := prior.get(bar.session_date)) is not None
        and _agrees(bar, before, agreement)
        else bar
        for bar in observed
    )


def _of(
    stored: Sequence[RawDailyBar], splits: Sequence[CorporateActionEvent], observed_at: datetime
) -> tuple[RawBarRebase, ...]:
    """Each split's rebase of the stored bars before it, in session order."""
    first = stored[0]
    return tuple(
        RawBarRebase(
            first.listing_id,
            first.provider,
            split.effective_date,
            first.session_date,
            max(bar.session_date for bar in stored if bar.session_date < split.effective_date),
            1.0 / float(split.new_shares_per_old_share or 1.0),
            float(split.new_shares_per_old_share or 1.0),
            0.0,
            observed_at,
        )
        for split in sorted(splits, key=lambda item: item.effective_date)
        if any(bar.session_date < split.effective_date for bar in stored)
    )


def recognize(
    stored: Sequence[RawDailyBar],
    observed: Sequence[RawDailyBar],
    splits: Sequence[CorporateActionEvent],
    agreement: tuple[float, float],
    observed_at: datetime,
) -> tuple[tuple[RawDailyBar, ...], tuple[RawBarRebase, ...], bool] | None:
    """The bars to write, the rebases and whether they cover the stored history, if one is.

    `splits` are the newly reported splits. Every observed bar before a split's ex-date must be
    the stored one times the factors of the splits after it, applied in session order, within
    the rule's rebase `agreement`; the bars from the last ex-date on must be as stored. The
    rebases cover the stored history when every stored session before the last ex-date was
    observed. None otherwise: a correction no split explains restates as before.

    Any ratio is kept. A power of two scales prices and volumes exactly in binary; any other
    can flip a tie or a rounded volume in a Feature, which the Feature engine's early cutoff
    finds and rebuilds, cell by cell.
    """
    if agreement[0] <= 0.0 or not splits or not stored:
        return None
    if observed_at.tzinfo is not None:  # as the store keeps it, so its receipt reads back
        observed_at = observed_at.astimezone(UTC).replace(tzinfo=None)
    rebases = _of(stored, splits, observed_at)
    prior = {bar.session_date: bar for bar in stored}
    written: dict[date, RawDailyBar] = {}
    largest = 0.0
    for bar in observed:
        before = prior.get(bar.session_date)
        if before is None:
            continue
        expected = _rescaled(before, (r for r in rebases if r.ex_date > bar.session_date))
        largest = max(largest, _difference(bar, expected))
        if not _agrees(bar, expected, agreement):
            return None
        if expected != before:
            written[bar.session_date] = expected
    if not written:
        return None
    bars = tuple(written.get(bar.session_date, bar) for bar in observed)
    seen, last = {bar.session_date for bar in observed}, max(r.ex_date for r in rebases)
    whole = all(bar.session_date in seen for bar in stored if bar.session_date < last)
    return bars, tuple(replace(r, largest_difference=largest) for r in rebases), whole


def apply(
    columns: Mapping[str, npt.NDArray[np.float64]],
    rows: slice,
    column: int,
    price_factor: float,
    volume_factor: float,
) -> None:
    """Scale one listing's rows of these OHLCV arrays as the store scaled its bars.

    `columns` maps each bar field to its sessions by listings array, scaled in place.
    """
    for name in PRICES:
        columns[name][rows, column] = columns[name][rows, column] * price_factor
    columns["volume"][rows, column] = np.floor(
        columns["volume"][rows, column] * volume_factor + HALF
    )


def record(connection: Any, rebases: Iterable[RawBarRebase], ingestion_id: str) -> None:
    """Keep each rebase, once by its receipt, with the ingestion that wrote it."""
    rows = [[item.receipt_hash, *asdict(item).values(), ingestion_id] for item in rebases]
    if not rows:
        return
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_bar_rebase (
            receipt_hash VARCHAR PRIMARY KEY, listing_id VARCHAR NOT NULL,
            provider VARCHAR NOT NULL, ex_date DATE NOT NULL, first_session DATE NOT NULL,
            last_session DATE NOT NULL, price_factor DOUBLE NOT NULL,
            volume_factor DOUBLE NOT NULL, largest_difference DOUBLE NOT NULL,
            observed_at TIMESTAMP NOT NULL, ingestion_id VARCHAR NOT NULL
        )
        """
    )
    connection.executemany(
        "INSERT INTO raw_bar_rebase VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT DO NOTHING",
        rows,
    )


def unrebased(
    connection: Any,
    changed: Sequence[tuple[Any, ...]],
    rebases: Sequence[RawBarRebase],
    ingestion_id: str,
) -> tuple[list[tuple[Any, ...]], int]:
    """The changed rows (listing, provider, session, ...) no rebase covers, and how many.

    The rebases are recorded with them: a rebased session is its rebase's, not a correction.
    """
    kept = [
        r for r in changed if all(not b.first_session <= r[2] <= b.last_session for b in rebases)
    ]
    record(connection, rebases, ingestion_id)
    return kept, len(kept)


def watermark(connection: Any, listing_ids: Sequence[str]) -> dict[str, Any]:
    """These listings' rebases in a source watermark: none while there are none, so it holds."""
    rebases = recorded(connection, listing_ids)
    return {RAW_REBASES: [item.receipt_hash for item in rebases]} if rebases else {}


def rebase_disclosure(rebases: Sequence[RawBarRebase]) -> dict[str, Any]:
    """Rebases as a change document discloses them: none while there are none."""
    return {RAW_REBASES: [disclosure(item) for item in rebases]} if rebases else {}


def rebase_changes(
    corrected: Sequence[date], rebases: Sequence[RawBarRebase], stored: Mapping[date, Any]
) -> list[date]:
    """A refresh's changed stored sessions: a rebased one is its rebase's, not a change.

    A split's rebase changes no stored return before its ex-date, so its change is the ex-date
    where it is stored: the windows that read across it.
    """
    covered = {s for s in corrected if any(r.first_session <= s <= r.last_session for r in rebases)}
    return sorted((set(corrected) - covered) | {r.ex_date for r in rebases if r.ex_date in stored})


def ex_date_deltas(
    connection: Any, manifest: Any, after: datetime, through: date
) -> list[tuple[Any, ...]]:
    """A manifest's rebases recorded after `after`, each a raw change at its ex-date.

    A rebase writes no correction per session, so a reader of the corrections since a point
    learns of the split, and of the windows across its ex-date, from its row.
    """
    listings, provider = [item.listing_id for item in manifest.listings], manifest.profile.provider
    return [
        (r.listing_id, r.ex_date, r.receipt_hash, r.receipt_hash, r.observed_at)
        for r in recorded(connection, listings)
        if r.provider == provider and r.observed_at > after and r.ex_date <= through
    ]


def recorded(
    connection: Any, listing_ids: Sequence[str], ingested: str = ""
) -> tuple[RawBarRebase, ...]:
    """These listings' rebases in the order the store applied them: observed, then by ex-date.

    With `ingested`, only those an ingestion id beginning so wrote: a maintenance cycle's own,
    which a restart after the bars' commit reads back to disclose.
    """
    if not connection.execute(
        "SELECT 1 FROM duckdb_tables() WHERE table_name = 'raw_bar_rebase'"
    ).fetchall():
        return ()
    rows = connection.execute(
        """
        SELECT listing_id, provider, ex_date, first_session, last_session, price_factor,
               volume_factor, largest_difference, observed_at
        FROM raw_bar_rebase
        WHERE listing_id IN (SELECT unnest(?)) AND starts_with(ingestion_id, ?)
        ORDER BY observed_at, ex_date, listing_id
        """,
        [list(listing_ids), ingested],
    ).fetchall()
    return tuple(RawBarRebase(*row) for row in rows)
