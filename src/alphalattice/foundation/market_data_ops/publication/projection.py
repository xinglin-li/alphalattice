"""Feature-ready dynamic projections; adjusted rows are never persisted."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, fields
from datetime import date
from fractions import Fraction
from itertools import pairwise
from typing import Final, Literal

from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    ProviderAdjustedClosePoint,
    RawDailyBar,
)

DailyPriceBasis = Literal["unadjusted", "split_adjusted"]

ADJUSTED_CLOSE_DIAGNOSTIC_POLICY_HASH = hashlib.sha256(
    json.dumps(
        {
            "comparison": "provider-backward-action-adjustment",
            "threshold_bps": 5.0,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


class FeatureAdmissionBlocked(ValueError):
    """A known action ambiguity must not be guessed into a feature series."""

    def __init__(self, code: str, detail: str) -> None:
        """Build a typed projection refusal with its failure code.

        Args:
            code: Stable feature admission failure code.
            detail: Human-readable reason for the refusal.

        """
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class ProjectedBar:
    """One frozen-snapshot row, projected from raw observations at read time."""

    listing_id: str
    session_date: date
    open_raw: float
    high_raw: float
    low_raw: float
    close_raw: float
    volume_raw: int
    cash_dividend: float
    new_shares_per_old_share: float
    # Compatibility vocabulary for existing formal feature consumers.
    split_ratio: float
    open_split_adjusted: float
    high_split_adjusted: float
    low_split_adjusted: float
    close_split_adjusted: float
    volume_split_adjusted: float
    close_total_return_adjusted: float
    action_set_hash: str
    anchor_session: date


@dataclass(frozen=True)
class ProviderAdjustedRatioDiagnostic:
    """Adjacent provider and internal gross-return ratio comparison."""

    previous_session: date
    session_date: date
    provider_gross_return: float
    internal_gross_return: float
    difference_bps: float


def action_set_hash(actions: Iterable[CorporateActionEvent]) -> str:
    """Hash corporate actions in canonical listing and event order.

    Args:
        actions: Corporate action events to bind into one set.

    Returns:
        SHA-256 digest of the sorted action payload.

    """
    payload = [
        {
            "listing_id": item.listing_id,
            "provider": item.provider,
            "effective_date": item.effective_date.isoformat(),
            "action_kind": item.action_kind,
            "new_shares_per_old_share": item.new_shares_per_old_share,
            "cash_amount": item.cash_amount,
            "provisional": item.provisional,
            "provenance": item.provenance,
        }
        for item in sorted(
            actions,
            key=lambda event: (
                event.listing_id,
                event.effective_date,
                event.action_kind,
                event.provenance,
            ),
        )
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def provider_adjusted_close_evidence_hash(
    points: Iterable[ProviderAdjustedClosePoint],
) -> str:
    """Hash an audit-only adjusted-close observation without persisting its rows."""
    payload = [
        {
            "listing_id": item.listing_id,
            "provider": item.provider,
            "session_date": item.session_date.isoformat(),
            "adjusted_close": item.adjusted_close,
        }
        for item in sorted(points, key=lambda item: item.session_date)
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _actions_by_session(
    bars: tuple[RawDailyBar, ...], actions: tuple[CorporateActionEvent, ...]
) -> tuple[dict[date, float], dict[date, float]]:
    sessions = {item.session_date for item in bars}
    multipliers = {item.session_date: 1.0 for item in bars}
    dividends = {item.session_date: 0.0 for item in bars}
    for event in actions:
        if event.effective_date not in sessions:
            raise FeatureAdmissionBlocked(
                "ACTION_OUTSIDE_SNAPSHOT",
                f"{event.action_kind} {event.effective_date.isoformat()} has no matching session",
            )
        if event.action_kind in {"CAPITAL_GAIN", "SPIN_OFF"}:
            raise FeatureAdmissionBlocked(
                "UNSUPPORTED_CORPORATE_ACTION",
                f"{event.action_kind} has no qualified share-unit semantics",
            )
        if event.action_kind == "SPLIT":
            value = event.new_shares_per_old_share
            if value is None or value <= 0:
                raise FeatureAdmissionBlocked("INVALID_SPLIT_MULTIPLIER", str(event.effective_date))
            multipliers[event.effective_date] *= value
        elif event.action_kind == "CASH_DIVIDEND":
            value = event.cash_amount
            if value is None or value < 0:
                raise FeatureAdmissionBlocked("INVALID_CASH_DIVIDEND", str(event.effective_date))
            dividends[event.effective_date] += value
    for session in sessions:
        if multipliers[session] != 1.0 and dividends[session] != 0.0:
            raise FeatureAdmissionBlocked(
                "AMBIGUOUS_SAME_DAY_ACTIONS",
                f"split and cash dividend on {session.isoformat()}",
            )
    return multipliers, dividends


def project_research_series(
    bars: Iterable[RawDailyBar],
    actions: Iterable[CorporateActionEvent],
    *,
    daily_price_basis: DailyPriceBasis = "unadjusted",
) -> tuple[ProjectedBar, ...]:
    """Project a full feature series with the last session as the TR anchor.

    The backward anchor intentionally makes the latest total-return close equal
    to the latest split-adjusted close, while a later action cannot mutate an
    already-written Parquet snapshot.  ``unadjusted`` inputs require dynamic
    split projection. ``split_adjusted`` inputs (the current yfinance contract)
    already use a current share basis, so applying an action split again would
    corrupt prices and volume; their OHLCV is projected unchanged.
    """
    ordered_bars = tuple(sorted(bars, key=lambda item: item.session_date))
    if not ordered_bars:
        return ()
    listing_ids = {item.listing_id for item in ordered_bars}
    if len(listing_ids) != 1:
        raise ValueError("projection accepts exactly one listing")
    ordered_actions = tuple(
        sorted(actions, key=lambda item: (item.effective_date, item.action_kind))
    )
    if any(item.listing_id != ordered_bars[0].listing_id for item in ordered_actions):
        raise ValueError("action listing does not match bar listing")
    multipliers, dividends = _actions_by_session(ordered_bars, ordered_actions)
    if daily_price_basis not in {"unadjusted", "split_adjusted"}:
        raise ValueError(f"unsupported daily price basis: {daily_price_basis}")
    projection_multipliers = (
        multipliers
        if daily_price_basis == "unadjusted"
        else {session: 1.0 for session in multipliers}
    )
    digest = action_set_hash(ordered_actions)

    later_multiplier = 1.0
    split_adjusted: list[tuple[float, float, float, float, float]] = [
        (0.0, 0.0, 0.0, 0.0, 0.0) for _ in ordered_bars
    ]
    for index in range(len(ordered_bars) - 1, -1, -1):
        bar = ordered_bars[index]
        split_adjusted[index] = (
            bar.open / later_multiplier,
            bar.high / later_multiplier,
            bar.low / later_multiplier,
            bar.close / later_multiplier,
            float(bar.volume) * later_multiplier,
        )
        later_multiplier *= projection_multipliers[bar.session_date]

    total_return = [0.0 for _ in ordered_bars]
    total_return[-1] = split_adjusted[-1][3]
    for index in range(len(ordered_bars) - 1, 0, -1):
        current = ordered_bars[index]
        previous = ordered_bars[index - 1]
        gross_return = (
            projection_multipliers[current.session_date]
            * (current.close + dividends[current.session_date])
            / previous.close
        )
        if gross_return <= 0:
            raise FeatureAdmissionBlocked("INVALID_GROSS_RETURN", current.session_date.isoformat())
        total_return[index - 1] = total_return[index] / gross_return

    anchor = ordered_bars[-1].session_date
    return tuple(
        ProjectedBar(
            listing_id=bar.listing_id,
            session_date=bar.session_date,
            open_raw=bar.open,
            high_raw=bar.high,
            low_raw=bar.low,
            close_raw=bar.close,
            volume_raw=bar.volume,
            cash_dividend=dividends[bar.session_date],
            new_shares_per_old_share=multipliers[bar.session_date],
            split_ratio=multipliers[bar.session_date],
            open_split_adjusted=split_adjusted[index][0],
            high_split_adjusted=split_adjusted[index][1],
            low_split_adjusted=split_adjusted[index][2],
            close_split_adjusted=split_adjusted[index][3],
            volume_split_adjusted=split_adjusted[index][4],
            close_total_return_adjusted=total_return[index],
            action_set_hash=digest,
            anchor_session=anchor,
        )
        for index, bar in enumerate(ordered_bars)
    )


SHARE_SPLIT_DENOMINATOR_LIMIT: Final = 100
"""The largest denominator a share count change's ratio takes, either way round."""

SHARE_SPLIT_RELATIVE_TOLERANCE: Final = 5e-6
"""How far a recorded ratio may sit from its simple fraction and still be one: the provider's
rounding of a reverse split (1/3 as 0.333333), never a spin-off's ratio."""


def is_share_split(ratio: float) -> bool:
    """Whether a recorded split ratio changes the listing's share count or only its price.

    The provider has no spin-off event: a spin-off arrives as a fractional split ratio (RTX
    1.589 for Carrier and Otis, GE 1.281 and 1.253, DHR 1.128, MMM 1.196) that re-bases the
    price as a split does and issues no share of this listing. A share split or a stock dividend
    gives whole shares for whole shares, so its ratio, or for a reverse split its reciprocal, is
    a simple fraction (2, 3/2, 50, 21/20, 1/8 as 8, 1/20 as 20); any other ratio is a price
    adjustment.

    Args:
        ratio: New shares per old share, as the provider recorded it.

    Returns:
        True for a change of the share count, False for a price adjustment.
    """

    def simple(value: float) -> bool:
        fraction = Fraction(value).limit_denominator(SHARE_SPLIT_DENOMINATOR_LIMIT)
        return fraction > 0 and abs(float(fraction) / value - 1.0) <= (
            SHARE_SPLIT_RELATIVE_TOLERANCE
        )

    return math.isfinite(ratio) and ratio > 0 and (simple(ratio) or simple(1.0 / ratio))


@dataclass(frozen=True)
class AsTradedBar:
    """One session's prices and share volume as they traded, and the indices that re-base them.

    Point in time: no later event changes a session's values. ``price_adjustment_index``
    is the product of every recorded ratio effective on or before the session and
    ``share_count_index`` the product of the share splits among them, so the price at
    session t as it read at a later session T is ``close_as_traded(t) * price_adjustment_index(t)
    price_adjustment_index(T)``, and the share volume ``volume_as_traded(t) *
    share_count_index(T) / share_count_index(t)``.
    """

    session_date: date
    open_as_traded: float
    high_as_traded: float
    low_as_traded: float
    close_as_traded: float
    volume_as_traded: float
    price_adjustment_index: float
    share_count_index: float


AS_TRADED_FIELDS: Final = tuple(item.name for item in fields(AsTradedBar))[1:]
"""The source fields an as-traded row adds to a listing's feature frame."""


def project_as_traded_series(
    bars: Iterable[RawDailyBar],
    actions: Iterable[CorporateActionEvent],
    *,
    daily_price_basis: DailyPriceBasis = "unadjusted",
) -> tuple[AsTradedBar, ...]:
    """Rebuild one listing's sessions as they traded, from its bars and every recorded split.

    ``unadjusted`` bars are as traded. ``split_adjusted`` bars (the current yfinance contract)
    are in the share basis of the download, every recorded ratio after a session applied to
    its prices and its volume as the provider's split mechanics apply them, so the session's
    values as traded multiply its prices by those ratios and divide its volume by them. The
    provider's share volume before a price adjustment is assumed scaled by it as by a split:
    that cannot be checked offline. ``actions`` must hold every recorded action of the
    listing, later ones included; a cash dividend moves no price here.

    Args:
        bars: The listing's daily bars.
        actions: Its recorded corporate actions, all of them.
        daily_price_basis: The basis its bars are stored in.

    Returns:
        One row per bar, in session order.

    Raises:
        FeatureAdmissionBlocked: A capital gain or a spin-off event, which have no share-unit
            semantics, or a split ratio that is not positive.
        ValueError: Bars of more than one listing, an action of another listing, or an
            unsupported price basis.
    """
    ordered = tuple(sorted(bars, key=lambda item: item.session_date))
    if not ordered:
        return ()
    if len({item.listing_id for item in ordered}) != 1:
        raise ValueError("projection accepts exactly one listing")
    if daily_price_basis not in {"unadjusted", "split_adjusted"}:
        raise ValueError(f"unsupported daily price basis: {daily_price_basis}")
    ratios: list[tuple[date, float, bool]] = []
    for event in actions:
        if event.listing_id != ordered[0].listing_id:
            raise ValueError("action listing does not match bar listing")
        if event.action_kind in {"CAPITAL_GAIN", "SPIN_OFF"}:
            raise FeatureAdmissionBlocked(
                "UNSUPPORTED_CORPORATE_ACTION",
                f"{event.action_kind} has no qualified share-unit semantics",
            )
        if event.action_kind == "SPLIT":
            value = event.new_shares_per_old_share
            if value is None or not math.isfinite(value) or value <= 0:
                raise FeatureAdmissionBlocked("INVALID_SPLIT_MULTIPLIER", str(event.effective_date))
            ratios.append((event.effective_date, value, is_share_split(value)))
    ratios.sort()

    def product(selected: Iterable[float]) -> float:
        value = 1.0
        for item in selected:
            value *= item
        return value

    rows = []
    for bar in ordered:
        later = (
            product(r for when, r, _ in ratios if when > bar.session_date)
            if daily_price_basis == "split_adjusted"
            else 1.0
        )
        rows.append(
            AsTradedBar(
                session_date=bar.session_date,
                open_as_traded=bar.open * later,
                high_as_traded=bar.high * later,
                low_as_traded=bar.low * later,
                close_as_traded=bar.close * later,
                volume_as_traded=float(bar.volume) / later,
                price_adjustment_index=product(
                    r for when, r, _ in ratios if when <= bar.session_date
                ),
                share_count_index=product(
                    r for when, r, share in ratios if share and when <= bar.session_date
                ),
            )
        )
    return tuple(rows)


def provider_adjusted_ratio_diagnostics(
    bars: Iterable[RawDailyBar],
    actions: Iterable[CorporateActionEvent],
    adjusted_closes: Iterable[ProviderAdjustedClosePoint],
    *,
    daily_price_basis: DailyPriceBasis = "unadjusted",
) -> tuple[ProviderAdjustedRatioDiagnostic, ...]:
    """Compare provider adjusted-close ratios with its implied action convention.

    This is deliberately an internal consistency diagnostic, not independent
    action authority: both inputs currently come from the same provider.

    Yahoo-style adjusted close is a backward price adjustment, not a wealth
    index.  On an ex-dividend session its adjacent ratio is equivalent to
    ``close[t] / (close[t-1] - dividend[t])`` (plus any still-required split
    multiplier), whereas the economically meaningful total return remains
    ``(close[t] + dividend[t]) / close[t-1]``.  Comparing those two different
    conventions creates false mismatches precisely on volatile ex-dividend
    sessions.  This diagnostic therefore predicts the provider convention;
    :func:`project_research_series` remains the authority for total return.
    """
    ordered_bars = tuple(sorted(bars, key=lambda item: item.session_date))
    if len(ordered_bars) < 2:
        return ()
    ordered_actions = tuple(
        sorted(actions, key=lambda item: (item.effective_date, item.action_kind))
    )
    observed = tuple(sorted(adjusted_closes, key=lambda item: item.session_date))
    if any(
        item.listing_id != ordered_bars[0].listing_id
        or item.provider != ordered_bars[0].provider
        or not math.isfinite(item.adjusted_close)
        or item.adjusted_close <= 0
        for item in observed
    ):
        raise ValueError("provider adjusted-close diagnostic has invalid identity or value")
    adjusted_by_session = {item.session_date: item.adjusted_close for item in observed}
    if len(adjusted_by_session) != len(observed):
        raise ValueError("provider adjusted-close diagnostic contains duplicate sessions")
    multipliers, dividends = _actions_by_session(ordered_bars, ordered_actions)
    if daily_price_basis not in {"unadjusted", "split_adjusted"}:
        raise ValueError(f"unsupported daily price basis: {daily_price_basis}")
    return_multipliers = (
        multipliers
        if daily_price_basis == "unadjusted"
        else {session: 1.0 for session in multipliers}
    )
    diagnostics: list[ProviderAdjustedRatioDiagnostic] = []
    for previous, current in pairwise(ordered_bars):
        previous_adjusted = adjusted_by_session.get(previous.session_date)
        current_adjusted = adjusted_by_session.get(current.session_date)
        if previous_adjusted is None or current_adjusted is None:
            continue
        provider_gross_return = current_adjusted / previous_adjusted
        backward_adjusted_denominator = previous.close - dividends[current.session_date]
        if backward_adjusted_denominator <= 0.0:
            raise ValueError(
                "provider adjusted-close diagnostic has a non-positive "
                "backward-adjusted denominator"
            )
        internal_gross_return = (
            return_multipliers[current.session_date] * current.close / backward_adjusted_denominator
        )
        difference_bps = abs(provider_gross_return / internal_gross_return - 1.0) * 10_000.0
        diagnostics.append(
            ProviderAdjustedRatioDiagnostic(
                previous.session_date,
                current.session_date,
                provider_gross_return,
                internal_gross_return,
                difference_bps,
            )
        )
    return tuple(diagnostics)
