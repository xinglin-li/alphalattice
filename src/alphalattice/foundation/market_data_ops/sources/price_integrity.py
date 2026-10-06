"""Deterministic integrity checks for provider split-adjusted price histories."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from typing import Literal

from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    RawDailyBar,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

#: Data-owned quarantine bound for a single cash dividend against the same
#: session's close. Deliberately broad: ordinary dividends sit far below one
#: percent of price and even large special distributions rarely approach a
#: quarter of it, so a breach is a data-defect signal, not a tuning surface.
#: This is a Data semantic and is named here rather than borrowed from the
#: Feature input policy, whose bps diagnostic answers a different question.
EXTREME_DIVIDEND_TO_CLOSE_RATIO = 0.25


class SplitAdjustedPriceIntegrityError(ValueError):
    """A provider history still exposes a split-sized discontinuity."""

    code = "data.unrepaired_split_adjustment"


def validate_split_adjusted_history(
    rows: Sequence[Mapping[str, object]],
    *,
    symbol: str,
    split_match_tolerance: float = 0.15,
    unexplained_ratio_limit: float = 5.0,
) -> None:
    """Reject action-aligned or unexplained split-sized open discontinuities."""
    if not 0.0 < split_match_tolerance < 1.0:
        raise ValueError("split match tolerance must be between zero and one")
    if unexplained_ratio_limit <= 1.0:
        raise ValueError("unexplained ratio limit must exceed one")
    ordered = sorted(rows, key=lambda item: _session(item.get("session_date")))
    for prior, current in pairwise(ordered):
        prior_open = _positive_number(prior.get("open"), "open")
        current_open = _positive_number(current.get("open"), "open")
        observed_ratio = current_open / prior_open
        split = _optional_number(current.get("split_ratio"), "split_ratio")
        current_session = _session(current.get("session_date"))
        if split is not None and split not in (0.0, 1.0):
            if split <= 0.0:
                raise SplitAdjustedPriceIntegrityError(
                    f"{symbol} has an invalid split ratio on {current_session.isoformat()}"
                )
            split_magnitude = max(split, 1.0 / split)
            if split_magnitude >= 2.0:
                unrepaired_ratio = 1.0 / split
                relative_error = abs(observed_ratio / unrepaired_ratio - 1.0)
                if relative_error <= split_match_tolerance:
                    raise SplitAdjustedPriceIntegrityError(
                        f"{symbol} open ratio {observed_ratio:.8g} remains close to "
                        f"unrepaired split ratio {unrepaired_ratio:.8g} on "
                        f"{current_session.isoformat()}"
                    )
        cash_event = any(
            (_optional_number(current.get(field), field) or 0.0) != 0.0
            for field in ("cash_dividend", "capital_gain")
        )
        if cash_event:
            continue
        if (
            observed_ratio >= unexplained_ratio_limit
            or observed_ratio <= 1.0 / unexplained_ratio_limit
        ):
            raise SplitAdjustedPriceIntegrityError(
                f"{symbol} has unexplained split-sized open ratio "
                f"{observed_ratio:.8g} on {current_session.isoformat()}"
            )


def _session(value: object) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise SplitAdjustedPriceIntegrityError("provider history has an invalid session date")


def _positive_number(value: object, field: str) -> float:
    number = _optional_number(value, field)
    if number is None or number <= 0.0:
        raise SplitAdjustedPriceIntegrityError(f"provider history has an invalid {field} value")
    return number


def _optional_number(value: object, field: str) -> float | None:
    if value is None:
        return None
    try:
        if not isinstance(value, (int, float, str)):
            raise TypeError
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SplitAdjustedPriceIntegrityError(
            f"provider history has a non-numeric {field} value"
        ) from exc
    if not math.isfinite(number):
        raise SplitAdjustedPriceIntegrityError(f"provider history has a non-finite {field} value")
    return number


@dataclass(frozen=True)
class TradingSessionAuthority:
    """The qualified ordered session axis a sentinel may anchor membership against.

    Data owns what the authority *is*; the Host owns where the sessions come
    from, because materializing an exchange calendar is a qualification act and
    a refresh runner that invented its own axis would be quarantining listings
    against a calendar nobody qualified.

    An authority has a scope, and says nothing outside it. Membership is checked
    only for dates within ``[first_session, last_session]``: an axis covering
    2016-2026 is not evidence about 2015, and treating it as such would turn a
    bounded fact into a false quarantine.
    """

    calendar_ids: tuple[str, ...]
    sessions: tuple[date, ...]
    authority_hash: str

    def __post_init__(self) -> None:
        """Verify canonical sessions, calendars, and their authority hash.

        Raises:
            ValueError: The axis is empty, noncanonical, or has an invalid hash.

        """
        if not self.calendar_ids or self.calendar_ids != tuple(sorted(set(self.calendar_ids))):
            raise ValueError("trading session authority calendar ids are not canonical")
        if not self.sessions or self.sessions != tuple(sorted(set(self.sessions))):
            raise ValueError("trading session authority sessions are not canonical")
        if self.authority_hash != _trading_session_authority_hash(self.calendar_ids, self.sessions):
            raise ValueError("trading session authority hash is invalid")

    @property
    def first_session(self) -> date:
        """Return the first session this authority may judge."""
        return self.sessions[0]

    @property
    def last_session(self) -> date:
        """Return the last session this authority may judge."""
        return self.sessions[-1]

    @property
    def session_set(self) -> frozenset[date]:
        """Return the qualified session dates as a lookup set."""
        return frozenset(self.sessions)

    def covers(self, value: date) -> bool:
        """Whether this authority is entitled to judge one date at all."""
        return self.first_session <= value <= self.last_session


def _trading_session_authority_hash(
    calendar_ids: tuple[str, ...], sessions: tuple[date, ...]
) -> str:
    return canonical_hash(
        {
            "kind": "TradingSessionAuthority",
            "calendar_ids": list(calendar_ids),
            "sessions": [value.isoformat() for value in sessions],
        }
    )


def build_trading_session_authority(
    *, calendar_ids: Sequence[str], sessions: Sequence[date]
) -> TradingSessionAuthority:
    """Seal one resolved session axis; the caller supplies sessions it qualified."""
    ordered_calendars = tuple(sorted(set(calendar_ids)))
    ordered_sessions = tuple(sorted(set(sessions)))
    return TradingSessionAuthority(
        calendar_ids=ordered_calendars,
        sessions=ordered_sessions,
        authority_hash=_trading_session_authority_hash(ordered_calendars, ordered_sessions),
    )


@dataclass(frozen=True)
class PriceActionSentinelPolicy:
    """Bounds the sentinel evaluates against, content-addressed for evidence."""

    extreme_dividend_to_close_ratio: float = EXTREME_DIVIDEND_TO_CLOSE_RATIO
    golden_relative_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        """Require both sentinel bounds to be proper fractions.

        Raises:
            ValueError: Either bound is outside the open unit interval.

        """
        if not 0.0 < self.extreme_dividend_to_close_ratio < 1.0:
            raise ValueError("extreme dividend ratio bound must be between zero and one")
        if not 0.0 < self.golden_relative_tolerance < 1.0:
            raise ValueError("golden relative tolerance must be between zero and one")

    @property
    def policy_hash(self) -> str:
        """Return the content identity of the sentinel bounds."""
        return canonical_hash(
            {
                "kind": "PriceActionSentinelPolicy",
                "extreme_dividend_to_close_ratio": self.extreme_dividend_to_close_ratio,
                "golden_relative_tolerance": self.golden_relative_tolerance,
            }
        )


@dataclass(frozen=True)
class KnownCorporateActionExpectation:
    """One externally known event the provider history must reproduce.

    Golden event data belongs in a case-study fixture, never in product
    constants: what this module owns is the comparison, not the truth.
    """

    listing_id: str
    effective_date: date
    action_kind: Literal["SPLIT", "CASH_DIVIDEND"]
    expected_value: float

    def __post_init__(self) -> None:
        """Require a finite, positive expected action value.

        Raises:
            ValueError: The expected value is nonfinite or nonpositive.

        """
        if not math.isfinite(self.expected_value) or self.expected_value <= 0.0:
            raise ValueError("known corporate action expectation must be finite and positive")


@dataclass(frozen=True)
class PriceActionSentinelFinding:
    """One listing and session that needs price-action review."""

    code: str
    listing_id: str
    session_date: date
    detail: str


@dataclass(frozen=True)
class PriceActionIntegritySentinelReport:
    """Typed, content-addressed outcome of one bounded sentinel evaluation."""

    provider: str
    as_of: date
    listing_ids: tuple[str, ...]
    input_identity: str
    policy_hash: str
    session_authority_hash: str | None
    """Identity of the qualified session axis membership was judged against, or
    ``None`` when no authority was supplied and membership was not evaluated."""

    findings: tuple[PriceActionSentinelFinding, ...]
    disposition: Literal["ANCHORED", "QUARANTINE_REVIEW_REQUIRED"]
    report_hash: str


def evaluate_price_action_integrity_sentinels(
    *,
    bars: Sequence[RawDailyBar],
    actions: Sequence[CorporateActionEvent],
    provider: str,
    as_of: date,
    trading_sessions: TradingSessionAuthority | None,
    known_actions: Sequence[KnownCorporateActionExpectation] = (),
    policy: PriceActionSentinelPolicy | None = None,
) -> PriceActionIntegritySentinelReport:
    """Evaluate bounded deterministic sentinels over one sanitized batch.

    A sentinel proves that the current Provider path remains internally and
    externally anchored at selected known points; it does not make one Provider
    universally authoritative. The checks here are only what the sanitization
    path does not already enforce: trading-session membership of bars and
    action effective dates, an extreme dividend-to-close quarantine bound, and
    fixture-supplied known split / cash-dividend expectations. Everything is a
    typed finding with a quarantine disposition rather than an exception, so
    the qualification consumer decides what a breach does; nothing here writes.
    """
    active_policy = policy or PriceActionSentinelPolicy()
    ordered_bars = tuple(sorted(bars, key=lambda item: (item.listing_id, item.session_date)))
    ordered_actions = tuple(
        sorted(actions, key=lambda item: (item.listing_id, item.effective_date, item.action_kind))
    )
    if not ordered_bars:
        raise ValueError("price-action sentinel evaluation requires at least one bar")
    close_by_key = {(bar.listing_id, bar.session_date): bar.close for bar in ordered_bars}
    findings: list[PriceActionSentinelFinding] = []

    if trading_sessions is not None:
        # Only dates the authority actually covers are judged. Outside its
        # range the authority is silent rather than negative, so a batch
        # reaching further back than the qualified axis is not quarantined for
        # a fact nobody established.
        members = trading_sessions.session_set
        for bar in ordered_bars:
            if trading_sessions.covers(bar.session_date) and bar.session_date not in members:
                findings.append(
                    PriceActionSentinelFinding(
                        code="data.sentinel.bar_session_not_trading",
                        listing_id=bar.listing_id,
                        session_date=bar.session_date,
                        detail="raw bar dated outside the qualified exchange session axis",
                    )
                )
        for event in ordered_actions:
            if trading_sessions.covers(event.effective_date) and (
                event.effective_date not in members
            ):
                findings.append(
                    PriceActionSentinelFinding(
                        code="data.sentinel.action_session_not_trading",
                        listing_id=event.listing_id,
                        session_date=event.effective_date,
                        detail=(
                            f"{event.action_kind} dated outside the qualified exchange session axis"
                        ),
                    )
                )

    for event in ordered_actions:
        if event.action_kind != "CASH_DIVIDEND":
            continue
        amount = event.cash_amount
        if amount is None or not math.isfinite(amount) or amount < 0.0:
            findings.append(
                PriceActionSentinelFinding(
                    code="data.sentinel.dividend_amount_invalid",
                    listing_id=event.listing_id,
                    session_date=event.effective_date,
                    detail="cash dividend amount is missing, non-finite, or negative",
                )
            )
            continue
        close = close_by_key.get((event.listing_id, event.effective_date))
        if close is None:
            findings.append(
                PriceActionSentinelFinding(
                    code="data.sentinel.dividend_without_bar",
                    listing_id=event.listing_id,
                    session_date=event.effective_date,
                    detail="cash dividend has no same-session raw bar to anchor against",
                )
            )
        elif amount / close > active_policy.extreme_dividend_to_close_ratio:
            findings.append(
                PriceActionSentinelFinding(
                    code="data.sentinel.extreme_dividend_ratio",
                    listing_id=event.listing_id,
                    session_date=event.effective_date,
                    detail=(
                        f"dividend {amount:.8g} is {amount / close:.4f} of close {close:.8g}, "
                        f"above the {active_policy.extreme_dividend_to_close_ratio} bound"
                    ),
                )
            )

    observed_by_key: dict[tuple[str, date, str], float | None] = {}
    for event in ordered_actions:
        value = (
            event.new_shares_per_old_share if event.action_kind == "SPLIT" else event.cash_amount
        )
        observed_by_key[(event.listing_id, event.effective_date, event.action_kind)] = value
    for expectation in sorted(
        known_actions,
        key=lambda item: (item.listing_id, item.effective_date, item.action_kind),
    ):
        key = (expectation.listing_id, expectation.effective_date, expectation.action_kind)
        observed = observed_by_key.get(key)
        if observed is None:
            findings.append(
                PriceActionSentinelFinding(
                    code="data.sentinel.golden_action_missing",
                    listing_id=expectation.listing_id,
                    session_date=expectation.effective_date,
                    detail=(
                        f"known {expectation.action_kind} "
                        f"{expectation.expected_value:.8g} is absent from the provider history"
                    ),
                )
            )
        elif (
            abs(observed / expectation.expected_value - 1.0)
            > active_policy.golden_relative_tolerance
        ):
            findings.append(
                PriceActionSentinelFinding(
                    code="data.sentinel.golden_action_mismatch",
                    listing_id=expectation.listing_id,
                    session_date=expectation.effective_date,
                    detail=(
                        f"known {expectation.action_kind} expected "
                        f"{expectation.expected_value:.8g}, provider reports {observed:.8g}"
                    ),
                )
            )

    input_identity = canonical_hash(
        {
            "kind": "PriceActionSentinelInput",
            "provider": provider,
            "bars": [
                (
                    bar.listing_id,
                    bar.session_date,
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                )
                for bar in ordered_bars
            ],
            "actions": [
                (
                    event.listing_id,
                    event.effective_date,
                    event.action_kind,
                    event.new_shares_per_old_share,
                    event.cash_amount,
                )
                for event in ordered_actions
            ],
        }
    )
    ordered_findings = tuple(
        sorted(findings, key=lambda item: (item.listing_id, item.session_date, item.code))
    )
    disposition: Literal["ANCHORED", "QUARANTINE_REVIEW_REQUIRED"] = (
        "QUARANTINE_REVIEW_REQUIRED" if ordered_findings else "ANCHORED"
    )
    listing_ids = tuple(sorted({bar.listing_id for bar in ordered_bars}))
    session_authority_hash = (
        trading_sessions.authority_hash if trading_sessions is not None else None
    )
    report_hash = canonical_hash(
        {
            "kind": "PriceActionIntegritySentinelReport",
            "provider": provider,
            "as_of": as_of,
            "listing_ids": listing_ids,
            "input_identity": input_identity,
            "policy_hash": active_policy.policy_hash,
            "session_authority_hash": session_authority_hash,
            "findings": [
                (item.code, item.listing_id, item.session_date, item.detail)
                for item in ordered_findings
            ],
            "disposition": disposition,
        }
    )
    return PriceActionIntegritySentinelReport(
        provider=provider,
        as_of=as_of,
        listing_ids=listing_ids,
        input_identity=input_identity,
        policy_hash=active_policy.policy_hash,
        session_authority_hash=session_authority_hash,
        findings=ordered_findings,
        disposition=disposition,
        report_hash=report_hash,
    )


__all__ = [
    "EXTREME_DIVIDEND_TO_CLOSE_RATIO",
    "KnownCorporateActionExpectation",
    "PriceActionIntegritySentinelReport",
    "PriceActionSentinelFinding",
    "PriceActionSentinelPolicy",
    "SplitAdjustedPriceIntegrityError",
    "TradingSessionAuthority",
    "build_trading_session_authority",
    "evaluate_price_action_integrity_sentinels",
    "validate_split_adjusted_history",
]
