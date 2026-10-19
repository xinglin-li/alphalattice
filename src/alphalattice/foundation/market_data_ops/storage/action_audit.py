"""What an action audit decides from its inputs alone, apart from the store that reads them.

The market-data store (`storage.duckdb`) reads each observation's evidence, set-based across the
observations of one call, and writes the receipts; this module holds the observation it is
given and the pure parts of its judgment: the observation's own scope, the mapping revision in
force and the receipt's payload.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from itertools import pairwise
from typing import Any

from alphalattice.foundation.market_data_ops.publication.projection import (
    ADJUSTED_CLOSE_DIAGNOSTIC_POLICY_HASH,
    action_set_hash,
    provider_adjusted_close_evidence_hash,
    provider_adjusted_ratio_diagnostics,
)
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    ProviderAdjustedClosePoint,
    RawDailyBar,
)
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest


@dataclass(frozen=True)
class ActionAuditObservation:
    """One listing's provider observation that an action audit reconciles over its scope."""

    listing_id: str
    provider: str
    actions: Sequence[CorporateActionEvent]
    adjusted_closes: Sequence[ProviderAdjustedClosePoint]
    history_start: date
    history_end: date
    observed_at: datetime
    requested_as_of: date | None = None
    """The session the audit is for; None is its ``history_end``."""
    adjusted_tolerance: float = 0.0
    """The data rule's relative precision for stored adjusted closes observed again: within it
    the stored value stands. Zero when the refresh corrected a raw bar."""

    @property
    def as_of(self) -> date:
        """The session the audit is for."""
        return self.history_end if self.requested_as_of is None else self.requested_as_of


def audited_natural_keys(item: ActionAuditObservation) -> set[tuple[date, str]]:
    """The observation's action keys, once its scope and keys are proven its own."""
    if item.history_start > item.history_end or item.history_end < item.as_of:
        raise ValueError("action audit range must cover requested as_of")
    scope = (item.listing_id, item.provider)
    if any(
        (event.listing_id, event.provider) != scope
        or not item.history_start <= event.effective_date <= item.history_end
        for event in item.actions
    ):
        raise ValueError("action audit observation is outside its declared scope")
    keys = {(event.effective_date, event.action_kind) for event in item.actions}
    if len(keys) != len(item.actions):
        raise ValueError("action audit contains duplicate natural keys")
    if any(
        (point.listing_id, point.provider) != scope
        or not item.history_start <= point.session_date <= item.history_end
        for point in item.adjusted_closes
    ):
        raise ValueError("adjusted-close audit observation is outside its declared scope")
    return keys


def mapping_in_force(rows: Sequence[tuple[Any, ...]], as_of: date) -> list[tuple[Any, ...]]:
    """The mapping rows a listing's revision binds at `as_of`: those in force, else all of them.

    This current-only manifest has no historical mapping lifecycle yet; its one active mapping
    is a bootstrap identity, which future lifecycle authority must replace, not infer from.
    """
    current = [row for row in rows if row[1] <= as_of and (row[2] is None or row[2] >= as_of)]
    if not current and not rows:
        raise ValueError("listing has no provider symbol mapping")
    return current or list(rows)


def receipt_payload(
    manifest: UniverseManifest,
    item: ActionAuditObservation,
    *,
    active: tuple[CorporateActionEvent, ...],
    bars: tuple[RawDailyBar, ...],
    mapping_revision: str,
    raw_digest: str,
    action_evidence: str,
) -> dict[str, object]:
    """One observation's receipt payload over bars that cover its adjusted closes exactly."""
    diagnostics = provider_adjusted_ratio_diagnostics(
        bars,
        tuple(
            event
            for event in active
            if item.history_start <= event.effective_date <= item.history_end
        ),
        item.adjusted_closes,
        daily_price_basis=manifest.profile.daily_price_basis,
    )
    mismatches = tuple(entry for entry in diagnostics if entry.difference_bps > 5.0)
    return {
        "listing_id": item.listing_id,
        "provider": item.provider,
        "manifest_revision": manifest.revision_sha256,
        "mapping_revision": mapping_revision,
        "requested_as_of": item.as_of,
        "history_start": item.history_start,
        "history_end": item.history_end,
        "raw_evidence_hash": raw_digest,
        "action_set_hash": action_set_hash(active),
        "action_evidence_hash": action_evidence,
        "provider_adjusted_close_evidence_hash": provider_adjusted_close_evidence_hash(
            item.adjusted_closes
        ),
        "max_adjusted_close_difference_bps": max(
            (entry.difference_bps for entry in diagnostics), default=0.0
        ),
        "adjusted_close_mismatch_count": len(mismatches),
        "first_adjusted_close_mismatch_session": (
            mismatches[0].session_date if mismatches else None
        ),
        "diagnostic_policy_hash": ADJUSTED_CLOSE_DIAGNOSTIC_POLICY_HASH,
        "observed_at": item.observed_at,
    }


def adjusted_return_changes(
    prior: Mapping[date, float],
    current: Mapping[date, float],
    *,
    refreshed_from: date | None = None,
) -> tuple[date, ...]:
    """The sessions whose log return differs between two adjusted series.

    `refreshed_from` names the first session a rolling refresh replaced, every earlier one
    kept as it was: a return can then change only from the kept session before it, so the
    earlier returns, which read the same two closes on both sides, are not compared.
    """
    if refreshed_from is not None:
        since = max((session for session in prior if session < refreshed_from), default=None)
        if since is not None:
            prior = {session: close for session, close in prior.items() if session >= since}
            current = {session: close for session, close in current.items() if session >= since}

    def returns(values: Mapping[date, float]) -> dict[date, float]:
        sessions = sorted(values)
        return {
            session: math.log(float(values[session]) / float(values[previous]))
            for previous, session in pairwise(sessions)
        }

    old_returns = returns(prior)
    new_returns = returns(current)
    return tuple(
        session
        for session in sorted(set(old_returns) | set(new_returns))
        if session not in old_returns
        or session not in new_returns
        or abs(new_returns[session] - old_returns[session]) > 1e-12
    )


def _within_precision(
    prior: Mapping[date, float], observed: Mapping[date, float], tolerance: float
) -> float | None:
    """The largest relative difference of stored closes observed again, when inside `tolerance`.

    None when nothing differs, when nothing is shared, or when any difference passes it.
    """
    shared = observed.keys() & prior.keys()
    if tolerance <= 0.0 or not shared:
        return None
    largest = max(abs(observed[session] - prior[session]) / prior[session] for session in shared)
    return largest if 0.0 < largest <= tolerance else None


def standing(
    prior: Mapping[date, float],
    observed: dict[date, float],
    added: set[date],
    tolerance: float,
    *,
    replaces: bool,
) -> tuple[dict[date, float], dict[date, float], float | None]:
    """The closes a refresh writes, the series it leaves, and how far stored closes stood.

    A stored close observed again within the data rule's precision stands: a later observation
    does not rewrite a published session, so only the new sessions are written then. Otherwise
    a full refresh (`replaces`) leaves its own series and a rolling one the stored one updated.
    """
    kept = _within_precision(prior, observed, tolerance)
    if kept is None:
        return observed, (dict(observed) if replaces else {**prior, **observed}), kept
    written = {item: observed[item] for item in added}
    return written, {**prior, **written}, kept


def closed_years(digests: Mapping[int, str], scope_end: date) -> dict[int, str]:
    """The digests of the years before `scope_end`'s, which a refresh renews as their facts."""
    return {year: digest for year, digest in digests.items() if year < scope_end.year}
