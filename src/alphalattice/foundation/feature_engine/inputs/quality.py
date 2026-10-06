"""Deterministic evidence preparation for the Feature Input Gateway.

This owner reads already-sanitized facts and emits a bounded evidence summary.
It never repairs, clips, interpolates, or persists a market observation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from math import isfinite

import numpy as np

from alphalattice.foundation.feature_engine.contracts import canonical_hash
from alphalattice.foundation.feature_engine.inputs.contracts import MaterializedFeatureQualification
from alphalattice.foundation.feature_engine.inputs.gateway import (
    ListingQualityEvidence,
    UnexplainedRawMove,
)
from alphalattice.foundation.feature_engine.storage.contracts import FeatureIneligibilityRun
from alphalattice.foundation.market_data_ops.sources.universe import (
    ORIGINAL_RESEARCH_WHITELIST_STANDARD,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import CurrentUniverseQualityAdmission


def qualify_materialized_features(
    *,
    session: date,
    catalog_hash: str,
    factor_ids: Sequence[str],
    listing_ids: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    ineligibility: Sequence[FeatureIneligibilityRun],
    factor_catalogs: Mapping[str, str] | None = None,
) -> MaterializedFeatureQualification:
    """Reuse stored values/reasons; never recompute, fill or infer failed formulas.

    A missing materialized row is pending, not a corrupt stock. An undefined
    selected value needs its matching catalog/date ineligibility evidence; a
    missing reason is an integrity defect, not permission to exclude the name.
    A layered catalog's evidence is recorded by the part that computes each
    factor (``factor_catalogs``, V92).
    """
    factors, listings = tuple(sorted(set(factor_ids))), tuple(sorted(set(listing_ids)))
    if not factors or not listings:
        raise ValueError("feature.qualification_scope_empty")
    by_listing: dict[str, Mapping[str, object]] = {}
    for row in rows:
        listing = str(row["listing_id"])
        if (
            listing not in listings
            or listing in by_listing
            or row["session_date"] != session
            or row["catalog_hash"] != catalog_hash
            or not set(factors).issubset(row)
        ):
            raise ValueError("feature.qualification_scope_mismatch")
        by_listing[listing] = row
    reasons = {
        (run.listing_id, run.factor_id): run.reason
        for run in ineligibility
        if run.catalog_hash == (factor_catalogs or {}).get(run.factor_id, catalog_hash)
        and run.first_session <= session <= run.last_session
    }
    eligible: list[str] = []
    pending: list[str] = []
    exclusions: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for listing in listings:
        row = by_listing.get(listing)
        if row is None:
            pending.append(listing)
            continue
        missing = tuple(
            factor
            for factor in factors
            if row[factor] is None or not isfinite(float(row[factor]))  # type: ignore[arg-type]
        )
        if any((listing, factor) not in reasons for factor in missing):
            raise ValueError("feature.qualification_unavailability_undocumented")
        if missing:
            exclusions.append(
                (listing, tuple((factor, reasons[listing, factor]) for factor in missing))
            )
        else:
            eligible.append(listing)
    return MaterializedFeatureQualification(
        session,
        catalog_hash,
        factors,
        tuple(eligible),
        tuple(pending),
        tuple(exclusions),
        canonical_hash(
            {
                "session": session,
                "catalog_hash": catalog_hash,
                "factor_ids": factors,
                "rows": [
                    (listing, by_listing[listing]["row_hash"] if listing in by_listing else None)
                    for listing in listings
                ],
                "exclusions": exclusions,
            }
        ),
    )


@dataclass(frozen=True, eq=False)
class RawCloseSeries:
    """One listing's dated, finite positive raw closes used for quality evaluation, as columns.

    ``sessions`` are days (``datetime64[D]``), ``closes`` the raw closes (``float64``), one per
    session in the order given. Governance reads every listing's closes on each of its passes;
    an object per session cost a first use's governance 1.9 s a pass to build (1.25M of them),
    so the series is held by column and the rule a point once kept is kept here for the whole
    series at once.
    """

    sessions: np.ndarray
    closes: np.ndarray

    def __post_init__(self) -> None:
        """Require one finite positive raw close per dated session."""
        if (
            self.sessions.dtype != np.dtype("datetime64[D]")
            or self.closes.dtype != np.dtype(np.float64)
            or len(self.sessions) != len(self.closes)
        ):
            raise ValueError("feature-input raw closes must pair each session with one close")
        if not bool(np.all(np.isfinite(self.closes) & (self.closes > 0))):
            raise ValueError("feature-input raw close must be finite and positive")

    @classmethod
    def of(cls, points: Iterable[tuple[date, float]]) -> RawCloseSeries:
        """The series of ``(session, close)`` points, in the order given."""
        pairs = tuple(points)
        return cls(
            np.array([session for session, _close in pairs], dtype="datetime64[D]"),
            np.array([close for _session, close in pairs], dtype=np.float64),
        )

    def __len__(self) -> int:
        """How many sessions the series holds."""
        return len(self.closes)

    def session(self, index: int) -> date:
        """The session at `index`, as a date."""
        value = self.sessions[index].item()
        assert isinstance(value, date)
        return value


@dataclass(frozen=True)
class FeatureInputQualityPolicy:
    """Thresholds for raw-input quality and anomaly evidence."""

    maximum_missing_ratio: float = ORIGINAL_RESEARCH_WHITELIST_STANDARD.maximum_missing_ratio
    maximum_consecutive_gap: int = (
        ORIGINAL_RESEARCH_WHITELIST_STANDARD.maximum_consecutive_missing_sessions
    )
    maximum_adjusted_diagnostic_bps: float = 5.0
    unexplained_raw_move_fraction: float = 0.50

    @property
    def policy_hash(self) -> str:
        """Hash the thresholds that governed this quality assessment."""
        return canonical_hash(
            {
                "maximum_missing_ratio": self.maximum_missing_ratio,
                "maximum_consecutive_gap": self.maximum_consecutive_gap,
                "maximum_adjusted_diagnostic_bps": self.maximum_adjusted_diagnostic_bps,
                "unexplained_raw_move_fraction": self.unexplained_raw_move_fraction,
            }
        )


@dataclass(frozen=True)
class ExtremeMoveSummary:
    """The moves above the policy fraction, and two identities over them.

    ``summary_hash`` is the sessions and the largest move, part of the
    evidence identity since the first evaluator. ``anomaly_signature_hash``
    is the unexplained moves with the two observations each is made of, the
    explained sessions and the policy -- no range, no other close -- so the
    same anomaly reads the same on every later day and a revised observation
    or a new session reads differently.
    """

    unexplained_sessions: tuple[date, ...]
    action_explained_sessions: tuple[date, ...]
    largest_absolute_move: float
    summary_hash: str
    unexplained_moves: tuple[UnexplainedRawMove, ...] = ()
    anomaly_signature_hash: str | None = None


class FeatureInputQualityEvaluator:
    """Convert quality/audit facts into one safe per-listing Gateway input."""

    def __init__(self, policy: FeatureInputQualityPolicy | None = None) -> None:
        """Install the input-quality policy used to prepare evidence."""
        self.policy = policy or FeatureInputQualityPolicy()

    def evaluate(
        self,
        *,
        admission: CurrentUniverseQualityAdmission,
        provider: str,
        range_start: date,
        range_end: date,
        raw_closes: RawCloseSeries,
        corporate_action_sessions: frozenset[date],
        action_audit_completed: bool,
        adjusted_diagnostic_max_bps: float | None,
        identity_verified: bool,
        retry_exhausted: bool,
        provider_correction_observed: bool = False,
        action_evidence_hash: str | None = None,
    ) -> ListingQualityEvidence:
        """Evaluate sanitized market facts into one listing's gateway evidence."""
        if range_start > range_end:
            raise ValueError("feature-input evaluation range is reversed")
        if not len(raw_closes):
            raise ValueError("feature-input evaluation requires raw closes")
        if action_evidence_hash is not None and (
            len(action_evidence_hash) != 64
            or any(c not in "0123456789abcdef" for c in action_evidence_hash)
        ):
            raise ValueError("feature-input action evidence hash is invalid")
        # Unique and ordered is strictly increasing.
        if not bool(np.all(raw_closes.sessions[1:] > raw_closes.sessions[:-1])):
            raise ValueError("feature-input raw closes must be unique and ordered")
        if raw_closes.session(0) < range_start or raw_closes.session(-1) > range_end:
            raise ValueError("feature-input raw closes exceed the evaluated range")
        extreme = self._extreme_moves(raw_closes, corporate_action_sessions)
        reasons: set[str] = set()
        if (
            not admission.eligible
            or admission.missing_ratio > self.policy.maximum_missing_ratio
            or admission.maximum_consecutive_gap > self.policy.maximum_consecutive_gap
        ):
            reasons.add("CURRENT_UNIVERSE_QUALITY_FAILED")
        if not identity_verified:
            reasons.add("IDENTITY_UNVERIFIED")
        if not action_audit_completed:
            reasons.add("ACTION_AUDIT_INCOMPLETE")
        if adjusted_diagnostic_max_bps is None or not isfinite(adjusted_diagnostic_max_bps):
            reasons.add("ADJUSTED_DIAGNOSTIC_UNAVAILABLE")
        elif adjusted_diagnostic_max_bps > self.policy.maximum_adjusted_diagnostic_bps:
            reasons.add("ADJUSTED_CLOSE_DIAGNOSTIC_MISMATCH")
        if extreme.unexplained_sessions:
            reasons.add("UNEXPLAINED_RAW_MOVE")
        failure_code = self._failure_code(reasons)
        identity = {
            "listing_id": admission.listing_id,
            "provider": provider,
            "range": [range_start, range_end],
            "quality": {
                "eligible": admission.eligible,
                "missing_ratio": admission.missing_ratio,
                "maximum_consecutive_gap": admission.maximum_consecutive_gap,
                "reasons": admission.reasons,
            },
            # ``str(date)`` is what the canonical encoder writes for a date;
            # spelled here, for the whole series at once, so the encoder stays
            # in C over ~2,500 points.
            "raw_close_hash": canonical_hash(
                list(
                    zip(
                        np.datetime_as_string(raw_closes.sessions, unit="D").tolist(),
                        raw_closes.closes.tolist(),
                        strict=True,
                    )
                )
            ),
            "extreme_move_summary_hash": extreme.summary_hash,
            "action_audit_completed": action_audit_completed,
            "adjusted_diagnostic_max_bps": adjusted_diagnostic_max_bps,
            "identity_verified": identity_verified,
            "provider_correction_observed": provider_correction_observed,
            "policy_hash": self.policy.policy_hash,
        }
        if action_evidence_hash is not None:
            identity["action_evidence_hash"] = action_evidence_hash
        evidence_hash = canonical_hash(identity)
        return ListingQualityEvidence(
            listing_id=admission.listing_id,
            provider=provider,
            range_start=range_start,
            range_end=range_end,
            evidence_hash=evidence_hash,
            failure_code=failure_code,
            reason_codes=tuple(sorted(reasons)),
            retry_exhausted=retry_exhausted if failure_code is not None else False,
            extreme_move_unexplained=bool(extreme.unexplained_sessions),
            provider_correction_observed=provider_correction_observed,
            action_explained=bool(extreme.action_explained_sessions),
            identity_verified=identity_verified,
            unexplained_sessions=extreme.unexplained_sessions,
            largest_absolute_move=extreme.largest_absolute_move,
            unexplained_moves=extreme.unexplained_moves,
            anomaly_signature_hash=extreme.anomaly_signature_hash,
            qualification_receipt_hash=(
                canonical_hash(
                    [admission.onboarding_id, admission.listing_id, evidence_hash, "QUALIFIED"]
                )
                if failure_code is None
                else None
            ),
        )

    def _extreme_moves(
        self,
        raw_closes: RawCloseSeries,
        corporate_action_sessions: frozenset[date],
    ) -> ExtremeMoveSummary:
        unexplained: list[UnexplainedRawMove] = []
        explained: list[date] = []
        # Each session's move from the one before, the same division and subtraction a
        # float per pair made; only the moves past the threshold are then visited.
        closes = raw_closes.closes
        moves = np.abs(closes[1:] / closes[:-1] - 1.0)
        largest = float(moves.max()) if len(moves) else 0.0
        for index in np.flatnonzero(moves > self.policy.unexplained_raw_move_fraction):
            session = raw_closes.session(int(index) + 1)
            if session in corporate_action_sessions:
                explained.append(session)
            else:
                unexplained.append(
                    UnexplainedRawMove(
                        previous_session=raw_closes.session(int(index)),
                        previous_close=float(closes[index]),
                        session=session,
                        close=float(closes[index + 1]),
                    )
                )
        sessions = tuple(item.session for item in unexplained)
        payload = {
            "unexplained_sessions": list(sessions),
            "action_explained_sessions": explained,
            "largest_absolute_move": largest,
            "policy_hash": self.policy.policy_hash,
        }
        signature = {
            "unexplained_moves": [
                [item.previous_session, item.previous_close, item.session, item.close]
                for item in unexplained
            ],
            "action_explained_sessions": explained,
            "policy_hash": self.policy.policy_hash,
        }
        return ExtremeMoveSummary(
            unexplained_sessions=sessions,
            action_explained_sessions=tuple(explained),
            largest_absolute_move=largest,
            summary_hash=canonical_hash(payload),
            unexplained_moves=tuple(unexplained),
            anomaly_signature_hash=canonical_hash(signature),
        )

    @staticmethod
    def _failure_code(reasons: set[str]) -> str | None:
        for reason, code in (
            ("IDENTITY_UNVERIFIED", "data.identity_unverified"),
            ("ACTION_AUDIT_INCOMPLETE", "data.action_audit_incomplete"),
            ("ADJUSTED_CLOSE_DIAGNOSTIC_MISMATCH", "data.adjusted_close_mismatch"),
            ("ADJUSTED_DIAGNOSTIC_UNAVAILABLE", "data.adjusted_diagnostic_unavailable"),
            ("UNEXPLAINED_RAW_MOVE", "data.unexplained_raw_move"),
            ("CURRENT_UNIVERSE_QUALITY_FAILED", "data.current_universe_quality_failed"),
        ):
            if reason in reasons:
                return code
        return None
