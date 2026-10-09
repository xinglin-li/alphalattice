"""Governed quality admission between mutable data work and Factor Research.

The Gateway owns decisions, not calculations.  Raw observations, corporate
actions, base features, and panel math remain with their existing deterministic
owners.  This module turns their bounded evidence into one immutable research
universe admission, a provider-wide pause, or a token-scoped Data Engineer
case.  It never edits a value in order to make data pass.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from math import ceil
from typing import TYPE_CHECKING, Literal, cast

from pydantic import TypeAdapter

from alphalattice.foundation.feature_engine.catalog.contracts import desktop_core_feature_bundle
from alphalattice.foundation.feature_engine.contracts import (
    TemporalKnowledgeBoundary,
    canonical_hash,
)
from alphalattice.foundation.market_data_ops.runtime.remediation import (
    EscalateForHumanArgs,
    ExcludeFromNextManifestArgs,
    PolicyDisposition,
    QuarantineListingArgs,
    RefreshAdjustmentDiagnosticArgs,
    RemediationAction,
    RemediationOption,
    RetainRawValueWithCaveatArgs,
    RetryPrimaryArgs,
    UseLastKnownGoodArgs,
    VerifiedAliasArgs,
    WaitThenRetryArgs,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    UniverseManifest,
    build_quality_filtered_research_manifest,
    qualification_obligation,
)
from alphalattice.foundation.market_data_ops.sources.membership import (
    journal_members,
    membership_identity,
)

from .contracts import (
    FEATURE_QUALIFICATION_DOMAINS,
    ListingQuarantine,
    MaterializedFeatureQualification,
)

if TYPE_CHECKING:
    from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
    from alphalattice.foundation.feature_engine.storage.contracts import SectorReferenceState
    from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
    from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository


class FeatureInputAssessment(StrEnum):
    """Fail-closed outcomes of a Feature input assessment."""

    ADMITTED = "admitted"
    DEFERRED = "deferred"
    QUARANTINED = "quarantined"
    PROVIDER_COHORT_DEFERRED = "provider_cohort_deferred"
    DATA_TRUTH_REVIEW = "data_truth_review"


@dataclass(frozen=True)
class FeatureInputPolicy:
    """Hash-bound operational policy; thresholds are explicit, not statistical truth."""

    ordinary_fraction: float = 0.10
    ordinary_floor: int = 10
    provider_macro_fraction: float = 0.15
    provider_macro_floor: int = 25
    sector_macro_fraction: float = 0.35
    sector_macro_floor: int = 5
    provider_retry_delay_seconds: int = 300
    minimum_sector_size: int = 5
    minimum_panel_coverage: float = 0.98
    policy_hash: str = ""

    def __post_init__(self) -> None:
        """Validate thresholds and seal the operational policy hash."""
        if not 0 < self.ordinary_fraction < self.provider_macro_fraction < 1:
            raise ValueError("ordinary and provider macro fractions are invalid")
        if not 0 < self.sector_macro_fraction < 1:
            raise ValueError("sector macro fraction is invalid")
        if self.provider_retry_delay_seconds < 1:
            raise ValueError("provider retry delay must be positive")
        payload = asdict(self)
        payload.pop("policy_hash")
        expected = canonical_hash(payload)
        if self.policy_hash and self.policy_hash != expected:
            raise ValueError("feature-input policy hash does not match its values")
        object.__setattr__(self, "policy_hash", expected)

    def ordinary_threshold(self, active_count: int) -> int:
        """Return the ordinary incident threshold for an active cohort."""
        return max(self.ordinary_floor, ceil(active_count * self.ordinary_fraction))

    def provider_macro_threshold(self, active_count: int) -> int:
        """Return the provider-wide incident threshold for an active cohort."""
        return max(self.provider_macro_floor, ceil(active_count * self.provider_macro_fraction))

    def sector_macro_threshold(self, sector_count: int) -> int:
        """Return the incident threshold for one sector cohort."""
        return max(self.sector_macro_floor, ceil(sector_count * self.sector_macro_fraction))


@dataclass(frozen=True)
class UnexplainedRawMove:
    """One raw move above the policy fraction with no corporate action on its session.

    The two observations that make the move are the anomaly's substance: a
    revision of either is a different anomaly, a new session after them is
    not.
    """

    previous_session: date
    previous_close: float
    session: date
    close: float

    def __post_init__(self) -> None:
        """Require the previous observation to precede the move session."""
        if self.previous_session >= self.session:
            raise ValueError("unexplained raw move sessions are not ordered")


@dataclass(frozen=True)
class ListingQualityEvidence:
    """Safe deterministic facts for one listing; no raw matrix crosses this boundary.

    ``evidence_hash`` is the identity of everything observed, and it moves
    with every appended session. ``anomaly_signature_hash`` is the identity
    of the anomaly alone (the unexplained moves with their observations, the
    action-explained sessions and the evaluator policy), so an unchanged
    historical anomaly reads the same across windows; it is a comparison
    basis for a standing decision, never an admission identity.
    """

    listing_id: str
    provider: str
    range_start: date
    range_end: date
    evidence_hash: str
    failure_code: str | None = None
    reason_codes: tuple[str, ...] = ()
    retry_exhausted: bool = False
    extreme_move_unexplained: bool = False
    provider_correction_observed: bool = False
    action_explained: bool = False
    identity_verified: bool = True
    qualification_receipt_hash: str | None = None
    verified_alias_candidate_id: str | None = None
    verified_alias_provider_symbol: str | None = None
    verified_alias_evidence_hash: str | None = None
    unexplained_sessions: tuple[date, ...] = ()
    largest_absolute_move: float | None = None
    caveat_receipt_hash: str | None = None
    unexplained_moves: tuple[UnexplainedRawMove, ...] = ()
    anomaly_signature_hash: str | None = None

    def __post_init__(self) -> None:
        """Validate the evidence range, identities and failure facts."""
        if self.range_start > self.range_end:
            raise ValueError("quality evidence range is reversed")
        if len(self.evidence_hash) != 64:
            raise ValueError("quality evidence requires a SHA-256 identity")
        if self.anomaly_signature_hash is not None and len(self.anomaly_signature_hash) != 64:
            raise ValueError("anomaly signature requires a SHA-256 identity")
        if self.unexplained_moves and (
            tuple(item.session for item in self.unexplained_moves) != self.unexplained_sessions
        ):
            raise ValueError("unexplained moves must name the unexplained sessions")
        if self.failure_code is None and self.reason_codes:
            raise ValueError("admitted evidence cannot contain failure reasons")
        if self.failure_code is not None and not self.reason_codes:
            raise ValueError("failed evidence requires at least one reason code")
        alias_fields = (
            self.verified_alias_candidate_id,
            self.verified_alias_provider_symbol,
            self.verified_alias_evidence_hash,
        )
        if any(alias_fields) and not all(alias_fields):
            raise ValueError("verified alias evidence must be complete")
        if (
            self.verified_alias_evidence_hash is not None
            and len(self.verified_alias_evidence_hash) != 64
        ):
            raise ValueError("verified alias evidence requires a SHA-256 identity")

    @property
    def admitted(self) -> bool:
        """Report whether this listing has passing qualification evidence."""
        return self.failure_code is None


@dataclass(frozen=True)
class ListingEligibilityDecision:
    """Hash-bound eligibility judgment for one listing's input evidence."""

    listing_id: str
    assessment: FeatureInputAssessment
    reason_codes: tuple[str, ...]
    evidence_hash: str
    valid_until: datetime | None
    requalification_conditions: tuple[str, ...]
    decision_hash: str

    @classmethod
    def create(
        cls,
        *,
        listing_id: str,
        assessment: FeatureInputAssessment,
        reason_codes: tuple[str, ...],
        evidence_hash: str,
        valid_until: datetime | None = None,
        requalification_conditions: tuple[str, ...] = (),
    ) -> ListingEligibilityDecision:
        """Seal one listing's assessment and requalification conditions."""
        payload = {
            "listing_id": listing_id,
            "assessment": assessment.value,
            "reason_codes": sorted(set(reason_codes)),
            "evidence_hash": evidence_hash,
            "valid_until": valid_until.isoformat() if valid_until else None,
            "requalification_conditions": sorted(set(requalification_conditions)),
        }
        return cls(
            listing_id=listing_id,
            assessment=assessment,
            reason_codes=tuple(payload["reason_codes"]),
            evidence_hash=evidence_hash,
            valid_until=valid_until,
            requalification_conditions=tuple(payload["requalification_conditions"]),
            decision_hash=canonical_hash(payload),
        )


@dataclass(frozen=True)
class PanelImpactProjection:
    """Host-owned preflight of a proposed next-manifest listing set."""

    sector_distribution: Mapping[str, int]
    panel_coverage: float
    available_factor_count: int
    lineage_compatible: bool
    projection_hash: str = ""

    def __post_init__(self) -> None:
        """Seal or verify projected Panel coverage and sector distribution."""
        payload = {
            "sector_distribution": dict(sorted(self.sector_distribution.items())),
            "panel_coverage": self.panel_coverage,
            "available_factor_count": self.available_factor_count,
            "lineage_compatible": self.lineage_compatible,
        }
        expected = canonical_hash(payload)
        if self.projection_hash and self.projection_hash != expected:
            raise ValueError("panel impact projection hash does not match its values")
        object.__setattr__(self, "projection_hash", expected)

    def blocking_reasons(self, policy: FeatureInputPolicy) -> tuple[str, ...]:
        """Return policy failures that prevent projected Panel use."""
        reasons: set[str] = set()
        if any(count < policy.minimum_sector_size for count in self.sector_distribution.values()):
            reasons.add("PANEL_SECTOR_BELOW_MINIMUM")
        if self.panel_coverage < policy.minimum_panel_coverage:
            reasons.add("PANEL_COVERAGE_BELOW_MINIMUM")
        if self.available_factor_count < 1:
            reasons.add("PANEL_ALL_FACTORS_UNAVAILABLE")
        if not self.lineage_compatible:
            reasons.add("PANEL_BINDING_MISMATCH")
        return tuple(sorted(reasons))


@dataclass(frozen=True)
class FeatureInputAdmission:
    """Sealed cohort admission, quarantine counts and temporal scope."""

    candidate_manifest_revision: str
    admitted_listing_ids: tuple[str, ...]
    quarantined_listing_ids: tuple[str, ...]
    quarantine_reason_counts: Mapping[str, int]
    quality_policy_hash: str
    temporal_identity_hash: str
    knowledge_cutoff_at: datetime
    admission_hash: str
    evidence_scope: Literal[
        "FULL_QUALITY", "SECTOR_ONLY", "BASE_FEATURES_ONLY", "BASE_FEATURE_CANDIDATES"
    ] = "FULL_QUALITY"
    caveat_receipts: tuple[tuple[str, str], ...] = ()
    feature_qualification_hash: str | None = None
    nominal_membership_hash: str | None = None

    @classmethod
    def create(
        cls,
        *,
        candidate_manifest_revision: str,
        admitted_listing_ids: tuple[str, ...],
        quarantines: tuple[ListingQuarantine, ...],
        quality_policy_hash: str,
        temporal_boundary: TemporalKnowledgeBoundary,
        evidence_scope: Literal[
            "FULL_QUALITY", "SECTOR_ONLY", "BASE_FEATURES_ONLY", "BASE_FEATURE_CANDIDATES"
        ] = "FULL_QUALITY",
        caveat_receipts: tuple[tuple[str, str], ...] = (),
        feature_qualification_hash: str | None = None,
        nominal_membership_hash: str | None = None,
    ) -> FeatureInputAdmission:
        """Seal admitted listings and quarantine counts under a temporal boundary."""
        if (
            evidence_scope in {"BASE_FEATURES_ONLY", "BASE_FEATURE_CANDIDATES"}
            and feature_qualification_hash is None
        ):
            raise ValueError("base Feature admission requires materialized qualification evidence")
        if feature_qualification_hash is not None and (
            len(feature_qualification_hash) != 64
            or any(value not in "0123456789abcdef" for value in feature_qualification_hash)
        ):
            raise ValueError("base Feature qualification identity is invalid")
        counts: dict[str, int] = defaultdict(int)
        for quarantine in quarantines:
            for reason in quarantine.reason_codes:
                counts[reason] += 1
        payload = {
            "candidate_manifest_revision": candidate_manifest_revision,
            "admitted_listing_ids": sorted(set(admitted_listing_ids)),
            "quarantined_listing_ids": sorted({item.listing_id for item in quarantines}),
            "quarantine_reason_counts": dict(sorted(counts.items())),
            "quality_policy_hash": quality_policy_hash,
            "temporal_identity_hash": temporal_boundary.identity_hash(),
            "knowledge_cutoff_at": temporal_boundary.knowledge_cutoff_at.isoformat(),
        }
        if evidence_scope != "FULL_QUALITY":
            payload["evidence_scope"] = evidence_scope
        if caveat_receipts:
            payload["caveat_receipts"] = sorted(caveat_receipts)
        if feature_qualification_hash is not None:
            payload["feature_qualification_hash"] = feature_qualification_hash
        if nominal_membership_hash is not None:
            payload["nominal_membership_hash"] = nominal_membership_hash
        return cls(
            candidate_manifest_revision=candidate_manifest_revision,
            admitted_listing_ids=tuple(payload["admitted_listing_ids"]),
            quarantined_listing_ids=tuple(payload["quarantined_listing_ids"]),
            quarantine_reason_counts=dict(payload["quarantine_reason_counts"]),
            quality_policy_hash=quality_policy_hash,
            temporal_identity_hash=temporal_boundary.identity_hash(),
            knowledge_cutoff_at=temporal_boundary.knowledge_cutoff_at,
            admission_hash=canonical_hash(payload),
            evidence_scope=evidence_scope,
            caveat_receipts=tuple(sorted(caveat_receipts)),
            feature_qualification_hash=feature_qualification_hash,
            nominal_membership_hash=nominal_membership_hash,
        )


@dataclass(frozen=True)
class ProviderCohortDeferred:
    """Provider-wide retry state with affected listings and transport policy."""

    deferred_retry_id: str
    provider: str
    failure_code: str
    affected_listing_ids: tuple[str, ...]
    retry_after_at: datetime
    observed_workers: int
    next_workers: int
    evidence_hash: str
    policy_hash: str


@dataclass(frozen=True)
class FeatureInputAgentCase:
    """Bounded diagnosis case and immutable remediation options for an agent."""

    run_id: str
    case_token: str
    case_kind: str
    manifest_revision: str
    listing_ids: tuple[str, ...]
    failure_code: str
    evidence_hash: str
    rediagnosis_count: int
    tool_names: tuple[str, ...]
    options: tuple[RemediationOption, ...]
    policy_hash: str = ""
    evidence: tuple[ListingQualityEvidence, ...] = ()

    def __post_init__(self) -> None:
        """Verify the case token against its options and policy when durable."""
        if not self.policy_hash:  # retained in-memory callers; no durable resolution authority
            return
        expected = canonical_hash(
            [
                self.manifest_revision,
                self.case_kind,
                self.failure_code,
                self.evidence_hash,
                tuple(item.option_hash for item in self.options),
                self.policy_hash,
            ]
        )
        scope = tuple(sorted(self.evidence, key=lambda item: item.listing_id))
        if (
            self.case_token != expected
            or self.run_id != f"feature-input:{expected[:24]}"
            or tuple(item.listing_id for item in scope) != self.listing_ids
            or canonical_hash([(item.listing_id, item.evidence_hash) for item in scope])
            != self.evidence_hash
        ):
            raise ValueError("feature_input.case_identity_mismatch")

    @property
    def issue_hash(self) -> str:
        """Same evidence/options apart from the rolling quarantine deadline.

        This groups a pending problem, not an approval. A decision still names
        its complete immutable case/option token, including that deadline.
        """
        options = []
        for option in self.options:
            payload = option.model_dump(mode="json", exclude={"option_hash"})
            payload["policy_args"].pop("recheck_after_at", None)
            options.append(payload)
        return str(
            canonical_hash(
                [
                    self.manifest_revision,
                    self.case_kind,
                    self.failure_code,
                    self.listing_ids,
                    self.evidence_hash,
                    self.policy_hash,
                    options,
                ]
            )
        )

    def document(self) -> dict[str, object]:
        """Serialize a bounded agent case as a safe JSON-compatible document."""
        return cast(
            dict[str, object], TypeAdapter(FeatureInputAgentCase).dump_python(self, mode="json")
        )

    @classmethod
    def read_document(cls, document: str) -> FeatureInputAgentCase:
        """Read a durable case only when resolution authority is present."""
        case: FeatureInputAgentCase = TypeAdapter(cls).validate_json(document)
        if not case.policy_hash or not case.evidence:
            raise ValueError("feature_input.case_resolution_authority_absent")
        return case

    def options_current_at(self, observed_at: datetime) -> bool:
        """Check whether the offered options remain valid at observation time."""
        return all(
            not isinstance(option.policy_args, QuarantineListingArgs | ExcludeFromNextManifestArgs)
            or option.policy_args.recheck_after_at is None
            or option.policy_args.recheck_after_at > observed_at
            for option in self.options
        )


QUARANTINE_CONTINUATION_RULE = "recoverable_quarantine.unchanged_unexplained_move.v1"
CONTINUABLE_OPTION_IDS: frozenset[str] = frozenset({"recoverable_quarantine"})
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class QuarantineContinuation:
    """A standing quarantine carried to its next recheck over re-examined evidence.

    Not a new decision: the authority is the original execution receipt, the
    basis is the original case's frozen evidence, ``rule`` names what was
    compared, ``evidence`` is what the recheck examined, and ``quarantine`` is
    the continued row, chained to the row it continues and due again after
    ``recheck_after_at``. Reads back as original decision, new evidence, rule,
    disposition.
    """

    rule: str
    listing_id: str
    original_case_token: str
    original_option_id: str
    original_execution_receipt_hash: str
    original_evidence_hash: str
    continued_from_quarantine_hash: str
    evidence: ListingQualityEvidence
    observed_at: datetime
    recheck_after_at: datetime
    quarantine: ListingQuarantine
    continuation_hash: str = ""

    def __post_init__(self) -> None:
        # Creation seals the identity; an identity given must match it. A
        # persisted record is read only through ``read_document``, which
        # requires the identity to be present -- nothing is sealed on read.
        """Seal and cross-check a carried quarantine against its evidence."""
        expected = canonical_hash(self._identity())
        if self.continuation_hash and self.continuation_hash != expected:
            raise ValueError("feature_input.continuation_identity_mismatch")
        object.__setattr__(self, "continuation_hash", expected)
        # The record, its row and its evidence describe one disposition: the
        # row continues the row named, under the receipt named, over this
        # listing and exactly the evidence the recheck examined, due when the
        # record says; and the row's own identity recomputes from its fields.
        row, evidence = self.quarantine, self.evidence
        if (
            self.rule != QUARANTINE_CONTINUATION_RULE
            or row.continued_from_quarantine_hash != self.continued_from_quarantine_hash
            or row.execution_receipt_hash != self.original_execution_receipt_hash
            or row.listing_id != self.listing_id
            or evidence.listing_id != self.listing_id
            or row.evidence_hash != evidence.evidence_hash
            or row.reason_codes != evidence.reason_codes
            or row.recheck_after_at != self.recheck_after_at
            or evidence.admitted
            or self.continued_from_quarantine_hash == row.quarantine_hash
            or row
            != ListingQuarantine.create(
                listing_id=row.listing_id,
                reason_codes=row.reason_codes,
                evidence_hash=row.evidence_hash,
                execution_receipt_hash=row.execution_receipt_hash,
                recheck_after_at=row.recheck_after_at,
                agent_proposal_hash=row.agent_proposal_hash,
                continued_from_quarantine_hash=row.continued_from_quarantine_hash,
            )
        ):
            raise ValueError("feature_input.continuation_identity_mismatch")

    def _identity(self) -> dict[str, object]:
        payload = TypeAdapter(QuarantineContinuation).dump_python(self, mode="json")
        payload.pop("continuation_hash")
        return cast(dict[str, object], payload)

    def document(self) -> dict[str, object]:
        """Serialize the sealed quarantine continuation for storage."""
        return cast(
            dict[str, object], TypeAdapter(QuarantineContinuation).dump_python(self, mode="json")
        )

    @classmethod
    def read_document(cls, document: str) -> QuarantineContinuation:
        """Read a persisted record whose identity is present, well-formed and matching.

        A document without its ``continuation_hash`` (absent, empty or not a
        SHA-256) is refused before any value is read; one whose identity does
        not match its content is refused by the construction. The reader
        never supplies an identity a document lacks.
        """
        payload = json.loads(document)
        given = payload.get("continuation_hash") if isinstance(payload, dict) else None
        if not isinstance(given, str) or _SHA256_HEX.fullmatch(given) is None:
            raise ValueError("feature_input.continuation_identity_absent")
        record: QuarantineContinuation = TypeAdapter(cls).validate_json(document)
        if record.continuation_hash != given:
            raise ValueError("feature_input.continuation_identity_mismatch")
        return record


@dataclass(frozen=True)
class QuarantineContinuationRefusal:
    """Why a standing quarantine is not continued; its listing re-enters the existing path."""

    listing_id: str
    quarantine_hash: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require at least one reason for refusing continuation."""
        if not self.reasons:
            raise ValueError("a continuation refusal names at least one reason")


@dataclass(frozen=True)
class FeatureInputGatewayResult:
    """Assessment, decisions and any admission, quarantine or deferral."""

    assessment: FeatureInputAssessment
    decisions: tuple[ListingEligibilityDecision, ...]
    admission: FeatureInputAdmission | None = None
    research_manifest: UniverseManifest | None = None
    quarantines: tuple[ListingQuarantine, ...] = ()
    deferred: ProviderCohortDeferred | None = None
    agent_cases: tuple[FeatureInputAgentCase, ...] = ()
    failure_reasons: tuple[str, ...] = ()


class FeatureInputExecutionStatus(StrEnum):
    """Deterministic status of an executed Feature input policy option."""

    RETRY_SCHEDULED = "retry_scheduled"
    DEFERRED = "deferred"
    QUARANTINE_READY = "quarantine_ready"
    LAST_KNOWN_GOOD_BOUND = "last_known_good_bound"
    REQUALIFICATION_READY = "requalification_ready"
    DATA_TRUTH_REVIEW = "data_truth_review"
    RAW_VALUE_RETAINED = "raw_value_retained_with_caveat"


@dataclass(frozen=True)
class FeatureInputPolicyExecution:
    """Sealed effect of one validated Feature input remediation option."""

    status: FeatureInputExecutionStatus
    option_id: str
    option_hash: str
    policy_args_hash: str
    execution_receipt_hash: str
    quarantines: tuple[ListingQuarantine, ...] = ()
    retry_after_at: datetime | None = None
    snapshot_ref: str | None = None
    bound_market_as_of_session: date | None = None
    failure_reasons: tuple[str, ...] = ()
    retained_evidence: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class RawRetentionDecisionProof:
    """Verified historical authority to retain one unchanged raw-move finding."""

    case: FeatureInputAgentCase
    receipt_hash: str
    execution_receipt_hash: str
    execution_policy_hash: str
    actor_kind: str
    actor_id: str
    option_hash: str
    retained_evidence: tuple[tuple[str, str], ...]
    receipt_document: str
    effect_document: str
    market_profile_id: str
    membership_fingerprint: str | None
    universe_membership_basis: str
    universe_policy_type: str
    universe_components: tuple[str, ...]
    research_use_class: str
    source_listing_ids: tuple[str, ...]


def matching_raw_retention_proof(
    *,
    case: FeatureInputAgentCase,
    candidate_manifest: UniverseManifest,
    proofs: tuple[RawRetentionDecisionProof, ...],
    policy_hash: str,
) -> RawRetentionDecisionProof | None:
    """Return the unique verified prior choice for this exact case and source membership."""
    if (
        case.case_kind not in {"LISTING", "AGGREGATED"}
        or case.failure_code != "data.unexplained_raw_move"
        or case.policy_hash != policy_hash
    ):
        return None
    current_option = next(
        (
            value
            for value in case.options
            if value.option_id == "retain_isolated_raw_move_with_caveat"
        ),
        None,
    )
    if (
        current_option is None
        or tuple(current_option.target_listing_ids) != case.listing_ids
        or any(
            item.failure_code != "data.unexplained_raw_move"
            or item.reason_codes != ("UNEXPLAINED_RAW_MOVE",)
            for item in case.evidence
        )
    ):
        return None
    current_evidence = tuple(
        sorted((item.listing_id, item.evidence_hash) for item in case.evidence)
    )
    candidates = []
    for proof in proofs:
        prior = proof.case
        option = next(
            (
                value
                for value in prior.options
                if value.option_id == "retain_isolated_raw_move_with_caveat"
            ),
            None,
        )
        if (
            option is None
            or prior.case_kind != case.case_kind
            or prior.failure_code != case.failure_code
            or prior.policy_hash != case.policy_hash
            or prior.listing_ids != case.listing_ids
            or prior.evidence_hash != case.evidence_hash
            or option.option_hash != current_option.option_hash
            or option.option_hash != proof.option_hash
            or tuple(sorted(proof.retained_evidence)) != current_evidence
            or proof.execution_policy_hash != policy_hash
            or proof.actor_kind not in {"HUMAN", "EXTERNAL_AUTOMATION"}
            or not proof.actor_id
            or not proof.receipt_hash
            or not proof.execution_receipt_hash
            or proof.market_profile_id != candidate_manifest.profile.market_profile_id
            or proof.membership_fingerprint is None
            or proof.membership_fingerprint != candidate_manifest.membership_fingerprint
            or proof.universe_membership_basis != candidate_manifest.universe_membership_basis
            or proof.universe_policy_type != candidate_manifest.universe_policy_type
            or proof.universe_components != candidate_manifest.universe_components
            or proof.research_use_class != candidate_manifest.research_use_class
            or not set(case.listing_ids).issubset(proof.source_listing_ids)
            or not set(case.listing_ids).issubset(
                {item.listing_id for item in candidate_manifest.listings}
            )
        ):
            continue
        candidates.append(proof)
    if len({proof.receipt_hash for proof in candidates}) != 1:
        return None
    return candidates[0] if candidates else None


_INSPECTION_TOOLS: tuple[str, ...] = ()


class FeatureInputGateway:
    """Make fail-closed, replayable feature-input admission decisions."""

    def __init__(self, policy: FeatureInputPolicy | None = None) -> None:
        """Install the explicit input-assessment policy."""
        self.policy = policy or FeatureInputPolicy()

    def assess(
        self,
        *,
        candidate_manifest: UniverseManifest,
        evidence: tuple[ListingQualityEvidence, ...],
        sector_by_listing_id: Mapping[str, str],
        temporal_boundary: TemporalKnowledgeBoundary,
        observed_at: datetime,
        panel_impact: PanelImpactProjection | None = None,
        existing_quarantines: tuple[ListingQuarantine, ...] = (),
        macro_retry_exhausted: bool = False,
        observed_workers: int = 4,
        last_known_good_snapshot_ref: str | None = None,
        nominal_membership_hash: str | None = None,
        sector_unknown_listing_ids: frozenset[str] = frozenset(),
    ) -> FeatureInputGatewayResult:
        """Assess one candidate manifest over complete per-listing evidence.

        ``sector_unknown_listing_ids`` declares listings whose Sector evidence
        does not exist yet (a parent bound for a candidate recheck, whose
        entrants get their Sector from the Feature build). Nothing is
        invented for them: they take no part in the sector-macro judgement
        (the provider-wide one still counts them), no case offers an
        exclusion option since the Panel impact of an exclusion cannot be
        projected, and the manifest is not admitted -- only its failures are
        governed.
        """
        if observed_at.tzinfo is None:
            raise ValueError("feature-input observation time must be timezone-aware")
        known = {item.listing_id for item in candidate_manifest.listings}
        if nominal_membership_hash is not None and nominal_membership_hash != membership_identity(
            known
        ):
            raise ValueError("feature-input nominal membership differs from the qualified journal")
        by_listing = {item.listing_id: item for item in evidence}
        if set(by_listing) != known or len(by_listing) != len(evidence):
            raise ValueError("feature-input evidence must cover the candidate manifest exactly")
        if (
            set(sector_by_listing_id) | sector_unknown_listing_ids != known
            or set(sector_by_listing_id) & sector_unknown_listing_ids
        ):
            raise ValueError("sector evidence must cover the candidate manifest exactly")
        if sector_unknown_listing_ids and panel_impact is not None:
            raise ValueError("panel impact cannot be projected without complete sector evidence")
        quarantines = tuple(item for item in existing_quarantines if item.listing_id in known)
        held_quarantine_ids = {
            item.listing_id
            for item in quarantines
            if observed_at.astimezone(UTC) < item.recheck_after_at
        }
        # A recoverable quarantine is an intentional bounded state, not a new
        # failure storm on every daily cycle.  Re-diagnosis becomes eligible
        # only when its deterministic recheck time is due.
        failures = tuple(
            item
            for item in evidence
            if not item.admitted and item.listing_id not in held_quarantine_ids
        )
        macro = self._macro_cohort(
            failures,
            active_count=len(known),
            sector_by_listing_id=sector_by_listing_id,
        )
        if macro is not None:
            provider, failure_code, cohort = macro
            decisions = tuple(
                self._decision(
                    item,
                    FeatureInputAssessment.DEFERRED
                    if item.listing_id in cohort
                    else FeatureInputAssessment.ADMITTED,
                )
                for item in evidence
            )
            cohort_evidence_hash = canonical_hash(
                [
                    (item.listing_id, item.evidence_hash)
                    for item in failures
                    if item.listing_id in cohort
                ]
            )
            if not macro_retry_exhausted:
                retry_after = observed_at.astimezone(UTC) + timedelta(
                    seconds=self.policy.provider_retry_delay_seconds
                )
                deferred = ProviderCohortDeferred(
                    deferred_retry_id=canonical_hash(
                        [candidate_manifest.revision_sha256, cohort_evidence_hash, retry_after]
                    ),
                    provider=provider,
                    failure_code=failure_code,
                    affected_listing_ids=cohort,
                    retry_after_at=retry_after,
                    observed_workers=observed_workers,
                    next_workers=max(1, observed_workers // 2),
                    evidence_hash=cohort_evidence_hash,
                    policy_hash=self.policy.policy_hash,
                )
                return FeatureInputGatewayResult(
                    assessment=FeatureInputAssessment.PROVIDER_COHORT_DEFERRED,
                    decisions=decisions,
                    deferred=deferred,
                )
            case = self._case(
                candidate_manifest=candidate_manifest,
                evidence=tuple(item for item in failures if item.listing_id in cohort),
                case_kind="MARKETWIDE",
                failure_code=failure_code,
                rediagnosis_count=0,
                observed_at=observed_at,
                last_known_good_snapshot_ref=last_known_good_snapshot_ref,
            )
            return FeatureInputGatewayResult(
                assessment=FeatureInputAssessment.DEFERRED,
                decisions=decisions,
                agent_cases=(case,),
            )
        if failures:
            cases = self._ordinary_cases(
                candidate_manifest,
                failures,
                observed_at=observed_at,
                last_known_good_snapshot_ref=last_known_good_snapshot_ref,
                exclusion_options=not sector_unknown_listing_ids,
            )
            decisions = tuple(
                self._decision(
                    item,
                    FeatureInputAssessment.DEFERRED
                    if not item.admitted
                    else FeatureInputAssessment.ADMITTED,
                )
                for item in evidence
            )
            return FeatureInputGatewayResult(
                assessment=FeatureInputAssessment.DEFERRED,
                decisions=decisions,
                agent_cases=cases,
            )
        if sector_unknown_listing_ids:
            # Nothing left to govern and no Sector evidence to admit on: the
            # caller asked for failures only, and there are none outside the
            # standing quarantines.
            return FeatureInputGatewayResult(
                assessment=FeatureInputAssessment.DEFERRED,
                decisions=tuple(
                    self._decision(item, FeatureInputAssessment.DEFERRED) for item in evidence
                ),
                quarantines=quarantines,
                failure_reasons=("SECTOR_EVIDENCE_INCOMPLETE",),
            )
        admitted = tuple(sorted(known - {item.listing_id for item in quarantines}))
        if panel_impact is not None:
            blocking = panel_impact.blocking_reasons(self.policy)
            if blocking:
                return FeatureInputGatewayResult(
                    assessment=FeatureInputAssessment.DEFERRED,
                    decisions=tuple(
                        self._decision(item, FeatureInputAssessment.DEFERRED) for item in evidence
                    ),
                    quarantines=quarantines,
                    failure_reasons=blocking,
                )
        admission = FeatureInputAdmission.create(
            candidate_manifest_revision=candidate_manifest.revision_sha256,
            admitted_listing_ids=admitted,
            quarantines=quarantines,
            quality_policy_hash=self.policy.policy_hash,
            temporal_boundary=temporal_boundary,
            caveat_receipts=tuple(
                (item.listing_id, item.caveat_receipt_hash)
                for item in evidence
                if item.caveat_receipt_hash is not None
            ),
            nominal_membership_hash=nominal_membership_hash,
        )
        # The obligations this membership satisfied, by name: the candidate's
        # own (the history-quality standard) and this Gateway's policy. One
        # spelling for every derivation owner, so a child that re-satisfies
        # an obligation already met is the same manifest, and a child that
        # adds one carries the old ones.
        research_manifest = build_quality_filtered_research_manifest(
            candidate_manifest,
            eligible_listing_ids=tuple(sorted(known))
            if nominal_membership_hash is not None
            else admitted,
            quality_admission_hash=admission.admission_hash,
            qualification_obligations=(
                qualification_obligation("feature_input_policy", self.policy.policy_hash),
            ),
        )
        decisions = tuple(
            ListingEligibilityDecision.create(
                listing_id=item.listing_id,
                assessment=(
                    FeatureInputAssessment.QUARANTINED
                    if item.listing_id in set(admission.quarantined_listing_ids)
                    else FeatureInputAssessment.ADMITTED
                ),
                reason_codes=next(
                    (
                        quarantine.reason_codes
                        for quarantine in quarantines
                        if quarantine.listing_id == item.listing_id
                    ),
                    (),
                ),
                evidence_hash=item.evidence_hash,
                valid_until=next(
                    (
                        quarantine.recheck_after_at
                        for quarantine in quarantines
                        if quarantine.listing_id == item.listing_id
                    ),
                    None,
                ),
                requalification_conditions=(
                    ("fresh_quality_and_action_audit",)
                    if item.listing_id in set(admission.quarantined_listing_ids)
                    else ()
                ),
            )
            for item in evidence
        )
        return FeatureInputGatewayResult(
            assessment=(
                FeatureInputAssessment.QUARANTINED
                if quarantines
                else FeatureInputAssessment.ADMITTED
            ),
            decisions=decisions,
            admission=admission,
            research_manifest=research_manifest,
            quarantines=quarantines,
        )

    def quarantine(
        self,
        *,
        evidence: ListingQualityEvidence,
        recheck_after_at: datetime,
        execution_receipt_hash: str,
        agent_proposal_hash: str | None = None,
    ) -> ListingQuarantine:
        """Create a dated quarantine for nonadmitted listing evidence."""
        if evidence.admitted:
            raise ValueError("admitted listing cannot be quarantined")
        return ListingQuarantine.create(
            listing_id=evidence.listing_id,
            reason_codes=evidence.reason_codes,
            evidence_hash=evidence.evidence_hash,
            agent_proposal_hash=agent_proposal_hash,
            execution_receipt_hash=execution_receipt_hash,
            recheck_after_at=recheck_after_at,
        )

    def continue_quarantine(
        self,
        *,
        quarantine: ListingQuarantine,
        original_case: FeatureInputAgentCase,
        original_option_id: str,
        original_execution_policy_hash: str,
        original_execution: FeatureInputPolicyExecution,
        current: ListingQualityEvidence,
        observed_at: datetime,
    ) -> QuarantineContinuation | QuarantineContinuationRefusal:
        """Carry a due, still-failing quarantine to its next recheck when nothing moved.

        The recheck happened: ``current`` is this cycle's evidence over the
        whole history. It is continued, without a new decision, only when
        the original decision is real and executed (``recoverable_quarantine``,
        its effect sealed under the receipt the quarantine carries, not
        refused), the listing is the same source (identity, provider, alias,
        history start), the anomaly is the same in substance (the unexplained
        moves with their two observations, the action-explained sessions, the
        reasons and failure code -- compared by value, not by the signature
        alone), nothing new was observed (no correction, no new reason), the
        policies are the ones the decision was made under, and the listing
        still fails. A decision recorded before evidence carried the anomaly
        signature cannot be judged and is re-confirmed on the existing path.
        Every refusal names its reasons. Continuing never widens the
        disposition: same listing, same reasons, one more recheck period.
        """
        if observed_at.tzinfo is None:
            raise ValueError("continuation time must be timezone-aware")
        reasons: list[str] = []
        original = next(
            (item for item in original_case.evidence if item.listing_id == quarantine.listing_id),
            None,
        )
        if original is None or quarantine.listing_id not in original_case.listing_ids:
            return QuarantineContinuationRefusal(
                quarantine.listing_id,
                quarantine.quarantine_hash,
                ("ORIGINAL_DECISION_UNAVAILABLE",),
            )
        if (
            original_execution.status is not FeatureInputExecutionStatus.QUARANTINE_READY
            or original_execution.execution_receipt_hash != quarantine.execution_receipt_hash
            or original_execution.failure_reasons
            or original_execution.option_id != original_option_id
            or quarantine.listing_id
            not in {item.listing_id for item in original_execution.quarantines}
        ):
            reasons.append("ORIGINAL_DECISION_NOT_EXECUTED")
        if original_option_id not in CONTINUABLE_OPTION_IDS or not any(
            option.option_id == original_option_id
            and quarantine.listing_id in option.target_listing_ids
            for option in original_case.options
        ):
            reasons.append("ORIGINAL_OPTION_NOT_CONTINUABLE")
        if (
            original_case.policy_hash != self.policy.policy_hash
            or original_execution_policy_hash != self.policy.policy_hash
        ):
            reasons.append("POLICY_CHANGED")
        if observed_at.astimezone(UTC) < quarantine.recheck_after_at:
            reasons.append("QUARANTINE_NOT_DUE")
        if current.listing_id != quarantine.listing_id:
            raise ValueError("continuation listing identity mismatch")
        if current.admitted:
            reasons.append("EVIDENCE_ADMITTED")
        if (
            current.provider != original.provider
            or not current.identity_verified
            or not original.identity_verified
            or current.verified_alias_candidate_id is not None
            or original.verified_alias_candidate_id is not None
        ):
            reasons.append("LISTING_IDENTITY_CHANGED")
        if current.range_start != original.range_start:
            reasons.append("HISTORY_RANGE_START_CHANGED")
        if current.provider_correction_observed:
            reasons.append("PROVIDER_CORRECTION_OBSERVED")
        if current.failure_code != original.failure_code:
            reasons.append("FAILURE_CODE_CHANGED")
        if current.reason_codes != original.reason_codes or current.reason_codes != tuple(
            quarantine.reason_codes
        ):
            reasons.append("REASON_CODES_CHANGED")
        if original.anomaly_signature_hash is None or not original.unexplained_moves:
            reasons.append("ORIGINAL_ANOMALY_SIGNATURE_ABSENT")
        elif current.anomaly_signature_hash is None or not current.unexplained_moves:
            reasons.append("CURRENT_ANOMALY_SIGNATURE_ABSENT")
        elif (
            current.unexplained_moves != original.unexplained_moves
            or current.unexplained_sessions != original.unexplained_sessions
            or current.action_explained != original.action_explained
        ):
            reasons.append("ANOMALY_CHANGED")
        elif current.anomaly_signature_hash != original.anomaly_signature_hash:
            # Same moves, same explanations: only the evaluator policy inside
            # the signature can differ.
            reasons.append("POLICY_CHANGED")
        if reasons:
            return QuarantineContinuationRefusal(
                quarantine.listing_id, quarantine.quarantine_hash, tuple(dict.fromkeys(reasons))
            )
        recheck = self._recheck_after(current.range_end, observed_at)
        continued = ListingQuarantine.create(
            listing_id=quarantine.listing_id,
            reason_codes=current.reason_codes,
            evidence_hash=current.evidence_hash,
            execution_receipt_hash=quarantine.execution_receipt_hash,
            recheck_after_at=recheck,
            agent_proposal_hash=quarantine.agent_proposal_hash,
            continued_from_quarantine_hash=quarantine.quarantine_hash,
        )
        return QuarantineContinuation(
            rule=QUARANTINE_CONTINUATION_RULE,
            listing_id=quarantine.listing_id,
            original_case_token=original_case.case_token,
            original_option_id=original_option_id,
            original_execution_receipt_hash=quarantine.execution_receipt_hash,
            original_evidence_hash=original.evidence_hash,
            continued_from_quarantine_hash=quarantine.quarantine_hash,
            evidence=current,
            observed_at=observed_at.astimezone(UTC),
            recheck_after_at=recheck,
            quarantine=continued,
        )

    @staticmethod
    def _recheck_after(latest: date, observed_at: datetime) -> datetime:
        """Schedule the next recheck after both latest evidence and observation."""
        return max(
            datetime(latest.year, latest.month, latest.day, tzinfo=UTC) + timedelta(days=1),
            observed_at.astimezone(UTC) + timedelta(days=1),
        )

    @staticmethod
    def requalify(
        quarantine: ListingQuarantine,
        evidence: ListingQualityEvidence,
        *,
        observed_at: datetime,
    ) -> ListingEligibilityDecision:
        """Admit a due quarantine only on fresh passing qualification."""
        if observed_at.tzinfo is None:
            raise ValueError("requalification time must be timezone-aware")
        if evidence.listing_id != quarantine.listing_id:
            raise ValueError("requalification listing identity mismatch")
        if observed_at < quarantine.recheck_after_at:
            raise ValueError("listing is not due for requalification")
        if not evidence.admitted or not evidence.qualification_receipt_hash:
            raise ValueError("requalification requires a fresh passing qualification receipt")
        return ListingEligibilityDecision.create(
            listing_id=evidence.listing_id,
            assessment=FeatureInputAssessment.ADMITTED,
            reason_codes=(),
            evidence_hash=evidence.evidence_hash,
        )

    def validate_agent_selection(
        self,
        case: FeatureInputAgentCase,
        *,
        run_id: str,
        case_token: str,
        evidence_hash: str,
        option_id: str,
        current_evidence_hash: str,
        rediagnosis_count: int,
    ) -> RemediationOption:
        """Reject forged, stale or out-of-catalog remediation selections."""
        if (
            case.run_id != run_id
            or case.case_token != case_token
            or case.evidence_hash != evidence_hash
        ):
            raise ValueError("forged Data Engineer feature-input selection")
        if current_evidence_hash != evidence_hash:
            if rediagnosis_count >= 1:
                raise ValueError("feature-input evidence churn requires a deferred outcome")
            raise ValueError("stale Data Engineer feature-input selection")
        option = next((item for item in case.options if item.option_id == option_id), None)
        if option is None:
            raise ValueError("Data Engineer selected an option outside the immutable catalog")
        return option

    def reconcile_stale_case(
        self,
        *,
        candidate_manifest: UniverseManifest,
        prior_case: FeatureInputAgentCase,
        current_evidence: tuple[ListingQualityEvidence, ...],
        observed_at: datetime,
        observed_workers: int = 1,
        last_known_good_snapshot_ref: str | None = None,
    ) -> FeatureInputAgentCase | ProviderCohortDeferred:
        """Allow one fresh diagnosis, then defer evidence churn without another Agent."""
        if candidate_manifest.revision_sha256 != prior_case.manifest_revision:
            raise ValueError("stale case manifest identity changed")
        if tuple(sorted(item.listing_id for item in current_evidence)) != prior_case.listing_ids:
            raise ValueError("stale case listing scope changed")
        current_hash = canonical_hash(
            [
                (item.listing_id, item.evidence_hash)
                for item in sorted(current_evidence, key=lambda item: item.listing_id)
            ]
        )
        if current_hash == prior_case.evidence_hash:
            return prior_case
        if prior_case.rediagnosis_count < 1:
            return self._case(
                candidate_manifest=candidate_manifest,
                evidence=current_evidence,
                case_kind=prior_case.case_kind,
                failure_code=prior_case.failure_code,
                rediagnosis_count=prior_case.rediagnosis_count + 1,
                observed_at=observed_at,
                last_known_good_snapshot_ref=last_known_good_snapshot_ref,
                exclusion_options=any(
                    option.option_id == "recoverable_quarantine" for option in prior_case.options
                ),
            )
        retry_after = observed_at.astimezone(UTC) + timedelta(
            seconds=self.policy.provider_retry_delay_seconds
        )
        return ProviderCohortDeferred(
            deferred_retry_id=canonical_hash(
                [prior_case.case_token, current_hash, "evidence_churn", retry_after]
            ),
            provider=current_evidence[0].provider,
            failure_code="data.evidence_churn",
            affected_listing_ids=prior_case.listing_ids,
            retry_after_at=retry_after,
            observed_workers=observed_workers,
            next_workers=max(1, observed_workers // 2),
            evidence_hash=current_hash,
            policy_hash=self.policy.policy_hash,
        )

    def _macro_cohort(
        self,
        failures: tuple[ListingQualityEvidence, ...],
        *,
        active_count: int,
        sector_by_listing_id: Mapping[str, str],
    ) -> tuple[str, str, tuple[str, ...]] | None:
        grouped: dict[tuple[str, str, date, date], set[str]] = defaultdict(set)
        for item in failures:
            assert item.failure_code is not None
            grouped[(item.provider, item.failure_code, item.range_start, item.range_end)].add(
                item.listing_id
            )
        provider_threshold = self.policy.provider_macro_threshold(active_count)
        for (provider, code, _, _), listing_ids in sorted(grouped.items()):
            if len(listing_ids) >= provider_threshold:
                return provider, code, tuple(sorted(listing_ids))
        sector_sizes: dict[str, int] = defaultdict(int)
        for sector in sector_by_listing_id.values():
            sector_sizes[sector] += 1
        for (provider, code, _, _), listing_ids in sorted(grouped.items()):
            by_sector: dict[str, set[str]] = defaultdict(set)
            for listing_id in listing_ids:
                if listing_id in sector_by_listing_id:  # no sector evidence, no sector judgement
                    by_sector[sector_by_listing_id[listing_id]].add(listing_id)
            for sector, members in sorted(by_sector.items()):
                if len(members) >= self.policy.sector_macro_threshold(sector_sizes[sector]):
                    return provider, code, tuple(sorted(members))
        return None

    def _ordinary_cases(
        self,
        manifest: UniverseManifest,
        failures: tuple[ListingQualityEvidence, ...],
        *,
        observed_at: datetime,
        last_known_good_snapshot_ref: str | None,
        exclusion_options: bool = True,
    ) -> tuple[FeatureInputAgentCase, ...]:
        grouped: dict[tuple[str, str, date, date], list[ListingQualityEvidence]] = defaultdict(list)
        for item in failures:
            if not item.retry_exhausted:
                continue
            assert item.failure_code is not None
            grouped[(item.provider, item.failure_code, item.range_start, item.range_end)].append(
                item
            )
        threshold = self.policy.ordinary_threshold(len(manifest.listings))
        cases: list[FeatureInputAgentCase] = []
        for (_, failure_code, _, _), items in sorted(grouped.items()):
            if len(items) >= threshold:
                cases.append(
                    self._case(
                        candidate_manifest=manifest,
                        evidence=tuple(items),
                        case_kind="AGGREGATED",
                        failure_code=failure_code,
                        rediagnosis_count=0,
                        observed_at=observed_at,
                        last_known_good_snapshot_ref=last_known_good_snapshot_ref,
                        exclusion_options=exclusion_options,
                    )
                )
                continue
            for item in items:
                cases.append(
                    self._case(
                        candidate_manifest=manifest,
                        evidence=(item,),
                        case_kind="LISTING",
                        failure_code=failure_code,
                        rediagnosis_count=0,
                        observed_at=observed_at,
                        last_known_good_snapshot_ref=last_known_good_snapshot_ref,
                        exclusion_options=exclusion_options,
                    )
                )
        return tuple(cases)

    def _case(
        self,
        *,
        candidate_manifest: UniverseManifest,
        evidence: tuple[ListingQualityEvidence, ...],
        case_kind: str,
        failure_code: str,
        rediagnosis_count: int,
        observed_at: datetime,
        last_known_good_snapshot_ref: str | None,
        exclusion_options: bool = True,
    ) -> FeatureInputAgentCase:
        listing_ids = tuple(sorted(item.listing_id for item in evidence))
        evidence_hash = canonical_hash(
            [
                (item.listing_id, item.evidence_hash)
                for item in sorted(evidence, key=lambda x: x.listing_id)
            ]
        )
        options = self._options(
            evidence=evidence,
            case_kind=case_kind,
            observed_at=observed_at,
            last_known_good_snapshot_ref=last_known_good_snapshot_ref,
            exclusion_options=exclusion_options,
        )
        case_token = canonical_hash(
            [
                candidate_manifest.revision_sha256,
                case_kind,
                failure_code,
                evidence_hash,
                tuple(item.option_hash for item in options),
                self.policy.policy_hash,
            ]
        )
        return FeatureInputAgentCase(
            run_id=f"feature-input:{case_token[:24]}",
            case_token=case_token,
            case_kind=case_kind,
            manifest_revision=candidate_manifest.revision_sha256,
            listing_ids=listing_ids,
            failure_code=failure_code,
            evidence_hash=evidence_hash,
            rediagnosis_count=rediagnosis_count,
            tool_names=_INSPECTION_TOOLS,
            options=options,
            policy_hash=self.policy.policy_hash,
            evidence=tuple(sorted(evidence, key=lambda item: item.listing_id)),
        )

    def _options(
        self,
        *,
        evidence: tuple[ListingQualityEvidence, ...],
        case_kind: str,
        observed_at: datetime,
        last_known_good_snapshot_ref: str | None,
        exclusion_options: bool = True,
    ) -> tuple[RemediationOption, ...]:
        first = evidence[0]
        listing_ids = tuple(sorted(item.listing_id for item in evidence))
        options = [
            RemediationOption.create(
                option_id="bounded_full_history_retry",
                target_listing_ids=listing_ids,
                disposition=PolicyDisposition.AUTO,
                policy_args=RetryPrimaryArgs(
                    action=RemediationAction.RETRY_PRIMARY,
                    provider=first.provider,
                    range_start=min(item.range_start for item in evidence),
                    range_end=max(item.range_end for item in evidence),
                    max_attempts=1,
                ),
            ),
            RemediationOption.create(
                option_id="wait_for_provider_recovery",
                target_listing_ids=listing_ids,
                disposition=PolicyDisposition.AUTO,
                policy_args=WaitThenRetryArgs(
                    action=RemediationAction.WAIT_THEN_RETRY,
                    retry_after_seconds=self.policy.provider_retry_delay_seconds,
                    provider=first.provider,
                    range_start=min(item.range_start for item in evidence),
                    range_end=max(item.range_end for item in evidence),
                    max_attempts=1,
                ),
            ),
        ]
        if any(
            "ADJUST" in reason or "ACTION" in reason
            for item in evidence
            for reason in item.reason_codes
        ):
            options.append(
                RemediationOption.create(
                    option_id="refresh_adjustment_diagnostic",
                    target_listing_ids=listing_ids,
                    disposition=PolicyDisposition.AUTO,
                    policy_args=RefreshAdjustmentDiagnosticArgs(
                        action=RemediationAction.REFRESH_ADJUSTMENT_DIAGNOSTIC,
                        provider=first.provider,
                        range_start=min(item.range_start for item in evidence),
                        range_end=max(item.range_end for item in evidence),
                    ),
                )
            )
        if case_kind != "MARKETWIDE" and exclusion_options:
            recheck = self._recheck_after(max(item.range_end for item in evidence), observed_at)
            options.extend(
                (
                    RemediationOption.create(
                        option_id="recoverable_quarantine",
                        target_listing_ids=listing_ids,
                        disposition=PolicyDisposition.AUTO,
                        policy_args=QuarantineListingArgs(
                            action=RemediationAction.QUARANTINE_LISTING,
                            reason_code="QUALITY_REQUALIFICATION_REQUIRED",
                            recheck_after_at=recheck,
                        ),
                    ),
                    RemediationOption.create(
                        option_id="exclude_from_next_manifest",
                        target_listing_ids=listing_ids,
                        disposition=PolicyDisposition.AUTO,
                        policy_args=ExcludeFromNextManifestArgs(
                            action=RemediationAction.EXCLUDE_FROM_NEXT_MANIFEST,
                            reason_code="QUALITY_REQUALIFICATION_REQUIRED",
                            recheck_after_at=recheck,
                        ),
                    ),
                )
            )
        if (
            case_kind != "MARKETWIDE"
            and len(evidence) == 1
            and first.verified_alias_candidate_id is not None
        ):
            assert first.verified_alias_provider_symbol is not None
            assert first.verified_alias_evidence_hash is not None
            options.append(
                RemediationOption.create(
                    option_id="use_unique_verified_alias_for_this_run",
                    target_listing_ids=listing_ids,
                    disposition=PolicyDisposition.AUTO,
                    policy_args=VerifiedAliasArgs(
                        action=RemediationAction.USE_VERIFIED_ALIAS,
                        candidate_id=first.verified_alias_candidate_id,
                        provider_symbol=first.verified_alias_provider_symbol,
                        identity_evidence_hash=first.verified_alias_evidence_hash,
                    ),
                )
            )
        if any(item.extreme_move_unexplained for item in evidence):
            options.append(
                RemediationOption.create(
                    option_id="retain_isolated_raw_move_with_caveat",
                    target_listing_ids=listing_ids,
                    disposition=PolicyDisposition.HUMAN_REVIEW,
                    policy_args=RetainRawValueWithCaveatArgs(
                        action=RemediationAction.RETAIN_RAW_VALUE_WITH_CAVEAT,
                        caveat_code="UNEXPLAINED_ISOLATED_RAW_MOVE",
                    ),
                )
            )
        if case_kind == "MARKETWIDE" and last_known_good_snapshot_ref:
            options.append(
                RemediationOption.create(
                    option_id="retain_last_known_good_frozen_snapshot",
                    target_listing_ids=listing_ids,
                    disposition=PolicyDisposition.AUTO,
                    policy_args=UseLastKnownGoodArgs(
                        action=RemediationAction.USE_LAST_KNOWN_GOOD,
                        snapshot_ref=last_known_good_snapshot_ref,
                        bound_market_as_of_session=first.range_end,
                    ),
                )
            )
        options.append(
            RemediationOption.create(
                option_id="escalate_data_truth_conflict",
                target_listing_ids=listing_ids,
                disposition=PolicyDisposition.HUMAN_REVIEW,
                policy_args=EscalateForHumanArgs(
                    action=RemediationAction.ESCALATE_FOR_HUMAN,
                    review_kind="feature_input_data_truth_conflict",
                ),
            )
        )
        return tuple(options)

    @staticmethod
    def _decision(
        evidence: ListingQualityEvidence,
        assessment: FeatureInputAssessment,
    ) -> ListingEligibilityDecision:
        return ListingEligibilityDecision.create(
            listing_id=evidence.listing_id,
            assessment=assessment,
            reason_codes=evidence.reason_codes,
            evidence_hash=evidence.evidence_hash,
            requalification_conditions=(
                ("deterministic_retry_or_host_remediation",)
                if assessment is FeatureInputAssessment.DEFERRED
                else ()
            ),
        )


class FeatureInputRemediationExecutor:
    """Translate one validated option into deterministic host-owned effects."""

    def __init__(self, gateway: FeatureInputGateway) -> None:
        """Bind deterministic remediation effects to the input gateway."""
        self.gateway = gateway

    def execute(
        self,
        *,
        case: FeatureInputAgentCase,
        option: RemediationOption,
        evidence_by_listing_id: Mapping[str, ListingQualityEvidence],
        observed_at: datetime,
        panel_impact_after_exclusion: PanelImpactProjection | None = None,
        proposal_hash: str | None = None,
        human_confirmed: bool = False,
    ) -> FeatureInputPolicyExecution:
        """Execute one validated remediation option over its exact evidence."""
        if observed_at.tzinfo is None:
            raise ValueError("remediation execution time must be timezone-aware")
        if tuple(sorted(option.target_listing_ids)) != tuple(sorted(case.listing_ids)):
            raise ValueError("remediation option target scope does not match its case")
        if set(option.target_listing_ids) - set(evidence_by_listing_id):
            raise ValueError("remediation execution is missing listing evidence")
        action = option.policy_args.action
        policy_args_hash = canonical_hash(option.policy_args.model_dump(mode="json"))
        receipt = canonical_hash(
            [case.case_token, case.evidence_hash, option.option_hash, policy_args_hash]
        )
        if action is RemediationAction.RETAIN_RAW_VALUE_WITH_CAVEAT and human_confirmed:
            selected = tuple(evidence_by_listing_id[key] for key in option.target_listing_ids)
            if not proposal_hash or any(
                item.reason_codes != ("UNEXPLAINED_RAW_MOVE",)
                or not item.extreme_move_unexplained
                or not item.identity_verified
                for item in selected
            ):
                raise ValueError("feature_input.raw_caveat_requires_isolated_verified_move")
            return FeatureInputPolicyExecution(
                status=FeatureInputExecutionStatus.RAW_VALUE_RETAINED,
                option_id=option.option_id,
                option_hash=option.option_hash,
                policy_args_hash=policy_args_hash,
                execution_receipt_hash=canonical_hash([receipt, "HUMAN_CONFIRMED", proposal_hash]),
                retained_evidence=tuple((item.listing_id, item.evidence_hash) for item in selected),
            )
        if option.disposition is PolicyDisposition.HUMAN_REVIEW:
            return FeatureInputPolicyExecution(
                status=FeatureInputExecutionStatus.DATA_TRUTH_REVIEW,
                option_id=option.option_id,
                option_hash=option.option_hash,
                policy_args_hash=policy_args_hash,
                execution_receipt_hash=receipt,
            )
        if action in {
            RemediationAction.RETRY_PRIMARY,
            RemediationAction.REFRESH_ADJUSTMENT_DIAGNOSTIC,
            RemediationAction.USE_VERIFIED_ALIAS,
        }:
            return FeatureInputPolicyExecution(
                status=FeatureInputExecutionStatus.RETRY_SCHEDULED,
                option_id=option.option_id,
                option_hash=option.option_hash,
                policy_args_hash=policy_args_hash,
                execution_receipt_hash=receipt,
            )
        if action is RemediationAction.WAIT_THEN_RETRY:
            seconds = option.policy_args.retry_after_seconds  # type: ignore[union-attr]
            return FeatureInputPolicyExecution(
                status=FeatureInputExecutionStatus.DEFERRED,
                option_id=option.option_id,
                option_hash=option.option_hash,
                policy_args_hash=policy_args_hash,
                execution_receipt_hash=receipt,
                retry_after_at=observed_at.astimezone(UTC) + timedelta(seconds=seconds),
            )
        if action in {
            RemediationAction.QUARANTINE_LISTING,
            RemediationAction.EXCLUDE_FROM_NEXT_MANIFEST,
        }:
            if panel_impact_after_exclusion is None:
                raise ValueError("quarantine execution requires a panel-impact preflight")
            blocking = panel_impact_after_exclusion.blocking_reasons(self.gateway.policy)
            if blocking:
                return FeatureInputPolicyExecution(
                    status=FeatureInputExecutionStatus.DEFERRED,
                    option_id=option.option_id,
                    option_hash=option.option_hash,
                    policy_args_hash=policy_args_hash,
                    execution_receipt_hash=receipt,
                    failure_reasons=blocking,
                )
            recheck = option.policy_args.recheck_after_at  # type: ignore[union-attr]
            if recheck is None:
                raise ValueError("recoverable quarantine has no recheck time")
            quarantines = tuple(
                self.gateway.quarantine(
                    evidence=evidence_by_listing_id[listing_id],
                    recheck_after_at=recheck,
                    execution_receipt_hash=receipt,
                    agent_proposal_hash=proposal_hash,
                )
                for listing_id in option.target_listing_ids
            )
            return FeatureInputPolicyExecution(
                status=FeatureInputExecutionStatus.QUARANTINE_READY,
                option_id=option.option_id,
                option_hash=option.option_hash,
                policy_args_hash=policy_args_hash,
                execution_receipt_hash=receipt,
                quarantines=quarantines,
            )
        if action is RemediationAction.USE_LAST_KNOWN_GOOD:
            return FeatureInputPolicyExecution(
                status=FeatureInputExecutionStatus.LAST_KNOWN_GOOD_BOUND,
                option_id=option.option_id,
                option_hash=option.option_hash,
                policy_args_hash=policy_args_hash,
                execution_receipt_hash=receipt,
                snapshot_ref=option.policy_args.snapshot_ref,  # type: ignore[union-attr]
                bound_market_as_of_session=(
                    option.policy_args.bound_market_as_of_session  # type: ignore[union-attr]
                ),
            )
        if action is RemediationAction.REQUALIFY_LISTING:
            return FeatureInputPolicyExecution(
                status=FeatureInputExecutionStatus.REQUALIFICATION_READY,
                option_id=option.option_id,
                option_hash=option.option_hash,
                policy_args_hash=policy_args_hash,
                execution_receipt_hash=receipt,
            )
        raise ValueError("unsupported Feature Input remediation action")


class FeatureBaselinePopulationError(ValueError):
    """Too few candidates qualify for the baseline: its counts and why, beside the stable code.

    `cause` is in a Task stage's failure-cause fields, so the preparation that stops on it shows
    how many qualified against what the rule needs and why the others did not.
    """

    def __init__(self, cause: dict[str, object]) -> None:
        """Keep the shortfall's cause under the stable refusal code."""
        super().__init__("feature.baseline_qualified_population_insufficient")
        self.cause = cause


def _population_shortfall(
    *,
    session: date,
    candidates: int,
    counts: Mapping[str, int],
    minimum: int,
    excluded: Mapping[str, tuple[str, ...]],
    latest_bar: date | None,
) -> dict[str, object]:
    """The baseline's shortfall in one sentence, which the Task's failure cause bounds:
    qualified against the rule, the sectors short of it, and the unqualified grouped by their
    first reason, the largest group first."""
    reasons: dict[str, int] = defaultdict(int)
    for codes in excluded.values():
        kind, _, rest = codes[0].partition(":")
        # A feature's reason without its factor, so one cause over many factors counts once.
        why = rest.partition(":")[2]
        reasons[f"{kind}:{why}" if why else kind] += 1
    short = sorted((name, count) for name, count in counts.items() if count < minimum)

    def worded(reason: str) -> str:
        if reason == "BASE_MARKET_OBSERVATION_UNAVAILABLE":
            latest = f" (latest bar {latest_bar})" if latest_bar is not None else ""
            return f"no market observation at the session{latest}"
        kind, _, why = reason.partition(":")
        return f"a base feature not computable ({why})" if why else kind

    detail = (
        f"{sum(counts.values())} of {candidates} candidates qualify at {session}; the baseline "
        f"needs at least one, and {minimum} in each sector present."
        + (
            " Short: " + ", ".join(f"{name} {count}" for name, count in short) + "."
            if short
            else ""
        )
        + (
            " Not qualified: "
            + "; ".join(
                f"{count} {worded(reason)}"
                for reason, count in sorted(reasons.items(), key=lambda item: -item[1])
            )
            + "."
            if reasons
            else ""
        )
    )
    return {
        "exception_type": FeatureBaselinePopulationError.__name__,
        "detail": detail,
        "step": "baseline_qualification",
    }


@dataclass
class FeatureInputGovernanceService:
    """Composition owner that serializes durable Gateway results through one writer."""

    market_data: MarketDataRepository
    panel_state: PanelStateRepository
    mutation_gate: WorkspaceMutationGate
    gateway: FeatureInputGateway

    def admit_feature_baseline(
        self,
        *,
        candidate_manifest: UniverseManifest,
        qualification: MaterializedFeatureQualification,
        sector: SectorReferenceState,
        temporal_boundary: TemporalKnowledgeBoundary,
        observed_at: datetime,
    ) -> FeatureInputGatewayResult:
        """Qualify initial members or new candidates; never requalify history by implication.

        Before initial-cohort admission, the candidate pass rate is not Panel coverage.
        Panel coverage
        is subsequently measured on the qualified cohort without changing its
        floors. After initial-cohort admission only new candidates are checked;
        existing members keep
        their prior qualification and dated usability. The journal is committed
        separately by the membership caller, not by this Feature admission.
        """
        parent = candidate_manifest
        sector_by_listing_id = sector.sector_by_listing_id
        bundle = desktop_core_feature_bundle()
        boundary = temporal_boundary
        bootstrap = self.market_data.universe_bootstrap(parent.profile.market_profile_id)
        prior = (
            set(
                journal_members(
                    bootstrap, self.market_data.membership_events(parent.profile.market_profile_id)
                )
            )
            if bootstrap is not None
            else set()
        )
        candidates = {item.listing_id for item in parent.listings}
        scope = (
            set(qualification.eligible_listing_ids)
            | set(qualification.pending_listing_ids)
            | {listing for listing, _reasons in qualification.exclusions}
        )
        if (
            scope != candidates - prior
            or qualification.session != boundary.market_as_of_session
            or qualification.factor_ids != tuple(sorted(bundle.factor_ids))
        ):
            raise ValueError("feature.baseline_qualification_scope_mismatch")
        obligation = qualification_obligation("baseline_features", bundle.bundle_hash)
        excluded = {
            listing: tuple(
                f"BASE_FEATURE_UNAVAILABLE:{factor}:{reason}" for factor, reason in reasons
            )
            for listing, reasons in qualification.exclusions
        }
        latest_bar: date | None = None
        for listing in qualification.pending_listing_ids:
            sessions = self.market_data.raw_bar_sessions(listing, through=qualification.session)
            if qualification.session in sessions:
                raise ValueError("feature.baseline_member_row_not_materialized")
            excluded[listing] = ("BASE_MARKET_OBSERVATION_UNAVAILABLE",)
            if sessions and (latest_bar is None or sessions[-1] > latest_bar):
                latest_bar = sessions[-1]
        admitted = tuple(sorted(set(qualification.eligible_listing_ids) | (candidates & prior)))
        counts: dict[str, int] = {}
        for listing in admitted:
            name = sector_by_listing_id[listing]
            counts[name] = counts.get(name, 0) + 1
        policy = self.gateway.policy
        if not admitted or any(count < policy.minimum_sector_size for count in counts.values()):
            raise FeatureBaselinePopulationError(
                _population_shortfall(
                    session=qualification.session,
                    candidates=len(candidates - prior),
                    counts=counts,
                    minimum=policy.minimum_sector_size,
                    excluded=excluded,
                    latest_bar=latest_bar,
                )
            )
        for domain in FEATURE_QUALIFICATION_DOMAINS:
            for held in self.panel_state.active_listing_quarantines(
                parent.revision_sha256,
                include_profile_history=True,
                qualification_domain=domain,
            ):
                if held.listing_id in qualification.eligible_listing_ids:
                    self.mutation_gate.run(
                        self.panel_state.clear_listing_quarantine,
                        quarantine_hash=held.quarantine_hash,
                        qualification_receipt_hash=(
                            sector.sector_revision
                            if domain == "SECTOR_REFERENCE"
                            else qualification.evidence_hash
                        ),
                        cleared_at=observed_at,
                        effective_session=qualification.session,
                        qualification_domain=domain,
                    )
        quarantines = tuple(
            ListingQuarantine.create(
                listing_id=listing,
                reason_codes=reasons,
                evidence_hash=canonical_hash([qualification.evidence_hash, listing]),
                execution_receipt_hash=qualification.evidence_hash,
                recheck_after_at=observed_at + timedelta(days=1),
            )
            for listing, reasons in sorted(excluded.items())
        )
        reduced = (
            parent
            if bootstrap is not None and set(admitted) == candidates
            else build_quality_filtered_research_manifest(
                parent,
                eligible_listing_ids=admitted,
                qualification_obligations=(obligation,) if bootstrap is None else (),
            )
        )
        admission = FeatureInputAdmission.create(
            candidate_manifest_revision=parent.revision_sha256,
            admitted_listing_ids=admitted,
            quarantines=quarantines,
            quality_policy_hash=bundle.bundle_hash,
            temporal_boundary=boundary,
            evidence_scope="BASE_FEATURES_ONLY" if bootstrap is None else "BASE_FEATURE_CANDIDATES",
            feature_qualification_hash=qualification.evidence_hash,
        )
        result = FeatureInputGatewayResult(
            assessment=FeatureInputAssessment.QUARANTINED
            if excluded
            else FeatureInputAssessment.ADMITTED,
            decisions=tuple(
                ListingEligibilityDecision.create(
                    listing_id=listing.listing_id,
                    assessment=FeatureInputAssessment.QUARANTINED
                    if listing.listing_id in excluded
                    else FeatureInputAssessment.ADMITTED,
                    reason_codes=excluded.get(listing.listing_id, ()),
                    evidence_hash=canonical_hash([qualification.evidence_hash, listing.listing_id]),
                )
                for listing in parent.listings
            ),
            admission=admission,
            research_manifest=reduced,
            quarantines=quarantines,
        )
        self.mutation_gate.run(
            self.panel_state.record_feature_input_gateway_result,
            candidate_manifest=parent,
            result=result,
            temporal_boundary=boundary,
            observed_at=observed_at,
        )
        return result

    def assess_and_record(
        self,
        *,
        candidate_manifest: UniverseManifest,
        evidence: tuple[ListingQualityEvidence, ...],
        sector_by_listing_id: Mapping[str, str],
        temporal_boundary: TemporalKnowledgeBoundary,
        observed_at: datetime,
        panel_impact: PanelImpactProjection | None = None,
        macro_retry_exhausted: bool = False,
        observed_workers: int = 4,
        last_known_good_snapshot_ref: str | None = None,
        sector_unknown_listing_ids: frozenset[str] = frozenset(),
        raw_retention_proofs: tuple[RawRetentionDecisionProof, ...] = (),
    ) -> FeatureInputGatewayResult:
        """Assess candidate inputs and persist the resulting governance state."""
        self.mutation_gate.run(self.market_data.bootstrap, candidate_manifest)
        retained_moves: dict[str, tuple[str, str]] = {}
        for effect in self.panel_state.feature_input_effect_documents(
            candidate_manifest.revision_sha256
        ):
            execution = TypeAdapter(FeatureInputPolicyExecution).validate_json(effect)
            if execution.status is FeatureInputExecutionStatus.RAW_VALUE_RETAINED:
                for listing_id, evidence_hash in execution.retained_evidence:
                    retained_moves[listing_id] = (evidence_hash, execution.execution_receipt_hash)
        evidence = tuple(
            replace(
                item,
                failure_code=None,
                reason_codes=(),
                retry_exhausted=False,
                caveat_receipt_hash=retained_moves[item.listing_id][1],
            )
            if (
                item.listing_id in retained_moves
                and retained_moves[item.listing_id][0] == item.evidence_hash
                and item.reason_codes == ("UNEXPLAINED_RAW_MOVE",)
                and item.failure_code == "data.unexplained_raw_move"
            )
            else item
            for item in evidence
        )
        bootstrap = self.market_data.universe_bootstrap(
            candidate_manifest.profile.market_profile_id
        )
        nominal = (
            journal_members(
                bootstrap,
                self.market_data.membership_events(candidate_manifest.profile.market_profile_id),
            )
            if bootstrap is not None
            else ()
        )
        nominal_hash = (
            membership_identity(nominal)
            if bootstrap is not None
            and set(nominal) == {item.listing_id for item in candidate_manifest.listings}
            else None
        )
        existing = self.panel_state.active_listing_quarantines(
            candidate_manifest.revision_sha256,
            include_profile_history=bootstrap is not None,
            qualification_domain="MARKET_DATA",
        )
        evidence_by_listing = {item.listing_id: item for item in evidence}
        retained = []
        for quarantine in existing:
            current = evidence_by_listing.get(quarantine.listing_id)
            if current is None or observed_at.astimezone(UTC) < quarantine.recheck_after_at:
                retained.append(quarantine)
            elif current.admitted and current.qualification_receipt_hash:
                self.gateway.requalify(
                    quarantine,
                    current,
                    observed_at=observed_at,
                )
                self.mutation_gate.run(
                    self.panel_state.clear_listing_quarantine,
                    quarantine_hash=quarantine.quarantine_hash,
                    qualification_receipt_hash=current.qualification_receipt_hash,
                    cleared_at=observed_at,
                    effective_session=current.range_end,
                    qualification_domain="MARKET_DATA",
                )
            elif not current.admitted:
                # The recheck ran and the listing still fails. A standing
                # decision whose basis is unchanged is continued to the next
                # recheck; otherwise the listing re-enters the existing path
                # and the refusal is readable by the same judgement.
                judged = self.judge_standing_quarantine(
                    candidate_manifest=candidate_manifest,
                    quarantine=quarantine,
                    current=current,
                    observed_at=observed_at,
                )
                if isinstance(judged, QuarantineContinuation):
                    self.mutation_gate.run(
                        self.panel_state.record_quarantine_continuation,
                        candidate_manifest_revision=candidate_manifest.revision_sha256,
                        continuation=judged,
                        observed_at=observed_at,
                        effective_session=current.range_end,
                    )
                    retained.append(judged.quarantine)
                else:
                    retained.append(quarantine)
            else:
                retained.append(quarantine)  # due, admitted, no fresh receipt: not requalified
        result = self.gateway.assess(
            candidate_manifest=candidate_manifest,
            evidence=evidence,
            sector_by_listing_id=sector_by_listing_id,
            temporal_boundary=temporal_boundary,
            observed_at=observed_at,
            panel_impact=panel_impact,
            existing_quarantines=tuple(retained),  # type: ignore[arg-type]
            macro_retry_exhausted=macro_retry_exhausted,
            observed_workers=observed_workers,
            last_known_good_snapshot_ref=last_known_good_snapshot_ref,
            nominal_membership_hash=nominal_hash,
            sector_unknown_listing_ids=sector_unknown_listing_ids,
        )
        historical_moves: dict[str, tuple[str, str]] = {}
        if raw_retention_proofs:
            for current_case in result.agent_cases:
                proof = matching_raw_retention_proof(
                    case=current_case,
                    candidate_manifest=candidate_manifest,
                    proofs=raw_retention_proofs,
                    policy_hash=self.gateway.policy.policy_hash,
                )
                if proof is None:
                    continue
                for listing_id, evidence_hash in proof.retained_evidence:
                    historical_moves[listing_id] = (
                        evidence_hash,
                        proof.execution_receipt_hash,
                    )
        if historical_moves:
            evidence = tuple(
                replace(
                    item,
                    failure_code=None,
                    reason_codes=(),
                    retry_exhausted=False,
                    caveat_receipt_hash=historical_moves[item.listing_id][1],
                )
                if (
                    item.listing_id in historical_moves
                    and historical_moves[item.listing_id][0] == item.evidence_hash
                    and item.failure_code == "data.unexplained_raw_move"
                    and item.reason_codes == ("UNEXPLAINED_RAW_MOVE",)
                )
                else item
                for item in evidence
            )
            result = self.gateway.assess(
                candidate_manifest=candidate_manifest,
                evidence=evidence,
                sector_by_listing_id=sector_by_listing_id,
                temporal_boundary=temporal_boundary,
                observed_at=observed_at,
                panel_impact=panel_impact,
                existing_quarantines=tuple(retained),  # type: ignore[arg-type]
                macro_retry_exhausted=macro_retry_exhausted,
                observed_workers=observed_workers,
                last_known_good_snapshot_ref=last_known_good_snapshot_ref,
                nominal_membership_hash=nominal_hash,
                sector_unknown_listing_ids=sector_unknown_listing_ids,
            )
        # Reopen the exact pending catalog after a poll/restart. Regenerating
        # timestamped options is not a new problem or a new decision authority.
        prior = {}
        for document in self.panel_state.feature_input_case_documents(
            candidate_manifest.revision_sha256
        ):
            case = FeatureInputAgentCase.read_document(document)
            resolution = self.panel_state.feature_input_resolution(case.case_token)
            effect = (resolution or {}).get("effect", {})
            retry_at = effect.get("retry_after_at")
            if retry_at is not None and datetime.fromisoformat(str(retry_at)) <= observed_at:
                continue  # A wait was consumed once; re-evaluate, never silently renew it.
            if case.options_current_at(observed_at):
                prior[case.issue_hash] = case
        result = replace(
            result,
            agent_cases=tuple(prior.get(case.issue_hash, case) for case in result.agent_cases),
        )
        self.mutation_gate.run(
            self.panel_state.record_feature_input_gateway_result,
            candidate_manifest=candidate_manifest,
            result=result,
            temporal_boundary=temporal_boundary,
            observed_at=observed_at,
        )
        return result

    def judge_standing_quarantine(
        self,
        *,
        candidate_manifest: UniverseManifest,
        quarantine: ListingQuarantine,
        current: ListingQualityEvidence,
        observed_at: datetime,
    ) -> QuarantineContinuation | QuarantineContinuationRefusal:
        """Read the decision a standing quarantine carries and let the Gateway judge it.

        Pure: the original case, its sealed choice and its executed effect
        come from the case store by the receipt the quarantine names; nothing
        is written. The maintenance cycle records a continuation; a readback
        explains a refusal by the same judgement over the same frozen inputs.
        """
        found = self.panel_state.feature_input_decision_for_receipt(
            quarantine.execution_receipt_hash
        )
        if found is None:
            return QuarantineContinuationRefusal(
                quarantine.listing_id,
                quarantine.quarantine_hash,
                ("ORIGINAL_DECISION_UNAVAILABLE",),
            )
        case_document, resolution = found
        case = FeatureInputAgentCase.read_document(case_document)
        receipt = resolution.get("receipt")
        effect = resolution.get("effect")
        decision = receipt.get("policy_decision") if isinstance(receipt, dict) else None
        if (
            not isinstance(receipt, dict)
            or not isinstance(decision, dict)
            or not isinstance(effect, dict)
        ):
            return QuarantineContinuationRefusal(
                quarantine.listing_id,
                quarantine.quarantine_hash,
                ("ORIGINAL_DECISION_NOT_EXECUTED",),
            )
        try:
            original_manifest = self.market_data.load_universe_manifest_revision(
                case.manifest_revision
            )
        except ValueError:
            original_manifest = None
        if (
            original_manifest is None
            or original_manifest.profile.market_profile_id
            != candidate_manifest.profile.market_profile_id
        ):
            return QuarantineContinuationRefusal(
                quarantine.listing_id, quarantine.quarantine_hash, ("WORKSPACE_PROFILE_CHANGED",)
            )
        return self.gateway.continue_quarantine(
            quarantine=quarantine,
            original_case=case,
            original_option_id=str(decision.get("option_id")),
            original_execution_policy_hash=str(receipt.get("execution_policy_hash")),
            original_execution=TypeAdapter(FeatureInputPolicyExecution).validate_python(effect),
            current=current,
            observed_at=observed_at,
        )

    def record_policy_execution(
        self,
        *,
        candidate_manifest_revision: str,
        execution: FeatureInputPolicyExecution,
        observed_at: datetime,
        effective_session: date | None = None,
    ) -> bool:
        """Commit one idempotent host effect after Agent selection validation."""
        applied = self.mutation_gate.run(
            self.market_data.record_remediation_execution,
            execution.execution_receipt_hash,
            execution.option_id,
            executed_at=observed_at,
        )
        if execution.quarantines:
            self.mutation_gate.run(
                self.panel_state.record_listing_quarantines,
                candidate_manifest_revision=candidate_manifest_revision,
                quarantines=execution.quarantines,
                observed_at=observed_at,
                effective_session=effective_session,
            )
        return bool(applied)
