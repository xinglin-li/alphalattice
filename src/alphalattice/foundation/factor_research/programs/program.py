"""Deterministic one-session Factor Research program composition."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Protocol, cast

import pyarrow as pa
import pyarrow.compute as pc
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.causal_outcomes.execution.contracts import (
    CausalExecutionOutcomeManifest,
)
from alphalattice.foundation.causal_outcomes.execution.methods import (
    ExecutionOutcomeMethodSeal,
)
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.foundation.factor_research.evaluation.oos_evidence import (
    FactorEvidencePolicy,
    FactorOosEvidenceReport,
    compute_factor_oos_evidence,
)
from alphalattice.foundation.factor_research.evaluation.redundancy import (
    FactorRedundancyPolicy,
    FactorRedundancyStructure,
    build_factor_redundancy_structure,
)
from alphalattice.foundation.factor_research.inputs.execution_target import (
    FactorTargetPolicy,
    FactorTargetQualityReport,
    FactorTargetSurface,
    FactorTargetSurfaceManifest,
    compile_factor_target_surface,
)
from alphalattice.foundation.factor_research.programs.sealed import seal_contract
from alphalattice.foundation.factor_research.programs.walk_forward import (
    FactorWalkForwardPlan,
    FactorWalkForwardPolicy,
    compile_factor_walk_forward_plan,
    factor_walk_forward_development_sessions,
)
from alphalattice.foundation.feature_engine.contracts import panel_member_counts
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReadRequest
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class FactorResearchProgramBoundaryError(ValueError):
    """Stable failure raised before deterministic program publication."""


class _FeatureReader(Protocol):
    def available_sessions(self, manifest_ref: str) -> tuple[date, ...]: ...

    def batches(self, request: FeaturePanelReadRequest) -> Iterator[pa.RecordBatch]: ...


class _OutcomeReader(Protocol):
    def load_manifest(self, snapshot_hash: str) -> CausalExecutionOutcomeManifest: ...

    def read_development_sessions(
        self, manifest_ref: str, sessions: tuple[date, ...]
    ) -> pa.Table: ...


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class FactorResearchProgramSpec(_Contract):
    """Freeze qualified inputs and deterministic policies for one Factor research program.

    Attributes:
        kind: Program-specification discriminator.
        feature_panel_snapshot_hash: Admitted Feature Panel identity.
        feature_panel_manifest_ref: Reference read by the Feature reader.
        causal_outcome_snapshot_hash: Admitted execution-outcome identity.
        causal_outcome_manifest_ref: Reference read by the development outcome reader.
        factor_ids: Nonempty sorted unique catalog factor axis.
        frozen_at: Timezone-aware evidence clock.
        target_policy: Target construction and execution semantics.
        walk_forward_policy: Development split and reserved-holdout policy.
        evidence_policy: Complete-family out-of-sample statistical policy.
        redundancy_policy: Measured cross-factor clustering policy.
        data_validity_class: Current-universe research scope of the inputs.
        listing_ids: Optional unique exploration sample; target construction retains the whole
            panel.
        program_hash: Canonical identity of program inputs and policies.
    """

    kind: Literal["FactorResearchProgramSpec"] = "FactorResearchProgramSpec"
    feature_panel_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_panel_manifest_ref: str = Field(min_length=1)
    causal_outcome_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    causal_outcome_manifest_ref: str = Field(min_length=1)
    factor_ids: tuple[str, ...] = Field(min_length=1)
    frozen_at: datetime
    target_policy: FactorTargetPolicy
    walk_forward_policy: FactorWalkForwardPolicy
    evidence_policy: FactorEvidencePolicy
    redundancy_policy: FactorRedundancyPolicy
    data_validity_class: Literal["CURRENT_UNIVERSE_RESEARCH_ONLY"] = (
        "CURRENT_UNIVERSE_RESEARCH_ONLY"
    )
    listing_ids: tuple[str, ...] | None = Field(default=None, exclude_if=lambda v: v is None)
    """The names the evidence is computed over, for an exploration sample (binding plan,
    decision 4); None for the whole Panel. The targets stay the whole cross-section's, so a
    sample changes which names are ranked, not what a name's target is. Absent from the
    identity of a whole-Panel program, so every program sealed before it keeps its hash."""
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_spec(self) -> FactorResearchProgramSpec:
        """Validate the factor axis, optional sample, frozen clock, and program identity.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: Factors are noncanonical, sample is empty or repeated, clock is naive, or
                the hash is invalid.
        """
        if self.factor_ids != tuple(sorted(set(self.factor_ids))):
            raise ValueError("Factor Research program factor axis is not canonical")
        if self.listing_ids is not None and (
            not self.listing_ids or len(set(self.listing_ids)) != len(self.listing_ids)
        ):
            raise ValueError("Factor Research program listing sample is invalid")
        if self.frozen_at.tzinfo is None or self.frozen_at.utcoffset() is None:
            raise ValueError("Factor Research program clock is invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"program_hash"}))
        if self.program_hash != expected:
            raise ValueError("Factor Research program hash is invalid")
        return self


@dataclass(frozen=True, slots=True)
class FactorResearchDeterministicResult:
    """Carry computed development statistics and their source-population counts.

    Attributes:
        program: Qualified specification executed by the Host.
        target_surface: Verified complete execution-return targets.
        walk_forward_plan: Formal folds and verified training-label availability.
        evidence_report: Out-of-sample statistics for the complete factor family.
        redundancy_structure: Measured redundancy clusters on that family.
        feature_table: Development Feature rows after any declared exploration sample.
        feature_batch_count: Number of source batches consumed.
        feature_row_count: Number of Feature rows used for evidence.
        target_row_count: Number of full-panel target rows.
        development_sessions: Ordered union of training and validation sessions in complete folds.
    """

    program: FactorResearchProgramSpec
    target_surface: FactorTargetSurface
    walk_forward_plan: FactorWalkForwardPlan
    evidence_report: FactorOosEvidenceReport
    redundancy_structure: FactorRedundancyStructure
    feature_table: pa.Table
    feature_batch_count: int
    feature_row_count: int
    target_row_count: int
    development_sessions: tuple[date, ...]
    """The sessions the complete development folds consumed, in order: the
    formation axis this run was computed over, read off the same admitted
    calendar the rows were read for."""


class FactorResearchDeterministicEvidence(_Contract):
    """Durable Host checkpoint between expensive statistics and Agent review."""

    kind: Literal["FactorResearchDeterministicEvidence"] = "FactorResearchDeterministicEvidence"
    program: FactorResearchProgramSpec
    target_quality: FactorTargetQualityReport
    target_surface_manifest: FactorTargetSurfaceManifest
    walk_forward_plan: FactorWalkForwardPlan
    evidence_report: FactorOosEvidenceReport
    redundancy_structure: FactorRedundancyStructure
    feature_batch_count: int = Field(ge=1)
    feature_row_count: int = Field(ge=1)
    target_row_count: int = Field(ge=1)
    execution_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_checkpoint(self) -> FactorResearchDeterministicEvidence:
        """Verify target lineage, complete factor axes, counts, and checkpoint identity.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: Quality lineage, evidence/redundancy axes, target count, or canonical hash
                differs.
        """
        if self.target_surface_manifest.quality_hash != self.target_quality.quality_hash:
            raise ValueError("Factor deterministic checkpoint target lineage is invalid")
        if self.evidence_report.factor_ids != self.program.factor_ids:
            raise ValueError("Factor deterministic checkpoint evidence axis is invalid")
        if self.redundancy_structure.factor_ids != self.program.factor_ids:
            raise ValueError("Factor deterministic checkpoint redundancy axis is invalid")
        if self.target_row_count != self.target_surface_manifest.row_count:
            raise ValueError("Factor deterministic checkpoint target count is invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"checkpoint_hash"}))
        if self.checkpoint_hash != expected:
            raise ValueError("Factor deterministic checkpoint hash is invalid")
        return self


def seal_factor_research_deterministic_evidence(
    *,
    result: FactorResearchDeterministicResult,
    execution_binding_hash: str,
) -> FactorResearchDeterministicEvidence:
    """Seal a durable checkpoint from the Host's computed result and execution binding.

    Args:
        result: Completed deterministic statistics with their admitted source lineage.
        execution_binding_hash: Qualified execution binding for the recorded computation.

    Returns:
        Immutable checkpoint carrying policies, quality, folds, evidence, clusters,
        row/batch counts, and a canonical content identity.

    Raises:
        ValueError: Lineage, axes, counts, or binding fields fail checkpoint validation.
    """
    values = {
        "kind": "FactorResearchDeterministicEvidence",
        "program": result.program,
        "target_quality": result.target_surface.quality,
        "target_surface_manifest": result.target_surface.manifest,
        "walk_forward_plan": result.walk_forward_plan,
        "evidence_report": result.evidence_report,
        "redundancy_structure": result.redundancy_structure,
        "feature_batch_count": result.feature_batch_count,
        "feature_row_count": result.feature_row_count,
        "target_row_count": result.target_row_count,
        "execution_binding_hash": execution_binding_hash,
    }
    return cast(
        FactorResearchDeterministicEvidence,
        seal_contract(FactorResearchDeterministicEvidence, "checkpoint_hash", **values),
    )


def build_factor_research_program_spec(
    *,
    feature_panel_snapshot_hash: str,
    feature_panel_manifest_ref: str,
    causal_outcome_snapshot_hash: str,
    causal_outcome_manifest_ref: str,
    factor_ids: tuple[str, ...],
    frozen_at: datetime,
    target_policy: FactorTargetPolicy,
    walk_forward_policy: FactorWalkForwardPolicy,
    evidence_policy: FactorEvidencePolicy,
    redundancy_policy: FactorRedundancyPolicy,
    listing_ids: tuple[str, ...] | None = None,
) -> FactorResearchProgramSpec:
    """Seal a declared Factor program without reading inputs or executing statistics.

    Args:
        feature_panel_snapshot_hash: Admitted Feature Panel identity.
        feature_panel_manifest_ref: Reference for reopening the panel.
        causal_outcome_snapshot_hash: Admitted execution-outcome identity.
        causal_outcome_manifest_ref: Reference for reopening the outcomes.
        factor_ids: Canonical factor axis expected in the panel catalog.
        frozen_at: Timezone-aware research evidence clock.
        target_policy: Qualified execution-return target policy.
        walk_forward_policy: Qualified development split policy.
        evidence_policy: Qualified statistical evidence policy.
        redundancy_policy: Qualified cross-factor clustering policy.
        listing_ids: Optional unique exploration sample; None selects the whole panel.

    Returns:
        Validated current-universe program with a canonical identity.

    Raises:
        ValueError: A factor/sample axis, clock, policy, reference, or identity is invalid.
    """
    return cast(
        FactorResearchProgramSpec,
        seal_contract(
            FactorResearchProgramSpec,
            "program_hash",
            feature_panel_snapshot_hash=feature_panel_snapshot_hash,
            feature_panel_manifest_ref=feature_panel_manifest_ref,
            causal_outcome_snapshot_hash=causal_outcome_snapshot_hash,
            causal_outcome_manifest_ref=causal_outcome_manifest_ref,
            factor_ids=factor_ids,
            frozen_at=frozen_at,
            target_policy=target_policy,
            walk_forward_policy=walk_forward_policy,
            evidence_policy=evidence_policy,
            redundancy_policy=redundancy_policy,
            data_validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
            listing_ids=listing_ids,
        ),
    )


def _validate_panel_manifest(
    manifest: Mapping[str, object], program: FactorResearchProgramSpec
) -> None:
    if manifest.get("snapshot_hash") != program.feature_panel_snapshot_hash:
        raise FactorResearchProgramBoundaryError("factor_research.program_panel_identity_mismatch")
    summary = manifest.get("safe_summary")
    factor_summary = summary.get("factor_catalog_summary") if isinstance(summary, dict) else None
    if not isinstance(factor_summary, dict):
        raise FactorResearchProgramBoundaryError("factor_research.program_factor_catalog_missing")
    if tuple(sorted(str(value) for value in factor_summary)) != program.factor_ids:
        raise FactorResearchProgramBoundaryError("factor_research.program_factor_axis_mismatch")


def execute_factor_research_program(
    *,
    program: FactorResearchProgramSpec,
    panel_manifest: Mapping[str, object],
    feature_reader: _FeatureReader,
    outcome_reader: _OutcomeReader,
) -> FactorResearchDeterministicResult:
    """Read admitted development rows once and compute deterministic research evidence.

    Args:
        program: Frozen source references, factor axis, and deterministic policies.
        panel_manifest: Admitted panel manifest with its catalog and membership summary.
        feature_reader: Reader serving exact development sessions and projected factor columns.
        outcome_reader: Development reader serving the admitted outcomes and method seal.

    Returns:
        Verified targets, complete folds, statistical evidence, redundancy clusters, and
        Feature rows/counts. An optional exploration sample filters Feature rows only.

    Raises:
        FactorResearchProgramBoundaryError: Input identities, catalog axes, or expected
            source rows disagree, or the declared sample has no Feature rows.
        ValueError: A program or a downstream target/split/evidence contract is invalid.
    """
    program = FactorResearchProgramSpec.model_validate(program)
    _validate_panel_manifest(panel_manifest, program)
    panel_sessions = tuple(feature_reader.available_sessions(program.feature_panel_manifest_ref))
    development_sessions = factor_walk_forward_development_sessions(
        panel_sessions=panel_sessions,
        frozen_at=program.frozen_at,
        policy=program.walk_forward_policy,
    )
    outcome_manifest = outcome_reader.load_manifest(program.causal_outcome_snapshot_hash)
    if outcome_manifest.snapshot_hash != program.causal_outcome_snapshot_hash:
        raise FactorResearchProgramBoundaryError(
            "factor_research.program_outcome_identity_mismatch"
        )
    source_outcomes = outcome_reader.read_development_sessions(
        program.causal_outcome_manifest_ref,
        development_sessions,
    )
    target_surface = compile_factor_target_surface(
        source_table=source_outcomes,
        source_manifest=outcome_manifest,
        source_manifest_ref=program.causal_outcome_manifest_ref,
        policy=program.target_policy,
        outcome_method=_resolved_outcome_seal(outcome_reader, program.causal_outcome_snapshot_hash),
    )
    walk_forward_plan = compile_factor_walk_forward_plan(
        panel_sessions=panel_sessions,
        target_surface=target_surface,
        frozen_at=program.frozen_at,
        policy=program.walk_forward_policy,
    )
    request = FeaturePanelReadRequest(
        manifest_ref=program.feature_panel_manifest_ref,
        start_session=development_sessions[0],
        end_session=development_sessions[-1],
        exact_sessions=development_sessions,
        factor_columns=program.factor_ids,
        include_row_hash=False,
    )
    batches = tuple(feature_reader.batches(request))
    if not batches:
        raise FactorResearchProgramBoundaryError("factor_research.program_feature_rows_missing")
    feature_table = pa.Table.from_batches(batches).combine_chunks()
    summary = panel_manifest.get("safe_summary")
    if isinstance(summary, Mapping) and isinstance(summary.get("membership"), Mapping):
        expected = panel_member_counts(panel_manifest, development_sessions)
        observed = feature_table.group_by("session_date").aggregate([("listing_id", "count")])
        if {
            row["session_date"]: int(row["listing_id_count"]) for row in observed.to_pylist()
        } != expected:
            raise FactorResearchProgramBoundaryError("factor_research.program_member_rows_missing")
    if program.listing_ids is not None:
        feature_table = feature_table.filter(
            pc.is_in(feature_table["listing_id"], value_set=pa.array(program.listing_ids))
        )
        if feature_table.num_rows == 0:
            raise FactorResearchProgramBoundaryError("factor_research.program_sample_rows_missing")
    evidence_report = compute_factor_oos_evidence(
        feature_table=feature_table,
        feature_panel_snapshot_hash=program.feature_panel_snapshot_hash,
        feature_panel_manifest_ref=program.feature_panel_manifest_ref,
        target_surface=target_surface,
        walk_forward_plan=walk_forward_plan,
        factor_ids=program.factor_ids,
        policy=program.evidence_policy,
    )
    redundancy_structure = build_factor_redundancy_structure(
        feature_table=feature_table,
        feature_panel_snapshot_hash=program.feature_panel_snapshot_hash,
        feature_panel_manifest_ref=program.feature_panel_manifest_ref,
        factor_ids=program.factor_ids,
        policy=program.redundancy_policy,
    )
    return FactorResearchDeterministicResult(
        program=program,
        target_surface=target_surface,
        walk_forward_plan=walk_forward_plan,
        evidence_report=evidence_report,
        redundancy_structure=redundancy_structure,
        feature_table=feature_table,
        feature_batch_count=len(batches),
        feature_row_count=feature_table.num_rows,
        target_row_count=target_surface.table.num_rows,
        development_sessions=development_sessions,
    )


__all__ = [
    "FactorResearchDeterministicEvidence",
    "FactorResearchDeterministicResult",
    "FactorResearchProgramBoundaryError",
    "FactorResearchProgramSpec",
    "build_factor_research_program_spec",
    "execute_factor_research_program",
    "seal_factor_research_deterministic_evidence",
]


def _resolved_outcome_seal(
    outcome_reader: CausalExecutionOutcomeDevelopmentReader,
    snapshot_hash: str,
) -> ExecutionOutcomeMethodSeal | None:
    """Resolve the snapshot's method authority, or ``None`` for legacy evidence.

    Read and evaluation paths must keep serving snapshots published before the
    method seam existed -- frozen artifacts stay readable under their original
    identities -- so a ``LEGACY_READBACK_ONLY`` disposition becomes ``None``,
    the explicitly named legacy compatibility route, rather than being passed
    off as method authority. Development writers refuse that same disposition.
    """
    seal = outcome_reader.resolve_method_seal(snapshot_hash)
    return seal if seal.disposition == "METHOD_BOUND" else None
