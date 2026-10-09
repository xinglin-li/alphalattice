"""Resolve a published simple score by re-deriving it, never by reading it back.

The discipline here is the one the Stage-6 campaign enforced on the Alpha
calibration (the campaign retired with R01), for the same reason: a
stored-values readback establishes only that the writer agreed with itself. So
this re-reads the Panel, recomputes the standardization from the binding's own
declared axis, and reports ``rederived=True`` only when the bytes match the
published identity.

A resolution that cannot re-derive is not an error at consume time -- it is a
*disposition*. The campaign refused it before sealing a Program and its replay
recorded it as a readback-only reason, downgrading rather than raising, because
a panel snapshot that has since been retired makes a published graph less
verifiable without making it forged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from alphalattice.capabilities.portfolio_inputs.signed_score.contracts import (
    SIGNED_SCORE_SEMANTICS,
    ResolvedSignedScore,
    ScoreObservationAuthority,
)
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver

from .contracts import (
    SimpleSignedScoreBinding,
    SimpleSignedScoreMethodId,
    feature_id_for_simple_score_method,
)
from .materialization import materialize_simple_score
from .provenance import resolve_feature_clock
from .standardize import (
    FloatArray,
    SimpleSignalError,
    score_values_identity,
    simple_score_matrix,
)

REDERIVED = "REDERIVED_FROM_PANEL_UNDER_INSTALLED_STANDARDIZATION"
PANEL_UNAVAILABLE = "SIMPLE_SCORE_PANEL_SNAPSHOT_UNAVAILABLE"
VALUES_NOT_REDERIVABLE = "SIMPLE_SCORE_VALUES_NOT_REDERIVABLE"
IMPLEMENTATION_MOVED = "SIMPLE_SCORE_IMPLEMENTATION_CLOSURE_MOVED"


@dataclass(frozen=True, slots=True)
class ResolvedSimpleSignedScore:
    """One published score, and whether this build could rebuild it."""

    binding: SimpleSignedScoreBinding
    disposition: str
    rederived: bool
    standardized_values: FloatArray | None
    """``None`` whenever ``rederived`` is false. There is no partial lane."""


@dataclass(frozen=True, slots=True)
class MaterializedSimpleSignedScore:
    """A run-time score plus the owner contract needed for durable readback."""

    binding: SimpleSignedScoreBinding
    standardized_values: FloatArray


def implementation_closure_hash() -> str:
    """Bytes of the modules whose edits move a simple score's numbers."""
    from alphalattice.kernel.shared_kernel.source_identity import (
        source_component_id,
        switched_source_identity,
    )

    from . import materialization, standardize

    return str(
        switched_source_identity(
            {
                source_component_id(package_id="alpha_research", source_path=Path(path)): Path(path)
                for path in (standardize.__file__, materialization.__file__)
            },
            semantic_owner="alpha_research",
            numerical_role="simple-signed-score",
        )
    )


class WorkspaceSimpleSignalMaterializationAuthority:
    """Host-injected location authority for an installed simple score method.

    The research request selects a method id.  The Host supplies this object,
    which owns the immutable Panel location and snapshot identity.  Numerical
    materialization therefore happens only during ``run`` and no Campaign is
    given an absolute path, a matrix, or a private callback.
    """

    def __init__(
        self,
        *,
        panel_artifacts_root: Path,
        panel_manifest_ref: str,
        panel_snapshot_identity: str,
    ) -> None:
        """Bind simple-signal materialization to an explicit panel artifact authority.

        Args:
            panel_artifacts_root: Caller-owned panel artifact root resolved for readback.
            panel_manifest_ref: Exact retained panel manifest reference.
            panel_snapshot_identity: Required declared panel snapshot identity.
        """
        self._resolver = ArtifactResolver(panel_artifacts_root.resolve())
        self._panel_manifest_ref = panel_manifest_ref
        self._panel_snapshot_identity = panel_snapshot_identity

    def materialize(
        self,
        *,
        method_id: SimpleSignedScoreMethodId,
        ordered_formation_sessions: tuple[date, ...],
        ordered_listing_ids: tuple[str, ...],
    ) -> MaterializedSimpleSignedScore:
        """Materialize and bind one target-free score on the consumer's axis."""
        feature_id = feature_id_for_simple_score_method(method_id)
        materialized = materialize_simple_score(
            resolver=self._resolver,
            panel_manifest_ref=self._panel_manifest_ref,
            feature_id=feature_id,
            ordered_formation_sessions=ordered_formation_sessions,
            ordered_listing_ids=ordered_listing_ids,
        )
        clock = resolve_feature_clock(feature_id)
        binding = SimpleSignedScoreBinding.create(
            feature_id=feature_id,
            panel_manifest_ref=self._panel_manifest_ref,
            panel_snapshot_identity=self._panel_snapshot_identity,
            ordered_formation_sessions=ordered_formation_sessions,
            ordered_listing_ids=ordered_listing_ids,
            minimum_finite_listings=2,
            feature_formula_ref=clock.formula_ref,
            feature_formula=clock.formula,
            feature_window_sessions=clock.window_sessions,
            feature_source_interval=clock.source_interval,
            observation_session_offset_sessions=(clock.observation_session_offset_sessions),
            availability_delay_sessions=clock.availability_delay_sessions,
            availability_policy_id=clock.availability_policy_id,
            availability_policy_hash=clock.availability_policy_hash,
            observation_clock_hash=clock.observation_clock_hash,
            feature_methodology_identity=clock.methodology_identity,
            feature_catalog_binding_hash=clock.catalog_binding_hash,
            values_identity=materialized.values_identity,
            implementation_closure_hash=implementation_closure_hash(),
            resolved_cell_count=materialized.resolved_cell_count,
            minimum_finite_listings_observed=(materialized.minimum_finite_listings_observed),
        )
        return MaterializedSimpleSignedScore(
            binding=binding,
            standardized_values=materialized.standardized_values,
        )


def resolve_simple_signed_score(
    *, binding: SimpleSignedScoreBinding, panel_artifacts_root: Path
) -> ResolvedSimpleSignedScore:
    """Rebuild one published score from the Panel, or say why this build cannot."""
    if binding.implementation_closure_hash != implementation_closure_hash():
        # The standardization moved under a binding that stands still. The score
        # is still readable and is no longer reproducible here, which is a
        # different statement from either "fine" or "forged".
        return ResolvedSimpleSignedScore(
            binding=binding,
            disposition=IMPLEMENTATION_MOVED,
            rederived=False,
            standardized_values=None,
        )
    try:
        materialized = materialize_simple_score(
            resolver=ArtifactResolver(panel_artifacts_root),
            panel_manifest_ref=binding.panel_manifest_ref,
            feature_id=binding.feature_id,
            ordered_formation_sessions=binding.ordered_formation_sessions,
            ordered_listing_ids=binding.ordered_listing_ids,
        )
    except (ValueError, SimpleSignalError):
        # ``FeaturePanelReader`` refuses a snapshot that is not ACTIVE and
        # physically available. That is an availability fact about this machine,
        # not a defect in the graph.
        return ResolvedSimpleSignedScore(
            binding=binding,
            disposition=PANEL_UNAVAILABLE,
            rederived=False,
            standardized_values=None,
        )
    if score_values_identity(materialized.standardized_values) != binding.values_identity:
        return ResolvedSimpleSignedScore(
            binding=binding,
            disposition=VALUES_NOT_REDERIVABLE,
            rederived=False,
            standardized_values=None,
        )
    return ResolvedSimpleSignedScore(
        binding=binding,
        disposition=REDERIVED,
        rederived=True,
        standardized_values=materialized.standardized_values,
    )


def score_observation_authority(binding: SimpleSignedScoreBinding) -> ScoreObservationAuthority:
    """State this score's observation semantics in the neutral contract's terms.

    Every field is copied from the binding, and every field on the binding was
    re-resolved from the Feature owner when it validated. Nothing is decided
    here, which is why this is a projection and not a second declaration.

    What it does *not* build is an execution clock. This function used to return
    a contract carrying ``FORMATION_SESSION_CLOSE``, ``entry_offset_sessions=1``
    and ``NEXT_COMMON_SESSION_OFFICIAL_OPEN`` -- three constants copied out of
    ``causal_outcomes.execution`` into a score producer, where a later change to
    the execution method would have left them silently stale and a reader would
    have had two sources for one fact.
    """
    return ScoreObservationAuthority.create(
        observation_session_offset_sessions=int(binding.observation_session_offset_sessions),
        availability_delay_sessions=int(binding.availability_delay_sessions),
        availability_policy_id=binding.availability_policy_id,
        availability_policy_hash=binding.availability_policy_hash,
        observation_clock_hash=binding.observation_clock_hash,
        formula_observation_semantics=(
            f"{binding.feature_formula} over {binding.feature_source_interval}"
        ),
        source_authority_id=(
            f"alpha_research.simple_signal:{binding.feature_id}:{binding.feature_formula_ref}"
        ),
        methodology_identity=binding.feature_methodology_identity,
        strategy_scope=binding.strategy_scope,
    )


class WorkspaceSimpleSignalSource:
    """Location-only composition: which roots, never which score.

    Satisfies ``portfolio_inputs.signed_score.contracts.SignedScoreSource``
    structurally. The Protocol is ``runtime_checkable`` and this class does not
    inherit from it, so the dependency runs one way: a score producer answers a
    portfolio consumer's question without knowing a portfolio exists.
    """

    def __init__(self, *, score_evidence_root: Path, panel_artifacts_root: Path) -> None:
        """Bind simple-signal source readback to explicit evidence and panel roots.

        Args:
            score_evidence_root: Caller-owned score evidence root.
            panel_artifacts_root: Caller-owned panel artifact root.
        """
        self._score_evidence_root = score_evidence_root.resolve()
        self._panel_artifacts_root = panel_artifacts_root.resolve()

    def _load(self, binding_hash: str) -> SimpleSignedScoreBinding:
        from alphalattice.investment.alpha_research.experiments.development_artifacts import (
            AlphaDevelopmentArtifactStore,
        )

        loaded: SimpleSignedScoreBinding = AlphaDevelopmentArtifactStore(
            self._score_evidence_root
        ).load_simple_signed_score(binding_hash)
        return loaded

    def load_durable(
        self,
        *,
        artifact_hash: str | None,
        binding_hash: str | None,
    ) -> MaterializedSimpleSignedScore:
        """Read exact Alpha-owned bytes; a strong consumer may not re-open Panel."""
        if artifact_hash is None or binding_hash is None:
            raise SimpleSignalError("alpha_research.simple_signal_value_artifact_handle_required")
        from alphalattice.investment.alpha_research.experiments.development_artifacts import (
            AlphaDevelopmentArtifactReadbackError,
            AlphaDevelopmentArtifactStore,
        )

        try:
            binding, _artifact, values = AlphaDevelopmentArtifactStore(
                self._score_evidence_root
            ).load_simple_signed_score_values(
                artifact_hash=artifact_hash,
                binding_hash=binding_hash,
            )
        except AlphaDevelopmentArtifactReadbackError as error:
            raise SimpleSignalError(
                "alpha_research.simple_signal_value_artifact_not_admitted"
            ) from error
        return MaterializedSimpleSignedScore(binding=binding, standardized_values=values)

    def load_durable_projection(
        self,
        *,
        artifact_hash: str | None,
        binding_hash: str | None,
        formation_sessions: Sequence[date],
        ordered_listing_ids: Sequence[str],
    ) -> MaterializedSimpleSignedScore:
        """Verify the Alpha artifact, then project it onto an exact consumer axis."""
        materialized = self.load_durable(
            artifact_hash=artifact_hash,
            binding_hash=binding_hash,
        )
        binding = materialized.binding
        projected = simple_score_matrix(
            standardized_values=materialized.standardized_values,
            score_sessions=binding.ordered_formation_sessions,
            score_listing_ids=binding.ordered_listing_ids,
            formation_sessions=tuple(formation_sessions),
            ordered_listing_ids=tuple(ordered_listing_ids),
        )
        return MaterializedSimpleSignedScore(
            binding=binding,
            standardized_values=projected,
        )

    def resolve_simple(self, *, binding_hash: str) -> ResolvedSimpleSignedScore:
        """The producer-shaped resolution, with the values attached."""
        return resolve_simple_signed_score(
            binding=self._load(binding_hash), panel_artifacts_root=self._panel_artifacts_root
        )

    def resolve(self, *, binding_hash: str) -> ResolvedSignedScore:
        """The consumer-shaped resolution: axes, identity, clock, disposition."""
        resolved = self.resolve_simple(binding_hash=binding_hash)
        binding = resolved.binding
        return ResolvedSignedScore(
            semantics=SIGNED_SCORE_SEMANTICS,
            binding_hash=binding.binding_hash,
            values_identity=binding.values_identity,
            ordered_formation_sessions=binding.ordered_formation_sessions,
            ordered_listing_ids=binding.ordered_listing_ids,
            observation=score_observation_authority(binding),
            rederived=resolved.rederived,
            disposition=resolved.disposition,
        )

    def projection(
        self,
        *,
        binding_hash: str,
        formation_sessions: Sequence[date],
        ordered_listing_ids: Sequence[str],
    ) -> FloatArray:
        """The score on the axis the campaign decided on, re-derived not read.

        Refuses rather than returning a partial surface: a consumer that got
        ``None`` here would have to decide what an unresolvable score means, and
        that decision belongs to the campaign's authority resolution.
        """
        resolved = self.resolve_simple(binding_hash=binding_hash)
        if not resolved.rederived or resolved.standardized_values is None:
            raise SimpleSignalError(
                "alpha_research.simple_signal_not_rederivable:" + resolved.disposition
            )
        return simple_score_matrix(
            standardized_values=resolved.standardized_values,
            score_sessions=resolved.binding.ordered_formation_sessions,
            score_listing_ids=resolved.binding.ordered_listing_ids,
            formation_sessions=tuple(formation_sessions),
            ordered_listing_ids=tuple(ordered_listing_ids),
        )


__all__ = [
    "IMPLEMENTATION_MOVED",
    "PANEL_UNAVAILABLE",
    "REDERIVED",
    "VALUES_NOT_REDERIVABLE",
    "MaterializedSimpleSignedScore",
    "ResolvedSimpleSignedScore",
    "WorkspaceSimpleSignalMaterializationAuthority",
    "WorkspaceSimpleSignalSource",
    "implementation_closure_hash",
    "resolve_simple_signed_score",
    "score_observation_authority",
]
