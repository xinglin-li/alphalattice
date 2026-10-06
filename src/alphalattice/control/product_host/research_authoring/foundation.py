"""Assemble the input authority a development Alpha run actually has.

``research_foundation/mandate/foundation.py`` assembles the frozen constitution
from four pieces of admission evidence, and it is reachable only through
``research_foundation/publication/sponsorship.py`` -- an owner that publishes.
A development run has neither the evidence nor the authority, and borrowing that
assembler would mean either fabricating the four lineage values or acquiring
publication reach. This does neither.

What a development run genuinely has is read here, from three roots the Host is
told about explicitly:

- the published Feature Panel manifest and its logical identity,
- the published causal execution outcomes,
- one Factor development checkpoint, named by hash.

The three roots must be distinct and non-nested. That is not tidiness: a Factor
evidence root nested inside the source workspace makes a development artifact
look like workspace state, and one nested inside the Alpha output workspace makes
this run's own output eligible as next run's authority.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alphalattice.control.research_program.authoring.workflow import ResearchEvidenceStore
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.causal_outcomes.execution.methods import (
    build_installed_execution_outcome_publication_policy_catalog,
)
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
    DevelopmentOnlyExecutionOutcomeReader,
)
from alphalattice.foundation.causal_outcomes.selection.listing_set import (
    resolve_development_only_execution_outcome,
    resolve_published_execution_outcome,
    selected_snapshot_hash,
)
from alphalattice.foundation.factor_research.experiments.authoring import (
    factor_inventory_from_panel_manifest,
)
from alphalattice.foundation.factor_research.experiments.development_evidence import (
    FactorDevelopmentReceiptReader,
    verify_factor_development_receipt,
)
from alphalattice.foundation.feature_engine.panels.closure import PanelClosureArtifactStore
from alphalattice.foundation.feature_engine.panels.development_input import (
    ResolvedDevelopmentFeatureInput,
)
from alphalattice.foundation.feature_engine.panels.logical_identity import (
    PanelLogicalArtifactStore,
)
from alphalattice.investment.alpha_research.inputs.development_foundation import (
    AlphaDevelopmentExecutionOutcomeRef,
    AlphaDevelopmentFoundationBinding,
)
from alphalattice.kernel.validation.enums import RebalanceFrequency
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExecutionEvidence,
)


def assert_distinct_roots(*roots: Path) -> None:
    """Refuse any pair of roots where one contains the other, or they are equal."""
    resolved = [Path(value).resolve() for value in roots]
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if left == right:
                raise AuthoringError("research_authoring.evidence_root_identical")
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise AuthoringError("research_authoring.evidence_root_nested")


def _generic_evidence_for(evidence_root: Path, program_hash: str) -> ResearchExecutionEvidence:
    """Re-read the generic execution evidence that recorded this Factor run.

    Read from the same output workspace the Factor run wrote to, by
    ``program_hash``. It is handed to the verifier whole rather than reduced to a
    dict of four strings: the verifier decides which of its identities matter,
    and a caller that pre-selects them decides that question for it.
    """

    evidence = ResearchEvidenceStore(Path(evidence_root)).load_for_program(program_hash)
    if evidence is None:
        raise AuthoringError("research_authoring.factor_evidence_program_unavailable")
    return evidence


def build_alpha_development_foundation(
    *,
    panel_manifest: dict[str, Any],
    panel_snapshot_hash: str,
    artifact_root: Path,
    factor_evidence_root: Path,
    factor_evidence_handle: str,
    source_workspace: Path,
    output_workspace: Path,
    sector_revision: str | None = None,
    sector_coverage_hash: str | None = None,
    execution_outcome_recipe_id: str | None = None,
    outcome_artifact_root: Path | None = None,
    causal_outcome_snapshot_handle: str | None = None,
    feature_authority_outcome_snapshot_handle: str | None = None,
    feature_input: ResolvedDevelopmentFeatureInput | None = None,
) -> tuple[AlphaDevelopmentFoundationBinding, str]:
    """Bind one development Alpha run to the artifacts that authorize it.

    Returns the binding together with the causal outcome manifest ref, because a
    caller needs both and pairing a snapshot hash with someone else's ref is
    exactly the mistake handing them back separately invites.

    ``outcome_artifact_root`` separates where outcomes are *read* from where the
    Panel and its logical identity are read. They are the same root for an
    ordinary run and differ when a study prepared its outcome evidence into its
    own input root rather than writing into a source workspace it may not touch.
    """
    assert_distinct_roots(factor_evidence_root, source_workspace, output_workspace)

    # The receipt, not the bare child. The child is self-identifying and says
    # nothing about which authored Program produced it, so consuming it alone
    # meant "a Factor checkpoint exists" rather than "this Factor Program
    # authorized this run".
    receipt, evidence = FactorDevelopmentReceiptReader(factor_evidence_root).load(
        factor_evidence_handle
    )
    # Re-derived here, against the artifacts that actually own each identity. The
    # receipt validating its own hash proves only that nobody edited it; a
    # receipt that faithfully recorded the wrong Program passes every self-check
    # it has. Self-consistency is not authority.
    verify_factor_development_receipt(
        receipt=receipt,
        child=evidence,
        evidence=_generic_evidence_for(factor_evidence_root, receipt.program_hash),
        # Straight off the published Panel, in its published order, so the
        # comparison has two independent sides. A receipt checked against a
        # summary built from itself would agree with itself.
        panel_inventory=factor_inventory_from_panel_manifest(panel_manifest),
        panel_snapshot_hash=panel_snapshot_hash,
    )
    if evidence.program.feature_panel_snapshot_hash != panel_snapshot_hash:
        # The Factor checkpoint answered for a different Panel, so its factor axis
        # is not evidence about the columns this run would read.
        raise AuthoringError("research_authoring.factor_evidence_panel_mismatch")

    # Logical identity is published beside the Panel rather than inside its
    # manifest, so it is read from the owner that publishes it. The frozen
    # constitution treats the pair as all-or-nothing and so does this: a Panel
    # with only half a logical identity cannot answer for an array surface.
    resolver = ArtifactResolver(artifact_root)
    marker = PanelLogicalArtifactStore(PanelClosureArtifactStore(resolver)).marker_for_snapshot(
        feature_input.base_panel_snapshot_hash if feature_input else panel_snapshot_hash
    )
    if marker is None:
        raise AuthoringError("research_authoring.panel_logical_identity_unavailable")

    outcome_root = Path(outcome_artifact_root) if outcome_artifact_root else Path(artifact_root)
    listing_set_hash = str(panel_manifest.get("listing_set_hash", ""))
    # The workspace's frozen-contract snapshot is always resolved: it is the
    # clock the Factor evidence screened on, and the equality below is what
    # makes the feature axis evidence about the question rather than a label.
    reference_outcome = resolve_published_execution_outcome(
        artifact_root=outcome_root,
        listing_set_hash=listing_set_hash,
        snapshot_handle=feature_authority_outcome_snapshot_handle,
    )
    if evidence.program.causal_outcome_snapshot_hash != reference_outcome.snapshot_hash:
        raise AuthoringError("research_authoring.factor_evidence_outcome_mismatch")

    development_only_scope = False
    if execution_outcome_recipe_id is not None:
        policy = build_installed_execution_outcome_publication_policy_catalog().resolve(
            execution_outcome_recipe_id
        )
        development_only_scope = policy.publication_scope == "DEVELOPMENT_ONLY"

    feature_authority_snapshot: str | None = None
    if development_only_scope:
        # A longer-horizon development run reads the development-only successor
        # snapshot, and states openly that its feature authority was screened on
        # the one-session clock. Silence here would be a cross-clock borrowing
        # nobody could see; the explicit field is what a reader weighs.
        development_outcome = resolve_development_only_execution_outcome(
            artifact_root=outcome_root,
            listing_set_hash=listing_set_hash,
            snapshot_handle=causal_outcome_snapshot_handle,
        )
        outcome_snapshot_hash = str(development_outcome.snapshot_hash)
        outcome_listing_set_hash = str(development_outcome.listing_set_hash)
        feature_authority_snapshot = str(reference_outcome.snapshot_hash)
        manifest_ref = DevelopmentOnlyExecutionOutcomeReader(outcome_root).manifest_uri(
            outcome_snapshot_hash
        )
    else:
        if (
            causal_outcome_snapshot_handle is not None
            and selected_snapshot_hash(
                causal_outcome_snapshot_handle,
                manifest_uri=CausalExecutionOutcomeDevelopmentReader(outcome_root).manifest_uri,
            )
            != reference_outcome.snapshot_hash
        ):
            # A one-session arm reads the same snapshot its Factor evidence was
            # screened on. Naming a different one is a request nobody can honor
            # without also changing what the feature axis is evidence about.
            raise AuthoringError("research_authoring.factor_evidence_outcome_mismatch")
        outcome_snapshot_hash = str(reference_outcome.snapshot_hash)
        outcome_listing_set_hash = str(reference_outcome.listing_set_hash)
        manifest_ref = CausalExecutionOutcomeDevelopmentReader(outcome_root).manifest_uri(
            outcome_snapshot_hash
        )

    binding = AlphaDevelopmentFoundationBinding.create(
        research_cadence=RebalanceFrequency.DAILY,
        feature_panel_snapshot_hash=panel_snapshot_hash,
        logical_panel_hash=feature_input.logical_panel_hash
        if feature_input
        else marker.logical_panel_hash,
        logical_semantic_index_hash=feature_input.logical_semantic_index_hash
        if feature_input
        else marker.logical_semantic_index_hash,
        execution_outcome=AlphaDevelopmentExecutionOutcomeRef(
            research_cadence=RebalanceFrequency.DAILY,
            snapshot_hash=outcome_snapshot_hash,
            manifest_ref=manifest_ref,
            listing_set_hash=outcome_listing_set_hash,
        ),
        # The statistical universe the Factor checkpoint ran over. Kept whole
        # because it is upstream authority: it records what the selection was
        # made *from*, and narrowing it here to look like the selection would
        # destroy exactly that.
        ordered_factor_ids=tuple(evidence.program.factor_ids),
        # What this Factor run actually reported on, and therefore the only axis
        # an Alpha document may draw its features from. Distinct from the axis
        # above, and named so no reader can take one for the other.
        selected_factor_ids=receipt.selected_factor_ids,
        factor_development_receipt_hash=receipt.receipt_hash,
        factor_development_checkpoint_hash=evidence.checkpoint_hash,
        factor_development_program_hash=evidence.program.program_hash,
        factor_development_binding_hash=evidence.execution_binding_hash,
        sector_revision=sector_revision,
        sector_coverage_hash=sector_coverage_hash,
        feature_authority_outcome_snapshot_hash=feature_authority_snapshot,
    )
    return binding, manifest_ref


__all__ = [
    "assert_distinct_roots",
    "build_alpha_development_foundation",
]
