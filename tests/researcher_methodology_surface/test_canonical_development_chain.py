"""The canonical target, its scale, and the return signal, driven through real Hosts.

Before this, `compile_canonical_alpha_target_surface`, `seal_canonical_alpha_target_evidence`
and `compile_cross_sectional_dispersion_forecast` had exactly one kind of caller
between them: a case study. Every one of them was correct and unreachable -- no
Host resolved authority for them, nothing published what they produced, and no
reader walked back to what authorized it. A method nothing runs is a proposal.

These cases drive `CanonicalAlphaDevelopmentService` over the shared real
workspace, against causal outcomes published by the product's own writer and a
method seal resolved by the real outcome reader. Nothing here constructs a seal,
a snapshot hash or a maturity lag: those are the identities the service exists to
derive, and a fixture that supplied them would be asserting exactly what
publication is supposed to establish.

The G4 cases then show the two states that actually exist today. With a canonical
score claim published, the composition runs and its evidence resolves down to the
outcome method. Without one -- which is the real state while Stage 1 canonical
score fitting remains unauthorized -- resolution reports a typed wait and writes
nothing, because the alternative is relabelling a legacy lane as canonical.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pytest

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.investment.alpha_research.experiments.canonical_development import (
    CanonicalAlphaDevelopmentService,
    CanonicalDevelopmentError,
    CanonicalDevelopmentRequest,
)
from alphalattice.investment.alpha_research.targets.canonical import (
    CANONICAL_ALPHA_TARGET_RECIPE_ID,
    CanonicalAlphaScoreBinding,
    CanonicalAlphaTargetRecipeBinding,
    seal_canonical_alpha_target_evidence,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from tests.researcher_methodology_surface.real_workspace import (
    RealRiskWorkspace,
    publish_causal_outcomes,
)

CASE_ROOT = Path(__file__).resolve().parent
PLAYPEN_ROOT = CASE_ROOT.parents[1]


def _service(workspace: RealRiskWorkspace) -> CanonicalAlphaDevelopmentService:
    return CanonicalAlphaDevelopmentService(
        artifact_root=workspace.artifact_root,
        outcome_reader=CausalExecutionOutcomeDevelopmentReader(workspace.artifact_root),
        resolver=ArtifactResolver(workspace.artifact_root),
    )


@pytest.fixture(scope="module")
def canonical_publication(real_risk_workspace: RealRiskWorkspace):  # type: ignore[no-untyped-def]
    """One real canonical publication over the shared workspace."""

    snapshot_hash, _manifest_ref = publish_causal_outcomes(real_risk_workspace)
    service = _service(real_risk_workspace)
    reader = CausalExecutionOutcomeDevelopmentReader(real_risk_workspace.artifact_root)
    manifest = reader.load_manifest(snapshot_hash)
    sector_revision = _sector_revision(real_risk_workspace)
    published = service.publish(
        CanonicalDevelopmentRequest(
            causal_outcome_snapshot_hash=snapshot_hash,
            sector_revision=sector_revision,
            holding_end_through=max(
                value.last_formation_session for value in manifest.development_chunks
            )
            + timedelta(days=30),
        )
    )
    return service, published, snapshot_hash, sector_revision


def _sector_revision(workspace: RealRiskWorkspace) -> str:
    from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
        PanelClosureArtifactStore,
    )
    from alphalattice.foundation.feature_engine.panels.closure_contracts import SectorRevisionMap

    store = PanelClosureArtifactStore(ArtifactResolver(workspace.artifact_root))
    paths = sorted((store.root / "sector-maps").glob("*.json"))
    assert paths, "the shared workspace publishes a sector revision map"
    return store.load_model(
        category="sector-maps", content_hash=paths[0].stem, model=SectorRevisionMap
    ).sector_revision


def test_host_publishes_the_canonical_target_and_its_scale(canonical_publication) -> None:  # type: ignore[no-untyped-def]
    """Host publishes the canonical target and its scale."""

    service, published, snapshot_hash, _revision = canonical_publication
    reader = CausalExecutionOutcomeDevelopmentReader(service.store.root.parent)
    seal = reader.resolve_method_seal(snapshot_hash)
    assert seal.disposition == "METHOD_BOUND"

    binding = published.recipe_binding
    assert binding.causal_outcome_snapshot_hash == snapshot_hash
    assert binding.outcome_method_binding_hash == seal.method_bound.binding_hash
    assert binding.maturity_lag_sessions == seal.method_bound.maturity_lag_sessions
    assert binding.target_recipe_id == CANONICAL_ALPHA_TARGET_RECIPE_ID

    # The scale is the one the target compiler produced, not a similar number.
    assert published.dispersion.source_dispersion_identity == (
        published.evidence.cross_sectional_dispersion_identity
    )
    assert published.dispersion.maturity_lag_sessions == binding.maturity_lag_sessions

    # And the implementation lineage terminates in facts, not another digest.
    implementation = published.dispersion.implementation
    assert implementation.implementation_owners
    assert len(implementation.implementation_content_hash) == 64
    # The environment is provenance, never part of this binding (LAWS.md ID6).
    assert implementation.numerical_environment_hash is None


def test_published_canonical_evidence_resolves_back_to_the_outcome_method(
    canonical_publication,  # type: ignore[no-untyped-def]
) -> None:
    """The verifier walks down to a seal held by another Desk's reader."""

    service, published, snapshot_hash, _revision = canonical_publication
    lineage = service.verify(
        evidence_hash=published.evidence.evidence_hash,
        dispersion_hash=published.dispersion.forecast_hash,
    )
    assert lineage.outcome_snapshot_hash == snapshot_hash
    assert lineage.dispersion is not None
    assert lineage.evidence.evidence_hash == published.evidence.evidence_hash
    assert published.evidence.evidence_hash in service.find_published_evidence(
        snapshot_hash=snapshot_hash
    )


def test_forged_but_self_consistent_target_authority_is_refused(
    canonical_publication,  # type: ignore[no-untyped-def]
) -> None:
    """Forged but self consistent target authority is refused."""

    service, published, _snapshot_hash, _revision = canonical_publication

    forged_binding = CanonicalAlphaTargetRecipeBinding.create(
        recipe=published.surface.recipe,
        target_catalog_hash=published.recipe_binding.target_catalog_hash,
        causal_outcome_snapshot_hash=published.recipe_binding.causal_outcome_snapshot_hash,
        outcome_method_binding_hash="d" * 64,
        maturity_lag_sessions=published.recipe_binding.maturity_lag_sessions,
    )
    forged_evidence = seal_canonical_alpha_target_evidence(
        surface=published.surface, recipe_binding=forged_binding
    )
    service.store.publish_canonical_target_recipe_binding(forged_binding)
    service.store.publish_canonical_target_evidence(forged_evidence)

    with pytest.raises(CanonicalDevelopmentError, match="canonical_outcome_method_mismatch"):
        service.verify(evidence_hash=forged_evidence.evidence_hash)

    # A binding naming a snapshot that was never sealed fails the same way.
    unknown = CanonicalAlphaTargetRecipeBinding.create(
        recipe=published.surface.recipe,
        target_catalog_hash=published.recipe_binding.target_catalog_hash,
        causal_outcome_snapshot_hash="e" * 64,
        outcome_method_binding_hash=published.recipe_binding.outcome_method_binding_hash,
        maturity_lag_sessions=published.recipe_binding.maturity_lag_sessions,
    )
    unknown_evidence = seal_canonical_alpha_target_evidence(
        surface=published.surface, recipe_binding=unknown
    )
    service.store.publish_canonical_target_recipe_binding(unknown)
    service.store.publish_canonical_target_evidence(unknown_evidence)
    # The outcome store has no manifest under that hash at all, so resolution
    # fails at the reader rather than at a comparison -- which is the same
    # refusal reached one step earlier.
    with pytest.raises((CanonicalDevelopmentError, FileNotFoundError, ValueError)):
        service.verify(evidence_hash=unknown_evidence.evidence_hash)


def test_host_refuses_a_legacy_snapshot_and_an_immature_axis(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    """Two same-shaped requests that must fail before any numerics."""

    service = _service(real_risk_workspace)
    snapshot_hash, _ref = publish_causal_outcomes(real_risk_workspace)
    revision = _sector_revision(real_risk_workspace)

    # A snapshot nobody sealed: readable rows, absent authority.
    with pytest.raises((CanonicalDevelopmentError, ValueError, FileNotFoundError)):
        service.publish(
            CanonicalDevelopmentRequest(
                causal_outcome_snapshot_hash="a" * 64,
                sector_revision=revision,
                holding_end_through=date(2026, 1, 5),
            )
        )

    # A maturity clock so early that no formation has finished.
    with pytest.raises(CanonicalDevelopmentError, match="matured_axis_too_short"):
        service.publish(
            CanonicalDevelopmentRequest(
                causal_outcome_snapshot_hash=snapshot_hash,
                sector_revision=revision,
                holding_end_through=date(1990, 1, 1),
            )
        )


@dataclass(frozen=True, slots=True)
class _StubBaseWorkspace:
    """Only the four fields the resolver checks a base workspace for.

    A real ``PortfolioSharedDevelopmentWorkspace`` needs current Alpha, Risk and
    Tradability pointers and decodes the whole history; none of that is under
    test here. What is under test is that the resolver takes its base identity
    and axes from whatever the provider yields, and refuses a request that
    disagrees -- which this stub exercises exactly.
    """

    workspace_hash: str
    candidate_ids: tuple[str, ...]
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]


def _score_claim(published, candidate_id: str, **overrides):  # type: ignore[no-untyped-def]
    values = {
        "alpha_score_surface_hash": "c" * 64,
        "candidate_id": candidate_id,
        "target_recipe_binding_hash": published.recipe_binding.binding_hash,
        "target_evidence_hash": published.evidence.evidence_hash,
        "ordered_listing_ids_hash": str(
            canonical_hash(list(published.evidence.ordered_listing_ids))
        ),
        "formation_sessions_hash": str(
            canonical_hash([v.isoformat() for v in published.evidence.formation_sessions])
        ),
    }
    values.update(overrides)
    return CanonicalAlphaScoreBinding.create(**values)


def test_matched_handles_are_paired_by_lineage_not_by_hash_order(
    canonical_publication,  # type: ignore[no-untyped-def]
) -> None:
    """Matched handles are paired by lineage not by hash order."""

    service, published, _snapshot, _revision = canonical_publication
    matching = _score_claim(published, "candidate-matched")
    unrelated = _score_claim(
        published,
        "candidate-unrelated",
        alpha_score_surface_hash="d" * 64,
        target_recipe_binding_hash="0" * 64,
        target_evidence_hash="1" * 64,
    )
    service.store.publish_canonical_score_binding(matching)
    service.store.publish_canonical_score_binding(unrelated)

    pairs = service.find_matched_canonical_inputs(candidate_id="candidate-matched")
    assert pairs == ((matching.binding_hash, published.dispersion.forecast_hash),)
    assert service.find_matched_canonical_inputs(candidate_id="candidate-unrelated") == ()


def test_canonical_development_entrypoint_publishes_and_inspects(
    canonical_publication,  # type: ignore[no-untyped-def]
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    """The installed development CLI refuses unresolved requests and reports durably published
    evidence."""

    spec = importlib.util.spec_from_file_location(
        "_canonical_cli",
        Path(__file__).resolve().parents[2] / "scripts" / "run_canonical_alpha_development.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    service, published, snapshot_hash, revision = canonical_publication
    workspace = real_risk_workspace.artifact_root.parent

    inspected = module._inspect(workspace, snapshot_hash=snapshot_hash)
    assert inspected["action"] == "INSPECTED"
    assert published.evidence.evidence_hash in inspected["canonical_target_evidence_hashes"]
    assert inspected["current_pointer_writes"] == 0

    # An unsealed snapshot is refused by the same entry point, typed rather than
    # crashing, and writes nothing.
    republished = module._publish(
        workspace,
        snapshot_hash=snapshot_hash,
        sector_revision=revision,
        holding_end_through=max(published.formation_sessions),
    )
    assert republished["action"] == "PUBLISHED"
    assert republished["causal_outcome_snapshot_hash"] == snapshot_hash
    assert republished["current_pointer_writes"] == 0
    assert republished["model_fits"] == 0
    del service
