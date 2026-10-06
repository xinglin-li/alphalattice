"""Focused correctness gates for the 1D Factor Research target and split."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from alphalattice.foundation.causal_outcomes.execution.contracts import (
    CausalExecutionOutcomeManifest,
    CausalExecutionOutcomeMarker,
)
from alphalattice.foundation.causal_outcomes.execution.methods import (
    ExecutionOutcomeMethodBinding,
    ExecutionOutcomeMethodRecipe,
    ExecutionOutcomeMethodSeal,
    build_execution_outcome_method_binding,
    build_execution_outcome_method_seal_marker,
    build_five_session_recipe,
    build_installed_execution_outcome_method_catalog,
    build_installed_execution_outcome_publication_policy,
    build_one_session_recipe,
)
from alphalattice.foundation.factor_research.evaluation.oos_evidence import (
    FactorEvidenceClassification,
    _classification,
    build_factor_evidence_policy,
    compute_factor_oos_evidence,
)
from alphalattice.foundation.factor_research.evaluation.rank_correlation import (
    average_ranks_over_rows,
    ordinal_percentile_columns,
    pairwise_rank_correlations,
    period_pair_correlation_matrix,
    stable_column_orders,
)
from alphalattice.foundation.factor_research.evaluation.redundancy import (
    FactorRedundancyBoundaryError,
    build_factor_redundancy_policy,
    build_factor_redundancy_structure,
)
from alphalattice.foundation.factor_research.inputs.execution_target import (
    FactorTargetBoundaryError,
    FactorTargetPolicy,
    build_factor_target_policy,
    compile_factor_target_surface,
)
from alphalattice.foundation.factor_research.inputs.research_input import (
    FactorResearchCandidateRole,
    FactorResearchInputBoundaryError,
    FactorResearchProposalChoice,
    build_factor_research_proposal,
    compile_factor_horizon_research_input,
)
from alphalattice.foundation.factor_research.programs.walk_forward import (
    FactorWalkForwardBoundaryError,
    build_factor_walk_forward_policy,
    compile_factor_walk_forward_plan,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.validation.screening_statistics import (
    average_ranks,
    gross_decile_spread,
    spearman_rank_correlation,
)
from tests.factor_research_pipeline_correctness.walk_forward_source import (
    _manifest,
    _source_table,
)


def _surface(
    formations: tuple[date, ...], listing_ids: tuple[str, ...] = ("listing-a", "listing-b")
):
    return compile_factor_target_surface(
        source_table=_source_table(formations, listing_ids),
        source_manifest=_manifest(listing_ids),
        source_manifest_ref="playpen://outcomes/manifest",
        policy=build_factor_target_policy(),
    )


def test_target_preserves_simple_economics_and_projects_log_fit_target() -> None:
    formation = date(2026, 1, 2)
    surface = _surface((formation,))

    simple = np.asarray(surface.table["simple_economic_return"].to_numpy())
    fit_target = np.asarray(surface.table["fit_target"].to_numpy())

    assert np.array_equal(fit_target, np.log1p(simple))
    assert surface.quality.valid_target_count == 2
    assert surface.quality.typed_missing_count == 0
    assert surface.manifest.formation_count == 1
    assert surface.manifest.listing_count == 2


def test_target_keeps_unexecutable_rows_as_typed_missing() -> None:
    listings = ("listing-a", "listing-b")
    source = _source_table((date(2026, 1, 2),), listings, missing_listing="listing-b")
    surface = compile_factor_target_surface(
        source_table=source,
        source_manifest=_manifest(listings),
        source_manifest_ref="playpen://outcomes/manifest",
        policy=build_factor_target_policy(),
    )

    assert surface.quality.valid_target_count == 1
    assert surface.quality.typed_missing_count == 1
    assert surface.quality.missing_reasons[0].reason == "ENTRY_NOT_EXECUTABLE"
    assert surface.table["fit_target"].null_count == 1


def test_target_rejects_duplicate_rows_and_formula_tamper() -> None:
    formation = date(2026, 1, 2)
    source = _source_table((formation,), ("listing-a",))
    duplicate = pa.concat_tables([source, source])
    with pytest.raises(FactorTargetBoundaryError, match="target_duplicate_row"):
        compile_factor_target_surface(
            source_table=duplicate,
            source_manifest=_manifest(("listing-a",)),
            source_manifest_ref="playpen://outcomes/manifest",
            policy=build_factor_target_policy(),
        )

    tampered = source.set_column(
        source.schema.get_field_index("simple_return"),
        "simple_return",
        pa.array([0.5], type=pa.float64()),
    )
    with pytest.raises(FactorTargetBoundaryError, match="target_formula_mismatch"):
        compile_factor_target_surface(
            source_table=tampered,
            source_manifest=_manifest(("listing-a",)),
            source_manifest_ref="playpen://outcomes/manifest",
            policy=build_factor_target_policy(),
        )


def test_target_policy_identity_rejects_tamper() -> None:
    policy = build_factor_target_policy()
    with pytest.raises(ValueError, match="policy hash"):
        FactorTargetPolicy(**{**policy.model_dump(), "policy_hash": "0" * 64})


def _binding(
    manifest: CausalExecutionOutcomeManifest,
    recipe: ExecutionOutcomeMethodRecipe,
    *,
    schedule_hash: str | None = None,
    catalog_hash: str | None = None,
    publication_policy_hash: str | None = None,
) -> ExecutionOutcomeMethodBinding:
    """Build a seal the way the Host would.

    The catalog and policy hashes default to the installed ones rather than to
    arbitrary values: a fixture that invents them would be asserting exactly the
    authority the consumer is supposed to re-derive, and the earlier version of
    this helper did, which is how a binding citing an uninstalled catalog was
    able to reach numerical work.
    """

    catalog = build_installed_execution_outcome_method_catalog()
    return build_execution_outcome_method_binding(
        recipe=recipe,
        catalog_hash=catalog_hash or catalog.binding.catalog_hash,
        publication_policy_hash=(
            publication_policy_hash
            or build_installed_execution_outcome_publication_policy().policy_hash
        ),
        snapshot_hash=manifest.snapshot_hash,
        schedule_hash=schedule_hash or manifest.schedule_hash,
        ordered_session_triples_hash=manifest.ordered_session_triples_hash,
        source_watermark_hash="7" * 64,
        action_lineage="6" * 64,
    )


def _seal(
    manifest: CausalExecutionOutcomeManifest,
    recipe: ExecutionOutcomeMethodRecipe,
    *,
    manifest_ref: str = "playpen://outcomes/manifest",
    schedule_hash: str | None = None,
    catalog_hash: str | None = None,
    publication_policy_hash: str | None = None,
) -> ExecutionOutcomeMethodSeal:
    """Build the resolved seal graph the way an authoritative reader would.

    The compiler re-derives everything from the installed catalog, policy and
    the manifest it is handed, so an honest graph passes without any store
    behind it and a forged one fails on the same whole-model comparison a real
    resolution would.
    """

    binding = _binding(
        manifest,
        recipe,
        schedule_hash=schedule_hash,
        catalog_hash=catalog_hash,
        publication_policy_hash=publication_policy_hash,
    )
    marker_values = {
        "kind": "CausalExecutionOutcomeMarker",
        "snapshot_hash": manifest.snapshot_hash,
        "manifest_ref": manifest_ref,
        "schedule_hash": manifest.schedule_hash,
        "listing_set_hash": manifest.listing_set_hash,
    }
    outcome_marker = CausalExecutionOutcomeMarker(
        **marker_values, marker_hash=canonical_hash(marker_values)
    )
    seal_marker = build_execution_outcome_method_seal_marker(
        outcome_marker_hash=outcome_marker.marker_hash,
        outcome_marker_ref="playpen://outcomes/markers",
        snapshot_hash=manifest.snapshot_hash,
        manifest_ref=manifest_ref,
        binding=binding,
        binding_ref="playpen://outcomes/method-bindings",
    )
    return ExecutionOutcomeMethodSeal(
        disposition="METHOD_BOUND",
        seal_marker=seal_marker,
        outcome_marker=outcome_marker,
        binding=binding,
    )


def test_frozen_one_session_target_policy_identity_is_unchanged() -> None:
    """The frozen policy bytes are the compatibility promise of this Gate.

    Pinned as a literal rather than recomputed, because a check that derived the
    expected value from the same fields it is verifying would pass through any
    field change at all.
    """

    assert build_factor_target_policy().policy_hash == (
        "b98fa43b091ebacb5a1b1c997f6e92ffdf9c715e87f52802e0122034d0e7c72d"
    )


def test_sealed_one_session_method_reproduces_the_legacy_default_surface() -> None:
    formations = (date(2026, 1, 2), date(2026, 1, 3))
    listings = ("listing-a", "listing-b")
    manifest = _manifest(listings)
    source = _source_table(formations, listings)

    legacy = compile_factor_target_surface(
        source_table=source,
        source_manifest=manifest,
        source_manifest_ref="playpen://outcomes/manifest",
        policy=build_factor_target_policy(),
    )
    sealed = compile_factor_target_surface(
        source_table=source,
        source_manifest=manifest,
        source_manifest_ref="playpen://outcomes/manifest",
        policy=build_factor_target_policy(),
        outcome_method=_seal(manifest, build_one_session_recipe()),
    )

    assert sealed.manifest.surface_hash == legacy.manifest.surface_hash
    assert sealed.quality.quality_hash == legacy.quality.quality_hash


def test_five_session_source_fails_on_method_before_any_numerical_work() -> None:
    """A longer-span surface is refused for being the wrong method, not the wrong shape.

    The two failures are deliberately distinct. A five-session seal cannot even
    bind to a frozen one-session manifest: the manifest's formula is a
    single-value Literal, so the seal's child graph disagrees with the surface
    it claims and is refused before a single column is read. Handed no seal at
    all, the same rows are refused later by the span the legacy default
    asserts.
    """

    listings = ("listing-a",)
    manifest = _manifest(listings)
    source = _source_table((date(2026, 1, 2),), listings)
    span_six = source.set_column(
        source.schema.get_field_index("actual_session_span"),
        "actual_session_span",
        pa.array([6] * source.num_rows, type=source.schema.field("actual_session_span").type),
    )

    with pytest.raises(FactorTargetBoundaryError, match="target_outcome_method_unbound"):
        compile_factor_target_surface(
            source_table=span_six,
            source_manifest=manifest,
            source_manifest_ref="playpen://outcomes/manifest",
            policy=build_factor_target_policy(),
            outcome_method=_seal(manifest, build_five_session_recipe()),
        )

    with pytest.raises(FactorTargetBoundaryError, match="target_horizon_mismatch"):
        compile_factor_target_surface(
            source_table=span_six,
            source_manifest=manifest,
            source_manifest_ref="playpen://outcomes/manifest",
            policy=build_factor_target_policy(),
        )


def test_seal_from_another_snapshot_is_refused() -> None:
    """A well-formed seal is not authority over rows it does not describe."""

    listings = ("listing-a",)
    manifest = _manifest(listings)
    source = _source_table((date(2026, 1, 2),), listings)
    foreign = _seal(manifest, build_one_session_recipe(), schedule_hash="2" * 64)

    with pytest.raises(FactorTargetBoundaryError, match="target_outcome_method_unbound"):
        compile_factor_target_surface(
            source_table=source,
            source_manifest=manifest,
            source_manifest_ref="playpen://outcomes/manifest",
            policy=build_factor_target_policy(),
            outcome_method=foreign,
        )


def test_seal_citing_an_uninstalled_catalog_or_policy_cannot_reach_numerical_work() -> None:
    """The consumer re-derives authority instead of comparing a subset of fields.

    Both forgeries below are internally consistent and agree with the manifest
    on snapshot, schedule and session axis -- everything a field comparison
    looked at. They differ only in claiming a catalog or a publication policy
    the Host never installed.
    """

    listings = ("listing-a",)
    manifest = _manifest(listings)
    source = _source_table((date(2026, 1, 2),), listings)
    recipe = build_one_session_recipe()

    for forged in (
        _seal(manifest, recipe, catalog_hash="9" * 64),
        _seal(manifest, recipe, publication_policy_hash="8" * 64),
    ):
        assert forged.method_bound.snapshot_hash == manifest.snapshot_hash
        assert forged.method_bound.schedule_hash == manifest.schedule_hash
        with pytest.raises(FactorTargetBoundaryError, match="target_outcome_method_unbound"):
            compile_factor_target_surface(
                source_table=source,
                source_manifest=manifest,
                source_manifest_ref="playpen://outcomes/manifest",
                policy=build_factor_target_policy(),
                outcome_method=forged,
            )


def test_bare_binding_without_terminal_seal_is_not_method_authority() -> None:
    """A raw binding is a well-formed file, not a resolved publication.

    Before this Gate the method-aware route accepted the binding object alone,
    which meant anything that could construct one -- honestly or otherwise --
    spoke with a publication's voice. The route now requires the resolved seal
    graph, so the same Host-perfect binding, handed in bare, is refused with a
    typed error rather than being promoted.
    """

    listings = ("listing-a",)
    manifest = _manifest(listings)
    source = _source_table((date(2026, 1, 2),), listings)
    binding = _binding(manifest, build_one_session_recipe())

    with pytest.raises(FactorTargetBoundaryError, match="target_outcome_method_unbound"):
        compile_factor_target_surface(
            source_table=source,
            source_manifest=manifest,
            source_manifest_ref="playpen://outcomes/manifest",
            policy=build_factor_target_policy(),
            outcome_method=binding,  # type: ignore[arg-type]
        )


def test_legacy_readback_disposition_is_refused_as_method_authority() -> None:
    """``LEGACY_READBACK_ONLY`` says rows are readable, not that a method is bound."""

    listings = ("listing-a",)
    manifest = _manifest(listings)
    source = _source_table((date(2026, 1, 2),), listings)

    with pytest.raises(FactorTargetBoundaryError, match="target_outcome_method_unbound"):
        compile_factor_target_surface(
            source_table=source,
            source_manifest=manifest,
            source_manifest_ref="playpen://outcomes/manifest",
            policy=build_factor_target_policy(),
            outcome_method=ExecutionOutcomeMethodSeal(disposition="LEGACY_READBACK_ONLY"),
        )


def test_seal_naming_a_different_manifest_ref_is_refused() -> None:
    """The terminal marker must name the exact manifest ref the compiler was handed."""

    listings = ("listing-a",)
    manifest = _manifest(listings)
    source = _source_table((date(2026, 1, 2),), listings)
    foreign_ref = _seal(
        manifest,
        build_one_session_recipe(),
        manifest_ref="playpen://outcomes/another-manifest",
    )

    with pytest.raises(FactorTargetBoundaryError, match="target_outcome_method_unbound"):
        compile_factor_target_surface(
            source_table=source,
            source_manifest=manifest,
            source_manifest_ref="playpen://outcomes/manifest",
            policy=build_factor_target_policy(),
            outcome_method=foreign_ref,
        )


def test_shifted_schedule_axis_is_refused_before_target_compilation() -> None:
    """Same shape, same span, shifted session: still refused.

    Nothing about the surface looks irregular row by row; what fails is that one
    row's entry session disagrees with the rest of its own formation group.
    """

    listings = ("listing-a", "listing-b")
    manifest = _manifest(listings)
    source = _source_table((date(2026, 1, 2),), listings)
    shifted_entry = source["entry_session"].to_pylist()
    shifted_entry[0] = shifted_entry[0] + timedelta(days=1)
    shifted = source.set_column(
        source.schema.get_field_index("entry_session"),
        "entry_session",
        pa.array(shifted_entry, type=source.schema.field("entry_session").type),
    )

    with pytest.raises(FactorTargetBoundaryError, match="target_schedule_axis_mismatch"):
        compile_factor_target_surface(
            source_table=shifted,
            source_manifest=manifest,
            source_manifest_ref="playpen://outcomes/manifest",
            policy=build_factor_target_policy(),
        )


def test_walk_forward_uses_all_complete_folds_and_exact_one_day_purge() -> None:
    panel_sessions = tuple(date(2026, 1, 2) + timedelta(days=index) for index in range(11))
    surface = _surface(panel_sessions[:8], ("listing-a",))
    policy = build_factor_walk_forward_policy(
        train_sessions=3,
        validation_sessions=2,
        step_sessions=2,
        sealed_holdout_sessions=2,
        minimum_folds=2,
    )

    plan = compile_factor_walk_forward_plan(
        panel_sessions=panel_sessions,
        target_surface=surface,
        frozen_at=datetime(2027, 1, 1, tzinfo=UTC),
        policy=policy,
    )

    assert len(plan.formal_split.windows) == 2
    assert all(len(window.purge_sessions) == 1 for window in plan.formal_split.windows)
    assert plan.consumed_development_session_count == 8


def test_walk_forward_rejects_missing_consumed_label() -> None:
    panel_sessions = tuple(date(2026, 1, 2) + timedelta(days=index) for index in range(11))
    surface = _surface(panel_sessions[:7], ("listing-a",))
    policy = build_factor_walk_forward_policy(
        train_sessions=3,
        validation_sessions=2,
        step_sessions=2,
        sealed_holdout_sessions=2,
        minimum_folds=2,
    )
    with pytest.raises(FactorWalkForwardBoundaryError, match="label_availability_incomplete"):
        compile_factor_walk_forward_plan(
            panel_sessions=panel_sessions,
            target_surface=surface,
            frozen_at=datetime(2027, 1, 1, tzinfo=UTC),
            policy=policy,
        )


def test_oos_evidence_freezes_training_orientation_and_uses_full_by_family() -> None:
    panel_sessions = tuple(date(2026, 1, 2) + timedelta(days=index) for index in range(14))
    consumed_sessions = (*panel_sessions[:4], *panel_sessions[5:11])
    listing_ids = tuple(f"listing-{index:03d}" for index in range(120))
    target_surface = _surface(consumed_sessions, listing_ids)
    walk_policy = build_factor_walk_forward_policy(
        train_sessions=4,
        validation_sessions=6,
        step_sessions=6,
        sealed_holdout_sessions=2,
        minimum_folds=1,
    )
    plan = compile_factor_walk_forward_plan(
        panel_sessions=panel_sessions,
        target_surface=target_surface,
        frozen_at=datetime(2027, 1, 1, tzinfo=UTC),
        policy=walk_policy,
    )
    rows: list[dict[str, object]] = []
    for session in consumed_sessions:
        is_validation = session in panel_sessions[5:11]
        for listing_index, listing_id in enumerate(listing_ids):
            stable_score = -float(listing_index)
            rows.append(
                {
                    "session_date": session,
                    "listing_id": listing_id,
                    "factor_missing": None,
                    "factor_regime_flip": -stable_score if is_validation else stable_score,
                    "factor_stable": stable_score + (listing_index % 7) * 0.001,
                    "factor_stable_peer": stable_score + (listing_index % 7) * 0.001,
                }
            )
    factor_ids = (
        "factor_missing",
        "factor_regime_flip",
        "factor_stable",
        "factor_stable_peer",
    )
    report = compute_factor_oos_evidence(
        feature_table=pa.Table.from_pylist(rows),
        feature_panel_snapshot_hash="9" * 64,
        feature_panel_manifest_ref="playpen://feature-panel/manifest",
        target_surface=target_surface,
        walk_forward_plan=plan,
        factor_ids=factor_ids,
        policy=build_factor_evidence_policy(minimum_cross_section_observations=100),
    )

    by_factor = {item.factor_id: item for item in report.items}
    flipped = by_factor["factor_regime_flip"]
    assert report.hypothesis_count == 4
    assert flipped.fold_evidence[0].orientation == 1
    assert flipped.mean_oriented_rank_ic is not None
    assert flipped.mean_oriented_rank_ic < 0.0
    assert flipped.mean_oriented_simple_spread is not None
    assert flipped.mean_oriented_simple_spread < 0.0
    assert flipped.classification is FactorEvidenceClassification.NO_DETECTABLE_EFFECT
    assert flipped.reason_codes == ("NONPOSITIVE_DIRECTION_NOT_BY_CONFIRMED",)
    assert (
        by_factor["factor_missing"].classification
        is FactorEvidenceClassification.INSUFFICIENT_EVIDENCE
    )
    assert set(item.factor_id for item in report.items) == set(factor_ids)

    redundancy_rows = [
        {**row, "factor_missing": float(index % len(listing_ids))} for index, row in enumerate(rows)
    ]
    redundancy = build_factor_redundancy_structure(
        feature_table=pa.Table.from_pylist(redundancy_rows),
        feature_panel_snapshot_hash="9" * 64,
        feature_panel_manifest_ref="playpen://feature-panel/manifest",
        factor_ids=factor_ids,
        policy=build_factor_redundancy_policy(
            minimum_common_listings=100,
            minimum_formal_periods=3,
            distance_cut=0.1,
        ),
    )
    cluster_by_factor = {
        factor_id: cluster
        for cluster in redundancy.clusters
        for factor_id in cluster.member_factor_ids
    }
    stable = by_factor["factor_stable"]
    stable_peer = by_factor["factor_stable_peer"]
    assert stable.classification is FactorEvidenceClassification.POSITIVE_OOS_EVIDENCE
    assert stable_peer.classification is FactorEvidenceClassification.POSITIVE_OOS_EVIDENCE
    assert stable.reason_codes == ("DIRECTIONALLY_POSITIVE_NOT_BY_CONFIRMED",)
    assert (
        cluster_by_factor["factor_stable"].cluster_id
        == cluster_by_factor["factor_stable_peer"].cluster_id
    )
    choice = FactorResearchProposalChoice(
        factor_id="factor_stable",
        role=FactorResearchCandidateRole.CORE,
        evidence_hash=stable.evidence_hash,
        cluster_id=cluster_by_factor["factor_stable"].cluster_id,
        rationale="Stable favorable direction; BY strength remains explicitly unconfirmed.",
    )
    peer_choice = FactorResearchProposalChoice(
        factor_id="factor_stable_peer",
        role=FactorResearchCandidateRole.CORE,
        evidence_hash=stable_peer.evidence_hash,
        cluster_id=cluster_by_factor["factor_stable_peer"].cluster_id,
        rationale="Same cluster but independently retained for downstream joint modeling.",
    )
    proposal = build_factor_research_proposal(
        evidence_report_hash=report.report_hash,
        redundancy_structure_hash=redundancy.structure_hash,
        choices=(choice, peer_choice),
        limitations_acknowledged=("CURRENT_UNIVERSE_RESEARCH_ONLY",),
    )
    research_input = compile_factor_horizon_research_input(
        evidence=report,
        redundancy=redundancy,
        proposal=proposal,
        limitations=("CURRENT_UNIVERSE_RESEARCH_ONLY",),
    )
    assert research_input.core_factor_ids == ("factor_stable", "factor_stable_peer")
    assert research_input.conditional_factor_ids == ()
    assert len(research_input.complete_factor_evidence_refs) == 4

    invalid_choice = FactorResearchProposalChoice(
        factor_id="factor_regime_flip",
        role=FactorResearchCandidateRole.CONDITIONAL,
        evidence_hash=flipped.evidence_hash,
        cluster_id=cluster_by_factor["factor_regime_flip"].cluster_id,
        rationale="Attempt to revive statistically undetectable nonpositive evidence.",
    )
    invalid_proposal = build_factor_research_proposal(
        evidence_report_hash=report.report_hash,
        redundancy_structure_hash=redundancy.structure_hash,
        choices=(invalid_choice,),
        limitations_acknowledged=("CURRENT_UNIVERSE_RESEARCH_ONLY",),
    )
    with pytest.raises(
        FactorResearchInputBoundaryError,
        match="conditional_evidence_not_mixed",
    ):
        compile_factor_horizon_research_input(
            evidence=report,
            redundancy=redundancy,
            proposal=invalid_proposal,
            limitations=("CURRENT_UNIVERSE_RESEARCH_ONLY",),
        )


@pytest.mark.parametrize(
    ("mean_ic", "mean_spread", "q_value", "classification", "reason"),
    (
        (
            0.01,
            0.02,
            0.01,
            FactorEvidenceClassification.POSITIVE_OOS_EVIDENCE,
            "DIRECTIONALLY_POSITIVE_BY_CONFIRMED",
        ),
        (
            -0.01,
            -0.02,
            0.01,
            FactorEvidenceClassification.NEGATIVE_OOS_EVIDENCE,
            "DIRECTIONALLY_NEGATIVE_BY_CONFIRMED",
        ),
        (
            -0.01,
            -0.02,
            0.50,
            FactorEvidenceClassification.NO_DETECTABLE_EFFECT,
            "NONPOSITIVE_DIRECTION_NOT_BY_CONFIRMED",
        ),
        (
            0.01,
            -0.02,
            0.50,
            FactorEvidenceClassification.MIXED_OOS_EVIDENCE,
            "MIXED_OOS_DIRECTION",
        ),
    ),
)
def test_evidence_classification_distinguishes_direction_from_by_strength(
    mean_ic: float,
    mean_spread: float,
    q_value: float,
    classification: FactorEvidenceClassification,
    reason: str,
) -> None:
    computation = SimpleNamespace(
        folds=(SimpleNamespace(orientation=1),),
        validation_ics=(mean_ic,),
        validation_spreads=(mean_spread,),
    )
    observed, reasons = _classification(
        computation=computation,
        mean_ic=mean_ic,
        mean_spread=mean_spread,
        q_value=q_value,
        coverage=1.0,
        policy=build_factor_evidence_policy(),
    )
    assert observed is classification
    assert reasons == (reason,)


def _bits(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def _ordinal_percentiles_one_column(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The one-column ordinal percentile rule: rank the finite values alone, stably."""

    finite = np.isfinite(values)
    expected = np.full(values.shape, np.nan)
    count = int(finite.sum())
    if count:
        order = np.argsort(values[finite], kind="stable")
        ordinal = np.empty(count)
        ordinal[order] = np.arange(count, dtype=np.float64)
        expected[finite] = (ordinal + 0.5) / count
    return expected, finite


def test_ordinal_percentiles_rank_only_the_finite_values_of_a_column() -> None:
    """The counterexample: a ``-inf`` sorts before the finite values and must not
    take an ordinal position from them; ``+inf`` and NaN sort after them and
    must not either. Two sessions whose finite scores and order are identical
    give a zero turnover whatever unavailable value the other rows carry."""

    percentiles, finite = ordinal_percentile_columns(np.asarray([[-np.inf], [1.0], [2.0]]))
    assert np.array_equal(finite[:, 0], [False, True, True])
    assert np.array_equal(percentiles[1:, 0], [0.25, 0.75]) and np.isnan(percentiles[0, 0])
    oriented, _finite = ordinal_percentile_columns(np.asarray([[np.inf], [1.0], [2.0]]) * -1.0)
    assert np.array_equal(oriented[1:, 0], [0.75, 0.25]) and np.isnan(oriented[0, 0])
    previous, previous_finite = ordinal_percentile_columns(np.asarray([[-np.inf], [1.0], [2.0]]))
    current, current_finite = ordinal_percentile_columns(np.asarray([[np.nan], [1.0], [2.0]]))
    common = previous_finite[:, 0] & current_finite[:, 0]
    assert float(np.mean(np.abs(current[common, 0] - previous[common, 0]))) == 0.0
    for column in (
        np.asarray([np.inf, -np.inf, np.nan]),
        np.asarray([-np.inf, 5.0]),
        np.asarray([3.0, 3.0, -np.inf, 3.0]),
        np.asarray([-0.0, 0.0, -np.inf, np.inf, np.nan, 0.0]),
    ):
        got, got_finite = ordinal_percentile_columns(column[:, None])
        expected, expected_finite = _ordinal_percentiles_one_column(column)
        assert np.array_equal(got_finite[:, 0], expected_finite)
        assert np.array_equal(_bits(got[expected_finite, 0]), _bits(expected[expected_finite]))
        assert np.isnan(got[~expected_finite, 0]).all()


def test_ranks_read_off_one_column_order_are_the_formal_ranks_over_any_row_mask() -> None:
    """The per-session sort is done once; every masked rank vector must be the
    formal owner's, bit for bit, with ties, signed zeros, NaN, ``+inf`` and
    ``-inf`` rows (positive and negative orientation) and any mask."""

    rng = np.random.default_rng(20260915)
    for trial in range(200):
        rows, columns = int(rng.integers(2, 70)), int(rng.integers(1, 9))
        block = np.round(rng.normal(size=(rows, columns)) * 3.0, int(rng.integers(0, 2)))
        block[rng.random(block.shape) < 0.15] = np.nan
        block[rng.random(block.shape) < 0.05] = np.inf
        block[rng.random(block.shape) < 0.05] = -np.inf
        block[rng.random(rows) < 0.2, 0] = -0.0
        if trial % 3 == 0:
            block[:, -1] = 1.0
        if trial % 4 == 0:
            block = block * -1.0
        orders = stable_column_orders(block)
        finite = np.isfinite(block)
        width = int(rng.integers(1, columns + 1))
        selected = sorted(rng.choice(columns, size=width, replace=False))
        mask = finite[:, selected].all(axis=1) & (rng.random(rows) < 0.85)
        if not mask.any():
            continue
        ranked = average_ranks_over_rows(block, orders, mask, selected)
        assert ranked.shape == (len(selected), int(mask.sum())) and ranked.flags.c_contiguous
        for position, column in enumerate(selected):
            expected_ranks = average_ranks(block[mask, column])
            assert np.array_equal(_bits(ranked[position]), _bits(expected_ranks))
        percentiles, percentile_finite = ordinal_percentile_columns(block)
        for column in range(columns):
            expected, expected_finite = _ordinal_percentiles_one_column(block[:, column])
            assert np.array_equal(percentile_finite[:, column], expected_finite)
            assert np.array_equal(expected_finite, finite[:, column])
            assert np.array_equal(
                _bits(percentiles[expected_finite, column]), _bits(expected[expected_finite])
            )
            assert np.isnan(percentiles[~expected_finite, column]).all()


def test_period_pair_matrix_keeps_both_pairwise_formulas_bit_for_bit() -> None:
    """Pairs sharing a finite-row mask keep the matrix formula the pairwise owner
    always used for them (bitwise the column-stacked ranks); pairs across two
    masks keep the per-pair formula, which is the formal Spearman exactly."""

    rng = np.random.default_rng(7)
    for _trial in range(40):
        rows, columns = 140, 7
        block = np.round(rng.normal(size=(rows, columns)), 2)
        block[rng.random(rows) < 0.05, 1] = np.nan
        block[rng.random(rows) < 0.05, 2] = np.nan
        block[np.isnan(block[:, 1]), 3] = np.nan
        finite = np.isfinite(block)
        correlations, counts = period_pair_correlation_matrix(block, minimum_common_count=100)
        for left, right in combinations(range(columns), 2):
            common = finite[:, left] & finite[:, right]
            assert counts[left, right] == int(common.sum())
            formal = spearman_rank_correlation(block[common, left], block[common, right])
            if np.array_equal(finite[:, left], finite[:, right]):
                members = [
                    column for column in range(columns) if np.array_equal(finite[:, column], common)
                ]
                stacked = pairwise_rank_correlations(
                    np.column_stack([average_ranks(block[common, member]) for member in members])
                )
                expected = stacked[members.index(left), members.index(right)]
                assert np.float64(correlations[left, right]).view(np.uint64) == np.float64(
                    expected
                ).view(np.uint64)
                assert formal is not None and abs(correlations[left, right] - formal) < 1e-12
            else:
                assert np.float64(correlations[left, right]).view(np.uint64) == np.float64(
                    formal
                ).view(np.uint64)
            assert correlations[left, right] == correlations[right, left]


def test_session_major_oos_evidence_is_the_one_factor_statistic_bit_for_bit() -> None:
    """Every fold field of every factor equals the one-factor rule computed with
    the formal owners on that factor alone: rank IC on the rows finite in both
    the factor and the target, oriented by the fold's training mean, decile
    spread on the simple return, pair coverage, and ordinal-rank turnover."""

    panel_sessions = tuple(date(2026, 1, 2) + timedelta(days=index) for index in range(14))
    consumed = (*panel_sessions[:4], *panel_sessions[5:11])
    listing_ids = tuple(f"listing-{index:03d}" for index in range(130))
    target_surface = _surface(consumed, listing_ids)
    plan = compile_factor_walk_forward_plan(
        panel_sessions=panel_sessions,
        target_surface=target_surface,
        frozen_at=datetime(2027, 1, 1, tzinfo=UTC),
        policy=build_factor_walk_forward_policy(
            train_sessions=4,
            validation_sessions=6,
            step_sessions=6,
            sealed_holdout_sessions=2,
            minimum_folds=1,
        ),
    )
    rng = np.random.default_rng(3)
    factor_ids = ("factor_flip", "factor_infinite", "factor_sparse", "factor_tied", "factor_up")
    rows: list[dict[str, object]] = []
    for session_index, session in enumerate(consumed):
        for listing_index, listing_id in enumerate(listing_ids):
            up = float(listing_index) + rng.normal() * 40.0
            # Unavailable values of every kind, on both sides of the finite
            # values, on a factor whose training orientation is negative (so
            # the oriented +inf becomes -inf) and on one whose is positive.
            draw = rng.random()
            unavailable = -np.inf if draw < 0.03 else np.inf if draw < 0.06 else None
            rows.append(
                {
                    "session_date": session,
                    "listing_id": listing_id,
                    "factor_flip": -up if session_index >= 4 else up,
                    "factor_infinite": unavailable if unavailable is not None else -up,
                    "factor_sparse": (
                        None
                        if rng.random() < 0.08
                        else (unavailable if unavailable is not None else up + rng.normal())
                    ),
                    "factor_tied": float(listing_index % 9) - 4.0,
                    "factor_up": up,
                }
            )
    policy = build_factor_evidence_policy(minimum_cross_section_observations=100)
    report = compute_factor_oos_evidence(
        feature_table=pa.Table.from_pylist(rows),
        feature_panel_snapshot_hash="9" * 64,
        feature_panel_manifest_ref="playpen://feature-panel/manifest",
        target_surface=target_surface,
        walk_forward_plan=plan,
        factor_ids=factor_ids,
        policy=policy,
    )
    by_session = {
        session: {
            "scores": np.asarray(
                [
                    [np.nan if row[factor] is None else row[factor] for factor in factor_ids]
                    for row in rows
                    if row["session_date"] == session
                ],
                dtype=np.float64,
            )
        }
        for session in consumed
    }
    fit = {
        (row["formation_session"], row["listing_id"]): (
            row["fit_target"],
            row["simple_economic_return"],
        )
        for row in target_surface.table.to_pylist()
    }
    for session in consumed:
        by_session[session]["fit"] = np.asarray(
            [
                np.nan if fit[session, lid][0] is None else fit[session, lid][0]
                for lid in listing_ids
            ],
            dtype=np.float64,
        )
        by_session[session]["simple"] = np.asarray(
            [
                np.nan if fit[session, lid][1] is None else fit[session, lid][1]
                for lid in listing_ids
            ],
            dtype=np.float64,
        )

    def rank_ic(scores: np.ndarray, targets: np.ndarray) -> float | None:
        mask = np.isfinite(scores) & np.isfinite(targets)
        if int(mask.sum()) < 100:
            return None
        return spearman_rank_correlation(scores[mask], targets[mask])

    def percentiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        finite = np.isfinite(values)
        result = np.full(values.shape, np.nan)
        count = int(finite.sum())
        if count:
            order = np.argsort(values[finite], kind="stable")
            ordinal = np.empty(count)
            ordinal[order] = np.arange(count, dtype=np.float64)
            result[finite] = (ordinal + 0.5) / count
        return result, finite

    for factor_index, factor_id in enumerate(factor_ids):
        item = next(candidate for candidate in report.items if candidate.factor_id == factor_id)
        for window, fold in zip(plan.formal_split.windows, item.fold_evidence, strict=True):
            training = [
                value
                for value in (
                    rank_ic(
                        by_session[session]["scores"][:, factor_index], by_session[session]["fit"]
                    )
                    for session in window.train_sessions
                )
                if value is not None
            ]
            mean = float(np.mean(training)) if training else None
            orientation = None if mean is None else (1 if mean >= 0.0 else -1)
            assert fold.orientation == orientation
            assert fold.training_period_count == len(training)
            assert fold.training_mean_rank_ic == mean
            ics: list[float] = []
            spreads: list[float] = []
            coverages: list[float] = []
            turnovers: list[float] = []
            previous = None
            for session in window.validation_sessions:
                scores = by_session[session]["scores"][:, factor_index]
                oriented = scores if orientation is None else scores * orientation
                common = np.isfinite(scores) & np.isfinite(by_session[session]["fit"])
                coverages.append(float(common.sum() / len(listing_ids)))
                value = rank_ic(oriented, by_session[session]["fit"])
                if value is not None:
                    ics.append(value)
                spread = gross_decile_spread(oriented, by_session[session]["simple"])
                if spread is not None:
                    spreads.append(spread)
                if previous is not None:
                    left, left_finite = percentiles(previous)
                    right, right_finite = percentiles(oriented)
                    both = left_finite & right_finite
                    if int(both.sum()) >= 2:
                        turnovers.append(float(np.mean(np.abs(right[both] - left[both]))))
                previous = oriented
            assert fold.validation_period_count == len(ics)
            assert fold.validation_oriented_mean_rank_ic == (float(np.mean(ics)) if ics else None)
            assert fold.validation_oriented_mean_simple_spread == (
                float(np.mean(spreads)) if spreads else None
            )
            assert fold.validation_pair_coverage_mean == float(np.mean(coverages))
            assert fold.validation_rank_turnover_mean == (
                float(np.mean(turnovers)) if turnovers else None
            )
    flipped = next(item for item in report.items if item.factor_id == "factor_flip")
    assert flipped.fold_evidence[0].orientation is not None
    assert flipped.mean_oriented_rank_ic is not None and flipped.mean_oriented_rank_ic < 0.0
    orientations = {
        item.factor_id: item.fold_evidence[0].orientation
        for item in report.items
        if item.factor_id in {"factor_infinite", "factor_sparse"}
    }
    assert set(orientations.values()) == {-1, 1}, orientations
    assert all(
        fold.validation_rank_turnover_mean is not None
        for item in report.items
        if item.factor_id in {"factor_infinite", "factor_sparse"}
        for fold in item.fold_evidence
    )


def test_walk_forward_refuses_a_formation_whose_label_availability_differs_by_row() -> None:
    panel_sessions = tuple(date(2026, 1, 2) + timedelta(days=index) for index in range(11))
    listing_ids = ("listing-a", "listing-b")
    source = _source_table(panel_sessions[:8], listing_ids)
    shifted = source.set_column(
        source.schema.get_field_index("holding_end_open_at"),
        "holding_end_open_at",
        pa.array(
            [
                value + timedelta(hours=1) if index == 1 else value
                for index, value in enumerate(source["holding_end_open_at"].to_pylist())
            ],
            type=source.schema.field("holding_end_open_at").type,
        ),
    )
    surface = compile_factor_target_surface(
        source_table=shifted,
        source_manifest=_manifest(listing_ids),
        source_manifest_ref="playpen://outcomes/manifest",
        policy=build_factor_target_policy(),
    )
    with pytest.raises(FactorWalkForwardBoundaryError, match="label_availability_axis_mismatch"):
        compile_factor_walk_forward_plan(
            panel_sessions=panel_sessions,
            target_surface=surface,
            frozen_at=datetime(2027, 1, 1, tzinfo=UTC),
            policy=build_factor_walk_forward_policy(
                train_sessions=3,
                validation_sessions=2,
                step_sessions=2,
                sealed_holdout_sessions=2,
                minimum_folds=2,
            ),
        )


def test_redundancy_uses_absolute_rank_distance_and_average_linkage() -> None:
    rows: list[dict[str, object]] = []
    for session_index in range(5):
        session = date(2026, 2, 2) + timedelta(days=session_index)
        for listing_index in range(120):
            rows.append(
                {
                    "session_date": session,
                    "listing_id": f"listing-{listing_index:03d}",
                    "factor_a": float(listing_index),
                    "factor_b": -float(listing_index),
                    "factor_c": float((listing_index * 37 + session_index * 13) % 120),
                }
            )
    table = pa.Table.from_pylist(rows)
    factor_ids = ("factor_a", "factor_b", "factor_c")
    policy = build_factor_redundancy_policy(
        minimum_common_listings=100,
        minimum_formal_periods=3,
        distance_cut=0.1,
    )

    structure = build_factor_redundancy_structure(
        feature_table=table,
        feature_panel_snapshot_hash="8" * 64,
        feature_panel_manifest_ref="playpen://feature-panel/manifest",
        factor_ids=factor_ids,
        policy=policy,
    )
    shuffled = build_factor_redundancy_structure(
        feature_table=table.take(pa.array(list(reversed(range(table.num_rows))))),
        feature_panel_snapshot_hash="8" * 64,
        feature_panel_manifest_ref="playpen://feature-panel/manifest",
        factor_ids=factor_ids,
        policy=policy,
    )

    assert structure.structure_hash == shuffled.structure_hash
    assert tuple(cluster.member_factor_ids for cluster in structure.clusters) == (
        ("factor_a", "factor_b"),
        ("factor_c",),
    )
    inverse_pair = next(
        pair
        for pair in structure.pair_evidence
        if (pair.left_factor_id, pair.right_factor_id) == ("factor_a", "factor_b")
    )
    assert inverse_pair.median_cross_section_spearman == -1.0
    assert inverse_pair.distance == 0.0


def test_redundancy_fails_closed_when_a_formal_pair_is_unresolved() -> None:
    table = pa.Table.from_pylist(
        [
            {
                "session_date": date(2026, 3, 2),
                "listing_id": f"listing-{index:03d}",
                "factor_a": float(index),
                "factor_b": float(index),
            }
            for index in range(120)
        ]
    )
    with pytest.raises(
        FactorRedundancyBoundaryError,
        match=r"factor_research\.redundancy_pair_unresolved",
    ):
        build_factor_redundancy_structure(
            feature_table=table,
            feature_panel_snapshot_hash="7" * 64,
            feature_panel_manifest_ref="playpen://feature-panel/manifest",
            factor_ids=("factor_a", "factor_b"),
            policy=build_factor_redundancy_policy(minimum_formal_periods=2),
        )
