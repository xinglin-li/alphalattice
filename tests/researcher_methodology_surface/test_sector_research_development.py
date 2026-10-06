"""The deterministic Sector Research capability, driven through the real Host.

The unit-level cases pin the arithmetic to hand computations: the clean target
is the equal-weight sector mean and nothing else, ZERO is exactly zero, the
EWMA is the normalized exponentially weighted mean with an explicit warmup that
flips at precisely the twenty-first matured observation, and an immature label
is refused at input construction -- before any adapter can run.

The service cases then drive ``SectorResearchDevelopmentService`` over the
shared real workspace, against causal outcomes published by the product's own
writer and a method seal resolved by the real outcome reader. Nothing here
constructs a seal, a snapshot hash or a maturity lag: those are the identities
the service exists to derive, and a fixture that supplied them would be
asserting exactly what publication is supposed to establish.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.investment.sector_research.contracts import (
    SectorResearchError,
    grid_to_matrix,
)
from alphalattice.investment.sector_research.experiments.campaign import (
    SECTOR_CAMPAIGN_METHOD_IDS,
    SectorCampaignRequest,
)
from alphalattice.investment.sector_research.experiments.campaign_evidence import (
    SectorForecastSelectionSubmission,
)
from alphalattice.investment.sector_research.experiments.development_artifacts import (
    SECTOR_EXPERIMENT_EVIDENCE_CATEGORY,
    SECTOR_FORECAST_EVALUATION_CATEGORY,
    SECTOR_FORECAST_SURFACE_CATEGORY,
    SECTOR_TARGET_EVIDENCE_CATEGORY,
    SectorExperimentEvidence,
    sector_artifact_uri,
)
from alphalattice.investment.sector_research.experiments.service import (
    SectorExperimentRequest,
    SectorResearchDevelopmentService,
)
from alphalattice.investment.sector_research.models.catalog import (
    build_installed_sector_forecast_catalog,
)
from alphalattice.investment.sector_research.models.contracts import (
    INSUFFICIENT_MATURED_HISTORY,
    BoundSectorForecastInput,
)
from alphalattice.investment.sector_research.targets.execution import (
    build_sector_target_recipe,
    compile_sector_target_surface,
    seal_sector_target_evidence,
)
from alphalattice.protocols.actor_execution import ActorKind
from tests.researcher_methodology_surface.real_workspace import (
    RealRiskWorkspace,
    publish_causal_outcomes,
)

# --------------------------------------------------------------------- helpers


def _synthetic_target_table(
    *, session_count: int = 30, listing_count: int = 10
) -> tuple[pa.Table, dict[str, str], tuple[date, ...]]:
    """A complete formation-by-listing grid with the outcome seam's clocks."""

    sessions = tuple(date(2026, 1, 5) + timedelta(days=index) for index in range(session_count))
    listings = tuple(f"L{index}" for index in range(listing_count))
    sectors = {listing: f"S{index // 5}" for index, listing in enumerate(listings)}
    rng = np.random.default_rng(7)
    rows = [
        {
            "formation_session": session,
            "listing_id": listing,
            "fit_target": float(rng.normal(0.0, 0.01)),
            "simple_economic_return": float(rng.normal(0.0, 0.01)),
            "entry_session": session + timedelta(days=1),
            "holding_end_session": session + timedelta(days=2),
        }
        for session in sessions
        for listing in listings
    ]
    return pa.Table.from_pylist(rows), sectors, sessions


def _synthetic_evidence(*, session_count: int = 30):  # type: ignore[no-untyped-def]
    table, sectors, sessions = _synthetic_target_table(session_count=session_count)
    recipe = build_sector_target_recipe(
        execution_outcome_recipe_id="ONE_SESSION_OPEN_TO_OPEN",
        sector_revision="a" * 64,
    )
    surface = compile_sector_target_surface(
        source_table=table, recipe=recipe, sector_by_listing_id=sectors
    )
    evidence = seal_sector_target_evidence(
        surface=surface,
        causal_outcome_snapshot_hash="b" * 64,
        outcome_method_binding_hash="c" * 64,
        maturity_lag_sessions=2,
        actual_session_span=2,
        forecast_horizon_sessions=1,
    )
    return table, sectors, sessions, surface, evidence


def _bound_input(evidence, sessions, matrix, *, rows: int, formation_index: int = 24):  # type: ignore[no-untyped-def]
    return BoundSectorForecastInput.create(
        target_evidence_hash=evidence.evidence_hash,
        ordered_sectors=evidence.ordered_sectors,
        forecast_formation_at=sessions[formation_index],
        training_formation_sessions=tuple(sessions[:rows]),
        training_target_available_sessions=tuple(
            value + timedelta(days=2) for value in sessions[:rows]
        ),
        training_values=matrix[:rows, :],
    )


# ------------------------------------------------------------ unit-level cases


def test_clean_target_is_the_hand_computed_sector_mean_and_nothing_else() -> None:
    """Equal-weight mean of constituent log returns: no winsor, no rank, no demean.

    The strongest statement a golden can make is exact float equality against
    an independent composition, so that is what is asserted -- for both the fit
    lane and the raw economic diagnostic lane.
    """

    table, sectors, sessions, surface, evidence = _synthetic_evidence()
    ordered = table.sort_by([("formation_session", "ascending"), ("listing_id", "ascending")])
    fit = np.asarray(ordered["fit_target"]).reshape(len(sessions), 10)
    simple = np.asarray(ordered["simple_economic_return"]).reshape(len(sessions), 10)
    for row in (0, 7, len(sessions) - 1):
        assert surface.sector_target[row, 0] == float(np.mean(fit[row, :5]))
        assert surface.sector_target[row, 1] == float(np.mean(fit[row, 5:]))
        assert surface.economic_diagnostic[row, 0] == float(np.mean(simple[row, :5]))
    # The mean is of the raw values: had any bounding or demeaning slipped into
    # the sequence, exact equality against the raw composition would fail.
    assert evidence.recipe.sequence == ("equal_weight_sector_mean",)
    assert evidence.target_available_sessions == evidence.exit_sessions

    # Availability is typed, not NaN-shaped: empty a sector below its floor.
    broken = table.to_pylist()
    for entry in broken:
        if entry["formation_session"] == sessions[3] and entry["listing_id"] in {"L0", "L1"}:
            entry["fit_target"] = None
            entry["simple_economic_return"] = None
    partial = compile_sector_target_surface(
        source_table=pa.Table.from_pylist(broken),
        recipe=surface.recipe,
        sector_by_listing_id=sectors,
    )
    assert partial.available[3][0] is False
    assert partial.unavailable_reasons[3][0] == "SECTOR_SAMPLE_BELOW_MINIMUM"
    assert partial.available[3][1] is True


def test_same_shaped_wrong_inputs_are_refused_before_any_model() -> None:
    """A wrong date, membership, schedule or duplicate fails at the boundary."""

    table, sectors, _sessions, surface, _evidence = _synthetic_evidence()

    incomplete_membership = dict(sectors)
    incomplete_membership.pop("L0")
    with pytest.raises(SectorResearchError, match="sector_membership_incomplete"):
        compile_sector_target_surface(
            source_table=table, recipe=surface.recipe, sector_by_listing_id=incomplete_membership
        )

    duplicated = pa.concat_tables([table, table.slice(0, 1)])
    with pytest.raises(SectorResearchError, match="target_source_duplicated"):
        compile_sector_target_surface(
            source_table=duplicated, recipe=surface.recipe, sector_by_listing_id=sectors
        )

    with pytest.raises(SectorResearchError, match="target_surface_incomplete"):
        compile_sector_target_surface(
            source_table=table.slice(0, table.num_rows - 1),
            recipe=surface.recipe,
            sector_by_listing_id=sectors,
        )

    # One listing of one formation disagreeing about the exit clock means the
    # source is not the single-schedule seam this target is defined over.
    rows = table.to_pylist()
    rows[5]["holding_end_session"] = rows[5]["holding_end_session"] + timedelta(days=1)
    with pytest.raises(SectorResearchError, match="target_schedule_axis_mismatch"):
        compile_sector_target_surface(
            source_table=pa.Table.from_pylist(rows),
            recipe=surface.recipe,
            sector_by_listing_id=sectors,
        )


def test_zero_and_ewma_goldens_with_the_warmup_boundary_at_twenty_one() -> None:
    """ZERO is exactly zero; the EWMA equals its hand-built weighted mean.

    The warmup boundary is asserted at both edges: the twentieth matured
    observation refuses with the typed reason, the twenty-first emits a value.
    No zero fill, no backfill, no seeded fallback.
    """

    _table, _sectors, sessions, _surface, evidence = _synthetic_evidence()
    matrix = grid_to_matrix(evidence.sector_target_values)
    catalog = build_installed_sector_forecast_catalog()
    zero_recipe = catalog.seal_recipe(method_id="ZERO_SECTOR_FORECAST")
    ewma_recipe = catalog.seal_recipe(method_id="EWMA_SECTOR_MEAN")

    bound = _bound_input(evidence, sessions, matrix, rows=22)
    zero = catalog.resolve(zero_recipe).forecast(bound_input=bound, recipe=zero_recipe)
    assert zero.values == (0.0, 0.0)
    assert zero.unavailable_reasons == (None, None)

    ewma = catalog.resolve(ewma_recipe).forecast(bound_input=bound, recipe=ewma_recipe)
    weights = 0.5 ** (np.arange(21, -1, -1, dtype=np.float64) / 21.0)
    for column in (0, 1):
        hand = float(np.sum(weights * matrix[:22, column]) / np.sum(weights))
        assert ewma.values[column] == hand

    twenty = catalog.resolve(ewma_recipe).forecast(
        bound_input=_bound_input(evidence, sessions, matrix, rows=20), recipe=ewma_recipe
    )
    assert twenty.values == (None, None)
    assert twenty.unavailable_reasons == (
        INSUFFICIENT_MATURED_HISTORY,
        INSUFFICIENT_MATURED_HISTORY,
    )
    twenty_one = catalog.resolve(ewma_recipe).forecast(
        bound_input=_bound_input(evidence, sessions, matrix, rows=21), recipe=ewma_recipe
    )
    assert twenty_one.values[0] is not None
    assert twenty_one.unavailable_reasons == (None, None)


@pytest.mark.parametrize(
    ("method_id", "session_count", "minimum_history"),
    [
        ("SECTOR_RELATIVE_STRENGTH_126", 200, 126),
        ("DISTRIBUTED_LAG_ELASTIC_NET", 400, 252),
    ],
)
def test_new_batch_methods_match_their_formula_and_their_warmup_boundary(
    method_id: str, session_count: int, minimum_history: int
) -> None:
    """Each new method equals its documented formula, computed independently here.

    The expectations are rebuilt from the Formula Specification rather than by
    calling the adapter's own helpers, so the lag ordering, the cross-sector
    demeaning, the per-session rescaling and the ``alpha_max``-relative penalty
    are all pinned by something that would not move with them. The warmup
    boundary is asserted at both edges: one observation short refuses with the
    typed reason, and the exact minimum emits.
    """

    _table, _sectors, sessions, _surface, evidence = _synthetic_evidence(
        session_count=session_count
    )
    matrix = grid_to_matrix(evidence.sector_target_values)
    catalog = build_installed_sector_forecast_catalog()
    recipe = catalog.seal_recipe(method_id=method_id)
    adapter = catalog.resolve(recipe)

    rows = minimum_history + 5
    bound = _bound_input(evidence, sessions, matrix, rows=rows, formation_index=session_count - 1)
    produced = adapter.forecast(bound_input=bound, recipe=recipe)

    if method_id == "SECTOR_RELATIVE_STRENGTH_126":
        lookback = recipe.parameters["lookback_sessions"]
        window = matrix[rows - lookback : rows, :]
        strengths = [float(np.sum(window[:, column])) for column in range(window.shape[1])]
        breadth = float(np.mean(np.asarray(strengths, dtype=np.float64)))
        for column, strength in enumerate(strengths):
            assert produced.values[column] == (strength - breadth) / float(lookback)
        # Relative by construction: the method takes no view on the market level.
        assert sum(cast_float(value) for value in produced.values) == pytest.approx(0.0, abs=1e-15)
    else:
        from sklearn.linear_model import ElasticNet

        lag_count = recipe.parameters["lag_count"]
        l1_ratio = recipe.parameters["l1_ratio_percent"] / 100.0
        multiplier = recipe.parameters["alpha_max_multiplier_basis_points"] / 10_000.0
        history = matrix[:rows, :]
        # Built by an explicit loop rather than the adapter's vectorized helper,
        # so a change to the lag block order fails here.
        design = np.asarray(
            [
                np.concatenate([history[index - lag, :] for lag in range(1, lag_count + 1)])
                for index in range(lag_count, rows)
            ],
            dtype=np.float64,
        )
        targets = history[lag_count:, :]
        latest = np.concatenate(
            [history[rows - lag, :] for lag in range(1, lag_count + 1)]
        ).reshape(1, -1)
        for column in range(targets.shape[1]):
            response = targets[:, column]
            centred = design - design.mean(axis=0)
            alpha_max = float(
                np.max(np.abs(centred.T @ (response - response.mean())))
                / (len(response) * l1_ratio)
            )
            estimator = ElasticNet(
                alpha=alpha_max * multiplier,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                max_iter=10_000,
                tol=1e-8,
                selection="cyclic",
            )
            estimator.fit(design, response)
            assert produced.values[column] == float(estimator.predict(latest)[0])

    short = adapter.forecast(
        bound_input=_bound_input(
            evidence,
            sessions,
            matrix,
            rows=minimum_history - 1,
            formation_index=session_count - 1,
        ),
        recipe=recipe,
    )
    assert set(short.values) == {None}
    assert set(short.unavailable_reasons) == {INSUFFICIENT_MATURED_HISTORY}
    exact = adapter.forecast(
        bound_input=_bound_input(
            evidence,
            sessions,
            matrix,
            rows=minimum_history + (recipe.parameters.get("lag_count", 0)),
            formation_index=session_count - 1,
        ),
        recipe=recipe,
    )
    assert None not in exact.values


def cast_float(value: float | None) -> float:
    assert value is not None
    return value


def test_immature_label_is_refused_before_the_adapter_and_singleton_is_frozen() -> None:
    """The causal boundary is structural and the parameter space is one point."""

    _table, _sectors, sessions, _surface, evidence = _synthetic_evidence()
    matrix = grid_to_matrix(evidence.sector_target_values)

    # Training labels through session 5, but the label for session 5 matures at
    # session 7 -- constructing the input already refuses, so no adapter is
    # ever consulted about a future it should not have seen.
    with pytest.raises(SectorResearchError, match="training_label_immature"):
        _bound_input(evidence, sessions, matrix, rows=6, formation_index=5)

    catalog = build_installed_sector_forecast_catalog()
    with pytest.raises(SectorResearchError, match="recipe_outside_frozen_singleton"):
        catalog.seal_recipe(method_id="EWMA_SECTOR_MEAN", parameters={"half_life_sessions": 10})
    with pytest.raises(SectorResearchError, match="method_not_installed"):
        catalog.seal_recipe(method_id="SECTOR_EXCESS_EMA_LEGACY")


# ------------------------------------------------------------- service fixture


def _service(workspace: RealRiskWorkspace) -> SectorResearchDevelopmentService:
    return SectorResearchDevelopmentService(
        artifact_root=workspace.artifact_root,
        outcome_reader=CausalExecutionOutcomeDevelopmentReader(workspace.artifact_root),
        resolver=ArtifactResolver(workspace.artifact_root),
    )


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


@pytest.fixture(scope="module")
def sector_publication(real_risk_workspace: RealRiskWorkspace):  # type: ignore[no-untyped-def]
    """One real EWMA experiment and one ZERO control over the shared workspace."""

    snapshot_hash, _manifest_ref = publish_causal_outcomes(real_risk_workspace)
    service = _service(real_risk_workspace)
    reader = CausalExecutionOutcomeDevelopmentReader(real_risk_workspace.artifact_root)
    manifest = reader.load_manifest(snapshot_hash)
    first = min(value.first_formation_session for value in manifest.development_chunks)
    last = max(value.last_formation_session for value in manifest.development_chunks)
    revision = _sector_revision(real_risk_workspace)

    def request(method_id: str) -> SectorExperimentRequest:
        return SectorExperimentRequest(
            causal_outcome_snapshot_hash=snapshot_hash,
            sector_revision=revision,
            method_id=method_id,
            training_start=first,
            training_end=last,
            forecast_start=first,
            forecast_end=last + timedelta(days=30),
        )

    ewma = service.publish(request("EWMA_SECTOR_MEAN"))
    zero = service.publish(request("ZERO_SECTOR_FORECAST"))
    return service, snapshot_hash, revision, ewma, zero, (first, last)


# -------------------------------------------------------------- service cases


def test_host_derives_every_identity_and_the_verifier_walks_to_the_seal(  # type: ignore[no-untyped-def]
    sector_publication, real_risk_workspace: RealRiskWorkspace
) -> None:
    """The outcome method, maturity lag and catalog identity come from evidence.

    None of them was available to the caller to supply, and the verifier's
    terminal edge re-derives the seal from the outcome reader rather than from
    anything Sector Research wrote.
    """

    service, snapshot_hash, revision, ewma, _zero, _range = sector_publication
    reader = CausalExecutionOutcomeDevelopmentReader(real_risk_workspace.artifact_root)
    seal = reader.resolve_method_seal(snapshot_hash)
    assert seal.disposition == "METHOD_BOUND"

    evidence = ewma.target_evidence
    assert evidence.causal_outcome_snapshot_hash == snapshot_hash
    assert evidence.outcome_method_binding_hash == seal.method_bound.binding_hash
    assert evidence.maturity_lag_sessions == seal.method_bound.maturity_lag_sessions
    assert evidence.recipe.sector_revision == revision
    catalog = build_installed_sector_forecast_catalog()
    assert ewma.surface.program_binding.catalog_hash == catalog.binding.catalog_hash

    lineage = service.verify(experiment_hash=ewma.experiment.experiment_hash)
    assert lineage.outcome_method_binding_hash == seal.method_bound.binding_hash
    assert lineage.surface.surface_hash == ewma.surface.surface_hash

    # The explicit warmup is visible in the real run: the earliest refits have
    # fewer than twenty-one matured labels and refuse; later ones emit values.
    first_row = ewma.surface.values[0]
    assert all(value is None for value in first_row)
    assert all(
        reason == INSUFFICIENT_MATURED_HISTORY for reason in ewma.surface.unavailable_reasons[0]
    )
    assert any(value is not None for row in ewma.surface.values for value in row)

    # Training row counts grow with maturity and never include an immature
    # label: the count at each row equals the verifier's own re-derivation,
    # which service.verify above already held the surface to.
    counts = ewma.surface.training_row_counts
    assert counts[0] <= counts[-1]


def test_the_zero_control_is_resolved_through_the_sector_owner(  # type: ignore[no-untyped-def]
    sector_publication,
) -> None:
    """ZERO publishes the same evidence graph as any method, all zeros."""

    service, _snapshot, _revision, ewma, zero, _range = sector_publication
    assert all(value == 0.0 for row in zero.surface.values for value in row)
    # Both experiments share the one target evidence: same snapshot, same
    # revision, same clean-target recipe -- the content address proves it.
    assert zero.target_evidence.evidence_hash == ewma.target_evidence.evidence_hash
    lineage = service.verify(experiment_hash=zero.experiment.experiment_hash)
    assert lineage.evaluation.evaluated_counts != ()


def test_tamper_missing_child_and_wrong_lineage_are_refused(  # type: ignore[no-untyped-def]
    sector_publication,
) -> None:
    """The graph refuses a flipped byte, a deleted child, and unrelated children."""

    service, snapshot_hash, revision, ewma, zero, _range = sector_publication
    store = service.store

    surface_path = (
        store.root / SECTOR_FORECAST_SURFACE_CATEGORY / f"{ewma.surface.surface_hash}.json"
    )
    original = surface_path.read_bytes()
    try:
        payload = json.loads(original)
        payload["training_row_counts"][-1] = int(payload["training_row_counts"][-1]) + 1
        surface_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        with pytest.raises(SectorResearchError, match="evidence_child_unverifiable"):
            service.verify(experiment_hash=ewma.experiment.experiment_hash)
    finally:
        surface_path.write_bytes(original)
    service.verify(experiment_hash=ewma.experiment.experiment_hash)

    evaluation_path = (
        store.root / SECTOR_FORECAST_EVALUATION_CATEGORY / f"{ewma.evaluation.evaluation_hash}.json"
    )
    saved = evaluation_path.read_bytes()
    try:
        evaluation_path.unlink()
        with pytest.raises(SectorResearchError, match="evidence_child_missing"):
            service.verify(experiment_hash=ewma.experiment.experiment_hash)
    finally:
        evaluation_path.write_bytes(saved)
    service.verify(experiment_hash=ewma.experiment.experiment_hash)

    # Reordered children cannot even be sealed: order is part of the contract.
    # (Pydantic wraps a validator's refusal in its own ValidationError, which
    # is still a ValueError carrying the stable code.)
    with pytest.raises(ValueError, match="experiment_children_invalid"):
        SectorExperimentEvidence.create(
            kind="SectorExperimentEvidence",
            causal_outcome_snapshot_hash=snapshot_hash,
            sector_revision=revision,
            method_id="EWMA_SECTOR_MEAN",
            catalog_hash=ewma.experiment.catalog_hash,
            target_evidence_hash=ewma.experiment.target_evidence_hash,
            forecast_surface_hash=ewma.experiment.forecast_surface_hash,
            evaluation_hash=ewma.experiment.evaluation_hash,
            ordered_child_uris=(
                sector_artifact_uri(
                    SECTOR_FORECAST_SURFACE_CATEGORY, ewma.experiment.forecast_surface_hash
                ),
                sector_artifact_uri(
                    SECTOR_TARGET_EVIDENCE_CATEGORY, ewma.experiment.target_evidence_hash
                ),
                sector_artifact_uri(
                    SECTOR_FORECAST_EVALUATION_CATEGORY, ewma.experiment.evaluation_hash
                ),
            ),
        )

    # Valid children that do not belong to each other: an experiment naming
    # EWMA's surface beside ZERO's evaluation is internally hash-consistent and
    # still refused, because the evaluation's own parent is a different surface.
    forged = SectorExperimentEvidence.create(
        kind="SectorExperimentEvidence",
        causal_outcome_snapshot_hash=snapshot_hash,
        sector_revision=revision,
        method_id="EWMA_SECTOR_MEAN",
        catalog_hash=ewma.experiment.catalog_hash,
        target_evidence_hash=ewma.experiment.target_evidence_hash,
        forecast_surface_hash=ewma.experiment.forecast_surface_hash,
        evaluation_hash=zero.experiment.evaluation_hash,
        ordered_child_uris=(
            sector_artifact_uri(
                SECTOR_TARGET_EVIDENCE_CATEGORY, ewma.experiment.target_evidence_hash
            ),
            sector_artifact_uri(
                SECTOR_FORECAST_SURFACE_CATEGORY, ewma.experiment.forecast_surface_hash
            ),
            sector_artifact_uri(
                SECTOR_FORECAST_EVALUATION_CATEGORY, zero.experiment.evaluation_hash
            ),
        ),
    )
    forged_path = (
        store.root / SECTOR_EXPERIMENT_EVIDENCE_CATEGORY / f"{forged.experiment_hash}.json"
    )
    try:
        store.publish_experiment_evidence(forged)
        with pytest.raises(SectorResearchError, match="evaluation_not_this_surface"):
            service.verify(experiment_hash=forged.experiment_hash)
    finally:
        # The forged root must not survive this case: a later directory scan --
        # the CLI's inspect, for one -- verifies everything it finds.
        forged_path.unlink(missing_ok=True)


def test_selection_resolution_is_owned_by_sector_and_refuses_wrong_axes(  # type: ignore[no-untyped-def]
    sector_publication, real_risk_workspace: RealRiskWorkspace
) -> None:
    """A consumer holds a selection; the values come from this Desk, verified.

    The ZERO control is synthesized here against the requested axis -- never by
    the consumer from a method id. An evidence handle is walked down to the
    outcome seal before a single value is mapped, and every listing receives
    exactly its own sector's value through the same membership revision the
    forecast was built against. A session outside the forecast axis, a listing
    the revision does not cover, and a selection naming the wrong method are
    each refused with their own code.
    """

    from alphalattice.investment.sector_research.experiments.resolution import (
        SectorForecastResolver,
        SectorForecastSelection,
    )
    from alphalattice.investment.sector_research.inputs.membership import (
        load_sector_membership,
    )

    _service_obj, _snapshot, revision, ewma, _zero, (first, _last) = sector_publication
    artifact_root = real_risk_workspace.artifact_root
    seam = SectorForecastResolver(
        artifact_root=artifact_root,
        outcome_reader=CausalExecutionOutcomeDevelopmentReader(artifact_root),
        resolver=ArtifactResolver(artifact_root),
    )
    membership = load_sector_membership(
        resolver=ArtifactResolver(artifact_root), sector_revision=revision
    )
    sessions = ewma.forecast_formation_sessions[-5:]
    listings = tuple(sorted(membership))

    selection = SectorForecastSelection.from_experiment(
        method_id="EWMA_SECTOR_MEAN",
        sector_experiment_hash=ewma.experiment.experiment_hash,
    )
    resolved = seam.resolve(
        selection=selection, formation_sessions=sessions, ordered_listing_ids=listings
    )
    assert resolved.sector_target_evidence_hash == ewma.target_evidence.evidence_hash
    assert not resolved.expected_return_by_listing.flags.writeable
    sectors = ewma.target_evidence.ordered_sectors
    for row, session in enumerate(sessions):
        source_row = ewma.surface.forecast_formation_sessions.index(session)
        for column, listing in enumerate(listings):
            expected = ewma.surface.values[source_row][sectors.index(membership[listing])]
            actual = resolved.expected_return_by_listing[row, column]
            assert (expected is None and bool(np.isnan(actual))) or actual == expected

    zeros = seam.resolve(
        selection=SectorForecastSelection.control_zero(),
        formation_sessions=sessions,
        ordered_listing_ids=listings,
    )
    assert bool(np.all(zeros.expected_return_by_listing == 0.0))
    assert zeros.sector_experiment_hash is None

    with pytest.raises(SectorResearchError, match="resolution_session_axis_mismatch"):
        seam.resolve(
            selection=selection,
            formation_sessions=(first - timedelta(days=365),),
            ordered_listing_ids=listings,
        )
    with pytest.raises(SectorResearchError, match="resolution_membership_incomplete"):
        seam.resolve(
            selection=selection,
            formation_sessions=sessions,
            ordered_listing_ids=("ZZZ-NOT-A-LISTING",),
        )
    with pytest.raises(SectorResearchError, match="resolution_method_mismatch"):
        seam.resolve(
            selection=SectorForecastSelection.from_experiment(
                method_id="ZERO_SECTOR_FORECAST",
                sector_experiment_hash=ewma.experiment.experiment_hash,
            ),
            formation_sessions=sessions,
            ordered_listing_ids=listings,
        )
    # A non-control method without evidence cannot even be constructed.
    with pytest.raises(ValueError, match="selection_evidence_required"):
        SectorForecastSelection(
            kind="SectorForecastSelection",
            method_id="EWMA_SECTOR_MEAN",
            sector_experiment_hash=None,
            selection_hash="0" * 64,
        )


def test_the_installed_cli_publishes_and_inspects(  # type: ignore[no-untyped-def]
    sector_publication, real_risk_workspace: RealRiskWorkspace
) -> None:
    """A scripted publish and inspect through the real entry point.

    The publish re-states the ZERO run the fixture already sealed, which also
    proves idempotency: identical content re-publishes to identical hashes and
    the store's reuse-identical path accepts it.
    """

    _service_obj, snapshot_hash, revision, _ewma, zero, (first, last) = sector_publication
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_sector_research_development.py"
    environment = dict(os.environ)
    environment["ALPHALATTICE_NETWORK_DISABLED"] = "1"

    published = subprocess.run(
        [
            sys.executable,
            str(script),
            "--workspace",
            str(real_risk_workspace.workspace),
            "--publish",
            "--snapshot-hash",
            snapshot_hash,
            "--sector-revision",
            revision,
            "--method",
            "ZERO_SECTOR_FORECAST",
            "--training-start",
            first.isoformat(),
            "--training-end",
            last.isoformat(),
            "--forecast-start",
            first.isoformat(),
            "--forecast-end",
            (last + timedelta(days=30)).isoformat(),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )
    result = json.loads(published.stdout)
    assert result["action"] == "PUBLISHED"
    assert result["experiment_hash"] == zero.experiment.experiment_hash
    assert result["current_pointer_writes"] == 0
    assert result["alpha_model_fits"] == 0

    inspected = subprocess.run(
        [
            sys.executable,
            str(script),
            "--workspace",
            str(real_risk_workspace.workspace),
            "--inspect",
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )
    report = json.loads(inspected.stdout)
    assert report["action"] == "INSPECTED"
    verified = {entry["experiment_hash"] for entry in report["experiments"]}
    assert zero.experiment.experiment_hash in verified


# ----------------------------------------------------- adversarial regressions


def _reseal_forged_experiment(  # type: ignore[no-untyped-def]
    store,
    real,
    *,
    recipe=None,
    target_evidence=None,
    closure_hash=None,
):
    """Re-seal a fully self-consistent experiment graph with one authority forged.

    Every hash in the result re-derives perfectly: the recipe validates its own
    identity, the program binding and surface revalidate on parse, the
    evaluation is the genuine re-derivation over the (possibly forged) target,
    and the root names its children in order. Nothing in the graph is
    internally wrong -- which is exactly what makes these cases worth having,
    because only a comparison against what is *installed* can refuse them.

    Returns the forged root hash and every path written, so the caller can
    remove the forgery before any later directory scan verifies the store.
    """

    from alphalattice.investment.sector_research.experiments.development_artifacts import (
        SectorForecastProgramBinding,
        SectorForecastSurface,
        evaluate_sector_forecast_surface,
    )

    base_surface = real.surface
    base_binding = base_surface.program_binding
    target = target_evidence if target_evidence is not None else real.target_evidence
    forged_recipe = recipe if recipe is not None else base_surface.recipe

    binding = SectorForecastProgramBinding.create(
        kind="SectorForecastProgramBinding",
        target_evidence_hash=target.evidence_hash,
        catalog_hash=base_binding.catalog_hash,
        method_id=forged_recipe.method_id,
        recipe_hash=forged_recipe.recipe_hash,
        numerical_binding_hash=base_binding.numerical_binding_hash,
        training_start=base_binding.training_start,
        training_end=base_binding.training_end,
        forecast_start=base_binding.forecast_start,
        forecast_end=base_binding.forecast_end,
        forecast_horizon_sessions=base_binding.forecast_horizon_sessions,
        maturity_lag_sessions=base_binding.maturity_lag_sessions,
        refit_every_sessions=forged_recipe.parameters.get("refit_every_sessions", 0),
        minimum_history_sessions=forged_recipe.parameters.get("minimum_history_sessions", 0),
        forecast_formation_sessions=base_binding.forecast_formation_sessions,
        development_source_closure_hash=(
            closure_hash
            if closure_hash is not None
            else base_binding.development_source_closure_hash
        ),
    )
    surface = SectorForecastSurface.create(
        kind="SectorForecastSurface",
        identity_class="DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED",
        target_evidence_hash=target.evidence_hash,
        recipe=forged_recipe,
        program_binding=binding,
        ordered_sectors=base_surface.ordered_sectors,
        forecast_formation_sessions=base_surface.forecast_formation_sessions,
        values=base_surface.values,
        unavailable_reasons=base_surface.unavailable_reasons,
        training_row_counts=base_surface.training_row_counts,
        values_identity=base_surface.values_identity,
    )
    evaluation = evaluate_sector_forecast_surface(evidence=target, surface=surface)
    root = SectorExperimentEvidence.create(
        kind="SectorExperimentEvidence",
        causal_outcome_snapshot_hash=target.causal_outcome_snapshot_hash,
        sector_revision=target.recipe.sector_revision,
        method_id=forged_recipe.method_id,
        catalog_hash=binding.catalog_hash,
        target_evidence_hash=target.evidence_hash,
        forecast_surface_hash=surface.surface_hash,
        evaluation_hash=evaluation.evaluation_hash,
        ordered_child_uris=(
            sector_artifact_uri(SECTOR_TARGET_EVIDENCE_CATEGORY, target.evidence_hash),
            sector_artifact_uri(SECTOR_FORECAST_SURFACE_CATEGORY, surface.surface_hash),
            sector_artifact_uri(SECTOR_FORECAST_EVALUATION_CATEGORY, evaluation.evaluation_hash),
        ),
    )
    written = []
    if target_evidence is not None:
        store.publish_target_evidence(target)
        written.append(
            store.root / SECTOR_TARGET_EVIDENCE_CATEGORY / f"{target.evidence_hash}.json"
        )
    store.publish_forecast_surface(surface)
    store.publish_evaluation(evaluation)
    store.publish_experiment_evidence(root)
    written.extend(
        (
            store.root / SECTOR_FORECAST_SURFACE_CATEGORY / f"{surface.surface_hash}.json",
            store.root / SECTOR_FORECAST_EVALUATION_CATEGORY / f"{evaluation.evaluation_hash}.json",
            store.root / SECTOR_EXPERIMENT_EVIDENCE_CATEGORY / f"{root.experiment_hash}.json",
        )
    )
    return root.experiment_hash, written


def test_forged_recipe_policy_and_closure_cannot_pass_the_verifier(  # type: ignore[no-untyped-def]
    sector_publication,
) -> None:
    """Three internally perfect forgeries, three installed-authority refusals.

    A re-sealed surface carrying a 999-session half-life is refused because the
    frozen singleton is inside catalog identity, not merely the implementation.
    A target evidence compiled under a one-listing sector floor is refused
    because the clean-target policy is re-derived from the installed builder.
    And a program binding naming a different Host execution closure is refused
    because the compiler, selection, evaluation and service bytes that ran are
    part of what the program *is*. Before these checks, every one of these
    graphs verified: each hash re-derived, each child belonged to its parent,
    and the terminal seal held.
    """

    from alphalattice.investment.sector_research.models.contracts import SectorForecastRecipe
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    service, _snapshot, _revision, ewma, _zero, _range = sector_publication
    store = service.store
    written: list[Path] = []
    try:
        rogue_recipe = SectorForecastRecipe.create(
            method_id="EWMA_SECTOR_MEAN",
            recipe_schema_id="sector-forecast/ewma-sector-mean@1",
            parameters={
                "half_life_sessions": 999,
                "minimum_history_sessions": 21,
                "refit_every_sessions": 21,
                "forecast_horizon_sessions": 1,
            },
        )
        forged, paths = _reseal_forged_experiment(store, ewma, recipe=rogue_recipe)
        written.extend(paths)
        with pytest.raises(SectorResearchError, match="evidence_recipe_not_admitted"):
            service.verify(experiment_hash=forged)

        payload = ewma.target_evidence.model_dump(mode="json")
        payload["recipe"] = build_sector_target_recipe(
            execution_outcome_recipe_id=ewma.target_evidence.recipe.execution_outcome_recipe_id,
            sector_revision=ewma.target_evidence.recipe.sector_revision,
            minimum_sector_sample=1,
        ).model_dump(mode="json")
        identity = {key: value for key, value in payload.items() if key != "evidence_hash"}
        from alphalattice.investment.sector_research.targets.execution import SectorTargetEvidence

        rogue_target = SectorTargetEvidence.model_validate(
            {**identity, "evidence_hash": canonical_hash(identity)}
        )
        forged, paths = _reseal_forged_experiment(store, ewma, target_evidence=rogue_target)
        written.extend(paths)
        with pytest.raises(SectorResearchError, match="evidence_target_policy_not_installed"):
            service.verify(experiment_hash=forged)

        forged, paths = _reseal_forged_experiment(store, ewma, closure_hash="f" * 64)
        written.extend(paths)
        with pytest.raises(SectorResearchError, match="evidence_execution_closure_drift"):
            service.verify(experiment_hash=forged)

        # The genuine publication still verifies beside the removed forgeries.
        service.verify(experiment_hash=ewma.experiment.experiment_hash)
    finally:
        # Forged artifacts must not survive this case: later directory scans --
        # the CLI's inspect, for one -- verify everything they find.
        for path in written:
            path.unlink(missing_ok=True)


# ------------------------------------------------------------ campaign acceptance


def test_the_campaign_compares_calibrates_decides_and_replays(  # type: ignore[no-untyped-def]
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    """One Campaign, end to end, through the real Host on the real workspace.

    The whole chain in one path, because the parts are only worth what they are
    worth together: every installed method against one compiled target, a
    cross-fitted shrink calibration each, the frozen comparison rules, an
    actor-neutral submission the Host validates before sealing, and a replay
    that recomputes all of it and performs no numerical call.

    The decision is allowed to be ``ZERO``. A test that required a non-zero
    winner would be requiring a scientific result, which is not a property of
    the software.
    """

    snapshot_hash, _manifest_ref = publish_causal_outcomes(real_risk_workspace)
    service = _service(real_risk_workspace)
    reader = CausalExecutionOutcomeDevelopmentReader(real_risk_workspace.artifact_root)
    manifest = reader.load_manifest(snapshot_hash)
    last = max(value.last_formation_session for value in manifest.development_chunks)
    sessions = reader.available_development_sessions(
        reader.manifest_uri(snapshot_hash), holding_end_through=last + timedelta(days=30)
    )
    # Far enough in that the slowest method's warmup is satisfied, and short
    # enough that the acceptance path stays a test rather than a campaign.
    request = SectorCampaignRequest(
        causal_outcome_snapshot_hash=snapshot_hash,
        sector_revision=_sector_revision(real_risk_workspace),
        method_ids=SECTOR_CAMPAIGN_METHOD_IDS,
        training_start=sessions[0],
        training_end=sessions[-1],
        forecast_start=sessions[300],
        forecast_end=sessions[420],
        calibration_fold_count=3,
        development_only=True,
    )
    published = service.run_campaign(request)
    comparison = published.comparison

    # Every selected method reached the comparable table against one target.
    assert tuple(value.method_id for value in comparison.rows) == tuple(
        sorted(SECTOR_CAMPAIGN_METHOD_IDS)
    )
    assert {value.disposition for value in comparison.rows} == {"EVALUATED"}
    assert all(
        value.target_evidence_hash == published.target_evidence.evidence_hash
        for value in published.calibrations
    )
    # The calibration is a shrink, never a lever, and it is cross-fitted.
    for calibration in published.calibrations:
        assert calibration.fold_count == 3
        assert all(0.0 <= fold.slope <= 1.0 for fold in calibration.folds)
        assert calibration.evaluated_pair_count == sum(
            fold.evaluated_pair_count for fold in calibration.folds
        )
    # The conditional family is not installed and did not run.
    assert set(published.dossier.conditional_dispositions) == {
        "REGULARIZED_VAR",
        "PCA_FACTOR_VAR",
    }
    assert comparison.joint_dynamics_triggered == (
        "EVALUATED" in set(published.dossier.conditional_dispositions.values())
    )
    assert published.dossier.provider_call_count == 0
    assert published.dossier.holdout_access_count == 0
    assert published.dossier.pointer_mutation_count == 0

    submission = SectorForecastSelectionSubmission.create(
        dossier_hash=published.dossier.dossier_hash,
        selected_method_id=comparison.recommended_method_id,
        rationale="case-study submission",
    )
    receipt = service.seal_campaign_decision(
        dossier=published.dossier,
        comparison=comparison,
        submission=submission,
        actor_kind=ActorKind.EXTERNAL_AUTOMATION,
        actor_id="case-study-external-automation",
    )
    assert receipt.actor_binding.actor_kind is ActorKind.EXTERNAL_AUTOMATION
    assert receipt.actor_binding.agent_execution is None
    assert receipt.selection.method_id == comparison.recommended_method_id

    # A submitter may propose; it may not decide. Naming any other installed
    # method than the evidence supports is refused rather than sealed.
    contradicting = next(
        value.method_id
        for value in comparison.rows
        if value.method_id != comparison.recommended_method_id
    )
    with pytest.raises(SectorResearchError, match="submission_contradicts_evidence"):
        service.seal_campaign_decision(
            dossier=published.dossier,
            comparison=comparison,
            submission=SectorForecastSelectionSubmission.create(
                dossier_hash=published.dossier.dossier_hash,
                selected_method_id=contradicting,
                rationale="a proposal the evidence does not support",
            ),
            actor_kind=ActorKind.HUMAN,
            actor_id="case-study-human",
        )

    replay = service.verify_campaign(
        dossier_hash=published.dossier.dossier_hash,
        decision_receipt_hash=receipt.receipt_hash,
    )
    assert replay.disposition == "REUSED_EXACT"
    assert replay.verified_child_count >= 4 * len(SECTOR_CAMPAIGN_METHOD_IDS)
    assert (
        replay.forecast_call_count,
        replay.fit_call_count,
        replay.metric_call_count,
        replay.provider_call_count,
        replay.holdout_access_count,
        replay.pointer_mutation_count,
    ) == (0, 0, 0, 0, 0, 0)


def test_a_campaign_document_is_read_by_the_one_declaration_loader() -> None:
    """regression (V276): the Sector campaign read its document with a plain `yaml.safe_load`,
    which keeps the last of a key written twice; the one declaration loader refuses it."""

    import yaml

    with pytest.raises(yaml.YAMLError, match="twice"):
        SectorCampaignRequest.from_yaml(
            "campaign_id: a\ncampaign_id: b\n",
            causal_outcome_snapshot_hash="0" * 64,
            sector_revision="0" * 64,
        )
