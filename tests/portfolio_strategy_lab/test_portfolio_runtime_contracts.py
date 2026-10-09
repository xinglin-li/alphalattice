"""Progress publication and retired-lab readback tests for Strategy Lab."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pytest

import alphalattice.control.observation_runtime.telemetry.progress as progress_module
from alphalattice.control.observation_runtime.telemetry.progress import (
    WorkProgressProjection,
    WorkProgressUpdate,
    WorkspaceProgressPublisher,
)
from alphalattice.investment.portfolio_strategy_lab.contracts import (
    CurrentPortfolioResearchMarker,
    CurrentPortfolioResearchPointer,
    PortfolioEvidenceDossier,
    PortfolioExperimentProgram,
    PortfolioExperimentStratum,
    PortfolioResearchOutcome,
    PortfolioResearchProjectionBundle,
    PortfolioResearchReview,
    PortfolioResearchRFC,
    PortfolioScientificStop,
    PortfolioScoreMode,
    PortfolioTrialLedger,
    PortfolioTrialLedgerEntry,
    TopKEqualWeightPolicy,
    TopKMinimumVariancePolicy,
    TopKScoreRiskCostPolicy,
    TrialState,
    seal_contract,
)
from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
    PortfolioPublicationError,
    PortfolioResearchArtifactStore,
)

PLAYPEN_ROOT = Path(__file__).resolve().parents[2]
PLAYPEN_SRC = PLAYPEN_ROOT / "src"

_HASHES = tuple(f"{value:064x}" for value in range(1, 50))


def _program(offset: int = 0) -> PortfolioExperimentProgram:
    """A retired-lab Program record as its builder sealed it, for the readback tests."""

    strata = tuple(
        PortfolioExperimentStratum(
            candidate_id=candidate_id,
            score_mode=mode,
            stratum_id=f"{candidate_id}:{mode.value}",
        )
        for candidate_id in ("ridge", "lasso", "elastic-net")
        for mode in (
            PortfolioScoreMode.STOCK_ONLY,
            PortfolioScoreMode.STOCK_PLUS_SECTOR_COMPONENT,
        )
    )
    return seal_contract(
        PortfolioExperimentProgram,
        "program_hash",
        mandate_hash=_HASHES[offset],
        strata=strata,
        baseline_specs=(
            TopKEqualWeightPolicy(top_k=50, maximum_weight=0.05),
            TopKMinimumVariancePolicy(top_k=50, maximum_weight=0.05),
            TopKScoreRiskCostPolicy(
                top_k=50,
                maximum_weight=0.05,
                risk_aversion=1_000.0,
                turnover_regularization=0.1,
            ),
        ),
        initial_attempt_budget=len(strata) * 24,
        structural_attempt_budget=len(strata) * 12,
        execution_binding_hash=_HASHES[offset + 1],
        numerical_environment_hash=_HASHES[offset + 2],
    )


def _ledger(program_hash: str) -> PortfolioTrialLedger:
    policy = TopKScoreRiskCostPolicy(
        top_k=50,
        maximum_weight=0.05,
        risk_aversion=1000,
        turnover_regularization=0.1,
    )
    entries = tuple(
        PortfolioTrialLedgerEntry(
            stratum_id=f"stratum-{index // 24}",
            trial_number=index % 24,
            policy=policy,
            search_parameters={
                "top_k": 50,
                "cap_multiplier": 2.5,
                "risk_aversion": 1000.0,
                "turnover_regularization": 0.1,
            },
            state=TrialState.FAILED,
            evidence_hash=f"{1000 + index:064x}",
            failure_code="portfolio_strategy_lab.fixture_failure",
        )
        for index in range(144)
    )
    return seal_contract(
        PortfolioTrialLedger,
        "ledger_hash",
        program_hash=program_hash,
        phase="INITIAL_SEARCH",
        attempt_budget=144,
        entries=entries,
    )


def _dossier(program_hash: str) -> PortfolioEvidenceDossier:
    return seal_contract(
        PortfolioEvidenceDossier,
        "dossier_hash",
        program_hash=program_hash,
        baseline_count=18,
        attempted_count=144,
        completed_count=0,
        failed_count=144,
        pareto_region_count=0,
        robust_region_count=0,
        baseline_summary=("All fixed baselines were evaluated.",),
        stratum_summaries=("No numerical trial completed.",),
        contradictions=(),
        falsification_ledger=("No objective was forged for failed trials.",),
        owner_routing_evidence=("Repair numerical implementation.",),
        limitations=("Development only.",),
    )


def _publication_values(
    program: PortfolioExperimentProgram,
    ledger: PortfolioTrialLedger,
):
    program_hash = program.program_hash
    dossier = _dossier(program_hash)
    rfc = seal_contract(
        PortfolioResearchRFC,
        "rfc_hash",
        dossier_hash=dossier.dossier_hash,
        competing_hypotheses=("No completed numerical evidence is available.",),
        observed_anomaly_signature="All numerical attempts failed.",
        proposed_outcome=PortfolioResearchOutcome.STOP_WITH_EVIDENCE,
        expected_uncertainty_reduction="No further same-program compute is justified.",
        estimated_compute_budget="Zero additional attempts.",
        posterior_branching_rule="Stop and repair the responsible owner.",
        limitations_acknowledged=True,
    )
    review = seal_contract(
        PortfolioResearchReview,
        "review_hash",
        review_binding_hash=_HASHES[30],
        dossier_hash=dossier.dossier_hash,
        rfc_hash=rfc.rfc_hash,
        terminal_outcome=PortfolioResearchOutcome.STOP_WITH_EVIDENCE,
        counterfactual_outcome=PortfolioResearchOutcome.STOP_WITH_EVIDENCE,
        agent_changed_route=False,
        structural_experiment_executed=False,
        additional_attempt_count=0,
        summary="No completed numerical trials.",
        model_calls=2,
        executed_tool_names=(
            "submit_portfolio_research_rfc",
            "submit_portfolio_research_outcome",
        ),
        reviewed_at=datetime(2026, 8, 11, tzinfo=UTC),
    )
    stop = seal_contract(
        PortfolioScientificStop,
        "stop_hash",
        program_hash=program_hash,
        dossier_hash=dossier.dossier_hash,
        rfc_hash=rfc.rfc_hash,
        review_hash=review.review_hash,
        outcome=PortfolioResearchOutcome.STOP_WITH_EVIDENCE,
        reason="No completed numerical trials.",
    )
    bundle = seal_contract(
        PortfolioResearchProjectionBundle,
        "bundle_hash",
        program_hash=program_hash,
        ledger_hash=ledger.ledger_hash,
        dossier_hash=dossier.dossier_hash,
        review_hash=review.review_hash,
        terminal_artifact_hash=stop.stop_hash,
        terminal_kind="SCIENTIFIC_STOP",
        status="NO_PORTFOLIO_POLICY_READY_FOR_VALIDATION",
        attempted_count=144,
        completed_count=0,
        candidate_count=0,
        model_call_count=1,
        limitations=("Development only.",),
    )
    return dossier, rfc, review, stop, bundle


def _publish_children(
    store: PortfolioResearchArtifactStore,
    program: PortfolioExperimentProgram,
    values: tuple[
        PortfolioEvidenceDossier,
        PortfolioResearchRFC,
        PortfolioResearchReview,
        PortfolioScientificStop,
        PortfolioResearchProjectionBundle,
    ],
) -> None:
    dossier, rfc, review, stop, _ = values
    store.publish(
        category="programs",
        value=program,
        identity_field="program_hash",
    )
    store.publish(category="dossiers", value=dossier, identity_field="dossier_hash")
    store.publish(category="research-rfcs", value=rfc, identity_field="rfc_hash")
    store.publish(category="reviews", value=review, identity_field="review_hash")
    store.publish(category="scientific-stops", value=stop, identity_field="stop_hash")


def _install_historical_current_fixture(
    store: PortfolioResearchArtifactStore,
    *,
    bundle: PortfolioResearchProjectionBundle,
    ledger: PortfolioTrialLedger,
) -> None:
    """Install an old current pointer for readback tests without a production writer."""

    store.publish(category="projection-bundles", value=bundle, identity_field="bundle_hash")
    store.publish_ledger(ledger)
    marker = seal_contract(
        CurrentPortfolioResearchMarker,
        "marker_hash",
        program_hash=bundle.program_hash,
        ledger_hash=ledger.ledger_hash,
        bundle_hash=bundle.bundle_hash,
        status=bundle.status,
        published_at=datetime(2026, 8, 11, tzinfo=UTC),
    )
    store.publish(category="current/markers", value=marker, identity_field="marker_hash")
    pointer = seal_contract(
        CurrentPortfolioResearchPointer,
        "pointer_hash",
        marker_hash=marker.marker_hash,
    )
    store._atomic_write(
        store.pointer_path,
        json.dumps(pointer.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode(),
    )


def test_progress_publication_retries_a_transient_windows_reader_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_replace = progress_module.os.replace
    calls = 0

    def intermittently_locked(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError(5, "sharing violation")
        real_replace(source, destination)

    monkeypatch.setattr(progress_module.os, "name", "nt")
    monkeypatch.setattr(progress_module.os, "replace", intermittently_locked)
    projection = WorkspaceProgressPublisher(tmp_path).publish(
        WorkProgressUpdate(
            operation_id="portfolio-progress-retry",
            stage_id="execute_numerical_search",
            status="RUNNING",
            completed_units=1,
            total_units=144,
            unit_name="attempt",
        )
    )
    assert calls == 3
    assert projection.completed_units == 1


def test_a_read_back_projection_keeps_the_update_owner_invariants(tmp_path: Path) -> None:
    """A readback projection preserves the update owner's invariants."""

    from pydantic import ValidationError

    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    published = WorkspaceProgressPublisher(tmp_path).publish(
        WorkProgressUpdate(
            operation_id="portfolio-progress-invariants",
            stage_id="execute_numerical_search",
            status="RUNNING",
            completed_units=1,
            total_units=2,
            unit_name="attempt",
        )
    )
    document = published.model_dump(mode="json")
    assert WorkProgressProjection.model_validate(document) == published

    def sealed(**changes: object) -> dict[str, object]:
        values = {k: v for k, v in document.items() if k != "projection_hash"}
        values.update(changes)
        return {**values, "projection_hash": canonical_hash(values)}

    for malformed in (
        sealed(completed_units=0, total_units=0, percent_complete=0.0),
        sealed(completed_units=-1, total_units=-2, percent_complete=50.0),
        sealed(completed_units=3, total_units=2, percent_complete=150.0),
        sealed(updated_at="2026-08-02T06:30:00"),
        sealed(started_at="2026-08-02T06:30:00"),
        sealed(status="SUCCEEDED"),
        sealed(counters={"fits": -1}),
    ):
        with pytest.raises(ValidationError):
            WorkProgressProjection.model_validate(malformed)
    assert issubclass(ValidationError, ValueError)


def test_current_readback_rejects_mixed_valid_children(tmp_path: Path) -> None:
    store = PortfolioResearchArtifactStore(tmp_path)
    first_program = _program()
    first_ledger = _ledger(first_program.program_hash)
    first_values = _publication_values(first_program, first_ledger)
    _, _, _, _, first_bundle = first_values
    _publish_children(store, first_program, first_values)
    _install_historical_current_fixture(
        store,
        bundle=first_bundle,
        ledger=first_ledger,
    )
    first = store.read_current()
    assert first is not None
    assert first.bundle.model_view()["candidate_count"] == 0
    second_program = _program(3)
    second_ledger = _ledger(second_program.program_hash)
    second_values = _publication_values(second_program, second_ledger)
    _, _, _, _, second_bundle = second_values
    store.publish_ledger(second_ledger)
    _publish_children(store, second_program, second_values)
    store.publish(category="projection-bundles", value=second_bundle, identity_field="bundle_hash")
    mixed_marker = seal_contract(
        CurrentPortfolioResearchMarker,
        "marker_hash",
        program_hash=second_bundle.program_hash,
        ledger_hash=first_ledger.ledger_hash,
        bundle_hash=second_bundle.bundle_hash,
        status=second_bundle.status,
        published_at=datetime(2026, 8, 11, tzinfo=UTC),
    )
    store.publish(category="current/markers", value=mixed_marker, identity_field="marker_hash")
    pointer = seal_contract(
        CurrentPortfolioResearchPointer,
        "pointer_hash",
        marker_hash=mixed_marker.marker_hash,
    )
    store._atomic_write(
        store.pointer_path,
        json.dumps(pointer.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode(),
    )
    with pytest.raises(PortfolioPublicationError, match="current_lineage_invalid"):
        store.read_current()


def test_current_readback_requires_complete_dossier_rfc_and_review(tmp_path: Path) -> None:
    store = PortfolioResearchArtifactStore(tmp_path)
    program = _program()
    ledger = _ledger(program.program_hash)
    values = _publication_values(program, ledger)
    _, _, review, _, bundle = values
    _publish_children(store, program, values)
    _install_historical_current_fixture(store, bundle=bundle, ledger=ledger)
    (store.root / "reviews" / f"{review.review_hash}.json").unlink()
    # Missing sealed work offers recovery; it does not accuse a changed workspace.
    with pytest.raises(PortfolioPublicationError, match="artifact_absent"):
        store.read_current()


@pytest.mark.parametrize("reader", ["model", "columns", "packed", "document"])
def test_portfolio_artifact_readers_keep_missing_corrupt_and_invalid_distinct(
    tmp_path: Path, reader: str
) -> None:
    """BEHAVIOUR: every adapter retains the sealed-work recovery route for absence."""
    from alphalattice.interface.local_application.cli_contract import refusal_words

    store = PortfolioResearchArtifactStore(tmp_path)
    identity = "a" * 64
    category = "trial-evidence" if reader == "model" else "qa"
    load = {
        "model": lambda value: store.load_trial_evidence(value),
        "columns": lambda value: store.load_columns(category=category, content_hash=value),
        "packed": lambda value: store.load_packed_bytes(category=category, content_hash=value),
        "document": lambda value: store.load_document(category=category, content_hash=value),
    }[reader]
    with pytest.raises(
        PortfolioPublicationError, match=r"^portfolio_strategy_lab\.artifact_absent$"
    ):
        load(identity)
    assert "restore" in refusal_words("portfolio_strategy_lab.artifact_absent")["detail"]
    with pytest.raises(PortfolioPublicationError, match=r"^content_store\.identity_invalid$"):
        load("bad-hash")
    folder = store.root / category
    folder.mkdir(parents=True)
    extension = "parquet" if reader in {"columns", "packed"} else "json"
    (folder / f"{identity}.{extension}").write_bytes(b"corrupt")
    with pytest.raises(
        PortfolioPublicationError, match=r"^portfolio_strategy_lab\.artifact_tampered$"
    ):
        load(identity)


@pytest.mark.parametrize("lookup", ["execution", "program", "plan", "result", "outcome", "receipt"])
def test_indexed_missing_content_is_lost_work_and_not_a_tampered_index(
    tmp_path: Path, lookup: str
) -> None:
    """BEHAVIOUR: all six persisted index readers distinguish a lost child from a bad index."""

    from alphalattice.control.workspace_runtime.content_store import ContentAddressedStoreError
    from alphalattice.investment.portfolio_strategy_lab.publication.advancement_ledger import (
        AdvancementLedgerStore,
    )
    from alphalattice.investment.portfolio_strategy_lab.publication.portfolio_ledger import (
        PortfolioLedgerStore,
    )
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    program, child = "a" * 64, "b" * 64
    spec, authorities = "c" * 64, "d" * 64
    store = PortfolioLedgerStore(tmp_path)
    advancement = AdvancementLedgerStore(tmp_path)
    if lookup == "execution":
        path = store.root / "by-program-execution" / f"{program}.json"
        entry = {"program_hash": program, "ledger_hash": child}
        load = partial(store.find_execution_for_program, program)
    elif lookup == "program":
        key = canonical_hash({"holdings_spec_hash": spec, "authorities_hash": authorities})
        path = store.root / "by-holdings" / f"{key}.json"
        entry = {"holdings_spec_hash": spec, "authorities_hash": authorities, "program_hash": child}
        load = partial(
            store.find_program_for, holdings_spec_hash=spec, authorities_hash=authorities
        )
    elif lookup == "plan":
        key = canonical_hash({"spec_hash": spec, "authorities_hash": authorities})
        path = store.root / "by-plan" / f"{key}.json"
        entry = {"spec_hash": spec, "authorities_hash": authorities, "result_hash": child}
        load = partial(store.find_planned_result, spec_hash=spec, authorities_hash=authorities)
    elif lookup == "result":
        key = canonical_hash({"program_hash": program, "spec_hash": spec})
        path = store.root / "by-request" / f"{key}.json"
        entry = {"program_hash": program, "spec_hash": spec, "result_hash": child}
        load = partial(store.find_result_for, program_hash=program, spec_hash=spec)
    elif lookup == "outcome":
        path = advancement.root / "index/by-program-lane" / program / "RISK.json"
        entry = {"program_hash": program, "lane": "RISK", "receipt_hash": child}
        load = partial(advancement.find_outcome_for, program_hash=program, lane="RISK")
    else:
        path = advancement.root / "index/by-program" / f"{program}.json"
        entry = {"program_hash": program, "receipt_hash": child}
        load = partial(advancement.find_receipt_for, program)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(entry))
    with pytest.raises(ContentAddressedStoreError) as lost:
        load()
    assert str(lost.value) == f"content_store.artifact_missing:{child}"
    path.write_text("{}")
    with pytest.raises(ValueError, match="index_tampered"):
        load()
