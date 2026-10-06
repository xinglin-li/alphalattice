"""Dated Risk axes compose existing finite estimators and replayable artifact graphs."""

from datetime import date, timedelta
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pyarrow as pa
import pytest

from alphalattice.control.product_host.research_authoring.risk_reports import RiskReportLinks
from alphalattice.investment.risk_research.contracts import (
    CausalRiskReturnSurface,
    RiskUniverseEpoch,
    seal_contract,
)
from alphalattice.investment.risk_research.estimators.capability import CANONICAL_NO_RANDOMNESS_SEED
from alphalattice.investment.risk_research.estimators.catalog import (
    build_installed_risk_estimator_catalog,
)
from alphalattice.investment.risk_research.estimators.domains import COVARIANCE_RECIPE_SCHEMA_ID
from alphalattice.investment.risk_research.experiments.compiler import RiskExperimentCompiler
from alphalattice.investment.risk_research.experiments.execution import (
    RiskDevelopmentExecutor,
    RiskExperimentExecutor,
)
from alphalattice.investment.risk_research.experiments.series import (
    SERIES_CATEGORY,
    RiskDevelopmentSeries,
)
from alphalattice.investment.risk_research.experiments.verification import RiskEvidenceVerifier
from alphalattice.investment.risk_research.surfaces.artifacts import RiskArtifactStore
from alphalattice.investment.risk_research.surfaces.returns import (
    CausalRiskReturnReader,
    RiskReturnArtifactStore,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExecutionEvidence,
    ResearchExperimentEnvelope,
    ResolvedListingScope,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)


def _authority():
    days = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(318))
    authority = ResolvedResearchAuthority.create(
        data_snapshot_handle="fixture",
        universe_handle="fixture",
        panel_snapshot_hash="a" * 64,
        panel_manifest_ref="playpen://fixture/panel",
        universe_revision_sha256="b" * 64,
        ordered_listing_ids=("A", "B", "C"),
        sessions=days[314:317],
        source_watermark_hash="c" * 64,
    )
    return days, authority


def test_resolved_scopes_keep_legacy_identity_and_reject_missing_or_reordered_coverage():
    days, old = _authority()
    payload = old.model_dump(mode="json")
    assert "listing_scopes" not in payload
    assert old.authority_hash == canonical_hash(
        {k: v for k, v in payload.items() if k != "authority_hash"}
    )
    scopes = (
        ResolvedListingScope(
            first_session=days[314], last_session=days[314], ordered_listing_ids=("A", "B", "C")
        ),
        ResolvedListingScope(
            first_session=days[315], last_session=days[316], ordered_listing_ids=("A", "B")
        ),
    )
    current = ResolvedResearchAuthority.create(
        **{**old.model_dump(exclude={"authority_hash"}), "listing_scopes": scopes}
    )
    assert current.authority_hash != old.authority_hash
    assert current.for_scope(scopes[1]).ordered_listing_ids == ("A", "B")
    assert current.for_scope(scopes[1]).sessions == days[315:317]
    for invalid in (scopes[:1], tuple(reversed(scopes))):
        with pytest.raises(ValueError, match="listing_scope_coverage_invalid"):
            ResolvedResearchAuthority.create(
                **{**old.model_dump(exclude={"authority_hash"}), "listing_scopes": invalid}
            )


def test_risk_series_computes_each_scope_once_and_verifies_every_child(tmp_path, monkeypatch):
    days, base = _authority()
    scopes = (
        ResolvedListingScope(
            first_session=days[314], last_session=days[314], ordered_listing_ids=("A", "B", "C")
        ),
        ResolvedListingScope(
            first_session=days[315], last_session=days[316], ordered_listing_ids=("A", "B")
        ),
    )
    authority = ResolvedResearchAuthority.create(
        **{**base.model_dump(exclude={"authority_hash"}), "listing_scopes": scopes}
    )
    values = np.random.default_rng(91).normal(0, 0.01, (len(days), 3))
    input_store = RiskReturnArtifactStore(tmp_path / "input")
    surfaces = {}
    for ids in (scopes[0].ordered_listing_ids, scopes[1].ordered_listing_ids):
        rows = []
        for i, day in enumerate(days):
            for j, listing in enumerate(ids):
                row = dict(
                    formation_session=day,
                    return_start_session=day - timedelta(days=1),
                    listing_id=listing,
                    symbol=listing,
                    open_total_return_log=float(values[i, j]),
                    entry_source_row_hash=canonical_hash((i, listing, "entry")),
                    exit_source_row_hash=canonical_hash((i, listing, "exit")),
                    action_set_hash="0" * 64,
                )
                rows.append({**row, "row_hash": canonical_hash(row)})
        chunk = input_store.publish_chunk(pa.Table.from_pylist(rows))
        epoch = seal_contract(
            RiskUniverseEpoch,
            "epoch_hash",
            market_profile_id="fixture",
            universe_manifest_revision=base.universe_revision_sha256,
            panel_snapshot_hash=base.panel_snapshot_hash,
            ordered_listing_ids=ids,
        )
        surfaces[ids] = seal_contract(
            CausalRiskReturnSurface,
            "surface_hash",
            epoch=epoch,
            first_formation_session=days[0],
            last_formation_session=days[-1],
            formation_count=len(days),
            source_watermark_hash="d" * 64,
            chunks=(chunk,),
            limitations=("SYNTHETIC_QA",),
        )
    envelope = ResearchExperimentEnvelope.create(
        kind="risk.covariance-development",
        schema_id="research-experiment",
        data_snapshot_handle="fixture",
        universe_handle="fixture",
        sessions={
            "start": days[314],
            "end": days[316],
            "as_of": {"session": days[-1], "phase": "OFFICIAL_CLOSE"},
        },
        budget={"maximum_candidates": 1, "maximum_numerical_calls": 3},
        determinism={
            "seed": CANONICAL_NO_RANDOMNESS_SEED,
            "thread_limit": 1,
            "network_disabled": True,
        },
        output_workspace="risk-output",
    )
    document = {
        "experiment": envelope.model_dump(mode="json"),
        "risk": {
            "estimator": {
                "capability": COVARIANCE_RECIPE_SCHEMA_ID,
                "parameters": {"ewma_decay": 0.94},
            }
        },
    }
    catalog = build_installed_risk_estimator_catalog()
    compiled = RiskExperimentCompiler(catalog).compile_development_program(
        envelope=envelope, document=document, authority=authority
    )
    binding = compiled.binding
    program = SealedResearchProgram.create(
        kind=envelope.kind,
        envelope_hash=envelope.envelope_hash,
        desk_program_hash=compiled.desk_program_hash,
        resolved_sessions=authority.sessions,
        catalog_hash=binding.catalog_hash,
        method_binding_hash=binding.development_binding_hash,
        parameter_domain_hash=binding.parameter_domain_hash,
        authority_hash=authority.authority_hash,
    )

    def freshness(**_):
        return "d" * 64

    executor = RiskExperimentExecutor(
        surface_provider=SimpleNamespace(published_surfaces=lambda: tuple(surfaces.values())),
        return_reader=CausalRiskReturnReader(tmp_path / "input"),
        sector_by_listing_id={"A": "one", "B": "one", "C": "two"},
        freshness_probe=freshness,
        estimators=catalog,
        scope_provider=lambda selected: ((surfaces[selected.ordered_listing_ids],), freshness),
    )
    output = tmp_path / "output"
    # Missing the second scope's realized next session must refuse before even
    # the valid first scope computes. The input writer's finite rules stay intact.
    second = surfaces[scopes[1].ordered_listing_ids]
    incomplete_chunk = input_store.publish_chunk(pa.Table.from_pylist(rows[:-2]))
    incomplete = seal_contract(
        CausalRiskReturnSurface,
        "surface_hash",
        **{
            **second.model_dump(exclude={"surface_hash"}),
            "epoch": second.epoch,
            "last_formation_session": days[-2],
            "formation_count": len(days) - 1,
            "chunks": (incomplete_chunk,),
        },
    )
    with monkeypatch.context() as patch:
        patch.setitem(surfaces, scopes[1].ordered_listing_ids, incomplete)
        patch.setattr(
            RiskDevelopmentExecutor, "execute", lambda *a, **k: pytest.fail("premature estimate")
        )
        with pytest.raises(AuthoringError, match="insufficient_next_sessions"):
            executor.execute(
                program=program, document=document, authority=authority, output_workspace=output
            )
    assert not output.exists()
    execute_scope = RiskDevelopmentExecutor.execute
    scope_calls = []

    def interrupted(self, **kwargs):
        if scope_calls:
            raise RuntimeError("synthetic_interruption_between_scopes")
        completed = execute_scope(self, **kwargs)
        scope_calls.append(completed.estimate_calls)
        return completed

    with monkeypatch.context() as patch:
        patch.setattr(RiskDevelopmentExecutor, "execute", interrupted)
        with pytest.raises(RuntimeError, match="synthetic_interruption_between_scopes"):
            executor.execute(
                program=program, document=document, authority=authority, output_workspace=output
            )
    assert scope_calls == [1]
    assert not (output / SERIES_CATEGORY).exists()
    result = executor.execute(
        program=program, document=document, authority=authority, output_workspace=output
    )
    assert result.numerical_call_count == 2  # The completed first scope is not recomputed.
    evidence = ResearchExecutionEvidence.create(
        kind=envelope.kind,
        program_hash=program.program_hash,
        desk_program_hash=program.desk_program_hash,
        method_binding_hash=program.method_binding_hash,
        authority_hash=authority.authority_hash,
        disposition="COMPUTED",
        numerical_call_count=result.numerical_call_count,
        artifact_uris=result.artifact_uris,
        formation_sessions=result.formation_sessions,
        desk_input_binding_hash=result.desk_input_binding_hash,
    )
    verifier = RiskEvidenceVerifier()

    def verify(value):
        verifier.verify(program=program, evidence=value, authority=None, output_workspace=output)

    verify(evidence)
    projected = verifier.projection(evidence=evidence, output_workspace=output)
    assert [len(v["ordered_listing_ids"]) for v in projected["risk_surface"]["scope_surfaces"]] == [
        3,
        2,
    ]
    assert len(projected["result"]["evaluations"]) == 3
    assert projected["result"]["evaluations"][1]["trace_ratio"] is None
    resumed = executor.execute(
        program=program, document=document, authority=authority, output_workspace=output
    )
    assert resumed.artifact_uris == result.artifact_uris and resumed.numerical_call_count == 0
    store = RiskArtifactStore(output)
    series = RiskDevelopmentSeries.model_validate(
        store.load_json(
            category=SERIES_CATEGORY, uri=result.artifact_uris[0], identity_field="series_hash"
        )
    )
    swapped = RiskDevelopmentSeries.create(
        **{
            **{k: getattr(series, k) for k in type(series).model_fields if k != "series_hash"},
            "members": tuple(reversed(series.members)),
        }
    )
    store.publish_json(
        category=SERIES_CATEGORY,
        payload=swapped.model_dump(mode="json"),
        identity_field="series_hash",
    )
    wrong = ResearchExecutionEvidence.create(
        **{
            **evidence.model_dump(exclude={"evidence_hash"}),
            "artifact_uris": (store.uri(SERIES_CATEGORY, swapped.series_hash),),
            "desk_input_binding_hash": swapped.input_binding_hash,
        }
    )
    with pytest.raises(AuthoringError, match="series_scope_mismatch"):
        verify(wrong)
    chunk = projected["risk_surface"]["scope_surfaces"][1]["chunks"][0]
    path = store.root / "covariance/chunks" / f"{chunk['content_hash']}.bin"
    original = path.read_bytes()
    try:
        path.unlink()
        with pytest.raises(AuthoringError, match="evidence_artifact_unverifiable"):
            verify(evidence)
    finally:
        path.write_bytes(original)
    verify(evidence)


def _synthetic_run(root, days, authority, values):
    """One development run over a synthetic full-axis surface; its matrix hashes and evaluations."""

    ids = authority.ordered_listing_ids
    input_store = RiskReturnArtifactStore(root / "input")
    rows = []
    for i, day in enumerate(days):
        for j, listing in enumerate(ids):
            row = dict(
                formation_session=day,
                return_start_session=day - timedelta(days=1),
                listing_id=listing,
                symbol=listing,
                open_total_return_log=float(values[i, j]),
                entry_source_row_hash=canonical_hash((i, listing, "entry")),
                exit_source_row_hash=canonical_hash((i, listing, "exit")),
                action_set_hash="0" * 64,
            )
            rows.append({**row, "row_hash": canonical_hash(row)})
    chunk = input_store.publish_chunk(pa.Table.from_pylist(rows))
    epoch = seal_contract(
        RiskUniverseEpoch,
        "epoch_hash",
        market_profile_id="fixture",
        universe_manifest_revision=authority.universe_revision_sha256,
        panel_snapshot_hash=authority.panel_snapshot_hash,
        ordered_listing_ids=ids,
    )
    surface = seal_contract(
        CausalRiskReturnSurface,
        "surface_hash",
        epoch=epoch,
        first_formation_session=days[0],
        last_formation_session=days[-1],
        formation_count=len(days),
        source_watermark_hash="d" * 64,
        chunks=(chunk,),
        limitations=("SYNTHETIC_QA",),
    )
    envelope = ResearchExperimentEnvelope.create(
        kind="risk.covariance-development",
        schema_id="research-experiment",
        data_snapshot_handle="fixture",
        universe_handle="fixture",
        sessions={
            "start": authority.sessions[0],
            "end": authority.sessions[-1],
            "as_of": {"session": days[-1], "phase": "OFFICIAL_CLOSE"},
        },
        budget={"maximum_candidates": 1, "maximum_numerical_calls": len(authority.sessions)},
        determinism={
            "seed": CANONICAL_NO_RANDOMNESS_SEED,
            "thread_limit": 1,
            "network_disabled": True,
        },
        output_workspace="risk-output",
    )
    document = {
        "experiment": envelope.model_dump(mode="json"),
        "risk": {
            "estimator": {
                "capability": COVARIANCE_RECIPE_SCHEMA_ID,
                "parameters": {"ewma_decay": 0.94},
            }
        },
    }
    catalog = build_installed_risk_estimator_catalog()
    compiled = RiskExperimentCompiler(catalog).compile_development_program(
        envelope=envelope, document=document, authority=authority
    )
    program = SealedResearchProgram.create(
        kind=envelope.kind,
        envelope_hash=envelope.envelope_hash,
        desk_program_hash=compiled.desk_program_hash,
        resolved_sessions=authority.sessions,
        catalog_hash=compiled.binding.catalog_hash,
        method_binding_hash=compiled.binding.development_binding_hash,
        parameter_domain_hash=compiled.binding.parameter_domain_hash,
        authority_hash=authority.authority_hash,
    )

    def freshness(**_):
        return "d" * 64

    executor = RiskExperimentExecutor(
        surface_provider=SimpleNamespace(published_surfaces=lambda: (surface,)),
        return_reader=CausalRiskReturnReader(root / "input"),
        sector_by_listing_id={"A": "one", "B": "one", "C": "two"},
        freshness_probe=freshness,
        estimators=catalog,
        scope_provider=lambda selected: ((surface,), freshness),
    )
    output = root / "output"
    result = executor.execute(
        program=program, document=document, authority=authority, output_workspace=output
    )
    evidence = ResearchExecutionEvidence.create(
        kind=envelope.kind,
        program_hash=program.program_hash,
        desk_program_hash=program.desk_program_hash,
        method_binding_hash=program.method_binding_hash,
        authority_hash=authority.authority_hash,
        disposition="COMPUTED",
        numerical_call_count=result.numerical_call_count,
        artifact_uris=result.artifact_uris,
        formation_sessions=result.formation_sessions,
        desk_input_binding_hash=result.desk_input_binding_hash,
    )
    projected = RiskEvidenceVerifier().projection(evidence=evidence, output_workspace=output)
    matrix_hashes = tuple(
        h for chunk in projected["risk_surface"]["chunks"] for h in chunk["matrix_hashes"]
    )
    return matrix_hashes, projected["result"]["evaluations"]


def test_the_realized_next_session_never_enters_the_formations_covariance(tmp_path):
    """T's covariance is estimated from returns through T; the realized T+1 row only
    evaluates it. The only row that may change a matrix at T is a row at or before T."""

    days, authority = _authority()
    formations = authority.sessions  # days[314:317]; the realized row of the last is days[317]
    values = np.random.default_rng(91).normal(0, 0.01, (len(days), 3))
    matrices, evaluations = _synthetic_run(tmp_path / "base", days, authority, values)
    assert len(matrices) == len(formations) == 3
    assert [e["next_session"] for e in evaluations] == [str(d) for d in days[315:318]]

    # The realized row after the last formation moves: every matrix stays.
    realized = values.copy()
    realized[317] += 0.05
    moved_matrices, moved = _synthetic_run(tmp_path / "realized", days, authority, realized)
    assert moved_matrices == matrices
    assert moved[:2] == evaluations[:2]
    realized_only = {
        "equal_weight_realized_squared_return",
        "sector_balanced_realized_squared_return",
        "gaussian_log_score_per_asset",
    }
    assert {k: v for k, v in moved[2].items() if k not in realized_only} == {
        k: v for k, v in evaluations[2].items() if k not in realized_only
    }
    assert all(moved[2][k] != evaluations[2][k] for k in realized_only)

    # The last formation's own row moves: only that formation's matrix moves, and the
    # earlier formation it is the realized row of is evaluated differently.
    own = values.copy()
    own[316] += 0.05
    own_matrices, own_evaluations = _synthetic_run(tmp_path / "own", days, authority, own)
    assert own_matrices[:2] == matrices[:2] and own_matrices[2] != matrices[2]
    assert own_evaluations[0] == evaluations[0]
    assert (
        own_evaluations[1]["equal_weight_realized_squared_return"]
        != evaluations[1]["equal_weight_realized_squared_return"]
    )
    assert own_evaluations[1]["matrix_hash"] == evaluations[1]["matrix_hash"]


def test_risk_report_links_disclose_subset_scopes_without_changing_legacy_shape():
    ids = ["A", "B", "C"]
    days = ["2026-08-03", "2026-08-04"]
    document = {"experiment": {"sessions": {"as_of": {"session": "2026-08-05"}}}}
    portfolio = {
        "task_id": str(UUID(int=1)),
        "receipt": {"receipt_hash": "1" * 64},
        "document": document,
        "portfolio_source": {
            "input_binding_hash": "2" * 64,
            "panel_snapshot_hash": "3" * 64,
            "universe_revision": "4" * 64,
            "ordered_listing_ids": ids,
            "formation_sessions": days,
        },
    }
    surface = {"surface_hash": "5" * 64, "ordered_listing_ids": ids, "formation_sessions": days}
    risk = {
        "status": "EXPERIMENT_PUBLISHED",
        "task_id": str(UUID(int=2)),
        "input_binding_hash": "2" * 64,
        "document": document,
        "risk_surface": surface,
        "risk_input": {"panel_snapshot_hash": "3" * 64, "universe_revision_sha256": "4" * 64},
        "result": {"diagnostics_hash": "6" * 64},
    }
    legacy = RiskReportLinks._link(portfolio, risk, "HUMAN").model_dump(mode="json")
    assert "assessed_listing_scopes" not in legacy
    assert legacy["link_hash"] == canonical_hash(
        {k: v for k, v in legacy.items() if k != "link_hash"}
    )
    retrospective = RiskReportLinks._link(
        portfolio,
        {
            **risk,
            "result": {
                **risk["result"],
                "evaluations": [{"formation_session": day} for day in days],
            },
        },
        "HUMAN",
        portfolio_window=True,
    )
    assert retrospective.report_window.formation_sessions == tuple(map(date.fromisoformat, days))
    assert retrospective.risk_surface_hash == legacy["risk_surface_hash"]
    assert retrospective.portfolio_receipt_hash == legacy["portfolio_receipt_hash"]
    for axis in (["A", "B"], ids):
        scoped = {
            **surface,
            "ordered_listing_ids": axis,
            "scope_surfaces": [
                {"ordered_listing_ids": axis, "formation_sessions": days[:1]},
                {"ordered_listing_ids": ["A", "B"], "formation_sessions": days[1:]},
            ],
        }
        linked = RiskReportLinks._link(portfolio, {**risk, "risk_surface": scoped}, "HUMAN")
        assert len(linked.assessed_listing_scopes) == 2
        assert linked.assessed_listing_scopes[-1].ordered_listing_ids == ("A", "B")
        assert linked.claim == "REPORT_REFERENCE_ONLY_NOT_ALLOCATION_OR_CURRENT_RISK"
    for axis in (["B", "A"], ["A", "D"], []):
        with pytest.raises(AuthoringError, match="input_or_axis_mismatch"):
            RiskReportLinks._link(
                portfolio,
                {**risk, "risk_surface": {**surface, "ordered_listing_ids": axis}},
                "HUMAN",
            )


def test_risk_report_link_listing_refuses_bad_siblings_and_keeps_valid_link(tmp_path):
    from alphalattice.control.product_host.research_authoring.risk_reports import (
        CATEGORY,
        RiskReportLink,
    )
    from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
        PortfolioResearchArtifactStore,
    )

    workspace = tmp_path / "workspace"
    output = workspace / "portfolio"
    output.mkdir(parents=True)
    portfolio_task_id = UUID(int=101)
    portfolio = {
        "status": "EXPERIMENT_PUBLISHED",
        "task_id": str(portfolio_task_id),
        "receipt": {"receipt_hash": "1" * 64},
        "document": {"experiment": {"output_workspace": "portfolio"}},
        "portfolio_source": {"fixture": True},
    }
    links = RiskReportLinks(workspace, lambda _task_id: portfolio)
    store = PortfolioResearchArtifactStore(output)

    def make_link(*, receipt_hash: str, risk_task_id: int):
        return RiskReportLink.create(
            portfolio_task_id=portfolio_task_id,
            portfolio_receipt_hash=receipt_hash,
            risk_task_id=UUID(int=risk_task_id),
            risk_surface_hash="2" * 64,
            risk_diagnostics_hash="3" * 64,
            input_binding_hash="4" * 64,
            risk_start=date(2026, 8, 3),
            risk_end=date(2026, 8, 3),
            risk_formation_count=1,
            portfolio_formation_count=1,
            attached_by="HUMAN",
        )

    healthy = make_link(receipt_hash=portfolio["receipt"]["receipt_hash"], risk_task_id=201)
    mismatched = make_link(receipt_hash="5" * 64, risk_task_id=202)
    for link in (healthy, mismatched):
        store.publish(category=CATEGORY, value=link, identity_field="link_hash")

    corrupt_hash = "a" * 64
    corrupt_path = store.root / CATEGORY / f"{corrupt_hash}.json"
    corrupt_path.write_text("{not json", encoding="utf-8")

    answer = links.listing(portfolio_task_id)

    assert answer["status"] == "AVAILABLE"
    assert [item["link_hash"] for item in answer["links"]] == [healthy.link_hash]
    assert [item["risk_report_hash"] for item in answer["export_requests"]] == [healthy.link_hash]
    refused = {item["link_hash"]: item for item in answer["refused_links"]}
    assert set(refused) == {mismatched.link_hash, corrupt_hash}
    assert refused[mismatched.link_hash]["failure_code"] == "risk_report.stored_subject_mismatch"
    assert refused[corrupt_hash]["failure_code"] == "portfolio_strategy_lab.artifact_tampered"
    for item in refused.values():
        assert item["status"] == "REFUSED"
        assert item["next_requests"]["links"] == {
            "operation": "EXPERIMENT_RISK_LINKS",
            "task_id": str(portfolio_task_id),
        }
        assert item["next_requests"]["workspace"]["operation"] == "WORKSPACE_SHOW"
        assert item["next_requests"]["backups"]["operation"] == "WORKSPACE_BACKUPS"


@pytest.mark.parametrize(
    "task_kind", ("research_strategy_preparation", "portfolio_public_development_replay")
)
def test_an_installed_book_names_the_admitted_risk_report_subject_before_the_action(
    tmp_path, capsys, task_kind
):
    """regression (V616): installed books offer the study boundary before confirmation;
    every association door refuses that selector by name and its saved way continues."""
    import gc
    import json
    import shutil
    from datetime import UTC, datetime
    from uuid import uuid4

    from alphalattice.control.task_control.contracts import (
        ResearchGoal,
        ResearchPlan,
        TaskInputEnvelope,
        WorkItemDefinition,
    )
    from alphalattice.interface.local_application.cli import main
    from tests.portfolio_strategy_lab.strategy_dates_support import date_host

    workspace = tmp_path / "workspace"
    try:
        with date_host(workspace) as live:
            registry = live.session.task_control_registry
            envelope = TaskInputEnvelope.create(
                task_kind=task_kind,
                input_schema_id="research-strategy-input",
                payload={"declaration": "synthetic sealed preparation"},
            )
            goal = ResearchGoal.create(
                goal_kind="PREPARE_RESEARCH_STRATEGY",
                input_hash=envelope.input_hash,
                deliverable_kind="LocalLifecyclePortfolioAuthority",
                summary="Synthetic installed-book selection",
            )
            plan = ResearchPlan.create(
                goal_hash=goal.goal_hash,
                workflow_definition_hash="1" * 64,
                verifier_catalog_hash="2" * 64,
                work_items=(
                    WorkItemDefinition.create(
                        stage_id="prepare", dependency_ids=(), verifier_id="synthetic.verify"
                    ),
                ),
            )
            task = registry.admit(
                input_envelope=envelope,
                goal=goal,
                plan=plan,
                observed_at=datetime(2026, 10, 3, 12, tzinfo=UTC),
            ).record
            task_count = len(registry.tasks())

            def cli(*arguments):
                code = main(
                    ["--workspace", str(workspace), "--view", "full", *arguments],
                    serve=lambda _: 99,
                )
                return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])

            for package in live.application.resolver.installed_packages().values():
                code, controls = cli("strategy-book", "controls", "--package", package.strategy_id)
                assert code == 0, controls
                selection = controls["data"]["report_reference_selection"]
                assert selection["available"] is False
                assert selection["admitted_subject"] == "COMPLETED_PORTFOLIO_STUDY"
                assert selection["failure_code"] == "risk_report.completed_portfolio_required"
                assert "installed strategy book" in selection["detail"]
                assert selection["next_action"] and selection["next_requests"] == {
                    "studies": {"operation": "EXPERIMENTS"}
                }

            for index, arguments in enumerate(
                (
                    ("add", "--risk-study", str(uuid4())),
                    (
                        "add",
                        "--risk-study",
                        str(uuid4()),
                        "--scope",
                        "POST_OBSERVED_PORTFOLIO_WINDOW",
                    ),
                    ("list",),
                    ("export", "--risk-report", "f" * 64),
                )
            ):
                saved = tmp_path / f"refusal-{index}.json"
                code, refused = cli(
                    "risk-link", *arguments, "--task", str(task.task_id), "--output", str(saved)
                )
                assert (code, refused["failure_code"]) == (
                    2,
                    "risk_report.completed_portfolio_required",
                ), refused
                assert "sealed receipt" in refused["detail"]
                assert "captured input" in refused["detail"]
                assert refused["data"]["next_action"]
                assert refused["next_requests"] == {"studies": {"operation": "EXPERIMENTS"}}
                code, continued = cli("request", "--from", str(saved), "--action", "studies")
                assert (code, continued["status"]) == (0, "AVAILABLE"), continued
                assert len(registry.tasks()) == task_count
    finally:
        gc.collect()
        if workspace.exists():
            assert workspace.resolve().is_relative_to(tmp_path.resolve())
            shutil.rmtree(workspace)


@pytest.mark.parametrize(
    ("risk", "named"),
    [
        (
            {
                "estimator": {
                    "capability": COVARIANCE_RECIPE_SCHEMA_ID,
                    "parameters": {},
                    "ewma_decay": 0.97,
                }
            },
            "risk.estimator.ewma_decay",
        ),
        (
            {
                "estimator": {"capability": COVARIANCE_RECIPE_SCHEMA_ID, "parameters": {}},
                "parameters": {"ewma_decay": 0.97},
            },
            "risk.parameters",
        ),
    ],
)
def test_a_key_the_compiler_does_not_read_is_refused_where_it_was_written(risk, named):
    """regression (V134, V249): a parameter written beside `parameters` entered the plan's
    identity and moved it while no number moved; it is refused at its place with the one code,
    and the refusal's words say where a parameter is written."""

    from alphalattice.control.product_host.composition.plain_refusals import explain

    days, authority = _authority()
    envelope = ResearchExperimentEnvelope.create(
        kind="risk.covariance-development",
        schema_id="research-experiment",
        data_snapshot_handle="fixture",
        universe_handle="fixture",
        sessions={
            "start": authority.sessions[0],
            "end": authority.sessions[-1],
            "as_of": {"session": days[-1], "phase": "OFFICIAL_CLOSE"},
        },
        budget={"maximum_candidates": 1, "maximum_numerical_calls": 3},
        determinism={
            "seed": CANONICAL_NO_RANDOMNESS_SEED,
            "thread_limit": 1,
            "network_disabled": True,
        },
        output_workspace="risk-output",
    )
    compiler = RiskExperimentCompiler(build_installed_risk_estimator_catalog())
    with pytest.raises(AuthoringError) as refused:
        compiler.compile_development_program(
            envelope=envelope,
            document={"experiment": envelope.model_dump(mode="json"), "risk": risk},
            authority=authority,
        )
    assert str(refused.value) == f"research_authoring.section_key_unknown:{named}"
    assert "inside `parameters`" in explain(str(refused.value))["detail"]
