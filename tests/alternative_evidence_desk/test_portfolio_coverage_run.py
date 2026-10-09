"""A fifty-name book reviewed once: every listing accounted for, in bounded units.

The controlled fifty-issuer fixture (`portfolio_coverage_support`) proves the
route's scheduling, isolation, accounting and reuse at fifty-name scale through
the real local service -- socket, session token, the one dispatcher, Task
Control -- with recorded documents, a citing automation actor for the analyst
answer and submitted CRO assessments. It proves nothing about financial recall
or real-world latency; those are the retained-filing drivers' to prove.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import urllib.parse
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from alphalattice.control.observation_runtime.telemetry.progress import WorkProgressProjection
from alphalattice.control.product_host.composition.evidence_review_application import PACKAGE_RULE
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.evidence.alternative_evidence.analysis.contracts import EvidenceTopic
from alphalattice.evidence.alternative_evidence.publication.contracts import (
    AlternativeEvidenceAnalysisPublication,
)
from alphalattice.evidence.alternative_evidence.runtime.coverage import (
    COVERAGE_RUN_PURPOSE,
    UNIT_LIMIT,
    AlternativeEvidenceCoverageRun,
    AlternativeEvidenceUnitFailure,
    coverage_intent_hash,
)
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    UNIT_FAILURE_CATEGORY,
    AlternativeEvidenceDocumentTaskAdapter,
)
from alphalattice.interface.local_application.cli import main as cli_main
from alphalattice.interface.local_application.cli_contract import refusal_words
from alphalattice.investment.portfolio_strategy_lab.reporting.static import format_book_weight
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    PortfolioReviewDossier,
)
from tests.alternative_evidence_desk.planted_corpus import _NOW, PLANTED_CLAIMS
from tests.alternative_evidence_desk.portfolio_coverage_support import (
    COVERAGE_BOOK_SIZE,
    COVERAGE_ENTITIES,
    COVERAGE_TOPICS,
    UNIT_COUNT,
    answer_unit,
    build_coverage_workspace,
    coverage_authority,
    coverage_documents,
    coverage_registry,
    http_body,
    run_one,
    start_coverage_service,
)
from tests.alternative_evidence_desk.review_dossiers import (
    _issue,
    _submission,
    answer_from_issues,
)
from tests.alternative_evidence_desk.review_http_support import _raise_interruption

_run_one = run_one
_http = http_body
_answer_unit = answer_unit


@pytest.fixture(scope="module")
def built_book(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """One real Portfolio path over the wide universe; built once for the module."""

    return build_coverage_workspace(tmp_path_factory.mktemp("coverage-book"))


@pytest.fixture
def book(built_book: Any, tmp_path: Path) -> Any:
    """Each test's own copy of the built workspace: its runtime, registry and
    artifacts start empty and end where the test left them."""

    workspace, report = built_book
    copy = tmp_path / "workspace"
    shutil.copytree(workspace, copy)
    return copy, report


def _embedding_passes(service: Any) -> int:
    return int(service.review.evidence_task_adapter.runtime.retrieval.passage_embedding_pass_count)


@pytest.mark.parametrize(
    "top_k,universe,rotation,held_count,unit_count,unreviewed_count",
    [(75, 300, 75, 150, 19, 142), (50, 120, 0, 50, 7, 42)],
    ids=["REV-19-units", "short-inline"],
)
def test_the_cro_cli_bundle_names_every_unreviewed_holding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    top_k: int,
    universe: int,
    rotation: int,
    held_count: int,
    unit_count: int,
    unreviewed_count: int,
) -> None:
    """CONTRACT (V609): naturally aggregate failed units, then read every real CLI file.

    This is REV's rotating three-tranche book, with one analysed eight-issuer
    unit and unavailable sources for the remainder; no dossier or renderer is
    injected. The short case keeps unavailable units inline; other lists may
    independently overflow.
    """
    entities = tuple(f"QAY{number:03d}" for number in range(1, held_count + 1))
    topics = tuple(EvidenceTopic)
    for index, entity in enumerate(entities):
        monkeypatch.setitem(COVERAGE_TOPICS, entity, topics[index % len(topics)])
    book_parameters = {
        "top_k": top_k,
        "universe": universe,
        "tranches": 3,
        "exit_rank": top_k * 2,
        "score_rotation": rotation,
    }
    workspace, report = build_coverage_workspace(tmp_path / "fixture", **book_parameters)
    planner = start_coverage_service(
        workspace,
        coverage_authority(workspace, report, entities=entities, documents=()),
        tmp_path / "planner",
        **book_parameters,
    )
    try:
        scope = planner.review.resolve_book(planner.review.default_selector()).scope
        healthy = tuple(
            issuer.entity_id
            for issuer in sorted(scope.selected_issuers, key=lambda value: value.weight_rank)[:8]
        )
    finally:
        planner.session.stop()
    service = start_coverage_service(
        workspace,
        coverage_authority(
            workspace,
            report,
            entities=entities,
            documents=coverage_documents(healthy),
            minimum_entity_coverage=1.0,
        ),
        tmp_path / "service",
        **book_parameters,
    )
    try:
        preview = run_one(service, {"operation": "EVIDENCE_PREVIEW"})
        assert preview["status"] == "EVIDENCE_PREPARATION_READY", preview
        assert len(preview["coverage"]["units"]) == unit_count
        run_one(service, preview["next_requests"]["prepare"])
        service.drain()
        section = run_one(service, {"operation": "EVIDENCE_CRO"})
        packets = [
            value for key, value in section["next_requests"].items() if key.startswith("packet_")
        ]
        assert len(packets) == 1
        answer_unit(service, packets[0])
        service.drain()
        section = run_one(service, {"operation": "EVIDENCE_CRO"})
        dossier = run_one(service, section["next_requests"]["dossier"])["dossier"]
        assert dossier["held_count"] == held_count
        reasons = dossier["coverage"]["unavailable_reasons"]
        assert len(reasons) == unit_count - 1
        unreviewed = {
            entity
            for reason in reasons
            for entity in reason.partition("(")[2].partition(")")[0].split(", ")
        }
        assert len(unreviewed) == unreviewed_count
        assert unreviewed == set(entities) - set(healthy)
        directory = tmp_path / "cro-bundle"
        request_file = tmp_path / "request.json"
        request_file.write_text(
            json.dumps(
                {
                    "operation": "AGENT_BUNDLE_PREPARE",
                    "agent_role": "CRO",
                    "bundle_directory": str(directory),
                }
            ),
            encoding="utf-8",
            newline="\n",
        )
        capsys.readouterr()
        assert (
            cli_main(
                [
                    "--workspace",
                    str(workspace),
                    "--view",
                    "full",
                    "request",
                    "--file",
                    str(request_file),
                ],
                serve=lambda _arguments: 99,
            )
            == 0
        )
        bundle = json.loads(capsys.readouterr().out)["data"]
        assert bundle["status"] == "AGENT_BUNDLE_READY", bundle
        files = {
            entry["name"]: (directory / entry["name"]).read_text(encoding="utf-8")
            for entry in bundle["files"]
        }
        assert set(files) == {path.name for path in directory.iterdir()}
        all_text = " ".join(" ".join(files.values()).split())
        assert {entity for entity in unreviewed if entity in all_text} == unreviewed
        for reason in reasons:
            assert " ".join(reason.split()) in all_text
        index = files["README.md"]
        overflow = [name for name in files if name.startswith("coverage-")]
        if unit_count > 13:
            assert overflow
            assert "6 more unavailable reason(s)" in index
            assert "complete list, with each unit's issuers and reason" in index
            assert all(f"`{name}`" in index for name in overflow)
            complete = " ".join(" ".join(files[name] for name in overflow).split())
            assert all(" ".join(reason.split()) in complete for reason in reasons)
        else:
            assert "more unavailable" not in index
            assert "\n".join(f"- {reason}" for reason in reasons) in index
            assert all("## Unavailable units and issuers" not in files[name] for name in overflow)
    finally:
        service.session.stop()


def test_a_fifty_name_book_is_prepared_analysed_and_reviewed_as_one_request(
    book: Any, tmp_path: Path
) -> None:
    """requirement: one review request over a fifty-listing book accounts for every
    listing; the eight-issuer axis cuts execution units, never the scope; the
    packets, analyses and the CRO dossier are one book's, exactly."""

    workspace, report = book
    authority = coverage_authority(workspace, report)
    service = start_coverage_service(workspace, authority, tmp_path)
    try:
        selected = {"result_hash": service.result_hash()}
        section = service.evidence_cro()
        assert section["state"] == "AWAITING_ALTERNATIVE_EVIDENCE"
        progress = section["coverage_progress"]
        assert progress["book_listings"] == COVERAGE_BOOK_SIZE
        assert progress["mapped_issuers"] == COVERAGE_BOOK_SIZE
        assert progress["unmapped_listings"] == 0
        assert progress["units_total"] == UNIT_COUNT
        assert progress["units_prepared"] == 0 and progress["complete"] is False
        assert [unit["state"] for unit in progress["units"]] == ["NOT_STARTED"] * UNIT_COUNT
        assert sum(unit["listing_count"] for unit in progress["units"]) == COVERAGE_BOOK_SIZE
        assert {len(unit["entity_ids"]) for unit in progress["units"]} == {UNIT_LIMIT, 2}

        # The preview describes the whole run: seven units, one binding, and
        # the same document over HTTP and the CLI path.
        preview = _run_one(service, section["next_requests"]["preview"])
        assert service.get("/api/evidence/preview?" + urllib.parse.urlencode(selected)) == preview
        assert preview["status"] == "EVIDENCE_PREPARATION_READY", preview
        assert preview["coverage"]["unit_count"] == UNIT_COUNT
        assert preview["coverage"]["units_prepared"] == 0
        assert preview["recorded_candidate_count"] == 2 * COVERAGE_BOOK_SIZE
        # The preview hands out what was approved; the run is the Host's packing.
        run_hash = preview["preparation_binding"]["run_hash"]
        assert (
            preview["next_requests"]["prepare"]["preparation_binding_hash"]
            == (preview["preparation_binding_hash"])
        )
        # The source inventory: two documents per issuer, none deferred -- an
        # issuer's capacity is its own policy budget, never the set divided by
        # the unit's issuers -- no issuer without; the units were packed from
        # each issuer's own count (two), so eight issuers of two fit one unit.
        inventory = preview["source_inventory"]
        assert inventory["issuers_with_source"] == COVERAGE_BOOK_SIZE
        assert inventory["issuers_without_source"] == []
        assert {row["documents"] for row in inventory["issuers"]} == {2}
        assert {unit["capacity_per_issuer"] for unit in inventory["units"]} == {12}
        assert all(unit["deferrals"] == [] for unit in inventory["units"])
        packing = preview["coverage"]["packing"]
        assert packing["rules_id"] == "alternative-evidence.coverage-unit-packing.v2"
        assert set(packing["source_counts"].values()) == {2}
        assert all("recorded library: 2" in v for v in packing["source_count_basis"].values())
        assert preview["preparation_binding"]["packing_rules_id"] == packing["rules_id"]
        assert {
            entity for unit in preview["coverage"]["units"] for entity in unit["ordered_entity_ids"]
        } == set(COVERAGE_ENTITIES)
        ids = [unit["unit_id"] for unit in preview["coverage"]["units"]]
        assert sorted(ids) == [f"u{index:02d}" for index in range(1, UNIT_COUNT + 1)]

        prepared = service.post("/api/evidence/prepare", _http(preview["next_requests"]["prepare"]))
        assert prepared["disposition"] == "ADMITTED", prepared
        task_id = UUID(prepared["task_id"])
        # Each packet comes with its Analyst bundle request (V295).
        assert set(prepared["next_requests"]) == {
            f"{kind}_{unit}" for unit in ids for kind in ("packet", "analyst_bundle")
        }
        assert service.evidence_cro()["state"] == "EVIDENCE_REFRESH_IN_PROGRESS"
        service.drain()
        task = service.registry.task(task_id)
        assert task.lifecycle is TaskLifecycle.SUCCEEDED
        assert task.input.payload["purpose"] == COVERAGE_RUN_PURPOSE
        assert task.input.payload["run_hash"] == run_hash
        adapter = service.review.evidence_task_adapter
        # The run's counts reached the workspace's progress projection (S3); every
        # unit's counted stages ended whole.
        latest = WorkProgressProjection.model_validate_json(
            (workspace / "artifacts" / "run-monitor" / "work-progress.json").read_bytes()
        )
        assert latest.operation_id == str(task_id) and latest.status == "SUCCEEDED"
        reported = adapter.progress.units(task_id)
        assert set(reported) == set(ids)
        assert all(not work.running and work.completed == work.total for work in reported.values())
        states = adapter.unit_states(task)
        assert {value["state"] for value in states.values()} == {"PREPARED"}, states
        assert _embedding_passes(service) == UNIT_COUNT, "one corpus pass per unit"
        # Units run heaviest holding first (the plan's order, which the runner
        # follows strictly); a unit's name is its membership, not its turn.
        first_stage = task.plan.work_items[0].stage_id
        scope = service.review.resolve_book(service.review.default_selector()).scope
        first_unit = adapter.run_of(task).unit(first_stage[:3])
        heaviest = min(scope.selected_issuers, key=lambda issuer: issuer.weight_rank)
        assert heaviest.entity_id in first_unit.ordered_entity_ids

        # Prepared: the section names every unit's packet and hands out the
        # bound read of each; asking to prepare again is the completed run.
        section = service.evidence_cro()
        assert section["state"] == "ANALYST_PACKET_PREPARED"
        progress = section["coverage_progress"]
        assert progress["run_hash"] == run_hash
        assert progress["units_prepared"] == UNIT_COUNT and progress["complete"] is True
        assert progress["issuers_prepared"] == COVERAGE_BOOK_SIZE
        resolved = service.review.resolve_book(service.review.default_selector())
        assert progress["prepared_weight"] == format_book_weight(
            math.fsum(value.ending_weight for value in resolved.projection.positions)
        )
        assert progress["analyzed_weight"] == format_book_weight(0.0)
        assert progress["units_analyzed"] == 0
        packets = {k: v for k, v in section["next_requests"].items() if k.startswith("packet_")}
        assert len(packets) == UNIT_COUNT
        again = _run_one(service, preview["next_requests"]["prepare"])
        assert again["disposition"] == "REUSED_EXACT" and again["task_id"] == str(task_id)
        assert _embedding_passes(service) == UNIT_COUNT
        preview_again = _run_one(service, section["next_requests"]["preview"])
        assert preview_again["prepared_task_id"] == str(task_id)
        assert preview_again["coverage"]["units_prepared"] == UNIT_COUNT
        # The run is the packing at its cutoff: when the counts the packing
        # is derived from move afterwards -- a run's own acquisition seals
        # plans at its cutoff, a later delta seals more -- its packets, its
        # captured intent and its progress still resolve to the run's own
        # units (the book journey had refused a nine-unit run's packets as
        # another book's once its own plans packed the cutoff into eight).
        review = service.review
        counting = review._source_counts
        review._source_counts = lambda scope, *, evidence_as_of: (
            dict.fromkeys(scope.ordered_entity_ids, 12),
            dict.fromkeys(scope.ordered_entity_ids, "drifted: 12"),
        )
        review._unit_obligation_cache.clear()
        try:
            drifted_units = review._units(
                resolved.scope, evidence_as_of=adapter.run_of(task).evidence_as_of
            )
            assert len(drifted_units) == UNIT_COUNT, "the sealed run, not the drifted counts"
            first_packet = next(iter(sorted(packets)))
            drifted_read = service.get(
                "/api/evidence/packet?"
                + urllib.parse.urlencode(
                    {
                        **selected,
                        "task_id": str(task_id),
                        "evidence_unit_id": packets[first_packet]["evidence_unit_id"],
                    }
                )
            )
            assert drifted_read["coverage_unit"]["unit_count"] == UNIT_COUNT
            still = _run_one(service, preview["next_requests"]["prepare"])
            assert still["disposition"] == "REUSED_EXACT" and still["task_id"] == str(task_id)
            assert service.evidence_cro()["coverage_progress"]["units_total"] == UNIT_COUNT
        finally:
            review._source_counts = counting
            review._unit_obligation_cache.clear()

        # Each unit is answered on its own packet; the analyses are published
        # one per unit and the section counts them apart from preparation.
        for answered, key in enumerate(sorted(packets), start=1):
            unit_id = packets[key]["evidence_unit_id"]
            exported = service.get(
                "/api/evidence/packet?"
                + urllib.parse.urlencode(
                    {**selected, "task_id": str(task_id), "evidence_unit_id": unit_id}
                )
            )
            assert exported == _run_one(service, packets[key])
            assert exported["coverage_unit"]["unit_id"] == unit_id
            assert exported["coverage_unit"]["unit_count"] == UNIT_COUNT
            assert exported["prepared_unit_id"] == unit_id
            assert exported["submission_template"]["evidence_unit_id"] == unit_id
            submitted = _answer_unit(service, packets[key])
            assert submitted["disposition"] == "ADMITTED"
            service.drain()
            if answered == 1:
                # Honest partial along the way: one unit analysed, the rest
                # prepared, and the review offered over what is there.
                partial = service.evidence_cro()
                assert partial["state"] == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW"
                assert partial["coverage_progress"]["units_analyzed"] == 1
                assert partial["coverage_progress"]["units_prepared"] == UNIT_COUNT
                assert "1 of" in partial["explanation"]
                assert (
                    len([k for k in partial["next_requests"] if k.startswith("packet_")])
                    == UNIT_COUNT - 1
                )
        analyses = service.review.artifacts.values(
            "analysis-publications", AlternativeEvidenceAnalysisPublication
        )
        assert len(analyses) == UNIT_COUNT
        assert _embedding_passes(service) == UNIT_COUNT, "answers embed nothing"
        section = service.evidence_cro()
        assert section["state"] == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW", section["state"]
        progress = section["coverage_progress"]
        assert progress["units_analyzed"] == UNIT_COUNT
        assert progress["issuers_analyzed"] == COVERAGE_BOOK_SIZE
        assert progress["units_reviewed"] == 0
        assert {unit["state"] for unit in progress["units"]} == {"ANALYZED"}
        assert all(unit["analysis_publication_hash"] for unit in progress["units"])

        # The dossier is one book's: seven exact children, every finding and
        # span qualified by its publication, coverage over all fifty listings.
        dossier_body = _run_one(service, section["next_requests"]["dossier"])
        assert dossier_body["status"] == "CRO_DOSSIER_READY", dossier_body
        assert (
            service.get("/api/cro/dossier?" + urllib.parse.urlencode(selected))["dossier"]
            == dossier_body["dossier"]
        )
        dossier = PortfolioReviewDossier.model_validate(dossier_body["dossier"])
        assert len(dossier.evidence_children) == UNIT_COUNT
        assert len(dossier.issuers) == COVERAGE_BOOK_SIZE
        assert dossier.coverage.selected_issuer_coverage == 1.0
        assert dossier.coverage.reviewed_ending_weight_coverage == 1.0
        assert dossier.coverage.mapping_coverage == 1.0
        assert dossier.mapping_failure_count == 0
        assert dossier.coverage.unavailable_reasons == ()
        assert {
            entity for finding in dossier.findings for entity in finding.affected_entities
        } == set(COVERAGE_ENTITIES)
        qualifiers = {
            child.analysis_publication_hash[:8].upper() for child in dossier.evidence_children
        }
        assert all(
            finding.finding_handle.split("-")[1].removeprefix("P") in qualifiers
            for finding in dossier.findings
        )
        assert all(
            citation.span_handle.split(":")[0][1:] in qualifiers for citation in dossier.citations
        )
        assert len({citation.span_handle for citation in dossier.citations}) == len(
            dossier.citations
        )
        assert dossier.analysis_publication_hash not in {
            child.analysis_publication_hash for child in dossier.evidence_children
        }

        # The CRO answer cites a qualified handle and is published against
        # exactly these seven analyses; the export replays each of them.
        finding = next(f for f in dossier.findings if COVERAGE_ENTITIES[0] in f.affected_entities)
        assessment = answer_from_issues(
            dossier, _submission(_issue(finding.finding_handle, COVERAGE_ENTITIES[0]))
        ).model_dump(mode="json")
        reviewed = service.post(
            "/api/cro/assessment",
            _http({**dossier_body["submission_template"], "review_answer": assessment}),
        )
        assert reviewed["disposition"] == "ADMITTED", reviewed
        service.drain()
        section = service.evidence_cro()
        assert section["state"] == "REVIEW_PUBLISHED", section
        assert section["coverage_progress"]["units_reviewed"] == UNIT_COUNT
        assert {unit["state"] for unit in section["coverage_progress"]["units"]} == {"REVIEWED"}
        publication_hash = section["review_publication_hash"]
        export = service.get(
            "/api/evidence-cro/export?"
            + urllib.parse.urlencode({**selected, "review_publication_hash": publication_hash})
        )
        assert export["review_status"] == "EXACT_HISTORICAL_READBACK"
        assert export["evidence"]["aggregate_publication_hash"] == dossier.analysis_publication_hash
        assert [child["unit_id"] for child in export["evidence"]["children"]] == [
            child.unit_id for child in dossier.evidence_children
        ]
        exported_handles = {span["span_handle"] for span in export["evidence"]["verified_spans"]}
        assert {citation.span_handle for citation in dossier.citations} <= exported_handles
        assert finding.finding_handle in export["html"]
        # The historical review reopens exactly, with no current eligibility.
        historical = service.evidence_cro(review_publication_hash=publication_hash)
        assert historical["state"] == "REVIEW_PUBLISHED"
        assert historical["available_actions"] == []
    finally:
        service.session.stop()


def test_one_unit_failure_is_recorded_and_the_rest_of_the_book_goes_on(
    book: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """requirement: an independent unit failure does not block healthy issuers;
    the failed unit stays in every denominator with its owner's failure code,
    and preparing again keeps the units that completed. The unit short of its
    sources names the coverage it reached and needed and the issuer without one,
    the preview predicted it, and the readback words it with its ways on (V541).
    Once its source is restored, the readback uses the verified retry and keeps
    the original failed unit as history. A completed Task names its failed
    units, without equating verified failure receipts with prepared units."""

    workspace, report = book
    missing = COVERAGE_ENTITIES[16]
    documents = coverage_documents(tuple(e for e in COVERAGE_ENTITIES if e != missing))
    authority = coverage_authority(
        workspace, report, documents=documents, minimum_entity_coverage=1.0
    )
    service = start_coverage_service(workspace, authority, tmp_path)
    try:
        preview = _run_one(service, {"operation": "EVIDENCE_PREVIEW"})
        assert preview["source_inventory"]["issuers_without_source"] == [missing]
        assert preview["source_inventory"]["issuers_with_source"] == COVERAGE_BOOK_SIZE - 1
        # Said before the run: the one unit short of the floor, with the code it fails and the
        # issuer without a source, and the ways on; the other units are offered (V541).
        units = preview["coverage"]["units"]
        unit = next(value for value in units if missing in value["ordered_entity_ids"])
        issuers = len(unit["ordered_entity_ids"])
        code = (
            "alternative_evidence.minimum_entity_coverage_not_met:"
            f"{issuers - 1} of {issuers} issuers hold a source, {issuers} needed"
        )
        assert preview["coverage"]["units_short_of_sources"] == [
            {"unit_id": unit["unit_id"], "failure_code": code, "issuers_without_source": [missing]}
        ]
        assert f"1 of {UNIT_COUNT} units fall short" in preview["source_limit"]
        # Official acquisition first; a book of seven units takes no package, since units under
        # different packages cannot be reviewed together (V546).
        ways = preview["source_ways"]
        assert "package" not in ways and ways["package_rule"] == PACKAGE_RULE
        assert ways["official"]["serve"].endswith("serve --sec-network-consent")
        assert preview["status"] == "EVIDENCE_PREPARATION_READY" and "failure_code" not in preview
        prepared = _run_one(service, preview["next_requests"]["prepare"])
        assert prepared["disposition"] == "ADMITTED"
        service.drain()
        task_id = UUID(prepared["task_id"])
        task = service.registry.task(task_id)
        assert task.lifecycle is TaskLifecycle.SUCCEEDED, task.failure_code
        adapter = service.review.evidence_task_adapter
        states = adapter.unit_states(task)
        failed = [value for value in states.values() if value["state"] == "FAILED"]
        assert len(failed) == 1 and missing in failed[0]["ordered_entity_ids"]
        assert failed[0]["failure_code"] == code
        assert failed[0]["failed_stage"] == "acquire_source_evidence"
        assert failed[0]["uncovered_entity_ids"] == (missing,)
        assert sum(1 for value in states.values() if value["state"] == "PREPARED") == UNIT_COUNT - 1
        assert _embedding_passes(service) == UNIT_COUNT - 1
        failures = service.review.artifacts.values(
            UNIT_FAILURE_CATEGORY, AlternativeEvidenceUnitFailure
        )
        assert len(failures) == 1 and failures[0].unit_id == failed[0]["unit_id"]
        assert failures[0].uncovered_entity_ids == (missing,)
        assert failures[0].exception_type == "UnitSourcesShort"
        assert failures[0].exception_message == code

        completed = _run_one(service, {"operation": "STATUS", "task_id": task_id})
        assert completed["lifecycle"] == "SUCCEEDED"
        assert completed["verified_stage_count"] == completed["total_stage_count"]
        assert completed["units_failed"] == 1
        assert completed["detail"].startswith(
            f"Evidence preparation completed with 1 failed unit(s): "
            f"{failed[0]['unit_id']} ({issuers} issuers)."
        )
        assert completed["failed_units"] == [
            {
                "unit_id": failed[0]["unit_id"],
                "issuer_count": issuers,
                "failure_code": code,
                **refusal_words(code),
            }
        ]
        assert completed["next_requests"]["coverage"] == {"operation": "EVIDENCE_CRO"}
        assert completed["next_requests"]["network"] == {"operation": "NETWORK_ACCESS"}
        assert "preview" not in completed["next_requests"]
        assert completed["source_ways"] == ways
        listed = _run_one(service, {"operation": "TASKS"})
        assert (
            next(row for row in listed["tasks"] if row["task_id"] == str(task_id))["detail"]
            == completed["detail"]
        )
        capsys.readouterr()
        assert (
            cli_main(
                [
                    "--workspace",
                    str(workspace),
                    "--view",
                    "full",
                    "--lang",
                    "zh",
                    "task",
                    "show",
                    str(task_id),
                ],
                serve=lambda _arguments: 99,
            )
            == 0
        )
        said = json.loads(capsys.readouterr().out)
        assert said["data"]["failed_units"] == completed["failed_units"]
        assert said["detail"].startswith("证据准备已完成\uff0c其中 1 个单元失败\uff1a")
        assert f"{failed[0]['unit_id']}\uff08{issuers} 家发行人\uff09" in said["detail"]

        section = _run_one(service, completed["next_requests"]["coverage"])
        assert section["state"] == "ANALYST_PACKET_PREPARED"
        progress = section["coverage_progress"]
        assert progress["units_failed"] == 1 and progress["complete"] is False
        assert progress["units_prepared"] == UNIT_COUNT - 1
        assert progress["issuers_failed"] == len(failed[0]["ordered_entity_ids"])
        assert progress["issuers_prepared"] + progress["issuers_failed"] == COVERAGE_BOOK_SIZE
        row = next(unit for unit in progress["units"] if unit["state"] == "FAILED")
        assert row["failure_code"] == failed[0]["failure_code"]
        assert row["packet_task_id"] is None
        # The failed unit says why and what to do, and names the issuer without a source; the
        # book offers the ways to cover it beside the units it can review (V541).
        assert row["issuers_without_source"] == [missing]
        assert row["detail"] == refusal_words(code)["detail"] and "`source_ways`" in row["detail"]
        assert row["next_action"] == "ASK_FOR_OFFICIAL_ACQUISITION_OR_A_COVERING_PACKAGE"
        entities = list(failed[0]["ordered_entity_ids"])
        assert "package" not in section["source_ways"] and entities
        assert all("detail" not in unit for unit in progress["units"] if unit["state"] != "FAILED")
        packets = {k: v for k, v in section["next_requests"].items() if k.startswith("packet_")}
        assert len(packets) == UNIT_COUNT - 1
        assert f"packet_{failed[0]['unit_id']}" not in packets
        # The failed unit's packet does not read: the answer is a refusal naming the code, with
        # the book's coverage read as the way on (V388).
        refusal = _run_one(
            service,
            {
                "operation": "EVIDENCE_PACKET",
                "task_id": str(task_id),
                "evidence_unit_id": failed[0]["unit_id"],
            },
        )
        assert refusal["status"] == "REFUSED", refusal
        assert refusal["failure_code"] == "alternative_evidence.unit_not_prepared:" + code, refusal
        assert refusal_words(code)["detail"] in refusal["detail"], refusal
        assert refusal["next_requests"]["coverage"]["operation"] == "EVIDENCE_CRO", refusal

        # Preparing again is not exact reuse (a unit is missing) and costs no
        # embedding: every completed unit is carried in; the failure recurs.
        again = _run_one(service, preview["next_requests"]["prepare"])
        assert again["disposition"] == "ADMITTED", again
        service.drain()
        second = service.registry.task(UUID(again["task_id"]))
        assert second.lifecycle is TaskLifecycle.SUCCEEDED
        retried = _run_one(service, {"operation": "STATUS", "task_id": second.task_id})
        assert retried["failed_units"] == completed["failed_units"]
        assert _embedding_passes(service) == UNIT_COUNT - 1
        first_receipts = {
            r.stage_id: r.evidence[0].content_hash for r in service.registry.stage_receipts(task_id)
        }
        second_receipts = {
            r.stage_id: r.evidence[0].content_hash
            for r in service.registry.stage_receipts(second.task_id)
        }
        prepared_stages = {
            key for key, value in first_receipts.items() if not key.startswith(failed[0]["unit_id"])
        }
        assert {key: second_receipts[key] for key in prepared_stages} == {
            key: first_receipts[key] for key in prepared_stages
        }

        # The healthy units are analysed and reviewed; the failed unit is
        # named as not reviewed and its listings stay in the denominator.
        for key in sorted(packets):
            _answer_unit(service, packets[key])
            service.drain()
        section = service.evidence_cro()
        assert section["state"] == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW"
        assert "of" in section["explanation"]
        dossier_body = _run_one(service, section["next_requests"]["dossier"])
        dossier = PortfolioReviewDossier.model_validate(dossier_body["dossier"])
        assert len(dossier.evidence_children) == UNIT_COUNT - 1
        assert len(dossier.issuers) == COVERAGE_BOOK_SIZE - len(failed[0]["ordered_entity_ids"])
        assert dossier.coverage.selected_issuer_coverage < 1.0
        assert dossier.coverage.reviewed_ending_weight_coverage < 1.0
        assert dossier.coverage.mapping_coverage == 1.0
        assert any(
            failed[0]["unit_id"] in reason and "not reviewed" in reason
            for reason in dossier.coverage.unavailable_reasons
        )
        assert missing not in {issuer.entity_id for issuer in dossier.issuers}
    finally:
        service.session.stop()

    service = start_coverage_service(
        workspace,
        coverage_authority(
            workspace, report, documents=coverage_documents(), minimum_entity_coverage=1.0
        ),
        tmp_path,
        clock=lambda: _NOW + timedelta(minutes=1),
    )
    try:
        preview = _run_one(service, {"operation": "EVIDENCE_PREVIEW"})
        restored = _run_one(service, preview["next_requests"]["prepare"])
        service.drain()
        adapter = service.review.evidence_task_adapter
        newest = service.registry.task(UUID(restored["task_id"]))
        assert newest.lifecycle is TaskLifecycle.SUCCEEDED
        finished = _run_one(service, {"operation": "STATUS", "task_id": newest.task_id})
        assert "failed_units" not in finished and "units_failed" not in finished
        assert {value["state"] for value in adapter.unit_states(newest).values()} == {"PREPARED"}
        assert adapter.unit_states(service.registry.task(task_id)) == states
        assert failures[0] in service.review.artifacts.values(
            UNIT_FAILURE_CATEGORY, AlternativeEvidenceUnitFailure
        )
        section = _run_one(service, {"operation": "EVIDENCE_CRO"})
        progress = section["coverage_progress"]
        assert progress["units_failed"] == 0 and progress["units_prepared"] == UNIT_COUNT
        row = next(unit for unit in progress["units"] if unit["unit_id"] == failed[0]["unit_id"])
        assert row["state"] == "PREPARED" and row["packet_task_id"] == restored["task_id"]
        request = section["next_requests"][f"packet_{row['unit_id']}"]
        assert request["task_id"] == restored["task_id"]
        packet = _run_one(service, request)
        assert packet["status"] == "EVIDENCE_ANALYST_PACKET_READY"
        assert packet["prepared_task_id"] == restored["task_id"]
        assert packet["submission_template"]["evidence_unit_id"] == row["unit_id"]
    finally:
        service.session.stop()


def test_a_book_its_sources_cannot_cover_says_so_with_its_ways_on(
    book: Any, tmp_path: Path
) -> None:
    """regression (V541, RR5d's FINDING 20:13): a real SEC package held filings for 3 of a
    78-issuer book's issuers; the preview offered the run, every unit then failed
    `alternative_evidence.minimum_entity_coverage_not_met` with no words, and the readback's
    only request was the same preview. The preview refuses a run no unit can prepare, naming
    each unit's code and issuers without a source, with the ways on; a run sent anyway fails
    each unit naming its reach and need; the readback says the book cannot be reviewed under
    the installed sources, with each unit's words and the ways on."""

    workspace, report = book
    held = COVERAGE_ENTITIES[0]
    authority = coverage_authority(
        workspace, report, documents=coverage_documents((held,)), minimum_entity_coverage=0.6
    )
    service = start_coverage_service(workspace, authority, tmp_path)
    try:
        preview = _run_one(service, {"operation": "EVIDENCE_PREVIEW"})
        code = "alternative_evidence.book_sources_short"
        assert preview["status"] == "EVIDENCE_PREREQUISITES_MISSING", preview["status"]
        assert preview["failure_code"] == code
        assert preview["detail"] == refusal_words(code)["detail"]
        assert preview["next_action"] == "ASK_FOR_OFFICIAL_ACQUISITION_OR_A_COVERING_PACKAGE"
        assert "prepare" not in preview["next_requests"]
        units = preview["coverage"]["units"]
        short = preview["coverage"]["units_short_of_sources"]
        assert [value["unit_id"] for value in short] == [value["unit_id"] for value in units]
        assert "package" not in preview["source_ways"]

        # A run sent anyway, as a preview saved before would send it, fails each unit by name.
        sent = _run_one(
            service,
            {
                "operation": "EVIDENCE_PREPARE",
                "result_hash": service.result_hash(),
                "evidence_as_of": preview["evidence_as_of"],
                "preparation_binding_hash": preview["preparation_binding_hash"],
            },
        )
        assert sent["disposition"] == "ADMITTED", sent
        service.drain()
        section = service.evidence_cro()
        assert section["state"] == "AWAITING_ALTERNATIVE_EVIDENCE"
        assert section["explanation"].startswith(
            "This book cannot be reviewed under the installed sources:"
        ), section["explanation"]
        assert set(section["next_requests"]) == {"preview"}
        rows = section["coverage_progress"]["units"]
        assert [row["state"] for row in rows] == ["FAILED"] * UNIT_COUNT
        for row in rows:
            issuers = len(row["entity_ids"])
            sourced = int(held in row["entity_ids"])
            needed = next(count for count in range(issuers + 1) if count / issuers >= 0.6)
            assert row["failure_code"] == (
                "alternative_evidence.minimum_entity_coverage_not_met:"
                f"{sourced} of {issuers} issuers hold a source, {needed} needed"
            )
            assert row["issuers_without_source"] == [e for e in row["entity_ids"] if e != held]
            assert row["detail"] == refusal_words(row["failure_code"])["detail"]
            assert row["next_action"] == "ASK_FOR_OFFICIAL_ACQUISITION_OR_A_COVERING_PACKAGE"
        assert [row["unit_id"] for row in rows] == [value["unit_id"] for value in short]
        assert "package" not in section["source_ways"]
        assert section["source_ways"]["official"]["serve"].endswith("--sec-network-consent")
    finally:
        service.session.stop()


def test_units_prepared_under_two_packages_say_why_they_cannot_be_reviewed_together(
    book: Any, tmp_path: Path
) -> None:
    """regression (V546, V547; RR5d's FINDINGS 21:24 and 21:41): the lead installed one recorded
    package per unit, each replacing the last. A packet prepared under the first went stale on
    the next install, and its Analyst's answer was refused with a bare
    `alternative_evidence.task_resource_authority_mismatch`; the units answered under each
    package then stood at different cutoffs, the review's dossier refused with a bare
    `chief_risk_officer.dossier_cutoff_invalid`, and the book's readback failed. The stale packet
    is said where it is listed, and its answer is refused naming what moved, with its unit and
    the preview that prepares it again; the dossier, the CRO's bundle and the book's readback
    name the cutoff, the late citations and the earliest of their dates, with the way on. The
    invariant stands: no citation is read as of a cutoff before it was available."""

    workspace, report = book
    # The book's units, heaviest first, as its preview plans them: each package below covers
    # its own units, as RR5d's did.
    probe = start_coverage_service(workspace, coverage_authority(workspace, report), tmp_path)
    try:
        planned = _run_one(probe, {"operation": "EVIDENCE_PREVIEW"})["coverage"]["units"]
    finally:
        probe.session.stop()
    (answered, answered_ids), (pending, pending_ids), (later, later_ids) = (
        (unit["unit_id"], tuple(unit["ordered_entity_ids"])) for unit in planned[:3]
    )
    first = start_coverage_service(
        workspace,
        coverage_authority(
            workspace, report, documents=coverage_documents(answered_ids + pending_ids)
        ),
        tmp_path,
    )
    directory = tmp_path / "analyst"
    try:
        preview = _run_one(first, {"operation": "EVIDENCE_PREVIEW"})
        prepared = _run_one(first, preview["next_requests"]["prepare"])
        first.drain()
        _answer_unit(first, first.evidence_cro()["next_requests"][f"packet_{answered}"])
        first.drain()
        bundle = _run_one(
            first,
            {
                "operation": "AGENT_BUNDLE_PREPARE",
                "agent_role": "ANALYST",
                "task_id": prepared["task_id"],
                "evidence_unit_id": pending,
                "bundle_directory": str(directory),
            },
        )
        assert bundle["status"] == "AGENT_BUNDLE_READY", bundle
        adapter = first.review.evidence_task_adapter
        packet = adapter.prepared_packet(
            UUID(prepared["task_id"]), now=first.review.clock(), unit_id=pending
        )
        answer = adapter.resources.analysis_actor(packet=packet).answer.model_dump(mode="json")
    finally:
        first.session.stop()

    # Another package installed for the third unit, its filings available an hour after the
    # first preparation's cutoff, and the Host served again an hour after that.
    available = _NOW + timedelta(hours=1)
    documents = tuple(
        document.model_copy(update={"captured_at": available, "available_at": available})
        for document in coverage_documents(later_ids)
    )
    second = start_coverage_service(
        workspace,
        coverage_authority(workspace, report, documents=documents),
        tmp_path,
        clock=lambda: _NOW + timedelta(hours=2),
    )
    stale = "alternative_evidence.task_resource_authority_mismatch:package"
    try:
        # The packet prepared under the replaced package is said to be stale where it is listed;
        # the unit answered under it keeps its analysis.
        section = second.evidence_cro()
        rows = {row["unit_id"]: row for row in section["coverage_progress"]["units"]}
        assert rows[pending]["state"] == "PREPARED"
        assert rows[pending]["detail"] == refusal_words(stale)["detail"]
        assert rows[pending]["next_action"] == "PREPARE_THE_UNIT_AGAIN"
        assert "detail" not in rows[answered]
        # Neither its packet nor its Analyst's bundle is offered, which the Host would refuse
        # (U79); the preview its row names is offered bound (V567).
        offered = section["next_requests"]
        assert not {f"packet_{pending}", f"analyst_bundle_{pending}"} & set(offered), offered
        assert offered["preview"] == {**offered["dossier"], "operation": "EVIDENCE_PREVIEW"}
        # Its Analyst's answer is refused naming what moved, with its unit and the way on.
        refusal = _run_one(
            second,
            {
                "operation": "AGENT_ANSWER_SUBMIT",
                "bundle_directory": str(directory),
                "agent_answer": answer,
            },
        )
        assert (refusal["status"], refusal["failure_code"]) == ("REFUSED", stale), refusal
        assert refusal["detail"] == refusal_words(stale)["detail"]
        assert refusal["evidence_unit_id"] == pending
        assert refusal["next_requests"]["preview"]["operation"] == "EVIDENCE_PREVIEW"

        # The third unit prepared and answered under the new package, at its own cutoff.
        preview = _run_one(second, refusal["next_requests"]["preview"])
        assert preview["evidence_as_of"].startswith("2026-08-12T18:00:00"), preview
        _run_one(second, preview["next_requests"]["prepare"])
        second.drain()
        _answer_unit(second, second.evidence_cro()["next_requests"][f"packet_{later}"])
        second.drain()

        # Two analyses at different cutoffs: no review reads them together, and every read on
        # the route says so in words, with the way on, where the book's read had failed.
        cutoff, earliest = "2026-08-12T16:00:00Z", "2026-08-12T17:00:00Z"
        section = second.evidence_cro()
        assert section["state"] == "AWAITING_ALTERNATIVE_EVIDENCE", section["state"]
        assert set(section["next_requests"]) == {"preview"}
        dossier = _run_one(second, {"operation": "CRO_REVIEW_DOSSIER"})
        assert dossier["disposition"] == "REFUSED_EVIDENCE_CUTOFFS_DIFFER", dossier
        late = re.fullmatch(
            r"chief_risk_officer\.dossier_cutoff_invalid:(\d+) citations? of 1 unit after the "
            rf"cutoff {cutoff}, the earliest {earliest}",
            dossier["failure_code"],
        )
        assert late is not None and int(late[1]) >= 1, dossier["failure_code"]
        assert dossier["detail"] == section["explanation"]
        assert f"the earliest, {cutoff} ({answered})" in dossier["detail"]
        assert (
            f"of {later} became available after it, the earliest at {earliest}"
            in (dossier["detail"])
        )
        assert dossier["detail"].endswith(PACKAGE_RULE)
        assert dossier["next_requests"] == section["next_requests"]
        cro = _run_one(
            second,
            {
                "operation": "AGENT_BUNDLE_PREPARE",
                "agent_role": "CRO",
                "bundle_directory": str(tmp_path / "cro"),
            },
        )
        assert (cro["failure_code"], cro["detail"]) == (dossier["failure_code"], dossier["detail"])
    finally:
        second.session.stop()


def test_official_acquisition_reviews_a_wide_book_with_quiet_holdings_at_one_cutoff(
    book: Any, tmp_path: Path
) -> None:
    """requirement (V541's W1 question, V546; decided on evidence before RR5's repeat): official
    acquisition reads every holding's filing index at one cutoff, names the holdings that filed
    nothing in the window and packs only the others, so a quiet holding never counts against
    the installed floor. At the strictest floor -- every issuer of a unit holding a source -- a
    fifty-issuer book with three quiet holdings packs the other 47 into six units, prepares
    every one at that one cutoff, and its whole review's dossier compiles, the quiet holdings
    read as nothing filed and nothing left unreached."""

    from dataclasses import replace

    from alphalattice.control.product_host.composition.evidence_review_workspace import (
        live_evidence_policy,
    )
    from alphalattice.evidence.alternative_evidence.sources.sec_edgar import SecEdgarSource
    from tests.alternative_evidence_desk.sec_scenario_transport import (
        ScenarioFiling,
        SecScenarioTransport,
        filing_body,
    )

    workspace, report = book
    # One quiet holding in each of three of the planned units: each filed its two periodic
    # reports a quarter before the cutoff, outside the 30-day window.
    quiet = ("QA03", "QA20", "QA41")
    registry = {item.ticker: (item.cik, item.legal_name) for item in coverage_registry().entries}
    filings: dict[str, list[ScenarioFiling]] = {}
    bodies: dict[str, bytes] = {}
    for entity, (cik, _name) in registry.items():
        months = ("05", "04") if entity in quiet else ("08", "07")
        filings[cik] = [
            ScenarioFiling(
                f"{cik}-26-{50 + index:06d}",
                form,
                f"2026-{month}-01",
                f"2026-{month}-01T20:00:00.000Z",
                f"{entity.casefold()}-{form.casefold()}.htm",
                period,
            )
            for index, (form, month, period) in enumerate(
                zip(("10-Q", "10-K"), months, ("2026-03-28", "2025-09-27"), strict=True)
            )
        ]
        claim = PLANTED_CLAIMS[COVERAGE_TOPICS[entity]]
        for filing in filings[cik]:
            narrative = filing_body(filing.accession).decode()
            bodies[SecScenarioTransport.locator(cik, filing)] = narrative.replace(
                "<body>", f"<body><p>{claim}</p>", 1
            ).encode()
    transport = SecScenarioTransport(registry=registry, filings=filings, bodies=bodies)
    transport.retrieved_at = _NOW
    recorded = coverage_authority(workspace, report, minimum_entity_coverage=1.0)
    assert recorded.evidence_resources is not None
    authority = replace(
        recorded,
        evidence_resources=replace(
            recorded.evidence_resources, live_source=SecEdgarSource(transport)
        ),
        evidence_policy=live_evidence_policy(recorded.evidence_policy),
    )
    service = start_coverage_service(workspace, authority, tmp_path)
    try:
        preview = _run_one(service, {"operation": "EVIDENCE_PREVIEW"})
        assert preview["status"] == "EVIDENCE_PREPARATION_READY", preview
        assert preview.get("source_ways") is None
        prepared = _run_one(service, preview["next_requests"]["prepare"])
        assert prepared["disposition"] == "ADMITTED", prepared
        service.drain()
        task = service.registry.task(UUID(prepared["task_id"]))
        assert task.lifecycle is TaskLifecycle.SUCCEEDED, task.failure_code
        adapter = service.review.evidence_task_adapter
        run = adapter.run_of(task)
        assert run is not None and tuple(sorted(run.nothing_filed)) == quiet
        assert not set(quiet) & set(run.ordered_entity_ids)
        assert len(run.units) == -(-(COVERAGE_BOOK_SIZE - len(quiet)) // 8)
        assert {unit.request.evidence_as_of for unit in run.units} == {run.evidence_as_of}
        assert {value["state"] for value in adapter.unit_states(task).values()} == {"PREPARED"}

        section = service.evidence_cro()
        packets = {k: v for k, v in section["next_requests"].items() if k.startswith("packet_")}
        assert len(packets) == len(run.units)
        for key in sorted(packets):
            _answer_unit(service, packets[key])
            service.drain()
        section = service.evidence_cro()
        assert section["state"] == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW", section["explanation"]
        body = _run_one(service, section["next_requests"]["dossier"])
        dossier = PortfolioReviewDossier.model_validate(body["dossier"])
        assert dossier.evidence_as_of == run.evidence_as_of
        assert len(dossier.evidence_children) == len(run.units)
        assert {child.evidence_as_of for child in dossier.evidence_children} == {run.evidence_as_of}
        assert dossier.citations
        assert all(value.available_at <= dossier.evidence_as_of for value in dossier.citations)
        states = {issuer.entity_id: issuer.review_state for issuer in dossier.issuers}
        assert {entity: states[entity] for entity in quiet} == dict.fromkeys(quiet, "NOTHING_FILED")
        assert (dossier.coverage.nothing_filed_ending_weight_coverage or 0.0) > 0.0
        assert dossier.coverage.unreached_ending_weight_coverage == 0.0
    finally:
        service.session.stop()


def test_every_coverage_floor_is_judged_by_one_rule_that_leaves_quiet_holdings_out() -> None:
    """requirement (V587, TE12): the coverage floor is judged by one rule, `sources_short`,
    wherever it is judged -- a unit's preparation, the preview that predicts it and the
    installer's package -- and no owner divides by its issuers against the floor by hand. Each
    judge that reads an acquisition hands the rule the issuers an index read at the cutoff found
    quiet (`index_quiet`), asked only when the share falls short; the preview's units come from
    a packing that already left them out (V541). A quiet issuer leaves the share, a failed one
    stays in it."""

    import ast

    from alphalattice.evidence.alternative_evidence.runtime.coverage import sources_short

    root = Path(__file__).resolve().parents[2] / "src" / "alphalattice"
    judges: dict[str, bool] = {}
    by_hand: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        for function in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "sources_short":
                    judges[f"{relative}::{function.name}"] = any(
                        keyword.arg == "quiet" for keyword in node.keywords
                    )
                parts = list(ast.walk(node)) if isinstance(node, ast.Compare) else []
                divides = any(isinstance(p, ast.BinOp) and isinstance(p.op, ast.Div) for p in parts)
                if divides and any(
                    (isinstance(p, ast.Name) and p.id == "floor")
                    or (isinstance(p, ast.Attribute) and p.attr == "minimum_entity_coverage")
                    for p in parts
                ):
                    by_hand.add(f"{relative}::{function.name}")
    composition = "control/product_host/composition"
    assert judges == {
        f"{composition}/evidence_authority_setup.py::_materialize": True,
        f"{composition}/evidence_review_application.py::units_short_of_sources": False,
        "evidence/alternative_evidence/runtime/task_adapter.py::_execute_stage": True,
    }
    assert by_hand == {"evidence/alternative_evidence/runtime/coverage.py::sources_short"}

    asked: list[tuple[str, ...]] = []

    def quiet(uncovered: tuple[str, ...]) -> frozenset[str]:
        asked.append(uncovered)
        return frozenset({"E", "F"})

    book = ("A", "B", "C", "D", "E", "F", "G", "H")
    assert sources_short(book, {"A", "B", "C", "D", "G"}, floor=0.6, quiet=quiet) is None
    assert asked == [], "the quiet issuers are asked for only when the share falls short"
    assert sources_short(book, {"A", "B", "C", "D"}, floor=0.6, quiet=quiet) is None
    short = sources_short(book, {"A", "B", "C", "D"}, floor=0.7, quiet=quiet)
    assert short is not None and asked[-1] == ("E", "F", "G", "H")
    assert str(short) == (
        "alternative_evidence.minimum_entity_coverage_not_met:"
        "4 of 6 issuers hold a source, 5 needed, 2 filed nothing"
    )
    assert (short.held, short.issuers, short.needed, short.quiet) == (4, 6, 5, 2)
    assert short.uncovered == ("G", "H")


def _one_core(service: Any) -> None:
    """The workspace's CPU budget at one core, set through the owner the CLI's
    `cpu-budget --set 1` reaches: one unit prepares at a time (F1)."""

    body = service.session.operations.set_cpu_budget("1", chosen_by="HUMAN")
    assert (body["status"], body["cpu_budget"], body["chosen_by"]) == ("CPU_BUDGET", 1, "HUMAN")
    assert body["machine"]["processors"] >= 1 and body["a_book_now"]["units_at_once_at_most"] == 1


def test_an_interrupted_run_resumes_after_restart_without_repeating_completed_units(
    book: Any, tmp_path: Path
) -> None:
    """requirement: committed units survive an interruption and a real restart;
    recovery prepares only what was not committed, and the reweighted book
    reads the same units."""

    workspace, report = book
    authority = coverage_authority(workspace, report)
    first = start_coverage_service(workspace, authority, tmp_path)
    # The interruption is placed by the order the units run in: one at a time,
    # on a CPU budget of one core.
    _one_core(first)
    original = AlternativeEvidenceDocumentTaskAdapter.verify_stage

    def interrupt_fourth_unit(
        self: Any, *, task: Any, execution: Any, work_item: Any, evidence: Any
    ) -> Any:
        if work_item.stage_id.endswith("_admit_evidence_request") and (
            len({r.stage_id[:3] for r in self.registry.stage_receipts(task.task_id)}) == 3
        ):
            _raise_interruption()
        return original(
            self, task=task, execution=execution, work_item=work_item, evidence=evidence
        )

    AlternativeEvidenceDocumentTaskAdapter.verify_stage = interrupt_fourth_unit  # type: ignore[method-assign]
    try:
        preview = _run_one(first, {"operation": "EVIDENCE_PREVIEW"})
        prepared = _run_one(first, preview["next_requests"]["prepare"])
        first.drain()
    finally:
        AlternativeEvidenceDocumentTaskAdapter.verify_stage = original  # type: ignore[method-assign]
    task_id = UUID(prepared["task_id"])
    try:
        assert first.registry.task(task_id).lifecycle is TaskLifecycle.RECOVERY_REQUIRED
        assert _embedding_passes(first) == 3
        section = first.evidence_cro()
        assert section["state"] == "EVIDENCE_REFRESH_IN_PROGRESS"
        progress = section["coverage_progress"]
        assert progress["units_prepared"] == 3 and progress["units_pending"] == UNIT_COUNT - 3
        assert section["task_id"] == str(task_id)
    finally:
        first.session.stop()

    second = start_coverage_service(workspace, authority, tmp_path)
    _one_core(second)
    try:
        assert second.session.resumed_task_ids == (task_id,)
        second.drain()
        assert second.registry.task(task_id).lifecycle is TaskLifecycle.SUCCEEDED
        assert _embedding_passes(second) == UNIT_COUNT, "the three committed units were not redone"
        states = second.review.evidence_task_adapter.unit_states(second.registry.task(task_id))
        assert {value["state"] for value in states.values()} == {"PREPARED"}
        section = second.evidence_cro()
        assert section["state"] == "ANALYST_PACKET_PREPARED"
        assert section["coverage_progress"]["complete"] is True
        again = _run_one(second, preview["next_requests"]["prepare"])
        assert again["disposition"] == "REUSED_EXACT" and again["task_id"] == str(task_id)
        evidence_tasks = [
            task
            for task in second.registry.tasks()
            if task.task_kind == AlternativeEvidenceDocumentTaskAdapter.task_kind
        ]
        assert len(evidence_tasks) == 1, "resumed once, never admitted twice"
        # What the resumed preparation ran with, and why, is its receipt.
        ran = second.session.operations.cpu_budget()["last_preparation"]
        assert ran["task_id"] == str(task_id) and ran["units"] == UNIT_COUNT
        assert (ran["cpu_budget"], ran["units_at_once"], ran["threads_per_session"]) == (1, 1, 1)
        assert ran["reason"].startswith("set to 1 cores")
    finally:
        second.session.stop()


def test_a_single_unit_book_is_a_run_of_one_unit(tmp_path: Path) -> None:
    """requirement (C2, 2026-09-23): every book is a coverage run. A book that
    fits one unit is a run of one unit, with the run, the per-unit packet
    request and the readers a wider book has; its lone unit carries exactly
    the request, obligation and intent a single preparation of the book
    carried, so a single preparation's completion is that unit's completion.
    One unit has no progress table, and the preview still names its source
    check where a reader of a one-unit book looks for it."""

    from tests.alternative_evidence_desk.review_http_support import (
        build_authority,
        build_workspace,
        start_service,
    )

    workspace, report = build_workspace(tmp_path)
    authority = build_authority(tmp_path=tmp_path, report=report)
    service = start_service(workspace, authority, tmp_path)
    try:
        preview = _run_one(service, {"operation": "EVIDENCE_PREVIEW"})
        assert preview["coverage"]["unit_count"] == 1
        (planned,) = preview["coverage"]["units"]
        assert (
            preview["next_requests"]["prepare"]["preparation_binding_hash"]
            == (preview["preparation_binding_hash"])
        )
        section = service.evidence_cro()
        assert section["coverage_progress"] is None
        prepared = _run_one(service, preview["next_requests"]["prepare"])
        service.drain()
        task = service.registry.task(UUID(prepared["task_id"]))
        assert task.input.payload["purpose"] == COVERAGE_RUN_PURPOSE
        assert task.input.payload["unit_count"] == 1
        (run,) = service.review.artifacts.values(
            "evidence-coverage-runs", AlternativeEvidenceCoverageRun
        )
        (unit,) = run.units
        assert unit.preparation_intent_hash == planned["preparation_intent_hash"]
        packets = [
            value
            for value in prepared["next_requests"].values()
            if value.get("operation") == "EVIDENCE_PACKET"
        ]
        assert packets == [prepared["next_requests"][f"packet_{unit.unit_id}"]]
        assert packets[0]["task_id"] == prepared["task_id"]
        assert packets[0]["evidence_unit_id"] == unit.unit_id
        adapter = service.review.evidence_task_adapter
        assert adapter.completed_unit(unit.preparation_intent_hash) == (task.task_id, unit.unit_id)
        again = _run_one(service, {"operation": "EVIDENCE_PREVIEW"})
        assert again["source_check"] == again["coverage"]["units"][0]["source_check"]
        assert again["prepared_task_id"] == prepared["task_id"]
    finally:
        service.session.stop()


def test_a_preparation_admitted_before_every_book_was_a_run_is_carried_into_the_run(
    tmp_path: Path,
) -> None:
    """requirement (C2, retirement row H7): a one-unit book prepared as a single
    request before C2 -- the Task a workspace on the accepted baseline may hold
    -- is not prepared twice: the book's run finds that completion by the
    unit's intent and carries it, embedding nothing again. The single Task
    stays readable by its id; no new single preparation is ever admitted."""

    from dataclasses import replace

    from alphalattice.control.product_host.composition.evidence_review_application import (
        AlternativeEvidenceRefreshCommand,
    )
    from tests.alternative_evidence_desk.review_http_support import (
        build_authority,
        build_workspace,
        start_service,
    )

    workspace, report = build_workspace(tmp_path)
    authority = build_authority(tmp_path=tmp_path, report=report)
    service = start_service(workspace, authority, tmp_path)
    try:
        review = service.review
        chosen = review.default_selector(None)
        now = review.clock()
        coverage = review.resolve_coverage(chosen, evidence_as_of=now, prepare_only=True)
        (unit,) = coverage.run.units
        # Exactly the Task a pre-C2 build admitted for a one-unit book.
        policy = replace(review.evidence_policy, admit_model_review=False)
        legacy = AlternativeEvidenceRefreshCommand(
            application=review,
            obligation=unit.obligation,
            request=unit.request,
            admission=policy.admission(request=unit.request, admitted_at=now),
            prepare_only=True,
        )
        legacy_task = service.session.dispatcher.submit(legacy).task_id
        service.drain()
        single = service.registry.task(legacy_task)
        assert single.lifecycle is TaskLifecycle.SUCCEEDED
        assert single.input.payload["purpose"] == "PREPARE_PACKET"
        adapter = review.evidence_task_adapter
        assert adapter.completed_unit(unit.preparation_intent_hash) == (legacy_task, None)
        passes = _embedding_passes(service)

        # The run at the single preparation's own cutoff, as a preview captured then.
        template = _run_one(service, {"operation": "EVIDENCE_PREVIEW"})["next_requests"]["prepare"]
        prepared = _run_one(
            service,
            {
                **template,
                "evidence_as_of": now.isoformat(),
                "preparation_binding_hash": coverage_intent_hash(coverage.run),
            },
        )
        service.drain()
        task = service.registry.task(UUID(prepared["task_id"]))
        assert task.input.payload["purpose"] == COVERAGE_RUN_PURPOSE
        assert task.lifecycle is TaskLifecycle.SUCCEEDED
        assert _embedding_passes(service) == passes, "the single preparation was carried"
        states = adapter.unit_states(task)
        assert {value["state"] for value in states.values()} == {"PREPARED"}
        packet = _run_one(service, prepared["next_requests"][f"packet_{unit.unit_id}"])
        assert packet["status"] == "EVIDENCE_ANALYST_PACKET_READY"
        assert packet["prepared_task_id"] == prepared["task_id"]
        # the single Task's own packet still reads by its id
        assert adapter.prepared_packet(legacy_task, now=review.clock()) is not None
    finally:
        service.session.stop()


def test_the_book_ledger_reads_every_group_of_a_wide_book(book: Any, tmp_path: Path) -> None:
    """requirement (first-release integration T4, A11): a wide book's ledger
    is one read of every group of its newest coverage Task -- each group's
    cells the unit packet's own coverage cells, its packet named by Task and
    unit -- twenty groups a page; the read adds no Task and no write, and a
    page past the last refuses by name."""

    from dataclasses import replace

    from alphalattice.evidence.alternative_evidence.analysis.routing import TOPICS
    from alphalattice.evidence.alternative_evidence.contracts import (
        MATTER_FAMILY_CORPORATE_EVENT,
        MATTER_FAMILY_FINANCING,
        MATTER_FAMILY_LITIGATION,
        MATTER_SELECTION_INTEGRATED,
        MatterSelectionPolicy,
    )
    from alphalattice.evidence.alternative_evidence.runtime.policy import AdmittedEvidencePolicy

    workspace, report = book
    integrated = MatterSelectionPolicy(
        method=MATTER_SELECTION_INTEGRATED,
        families=(MATTER_FAMILY_LITIGATION, MATTER_FAMILY_CORPORATE_EVENT, MATTER_FAMILY_FINANCING),
    )
    authority = replace(
        coverage_authority(workspace, report),
        evidence_policy=AdmittedEvidencePolicy(matter_selection=integrated),
    )
    service = start_coverage_service(workspace, authority, tmp_path)
    try:
        preview = _run_one(service, {"operation": "EVIDENCE_PREVIEW"})
        prepared = _run_one(service, preview["next_requests"]["prepare"])
        assert prepared["disposition"] == "ADMITTED", prepared
        service.drain()
        task_id = prepared["task_id"]
        adapter = service.review.evidence_task_adapter
        states = adapter.unit_states(service.registry.task(UUID(task_id)))
        unchanged = (len(service.registry.tasks()), service.review.artifacts.write_count)
        ledger = _run_one(service, {"operation": "EVIDENCE_LEDGER"})
        assert ledger["status"] == "EVIDENCE_BOOK_LEDGER"
        assert (ledger["total_groups"], ledger["page_count"], ledger["groups_per_page"]) == (
            UNIT_COUNT,
            1,
            20,
        )
        assert ledger["next_request"] is None
        groups = ledger["groups"]
        # In the section's own order (the book's priority), so the Reading
        # map and the progress list the same groups in the same places.
        progress = service.evidence_cro()["coverage_progress"]
        assert [group["group_id"] for group in groups] == [
            unit["unit_id"] for unit in progress["units"]
        ]
        assert sorted(group["group_id"] for group in groups) == sorted(states)
        for group in groups:
            unit = states[group["group_id"]]
            assert group["ordered_entity_ids"] == list(unit["ordered_entity_ids"])
            assert (group["state"], group["packet_task_id"], group["packet_unit_id"]) == (
                "PREPARED",
                task_id,
                group["group_id"],
            )
            assert {(cell["entity_id"], cell["topic"]) for cell in group["cells"]} == {
                (entity, str(topic)) for entity in group["ordered_entity_ids"] for topic in TOPICS
            }
            assert sum(group["cells_by_state"].values()) == len(group["cells"])
        # A group's cells are its unit packet's own coverage cells.
        last = groups[-1]
        detail = _run_one(
            service,
            {
                "operation": "EVIDENCE_PACKET",
                "task_id": task_id,
                "evidence_unit_id": last["group_id"],
                "evidence_detail": "topic_coverage",
            },
        )
        # R20: the ledger whole, as the packet read's coverage shows it -- the cells and
        # the topics -- with the continuation scope whole and what else the map read per packet.
        assert last["cells"] == detail["evidence_view"]["coverage"]["cells"]
        assert last["topics"] == detail["evidence_view"]["coverage"]["topics"]
        assert last["continuation"] == detail["continuation_scope"]
        assert {
            key: last[key] for key in ("coverage_unit", "delivery", "continuation_request")
        } == {key: detail[key] for key in ("coverage_unit", "delivery", "continuation_request")}
        assert (len(service.registry.tasks()), service.review.artifacts.write_count) == unchanged
        with pytest.raises(ValueError, match="ledger_page_out_of_range:2 of 1"):
            _run_one(service, {"operation": "EVIDENCE_LEDGER", "ledger_page": 2})
    finally:
        service.session.stop()


def test_the_book_ledger_refuses_one_unreadable_receipt_and_keeps_healthy_groups(
    book: Any, tmp_path: Path
) -> None:
    """An unreadable unit receipt refuses in place while other unit ledgers remain readable."""
    workspace, report = book
    service = start_coverage_service(workspace, coverage_authority(workspace, report), tmp_path)
    try:
        preview = _run_one(service, {"operation": "EVIDENCE_PREVIEW"})
        prepared = _run_one(service, preview["next_requests"]["prepare"])
        assert prepared["disposition"] == "ADMITTED", prepared
        service.drain()
        task_id = prepared["task_id"]
        adapter = service.review.evidence_task_adapter
        states = adapter.unit_states(service.registry.task(UUID(task_id)))
        unit_id = next(iter(states))
        receipt_hash, _span_set_hash = adapter.prepared_receipt_identity(
            UUID(task_id), unit_id=unit_id
        )
        receipt_path = (
            adapter.runtime.artifacts.root / "retrieval-access-receipts" / f"{receipt_hash}.json"
        )
    finally:
        service.session.stop()

    receipt_path.write_text("{", encoding="utf-8")
    reopened = coverage_authority(workspace, report)
    service = start_coverage_service(workspace, reopened, tmp_path)
    try:
        ledger = _run_one(service, {"operation": "EVIDENCE_LEDGER"})
        assert ledger["status"] == "EVIDENCE_BOOK_LEDGER"
        assert len(ledger["groups"]) == UNIT_COUNT
        refused = next(group for group in ledger["groups"] if group["group_id"] == unit_id)
        assert refused["status"] == "REFUSED"
        assert refused["refusal"] == refused["failure_code"]
        assert refused["failure_code"].startswith("alternative_evidence.artifact_tampered")
        assert refused["packet_task_id"] == task_id
        assert "cells" not in refused
        requests = refused["next_requests"]
        assert requests["ledger"] == {
            "operation": "EVIDENCE_LEDGER",
            "result_hash": service.result_hash(),
            "ledger_page": 1,
        }
        assert requests["task"] == {"operation": "TASK_RECOVERY", "task_id": task_id}
        assert requests["storage"] == {"operation": "STORAGE_READBACK"}
        assert requests["workspace"] == {"operation": "WORKSPACE_SHOW"}
        assert requests["backups"] == {"operation": "WORKSPACE_BACKUPS"}
        healthy = [group for group in ledger["groups"] if group["group_id"] != unit_id]
        assert len(healthy) == UNIT_COUNT - 1
        assert all(group["cells"] for group in healthy)
    finally:
        service.session.stop()
