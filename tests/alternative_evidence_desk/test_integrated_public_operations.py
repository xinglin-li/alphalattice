"""The integrated selection through the public operations: the preview
discloses the routing under the one allowance; PREPARE seals a receipt
whose routing ledger reaches the packet's coverage detail; the read-only
time view resolves the last 30 days ending at the evidence cutoff on the
acceptance basis, keeps an old filing's exact repeat as labelled
historical context, counts what falls outside, and adds no model work;
the finding package carries the bundle's topics, method and comparison
beside the actor's claim; a stale detail name and a filter on the wrong
detail refuse by name. The first-release reads (the preview's bounds,
campaign and reuse; the book ledger; the index readers; the dossier's
sources and checks; the citation and span pages) are projections of the
same sealed work."""

from __future__ import annotations

import json
import urllib.parse
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

from alphalattice.evidence.alternative_evidence.analysis.matters import (
    TOPIC_LANES_ALLOCATION_ID,
)
from alphalattice.evidence.alternative_evidence.analysis.routing import ROUTING_RULES_ID, TOPICS
from alphalattice.evidence.alternative_evidence.contracts import (
    MATTER_FAMILY_CORPORATE_EVENT,
    MATTER_FAMILY_FINANCING,
    MATTER_FAMILY_LITIGATION,
    MATTER_SELECTION_INTEGRATED,
    MatterSelectionPolicy,
)
from alphalattice.evidence.alternative_evidence.runtime.policy import AdmittedEvidencePolicy
from alphalattice.evidence.alternative_evidence.sources.recorded import RecordedEvidenceDocument
from tests.alternative_evidence_desk.matter_selection_support import (
    _filings_with_debt,
    _repeating_filings,
)
from tests.alternative_evidence_desk.planted_corpus import _NOW, _CitingActor
from tests.alternative_evidence_desk.review_dossiers import _issue, _submission, answer_body
from tests.alternative_evidence_desk.review_http_support import (
    _operation,
    _parts,
    _receipt,
    build_authority,
    build_workspace,
    delivered_aliases,
    start_service,
)

INTEGRATED = MatterSelectionPolicy(
    method=MATTER_SELECTION_INTEGRATED,
    families=(MATTER_FAMILY_LITIGATION, MATTER_FAMILY_CORPORATE_EVENT, MATTER_FAMILY_FINANCING),
)


ONE_UNIT = "u01"
"""The book's only unit: every book is prepared as a coverage run."""


def _book(entities: tuple[str, ...]) -> tuple[RecordedEvidenceDocument, ...]:
    """The first issuer's filings with the debt note; the second issuer's two
    10-Qs restating the same matters a quarter apart."""

    second = entities[1] if len(entities) > 1 else entities[0]
    repeats = tuple(
        document.model_copy(update={"entity_id": second}) for document in _repeating_filings()
    )
    return (*_filings_with_debt(entities), *repeats)


def _authority(tmp_path: Path, report: Any) -> Any:
    authority = build_authority(
        tmp_path=tmp_path, report=report, extra_documents=_book, model_authority_admitted=False
    )
    return authority.__class__(
        **{
            **{f: getattr(authority, f) for f in authority.__dataclass_fields__},
            "evidence_policy": AdmittedEvidencePolicy(
                admit_model_review=False, matter_selection=INTEGRATED
            ),
        }
    )


def test_the_integrated_selection_flows_through_the_public_operations(tmp_path: Path) -> None:
    """requirement (G, §3.4): preview -> captured PREPARE -> coverage ledger
    -> time view -> finding package, every projection a view of one sealed
    evidence set, filtering adding no acquisition or model work."""

    workspace, report = build_workspace(tmp_path)
    service = start_service(workspace, _authority(tmp_path, report), tmp_path)
    try:
        selected = {"result_hash": service.result_hash()}
        query = "&".join(f"{k}={v}" for k, v in selected.items())
        preview = service.get("/api/evidence/preview?" + query)
        disclosed = preview["matter_selection"]
        assert disclosed["method"] == MATTER_SELECTION_INTEGRATED
        assert disclosed["allocation_rules_id"] == TOPIC_LANES_ALLOCATION_ID
        assert disclosed["routing"]["rules_id"] == ROUTING_RULES_ID
        assert disclosed["routing"]["topics"] == [str(t) for t in TOPICS]
        assert disclosed["per_session_allowance"]["windows"] == 64, "no allowance is added"
        prepare_request = preview["next_requests"]["prepare"]
        prepared = service.post(
            "/api/evidence/prepare", {k: v for k, v in prepare_request.items() if k != "operation"}
        )
        service.drain()
        assert prepared["disposition"] == "ADMITTED", prepared
        task_id = prepared["task_id"]
        adapter = service.review.evidence_task_adapter
        (unit,) = adapter.run_of(service.registry.task(UUID(task_id))).units
        recorded = unit.request
        assert recorded.matter_selection == INTEGRATED
        receipt_hash, _ = adapter.prepared_receipt_identity(UUID(task_id), unit_id=ONE_UNIT)
        receipt = _receipt(service, receipt_hash)
        assert receipt.routing is not None and receipt.routing.rules_id == ROUTING_RULES_ID
        entities = tuple(recorded.ordered_entity_ids)
        # The fixture book is built over the listing's sorted tickers: the
        # first holds the filings with the debt note, the second the two 10-Qs.
        listed = tuple(sorted(entities))
        packet_request = {
            "operation": "EVIDENCE_PACKET",
            **selected,
            "task_id": task_id,
            "evidence_unit_id": ONE_UNIT,
        }
        parts = _parts(service, packet_request)
        payload = json.loads(parts[0]["packet"].split("\n\n", 2)[1])
        assert payload["litigation_matters"]["allocation_rules_id"] == TOPIC_LANES_ALLOCATION_ID
        # The coverage ledger: every cell of the book with its states.
        coverage = _operation(service, {**packet_request, "evidence_detail": "topic_coverage"})
        assert coverage["status"] == "EVIDENCE_TOPIC_COVERAGE"
        view = coverage["evidence_view"]
        ledger = view["coverage"]
        assert ledger["rules_id"] == ROUTING_RULES_ID
        assert {(c["entity_id"], c["topic"]) for c in ledger["cells"]} == {
            (entity, str(topic)) for entity in entities for topic in TOPICS
        }
        assert all(
            c["delivery_state"] in {"COMPLETE", "PARTIAL", "PENDING", "NO_ROUTE", "NO_SOURCE"}
            for c in ledger["cells"]
        )
        # The other issuers hold only their planted release: what they filed
        # recently. A periodic report outside the window is no gap (W1), so no
        # cell names one; a gap named beside a cell is still never a COMPLETE
        # cell (the first-release safety closeout).
        for entity in listed[2:]:
            cells = [c for c in ledger["cells"] if c["entity_id"] == entity]
            assert len(cells) == len(TOPICS)
            assert not any(g.startswith("SOURCE_GAP") for c in cells for g in c["gaps"])
            assert all(c["delivery_state"] != "COMPLETE" for c in cells if c["gaps"])
        assert all(c["delivery_state"] != "COMPLETE" or not c["gaps"] for c in ledger["cells"]), (
            "COMPLETE and a gap never share a cell"
        )
        # The first issuer's routed regions hold tables the canonical text did
        # not carry and whose original is not retained: those cells are
        # delivered in part and say so, whatever else was read.
        without_original = [c for c in ledger["cells"] if c["tables_without_original"]]
        assert without_original
        assert all(
            c["delivery_state"] == "PARTIAL" and "TABLES_WITHOUT_ORIGINAL" in c["incomplete"]
            for c in without_original
        )
        assert all(
            c["tables"] == c["tables_delivered"] + c["tables_unrenderable"] + c["tables_pending"]
            for c in ledger["cells"]
        ), "every routed table is delivered, refused by name, or pending -- never unaccounted"
        assert [t["topic"] for t in ledger["topics"]] == [str(t) for t in TOPICS]
        assert ledger["residual_search"]["questions_run"] == receipt.search_call_count
        assert ledger["exact_repeats_collapsed"] == 2
        bundles = view["bundles"]
        assert len(bundles) == len(receipt.delivered_span_handles)
        assert {b["span_handle"] for b in bundles} == set(receipt.delivered_span_handles)
        assert coverage["delivery"]["model_work"] == "NONE"
        methods = {b["method"] for b in bundles}
        assert {"UNIT_WINDOW", "RESIDUAL_SEARCH"} <= methods
        unit_bundles = [b for b in bundles if b["method"] == "UNIT_WINDOW"]
        assert all(b["topics"] for b in unit_bundles), "every window serves a topic"
        assert all(b["time"]["published_at"] for b in bundles)
        assert all(b["comparison"] is not None for b in unit_bundles)
        # The time view: the last 30 days ending at the evidence cutoff; the
        # second issuer's later 10-Q (published 10 days before) is in, its
        # earlier one (100 days before) is out, and the earlier's exact
        # repeats accompany the later windows as historical context.
        second = listed[1]
        timed = _operation(
            service,
            {
                **packet_request,
                "evidence_detail": "time_view",
                "view_entity_id": second,
                "view_topic": "LEGAL_REGULATORY",
            },
        )
        assert timed["status"] == "EVIDENCE_TIME_VIEW"
        tv = timed["evidence_view"]
        assert tv["interval"]["basis"] == "LAST_30_DAYS_ENDING_AT_CUTOFF"
        assert tv["interval"]["to"] == tv["interval"]["evidence_cutoff"]
        assert tv["interval"]["time_basis"] == "ACCEPTANCE" and tv["interval"]["timezone"] == "UTC"
        assert tv["filters"] == {"entity_id": second, "topic": "LEGAL_REGULATORY"}
        inside = tv["bundles"]
        assert inside and all(b["entity_id"] == second for b in inside)
        assert all("LEGAL_REGULATORY" in b["topics"] for b in inside)
        assert all(b["placement"] == "IN" for b in inside)
        assert all(b["time_basis"] == "PUBLICATION_INSTANT" for b in inside), (
            "a recorded text states no acceptance: publication at its precision places it"
        )
        assert tv["outside_interval"] >= 1, "the 100-day-old filing's windows fall outside"
        assert tv["unknown_time"] == [] and tv["boundary"] == []
        context = tv["historical_context"]
        assert context, "the earlier filing's restated units accompany the recent ones"
        assert all(c["placement"] == "HISTORICAL_CONTEXT" for c in context)
        assert all(c["relation"] == "EXACT_REPEAT" for c in context)
        assert all(c["context_for"] in {b["span_handle"] for b in inside} for c in context)
        assert tv["coverage"]["cells"] == ledger["cells"], "the unfiltered ledger travels"
        assert timed["delivery"]["model_work"] == "NONE"
        assert adapter.runtime.selection_reuse_count == 0
        # An explicit interval whose end passes the cutoff is clamped to it;
        # a filter on the coverage detail refuses by name; an unknown detail
        # refuses by name.
        clamped = _operation(
            service,
            {
                **packet_request,
                "evidence_detail": "time_view",
                "view_from": "2020-01-01T00:00:00+00:00",
                "view_to": "2999-01-01T00:00:00+00:00",
            },
        )
        assert clamped["evidence_view"]["interval"]["to"] == tv["interval"]["evidence_cutoff"]
        assert clamped["evidence_view"]["interval"]["basis"] == "EXPLICIT_INTERVAL"
        assert clamped["evidence_view"]["outside_interval"] == 0
        refused = service.request(
            "/api/evidence/packet?"
            + "&".join(
                f"{k}={v}"
                for k, v in {
                    **selected,
                    "task_id": task_id,
                    "evidence_unit_id": ONE_UNIT,
                    "evidence_detail": "topic_coverage",
                    "view_last_days": 7,
                }.items()
            )
        )
        assert refused[0] != 200 and "view_filters_not_applicable" in json.dumps(refused[1])
        unknown = service.request(
            "/api/evidence/packet?"
            + "&".join(
                f"{k}={v}"
                for k, v in {
                    **selected,
                    "task_id": task_id,
                    "evidence_unit_id": ONE_UNIT,
                    "evidence_detail": "ledger",
                }.items()
            )
        )
        assert unknown[0] != 200
        # A brief citing a unit window: the finding package carries the
        # bundle's topics, method and comparison beside the claim.
        cited = next(
            b
            for b in inside
            if b["method"] == "UNIT_WINDOW" and b["comparison"]["state"] == "EXACT_REPEAT"
        )
        exported = _operation(service, packet_request)
        alias_of = delivered_aliases(_parts(service, packet_request))
        brief = {
            "findings": [
                {
                    "issuer": second,
                    "topic": "LEGAL_REGULATORY",
                    "lifecycle": "ONGOING",
                    "direction": "ADVERSE",
                    "summary": f"{second}: the filing names the proceeding in the cited window.",
                    "cite": [alias_of[cited["span_handle"]]],
                }
            ],
            "notes": "QA-controlled answer over one delivered unit window.",
        }
        document = {**exported["submission_template"], "analysis_answer": brief}
        submitted = service.post(
            "/api/evidence/analysis", {k: v for k, v in document.items() if k != "operation"}
        )
        service.drain()
        assert submitted["disposition"] == "ADMITTED", submitted
        status_code, dossier = service.request("/api/cro/dossier?" + query)
        assert status_code == 200, dossier
        handle = dossier["dossier"]["findings"][0]["finding_handle"]
        # The machine ledger reaches the dossier the review stands on: the
        # cells' unread needs and gaps travel as compact lines beside the
        # document gaps, whatever the actor reported.
        missing = dossier["dossier"]["coverage"]["missing_evidence"]
        cell_lines = [line for line in missing if any(line.startswith(f"{e} ") for e in entities)]
        assert cell_lines, missing
        assert any("unit need(s) unread" in line for line in cell_lines)
        # No periodic report is a baseline since W1: none is named missing.
        assert not any("periodic filing" in line for line in cell_lines)
        package = service.get(f"/api/cro/finding?{query}&finding_handle={handle}")
        assert package["status"] == "FINDING_EVIDENCE_PACKAGE"
        (support,) = package["support"]
        assert support["span_handle"] == cited["span_handle"] and support["source_verified"]
        assert support["method"] == "UNIT_WINDOW"
        assert "LEGAL_REGULATORY" in support["topics"]
        assert support["comparison"]["state"] == "EXACT_REPEAT"
        assert support["comparison"]["against"] == cited["comparison"]["against"]
        assert support["time"]["published_at"] == cited["time"]["published_at"]
    finally:
        service.session.stop()


def test_the_first_release_reads_are_projections_of_the_sealed_evidence(tmp_path: Path) -> None:
    """The first release reads are projections of the sealed evidence."""

    from alphalattice.control.product_host.composition.local_web_session import (
        _campaign_reader,
    )
    from alphalattice.evidence.alternative_evidence.sources.campaign import (
        SecCampaignDeclaration,
        SecCampaignLedger,
    )

    workspace, report = build_workspace(tmp_path)
    now = [_NOW]
    service = start_service(workspace, _authority(tmp_path, report), tmp_path, clock=lambda: now[0])
    try:
        selected = {"result_hash": service.result_hash()}
        query = urllib.parse.urlencode(selected)
        # A2: the bounds every preparation runs under, stated before any
        # request; no campaign where none was named; nothing reused yet.
        preview = service.get("/api/evidence/preview?" + query)
        policy = service.review.evidence_policy
        assert preview["admission"] == {
            "maximum_document_bytes": policy.source_policy.maximum_document_bytes,
            "acquisition_window_seconds": policy.acquisition_window_seconds,
        }
        assert preview["campaign"] is None and preview["reuse"] is None
        # The campaign balance is read from its ledger at each read, never
        # the admission's snapshot: a reservation between two reads shows.
        declaration = SecCampaignDeclaration(
            campaign_id="qa-campaign",
            maximum_total_attempts=4,
            maximum_total_response_bytes=1_500,
            maximum_body_resources=2,
            maximum_document_bytes=600,
        )
        with SecCampaignLedger.open(tmp_path / "campaign.jsonl", declaration=declaration) as ledger:
            service.review.campaign_summary = _campaign_reader(
                cast(Any, SimpleNamespace(campaign_ledger=ledger))
            )
            assert service.get("/api/evidence/preview?" + query)["campaign"] == ledger.summary()
            sequence = ledger.reserve(
                url="https://www.sec.gov/qa", resource="qa-1", maximum_bytes=500
            )
            held = service.get("/api/evidence/preview?" + query)["campaign"]
            assert (held["unsettled_reservations"], held["bytes_remaining"]) == (1, 1_000)
            ledger.settle(sequence, actual_bytes=400, outcome="COMPLETE")
            settled = service.get("/api/evidence/preview?" + query)["campaign"]
            assert (settled["unsettled_reservations"], settled["bytes_consumed"]) == (0, 400)
        service.review.campaign_summary = None
        prepare_request = preview["next_requests"]["prepare"]
        prepared = service.post(
            "/api/evidence/prepare", {k: v for k, v in prepare_request.items() if k != "operation"}
        )
        service.drain()
        assert prepared["disposition"] == "ADMITTED", prepared
        task_id = prepared["task_id"]
        adapter = service.review.evidence_task_adapter
        receipt_hash, _ = adapter.prepared_receipt_identity(UUID(task_id), unit_id=ONE_UNIT)
        # A7: the named preparation made its own selection; a recorded
        # preparation checks no official source, so it counts no documents.
        reuse = service.get("/api/evidence/preview?" + query)["reuse"]
        assert reuse["preparations"] == [
            {
                "task_id": task_id,
                "unit_id": ONE_UNIT,
                "selection": "SELECTED",
                "reused_from_receipt_hash": None,
            }
        ]
        assert (reuse["selections_made"], reuse["selections_reused"]) == (1, 0)
        assert reuse["documents_reused"] is None and reuse["documents_fetched"] is None
        assert reuse["service_counters"] == adapter.runtime.reuse_accounting()
        unchanged = (len(service.registry.tasks()), service.review.artifacts.write_count)
        # A11: the book's ledger is each group's packet coverage cells.
        book_ledger = service.get("/api/evidence-cro/ledger?" + query)
        assert book_ledger["status"] == "EVIDENCE_BOOK_LEDGER"
        assert (book_ledger["page"], book_ledger["page_count"], book_ledger["total_groups"]) == (
            1,
            1,
            1,
        )
        assert book_ledger["next_request"] is None
        (group,) = book_ledger["groups"]
        assert (group["state"], group["packet_task_id"], group["packet_unit_id"]) == (
            "PREPARED",
            task_id,
            ONE_UNIT,
        )
        packet_request = {
            "operation": "EVIDENCE_PACKET",
            **selected,
            "task_id": task_id,
            "evidence_unit_id": ONE_UNIT,
        }
        coverage = _operation(service, {**packet_request, "evidence_detail": "topic_coverage"})
        cells = coverage["evidence_view"]["coverage"]["cells"]
        assert group["rules_id"] == ROUTING_RULES_ID
        # Each group carries its packet's cells whole (R20).
        assert group["cells"] == cells
        assert sum(group["cells_by_state"].values()) == len(cells)
        # And its continuation scope whole (R20), these among its fields.
        assert {"state", "pending_windows", "pending_candidates", "remainders"} <= set(
            group["continuation"]
        )
        for page, code in ((2, "ledger_page_out_of_range:2 of 1"), (0, "ledger_page_invalid")):
            status, refused = service.request(
                f"/api/evidence-cro/ledger?{query}&ledger_page={page}"
            )
            assert status != 200 and code in json.dumps(refused), refused
        assert (len(service.registry.tasks()), service.review.artifacts.write_count) == unchanged
        # A3/A4, restated by the answer format: the answer reports no check;
        # each issuer's state beside the dossier is a delivery fact.
        exported = _operation(service, packet_request)
        packet = adapter.prepared_packet(
            UUID(task_id), now=service.review.clock(), unit_id=ONE_UNIT
        )
        brief = _CitingActor()(packet=packet).answer.model_dump(mode="json")
        answered = {finding["issuer"] for finding in brief["findings"]}
        document = {**exported["submission_template"], "analysis_answer": brief}
        submitted = service.post(
            "/api/evidence/analysis", {k: v for k, v in document.items() if k != "operation"}
        )
        service.drain()
        assert submitted["disposition"] == "ADMITTED", submitted
        publication = adapter.published_analysis(
            UUID(submitted["task_id"]), now=service.review.clock()
        ).publication
        status, dossier = service.request("/api/cro/dossier?" + query)
        assert status == 200, dossier
        (source,) = dossier["evidence_sources"]
        required = source.pop("required_checks")
        assert required and source == {
            "analysis_publication_hash": publication.publication_hash,
            "unit_id": None,
            "access_receipt_hash": receipt_hash,
            "prepared_task_id": task_id,
            "prepared_unit_id": ONE_UNIT,
            "completion_schema": "issuer-delivery-facts-v3",
        }
        checks = dossier["issuer_checks"]
        assert answered and answered <= set(checks), (answered, sorted(checks))
        for entity in answered:
            mine = checks[entity]
            assert mine["state"] == "EXECUTED_WITH_FINDINGS"
            assert mine["executed"] == mine["deferred"] == mine["unreported"] == []
            assert mine["analysis_publication_hash"] == publication.publication_hash
        assert not {"evidence_sources", "issuer_checks"} & set(dossier["dossier"]), (
            "beside the sealed dossier, never inside it"
        )
        finding = dossier["dossier"]["findings"][0]
        submission = _submission(
            _issue(finding["finding_handle"], finding["affected_entities"][0])
        ).model_dump(mode="json")
        assessment = {
            **dossier["submission_template"],
            "review_answer": answer_body(dossier["dossier"], submission),
        }
        service.post(
            "/api/cro/assessment", {k: v for k, v in assessment.items() if k != "operation"}
        )
        service.drain()
        section = service.evidence_cro(selected["result_hash"])
        assert section["state"] == "REVIEW_PUBLISHED", section["state"]
        # A10: the section's citations, a page for one issuer or one group.
        unchanged = (len(service.registry.tasks()), service.review.artifacts.write_count)
        citations = section["citations"]
        assert citations

        def cro_page(**fields: object) -> tuple[int, dict[str, Any]]:
            return service.request(
                "/api/evidence-cro?" + urllib.parse.urlencode({**selected, **fields})
            )

        status, first = cro_page(citation_page=1)
        assert status == 200, first
        assert first["status"] == "EVIDENCE_CRO_CITATIONS_PAGE"
        assert first["review_publication_hash"] == section["review_publication_hash"]
        assert first["citations"] == citations[:20]
        assert (first["total"], first["total_unfiltered"], first["per_page"]) == (
            len(citations),
            len(citations),
            20,
        )
        issuer = citations[-1]["entity_id"]
        status, theirs = cro_page(citation_entity_id=issuer)
        assert theirs["citations"] == [c for c in citations if c["entity_id"] == issuer][:20]
        assert theirs["filter"] == {"entity_id": issuer, "unit_id": None}
        status, group_page = cro_page(citation_unit_id="u01")
        assert group_page["total"] == len(citations), "a book of one unit: the group is the book"
        for fields, code in (
            ({"citation_page": first["page_count"] + 1}, "citation_page_out_of_range"),
            (
                {"citation_entity_id": issuer, "citation_unit_id": "u01"},
                "citation_filter_ambiguous",
            ),
            ({"citation_page": 0}, "citation_page_invalid"),
        ):
            status, refused = cro_page(**fields)
            assert status != 200 and code in json.dumps(refused), (fields, refused)
        assert (len(service.registry.tasks()), service.review.artifacts.write_count) == unchanged
        # regression: one operation verifies each Evidence record once; the
        # section read replayed each analysis from several places (68 records 388 times).
        store = service.review.artifacts
        asked: set[tuple[str, ...]] = set()
        originals = {name: getattr(store, name) for name in ("load", "load_retrieval_generation")}

        def asking(name: str) -> Any:
            def ask(*args: Any) -> Any:
                asked.add((name, *map(str, args[:2])))
                return originals[name](*args)

            return ask

        for name in originals:
            setattr(store, name, asking(name))
        try:
            reads = store.read_count
            service.evidence_cro(selected["result_hash"])
            assert asked and store.read_count - reads == len(asked)
        finally:
            for name in originals:
                delattr(store, name)
        # The export's verified spans: the same export, a page of it.
        export_request = section["next_requests"]["export"]
        whole = _operation(service, export_request)
        spans = whole["evidence"]["verified_spans"]
        # regression: a page slices the export the whole export sealed, reading nothing.
        reads = service.review.artifacts.read_count
        spans_page = _operation(service, {**export_request, "citation_entity_id": issuer})
        assert service.review.artifacts.read_count == reads
        assert spans_page["verification_basis"].startswith("SEALED_EXPORT ")
        assert spans_page["status"] == "EVIDENCE_CRO_EXPORT_SPANS_PAGE"
        assert spans_page["export_hash"] == whole["export_hash"]
        assert spans_page["review_publication_hash"] == section["review_publication_hash"]
        assert spans_page["verified_spans"] == [s for s in spans if s["entity_id"] == issuer][:20]
        assert spans_page["total_unfiltered"] == len(spans)
        assert service.evidence_cro(selected["result_hash"])["state"] == "REVIEW_PUBLISHED"
        # A7 again: a later cutoff over the same corpus reuses the selection
        # whole, and the preview says from which receipt.
        now[0] = now[0] + timedelta(hours=1)
        again = service.post("/api/evidence/prepare", selected)
        service.drain()
        assert again["disposition"] == "ADMITTED" and again["task_id"] != task_id, again
        later = service.get("/api/evidence/preview?" + query)["reuse"]
        assert later["preparations"] == [
            {
                "task_id": again["task_id"],
                "unit_id": ONE_UNIT,
                "selection": "REUSED",
                "reused_from_receipt_hash": receipt_hash,
            }
        ]
        assert (later["selections_made"], later["selections_reused"]) == (0, 1)
        # The production plan is retired (first-release integration T5): a
        # workspace switched back to it prepares nothing, refused by name, and
        # no Task is admitted; nothing runs under the integrated selection in
        # its place.
        service.review.evidence_policy = AdmittedEvidencePolicy(
            admit_model_review=False, matter_selection=None
        )
        now[0] = now[0] + timedelta(hours=1)
        count = len(service.registry.tasks())
        plain = service.post("/api/evidence/prepare", selected)
        assert plain["disposition"] == "REFUSED_MATTER_SELECTION_RETIRED", plain
        assert plain["failure_code"] == "alternative_evidence.matter_selection_policy_retired"
        assert len(service.registry.tasks()) == count
    finally:
        service.session.stop()


def test_a_published_reading_filters_the_exact_review_lineage_without_rewriting_export(
    tmp_path: Path,
) -> None:
    """A published reading filters the exact review lineage without rewriting export."""

    from alphalattice.evidence.alternative_evidence.analysis.read_model import time_view

    workspace, report = build_workspace(tmp_path)
    now = [_NOW]
    service = start_service(workspace, _authority(tmp_path, report), tmp_path, clock=lambda: now[0])
    try:
        selected = {"result_hash": service.result_hash()}
        query = urllib.parse.urlencode(selected)
        prepare = service.get("/api/evidence/preview?" + query)["next_requests"]["prepare"]
        prepared = service.post(
            "/api/evidence/prepare", {k: v for k, v in prepare.items() if k != "operation"}
        )
        service.drain()
        assert prepared["disposition"] == "ADMITTED", prepared
        task_id = prepared["task_id"]
        packet_request = {
            "operation": "EVIDENCE_PACKET",
            **selected,
            "task_id": task_id,
            "evidence_unit_id": ONE_UNIT,
        }
        adapter = service.review.evidence_task_adapter
        packet = adapter.prepared_packet(UUID(task_id), now=now[0], unit_id=ONE_UNIT)
        exported_packet = _operation(service, packet_request)
        submitted = service.post(
            "/api/evidence/analysis",
            {
                **{
                    k: v
                    for k, v in exported_packet["submission_template"].items()
                    if k != "operation"
                },
                "analysis_answer": _CitingActor()(packet=packet).answer.model_dump(mode="json"),
            },
        )
        service.drain()
        assert submitted["disposition"] == "ADMITTED", submitted
        analysis_hash = adapter.published_analysis(
            UUID(submitted["task_id"]), now=now[0]
        ).publication.publication_hash
        dossier = service.get("/api/cro/dossier?" + query)
        finding = dossier["dossier"]["findings"][0]
        answer = _submission(
            _issue(finding["finding_handle"], finding["affected_entities"][0])
        ).model_dump(mode="json")
        assessed = service.post(
            "/api/cro/assessment",
            {
                **{k: v for k, v in dossier["submission_template"].items() if k != "operation"},
                "review_answer": answer_body(dossier["dossier"], answer),
            },
        )
        service.drain()
        assert assessed["disposition"] == "ADMITTED", assessed
        published = service.evidence_cro(selected["result_hash"])
        assert published["state"] == "REVIEW_PUBLISHED", published
        review_hash = published["review_publication_hash"]
        export_request = published["next_requests"]["export"]
        export_before = _operation(service, export_request)

        def reading(**filters: object) -> dict[str, Any]:
            body = service.get(
                "/api/evidence-cro?"
                + urllib.parse.urlencode({**selected, "evidence_detail": "time_view", **filters})
            )
            assert body["state"] == "REVIEW_PUBLISHED", body
            value = body["published_reading"]
            assert value["review_publication_hash"] == review_hash
            return value

        all_reading = reading()
        assert len(all_reading["bundles"]) == len(export_before["evidence"]["verified_spans"])
        assert all_reading["delivered_total"] == len(all_reading["bundles"])
        assert all(b["placement"] == "DELIVERED" for b in all_reading["bundles"])
        (analysis,) = all_reading["analyses"]
        assert analysis == {
            "analysis_publication_hash": analysis_hash,
            "unit_id": None,
            "interval": None,
            "evidence_cutoff": _NOW.isoformat(),
        }
        second = sorted(packet.request.ordered_entity_ids)[1]
        topic = "LEGAL_REGULATORY"
        filters = {"view_entity_id": second, "view_topic": topic}
        cell = reading(**filters)
        assert cell["filters"] == {"entity_id": second, "topic": topic, "last_days": None}
        assert cell["bundles"] == [
            b for b in all_reading["bundles"] if b["entity_id"] == second and topic in b["topics"]
        ]
        assert cell["bundles"] and len(cell["bundles"]) < len(all_reading["bundles"])
        assert reading(view_entity_id="NOT-IN-THIS-BOOK")["bundles"] == []

        timed = reading(**filters, view_last_days=30)
        # The published list is the immutable export's verified citations, not
        # every prepared passage: this fixture's actor cites the first eight per
        # issuer. Read their original receipt's time rule in the export's order.
        by_handle = {span.span_handle: span for span in packet.spans}
        published_packet = replace(
            packet,
            spans=tuple(
                by_handle[span["span_handle"]]
                for span in export_before["evidence"]["verified_spans"]
            ),
        )
        original = json.loads(
            json.dumps(time_view(published_packet, entity_id=second, topic=topic, last_days=30))
        )
        for name in ("bundles", "boundary", "unknown_time", "historical_context"):
            expected = [
                {
                    **bundle,
                    "analysis_publication_hash": analysis_hash,
                }
                for bundle in original[name]
            ]
            assert timed[name] == expected, name
        assert timed["outside_interval"] == original["outside_interval"]
        # All this fixture's cited sources were published at least two days
        # before the cutoff. A one-day view must move them out, not ignore days.
        recent = reading(**filters, view_last_days=1)
        assert recent["bundles"] == []
        assert recent["outside_interval"] == len(cell["bundles"]) > 0
        assert timed["analyses"][0]["interval"] == original["interval"]
        assert _operation(service, export_request) == export_before

        # The newest prepared receipt has a different cutoff; the pinned published
        # read continues from exactly the analysed receipt, not this newer preparation.
        now[0] += timedelta(hours=1)
        newer = service.post("/api/evidence/prepare", selected)
        service.drain()
        assert newer["disposition"] == "ADMITTED" and newer["task_id"] != task_id, newer
        newest = adapter.prepared_packet(UUID(newer["task_id"]), now=now[0], unit_id=ONE_UNIT)
        assert newest.request.evidence_as_of == now[0]
        unchanged = len(service.registry.tasks()), service.review.artifacts.write_count
        pinned = reading(review_publication_hash=review_hash, **filters, view_last_days=30)
        assert pinned == timed
        assert _operation(service, export_request) == export_before
        assert (len(service.registry.tasks()), service.review.artifacts.write_count) == unchanged
    finally:
        service.session.stop()


def _every_issuer_filed(entities: tuple[str, ...]) -> tuple[RecordedEvidenceDocument, ...]:
    """Every issuer holds the same routed periodic filings (the first issuer's
    10-Q, 10-Ks and 8-K restated under its own id), so every issuer is read."""

    base = _filings_with_debt(entities)
    return tuple(
        document.model_copy(update={"entity_id": entity})
        for entity in entities
        for document in base
    )


def test_the_integrated_default_judges_first_and_states_its_gaps(tmp_path: Path) -> None:
    """The integrated default judges first and states its gaps."""

    workspace, report = build_workspace(tmp_path)
    authority = build_authority(
        tmp_path=tmp_path,
        report=report,
        extra_documents=_every_issuer_filed,
        model_authority_admitted=False,
    )
    authority = authority.__class__(
        **{
            **{f: getattr(authority, f) for f in authority.__dataclass_fields__},
            "evidence_policy": AdmittedEvidencePolicy(
                admit_model_review=False, matter_selection=INTEGRATED
            ),
        }
    )
    service = start_service(workspace, authority, tmp_path)
    try:
        selected = {"result_hash": service.result_hash()}
        query = urllib.parse.urlencode(selected)
        prepare = service.get("/api/evidence/preview?" + query)["next_requests"]["prepare"]
        service.post(
            "/api/evidence/prepare", {k: v for k, v in prepare.items() if k != "operation"}
        )
        service.drain()
        adapter = service.review.evidence_task_adapter
        section = service.evidence_cro(selected["result_hash"])
        packets = [v for k, v in section["next_requests"].items() if k.startswith("packet")]
        for request in packets:
            exported = _operation(service, request)
            packet = adapter.prepared_packet(
                UUID(request["task_id"]),
                now=service.review.clock(),
                unit_id=request.get("evidence_unit_id"),
            )
            answer = _CitingActor()(packet=packet).answer.model_dump(mode="json")
            service.post(
                "/api/evidence/analysis",
                {
                    k: v
                    for k, v in {
                        **exported["submission_template"],
                        "analysis_answer": answer,
                    }.items()
                    if k != "operation"
                },
            )
            service.drain()
        status, dossier = service.request("/api/cro/dossier?" + query)
        assert status == 200, dossier
        body = dossier["dossier"]
        assert body["coverage"]["missing_evidence"], "the integrated default records its gaps"
        bands = {value["entity_id"]: value["exposure_band"] for value in body["issuers"]}
        aliases = {handle: alias for alias, handle in dossier["finding_aliases"].items()}
        supported = next(
            value
            for value in body["findings"]
            if value["structure"] == "SUPPORTED"
            and bands[value["affected_entities"][0]] in {"HIGH", "CRITICAL"}
        )
        risk = {
            "findings": [aliases[supported["finding_handle"]]],
            "why": "The cited filings state a matter that bears on this position.",
            "severity": "HIGH",
            "confidence": "SUPPORTED",
            "recommendation": "Reconsider the position before relying on this book.",
        }
        views = []
        for answer in ({"risks": [risk]}, {"risks": []}):
            submitted = service.post(
                "/api/cro/assessment",
                {
                    k: v
                    for k, v in {**dossier["submission_template"], "review_answer": answer}.items()
                    if k != "operation"
                },
            )
            assert submitted["disposition"] == "ADMITTED", submitted
            service.drain()
            key = service.registry.task(UUID(submitted["task_id"])).input.input_hash
            views.append(service.review.review_publications.find_for_review_key(key))
        alert, clear = views
        assert alert.recommendation.route == "MATERIAL_OBJECTION"
        assert alert.receipt.outcome.rule_ids == ("R-OBJECTION",)
        assert any("gap(s) in what could be read" in v for v in alert.recommendation.limitations)
        assert clear.recommendation.route == "NO_MATERIAL_OBJECTION"
        assert clear.recommendation.reasons[0].startswith(
            "No major negative was found in the evidence read."
        )
        assert any("not named as a risk" in v for v in clear.recommendation.limitations)
        for view in views:
            assert not any(
                str(value.action) == "REFRESH_EVIDENCE"
                for value in view.recommendation.required_actions
            )
    finally:
        service.session.stop()
