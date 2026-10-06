"""SEC filings are planned before capacity is spent: what each issuer filed recently.

Since W1 (2026-09-24) the plan is the filings accepted in the 30 days before the
cutoff -- major negatives first, then periodic reports, then the other current
reports, each newest first -- as far as the capacity reaches, every discovered
filing accounted for, and nothing older kept as a baseline: an issuer that filed
nothing in the window has an empty plan. A request sealed under the retired
90-day policy with fixed baselines is refused at planning. Recorded fixture
transport; no network.
"""

from __future__ import annotations

import json
import warnings
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from alphalattice.evidence.alternative_evidence.contracts import (
    ADMITTED_DOCUMENT_CAPACITY,
    AlternativeEvidenceAdmission,
    AlternativeEvidenceClass,
    AlternativeEvidenceMode,
    AlternativeEvidenceReadFiling,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSourcePolicy,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.runtime.coverage import index_quiet
from alphalattice.evidence.alternative_evidence.sources.acquisition import (
    AlternativeEvidenceAcquisitionService,
)
from alphalattice.evidence.alternative_evidence.sources.contracts import (
    SecFilingInventoryRead,
    SecFilingSelectionPlan,
    inventory_read_key,
)
from alphalattice.evidence.alternative_evidence.sources.sec_edgar import (
    SecEdgarSource,
    SecOfficialResponse,
    apply_unit_capacity,
    plan_sec_filing_selection,
)
from tests.alternative_evidence_desk.sec_fixture_transport import NOW, SecFixtureTransport

WIDE = ("AAPL", "MSFT", "A3", "A4", "A5", "A6", "A7", "A8")


def _live_admission(request: Any) -> AlternativeEvidenceAdmission:
    return seal_contract(
        AlternativeEvidenceAdmission,
        "admission_hash",
        request_hash=request.request_hash,
        network_consent=True,
        admit_live_official=True,
        admit_model_review=False,
        admitted_at=NOW,
    )


def _submissions(rows: list[tuple[str, ...]]) -> bytes:
    """Rows of (accession, form, accepted ISO time, report date[, items]) in any order."""

    return json.dumps(
        {
            "filings": {
                "recent": {
                    "accessionNumber": [row[0] for row in rows],
                    "filingDate": [row[2][:10] for row in rows],
                    "acceptanceDateTime": [row[2] for row in rows],
                    "form": [row[1] for row in rows],
                    "primaryDocument": [f"{row[0]}.htm" for row in rows],
                    "reportDate": [row[3] for row in rows],
                    "items": [row[4] if len(row) > 4 else "" for row in rows],
                }
            }
        }
    ).encode()


def _at(days_before_cutoff: int) -> str:
    return (NOW - timedelta(days=days_before_cutoff)).strftime("%Y-%m-%dT%H:%M:%SZ")


FLOOD: list[tuple[str, str, str, str]] = [
    # Old periodic reports, all outside the window: no baseline is kept.
    ("0000320193-24-000010", "10-K", _at(600), "2024-09-30"),
    ("0000320193-25-000010", "10-K", _at(320), "2025-09-30"),
    ("0000320193-25-000011", "10-K/A", _at(300), "2025-09-30"),
    ("0000320193-25-000020", "10-Q", _at(200), "2025-12-31"),
    ("0000320193-26-000021", "10-Q/A", _at(100), "2026-03-31"),
    # Current reports: six inside the 30-day window, three outside, one after the cutoff.
    *[(f"0000320193-26-0001{index:02d}", "8-K", _at(5 * index + 3), "") for index in range(7)],
    ("0000320193-26-000200", "8-K", _at(120), ""),
    ("0000320193-26-000201", "8-K/A", _at(150), ""),
    ("0000320193-26-000300", "8-K", (NOW + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), ""),
]
WINDOWED = [f"0000320193-26-0001{index:02d}" for index in range(6)]


class _FloodTransport(SecFixtureTransport):
    def __init__(self, rows: list[tuple[str, ...]]) -> None:
        super().__init__()
        self.rows = rows

    def get(self, url: str, *, maximum_bytes: int) -> SecOfficialResponse:
        if "/submissions/" in url:
            self.calls.append(url)
            return SecOfficialResponse(
                status_code=200,
                content_type="application/json",
                content=_submissions(self.rows),
                retrieved_at=NOW,
            )
        return super().get(url, maximum_bytes=maximum_bytes)


def _request(
    *,
    issuers: tuple[str, ...] = ("AAPL",),
    budget: int = 12,
    policy: AlternativeEvidenceSourcePolicy | None = None,
    read_filings: tuple[AlternativeEvidenceReadFiling, ...] = (),
) -> Any:
    return seal_contract(
        AlternativeEvidenceRequest,
        "request_hash",
        ordered_entity_ids=issuers,
        evidence_as_of=NOW,
        acquisition_deadline=NOW + timedelta(hours=1),
        evidence_classes=(AlternativeEvidenceClass.SEC_FILING,),
        source_policy=policy
        or AlternativeEvidenceSourcePolicy(maximum_documents_per_issuer=budget),
        ttl_seconds=3600,
        mode=AlternativeEvidenceMode.LIVE_OFFICIAL,
        read_filings=read_filings,
    )


def _plan(
    transport: _FloodTransport,
    request: Any,
    *,
    accession_scope: frozenset[str] | None = None,
    read_earlier: frozenset[str] | None = None,
) -> SecFilingSelectionPlan:
    source = SecEdgarSource(transport)
    registry = source.acquire_registry(captured_at=NOW)
    return source.plan_entity_filings(
        request=request,
        registry=registry,
        entity_id="AAPL",
        accession_scope=accession_scope,
        read_earlier=read_earlier,
    )


def test_the_window_is_what_the_issuer_filed_recently_and_nothing_older_is_kept() -> None:
    plan = _plan(_FloodTransport(FLOOD), _request(budget=4))
    assert plan.discovered_count == 14, "the filing after the cutoff is not discovered"
    assert plan.event_window_days == 30 and plan.required == ()
    assert plan.capacity == 4 and plan.unit_capacity == ADMITTED_DOCUMENT_CAPACITY
    # The four newest of the six filings in the window, newest first.
    assert [value.accession for value in plan.events] == WINDOWED[:4]
    assert [(value.accession, value.reason) for value in plan.deferred] == [
        (accession, "BEYOND_CAPACITY") for accession in WINDOWED[4:]
    ]
    assert any("inside the 30-day window deferred" in value for value in plan.limitations)
    # Every older filing is counted outside the window; none is a baseline.
    assert plan.superseded_baseline_count == 0
    assert plan.outside_window_event_count == 8
    assert plan.selected == plan.events


def test_a_filing_read_earlier_gives_its_capacity_to_what_is_new() -> None:
    """requirement (W3): the filing is the unit of reuse. A filing an earlier
    analysis read is not selected again while it stays in the window: it is
    deferred as read earlier, and the capacity goes to what is new -- the
    request names what was read, and the plan is the request's own."""

    read = frozenset(WINDOWED[:2])
    plan = _plan(_FloodTransport(FLOOD), _request(budget=4), read_earlier=read)
    assert [value.accession for value in plan.events] == WINDOWED[2:6]
    assert [(value.accession, value.reason) for value in plan.deferred] == [
        (accession, "READ_EARLIER") for accession in WINDOWED[:2]
    ]
    assert plan.limitations == (), "a filing read earlier is carried, never a limitation"
    assert plan.outside_window_event_count == 8
    named = tuple(
        AlternativeEvidenceReadFiling(
            entity_id="AAPL", accession=accession, publication_hash="a" * 64
        )
        for accession in sorted(read)
    )
    request = _request(budget=4, read_filings=named)
    assert request.read_accessions("AAPL") == read
    assert _plan(_FloodTransport(FLOOD), request).plan_hash == plan.plan_hash
    with pytest.raises(ValueError, match="read_filings_invalid"):
        _request(budget=4, read_filings=(named[0].model_copy(update={"entity_id": "MSFT"}),))
    with pytest.raises(ValueError, match="read_filings_invalid"):
        _request(budget=4, read_filings=(named[0], named[0]))
    # Nothing read: the plan an earlier request sealed, hash and all.
    assert (
        _plan(_FloodTransport(FLOOD), _request(budget=4)).plan_hash
        == _plan(_FloodTransport(FLOOD), _request(budget=4), read_earlier=frozenset()).plan_hash
    )


def test_an_issuer_that_filed_nothing_in_the_window_has_an_empty_plan() -> None:
    """requirement (W1): nothing in the window, nothing read -- the plan is
    empty and every discovered filing is counted outside it; the packing names
    the issuer as nothing filed from the read that holds this plan."""

    plan = _plan(_FloodTransport(FLOOD[:5]), _request())
    assert plan.events == () and plan.deferred == () and plan.required == ()
    assert plan.outside_window_event_count == plan.discovered_count == 5
    read = SecFilingInventoryRead(
        plan=plan,
        read_hash=inventory_read_key(
            entity_id=plan.entity_id,
            evidence_as_of=plan.evidence_as_of,
            event_window_days=plan.event_window_days,
            policy_budget=plan.policy_budget,
            unit_capacity=plan.unit_capacity,
        ),
    )
    assert read.nothing_filed
    with pytest.raises(ValueError, match="inventory_read_invalid"):
        SecFilingInventoryRead(plan=plan, read_hash="0" * 64)
    # The plan says so itself, the one predicate every floor judge reads (V587), and an index
    # read establishes it at its own cutoff and window only. A plan scoped to named accessions
    # reads those, not the window, so it never says so -- even one the index holds nothing for.
    assert plan.nothing_filed
    held = {plan.entity_id: [plan]}
    cutoff, window = plan.evidence_as_of, plan.event_window_days
    assert index_quiet(held, evidence_as_of=cutoff, window_days=window) == {plan.entity_id}
    later = cutoff + timedelta(days=1)
    assert not index_quiet(held, evidence_as_of=later, window_days=window)
    assert not index_quiet(held, evidence_as_of=cutoff, window_days=window + 1)
    scoped = plan_sec_filing_selection(
        (),
        entity_id=plan.entity_id,
        cik=plan.cik,
        evidence_as_of=cutoff,
        event_window_days=window,
        policy_budget=plan.policy_budget,
        unit_capacity=plan.unit_capacity,
        accession_scope=frozenset({"0000320193-26-000009"}),
    )
    assert scoped.selected == () and scoped.deferred == () and not scoped.nothing_filed


def test_a_request_under_the_retired_policy_is_refused_at_planning() -> None:
    """requirement (W1): the 90-day window with fixed baselines is retired; a
    request sealed under it still validates (and reads back), is refused if
    planned again, and the window and the baselines cannot be mixed."""

    retired = AlternativeEvidenceSourcePolicy(
        sec_recent_8k_days=90, include_latest_10q=True, include_latest_10k=True
    )
    assert retired.retired
    with pytest.raises(ValueError, match="source_policy_retired"):
        _plan(_FloodTransport(FLOOD), _request(policy=retired))
    with pytest.raises(ValueError, match="source_policy_invalid"):
        AlternativeEvidenceSourcePolicy(include_latest_10k=True)
    with pytest.raises(ValueError, match="source_policy_invalid"):
        AlternativeEvidenceSourcePolicy(sec_recent_8k_days=90)


def test_an_issuer_s_plan_is_its_own_whatever_the_width_of_the_request() -> None:
    """requirement (6A): the logical selection is decided under the issuer's
    policy budget before any packing, so an issuer of an eight-issuer request
    plans the same filings as an issuer alone; the admitted set bounds the
    unit's plans together (`apply_unit_capacity`), never by division."""

    wide = _plan(_FloodTransport(FLOOD), _request(issuers=WIDE, budget=12))
    alone = _plan(_FloodTransport(FLOOD), _request(budget=12))
    assert wide.unit_capacity == ADMITTED_DOCUMENT_CAPACITY and wide.capacity == 12
    assert wide.selected == alone.selected and wide.deferred == alone.deferred
    assert wide.required == () and len(wide.events) == 6
    assert not any("admitted-set share" in value for value in wide.limitations)


def _windowed_plans(rows: list[tuple[str, ...]], issuers: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(
        plan_sec_filing_selection(
            _inventory(
                [(row[0].replace("0000320193", f"00003201{i:02d}"), *row[1:]) for row in rows],
                entity_id=entity,
                cik=f"00003201{i:02d}",
            ),
            entity_id=entity,
            cik=f"00003201{i:02d}",
            evidence_as_of=NOW,
            event_window_days=30,
            policy_budget=12,
            unit_capacity=ADMITTED_DOCUMENT_CAPACITY,
        )
        for i, entity in enumerate(issuers)
    )


def test_a_unit_whose_plans_exceed_the_admitted_set_defers_filings_round_robin() -> None:
    """requirement (6A): eight issuers' own plans of six filings each are
    bounded together: filings are deferred BEYOND_UNIT_CAPACITY from the
    issuer holding the most, oldest first, until the unit's selections fit the
    24-document set; a plan that fits is returned unchanged; the result is the
    same whatever order the plans arrive in."""

    plans = _windowed_plans(FLOOD, WIDE)
    assert all(len(plan.selected) == 6 for plan in plans)
    bounded = apply_unit_capacity(plans)
    assert sum(len(plan.selected) for plan in bounded) == ADMITTED_DOCUMENT_CAPACITY
    assert [plan.entity_id for plan in bounded] == list(WIDE), "order preserved"
    for before, after in zip(plans, bounded, strict=True):
        moved = [d for d in after.deferred if d.reason == "BEYOND_UNIT_CAPACITY"]
        assert len(moved) == 3 and len(after.events) == 3
        # The three oldest go first; the three newest stay.
        assert [e.accession[-3:] for e in after.events] == ["100", "101", "102"]
        assert {d.accession for d in moved} == {e.accession for e in before.events[3:]}
        assert any("pack fewer issuers per unit" in value for value in after.limitations)
        assert after.discovered_count == before.discovered_count
    reversed_plans = apply_unit_capacity(tuple(reversed(plans)))
    assert {p.plan_hash for p in reversed_plans} == {p.plan_hash for p in bounded}
    fits = apply_unit_capacity(plans[:4])
    assert fits == plans[:4], "a unit within the set is returned as it is"
    partial = apply_unit_capacity(plans[:5])
    assert sum(len(plan.selected) for plan in partial) == ADMITTED_DOCUMENT_CAPACITY
    assert sorted(len(plan.events) for plan in partial) == [4, 5, 5, 5, 5]


NEGATIVE: list[tuple[str, ...]] = [
    # Outside the window: an annual report and an older late-filing notice.
    ("0000320193-25-000010", "10-K", _at(320), "2025-09-30"),
    ("0000320193-25-000107", "NT 10-K", _at(200), "2025-09-30"),
    # Inside the window, newest first: a routine 8-K, a late-filing notice, a non-reliance
    # 8-K, another routine 8-K, a delisting notice and a quarterly report; a prospectus the
    # selection ignores.
    ("0000320193-26-000101", "8-K", _at(3), "", "8.01,9.01"),
    ("0000320193-26-000102", "NT 10-Q", _at(10), "2026-06-30"),
    ("0000320193-26-000103", "8-K", _at(15), "", "4.02"),
    ("0000320193-26-000104", "8-K", _at(20), "", "2.02,9.01"),
    ("0000320193-26-000105", "8-K", _at(25), "", "3.01,3.02"),
    ("0000320193-26-000020", "10-Q", _at(28), "2026-06-30"),
    ("0000320193-26-000106", "424B5", _at(5), ""),
]


def test_the_inventory_carries_an_eight_k_s_items_and_admits_late_filing_notices() -> None:
    """requirement (Q3): the submissions index states each current report's
    items at no extra request, so the inventory records them; a late-filing
    notice (NT 10-K, NT 10-Q) is a filing beside the 8-Ks; a form the
    selection does not read (a prospectus) is not discovered."""

    plan = _plan(_FloodTransport(NEGATIVE), _request(budget=12))
    assert plan.discovered_count == 8, "the prospectus is not discovered"
    by_accession = {entry.accession: entry for entry in plan.selected}
    assert by_accession["0000320193-26-000101"].items == ("8.01", "9.01")
    assert by_accession["0000320193-26-000105"].items == ("3.01", "3.02")
    assert by_accession["0000320193-26-000102"].form == "NT 10-Q"
    assert by_accession["0000320193-26-000102"].items == ()
    assert plan.outside_window_event_count == 2, "the 10-K and the old NT 10-K"
    # An empty item list stays out of the sealed form, so a plan sealed before
    # items were recorded reads back under the hash it was sealed with.
    assert "items" not in by_accession["0000320193-26-000020"].model_dump(mode="json")
    assert by_accession["0000320193-26-000103"].model_dump(mode="json")["items"] == ["4.02"]


def test_major_negatives_come_first_then_periodic_reports_then_routine_events() -> None:
    """requirement (Q3, W1): recency was the only order, so a non-reliance or a
    delisting notice followed by one routine 8-K was deferred behind it. Under
    capacity the late-filing notices and the 8-Ks carrying a negative item
    (1.03, 2.04, 2.06, 3.01, 4.01, 4.02) come first, then the periodic reports,
    then the routine current reports, each newest first."""

    plan = _plan(_FloodTransport(NEGATIVE), _request(budget=5))
    assert [entry.accession[-3:] for entry in plan.events] == ["102", "103", "105", "020", "101"]
    assert [(value.accession, value.reason) for value in plan.deferred] == [
        ("0000320193-26-000104", "BEYOND_CAPACITY"),
    ]


def test_the_unit_defers_routine_events_before_periodic_reports_and_major_negatives() -> None:
    """requirement (Q3): packing a unit defers from the issuer holding the most
    selected filings, its routine events oldest first, and a periodic report or
    a major negative only when that issuer has no routine event left;
    re-sealing a plan keeps its entries as they are (no serializer warnings
    from dumped dictionaries)."""

    plans = _windowed_plans(NEGATIVE, WIDE[:5])
    assert all(len(plan.events) == 6 for plan in plans)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bounded = apply_unit_capacity(plans)
    assert sum(len(plan.selected) for plan in bounded) == ADMITTED_DOCUMENT_CAPACITY
    moved = [
        [d.accession[-3:] for d in plan.deferred if d.reason == "BEYOND_UNIT_CAPACITY"]
        for plan in bounded
    ]
    # Thirty filings for a set of 24: every issuer's oldest routine event goes
    # first, then one issuer's other; no negative and no quarterly report moves.
    assert all(values[0] == "104" for values in moved)
    assert sorted(len(values) for values in moved) == [1, 1, 1, 1, 2]
    assert all(value in {"104", "101"} for values in moved for value in values)


def _inventory(
    rows: list[tuple[str, ...]], *, entity_id: str = "AAPL", cik: str = "0000320193"
) -> Any:
    from alphalattice.evidence.alternative_evidence.sources.sec_edgar import (
        _material_inventory_entries,
    )

    return _material_inventory_entries(
        json.loads(_submissions(rows)), entity_id=entity_id, cik=cik, evidence_as_of=NOW
    )


def test_acquisition_fetches_exactly_the_plan_and_records_it(tmp_path: Path) -> None:
    transport = _FloodTransport(FLOOD)
    request = _request(budget=5)
    service = AlternativeEvidenceAcquisitionService(artifact_root=tmp_path)
    registry, snapshot, documents = service.build_live_evidence(
        request=request,
        admission=_live_admission(request),
        source=SecEdgarSource(transport),
        published_at=NOW,
        clock=lambda: NOW,
    )
    fetched = [
        url.rsplit("/", 1)[-1].removesuffix(".htm")
        for url in transport.calls
        if "/Archives/" in url
    ]
    plans = service.artifacts.values("sec-selection-plans", SecFilingSelectionPlan)
    assert len(plans) == 1
    plan = plans[0]
    assert fetched == [value.accession for value in plan.selected] == WINDOWED[:5]
    assert [value.revision for value in documents] == fetched
    assert plan.required == ()
    assert snapshot.status == "COMPLETE"
    assert any(plan.plan_hash[:12] in value for value in snapshot.limitations)
    assert any("deferred" in value for value in snapshot.limitations)
    assert all(value.accepted_at <= request.evidence_as_of for value in snapshot.citations)
    del registry


def test_a_scoped_acquisition_fetches_the_named_originals_as_the_index_holds_them(
    tmp_path: Path,
) -> None:
    """requirement: named accessions are fetched exactly as the issuer's own
    official index holds them (form, primary document, ownership by CIK),
    every other discovered filing is deferred as outside the scope, an
    accession the index does not hold is named and never fetched from a
    guessed location, and each acquired document keeps its time apart: the
    filing date at DATE precision, the official acceptance, the reported
    period from the official metadata and the retrieval clock."""

    from alphalattice.evidence.alternative_evidence.documents.workspace import (
        AlternativeEvidenceDocumentPublisher,
    )

    transport = _FloodTransport(FLOOD)
    request = _request(budget=3)
    scope = frozenset({"0000320193-25-000010", "0000320193-25-000020", "0000320193-99-000001"})
    service = AlternativeEvidenceAcquisitionService(artifact_root=tmp_path)
    _registry, snapshot, documents = service.build_live_evidence(
        request=request,
        admission=_live_admission(request),
        source=SecEdgarSource(transport),
        published_at=NOW,
        clock=lambda: NOW,
        accession_scopes={"AAPL": scope},
    )
    (plan,) = service.artifacts.values("sec-selection-plans", SecFilingSelectionPlan)
    assert plan.accession_scope == tuple(sorted(scope))
    assert [v.accession for v in plan.selected] == ["0000320193-25-000010", "0000320193-25-000020"]
    assert plan.events == () and {v.reason for v in plan.deferred} == {"OUTSIDE_ACCESSION_SCOPE"}
    assert plan.discovered_count == len(plan.selected) + len(plan.deferred)
    assert any("0000320193-99-000001 is not in the cutoff-valid" in v for v in plan.limitations)
    fetched = [url.rsplit("/", 1)[-1] for url in transport.calls if "/Archives/" in url]
    assert fetched == ["0000320193-25-000010.htm", "0000320193-25-000020.htm"]
    assert snapshot.status == "COMPLETE"
    ten_k, ten_q = documents
    assert (ten_k.document_type, ten_q.document_type) == ("10-K", "10-Q")
    assert ten_k.published_precision == "DATE"
    assert ten_k.published_at == datetime.combine(
        ten_k.accepted_at.date(), datetime.min.time(), tzinfo=UTC
    )
    assert ten_k.report_period_end == date(2025, 9, 30)
    assert ten_q.report_period_end == date(2025, 12, 31)
    assert ten_k.retrieved_at == NOW and ten_k.available_at == ten_k.accepted_at
    # The retained bytes keep the same time; a reference sealed without the
    # new fields is a document whose time is unknown, not misread.
    publisher = AlternativeEvidenceDocumentPublisher(tmp_path / "workspace")
    reference = publisher.publish_source_object(ten_k)
    assert reference.report_period_end == date(2025, 9, 30)
    assert reference.published_precision == "DATE" and reference.retrieved_at == NOW
    assert publisher.resolve_source_object(reference) == ten_k
    legacy = reference.model_dump(mode="json")
    for name in ("published_precision", "report_period_end", "retrieved_at"):
        legacy.pop(name)
    old_form = type(reference).model_validate(legacy)
    assert old_form.published_precision is None and old_form.report_period_end is None
    assert old_form.model_dump(mode="json") == legacy
    with pytest.raises(ValueError, match="selection_accession_scope_invalid"):
        _plan(_FloodTransport(FLOOD), _request(budget=3), accession_scope=frozenset())
    with pytest.raises(ValueError, match="issuer_not_requested"):
        service.build_live_evidence(
            request=request,
            admission=_live_admission(request),
            source=SecEdgarSource(_FloodTransport(FLOOD)),
            published_at=NOW,
            clock=lambda: NOW,
            accession_scopes={"MSFT": scope},
        )
