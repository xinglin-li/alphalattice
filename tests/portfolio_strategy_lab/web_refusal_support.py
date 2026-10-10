"""Real Local Web refusal probes over every registered route and public dependency."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import dataclass
from http.client import HTTPConnection
from typing import Any, Protocol
from urllib.parse import urlencode

from alphalattice.control.product_host.composition.goal_case_routes import CASE_ROUTES, case_request
from alphalattice.control.product_host.composition.local_web_session import (
    HANDLED_OPERATION_ROUTES,
    OPERATION_ROUTES,
)
from alphalattice.control.research_program.authoring import document as authoring_document
from alphalattice.interface.local_application import native_setup
from alphalattice.interface.local_application.activity import (
    ACTIVITY_MAXIMUM_LIMIT,
    ActivityReadQuery,
)
from alphalattice.interface.local_application.cli_contract import (
    AGENT_SESSION_HEADER,
    AGENT_VENDOR_HEADER,
    CLIENT_HEADER,
    INSTANCE_HEADER,
    WORKSPACE_HEADER,
    command_table,
    refusal_words,
)
from alphalattice.interface.local_application.client import workspace_connection_key
from alphalattice.interface.local_application.failure_codes import untyped_failure
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument,
)
from alphalattice.interface.local_application.web import SESSION_COOKIE, SESSION_HEADER

HASH = "a" * 64
TASK = "00000000-0000-4000-8000-000000000001"
UUID_FIELDS = frozenset(
    {
        "task_id",
        "origin_task_id",
        "factor_task_id",
        "risk_task_id",
        "left_task_id",
        "right_task_id",
        "goal_id",
        "update_task_id",
        "experiment_task_id",
    }
)
HASH_FIELDS = frozenset(
    {
        "incident_key",
        "data_issue_case_token",
        "evidence_index_id",
        "feature_trial_id",
        "continuation_of",
        "continuation_spans",
    }
)
LITERAL_SEEDS = {
    "storage_cap_bytes": "auto",
    "finding_handle": "FIND-PROBE",
    "remedy": "CANCEL",
    "experiment_yaml": "{}",
    "evidence_unit_id": "u001",
    "portfolio_session": "2024-08-09",
    "formation_session": "2024-08-09",
    "observed_through": "2024-08-09",
    "feature_output": "RAW_VALUES",
    "risk_report_scope": "POST_OBSERVED_PORTFOLIO_WINDOW",
    "experiment_curation": {
        "expected_receipt_hash": HASH,
        "choices": [],
        "limitations_acknowledged": [],
    },
}
# The fifteen own operation handlers omitted from the public operation-route tables.
LEGACY_OPERATION_ROUTES = (
    ("GET", "/api/data-update", "DATA_UPDATE_READBACK"),
    ("GET", "/api/strategy-score", "STRATEGY_SCORE_READBACK"),
    ("GET", "/api/research-update", "RESEARCH_UPDATE_READBACK"),
    ("GET", "/api/portfolio-update", "PORTFOLIO_UPDATE_READBACK"),
    ("GET", "/api/strategy-calibration", "STRATEGY_CALIBRATION_READBACK"),
    ("GET", "/api/controls", "CONTROLS"),
    ("GET", "/api/status", "STATUS"),
    ("POST", "/api/cancel", "CANCEL"),
    ("GET", "/api/results", "RESULTS"),
    ("GET", "/api/report", "REPORT"),
    ("GET", "/api/compare", "COMPARE"),
    ("POST", "/api/freeze", "FREEZE"),
    ("GET", "/api/finalization", "FINALIZATION"),
    ("GET", "/api/evidence-cro", "EVIDENCE_CRO"),
    ("GET", "/api/export", "EXPORT"),
)

REQUIRED_SEED_FIELDS = frozenset(
    {
        "analysis_answer",
        "analysis_context_hash",
        "analysis_publication_hash",
        "automation_enabled",
        "automation_package_ids",
        "calibration_plan_hash",
        "candidate_hash",
        "candidate_id",
        "change_reason",
        "component_id",
        "continuation_of",
        "continuation_spans",
        "curation_receipt_hash",
        "data_issue_case_token",
        "data_issue_evidence_hash",
        "data_issue_grant_hash",
        "data_issue_option_hash",
        "data_issue_option_id",
        "evidence_documents_per_issuer",
        "evidence_index_id",
        "evidence_setup",
        "experiment_curation",
        "experiment_document",
        "experiment_plan_hash",
        "experiment_receipt_hash",
        "feature_document",
        "feature_factor_id",
        "feature_plan_hash",
        "feature_trial_id",
        "finding_handle",
        "foundation_admission_hash",
        "goal_declaration",
        "goal_hash",
        "goal_reference",
        "goal_reference_id",
        "goal_statement",
        "incident_key",
        "input_binding_hash",
        "input_pinned",
        "left_candidate_id",
        "left_result_hash",
        "left_task_id",
        "model_id",
        "network_enabled",
        "portfolio_session",
        "preparation_plan_hash",
        "remedy",
        "research_input_id",
        "research_input_plan_hash",
        "result_hash",
        "review_answer",
        "review_dossier_hash",
        "review_policy_hash",
        "review_schema_hash",
        "right_candidate_id",
        "right_result_hash",
        "right_task_id",
        "risk_report_hash",
        "risk_task_id",
        "score_plan_hash",
        "score_snapshot_hash",
        "session_limit",
        "spec",
        "storage_plan_hash",
        "storage_cap_bytes",
        "strategy_package_id",
        "task_id",
        "update_plan_hash",
        "update_task_id",
        "upgrade_set_hash",
        "usage_reading_enabled",
        "window_limit",
    }
)
PUBLIC_SEAMS = frozenset(
    {
        "execute",
        "session_projection",
        "open_html",
        "load_authoring_document",
        "read",
        "read_external",
        "admitted_session_project",
    }
)
ACTIVATION_OPERATIONS = frozenset({"STRATEGY_ACTIVATE", "STRATEGY_DEACTIVATE"})
ACTIVATION_SEAMS = frozenset({"activate", "deactivate"})
RETURNED_TASK_AUTHORITY_ERRORS = frozenset(
    {
        "alphalattice.control.task_control.registry.TaskQueueHeadAuthorityError",
        "alphalattice.control.task_control.registry.TaskRecordAuthorityError",
    }
)


class RefusalCase(Protocol):
    """The real owner exception inventory's public case interface."""

    @property
    def name(self) -> str: ...

    @property
    def factory(self) -> Callable[[], Exception]: ...

    @property
    def code(self) -> str: ...

    @property
    def private_markers(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class FaultCase:
    """An ordinary fault with synthetic private text for the withholding check."""

    name: str
    factory: Callable[[], Exception]
    code: str = ""
    private_markers: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteProbe:
    route: Any
    fields: dict[str, Any]
    seam: str
    operation: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.route.method, self.route.path


def seed_field(name: str, types: dict[str, list[str]]) -> Any:
    if name in LITERAL_SEEDS:
        return json.loads(json.dumps(LITERAL_SEEDS[name]))
    if name in UUID_FIELDS:
        return TASK
    if name.endswith("_hash") or name in HASH_FIELDS:
        return HASH
    seeds = {
        "boolean": True,
        "integer": 1,
        "number": 1.0,
        "array": [],
        "object": {},
        "string": "probe",
    }
    kinds = types[name]
    assert len(kinds) == 1 and kinds[0] in seeds, ("new field needs seed", name, kinds)
    return seeds[kinds[0]]


def seed_operation(operation: str) -> dict[str, Any]:
    """Use public required fields/types and prove the request envelope first."""
    table = command_table()
    fields = {
        name: seed_field(name, table["types"]) for name in table["fields"][operation]["required"]
    }
    alternatives = table.get("alternatives", {}).get(operation)
    if alternatives and not any(set(option) <= fields.keys() for option in alternatives):
        chosen = next(
            (option for option in alternatives if "experiment_document" in option), alternatives[0]
        )
        fields.update({name: seed_field(name, table["types"]) for name in chosen})
    PortfolioResearchRequestDocument.model_validate({"operation": operation, **fields})
    return fields


def seed_case(operation: str) -> tuple[dict[str, Any], str]:
    fields = {
        "CASE_SAVE": {"case_id": TASK, "case_document": {}, "change_reason": "probe"},
        "CASE_ATTACH": {"case_reference": {}, "change_reason": "probe"},
        "CASE_NOTE": {"case_statement": {}, "change_reason": "probe"},
    }.get(operation, {})
    return fields, case_request(operation, fields).operation


def registered_route_probes(live: Any) -> tuple[RouteProbe, ...]:
    """Enumerate actual routes, retaining each original handler."""
    assert live.web is not None
    routes = live.web.application.routes
    plans = {}
    seeded_fields: set[str] = set()
    for method, path, operation in (
        *OPERATION_ROUTES,
        *HANDLED_OPERATION_ROUTES,
        *LEGACY_OPERATION_ROUTES,
    ):
        fields = seed_operation(operation)
        seeded_fields.update(fields)
        if path == "/api/compare":
            fields = {"left": fields["left_result_hash"], "right": fields["right_result_hash"]}
        plans[(method, path)] = fields, "execute", operation
    for method, path, operation in CASE_ROUTES:
        fields, translated = seed_case(operation)
        plans[(method, path)] = fields, "execute", translated
    plans.update(
        {
            ("POST", "/api/client/operations"): (
                {"operation": "OPERATION_LIST", **seed_operation("OPERATION_LIST")},
                "execute",
                "OPERATION_LIST",
            ),
            ("GET", "/api/workbench/portfolio"): (
                {"task_id": TASK},
                "execute",
                "PORTFOLIO_READBACK",
            ),
            ("GET", "/api/session"): ({}, "session_projection", None),
            ("GET", "/report"): ({"result_hash": HASH}, "open_html", None),
            ("GET", "/api/client/report"): ({"result_hash": HASH}, "open_html", None),
            ("POST", "/api/experiments/declaration"): (
                {"experiment_yaml": "{}"},
                "load_authoring_document",
                None,
            ),
            ("GET", "/api/activity"): ({}, "read", None),
            ("GET", "/api/activity/external"): ({}, "read_external", None),
            ("POST", "/api/client/session/bind"): (
                {"project": str(live.workspace.resolve())},
                "admitted_session_project",
                None,
            ),
            ("POST", "/api/client/session/event"): (
                {"project": str(live.workspace.resolve()), "event": {}},
                "admitted_session_project",
                None,
            ),
        }
    )
    assert seeded_fields == REQUIRED_SEED_FIELDS, {
        "new_seed_fields": sorted(seeded_fields - REQUIRED_SEED_FIELDS),
        "retired_seed_fields": sorted(REQUIRED_SEED_FIELDS - seeded_fields),
    }
    # New registrations must be covered by a valid request and a public seam.
    assert set(plans) == set(routes), {
        "unseeded": sorted(set(routes) - set(plans)),
        "retired": sorted(set(plans) - set(routes)),
    }
    probes = tuple(RouteProbe(routes[key], *plans[key]) for key in sorted(routes))
    assert {probe.seam for probe in probes} == PUBLIC_SEAMS
    return probes


def public_seam(live: Any, seam: str) -> tuple[Any, str, Any | None]:
    if seam == "load_authoring_document":
        return authoring_document, seam, None
    if seam == "admitted_session_project":
        return native_setup, seam, None
    if seam in {"execute", "session_projection"}:
        owner = live.operations
    elif seam == "open_html":
        owner = live.service
    elif seam in {"read", "read_external"}:
        owner = live.activity
    elif seam in {"activate", "deactivate"}:
        owner = live.operations.activations
    elif seam == "configure":
        owner = live.operations.automation
    else:
        raise AssertionError(("unknown public seam", seam))
    assert owner is not None, seam
    # Public class method patch is compatible with slots owners.
    return type(owner), seam, owner


def loopback_request(
    connection: HTTPConnection, live: Any, probe: RouteProbe
) -> tuple[int, dict[str, Any]]:
    route, fields = probe.route, dict(probe.fields)
    target, body = route.path, b""
    if route.method == "GET" and fields:
        assert all(isinstance(value, str | int | float | bool) for value in fields.values())
        target += "?" + urlencode(fields)
    elif route.method != "GET":
        body = json.dumps(fields, ensure_ascii=False).encode("utf-8")
    assert live.web is not None
    application = live.web.application
    headers = {"Content-Type": "application/json"}
    if route.external_client:
        client = live.client_connection
        assert client is not None
        headers.update(
            {
                CLIENT_HEADER: client.token,
                INSTANCE_HEADER: client.instance,
                WORKSPACE_HEADER: workspace_connection_key(live.workspace),
            }
        )
        if probe.seam == "admitted_session_project":
            headers.update({AGENT_VENDOR_HEADER: "codex", AGENT_SESSION_HEADER: TASK})
    else:
        headers.update(
            {
                "Cookie": f"{SESSION_COOKIE}={application.session_token}",
                SESSION_HEADER: application.session_token,
                "Origin": f"http://127.0.0.1:{live.web.bound_port}",
            }
        )
    connection.request(route.method, target, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    assert response.getheader("Content-Type", "").startswith("application/json"), (
        probe.key,
        response.status,
        raw[:200],
    )
    answer = json.loads(raw)
    assert isinstance(answer, dict), (probe.key, answer)
    return response.status, answer


def assert_owner_refusal(answer: dict[str, Any], case: RefusalCase) -> None:
    """Keep every owner's code and the exact catalogue words where declared."""
    code = answer.get("failure_code") or answer.get("refused")
    assert code == case.code and not untyped_failure(str(code)), (case.name, answer)
    words = refusal_words(case.code)
    if words:
        assert answer.get("status") == "REFUSED", (case.name, answer)
        assert answer.get("detail") == words["detail"], (case.name, answer, words)
        if answer.get("next_action"):
            assert answer["next_action"] == words["next_action"], (case.name, answer, words)
        else:
            # A bound owner offer is the documented alternative to a manual action.
            # Hold its admitted request, rather than demanding a redundant action label.
            from alphalattice.interface.local_application.cli_contract import (
                offered_requests,
                request_problem,
            )

            requests = offered_requests(answer)
            assert requests and all(request_problem(item) is None for item in requests.values()), (
                case.name,
                answer,
            )
    for marker in case.private_markers:
        assert marker not in json.dumps(answer, ensure_ascii=False), (case.name, marker, answer)


def inject_public_refusal(
    patch: Any, live: Any, probe: RouteProbe, case: RefusalCase
) -> dict[str, int]:
    target, attribute, owner = public_seam(live, probe.seam)
    counter = {"calls": 0}

    def raise_refusal(*args: Any, **kwargs: Any) -> Any:
        if owner is not None:
            assert args[0] is owner, (probe.key, case.name, probe.seam)
        if probe.seam == "execute":
            assert args[1].operation == probe.operation, (probe.key, args[1].operation)
        counter["calls"] += 1
        raise case.factory()

    patch.setattr(target, attribute, raise_refusal)
    return counter


def exercise_matrix(
    live: Any,
    monkeypatch: Any,
    probes: Iterable[RouteProbe],
    cases: Iterable[RefusalCase],
    expected_status: int,
) -> None:
    cases = tuple(cases)
    assert cases and live.web is not None
    # One real persistent loopback connection avoids thousands of TIME_WAIT ports.
    with closing(HTTPConnection("127.0.0.1", live.web.bound_port, timeout=10)) as connection:
        for probe in probes:
            original_handler = probe.route.handler
            for case in cases:
                with monkeypatch.context() as patch:
                    counter = inject_public_refusal(patch, live, probe, case)
                    status, answer = loopback_request(connection, live, probe)
                label = (probe.key, case.name, probe.seam)
                assert counter["calls"] == 1, (label, "owner not reached", status, answer)
                assert live.web.application.routes[probe.key].handler is original_handler, label
                # Native setup and Task Control publish these refusals as owner answers;
                # other raised owner refusals keep the HTTP400 boundary.
                owner_returned = (
                    probe.seam == "admitted_session_project"
                    and case.name
                    == "alphalattice.interface.local_application.native_bridge.NativeBridgeError"
                ) or (
                    probe.seam == "load_authoring_document"
                    and case.name in RETURNED_TASK_AUTHORITY_ERRORS
                )
                assert status == (200 if owner_returned else expected_status), (
                    label,
                    status,
                    answer,
                )
                assert_owner_refusal(answer, case)


def exercise_registered_handler_matrix(
    live: Any, monkeypatch: Any, cases: Iterable[RefusalCase]
) -> None:
    """Every registered handler x every eligible inventoried typed owner error."""
    exercise_matrix(live, monkeypatch, registered_route_probes(live), cases, 400)


def activation_owner_probes(live: Any) -> tuple[RouteProbe, ...]:
    """Keep real operations.execute and enter the activation owner's public methods."""
    assert live.web is not None and live.operations.activations is not None
    routes = live.web.application.routes
    probes = tuple(
        RouteProbe(
            routes[(method, path)],
            seed_operation(operation),
            "activate" if operation == "STRATEGY_ACTIVATE" else "deactivate",
            operation,
        )
        for method, path, operation in OPERATION_ROUTES
        if operation in ACTIVATION_OPERATIONS
    )
    assert {probe.operation for probe in probes} == ACTIVATION_OPERATIONS
    assert {probe.seam for probe in probes} == ACTIVATION_SEAMS
    return probes


def exercise_activation_owner_matrix(
    live: Any, monkeypatch: Any, cases: Iterable[RefusalCase]
) -> None:
    """Every typed owner error passes through the activation owner's real catch."""
    # Returned REFUSED dictionaries retain this route's HTTP200 answer contract.
    exercise_matrix(live, monkeypatch, activation_owner_probes(live), cases, 200)


def exercise_observed_owner_matrix(
    live: Any, monkeypatch: Any, cases: Iterable[RefusalCase]
) -> None:
    """A raised refusal crosses real execute/observer without becoming a returned answer."""
    cases = tuple(cases)
    assert live.web is not None
    key = ("POST", "/api/research-update/automation")
    probe = RouteProbe(
        live.web.application.routes[key],
        {"automation_enabled": False, "automation_package_ids": []},
        "configure",
        "RESEARCH_UPDATE_AUTOMATION_CONFIGURE",
    )
    # Task Control already publishes these two domain refusals as canonical owner
    # answers with per-record recovery. Keep that meaning as well as raised transport.
    from tests.portfolio_strategy_lab.typed_owner_refusals import OwnerRefusalCase

    preserved = tuple(case for case in cases if case.name not in RETURNED_TASK_AUTHORITY_ERRORS)
    canonical = tuple(
        OwnerRefusalCase(
            case.name,
            case.factory,
            "task_control.database_authority_unreadable",
            case.private_markers,
        )
        for case in cases
        if case.name in RETURNED_TASK_AUTHORITY_ERRORS
    )
    assert {case.name for case in canonical} == RETURNED_TASK_AUTHORITY_ERRORS
    initial = live.activity.read(ActivityReadQuery(limit=ACTIVITY_MAXIMUM_LIMIT))
    assert initial["cursor"] is not None, initial
    exercise_matrix(live, monkeypatch, (probe,), preserved, 400)
    exercise_matrix(live, monkeypatch, (probe,), canonical, 200)
    # An initial unaddressed read is a bounded tail; continue from the pre-test
    # cursor to verify every retained phase instead of only the newest page.
    query = {"limit": [str(ACTIVITY_MAXIMUM_LIMIT)], "after": [initial["cursor"]]}
    recorded = []
    while True:
        page = live.activity.read(ActivityReadQuery.from_query(query))
        recorded.extend(
            item["payload"]
            for item in page["items"]
            if isinstance(item.get("payload"), dict)
            and item["payload"].get("phase") in {"FAILED", "RETURNED"}
            and item["payload"].get("operation") == probe.operation
        )
        if not page["more"]:
            break
        query["after"] = [page["cursor"]]
    actual = Counter((item["phase"], item["failure_code"]) for item in recorded)
    expected = Counter(
        [("FAILED", case.code) for case in preserved]
        + [("RETURNED", case.code) for case in canonical]
    )
    assert actual == expected, (actual, expected)
    rendered = json.dumps(recorded, ensure_ascii=False)
    assert all(marker not in rendered for case in cases for marker in case.private_markers)


def exercise_untyped_fault_matrix(live: Any, monkeypatch: Any) -> None:
    """Genuine faults remain HTTP500, with fingerprints and withheld private text."""
    marker = "PRIVATE-WEB-FAULT-PAYLOAD/never-serve-this"
    fault = FaultCase(
        "ordinary RuntimeError", lambda: RuntimeError(marker), private_markers=(marker,)
    )
    representatives: dict[str, RouteProbe] = {}
    for probe in registered_route_probes(live):
        representatives.setdefault(probe.seam, probe)
    assert set(representatives) == PUBLIC_SEAMS
    for probe in activation_owner_probes(live):
        assert probe.seam not in representatives, probe.seam
        representatives[probe.seam] = probe
    assert set(representatives) == PUBLIC_SEAMS | ACTIVATION_SEAMS
    assert live.web is not None
    with closing(HTTPConnection("127.0.0.1", live.web.bound_port, timeout=10)) as connection:
        for probe in representatives.values():
            original_handler = probe.route.handler
            with monkeypatch.context() as patch:
                counter = inject_public_refusal(patch, live, probe, fault)
                status, answer = loopback_request(connection, live, probe)
            label = (probe.key, probe.seam)
            assert counter["calls"] == 1, (label, "owner not reached", status, answer)
            assert live.web.application.routes[probe.key].handler is original_handler, label
            assert status == 500, (label, status, answer)
            code = answer.get("failure_code") or answer.get("refused")
            assert untyped_failure(str(code)), (label, answer)
            assert str(code).startswith("local_web.handler_failed:RuntimeError:"), (label, answer)
            assert marker not in json.dumps(answer, ensure_ascii=False), (label, answer)


__all__ = [
    "RefusalCase",
    "activation_owner_probes",
    "exercise_activation_owner_matrix",
    "exercise_registered_handler_matrix",
    "exercise_untyped_fault_matrix",
    "registered_route_probes",
]
