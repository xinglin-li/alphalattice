"""Shared loopback Evidence/CRO fixture; recorded documents and submitted actors only."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from alphalattice.control.product_host.composition.local_web_session import (
    EvidenceReviewAuthority,
    LocalPortfolioWebSession,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
)
from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    AlternativeEvidenceRetrievalAccessReceipt,
)
from alphalattice.evidence.alternative_evidence.runtime.policy import AdmittedEvidencePolicy
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    AlternativeEvidenceDocumentTaskResources,
)
from alphalattice.evidence.alternative_evidence.sources.recorded import RecordedEvidenceDocument
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
    PortfolioResearchRequestDocument,
)
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument as PortfolioResearchAgentRequest,
)
from alphalattice.interface.local_application.web import SESSION_COOKIE, SESSION_HEADER
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioResearchSpec,
)
from alphalattice.protocols.actor_execution import ActorKind
from tests.alternative_evidence_desk.issuer_listing import (
    ISSUER_TOPICS,
    _documents,
    _listing_authority,
    _registry,
)
from tests.alternative_evidence_desk.planted_corpus import _NOW, _CitingActor, _runtime
from tests.alternative_evidence_desk.review_dossiers import CitingReviewActor, ControlledRisk
from tests.portfolio_strategy_lab.local_web_support import (
    TEST_PACKAGE,
    InstalledAgent,
    _harness,
    _resolved,
    _Resolver,
    _run,
)


def _workspace_manifest(workspace_id: str) -> ResearchWorkspaceManifest:
    return ResearchWorkspaceManifest.create(
        workspace_id=workspace_id,
        default_strategy_package_id=TEST_PACKAGE.strategy_id,
        default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
        strategy_artifacts=(),
    )


# ==================================================================== fixture


class _Service:
    """One running local service over a workspace that holds a development result."""

    def __init__(self, session: LocalPortfolioWebSession, tmp_path: Path) -> None:
        self.session = session
        self.tmp_path = tmp_path

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        token: str | None = "valid",
    ) -> tuple[int, dict[str, Any]]:
        application = self.session.web.application  # type: ignore[union-attr]
        port = self.session.web.bound_port  # type: ignore[union-attr]
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(f"{self.session.url}{path}", data=body, method=method)
        request.add_header("Host", f"127.0.0.1:{port}")
        if method == "POST":
            request.add_header("Origin", f"http://127.0.0.1:{port}")
        if body is not None:
            request.add_header("Content-Type", "application/json")
        resolved = application.session_token if token == "valid" else token
        if resolved is not None:
            request.add_header(SESSION_HEADER, resolved)
            request.add_header("Cookie", f"{SESSION_COOKIE}={resolved}")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read() or b"{}")

    def post(self, path: str, payload: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        status, body = self.request(path, method="POST", payload=payload or {}, **kwargs)
        assert status == 200, (status, body)
        return body

    def get(self, path: str, **kwargs: Any) -> dict[str, Any]:
        status, body = self.request(path, **kwargs)
        assert status == 200, (status, body)
        return body

    def drain(self) -> None:
        self.session.dispatcher.drain_for_tests()  # type: ignore[union-attr]

    @property
    def review(self) -> Any:
        return self.session.review

    @property
    def registry(self) -> Any:
        return self.session.session.task_control_registry  # type: ignore[union-attr]

    def evidence_cro(
        self, result_hash: str | None = None, *, review_publication_hash: str | None = None
    ) -> dict[str, Any]:
        query = {
            key: value
            for key, value in (
                ("result_hash", result_hash),
                ("review_publication_hash", review_publication_hash),
            )
            if value is not None
        }
        suffix = "" if not query else "?" + urllib.parse.urlencode(query)
        return self.get(f"/api/evidence-cro{suffix}")

    def result_hash(self) -> str:
        results = self.get("/api/results")["results"]
        assert len(results) == 1
        return str(results[0]["result_hash"])

    def agent(self, request: PortfolioResearchAgentRequest) -> Any:
        assert self.session.operations is not None
        return json.loads(InstalledAgent(self.session.operations).invoke(request))


class _RefusingActor:
    """Any call is a defect: without a credential no model work may happen."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"a model actor was called without an admitted credential: {name}")


def build_authority(
    *,
    tmp_path: Path,
    report: Any,
    with_runtime: bool = True,
    with_actor: bool = True,
    model_authority_admitted: bool = True,
    document_suffix: str = "",
    extra_documents: Callable[[tuple[str, ...]], tuple[RecordedEvidenceDocument, ...]]
    | None = None,
) -> EvidenceReviewAuthority:
    """Everything a host must be handed, and nothing bound to a session.

    Constructible before any service exists and reusable across restarts: the
    evidence *runtime* and its typed resources are held here, and the
    composition owner binds them to the session's registry once it has one.
    `extra_documents` adds recorded filings for the book's admitted entities
    beside the planted releases (a 10-K with a litigation note, say).
    """

    listings = tuple(value.listing_id for value in report.window_end_book.positions)
    runtime = _runtime(tmp_path / "evidence")
    registry_snapshot = _registry()
    listing_authority = _listing_authority(listings)
    entities = tuple(sorted({value.ticker for value in listing_authority.entries}))
    resources = AlternativeEvidenceDocumentTaskResources(
        recorded_registry=registry_snapshot,
        recorded_documents=(
            *_documents(entities, suffix=document_suffix),
            *(() if extra_documents is None else extra_documents(entities)),
        ),
        analysis_actor=(
            _CitingActor(topics=ISSUER_TOPICS)
            if model_authority_admitted
            else cast(Any, _RefusingActor())
        ),
    )
    return EvidenceReviewAuthority(
        model_authority_admitted=model_authority_admitted,
        registry=registry_snapshot,
        listing_authority=listing_authority,
        artifacts=runtime.artifacts,
        evidence_publications=runtime.publications if with_runtime else None,
        evidence_runtime=runtime if with_runtime else None,
        evidence_resources=resources if with_runtime else None,
        evidence_policy=AdmittedEvidencePolicy(),
        review_actor=(
            cast(Any, _RefusingActor())
            if not model_authority_admitted
            else CitingReviewActor(
                risks=(ControlledRisk("AAPL"),),
                actor_kind=ActorKind.HUMAN,
                actor_id="gate-9c5-http",
            )
            if with_actor
            else None
        ),
    )


def build_workspace(tmp_path: Path) -> Any:
    """One real Portfolio path, published, with its lease released again."""

    portfolio = tmp_path / "portfolio"
    portfolio.mkdir(parents=True, exist_ok=True)
    with _harness(portfolio) as harness:
        result = _run(harness, PortfolioResearchSpec.default())
        report = harness.application.report(result.result_hash)
        return harness.workspace, report


def start_service(
    workspace: Path,
    authority: EvidenceReviewAuthority | None,
    tmp_path: Path,
    *,
    clock: Any = None,
) -> _Service:
    session = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_workspace_manifest("qa-gate-9c5-http"),
        resolver=_Resolver(_resolved()),
        clock=clock or (lambda: _NOW),
        review_authority=authority,
    )
    session.start()
    return _Service(session, tmp_path)


def _raise_interruption(*args: object, **kwargs: object) -> object:
    del args, kwargs
    raise RuntimeError("simulated interruption")


def _operation(service: _Service, document: dict[str, Any]) -> dict[str, Any]:
    """The CLI's path: the one operation owner, in process."""

    assert service.session.operations is not None
    return service.session.operations.execute(
        PortfolioResearchOperationRequest(**document), caller="HUMAN"
    )


def _parts(service: _Service, document: dict[str, Any]) -> list[dict[str, Any]]:
    """Every delivered part of a packet, followed through the executable
    pagination requests: what a consumer actually receives."""

    parts = [_operation(service, document)]
    while parts[-1]["delivery"]["next_request"] is not None:
        parts.append(_operation(service, parts[-1]["delivery"]["next_request"]))
    return parts


def delivered_aliases(parts: list[dict[str, Any]]) -> dict[str, str]:
    """Each delivered span's alias, as the packet parts a consumer received
    name it: what an answer cites."""

    return {
        row["span_handle"]: row["alias"]
        for part in parts
        for row in json.loads(part["packet"].split("\n\n", 2)[1])["spans"]
    }


def _summary(body: dict[str, Any]) -> dict[str, Any]:
    """The compact litigation block of a delivered packet part."""

    payload = json.loads(body["packet"].split("\n\n", 2)[1])
    return payload["litigation_matters"]


def _receipt(service: _Service, receipt_hash: str) -> AlternativeEvidenceRetrievalAccessReceipt:
    adapter = service.review.evidence_task_adapter
    return adapter.runtime.artifacts.load(
        "retrieval-access-receipts", receipt_hash, AlternativeEvidenceRetrievalAccessReceipt
    )


def _continue(service: _Service, request: dict[str, Any], **limits: int) -> dict[str, Any]:
    """Execute a delivery's continuation request as the product returned it --
    every field it carries, only the limits filled -- through the typed
    operation document (a field the operation does not accept is refused
    there) and the operations owner, and drain the Task it admitted."""

    document = PortfolioResearchRequestDocument.model_validate({**request, **limits})
    assert service.session.operations is not None
    body = service.session.operations.execute(document.to_operation_request(), caller="HUMAN")
    if body.get("task_id"):
        service.drain()
    return body
