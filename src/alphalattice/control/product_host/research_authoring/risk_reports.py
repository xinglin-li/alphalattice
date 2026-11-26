"""Exact report-only links; neither Portfolio economics nor Risk admission is changed."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, Literal, Self, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.product_host.composition.plain_refusals import explain
from alphalattice.control.product_host.research_authoring.factor_inputs import confined
from alphalattice.interface.local_application.cli_contract import refusal_words
from alphalattice.interface.local_application.failure_codes import public_failure
from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
    PortfolioPublicationError,
    PortfolioResearchArtifactStore,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import AuthoringError, ResolvedListingScope

CATEGORY = "development/risk-report-links"
_HASH = r"^[0-9a-f]{64}$"


def _risk_link_refusal(
    link_hash: str,
    failure_code: str,
    detail: str,
    portfolio_task_id: UUID,
) -> dict[str, object]:
    """Name one unreadable or misbound stored link and the read/recovery routes."""
    return {
        "status": "REFUSED",
        "link_hash": link_hash,
        "failure_code": failure_code,
        "detail": detail,
        "next_requests": {
            "links": {"operation": "EXPERIMENT_RISK_LINKS", "task_id": str(portfolio_task_id)},
            "studies": {"operation": "EXPERIMENTS"},
            "workspace": {"operation": "WORKSPACE_SHOW"},
            "backups": {"operation": "WORKSPACE_BACKUPS"},
        },
    }


class RiskReportWindow(BaseModel):  # type: ignore[misc]
    """An explicit retrospective view of a source, never a replacement surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    source_as_of: date
    portfolio_as_of: date
    claim: Literal["POST_OBSERVED_REFERENCE_NOT_AVAILABLE_AT_PORTFOLIO_CUTOFF"] = (
        "POST_OBSERVED_REFERENCE_NOT_AVAILABLE_AT_PORTFOLIO_CUTOFF"
    )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_axis(self) -> Self:
        """Require sorted unique report formation sessions.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Formation axis is unordered or duplicated.
        """
        if self.formation_sessions != tuple(sorted(set(self.formation_sessions))):
            raise ValueError("risk_report.window_axis_invalid")
        return self


class RiskReportLink(BaseModel):  # type: ignore[misc]
    """Seal an exact report reference with no allocation or current-risk authority.

    Seal an exact Portfolio-to-Risk report reference without allocation or current-risk authority.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    portfolio_task_id: UUID
    portfolio_receipt_hash: str = Field(pattern=_HASH)
    risk_task_id: UUID
    risk_surface_hash: str = Field(pattern=_HASH)
    risk_diagnostics_hash: str = Field(pattern=_HASH)
    input_binding_hash: str = Field(pattern=_HASH)
    risk_start: date
    risk_end: date
    risk_formation_count: int = Field(ge=1)
    portfolio_formation_count: int = Field(ge=1)
    assessed_listing_scopes: tuple[ResolvedListingScope, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    report_window: RiskReportWindow | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    attached_by: Literal["HUMAN", "INSTALLED_AGENT", "EXTERNAL_AUTOMATION"]
    claim: Literal["REPORT_REFERENCE_ONLY_NOT_ALLOCATION_OR_CURRENT_RISK"] = (
        "REPORT_REFERENCE_ONLY_NOT_ALLOCATION_OR_CURRENT_RISK"
    )
    link_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal an explicit Portfolio-to-Risk report reference.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical link_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        draft = cls.model_construct(**values, link_hash="")
        return cls(
            **values,
            link_hash=str(canonical_hash(draft.model_dump(mode="json", exclude={"link_hash"}))),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require ordered Risk interval and exact report link identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Risk interval is reversed or canonical link_hash differs.
        """
        if self.risk_start > self.risk_end or self.link_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"link_hash"})
        ):
            raise ValueError("risk_report.link_identity_invalid")
        return self

    def export_request(self) -> dict[str, str]:
        """Use the link identity, not its Risk surface or diagnostics identity."""
        return {
            "operation": "EXPERIMENT_RISK_EXPORT",
            "task_id": str(self.portfolio_task_id),
            "risk_report_hash": self.link_hash,
        }


class RiskReportLinks:
    """Host join over existing verified readback and the Portfolio artifact store."""

    def __init__(self, workspace: Path, read: Callable[[UUID], dict[str, object]]):
        """Wire confined report reference storage and exact study readback.

        Args:
            workspace: Caller-owned admitted workspace.
            read: Deterministic exact experiment readback callback.
        """
        self.workspace, self.read = workspace, read

    @staticmethod
    def selection(*, task_kind: str, completed_portfolio: bool) -> dict[str, object]:
        """Name the admitted report subject before an association is offered.

        Installed strategy books do not carry a Portfolio experiment's receipt and
        captured-input binding. Selection grants no Risk authority: an admitted study
        still undergoes the exact receipt, input, listing-axis and window checks.
        """
        body: dict[str, object] = {
            "available": completed_portfolio,
            "admitted_subject": "COMPLETED_PORTFOLIO_STUDY",
            "selected_task_kind": task_kind,
        }
        if not completed_portfolio:
            code = "risk_report.completed_portfolio_required"
            body.update(
                {
                    "failure_code": code,
                    **explain(code, kind=task_kind),
                    "next_requests": {"studies": {"operation": "EXPERIMENTS"}},
                }
            )
        return body

    def _portfolio(self, task_id: UUID) -> tuple[dict[str, Any], PortfolioResearchArtifactStore]:
        body = cast(dict[str, Any], self.read(task_id))
        if body.get("status") != "EXPERIMENT_PUBLISHED" or not body.get("portfolio_source"):
            raise AuthoringError("risk_report.completed_portfolio_required")
        output = confined(self.workspace, body["document"]["experiment"]["output_workspace"])
        return body, PortfolioResearchArtifactStore(output)

    @staticmethod
    def _require_listing_axis(child: dict[str, Any], portfolio_axis: list[str]) -> None:
        assessed = set(child["ordered_listing_ids"])
        if not assessed or child["ordered_listing_ids"] != [
            listing for listing in portfolio_axis if listing in assessed
        ]:
            raise AuthoringError("risk_report.input_or_axis_mismatch")

    @staticmethod
    def _link(
        portfolio: dict[str, Any],
        risk: dict[str, Any],
        caller: str,
        *,
        portfolio_window: bool = False,
    ) -> RiskReportLink:
        if risk.get("status") != "EXPERIMENT_PUBLISHED" or not risk.get("risk_surface"):
            raise AuthoringError("risk_report.completed_risk_required")
        source, surface, risk_input = (
            portfolio["portfolio_source"],
            risk["risk_surface"],
            risk["risk_input"],
        )
        if (
            source["input_binding_hash"] != risk["input_binding_hash"]
            or source["panel_snapshot_hash"] != risk_input["panel_snapshot_hash"]
            or source["universe_revision"] != risk_input["universe_revision_sha256"]
        ):
            raise AuthoringError("risk_report.input_or_axis_mismatch")
        portfolio_axis = source["ordered_listing_ids"]
        if not portfolio_window:
            RiskReportLinks._require_listing_axis(surface, portfolio_axis)
        sessions = surface["formation_sessions"]
        risk_as_of = risk["document"]["experiment"]["sessions"]["as_of"]["session"]
        portfolio_as_of = portfolio["document"]["experiment"]["sessions"]["as_of"]["session"]
        window = None
        children = surface.get("scope_surfaces") or (surface,)
        if portfolio_window:
            sessions = source["formation_sessions"]
            requested = set(sessions)
            if not sessions or not requested <= set(surface["formation_sessions"]):
                raise AuthoringError("risk_report.portfolio_coverage_incomplete")
            window = RiskReportWindow(
                formation_sessions=sessions,
                source_as_of=risk_as_of,
                portfolio_as_of=portfolio_as_of,
            )
            children = tuple(
                {
                    **child,
                    "formation_sessions": [
                        s for s in child["formation_sessions"] if s in requested
                    ],
                }
                for child in children
                if set(child["formation_sessions"]) & requested
            )
            covered = [s for child in children for s in child["formation_sessions"]]
            evaluations = [
                row["formation_session"]
                for row in risk["result"]["evaluations"]
                if row["formation_session"] in requested
            ]
            if sorted(covered) != sessions or sorted(evaluations) != sessions:
                raise AuthoringError("risk_report.portfolio_coverage_incomplete")
            for child in children:
                RiskReportLinks._require_listing_axis(child, portfolio_axis)
        else:
            if not set(sessions) <= set(source["formation_sessions"]):
                raise AuthoringError("risk_report.sessions_outside_portfolio")
        if risk_as_of > portfolio_as_of:
            raise AuthoringError("risk_report.cutoff_after_portfolio")
        return RiskReportLink.create(
            portfolio_task_id=UUID(portfolio["task_id"]),
            portfolio_receipt_hash=portfolio["receipt"]["receipt_hash"],
            risk_task_id=UUID(risk["task_id"]),
            risk_surface_hash=surface["surface_hash"],
            risk_diagnostics_hash=risk["result"]["diagnostics_hash"],
            input_binding_hash=risk["input_binding_hash"],
            risk_start=date.fromisoformat(sessions[0]),
            risk_end=date.fromisoformat(sessions[-1]),
            risk_formation_count=len(sessions),
            portfolio_formation_count=len(source["formation_sessions"]),
            assessed_listing_scopes=(
                tuple(
                    ResolvedListingScope(
                        first_session=child["formation_sessions"][0],
                        last_session=child["formation_sessions"][-1],
                        ordered_listing_ids=child["ordered_listing_ids"],
                    )
                    for child in children
                )
                if portfolio_window
                or surface.get("scope_surfaces")
                or surface["ordered_listing_ids"] != portfolio_axis
                else ()
            ),
            report_window=window,
            attached_by=caller,
        )

    def attach(
        self,
        portfolio_task_id: UUID,
        risk_task_id: UUID,
        *,
        caller: str,
        portfolio_window: bool = False,
    ) -> dict[str, object]:
        """Publish or reuse an exact verified Portfolio-to-Risk report link.

        Args:
            portfolio_task_id: Exact selected Portfolio study.
            risk_task_id: Exact selected Risk study.
            caller: Declared attaching actor.
            portfolio_window: Whether to assess the post-observed Portfolio window.

        Returns:
            Exact retained link and export request; no task or numerical call is created.
        """
        portfolio, store = self._portfolio(portfolio_task_id)
        link = self._link(
            portfolio, self.read(risk_task_id), caller, portfolio_window=portfolio_window
        )
        existed = (store.root / CATEGORY / f"{link.link_hash}.json").is_file()
        store.publish(category=CATEGORY, value=link, identity_field="link_hash")
        export_request = link.export_request()
        return {
            "status": "REUSED_EXACT" if existed else "RISK_REPORT_LINKED",
            "link": link.model_dump(mode="json"),
            "export_request": export_request,
            # The links read again from this receipt, bound to its book's Task, which the
            # receipt names only inside its link (V479).
            "next_requests": {
                "export": export_request,
                "links": {"operation": "EXPERIMENT_RISK_LINKS", "task_id": str(portfolio_task_id)},
            },
            "task_id": None,
            "numerical_call_count": 0,
        }

    def listing(self, portfolio_task_id: UUID) -> dict[str, object]:
        """Read stored report links and require their exact selected Portfolio receipt binding.

        Args:
            portfolio_task_id: Exact selected Portfolio study.

        Returns:
            Verified link declarations and exact export requests.

        Raises:
            AuthoringError: Stored link names a different Portfolio subject receipt.
        """
        portfolio, store = self._portfolio(portfolio_task_id)
        links = []
        export_requests = []
        refused_links: list[dict[str, object]] = []
        for path in sorted((store.root / CATEGORY).glob("*.json")):
            try:
                link = store.load(
                    category=CATEGORY,
                    content_hash=path.stem,
                    model=RiskReportLink,
                    identity_field="link_hash",
                )
            except PortfolioPublicationError as error:
                code = public_failure(error, "portfolio_strategy_lab.artifact_tampered")
                if code == "content_store.identity_invalid":
                    code = "portfolio_strategy_lab.artifact_identity_invalid"
                words = refusal_words(code)
                reason = words.get("detail") or (
                    "The saved Risk link does not have a readable content-addressed identity."
                )
                refused_links.append(
                    _risk_link_refusal(
                        path.stem,
                        code,
                        f"Saved Risk link {path.stem} was omitted because its record could not "
                        f"be read. {reason}",
                        portfolio_task_id,
                    )
                )
                continue
            if link.portfolio_task_id != portfolio_task_id:
                continue
            if link.portfolio_receipt_hash != portfolio["receipt"]["receipt_hash"]:
                refused_links.append(
                    _risk_link_refusal(
                        path.stem,
                        "risk_report.stored_subject_mismatch",
                        f"Saved Risk link {path.stem} names a different Portfolio receipt than "
                        f"Task {portfolio_task_id}; it is omitted from this book's report links.",
                        portfolio_task_id,
                    )
                )
                continue
            links.append(link.model_dump(mode="json"))
            export_requests.append(link.export_request())
        answer: dict[str, object] = {
            "status": "AVAILABLE",
            "links": links,
            "export_requests": export_requests,
        }
        if refused_links:
            answer["refused_links"] = refused_links
        return answer

    def read_link(self, portfolio_task_id: UUID, link_hash: str) -> dict[str, object]:
        """Verify a selected association once, independently of its HTML presentation."""
        portfolio, store = self._portfolio(portfolio_task_id)
        link = store.load(
            category=CATEGORY,
            content_hash=link_hash,
            model=RiskReportLink,
            identity_field="link_hash",
        )
        risk = self.read(link.risk_task_id)
        if link != self._link(
            portfolio, risk, link.attached_by, portfolio_window=link.report_window is not None
        ):
            raise AuthoringError("risk_report.stored_subject_mismatch")
        result = {"link": link.model_dump(mode="json"), "portfolio": portfolio, "risk": risk}
        if link.report_window is not None:
            sessions = {day.isoformat() for day in link.report_window.formation_sessions}
            result["report_projection"] = {
                "window": link.report_window.model_dump(mode="json"),
                "evaluations": [
                    row
                    for row in cast(dict[str, Any], risk)["result"]["evaluations"]
                    if row["formation_session"] in sessions
                ],
                "assessed_listing_scopes": [
                    scope.model_dump(mode="json") for scope in link.assessed_listing_scopes
                ],
                "source_surface_hash": link.risk_surface_hash,
                "source_diagnostics_hash": link.risk_diagnostics_hash,
            }
        return result

    def export(self, portfolio_task_id: UUID, link_hash: str) -> dict[str, object]:
        """Read an exact linked report snapshot and seal its rendered HTML export.

        Args:
            portfolio_task_id: Exact selected Portfolio study.
            link_hash: Exact retained report link.

        Returns:
            Verified linked snapshot, rendered HTML and canonical export identity.
        """
        from alphalattice.interface.local_application.experiment_report import (
            render_linked_risk_report,
        )

        snapshot = self.read_link(portfolio_task_id, link_hash)
        rendered = render_linked_risk_report(snapshot)
        result = {**snapshot, "html": rendered}
        return {**result, "export_hash": str(canonical_hash(result))}
