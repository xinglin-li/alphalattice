"""The review verb's captured preparation and resumed Task readback."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

from alphalattice.interface.local_application.cli_contract import offered_requests, outcome_of
from alphalattice.interface.local_application.failure_codes import LocalResearchClientError


def select_preparation(
    chain: Any, captured: dict[str, Any], request: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Prefer the original Task only when the owner offers the captured identity."""
    if any(captured.values()) and not all(captured.values()):
        raise LocalResearchClientError("local_client.request_invalid")
    preview = chain.send("evidence_preview", request)
    resume = chain.offered(preview, "resume")
    if any(captured.values()):
        retained = preview.get("resumable_preparation") or {}
        try:
            cutoff = captured["evidence_as_of"]
            same = isinstance(cutoff, str) and datetime.fromisoformat(
                cutoff
            ) == datetime.fromisoformat(retained.get("evidence_as_of", ""))
        except (TypeError, ValueError):
            same = False
        if (
            same
            and retained.get("preparation_binding_hash") == captured["preparation_binding_hash"]
            and resume
        ):
            return preview, resume
        return preview, {**request, "operation": "EVIDENCE_PREPARE", **captured}
    prepare = resume or chain.offered(preview, "prepare")
    if prepare is None or (
        prepare.get("operation") != "STATUS"
        and preview.get("status") != "EVIDENCE_PREPARATION_READY"
    ):
        return preview, None
    return preview, prepare


def read_resumed(
    chain: Any,
    prepared: dict[str, Any],
    prepare: dict[str, Any],
    request: dict[str, Any],
    pending: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Read a resumed Task and only its own prepared-unit offers."""
    if pending:
        prepared = chain.send("evidence_status", prepare)
        if outcome_of(prepared) != "OK":
            return prepared, {}, chain.stop("evidence", prepared)
    current = chain.send("evidence_prepared", request)
    retained = current.get("resumable_preparation")
    if not isinstance(retained, dict) or retained.get("task_id") != prepared.get("task_id"):
        return prepared, current, chain.stop("evidence_prepared", current)
    current["next_requests"] = {
        name: offer
        for name, offer in offered_requests(current).items()
        if not name.startswith("analyst_bundle")
        or (isinstance(offer, dict) and offer.get("task_id") == prepared.get("task_id"))
    }
    if prepared.get("units_failed") and not any(
        name.startswith("analyst_bundle") for name in current["next_requests"]
    ):
        return prepared, current, chain.stop("evidence", prepared)
    return prepared, current, None


def review_evidence_view(
    preview: dict[str, Any], current: dict[str, Any], prepared: dict[str, Any], *, resuming: bool
) -> dict[str, Any]:
    """Name the original cutoff and failures without attributing a newer preview to it."""
    retained = preview.get("resumable_preparation") or {}
    cutoff = (
        retained.get("evidence_as_of")
        if retained.get("task_id") == prepared.get("task_id")
        else None
    )
    return {
        "status": current.get("status") or current.get("disposition"),
        "evidence_as_of": cutoff
        if resuming
        else current.get("evidence_as_of") or prepared.get("evidence_as_of"),
        "coverage": None if resuming else current.get("coverage") or preview.get("coverage"),
        "prepared_task_id": prepared.get("task_id"),
        **(
            {"units_failed": prepared["units_failed"], "failed_units": prepared["failed_units"]}
            if prepared.get("units_failed")
            else {}
        ),
    }


def review_prepared(
    chain: Any,
    *,
    root: Path,
    readback: dict[str, Any],
    package: str,
    update: Any,
    subject: dict[str, Any],
    preview: dict[str, Any],
    current: dict[str, Any],
    prepared: dict[str, Any],
    resuming: bool,
) -> dict[str, Any]:
    """Return the prepared units and any sealed failures in the agent answer."""
    bundles = chain.bundles(current, root)
    if isinstance(bundles, dict):
        return bundles
    if prepared.get("units_failed") and not bundles:
        return cast(dict[str, Any], chain.stop("evidence", prepared))
    review = chain.offered(readback, "review")
    return {
        "status": "BOOK_REVIEW_READY" if bundles else "EVIDENCE_PREPARED",
        "strategy_package_id": package,
        ("positions" if update else "book"): subject,
        "evidence": review_evidence_view(preview, current, prepared, resuming=resuming),
        "analyst_bundles": bundles,
        "steps": chain.steps,
        "next_action": "DISPATCH_ANALYSTS" if bundles else "READ_EVIDENCE",
        "detail": (
            "Start each Analyst with its bundle: each submits its answer by its route. With their "
            "receipts back, `review continue --dir` reads them, prepares the CRO's bundle."
            if bundles
            else "Evidence is prepared with no Analyst bundle to answer; read its review state."
        ),
        "next_requests": {} if review is None else {"review": review},
    }
