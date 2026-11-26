"""Read-only clock disclosure; no new calendar, admission or numerical identity."""

from __future__ import annotations

from datetime import date
from functools import cache
from pathlib import Path
from typing import Any, TypedDict, cast

from alphalattice.control.product_host.maintenance.data_update import PROFILE
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    factor_input_paths,
    read_factor_bundle,
)
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    SourceAvailabilityCatalog,
)
from alphalattice.foundation.feature_engine.publication.temporal_statement import (
    TemporalStatement,
)
from alphalattice.foundation.market_data_ops.sources.manifest import read_market_profile


class ResearchTimingSummary(TypedDict):
    """Presentation metadata, never a caller-supplied execution clock."""

    claim: str
    declared: dict[str, Any]
    evaluated: dict[str, Any]
    input: dict[str, Any]
    availability: dict[str, Any]
    outcome: dict[str, Any]
    selected_event: dict[str, Any]
    training: dict[str, Any]
    temporal_scope: dict[str, Any]
    """What the window can claim about time, from its Panel's marks."""
    notices: list[str]


@cache
def installed_price_basis() -> str:
    """The installed market profile's price basis, which the temporal statement names."""
    basis: str = read_market_profile(PROFILE).daily_price_basis
    return basis


def temporal_scope(
    panel: dict[str, Any], *, window_start: object, window_end: object, data_start: object
) -> dict[str, Any]:
    """A window's temporal statement from its Panel manifest.

    Args:
        panel: The Panel manifest the result read.
        window_start: The first session the result claims.
        window_end: The last.
        data_start: The first session its inputs read.

    Returns:
        The statement's marks and sentences, as a readback carries them.
    """
    statement = TemporalStatement.from_panel(
        cast(dict[str, Any], panel.get("safe_summary", {})),
        window_start=_day(window_start),
        window_end=_day(window_end),
        data_start=_day(data_start),
        price_basis=installed_price_basis(),
    )
    return {"status": "RECORDED", **statement.model_dump(mode="json")}


def binding_temporal_scope(
    workspace: Path, binding_hash: str, *, window_start: object, window_end: object
) -> dict[str, Any]:
    """The temporal statement of a result bound to one research input, from that input's Panel.

    Args:
        workspace: The Host's workspace.
        binding_hash: The research input the result read.
        window_start: The first session the result claims.
        window_end: The last.

    Returns:
        The statement, or `NOT_RECORDED` when the input or its Panel is not kept.
    """
    try:
        bundle = read_factor_bundle(workspace, binding_hash, verify=False)
        _, artifacts = factor_input_paths(workspace, binding_hash)
        resolver = ArtifactResolver(artifacts)
        panel = resolver.load_feature_panel_manifest(
            resolver.feature_panel_manifest_uri(bundle.panel_snapshot_hash)
        )
    except FileNotFoundError:
        return {"status": "NOT_RECORDED"}
    return temporal_scope(
        panel,
        window_start=window_start or bundle.sessions[0],
        window_end=window_end or bundle.sessions[-1],
        data_start=bundle.sessions[0],
    )


def _day(value: object) -> date | None:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10]) if value else None


def research_timing(
    *,
    workspace: Path,
    binding_hash: str,
    document: dict[str, Any],
    preview: dict[str, Any],
    formations: tuple[str, ...] = (),
    selected_session: str | None = None,
) -> ResearchTimingSummary:
    """Project exact input metadata and the Outcome owner's stored schedule.

    Called after PLAN/readback admission, not from a History list. Manifest
    identity is checked here; this does not repeat input-array verification or
    claim that the provider delivered each historical row at its policy time.
    Legacy metadata absence is explicit. Corrupt identities still raise.
    """
    experiment = document.get("experiment", {})
    declared = experiment.get("sessions", {})
    kind = experiment.get("kind")
    result: ResearchTimingSummary = {
        "claim": "READ_ONLY_CLOCK_PROJECTION_NOT_EXECUTION_AUTHORITY",
        "declared": {**declared, "meaning": "RESEARCH_CUTOFF_NOT_EACH_HISTORICAL_DECISION"},
        "evaluated": {
            "start": formations[0] if formations else preview.get("statistical_start"),
            "end": formations[-1] if formations else preview.get("statistical_end"),
            "count": len(formations) if formations else preview.get("statistical_session_count"),
            "meaning": preview.get("interval_semantics"),
        },
        "input": {"binding_hash": binding_hash, "status": "NOT_RECORDED"},
        "availability": {"status": "NOT_RECORDED", "provider_arrival": "NOT_MEASURED"},
        "outcome": {"status": "NOT_RECORDED"},
        "selected_event": {"status": "NOT_RECORDED"},
        "training": {
            "split_policy": preview.get("split_policy") or preview.get("window_policy"),
            "folds": preview.get("folds"),
            "maturity_lag_sessions": preview.get("maturity_lag_sessions"),
            "lifecycle": document.get("alpha", {}).get("lifecycle"),
            "meaning": "REFIT_CADENCE_IS_NOT_REBALANCE_CADENCE",
        },
        "temporal_scope": {"status": "NOT_RECORDED"},
        "notices": [
            "The research cutoff is not each historical decision or the download time.",
            "Source availability is declared policy, not measured provider arrival or PIT proof.",
            "A mature outcome may inform later research, never its own earlier decision.",
            "An outcome interval is not an instruction to liquidate the whole book.",
        ],
    }
    try:
        bundle = read_factor_bundle(workspace, binding_hash, verify=False)
    except FileNotFoundError:
        return result
    result["input"].update(
        status="IDENTITY_BOUND_MANIFEST",
        start=str(bundle.sessions[0]),
        end=str(bundle.sessions[-1]),
        panel_snapshot_hash=bundle.panel_snapshot_hash,
        outcome_snapshot_hash=bundle.outcome_snapshot_hash,
    )
    _, artifacts = factor_input_paths(workspace, binding_hash)
    resolver = ArtifactResolver(artifacts)
    try:
        panel = resolver.load_feature_panel_manifest(
            resolver.feature_panel_manifest_uri(bundle.panel_snapshot_hash)
        )
    except FileNotFoundError:
        panel = {}
    if panel:
        result["temporal_scope"] = temporal_scope(
            panel,
            window_start=result["evaluated"]["start"] or bundle.sessions[0],
            window_end=result["evaluated"]["end"] or bundle.sessions[-1],
            data_start=bundle.sessions[0],
        )
    lineage = cast(dict[str, Any], panel.get("safe_summary", {})).get("lineage", {})
    recorded = lineage.get("source_authorities")
    if recorded is not None:
        catalog = SourceAvailabilityCatalog.model_validate(recorded)
        result["availability"].update(
            status="RECORDED_SOURCE_POLICY",
            policy_hash=catalog.catalog_hash,
            sources=[
                {**v.model_dump(mode="json"), "phase_name": v.available_after_phase.name}
                for v in catalog.owners
            ],
        )
    # Risk evaluates its own next-session return surface, not the Panel's
    # forward open-to-open target. Never label a covariance diagnostic as a trade.
    if kind == "risk.covariance-development":
        result["outcome"] = {
            "status": "RISK_DIAGNOSTIC_NOT_EXECUTION",
            "latest_outcome_session": preview.get("latest_outcome_session"),
            "lookback_sessions": preview.get("lookback_sessions"),
        }
        return result
    reader = CausalExecutionOutcomeDevelopmentReader(artifacts)
    seal = reader.resolve_method_seal(bundle.outcome_snapshot_hash)
    if seal.binding is not None:
        result["outcome"] = {"status": seal.disposition, **seal.binding.model_dump(mode="json")}
    else:
        result["outcome"] = {"status": seal.disposition}
    # Only a Portfolio selection claims execution events. Other studies keep
    # the outcome recipe separate from their target transformation/maturity.
    if kind == "portfolio.policy-development":
        day = selected_session or result["evaluated"]["start"]
        if day is not None:
            schedule = reader.read_development_schedule(
                reader.manifest_uri(bundle.outcome_snapshot_hash)
            )
            rows = [r for r in schedule.to_pylist() if str(r["formation_session"]) == day]
            if len(rows) != 1:
                raise ValueError("research_timing.selected_formation_not_in_outcome_schedule")
            result["selected_event"] = {
                "status": "RECORDED_OUTCOME_SCHEDULE",
                "purpose": "SELECTED_PORTFOLIO_FORMATION" if selected_session else "PLAN_EXAMPLE",
                **{k: v.isoformat() if hasattr(v, "isoformat") else v for k, v in rows[0].items()},
            }
    return result
