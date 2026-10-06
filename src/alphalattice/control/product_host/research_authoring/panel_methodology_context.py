"""Discover the runtime locations a Panel methodology request needs.

`PanelMethodologySourceRoots` is deliberately operational: no path is admitted
into scientific configuration. What was missing is the other half of that
contract -- a way to *find* those locations. Seven had to be supplied on the
command line, none was documented, and the only way to learn them was to read
the code that opens them. A researcher expressing an ordinary experiment cannot
be expected to do that.

This module answers the question at the owner. Each location is probed by the
structure its own reader requires, and the probe reports what it checked rather
than asserting a convention. Nothing here opens a numerical surface, resolves a
Program, or touches a pointer.

**A manifest is operational routing and never scientific identity.** It carries
where to look. The identities that enter a Program are the ones obtained by
actually opening the artifacts during preflight and execution, never the ones a
manifest claims.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

type PanelMethodologyLocationStatus = Literal[
    "RESOLVED",
    "NOT_FOUND_IN_SEARCH_SCOPE",
    "REJECTED_BY_OWNER",
    "AMBIGUOUS",
]

__all__ = [
    "PANEL_METHODOLOGY_LOCATION_IDS",
    "PanelMethodologyContextError",
    "PanelMethodologyLocationProbe",
    "discover_panel_methodology_context",
    "load_panel_methodology_context_manifest",
    "render_panel_methodology_context",
]


class PanelMethodologyContextError(ValueError):
    """A manifest that cannot route, or a discovery that cannot be trusted."""


PANEL_METHODOLOGY_LOCATION_IDS: Final[tuple[str, ...]] = (
    "repository_root",
    "panel_artifact_root",
    "legacy_panel_artifact_root",
    "feature_artifact_root",
    "failed_baseline_workspace",
    "sector_context_artifact_root",
    "execution_outcome_artifact_root",
)

_OPTIONAL_LOCATION_IDS: Final[frozenset[str]] = frozenset(
    {
        "portfolio_market_workspace",
        "portfolio_tradability_artifact_root",
        "r0_evidence_root",
        "r1_evidence_root",
        "risk_return_artifact_root",
    }
)

_ARTIFACT_DIRECTORY: Final = "artifacts"
_FACTOR_RESEARCH_STORE: Final = "factor-research"
_ALPHA_RESEARCH_STORE: Final = "alpha-research"
_FEATURE_PANEL_STORE: Final = "feature-panel"
_EXECUTION_OUTCOME_MANIFESTS: Final = Path("data-operations") / "execution-outcomes" / "manifests"
# The baseline reader's own criterion. Probing for an `alpha-research` store
# instead accepted workspaces that hold no report, which is how a location that
# exists one directory away was reported as absent.
_BASELINE_REPORT: Final = Path("reports") / "dynamic-panel-score-aggregation.json"


@dataclass(frozen=True, slots=True)
class PanelMethodologyLocationProbe:
    """One operational location, and the evidence behind its status.

    ``evidence`` names the structure that was checked, so a ``MISSING`` answer
    tells a researcher what to create or where else to look rather than only
    that something is absent.
    """

    location_id: str
    status: PanelMethodologyLocationStatus
    path: Path | None
    candidates: tuple[Path, ...]
    owner: str
    evidence: str

    def as_payload(self) -> dict[str, object]:
        """Serialize explicit source location status and its owner/evidence metadata.

        Returns:
            Location identifier, status, optional path, candidate paths and recorded owner/evidence.
        """
        return {
            "location_id": self.location_id,
            "status": self.status,
            "path": None if self.path is None else str(self.path),
            "candidates": [str(value) for value in self.candidates],
            "owner": self.owner,
            "evidence": self.evidence,
        }


def _workspaces_under(search_root: Path) -> tuple[Path, ...]:
    """A workspace is a directory that publishes an ``artifacts`` child."""

    if not search_root.is_dir():
        return ()

    def qualifies(candidate: Path) -> bool:
        return (candidate / _ARTIFACT_DIRECTORY).is_dir() or (
            candidate / _BASELINE_REPORT
        ).is_file()

    found = [search_root] if qualifies(search_root) else []
    found.extend(
        child for child in sorted(search_root.iterdir()) if child.is_dir() and qualifies(child)
    )
    return tuple(dict.fromkeys(found))


def _panel_stores_resolvable_at_their_owner(
    artifact_roots: tuple[Path, ...],
) -> tuple[tuple[Path, ...], str]:
    """Accept a Panel root only if its own resolver finds one complete Panel.

    Holding a `feature-panel` directory is not the property that matters. The
    owner requires exactly one Panel whose derivation closure reads back, and a
    Panel published before the installed observation clock has no clock identity
    to read -- so a directory full of legacy Panels looks present and resolves to
    nothing. Asking the owner turns that from a false `RESOLVED` into a refusal
    that names the cause, and it opens no numerical surface.
    """

    from alphalattice.control.product_host.research_authoring.panel_methodology_sources import (
        PanelMethodologySourceRoots,
        resolve_corrected_panel_snapshot_hash,
    )

    resolvable: list[Path] = []
    present = 0
    for root in artifact_roots:
        if not (root / _FEATURE_PANEL_STORE).is_dir():
            continue
        present += 1
        probe = PanelMethodologySourceRoots(
            repository_root=root.parent,
            panel_artifact_root=root,
            legacy_panel_artifact_root=root,
            feature_artifact_root=root,
            failed_baseline_workspace=root.parent,
            sector_context_artifact_root=root,
        )
        try:
            resolve_corrected_panel_snapshot_hash(roots=probe)
        except Exception:
            continue
        resolvable.append(root)
    evidence = (
        f"a {_FEATURE_PANEL_STORE} store whose owner resolves exactly one Panel with a "
        f"complete derivation closure; {present} store(s) present, {len(resolvable)} resolvable"
    )
    return tuple(resolvable), evidence


def _feature_stores_readable_at_their_owner(
    artifact_roots: tuple[Path, ...],
) -> tuple[tuple[Path, ...], str]:
    """A Feature root serves two readers, and a `factor-research` store proves one.

    `FactorResearchArtifactStore` is rooted here, and so is the development
    methodology-surface reader that resolves a Panel's `relative_surface_hash`.
    Probing only for `factor-research` accepted a workspace that publishes no
    methodology surface: it reported `RESOLVED`, and preflight then refused
    minutes later with a message about the baseline graph rather than about the
    root that was actually wrong.
    """

    from alphalattice.foundation.feature_engine.panels.development_overlay import (
        DEVELOPMENT_METHODOLOGY_SURFACE_CATEGORY,
    )

    present = tuple(value for value in artifact_roots if (value / _FACTOR_RESEARCH_STORE).is_dir())
    readable = tuple(
        value for value in present if (value / DEVELOPMENT_METHODOLOGY_SURFACE_CATEGORY).is_dir()
    )
    evidence = (
        f"an artifacts directory holding both a {_FACTOR_RESEARCH_STORE} store and a "
        f"{DEVELOPMENT_METHODOLOGY_SURFACE_CATEGORY} store; "
        f"{len(present)} store(s) present, {len(readable)} readable"
    )
    return readable, evidence


def _sector_context_stores_readable_at_their_owner(
    artifact_roots: tuple[Path, ...],
) -> tuple[tuple[Path, ...], int, str]:
    """Ask `SectorContextStore` where it would read, and whether anything is there.

    This is a different root from the Feature one. Both used to be probed as "an
    artifacts directory holding a factor-research store", which made them
    interchangeable in the report and they are not: in this repository the
    methodology surface and the Sector context surfaces live in two different
    workspaces, and only one of the two candidates satisfies each reader.
    """

    from alphalattice.investment.sector_research.inputs.storage import SectorContextStore

    present = 0
    readable: list[Path] = []
    for root in artifact_roots:
        manifests = SectorContextStore(root).root / "manifests"
        if not manifests.is_dir():
            continue
        present += 1
        if any(manifests.glob("*.json")):
            readable.append(root)
    evidence = (
        "an artifacts directory whose SectorContextStore holds at least one published "
        f"Sector context manifest; {present} store(s) present, {len(readable)} readable"
    )
    return tuple(readable), present, evidence


def _execution_outcome_stores_readable_at_their_owner(
    artifact_roots: tuple[Path, ...],
) -> tuple[tuple[Path, ...], int, str]:
    """Recognize roots that the causal execution-outcome reader can open."""

    present = 0
    readable: list[Path] = []
    for root in artifact_roots:
        manifests = root / _EXECUTION_OUTCOME_MANIFESTS
        if not manifests.is_dir():
            continue
        present += 1
        if any(manifests.glob("*.json")):
            readable.append(root)
    evidence = (
        "an artifacts directory whose CausalExecutionOutcomeDevelopmentReader "
        "holds at least one immutable execution manifest; "
        f"{present} store(s) present, {len(readable)} readable"
    )
    return tuple(readable), present, evidence


def _decide(
    location_id: str,
    owner: str,
    evidence: str,
    candidates: tuple[Path, ...],
    *,
    present: int | None = None,
) -> PanelMethodologyLocationProbe:
    """Separate "nothing here to look at" from "its owner refused what is here".

    Collapsing the two reads as a claim that the artifact does not exist, when
    all that was established is that it is not under the roots that were
    searched. `present` counts structurally matching candidates; `candidates`
    counts the ones their owner accepted.
    """

    structural = len(candidates) if present is None else present
    status: PanelMethodologyLocationStatus
    path: Path | None
    if structural == 0:
        status, path = "NOT_FOUND_IN_SEARCH_SCOPE", None
    elif not candidates:
        status, path = "REJECTED_BY_OWNER", None
    elif len(candidates) == 1:
        status, path = "RESOLVED", candidates[0]
    else:
        status, path = "AMBIGUOUS", None
    return PanelMethodologyLocationProbe(
        location_id=location_id,
        status=status,
        path=path,
        candidates=candidates,
        owner=owner,
        evidence=evidence,
    )


def discover_panel_methodology_context(
    *,
    search_roots: tuple[Path, ...] = (),
    named_locations: Mapping[str, Path] | None = None,
) -> tuple[PanelMethodologyLocationProbe, ...]:
    """Probe every required location, and decide each one at its owner.

    A location an operator names is **not** exempt from its owner's probe.
    Naming a path supplies *where to look*, never *what is there*, so a named
    location becomes the search scope for that one id and the same owner
    predicate decides it.

    That distinction is what makes an externally published artifact verifiable
    at all. A Panel published by another owner routinely lives outside every
    root a Desk would think to walk, so no scoped search can reach it and no
    scoped search can establish its absence either. The operator can say where
    it is; the owner still has the last word on whether it is what it claims.
    """
    named = {key: value for key, value in dict(named_locations or {}).items()}
    unknown = sorted(key for key in named if key not in PANEL_METHODOLOGY_LOCATION_IDS)
    if unknown:
        raise PanelMethodologyContextError(
            "research_authoring.context_field_not_a_location:" + ",".join(unknown)
        )
    if not search_roots and not named:
        raise PanelMethodologyContextError("research_authoring.context_search_root_required")

    workspaces: list[Path] = []
    for root in search_roots:
        workspaces.extend(_workspaces_under(root.resolve()))
    ordered = tuple(dict.fromkeys(workspaces))
    artifact_roots = tuple(value / _ARTIFACT_DIRECTORY for value in ordered)

    def scope(location_id: str, discovered: tuple[Path, ...]) -> tuple[Path, ...]:
        override = named.get(location_id)
        return (override.resolve(),) if override is not None else discovered

    def holding(scoped: tuple[Path, ...], store: str) -> tuple[Path, ...]:
        return tuple(value for value in scoped if (value / store).is_dir())

    panel_scope = scope("panel_artifact_root", artifact_roots)
    feature_scope = scope("feature_artifact_root", artifact_roots)
    sector_scope = scope("sector_context_artifact_root", artifact_roots)
    execution_scope = scope("execution_outcome_artifact_root", artifact_roots)
    baseline_scope = scope("failed_baseline_workspace", ordered)
    panel_stores, panel_evidence = _panel_stores_resolvable_at_their_owner(panel_scope)
    feature_stores, feature_evidence = _feature_stores_readable_at_their_owner(feature_scope)
    sector_stores, sector_present, sector_evidence = _sector_context_stores_readable_at_their_owner(
        sector_scope
    )
    execution_stores, execution_present, execution_evidence = (
        _execution_outcome_stores_readable_at_their_owner(execution_scope)
    )
    return (
        _decide(
            "repository_root",
            "alpha_research.DynamicPanelAlphaDevelopmentService",
            "a workspace publishing an artifacts directory",
            scope("repository_root", ordered),
        ),
        _decide(
            "panel_artifact_root",
            "alpha_research.resolve_corrected_panel_snapshot_hash",
            panel_evidence,
            panel_stores,
            present=len(holding(panel_scope, _FEATURE_PANEL_STORE)),
        ),
        _decide(
            "legacy_panel_artifact_root",
            "alpha_research.DynamicPanelAlphaDevelopmentService",
            f"an artifacts directory holding a {_FEATURE_PANEL_STORE} store; the legacy "
            "baseline is a different role from the corrected Panel and is not required "
            "to resolve as one",
            holding(scope("legacy_panel_artifact_root", artifact_roots), _FEATURE_PANEL_STORE),
        ),
        _decide(
            "feature_artifact_root",
            "feature_engine.load_development_methodology_surface_manifest",
            feature_evidence,
            feature_stores,
            present=len(holding(feature_scope, _FACTOR_RESEARCH_STORE)),
        ),
        _decide(
            "failed_baseline_workspace",
            "alpha_research.DynamicPanelArtifactStore",
            f"a workspace holding {_BASELINE_REPORT.as_posix()}",
            tuple(value for value in baseline_scope if (value / _BASELINE_REPORT).is_file()),
        ),
        _decide(
            "sector_context_artifact_root",
            "sector_research.SectorContextStore",
            sector_evidence,
            sector_stores,
            present=sector_present,
        ),
        _decide(
            "execution_outcome_artifact_root",
            "causal_outcomes.CausalExecutionOutcomeDevelopmentReader",
            execution_evidence,
            execution_stores,
            present=execution_present,
        ),
    )


def render_panel_methodology_context(
    probes: tuple[PanelMethodologyLocationProbe, ...],
    *,
    searched_roots: tuple[Path, ...] = (),
) -> dict[str, object]:
    """The printable report, including a manifest when every location resolved."""
    unresolved = tuple(value for value in probes if value.status != "RESOLVED")
    manifest: dict[str, str] | None = (
        None
        if unresolved
        else {value.location_id: str(value.path) for value in probes if value.path is not None}
    )
    return {
        "command": "print-context",
        # A refusal below establishes only that the location is not under these
        # roots. Artifacts published by other owners routinely live outside a
        # Desk's own workspace, so absence here is never evidence of absence.
        "searched_roots": [str(value) for value in searched_roots],
        "locations": [value.as_payload() for value in probes],
        "manifest": manifest,
        "refusal": (
            None
            if manifest is not None
            else "research_authoring.context_not_resolved:"
            + ",".join(sorted(f"{value.location_id}={value.status}" for value in unresolved))
        ),
        "numerical_calls": 0,
    }


def load_panel_methodology_context_manifest(path: Path) -> dict[str, Path]:
    """Read an operational routing manifest. It carries locations, never identity.

    A manifest that names an identity, a hash, or a publication pointer is
    refused: those belong to the artifacts a run actually opens, and admitting
    them here would let a routing file assert authority it never held.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PanelMethodologyContextError(
            "research_authoring.context_manifest_unreadable"
        ) from error
    if not isinstance(payload, dict):
        raise PanelMethodologyContextError("research_authoring.context_manifest_invalid")
    entries = payload.get("manifest", payload)
    if not isinstance(entries, dict) or not entries:
        raise PanelMethodologyContextError("research_authoring.context_manifest_invalid")
    admitted = set(PANEL_METHODOLOGY_LOCATION_IDS) | _OPTIONAL_LOCATION_IDS
    unknown = sorted(str(key) for key in entries if str(key) not in admitted)
    if unknown:
        raise PanelMethodologyContextError(
            "research_authoring.context_manifest_field_not_a_location:" + ",".join(unknown)
        )
    missing = sorted(value for value in PANEL_METHODOLOGY_LOCATION_IDS if value not in entries)
    if missing:
        raise PanelMethodologyContextError(
            "research_authoring.context_manifest_incomplete:" + ",".join(missing)
        )
    resolved: dict[str, Path] = {}
    for key, value in entries.items():
        if not isinstance(value, str) or not value:
            raise PanelMethodologyContextError("research_authoring.context_manifest_invalid")
        candidate = Path(value)
        if not candidate.exists():
            raise PanelMethodologyContextError(
                f"research_authoring.context_manifest_location_absent:{key}"
            )
        resolved[str(key)] = candidate
    return resolved
