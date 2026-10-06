"""A bounded real Risk authority and a Risk development program binding over
the real workspace.

The package's session fixture and the Risk development suite both construct
them; the fixture used to import them from the suite.
Test support beside its owner; nothing here is product authority or evidence.
"""

from __future__ import annotations

from pathlib import Path

from alphalattice.investment.risk_research.contracts import (
    default_covariance_recipe,
)
from alphalattice.investment.risk_research.estimators.catalog import (
    build_installed_risk_estimator_catalog,
)
from alphalattice.investment.risk_research.estimators.domains import (
    COVARIANCE_PARAMETER_DOMAIN,
)
from alphalattice.investment.risk_research.experiments.contracts import (
    RiskDevelopmentProgramBinding,
)
from alphalattice.investment.risk_research.experiments.window import (
    REQUIRED_LOOKBACK_SESSIONS,
)
from alphalattice.protocols.research_authoring.contracts import (
    ResolvedResearchAuthority,
)
from tests.researcher_methodology_surface.real_workspace import RealRiskWorkspace

PLAYPEN_ROOT = Path(__file__).resolve().parents[2]


def _bounded_authority(
    workspace: RealRiskWorkspace, *, count: int
) -> tuple[ResolvedResearchAuthority, tuple[object, ...]]:
    """A real authority over a deliberately small slice of the real surface.

    The first admissible formation sits at index ``REQUIRED_LOOKBACK_SESSIONS``,
    so requesting from there gives a window the numerical path can actually
    serve while staying far smaller than the surface.
    """

    available = workspace.return_reader.available_sessions(workspace.return_surface)
    requested = available[REQUIRED_LOOKBACK_SESSIONS : REQUIRED_LOOKBACK_SESSIONS + count]
    authority = ResolvedResearchAuthority.create(
        data_snapshot_handle="current",
        universe_handle="us-current-index-research",
        panel_snapshot_hash=workspace.panel_snapshot_hash,
        panel_manifest_ref=workspace.panel_manifest_ref,
        universe_revision_sha256=workspace.manifest.revision_sha256,
        ordered_listing_ids=tuple(workspace.return_surface.epoch.ordered_listing_ids),
        sessions=requested,
        # The authority's own watermark scope. It is bound into identity and is
        # deliberately never compared with the surface's.
        source_watermark_hash="0" * 64,
    )
    return authority, requested


def _binding(**overrides: object) -> RiskDevelopmentProgramBinding:
    catalog = build_installed_risk_estimator_catalog().binding
    selected = catalog.ordered_capabilities[0]
    values: dict[str, object] = {
        "catalog_hash": catalog.catalog_hash,
        "selected_adapter_id": selected.adapter_id,
        "selected_numerical_binding_hash": selected.numerical_binding_hash,
        "recipe_hash": default_covariance_recipe().recipe_hash,
        "parameter_domain_hash": COVARIANCE_PARAMETER_DOMAIN.domain_hash,
    }
    values.update(overrides)
    return RiskDevelopmentProgramBinding.create(**values)  # type: ignore[arg-type]
