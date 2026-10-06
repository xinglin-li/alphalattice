"""A Feature method a researcher adds reaches a real published panel.

Registration was already proven: ``test_feature_factor_authoring`` shows a kernel
installs into ``FeatureKernelRegistry`` and an unregistered id is refused. That is
necessary and nowhere near sufficient. A registry entry nothing materializes is a
lookup table, and the claim that matters to a researcher is the other one: *the
method computes, lands in a panel, and the panel records which code produced it.*

So this drives the real path -- onboarding, genesis, the maintenance coordinator,
``FeatureFoundationService`` and ``FeaturePanelSnapshotPublisher`` -- over a
workspace composed with a catalog revision naming ``overnight_return``. No fixture
materializer and no ``DirectFixture`` bypass; the only fixture is the provider,
which is the network edge.

Two cases, not six. Every claim below is an assertion, but they divide into
exactly two questions -- did the new method run, and did installing it disturb
anything -- so they are grouped that way rather than one per field.

What is deliberately *not* claimed: this factor is not admitted to production. The
shipped catalog does not name it. "Developable, registrable, runnable" is the
claim; "current" is a separate authority nothing here touches.
"""

from __future__ import annotations

from typing import Any

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.catalog.contracts import (
    FeatureCatalog,
    desktop_core_feature_bundle,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
    extension_factor_specs,
)
from alphalattice.foundation.feature_engine.producers.factors.open_intraday import (
    OVERNIGHT_RETURN_FACTOR_ID,
    OVERNIGHT_RETURN_ID,
    overnight_return_factor_spec,
)
from tests.researcher_methodology_surface.real_workspace import (
    RealRiskWorkspace,
    development_feature_catalog,
)


def _factor_summary(workspace: RealRiskWorkspace) -> dict[str, Any]:
    """Read the published manifest exactly as every consumer reads it."""

    payload = ArtifactResolver(workspace.artifact_root).load_feature_panel_manifest(
        workspace.panel_manifest_ref
    )
    summary = payload["safe_summary"]
    assert isinstance(summary, dict)
    factors = summary["factor_catalog_summary"]
    assert isinstance(factors, dict)
    return factors


def test_the_extension_factor_materializes_through_the_real_service(
    extension_feature_workspace: RealRiskWorkspace,
) -> None:
    """The decisive case: the method computed, and the panel says which code did.

    The availability summary is derived from what was **materialized**, not from
    what the catalog declared, so a factor whose kernel never ran would appear
    with no available sessions and a null ratio of one. That is what makes this a
    materialization proof rather than a restatement of the catalog.

    The recipe is asserted to be product-owned in the same place, because the two
    are one claim: a ``FactorSpec`` assembled in a fixture would make "add a
    method in the domain directory" false -- the mathematics would live in the
    product while the recipe deciding its window, lag and inputs lived in a test.
    """

    spec = overnight_return_factor_spec()
    kernel = default_extension_kernel_registry().resolve(spec.formula_ref)

    assert spec in extension_factor_specs()
    assert spec.formula_ref == OVERNIGHT_RETURN_ID
    # The registry refuses a spec whose declared inputs differ from its kernel's,
    # so this equality is what lets recipe and mathematics be reviewed apart
    # without drifting.
    assert spec.required_fields == tuple(sorted(kernel.required_fields))

    entry = _factor_summary(extension_feature_workspace)[OVERNIGHT_RETURN_FACTOR_ID]

    # It carries the identity of the code that computed it, not merely its id: a
    # catalog binding only factor ids lets an implementation change under a
    # stable id while old evidence still looks reusable.
    #
    # Compared against the identity the registry derives, not the summary the
    # kernel declares. Since Factor identity was bound to the measured source
    # bytes, the declared summary is one input to that identity rather than the
    # identity itself -- asserting the declared value here would re-assert the
    # property that binding removed.
    registry = default_extension_kernel_registry()
    assert entry["implementation_hash"] == str(
        registry.implementation_hash(spec, core_bundle=desktop_core_feature_bundle())
    )
    assert entry["implementation_hash"] != kernel.implementation_hash
    # And it genuinely computed.
    assert int(entry["available_session_count"]) > 0
    assert float(entry["null_ratio"]) < 1.0


def test_installing_the_method_disturbs_no_shipped_identity(
    real_risk_workspace: RealRiskWorkspace,
    extension_feature_workspace: RealRiskWorkspace,
) -> None:
    """Readback compatibility, stated as a comparison between two real panels.

    The shared workspace is built from the shipped catalog and knows nothing about
    the extension. Its manifest must not mention the new factor, and every factor
    it does carry must report the same implementation identity in both panels --
    otherwise installing a method would silently restate the identity of methods
    nobody touched, and every panel published before this one would stop being
    comparable.
    """

    shipped = FeatureCatalog.load()
    development = development_feature_catalog()

    assert OVERNIGHT_RETURN_FACTOR_ID not in shipped.factor_ids
    assert OVERNIGHT_RETURN_FACTOR_ID in development.factor_ids
    assert development.binding.catalog_content_hash != shipped.binding.catalog_content_hash
    # The revision was composed in memory, not written back over the governed
    # resource. Without this the failure would stay invisible until some
    # unrelated build published a moved panel binding.
    reloaded = FeatureCatalog.load().binding.catalog_content_hash
    assert reloaded == shipped.binding.catalog_content_hash
    # Installing one factor is not an edit to the other fifty-three.
    for factor_id in shipped.factor_ids:
        assert development.contracts_by_factor[factor_id] == shipped.contracts_by_factor[factor_id]

    shipped_summary = _factor_summary(real_risk_workspace)
    extended_summary = _factor_summary(extension_feature_workspace)

    assert set(extended_summary) - set(shipped_summary) == {OVERNIGHT_RETURN_FACTOR_ID}
    for factor_id, entry in shipped_summary.items():
        assert entry["implementation_hash"] == extended_summary[factor_id]["implementation_hash"], (
            factor_id
        )
