"""Install the extension factor declared for development workspace tests."""

from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.producers.factors.open_intraday import (
    overnight_return_factor_spec,
)


def development_feature_catalog() -> FeatureCatalog:
    """The shipped catalog plus the one persistent extension under test.

    This is the whole of "a researcher installs a method": take the governed
    catalog, add the typed recipe the Feature capability already declares, and
    hand the revision to the writers. The mathematics, the recipe and its
    implementation identity are all product-owned -- ``overnight_return_factor_spec``
    lives beside its kernel in ``feature_engine/producers``. Nothing is invented
    here, and a spec assembled here instead would make "add a method in the
    domain directory" false.

    ``from_payload`` requires the factor axis sorted, so the merge sorts rather
    than appends: the axis is positional in everything downstream that quotes it.

    The shipped ``desktop-feature-catalog.json`` is not written to. This revision
    exists only in the workspace that installs it, which is why no published
    manifest and no current pointer moves.
    """

    shipped = FeatureCatalog.load()
    payload = shipped.to_payload()
    payload["factors"] = sorted(
        [*payload["factors"], overnight_return_factor_spec().model_dump(mode="json")],
        key=lambda factor: str(factor["factor_id"]),
    )
    return FeatureCatalog.from_payload(payload)
