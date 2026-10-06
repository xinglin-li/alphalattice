"""The universe-centred robust Z: the Panel kernel with the whole universe as its one group (V346).

A quantity whose Sector level is structural (liquidity, turnover) takes the Sector demean
(`ROBUST_SECTOR_NEUTRAL_Z`); a return, a price move or a move size has no Sector level to remove,
and centring it on its Sector would take out part of what it measures. This method runs the same
kernel -- the median/MAD winsor, the equal-weight demean and the global robust Z over each
session's members -- with every member in one group, so the demean is the universe's and the
Sector plays no part: no classification, and so no reclassification, reaches its values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from functools import lru_cache
from typing import Final

import pandas as pd

from alphalattice.foundation.feature_engine.contracts import FeaturePanelBinding
from alphalattice.foundation.feature_engine.producers.arithmetic_identity import (
    PREPROCESSING_WALK_EXCLUDED,
    feature_component_identity,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.contracts import (
    PanelPreprocessingImplementationBinding,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.robust_cross_section import (
    PanelCrossSectionKernel,
    PanelMaterialization,
)

ROBUST_UNIVERSE_Z_IMPLEMENTATION: Final = "robust_universe_z.v1"
UNIVERSE_GROUP: Final = "UNIVERSE"
"""The one group every member joins: the kernel's Sector axis holds it alone."""
_OWNERS: Final = (
    "alphalattice.foundation.feature_engine.producers.preprocessing.robust_universe",
    "alphalattice.foundation.feature_engine.producers.preprocessing.robust_cross_section",
    "alphalattice.kernel.quant.cross_section",
)


@lru_cache(maxsize=1)
def _content_hash() -> str:
    """The identity of the code that computes the universe-centred transformation."""
    identity: str = feature_component_identity(
        "PANEL_PREPROCESSING:robust_universe_z",
        owners=_OWNERS,
        excluded=PREPROCESSING_WALK_EXCLUDED,
        semantic_owner="feature_engine.producers.preprocessing",
        numerical_role="PANEL_PREPROCESSING",
    )
    return identity


class RobustUniverseZAdapter:
    """Run the robust cross-section kernel with the universe as the one group."""

    def __init__(self) -> None:
        """Create the robust cross-section numerical kernel it runs."""
        self._kernel = PanelCrossSectionKernel()

    @property
    def implementation_id(self) -> str:
        """Return this adapter's stable installed executable handle."""
        return ROBUST_UNIVERSE_Z_IMPLEMENTATION

    def describe_implementation_binding(self) -> PanelPreprocessingImplementationBinding:
        """Describe this executable by its measured implementation content.

        Returns:
            Canonical handle, owners, and content binding for the code this adapter runs.
        """
        return PanelPreprocessingImplementationBinding.create(
            implementation_id=ROBUST_UNIVERSE_Z_IMPLEMENTATION,
            implementation_owners=_OWNERS,
            implementation_content_hash=_content_hash(),
        )

    def materialize(
        self,
        *,
        feature_rows: list[dict[str, object]] | pd.DataFrame,
        active_listing_ids: tuple[str, ...],
        sector_by_listing_id: Mapping[str, str],
        factor_ids: tuple[str, ...],
        binding: FeaturePanelBinding,
        members_by_session: Mapping[date, Sequence[str]] | None = None,
        source_exclusions_by_session: Mapping[date, Sequence[str]] | None = None,
    ) -> PanelMaterialization:
        """Transform each session's members centred on the universe, not on their Sectors.

        Args:
            feature_rows: Listing-session source feature values.
            active_listing_ids: Declared calculation axis in execution order.
            sector_by_listing_id: Protocol argument; no Sector enters this method.
            factor_ids: Ordered factor scope to transform.
            binding: Panel lineage and recorded numerical policy authority.
            members_by_session: Nominal session members, or the full uniform axis when omitted.
            source_exclusions_by_session: Evidenced exclusions by session, if present.

        Returns:
            Transformed rows, availability and admission, measured clipping observations,
                and ordered input/output identities bound to the supplied Panel lineage.

        Raises:
            KeyError: A required listing, session, or factor source column is absent.
            ValueError: Rows are empty or duplicated, axes are invalid, temporal membership is
                incomplete, or source eligibility contradicts the recorded policy.
        """
        del sector_by_listing_id
        return self._kernel.materialize(
            feature_rows=feature_rows,
            active_listing_ids=active_listing_ids,
            sector_by_listing_id=dict.fromkeys(active_listing_ids, UNIVERSE_GROUP),
            factor_ids=factor_ids,
            binding=binding,
            members_by_session=members_by_session,
            source_exclusions_by_session=source_exclusions_by_session,
        )


__all__ = [
    "ROBUST_UNIVERSE_Z_IMPLEMENTATION",
    "UNIVERSE_GROUP",
    "RobustUniverseZAdapter",
]
