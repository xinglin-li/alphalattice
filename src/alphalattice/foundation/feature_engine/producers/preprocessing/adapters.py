"""The numerical implementations an installed preprocessing recipe may run.

A catalog that names methods but cannot say which code computes them is
metadata, not authority: a build could seal a new recipe's identity and still
execute the previous kernel. The adapter is the missing edge -- one narrow
Protocol, one implementation per installed recipe, resolved through the catalog
so the method that was named is the method that runs.

Deliberately not a plugin framework: no discovery, no registry, no dynamic
import. Adding an implementation is an entry in the installed catalog.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from functools import lru_cache
from typing import Protocol

import pandas as pd

from alphalattice.foundation.feature_engine.contracts import FeaturePanelBinding
from alphalattice.foundation.feature_engine.producers.arithmetic_identity import (
    PREPROCESSING_WALK_EXCLUDED,
    feature_component_identity,
)
from alphalattice.foundation.feature_engine.producers.preprocessing import (
    development,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.contracts import (
    PanelPreprocessingError,
    PanelPreprocessingImplementationBinding,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.robust_cross_section import (
    PanelCrossSectionKernel,
    PanelMaterialization,
)

ROBUST_SECTOR_NEUTRAL_Z_IMPLEMENTATION = (
    "feature_engine.producers.preprocessing.robust_cross_section.PanelCrossSectionKernel"
)
TIME_SERIES_ABSOLUTE_STATE_ROBUST_IMPLEMENTATION = (
    "feature_engine.producers.preprocessing.development.TimeSeriesAbsoluteStateRobustAdapter"
)
STATE_INTERACTION_BLOCK_IMPLEMENTATION = (
    "feature_engine.producers.preprocessing.development.StateInteractionBlockAdapter"
)

_ROBUST_SECTOR_NEUTRAL_Z_OWNERS = (
    "alphalattice.foundation.feature_engine.producers.preprocessing.robust_cross_section",
    "alphalattice.kernel.quant.cross_section",
    "alphalattice.foundation.feature_engine.producers.preprocessing.adapters",
)
_DEVELOPMENT_PREPROCESSING_OWNERS = (
    "alphalattice.foundation.feature_engine.producers.preprocessing.development",
    "alphalattice.foundation.feature_engine.producers.preprocessing.adapters",
)


@lru_cache(maxsize=1)
def _robust_sector_neutral_z_content_hash() -> str:
    """The identity of the code that computes the robust sector-neutral transformation.

    Its rule closure from the Panel kernel, the shared mathematics it runs and this adapter,
    kept at its byte value while its rule is the one the switch recorded (V70).
    """
    return feature_component_identity(
        "PANEL_PREPROCESSING:robust_sector_neutral_z",
        owners=(
            "alphalattice.foundation.feature_engine.producers.preprocessing.robust_cross_section",
            "alphalattice.kernel.quant.cross_section",
            "alphalattice.foundation.feature_engine.producers.preprocessing.adapters",
        ),
        excluded=PREPROCESSING_WALK_EXCLUDED,
        semantic_owner="feature_engine.producers.preprocessing",
        numerical_role="PANEL_PREPROCESSING",
    )


@lru_cache(maxsize=1)
def _development_preprocessing_content_hash() -> str:
    """The identity of the development preprocessing, with the Panel kernel it runs (V232)."""
    return feature_component_identity(
        "DEVELOPMENT_PANEL_PREPROCESSING:development",
        owners=(
            "alphalattice.foundation.feature_engine.producers.preprocessing.development",
            "alphalattice.foundation.feature_engine.producers.preprocessing.adapters",
        ),
        excluded=PREPROCESSING_WALK_EXCLUDED,
        semantic_owner="feature_engine.producers.preprocessing",
        numerical_role="DEVELOPMENT_PANEL_PREPROCESSING",
    )


class PanelPreprocessingAdapter(Protocol):
    """What every installed preprocessing implementation must offer.

    Exactly the call the Panel build already makes, so an implementation is a
    numerical owner rather than a lifecycle to manage.
    """

    @property
    def implementation_id(self) -> str:
        """Return the stable installed handle of this numerical implementation."""
        ...

    def describe_implementation_binding(self) -> PanelPreprocessingImplementationBinding:
        """The content identity of this implementation.

        Required of every adapter, because a catalog that can only report the
        name an implementation gives itself cannot tell a rewrite from a rename.
        """
        ...

    def materialize(
        self,
        *,
        feature_rows: list[dict[str, object]] | pd.DataFrame,
        active_listing_ids: tuple[str, ...],
        sector_by_listing_id: dict[str, str],
        factor_ids: tuple[str, ...],
        binding: FeaturePanelBinding,
        members_by_session: Mapping[date, Sequence[str]] | None = None,
        source_exclusions_by_session: Mapping[date, Sequence[str]] | None = None,
    ) -> PanelMaterialization:
        """Transform the rows; ``members_by_session`` names each session's members.

        Every implementation accepts the argument. One that transforms a
        dense grid only refuses a membership that is not the whole axis on
        every session, by name, rather than computing over rows that are not
        in the cross-section.

        Transform rows under the installed implementation's membership contract.

        Args:
            feature_rows: Listing-session source feature values.
            active_listing_ids: Declared calculation axis in execution order.
            sector_by_listing_id: Sector authority for every axis listing.
            factor_ids: Ordered factor scope to transform.
            binding: Panel lineage and recorded numerical policy authority.
            members_by_session: Nominal session members, or the full uniform axis when omitted.
            source_exclusions_by_session: Evidenced exclusions by session, if present.

        Returns:
            Transformed rows, availability and admission, measured clipping observations,
                and ordered input/output identities bound to the supplied Panel lineage.

        Raises:
            KeyError: A required listing, session, or factor source column is absent.
            ValueError: Inputs violate the resolved implementation's materialization contract.
            PanelPreprocessingError: A dense implementation cannot honor the declared membership.
        """
        ...


def require_uniform_membership(
    members_by_session: Mapping[date, Sequence[str]] | None,
    active_listing_ids: tuple[str, ...],
    source_exclusions_by_session: Mapping[date, Sequence[str]] | None = None,
) -> None:
    """Refuse per-session membership an implementation cannot honour.

    Args:
        members_by_session: Per-session members, or no temporal declaration for a dense axis.
        active_listing_ids: Ordered calculation axis every declared session must hold.
        source_exclusions_by_session: Source exclusions, which this dense
            implementation cannot honor.

    Raises:
        PanelPreprocessingError: Any source exclusion is nonempty, or a session's ordered
            members differ from the full calculation axis.
    """
    if source_exclusions_by_session and any(source_exclusions_by_session.values()):
        raise PanelPreprocessingError("PANEL_PREPROCESSING_SOURCE_ELIGIBILITY_NOT_SUPPORTED")
    if members_by_session is None:
        return
    axis = tuple(active_listing_ids)
    if any(tuple(members) != axis for members in members_by_session.values()):
        raise PanelPreprocessingError(
            "PANEL_PREPROCESSING_IMPLEMENTATION_REQUIRES_UNIFORM_MEMBERSHIP"
        )


class RobustSectorNeutralZAdapter:
    """Adapt the installed robust sector-neutral kernel to the preprocessing contract."""

    @property
    def implementation_id(self) -> str:
        """Return this adapter's stable installed executable handle."""
        return ROBUST_SECTOR_NEUTRAL_Z_IMPLEMENTATION

    def describe_implementation_binding(self) -> PanelPreprocessingImplementationBinding:
        """Describe this executable by its measured implementation content.

        Returns:
            Canonical handle, owner, and content binding for the code this adapter runs.
        """
        return PanelPreprocessingImplementationBinding.create(
            implementation_id=ROBUST_SECTOR_NEUTRAL_Z_IMPLEMENTATION,
            implementation_owners=_ROBUST_SECTOR_NEUTRAL_Z_OWNERS,
            implementation_content_hash=_robust_sector_neutral_z_content_hash(),
        )

    def __init__(self) -> None:
        """Create the explicit robust cross-section numerical kernel."""
        self._kernel = PanelCrossSectionKernel()

    def materialize(
        self,
        *,
        feature_rows: list[dict[str, object]] | pd.DataFrame,
        active_listing_ids: tuple[str, ...],
        sector_by_listing_id: dict[str, str],
        factor_ids: tuple[str, ...],
        binding: FeaturePanelBinding,
        members_by_session: Mapping[date, Sequence[str]] | None = None,
        source_exclusions_by_session: Mapping[date, Sequence[str]] | None = None,
    ) -> PanelMaterialization:
        """Transform each session through the robust sector-neutral numerical kernel.

        Args:
            feature_rows: Listing-session source feature values.
            active_listing_ids: Declared calculation axis in execution order.
            sector_by_listing_id: Sector authority for every axis listing.
            factor_ids: Ordered factor scope to transform.
            binding: Panel lineage and recorded numerical policy authority.
            members_by_session: Nominal session members, or the full uniform axis when omitted.
            source_exclusions_by_session: Evidenced exclusions by session, if present.

        Returns:
            Transformed rows, availability and admission, measured clipping observations,
                and ordered input/output identities bound to the supplied Panel lineage.

        Raises:
            KeyError: A required listing, session, or factor source column is absent.
            ValueError: Rows are empty or duplicated, axes or sectors are invalid, temporal
                membership is incomplete, or source eligibility contradicts the recorded policy.
        """
        return self._kernel.materialize(
            feature_rows=feature_rows,
            active_listing_ids=active_listing_ids,
            sector_by_listing_id=sector_by_listing_id,
            factor_ids=factor_ids,
            binding=binding,
            members_by_session=members_by_session,
            source_exclusions_by_session=source_exclusions_by_session,
        )


class TimeSeriesAbsoluteStateRobustAdapter:
    """Apply trailing listing median/MAD scaling to a uniform development Panel.

    Sector labels do not enter this time-series transformation.
    """

    @property
    def implementation_id(self) -> str:
        """Return this adapter's stable installed executable handle."""
        return TIME_SERIES_ABSOLUTE_STATE_ROBUST_IMPLEMENTATION

    def describe_implementation_binding(self) -> PanelPreprocessingImplementationBinding:
        """Describe this executable by its measured implementation content.

        Returns:
            Canonical handle, owner, and content binding for the code this adapter runs.
        """
        return PanelPreprocessingImplementationBinding.create(
            implementation_id=self.implementation_id,
            implementation_owners=_DEVELOPMENT_PREPROCESSING_OWNERS,
            implementation_content_hash=_development_preprocessing_content_hash(),
        )

    def materialize(
        self,
        *,
        feature_rows: list[dict[str, object]] | pd.DataFrame,
        active_listing_ids: tuple[str, ...],
        sector_by_listing_id: dict[str, str],
        factor_ids: tuple[str, ...],
        binding: FeaturePanelBinding,
        members_by_session: Mapping[date, Sequence[str]] | None = None,
        source_exclusions_by_session: Mapping[date, Sequence[str]] | None = None,
    ) -> PanelMaterialization:
        """Transform uniform listing histories by trailing median/MAD scaling.

        Args:
            feature_rows: Listing-session source feature values.
            active_listing_ids: Declared calculation axis in execution order.
            sector_by_listing_id: Protocol argument, unused by the listing time-series method.
            factor_ids: Ordered factor scope to transform.
            binding: Panel lineage and recorded numerical policy authority.
            members_by_session: Nominal session members, or the full uniform axis when omitted.
            source_exclusions_by_session: Evidenced exclusions by session, if present.

        Returns:
            Transformed rows, availability and admission, measured clipping observations,
                and ordered input/output identities bound to the supplied Panel lineage.

        Raises:
            KeyError: A required listing, session, or factor source column is absent.
            ValueError: Source rows, factor or sector axes, or required inputs are invalid.
            PanelPreprocessingError: Membership is nonuniform or source exclusions are nonempty.
        """
        del sector_by_listing_id
        require_uniform_membership(
            members_by_session, active_listing_ids, source_exclusions_by_session
        )
        return development.materialize_absolute_state(
            feature_rows=feature_rows,
            active_listing_ids=active_listing_ids,
            factor_ids=factor_ids,
            binding=binding,
        )


class StateInteractionBlockAdapter:
    """Combine robust sector-neutral stock values with supplied state values on a uniform Panel."""

    @property
    def implementation_id(self) -> str:
        """Return this adapter's stable installed executable handle."""
        return STATE_INTERACTION_BLOCK_IMPLEMENTATION

    def describe_implementation_binding(self) -> PanelPreprocessingImplementationBinding:
        """Describe this executable by its measured implementation content.

        Returns:
            Canonical handle, owner, and content binding for the code this adapter runs.
        """
        return PanelPreprocessingImplementationBinding.create(
            implementation_id=self.implementation_id,
            implementation_owners=_DEVELOPMENT_PREPROCESSING_OWNERS,
            implementation_content_hash=_development_preprocessing_content_hash(),
        )

    def materialize(
        self,
        *,
        feature_rows: list[dict[str, object]] | pd.DataFrame,
        active_listing_ids: tuple[str, ...],
        sector_by_listing_id: dict[str, str],
        factor_ids: tuple[str, ...],
        binding: FeaturePanelBinding,
        members_by_session: Mapping[date, Sequence[str]] | None = None,
        source_exclusions_by_session: Mapping[date, Sequence[str]] | None = None,
    ) -> PanelMaterialization:
        """Multiply robust stock values by their state children and clip the interaction.

        Args:
            feature_rows: Listing-session source feature values.
            active_listing_ids: Declared calculation axis in execution order.
            sector_by_listing_id: Sector authority for every axis listing.
            factor_ids: Ordered factor scope to transform.
            binding: Panel lineage and recorded numerical policy authority.
            members_by_session: Nominal session members, or the full uniform axis when omitted.
            source_exclusions_by_session: Evidenced exclusions by session, if present.

        Returns:
            Transformed rows, availability and admission, measured clipping observations,
                and ordered input/output identities bound to the supplied Panel lineage.

        Raises:
            KeyError: A required listing, session, or factor source column is absent.
            ValueError: Source rows, factor or sector axes, or required inputs are invalid.
            PanelPreprocessingError: Membership is nonuniform or source exclusions are nonempty.
        """
        require_uniform_membership(
            members_by_session, active_listing_ids, source_exclusions_by_session
        )
        return development.materialize_state_interactions(
            feature_rows=feature_rows,
            active_listing_ids=active_listing_ids,
            sector_by_listing_id=sector_by_listing_id,
            factor_ids=factor_ids,
            binding=binding,
        )


__all__ = [
    "ROBUST_SECTOR_NEUTRAL_Z_IMPLEMENTATION",
    "STATE_INTERACTION_BLOCK_IMPLEMENTATION",
    "TIME_SERIES_ABSOLUTE_STATE_ROBUST_IMPLEMENTATION",
    "PanelPreprocessingAdapter",
    "RobustSectorNeutralZAdapter",
    "StateInteractionBlockAdapter",
    "TimeSeriesAbsoluteStateRobustAdapter",
    "require_uniform_membership",
]
