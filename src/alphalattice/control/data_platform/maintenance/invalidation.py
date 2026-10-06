"""Domain-owned source-to-feature invalidation topology."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.contracts import FeatureInvalidation, canonical_hash

from .contracts import MarketDataChangeSet


@dataclass(frozen=True)
class FeatureInvalidationPlan:
    """Seal dependency-ordered invalidation effects against one market-data change identity.

    Attributes:
        ordered_stages: Installed stage order for dependency propagation.
        invalidations: Deduplicated sorted invalidation records.
        market_dependent_factor_ids: Factors depending on benchmark/market evidence.
        source_change_set_hash: Observed input change identity.
        plan_hash: Canonical invalidation plan identity.
    """

    ordered_stages: tuple[str, ...]
    invalidations: tuple[FeatureInvalidation, ...]
    market_dependent_factor_ids: tuple[str, ...]
    source_change_set_hash: str
    plan_hash: str


class FeatureInvalidationTopology:
    """A fixed source-field topology, not a general graph framework."""

    _ORDER = (
        "provider_series_delta",
        "base_feature",
        "market_dependent_base_feature",
        "sector_neutral_panel",
        "active_panel_binding",
    )

    def __init__(self, catalog: FeatureCatalog) -> None:
        """Resolve invalidation dependencies from the installed Feature catalog.

        Args:
            catalog: Installed Feature catalog defining factor input dependencies.
        """
        self.catalog = catalog

    def _factors_for_fields(self, fields: set[str]) -> tuple[str, ...]:
        return tuple(
            sorted(
                contract.factor_id
                for contract in self.catalog.maintenance_contracts
                if fields.intersection(contract.required_fields)
            )
        )

    def plan(
        self,
        change_set: MarketDataChangeSet,
        *,
        history_start: date,
        as_of_session: date,
    ) -> FeatureInvalidationPlan:
        """Derive ordered invalidation effects from actual input and membership changes.

        Args:
            change_set: Observed market-data/membership changes to propagate.
            history_start: First market session in the bounded history.
            as_of_session: Effective market session for membership changes and invalidation.

        Returns:
            The sealed invalidation plan, preserving unaffected historical cross-sections.

        Raises:
            ValueError: The history range is reversed.
        """
        if history_start > as_of_session:
            raise ValueError("feature invalidation history range is reversed")
        items: list[FeatureInvalidation] = []
        all_factors = tuple(sorted(self.catalog.factor_ids))
        adjusted_factors = self._factors_for_fields({"provider_adjusted_close"})
        raw_fields = {
            "open_raw",
            "high_raw",
            "low_raw",
            "close_raw",
            "volume_raw",
            "open_split_adjusted",
            "high_split_adjusted",
            "low_split_adjusted",
            "close_split_adjusted",
        }
        raw_factors = self._factors_for_fields(raw_fields)
        for change in change_set.listing_changes:
            receipt = change.source_receipt_hashes[0] if change.source_receipt_hashes else None
            if change.new_session_start is not None:
                new_sessions = change.new_sessions or (change.new_session_start,)
                items.append(
                    FeatureInvalidation(
                        "normal_new_session",
                        listing_id=change.listing_id,
                        earliest_session=change.new_session_start,
                        affected_sessions=new_sessions,
                        factor_ids=all_factors,
                        source_receipt_hash=receipt,
                    )
                )
            raw_sessions = change.raw_correction_sessions or (
                (change.raw_correction_start,) if change.raw_correction_start else ()
            )
            if raw_sessions:
                items.append(
                    FeatureInvalidation(
                        "ohlc_correction",
                        listing_id=change.listing_id,
                        earliest_session=min(raw_sessions),
                        affected_sessions=tuple(sorted(set(raw_sessions))),
                        source_fields=tuple(sorted(raw_fields)),
                        factor_ids=raw_factors,
                        source_receipt_hash=receipt,
                    )
                )
            if change.adjusted_return_change_sessions:
                items.append(
                    FeatureInvalidation(
                        "adjusted_return_correction",
                        listing_id=change.listing_id,
                        earliest_session=min(change.adjusted_return_change_sessions),
                        affected_sessions=change.adjusted_return_change_sessions,
                        source_fields=("provider_adjusted_close",),
                        factor_ids=adjusted_factors,
                        source_receipt_hash=receipt,
                    )
                )
            # Corporate-action evidence is intentionally absent here.  It is
            # provenance unless provider OHLCV or adjusted returns also changed.
        if change_set.spy_correction_start is not None:
            spy_sessions = change_set.spy_return_change_sessions or (
                change_set.spy_correction_start,
            )
            items.append(
                FeatureInvalidation(
                    "spy_correction",
                    earliest_session=change_set.spy_correction_start,
                    affected_sessions=spy_sessions,
                    factor_ids=tuple(
                        sorted(item.factor_id for item in self.catalog.market_dependent)
                    ),
                    source_receipt_hash=(
                        change_set.receipt_hashes[0] if change_set.receipt_hashes else None
                    ),
                )
            )
        if change_set.sector_revision_changed:
            items.append(
                FeatureInvalidation("sector_revision_change", earliest_session=history_start)
            )
        # A membership change takes effect at the session it is decided for.
        # The entrant's own base Formula values are computed over its whole
        # history (the Formula windows need it); the Panel's cross-sections
        # change from the effective session on and nothing before it moves.
        items.extend(
            FeatureInvalidation(
                "manifest_addition",
                listing_id=listing_id,
                earliest_session=as_of_session,
                factor_ids=all_factors,
            )
            for listing_id in change_set.membership_additions
        )
        items.extend(
            FeatureInvalidation(
                "manifest_removal", listing_id=listing_id, earliest_session=as_of_session
            )
            for listing_id in change_set.membership_removals
        )
        deduplicated = tuple(
            sorted(
                set(items),
                key=lambda item: (
                    item.kind,
                    item.listing_id or "",
                    item.earliest_session or history_start,
                ),
            )
        )
        market_factors = tuple(item.factor_id for item in self.catalog.market_dependent)
        payload = {
            "ordered_stages": self._ORDER,
            "invalidations": [asdict(item) for item in deduplicated],
            "market_dependent_factor_ids": market_factors,
            "source_change_set_hash": change_set.change_set_hash,
        }
        return FeatureInvalidationPlan(
            ordered_stages=self._ORDER,
            invalidations=deduplicated,
            market_dependent_factor_ids=market_factors,
            source_change_set_hash=change_set.change_set_hash,
            plan_hash=canonical_hash(payload),
        )
