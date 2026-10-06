"""A catalog that only adds columns to the shipped one, layered on it (V92).

A Feature row's identity binds its whole catalog (`feature_row_content_hash`), so a catalog that
only adds a column to the shipped one, a person's activation, would re-identify and rewrite every
row the store holds and copy every value into a new closure. Layered, the shipped catalog stays
the store's **base** and each added factor is a **column catalog** of its own: the catalog
restricted to that factor, its recipe and the maintenance entries naming it, under the catalog's
observation clock. Each part keeps its own rows and its own Feature closure, and the installed
catalog's rows are presented composed from them (`current_storage.py`). A catalog that changes
anything but added columns is its own base, unlayered, as every catalog was before.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog

_FACTOR_LISTS = (
    "market_dependent_factor_ids",
    "finite_calendar_factor_ids",
    "window_local_cumulative_factor_ids",
)


def restricted_catalog(catalog: FeatureCatalog, factor_ids: Iterable[str]) -> FeatureCatalog:
    """The catalog over some of its factors: their recipes and the policy entries naming them.

    Every catalog-level authority (the observation clock, the stable id, the policy's other
    settings) is kept, so the restriction binds exactly what decides those factors' values.

    Args:
        catalog: The catalog.
        factor_ids: The factors kept, all of them the catalog's.

    Returns:
        The restricted catalog, qualified as any catalog is.

    Raises:
        ValueError: No factor is kept, or one is not the catalog's.
    """
    kept = frozenset(factor_ids)
    if not kept or not kept <= set(catalog.factor_ids):
        raise ValueError("feature.catalog_restriction_outside_axis")
    payload = catalog.to_payload()
    payload["factors"] = [item for item in payload["factors"] if item["factor_id"] in kept]
    policy = payload["maintenance_policy"]
    for key in _FACTOR_LISTS:
        policy[key] = [value for value in policy[key] if value in kept]
    policy["maximum_invalidation_overrides"] = {
        key: value for key, value in policy["maximum_invalidation_overrides"].items() if key in kept
    }
    return FeatureCatalog.from_payload(payload)


@dataclass(frozen=True)
class FeatureCatalogLayer:
    """The catalogs whose rows compose an installed catalog's rows.

    Attributes:
        catalog: The installed catalog, whose rows readers read.
        base: The part holding the base's factors; the catalog itself when unlayered.
        columns: One column catalog per added factor, in the catalog's factor order.
    """

    catalog: FeatureCatalog
    base: FeatureCatalog
    columns: tuple[FeatureCatalog, ...] = ()

    @classmethod
    def over(
        cls, catalog: FeatureCatalog, base: FeatureCatalog | None = None
    ) -> FeatureCatalogLayer:
        """Layer a catalog on a base when it only adds columns to it.

        Args:
            catalog: The installed catalog.
            base: The base; the shipped catalog unless given.

        Returns:
            The layer; unlayered when the catalog is the base or changes more than columns.
        """
        shipped = base if base is not None else FeatureCatalog.load()
        held = set(shipped.factor_ids)
        added = tuple(factor_id for factor_id in catalog.factor_ids if factor_id not in held)
        if (
            not added
            or not held <= set(catalog.factor_ids)
            or restricted_catalog(catalog, held).binding.catalog_hash
            != shipped.binding.catalog_hash
        ):
            return cls(catalog=catalog, base=catalog)
        return cls(
            catalog=catalog,
            base=shipped,
            columns=tuple(restricted_catalog(catalog, (factor_id,)) for factor_id in added),
        )

    @property
    def layered(self) -> bool:
        """Whether the catalog's rows are composed of more than one part."""
        return bool(self.columns)

    @property
    def parts(self) -> tuple[FeatureCatalog, ...]:
        """The catalogs that own rows and closures, the base first."""
        return (self.base, *self.columns)

    @property
    def part_hashes(self) -> tuple[str, ...]:
        """Each part's catalog hash, the base first."""
        return tuple(part.binding.catalog_hash for part in self.parts)

    def part(self, catalog_hash: str) -> FeatureCatalog:
        """The part with a catalog hash.

        Args:
            catalog_hash: A part's catalog hash.

        Returns:
            The part.

        Raises:
            ValueError: No part has the hash.
        """
        for part in self.parts:
            if part.binding.catalog_hash == catalog_hash:
                return part
        raise ValueError("feature.catalog_outside_layer")

    def factor_parts(self) -> dict[str, str]:
        """The catalog hash of the part that computes each factor."""
        return {
            factor_id: part.binding.catalog_hash
            for part in self.parts
            for factor_id in part.factor_ids
        }

    def storage_parts(self) -> tuple[tuple[str, tuple[str, ...]], ...] | None:
        """Each part's hash and factor axis for the store's composed view; None when unlayered."""
        if not self.layered:
            return None
        return tuple((part.binding.catalog_hash, part.factor_ids) for part in self.parts)


__all__ = ["FeatureCatalogLayer", "restricted_catalog"]
