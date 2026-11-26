"""Explicit immutable catalog of Alpha target standardizations installed by the Host."""

from __future__ import annotations

from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .contracts import AlphaTargetStandardizationAdapter


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class AlphaTargetCatalogBinding(_Contract):
    """Content identity of the explicitly installed standardization capabilities."""

    ordered_standardization_ids: tuple[str, ...] = Field(min_length=1)
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaTargetCatalogBinding:
        """Require unique ordered target standardizations and exact catalog identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: The standardization axis repeats or its declared catalog hash differs.
        """
        if len(set(self.ordered_standardization_ids)) != len(self.ordered_standardization_ids):
            raise ValueError("ALPHA_TARGET_CATALOG_CAPABILITY_DUPLICATED")
        if self.catalog_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"catalog_hash"})
        ):
            raise ValueError("ALPHA_TARGET_CATALOG_IDENTITY_INVALID")
        return self


class AlphaTargetCatalog:
    """Route one policy's declared standardization to its installed adapter."""

    def __init__(self, adapters: tuple[AlphaTargetStandardizationAdapter, ...]) -> None:
        """Index a nonempty immutable catalog of unique target standardizations.

        Args:
            adapters: Explicit adapters in retained standardization order.

        Raises:
            ValueError: The adapter catalog is empty or a standardization identity repeats.
        """
        indexed = {value.standardization_id: value for value in adapters}
        if not adapters or len(indexed) != len(adapters):
            raise ValueError("ALPHA_TARGET_CATALOG_INVALID")
        self._adapters = MappingProxyType(indexed)

    @property
    def standardization_ids(self) -> tuple[str, ...]:
        """Read installed standardization identities in retained catalog order.

        Returns:
            Immutable ordered tuple of standardization IDs.
        """
        return tuple(self._adapters)

    @property
    def adapters(self) -> tuple[AlphaTargetStandardizationAdapter, ...]:
        """Read installed target standardization adapters in retained order.

        Returns:
            Immutable tuple of the indexed adapters.
        """
        return tuple(self._adapters.values())

    @property
    def binding(self) -> AlphaTargetCatalogBinding:
        """Seal the catalog ordered standardization identity axis.

        Returns:
            Validated catalog binding; numerical adapter content is not recomputed here.
        """
        identity = {"ordered_standardization_ids": list(self.standardization_ids)}
        return AlphaTargetCatalogBinding(
            ordered_standardization_ids=self.standardization_ids,
            catalog_hash=canonical_hash(identity),
        )

    def resolve(self, standardization_id: str) -> AlphaTargetStandardizationAdapter:
        """Resolve one explicitly installed target standardization adapter.

        Args:
            standardization_id: Exact installed standardization identity.

        Returns:
            The indexed adapter.

        Raises:
            ValueError: The standardization identity is not installed.
        """
        try:
            return self._adapters[standardization_id]
        except KeyError as error:
            raise ValueError("ALPHA_TARGET_STANDARDIZATION_NOT_INSTALLED") from error


def build_installed_alpha_target_catalog() -> AlphaTargetCatalog:
    """Construct the explicit installed catalog; no discovery or runtime mutation."""
    from .standardization import (
        CrossSectionalStdZStandardization,
        RankGaussStandardization,
        RobustZStandardization,
    )

    return AlphaTargetCatalog(
        (
            RankGaussStandardization(),
            RobustZStandardization(),
            CrossSectionalStdZStandardization(),
        )
    )


__all__ = [
    "AlphaTargetCatalog",
    "AlphaTargetCatalogBinding",
    "build_installed_alpha_target_catalog",
]
