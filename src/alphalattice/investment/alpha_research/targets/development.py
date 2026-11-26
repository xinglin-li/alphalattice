"""A development target recipe that names its standardization outright.

``AlphaTargetPolicy.standardization_id`` is a *property*, derived from a closed
four-lane enum. That is correct for the frozen lanes -- it is why every published
target artifact's ``policy_hash`` never moved when the routing key was introduced
-- but it means the set of nameable standardizations is exactly two, and a third
one cannot be selected at all.

The observable consequence was a case-study standardization that set
``standardization_id = RANK_GAUSS_STANDARDIZATION_ID`` on itself in order to be
routed. That is not an extension seam; it is a disguise. The evidence it produced
described a method that had not run, which is the same class of defect as an
adapter computing under another implementation's identity.

So a development recipe carries the routing key as a **field**. The lane stays
what it always was -- winsorization, neutralization and the raw economic-return
column are lane properties and are unchanged -- and only the standardization step
becomes selectable.

Nothing here is published. ``AlphaTargetPolicy``, its derived property, the
current candidate and every frozen artifact are untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol, Self

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.investment.alpha_research.targets.catalog import (
    AlphaTargetCatalog,
    build_installed_alpha_target_catalog,
)
from alphalattice.investment.alpha_research.targets.execution_outcome import (
    AlphaTargetLane,
    AlphaTargetPolicy,
    build_alpha_target_policy,
    compile_alpha_target_surface,
)
from alphalattice.investment.alpha_research.targets.standardization import (
    RANK_GAUSS_STANDARDIZATION_ID,
    ROBUST_Z_STANDARDIZATION_ID,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sector_treatment import (
    SECTOR_HISTORY_BACKFILLED,
    SectorHistoryTreatment,
)


class AlphaDevelopmentTargetRecipe(BaseModel):  # type: ignore[misc]
    """One development target: a frozen lane policy plus a named standardization.

    Both identities are carried, and they answer different questions.
    ``policy.policy_hash`` is the lane -- what the target *is*. ``recipe_hash``
    covers the lane together with the standardization actually selected, so two
    recipes over one lane with different standardizations are two recipes rather
    than one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["AlphaDevelopmentTargetRecipe"] = "AlphaDevelopmentTargetRecipe"
    policy: AlphaTargetPolicy
    standardization_id: str = Field(min_length=1, max_length=128)
    """The installed standardization this recipe selects, stated rather than derived."""

    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, *, policy: AlphaTargetPolicy, standardization_id: str) -> Self:
        """Seal a development target policy and standardization identity.

        Args:
            policy: Typed declared target policy.
            standardization_id: Exact target standardization route.

        Returns:
            Validated recipe with canonical policy/standardization identity.
        """
        values = {
            "kind": "AlphaDevelopmentTargetRecipe",
            "policy": policy.model_dump(mode="json"),
            "standardization_id": standardization_id,
        }
        return cls(
            policy=policy,
            standardization_id=standardization_id,
            recipe_hash=str(canonical_hash(values)),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact development target recipe identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: The canonical recipe payload differs from its hash.
        """
        expected = canonical_hash(self.model_dump(mode="json", exclude={"recipe_hash"}))
        if self.recipe_hash != expected:
            raise ValueError("alpha_research.development_target_recipe_identity_invalid")
        return self


def compile_alpha_development_target_surface(
    *,
    source_table: pa.Table,
    recipe: AlphaDevelopmentTargetRecipe,
    sector_by_listing_id: Mapping[str, str],
    standardizations: AlphaTargetCatalog | None = None,
) -> pa.Table:
    """Compile a development target lane through the recipe's named standardization.

    The mathematics is not reimplemented. This resolves the recipe's declared
    standardization against the installed catalog -- failing closed if it is not
    installed -- and hands it to the same compiler the frozen lanes use, so
    winsorization, sector neutralization, coverage masking and the retained raw
    economic-return column are byte-identical to production behaviour.
    """
    catalog = standardizations or build_installed_alpha_target_catalog()
    # Resolved here, before any numbers move, so an uninstalled standardization
    # is refused rather than silently falling back to the lane's derived key.
    catalog.resolve(recipe.standardization_id)
    return compile_alpha_target_surface(
        source_table=source_table,
        policy=recipe.policy,
        sector_by_listing_id=sector_by_listing_id,
        standardizations=catalog,
        standardization_id=recipe.standardization_id,
    )


class AlphaDevelopmentTargetFoldRecord(BaseModel):  # type: ignore[misc]
    """The transformed target values of exactly one fold, with their own edges.

    A single digest over every fold's bytes concatenated end to end has no
    internal boundaries: it says nothing about which values belonged to which
    fold, and two differently-shaped splits can produce the same byte stream and
    therefore the same hash. Recording each fold separately, with the dtype and
    shape that gave those bytes their meaning, is what makes the digest a
    statement about arrays rather than about a buffer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["AlphaDevelopmentTargetFoldRecord"] = "AlphaDevelopmentTargetFoldRecord"
    fold_index: int = Field(ge=0)
    fold_commitment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_dtype: str = Field(min_length=1, max_length=32)
    training_shape: tuple[int, ...] = Field(min_length=1)
    training_value_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_dtype: str = Field(min_length=1, max_length=32)
    validation_shape: tuple[int, ...] = Field(min_length=1)
    validation_value_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        fold_index: int,
        fold_commitment_hash: str,
        training_dtype: str,
        training_shape: tuple[int, ...],
        training_value_hash: str,
        validation_dtype: str,
        validation_shape: tuple[int, ...],
        validation_value_hash: str,
    ) -> Self:
        """Seal one development fold training/validation dtype, shape and value record.

        Args:
            fold_index: Declared integer fold index.
            fold_commitment_hash: Exact causal fold commitment.
            training_dtype: Training target dtype.
            training_shape: Ordered integer training shape.
            training_value_hash: Exact transformed training values.
            validation_dtype: Validation target dtype.
            validation_shape: Ordered integer validation shape.
            validation_value_hash: Exact transformed validation values.

        Returns:
            Validated fold record with canonical normalized metadata identity.
        """
        values = {
            "kind": "AlphaDevelopmentTargetFoldRecord",
            "fold_index": int(fold_index),
            "fold_commitment_hash": fold_commitment_hash,
            "training_dtype": training_dtype,
            "training_shape": [int(value) for value in training_shape],
            "training_value_hash": training_value_hash,
            "validation_dtype": validation_dtype,
            "validation_shape": [int(value) for value in validation_shape],
            "validation_value_hash": validation_value_hash,
        }
        return cls(
            fold_index=int(fold_index),
            fold_commitment_hash=fold_commitment_hash,
            training_dtype=training_dtype,
            training_shape=tuple(int(value) for value in training_shape),
            training_value_hash=training_value_hash,
            validation_dtype=validation_dtype,
            validation_shape=tuple(int(value) for value in validation_shape),
            validation_value_hash=validation_value_hash,
            record_hash=str(canonical_hash(values)),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact transformed target fold-record identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: The canonical fold record differs from its hash.
        """
        if self.record_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"record_hash"})
        ):
            raise ValueError("alpha_research.development_target_fold_record_invalid")
        return self


class _TargetRecipeBindingLike(Protocol):
    @property
    def binding_hash(self) -> str: ...

    @property
    def standardization_id(self) -> str: ...


class AlphaDevelopmentTargetMaterializationBinding(BaseModel):  # type: ignore[misc]
    """What the declared method actually produced, sealed after the surface exists.

    The companion to the recipe binding, and deliberately a second object rather
    than four more fields on the first. A transformed-value hash is not knowable
    when a fold plan is built -- no array has been materialized yet -- so a single
    binding would have to carry either an absent value or an invented one, and an
    invented one is the defect this whole gate is about.

    Sealed after the target surface is compiled and before any fit, so it answers
    "these values, over this axis, came from that declared recipe".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["AlphaDevelopmentTargetMaterializationBinding"] = (
        "AlphaDevelopmentTargetMaterializationBinding"
    )
    recipe_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    standardization_id: str = Field(min_length=1, max_length=128)
    ordered_base_feature_ids: tuple[str, ...] = Field(min_length=1)
    fold_records: tuple[AlphaDevelopmentTargetFoldRecord, ...] = Field(min_length=1)
    training_row_count: int = Field(ge=1)
    transformed_target_value_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Derived from the ordered fold records, so it is a summary of them rather
    than a second, independently computable opinion about the same values."""

    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @staticmethod
    def value_hash_for(records: tuple[AlphaDevelopmentTargetFoldRecord, ...]) -> str:
        """Hash ordered transformed target fold-record identities.

        Args:
            records: Fold records in materialization order.

        Returns:
            Canonical AlphaDevelopmentTargetValues identity preserving the declared order.
        """
        return str(
            canonical_hash(
                {
                    "kind": "AlphaDevelopmentTargetValues",
                    "ordered_fold_records": [value.record_hash for value in records],
                }
            )
        )

    @property
    def fold_commitment_hashes(self) -> tuple[str, ...]:
        """The commitments, in fold order, for readers that only need the axis."""
        return tuple(value.fold_commitment_hash for value in self.fold_records)

    @classmethod
    def create(
        cls,
        *,
        recipe_binding: _TargetRecipeBindingLike,
        ordered_base_feature_ids: tuple[str, ...],
        fold_records: tuple[AlphaDevelopmentTargetFoldRecord, ...],
        training_row_count: int,
    ) -> Self:
        """Seal transformed targets against recipe, feature axis and ordered fold records.

        Args:
            recipe_binding: Exact target recipe/source binding and standardization.
            ordered_base_feature_ids: Declared ordered base-feature axis.
            fold_records: Records in materialization/fit order.
            training_row_count: Declared integer training row count.

        Returns:
            Validated materialization binding with derived transformed-value and binding hashes.
        """
        records = tuple(fold_records)
        value_hash = cls.value_hash_for(records)
        values = {
            "kind": "AlphaDevelopmentTargetMaterializationBinding",
            "recipe_binding_hash": recipe_binding.binding_hash,
            "standardization_id": recipe_binding.standardization_id,
            "ordered_base_feature_ids": list(ordered_base_feature_ids),
            "fold_records": [value.model_dump(mode="json") for value in records],
            "training_row_count": int(training_row_count),
            "transformed_target_value_hash": value_hash,
        }
        return cls(
            recipe_binding_hash=recipe_binding.binding_hash,
            standardization_id=recipe_binding.standardization_id,
            ordered_base_feature_ids=tuple(ordered_base_feature_ids),
            fold_records=records,
            training_row_count=int(training_row_count),
            transformed_target_value_hash=value_hash,
            binding_hash=str(canonical_hash(values)),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require contiguous fold order and exact target value/materialization identities.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Fold indices are not contiguous from zero or the ordered value/binding
                hashes differ.
        """
        indices = tuple(value.fold_index for value in self.fold_records)
        if indices != tuple(range(len(indices))):
            # Fold order is the order the arrays were built and fitted in, so a
            # gap or a permutation describes a different materialization.
            raise ValueError("alpha_research.development_target_fold_records_unordered")
        if self.transformed_target_value_hash != self.value_hash_for(self.fold_records):
            raise ValueError("alpha_research.development_target_value_hash_invalid")
        if self.binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"binding_hash"})
        ):
            raise ValueError("alpha_research.development_target_materialization_binding_invalid")
        return self


DEFAULT_DEVELOPMENT_TARGET_RECIPE_ID = "SECTOR_RESIDUAL_RANK_GAUSS"
"""The lane and standardization the frozen default pairing already implies."""

CROSS_STANDARDIZED_DEVELOPMENT_TARGET_RECIPE_ID = "SECTOR_RESIDUAL_ROBUST_Z"
"""A sector-residual lane standardized by robust-z instead of rank-gauss.

The non-default recipe, and non-default in the way that matters: no
``AlphaTargetPolicy`` can express it. The lane's derived ``standardization_id``
is rank-gauss and always will be, so before a recipe carried the routing key as a
field this pairing was unnameable rather than merely unused. Both halves are
product-installed -- the lane is frozen, the standardization is one of the two in
the installed catalog -- so proving the seam needs no case-study method in the
product.
"""


class AlphaDevelopmentTargetRecipeCatalog:
    """The development target recipes a Host installs, keyed by declared id.

    Explicit, like every other catalog on this surface: no discovery, no entry
    points, no dynamic import. Installing a recipe is an entry in
    ``installed_alpha_development_target_recipes`` and nothing else.

    Parameterized by ``sector_revision`` because a target policy binds the sector
    map it neutralized against. That is authority the workspace owns, not a
    constant a Desk may assume, and a catalog that defaulted it would let two runs
    against different sector maps produce the same recipe identity.
    """

    def __init__(self, recipes: Mapping[str, AlphaDevelopmentTargetRecipe]) -> None:
        """Retain a nonempty explicit catalog of development target recipes.

        Args:
            recipes: Declared recipe mapping to copy.

        Raises:
            ValueError: The recipe catalog is empty.
        """
        if not recipes:
            raise ValueError("ALPHA_DEVELOPMENT_TARGET_RECIPE_CATALOG_EMPTY")
        self._recipes = dict(recipes)

    @property
    def recipe_ids(self) -> tuple[str, ...]:
        """Read installed development target recipe identities in sorted order.

        Returns:
            Sorted immutable tuple of recipe keys.
        """
        return tuple(sorted(self._recipes))

    def resolve(self, recipe_id: str) -> AlphaDevelopmentTargetRecipe:
        """Resolve one explicitly installed development target recipe.

        Args:
            recipe_id: Exact installed recipe key.

        Returns:
            The indexed recipe.

        Raises:
            ValueError: The requested development target recipe is not installed.
        """
        try:
            return self._recipes[recipe_id]
        except KeyError as error:
            raise ValueError("ALPHA_DEVELOPMENT_TARGET_RECIPE_NOT_INSTALLED") from error

    @property
    def catalog_hash(self) -> str:
        """Identity of the installed set, so adding a recipe moves Program identity."""
        return str(
            canonical_hash(
                {
                    "kind": "AlphaDevelopmentTargetRecipeCatalog",
                    "ordered_recipes": [
                        {"recipe_id": key, "recipe_hash": self._recipes[key].recipe_hash}
                        for key in sorted(self._recipes)
                    ],
                }
            )
        )


def installed_alpha_development_target_recipes(
    *,
    sector_revision: str,
    sector_history_treatment: SectorHistoryTreatment = SECTOR_HISTORY_BACKFILLED,
) -> AlphaDevelopmentTargetRecipeCatalog:
    """Every development target recipe this build installs, in declaration order."""
    lane = AlphaTargetLane.SECTOR_RESIDUAL_RANK_GAUSS
    policy = build_alpha_target_policy(
        lane=lane,
        sector_revision=sector_revision,
        sector_history_treatment=sector_history_treatment,
    )
    return AlphaDevelopmentTargetRecipeCatalog(
        {
            DEFAULT_DEVELOPMENT_TARGET_RECIPE_ID: AlphaDevelopmentTargetRecipe.create(
                policy=policy,
                standardization_id=RANK_GAUSS_STANDARDIZATION_ID,
            ),
            CROSS_STANDARDIZED_DEVELOPMENT_TARGET_RECIPE_ID: AlphaDevelopmentTargetRecipe.create(
                policy=policy,
                standardization_id=ROBUST_Z_STANDARDIZATION_ID,
            ),
        }
    )


__all__ = [
    "CROSS_STANDARDIZED_DEVELOPMENT_TARGET_RECIPE_ID",
    "DEFAULT_DEVELOPMENT_TARGET_RECIPE_ID",
    "AlphaDevelopmentTargetFoldRecord",
    "AlphaDevelopmentTargetMaterializationBinding",
    "AlphaDevelopmentTargetRecipe",
    "AlphaDevelopmentTargetRecipeCatalog",
    "compile_alpha_development_target_surface",
    "installed_alpha_development_target_recipes",
]
