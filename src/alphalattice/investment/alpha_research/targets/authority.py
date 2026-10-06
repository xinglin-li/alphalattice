"""One installed target method, as the compiler and the executor need to see it.

The development path was built around ``AlphaDevelopmentTargetRecipe``, which is
a frozen ``AlphaTargetLane`` policy plus a named standardization. That shape is
correct for the four frozen lanes and cannot express the canonical target at all:
``SECTOR_RESIDUAL_CROSS_SECTIONAL_STD_Z`` bounds the residual rather than the raw
return and re-demeans afterwards, so it is a different composition, not a
different standardization of the same one. Adding it to the lane enum would
rotate every frozen policy hash that derives from that enum.

So the boundary moves instead of the enum. Everything the generic path actually
needs from a target -- what it is called, what identity it has, which
standardization it names, which model lanes it admits, and how to compile one
surface -- is stated here as a Protocol. The executor asks a target method those
questions and never asks which family it belongs to, which is what lets one
executor run all three arms.

Two things are deliberately *not* on this Protocol. There is no way to ask for a
lane, because two of the three implementations have none. And there is no way to
ask a method to describe its own admission: what a target admits is checked by
the model mandate against the binding in ``experiments/mandate.py``, so a target
that could answer "yes, I am admitted" would be answering the question that
authority exists to ask.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, Self

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sector_treatment import (
    SECTOR_HISTORY_BACKFILLED,
    SectorHistoryTreatment,
)

from .canonical import (
    CANONICAL_ALPHA_TARGET_RECIPE_ID,
    JOINT_PRIMARY_ALPHA_TARGET_RECIPE_ID,
    CanonicalAlphaTargetRecipe,
    build_canonical_alpha_target_recipe,
    build_joint_primary_alpha_target_recipe,
    compile_canonical_alpha_target_surface,
)
from .catalog import AlphaTargetCatalog
from .development import (
    AlphaDevelopmentTargetRecipe,
    compile_alpha_development_target_surface,
    installed_alpha_development_target_recipes,
)
from .execution_outcome import AlphaTargetBoundaryError, AlphaTargetLane
from .total_return import (
    TOTAL_RETURN_ALPHA_TARGET_RECIPE_ID,
    TotalReturnAlphaTargetRecipe,
    build_total_return_alpha_target_recipe,
    compile_total_return_alpha_target_surface,
)
from .unbounded import (
    UNBOUNDED_SENSITIVITY_TARGET_RECIPE_ID,
    UnboundedSensitivityAlphaTargetRecipe,
    build_unbounded_sensitivity_target_recipe,
    compile_unbounded_sensitivity_target_surface,
)


class AlphaTargetMethod(Protocol):
    """The complete target boundary the generic Alpha path depends on."""

    @property
    def target_recipe_id(self) -> str:
        """The installed method id a document may name."""

    @property
    def target_method_hash(self) -> str:
        """Content identity of the whole method, constants included."""

    @property
    def standardization_id(self) -> str:
        """The installed standardization this method routes through."""

    @property
    def sector_revision(self) -> str:
        """The Sector map this method neutralizes against.

        On the boundary because a verifier rebuilding the installed method needs
        it, and it is workspace authority rather than a constant: two runs over
        different Sector maps must not share a method identity.
        """

    @property
    def cross_sectional_authority_id(self) -> str:
        """What defines the cross-section the target transforms."""

    @property
    def admitted_model_lanes(self) -> tuple[AlphaTargetLane, ...] | None:
        """Frozen lanes a model recipe may be admitted against, or ``None``.

        ``None`` means "this method is not one of the frozen lanes" and never
        means "unconstrained": a model admitted under it is bound to the target
        by ``AlphaDevelopmentModelMethodBinding`` instead, which carries strictly
        more identity than a lane does.
        """

    def compile_target_surface(
        self,
        *,
        source_table: pa.Table,
        sector_by_listing_id: Mapping[str, str],
        standardizations: AlphaTargetCatalog | None = None,
    ) -> pa.Table:
        """Compile the fit surface this method defines, retaining its raw lanes."""


class AlphaTargetMethodBinding(BaseModel):  # type: ignore[misc]
    """What target method a run declared, bound to the outcome it transformed.

    The successor to ``AlphaDevelopmentTargetRecipeBinding`` for methods that
    have no lane. The older binding stays exactly as it is: it carries
    ``target_policy_hash``, which is a real fact about a frozen lane and would be
    a fabrication for a method that has no policy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["AlphaTargetMethodBinding"] = "AlphaTargetMethodBinding"
    target_recipe_id: str = Field(min_length=1, max_length=96)
    target_method_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    standardization_id: str = Field(min_length=1, max_length=128)
    sector_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_outcome_recipe_id: str = Field(min_length=1, max_length=96)
    """The two per-run facts a verifier needs to rebuild this method from the
    installed catalog. Without them it could only check the method hash against
    itself, which a re-sealed binding also passes."""

    causal_outcome_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome_method_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        method: AlphaTargetMethod,
        execution_outcome_recipe_id: str,
        causal_outcome_snapshot_hash: str,
        outcome_method_binding_hash: str,
    ) -> Self:
        """Seal a declared target method against its exact execution-outcome authority.

        Args:
            method: Resolved sector-aware target method.
            execution_outcome_recipe_id: Outcome recipe transformed by the method.
            causal_outcome_snapshot_hash: Exact causal source outcome snapshot.
            outcome_method_binding_hash: Exact source outcome-method binding.

        Returns:
            Validated binding with method, standardization, sector and source identities.
        """
        values: dict[str, object] = {
            "kind": "AlphaTargetMethodBinding",
            "target_recipe_id": method.target_recipe_id,
            "target_method_hash": method.target_method_hash,
            "standardization_id": method.standardization_id,
            "sector_revision": method.sector_revision,
            "execution_outcome_recipe_id": execution_outcome_recipe_id,
            "causal_outcome_snapshot_hash": causal_outcome_snapshot_hash,
            "outcome_method_binding_hash": outcome_method_binding_hash,
        }
        return cls(**values, binding_hash=str(canonical_hash(values)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact target method and outcome binding identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaTargetBoundaryError: The canonical method/source binding differs from its declared
                hash.
        """
        if self.binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"binding_hash"})
        ):
            raise AlphaTargetBoundaryError("alpha_research.target_method_binding_identity_invalid")
        return self


class WholeUniverseAlphaTargetMethodBinding(BaseModel):  # type: ignore[misc]
    """Successor binding for targets whose authority is the whole Universe.

    Kept separate from ``AlphaTargetMethodBinding`` so existing Sector-bound
    identities do not rotate and the total-return target never fabricates a
    Sector revision it does not consume.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["WholeUniverseAlphaTargetMethodBinding"] = "WholeUniverseAlphaTargetMethodBinding"
    target_recipe_id: Literal["CROSS_SECTIONAL_TOTAL_RETURN_STD_Z"]
    target_method_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    standardization_id: str = Field(min_length=1, max_length=128)
    cross_sectional_authority_id: Literal["WHOLE_ACTIVE_UNIVERSE"]
    listing_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_outcome_recipe_id: str = Field(min_length=1, max_length=96)
    causal_outcome_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome_method_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    maturity_lag_sessions: int = Field(ge=2)
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        method: TotalReturnTargetMethod,
        listing_set_hash: str,
        execution_outcome_recipe_id: str,
        causal_outcome_snapshot_hash: str,
        outcome_method_binding_hash: str,
        maturity_lag_sessions: int,
    ) -> Self:
        """Seal a total-return target method against whole-universe outcome authority.

        Args:
            method: Resolved total-return target method without sector authority.
            listing_set_hash: Required whole-universe listing population.
            execution_outcome_recipe_id: Exact transformed outcome recipe.
            causal_outcome_snapshot_hash: Exact causal source snapshot.
            outcome_method_binding_hash: Exact source outcome-method binding.
            maturity_lag_sessions: Declared label maturity lag.

        Returns:
            Validated whole-universe method/source binding with canonical identity.
        """
        values: dict[str, object] = {
            "kind": "WholeUniverseAlphaTargetMethodBinding",
            "target_recipe_id": method.target_recipe_id,
            "target_method_hash": method.target_method_hash,
            "standardization_id": method.standardization_id,
            "cross_sectional_authority_id": method.cross_sectional_authority_id,
            "listing_set_hash": listing_set_hash,
            "execution_outcome_recipe_id": execution_outcome_recipe_id,
            "causal_outcome_snapshot_hash": causal_outcome_snapshot_hash,
            "outcome_method_binding_hash": outcome_method_binding_hash,
            "maturity_lag_sessions": maturity_lag_sessions,
        }
        return cls(**values, binding_hash=str(canonical_hash(values)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact whole-universe target and outcome binding identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaTargetBoundaryError: The canonical method/source binding differs from its declared
                hash.
        """
        if self.binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"binding_hash"})
        ):
            raise AlphaTargetBoundaryError(
                "alpha_research.whole_universe_target_binding_identity_invalid"
            )
        return self


type InstalledAlphaTargetMethodBinding = (
    AlphaTargetMethodBinding | WholeUniverseAlphaTargetMethodBinding
)


@dataclass(frozen=True, slots=True)
class InstalledLaneTargetMethod:
    """A frozen lane and its named standardization, behind the shared boundary.

    Adapts rather than replaces. The recipe travels whole into the same
    development compiler it always used, so every frozen lane's numbers and
    hashes are exactly what they were.
    """

    recipe_id: str
    """The declared catalog id, carried rather than derived from the lane.

    Two installed recipes share the rank-gauss *lane* and differ only by the
    standardization they name, so the lane cannot identify either of them. A
    method that derived its id from the lane would answer with the same name for
    both, which is precisely the confusion the recipe id exists to prevent.
    """

    recipe: AlphaDevelopmentTargetRecipe

    @property
    def target_recipe_id(self) -> str:
        """Read the installed target recipe identity.

        Returns:
            The declared recipe_id.
        """
        return self.recipe_id

    @property
    def target_method_hash(self) -> str:
        """Read the complete target method content identity.

        Returns:
            The retained recipe_hash.
        """
        return str(self.recipe.recipe_hash)

    @property
    def standardization_id(self) -> str:
        """Read the standardization identity declared by this target method.

        Returns:
            The retained standardization_id.
        """
        return str(self.recipe.standardization_id)

    @property
    def sector_revision(self) -> str:
        """Read the sector classification revision used by this target method.

        Returns:
            The retained policy sector_revision.
        """
        return str(self.recipe.policy.sector_revision)

    @property
    def cross_sectional_authority_id(self) -> str:
        """Read the authority defining the target cross-section.

        Returns:
            CURRENT_SECTOR_REVISION for this sector-aware method.
        """
        return "CURRENT_SECTOR_REVISION"

    @property
    def admitted_model_lanes(self) -> tuple[AlphaTargetLane, ...] | None:
        """Read the frozen lanes admitted by this target method.

        Returns:
            The declared policy lane as a one-item tuple.
        """
        return (self.recipe.policy.lane,)

    def compile_target_surface(
        self,
        *,
        source_table: pa.Table,
        sector_by_listing_id: Mapping[str, str],
        standardizations: AlphaTargetCatalog | None = None,
    ) -> pa.Table:
        """Compile the declared frozen-lane development target surface.

        Args:
            source_table: Declared causal execution-outcome rows.
            sector_by_listing_id: Current sector classification; unused by total-return compilation.
            standardizations: Optional explicit target standardization catalog where the selected
                owner admits it.

        Returns:
            Arrow target surface produced by the installed lane compiler.
        """
        return compile_alpha_development_target_surface(
            source_table=source_table,
            recipe=self.recipe,
            sector_by_listing_id=sector_by_listing_id,
            standardizations=standardizations,
        )


@dataclass(frozen=True, slots=True)
class CanonicalTargetMethod:
    """The canonical bounded target: residual first, bounded, re-demeaned, std-Z."""

    recipe: CanonicalAlphaTargetRecipe

    @property
    def target_recipe_id(self) -> str:
        """Read the installed target recipe identity.

        Returns:
            The target_recipe_id retained by this recipe.
        """
        return str(self.recipe.target_recipe_id)

    @property
    def target_method_hash(self) -> str:
        """Read the complete target method content identity.

        Returns:
            The retained recipe_hash.
        """
        return str(self.recipe.recipe_hash)

    @property
    def standardization_id(self) -> str:
        """Read the standardization identity declared by this target method.

        Returns:
            The retained standardization_id.
        """
        return str(self.recipe.standardization_id)

    @property
    def sector_revision(self) -> str:
        """Read the sector classification revision used by this target method.

        Returns:
            The recipe sector_revision.
        """
        return str(self.recipe.sector_revision)

    @property
    def cross_sectional_authority_id(self) -> str:
        """Read the authority defining the target cross-section.

        Returns:
            CURRENT_SECTOR_REVISION for this sector-aware method.
        """
        return "CURRENT_SECTOR_REVISION"

    @property
    def admitted_model_lanes(self) -> tuple[AlphaTargetLane, ...] | None:
        """Read the frozen lanes admitted by this target method.

        Returns:
            None: models bind the complete target method instead of a frozen lane.
        """
        return None

    def compile_target_surface(
        self,
        *,
        source_table: pa.Table,
        sector_by_listing_id: Mapping[str, str],
        standardizations: AlphaTargetCatalog | None = None,
    ) -> pa.Table:
        # The canonical compiler is the owner of this composition and returns a
        # whole surface -- targets, per-session dispersion and lane identities.
        # Only the fit lane is handed to the array path; the scale is published
        # by the canonical development service, and recomputing it here would
        # create a second producer of a number that G4 binds by identity.
        """Compile canonical target lanes through their single deterministic owner.

        Args:
            source_table: Declared causal execution-outcome rows.
            sector_by_listing_id: Current sector classification; unused by total-return compilation.
            standardizations: Optional explicit target standardization catalog where the selected
                owner admits it.

        Returns:
            Fit-target table from the complete canonical surface; its dispersion remains with the
            canonical owner.
        """
        return compile_canonical_alpha_target_surface(
            source_table=source_table,
            recipe=self.recipe,
            sector_by_listing_id=sector_by_listing_id,
            standardizations=standardizations,
        ).targets


@dataclass(frozen=True, slots=True)
class UnboundedSensitivityTargetMethod:
    """The same composition with the bounding step removed, for sensitivity only."""

    recipe: UnboundedSensitivityAlphaTargetRecipe

    @property
    def target_recipe_id(self) -> str:
        """Read the installed target recipe identity.

        Returns:
            The target_recipe_id retained by this recipe.
        """
        return str(self.recipe.target_recipe_id)

    @property
    def target_method_hash(self) -> str:
        """Read the complete target method content identity.

        Returns:
            The retained recipe_hash.
        """
        return str(self.recipe.recipe_hash)

    @property
    def standardization_id(self) -> str:
        """Read the standardization identity declared by this target method.

        Returns:
            The retained standardization_id.
        """
        return str(self.recipe.standardization_id)

    @property
    def sector_revision(self) -> str:
        """Read the sector classification revision used by this target method.

        Returns:
            The recipe sector_revision.
        """
        return str(self.recipe.sector_revision)

    @property
    def cross_sectional_authority_id(self) -> str:
        """Read the authority defining the target cross-section.

        Returns:
            CURRENT_SECTOR_REVISION for this sector-aware method.
        """
        return "CURRENT_SECTOR_REVISION"

    @property
    def admitted_model_lanes(self) -> tuple[AlphaTargetLane, ...] | None:
        """Read the frozen lanes admitted by this target method.

        Returns:
            None: models bind the complete target method instead of a frozen lane.
        """
        return None

    def compile_target_surface(
        self,
        *,
        source_table: pa.Table,
        sector_by_listing_id: Mapping[str, str],
        standardizations: AlphaTargetCatalog | None = None,
    ) -> pa.Table:
        """Compile the declared unbounded sensitivity target through its owner.

        Args:
            source_table: Declared causal execution-outcome rows.
            sector_by_listing_id: Current sector classification; unused by total-return compilation.
            standardizations: Optional explicit target standardization catalog where the selected
                owner admits it.

        Returns:
            Fit-target table from the complete unbounded sensitivity surface.
        """
        return compile_unbounded_sensitivity_target_surface(
            source_table=source_table,
            recipe=self.recipe,
            sector_by_listing_id=sector_by_listing_id,
            standardizations=standardizations,
        ).targets


@dataclass(frozen=True, slots=True)
class TotalReturnTargetMethod:
    """Whole-Universe total-return target; no Sector authority is consumed."""

    recipe: TotalReturnAlphaTargetRecipe

    @property
    def target_recipe_id(self) -> str:
        """Read the installed target recipe identity.

        Returns:
            The target_recipe_id retained by this recipe.
        """
        return self.recipe.target_recipe_id

    @property
    def target_method_hash(self) -> str:
        """Read the complete target method content identity.

        Returns:
            The retained recipe_hash.
        """
        return self.recipe.recipe_hash

    @property
    def standardization_id(self) -> str:
        """Read the standardization identity declared by this target method.

        Returns:
            The retained standardization_id.
        """
        return self.recipe.standardization_id

    @property
    def sector_revision(self) -> str:
        """Refuse a sector revision for a whole-universe total-return target.

        Raises:
            AlphaTargetBoundaryError: This method has no sector authority.
        """
        raise AlphaTargetBoundaryError("alpha_research.total_return_target_has_no_sector_authority")

    @property
    def cross_sectional_authority_id(self) -> str:
        """Read the authority defining the target cross-section.

        Returns:
            The declared whole-universe cross-sectional authority ID.
        """
        return self.recipe.cross_sectional_authority_id

    @property
    def admitted_model_lanes(self) -> tuple[AlphaTargetLane, ...] | None:
        """Read the frozen lanes admitted by this target method.

        Returns:
            None: models bind the complete target method instead of a frozen lane.
        """
        return None

    def compile_target_surface(
        self,
        *,
        source_table: pa.Table,
        sector_by_listing_id: Mapping[str, str],
        standardizations: AlphaTargetCatalog | None = None,
    ) -> pa.Table:
        """Compile whole-universe total-return targets without sector neutralization.

        Args:
            source_table: Declared causal execution-outcome rows.
            sector_by_listing_id: Current sector classification; unused by total-return compilation.
            standardizations: Optional explicit target standardization catalog where the selected
                owner admits it.

        Returns:
            Fit-target table from the complete total-return surface; sector mapping and optional
            standardizations are unused.
        """
        del sector_by_listing_id, standardizations
        return compile_total_return_alpha_target_surface(
            source_table=source_table, recipe=self.recipe
        ).targets


class AlphaTargetMethodCatalog:
    """Every target method a Host installs, keyed by the id a document may name.

    Explicit, like every other catalog on this surface: no discovery, no entry
    points, no dynamic import. It is the single place that answers "may this
    document name this target", which is why the compiler and the executor
    resolve through the same object rather than each knowing a set of families.
    """

    def __init__(self, methods: dict[str, AlphaTargetMethod]) -> None:
        """Index a nonempty target-method catalog with exact method/key agreement.

        Args:
            methods: Declared installed methods keyed by their target_recipe_id.

        Raises:
            AlphaTargetBoundaryError: The catalog is empty or a key disagrees with its method
                identity.
        """
        if not methods:
            raise AlphaTargetBoundaryError("alpha_research.target_method_catalog_empty")
        for declared, method in methods.items():
            if method.target_recipe_id != declared:
                # A method filed under a name it does not answer to would make
                # the catalog key and the sealed identity disagree, and every
                # hash downstream would still validate.
                raise AlphaTargetBoundaryError("alpha_research.target_method_catalog_misfiled")
        self._methods = dict(methods)

    @property
    def method_ids(self) -> tuple[str, ...]:
        """Read installed target method identities in sorted order.

        Returns:
            Sorted immutable tuple of target recipe IDs.
        """
        return tuple(sorted(self._methods))

    def resolve(self, target_recipe_id: str) -> AlphaTargetMethod:
        """Resolve the exact declared installed target method.

        Args:
            target_recipe_id: Installed method identity to resolve.

        Returns:
            The indexed target method.

        Raises:
            AlphaTargetBoundaryError: The requested method is not installed.
        """
        try:
            return self._methods[target_recipe_id]
        except KeyError as error:
            raise AlphaTargetBoundaryError("alpha_research.target_method_not_installed") from error

    @property
    def catalog_hash(self) -> str:
        """Identity of the installed set, so adding a method moves Program identity."""
        return str(
            canonical_hash(
                {
                    "kind": "AlphaTargetMethodCatalog",
                    "ordered_methods": [
                        {
                            "target_recipe_id": key,
                            "target_method_hash": self._methods[key].target_method_hash,
                        }
                        for key in sorted(self._methods)
                    ],
                }
            )
        )


def installed_alpha_target_methods(
    *,
    sector_revision: str,
    execution_outcome_recipe_id: str,
    sector_history_treatment: SectorHistoryTreatment = SECTOR_HISTORY_BACKFILLED,
) -> AlphaTargetMethodCatalog:
    """Every target method this build installs: two frozen lanes and two successors.

    The successors are parameterized by the outcome method they transform,
    because a target is a transformation *of* a causal outcome and the same
    composition over a different span is a different method. The frozen lanes are
    not: their policies predate the seam and their identities must not move.
    """
    # The frozen lanes state what their sessions read (V346); the successors bind the Panel.
    lane_recipes = installed_alpha_development_target_recipes(
        sector_revision=sector_revision, sector_history_treatment=sector_history_treatment
    )
    methods: dict[str, AlphaTargetMethod] = {
        recipe_id: InstalledLaneTargetMethod(
            recipe_id=recipe_id, recipe=lane_recipes.resolve(recipe_id)
        )
        for recipe_id in lane_recipes.recipe_ids
    }
    methods[CANONICAL_ALPHA_TARGET_RECIPE_ID] = CanonicalTargetMethod(
        recipe=build_canonical_alpha_target_recipe(
            execution_outcome_recipe_id=execution_outcome_recipe_id,
            sector_revision=sector_revision,
        )
    )
    methods[JOINT_PRIMARY_ALPHA_TARGET_RECIPE_ID] = CanonicalTargetMethod(
        recipe=build_joint_primary_alpha_target_recipe(
            execution_outcome_recipe_id=execution_outcome_recipe_id,
            sector_revision=sector_revision,
        )
    )
    methods[UNBOUNDED_SENSITIVITY_TARGET_RECIPE_ID] = UnboundedSensitivityTargetMethod(
        recipe=build_unbounded_sensitivity_target_recipe(
            execution_outcome_recipe_id=execution_outcome_recipe_id,
            sector_revision=sector_revision,
        )
    )
    methods[TOTAL_RETURN_ALPHA_TARGET_RECIPE_ID] = TotalReturnTargetMethod(
        recipe=build_total_return_alpha_target_recipe(
            execution_outcome_recipe_id=execution_outcome_recipe_id
        )
    )
    return AlphaTargetMethodCatalog(methods)


def resolve_installed_alpha_target_method(
    *,
    target_recipe_id: str,
    execution_outcome_recipe_id: str,
    sector_revision: str | None,
    sector_history_treatment: SectorHistoryTreatment = SECTOR_HISTORY_BACKFILLED,
) -> AlphaTargetMethod:
    """Resolve one installed target without inventing irrelevant authority."""
    if target_recipe_id == TOTAL_RETURN_ALPHA_TARGET_RECIPE_ID:
        return TotalReturnTargetMethod(
            recipe=build_total_return_alpha_target_recipe(
                execution_outcome_recipe_id=execution_outcome_recipe_id
            )
        )
    if sector_revision is None:
        raise AlphaTargetBoundaryError("alpha_research.sector_bound_target_authority_missing")
    return installed_alpha_target_methods(
        sector_revision=sector_revision,
        execution_outcome_recipe_id=execution_outcome_recipe_id,
        sector_history_treatment=sector_history_treatment,
    ).resolve(target_recipe_id)


__all__ = [
    "AlphaTargetMethod",
    "AlphaTargetMethodBinding",
    "AlphaTargetMethodCatalog",
    "CanonicalTargetMethod",
    "InstalledAlphaTargetMethodBinding",
    "InstalledLaneTargetMethod",
    "TotalReturnTargetMethod",
    "UnboundedSensitivityTargetMethod",
    "WholeUniverseAlphaTargetMethodBinding",
    "installed_alpha_target_methods",
    "resolve_installed_alpha_target_method",
]
