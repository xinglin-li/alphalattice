"""The explicitly installed Panel preprocessing methods.

Explicit construction only: no filesystem discovery, no entry points, no
dynamic import. Installing a method is an entry in
``build_installed_panel_preprocessing_catalog`` and nothing else, so the set of
transformations a Panel can be built by is readable in one place and changes
only when someone edits it.

The Feature build resolves a method here and then executes the kernel. Neither
the Host nor the Panel publisher branches on a recipe id: which method ran is
carried by the binding and the evidence, never by a conditional in a writer.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import cast

from alphalattice.foundation.feature_engine.producers.preprocessing.adapters import (
    PanelPreprocessingAdapter,
    RobustSectorNeutralZAdapter,
    StateInteractionBlockAdapter,
    TimeSeriesAbsoluteStateRobustAdapter,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.contracts import (
    PanelPreprocessingCapability,
    PanelPreprocessingError,
    PanelPreprocessingImplementationBinding,
    PanelPreprocessingRecipe,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.joint_primary import (
    JointPrimaryRelativeFactorStdZAdapter,
    JointPrimaryStateInteractionBlockAdapter,
    render_formula_pretransform_step,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.robust_universe import (
    RobustUniverseZAdapter,
)
from alphalattice.kernel.quant.cross_section import (
    MAD_SCALE,
    MIN_COVERAGE,
    MIN_SECTOR_SAMPLE,
    WINSOR_MULTIPLIER,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.identity_successors import is_current

ROBUST_SECTOR_NEUTRAL_Z = "ROBUST_SECTOR_NEUTRAL_Z"
ROBUST_UNIVERSE_Z = "ROBUST_UNIVERSE_Z"
TIME_SERIES_ABSOLUTE_STATE_ROBUST = "TIME_SERIES_ABSOLUTE_STATE_ROBUST"
STATE_INTERACTION_BLOCK = "STATE_INTERACTION_BLOCK"
JOINT_PRIMARY_RELATIVE_FACTOR_STD_Z = "JOINT_PRIMARY_RELATIVE_FACTOR_STD_Z"
JOINT_PRIMARY_STATE_INTERACTION_BLOCK = "JOINT_PRIMARY_STATE_INTERACTION_BLOCK"

PREPROCESSING_CATALOG_ROLE = "feature_engine.preprocessing_catalog"
"""The identity role of the installed catalog's hash (`config/identity-roles.json`)."""


def implementation_role(recipe_id: str) -> str:
    """The identity role of one installed recipe's implementation binding."""
    return f"feature_engine.preprocessing_implementation.{recipe_id}"


def build_robust_sector_neutral_z_recipe() -> PanelPreprocessingRecipe:
    """The one method the Panel has always run, now stated rather than implied.

    Every constant is read from the kernel rather than respelled here. A second
    spelling that drifted by one digit would describe a transformation the code
    does not perform, which is precisely the failure this contract exists to
    make impossible.
    """
    values: dict[str, object] = {
        "kind": "PanelPreprocessingRecipe",
        "recipe_id": ROBUST_SECTOR_NEUTRAL_Z,
        "sequence": ("median_mad_winsor", "equal_sector_demean", "global_robust_zscore"),
        "winsor_multiplier": WINSOR_MULTIPLIER,
        "mad_scale": MAD_SCALE,
        "minimum_coverage": MIN_COVERAGE,
        "minimum_sector_sample": MIN_SECTOR_SAMPLE,
        "neutralization": "EQUAL_SECTOR_DEMEAN",
        "standardization": "GLOBAL_ROBUST_Z",
        "lookback_sessions": None,
        "minimum_finite_observations": None,
        "interaction_clip": None,
        "partial_universe": "fail_closed",
    }
    return PanelPreprocessingRecipe(**values, recipe_hash=canonical_hash(values))


def build_robust_universe_z_recipe() -> PanelPreprocessingRecipe:
    """The universe-centred robust Z: the Panel kernel with the universe as its one group.

    For a return, a price move or a move size, which has no Sector level to remove: the same
    winsor and global robust Z as the sector-neutral method, the demean the universe's, so no
    classification reaches its values. The constants are the kernel's, as that method's are.

    Returns:
        The recipe, identified by its content.
    """
    values: dict[str, object] = {
        "kind": "PanelPreprocessingRecipe",
        "recipe_id": ROBUST_UNIVERSE_Z,
        "sequence": ("median_mad_winsor", "equal_universe_demean", "global_robust_zscore"),
        "winsor_multiplier": WINSOR_MULTIPLIER,
        "mad_scale": MAD_SCALE,
        "minimum_coverage": MIN_COVERAGE,
        "minimum_sector_sample": MIN_SECTOR_SAMPLE,
        "neutralization": "NONE",
        "standardization": "GLOBAL_ROBUST_Z",
        "lookback_sessions": None,
        "minimum_finite_observations": None,
        "interaction_clip": None,
        "partial_universe": "fail_closed",
    }
    return PanelPreprocessingRecipe(**values, recipe_hash=canonical_hash(values))


def build_joint_primary_relative_factor_std_z_recipe() -> PanelPreprocessingRecipe:
    """Build the installed joint-primary relative standardization recipe."""
    values: dict[str, object] = {
        "kind": "PanelPreprocessingRecipe",
        "recipe_id": JOINT_PRIMARY_RELATIVE_FACTOR_STD_Z,
        "sequence": (
            render_formula_pretransform_step(),
            "universe_normalized_mad_winsor_3_5",
            "equal_sector_demean",
            "cross_sectional_sample_std_z_ddof_1",
        ),
        "winsor_multiplier": 3.5,
        "mad_scale": 1.4826,
        "minimum_coverage": MIN_COVERAGE,
        "minimum_sector_sample": MIN_SECTOR_SAMPLE,
        "neutralization": "EQUAL_SECTOR_DEMEAN",
        "standardization": "CROSS_SECTIONAL_STD_Z",
        "lookback_sessions": None,
        "minimum_finite_observations": None,
        "interaction_clip": None,
        "partial_universe": "fail_closed",
    }
    return PanelPreprocessingRecipe(**values, recipe_hash=canonical_hash(values))


def build_time_series_absolute_state_robust_recipe() -> PanelPreprocessingRecipe:
    """Build the installed robust absolute-state time-series recipe."""
    values: dict[str, object] = {
        "kind": "PanelPreprocessingRecipe",
        "recipe_id": TIME_SERIES_ABSOLUTE_STATE_ROBUST,
        "sequence": ("trailing_252_finite", "median_mad_winsor_current", "listing_robust_z"),
        "winsor_multiplier": 5.0,
        "mad_scale": 1.4826,
        "minimum_coverage": 0.0 + 1e-12,
        "minimum_sector_sample": 0,
        "neutralization": "NONE",
        "standardization": "TRAILING_LISTING_ROBUST_Z",
        "lookback_sessions": 252,
        "minimum_finite_observations": 126,
        "interaction_clip": None,
        "partial_universe": "fail_closed",
    }
    return PanelPreprocessingRecipe(**values, recipe_hash=canonical_hash(values))


def build_state_interaction_block_recipe() -> PanelPreprocessingRecipe:
    """Build the installed state-interaction preprocessing recipe."""
    values: dict[str, object] = {
        "kind": "PanelPreprocessingRecipe",
        "recipe_id": STATE_INTERACTION_BLOCK,
        "sequence": (
            "verified_cross_sectional_child",
            "verified_market_state_child",
            "product",
            "clip_5",
        ),
        "winsor_multiplier": 5.0,
        "mad_scale": 1.4826,
        "minimum_coverage": 0.98,
        "minimum_sector_sample": 5,
        "neutralization": "NONE",
        "standardization": "NONE",
        "lookback_sessions": None,
        "minimum_finite_observations": None,
        "interaction_clip": 5.0,
        "partial_universe": "fail_closed",
    }
    return PanelPreprocessingRecipe(**values, recipe_hash=canonical_hash(values))


def build_joint_primary_state_interaction_block_recipe() -> PanelPreprocessingRecipe:
    """Build the joint-primary interaction recipe from the verified child recipe."""
    values = build_state_interaction_block_recipe().model_dump(mode="json", exclude={"recipe_hash"})
    values["recipe_id"] = JOINT_PRIMARY_STATE_INTERACTION_BLOCK
    values["sequence"] = (
        "verified_joint_primary_relative_child",
        "verified_market_state_child",
        "product",
        "clip_5",
    )
    return PanelPreprocessingRecipe(**values, recipe_hash=canonical_hash(values))


class PanelPreprocessingCatalog:
    """Immutable index of installed Panel preprocessing methods and their code.

    Each entry pairs a recipe with the adapter that computes it. Resolving
    returns the pair, so a caller cannot obtain a recipe identity without also
    obtaining the implementation bound to it -- which is what stops a build from
    sealing one method and running another.

    The pairing is by *content*, not by name. ``catalog_hash`` folds each
    adapter's implementation binding, so replacing the code behind an unchanged
    ``implementation_id`` rotates catalog identity rather than passing silently.
    Two distinct adapter instances declaring the same content stay equal, because
    what is compared is the declared identity and not the object.
    """

    def __init__(
        self,
        entries: tuple[tuple[PanelPreprocessingRecipe, PanelPreprocessingAdapter], ...],
    ) -> None:
        """Index installed preprocessing recipes and adapters by recipe identifier."""
        indexed = {recipe.recipe_id: (recipe, adapter) for recipe, adapter in entries}
        if not entries or len(indexed) != len(entries):
            raise PanelPreprocessingError("PANEL_PREPROCESSING_CATALOG_INVALID")
        self._entries = MappingProxyType(indexed)
        self._recipes = MappingProxyType({key: value[0] for key, value in indexed.items()})

    @property
    def recipe_ids(self) -> tuple[str, ...]:
        """Return installed preprocessing recipe identifiers in catalog order."""
        return tuple(self._recipes)

    @property
    def catalog_hash(self) -> str:
        """Identity of what is installed -- methods *and* the code that runs them.

        Folding the implementation binding is what makes this more than a list of
        names. Hashing only recipe ids and recipe hashes left an adapter swap
        invisible: the same catalog identity could describe two different
        arithmetics, which is precisely the substitution the catalog exists to
        make impossible.
        """
        return str(
            canonical_hash(
                {
                    "kind": "PanelPreprocessingCatalog",
                    "ordered_recipes": [
                        {
                            "recipe_id": key,
                            "recipe_hash": self._recipes[key].recipe_hash,
                            "implementation_binding_hash": (
                                self._entries[key][1]
                                .describe_implementation_binding()
                                .implementation_binding_hash
                            ),
                        }
                        for key in sorted(self._recipes)
                    ],
                }
            )
        )

    def catalog_current(self, recorded: str) -> bool:
        """Whether a recorded catalog hash names this catalog, by value or recorded moves."""
        return is_current(PREPROCESSING_CATALOG_ROLE, recorded, self.catalog_hash)

    def implementation_current(self, recipe_id: str, recorded: str) -> bool:
        """Whether a recorded implementation binding names this recipe's installed one."""
        installed = self.implementation_binding(recipe_id).implementation_binding_hash
        return is_current(implementation_role(recipe_id), recorded, installed)

    def resolve(self, recipe_id: str) -> PanelPreprocessingRecipe:
        """The only route from an authored id to a transformation.

        Re-parsed from serialized fields rather than returned directly:
        ``model_validate`` on a same-class instance re-runs only the
        after-validators, so a recipe built through ``model_construct`` with an
        out-of-range constant and a recomputed hash would otherwise survive.
        """
        recipe, _adapter = self.resolve_executable(recipe_id)
        return recipe

    def resolve_executable(
        self, recipe_id: str
    ) -> tuple[PanelPreprocessingRecipe, PanelPreprocessingAdapter]:
        """Resolve the recipe together with the implementation that computes it."""
        entry = self._entries.get(recipe_id)
        if entry is None:
            raise PanelPreprocessingError("PANEL_PREPROCESSING_RECIPE_NOT_INSTALLED")
        recipe, adapter = entry
        return (
            PanelPreprocessingRecipe.model_validate(recipe.model_dump(mode="json")),
            adapter,
        )

    def admit_for_active_panel(self, recipe_id: str) -> PanelPreprocessingCapability:
        """Resolve a capability and require it to be admitted for the active Panel.

        The admission is checked here rather than merely reported, because a
        capability flag nobody enforces is documentation. A recipe that is
        installed but not admitted fails before a Panel is built from it.
        """
        capability = self.capability(recipe_id)
        if not capability.admitted_for_active_panel:
            raise PanelPreprocessingError("PANEL_PREPROCESSING_RECIPE_NOT_ADMITTED")
        return capability

    def admit_for_development_overlay(self, recipe_id: str) -> PanelPreprocessingCapability:
        """Return a recipe capability only when development overlays admit it."""
        capability = self.capability(recipe_id)
        if not capability.admitted_for_development_overlay:
            raise PanelPreprocessingError("PANEL_PREPROCESSING_RECIPE_NOT_DEVELOPMENT_ADMITTED")
        return capability

    def capability(self, recipe_id: str) -> PanelPreprocessingCapability:
        """Describe an installed recipe, implementation binding and admission scope."""
        implementation = self.implementation_binding(recipe_id)
        recipe, adapter = self.resolve_executable(recipe_id)
        return PanelPreprocessingCapability(
            recipe_id=recipe.recipe_id,
            recipe_hash=recipe.recipe_hash,
            implementation_id=adapter.implementation_id,
            implementation_binding_hash=implementation.implementation_binding_hash,
            admitted_for_active_panel=recipe.recipe_id == ROBUST_SECTOR_NEUTRAL_Z,
            admitted_for_development_overlay=recipe.recipe_id
            in {
                ROBUST_SECTOR_NEUTRAL_Z,
                ROBUST_UNIVERSE_Z,
                TIME_SERIES_ABSOLUTE_STATE_ROBUST,
                STATE_INTERACTION_BLOCK,
                JOINT_PRIMARY_RELATIVE_FACTOR_STD_Z,
                JOINT_PRIMARY_STATE_INTERACTION_BLOCK,
            },
        )

    def implementation_binding(self, recipe_id: str) -> PanelPreprocessingImplementationBinding:
        """The executable identity bound to one installed recipe.

        Re-parsed from serialized fields for the same reason ``resolve`` is:
        ``model_validate`` on a same-class instance re-runs only the
        after-validators, so an adapter returning a hand-built binding with a
        recomputed hash would otherwise pass unexamined.
        """
        _recipe, adapter = self.resolve_executable(recipe_id)
        declared = adapter.describe_implementation_binding()
        return cast(
            PanelPreprocessingImplementationBinding,
            PanelPreprocessingImplementationBinding.model_validate_json(declared.model_dump_json()),
        )


def installed_implementation_binding_hash(recipe_id: str) -> str:
    """The implementation binding this build installs for one recipe: its role's value."""
    catalog = build_installed_panel_preprocessing_catalog()
    return catalog.implementation_binding(recipe_id).implementation_binding_hash


def build_installed_panel_preprocessing_catalog() -> PanelPreprocessingCatalog:
    """Install exactly the methods this build offers, each with its implementation."""
    return PanelPreprocessingCatalog(
        (
            (build_robust_sector_neutral_z_recipe(), RobustSectorNeutralZAdapter()),
            (build_robust_universe_z_recipe(), RobustUniverseZAdapter()),
            (
                build_time_series_absolute_state_robust_recipe(),
                TimeSeriesAbsoluteStateRobustAdapter(),
            ),
            (build_state_interaction_block_recipe(), StateInteractionBlockAdapter()),
            (
                build_joint_primary_relative_factor_std_z_recipe(),
                JointPrimaryRelativeFactorStdZAdapter(),
            ),
            (
                build_joint_primary_state_interaction_block_recipe(),
                JointPrimaryStateInteractionBlockAdapter(),
            ),
        )
    )


__all__ = [
    "JOINT_PRIMARY_RELATIVE_FACTOR_STD_Z",
    "JOINT_PRIMARY_STATE_INTERACTION_BLOCK",
    "ROBUST_SECTOR_NEUTRAL_Z",
    "ROBUST_UNIVERSE_Z",
    "STATE_INTERACTION_BLOCK",
    "TIME_SERIES_ABSOLUTE_STATE_ROBUST",
    "PanelPreprocessingCatalog",
    "build_installed_panel_preprocessing_catalog",
    "build_joint_primary_relative_factor_std_z_recipe",
    "build_joint_primary_state_interaction_block_recipe",
    "build_robust_sector_neutral_z_recipe",
    "build_robust_universe_z_recipe",
    "build_state_interaction_block_recipe",
    "build_time_series_absolute_state_robust_recipe",
]
