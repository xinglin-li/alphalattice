"""The public Risk representation: one factor-plus-idiosyncratic surface, two consumers.

The installed Risk package is a dense-covariance research runtime. Its catalog
holds three matrix estimators, its durable surface persists packed dense
matrices, and its current projection is pinned to the historical R0 recipe.
Nothing in it publishes factor exposures, factor covariance, idiosyncratic
variance, or the two distinct projections the public desktop needs, so this is a
new typed owner rather than an old capability under a new label.

The representation is

    Sigma_public = (G B) F (G B)' + G E G

where ``B`` is the admitted exposure axis, ``F`` the factor covariance, ``E`` the
diagonal idiosyncratic variance, and ``G`` the diagonal rescaling that puts the
surface on the conditional-volatility clock.

**Two consumers, one source, deliberately distinct identities.** The policy reads
only per-name scale; reports read the factor and industry structure. They share
verified lineage but never authority, which is what lets a report-only change to
the attribution rotate its own identity without touching a sealed holdings
ledger.

**No dense N x N is ever materialised.** Every quantity the product needs is a
quadratic form or a diagonal, and both are computable from ``B``, ``F``, ``E``
and ``G`` directly at `O(nk + k^2)`. A five-hundred name book would otherwise
carry a 250,000-entry matrix to answer questions that never required one.

Estimation is owned by ``surfaces.producer``.  The recipe below binds its exact
implementation closure rather than asking a consumer to reconstruct an
algorithm from labels.  This module remains the representation/projection owner;
the producer is the only place that turns returns plus Foundation classification
into these arrays.

On that prototype specifically: it supports the diagonal and book-calibration
part of the idea, and its GMV realized-over-predicted ratio of `4.0355` did
**not** meet its own pre-registered `< 3` threshold. Nothing here claims
otherwise, and nothing here is validated for matrix inversion, minimum variance
or QP use.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Final, Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]

RISK_DECOMPOSITION_RECIPE_ID: Final = "SECTOR_FACTOR_IDIOSYNCRATIC_EWMA_RESCALED"
RISK_IMPLEMENTATION_SOURCE_COMMIT: Final = "d6fd71854e01a15d057fb5162891afa87bb4355e"
RISK_IMPLEMENTATION_SOURCE_BLOB: Final = "df83971dfc3df3d836b4c604bb83a96335e90bad"

RECONSTRUCTION_TOLERANCE: Final = 1e-9
"""Relative tolerance for the diagonal identity below.

Loose enough for float64 accumulation over a few hundred names and a dozen
factors, tight enough that a wrong ``G`` cannot hide behind it.
"""

PSD_EIGENVALUE_TOLERANCE: Final = -1e-12
"""How negative the smallest factor eigenvalue may be before the surface is refused.

Only float round-trip noise is admitted. A genuinely indefinite factor block
produces negative portfolio variance for some long-only book, and a risk model
that can report a negative variance is not a risk model.
"""

_VARIANCE_FLOOR: Final = 1e-16


class RiskDecompositionError(ValueError):
    """Stable refusal for a Risk decomposition identity or consistency failure."""


def _array_hash(values: npt.NDArray[np.generic], *, dtype: npt.DTypeLike) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def _frozen(values: npt.ArrayLike) -> FloatArray:
    """Own the bytes, then make them unwritable.

    ``np.asarray`` keeps the caller's buffer, so a surface built from it stays
    live to whatever the caller does next: mutate the exposures afterwards and
    the surface changes while its content hash does not. That is the opposite of
    a content-addressed artifact, so every array is copied on the way in and
    sealed against later writes.
    """

    owned = np.array(values, dtype=np.float64, copy=True)
    # Backed by ``bytes``, which is immutable at the object level and therefore
    # cannot be reopened at any depth.
    #
    # Two weaker attempts came first and both failed to a probe.
    # ``setflags(write=False)`` on an array that owns its buffer is reversible by
    # whoever holds it. Handing out a read-only *view* of such an array only
    # moves the problem: the view refuses ``setflags(write=True)``, but its
    # ``.base`` is the owning array, and thawing the base thaws the view with it.
    # A ``bytes`` buffer ends that chain, because the terminal base is not an
    # ndarray and has no flags to set.
    frozen: FloatArray = np.frombuffer(owned.tobytes(), dtype=np.float64).reshape(owned.shape)
    return frozen


class RiskDecompositionRecipe(BaseModel):  # type: ignore[misc]
    """The frozen semantics of the public Risk representation.

    Every field is a statement the installed producer executes and verifies.
    The source commit/blob pair is the decision receipt that promotes the exact
    Batch 108 implementation closure; the product never imports that worktree or
    copies its arrays at runtime.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["RiskDecompositionRecipe"] = "RiskDecompositionRecipe"
    recipe_id: Literal["SECTOR_FACTOR_IDIOSYNCRATIC_EWMA_RESCALED"] = RISK_DECOMPOSITION_RECIPE_ID
    return_unit: Literal["ONE_SESSION_OPEN_TO_OPEN_LOG_RETURN"] = (
        "ONE_SESSION_OPEN_TO_OPEN_LOG_RETURN"
    )
    conditional_volatility_decay: float = 0.94
    conditional_volatility_initialization_sessions: int = 63
    factor_fit_sessions: int = 504
    factor_fit_weighting: Literal["EQUAL_WEIGHT"] = "EQUAL_WEIGHT"
    residual_standardization: Literal["EWMA_CONDITIONAL_VOLATILITY"] = "EWMA_CONDITIONAL_VOLATILITY"
    exposure_construction: Literal["ONE_HOT_FOUNDATION_SECTOR_KEY"] = (
        "ONE_HOT_FOUNDATION_SECTOR_KEY"
    )
    factor_return_estimator: Literal["EQUAL_WEIGHT_SECTOR_MEAN"] = "EQUAL_WEIGHT_SECTOR_MEAN"
    factor_covariance_estimator: Literal["SAMPLE_COVARIANCE_DDOF_1"] = "SAMPLE_COVARIANCE_DDOF_1"
    missing_data_policy: Literal["ZERO_RESIDUAL_EWMA_STATE_HOLD"] = "ZERO_RESIDUAL_EWMA_STATE_HOLD"
    residual_clip_bound: float = 10.0
    idiosyncratic_estimator: Literal["SUM_SQUARES_OVER_ROWS_MINUS_SECTORS"] = (
        "SUM_SQUARES_OVER_ROWS_MINUS_SECTORS"
    )
    variance_floor: float = 1e-16
    implementation_source_commit: Literal["d6fd71854e01a15d057fb5162891afa87bb4355e"] = (
        RISK_IMPLEMENTATION_SOURCE_COMMIT
    )
    implementation_source_blob: Literal["df83971dfc3df3d836b4c604bb83a96335e90bad"] = (
        RISK_IMPLEMENTATION_SOURCE_BLOB
    )
    dense_materialization: Literal["NOT_REQUIRED"] = "NOT_REQUIRED"
    classification_authority: Literal["FOUNDATION_SECTOR_REVISION_MAP"] = (
        "FOUNDATION_SECTOR_REVISION_MAP"
    )
    """Industry membership is Data/Foundation authority, read rather than owned.

    Named here so that a reader cannot mistake the factor axis for a Sector
    Forecast capability. Risk consumes the published classification; it does not
    forecast one, and no Sector module is implied by this recipe.
    """

    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def installed(cls) -> Self:
        """Seal the installed fixed decomposition math and source identities.

        Returns:
            Frozen installed recipe with its declared windows, clip/floor rules and canonical hash.
        """
        values: dict[str, object] = {
            "kind": "RiskDecompositionRecipe",
            "recipe_id": RISK_DECOMPOSITION_RECIPE_ID,
            "return_unit": "ONE_SESSION_OPEN_TO_OPEN_LOG_RETURN",
            "conditional_volatility_decay": 0.94,
            "conditional_volatility_initialization_sessions": 63,
            "factor_fit_sessions": 504,
            "factor_fit_weighting": "EQUAL_WEIGHT",
            "residual_standardization": "EWMA_CONDITIONAL_VOLATILITY",
            "exposure_construction": "ONE_HOT_FOUNDATION_SECTOR_KEY",
            "factor_return_estimator": "EQUAL_WEIGHT_SECTOR_MEAN",
            "factor_covariance_estimator": "SAMPLE_COVARIANCE_DDOF_1",
            "missing_data_policy": "ZERO_RESIDUAL_EWMA_STATE_HOLD",
            "residual_clip_bound": 10.0,
            "idiosyncratic_estimator": "SUM_SQUARES_OVER_ROWS_MINUS_SECTORS",
            "variance_floor": 1e-16,
            "implementation_source_commit": RISK_IMPLEMENTATION_SOURCE_COMMIT,
            "implementation_source_blob": RISK_IMPLEMENTATION_SOURCE_BLOB,
            "dense_materialization": "NOT_REQUIRED",
            "classification_authority": "FOUNDATION_SECTOR_REVISION_MAP",
        }
        return cls(**values, recipe_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_recipe(self) -> Self:
        """Require admissible windows/decay, fixed clip/floor rules and exact recipe identity.

        Returns:
            This recipe after fixed numerical-rule and recipe_hash validation.

        Raises:
            RiskDecompositionError: Decay/windows, the fixed clip/floor values or recipe_hash
                violate the recipe contract.
        """
        if not 0.0 < self.conditional_volatility_decay < 1.0:
            raise RiskDecompositionError("risk_research.decomposition_decay_invalid")
        if self.conditional_volatility_initialization_sessions < 1 or self.factor_fit_sessions < 1:
            raise RiskDecompositionError("risk_research.decomposition_window_invalid")
        if self.residual_clip_bound != 10.0 or self.variance_floor != _VARIANCE_FLOOR:
            raise RiskDecompositionError("risk_research.decomposition_estimator_not_frozen")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"recipe_hash"}))
        if self.recipe_hash != expected:
            raise RiskDecompositionError("risk_research.decomposition_recipe_identity_invalid")
        return self


@dataclass(frozen=True, slots=True)
class FactorIdiosyncraticRiskSurface:
    """One formation's factor-plus-idiosyncratic representation, content addressed.

    The arrays arrive from an admitted producer. What this type guarantees is
    that they are mutually consistent, and in particular that the rescaling
    really does put the surface on the conditional-volatility clock it claims.
    """

    recipe_hash: str
    formation_session: date
    ordered_listing_ids: tuple[str, ...]
    ordered_factor_ids: tuple[str, ...]
    exposures: FloatArray
    factor_covariance: FloatArray
    idiosyncratic_variance: FloatArray
    conditional_volatility: FloatArray
    rescaling: FloatArray
    producer_identity: str
    surface_hash: str

    @classmethod
    def create(
        cls,
        *,
        recipe: RiskDecompositionRecipe,
        formation_session: date,
        ordered_listing_ids: tuple[str, ...],
        ordered_factor_ids: tuple[str, ...],
        exposures: FloatArray,
        factor_covariance: FloatArray,
        idiosyncratic_variance: FloatArray,
        conditional_volatility: FloatArray,
        producer_identity: str,
    ) -> Self:
        """Derive ``G`` from the arrays rather than accepting one.

        ``G`` is not independent information: it is exactly the ratio that moves
        the base representation onto the conditional-volatility clock. Accepting
        a supplied one would let a caller assert a rescaling the surface does not
        actually have, and the diagonal check below would then be validating the
        caller's arithmetic instead of the surface.
        """
        base = _base_diagonal(exposures, factor_covariance, idiosyncratic_variance)
        if np.any(base <= _VARIANCE_FLOOR):
            raise RiskDecompositionError("risk_research.decomposition_base_variance_degenerate")
        rescaling = np.asarray(
            np.asarray(conditional_volatility, dtype=np.float64) / np.sqrt(base),
            dtype=np.float64,
        )
        surface_hash = canonical_hash(
            {
                "kind": "FactorIdiosyncraticRiskSurface",
                "recipe_hash": recipe.recipe_hash,
                "formation_session": formation_session.isoformat(),
                "ordered_listing_ids": list(ordered_listing_ids),
                "ordered_factor_ids": list(ordered_factor_ids),
                "exposures": _array_hash(exposures, dtype="<f8"),
                "factor_covariance": _array_hash(factor_covariance, dtype="<f8"),
                "idiosyncratic_variance": _array_hash(idiosyncratic_variance, dtype="<f8"),
                "conditional_volatility": _array_hash(conditional_volatility, dtype="<f8"),
                "rescaling": _array_hash(rescaling, dtype="<f8"),
                "producer_identity": producer_identity,
            }
        )
        surface = cls(
            recipe_hash=recipe.recipe_hash,
            formation_session=formation_session,
            ordered_listing_ids=ordered_listing_ids,
            ordered_factor_ids=ordered_factor_ids,
            exposures=_frozen(exposures),
            factor_covariance=_frozen(factor_covariance),
            idiosyncratic_variance=_frozen(idiosyncratic_variance),
            conditional_volatility=_frozen(conditional_volatility),
            rescaling=_frozen(rescaling),
            producer_identity=producer_identity,
            surface_hash=surface_hash,
        )
        surface.validate_surface()
        return surface

    def validate_surface(self) -> None:
        """Shapes, finiteness, symmetry -- then the identity that matters."""
        names = len(self.ordered_listing_ids)
        factors = len(self.ordered_factor_ids)
        if names == 0 or factors == 0:
            raise RiskDecompositionError("risk_research.decomposition_axis_empty")
        if len(set(self.ordered_listing_ids)) != names or len(set(self.ordered_factor_ids)) != (
            factors
        ):
            raise RiskDecompositionError("risk_research.decomposition_axis_duplicated")
        if (
            self.exposures.shape != (names, factors)
            or self.factor_covariance.shape != (factors, factors)
            or self.idiosyncratic_variance.shape != (names,)
            or self.conditional_volatility.shape != (names,)
            or self.rescaling.shape != (names,)
        ):
            raise RiskDecompositionError("risk_research.decomposition_shape_invalid")
        for values in (
            self.exposures,
            self.factor_covariance,
            self.idiosyncratic_variance,
            self.conditional_volatility,
            self.rescaling,
        ):
            if not np.isfinite(values).all():
                raise RiskDecompositionError("risk_research.decomposition_values_not_finite")
        if not np.allclose(self.factor_covariance, self.factor_covariance.T, atol=1e-12):
            raise RiskDecompositionError("risk_research.decomposition_factor_covariance_asymmetric")
        # Symmetry alone admits an indefinite block. An indefinite F gives some
        # long-only book a negative variance, which then either surfaces as a
        # nonsensical number or gets clamped to zero and reports a risky book as
        # riskless. Neither is acceptable, so it is refused at admission.
        if float(np.linalg.eigvalsh(self.factor_covariance).min()) < PSD_EIGENVALUE_TOLERANCE:
            raise RiskDecompositionError(
                "risk_research.decomposition_factor_covariance_not_positive_semidefinite"
            )
        if np.any(self.idiosyncratic_variance < 0.0):
            raise RiskDecompositionError("risk_research.decomposition_idiosyncratic_negative")
        if np.any(self.conditional_volatility <= 0.0):
            raise RiskDecompositionError("risk_research.decomposition_volatility_not_positive")
        self.verify_reconstruction()

    def verify_reconstruction(self) -> None:
        """The one check that proves both projections come from the same surface.

        By construction ``g_i = sigma_i / sqrt((B F B' + E)_ii)``, so the public
        representation's own diagonal must return exactly ``sigma_i^2``:

            diag((G B) F (G B)' + G E G)_i = g_i^2 * ((B F B')_ii + E_i)
                                           = sigma_i^2

        That makes the policy's per-name scale and the report's factor structure
        two readings of one object rather than two numbers that happen to agree.
        If a producer ever supplied a rescaling inconsistent with its own factor
        block, this is where it fails -- before any weight is decided.
        """
        rebuilt = self.per_name_variance()
        expected = np.square(self.conditional_volatility)
        if not np.allclose(rebuilt, expected, rtol=RECONSTRUCTION_TOLERANCE, atol=0.0):
            raise RiskDecompositionError("risk_research.decomposition_reconstruction_failed")

    def verify_content(self) -> None:
        """Recompute the content hash and compare, for a readback boundary.

        Freezing the arrays stops accidental mutation; this catches a surface
        that was assembled or restored with contents its hash does not describe.
        """
        expected = canonical_hash(
            {
                "kind": "FactorIdiosyncraticRiskSurface",
                "recipe_hash": self.recipe_hash,
                "formation_session": self.formation_session.isoformat(),
                "ordered_listing_ids": list(self.ordered_listing_ids),
                "ordered_factor_ids": list(self.ordered_factor_ids),
                "exposures": _array_hash(self.exposures, dtype="<f8"),
                "factor_covariance": _array_hash(self.factor_covariance, dtype="<f8"),
                "idiosyncratic_variance": _array_hash(self.idiosyncratic_variance, dtype="<f8"),
                "conditional_volatility": _array_hash(self.conditional_volatility, dtype="<f8"),
                "rescaling": _array_hash(self.rescaling, dtype="<f8"),
                "producer_identity": self.producer_identity,
            }
        )
        if expected != self.surface_hash:
            raise RiskDecompositionError("risk_research.decomposition_content_hash_mismatch")
        # The hash proves the bytes are the ones recorded. It does not prove the
        # derived block is the one those bytes imply, so G is rebuilt from the
        # factor and idiosyncratic blocks and compared, and the diagonal identity
        # is re-checked. Without this a readback could restore a surface whose
        # rescaling was replaced wholesale and whose predicted variances are
        # therefore meaningless while every recorded hash still matches.
        rebuilt = np.asarray(
            self.conditional_volatility
            / np.sqrt(
                _base_diagonal(self.exposures, self.factor_covariance, self.idiosyncratic_variance)
            ),
            dtype=np.float64,
        )
        if not np.allclose(rebuilt, self.rescaling, rtol=RECONSTRUCTION_TOLERANCE, atol=0.0):
            raise RiskDecompositionError("risk_research.decomposition_rescaling_not_derived")
        self.verify_reconstruction()

    @property
    def scaled_exposures(self) -> FloatArray:
        """``G B``, the exposures on the conditional-volatility clock."""
        return np.asarray(self.rescaling[:, None] * self.exposures, dtype=np.float64)

    def per_name_variance(self) -> FloatArray:
        """``diag(Sigma_public)`` without materialising ``Sigma_public``."""
        scaled = self.scaled_exposures
        systematic = np.einsum("ij,jk,ik->i", scaled, self.factor_covariance, scaled)
        specific = np.square(self.rescaling) * self.idiosyncratic_variance
        return np.asarray(systematic + specific, dtype=np.float64)

    def book_variance(self, weights: FloatArray) -> float:
        """``w' Sigma_public w`` at `O(nk + k^2)`, never `O(n^2)`.

        A negative total is refused rather than floored. PSD admission should
        make it unreachable, and if it ever is reached the honest answer is that
        the surface is wrong -- silently returning zero would publish a riskless
        reading for a book that is not.
        """
        systematic, specific = self.variance_split(weights)
        total = systematic + specific
        if total < 0.0:
            raise RiskDecompositionError("risk_research.decomposition_book_variance_negative")
        return total

    def variance_split(self, weights: FloatArray) -> tuple[float, float]:
        """The systematic and specific halves the Risk page reports."""
        values = np.asarray(weights, dtype=np.float64)
        if values.shape != (len(self.ordered_listing_ids),):
            raise RiskDecompositionError("risk_research.decomposition_weight_axis_invalid")
        factor_loadings = self.scaled_exposures.T @ values
        systematic = float(factor_loadings @ self.factor_covariance @ factor_loadings)
        specific = float(np.square(self.rescaling * values) @ self.idiosyncratic_variance)
        return systematic, specific

    def industry_exposure(self, weights: FloatArray) -> FloatArray:
        """The book's exposure to each admitted factor, for the Risk page.

        Explanatory only. It is not an instruction to rotate, and the record is
        explicit that sector-neutralising this score removes the book's entire
        alpha.
        """
        values = np.asarray(weights, dtype=np.float64)
        if values.shape != (len(self.ordered_listing_ids),):
            raise RiskDecompositionError("risk_research.decomposition_weight_axis_invalid")
        return np.asarray(self.exposures.T @ values, dtype=np.float64)

    def factor_variance_contribution(self, weights: FloatArray) -> FloatArray:
        """Euler contribution of each factor to systematic variance."""
        values = np.asarray(weights, dtype=np.float64)
        if values.shape != (len(self.ordered_listing_ids),):
            raise RiskDecompositionError("risk_research.decomposition_weight_axis_invalid")
        loadings = self.scaled_exposures.T @ values
        return np.asarray(loadings * (self.factor_covariance @ loadings), dtype=np.float64)

    def variance_shares(self, weights: FloatArray) -> FloatArray:
        """Each name's Euler share of the book variance: its weight times `Sigma w` over the total.

        The shares sum to one; a hedge's share is negative.
        """
        values = np.asarray(weights, dtype=np.float64)
        loadings = self.scaled_exposures.T @ values
        marginal = self.scaled_exposures @ (self.factor_covariance @ loadings)
        marginal = marginal + np.square(self.rescaling) * self.idiosyncratic_variance * values
        return np.asarray(values * marginal / self.book_variance(values), dtype=np.float64)


def _base_diagonal(
    exposures: FloatArray,
    factor_covariance: FloatArray,
    idiosyncratic_variance: FloatArray,
) -> FloatArray:
    """``diag(B F B' + E)`` before any rescaling."""

    loadings = np.asarray(exposures, dtype=np.float64)
    factors = np.asarray(factor_covariance, dtype=np.float64)
    specific = np.asarray(idiosyncratic_variance, dtype=np.float64)
    if loadings.ndim != 2 or factors.ndim != 2 or specific.ndim != 1:
        raise RiskDecompositionError("risk_research.decomposition_shape_invalid")
    if loadings.shape[1] != factors.shape[0] or loadings.shape[0] != specific.size:
        raise RiskDecompositionError("risk_research.decomposition_shape_invalid")
    systematic = np.einsum("ij,jk,ik->i", loadings, factors, loadings)
    return np.asarray(systematic + specific, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class RiskAllocationProjection:
    """Exactly what the installed closed-form policy consumes, and nothing else.

    Deliberately narrow. A projection that also carried the factor block would
    let a report-only change to exposures or factor covariance invalidate a
    sealed holdings ledger, which is the coupling the delivery plan separates
    these two projections to prevent.
    """

    surface_hash: str
    """Lineage, deliberately *not* part of this projection's identity.

    Recorded so the source is always reopenable, and excluded from the hash
    below so that a report-only change to the factor block does not rotate a
    projection whose consumed values did not move.
    """

    recipe_hash: str
    formation_session: date
    ordered_listing_ids: tuple[str, ...]
    per_name_volatility: FloatArray
    projection_hash: str

    @classmethod
    def of(cls, surface: FactorIdiosyncraticRiskSurface) -> Self:
        # Read the conditional-volatility lane directly rather than rebuilding it
        # through the factor block. The two are equal by the surface's own
        # reconstruction check, but they are not equal *bit for bit*: recomputing
        # routes the value through B, F and E, so a report-only revision to any
        # of them moves the last bits and rotates this identity for a value that
        # did not change. The check proves the agreement; the lane supplies it.
        """Project the direct conditional-volatility lane for portfolio allocation.

        Allocation identity binds recipe, formation, listing axis and per-name volatility content.
        The full surface identity is retained as lineage and is excluded from allocation identity;
        factor components do not reconstruct this lane.

        Args:
            surface: Verified factor/idiosyncratic surface with direct conditional volatility.

        Returns:
            Content-verified allocation projection using the surface volatility lane.

        Raises:
            RiskDecompositionError: The resulting projection content is inconsistent.
        """
        volatility = np.asarray(surface.conditional_volatility, dtype=np.float64)
        # Identity covers the recipe, the axis and the consumed values -- nothing
        # else. Including the surface hash would have made every exposure or
        # factor-covariance revision look like a new holdings input, which is the
        # coupling these two projections exist to break.
        projection_hash = canonical_hash(
            {
                "kind": "RiskAllocationProjection",
                "recipe_hash": surface.recipe_hash,
                "formation_session": surface.formation_session.isoformat(),
                "ordered_listing_ids": list(surface.ordered_listing_ids),
                "per_name_volatility": _array_hash(volatility, dtype="<f8"),
            }
        )
        projection = cls(
            surface_hash=surface.surface_hash,
            recipe_hash=surface.recipe_hash,
            formation_session=surface.formation_session,
            ordered_listing_ids=surface.ordered_listing_ids,
            per_name_volatility=volatility,
            projection_hash=projection_hash,
        )
        projection.verify_content()
        return projection

    def verify_content(self) -> None:
        """Refuse a readback whose values or axis do not match its identity."""
        hashes = (self.surface_hash, self.recipe_hash, self.projection_hash)
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise RiskDecompositionError("risk_research.allocation_projection_hash_invalid")
        names = len(self.ordered_listing_ids)
        if names == 0 or len(set(self.ordered_listing_ids)) != names:
            raise RiskDecompositionError("risk_research.allocation_projection_axis_invalid")
        volatility = np.asarray(self.per_name_volatility, dtype=np.float64)
        if volatility.shape != (names,):
            raise RiskDecompositionError("risk_research.allocation_projection_shape_invalid")
        if not np.isfinite(volatility).all() or np.any(volatility <= 0.0):
            raise RiskDecompositionError("risk_research.allocation_projection_values_invalid")
        expected = canonical_hash(
            {
                "kind": "RiskAllocationProjection",
                "recipe_hash": self.recipe_hash,
                "formation_session": self.formation_session.isoformat(),
                "ordered_listing_ids": list(self.ordered_listing_ids),
                "per_name_volatility": _array_hash(volatility, dtype="<f8"),
            }
        )
        if expected != self.projection_hash:
            raise RiskDecompositionError("risk_research.allocation_projection_content_mismatch")

    def diagonal_covariance(self) -> FloatArray:
        """The per-name variances a diagonal-consuming policy reads.

        Returned as a vector. The policy needs a diagonal, and handing it a
        square matrix would invite a caller to read an off-diagonal that this
        projection does not carry and the public path never validated.
        """
        return np.asarray(np.square(self.per_name_volatility), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class RiskAttributionProjection:
    """The factor and industry facts a report consumes. Never a holdings input."""

    surface_hash: str
    recipe_hash: str
    formation_session: date
    ordered_listing_ids: tuple[str, ...]
    ordered_factor_ids: tuple[str, ...]
    classification_authority: str
    projection_hash: str
    _surface: FactorIdiosyncraticRiskSurface

    @classmethod
    def of(
        cls,
        surface: FactorIdiosyncraticRiskSurface,
        *,
        classification_authority: str = "FOUNDATION_SECTOR_REVISION_MAP",
    ) -> Self:
        """Project the full surface and classification authority for portfolio attribution.

        Args:
            surface: Factor/idiosyncratic surface retained for attribution calculations.
            classification_authority: Declared classification owner used by the attribution
                projection.

        Returns:
            Attribution projection binding full surface content, factor axis and classification
            authority.
        """
        projection_hash = canonical_hash(
            {
                "kind": "RiskAttributionProjection",
                "surface_hash": surface.surface_hash,
                "classification_authority": classification_authority,
                "ordered_factor_ids": list(surface.ordered_factor_ids),
            }
        )
        return cls(
            surface_hash=surface.surface_hash,
            recipe_hash=surface.recipe_hash,
            formation_session=surface.formation_session,
            ordered_listing_ids=surface.ordered_listing_ids,
            ordered_factor_ids=surface.ordered_factor_ids,
            classification_authority=classification_authority,
            projection_hash=projection_hash,
            _surface=surface,
        )

    def variance_split(self, weights: FloatArray) -> tuple[float, float]:
        """Split portfolio variance into factor and idiosyncratic contributions.

        Args:
            weights: Portfolio weights on the retained surface ordered listing axis.

        Returns:
            Factor variance and idiosyncratic variance from the retained surface.
        """
        return self._surface.variance_split(weights)

    def industry_exposure(self, weights: FloatArray) -> FloatArray:
        """Compute portfolio industry-factor exposure from the retained surface.

        Args:
            weights: Portfolio weights on the retained surface ordered listing axis.

        Returns:
            Exposure vector in the declared factor axis order.
        """
        return self._surface.industry_exposure(weights)

    def factor_variance_contribution(self, weights: FloatArray) -> FloatArray:
        """Compute each factor contribution to portfolio variance.

        Args:
            weights: Portfolio weights on the retained surface ordered listing axis.

        Returns:
            Per-factor contributions in the declared factor axis order.
        """
        return self._surface.factor_variance_contribution(weights)

    def total_predicted_volatility(self, weights: FloatArray) -> float:
        """No clamp. ``book_variance`` refuses a negative rather than hiding one."""
        return float(np.sqrt(self._surface.book_variance(weights)))


INSTALLED_RISK_DECOMPOSITION_RECIPE: Final = RiskDecompositionRecipe.installed()
"""The one public Risk identity. Composition injects it; nothing discovers it."""


__all__ = [
    "INSTALLED_RISK_DECOMPOSITION_RECIPE",
    "PSD_EIGENVALUE_TOLERANCE",
    "RECONSTRUCTION_TOLERANCE",
    "RISK_DECOMPOSITION_RECIPE_ID",
    "FactorIdiosyncraticRiskSurface",
    "RiskAllocationProjection",
    "RiskAttributionProjection",
    "RiskDecompositionError",
    "RiskDecompositionRecipe",
]
