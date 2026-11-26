"""Typed, path-free contracts for the Risk Desk boundary."""

from __future__ import annotations

from datetime import date
from typing import Final, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

Hash = str
_HASH_PATTERN = r"^[0-9a-f]{64}$"


class RiskContract(BaseModel):  # type: ignore[misc]
    """Provide frozen Risk contracts that reject undeclared fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def seal_contract[ContractT: RiskContract](
    model_type: type[ContractT], identity_field: str, /, **values: object
) -> ContractT:
    """Construct one immutable content-addressed Risk contract."""
    draft = model_type.model_construct(**values, **{identity_field: "0" * 64})
    identity = draft.model_dump(mode="json", exclude={identity_field})
    payload = draft.model_dump(exclude={identity_field})
    return cast(
        ContractT,
        model_type.model_validate({**payload, identity_field: canonical_hash(identity)}),
    )


class RiskUniverseEpoch(RiskContract):
    """Bind an ordered current-universe listing axis to its source revision and Panel.

    The contract declares current-universe research validity and a minimum of 315 valid return
    sessions. The axis is unique; its supplied order is preserved.
    """

    kind: Literal["RiskUniverseEpoch"] = "RiskUniverseEpoch"
    market_profile_id: str = Field(min_length=1)
    universe_manifest_revision: Hash = Field(pattern=_HASH_PATTERN)
    panel_snapshot_hash: Hash = Field(pattern=_HASH_PATTERN)
    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)
    data_validity_class: Literal["CURRENT_UNIVERSE_RESEARCH_ONLY"] = (
        "CURRENT_UNIVERSE_RESEARCH_ONLY"
    )
    minimum_valid_return_sessions: Literal[315] = 315
    epoch_hash: Hash = Field(pattern=_HASH_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_epoch(self) -> Self:
        """Require a unique ordered listing axis and exact epoch identity.

        Returns:
            This contract after consistency and applicable identity checks.

        Raises:
            ValueError: A listing repeats or epoch_hash differs from the complete payload.
        """
        if len(self.ordered_listing_ids) != len(set(self.ordered_listing_ids)):
            raise ValueError("risk_research.universe_epoch_axis_invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"epoch_hash"}))
        if self.epoch_hash != expected:
            raise ValueError("risk_research.universe_epoch_identity_invalid")
        return self


class CausalRiskReturnChunk(RiskContract):
    """Locate one yearly causal-return chunk by content and metadata identities.

    The record retains row count, formation bounds and the publication URI. The artifact reader
    separately verifies its bytes and rows.
    """

    kind: Literal["CausalRiskReturnChunk"] = "CausalRiskReturnChunk"
    year: int = Field(ge=1900, le=2200)
    row_count: int = Field(ge=1)
    first_formation_session: date
    last_formation_session: date
    content_hash: Hash = Field(pattern=_HASH_PATTERN)
    metadata_hash: Hash = Field(pattern=_HASH_PATTERN)
    uri: str = Field(min_length=1)


class CausalRiskReturnSurface(RiskContract):
    """Seal causal one-session open-to-open log returns for a declared universe epoch.

    The formula uses split-adjusted opens and period dividends. Source watermark, yearly chunks,
    formation coverage and limitations form the sealed surface.
    """

    kind: Literal["CausalRiskReturnSurface"] = "CausalRiskReturnSurface"
    epoch: RiskUniverseEpoch
    first_formation_session: date
    last_formation_session: date
    formation_count: int = Field(ge=315)
    return_unit: Literal["one-session-open-to-open-log-return"] = (
        "one-session-open-to-open-log-return"
    )
    formula_identity: Literal["split-adjusted-open-plus-dividend-log-gross-return"] = (
        "split-adjusted-open-plus-dividend-log-gross-return"
    )
    source_watermark_hash: Hash = Field(pattern=_HASH_PATTERN)
    chunks: tuple[CausalRiskReturnChunk, ...] = Field(min_length=1)
    limitations: tuple[str, ...]
    surface_hash: Hash = Field(pattern=_HASH_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_surface(self) -> Self:
        """Require ordered coverage, complete axis row counts and exact return identity.

        Returns:
            This contract after consistency and applicable identity checks.

        Raises:
            ValueError: Formation bounds are reversed, chunk rows differ from formations times
                listings or surface_hash is inconsistent.
        """
        if self.first_formation_session > self.last_formation_session:
            raise ValueError("risk_research.return_surface_range_invalid")
        if sum(chunk.row_count for chunk in self.chunks) != (
            self.formation_count * len(self.epoch.ordered_listing_ids)
        ):
            raise ValueError("risk_research.return_surface_row_count_invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"surface_hash"}))
        if self.surface_hash != expected:
            raise ValueError("risk_research.return_surface_identity_invalid")
        return self


ADMISSIBLE_EWMA_DECAY: Final = (0.94, 0.97)
"""The decay factors this recipe schema will accept.

Two points, not one, and the difference is the whole reason this tuple exists.
While the only admissible value equalled the default, an authored recipe was
byte-identical to the recipe the builder constructed for itself -- so a builder
that ignored the authored recipe entirely produced exactly the same numbers, and
no test could tell the two apart. A schema that admits a second value makes the
routing observable in the estimates themselves.

Both are the standard RiskMetrics factors: 0.94 for daily returns, 0.97 for
monthly. ``DEFAULT_EWMA_DECAY`` stays 0.94, so every published surface keeps its
identity and production continues to seal exactly the recipe it always sealed.
Widening the set is deliberately identity-moving for the *source closure* --
these bytes are inside it -- and is recorded as such; it moves no published
artifact, because artifact identity is over field values and 0.94 still hashes
to what it always did.
"""

DEFAULT_EWMA_DECAY: Final = 0.94


class CovarianceRecipe(RiskContract):
    """Seal the admitted EWMA-standardized Ledoit-Wolf correlation recipe.

    Initialization and residual windows are fixed at 63 and 252 sessions. Decay must be an exactly
    declared admissible value; output units and Ledoit-Wolf options remain fixed.
    """

    kind: Literal["CovarianceRecipe"] = "CovarianceRecipe"
    recipe_name: Literal["EWMA_STANDARDIZED_LEDOIT_WOLF_CORRELATION"] = (
        "EWMA_STANDARDIZED_LEDOIT_WOLF_CORRELATION"
    )
    initialization_sessions: Literal[63] = 63
    standardized_residual_sessions: Literal[252] = 252
    ewma_decay: float = Field(default=DEFAULT_EWMA_DECAY, ge=0.0, lt=1.0)
    ledoit_wolf_store_precision: Literal[False] = False
    ledoit_wolf_assume_centered: Literal[False] = False
    output_unit: Literal["one-session-open-to-open-log-return-covariance"] = (
        "one-session-open-to-open-log-return-covariance"
    )
    recipe_hash: Hash = Field(pattern=_HASH_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_recipe(self) -> Self:
        # Membership, and by exact float equality: an approximate check would
        # admit values that hash differently and so would silently create
        # recipe identities the declared domain never named.
        """Require an exactly admitted EWMA decay and canonical covariance recipe.

        Returns:
            This contract after consistency and applicable identity checks.

        Raises:
            ValueError: Decay is outside the declared exact values or recipe_hash is inconsistent.
        """
        if not any(self.ewma_decay == candidate for candidate in ADMISSIBLE_EWMA_DECAY):
            raise ValueError("risk_research.recipe_decay_invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"recipe_hash"}))
        if self.recipe_hash != expected:
            raise ValueError("risk_research.recipe_identity_invalid")
        return self


class CovarianceDiagnostics(RiskContract):
    """Record numerical diagnostics and provenance for one formation covariance.

    The diagnostics retain shrinkage, eigenvalue/condition summaries, trace, correlations,
    eigenvalue shares, annualized volatility bounds and matrix/environment identities.
    """

    formation_session: date
    asset_count: int = Field(ge=1)
    shrinkage: float = Field(ge=0.0, le=1.0)
    minimum_eigenvalue: float
    maximum_eigenvalue: float = Field(gt=0.0)
    condition_number: float = Field(gt=0.0, le=1e12)
    trace: float = Field(gt=0.0)
    average_correlation: float = Field(ge=-1.0, le=1.0)
    top_one_eigenvalue_share: float = Field(ge=0.0, le=1.0)
    top_five_eigenvalue_share: float = Field(ge=0.0, le=1.0)
    annualized_volatility_minimum: float = Field(gt=0.0)
    annualized_volatility_median: float = Field(gt=0.0)
    annualized_volatility_maximum: float = Field(gt=0.0)
    numerical_environment_hash: Hash = Field(pattern=_HASH_PATTERN)
    matrix_hash: Hash = Field(pattern=_HASH_PATTERN)


class HistoricalCovarianceChunk(RiskContract):
    """Bind a bounded session sequence to packed symmetric covariance matrices.

    The chunk retains each matrix identity, the packed little-endian float64 byte identity, asset
    count and storage URI. At most 21 formations belong to one chunk.
    """

    kind: Literal["HistoricalCovarianceChunk"] = "HistoricalCovarianceChunk"
    formation_sessions: tuple[date, ...] = Field(min_length=1, max_length=21)
    matrix_count: int = Field(ge=1, le=21)
    asset_count: int = Field(ge=1)
    packed_value_count: int = Field(ge=1)
    matrix_hashes: tuple[Hash, ...] = Field(min_length=1, max_length=21)
    packed_bytes_sha256: Hash = Field(pattern=_HASH_PATTERN)
    content_hash: Hash = Field(pattern=_HASH_PATTERN)
    uri: str = Field(min_length=1)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_chunk(self) -> Self:
        """Require aligned matrix/session counts, packed size and exact chunk content identity.

        Returns:
            This contract after consistency and applicable identity checks.

        Raises:
            ValueError: Counts, asset-derived packed size or the canonical chunk payload differ from
                the declaration.
        """
        expected_values = self.asset_count * (self.asset_count + 1) // 2
        if self.matrix_count != len(self.formation_sessions):
            raise ValueError("risk_research.covariance_chunk_count_invalid")
        if self.matrix_count != len(self.matrix_hashes):
            raise ValueError("risk_research.covariance_chunk_hash_count_invalid")
        if self.packed_value_count != self.matrix_count * expected_values:
            raise ValueError("risk_research.covariance_chunk_shape_invalid")
        expected_hash = canonical_hash(
            {
                "dtype": "little-endian-float64",
                "asset_count": self.asset_count,
                "formation_sessions": self.formation_sessions,
                "matrix_hashes": self.matrix_hashes,
                "packed_bytes_sha256": self.packed_bytes_sha256,
            }
        )
        if self.content_hash != expected_hash:
            raise ValueError("risk_research.covariance_chunk_identity_invalid")
        return self


class HistoricalCovarianceSurface(RiskContract):
    """Seal an ordered historical covariance publication and its source lineage.

    The surface binds epoch, causal returns, recipe, execution/environment provenance, diagnostics,
    packed chunks, published bytes and declared limitations.
    """

    kind: Literal["HistoricalCovarianceSurface"] = "HistoricalCovarianceSurface"
    epoch_hash: Hash = Field(pattern=_HASH_PATTERN)
    return_surface_hash: Hash = Field(pattern=_HASH_PATTERN)
    recipe: CovarianceRecipe
    execution_binding_hash: Hash = Field(pattern=_HASH_PATTERN)
    numerical_environment_hash: Hash = Field(pattern=_HASH_PATTERN)
    first_formation_session: date
    last_formation_session: date
    formation_count: int = Field(ge=1)
    asset_count: int = Field(ge=1)
    chunks: tuple[HistoricalCovarianceChunk, ...] = Field(min_length=1)
    diagnostics_hash: Hash = Field(pattern=_HASH_PATTERN)
    matrix_bytes: int = Field(ge=1)
    limitations: tuple[str, ...]
    surface_hash: Hash = Field(pattern=_HASH_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_surface(self) -> Self:
        """Require complete chunk coverage, matching assets and exact surface identity.

        Returns:
            This contract after consistency and applicable identity checks.

        Raises:
            ValueError: Chunk counts, an asset count or surface_hash differs from the surface
                declaration.
        """
        if sum(chunk.matrix_count for chunk in self.chunks) != self.formation_count:
            raise ValueError("risk_research.covariance_surface_count_invalid")
        if any(chunk.asset_count != self.asset_count for chunk in self.chunks):
            raise ValueError("risk_research.covariance_surface_axis_invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"surface_hash"}))
        if self.surface_hash != expected:
            raise ValueError("risk_research.covariance_surface_identity_invalid")
        return self


class RiskFormationEvaluation(RiskContract):
    """Record formation diagnostics and subsequent realized-return calibration.

    Optional annualized volatility diagnostics appear together. Relative matrix changes can be
    unavailable when no preceding covariance was supplied.
    """

    formation_session: date
    next_session: date
    matrix_hash: Hash = Field(pattern=_HASH_PATTERN)
    shrinkage: float = Field(ge=0.0, le=1.0)
    minimum_eigenvalue: float
    maximum_eigenvalue: float = Field(gt=0.0)
    condition_number: float = Field(gt=0.0, le=1e12)
    trace: float = Field(gt=0.0)
    average_correlation: float = Field(ge=-1.0, le=1.0)
    top_one_eigenvalue_share: float = Field(ge=0.0, le=1.0)
    top_five_eigenvalue_share: float = Field(ge=0.0, le=1.0)
    # These diagnostics were added after the first immutable Risk publication.
    # An all-None group is accepted only for historical readback; current
    # numerical production always supplies the complete group.
    annualized_volatility_minimum: float | None = Field(default=None, gt=0.0)
    annualized_volatility_median: float | None = Field(default=None, gt=0.0)
    annualized_volatility_maximum: float | None = Field(default=None, gt=0.0)
    gaussian_log_score_per_asset: float
    equal_weight_predicted_variance: float = Field(gt=0.0)
    equal_weight_realized_squared_return: float = Field(ge=0.0)
    sector_balanced_predicted_variance: float = Field(gt=0.0)
    sector_balanced_realized_squared_return: float = Field(ge=0.0)
    trace_ratio: float | None = None
    frobenius_delta_ratio: float | None = Field(default=None, ge=0.0)
    leading_eigenvalue_delta_ratio: float | None = None

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_compatible_diagnostics(self) -> Self:
        """Require annualized volatility diagnostics to be present together or absent together.

        Returns:
            This contract after consistency and applicable identity checks.

        Raises:
            ValueError: The minimum, median and maximum volatility fields are only partially
                supplied.
        """
        annualized = (
            self.annualized_volatility_minimum,
            self.annualized_volatility_median,
            self.annualized_volatility_maximum,
        )
        if any(value is None for value in annualized) and not all(
            value is None for value in annualized
        ):
            raise ValueError("risk_research.formation_diagnostics_incomplete")
        return self


class HistoricalCovarianceBuildCheckpoint(RiskContract):
    """Small verified prefix for one resumable historical build."""

    kind: Literal["HistoricalCovarianceBuildCheckpoint"] = "HistoricalCovarianceBuildCheckpoint"
    program_hash: Hash = Field(pattern=_HASH_PATTERN)
    expected_formation_count: int = Field(ge=1)
    attempt_count: int = Field(ge=1)
    cumulative_numerical_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    chunks: tuple[HistoricalCovarianceChunk, ...]
    evaluations: tuple[RiskFormationEvaluation, ...]
    checkpoint_hash: Hash = Field(pattern=_HASH_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_checkpoint(self) -> Self:
        """Require a complete aligned checkpoint prefix and exact checkpoint identity.

        Returns:
            This contract after consistency and applicable identity checks.

        Raises:
            ValueError: Chunk/evaluation counts exceed or differ from the expected prefix, session
                axes differ or checkpoint_hash is inconsistent.
        """
        completed = sum(chunk.matrix_count for chunk in self.chunks)
        if completed != len(self.evaluations) or completed > self.expected_formation_count:
            raise ValueError("risk_research.covariance_checkpoint_count_invalid")
        sessions = tuple(item.formation_session for item in self.evaluations)
        chunk_sessions = tuple(
            session for chunk in self.chunks for session in chunk.formation_sessions
        )
        if sessions != chunk_sessions:
            raise ValueError("risk_research.covariance_checkpoint_axis_invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"checkpoint_hash"}))
        if self.checkpoint_hash != expected:
            raise ValueError("risk_research.covariance_checkpoint_identity_invalid")
        return self


class RiskHistoricalDiagnostics(RiskContract):
    """Seal a sorted unique formation axis of covariance evaluation evidence.

    Identity validation admits the full current serialization and the historical exclude-unset
    serialization for compatibility.
    """

    kind: Literal["RiskHistoricalDiagnostics"] = "RiskHistoricalDiagnostics"
    epoch_hash: Hash = Field(pattern=_HASH_PATTERN)
    recipe_hash: Hash = Field(pattern=_HASH_PATTERN)
    evaluations: tuple[RiskFormationEvaluation, ...] = Field(min_length=1)
    diagnostics_hash: Hash = Field(pattern=_HASH_PATTERN)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_diagnostics(self) -> Self:
        """Require a sorted unique formation axis and a compatible exact diagnostics identity.

        Returns:
            This contract after consistency and applicable identity checks.

        Raises:
            ValueError: The formation axis repeats or is unordered, or neither current nor
                historical exclude-unset serialization matches diagnostics_hash.
        """
        sessions = tuple(item.formation_session for item in self.evaluations)
        if sessions != tuple(sorted(set(sessions))):
            raise ValueError("risk_research.historical_diagnostics_axis_invalid")
        current = canonical_hash(self.model_dump(mode="json", exclude={"diagnostics_hash"}))
        historical = canonical_hash(
            self.model_dump(mode="json", exclude={"diagnostics_hash"}, exclude_unset=True)
        )
        if self.diagnostics_hash not in {current, historical}:
            raise ValueError("risk_research.historical_diagnostics_identity_invalid")
        return self


def default_covariance_recipe() -> CovarianceRecipe:
    """Seal the installed default EWMA-standardized covariance recipe.

    Returns:
        Validated default covariance recipe with its canonical recipe identity.
    """
    return seal_contract(CovarianceRecipe, "recipe_hash")


__all__ = [
    "CausalRiskReturnChunk",
    "CausalRiskReturnSurface",
    "CovarianceDiagnostics",
    "CovarianceRecipe",
    "HistoricalCovarianceBuildCheckpoint",
    "HistoricalCovarianceChunk",
    "HistoricalCovarianceSurface",
    "RiskContract",
    "RiskFormationEvaluation",
    "RiskHistoricalDiagnostics",
    "RiskUniverseEpoch",
    "default_covariance_recipe",
    "seal_contract",
]
