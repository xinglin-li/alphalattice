"""Explicit immutable catalog of Portfolio policies installed by the Host."""

from __future__ import annotations

from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .contracts import PortfolioPolicyAdapter, PortfolioPolicyRecipe


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class PortfolioPolicyCapabilityIdentity(_Contract):
    """Declare one installed policy identity and whether its allocation uses a solver."""

    policy_id: str = Field(min_length=1, max_length=96)
    solver_backed: bool


class PortfolioPolicyCatalogBinding(_Contract):
    """Content identity of the explicitly installed Host policy capabilities."""

    ordered_capabilities: tuple[PortfolioPolicyCapabilityIdentity, ...] = Field(min_length=1)
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> PortfolioPolicyCatalogBinding:
        """Require unique ordered capability declarations and exact catalog identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Policy IDs repeat or catalog_hash differs.
        """
        keys = tuple(value.policy_id for value in self.ordered_capabilities)
        if len(set(keys)) != len(keys):
            raise ValueError("PORTFOLIO_POLICY_CATALOG_CAPABILITY_DUPLICATED")
        if self.catalog_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"catalog_hash"})
        ):
            raise ValueError("PORTFOLIO_POLICY_CATALOG_IDENTITY_INVALID")
        return self


class PortfolioPolicyCatalog:
    """Immutable installed-policy index resolved by adapter-declared policy id."""

    def __init__(self, adapters: tuple[PortfolioPolicyAdapter, ...]) -> None:
        """Index a nonempty uniquely identified adapter catalog in registration order.

        Args:
            adapters: Explicit installed deterministic policy adapters.

        Raises:
            ValueError: Catalog is empty or policy IDs repeat.
        """
        indexed = {value.policy_id: value for value in adapters}
        if not adapters or len(indexed) != len(adapters):
            raise ValueError("PORTFOLIO_POLICY_CATALOG_INVALID")
        self._adapters = MappingProxyType(indexed)

    @property
    def policy_ids(self) -> tuple[str, ...]:
        """Read installed policy identities in registration order.

        Returns:
            Ordered adapter mapping keys.
        """
        return tuple(self._adapters)

    @property
    def adapters(self) -> tuple[PortfolioPolicyAdapter, ...]:
        """Read installed adapters in registration order.

        Returns:
            Ordered deterministic adapter owners.
        """
        return tuple(self._adapters.values())

    @property
    def binding(self) -> PortfolioPolicyCatalogBinding:
        """Seal ordered policy capability declarations without runtime roots.

        Returns:
            Catalog binding over each registered policy ID and solver-backed flag.
        """
        values = tuple(
            PortfolioPolicyCapabilityIdentity(
                policy_id=adapter.policy_id,
                solver_backed=adapter.solver_backed,
            )
            for adapter in self._adapters.values()
        )
        identity = {"ordered_capabilities": [value.model_dump(mode="json") for value in values]}
        return PortfolioPolicyCatalogBinding(
            ordered_capabilities=values,
            catalog_hash=canonical_hash(identity),
        )

    def resolve(self, policy: PortfolioPolicyRecipe) -> PortfolioPolicyAdapter:
        """Resolve the installed deterministic adapter for a recipe.

        Args:
            policy: Recipe exposing its installed policy_id.

        Returns:
            Matching installed adapter.

        Raises:
            ValueError: The recipe's policy adapter is not installed.
        """
        try:
            return self._adapters[policy.policy_id]
        except KeyError as error:
            raise ValueError("PORTFOLIO_POLICY_ADAPTER_NOT_INSTALLED") from error


def build_public_portfolio_policy_catalog() -> PortfolioPolicyCatalog:
    """The public desktop's catalog: one closed-form adapter and nothing else.

    Separate from the research catalog below rather than a filtered view of it,
    because the difference is an installation decision and not a query. Building
    this one imports no optimizer, no solver and no research adapter, so a public
    task cannot resolve a solver-backed policy id merely because its source is on
    disk.
    """
    from .tranche_book import TrancheBookAdapter

    return PortfolioPolicyCatalog((TrancheBookAdapter(),))


def build_installed_portfolio_policy_catalog() -> PortfolioPolicyCatalog:
    """Construct the explicit installed catalog; no discovery or runtime mutation."""
    from .buffered_equal_weight import WholeBookHysteresisEqualWeightAdapter
    from .buffered_inverse_volatility import WholeBookHysteresisInverseVolatilityAdapter
    from .buffered_rank_return import WholeBookHysteresisCausalRankMuAdapter
    from .current_universe_equal_weight import CurrentUniverseEqualWeightAdapter
    from .minimum_variance import TopKMinimumVarianceAdapter
    from .return_scaled_total_signal import ReturnScaledTotalSignalAdapter
    from .score_risk_cost import (
        RankBufferedScoreRiskCostAdapter,
        SectorDeviationPenaltyAdapter,
        TopKScoreRiskCostAdapter,
    )
    from .stratified_equal_weight import StratifiedTopKEqualWeightAdapter
    from .top_k_equal_weight import TopKEqualWeightAdapter

    # Order is declaration order and it is inside ``catalog_hash``. The four
    # pre-existing adapters keep their positions so the Stage 6 installation is
    # an append: reordering them would move a catalog identity that published
    # Portfolio evidence already names, for no reason anyone could point at.
    return PortfolioPolicyCatalog(
        (
            TopKEqualWeightAdapter(),
            TopKMinimumVarianceAdapter(),
            TopKScoreRiskCostAdapter(),
            SectorDeviationPenaltyAdapter(),
            CurrentUniverseEqualWeightAdapter(),
            StratifiedTopKEqualWeightAdapter(),
            ReturnScaledTotalSignalAdapter(),
            RankBufferedScoreRiskCostAdapter(),
            WholeBookHysteresisEqualWeightAdapter(),
            WholeBookHysteresisInverseVolatilityAdapter(),
            WholeBookHysteresisCausalRankMuAdapter(),
        )
    )


__all__ = [
    "PortfolioPolicyCapabilityIdentity",
    "PortfolioPolicyCatalog",
    "PortfolioPolicyCatalogBinding",
    "build_installed_portfolio_policy_catalog",
    "build_public_portfolio_policy_catalog",
]
