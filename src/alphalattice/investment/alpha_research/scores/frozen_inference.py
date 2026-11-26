"""Single-component inference authority, independent of a Portfolio book.

Reuses the admitted child/model/Feature scoring contracts. It installs neither
a current Portfolio mode nor a research decision, and never fits a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from alphalattice.investment.alpha_research.inputs.frozen_price_volume import (
    FrozenPriceVolumeInputs,
    apply_frozen_price_volume_scale,
    prepare_frozen_price_volume_features,
    prepare_frozen_price_volume_history,
)
from alphalattice.investment.alpha_research.publication.contracts import (
    _Contract,
    seal_current_contract,
)
from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
    INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
    LIVE_COMPONENT_MODEL_SET_HASHES,
    LIVE_SCORE_CLOSURE_RECEIPT_HASH,
    HeterogeneousChildModelIdentity,
    HeterogeneousComponentModelSet,
)
from alphalattice.investment.alpha_research.scores.heterogeneous_replay import (
    AlphaRuntimeHeterogeneousPredictionOwner,
    HeterogeneousClosureChild,
    read_verified_closure_model,
)
from alphalattice.investment.alpha_research.scores.product_replay import (
    AlphaProductScoreProjection,
    HeterogeneousFormationScoreInput,
    HeterogeneousLiveModel,
    HeterogeneousVintageFeatureSurface,
    live_vintages,
    score_heterogeneous_component,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class FrozenInferenceError(ValueError):
    """A named missing or inconsistent inference authority, before prediction."""


class FrozenMarketScale(_Contract):
    """Retain one vintage market scaler with causal training sessions and source binding.

    The three centers/scales are finite and scales are nonnegative. Training must end before the
    vintage month starts.
    """

    vintage: str
    training_sessions: tuple[date, ...] = Field(min_length=1)
    source_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    center: tuple[float, float, float]
    scale: tuple[float, float, float]

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def check_training(self) -> Self:
        """Require canonical training before the vintage boundary and finite nonnegative scales.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            FrozenInferenceError: Training sessions are noncanonical/noncausal or scaler
                centers/scales are invalid.
        """
        if (
            self.training_sessions != tuple(sorted(set(self.training_sessions)))
            or self.training_sessions[-1] >= date.fromisoformat(self.vintage + "-01")
            or not np.isfinite(self.center).all()
            or not np.isfinite(self.scale).all()
            or min(self.scale) < 0.0
        ):
            raise FrozenInferenceError("alpha_research.frozen_scale_authority_invalid")
        return self


class FrozenComponentInferenceAuthority(_Contract):
    """Seal the admitted frozen component models and market scalers for local input QA.

    The authority binds installed component/gate receipts, exact child identities and observed
    numerical provenance. Its purpose is local input scoring; the declaration does not promote
    research results.
    """

    kind: Literal["FrozenComponentInferenceAuthority"] = "FrozenComponentInferenceAuthority"
    purpose: Literal["LOCAL_INPUT_SCORING_QA"] = "LOCAL_INPUT_SCORING_QA"
    model_set: HeterogeneousComponentModelSet
    children: tuple[HeterogeneousClosureChild, ...]
    market_scales: tuple[FrozenMarketScale, ...]
    gate_m_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_environment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    authority_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal the declared frozen local-scoring model and scaler authority.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical authority_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return seal_current_contract(cls, values, "authority_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def check_identity(self) -> Self:
        """Require the installed component closure, exact vintage scalers and authority identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            FrozenInferenceError: Installed component/gate/model identities, ordered vintage/child
                declarations or authority_hash differ from the admitted closure.
        """
        model = self.model_set
        if (
            model.component_id != "G6_R0_FAST_REBOUND"
            or self.gate_m_receipt_hash != LIVE_SCORE_CLOSURE_RECEIPT_HASH
            or model.model_set_hash != LIVE_COMPONENT_MODEL_SET_HASHES[model.component_id]
            or tuple(value.vintage for value in self.market_scales) != model.vintages
            or tuple(
                HeterogeneousChildModelIdentity(
                    vintage=child.vintage,
                    seed=child.seed,
                    recipe_hash=child.recipe_hash,
                    content_hash=child.content_hash,
                    lineage_hash=child.lineage_hash,
                )
                for child in self.children
            )
            != model.children
            or self.authority_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"authority_hash"}))
        ):
            raise FrozenInferenceError("alpha_research.frozen_inference_authority_invalid")
        return self

    def supports(self, formation: date) -> bool:
        """Check whether a formation selects exactly this admitted live-vintage sequence.

        Args:
            formation: Formation whose installed live vintage window is queried.

        Returns:
            True only when the requested ordered vintages equal the model-set vintages.
        """
        return bool(
            live_vintages(formation, count=self.model_set.vintage_count) == self.model_set.vintages
        )


@dataclass
class AdmittedFrozenInference:
    """Retain verified frozen models, local scoring authority and its prediction owner."""

    authority: FrozenComponentInferenceAuthority
    models: tuple[HeterogeneousLiveModel, ...]
    prediction_owner: AlphaRuntimeHeterogeneousPredictionOwner

    def features(
        self, source: FrozenPriceVolumeInputs, formation: date
    ) -> tuple[HeterogeneousVintageFeatureSurface, ...]:
        """Prepare formation features under the exact admitted vintage and scaler authority.

        Args:
            source: Dated price/volume/context inputs admitted by the frozen feature owner.
            formation: Requested formation within the admitted model epoch.

        Returns:
            Ordered vintage feature surfaces on the admitted model feature axis.

        Raises:
            FrozenInferenceError: The formation has no admitted model epoch or feature preparation
                fails.
        """
        if not self.authority.supports(formation):
            raise FrozenInferenceError("alpha_research.frozen_model_epoch_unavailable")
        feature_ids = self.models[0].estimator.ordered_feature_ids
        return component_feature_surfaces(
            source,
            formation=formation,
            feature_ids=feature_ids,
            scales=self.authority.market_scales,
            authority_hash=self.authority.authority_hash,
        )

    def score(
        self,
        *,
        formation: date,
        listing_ids: tuple[str, ...],
        eligible: npt.NDArray[np.bool_],
        surfaces: tuple[HeterogeneousVintageFeatureSurface, ...],
        raw_12_1_momentum: npt.NDArray[np.float64] | None = None,
    ) -> AlphaProductScoreProjection:
        """Score the admitted frozen component through its deterministic prediction owner.

        Args:
            formation: Formation within the admitted model epoch.
            listing_ids: Exact ordered listing axis.
            eligible: Decision-eligibility mask on that axis.
            surfaces: Admitted vintage feature surfaces.
            raw_12_1_momentum: Optional raw momentum lane used by the component policy.

        Returns:
            Component score projection from bound feature/model/eligibility authority.

        Raises:
            FrozenInferenceError: The requested model epoch is unavailable.
            HeterogeneousAlphaError: Bound score-input or component numerical admission fails.
        """
        if not self.authority.supports(formation):
            raise FrozenInferenceError("alpha_research.frozen_model_epoch_unavailable")
        return score_heterogeneous_component(
            component=self.authority.model_set,
            inputs=HeterogeneousFormationScoreInput.create(
                formation_session=formation,
                ordered_listing_ids=listing_ids,
                decision_eligible=eligible,
                raw_12_1_momentum=raw_12_1_momentum,
                feature_surfaces=surfaces,
                models=self.models,
                model_set_manifest_hash=self.authority.authority_hash,
            ),
            prediction_owner=self.prediction_owner,
        )


def component_feature_surfaces(
    source: FrozenPriceVolumeInputs,
    *,
    formation: date,
    feature_ids: tuple[str, ...],
    scales: tuple[FrozenMarketScale, ...],
    authority_hash: str,
) -> tuple[HeterogeneousVintageFeatureSurface, ...]:
    """The shared input owner; a renewed model changes its scale, not feature arithmetic."""
    prepared = prepare_frozen_price_volume_features(
        source, ordered_feature_ids=feature_ids, formation_session=formation
    )
    return _scaled_component_surfaces(
        source,
        prepared=prepared,
        feature_ids=feature_ids,
        scales=scales,
        authority_hash=authority_hash,
    )


class ComponentFeatureHistory:
    """A value for one execution over one verified source, not a stored cache."""

    def __init__(
        self, source: FrozenPriceVolumeInputs, *, feature_ids: tuple[str, ...], through: date
    ):
        """Prepare dated price/volume history once on an explicit feature axis and cutoff.

        Args:
            source: Frozen dated input tensors.
            feature_ids: Ordered feature formulas required by the admitted models.
            through: Last source formation included in preparation.
        """
        self.source, self.feature_ids = source, feature_ids
        self.values = prepare_frozen_price_volume_history(
            source, ordered_feature_ids=feature_ids, through=through
        )
        self.positions = {
            day: i for i, day in enumerate(source.formation_sessions[: len(self.values)])
        }

    def surfaces(
        self, formation: date, *, scales: tuple[FrozenMarketScale, ...], authority_hash: str
    ) -> tuple[HeterogeneousVintageFeatureSurface, ...]:
        """Scale one prepared formation into its admitted vintage feature surfaces.

        Args:
            formation: Formation present in the prepared history.
            scales: Ordered admitted market scalers for the requested model vintages.
            authority_hash: Exact inference authority identity retained in the surface binding.

        Returns:
            Scaled vintage feature surfaces for the selected prepared formation.

        Raises:
            FrozenInferenceError: The requested formation was not prepared or feature/scaler
                admission fails.
        """
        if formation not in self.positions:
            raise FrozenInferenceError("alpha_research.feature_history_formation_unavailable")
        return _scaled_component_surfaces(
            self.source,
            prepared=self.values[self.positions[formation]],
            feature_ids=self.feature_ids,
            scales=scales,
            authority_hash=authority_hash,
        )


def _scaled_component_surfaces(
    source: FrozenPriceVolumeInputs,
    *,
    prepared: npt.NDArray[np.float64],
    feature_ids: tuple[str, ...],
    scales: tuple[FrozenMarketScale, ...],
    authority_hash: str,
) -> tuple[HeterogeneousVintageFeatureSurface, ...]:
    return tuple(
        HeterogeneousVintageFeatureSurface.create(
            vintage=scale.vintage,
            ordered_listing_ids=source.ordered_listing_ids,
            ordered_feature_ids=feature_ids,
            features=apply_frozen_price_volume_scale(
                prepared,
                ordered_feature_ids=feature_ids,
                market_center=scale.center,
                market_scale=scale.scale,
            ),
            source_binding_hash=canonical_hash(
                {
                    "source": source.source_binding_hash,
                    "scale": scale.model_dump(mode="json"),
                    "authority": authority_hash,
                }
            ),
        )
        for scale in scales
    )


def admit_frozen_inference(root: Path, *, expected_hash: str) -> AdmittedFrozenInference:
    """Read and verify exact local inference authority and each declared closure model.

    Args:
        root: Caller-owned frozen authority/model directory.
        expected_hash: Required authority identity to compare before model loading.

    Returns:
        Admitted authority, verified live models and a prediction owner restricted to its component.

    Raises:
        FrozenInferenceError: Authority JSON/model is unreadable, expected identity differs or a
            closure model cannot be verified.
    """
    try:
        manifest = FrozenComponentInferenceAuthority.model_validate_json(
            (root / "inference-authority.json").read_bytes()
        )
    except (OSError, ValueError) as error:
        raise FrozenInferenceError("alpha_research.frozen_inference_manifest_invalid") from error
    if manifest.authority_hash != expected_hash:
        raise FrozenInferenceError("alpha_research.frozen_inference_binding_mismatch")
    loaded = []
    for child in manifest.children:
        expected_axis = (
            manifest.model_set.ordered_feature_ids
            or INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(
                manifest.model_set.component_id
            ).ordered_feature_ids
        )
        loaded.append(
            read_verified_closure_model(
                root,
                child,
                feature_axis_hash=manifest.model_set.feature_axis_hash,
                feature_count=manifest.model_set.feature_count,
                ordered_feature_ids=expected_axis,
            )
        )
    return AdmittedFrozenInference(
        manifest,
        tuple(loaded),
        AlphaRuntimeHeterogeneousPredictionOwner(
            component_ids=(manifest.model_set.component_id,),
        ),
    )
