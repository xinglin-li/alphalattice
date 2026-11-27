"""Fail-closed readback of the heterogeneous strategy's admitted evidence.

Historical replay reads a sealed manifest and its four component score arrays.
The current closure is the product-readable materialisation
of the forty-eight admitted children, their per-vintage Feature surfaces and
the raw momentum lane, derived once under a granted authority from the exact
files the Gate M source readback names. Neither is installation authority by
itself: the successor recipe in :mod:`heterogeneous_product` supplies the
strategy, and each reader here proves a caller opened exactly the evidence its
manifest names -- every file by hash, every payload by content identity, every
recipe against the installed strategy -- before anything numerical is built.
Paths are explicit caller inputs. Nothing is discovered, copied or silently
substituted from the absolute paths serialised by a research run.

The current closure's reader is also the only place the installed Alpha
runtime is reached from the product path: one prediction owner that reopens a
payload through the installed adapter and refuses a numerical environment
other than the one the closure was sealed under.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Final, Literal, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
    DynamicPanelLightGBMAdapter,
    build_dynamic_panel_lightgbm_recipe,
    build_dynamic_panel_lightgbm_search_domain,
)
from alphalattice.capabilities.alpha_modeling.catalog import AlphaModelCatalog
from alphalattice.capabilities.alpha_modeling.contracts import (
    AlphaEstimatorContent,
    BoundAlphaPredictionInput,
)
from alphalattice.capabilities.alpha_modeling.runtime.numerical_environment import (
    resolve_alpha_model_numerical_environment,
)
from alphalattice.capabilities.alpha_modeling.runtime.service import (
    AlphaModelRuntimeService,
)
from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
    COMPONENT_IDS,
    INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
    LIVE_MODEL_VINTAGES,
    ComponentId,
    HeterogeneousLaneValueBinding,
    HeterogeneousModelSetAuthority,
    HeterogeneousSurfaceBinding,
    component_recipe_is_current,
    heterogeneous_model_set_authority,
    live_model_set_hash,
    strategy_recipe_is_current,
)
from alphalattice.investment.alpha_research.scores.product_replay import (
    HeterogeneousComponentScoringRecipe,
    HeterogeneousLiveModel,
    HeterogeneousScoringError,
    HeterogeneousVintageFeatureSurface,
    live_vintages,
)
from alphalattice.kernel.identity import (
    BLAKE3_PATTERN,
    FileDigestTrust,
    integrity_blake3,
    tagged_blake3,
)

type FloatArray = npt.NDArray[np.float64]


class HeterogeneousReplayError(ValueError):
    """Stable refusal for package, array, axis or readback drift."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HeterogeneousReplayError("alpha_research.heterogeneous_json_invalid") from error
    if not isinstance(value, dict):
        raise HeterogeneousReplayError("alpha_research.heterogeneous_json_object_required")
    return cast(dict[str, Any], value)


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class HeterogeneousHistoricalScoreEvidence(_Contract):
    """Name the installed strategy, support axes and exact bytes of each replay surface."""

    kind: Literal["HeterogeneousHistoricalScoreEvidence"] = "HeterogeneousHistoricalScoreEvidence"
    strategy_hash: str = Field(pattern=BLAKE3_PATTERN)
    session_axis_hash: str = Field(pattern=BLAKE3_PATTERN)
    listing_axis_hash: str = Field(pattern=BLAKE3_PATTERN)
    component_digests: dict[ComponentId, Annotated[str, Field(pattern=BLAKE3_PATTERN)]]
    package_hash: str = Field(pattern=BLAKE3_PATTERN)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if set(self.component_digests) != set(COMPONENT_IDS):
            raise HeterogeneousReplayError("alpha_research.heterogeneous_component_set_invalid")
        if self.strategy_hash != INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.strategy_hash:
            raise HeterogeneousReplayError("alpha_research.heterogeneous_strategy_contract_invalid")
        if (
            canonical_hash(self.model_dump(mode="json", exclude={"package_hash"}))
            != self.package_hash
        ):
            raise HeterogeneousReplayError("alpha_research.heterogeneous_evidence_identity_invalid")
        return self


def read_successor_evidence(root: Path) -> HeterogeneousHistoricalScoreEvidence:
    """Admit the replay manifest before reading any array named by the installation."""
    payload = _json(root / "frozen-strategy-package.json")
    seal = payload.get("package_hash")
    if isinstance(seal, str) and re.fullmatch(r"[0-9a-f]{64}", seal):
        raise HeterogeneousReplayError("alpha_research.heterogeneous_evidence_before_blake3")
    try:
        return HeterogeneousHistoricalScoreEvidence.model_validate(payload)
    except ValidationError as error:
        cause = next(
            (
                entry.get("ctx", {}).get("error")
                for entry in error.errors()
                if isinstance(entry.get("ctx", {}).get("error"), HeterogeneousReplayError)
            ),
            None,
        )
        if isinstance(cause, HeterogeneousReplayError):
            raise cause from error
        raise HeterogeneousReplayError(
            "alpha_research.heterogeneous_evidence_identity_invalid"
        ) from error


def _immutable_scores(path: Path, *, shape: tuple[int, ...], expected: str) -> FloatArray:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise HeterogeneousReplayError("alpha_research.heterogeneous_evidence_absent") from error
    if tagged_blake3(integrity_blake3(payload)) != expected:
        raise HeterogeneousReplayError("alpha_research.heterogeneous_array_hash_invalid")
    try:
        value = np.load(BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise HeterogeneousReplayError(
            "alpha_research.heterogeneous_array_contract_invalid"
        ) from error
    if (
        not isinstance(value, np.ndarray)
        or value.shape != shape
        or value.dtype != np.dtype("float64")
    ):
        if isinstance(value, np.lib.npyio.NpzFile):
            value.close()
        raise HeterogeneousReplayError("alpha_research.heterogeneous_array_contract_invalid")
    value.setflags(write=False)
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class AdmittedSuccessorEvidence:
    """The historical component scores over the support the installed book asked to read."""

    component_scores: Mapping[ComponentId, FloatArray]

    def component(self, component_id: ComponentId) -> FloatArray:
        """Read a component's immutable score array."""
        try:
            return self.component_scores[component_id]
        except KeyError as error:
            raise HeterogeneousReplayError(
                "alpha_research.heterogeneous_component_score_absent"
            ) from error


def admit_successor_evidence(
    evidence: HeterogeneousHistoricalScoreEvidence,
    *,
    formation_sessions: tuple[date, ...],
    ordered_listing_ids: tuple[str, ...],
    component_score_paths: Mapping[ComponentId, Path],
) -> AdmittedSuccessorEvidence:
    """Verify the held manifest's support and read its four explicitly supplied score arrays."""
    if (
        canonical_hash(tuple(value.isoformat() for value in formation_sessions))
        != evidence.session_axis_hash
        or canonical_hash(ordered_listing_ids) != evidence.listing_axis_hash
    ):
        raise HeterogeneousReplayError("alpha_research.heterogeneous_axis_identity_invalid")
    if set(component_score_paths) != set(COMPONENT_IDS):
        raise HeterogeneousReplayError("alpha_research.heterogeneous_component_set_invalid")
    shape = (len(formation_sessions), len(ordered_listing_ids))
    return AdmittedSuccessorEvidence(
        component_scores={
            component: _immutable_scores(
                component_score_paths[component],
                shape=shape,
                expected=evidence.component_digests[component],
            )
            for component in COMPONENT_IDS
        }
    )


# ------------------------------------------------------- the current closure


_HEX64 = BLAKE3_PATTERN


class HeterogeneousClosureFile(_Contract):
    """One artifact under the closure root, by relative path, bytes and digest."""

    relative_path: str = Field(min_length=1)
    digest: str = Field(pattern=_HEX64)
    byte_count: int = Field(ge=1)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        """Keep a closure file canonical and relative on every supported host."""
        posix = PurePosixPath(value)
        windows = PureWindowsPath(value)
        if (
            value in {".", ".."}
            or "\\" in value
            or value != posix.as_posix()
            or posix.is_absolute()
            or windows.is_absolute()
            or windows.drive
            or any(part in {".", ".."} for part in posix.parts)
            or any(part in {".", ".."} for part in windows.parts)
        ):
            raise ValueError("alpha_research.heterogeneous_closure_file_path_invalid")
        return value

    def resolve_under(self, root: Path) -> Path:
        """Resolve one file without allowing a link or junction to escape the root."""
        resolved_root = root.resolve()
        resolved = (resolved_root / self.relative_path).resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError as error:
            raise HeterogeneousReplayError(
                "alpha_research.heterogeneous_closure_file_outside_root"
            ) from error
        return resolved


class HeterogeneousClosureChild(_Contract):
    """Bind one vintage/seed model payload to recipe, content, lineage and training authority."""

    vintage: str = Field(pattern=r"^\d{4}-\d{2}$")
    seed: int
    recipe_hash: str = Field(pattern=_HEX64)
    content_hash: str = Field(pattern=_HEX64)
    lineage_hash: str = Field(pattern=_HEX64)
    training_binding_hash: str = Field(pattern=_HEX64)
    payload: HeterogeneousClosureFile


class HeterogeneousClosureSurface(_Contract):
    """One vintage's Feature surface for every closure formation, in one array."""

    vintage: str = Field(pattern=r"^\d{4}-\d{2}$")
    source_binding_hash: str = Field(pattern=_HEX64)
    feature_values_hashes: tuple[Annotated[str, Field(pattern=_HEX64)], ...] = Field(min_length=1)
    """One per closure formation, in formation order, as the live surface hashes them."""

    file: HeterogeneousClosureFile
    """An `.npz` whose `features` is `(formations, listings, feature_count)` float64."""


class HeterogeneousClosureComponent(_Contract):
    """Declare one component model set, feature axis, children and replay surfaces."""

    component_id: ComponentId
    recipe_hash: str = Field(pattern=_HEX64)
    feature_axis_hash: str = Field(pattern=_HEX64)
    feature_count: int = Field(ge=1)
    model_set_hash: str = Field(pattern=_HEX64)
    children: tuple[HeterogeneousClosureChild, ...] = Field(min_length=1)
    surfaces: tuple[HeterogeneousClosureSurface, ...] = Field(min_length=1)


class HeterogeneousClosureMomentum(_Contract):
    """The raw 12-1 momentum lane the G2 and G7 candidate rules read."""

    factor_id: str = Field(min_length=1)
    source_identity_hash: str = Field(pattern=_HEX64)
    value_hashes: tuple[Annotated[str, Field(pattern=_HEX64)], ...] = Field(min_length=1)
    file: HeterogeneousClosureFile
    """An `.npy` of `(formations, listings)` float64."""


class HeterogeneousCurrentClosureManifest(_Contract):
    """The detailed authority owner for current scoring: sealed, content-addressed, local.

    Everything a prediction depends on is a row here -- the forty-eight
    children with their payload digests, each component's per-vintage surfaces
    with their source and value identities, the momentum lane, the formation
    and listing axes, the numerical environment the closure was sealed under,
    and the Gate M and Gate L identities it was derived from. The reader
    verifies every row before it builds a model, and the authority a Program
    binds is constructed from this manifest, never restated elsewhere.
    """

    kind: Literal["HeterogeneousCurrentClosureManifest"] = "HeterogeneousCurrentClosureManifest"
    schema_version: Literal[2] = 2
    strategy_hash: str = Field(pattern=_HEX64)
    gate_m_receipt_hash: str = Field(pattern=_HEX64)
    gate_m_source_readback_digest: str = Field(pattern=_HEX64)
    gate_l_manifest_hash: str = Field(pattern=_HEX64)
    live_vintages: tuple[str, ...]
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    ordered_listing_ids: tuple[str, ...] = Field(min_length=2)
    listing_axis_hash: str = Field(pattern=_HEX64)
    numerical_environment_hash: str = Field(pattern=_HEX64)
    maximum_raw_prediction_gap: float = Field(ge=0.0)
    """Reopened payloads against the research predictions on the closure formations."""

    source_provenance: dict[str, str]
    """Named identities of the sources the closure was derived from."""

    components: tuple[HeterogeneousClosureComponent, ...]
    momentum: HeterogeneousClosureMomentum
    manifest_hash: str = Field(pattern=_HEX64)

    @classmethod
    def seal(cls, **values: object) -> Self:
        """Seal the declared dated heterogeneous replay closure manifest.

        Args:
            values: Explicit manifest fields excluding manifest_hash.

        Returns:
            Validated manifest and its canonical complete-payload identity.

        Raises:
            pydantic.ValidationError: Manifest fields or declared closure consistency fail.
        """
        identity = cls.model_construct(**values, manifest_hash="0" * 64).model_dump(
            mode="json", exclude={"manifest_hash"}
        )
        return cls(**identity, manifest_hash=canonical_hash(identity))

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        """Require exact session/listing/vintage axes and complete component replay closure.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            HeterogeneousReplayError: Calendar/listing identity, four live vintages,
                component/surface/child populations, momentum length, model-set identities or
                manifest_hash disagree.
        """
        sessions = self.formation_sessions
        count = len(sessions)
        if (
            tuple(sorted(set(sessions))) != sessions
            or len(set(self.ordered_listing_ids)) != len(self.ordered_listing_ids)
            or self.listing_axis_hash != canonical_hash(list(self.ordered_listing_ids))
            or len(self.live_vintages) != 4
            or tuple(value.component_id for value in self.components) != COMPONENT_IDS
            or len(self.momentum.value_hashes) != count
            or any(live_vintages(value, count=4) != self.live_vintages for value in sessions)
        ):
            raise HeterogeneousReplayError("alpha_research.heterogeneous_closure_manifest_invalid")
        for component in self.components:
            recipe = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(component.component_id)
            if (
                tuple(value.vintage for value in component.surfaces) != self.live_vintages
                or any(len(value.feature_values_hashes) != count for value in component.surfaces)
                or tuple((value.vintage, value.seed) for value in component.children)
                != tuple((vintage, seed) for vintage in self.live_vintages for seed in recipe.seeds)
                or component.model_set_hash
                != live_model_set_hash(
                    recipe_hash=component.recipe_hash,
                    content_hashes=tuple(value.content_hash for value in component.children),
                )
            ):
                raise HeterogeneousReplayError(
                    "alpha_research.heterogeneous_closure_manifest_invalid"
                )
        if self.manifest_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"manifest_hash"})
        ):
            raise HeterogeneousReplayError(
                "alpha_research.heterogeneous_closure_manifest_identity_invalid"
            )
        return self


CLOSURE_MANIFEST_FILE_NAME = "closure-manifest.json"
ADMITTED_CURRENT_CLOSURE_HASH: Final[str | None] = None
"""An independently admitted closure manifest. No BLAKE3 closure has been admitted."""


class AlphaRuntimeHeterogeneousPredictionOwner:
    """Reopen one admitted payload through the installed adapter and predict.

    The only route from the product path to the Alpha runtime. It refuses a
    payload whose recipe is not the one its seed resolves to, and -- when it was
    given the environment the closure was sealed under -- any prediction made
    in a different numerical environment, because the closure's zero-gap proof
    holds for those bits and no others.
    """

    def __init__(
        self,
        *,
        component_ids: tuple[ComponentId, ...] = COMPONENT_IDS,
    ) -> None:
        """Admit unique component routes and one coherent installed numerical environment.

        The observed execution environment is provenance. It is not compared with a frozen model
        closure historical environment; recipes and model content retain their own numerical
        authority.

        Args:
            component_ids: Nonempty unique installed component identifiers admitted for prediction.

        Raises:
            HeterogeneousScoringError: Component routes are invalid or numerical environment
                resolution is unavailable/incoherent.
        """
        if not component_ids or len(set(component_ids)) != len(component_ids):
            raise HeterogeneousScoringError("alpha_research.heterogeneous_live_component_invalid")
        self._component_ids = component_ids
        self._runtime = AlphaModelRuntimeService(
            AlphaModelCatalog((DynamicPanelLightGBMAdapter(),))
        )
        self._domain = build_dynamic_panel_lightgbm_search_domain()
        try:
            environments = {
                resolve_alpha_model_numerical_environment(
                    self._runtime.resolve_numerical_binding(
                        recipe=build_dynamic_panel_lightgbm_recipe(
                            INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(
                                component_id
                            ).estimator_point.resolve(seed=seed)
                        ),
                        domain=self._domain,
                    )
                ).environment_hash
                for component_id in component_ids
                for seed in INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(component_id).seeds
            }
        except (RuntimeError, ValueError) as error:
            raise HeterogeneousScoringError(
                "alpha_research.heterogeneous_live_numerical_environment_unavailable"
            ) from error
        if len(environments) != 1:
            raise HeterogeneousScoringError(
                "alpha_research.heterogeneous_live_numerical_environment_incoherent"
            )
        # The environment these predictions run in, recorded beside them; a frozen closure's
        # recorded one is its provenance, never compared with it (LAWS.md ID6).
        self.environment_hash = environments.pop()
        self.predictions = 0

    def __call__(
        self,
        *,
        component: HeterogeneousComponentScoringRecipe,
        model: HeterogeneousLiveModel,
        features: FloatArray,
    ) -> FloatArray:
        """Predict admitted component features through the installed Alpha model runtime.

        Args:
            component: Component scoring recipe within this owner admitted component set.
            model: Verified live model with seed, recipe, estimator content and training binding.
            features: Bound feature matrix on the retained estimator feature axis.

        Returns:
            Float64 predictions; successful calls update observed environment and prediction count.

        Raises:
            HeterogeneousScoringError: Component admission, exact model recipe/content or bound
                prediction fails.
        """
        if component.component_id not in self._component_ids:
            raise HeterogeneousScoringError(
                "alpha_research.heterogeneous_live_component_not_admitted"
            )
        installed = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(
            cast(ComponentId, component.component_id)
        )
        recipe = build_dynamic_panel_lightgbm_recipe(
            installed.estimator_point.resolve(seed=model.seed)
        )
        if recipe.recipe_hash != model.recipe_hash:
            raise HeterogeneousScoringError(
                "alpha_research.heterogeneous_live_model_recipe_mismatch"
            )
        estimator = model.estimator
        if not isinstance(estimator, AlphaEstimatorContent):
            raise HeterogeneousScoringError(
                "alpha_research.heterogeneous_live_estimator_content_invalid"
            )
        try:
            result = self._runtime.predict(
                recipe=recipe,
                domain=self._domain,
                estimator=estimator,
                prediction_input=BoundAlphaPredictionInput(
                    training_binding_hash=model.training_binding_hash,
                    ordered_feature_ids=estimator.ordered_feature_ids,
                    features=features,
                ),
            )
        except ValueError as error:
            raise HeterogeneousScoringError(
                "alpha_research.heterogeneous_live_model_prediction_refused"
            ) from error
        self.environment_hash = str(result.numerical_environment.environment_hash)
        self.predictions += 1
        return np.asarray(result.prediction.predictions, dtype=np.float64)


@dataclass
class AdmittedHeterogeneousCurrentClosure:
    """A verified closure root, answering the current-scoring closure protocol.

    Built only by `admit_heterogeneous_current_closure`, after every file and
    every pin was checked. Arrays are read lazily and once; `reads` counts the
    file reads made after admission so a caller can prove an exact reuse read
    nothing.
    """

    root: Path
    manifest: HeterogeneousCurrentClosureManifest
    authority: HeterogeneousModelSetAuthority
    _models: Mapping[ComponentId, tuple[HeterogeneousLiveModel, ...]]
    _surfaces: dict[tuple[ComponentId, str], FloatArray] = field(default_factory=dict)
    _momentum: FloatArray | None = None
    _owner: AlphaRuntimeHeterogeneousPredictionOwner | None = None
    reads: int = 0

    @property
    def formation_sessions(self) -> tuple[date, ...]:
        """Read the replay closure admitted formation axis.

        Returns:
            Manifest formation sessions in declared order.
        """
        return self.manifest.formation_sessions

    @property
    def ordered_listing_ids(self) -> tuple[str, ...]:
        """Read the replay closure exact listing axis.

        Returns:
            Manifest listing identifiers in declared order.
        """
        return self.manifest.ordered_listing_ids

    @property
    def models_by_component(self) -> Mapping[ComponentId, tuple[HeterogeneousLiveModel, ...]]:
        """Read the admitted component-to-live-model mapping.

        Returns:
            Retained verified live models keyed by component identifier.
        """
        return self._models

    @property
    def prediction_owner(self) -> AlphaRuntimeHeterogeneousPredictionOwner:
        """Lazily admit and retain the deterministic heterogeneous prediction owner.

        Returns:
            Prediction owner with coherent installed numerical environment.
        """
        if self._owner is None:
            self._owner = AlphaRuntimeHeterogeneousPredictionOwner()
        return self._owner

    def _row(self, formation_session: date) -> int:
        try:
            return self.manifest.formation_sessions.index(formation_session)
        except ValueError as error:
            raise HeterogeneousReplayError(
                "alpha_research.heterogeneous_closure_formation_absent"
            ) from error

    def feature_surfaces(
        self,
        *,
        component: ComponentId,
        formation_session: date,
        ordered_listing_ids: tuple[str, ...],
    ) -> tuple[HeterogeneousVintageFeatureSurface, ...]:
        """Read one admitted formation from declared vintage feature payloads and cached arrays.

        Args:
            component: Declared component selecting model feature authority and surface files.
            formation_session: Formation present in the admitted replay calendar.
            ordered_listing_ids: Exact manifest listing axis; reordering is refused.

        Returns:
            Constructed vintage feature surfaces in declared component-surface order.

        Raises:
            HeterogeneousReplayError: Listing axis or requested formation differs from the admitted
                closure.
        """
        if ordered_listing_ids != self.manifest.ordered_listing_ids:
            raise HeterogeneousReplayError(
                "alpha_research.heterogeneous_closure_listing_axis_invalid"
            )
        row = self._row(formation_session)
        declared = {value.component_id: value for value in self.manifest.components}[component]
        axis = self._models[component][0].estimator.ordered_feature_ids
        surfaces = []
        for surface in declared.surfaces:
            key = (component, surface.vintage)
            if key not in self._surfaces:
                with np.load(self.root / surface.file.relative_path, allow_pickle=False) as payload:
                    self._surfaces[key] = np.asarray(payload["features"], dtype=np.float64)
                self.reads += 1
            surfaces.append(
                HeterogeneousVintageFeatureSurface.create(
                    vintage=surface.vintage,
                    ordered_listing_ids=ordered_listing_ids,
                    ordered_feature_ids=axis,
                    features=self._surfaces[key][row],
                    source_binding_hash=surface.source_binding_hash,
                )
            )
        return tuple(surfaces)

    def raw_12_1_momentum(
        self, *, formation_session: date, ordered_listing_ids: tuple[str, ...]
    ) -> FloatArray:
        """Copy one formation raw momentum vector from the retained replay payload.

        Args:
            formation_session: Formation present in the admitted replay calendar.
            ordered_listing_ids: Exact manifest listing axis.

        Returns:
            Independent float64 copy of the selected cached momentum row.

        Raises:
            HeterogeneousReplayError: Listing axis or requested formation differs from the admitted
                closure.
        """
        if ordered_listing_ids != self.manifest.ordered_listing_ids:
            raise HeterogeneousReplayError(
                "alpha_research.heterogeneous_closure_listing_axis_invalid"
            )
        row = self._row(formation_session)
        if self._momentum is None:
            self._momentum = np.asarray(
                np.load(self.root / self.manifest.momentum.file.relative_path, allow_pickle=False),
                dtype=np.float64,
            )
            self.reads += 1
        return np.array(self._momentum[row], dtype=np.float64, copy=True)


def _verified_file(root: Path, declared: HeterogeneousClosureFile) -> Path:
    path = declared.resolve_under(root)
    if not path.is_file() or path.stat().st_size != declared.byte_count:
        raise HeterogeneousReplayError("alpha_research.heterogeneous_closure_file_absent")
    if FileDigestTrust.fresh_digest(path) != declared.digest:
        raise HeterogeneousReplayError("alpha_research.heterogeneous_closure_file_hash_invalid")
    return path


def read_verified_closure_model(
    root: Path,
    child: HeterogeneousClosureChild,
    *,
    feature_axis_hash: str,
    feature_count: int,
    ordered_feature_ids: tuple[str, ...] = (),
) -> HeterogeneousLiveModel:
    """The shared payload/axis reader for a component or a composite closure."""
    payload = _verified_file(root, child.payload)
    try:
        estimator = AlphaEstimatorContent.model_validate_json(payload.read_text(encoding="utf-8"))
    except ValueError as error:
        raise HeterogeneousReplayError(
            "alpha_research.heterogeneous_closure_payload_invalid"
        ) from error
    if (
        estimator.content_hash != child.content_hash
        or len(estimator.ordered_feature_ids) != feature_count
        or canonical_hash(list(estimator.ordered_feature_ids)) != feature_axis_hash
        or (ordered_feature_ids and estimator.ordered_feature_ids != ordered_feature_ids)
    ):
        raise HeterogeneousReplayError(
            "alpha_research.heterogeneous_closure_payload_identity_invalid"
        )
    return HeterogeneousLiveModel(
        vintage=child.vintage,
        seed=child.seed,
        recipe_hash=child.recipe_hash,
        training_binding_hash=child.training_binding_hash,
        lineage_hash=child.lineage_hash,
        estimator=estimator,
    )


def admit_heterogeneous_current_closure(root: Path) -> AdmittedHeterogeneousCurrentClosure:
    """Open one sealed closure root, or refuse before a single model exists.

    Order matters and is deliberate: the manifest's own identity, then every
    pin against the installed constants, then every file by digest, then every
    payload by content identity -- and only then the models and the authority.
    A closure that fails any step yields nothing a package could install.
    """
    resolved = root.resolve()
    path = resolved / CLOSURE_MANIFEST_FILE_NAME
    if not path.is_file():
        raise HeterogeneousReplayError("alpha_research.heterogeneous_closure_manifest_absent")
    payload = _json(path)
    if payload.get("schema_version") == 1:
        raise HeterogeneousReplayError(
            "alpha_research.heterogeneous_closure_manifest_invalid: "
            "closure was made before BLAKE3; start a fresh data root"
        )
    try:
        manifest = HeterogeneousCurrentClosureManifest.model_validate(payload)
    except ValueError as error:
        raise HeterogeneousReplayError(
            "alpha_research.heterogeneous_closure_manifest_invalid"
        ) from error
    strategy = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY
    if (
        # A closure sealed before a recorded move of the strategy's or a component's recipe
        # role still names the installed one (NM1): the heterogeneous closure is a
        # sealed root the tree keeps reading.
        not strategy_recipe_is_current(manifest.strategy_hash, strategy.strategy_hash)
        or manifest.manifest_hash != ADMITTED_CURRENT_CLOSURE_HASH
        or manifest.live_vintages != LIVE_MODEL_VINTAGES
        or manifest.maximum_raw_prediction_gap != 0.0
    ):
        raise HeterogeneousReplayError("alpha_research.heterogeneous_closure_pins_invalid")
    for component in manifest.components:
        recipe = strategy.component(component.component_id)
        if (
            not component_recipe_is_current(
                component.component_id, component.recipe_hash, recipe.recipe_hash
            )
            or component.feature_axis_hash != recipe.feature_axis_hash
            or component.feature_count != recipe.feature_count
        ):
            raise HeterogeneousReplayError("alpha_research.heterogeneous_closure_pins_invalid")
    for component in manifest.components:
        for surface in component.surfaces:
            _verified_file(resolved, surface.file)
    _verified_file(resolved, manifest.momentum.file)
    models: dict[ComponentId, tuple[HeterogeneousLiveModel, ...]] = {}
    children: dict[ComponentId, tuple[tuple[str, int, str, str, str], ...]] = {}
    for component in manifest.components:
        loaded = []
        for child in component.children:
            loaded.append(
                read_verified_closure_model(
                    resolved,
                    child,
                    feature_axis_hash=component.feature_axis_hash,
                    feature_count=component.feature_count,
                )
            )
        models[component.component_id] = tuple(loaded)
        children[component.component_id] = tuple(
            (value.vintage, value.seed, value.recipe_hash, value.content_hash, value.lineage_hash)
            for value in component.children
        )
    authority = heterogeneous_model_set_authority(
        children,
        live_vintages=manifest.live_vintages,
        closure_receipt_hash=manifest.gate_m_receipt_hash,
        closure_manifest_hash=manifest.manifest_hash,
        numerical_environment_hash=manifest.numerical_environment_hash,
        formation_sessions=manifest.formation_sessions,
        ordered_listing_ids=manifest.ordered_listing_ids,
        surface_bindings=tuple(
            HeterogeneousSurfaceBinding(
                component_id=component.component_id,
                vintage=surface.vintage,
                formation_session=session,
                source_binding_hash=surface.source_binding_hash,
                feature_values_hash=surface.feature_values_hashes[index],
            )
            for component in manifest.components
            for surface in component.surfaces
            for index, session in enumerate(manifest.formation_sessions)
        ),
        momentum_source_identity_hash=manifest.momentum.source_identity_hash,
        momentum_value_bindings=tuple(
            HeterogeneousLaneValueBinding(
                formation_session=session, value_hash=manifest.momentum.value_hashes[index]
            )
            for index, session in enumerate(manifest.formation_sessions)
        ),
    )
    try:
        owner = AlphaRuntimeHeterogeneousPredictionOwner()
    except HeterogeneousScoringError as error:
        raise HeterogeneousReplayError(
            "alpha_research.heterogeneous_closure_numerical_environment_invalid"
        ) from error
    return AdmittedHeterogeneousCurrentClosure(
        root=resolved, manifest=manifest, authority=authority, _models=models, _owner=owner
    )


__all__ = [
    "CLOSURE_MANIFEST_FILE_NAME",
    "AdmittedHeterogeneousCurrentClosure",
    "AdmittedSuccessorEvidence",
    "AlphaRuntimeHeterogeneousPredictionOwner",
    "HeterogeneousClosureChild",
    "HeterogeneousClosureComponent",
    "HeterogeneousClosureFile",
    "HeterogeneousClosureMomentum",
    "HeterogeneousClosureSurface",
    "HeterogeneousCurrentClosureManifest",
    "HeterogeneousReplayError",
    "admit_heterogeneous_current_closure",
    "admit_successor_evidence",
    "read_successor_evidence",
]
