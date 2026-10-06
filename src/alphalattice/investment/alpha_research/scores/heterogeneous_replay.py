"""Fail-closed readback of the heterogeneous strategy's admitted evidence.

Two roots, one discipline. The Gate I research package is historical value
evidence; the sealed current closure is the product-readable materialisation
of the forty-eight admitted children, their per-vintage Feature surfaces and
the raw momentum lane, derived once under a granted authority from the exact
files the Gate M source readback names. Neither is installation authority by
itself: the successor recipe in :mod:`heterogeneous_product` supplies the
strategy, and each reader here proves a caller opened exactly the evidence its
manifest names -- every file by hash, every payload by content identity, every
pin against the installed constants -- before anything numerical is built.
Paths are explicit caller inputs. Nothing is discovered, copied or silently
substituted from the absolute paths serialised by a research run.

The current closure's reader is also the only place the installed Alpha
runtime is reached from the product path: one prediction owner that reopens a
payload through the installed adapter and refuses a numerical environment
other than the one the closure was sealed under.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
    COMPONENT_IDS,
    INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
    LIVE_COMPONENT_MODEL_SET_HASHES,
    LIVE_MODEL_CLOSURE_MANIFEST_HASH,
    LIVE_MODEL_VINTAGES,
    LIVE_SCORE_CLOSURE_RECEIPT_HASH,
    LIVE_SOURCE_READBACK_SHA256,
    SUCCESSOR_PACKAGE_HASH,
    SUCCESSOR_REPLAY_RESULT_HASH,
    SUCCESSOR_REPORT_CONTRACT_HASH,
    SUCCESSOR_VALIDATION_CONTRACT_HASH,
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
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]

_PACKAGE_FILE_SHA256 = "4c43e0d46df636e5ef8f8ba8a1f1642874f630e52264fb028ed77b2b477f7a19"
_WEIGHTS_SHA256 = "2d2828ab2d709d2974aec57eb345dcc1884bfe4128166bb236eddea409d21400"
_LEDGER_SHA256 = "3a625328342da27ed5ade9df1f9398b702adcb75d07244c725244977033bbacb"
_REPORT_FILE_SHA256 = "ed5e292917f7e865cc0e7f3bc7362e29b077ea929054ed6fc877129dc6e67f99"
_VALIDATION_FILE_SHA256 = "ecfacc31d9cac0ed45e821401133d12ae80f7157b2b94dbceec287ae3ce3753c"
_REPLAY_FILE_SHA256 = "e1195d370d8a10e1937d83e650e20adaeac7b20b12a71ffe71c2a9488efea32c"


class HeterogeneousReplayError(ValueError):
    """Stable refusal for package, array, axis or readback drift."""


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise HeterogeneousReplayError("alpha_research.heterogeneous_evidence_absent")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HeterogeneousReplayError("alpha_research.heterogeneous_json_invalid") from error
    if not isinstance(value, dict):
        raise HeterogeneousReplayError("alpha_research.heterogeneous_json_object_required")
    return cast(dict[str, Any], value)


def _sealed(payload: Mapping[str, Any], field: str, expected: str) -> None:
    measured = canonical_hash({key: value for key, value in payload.items() if key != field})
    if payload.get(field) != expected or measured != expected:
        raise HeterogeneousReplayError("alpha_research.heterogeneous_evidence_identity_invalid")


def _immutable_memmap(path: Path, *, shape: tuple[int, ...], sha256: str) -> FloatArray:
    if _sha256(path) != sha256:
        raise HeterogeneousReplayError("alpha_research.heterogeneous_array_hash_invalid")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.shape != shape or value.dtype != np.dtype("float64") or value.flags.writeable:
        raise HeterogeneousReplayError("alpha_research.heterogeneous_array_contract_invalid")
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class AdmittedSuccessorEvidence:
    """Exact historical component scores owned by Gate I, over one proven axis.

    The package's merged weights and ledger are still hash-verified on the way
    in -- they are part of the package identity -- but they are not loaded: the
    product forms its own books from the component scores, and the Gate K
    parity runner that read the sealed weights is retired.
    """

    root: Path
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    component_scores: Mapping[ComponentId, FloatArray]

    def component(self, component_id: ComponentId) -> FloatArray:
        """Read one component score vector from admitted replay evidence.

        Args:
            component_id: Component identifier selecting the retained score vector.

        Returns:
            Retained component score array.

        Raises:
            HeterogeneousReplayError: The component score is absent from this evidence.
        """
        try:
            return self.component_scores[component_id]
        except KeyError as error:
            raise HeterogeneousReplayError(
                "alpha_research.heterogeneous_component_score_absent"
            ) from error


def admit_successor_evidence(
    root: Path,
    *,
    formation_sessions: tuple[date, ...],
    ordered_listing_ids: tuple[str, ...],
    component_score_paths: Mapping[ComponentId, Path],
) -> AdmittedSuccessorEvidence:
    """Open one exact package and its four explicitly supplied score surfaces."""
    resolved = root.resolve()
    files = {
        "package": resolved / "frozen-strategy-package.json",
        "weights": resolved / "frozen-strategy-weights.npy",
        "ledger": resolved / "frozen-strategy-ledger.parquet",
        "report": resolved / "report-projection-contract.json",
        "validation": resolved / "validation-admission-contract.json",
        "replay": resolved / "gate-i-replay-result.json",
    }
    expected_file_hashes = {
        "package": _PACKAGE_FILE_SHA256,
        "weights": _WEIGHTS_SHA256,
        "ledger": _LEDGER_SHA256,
        "report": _REPORT_FILE_SHA256,
        "validation": _VALIDATION_FILE_SHA256,
        "replay": _REPLAY_FILE_SHA256,
    }
    for key, expected in expected_file_hashes.items():
        if _sha256(files[key]) != expected:
            raise HeterogeneousReplayError("alpha_research.heterogeneous_package_file_hash_invalid")

    package = _json(files["package"])
    _sealed(package, "package_hash", SUCCESSOR_PACKAGE_HASH)
    report = _json(files["report"])
    _sealed(report, "contract_hash", SUCCESSOR_REPORT_CONTRACT_HASH)
    validation = _json(files["validation"])
    _sealed(validation, "contract_hash", SUCCESSOR_VALIDATION_CONTRACT_HASH)
    replay = _json(files["replay"])
    _sealed(replay, "result_hash", SUCCESSOR_REPLAY_RESULT_HASH)

    strategy = cast(Mapping[str, Any], package.get("strategy_spec", {}))
    if (
        strategy.get("strategy_id") != INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.strategy_id
        or strategy.get("weighting_rule") != "ew"
        or strategy.get("hazard_overlay") != "NONE"
        or strategy.get("merge_semantics") != "POST_TRADE_LISTING_WEIGHTS_THEN_RECOMPUTE_ECONOMICS"
    ):
        raise HeterogeneousReplayError("alpha_research.heterogeneous_strategy_contract_invalid")
    allocations = cast(Mapping[str, Any], strategy.get("components", {}))
    if tuple(allocations) != COMPONENT_IDS or any(
        float(allocations[value]) != 0.25 for value in COMPONENT_IDS
    ):
        raise HeterogeneousReplayError("alpha_research.heterogeneous_component_set_invalid")

    axis = cast(Mapping[str, Any], package.get("axis", {}))
    expected_shape = (len(formation_sessions), len(ordered_listing_ids))
    if expected_shape != (1751, 466):
        raise HeterogeneousReplayError("alpha_research.heterogeneous_axis_shape_invalid")
    if (
        axis.get("session_count") != expected_shape[0]
        or axis.get("listing_count") != expected_shape[1]
        or axis.get("session_axis_hash")
        != canonical_hash(tuple(value.isoformat() for value in formation_sessions))
        or axis.get("listing_axis_hash") != canonical_hash(ordered_listing_ids)
    ):
        raise HeterogeneousReplayError("alpha_research.heterogeneous_axis_identity_invalid")

    declared_components = cast(Mapping[str, Any], package.get("components", {}))
    if tuple(component_score_paths) != COMPONENT_IDS or tuple(declared_components) != COMPONENT_IDS:
        raise HeterogeneousReplayError("alpha_research.heterogeneous_component_set_invalid")
    loaded: dict[ComponentId, FloatArray] = {}
    for component_id in COMPONENT_IDS:
        declaration = cast(Mapping[str, Any], declared_components[component_id])
        artifact = cast(Mapping[str, Any], declaration.get("score_artifact", {}))
        recipe = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(component_id)
        if (
            artifact.get("sha256") != recipe.evidence_daily_parquet_sha256
            or artifact.get("shape") != [1751, 466]
            or artifact.get("dtype") != "float64"
        ):
            raise HeterogeneousReplayError(
                "alpha_research.heterogeneous_component_declaration_invalid"
            )
        loaded[component_id] = _immutable_memmap(
            component_score_paths[component_id],
            shape=expected_shape,
            sha256=recipe.evidence_daily_parquet_sha256,
        )

    return AdmittedSuccessorEvidence(
        root=resolved,
        formation_sessions=formation_sessions,
        ordered_listing_ids=ordered_listing_ids,
        component_scores=loaded,
    )


# ------------------------------------------------------- the current closure


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


_HEX64 = r"^[0-9a-f]{64}$"


class HeterogeneousClosureFile(_Contract):
    """One artifact under the closure root, by relative path, bytes and digest."""

    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=_HEX64)
    byte_count: int = Field(ge=1)

    @field_validator("relative_path")  # type: ignore[untyped-decorator]
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
    feature_values_hashes: tuple[str, ...] = Field(min_length=1)
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
    value_hashes: tuple[str, ...] = Field(min_length=1)
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
    schema_version: Literal[1] = 1
    strategy_hash: str = Field(pattern=_HEX64)
    gate_m_receipt_hash: str = Field(pattern=_HEX64)
    gate_m_source_readback_sha256: str = Field(pattern=_HEX64)
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

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
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
    if _sha256(path) != declared.sha256:
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
    try:
        manifest = HeterogeneousCurrentClosureManifest.model_validate(_json(path))
    except ValueError as error:
        raise HeterogeneousReplayError(
            "alpha_research.heterogeneous_closure_manifest_invalid"
        ) from error
    strategy = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY
    if (
        # A closure sealed before a recorded move of the strategy's or a component's recipe
        # role still names the installed one (NM1, V451): the heterogeneous closure is a
        # sealed root the tree keeps reading.
        not strategy_recipe_is_current(manifest.strategy_hash, strategy.strategy_hash)
        or manifest.gate_m_receipt_hash != LIVE_SCORE_CLOSURE_RECEIPT_HASH
        or manifest.gate_m_source_readback_sha256 != LIVE_SOURCE_READBACK_SHA256
        or manifest.gate_l_manifest_hash != LIVE_MODEL_CLOSURE_MANIFEST_HASH
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
            or component.model_set_hash != LIVE_COMPONENT_MODEL_SET_HASHES[component.component_id]
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
]
