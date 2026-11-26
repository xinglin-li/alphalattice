"""Relocatable historical authority for post-observed frozen strategies.

The research worktree is an input to one materialization command, never a
runtime dependency.  This module owns the product-side manifest, verifies every
byte that can affect the two historical books, and exposes one parameterized
score source.  Gate V weights travel with the authority as parity oracles; the
score source never reads them while forming a book.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Final, Literal, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioResearchSpec,
    ScoreSourceMode,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    FrozenSharedMarketInputs,
    ScoreSupport,
    SharedPortfolioInputs,
    StrategyComponentResolution,
    StrategyPackageError,
)
from alphalattice.investment.portfolio_strategy_lab.application.tranche_book_execution import (
    CappedSleeveComponent,
    TrancheFormationInputs,
)
from alphalattice.investment.portfolio_strategy_lab.policies.contracts import (
    CausalRankReturnCurveSlice,
)
from alphalattice.investment.portfolio_strategy_lab.policies.tranche_book import (
    FrozenSleeveWeightRule,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]

POST_OBSERVED_AUTHORITY_MANIFEST_NAME: Final = "post-observed-strategy-authority.json"
POST_OBSERVED_RESEARCH_SHA: Final = "6fc04e3412dba5b4f95439bcac64ccf75587a873"
POST_OBSERVED_DESCRIPTOR_SHA256: Final = (
    "7efd550a89b40b254a7ae524edcb62635b40e902530e0295feecae12c066e92a"
)
POST_OBSERVED_QV_MAP_SHA256: Final = (
    "1ca7310f5519dc8090734599d4e1536ada98d9ac11b8d66f7adbd1e59075885b"
)
POST_OBSERVED_MODEL_INDEX_SHA256: Final = (
    "2541eb51f81531060230dd533e872bc296a92ffc9970b3adec7013d5e2819d71"
)
POST_OBSERVED_ASSET_INDEX_SHA256: Final = (
    "8beed2894aa481d8b15500ec68f78fb0387c73a3e303ec95a03e66223bd2f7b0"
)
POST_OBSERVED_FINAL_RECEIPT_SHA256: Final = (
    "a8f065a8da474b6b8247d7f1c7e1bd7e0fd60e06d7369b5b2012b3b29019d05c"
)
POST_OBSERVED_AUTHORITY_HASH: Final = (
    "552f149da1e080bc0b62ec715489a21f979dd68d6ef80efa493221907dd86930"
)

TREND_CANDIDATE_COMPONENT_ID: Final = "G2_R0_TREND"
FAST_REBOUND_COMPONENT_ID: Final = "G6_R0_FAST_REBOUND"
POST_OBSERVED_COMPONENT_IDS: Final = (TREND_CANDIDATE_COMPONENT_ID, FAST_REBOUND_COMPONENT_ID)


class PostObservedAuthorityError(ValueError):
    """Stable refusal for a missing, moved or substituted historical authority."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class FrozenAuthorityArtifact(_Contract):
    """One product-owned file named by relative path and exact content."""

    artifact_id: str = Field(min_length=1, max_length=120)
    relative_path: str = Field(min_length=1, max_length=512)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(gt=0)
    dtype: str | None = None
    shape: tuple[int, ...] | None = None

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_path_and_array_shape(self) -> Self:
        """Require safe relative artifact paths and paired dtype/shape declarations.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PostObservedAuthorityError: Path is absolute, contains backslashes/dot traversal or
                dtype/shape is declared alone.
        """
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or "\\" in self.relative_path
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            raise PostObservedAuthorityError(
                "portfolio_application.post_observed_artifact_path_invalid"
            )
        if (self.dtype is None) != (self.shape is None):
            raise PostObservedAuthorityError(
                "portfolio_application.post_observed_artifact_shape_invalid"
            )
        return self


class PostObservedStrategyAuthorityManifest(_Contract):
    """Complete runtime authority, with no source-worktree path in its payload."""

    kind: Literal["PostObservedStrategyAuthorityManifest"] = "PostObservedStrategyAuthorityManifest"
    schema_version: Literal[1] = 1
    status: Literal["POST_OBSERVED_HISTORICAL_AUTHORITY"] = "POST_OBSERVED_HISTORICAL_AUTHORITY"
    research_repository_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_descriptor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_qv_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_model_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asset_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_final_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_shared_input_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)
    source_score_prefix_rows: Literal[4] = 4
    maturity_start_formation: Literal[521] = 521
    review_phase: Literal[1] = 1
    decision_cut: Literal["CLOSE_T"] = "CLOSE_T"
    execution: Literal["OPEN_T_PLUS_1"] = "OPEN_T_PLUS_1"
    primary_cost_bps_per_side: Literal[5] = 5
    sensitivity_cost_bps_per_side: Literal[10] = 10
    execution_availability_disposition: Literal[
        "ASSUMED_FULL_FILL_FIXED_COST_HISTORICAL_REPLAY"
    ] = "ASSUMED_FULL_FILL_FIXED_COST_HISTORICAL_REPLAY"
    capacity_disposition: Literal["NOT_ADMITTED"] = "NOT_ADMITTED"
    artifacts: tuple[FrozenAuthorityArtifact, ...] = Field(min_length=1)
    authority_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one post-observed strategy authority manifest.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical authority_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, authority_hash="0" * 64).model_dump(
            mode="json", exclude={"authority_hash"}
        )
        return cls(**identity, authority_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity_and_frozen_source(self) -> Self:
        """Require exact frozen source pins, support axes, artifact set and authority identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PostObservedAuthorityError: Installed research/source pins, authority, ordered unique
                support/count/endpoints, complete sorted artifact set or canonical identity differs.
        """
        required_source = (
            self.research_repository_sha == POST_OBSERVED_RESEARCH_SHA
            and self.source_descriptor_sha256 == POST_OBSERVED_DESCRIPTOR_SHA256
            and self.source_qv_map_sha256 == POST_OBSERVED_QV_MAP_SHA256
            and self.source_model_index_sha256 == POST_OBSERVED_MODEL_INDEX_SHA256
            and self.source_asset_index_sha256 == POST_OBSERVED_ASSET_INDEX_SHA256
            and self.source_final_receipt_sha256 == POST_OBSERVED_FINAL_RECEIPT_SHA256
        )
        if not required_source or self.authority_hash != POST_OBSERVED_AUTHORITY_HASH:
            raise PostObservedAuthorityError(
                "portfolio_application.post_observed_source_authority_invalid"
            )
        sessions = self.formation_sessions
        if (
            sessions != tuple(sorted(sessions))
            or len(set(sessions)) != len(sessions)
            or sessions[0] != date(2019, 8, 9)
            or sessions[-1] != date(2026, 7, 29)
            or len(sessions) != 1_747
        ):
            raise PostObservedAuthorityError(
                "portfolio_application.post_observed_session_axis_invalid"
            )
        listings = self.ordered_listing_ids
        if len(listings) != 466 or len(set(listings)) != len(listings):
            raise PostObservedAuthorityError(
                "portfolio_application.post_observed_listing_axis_invalid"
            )
        ids = tuple(value.artifact_id for value in self.artifacts)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise PostObservedAuthorityError(
                "portfolio_application.post_observed_artifact_set_invalid"
            )
        required = {
            "balanced_daily_10bps",
            "balanced_daily_5bps",
            "balanced_weights_oracle",
            "g2_score_manifest",
            "g2_scores",
            "g6_causal_curve",
            "g6_latest_calibration_row",
            "g6_score_manifest",
            "g6_scores",
            "return_daily_10bps",
            "return_daily_5bps",
            "return_weights_oracle",
            "shared_decision_eligible",
            "shared_execution_available",
            "shared_realized_returns",
        }
        if set(ids) != required:
            raise PostObservedAuthorityError(
                "portfolio_application.post_observed_artifact_set_invalid"
            )
        if self.authority_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"authority_hash"})
        ):
            raise PostObservedAuthorityError(
                "portfolio_application.post_observed_authority_identity_invalid"
            )
        return self

    def artifact(self, artifact_id: str) -> FrozenAuthorityArtifact:
        """Resolve one explicitly declared frozen authority artifact.

        Args:
            artifact_id: Exact registered artifact identity.

        Returns:
            Matching artifact declaration.

        Raises:
            PostObservedAuthorityError: Requested artifact is absent.
        """
        for value in self.artifacts:
            if value.artifact_id == artifact_id:
                return value
        raise PostObservedAuthorityError(
            f"portfolio_application.post_observed_artifact_absent:{artifact_id}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PostObservedAuthorityError(
            "portfolio_application.post_observed_artifact_unreadable"
        ) from error
    return digest.hexdigest()


class AdmittedPostObservedStrategyAuthority:
    """Reader that re-verifies the relocatable bundle before numerical work."""

    def __init__(self, root: Path) -> None:
        """Read and verify frozen strategy authority from a caller-owned evidence copy.

        Args:
            root: Root of the frozen authority evidence copy.

        Raises:
            PostObservedAuthorityError: Manifest read/parse/validation or exact artifact
                verification is refused.
        """
        self.root = root.resolve()
        manifest_path = self.root / POST_OBSERVED_AUTHORITY_MANIFEST_NAME
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.manifest = cast(
                PostObservedStrategyAuthorityManifest,
                PostObservedStrategyAuthorityManifest.model_validate(payload),
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise PostObservedAuthorityError(
                "portfolio_application.post_observed_manifest_refused"
            ) from error
        self.verify()

    @property
    def authority_hash(self) -> str:
        """Read the admitted frozen strategy authority identity.

        Returns:
            Exact manifest authority_hash.
        """
        return self.manifest.authority_hash

    def _path(self, artifact: FrozenAuthorityArtifact) -> Path:
        resolved = (self.root / Path(*PurePosixPath(artifact.relative_path).parts)).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise PostObservedAuthorityError(
                "portfolio_application.post_observed_artifact_escapes_root"
            ) from error
        return resolved

    def verify(self) -> None:
        """Verify every declared byte and every array axis before scoring."""
        for artifact in self.manifest.artifacts:
            path = self._path(artifact)
            try:
                size = path.stat().st_size
            except OSError as error:
                raise PostObservedAuthorityError(
                    f"portfolio_application.post_observed_artifact_unreadable:{artifact.artifact_id}"
                ) from error
            if size != artifact.bytes or _sha256(path) != artifact.sha256:
                raise PostObservedAuthorityError(
                    f"portfolio_application.post_observed_artifact_tampered:{artifact.artifact_id}"
                )
            if artifact.shape is not None:
                self._load_array_artifact(artifact)

    def _load_array_artifact(self, artifact: FrozenAuthorityArtifact) -> npt.NDArray[np.generic]:
        path = self._path(artifact)
        try:
            value = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise PostObservedAuthorityError(
                f"portfolio_application.post_observed_array_unreadable:{artifact.artifact_id}"
            ) from error
        if tuple(value.shape) != artifact.shape or value.dtype.str != artifact.dtype:
            raise PostObservedAuthorityError(
                f"portfolio_application.post_observed_array_axis_invalid:{artifact.artifact_id}"
            )
        return value

    def array(self, artifact_id: str) -> npt.NDArray[np.generic]:
        """Reopen one already-admitted array and recheck its content identity."""
        artifact = self.manifest.artifact(artifact_id)
        if artifact.shape is None:
            raise PostObservedAuthorityError(
                f"portfolio_application.post_observed_artifact_not_array:{artifact_id}"
            )
        path = self._path(artifact)
        if _sha256(path) != artifact.sha256:
            raise PostObservedAuthorityError(
                f"portfolio_application.post_observed_artifact_tampered:{artifact_id}"
            )
        return self._load_array_artifact(artifact)


class PostObservedHistoricalSharedMarketSource:
    """One parameterized shared-lane source for every package in this bundle."""

    def __init__(self, authority: AdmittedPostObservedStrategyAuthority) -> None:
        """Bind shared market readback to already admitted frozen authority.

        Args:
            authority: Verified frozen strategy authority owner.
        """
        self._authority = authority

    @property
    def formation_sessions(self) -> tuple[date, ...]:
        """Read the frozen shared market formation axis.

        Returns:
            Manifest formation sessions.
        """
        return self._authority.manifest.formation_sessions

    @property
    def ordered_listing_ids(self) -> tuple[str, ...]:
        """Read the frozen shared market listing axis.

        Returns:
            Manifest ordered listing identities.
        """
        return self._authority.manifest.ordered_listing_ids

    @property
    def tradability_decision_hash(self) -> str:
        """Read the frozen decision eligibility artifact identity.

        Returns:
            Declared shared_decision_eligible SHA256.
        """
        return self._authority.manifest.artifact("shared_decision_eligible").sha256

    @property
    def execution_outcome_manifest_hash(self) -> str:
        """Read the frozen realized-return artifact identity.

        Returns:
            Declared shared_realized_returns SHA256.
        """
        return self._authority.manifest.artifact("shared_realized_returns").sha256

    def resolve(self) -> FrozenSharedMarketInputs:
        """Verify authority and load exact common eligibility, execution and realized-return arrays.

        Returns:
            Contiguous declared market inputs with no invented ADV lane.

        Raises:
            PostObservedAuthorityError: Arrays differ from common shapes, realized returns are
                nonfinite or authority verification fails.
        """
        self._authority.verify()
        shape = (len(self.formation_sessions), len(self.ordered_listing_ids))
        decision = np.asarray(self._authority.array("shared_decision_eligible"), dtype=np.bool_)
        execution = np.asarray(self._authority.array("shared_execution_available"), dtype=np.bool_)
        realized = np.asarray(self._authority.array("shared_realized_returns"), dtype=np.float64)
        if (
            decision.shape != shape
            or execution.shape != shape
            or realized.shape != shape
            or not np.isfinite(realized).all()
        ):
            raise PostObservedAuthorityError(
                "portfolio_application.post_observed_shared_axis_invalid"
            )
        return FrozenSharedMarketInputs(
            formation_sessions=self.formation_sessions,
            ordered_listing_ids=self.ordered_listing_ids,
            decision_eligible=np.ascontiguousarray(decision),
            execution_available=np.ascontiguousarray(execution),
            realized_simple_returns=np.ascontiguousarray(realized),
            causal_adv20=None,
        )


@dataclass(frozen=True, slots=True)
class FrozenHistoricalComponentRecipe:
    """One component declaration consumed by the shared capped-sleeve owner."""

    component_id: str
    allocation_basis_points: int
    weight_rule: FrozenSleeveWeightRule
    outsider_sentinel: float | None = None


@dataclass(frozen=True, slots=True)
class FrozenHistoricalBookRecipe:
    """A generic historical package recipe, independent of a strategy name."""

    strategy_id: str
    components: tuple[FrozenHistoricalComponentRecipe, ...]
    top_k: int = 35
    exit_rank: int = 70
    tranches: int = 3
    review_phase: int = 1
    aggregate_name_cap: float = 0.06
    aggregate_cap_start_formation: int = 521
    sizing_activation_formation: int = 521
    sleeve_cap_equal_weight_multiple: float = 2.1

    @property
    def recipe_hash(self) -> str:
        """Hash the components in order, their allocations and the frozen controls.

        The sleeve, cap and sizing controls are hashed with them; the strategy's and the
        components' names stay out (ID10).

        Returns:
            Canonical recipe identity independent of evidence-copy runtime roots.
        """
        return str(
            canonical_hash(
                {
                    "kind": "FrozenHistoricalBookRecipe",
                    "components": [
                        {
                            "allocation_basis_points": value.allocation_basis_points,
                            "weight_rule": value.weight_rule,
                            "outsider_sentinel": value.outsider_sentinel,
                        }
                        for value in self.components
                    ],
                    "top_k": self.top_k,
                    "exit_rank": self.exit_rank,
                    "tranches": self.tranches,
                    "review_phase": self.review_phase,
                    "aggregate_name_cap": self.aggregate_name_cap,
                    "aggregate_cap_start_formation": self.aggregate_cap_start_formation,
                    "sizing_activation_formation": self.sizing_activation_formation,
                    "sleeve_cap_equal_weight_multiple": self.sleeve_cap_equal_weight_multiple,
                }
            )
        )


def _score_receipt(
    *, authority_hash: str, strategy_id: str, session: date, rows: tuple[FloatArray, ...]
) -> str:
    return str(
        canonical_hash(
            {
                "kind": "PostObservedHistoricalScoreReceipt",
                "authority_hash": authority_hash,
                "strategy_id": strategy_id,
                "formation_session": session.isoformat(),
                "component_row_hashes": [
                    hashlib.sha256(np.ascontiguousarray(row, dtype="<f8").tobytes()).hexdigest()
                    for row in rows
                ],
            }
        )
    )


class PostObservedHistoricalScoreSource:
    """One historical score source parameterized by an admitted book recipe."""

    def __init__(
        self,
        *,
        authority: AdmittedPostObservedStrategyAuthority,
        recipe: FrozenHistoricalBookRecipe,
    ) -> None:
        """Bind frozen component scores to a complete unique admitted allocation recipe.

        Args:
            authority: Verified frozen strategy authority owner.
            recipe: Nonempty unique installed component book totaling 10000 basis points.

        Raises:
            StrategyPackageError: Component allocation total, uniqueness or installed component
                membership differs.
        """
        if (
            not recipe.components
            or sum(value.allocation_basis_points for value in recipe.components) != 10_000
        ):
            raise StrategyPackageError(
                "portfolio_application.post_observed_component_allocation_invalid"
            )
        if len({value.component_id for value in recipe.components}) != len(recipe.components):
            raise StrategyPackageError("portfolio_application.post_observed_component_set_invalid")
        if any(
            value.component_id not in POST_OBSERVED_COMPONENT_IDS for value in recipe.components
        ):
            raise StrategyPackageError("portfolio_application.post_observed_component_set_invalid")
        self._authority = authority
        self._recipe = recipe

    @property
    def mode(self) -> ScoreSourceMode:
        """Declare historical array replay for frozen post-observed component scores.

        Returns:
            HISTORICAL_ARRAY_REPLAY.
        """
        return "HISTORICAL_ARRAY_REPLAY"

    @property
    def evidence_identity_hash(self) -> str:
        """Read the admitted frozen evidence authority identity.

        Returns:
            Exact retained authority_hash.
        """
        return self._authority.authority_hash

    def support(self, *, spec: PortfolioResearchSpec) -> ScoreSupport:
        """Expose frozen score axes and shared market source under one authority.

        Args:
            spec: Research controls unused by this frozen support declaration.

        Returns:
            Frozen component score support and shared historical market owner.
        """
        del spec
        manifest = self._authority.manifest
        return ScoreSupport(
            owner_id="post_observed_historical_authority",
            lane="POST_OBSERVED_COMPONENT_SCORE_ARRAY",
            identity_hash=manifest.authority_hash,
            formation_sessions=manifest.formation_sessions,
            ordered_listing_ids=manifest.ordered_listing_ids,
            frozen_shared_market_source=PostObservedHistoricalSharedMarketSource(self._authority),
        )

    def resolve_components(
        self, *, shared: SharedPortfolioInputs, spec: PortfolioResearchSpec
    ) -> StrategyComponentResolution:
        """Verify frozen authority and align score/curve/calibration arrays to shared support.

        Args:
            shared: Exact common portfolio source, listing and execution support.
            spec: Admitted research controls selecting declared input consumption.

        Returns:
            Frozen book component resolution after source-prefix and exact session alignment.

        Raises:
            StrategyPackageError: Shared listing/session support or curve/calibration shape differs.
        """
        del spec
        self._authority.verify()
        manifest = self._authority.manifest
        if shared.ordered_listing_ids != manifest.ordered_listing_ids:
            raise StrategyPackageError("portfolio_application.post_observed_listing_axis_mismatch")
        positions = {session: index for index, session in enumerate(manifest.formation_sessions)}
        if any(session not in positions for session in shared.formation_sessions):
            raise StrategyPackageError("portfolio_application.post_observed_session_axis_mismatch")
        rows: npt.NDArray[np.int64] = np.asarray(
            [positions[session] for session in shared.formation_sessions], dtype=np.int64
        )
        prefix = manifest.source_score_prefix_rows
        score_arrays: dict[str, FloatArray] = {}
        for component in self._recipe.components:
            raw = np.asarray(
                self._authority.array(
                    "g2_scores"
                    if component.component_id == TREND_CANDIDATE_COMPONENT_ID
                    else "g6_scores"
                ),
                dtype=np.float64,
            )
            score_arrays[component.component_id] = np.ascontiguousarray(raw[prefix + rows])
        curve = np.asarray(self._authority.array("g6_causal_curve"), dtype=np.float64)
        latest = np.asarray(self._authority.array("g6_latest_calibration_row"), dtype=np.int64)
        if curve.shape != (len(manifest.formation_sessions), 20) or latest.shape != (
            len(manifest.formation_sessions),
        ):
            raise StrategyPackageError("portfolio_application.post_observed_curve_axis_invalid")
        return resolve_frozen_book_components(
            recipe=self._recipe,
            authority_hash=manifest.authority_hash,
            authority_sessions=manifest.formation_sessions,
            ordered_listing_ids=manifest.ordered_listing_ids,
            score_arrays=score_arrays,
            curve=curve,
            latest=latest,
            shared=shared,
        )


def resolve_frozen_book_components(
    *,
    recipe: FrozenHistoricalBookRecipe,
    authority_hash: str,
    authority_sessions: tuple[date, ...],
    ordered_listing_ids: tuple[str, ...],
    score_arrays: dict[str, FloatArray],
    curve: FloatArray,
    latest: npt.NDArray[np.int64],
    shared: SharedPortfolioInputs,
) -> StrategyComponentResolution:
    """One frozen book construction rule, over independently verified score sources."""
    positions = {day: i for i, day in enumerate(authority_sessions)}
    if shared.ordered_listing_ids != ordered_listing_ids or any(
        day not in positions for day in shared.formation_sessions
    ):
        raise StrategyPackageError("portfolio_application.frozen_book_axis_mismatch")
    rows: npt.NDArray[np.int64] = np.asarray(
        [positions[day] for day in shared.formation_sessions], dtype=np.int64
    )
    if (
        curve.shape != (len(authority_sessions), 20)
        or latest.shape != (len(authority_sessions),)
        or any(
            score_arrays[value.component_id].shape
            != (len(shared.formation_sessions), len(ordered_listing_ids))
            for value in recipe.components
        )
    ):
        raise StrategyPackageError("portfolio_application.frozen_book_array_axis_invalid")
    components: list[CappedSleeveComponent] = []
    eligible = np.asarray(shared.decision_eligible, dtype=np.bool_)
    for component in recipe.components:
        component_rows = score_arrays[component.component_id]
        formation_inputs: list[TrancheFormationInputs] = []
        for local_index, session in enumerate(shared.formation_sessions):
            scores = np.asarray(component_rows[local_index], dtype=np.float64)
            decision_eligible = eligible[local_index] & np.isfinite(scores)
            if component.outsider_sentinel is not None:
                decision_eligible &= scores != component.outsider_sentinel
            curve_slice: CausalRankReturnCurveSlice | None = None
            if component.weight_rule == "mu.iv0":
                authority_index = int(rows[local_index])
                bucket_means = np.asarray(curve[authority_index], dtype=np.float64)
                if authority_index >= recipe.sizing_activation_formation and (
                    not np.isfinite(bucket_means).all() or latest[authority_index] < 0
                ):
                    raise StrategyPackageError("portfolio_application.post_observed_curve_invalid")
                curve_slice = CausalRankReturnCurveSlice(
                    formation_index=local_index,
                    formation_session=session,
                    bucket_means=bucket_means,
                    bucket_support_counts=tuple(0 for _ in range(bucket_means.size)),
                    admitted_formation_count=max(0, int(latest[authority_index]) + 1),
                    disposition=(
                        "AVAILABLE"
                        if authority_index >= recipe.sizing_activation_formation
                        else "INSUFFICIENT_MATURED_FORMATION_HISTORY"
                    ),
                    curve_hash=str(
                        canonical_hash(
                            {
                                "authority_hash": authority_hash,
                                "formation_session": session.isoformat(),
                                "bucket_means": bucket_means.tolist(),
                                "latest_calibration_row": int(latest[authority_index]),
                            }
                        )
                    ),
                )
            formation_inputs.append(
                TrancheFormationInputs(
                    formation_session=session,
                    scores=scores,
                    decision_eligible=np.asarray(decision_eligible, dtype=np.bool_),
                    risk_allocation=None,
                    risk_attribution=None,
                    causal_rank_return_curve=curve_slice,
                )
            )
        components.append(
            CappedSleeveComponent(
                component_id=component.component_id,
                allocation_basis_points=component.allocation_basis_points,
                top_k=recipe.top_k,
                exit_rank=recipe.exit_rank,
                tranches=recipe.tranches,
                aggregate_name_cap=recipe.aggregate_name_cap,
                aggregate_cap_start_formation=recipe.aggregate_cap_start_formation,
                formations=tuple(formation_inputs),
                ordered_listing_ids=ordered_listing_ids,
                weight_rule=component.weight_rule,
                review_phase=recipe.review_phase,
                sizing_activation_formation=(
                    recipe.sizing_activation_formation
                    if component.weight_rule == "mu.iv0"
                    else None
                ),
                sleeve_cap_equal_weight_multiple=(recipe.sleeve_cap_equal_weight_multiple),
            )
        )
    receipts = tuple(
        _score_receipt(
            authority_hash=authority_hash,
            strategy_id=recipe.strategy_id,
            session=session,
            rows=tuple(
                np.asarray(score_arrays[value.component_id][index], dtype=np.float64)
                for value in recipe.components
            ),
        )
        for index, session in enumerate(shared.formation_sessions)
    )
    return StrategyComponentResolution(components=tuple(components), score_receipt_hashes=receipts)


def admit_post_observed_strategy_authority(
    root: Path,
) -> AdmittedPostObservedStrategyAuthority:
    """Public installation seam for the one product-owned historical bundle."""
    return AdmittedPostObservedStrategyAuthority(root)


__all__ = [
    "FAST_REBOUND_COMPONENT_ID",
    "POST_OBSERVED_ASSET_INDEX_SHA256",
    "POST_OBSERVED_AUTHORITY_HASH",
    "POST_OBSERVED_AUTHORITY_MANIFEST_NAME",
    "POST_OBSERVED_COMPONENT_IDS",
    "POST_OBSERVED_DESCRIPTOR_SHA256",
    "POST_OBSERVED_FINAL_RECEIPT_SHA256",
    "POST_OBSERVED_MODEL_INDEX_SHA256",
    "POST_OBSERVED_QV_MAP_SHA256",
    "POST_OBSERVED_RESEARCH_SHA",
    "TREND_CANDIDATE_COMPONENT_ID",
    "AdmittedPostObservedStrategyAuthority",
    "FrozenAuthorityArtifact",
    "FrozenHistoricalBookRecipe",
    "FrozenHistoricalComponentRecipe",
    "PostObservedAuthorityError",
    "PostObservedHistoricalScoreSource",
    "PostObservedHistoricalSharedMarketSource",
    "PostObservedStrategyAuthorityManifest",
    "admit_post_observed_strategy_authority",
]
