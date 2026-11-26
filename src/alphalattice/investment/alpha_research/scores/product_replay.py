"""Reuse the admitted IW184 score cache and aggregate it into one product surface.

This is `DEVELOPMENT_REPLAY`, and the distinction is the whole contract: **no fit
happens here**. The admitted evidence package already contains one score vector
per live model, so the product consumes those exact arrays, verifies that each is
the artifact its receipt describes, and combines them. Nothing in this module
calls a model, and a caller that wanted a fresh fit would need raw Panel inputs
this path never touches.

The three layers are fixed and ordered, and each one is a different kind of
average for a different reason:

1. **within-session percentile rank**, per model, over the live set only -- the
   scores are model outputs on incomparable scales, so they are made comparable
   before anything is averaged;
2. **equal mean across the three seeds** inside a vintage -- seed noise is larger
   than most effects this programme measures, and averaging is the cheapest way
   to remove it, which is why all three seeds are load-bearing;
3. **`4:3:2:1` weighted mean across the four live vintages, newest first** --
   the declared decay.

Every admitted cache vector must be wholly finite, matching the producer's own
admission rule. The formation live set is the caller's eligibility projected
onto rows present on all twelve model axes; substituting, filling or dropping a
model and renormalising would each let the surface report a score no admitted
model produced.

Product code never names the evidence location. The host injects a typed
read-only root, which is what keeps a development replay from hard-coding a
worktree that only exists on one machine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Final, Literal, Protocol, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type ProjectionAggregationSemantics = Literal[
    "WITHIN_SESSION_PERCENTILE_RANK_THEN_EQUAL_SEED_MEAN_THEN_WEIGHTED_VINTAGE_MEAN",
    "EQUAL_RAW_SEED_MEAN_THEN_CANDIDATE_PERCENTILE_1_N_TO_1_THEN_WEIGHTED_VINTAGE_MEAN_WITH_OUTSIDER_MINUS_ONE",
]

EVIDENCE_PACKAGE_ID: Final = "iw184-quarterly-5y-ensemble"
INSTALLED_EVIDENCE_MANIFEST_SHA256: Final = (
    "ded3268421a00ead39450eaf4cabc1d9cb844dfc7ea5cd8614af4850f1edb284"
)
"""Content identity of the exact package manifest admitted by the Stage 8 Gate."""

SEED_MODEL_KIND: Final = "IW184Fixed1260QuarterlySeedModel"
PREDICTION_AXIS_KIND: Final = "IW184QuarterlyPredictionAxis"

SCORE_AGGREGATION_SEMANTICS: Final[ProjectionAggregationSemantics] = (
    "WITHIN_SESSION_PERCENTILE_RANK_THEN_EQUAL_SEED_MEAN_THEN_WEIGHTED_VINTAGE_MEAN"
)
"""The ordered three-layer statement, so a reader cannot infer a different order."""

SPECIALIST_SCORE_AGGREGATION_SEMANTICS: Final[ProjectionAggregationSemantics] = (
    "EQUAL_RAW_SEED_MEAN_THEN_CANDIDATE_PERCENTILE_1_N_TO_1_THEN_"
    "WEIGHTED_VINTAGE_MEAN_WITH_OUTSIDER_MINUS_ONE"
)

MINIMUM_LIVE_NAMES: Final = 2
"""A percentile rank over fewer than two names is not a ranking."""


class AlphaScoreReplayError(ValueError):
    """Stable refusal for an evidence, identity, axis or support failure."""


class AlphaProductRecipeView(Protocol):
    """The replay-only view of the frozen recipe.

    Keeping this structural prevents a score reader on the public executor path
    from importing the estimator adapter and runtime that validate fresh-fit
    parameters. Replay consumes their sealed identities; it never fits them.
    """

    recipe_hash: str
    feature_axis_id: str
    feature_axis_hash: str
    feature_count: int
    training_window_sessions: int
    purge_sessions: int
    refit_quarter_start_months: tuple[int, ...]
    seeds: tuple[int, ...]
    vintage_count: int
    vintage_weights: tuple[int, ...]
    live_model_count: int
    score_aggregation: str
    evidence_disposition: str
    evidence_daily_parquet_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(values: npt.NDArray[np.generic], *, dtype: npt.DTypeLike) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def _frozen(values: npt.ArrayLike, *, dtype: npt.DTypeLike) -> npt.NDArray[np.generic]:
    """Copy into an immutable bytes-backed array whose base cannot be thawed."""

    owned = np.array(values, dtype=dtype, copy=True)
    return np.frombuffer(owned.tobytes(), dtype=dtype).reshape(owned.shape)


def _is_immutable_bytes_backed(values: npt.NDArray[np.generic]) -> bool:
    current: object = values
    while isinstance(current, np.ndarray):
        if current.flags.writeable:
            return False
        current = current.base
    return isinstance(current, bytes)


def quarter_start(session: date) -> str:
    """The refit quarter containing a session, as ``YYYY-MM``."""
    return f"{session.year:04d}-{1 + 3 * ((session.month - 1) // 3):02d}"


def shift_quarter(value: str, offset: int) -> str:
    """Shift a year-month label by a signed number of calendar quarters.

    Args:
        value: Year-month label whose leading year and month are parsed.
        offset: Signed number of three-month shifts.

    Returns:
        Shifted year-month label with year rollover applied.

    Raises:
        ValueError: The year or month substring cannot be parsed as an integer.
    """
    year, month = int(value[:4]), int(value[5:7])
    ordinal = year * 12 + month - 1 + 3 * offset
    return f"{ordinal // 12:04d}-{ordinal % 12 + 1:02d}"


def live_vintages(session: date, *, count: int) -> tuple[str, ...]:
    """The current quarter and the preceding ones, newest first.

    Newest first because the declared weights are newest first. Reversing one
    without the other silently inverts the decay, and both orderings produce a
    plausible surface.
    """
    if count < 1:
        raise AlphaScoreReplayError("alpha_research.replay_vintage_count_invalid")
    return calendar_model_vintages(session, count=count, month_interval=3, anchor_month=1)


def calendar_model_vintages(
    session: date, *, count: int, month_interval: int, anchor_month: int
) -> tuple[str, ...]:
    """One calendar for frozen quarterly replay and declared renewal policies."""
    if count < 1 or month_interval not in {1, 3, 12} or not 1 <= anchor_month <= 12:
        raise AlphaScoreReplayError("alpha_research.model_calendar_invalid")
    month = session.year * 12 + session.month - 1
    newest = month - (month - (anchor_month - 1)) % month_interval
    values = tuple(newest - i * month_interval for i in range(count))
    return tuple(f"{value // 12:04d}-{value % 12 + 1:02d}" for value in values)


def percentile_rank(values: FloatArray) -> FloatArray:
    """Within-session percentile rank on ``[0, 1]`` with average ties.

    Average ties rather than ordinal: two names a model scored identically must
    receive identical weight, and an ordinal rank would break that tie by
    position on the listing axis, which is not information.
    """
    count = values.size
    if count < MINIMUM_LIVE_NAMES:
        return np.zeros(count, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(count, dtype=np.float64)
    ranks[order] = np.arange(1.0, count + 1.0, dtype=np.float64)
    # Average the ranks inside each tie group, which is what
    # ``scipy.stats.rankdata(method="average")`` does.
    sorted_values = values[order]
    start = 0
    for index in range(1, count + 1):
        if index == count or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                group = order[start:index]
                ranks[group] = ranks[group].mean()
            start = index
    return np.asarray((ranks - 1.0) / (count - 1.0), dtype=np.float64)


class PredictionAxisReceipt(BaseModel):  # type: ignore[misc]
    """One vintage's prediction row axis, as its receipt describes it."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    kind: Literal["IW184QuarterlyPredictionAxis"]
    quarter_start: str
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(ge=1)
    transform_session_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    operational_months: tuple[str, ...]
    prediction_months: tuple[str, ...]


class EvidenceDefinition(BaseModel):  # type: ignore[misc]
    """The product fields the exact package manifest must agree with."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    feature_axis: str
    feature_count: int
    live_model_count: int
    training_window_sessions: int
    purge_session_count: int
    seeds: tuple[int, ...]
    vintage_count: int
    vintage_weights: tuple[int, ...]
    seed_aggregation: str
    score_aggregation: str


class DailyEvidenceLineage(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    row_count: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ManifestSeedLineage(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimator_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_first: date
    training_last: date
    training_row_count: int = Field(ge=1)


class ManifestVintageLineage(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction_axis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(ge=1)
    seeds: dict[str, ManifestSeedLineage]


class ProductEvidenceManifest(BaseModel):  # type: ignore[misc]
    """The exact package-level trust anchor admitted by the Stage 8 Gate."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    kind: Literal["IW184Fixed1260QuarterlySeed3Vintage4WeightSensitivityDiagnostic"]
    disposition: Literal["DEVELOPMENT_DIAGNOSTIC_CANDIDATE_NOT_INSTALLED"]
    definition: EvidenceDefinition
    daily_evidence: DailyEvidenceLineage
    model_lineage: dict[str, ManifestVintageLineage]


class SeedModelReceipt(BaseModel):  # type: ignore[misc]
    """One live model: which fit produced it, on which axis, from which recipe."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    kind: Literal["IW184Fixed1260QuarterlySeedModel"]
    candidate: str
    candidate_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_feature_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_view_id: str
    annual_root_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_resolution_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimator_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_axis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    quarter_start: str
    prediction_row_count: int = Field(ge=1)
    training_first: date
    training_last: date
    training_session_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_row_count: int = Field(ge=1)
    purge_session: date
    training_window_sessions: int
    purge_session_count: int
    prediction_months: tuple[str, ...]
    operational_months: tuple[str, ...]
    transform_session_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_aggregation: str

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_receipt(self) -> Self:
        """Verify the receipt's Feature axis and canonical fit identity.

        Returns:
            This receipt after its declared lineage fields reproduce its identity.

        Raises:
            AlphaScoreReplayError: The candidate axis disagrees or the canonical
                fit identity differs from the stored identity.
        """
        if self.candidate_axis_hash != self.ordered_feature_ids_hash:
            raise AlphaScoreReplayError("alpha_research.replay_receipt_axis_disagrees")
        identity = {
            "kind": self.kind,
            "candidate": self.candidate,
            "candidate_axis_hash": self.candidate_axis_hash,
            "feature_view_id": self.feature_view_id,
            "annual_root_hash": self.annual_root_hash,
            "source_resolution_hash": self.source_resolution_hash,
            "recipe_hash": self.recipe_hash,
            "seed": self.seed,
            "quarter_start": self.quarter_start,
            "training_window_sessions": self.training_window_sessions,
            "training_first": self.training_first.isoformat(),
            "training_last": self.training_last.isoformat(),
            "training_session_axis_hash": self.training_session_axis_hash,
            "purge_session": self.purge_session.isoformat(),
            "purge_session_count": self.purge_session_count,
            "prediction_months": list(self.prediction_months),
            "operational_months": list(self.operational_months),
            "transform_session_axis_hash": self.transform_session_axis_hash,
            "score_aggregation": self.score_aggregation,
        }
        if canonical_hash(identity) != self.identity_hash:
            raise AlphaScoreReplayError("alpha_research.replay_receipt_identity_invalid")
        return self


@dataclass(frozen=True, slots=True)
class LoadedPredictionAxis:
    """One verified vintage axis, read and frozen once."""

    receipt: PredictionAxisReceipt
    row_sessions: tuple[date, ...]
    row_listing_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LoadedSeedModel:
    """One verified model's scores on its vintage row axis."""

    vintage: str
    seed: int
    receipt: SeedModelReceipt
    row_sessions: tuple[date, ...]
    row_listing_ids: tuple[str, ...]
    scores: FloatArray


@dataclass(frozen=True, slots=True)
class AlphaEvidenceClosure:
    """Read-only proof that the manifest and every child it names reopened."""

    manifest_sha256: str
    child_count: int
    closure_hash: str


@dataclass
class AdmittedProductEvidence:
    """A read-only reader over the admitted package, injected by the host.

    The root arrives as a parameter and is never named in this module. A
    development replay that hard-coded a worktree would run on one machine and
    fail everywhere else, and the failure would look like missing evidence
    rather than a missing injection.
    """

    root: Path
    recipe: AlphaProductRecipeView
    expected_manifest_sha256: str
    _manifest: ProductEvidenceManifest | None = field(default=None, repr=False)
    _axis_cache: dict[str, LoadedPredictionAxis] = field(default_factory=dict, repr=False)
    _model_cache: dict[tuple[str, int], LoadedSeedModel] = field(default_factory=dict, repr=False)
    _reads: list[tuple[str, int]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Require the injected manifest identity to be a hexadecimal SHA-256.

        Raises:
            AlphaScoreReplayError: The expected manifest identity is malformed.
        """
        if len(self.expected_manifest_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.expected_manifest_sha256
        ):
            raise AlphaScoreReplayError("alpha_research.replay_manifest_identity_invalid")

    def manifest(self) -> ProductEvidenceManifest:
        """Load the package trust anchor before trusting any sibling receipt."""
        if self._manifest is not None:
            return self._manifest
        path = self.root / f"{EVIDENCE_PACKAGE_ID}.json"
        if not path.is_file():
            raise AlphaScoreReplayError("alpha_research.replay_manifest_absent")
        if _sha256(path) != self.expected_manifest_sha256:
            raise AlphaScoreReplayError("alpha_research.replay_manifest_sha256_mismatch")
        try:
            manifest = ProductEvidenceManifest.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except ValueError as error:
            raise AlphaScoreReplayError("alpha_research.replay_manifest_invalid") from error
        definition = manifest.definition
        recipe = self.recipe
        if (
            manifest.disposition != recipe.evidence_disposition
            or definition.feature_axis != recipe.feature_axis_id
            or definition.feature_count != recipe.feature_count
            or definition.live_model_count != recipe.live_model_count
            or definition.training_window_sessions != recipe.training_window_sessions
            or definition.purge_session_count != recipe.purge_sessions
            or definition.seeds != recipe.seeds
            or definition.vintage_count != recipe.vintage_count
            or definition.vintage_weights != recipe.vintage_weights
            or definition.seed_aggregation != "EQUAL_MEAN_WITHIN_VINTAGE"
            or definition.score_aggregation != recipe.score_aggregation
            or manifest.daily_evidence.sha256 != recipe.evidence_daily_parquet_sha256
        ):
            raise AlphaScoreReplayError("alpha_research.replay_manifest_recipe_mismatch")
        self._manifest = manifest
        return manifest

    def verify_closure(self) -> AlphaEvidenceClosure:
        """Open the package trust anchor and every artifact below it.

        This is readback, not scoring: it validates the daily evidence bytes,
        every prediction axis, and every seed receipt/payload named by the
        manifest.  Returning only a compact identity keeps arrays on the Alpha
        side of the boundary while letting a caller distinguish a manifest that
        exists from a package whose children are actually complete.
        """
        manifest = self.manifest()
        root = self.root.resolve()
        daily = (root / manifest.daily_evidence.name).resolve()
        if not daily.is_relative_to(root) or not daily.is_file():
            raise AlphaScoreReplayError("alpha_research.replay_daily_evidence_absent")
        if _sha256(daily) != manifest.daily_evidence.sha256:
            raise AlphaScoreReplayError("alpha_research.replay_daily_evidence_sha256_mismatch")

        expected_seeds = {str(value) for value in self.recipe.seeds}
        children: list[tuple[str, str]] = [("daily_evidence", manifest.daily_evidence.sha256)]
        if not manifest.model_lineage:
            raise AlphaScoreReplayError("alpha_research.replay_manifest_model_lineage_empty")
        if len(manifest.model_lineage) != self.recipe.vintage_count:
            raise AlphaScoreReplayError("alpha_research.replay_manifest_vintage_count_mismatch")
        for vintage, lineage in sorted(manifest.model_lineage.items()):
            if set(lineage.seeds) != expected_seeds:
                raise AlphaScoreReplayError("alpha_research.replay_manifest_seed_set_mismatch")
            axis = self.prediction_axis(vintage)
            children.append((f"axis:{vintage}", axis.receipt.artifact_sha256))
            for seed in self.recipe.seeds:
                model = self.seed_model(vintage=vintage, seed=seed)
                children.append(
                    (
                        f"model:{vintage}:{seed}",
                        canonical_hash(
                            {
                                "payload_sha256": model.receipt.artifact_sha256,
                                "identity_hash": model.receipt.identity_hash,
                            }
                        ),
                    )
                )
        if len(children) != 1 + self.recipe.vintage_count + self.recipe.live_model_count:
            raise AlphaScoreReplayError("alpha_research.replay_manifest_model_count_mismatch")
        identity = {
            "kind": "AlphaEvidenceClosure",
            "manifest_sha256": self.expected_manifest_sha256,
            "children": tuple(children),
        }
        return AlphaEvidenceClosure(
            manifest_sha256=self.expected_manifest_sha256,
            child_count=len(children),
            closure_hash=canonical_hash(identity),
        )

    @property
    def artifact_read_log(self) -> tuple[tuple[str, int], ...]:
        """Every (vintage, seed) actually read from disk, in order.

        Exposed so cache reuse is observable. A reuse claim that cannot be
        measured is a comment.
        """
        return tuple(self._reads)

    def invalidate(self, *, vintage: str, seed: int | None = None) -> None:
        """Drop one model, or one whole vintage, from the cache.

        Targeted because that is what identity change means: a new estimator
        content hash for one seed invalidates that seed, not the other eleven
        models that did not move.
        """
        if seed is None:
            for key in [item for item in self._model_cache if item[0] == vintage]:
                del self._model_cache[key]
            self._axis_cache.pop(vintage, None)
            return
        self._model_cache.pop((vintage, seed), None)

    def _vintage_dir(self, vintage: str) -> Path:
        return self.root / "model-predictions" / vintage

    def prediction_axis(self, vintage: str) -> LoadedPredictionAxis:
        """Read and cache a vintage's verified prediction row axis.

        Args:
            vintage: Exact quarterly vintage key in the admitted manifest.

        Returns:
            Verified receipt, ordered row sessions and listing identifiers. A
            previously admitted axis is reused from this reader's cache.

        Raises:
            AlphaScoreReplayError: The vintage, artifact, receipt, byte digest,
                row uniqueness or session lineage is missing or inconsistent.
        """
        if vintage in self._axis_cache:
            return self._axis_cache[vintage]
        manifest = self.manifest()
        lineage = manifest.model_lineage.get(vintage)
        if lineage is None:
            raise AlphaScoreReplayError("alpha_research.replay_manifest_vintage_absent")
        meta = self.root / "prediction-axes" / f"{vintage}.json"
        payload = self.root / "prediction-axes" / f"{vintage}.npz"
        if not meta.is_file() or not payload.is_file():
            raise AlphaScoreReplayError("alpha_research.replay_prediction_axis_absent")
        try:
            receipt = PredictionAxisReceipt.model_validate(
                json.loads(meta.read_text(encoding="utf-8"))
            )
        except ValueError as error:
            raise AlphaScoreReplayError(
                "alpha_research.replay_prediction_axis_receipt_invalid"
            ) from error
        if receipt.kind != PREDICTION_AXIS_KIND or receipt.quarter_start != vintage:
            raise AlphaScoreReplayError("alpha_research.replay_prediction_axis_identity_invalid")
        if (
            receipt.artifact_sha256 != lineage.prediction_axis_sha256
            or receipt.row_count != lineage.row_count
        ):
            raise AlphaScoreReplayError("alpha_research.replay_manifest_axis_lineage_mismatch")
        if _sha256(payload) != receipt.artifact_sha256:
            raise AlphaScoreReplayError("alpha_research.replay_prediction_axis_sha256_mismatch")
        with np.load(payload, allow_pickle=False) as axis:
            sessions = tuple(
                value.astype("datetime64[D]").astype(date) for value in axis["row_sessions"]
            )
            listing_ids = tuple(str(value) for value in axis["row_listing_ids"])
        if len(sessions) != receipt.row_count or len(listing_ids) != receipt.row_count:
            raise AlphaScoreReplayError("alpha_research.replay_axis_row_count_mismatch")
        if len(set(zip(sessions, listing_ids, strict=True))) != receipt.row_count:
            raise AlphaScoreReplayError("alpha_research.replay_axis_rows_not_unique")
        session_axis = tuple(dict.fromkeys(sessions))
        if canonical_hash([value.isoformat() for value in session_axis]) != (
            receipt.transform_session_axis_hash
        ):
            raise AlphaScoreReplayError("alpha_research.replay_axis_session_identity_mismatch")
        loaded = LoadedPredictionAxis(
            receipt=receipt,
            row_sessions=sessions,
            row_listing_ids=listing_ids,
        )
        self._axis_cache[vintage] = loaded
        return loaded

    def seed_model(self, *, vintage: str, seed: int) -> LoadedSeedModel:
        """Load and verify one live model, reusing an already verified one."""
        key = (vintage, seed)
        if key in self._model_cache:
            return self._model_cache[key]

        manifest = self.manifest()
        vintage_lineage = manifest.model_lineage.get(vintage)
        if vintage_lineage is None:
            raise AlphaScoreReplayError("alpha_research.replay_manifest_vintage_absent")
        manifest_lineage = vintage_lineage.seeds.get(str(seed))
        if manifest_lineage is None:
            raise AlphaScoreReplayError("alpha_research.replay_manifest_seed_absent")
        axis = self.prediction_axis(vintage)
        axis_receipt = axis.receipt
        meta = self._vintage_dir(vintage) / f"seed-{seed}.json"
        payload = self._vintage_dir(vintage) / f"seed-{seed}.npz"
        if not meta.is_file() or not payload.is_file():
            raise AlphaScoreReplayError("alpha_research.replay_seed_model_absent")

        try:
            receipt = SeedModelReceipt.model_validate(json.loads(meta.read_text(encoding="utf-8")))
        except ValueError as error:
            raise AlphaScoreReplayError("alpha_research.replay_seed_receipt_invalid") from error
        if receipt.kind != SEED_MODEL_KIND:
            raise AlphaScoreReplayError("alpha_research.replay_seed_model_kind_invalid")
        if receipt.seed != seed or receipt.quarter_start != vintage:
            raise AlphaScoreReplayError("alpha_research.replay_seed_model_identity_invalid")
        if receipt.candidate != self.recipe.feature_axis_id:
            raise AlphaScoreReplayError("alpha_research.replay_feature_axis_id_mismatch")
        if receipt.candidate_axis_hash != self.recipe.feature_axis_hash:
            raise AlphaScoreReplayError("alpha_research.replay_feature_axis_hash_mismatch")
        if receipt.training_window_sessions != self.recipe.training_window_sessions:
            raise AlphaScoreReplayError("alpha_research.replay_training_window_mismatch")
        if receipt.purge_session_count != self.recipe.purge_sessions:
            raise AlphaScoreReplayError("alpha_research.replay_purge_mismatch")
        if receipt.score_aggregation != self.recipe.score_aggregation:
            raise AlphaScoreReplayError("alpha_research.replay_score_aggregation_mismatch")
        if (
            receipt.prediction_months != axis_receipt.prediction_months
            or receipt.operational_months != axis_receipt.operational_months
            or receipt.transform_session_axis_hash != axis_receipt.transform_session_axis_hash
        ):
            raise AlphaScoreReplayError("alpha_research.replay_seed_axis_semantics_mismatch")
        if receipt.prediction_axis_sha256 != axis_receipt.artifact_sha256:
            raise AlphaScoreReplayError("alpha_research.replay_prediction_axis_binding_invalid")
        if (
            receipt.artifact_sha256 != manifest_lineage.artifact_sha256
            or receipt.estimator_content_hash != manifest_lineage.estimator_content_hash
            or receipt.identity_hash != manifest_lineage.identity_hash
            or receipt.training_first != manifest_lineage.training_first
            or receipt.training_last != manifest_lineage.training_last
            or receipt.training_row_count != manifest_lineage.training_row_count
        ):
            raise AlphaScoreReplayError("alpha_research.replay_manifest_seed_lineage_mismatch")
        if _sha256(payload) != receipt.artifact_sha256:
            raise AlphaScoreReplayError("alpha_research.replay_seed_model_sha256_mismatch")

        with np.load(payload, allow_pickle=False) as bundle:
            scores = np.asarray(bundle["scores"], dtype=np.float64)
            stored_identity = str(bundle["identity_hash"].item())
        if stored_identity != receipt.identity_hash:
            raise AlphaScoreReplayError("alpha_research.replay_identity_hash_mismatch")
        if scores.shape != (receipt.prediction_row_count,):
            raise AlphaScoreReplayError("alpha_research.replay_score_row_count_mismatch")
        if not np.isfinite(scores).all():
            raise AlphaScoreReplayError("alpha_research.replay_seed_model_values_invalid")
        if len(axis.row_sessions) != receipt.prediction_row_count:
            raise AlphaScoreReplayError("alpha_research.replay_axis_row_count_mismatch")

        loaded = LoadedSeedModel(
            vintage=vintage,
            seed=seed,
            receipt=receipt,
            row_sessions=axis.row_sessions,
            row_listing_ids=axis.row_listing_ids,
            scores=np.asarray(_frozen(scores, dtype=np.dtype("<f8")), dtype=np.float64),
        )
        self._model_cache[key] = loaded
        self._reads.append(key)
        return loaded


class AlphaProductScoreProjection(BaseModel):  # type: ignore[misc]
    """One formation's aggregated product score, with the lineage behind it."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    kind: Literal["AlphaProductScoreProjection"] = "AlphaProductScoreProjection"
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    formation_session: date
    ordered_listing_ids: tuple[str, ...]
    live_vintages: tuple[str, ...]
    seeds: tuple[int, ...]
    vintage_weights: tuple[int, ...]
    aggregation_semantics: ProjectionAggregationSemantics = SCORE_AGGREGATION_SEMANTICS
    model_identity_hashes: tuple[str, ...]
    model_lineage_hashes: tuple[str, ...]
    live_name_count: int = Field(ge=0)
    projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    scores: FloatArray = Field(exclude=True)
    live: BoolArray = Field(exclude=True)

    @model_validator(mode="before")  # type: ignore[untyped-decorator]
    @classmethod
    def own_arrays(cls, values: object) -> object:
        """Copy supplied score and live arrays into immutable owned storage.

        Args:
            values: Incoming model values; a non-mapping value passes through.

        Returns:
            Values with supplied arrays frozen in their canonical score and mask dtypes.
        """
        if not isinstance(values, dict):
            return values
        owned = dict(values)
        if "scores" in owned:
            owned["scores"] = _frozen(owned["scores"], dtype=np.dtype("<f8"))
        if "live" in owned:
            owned["live"] = _frozen(owned["live"], dtype=np.dtype("?"))
        return owned

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_projection(self) -> Self:
        """Verify score content against the declared projection identity.

        Returns:
            This projection after its owned arrays and lineage pass verification.

        Raises:
            AlphaScoreReplayError: The projection content or axes are inconsistent.
        """
        self.verify_content()
        return self

    @classmethod
    def create_live(
        cls,
        *,
        recipe_hash: str,
        evidence_manifest_sha256: str,
        formation_session: date,
        ordered_listing_ids: tuple[str, ...],
        live_vintages: tuple[str, ...],
        seeds: tuple[int, ...],
        vintage_weights: tuple[int, ...],
        aggregation_semantics: ProjectionAggregationSemantics,
        model_identity_hashes: tuple[str, ...],
        model_lineage_hashes: tuple[str, ...],
        scores: npt.ArrayLike,
        live: npt.ArrayLike,
    ) -> AlphaProductScoreProjection:
        """Seal an already-composed live score without reopening evidence arrays."""
        frozen_scores = np.asarray(_frozen(scores, dtype=np.dtype("<f8")), dtype=np.float64)
        frozen_live = np.asarray(_frozen(live, dtype=np.dtype("?")), dtype=np.bool_)
        return cls(
            recipe_hash=recipe_hash,
            evidence_manifest_sha256=evidence_manifest_sha256,
            formation_session=formation_session,
            ordered_listing_ids=ordered_listing_ids,
            live_vintages=live_vintages,
            seeds=seeds,
            vintage_weights=vintage_weights,
            aggregation_semantics=aggregation_semantics,
            model_identity_hashes=model_identity_hashes,
            model_lineage_hashes=model_lineage_hashes,
            live_name_count=int(frozen_live.sum()),
            projection_hash=_identity(
                recipe_hash=recipe_hash,
                manifest_sha256=evidence_manifest_sha256,
                session=formation_session,
                listing_ids=ordered_listing_ids,
                vintages=live_vintages,
                seeds=seeds,
                vintage_weights=vintage_weights,
                aggregation_semantics=aggregation_semantics,
                identity_hashes=model_identity_hashes,
                lineage_hashes=model_lineage_hashes,
                scores=frozen_scores,
                live=frozen_live,
            ),
            scores=frozen_scores,
            live=frozen_live,
        )

    def verify_content(self) -> None:
        """Recompute the identity immediately before a consumer reads scores."""
        names = len(self.ordered_listing_ids)
        if names == 0 or len(set(self.ordered_listing_ids)) != names:
            raise AlphaScoreReplayError("alpha_research.replay_projection_listing_axis_invalid")
        if self.scores.shape != (names,) or self.live.shape != (names,):
            raise AlphaScoreReplayError("alpha_research.replay_projection_axis_invalid")
        if not _is_immutable_bytes_backed(self.scores) or not _is_immutable_bytes_backed(self.live):
            raise AlphaScoreReplayError("alpha_research.replay_projection_arrays_mutable")
        finite = np.isfinite(self.scores)
        if not np.array_equal(finite, self.live) or int(self.live.sum()) != self.live_name_count:
            raise AlphaScoreReplayError("alpha_research.replay_projection_support_invalid")
        models = len(self.live_vintages) * len(self.seeds)
        if (
            not self.live_vintages
            or not self.seeds
            or len(self.vintage_weights) != len(self.live_vintages)
            or len(self.model_identity_hashes) != models
            or len(self.model_lineage_hashes) != models
        ):
            raise AlphaScoreReplayError("alpha_research.replay_projection_model_count_invalid")
        expected = _identity(
            recipe_hash=self.recipe_hash,
            manifest_sha256=self.evidence_manifest_sha256,
            session=self.formation_session,
            listing_ids=self.ordered_listing_ids,
            vintages=self.live_vintages,
            seeds=self.seeds,
            vintage_weights=self.vintage_weights,
            aggregation_semantics=self.aggregation_semantics,
            identity_hashes=self.model_identity_hashes,
            lineage_hashes=self.model_lineage_hashes,
            scores=self.scores,
            live=self.live,
        )
        if expected != self.projection_hash:
            raise AlphaScoreReplayError("alpha_research.replay_projection_content_mismatch")


def _identity(
    *,
    recipe_hash: str,
    manifest_sha256: str,
    session: date,
    listing_ids: tuple[str, ...],
    vintages: tuple[str, ...],
    seeds: tuple[int, ...],
    vintage_weights: tuple[int, ...],
    aggregation_semantics: str,
    identity_hashes: tuple[str, ...],
    lineage_hashes: tuple[str, ...],
    scores: FloatArray,
    live: BoolArray,
) -> str:
    return canonical_hash(
        {
            "kind": "AlphaProductScoreProjection",
            "recipe_hash": recipe_hash,
            "evidence_manifest_sha256": manifest_sha256,
            "formation_session": session.isoformat(),
            "ordered_listing_ids": list(listing_ids),
            "live_vintages": list(vintages),
            "seeds": list(seeds),
            "vintage_weights": list(vintage_weights),
            "aggregation_semantics": aggregation_semantics,
            "model_identity_hashes": list(identity_hashes),
            "model_lineage_hashes": list(lineage_hashes),
            "scores": _array_hash(scores, dtype=np.dtype("<f8")),
            "live": _array_hash(live, dtype=np.dtype("?")),
        }
    )


def _model_lineage_hash(model: LoadedSeedModel) -> str:
    return canonical_hash(
        {
            "vintage": model.vintage,
            "seed": model.seed,
            "receipt": model.receipt.model_dump(mode="json"),
        }
    )


def replay_formation_scores(
    *,
    evidence: AdmittedProductEvidence,
    formation_session: date,
    ordered_listing_ids: tuple[str, ...],
    decision_eligible: BoolArray,
) -> AlphaProductScoreProjection:
    """Aggregate the twelve live models into one formation's product score.

    Fails closed on every way the evidence can be wrong: a missing vintage or
    seed, a row axis that disagrees with the decision axis, a formation the
    package does not cover, or a live set too small to rank.
    """
    recipe = evidence.recipe
    evidence.manifest()
    vintages = live_vintages(formation_session, count=recipe.vintage_count)
    eligible = np.asarray(decision_eligible, dtype=np.bool_)
    if eligible.shape != (len(ordered_listing_ids),):
        raise AlphaScoreReplayError("alpha_research.replay_eligibility_axis_invalid")

    live = eligible.copy()
    per_model: dict[tuple[str, int], FloatArray] = {}
    identity_hashes: list[str] = []
    lineage_hashes: list[str] = []
    formation_month = formation_session.strftime("%Y-%m")
    for vintage in vintages:
        for seed in recipe.seeds:
            model = evidence.seed_model(vintage=vintage, seed=seed)
            if formation_month not in model.receipt.operational_months:
                raise AlphaScoreReplayError("alpha_research.replay_formation_not_operational")
            row = _formation_rows(model, formation_session, ordered_listing_ids)
            per_model[(vintage, seed)] = row
            live &= np.isfinite(row)
            identity_hashes.append(model.receipt.identity_hash)
            lineage_hashes.append(_model_lineage_hash(model))

    positions = np.flatnonzero(live)
    scores: FloatArray = np.full(len(ordered_listing_ids), np.nan, dtype=np.float64)
    if positions.size >= MINIMUM_LIVE_NAMES:
        weighted: FloatArray = np.zeros(positions.size, dtype=np.float64)
        total = float(sum(recipe.vintage_weights))
        for vintage, weight in zip(vintages, recipe.vintage_weights, strict=True):
            seed_ranks = np.vstack(
                [percentile_rank(per_model[(vintage, seed)][positions]) for seed in recipe.seeds]
            )
            weighted += float(weight) * seed_ranks.mean(axis=0)
        scores[positions] = weighted / total
    else:
        live = np.zeros(len(ordered_listing_ids), dtype=np.bool_)

    frozen_scores = np.asarray(_frozen(scores, dtype=np.dtype("<f8")), dtype=np.float64)
    frozen_live = np.asarray(_frozen(live, dtype=np.dtype("?")), dtype=np.bool_)
    return AlphaProductScoreProjection(
        recipe_hash=recipe.recipe_hash,
        evidence_manifest_sha256=evidence.expected_manifest_sha256,
        formation_session=formation_session,
        ordered_listing_ids=ordered_listing_ids,
        live_vintages=vintages,
        seeds=recipe.seeds,
        vintage_weights=recipe.vintage_weights,
        model_identity_hashes=tuple(identity_hashes),
        model_lineage_hashes=tuple(lineage_hashes),
        live_name_count=int(frozen_live.sum()),
        projection_hash=_identity(
            recipe_hash=recipe.recipe_hash,
            manifest_sha256=evidence.expected_manifest_sha256,
            session=formation_session,
            listing_ids=ordered_listing_ids,
            vintages=vintages,
            seeds=recipe.seeds,
            vintage_weights=recipe.vintage_weights,
            aggregation_semantics=SCORE_AGGREGATION_SEMANTICS,
            identity_hashes=tuple(identity_hashes),
            lineage_hashes=tuple(lineage_hashes),
            scores=frozen_scores,
            live=frozen_live,
        ),
        scores=frozen_scores,
        live=frozen_live,
    )


def _formation_rows(
    model: LoadedSeedModel,
    formation_session: date,
    ordered_listing_ids: tuple[str, ...],
) -> FloatArray:
    """Project one model's long-format rows onto the decision listing axis.

    A listing absent from this model's rows at the formation stays ``NaN`` and
    drops out of the live set. A non-finite value inside an admitted cache is a
    different condition and is rejected while loading the whole model.
    """

    session_rows = [
        index for index, value in enumerate(model.row_sessions) if value == formation_session
    ]
    if not session_rows:
        raise AlphaScoreReplayError("alpha_research.replay_formation_not_in_prediction_axis")
    by_listing = {model.row_listing_ids[index]: model.scores[index] for index in session_rows}
    if len(by_listing) != len(session_rows):
        raise AlphaScoreReplayError("alpha_research.replay_duplicate_listing_row")
    return np.asarray(
        [by_listing.get(listing_id, np.nan) for listing_id in ordered_listing_ids],
        dtype=np.float64,
    )


class HeterogeneousScoringError(ValueError):
    """Stable refusal for live model, Feature, candidate, or aggregation drift."""


class HeterogeneousEstimatorContent(Protocol):
    """Estimator content exposing its ordered Feature axis and content identity."""

    ordered_feature_ids: tuple[str, ...]
    content_hash: str


class HeterogeneousComponentScoringRecipe(Protocol):
    """Component recipe declaring Feature, seed, vintage and aggregation axes."""

    component_id: str
    recipe_hash: str
    feature_axis_hash: str
    feature_count: int
    ordered_feature_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    vintage_count: int
    vintage_weights: tuple[int, ...]
    candidate_semantics: str
    score_aggregation: str


def _valid_live_hash(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _alpha_live_array_hash(values: npt.NDArray[np.generic]) -> str:
    """The installed Alpha training-value identity, without importing its runtime."""

    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class HeterogeneousVintageFeatureSurface:
    """One formation's full listing axis under one vintage's scaling state."""

    vintage: str
    ordered_listing_ids: tuple[str, ...]
    ordered_feature_ids: tuple[str, ...]
    features: FloatArray
    source_binding_hash: str
    feature_values_hash: str

    @classmethod
    def create(
        cls,
        *,
        vintage: str,
        ordered_listing_ids: tuple[str, ...],
        ordered_feature_ids: tuple[str, ...],
        features: npt.ArrayLike,
        source_binding_hash: str,
    ) -> HeterogeneousVintageFeatureSurface:
        """Own a vintage's Feature matrix and bind its values to the declared axes.

        Args:
            vintage: Year-month vintage label.
            ordered_listing_ids: Unique listing row axis.
            ordered_feature_ids: Unique Feature column axis.
            features: Matrix aligned to the two declared axes.
            source_binding_hash: Exact source binding admitted for this surface.

        Returns:
            Immutable surface with its computed Feature-value digest.

        Raises:
            HeterogeneousScoringError: The axes, vintage, shape or binding are invalid.
        """
        frozen = np.asarray(_frozen(features, dtype=np.dtype("<f8")), dtype=np.float64)
        return cls(
            vintage=vintage,
            ordered_listing_ids=ordered_listing_ids,
            ordered_feature_ids=ordered_feature_ids,
            features=frozen,
            source_binding_hash=source_binding_hash,
            feature_values_hash=_alpha_live_array_hash(frozen),
        )

    def __post_init__(self) -> None:
        """Verify the immutable matrix's shape, unique axes and value/source bindings.

        Raises:
            HeterogeneousScoringError: The surface fails an axis or identity check.
        """
        if (
            len(self.vintage) != 7
            or self.vintage[4] != "-"
            or len(set(self.ordered_listing_ids)) != len(self.ordered_listing_ids)
            or not self.ordered_feature_ids
            or len(set(self.ordered_feature_ids)) != len(self.ordered_feature_ids)
            or self.features.shape != (len(self.ordered_listing_ids), len(self.ordered_feature_ids))
            or self.features.flags.writeable
            or not _valid_live_hash(self.source_binding_hash)
            or self.feature_values_hash != _alpha_live_array_hash(self.features)
        ):
            raise HeterogeneousScoringError(
                "alpha_research.heterogeneous_live_feature_surface_invalid"
            )


@dataclass(frozen=True, slots=True)
class HeterogeneousLiveModel:
    """One vintage and seed's estimator with recipe, training and lineage bindings."""

    vintage: str
    seed: int
    recipe_hash: str
    training_binding_hash: str
    lineage_hash: str
    estimator: HeterogeneousEstimatorContent

    def __post_init__(self) -> None:
        """Require the recipe, training and lineage bindings to be valid digests.

        Raises:
            HeterogeneousScoringError: A declared model identity is malformed.
        """
        if not all(
            _valid_live_hash(value)
            for value in (self.recipe_hash, self.training_binding_hash, self.lineage_hash)
        ):
            raise HeterogeneousScoringError(
                "alpha_research.heterogeneous_live_model_identity_invalid"
            )


@dataclass(frozen=True, slots=True)
class HeterogeneousFormationScoreInput:
    """Owned formation axes, eligibility and Feature/model inputs for live scoring."""

    formation_session: date
    ordered_listing_ids: tuple[str, ...]
    decision_eligible: BoolArray
    raw_12_1_momentum: FloatArray | None
    feature_surfaces: tuple[HeterogeneousVintageFeatureSurface, ...]
    models: tuple[HeterogeneousLiveModel, ...]
    model_set_manifest_hash: str

    @classmethod
    def create(
        cls,
        *,
        formation_session: date,
        ordered_listing_ids: tuple[str, ...],
        decision_eligible: npt.ArrayLike,
        raw_12_1_momentum: npt.ArrayLike | None,
        feature_surfaces: tuple[HeterogeneousVintageFeatureSurface, ...],
        models: tuple[HeterogeneousLiveModel, ...],
        model_set_manifest_hash: str,
    ) -> HeterogeneousFormationScoreInput:
        """Own formation masks and optional momentum while retaining admitted inputs.

        Args:
            formation_session: Formation whose inputs are being declared.
            ordered_listing_ids: Unique listing axis shared by the formation arrays.
            decision_eligible: Eligibility mask on that axis.
            raw_12_1_momentum: Optional raw momentum values on the same axis.
            feature_surfaces: Nonempty set of declared vintage Feature surfaces.
            models: Nonempty set of bound live models.
            model_set_manifest_hash: Exact model-set manifest identity.

        Returns:
            Formation input with immutable eligibility and optional momentum arrays.

        Raises:
            HeterogeneousScoringError: The listing axis, arrays, input sets or
                manifest identity fail validation.
        """
        eligible = np.asarray(_frozen(decision_eligible, dtype=np.dtype("?")), dtype=np.bool_)
        momentum = (
            None
            if raw_12_1_momentum is None
            else np.asarray(_frozen(raw_12_1_momentum, dtype=np.dtype("<f8")), dtype=np.float64)
        )
        return cls(
            formation_session=formation_session,
            ordered_listing_ids=ordered_listing_ids,
            decision_eligible=eligible,
            raw_12_1_momentum=momentum,
            feature_surfaces=feature_surfaces,
            models=models,
            model_set_manifest_hash=model_set_manifest_hash,
        )

    def __post_init__(self) -> None:
        """Check formation axis lengths, immutable arrays and required input bindings.

        Raises:
            HeterogeneousScoringError: The formation has invalid axes or inputs.
        """
        names = len(self.ordered_listing_ids)
        if (
            names < 2
            or len(set(self.ordered_listing_ids)) != names
            or self.decision_eligible.shape != (names,)
            or self.decision_eligible.flags.writeable
            or (
                self.raw_12_1_momentum is not None
                and (
                    self.raw_12_1_momentum.shape != (names,)
                    or self.raw_12_1_momentum.flags.writeable
                )
            )
            or not self.feature_surfaces
            or not self.models
            or not _valid_live_hash(self.model_set_manifest_hash)
        ):
            raise HeterogeneousScoringError(
                "alpha_research.heterogeneous_live_formation_axis_invalid"
            )


class HeterogeneousPredictionOwner(Protocol):
    """Injected owner that predicts one bound model on its supplied Feature matrix."""

    def __call__(
        self,
        *,
        component: HeterogeneousComponentScoringRecipe,
        model: HeterogeneousLiveModel,
        features: FloatArray,
    ) -> FloatArray:
        """Predict through the admitted component-specific numerical owner.

        Args:
            component: Recipe declaring the component's scientific scoring semantics.
            model: Bound vintage and seed estimator to evaluate.
            features: Feature matrix supplied for that model's prediction.

        Returns:
            Prediction values produced by the admitted numerical owner.
        """
        ...


def _specialist_percentile(values: FloatArray) -> FloatArray:
    count = len(values)
    if count < 2 or not np.isfinite(values).all():
        raise HeterogeneousScoringError("alpha_research.heterogeneous_live_rank_input_invalid")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(count, dtype=np.float64)
    ranks[order] = np.arange(1.0, count + 1.0, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for index in range(1, count + 1):
        if index == count or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                group = order[start:index]
                ranks[group] = ranks[group].mean()
            start = index
    return np.asarray(ranks / count, dtype=np.float64)


def _heterogeneous_candidate(
    component_id: str, *, eligible: BoolArray, momentum: FloatArray | None
) -> BoolArray:
    if component_id in {"G0_IW184", "G6_R0_FAST_REBOUND"}:
        return eligible.copy()
    if momentum is None:
        raise HeterogeneousScoringError("alpha_research.heterogeneous_live_candidate_source_absent")
    live = np.flatnonzero(eligible & np.isfinite(momentum))
    output: BoolArray = np.zeros(len(eligible), dtype=np.bool_)
    if component_id == "G2_R0_TREND":
        if len(live) < 100:
            raise HeterogeneousScoringError(
                "alpha_research.heterogeneous_live_trend_candidate_support_insufficient"
            )
        ordered = live[np.lexsort((live, -momentum[live]))]
        output[live[momentum[live] >= momentum[ordered[99]]]] = True
        return output
    if component_id == "G7_R1_CONTEXTUAL_MOMENTUM":
        if len(live) < 70:
            raise HeterogeneousScoringError(
                "alpha_research.heterogeneous_live_contextual_momentum_candidate_support_insufficient"
            )
        output[live[_specialist_percentile(momentum[live]) >= 0.50]] = True
        return output
    raise HeterogeneousScoringError("alpha_research.heterogeneous_live_component_invalid")


def _heterogeneous_inputs(
    *,
    component: HeterogeneousComponentScoringRecipe,
    inputs: HeterogeneousFormationScoreInput,
    declared_vintages: tuple[str, ...] | None = None,
) -> tuple[tuple[str, ...], dict[str, HeterogeneousVintageFeatureSurface]]:
    vintages = (
        live_vintages(inputs.formation_session, count=component.vintage_count)
        if declared_vintages is None
        else declared_vintages
    )
    if len(vintages) != component.vintage_count or len(set(vintages)) != len(vintages):
        raise HeterogeneousScoringError("alpha_research.heterogeneous_live_vintage_axis_invalid")
    if tuple(value.vintage for value in inputs.feature_surfaces) != vintages:
        raise HeterogeneousScoringError(
            "alpha_research.heterogeneous_live_feature_vintage_axis_invalid"
        )
    by_vintage = {value.vintage: value for value in inputs.feature_surfaces}
    if tuple((model.vintage, model.seed) for model in inputs.models) != tuple(
        (vintage, seed) for vintage in vintages for seed in component.seeds
    ):
        raise HeterogeneousScoringError("alpha_research.heterogeneous_live_model_set_incomplete")
    expected_axis = component.ordered_feature_ids or inputs.models[0].estimator.ordered_feature_ids
    if (
        len(expected_axis) != component.feature_count
        or canonical_hash(list(expected_axis)) != component.feature_axis_hash
        or any(model.estimator.ordered_feature_ids != expected_axis for model in inputs.models)
        or any(
            surface.ordered_listing_ids != inputs.ordered_listing_ids
            or surface.ordered_feature_ids != expected_axis
            for surface in inputs.feature_surfaces
        )
    ):
        raise HeterogeneousScoringError("alpha_research.heterogeneous_live_feature_axis_mismatch")
    return vintages, by_vintage


def score_heterogeneous_component(
    *,
    component: HeterogeneousComponentScoringRecipe,
    inputs: HeterogeneousFormationScoreInput,
    prediction_owner: HeterogeneousPredictionOwner,
    declared_vintages: tuple[str, ...] | None = None,
) -> AlphaProductScoreProjection:
    """Score one complete component; missing children or axes never renormalize."""
    vintages, surfaces = _heterogeneous_inputs(
        component=component, inputs=inputs, declared_vintages=declared_vintages
    )
    eligible = np.asarray(inputs.decision_eligible, dtype=np.bool_)
    candidate = _heterogeneous_candidate(
        component.component_id, eligible=eligible, momentum=inputs.raw_12_1_momentum
    )
    models = {(value.vintage, value.seed): value for value in inputs.models}
    scores: FloatArray = np.full(len(inputs.ordered_listing_ids), np.nan, dtype=np.float64)
    aggregation_semantics: ProjectionAggregationSemantics

    def raw(vintage: str, seed: int, positions: npt.NDArray[np.intp]) -> FloatArray:
        values = prediction_owner(
            component=component,
            model=models[(vintage, seed)],
            features=np.asarray(
                _frozen(surfaces[vintage].features[positions], dtype=np.dtype("<f8")),
                dtype=np.float64,
            ),
        )
        if values.shape != (len(positions),) or not np.isfinite(values).all():
            raise HeterogeneousScoringError(
                "alpha_research.heterogeneous_live_prediction_values_invalid"
            )
        return values

    if component.score_aggregation == (
        "PER_MODEL_PERCENTILE_0_1_THEN_EQUAL_SEED_MEAN_THEN_WEIGHTED_VINTAGE_MEAN"
    ):
        live = candidate.copy()
        for surface in inputs.feature_surfaces:
            live &= np.isfinite(surface.features).all(axis=1)
        positions = np.flatnonzero(live)
        if len(positions) < 2:
            raise HeterogeneousScoringError(
                "alpha_research.heterogeneous_live_score_support_insufficient"
            )
        weighted = np.zeros(len(positions), dtype=np.float64)
        for vintage, weight in zip(vintages, component.vintage_weights, strict=True):
            weighted += float(weight) * np.mean(
                np.vstack(
                    [percentile_rank(raw(vintage, seed, positions)) for seed in component.seeds]
                ),
                axis=0,
            )
        scores[positions] = weighted / float(sum(component.vintage_weights))
        aggregation_semantics = SCORE_AGGREGATION_SEMANTICS
    elif component.score_aggregation == (
        "EQUAL_RAW_SEED_MEAN_THEN_CANDIDATE_PERCENTILE_1_N_TO_1_THEN_"
        "WEIGHTED_VINTAGE_MEAN_WITH_OUTSIDER_MINUS_ONE"
    ):
        vintage_scores = []
        for vintage in vintages:
            positions = np.flatnonzero(
                candidate & np.isfinite(surfaces[vintage].features).all(axis=1)
            )
            if len(positions) < 2:
                raise HeterogeneousScoringError(
                    "alpha_research.heterogeneous_live_score_support_insufficient"
                )
            ranked = np.full(len(scores), -1.0, dtype=np.float64)
            ranked[positions] = _specialist_percentile(
                np.mean(
                    np.vstack([raw(vintage, seed, positions) for seed in component.seeds]), axis=0
                )
            )
            vintage_scores.append(ranked[eligible])
        scores[eligible] = np.average(
            np.vstack(vintage_scores),
            axis=0,
            weights=np.asarray(component.vintage_weights, dtype=np.float64),
        )
        live = eligible
        aggregation_semantics = SPECIALIST_SCORE_AGGREGATION_SEMANTICS
    else:
        raise HeterogeneousScoringError(
            "alpha_research.heterogeneous_live_score_aggregation_invalid"
        )

    candidate_hash = canonical_hash(
        {
            "component_id": component.component_id,
            "formation_session": inputs.formation_session.isoformat(),
            "candidate_semantics": component.candidate_semantics,
            "candidate_mask": _alpha_live_array_hash(candidate),
        }
    )
    lineage = tuple(
        canonical_hash(
            {
                "model_lineage_hash": models[(vintage, seed)].lineage_hash,
                "feature_source_binding_hash": surfaces[vintage].source_binding_hash,
                "feature_values_hash": surfaces[vintage].feature_values_hash,
                "candidate_binding_hash": candidate_hash,
            }
        )
        for vintage in vintages
        for seed in component.seeds
    )
    return AlphaProductScoreProjection.create_live(
        recipe_hash=component.recipe_hash,
        evidence_manifest_sha256=inputs.model_set_manifest_hash,
        formation_session=inputs.formation_session,
        ordered_listing_ids=inputs.ordered_listing_ids,
        live_vintages=vintages,
        seeds=component.seeds,
        vintage_weights=component.vintage_weights,
        aggregation_semantics=aggregation_semantics,
        model_identity_hashes=tuple(value.estimator.content_hash for value in inputs.models),
        model_lineage_hashes=lineage,
        scores=scores,
        live=live,
    )


__all__ = [
    "EVIDENCE_PACKAGE_ID",
    "INSTALLED_EVIDENCE_MANIFEST_SHA256",
    "MINIMUM_LIVE_NAMES",
    "SCORE_AGGREGATION_SEMANTICS",
    "AdmittedProductEvidence",
    "AlphaEvidenceClosure",
    "AlphaProductRecipeView",
    "AlphaProductScoreProjection",
    "AlphaScoreReplayError",
    "HeterogeneousComponentScoringRecipe",
    "HeterogeneousEstimatorContent",
    "HeterogeneousFormationScoreInput",
    "HeterogeneousLiveModel",
    "HeterogeneousPredictionOwner",
    "HeterogeneousScoringError",
    "HeterogeneousVintageFeatureSurface",
    "LoadedSeedModel",
    "PredictionAxisReceipt",
    "SeedModelReceipt",
    "live_vintages",
    "percentile_rank",
    "quarter_start",
    "replay_formation_scores",
    "score_heterogeneous_component",
    "shift_quarter",
]
