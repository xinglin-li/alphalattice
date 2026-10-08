"""Content-addressed development Alpha evidence independent of Agent and publication."""

from __future__ import annotations

import json
import os
import pathlib
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import date
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter
from typing import Any, Self, TypeVar, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.alpha_modeling.contracts import (
    AlphaEstimatorContent,
    AlphaFitProvenanceReceipt,
    AlphaModelFitSidecar,
    alpha_model_array_content_hash,
)
from alphalattice.capabilities.alpha_modeling.runtime.numerical_environment import (
    AlphaModelNumericalEnvironment,
)
from alphalattice.capabilities.causal_inputs.contracts import (
    AnchoredInstantPolicy,
    SessionAnchor,
)
from alphalattice.control.workspace_runtime.content_store import (
    ContentAddressedStore,
    ContentAddressedStoreError,
    verified_npz_arrays,
    verified_request_value,
    verified_source_value,
)
from alphalattice.investment.alpha_research.calibration.return_unit import (
    RETURN_UNIT_CALIBRATION_CATEGORY,
    AlphaReturnUnitCalibrationEvidence,
)
from alphalattice.investment.alpha_research.scaling.contracts import (
    CrossSectionalDispersionForecast,
)
from alphalattice.investment.alpha_research.scores.temporal_aggregation import (
    FIXED_ALPHA_ROW_AXIS_LANE_CATEGORY,
    FIXED_ALPHA_ROW_AXIS_RECEIPT_CATEGORY,
    PANEL_SCORE_READINESS_POLICY,
    PANEL_SCORE_READINESS_RECEIPT_CATEGORY,
    SEALED_PANEL_ALPHA_DECISION_SESSION_COUNT,
    SEALED_PANEL_ALPHA_FEATURE_COLUMN_COUNT,
    SEALED_PANEL_ALPHA_MODEL_RECIPE_ID,
    SEALED_PANEL_ALPHA_VIEW_ID,
    AlphaPanelSourceIdentity,
    AlphaScoreAggregationEvidence,
    AlphaScoreAggregationProgram,
    AlphaScoreAggregationSurface,
    FixedAlphaFoldRowAxis,
    FixedAlphaRowAxisReceipt,
    PanelScoreFoldIdentity,
    PanelScoreFormationReadinessReceipt,
    panel_score_surface_hash,
    resolve_panel_score_readiness_policy,
    row_axis_hash,
    score_value_hash,
)
from alphalattice.investment.alpha_research.simple_signal.contracts import (
    SIMPLE_SCORE_CATEGORY,
    SIMPLE_SCORE_VALUE_ARTIFACT_CATEGORY,
    SIMPLE_SCORE_VALUE_BINDING_CATEGORY,
    SIMPLE_SCORE_VALUE_PAYLOAD_CATEGORY,
    SimpleSignedScoreBinding,
    SimpleSignedScoreValueArtifact,
)
from alphalattice.investment.alpha_research.simple_signal.standardize import (
    score_values_identity,
)
from alphalattice.investment.alpha_research.targets.authority import (
    WholeUniverseAlphaTargetMethodBinding,
)
from alphalattice.investment.alpha_research.targets.canonical import (
    CanonicalAlphaScoreBinding,
    CanonicalAlphaTargetEvidence,
    CanonicalAlphaTargetRecipeBinding,
)
from alphalattice.investment.alpha_research.targets.total_return import (
    TotalReturnAlphaTargetEvidence,
    TotalReturnAlphaTargetSurface,
    total_return_array_identity,
)
from alphalattice.kernel.shared_kernel.arrow_identity import canonical_hash_with_rows
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.spans import span, spanned

from .campaign import AlphaDevelopmentProgram
from .campaign_evidence import (
    AlphaCampaignDecisionReceipt,
    AlphaDevelopmentDecisionDossier,
    AlphaDevelopmentMethodEvidence,
    AlphaDevelopmentTrialEvidence,
    AlphaInnerSelectionRecord,
    AlphaRecursiveReplayReceipt,
)
from .campaign_execution import (
    AlphaCampaignPredictionSurface,
    alpha_campaign_prediction_artifact_hash,
)
from .contracts import AlphaDevelopmentFitEvidence, AlphaExperimentBatchResult
from .development_contracts import (
    AlphaCandidateDevelopmentReport,
    AlphaCandidateDevelopmentScoreChunkRef,
    AlphaCandidateExecutionBinding,
    AlphaCandidateFoldEvidence,
    AlphaCandidateInferenceEvidence,
    AlphaCandidateNumericalFoldResult,
    AlphaCurrentRefitDiagnosticReport,
    AlphaDecisionReproducibilityReport,
    AlphaDevelopmentEstimatorState,
    AlphaDevelopmentExecutionReceipt,
    AlphaDevelopmentFoldSurface,
    AlphaDevelopmentSurfaceBinding,
    AlphaDevelopmentSurfaceManifest,
    AlphaDevelopmentValidationChunkRef,
    AlphaEstimatorState,
    AlphaModelSelectionDecision,
    AlphaModelSelectionProposal,
    AlphaModelViabilityAssessment,
    AlphaNumericalDevelopmentScoreChunkRef,
    AlphaParentRequestBinding,
    CanonicalAlphaScoreSurface,
    LegacyAlphaCandidateFoldEvidence,
)
from .fit_plan import AlphaModelFitPlan
from .target_preprocessing import (
    AlphaTargetClippingDetail,
    AlphaTargetPreprocessingComparisonEvidence,
    AlphaTargetPreprocessingComparisonProgram,
    AlphaTargetPreprocessingReceipt,
)

CANONICAL_SCORE_SURFACE_CATEGORY = "canonical-score-surfaces"
"""Where the aggregated prediction surface lives, spelled once.

A consumer filtering evidence URIs needs the same spelling the store publishes
under; two spellings of one category is two definitions of where artifacts are.
"""

CANONICAL_SCORE_BINDING_CATEGORY = "canonical-score-bindings"


class PanelScoreSurfaceArtifact(BaseModel):  # type: ignore[misc]
    """Packed Alpha score lanes and their exact row/producer identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = "PanelScoreSurfaceArtifact"
    source_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_index: int = Field(ge=0)
    surface_id: str
    span_sessions: int = Field(ge=1)
    row_count: int = Field(ge=1)
    row_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_lane_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_z_lane_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_simple_return_lane_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_recipe_id: str | None = None
    producer_spec_id: str | None = None
    upstream_raw_score_artifact_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    filter_spec_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    filter_evidence_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    fold_selection_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    output_score_value_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, artifact_hash="0" * 64)
        return cls(
            **values,
            artifact_hash=str(
                canonical_hash(
                    provisional.model_dump(
                        mode="json", exclude={"artifact_hash"}, exclude_none=True
                    )
                )
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_lineage(self) -> Self:
        lineage = (
            self.upstream_raw_score_artifact_hash,
            self.filter_spec_hash,
            self.filter_evidence_hash,
            self.fold_selection_hash,
            self.output_score_value_hash,
        )
        filtered = self.producer_spec_id is not None
        if (
            any(value is not None for value in lineage)
            != all(value is not None for value in lineage)
            or filtered != all(value is not None for value in lineage)
            or self.artifact_hash
            != canonical_hash(
                self.model_dump(mode="json", exclude={"artifact_hash"}, exclude_none=True)
            )
        ):
            raise ValueError("alpha_research.score_surface_artifact_invalid")
        return self


ALPHA_DEVELOPMENT_EXECUTION_RECEIPT_CATEGORY = "development-execution-receipts"
"""Category of the development authority parent, in this store's namespace.

Its own category rather than a field on an existing one: the receipt is written
after the children it names, and a category that mixed the two would have to be
written twice.
"""

ALPHA_DEVELOPMENT_BATCH_RESULT_CATEGORY = "development-batch-results"
"""The executed child axis, sealed after the run and content-addressed.

``AlphaExperimentBatchResult`` was returned to its caller and written nowhere, so
a receipt naming a ``batch_result_hash`` named nothing resolvable -- and a reader
had no way to learn which children the batch actually produced. It is published
unchanged; only a path to it is added.
"""

ALPHA_SCORE_AGGREGATION_PROGRAM_CATEGORY = "development/score-aggregation/programs"
ALPHA_SCORE_AGGREGATION_EVIDENCE_CATEGORY = "development/score-aggregation/evidence"
ALPHA_SCORE_AGGREGATION_SURFACE_CATEGORY = "development/score-aggregation/surfaces"
TOTAL_RETURN_TARGET_BINDING_CATEGORY = "development/total-return-target/bindings"
TOTAL_RETURN_TARGET_EVIDENCE_CATEGORY = "development/total-return-target/evidence"
TOTAL_RETURN_TARGET_SURFACE_CATEGORY = "development/total-return-target/surfaces"


_ModelT = TypeVar("_ModelT", bound=BaseModel)
"""The artifact model a reader asked for, so it comes back as that model.

Every model here is imported, and imported names are `Any` to the type checker
under this repository's `follow_imports = "skip"`. That is why 37 readers ended
up spelling their own `cast` around a `BaseModel`-typed load. One type variable
puts the cast in one place instead.
"""


def _chunk_content_hash(table: pa.Table, identity: dict[str, object]) -> str:
    """The content identity of one development chunk: its identity, schema and rows.

    The same canonical document the chunk was first sealed with -- identity,
    the schema text and every row as a sorted-key object -- hashed from the
    Arrow columns (``canonical_hash_with_rows``, bit for bit the encoder's
    output) rather than from a Python copy of every row.
    """

    return canonical_hash_with_rows(
        {"identity": identity, "schema": str(table.schema)}, key="rows", table=table
    )


_VERIFIED_CHUNKS: dict[tuple[str, str], str] = {}
"""Each development chunk this process derived in full: its file's SHA-256 and its identity's
hash, to the content hash derived from those bytes (V461). The derivation is a function of the
bytes alone, so a chunk whose bytes hash the same was derived before; its read still hashes the
file and runs every other check. Kept in memory only: nothing on disk vouches for a file. Each
entry is two digests, one per chunk file the process read, so the workspace's chunks bound it."""

_VERIFIED_CHUNK_LOCK = Lock()


def _verified_content_hash(raw: bytes, table: pa.Table, identity: dict[str, object]) -> str:
    """The chunk's content hash, derived in full the first time this process reads its bytes."""

    key = (sha256(raw).hexdigest(), str(canonical_hash(identity)))
    with _VERIFIED_CHUNK_LOCK:
        derived = _VERIFIED_CHUNKS.get(key)
    if derived is None:
        derived = _chunk_content_hash(table, identity)
        with _VERIFIED_CHUNK_LOCK:
            _VERIFIED_CHUNKS[key] = derived
    return derived


class AlphaDevelopmentArtifactReadbackError(ValueError):
    """A development Alpha parent/child chain is missing, stale, or tampered."""


class AlphaDevelopmentArtifactStore:
    """Immutable development evidence in the established Alpha CAS namespace."""

    _PREFIX = "playpen://alpha-research/"

    _CURRENT_SCORE_COLUMNS = (
        "formation_session",
        "listing_id",
        "score",
        "availability",
    )
    _DEVELOPMENT_VALIDATION_COLUMNS = (
        "row_index",
        "formation_session",
        "listing_id",
        "target",
        "feature_available",
        "outcome_available",
        "feature_row_hash",
        "causal_outcome_row_hash",
    )
    _DEVELOPMENT_ECONOMIC_VALIDATION_COLUMNS = (
        "row_index",
        "formation_session",
        "listing_id",
        "target",
        "economic_return",
        "feature_available",
        "outcome_available",
        "feature_row_hash",
        "causal_outcome_row_hash",
    )
    _DEVELOPMENT_SCORE_COLUMNS = ("row_index", "score", "availability")

    def __init__(
        self,
        artifact_root: Path,
        *,
        packed_capacity: Callable[[int], None] = lambda _bytes: None,
        sharing_workspace: Path | None = None,
    ) -> None:
        self.root = artifact_root.resolve() / "alpha-research"
        self._packed_capacity = packed_capacity
        # A fold's listing positions and a score's values are numerical lanes: Parquet named by
        # the digest of their packed bytes; a store written before V294 reads its `.bin` files.
        self._lanes = ContentAddressedStore(
            self.root, uri_prefix=self._PREFIX, capacity=packed_capacity
        )
        self._sharing_workspace = sharing_workspace.resolve() if sharing_workspace else None
        if self._sharing_workspace is not None and not self.root.is_relative_to(
            self._sharing_workspace
        ):
            raise ValueError("alpha_research.artifact_root_outside_sharing_workspace")

    @staticmethod
    def _require_hash(value: str) -> None:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("Alpha artifact identity is not SHA-256")

    def _path(self, category: str, content_hash: str, suffix: str) -> Path:
        self._require_hash(content_hash)
        return self.root / category / f"{content_hash}.{suffix}"

    @classmethod
    def uri(cls, category: str, content_hash: str) -> str:
        return f"{cls._PREFIX}{category}/{content_hash}"

    @classmethod
    def _hash_from_uri(cls, uri: str, category: str) -> str:
        prefix = f"{cls._PREFIX}{category}/"
        if not uri.startswith(prefix):
            raise AlphaDevelopmentArtifactReadbackError("unsupported Alpha artifact URI")
        value = uri[len(prefix) :]
        cls._require_hash(value)
        return value

    @staticmethod
    def _json_bytes(payload: Mapping[str, object]) -> bytes:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")

    @staticmethod
    def _atomic_write(target: Path, content: bytes) -> None:
        with span("write", "alpha_artifact"):
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            staged.write_bytes(content)
            os.replace(staged, target)
            staged.unlink(missing_ok=True)

    def _publish_identity_json(
        self,
        *,
        category: str,
        value: BaseModel,
        identity_field: str,
    ) -> str:
        with span("serialize", "alpha_json"):
            payload = value.model_dump(mode="json")
        content_hash = str(payload[identity_field])
        self._require_hash(content_hash)
        identity = dict(payload)
        identity.pop(identity_field)
        with span("hash", "alpha_json"):
            if canonical_hash(identity) != content_hash:
                raise ValueError("Alpha JSON identity field is invalid")
        with span("serialize", "alpha_json"):
            serialized = self._json_bytes(payload)
        target = self._path(category, content_hash, "json")
        if target.exists():
            if target.read_bytes() != serialized:
                raise ValueError("Alpha content identity was reused with different JSON")
        else:
            self._atomic_write(target, serialized)
        return self.uri(category, content_hash)

    def _publish_packed_bytes(self, *, category: str, payload: bytes) -> str:
        with span("hash", "alpha_packed"):
            content_hash = sha256(payload).hexdigest()
        target = self._path(category, content_hash, "bin")
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError("Alpha packed content identity was reused")
        else:
            self._packed_capacity(len(payload))
            self._atomic_write(target, payload)
        return content_hash

    def import_packed(
        self, source: AlphaDevelopmentArtifactStore, *, category: str, content_hash: str
    ) -> str:
        """Keep an independently retained path to verified immutable input bytes.

        Sharing is opt-in and confined to one Host-bound workspace. A link's
        staged bytes are checked before publication; unsupported filesystems
        copy through the usual capacity-checked publisher. No artifact, receipt
        or reader identity changes, and no existing file is replaced.
        """
        origin = source._path("current/" + category, content_hash, "bin")
        target = self._path("current/" + category, content_hash, "bin")
        if self._sharing_workspace is not None and not all(
            path.resolve().is_relative_to(self._sharing_workspace) for path in (origin, target)
        ):
            raise self._packed_readback_error("alpha_research.shared_input_outside_workspace")
        payload = source._frozen_payload(category, content_hash)
        if target.exists() or self._sharing_workspace is None:
            return self._publish_packed_bytes(category="current/" + category, payload=payload)
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=target.parent, prefix=".input-") as temporary:
            staged = Path(temporary) / "payload"
            try:
                os.link(origin, staged)
            except OSError:
                return self._publish_packed_bytes(category="current/" + category, payload=payload)
            if sha256(staged.read_bytes()).hexdigest() != content_hash:
                raise self._packed_readback_error("alpha_research.frozen_input_content_invalid")
            try:
                # link, not replace: a concurrently present name must be proved.
                os.link(staged, target)
            except FileExistsError:
                return self._publish_packed_bytes(category="current/" + category, payload=payload)
        return content_hash

    _packed_readback_error = AlphaDevelopmentArtifactReadbackError

    def _frozen_payload(self, category: str, content_hash: str) -> bytes:
        """Read immutable model/input bytes; this grants no current-pointer authority."""
        payload, _ = self._frozen_payload_with_identity(category, content_hash)
        return payload

    def _frozen_payload_with_identity(
        self, category: str, content_hash: str
    ) -> tuple[bytes, tuple[str, ...]]:
        """Return exact verified bytes without retaining a second packed-byte cache.

        The descriptor identity distinguishes atomic replacements at one resolved path.
        Each cold or invalidated read computes the digest in full. Path metadata alone never
        permits reuse; write/admission paths and platforms without leases always read in full.
        """
        path = self._path(f"current/{category}", content_hash, "bin")
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root.parent):
            raise self._packed_readback_error("alpha_research.frozen_input_path_outside_root")

        with span("read", "alpha_packed"), path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            payload = stream.read()
            after = os.fstat(stream.fileno())
        with span("hash", "alpha_packed"):
            observed_hash = sha256(payload).hexdigest()
        if (
            observed_hash != content_hash
            or len(payload) != before.st_size
            or (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
        ):
            raise self._packed_readback_error("alpha_research.frozen_input_content_invalid")
        identity = (
            str(resolved),
            str(after.st_dev),
            str(after.st_ino),
            str(after.st_size),
            observed_hash,
            content_hash,
        )
        return payload, identity

    def _frozen_arrays_with_identity(
        self, category: str, content_hash: str
    ) -> tuple[dict[str, npt.NDArray[Any]], tuple[str, ...]]:
        """Retain only immutable decoded arrays behind their exact packed-source lease."""
        path = self._path(f"current/{category}", content_hash, "bin")

        def verify() -> tuple[tuple[tuple[str, npt.NDArray[Any]], ...], tuple[str, ...]]:
            payload, file_identity = self._frozen_payload_with_identity(category, content_hash)
            arrays = verified_npz_arrays(
                payload,
                identity=("alpha-packed-arrays", str(self.root), category, *file_identity),
            )
            return tuple(arrays.items()), file_identity

        arrays, file_identity = verified_source_value(
            ("alpha-packed-arrays", str(self.root), category, content_hash),
            (path,),
            verify,
            nbytes=lambda value: sum(array.nbytes for _, array in value[0]),
        )
        return dict(arrays), file_identity

    def _load_identity_json(
        self,
        *,
        uri: str,
        category: str,
        identity_field: str,
    ) -> dict[str, object]:
        value, _ = self._load_identity_json_with_identity(
            uri=uri, category=category, identity_field=identity_field
        )
        return value

    def _load_identity_json_with_identity(
        self,
        *,
        uri: str,
        category: str,
        identity_field: str,
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        """Read and verify one exact JSON file snapshot, returning its descriptor identity."""
        content_hash = self._hash_from_uri(uri, category)
        target = self._path(category, content_hash, "json")
        resolved = target.resolve()
        if not resolved.is_relative_to(self.root.parent):
            raise AlphaDevelopmentArtifactReadbackError("Alpha JSON artifact is outside root")

        def verify() -> tuple[bytes, tuple[str, ...]]:
            try:
                with span("read", "alpha_json"), target.open("rb") as stream:
                    before = os.fstat(stream.fileno())
                    raw = stream.read()
                    after = os.fstat(stream.fileno())
                current = target.stat()
            except FileNotFoundError as error:
                raise FileNotFoundError("Alpha JSON artifact is missing") from error
            if (
                len(raw) != before.st_size
                or (before.st_dev, before.st_ino, before.st_size)
                != (after.st_dev, after.st_ino, after.st_size)
                or (before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_mtime_ns, after.st_ctime_ns)
                or (after.st_dev, after.st_ino, after.st_size)
                != (current.st_dev, current.st_ino, current.st_size)
                or target.resolve() != resolved
            ):
                raise AlphaDevelopmentArtifactReadbackError(
                    "Alpha JSON artifact changed during read"
                )
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict) or payload.get(identity_field) != content_hash:
                raise AlphaDevelopmentArtifactReadbackError(
                    "Alpha JSON requested identity is invalid"
                )
            identity = dict(payload)
            identity.pop(identity_field)
            with span("hash", "alpha_json"):
                if canonical_hash(identity) != content_hash:
                    raise AlphaDevelopmentArtifactReadbackError(
                        "Alpha JSON payload hash is invalid"
                    )
            file_identity = (
                str(resolved),
                str(after.st_dev),
                str(after.st_ino),
                str(after.st_size),
                str(after.st_mtime_ns),
                str(after.st_ctime_ns),
                sha256(raw).hexdigest(),
                content_hash,
                identity_field,
            )
            return raw, file_identity

        raw, file_identity = verified_source_value(
            ("alpha-identity-json", str(self.root), category, content_hash, identity_field),
            (target,),
            verify,
            nbytes=lambda value: len(value[0]),
        )
        return cast(dict[str, object], json.loads(raw)), file_identity

    def _read_identity_json_with_identity(
        self,
        model: type[_ModelT],
        *,
        category: str,
        content_hash: str,
        identity_field: str,
    ) -> tuple[_ModelT, tuple[str, ...]]:
        """Reuse complete immutable Python-mode validation of exact verified bytes.

        The original descriptor, canonical content and model checks all run on
        a cold or invalidated read. Mutable models and ordinary non-opted reads
        retain their fresh parsing; the shared store's deep immutability check
        admits only fully immutable results to its existing bounded cache.
        """
        payload, file_identity = self._load_identity_json_with_identity(
            uri=self.uri(category, content_hash),
            category=category,
            identity_field=identity_field,
        )

        def verify() -> tuple[_ModelT, tuple[str, ...]]:
            return cast(_ModelT, model.model_validate(payload)), file_identity

        return verified_request_value(
            (
                "alpha-identity-model-python",
                str(self.root),
                category,
                content_hash,
                model,
                identity_field,
                *file_identity,
            ),
            verify,
            nbytes=int(file_identity[3]),
        )

    def _publish(self, category: str, value: BaseModel, identity_field: str) -> str:
        return str(
            self._publish_identity_json(
                category=f"current/{category}",
                value=value,
                identity_field=identity_field,
            )
        )

    def _read_identity_json(
        self,
        model: type[_ModelT],
        *,
        category: str,
        content_hash: str,
        identity_field: str,
    ) -> _ModelT:
        """One identity-checked JSON artifact, parsed in Python mode.

        `_load_identity_json` still proves the URI, the file, the requested
        identity and the payload hash; this adds the model and the return type.
        A reader states what it expects once, instead of building the URI,
        loading, validating and then casting the result back to the type it
        already named in its own signature.
        """

        value, _ = self._read_identity_json_with_identity(
            model,
            category=category,
            content_hash=content_hash,
            identity_field=identity_field,
        )
        return value

    def _read_identity_json_text(
        self,
        model: type[_ModelT],
        *,
        category: str,
        content_hash: str,
        identity_field: str,
    ) -> _ModelT:
        """The same artifact, parsed by the model's own JSON parser.

        Deliberately not folded into `_read_identity_json`. Python-mode and
        JSON-mode validation are two parsers with two coercion tables, and the
        models that arrive here were written against the JSON one. Merging them
        would be a silent parsing change wearing deduplication as a disguise.
        """

        return cast(
            _ModelT,
            model.model_validate_json(
                json.dumps(
                    self._load_identity_json(
                        uri=self.uri(category, content_hash),
                        category=category,
                        identity_field=identity_field,
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        )

    def _load(
        self,
        category: str,
        content_hash: str,
        identity_field: str,
        model: type[_ModelT],
    ) -> _ModelT:
        """A `current/` category artifact, under the name the subclass calls.

        The signature is unchanged -- same order, same positional arguments --
        because `AlphaCurrentArtifactStore` inherits this and calls it for its
        own categories. Only the return type is sharper: it is the model the
        caller handed in, which is what let every caller here stop casting.
        """

        return self._read_identity_json(
            model,
            category=f"current/{category}",
            content_hash=content_hash,
            identity_field=identity_field,
        )

    def publish_campaign_program(self, value: AlphaDevelopmentProgram) -> str:
        return self._publish_identity_json(
            category="development/campaign/programs",
            value=value,
            identity_field="program_hash",
        )

    def publish_total_return_target_binding(
        self, value: WholeUniverseAlphaTargetMethodBinding
    ) -> str:
        return self._publish_identity_json(
            category=TOTAL_RETURN_TARGET_BINDING_CATEGORY,
            value=value,
            identity_field="binding_hash",
        )

    def load_total_return_target_binding(
        self, content_hash: str
    ) -> WholeUniverseAlphaTargetMethodBinding:
        return self._read_identity_json(
            WholeUniverseAlphaTargetMethodBinding,
            category=TOTAL_RETURN_TARGET_BINDING_CATEGORY,
            content_hash=content_hash,
            identity_field="binding_hash",
        )

    def publish_total_return_target_evidence(self, value: TotalReturnAlphaTargetEvidence) -> str:
        return self._publish_identity_json(
            category=TOTAL_RETURN_TARGET_EVIDENCE_CATEGORY,
            value=value,
            identity_field="evidence_hash",
        )

    def load_total_return_target_evidence(
        self, content_hash: str
    ) -> TotalReturnAlphaTargetEvidence:
        return self._read_identity_json(
            TotalReturnAlphaTargetEvidence,
            category=TOTAL_RETURN_TARGET_EVIDENCE_CATEGORY,
            content_hash=content_hash,
            identity_field="evidence_hash",
        )

    def publish_total_return_target_surface(
        self,
        *,
        evidence: TotalReturnAlphaTargetEvidence,
        surface: TotalReturnAlphaTargetSurface,
    ) -> str:
        if (
            surface.lane_identity.lane_identity_hash != evidence.lane_identity_hash
            or surface.formation_sessions != evidence.formation_sessions
            or surface.ordered_listing_ids != evidence.ordered_listing_ids
        ):
            raise ValueError("TOTAL_RETURN_TARGET_SURFACE_EVIDENCE_MISMATCH")
        directory = self.root / TOTAL_RETURN_TARGET_SURFACE_CATEGORY / evidence.evidence_hash
        files = {
            "targets.parquet": surface.targets,
            "dispersion.parquet": surface.dispersion,
            "bounds.parquet": surface.bounds,
        }
        directory.mkdir(parents=True, exist_ok=True)
        for name, table in files.items():
            target = directory / name
            if not target.exists():
                staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
                pq.write_table(table, staged, compression="zstd")
                os.replace(staged, target)
                staged.unlink(missing_ok=True)
        self.load_total_return_target_surface(evidence.evidence_hash)
        return self.uri(TOTAL_RETURN_TARGET_SURFACE_CATEGORY, evidence.evidence_hash)

    def load_total_return_target_surface(
        self, evidence_hash: str
    ) -> tuple[TotalReturnAlphaTargetEvidence, pa.Table, pa.Table, pa.Table]:
        evidence = self.load_total_return_target_evidence(evidence_hash)
        directory = self.root / TOTAL_RETURN_TARGET_SURFACE_CATEGORY / evidence_hash
        targets = pq.read_table(directory / "targets.parquet")
        dispersion = pq.read_table(directory / "dispersion.parquet")
        bounds = pq.read_table(directory / "bounds.parquet")
        sessions = len(evidence.formation_sessions)
        listings = len(evidence.ordered_listing_ids)
        shape = (sessions, listings)
        identities = {
            "raw_log_execution_return_identity": total_return_array_identity(
                np.asarray(
                    targets["raw_log_execution_return"].to_numpy(),
                    dtype=np.float64,
                ).reshape(shape)
            ),
            "simple_economic_return_identity": total_return_array_identity(
                np.asarray(targets["simple_economic_return"].to_numpy(), dtype=np.float64).reshape(
                    shape
                )
            ),
            "bounded_log_execution_return_identity": total_return_array_identity(
                np.asarray(
                    targets["bounded_log_execution_return"].to_numpy(), dtype=np.float64
                ).reshape(shape)
            ),
            "universe_centered_return_identity": total_return_array_identity(
                np.asarray(
                    targets["universe_centered_return"].to_numpy(),
                    dtype=np.float64,
                ).reshape(shape)
            ),
            "fit_target_identity": total_return_array_identity(
                np.asarray(targets["fit_target"].to_numpy(), dtype=np.float64).reshape(shape)
            ),
            "cross_sectional_dispersion_identity": total_return_array_identity(
                np.asarray(
                    dispersion["cross_sectional_dispersion"].to_numpy(zero_copy_only=False),
                    dtype=np.float64,
                )
            ),
            "clipping_boundary_identity": str(
                canonical_hash(
                    [
                        (float(left), float(right))
                        for left, right in zip(
                            bounds["lower_bound"].to_pylist(),
                            bounds["upper_bound"].to_pylist(),
                            strict=True,
                        )
                    ]
                )
            ),
            "holding_end_sessions_hash": str(
                canonical_hash(
                    [value.isoformat() for value in targets["holding_end_session"].to_pylist()]
                )
            ),
        }
        if any(getattr(evidence, key) != value for key, value in identities.items()):
            raise AlphaDevelopmentArtifactReadbackError(
                "Total-return target surface identity is invalid"
            )
        return evidence, targets, dispersion, bounds

    def load_total_return_simple_return_lane(
        self, evidence_hash: str
    ) -> tuple[TotalReturnAlphaTargetEvidence, tuple[date, ...], np.ndarray]:
        """Read the Portfolio target lane without opening Feature values.

        The full target reader verifies every diagnostic lane because Alpha
        training consumes them all. Portfolio needs only the exact simple
        execution-return lane and its row/holding axes. Reading those four
        parquet columns through the same evidence owner prevents a
        PORTFOLIO_ONLY route from paying for unrelated target diagnostics,
        while the durable lane identity is still re-derived before return.
        """

        evidence = self.load_total_return_target_evidence(evidence_hash)
        directory = self.root / TOTAL_RETURN_TARGET_SURFACE_CATEGORY / evidence_hash
        targets = pq.read_table(
            directory / "targets.parquet",
            columns=[
                "formation_session",
                "holding_end_session",
                "listing_id",
                "simple_economic_return",
            ],
        )
        sessions = evidence.formation_sessions
        listings = evidence.ordered_listing_ids
        expected_rows = len(sessions) * len(listings)
        observed_sessions = tuple(targets["formation_session"].to_pylist())
        observed_listings = tuple(str(value) for value in targets["listing_id"].to_pylist())
        expected_sessions = tuple(session for session in sessions for _listing in listings)
        expected_listings = listings * len(sessions)
        if (
            targets.num_rows != expected_rows
            or observed_sessions != expected_sessions
            or observed_listings != expected_listings
        ):
            raise AlphaDevelopmentArtifactReadbackError(
                "total return simple-return row axis changed"
            )
        holding_rows = tuple(targets["holding_end_session"].to_pylist())
        holding_sessions: list[date] = []
        for position in range(len(sessions)):
            start = position * len(listings)
            values = holding_rows[start : start + len(listings)]
            if not values or len(set(values)) != 1:
                raise AlphaDevelopmentArtifactReadbackError("total return holding-end axis changed")
            holding_sessions.append(cast(date, values[0]))
        if canonical_hash([value.isoformat() for value in holding_rows]) != (
            evidence.holding_end_sessions_hash
        ):
            raise AlphaDevelopmentArtifactReadbackError("total return holding-end identity changed")
        simple = np.ascontiguousarray(
            np.asarray(
                targets["simple_economic_return"].to_numpy(zero_copy_only=False),
                dtype=np.float64,
            ).reshape(len(sessions), len(listings))
        )
        if total_return_array_identity(simple) != evidence.simple_economic_return_identity:
            raise AlphaDevelopmentArtifactReadbackError(
                "total return simple-return identity changed"
            )
        simple.setflags(write=False)
        return evidence, tuple(holding_sessions), simple

    def load_score_aggregation_program(self, content_hash: str) -> AlphaScoreAggregationProgram:
        return self._read_identity_json(
            AlphaScoreAggregationProgram,
            category=ALPHA_SCORE_AGGREGATION_PROGRAM_CATEGORY,
            content_hash=content_hash,
            identity_field="program_hash",
        )

    def load_score_aggregation_evidence(self, content_hash: str) -> AlphaScoreAggregationEvidence:
        return self._read_identity_json(
            AlphaScoreAggregationEvidence,
            category=ALPHA_SCORE_AGGREGATION_EVIDENCE_CATEGORY,
            content_hash=content_hash,
            identity_field="evidence_hash",
        )

    def load_score_aggregation_surface(
        self, content_hash: str
    ) -> tuple[AlphaScoreAggregationSurface, tuple[str, ...], np.ndarray]:
        surface = self._read_identity_json(
            AlphaScoreAggregationSurface,
            category=ALPHA_SCORE_AGGREGATION_SURFACE_CATEGORY,
            content_hash=content_hash,
            identity_field="surface_hash",
        )
        target = self._path(ALPHA_SCORE_AGGREGATION_SURFACE_CATEGORY, content_hash, "parquet")
        if not target.is_file():
            raise FileNotFoundError("Alpha score aggregation surface is missing")
        table = pq.read_table(target)
        if table.column_names != ["row_id", "aggregated_score"]:
            raise AlphaDevelopmentArtifactReadbackError(
                "Alpha score aggregation surface schema is invalid"
            )
        row_ids = tuple(str(value) for value in table["row_id"].to_pylist())
        values = np.ascontiguousarray(table["aggregated_score"].to_numpy(), dtype=np.float64)
        sessions = tuple(date.fromisoformat(value.split("|", 1)[0]) for value in row_ids)
        listings = tuple(value.split("|", 1)[1] for value in row_ids)
        if (
            len(row_ids) != surface.row_count
            or row_axis_hash(sessions, listings) != surface.row_axis_hash
            or score_value_hash(values) != surface.aggregated_score_value_hash
        ):
            raise AlphaDevelopmentArtifactReadbackError(
                "Alpha score aggregation surface identity is invalid"
            )
        values.setflags(write=False)
        return surface, row_ids, values

    def publish_campaign_trial_evidence(self, value: AlphaDevelopmentTrialEvidence) -> str:
        return self._publish_identity_json(
            category="development/campaign/trial-evidence",
            value=value,
            identity_field="evidence_hash",
        )

    def publish_campaign_prediction_surface(
        self,
        *,
        evidence: AlphaDevelopmentTrialEvidence,
        surface: AlphaCampaignPredictionSurface,
    ) -> str:
        if surface.trial_evidence_hash != evidence.evidence_hash:
            raise ValueError("ALPHA_CAMPAIGN_PREDICTION_EVIDENCE_MISMATCH")
        values = np.asarray(surface.predictions, dtype=np.float64)
        value_hash = alpha_model_array_content_hash(values)
        row_axis_hash = str(canonical_hash(surface.validation_row_ids))
        artifact_hash = alpha_campaign_prediction_artifact_hash(
            program_hash=evidence.program_hash,
            horizon_sessions=evidence.horizon_sessions,
            trial_hash=evidence.trial_hash,
            fold_index=evidence.fold_index,
            prediction_value_hash=value_hash,
            validation_row_axis_hash=row_axis_hash,
        )
        if (
            value_hash != evidence.prediction_value_hash
            or row_axis_hash != evidence.validation_row_axis_hash
            or artifact_hash != evidence.prediction_artifact_hash
            or artifact_hash != surface.artifact_hash
        ):
            raise ValueError("ALPHA_CAMPAIGN_PREDICTION_IDENTITY_MISMATCH")
        target = self._path("development/campaign/prediction-surfaces", artifact_hash, "parquet")
        table = pa.table(
            {
                "row_id": pa.array(surface.validation_row_ids, type=pa.string()),
                "prediction": pa.array(values, type=pa.float64()),
            }
        )
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            pq.write_table(table, staged, compression="zstd")
            os.replace(staged, target)
            staged.unlink(missing_ok=True)
        self.load_campaign_prediction_surface(evidence=evidence)
        return self.uri("development/campaign/prediction-surfaces", artifact_hash)

    def load_campaign_prediction_surface(
        self, *, evidence: AlphaDevelopmentTrialEvidence
    ) -> tuple[tuple[str, ...], np.ndarray]:
        target = self._path(
            "development/campaign/prediction-surfaces",
            evidence.prediction_artifact_hash,
            "parquet",
        )
        if not target.is_file():
            raise FileNotFoundError("Alpha campaign prediction surface is missing")
        table = pq.read_table(target)
        if table.column_names != ["row_id", "prediction"]:
            raise AlphaDevelopmentArtifactReadbackError(
                "Alpha campaign prediction surface schema is invalid"
            )
        row_ids = tuple(str(value) for value in table["row_id"].to_pylist())
        values = np.asarray(table["prediction"].to_numpy(), dtype=np.float64)
        row_axis_hash = str(canonical_hash(row_ids))
        value_hash = alpha_model_array_content_hash(values)
        artifact_hash = alpha_campaign_prediction_artifact_hash(
            program_hash=evidence.program_hash,
            horizon_sessions=evidence.horizon_sessions,
            trial_hash=evidence.trial_hash,
            fold_index=evidence.fold_index,
            prediction_value_hash=value_hash,
            validation_row_axis_hash=row_axis_hash,
        )
        if (
            row_axis_hash != evidence.validation_row_axis_hash
            or value_hash != evidence.prediction_value_hash
            or artifact_hash != evidence.prediction_artifact_hash
        ):
            raise AlphaDevelopmentArtifactReadbackError(
                "Alpha campaign prediction surface identity is invalid"
            )
        values.setflags(write=False)
        return row_ids, values

    def publish_campaign_method_evidence(self, value: AlphaDevelopmentMethodEvidence) -> str:
        return self._publish_identity_json(
            category="development/campaign/method-evidence",
            value=value,
            identity_field="evidence_hash",
        )

    def publish_campaign_inner_selection_record(self, value: AlphaInnerSelectionRecord) -> str:
        return self._publish_identity_json(
            category="development/campaign/inner-selection-records",
            value=value,
            identity_field="record_hash",
        )

    def load_campaign_inner_selection_record(self, content_hash: str) -> AlphaInnerSelectionRecord:
        return self._read_identity_json_text(
            AlphaInnerSelectionRecord,
            category="development/campaign/inner-selection-records",
            content_hash=content_hash,
            identity_field="record_hash",
        )

    def publish_campaign_dossier(self, value: AlphaDevelopmentDecisionDossier) -> str:
        self.publish_campaign_program(value.program)
        for record in value.inner_selection_records:
            self.publish_campaign_inner_selection_record(record)
        for child in value.trial_evidence:
            self.publish_campaign_trial_evidence(child)
        for child in value.method_evidence:
            self.publish_campaign_method_evidence(child)
        return self._publish_identity_json(
            category="development/campaign/dossiers",
            value=value,
            identity_field="dossier_hash",
        )

    def load_campaign_dossier(self, content_hash: str) -> AlphaDevelopmentDecisionDossier:
        dossier = self._read_identity_json_text(
            AlphaDevelopmentDecisionDossier,
            category="development/campaign/dossiers",
            content_hash=content_hash,
            identity_field="dossier_hash",
        )
        self._load_identity_json(
            uri=self.uri("development/campaign/programs", dossier.program.program_hash),
            category="development/campaign/programs",
            identity_field="program_hash",
        )
        for record in dossier.inner_selection_records:
            self._load_identity_json(
                uri=self.uri("development/campaign/inner-selection-records", record.record_hash),
                category="development/campaign/inner-selection-records",
                identity_field="record_hash",
            )
        for child in dossier.trial_evidence:
            self._load_identity_json(
                uri=self.uri("development/campaign/trial-evidence", child.evidence_hash),
                category="development/campaign/trial-evidence",
                identity_field="evidence_hash",
            )
            self.load_campaign_prediction_surface(evidence=child)
        for child in dossier.method_evidence:
            self._load_identity_json(
                uri=self.uri("development/campaign/method-evidence", child.evidence_hash),
                category="development/campaign/method-evidence",
                identity_field="evidence_hash",
            )
        for forecast_hash in dossier.scale_forecast_hashes:
            self.load_canonical_dispersion_forecast(forecast_hash)
        return dossier

    def publish_simple_signed_score(self, value: SimpleSignedScoreBinding) -> str:
        """Publish one simple signed score binding into this same store.

        A method here rather than a second store. The score is Alpha-owned
        evidence with the same content-addressed, write-once, atomic discipline
        as everything else in this file, and a parallel store would be a second
        artifact runtime for one document family.
        """

        return self._publish_identity_json(
            category=SIMPLE_SCORE_CATEGORY,
            value=value,
            identity_field="binding_hash",
        )

    def load_simple_signed_score(self, content_hash: str) -> SimpleSignedScoreBinding:
        return self._read_identity_json(
            SimpleSignedScoreBinding,
            category=SIMPLE_SCORE_CATEGORY,
            content_hash=content_hash,
            identity_field="binding_hash",
        )

    def load_fixed_alpha_row_axis_receipt(self, content_hash: str) -> FixedAlphaRowAxisReceipt:
        """Read one exact Alpha-owned historical row-axis successor."""

        return self._read_identity_json(
            FixedAlphaRowAxisReceipt,
            category=FIXED_ALPHA_ROW_AXIS_RECEIPT_CATEGORY,
            content_hash=content_hash,
            identity_field="receipt_hash",
        )

    def load_fixed_alpha_row_axis_lane(self, content_hash: str) -> bytes:
        """Read one exact packed listing-position lane by content hash."""

        try:
            return self._lanes.load_packed_bytes(
                category=FIXED_ALPHA_ROW_AXIS_LANE_CATEGORY, content_hash=content_hash
            )
        except ContentAddressedStoreError as error:
            if str(error).partition(":")[0] == "content_store.artifact_missing":
                raise
            raise AlphaDevelopmentArtifactReadbackError(
                "alpha_research.fixed_alpha_row_axis_lane_hash_invalid"
            ) from error

    def publish_panel_score_authority_group(
        self,
        *,
        row_axis_receipt: FixedAlphaRowAxisReceipt,
        row_axis_lane_payloads: tuple[bytes, ...],
        simple_score_binding: SimpleSignedScoreBinding,
        simple_score_values: npt.NDArray[np.float64],
        readiness_receipt: PanelScoreFormationReadinessReceipt,
    ) -> tuple[str, str, str, str]:
        """Publish one internally consistent Alpha successor authority group."""

        formation_sessions = tuple(
            session for fold in row_axis_receipt.folds for session in fold.validation_sessions
        )
        if (
            len(row_axis_lane_payloads) != len(row_axis_receipt.folds)
            or simple_score_binding.ordered_formation_sessions != tuple(sorted(formation_sessions))
            or simple_score_binding.ordered_listing_ids != row_axis_receipt.ordered_listing_ids
            or readiness_receipt.program_hash != row_axis_receipt.program_hash
            or readiness_receipt.formation_count != len(formation_sessions)
            or readiness_receipt.formation_axis_hash
            != canonical_hash([value.isoformat() for value in sorted(formation_sessions)])
        ):
            raise ValueError("alpha_research.panel_score_authority_group_invalid")
        for fold, payload in zip(row_axis_receipt.folds, row_axis_lane_payloads, strict=True):
            if (
                len(payload) != fold.row_count * np.dtype(fold.listing_position_dtype).itemsize
                or sha256(payload).hexdigest() != fold.listing_position_lane_hash
                or self._lanes.publish_columns(
                    category=FIXED_ALPHA_ROW_AXIS_LANE_CATEGORY,
                    columns={
                        "listing_position": np.frombuffer(
                            payload, dtype=np.dtype(fold.listing_position_dtype)
                        )
                    },
                )
                != fold.listing_position_lane_hash
            ):
                raise ValueError("alpha_research.fixed_alpha_row_axis_lane_invalid")
        self._publish_identity_json(
            category=FIXED_ALPHA_ROW_AXIS_RECEIPT_CATEGORY,
            value=row_axis_receipt,
            identity_field="receipt_hash",
        )

        values = np.ascontiguousarray(simple_score_values, dtype="<f8")
        expected_shape = (
            len(simple_score_binding.ordered_formation_sessions),
            len(simple_score_binding.ordered_listing_ids),
        )
        if (
            values.shape != expected_shape
            or not np.isfinite(values).all()
            or score_values_identity(values) != simple_score_binding.values_identity
        ):
            raise ValueError("Alpha simple-score values do not match their binding")
        payload_hash = self._lanes.publish_columns(
            category=SIMPLE_SCORE_VALUE_PAYLOAD_CATEGORY, columns={"score": values}
        )
        self._publish_identity_json(
            category=SIMPLE_SCORE_VALUE_BINDING_CATEGORY,
            value=simple_score_binding,
            identity_field="binding_hash",
        )
        value_artifact = SimpleSignedScoreValueArtifact.create(
            binding_hash=simple_score_binding.binding_hash,
            values_identity=simple_score_binding.values_identity,
            payload_hash=payload_hash,
            dtype="<f8",
            shape=expected_shape,
        )
        self._publish_identity_json(
            category=SIMPLE_SCORE_VALUE_ARTIFACT_CATEGORY,
            value=value_artifact,
            identity_field="artifact_hash",
        )
        self._publish_identity_json(
            category=PANEL_SCORE_READINESS_RECEIPT_CATEGORY,
            value=readiness_receipt,
            identity_field="receipt_hash",
        )
        handles = (
            row_axis_receipt.receipt_hash,
            simple_score_binding.binding_hash,
            value_artifact.artifact_hash,
            readiness_receipt.receipt_hash,
        )
        loaded_binding, _loaded_artifact, loaded_values = self.load_simple_signed_score_values(
            artifact_hash=handles[2], binding_hash=handles[1]
        )
        if (
            self.load_fixed_alpha_row_axis_receipt(handles[0]) != row_axis_receipt
            or self.load_panel_score_formation_readiness(handles[3]) != readiness_receipt
            or loaded_binding != simple_score_binding
            or not np.array_equal(loaded_values, values)
        ):
            raise ValueError("alpha_research.panel_score_successor_readback_failed")
        return handles

    def publish_panel_score_successor_authority(
        self,
        *,
        program_hash: str,
        source: AlphaPanelSourceIdentity,
        feature_column_count: int,
        folds: tuple[
            tuple[
                int,
                tuple[date, ...],
                tuple[str, ...],
                PanelScoreSurfaceArtifact,
                PanelScoreSurfaceArtifact,
            ],
            ...,
        ],
        historical_artifacts: tuple[PanelScoreSurfaceArtifact, ...],
        simple_score_materializer: Callable[
            [tuple[date, ...], tuple[str, ...]],
            tuple[SimpleSignedScoreBinding, npt.NDArray[np.float64]],
        ],
    ) -> tuple[str, str, str, str, str, float]:
        """Prove parity and publish all Alpha children before the caller's root marker."""

        if len(folds) != 5 or feature_column_count != SEALED_PANEL_ALPHA_FEATURE_COLUMN_COUNT:
            raise ValueError("alpha_research.fixed_alpha_successor_parity_failed")
        historical = {
            value.fold_index: value
            for value in historical_artifacts
            if value.model_recipe_id == SEALED_PANEL_ALPHA_MODEL_RECIPE_ID
            and value.producer_spec_id is None
        }
        identities = tuple(
            PanelScoreFoldIdentity.create(
                fold_index=fold_index,
                artifact_hash=fixed.artifact_hash,
                source_surface_hash=fixed.source_surface_hash,
                validation_session_count=len(tuple(dict.fromkeys(row_sessions))),
                validation_session_axis_hash=str(
                    canonical_hash([value.isoformat() for value in dict.fromkeys(row_sessions)])
                ),
                row_count=fixed.row_count,
                row_axis_hash=fixed.row_axis_hash,
                score_value_hash=fixed.score_lane_hash,
            )
            for fold_index, row_sessions, _row_listings, fixed, _control in folds
        )
        if any(
            historical.get(identity.fold_index) is None
            or historical[identity.fold_index].row_count != identity.row_count
            or historical[identity.fold_index].row_axis_hash != identity.row_axis_hash
            or historical[identity.fold_index].score_lane_hash != identity.score_value_hash
            for identity in identities
        ):
            raise ValueError("alpha_research.fixed_alpha_successor_parity_failed")
        formation_sessions = tuple(
            sorted(
                {session for _index, rows, _listings, _fixed, _control in folds for session in rows}
            )
        )
        ordered_listings = tuple(
            sorted(
                {
                    listing
                    for _index, _rows, listings, _fixed, _control in folds
                    for listing in listings
                }
            )
        )
        if (
            len(formation_sessions) != SEALED_PANEL_ALPHA_DECISION_SESSION_COUNT
            or len(ordered_listings) > np.iinfo(np.uint16).max
        ):
            raise ValueError("alpha_research.fixed_alpha_successor_axis_invalid")
        listing_at = {value: index for index, value in enumerate(ordered_listings)}
        row_folds: list[FixedAlphaFoldRowAxis] = []
        lane_payloads: list[bytes] = []
        for (fold_index, row_sessions, row_listings, fixed, control), identity in zip(
            folds, identities, strict=True
        ):
            sessions = tuple(dict.fromkeys(row_sessions))
            counts = Counter(row_sessions)
            if (
                tuple(session for session in sessions for _ in range(counts[session]))
                != row_sessions
                or fixed.row_axis_hash != control.row_axis_hash
            ):
                raise ValueError("alpha_research.fixed_alpha_successor_axis_invalid")
            payload = np.asarray(
                [listing_at[value] for value in row_listings], dtype="<u2"
            ).tobytes(order="C")
            lane_payloads.append(payload)
            row_folds.append(
                FixedAlphaFoldRowAxis(
                    fold_index=fold_index,
                    validation_sessions=sessions,
                    row_count_by_session=tuple(counts[value] for value in sessions),
                    listing_position_lane_hash=sha256(payload).hexdigest(),
                    row_count=identity.row_count,
                    row_axis_hash=identity.row_axis_hash,
                    fixed_score_artifact_hash=fixed.artifact_hash,
                    fixed_score_value_hash=fixed.score_lane_hash,
                    control_score_artifact_hash=control.artifact_hash,
                    control_score_value_hash=control.score_lane_hash,
                )
            )
        row_axis_receipt = FixedAlphaRowAxisReceipt.create(
            program_hash=program_hash,
            source_resolution_hash=source.source_resolution_hash,
            panel_snapshot_hash=source.panel_snapshot_hash,
            fixed_model_recipe_id=SEALED_PANEL_ALPHA_MODEL_RECIPE_ID,
            fixed_feature_view_id=SEALED_PANEL_ALPHA_VIEW_ID,
            ordered_listing_ids=ordered_listings,
            listing_axis_hash=str(canonical_hash(list(ordered_listings))),
            folds=tuple(row_folds),
        )
        simple_started = perf_counter()
        simple_score_binding, simple_score_values = simple_score_materializer(
            formation_sessions, ordered_listings
        )
        simple_seconds = perf_counter() - simple_started
        score_surface = panel_score_surface_hash(identities)
        readiness_receipt = PanelScoreFormationReadinessReceipt.create(
            program_hash=program_hash,
            source_identity_hash=source.identity_hash,
            score_surface_hash=score_surface,
            formation_count=len(formation_sessions),
            formation_axis_hash=str(
                canonical_hash([value.isoformat() for value in formation_sessions])
            ),
            observed_through=AnchoredInstantPolicy.exchange_event(
                offset_sessions=0,
                event="OFFICIAL_CLOSE",
                policy_id="SCORE_OBSERVATION_SESSION_CLOSE",
            ),
            source_available=AnchoredInstantPolicy.create(
                policy_id=source.governing_availability_policy_id,
                basis="INSTALLED_SOURCE_AVAILABILITY_POLICY",
                anchor=SessionAnchor(offset_sessions=0, event="OFFICIAL_CLOSE"),
                minutes_after_anchor=0,
                rationale=(
                    "installed source availability policy "
                    f"{source.governing_availability_policy_id}, 0 sessions after the "
                    "observation session close"
                ),
            ),
            derived_ready=resolve_panel_score_readiness_policy(PANEL_SCORE_READINESS_POLICY),
        )
        handles = self.publish_panel_score_authority_group(
            row_axis_receipt=row_axis_receipt,
            row_axis_lane_payloads=tuple(lane_payloads),
            simple_score_binding=simple_score_binding,
            simple_score_values=simple_score_values,
            readiness_receipt=readiness_receipt,
        )
        return (score_surface, *handles, simple_seconds)

    def load_panel_score_formation_readiness(
        self,
        content_hash: str,
        *,
        program_hash: str | None = None,
        score_surface_hash: str | None = None,
        formation_sessions: tuple[date, ...] | None = None,
    ) -> PanelScoreFormationReadinessReceipt:
        receipt = self._read_identity_json(
            PanelScoreFormationReadinessReceipt,
            category=PANEL_SCORE_READINESS_RECEIPT_CATEGORY,
            content_hash=content_hash,
            identity_field="receipt_hash",
        )
        expected = (program_hash, score_surface_hash, formation_sessions)
        if any(value is None for value in expected) != all(value is None for value in expected):
            raise AlphaDevelopmentArtifactReadbackError(
                "alpha_research.panel_score_readiness_expectation_incomplete"
            )
        if program_hash is not None and (
            receipt.program_hash != program_hash
            or receipt.score_surface_hash != score_surface_hash
            or receipt.formation_count != len(cast(tuple[date, ...], formation_sessions))
            or receipt.formation_axis_hash
            != canonical_hash(
                [value.isoformat() for value in cast(tuple[date, ...], formation_sessions)]
            )
        ):
            raise AlphaDevelopmentArtifactReadbackError(
                "alpha_research.panel_score_formation_readiness_unverified"
            )
        return receipt

    def load_simple_signed_score_values(
        self,
        *,
        artifact_hash: str,
        binding_hash: str,
    ) -> tuple[
        SimpleSignedScoreBinding,
        SimpleSignedScoreValueArtifact,
        npt.NDArray[np.float64],
    ]:
        """Load one exact value artifact without opening its source Panel."""

        try:
            artifact = self._read_identity_json(
                SimpleSignedScoreValueArtifact,
                category=SIMPLE_SCORE_VALUE_ARTIFACT_CATEGORY,
                content_hash=artifact_hash,
                identity_field="artifact_hash",
            )
            binding = self._read_identity_json(
                SimpleSignedScoreBinding,
                category=SIMPLE_SCORE_VALUE_BINDING_CATEGORY,
                content_hash=binding_hash,
                identity_field="binding_hash",
            )
            if artifact.binding_hash != binding_hash:
                raise ValueError("binding identity mismatch")
            expected_shape = (
                len(binding.ordered_formation_sessions),
                len(binding.ordered_listing_ids),
            )
            if (
                artifact.shape != expected_shape
                or artifact.values_identity != binding.values_identity
            ):
                raise ValueError("value artifact axis or binding mismatch")
            payload = self._lanes.load_packed_bytes(
                category=SIMPLE_SCORE_VALUE_PAYLOAD_CATEGORY, content_hash=artifact.payload_hash
            )
            dtype = np.dtype(artifact.dtype)
            if (
                sha256(payload).hexdigest() != artifact.payload_hash
                or len(payload) != int(np.prod(artifact.shape, dtype=np.int64)) * dtype.itemsize
            ):
                raise ValueError("value artifact payload mismatch")
            values: npt.NDArray[np.float64] = np.frombuffer(payload, dtype=dtype).reshape(
                artifact.shape
            )
            if (
                not np.isfinite(values).all()
                or score_values_identity(values) != artifact.values_identity
            ):
                raise ValueError("value artifact numerical identity mismatch")
            values.setflags(write=False)
            return binding, artifact, values
        except Exception as error:
            if isinstance(error, AlphaDevelopmentArtifactReadbackError):
                raise
            raise AlphaDevelopmentArtifactReadbackError(
                "alpha_research.simple_signal_value_artifact_readback_failed"
            ) from error

    def load_simple_signed_score_metadata(
        self,
        *,
        artifact_hash: str,
        binding_hash: str,
    ) -> tuple[SimpleSignedScoreBinding, SimpleSignedScoreValueArtifact, int]:
        """Reopen one exact value handle without constructing its score matrix."""

        try:
            artifact = self._read_identity_json(
                SimpleSignedScoreValueArtifact,
                category=SIMPLE_SCORE_VALUE_ARTIFACT_CATEGORY,
                content_hash=artifact_hash,
                identity_field="artifact_hash",
            )
            binding = self._read_identity_json(
                SimpleSignedScoreBinding,
                category=SIMPLE_SCORE_VALUE_BINDING_CATEGORY,
                content_hash=binding_hash,
                identity_field="binding_hash",
            )
            payload = self._lanes.load_packed_bytes(
                category=SIMPLE_SCORE_VALUE_PAYLOAD_CATEGORY, content_hash=artifact.payload_hash
            )
            expected_shape = (
                len(binding.ordered_formation_sessions),
                len(binding.ordered_listing_ids),
            )
            expected_bytes = (
                int(np.prod(artifact.shape, dtype=np.int64)) * np.dtype(artifact.dtype).itemsize
            )
            if (
                artifact.binding_hash != binding_hash
                or artifact.shape != expected_shape
                or artifact.values_identity != binding.values_identity
                or len(payload) != expected_bytes
                or sha256(payload).hexdigest() != artifact.payload_hash
            ):
                raise ValueError("value artifact metadata mismatch")
            return binding, artifact, len(payload)
        except Exception as error:
            if isinstance(error, AlphaDevelopmentArtifactReadbackError):
                raise
            raise AlphaDevelopmentArtifactReadbackError(
                "alpha_research.simple_signal_value_artifact_readback_failed"
            ) from error

    def publish_campaign_decision(self, value: AlphaCampaignDecisionReceipt) -> str:
        return self._publish_identity_json(
            category="development/campaign/decisions",
            value=value,
            identity_field="receipt_hash",
        )

    def load_campaign_decision(self, content_hash: str) -> AlphaCampaignDecisionReceipt:
        return self._read_identity_json_text(
            AlphaCampaignDecisionReceipt,
            category="development/campaign/decisions",
            content_hash=content_hash,
            identity_field="receipt_hash",
        )

    def publish_campaign_replay(self, value: AlphaRecursiveReplayReceipt) -> str:
        return self._publish_identity_json(
            category="development/campaign/replays",
            value=value,
            identity_field="replay_hash",
        )

    def load_campaign_replay(self, content_hash: str) -> AlphaRecursiveReplayReceipt:
        return self._read_identity_json_text(
            AlphaRecursiveReplayReceipt,
            category="development/campaign/replays",
            content_hash=content_hash,
            identity_field="replay_hash",
        )

    def publish_model_fit_sidecar(
        self,
        *,
        operation_binding_hash: str,
        estimator_content: AlphaEstimatorContent,
        fit_provenance: AlphaFitProvenanceReceipt,
    ) -> AlphaModelFitSidecar:
        """Publish additive model evidence without changing frozen state identities."""

        if fit_provenance.estimator_content_hash != estimator_content.content_hash:
            raise ValueError("ALPHA_MODEL_FIT_SIDECAR_CONTENT_MISMATCH")
        self._publish("model-estimator-content", estimator_content, "content_hash")
        self._publish("model-fit-provenance", fit_provenance, "provenance_hash")
        sidecar = AlphaModelFitSidecar.create(
            operation_binding_hash=operation_binding_hash,
            estimator_content_hash=estimator_content.content_hash,
            fit_provenance_hash=fit_provenance.provenance_hash,
        )
        self._publish("model-fit-sidecars", sidecar, "sidecar_hash")
        pointer = (
            self.root
            / "current"
            / "model-fit-sidecar-by-operation"
            / f"{operation_binding_hash}.json"
        )
        payload = sidecar.model_dump(mode="json")
        if pointer.is_file():
            existing = json.loads(pointer.read_text(encoding="utf-8"))
            if existing != payload:
                raise AlphaDevelopmentArtifactReadbackError(
                    "model fit operation has divergent sidecar evidence"
                )
        else:
            self._atomic_write(pointer, self._json_bytes(payload))
        return sidecar

    @spanned("verify", "model_fit_sidecar")
    def load_model_fit_sidecar(
        self,
        operation_binding_hash: str,
    ) -> tuple[AlphaModelFitSidecar, AlphaEstimatorContent, AlphaFitProvenanceReceipt]:
        pointer = (
            self.root
            / "current"
            / "model-fit-sidecar-by-operation"
            / f"{operation_binding_hash}.json"
        )
        if not pointer.is_file():
            raise AlphaDevelopmentArtifactReadbackError("model fit sidecar is unavailable")
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        sidecar = AlphaModelFitSidecar.model_validate(payload)
        if sidecar.operation_binding_hash != operation_binding_hash:
            raise AlphaDevelopmentArtifactReadbackError("model fit sidecar operation changed")
        estimator = self._load(
            "model-estimator-content",
            sidecar.estimator_content_hash,
            "content_hash",
            AlphaEstimatorContent,
        )
        provenance = self._load(
            "model-fit-provenance",
            sidecar.fit_provenance_hash,
            "provenance_hash",
            AlphaFitProvenanceReceipt,
        )
        if provenance.estimator_content_hash != estimator.content_hash:
            raise AlphaDevelopmentArtifactReadbackError("model fit sidecar content changed")
        return sidecar, estimator, provenance

    def publish_surface(self, value: AlphaDevelopmentSurfaceManifest) -> str:
        return self._publish("development-surfaces", value, "surface_hash")

    def publish_surface_binding(self, value: AlphaDevelopmentSurfaceBinding) -> str:
        return self._publish("development-surface-bindings", value, "binding_hash")

    def load_surface_binding(self, value: str) -> AlphaDevelopmentSurfaceBinding:
        return self._load(
            "development-surface-bindings",
            value,
            "binding_hash",
            AlphaDevelopmentSurfaceBinding,
        )

    def publish_candidate_execution_binding(self, value: AlphaCandidateExecutionBinding) -> str:
        return self._publish("candidate-execution-bindings", value, "execution_binding_hash")

    def publish_parent_request_binding(self, value: AlphaParentRequestBinding) -> str:
        return self._publish("parent-request-bindings", value, "binding_hash")

    def publish_current_refit_diagnostic(self, value: AlphaCurrentRefitDiagnosticReport) -> str:
        return self._publish("current-refit-diagnostics", value, "report_hash")

    def load_current_refit_diagnostic(self, value: str) -> AlphaCurrentRefitDiagnosticReport:
        return self._load(
            "current-refit-diagnostics",
            value,
            "report_hash",
            AlphaCurrentRefitDiagnosticReport,
        )

    def current_refit_diagnostics(self) -> tuple[AlphaCurrentRefitDiagnosticReport, ...]:
        """Read the small content-addressed diagnostic inventory in canonical order."""

        root = self.root / "current" / "current-refit-diagnostics"
        if not root.is_dir():
            return ()
        return tuple(
            self.load_current_refit_diagnostic(path.stem)
            for path in sorted(root.glob("*.json"), key=lambda value: value.name)
        )

    def load_surface(self, value: str) -> AlphaDevelopmentSurfaceManifest:
        return self._load(
            "development-surfaces",
            value,
            "surface_hash",
            AlphaDevelopmentSurfaceManifest,
        )

    def publish_validation_chunk(
        self,
        table: pa.Table,
        *,
        request_hash: str,
        development_surface_hash: str,
        fold_index: int,
        fold_commitment_hash: str,
    ) -> AlphaDevelopmentValidationChunkRef:
        identity = {
            "request_hash": request_hash,
            "development_surface_hash": development_surface_hash,
            "fold_index": fold_index,
            "fold_commitment_hash": fold_commitment_hash,
        }
        columns = tuple(table.schema.names)
        if columns not in (
            self._DEVELOPMENT_VALIDATION_COLUMNS,
            self._DEVELOPMENT_ECONOMIC_VALIDATION_COLUMNS,
        ):
            raise ValueError("AlphaDevelopmentValidationChunk schema is invalid")
        return cast(
            AlphaDevelopmentValidationChunkRef,
            self._publish_development_chunk(
                table=table,
                columns=columns,
                category="development-validation-chunks",
                kind="AlphaDevelopmentValidationChunk",
                identity=identity,
                model=AlphaDevelopmentValidationChunkRef,
            ),
        )

    def resolve_validation_chunk(self, value: AlphaDevelopmentValidationChunkRef) -> pa.Table:
        target = self._path("current/development-validation-chunks", value.content_hash, "parquet")
        columns = tuple(pq.ParquetFile(target).schema_arrow.names)
        if columns not in (
            self._DEVELOPMENT_VALIDATION_COLUMNS,
            self._DEVELOPMENT_ECONOMIC_VALIDATION_COLUMNS,
        ):
            raise AlphaDevelopmentArtifactReadbackError(
                "AlphaDevelopmentValidationChunk schema changed"
            )
        return self._resolve_development_chunk(
            reference=value,
            columns=columns,
            category="development-validation-chunks",
            kind="AlphaDevelopmentValidationChunk",
            identity={
                "request_hash": value.request_hash,
                "development_surface_hash": value.development_surface_hash,
                "fold_index": value.fold_index,
                "fold_commitment_hash": value.fold_commitment_hash,
            },
        )

    def publish_fold_surface(self, value: AlphaDevelopmentFoldSurface) -> str:
        return self._publish("development-fold-surfaces", value, "surface_fold_hash")

    def load_fold_surface(self, value: str) -> AlphaDevelopmentFoldSurface:
        return self.read_fold_surface(value)[0]

    def read_fold_surface(self, value: str) -> tuple[AlphaDevelopmentFoldSurface, pa.Table]:
        """The fold surface with its validation chunk, resolved once for both.

        Loading the surface proves its chunk; a caller that goes on to read the
        chunk's rows takes them from this same resolution rather than proving
        the chunk a second time.
        """

        result = self._load(
            "development-fold-surfaces",
            value,
            "surface_fold_hash",
            AlphaDevelopmentFoldSurface,
        )
        return result, self.resolve_validation_chunk(result.validation_chunk)

    def publish_candidate_score_chunk(
        self,
        table: pa.Table,
        *,
        request_hash: str,
        development_surface_hash: str,
        candidate_id: str,
        candidate_card_hash: str,
        fold_index: int,
        fold_commitment_hash: str,
    ) -> AlphaCandidateDevelopmentScoreChunkRef:
        identity = {
            "request_hash": request_hash,
            "development_surface_hash": development_surface_hash,
            "candidate_id": candidate_id,
            "candidate_card_hash": candidate_card_hash,
            "fold_index": fold_index,
            "fold_commitment_hash": fold_commitment_hash,
        }
        return cast(
            AlphaCandidateDevelopmentScoreChunkRef,
            self._publish_development_chunk(
                table=table,
                columns=self._DEVELOPMENT_SCORE_COLUMNS,
                category="development-score-chunks",
                kind="AlphaCandidateDevelopmentScoreChunk",
                identity=identity,
                model=AlphaCandidateDevelopmentScoreChunkRef,
            ),
        )

    def resolve_candidate_score_chunk(
        self,
        value: AlphaCandidateDevelopmentScoreChunkRef | AlphaNumericalDevelopmentScoreChunkRef,
    ) -> pa.Table:
        if isinstance(value, AlphaNumericalDevelopmentScoreChunkRef):
            return self.resolve_numerical_score_chunk(value)
        return self._resolve_development_chunk(
            reference=value,
            columns=self._DEVELOPMENT_SCORE_COLUMNS,
            category="development-score-chunks",
            kind="AlphaCandidateDevelopmentScoreChunk",
            identity={
                "request_hash": value.request_hash,
                "development_surface_hash": value.development_surface_hash,
                "candidate_id": value.candidate_id,
                "candidate_card_hash": value.candidate_card_hash,
                "fold_index": value.fold_index,
                "fold_commitment_hash": value.fold_commitment_hash,
            },
        )

    def publish_numerical_score_chunk(
        self,
        table: pa.Table,
        *,
        execution_binding_hash: str,
        development_surface_binding_hash: str,
        candidate_id: str,
        candidate_card_hash: str,
        fold_index: int,
        fold_commitment_hash: str,
    ) -> AlphaNumericalDevelopmentScoreChunkRef:
        identity = {
            "execution_binding_hash": execution_binding_hash,
            "development_surface_binding_hash": development_surface_binding_hash,
            "candidate_id": candidate_id,
            "candidate_card_hash": candidate_card_hash,
            "fold_index": fold_index,
            "fold_commitment_hash": fold_commitment_hash,
        }
        return cast(
            AlphaNumericalDevelopmentScoreChunkRef,
            self._publish_development_chunk(
                table=table,
                columns=self._DEVELOPMENT_SCORE_COLUMNS,
                category="numerical-development-score-chunks",
                kind="AlphaNumericalDevelopmentScoreChunk",
                identity=identity,
                model=AlphaNumericalDevelopmentScoreChunkRef,
            ),
        )

    def resolve_numerical_score_chunk(
        self,
        value: AlphaNumericalDevelopmentScoreChunkRef,
    ) -> pa.Table:
        return self._resolve_development_chunk(
            reference=value,
            columns=self._DEVELOPMENT_SCORE_COLUMNS,
            category="numerical-development-score-chunks",
            kind="AlphaNumericalDevelopmentScoreChunk",
            identity={
                "execution_binding_hash": value.execution_binding_hash,
                "development_surface_binding_hash": value.development_surface_binding_hash,
                "candidate_id": value.candidate_id,
                "candidate_card_hash": value.candidate_card_hash,
                "fold_index": value.fold_index,
                "fold_commitment_hash": value.fold_commitment_hash,
            },
        )

    def publish_canonical_target_recipe_binding(
        self, value: CanonicalAlphaTargetRecipeBinding
    ) -> str:
        return self._publish("canonical-target-recipe-bindings", value, "binding_hash")

    def load_canonical_target_recipe_binding(self, value: str) -> CanonicalAlphaTargetRecipeBinding:
        return self._load(
            "canonical-target-recipe-bindings",
            value,
            "binding_hash",
            CanonicalAlphaTargetRecipeBinding,
        )

    def publish_canonical_target_evidence(self, value: CanonicalAlphaTargetEvidence) -> str:
        return self._publish("canonical-target-evidence", value, "evidence_hash")

    def load_canonical_target_evidence(self, value: str) -> CanonicalAlphaTargetEvidence:
        return self._load(
            "canonical-target-evidence",
            value,
            "evidence_hash",
            CanonicalAlphaTargetEvidence,
        )

    def publish_return_unit_calibration(self, value: AlphaReturnUnitCalibrationEvidence) -> str:
        return self._publish(RETURN_UNIT_CALIBRATION_CATEGORY, value, "evidence_hash")

    def load_return_unit_calibration(self, value: str) -> AlphaReturnUnitCalibrationEvidence:
        return self._load(
            RETURN_UNIT_CALIBRATION_CATEGORY,
            value,
            "evidence_hash",
            AlphaReturnUnitCalibrationEvidence,
        )

    def publish_canonical_score_surface(self, value: CanonicalAlphaScoreSurface) -> str:
        return self._publish(CANONICAL_SCORE_SURFACE_CATEGORY, value, "surface_hash")

    def load_canonical_score_surface(self, value: str) -> CanonicalAlphaScoreSurface:
        return self._load(
            CANONICAL_SCORE_SURFACE_CATEGORY,
            value,
            "surface_hash",
            CanonicalAlphaScoreSurface,
        )

    def publish_target_preprocessing_detail(self, value: AlphaTargetClippingDetail) -> str:
        return self._publish("target-clipping-details", value, "detail_hash")

    def load_target_preprocessing_detail(self, value: str) -> AlphaTargetClippingDetail:
        return self._load(
            "target-clipping-details",
            value,
            "detail_hash",
            AlphaTargetClippingDetail,
        )

    def publish_target_preprocessing_receipt(self, value: AlphaTargetPreprocessingReceipt) -> str:
        return self._publish("target-preprocessing-receipts", value, "receipt_hash")

    def load_target_preprocessing_receipt(self, value: str) -> AlphaTargetPreprocessingReceipt:
        return self._load(
            "target-preprocessing-receipts",
            value,
            "receipt_hash",
            AlphaTargetPreprocessingReceipt,
        )

    def publish_comparison_program(self, value: AlphaTargetPreprocessingComparisonProgram) -> str:
        return self._publish("comparison-programs", value, "program_hash")

    def load_comparison_program(self, value: str) -> AlphaTargetPreprocessingComparisonProgram:
        return self._load(
            "comparison-programs",
            value,
            "program_hash",
            AlphaTargetPreprocessingComparisonProgram,
        )

    def development_category_dir(self, category: str) -> pathlib.Path:
        """Where one development category lives, spelled by the store itself.

        The writer wraps every development category under ``current/``; a caller
        assembling ``root / category`` by hand looks one level too high and
        silently reads an empty directory where published artifacts exist.
        """

        return self.root / "current" / category

    def study_root_hashes(self) -> tuple[str, ...]:
        """Every published study root's hash, sorted -- an axis, never a choice.

        Listing is for a caller that must refuse ambiguity: a reader that picked
        one of several roots by directory order would be selecting evidence.
        """

        category = self.development_category_dir("target-study-roots")
        if not category.is_dir():
            return ()
        return tuple(sorted(path.stem for path in category.glob("*.json")))

    def publish_comparison_evidence(self, value: AlphaTargetPreprocessingComparisonEvidence) -> str:
        return self._publish("comparison-evidence", value, "evidence_hash")

    def load_comparison_evidence(self, value: str) -> AlphaTargetPreprocessingComparisonEvidence:
        return self._load(
            "comparison-evidence",
            value,
            "evidence_hash",
            AlphaTargetPreprocessingComparisonEvidence,
        )

    def publish_canonical_score_binding(self, value: CanonicalAlphaScoreBinding) -> str:
        return self._publish(CANONICAL_SCORE_BINDING_CATEGORY, value, "binding_hash")

    def load_canonical_score_binding(self, value: str) -> CanonicalAlphaScoreBinding:
        return self._load(
            CANONICAL_SCORE_BINDING_CATEGORY,
            value,
            "binding_hash",
            CanonicalAlphaScoreBinding,
        )

    def publish_canonical_dispersion_forecast(self, value: CrossSectionalDispersionForecast) -> str:
        return self._publish("canonical-dispersion-forecasts", value, "forecast_hash")

    def load_canonical_dispersion_forecast(self, value: str) -> CrossSectionalDispersionForecast:
        return self._load(
            "canonical-dispersion-forecasts",
            value,
            "forecast_hash",
            CrossSectionalDispersionForecast,
        )

    def publish_candidate_fold_evidence(self, value: AlphaCandidateFoldEvidence) -> str:
        if not isinstance(value, AlphaCandidateFoldEvidence):
            raise TypeError("active Alpha fold evidence publisher requires the thin contract")
        return self._publish("candidate-fold-evidence", value, "fold_evidence_hash")

    def load_candidate_fold_evidence(
        self,
        value: str,
    ) -> AlphaCandidateFoldEvidence | LegacyAlphaCandidateFoldEvidence:
        uri = self.uri("current/candidate-fold-evidence", value)
        payload = self._load_identity_json(
            uri=uri,
            category="current/candidate-fold-evidence",
            identity_field="fold_evidence_hash",
        )
        if any(
            field in payload
            for field in (
                "score_chunk",
                "metrics",
                "session_rank_ics",
                "session_spreads",
                "fit_ledger",
                "estimator_state_hash",
                "failure",
            )
        ):
            result = cast(
                LegacyAlphaCandidateFoldEvidence,
                LegacyAlphaCandidateFoldEvidence.model_validate(payload),
            )
            if result.score_chunk is not None:
                if isinstance(result.score_chunk, AlphaNumericalDevelopmentScoreChunkRef):
                    self.resolve_numerical_score_chunk(result.score_chunk)
                else:
                    self.resolve_candidate_score_chunk(result.score_chunk)
            if result.estimator_state_hash is not None:
                self.load_any_development_estimator_state(result.estimator_state_hash)
            return result
        return cast(
            AlphaCandidateFoldEvidence,
            AlphaCandidateFoldEvidence.model_validate(payload),
        )

    def publish_candidate_numerical_fold_result(
        self,
        value: AlphaCandidateNumericalFoldResult,
    ) -> str:
        uri = self._publish("candidate-numerical-fold-results", value, "numerical_result_hash")
        pointer = (
            self.root
            / "current"
            / "candidate-numerical-fold-by-binding"
            / f"{value.execution_binding_hash}.json"
        )
        payload = {
            "execution_binding_hash": value.execution_binding_hash,
            "numerical_result_hash": value.numerical_result_hash,
            "numerical_result_ref": uri,
        }
        if pointer.is_file():
            existing = json.loads(pointer.read_text(encoding="utf-8"))
            if existing != payload:
                raise AlphaDevelopmentArtifactReadbackError(
                    "candidate-fold execution binding has divergent numerical evidence"
                )
        else:
            self._atomic_write(pointer, self._json_bytes(payload))
        return uri

    def load_candidate_numerical_fold_result(
        self,
        value: str,
    ) -> AlphaCandidateNumericalFoldResult:
        return self.read_candidate_numerical_fold_result(value)[0]

    def read_candidate_numerical_fold_result(
        self,
        value: str,
    ) -> tuple[AlphaCandidateNumericalFoldResult, pa.Table | None]:
        """The numerical fold result with its score chunk, resolved once for both.

        The chunk is proved as part of loading the result; a caller that reads
        the scores takes the rows of that same resolution. ``None`` when the
        result carries no score chunk.
        """

        result = self._load(
            "candidate-numerical-fold-results",
            value,
            "numerical_result_hash",
            AlphaCandidateNumericalFoldResult,
        )
        scores = (
            None
            if result.score_chunk is None
            else self.resolve_numerical_score_chunk(result.score_chunk)
        )
        if result.estimator_state_hash is not None:
            state = self.load_development_estimator_state(result.estimator_state_hash)
            if (
                state.execution_binding_hash != result.execution_binding_hash
                or state.development_surface_binding_hash != result.development_surface_binding_hash
                or state.candidate_id != result.candidate_id
                or state.fold_index != result.fold_index
            ):
                raise AlphaDevelopmentArtifactReadbackError(
                    "candidate numerical estimator binding changed"
                )
            if state.fit_evidence_hash is not None:
                if result.score_chunk is None or result.metrics is None:
                    raise AlphaDevelopmentArtifactReadbackError(
                        "candidate numerical fit evidence is incomplete"
                    )
                evidence = self.load_development_fit_evidence(state.fit_evidence_hash)
                score_evidence_hash = canonical_hash(
                    {
                        "score_chunk_content_hash": result.score_chunk.content_hash,
                        "fold_metrics_hash": result.metrics.metrics_hash,
                    }
                )
                if evidence.score_evidence_hash != score_evidence_hash:
                    raise AlphaDevelopmentArtifactReadbackError(
                        "candidate numerical score evidence changed"
                    )
        return result, scores

    def find_candidate_numerical_fold_result(
        self,
        execution_binding_hash: str,
    ) -> AlphaCandidateNumericalFoldResult | None:
        pointer = (
            self.root
            / "current"
            / "candidate-numerical-fold-by-binding"
            / f"{execution_binding_hash}.json"
        )
        if not pointer.is_file():
            return None
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        if str(payload.get("execution_binding_hash", "")) != execution_binding_hash:
            raise AlphaDevelopmentArtifactReadbackError("candidate-fold pointer binding changed")
        value = self.load_candidate_numerical_fold_result(
            str(payload.get("numerical_result_hash", ""))
        )
        if value.execution_binding_hash != execution_binding_hash:
            raise AlphaDevelopmentArtifactReadbackError("candidate-fold numerical binding changed")
        return value

    def publish_development_execution_receipt(
        self,
        value: AlphaDevelopmentExecutionReceipt,
    ) -> str:
        """Write the development authority parent through the store that owns the children.

        A second store beside this one would be a second answer to where Alpha
        development evidence lives, and the receipt exists precisely to be found
        next to the artifacts it names.
        """

        return self._publish(ALPHA_DEVELOPMENT_EXECUTION_RECEIPT_CATEGORY, value, "receipt_hash")

    def load_development_execution_receipt(self, value: str) -> AlphaDevelopmentExecutionReceipt:
        return self._load(
            ALPHA_DEVELOPMENT_EXECUTION_RECEIPT_CATEGORY,
            value,
            "receipt_hash",
            AlphaDevelopmentExecutionReceipt,
        )

    def publish_batch_result(self, value: AlphaExperimentBatchResult) -> str:
        """Seal the executed child axis so a reader can ask what the batch produced."""

        return self._publish(ALPHA_DEVELOPMENT_BATCH_RESULT_CATEGORY, value, "result_hash")

    def load_batch_result(self, value: str) -> AlphaExperimentBatchResult:
        return self._load(
            ALPHA_DEVELOPMENT_BATCH_RESULT_CATEGORY,
            value,
            "result_hash",
            AlphaExperimentBatchResult,
        )

    def publish_candidate_report(self, value: AlphaCandidateDevelopmentReport) -> str:
        return self._publish("candidate-development-reports", value, "report_hash")

    def load_candidate_report(self, value: str) -> AlphaCandidateDevelopmentReport:
        return self._load(
            "candidate-development-reports",
            value,
            "report_hash",
            AlphaCandidateDevelopmentReport,
        )

    def publish_candidate_inference_evidence(
        self,
        value: AlphaCandidateInferenceEvidence,
    ) -> str:
        return self._publish("candidate-inference-evidence", value, "evidence_hash")

    def load_candidate_inference_evidence(self, value: str) -> AlphaCandidateInferenceEvidence:
        return self._load(
            "candidate-inference-evidence",
            value,
            "evidence_hash",
            AlphaCandidateInferenceEvidence,
        )

    def _publish_development_chunk(
        self,
        *,
        table: pa.Table,
        columns: tuple[str, ...],
        category: str,
        kind: str,
        identity: dict[str, object],
        model: type[BaseModel],
    ) -> BaseModel:
        if tuple(table.schema.names) != columns or table.num_rows < 1:
            raise ValueError(f"{kind} schema is invalid")
        row_indices = tuple(int(value) for value in table["row_index"].to_pylist())
        if row_indices != tuple(range(table.num_rows)):
            raise ValueError(f"{kind} row index is not canonical")
        content_hash = _chunk_content_hash(table, identity)
        uri = self.uri(f"current/{category}", content_hash)
        values = {
            **identity,
            "row_count": table.num_rows,
            "content_hash": content_hash,
            "metadata_hash": "0" * 64,
            "uri": uri,
        }
        provisional = model.model_construct(**values)
        metadata_hash = canonical_hash(
            provisional.model_dump(mode="json", exclude={"metadata_hash", "uri"})
        )
        reference = model.model_validate({**values, "metadata_hash": metadata_hash})
        target = self._path(f"current/{category}", content_hash, "parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            b"alphalattice.snapshot_kind": kind.encode(),
            b"alphalattice.content_hash": content_hash.encode(),
            b"alphalattice.metadata_hash": metadata_hash.encode(),
        }
        staged = target.with_name(f".{content_hash}.{os.getpid()}.tmp")
        if not target.exists():
            pq.write_table(table.replace_schema_metadata(metadata), staged, compression="zstd")
            os.replace(staged, target)
        staged.unlink(missing_ok=True)
        self._resolve_development_chunk(
            reference=reference,
            columns=columns,
            category=category,
            kind=kind,
            identity=identity,
        )
        return reference

    def _resolve_development_chunk(
        self,
        *,
        reference: BaseModel,
        columns: tuple[str, ...],
        category: str,
        kind: str,
        identity: dict[str, object],
    ) -> pa.Table:
        content_hash = str(reference.content_hash)
        metadata_hash = str(reference.metadata_hash)
        uri = str(reference.uri)
        row_count = int(reference.row_count)
        if uri != self.uri(f"current/{category}", content_hash):
            raise AlphaDevelopmentArtifactReadbackError(f"{kind} URI changed")
        target = self._path(f"current/{category}", content_hash, "parquet")
        raw = target.read_bytes()
        parquet = pq.ParquetFile(pa.BufferReader(raw))
        metadata = parquet.schema_arrow.metadata or {}
        if (
            metadata.get(b"alphalattice.snapshot_kind") != kind.encode()
            or metadata.get(b"alphalattice.content_hash") != content_hash.encode()
            or metadata.get(b"alphalattice.metadata_hash") != metadata_hash.encode()
        ):
            raise AlphaDevelopmentArtifactReadbackError(f"{kind} metadata changed")
        table = pq.read_table(pa.BufferReader(raw)).replace_schema_metadata(None)
        if (
            tuple(table.schema.names) != columns
            or table.num_rows != row_count
            or _verified_content_hash(raw, table, identity) != content_hash
        ):
            raise AlphaDevelopmentArtifactReadbackError(f"{kind} content changed")
        for field, expected in identity.items():
            if getattr(reference, field) != expected:
                raise AlphaDevelopmentArtifactReadbackError(f"{kind} identity changed")
        return table

    def publish_estimator_state(self, value: AlphaEstimatorState) -> str:
        return self._publish("estimator-states", value, "state_hash")

    def publish_model_text(self, model_text: str) -> str:
        """Publish one opaque tree model payload under its canonical content hash."""

        content_hash = str(canonical_hash(model_text))
        path = self.root / "current" / "model-text" / f"{content_hash}.txt"
        encoded = model_text.encode("utf-8")
        if path.is_file():
            if path.read_bytes() != encoded:
                raise AlphaDevelopmentArtifactReadbackError("Alpha model-text identity was reused")
        else:
            self._atomic_write(path, encoded)
        if path.read_bytes() != encoded:
            raise AlphaDevelopmentArtifactReadbackError("Alpha model-text readback failed")
        return content_hash

    def load_model_text(self, content_hash: str) -> str:
        path: Path = self.root / "current" / "model-text" / f"{content_hash}.txt"
        value: str = path.read_text(encoding="utf-8")
        if canonical_hash(value) != content_hash:
            raise AlphaDevelopmentArtifactReadbackError("Alpha model-text content changed")
        return value

    def load_estimator_state(self, value: str) -> AlphaEstimatorState:
        result = self._load("estimator-states", value, "state_hash", AlphaEstimatorState)
        if result.model_text_hash is not None:
            self.load_model_text(result.model_text_hash)
        return result

    def publish_development_estimator_state(
        self,
        value: AlphaDevelopmentEstimatorState,
    ) -> str:
        return self._publish("development-estimator-states", value, "state_hash")

    def publish_development_fit_evidence(
        self,
        value: AlphaDevelopmentFitEvidence,
    ) -> str:
        return self._publish("development-fit-evidence", value, "evidence_hash")

    def publish_model_fit_plan(self, value: AlphaModelFitPlan) -> str:
        return self._publish("model-fit-plans", value, "fit_plan_hash")

    def publish_model_numerical_environment(
        self,
        value: AlphaModelNumericalEnvironment,
    ) -> str:
        return self._publish(
            "model-numerical-environments",
            value,
            "environment_hash",
        )

    def load_model_fit_plan(self, value: str) -> AlphaModelFitPlan:
        return self._load(
            "model-fit-plans",
            value,
            "fit_plan_hash",
            AlphaModelFitPlan,
        )

    def load_model_numerical_environment(
        self,
        value: str,
    ) -> AlphaModelNumericalEnvironment:
        return self._load(
            "model-numerical-environments",
            value,
            "environment_hash",
            AlphaModelNumericalEnvironment,
        )

    def load_development_fit_evidence(self, value: str) -> AlphaDevelopmentFitEvidence:
        return self._load(
            "development-fit-evidence",
            value,
            "evidence_hash",
            AlphaDevelopmentFitEvidence,
        )

    def load_development_estimator_state(
        self,
        value: str,
    ) -> AlphaDevelopmentEstimatorState:
        result = self._load(
            "development-estimator-states",
            value,
            "state_hash",
            AlphaDevelopmentEstimatorState,
        )
        if result.model_text_hash is not None:
            self.load_model_text(result.model_text_hash)
        if result.fit_evidence_hash is not None:
            evidence = self.load_development_fit_evidence(result.fit_evidence_hash)
            if evidence.fit_plan_hash is not None:
                fit_plan = self.load_model_fit_plan(evidence.fit_plan_hash)
                if (
                    fit_plan.parent_training_binding_hash != evidence.training_binding_hash
                    or fit_plan.numerical_binding_hash != evidence.numerical_binding_hash
                    or fit_plan.ordered_feature_ids != result.ordered_factor_ids
                ):
                    raise AlphaDevelopmentArtifactReadbackError(
                        "Alpha model fit plan binding changed"
                    )
            if evidence.numerical_environment_hash is not None:
                environment = self.load_model_numerical_environment(
                    evidence.numerical_environment_hash
                )
                if environment.numerical_binding_hash != evidence.numerical_binding_hash:
                    raise AlphaDevelopmentArtifactReadbackError(
                        "Alpha model numerical environment binding changed"
                    )
            if (
                evidence.execution_binding_hash != result.execution_binding_hash
                or evidence.candidate_id != result.candidate_id
                or evidence.fold_index != result.fold_index
                or evidence.adapter_id != result.adapter_id
                or evidence.numerical_binding_hash != result.numerical_binding_hash
                or evidence.state_projection_hash != result.state_projection_hash
            ):
                raise AlphaDevelopmentArtifactReadbackError(
                    "Alpha development fit evidence binding changed"
                )
            sidecar, estimator, provenance = self.load_model_fit_sidecar(
                evidence.execution_binding_hash
            )
            if (
                sidecar.estimator_content_hash != evidence.estimator_content_hash
                or sidecar.fit_provenance_hash != evidence.fit_provenance_hash
                or estimator.adapter_id != evidence.adapter_id
                or estimator.ordered_feature_ids != result.ordered_factor_ids
                or provenance.recipe_hash != evidence.recipe_hash
                or provenance.training_binding_hash != evidence.training_binding_hash
                or provenance.fit_plan_hash != evidence.fit_plan_hash
                or provenance.numerical_environment_hash != evidence.numerical_environment_hash
                or provenance.estimator_content_hash != evidence.estimator_content_hash
                or provenance.fit_call_count != evidence.fit_call_count
                or provenance.predict_call_count != evidence.predict_call_count
            ):
                raise AlphaDevelopmentArtifactReadbackError(
                    "Alpha development model evidence binding changed"
                )
        return result

    def load_any_development_estimator_state(
        self,
        value: str,
    ) -> AlphaDevelopmentEstimatorState | AlphaEstimatorState:
        development_path = self._path("current/development-estimator-states", value, "json")
        if development_path.is_file():
            return self.load_development_estimator_state(value)
        return self.load_estimator_state(value)

    def publish_viability(self, value: AlphaModelViabilityAssessment) -> str:
        return self._publish("viability", value, "assessment_hash")

    def load_viability(self, value: str) -> AlphaModelViabilityAssessment:
        return self._load(
            "viability",
            value,
            "assessment_hash",
            AlphaModelViabilityAssessment,
        )

    def publish_proposal(self, value: AlphaModelSelectionProposal) -> str:
        return self._publish("selection-proposals", value, "proposal_hash")

    def load_proposal(self, value: str) -> AlphaModelSelectionProposal:
        return self._load(
            "selection-proposals",
            value,
            "proposal_hash",
            AlphaModelSelectionProposal,
        )

    def publish_decision(self, value: AlphaModelSelectionDecision) -> str:
        return self._publish("selection-decisions", value, "decision_hash")

    def load_decision(self, value: str) -> AlphaModelSelectionDecision:
        return self._load(
            "selection-decisions",
            value,
            "decision_hash",
            AlphaModelSelectionDecision,
        )

    def publish_reproducibility(self, value: AlphaDecisionReproducibilityReport) -> str:
        return self._publish("decision-reproducibility", value, "report_hash")

    def load_reproducibility(self, value: str) -> AlphaDecisionReproducibilityReport:
        return self._load(
            "decision-reproducibility",
            value,
            "report_hash",
            AlphaDecisionReproducibilityReport,
        )


__all__ = [
    "AlphaDevelopmentArtifactReadbackError",
    "AlphaDevelopmentArtifactStore",
]
