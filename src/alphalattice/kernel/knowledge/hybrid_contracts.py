"""Strict native contracts for Workspace-owned local Hybrid v2 retrieval."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, StringConstraints, field_validator, model_validator

from alphalattice.kernel.knowledge.retrieval_contracts import (
    KnowledgeAccessClass,
    KnowledgeNamespace,
    KnowledgeRetrievalHit,
    LexicalIndexSpec,
    RetrievalChannel,
    RetrievalStatus,
    StableKnowledgeId,
)
from alphalattice.kernel.shared_kernel.domain.base import (
    DomainModel,
    NonEmptyString,
    Sha256Hex,
    ShortString,
    UtcDatetime,
    Uuid4,
)
from alphalattice.kernel.shared_kernel.domain.serialization import canonical_json_bytes, sha256_hex

PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
TopK = Annotated[int, Field(ge=1, le=20)]
RerankerDepth = Annotated[int, Field(ge=1, le=32)]
FiniteScore = Annotated[float, Field(allow_inf_nan=False)]
RECIPE_MINILM_CPU = "hybrid-v2-minilm"
RECIPE_BGE_SMALL_CPU = "hybrid-v3-bge-small-en"
RECIPE_QWEN3_ENCODER_GPU = "hybrid-v3-qwen3-embedding-gpu"
RECIPE_QWEN3_RERANKER_GPU = "hybrid-v3-minilm-qwen3-reranker-gpu"
RECIPE_QWEN3_GPU = "hybrid-v3-qwen3-gpu"
SUPPORTED_RECIPES: tuple[str, ...] = (
    RECIPE_MINILM_CPU,
    RECIPE_BGE_SMALL_CPU,
    RECIPE_QWEN3_ENCODER_GPU,
    RECIPE_QWEN3_RERANKER_GPU,
    RECIPE_QWEN3_GPU,
)
"""The bounded set of retrieval recipes: one payload each, no other pairing.

`hybrid-v2-minilm` is the payload every existing index and generation was
built under; the four `v3` recipes are the comparison arms of the evidence
initiative (a CPU encoder against the retained reranker; the Qwen3 encoder,
the Qwen3 reranker, and both, on the one supported local GPU backend). A
recipe is admitted by name; its identity is its whole payload."""
FixedPolicyId = Annotated[
    str,
    StringConstraints(
        pattern=(
            r"^hybrid-(v2-minilm|v3-bge-small-en|v3-qwen3-embedding-gpu"
            r"|v3-minilm-qwen3-reranker-gpu|v3-qwen3-gpu)$"
        )
    ),
]
FixedTokenNormalization = Annotated[
    str,
    StringConstraints(
        pattern=r"^(XLM_ROBERTA_SENTENCEPIECE_V1|BERT_WORDPIECE_TOKENIZERS_V1|QWEN2_BPE_TOKENIZERS_V1)$"
    ),
]
FixedPooling = Annotated[
    str, StringConstraints(pattern=r"^(MEAN_NON_PADDING|CLS_TOKEN|LAST_TOKEN)$")
]
FixedNormalization = Annotated[str, StringConstraints(pattern=r"^L2_FLOAT32$")]
FixedQuantization = Annotated[str, StringConstraints(pattern=r"^NONE$")]
FixedProvider = Annotated[str, StringConstraints(pattern=r"^(CPUExecutionProvider|TORCH_CUDA)$")]
FixedExecutionMode = Annotated[str, StringConstraints(pattern=r"^ORT_SEQUENTIAL$")]
EncoderRuntime = Annotated[str, StringConstraints(pattern=r"^(ONNX_CPU|TORCH_CUDA)$")]
RerankerRuntime = Annotated[
    str, StringConstraints(pattern=r"^(FASTEMBED_ONNX_CPU|TORCH_CUDA_CAUSAL_LM)$")
]
ENCODER_RUNTIME_ONNX_CPU = "ONNX_CPU"
ENCODER_RUNTIME_TORCH_CUDA = "TORCH_CUDA"
RERANKER_RUNTIME_FASTEMBED_CPU = "FASTEMBED_ONNX_CPU"
RERANKER_RUNTIME_TORCH_CUDA = "TORCH_CUDA_CAUSAL_LM"
FixedScoreType = Annotated[str, StringConstraints(pattern=r"^(?:RRF_RANK|CROSS_ENCODER_LOGIT)$")]
ImmutableModelRevision = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{40}$"),
]

RERANKER_MODEL_ID = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL_REVISION = "a09144355adeed5f58c8ed011d209bf8ee5a1fec"
RERANKER_PACK: tuple[tuple[str, str], ...] = (
    # Every file the installed FastEmbed loader reads before or during
    # inference. `load_tokenizer` consumes the three JSON configurations and the
    # tokenizer; the graph is run last. Verifying only the graph and the
    # tokenizer admitted a pack whose `model_max_length` was 16.
    ("config.json", "d827779a72d27ae68cf878a6fc2e954542663fe21ca515d9f4783fc96be2d37e"),
    ("onnx/model.onnx", "c623d0bcb99f4622beb413eaef00cfbe5db20df9f1dd982da4b4f26022881870"),
    ("special_tokens_map.json", "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3"),
    ("tokenizer.json", "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
    ("tokenizer_config.json", "0b29c7bfc889e53b36d9dd3e686dd4300f6525110eaa98c76a5dafceb2029f53"),
)
RERANKER_PACK_SHA256 = sha256_hex(
    canonical_json_bytes({name: digest for name, digest in sorted(RERANKER_PACK)})
)
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MODEL_ARTIFACT_SHA256 = "10f7a088420252b26caf819236ca2c9d2987afd0fc06fec7553b542a5655a05a"
TOKENIZER_SHA256 = "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"

# The BAAI encoder: the upstream fp32 ONNX export in the official repository
# (no converted or quantized third-party distribution) and its BERT WordPiece
# tokenizer file, the two files the ONNX adapter consumes.
BGE_MODEL_ID = "BAAI/bge-small-en-v1.5"
BGE_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_MODEL_ARTIFACT_SHA256 = "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"
BGE_TOKENIZER_SHA256 = "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"

# The Qwen3 packs: every file the transformers loader reads before or during
# inference (configuration, weights, tokenizer files, the reranker's chat
# template), hashed as one closure each, as the ms-marco pack is.
QWEN3_EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
QWEN3_EMBEDDING_MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QWEN3_EMBEDDING_PACK: tuple[tuple[str, str], ...] = (
    ("config.json", "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9"),
    ("generation_config.json", "28396d421a2108acce96383f6a7de78008f7f1b17f807958f3c14c51dbfb65fb"),
    ("merges.txt", "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    ("model.safetensors", "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"),
    ("tokenizer.json", "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"),
    ("tokenizer_config.json", "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0"),
    ("vocab.json", "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
)
QWEN3_EMBEDDING_PACK_SHA256 = sha256_hex(
    canonical_json_bytes({name: digest for name, digest in sorted(QWEN3_EMBEDDING_PACK)})
)
QWEN3_RERANKER_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
QWEN3_RERANKER_MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
QWEN3_RERANKER_PACK: tuple[tuple[str, str], ...] = (
    ("chat_template.jinja", "6f682162495ec5b39fd9005c01b6aa2a74669379fe967039f1e2cbbe8752369d"),
    ("config.json", "d479c427a9ca5295218063d4f9aca4f297ab4ac27487cca7af42c84643d51ef0"),
    ("generation_config.json", "81051cd3f6e77013827148d0b8a6ead93f8ac390d5ab805f849199f0af6a08db"),
    ("merges.txt", "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    ("model.safetensors", "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b"),
    ("tokenizer.json", "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    ("tokenizer_config.json", "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0"),
    ("vocab.json", "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
)
QWEN3_RERANKER_PACK_SHA256 = sha256_hex(
    canonical_json_bytes({name: digest for name, digest in sorted(QWEN3_RERANKER_PACK)})
)
# The one supported local GPU runtime, as locked in config/requirements-gpu.lock.
TORCH_VERSION = "2.14.0+cu130"
TRANSFORMERS_VERSION = "5.17.0"
CUDA_VERSION = "13.0"


def _logical_hash(value: DomainModel, *, exclude: set[str]) -> str:
    return sha256_hex(canonical_json_bytes(value.model_dump(mode="python", exclude=exclude)))


def _require_sorted_unique(values: tuple[object, ...], label: str) -> tuple[object, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    if values != tuple(sorted(values, key=str)):
        raise ValueError(f"{label} must be sorted")
    return values


def safe_intra_op_threads(cpu_count: int | None = None) -> int:
    """Choose the default intra-operation thread bound without a CPU budget.

    The desktop-safe intra-op threads of one model session: the default a
    session runs with when no CPU budget says otherwise.

    Args:
        cpu_count: Optional available CPU count; None uses the installed default discovery.

    Returns:
        Default positive intra-operation thread bound.
    """
    detected = os.cpu_count() if cpu_count is None else cpu_count
    return max(1, min(4, detected or 1))


MAXIMUM_SESSION_THREADS = 16
"""The most intra-op threads one session is given (S2 measured 1-16, identical)."""

RETRIEVAL_CANARY_TEXT = (
    "The registrant recorded a goodwill impairment charge in the quarter and disclosed that its "
    "credit facility's leverage covenant was amended after the reporting date."
)
RETRIEVAL_CANARY_QUERY = "Did the issuer disclose a cybersecurity incident?"
RETRIEVAL_CANARY_PASSAGE = (
    "On March 3 the company detected unauthorized access to part of its information systems, "
    "activated its incident response plan and notified law enforcement."
)
EMBEDDING_CANARIES: dict[str, str] = {
    # The multilingual MiniLM encoder's context (every recipe that uses it): the
    # canary text embedded as a passage, SHA-256 of its float32 little-endian
    # bytes, as the four threads every spec sealed before F1 named produced it
    # (identical at 1, 2, 8 and 16 threads; record F1).
    "8764a88dff566bf1a5eb63c6098ac0d7585901f96586a62d8efdf46a01e96d6b": (
        "eb627c3432fae3316dbfb65b50619b3c565e8ecb35bf348435d3294b31d1d51c"
    ),
}
"""The sealed canary vector of each ONNX encoder context, by `embedding_context_hash`."""
RERANKER_CANARIES: dict[str, str] = {
    # The ms-marco MiniLM cross-encoder's context: the canary pair's raw score
    # (-4.206943...), SHA-256 of its float32 little-endian bytes, at four threads
    # (identical at 1-16).
    "e844d0bfdce9e829dbcb5fc4a8013f6b4634c178b434cd70c76cf6e9e67535dd": (
        "9273c67a078a2b4ca9a288e62300e5d4913c9980ce50d5a4aa373b53b811d37a"
    ),
}
"""The sealed canary score of each ONNX cross-encoder context, by `reranker_context_hash`."""


class KnowledgeRetrievalMode(StrEnum):
    """Distinguish semantic hybrid retrieval from explicit lexical-only mode."""

    STANDARD_HYBRID = "STANDARD_HYBRID"
    LEXICAL_ONLY = "LEXICAL_ONLY"


class SemanticPackStatus(StrEnum):
    """Report ready, unavailable or lexical-only semantic-pack disposition."""

    READY = "READY"
    SEMANTIC_PACK_UNAVAILABLE = "SEMANTIC_PACK_UNAVAILABLE"
    LEXICAL_ONLY = "LEXICAL_ONLY"


class EmbeddingInputPolicy(StrEnum):
    """Identify the recipe-selected transformation of query and passage text.

    Policies distinguish E5 asymmetric prefixes, unchanged inputs, BGE query instructions and Qwen3
    retrieval instructions.
    """

    E5_ASYMMETRIC_V1 = "E5_ASYMMETRIC_V1"
    IDENTITY_V1 = "IDENTITY_V1"
    BGE_QUERY_INSTRUCTION_V1 = "BGE_QUERY_INSTRUCTION_V1"
    """Queries carry the BAAI retrieval instruction; passages are the text."""
    QWEN3_RETRIEVAL_INSTRUCT_V1 = "QWEN3_RETRIEVAL_INSTRUCT_V1"
    """Queries carry the Qwen3 `Instruct: … Query:` prefix; passages are the text."""


class RetrievalScoreSemantics(DomainModel):
    """Declare fixed ranking-score meaning without calibration or confidence percentages.

    RRF_RANK and CROSS_ENCODER_LOGIT both rank larger values first. Neither represents a probability
    or a calibrated confidence.
    """

    score_type: FixedScoreType
    calibrated: bool
    higher_is_better: bool
    confidence_percentage_allowed: bool

    @model_validator(mode="after")
    def validate_fixed_semantics(self) -> Self:
        """Require fixed, uncalibrated RRF-rank or cross-encoder-logit semantics.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: The score type or its calibration, ordering or percentage flags differ from
                the fixed policy.
        """
        actual = self.model_dump(mode="python", exclude={"schema_version"})
        allowed = (
            {
                "score_type": "RRF_RANK",
                "calibrated": False,
                "higher_is_better": True,
                "confidence_percentage_allowed": False,
            },
            {
                "score_type": "CROSS_ENCODER_LOGIT",
                "calibrated": False,
                "higher_is_better": True,
                "confidence_percentage_allowed": False,
            },
        )
        if actual not in allowed:
            raise ValueError("retrieval score semantics must describe an uncalibrated fixed score")
        return self

    @classmethod
    def cross_encoder_logit(cls) -> RetrievalScoreSemantics:
        """A cross-encoder relevance logit.

        Comparable across one model's queries; not a probability.
        """
        return cls(
            score_type="CROSS_ENCODER_LOGIT",
            calibrated=False,
            higher_is_better=True,
            confidence_percentage_allowed=False,
        )

    @classmethod
    def rrf_rank(cls) -> RetrievalScoreSemantics:
        """Construct the fixed uncalibrated reciprocal-rank-fusion score semantics.

        Returns:
            RRF_RANK semantics with higher-is-better and confidence percentages disabled.
        """
        return cls(
            score_type="RRF_RANK",
            calibrated=False,
            higher_is_better=True,
            confidence_percentage_allowed=False,
        )


HYBRID_INDEX_SCHEMA_VERSION = "knowledge-hybrid-v4"
"""The index schema the active manifest belongs to.

It is folded into `index_id` and into the database path, so a change to what a
row commits to -- the citation gaining an exact source range, for instance --
lands as a distinct generation in a distinct file rather than colliding with
its predecessor on one path and refusing.

`knowledge-hybrid-v4` keys a generation by the corpus it embeds (the sorted
revision commitments of a snapshot) and the spec, not by the snapshot that
happened to publish it, stores corpus-level citations, and commits to the
passage vectors as a separate content-addressed payload, so a request whose
snapshot resolves to the same corpus reuses the generation and a cold open
verifies the committed vectors instead of embedding the corpus again.
"""

LEGACY_HYBRID_INDEX_SCHEMA_VERSION = "knowledge-hybrid-v3"
"""The durable index schema of generations built before the corpus commitment.

Keyed by the Workspace snapshot, citations carry the snapshot, and the vectors
live only inside the database, so a cold open of such a generation has to
embed the corpus to verify it. Readable through its own manifest and the
legacy open path while a retained workspace still holds one; never built again.
"""

HybridIndexSchema = Literal["knowledge-hybrid-v3", "knowledge-hybrid-v4"]
VECTOR_PAYLOAD_ROOT = ".system/knowledge-vectors"
PAIR_SCORE_ROOT = ".system/knowledge-scores"
"""Under the workspace: the sealed blocks of cross-encoder pair scores, one
directory per reranker context (`reranker_context_hash`)."""
PAIR_SCORE_BLOCK_SUFFIX = ".scores"


@dataclass(frozen=True, slots=True)
class PairScoreAdmission:
    """Declare which sealed pair scores may be reused and which new bytes may be admitted.

    What a reader is allowed to reuse and to seal: the block names a
    sealed commitment of the workspace admits (a block on disk under any
    other name is a claim, never consulted), and the workspace's storage
    admission a new block is placed under (`(additional_bytes)`, raising
    to refuse; None admits nothing and seals nothing). A reader without
    an admission keeps its in-process cache only.
    """

    admitted: frozenset[str]
    admit: Callable[[int], None] | None = None


@dataclass(frozen=True, slots=True)
class SealedPairScoreBlock:
    """One block a reader sealed at its close: what the caller commits."""

    name: str
    pair_count: int
    byte_length: int


class HybridCapabilityReport(DomainModel):
    """Seal semantic-pack capability, exact model/runtime identities and provider truth.

    READY requires complete identities and STANDARD_HYBRID mode. LEXICAL_ONLY names no semantic
    identities or semantic provider; an unavailable report remains a hybrid-mode outcome.
    """

    mode: KnowledgeRetrievalMode
    status: SemanticPackStatus
    model_id: ShortString | None
    model_revision: ImmutableModelRevision | None
    model_artifact_sha256: Sha256Hex | None
    tokenizer_sha256: Sha256Hex | None
    onnx_runtime_version: ShortString | None
    sentencepiece_version: ShortString | None
    vector_extension_version: ShortString | None
    cpu_execution_provider_only: bool
    details: tuple[ShortString, ...]
    encoder_runtime: EncoderRuntime = Field(
        default=ENCODER_RUNTIME_ONNX_CPU,
        exclude_if=lambda value: value == ENCODER_RUNTIME_ONNX_CPU,
    )
    reranker_runtime: RerankerRuntime = Field(
        default=RERANKER_RUNTIME_FASTEMBED_CPU,
        exclude_if=lambda value: value == RERANKER_RUNTIME_FASTEMBED_CPU,
    )
    """The runtimes the report proved; absent from the identity at the CPU
    defaults, so every manifest that bound a CPU report keeps its hash."""
    logical_hash: Sha256Hex

    @field_validator("details")
    @classmethod
    def validate_details(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require sorted unique capability details.

        Args:
            values: Capability detail strings in canonical order.

        Returns:
            The unchanged detail tuple.

        Raises:
            ValueError: Details are unordered or duplicated.
        """
        return _require_sorted_unique(values, "capability details")  # type: ignore[return-value]

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        """Require status-consistent semantic identities, provider claims and report hash.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: Mode/status, identity completeness, execution-provider truth or logical_hash
                are inconsistent.
        """
        identities = (
            self.model_id,
            self.model_revision,
            self.model_artifact_sha256,
            self.tokenizer_sha256,
            self.onnx_runtime_version,
            self.sentencepiece_version,
            self.vector_extension_version,
        )
        if self.status is SemanticPackStatus.READY:
            if self.mode is not KnowledgeRetrievalMode.STANDARD_HYBRID:
                raise ValueError("READY semantic pack requires STANDARD_HYBRID mode")
            if any(value is None for value in identities):
                raise ValueError("READY semantic pack requires complete identities")
            gpu = ENCODER_RUNTIME_TORCH_CUDA in (self.encoder_runtime, self.reranker_runtime) or (
                self.reranker_runtime == RERANKER_RUNTIME_TORCH_CUDA
            )
            if self.cpu_execution_provider_only == gpu:
                raise ValueError("READY semantic pack must state its execution provider truthfully")
        elif self.status is SemanticPackStatus.LEXICAL_ONLY:
            if self.mode is not KnowledgeRetrievalMode.LEXICAL_ONLY:
                raise ValueError("LEXICAL_ONLY status requires LEXICAL_ONLY mode")
            if any(value is not None for value in identities):
                raise ValueError("LEXICAL_ONLY report must not name semantic identities")
            if self.cpu_execution_provider_only:
                raise ValueError("LEXICAL_ONLY report must not claim a semantic provider")
        elif self.mode is not KnowledgeRetrievalMode.STANDARD_HYBRID:
            raise ValueError("unavailable semantic pack requires STANDARD_HYBRID mode")
        if self.logical_hash != _logical_hash(self, exclude={"logical_hash"}):
            raise ValueError("capability report logical_hash is inconsistent")
        return self

    @classmethod
    def ready(cls, spec: HybridIndexSpec) -> HybridCapabilityReport:
        """Project a sealed READY capability report from a supplied recipe.

        This factory records recipe identities and provider flags; its caller is responsible for
        establishing local capability before using a READY claim.

        Args:
            spec: Exact installed encoder/reranker recipe whose capability was established by the
                caller.

        Returns:
            READY STANDARD_HYBRID report with a canonical logical hash.
        """
        cpu_only = spec.execution_provider == "CPUExecutionProvider" and (
            spec.reranker_runtime == RERANKER_RUNTIME_FASTEMBED_CPU
        )
        payload = {
            "schema_version": "1",
            "mode": KnowledgeRetrievalMode.STANDARD_HYBRID,
            "status": SemanticPackStatus.READY,
            "model_id": spec.model_id,
            "model_revision": spec.model_revision,
            "model_artifact_sha256": spec.model_artifact_sha256,
            "tokenizer_sha256": spec.tokenizer_sha256,
            "onnx_runtime_version": spec.onnx_runtime_version,
            "sentencepiece_version": spec.sentencepiece_version,
            "vector_extension_version": spec.vector_extension_version,
            "cpu_execution_provider_only": cpu_only,
            "details": (
                "model:verified",
                "onnxruntime:verified",
                "sentencepiece:verified",
                "sqlite-vec:verified",
                "tokenizer:verified",
            ),
        }
        if spec.encoder_runtime != ENCODER_RUNTIME_ONNX_CPU:
            payload["encoder_runtime"] = spec.encoder_runtime
        if spec.reranker_runtime != RERANKER_RUNTIME_FASTEMBED_CPU:
            payload["reranker_runtime"] = spec.reranker_runtime
        return cls.model_validate(
            {**payload, "logical_hash": sha256_hex(canonical_json_bytes(payload))}
        )


class HybridIndexSpec(DomainModel):
    """Bind the installed lexical, encoder and reranker recipe to one canonical identity.

    The recipe fixes model/tokenizer pins, input policy, vector shape, normalization,
    runtime/provider, candidate and reranker depths, channel fusion and score rounding. Legacy
    execution fields are admitted only by the installed legacy profile; model construction does not
    probe local artifacts.
    """

    policy_id: FixedPolicyId
    lexical_spec: LexicalIndexSpec
    model_id: ShortString
    model_revision: ImmutableModelRevision
    model_artifact_sha256: Sha256Hex
    tokenizer_sha256: Sha256Hex
    license_id: ShortString
    embedding_input_policy: EmbeddingInputPolicy
    token_normalization: FixedTokenNormalization
    embedding_dimension: PositiveInt
    maximum_tokens: PositiveInt
    pooling: FixedPooling
    vector_normalization: FixedNormalization
    quantization: FixedQuantization
    onnx_runtime_version: ShortString
    sentencepiece_version: ShortString
    vector_extension_version: ShortString
    execution_provider: FixedProvider
    # Execution fields. A spec sealed before the final close-out (F1) named
    # the threads its sessions ran with; since F1 they are the operator's (the
    # CPU budget: `session_threads` on each session), proved on every machine
    # by the retrieval canary and absent from every new identity. A sealed
    # legacy spec reads back with its own values and its own hash.
    execution_mode: FixedExecutionMode | None = Field(default=None, exclude_if=lambda v: v is None)
    inter_op_num_threads: PositiveInt | None = Field(default=None, exclude_if=lambda v: v is None)
    intra_op_num_threads: PositiveInt | None = Field(default=None, exclude_if=lambda v: v is None)
    intra_op_spinning: bool | None = Field(default=None, exclude_if=lambda v: v is None)
    inter_op_spinning: bool | None = Field(default=None, exclude_if=lambda v: v is None)
    batch_size: PositiveInt
    candidate_depth_per_document: PositiveInt
    reranker_model_id: ShortString
    reranker_model_revision: ImmutableModelRevision
    reranker_pack_sha256: Sha256Hex
    reranker_runtime_version: ShortString
    reranker_tokenizer_runtime_version: ShortString
    reranker_depth_per_document: PositiveInt
    reranker_score_decimal_places: Annotated[int, Field(ge=0, le=15)]
    rrf_constant: PositiveInt
    term_all_weight: FiniteScore
    term_any_weight: FiniteScore
    cjk_bigram_weight: FiniteScore
    trigram_weight: FiniteScore
    dense_weight: FiniteScore
    dense_distance_decimal_places: Annotated[int, Field(ge=0, le=15)]
    vector_tolerance: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    # Recipe fields, absent from the identity at the CPU defaults so the
    # `hybrid-v2-minilm` payload, its embedding context, its indexes and its
    # generations keep the hashes they were published under.
    encoder_runtime: EncoderRuntime = Field(
        default=ENCODER_RUNTIME_ONNX_CPU,
        exclude_if=lambda value: value == ENCODER_RUNTIME_ONNX_CPU,
    )
    encoder_pack_sha256: Sha256Hex | None = Field(default=None, exclude_if=lambda v: v is None)
    """The whole consumed encoder pack for a multi-file (torch) pack; the two
    hashes above name the weights and the tokenizer inside it."""
    reranker_runtime: RerankerRuntime = Field(
        default=RERANKER_RUNTIME_FASTEMBED_CPU,
        exclude_if=lambda value: value == RERANKER_RUNTIME_FASTEMBED_CPU,
    )
    reranker_maximum_tokens: PositiveInt | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    # The torch runtime each side runs under, named per side: the encoder's
    # belongs to the embedding context (it can move the vectors), the
    # reranker's does not (a reranked order is never persisted).
    encoder_torch_version: ShortString | None = Field(default=None, exclude_if=lambda v: v is None)
    encoder_transformers_version: ShortString | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    encoder_cuda_version: ShortString | None = Field(default=None, exclude_if=lambda v: v is None)
    encoder_torch_dtype: ShortString | None = Field(default=None, exclude_if=lambda v: v is None)
    reranker_torch_version: ShortString | None = Field(default=None, exclude_if=lambda v: v is None)
    reranker_transformers_version: ShortString | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    reranker_cuda_version: ShortString | None = Field(default=None, exclude_if=lambda v: v is None)
    reranker_torch_dtype: ShortString | None = Field(default=None, exclude_if=lambda v: v is None)
    logical_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_recipe(self) -> Self:
        """Require the exact installed numerical recipe and its canonical identity.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: Recipe values, allowed legacy execution fields or logical_hash differ from
                the installed recipe.
        """
        actual = self.model_dump(mode="python", exclude={"schema_version", "logical_hash"})
        expected = recipe_payload(self.policy_id)
        if self.intra_op_num_threads is not None:
            expected.update(legacy_execution_payload(self.intra_op_num_threads))
        expected.pop("schema_version")
        lexical_spec = expected["lexical_spec"]
        assert isinstance(lexical_spec, LexicalIndexSpec)
        expected["lexical_spec"] = lexical_spec.model_dump(mode="python")
        if actual != expected:
            raise ValueError(f"HybridIndexSpec must match the recipe {self.policy_id}")
        if self.logical_hash != _logical_hash(self, exclude={"logical_hash"}):
            raise ValueError("hybrid index spec logical_hash is inconsistent")
        return self

    @classmethod
    def fixed_v2(cls) -> HybridIndexSpec:
        """Construct the installed MiniLM CPU hybrid recipe.

        Returns:
            Exact RECIPE_MINILM_CPU specification with its canonical logical identity.
        """
        return cls.for_recipe(RECIPE_MINILM_CPU)

    @classmethod
    def for_recipe(cls, policy_id: str) -> HybridIndexSpec:
        """Construct an exact numerical recipe without execution-width fields.

        The one spec a recipe name admits: what decides a result, never how
        many threads compute it.

        Args:
            policy_id: Installed numerical recipe identifier.

        Returns:
            Validated recipe with its canonical identity and no execution-width fields.
        """
        payload = recipe_payload(policy_id)
        return cls.model_validate(
            {**payload, "logical_hash": sha256_hex(canonical_json_bytes(payload))}
        )

    @classmethod
    def sealed_before_f1(cls, policy_id: str, intra_op_num_threads: int) -> HybridIndexSpec:
        """Construct the legacy thread-bound specification for sealed readback.

        The spec a recipe was sealed under before F1, naming its sessions'
        threads: what a legacy manifest holds, for reading it back.

        Args:
            policy_id: Installed recipe identifier.
            intra_op_num_threads: Thread width recorded by the legacy sealed specification.

        Returns:
            Validated legacy recipe preserving the execution fields required for readback.
        """
        payload = {**recipe_payload(policy_id), **legacy_execution_payload(intra_op_num_threads)}
        return cls.model_validate(
            {**payload, "logical_hash": sha256_hex(canonical_json_bytes(payload))}
        )

    @property
    def names_execution(self) -> bool:
        """A legacy spec: sealed before F1 with its sessions' threads."""
        return self.intra_op_num_threads is not None

    def without_execution(self) -> HybridIndexSpec:
        """Project the numerical recipe after removing legacy execution fields.

        This spec as F1 states it: a legacy spec differs from its recipe
        only by the execution fields (the validator holds that), so its
        neutral form is the recipe's own spec.

        Returns:
            Validated numerical recipe with legacy execution fields removed.
        """
        return HybridIndexSpec.for_recipe(self.policy_id) if self.names_execution else self


def legacy_execution_payload(intra_op_num_threads: int) -> dict[str, object]:
    """The five execution fields every spec sealed before F1 carried."""
    return {
        "execution_mode": "ORT_SEQUENTIAL",
        "inter_op_num_threads": 1,
        "intra_op_num_threads": intra_op_num_threads,
        "intra_op_spinning": False,
        "inter_op_spinning": False,
    }


def recipe_payload(policy_id: str) -> dict[str, object]:
    """The payload of one supported recipe, or a refusal naming the unknown one."""
    if policy_id == RECIPE_MINILM_CPU:
        return _fixed_spec_payload()
    if policy_id == RECIPE_BGE_SMALL_CPU:
        return {
            **_fixed_spec_payload(),
            "policy_id": RECIPE_BGE_SMALL_CPU,
            "model_id": BGE_MODEL_ID,
            "model_revision": BGE_MODEL_REVISION,
            "model_artifact_sha256": BGE_MODEL_ARTIFACT_SHA256,
            "tokenizer_sha256": BGE_TOKENIZER_SHA256,
            "license_id": "MIT",
            "embedding_input_policy": EmbeddingInputPolicy.BGE_QUERY_INSTRUCTION_V1,
            "token_normalization": "BERT_WORDPIECE_TOKENIZERS_V1",
            "embedding_dimension": 384,
            "pooling": "CLS_TOKEN",
        }
    if policy_id in (RECIPE_QWEN3_ENCODER_GPU, RECIPE_QWEN3_GPU):
        payload: dict[str, object] = {
            **_fixed_spec_payload(),
            "policy_id": policy_id,
            "model_id": QWEN3_EMBEDDING_MODEL_ID,
            "model_revision": QWEN3_EMBEDDING_MODEL_REVISION,
            "model_artifact_sha256": dict(QWEN3_EMBEDDING_PACK)["model.safetensors"],
            "tokenizer_sha256": dict(QWEN3_EMBEDDING_PACK)["tokenizer.json"],
            "license_id": "Apache-2.0",
            "embedding_input_policy": EmbeddingInputPolicy.QWEN3_RETRIEVAL_INSTRUCT_V1,
            "token_normalization": "QWEN2_BPE_TOKENIZERS_V1",
            "embedding_dimension": 1024,
            "pooling": "LAST_TOKEN",
            "execution_provider": "TORCH_CUDA",
            "batch_size": 16,
            "encoder_runtime": ENCODER_RUNTIME_TORCH_CUDA,
            "encoder_pack_sha256": QWEN3_EMBEDDING_PACK_SHA256,
            "encoder_torch_version": TORCH_VERSION,
            "encoder_transformers_version": TRANSFORMERS_VERSION,
            "encoder_cuda_version": CUDA_VERSION,
            "encoder_torch_dtype": "float32",
        }
        if policy_id == RECIPE_QWEN3_GPU:
            payload.update(_qwen3_reranker_fields())
        return payload
    if policy_id == RECIPE_QWEN3_RERANKER_GPU:
        return {
            **_fixed_spec_payload(),
            "policy_id": RECIPE_QWEN3_RERANKER_GPU,
            **_qwen3_reranker_fields(),
        }
    raise ValueError(f"unsupported retrieval recipe: {policy_id}")


def _qwen3_reranker_fields() -> dict[str, object]:
    return {
        "reranker_model_id": QWEN3_RERANKER_MODEL_ID,
        "reranker_model_revision": QWEN3_RERANKER_MODEL_REVISION,
        "reranker_pack_sha256": QWEN3_RERANKER_PACK_SHA256,
        "reranker_runtime": RERANKER_RUNTIME_TORCH_CUDA,
        "reranker_maximum_tokens": 1024,
        "reranker_torch_version": TORCH_VERSION,
        "reranker_transformers_version": TRANSFORMERS_VERSION,
        "reranker_cuda_version": CUDA_VERSION,
        "reranker_torch_dtype": "float32",
    }


def _fixed_spec_payload() -> dict[str, object]:
    return {
        "schema_version": "1",
        "policy_id": "hybrid-v2-minilm",
        "lexical_spec": LexicalIndexSpec.fixed_v1(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_artifact_sha256": MODEL_ARTIFACT_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "license_id": "Apache-2.0",
        "embedding_input_policy": EmbeddingInputPolicy.IDENTITY_V1,
        "token_normalization": "XLM_ROBERTA_SENTENCEPIECE_V1",
        "embedding_dimension": 384,
        "maximum_tokens": 512,
        "pooling": "MEAN_NON_PADDING",
        "vector_normalization": "L2_FLOAT32",
        "quantization": "NONE",
        "onnx_runtime_version": "1.27.0",
        "sentencepiece_version": "0.2.1",
        "vector_extension_version": "0.1.9",
        "execution_provider": "CPUExecutionProvider",
        "batch_size": 8,
        "candidate_depth_per_document": 32,
        "reranker_model_id": RERANKER_MODEL_ID,
        "reranker_model_revision": RERANKER_MODEL_REVISION,
        "reranker_pack_sha256": RERANKER_PACK_SHA256,
        "reranker_runtime_version": "0.8.0",
        "reranker_tokenizer_runtime_version": "0.23.2",
        "reranker_depth_per_document": 6,
        "reranker_score_decimal_places": 4,
        "rrf_constant": 60,
        "term_all_weight": 1.25,
        "term_any_weight": 1.0,
        "cjk_bigram_weight": 0.9,
        "trigram_weight": 0.5,
        "dense_weight": 1.25,
        "dense_distance_decimal_places": 8,
        "vector_tolerance": 0.00001,
    }


VECTOR_BLOCK_SIDECAR_SUFFIX = ".blocks.json"
"""Under the vector root, named by a generation's manifest hash: which
per-revision blocks compose that generation's payload."""

_EMBEDDING_CONTEXT_FIELDS = (
    "model_id",
    "model_revision",
    "model_artifact_sha256",
    "tokenizer_sha256",
    "embedding_input_policy",
    "token_normalization",
    "embedding_dimension",
    "maximum_tokens",
    "pooling",
    "vector_normalization",
    "quantization",
    "onnx_runtime_version",
    "sentencepiece_version",
    "execution_provider",
    "execution_mode",
    "inter_op_num_threads",
    "intra_op_num_threads",
    "intra_op_spinning",
    "inter_op_spinning",
    "batch_size",
    # Recipe fields: excluded from a dump at their CPU defaults, so the v2
    # context hash is unchanged and a torch recipe's context names its runtime.
    "encoder_runtime",
    "encoder_pack_sha256",
    "encoder_torch_version",
    "encoder_transformers_version",
    "encoder_cuda_version",
    "encoder_torch_dtype",
)


def embedding_context_hash(spec: HybridIndexSpec) -> str:
    """The identity of everything that can move an encoder's output.

    The pinned model and tokenizer, the input policy and normalization, the
    pooling and vector normalization, the runtime versions, the execution
    provider and the batch size -- every spec field the passage model's
    numbers depend on, and none of the fields that only decide how chunks are
    cut (those are in the inputs) or how hits are ranked. The threads are not
    among them since F1: the retrieval canary proves on each session that they
    move no number; a legacy spec's context still names the ones it sealed.
    Two vectors share an asset only under one context.
    """
    payload = spec.model_dump(mode="python", include=set(_EMBEDDING_CONTEXT_FIELDS))
    return sha256_hex(canonical_json_bytes({"kind": "hybrid-embedding-context", **payload}))


_RERANKER_CONTEXT_FIELDS = (
    "reranker_model_id",
    "reranker_model_revision",
    "reranker_pack_sha256",
    "reranker_runtime_version",
    "reranker_tokenizer_runtime_version",
    "reranker_score_decimal_places",
    "execution_provider",
    "execution_mode",
    "inter_op_num_threads",
    "intra_op_num_threads",
    "intra_op_spinning",
    "inter_op_spinning",
    "batch_size",
    # Recipe fields: excluded from a dump at their CPU defaults, so a torch
    # recipe's context names its runtime and the CPU context is unchanged.
    "reranker_runtime",
    "reranker_maximum_tokens",
    "reranker_torch_version",
    "reranker_transformers_version",
    "reranker_cuda_version",
    "reranker_torch_dtype",
)


def reranker_context_hash(spec: HybridIndexSpec) -> str:
    """The identity of everything that can move a cross-encoder's score.

    The pinned reranker pack and its runtime versions, the rounding, the
    execution provider and mode, the thread and spinning settings and the
    batch size -- every spec field a pair's score depends on, and none of
    the fields that decide what is scored (the depth) or how the channels
    fuse. A pair's score is reused only under one context.
    """
    payload = spec.model_dump(mode="python", include=set(_RERANKER_CONTEXT_FIELDS))
    return sha256_hex(canonical_json_bytes({"kind": "hybrid-reranker-context", **payload}))


def vector_asset_key(context_hash: str, embedding_input_hashes: tuple[str, ...]) -> str:
    """One revision's vectors under one context: the reuse key of a block.

    Keyed by the exact encoder inputs (each chunk's embedding input, in order)
    and the complete context, never by a file name, a revision label or the
    time a document was last used; a revised document with other inputs is
    another key, and a changed model or profile is another context.
    """
    return sha256_hex(
        canonical_json_bytes(
            {
                "kind": "hybrid-vector-block",
                "embedding_context_hash": context_hash,
                "embedding_input_hashes": list(embedding_input_hashes),
            }
        )
    )


class HybridVectorBlock(DomainModel):
    """The vectors of one revision's chunks, in chunk order, as one immutable object."""

    revision_logical_hash: Sha256Hex
    asset_key: Sha256Hex
    block_sha256: Sha256Hex
    vector_count: PositiveInt
    byte_length: PositiveInt

    @property
    def block_path(self) -> str:
        """Return the content-addressed float32 block path.

        Returns:
            Relative vector-payload path derived from block_sha256.
        """
        return f"{VECTOR_PAYLOAD_ROOT}/{self.block_sha256}.f32"


class HybridVectorBlockSidecar(DomainModel):
    """Which blocks compose one generation's committed payload.

    A payload built from blocks is never written as one file: its bytes are
    the blocks concatenated in this order, and its digest -- the one the
    manifest and the sealed record commit to -- is the digest of that
    concatenation. Kept under the generation's manifest hash, because two
    generations may hold the same bytes cut at different revisions. This
    file is not an authority: it is verified by reading every block it
    names and hashing them back to the committed digest.
    """

    kind: Literal["hybrid-vector-blocks"] = "hybrid-vector-blocks"
    manifest_logical_hash: Sha256Hex
    payload_sha256: Sha256Hex
    embedding_context_hash: Sha256Hex
    embedding_dimension: PositiveInt
    blocks: tuple[HybridVectorBlock, ...]

    @model_validator(mode="after")
    def validate_sidecar(self) -> Self:
        """Require every block byte length to match its float32 vector shape.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: A block length differs from vector_count times embedding_dimension times
                four bytes.
        """
        width = self.embedding_dimension * 4
        for block in self.blocks:
            if block.byte_length != block.vector_count * width:
                raise ValueError("vector block byte length differs from its shape")
        return self


class HybridVectorCommitment(DomainModel):
    """What the pinned passage model produced, committed as a payload of its own.

    The payload is the serialized vectors in manifest order -- the exact bytes
    the database rows hold -- addressed by its SHA-256 and kept beside the
    index, so a reopened or rebuilt generation verifies these bytes against
    this commitment instead of running the model over the corpus again.
    """

    embedding_dimension: PositiveInt
    dtype: Annotated[str, StringConstraints(pattern=r"^float32$")]
    byte_order: Annotated[str, StringConstraints(pattern=r"^little$")]
    vector_normalization: FixedNormalization
    vector_count: NonNegativeInt
    payload_bytes: NonNegativeInt
    payload_sha256: Sha256Hex
    payload_path: NonEmptyString

    @model_validator(mode="after")
    def validate_commitment(self) -> Self:
        """Require float32 payload size and the content-addressed payload path.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: Payload bytes differ from the vector shape or payload_path differs from the
                committed digest.
        """
        if self.payload_bytes != self.vector_count * self.embedding_dimension * 4:
            raise ValueError("vector payload byte count differs from its shape")
        if self.payload_path != f"{VECTOR_PAYLOAD_ROOT}/{self.payload_sha256}.f32":
            raise ValueError("vector payload_path is not content-addressed")
        return self


class HybridKnowledgeIndexManifest(DomainModel):
    """One `knowledge-hybrid-v4` generation: a corpus, a spec, and what was built."""

    index_id: Sha256Hex
    database_path: NonEmptyString
    corpus_logical_hash: Sha256Hex
    index_spec: HybridIndexSpec
    document_count: NonNegativeInt
    chunk_count: NonNegativeInt
    ordered_chunk_ids: tuple[Sha256Hex, ...]
    ordered_citation_hashes: tuple[Sha256Hex, ...]
    ordered_embedding_input_hashes: tuple[Sha256Hex, ...]
    projection_logical_hash: Sha256Hex
    vector_commitment: HybridVectorCommitment
    logical_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        """Require aligned chunk commitments, canonical path, vector shape and manifest hash.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: Commitment counts, unique chunk IDs, database path, vector
                shape/normalization or hash are invalid.
        """
        commitments = (
            self.ordered_chunk_ids,
            self.ordered_citation_hashes,
            self.ordered_embedding_input_hashes,
        )
        if any(len(values) != self.chunk_count for values in commitments):
            raise ValueError("hybrid manifest commitment count differs from chunk_count")
        if len(set(self.ordered_chunk_ids)) != len(self.ordered_chunk_ids):
            raise ValueError("hybrid chunk IDs must be unique")
        expected_path = (
            f".system/knowledge-indexes/{HYBRID_INDEX_SCHEMA_VERSION}/"
            f"{self.corpus_logical_hash}-{self.index_spec.logical_hash}.db"
        )
        if self.database_path != expected_path:
            raise ValueError("hybrid database_path is not canonical")
        commitment = self.vector_commitment
        if (
            commitment.vector_count != self.chunk_count
            or commitment.embedding_dimension != self.index_spec.embedding_dimension
            or commitment.vector_normalization != self.index_spec.vector_normalization
        ):
            raise ValueError("hybrid vector commitment differs from the manifest shape")
        if self.logical_hash != _logical_hash(self, exclude={"logical_hash"}):
            raise ValueError("hybrid manifest logical_hash is inconsistent")
        return self

    def source_commitments(self) -> dict[str, object]:
        """The part a source re-derivation must reproduce; the vectors are separate."""
        return {
            "index_id": self.index_id,
            "corpus_logical_hash": self.corpus_logical_hash,
            "index_spec_logical_hash": self.index_spec.logical_hash,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "ordered_chunk_ids": self.ordered_chunk_ids,
            "ordered_citation_hashes": self.ordered_citation_hashes,
            "ordered_embedding_input_hashes": self.ordered_embedding_input_hashes,
            "projection_logical_hash": self.projection_logical_hash,
        }


class HybridV3KnowledgeIndexManifest(DomainModel):
    """The `knowledge-hybrid-v3` manifest: keyed by its snapshot, vectors in the DB only."""

    index_id: Sha256Hex
    database_path: NonEmptyString
    snapshot_id: Uuid4
    snapshot_logical_hash: Sha256Hex
    index_spec: HybridIndexSpec
    document_count: NonNegativeInt
    chunk_count: NonNegativeInt
    ordered_chunk_ids: tuple[Sha256Hex, ...]
    ordered_citation_hashes: tuple[Sha256Hex, ...]
    ordered_embedding_input_hashes: tuple[Sha256Hex, ...]
    projection_logical_hash: Sha256Hex
    logical_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        """Require aligned legacy chunk commitments, canonical path and manifest hash.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: Commitment counts, unique chunk IDs, snapshot-addressed database path or
                hash are invalid.
        """
        commitments = (
            self.ordered_chunk_ids,
            self.ordered_citation_hashes,
            self.ordered_embedding_input_hashes,
        )
        if any(len(values) != self.chunk_count for values in commitments):
            raise ValueError("hybrid manifest commitment count differs from chunk_count")
        if len(set(self.ordered_chunk_ids)) != len(self.ordered_chunk_ids):
            raise ValueError("hybrid chunk IDs must be unique")
        expected_path = (
            f".system/knowledge-indexes/{LEGACY_HYBRID_INDEX_SCHEMA_VERSION}/"
            f"{self.snapshot_logical_hash}-{self.index_spec.logical_hash}.db"
        )
        if self.database_path != expected_path:
            raise ValueError("hybrid database_path is not canonical")
        if self.logical_hash != _logical_hash(self, exclude={"logical_hash"}):
            raise ValueError("hybrid manifest logical_hash is inconsistent")
        return self


class HybridGenerationAnchor(DomainModel):
    """Hold the caller-sealed manifest and vector-payload commitment for one generation.

    The durable commitment a caller holds for one `knowledge-hybrid-v4`
    generation: the manifest it sealed and the payload that manifest names.

    The Workspace never takes a generation's identity from what it finds on
    disk. A database, its in-database manifest and an eviction marker are
    mutable files that can be rewritten together; only a commitment the
    caller sealed elsewhere -- and re-read, verified, at the moment of use --
    says which vectors a corpus's generation holds. Reuse, restoration and
    rebuild all verify against this anchor; a generation without one is
    recomputed under the admitted build, never trusted from disk.
    """

    manifest_logical_hash: Sha256Hex
    payload_sha256: Sha256Hex


class HybridIndexEvictionMarker(DomainModel):
    """Record the generation and payload removed by an approved cleanup plan.

    What an approved cleanup left where an index was: which generation it
    removed and under which plan. It names the payload that was committed so
    the inventory can account for it; restoration never takes its commitment
    from here but from the caller's sealed anchor.
    """

    index_schema: HybridIndexSchema
    index_id: Sha256Hex
    manifest_logical_hash: Sha256Hex
    vector_commitment: HybridVectorCommitment | None
    database_path: NonEmptyString
    eviction_plan_hash: Sha256Hex
    evicted_at: UtcDatetime

    @model_validator(mode="after")
    def validate_marker(self) -> Self:
        """Require a vector commitment exactly when the marker names the current schema.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: Index schema and vector-commitment presence are inconsistent.
        """
        committed = self.index_schema == HYBRID_INDEX_SCHEMA_VERSION
        if committed != (self.vector_commitment is not None):
            raise ValueError("eviction marker commitment does not match its index schema")
        return self


class HybridKnowledgeRetrievalRequest(DomainModel):
    """Seal a hybrid query against exact source, recipe and index commitments.

    Current-schema requests carry both corpus and manifest hashes. Namespace/access filters are
    nonempty and sorted; optional document groups and ordinal scopes restrict retrieval within the
    admitted source. as_of controls source availability.
    """

    mode: KnowledgeRetrievalMode
    index_schema: HybridIndexSchema
    index_id: Sha256Hex
    snapshot_id: Uuid4
    snapshot_logical_hash: Sha256Hex
    index_spec_logical_hash: Sha256Hex
    corpus_logical_hash: Sha256Hex | None
    """`knowledge-hybrid-v4`: the corpus the request's snapshot must resolve to."""
    index_manifest_logical_hash: Sha256Hex | None
    """`knowledge-hybrid-v4`: the caller's sealed expectation of the generation
    manifest, so the database's own copy is never the authority for itself."""
    query: NonEmptyString
    allowed_namespaces: tuple[KnowledgeNamespace, ...]
    allowed_access_classes: tuple[KnowledgeAccessClass, ...]
    as_of: UtcDatetime
    top_k: TopK
    document_groups: tuple[tuple[StableKnowledgeId, ...], ...] = ()
    """Groups of document ids that each receive their own `top_k` of the final
    reranked order (an issuer's filings, for instance), so a group whose best
    passage ranks below another group's twentieth is still returned. Documents
    in no group form one trailing group. Empty: one global `top_k`, as before.
    Ranks stay global: the hits are the reranked order, cut per group."""
    reranker_depth_per_document: RerankerDepth | None = None
    """How many of each document's best fused candidates the cross-encoder
    reads for this request; `None` is the index spec's depth. A query-time
    choice, so it lives on the request and not in the index identity: the
    same generation serves a program that reads deeper. Bounded by the
    candidate window (`candidate_depth_per_document`), which is what the
    channels can supply per document."""
    scope: tuple[tuple[StableKnowledgeId, int, int], ...] = ()
    """The source ranges this request may consider, as `(document id, first
    ordinal, last ordinal)` chunk ranges (inclusive; a document's whole
    range is `(id, 0, last)`), sorted and non-overlapping within a
    document. Enforced in every channel's SQL filter -- before any
    channel's candidate limit, before fusion and before the cross-encoder
    reads a pair -- so a candidate outside the scope is never scored; the
    `document_groups` cut of the final order is not a scope. Empty: no
    scope, every admitted chunk as before, so earlier requests keep their
    identity. Part of the request's logical hash, which names one search
    and seals nothing durable."""
    logical_hash: Sha256Hex

    @field_validator("allowed_namespaces", "allowed_access_classes")
    @classmethod
    def validate_filters(cls, values: tuple[object, ...]) -> tuple[object, ...]:
        """Require nonempty sorted unique namespace or access filters.

        Args:
            values: One request filter tuple.

        Returns:
            The unchanged filter tuple.

        Raises:
            ValueError: The filter is empty, unordered or duplicated.
        """
        if not values:
            raise ValueError("retrieval filters must not be empty")
        return _require_sorted_unique(values, "retrieval filters")

    @field_validator("document_groups")
    @classmethod
    def validate_groups(cls, groups: tuple[tuple[str, ...], ...]) -> tuple[tuple[str, ...], ...]:
        """Require at most 64 ordered disjoint groups of unique document identifiers.

        Args:
            groups: Groups ordered by their first identifier, with sorted unique members.

        Returns:
            The unchanged document groups.

        Raises:
            ValueError: A group is empty/unordered, groups overlap or ordering/count limits fail.
        """
        if len(groups) > 64:
            raise ValueError("retrieval document groups exceed 64")
        seen: set[str] = set()
        for group in groups:
            if not group:
                raise ValueError("retrieval document group must not be empty")
            if tuple(sorted(group, key=str)) != group or len(set(group)) != len(group):
                raise ValueError("retrieval document group must be sorted and unique")
            if seen & set(group):
                raise ValueError("retrieval document groups must be disjoint")
            seen.update(group)
        if groups != tuple(sorted(groups, key=lambda group: str(group[0]))):
            raise ValueError("retrieval document groups must be sorted by their first id")
        return groups

    @field_validator("scope")
    @classmethod
    def validate_scope(
        cls, scope: tuple[tuple[str, int, int], ...]
    ) -> tuple[tuple[str, int, int], ...]:
        """Require bounded ordered nonoverlapping document ordinal ranges.

        Args:
            scope: Up to 4096 document, first-ordinal and last-ordinal triples.

        Returns:
            The unchanged scope tuple.

        Raises:
            ValueError: Scope ordering, uniqueness, count or nonnegative inclusive ranges are
                invalid.
        """
        if len(scope) > 4096:
            raise ValueError("retrieval scope exceeds 4096 ranges")
        if tuple(sorted(scope)) != scope:
            raise ValueError("retrieval scope must be sorted")
        previous: tuple[str, int, int] | None = None
        for entry in scope:
            document_id, first, last = entry
            if first < 0 or last < first:
                raise ValueError("retrieval scope range is invalid")
            if previous is not None and previous[0] == document_id and first <= previous[2]:
                raise ValueError("retrieval scope ranges overlap")
            previous = entry
        return scope

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        """Require hybrid mode, bounded query, schema-consistent commitments and request hash.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: Mode, query length, corpus/manifest commitment presence or logical_hash are
                invalid.
        """
        if self.mode is not KnowledgeRetrievalMode.STANDARD_HYBRID:
            raise ValueError("Hybrid v2 request requires STANDARD_HYBRID mode")
        if len(self.query) > 512:
            raise ValueError("query exceeds hybrid-v2 character limit")
        committed = self.index_schema == HYBRID_INDEX_SCHEMA_VERSION
        if committed != (
            self.corpus_logical_hash is not None and self.index_manifest_logical_hash is not None
        ):
            raise ValueError("hybrid request corpus commitments do not match its index schema")
        if self.logical_hash != _logical_hash(self, exclude={"logical_hash"}):
            raise ValueError("hybrid request logical_hash is inconsistent")
        return self

    @classmethod
    def create(
        cls,
        *,
        index_id: str,
        snapshot_id: UUID,
        snapshot_logical_hash: str,
        index_spec_logical_hash: str,
        query: str,
        allowed_namespaces: tuple[KnowledgeNamespace, ...],
        allowed_access_classes: tuple[KnowledgeAccessClass, ...],
        as_of: datetime,
        top_k: int,
        index_schema: HybridIndexSchema = HYBRID_INDEX_SCHEMA_VERSION,
        corpus_logical_hash: str | None = None,
        index_manifest_logical_hash: str | None = None,
        document_groups: tuple[tuple[str, ...], ...] = (),
        reranker_depth_per_document: int | None = None,
        scope: tuple[tuple[str, int, int], ...] = (),
    ) -> HybridKnowledgeRetrievalRequest:
        """Seal a standard-hybrid request from explicit query and source commitments.

        Args:
            index_id: Exact index identifier.
            snapshot_id: Admitted knowledge snapshot identifier.
            snapshot_logical_hash: Exact admitted snapshot identity.
            index_spec_logical_hash: Exact numerical recipe identity.
            query: Raw query text within the hybrid character limit.
            allowed_namespaces: Nonempty sorted unique namespace filter.
            allowed_access_classes: Nonempty sorted unique access filter.
            as_of: UTC source-availability cutoff.
            top_k: Requested bounded hit count.
            index_schema: Current or supported legacy index schema.
            corpus_logical_hash: Corpus commitment required by the current schema.
            index_manifest_logical_hash: Generation commitment required by the current schema.
            document_groups: Optional ordered disjoint document groups.
            reranker_depth_per_document: Optional request-specific reranking depth.
            scope: Optional inclusive document ordinal ranges.

        Returns:
            Validated request whose logical_hash binds the complete declared payload.

        Raises:
            pydantic.ValidationError: Values or request consistency violate the contract.
        """
        payload = {
            "schema_version": "1",
            "mode": KnowledgeRetrievalMode.STANDARD_HYBRID,
            "index_schema": index_schema,
            "index_id": index_id,
            "snapshot_id": snapshot_id,
            "snapshot_logical_hash": snapshot_logical_hash,
            "index_spec_logical_hash": index_spec_logical_hash,
            "corpus_logical_hash": corpus_logical_hash,
            "index_manifest_logical_hash": index_manifest_logical_hash,
            "query": query,
            "allowed_namespaces": allowed_namespaces,
            "allowed_access_classes": allowed_access_classes,
            "as_of": as_of,
            "top_k": top_k,
            "document_groups": document_groups,
            "reranker_depth_per_document": reranker_depth_per_document,
            "scope": scope,
        }
        return cls.model_validate(
            {**payload, "logical_hash": sha256_hex(canonical_json_bytes(payload))}
        )


class HybridKnowledgeRetrievalTrace(DomainModel):
    """Seal query compilation, queried channels, candidate counts and score semantics.

    The dense channel must be represented. Sorted unique terms/channels and the exact trace hash
    make the retrieval decisions inspectable without interpreting scores as confidence.
    """

    normalized_query: NonEmptyString
    compiled_terms: tuple[NonEmptyString, ...]
    channels_queried: tuple[RetrievalChannel, ...]
    candidate_count: NonNegativeInt
    score_semantics: RetrievalScoreSemantics
    reranked_count: NonNegativeInt | None = None
    """How many (query, passage) pairs the cross-encoder scored for this
    request: the fused candidates read to the per-document depth -- what a
    pair budget is spent on. Absent on traces sealed before it was
    recorded."""
    logical_hash: Sha256Hex

    @field_validator("compiled_terms", "channels_queried")
    @classmethod
    def validate_sorted_unique(cls, values: tuple[object, ...]) -> tuple[object, ...]:
        """Require sorted unique compiled terms or queried channels.

        Args:
            values: Trace tuple requiring canonical ordering.

        Returns:
            The unchanged trace tuple.

        Raises:
            ValueError: Values are unordered or duplicated.
        """
        return _require_sorted_unique(values, "hybrid trace tuple")

    @model_validator(mode="after")
    def validate_trace(self) -> Self:
        """Require dense-channel evidence, fixed score semantics and the exact trace hash.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: The dense channel is absent, score semantics are invalid or logical_hash is
                inconsistent.
        """
        if RetrievalChannel.DENSE not in self.channels_queried:
            raise ValueError("Hybrid v2 trace must include DENSE")
        if self.score_semantics not in (
            RetrievalScoreSemantics.rrf_rank(),
            RetrievalScoreSemantics.cross_encoder_logit(),
        ):
            raise ValueError("Hybrid v2 trace score semantics are not a fixed retrieval score")
        if self.logical_hash != _logical_hash(self, exclude={"logical_hash"}):
            raise ValueError("hybrid trace logical_hash is inconsistent")
        return self


class HybridKnowledgeRetrievalResult(DomainModel):
    """Seal ranked cited hits and their hybrid retrieval trace.

    EMPTY has no hits; RESOLVED has hits ranked consecutively from one. logical_hash binds the
    request, status, hits and trace.
    """

    request: HybridKnowledgeRetrievalRequest
    status: RetrievalStatus
    hits: tuple[KnowledgeRetrievalHit, ...]
    trace: HybridKnowledgeRetrievalTrace
    logical_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        """Require status-consistent hits, contiguous ranks and the exact result hash.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: EMPTY/RESOLVED disagrees with hit presence, ranks are not consecutive from
                one or hash is invalid.
        """
        if self.status is RetrievalStatus.EMPTY and self.hits:
            raise ValueError("EMPTY result must not contain hits")
        if self.status is RetrievalStatus.RESOLVED and not self.hits:
            raise ValueError("RESOLVED result requires hits")
        if tuple(hit.rank for hit in self.hits) != tuple(range(1, len(self.hits) + 1)):
            raise ValueError("hit ranks must be contiguous")
        if self.logical_hash != _logical_hash(self, exclude={"logical_hash"}):
            raise ValueError("hybrid result logical_hash is inconsistent")
        return self


__all__ = [
    "BGE_MODEL_ARTIFACT_SHA256",
    "BGE_MODEL_ID",
    "BGE_MODEL_REVISION",
    "BGE_TOKENIZER_SHA256",
    "CUDA_VERSION",
    "ENCODER_RUNTIME_ONNX_CPU",
    "ENCODER_RUNTIME_TORCH_CUDA",
    "HYBRID_INDEX_SCHEMA_VERSION",
    "LEGACY_HYBRID_INDEX_SCHEMA_VERSION",
    "MODEL_ARTIFACT_SHA256",
    "MODEL_ID",
    "MODEL_REVISION",
    "PAIR_SCORE_BLOCK_SUFFIX",
    "PAIR_SCORE_ROOT",
    "QWEN3_EMBEDDING_MODEL_ID",
    "QWEN3_EMBEDDING_MODEL_REVISION",
    "QWEN3_EMBEDDING_PACK",
    "QWEN3_EMBEDDING_PACK_SHA256",
    "QWEN3_RERANKER_MODEL_ID",
    "QWEN3_RERANKER_MODEL_REVISION",
    "QWEN3_RERANKER_PACK",
    "QWEN3_RERANKER_PACK_SHA256",
    "RECIPE_BGE_SMALL_CPU",
    "RECIPE_MINILM_CPU",
    "RECIPE_QWEN3_ENCODER_GPU",
    "RECIPE_QWEN3_GPU",
    "RECIPE_QWEN3_RERANKER_GPU",
    "RERANKER_RUNTIME_FASTEMBED_CPU",
    "RERANKER_RUNTIME_TORCH_CUDA",
    "SUPPORTED_RECIPES",
    "TOKENIZER_SHA256",
    "TORCH_VERSION",
    "TRANSFORMERS_VERSION",
    "VECTOR_BLOCK_SIDECAR_SUFFIX",
    "VECTOR_PAYLOAD_ROOT",
    "EmbeddingInputPolicy",
    "HybridCapabilityReport",
    "HybridGenerationAnchor",
    "HybridIndexEvictionMarker",
    "HybridIndexSchema",
    "HybridIndexSpec",
    "HybridKnowledgeIndexManifest",
    "HybridKnowledgeRetrievalRequest",
    "HybridKnowledgeRetrievalResult",
    "HybridKnowledgeRetrievalTrace",
    "HybridV3KnowledgeIndexManifest",
    "HybridVectorBlock",
    "HybridVectorBlockSidecar",
    "HybridVectorCommitment",
    "KnowledgeRetrievalMode",
    "PairScoreAdmission",
    "RetrievalScoreSemantics",
    "SealedPairScoreBlock",
    "SemanticPackStatus",
    "embedding_context_hash",
    "recipe_payload",
    "reranker_context_hash",
    "safe_intra_op_threads",
    "vector_asset_key",
]
