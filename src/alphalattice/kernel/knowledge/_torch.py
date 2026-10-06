"""The one supported local GPU runtime for the enhanced retrieval recipes.

PyTorch with CUDA and the transformers loader, used for exactly two things:
the Qwen3 embedding model (last-token pooling over a left-padded batch, L2
normalised, the retrieval instruction on queries only) and the Qwen3 reranker
(a causal language model asked whether a document meets a query, scored as
the log-odds of its "yes" against its "no" at the last position -- a logit,
uncalibrated, comparable within one model, as the cross-encoder's is).

The runtime is optional and isolated (`.venv-gpu`, `config/requirements-gpu.lock`);
the CPU recipes never import it. A missing runtime, a missing GPU, a version
other than the spec's or a pack whose files differ from the pinned closure is
a typed refusal naming what is missing, never a silent fallback to another
model. Every file the loader reads is hashed before it is read; the pack
directory may hold nothing else; `trust_remote_code` is never enabled and
`local_files_only` always is. Weights are loaded in float32 so a passage's
vector is reproducible for the same batch composition, which the incremental
block owner guarantees by embedding one revision's chunks at a time.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib
import importlib.metadata
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Lock
from typing import Any

from alphalattice.kernel.knowledge._embeddings import (
    PairScoreCache,
    score_with_cache,
)
from alphalattice.kernel.knowledge.hybrid_contracts import (
    ENCODER_RUNTIME_TORCH_CUDA,
    QWEN3_EMBEDDING_PACK,
    QWEN3_RERANKER_PACK,
    RERANKER_RUNTIME_TORCH_CUDA,
    HybridCapabilityReport,
    HybridIndexSpec,
    KnowledgeRetrievalMode,
    SemanticPackStatus,
)
from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError
from alphalattice.kernel.shared_kernel.domain.serialization import canonical_json_bytes, sha256_hex

QWEN3_RERANKER_TASK = "Given a web search query, retrieve relevant passages that answer the query"
_RERANKER_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _error(message: str, *, cause: Exception | None = None) -> KnowledgeRetrievalError:
    error = KnowledgeRetrievalError(
        message, code="retrieval.semantic_pack_unavailable", retryable=True
    )
    if cause is not None:
        error.__cause__ = cause
    return error


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise _error("torch model pack is unavailable", cause=exc) from exc
    return digest.hexdigest()


def verify_pack(root: Path, manifest: tuple[tuple[str, str], ...], expected_hash: str) -> None:
    """Every consumed file present with its pinned hash, nothing unadmitted,
    the manifest's own hash equal to the spec's."""

    try:
        present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    except OSError as exc:
        raise _error("torch model pack is unavailable", cause=exc) from exc
    expected = {name for name, _digest in manifest}
    missing = sorted(expected - present)
    if missing:
        raise _error(f"torch model pack is missing a consumed artifact: {missing[0]}")
    unexpected = sorted(present - expected)
    if unexpected:
        raise _error(f"torch model pack holds an unadmitted artifact: {unexpected[0]}")
    observed: dict[str, str] = {}
    for name, digest in manifest:
        observed[name] = _hash_file(root / name)
        if observed[name] != digest:
            raise _error(f"torch pack artifact hash differs from the index spec: {name}")
    if sha256_hex(canonical_json_bytes(dict(sorted(observed.items())))) != expected_hash:
        raise _error("torch pack manifest differs from the index spec")


def torch_encoder_pack_paths(root: Path) -> tuple[Path, ...]:
    return (root, *(root / name for name, _digest in QWEN3_EMBEDDING_PACK))


def torch_reranker_pack_paths(root: Path) -> tuple[Path, ...]:
    return (root, *(root / name for name, _digest in QWEN3_RERANKER_PACK))


def runtime_for(
    torch_version: str | None, transformers_version: str | None, cuda_version: str | None
) -> tuple[Any, Any]:
    """Import the runtime and prove it is the locked one on a usable GPU."""

    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise _error(
            "the GPU retrieval runtime is not installed (scripts/create_gpu_environment.py)",
            cause=exc,
        ) from exc
    observed = (
        importlib.metadata.version("torch"),
        importlib.metadata.version("transformers"),
        str(torch.version.cuda),
    )
    if observed != (torch_version, transformers_version, cuda_version):
        raise _error(
            f"GPU runtime {observed} differs from the index spec "
            f"({torch_version}, {transformers_version}, {cuda_version})"
        )
    if not torch.cuda.is_available():
        raise _error(
            "no CUDA device is available for the GPU retrieval recipe "
            f"(torch {observed[0]}, CUDA {observed[2]}; the device is absent, lost or unsupported)"
        )
    return torch, transformers


def probe_torch_encoder(model_root: Path, spec: HybridIndexSpec) -> HybridCapabilityReport:
    """The torch encoder's readiness: pack, runtime, device, and the sqlite
    vector extension the index still needs."""

    from alphalattice.kernel.knowledge._embeddings import (
        encoder_root,
        load_vector_extension,
    )

    root = encoder_root(model_root, spec)
    assert spec.encoder_pack_sha256 is not None
    verify_pack(root, QWEN3_EMBEDDING_PACK, spec.encoder_pack_sha256)
    runtime_for(
        spec.encoder_torch_version, spec.encoder_transformers_version, spec.encoder_cuda_version
    )
    import sqlite3
    from contextlib import closing

    try:
        vector_version = importlib.metadata.version("sqlite-vec")
        with closing(sqlite3.connect(":memory:")) as connection:
            loaded = load_vector_extension(connection)
    except (ImportError, sqlite3.Error) as exc:
        raise _error("sqlite-vec extension is unavailable", cause=exc) from exc
    if vector_version != spec.vector_extension_version or loaded != spec.vector_extension_version:
        raise _error("loaded sqlite-vec extension differs from the index spec")
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
        "cpu_execution_provider_only": False,
        "details": tuple(
            sorted(
                (
                    "model:verified",
                    "tokenizer:verified",
                    "encoder-pack:verified",
                    "torch:verified",
                    "sqlite-vec:verified",
                    "cuda-device:verified",
                )
            )
        ),
        "encoder_runtime": ENCODER_RUNTIME_TORCH_CUDA,
    }
    if spec.reranker_runtime == RERANKER_RUNTIME_TORCH_CUDA:
        payload["reranker_runtime"] = RERANKER_RUNTIME_TORCH_CUDA
    return HybridCapabilityReport.model_validate(
        {**payload, "logical_hash": sha256_hex(canonical_json_bytes(payload))}
    )


class TorchEmbeddingAdapter:
    """One verified Qwen3 embedding session on the GPU, serialised."""

    def __init__(self, root: Path, spec: HybridIndexSpec) -> None:
        from alphalattice.kernel.knowledge._embeddings import (
            MODEL_WORK,
            prepare_embedding_input,
        )

        self.spec = spec
        self._prepare = prepare_embedding_input
        self._work = MODEL_WORK
        self._lock = Lock()
        self._closed = False
        assert spec.encoder_pack_sha256 is not None
        verify_pack(root, QWEN3_EMBEDDING_PACK, spec.encoder_pack_sha256)
        self._torch, transformers = runtime_for(
            spec.encoder_torch_version,
            spec.encoder_transformers_version,
            spec.encoder_cuda_version,
        )
        try:
            MODEL_WORK.embedding_model_loads += 1
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(root), local_files_only=True, padding_side="left"
            )
            self._model = (
                transformers.AutoModel.from_pretrained(
                    str(root), local_files_only=True, dtype=self._torch.float32
                )
                .cuda()
                .eval()
            )
        except Exception as exc:
            self.close()
            raise _error("torch embedding model could not be loaded", cause=exc) from exc

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._model = None
        self._tokenizer = None
        gc.collect()
        # Releasing a dead context must not raise.
        with contextlib.suppress(Exception):
            self._torch.cuda.empty_cache()

    def embed_passages(
        self, texts: Sequence[str], *, cancelled: Callable[[], bool] | None = None
    ) -> tuple[tuple[float, ...], ...]:
        return self._embed(texts, is_query=False, cancelled=cancelled)

    def embed_query(
        self, text: str, *, cancelled: Callable[[], bool] | None = None
    ) -> tuple[float, ...]:
        return self._embed((text,), is_query=True, cancelled=cancelled)[0]

    def _embed(
        self, texts: Sequence[str], *, is_query: bool, cancelled: Callable[[], bool] | None
    ) -> tuple[tuple[float, ...], ...]:
        if self._closed:
            raise RuntimeError("embedding adapter is closed")
        if not texts:
            return ()
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent embedding calls are forbidden")
        work = self._work
        if is_query:
            work.query_embeddings += len(texts)
        else:
            work.passage_calls_attempted += 1
            work.passages_attempted += len(texts)
        torch = self._torch
        try:
            assert self._model is not None and self._tokenizer is not None
            prepared = [
                self._prepare(text, policy=self.spec.embedding_input_policy, is_query=is_query)
                for text in texts
            ]
            rows: list[tuple[float, ...]] = []
            for start in range(0, len(prepared), self.spec.batch_size):
                if cancelled is not None and cancelled():
                    raise CancelledError("embedding cancelled")
                batch = self._tokenizer(
                    prepared[start : start + self.spec.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.spec.maximum_tokens,
                    return_tensors="pt",
                ).to("cuda")
                with torch.no_grad():
                    hidden = self._model(**batch).last_hidden_state
                pooled = hidden[:, -1]  # left padding: the last position is the last token
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1).float().cpu()
                if (
                    normalized.ndim != 2
                    or normalized.shape[1] != self.spec.embedding_dimension
                    or not torch.isfinite(normalized).all()
                ):
                    raise _error("torch runtime returned invalid embeddings")
                rows.extend(tuple(float(value) for value in row) for row in normalized.tolist())
            if not is_query:
                work.passage_calls_completed += 1
                work.passages_completed += len(texts)
            return tuple(rows)
        except CancelledError:
            if not is_query:
                work.passage_calls_cancelled += 1
            self.close()
            raise
        except KnowledgeRetrievalError:
            if not is_query:
                work.passage_calls_failed += 1
            self.close()
            raise
        except Exception as exc:
            if not is_query:
                work.passage_calls_failed += 1
            self.close()
            raise _error("torch embedding inference failed", cause=exc) from exc
        finally:
            self._lock.release()


class TorchCausalRerankerAdapter:
    """Score (query, passage) pairs with the verified Qwen3 reranker on the GPU."""

    def __init__(self, root: Path, spec: HybridIndexSpec) -> None:
        from alphalattice.kernel.knowledge._embeddings import MODEL_WORK

        self._work = MODEL_WORK
        verify_pack(root, QWEN3_RERANKER_PACK, spec.reranker_pack_sha256)
        self._torch, transformers = runtime_for(
            spec.reranker_torch_version,
            spec.reranker_transformers_version,
            spec.reranker_cuda_version,
        )
        self._decimals = spec.reranker_score_decimal_places
        self._batch_size = spec.batch_size
        self._maximum_tokens = spec.reranker_maximum_tokens or 1024
        self._cache = PairScoreCache()
        try:
            MODEL_WORK.reranker_model_loads += 1
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(root), local_files_only=True, padding_side="left"
            )
            self._model = (
                transformers.AutoModelForCausalLM.from_pretrained(
                    str(root), local_files_only=True, dtype=self._torch.float32
                )
                .cuda()
                .eval()
            )
            self._yes = int(self._tokenizer.convert_tokens_to_ids("yes"))
            self._no = int(self._tokenizer.convert_tokens_to_ids("no"))
        except Exception as exc:
            self.close()
            raise _error("torch reranker model could not be loaded", cause=exc) from exc

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        """One log-odds per passage, quantised so ordering does not ride on float noise."""

        if not passages:
            return ()
        self._work.reranker_calls += 1
        self._work.reranker_pairs += len(passages)
        torch = self._torch

        def infer(pending: Sequence[str]) -> Sequence[float]:
            texts = [
                f"{_RERANKER_PREFIX}<Instruct>: {QWEN3_RERANKER_TASK}\n<Query>: {query}\n"
                f"<Document>: {passage}{_RERANKER_SUFFIX}"
                for passage in pending
            ]
            try:
                assert self._model is not None and self._tokenizer is not None
                scores: list[float] = []
                for start in range(0, len(texts), self._batch_size):
                    batch = self._tokenizer(
                        texts[start : start + self._batch_size],
                        padding=True,
                        truncation=True,
                        max_length=self._maximum_tokens,
                        return_tensors="pt",
                    ).to("cuda")
                    with torch.no_grad():
                        logits = self._model(**batch).logits[:, -1, :]
                    odds = (logits[:, self._yes] - logits[:, self._no]).float().cpu().tolist()
                    scores.extend(round(float(value), self._decimals) for value in odds)
                return scores
            except Exception as exc:
                raise KnowledgeRetrievalError(
                    "torch reranker inference failed",
                    code="retrieval.index_unavailable",
                    retryable=False,
                ) from exc

        return tuple(
            float(value) for value in score_with_cache(self._cache, query, passages, infer)
        )


__all__ = [
    "QWEN3_RERANKER_TASK",
    "TorchCausalRerankerAdapter",
    "TorchEmbeddingAdapter",
    "probe_torch_encoder",
    "torch_encoder_pack_paths",
    "torch_reranker_pack_paths",
    "verify_pack",
]
