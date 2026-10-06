"""Cross-encoder reranking over a locally verified model pack.

Fusion decides which passages are considered and a cross-encoder decides their
order, because it reads the query and the passage together rather than each
alone. Whether that ordering is better for a given corpus is a measured
question, not a property of the architecture.

The pack is pinned by content. `verify_reranker_pack` hashes **every file the
installed loader consumes** -- `fastembed.common.preprocessor_utils.load_tokenizer`
reads `config.json` for `pad_token_id`, `tokenizer_config.json` for
`model_max_length`/`max_length` and `pad_token`, `special_tokens_map.json` for
the added tokens, and `tokenizer.json` for the tokenizer itself, before
`onnx/model.onnx` is ever run -- and refuses a directory holding any other
file, so nothing outside the manifest can reach the loader. The manifest's own
hash is in the index spec, so altering it moves the index identity rather than
the inference.

That closure is the point. Verifying only the two largest files admitted a pack
whose `tokenizer_config.json` said `model_max_length: 16`: identical
verification output, and one pair's score moved from 0.5765 to 8.3125 because
the passage was truncated to sixteen tokens.

FastEmbed owns tokenization, pair construction, truncation and output
semantics; `tokenizers` applies the truncation and padding, so both versions
are bound too. This module owns the identity check and the deterministic
ordering, and nothing else.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
import struct
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, cast
from uuid import uuid4

import numpy as np

from alphalattice.kernel.knowledge._embeddings import (
    MODEL_WORK,
    VERIFIED_PACKS,
    PackLease,
    PairScoreCache,
    canary_mismatch,
    score_with_cache,
)
from alphalattice.kernel.knowledge.hybrid_contracts import (
    PAIR_SCORE_BLOCK_SUFFIX,
    PAIR_SCORE_ROOT,
    RERANKER_CANARIES,
    RERANKER_PACK,
    RERANKER_RUNTIME_TORCH_CUDA,
    RETRIEVAL_CANARY_PASSAGE,
    RETRIEVAL_CANARY_QUERY,
    HybridIndexSpec,
    PairScoreAdmission,
    SealedPairScoreBlock,
    reranker_context_hash,
    safe_intra_op_threads,
)
from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError
from alphalattice.kernel.shared_kernel.domain.serialization import canonical_json_bytes, sha256_hex

RERANKER_PACK_DIRECTORY = "reranker"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _error(message: str, *, cause: Exception | None = None) -> KnowledgeRetrievalError:
    error = KnowledgeRetrievalError(message, code="retrieval.index_unavailable", retryable=False)
    if cause is not None:
        error.__cause__ = cause
    return error


def _runtime_versions() -> tuple[str, str]:
    """The two libraries whose versions decide tokenization and inference.

    A seam, so a test that stands in for the model bytes can stand in for these
    two strings as well without replacing any of the structural verification.
    """

    try:
        importlib.import_module("fastembed.rerank.cross_encoder")
        return (
            importlib.metadata.version("fastembed"),
            importlib.metadata.version("tokenizers"),
        )
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise _error(
            "the reranking runtime is not installed in this environment", cause=exc
        ) from exc


def reranker_pack_root(model_root: Path) -> Path:
    return model_root / RERANKER_PACK_DIRECTORY


def reranker_pack_hash(files: dict[str, str]) -> str:
    """One identity for the whole consumed closure, ordered so it is stable."""

    return str(sha256_hex(canonical_json_bytes(dict(sorted(files.items())))))


def verify_reranker_pack(model_root: Path, spec: HybridIndexSpec) -> tuple[str, ...]:
    """Prove the pack on disk is the one the spec admits; return probe details."""

    if spec.reranker_runtime == RERANKER_RUNTIME_TORCH_CUDA:
        from alphalattice.kernel.knowledge._torch import (
            QWEN3_RERANKER_PACK,
            runtime_for,
            verify_pack,
        )

        verify_pack(reranker_pack_root(model_root), QWEN3_RERANKER_PACK, spec.reranker_pack_sha256)
        # The pack alone is not readiness: the locked runtime and a usable
        # device are proved here, so a lost or absent GPU is named now, not
        # at the first pair scored.
        runtime_for(
            spec.reranker_torch_version,
            spec.reranker_transformers_version,
            spec.reranker_cuda_version,
        )
        return (
            "cuda-device:verified",
            f"reranker-pack:{len(QWEN3_RERANKER_PACK)}-artifacts-verified",
            "reranker-torch:verified",
        )
    pack = reranker_pack_root(model_root)
    try:
        present = {path.relative_to(pack).as_posix() for path in pack.rglob("*") if path.is_file()}
    except OSError as exc:
        raise _error("reranker model pack is unavailable", cause=exc) from exc
    expected = {name for name, _digest in RERANKER_PACK}
    missing = sorted(expected - present)
    if missing:
        raise _error(f"reranker model pack is missing a consumed artifact: {missing[0]}")
    unexpected = sorted(present - expected)
    if unexpected:
        raise _error(f"reranker model pack holds an unadmitted artifact: {unexpected[0]}")

    observed: dict[str, str] = {}
    for name, digest in RERANKER_PACK:
        try:
            observed[name] = _hash_file(pack / name)
        except OSError as exc:
            raise _error("reranker model pack is unavailable", cause=exc) from exc
        if observed[name] != digest:
            raise _error(f"reranker pack artifact hash differs from the index spec: {name}")
    if reranker_pack_hash(observed) != spec.reranker_pack_sha256:
        raise _error("reranker pack manifest differs from the index spec")

    fastembed_version, tokenizers_version = _runtime_versions()
    if fastembed_version != spec.reranker_runtime_version:
        raise _error("fastembed version differs from the index spec")
    if tokenizers_version != spec.reranker_tokenizer_runtime_version:
        raise _error("tokenizers version differs from the index spec")
    return (
        f"reranker-pack:{len(observed)}-artifacts-verified",
        "fastembed:verified",
        "tokenizers:verified",
    )


def reranker_pack_paths(model_root: Path, spec: HybridIndexSpec) -> tuple[Path, ...]:
    """The pack directory (its listing is part of the proof) and every consumed file."""

    pack = reranker_pack_root(model_root)
    if spec.reranker_runtime == RERANKER_RUNTIME_TORCH_CUDA:
        from alphalattice.kernel.knowledge._torch import torch_reranker_pack_paths

        return torch_reranker_pack_paths(pack)
    return (pack, *(pack / name for name, _digest in RERANKER_PACK))


def _load_reranker(model_root: Path, spec: HybridIndexSpec, threads: int) -> Any:
    if spec.reranker_runtime == RERANKER_RUNTIME_TORCH_CUDA:
        from alphalattice.kernel.knowledge._torch import TorchCausalRerankerAdapter

        return TorchCausalRerankerAdapter(reranker_pack_root(model_root), spec)
    return LocalCrossEncoderAdapter(model_root, spec, threads=threads)


class SharedRerankerAdapter:
    """One consumer's lease on the process's held cross-encoder; calls serialized
    on the session's shared lock; closing releases the lease."""

    def __init__(self, lease: PackLease) -> None:
        self._lease = lease

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        with self._lease.lock:
            return cast(tuple[float, ...], self._lease.value.score(query, passages))

    def close(self) -> None:
        self._lease.release()


def reranker_session_threads(spec: HybridIndexSpec) -> int:
    """The threads a cross-encoder session of `spec` loads with: the CPU
    budget's share when it has a sealed canary, else the safe default."""

    if reranker_context_hash(spec.without_execution()) in RERANKER_CANARIES:
        return VERIFIED_PACKS.threads()
    return safe_intra_op_threads()


def shared_reranker_adapter(model_root: Path, spec: HybridIndexSpec) -> SharedRerankerAdapter:
    """A lease on the held cross-encoder for this pack, loaded once per process."""

    threads = reranker_session_threads(spec)
    lease = VERIFIED_PACKS.lease(
        ("reranker", model_root.resolve().as_posix(), spec.logical_hash),
        consumed=reranker_pack_paths(model_root, spec),
        load=lambda: _load_reranker(model_root, spec, threads),
        usable=lambda encoder: getattr(encoder, "threads", threads) == threads,
    )
    if lease.reused:
        MODEL_WORK.reranker_model_reuses += 1
    return SharedRerankerAdapter(lease)


PAIR_SCORE_STORE_CAP_BYTES = 64 * 1024 * 1024
"""The most a workspace holds of sealed pair scores under one reranker
context: at 36 bytes a pair, 1.86 million pairs, some six hundred first
responses at the residual pair budget. A full store seals nothing more and
says so; nothing is evicted here."""
PAIR_SCORE_BLOCK_MINIMUM = 16
"""Fewer new pairs than this at a session's close stay in the process's
cache rather than becoming a block of their own."""
PAIR_SCORE_FLUSH_PAIRS = 4096
"""New pairs are sealed in blocks of at most this many, so one block is
one session's worth of scoring at the residual pair budget."""
PAIR_SCORE_SAMPLE = 4
"""Served scores a session re-derives with the model once: a diagnostic
of runtime drift (a pack or runtime that no longer scores as the sealed
context did), never a certificate of the values it did not examine --
the authority to serve a block is the sealed commitment that admits it."""
_BLOCK_MAGIC = b"ALPHALATTICE-PAIR-SCORES/1\n"
_ENTRY = np.dtype([("key", "S32"), ("score", "<i4")])


def _store_error(message: str, code: str) -> KnowledgeRetrievalError:
    return KnowledgeRetrievalError(message, code=code, retryable=False)


@dataclass(frozen=True, slots=True)
class _ScoreBlock:
    name: str
    keys: Any
    scores: Any
    byte_length: int


class PairScoreStore:
    """Sealed cross-encoder pair scores of one workspace under one reranker
    context, read before the model is asked and sealed after it answers.

    A pair's score is a function of the reranker context, the query text and
    the passage text alone (`PairScoreCache`), so a score sealed by one
    session serves every later session of every later process: an unchanged
    or repacked refresh, a restart, another unit holding the same filing --
    the twenty questions ask for the same pairs, which are content-addressed
    whatever unit they sit in. Measured cold, the cross-encoder was 23.7 s of
    a 53.9 s unit selection, all of it pairs a previous process had scored.

    A block is an immutable file named by the sha256 of its bytes: the magic,
    a header naming the context and the rounding, then the entries sorted by
    key (the sha256 of query and passage) with the score as an integer at the
    context's rounding, exact. A block's name is a lookup key, not the
    authority to serve it: the store loads only the blocks the caller's
    admission names -- the evidence runtime names those its sealed
    commitments bind, each commitment named by a sealed receipt -- and a
    block on disk under any other name is counted (`unanchored`) and never
    consulted. An admitted block is verified whole at open (hash, header,
    order) and refused by name when it does not verify
    (`retrieval.pair_score_block_tampered`): a damaged anchored artifact is
    never a miss. An admitted block missing from disk is a missing proof
    (`missing`; the pair is scored again). A pair two admitted blocks score
    differently is a conflict, refused by name
    (`retrieval.pair_score_conflict`), never an arbitrary choice. Each
    session also re-derives a sample of the served scores with the model
    (`PAIR_SCORE_SAMPLE`): a diagnostic of runtime drift, no certificate.
    New blocks are sealed at close under the workspace's storage admission
    (refused: `unsealed`, the pairs stay served from this process's cache)
    and under the store's own cap (`PAIR_SCORE_STORE_CAP_BYTES`; full: no
    more blocks); the bytes live under the managed evidence root and count.
    """

    def __init__(
        self,
        root: Path,
        context: str,
        decimals: int,
        cap_bytes: int,
        *,
        admission: PairScoreAdmission,
    ) -> None:
        self.root = root
        self.context = context
        self.decimals = decimals
        self.cap_bytes = cap_bytes
        self.admission = admission
        self._blocks: tuple[_ScoreBlock, ...] = ()
        self._pending: dict[bytes, int] = {}
        self._lock = Lock()
        self.held_bytes = 0
        self.hits = 0
        self.misses = 0
        self.sealed_pairs = 0
        self.refused_full = 0
        self.unanchored = 0
        """Blocks on disk under this context that no commitment admits."""
        self.missing = 0
        """Admitted blocks not on disk: missing proofs, scored again."""
        self.unsealed = 0
        """Pairs the storage admission refused to seal."""

    @property
    def block_count(self) -> int:
        return len(self._blocks)

    @property
    def pending_pairs(self) -> int:
        return len(self._pending)

    # ---- opening: every block verified.
    @staticmethod
    def block_paths(root: Path) -> tuple[Path, ...]:
        if not root.is_dir():
            return ()
        return tuple(sorted(p for p in root.iterdir() if p.name.endswith(PAIR_SCORE_BLOCK_SUFFIX)))

    def load(self) -> None:
        """Load the admitted blocks: every one verified whole; unadmitted files
        counted and left; admitted names absent counted as missing; a pair
        two admitted blocks disagree on refused."""

        on_disk = {path.name: path for path in self.block_paths(self.root)}
        self.unanchored = sum(1 for name in on_disk if name not in self.admission.admitted)
        blocks = []
        held = 0
        for name in sorted(self.admission.admitted):
            path = on_disk.get(name)
            if path is None:
                self.missing += 1
                continue
            content = path.read_bytes()
            blocks.append(self._verify(name, content))
            held += len(content)
        self._refuse_conflicts(blocks)
        self._blocks = tuple(blocks)
        self.held_bytes = held

    @staticmethod
    def _refuse_conflicts(blocks: Sequence[_ScoreBlock]) -> None:
        if len(blocks) < 2:
            return
        keys = np.concatenate([block.keys for block in blocks])
        scores = np.concatenate([block.scores for block in blocks])
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        sorted_scores = scores[order]
        same = sorted_keys[1:] == sorted_keys[:-1]
        if bool(np.any(same & (sorted_scores[1:] != sorted_scores[:-1]))):
            raise _store_error(
                "two admitted pair score blocks score one pair differently",
                code="retrieval.pair_score_conflict",
            )

    def _verify(self, name: str, content: bytes) -> _ScoreBlock:
        if sha256_hex(content) + PAIR_SCORE_BLOCK_SUFFIX != name:
            raise _store_error(
                f"pair score block {name} does not hash to its name",
                code="retrieval.pair_score_block_tampered",
            )
        if not content.startswith(_BLOCK_MAGIC):
            raise _store_error(
                f"pair score block {name} is not a pair score block",
                code="retrieval.pair_score_block_tampered",
            )
        header_end = content.index(b"\n", len(_BLOCK_MAGIC))
        header = content[len(_BLOCK_MAGIC) : header_end].decode("ascii").split(" ")
        entries = content[header_end + 1 :]
        if (
            len(header) != 3
            or header[0] != self.context
            or header[1] != str(self.decimals)
            or len(entries) % _ENTRY.itemsize
            or header[2] != str(len(entries) // _ENTRY.itemsize)
        ):
            raise _store_error(
                f"pair score block {name} is not this reranker context's",
                code="retrieval.pair_score_block_tampered",
            )
        table = np.frombuffer(entries, dtype=_ENTRY)
        keys = table["key"]
        if len(keys) == 0 or not bool(np.all(keys[1:] > keys[:-1])):
            raise _store_error(
                f"pair score block {name} is not sorted by key",
                code="retrieval.pair_score_block_tampered",
            )
        return _ScoreBlock(name=name, keys=keys, scores=table["score"], byte_length=len(content))

    # ---- reading.
    def lookup(self, keys: Sequence[bytes]) -> list[int | None]:
        """The sealed score of each key, in order; None where none is sealed."""

        found: list[int | None] = [None] * len(keys)
        if not keys:
            return found
        with self._lock:
            for index, key in enumerate(keys):
                value = self._pending.get(key)
                if value is not None:
                    found[index] = value
            blocks = self._blocks
        wanted = np.array(keys, dtype="S32")
        for block in blocks:
            positions = np.searchsorted(block.keys, wanted)
            inside = positions < len(block.keys)
            positions[~inside] = 0
            matched = inside & (block.keys[positions] == wanted)
            for index in np.flatnonzero(matched):
                if found[index] is None:
                    found[index] = int(block.scores[positions[index]])
        hits = sum(1 for value in found if value is not None)
        self.hits += hits
        self.misses += len(keys) - hits
        MODEL_WORK.reranker_pair_store_hits += hits
        return found

    # ---- sealing.
    def record(self, scored: Sequence[tuple[bytes, int]]) -> None:
        """Scores the model just produced, to be sealed at the flush."""

        with self._lock:
            for key, value in scored:
                self._pending.setdefault(key, value)

    def flush(self) -> tuple[SealedPairScoreBlock, ...]:
        """Seal the pending pairs in blocks of at most `PAIR_SCORE_FLUSH_PAIRS`
        under the caller's storage admission; the blocks sealed, for the
        caller to commit. Fewer than `PAIR_SCORE_BLOCK_MINIMUM` pairs stay
        unsealed; a full store or a refused admission seals nothing and
        counts it; nothing is placed that was not admitted first. A block
        sealed here is served by this store for the rest of the session and
        by a later one only once committed."""

        sealed: list[SealedPairScoreBlock] = []
        with self._lock:
            while len(self._pending) >= PAIR_SCORE_BLOCK_MINIMUM:
                batch = sorted(self._pending.items())[:PAIR_SCORE_FLUSH_PAIRS]
                content = self._encode(batch)
                for key, _value in batch:
                    del self._pending[key]
                if self.held_bytes + len(content) > self.cap_bytes:
                    self.refused_full += len(batch)
                    continue
                if self.admission.admit is not None:
                    try:
                        self.admission.admit(len(content))
                    except Exception:
                        self.unsealed += len(batch)
                        continue
                name = sha256_hex(content) + PAIR_SCORE_BLOCK_SUFFIX
                self._place(name, content)
                self._blocks = (*self._blocks, self._verify(name, content))
                self.held_bytes += len(content)
                self.sealed_pairs += len(batch)
                sealed.append(
                    SealedPairScoreBlock(name=name, pair_count=len(batch), byte_length=len(content))
                )
                MODEL_WORK.reranker_pair_store_writes += len(batch)
        return tuple(sealed)

    def _encode(self, batch: Sequence[tuple[bytes, int]]) -> bytes:
        if any(not -(2**31) <= value < 2**31 for _key, value in batch):
            raise _store_error(
                "a pair score is outside the sealed integer representation",
                code="retrieval.pair_score_out_of_range",
            )
        table = np.empty(len(batch), dtype=_ENTRY)
        table["key"] = [key for key, _value in batch]
        table["score"] = [value for _key, value in batch]
        header = f"{self.context} {self.decimals} {len(batch)}\n".encode("ascii")
        return _BLOCK_MAGIC + header + table.tobytes()

    def _place(self, name: str, content: bytes) -> None:
        target = self.root / name
        if target.is_file():
            if target.read_bytes() == content:
                return
            raise _store_error(
                f"pair score block {name} holds other bytes",
                code="retrieval.pair_score_block_tampered",
            )
        self.root.mkdir(parents=True, exist_ok=True)
        staging = self.root / f".{uuid4()}.tmp"
        try:
            with staging.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if staging.read_bytes() != content:
                raise _store_error(
                    f"pair score block {name} did not read back as written",
                    code="retrieval.pair_score_block_tampered",
                )
            os.replace(staging, target)
        finally:
            with suppress(OSError):
                staging.unlink(missing_ok=True)

    # ---- serving a reranker call.
    def serve(
        self,
        query: str,
        passages: Sequence[str],
        score_fresh: Callable[[Sequence[str]], Sequence[float]],
        *,
        verify_sample: bool,
    ) -> tuple[tuple[float, ...], int]:
        """Every pair's score in order, and how many served scores were
        re-derived: admitted scores from the store, the rest from
        `score_fresh` in one call and recorded for sealing. With
        `verify_sample`, up to `PAIR_SCORE_SAMPLE` of the served scores are
        re-derived by the model and a difference refuses the store
        (`retrieval.pair_score_runtime_drift`): the runtime no longer scores
        as the sealed context did. The sample diagnoses drift; the
        admission is what authorizes the values."""

        if not passages:
            return (), 0
        scale = 10**self.decimals
        keys = [bytes.fromhex(PairScoreCache.key(query, passage)) for passage in passages]
        found = self.lookup(keys)
        missing = [index for index, value in enumerate(found) if value is None]
        if missing:
            fresh = score_fresh([passages[index] for index in missing])
            scored = []
            for index, value in zip(missing, fresh, strict=True):
                quantized = round(float(value) * scale)
                found[index] = quantized
                scored.append((keys[index], quantized))
            self.record(scored)
        fresh_indexes = set(missing)
        served = [index for index in range(len(passages)) if index not in fresh_indexes]
        sampled = 0
        if verify_sample and served:
            sample = sorted(served, key=lambda index: keys[index])[:PAIR_SCORE_SAMPLE]
            derived = score_fresh([passages[index] for index in sample])
            for index, value in zip(sample, derived, strict=True):
                if round(float(value) * scale) != found[index]:
                    raise _store_error(
                        "a sealed pair score differs from the model's: runtime drift",
                        code="retrieval.pair_score_runtime_drift",
                    )
            sampled = len(sample)
        return tuple(cast(int, value) / scale for value in found), sampled


def pair_score_store(
    workspace_root: Path, spec: HybridIndexSpec, admission: PairScoreAdmission
) -> PairScoreStore:
    """A reader's store for this workspace and reranker context: the blocks
    the admission names, loaded and verified for this reader alone -- no
    process-wide hold, nothing trusted across sessions but the sealed
    commitments the caller read them from."""

    context = reranker_context_hash(spec)
    root = workspace_root / PAIR_SCORE_ROOT / context
    store = PairScoreStore(
        root,
        context,
        spec.reranker_score_decimal_places,
        PAIR_SCORE_STORE_CAP_BYTES,
        admission=admission,
    )
    store.load()
    return store


class LocalCrossEncoderAdapter:
    """Score (query, passage) pairs with the verified local cross-encoder."""

    def __init__(
        self, model_root: Path, spec: HybridIndexSpec, *, threads: int | None = None
    ) -> None:
        verify_reranker_pack(model_root, spec)
        module = importlib.import_module("fastembed.rerank.cross_encoder")
        MODEL_WORK.reranker_model_loads += 1
        self.threads = threads or safe_intra_op_threads()
        self._encoder: Any = module.TextCrossEncoder(
            spec.reranker_model_id,
            specific_model_path=str(reranker_pack_root(model_root)),
            local_files_only=True,
            threads=self.threads,
        )
        self._decimals = spec.reranker_score_decimal_places
        self._batch_size = spec.batch_size
        self._cache = PairScoreCache()
        sealed = RERANKER_CANARIES.get(reranker_context_hash(spec.without_execution()))
        if sealed is not None:
            # The canary pair's raw score, before rounding: a session that moves
            # a number on this machine at these threads is refused by name.
            raw = list(
                self._encoder.rerank(
                    RETRIEVAL_CANARY_QUERY, [RETRIEVAL_CANARY_PASSAGE], batch_size=self._batch_size
                )
            )
            if not raw or (hashlib.sha256(struct.pack("<f", float(raw[0]))).hexdigest() != sealed):
                raise canary_mismatch("cross-encoder", self.threads)

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...]:
        """One score per passage, quantized so ordering does not ride on float
        noise; pairs scored before under this session are answered from its
        bounded pair cache, the rest in one inference."""

        if not passages:
            return ()
        MODEL_WORK.reranker_calls += 1
        MODEL_WORK.reranker_pairs += len(passages)

        def infer(batch: Sequence[str]) -> Sequence[float]:
            try:
                raw = self._encoder.rerank(query, list(batch), batch_size=self._batch_size)
                return [round(float(value), self._decimals) for value in raw]
            except (OSError, RuntimeError, ValueError) as exc:
                raise _error("cross-encoder inference failed", cause=exc) from exc

        return score_with_cache(self._cache, query, passages, infer)


__all__ = [
    "PAIR_SCORE_BLOCK_MINIMUM",
    "PAIR_SCORE_FLUSH_PAIRS",
    "PAIR_SCORE_SAMPLE",
    "PAIR_SCORE_STORE_CAP_BYTES",
    "RERANKER_PACK_DIRECTORY",
    "LocalCrossEncoderAdapter",
    "PairScoreStore",
    "_runtime_versions",
    "pair_score_store",
    "reranker_pack_hash",
    "reranker_pack_root",
    "verify_reranker_pack",
]
