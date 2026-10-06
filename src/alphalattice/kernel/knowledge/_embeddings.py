"""Private, network-independent embedding adapter for local Hybrid v2 retrieval."""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import sqlite3
import struct
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import CancelledError
from contextlib import AbstractContextManager, closing, contextmanager
from pathlib import Path
from threading import Condition, Lock, get_ident, local
from time import monotonic
from typing import Any, cast

from alphalattice.kernel.knowledge.hybrid_contracts import (
    EMBEDDING_CANARIES,
    RECIPE_MINILM_CPU,
    RETRIEVAL_CANARY_TEXT,
    EmbeddingInputPolicy,
    HybridCapabilityReport,
    HybridIndexSpec,
    KnowledgeRetrievalMode,
    SemanticPackStatus,
    embedding_context_hash,
    safe_intra_op_threads,
)
from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError
from alphalattice.kernel.shared_kernel.domain.serialization import canonical_json_bytes, sha256_hex

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
QWEN3_QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise _semantic_error("semantic model artifact is unavailable", cause=exc) from exc
    return digest.hexdigest()


def _semantic_error(
    message: str,
    *,
    cause: Exception | None = None,
) -> KnowledgeRetrievalError:
    error = KnowledgeRetrievalError(
        message,
        code="retrieval.semantic_pack_unavailable",
        retryable=True,
    )
    if cause is not None:
        error.__cause__ = cause
    return error


def prepare_embedding_input(
    text: str,
    *,
    policy: EmbeddingInputPolicy,
    is_query: bool,
) -> str:
    """Apply the exact adapter-owned input policy without rewriting user text."""

    if policy is EmbeddingInputPolicy.E5_ASYMMETRIC_V1:
        return f"{'query' if is_query else 'passage'}: {text}"
    if policy is EmbeddingInputPolicy.IDENTITY_V1:
        return text
    if policy is EmbeddingInputPolicy.BGE_QUERY_INSTRUCTION_V1:
        return f"{BGE_QUERY_INSTRUCTION}{text}" if is_query else text
    if policy is EmbeddingInputPolicy.QWEN3_RETRIEVAL_INSTRUCT_V1:
        return f"{QWEN3_QUERY_INSTRUCTION}{text}" if is_query else text
    raise ValueError(f"unsupported embedding input policy: {policy}")


def encoder_root(model_root: Path, spec: HybridIndexSpec) -> Path:
    """Where the encoder's files sit under a pack root: at the root for the
    retained `hybrid-v2-minilm` layout, under `encoder/` for every later
    recipe (the reranker sits under `reranker/` for all of them)."""

    return model_root if spec.policy_id == RECIPE_MINILM_CPU else model_root / "encoder"


def encoder_tokenizer_path(root: Path, spec: HybridIndexSpec) -> Path:
    if spec.token_normalization == "XLM_ROBERTA_SENTENCEPIECE_V1":
        return root / "sentencepiece.bpe.model"
    return root / "tokenizer.json"


def _capability_report(
    *,
    spec: HybridIndexSpec,
    status: SemanticPackStatus,
    details: tuple[str, ...],
    onnx_runtime_version: str | None,
    sentencepiece_version: str | None,
    vector_extension_version: str | None,
    cpu_only: bool,
) -> HybridCapabilityReport:
    payload = {
        "schema_version": "1",
        "mode": KnowledgeRetrievalMode.STANDARD_HYBRID,
        "status": status,
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "model_artifact_sha256": spec.model_artifact_sha256,
        "tokenizer_sha256": spec.tokenizer_sha256,
        "onnx_runtime_version": onnx_runtime_version,
        "sentencepiece_version": sentencepiece_version,
        "vector_extension_version": vector_extension_version,
        "cpu_execution_provider_only": cpu_only,
        "details": tuple(sorted(details)),
    }
    return HybridCapabilityReport.model_validate(
        {**payload, "logical_hash": sha256_hex(canonical_json_bytes(payload))}
    )


def probe_semantic_pack(
    model_root: Path,
    spec: HybridIndexSpec,
) -> HybridCapabilityReport:
    """Verify the exact local ONNX model and native runtime inventory without
    networking (the torch recipes are probed by their own module; the kernel
    owner dispatches)."""

    details: list[str] = []
    try:
        root = encoder_root(model_root, spec)
        model_path = root / "onnx" / "model.onnx"
        tokenizer_path = encoder_tokenizer_path(root, spec)
        if _hash_file(model_path) != spec.model_artifact_sha256:
            raise _semantic_error("semantic model artifact hash differs from the index spec")
        if _hash_file(tokenizer_path) != spec.tokenizer_sha256:
            raise _semantic_error("semantic tokenizer hash differs from the index spec")
        details.extend(("model:verified", "tokenizer:verified"))

        ort = importlib.import_module("onnxruntime")
        sentencepiece = importlib.import_module("sentencepiece")
        sqlite_vec = importlib.import_module("sqlite_vec")
        onnx_version = importlib.metadata.version("onnxruntime")
        sentencepiece_version = importlib.metadata.version("sentencepiece")
        vector_version = importlib.metadata.version("sqlite-vec")
        if (
            onnx_version != spec.onnx_runtime_version
            or sentencepiece_version != spec.sentencepiece_version
            or vector_version != spec.vector_extension_version
        ):
            raise _semantic_error("semantic runtime version differs from the index spec")
        if "CPUExecutionProvider" not in ort.get_available_providers():
            raise _semantic_error("ONNX CPUExecutionProvider is unavailable")
        if spec.token_normalization == "BERT_WORDPIECE_TOKENIZERS_V1":
            importlib.import_module("tokenizers")
        details.extend(
            (
                "onnxruntime:verified",
                "sqlite-vec:verified",
                "sentencepiece:verified",
            )
        )

        with closing(sqlite3.connect(":memory:")) as connection:
            connection.enable_load_extension(True)
            sqlite_vec.load(connection)
            connection.enable_load_extension(False)
            loaded_version = str(connection.execute("SELECT vec_version()").fetchone()[0])
        if loaded_version.lstrip("v") != spec.vector_extension_version:
            raise _semantic_error("loaded sqlite-vec extension differs from the index spec")
        del sentencepiece
        return _capability_report(
            spec=spec,
            status=SemanticPackStatus.READY,
            details=tuple(details),
            onnx_runtime_version=onnx_version,
            sentencepiece_version=sentencepiece_version,
            vector_extension_version=vector_version,
            cpu_only=True,
        )
    except KnowledgeRetrievalError:
        raise
    except (ImportError, OSError, RuntimeError, sqlite3.Error) as exc:
        raise _semantic_error("semantic runtime or model pack is unavailable", cause=exc) from exc


class ModelWorkLedger:
    """What the local models were asked to do, counted when asked, not when a
    build was published: passage calls and passages attempted, completed,
    cancelled or failed; query embeddings; cross-encoder pairs; model loads
    and the loads a held session answered instead. A count that moves only
    on success cannot represent the work a failed or cancelled build cost,
    which is the number a person pays for."""

    __slots__ = (
        "embedding_model_loads",
        "embedding_model_reuses",
        "pack_probe_reuses",
        "passage_calls_attempted",
        "passage_calls_cancelled",
        "passage_calls_completed",
        "passage_calls_failed",
        "passages_attempted",
        "passages_completed",
        "query_embeddings",
        "reranker_calls",
        "reranker_model_loads",
        "reranker_model_reuses",
        "reranker_pair_cache_hits",
        "reranker_pair_store_hits",
        "reranker_pair_store_writes",
        "reranker_pairs",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.embedding_model_loads = 0
        self.embedding_model_reuses = 0
        self.pack_probe_reuses = 0
        self.reranker_model_loads = 0
        self.reranker_model_reuses = 0
        self.passage_calls_attempted = 0
        self.passage_calls_completed = 0
        self.passage_calls_cancelled = 0
        self.passage_calls_failed = 0
        self.passages_attempted = 0
        self.passages_completed = 0
        self.query_embeddings = 0
        self.reranker_calls = 0
        self.reranker_pairs = 0
        self.reranker_pair_cache_hits = 0
        self.reranker_pair_store_hits = 0
        self.reranker_pair_store_writes = 0

    def snapshot(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__slots__}


MODEL_WORK = ModelWorkLedger()
"""The process's ledger; the retrieval owner reports it beside its own counts."""


class PairScoreCache:
    """Scores of exact (query, passage) pairs under one reranker, held in the
    process and bounded.

    A pair's score is a function of the model, the query text and the
    passage text alone -- measured, a batch of eight and a batch of
    thirty-two return the same scores to the last digit -- so a score once
    computed serves any later request for the same pair: the program's
    twenty questions over an unchanged generation ask for the same pairs,
    and a repeat costs the candidate SQL rather than the cross-encoder. The
    key is the hash of both texts; the value is the rounded score the
    adapter returned. Nothing else is remembered: no eligibility, no result
    list, no candidate set, which are regenerated every time. Cancelled or
    failed work leaves nothing behind. The oldest entries leave first.
    """

    __slots__ = ("_entries", "_limit")

    def __init__(self, limit: int = 250_000) -> None:
        self._entries: dict[str, float] = {}
        self._limit = limit

    @staticmethod
    def key(query: str, passage: str) -> str:
        return hashlib.sha256(
            len(query).to_bytes(4, "big") + query.encode("utf-8") + passage.encode("utf-8")
        ).hexdigest()

    def get(self, key: str) -> float | None:
        value = self._entries.get(key)
        if value is not None:
            # Refresh: the dict keeps insertion order, so the oldest is first.
            del self._entries[key]
            self._entries[key] = value
        return value

    def put(self, key: str, value: float) -> None:
        if key in self._entries:
            del self._entries[key]
        elif len(self._entries) >= self._limit:
            del self._entries[next(iter(self._entries))]
        self._entries[key] = value

    def __len__(self) -> int:
        return len(self._entries)


def score_with_cache(
    cache: PairScoreCache,
    query: str,
    passages: Sequence[str],
    score_batch: Callable[[Sequence[str]], Sequence[float]],
) -> tuple[float, ...]:
    """Score the pairs not yet known in one call; return every score in order."""

    keys = [PairScoreCache.key(query, passage) for passage in passages]
    scores: list[float | None] = [cache.get(key) for key in keys]
    MODEL_WORK.reranker_pair_cache_hits += sum(1 for value in scores if value is not None)
    missing = [index for index, value in enumerate(scores) if value is None]
    if missing:
        fresh = score_batch([passages[index] for index in missing])
        for index, value in zip(missing, fresh, strict=True):
            scores[index] = float(value)
            cache.put(keys[index], float(value))
    return tuple(float(value) for value in scores if value is not None)


PackFacts = tuple[tuple[str, int, int, int, int], ...]
"""Per consumed path: posix name, size, mtime_ns, ctime_ns, inode -- what a
pack proven by content looked like on disk at that proof."""


def pack_facts(paths: Sequence[Path]) -> PackFacts:
    """The lstat facts of every consumed path, in order; an unreadable path
    records -1 so the next proof runs and names the failure."""

    facts = []
    for path in paths:
        try:
            stat = path.lstat()
            facts.append(
                (path.as_posix(), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
            )
        except OSError:
            facts.append((path.as_posix(), -1, -1, -1, -1))
    return tuple(facts)


class _Held:
    """One held value: its proof facts, the lock its users share, the leases
    outstanding on it, and whether it has been retired (replaced or found
    unusable) -- a retired value closes when its last lease ends."""

    __slots__ = ("facts", "leases", "lock", "retired", "value")

    def __init__(self, facts: PackFacts, value: Any) -> None:
        self.facts = facts
        self.value = value
        self.lock = Lock()
        self.leases = 0
        self.retired = False


def _close_value(value: Any) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()


_INFERENCE = local()
"""Per thread: what a caller gives up while it waits for and runs a held model, the
threads its sessions are loaded with, and what it waits for before its first model call."""

FOREGROUND_WAIT_SECONDS = 30.0
"""The longest model work waits for a foreground reader (`VerifiedPackRegistry.foreground`)."""


@contextmanager
def before_first_inference(gate: Callable[[], None] | None) -> Iterator[None]:
    """`gate()` runs once, at this thread's first model call in the scope, after the
    caller's lock is given up (`yielding_during_inference`): what comes before the
    model -- reading, cutting, loading a session -- runs meanwhile. A call that keeps
    the caller's lock does not wait there; the next one that yields does."""

    previous = getattr(_INFERENCE, "gate", None)
    _INFERENCE.gate = gate
    try:
        yield
    finally:
        _INFERENCE.gate = previous


@contextmanager
def yielding_during_inference(
    release: Callable[[], AbstractContextManager[object]],
) -> Iterator[None]:
    """While this thread waits for and runs a held model, `release()` is entered.

    A caller that holds a lock over shared state -- the evidence runtime's
    writer, one preparation stage at a time -- gives it to other threads for
    the one span in which it touches nothing shared, and takes it back before
    its next step. Only the model call yields, on the thread that entered
    this scope; a scope nested inside restores the outer one when it ends.
    """

    previous = getattr(_INFERENCE, "release", None)
    _INFERENCE.release = release
    try:
        yield
    finally:
        _INFERENCE.release = previous


@contextmanager
def observing_passages(observer: Callable[[int], None] | None) -> Iterator[None]:
    """While this thread embeds passages, `observer(n)` hears each batch of `n`
    as it is embedded: a build's progress, counted where the model runs."""

    previous = getattr(_INFERENCE, "passages", None)
    _INFERENCE.passages = observer
    try:
        yield
    finally:
        _INFERENCE.passages = previous


def withhold_inference_yield() -> None:
    """For the rest of this thread's yielding scope, a model call keeps the
    caller's lock: the caller now holds another lock (the workspace's) that a
    thread it would let in could ask for."""

    _INFERENCE.release = None


class _InferenceHold:
    """A held value's lock, taken for one call; the caller's yield given up first."""

    __slots__ = ("_lock", "_yielded")

    def __init__(self, lock: Lock) -> None:
        self._lock = lock
        self._yielded: AbstractContextManager[object] | None = None

    def __enter__(self) -> None:
        release = getattr(_INFERENCE, "release", None)
        if release is not None:
            self._yielded = release()
            self._yielded.__enter__()
            gate = getattr(_INFERENCE, "gate", None)
            if gate is not None:
                _INFERENCE.gate = None
                gate()
        VERIFIED_PACKS.yield_to_foreground()
        self._lock.acquire()

    def __exit__(self, *_exc: object) -> None:
        self._lock.release()
        yielded, self._yielded = self._yielded, None
        if yielded is not None:
            yielded.__exit__(None, None, None)


class PackLease:
    """One consumer's hold on a held value.

    The value stays open at least until every lease on it is released; calls
    through the lease take the held value's lock, so consumers of one
    session wait for each other rather than meeting the session's own
    single-caller refusal, and a caller that yields during inference gives
    up its own lock while it waits and runs. Releasing is idempotent.
    """

    __slots__ = ("_held", "_registry", "_released", "reused")

    def __init__(self, registry: VerifiedPackRegistry, held: _Held, *, reused: bool) -> None:
        self._registry = registry
        self._held = held
        self._released = False
        self.reused = reused

    @property
    def value(self) -> Any:
        return self._held.value

    @property
    def lock(self) -> AbstractContextManager[None]:
        return _InferenceHold(self._held.lock)

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._registry._release_lease(self._held)


class VerifiedPackRegistry:
    """What this process has proven about a model pack, held for reuse.

    Proving a pack means hashing every file its loaders consume -- a
    475 MB model is two seconds and half a gigabyte of reads -- and loading
    a session means reading it again. Measured on the real-copy flow: every
    build, every retrieval session and every capability check proved and
    loaded the packs afresh, 2.6 GB of reads and six seconds per issuer unit
    that embedded nothing. A proof made once in this process is kept with
    the lstat facts (size, mtime, ctime, inode) of every consumed path and of
    the pack directories; a later request with the same facts reuses the
    proof and the session it loaded, and any difference -- a replaced,
    grown, touched, added or removed file -- makes the request prove again.
    Up to `sessions_per_pack` sessions per (root, spec) are held -- one
    unless a Host that runs work items at once raises it -- so a process
    holds one pack's sessions, not a history of them: a lease takes a held
    session no other lease holds, loads another while fewer than that many
    are held, and otherwise shares the least-leased one, its calls waiting
    for each other. Each is the same pinned pack under the same session
    options, so which one serves a call changes no number.

    A session is handed out only while it is usable: the embedding adapter
    closes itself on cancellation and on inference failure, and a request
    after that loads a fresh session instead of receiving the dead one. A
    replaced or unusable session is retired: it closes when the last lease on
    it is released, never under a consumer still using it.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._entries: dict[tuple[str, ...], _Held] = {}
        self._pools: dict[tuple[str, ...], list[_Held]] = {}
        self.sessions_per_pack = 1
        self.session_threads: int | None = None
        """The intra-op threads a model session this process loads is given: the
        CPU budget's share the running preparations set (`use_threads`), else the
        machine's safe default; a thread inside `threads_for` loads its own. Never
        sealed; each session proves its canary."""
        self._foreground = Condition(Lock())
        self._foreground_threads: dict[int, int] = {}

    def use_threads(self, threads: int | None) -> None:
        """Load sessions with `threads` from now on (None: the safe default); a
        held session of another count is retired when next asked for, closing
        with its last lease."""

        with self._lock:
            self.session_threads = None if threads is None else max(1, int(threads))

    @contextmanager
    def threads_for(self, threads: int | None) -> Iterator[None]:
        """Sessions this thread leases inside the scope are of `threads` intra-op
        threads: its unit's share of the CPU budget. Held sessions of another width
        stay held for their own callers (`lease`'s `fits`)."""

        previous = getattr(_INFERENCE, "threads", None)
        _INFERENCE.threads = None if threads is None else max(1, int(threads))
        try:
            yield
        finally:
            _INFERENCE.threads = previous

    def threads(self) -> int:
        return (
            getattr(_INFERENCE, "threads", None) or self.session_threads or safe_intra_op_threads()
        )

    @contextmanager
    def foreground(self) -> Iterator[None]:
        """While a reader the lead waits on runs (an Analyst's bundle), model work on
        other threads waits at its next call or batch, at most
        `FOREGROUND_WAIT_SECONDS`: the reader has the machine (final close-out, F2)."""

        me = get_ident()
        with self._foreground:
            self._foreground_threads[me] = self._foreground_threads.get(me, 0) + 1
        try:
            yield
        finally:
            with self._foreground:
                left = self._foreground_threads.pop(me) - 1
                if left:
                    self._foreground_threads[me] = left
                self._foreground.notify_all()

    def yield_to_foreground(self) -> None:
        """Wait while another thread reads in the foreground, at most the bound."""

        if not self._foreground_threads:
            return
        me, deadline = get_ident(), monotonic() + FOREGROUND_WAIT_SECONDS
        with self._foreground:
            while self._foreground_threads and me not in self._foreground_threads:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return
                self._foreground.wait(remaining)

    def hold_up_to(self, sessions: int) -> None:
        """Hold up to `sessions` leased values per key (never fewer than before):
        the work items a Host runs at once, each with a session of its own."""

        with self._lock:
            self.sessions_per_pack = max(self.sessions_per_pack, int(sessions))

    def resolve(
        self,
        key: tuple[str, ...],
        *,
        consumed: Sequence[Path],
        load: Callable[[], Any],
        release: Callable[[Any], None] | None = None,
    ) -> tuple[Any, bool]:
        """The held value for `key` while `consumed` is unchanged, else `load()`'s.

        For values without a lifecycle (a capability report). Returns
        `(value, reused)`; a replaced value is handed to `release`.
        """

        facts = pack_facts(consumed)
        with self._lock:
            held = self._entries.get(key)
            if held is not None and held.facts == facts:
                return held.value, True
            value = load()
            self._entries[key] = _Held(facts, value)
        if held is not None and release is not None:
            release(held.value)
        return value, False

    def lease(
        self,
        key: tuple[str, ...],
        *,
        consumed: Sequence[Path],
        load: Callable[[], Any],
        usable: Callable[[Any], bool] | None = None,
        fits: Callable[[Any], bool] | None = None,
    ) -> PackLease:
        """A lease on the held value for `key`, loaded when there is none
        usable: the facts changed, or `usable` says the held value is dead.
        With `fits`, only a held value it admits is leased (a session of the
        caller's width); the others stay held for their own callers, an idle one
        giving its place when the pool is full.

        Loading happens under the registry lock, so two requests for one
        pack load it once. A previous value is retired, not closed here.
        """

        facts = pack_facts(consumed)
        with self._lock:
            pool = self._pools.setdefault(key, [])
            for held in list(pool):
                if held.facts != facts or (usable is not None and not usable(held.value)):
                    self._retire_locked(key, held)
            fitting = [held for held in pool if fits is None or fits(held.value)]
            idle = next((held for held in fitting if held.leases == 0), None)
            reused = idle is not None or len(fitting) >= self.sessions_per_pack
            if idle is not None:
                held = idle
            elif reused:
                held = min(fitting, key=lambda value: value.leases)
            else:
                for spare in [value for value in pool if value not in fitting]:
                    if len(pool) < self.sessions_per_pack:
                        break
                    if spare.leases == 0:
                        self._retire_locked(key, spare)
                held = _Held(facts, load())
                pool.append(held)
            held.leases += 1
        return PackLease(self, held, reused=reused)

    def _retire_locked(self, key: tuple[str, ...], held: _Held) -> None:
        held.retired = True
        pool = self._pools.get(key, [])
        if held in pool:
            pool.remove(held)
        if held.leases == 0:
            _close_value(held.value)

    def _release_lease(self, held: _Held) -> None:
        with self._lock:
            held.leases -= 1
            close = held.retired and held.leases == 0
        if close:
            _close_value(held.value)

    def release(self) -> None:
        """Drop every held value (process end, or a test that must reload)."""

        with self._lock:
            entries = [
                *self._entries.values(),
                *(held for pool in self._pools.values() for held in pool),
            ]
            self._entries.clear()
            self._pools.clear()
        for held in entries:
            held.retired = True
            if held.leases == 0:
                _close_value(held.value)


VERIFIED_PACKS = VerifiedPackRegistry()
"""The process's held pack proofs and sessions."""


class SharedEmbeddingAdapter:
    """One consumer's lease on the process's held embedding session.

    Closing the lease releases it; the session stays the registry's until it
    is retired and unleased. Calls take the session's shared lock, so two
    consumers wait for each other. A session that closed itself (cancelled
    or failed work) refuses the consumer that killed it and is not leased
    again.
    """

    def __init__(self, lease: PackLease) -> None:
        self._lease = lease
        self.spec = lease.value.spec

    @property
    def closed(self) -> bool:
        return self._lease.released

    def close(self) -> None:
        self._lease.release()

    def __enter__(self) -> SharedEmbeddingAdapter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def embed_passages(
        self,
        texts: Sequence[str],
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        with self._lease.lock:
            return cast(
                tuple[tuple[float, ...], ...],
                self._lease.value.embed_passages(texts, cancelled=cancelled),
            )

    def embed_query(
        self,
        text: str,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[float, ...]:
        with self._lease.lock:
            return cast(tuple[float, ...], self._lease.value.embed_query(text, cancelled=cancelled))


def semantic_pack_paths(model_root: Path, spec: HybridIndexSpec) -> tuple[Path, ...]:
    """The paths the ONNX embedding loader consumes, and the directory holding
    the model."""

    root = encoder_root(model_root, spec)
    return (root / "onnx", root / "onnx" / "model.onnx", encoder_tokenizer_path(root, spec))


def shared_embedding_adapter(
    model_root: Path,
    spec: HybridIndexSpec,
    *,
    consumed: Sequence[Path] | None = None,
    load: Callable[[], Any] | None = None,
) -> SharedEmbeddingAdapter:
    """A lease on the held session for this pack, loaded once per process
    and again after the held one closed itself. The ONNX session by default;
    a runtime with its own loader passes the paths it consumes and its loader."""

    threads = embedding_session_threads(spec)
    lease = VERIFIED_PACKS.lease(
        ("embedding", model_root.resolve().as_posix(), spec.logical_hash),
        consumed=semantic_pack_paths(model_root, spec) if consumed is None else consumed,
        load=(
            (lambda: LocalOnnxEmbeddingAdapter(model_root, spec, threads=threads))
            if load is None
            else load
        ),
        usable=lambda adapter: not adapter.closed,
        fits=lambda adapter: getattr(adapter, "threads", threads) == threads,
    )
    if lease.reused:
        MODEL_WORK.embedding_model_reuses += 1
    return SharedEmbeddingAdapter(lease)


def load_vector_extension(connection: sqlite3.Connection) -> str:
    sqlite_vec = importlib.import_module("sqlite_vec")
    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
    except (AttributeError, ImportError, OSError, sqlite3.Error) as exc:
        raise _semantic_error("sqlite-vec extension is unavailable", cause=exc) from exc
    finally:
        connection.enable_load_extension(False)
    row = connection.execute("SELECT vec_version()").fetchone()
    if row is None:
        raise _semantic_error("sqlite-vec version probe returned no row")
    return str(row[0]).lstrip("v")


def serialize_vector(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def deserialize_vector(blob: bytes, *, dimension: int) -> tuple[float, ...]:
    if len(blob) != dimension * 4:
        raise ValueError("serialized vector has an invalid byte length")
    return tuple(item[0] for item in struct.iter_unpack("<f", blob))


def create_session_options(runtime: Any, threads: int) -> Any:
    """One sequential session on `threads` intra-op threads, no spinning: how
    fast, never what -- the session's canary proves the numbers."""

    options = runtime.SessionOptions()
    options.execution_mode = runtime.ExecutionMode.ORT_SEQUENTIAL
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = threads
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options


def embedding_session_threads(spec: HybridIndexSpec) -> int:
    """The threads an embedding session of `spec` loads with: the CPU budget's
    share when its encoder has a sealed canary to prove the numbers on, else
    the safe default every spec sealed before F1 named."""

    if embedding_context_hash(spec.without_execution()) in EMBEDDING_CANARIES:
        return VERIFIED_PACKS.threads()
    return safe_intra_op_threads()


def canary_mismatch(kind: str, threads: int) -> KnowledgeRetrievalError:
    """A session whose canary is not the sealed one: refused by name, with the
    threads it ran on -- a machine or runtime that moves the numbers."""

    return KnowledgeRetrievalError(
        f"the {kind} session's canary differs from the sealed canary at {threads} "
        "intra-op threads, the CPU budget's share: this machine's runtime moves a retrieval "
        "number; prepare again with the budget auto, and report it if it refuses again",
        code="retrieval.execution_canary_mismatch",
        retryable=False,
    )


class LocalOnnxEmbeddingAdapter:
    """One verified, serialized CPU ONNX session."""

    def __init__(
        self,
        model_root: Path,
        spec: HybridIndexSpec,
        *,
        threads: int | None = None,
        input_observer: Callable[[str], None] | None = None,
        runtime_module: Any | None = None,
        processor_class: Any | None = None,
        numpy_module: Any | None = None,
    ) -> None:
        self.spec = spec
        self.threads = threads or safe_intra_op_threads()
        self._input_observer = input_observer
        self._lock = Lock()
        self._closed = False
        self._tokenizer: Any | None = None
        self._session: Any | None = None
        self._numpy = numpy_module or importlib.import_module("numpy")
        self._runtime = runtime_module or importlib.import_module("onnxruntime")
        self._wordpiece = spec.token_normalization == "BERT_WORDPIECE_TOKENIZERS_V1"
        if processor_class is None and not self._wordpiece:
            processor_class = importlib.import_module("sentencepiece").SentencePieceProcessor

        root = encoder_root(model_root, spec)
        model_path = root / "onnx" / "model.onnx"
        tokenizer_path = encoder_tokenizer_path(root, spec)
        if _hash_file(model_path) != spec.model_artifact_sha256:
            raise _semantic_error("semantic model artifact hash differs from the index spec")
        if _hash_file(tokenizer_path) != spec.tokenizer_sha256:
            raise _semantic_error("semantic tokenizer hash differs from the index spec")

        try:
            MODEL_WORK.embedding_model_loads += 1
            if self._wordpiece:
                # The BERT tokenizer file carries its own normaliser, the
                # WordPiece vocabulary and the [CLS] … [SEP] post-processor;
                # truncation is set here, to the spec, never read from a
                # configuration file the loader would otherwise consult.
                tokenizers = importlib.import_module("tokenizers")
                self._tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
                self._tokenizer.no_padding()
                self._tokenizer.enable_truncation(max_length=spec.maximum_tokens)
            else:
                assert processor_class is not None
                self._tokenizer = processor_class(model_file=str(tokenizer_path))
            options = create_session_options(self._runtime, self.threads)
            self._session_options = options
            self._session = self._runtime.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=[spec.execution_provider],
            )
            if self._session.get_providers() != [spec.execution_provider]:
                raise _semantic_error("ONNX session did not bind the CPU-only provider")
            self._input_names = {item.name for item in self._session.get_inputs()}
            sealed = EMBEDDING_CANARIES.get(embedding_context_hash(spec.without_execution()))
            if sealed is not None and runtime_module is None:
                # The real runtime proves its numbers on this machine at these
                # threads; a stand-in runtime has no sealed numbers to prove.
                self._prove_canary(sealed)
        except KnowledgeRetrievalError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise _semantic_error(
                "semantic tokenizer or ONNX session is unavailable",
                cause=exc,
            ) from exc

    @property
    def session_options(self) -> Any:
        """Expose private options for adapter-level deterministic tests."""

        return self._session_options

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._session = None
        self._tokenizer = None
        gc.collect()

    def __enter__(self) -> LocalOnnxEmbeddingAdapter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def embed_passages(
        self,
        texts: Sequence[str],
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        return self._embed(texts, is_query=False, cancelled=cancelled)

    def embed_query(
        self,
        text: str,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[float, ...]:
        return self._embed((text,), is_query=True, cancelled=cancelled)[0]

    def _prove_canary(self, sealed: str) -> None:
        prepared = prepare_embedding_input(
            RETRIEVAL_CANARY_TEXT, policy=self.spec.embedding_input_policy, is_query=False
        )
        row = self._rows((self._encode(prepared),))[0]
        if hashlib.sha256(row.astype("<f4").tobytes()).hexdigest() != sealed:
            raise canary_mismatch("embedding", self.threads)

    def _rows(self, encoded: tuple[list[int], ...]) -> Any:
        """One batch's normalized float32 rows."""

        assert self._session is not None
        width = max(len(token_ids) for token_ids in encoded)
        padding = 0 if self._wordpiece else 1
        input_ids = [token_ids + [padding] * (width - len(token_ids)) for token_ids in encoded]
        attention_masks = [
            [1] * len(token_ids) + [0] * (width - len(token_ids)) for token_ids in encoded
        ]
        feeds = {
            "input_ids": self._numpy.asarray(input_ids, dtype=self._numpy.int64),
            "attention_mask": self._numpy.asarray(attention_masks, dtype=self._numpy.int64),
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = self._numpy.asarray(
                [[0] * width for _ in encoded], dtype=self._numpy.int64
            )
        output = self._numpy.asarray(self._session.run(None, feeds)[0], dtype=self._numpy.float32)
        if self.spec.pooling == "CLS_TOKEN":
            pooled = output[:, 0, :]
        else:
            mask = feeds["attention_mask"].astype(self._numpy.float32)[..., None]
            pooled = (output * mask).sum(axis=1) / self._numpy.maximum(mask.sum(axis=1), 1.0)
        normalized = pooled / self._numpy.maximum(
            self._numpy.linalg.norm(pooled, axis=1, keepdims=True), 1e-12
        )
        if (
            normalized.ndim != 2
            or normalized.shape[1] != self.spec.embedding_dimension
            or not self._numpy.isfinite(normalized).all()
        ):
            raise _semantic_error("semantic runtime returned invalid embeddings")
        return normalized.astype(self._numpy.float32)

    def _embed(
        self,
        texts: Sequence[str],
        *,
        is_query: bool,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[tuple[float, ...], ...]:
        if self._closed:
            raise RuntimeError("embedding adapter is closed")
        if not texts:
            return ()
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent embedding calls are forbidden")
        if is_query:
            MODEL_WORK.query_embeddings += len(texts)
        else:
            MODEL_WORK.passage_calls_attempted += 1
            MODEL_WORK.passages_attempted += len(texts)
        try:
            assert self._session is not None
            rows: list[tuple[float, ...]] = []
            prepared = tuple(
                prepare_embedding_input(
                    text,
                    policy=self.spec.embedding_input_policy,
                    is_query=is_query,
                )
                for text in texts
            )
            if self._input_observer is not None:
                for text in prepared:
                    self._input_observer(text)
            for start in range(0, len(prepared), self.spec.batch_size):
                VERIFIED_PACKS.yield_to_foreground()
                if cancelled is not None and cancelled():
                    raise CancelledError("embedding cancelled")
                encoded = tuple(
                    self._encode(text) for text in prepared[start : start + self.spec.batch_size]
                )
                rows.extend(tuple(float(value) for value in row) for row in self._rows(encoded))
                observer = None if is_query else getattr(_INFERENCE, "passages", None)
                if observer is not None:
                    observer(len(encoded))
            if not is_query:
                MODEL_WORK.passage_calls_completed += 1
                MODEL_WORK.passages_completed += len(texts)
            return tuple(rows)
        except CancelledError:
            if not is_query:
                MODEL_WORK.passage_calls_cancelled += 1
            self.close()
            raise
        except KnowledgeRetrievalError:
            if not is_query:
                MODEL_WORK.passage_calls_failed += 1
            self.close()
            raise
        except Exception as exc:
            if not is_query:
                MODEL_WORK.passage_calls_failed += 1
            self.close()
            raise _semantic_error("semantic inference failed", cause=exc) from exc
        finally:
            self._lock.release()

    def _encode(self, text: str) -> list[int]:
        assert self._tokenizer is not None
        if self._wordpiece:
            return list(self._tokenizer.encode(text).ids)
        piece_ids = self._tokenizer.encode(text, out_type=int)
        converted = [3 if piece_id == 0 else piece_id + 1 for piece_id in piece_ids]
        return [0, *converted[: self.spec.maximum_tokens - 2], 2]


__all__ = [
    "BGE_QUERY_INSTRUCTION",
    "FOREGROUND_WAIT_SECONDS",
    "MODEL_WORK",
    "QWEN3_QUERY_INSTRUCTION",
    "VERIFIED_PACKS",
    "LocalOnnxEmbeddingAdapter",
    "ModelWorkLedger",
    "PackLease",
    "PairScoreCache",
    "SharedEmbeddingAdapter",
    "VerifiedPackRegistry",
    "before_first_inference",
    "canary_mismatch",
    "create_session_options",
    "deserialize_vector",
    "embedding_session_threads",
    "encoder_root",
    "encoder_tokenizer_path",
    "load_vector_extension",
    "observing_passages",
    "pack_facts",
    "prepare_embedding_input",
    "probe_semantic_pack",
    "score_with_cache",
    "semantic_pack_paths",
    "serialize_vector",
    "shared_embedding_adapter",
    "withhold_inference_yield",
    "yielding_during_inference",
]
