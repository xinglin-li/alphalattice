"""A model pack proven once per process is held and reused while unchanged.

Every build, retrieval session and capability check re-hashed a 475 MB pack
and re-loaded its ONNX session (2.6 GB of reads and six seconds per issuer
unit on the real-copy flow, for a unit that embedded nothing). The registry
keeps one proof and one session per pack, reuses them while the lstat facts
of every consumed path are those of the proof, and proves again on any
change; a lease's close keeps the session; the ledger counts the reuses.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from alphalattice.kernel.knowledge import _embeddings as embeddings
from alphalattice.kernel.knowledge import _reranking as reranking
from alphalattice.kernel.knowledge import hybrid
from alphalattice.kernel.knowledge.hybrid_contracts import RERANKER_PACK, HybridIndexSpec


class _Session:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _touch(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))


def test_a_proof_is_reused_while_the_consumed_files_are_unchanged(tmp_path: Path) -> None:
    """requirement: one load per pack per process; any change proves again."""

    registry = embeddings.VerifiedPackRegistry()
    consumed = (tmp_path / "model.bin", tmp_path / "tokenizer.bin")
    for path in consumed:
        path.write_bytes(b"pack")
    loads: list[str] = []

    def load(name: str) -> _Session:
        loads.append(name)
        return _Session(name)

    first, reused = registry.resolve(("k",), consumed=consumed, load=lambda: load("first"))
    assert not reused and loads == ["first"]
    again, reused = registry.resolve(("k",), consumed=consumed, load=lambda: load("again"))
    assert reused and again is first and loads == ["first"]
    # A replaced file (same size, later mtime) proves again and releases the old value.
    _touch(consumed[0], b"pack")
    replaced, reused = registry.resolve(
        ("k",), consumed=consumed, load=lambda: load("replaced"), release=lambda v: v.close()
    )
    assert not reused and replaced is not first and first.closed and loads[-1] == "replaced"
    # A missing consumed path never matches a held proof: the loader runs and may refuse.
    consumed[1].unlink()
    with pytest.raises(RuntimeError, match="absent"):
        registry.resolve(
            ("k",), consumed=consumed, load=lambda: (_ for _ in ()).throw(RuntimeError("absent"))
        )
    registry.release()
    assert replaced.closed


def test_shared_adapters_lease_one_session_and_count_reuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement: the default factories hand out leases on held sessions; a
    lease's close keeps the session; a changed pack file loads a new one."""

    spec = HybridIndexSpec.fixed_v2()
    model_root = tmp_path / "pack"
    (model_root / "onnx").mkdir(parents=True)
    (model_root / "onnx" / "model.onnx").write_bytes(b"model")
    (model_root / "sentencepiece.bpe.model").write_bytes(b"tokenizer")
    pack = reranking.reranker_pack_root(model_root)
    pack.mkdir(parents=True)
    for name, _digest in RERANKER_PACK:
        (pack / name).parent.mkdir(parents=True, exist_ok=True)
        (pack / name).write_bytes(b"reranker")
    loaded: list[str] = []

    class _FakeEmbedding:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            loaded.append("embedding")
            self.spec = spec
            self.closed = False

        def embed_query(self, text: str, *, cancelled: Any = None) -> tuple[float, ...]:
            return (float(len(text)),)

        def embed_passages(self, texts: Any, *, cancelled: Any = None) -> tuple[Any, ...]:
            return tuple((float(len(t)),) for t in texts)

        def close(self) -> None:
            self.closed = True

    class _FakeReranker:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            loaded.append("reranker")

        def score(self, query: str, passages: Any) -> tuple[float, ...]:
            return tuple(float(len(p)) for p in passages)

    monkeypatch.setattr(embeddings, "LocalOnnxEmbeddingAdapter", _FakeEmbedding)
    monkeypatch.setattr(reranking, "LocalCrossEncoderAdapter", _FakeReranker)
    monkeypatch.setattr(embeddings, "VERIFIED_PACKS", embeddings.VerifiedPackRegistry())
    monkeypatch.setattr(reranking, "VERIFIED_PACKS", embeddings.VERIFIED_PACKS)
    before = embeddings.MODEL_WORK.snapshot()

    lease = hybrid._default_adapter_factory(model_root, spec)
    assert lease.embed_query("abc") == (3.0,) and loaded == ["embedding"]
    session = lease._lease.value
    lease.close()
    assert lease.closed and not session.closed, "closing the lease keeps the session"
    second = hybrid._default_adapter_factory(model_root, spec)
    assert second._lease.value is session and loaded == ["embedding"]
    reranker = hybrid._default_reranker_factory(model_root, spec)
    assert reranker.score("q", ("aa", "b")) == (2.0, 1.0) and loaded == ["embedding", "reranker"]
    encoder = reranker._lease.value
    reranker.close()
    assert hybrid._default_reranker_factory(model_root, spec)._lease.value is encoder
    after = embeddings.MODEL_WORK.snapshot()
    assert after["embedding_model_reuses"] - before["embedding_model_reuses"] == 1
    assert after["reranker_model_reuses"] - before["reranker_model_reuses"] == 1
    # A grown model file is a different pack: the next lease loads again; the
    # replaced session, still leased by `second`, closes with that lease.
    (model_root / "onnx" / "model.onnx").write_bytes(b"model-v2")
    third = hybrid._default_adapter_factory(model_root, spec)
    assert third._lease.value is not session and not session.closed
    second.close()
    assert session.closed
    assert loaded == ["embedding", "reranker", "embedding"]


def test_capability_probe_is_proven_once_and_again_after_a_pack_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement: the readiness proof is held with both packs' facts."""

    spec = HybridIndexSpec.fixed_v2()
    model_root = tmp_path / "pack"
    (model_root / "onnx").mkdir(parents=True)
    (model_root / "onnx" / "model.onnx").write_bytes(b"model")
    (model_root / "sentencepiece.bpe.model").write_bytes(b"tokenizer")
    pack = reranking.reranker_pack_root(model_root)
    pack.mkdir(parents=True)
    for name, _digest in RERANKER_PACK:
        (pack / name).parent.mkdir(parents=True, exist_ok=True)
        (pack / name).write_bytes(b"reranker")
    proofs: list[int] = []
    monkeypatch.setattr(embeddings, "VERIFIED_PACKS", embeddings.VerifiedPackRegistry())
    monkeypatch.setattr(
        hybrid, "_prove_capability", lambda root, spec_: proofs.append(1) or ("report", len(proofs))
    )
    before = embeddings.MODEL_WORK.snapshot()["pack_probe_reuses"]
    assert hybrid._default_capability_probe(model_root, spec) == ("report", 1)
    assert hybrid._default_capability_probe(model_root, spec) == ("report", 1)
    assert embeddings.MODEL_WORK.snapshot()["pack_probe_reuses"] - before == 1
    # A file added to the reranker pack directory changes the directory's facts.
    (pack / "unadmitted.txt").write_bytes(b"extra")
    stat = pack.stat()
    os.utime(pack, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert hybrid._default_capability_probe(model_root, spec) == ("report", 2)


def test_a_host_that_runs_items_at_once_holds_a_session_for_each(tmp_path: Path) -> None:
    """requirement (first-day speed): one session per pack unless the Host
    raises it; raised, leases held at once each take a session of their own up
    to the bound, then share the least-leased; a released session is taken
    again before another loads; a changed pack retires every one of them."""

    registry = embeddings.VerifiedPackRegistry()
    consumed = (tmp_path / "model.bin",)
    consumed[0].write_bytes(b"pack")
    loaded: list[_Session] = []

    def load() -> _Session:
        loaded.append(_Session(str(len(loaded))))
        return loaded[-1]

    def lease() -> embeddings.PackLease:
        return registry.lease(("k",), consumed=consumed, load=load)

    one, two = lease(), lease()
    assert two.value is one.value and len(loaded) == 1, "by default one session is shared"
    registry.hold_up_to(2)
    three = lease()
    assert three.value is not one.value and not three.reused and len(loaded) == 2
    four = lease()
    assert four.value is three.value and four.reused, "past the bound: the least-leased"
    for value in (one, two, three, four):
        value.release()
    again = lease()
    assert again.reused and len(loaded) == 2 and again.value in loaded
    _touch(consumed[0], b"pack")
    fresh = lease()
    assert fresh.value is loaded[-1] and len(loaded) == 3
    assert all(value.closed for value in loaded[:2] if value is not again.value)
    again.release()
    assert all(value.closed for value in loaded[:2])
    fresh.release()


def test_a_model_call_gives_up_the_callers_lock_only_where_it_yields() -> None:
    """Requirement: a caller inside `yielding_during_inference` gives its
    lock up while the model runs and holds it again after; outside that scope,
    or once a build withholds it (a committed generation materialized under the
    workspace's lock), the model call keeps the caller's lock."""

    import threading
    from collections.abc import Iterator
    from contextlib import contextmanager

    writer = threading.Lock()
    observed: list[bool] = []
    spec = HybridIndexSpec.fixed_v2()

    class _Model:
        closed = False

        def __init__(self) -> None:
            self.spec = spec

        def embed_query(self, text: str, *, cancelled: Any = None) -> tuple[float, ...]:
            observed.append(writer.locked())
            return (float(len(text)),)

    @contextmanager
    def released() -> Iterator[None]:
        writer.release()
        try:
            yield
        finally:
            writer.acquire()

    registry = embeddings.VerifiedPackRegistry()
    adapter = embeddings.SharedEmbeddingAdapter(registry.lease(("k",), consumed=(), load=_Model))
    with writer:
        adapter.embed_query("a")
        with embeddings.yielding_during_inference(released):
            adapter.embed_query("b")
            embeddings.withhold_inference_yield()
            adapter.embed_query("c")
        assert writer.locked()
    adapter.close()
    assert observed == [True, False, True]


def test_sessions_of_two_widths_are_held_side_by_side(tmp_path: Path) -> None:
    """requirement (the final close-out, F2): a unit's sessions are its own share of
    the CPU budget. A lease takes a held session of its width, loading one while
    fewer than the bound are held, and leaves the others to their own callers; an
    idle one of another width gives its place when the pool is full."""

    registry = embeddings.VerifiedPackRegistry()
    registry.hold_up_to(2)
    consumed = (tmp_path / "model.bin",)
    consumed[0].write_bytes(b"pack")
    loaded: list[_Session] = []

    def lease(threads: int) -> embeddings.PackLease:
        def load() -> _Session:
            loaded.append(_Session(str(threads)))
            return loaded[-1]

        return registry.lease(
            ("k",), consumed=consumed, load=load, fits=lambda value: value.name == str(threads)
        )

    wide, narrow = lease(16), lease(4)
    assert (wide.value.name, narrow.value.name, len(loaded)) == ("16", "4", 2)
    wide.release()
    narrow.release()
    again = lease(4)
    assert again.value is narrow.value and again.reused
    other = lease(8)
    assert other.value.name == "8" and loaded[0].closed and not narrow.value.closed
    again.release()
    other.release()


def test_model_work_waits_for_the_foreground_and_a_gate_at_its_first_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """requirement (F2): while a reader the lead waits on runs (a bundle), another
    thread's model call waits and goes on when the reader ends -- the reader's own
    calls do not wait; and a thread's gate runs once, at its first model call that
    gives up the caller's lock, never while the lock is held."""

    import threading
    from collections.abc import Iterator
    from contextlib import contextmanager

    registry = embeddings.VerifiedPackRegistry()
    monkeypatch.setattr(embeddings, "VERIFIED_PACKS", registry)
    spec = HybridIndexSpec.fixed_v2()
    calls: list[str] = []

    class _Model:
        closed = False

        def __init__(self) -> None:
            self.spec = spec

        def embed_query(self, text: str, *, cancelled: Any = None) -> tuple[float, ...]:
            calls.append(text)
            return (float(len(text)),)

    registry.hold_up_to(2)
    first = embeddings.SharedEmbeddingAdapter(registry.lease(("k",), consumed=(), load=_Model))
    second = embeddings.SharedEmbeddingAdapter(registry.lease(("k",), consumed=(), load=_Model))
    background = threading.Thread(target=lambda: second.embed_query("later"))
    with registry.foreground():
        background.start()
        background.join(timeout=0.3)
        assert background.is_alive() and calls == [], "the other thread's call waits"
        first.embed_query("bundle")
        assert calls == ["bundle"], "the reader's own call does not"
    background.join(timeout=5.0)
    assert calls == ["bundle", "later"]

    writer = threading.Lock()
    gated: list[bool] = []

    @contextmanager
    def released() -> Iterator[None]:
        writer.release()
        try:
            yield
        finally:
            writer.acquire()

    with writer, embeddings.before_first_inference(lambda: gated.append(writer.locked())):
        first.embed_query("a")
        assert gated == [], "a call that keeps the lock does not wait there"
        with embeddings.yielding_during_inference(released):
            first.embed_query("b")
            first.embed_query("c")
    assert gated == [False], "once, with the lock given up"
    first.close()
    second.close()
