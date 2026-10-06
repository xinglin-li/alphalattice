"""A held model session that failed or was cancelled is not handed out again.

The real `LocalOnnxEmbeddingAdapter` closes itself on cancellation and on
inference failure. The registry that holds one session per pack per process
must see that, load a fresh session for the next request in the same
process, keep a healthy session's reuse, and never close a session another
consumer still leases. The adapter here is the real one over an injected
inference runtime, so the real close path runs; only the ONNX graph is
stood in for.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import CancelledError
from functools import partial
from pathlib import Path
from typing import Any

import numpy
import pytest

from alphalattice.kernel.knowledge import _embeddings as embeddings
from alphalattice.kernel.knowledge import hybrid
from alphalattice.kernel.knowledge.hybrid_contracts import (
    MODEL_ARTIFACT_SHA256,
    TOKENIZER_SHA256,
    HybridIndexSpec,
)
from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError

SPEC = HybridIndexSpec.fixed_v2()


class _Options:
    def __init__(self) -> None:
        self.entries: dict[str, str] = {}

    def add_session_config_entry(self, key: str, value: str) -> None:
        self.entries[key] = value


class _Input:
    def __init__(self, name: str) -> None:
        self.name = name


class _Session:
    """An inference session whose `run` can be made to fail or to linger."""

    failing = False
    delay = 0.0
    runs = 0

    def __init__(self, path: str, *, sess_options: Any, providers: list[str]) -> None:
        self.path = path
        self.providers = providers

    def get_providers(self) -> list[str]:
        return list(self.providers)

    def get_inputs(self) -> list[_Input]:
        return [_Input("input_ids"), _Input("attention_mask")]

    def run(self, _outputs: None, feeds: dict[str, Any]) -> list[Any]:
        type(self).runs += 1
        if type(self).delay:
            time.sleep(type(self).delay)
        if type(self).failing:
            raise RuntimeError("inference failed")
        batch, width = feeds["input_ids"].shape
        vectors = numpy.zeros((batch, width, SPEC.embedding_dimension), dtype=numpy.float32)
        vectors[:, :, 0] = 1.0
        return [vectors]


class _Runtime:
    SessionOptions = _Options
    InferenceSession = _Session

    class ExecutionMode:
        ORT_SEQUENTIAL = "sequential"


class _Processor:
    def __init__(self, *, model_file: str) -> None:
        self.model_file = model_file

    def encode(self, text: str, *, out_type: type) -> list[int]:
        del out_type
        return [7 + index for index in range(max(1, min(len(text.split()), 6)))]


@pytest.fixture
def pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A pack whose two file hashes are stood in for; everything else is real."""

    model_root = tmp_path / "pack"
    (model_root / "onnx").mkdir(parents=True)
    (model_root / "onnx" / "model.onnx").write_bytes(b"absent-model-pack-placeholder")
    (model_root / "sentencepiece.bpe.model").write_bytes(b"absent-tokenizer-placeholder")
    pinned = {"model.onnx": MODEL_ARTIFACT_SHA256, "sentencepiece.bpe.model": TOKENIZER_SHA256}
    real = embeddings._hash_file
    monkeypatch.setattr(
        embeddings, "_hash_file", lambda path: pinned.get(path.name) or str(real(path))
    )
    monkeypatch.setattr(
        embeddings,
        "LocalOnnxEmbeddingAdapter",
        partial(
            embeddings.LocalOnnxEmbeddingAdapter,
            runtime_module=_Runtime,
            processor_class=_Processor,
            numpy_module=numpy,
        ),
    )
    monkeypatch.setattr(embeddings, "VERIFIED_PACKS", embeddings.VerifiedPackRegistry())
    _Session.failing = False
    _Session.delay = 0.0
    _Session.runs = 0
    return model_root


def _loads() -> int:
    return embeddings.MODEL_WORK.snapshot()["embedding_model_loads"]


def _reuses() -> int:
    return embeddings.MODEL_WORK.snapshot()["embedding_model_reuses"]


def test_a_healthy_session_is_reused_and_a_cancelled_one_is_replaced(pack: Path) -> None:
    """requirement: normal reuse loads once; cancellation during adapter work
    closes the real session, and the next normal request gets a usable one."""

    loads, reuses = _loads(), _reuses()
    first = hybrid._default_adapter_factory(pack, SPEC)
    assert len(first.embed_query("alpha beta")) == SPEC.embedding_dimension
    second = hybrid._default_adapter_factory(pack, SPEC)
    assert len(second.embed_query("gamma")) == SPEC.embedding_dimension
    assert _loads() - loads == 1 and _reuses() - reuses == 1, "one load, one reuse"
    first.close()
    second.close()

    # Cancellation inside the real adapter's work: it closes itself.
    working = hybrid._default_adapter_factory(pack, SPEC)
    with pytest.raises(CancelledError):
        working.embed_passages(("one", "two", "three"), cancelled=lambda: True)
    with pytest.raises(RuntimeError, match="closed"):
        working.embed_query("still on the dead session")
    working.close()

    # The next normal request in the same process must not receive that session.
    recovered = hybrid._default_adapter_factory(pack, SPEC)
    assert len(recovered.embed_query("after cancellation")) == SPEC.embedding_dimension
    assert _loads() - loads == 2, "the dead session was replaced by one fresh load"
    recovered.close()


def test_an_inference_failure_is_replaced_on_the_next_request(pack: Path) -> None:
    """requirement: a session that failed mid-inference is closed by the
    adapter; the following normal request recovers within the process."""

    loads = _loads()
    lease = hybrid._default_adapter_factory(pack, SPEC)
    assert len(lease.embed_query("fine")) == SPEC.embedding_dimension
    _Session.failing = True
    with pytest.raises(KnowledgeRetrievalError):
        lease.embed_passages(("a passage",))
    _Session.failing = False
    with pytest.raises(RuntimeError, match="closed"):
        lease.embed_query("dead")
    lease.close()
    recovered = hybrid._default_adapter_factory(pack, SPEC)
    assert len(recovered.embed_passages(("a passage",))) == 1
    assert _loads() - loads == 2
    again = hybrid._default_adapter_factory(pack, SPEC)
    assert again.embed_query("reused") and _loads() - loads == 2, "healthy reuse resumes"
    recovered.close()
    again.close()


def test_replacement_waits_for_outstanding_leases_and_calls_serialize(pack: Path) -> None:
    """requirement: a replaced session stays open for the consumer that still
    leases it and closes when that lease ends; concurrent consumers of one
    session wait for each other instead of being refused."""

    holder = hybrid._default_adapter_factory(pack, SPEC)
    held = holder._lease.value
    (pack / "onnx" / "model.onnx").write_bytes(b"absent-model-pack-placeholder-v2")
    replacement = hybrid._default_adapter_factory(pack, SPEC)
    assert replacement._lease.value is not held
    assert not held.closed, "an outstanding lease keeps its session open"
    assert len(holder.embed_query("still served")) == SPEC.embedding_dimension
    holder.close()
    assert held.closed, "the retired session closes with its last lease"
    replacement.close()

    # Two threads through two leases on one session: both complete.
    first = hybrid._default_adapter_factory(pack, SPEC)
    second = hybrid._default_adapter_factory(pack, SPEC)
    assert first._lease.value is second._lease.value
    _Session.delay = 0.05
    results: list[int] = []
    errors: list[BaseException] = []

    def call(lease: Any) -> None:
        try:
            results.append(len(lease.embed_query("concurrent")))
        except BaseException as error:  # the refusal is what this proves absent
            errors.append(error)

    threads = [threading.Thread(target=call, args=(lease,)) for lease in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    _Session.delay = 0.0
    assert errors == [] and results == [SPEC.embedding_dimension] * 2
    first.close()
    second.close()
