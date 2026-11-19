"""The retrieval store's two CPU models, run from their verified upstream files through onnxruntime.

bekko-static 512 int8 encodes passages and queries alike. Ettin17 int8 scores an English shortlist
of at most 20 (query, passage) pairs at 256 tokens, one pair a run: its published backbone graph,
then its published head (the first token, Dense, GELU, LayerNorm, Dense) in NumPy from the pack's
safetensors, as the retrieval measurement ran it. A shortlist that takes longer than one
second returns no scores, and the caller keeps the hybrid order.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Any, Final

import numpy as np

from alphalattice.kernel.data.errors import DataError

DIMENSION: Final = 512
SHORTLIST: Final = 20
PAIR_TOKENS: Final = 256
DEADLINE_SECONDS: Final = 1.0
THREADS: Final = min(4, os.cpu_count() or 1)
"""Intra-op threads of each session: the four cores the speed target is measured on, or fewer."""


def _refuse(message: str) -> DataError:
    return DataError(message, code="data_layer.knowledge_model_unavailable")


def _session(graph: Path) -> Any:
    """One sequential CPU session on THREADS threads that never spins."""
    runtime = importlib.import_module("onnxruntime")  # lazy: the semantic extra
    options = runtime.SessionOptions()
    options.execution_mode = runtime.ExecutionMode.ORT_SEQUENTIAL
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = THREADS
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.graph_optimization_level = runtime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = runtime.InferenceSession(
        str(graph), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise _refuse(f"{graph.name} did not bind the CPU provider")
    return session


def _tokenizer(path: Path) -> Any:
    tokenizers = importlib.import_module("tokenizers")  # lazy: the semantic extra
    tokenizer = tokenizers.Tokenizer.from_file(str(path))
    tokenizer.no_padding()
    return tokenizer


def _padded(encoded: Sequence[Any], pad: int) -> tuple[np.ndarray, np.ndarray]:
    width = max(1, max(len(item.ids) for item in encoded))
    ids = np.full((len(encoded), width), pad, dtype=np.int64)
    mask = np.zeros_like(ids)
    for row, item in enumerate(encoded):
        ids[row, : len(item.ids)] = item.ids
        mask[row, : len(item.ids)] = 1
    return ids, mask


def safetensors(path: Path) -> dict[str, np.ndarray]:
    """The float32 tensors of a safetensors file: an 8-byte little-endian header length, the JSON
    header naming each tensor's dtype, shape and byte range, then the bytes."""
    data = path.read_bytes()
    size = int.from_bytes(data[:8], "little")
    body = memoryview(data)[8 + size :]
    tensors = {}
    for name, spec in json.loads(data[8 : 8 + size]).items():
        if name == "__metadata__":
            continue
        if spec["dtype"] != "F32":
            raise _refuse(f"{path.name} holds {spec['dtype']} tensors, not float32")
        start, end = spec["data_offsets"]
        tensors[name] = np.frombuffer(body[start:end], dtype="<f4").reshape(spec["shape"])
    return tensors


def head(
    hidden: np.ndarray,
    first: np.ndarray,
    scale: np.ndarray,
    shift: np.ndarray,
    last: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """The reranker's published head over its backbone's hidden states: each row's first token,
    a Dense layer without bias, exact (erf) GELU, LayerNorm at epsilon 1e-5, and a Dense layer to
    one logit."""
    dense = hidden[:, 0, :] @ first
    gelu = 0.5 * dense * (1.0 + np.vectorize(math.erf, otypes=[np.float32])(dense / 2**0.5))
    mean = gelu.mean(axis=1, keepdims=True)
    variance = ((gelu - mean) ** 2).mean(axis=1, keepdims=True)
    normed = (gelu - mean) / np.sqrt(variance + 1e-5) * scale + shift
    return (normed @ last + bias).reshape(-1)


class StaticEncoder:
    """bekko-static 512 int8: one padded batch in, unit-length float32 rows out."""

    def __init__(self, pack: Path) -> None:
        """Load the pack's graph and tokenizer; the pack's files are verified by its installer."""
        self._tokenizer = _tokenizer(pack / "tokenizer.json")
        self._session = _session(pack / "onnx" / "model_int8_dim512.onnx")
        self._lock = Lock()

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Each text's unit vector; an empty or non-finite one is refused by name."""
        if not texts:
            return np.empty((0, DIMENSION), dtype=np.float32)
        with self._lock:
            encoded = self._tokenizer.encode_batch(list(texts), add_special_tokens=False)
            ids, mask = _padded(encoded, 0)
            (values,) = self._session.run(
                ["sentence_embedding"], {"input_ids": ids, "attention_mask": mask}
            )
        values = np.asarray(values, dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if values.shape != (len(texts), DIMENSION) or not (np.isfinite(norms) & (norms > 0)).all():
            raise _refuse("the encoder returned an invalid vector")
        return values / norms

    def embed_query(self, text: str) -> np.ndarray:
        """A query's vector, by the passages' own encoder."""
        return self.embed_passages((text,))[0]


class Reranker:
    """Ettin17 int8: the published backbone, then its head in NumPy."""

    def __init__(self, pack: Path) -> None:
        """Load the backbone graph, the tokenizer and the head's three published tensors files."""
        self._tokenizer = _tokenizer(pack / "tokenizer.json")
        self._tokenizer.enable_truncation(
            max_length=PAIR_TOKENS, strategy="longest_first", direction="right"
        )
        pad = self._tokenizer.token_to_id("[PAD]")
        if pad is None:
            raise _refuse("the reranker's tokenizer has no padding token")
        self._pad = pad
        self._session = _session(pack / "onnx" / "model_qint8_avx512.onnx")
        self._first = safetensors(pack / "2_Dense" / "model.safetensors")["linear.weight"].T
        norm = safetensors(pack / "3_LayerNorm" / "model.safetensors")
        self._scale, self._shift = norm["norm.weight"], norm["norm.bias"]
        last = safetensors(pack / "4_Dense" / "model.safetensors")
        self._last, self._bias = last["linear.weight"].T, last["linear.bias"]
        self._lock = Lock()

    def _head(self, hidden: np.ndarray) -> np.ndarray:
        return head(hidden, self._first, self._scale, self._shift, self._last, self._bias)

    def score(self, query: str, passages: Sequence[str]) -> tuple[float, ...] | None:
        """One logit per passage, each pair run alone: the backbone quantizes its activations over
        the whole batch, so a pair batched with others would score by its neighbours. None once
        the shortlist passes the deadline."""
        if len(passages) > SHORTLIST:
            raise DataError(
                f"the reranker scores at most {SHORTLIST} passages",
                code="data_layer.knowledge_query_invalid",
            )
        started = time.monotonic()
        scores = []
        with self._lock:
            for item in self._tokenizer.encode_batch([(query, passage) for passage in passages]):
                if time.monotonic() - started > DEADLINE_SECONDS:
                    return None
                ids, mask = _padded((item,), self._pad)
                (hidden,) = self._session.run(
                    ["last_hidden_state"], {"input_ids": ids, "attention_mask": mask}
                )
                scores.append(self._head(np.asarray(hidden, dtype=np.float32))[0])
        if time.monotonic() - started > DEADLINE_SECONDS:
            return None
        if not np.isfinite(scores).all():
            raise _refuse("the reranker returned an invalid score")
        return tuple(float(value) for value in scores)
