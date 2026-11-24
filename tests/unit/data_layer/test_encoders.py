"""The retrieval store's model files, read without a model runtime."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer.encoders import head, safetensors

TEST_SOURCE = "src/alphalattice/foundation/data_layer/encoders.py"


def _file(path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, body = {"__metadata__": {"format": "pt"}}, b""
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(body), len(body) + len(data)],
        }
        body += data
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)


@pytest.mark.proves("data_layer.model_head_tensors_read_as_float32")
def test_a_head_file_reads_its_float32_tensors_by_name_and_refuses_another_dtype(tmp_path) -> None:
    """Each named tensor comes back with its shape from its byte range, the metadata entry is not
    a tensor, and a tensor of another dtype is refused by name."""
    weight = np.arange(6, dtype="<f4").reshape(2, 3)
    bias = np.asarray([0.5], dtype="<f4")
    head = tmp_path / "head.safetensors"
    _file(
        head,
        {
            "linear.bias": ("F32", [1], bias.tobytes()),
            "linear.weight": ("F32", [2, 3], weight.tobytes()),
        },
    )
    read = safetensors(head)
    assert set(read) == {"linear.bias", "linear.weight"}
    assert np.array_equal(read["linear.weight"], weight) and np.array_equal(
        read["linear.bias"], bias
    )
    half = tmp_path / "half.safetensors"
    _file(half, {"w": ("F16", [2], np.zeros(2, dtype="<f2").tobytes())})
    with pytest.raises(DataError) as refused:
        safetensors(half)
    assert refused.value.failure.code == "data_layer.knowledge_model_unavailable"


@pytest.mark.proves("data_layer.reranker_head_scores_the_first_token")
def test_the_head_scores_each_row_by_its_first_token_through_gelu_and_layer_norm() -> None:
    """Only each row's first token reaches the score: Dense, exact GELU, LayerNorm, then the
    final Dense with its bias; the second row's other tokens change nothing."""
    hidden = np.asarray([[[1.0, -1.0], [9.0, 9.0]], [[1.0, -1.0], [-7.0, 3.0]]], dtype=np.float32)
    identity, ones, zeros = (
        np.eye(2, dtype=np.float32),
        np.ones(2, np.float32),
        np.zeros(2, np.float32),
    )
    last, bias = np.asarray([[1.0], [-1.0]], dtype=np.float32), np.asarray([0.5], np.float32)
    scores = head(hidden, identity, ones, zeros, last, bias)
    # GELU(1) = 0.8413 and GELU(-1) = -0.1587 sit 0.5 either side of their mean, so LayerNorm
    # makes them +1 and -1 (less epsilon's share), and the final Dense gives 1 - (-1) + 0.5.
    expected = 2 * 0.5 / np.sqrt(0.25 + 1e-5) + 0.5
    assert scores.shape == (2,) and np.allclose(scores, expected, atol=1e-5)
