"""The knowledge store's fusion of its lexical and vector channels, and its lexical words."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest

from alphalattice.foundation.data_layer import knowledge
from alphalattice.foundation.data_layer.knowledge import Chunk, SearchHit, WordCounts, fuse

TEST_SOURCE = "src/alphalattice/foundation/data_layer/knowledge.py"


@pytest.mark.proves("data_layer.knowledge_chunks_preserve_character_offsets")
@pytest.mark.parametrize(
    ("body", "capacity", "overlap", "trim", "chunks"),
    [
        ("One.\n\nTwo. three four", 12, 0, True, [(0, "One."), (6, "Two."), (11, "three four")]),
        ("One.\n\n\nTwo.\n\nThree end", 16, 0, True, [(0, "One."), (7, "Two.\n\nThree end")]),
        ("甲乙。丙丁戊己庚", 5, 0, True, [(0, "甲乙。"), (3, "丙丁戊己庚")]),
        (" alpha beta gamma", 9, 0, True, [(1, "alpha"), (7, "beta"), (12, "gamma")]),
        ("   alpha beta", 10, 0, True, [(3, "alpha beta")]),
        ("abcdefgh", 4, 2, False, [(0, "abcd"), (2, "cdef"), (4, "efgh")]),
        (" A.\n\n BCDEF", 6, 0, False, [(0, " A.\n\n"), (5, " BCDEF")]),
        (" \t\n", 2, 0, True, []),
    ],
    ids=(
        "paragraph-before-sentence",
        "paragraph-levels",
        "unicode-sentence",
        "word",
        "leading-trim",
        "overlap",
        "untrimmed",
        "blank",
    ),
)
def test_chunk_slices_preserve_boundary_offsets(body, capacity, overlap, trim, chunks) -> None:
    """The preferred boundaries, overlap and trimming preserve positions in the original text."""
    assert list(knowledge._chunk_slices(body, capacity, overlap, trim=trim)) == chunks


def _hit(chunk_id: str, score: float, rowid: int) -> SearchHit:
    return SearchHit("s", "1", rowid, Chunk(chunk_id, 0, 1, "d" + chunk_id), score, rowid)


@pytest.mark.proves("data_layer.knowledge_fusion_weighs_both_channels")
def test_fusion_ranks_a_chunk_both_channels_find_over_one_found_once() -> None:
    """Each channel's scores are scaled to [0, 1] before the equal-weight sum, so a chunk both
    channels find outranks a top hit of one channel alone; equal scores keep source row order, and
    the limit cuts the fused order."""
    lexical = (_hit("a", 12.0, 3), _hit("b", 4.0, 1), _hit("c", 2.0, 2))
    vector = (_hit("b", 0.9, 1), _hit("d", 0.1, 4))
    fused = fuse(lexical, vector)
    assert [hit.chunk.chunk_id for hit in fused] == ["b", "a", "c", "d"]
    assert [round(hit.score, 4) for hit in fused] == [0.6, 0.5, 0.0, 0.0]
    assert [hit.chunk.chunk_id for hit in fuse(lexical, vector, limit=2)] == ["b", "a"]
    assert fuse((), ()) == ()


@pytest.mark.proves("data_layer.lexical_leaves_out_common_words")
def test_a_large_index_leaves_out_the_words_most_chunks_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """From the large-index size a word at least half the chunks hold is common, each word read
    once and counted again after a chunk is added; a smaller index keeps every word."""
    texts = ["alpha beta", "alpha gamma", "delta"]
    reads: list[str] = []

    def holding(word: str) -> int:
        reads.append(word)
        return sum(word in text.split() for text in texts)

    def common(*words: str) -> frozenset[str]:
        return counts.common(len(texts), lambda: len(texts), holding, list(words))

    counts = WordCounts()
    assert common("alpha", "gamma") == frozenset()  # a small index
    monkeypatch.setattr(knowledge, "COMMON_WORDS_FROM", 1)
    assert common("alpha", "gamma") == {"alpha"}  # two of three
    assert common("alpha") == {"alpha"} and reads == ["alpha", "gamma"]  # read once
    texts += ["delta zeta", "delta eta"]
    assert common("alpha", "delta") == {"delta"}  # alpha: two of five


@pytest.mark.proves("data_layer.knowledge_batches_decode_once")
def test_writing_the_index_reads_each_batch_once_into_its_exact_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each batch is read once, in steps, and each row is the value a whole-batch decode gives."""
    encoded = {
        "a": np.asarray([[1, -2, 3], [127, 0, -127], [5, 6, -7]], dtype="i1"),
        "b": np.asarray([[-1, 1, 0]], dtype="i1"),
    }
    scales = {"a": np.asarray([0.5, 0.25, 0.125], "<f4"), "b": np.asarray([1, 2, 3], "<f4")}
    reads: list[str] = []

    class Log:
        @contextmanager
        def reading(self):
            yield self

        def execute(self, _sql: str, args: tuple[str]):
            reads.append(args[0])
            batch = args[0]
            values = (encoded[batch].tobytes(), scales[batch].tobytes(), len(encoded[batch]))
            return type("Found", (), {"fetchone": lambda _: values})()

    monkeypatch.setattr(knowledge, "DECODED", 1)
    rows = [(None,) * 7 + (batch, ordinal) for batch, ordinal in (("a", 2), ("b", 0), ("a", 0))]
    into = np.zeros((4, 3), dtype=np.float32)
    knowledge.decode_batches(Log(), rows, into, 1, 3, "int8")
    whole = {batch: encoded[batch].astype(np.float32) * scales[batch] for batch in encoded}
    np.testing.assert_array_equal(into, [[0, 0, 0], whole["a"][2], whole["b"][0], whole["a"][0]])
    assert sorted(reads) == ["a", "b"]
