"""The knowledge store's fusion of its lexical and vector channels, and its lexical words."""

from __future__ import annotations

import pytest

from alphalattice.kernel.data_layer import knowledge
from alphalattice.kernel.data_layer.knowledge import Chunk, SearchHit, WordCounts, fuse

TEST_SOURCE = "src/alphalattice/kernel/data_layer/knowledge.py"


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
