"""The Host's shared retrieval cache keeps its recipe, model and encoded bytes."""

from __future__ import annotations

import random
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import create_data_root, knowledge, recall
from alphalattice.kernel.data_layer.chain import digest
from alphalattice.kernel.data_layer.encoders import Reranker, StaticEncoder
from alphalattice.kernel.data_layer.knowledge import HybridSearch, KnowledgeStore, SourceOrigin
from alphalattice.kernel.data_layer.log import HostLog, uuid7

TEST_SOURCE = "src/alphalattice/kernel/data_layer/knowledge.py"


@pytest.mark.proves("data_layer.knowledge_embedding_cache_survives_its_readers")
@pytest.mark.parametrize("precision", ("float16", "int8"), ids=("float16", "per-dimension-int8"))
def test_another_reader_reuses_the_encoded_chunks(tmp_path: Path, precision: str) -> None:
    """Duplicate inputs and another store reader never send a held chunk back to the encoder."""
    create_data_root(tmp_path)
    keys = (digest(b"shared document"), digest(b"another document"))
    vectors = np.asarray(((0.25, 0.5, -0.75), (0.0, 1.0, 0.0)), dtype=np.float32)
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    calls: list[tuple[str, ...]] = []

    def encode(missing: tuple[str, ...]) -> np.ndarray:
        """Only the model is doubled; the cache, writer and restart read real SQLite bytes."""
        calls.append(missing)
        return vectors[np.asarray([keys.index(key) for key in missing])]

    with HostLog.open(tmp_path) as log:
        first = KnowledgeStore(log)
        recipe = first.recipe({"splitter": "semantic-text-splitter", "version": "0.33.3"})
        manifest = {"graph": "fixture.onnx", "sha256": "0" * 64}
        model = first.model(manifest, dimension=3, precision=precision)
        built = first.embeddings(keys, recipe_id=recipe, model_id=model, encode=encode)
        second = KnowledgeStore(log)
        assert second.recipe({"version": "0.33.3", "splitter": "semantic-text-splitter"}) == recipe
        assert second.model(manifest, dimension=3, precision=precision) == model
        reused = second.embeddings(
            (keys[1], keys[0], keys[1]), recipe_id=recipe, model_id=model, encode=encode
        )
        assert calls == [keys]
        np.testing.assert_allclose(built.vectors, normalized, rtol=0, atol=1 / 127)
        np.testing.assert_array_equal(reused.vectors, built.vectors[[1, 0, 1]])
        assert (built.embedded, built.reused, reused.embedded, reused.reused) == (2, 0, 0, 3)
        with pytest.raises(DataError, match="zero"):
            first.embeddings(
                (digest(b"zero"),),
                recipe_id=recipe,
                model_id=model,
                encode=lambda missing: np.zeros((len(missing), 3), dtype=np.float32),
            )
        with log.reading() as con:
            count, byte_count, scales = con.execute(
                "SELECT count(*), sum(length(vectors)), sum(coalesce(length(scales), 0))"
                " FROM knowledge_embedding_batch"
            ).fetchone()
            expected = (1, 12, 0) if precision == "float16" else (1, 6, 12)
            assert (count, byte_count, scales) == expected
            ids = con.execute(
                "SELECT recipe_id FROM knowledge_recipe UNION ALL"
                " SELECT model_id FROM knowledge_model UNION ALL"
                " SELECT batch_id FROM knowledge_embedding_batch UNION ALL"
                " SELECT embedding_id FROM knowledge_embedding"
            ).fetchall()
            assert len(ids) == 5 and all(UUID(row[0]).version == 7 for row in ids)


@pytest.mark.proves("data_layer.knowledge_publications_replay_with_their_sources")
def test_a_published_revision_and_snapshot_are_rebuilt_from_the_log(tmp_path: Path) -> None:
    """Replay crosses the real SQLite log and its source projections.

    An in-memory double cannot prove the publication transaction or its reconstruction.
    """
    from alphalattice.kernel.data_layer.knowledge import fuse
    from alphalattice.kernel.data_layer.log import replay
    from alphalattice.kernel.data_layer.project import differences

    root = tmp_path / "source"
    create_data_root(root)
    metadata = {"document": {"document_id": "reference.guide"}, "revision": 1}
    snapshot = {"snapshot_id": "retained-snapshot-name", "revisions": [metadata]}
    with HostLog.open(root) as log:
        log.record("workspace.created", ("workspace", "w1"), {"name": "first"})

        def visible(cutoff: int) -> frozenset[tuple[str, str]]:
            return frozenset(
                tuple(row)
                for row in log.read(
                    "SELECT s.source_id, v.version FROM knowledge_source_version v"
                    " JOIN knowledge_source s USING (source_id) WHERE s.kind = 'evidence'"
                    " AND v.known_us <= ? AND (v.superseded_us IS NULL OR v.superseded_us > ?)",
                    (cutoff, cutoff),
                )
            )

        store = KnowledgeStore(log, evidence_asof=visible)
        version_id = store.publish_revision(
            workspace_id="w1", metadata=metadata, content="Keep the source of each conclusion."
        )
        store.publish_snapshot(workspace_id="w1", metadata=snapshot)
        original = store.read_revision("w1", "reference.guide", 1)
        assert original.metadata == metadata
        assert original.content == "Keep the source of each conclusion."
        assert UUID(original.origin.source_id).version == 7
        assert UUID(version_id).version == 7
        assert store.read_snapshot("w1", snapshot["snapshot_id"]) == snapshot
        assert (
            store.publish_revision(workspace_id="w1", metadata=metadata, content=original.content)
            == version_id
        )
        log.record("workspace.created", ("workspace", "w2"), {"name": "second"})
        external = {"document": {"document_id": "filing.reference"}, "revision": 1}
        origin = SourceOrigin(uuid7(), uuid7(), "evidence", 123)
        body = "The canonical evidence body stays in the world store. 市场风险已经增加。"
        first = store.publish_revision(
            workspace_id="w1", metadata=external, content=body, origin=origin
        )
        assert (
            store.publish_revision(
                workspace_id="w2", metadata=external, content=body, origin=origin
            )
            == first
        )
        alias = {"document": {"document_id": "filing.local-alias"}, "revision": 2}
        assert (
            store.publish_revision(workspace_id="w2", metadata=alias, content=body, origin=origin)
            == first
        )
        assert store.read_revision("w2", "filing.local-alias", 2).metadata == alias
        alias_snapshot = {"snapshot_id": "alias-snapshot", "revisions": [alias]}
        store.publish_snapshot(workspace_id="w2", metadata=alias_snapshot)
        assert store.read_snapshot("w2", "alias-snapshot") == alias_snapshot
        spec = {**recall._recipe(), "capacity": 80}
        recipe = store.recipe(spec)
        legacy_recipe = store.recipe(
            {**spec, "splitter": "semantic-text-splitter", "version": "0.33.3"}
        )
        assert recipe != legacy_recipe
        with pytest.raises(DataError, match="index under a new recipe"):
            store.index_chunks(version_id=first, recipe_id=legacy_recipe, content=body)
        chunks = store.index_chunks(version_id=first, recipe_id=recipe, content=body)
        with log.reading() as con:
            assert (
                con.execute("SELECT v FROM knowledge_fts_config WHERE k = 'automerge'").fetchone()[
                    0
                ]
                == 0
            )
        store.optimize()
        assert chunks == store.index_chunks(version_id=first, recipe_id=recipe, content=body)
        assert all(
            digest(body[chunk.start : chunk.end].encode()) == chunk.digest for chunk in chunks
        )
        assert [
            hit.source_id
            for hit in store.lexical("市场风险", store.scope(workspace="w2", cutoff_us=123))
        ] == [origin.source_id]
        assert [
            hit.source_id
            for hit in store.lexical("风险市场", store.scope(workspace="w2", cutoff_us=123))
        ] == [origin.source_id]
        # In a large index a phrase no chunk holds is matched only by its characters fewer than
        # half the chunks hold; here every chunk that has Chinese holds them all. The precise
        # mode's per-source window and a sealed read (evidence review's references) match every
        # character, as their inputs name no index-wide count.
        sealed = SourceOrigin(uuid7(), uuid7(), knowledge.SEALED_KIND, 123)
        sealed_body = "A reference read as evidence. 市场风险已经增加。"
        sealed_id = store.publish_revision(
            workspace_id="w2",
            metadata={"document": {"document_id": "filing.sealed"}, "revision": 1},
            content=sealed_body,
            origin=sealed,
        )
        sealed_chunks = store.index_chunks(
            version_id=sealed_id, recipe_id=recipe, content=sealed_body
        )
        reading = {"workspace": "w2", "cutoff_us": 123}
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(knowledge, "COMMON_WORDS_FROM", 1)
            assert store.lexical("风险市场", store.scope(kind="evidence", **reading)) == ()
            assert [
                hit.source_id
                for hit in store.lexical(
                    "风险市场", store.scope(kind="evidence", **reading), per_source=True
                )
            ] == [origin.source_id]
            assert [
                hit.source_id
                for hit in store.lexical(
                    "风险市场", store.scope(kind=knowledge.SEALED_KIND, **reading)
                )
            ] == [sealed.source_id]
            assert [
                hit.source_id
                for hit in store.lexical("市场风险", store.scope(kind="evidence", **reading))
            ] == [origin.source_id]
        assert store.lexical("市场风险", store.scope(workspace="w2", cutoff_us=122)) == ()
        with pytest.raises(DataError, match="source"):
            store.index_chunks(version_id="", recipe_id=recipe, content=body)
        later = SourceOrigin(origin.source_id, uuid7(), "evidence", 300)
        later_metadata = {"document": {"document_id": "filing.reference"}, "revision": 3}
        later_body = "市场风险已下降。"
        later_id = store.publish_revision(
            workspace_id="w1", metadata=later_metadata, content=later_body, origin=later
        )
        store.index_chunks(version_id=later_id, recipe_id=recipe, content=later_body)
        intermediate = SourceOrigin(origin.source_id, uuid7(), "evidence", 200)
        middle_metadata = {"document": {"document_id": "filing.reference"}, "revision": 2}
        middle_body = "市场风险维持不变。"
        middle_id = store.publish_revision(
            workspace_id="w1", metadata=middle_metadata, content=middle_body, origin=intermediate
        )
        store.index_chunks(version_id=middle_id, recipe_id=recipe, content=middle_body)
        assert [
            hit.version
            for hit in store.lexical("市场风险", store.scope(workspace="w1", cutoff_us=199))
        ] == [origin.version]
        assert [
            hit.version
            for hit in store.lexical("市场风险", store.scope(workspace="w1", cutoff_us=200))
        ] == [intermediate.version]
        assert [
            hit.version
            for hit in store.lexical("市场风险", store.scope(workspace="w1", cutoff_us=300))
        ] == [later.version]
        all_chunks = (
            *chunks,
            *sealed_chunks,
            *store.index_chunks(version_id=middle_id, recipe_id=recipe, content=middle_body),
            *store.index_chunks(version_id=later_id, recipe_id=recipe, content=later_body),
        )
        model = store.model({"graph": "fixture.onnx"}, dimension=3)
        keys = tuple(chunk.digest for chunk in all_chunks)
        store.embeddings(
            keys,
            recipe_id=recipe,
            model_id=model,
            encode=lambda missing: np.asarray([(1.0, 0.0, 0.0)] * len(missing), dtype=np.float32),
        )
        with store.vector_index(recipe_id=recipe, model_id=model, directory=tmp_path) as index:
            assert not index.matrix.flags.writeable
            query_vector = np.asarray([1, 0, 0], dtype=np.float32)
            first_hit = index.search(query_vector, store.scope(workspace="w1", cutoff_us=123))[0]
            middle_hit = index.search(query_vector, store.scope(workspace="w1", cutoff_us=200))[0]
            later_hit = index.search(query_vector, store.scope(workspace="w1", cutoff_us=300))[0]
            assert middle_hit.version == intermediate.version
            fused = fuse(
                (replace(first_hit, score=5), replace(middle_hit, score=10)),
                (replace(first_hit, score=10), replace(later_hit, score=5)),
            )
            assert [(hit.version, hit.score) for hit in fused] == [
                (origin.version, 0.5),
                (intermediate.version, 0.5),
                (later.version, 0.0),
            ]
            # Each fused hit names the channels that found it.
            assert [hit.channels for hit in fused] == [
                ("lexical", "vector"),
                ("lexical",),
                ("vector",),
            ]
            # A range scope admits only the chunks inside it, in both channels.
            inside = replace(
                store.scope(workspace="w1", cutoff_us=123),
                ranges=((origin.source_id, first_hit.ordinal, first_hit.ordinal),),
            )
            outside = replace(inside, ranges=((origin.source_id, 999, 999),))
            assert [hit.chunk for hit in index.search(query_vector, inside)] == [first_hit.chunk]
            assert index.search(query_vector, outside) == ()
            assert {hit.ordinal for hit in store.lexical("evidence", inside)} <= {first_hit.ordinal}
            assert store.lexical("evidence", outside) == ()
            # A per-source window is each source's best, in one order.
            assert len(index.search(query_vector, inside, limit=1, per_source=True)) == 1
            newest = SourceOrigin(origin.source_id, uuid7(), "evidence", 400)
            newest_id = store.publish_revision(
                workspace_id="w1",
                metadata=later_metadata | {"revision": 4},
                content="市场风险再次增加。",
                origin=newest,
            )
            newest_chunks = store.index_chunks(
                version_id=newest_id, recipe_id=recipe, content="市场风险再次增加。"
            )
            store.embeddings(
                tuple(chunk.digest for chunk in newest_chunks),
                recipe_id=recipe,
                model_id=model,
                encode=lambda missing: np.asarray(
                    [(1.0, 0.0, 0.0)] * len(missing), dtype=np.float32
                ),
            )
            assert index.refresh()
            assert (
                index.search(query_vector, store.scope(workspace="w1", cutoff_us=400))[0].version
                == newest.version
            )
            assert (
                index.search(
                    np.asarray([1, 0, 0], dtype=np.float32),
                    store.scope(workspace="other", cutoff_us=200),
                )
                == ()
            )

            class Encoder:
                def embed_query(self, _: str) -> np.ndarray:
                    return query_vector

            class Reranker:
                calls = 0
                deadline = True

                def score(self, _: str, passages: tuple[str, ...]) -> list[float] | None:
                    self.calls += 1
                    assert passages
                    return None if self.deadline else [0.25] * len(passages)

            bodies = {
                origin.version: body,
                intermediate.version: middle_body,
                later.version: later_body,
                newest.version: "市场风险再次增加。",
                sealed.version: sealed_body,
            }
            other_recipe = store.recipe({**recall._recipe(), "capacity": 24})
            other_chunks = store.index_chunks(
                version_id=first, recipe_id=other_recipe, content=body
            )
            assert other_chunks != chunks
            lexical = store.lexical(
                "evidence", store.scope(workspace="w1", cutoff_us=123), recipe_id=recipe
            )
            assert lexical and all(
                hit.chunk.chunk_id in {chunk.chunk_id for chunk in chunks} for hit in lexical
            )
            reranker = Reranker()
            search = HybridSearch(
                store, index, Encoder(), reranker, lambda hit, _: bodies[hit.version]
            )
            chinese = search.retrieve("市场风险", language="zh", workspace="w1", cutoff_us=400)
            assert chinese.hits[0].hit.version == newest.version
            assert chinese.hits[0].text == bodies[newest.version]
            assert chinese.score_type == "LINEAR_FUSION"
            assert reranker.calls == 0
            # The default path is the fused order: no cross-encoder call.
            english = search.retrieve("evidence", language="en", workspace="w1", cutoff_us=123)
            assert english.hits[0].hit.version == origin.version
            assert all(
                hit.hit.chunk.chunk_id in {chunk.chunk_id for chunk in chunks}
                for hit in english.hits
            )
            assert english.score_type == "LINEAR_FUSION"
            assert reranker.calls == 0
            # A `depth` asks for the precise mode: English reranked, the fused order standing
            # past the deadline.
            precise = {"language": "en", "workspace": "w1", "cutoff_us": 123, "depth": 5}
            assert search.retrieve("evidence", **precise).score_type == "LINEAR_FUSION"
            assert reranker.calls == 1
            reranker.deadline = False
            english = search.retrieve("evidence", **precise)
            assert english.score_type == "CROSS_ENCODER_LOGIT"
            assert english.hits[0].hit.score == 0.25
            # A sealed read reranks English without a depth, as evidence review's reads did.
            calls = reranker.calls
            sealed_read = search.retrieve(
                "evidence", language="en", kind=knowledge.SEALED_KIND, **reading
            )
            assert sealed_read.score_type == "CROSS_ENCODER_LOGIT"
            assert [hit.hit.source_id for hit in sealed_read.hits] == [sealed.source_id]
            assert reranker.calls == calls + 1
            # Read per document: each source's best `depth` candidates are reranked, and each
            # group of sources takes its own `limit` of that order.
            calls = reranker.calls
            deep = search.retrieve(
                "evidence",
                language="en",
                workspace="w1",
                cutoff_us=123,
                limit=1,
                depth=1,
                groups=((origin.source_id,), ("another source",)),
            )
            assert [hit.hit.source_id for hit in deep.hits] == [origin.source_id]
            assert deep.score_type == "CROSS_ENCODER_LOGIT" and reranker.calls == calls + 1
            scoped = search.retrieve(
                "evidence",
                language="en",
                workspace="w1",
                cutoff_us=123,
                ranges=((origin.source_id, 999, 999),),
            )
            assert scoped.hits == ()
            with pytest.raises(DataError, match="query"):
                search.retrieve("x" * 513, language="en", workspace="w1", cutoff_us=123)
            retirement = store.retire(origin.source_id, known_us=500)
            assert index.refresh()
            assert index.search(query_vector, store.scope(workspace="w1", cutoff_us=500)) == ()
            assert (
                index.search(
                    query_vector, store.scope(workspace="w1", cutoff_us=500, superseded=True)
                )
                == ()
            )
        assert store.retire(origin.source_id, known_us=500) == retirement
        assert [
            hit.version
            for hit in store.lexical("市场风险", store.scope(workspace="w1", cutoff_us=499))
        ] == [newest.version]
        assert store.lexical("市场风险", store.scope(workspace="w1", cutoff_us=500)) == ()
        assert (
            store.lexical("市场风险", store.scope(workspace="w1", cutoff_us=500, superseded=True))
            == ()
        )
        late = SourceOrigin(origin.source_id, uuid7(), "evidence", 450)
        late_id = store.publish_revision(
            workspace_id="w1",
            metadata=later_metadata | {"revision": 5},
            content="A source learned out of order.",
            origin=late,
        )
        with log.reading() as con:
            assert (
                con.execute(
                    "SELECT superseded_us FROM knowledge_source_version WHERE version_id = ?",
                    (late_id,),
                ).fetchone()[0]
                == 500
            )
        wrong_kind = SourceOrigin(origin.source_id, uuid7(), "experience", 500)
        with pytest.raises(DataError, match="source"):
            store.publish_revision(
                workspace_id="w1",
                metadata=later_metadata | {"revision": 6},
                content=later_body,
                origin=wrong_kind,
            )
        external_read = store.read_revision("w2", "filing.reference", 1)
        assert external_read.origin == origin and external_read.content is None
        with log.reading() as con:
            columns = {row[1] for row in con.execute("PRAGMA table_info(knowledge_chunk)")}
            assert "text" not in columns and "fts_text" not in columns
            assert (
                con.execute(
                    "SELECT count(*) FROM event_text WHERE instr(text, ?) > 0", (body,)
                ).fetchone()[0]
                == 0
            )
        log.flush()
        copy = tmp_path / "replayed"
        (copy / "machine").mkdir(parents=True)
        with (
            log.reading() as source,
            closing(replay(source, copy / "machine/log.sqlite")) as target,
        ):
            assert differences(source, target) == []
    with HostLog.open(copy) as log:
        restored = KnowledgeStore(log)
        assert restored.read_revision("w1", "reference.guide", 1) == original
        assert restored.read_revision("w2", "filing.reference", 1) == external_read
        assert restored.read_revision("w2", "filing.local-alias", 2).metadata == alias
        assert restored.read_snapshot("w2", "alias-snapshot") == alias_snapshot
        assert restored.read_snapshot("w1", snapshot["snapshot_id"]) == snapshot


# The retrieval store's speed target, read from the nightly: at 100,000 chunks on four cores, a
# query's p95 is under one second in English and in Chinese, cold and warm, with the real models.

CHUNKS = 100_000
BUDGET = 1.0
EN = [
    "margin",
    "revenue",
    "guidance",
    "quarter",
    "costs",
    "pricing",
    "demand",
    "supply",
    "inventory",
    "debt",
    "equity",
    "cash",
    "flow",
    "dividend",
    "buyback",
    "risk",
    "exposure",
    "hedge",
    "rate",
    "inflation",
    "momentum",
    "value",
    "growth",
    "volatility",
    "factor",
    "earnings",
    "outlook",
    "segment",
    "acquisition",
    "impairment",
    "litigation",
    "regulator",
    "tariff",
    "currency",
    "China",
    "semiconductor",
    "energy",
    "bank",
    "insurer",
    "retail",
    "consumer",
    "software",
    "cloud",
    "capital",
    "expenditure",
]
ZH = list(
    "市场风险收益利润营收成本价格需求供给库存债务股权现金流分红回购敞口对冲利率通胀动量价值增长"
    "波动因子盈利展望并购减值诉讼监管关税汇率半导体能源银行保险零售消费软件资本开支"
)


def _corpus(rng: random.Random) -> str:
    """One source of mixed English and Chinese sentences: few distinct words, so each FTS term
    has a long posting list, slower than real filings."""
    sentences = []
    for _ in range(160):
        if rng.random() < 0.5:
            words = " ".join(rng.choice(EN) for _ in range(rng.randint(8, 18)))
            sentences.append(words.capitalize() + ". ")
        else:
            sentences.append("".join(rng.choice(ZH) for _ in range(rng.randint(12, 30))) + "。")
    return "".join(sentences)


@pytest.mark.proves("data_layer.recall_meets_its_speed_at_100k")
def test_recall_answers_within_a_second_at_100k_chunks(tmp_path: Path) -> None:
    """With the installed packs, a store of 100,000 chunks answers its first query in each
    language and the p95 of thirty more within one second, English reranked."""
    encoder_pack, reranker_pack, manifest = recall._packs()  # an absent pack is red, never a skip
    root, rng, bodies = tmp_path / "data", random.Random(7), {}
    create_data_root(root)
    with HostLog.open(root) as log:
        log.record("workspace.created", ("workspace", "w1"), {"name": "w1"})
        store, encoder = KnowledgeStore(log), StaticEncoder(encoder_pack)
        recipe = store.recipe(recall._recipe())
        model = store.model(manifest, dimension=512, precision=recall.PRECISION)
        held = 0
        while held < CHUNKS:
            body, source = _corpus(rng), str(uuid7())
            bodies[source] = body
            version_id = store.publish_revision(
                workspace_id="w1",
                metadata={"document": {"document_id": source}, "revision": 1},
                content=body,
                origin=SourceOrigin(source, "1", "reference", 1),
            )
            chunks = store.index_chunks(version_id=version_id, recipe_id=recipe, content=body)
            texts = {chunk.digest: body[chunk.start : chunk.end] for chunk in chunks}
            store.embeddings(
                tuple(texts),
                recipe_id=recipe,
                model_id=model,
                encode=lambda missing, texts=texts: encoder.embed_passages(
                    [texts[d] for d in missing]
                ),
            )
            held += len(chunks)
        store.optimize()
        with store.vector_index(recipe_id=recipe, model_id=model, directory=root) as index:
            search = HybridSearch(
                store, index, encoder, Reranker(reranker_pack), lambda hit, _: bodies[hit.source_id]
            )
            measured: dict[str, float] = {}
            for language, words, joined in (("en", EN, " "), ("zh", ZH, "")):
                seconds = []
                for _ in range(31):
                    query = joined.join(rng.choice(words) for _ in range(rng.randint(2, 6)))
                    started = time.perf_counter()
                    found = search.retrieve(
                        query, language=language, workspace="w1", cutoff_us=2**62
                    )
                    seconds.append(time.perf_counter() - started)
                    assert found.hits
                measured[f"{language} cold"] = seconds[0]
                measured[f"{language} warm p95"] = sorted(seconds[1:])[28]
    # One verdict naming every measured second, so a nightly red reads as numbers.
    assert all(value < BUDGET for value in measured.values()), {
        name: round(value, 3) for name, value in measured.items()
    }
