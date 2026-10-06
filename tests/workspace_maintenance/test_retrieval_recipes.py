"""Five retrieval recipes are admitted by name; the retained one keeps every
hash it was published under.

A recipe is one whole payload (encoder, reranker, runtime, input policy,
pooling, dimension), never a free pairing. The new fields are absent from a
payload at their CPU defaults, so the `hybrid-v2-minilm` spec hash, its
embedding context hash and the capability report a workspace manifest binds
are the values they were before recipes existed; two recipes that share an
encoder share the embedding context, so their vectors are one asset; a torch
pack is proved as a whole closure; and a reranker's exact pair scores are
answered from a bounded cache that never holds work that did not complete.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from alphalattice.kernel.knowledge import _embeddings as embeddings
from alphalattice.kernel.knowledge import _torch as torch_runtime
from alphalattice.kernel.knowledge.hybrid_contracts import (
    EMBEDDING_CANARIES,
    RECIPE_BGE_SMALL_CPU,
    RECIPE_MINILM_CPU,
    RECIPE_QWEN3_ENCODER_GPU,
    RECIPE_QWEN3_GPU,
    RECIPE_QWEN3_RERANKER_GPU,
    RERANKER_CANARIES,
    RETRIEVAL_CANARY_PASSAGE,
    RETRIEVAL_CANARY_QUERY,
    SUPPORTED_RECIPES,
    EmbeddingInputPolicy,
    HybridCapabilityReport,
    HybridIndexSpec,
    embedding_context_hash,
    reranker_context_hash,
)
from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError
from alphalattice.kernel.shared_kernel.domain.serialization import canonical_json_bytes, sha256_hex

# The recipe's identity since the final close-out (F1): no execution field.
V2_SPEC_HASH = "e52d05c1bfb5e285c4243c298e0215c0e74356671673167a5f173c7b14dd456a"
# The identity every index and generation before F1 was built under, naming the
# four intra-op threads of the host it was sealed on: the index databases of the
# real-material QA copies are named by it, and it reads back as sealed.
LEGACY_V2_SPEC_HASH = "6b3d391490a72f1857eca2336ee50afcc615e900feb2a69640e26a47f8a36c9a"
LEGACY_V2_EMBEDDING_CONTEXT = "2fdc2c5f41264c41"
# The capability hash the retained real-material authority manifest binds
# (`semantic_capability_hash` in evidence-review-authority.json of the eight
# issuer package): the probe's report with the reranker's details folded in.
BOUND_CAPABILITY_HASH = "508b3cd1b9c3910c499c7b3f16b66cce01f648151946f6fe10c210de4dcc412c"


def test_the_retained_recipe_keeps_its_hashes() -> None:
    spec = HybridIndexSpec.fixed_v2()
    assert spec.policy_id == RECIPE_MINILM_CPU
    assert spec.logical_hash == V2_SPEC_HASH
    assert HybridIndexSpec.for_recipe(RECIPE_MINILM_CPU).logical_hash == V2_SPEC_HASH
    dumped = spec.model_dump(mode="python")
    for name in (
        "encoder_runtime",
        "encoder_pack_sha256",
        "reranker_runtime",
        "reranker_maximum_tokens",
        "encoder_torch_version",
        "reranker_torch_version",
    ):
        assert name not in dumped, name
    report = HybridCapabilityReport.ready(spec)
    details = tuple(
        sorted(
            (
                *report.details,
                "reranker-pack:5-artifacts-verified",
                "fastembed:verified",
                "tokenizers:verified",
            )
        )
    )
    payload = {**report.model_dump(mode="python", exclude={"logical_hash"}), "details": details}
    bound = HybridCapabilityReport.model_validate(
        {**payload, "logical_hash": sha256_hex(canonical_json_bytes(payload))}
    )
    assert bound.logical_hash == BOUND_CAPABILITY_HASH
    assert bound.cpu_execution_provider_only


def test_the_threads_left_the_identity_and_a_spec_that_named_them_reads_back() -> None:
    """requirement (the final close-out, F1): a spec names what decides a result,
    never how many threads compute it; a spec sealed before F1 with its threads
    parses with its own hash, states the recipe's spec without them, and its
    encoder's canary is sealed under that context."""

    spec = HybridIndexSpec.fixed_v2()
    assert not spec.names_execution
    for name in (
        "execution_mode",
        "inter_op_num_threads",
        "intra_op_num_threads",
        "intra_op_spinning",
        "inter_op_spinning",
    ):
        assert name not in spec.model_dump(mode="json"), name
    legacy = HybridIndexSpec.sealed_before_f1(RECIPE_MINILM_CPU, 4)
    assert legacy.names_execution and legacy.logical_hash == LEGACY_V2_SPEC_HASH
    assert HybridIndexSpec.model_validate_json(legacy.model_dump_json()) == legacy
    assert embedding_context_hash(legacy).startswith(LEGACY_V2_EMBEDDING_CONTEXT)
    assert legacy.without_execution() == spec
    assert embedding_context_hash(spec) in EMBEDDING_CANARIES
    assert reranker_context_hash(spec) in RERANKER_CANARIES
    assert reranker_context_hash(legacy.without_execution()) in RERANKER_CANARIES


def test_every_recipe_is_one_admitted_payload_and_nothing_else() -> None:
    specs = {recipe: HybridIndexSpec.for_recipe(recipe) for recipe in SUPPORTED_RECIPES}
    assert len({spec.logical_hash for spec in specs.values()}) == len(SUPPORTED_RECIPES)
    with pytest.raises(ValueError, match="unsupported retrieval recipe"):
        HybridIndexSpec.for_recipe("hybrid-v3-anything-goes")
    # A payload under one name with another's model is refused: the identity
    # is the whole payload.
    bge = specs[RECIPE_BGE_SMALL_CPU]
    forged = {**bge.model_dump(mode="python"), "model_id": specs[RECIPE_MINILM_CPU].model_id}
    forged.pop("logical_hash")
    with pytest.raises(ValueError, match="must match the recipe"):
        HybridIndexSpec.model_validate(
            {**forged, "logical_hash": sha256_hex(canonical_json_bytes(forged))}
        )


def test_recipes_that_share_an_encoder_share_the_embedding_context() -> None:
    v2 = HybridIndexSpec.for_recipe(RECIPE_MINILM_CPU)
    reranker_only = HybridIndexSpec.for_recipe(RECIPE_QWEN3_RERANKER_GPU)
    encoder_only = HybridIndexSpec.for_recipe(RECIPE_QWEN3_ENCODER_GPU)
    both = HybridIndexSpec.for_recipe(RECIPE_QWEN3_GPU)
    bge = HybridIndexSpec.for_recipe(RECIPE_BGE_SMALL_CPU)
    # Same encoder, other reranker: the vectors are the same asset.
    assert embedding_context_hash(reranker_only) == embedding_context_hash(v2)
    assert embedding_context_hash(both) == embedding_context_hash(encoder_only)
    # Another encoder is another context.
    assert len({embedding_context_hash(s) for s in (v2, encoder_only, bge)}) == 3
    assert reranker_only.logical_hash != v2.logical_hash
    assert encoder_only.embedding_dimension == 1024 and bge.embedding_dimension == 384


def test_a_gpu_recipe_reports_its_provider_truthfully() -> None:
    for recipe in (RECIPE_QWEN3_ENCODER_GPU, RECIPE_QWEN3_RERANKER_GPU, RECIPE_QWEN3_GPU):
        report = HybridCapabilityReport.ready(HybridIndexSpec.for_recipe(recipe))
        assert not report.cpu_execution_provider_only
    cpu = HybridCapabilityReport.ready(HybridIndexSpec.for_recipe(RECIPE_BGE_SMALL_CPU))
    assert cpu.cpu_execution_provider_only
    payload = cpu.model_dump(mode="python", exclude={"logical_hash"})
    payload["cpu_execution_provider_only"] = False
    with pytest.raises(ValueError, match="truthfully"):
        HybridCapabilityReport.model_validate(
            {**payload, "logical_hash": sha256_hex(canonical_json_bytes(payload))}
        )


def test_the_input_policies_prefix_queries_only() -> None:
    prepare = embeddings.prepare_embedding_input
    text = "Was a lawsuit filed?"
    for policy, prefix in (
        (EmbeddingInputPolicy.BGE_QUERY_INSTRUCTION_V1, embeddings.BGE_QUERY_INSTRUCTION),
        (EmbeddingInputPolicy.QWEN3_RETRIEVAL_INSTRUCT_V1, embeddings.QWEN3_QUERY_INSTRUCTION),
    ):
        assert prepare(text, policy=policy, is_query=True) == prefix + text
        assert prepare(text, policy=policy, is_query=False) == text


def _manifest(root: Path, files: dict[str, bytes]) -> tuple[tuple[tuple[str, str], ...], str]:
    entries = []
    for name, payload in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        entries.append((name, hashlib.sha256(payload).hexdigest()))
    manifest = tuple(sorted(entries))
    return manifest, sha256_hex(canonical_json_bytes(dict(manifest)))


def test_a_torch_pack_is_proved_as_a_whole_closure(tmp_path: Path) -> None:
    manifest, digest = _manifest(
        tmp_path, {"config.json": b"{}", "model.safetensors": b"weights", "tokenizer.json": b"{}"}
    )
    torch_runtime.verify_pack(tmp_path, manifest, digest)
    (tmp_path / "extra.py").write_bytes(b"print('hi')")
    with pytest.raises(KnowledgeRetrievalError, match=r"unadmitted artifact: extra\.py"):
        torch_runtime.verify_pack(tmp_path, manifest, digest)
    (tmp_path / "extra.py").unlink()
    (tmp_path / "config.json").write_bytes(b'{"tampered": true}')
    with pytest.raises(KnowledgeRetrievalError, match=r"differs from the index spec: config\.json"):
        torch_runtime.verify_pack(tmp_path, manifest, digest)
    (tmp_path / "config.json").unlink()
    with pytest.raises(KnowledgeRetrievalError, match=r"missing a consumed artifact: config\.json"):
        torch_runtime.verify_pack(tmp_path, manifest, digest)


def test_a_missing_gpu_runtime_is_a_named_refusal_not_a_fallback() -> None:
    """The default environment holds no torch; the recipe names the setup step."""

    pytest.importorskip("onnxruntime")
    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(KnowledgeRetrievalError, match="create_gpu_environment"):
            torch_runtime.runtime_for("2.14.0+cu130", "5.17.0", "13.0")
    else:
        pytest.skip("torch is installed in this environment")


def test_an_unusable_runtime_is_named_in_the_capability_and_the_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement: incompatible or lost hardware is explicit, for both GPU
    recipe shapes.

    The reranker-only GPU recipe verified its pack and reported READY while
    the device was lost, failing at the first pair scored; and an unavailable
    report said only that, not why. Now the torch reranker's verification
    proves the runtime as the encoder's probe does, and the report carries the
    probe's own words, which the Host's refusal repeats.
    """

    from alphalattice.kernel.knowledge import _reranking as reranking
    from alphalattice.kernel.knowledge import hybrid

    calls: list[tuple[str | None, ...]] = []

    def unusable(*versions: str | None) -> tuple[object, object]:
        calls.append(versions)
        raise KnowledgeRetrievalError(
            "no CUDA device is available for the GPU retrieval recipe (lost)",
            code="retrieval.semantic_pack_unavailable",
            retryable=True,
        )

    monkeypatch.setattr(torch_runtime, "runtime_for", unusable)
    monkeypatch.setattr(torch_runtime, "verify_pack", lambda *_: None)
    spec = HybridIndexSpec.for_recipe(RECIPE_QWEN3_RERANKER_GPU)
    with pytest.raises(KnowledgeRetrievalError, match="no CUDA device"):
        reranking.verify_reranker_pack(tmp_path, spec)
    versions = (
        spec.reranker_torch_version,
        spec.reranker_transformers_version,
        spec.reranker_cuda_version,
    )
    assert calls == [versions], "the reranker's own locked versions are what is proved"

    report = hybrid._semantic_unavailable_capability(spec, cause="no CUDA device (lost)")
    assert str(report.status) == "SEMANTIC_PACK_UNAVAILABLE"
    assert report.details == ("cause:no CUDA device (lost)", "semantic-pack:unavailable")
    # The retained CPU recipe's report is untouched by the cause (its hash is bound).
    assert "cause:" not in " ".join(hybrid._semantic_unavailable_capability(spec).details)


def test_the_installer_lets_the_stores_typed_refusal_through(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """requirement: a cancelled, bad or tampered install is unselectable, by
    name -- the store's code reaches the person, not a generic setup failure."""

    import json

    from scripts import materialize_evidence_cro_authority as setup

    def refused(_arguments: object) -> dict[str, object]:
        raise ValueError("model_store.pack_not_installed:bge-small-en-v1.5:TAMPERED")

    monkeypatch.setattr(setup, "materialize", refused)
    assert setup.main(["--workspace", "x"]) == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "REFUSED"
    assert printed["failure_code"] == "model_store.pack_not_installed:bge-small-en-v1.5:TAMPERED"


def test_pair_scores_are_answered_from_a_bounded_cache() -> None:
    cache = embeddings.PairScoreCache(limit=3)
    calls: list[list[str]] = []

    def infer(pending: Any) -> list[float]:
        calls.append(list(pending))
        return [float(len(p)) for p in pending]

    ledger = embeddings.MODEL_WORK
    hits_before = ledger.reranker_pair_cache_hits
    first = embeddings.score_with_cache(cache, "q", ("aa", "bbb", "c"), infer)
    assert first == (2.0, 3.0, 1.0) and calls == [["aa", "bbb", "c"]]
    # A repeat asks nothing of the model and keeps the order of the request.
    again = embeddings.score_with_cache(cache, "q", ("c", "aa", "bbb"), infer)
    assert again == (1.0, 2.0, 3.0) and len(calls) == 1
    assert ledger.reranker_pair_cache_hits - hits_before == 3
    # A new pair is scored once, in a batch of the misses only; the query is
    # part of the key.
    mixed = embeddings.score_with_cache(cache, "q", ("aa", "dddd"), infer)
    assert mixed == (2.0, 4.0) and calls[-1] == ["dddd"]
    other = embeddings.score_with_cache(cache, "other question", ("aa",), infer)
    assert other == (2.0,) and calls[-1] == ["aa"]
    # Bounded: the oldest entries left, so the cache never grows past its limit.
    assert len(cache) == 3

    def failing(pending: Any) -> list[float]:
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError):
        embeddings.score_with_cache(embeddings.PairScoreCache(), "q", ("zz",), failing)
    assert embeddings.score_with_cache(cache, "q", ("zz",), infer) == (2.0,)


def test_a_session_whose_canary_moved_is_refused_by_name_with_its_threads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """requirement (the final close-out, F1): a cross-encoder session scores the
    sealed canary pair when it loads; the sealed raw score loads it, any other
    refuses by name with the threads it ran on."""

    import importlib
    from types import SimpleNamespace

    from alphalattice.kernel.knowledge import _reranking as reranking

    scores = iter((-4.206943035125732, -4.2069))

    class _Encoder:
        def __init__(self, *_args: Any, threads: int, **_kwargs: Any) -> None:
            self.threads = threads

        def rerank(self, query: str, passages: list[str], batch_size: int) -> list[float]:
            assert (query, passages) == (RETRIEVAL_CANARY_QUERY, [RETRIEVAL_CANARY_PASSAGE])
            return [next(scores)]

    load = importlib.import_module
    monkeypatch.setattr(reranking, "verify_reranker_pack", lambda *_args: None)
    monkeypatch.setattr(
        reranking.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(TextCrossEncoder=_Encoder)
            if name == "fastembed.rerank.cross_encoder"
            else load(name)
        ),
    )
    spec = HybridIndexSpec.fixed_v2()
    assert reranking.LocalCrossEncoderAdapter(tmp_path, spec, threads=8).threads == 8
    with pytest.raises(KnowledgeRetrievalError, match="at 8 intra-op threads") as refused:
        reranking.LocalCrossEncoderAdapter(tmp_path, spec, threads=8)
    assert refused.value.failure.code == "retrieval.execution_canary_mismatch"
