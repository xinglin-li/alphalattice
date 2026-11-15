"""The int8 retrieval recipe: the retained models' upstream int8 graphs, proved live.

`hybrid-v3-minilm-int8` names the retained recipe's two pinned models through their repositories'
own int8 graphs, installed beside the fp32 packs. Integer kernels differ by instruction set, so no
sealed canary holds its numbers on every machine: a session proves at open that its threads give
the numbers one thread gives here. A new Evidence authority is set up with it; an installed one
keeps its own recipe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alphalattice.kernel.knowledge import model_store
from alphalattice.kernel.knowledge.hybrid_contracts import (
    EMBEDDING_CANARIES,
    MODEL_INT8_GRAPH,
    NEW_BUILD_RECIPE,
    RECIPE_MINILM_CPU,
    RECIPE_MINILM_INT8_CPU,
    RERANKER_CANARIES,
    RERANKER_INT8_PACK,
    RERANKER_PACK,
    SUPPORTED_RECIPES,
    HybridIndexSpec,
    canary_to_prove,
    embedding_context_hash,
    encoder_graph,
    live_canary,
    machine_class,
    recipe_payload,
    reranker_context_hash,
    reranker_pack_files,
    setup_recipe,
)
from alphalattice.kernel.shared_kernel.domain.serialization import canonical_json_bytes, sha256_hex

TEST_SOURCE = "src/alphalattice/kernel/knowledge/hybrid_contracts.py"


def _identity(payload: dict[str, object]) -> str:
    return sha256_hex(canonical_json_bytes(payload))


INT8 = HybridIndexSpec.for_recipe(RECIPE_MINILM_INT8_CPU)
RETAINED = HybridIndexSpec.fixed_v2()


@pytest.mark.proves("knowledge.int8_recipe_is_the_retained_models_int8_graphs")
def test_the_int8_recipe_is_the_retained_models_int8_graphs_in_their_own_directories() -> None:
    same = (
        "model_id",
        "model_revision",
        "tokenizer_sha256",
        "reranker_model_id",
        "embedding_dimension",
    )
    assert all(getattr(INT8, name) == getattr(RETAINED, name) for name in same)
    assert (INT8.quantization, RETAINED.quantization) == ("INT8_DYNAMIC", "NONE")
    assert (encoder_graph(INT8), encoder_graph(RETAINED)) == (MODEL_INT8_GRAPH, "onnx/model.onnx")
    assert (reranker_pack_files(INT8), reranker_pack_files(RETAINED)) == (
        RERANKER_INT8_PACK,
        RERANKER_PACK,
    )
    # Its numbers are another asset, which no sealed canary claims and its machine class names.
    assert embedding_context_hash(INT8) != embedding_context_hash(RETAINED)
    assert embedding_context_hash(INT8) not in EMBEDDING_CANARIES
    assert reranker_context_hash(INT8) not in RERANKER_CANARIES
    specs = [HybridIndexSpec.for_recipe(recipe) for recipe in SUPPORTED_RECIPES]
    assert [spec.policy_id for spec in specs if live_canary(spec)] == [RECIPE_MINILM_INT8_CPU]
    assert [spec.policy_id for spec in specs if spec.execution_class] == [RECIPE_MINILM_INT8_CPU]
    assert INT8.execution_class == machine_class()
    # Another machine class's spec reads back under its own identities, so none of its vectors,
    # pair scores or indexes is reused here; a sealed-canary recipe never names a class.
    there = {**recipe_payload(RECIPE_MINILM_INT8_CPU), "execution_class": "arm64; arm; NEON"}
    other = HybridIndexSpec.model_validate({**there, "logical_hash": _identity(there)})
    assert other.logical_hash != INT8.logical_hash
    assert embedding_context_hash(other) != embedding_context_hash(INT8)
    assert reranker_context_hash(other) != reranker_context_hash(INT8)
    named = {**recipe_payload(RECIPE_MINILM_CPU), "execution_class": "arm64; arm; NEON"}
    with pytest.raises(ValueError, match="machine class only for a live canary"):
        HybridIndexSpec.model_validate({**named, "logical_hash": _identity(named)})
    # The same repository and revision, installed beside the fp32 pack, never over it.
    store = Path("store")
    for fp32, quantized in zip(
        model_store.RECIPE_PACKS[RECIPE_MINILM_CPU],
        model_store.RECIPE_PACKS[RECIPE_MINILM_INT8_CPU],
        strict=True,
    ):
        beside = model_store.pack_directory(store, fp32).with_name(f"{fp32.revision}-int8")
        assert model_store.pack_directory(store, quantized) == beside


@pytest.mark.proves("knowledge.int8_canary_is_proved_live_at_session_open")
def test_a_session_proves_the_sealed_canary_else_its_machines_one_thread_canary() -> None:
    asked: list[str] = []

    def one_thread() -> str:
        asked.append("one thread")
        return "a" * 64

    assert canary_to_prove(INT8, None, one_thread) == "a" * 64 and asked == ["one thread"]
    # A sealed canary is proved as sealed, and a context with none and no live canary proves
    # nothing; neither opens a one-thread session.
    assert canary_to_prove(RETAINED, "b" * 64, one_thread) == "b" * 64
    assert canary_to_prove(RETAINED, None, one_thread) is None
    assert asked == ["one thread"]


@pytest.mark.proves("knowledge.a_new_authority_takes_the_int8_recipe")
def test_a_new_authority_takes_the_int8_recipe_and_an_installed_one_keeps_its_own() -> None:
    assert NEW_BUILD_RECIPE == RECIPE_MINILM_INT8_CPU
    assert setup_recipe(None, installed=None, retained_layout=False) == RECIPE_MINILM_INT8_CPU
    # A pack directory of the retained layout is read only by the retained recipe.
    assert setup_recipe(None, installed=None, retained_layout=True) == RECIPE_MINILM_CPU
    assert (
        setup_recipe(None, installed=RECIPE_MINILM_CPU, retained_layout=False) == RECIPE_MINILM_CPU
    )
    named = "hybrid-v3-bge-small-en"
    assert setup_recipe(named, installed=RECIPE_MINILM_CPU, retained_layout=True) == named
