"""The retrieval recipes and model packs an installed Evidence authority names.

The model store installs and binds these packs, and a workspace's manifest names its recipe; the
Host's retrieval store reads its own packs (`model_store.RETRIEVAL_STORE_PACKS`)."""

from __future__ import annotations

RECIPE_MINILM_CPU = "hybrid-v2-minilm"
RECIPE_BGE_SMALL_CPU = "hybrid-v3-bge-small-en"
RECIPE_QWEN3_ENCODER_GPU = "hybrid-v3-qwen3-embedding-gpu"
RECIPE_QWEN3_RERANKER_GPU = "hybrid-v3-minilm-qwen3-reranker-gpu"
RECIPE_QWEN3_GPU = "hybrid-v3-qwen3-gpu"
RECIPE_MINILM_INT8_CPU = "hybrid-v3-minilm-int8"
NEW_BUILD_RECIPE = RECIPE_MINILM_INT8_CPU
"""The recipe a new Evidence authority is set up with; an installed one keeps its own."""
SUPPORTED_RECIPES: tuple[str, ...] = (
    RECIPE_MINILM_CPU,
    RECIPE_BGE_SMALL_CPU,
    RECIPE_QWEN3_ENCODER_GPU,
    RECIPE_QWEN3_RERANKER_GPU,
    RECIPE_QWEN3_GPU,
    RECIPE_MINILM_INT8_CPU,
)
"""The bounded set of retrieval recipes: one payload each, no other pairing.

`hybrid-v2-minilm` is the payload every existing index and generation was
built under; the four `v3` recipes are the comparison arms of the evidence
initiative (a CPU encoder against the retained reranker; the Qwen3 encoder,
the Qwen3 reranker, and both, on the one supported local GPU backend).
`hybrid-v3-minilm-int8` is the same two models as the upstream int8 graphs
of the same pinned revisions. A recipe is admitted by name; its identity is
its whole payload."""

RERANKER_MODEL_ID = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL_REVISION = "a09144355adeed5f58c8ed011d209bf8ee5a1fec"
RERANKER_PACK: tuple[tuple[str, str], ...] = (
    # Every file the installed FastEmbed loader reads before or during
    # inference. `load_tokenizer` consumes the three JSON configurations and the
    # tokenizer; the graph is run last. Verifying only the graph and the
    # tokenizer admitted a pack whose `model_max_length` was 16.
    ("config.json", "d827779a72d27ae68cf878a6fc2e954542663fe21ca515d9f4783fc96be2d37e"),
    ("onnx/model.onnx", "c623d0bcb99f4622beb413eaef00cfbe5db20df9f1dd982da4b4f26022881870"),
    ("special_tokens_map.json", "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3"),
    ("tokenizer.json", "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
    ("tokenizer_config.json", "0b29c7bfc889e53b36d9dd3e686dd4300f6525110eaa98c76a5dafceb2029f53"),
)
# The same revision's upstream int8 graph in place of the fp32 one; the other four
# files are the fp32 pack's.
RERANKER_INT8_GRAPH = "onnx/model_int8.onnx"
RERANKER_INT8_PACK: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (
            *((name, digest) for name, digest in RERANKER_PACK if name != "onnx/model.onnx"),
            (
                RERANKER_INT8_GRAPH,
                "a13ec391ca99f49886694e12d3e800521f36d4267d7d448c34421c541a2baf50",
            ),
        )
    )
)
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MODEL_ARTIFACT_SHA256 = "10f7a088420252b26caf819236ca2c9d2987afd0fc06fec7553b542a5655a05a"
TOKENIZER_SHA256 = "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"
# The same revision's upstream int8 graph (dynamic quantization; the repository's
# `_arm64` and `_avx512_vnni` files are these bytes too).
MODEL_INT8_GRAPH = "onnx/model_qint8_avx512.onnx"
MODEL_INT8_ARTIFACT_SHA256 = "783fea82d71a58179b830a4dbd2d58447e640609e98eedf9ffa12622d375a672"

# The BAAI encoder: the upstream fp32 ONNX export in the official repository
# (no converted or quantized third-party distribution) and its BERT WordPiece
# tokenizer file, the two files the ONNX adapter consumes.
BGE_MODEL_ID = "BAAI/bge-small-en-v1.5"
BGE_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_MODEL_ARTIFACT_SHA256 = "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"
BGE_TOKENIZER_SHA256 = "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"

# The Qwen3 packs: every file the transformers loader reads before or during
# inference (configuration, weights, tokenizer files, the reranker's chat
# template), hashed as one closure each, as the ms-marco pack is.
QWEN3_EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
QWEN3_EMBEDDING_MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QWEN3_EMBEDDING_PACK: tuple[tuple[str, str], ...] = (
    ("config.json", "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9"),
    ("generation_config.json", "28396d421a2108acce96383f6a7de78008f7f1b17f807958f3c14c51dbfb65fb"),
    ("merges.txt", "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    ("model.safetensors", "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"),
    ("tokenizer.json", "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"),
    ("tokenizer_config.json", "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0"),
    ("vocab.json", "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
)
QWEN3_RERANKER_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
QWEN3_RERANKER_MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
QWEN3_RERANKER_PACK: tuple[tuple[str, str], ...] = (
    ("chat_template.jinja", "6f682162495ec5b39fd9005c01b6aa2a74669379fe967039f1e2cbbe8752369d"),
    ("config.json", "d479c427a9ca5295218063d4f9aca4f297ab4ac27487cca7af42c84643d51ef0"),
    ("generation_config.json", "81051cd3f6e77013827148d0b8a6ead93f8ac390d5ab805f849199f0af6a08db"),
    ("merges.txt", "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    ("model.safetensors", "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b"),
    ("tokenizer.json", "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    ("tokenizer_config.json", "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0"),
    ("vocab.json", "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
)
# The one supported local GPU runtime, as locked in config/requirements-gpu.lock.


def setup_recipe(named: str | None, *, installed: str | None, retained_layout: bool) -> str:
    """Return the recipe an Evidence setup binds: the one named, else the installed one, else new.

    A new authority is set up with `NEW_BUILD_RECIPE`, unless its semantic model is a pack of the
    retained layout (`--semantic-model`), which only the retained recipe reads; an installed
    authority keeps its own recipe unless another is named.

    Args:
        named: The recipe the setup names (`--recipe`), if any.
        installed: The installed authority's recipe, if one is installed.
        retained_layout: Whether the setup names a pack directory of the retained layout.

    Returns:
        The recipe identifier the setup binds and seals.
    """
    if named:
        return named
    if installed:
        return installed
    return RECIPE_MINILM_CPU if retained_layout else NEW_BUILD_RECIPE


__all__ = [
    "BGE_MODEL_ARTIFACT_SHA256",
    "BGE_MODEL_ID",
    "BGE_MODEL_REVISION",
    "BGE_TOKENIZER_SHA256",
    "MODEL_ARTIFACT_SHA256",
    "MODEL_ID",
    "MODEL_INT8_ARTIFACT_SHA256",
    "MODEL_INT8_GRAPH",
    "MODEL_REVISION",
    "NEW_BUILD_RECIPE",
    "QWEN3_EMBEDDING_MODEL_ID",
    "QWEN3_EMBEDDING_MODEL_REVISION",
    "QWEN3_EMBEDDING_PACK",
    "QWEN3_RERANKER_MODEL_ID",
    "QWEN3_RERANKER_MODEL_REVISION",
    "QWEN3_RERANKER_PACK",
    "RECIPE_BGE_SMALL_CPU",
    "RECIPE_MINILM_CPU",
    "RECIPE_MINILM_INT8_CPU",
    "RECIPE_QWEN3_ENCODER_GPU",
    "RECIPE_QWEN3_GPU",
    "RECIPE_QWEN3_RERANKER_GPU",
    "RERANKER_INT8_GRAPH",
    "RERANKER_INT8_PACK",
    "SUPPORTED_RECIPES",
    "TOKENIZER_SHA256",
    "setup_recipe",
]
