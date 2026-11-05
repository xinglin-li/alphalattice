"""The retrieval recipe a workspace is bound to, as a reader sees it."""

from __future__ import annotations

from alphalattice.kernel.knowledge.hybrid_contracts import (
    ENCODER_RUNTIME_TORCH_CUDA,
    RERANKER_RUNTIME_TORCH_CUDA,
    HybridIndexSpec,
)
from alphalattice.kernel.knowledge.model_store import runtime_availability


def retrieval_recipe_view(spec: HybridIndexSpec) -> dict[str, object]:
    """Project the bound retrieval recipe and its resolved device requirements.

    The retrieval recipe a workspace is bound to, as a reader sees it: the
    encoder and reranker by name and revision, the device the recipe needs,
    and -- for a GPU recipe -- what this host resolves for it. Never a
    quality claim; a pack's presence is proven by the capability hash the
    manifest binds, not restated here.
    """
    gpu = (
        spec.encoder_runtime == ENCODER_RUNTIME_TORCH_CUDA
        or spec.reranker_runtime == RERANKER_RUNTIME_TORCH_CUDA
    )
    view: dict[str, object] = {
        "recipe": spec.policy_id,
        "encoder": {
            "model_id": spec.model_id,
            "revision": spec.model_revision,
            "runtime": spec.encoder_runtime,
            "dimension": spec.embedding_dimension,
        },
        "reranker": {
            "model_id": spec.reranker_model_id,
            "revision": spec.reranker_model_revision,
            "runtime": spec.reranker_runtime,
        },
        "device": "gpu" if gpu else "cpu",
    }
    if gpu:
        view["host_runtime"] = runtime_availability()
    return view
