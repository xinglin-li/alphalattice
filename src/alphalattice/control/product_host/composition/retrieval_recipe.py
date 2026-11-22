"""The retrieval recipe a workspace's review searches with, as a reader sees it."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def retrieval_recipe_view(spec: Mapping[str, Any]) -> dict[str, object]:
    """Project the review's retrieval spec (`recall.reference_spec`): the chunking recipe, the
    encoder and reranker by pack and revision, and the device, always the CPU. Never a quality
    claim; a pack's presence is proven by the capability hash the manifest binds."""
    recipe, model, reranker = spec["recipe"], spec["model"], spec["reranker"]
    return {
        "recipe": f"{recipe['splitter']} {recipe['version']}, {recipe['capacity']} characters",
        "encoder": {
            "model_id": model["manifest"]["pack"],
            "revision": model["manifest"]["revision"],
            "dimension": model["dimension"],
            "precision": model["precision"],
        },
        "reranker": {"model_id": reranker["pack"], "revision": reranker["revision"]},
        "device": "cpu",
    }
