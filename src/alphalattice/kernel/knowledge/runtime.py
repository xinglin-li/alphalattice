"""Public runtime handles shared by Knowledge and its consumers."""

from ._embeddings import (
    MODEL_WORK,
    VERIFIED_PACKS,
    before_first_inference,
    observing_passages,
    withhold_inference_yield,
    yielding_during_inference,
)
from .retrieval import _build_chunks as build_chunks

__all__ = [
    "MODEL_WORK",
    "VERIFIED_PACKS",
    "before_first_inference",
    "build_chunks",
    "observing_passages",
    "withhold_inference_yield",
    "yielding_during_inference",
]
