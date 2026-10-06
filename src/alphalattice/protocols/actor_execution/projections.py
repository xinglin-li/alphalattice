"""Framework-neutral, content-addressed actor-facing Markdown projections."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import tiktoken
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

_CL100K = tiktoken.get_encoding("cl100k_base")


def normalize_model_content(content: str) -> str:
    """Normalize model-facing text without changing evidence membership."""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).rstrip() + "\n"


def estimate_cl100k_proxy_tokens(content: str) -> int:
    """Return a stable diagnostic estimate, not provider token authority."""
    return len(_CL100K.encode(content))


def model_projection_semantics_hash(*, projection_id: str, semantics: object) -> str:
    """Hash stable renderer semantics without binding invocation-specific content."""
    return canonical_hash(
        {"projection_id": projection_id, "media_type": "text/markdown", "semantics": semantics}
    )


class ModelFacingProjection(BaseModel):
    """A content-addressed Markdown presentation admitted to one model call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    projection_id: str = Field(min_length=1, max_length=160)
    media_type: Literal["text/markdown"] = "text/markdown"
    semantics_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_hashes: tuple[str, ...]
    content: str = Field(min_length=1)
    utf8_bytes: int = Field(ge=1)
    token_estimate_method: Literal["CL100K_PROXY"] = "CL100K_PROXY"
    cl100k_proxy_tokens: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_projection(self) -> ModelFacingProjection:
        """Verify normalized content, source order, measurements, and hash."""
        if self.content != normalize_model_content(self.content):
            raise ValueError("model-facing projection content is not normalized")
        if self.source_hashes != tuple(dict.fromkeys(self.source_hashes)):
            raise ValueError("model-facing projection source hashes are not stable and unique")
        if self.utf8_bytes != len(self.content.encode("utf-8")):
            raise ValueError("model-facing projection byte count is invalid")
        if self.cl100k_proxy_tokens != estimate_cl100k_proxy_tokens(self.content):
            raise ValueError("model-facing projection token estimate is invalid")
        identity = {
            "projection_id": self.projection_id,
            "media_type": self.media_type,
            "source_hashes": self.source_hashes,
            "content": self.content,
        }
        if self.content_hash != canonical_hash(identity):
            raise ValueError("model-facing projection content hash is invalid")
        return self


def build_model_facing_projection(
    *,
    projection_id: str,
    semantics: object,
    content: str,
    source_hashes: Iterable[str],
) -> ModelFacingProjection:
    """Build a measured, content-addressed Markdown projection.

    Args:
        projection_id: Stable name of the projection.
        semantics: Renderer rules used to produce its content.
        content: Markdown text to normalize and measure.
        source_hashes: Source identities in presentation order.

    Returns:
        A validated projection with content and semantics hashes.

    """
    normalized = normalize_model_content(content)
    hashes = tuple(dict.fromkeys(source_hashes))
    identity = {
        "projection_id": projection_id,
        "media_type": "text/markdown",
        "source_hashes": hashes,
        "content": normalized,
    }
    return ModelFacingProjection(
        **identity,
        semantics_hash=model_projection_semantics_hash(
            projection_id=projection_id, semantics=semantics
        ),
        content_hash=canonical_hash(identity),
        utf8_bytes=len(normalized.encode("utf-8")),
        cl100k_proxy_tokens=estimate_cl100k_proxy_tokens(normalized),
    )


__all__ = [
    "ModelFacingProjection",
    "build_model_facing_projection",
    "estimate_cl100k_proxy_tokens",
    "model_projection_semantics_hash",
    "normalize_model_content",
]
