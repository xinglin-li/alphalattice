"""Framework-neutral, content-addressed actor-facing Markdown projections."""

from __future__ import annotations

import base64
import gzip
import hashlib
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Literal

import tiktoken
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tiktoken_ext import openai_public

from alphalattice.kernel.shared_kernel.identity import canonical_hash

ENCODING = Path(__file__).with_name("cl100k_base.tiktoken.gz")
"""The cl100k encoding the package ships: tiktoken's rank file (NOTICE), gzipped, read in memory
and checked by tiktoken's own SHA-256, so a count needs no download and writes no file."""
_RANKS_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
# tiktoken's published cl100k split pattern and special tokens (tiktoken_ext.openai_public).
_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+"""
    r"""|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
)
_SPECIAL = {
    openai_public.ENDOFTEXT: 100257,
    openai_public.FIM_PREFIX: 100258,
    openai_public.FIM_MIDDLE: 100259,
    openai_public.FIM_SUFFIX: 100260,
    openai_public.ENDOFPROMPT: 100276,
}


@cache
def _cl100k() -> tiktoken.Encoding:
    """The encoding, built once on the first count from the ranks the package ships."""
    data = gzip.decompress(ENCODING.read_bytes())
    if hashlib.sha256(data).hexdigest() != _RANKS_SHA256:
        raise ValueError("actor_execution.encoding_damaged")
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in data.splitlines() if line)
    }
    return tiktoken.Encoding(
        name="cl100k_base", pat_str=_PATTERN, mergeable_ranks=ranks, special_tokens=_SPECIAL
    )


def normalize_model_content(content: str) -> str:
    """Normalize model-facing text without changing evidence membership."""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).rstrip() + "\n"


def estimate_cl100k_proxy_tokens(content: str) -> int:
    """Return a stable diagnostic estimate, not provider token authority."""
    return len(_cl100k().encode(content))


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
