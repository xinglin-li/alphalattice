"""Content-addressed persistence for one causal Sector state surface.

The on-disk root stays ``alpha-research/sector-context`` even though the code now
lives in Sector Research. That string is where every existing surface and
manifest was written; renaming it would make them unreadable and buy nothing but
tidiness.
"""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .contracts import SectorContextManifest
from .surface import SectorContextBoundaryError, SectorContextSurface


class SectorContextStore:
    """Publish and verify identity-addressed Sector context manifests and Parquet payloads.

    The compatibility storage location is alpha-research/sector-context beneath the caller artifact
    root. Existing identities admit exact replay only; readback verifies both payload bytes and
    logical table content.
    """

    def __init__(self, artifact_root: Path) -> None:
        """Bind context publication/readback to the caller artifact root.

        Args:
            artifact_root: Caller-owned root beneath which the context store is located.
        """
        self.root = artifact_root.resolve() / "alpha-research" / "sector-context"

    @staticmethod
    def _write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        staged.write_bytes(content)
        os.replace(staged, path)
        staged.unlink(missing_ok=True)

    def publish(self, *, manifest: SectorContextManifest, payload: bytes) -> None:
        """Publish a context payload before its manifest, accepting only exact replay.

        Args:
            manifest: Validated policy/source/axis and content commitments.
            payload: Parquet bytes matching manifest.payload_sha256.

        Raises:
            SectorContextBoundaryError: Payload digest is invalid or an existing identity has
                different bytes.
        """
        if sha256(payload).hexdigest() != manifest.payload_sha256:
            raise SectorContextBoundaryError("alpha_research.sector_context_payload_invalid")
        data = self.root / "surfaces" / f"{manifest.manifest_hash}.parquet"
        meta = self.root / "manifests" / f"{manifest.manifest_hash}.json"
        serialized = json.dumps(
            manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        if data.is_file() and data.read_bytes() != payload:
            raise SectorContextBoundaryError("alpha_research.sector_context_identity_reused")
        if not data.is_file():
            self._write(data, payload)
        if meta.is_file() and meta.read_bytes() != serialized:
            raise SectorContextBoundaryError("alpha_research.sector_context_identity_reused")
        if not meta.is_file():
            self._write(meta, serialized)

    def load(self, manifest_hash: str) -> SectorContextSurface:
        """Verify persisted context bytes and reconstruct a read-only four-feature tensor.

        Args:
            manifest_hash: Identity selecting the stored context manifest and Parquet payload.

        Returns:
            Verified manifest and float64 values with shape sessions by sectors by four.

        Raises:
            SectorContextBoundaryError: Readback, payload/table identity or tensor reconstruction
                fails.
        """
        meta = self.root / "manifests" / f"{manifest_hash}.json"
        data = self.root / "surfaces" / f"{manifest_hash}.parquet"
        try:
            manifest = SectorContextManifest.model_validate_json(meta.read_bytes())
            payload = data.read_bytes()
            if sha256(payload).hexdigest() != manifest.payload_sha256:
                raise SectorContextBoundaryError("alpha_research.sector_context_payload_tampered")
            table = pq.read_table(data).combine_chunks()
            table_hash = str(
                canonical_hash({"schema": str(table.schema), "rows": table.to_pylist()})
            )
            if table_hash != manifest.table_content_hash:
                raise SectorContextBoundaryError("alpha_research.sector_context_table_tampered")
            values = np.column_stack(
                [
                    table[name].to_numpy(zero_copy_only=False)
                    for name in (
                        "sector_trend_20",
                        "sector_surprise_0",
                        "sector_surprise_1",
                        "sector_surprise_5",
                    )
                ]
            ).astype(np.float64, copy=False)
            values = values.reshape(len(manifest.formation_sessions), len(manifest.sector_ids), 4)
            values.setflags(write=False)
        except SectorContextBoundaryError:
            raise
        except Exception as error:
            raise SectorContextBoundaryError(
                "alpha_research.sector_context_readback_failed"
            ) from error
        return SectorContextSurface(manifest=manifest, values=values)


__all__ = ["SectorContextStore"]
