"""Publish a recovery sidecar before a Panel snapshot becomes operational."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import pyarrow.parquet as pq

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels.feature_closure_contracts import (
    PanelColumnClosureHead,
    PanelRecoveryClosureBinding,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    FeatureClosureLedger,
    identified,
)


class PanelRecoveryBindingPublisher:
    """Bind a Panel snapshot to exact Feature closure and sector evidence."""

    def __init__(self, *, ledger: FeatureClosureLedger, resolver: ArtifactResolver) -> None:
        """Bind the closure ledger and artifact resolver used for publication."""
        self._ledger = ledger
        self._resolver = resolver

    def publish(
        self,
        *,
        snapshot_hash: str,
        panel_content_hash: str,
        panel_binding_hash: str,
        catalog_hash: str,
        sector_revision: str,
        listing_ids: Sequence[str],
        factor_ids: Sequence[str],
        chunks: Sequence[Mapping[str, object]],
        panel_source_state_hash: str,
        parts: Sequence[str] = (),
    ) -> PanelRecoveryClosureBinding:
        """Seal and verify a Panel's exact closure and sector binding.

        Resolve each chunk to derive the session axis, then publish the
        binding through the ledger and read it back before returning. A
        layered catalog's rows are its ``parts``' (V92), the base first: the
        binding names each part's closure head.
        """
        closures = tuple(parts) or (catalog_hash,)
        heads = tuple(self._ledger.require_head(closure) for closure in closures)
        if any(self._ledger.pending_transition(closure) is not None for closure in closures):
            raise ValueError("feature_closure.panel_binding_incomplete")
        head = heads[0]
        sector_map = self._ledger.sector_map_for_head(head=head, sector_revision=sector_revision)
        sessions: set[date] = set()
        for chunk in chunks:
            path = self._resolver.resolve_feature_panel_chunk_ref(
                uri=str(chunk["uri"]),
                content_hash=str(chunk["chunk_hash"]),
                metadata_hash=str(chunk["metadata_hash"]),
            )
            session_column = pq.read_table(path, columns=["session_date"])["session_date"]
            sessions.update(session_column.to_pylist())
        binding = identified(
            PanelRecoveryClosureBinding,
            {
                "snapshot_hash": snapshot_hash,
                "panel_content_hash": panel_content_hash,
                "panel_binding_hash": panel_binding_hash,
                "catalog_hash": catalog_hash,
                "closure_head_hash": head.head_hash,
                "closure_transition_cursor": head.transition_cursor,
                "sector_map_hash": sector_map.map_hash,
                "listing_ids": tuple(listing_ids),
                "sessions": tuple(sorted(sessions)),
                "factor_ids": tuple(factor_ids),
                "panel_source_state_hash": panel_source_state_hash,
                "column_closures": tuple(
                    PanelColumnClosureHead(
                        catalog_hash=closure,
                        closure_head_hash=column.head_hash,
                        closure_transition_cursor=column.transition_cursor,
                    )
                    for closure, column in zip(closures[1:], heads[1:], strict=True)
                ),
            },
            "binding_hash",
        )
        try:
            self._ledger.publish_panel_binding(binding)
        except Exception as exc:
            raise ValueError("feature_closure.panel_binding_incomplete") from exc
        loaded = self._ledger.panel_binding(snapshot_hash)
        if loaded != binding:
            raise ValueError("feature_closure.panel_binding_incomplete")
        return binding


__all__ = ["PanelRecoveryBindingPublisher"]
