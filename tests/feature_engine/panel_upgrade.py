"""A fixture Panel's days, built by whichever source is on the path: the upgrade gate's driver.

Imported, it continues a store under this source. Run as a script under an older source's
path, it builds that source's store and prints each day's identities:
``panel_upgrade.py ROOT STEPS_JSON``, a step being ``[session, spy revision letter]``.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path


def build_days(root: Path, steps: list[list[str]], *, reopen: bool = False) -> list[dict]:
    """Build each step's day in the workspace at ``root``, as the daily journey does.

    The first step is the first build (seeded, unless ``reopen`` continues a store another
    source built); a later new session lands with its bars, and a repeated one is the same
    build run again.
    """
    from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
        PanelClosureArtifactStore,
    )
    from alphalattice.foundation.feature_engine.panels.logical_identity import (
        PanelLogicalArtifactStore,
    )
    from alphalattice.foundation.feature_engine.producers.reference_data import MarketReference
    from tests.feature_engine.panel_workspace import (
        NOW,
        build_and_publish,
        fixture_manifest,
        insert_fixture_bars,
        new_session_invalidations,
        panel_workspace,
    )

    manifest = fixture_manifest()
    first = date.fromisoformat(steps[0][0])
    workspace = panel_workspace(
        root, manifest, end=first, spy=steps[0][1] * 64, **({"seed": False} if reopen else {})
    )
    logical = PanelLogicalArtifactStore(PanelClosureArtifactStore(workspace.resolver))
    landed, days = {first}, []
    for session, letter in steps:
        day, spy = date.fromisoformat(session), letter * 64
        observed_at = NOW + timedelta(days=(day - first).days)
        if day not in landed:
            landed.add(day)
            insert_fixture_bars(
                workspace.market_data, manifest, workspace.provider, start=day, end=day
            )
            workspace.feature_state.upsert_market_reference(
                reference_id="SPY",
                listing_id=MarketReference.spy(manifest).listing_id,
                provider=workspace.provider.name,
                symbol="SPY",
                revision_hash=spy,
                action_audit_receipt_hash=None,
                latest_session=day,
                observed_at=observed_at,
            )
        outcome, published = build_and_publish(
            workspace,
            manifest,
            spy=spy,
            as_of=day,
            observed_at=observed_at,
            invalidations=() if day == first else new_session_invalidations(manifest, day),
            refresh_sector=day == first,
        )
        marker = logical.marker_for_snapshot(str(published["snapshot_hash"]))
        days.append(
            {
                "snapshot_hash": published["snapshot_hash"],
                "panel_content_hash": published["panel_content_hash"],
                "logical_panel_hash": marker.logical_panel_hash if marker else None,
                "chunks": [
                    [chunk["year"], chunk.get("part", 0), chunk["chunk_hash"]]
                    for chunk in published["chunks"]
                ],
                "written": outcome.coverage_summary["panel_partitions_written"],
            }
        )
    return days


if __name__ == "__main__":
    print(json.dumps(build_days(Path(sys.argv[1]), json.loads(sys.argv[2]))))
