"""What the readiness tests read of a workspace's storage."""

import json
from pathlib import Path


def published_panel_chunks(root: Path) -> set[str]:
    """Every Panel chunk a published generation's manifest names, as the storage plan names it."""

    return {
        f"artifacts/feature-panel/chunks/{chunk['chunk_hash']}.parquet"
        for manifest in (root / "artifacts/feature-panel/manifests").glob("*.json")
        for chunk in json.loads(manifest.read_text(encoding="utf-8"))["chunks"]
    }
