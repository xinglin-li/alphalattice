"""Canonical identity owner for one complete current-sector observation set."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping


def sector_revision_hash(
    *, manifest_revision: str, observations: Iterable[Mapping[str, object]]
) -> str:
    """Hash the manifest and normalized sector observations into a revision.

    Args:
        manifest_revision: Frozen manifest revision for the observation set.
        observations: Provider sector records keyed by listing identity.

    Returns:
        Canonical SHA-256 revision of the sorted observations.

    """
    safe_records = tuple(
        {
            "listing_id": str(item["listing_id"]),
            "provider": str(item["provider"]),
            "provider_symbol": str(item["provider_symbol"]),
            "sector_name": str(item["sector_name"]),
            "sector_key": str(item["sector_key"]) if item.get("sector_key") else None,
            "payload_hash": str(item["payload_hash"]),
            "evidence_hash": str(item["evidence_hash"]),
        }
        for item in sorted(observations, key=lambda row: str(row["listing_id"]))
    )
    payload = json.dumps(
        {
            "source": "YAHOO_CURRENT_SECTOR",
            "manifest": manifest_revision,
            "items": safe_records,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["sector_revision_hash"]
