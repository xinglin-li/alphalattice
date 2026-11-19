"""What the market source held for a Panel's listings when the Panel was published.

Its own module, outside the Feature numbers' rule closure: the record describes the source a
Panel read and decides no Feature value, so changing it rotates no Panel's implementation
identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository

PANEL_MARKET_OBSERVATION_VERSION = "panel-market-observation/v1"


@dataclass(frozen=True)
class PanelMarketObservation:
    """Which sessions the market source held no bar for when a Panel was published.

    Within each listing's own span of bars over the Panel's sessions, the
    sessions without a bar; every other session of that span traded, and a
    session outside it is not observed. A Panel whose listings traded on every
    session of their spans records none. The snapshot hash covers the record,
    which binds it to the Panel; ``source_watermark_hash`` binds it to the
    market source through ``through`` over the Panel's listing axis. It says
    what the source held, not why a bar is missing.
    """

    through: date
    source_watermark_hash: str
    absent_sessions: Mapping[str, tuple[date, ...]]

    def to_payload(self) -> dict[str, object]:
        """Serialize the record in listing and session order under its version."""
        return {
            "version": PANEL_MARKET_OBSERVATION_VERSION,
            "through": self.through.isoformat(),
            "source_watermark_hash": self.source_watermark_hash,
            "absent_sessions": {
                listing_id: [item.isoformat() for item in sorted(self.absent_sessions[listing_id])]
                for listing_id in sorted(self.absent_sessions)
            },
        }


def panel_market_observation(manifest: Mapping[str, object]) -> PanelMarketObservation | None:
    """The market observation a Panel snapshot records, or None when it records none.

    Args:
        manifest: Verified Panel snapshot payload.

    Returns:
        The recorded observation, or None for a Panel published without one.

    Raises:
        ValueError: The record has another version or an invalid shape.
    """
    summary = manifest.get("safe_summary")
    record = summary.get("market_observation") if isinstance(summary, Mapping) else None
    if record is None:
        return None
    try:
        if not isinstance(record, Mapping) or (
            record["version"] != PANEL_MARKET_OBSERVATION_VERSION
        ):
            raise ValueError
        absent = cast(Mapping[str, Sequence[str]], record["absent_sessions"])
        observation = PanelMarketObservation(
            through=date.fromisoformat(str(record["through"])),
            source_watermark_hash=str(record["source_watermark_hash"]),
            absent_sessions={
                str(listing_id): tuple(date.fromisoformat(item) for item in sessions)
                for listing_id, sessions in absent.items()
            },
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("feature_panel.market_observation_invalid") from error
    if not observation.absent_sessions or observation.to_payload() != dict(record):
        raise ValueError("feature_panel.market_observation_invalid")
    return observation


def sessions_without_a_bar(axis: Sequence[date], bars: pa.Table) -> dict[str, tuple[date, ...]]:
    """Each listing's sessions of `axis` without a bar, between its first and last bar on it.

    `bars` holds the (listing_id, session_date) of each bar on the axis. A session before a
    listing's first bar or after its last is outside its span and not named, and a listing
    whose span is complete is absent from the result. The spans are found over the whole table
    at once; only a listing with a gap is walked.
    """
    if bars.num_rows == 0:
        return {}
    ordered = np.asarray(axis, dtype="datetime64[D]")
    listings = bars.column("listing_id").to_numpy(zero_copy_only=False)
    positions = np.searchsorted(ordered, bars.column("session_date").to_numpy(zero_copy_only=False))
    spans = (
        pa.table({"listing_id": bars.column("listing_id"), "position": positions})
        .group_by("listing_id")
        .aggregate([("position", "min"), ("position", "max"), ("position", "count_distinct")])
    )
    gapped = spans.filter(
        pc.less(
            spans["position_count_distinct"],
            pc.add(pc.subtract(spans["position_max"], spans["position_min"]), 1),
        )
    )
    absent: dict[str, tuple[date, ...]] = {}
    for listing_id, first, last in sorted(
        zip(
            gapped["listing_id"].to_pylist(),
            gapped["position_min"].to_pylist(),
            gapped["position_max"].to_pylist(),
            strict=True,
        )
    ):
        held = set(positions[listings == listing_id].tolist())
        absent[str(listing_id)] = tuple(
            axis[index] for index in range(first, last + 1) if index not in held
        )
    return absent


def read_market_observation(
    market: MarketDataRepository,
    *,
    manifest_revision: str,
    listing_ids: tuple[str, ...],
    sessions: tuple[date, ...],
) -> PanelMarketObservation | None:
    """What the market held for a Panel's listings over its sessions, at publication.

    Each listing's sessions without a bar inside its own span of bars over `sessions`, bound to
    the source by its watermark through the last session over the Panel's listings under the
    Universe revision the Panel names. None when no listing misses a session of its span, so a
    Panel without a gap records nothing. Read through the market's own public reads.

    Args:
        market: The market store the Panel was built from.
        manifest_revision: Universe revision the Panel names as its source.
        listing_ids: The Panel's listing axis.
        sessions: The Panel's ordered session axis.

    Returns:
        The observation, or None when every span is complete.
    """
    with market.market_reads(listings=listing_ids) as (reading, tables):
        bars = reading.execute(
            f"""
            SELECT listing_id, session_date FROM {tables["bars"]}
            WHERE listing_id IN (SELECT unnest(?::VARCHAR[]))
              AND session_date IN (SELECT unnest(?::DATE[]))
            """,
            [list(listing_ids), list(sessions)],
        ).to_arrow_table()
    absent = sessions_without_a_bar(sessions, bars)
    if not absent:
        return None
    watermark = market.execution_source_watermark(
        market.load_universe_manifest_revision(manifest_revision),
        through=sessions[-1],
        listing_ids=listing_ids,
    )
    return PanelMarketObservation(
        through=sessions[-1],
        source_watermark_hash=str(watermark["watermark_hash"]),
        absent_sessions=absent,
    )
