"""A Panel sector map's classification read through world.sector: the one function its readers
call for each listing's Sector.

A map (closure_contracts.SectorRevisionMap) keeps its artifact as published: its entries, their
provenance (provider_symbol, payload_hash, evidence_hash) and its map_hash, which revisions and
frozen books hash. Maps are published in a task child, which records no selection, so the Host
records one at the map's first classification read in the Host: a selection whose spec names the
map's map_hash, at the log's clock, made only when world classifies every listing with exactly
the map's names at that cutoff. `classified` then returns the map's entries with each Sector
name as world knew it at that cutoff. A map with no selection (read only outside a Host so far,
or whose names world lacks or holds differently) reads its entries as the artifact holds them.
A world revision known after the cutoff never reaches the map's readers.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from alphalattice.foundation.data_layer import history
from alphalattice.foundation.feature_engine.panels.closure_contracts import (
    SectorRevisionEntry,
    SectorRevisionMap,
)
from alphalattice.foundation.market_data_ops.storage import world, world_reads


def classified(classification: SectorRevisionMap) -> tuple[SectorRevisionEntry, ...]:
    """Each listing's entry in `classification`, its Sector name read through world at the cutoff
    of the map's selection (made here, in the Host, at its first read: `_select`), else the
    artifact's own. Refused by name (`feature_closure.sector_map_unserved`) when a selection's
    cutoff names a listing world does not classify, or no world catalog is reachable: never
    answered from the artifact instead."""
    cutoff = world_reads.map_cutoff(classification.map_hash)
    if cutoff is None:
        cutoff = _select(classification)
    if cutoff is None:
        return classification.entries
    listings = tuple(entry.listing_id for entry in classification.entries)
    return renamed(classification.entries, _names(history.data_root(), listings, cutoff))


def renamed(
    entries: tuple[SectorRevisionEntry, ...], names: dict[str, str] | None
) -> tuple[SectorRevisionEntry, ...]:
    """`entries` with each listing's Sector name from `names`, world's; refused by name
    (`feature_closure.sector_map_unserved`) when world gave no names or misses a listing."""
    if names is None or any(entry.listing_id not in names for entry in entries):
        raise ValueError("feature_closure.sector_map_unserved")
    return tuple(
        entry.model_copy(update={"sector_name": names[entry.listing_id]}) for entry in entries
    )


def named(classification: SectorRevisionMap) -> SectorRevisionMap:
    """`classification` with its entries as `classified` reads them, for a pure reader that is
    handed the map (the risk producer's classification design): the same map, its names read
    through world where it is selected."""
    entries = classified(classification)
    if entries is classification.entries:
        return classification
    return classification.model_copy(update={"entries": entries})


def _select(sector_map: SectorRevisionMap) -> int | None:
    """In the Host, record the selection naming `sector_map`'s map_hash at the log's clock and
    return its cutoff, when world classifies every listing with exactly the map's names at that
    cutoff; None otherwise (outside a Host, or the names differ), so no value moves."""
    host = history.opened_log()
    if host is None:
        return None
    cutoff = host[0].clock()
    listings = tuple(entry.listing_id for entry in sector_map.entries)
    with world_reads.session() as cursor:
        if cursor is None:
            return None
        names = world.sector_map(listings, session=date.max, cutoff_us=cutoff, cursor=cursor)
    if names != {entry.listing_id: entry.sector_name for entry in sector_map.entries}:
        return None
    history.selection(
        {"cutoff_us": cutoff, "spec": {"map_hash": sector_map.map_hash}}, actor=("host", "host")
    )
    return cutoff


_NAMES_HELD = 64
"""The distinct (root, listings, cutoff) cuts one process reads names for: a Daily's few."""


@lru_cache(maxsize=_NAMES_HELD)
def _names(root: Path, listings: tuple[str, ...], cutoff_us: int) -> dict[str, str] | None:
    """World's Sector names of `listings` as known at `cutoff_us` in the data root `root`, or None
    with no world catalog; a known-time cut never changes, so each is read once per process and
    root, and at most _NAMES_HELD cuts are held."""
    with world_reads.session() as cursor:
        if cursor is None:
            return None
        return world.sector_map(listings, session=date.max, cutoff_us=cutoff_us, cursor=cursor)


__all__ = ["classified", "named", "renamed"]
