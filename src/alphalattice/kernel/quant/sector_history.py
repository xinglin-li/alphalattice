"""The Sector each session reads, as data a consumer applies.

A workspace's Sector classification is the provider's current one, observed at each refresh;
under the forward rule (`market_data_ops/sources/sector_forward.py` says from which session a
reclassification is in force) each listing reads the classification first recorded for it until
its first reclassification -- a disclosed backfill, not point in time -- and each later one from
its effective session. A history is that record: the current classification and the
reclassifications, each with its effective session. Every consumer applies it here, run by run of
sessions that read one map, so a consumer's identity binds this arithmetic with its own and the
history itself as content (LAWS ID8).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from itertools import groupby, pairwise
from typing import Any

import numpy as np
import numpy.typing as npt

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sector_treatment import (
    SectorHistoryTreatment,
    sector_treatment,
)


@dataclass(frozen=True)
class SectorReclassification:
    """One listing's move to another Sector, in force from its effective session on."""

    listing_id: str
    effective_session: date
    prior_sector: str
    sector: str


@dataclass(frozen=True)
class SectorHistory(Mapping[str, str]):
    """Every listing's Sector at every session: the backfill, then each reclassification.

    `current` is each listing's latest classification and `current_revision` the revision the
    store holds it under; `reclassifications` are ordered by effective session, and each one's
    prior Sector is the one the listing held before it, so the backfill is read back from them.
    As a mapping it is the current classification, so a reader of one session's map or of the
    names classified takes it as it took a map; a reader across sessions asks for its runs
    (`sector_slices`, `sector_positions`), where a plain map is one run.
    """

    current_revision: str
    current: Mapping[str, str]
    reclassifications: tuple[SectorReclassification, ...] = ()
    _runs: dict[tuple[date, ...], tuple[tuple[tuple[date, ...], dict[str, str]], ...]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        """Check the reclassifications chain from the backfill to the current classification.

        Raises:
            ValueError: `sector_history.invalid` when they are unordered, name a listing
                with no current Sector, or do not chain.
        """
        ordered = sorted(
            self.reclassifications, key=lambda item: (item.effective_session, item.listing_id)
        )
        if list(self.reclassifications) != ordered:
            raise ValueError("sector_history.invalid")
        held: dict[str, str] = {}
        for item in self.reclassifications:
            if item.listing_id not in self.current or item.prior_sector == item.sector:
                raise ValueError("sector_history.invalid")
            if held.get(item.listing_id, item.prior_sector) != item.prior_sector:
                raise ValueError("sector_history.invalid")
            held[item.listing_id] = item.sector
        if any(self.current[listing_id] != sector for listing_id, sector in held.items()):
            raise ValueError("sector_history.invalid")

    def __eq__(self, other: object) -> bool:
        """Whether another history, or a map read by every session, is this one.

        A map equals a history only while no reclassification is in force.

        Args:
            other: A history, or a map.

        Returns:
            Whether they classify every session alike.
        """
        if isinstance(other, SectorHistory):
            return (
                self.current_revision == other.current_revision
                and dict(self.current) == dict(other.current)
                and self.reclassifications == other.reclassifications
            )
        if isinstance(other, Mapping):
            return not self.reclassifications and dict(self.current) == dict(other)
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def __getitem__(self, listing_id: str) -> str:
        """The listing's current Sector."""
        return self.current[listing_id]

    def __iter__(self) -> Iterator[str]:
        """The listings classified."""
        return iter(self.current)

    def __len__(self) -> int:
        """How many listings are classified."""
        return len(self.current)

    @classmethod
    def of_panel(cls, lineage: Mapping[str, Any], current: Mapping[str, str]) -> SectorHistory:
        """The Sector each session of a Panel read, from its recorded lineage.

        Args:
            lineage: The Panel manifest's `safe_summary.lineage`: its `sector_revision` and the
                `sector_reclassifications` it read (absent while none is in force).
            current: That revision's classification of the listings asked for.

        Returns:
            The history over those listings.
        """
        return cls(
            current_revision=str(lineage["sector_revision"]),
            current=dict(current),
            reclassifications=tuple(
                SectorReclassification(
                    listing_id=str(item["listing_id"]),
                    effective_session=date.fromisoformat(str(item["effective_session"])),
                    prior_sector=str(item["prior_sector"]),
                    sector=str(item["sector"]),
                )
                for item in lineage.get("sector_reclassifications") or ()
                if str(item["listing_id"]) in current
            ),
        )

    def lineage_payload(self) -> list[dict[str, str]]:
        """The reclassifications as a Panel's lineage records them.

        Returns:
            One record per reclassification, in effective-session order.
        """
        return [
            {
                "listing_id": item.listing_id,
                "effective_session": item.effective_session.isoformat(),
                "prior_sector": item.prior_sector,
                "sector": item.sector,
            }
            for item in self.reclassifications
        ]

    def subset(self, listing_ids: Iterable[str]) -> SectorHistory:
        """The history of these listings alone, under the same revision.

        Args:
            listing_ids: Listings it classifies.

        Returns:
            Their current classification and their reclassifications.

        Raises:
            KeyError: For a listing it does not classify.
        """
        current = {listing_id: self.current[listing_id] for listing_id in listing_ids}
        return SectorHistory(
            current_revision=self.current_revision,
            current=current,
            reclassifications=tuple(
                item for item in self.reclassifications if item.listing_id in current
            ),
        )

    @property
    def sectors(self) -> tuple[str, ...]:
        """Every Sector some session reads, sorted."""
        return tuple(
            sorted(
                {*self.current.values(), *(item.prior_sector for item in self.reclassifications)}
            )
        )

    @property
    def base(self) -> dict[str, str]:
        """The classification each listing reads before its first reclassification."""
        values = dict(self.current)
        for item in reversed(self.reclassifications):
            values[item.listing_id] = item.prior_sector
        return values

    def at(self, session: date) -> dict[str, str]:
        """Each listing's Sector at one session.

        Args:
            session: The session.

        Returns:
            Listing ID to Sector.
        """
        values = self.base
        for item in self.reclassifications:
            if item.effective_session > session:
                break
            values[item.listing_id] = item.sector
        return values

    def runs(self, sessions: Sequence[date]) -> tuple[tuple[tuple[date, ...], dict[str, str]], ...]:
        """The sessions in order, grouped into contiguous runs that read one classification.

        Args:
            sessions: The sessions.

        Returns:
            Each run's sessions with the classification they read; one run while no
            reclassification falls inside them.
        """
        key = tuple(sorted(set(sessions)))
        held = self._runs.get(key)
        if held is not None:
            return held
        boundaries = [item.effective_session for item in self.reclassifications]

        def epoch(session: date) -> int:
            return sum(boundary <= session for boundary in boundaries)

        runs = tuple(
            (group, self.at(group[0]))
            for group in (tuple(values) for _, values in groupby(key, key=epoch))
        )
        self._runs[key] = runs
        return runs

    def slices(self, sessions: Sequence[date]) -> tuple[tuple[slice, dict[str, str]], ...]:
        """For an ascending session axis: each run's rows, as a slice, and the map they read.

        Args:
            sessions: The axis, ascending and without repeats.

        Returns:
            One slice per run; one covering the axis while no reclassification falls inside it.

        Raises:
            ValueError: `sector_history.axis_unordered` for another axis.
        """
        ordered = tuple(sessions)
        if any(left >= right for left, right in pairwise(ordered)):
            raise ValueError("sector_history.axis_unordered")
        slices: list[tuple[slice, dict[str, str]]] = []
        start = 0
        for run_sessions, sectors in self.runs(ordered):
            slices.append((slice(start, start + len(run_sessions)), sectors))
            start += len(run_sessions)
        return tuple(slices)

    @property
    def identity(self) -> str:
        """The current revision while no reclassification is in force; else the history's hash.

        A history of one classification names what the Panels built before the forward rule
        named, so their bindings stay what they were.
        """
        if not self.reclassifications:
            return self.current_revision
        identity: str = canonical_hash(
            {
                "kind": "SectorHistory",
                "current_revision": self.current_revision,
                "reclassifications": [
                    {
                        "listing_id": item.listing_id,
                        "effective_session": item.effective_session.isoformat(),
                        "prior_sector": item.prior_sector,
                        "sector": item.sector,
                    }
                    for item in self.reclassifications
                ],
            }
        )
        return identity


def sector_slices(
    sectors: Mapping[str, str], sessions: Sequence[date]
) -> tuple[tuple[slice, Mapping[str, str]], ...]:
    """Each run of an ascending session axis and the map it reads.

    Args:
        sectors: A `SectorHistory`, or a map read by every session.
        sessions: The axis, ascending and without repeats.

    Returns:
        The history's runs; one run over the axis for a plain map, the map itself.
    """
    if isinstance(sectors, SectorHistory):
        return sectors.slices(sessions)
    return ((slice(0, len(sessions)), sectors),)


def sector_positions(
    sectors: Mapping[str, str], sessions: Sequence[date], listing_ids: Sequence[str]
) -> tuple[tuple[slice, tuple[str, ...], tuple[npt.NDArray[np.int64], ...]], ...]:
    """For each run of a session axis: its rows, its Sectors and their listings' positions.

    Args:
        sectors: A `SectorHistory`, or a map read by every session.
        sessions: The axis, ascending and without repeats.
        listing_ids: The listing axis every run's positions index.

    Returns:
        Per run: its rows, the Sectors its listings read (sorted) and, per Sector, the positions
        of its listings in `listing_ids`.
    """
    runs: list[tuple[slice, tuple[str, ...], tuple[npt.NDArray[np.int64], ...]]] = []
    for rows, mapping in sector_slices(sectors, sessions):
        ordered = tuple(sorted({mapping[listing_id] for listing_id in listing_ids}))
        runs.append(
            (
                rows,
                ordered,
                tuple(
                    np.asarray(
                        [
                            index
                            for index, listing_id in enumerate(listing_ids)
                            if mapping[listing_id] == sector
                        ],
                        dtype=np.int64,
                    )
                    for sector in ordered
                ),
            )
        )
    return tuple(runs)


def sector_ids(sectors: Mapping[str, str]) -> tuple[str, ...]:
    """Every Sector some session reads, sorted: a history's, or a plain map's values.

    Args:
        sectors: A `SectorHistory`, or a map read by every session.

    Returns:
        The Sectors, sorted.
    """
    if isinstance(sectors, SectorHistory):
        return sectors.sectors
    return tuple(sorted(set(sectors.values())))


def sector_codes(
    sectors: Mapping[str, str],
    sessions: Sequence[date],
    listing_ids: Sequence[str],
    ordered_sector_ids: Sequence[str],
) -> npt.NDArray[np.int64]:
    """Each listing's Sector at each session, as its position in `ordered_sector_ids`.

    Args:
        sectors: A `SectorHistory`, or a map read by every session.
        sessions: The session axis, ascending and without repeats.
        listing_ids: The listing axis.
        ordered_sector_ids: The Sector axis the positions index; every Sector read is on it.

    Returns:
        One row per session, one column per listing.
    """
    position = {sector: index for index, sector in enumerate(ordered_sector_ids)}
    codes: npt.NDArray[np.int64] = np.empty((len(sessions), len(listing_ids)), dtype=np.int64)
    for rows, mapping in sector_slices(sectors, sessions):
        codes[rows] = np.asarray(
            [position[mapping[listing_id]] for listing_id in listing_ids], dtype=np.int64
        )
    return codes


def sector_subset(sectors: Mapping[str, str], listing_ids: Iterable[str]) -> Mapping[str, str]:
    """A map, or a history, over these listings alone.

    Args:
        sectors: A `SectorHistory`, or a map read by every session.
        listing_ids: Listings it classifies.

    Returns:
        The history's `subset`, or a new map.

    Raises:
        KeyError: For a listing it does not classify.
    """
    if isinstance(sectors, SectorHistory):
        return sectors.subset(listing_ids)
    return {listing_id: sectors[listing_id] for listing_id in listing_ids}


def reclassification_payload(sectors: Mapping[str, str]) -> dict[str, list[dict[str, str]]]:
    """What a hash binds beside a current map: the reclassifications, while any is in force.

    A map with none adds nothing, so an identity that bound the current map alone before the
    forward rule stays what it was.

    Args:
        sectors: A `SectorHistory`, or a map read by every session.

    Returns:
        `{"sector_reclassifications": [...]}` for a history holding one; else nothing.
    """
    if isinstance(sectors, SectorHistory) and sectors.reclassifications:
        return {"sector_reclassifications": sectors.lineage_payload()}
    return {}


def sector_treatment_of(sectors: Mapping[str, str]) -> SectorHistoryTreatment:
    """What a map's sessions read, as a record states it.

    Args:
        sectors: A `SectorHistory`, or a map read by every session.

    Returns:
        The forward treatment for a history holding a reclassification; else the backfill.
    """
    return sector_treatment(
        reclassified=isinstance(sectors, SectorHistory) and bool(sectors.reclassifications)
    )


def sector_exposure(
    sectors: Mapping[str, str],
    sessions: Sequence[date],
    listing_ids: Sequence[str],
    ordered_sector_ids: Sequence[str],
) -> npt.NDArray[np.float64]:
    """Each listing's exposure to its Sector: one row a Sector, one column a listing.

    Args:
        sectors: A `SectorHistory`, or a map read by every session.
        sessions: The session axis, ascending and without repeats; none reads the current map.
        listing_ids: The listing axis.
        ordered_sector_ids: The Sector axis; every Sector read is on it.

    Returns:
        One matrix while one map covers the sessions, else one per session stacked; read-only.
    """
    position = {sector: index for index, sector in enumerate(ordered_sector_ids)}

    def matrix(mapping: Mapping[str, str]) -> npt.NDArray[np.float64]:
        values: npt.NDArray[np.float64] = np.zeros(
            (len(ordered_sector_ids), len(listing_ids)), dtype=np.float64
        )
        for column, listing_id in enumerate(listing_ids):
            values[position[mapping[listing_id]], column] = 1.0
        return values

    runs: tuple[tuple[slice, Mapping[str, str]], ...] = (
        sector_slices(sectors, sessions) if len(sessions) else ((slice(0, 0), sectors),)
    )
    if len(runs) == 1:
        exposure = matrix(runs[0][1])
    else:
        exposure = np.empty(
            (len(sessions), len(ordered_sector_ids), len(listing_ids)), dtype=np.float64
        )
        for rows, mapping in runs:
            exposure[rows] = matrix(mapping)
    exposure.setflags(write=False)
    return exposure


def exposure_sector_ids(
    sectors: Mapping[str, str], sessions: Sequence[date], listing_ids: Sequence[str]
) -> tuple[str, ...]:
    """The Sectors these listings read at some session of the axis, sorted.

    Args:
        sectors: A `SectorHistory`, or a map read by every session.
        sessions: The session axis, ascending and without repeats; none reads the current map.
        listing_ids: The listings.

    Returns:
        The Sectors; the current map's alone while no reclassification falls inside the axis.
    """
    runs: tuple[tuple[slice, Mapping[str, str]], ...] = (
        sector_slices(sectors, sessions) if len(sessions) else ((slice(0, 0), sectors),)
    )
    return tuple(
        sorted({mapping[listing_id] for _rows, mapping in runs for listing_id in listing_ids})
    )


__all__ = [
    "SectorHistory",
    "SectorReclassification",
    "exposure_sector_ids",
    "reclassification_payload",
    "sector_codes",
    "sector_exposure",
    "sector_ids",
    "sector_positions",
    "sector_slices",
    "sector_subset",
    "sector_treatment_of",
]
