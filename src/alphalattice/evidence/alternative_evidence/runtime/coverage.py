"""One coverage run: every issuer of a scope, prepared as bounded units.

A request is one execution packet of at most eight issuers -- the axis the
acquisition, the packet program and the analyst were built and measured on.
A book of fifty names is not one request; it is one *run* of several units,
and the run is what the Host admits, the Task executes, progress reports and
the review consumes. This module owns the run's identity and how units are
formed. It does not schedule anything: Task Control runs the units as the
existing preparation stages, one unit after another.

A book's units are formed from the issuers sorted by entity id and packed
in that order under two bounds -- the issuer limit and the admitted document
capacity of one document set -- from each issuer's own logical source
selection (`source_counts`: how many documents the issuer's policy selects,
decided before any packing), so a weight change leaves every unit's
membership -- and therefore every unit's request, generation and packet --
exactly where it was; execution order follows the priority of a unit's
best-ranked issuer, so what the book cares about most is prepared first
without deciding what is prepared at all. Without counts (a caller that has
none) the units are cut at the issuer limit alone, as before, and each
issuer's share of the capacity is then the acquisition's own accounting.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from ..analysis.contracts import AlternativeEvidenceResearchObligation
from ..contracts import (
    ADMITTED_DOCUMENT_CAPACITY,
    AlternativeEvidenceClass,
    AlternativeEvidenceContract,
    AlternativeEvidenceMode,
    AlternativeEvidenceReadFiling,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSourcePolicy,
    MatterSelectionPolicy,
    seal_contract,
    validate_contract_identity,
)
from ..sources.contracts import SecFilingSelectionPlan

_HASH = r"^[0-9a-f]{64}$"

UNIT_LIMIT: int = 8
"""The request axis: the most issuers one unit prepares together. The
existing Alternative Evidence entity budget, kept as an execution bound."""

RUN_UNIT_LIMIT: int = 512
"""The most units one run carries: at most one per issuer, and the
Portfolio scope names at most 512 positions. Packing from each issuer's
source count can need far more units than the issuer-limit cut ever
produced (fourteen for a 64-issuer book whose selections are dense), and
the run's Task plan carries one set of stages per unit, so Task Control's
plan bound is sized for this many units across every stage
(`PLAN_WORK_ITEM_LIMIT`); the adapter proves the two agree."""

UNKNOWN_SELECTION_RESERVATION: int = ADMITTED_DOCUMENT_CAPACITY // UNIT_LIMIT
"""What the packing reserves for an issuer whose inventory has never been
read (no sealed selection plan at or before the cutoff): the share a unit
of eight gave every issuer before packing existed (three), so a first run
over a book packs as it always did; the acquisition then plans each issuer
under its own budget, defers by name what the unit cannot take, and the
next run packs from the sealed plans it left."""

UNIT_PACKING_RULES_ID = "alternative-evidence.coverage-unit-packing.v2"
"""v2 (2026-09-24, W1): units are packed by portfolio weight from each
issuer's logical source count. The issuers are taken heaviest first (weight
rank, ties by entity id); an issuer joins the open unit while the unit stays
within `UNIT_LIMIT` issuers and `ADMITTED_DOCUMENT_CAPACITY` selected
documents, otherwise it opens the next, so the units run heaviest first and
the lightest last. An issuer whose filing index at the cutoff holds nothing
in the window is not packed: the run names it as nothing filed. An issuer
whose own selection exceeds the capacity is refused by name, never cut. v1
(2026-09-22) packed by entity id, whatever the weights; a run sealed under it
keeps its hash, and a caller without counts still gets the issuer-limit cut."""

UNIT_ID_PATTERN = r"^u[0-9]{2,3}$"

COVERAGE_RUN_PURPOSE = "COVERAGE_RUN"
"""The Task purpose whose stages are unit-qualified: `u01_admit_evidence_request`."""


def coverage_units(
    ordered_entity_ids: tuple[str, ...],
    *,
    priority_rank: dict[str, int],
    unit_limit: int = UNIT_LIMIT,
    source_counts: Mapping[str, int] | None = None,
    document_capacity: int = ADMITTED_DOCUMENT_CAPACITY,
    weight_rank: Mapping[str, int] | None = None,
    nothing_filed: frozenset[str] = frozenset(),
    carried: frozenset[str] = frozenset(),
) -> tuple[tuple[str, ...], ...]:
    """Pack the issuers into units, in the order they run.

    With `source_counts` (each issuer's logical source selection, decided
    before packing) and `weight_rank` the units are packed under
    `UNIT_PACKING_RULES_ID`: the issuers heaviest first, each joining the open
    unit while it stays within the issuer limit and the document capacity, and
    `nothing_filed` -- issuers whose filing index at the cutoff holds nothing
    in the window -- and `carried` -- issuers whose every filing in the window
    an earlier analysis read -- not packed at all. The units run in the order they were
    packed, so the heaviest holdings are read first and the lightest last;
    inside a unit the issuers keep the order the scope gives.
    Without counts a book that fits the issuer limit is one request and a
    wider book is cut at the limit by entity id, ordered by the best priority
    rank inside each unit, as before.
    """
    if not 1 <= unit_limit <= UNIT_LIMIT:
        raise ValueError("alternative_evidence.coverage_unit_limit_invalid")
    if len(set(ordered_entity_ids)) != len(ordered_entity_ids) or not ordered_entity_ids:
        raise ValueError("alternative_evidence.coverage_entities_invalid")
    if source_counts is None:
        if len(ordered_entity_ids) <= unit_limit:
            return (tuple(ordered_entity_ids),)
        entities = tuple(sorted(ordered_entity_ids))
        units = [
            entities[start : start + unit_limit] for start in range(0, len(entities), unit_limit)
        ]
        units.sort(
            key=lambda unit: (min(priority_rank.get(entity, 1 << 30) for entity in unit), unit)
        )
        return tuple(units)
    if weight_rank is None:
        raise ValueError("alternative_evidence.coverage_weight_rank_missing")
    missing = [entity for entity in ordered_entity_ids if entity not in source_counts]
    if missing or not (nothing_filed | carried) <= set(ordered_entity_ids):
        raise ValueError(
            "alternative_evidence.coverage_source_counts_missing:" + ",".join(sorted(missing))
        )
    over = [
        entity
        for entity in ordered_entity_ids
        if source_counts[entity] > document_capacity or source_counts[entity] < 0
    ]
    if over:
        raise ValueError(
            "alternative_evidence.coverage_issuer_exceeds_unit_capacity:" + ",".join(sorted(over))
        )
    heaviest_first = sorted(
        (
            entity
            for entity in ordered_entity_ids
            if entity not in nothing_filed and entity not in carried
        ),
        key=lambda entity: (weight_rank.get(entity, 1 << 30), entity),
    )
    units = []
    current: list[str] = []
    held = 0
    for entity in heaviest_first:
        count = source_counts[entity]
        if current and (len(current) >= unit_limit or held + count > document_capacity):
            units.append(tuple(current))
            current, held = [], 0
        current.append(entity)
        held += count
    if current:
        units.append(tuple(current))
    # Inside a unit the issuers keep the scope's order: a book that fits one
    # unit is the one request, obligation and intent it always was.
    position = {entity: index for index, entity in enumerate(ordered_entity_ids)}
    return tuple(tuple(sorted(unit, key=position.__getitem__)) for unit in units)


@dataclass(frozen=True, slots=True)
class LogicalSourceCounts:
    """Describe the logical sources left for each issuer at a cutoff.

    Before any unit is packed, the record holds the logical source count:
    what remains to read, together with the
    basis of each, the issuers that filed nothing in the window, the issuers
    whose every filing in it an earlier analysis read (`carried`), and those
    earlier readings, by issuer and filing.
    """

    counts: dict[str, int]
    basis: dict[str, str]
    nothing_filed: frozenset[str] = frozenset()
    carried: frozenset[str] = frozenset()
    read_filings: tuple[AlternativeEvidenceReadFiling, ...] = field(default=())


class AlternativeEvidenceCoverageIntent(AlternativeEvidenceContract):
    """Capture what a person approves when a book is prepared.

    The intent names the book's scope, cutoff, evidence policy, source package,
    and permissions. The
    packing is the Host's, and the filing index read at the preparation
    refines it, so it is not part of what was approved.
    """

    kind: Literal["AlternativeEvidenceCoverageIntent"] = "AlternativeEvidenceCoverageIntent"
    scope_hash: str = Field(pattern=_HASH)
    evidence_as_of: datetime
    unit_limit: int = Field(ge=1, le=UNIT_LIMIT)
    resource_binding_hash: str = Field(pattern=_HASH)
    network_consent: bool
    admit_live_official: bool
    admit_model_review: bool
    source_policy: AlternativeEvidenceSourcePolicy
    evidence_classes: tuple[AlternativeEvidenceClass, ...]
    mode: AlternativeEvidenceMode
    ttl_seconds: int
    matter_selection: MatterSelectionPolicy | None = None
    intent_hash: str = Field(pattern=_HASH)


def coverage_intent_hash(
    run: AlternativeEvidenceCoverageRun, *, policy: AlternativeEvidenceRequest | None = None
) -> str:
    """Return the identity handed out by preview and named by preparation.

    The policy is the one its units were sealed under. A run with nothing left to read has
    no unit, and the caller names the policy it holds.
    """
    if run.units:
        request = run.units[0].request
    elif policy is not None:
        request = policy
    else:
        raise ValueError("alternative_evidence.coverage_intent_policy_missing")
    return seal_contract(
        AlternativeEvidenceCoverageIntent,
        "intent_hash",
        scope_hash=run.scope_hash,
        evidence_as_of=run.evidence_as_of,
        unit_limit=run.unit_limit,
        resource_binding_hash=run.resource_binding_hash,
        network_consent=run.network_consent,
        admit_live_official=run.admit_live_official,
        admit_model_review=run.admit_model_review,
        source_policy=request.source_policy,
        evidence_classes=request.evidence_classes,
        mode=request.mode,
        ttl_seconds=request.ttl_seconds,
        matter_selection=request.matter_selection,
    ).intent_hash


def coverage_unit_ids(units: tuple[tuple[str, ...], ...]) -> dict[tuple[str, ...], str]:
    """Name the units by membership, not by the order they run in.

    `u01` is the unit whose first issuer sorts first, whatever the book's
    weights say today, so a packet prepared as `u03` is still `u03` in the
    progress a reweighted book reports and in the dossier that reads it.
    """
    return {unit: f"u{index:02d}" for index, unit in enumerate(sorted(units), start=1)}


class AlternativeEvidenceCoverageUnit(AlternativeEvidenceContract):
    """One bounded execution packet of the run: its request and its question."""

    unit_id: str = Field(pattern=UNIT_ID_PATTERN)
    ordered_entity_ids: tuple[str, ...] = Field(min_length=1, max_length=UNIT_LIMIT)
    request: AlternativeEvidenceRequest
    obligation: AlternativeEvidenceResearchObligation
    preparation_intent_hash: str = Field(pattern=_HASH)
    """The intent this unit's preparation answers -- the same identity a
    standalone preparation of the same request carries, so the two are one
    completed unit wherever it was prepared."""

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_unit(self) -> Self:
        """Validate the unit's request, obligation, and issuer axis."""
        if (
            self.request.ordered_entity_ids != self.ordered_entity_ids
            or self.obligation.ordered_entity_ids != self.ordered_entity_ids
            or self.obligation.evidence_as_of != self.request.evidence_as_of
        ):
            raise ValueError("alternative_evidence.coverage_unit_axis_invalid")
        return self


class AlternativeEvidenceCoverageRun(AlternativeEvidenceContract):
    """Every unit one scope needs at one cutoff under one policy and binding.

    The run's identity carries no clock of its own: two admissions of the
    same book, cutoff, policy, source package and permissions are the same
    run, which is what lets a preview's captured intent be submitted later
    and what lets a completed unit be found again.
    """

    kind: Literal["AlternativeEvidenceCoverageRun"] = "AlternativeEvidenceCoverageRun"
    scope_hash: str = Field(pattern=_HASH)
    """The Portfolio issuer scope this run covers, by identity only."""
    evidence_as_of: datetime
    unit_limit: int = Field(ge=1, le=UNIT_LIMIT)
    resource_binding_hash: str = Field(pattern=_HASH)
    network_consent: bool
    admit_live_official: bool
    admit_model_review: bool
    units: tuple[AlternativeEvidenceCoverageUnit, ...] = Field(max_length=RUN_UNIT_LIMIT)
    """In execution order; the ids name membership (`coverage_unit_ids`). None
    when every holding is carried or filed nothing: nothing is left to read,
    and the run is sealed without a Task."""
    packing_rules_id: str | None = Field(
        default=None, max_length=80, exclude_if=lambda value: value is None
    )
    """`UNIT_PACKING_RULES_ID` when the units were packed from each issuer's
    logical source count; absent (and absent from the identity) on a run
    cut at the issuer limit alone, so every earlier run keeps its hash."""
    source_counts: tuple[tuple[str, int], ...] = Field(
        default=(), max_length=512, exclude_if=lambda value: value == ()
    )
    """Each issuer's logical source count the packing read, sorted by
    issuer: part of the run's identity, because two admissions of one book
    whose held metadata differ are two packings."""
    nothing_filed: tuple[str, ...] = Field(
        default=(), max_length=512, exclude_if=lambda value: value == ()
    )
    """The issuers whose filing index at the cutoff holds nothing in the
    window, sorted: not packed, and reported as nothing filed -- never as no
    risk. Absent from the identity when empty, so every earlier run keeps its
    hash."""
    carried: tuple[str, ...] = Field(
        default=(), max_length=512, exclude_if=lambda value: value == ()
    )
    """The issuers whose every filing in the window an earlier current
    analysis read, sorted: not packed -- nothing new is there to read -- and
    reviewed by the findings of those readings, which carry while their
    filings stay in the window (W3). Absent from the identity when empty."""
    read_filings: tuple[AlternativeEvidenceReadFiling, ...] = Field(
        default=(), max_length=4096, exclude_if=lambda value: value == ()
    )
    """Every filing of the scope's issuers in the window that an earlier
    current analysis read, with that analysis, sorted by issuer and filing: a
    unit's request names its issuers' share, and the review reads the rest
    from the analyses named. Absent from the identity when empty."""
    run_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_run(self) -> Self:
        """Validate packing, issuer membership, and the sealed run identity."""
        if self.evidence_as_of.tzinfo is None or self.evidence_as_of.utcoffset() is None:
            raise ValueError("alternative_evidence.coverage_run_clock_invalid")
        expected = coverage_unit_ids(tuple(unit.ordered_entity_ids for unit in self.units))
        if any(unit.unit_id != expected.get(unit.ordered_entity_ids) for unit in self.units):
            raise ValueError("alternative_evidence.coverage_unit_ids_invalid")
        entities = [entity for unit in self.units for entity in unit.ordered_entity_ids]
        if len(entities) != len(set(entities)):
            raise ValueError("alternative_evidence.coverage_units_overlap")
        if (self.packing_rules_id is None) != (self.source_counts == ()):
            raise ValueError("alternative_evidence.coverage_packing_invalid")
        unpacked = (*self.nothing_filed, *self.carried)
        for named in (self.nothing_filed, self.carried):
            if list(named) != sorted(set(named)):
                raise ValueError("alternative_evidence.coverage_packing_invalid")
        if len(unpacked) != len(set(unpacked)) or set(unpacked) & set(entities):
            raise ValueError("alternative_evidence.coverage_packing_invalid")
        if not self.units and not unpacked:
            raise ValueError("alternative_evidence.coverage_units_missing")
        if self.source_counts:
            counted = [entity for entity, _count in self.source_counts]
            if counted != sorted(counted) or set(counted) != set(entities) | set(unpacked):
                raise ValueError("alternative_evidence.coverage_packing_invalid")
            if any(count != 0 for entity, count in self.source_counts if entity in unpacked) or any(
                count < 0 or count > ADMITTED_DOCUMENT_CAPACITY for _e, count in self.source_counts
            ):
                raise ValueError("alternative_evidence.coverage_packing_invalid")
        elif unpacked:
            raise ValueError("alternative_evidence.coverage_packing_invalid")
        read = [(value.entity_id, value.accession) for value in self.read_filings]
        if (
            read != sorted(set(read))
            or not {entity for entity, _accession in read} <= set(entities) | set(self.carried)
            or not set(self.carried) <= {entity for entity, _accession in read}
            or any(
                unit.request.read_filings
                != tuple(
                    value
                    for value in self.read_filings
                    if value.entity_id in unit.ordered_entity_ids
                )
                for unit in self.units
            )
        ):
            raise ValueError("alternative_evidence.coverage_read_filings_invalid")
        if any(
            len(unit.ordered_entity_ids) > self.unit_limit
            or unit.request.evidence_as_of != self.evidence_as_of
            for unit in self.units
        ):
            raise ValueError("alternative_evidence.coverage_unit_invalid")
        validate_contract_identity(self, "run_hash")
        return self

    @property
    def ordered_entity_ids(self) -> tuple[str, ...]:
        """Return packed issuer IDs in execution order."""
        return tuple(entity for unit in self.units for entity in unit.ordered_entity_ids)

    @property
    def covered_entity_ids(self) -> frozenset[str]:
        """Every issuer the run accounts for: packed, carried or filed nothing."""
        return frozenset((*self.ordered_entity_ids, *self.carried, *self.nothing_filed))

    def unit(self, unit_id: str) -> AlternativeEvidenceCoverageUnit:
        """Return a unit by ID, raising ``KeyError`` when it is absent."""
        for value in self.units:
            if value.unit_id == unit_id:
                return value
        raise KeyError(unit_id)

    def unit_of(self, entity_id: str) -> AlternativeEvidenceCoverageUnit:
        """Return the unit containing an issuer, raising ``KeyError`` if absent."""
        for value in self.units:
            if entity_id in value.ordered_entity_ids:
                return value
        raise KeyError(entity_id)


def seal_coverage_run(**values: object) -> AlternativeEvidenceCoverageRun:
    """Seal a coverage run under its content identity."""
    return seal_contract(AlternativeEvidenceCoverageRun, "run_hash", **values)


class UnitSourcesShort(ValueError):
    """A unit whose issuers hold too few source documents for the installed floor (V541).

    Its code names the coverage the unit reached, the count the floor needed and, when any
    were left out of the share, the issuers that filed nothing (V587). The issuers without a
    source ride beside the code, on `uncovered`, and the unit's sealed failure keeps them: an
    entity id may be 32 characters and a recorded code is cut at 120.
    """

    def __init__(
        self, *, held: int, issuers: int, needed: int, uncovered: tuple[str, ...], quiet: int = 0
    ) -> None:
        """Name the reach, the need, the quiet issuers left out and the issuers without a source.

        Args:
            held: The counted issuers holding a source document.
            issuers: The counted issuers: the unit's, less those that filed nothing.
            needed: The fewest issuers holding a source that the floor admits.
            uncovered: The counted issuers without one, in the unit's order.
            quiet: The issuers left out of the share because they filed nothing.
        """
        super().__init__(
            "alternative_evidence.minimum_entity_coverage_not_met:"
            f"{held} of {issuers} issuers hold a source, {needed} needed"
            + (f", {quiet} filed nothing" if quiet else "")
        )
        self.held, self.issuers, self.needed, self.quiet = held, issuers, needed, quiet
        self.uncovered = uncovered


def index_quiet(
    plans: Mapping[str, Collection[SecFilingSelectionPlan]],
    *,
    evidence_as_of: datetime,
    window_days: int,
) -> frozenset[str]:
    """The issuers an index read at this cutoff found filing nothing in the window (V587).

    What `sources_short` leaves out of a share, wherever the floor is judged: an issuer is
    quiet only by a plan of its own filing index read at exactly this cutoff under this
    window that selected and deferred nothing (`SecFilingSelectionPlan.nothing_filed`, V541's
    NOTHING_FILED). An index read at another cutoff, a failed read or none is no such plan,
    and that issuer counts against the floor.

    Args:
        plans: Sealed selection plans by issuer.
        evidence_as_of: The cutoff the floor is judged at.
        window_days: The window the plans must have been read under.

    Returns:
        The quiet issuers among the plans' issuers.
    """
    return frozenset(
        entity
        for entity, held in plans.items()
        if any(
            plan.evidence_as_of == evidence_as_of
            and plan.event_window_days == window_days
            and plan.nothing_filed
            for plan in held
        )
    )


def sources_short(
    ordered_entity_ids: tuple[str, ...],
    sourced: Collection[str],
    *,
    floor: float,
    quiet: Callable[[tuple[str, ...]], Collection[str]] | None = None,
) -> UnitSourcesShort | None:
    """The refusal a unit meets when too few of its issuers hold a source, else None.

    The share of the issuers holding a source against the installed floor, as the
    acquisition always measured it, and the fewest such issuers that share admits: one rule
    for the unit's refusal, the preview that predicts it (V541) and the package the installer
    seals (V587). An issuer whose filing index at the cutoff shows nothing filed in the window
    holds nothing to count: `quiet` names those among the issuers without a source, asked only
    when the share falls short, and they leave it; a failed or unread acquisition stays in it.

    Args:
        ordered_entity_ids: The unit's issuers.
        sourced: The issuers holding a source document at the unit's cutoff.
        floor: The installed package's minimum issuer coverage.
        quiet: Which of the issuers given without a source filed nothing in the window, by an
            index read at the cutoff (`index_quiet`); None where no index was read.

    Returns:
        The refusal, naming the reach, the need, the quiet issuers left out and the counted
        issuers without a source; None when the counted issuers reach the floor.
    """
    held = sum(1 for entity in ordered_entity_ids if entity in sourced)
    uncovered = tuple(entity for entity in ordered_entity_ids if entity not in sourced)
    issuers = len(ordered_entity_ids)
    filed_nothing: frozenset[str] = frozenset()
    if issuers and held / issuers < floor and quiet is not None and uncovered:
        filed_nothing = frozenset(quiet(uncovered)).intersection(uncovered)
        uncovered = tuple(entity for entity in uncovered if entity not in filed_nothing)
        issuers -= len(filed_nothing)
    if not issuers or held / issuers >= floor:
        return None
    return UnitSourcesShort(
        held=held,
        issuers=issuers,
        needed=next(count for count in range(issuers + 1) if count / issuers >= floor),
        uncovered=uncovered,
        quiet=len(filed_nothing),
    )


class AlternativeEvidenceUnitFailure(AlternativeEvidenceContract):
    """A unit that did not complete, sealed so the run can go on without it.

    The run's other units are independent work; one unit's refusal (a source
    package without its issuer's documents, a capacity refusal, a document
    that no longer verifies) is recorded under the stage that refused, by the
    owner's own code, and every later stage of that unit carries this same
    record instead of pretending to complete. A person sees which issuers
    were not prepared and why; a later run of the same intent prepares the
    unit again.
    """

    kind: Literal["AlternativeEvidenceUnitFailure"] = "AlternativeEvidenceUnitFailure"
    run_hash: str = Field(pattern=_HASH)
    unit_id: str = Field(pattern=UNIT_ID_PATTERN)
    ordered_entity_ids: tuple[str, ...] = Field(min_length=1, max_length=UNIT_LIMIT)
    stage_id: str = Field(min_length=1, max_length=80)
    failure_code: str = Field(min_length=1, max_length=200)
    recorded_at: datetime
    uncovered_entity_ids: tuple[str, ...] = Field(
        default=(), max_length=UNIT_LIMIT, exclude_if=lambda value: not value
    )
    """The unit's issuers without a source document, when its sources fell short of the floor
    (V541): sealed with that refusal, and empty, absent from the identity, for every other
    refusal and every failure sealed before, which keep their hashes."""
    failure_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_failure(self) -> Self:
        """Validate the failure timestamp, its issuers and sealed identity."""
        if self.recorded_at.tzinfo is None or self.recorded_at.utcoffset() is None:
            raise ValueError("alternative_evidence.unit_failure_clock_invalid")
        if not set(self.uncovered_entity_ids) <= set(self.ordered_entity_ids):
            raise ValueError("alternative_evidence.unit_failure_issuers_invalid")
        validate_contract_identity(self, "failure_hash")
        return self


def seal_unit_failure(**values: object) -> AlternativeEvidenceUnitFailure:
    """Seal a failed unit under its content identity."""
    return seal_contract(AlternativeEvidenceUnitFailure, "failure_hash", **values)


__all__ = [
    "COVERAGE_RUN_PURPOSE",
    "UNIT_ID_PATTERN",
    "UNIT_LIMIT",
    "UNIT_PACKING_RULES_ID",
    "UNKNOWN_SELECTION_RESERVATION",
    "AlternativeEvidenceCoverageIntent",
    "AlternativeEvidenceCoverageRun",
    "AlternativeEvidenceCoverageUnit",
    "AlternativeEvidenceUnitFailure",
    "LogicalSourceCounts",
    "UnitSourcesShort",
    "coverage_intent_hash",
    "coverage_unit_ids",
    "coverage_units",
    "index_quiet",
    "seal_coverage_run",
    "seal_unit_failure",
    "sources_short",
]
