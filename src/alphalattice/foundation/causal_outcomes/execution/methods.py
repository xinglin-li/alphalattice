"""Installed causal execution-outcome methods and their artifact seal.

The holding span used to be an engine invariant: the publisher wrote
``index + 1`` / ``index + 2``, the Policy Holdout reader wrote ``common[1]`` /
``common[2]``, and the Factor target asserted ``spans == 2``. The generic
contract already allowed ``actual_session_span >= 2``, so the runtime, not the
contract, was what could not express a longer horizon.

This module owns the method itself. Three concepts stay separate, because
collapsing any pair of them is what lets configuration acquire authority:

- *installed capability* -- which methods exist at all -- is this module's
  catalog. It answers nothing about permission;
- *publication admission* -- which installed method may write development or
  sealed-holdout evidence today -- belongs to the publisher;
- *artifact method seal* -- which published evidence actually carries method
  authority -- is ``ExecutionOutcomeMethodBinding`` below, reachable only
  through ``ExecutionOutcomeMethodSealMarker`` and never minted by a reader.
  A binding is authority only when the Host can rebuild it exactly from the
  installed catalog, the installed publication policy and the manifest.

There is no discovery, no dynamic import, no plugin manager, no time DSL and no
second return kernel. A caller names a stable recipe id; the Host resolves it
and re-derives its identity before anything runs.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Literal, Protocol, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.market_data_ops.sources.contracts import CorporateActionEvent
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .contracts import (
    _ACTION_ID,
    _FORMULA_ID,
    CausalExecutionSchedulePoint,
)

ONE_SESSION_RECIPE_ID = "NEXT_OPEN_TO_OPEN_ONE_SESSION"
FIVE_SESSION_RECIPE_ID = "NEXT_OPEN_TO_OPEN_FIVE_SESSION"

_FIVE_SESSION_FORMULA_ID = (
    "formation-close-next-common-open-fifth-following-common-open-simple-return"
)

#: The one attribution rule installed for this family. Dividends are attributed
#: to *axis sessions* inside ``(entry_session, holding_end_session]``. For a
#: two-point span that is exactly the single holding-end lookup the publisher
#: performed before this module existed, so one-session rows are unchanged; for
#: a longer span it is the period rule that lookup was standing in for. A
#: dividend dated inside a holding interval on a day that is *not* an axis
#: session fails closed rather than being ignored, shifted to a neighbouring
#: session, or mapped by calendar guesswork: the event's effective date must
#: belong to the exact ordered common-session axis the method runs on, and
#: remediation belongs to Data qualification under a new Data identity.
AXIS_SESSION_HALF_OPEN_ATTRIBUTION = "AXIS_SESSION_HALF_OPEN_ENTRY_EXCLUSIVE_EXIT_INCLUSIVE"


class ExecutionOutcomeMethodError(ValueError):
    """Stable failure raised before any outcome evidence can be compiled."""


class SealableOutcomeManifest(Protocol):
    """The manifest fields the seal verifier compares, and only those.

    A Protocol rather than a union so the frozen snapshot contract and the
    development-only successor are verified by the *same* function without
    either importing the other's type. What makes a manifest sealable is that it
    can answer these seven questions, not which class it happens to be.
    """

    @property
    def snapshot_hash(self) -> str:
        """Return the sealed snapshot identity."""
        ...

    @property
    def schedule_hash(self) -> str:
        """Return the installed schedule identity."""
        ...

    @property
    def listing_set_hash(self) -> str:
        """Return the canonical listing-set identity."""
        ...

    @property
    def ordered_session_triples_hash(self) -> str:
        """Return the ordered formation, entry, and exit identity."""
        ...

    @property
    def price_basis(self) -> str:
        """Return the price convention used for execution rows."""
        ...

    @property
    def corporate_action_identity(self) -> str:
        """Return the corporate-action convention identity."""
        ...

    @property
    def return_formula_identity(self) -> str:
        """Return the installed return-formula identity."""
        ...


class SealableOutcomeMarker(Protocol):
    """The publication marker fields the seal verifier compares."""

    @property
    def snapshot_hash(self) -> str:
        """Return the snapshot named by the marker."""
        ...

    @property
    def manifest_ref(self) -> str:
        """Return the published manifest reference."""
        ...

    @property
    def schedule_hash(self) -> str:
        """Return the schedule identity named by the marker."""
        ...

    @property
    def listing_set_hash(self) -> str:
        """Return the listing-set identity named by the marker."""
        ...

    @property
    def marker_hash(self) -> str:
        """Return the terminal marker identity."""
        ...


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExecutionOutcomeMethodRecipe(_Contract):
    """One installed formation-to-exit method, identified by its own content.

    ``entry_offset_sessions`` and ``exit_offset_sessions`` are counted in
    *ordered exchange sessions* from the formation session, never in calendar
    days. ``actual_session_span`` is the span the schedule point carries and
    ``maturity_lag_sessions`` is how long after formation the outcome becomes
    observable; for this family both equal the exit offset, and the validator
    refuses a recipe that claims otherwise rather than letting a maturity rule
    drift away from the schedule it describes.
    """

    kind: Literal["ExecutionOutcomeMethodRecipe"] = "ExecutionOutcomeMethodRecipe"
    recipe_id: str = Field(min_length=1, max_length=96)
    research_cadence: Literal["DAILY"] = "DAILY"
    entry_offset_sessions: int = Field(ge=1)
    exit_offset_sessions: int = Field(ge=2)
    actual_session_span: int = Field(ge=2)
    maturity_lag_sessions: int = Field(ge=2)
    information_cutoff: Literal["FORMATION_OFFICIAL_CLOSE"] = "FORMATION_OFFICIAL_CLOSE"
    entry_timing: str = Field(min_length=1, max_length=96)
    exit_timing: str = Field(min_length=1, max_length=96)
    price_basis: Literal["open_split_adjusted"] = "open_split_adjusted"
    corporate_action_identity: str = Field(min_length=1, max_length=128)
    period_dividend_attribution: Literal[
        "AXIS_SESSION_HALF_OPEN_ENTRY_EXCLUSIVE_EXIT_INCLUSIVE"
    ] = AXIS_SESSION_HALF_OPEN_ATTRIBUTION  # type: ignore[assignment]
    return_formula_identity: str = Field(min_length=1, max_length=128)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require ordered offsets, matching span, and recipe identity."""
        if self.exit_offset_sessions <= self.entry_offset_sessions:
            raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_RECIPE_OFFSETS_INVALID")
        if (
            self.actual_session_span != self.exit_offset_sessions
            or self.maturity_lag_sessions != self.exit_offset_sessions
        ):
            raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_RECIPE_SPAN_INVALID")
        if self.recipe_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"recipe_hash"})
        ):
            raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_RECIPE_IDENTITY_INVALID")
        return self

    @property
    def minimum_axis_sessions(self) -> int:
        """Ordered sessions needed before one schedule point can be resolved."""
        return self.exit_offset_sessions + 1


def _seal_recipe(**values: object) -> ExecutionOutcomeMethodRecipe:
    draft = ExecutionOutcomeMethodRecipe.model_construct(**values, recipe_hash="0" * 64)
    identity = draft.model_dump(mode="json", exclude={"recipe_hash"})
    return ExecutionOutcomeMethodRecipe(**values, recipe_hash=canonical_hash(identity))


def build_one_session_recipe() -> ExecutionOutcomeMethodRecipe:
    """Build the first installed reference method.

    ``return_formula_identity`` and ``corporate_action_identity`` are the exact
    strings the frozen manifest already carries. They are read from
    ``contracts`` rather than respelled, because a second spelling that drifted
    by one character would silently rotate every published one-session identity.
    """
    return _seal_recipe(
        recipe_id=ONE_SESSION_RECIPE_ID,
        entry_offset_sessions=1,
        exit_offset_sessions=2,
        actual_session_span=2,
        maturity_lag_sessions=2,
        entry_timing="NEXT_COMMON_SESSION_OFFICIAL_OPEN",
        exit_timing="FOLLOWING_COMMON_SESSION_OFFICIAL_OPEN",
        corporate_action_identity=_ACTION_ID,
        return_formula_identity=_FORMULA_ID,
    )


def build_five_session_recipe() -> ExecutionOutcomeMethodRecipe:
    """Build the five-session candidate from formation close to exit T+6.

    Installed so the seam is real rather than hypothetical. It is deliberately
    not admitted to publication in this Gate -- see the publisher's admission
    policy -- because a span of six sessions overlaps labels across the
    development/sealed-holdout boundary and the embargo rule for that is Stage 1
    scientific policy, not infrastructure.
    """
    return _seal_recipe(
        recipe_id=FIVE_SESSION_RECIPE_ID,
        entry_offset_sessions=1,
        exit_offset_sessions=6,
        actual_session_span=6,
        maturity_lag_sessions=6,
        entry_timing="NEXT_COMMON_SESSION_OFFICIAL_OPEN",
        exit_timing="FIFTH_FOLLOWING_COMMON_SESSION_OFFICIAL_OPEN",
        corporate_action_identity=_ACTION_ID,
        return_formula_identity=_FIVE_SESSION_FORMULA_ID,
    )


def _rederive_recipe(recipe: ExecutionOutcomeMethodRecipe) -> ExecutionOutcomeMethodRecipe:
    """Re-parse a recipe from its serialized fields rather than trusting the object.

    ``model_validate`` on an instance of the same class returns that instance
    and re-runs only the ``mode="after"`` validators; field constraints are not
    re-checked. So a recipe built through ``model_construct`` with an
    out-of-range offset and a recomputed hash would survive it. Round-tripping
    through the dumped fields forces the whole schema to be applied again, which
    is what the catalog's downstream identity actually depends on.
    """
    return cast(
        ExecutionOutcomeMethodRecipe,
        ExecutionOutcomeMethodRecipe.model_validate(recipe.model_dump(mode="json")),
    )


class ExecutionOutcomeMethodIdentity(_Contract):
    """Pair an installed method ID with its sealed recipe hash."""

    recipe_id: str = Field(min_length=1, max_length=96)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExecutionOutcomeMethodCatalogBinding(_Contract):
    """Content identity of the explicitly installed outcome methods."""

    ordered_recipes: tuple[ExecutionOutcomeMethodIdentity, ...] = Field(min_length=1)
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require unique recipe IDs and the catalog's content hash."""
        keys = tuple(value.recipe_id for value in self.ordered_recipes)
        if len(set(keys)) != len(keys):
            raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_CATALOG_RECIPE_DUPLICATED")
        if self.catalog_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"catalog_hash"})
        ):
            raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_CATALOG_IDENTITY_INVALID")
        return self


class ExecutionOutcomeMethodCatalog:
    """Immutable index of installed outcome methods.

    It answers one question -- is this method installed -- and deliberately
    carries no permission field. Which installed method may write publication
    today is the publisher's policy, and which published evidence carries method
    authority is the seal. A catalog that also granted permission would let an
    installation decision silently become a publication decision.
    """

    def __init__(self, recipes: tuple[ExecutionOutcomeMethodRecipe, ...]) -> None:
        """Install unique, self-consistent outcome recipes.

        Args:
            recipes: Explicit recipe set in installation order.

        Raises:
            ExecutionOutcomeMethodError: If the set is empty or duplicated.

        """
        indexed = {value.recipe_id: value for value in recipes}
        if not recipes or len(indexed) != len(recipes):
            raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_CATALOG_INVALID")
        for recipe in recipes:
            _rederive_recipe(recipe)
        self._recipes = MappingProxyType(indexed)

    @property
    def recipe_ids(self) -> tuple[str, ...]:
        """Return installed recipe IDs in catalog order."""
        return tuple(self._recipes)

    @property
    def binding(self) -> ExecutionOutcomeMethodCatalogBinding:
        """Return the content seal of the ordered installed recipes."""
        values = tuple(
            ExecutionOutcomeMethodIdentity(
                recipe_id=recipe.recipe_id, recipe_hash=recipe.recipe_hash
            )
            for recipe in self._recipes.values()
        )
        identity = {"ordered_recipes": [value.model_dump(mode="json") for value in values]}
        return ExecutionOutcomeMethodCatalogBinding(
            ordered_recipes=values, catalog_hash=canonical_hash(identity)
        )

    def resolve(self, recipe_id: str) -> ExecutionOutcomeMethodRecipe:
        """Resolve an authored recipe ID through the installed catalog.

        Callers pass an id, never a recipe object, so a caller cannot present a
        method that shares an installed id while carrying different offsets.
        """
        recipe = self._recipes.get(recipe_id)
        if recipe is None:
            raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_RECIPE_NOT_INSTALLED")
        return _rederive_recipe(recipe)


def build_installed_execution_outcome_method_catalog() -> ExecutionOutcomeMethodCatalog:
    """Construct the explicit installed catalog; no discovery, no mutation."""
    return ExecutionOutcomeMethodCatalog((build_one_session_recipe(), build_five_session_recipe()))


def _ordered_axis(sessions: Sequence[date]) -> tuple[date, ...]:
    ordered = tuple(sessions)
    if ordered != tuple(sorted(set(ordered))):
        raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_SESSION_AXIS_NOT_CANONICAL")
    return ordered


class ExecutionOutcomeSessionAxis:
    """One validated session axis, prepared once for a whole publication.

    Attribution runs per listing and per schedule point, so a publication asks
    the same axis the same three questions more than a million times: is it
    canonical, does it contain this date, and which of its sessions fall inside
    this holding interval. Answering them from the raw sequence each time is
    what made the first real preparation spend most of its wall clock re-sorting
    a list that never changed.

    Preparing the axis does not relax anything. It is validated exactly once by
    the same ``_ordered_axis`` every caller used, and because the result is
    sorted and unique the interval can be cut by bisection -- which returns the
    identical tuple the scan returned, in the identical order, so every sum
    built from it is bit-for-bit what it was.
    """

    __slots__ = ("_members", "_ordered")

    def __init__(self, sessions: Sequence[date]) -> None:
        """Validate and cache a canonical ordered exchange-session axis."""
        self._ordered = _ordered_axis(sessions)
        self._members = frozenset(self._ordered)

    @property
    def ordered_sessions(self) -> tuple[date, ...]:
        """Return the canonical sessions in axis order."""
        return self._ordered

    def __contains__(self, session: object) -> bool:
        """Report whether the axis contains a session."""
        return session in self._members

    def half_open_interval(self, *, after: date, through: date) -> tuple[date, ...]:
        """Return axis sessions in ``(after, through]`` in order."""
        start = bisect_right(self._ordered, after)
        stop = bisect_right(self._ordered, through)
        return self._ordered[start:stop]


def _prepared_axis(
    sessions: Sequence[date] | ExecutionOutcomeSessionAxis,
) -> ExecutionOutcomeSessionAxis:
    if isinstance(sessions, ExecutionOutcomeSessionAxis):
        return sessions
    return ExecutionOutcomeSessionAxis(sessions)


def _clock(
    session_clocks: Mapping[date, Mapping[str, object]], session: date, field: str
) -> datetime:
    row = session_clocks.get(session)
    if row is None:
        raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_SESSION_CLOCK_MISSING")
    value = row.get(field)
    if not isinstance(value, datetime):
        raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_SESSION_CLOCK_INVALID")
    return value


def _schedule_point(
    *,
    recipe: ExecutionOutcomeMethodRecipe,
    ordered_sessions: tuple[date, ...],
    session_clocks: Mapping[date, Mapping[str, object]],
    index: int,
    sequence: int,
) -> CausalExecutionSchedulePoint:
    formation = ordered_sessions[index]
    entry = ordered_sessions[index + recipe.entry_offset_sessions]
    exit_session = ordered_sessions[index + recipe.exit_offset_sessions]
    return CausalExecutionSchedulePoint(
        sequence=sequence,
        formation_session=formation,
        formation_close_at=_clock(session_clocks, formation, "session_close_timestamp"),
        entry_session=entry,
        entry_open_at=_clock(session_clocks, entry, "session_open_timestamp"),
        holding_end_session=exit_session,
        holding_end_open_at=_clock(session_clocks, exit_session, "session_open_timestamp"),
        actual_session_span=recipe.actual_session_span,
    )


def resolve_schedule_points(
    *,
    recipe: ExecutionOutcomeMethodRecipe,
    ordered_sessions: Sequence[date],
    session_clocks: Mapping[date, Mapping[str, object]],
) -> tuple[CausalExecutionSchedulePoint, ...]:
    """Project one method onto an ordered exchange-session axis.

    Offsets index the session axis directly. No calendar-day arithmetic appears
    here, so a holiday or a half-session cannot shift an entry or exit.
    """
    axis = _ordered_axis(ordered_sessions)
    if len(axis) < recipe.minimum_axis_sessions:
        raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_SESSION_AXIS_TOO_SHORT")
    return tuple(
        _schedule_point(
            recipe=recipe,
            ordered_sessions=axis,
            session_clocks=session_clocks,
            index=index,
            sequence=index + 1,
        )
        for index in range(len(axis) - recipe.exit_offset_sessions)
    )


def resolve_final_schedule_point(
    *,
    recipe: ExecutionOutcomeMethodRecipe,
    ordered_sessions: Sequence[date],
    session_clocks: Mapping[date, Mapping[str, object]],
    formation_session: date,
) -> CausalExecutionSchedulePoint:
    """Resolve the single trailing point that follows one named formation."""
    axis = _ordered_axis(ordered_sessions)
    if len(axis) < recipe.minimum_axis_sessions or axis[0] != formation_session:
        raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_SESSION_AXIS_TOO_SHORT")
    return _schedule_point(
        recipe=recipe,
        ordered_sessions=axis,
        session_clocks=session_clocks,
        index=0,
        sequence=1,
    )


def period_dividend_for_point(
    *,
    recipe: ExecutionOutcomeMethodRecipe,
    dividends: Mapping[date, float],
    ordered_sessions: Sequence[date] | ExecutionOutcomeSessionAxis,
    point: CausalExecutionSchedulePoint,
) -> float:
    """Sum the admitted dividends attributed to one holding interval.

    The interval is half-open on axis sessions: ``entry_session < s <=
    holding_end_session``. For the one-session method that set is exactly
    ``{holding_end_session}``, so the value is bit-for-bit the one the previous
    ``dividends.get(holding_end_session, 0.0)`` produced.

    A dividend whose effective date falls inside the interval by calendar
    comparison but is not an axis session fails closed. Ignoring it would drop
    cash from the period return; shifting it would invent an effective date the
    provider never stated; both are Data defects wearing a method's clothes.
    Remediation is Data qualification under a new Data identity, not a mapping
    rule here.

    The axis may arrive raw or already prepared. A caller in a loop -- and every
    real caller is one -- prepares it once and hands the same object back, which
    changes only how often the axis is validated and never what this returns.
    """
    if recipe.period_dividend_attribution != AXIS_SESSION_HALF_OPEN_ATTRIBUTION:
        raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_DIVIDEND_ATTRIBUTION_NOT_INSTALLED")
    axis = _prepared_axis(ordered_sessions)
    # The off-axis check reads every dividend date once; its order changes
    # nothing, and sorting the same mapping for each of a listing's points
    # was a sort per row of the publication.
    for session in dividends:
        if point.entry_session < session <= point.holding_end_session and session not in axis:
            raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_DIVIDEND_EVENT_OFF_AXIS")
    interval = axis.half_open_interval(after=point.entry_session, through=point.holding_end_session)
    if len(interval) != recipe.exit_offset_sessions - recipe.entry_offset_sessions:
        raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_DIVIDEND_INTERVAL_INVALID")
    total = 0.0
    for session in interval:
        total += dividends.get(session, 0.0)
    return total


def action_lineage_hash(
    *,
    actions_by_listing: Mapping[str, Iterable[CorporateActionEvent]],
    intervals: Iterable[tuple[date, date]],
    ordered_sessions: Sequence[date],
) -> str:
    """Content identity of every admitted action that can move a period return.

    ``intervals`` are the schedule's ``(entry_session, exit_session]`` pairs, so
    the covered set is the whole entry-to-exit reach of the schedule rather than
    only its exit sessions. ``SPLIT`` events are included even though the
    dividend rule takes no cash from them: the provider basis is already
    split-adjusted, so a tampered split changes the prices this return is
    computed from and must be visible in identity. ``CAPITAL_GAIN`` and
    ``SPIN_OFF`` are rejected upstream by ``_action_dividends`` and never reach
    a published row.
    """
    axis = frozenset(_ordered_axis(ordered_sessions))
    covered: set[date] = set()
    for entry_session, exit_session in intervals:
        covered.update(session for session in axis if entry_session < session <= exit_session)
    rows: list[tuple[str, str, str, float | None, float | None]] = []
    for listing_id in sorted(actions_by_listing):
        events = [
            event for event in actions_by_listing[listing_id] if event.effective_date in covered
        ]
        for event in sorted(events, key=lambda value: (value.effective_date, value.action_kind)):
            rows.append(
                (
                    listing_id,
                    event.effective_date.isoformat(),
                    event.action_kind,
                    event.cash_amount,
                    event.new_shares_per_old_share,
                )
            )
    return cast(str, canonical_hash({"kind": "ExecutionOutcomeActionLineage", "events": rows}))


class ExecutionOutcomeMethodBinding(_Contract):
    """The artifact method seal: what method one published snapshot was made by.

    This is the only object that may assert method authority over a snapshot.
    It is published as its own content-addressed artifact rather than as fields
    on ``CausalExecutionOutcomeManifest``, whose bytes are frozen; and it is
    never minted by a reader, so a snapshot that was published before the seam
    existed stays legacy readback rather than acquiring authority by being read.
    """

    kind: Literal["ExecutionOutcomeMethodBinding"] = "ExecutionOutcomeMethodBinding"
    recipe_id: str = Field(min_length=1, max_length=96)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    publication_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    schedule_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_session_triples_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    entry_offset_sessions: int = Field(ge=1)
    exit_offset_sessions: int = Field(ge=2)
    actual_session_span: int = Field(ge=2)
    maturity_lag_sessions: int = Field(ge=2)
    information_cutoff: Literal["FORMATION_OFFICIAL_CLOSE"] = "FORMATION_OFFICIAL_CLOSE"
    entry_timing: str = Field(min_length=1, max_length=96)
    exit_timing: str = Field(min_length=1, max_length=96)
    price_basis: Literal["open_split_adjusted"] = "open_split_adjusted"
    corporate_action_identity: str = Field(min_length=1, max_length=128)
    period_dividend_attribution: Literal[
        "AXIS_SESSION_HALF_OPEN_ENTRY_EXCLUSIVE_EXIT_INCLUSIVE"
    ] = AXIS_SESSION_HALF_OPEN_ATTRIBUTION  # type: ignore[assignment]
    return_formula_identity: str = Field(min_length=1, max_length=128)
    source_watermark_hash: str = Field(min_length=1, max_length=128)
    action_lineage_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the recipe span and binding hash to agree with the seal."""
        if (
            self.exit_offset_sessions <= self.entry_offset_sessions
            or self.actual_session_span != self.exit_offset_sessions
            or self.maturity_lag_sessions != self.exit_offset_sessions
        ):
            raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_BINDING_SPAN_INVALID")
        if self.binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"binding_hash"})
        ):
            raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_BINDING_IDENTITY_INVALID")
        return self


def build_execution_outcome_method_binding(
    *,
    recipe: ExecutionOutcomeMethodRecipe,
    catalog_hash: str,
    publication_policy_hash: str,
    snapshot_hash: str,
    schedule_hash: str,
    ordered_session_triples_hash: str,
    source_watermark_hash: str,
    action_lineage: str,
) -> ExecutionOutcomeMethodBinding:
    """Seal one snapshot to the method that produced it."""
    values: dict[str, object] = {
        "recipe_id": recipe.recipe_id,
        "recipe_hash": recipe.recipe_hash,
        "catalog_hash": catalog_hash,
        "publication_policy_hash": publication_policy_hash,
        "snapshot_hash": snapshot_hash,
        "schedule_hash": schedule_hash,
        "ordered_session_triples_hash": ordered_session_triples_hash,
        "entry_offset_sessions": recipe.entry_offset_sessions,
        "exit_offset_sessions": recipe.exit_offset_sessions,
        "actual_session_span": recipe.actual_session_span,
        "maturity_lag_sessions": recipe.maturity_lag_sessions,
        "information_cutoff": recipe.information_cutoff,
        "entry_timing": recipe.entry_timing,
        "exit_timing": recipe.exit_timing,
        "price_basis": recipe.price_basis,
        "corporate_action_identity": recipe.corporate_action_identity,
        "period_dividend_attribution": recipe.period_dividend_attribution,
        "return_formula_identity": recipe.return_formula_identity,
        "source_watermark_hash": source_watermark_hash,
        "action_lineage_hash": action_lineage,
    }
    draft = ExecutionOutcomeMethodBinding.model_construct(**values, binding_hash="0" * 64)
    identity = draft.model_dump(mode="json", exclude={"binding_hash"})
    return ExecutionOutcomeMethodBinding(**values, binding_hash=canonical_hash(identity))


class ExecutionOutcomePublicationPolicy(_Contract):
    """Which installed methods may write development/sealed-holdout evidence.

    Deliberately *not* a field on the method catalog. The catalog answers whether
    a method exists at all; this answers whether the publisher may commit its
    output to the development and sealed-holdout split today. Collapsing the two
    would make installing a method the same act as granting it publication, and
    the five-session method is exactly where those must differ: installed and
    mechanically proven, and not permitted to publish, because a span of six
    sessions overlaps labels across the sealed boundary and the embargo rule for
    that is Stage 1 scientific policy rather than infrastructure.

    It lives beside the catalog rather than inside the publisher only so that a
    consumer verifying a seal can name the installed policy without importing
    the publisher -- which would drag artifact storage, the panel repository and
    the observation runtime into a pure target compiler. The publisher remains
    the sole caller of ``admit``: the decision is still the publisher's, the
    identity is merely readable.
    """

    kind: Literal["ExecutionOutcomePublicationPolicy"] = "ExecutionOutcomePublicationPolicy"
    admitted_recipe_ids: tuple[str, ...] = Field(min_length=1)
    development_split_semantics: Literal[
        "NON_OVERLAPPING_SINGLE_SESSION_LABEL",
        "OVERLAPPING_MULTI_SESSION_LABEL",
    ] = "NON_OVERLAPPING_SINGLE_SESSION_LABEL"
    """Widened to a union, never respelled. The frozen value keeps its exact
    string and its position as the default, so the one-session policy's
    ``policy_hash`` -- carried by every seal already on disk -- does not move."""

    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def publication_scope(self) -> Literal["DEVELOPMENT_AND_SEALED_HOLDOUT", "DEVELOPMENT_ONLY"]:
        """Which snapshot contract this policy admits.

        A property, not a field. Every field is inside ``policy_hash``, so adding
        one -- even defaulted -- would rotate the one-session policy identity
        that the seal chain compares against. The scope is a consequence of the
        label semantics rather than an independent choice: labels that overlap
        cannot be cut at a sealed boundary, which is the whole reason the
        successor contract exists.
        """
        return (
            "DEVELOPMENT_ONLY"
            if self.development_split_semantics == "OVERLAPPING_MULTI_SESSION_LABEL"
            else "DEVELOPMENT_AND_SEALED_HOLDOUT"
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> ExecutionOutcomePublicationPolicy:
        """Require canonical admission order and policy hash."""
        if self.admitted_recipe_ids != tuple(sorted(set(self.admitted_recipe_ids))):
            raise ValueError("causal execution publication policy order is not canonical")
        if self.policy_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"policy_hash"})
        ):
            raise ValueError("causal execution publication policy hash is invalid")
        return self

    def admit(self, recipe: ExecutionOutcomeMethodRecipe) -> None:
        """Refuse an installed-but-unadmitted method before any source is read."""
        if recipe.recipe_id not in self.admitted_recipe_ids:
            raise ExecutionOutcomeMethodError(
                "causal_outcomes.publication_contract_not_installed_for_recipe"
            )


def build_installed_execution_outcome_publication_policy() -> ExecutionOutcomePublicationPolicy:
    """Build the Host's one-session publication admission.

    Unchanged, field for field, so its ``policy_hash`` is exactly what every
    one-session seal already compares against. The five-session method is
    admitted by a *second* policy rather than by widening this one.
    """
    values: dict[str, object] = {
        "kind": "ExecutionOutcomePublicationPolicy",
        "admitted_recipe_ids": (ONE_SESSION_RECIPE_ID,),
        "development_split_semantics": "NON_OVERLAPPING_SINGLE_SESSION_LABEL",
    }
    return ExecutionOutcomePublicationPolicy(**values, policy_hash=canonical_hash(values))


def build_development_only_execution_outcome_publication_policy() -> (
    ExecutionOutcomePublicationPolicy
):
    """Development-only admission for the five-session method.

    Separate from the one-session policy rather than an extra id inside it, for
    two reasons that both matter. Its label semantics genuinely differ -- a
    six-session span overlaps labels, so the development split cannot be cut at a
    sealed boundary -- and merging the two would mean one identity covering two
    incompatible publication contracts, which is the shape of authority that
    cannot refuse anything.
    """
    values: dict[str, object] = {
        "kind": "ExecutionOutcomePublicationPolicy",
        "admitted_recipe_ids": (FIVE_SESSION_RECIPE_ID,),
        "development_split_semantics": "OVERLAPPING_MULTI_SESSION_LABEL",
    }
    return ExecutionOutcomePublicationPolicy(**values, policy_hash=canonical_hash(values))


class ExecutionOutcomePublicationPolicyCatalog:
    """The installed publication policies, indexed by the recipe each admits.

    Resolution is by recipe id, so a binding cannot name one method and carry
    another method's admission. A recipe admitted by two policies is refused at
    construction rather than resolved by whichever happened to be registered
    first -- ambiguous admission is not admission.
    """

    def __init__(self, policies: tuple[ExecutionOutcomePublicationPolicy, ...]) -> None:
        """Install policies with exactly one admission per recipe ID.

        Args:
            policies: Explicit publication policies to index.

        Raises:
            ExecutionOutcomeMethodError: If the set is empty or admission overlaps.

        """
        if not policies:
            raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_PUBLICATION_CATALOG_INVALID")
        indexed: dict[str, ExecutionOutcomePublicationPolicy] = {}
        for policy in policies:
            for recipe_id in policy.admitted_recipe_ids:
                if recipe_id in indexed:
                    raise ExecutionOutcomeMethodError(
                        "EXECUTION_OUTCOME_PUBLICATION_RECIPE_DOUBLY_ADMITTED"
                    )
                indexed[recipe_id] = policy
        self._by_recipe = MappingProxyType(indexed)
        self._policies = policies

    @property
    def policy_hashes(self) -> frozenset[str]:
        """Return the identities of installed publication policies."""
        return frozenset(value.policy_hash for value in self._policies)

    def resolve(self, recipe_id: str) -> ExecutionOutcomePublicationPolicy:
        """Return the policy admitting a recipe or refuse an unadmitted one."""
        policy = self._by_recipe.get(recipe_id)
        if policy is None:
            raise ExecutionOutcomeMethodError(
                "causal_outcomes.publication_contract_not_installed_for_recipe"
            )
        return policy


def build_installed_execution_outcome_publication_policy_catalog() -> (
    ExecutionOutcomePublicationPolicyCatalog
):
    """Both installed policies; no discovery and no mutation."""
    return ExecutionOutcomePublicationPolicyCatalog(
        (
            build_installed_execution_outcome_publication_policy(),
            build_development_only_execution_outcome_publication_policy(),
        )
    )


class ExecutionOutcomeMethodSealMarker(_Contract):
    """Terminal authority for one method-bound snapshot.

    The frozen ``CausalExecutionOutcomeMarker`` cannot carry a seal reference
    without changing its bytes, so the chain used to stop at the manifest and a
    reader had to scan a directory for a binding that happened to name the same
    snapshot. Scanning is a guess: it cannot tell "this snapshot was published
    under this method" from "somebody put a well-formed file here". This marker
    is the missing edge, and it is written last, so its presence is what makes
    a snapshot method-bound:

        seal marker -> outcome marker -> manifest -> exact binding
    """

    kind: Literal["ExecutionOutcomeMethodSealMarker"] = "ExecutionOutcomeMethodSealMarker"
    outcome_marker_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome_marker_ref: str = Field(min_length=1)
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_ref: str = Field(min_length=1)
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_ref: str = Field(min_length=1)
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    publication_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    seal_marker_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify the terminal method-seal marker hash."""
        if self.seal_marker_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"seal_marker_hash"})
        ):
            raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_SEAL_MARKER_IDENTITY_INVALID")
        return self


def build_execution_outcome_method_seal_marker(
    *,
    outcome_marker_hash: str,
    outcome_marker_ref: str,
    snapshot_hash: str,
    manifest_ref: str,
    binding: ExecutionOutcomeMethodBinding,
    binding_ref: str,
) -> ExecutionOutcomeMethodSealMarker:
    """Seal the terminal link from a snapshot marker to its method binding."""
    values: dict[str, object] = {
        "kind": "ExecutionOutcomeMethodSealMarker",
        "outcome_marker_hash": outcome_marker_hash,
        "outcome_marker_ref": outcome_marker_ref,
        "snapshot_hash": snapshot_hash,
        "manifest_ref": manifest_ref,
        "binding_hash": binding.binding_hash,
        "binding_ref": binding_ref,
        "catalog_hash": binding.catalog_hash,
        "publication_policy_hash": binding.publication_policy_hash,
    }
    return ExecutionOutcomeMethodSealMarker(**values, seal_marker_hash=canonical_hash(values))


def verify_execution_outcome_method_binding(
    *,
    binding: ExecutionOutcomeMethodBinding,
    snapshot_hash: str,
    schedule_hash: str,
    ordered_session_triples_hash: str,
    catalog: ExecutionOutcomeMethodCatalog,
    expected_catalog_hash: str,
    policies: ExecutionOutcomePublicationPolicyCatalog,
) -> ExecutionOutcomeMethodRecipe:
    """Rebuild the binding the Host would have sealed, and require it exactly.

    Comparing a subset of fields only proves internal consistency: a binding
    that names an uninstalled catalog, an unadmitted publication policy, or
    offsets the installed recipe does not have would pass every such check while
    describing a computation nobody authorized. So nothing here trusts a field
    on the supplied binding except the two values that are genuinely inputs
    rather than authority -- the source watermark and the action lineage. Every
    other field is re-derived from the installed catalog, the installed
    publication policy and the manifest, and the whole model must match.

    The policy arrives as the installed *catalog* rather than as one expected
    hash, and is resolved by the recipe the binding names. Handing in a single
    hash meant the caller decided which admission applied, so a five-session
    binding presented alongside the one-session policy hash would have been
    compared against an admission that never covered it.
    """
    recipe = catalog.resolve(binding.recipe_id)
    expected = build_execution_outcome_method_binding(
        recipe=recipe,
        catalog_hash=expected_catalog_hash,
        publication_policy_hash=policies.resolve(recipe.recipe_id).policy_hash,
        snapshot_hash=snapshot_hash,
        schedule_hash=schedule_hash,
        ordered_session_triples_hash=ordered_session_triples_hash,
        source_watermark_hash=binding.source_watermark_hash,
        action_lineage=binding.action_lineage_hash,
    )
    if binding != expected:
        raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_BINDING_NOT_HOST_AUTHORITATIVE")
    return recipe


def verify_execution_outcome_method_seal_marker(
    *,
    seal_marker: ExecutionOutcomeMethodSealMarker,
    outcome_marker: SealableOutcomeMarker,
    outcome_marker_ref: str,
    manifest: SealableOutcomeManifest,
    manifest_ref: str,
    binding: ExecutionOutcomeMethodBinding,
    binding_ref: str,
    catalog: ExecutionOutcomeMethodCatalog,
    expected_catalog_hash: str,
    policies: ExecutionOutcomePublicationPolicyCatalog,
) -> ExecutionOutcomeMethodRecipe:
    """Rebuild the terminal marker the Host would have written, and require it exactly.

    The one verifier every authority consumer shares -- publisher readback,
    reader resolution, and Factor admission. Its inputs are the Host-resolved
    graph, not the marker's own claims: the manifest and its ref come from the
    caller's authority, the outcome marker must name that manifest, the binding
    must be rebuildable from the installed catalog and publication policy, and
    only then is the expected terminal marker constructed and compared by whole
    model. A marker that is internally consistent but cites a wrong child hash,
    a wrong ref, an uninstalled catalog or policy, or another snapshot's graph
    therefore fails on the same single comparison rather than on whichever
    partial field check a particular caller happened to keep.
    """
    if (
        outcome_marker.snapshot_hash != manifest.snapshot_hash
        or outcome_marker.manifest_ref != manifest_ref
        or outcome_marker.schedule_hash != manifest.schedule_hash
        or outcome_marker.listing_set_hash != manifest.listing_set_hash
    ):
        raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_SEAL_CHILD_GRAPH_MISMATCH")
    if (
        binding.price_basis != manifest.price_basis
        or binding.corporate_action_identity != manifest.corporate_action_identity
        or binding.return_formula_identity != manifest.return_formula_identity
    ):
        raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_SEAL_CHILD_GRAPH_MISMATCH")
    recipe = verify_execution_outcome_method_binding(
        binding=binding,
        snapshot_hash=manifest.snapshot_hash,
        schedule_hash=manifest.schedule_hash,
        ordered_session_triples_hash=manifest.ordered_session_triples_hash,
        catalog=catalog,
        expected_catalog_hash=expected_catalog_hash,
        policies=policies,
    )
    expected = build_execution_outcome_method_seal_marker(
        outcome_marker_hash=outcome_marker.marker_hash,
        outcome_marker_ref=outcome_marker_ref,
        snapshot_hash=manifest.snapshot_hash,
        manifest_ref=manifest_ref,
        binding=binding,
        binding_ref=binding_ref,
    )
    if seal_marker != expected:
        raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_SEAL_MARKER_NOT_HOST_AUTHORITATIVE")
    return recipe


@dataclass(frozen=True, slots=True)
class ExecutionOutcomeMethodSeal:
    """What an authoritative resolver may say about one snapshot's method authority.

    Deliberately a disposition rather than an optional binding. ``None`` invites
    a caller to treat "no seal" as "no constraint"; ``LEGACY_READBACK_ONLY``
    says the rows are readable and the authority is absent, and ``method_bound``
    refuses to hand out a binding that does not exist.

    A method-bound seal carries the resolved graph -- terminal marker, outcome
    marker and exact binding -- so a downstream consumer can re-run the same
    full verifier against the manifest it was handed, instead of trusting that
    some earlier resolver did. The resolver that constructs one has already
    verified the graph; the fields here make that verifiable again, not assumed.
    """

    disposition: Literal["METHOD_BOUND", "LEGACY_READBACK_ONLY"]
    seal_marker: ExecutionOutcomeMethodSealMarker | None = None
    outcome_marker: SealableOutcomeMarker | None = None
    binding: ExecutionOutcomeMethodBinding | None = None

    def __post_init__(self) -> None:
        """Require a complete, consistent graph for method-bound authority."""
        seal_marker, outcome_marker, binding = (
            self.seal_marker,
            self.outcome_marker,
            self.binding,
        )
        if self.disposition == "LEGACY_READBACK_ONLY":
            if seal_marker is not None or outcome_marker is not None or binding is not None:
                raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_SEAL_LEGACY_CARRIES_AUTHORITY")
            return
        if seal_marker is None or outcome_marker is None or binding is None:
            raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_SEAL_GRAPH_INCOMPLETE")
        if (
            seal_marker.binding_hash != binding.binding_hash
            or seal_marker.snapshot_hash != binding.snapshot_hash
            or seal_marker.outcome_marker_hash != outcome_marker.marker_hash
            or seal_marker.snapshot_hash != outcome_marker.snapshot_hash
        ):
            raise ExecutionOutcomeMethodError("EXECUTION_OUTCOME_SEAL_GRAPH_INCONSISTENT")

    @property
    def method_bound(self) -> ExecutionOutcomeMethodBinding:
        """Return the verified binding or refuse legacy readback authority."""
        if self.disposition != "METHOD_BOUND" or self.binding is None:
            raise PermissionError("EXECUTION_OUTCOME_METHOD_AUTHORITY_UNAVAILABLE")
        return self.binding


__all__ = [
    "AXIS_SESSION_HALF_OPEN_ATTRIBUTION",
    "FIVE_SESSION_RECIPE_ID",
    "ONE_SESSION_RECIPE_ID",
    "ExecutionOutcomeMethodBinding",
    "ExecutionOutcomeMethodCatalog",
    "ExecutionOutcomeMethodCatalogBinding",
    "ExecutionOutcomeMethodError",
    "ExecutionOutcomeMethodIdentity",
    "ExecutionOutcomeMethodRecipe",
    "ExecutionOutcomeMethodSeal",
    "ExecutionOutcomeMethodSealMarker",
    "ExecutionOutcomePublicationPolicy",
    "ExecutionOutcomePublicationPolicyCatalog",
    "ExecutionOutcomeSessionAxis",
    "SealableOutcomeManifest",
    "SealableOutcomeMarker",
    "action_lineage_hash",
    "build_development_only_execution_outcome_publication_policy",
    "build_execution_outcome_method_binding",
    "build_execution_outcome_method_seal_marker",
    "build_five_session_recipe",
    "build_installed_execution_outcome_method_catalog",
    "build_installed_execution_outcome_publication_policy",
    "build_installed_execution_outcome_publication_policy_catalog",
    "build_one_session_recipe",
    "period_dividend_for_point",
    "resolve_final_schedule_point",
    "resolve_schedule_points",
    "verify_execution_outcome_method_binding",
    "verify_execution_outcome_method_seal_marker",
]
