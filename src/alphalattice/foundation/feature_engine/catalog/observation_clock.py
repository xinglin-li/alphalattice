"""Three clocks that must stay apart, and the one this module owns.

``Feature(T)`` is the value of a Formula on observation session ``T``. Feature
arithmetic knows nothing about strategy: no entry, no ``open(T+1)``, no rebalance
cadence, and no automatic shift for "tradability". A Formula that gives up recent
sessions does so because its own economics say to -- ``mom_252_21`` excludes the
most recent 21 sessions on purpose -- and that skip is Formula meaning, not
execution lag.

The three clocks, and who owns each:

``FeatureObservationClock``      *this module*. Which session a Formula's value
                                belongs to, what it actually consumed, and how
                                much history it needed. Nothing about when the
                                value may be read.
``FeatureAvailabilityPolicy``    *this module*, but one installed policy for the
                                whole catalog rather than seventy copies. It says
                                when an observation becomes selectable: for daily
                                provider bars, after the official close of the
                                observation session. A Provider publication delay
                                moves this policy and must not make any Formula's
                                arithmetic or methodology identity pretend to
                                have changed.
strategy / execution             *the Host*. ``close(T)`` decision, ``open(T+1)``
                                entry, ``open(T+2)`` exit. The Host asks which
                                observations are already available at a decision
                                event; it never earns a margin by editing a
                                Formula. This module imports nothing from
                                ``causal_outcomes.execution`` and the contracts
                                below forbid extra fields, so an entry or exit
                                offset cannot enter Feature identity.

Two owners are deliberately absent. Model-side transformations -- ``lag1``,
rolling means, deltas, EWMA -- start from a correct ``Feature(T)`` and belong to
the consuming Desk's input descriptor; ``shift(feature, k)`` is a projection of
an existing Feature, never a new one. Entry and exit belong to the execution
method.

One integer used to answer four of these questions at once. ``FactorSpec`` is a
frozen wire contract owned by the parent ``alphalattice.kernel.quant`` package, so its
``lag_sessions`` field cannot be renamed here; it is bridged into successor
semantics through exactly one function, :func:`formula_skip_sessions`, and every
active Feature-owned surface speaks of ``formula_skip_sessions`` instead.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from enum import StrEnum
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.quant.factor_contracts import FactorSpec
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import DecisionCutoff, MarketPhase

FEATURE_OBSERVATION_CLOCK_POLICY_ID = "feature-observation-clock.true-observation-session"
"""The installed successor semantics, named so an artifact cannot imply them.

A Panel published before this policy carries no such identity and cannot acquire
one by being read: the catalog refuses a payload that does not declare it.
"""

DAILY_PROVIDER_SESSION_CLOSE_AVAILABILITY = "feature-availability.daily-provider-session-close"

SOURCE_AVAILABILITY_CATALOG_ID = "feature-availability.installed-source-authorities"
"""The set of source-availability owners this build has installed.

One owner per *source authority*, never per Factor. The predecessor held a single
policy whose basis read ``PROVIDER_DAILY_OHLCV_AND_PROVIDER_ADJUSTED_CLOSE``,
which describes the price feed accurately and then silently answered for three
things that are not it: the Sector classification map, a Market or Sector row
derived from the feed, and one Formula consuming another Formula's published
output. Those have different owners and different revision stories, and a build
that cannot say so cannot refuse a source it has never heard of either.
"""

PROVIDER_DAILY_BARS_AUTHORITY = "source-authority.provider-daily-bars"
PROVIDER_AS_TRADED_AUTHORITY = "source-authority.provider-daily-bars-as-traded"
"""Each session's prices and share volume as they traded, rebuilt from the provider's bars and
its recorded splits (V345): the owner of the point-in-time fields."""
MARKET_REFERENCE_AUTHORITY = "source-authority.market-reference-derived"
SECTOR_AGGREGATE_AUTHORITY = "source-authority.sector-aggregate-derived"
SECTOR_CLASSIFICATION_AUTHORITY = "source-authority.sector-classification-map"
VERIFIED_PANEL_CHILD_AUTHORITY = "source-authority.verified-panel-child"

LEGACY_SKIP_WIRE_FIELD: Final = "lag_sessions"
"""The frozen ``FactorSpec`` field the successor reads through one bridge.

Named rather than spelled inline so the compatibility surface is greppable and
so no authoring site can reintroduce it as a general-purpose knob.
"""

NO_ECONOMIC_SKIP: Final = 0
"""What an authoring site writes when a Formula ends at its observation session.

Spelled as a constant at every recipe that declares it, because ``0`` on its own
is exactly the value the old execution-safety reading used to overwrite.
"""

LEGACY_FEATURE_CLOCK = "CLOSE_T_MINUS_1_MASQUERADING_AS_T"
"""What the measured pre-successor build did, recorded as a name, not a mode.

There is no switch that selects it. It exists so a report, a test and a refusal
can spell the defect the same way.
"""

NEW_FEATURE_OBSERVATION_CLOCK = "TRUE_OBSERVATION_SESSION"

SourceIntervalKind = Literal["CONTIGUOUS_SESSION_WINDOW", "CALENDAR_MONTH_SELECTION"]


class FeatureObservationClockError(ValueError):
    """Stable refusal for a clock that cannot be derived or does not match."""


class FeatureAvailabilityError(ValueError):
    """Stable refusal raised when a decision event cannot select observations."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class PointInTimeMark(StrEnum):
    """What a later event does to a source field's stored value at a session (V345).

    ``POINT_IN_TIME``: nothing; the value is the session's as it was published.
    ``POINT_IN_TIME_FROM_T0``: nothing from T0; before it the cohort's or the classification's
    first recorded state stands in, as a study's temporal statement says (V346, V347).
    ``RESCALED_BY_LATER_EVENTS``: a later split or dividend rescales the history before it by one
    factor, so a ratio of its values at or before a session holds and a level does not; the
    formula language admits it only in forms that hold.
    """

    POINT_IN_TIME = "POINT_IN_TIME"
    POINT_IN_TIME_FROM_T0 = "POINT_IN_TIME_FROM_T0"
    RESCALED_BY_LATER_EVENTS = "RESCALED_BY_LATER_EVENTS"


class FormulaSourceInterval(_Contract):
    """What a Formula actually consumed, stated by kind rather than generalised.

    The predecessor derived one interval for every Formula as
    ``[t-(minimum_observations-1), t-skip]``. That is true of a contiguous
    price/return window and false of everything else: ``seasonality_12m`` selects
    a calendar month twelve months back and reads no session in between, and
    ``market_drawdown_x_momentum`` needs 63 rows of history for its Market state
    child while its own stock leg reads only ``t`` and ``t-20``.

    So two things are separated here. ``latest_consumed_offset_sessions`` is
    exact and is the property a causal claim depends on -- ``None`` when the
    Formula's newest input is not expressible as a session offset at all.
    ``minimum_history_rows`` is a *bound*: no Formula reads further back than it,
    and only a mechanical window reads exactly that far.
    """

    kind: SourceIntervalKind
    minimum_history_rows: int = Field(ge=1)
    minimum_history_is_mechanical: bool
    """True when the first finite output lands exactly at ``minimum_history_rows - 1``.

    False for a calendar-driven Formula, whose first value waits on calendar
    months rather than on a row count, and whose declared row count is nominal.
    """

    latest_consumed_offset_sessions: int | None
    latest_consumed_event: str = Field(min_length=1)
    earliest_consumed_bound_offset_sessions: int | None
    notes: str = Field(min_length=1)
    interval_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Create an interval with its content hash."""
        identity = cls.model_construct(**values, interval_hash="0" * 64).model_dump(
            mode="json", exclude={"interval_hash"}
        )
        return cls(**values, interval_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_interval(self) -> Self:
        """Verify the interval bounds and content hash."""
        if self.kind == "CONTIGUOUS_SESSION_WINDOW":
            if self.latest_consumed_offset_sessions is None:
                raise FeatureObservationClockError(
                    "feature_engine.source_interval_contiguous_offset_absent"
                )
            if self.earliest_consumed_bound_offset_sessions is None:
                raise FeatureObservationClockError(
                    "feature_engine.source_interval_contiguous_bound_absent"
                )
            if self.earliest_consumed_bound_offset_sessions < self.latest_consumed_offset_sessions:
                raise FeatureObservationClockError("feature_engine.source_interval_inverted")
        elif self.minimum_history_is_mechanical:
            raise FeatureObservationClockError("feature_engine.source_interval_calendar_mechanical")
        if self.interval_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"interval_hash"})
        ):
            raise FeatureObservationClockError("feature_engine.source_interval_identity_invalid")
        return self

    @property
    def rendered(self) -> str:
        """The interval relative to ``t``, or the calendar rule when it is one."""
        if self.kind != "CONTIGUOUS_SESSION_WINDOW":
            return self.latest_consumed_event
        last = self.latest_consumed_offset_sessions or 0
        first = self.earliest_consumed_bound_offset_sessions or 0
        return f"[t-{first},{'t' if not last else f't-{last}'}]"


class FeatureObservationClock(_Contract):
    """One Formula's observation semantics, derived from the recipe that owns it.

    Availability is deliberately not a field. A Formula's clock says which
    session its value belongs to and what it read; when that value may be *used*
    is a property of the source, held once for the whole catalog. Folding the two
    together made seventy copies of one delay and would have rotated seventy
    method identities the day a Provider changed its publication schedule.
    """

    kind: Literal["FeatureObservationClock"] = "FeatureObservationClock"
    policy_id: str = Field(min_length=1, max_length=96)
    factor_id: str = Field(min_length=1, max_length=128)
    observation_session: Literal["FORMULA_FORMATION_SESSION"] = "FORMULA_FORMATION_SESSION"
    declared_window_sessions: int = Field(ge=1)
    """The recipe's declared span. Deliberately *not* the source-row count.

    The installed catalog uses it for two different things -- ``mom_252_21``
    declares the 252-session span it skips 21 of, while ``residual_mom_252_21``
    declares the 252 returns its estimation window holds -- so nothing derives an
    interval from it.
    """

    formula_skip_sessions: int = Field(ge=0)
    source_interval: FormulaSourceInterval
    clock_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Create a Formula observation clock with its content hash."""
        payload = {"kind": "FeatureObservationClock", **values}
        identity = cls.model_construct(**payload, clock_hash="0" * 64).model_dump(
            mode="json", exclude={"clock_hash"}
        )
        return cls(**payload, clock_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify the Formula skip and clock identity."""
        interval = self.source_interval
        if interval.minimum_history_rows <= self.formula_skip_sessions:
            raise FeatureObservationClockError("feature_engine.observation_clock_rows_invalid")
        if (
            interval.kind == "CONTIGUOUS_SESSION_WINDOW"
            and interval.latest_consumed_offset_sessions != self.formula_skip_sessions
        ):
            raise FeatureObservationClockError("feature_engine.observation_clock_skip_mismatch")
        if self.clock_hash != canonical_hash(self.model_dump(mode="json", exclude={"clock_hash"})):
            raise FeatureObservationClockError("feature_engine.observation_clock_identity_invalid")
        return self

    @property
    def latest_consumed_session_offset(self) -> int | None:
        """Sessions back from the observation session to the newest consumed row."""
        return self.source_interval.latest_consumed_offset_sessions

    @property
    def source_interval_rendered(self) -> str:
        """Render the source interval relative to its observation session."""
        return self.source_interval.rendered


class FeatureAvailabilityPolicy(_Contract):
    """When an observation session's Feature value becomes selectable.

    One policy per *source authority*, never one per Formula. Seventy per-Factor
    delay states would be a configuration surface whose copies can only disagree,
    and any change to one would drag seventy method identities behind it. The
    number of policies is therefore the number of distinct source owners the
    installed Formulas actually read -- five today -- and a Formula names the
    ones it needs rather than restating their schedules.

    ``revision_risk`` is stated rather than mitigated. The provider's adjusted
    series is a mutable backward projection, so a historical restatement can
    change values already published. A one-session lag protects against none of
    that: a restatement reaching ``T-1`` reaches ``T-2`` equally. Recording the
    residual risk is honest; paying a lag for it is not point-in-time authority.
    """

    policy_id: str = Field(min_length=1, max_length=96)
    observation_basis: str = Field(min_length=1, max_length=96)
    available_after_phase: MarketPhase = MarketPhase.OFFICIAL_CLOSE
    publication_delay_sessions: int = Field(ge=0)
    revision_risk: str = Field(min_length=1)
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Create a source availability policy with its content hash."""
        identity = cls.model_construct(**values, policy_hash="0" * 64).model_dump(
            mode="json", exclude={"policy_hash"}
        )
        return cls(**values, policy_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify the source availability policy hash."""
        if self.policy_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"policy_hash"})
        ):
            raise FeatureObservationClockError(
                "feature_engine.availability_policy_identity_invalid"
            )
        return self


def formula_skip_sessions(specification: FactorSpec) -> int:
    """Read the frozen wire field using successor skip semantics.

    ``FactorSpec`` belongs to the parent ``alphalattice.kernel.quant`` package and is
    shared by every worktree, so the field cannot be renamed here. This is the
    only place the Feature domain reads it, which is what keeps
    ``lag_sessions`` from drifting back into a general-purpose knob: every other
    surface -- maintenance contracts, materializer, invalidation, catalog audit,
    Panel lineage -- speaks of ``formula_skip_sessions``.
    """
    value = int(getattr(specification, LEGACY_SKIP_WIRE_FIELD))
    if value < 0:
        raise FeatureObservationClockError("feature_engine.formula_skip_sessions_invalid")
    return value


class SourceAvailabilityCatalog(_Contract):
    """Every installed source-availability owner, and which fields each answers for.

    The field map is the whole point. A Formula declares ``required_fields``; this
    resolves each one to the authority that publishes it, so a Formula's
    availability is *derived* from what it reads rather than assumed from what the
    majority of Formulas read. A field with no installed owner is refused, which
    is what makes an externally authored Formula over a non-OHLCV source a typed
    failure instead of a silent inheritance of the price feed's schedule.
    """

    kind: Literal["SourceAvailabilityCatalog"] = "SourceAvailabilityCatalog"
    catalog_id: str = Field(min_length=1, max_length=96)
    owners: tuple[FeatureAvailabilityPolicy, ...] = Field(min_length=1)
    field_authorities: Mapping[str, str]
    point_in_time: Mapping[str, PointInTimeMark] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    """Each field's point-in-time mark (V345); a catalog recorded before the marks holds none."""
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Create a source authority catalog with its content hash."""
        identity = cls.model_construct(**values, catalog_hash="0" * 64).model_dump(
            mode="json", exclude={"catalog_hash"}
        )
        return cls(**values, catalog_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_catalog(self) -> Self:
        """Verify the source owner axis, field map, and catalog hash."""
        ids = [item.policy_id for item in self.owners]
        if ids != sorted(set(ids)):
            raise FeatureObservationClockError(
                "feature_engine.source_availability_owner_axis_invalid"
            )
        installed = set(ids)
        for field, authority in self.field_authorities.items():
            if authority not in installed:
                raise FeatureObservationClockError(
                    f"feature_engine.source_availability_owner_uninstalled:{field}"
                )
        if self.point_in_time and set(self.point_in_time) != set(self.field_authorities):
            raise FeatureObservationClockError(
                "feature_engine.source_point_in_time_marks_incomplete"
            )
        if self.catalog_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"catalog_hash"})
        ):
            raise FeatureObservationClockError(
                "feature_engine.source_availability_catalog_identity_invalid"
            )
        return self

    def owner(self, policy_id: str) -> FeatureAvailabilityPolicy:
        """Return the installed source owner for a policy ID."""
        for item in self.owners:
            if item.policy_id == policy_id:
                return item
        raise FeatureAvailabilityError(
            f"feature_engine.source_availability_owner_uninstalled:{policy_id}"
        )

    def authorities_for(self, fields: Iterable[str]) -> tuple[str, ...]:
        """Return the ordered authority set one Formula depends on.

        Sorted and deduplicated, so the identity below is a property of *which*
        owners a Formula needs and not of the order its recipe happens to list
        its fields in.
        """
        resolved: set[str] = set()
        for field in fields:
            authority = self.field_authorities.get(field)
            if authority is None:
                raise FeatureAvailabilityError(
                    f"feature_engine.source_availability_owner_uninstalled:{field}"
                )
            resolved.add(authority)
        if not resolved:
            raise FeatureAvailabilityError(
                "feature_engine.source_availability_owner_uninstalled:<no-required-fields>"
            )
        return tuple(sorted(resolved))

    def effective_policy(self, authorities: Iterable[str]) -> FeatureAvailabilityPolicy:
        """Return the latest lawful availability event across the authorities given.

        A Formula is selectable only once *every* source it reads is, so the
        governing event is the maximum -- by publication delay first, then by
        market phase. A Panel takes the maximum across all of its Formulas, which
        is why the Host can still make one comparison.
        """
        owners = [self.owner(item) for item in authorities]
        if not owners:
            raise FeatureAvailabilityError(
                "feature_engine.source_availability_owner_uninstalled:<empty>"
            )
        return max(
            owners,
            key=lambda item: (item.publication_delay_sessions, int(item.available_after_phase)),
        )


def installed_source_availability_catalog() -> SourceAvailabilityCatalog:
    """Return the installed owners and the fields each answers for.

    **These schedules are installed policy, not measurement.** Every owner here
    declares official close with zero publication delay, and that is this
    repository asserting what it believes about its sources -- it is not derived
    from provider publication receipts, delivery timestamps or any observed
    lateness record, because no such evidence is collected today. The assertion
    is defensible for an end-of-day surface, and it is still an assertion. Read
    these as the schedule a run is admitted under, and treat a change to one as
    a decision someone made rather than a fact someone observed.

    Two consequences worth stating rather than leaving to be discovered:

    * ``VERIFIED_PANEL_CHILD_AUTHORITY`` says a Formula reading another Formula's
      output is available when its parent is. That is true and it is **not
      transitively derived**: this build does not walk a parent's own authority
      set and take the maximum over it. With every installed owner on the same
      event the two answers coincide, so nothing is wrong today; introduce an
      owner with a real delay and the transitive closure would have to be
      computed rather than assumed.
    * separating the owners is what makes a future schedule change expressible at
      all. It does not, by itself, make any of these five schedules evidence.
    """
    provider_revision = (
        "The provider's adjusted series is a mutable backward projection: a "
        "restatement can change values already published, and no session of lag "
        "bounds it -- a restatement reaching T-1 reaches T-2 equally, so paying a "
        "lag for it is not point-in-time authority. Stated rather than mitigated."
    )
    owners = (
        FeatureAvailabilityPolicy.create(
            policy_id=MARKET_REFERENCE_AUTHORITY,
            observation_basis="MARKET_REFERENCE_ROW_DERIVED_FROM_PROVIDER_DAILY",
            available_after_phase=MarketPhase.OFFICIAL_CLOSE,
            publication_delay_sessions=0,
            revision_risk=(
                "Derived inside this repository from the provider daily series, so "
                "it inherits that series' restatement risk exactly and adds none."
            ),
        ),
        FeatureAvailabilityPolicy.create(
            policy_id=PROVIDER_DAILY_BARS_AUTHORITY,
            observation_basis="PROVIDER_DAILY_OHLCV_AND_PROVIDER_ADJUSTED_CLOSE",
            available_after_phase=MarketPhase.OFFICIAL_CLOSE,
            publication_delay_sessions=0,
            revision_risk=provider_revision,
        ),
        FeatureAvailabilityPolicy.create(
            policy_id=PROVIDER_AS_TRADED_AUTHORITY,
            observation_basis="PROVIDER_DAILY_OHLCV_AS_TRADED_FROM_RECORDED_SPLITS",
            available_after_phase=MarketPhase.OFFICIAL_CLOSE,
            publication_delay_sessions=0,
            revision_risk=(
                "Point in time: each session's prices and share volume as they traded, rebuilt "
                "from the provider's split-basis bars and every split it recorded, so a later "
                "split moves nothing; a split the provider records late, or a ratio it restates, "
                "moves the sessions before it until recorded. A fractional ratio is a price "
                "adjustment, not a share split (V348): the share volume before one is assumed "
                "scaled by it as by a split, which cannot be checked offline."
            ),
        ),
        FeatureAvailabilityPolicy.create(
            policy_id=SECTOR_AGGREGATE_AUTHORITY,
            observation_basis="SECTOR_AGGREGATE_DERIVED_FROM_PROVIDER_DAILY_AND_SECTOR_MAP",
            available_after_phase=MarketPhase.OFFICIAL_CLOSE,
            publication_delay_sessions=0,
            revision_risk=(
                "Two parents: the provider daily series and the Sector "
                "classification map. It restates when either does, which makes it "
                "strictly riskier than the price feed alone."
            ),
        ),
        FeatureAvailabilityPolicy.create(
            policy_id=SECTOR_CLASSIFICATION_AUTHORITY,
            observation_basis="SECTOR_CLASSIFICATION_MAP_CURRENT_BACKFILLED",
            available_after_phase=MarketPhase.OFFICIAL_CLOSE,
            publication_delay_sessions=0,
            revision_risk=(
                "Not point-in-time. The installed map is the current "
                "classification backfilled across history, so a refresh restates "
                "the whole history rather than appending to it. This is the "
                "authority the single price-feed policy used to answer for "
                "silently, and it is the reason these are separate owners."
            ),
        ),
        FeatureAvailabilityPolicy.create(
            policy_id=VERIFIED_PANEL_CHILD_AUTHORITY,
            observation_basis="INSTALLED_FORMULA_OUTPUT_ON_THE_SAME_OBSERVATION_SESSION",
            available_after_phase=MarketPhase.OFFICIAL_CLOSE,
            publication_delay_sessions=0,
            revision_risk=(
                "A Formula reading another Formula's published value inherits "
                "every parent's risk transitively; it introduces no source of its "
                "own, and it is available exactly when its parent is."
            ),
        ),
    )
    provider_fields = (
        "open_raw",
        "high_raw",
        "low_raw",
        "close_raw",
        "volume_raw",
        "open_split_adjusted",
        "high_split_adjusted",
        "low_split_adjusted",
        "close_split_adjusted",
        "provider_adjusted_close",
    )
    as_traded_fields = (
        "open_as_traded",
        "high_as_traded",
        "low_as_traded",
        "close_as_traded",
        "volume_as_traded",
        "price_adjustment_index",
        "share_count_index",
    )
    field_authorities = {
        **{field: PROVIDER_DAILY_BARS_AUTHORITY for field in provider_fields},
        **{field: PROVIDER_AS_TRADED_AUTHORITY for field in as_traded_fields},
        "market_provider_adjusted_close": MARKET_REFERENCE_AUTHORITY,
        "market_return_log": MARKET_REFERENCE_AUTHORITY,
        "sector_return_log": SECTOR_AGGREGATE_AUTHORITY,
        "sector_membership_asof": SECTOR_CLASSIFICATION_AUTHORITY,
        "rev_5": VERIFIED_PANEL_CHILD_AUTHORITY,
    }
    # The provider's bars are in its share basis as of the download and its adjusted close is a
    # backward projection; the market reference and the Sector rows are built across a cohort
    # that is the first one before T0; a session's reversal is a ratio no later event moves.
    point_in_time = {
        **{field: PointInTimeMark.RESCALED_BY_LATER_EVENTS for field in provider_fields},
        **{field: PointInTimeMark.POINT_IN_TIME for field in as_traded_fields},
        "market_provider_adjusted_close": PointInTimeMark.RESCALED_BY_LATER_EVENTS,
        "market_return_log": PointInTimeMark.POINT_IN_TIME_FROM_T0,
        "sector_return_log": PointInTimeMark.POINT_IN_TIME_FROM_T0,
        "sector_membership_asof": PointInTimeMark.POINT_IN_TIME_FROM_T0,
        "rev_5": PointInTimeMark.POINT_IN_TIME,
    }
    return SourceAvailabilityCatalog.create(
        catalog_id=SOURCE_AVAILABILITY_CATALOG_ID,
        owners=owners,
        field_authorities=dict(sorted(field_authorities.items())),
        point_in_time=dict(sorted(point_in_time.items())),
    )


def formula_source_authority_binding_hash(
    authorities_by_factor: Mapping[str, tuple[str, ...]],
) -> str:
    """Identity over *which* owners each Formula needs, not over their schedules.

    This is the half that belongs beside a Formula. It moves when a Formula
    starts or stops reading a kind of source, and stands still when a Provider
    changes a publication time -- which is the separation the predecessor could
    not express, because it had one hash for both.
    """
    return str(
        canonical_hash(
            {
                "kind": "FormulaSourceAuthorityBinding",
                "authorities": {
                    factor_id: list(values)
                    for factor_id, values in sorted(authorities_by_factor.items())
                },
            }
        )
    )


def installed_feature_availability_policy() -> FeatureAvailabilityPolicy:
    """Return the governing availability event for the installed catalog.

    Derived, not declared: the latest event across every installed owner. The
    Host compares one decision cutoff against one policy, and that policy is now
    a consequence of which sources the Formulas read rather than an assertion
    that they all read the same one.
    """
    catalog = installed_source_availability_catalog()
    governing = catalog.effective_policy(item.policy_id for item in catalog.owners)
    return FeatureAvailabilityPolicy.create(
        policy_id=DAILY_PROVIDER_SESSION_CLOSE_AVAILABILITY,
        # Named for the derivation, not borrowed from one owner. Every installed
        # owner currently publishes at the same event, so ``max`` has a tie to
        # break and would otherwise hand this policy whichever owner happened to
        # sort first -- an arbitrary sentence that would move when an unrelated
        # owner was renamed.
        observation_basis="GOVERNING_EVENT_ACROSS_INSTALLED_SOURCE_AUTHORITIES",
        available_after_phase=governing.available_after_phase,
        publication_delay_sessions=governing.publication_delay_sessions,
        revision_risk=(
            "The union of every installed source authority's risk, which is not the "
            "provider feed's alone: the Sector classification map is a current "
            "classification backfilled across history, so a refresh restates history "
            "rather than appending to it, and a Formula reading it inherits that. "
            "Per-owner statements are on the owners; this is the governing event only."
        ),
    )


def _contiguous_interval(specification: FactorSpec) -> FormulaSourceInterval:
    skip = formula_skip_sessions(specification)
    rows = int(specification.minimum_observations)
    return FormulaSourceInterval.create(
        kind="CONTIGUOUS_SESSION_WINDOW",
        minimum_history_rows=rows,
        minimum_history_is_mechanical=True,
        latest_consumed_offset_sessions=skip,
        latest_consumed_event=(
            "official close of session t" if not skip else f"official close of session t-{skip}"
        ),
        earliest_consumed_bound_offset_sessions=rows - 1,
        notes=(
            "Ordered source rows ending at the observation session less this Formula's "
            "declared economic skip. The earliest offset is a bound: nothing older is "
            "read, and a mechanical window reads exactly that far."
        ),
    )


def _calendar_month_interval(specification: FactorSpec) -> FormulaSourceInterval:
    """Describe the interval for a Formula selecting a calendar month.

    ``minimum_history_rows`` carries the recipe's declared count and the notes
    below call it nominal, which is the whole of the claim: it is **not** the
    exact first finite row and **not** a lower bound on it. Thirteen calendar
    months of history is a calendar quantity and the number of trading sessions
    it spans varies; on the consumption audit's synthetic axis
    ``seasonality_12m`` first produces a finite value at row 280 against a
    declared 253. The audit therefore asserts only that a finite value appears
    (``FIRST_FINITE_EXISTS``) and never a row boundary in either direction.

    What the Formula does guarantee is the arithmetic, and that is unchanged:
    the observation month minus twelve, that month's final close measured
    against the preceding month's final close.

    This explanation is a docstring and not part of ``notes`` on purpose.
    ``notes`` is a field of ``FormulaSourceInterval`` and therefore inside
    ``interval_hash``, ``clock_hash``, ``formula_observation_policy_hash`` and
    ``catalog_hash``. Rewriting it to say something clearer would rotate every
    one of those and orphan a published Panel, which is far too high a price for
    a better sentence.
    """
    if formula_skip_sessions(specification) != NO_ECONOMIC_SKIP:
        raise FeatureObservationClockError("feature_engine.calendar_formula_declares_skip")
    return FormulaSourceInterval.create(
        kind="CALENDAR_MONTH_SELECTION",
        minimum_history_rows=int(specification.minimum_observations),
        minimum_history_is_mechanical=False,
        latest_consumed_offset_sessions=None,
        latest_consumed_event=(
            "official close of the last trading session of the calendar month twelve "
            "months before the observation month"
        ),
        earliest_consumed_bound_offset_sessions=None,
        notes=(
            "Selected by calendar month, not by a session offset: the value reads the "
            "final closes of the months twelve and thirteen back and no session in "
            "between, so no offset expresses it and the declared row count is nominal. "
            "The first finite observation waits on thirteen calendar months of history."
        ),
    )


def observation_clock_for(
    specification: FactorSpec,
    *,
    source_interval_kind: SourceIntervalKind = "CONTIGUOUS_SESSION_WINDOW",
) -> FeatureObservationClock:
    """Derive one Formula's clock from the recipe that already declares it.

    Derived rather than declared a second time. A clock stored beside the recipe
    is a number that can disagree with the window it describes, and the defect
    this module exists to close began as exactly that kind of disagreement. The
    interval *kind* is the one thing the recipe cannot state, and it comes from
    the catalog's own maintenance policy rather than from a second registry.
    """
    interval = (
        _calendar_month_interval(specification)
        if source_interval_kind == "CALENDAR_MONTH_SELECTION"
        else _contiguous_interval(specification)
    )
    return FeatureObservationClock.create(
        policy_id=FEATURE_OBSERVATION_CLOCK_POLICY_ID,
        factor_id=specification.factor_id,
        declared_window_sessions=int(specification.window_sessions),
        formula_skip_sessions=formula_skip_sessions(specification),
        source_interval=interval,
    )


def formula_observation_policy_hash(clocks: Iterable[FeatureObservationClock]) -> str:
    """One identity for a catalog's complete Formula observation authority.

    Availability is not in it. That separation is the point: changing a Provider
    publication schedule must move the availability authority and leave every
    Formula's observation identity exactly where it was.
    """
    ordered = sorted(clocks, key=lambda item: item.factor_id)
    if len({item.factor_id for item in ordered}) != len(ordered):
        raise FeatureObservationClockError("feature_engine.observation_clock_duplicated")
    return str(
        canonical_hash(
            {
                "kind": "FeatureFormulaObservationPolicy",
                "policy_id": FEATURE_OBSERVATION_CLOCK_POLICY_ID,
                "clocks": [
                    {"factor_id": item.factor_id, "clock_hash": item.clock_hash} for item in ordered
                ],
            }
        )
    )


def latest_selectable_observation_session(
    sessions: Sequence[date],
    *,
    decision_cutoff: DecisionCutoff,
    availability: FeatureAvailabilityPolicy,
) -> date | None:
    """Return the newest observation selectable at ``decision_cutoff``.

    The Host's selection rule, expressed once and here rather than at the Host,
    because "when is an observation available" is a property of the source. The
    Host supplies the decision event and compares; it never earns a margin by
    editing a Formula.

    A cutoff whose market phase is unstated is refused rather than assumed
    closed. A date alone cannot distinguish a pre-open decision from a post-close
    one, and guessing is exactly how a session of unearned information gets read.
    """
    if decision_cutoff.phase is MarketPhase.UNSPECIFIED:
        raise FeatureAvailabilityError("feature_engine.decision_cutoff_phase_ambiguous")
    ordered = sorted(set(sessions))
    eligible = [item for item in ordered if item <= decision_cutoff.session]
    if (
        eligible
        and eligible[-1] == decision_cutoff.session
        and decision_cutoff.phase < availability.available_after_phase
    ):
        # The decision is being taken before the session's observation exists.
        eligible = eligible[:-1]
    if not eligible:
        return None
    position = len(eligible) - 1 - availability.publication_delay_sessions
    return eligible[position] if position >= 0 else None


__all__ = [
    "DAILY_PROVIDER_SESSION_CLOSE_AVAILABILITY",
    "FEATURE_OBSERVATION_CLOCK_POLICY_ID",
    "LEGACY_FEATURE_CLOCK",
    "LEGACY_SKIP_WIRE_FIELD",
    "MARKET_REFERENCE_AUTHORITY",
    "NEW_FEATURE_OBSERVATION_CLOCK",
    "NO_ECONOMIC_SKIP",
    "PROVIDER_AS_TRADED_AUTHORITY",
    "PROVIDER_DAILY_BARS_AUTHORITY",
    "SECTOR_AGGREGATE_AUTHORITY",
    "SECTOR_CLASSIFICATION_AUTHORITY",
    "SOURCE_AVAILABILITY_CATALOG_ID",
    "VERIFIED_PANEL_CHILD_AUTHORITY",
    "FeatureAvailabilityError",
    "FeatureAvailabilityPolicy",
    "FeatureObservationClock",
    "FeatureObservationClockError",
    "FormulaSourceInterval",
    "PointInTimeMark",
    "SourceAvailabilityCatalog",
    "SourceIntervalKind",
    "formula_observation_policy_hash",
    "formula_skip_sessions",
    "formula_source_authority_binding_hash",
    "installed_feature_availability_policy",
    "installed_source_availability_catalog",
    "latest_selectable_observation_session",
    "observation_clock_for",
]
