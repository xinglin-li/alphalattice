"""What `ADVANCE_TO_WATERMARK` produces: lane coverage, domain receipts, one intersection.

Advancement and study slicing are different operations on different objects, and
the delivery plan separates them for a reason a date picker cannot express:
advancing asks upstream owners to *materialize* sessions and moves the common
support; re-slicing a study window asks nobody for anything and moves only a
report. Collapsing them into one `date_range` is how a cheap reporting change
starts refitting models.

Three rules give the types here their shape.

**Coverage is an exact axis, not an interval.** Two lanes can share a first
session, a last session and a count while disagreeing about a day in the middle,
and endpoints alone call that agreement. So every coverage binds the ordered
session axis and the ordered listing axis by content, and a gap, a duplicate or a
reordering changes the hash.

**Owners report their own work.** A receipt carries the counts its owner
measured. The Host never derives "three sessions requested, therefore three Risk
updates": that arithmetic is right until the day an owner reuses two of them, and
then it is a fabricated number in a durable artifact.

**A correction is not an extension.** Work can be required when the watermark
does not move at all, so a rebuilt lane has its own disposition and its own
lineage, and "no new sessions" never implies "no work".
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from itertools import pairwise
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sealing import seal_model_from_dump

AdvancementLaneId = Literal[
    "DATA_READINESS",
    "FEATURE",
    "ALPHA",
    "RISK",
    "MARK_TRADABILITY",
    "PORTFOLIO_STATE",
    "REPORT_PROJECTION",
]

ADVANCEMENT_LANE_ORDER: tuple[AdvancementLaneId, ...] = (
    "DATA_READINESS",
    "FEATURE",
    "ALPHA",
    "RISK",
    "MARK_TRADABILITY",
    "PORTFOLIO_STATE",
    "REPORT_PROJECTION",
)
"""Fixed, because the order is a dependency statement and not a preference.

Risk after Alpha is not a scheduling nicety: an advancement that scored a session
Risk cannot cover would publish a formation with no covariance, and the failure
would surface as a missing array deep inside a policy walk rather than as a
refusal naming the lane that ran out.
"""

PortfolioProductMode = Literal["DEVELOPMENT_REPLAY", "SYNTHETIC_QA"]
"""The two modes admitted before Stage 10.

`FORWARD_LOCAL_RESEARCH` is deliberately absent. Daily advancement, `Today`,
current holdings and intended trades belong to it, and it is admitted only after
Stage 10 by a separate activation receipt -- so there is no value here a caller
could pass to reach them.
"""

FORWARD_MODE_REFUSAL = "product_host.not_admitted_for_forward_mode"
"""What an ordinary replay route answers when asked to advance.

Not `NO_ADVANCE_EXACT_ZERO_WORK`. That answer would be true about the artifacts
and false about the product: it reads as "you are up to date", and the frozen
2024-08-12 workspace is not up to date with anything -- it is a historical
interval that was never intended to move.
"""

LaneDisposition = Literal[
    "REUSED_EXACT",
    "ADVANCED",
    "REBUILT_FROM_CORRECTION",
    "INPUT_NOT_MATERIALIZED",
]
AdvancementDisposition = Literal[
    "ADVANCED",
    "REBUILT_FROM_CORRECTION",
    "NO_ADVANCE_EXACT_ZERO_WORK",
    "INPUT_NOT_MATERIALIZED",
    "NOT_ADMITTED_FOR_FORWARD_MODE",
]


class AdvancementError(ValueError):
    """Stable refusal for an advancement identity, axis or ordering failure."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


def ordered_session_axis_hash(sessions: tuple[date, ...]) -> str:
    """Content identity of an exact ordered session axis.

    Refuses an axis that is not strictly increasing, because a duplicate or a
    reordering is a defect in the owner that produced it, and hashing it anyway
    would give that defect a durable identity.
    """
    if any(later <= earlier for earlier, later in pairwise(sessions)):
        raise AdvancementError("portfolio_advancement.session_axis_not_strictly_increasing")
    return str(
        canonical_hash(
            {
                "kind": "OrderedSessionAxis",
                "sessions": [value.isoformat() for value in sessions],
            }
        )
    )


def ordered_listing_axis_hash(listings: tuple[str, ...]) -> str:
    """Content identity of an exact ordered listing axis, order included."""
    if len(set(listings)) != len(listings):
        raise AdvancementError("portfolio_advancement.listing_axis_duplicated")
    return str(canonical_hash({"kind": "OrderedListingAxis", "listings": list(listings)}))


class LaneCoverage(_Contract):
    """One owner's exact axes, and how far its own source would let it go.

    ``reachable_end`` is the honest half. A lane that holds sessions through
    Tuesday but whose source only has Wednesday can advance one session, not to
    whatever the fastest lane holds, and a compiler that read only
    ``last_session`` would ask every owner for sessions half of them cannot make.
    """

    kind: Literal["LaneCoverage"] = "LaneCoverage"
    lane: AdvancementLaneId
    owner_id: str = Field(min_length=1, max_length=120)
    lane_label: str = Field(min_length=1, max_length=120)
    sessions: tuple[date, ...] = ()
    """The exact ordered axis this owner holds.

    Carried, not summarised. The common watermark is the *intersection* of these
    axes, and an intersection cannot be computed from endpoints and a count: a
    lane missing a Wednesday reports the same three numbers as a lane that never
    had that Wednesday, so a compiler reading only the ends would claim support
    neither of them has. A hash can prove two axes differ; only the axes
    themselves say what they still agree on.
    """

    first_session: date | None = None
    last_session: date | None = None
    session_count: int = Field(ge=0)
    session_axis_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    listing_count: int = Field(default=0, ge=0)
    listing_axis_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_bound: Literal["INDEPENDENT_SOURCE", "DOWNSTREAM_OF_PIPELINE"] = "INDEPENDENT_SOURCE"
    """Whether this lane has a source of its own that can limit the target.

    Data, Feature, Alpha, Risk and Tradability each read something outside the
    pipeline, so how far each of those reaches is a real constraint. The
    Portfolio path and the report have no such source -- they can only go as far
    as the lanes above them -- and letting them declare a bound would make the
    intersection describe the *previous* run rather than the reachable one.
    """

    reachable_end: date | None = None
    identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    coverage_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one declared owner-lane coverage.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical coverage_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, coverage_hash="0" * 64).model_dump(
            mode="json", exclude={"coverage_hash"}
        )
        return cls(**identity, coverage_hash=canonical_hash(identity))

    @classmethod
    def of(
        cls,
        *,
        lane: AdvancementLaneId,
        owner_id: str,
        lane_label: str,
        sessions: tuple[date, ...],
        identity_hash: str,
        listings: tuple[str, ...] = (),
        reachable_end: date | None = None,
        source_bound: str = "INDEPENDENT_SOURCE",
    ) -> Self:
        """Build coverage from the owner's exact axes rather than from endpoints."""
        return cls.create(
            lane=lane,
            owner_id=owner_id,
            lane_label=lane_label,
            sessions=sessions,
            first_session=sessions[0] if sessions else None,
            last_session=sessions[-1] if sessions else None,
            session_count=len(sessions),
            session_axis_hash=ordered_session_axis_hash(sessions) if sessions else None,
            listing_count=len(listings),
            listing_axis_hash=ordered_listing_axis_hash(listings) if listings else None,
            source_bound=source_bound,
            reachable_end=None if source_bound == "DOWNSTREAM_OF_PIPELINE" else reachable_end,
            identity_hash=identity_hash,
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact materialized lane axes and a coherent source reach bound.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AdvancementError: Empty/session/listing projections disagree, endpoints/count/hash
                contradict the session axis, a downstream lane declares a reach bound, reach
                precedes stored coverage, or coverage identity differs.
        """
        empty = self.session_count == 0
        if (
            empty != (self.first_session is None)
            or empty != (self.last_session is None)
            or empty != (self.session_axis_hash is None)
        ):
            raise AdvancementError("portfolio_advancement.lane_coverage_axis_invalid")
        # The axis is the authority; the endpoints, the count and the hash are
        # projections of it. Letting them disagree would let a coverage describe
        # one interval and intersect as another.
        if len(self.sessions) != self.session_count:
            raise AdvancementError("portfolio_advancement.lane_coverage_axis_count_mismatch")
        if self.sessions and (
            self.sessions[0] != self.first_session
            or self.sessions[-1] != self.last_session
            or ordered_session_axis_hash(self.sessions) != self.session_axis_hash
        ):
            raise AdvancementError("portfolio_advancement.lane_coverage_axis_mismatch")
        if (self.listing_count == 0) != (self.listing_axis_hash is None):
            raise AdvancementError("portfolio_advancement.lane_listing_axis_invalid")
        if (
            self.first_session is not None
            and self.last_session is not None
            and self.first_session > self.last_session
        ):
            raise AdvancementError("portfolio_advancement.lane_coverage_axis_invalid")
        if self.source_bound == "DOWNSTREAM_OF_PIPELINE" and self.reachable_end is not None:
            raise AdvancementError("portfolio_advancement.lane_downstream_declared_a_bound")
        if (
            self.reachable_end is not None
            and self.last_session is not None
            and self.reachable_end < self.last_session
        ):
            # A lane cannot reach less far than it already holds. When this
            # fires the adapter has confused "what my source has" with "what I
            # have left to do", and every downstream target would be short.
            raise AdvancementError("portfolio_advancement.lane_reachable_end_behind_materialized")
        if self.coverage_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"coverage_hash"})
        ):
            raise AdvancementError("portfolio_advancement.lane_coverage_identity_invalid")
        return self


def ordered_axis_intersection(axes: Sequence[tuple[date, ...]]) -> tuple[date, ...]:
    """The sessions every axis actually holds, in order.

    A real intersection rather than an interval. Two lanes sharing a first
    session, a last session and a count can still disagree about a Wednesday,
    and `max(first)`/`min(last)` calls that agreement -- which is how a watermark
    ends up ahead of one of the artifacts behind it.
    """
    if not axes:
        return ()
    common = set(axes[0])
    for axis in axes[1:]:
        common &= set(axis)
    return tuple(sorted(common))


def _common_axis_summary(
    axes: Sequence[tuple[date, ...]],
) -> tuple[date | None, date | None, int, str | None]:
    """Canonical endpoints, count and identity of one exact intersection."""

    common = ordered_axis_intersection(axes)
    if not common:
        return None, None, 0, None
    return common[0], common[-1], len(common), ordered_session_axis_hash(common)


class LaneWork(_Contract):
    """What an owner actually did, counted by the owner rather than inferred."""

    kind: Literal["LaneWork"] = "LaneWork"
    alpha_fits: int = Field(default=0, ge=0)
    score_sessions: int = Field(default=0, ge=0)
    risk_updates: int = Field(default=0, ge=0)
    policy_formations: int = Field(default=0, ge=0)
    publications: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        """Sum declared Alpha fits, scores, Risk updates, policy formations and publications.

        Returns:
            Total declared work units; no numerical operation is executed.
        """
        return (
            self.alpha_fits
            + self.score_sessions
            + self.risk_updates
            + self.policy_formations
            + self.publications
        )

    def plus(self, other: LaneWork) -> LaneWork:
        """Add work counters component by component without changing either input.

        Args:
            other: Another declared lane-work record.

        Returns:
            New work record with summed counters.
        """
        return LaneWork(
            alpha_fits=self.alpha_fits + other.alpha_fits,
            score_sessions=self.score_sessions + other.score_sessions,
            risk_updates=self.risk_updates + other.risk_updates,
            policy_formations=self.policy_formations + other.policy_formations,
            publications=self.publications + other.publications,
        )


class CorrectionLineage(_Contract):
    """Which upstream revision forced a rebuild, and what it superseded.

    Bound separately from the target sessions because a correction and an
    extension are different events that can occur alone or together. A receipt
    that expressed both as "sessions materialized" could not tell a rebuild of
    Tuesday from the arrival of Wednesday.
    """

    kind: Literal["CorrectionLineage"] = "CorrectionLineage"
    corrected_owner_id: str = Field(min_length=1, max_length=120)
    source_revision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    superseded_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rebuilt_sessions: tuple[date, ...] = ()
    lineage_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one explicit content supersession lineage.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical lineage_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, lineage_hash="0" * 64).model_dump(
            mode="json", exclude={"lineage_hash"}
        )
        return cls(**identity, lineage_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require an actual supersession and exact correction lineage identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AdvancementError: Source and superseded content identities are equal or lineage_hash
                differs.
        """
        if self.source_revision_hash == self.superseded_identity_hash:
            raise AdvancementError("portfolio_advancement.correction_did_not_supersede")
        if self.lineage_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"lineage_hash"})
        ):
            raise AdvancementError("portfolio_advancement.correction_identity_invalid")
        return self


class DomainLaneReceipt(_Contract):
    """One owner's own answer: what it was asked, what it made, what it cost.

    Produced by the domain owner, not by the Host. Everything the Host later
    reports about work comes from these fields, so an owner that reused two of
    three requested sessions says so and the receipt cannot claim three.
    """

    kind: Literal["DomainLaneReceipt"] = "DomainLaneReceipt"
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Which sealed advancement this answer belongs to.

    Carried on the receipt rather than only in an index key, so a lane result
    cannot be filed under a program it was never asked by -- which is the one way
    a resumed run could seal a receipt from another run's work.
    """

    lane: AdvancementLaneId
    owner_id: str = Field(min_length=1, max_length=120)
    disposition: LaneDisposition
    requested_sessions: tuple[date, ...] = ()
    materialized_sessions: tuple[date, ...] = ()
    produced_identities: tuple[str, ...] = ()
    reused_identities: tuple[str, ...] = ()
    work: LaneWork = LaneWork()
    coverage_before: LaneCoverage
    coverage_after: LaneCoverage
    correction_lineage: CorrectionLineage | None = None
    refusal_code: str | None = None
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one declared owner-lane advancement receipt.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical receipt_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return seal_model_from_dump(cls, values, field="receipt_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require lane authority, exact coverage movement and disposition/work evidence.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AdvancementError: Lane/owner/request/materialized axes disagree; advance/reuse/rebuild
                disposition contradicts produced content, work, correction lineage or coverage; or
                receipt identity differs.
        """
        if self.coverage_before.lane != self.lane or self.coverage_after.lane != self.lane:
            raise AdvancementError("portfolio_advancement.lane_receipt_lane_mismatch")
        if (self.disposition == "INPUT_NOT_MATERIALIZED") != (self.refusal_code is not None):
            raise AdvancementError("portfolio_advancement.lane_receipt_refusal_invalid")
        for axis in (self.requested_sessions, self.materialized_sessions):
            if any(later <= earlier for earlier, later in pairwise(axis)):
                raise AdvancementError("portfolio_advancement.lane_receipt_session_axis_invalid")
        if not set(self.materialized_sessions).issubset(self.requested_sessions):
            # An owner may materialize fewer sessions than it was asked for and
            # say so. It may not materialize a session nobody requested.
            raise AdvancementError("portfolio_advancement.lane_materialized_outside_request")
        if (self.disposition == "ADVANCED") != bool(self.materialized_sessions):
            raise AdvancementError("portfolio_advancement.lane_receipt_disposition_invalid")
        if self.disposition == "ADVANCED" and not self.produced_identities:
            raise AdvancementError("portfolio_advancement.lane_advanced_without_an_artifact")
        if self.disposition == "ADVANCED" and set(self.coverage_after.sessions) != set(
            self.coverage_before.sessions
        ) | set(self.materialized_sessions):
            # The axis after must be the axis before plus exactly what was
            # materialized. Anything else means the owner dropped a session it
            # held, or gained one it never reported making.
            raise AdvancementError("portfolio_advancement.lane_coverage_after_unaccounted")
        if self.disposition == "REUSED_EXACT" and (
            self.coverage_after.coverage_hash != self.coverage_before.coverage_hash
            or self.work.total != 0
            or self.produced_identities
        ):
            raise AdvancementError("portfolio_advancement.lane_reuse_is_not_exact")
        if self.disposition == "REBUILT_FROM_CORRECTION":
            if self.correction_lineage is None or not self.produced_identities:
                raise AdvancementError("portfolio_advancement.lane_rebuild_without_lineage")
            if self.coverage_after.session_axis_hash != self.coverage_before.session_axis_hash:
                # A rebuild replaces content over the same axis. Extending the
                # axis at the same time is an advancement, and conflating them
                # loses which of the two the work belonged to.
                raise AdvancementError("portfolio_advancement.lane_rebuild_moved_the_axis")
            if self.work.total == 0:
                raise AdvancementError("portfolio_advancement.lane_rebuild_reported_no_work")
        if self.correction_lineage is not None and self.disposition not in {
            "REBUILT_FROM_CORRECTION",
            "ADVANCED",
        }:
            raise AdvancementError("portfolio_advancement.lane_lineage_without_a_rebuild")
        if self.receipt_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        ):
            raise AdvancementError("portfolio_advancement.lane_receipt_identity_invalid")
        return self


class OperationalReceipt(_Contract):
    """The profile a run executed under. Deliberately outside scientific identity.

    Threads, workers and memory move wall-clock and nothing else, so binding them
    into a Program hash would make an identical experiment on a different machine
    look like a different experiment. They still have to be *recoverable*: a
    numerical output whose capacity plan cannot be read back leaves a runtime or
    capacity defect with nowhere to be diagnosed. Hence a separate receipt that
    the output names, rather than a field the Program hashes.
    """

    kind: Literal["OperationalReceipt"] = "OperationalReceipt"
    capacity_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_resolution: str = Field(min_length=1, max_length=40)
    process_logical_processor_limit: int = Field(ge=1)
    fold_workers: int = Field(ge=0)
    wall_seconds: float = Field(ge=0.0)
    peak_rss_bytes: int = Field(ge=0)
    process_tree_peak_rss_bytes: int = Field(ge=0)
    process_tree_peak_live_descendant_count: int = Field(ge=0)
    measurement_scope: Literal["PARENT_PROCESS_ONLY", "PARENT_AND_LIVE_DESCENDANTS"]
    instrumentation_limitation: str | None = None
    scientific_identity_disposition: Literal["OPERATIONAL_ONLY_NEVER_SCIENTIFIC_IDENTITY"] = (
        "OPERATIONAL_ONLY_NEVER_SCIENTIFIC_IDENTITY"
    )
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one resource-measurement receipt.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical receipt_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return seal_model_from_dump(cls, values, field="receipt_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require resource measurements consistent with their declared process scope.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AdvancementError: Parent-only measurement carries process-tree counters or receipt
                identity differs.
        """
        if self.measurement_scope == "PARENT_PROCESS_ONLY" and (
            self.process_tree_peak_rss_bytes or self.process_tree_peak_live_descendant_count
        ):
            raise AdvancementError("portfolio_advancement.operational_scope_inconsistent")
        if self.receipt_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        ):
            raise AdvancementError("portfolio_advancement.operational_receipt_identity_invalid")
        return self


class WatermarkAdvancementProgram(_Contract):
    """The compiled advancement: which lanes, which sessions, from which support.

    Compiled before anything runs and hashed without any operational field, so
    the same advancement asked for twice under different thread counts is the
    same program. The source revision is bound separately from the targets, which
    is what lets a corrected input rotate the program while the watermark stands
    still.
    """

    kind: Literal["WatermarkAdvancementProgram"] = "WatermarkAdvancementProgram"
    workspace_id: str = Field(min_length=1, max_length=120)
    product_mode: PortfolioProductMode
    lane_order: tuple[AdvancementLaneId, ...] = Field(min_length=1)
    previous_watermark_start: date | None = None
    previous_watermark_end: date | None = None
    previous_common_session_count: int = Field(default=0, ge=0)
    previous_common_axis_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The exact intersection this advancement started from.

    Bound separately from the endpoints because two workspaces can share a first
    and a last common session while disagreeing inside, and a program that named
    only the ends would compile to the same identity for both.
    """

    target_sessions: tuple[date, ...] = ()
    source_revision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """One digest over every lane's own artifact identity.

    Separate from the targets on purpose: a corrected Risk surface changes this
    and nothing else, so the program rotates, the stale receipt cannot be
    reopened, and the run that follows is a rebuild rather than a no-op.
    """

    alpha_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    risk_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    coverage: tuple[LaneCoverage, ...] = Field(min_length=1)
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one ordered multi-owner advancement program.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical program_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, program_hash="0" * 64).model_dump(
            mode="json", exclude={"program_hash"}
        )
        return cls(**identity, program_hash=canonical_hash(identity))

    @staticmethod
    def revision_of(coverage: tuple[LaneCoverage, ...]) -> str:
        """The source revision every lane's own identity contributes to."""
        return str(
            canonical_hash(
                {
                    "kind": "AdvancementSourceRevision",
                    "identities": [
                        {"owner_id": value.owner_id, "identity_hash": value.identity_hash}
                        for value in coverage
                    ],
                }
            )
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require ordered owner coverage and a strictly forward target/common axis.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AdvancementError: Lane/coverage order repeats or disagrees, source revision differs,
                target dates are unordered/not beyond the watermark, common intersection projections
                disagree, or program identity differs.
        """
        if tuple(dict.fromkeys(self.lane_order)) != self.lane_order:
            raise AdvancementError("portfolio_advancement.program_lane_order_invalid")
        if tuple(value.lane for value in self.coverage) != self.lane_order:
            raise AdvancementError("portfolio_advancement.program_coverage_order_invalid")
        if self.source_revision_hash != self.revision_of(self.coverage):
            raise AdvancementError("portfolio_advancement.program_source_revision_invalid")
        targets = self.target_sessions
        if any(later <= earlier for earlier, later in pairwise(targets)):
            raise AdvancementError("portfolio_advancement.program_target_axis_invalid")
        if (
            targets
            and self.previous_watermark_end is not None
            and targets[0] <= self.previous_watermark_end
        ):
            # Advancement extends; it never revisits. A target at or before the
            # previous watermark would rematerialize an admitted session.
            raise AdvancementError("portfolio_advancement.program_target_not_beyond_watermark")
        if (self.previous_watermark_start is None) != (self.previous_watermark_end is None):
            raise AdvancementError("portfolio_advancement.program_watermark_axis_invalid")
        expected_common = _common_axis_summary([value.sessions for value in self.coverage])
        stated_common = (
            self.previous_watermark_start,
            self.previous_watermark_end,
            self.previous_common_session_count,
            self.previous_common_axis_hash,
        )
        if stated_common != expected_common:
            raise AdvancementError("portfolio_advancement.program_common_axis_invalid")
        if self.program_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"program_hash"})
        ):
            raise AdvancementError("portfolio_advancement.program_identity_invalid")
        return self

    def require_lane_receipt(self, *, lane: AdvancementLaneId, receipt: DomainLaneReceipt) -> None:
        """Prove one durable answer belongs to this program's sealed question."""
        declared = next((value for value in self.coverage if value.lane == lane), None)
        if declared is None or receipt.lane != lane or receipt.program_hash != self.program_hash:
            raise AdvancementError("product_host.advancement_lane_receipt_mismatch:" + lane)
        if (
            receipt.owner_id != declared.owner_id
            or receipt.coverage_before.owner_id != declared.owner_id
            or receipt.coverage_after.owner_id != declared.owner_id
        ):
            raise AdvancementError(
                "product_host.advancement_lane_owner_mismatch:" + lane + ":" + receipt.owner_id
            )
        if receipt.requested_sessions != self.target_sessions:
            raise AdvancementError("product_host.advancement_lane_answered_another_request:" + lane)
        if receipt.coverage_before.coverage_hash != declared.coverage_hash:
            raise AdvancementError("product_host.advancement_lane_coverage_before_moved:" + lane)
        if receipt.disposition == "ADVANCED" and (
            receipt.coverage_after.last_session is None
            or (
                receipt.coverage_before.last_session is not None
                and receipt.coverage_after.last_session <= receipt.coverage_before.last_session
            )
        ):
            raise AdvancementError(
                "product_host.advancement_lane_did_not_extend:" + receipt.owner_id
            )

    def require_receipt(self, receipt: WatermarkAdvancementReceipt) -> None:
        """Bind a sealed advancement readback to its program and every lane answer."""
        if (
            receipt.program_hash != self.program_hash
            or receipt.workspace_id != self.workspace_id
            or receipt.product_mode != self.product_mode
            or receipt.previous_watermark_start != self.previous_watermark_start
            or receipt.previous_watermark_end != self.previous_watermark_end
            or receipt.previous_common_session_count != self.previous_common_session_count
            or receipt.previous_common_axis_hash != self.previous_common_axis_hash
        ):
            raise AdvancementError("portfolio_advancement.receipt_program_binding_invalid")
        for value in receipt.receipts:
            self.require_lane_receipt(lane=value.lane, receipt=value)


class WatermarkAdvancementReceipt(_Contract):
    """What the advancement did, per lane, and where the watermark ended up."""

    kind: Literal["WatermarkAdvancementReceipt"] = "WatermarkAdvancementReceipt"
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    workspace_id: str = Field(min_length=1, max_length=120)
    product_mode: PortfolioProductMode
    disposition: AdvancementDisposition
    receipts: tuple[DomainLaneReceipt, ...] = Field(min_length=1)
    previous_watermark_start: date | None = None
    previous_watermark_end: date | None = None
    previous_common_session_count: int = Field(default=0, ge=0)
    previous_common_axis_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The exact intersection the advancement started from.

    Carried beside the endpoints because a quiet run and a rebuild both leave the
    ends where they were: only the axis identity distinguishes "nothing moved"
    from "the same interval, different content".
    """

    new_watermark_start: date | None = None
    new_watermark_end: date | None = None
    new_common_session_count: int = Field(default=0, ge=0)
    new_common_axis_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The exact intersection the advancement ended at, for the same reason."""

    blocking_lane: AdvancementLaneId | None = None
    blocking_owner_id: str | None = None
    blocking_available_end: date | None = None
    refusal_code: str | None = None
    total_work: LaneWork = LaneWork()
    operational_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Bound, never hashed into a scientific identity. See ``OperationalReceipt``."""

    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one multi-owner advancement result.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical receipt_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return seal_model_from_dump(cls, values, field="receipt_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact owner receipts, common-watermark movement and summed work.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AdvancementError: Refusal owner/code/lane prefix is incomplete, receipt owner/program
                binding differs, common coverage or summed work disagrees,
                advance/rebuild/reuse/refusal disposition contradicts movement/work, or receipt
                identity differs.
        """
        refused = self.disposition in {"INPUT_NOT_MATERIALIZED", "NOT_ADMITTED_FOR_FORWARD_MODE"}
        if refused != (self.refusal_code is not None):
            raise AdvancementError("portfolio_advancement.receipt_refusal_invalid")
        if self.disposition == "INPUT_NOT_MATERIALIZED" and (
            self.blocking_lane is None or self.blocking_owner_id is None
        ):
            # The whole point of the disposition is that it names an owner. A
            # refusal without one is the opaque hash puzzle the plan forbids.
            raise AdvancementError("portfolio_advancement.receipt_refusal_owner_absent")
        lanes = tuple(value.lane for value in self.receipts)
        full_lane_set = lanes == ADVANCEMENT_LANE_ORDER
        if self.disposition == "INPUT_NOT_MATERIALIZED":
            prefix = ADVANCEMENT_LANE_ORDER[: len(lanes)]
            lawful_partial = lanes == prefix and bool(lanes) and self.blocking_lane == lanes[-1]
            if not full_lane_set and not lawful_partial:
                raise AdvancementError("portfolio_advancement.receipt_lane_set_invalid")
        elif not full_lane_set:
            raise AdvancementError("portfolio_advancement.receipt_lane_set_invalid")
        if any(
            value.program_hash != self.program_hash
            or value.owner_id != value.coverage_before.owner_id
            or value.owner_id != value.coverage_after.owner_id
            for value in self.receipts
        ):
            raise AdvancementError("portfolio_advancement.receipt_lane_binding_invalid")
        for count, axis_hash in (
            (self.previous_common_session_count, self.previous_common_axis_hash),
            (self.new_common_session_count, self.new_common_axis_hash),
        ):
            if (count == 0) != (axis_hash is None):
                raise AdvancementError("portfolio_advancement.receipt_common_axis_invalid")
        if full_lane_set:
            expected_before = _common_axis_summary(
                [value.coverage_before.sessions for value in self.receipts]
            )
            expected_after = _common_axis_summary(
                [value.coverage_after.sessions for value in self.receipts]
            )
            stated_before = (
                self.previous_watermark_start,
                self.previous_watermark_end,
                self.previous_common_session_count,
                self.previous_common_axis_hash,
            )
            stated_after = (
                self.new_watermark_start,
                self.new_watermark_end,
                self.new_common_session_count,
                self.new_common_axis_hash,
            )
            if stated_before != expected_before or stated_after != expected_after:
                raise AdvancementError("portfolio_advancement.receipt_common_axis_invalid")
        elif (
            self.new_watermark_start,
            self.new_watermark_end,
            self.new_common_session_count,
            self.new_common_axis_hash,
        ) != (
            self.previous_watermark_start,
            self.previous_watermark_end,
            self.previous_common_session_count,
            self.previous_common_axis_hash,
        ):
            # A blocked prefix cannot establish a new seven-lane intersection.
            raise AdvancementError("portfolio_advancement.receipt_refusal_moved_watermark")
        expected = LaneWork()
        for receipt in self.receipts:
            expected = expected.plus(receipt.work)
        if expected != self.total_work:
            raise AdvancementError("portfolio_advancement.receipt_work_total_invalid")
        if self.disposition == "NO_ADVANCE_EXACT_ZERO_WORK":
            if self.total_work.total != 0 or any(
                value.disposition != "REUSED_EXACT" for value in self.receipts
            ):
                raise AdvancementError("portfolio_advancement.receipt_zero_work_contradicted")
            if (
                self.new_watermark_start,
                self.new_watermark_end,
                self.new_common_session_count,
                self.new_common_axis_hash,
            ) != (
                self.previous_watermark_start,
                self.previous_watermark_end,
                self.previous_common_session_count,
                self.previous_common_axis_hash,
            ):
                raise AdvancementError("portfolio_advancement.receipt_zero_work_moved_watermark")
        if self.disposition == "ADVANCED" and (
            self.new_watermark_end is None
            or self.new_common_session_count <= self.previous_common_session_count
            or (
                self.previous_watermark_end is not None
                and self.new_watermark_end <= self.previous_watermark_end
            )
        ):
            # An advancement has to add sessions to the *intersection*, not only
            # push the last one out. A lane extending past a gap the others still
            # have would move the end without widening the support.
            raise AdvancementError("portfolio_advancement.receipt_advance_did_not_move_watermark")
        if self.disposition == "REBUILT_FROM_CORRECTION":
            if self.total_work.total == 0:
                # The defect this disposition exists to prevent: an empty target
                # list read as "nothing to do" while a corrected artifact sat
                # upstream of a stale report.
                raise AdvancementError("portfolio_advancement.receipt_rebuild_reported_no_work")
            if (
                self.new_watermark_start,
                self.new_watermark_end,
                self.new_common_session_count,
                self.new_common_axis_hash,
            ) != (
                self.previous_watermark_start,
                self.previous_watermark_end,
                self.previous_common_session_count,
                self.previous_common_axis_hash,
            ):
                raise AdvancementError("portfolio_advancement.receipt_rebuild_moved_watermark")
        if refused and (
            self.new_watermark_start,
            self.new_watermark_end,
            self.new_common_session_count,
            self.new_common_axis_hash,
        ) != (
            self.previous_watermark_start,
            self.previous_watermark_end,
            self.previous_common_session_count,
            self.previous_common_axis_hash,
        ):
            raise AdvancementError("portfolio_advancement.receipt_refusal_moved_watermark")
        if self.receipt_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        ):
            raise AdvancementError("portfolio_advancement.receipt_identity_invalid")
        return self

    def lane(self, lane: AdvancementLaneId) -> DomainLaneReceipt:
        """Resolve one declared owner-lane receipt from the retained result.

        Args:
            lane: Exact advancement lane identity.

        Returns:
            Matching domain-lane receipt.

        Raises:
            AdvancementError: The requested lane has no retained receipt.
        """
        for value in self.receipts:
            if value.lane == lane:
                return value
        raise AdvancementError("portfolio_advancement.receipt_lane_absent:" + lane)


class PortfolioLedgerCoverage(_Contract):
    """The current lifecycle's reporting geometry: the ledger's own exact axes.

    This replaces the fixed-1,260 Campaign geometry as the *authority* a current
    run reports against. The old codec pinned a formation count, which made a
    continuous path describable only at one length -- so a workspace holding 253
    formations, or 1,300, had no lawful way to report at all. The successor binds
    the exact ordered formation and listing axes instead, so the count is a fact
    about the path rather than a condition on it.

    The old geometry is not deleted: frozen Campaign artifacts still read back
    through it exactly. It simply stops being what a live path is measured with.
    """

    kind: Literal["PortfolioLedgerCoverage"] = "PortfolioLedgerCoverage"
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    first_formation_session: date
    last_formation_session: date
    formation_count: int = Field(gt=0)
    formation_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    listing_count: int = Field(gt=0)
    listing_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_coverage_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The `CommonWatermark`/owner-coverage identity this path was cut against.

    Without it the geometry describes a ledger but not the support that admitted
    it, and a path built over corrected inputs would be indistinguishable from
    one built over the inputs it claims.
    """

    geometry_disposition: Literal["CONTINUOUS_ADMITTED_LEDGER_NO_SELECTION"] = (
        "CONTINUOUS_ADMITTED_LEDGER_NO_SELECTION"
    )
    superseded_codec: Literal["FIXED_1260_FULL_PATH_REPORTING_HISTORICAL_READBACK_ONLY"] = (
        "FIXED_1260_FULL_PATH_REPORTING_HISTORICAL_READBACK_ONLY"
    )
    coverage_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one declared materialized ledger coverage.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical coverage_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, coverage_hash="0" * 64).model_dump(
            mode="json", exclude={"coverage_hash"}
        )
        return cls(**identity, coverage_hash=canonical_hash(identity))

    @classmethod
    def of(
        cls,
        *,
        program_hash: str,
        ledger_hash: str,
        formation_sessions: tuple[date, ...],
        ordered_listing_ids: tuple[str, ...],
        source_coverage_hash: str,
    ) -> Self:
        """Seal materialized ledger coverage from explicit nonempty formation/listing axes.

        Args:
            program_hash: Exact execution program.
            ledger_hash: Exact retained execution ledger.
            formation_sessions: Nonempty formation axis in declared order.
            ordered_listing_ids: Exact ordered listing axis.
            source_coverage_hash: Source-owner coverage used to materialize the ledger.

        Returns:
            Validated coverage with axis endpoints/counts/hashes.

        Raises:
            IndexError: No formation session is supplied.
            AdvancementError: Declared coverage consistency fails.
        """
        return cls.create(
            program_hash=program_hash,
            ledger_hash=ledger_hash,
            first_formation_session=formation_sessions[0],
            last_formation_session=formation_sessions[-1],
            formation_count=len(formation_sessions),
            formation_axis_hash=ordered_session_axis_hash(formation_sessions),
            listing_count=len(ordered_listing_ids),
            listing_axis_hash=ordered_listing_axis_hash(ordered_listing_ids),
            source_coverage_hash=source_coverage_hash,
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require an ordered declared ledger interval and exact coverage identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AdvancementError: Formation endpoints are reversed or coverage_hash differs.
        """
        if self.first_formation_session > self.last_formation_session:
            raise AdvancementError("portfolio_advancement.ledger_coverage_axis_invalid")
        if self.coverage_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"coverage_hash"})
        ):
            raise AdvancementError("portfolio_advancement.ledger_coverage_identity_invalid")
        return self

    def contains(self, *, start: date, end: date) -> bool:
        """Whether a requested study window is inside this materialized path."""
        return self.first_formation_session <= start and end <= self.last_formation_session

    def describes(self, formation_sessions: tuple[date, ...]) -> bool:
        """Whether an exact axis is the one this geometry was built from.

        Exact rather than endpoint-wise, because a path missing an interior
        session has the same first, last and count as the path that has it.
        """
        return (
            len(formation_sessions) == self.formation_count
            and ordered_session_axis_hash(formation_sessions) == self.formation_axis_hash
        )


__all__ = [
    "ADVANCEMENT_LANE_ORDER",
    "FORWARD_MODE_REFUSAL",
    "AdvancementDisposition",
    "AdvancementError",
    "AdvancementLaneId",
    "CorrectionLineage",
    "DomainLaneReceipt",
    "LaneCoverage",
    "LaneDisposition",
    "LaneWork",
    "OperationalReceipt",
    "PortfolioLedgerCoverage",
    "PortfolioProductMode",
    "WatermarkAdvancementProgram",
    "WatermarkAdvancementReceipt",
    "ordered_axis_intersection",
    "ordered_listing_axis_hash",
    "ordered_session_axis_hash",
]
