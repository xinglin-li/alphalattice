"""Authority-scoped readers for causal execution outcomes."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Protocol, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    RawDailyBar,
)
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.data.calendar import materialize_calendar_schedule
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .artifacts import (
    DEVELOPMENT_ONLY_MANIFEST_CATEGORY,
    DEVELOPMENT_ONLY_MARKER_CATEGORY,
    METHOD_BINDING_CATEGORY,
    _ExecutionOutcomeArtifactStore,
)
from .compile import _SCHEMA_ID, _action_dividends, _schema, derive_causal_execution_row
from .contracts import (
    CausalExecutionOutcomeChunk,
    CausalExecutionOutcomeManifest,
    CausalExecutionOutcomeMarker,
    CausalExecutionSchedulePoint,
    DevelopmentOnlyExecutionOutcomeManifest,
    DevelopmentOnlyExecutionOutcomeMarker,
    LocalQAMarketSnapshot,
    PolicyHoldoutExecutionOutcomeAuthority,
    PolicyHoldoutExecutionOutcomeRelease,
)
from .methods import (
    ExecutionOutcomeMethodBinding,
    ExecutionOutcomeMethodError,
    ExecutionOutcomeMethodSeal,
    ExecutionOutcomeSessionAxis,
    build_installed_execution_outcome_method_catalog,
    build_installed_execution_outcome_publication_policy_catalog,
    build_one_session_recipe,
    period_dividend_for_point,
    resolve_final_schedule_point,
    resolve_schedule_points,
    verify_execution_outcome_method_seal_marker,
)


def planned_local_qa_schedule(
    start: date, through: date
) -> tuple[CausalExecutionSchedulePoint, ...]:
    """Resolve intended sessions from the installed calendar, not from future bars.

    The calendar API filters completed sessions by its horizon. Here that horizon
    is explicitly a schedule lookup, never evidence of data availability.
    """
    end = through + timedelta(days=14)
    table = materialize_calendar_schedule(
        ("XNYS", "XNAS"),
        start=start,
        end=end,
        as_of_timestamp=datetime.combine(end, time(23, 59), UTC),
    )
    clocks: dict[date, dict[str, object]] = {}
    venues: dict[date, set[str]] = {}
    for row in table.to_pylist():
        day = row["session_date"]
        if day in clocks and any(
            clocks[day][k] != row[k] for k in ("session_open_timestamp", "session_close_timestamp")
        ):
            raise ValueError("causal_outcomes.qa_calendar_disagrees")
        clocks[day] = row
        venues.setdefault(day, set()).add(row["calendar_id"])
    axis = tuple(d for d in sorted(clocks) if venues[d] == {"XNYS", "XNAS"})
    return resolve_schedule_points(
        recipe=build_one_session_recipe(), ordered_sessions=axis, session_clocks=clocks
    )


def _qa_source_manifest(
    store: MarketDataRepository, listing_ids: tuple[str, ...], retained_listing_ids: tuple[str, ...]
) -> UniverseManifest:
    """Validate the declared candidate/retained coverage at the Data boundary."""
    manifest = store.current_quality_filtered_research_manifest(
        market_profile_id="us-current-index-research"
    )
    candidates = () if manifest is None else tuple(sorted(v.listing_id for v in manifest.listings))
    if (
        manifest is None
        or tuple(sorted(set(candidates) | set(retained_listing_ids))) != listing_ids
        or not set(retained_listing_ids) <= set(listing_ids)
    ):
        raise ValueError("causal_outcomes.qa_listing_epoch_mismatch")
    if retained_listing_ids:
        store.listing_scope(manifest, listing_ids=listing_ids)
    return manifest


def read_local_qa_market_snapshot(
    *,
    store: MarketDataRepository,
    start: date,
    through: date,
    listing_ids: tuple[str, ...],
    observed_at: datetime,
    retained_listing_ids: tuple[str, ...] = (),
) -> LocalQAMarketSnapshot:
    """Read only admitted daily bars, including an entry whose exit is still absent."""
    manifest = _qa_source_manifest(store, listing_ids, retained_listing_ids)
    schedule = planned_local_qa_schedule(start, through)
    point = next((v for v in schedule if v.formation_session == through), None)
    if (
        point is None
        or observed_at.tzinfo is None
        or observed_at.utcoffset() is None
        or point.formation_close_at > observed_at
    ):
        raise ValueError("causal_outcomes.qa_session_not_completed")
    covered = store.manifest_raw_range(manifest)
    if covered is None or through > covered[1] or start < covered[0]:
        raise ValueError("causal_outcomes.qa_source_support_absent")
    bars: list[RawDailyBar] = []
    actions: list[CorporateActionEvent] = []
    with store._connect(read_only=True) as connection:
        for listing in listing_ids:
            bars.extend(
                store.raw_bars(listing, start=start, through=through, _connection=connection)
            )
            actions.extend(
                v
                for v in store.actions(listing, _connection=connection)
                if v.effective_date <= through
            )
    return LocalQAMarketSnapshot.create(
        source_hash=manifest.revision_sha256,
        through=through,
        ordered_listing_ids=listing_ids,
        schedule=schedule,
        bars=tuple(sorted(bars, key=lambda v: (v.session_date, v.listing_id))),
        actions=tuple(actions),
    )


def local_qa_point_rows(
    snapshot: LocalQAMarketSnapshot, point: CausalExecutionSchedulePoint
) -> pa.Table:
    """Evaluate the installed entry/return rule from sealed, bounded observations."""
    bars = {(v.session_date, v.listing_id): v for v in snapshot.bars}
    axis = tuple(
        sorted(
            {
                d
                for v in snapshot.schedule
                for d in (v.formation_session, v.entry_session, v.holding_end_session)
            }
        )
    )
    rows = []
    for listing in snapshot.ordered_listing_ids:
        dividends = _action_dividends(
            tuple(v for v in snapshot.actions if v.listing_id == listing), through=snapshot.through
        )
        rows.append(
            derive_causal_execution_row(
                listing_id=listing,
                symbol=listing,
                point=point,
                entry_bar=bars.get((point.entry_session, listing)),
                holding_bar=bars.get((point.holding_end_session, listing)),
                period_dividend_split_adjusted=period_dividend_for_point(
                    recipe=build_one_session_recipe(),
                    dividends=dividends,
                    ordered_sessions=axis,
                    point=point,
                ),
            )
        )
    return pa.Table.from_pylist(rows, schema=_schema())


def local_qa_prefix(
    snapshot: LocalQAMarketSnapshot, *, through: date, start: date | None = None
) -> LocalQAMarketSnapshot:
    """Bound a captured input without pretending it was acquired at that cutoff."""
    if through > snapshot.through or through not in {
        v.formation_session for v in snapshot.schedule
    }:
        raise ValueError("causal_outcomes.qa_calendar_support_absent")
    return LocalQAMarketSnapshot.create(
        source_hash=snapshot.source_hash,
        through=through,
        ordered_listing_ids=snapshot.ordered_listing_ids,
        schedule=(
            snapshot.schedule
            if start is None
            else tuple(
                CausalExecutionSchedulePoint.model_validate({**v.model_dump(), "sequence": i + 1})
                for i, v in enumerate(p for p in snapshot.schedule if p.formation_session >= start)
            )
        ),
        bars=tuple(
            v
            for v in snapshot.bars
            if v.session_date <= through and (start is None or v.session_date >= start)
        ),
        actions=tuple(v for v in snapshot.actions if v.effective_date <= through),
    )


def local_qa_snapshot_rows(
    snapshot: LocalQAMarketSnapshot, *, sessions: tuple[date, ...], through: date
) -> tuple[pa.Table, dict[tuple[date, str], RawDailyBar], tuple[date, ...]]:
    """Derive completed labels from immutable recovery inputs."""
    prefix = local_qa_prefix(snapshot, through=through)
    axis = tuple(v.formation_session for v in prefix.schedule if v.formation_session <= through)
    if not sessions or sessions != tuple(sorted(set(sessions))) or not set(sessions) <= set(axis):
        raise ValueError("causal_outcomes.qa_session_axis_invalid")
    points = tuple(
        v
        for v in prefix.schedule
        if v.formation_session in sessions and v.holding_end_session <= through
    )
    bars = {(v.session_date, v.listing_id): v for v in prefix.bars}
    dividend_axis = ExecutionOutcomeSessionAxis(axis)
    method = build_one_session_recipe()
    rows = []
    for listing in prefix.ordered_listing_ids:
        dividends = _action_dividends(
            tuple(v for v in prefix.actions if v.listing_id == listing), through=through
        )
        for point in points:
            rows.append(
                derive_causal_execution_row(
                    listing_id=listing,
                    symbol=listing,
                    point=point,
                    entry_bar=bars.get((point.entry_session, listing)),
                    holding_bar=bars.get((point.holding_end_session, listing)),
                    period_dividend_split_adjusted=period_dividend_for_point(
                        recipe=method,
                        dividends=dividends,
                        ordered_sessions=dividend_axis,
                        point=point,
                    ),
                )
            )
    return pa.Table.from_pylist(rows, schema=_schema()), bars, axis


def read_local_qa_execution_rows(
    *,
    store: MarketDataRepository,
    sessions: tuple[date, ...],
    listing_ids: tuple[str, ...],
    through: date,
    observed_at: datetime,
    retained_listing_ids: tuple[str, ...] = (),
) -> tuple[pa.Table, dict[tuple[date, str], RawDailyBar], tuple[date, ...]]:
    """Derive completed QA observations from explicitly admitted local bars.

    This is not a development/holdout reader or publisher. The caller must hold
    workspace QA admission and seals the returned source-exact rows as QA input.
    No sealed artifact is opened, no tail is released, and no future bar is read.
    A missing holding end yields no observation rather than a zero return.
    """
    manifest = _qa_source_manifest(store, listing_ids, retained_listing_ids)
    if not sessions or sessions != tuple(sorted(set(sessions))) or sessions[-1] > through:
        raise ValueError("causal_outcomes.qa_session_axis_invalid")
    covered = store.manifest_raw_range(manifest)
    if covered is None or through > covered[1] or sessions[0] < covered[0]:
        raise ValueError("causal_outcomes.qa_source_support_absent")
    calendar = materialize_calendar_schedule(
        ("XNYS", "XNAS"), start=covered[0], end=through, as_of_timestamp=observed_at
    )
    clocks: dict[date, dict[str, object]] = {}
    venues: dict[date, set[str]] = {}
    for row in calendar.to_pylist():
        day = row["session_date"]
        if day in clocks and any(
            clocks[day][key] != row[key]
            for key in ("session_open_timestamp", "session_close_timestamp")
        ):
            raise ValueError("causal_outcomes.qa_calendar_disagrees")
        clocks[day] = row
        venues.setdefault(day, set()).add(row["calendar_id"])
    axis = tuple(day for day in sorted(clocks) if venues[day] == {"XNAS", "XNYS"})
    if through not in axis or not set(sessions) <= set(axis):
        raise ValueError("causal_outcomes.qa_calendar_support_absent")
    close_at = clocks[through]["session_close_timestamp"]
    if (
        observed_at.tzinfo is None
        or observed_at.utcoffset() is None
        or not isinstance(close_at, datetime)
        or close_at > observed_at
    ):
        raise ValueError("causal_outcomes.qa_session_not_completed")
    method = build_one_session_recipe()
    wanted = set(sessions)
    points = tuple(
        v
        for v in resolve_schedule_points(
            recipe=method, ordered_sessions=axis, session_clocks=clocks
        )
        if v.formation_session in wanted
    )
    start = axis[max(0, axis.index(sessions[0]) - 20)]
    bars: dict[tuple[date, str], RawDailyBar] = {}
    rows: list[dict[str, object]] = []
    dividend_axis = ExecutionOutcomeSessionAxis(axis)
    with store._connect(read_only=True) as connection:
        for listing in listing_ids:
            for bar in store.raw_bars(
                listing, start=start, through=through, _connection=connection
            ):
                bars[bar.session_date, listing] = bar
            dividends = _action_dividends(
                store.actions(listing, _connection=connection), through=through
            )
            for point in points:
                rows.append(
                    derive_causal_execution_row(
                        listing_id=listing,
                        symbol=listing,
                        point=point,
                        entry_bar=bars.get((point.entry_session, listing)),
                        holding_bar=bars.get((point.holding_end_session, listing)),
                        period_dividend_split_adjusted=period_dividend_for_point(
                            recipe=method,
                            dividends=dividends,
                            ordered_sessions=dividend_axis,
                            point=point,
                        ),
                    )
                )
    return pa.Table.from_pylist(rows, schema=_schema()), bars, axis


def _rows_for_sessions(
    store: _ExecutionOutcomeArtifactStore,
    chunks: tuple[CausalExecutionOutcomeChunk, ...],
    sessions: tuple[date, ...],
) -> pa.Table:
    """Read exactly the requested formations out of the year chunks that hold them."""
    years = {value.year for value in sessions}
    selected = tuple(value for value in chunks if value.year in years)
    if not selected:
        return pa.table({})
    table = pa.concat_tables([pq.read_table(store.resolve_chunk(value)) for value in selected])
    mask = pc.is_in(table["formation_session"], value_set=pa.array(sessions, type=pa.date32()))
    return table.filter(mask)


def _matured_sessions(
    store: _ExecutionOutcomeArtifactStore,
    chunks: tuple[CausalExecutionOutcomeChunk, ...],
    *,
    holding_end_through: date,
) -> tuple[date, ...]:
    """Find formations whose outcomes finished by an observation date.

    Compared on the exit clock the rows carry rather than by counting sessions
    back from the end, so the answer does not depend on the caller knowing the
    method's span.
    """
    tables = [
        pq.read_table(
            store.resolve_chunk(chunk), columns=["formation_session", "holding_end_session"]
        )
        for chunk in chunks
    ]
    if not tables:
        return ()
    table = pa.concat_tables(tables)
    eligible = table.filter(
        pc.less_equal(
            table["holding_end_session"], pa.scalar(holding_end_through, type=pa.date32())
        )
    )
    return tuple(sorted(set(eligible["formation_session"].to_pylist())))


class CausalOutcomeDevelopmentRows(Protocol):
    """What a development consumer may ask of causal outcome evidence.

    Satisfied by both the frozen one-session reader and the development-only
    successor. A consumer typed on this asks five questions -- which formations
    have matured, the rows for admitted sessions, the manifest, its ref, and the
    method the snapshot is sealed to -- and never which contract era answered.
    """

    def read_development_sessions(self, manifest_ref: str, sessions: tuple[date, ...]) -> pa.Table:
        """Read admitted development rows for ordered sessions."""
        ...

    def available_development_sessions(
        self, manifest_ref: str, *, holding_end_through: date
    ) -> tuple[date, ...]:
        """Return formations mature by the requested holding-end date."""
        ...

    def load_manifest(
        self, snapshot_hash: str
    ) -> CausalExecutionOutcomeManifest | DevelopmentOnlyExecutionOutcomeManifest:
        """Load the content-addressed development manifest."""
        ...

    def manifest_uri(self, snapshot_hash: str) -> str:
        """Return the artifact URI for a snapshot manifest."""
        ...

    def resolve_method_seal(self, snapshot_hash: str) -> ExecutionOutcomeMethodSeal:
        """Verify the method authority attached to a snapshot."""
        ...


class CausalExecutionOutcomeDevelopmentReader:
    """The only ordinary reader; it has no sealed-holdout release capability."""

    def __init__(self, artifact_root: Path) -> None:
        """Locate immutable outcome artifacts under a workspace root."""
        self.artifacts = _ExecutionOutcomeArtifactStore(artifact_root)

    def read_development_sessions(
        self,
        manifest_ref: str,
        sessions: tuple[date, ...],
    ) -> pa.Table:
        """Read only development chunks and rows needed by one admitted fold."""
        if not sessions or sessions != tuple(sorted(set(sessions))):
            raise ValueError("development session request must be ordered and non-empty")
        manifest = CausalExecutionOutcomeManifest.model_validate(
            self.artifacts.load_json(category="manifests", uri=manifest_ref)
        )
        return _rows_for_sessions(self.artifacts, manifest.development_chunks, sessions)

    def read_development_schedule(self, manifest_ref: str) -> pa.Table:
        """Read the development schedule once without decoding outcome values.

        Schedule columns repeat for every listing in an execution-outcome
        chunk.  A temporal admission needs one authoritative point per
        formation, not the full listing-by-session return surface.  Grouping in
        Arrow keeps that contraction inside the execution owner and validates
        that every listing published the same point before returning it.
        """
        manifest = CausalExecutionOutcomeManifest.model_validate(
            self.artifacts.load_json(category="manifests", uri=manifest_ref)
        )
        schedule_columns = (
            "sequence",
            "formation_session",
            "formation_close_at",
            "entry_session",
            "entry_open_at",
            "holding_end_session",
            "holding_end_open_at",
            "actual_session_span",
        )
        tables = tuple(
            pq.read_table(
                self.artifacts.resolve_chunk(chunk),
                columns=["listing_id", *schedule_columns],
            )
            for chunk in manifest.development_chunks
        )
        if not tables:
            raise ValueError("causal execution development schedule is absent")
        grouped = (
            pa.concat_tables(tables)
            .group_by(schedule_columns)
            .aggregate([("listing_id", "count")])
            .sort_by([("sequence", "ascending")])
        )
        rows = grouped.to_pylist()
        expected_listing_count = len(manifest.listing_ids)
        if len(rows) != manifest.development_formation_count or any(
            row["listing_id_count"] != expected_listing_count for row in rows
        ):
            raise ValueError("causal execution development schedule is inconsistent")
        points = tuple(
            CausalExecutionSchedulePoint.model_validate(
                {name: row[name] for name in schedule_columns}
            )
            for row in rows
        )
        if tuple(value.sequence for value in points) != tuple(
            range(points[0].sequence, points[0].sequence + len(points))
        ) or tuple(value.formation_session for value in points) != tuple(
            sorted({value.formation_session for value in points})
        ):
            raise ValueError("causal execution development schedule is inconsistent")
        return pa.Table.from_pylist([value.model_dump(mode="python") for value in points])

    def available_development_sessions(
        self,
        manifest_ref: str,
        *,
        holding_end_through: date,
    ) -> tuple[date, ...]:
        """Read only the causal session axis needed to derive a refit window."""
        manifest = CausalExecutionOutcomeManifest.model_validate(
            self.artifacts.load_json(category="manifests", uri=manifest_ref)
        )
        return _matured_sessions(
            self.artifacts, manifest.development_chunks, holding_end_through=holding_end_through
        )

    def read_sealed_holdout(self, _manifest_ref: str) -> pa.Table:
        """Refuse generic sealed-holdout access from a development reader."""
        raise PermissionError("SEALED_HOLDOUT_RELEASE_AUTHORITY_UNAVAILABLE")

    def manifest_uri(self, snapshot_hash: str) -> str:
        """Return the owner's published URI for one manifest.

        Exposed because a Host handing a manifest ref to a Desk would otherwise
        assemble the string itself, and a second spelling of a URI is a second
        definition of where artifacts live.

        Named ``manifest_uri`` rather than ``manifest_ref`` because the holdout
        reader below subclasses this one and binds ``manifest_ref`` as an
        instance attribute: the shorter name would shadow that attribute with a
        method on every holdout instance.
        """
        return str(self.artifacts._uri("manifests", snapshot_hash))

    def load_manifest(self, snapshot_hash: str) -> CausalExecutionOutcomeManifest:
        """Read only the content-addressed machine authority, without decoding rows."""
        return cast(
            CausalExecutionOutcomeManifest,
            CausalExecutionOutcomeManifest.model_validate(
                self.artifacts.load_json(
                    category="manifests",
                    uri=self.manifest_uri(snapshot_hash),
                )
            ),
        )

    def resolve_method_seal(self, snapshot_hash: str) -> ExecutionOutcomeMethodSeal:
        """Report which method one published snapshot is sealed to, if any.

        Authority is resolved through the terminal seal marker, never by
        scanning for a binding that mentions the snapshot: a file on disk is not
        a claim, and picking one out of several would let directory order decide
        what a snapshot means. The whole graph the marker names is then handed
        to the one shared Host verifier, with every expected ref derived by the
        Host from content identity -- the store's own spelling of where a
        manifest, marker or binding lives -- so a marker storing a
        plausible-but-wrong ref or citing a catalog or policy the Host never
        installed fails the whole-model comparison rather than being followed.

        A reader resolves a seal; it never mints one. A snapshot published
        before the seam therefore stays ``LEGACY_READBACK_ONLY`` permanently,
        and reading it again does not upgrade it.
        """
        manifest = self.load_manifest(snapshot_hash)
        markers = tuple(
            value
            for value in self.artifacts.method_seal_markers()
            if value.snapshot_hash == snapshot_hash
        )
        if not markers:
            return ExecutionOutcomeMethodSeal(disposition="LEGACY_READBACK_ONLY")
        if len(markers) != 1:
            raise ValueError("causal execution snapshot does not resolve one method seal")
        seal_marker = markers[0]
        outcome_marker = self.marker_for_snapshot(snapshot_hash)
        binding = ExecutionOutcomeMethodBinding.model_validate(
            self.artifacts.load_json(category=METHOD_BINDING_CATEGORY, uri=seal_marker.binding_ref)
        )
        catalog = build_installed_execution_outcome_method_catalog()
        verify_execution_outcome_method_seal_marker(
            seal_marker=seal_marker,
            outcome_marker=outcome_marker,
            outcome_marker_ref=self.artifacts._uri("markers", outcome_marker.marker_hash),
            manifest=manifest,
            manifest_ref=self.manifest_uri(snapshot_hash),
            binding=binding,
            binding_ref=self.artifacts._uri(METHOD_BINDING_CATEGORY, binding.binding_hash),
            catalog=catalog,
            expected_catalog_hash=catalog.binding.catalog_hash,
            policies=build_installed_execution_outcome_publication_policy_catalog(),
        )
        return ExecutionOutcomeMethodSeal(
            disposition="METHOD_BOUND",
            seal_marker=seal_marker,
            outcome_marker=outcome_marker,
            binding=binding,
        )

    def marker_for_snapshot(self, snapshot_hash: str) -> CausalExecutionOutcomeMarker:
        """Resolve the unique marker that authoritatively publishes a snapshot."""
        root = self.artifacts.root / "markers"
        matches = tuple(
            marker
            for path in (sorted(root.glob("*.json")) if root.is_dir() else ())
            if (
                marker := CausalExecutionOutcomeMarker.model_validate_json(path.read_bytes())
            ).snapshot_hash
            == snapshot_hash
        )
        if len(matches) != 1:
            raise ValueError("causal execution snapshot does not resolve one marker")
        return cast(CausalExecutionOutcomeMarker, matches[0])


class DevelopmentOnlyExecutionOutcomeReader:
    """Read one development-only snapshot, with no sealed-holdout capability at all.

    Deliberately not a subclass of the ordinary reader. Inheriting would give it
    ``read_sealed_holdout`` to override and a ``manifests`` category to share,
    and the point of the successor contract is that neither exists for it: there
    is no sealed child to release and no frozen manifest to be mistaken for.

    It offers the same three question-shapes the fold path asks -- which
    formations have matured, the rows for an admitted fold, and what method the
    snapshot is sealed to -- so a caller switches readers rather than switching
    code paths.
    """

    def __init__(self, artifact_root: Path) -> None:
        self.artifacts = _ExecutionOutcomeArtifactStore(artifact_root)

    def manifest_uri(self, snapshot_hash: str) -> str:
        return str(self.artifacts._uri(DEVELOPMENT_ONLY_MANIFEST_CATEGORY, snapshot_hash))

    def load_manifest(self, snapshot_hash: str) -> DevelopmentOnlyExecutionOutcomeManifest:
        return cast(
            DevelopmentOnlyExecutionOutcomeManifest,
            DevelopmentOnlyExecutionOutcomeManifest.model_validate(
                self.artifacts.load_json(
                    category=DEVELOPMENT_ONLY_MANIFEST_CATEGORY,
                    uri=self.manifest_uri(snapshot_hash),
                )
            ),
        )

    def _manifest_by_ref(self, manifest_ref: str) -> DevelopmentOnlyExecutionOutcomeManifest:
        return cast(
            DevelopmentOnlyExecutionOutcomeManifest,
            DevelopmentOnlyExecutionOutcomeManifest.model_validate(
                self.artifacts.load_json(
                    category=DEVELOPMENT_ONLY_MANIFEST_CATEGORY, uri=manifest_ref
                )
            ),
        )

    def read_development_sessions(self, manifest_ref: str, sessions: tuple[date, ...]) -> pa.Table:
        if not sessions or sessions != tuple(sorted(set(sessions))):
            raise ValueError("development session request must be ordered and non-empty")
        manifest = self._manifest_by_ref(manifest_ref)
        return _rows_for_sessions(self.artifacts, manifest.development_chunks, sessions)

    def available_development_sessions(
        self, manifest_ref: str, *, holding_end_through: date
    ) -> tuple[date, ...]:
        manifest = self._manifest_by_ref(manifest_ref)
        return _matured_sessions(
            self.artifacts, manifest.development_chunks, holding_end_through=holding_end_through
        )

    def read_sealed_holdout(self, _manifest_ref: str) -> pa.Table:
        """There is none, and saying so is different from refusing to release one."""
        raise PermissionError("DEVELOPMENT_ONLY_OUTCOME_HAS_NO_SEALED_HOLDOUT")

    def marker_for_snapshot(self, snapshot_hash: str) -> DevelopmentOnlyExecutionOutcomeMarker:
        matches = tuple(
            value
            for value in self.artifacts.development_only_markers()
            if value.snapshot_hash == snapshot_hash
        )
        if len(matches) != 1:
            raise ValueError("development-only snapshot does not resolve one marker")
        return matches[0]

    def resolve_method_seal(self, snapshot_hash: str) -> ExecutionOutcomeMethodSeal:
        """Resolve authority through the same terminal marker and the same verifier.

        A development-only snapshot with no seal is not "legacy" -- the contract
        postdates the seam, so an unsealed one was written by something that had
        no authority to write it. It refuses rather than degrading to readback.
        """
        manifest = self.load_manifest(snapshot_hash)
        markers = tuple(
            value
            for value in self.artifacts.method_seal_markers()
            if value.snapshot_hash == snapshot_hash
        )
        if len(markers) != 1:
            raise ExecutionOutcomeMethodError("DEVELOPMENT_ONLY_OUTCOME_SEAL_UNRESOLVED")
        seal_marker = markers[0]
        outcome_marker = self.marker_for_snapshot(snapshot_hash)
        binding = ExecutionOutcomeMethodBinding.model_validate(
            self.artifacts.load_json(category=METHOD_BINDING_CATEGORY, uri=seal_marker.binding_ref)
        )
        catalog = build_installed_execution_outcome_method_catalog()
        policies = build_installed_execution_outcome_publication_policy_catalog()
        if policies.resolve(binding.recipe_id).publication_scope != "DEVELOPMENT_ONLY":
            # A method whose admission covers a sealed holdout has no business
            # publishing under a contract that carries none.
            raise ExecutionOutcomeMethodError("DEVELOPMENT_ONLY_OUTCOME_SCOPE_NOT_ADMITTED")
        verify_execution_outcome_method_seal_marker(
            seal_marker=seal_marker,
            outcome_marker=outcome_marker,
            outcome_marker_ref=self.artifacts._uri(
                DEVELOPMENT_ONLY_MARKER_CATEGORY, outcome_marker.marker_hash
            ),
            manifest=manifest,
            manifest_ref=self.manifest_uri(snapshot_hash),
            binding=binding,
            binding_ref=self.artifacts._uri(METHOD_BINDING_CATEGORY, binding.binding_hash),
            catalog=catalog,
            expected_catalog_hash=catalog.binding.catalog_hash,
            policies=policies,
        )
        return ExecutionOutcomeMethodSeal(
            disposition="METHOD_BOUND",
            seal_marker=seal_marker,
            outcome_marker=outcome_marker,
            binding=binding,
        )


class CausalExecutionOutcomePolicyHoldoutReader(CausalExecutionOutcomeDevelopmentReader):
    """Read one claim-bound Policy Holdout without exposing a generic sealed reader."""

    _RELEASE_CATEGORY = "policy-holdout-releases"

    def __init__(
        self,
        *,
        market_store: PanelStateRepository,
        artifact_root: Path,
        authority: PolicyHoldoutExecutionOutcomeAuthority,
    ) -> None:
        """Verify claim authority and load its exact outcome release.

        Args:
            market_store: Source of the claimed market state.
            artifact_root: Root holding the durable outcome graph.
            authority: Claim-bound authority for this reader.

        Raises:
            ValueError: If the active claim or release cannot be verified.

        """
        super().__init__(artifact_root)
        self.market_store = market_store
        self.artifact_root = artifact_root.resolve()
        self.authority = authority
        self.manifest_ref = authority.source_manifest_ref
        self.claim_hash = authority.claim_hash
        self.embargo_session = authority.embargo_session
        self.formation_sessions = authority.formation_sessions
        self._verified_tables: dict[str, pa.Table] = {}
        self._validate_active_authority()
        self.release = self._obtain_release(authorized_at=authority.claimed_at)

    def _validate_active_authority(self) -> None:
        root = self.artifact_root / "model-validation" / "portfolio-policy"
        try:
            active = json.loads((root / "claims" / "active.json").read_text(encoding="utf-8"))
            if active != {"claim_hash": self.authority.claim_hash}:
                raise ValueError("active claim mismatch")
            claim = json.loads(
                (root / "claims" / "content" / f"{self.authority.claim_hash}.json").read_text(
                    encoding="utf-8"
                )
            )
            mandate = json.loads(
                (root / "mandates" / f"{self.authority.mandate_hash}.json").read_text(
                    encoding="utf-8"
                )
            )
            authority_index = json.loads(
                (
                    root
                    / "execution-outcome-authorities"
                    / "by-claim"
                    / f"{self.authority.claim_hash}.json"
                ).read_text(encoding="utf-8")
            )
            durable_authority = json.loads(
                (
                    root / "execution-outcome-authorities" / f"{self.authority.authority_hash}.json"
                ).read_text(encoding="utf-8")
            )
        except Exception as error:
            raise ValueError("policy holdout execution outcome authority is unavailable") from error
        claim_identity = {key: value for key, value in claim.items() if key != "claim_hash"}
        mandate_identity = {key: value for key, value in mandate.items() if key != "mandate_hash"}
        if (
            claim.get("claim_hash") != canonical_hash(claim_identity)
            or mandate.get("mandate_hash") != canonical_hash(mandate_identity)
            or claim.get("mandate_hash") != self.authority.mandate_hash
            or claim.get("slate_hash") != self.authority.slate_hash
            or claim.get("task_id") != self.authority.task_id
            or mandate.get("slate_hash") != self.authority.slate_hash
            or mandate.get("validation_numerical_binding_hash")
            != self.authority.validation_numerical_binding_hash
            or tuple(value["alpha_program_hash"] for value in mandate.get("alpha_recipes", ()))
            != self.authority.alpha_program_hashes
            or mandate.get("typed_user_authority") != "USER_AUTHORIZED_ONE_TIME_POLICY_HOLDOUT"
            or mandate.get("holdout_embargo_session") != self.authority.embargo_session.isoformat()
            or tuple(mandate.get("holdout_formation_sessions", ()))
            != tuple(value.isoformat() for value in self.authority.formation_sessions)
            or mandate.get("system_holdout_state") != "UNREAD"
            or authority_index
            != {
                "authority_hash": self.authority.authority_hash,
                "claim_hash": self.authority.claim_hash,
            }
            or durable_authority != self.authority.model_dump(mode="json")
        ):
            raise ValueError("policy holdout execution outcome authority is invalid")

    @classmethod
    def release_index_exists(cls, *, artifact_root: Path, claim_hash: str) -> bool:
        """Report physical release presence without treating corruption as absence."""
        artifacts = _ExecutionOutcomeArtifactStore(artifact_root)
        return (
            artifacts.root / cls._RELEASE_CATEGORY / "by-claim" / f"{claim_hash}.json"
        ).is_file()

    @classmethod
    def load_release(
        cls, *, artifact_root: Path, claim_hash: str
    ) -> PolicyHoldoutExecutionOutcomeRelease:
        """Load and verify the release indexed by one claim.

        Args:
            artifact_root: Root holding release artifacts.
            claim_hash: Authorized claim whose release is requested.

        Returns:
            The exact durable release for that claim.

        Raises:
            ValueError: If the release is missing, malformed, or misbound.

        """
        artifacts = _ExecutionOutcomeArtifactStore(artifact_root)
        index = artifacts.root / cls._RELEASE_CATEGORY / "by-claim" / f"{claim_hash}.json"
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
            if (
                set(payload) != {"claim_hash", "release_hash"}
                or payload["claim_hash"] != claim_hash
                or not isinstance(payload["release_hash"], str)
            ):
                raise ValueError("claim mismatch")
            value = PolicyHoldoutExecutionOutcomeRelease.model_validate(
                artifacts.load_json(
                    category=cls._RELEASE_CATEGORY,
                    uri=artifacts._uri(cls._RELEASE_CATEGORY, str(payload["release_hash"])),
                )
            )
        except Exception as error:
            raise ValueError("policy holdout execution outcome release is unavailable") from error
        if value.claim_hash != claim_hash:
            raise ValueError("policy holdout execution outcome release lineage is invalid")
        return cast(PolicyHoldoutExecutionOutcomeRelease, value)

    def _obtain_release(self, *, authorized_at: datetime) -> PolicyHoldoutExecutionOutcomeRelease:
        manifest = CausalExecutionOutcomeManifest.model_validate(
            self.artifacts.load_json(category="manifests", uri=self.manifest_ref)
        )
        if (
            manifest.snapshot_hash != self.authority.source_snapshot_hash
            or self.manifest_ref != self.authority.source_manifest_ref
            or self.formation_sessions != tuple(sorted(set(self.formation_sessions)))
            or len(self.formation_sessions) != 252
            or self.embargo_session >= self.formation_sessions[0]
        ):
            raise ValueError("policy holdout execution outcome axis is invalid")
        current = self.market_store.current_quality_filtered_research_manifest(
            market_profile_id="us-current-index-research"
        )
        if current is None:
            raise ValueError("policy holdout execution outcome manifest is unavailable")
        listing_ids = tuple(sorted(value.listing_id for value in current.listings))
        if (
            listing_ids != manifest.listing_ids
            or canonical_hash(listing_ids) != manifest.listing_set_hash
        ):
            raise ValueError("policy holdout execution outcome listing axis changed")
        final_point = self._final_schedule_point(authorized_at=authorized_at)
        watermark = self.market_store.execution_source_watermark(
            current, through=final_point.holding_end_session
        )
        release_index = (
            self.artifacts.root / self._RELEASE_CATEGORY / "by-claim" / f"{self.claim_hash}.json"
        )
        if release_index.is_file():
            durable = self.load_release(
                artifact_root=self.artifact_root, claim_hash=self.claim_hash
            )
            if (
                durable.source_snapshot_hash != manifest.snapshot_hash
                or durable.source_listing_set_hash != manifest.listing_set_hash
                or durable.universe_manifest_revision != current.revision_sha256
                or durable.source_watermark_hash != watermark["watermark_hash"]
                or durable.embargo_session != self.embargo_session
                or durable.formation_sessions != self.formation_sessions
            ):
                raise ValueError("policy holdout execution outcome release authority changed")
            self.artifacts.resolve_chunk(durable.extension_chunk)
            return durable

        extension = self._build_extension_chunk(
            current=current,
            point=final_point,
        )
        values = {
            "claim_hash": self.claim_hash,
            "source_snapshot_hash": manifest.snapshot_hash,
            "source_listing_set_hash": manifest.listing_set_hash,
            "universe_manifest_revision": current.revision_sha256,
            "source_watermark_hash": watermark["watermark_hash"],
            "embargo_session": self.embargo_session,
            "formation_sessions": self.formation_sessions,
            "extension_chunk": extension,
        }
        identity = PolicyHoldoutExecutionOutcomeRelease.model_construct(
            **values, release_hash=""
        ).model_dump(mode="json", exclude={"release_hash"})
        release = PolicyHoldoutExecutionOutcomeRelease(
            **values,
            release_hash=canonical_hash(identity),
        )
        self.artifacts.publish_json(
            category=self._RELEASE_CATEGORY,
            payload=release.model_dump(mode="json"),
            identity_field="release_hash",
        )
        index = (
            self.artifacts.root / self._RELEASE_CATEGORY / "by-claim" / f"{self.claim_hash}.json"
        )
        self.artifacts._atomic_write(
            index,
            json.dumps(
                {"claim_hash": self.claim_hash, "release_hash": release.release_hash},
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        )
        return self.load_release(artifact_root=self.artifact_root, claim_hash=self.claim_hash)

    def _final_schedule_point(self, *, authorized_at: datetime) -> CausalExecutionSchedulePoint:
        """Resolve the trailing point of the claim under the one-session method.

        The Policy Holdout release is frozen at the one-session clock, so this
        names that recipe explicitly rather than accepting one. What changed is
        that the offsets now come from the recipe instead of from ``common[1]``
        and ``common[2]``: the calendar is still the only source of sessions,
        but the arithmetic over it belongs to the method.
        """
        recipe = build_one_session_recipe()
        formation = self.formation_sessions[-1]
        # The lookahead is expressed in calendar days only to bound the calendar
        # query; every offset below indexes resolved ordered sessions. Sized so
        # the one-session window is exactly the seven days it always was.
        lookahead_days = 7 + 3 * (recipe.exit_offset_sessions - 2)
        calendar = materialize_calendar_schedule(
            ("XNYS", "XNAS"),
            start=formation,
            end=formation + date.resolution * lookahead_days,
            as_of_timestamp=authorized_at,
        )
        venues: dict[date, dict[str, dict[str, object]]] = {}
        for row in calendar.to_pylist():
            venues.setdefault(row["session_date"], {})[row["calendar_id"]] = row
        common: list[tuple[date, dict[str, object]]] = []
        for session, rows in sorted(venues.items()):
            if set(rows) != {"XNYS", "XNAS"}:
                continue
            xnys, xnas = rows["XNYS"], rows["XNAS"]
            if (
                xnys["session_open_timestamp"] != xnas["session_open_timestamp"]
                or xnys["session_close_timestamp"] != xnas["session_close_timestamp"]
            ):
                raise ValueError("common execution calendar venue clocks disagree")
            common.append((session, xnys))
        try:
            return resolve_final_schedule_point(
                recipe=recipe,
                ordered_sessions=tuple(session for session, _row in common),
                session_clocks={session: row for session, row in common},
                formation_session=formation,
            )
        except ExecutionOutcomeMethodError as error:
            raise ValueError("policy holdout final execution schedule is unavailable") from error

    def _build_extension_chunk(
        self,
        *,
        current: UniverseManifest,
        point: CausalExecutionSchedulePoint,
    ) -> CausalExecutionOutcomeChunk:
        listings = tuple(sorted(current.listings, key=lambda value: value.listing_id))
        recipe = build_one_session_recipe()
        axis = (point.formation_session, point.entry_session, point.holding_end_session)
        rows: list[dict[str, object]] = []
        connection = self.market_store._connect(read_only=True)
        try:
            for listing in listings:
                bars = self.market_store.raw_bars(
                    listing.listing_id,
                    start=point.entry_session,
                    through=point.holding_end_session,
                    _connection=connection,
                )
                by_session = {value.session_date: value for value in bars}
                actions = self.market_store.actions(listing.listing_id, _connection=connection)
                dividends = _action_dividends(actions, through=point.holding_end_session)
                rows.append(
                    derive_causal_execution_row(
                        listing_id=listing.listing_id,
                        symbol=listing.symbol,
                        point=point,
                        entry_bar=by_session.get(point.entry_session),
                        holding_bar=by_session.get(point.holding_end_session),
                        period_dividend_split_adjusted=period_dividend_for_point(
                            recipe=recipe,
                            dividends=dividends,
                            ordered_sessions=axis,
                            point=point,
                        ),
                    )
                )
        finally:
            connection.close()
        return self.artifacts.publish_chunk(
            split="SEALED_HOLDOUT",
            table=pa.Table.from_pylist(rows, schema=_schema()),
        )

    def _verified_table(self, chunk: CausalExecutionOutcomeChunk) -> pa.Table:
        cached = self._verified_tables.get(chunk.content_hash)
        if cached is not None:
            return cached
        table = (
            pq.read_table(self.artifacts.resolve_chunk(chunk))
            .combine_chunks()
            .sort_by([("formation_session", "ascending"), ("listing_id", "ascending")])
        )
        rows = table.to_pylist()
        if (
            table.num_rows != chunk.row_count
            or not rows
            or min(row["formation_session"] for row in rows) != chunk.first_formation_session
            or max(row["formation_session"] for row in rows) != chunk.last_formation_session
            or any(
                row["row_hash"]
                != canonical_hash({key: value for key, value in row.items() if key != "row_hash"})
                for row in rows
            )
            or canonical_hash(
                {
                    "schema": _SCHEMA_ID,
                    "rows": tuple(
                        (row["listing_id"], row["formation_session"], row["row_hash"])
                        for row in rows
                    ),
                }
            )
            != chunk.content_hash
        ):
            raise ValueError("policy holdout execution outcome chunk is tampered")
        self._verified_tables[chunk.content_hash] = table
        return table

    def _authorized_chunks(
        self, manifest: CausalExecutionOutcomeManifest
    ) -> tuple[CausalExecutionOutcomeChunk, ...]:
        return (
            *manifest.development_chunks,
            *manifest.sealed_holdout_chunks,
            self.release.extension_chunk,
        )

    def read_development_sessions(
        self,
        manifest_ref: str,
        sessions: tuple[date, ...],
    ) -> pa.Table:
        """Read only sessions admitted by this claim-bound release."""
        if (
            manifest_ref != self.manifest_ref
            or not sessions
            or sessions != tuple(sorted(set(sessions)))
        ):
            raise ValueError("policy holdout execution outcome request is invalid")
        manifest = CausalExecutionOutcomeManifest.model_validate(
            self.artifacts.load_json(category="manifests", uri=manifest_ref)
        )
        requested = set(sessions)
        chunks = tuple(
            chunk
            for chunk in self._authorized_chunks(manifest)
            if chunk.first_formation_session <= sessions[-1]
            and chunk.last_formation_session >= sessions[0]
        )
        tables = [self._verified_table(chunk) for chunk in chunks]
        if not tables:
            return pa.table({})
        table = pa.concat_tables(tables)
        filtered = table.filter(
            pc.is_in(
                table["formation_session"],
                value_set=pa.array(sessions, type=pa.date32()),
            )
        )
        observed = set(filtered["formation_session"].to_pylist())
        sealed = {
            value
            for value in observed
            if not any(
                chunk.first_formation_session <= value <= chunk.last_formation_session
                for chunk in manifest.development_chunks
            )
        }
        if (
            observed != requested
            or filtered.num_rows != len(sessions) * len(manifest.listing_ids)
            or not sealed.issubset(set(self.formation_sessions))
        ):
            raise ValueError("policy holdout execution outcome request exceeds authority")
        return filtered

    def available_development_sessions(
        self,
        manifest_ref: str,
        *,
        holding_end_through: date,
    ) -> tuple[date, ...]:
        """Return released formations mature by a holding-end date."""
        if manifest_ref != self.manifest_ref:
            raise ValueError("policy holdout execution outcome request is invalid")
        manifest = CausalExecutionOutcomeManifest.model_validate(
            self.artifacts.load_json(category="manifests", uri=manifest_ref)
        )
        tables = [
            pq.read_table(
                self.artifacts.resolve_chunk(chunk),
                columns=["formation_session", "holding_end_session"],
            )
            for chunk in self._authorized_chunks(manifest)
        ]
        table = pa.concat_tables(tables)
        eligible = table.filter(
            pc.less_equal(
                table["holding_end_session"],
                pa.scalar(holding_end_through, type=pa.date32()),
            )
        )
        return tuple(sorted(set(eligible["formation_session"].to_pylist())))

    def read_sealed_holdout(self, _manifest_ref: str) -> pa.Table:
        """Refuse generic sealed-holdout reads without a specific release."""
        raise PermissionError("GENERIC_SEALED_HOLDOUT_RELEASE_AUTHORITY_UNAVAILABLE")


__all__ = [
    "CausalExecutionOutcomeDevelopmentReader",
    "CausalExecutionOutcomePolicyHoldoutReader",
    "CausalOutcomeDevelopmentRows",
]
