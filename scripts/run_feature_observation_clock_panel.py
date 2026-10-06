"""Build one successor Feature Panel under the installed observation clock.

The source workspace is never written. A development workspace receives the
qualified inputs the source already holds -- raw bars, provider adjusted closes,
corporate actions, the universe manifest, the Sector map, the Market reference
and the Feature Input Gateway admission -- and the product's own owners recompute
everything derived from them under the installed catalog.

The derived plane is dropped rather than emptied, and dropped *before* the
installed catalog can differ. ``feature_daily_current`` in the source predates the
per-row catalog column, so leaving it in place would either be refused as
``feature_storage.row_catalog_identity_absent`` or -- worse, on an older build --
relabelled in place: 1.19M rows presented as this catalog's output with nothing
recomputed. Dropping the table lets the successor schema be created empty and
every row be computed by the code that claims it.

Stages are separate because the base closure is expensive and resumable:

``seed``     copy the source database and drop the derived plane
``build``    genesis, then the real ``FeatureFoundationService.build``
``publish``  the real snapshot publisher, including its semantic-index handoff
``verify``   re-derive the published Panel's clock authority from installed owners
``replay``   rebuild every Panel value from the durable closure and compare bytes

Nothing here calls a Provider: ``refresh_sector`` stays false and the Sector state
is the one the source workspace already qualified.

**This driver never deletes an existing admitted workspace, and it is worth
saying precisely what that does and does not mean.** It once took ``--fresh`` and
answered it with ``shutil.rmtree(output)`` on whatever path the caller passed,
which destroyed a published Panel exactly once before the capability was removed.

Removing ``--fresh`` was not sufficient. ``seed`` initializes a *copy* of the
source, and initializing it means dropping the derived plane the copy inherited:
the pre-successor base Features, their receipts, and the Panel publication tables.
That is a deliberate and necessary part of seeding a new workspace -- a copied
``feature_daily_current`` predates the per-row catalog column and no row of it may
answer for the installed catalog -- but it is emphatically not something that may
run against a workspace this driver has already seeded and that now holds a
published Panel. For one revision it could: admission returned ``RESUMED`` and
``seed`` dropped ``active_feature_panel_binding`` and
``feature_panel_snapshot_manifest`` anyway, while reporting zero destructive
operations in the same payload that listed what it had dropped.

So there are two statements, and only both together are true:

* the **source** workspace is never written, and that count is genuinely zero;
* the **output** workspace is initialized exactly once, at which point the stale
  derived objects the copy inherited are removed -- named and counted in the
  seed receipt rather than described as nothing.

``seed`` therefore accepts only a workspace it is initializing for the first
time. ``.successor-panel-workspace.json`` is written last, naming this driver and
the source; a later ``seed`` against a workspace carrying it is refused with
``successor_panel.output_workspace_already_seeded`` **before** anything is copied
and before the output database is opened. There is no reset or clean option: a
caller who wants a clean workspace passes a new ``--output-workspace``, which is
cheap and leaves the previous one intact to be compared against.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("ALPHALATTICE_NETWORK_DISABLED", "1")

PLAYPEN_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PLAYPEN_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import duckdb  # noqa: E402

from alphalattice.control.observation_runtime.telemetry.process_metrics import (  # noqa: E402
    bind_process_logical_processors,
    peak_rss_bytes,
    process_cpu_seconds,
)
from alphalattice.control.product_host.composition.workspace import WorkspaceRuntime  # noqa: E402
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver  # noqa: E402
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog  # noqa: E402
from alphalattice.foundation.feature_engine.contracts import (  # noqa: E402
    FeatureBuildRequest,
    FeatureInvalidation,
)
from alphalattice.foundation.feature_engine.inputs.closure_source import (  # noqa: E402
    FeatureClosureSourceRepository,
)
from alphalattice.foundation.feature_engine.panels.closure import (  # noqa: E402
    PanelClosurePublisher,
)
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (  # noqa: E402
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.closure_contracts import (  # noqa: E402
    PanelDerivationRecipe,
)
from alphalattice.foundation.feature_engine.panels.closure_source import (  # noqa: E402
    PanelClosureSourceRepository,
)
from alphalattice.foundation.feature_engine.panels.observation_clock_authority import (  # noqa: E402
    FeaturePanelObservationClockVerifier,
)
from alphalattice.foundation.feature_engine.panels.rematerialization import (  # noqa: E402
    ArtifactOnlyPanelRematerializer,
)
from alphalattice.foundation.feature_engine.producers.factors.specifications import (  # noqa: E402
    extension_factor_specs,
)
from alphalattice.foundation.feature_engine.runtime.closure_genesis import (  # noqa: E402
    FeatureClosureGenesisService,
)
from alphalattice.foundation.market_data_ops.sources.providers import (  # noqa: E402
    YFinanceMarketDataProvider,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import (  # noqa: E402
    MarketDataRepository,
)

WORKSPACE_OWNER_MARKER = ".successor-panel-workspace.json"
"""Names the driver that created a development workspace, its source, and itself.

Written last by ``seed``, so a workspace carrying it is one this driver finished
preparing rather than one it was interrupted in the middle of.

``output_workspace`` is the field that makes the marker a statement about *this*
location rather than a transferable badge. Without it a marker recorded only who
wrote it and from which source, so copying one verbatim into any other non-empty
directory made that directory answer as a workspace this runner had seeded --
and the stages behind the admission would then open whatever database happened to
be there. A marker now says "I was written for this path", and a copy says the
same thing about a path it is no longer in.
"""

WORKSPACE_OWNER_ID = "scripts/run_feature_observation_clock_panel.py"


class DevelopmentWorkspaceRefused(RuntimeError):
    """A typed refusal, so a caller can tell a boundary from an ordinary error."""


# There is deliberately no default output workspace. The one that used to be here
# named the round-two workspace and went stale the moment a later round seeded a
# different one, so a bare ``--stage verify`` refused at a path nobody was using
# any more -- a confusing failure that looked like a broken Panel rather than a
# stale constant. Replacing it with the current absolute path would only reset
# the clock on the same mistake. An output workspace is operational location, it
# changes whenever a new one is seeded, and the caller is the only party that
# knows which one it means.

DERIVED_TABLES = (
    # Base Feature plane: every value here was computed by the pre-successor
    # clock, and no row of it may answer for the installed catalog.
    "feature_daily_current",
    "feature_daily_revision",
    "feature_materialization_receipt",
    "feature_ineligibility_run",
    "feature_input_cutoff_set",
    "feature_catalog_current",
    # Publication plane: genesis refuses to invent an ancestor for a Panel that
    # already exists, and these are that Panel.
    "active_feature_panel_binding",
    "feature_panel_snapshot_manifest",
    "panel_factor_availability",
    "panel_factor_availability_revision",
    "panel_materialization_receipt",
    # Build/task bookkeeping for a plane that no longer exists.
    "feature_build_task",
    "feature_build_execution",
    "feature_build_event",
    "feature_build_deferred_retry",
    "feature_universe_rebuild_requirement",
    "sector_panel_rebuild_requirement",
)

DERIVED_VIEWS = ("feature_ineligibility", "feature_daily_runtime")


def _identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _write_evidence(output: Path, name: str, payload: dict[str, object]) -> Path:
    evidence = output.resolve() / "preparation-evidence"
    if not evidence.resolve().is_relative_to(output.resolve()):
        raise DevelopmentWorkspaceRefused("successor_panel.evidence_path_escapes_workspace")
    evidence.mkdir(parents=True, exist_ok=True)
    target = evidence / name
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return target


def _current_manifest(store: MarketDataRepository) -> Any:
    manifest = store.current_quality_filtered_research_manifest(
        market_profile_id="us-current-index-research"
    )
    if manifest is None:
        raise ValueError("SUCCESSOR_PANEL_UNIVERSE_MANIFEST_UNRESOLVED")
    return manifest


# ----------------------------------------------------------------------- seed


def _protected_roots(source: Path) -> tuple[Path, ...]:
    """Paths an output workspace may never be, resolved rather than compared raw.

    ``Path.resolve`` follows symlinks and junctions, which is the point: a link
    named ``scratch`` pointing at the repository is the case a string comparison
    misses.
    """

    return (
        source,
        PLAYPEN_ROOT,
        Path(__file__).resolve().parents[2],
        *source.parents,
        *PLAYPEN_ROOT.parents,
    )


def _read_seed_marker(output: Path) -> Path:
    """Parse and validate the seed marker in ``output``, returning its source.

    One parser, called by both admission paths. They ask different questions --
    ``_admit_output_workspace`` is deciding whether a directory is someone
    else's, ``_require_owned_seeded_workspace`` whether it has been seeded at
    all -- so each checks that the marker *exists* with its own failure code.
    Everything about the marker's *contents* is decided here, once, because two
    validators that agree today are two validators that can disagree later.

    ``output`` must already be resolved. The location check compares the
    canonical path the marker records against the canonical path it was found
    at, which is what a copied marker cannot satisfy.
    """

    marker = output / WORKSPACE_OWNER_MARKER
    try:
        declared = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DevelopmentWorkspaceRefused(
            "successor_panel.output_workspace_marker_unreadable"
        ) from error
    if not isinstance(declared, dict):
        raise DevelopmentWorkspaceRefused("successor_panel.output_workspace_marker_unreadable")
    if str(declared.get("owner")) != WORKSPACE_OWNER_ID:
        raise DevelopmentWorkspaceRefused(
            "successor_panel.output_workspace_not_owned_by_this_driver"
        )

    recorded_output = declared.get("output_workspace")
    if not isinstance(recorded_output, str) or not recorded_output:
        # A marker written before locations were bound. It is refused rather
        # than completed from the directory it was found in: inferring the value
        # from where the file sits is precisely the inference a copied marker
        # would satisfy, so it would grant exactly what it is meant to deny.
        # Nothing here rewrites or upgrades the marker on disk.
        raise DevelopmentWorkspaceRefused("successor_panel.output_workspace_marker_location_absent")
    declared_path = Path(recorded_output)
    if not declared_path.is_absolute() or declared_path.resolve() != output:
        # A relative path is refused without resolving it: what it names would
        # depend on the current working directory, so it is not a canonical
        # location even when it happens to point here.
        raise DevelopmentWorkspaceRefused(
            "successor_panel.output_workspace_marker_location_mismatch"
        )

    recorded_source = declared.get("source_workspace")
    if not isinstance(recorded_source, str) or not recorded_source:
        raise DevelopmentWorkspaceRefused("successor_panel.output_workspace_marker_unreadable")
    return Path(recorded_source).resolve()


def _admit_output_workspace(*, source: Path, output: Path) -> str:
    """Classify ``output`` for this driver, and refuse what it may not touch.

    Three recognised states: a path that does not exist yet, an empty directory,
    and a workspace this driver marked as its own having been seeded from this
    same source. Anything else is refused with a typed error rather than cleaned
    up, because a build driver that can delete is a build driver that will.

    ``RESUMED`` is a classification, not a permission, and ``seed`` is its only
    caller. Seeding means initializing a fresh copy, and running that against a
    workspace this driver already seeded is precisely the destructive act this
    admission exists to prevent -- so ``seed`` turns ``RESUMED`` into a refusal
    rather than proceeding on it. The later stages do not consult this function
    at all: they operate on a workspace that has already been seeded, and the
    marker is what tells a human which one that is.
    """

    if output == source:
        raise DevelopmentWorkspaceRefused("successor_panel.output_workspace_is_source_workspace")
    if source in output.parents:
        raise DevelopmentWorkspaceRefused("successor_panel.output_workspace_inside_source")
    if output in source.parents:
        raise DevelopmentWorkspaceRefused("successor_panel.output_workspace_contains_source")
    if output.parent == output:
        # A drive root has itself as a parent. Nothing may be seeded into one.
        raise DevelopmentWorkspaceRefused("successor_panel.output_workspace_is_filesystem_root")
    for protected in _protected_roots(source):
        if output == protected:
            raise DevelopmentWorkspaceRefused("successor_panel.output_workspace_is_protected_root")

    if not output.exists():
        return "CREATED"
    if not output.is_dir():
        raise DevelopmentWorkspaceRefused("successor_panel.output_workspace_is_not_a_directory")
    if not any(output.iterdir()):
        return "ADOPTED_EMPTY"

    marker = output / WORKSPACE_OWNER_MARKER
    if not marker.is_file():
        # Non-empty and not ours. This is the branch that used to be a recursive
        # delete, and the published Panel it destroyed is why it is a refusal.
        raise DevelopmentWorkspaceRefused(
            "successor_panel.output_workspace_not_owned_by_this_driver"
        )
    # Owner, recorded location and source shape are all decided by the one
    # parser; only the cross-check against the caller's source belongs here.
    if _read_seed_marker(output) != source:
        # Resuming across sources would silently mix two sets of qualified
        # inputs under one closure ledger.
        raise DevelopmentWorkspaceRefused(
            "successor_panel.output_workspace_seeded_from_another_source"
        )
    return "RESUMED"


def _require_owned_seeded_workspace(output: Path) -> Path:
    """Re-derive that ``output`` is a workspace this runner seeded, or refuse.

    Every stage that opens or writes the output workspace calls this first, and
    it is deliberately not enough to have guarded ``seed``. ``seed`` was the only
    stage with an admission, so a caller could skip it entirely --
    ``--stage build --output-workspace <the read-only source>`` reached
    ``MarketDataRepository`` and ``WorkspaceRuntime.create`` against the dogfood
    source with nothing in the way. Guarding one entrance is not guarding the
    building.

    The source workspace is read **from the marker**, never from a caller. A
    stage that accepted a declared source identity beside the path would let the
    caller supply both halves of the comparison, which is not a check.

    Returns the resolved output path so a caller uses the resolved one rather
    than re-deriving it.
    """

    output = Path(output).resolve()
    if not (output / WORKSPACE_OWNER_MARKER).is_file():
        # Covers "does not exist", "empty", and "someone else's directory": none
        # of them is a workspace this runner seeded.
        raise DevelopmentWorkspaceRefused("successor_panel.output_workspace_not_seeded")
    # The one parser: owner, the location the marker binds itself to, and the
    # shape of the source it names. A marker copied here from a workspace this
    # runner really did seed fails on the location, which is the whole point of
    # recording it.
    recorded_source = _read_seed_marker(output)

    # The same boundary rules ``seed`` is admitted under, re-run against the
    # source the marker names. One owner for those rules rather than two that can
    # drift, and a marker rewritten to point at a protected root or at the output
    # itself fails here exactly as it would have at seed time.
    admission = _admit_output_workspace(source=recorded_source, output=output)
    if admission != "RESUMED":
        raise DevelopmentWorkspaceRefused("successor_panel.output_workspace_not_seeded")
    return output


def seed(*, source: Path, output: Path) -> dict[str, object]:
    """Initialize one new development workspace from a read-only source.

    Initialization is destructive *to the copy*, by design: the derived plane the
    copy inherits was computed by a different catalog and no row of it may answer
    for the installed one. The receipt below names and counts every object
    removed rather than reporting the operation as globally non-destructive,
    which is what the predecessor did while dropping the publication tables.
    """

    source = source.resolve()
    output = output.resolve()
    admission = _admit_output_workspace(source=source, output=output)
    if admission not in {"CREATED", "ADOPTED_EMPTY"}:
        # Before any copy and before the output database is opened. A workspace
        # this driver already seeded holds a Panel that was published into it,
        # and initializing it again would drop `active_feature_panel_binding`
        # and `feature_panel_snapshot_manifest` along with the Feature plane.
        # There is no reset flag: the answer is a new --output-workspace.
        raise DevelopmentWorkspaceRefused("successor_panel.output_workspace_already_seeded")
    source_database = source / "market-data.duckdb"
    if source_database.with_suffix(".duckdb.wal").exists():
        raise DevelopmentWorkspaceRefused("successor_panel.source_database_not_quiescent")
    before = _identity(source_database)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "market-data.duckdb"
    started = time.perf_counter()
    if not target.exists():
        # Hold DuckDB's read-only lock throughout copying; another process must
        # not write the source while the destination snapshot is being formed.
        with duckdb.connect(str(source_database), read_only=True):
            shutil.copy2(source_database, target)
    staging = source / "staging" / "sector-reference"
    if staging.is_dir() and not (output / "staging" / "sector-reference").exists():
        shutil.copytree(staging, output / "staging" / "sector-reference")
    copied = time.perf_counter() - started

    dropped: list[str] = []
    # Read-write, and only here: this is the one step that edits the *copy*.
    with duckdb.connect(str(target), read_only=False) as connection:
        # Views first: ``feature_ineligibility`` and ``feature_daily_runtime``
        # read the tables below, so dropping the tables under them leaves the
        # schema owner unable to rebuild either. Both are recreated by
        # ``ensure_current_storage`` from the successor schema.
        for view in DERIVED_VIEWS:
            connection.execute(f"DROP VIEW IF EXISTS {view}")
            dropped.append(f"view:{view}")
        held = {
            str(row[0])
            for row in connection.execute("SELECT table_name FROM duckdb_tables()").fetchall()
        }
        for table in DERIVED_TABLES:
            if table in held:
                connection.execute(f"DROP TABLE {table}")
                dropped.append(table)
        connection.execute("CHECKPOINT")
        retained = connection.execute(
            """
            SELECT table_name, estimated_size FROM duckdb_tables()
            WHERE table_name IN (
                'raw_daily_bar_current', 'provider_adjusted_close_current',
                'corporate_action_current', 'sector_classification_current',
                'market_reference_current', 'universe_manifest',
                'feature_input_admission', 'current_universe_quality_admission'
            ) ORDER BY table_name
            """
        ).fetchall()
    after = _identity(source_database)
    if before != after:
        raise ValueError("SUCCESSOR_PANEL_SOURCE_WORKSPACE_MUTATED")
    # Written last, so an interrupted seed leaves a workspace this driver will
    # not later mistake for one it finished.
    (output / WORKSPACE_OWNER_MARKER).write_text(
        json.dumps(
            {
                "owner": WORKSPACE_OWNER_ID,
                "source_workspace": str(source),
                # Both paths are already resolved by ``seed``. Recording the
                # output binds the marker to the location it was written for,
                # so a verbatim copy elsewhere no longer answers for that
                # directory.
                "output_workspace": str(output),
                "seeded_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "kind": "SuccessorPanelSeed",
        "source_workspace": str(source),
        "output_workspace": str(output),
        "output_workspace_admission": admission,
        "source_identity_before": before,
        "source_identity_after": after,
        # Two counters, because one number cannot answer both questions and the
        # predecessor's single ``destructive_operations: 0`` was false about the
        # second while true about the first.
        "source_destructive_operations": 0,
        "source_writes": 0,
        "output_derived_objects_removed": len(dropped),
        "output_derived_objects_removed_names": dropped,
        "provider_calls": 0,
        "copy_seconds": round(copied, 3),
        "copied_bytes": target.stat().st_size,
        "retained_qualified_inputs": {name: size for name, size in retained},
    }


# --------------------------------------------------------------------- sector


def bind_sector(*, output: Path) -> dict[str, object]:
    """Publish the Sector revision map this workspace's own evidence implies.

    The Panel recovery binding resolves a Sector map from the closure ledger at
    publication, and a workspace with a new artifact root holds none. This is the
    Host's own step -- the maintenance coordinator calls the same owner method
    before it runs a build -- so it is a stage here rather than a repair.
    """

    output = _require_owned_seeded_workspace(output)
    catalog = FeatureCatalog.load()
    store = MarketDataRepository(output)
    manifest = _current_manifest(store)
    runtime = WorkspaceRuntime.create(
        workspace=output,
        manifest=manifest,
        provider=YFinanceMarketDataProvider(cache_root=output / "yfinance-cache"),
        artifact_root=output / "artifacts",
        feature_catalog=catalog,
    )
    started = time.perf_counter()
    try:
        bound = runtime.feature_foundation.sector_activation.bind_to_manifest(manifest)
    finally:
        runtime.close()
    return {
        "kind": "SuccessorPanelSectorBinding",
        "output_workspace": str(output),
        "manifest_revision": manifest.revision_sha256,
        "sector_revision": None if bound is None else bound[0],
        "store_receipt_hash": None if bound is None else bound[1],
        "wall_seconds": round(time.perf_counter() - started, 3),
        "provider_calls": 0,
        "source_writes": 0,
    }


# ---------------------------------------------------------------------- build


def build(*, output: Path, history_start: date, as_of: date | None) -> dict[str, object]:
    output = _require_owned_seeded_workspace(output)
    catalog = FeatureCatalog.load()
    artifact_root = output / "artifacts"
    store = MarketDataRepository(output)
    manifest = _current_manifest(store)
    # The real adapter, with the network disabled: any Provider call this route
    # made would fail closed and be visible rather than silently satisfied.
    provider = YFinanceMarketDataProvider(cache_root=output / "yfinance-cache")
    runtime = WorkspaceRuntime.create(
        workspace=output,
        manifest=manifest,
        provider=provider,
        artifact_root=artifact_root,
        feature_catalog=catalog,
    )
    started = time.perf_counter()
    stages: dict[str, float] = {}
    # The persistence layer measures its own stages and, until now, had nobody to
    # report them to. Collected here so a slow build is attributable from its own
    # receipt instead of from an investigation outside the process.
    persistence: dict[str, float] = {}

    def observe(stage: str, seconds: float) -> None:
        persistence[stage] = round(persistence.get(stage, 0.0) + seconds, 3)

    runtime.feature_foundation.persistence_timing_sink = observe
    try:
        # Schema before genesis: the closure source counts rows in
        # ``feature_daily_current`` and the ineligibility view reads a table the
        # market-data bootstrap owns. ``build`` does both in this order too; doing
        # it here is what lets genesis run first on a freshly re-seeded workspace.
        runtime.market_data.bootstrap(manifest)
        runtime.feature_state.ensure_current_storage()
        with duckdb.connect(str(store.database.path), read_only=True) as connection:
            watermark = connection.execute(
                "SELECT max(session_date) FROM raw_daily_bar_current"
            ).fetchone()
            spy = connection.execute(
                "SELECT revision_hash FROM market_reference_current LIMIT 1"
            ).fetchone()
        if watermark is None or spy is None:
            raise ValueError("SUCCESSOR_PANEL_SOURCE_STATE_INCOMPLETE")
        as_of_session = as_of or watermark[0]

        genesis_started = time.perf_counter()
        genesis = FeatureClosureGenesisService(
            panel_state=runtime.panel_state,
            source=FeatureClosureSourceRepository(runtime.database.path),
            ledger=runtime.closure_ledger,
        ).open_genesis(manifest=manifest, catalog=catalog)
        stages["genesis_seconds"] = round(time.perf_counter() - genesis_started, 3)

        request = FeatureBuildRequest.create(
            manifest_revision=manifest.revision_sha256,
            catalog=catalog.binding,
            spy_revision=str(spy[0]),
            history_start=history_start,
            as_of_session=as_of_session,
        )
        build_started = time.perf_counter()
        outcome = runtime.feature_foundation.build(
            request,
            observed_at=datetime.now(UTC),
            invalidations=(FeatureInvalidation("initial_backfill"),),
            refresh_sector=False,
        )
        stages["feature_build_seconds"] = round(time.perf_counter() - build_started, 3)
    finally:
        runtime.feature_foundation.persistence_timing_sink = None
        runtime.close()
    return {
        "kind": "SuccessorPanelBuild",
        "persistence_stage_seconds": dict(sorted(persistence.items())),
        "output_workspace": str(output),
        "catalog_hash": catalog.binding.catalog_hash,
        "formula_observation_policy_hash": catalog.binding.formula_observation_policy_hash,
        "source_availability_policy_hash": catalog.binding.source_availability_policy_hash,
        "manifest_revision": manifest.revision_sha256,
        "listing_count": len(manifest.listings),
        "history_start": history_start.isoformat(),
        "as_of_session": as_of_session.isoformat(),
        "genesis_disposition": genesis.disposition,
        "genesis_root_hash": genesis.root_hash,
        "genesis_head_hash": genesis.head_hash,
        "build_status": str(outcome.status),
        "build_stage": str(outcome.build_stage),
        "failure_code": outcome.failure_code,
        "materialization_receipt_hash": outcome.receipt_hash,
        "coverage_summary": outcome.coverage_summary,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "stage_seconds": stages,
        "peak_rss_bytes": peak_rss_bytes(),
        "provider_calls": 0,
        "source_writes": 0,
    }


# -------------------------------------------------------------------- publish


def publish(*, output: Path, history_start: date, as_of: date | None) -> dict[str, object]:
    output = _require_owned_seeded_workspace(output)
    catalog = FeatureCatalog.load()
    artifact_root = output / "artifacts"
    store = MarketDataRepository(output)
    manifest = _current_manifest(store)
    runtime = WorkspaceRuntime.create(
        workspace=output,
        manifest=manifest,
        provider=YFinanceMarketDataProvider(cache_root=output / "yfinance-cache"),
        artifact_root=artifact_root,
        feature_catalog=catalog,
    )
    started = time.perf_counter()
    try:
        with duckdb.connect(str(store.database.path), read_only=True) as connection:
            watermark = connection.execute(
                "SELECT max(session_date) FROM raw_daily_bar_current"
            ).fetchone()
        if watermark is None:
            raise ValueError("SUCCESSOR_PANEL_SOURCE_STATE_INCOMPLETE")
        published = runtime.panel_snapshot_publisher().publish(
            manifest=manifest,
            history_start=history_start,
            as_of_session=as_of or watermark[0],
            observed_at=datetime.now(UTC),
        )
    finally:
        runtime.close()
    snapshot = published.manifest
    governance = snapshot.safe_summary.get("quality_governance")
    resolver = ArtifactResolver(artifact_root)
    found = resolver.find_feature_panel_semantic_index(panel_snapshot_hash=snapshot.snapshot_hash)

    # The closure children, published by the product's own owner and published
    # *last*. Build and snapshot publication produce a genesis closure root,
    # which structurally carries no derivation recipe -- ``closure_root_for_head``
    # says so in as many words -- so a Panel published by them alone can be
    # verified at the relation layer and can never be numerically replayed.
    # ``PanelClosurePublisher`` is what freezes the base values, the availability
    # closure and the row receipts into immutable readback-verified children and
    # derives the recipe that names them. It runs here, after the Panel exists,
    # because a recipe is a statement *about* a published Panel: writing one
    # first would be inventing a causal parent.
    closure_started = time.perf_counter()
    closure_store = PanelClosureArtifactStore(resolver)
    publication = PanelClosurePublisher(
        resolver=resolver,
        source=PanelClosureSourceRepository(database_path=store.database.path, resolver=resolver),
        store=closure_store,
    ).publish()
    closure_seconds = time.perf_counter() - closure_started
    recipe = publication.recipes.get(snapshot.snapshot_hash)
    return {
        "kind": "SuccessorPanelPublication",
        "closure": {
            "recipe_hash": None if recipe is None else recipe.recipe_hash,
            "base_closure_hash": None if recipe is None else recipe.base_closure_hash,
            "availability_closure_hash": (
                None if recipe is None else recipe.availability_closure_hash
            ),
            "row_receipt_assignment_hash": (
                None if recipe is None else recipe.row_receipt_assignment_hash
            ),
            "sector_map_hash": None if recipe is None else recipe.sector_map_hash,
            "expected_chunk_count": None if recipe is None else len(recipe.expected_chunks),
            "rematerialization_policy": (
                None if recipe is None else recipe.rematerialization_policy
            ),
            "blocked_snapshot_hashes": sorted(publication.blocked_snapshots),
            "newly_written_bytes": publication.newly_written_bytes,
            "content_reused_artifact_count": publication.content_reused_artifact_count,
            "wall_seconds": round(closure_seconds, 3),
        },
        "output_workspace": str(output),
        "artifact_root": str(artifact_root),
        "snapshot_hash": snapshot.snapshot_hash,
        "panel_binding_hash": snapshot.panel_binding_hash,
        "panel_content_hash": snapshot.panel_content_hash,
        "manifest_ref": published.artifact.uri,
        "listing_set_hash": snapshot.listing_set_hash,
        "active_listing_count": snapshot.active_listing_count,
        "schema_hash": snapshot.schema_hash,
        "chunk_count": len(snapshot.chunks),
        "row_count": snapshot.safe_summary.get("row_count"),
        "date_range": snapshot.safe_summary.get("date_range"),
        "model_history_eligibility": snapshot.safe_summary.get("model_history_eligibility"),
        "gateway_qualified": (
            governance.get("gateway_qualified") if isinstance(governance, dict) else None
        ),
        "semantic_index_hash": (None if found is None else str(found[0]["index_hash"])),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "peak_rss_bytes": peak_rss_bytes(),
        "provider_calls": 0,
        "source_writes": 0,
    }


# --------------------------------------------------------------------- replay


def replay(*, output: Path, snapshot_hash: str) -> dict[str, object]:
    """Rematerialize the Panel from frozen base values and compare the bytes.

    Precisely what this is, because the shorter phrase "numerical replay" claims
    more than it does. ``ArtifactOnlyPanelRematerializer`` resolves the
    derivation recipe and its causal children, reads the **frozen base-value
    closure**, re-runs the Panel transformation over it, re-derives the row
    hashes, the schema hash and the chunk hash, writes the Parquet and compares
    the physical SHA-256 and byte count against what the published chunk
    recorded. The chunks are the expectation and the base closure is the input,
    so nothing is read back from the Panel and compared with itself.

    What it is **not** is an OHLCV-to-Formula replay. It starts at
    ``Feature(T)`` values that were already computed and frozen; it does not
    recompute a single Formula from raw bars. The evidence for that layer is the
    per-Formula perturbation audit in ``producers/consumption_audit.py``,
    which measures what each Formula actually consumes and where its window
    ends. Two different claims, two different owners, and neither substitutes
    for the other.

    It is pinned to the frozen transformation the Panel was built by rather than
    to whichever preprocessing recipe is installed today, which is what makes it
    a reproduction rather than a rebuild under a newer method.
    """

    catalog = FeatureCatalog.load()
    output = _require_owned_seeded_workspace(output)
    artifact_root = output / "artifacts"
    resolver = ArtifactResolver(artifact_root)
    store = PanelClosureArtifactStore(resolver)
    recipe_hash = _recipe_hash_for(store=store, artifact_root=artifact_root, snapshot=snapshot_hash)
    started = time.perf_counter()
    result = ArtifactOnlyPanelRematerializer(resolver=resolver, store=store).rematerialize(
        recipe_hash
    )
    elapsed = time.perf_counter() - started
    recipe = store.load_model(
        category="recipes", content_hash=recipe_hash, model=PanelDerivationRecipe
    )
    return {
        "kind": "SuccessorPanelRematerializationFromFrozenBaseValues",
        "artifact_root": str(artifact_root),
        "requested_snapshot_hash": snapshot_hash,
        "recipe_hash": recipe_hash,
        "base_closure_hash": recipe.base_closure_hash,
        "availability_closure_hash": recipe.availability_closure_hash,
        "row_receipt_assignment_hash": recipe.row_receipt_assignment_hash,
        "sector_map_hash": recipe.sector_map_hash,
        "catalog_hash": recipe.catalog_hash,
        "rematerialization_policy": recipe.rematerialization_policy,
        "replayed_snapshot_hash": result.snapshot_hash,
        "replayed_panel_content_hash": result.panel_content_hash,
        "snapshot_identity_exact": result.snapshot_hash == snapshot_hash,
        "logical_parity": result.logical_parity,
        "physical_parity": result.physical_parity,
        "recomputation_layer": "PANEL_TRANSFORM_OVER_FROZEN_BASE_VALUES",
        "recomputation_excludes": "OHLCV_TO_FORMULA_ARITHMETIC",
        # Counted from the installed surface rather than written down: a literal
        # went stale the moment a Formula was installed, and a receipt that
        # misstates how much evidence stands behind it is worse than one that
        # states none.
        "formula_arithmetic_evidence": (
            "producers/consumption_audit.py, "
            f"{len(catalog.factor_ids) + len(extension_factor_specs())}-Formula perturbation audit"
        ),
        # One call per published year, each a full re-run of that year's rows.
        # Reported as a count because "a replay ran" is not the same statement as
        # "every chunk was recomputed".
        "recomputation_calls": len(result.chunks),
        "recomputed_chunk_years": sorted({int(item.year) for item in result.chunks}),
        "recomputed_row_count": sum(int(item.row_count) for item in result.chunks),
        "wall_seconds": round(elapsed, 3),
        "peak_rss_bytes": peak_rss_bytes(),
        "provider_calls": 0,
        "source_writes": 0,
        "pointer_writes": 0,
    }


def _recipe_hash_for(
    *, store: PanelClosureArtifactStore, artifact_root: Path, snapshot: str
) -> str:
    """The recipe naming this snapshot, found by reading recipes rather than guessing.

    A recipe states which snapshot it derives; scanning them and matching on that
    field is the only lookup that cannot be satisfied by a recipe written for a
    different Panel.
    """

    directory = artifact_root / "feature-panel" / "closure" / "recipes"
    if not directory.is_dir():
        raise ValueError("feature_panel.closure_recipe_absent")
    for path in sorted(directory.glob("*.json")):
        recipe = store.load_model(
            category="recipes", content_hash=path.stem, model=PanelDerivationRecipe
        )
        if recipe.snapshot_hash == snapshot:
            return str(recipe.recipe_hash)
    raise ValueError("feature_panel.closure_recipe_absent")


# --------------------------------------------------------------------- verify


def verify(*, output: Path, snapshot_hash: str) -> dict[str, object]:
    """Re-derive one published Panel's clock authority from installed owners.

    Admitted like the write stages despite being read-only. The disposition it
    returns is a statement about *this runner's* successor Panel and it writes an
    evidence file naming that workspace, so pointing it at an arbitrary path
    would let it lend this runner's provenance to a Panel this runner never
    produced. Read-only is not the same as trustworthy-from-anywhere.

    A Panel outside a marked workspace is still verifiable: ``verify`` is a thin
    wrapper over ``FeaturePanelObservationClockVerifier``, which is a Feature
    owner and takes a resolver directly. What is refused is the claim of
    provenance, not the capability.
    """

    output = _require_owned_seeded_workspace(output)
    artifact_root = output / "artifacts"
    verifier = FeaturePanelObservationClockVerifier(resolver=ArtifactResolver(artifact_root))
    started = time.perf_counter()
    result = verifier.verify(snapshot_hash)
    return {
        "kind": "SuccessorPanelClockVerification",
        "artifact_root": str(artifact_root),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "provider_calls": 0,
        "source_writes": 0,
        "numerical_recomputation": "NONE",
        **asdict(result),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("seed", "sector", "build", "publish", "verify", "replay"),
        required=True,
    )
    parser.add_argument(
        "--source-workspace",
        type=Path,
        required=True,
        help="the source runtime to copy or read; an explicit operational location",
    )
    parser.add_argument(
        "--output-workspace",
        type=Path,
        required=True,
        help=(
            "the development workspace this stage acts on. Required, and it is "
            "operational location only: it is no part of Panel, catalog or Formula "
            "identity, and naming a different one never changes what a stage computes."
        ),
    )
    parser.add_argument("--history-start", type=date.fromisoformat, default=date(2016, 8, 1))
    parser.add_argument("--as-of", type=date.fromisoformat, default=None)
    parser.add_argument("--snapshot-hash", type=str, default=None)
    parser.add_argument(
        "--logical-processors",
        type=int,
        default=None,
        help=(
            "lower the budget below floor(cpu_count/2). It can only lower: a value "
            "above that ceiling, or above a narrower allowance this process already "
            "has, is refused rather than applied."
        ),
    )
    args = parser.parse_args()

    # Before any stage, and therefore before any copy, any DuckDB connection and
    # any numerical work. The predecessor set BLAS environment variables and
    # measured a 71% mean anyway, because those variables do not reach DuckDB;
    # a run that could not hold its ceiling was then capped by hand, which is an
    # operator action rather than a property of the runner. This is the runner
    # holding itself, and it refuses rather than proceeding unbounded.
    capacity = bind_process_logical_processors(args.logical_processors)
    started_cpu = process_cpu_seconds()

    if args.stage == "seed":
        payload = seed(source=args.source_workspace, output=args.output_workspace)
    elif args.stage == "sector":
        payload = bind_sector(output=args.output_workspace)
    elif args.stage == "build":
        payload = build(
            output=args.output_workspace, history_start=args.history_start, as_of=args.as_of
        )
    elif args.stage == "publish":
        payload = publish(
            output=args.output_workspace, history_start=args.history_start, as_of=args.as_of
        )
    elif args.stage == "replay":
        if args.snapshot_hash is None:
            parser.error("replay requires --snapshot-hash")
        payload = replay(output=args.output_workspace, snapshot_hash=args.snapshot_hash)
    else:
        if args.snapshot_hash is None:
            parser.error("verify requires --snapshot-hash")
        payload = verify(output=args.output_workspace, snapshot_hash=args.snapshot_hash)
    # Configured ceiling and observed consumption, side by side and never
    # summed: what a run was allowed and what it used are different facts, and
    # only the second can show the first was doing anything.
    payload = {
        **payload,
        "process_capacity": capacity,
        "process_cpu_seconds": round(process_cpu_seconds() - started_cpu, 3),
    }
    _write_evidence(args.output_workspace, f"gate-c-{args.stage}.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
