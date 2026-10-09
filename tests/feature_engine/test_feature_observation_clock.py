"""The Feature observation clock: whose session a value is, and when it is usable.

Every case here maps to one requirement of the successor boundary. The two
references each Formula is measured against are written in this file rather than
imported: comparing the materializer against itself would prove only that it is
deterministic.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from alphalattice.control.observation_runtime.telemetry import (
    process_metrics,
)
from alphalattice.control.observation_runtime.telemetry.process_metrics import (
    ProcessCapacityError,
    ProcessResourceMonitor,
    ResearchRuntimeRequest,
    RuntimeMachineCapacity,
    bind_process_logical_processors,
    current_allowed_logical_processors,
    default_logical_processor_budget,
    resolve_runtime_capacity_plan,
)
from alphalattice.control.workspace_runtime import reader_threads
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.database import WorkspaceDatabase
from alphalattice.foundation.causal_outcomes.execution.methods import build_one_session_recipe
from alphalattice.foundation.feature_engine.catalog.contracts import (
    FeatureCatalog,
    source_availability_binding,
)
from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    FEATURE_OBSERVATION_CLOCK_POLICY_ID,
    LEGACY_FEATURE_CLOCK,
    MARKET_REFERENCE_AUTHORITY,
    NEW_FEATURE_OBSERVATION_CLOCK,
    PROVIDER_AS_TRADED_AUTHORITY,
    PROVIDER_DAILY_BARS_AUTHORITY,
    SECTOR_AGGREGATE_AUTHORITY,
    SECTOR_CLASSIFICATION_AUTHORITY,
    VERIFIED_PANEL_CHILD_AUTHORITY,
    FeatureAvailabilityError,
    FeatureAvailabilityPolicy,
    FeatureObservationClock,
    formula_observation_policy_hash,
    formula_skip_sessions,
    formula_source_authority_binding_hash,
    installed_feature_availability_policy,
    installed_source_availability_catalog,
    latest_selectable_observation_session,
    observation_clock_for,
)
from alphalattice.foundation.feature_engine.contracts import FeaturePanelBinding
from alphalattice.foundation.feature_engine.panels.observation_clock_authority import (
    FeaturePanelObservationClockVerifier,
)
from alphalattice.foundation.feature_engine.producers.base_materializer import (
    BaseFeatureMaterializer,
)
from alphalattice.foundation.feature_engine.producers.consumption_audit import (
    FormulaConsumptionProbe,
    audit_installed_formula_consumption,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    catalog_implementation_hashes,
    catalog_methodology_hashes,
    default_extension_kernel_registry,
    extension_factor_specs,
    installed_formula_specs,
)
from alphalattice.foundation.feature_engine.producers.factors.specifications import (
    build_installed_factor_formula_specifications,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import DecisionCutoff, MarketPhase

ANNUALIZATION = math.sqrt(252.0)
_SKIP_MOMENTUM = ("mom_126_21", "mom_252_21", "residual_mom_252_21")


def _bars(periods: int, seed: int) -> pd.DataFrame:
    sessions = pd.bdate_range("2016-01-04", periods=periods)
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.012, periods))
    return pd.DataFrame(
        {
            "session_date": sessions,
            "open_raw": close * 0.997,
            "high_raw": close * 1.011,
            "low_raw": close * 0.989,
            "close_raw": close,
            "volume_raw": rng.integers(1_000_000, 5_000_000, periods).astype(float),
            "open_split_adjusted": close * 0.997,
            "high_split_adjusted": close * 1.011,
            "low_split_adjusted": close * 0.989,
            "close_split_adjusted": close,
            "provider_adjusted_close": close,
        }
    )


def _window_reference(
    values: np.ndarray, window: int, skip: int, function: Callable[[np.ndarray], float]
) -> np.ndarray:
    """The window whose final source row is ``t - skip``, evaluated independently."""

    output = np.full(len(values), np.nan, dtype=float)
    for end in range(window + skip - 1, len(values)):
        stop = end - skip + 1
        block = values[stop - window : stop]
        if np.isfinite(block).all():
            result = float(function(block))
            if np.isfinite(result):
                output[end] = result
    return output


def _log_returns(prices: np.ndarray) -> np.ndarray:
    output = np.full(len(prices), np.nan, dtype=float)
    positions = np.flatnonzero((prices[1:] > 0.0) & (prices[:-1] > 0.0)) + 1
    output[positions] = np.log(prices[positions] / prices[positions - 1])
    return output


def test_every_installed_formula_lands_its_window_on_the_observation_session() -> None:
    """requirement: Feature(T) is the Formula's value on observation session T."""

    catalog = FeatureCatalog.load()
    frame = _bars(400, seed=11)
    market = _bars(400, seed=13)
    block = BaseFeatureMaterializer(catalog).materialize_listing(
        listing_id="clock", projected_bars=frame, market_bars=market
    )
    prices = frame["provider_adjusted_close"].to_numpy(dtype=float)
    returns = _log_returns(prices)
    dollar_volume = frame["close_raw"].to_numpy(dtype=float) * frame["volume_raw"].to_numpy(
        dtype=float
    )
    cases = {
        "rev_5": (
            -_window_reference(returns, 5, 0, np.sum),
            -_window_reference(returns, 5, 1, np.sum),
        ),
        "vol_63": (
            _window_reference(returns, 63, 0, lambda item: float(np.std(item, ddof=1)))
            * ANNUALIZATION,
            _window_reference(returns, 63, 1, lambda item: float(np.std(item, ddof=1)))
            * ANNUALIZATION,
        ),
        "dollar_volume_21": (
            _window_reference(dollar_volume, 21, 0, np.mean),
            _window_reference(dollar_volume, 21, 1, np.mean),
        ),
    }
    for factor_id, (at_t, at_prior) in cases.items():
        observed = block.values[factor_id].to_numpy(dtype=float)
        np.testing.assert_allclose(observed, at_t, rtol=0.0, atol=1e-12, equal_nan=True)
        # regression: the pre-successor build matched this column instead, which
        # is what made a Feature named for session t one session stale.
        overlap = np.isfinite(observed) & np.isfinite(at_prior)
        assert overlap.any()
        assert float(np.max(np.abs(observed[overlap] - at_prior[overlap]))) > 0.0
    assert LEGACY_FEATURE_CLOCK == "CLOSE_T_MINUS_1_MASQUERADING_AS_T"
    assert NEW_FEATURE_OBSERVATION_CLOCK == "TRUE_OBSERVATION_SESSION"


def test_a_declared_economic_skip_survives_the_successor() -> None:
    """requirement: an economic skip is Formula meaning and is not removed."""

    catalog = FeatureCatalog.load()
    clocks = catalog.clocks_by_factor
    assert {clocks[factor_id].formula_skip_sessions for factor_id in _SKIP_MOMENTUM} == {21}
    # The interval is derived from the mechanical row minimum, not from the
    # declared span: 231 returns ending at r[t-21] need prices t-252..t-21.
    assert clocks["mom_252_21"].source_interval_rendered == "[t-252,t-21]"
    assert clocks["rev_5"].source_interval_rendered == "[t-5,t]"

    frame = _bars(400, seed=17)
    block = BaseFeatureMaterializer(catalog).materialize_listing(
        listing_id="skip", projected_bars=frame, market_bars=_bars(400, seed=19)
    )
    returns = _log_returns(frame["provider_adjusted_close"].to_numpy(dtype=float))
    np.testing.assert_allclose(
        block.values["mom_252_21"].to_numpy(dtype=float),
        _window_reference(returns, 231, 21, np.sum),
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    calendar_ids = catalog.calendar_factor_ids
    for spec in catalog.factors:
        clock = clocks[spec.factor_id]
        assert clock.policy_id == FEATURE_OBSERVATION_CLOCK_POLICY_ID
        if spec.factor_id in calendar_ids:
            # A calendar selection has no session offset at all, and saying it has
            # one is the generalisation this contract exists to refuse.
            assert clock.latest_consumed_session_offset is None
            assert not clock.source_interval.minimum_history_is_mechanical
        else:
            assert clock.latest_consumed_session_offset == formula_skip_sessions(spec)
            assert clock.source_interval.minimum_history_is_mechanical


def test_the_recorded_cutoff_names_the_latest_consumed_observation_session() -> None:
    """regression: the persisted cutoff said t-1 for every Formula that had no skip."""

    catalog = FeatureCatalog.load()
    frame = _bars(300, seed=23)
    block = BaseFeatureMaterializer(catalog).materialize_listing(
        listing_id="cutoff", projected_bars=frame, market_bars=_bars(300, seed=29)
    )
    labels = pd.to_datetime(frame["session_date"]).dt.strftime("%Y-%m-%d").to_numpy()
    for position in (0, 1, 21, 120, len(frame) - 1):
        recorded = json.loads(str(block.values.iloc[position]["input_cutoffs_json"]))
        assert recorded["rev_5"] == labels[position]
        assert recorded["mom_252_21"] == (labels[position - 21] if position >= 21 else None)


def test_the_canonical_axis_refuses_a_lagged_projection_of_a_formula() -> None:
    """The canonical axis refuses a lagged projection of a formula."""

    catalog = FeatureCatalog.load()
    payload = catalog.to_payload()
    donor = dict(payload["factors"][0])
    for suffix in ("_lag_1", "_lagged_5", "_shift_2"):
        projection = {**donor, "factor_id": f"{donor['factor_id']}{suffix}"}
        candidate = {
            **payload,
            "factors": sorted(
                [*payload["factors"], projection], key=lambda item: str(item["factor_id"])
            ),
        }
        with pytest.raises(ValueError, match="lagged projection"):
            FeatureCatalog.from_payload(candidate)

    # A Formula whose own mathematics happens to mention a lag is not a
    # projection, and the rule must not refuse one: the stem has to name an
    # installed Formula for the name to be a shift of anything.
    independent = {**donor, "factor_id": "zz_independent_ewma_lag_1"}
    candidate = {
        **payload,
        "factors": sorted(
            [*payload["factors"], independent], key=lambda item: str(item["factor_id"])
        ),
    }
    assert FeatureCatalog.from_payload(candidate).factor_ids[-1] == "zz_independent_ewma_lag_1"


def test_source_availability_is_owned_per_authority_not_per_factor() -> None:
    """Source availability is owned per authority not per factor."""

    catalog = installed_source_availability_catalog()

    # Owners are per authority: the five distinct source owners the installed Formulas
    # read, and the as-traded owner a formula reading a point-in-time leaf reads.
    assert {item.policy_id for item in catalog.owners} == {
        PROVIDER_AS_TRADED_AUTHORITY,
        PROVIDER_DAILY_BARS_AUTHORITY,
        MARKET_REFERENCE_AUTHORITY,
        SECTOR_AGGREGATE_AUTHORITY,
        SECTOR_CLASSIFICATION_AUTHORITY,
        VERIFIED_PANEL_CHILD_AUTHORITY,
    }

    specifications = build_installed_factor_formula_specifications()
    derived = {
        factor_id: specifications.resolve(factor_id).source_authorities
        for factor_id in specifications.factor_ids
    }
    # The Formulas the price-feed policy could not have answered for, named.
    assert derived["sector_leader_lag_5"] == (
        PROVIDER_DAILY_BARS_AUTHORITY,
        SECTOR_CLASSIFICATION_AUTHORITY,
    )
    assert derived["residual_reversal_vol_scaled_5"] == (
        PROVIDER_DAILY_BARS_AUTHORITY,
        SECTOR_AGGREGATE_AUTHORITY,
    )
    # Reads a Market row and another Formula's output, and no provider bar.
    assert derived["market_vol_ratio_x_reversal"] == (
        MARKET_REFERENCE_AUTHORITY,
        VERIFIED_PANEL_CHILD_AUTHORITY,
    )
    assert PROVIDER_DAILY_BARS_AUTHORITY not in derived["market_vol_ratio_x_reversal"]

    # A field no installed owner claims is refused rather than inheriting the
    # price feed's schedule. This is what an external non-OHLCV Formula meets.
    for fields in (("sec_filing_text",), ("provider_adjusted_close", "analyst_revision_feed")):
        with pytest.raises(FeatureAvailabilityError, match="source_availability_owner_uninstalled"):
            catalog.authorities_for(fields)

    # The governing event is the latest across the authorities, so a Panel and a
    # Formula are both answered by one comparison rather than by an assumption.
    governing = catalog.effective_policy(item.policy_id for item in catalog.owners)
    assert governing.available_after_phase is MarketPhase.OFFICIAL_CLOSE
    assert governing.publication_delay_sessions == 0

    installed = FeatureCatalog.load()
    # Schedules are bound once for the catalog; which owners a Formula needs is
    # bound per Formula. They are separate hashes because a Provider changing a
    # publication time and a Formula gaining a Sector dependency are different
    # events, and one hash could not tell them apart. The schedules' binding is held at the
    assert installed.binding.source_availability_policy_hash == source_availability_binding(
        catalog.catalog_hash
    )
    assert installed.binding.source_authority_binding_hash == (
        formula_source_authority_binding_hash(installed.formula_source_authorities)
    )
    # And neither of them touches what the Formula computes.
    assert installed.binding.formula_observation_policy_hash == formula_observation_policy_hash(
        installed.observation_clocks
    )


def test_the_panel_driver_never_deletes_a_development_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The panel driver never deletes a development workspace."""

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    path = scripts / "run_feature_observation_clock_panel.py"
    spec = importlib.util.spec_from_file_location("run_feature_observation_clock_panel", path)
    assert spec is not None and spec.loader is not None
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    # The capability is gone from the module, not merely unreachable from the CLI.
    # Names in the syntax tree, not text: the module's own docstring says the word
    # while explaining why it no longer does the thing.

    source = (tmp_path / "source").resolve()
    source.mkdir()
    repo_root = Path(__file__).resolve().parents[2]

    def admit(output: Path) -> str:
        return str(driver._admit_output_workspace(source=source, output=output.resolve()))

    for output, code in (
        (source, "output_workspace_is_source_workspace"),
        (source / "nested", "output_workspace_inside_source"),
        (source.parent, "output_workspace_contains_source"),
        (repo_root, "output_workspace_is_protected_root"),
        (Path(source.anchor), "output_workspace_contains_source"),
    ):
        with pytest.raises(driver.DevelopmentWorkspaceRefused, match=code):
            admit(output)

    # A drive that does not contain the source still cannot be an output root.
    other_root = Path("C:/") if source.anchor != "C:\\" else Path("D:/")
    with pytest.raises(driver.DevelopmentWorkspaceRefused, match="filesystem_root"):
        admit(other_root)

    assert admit(tmp_path / "not-yet-created") == "CREATED"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert admit(empty) == "ADOPTED_EMPTY"

    # Non-empty and unmarked is exactly the shape of a workspace holding someone
    # else's published Panel. It is refused, and its contents survive.
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "published.parquet").write_bytes(b"artifact")
    with pytest.raises(driver.DevelopmentWorkspaceRefused, match="not_owned_by_this_driver"):
        admit(foreign)
    assert (foreign / "published.parquet").read_bytes() == b"artifact"

    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / "market-data.duckdb").write_bytes(b"prior build")
    marker = owned / driver.WORKSPACE_OWNER_MARKER
    marker.write_text(
        json.dumps(
            {
                "owner": driver.WORKSPACE_OWNER_ID,
                "source_workspace": str(source),
                "output_workspace": str(owned.resolve()),
            }
        ),
        encoding="utf-8",
    )
    # Resuming our own workspace keeps every byte that was already there.
    assert admit(owned) == "RESUMED"
    assert (owned / "market-data.duckdb").read_bytes() == b"prior build"

    marker.write_text(
        json.dumps(
            {
                "owner": driver.WORKSPACE_OWNER_ID,
                "source_workspace": str(tmp_path),
                "output_workspace": str(owned.resolve()),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(driver.DevelopmentWorkspaceRefused, match="seeded_from_another_source"):
        admit(owned)

    # ------------------------------------------------------------------
    # The public route, twice. Classifying a workspace as RESUMED was never the
    # defect; proceeding to seed it was. `seed` initializes a copy, and
    # initializing means dropping the derived plane the copy inherited -- which
    # against an already-seeded workspace means dropping a published Panel's
    # `active_feature_panel_binding` and `feature_panel_snapshot_manifest`. It
    # did exactly that for one revision, and reported zero destructive
    # operations in the same payload that listed what it had dropped.
    #
    # Driven through `seed` rather than `_admit_output_workspace`, because the
    # admission call was already correct in isolation and the damage was in what
    # the caller did with its answer.
    fixture = tmp_path / "isolated-source"
    fixture.mkdir()
    with duckdb.connect(str(fixture / "market-data.duckdb")) as connection:
        # One retained qualified input and two derived objects, which is enough
        # to tell "initialized" from "destroyed".
        connection.execute("CREATE TABLE raw_daily_bar_current (listing_id VARCHAR)")
        connection.execute("INSERT INTO raw_daily_bar_current VALUES ('L0')")
        connection.execute("CREATE TABLE feature_daily_current (listing_id VARCHAR)")
        connection.execute("CREATE TABLE active_feature_panel_binding (snapshot_hash VARCHAR)")
        connection.execute("CREATE TABLE feature_panel_snapshot_manifest (snapshot_hash VARCHAR)")

    output = tmp_path / "isolated-output"
    first = driver.seed(source=fixture, output=output)
    evidence = driver._write_evidence(output, "seed.json", first)
    assert evidence.parent == output / "preparation-evidence"
    assert json.loads(evidence.read_text(encoding="utf-8")) == first
    assert not (fixture / "preparation-evidence").exists()

    # The first seed is the admitted initialization, and it reports what it did
    # to the copy rather than calling the whole operation non-destructive.
    assert first["output_workspace_admission"] == "CREATED"
    assert first["source_destructive_operations"] == 0
    assert first["source_writes"] == 0
    removed = list(first["output_derived_objects_removed_names"])  # type: ignore[call-overload]
    assert first["output_derived_objects_removed"] == len(removed)
    assert {"feature_daily_current", "active_feature_panel_binding"} <= set(removed)
    assert "destructive_operations" not in first

    # Now the workspace looks like one that has been built and published into.
    seeded_db = output / "market-data.duckdb"
    with duckdb.connect(str(seeded_db)) as connection:
        connection.execute("CREATE TABLE active_feature_panel_binding (snapshot_hash VARCHAR)")
        connection.execute("INSERT INTO active_feature_panel_binding VALUES ('published')")
    published_artifact = output / "artifacts" / "feature-panel" / "chunks" / "2024.parquet"
    published_artifact.parent.mkdir(parents=True, exist_ok=True)
    published_artifact.write_bytes(b"published chunk bytes")

    marker_before = (output / driver.WORKSPACE_OWNER_MARKER).read_bytes()
    database_before = seeded_db.read_bytes()
    artifact_before = published_artifact.read_bytes()
    mtime_before = seeded_db.stat().st_mtime_ns

    with pytest.raises(driver.DevelopmentWorkspaceRefused, match="output_workspace_already_seeded"):
        driver.seed(source=fixture, output=output)

    # Bytes, not timestamps: an open-for-write that dropped nothing would still
    # rewrite the database header, so comparing content is what proves the
    # refusal landed before the connection rather than after it.
    assert seeded_db.read_bytes() == database_before
    assert seeded_db.stat().st_mtime_ns == mtime_before
    assert (output / driver.WORKSPACE_OWNER_MARKER).read_bytes() == marker_before
    assert published_artifact.read_bytes() == artifact_before
    with duckdb.connect(str(seeded_db), read_only=True) as connection:
        held = {
            str(row[0])
            for row in connection.execute("SELECT table_name FROM duckdb_tables()").fetchall()
        }
        assert "active_feature_panel_binding" in held
        assert connection.execute(
            "SELECT count(*) FROM active_feature_panel_binding"
        ).fetchone() == (1,)

    # ------------------------------------------------------------------
    # Guarding `seed` guarded one entrance. Every stage that opens or writes the
    # output workspace takes `--output-workspace` too, so a caller could skip
    # seeding entirely -- `--stage build --output-workspace <the read-only
    # source>` reached `MarketDataRepository` and `WorkspaceRuntime.create`
    # against the dogfood source with nothing in the way.
    write_stages = {
        "bind_sector": lambda path: driver.bind_sector(output=path),
        "build": lambda path: driver.build(output=path, history_start=date(2016, 8, 1), as_of=None),
        "publish": lambda path: driver.publish(
            output=path, history_start=date(2016, 8, 1), as_of=None
        ),
        "verify": lambda path: driver.verify(output=path, snapshot_hash="a" * 64),
        "replay": lambda path: driver.replay(output=path, snapshot_hash="a" * 64),
    }

    # Every opener the stages reach for, replaced by a tripwire. If a refusal
    # arrives *after* one of these, the counter says so; a test that only checked
    # the exception could not tell the difference between refusing early and
    # refusing after opening the source database read-write.
    opened: list[str] = []

    def tripwire(name: str):  # type: ignore[no-untyped-def]
        def _fail(*_args: object, **_kwargs: object) -> object:
            opened.append(name)
            raise AssertionError(f"{name} was reached before the workspace was admitted")

        return _fail

    hostile = tmp_path / "hostile"
    hostile.mkdir()
    (hostile / "market-data.duckdb").write_bytes(b"not this driver's")

    relocated = tmp_path / "relocated"
    relocated.mkdir()
    (relocated / driver.WORKSPACE_OWNER_MARKER).write_text(
        json.dumps(
            # A marker rewritten so the workspace claims to have been seeded from
            # itself. The boundary rules must be re-run against what the marker
            # says, not assumed to have held once at seed time.
            {
                "owner": driver.WORKSPACE_OWNER_ID,
                "source_workspace": str(relocated),
                "output_workspace": str(relocated.resolve()),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    foreign_owner = tmp_path / "foreign-owner"
    foreign_owner.mkdir()
    (foreign_owner / driver.WORKSPACE_OWNER_MARKER).write_text(
        json.dumps({"owner": "some/other/driver.py", "source_workspace": str(fixture)}),
        encoding="utf-8",
    )

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / driver.WORKSPACE_OWNER_MARKER).write_text("{ not json", encoding="utf-8")

    refusals = {
        # The source workspace itself: the case that motivated this. It carries
        # no marker, so it is refused at the marker rather than at the path
        # comparison -- earlier and stricter than the rule that would have caught
        # it anyway. The `relocated` case below is what proves the path relations
        # are still re-derived for a workspace that does carry one.
        "source": (fixture, "output_workspace_not_seeded"),
        "unmarked": (hostile, "output_workspace_not_seeded"),
        "absent": (tmp_path / "never-existed", "output_workspace_not_seeded"),
        "foreign_owner": (foreign_owner, "output_workspace_not_owned_by_this_driver"),
        "malformed_marker": (malformed, "output_workspace_marker_unreadable"),
        "relocated": (relocated, "output_workspace_is_source_workspace"),
    }

    for stage_name, call in write_stages.items():
        for label, (workspace, code) in refusals.items():
            opened.clear()
            with monkeypatch.context() as patched:
                for attribute in (
                    "MarketDataRepository",
                    "WorkspaceRuntime",
                    "ArtifactResolver",
                    "PanelClosureArtifactStore",
                    "ArtifactOnlyPanelRematerializer",
                    "FeaturePanelObservationClockVerifier",
                    "duckdb",
                ):
                    patched.setattr(driver, attribute, tripwire(attribute), raising=True)
                with pytest.raises(driver.DevelopmentWorkspaceRefused, match=code):
                    call(workspace)
            assert opened == [], f"{stage_name}/{label} reached {opened} before refusing"

    # The source workspace was named as an output five times and is untouched.
    assert sorted(item.name for item in fixture.iterdir()) == ["market-data.duckdb"]

    # ------------------------------------------------------------------
    # The marker-copy attack, which every one of the hostile cases above misses.
    # They all fail on something *inside* the marker -- wrong owner, malformed
    # JSON, a source that is the output. A marker copied verbatim out of a
    # workspace this runner really did seed has none of those problems: the owner
    # is right, the JSON is right, and the source is the real source. Before the
    # marker bound its own location, that copy made any directory answer as a
    # seeded workspace and the stages behind it opened whatever database was
    # sitting there.
    stolen = tmp_path / "stolen"
    stolen.mkdir()
    (stolen / driver.WORKSPACE_OWNER_MARKER).write_bytes(
        (output / driver.WORKSPACE_OWNER_MARKER).read_bytes()
    )
    # Made to look worth opening, so a stage that got past admission would have
    # something to do rather than failing on an empty directory by luck.
    with duckdb.connect(str(stolen / "market-data.duckdb")) as connection:
        connection.execute("CREATE TABLE raw_daily_bar_current (listing_id VARCHAR)")
    stolen_artifact = stolen / "artifacts" / "feature-panel" / "manifests" / "x.json"
    stolen_artifact.parent.mkdir(parents=True, exist_ok=True)
    stolen_artifact.write_bytes(b'{"kind": "not ours"}')

    def _fingerprint(root: Path) -> dict[str, tuple[bytes, int]]:
        return {
            item.relative_to(root).as_posix(): (item.read_bytes(), item.stat().st_mtime_ns)
            for item in sorted(root.rglob("*"))
            if item.is_file()
        }

    stolen_before = _fingerprint(stolen)

    # A marker that records a location it is not in, and the variants a careless
    # reader might accept: no location at all, a relative one, one naming a
    # different workspace, and ones naming the source and a protected root.
    def _marker(directory: Path, **overrides: object) -> Path:
        payload = json.loads((output / driver.WORKSPACE_OWNER_MARKER).read_text(encoding="utf-8"))
        payload.update(overrides)
        target = directory
        target.mkdir(parents=True, exist_ok=True)
        (target / driver.WORKSPACE_OWNER_MARKER).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        (target / "market-data.duckdb").write_bytes(b"unrelated database")
        return target

    location_cases = {
        # The verbatim copy: every field correct except the one that says where
        # this marker belongs.
        "copied": (stolen, "output_workspace_marker_location_mismatch"),
        # A marker written before locations were bound. Refused, never completed
        # from the directory it was found in -- inferring it from the file's own
        # location is exactly what the copy would satisfy.
        "historic_no_location": (
            _marker(tmp_path / "historic-marker", output_workspace=None),
            "output_workspace_marker_location_absent",
        ),
        "empty_location": (
            _marker(tmp_path / "empty-location", output_workspace=""),
            "output_workspace_marker_location_absent",
        ),
        # Relative: what it names depends on the working directory, so it is not
        # a canonical location even if it happens to resolve here.
        "relative_location": (
            _marker(tmp_path / "relative-location", output_workspace="./relative-location"),
            "output_workspace_marker_location_mismatch",
        ),
        "another_workspace": (
            _marker(tmp_path / "another-workspace", output_workspace=str(output)),
            "output_workspace_marker_location_mismatch",
        ),
        "names_the_source": (
            _marker(tmp_path / "names-source", output_workspace=str(fixture)),
            "output_workspace_marker_location_mismatch",
        ),
        "names_protected_root": (
            _marker(tmp_path / "names-protected", output_workspace=str(repo_root)),
            "output_workspace_marker_location_mismatch",
        ),
    }

    for stage_name, call in write_stages.items():
        for label, (workspace, code) in location_cases.items():
            opened.clear()
            with monkeypatch.context() as patched:
                for attribute in (
                    "MarketDataRepository",
                    "WorkspaceRuntime",
                    "ArtifactResolver",
                    "PanelClosureArtifactStore",
                    "ArtifactOnlyPanelRematerializer",
                    "FeaturePanelObservationClockVerifier",
                    "duckdb",
                ):
                    patched.setattr(driver, attribute, tripwire(attribute), raising=True)
                with pytest.raises(driver.DevelopmentWorkspaceRefused, match=code):
                    call(workspace)
            assert opened == [], f"{stage_name}/{label} reached {opened} before refusing"

    # Bytes and mtime: the copied workspace was named as an output by five
    # stages and nothing in it was read open, written or touched.
    assert _fingerprint(stolen) == stolen_before

    # The workspace the marker was stolen *from* still admits, so the location
    # binding refuses the copy rather than invalidating the original.
    assert driver._require_owned_seeded_workspace(output) == output

    # And a legitimately seeded workspace still gets through admission into the
    # stage's own logic -- proven by the tripwire firing, which is the first
    # thing past the gate rather than a refusal.
    with monkeypatch.context() as patched:
        patched.setattr(driver, "ArtifactResolver", tripwire("ArtifactResolver"), raising=True)
        with pytest.raises(AssertionError, match="ArtifactResolver was reached"):
            driver.verify(output=output, snapshot_hash="a" * 64)

    # ------------------------------------------------------------------
    # There is no default output workspace, and there should not be one. The
    # constant that used to be here named the round-two workspace and went stale
    # as soon as a later round seeded a different one, so a bare `--stage verify`
    # refused at a path nobody used any more -- a failure that reads like a
    # broken Panel rather than a stale default. Replacing it with today's
    # absolute path would only restart the same clock.
    assert not hasattr(driver, "DEFAULT_OUTPUT")

    # argparse refuses the omission, and it does so before the capacity binding
    # and before any workspace is read: `parse_args` is the statement above it.
    with monkeypatch.context() as patched:
        patched.setattr(
            sys, "argv", ["run_feature_observation_clock_panel.py", "--stage", "verify"]
        )
        for attribute in ("MarketDataRepository", "WorkspaceRuntime", "ArtifactResolver", "duckdb"):
            patched.setattr(driver, attribute, tripwire(attribute), raising=True)
        patched.setattr(
            driver, "bind_process_logical_processors", tripwire("bind_process_logical_processors")
        )
        opened.clear()
        with pytest.raises(SystemExit) as exit_info:
            driver.main()
    assert exit_info.value.code == 2
    assert opened == []


def test_the_runner_is_capacity_bounded_before_any_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner is capacity bounded before any stage."""

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    path = scripts / "run_feature_observation_clock_panel.py"
    spec = importlib.util.spec_from_file_location("run_feature_observation_clock_panel", path)
    assert spec is not None and spec.loader is not None
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    detected = os.cpu_count() or 1
    assert default_logical_processor_budget() == max(1, detected // 2)
    assert default_logical_processor_budget() <= detected

    # This case narrows the affinity of the pytest process itself and the bound
    # is one-way by design, so it cannot be undone through the public API. Saved
    # here and restored in the `finally` below through the raw OS call, which is
    # environment hygiene for the rest of the session rather than a widening
    # anything in the product is allowed to do.
    restore_processors = current_allowed_logical_processors()
    restore_env = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }
    try:
        _assert_capacity_ceiling_holds(detected, monkeypatch, runner=path)
    finally:
        if os.name == "nt":
            process_metrics._windows_affinity(sum(1 << i for i in restore_processors))
        else:
            os.sched_setaffinity(0, set(restore_processors))
        for name, value in restore_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    assert current_allowed_logical_processors() == restore_processors


def test_runtime_plan_jointly_bounds_desktop_workers_models_and_duckdb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """requirement: one owner admits CPU/memory and DuckDB honors its receipt."""

    gib = 1024**3
    workstation = RuntimeMachineCapacity(
        detected_logical_processors=32,
        allowed_logical_processor_ids=tuple(range(32)),
        total_memory_bytes=128 * gib,
        available_memory_bytes=100 * gib,
    )
    plan = resolve_runtime_capacity_plan(
        request=ResearchRuntimeRequest(),
        machine=workstation,
        fold_count=5,
        parent_reservation_bytes=6 * gib,
        worker_reservation_bytes=12 * gib,
    )
    assert plan.admission == "ADMITTED"
    assert (plan.fold_workers, plan.lightgbm_threads_per_fit, plan.duckdb_threads) == (
        4,
        1,
        4,
    )
    assert plan.fold_workers * plan.lightgbm_threads_per_fit <= 16
    assert plan.estimated_peak_process_tree_rss_bytes == 54 * gib
    assert plan.process_tree_peak_rss_measurement == "NOT_MEASURED"

    same_policy_different_telemetry = resolve_runtime_capacity_plan(
        request=ResearchRuntimeRequest.model_validate({"capacity": {"maximum_memory_gib": 54.0}}),
        machine=workstation.model_copy(
            update={
                "total_memory_bytes": 100 * gib,
                "available_memory_bytes": 90 * gib,
            }
        ),
        fold_count=5,
        parent_reservation_bytes=6 * gib,
        worker_reservation_bytes=12 * gib,
    )
    fixed_policy = resolve_runtime_capacity_plan(
        request=ResearchRuntimeRequest.model_validate({"capacity": {"maximum_memory_gib": 54.0}}),
        machine=workstation,
        fold_count=5,
        parent_reservation_bytes=6 * gib,
        worker_reservation_bytes=12 * gib,
    )
    assert fixed_policy.admission == same_policy_different_telemetry.admission == "ADMITTED"
    assert fixed_policy.plan_hash == same_policy_different_telemetry.plan_hash
    insufficient_fraction = resolve_runtime_capacity_plan(
        request=ResearchRuntimeRequest.model_validate({"capacity": {"maximum_memory_gib": 54.0}}),
        machine=workstation.model_copy(
            update={
                "total_memory_bytes": 80 * gib,
                "available_memory_bytes": 70 * gib,
            }
        ),
        fold_count=5,
        parent_reservation_bytes=6 * gib,
        worker_reservation_bytes=12 * gib,
    )
    assert insufficient_fraction.admission == "REFUSED"
    assert insufficient_fraction.refusal_code == "process_capacity.memory_not_admitted"
    unavailable = resolve_runtime_capacity_plan(
        request=ResearchRuntimeRequest.model_validate({"capacity": {"maximum_memory_gib": 54.0}}),
        machine=workstation.model_copy(update={"available_memory_bytes": 50 * gib}),
        fold_count=5,
        parent_reservation_bytes=6 * gib,
        worker_reservation_bytes=12 * gib,
    )
    assert unavailable.admission == "REFUSED"
    assert unavailable.refusal_code == "process_capacity.memory_not_admitted"

    low_memory = RuntimeMachineCapacity(
        detected_logical_processors=8,
        allowed_logical_processor_ids=tuple(range(8)),
        total_memory_bytes=16 * gib,
        available_memory_bytes=12 * gib,
    )
    refused = resolve_runtime_capacity_plan(
        request=ResearchRuntimeRequest.model_validate({"profile": "LOW_MEMORY"}),
        machine=low_memory,
        fold_count=5,
        parent_reservation_bytes=6 * gib,
        worker_reservation_bytes=12 * gib,
    )
    assert refused.admission == "REFUSED"
    assert refused.fold_workers == 0
    assert refused.refusal_code == "process_capacity.memory_not_admitted"

    database = WorkspaceDatabase(tmp_path)
    with database.connect(read_only=False) as connection:
        connection.execute("CREATE TABLE observations(value INTEGER)")
        connection.execute("INSERT INTO observations VALUES (1)")
    before = database.path.read_bytes()
    monkeypatch.setattr(reader_threads, "_THREADS", [2])
    with database.connect(read_only=True) as connection:
        assert int(connection.execute("SELECT current_setting('threads')").fetchone()[0]) == 2
        assert connection.execute("SELECT SUM(value) FROM observations").fetchone()[0] == 1
    assert database.path.read_bytes() == before


def test_runtime_plan_is_workload_aware_across_desktop_and_throughput_profiles() -> None:
    """requirement: inactive Alpha workers cannot consume Portfolio capacity."""

    gib = 1024**3
    for processors in (4, 8, 16):
        desktop = RuntimeMachineCapacity(
            detected_logical_processors=processors,
            allowed_logical_processor_ids=tuple(range(processors)),
            total_memory_bytes=16 * gib,
            available_memory_bytes=14 * gib,
        )
        metadata = resolve_runtime_capacity_plan(
            request=ResearchRuntimeRequest(),
            machine=desktop,
            fold_count=5,
            parent_reservation_bytes=6 * gib,
            worker_reservation_bytes=12 * gib,
            workload="METADATA_PREFLIGHT",
        )
        assert metadata.admission == "ADMITTED"
        assert metadata.profile_resolution == "LOW_MEMORY"
        assert (metadata.fold_workers, metadata.lightgbm_threads_per_fit) == (0, 0)
        assert metadata.duckdb_threads <= metadata.process_logical_processor_limit

        alpha = resolve_runtime_capacity_plan(
            request=ResearchRuntimeRequest(),
            machine=desktop,
            fold_count=5,
            parent_reservation_bytes=6 * gib,
            worker_reservation_bytes=12 * gib,
            workload="ALPHA_FOLDS",
        )
        assert alpha.admission == "REFUSED"
        assert alpha.refusal_code == "process_capacity.memory_not_admitted"

        numerical = resolve_runtime_capacity_plan(
            request=ResearchRuntimeRequest(),
            machine=desktop,
            fold_count=5,
            parent_reservation_bytes=6 * gib,
            worker_reservation_bytes=12 * gib,
            workload="PORTFOLIO_NUMERICAL",
        )
        assert numerical.admission == "REFUSED"
        assert (
            numerical.refusal_code == "process_capacity.desktop_numerical_tree_measurement_required"
        )

    for processors in (4, 8, 16):
        throughput = RuntimeMachineCapacity(
            detected_logical_processors=processors,
            allowed_logical_processor_ids=tuple(range(processors)),
            total_memory_bytes=128 * gib,
            available_memory_bytes=120 * gib,
        )
        alpha = resolve_runtime_capacity_plan(
            request=ResearchRuntimeRequest.model_validate(
                {"parallelism": {"fold_workers": 4, "duckdb_threads": 4}}
            ),
            machine=throughput,
            fold_count=5,
            parent_reservation_bytes=6 * gib,
            worker_reservation_bytes=12 * gib,
            workload="ALPHA_FOLDS",
        )
        if alpha.admission == "ADMITTED":
            assert alpha.profile_resolution == "THROUGHPUT"
            assert (
                alpha.fold_workers * alpha.lightgbm_threads_per_fit + alpha.duckdb_threads
                <= alpha.process_logical_processor_limit
            )
        else:
            assert alpha.refusal_code == "process_capacity.processor_plan_not_admitted"

    with pytest.raises(ValueError, match="literal"):
        ResearchRuntimeRequest.model_validate({"parallelism": {"lightgbm_threads_per_fit": 2}})


def test_process_resource_monitor_reports_a_time_aligned_live_child_tree() -> None:
    """requirement: receipts sample parent and live descendants, not lifetime sums."""

    monitor = ProcessResourceMonitor(
        sample_interval_seconds=0.02,
        include_live_descendants=True,
    )
    monitor.start()
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; payload = bytearray(4 * 1024 * 1024); time.sleep(0.35)",
        ]
    )
    try:
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        usage = monitor.finish()

    assert usage.measurement_scope == "PARENT_AND_LIVE_DESCENDANTS"
    assert usage.process_tree_peak_live_descendant_count >= 1
    assert usage.process_tree_peak_rss_bytes >= usage.peak_rss_bytes
    assert usage.sample_interval_seconds == 0.02
    assert usage.instrumentation_limitation is None


def _assert_capacity_ceiling_holds(
    detected: int, monkeypatch: pytest.MonkeyPatch, *, runner: Path
) -> None:
    """The ceiling body, split out so the caller can restore process state."""

    receipt = bind_process_logical_processors()
    assert receipt["enforcement"] == "PROCESS_AFFINITY"
    assert receipt["configured_logical_processors"] == default_logical_processor_budget()
    # Read back from the operating system, not echoed from the request: a bound
    # that was asked for and silently ignored is worse than no bound, because
    # the run would then report a ceiling it is not under.
    assert (
        len(list(receipt["applied_logical_processors"]))
        == (  # type: ignore[call-overload]
            receipt["configured_logical_processors"]
        )
    )
    assert float(receipt["configured_share_of_machine"]) <= 0.5  # type: ignore[arg-type]
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert os.environ[name] == str(receipt["configured_logical_processors"])

    # The hole the review found: `limit` was validated against the machine total
    # rather than against the ceiling, so on a 32-processor host
    # `--logical-processors 32` was accepted and returned the whole machine while
    # the help text promised it could not. Every value above the half-machine
    # budget is refused now, including ones comfortably below `cpu_count`.
    half = default_logical_processor_budget()
    for refused in (detected + 1, detected, half + 1, 0, -1):
        with pytest.raises(ProcessCapacityError, match="logical_processor_budget_invalid"):
            bind_process_logical_processors(refused)

    # Lower is allowed, and lowering is one-way: once this process is narrowed to
    # four, the ceiling is four, and asking for the half-machine budget back is
    # refused rather than quietly widening a limit somebody else imposed.
    narrowed = bind_process_logical_processors(min(4, half))
    assert narrowed["configured_logical_processors"] == min(4, half)
    assert narrowed["logical_processors_already_allowed"] == half
    assert len(current_allowed_logical_processors()) == min(4, half)
    if half > 4:
        with pytest.raises(ProcessCapacityError, match="logical_processor_budget_invalid"):
            bind_process_logical_processors(half)

    # A read-back that disagrees with what was asked for is a refusal, not a
    # shrug: a ceiling that was requested and silently ignored would otherwise be
    # reported as being in force. Simulated at the platform call the code
    # actually reads back through, because patching the initial allowed-set read
    # would leave the read-back untouched and prove nothing.
    with monkeypatch.context() as patched:
        if os.name == "nt":
            patched.setattr(
                process_metrics,
                "_windows_affinity",
                lambda mask: (1 << (os.cpu_count() or 1)) - 1,
            )
        else:
            patched.setattr(os, "sched_setaffinity", lambda pid, cpus: None)
        with pytest.raises(ProcessCapacityError, match="logical_processor_bound_unenforceable"):
            process_metrics.bind_process_logical_processors(1)

    # Applied before any stage dispatches, so no copy, connection or numerical
    # call can precede it.
    source = ast.parse(runner.read_text(encoding="utf-8"))
    body = next(
        node.body
        for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [
        index
        for index, node in enumerate(body)
        if any(
            isinstance(inner, ast.Call)
            and getattr(inner.func, "id", "") == "bind_process_logical_processors"
            for inner in ast.walk(node)
        )
    ]
    dispatch = [
        index
        for index, node in enumerate(body)
        if isinstance(node, ast.If)
        and any(
            isinstance(inner, ast.Call) and getattr(inner.func, "id", "") in {"seed", "build"}
            for inner in ast.walk(node)
        )
    ]
    assert calls and dispatch and max(calls) < min(dispatch)

    # Operational authority, never methodology: the bound must not have reached
    # any identity the catalog quotes.
    assert "logical_processor" not in json.dumps(FeatureCatalog.load().binding.__dict__)


def test_availability_and_execution_are_separate_owners() -> None:
    """requirement: source availability, and entry/exit, are not Formula lag."""

    availability = installed_feature_availability_policy()
    assert availability.publication_delay_sessions == 0
    assert availability.available_after_phase is MarketPhase.OFFICIAL_CLOSE
    # The point-in-time claim belongs to the owner that makes it. The governing
    # policy states the union and delegates; the price feed's own policy is where
    # "a lag is not point-in-time authority" is asserted, and the Sector map's is
    # where the stronger admission -- that it is not point-in-time at all -- is.
    owners = installed_source_availability_catalog()
    assert "point-in-time" in owners.owner(PROVIDER_DAILY_BARS_AUTHORITY).revision_risk
    assert "Not point-in-time" in owners.owner(SECTOR_CLASSIFICATION_AUTHORITY).revision_risk

    sessions = tuple(date(2026, 1, 5) + timedelta(days=index) for index in range(5))

    def selectable(phase: MarketPhase, policy: FeatureAvailabilityPolicy) -> date | None:
        return latest_selectable_observation_session(
            sessions,
            decision_cutoff=DecisionCutoff(session=sessions[3], phase=phase),
            availability=policy,
        )

    # The phase decides a whole session of information, so it is the phase and
    # not the date that answers.
    assert selectable(MarketPhase.OFFICIAL_CLOSE, availability) == sessions[3]
    assert selectable(MarketPhase.POST_CLOSE, availability) == sessions[3]
    assert selectable(MarketPhase.INTRADAY, availability) == sessions[2]
    assert selectable(MarketPhase.PRE_OPEN, availability) == sessions[2]
    with pytest.raises(FeatureAvailabilityError, match="phase_ambiguous"):
        selectable(MarketPhase.UNSPECIFIED, availability)

    delayed = FeatureAvailabilityPolicy.create(
        policy_id="feature-availability.probe-one-session-delay",
        observation_basis="PROBE",
        available_after_phase=MarketPhase.OFFICIAL_CLOSE,
        publication_delay_sessions=1,
        revision_risk="probe only",
    )
    assert selectable(MarketPhase.OFFICIAL_CLOSE, delayed) == sessions[2]
    # A publication delay is a property of the source: it moves the availability
    # policy and leaves every Formula's observation identity untouched.
    assert delayed.policy_hash != availability.policy_hash
    assert FeatureCatalog.load().binding.formula_observation_policy_hash == (
        formula_observation_policy_hash(FeatureCatalog.load().observation_clocks)
    )

    # The execution recipe already supplies the causal isolation a Formula was
    # paying for a second time: formation at close(T), entry at open(T+1).
    recipe = build_one_session_recipe()
    assert recipe.information_cutoff == "FORMATION_OFFICIAL_CLOSE"
    assert (recipe.entry_offset_sessions, recipe.exit_offset_sessions) == (1, 2)
    clock = FeatureCatalog.load().clocks_by_factor["rev_5"]
    assert clock.formula_skip_sessions == 0
    # A clock cannot grow an execution offset: the contract forbids extras, so an
    # entry/exit field cannot be smuggled into Feature identity by a caller.
    with pytest.raises(ValueError):
        FeatureObservationClock.model_validate(
            {**clock.model_dump(mode="json"), "entry_offset_sessions": 1}
        )


def _publish_probe_panel(root: Path, *, catalog: FeatureCatalog) -> str:
    """A well-formed Panel whose identities the installed owners can re-derive."""

    resolver = ArtifactResolver(root)
    registry = default_extension_kernel_registry()
    implementations = catalog_implementation_hashes(catalog, registry=registry)
    methodologies = catalog_methodology_hashes(catalog, registry=registry)
    clocks = catalog.clocks_by_factor
    table = pa.table({"listing_id": ["L0"], "session_date": [date(2026, 1, 5)]})
    chunk_hash = canonical_hash({"probe": "chunk"})
    chunk = resolver.publish_feature_panel_chunk(
        table=table, content_hash=chunk_hash, metadata={"year": "2026"}
    )
    lineage = {
        "catalog_hash": catalog.binding.catalog_hash,
        "manifest_revision": "1" * 64,
        "sector_revision": "2" * 64,
        "spy_revision": "3" * 64,
        "policy_hash": "4" * 64,
        "formula_observation_policy_hash": catalog.binding.formula_observation_policy_hash,
        "source_availability": catalog.source_availability.model_dump(mode="json"),
        "source_authorities": catalog.source_authorities.model_dump(mode="json"),
        "source_authority_binding_hash": catalog.binding.source_authority_binding_hash,
    }
    binding = FeaturePanelBinding.create(
        manifest_revision=lineage["manifest_revision"],
        sector_revision=lineage["sector_revision"],
        catalog_hash=lineage["catalog_hash"],
        spy_revision=lineage["spy_revision"],
        policy_hash=lineage["policy_hash"],
    )
    payload: dict[str, Any] = {
        "kind": "FeaturePanelSnapshotManifest",
        "active_listing_count": 1,
        "listing_set_hash": canonical_hash(("L0",)),
        "panel_binding_hash": binding.panel_binding_hash,
        "panel_content_hash": canonical_hash({"probe": "content"}),
        "chunks": [
            {
                "year": 2026,
                "chunk_hash": chunk_hash,
                "metadata_hash": chunk.metadata_hash,
                "uri": chunk.uri,
                "row_count": 1,
                "first_session": "2026-01-05",
                "last_session": "2026-01-05",
            }
        ],
        "safe_summary": {
            "lineage": lineage,
            "factor_catalog_summary": {
                factor_id: {
                    "implementation_hash": implementations[factor_id],
                    "methodology_hash": methodologies[factor_id],
                    "observation_clock_hash": clocks[factor_id].clock_hash,
                    "source_authority_ids": list(catalog.formula_source_authorities[factor_id]),
                }
                for factor_id in catalog.factor_ids
            },
        },
    }
    snapshot_hash = canonical_hash(payload)
    resolver.publish_feature_panel_manifest(
        payload={**payload, "snapshot_hash": snapshot_hash}, snapshot_hash=snapshot_hash
    )
    index_identity = {
        "kind": "FeaturePanelSemanticIndex",
        "panel_snapshot_hash": snapshot_hash,
        "panel_content_hash": payload["panel_content_hash"],
        "catalog_hash": catalog.binding.catalog_hash,
        "listing_set_hash": payload["listing_set_hash"],
        "active_listing_count": 1,
        "calendar_hash": canonical_hash([date(2026, 1, 5)]),
        "slice_algorithm_identity": "ordered-session-row-hash-digest",
        "sessions": [
            {
                "session_date": "2026-01-05",
                "row_count": 1,
                "ordered_row_hash_digest": canonical_hash(["row"]),
            }
        ],
    }
    resolver.publish_feature_panel_semantic_index(
        payload={**index_identity, "index_hash": canonical_hash(index_identity)},
        index_hash=canonical_hash(index_identity),
    )
    return snapshot_hash


def _retarget_manifest(root: Path, snapshot_hash: str, mutate: Callable[[dict], None]) -> str:
    """Republish one manifest with a forged field under its own honest hash."""

    path = root / "feature-panel" / "manifests" / f"{snapshot_hash}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("snapshot_hash")
    mutate(payload)
    forged = canonical_hash(payload)
    ArtifactResolver(root).publish_feature_panel_manifest(
        payload={**payload, "snapshot_hash": forged}, snapshot_hash=forged
    )
    return forged


def test_recursive_verification_rejects_forged_clock_authority(tmp_path: Path) -> None:
    """regression: a Panel may not answer for its own clock, catalog or index."""

    catalog = FeatureCatalog.load()
    root = tmp_path / "artifacts"
    snapshot_hash = _publish_probe_panel(root, catalog=catalog)
    verifier = FeaturePanelObservationClockVerifier(resolver=ArtifactResolver(root))
    verified = verifier.verify(snapshot_hash)
    assert verified.disposition == "VERIFIED", verified.failure_detail
    assert "per_factor_clock_implementation_and_methodology" in verified.checks_passed
    assert "split_clock_availability_and_source_authority" in verified.checks_passed
    assert verified.verified_layer == "CLOCK_AUTHORITY_RELATIONS_VERIFIED"
    assert (
        verified.formula_observation_policy_hash == catalog.binding.formula_observation_policy_hash
    )
    assert (
        verified.source_availability_policy_hash == catalog.binding.source_availability_policy_hash
    )

    def forge_clock(payload: dict) -> None:
        payload["safe_summary"]["factor_catalog_summary"]["rev_5"]["observation_clock_hash"] = (
            "f" * 64
        )

    def drop_clock(payload: dict) -> None:
        for entry in payload["safe_summary"]["factor_catalog_summary"].values():
            entry.pop("observation_clock_hash")

    def forge_catalog(payload: dict) -> None:
        payload["safe_summary"]["lineage"]["catalog_hash"] = "e" * 64

    def forge_binding(payload: dict) -> None:
        payload["panel_binding_hash"] = "d" * 64

    for mutate, expected in (
        (forge_clock, "feature_panel.factor_authority_mismatch"),
        (drop_clock, "feature_panel.observation_clock_authority_absent"),
        (forge_catalog, "feature_panel.catalog_binding_mismatch"),
        (forge_binding, "feature_panel.panel_binding_mismatch"),
    ):
        forged = _retarget_manifest(root, snapshot_hash, mutate)
        result = verifier.verify(forged)
        assert result.disposition == "REFUSED"
        assert result.failure_code == expected
        # The forged manifest is self-consistent, so only re-derivation catches
        # it; and a forged snapshot has no semantic index of its own either.
        assert result.snapshot_hash == forged


def test_legacy_panel_readback_is_never_promoted_to_successor_authority(tmp_path: Path) -> None:
    """requirement: an old Panel stays readable and cannot satisfy the successor."""

    catalog = FeatureCatalog.load()
    root = tmp_path / "artifacts"
    snapshot_hash = _publish_probe_panel(root, catalog=catalog)
    legacy = _retarget_manifest(
        root,
        snapshot_hash,
        lambda payload: [
            entry.pop("observation_clock_hash")
            for entry in payload["safe_summary"]["factor_catalog_summary"].values()
        ],
    )
    resolver = ArtifactResolver(root)
    readback = resolver.load_feature_panel_manifest(resolver.feature_panel_manifest_uri(legacy))
    assert readback["snapshot_hash"] == legacy
    assert readback["safe_summary"]["factor_catalog_summary"]["rev_5"]["methodology_hash"]
    refused = FeaturePanelObservationClockVerifier(resolver=resolver).verify(legacy)
    assert refused.failure_code == "feature_panel.observation_clock_authority_absent"
    assert refused.checks_passed == ("manifest_payload_hash",)


@pytest.fixture(scope="module")
def consumption() -> dict[str, FormulaConsumptionProbe]:
    """One catalog-wide perturbation pass, shared by the cases that read it."""

    return {item.factor_id: item for item in audit_installed_formula_consumption()}


def test_every_installed_formula_consumes_exactly_what_its_clock_declares(
    consumption: dict[str, FormulaConsumptionProbe],
) -> None:
    """Every installed formula consumes exactly what its clock declares."""

    catalog = FeatureCatalog.load()
    expected = {spec.factor_id for spec in installed_formula_specs(catalog)}
    assert set(consumption) == expected

    failures = {
        factor_id: probe.failures for factor_id, probe in consumption.items() if probe.failures
    }
    assert failures == {}

    # Every Formula answers all four, and exactly one of them answers the
    # calendar variant of the first-finite question.
    for probe in consumption.values():
        assert "LATEST_CONSUMED_EXACT" in probe.checks_passed
        assert "NOTHING_LATER" in probe.checks_passed
        assert "NO_FUTURE_SOURCE" in probe.checks_passed
        assert "NOTHING_EARLIER" in probe.checks_passed
    mechanical = {
        factor_id
        for factor_id, probe in consumption.items()
        if "FIRST_FINITE_MECHANICAL" in probe.checks_passed
    }
    calendar = {
        factor_id
        for factor_id, probe in consumption.items()
        if "FIRST_FINITE_EXISTS" in probe.checks_passed
    }
    assert calendar == {"seasonality_12m"}
    assert mechanical == expected - calendar

    # The calendar Formula's declared row count is nominal, and this asserts that
    # it is *not* treated as a boundary in either direction. Its first finite
    # value lands later than the declaration, which is exactly why claiming "no
    # earlier than 253" would have been a claim the audit never measured.
    seasonality = consumption["seasonality_12m"]
    assert seasonality.minimum_history_is_mechanical is False
    assert seasonality.first_finite_position is not None
    assert seasonality.first_finite_position > seasonality.declared_minimum_history_rows - 1
    assert "FIRST_FINITE_MECHANICAL" not in seasonality.checks_passed


def test_the_measured_skip_partition_is_the_one_the_catalog_declares(
    consumption: dict[str, FormulaConsumptionProbe],
) -> None:
    """regression: an economic skip is preserved and a safety lag is not reinstated."""

    skipped = {
        factor_id
        for factor_id, probe in consumption.items()
        if probe.declared_skip_sessions not in (0, None)
    }
    assert skipped == set(_SKIP_MOMENTUM) | {
        "information_discreteness_252",
        "momentum_consistency_252",
    }
    for factor_id in skipped:
        probe = consumption[factor_id]
        assert probe.declared_skip_sessions == 21
        # Measured, not declared: the newest session that moves the value sits
        # exactly twenty-one sessions back.
        assert probe.measured_latest_consumed_position == probe.observation_position - 21

    for factor_id, probe in consumption.items():
        if factor_id in skipped or probe.source_interval_kind == "CALENDAR_MONTH_SELECTION":
            continue
        assert probe.measured_latest_consumed_position == probe.observation_position

    calendar = consumption["seasonality_12m"]
    assert calendar.declared_skip_sessions is None
    # A calendar selection reads a month twelve back; expressing that as any
    # session offset is the generalisation this branch removed.
    assert calendar.measured_latest_consumed_position is not None
    assert calendar.measured_latest_consumed_position < calendar.observation_position - 200


def test_every_installed_extension_recipe_declares_its_clock() -> None:
    """requirement: a researcher's Formula declares observation semantics once."""

    extensions = extension_factor_specs()
    clocks = {spec.factor_id: observation_clock_for(spec) for spec in extensions}
    assert set(clocks) == {spec.factor_id for spec in extensions}
    assert {clock.formula_skip_sessions for clock in clocks.values()} == {0}
    # None of them carries an availability field at all: that authority belongs to
    # the catalog, once, and a Formula that held a copy would rotate its method
    # identity every time a Provider changed its publication schedule.
    assert all(
        "availability" not in key
        for clock in clocks.values()
        for key in clock.model_dump(mode="json")
    )
    assert {clock.source_interval.kind for clock in clocks.values()} == {
        "CONTIGUOUS_SESSION_WINDOW"
    }
