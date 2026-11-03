"""The workspace a Risk experiment resolves against is written by the product.

Authority resolution is only worth proving against artifacts the product wrote
itself, so this case asserts the properties that make the fixture workspace a
real one: a Panel admitted through the Feature Input Gateway, a closure ledger
that answers for the revision actually activated, and a return surface readable
through the same reader production uses.

Workspace-writing cases drive onboarding, ten years of materialization and
maintenance fresh; those assertions share a module build. Other cases exercise
cache/lease metadata without a build. These are synthetic-provider tests, not
the separately selected real-evidence lane. Daily development selects the needed
cases explicitly rather than repeating this whole suite.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_contracts import (
    FeatureBaseClosureGenesisRoot,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    FeatureClosureLedger,
)
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from tests.researcher_methodology_surface.real_workspace import (
    AS_OF,
    MARKET_PROFILE_ID,
    SYMBOLS,
    GoldenWorkspaceError,
    RealRiskWorkspace,
    _golden_is_complete,
    _golden_key,
    _golden_root,
    build_real_risk_workspace,
    copy_fresh_workspace,
    session_fresh_seed,
)


@pytest.fixture
def fresh_workspace(tmp_path_factory: pytest.TempPathFactory, tmp_path: Path) -> RealRiskWorkspace:
    """A private copy of the product-written, session-fresh build."""

    return copy_fresh_workspace(tmp_path_factory, tmp_path / "fresh-real-workspace")


def test_panel_is_admitted_through_the_feature_input_gateway(
    fresh_workspace: RealRiskWorkspace,
) -> None:
    """Panel is admitted through the feature input gateway."""

    store = MarketDataRepository(fresh_workspace.workspace)
    panel_state = PanelStateRepository(store.database, market_data=store)
    disclosure = panel_state.feature_input_quality_disclosure(
        result_manifest_revision=fresh_workspace.manifest.revision_sha256
    )

    assert disclosure["gateway_qualified"] is True
    assert disclosure["admitted_listing_count"] == len(SYMBOLS)
    assert disclosure["quality_exclusion_count"] == 0
    assert disclosure["quality_admission_hash"]


def test_closure_ledger_answers_for_the_activated_sector_revision(
    fresh_workspace: RealRiskWorkspace,
) -> None:
    """Closure ledger answers for the activated sector revision."""

    store = MarketDataRepository(fresh_workspace.workspace)
    feature_state = FeatureStateRepository(store.database, market_data=store)
    ledger = FeatureClosureLedger(
        PanelClosureArtifactStore(ArtifactResolver(fresh_workspace.artifact_root))
    )
    sector = feature_state.current_sector_state(fresh_workspace.manifest)
    assert sector is not None

    binding = ledger.panel_binding(fresh_workspace.panel_snapshot_hash)
    assert binding is not None
    head = ledger.require_head(binding.catalog_hash)
    sector_map = ledger.sector_map_for_head(head=head, sector_revision=sector.sector_revision)

    assert sector_map.sector_revision == sector.sector_revision
    assert sector_map.manifest_revision == fresh_workspace.manifest.revision_sha256
    assert {entry.listing_id for entry in sector_map.entries} == set(
        fresh_workspace.sector_by_listing_id
    )


def test_closure_opened_at_genesis_and_never_claimed_admission(
    fresh_workspace: RealRiskWorkspace,
) -> None:
    """The closure ledger opens at genesis without claiming feature admission."""

    ledger = FeatureClosureLedger(
        PanelClosureArtifactStore(ArtifactResolver(fresh_workspace.artifact_root))
    )
    binding = ledger.panel_binding(fresh_workspace.panel_snapshot_hash)
    assert binding is not None
    head = ledger.require_head(binding.catalog_hash)

    assert ledger.is_genesis_root(head.root_hash)
    assert isinstance(ledger.closure_root_for_head(head), FeatureBaseClosureGenesisRoot)
    with pytest.raises(ValueError, match=r"feature_closure\.genesis_root_not_admissible"):
        ledger.root_for_head(head)


def test_return_surface_reads_back_through_the_production_reader(
    fresh_workspace: RealRiskWorkspace,
) -> None:
    """The surface a Risk experiment consumes resolves by hash, not by fixture."""

    surface = fresh_workspace.return_surface
    sessions = fresh_workspace.return_reader.available_sessions(surface)

    assert surface.formation_count == len(sessions)
    assert sessions[-1] <= AS_OF
    assert surface.surface_hash
    # A flat provider price would leave covariance degenerate and prove nothing,
    # so the seeded walk has to survive all the way to the read side.
    returns = fresh_workspace.return_reader.read_sessions(surface, sessions[-64:])
    assert returns.shape[0] == 64
    assert float(returns.std()) > 0.0


def test_panel_manifest_resolves_through_the_same_resolver_production_uses(
    fresh_workspace: RealRiskWorkspace,
) -> None:
    """No handle here is synthetic; the reference is a real content-addressed URI."""

    resolver = ArtifactResolver(fresh_workspace.artifact_root)
    payload = resolver.load_feature_panel_manifest(fresh_workspace.panel_manifest_ref)

    assert payload["snapshot_hash"] == fresh_workspace.panel_snapshot_hash
    assert payload["as_of_session"] == AS_OF.isoformat()
    assert payload["active_listing_count"] == len(SYMBOLS)
    assert payload["chunks"]

    store = MarketDataRepository(fresh_workspace.workspace)
    panel_state = PanelStateRepository(store.database, market_data=store)
    active = panel_state.active_feature_panel(MARKET_PROFILE_ID)
    assert active is not None
    assert str(active["manifest_revision"]) == fresh_workspace.manifest.revision_sha256
    assert str(active["panel_content_hash"]) == payload["panel_content_hash"]


def _envelope(
    fresh_workspace: RealRiskWorkspace,
    *,
    snapshot_handle: str | None = None,
    **session_overrides: object,
):
    """The shipped Risk document, retargeted at the fixture workspace's own axis."""

    from pathlib import Path as _Path

    from alphalattice.control.research_program.authoring.document import load_authoring_document
    from alphalattice.protocols.research_authoring.contracts import ResearchExperimentEnvelope

    document = load_authoring_document(
        (_Path(__file__).parent / "fixtures" / "risk_covariance_development.yaml").read_text(
            encoding="utf-8"
        )
    )
    experiment = dict(document["experiment"])
    request: dict[str, object] = {
        "start": experiment["sessions"]["start"],
        "end": AS_OF,
        "as_of": {"session": AS_OF, "phase": "OFFICIAL_CLOSE"},
    }
    request.update(session_overrides)
    experiment["sessions"] = request
    experiment["universe_handle"] = MARKET_PROFILE_ID
    experiment["data_snapshot_handle"] = snapshot_handle or fresh_workspace.panel_snapshot_hash
    return ResearchExperimentEnvelope.create(**experiment)


def test_the_host_admits_a_successor_panel_only_after_its_observation_close(
    fresh_workspace: RealRiskWorkspace,
) -> None:
    """The host admits a successor panel only after its observation close."""

    from alphalattice.control.product_host.research_authoring.authority import (
        WorkspaceResearchAuthorityResolver,
    )
    from alphalattice.protocols.research_authoring.contracts import AuthoringError

    resolver = WorkspaceResearchAuthorityResolver(workspace=fresh_workspace.workspace)
    at_close = resolver.resolve(_envelope(fresh_workspace))
    final = at_close.sessions[-1]

    before_close = resolver.resolve(
        _envelope(fresh_workspace, as_of={"session": final, "phase": "INTRADAY"})
    )
    assert before_close.sessions[-1] < final
    assert final not in before_close.sessions

    post = resolver.resolve(
        _envelope(fresh_workspace, as_of={"session": final, "phase": "POST_CLOSE"})
    )
    assert post.sessions[-1] == final

    # A date alone cannot say which of those three it was, so it is refused
    # rather than promoted to the most permissive reading.
    with pytest.raises(AuthoringError, match="decision_cutoff_phase_ambiguous"):
        resolver.resolve(_envelope(fresh_workspace, as_of=final))


def test_the_host_refuses_a_panel_without_feature_clock_authority(
    real_risk_workspace: RealRiskWorkspace, tmp_path
) -> None:
    """The host refuses a panel without feature clock authority."""

    from alphalattice.control.product_host.research_authoring.authority import (
        WorkspaceResearchAuthorityResolver,
    )
    from alphalattice.kernel.shared_kernel.identity import canonical_hash
    from alphalattice.protocols.research_authoring.contracts import AuthoringError

    private = tmp_path / "legacy-shaped"
    shutil.copytree(real_risk_workspace.artifact_root, private / "artifacts")
    manifests = private / "artifacts" / "feature-panel" / "manifests"
    payload = json.loads(
        (manifests / f"{real_risk_workspace.panel_snapshot_hash}.json").read_text(encoding="utf-8")
    )
    payload.pop("snapshot_hash")
    for entry in payload["safe_summary"]["factor_catalog_summary"].values():
        entry.pop("observation_clock_hash", None)
    forged = canonical_hash(payload)
    (manifests / f"{forged}.json").write_text(
        json.dumps(
            {**payload, "snapshot_hash": forged},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
        encoding="utf-8",
    )
    resolver = WorkspaceResearchAuthorityResolver(
        workspace=real_risk_workspace.workspace, artifact_root=private / "artifacts"
    )
    with pytest.raises(AuthoringError, match="panel_observation_clock_absent"):
        resolver.resolve(_envelope(real_risk_workspace, snapshot_handle=forged))

    # The same resolver admits the untouched Panel from the same private root,
    # so the refusal is about the missing clock authority and not about the copy.
    assert resolver.resolve(_envelope(real_risk_workspace)).panel_snapshot_hash == (
        real_risk_workspace.panel_snapshot_hash
    )


def test_a_second_consumer_restores_the_golden_workspace_with_the_same_identities(
    fresh_workspace: RealRiskWorkspace, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A second consumer restores the golden workspace with the same identities."""

    again = build_real_risk_workspace(tmp_path / "again")
    assert again.workspace != fresh_workspace.workspace
    assert again.manifest.revision_sha256 == fresh_workspace.manifest.revision_sha256
    assert again.panel_snapshot_hash == fresh_workspace.panel_snapshot_hash
    assert again.panel_manifest_ref == fresh_workspace.panel_manifest_ref
    assert again.return_surface.surface_hash == fresh_workspace.return_surface.surface_hash
    assert again.sector_by_listing_id == fresh_workspace.sector_by_listing_id
    assert again.return_reader.available_sessions(again.return_surface) == (
        fresh_workspace.return_reader.available_sessions(fresh_workspace.return_surface)
    )
    key = _golden_key(again.feature_catalog, again.feature_kernels, SYMBOLS)
    golden = _golden_root() / key
    assert (golden / "golden.json").is_file()
    assert not golden.with_name(golden.name + ".building").exists()
    assert not golden.with_name(golden.name + ".lock").exists()
    assert _golden_key(again.feature_catalog, again.feature_kernels, ("QA000", "QA001")) != key
    sibling = build_real_risk_workspace(tmp_path / "sibling")

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    seed = golden / "workspace/market-data.duckdb"
    original = digest(seed)
    with (again.workspace / "market-data.duckdb").open("ab") as stream:
        stream.write(b"private consumer mutation")
    assert digest(seed) == original
    assert digest(sibling.workspace / "market-data.duckdb") == original
    session_seed, _ = session_fresh_seed(tmp_path_factory)
    session_db = session_seed / "workspace/market-data.duckdb"
    session_original = digest(session_db)
    fresh_sibling = copy_fresh_workspace(tmp_path_factory, tmp_path / "fresh-sibling")
    with (fresh_workspace.workspace / "market-data.duckdb").open("ab") as stream:
        stream.write(b"fresh consumer mutation")
    assert digest(session_db) == session_original
    assert digest(fresh_sibling.workspace / "market-data.duckdb") == session_original
    assert digest(seed) == original
    assert digest(sibling.workspace / "market-data.duckdb") == original


def test_a_golden_that_disagrees_with_its_sidecar_is_refused_not_repaired(
    fresh_workspace: RealRiskWorkspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A golden that disagrees with its sidecar is refused not repaired."""

    key = _golden_key(fresh_workspace.feature_catalog, fresh_workspace.feature_kernels, SYMBOLS)
    build_real_risk_workspace(tmp_path / "published")  # publishes the golden when absent
    private_root = tmp_path / "goldens"
    golden = private_root / key
    shutil.copytree(_golden_root() / key, golden)
    monkeypatch.setenv("ALPHALATTICE_TEST_GOLDEN_ROOT", str(private_root))
    assert _golden_is_complete(golden, key)

    sidecar_path = golden / "golden.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    forged = dict(sidecar, manifest_revision="0" * 64)
    sidecar_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(GoldenWorkspaceError, match="manifest revision"):
        build_real_risk_workspace(tmp_path / "revision")

    forged = dict(sidecar, panel_snapshot_hash="1" * 64)
    sidecar_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(GoldenWorkspaceError, match="panel"):
        build_real_risk_workspace(tmp_path / "panel")

    sector_by_listing_id = dict(sidecar["sector_by_listing_id"])
    listing_id = next(iter(sector_by_listing_id))
    sector_by_listing_id[listing_id] = "FORGED-SECTOR"
    forged = dict(sidecar, sector_by_listing_id=sector_by_listing_id)
    sidecar_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(GoldenWorkspaceError, match="sector map"):
        build_real_risk_workspace(tmp_path / "sector")

    for field in ("source_tree", "environment"):
        forged = dict(sidecar, **{field: "foreign"})
        sidecar_path.write_text(json.dumps(forged), encoding="utf-8")
        with pytest.raises(GoldenWorkspaceError, match=field):
            build_real_risk_workspace(tmp_path / field)

    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    database = golden / "workspace/market-data.duckdb"
    size = database.stat().st_size
    try:
        with database.open("ab") as stream:
            stream.write(b"tampered cache payload")
        with pytest.raises(GoldenWorkspaceError, match="cached file content changed"):
            build_real_risk_workspace(tmp_path / "tampered-file")
    finally:
        with database.open("r+b") as stream:
            stream.truncate(size)

    # An incomplete *published* root is corrupt, not permission to rebuild it.
    assert not _golden_is_complete(golden, "0" * 24)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    shutil.rmtree(golden / "workspace")
    assert not _golden_is_complete(golden, key)
    with pytest.raises(GoldenWorkspaceError, match="invalid published cache"):
        build_real_risk_workspace(tmp_path / "missing-files")


def test_the_golden_key_binds_the_interpreter_and_the_installed_distributions(
    fresh_workspace: RealRiskWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement: the same source tree under another interpreter or numpy is another key."""

    key = _golden_key(fresh_workspace.feature_catalog, fresh_workspace.feature_kernels, SYMBOLS)
    assert (
        _golden_key(
            fresh_workspace.feature_catalog, fresh_workspace.feature_kernels, SYMBOLS, sector_size=6
        )
        != key
    )
    monkeypatch.setattr(
        "tests.researcher_methodology_surface.real_workspace._environment_digest",
        lambda: "another interpreter",
    )
    assert (
        _golden_key(fresh_workspace.feature_catalog, fresh_workspace.feature_kernels, SYMBOLS)
        != key
    )


def test_cache_dependencies_follow_build_imports_resources_and_identity_members(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from tests.researcher_methodology_surface import session_workspace as session

    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    factory, attempts, value = SimpleNamespace(getbasetemp=lambda: tmp_path), [], ["original"]
    make, copy = session.session_workspace, session.copy_workspace

    def build(root):
        root.mkdir()
        (root / "seed.txt").write_text(value[0], encoding="utf-8")
        attempts.append(value[0])
        if len(attempts) == 1:
            raise InterruptedError("worker stopped mid-build")
        return {"value": value[0]}

    with pytest.raises(InterruptedError, match="mid-build"):
        make(factory, "seed", build)
    try:
        first, original = make(factory, "seed", build)
    except RuntimeError as error:
        pytest.fail(f"the next worker could not rebuild an incomplete seed: {error}")
    assert original == {"value": "original"} and len(attempts) == 2
    assert make(factory, "seed", build) == (first, original) and len(attempts) == 2
    value[0] = "second"
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(make, factory, "seed", build, inputs="second")
        second, observed = future.result(timeout=10)
    assert (first / "seed.txt").read_text() == "original"
    assert first != second and observed == {"value": "second"}
    assert (copy(first, tmp_path / "first-copy") / "seed.txt").read_text() == "original"
    assert (second / "seed.txt").read_text() == "second"

    value[0] = "original"
    for field in ("name", "builder", "inputs", "", "missing"):
        marker, prior = first.parent / "complete.json", first
        record = json.loads(marker.read_text(encoding="utf-8"))
        if field == "missing":
            marker.unlink()
        else:
            marker.write_text(json.dumps(record | {field: "other"}) if field else "{")
        first, _ = make(factory, "seed", build)
        assert first != prior and (prior / "seed.txt").read_text() == "original"
    assert len(attempts) == 8 and make(factory, "seed", build, inputs="second") == (
        second,
        observed,
    )
    (first / "seed.txt").write_text("corrupted", encoding="utf-8")
    with pytest.raises(RuntimeError, match="session workspace was changed"):
        make(factory, "seed", build)
    index = tmp_path / "research-prerequisites" / "seed.json"
    index.write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeError, match="session workspace index unreadable"):
        make(factory, "seed", build)
    assert index.read_text(encoding="utf-8") == "{"
    with pytest.raises(ValueError):
        make(factory, "invalid", build, inputs=float("nan"))
    assert not (tmp_path / "research-prerequisites" / "invalid").exists()

    from tests.researcher_methodology_surface import real_workspace as cache

    def write(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        return path

    builder = write("tests/builder.py", "from alphalattice.foundation.inputs.build import run\n")
    data = write(
        "src/alphalattice/foundation/inputs/build.py",
        'SOURCE = "src/alphalattice/kernel/math/identity.py"\ndef run(): pass\n',
    )
    init = write("src/alphalattice/foundation/inputs/__init__.py", "VALUE = 1\n")
    identity = write("src/alphalattice/kernel/math/identity.py", "VALUE = 2\n")
    resource = write("src/alphalattice/foundation/inputs/resources/catalog.json", "{}")
    config = write("config/data-policy.json", "{}")
    ui = write("src/alphalattice/interface/local_application/assets/app.css", "body{}")
    page = write("src/alphalattice/interface/local_application/page.py", "VALUE = 0\n")
    baseline = write("config/internal-ownership-baseline.json", "{}")
    strict = write("config/mypy-strict-modules.txt", "one")
    write("pyproject.toml", "[project]\n")
    write("uv.lock", "version=1\n")
    monkeypatch.setattr(cache, "ROOT", tmp_path)
    monkeypatch.setattr(cache, "_BUILDER_MODULES", (builder,))

    def digest():
        cache._source_files.cache_clear()
        cache._source_tree_digest.cache_clear()
        return cache._source_tree_digest()

    try:
        original = digest()
        for path in (ui, baseline, strict):
            path.write_text("changed unrelated presentation/verification", encoding="utf-8")
        page.write_text("VALUE = 1\n", encoding="utf-8")
        assert digest() == original
        for path in (builder, data, init, identity, resource, config):
            before = path.read_text(encoding="utf-8")
            path.write_text(
                before + ("\n# changed\n" if path.suffix == ".py" else " "), encoding="utf-8"
            )
            assert digest() != original, path
            path.write_text(before, encoding="utf-8")
        extra = write("src/alphalattice/foundation/inputs/extra.py", "VALUE = 3\n")
        init.write_text("from .extra import VALUE\n", encoding="utf-8")
        assert extra in cache._source_files.__wrapped__()
    finally:
        cache._source_files.cache_clear()
        cache._source_tree_digest.cache_clear()


def test_a_cold_cache_build_refuses_an_unbound_source_read(tmp_path, monkeypatch):
    from tests.researcher_methodology_surface import real_workspace as cache

    source = tmp_path / "src/alphalattice/unbound.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(cache, "ROOT", tmp_path)
    monkeypatch.setattr(cache, "_source_files", lambda: ())
    with (
        pytest.raises(GoldenWorkspaceError, match="undeclared build source reads"),
        cache._checked_build_sources(),
    ):
        source.read_bytes()
    assert cache._BUILD_SOURCE_READS is None


def test_cache_wait_expiry_does_not_steal_an_active_lease(tmp_path, monkeypatch):
    from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease
    from tests.researcher_methodology_surface import real_workspace as cache

    key = "a" * 24
    root = tmp_path / ".locks" / key
    lease = WorkspaceWriterLease.acquire(root)
    monkeypatch.setattr(cache, "_GOLDEN_WAIT_SECONDS", 0)
    try:
        with (
            pytest.raises(GoldenWorkspaceError, match="builder busy"),
            cache._golden_lease(tmp_path / key),
        ):
            pytest.fail("a live builder's lease was stolen")
        assert lease.held and root.is_dir()
    finally:
        lease.close()
    with cache._golden_lease(tmp_path / key):
        assert root.is_dir()


def test_initial_universe_requires_computable_base_features_without_extra_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw qualification alone does not finalize the initial cohort:
    one known-bad sector remains a candidate.
    """
    from tests.researcher_methodology_surface.real_workspace import SeededWalkProvider

    # TTT has the same seeded walk as SPY and hence undefined idiosyncratic
    # skew. Keep this fixture focused on the five deliberately zero-volume names.
    symbols = tuple(chr(ord("A") + index) * 3 for index in range(21) if index != 19)
    fetch = SeededWalkProvider.fetch_daily

    def with_zero_volume(self, requested, *, start, end):
        data = fetch(self, requested, start=start, end=end)
        return {
            symbol: tuple(
                {**row, "volume": 0.0}
                if symbol in symbols[:5] and row["session_date"] == AS_OF.isoformat()
                else row
                for row in rows
            )
            for symbol, rows in data.items()
        }

    monkeypatch.setattr(SeededWalkProvider, "fetch_daily", with_zero_volume)
    source = build_real_risk_workspace(
        tmp_path / "initial-qualification", symbols=symbols, fresh=True
    )
    market = MarketDataRepository(source.workspace)
    bootstrap = market.universe_bootstrap(MARKET_PROFILE_ID)
    assert bootstrap is not None
    assert tuple(item.symbol for item in source.manifest.listings) == symbols[5:]
    assert set(bootstrap.cohort_listing_ids) == {
        item.listing_id for item in source.manifest.listings
    }
    with market._connect(read_only=True) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM current_universe_onboarding_listing "
                "WHERE state = 'FEATURE_READY'"
            ).fetchone()[0]
            == 20
        )
        admissions = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT admission_json FROM feature_input_admission"
            ).fetchall()
        ]
    baseline = [item for item in admissions if item.get("evidence_scope") == "BASE_FEATURES_ONLY"]
    assert baseline and len(baseline[0]["admitted_listing_ids"]) == 15
    assert len(baseline[0]["quarantined_listing_ids"]) == 5
    assert baseline[0]["feature_qualification_hash"]
    assert (
        PanelStateRepository(market.database, market_data=market).feature_input_quality_disclosure(
            result_manifest_revision=source.manifest.revision_sha256
        )["gateway_qualified"]
        is True
    )


@pytest.mark.parametrize("case", ["digest", "session-seed"])
def test_cache_environment_tracks_thread_settings_in_the_current_process(
    monkeypatch, tmp_path, case
):
    from types import SimpleNamespace

    from tests.researcher_methodology_surface import real_workspace as cache

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    first = cache._environment_digest()
    if case == "session-seed":

        def build(root, *, fresh):
            assert fresh
            root.mkdir()
            return SimpleNamespace(
                workspace=root,
                feature_catalog=cache.FeatureCatalog.load(),
                feature_kernels=cache.default_extension_kernel_registry(),
                manifest=SimpleNamespace(revision_sha256="fixture"),
                panel_manifest_ref="fixture",
                panel_snapshot_hash="fixture",
                return_surface=SimpleNamespace(surface_hash="fixture"),
                sector_by_listing_id={},
            )

        monkeypatch.setattr(cache, "build_real_risk_workspace", build)
        factory = SimpleNamespace(getbasetemp=lambda: tmp_path)
        seed, metadata = cache.session_fresh_seed(factory)
        assert metadata["environment"] == first
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert cache._environment_digest() != first
    if case == "session-seed":
        changed_seed, changed = cache.session_fresh_seed(factory)
        assert changed_seed != seed and changed["environment"] == cache._environment_digest()


@pytest.mark.proves("evaluation.persistent_golden_root_follows_its_setting")
def test_the_persistent_golden_root_follows_its_shared_setting(tmp_path, monkeypatch):
    """Checkouts share one golden root through the setting; without it each keeps its own."""
    from devtools.evaluation.golden_workspaces import GOLDEN_ROOT_VARIABLE, persistent_golden_root

    monkeypatch.delenv(GOLDEN_ROOT_VARIABLE, raising=False)
    assert persistent_golden_root(tmp_path / "own") == tmp_path / "own"
    monkeypatch.setenv(GOLDEN_ROOT_VARIABLE, str(tmp_path / "shared"))
    assert persistent_golden_root(tmp_path / "own") == tmp_path / "shared"
