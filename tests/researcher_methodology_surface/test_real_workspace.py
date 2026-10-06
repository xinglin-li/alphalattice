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
    _GOLDEN_ROOT,
    AS_OF,
    MARKET_PROFILE_ID,
    SYMBOLS,
    GoldenWorkspaceError,
    RealRiskWorkspace,
    _golden_is_complete,
    _golden_key,
    build_real_risk_workspace,
)


@pytest.fixture(scope="module")
def fresh_workspace(tmp_path_factory: pytest.TempPathFactory) -> RealRiskWorkspace:
    """The build itself, driven in this process; the golden cache is bypassed."""

    return build_real_risk_workspace(tmp_path_factory.mktemp("fresh-real-workspace"), fresh=True)


def test_panel_is_admitted_through_the_feature_input_gateway(
    fresh_workspace: RealRiskWorkspace,
) -> None:
    """Without an admission the Panel is unreadable by Factor and Risk.

    A freshly initialized workspace used to be unable to obtain one at all: the
    first cycle skipped governance for want of an active Panel, and every later
    cycle found market data already current. The disclosure is the observable
    end of that fix.
    """

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
    """Quality governance rebinds the sector revision onto its child manifest.

    The revision is bound to manifest identity, so rebinding activates one the
    ledger has never seen. Publishing the map first is what keeps the recovery
    binding resolvable; without it the Panel cannot be published at all.
    """

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
    """A greenfield workspace opens a genesis root, not an admitted one.

    The two roots are separate contracts on purpose: a genesis root structurally
    cannot answer for a Panel snapshot or a retention assessment, so it must
    never be mistaken for admitted history.
    """

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
    """requirement: the Host compares a typed decision event with source availability.

    The envelope already requires ``as_of >= end``, so the resolver's old
    ``min(end, as_of)`` bound was always ``end`` and availability never removed
    anything however early in the day the decision was taken. A phase is what
    makes the comparison real: a daily Feature for session ``T`` exists only
    after ``close(T)``.
    """

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
    fresh_workspace: RealRiskWorkspace, tmp_path
) -> None:
    """requirement: the verifier runs on the real route, before any Desk or numerical call.

    The Panel is republished into a private artifact root with its per-Formula
    observation identities removed -- exactly the shape every pre-successor Panel
    has. It stays readable under its own stored identity and must not resolve.
    """

    import json
    import shutil

    from alphalattice.control.product_host.research_authoring.authority import (
        WorkspaceResearchAuthorityResolver,
    )
    from alphalattice.kernel.shared_kernel.identity import canonical_hash
    from alphalattice.protocols.research_authoring.contracts import AuthoringError

    private = tmp_path / "legacy-shaped"
    shutil.copytree(fresh_workspace.artifact_root, private / "artifacts")
    manifests = private / "artifacts" / "feature-panel" / "manifests"
    payload = json.loads(
        (manifests / f"{fresh_workspace.panel_snapshot_hash}.json").read_text(encoding="utf-8")
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
        workspace=fresh_workspace.workspace, artifact_root=private / "artifacts"
    )
    with pytest.raises(AuthoringError, match="panel_observation_clock_absent"):
        resolver.resolve(_envelope(fresh_workspace, snapshot_handle=forged))

    # The same resolver admits the untouched Panel from the same private root,
    # so the refusal is about the missing clock authority and not about the copy.
    assert resolver.resolve(_envelope(fresh_workspace)).panel_snapshot_hash == (
        fresh_workspace.panel_snapshot_hash
    )


def test_a_second_consumer_restores_the_golden_workspace_with_the_same_identities(
    fresh_workspace: RealRiskWorkspace, tmp_path: Path
) -> None:
    """requirement: the cache hands every consumer the build's identities, through product readers.

    The module's workspace was built fresh in this process; the golden copy for
    the same key is a restore. Their manifest, panel, surface and sector map
    must agree, no half-built directory or lock may remain, and the key must
    move with the universe it was built for.
    """

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
    golden = _GOLDEN_ROOT / key
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


def test_a_golden_that_disagrees_with_its_sidecar_is_refused_not_repaired(
    fresh_workspace: RealRiskWorkspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement: a complete golden whose identities do not match is a refusal, never a rebuild.

    The sidecar is what makes a golden complete, and it is bound to its key; a
    complete golden whose build environment, manifest, panel or sector state
    disagree with the consumer is corrupt or foreign, and silently rebuilding
    it would hide that.
    """

    key = _golden_key(fresh_workspace.feature_catalog, fresh_workspace.feature_kernels, SYMBOLS)
    private_root = tmp_path / "goldens"
    golden = private_root / key
    shutil.copytree(_GOLDEN_ROOT / key, golden)
    monkeypatch.setattr(
        "tests.researcher_methodology_surface.real_workspace._GOLDEN_ROOT", private_root
    )
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
    """Raw qualification alone is not U0: one known-bad sector remains a candidate."""
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


def test_cache_environment_tracks_thread_settings_in_the_current_process(monkeypatch):
    from tests.researcher_methodology_surface import real_workspace as cache

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    first = cache._environment_digest()
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert cache._environment_digest() != first
