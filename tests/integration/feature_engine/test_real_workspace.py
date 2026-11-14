"""The workspace a Risk experiment resolves against is written by the product.

Authority resolution is only worth proving against artifacts the product wrote
itself, so this case asserts the properties that make the fixture workspace a
real one: a closure ledger opened at genesis, a Panel manifest and a return
surface resolved through the same resolver and reader production uses. The
Gateway's admission and the ledger's sector map are held by the Feature tests,
which kill every mutant the cases that asserted them here killed.

Workspace-writing cases drive onboarding, ten years of materialization and
maintenance fresh; those assertions share a module build. Other cases exercise
cache/lease metadata without a build. These are synthetic-provider tests, not
the separately selected real-evidence lane. Daily development selects the needed
cases explicitly rather than repeating this whole suite.
"""

from __future__ import annotations

import hashlib
import json
import os
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
        (
            (_Path(__file__).resolve().parents[3] / "tests/researcher_methodology_surface")
            / "fixtures"
            / "risk_covariance_development.yaml"
        ).read_text(encoding="utf-8")
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
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    """The host admits a successor panel only after its observation close."""

    from alphalattice.control.product_host.research_authoring.authority import (
        WorkspaceResearchAuthorityResolver,
    )
    from alphalattice.protocols.research_authoring.contracts import AuthoringError

    resolver = WorkspaceResearchAuthorityResolver(workspace=real_risk_workspace.workspace)
    at_close = resolver.resolve(_envelope(real_risk_workspace))
    final = at_close.sessions[-1]

    before_close = resolver.resolve(
        _envelope(real_risk_workspace, as_of={"session": final, "phase": "INTRADAY"})
    )
    assert before_close.sessions[-1] < final
    assert final not in before_close.sessions

    post = resolver.resolve(
        _envelope(real_risk_workspace, as_of={"session": final, "phase": "POST_CLOSE"})
    )
    assert post.sessions[-1] == final

    # A date alone cannot say which of those three it was, so it is refused
    # rather than promoted to the most permissive reading.
    with pytest.raises(AuthoringError, match="decision_cutoff_phase_ambiguous"):
        resolver.resolve(_envelope(real_risk_workspace, as_of=final))


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


@pytest.mark.private_data
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


@pytest.mark.private_data
def test_a_golden_that_disagrees_with_its_sidecar_is_refused_not_repaired(
    fresh_workspace: RealRiskWorkspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A foreign scientific sidecar identity is refused."""

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

    # Payload seal mismatches and incomplete generations are covered by the small seal case below.


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
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import nullcontext
    from types import SimpleNamespace

    from tests import conftest as hooks
    from tests.researcher_methodology_surface import session_workspace as session

    # Collection warms the selected seven-listing variant, and preserves the default in mixed runs.
    members = SYMBOLS[:7]
    member = SimpleNamespace(
        fixturenames=["stopped_change"], module=SimpleNamespace(MEMBERS=members)
    )
    ordinary = SimpleNamespace(fixturenames=[], module=SimpleNamespace())
    external = SimpleNamespace(fixturenames=["external_factor_workspace"], module=SimpleNamespace())
    readiness = SimpleNamespace(fixturenames=["revision_seed"], module=SimpleNamespace())
    with monkeypatch.context() as patch:
        calls = []
        fake = SimpleNamespace(
            _golden_root=lambda: tmp_path / "warmup",
            warm_real_risk_golden=lambda **kw: calls.append(kw),
            READINESS_UNIVERSE=(members, 6),
        )
        patch.setitem(sys.modules, "tests.researcher_methodology_surface.real_workspace", fake)
        patch.setattr(hooks.environment, "project_scope", lambda *_: nullcontext())
        config = SimpleNamespace(option=SimpleNamespace(collectonly=False))
        for items, expected in (
            ([member], [{"symbols": members, "sector_size": 7}]),
            ([member, member], [{"symbols": members, "sector_size": 7}]),
            ([member, ordinary], [{"external": False}, {"symbols": members, "sector_size": 7}]),
            ([member, external], [{"external": True}, {"symbols": members, "sector_size": 7}]),
            ([ordinary], [{"external": False}]),
            ([external], [{"external": True}]),
            ([readiness], [{"external": False}, {"symbols": members, "sector_size": 6}]),
        ):
            calls.clear()
            hooks.pytest_collection_finish(SimpleNamespace(config=config, items=items))
            assert calls == expected

    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    factory, attempts, value = SimpleNamespace(getbasetemp=lambda: tmp_path), [], ["original"]
    make, copy = session.session_workspace, session.copy_workspace
    monkeypatch.setattr(session, "LINKED_SEEDS", set())  # this holder's seeds, not the session's

    def build(root):
        root.mkdir()
        (root / "seed.txt").write_text(value[0], encoding="utf-8")
        (root / "pool.duckdb").write_bytes(b"db")  # a linked input, as the product binds one
        (root / "source.duckdb").hardlink_to(root / "pool.duckdb")
        (root / "market-data.duckdb").write_bytes(b"market data")
        digest = hashlib.sha256(b"sealed arrays").hexdigest()
        for category in ("frozen-observation-arrays", "lifecycle-arrays", "unregistered-arrays"):
            folder = root / "artifacts/alpha-research/current" / category
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{digest}.bin").write_bytes(b"sealed arrays")
        (root / "artifacts/alpha-research/current/lifecycle-arrays/not-a-digest.bin").write_bytes(
            b"other"
        )
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
    copied = copy(first, tmp_path / "first-copy")
    assert (copied / "seed.txt").read_text() == "original"
    # The product's link is kept inside the copy, and never reaches back to the seed.
    pool = copied / "pool.duckdb"
    assert pool.samefile(copied / "source.duckdb") and not pool.samefile(first / "pool.duckdb")
    assert (second / "seed.txt").read_text() == "second"
    # Read in place: the seed's own file, refusing a writer through any link; the copy still
    # goes whole (on Windows too), the seed still refusing writers, and a change is named.
    in_place = copy(second, tmp_path / "in-place", link_data=True)
    linked = in_place / "market-data.duckdb"
    assert linked.samefile(second / "market-data.duckdb") and session.seeds_changed() == []
    with pytest.raises(PermissionError):
        linked.open("r+b")
    digest = hashlib.sha256(b"sealed arrays").hexdigest()
    for category in ("frozen-observation-arrays", "lifecycle-arrays"):
        relative = Path("artifacts/alpha-research/current") / category / f"{digest}.bin"
        assert (in_place / relative).samefile(second / relative)
        with pytest.raises(PermissionError):
            (in_place / relative).open("r+b")
        nested = tmp_path / (category + ".bin")
        session.keep_seed_links(str(in_place / relative), str(nested))
        assert nested.samefile(second / relative)
    for relative in (
        Path("artifacts/alpha-research/current/unregistered-arrays") / f"{digest}.bin",
        Path("artifacts/alpha-research/current/lifecycle-arrays/not-a-digest.bin"),
    ):
        assert not (in_place / relative).samefile(second / relative)
    moved = in_place.rename(tmp_path / "in-place-moved")  # a folder of sealed links still moves
    shutil.rmtree(moved)
    with pytest.raises(PermissionError):
        (second / "market-data.duckdb").open("r+b")
    (second / "seed.txt").write_text("changed", encoding="utf-8")
    assert session.seeds_changed() == [f"{second}: seed.txt"]
    (second / "seed.txt").write_text("second", encoding="utf-8")  # the seed is reused below
    session.LINKED_SEEDS.clear()
    with pytest.raises(RuntimeError, match="private_copy_unmarked"):  # unmarked: no private copy
        copy(second, tmp_path / "private", link_data=False)
    with monkeypatch.context() as private_test:
        private_test.setitem(session.RUNNING_TEST, "private", True)
        private = copy(second, tmp_path / "private-arrays", link_data=False)
        relative = Path("artifacts/alpha-research/current/lifecycle-arrays") / f"{digest}.bin"
        assert not (private / relative).samefile(second / relative)
        (private / relative).write_bytes(b"private change")
        assert (second / relative).read_bytes() == b"sealed arrays"

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

    # A durable seed is built once for a source snapshot, then verified across sessions.
    import inspect

    assert "persistent" in inspect.signature(make).parameters, (
        "session seed has no persistent golden owner"
    )
    from devtools.evaluation.golden_workspaces import GoldenInputs

    shared = tmp_path / "persistent-golden"
    monkeypatch.setenv("ALPHALATTICE_TEST_GOLDEN_ROOT", str(shared))
    source = tmp_path / "seed-input.txt"
    source.write_bytes(b"recipe-one")
    snapshot = GoldenInputs(
        tmp_path, (source.name,), {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
    )
    built_count = []

    def build_persistent(root):
        root.mkdir()
        (root / "value.txt").write_bytes(b"immutable")
        (root / "market-data.duckdb").write_bytes(b"data")
        built_count.append(1)
        return {"value": "verified"}

    first_base, second_base = tmp_path / "session-one", tmp_path / "session-two"
    first_base.mkdir()
    second_base.mkdir()
    a = SimpleNamespace(getbasetemp=lambda: first_base)
    b = SimpleNamespace(getbasetemp=lambda: second_base)
    from devtools.evaluation.golden_workspaces import golden_run
    from tests.pytest_environment import golden_runtime

    with golden_run(first_base / "owned-run", runtime=golden_runtime()):
        durable, metadata = make(a, "durable_seed", build_persistent, persistent=snapshot)
        again, same = make(b, "durable_seed", build_persistent, persistent=snapshot)
        assert durable == again and metadata == same == {"value": "verified"} and built_count == [1]
        assert not (durable / "value.txt").stat().st_mode & 0o200
        with pytest.raises(PermissionError):
            (durable / "value.txt").open("r+b")
        with monkeypatch.context() as private_test:
            private_test.setitem(session.RUNNING_TEST, "private", True)
            private = copy(durable, tmp_path / "durable-private", link_data=False)
        (private / "value.txt").write_text("private")
        assert (durable / "value.txt").read_bytes() == b"immutable"
        payload = durable.parent / "payload.json"
        payload.chmod(0o600)
        payload.write_text("{}")
        with pytest.raises(RuntimeError, match="payload verification refused"):
            make(b, "durable_seed", build_persistent, persistent=snapshot)
        try:
            rebuilt, rebuilt_metadata = make(
                b, "durable_seed", build_persistent, persistent=snapshot
            )
        except RuntimeError as error:
            pytest.fail(f"a refused persistent seed was not rebuilt: {error}")
        assert rebuilt != durable and rebuilt_metadata == metadata and built_count == [1, 1]
        assert payload.read_bytes() == b"{}"  # the refused generation was not mutated
        durable = rebuilt
        payload = durable.parent / "payload.json"
        read_bytes = Path.read_bytes

        def unreadable(path):
            if path == payload:
                raise PermissionError("unreadable")
            return read_bytes(path)

        with monkeypatch.context() as denied:
            denied.setattr(Path, "read_bytes", unreadable)
            with pytest.raises(RuntimeError, match="verification unmeasured"):
                make(b, "durable_seed", build_persistent, persistent=snapshot)
        assert make(b, "durable_seed", build_persistent, persistent=snapshot)[0] == durable
        assert built_count == [1, 1]  # unknown never causes replacement
        marker = durable.parent / "complete.json"
        marker.chmod(0o600)
        with pytest.raises(RuntimeError, match="published marker writable"):
            make(b, "durable_seed", build_persistent, persistent=snapshot)
        recovered, _ = make(b, "durable_seed", build_persistent, persistent=snapshot)
        assert recovered != durable and marker.stat().st_mode & 0o200
        usage = shared / "session-prerequisites/durable_seed.last-use.json"
        last_use = json.loads(usage.read_bytes())
        last_use[durable.parent.name] = 0
        usage.write_text(json.dumps(last_use), encoding="utf-8")
        make(b, "durable_seed", build_persistent, inputs="other", persistent=snapshot)
        assert durable.exists()  # the live invocation still leases the refused generation
        source.write_bytes(b"changed")
        with pytest.raises(RuntimeError, match="inputs changed"):
            make(b, "durable_seed", build_persistent, persistent=snapshot)
        source.write_bytes(b"recipe-one")

    last_use = json.loads(usage.read_bytes())
    assert last_use[durable.parent.name] > 0  # release recorded its actual last use
    last_use[durable.parent.name] = 0  # drive the three-day boundary, never sleep
    usage.write_text(json.dumps(last_use), encoding="utf-8")
    with golden_run(second_base / "owned-run", runtime=golden_runtime()):
        make(b, "durable_seed", build_persistent, inputs="later", persistent=snapshot)
        assert not durable.exists()  # the last reader released its operating-system lease
        assert (private / "value.txt").read_text() == "private"

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


@pytest.mark.parametrize("case", ["digest", "session-seed", "seal", "real-link"])
def test_cache_environment_tracks_thread_settings_in_the_current_process(
    monkeypatch, tmp_path, tmp_path_factory, case
):
    from types import SimpleNamespace

    from tests.researcher_methodology_surface import real_workspace as cache

    if case == "real-link":
        from tests.researcher_methodology_surface import session_workspace as session

        record = cache.build_real_risk_workspace(tmp_path / "real-linked")
        database = record.workspace / "market-data.duckdb"
        matches = [
            seed
            for seed in session.LINKED_SEEDS
            if (seed / database.name).is_file() and database.samefile(seed / database.name)
        ]
        golden = cache._golden_root() / cache._golden_key(
            record.feature_catalog, record.feature_kernels, cache.SYMBOLS
        )
        assert len(matches) == 1 and not database.samefile(golden / "workspace" / database.name)
        assert record.return_reader.available_sessions(record.return_surface)
        return

    if case == "seal":
        key, builds = "a" * 24, []
        golden = tmp_path / "golden"
        verify = getattr(cache, "verify_golden_seal", None)
        assert callable(verify), "a persistent golden must verify its recorded seal"

        def build():
            builds.append(1)
            workspace = golden / "workspace"
            workspace.mkdir(parents=True)
            payload = b"market"
            (workspace / "market-data.duckdb").write_bytes(payload)
            files = {"market-data.duckdb": hashlib.sha256(payload).hexdigest()}
            seal = hashlib.sha256(
                json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            (golden / "golden.json").write_text(
                json.dumps({"format": 2, "key": key, "files": files, "seal_hash": seal}),
                encoding="utf-8",
            )

        assert verify(golden, key, build)[0] is False
        assert verify(golden, key, build)[0] is True and len(builds) == 1
        scan = os.scandir

        def denied(path):
            if Path(path) == golden / "workspace":
                raise PermissionError("unreadable")
            return scan(path)

        with monkeypatch.context() as unavailable:
            unavailable.setattr(os, "scandir", denied)
            with pytest.raises(GoldenWorkspaceError, match="cache unreadable"):
                verify(golden, key, build)
        assert (
            len(builds) == 1 and (golden / "workspace/market-data.duckdb").read_bytes() == b"market"
        )
        (golden / "workspace/market-data.duckdb").write_bytes(b"corrupt")
        assert verify(golden, key, build)[0] is False and len(builds) == 2
        shutil.rmtree(golden / "workspace")
        assert verify(golden, key, build)[0] is False and len(builds) == 3
        marker = golden / "golden.json"
        saved = marker.read_bytes()
        marker.write_text("{", encoding="utf-8")
        with pytest.raises(GoldenWorkspaceError, match="invalid published cache"):
            verify(golden, key, build)
        marker.write_bytes(saved)
        (golden / "workspace/market-data.duckdb").write_bytes(b"changed")

        def stopped():
            raise OSError("stopped")

        with pytest.raises(GoldenWorkspaceError, match="rebuild failed; inspect"):
            verify(golden, key, stopped)
        retained = list(tmp_path.glob("golden.mismatch-*"))
        assert (
            len(retained) == 1
            and (retained[0] / "workspace/market-data.duckdb").read_bytes() == b"changed"
        )

        seed = tmp_path / "seed"
        seed.mkdir()
        data = seed / "market-data.duckdb"
        data.write_bytes(b"market")
        target = tmp_path / "linked.duckdb"
        os.link(data, target)
        cache.track_test_seed(
            "fixture-reader", seed, {"market-data.duckdb": hashlib.sha256(b"market").hexdigest()}
        )
        target.write_bytes(b"changed")
        with pytest.raises(
            AssertionError, match=r"test_datasets\.write_through.*market-data\.duckdb"
        ) as failure:
            cache.verify_test_seed("fixture-reader")
        assert "fixture-reader" in str(failure.value)
        return

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
    """Checkouts share one golden root through the setting; without it each keeps its own.
    A pipeline's warm-up builds into that shared root, confined to it as a session's build."""
    from devtools.evaluation.golden_workspaces import GOLDEN_ROOT_VARIABLE, persistent_golden_root
    from tests import pytest_environment as environment
    from tests.researcher_methodology_surface import real_workspace as builder

    monkeypatch.delenv(GOLDEN_ROOT_VARIABLE, raising=False)
    assert persistent_golden_root(tmp_path / "own") == tmp_path / "own"
    monkeypatch.setenv(GOLDEN_ROOT_VARIABLE, str(tmp_path / "shared"))
    assert persistent_golden_root(tmp_path / "own") == tmp_path / "shared"
    scopes = []
    monkeypatch.setattr(
        builder, "warm_real_risk_golden", lambda: scopes.append(environment.PROJECT_ROOT.get())
    )
    builder.warm_before_tests()
    assert scopes == [(tmp_path / "shared").resolve()] and (tmp_path / "shared").is_dir()
    # The warm-up and a pytest session (src on its path) share that golden only with one key; a
    # separate installation of the same version is still another key.
    import importlib.metadata
    import sys
    from types import SimpleNamespace

    from tests.researcher_methodology_surface.real_workspace import _environment_digest

    checkout, prefix = tmp_path / "checkout", tmp_path / "checkout/.venv"
    url = '{"url": "' + checkout.resolve().as_uri() + '", "dir_info": {"editable": true}}'

    def record(where, direct=None, metadata_name="alphalattice.egg-info"):
        return SimpleNamespace(
            metadata={"Name": "alphalattice"},
            version="0.2.4",
            locate_file=lambda _: where,
            read_text=lambda _: direct,
            _path=where / metadata_name,
        )

    installed = record(prefix / "Lib/site-packages", url)
    with monkeypatch.context() as path:
        path.setattr(sys, "prefix", str(prefix))
        path.setattr(importlib.metadata, "distributions", lambda: [installed])
        once = _environment_digest()
        path.setattr(
            importlib.metadata, "distributions", lambda: [installed, record(checkout / "src")]
        )
        aliased = _environment_digest()  # the checkout's own metadata, as a pytest session sees it
        path.setattr(
            importlib.metadata, "distributions", lambda: [installed, record(tmp_path / "other")]
        )
        separate = _environment_digest()  # another installation of the same version
        path.setattr(
            importlib.metadata, "distributions", lambda: [installed, record(checkout / "vendor")]
        )
        nested = _environment_digest()  # a separate installation inside the checkout
        path.setattr(
            importlib.metadata,
            "distributions",
            lambda: [
                installed,
                record(checkout / "src", metadata_name="alphalattice-0.2.4.dist-info"),
            ],
        )
        source_install = _environment_digest()  # a separate wheel, not the checkout's egg-info
    assert once == aliased and once != separate == nested == source_install
