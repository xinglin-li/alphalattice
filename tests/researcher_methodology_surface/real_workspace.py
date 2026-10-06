"""Build a small but genuine workspace with the product's own writers.

Authority resolution is only worth proving against artifacts that the product
actually wrote. Every step below is a production writer: universe onboarding,
the composed workspace runtime, the feature foundation, the panel snapshot
publisher, and the causal risk return publisher. The single fixture is the
provider, which is the network edge, and it returns a seeded random walk rather
than a flat price so covariance is non-degenerate.

Nothing here hand-writes an artifact or mocks a handle, so the same
``ArtifactResolver``, panel manifest, and universe manifest that production
resolves are the ones a test resolves.
"""

from __future__ import annotations

import ast
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import cache
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import numpy as np

from alphalattice.control.data_platform.maintenance.contracts import (
    MaintenanceStatus,
    MaintenanceTrigger,
    WorkspaceMaintenanceRequest,
)
from alphalattice.control.data_platform.readiness import (
    WorkspaceConsentAction,
    WorkspaceReadinessConsent,
    WorkspaceReadinessGate,
)
from alphalattice.control.product_host.composition.workspace import WorkspaceRuntime
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease
from alphalattice.foundation.causal_outcomes.execution.publication import (
    CausalExecutionOutcomePublisher,
)
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.inputs.closure_source import (
    FeatureClosureSourceRepository,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
)
from alphalattice.foundation.feature_engine.producers.factors.open_intraday import (
    overnight_return_factor_spec,
)
from alphalattice.foundation.feature_engine.producers.factors.registry import FeatureKernelRegistry
from alphalattice.foundation.feature_engine.runtime.closure_genesis import (
    FeatureClosureGenesisService,
)
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.runtime.universe_onboarding import (
    CurrentUniverseOnboardingStatus,
)
from alphalattice.foundation.market_data_ops.sources.contracts import ProviderAdjustedClosePoint
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest
from alphalattice.foundation.market_data_ops.sources.providers import (
    ProviderFetchError,
    SectorObservation,
)
from alphalattice.foundation.market_data_ops.sources.universe import (
    ORIGINAL_RESEARCH_WHITELIST_STANDARD,
    CandidateMembershipEvidence,
    CurrentUniverseBootstrap,
    CurrentUniverseCandidate,
    CurrentUniverseCandidateManifest,
    candidate_manifest_document,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.investment.risk_research.contracts import CausalRiskReturnSurface
from alphalattice.investment.risk_research.surfaces.returns import (
    CausalRiskReturnReader,
    CausalRiskReturnSurfacePublisher,
    RiskReturnArtifactStore,
)
from alphalattice.kernel.data.calendar import materialize_calendar_schedule
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from devtools.architecture.structural import (
    dependency_closure,
    discover_module_dependencies,
    discover_python_imports,
    discover_python_modules,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "config" / "market-profiles" / "us-current-index-research.yaml"
MARKET_PROFILE_ID = "us-current-index-research"

# The readiness gate applies its own ten-year initial history policy, so the
# provider must be able to answer over that whole window: a narrower fixture
# fails every listing on coverage and onboarding blocks with zero feature-ready
# listings. The panel's own floors (316 index sessions, 274 observations for the
# widest factor) are cleared many times over by this range.
HISTORY_START = date(2016, 7, 31)
AS_OF = date(2026, 7, 31)
OBSERVED_AT = datetime(2026, 8, 2, tzinfo=UTC)

# Two sectors of five, which is the smallest shape that satisfies the
# five-member sector floor on both sides of the sector-balanced weighting.
SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ")


@cache
def _intraday_offsets(seed: int, ordinal: int) -> tuple[float, float, float]:
    """Immutable synthetic values shared by raw and adjusted-history reads.

    Cache only the three floats: each fetch still creates its own writable rows,
    and corrections to a provider's volumes or sectors remain independent.
    """

    rng = np.random.default_rng((seed, ordinal))
    above = float(rng.uniform(0.002, 0.020))
    below = float(rng.uniform(0.002, 0.020))
    opening = float(rng.uniform(-below * 0.8, above * 0.8))
    return opening, above, below


class SeededWalkProvider:
    """Deterministic non-degenerate prices, plus the sector capability."""

    name = "yfinance"

    def __init__(
        self, symbols: Sequence[str], sessions: Sequence[date], *, sector_size: int = 5
    ) -> None:
        if sector_size < 1:
            raise ValueError("fixture sector size must be positive")
        self._symbols = tuple(symbols)
        self.sessions = tuple(sessions)
        self.sectors = {
            symbol: f"Sector-{index // sector_size}" for index, symbol in enumerate(self._symbols)
        }
        self.sectors["SPY"] = "Reference"

    @staticmethod
    def _seed(symbol: str) -> int:
        return sum(ord(item) for item in symbol)

    def _closes(self, symbol: str, end: date) -> dict[date, float]:
        # Build from the full history so a narrower refresh window never moves
        # a bar that was already published.
        rng = np.random.default_rng(self._seed(symbol))
        history = tuple(session for session in self.sessions if session <= end)
        prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.012, len(history)))
        return dict(zip(history, (float(value) for value in prices), strict=True))

    def _intraday(self, symbol: str, session: date) -> tuple[float, float, float]:
        """Asymmetric open/high/low offsets, drawn per session.

        A separate per-session generator keeps a bar stable no matter how wide
        the requested window is, which one shared stream over the window would
        not. The asymmetry matters: with ``high`` and ``low`` equidistant from
        ``close``, Chaikin's money-flow multiplier is identically zero for every
        listing and ``cmf_21`` is rejected as ``feature_mad_zero``.
        """

        return _intraday_offsets(self._seed(symbol), session.toordinal())

    def fetch_daily(self, symbols: Sequence[str], *, start: date, end: date):  # type: ignore[no-untyped-def]
        result = {}
        for symbol in symbols:
            closes = self._closes(symbol, end)
            seed = self._seed(symbol)
            selected = tuple(session for session in self.sessions if start <= session <= end)
            rows = []
            for index, session in enumerate(selected):
                close = closes[session]
                opening, above, below = self._intraday(symbol, session)
                rows.append(
                    {
                        "session_date": session.isoformat(),
                        "open": close * (1.0 + opening),
                        "high": close * (1.0 + above),
                        "low": close * (1.0 - below),
                        "close": close,
                        # Volume must differ across listings on a session, or the
                        # cross-sectional MAD collapses and every volume factor
                        # is rejected as feature_mad_zero.
                        "volume": 1_000_000 + seed + ((index * (seed % 97 + 3)) % 100_000),
                        "split_ratio": 0.0,
                        "cash_dividend": 0.0,
                        "capital_gain": 0.0,
                    }
                )
            result[symbol] = tuple(rows)
        return result

    def fetch_action_history(
        self, *, listing_id: str, provider_symbol: str, start: date, end: date
    ):  # type: ignore[no-untyped-def]
        del listing_id, provider_symbol, start, end
        return ()

    def fetch_adjusted_close_history(  # type: ignore[no-untyped-def]
        self, *, listing_id: str, provider_symbol: str, start: date, end: date
    ):
        payload = self.fetch_daily((provider_symbol,), start=start, end=end)[provider_symbol]
        return tuple(
            ProviderAdjustedClosePoint(
                listing_id=listing_id,
                provider=self.name,
                session_date=date.fromisoformat(str(item["session_date"])),
                adjusted_close=float(item["close"]),
            )
            for item in payload
        )

    def fetch_current_sector(self, *, provider_symbol: str) -> SectorObservation:
        if provider_symbol not in self.sectors:
            raise ProviderFetchError(
                "sector.missing_current_sector", "fixture missing sector", retryable=False
            )
        sector = self.sectors[provider_symbol]
        return SectorObservation(
            provider=self.name,
            provider_symbol=provider_symbol,
            sector_name=sector,
            sector_key=sector.casefold(),
            payload_hash=(sector.encode().hex() * 16)[:64],
        )


def _bootstrap(symbols: Sequence[str]) -> CurrentUniverseBootstrap:
    candidates = tuple(
        CurrentUniverseCandidate(
            symbol=symbol,
            provider_symbol=symbol,
            source_memberships=(
                CandidateMembershipEvidence(index="NASDAQ100", company_name=symbol),
            ),
        )
        for symbol in symbols
    )
    provisional = CurrentUniverseCandidateManifest(
        created_at=OBSERVED_AT,
        as_of_timestamp=OBSERVED_AT,
        construction_rule=ORIGINAL_RESEARCH_WHITELIST_STANDARD.construction_rule,
        data_validity_class=ORIGINAL_RESEARCH_WHITELIST_STANDARD.data_validity_class,
        sources=(),
        candidates=candidates,
        content_hash="",
    )
    document = candidate_manifest_document(provisional)
    document.pop("content_hash")
    return CurrentUniverseBootstrap(
        source_manifest=CurrentUniverseCandidateManifest(
            created_at=provisional.created_at,
            as_of_timestamp=provisional.as_of_timestamp,
            construction_rule=provisional.construction_rule,
            data_validity_class=provisional.data_validity_class,
            sources=provisional.sources,
            candidates=provisional.candidates,
            content_hash=sha256(
                json.dumps(
                    document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        ),
        standard=ORIGINAL_RESEARCH_WHITELIST_STANDARD,
    )


def _require_current_manifest(store: MarketDataRepository) -> UniverseManifest:
    """The manifest the workspace currently answers for, never a cached one."""

    manifest = store.current_quality_filtered_research_manifest(market_profile_id=MARKET_PROFILE_ID)
    if manifest is None:
        raise AssertionError("workspace published no quality-filtered research manifest")
    return manifest


class _MovingClock:
    """The injected wall clock for everything the coordinator composes."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _source_loader_for(symbols: Sequence[str]):  # type: ignore[no-untyped-def]
    """Stand in for the three-index web fetch, and only for that, over one universe."""

    def load(**_kwargs: object) -> CurrentUniverseBootstrap:
        return _bootstrap(symbols)

    return load


def publish_causal_outcomes(
    workspace: RealRiskWorkspace, *, at: datetime = OBSERVED_AT, recipe_id: str | None = None
) -> tuple[str, str]:
    """Publish the causal execution outcomes a Factor program reads, for real.

    Kept out of ``build_real_risk_workspace`` deliberately. Risk needs a return
    surface and never touches execution outcomes, so publishing them in the
    shared builder would slow every Risk case for an artifact none of them reads.

    Returns ``(snapshot_hash, manifest_ref)`` because a Factor program binds both
    and the reader resolves a manifest by hash rather than by URI -- the two are
    handed back together so a caller cannot pair a hash with someone else's ref.
    """

    store = MarketDataRepository(workspace.workspace)
    published = CausalExecutionOutcomePublisher(
        # Annotated ``PanelStateRepository`` on the publisher, but every method it
        # reaches for -- the research manifest, the source watermark, raw bars and
        # actions -- belongs to the market store. The annotation is stale; the
        # market repository is what the code actually requires.
        store=store,  # type: ignore[arg-type]
        resolver=ArtifactResolver(workspace.artifact_root),
        artifact_root=workspace.artifact_root,
        mutation_gate=WorkspaceMutationGate(),
    ).publish_daily(
        panel_manifest_ref=workspace.panel_manifest_ref,
        completed_at=at,
        **({} if recipe_id is None else {"recipe_id": recipe_id}),
    )
    return str(published.manifest.snapshot_hash), str(published.manifest_artifact.uri)


def development_feature_catalog() -> FeatureCatalog:
    """The shipped catalog plus the one persistent extension under test.

    This is the whole of "a researcher installs a method": take the governed
    catalog, add the typed recipe the Feature capability already declares, and
    hand the revision to the writers. The mathematics, the recipe and its
    implementation identity are all product-owned -- ``overnight_return_factor_spec``
    lives beside its kernel in ``feature_engine/producers``. Nothing is invented
    here, and a spec assembled here instead would make "add a method in the
    domain directory" false.

    ``from_payload`` requires the factor axis sorted, so the merge sorts rather
    than appends: the axis is positional in everything downstream that quotes it.

    The shipped ``desktop-feature-catalog.json`` is not written to. This revision
    exists only in the workspace that installs it, which is why no published
    manifest and no current pointer moves.
    """

    shipped = FeatureCatalog.load()
    payload = shipped.to_payload()
    payload["factors"] = sorted(
        [*payload["factors"], overnight_return_factor_spec().model_dump(mode="json")],
        key=lambda factor: str(factor["factor_id"]),
    )
    return FeatureCatalog.from_payload(payload)


@dataclass(frozen=True, slots=True)
class RealRiskWorkspace:
    """Everything a Risk experiment needs, all of it product-written."""

    workspace: Path
    artifact_root: Path
    feature_catalog: FeatureCatalog
    """The catalog revision every writer in this workspace was composed with."""

    feature_kernels: FeatureKernelRegistry
    """The kernel set that travels with it.

    Carried for the same reason the catalog is: a reader that re-derives what the
    writers published needs the owners they were composed with, and the Host's
    authority resolver now verifies a Panel's clock authority before resolving it.
    """

    manifest: UniverseManifest
    panel_manifest_ref: str
    panel_snapshot_hash: str
    return_surface: CausalRiskReturnSurface
    return_reader: CausalRiskReturnReader
    sector_by_listing_id: dict[str, str]

    def freshness_probe(self, *, through: date) -> str:
        """Recompute the surface-scope watermark exactly as the publisher does.

        The same projection the Host installs -- ``watermark_hash`` from the
        market store -- so tests exercise the real causal check rather than a
        stand-in that would agree with anything.
        """

        store = MarketDataRepository(self.workspace)
        return str(
            store.execution_source_watermark(self.manifest, through=through)["watermark_hash"]
        )


def build_real_risk_workspace(
    root: Path,
    *,
    feature_catalog: FeatureCatalog | None = None,
    feature_kernels: FeatureKernelRegistry | None = None,
    symbols: Sequence[str] = SYMBOLS,
    fresh: bool = False,
    sector_size: int = 5,
) -> RealRiskWorkspace:
    """Drive onboarding, features, the panel, and the return surface for real.

    ``feature_catalog`` installs one catalog revision across every writer this
    builds -- genesis, the materializer, the persistence factor axis and the
    panel publisher. It defaults to the shipped catalog, so the ordinary path is
    unchanged; passing a revision is how a development workspace materializes an
    extension factor without the shipped catalog, or any panel binding that
    quotes its hash, moving at all.

    ``feature_kernels`` is the other half of that installation and travels with
    it. A catalog revision naming a ``formula_ref`` no registry resolves fails
    closed in the materializer, so a caller installing a formula the product does
    not own supplies both or neither. It is the seam an external consumer uses:
    the product's own registry stays exactly what it was, and nothing about the
    extra kernel reaches a shipped catalog, a specification catalog or a pointer.

    Threading them here rather than at each call site is the point: the writers
    disagreeing about which factors exist is exactly the failure this prevents.

    ``symbols`` is the universe the seeded walk serves; the shipped ten names by
    default, the eighty-name product QA universe for the suites that need it.

    The build runs once per source tree: its result is materialized under
    ``tmp/golden-workspaces/<key>`` and every caller receives a private copy,
    restored through the product's own readers. The key binds the builder's
    import/source-identity/resource closure, runtime configuration, lock and
    fixture modules plus the catalog, kernel and
    universe identities, so a stale copy cannot survive a source change; a
    gate that runs four processes needing the same workspace builds it once.

    ``fresh`` bypasses the cache and drives the writers into ``root`` directly.
    It is for the one suite whose claim is the build itself: a workspace the
    product wrote is only proved real by writing it, so that suite pays the
    build every run and the routed lane keeps executing the onboarding,
    maintenance and publication path once per gate.
    """

    installed_catalog = feature_catalog if feature_catalog is not None else FeatureCatalog.load()
    installed_kernels = (
        feature_kernels if feature_kernels is not None else default_extension_kernel_registry()
    )
    if fresh:
        return _build_workspace(
            root / "workspace", installed_catalog, installed_kernels, symbols, sector_size
        )
    key = _golden_key(installed_catalog, installed_kernels, symbols, sector_size)
    golden = _GOLDEN_ROOT / key
    workspace = root / "workspace"
    started = time.perf_counter()
    with _golden_lease(golden):
        hit = _golden_is_complete(golden, key)
        if golden.exists() and not hit:
            raise GoldenWorkspaceError(f"golden workspace {key}: invalid published cache")
        if not hit:
            _materialize_golden(
                golden, key, installed_catalog, installed_kernels, symbols, sector_size
            )
        sidecar = _load_sidecar(golden)
        if sidecar is None:
            raise GoldenWorkspaceError(f"golden workspace {key}: no sidecar after materialization")
        shutil.copytree(golden / "workspace", workspace)
        (golden / "golden.json").touch()
    print(
        f"PLAYPEN_WORKSPACE_CACHE {'HIT' if hit else 'BUILD'} {key} "
        f"{time.perf_counter() - started:.3f}s",
        flush=True,
    )
    return _restore(workspace, installed_catalog, installed_kernels, symbols, key, sidecar)


_GOLDEN_ROOT = ROOT / "tmp" / "golden-workspaces"
_GOLDEN_FORMAT = 2
_GOLDEN_WAIT_SECONDS = 900.0
_GOLDEN_KEEP_DAYS = 3
_BUILDER_MODULES = (Path(__file__), Path(__file__).with_name("external_factor_method.py"))


class GoldenWorkspaceError(RuntimeError):
    """A complete golden workspace that does not match what its consumer asked for."""


def _environment_digest() -> str:
    """The interpreter, the platform and every installed distribution a build ran under.

    The source tree and the lock say what *should* be installed; this says what
    is, so a golden built under one interpreter or one numpy is never restored
    under another.
    """

    distributions = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    )
    identity = {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "distributions": distributions,
        "threads": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@cache
def _source_files() -> tuple[Path, ...]:
    """Bind the actual builder, not unrelated UI/Agent code or verification counters.

    Imports use the existing structural scanner. Initializers execute too;
    literal source-identity members are data dependencies even without imports.
    Package resources and runtime config are conservatively included. No
    production identity or source closure is changed by this test-cache key.
    """
    src = ROOT / "src"
    paths = tuple(sorted(src.rglob("*.py")))
    modules = {name: path for path, name in discover_python_modules(paths, source_root=src)}
    edges = discover_module_dependencies(paths, source_root=src)
    files = set(_BUILDER_MODULES)
    pending = list(files)
    seeds: set[str] = set()
    while pending:
        path = pending.pop()
        for name in discover_python_imports(path, source_root=ROOT):
            if name in modules:
                seeds.add(name)
            elif name.startswith("tests."):
                candidate = ROOT / Path(*name.split("."))
                candidate = (
                    candidate.with_suffix(".py")
                    if candidate.with_suffix(".py").is_file()
                    else candidate / "__init__.py"
                )
                if candidate.is_file() and candidate not in files:
                    files.add(candidate)
                    pending.append(candidate)
    while True:
        previous = set(files)
        reached = dependency_closure(edges, seeds=tuple(sorted(seeds)))
        files.update(modules[name] for name in reached)
        for path in tuple(files):
            if not path.is_relative_to(src):
                continue
            for parent in path.parents:
                if parent == src:
                    break
                init = parent / "__init__.py"
                if init.is_file():
                    files.add(init)
                    seeds.update(
                        name
                        for name in discover_python_imports(init, source_root=src)
                        if name in modules
                    )
        if files == previous:
            break
    # These paths are explicitly hashed by scientific/serialization owners.
    # They need not execute to be load-bearing for the artifacts being built.
    for path in tuple(files):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            value = node.value
            if value.startswith(("src/", "config/")) and "\n" not in value:
                target = ROOT / value
                if target.is_file():
                    files.add(target)
    packages = {
        ROOT / "src/alphalattice" / Path(*path.relative_to(ROOT / "src/alphalattice").parts[:2])
        for path in files
        if path.is_relative_to(ROOT / "src/alphalattice")
        and len(path.relative_to(ROOT / "src/alphalattice").parts) >= 3
    }
    files.update(
        path
        for package in packages
        for path in package.rglob("*")
        if path.is_file() and path.suffix not in {".py", ".pyc"} and "__pycache__" not in path.parts
    )
    verification_only = {"internal-ownership-baseline.json", "mypy-strict-modules.txt"}
    files.update(
        path
        for path in (ROOT / "config").rglob("*")
        if path.is_file() and path.name not in verification_only
    )
    files.update((ROOT / "uv.lock", ROOT / "pyproject.toml"))
    return tuple(sorted(files))


@cache
def _source_tree_digest() -> str:
    digest = hashlib.sha256()
    for path in _source_files():
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _workspace_files(workspace: Path) -> dict[str, str]:
    """Content verification of an immutable test seed, not a scientific identity."""
    files = sorted(path for path in workspace.rglob("*") if path.is_file())
    hashes = {}
    for path in files:
        with path.open("rb") as stream:
            hashes[path.relative_to(workspace).as_posix()] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return hashes


_BUILD_SOURCE_READS: set[Path] | None = None
_SOURCE_AUDIT_INSTALLED = False


def _source_read_audit(event, arguments):
    if event != "open" or _BUILD_SOURCE_READS is None or not isinstance(arguments[0], (str, bytes)):
        return
    path = Path(os.fsdecode(arguments[0])).resolve()
    if not (path.is_relative_to(ROOT / "src") or path.is_relative_to(ROOT / "config")):
        return
    if path.suffix == ".pyc":
        path = Path(importlib.util.source_from_cache(str(path)))
    _BUILD_SOURCE_READS.add(path)


@contextmanager
def _checked_build_sources():
    """A real cold build may not publish a cache with unbound owned-source reads."""
    global _BUILD_SOURCE_READS, _SOURCE_AUDIT_INSTALLED
    if not _SOURCE_AUDIT_INSTALLED:
        sys.addaudithook(_source_read_audit)
        _SOURCE_AUDIT_INSTALLED = True
    if _BUILD_SOURCE_READS is not None:
        raise GoldenWorkspaceError("nested cache build is not supported")
    declared = set(_source_files())
    observed: set[Path] = set()
    _BUILD_SOURCE_READS = observed
    try:
        yield
    finally:
        _BUILD_SOURCE_READS = None
    missing = observed - declared
    if missing:
        raise GoldenWorkspaceError(
            "undeclared build source reads: "
            + ", ".join(str(p.relative_to(ROOT)) for p in sorted(missing))
        )


def _golden_key(
    catalog: FeatureCatalog,
    kernels: FeatureKernelRegistry,
    symbols: Sequence[str],
    sector_size: int = 5,
) -> str:
    identity = {
        "format": _GOLDEN_FORMAT,
        "source_tree": _source_tree_digest(),
        "environment": _environment_digest(),
        "catalog_hash": catalog.binding.catalog_hash,
        "kernels_hash": kernels.installed_capability_hash,
        "symbols": list(symbols),
        "sector_size": sector_size,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]


def _load_sidecar(golden: Path) -> dict[str, object] | None:
    try:
        payload = json.loads((golden / "golden.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _golden_is_complete(golden: Path, key: str) -> bool:
    """Complete means published: a sidecar written last, bound to this key, over a workspace.

    Anything less is never restored. The caller builds an absent key and
    refuses an incomplete published root.
    """

    sidecar = _load_sidecar(golden)
    return (
        sidecar is not None
        and sidecar.get("format") == _GOLDEN_FORMAT
        and sidecar.get("key") == key
        and (golden / "workspace").is_dir()
    )


def _sidecar(record: RealRiskWorkspace, symbols: Sequence[str], key: str) -> dict[str, object]:
    """What a restore must agree with: the identities the build resolved, bound to the key."""

    return {
        "format": _GOLDEN_FORMAT,
        "key": key,
        "source_tree": _source_tree_digest(),
        "environment": _environment_digest(),
        "symbols": list(symbols),
        "catalog_hash": record.feature_catalog.binding.catalog_hash,
        "kernels_hash": record.feature_kernels.installed_capability_hash,
        "manifest_revision": record.manifest.revision_sha256,
        "panel_manifest_ref": record.panel_manifest_ref,
        "panel_snapshot_hash": record.panel_snapshot_hash,
        "surface_hash": record.return_surface.surface_hash,
        "sector_by_listing_id": record.sector_by_listing_id,
        "files": _workspace_files(record.workspace),
        "built_at": time.time(),
    }


def _restore(
    workspace: Path,
    catalog: FeatureCatalog,
    kernels: FeatureKernelRegistry,
    symbols: Sequence[str],
    key: str,
    sidecar: dict[str, object],
) -> RealRiskWorkspace:
    """Rebuild the record from a copied workspace through the readers the product uses.

    Every identity the sidecar carries is checked against what the consumer asked
    for and against what the copied files say through production readers: the
    manifest re-read from the market store, the panel manifest resolved by the
    artifact resolver and matched to the active panel record, the return
    surface reloaded and verified by the artifact store. A disagreement is a
    corrupt or foreign golden and is refused; it is never repaired or rebuilt
    here, because a cache that silently rebuilds on corruption hides the
    corruption.
    """

    expected = {
        "format": _GOLDEN_FORMAT,
        "key": key,
        "source_tree": _source_tree_digest(),
        "environment": _environment_digest(),
        "symbols": list(symbols),
        "catalog_hash": catalog.binding.catalog_hash,
        "kernels_hash": kernels.installed_capability_hash,
    }
    for field, value in expected.items():
        if sidecar.get(field) != value:
            raise GoldenWorkspaceError(
                f"golden workspace {key}: sidecar {field} {sidecar.get(field)!r} != {value!r}"
            )
    if _workspace_files(workspace) != sidecar.get("files"):
        raise GoldenWorkspaceError(f"golden workspace {key}: cached file content changed")
    artifact_root = workspace / "artifacts"
    store = MarketDataRepository(workspace)
    manifest = _require_current_manifest(store)
    if manifest.revision_sha256 != sidecar.get("manifest_revision"):
        raise GoldenWorkspaceError(
            f"golden workspace {key}: manifest revision {manifest.revision_sha256} != "
            f"{sidecar.get('manifest_revision')!r}"
        )
    panel_manifest_ref = str(sidecar["panel_manifest_ref"])
    panel_snapshot_hash = str(sidecar["panel_snapshot_hash"])
    payload = ArtifactResolver(artifact_root).load_feature_panel_manifest(panel_manifest_ref)
    if payload["snapshot_hash"] != panel_snapshot_hash:
        raise GoldenWorkspaceError(
            f"golden workspace {key}: panel {payload['snapshot_hash']} != {panel_snapshot_hash}"
        )
    active = PanelStateRepository(store.database, market_data=store).active_feature_panel(
        MARKET_PROFILE_ID
    )
    if (
        active is None
        or str(active["manifest_revision"]) != manifest.revision_sha256
        or str(active["panel_content_hash"]) != payload["panel_content_hash"]
    ):
        raise GoldenWorkspaceError(f"golden workspace {key}: active panel does not match")
    sector = FeatureStateRepository(
        store.database,
        market_data=store,
        installed_catalog=catalog,
    ).current_sector_state(manifest)
    if sector is None:
        raise GoldenWorkspaceError(f"golden workspace {key}: sector state is absent")
    sector_by_listing_id = dict(sector.sector_by_listing_id)
    if sidecar.get("sector_by_listing_id") != sector_by_listing_id:
        raise GoldenWorkspaceError(f"golden workspace {key}: sector map does not match")
    surface = RiskReturnArtifactStore(artifact_root).load_manifest(str(sidecar["surface_hash"]))
    if surface.surface_hash != sidecar["surface_hash"]:
        raise GoldenWorkspaceError(f"golden workspace {key}: surface hash does not match")
    return RealRiskWorkspace(
        workspace=workspace,
        artifact_root=artifact_root,
        feature_catalog=catalog,
        feature_kernels=kernels,
        manifest=manifest,
        panel_manifest_ref=panel_manifest_ref,
        panel_snapshot_hash=panel_snapshot_hash,
        return_surface=surface,
        return_reader=CausalRiskReturnReader(artifact_root),
        sector_by_listing_id=sector_by_listing_id,
    )


@contextmanager
def _golden_lease(golden: Path):
    """Reuse the OS lease; a slow live builder is never evicted by its age."""
    started = time.monotonic()
    while True:
        try:
            lease = WorkspaceWriterLease.acquire(golden.parent / ".locks" / golden.name)
            break
        except RuntimeError as error:
            if time.monotonic() - started > _GOLDEN_WAIT_SECONDS:
                raise GoldenWorkspaceError(
                    f"golden workspace {golden.name}: builder busy"
                ) from error
            time.sleep(0.25)
    try:
        yield
    finally:
        lease.close()


def _materialize_golden(
    golden: Path,
    key: str,
    catalog: FeatureCatalog,
    kernels: FeatureKernelRegistry,
    symbols: Sequence[str],
    sector_size: int = 5,
) -> None:
    """Called under the same lease that protects copying and cache retention."""
    building = golden.with_name(golden.name + ".building")
    try:
        if _golden_is_complete(golden, key):
            return
        shutil.rmtree(building, ignore_errors=True)
        building.mkdir()
        with _checked_build_sources():
            record = _build_workspace(
                building / "workspace", catalog, kernels, symbols, sector_size
            )
        sidecar = _sidecar(record, symbols, key)
        del record
        gc.collect()  # the build's store handles must be closed before the rename
        (building / "golden.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        _publish(building, golden)
        _prune_goldens(keep=golden)
    finally:
        shutil.rmtree(building, ignore_errors=True)


def _publish(building: Path, golden: Path) -> None:
    """Rename the finished build into place, or fail; never copy into the final name.

    On Windows a handle the build left open refuses the rename. The way around
    it is a copy to a fresh sibling, which no process has open, and a rename of
    that. A copy straight into the final name would let a concurrent reader see
    the sidecar before the workspace it describes, so it is not an option.
    """

    for attempt in range(5):
        try:
            os.replace(building, golden)
            return
        except PermissionError:
            gc.collect()
            time.sleep(0.5 * (attempt + 1))
    staged = golden.with_name(f"{golden.name}.staged-{os.getpid()}")
    shutil.rmtree(staged, ignore_errors=True)
    shutil.copytree(building, staged)
    try:
        os.replace(staged, golden)
    except OSError as error:
        shutil.rmtree(staged, ignore_errors=True)
        raise GoldenWorkspaceError(f"golden workspace {golden.name}: cannot publish") from error


def _prune_goldens(*, keep: Path) -> None:
    """Drop goldens of source trees nobody has used for days; best effort."""

    horizon = time.time() - _GOLDEN_KEEP_DAYS * 86400
    for candidate in _GOLDEN_ROOT.iterdir():
        sidecar = candidate / "golden.json"
        if (
            candidate == keep
            or len(candidate.name) != 24
            or any(c not in "0123456789abcdef" for c in candidate.name)
            or not sidecar.exists()
        ):
            continue
        if sidecar.stat().st_mtime < horizon:
            try:
                lease = WorkspaceWriterLease.acquire(_GOLDEN_ROOT / ".locks" / candidate.name)
            except RuntimeError:
                continue
            try:
                if sidecar.exists() and sidecar.stat().st_mtime < horizon:
                    shutil.rmtree(candidate, ignore_errors=True)
            finally:
                lease.close()


GOLDEN_KEEP_KEYS = 3
"""Golden workspaces the cache keeps by last use (V235): every identity move made a key, and 78
of them held 35 GB before the rule was kept by hand."""


def retain_golden_workspaces(
    *, keep: int = GOLDEN_KEEP_KEYS, apply: bool = False, root: Path | None = None
) -> list[dict[str, object]]:
    """The golden workspaces by last use, newest first, each kept or dropped (V235).

    The newest ``keep`` stay. With ``apply`` every other one is deleted under its own lease, and
    one a builder holds is left ``BUSY``; without it nothing is deleted (``TO_DROP``).

    Args:
        keep: How many of the most recently used goldens stay.
        apply: Whether to delete the others.
        root: The cache's root; the checkout's own when None.

    Returns:
        One row per golden: its key, last use, bytes and disposition.
    """

    base = _GOLDEN_ROOT if root is None else root
    found: list[tuple[float, Path]] = []
    for candidate in sorted(base.iterdir()) if base.is_dir() else ():
        sidecar = candidate / "golden.json"
        if (
            len(candidate.name) == 24
            and all(c in "0123456789abcdef" for c in candidate.name)
            and sidecar.exists()
        ):
            found.append((sidecar.stat().st_mtime, candidate))
    rows: list[dict[str, object]] = []
    for index, (used, candidate) in enumerate(sorted(found, reverse=True)):
        size = sum(path.stat().st_size for path in candidate.rglob("*") if path.is_file())
        disposition = "KEPT" if index < keep else "TO_DROP"
        if disposition == "TO_DROP" and apply:
            try:
                lease = WorkspaceWriterLease.acquire(base / ".locks" / candidate.name)
            except RuntimeError:
                disposition = "BUSY"
            else:
                try:
                    shutil.rmtree(candidate, ignore_errors=True)
                    disposition = "TO_DROP" if candidate.exists() else "DROPPED"
                finally:
                    lease.close()
        rows.append(
            {
                "key": candidate.name,
                "last_used": datetime.fromtimestamp(used, UTC).isoformat(),
                "bytes": size,
                "disposition": disposition,
            }
        )
    return rows


def _build_workspace(
    workspace: Path,
    installed_catalog: FeatureCatalog,
    installed_kernels: FeatureKernelRegistry,
    symbols: Sequence[str],
    sector_size: int = 5,
) -> RealRiskWorkspace:
    """Drive onboarding, features, the panel and the return surface for real, into ``workspace``."""

    # The product's own layout: artifacts live under the workspace, which is what
    # every reader that takes only a workspace path assumes. A sibling directory
    # works only for callers that are told about it separately.
    artifact_root = workspace / "artifacts"
    schedule = materialize_calendar_schedule(
        ("XNAS",), start=HISTORY_START, end=AS_OF, as_of_timestamp=OBSERVED_AT
    )
    sessions = tuple(row["session_date"] for row in schedule.to_pylist())
    provider = SeededWalkProvider(symbols, sessions, sector_size=sector_size)

    store = MarketDataRepository(workspace)
    # Onboarding installs the current-feature schema, so this repository has to
    # know the revision too -- otherwise the table is created on the shipped
    # factor axis and the runtime later computes a factor it cannot store.
    feature_state = FeatureStateRepository(
        store.database, market_data=store, installed_catalog=installed_catalog
    )
    panel_state = PanelStateRepository(store.database, market_data=store)

    # Onboard through the readiness gate rather than by constructing the runner
    # directly: the gate is what records workspace readiness, and without that
    # record every later maintenance cycle blocks on initialization consent.
    gate = WorkspaceReadinessGate(
        market_data=store,
        feature_state=feature_state,
        panel_state=panel_state,
        profile_path=PROFILE_PATH,
        provider=provider,
        source_loader=_source_loader_for(symbols),
    )
    approved = gate.start_approved_onboarding(
        WorkspaceReadinessConsent(
            consent_id=uuid4(),
            market_profile_id=MARKET_PROFILE_ID,
            action=WorkspaceConsentAction.INITIALIZE,
            approved_at=OBSERVED_AT,
        ),
    )
    onboarding = approved.runner.run(observed_at=OBSERVED_AT)
    if onboarding.status is not CurrentUniverseOnboardingStatus.COMPLETED:
        raise AssertionError(f"onboarding did not complete: {onboarding.status}")
    # Promote the frozen manifest, or readiness stays ONBOARDING_IN_PROGRESS and
    # the coordinator tries to resume a task that has already finished.
    gate.complete_onboarding(approved, onboarding, observed_at=OBSERVED_AT)
    manifest = _require_current_manifest(store)

    runtime = WorkspaceRuntime.create(
        workspace=workspace,
        manifest=manifest,
        provider=provider,
        artifact_root=artifact_root,
        feature_catalog=installed_catalog,
        feature_kernels=installed_kernels,
    )
    try:
        # A workspace this new has no closure, and persistence requires one: one for each
        # part of the installed catalog's layer, the shipped base and the extension's column.
        for part in runtime.feature_foundation.layer.parts:
            genesis = FeatureClosureGenesisService(
                panel_state=runtime.panel_state,
                source=FeatureClosureSourceRepository(runtime.database.path),
                ledger=runtime.closure_ledger,
            ).open_genesis(manifest=manifest, catalog=part)
            if genesis.disposition != "GENESIS_READY":
                raise AssertionError(f"unexpected genesis disposition: {genesis.disposition}")

        # The maintenance coordinator is the production orchestrator: it runs
        # the Feature Input Gateway admission, the build, and the panel
        # publication in the one order the panel reader will accept. Hand
        # sequencing those calls skips the gateway and produces a panel that
        # Factor and Risk readers reject.
        # The nested Feature task runner reads its own clock rather than the
        # coordinator's observation time, and it is what stamps the sector
        # revision. Left at wall clock it would stamp today, which is two weeks
        # past the audit receipts written at OBSERVED_AT, and their twenty-four
        # hour reuse window would reject every one of them.
        clock = _MovingClock(OBSERVED_AT)
        coordinator = runtime.maintenance_coordinator(
            readiness_gate=WorkspaceReadinessGate(
                market_data=runtime.market_data,
                feature_state=runtime.feature_state,
                panel_state=runtime.panel_state,
                profile_path=PROFILE_PATH,
                provider=provider,
                # The coordinator assesses readiness without naming a loader, so
                # a stale source check would otherwise reach the live network.
                source_loader=_source_loader_for(symbols),
            ),
            clock=clock,
        )
        # Initial Feature/Sector materialization may yield RUNNING before the
        # same maintenance cycle performs Gateway admission. Drive that owed
        # continuation; a completed request remains idempotent.
        for attempt in range(1, 5):
            observed = OBSERVED_AT + timedelta(seconds=attempt - 1)
            clock.now = observed
            # Admission derives a quality-filtered child manifest and makes it
            # current, so a request still naming the parent is rejected as a
            # membership mismatch. Reading the current one each cycle is what a
            # host does, and it is also where the admission is disclosed.
            manifest = _require_current_manifest(runtime.market_data)
            request = WorkspaceMaintenanceRequest.create(
                market_profile_id=MARKET_PROFILE_ID,
                target_market_session=AS_OF,
                knowledge_cutoff_at=observed,
                trigger=MaintenanceTrigger.STARTUP,
                membership_revision=manifest.revision_sha256,
                data_policy_hash=canonical_hash({"fixture": "researcher-methodology-surface"}),
                feature_policy_hash=canonical_hash(
                    {
                        "catalog": runtime.feature_foundation.catalog.binding.catalog_hash,
                        "invalidation": "domain-topology",
                    }
                ),
            )
            outcome = coordinator.run(request, observed_at=observed)
            while outcome.status is MaintenanceStatus.RUNNING:
                outcome = coordinator.run(request, observed_at=observed)
            if outcome.status not in {MaintenanceStatus.COMPLETED, MaintenanceStatus.NOOP}:
                raise AssertionError(f"maintenance cycle {attempt} did not complete: {outcome}")
            manifest = _require_current_manifest(runtime.market_data)
            quality = runtime.panel_state.feature_input_quality_disclosure(
                result_manifest_revision=manifest.revision_sha256
            )
            if quality.get("gateway_qualified") is True:
                break
        else:
            raise AssertionError("no maintenance cycle produced a Feature Input Gateway admission")

        snapshot = runtime.panel_state.feature_panel_snapshot_for_active(MARKET_PROFILE_ID)
        if snapshot is None:
            raise AssertionError("maintenance completed without publishing a panel snapshot")
        panel_snapshot_hash = str(snapshot["snapshot_hash"])
        panel_manifest_ref = str(snapshot["manifest_uri"])

        sector = runtime.feature_state.current_sector_state(manifest)
        if sector is None:
            raise AssertionError("sector state missing after maintenance")
        sector_by_listing_id = dict(sector.sector_by_listing_id)
    finally:
        runtime.close()

    resolver = ArtifactResolver(artifact_root)
    surface, _descriptor = CausalRiskReturnSurfacePublisher(
        store=store,
        resolver=resolver,
        artifact_root=artifact_root,
        mutation_gate=WorkspaceMutationGate(),
    ).publish(
        panel_manifest_ref=panel_manifest_ref,
        market_profile_id=MARKET_PROFILE_ID,
    )
    return RealRiskWorkspace(
        workspace=workspace,
        artifact_root=artifact_root,
        feature_catalog=installed_catalog,
        feature_kernels=installed_kernels,
        manifest=manifest,
        panel_manifest_ref=panel_manifest_ref,
        panel_snapshot_hash=panel_snapshot_hash,
        return_surface=surface,
        return_reader=CausalRiskReturnReader(artifact_root),
        sector_by_listing_id=sector_by_listing_id,
    )


__all__ = [
    "AS_OF",
    "GOLDEN_KEEP_KEYS",
    "HISTORY_START",
    "MARKET_PROFILE_ID",
    "OBSERVED_AT",
    "SYMBOLS",
    "GoldenWorkspaceError",
    "RealRiskWorkspace",
    "SeededWalkProvider",
    "build_real_risk_workspace",
    "development_feature_catalog",
    "publish_causal_outcomes",
    "retain_golden_workspaces",
]
