"""Own pytest's temporary roots and the QA launcher's confined project discovery.

Also a pytest plugin of the root conftest: a fixture narrower than the session takes the
temporary roots it made with it (pytest_fixture_setup)."""

from __future__ import annotations

import functools
import getpass
import json
import os
import stat
import sys
import tempfile
import time
import types
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING, Any, TextIO, cast
from unittest.mock import patch

import pytest

from devtools.evaluation.golden_workspaces import payload_path, remove_tree

if TYPE_CHECKING:
    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceManifest,
    )
    from alphalattice.interface.local_application.client import LocalResearchClient

PROJECT_ROOT: ContextVar[Path | None] = ContextVar("pytest_project_root", default=None)
# The running test, set by the root conftest: the session seeds it copies read their market data
# and Parquet in place unless it is marked private_data (it writes them, so it gets a copy).
RUNNING_TEST: dict[str, object] = {"id": None, "private": False, "factory": None}
# This process's backup root: one per process, named once at configure. A test's own workspaces'
# generations go with the passed test, and the root with a clean session (backups_made).
BACKUP_ROOT: Path | None = None
DATA_ROOT: Path | None = None
MODEL_STORE: Path | None = None
# The places a test may resolve a Host root to: the run's temporary roots, and for the model
# store also an admitted shared kit. The person's own data, backup and model folders are never
# among them, whatever a variable the run inherited names.
OWNED_ROOTS: list[Path] = []
MODEL_ROOTS: list[Path] = []
# A workshop test host names its read-only cache of model packs in this test-only variable as
# well as in ALPHALATTICE_MODEL_STORE; the product's own variable, which a person's shell may
# set, is never its own authority, whatever its path.
SHARED_MODEL_KIT = "ALPHALATTICE_TEST_MODEL_KIT"
PERSON_RESOLVERS = (  # module, resolver, the variable naming its root, whether it reads home
    ("alphalattice.foundation.data_layer.history", "data_root", "ALPHALATTICE_DATA_ROOT", False),
    (
        "alphalattice.control.product_host.storage.backup",
        "default_backup_root",
        "ALPHALATTICE_BACKUP_ROOT",
        False,
    ),
    (
        "alphalattice.capabilities.knowledge.model_store",
        "default_store_root",
        "ALPHALATTICE_MODEL_STORE",
        True,  # its pointer file lies under the home folder
    ),
)


@contextmanager
def owning(*roots: Path) -> Iterator[None]:
    """Admit roots a test helper is handed by path, a golden's and its copy's served roots, while
    it works on them; never a default the run did not name."""
    OWNED_ROOTS.extend(roots)
    try:
        yield
    finally:
        for root in roots:
            OWNED_ROOTS.remove(root)


def _owned(path: Path, roots: list[Path]) -> bool:
    """Whether ``path``'s resolved target lies in a root: a link under a root that leads out of
    it is outside."""
    place = Path(os.path.realpath(path))
    return any(place.is_relative_to(Path(os.path.realpath(root))) for root in roots)


def model_store_root(named: str, kit: str, make: Any) -> Path:
    """The run's model store: the inherited store when the test host named that same folder as
    its shared kit, else ``make()``, a store of the run's own."""
    if named and kit and os.path.realpath(named) == os.path.realpath(kit):
        return Path(os.path.realpath(named))
    return Path(make())


def _default_bases(reads_home: bool) -> list[Path]:
    """The folders a resolver's platform default stands on, as the product chooses them."""
    key = "LOCALAPPDATA" if os.name == "nt" else "XDG_DATA_HOME"
    bases = [Path(os.environ[key])] if os.environ.get(key) else [Path.home()]
    return bases + ([Path.home()] if reads_home else [])


GUARDED: dict[Any, Any] = {}  # each wrapped resolver, and its guard


def _owned_only(
    resolve: Any, where: str, variable: str, reads_home: bool, roots: list[Path]
) -> Any:
    @functools.wraps(resolve)
    def owned(*args: Any, **kwargs: Any) -> Any:
        # judged before the resolver runs, so a person's pointer or folder is never even read
        named = os.environ.get(variable, "").strip()
        bases = [] if named else _default_bases(reads_home)
        outside = [base for base in bases if not _owned(base, roots)]
        if not outside and _owned(found := resolve(*args, **kwargs), roots):
            return found
        raise AssertionError(
            f"tests.person_root_resolved: {RUNNING_TEST.get('id') or 'a test'} reached "
            f"{outside[0] if outside else found} through {where}, outside this run's own roots; "
            f"keep {variable} set, or point its default's folders at tmp_path"
        )

    return owned


def require_named_roots(entry: str) -> None:
    """Refuse, by name, a test helper run where no root of its own is named.

    A test process names each Host root at its start; a plain python run of a builder or a
    proof names none, and each resolver's default is the person's own folder, which the run
    would then open and write. Called first by every helper a script can run, before any root
    is resolved, so the person's folders are never read.

    Raises:
        RuntimeError: `tests.person_root_unnamed`, naming the entry and each variable unset.
    """
    unset = [
        variable
        for _module, _resolver, variable, _home in PERSON_RESOLVERS
        if not os.environ.get(variable, "").strip()
    ]
    if unset:
        raise RuntimeError(
            f"tests.person_root_unnamed: {entry} would open the person's own root, because "
            f"{', '.join(unset)} {'is' if len(unset) == 1 else 'are'} unset; name a root of "
            "this run's own for each "
            "(wk.core.workshop_roots(run) gives all three), or run it under pytest"
        )


def guard_person_roots() -> None:
    """Wrap each loaded Host root resolver so a test that reaches a default outside this run's
    roots fails by name; called before each test, so a resolver imported since is wrapped too,
    and so is a module's own name for one it imported (`from ... import data_root`)."""
    for name, attribute, variable, reads_home in PERSON_RESOLVERS:
        resolve = getattr(sys.modules.get(name), attribute, None)
        if resolve is not None and resolve not in GUARDED.values():
            roots = MODEL_ROOTS if variable == "ALPHALATTICE_MODEL_STORE" else OWNED_ROOTS
            where = f"{name}.{attribute}"
            GUARDED[resolve] = _owned_only(resolve, where, variable, reads_home, roots)
            setattr(sys.modules[name], attribute, GUARDED[resolve])
    names = {attribute for _, attribute, _, _ in PERSON_RESOLVERS}
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("alphalattice."):
            space = vars(module)
            for attribute in names & space.keys():
                if isinstance(space[attribute], types.FunctionType) and space[attribute] in GUARDED:
                    space[attribute] = GUARDED[space[attribute]]


def backups_made() -> frozenset[str]:
    """The workspaces this process's backup root holds generations for."""
    try:
        return frozenset(e.name for e in os.scandir(BACKUP_ROOT)) if BACKUP_ROOT else frozenset()
    except FileNotFoundError:
        return frozenset()


def golden_runtime():
    """Supply public OS drivers without importing the private explorer into pytest setup."""
    import portalocker  # lazy: only contemporary golden counterfactuals need these drivers.
    import psutil

    from alphalattice.control.workspace_runtime.paths import resolve_confined
    from devtools.evaluation import GoldenRuntime

    return GoldenRuntime.from_os(psutil, portalocker, resolve_confined)


def unplaced_workers(numprocesses: object, environ: dict[str, str], above: Iterable[Path]) -> str:
    """The warning for a run asking more than 2 xdist workers without a heavy place, or "".

    Only inside the private workshop, found by its tools/heavy_slot.py in ``above`` (the folders
    above the checkout): the
    public tree and CI have none, and there the place variable means nothing. It never refuses,
    since a minute-long run of at most 2 workers needs no place."""
    if environ.get("WK_HEAVY_PLACE") or not numprocesses:
        return ""
    if isinstance(numprocesses, int) and numprocesses <= 2:
        return ""
    try:
        slot = next(
            (up / "tools/heavy_slot.py" for up in above if (up / "tools/heavy_slot.py").is_file()),
            None,
        )
    # an unreadable marker is unknown: neither absence nor a reason to stop
    except OSError as error:
        return (
            f"pytest asks for {numprocesses} workers without a heavy place, and whether this is "
            f"the workshop is unknown ({error}); if it is, run it through tools/heavy_slot.py run"
            ' --line <line> --what "<what>" -- <this command>'
        )
    if slot is None:
        return ""
    return (
        f"pytest asks for {numprocesses} workers without a heavy place; run it through one: "
        f'python "{slot.as_posix()}" run --line <line> --what "<what>" -- <this command>'
    )


def warn_unplaced(config: object, above: Iterable[Path], stream: TextIO) -> None:
    """pytest_configure's controller step: write the unplaced-workers warning, if any, to
    ``stream``; it never stops the run."""
    numprocesses = config.getoption("numprocesses", None)  # type: ignore[attr-defined]
    if warning := unplaced_workers(numprocesses, dict(os.environ), above):
        print(warning, file=stream, flush=True)


def configure_temporary_roots(checkout: Path, basetemp: str | None) -> None:
    """Refuse checkout temp before collection and confine the existing backup root."""
    if basetemp is not None:
        roots = {"--basetemp": Path(os.path.abspath(str(basetemp)))}
    else:
        temproot = Path(os.environ.get("PYTEST_DEBUG_TEMPROOT") or tempfile.gettempdir())
        try:
            user = getpass.getuser() or "unknown"
        except (OSError, KeyError):
            user = "unknown"
        roots = {
            "pytest temporary root": temproot,
            "pytest user temporary root": temproot / f"pytest-of-{user}",
            "pytest fallback temporary root": temproot / "pytest-of-unknown",
        }
    for source, path in roots.items():
        if (resolved := path.resolve()).is_relative_to(checkout):
            raise ValueError(
                f"basetemp_inside_checkout: {source} resolves to {resolved}, inside checkout "
                f"{checkout}. Refused before collection: checkout artifacts and hooks would "
                "contaminate test results. Use --basetemp with an absolute directory outside "
                "the checkout, or move PYTEST_DEBUG_TEMPROOT / TMPDIR / TEMP / TMP outside it."
            )
    backup_parent = Path(tempfile.gettempdir()).resolve()
    if backup_parent.is_relative_to(checkout):
        backup_parent = (
            Path(basetemp).resolve().parent if basetemp is not None else temproot.resolve()
        )
    # Printed commands are quoted for the shell a run started in, and a test replays one by POSIX
    # rules: under PowerShell quoting that loses a path's backslashes, and the file lands at the
    # drive root. The run's own shell must not decide that; a test of another shell sets its own.
    os.environ.setdefault("ALPHALATTICE_SHELL", "posix")
    global BACKUP_ROOT, DATA_ROOT
    BACKUP_ROOT = Path(tempfile.mkdtemp(prefix="alphalattice-test-backups-", dir=backup_parent))
    os.environ["ALPHALATTICE_BACKUP_ROOT"] = str(BACKUP_ROOT)
    # The Host's research history: one data root per test process, never the machine's own.
    DATA_ROOT = Path(tempfile.mkdtemp(prefix="alphalattice-test-data-", dir=backup_parent))
    os.environ["ALPHALATTICE_DATA_ROOT"] = str(DATA_ROOT)
    # The Host's model packs: an admitted shared kit (a read-only cache of packs), else one per
    # test process; never the person's own, which a run once loaded and filled.
    global MODEL_STORE
    MODEL_STORE = model_store_root(
        os.environ.get("ALPHALATTICE_MODEL_STORE", "").strip(),
        os.environ.get(SHARED_MODEL_KIT, "").strip(),
        lambda: tempfile.mkdtemp(prefix="alphalattice-test-models-", dir=backup_parent),
    )
    os.environ["ALPHALATTICE_MODEL_STORE"] = str(MODEL_STORE)
    _declare_environment(
        {
            "ALPHALATTICE_SHELL",
            "ALPHALATTICE_DATA_ROOT",
            "ALPHALATTICE_BACKUP_ROOT",
            "ALPHALATTICE_MODEL_STORE",
        },
        "test-worker-roots",
    )
    OWNED_ROOTS[:] = [backup_parent, Path(tempfile.gettempdir())]
    if basetemp is not None:
        OWNED_ROOTS.append(Path(os.path.abspath(str(basetemp))).parent)
    MODEL_ROOTS[:] = [*OWNED_ROOTS, MODEL_STORE]


def _declare_environment(names: set[str], kind: str) -> None:
    """Keep pytest's explicit OS overrides visible to the frozen product configuration."""
    from alphalattice.kernel.config import Change, declare

    declare(Change(kind, {name: os.environ.get(name) for name in names}))


class ConfigMonkeyPatch(pytest.MonkeyPatch):
    """Apply existing pytest env fixtures as their own declared configuration changes."""

    def __init__(self) -> None:
        super().__init__()
        self._environment_names: set[str] = set()

    def setitem(self, mapping: Any, name: Any, value: Any) -> None:
        super().setitem(mapping, name, value)
        self._note(mapping, name)

    def delitem(self, mapping: Any, name: Any, raising: bool = True) -> None:
        super().delitem(mapping, name, raising)
        self._note(mapping, name)

    def _note(self, mapping: Any, name: Any) -> None:
        if mapping is os.environ:
            name = name.upper() if os.name == "nt" else name
            self._environment_names.add(name)
            _declare_environment({name}, "test-override")

    def undo(self) -> None:
        super().undo()
        if self._environment_names:
            _declare_environment(self._environment_names, "test-override-restored")
            self._environment_names.clear()


@pytest.fixture
def monkeypatch() -> Iterator[pytest.MonkeyPatch]:
    """Expose existing test overrides to Python children and the product's settings port."""
    with ConfigMonkeyPatch.context() as patcher:
        yield patcher


def served_root(workspace: Path) -> Path:
    """The data root of a Host served from `workspace` in another process: its own, beside the
    workspace, since a process holds one data root and this one may hold its own."""
    return workspace.parent / f"{workspace.name}.data-root"


def served_client(workspace: Path) -> LocalResearchClient:
    """A client of the Host served apart from `workspace`: the product's own reader finds that
    Host's endpoint record where it writes it, on its data root, never on this process's."""
    from alphalattice.foundation.data_layer import history
    from alphalattice.interface.local_application.client import LocalResearchClient

    with history.own_root(served_root(workspace)):  # the client reads its record once, here
        return LocalResearchClient(workspace)


def serve_apart(workspace: Path) -> Path:
    """Make `workspace` ready for a Host served in another process, and that Host's data root:
    the Tasks this process's log holds for the workspace are carried into it, so that Host finds
    them."""
    from tests._support import require_isolated_roots

    require_isolated_roots()
    from alphalattice.control.task_control.upgrade import carry_tasks  # lazy: a served Host's only
    from alphalattice.foundation.data_layer.history import (
        carry_arrays,
        carry_knowledge_snapshots,
        carry_research_inputs,
    )

    workspace.mkdir(parents=True, exist_ok=True)
    carry_tasks(workspace, workspace)
    carry_arrays(workspace, workspace)
    carry_research_inputs(workspace, workspace)
    carry_knowledge_snapshots(workspace, workspace)
    return served_root(workspace)


def released(root: Path) -> None:
    """Return once a killed Host's hold on the data root `root` is gone: Windows frees a dead
    process's file locks a moment after the process ends (about 16 ms idle, more under load)."""
    from alphalattice.foundation.data_layer.log import LOCK_NAME, hold  # lazy: a killed Host's only

    lock, deadline = root / "machine" / LOCK_NAME, time.monotonic() + 30
    if not lock.is_file():
        return  # it never held this root
    while (handle := hold(lock)) is None:
        assert time.monotonic() < deadline, f"a dead Host still holds {root}"
        time.sleep(0.01)
    handle.close()


def with_its_tasks(workspace: Path) -> Path:
    """Carry the served Host's records and Evidence originals into the workspace to seal."""
    from alphalattice.control.task_control.upgrade import carry_tasks  # lazy: a golden seal's only
    from alphalattice.evidence.alternative_evidence.storage.world import carry_originals
    from alphalattice.foundation.data_layer.history import (
        carry_arrays,
        carry_knowledge_snapshots,
        carry_research_inputs,
    )

    root = served_root(workspace)
    carry_tasks(workspace, workspace, root=root)
    carry_arrays(workspace, workspace, root=root)
    carry_research_inputs(workspace, workspace, root=root)
    carry_knowledge_snapshots(workspace, workspace, root=root)
    for _ in carry_originals(workspace, workspace, root=root):
        pass
    return workspace


def golden_payload(workspace: Path, parent: Path | None) -> Path:
    """Carry the Host's records and give the golden its own copies of the Host's file aliases."""
    from alphalattice.foundation.data_layer.inputs import folder_files

    def linked(root: Path) -> Iterator[tuple[Path, tuple[int, int]]]:
        for _, file in folder_files(payload_path(root)):
            info = file.lstat()
            if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
                yield file, (info.st_dev, info.st_ino)

    with_its_tasks(workspace)
    source = payload_path(served_root(workspace))
    if not source.is_dir():
        return workspace
    shared = {inode for _, inode in linked(source)}
    if shared and parent is not None:
        shared.difference_update(inode for _, inode in linked(parent))
    if shared:
        with tempfile.TemporaryDirectory(dir=source.parent) as pending:
            staged = Path(pending) / "body"
            for file, inode in linked(workspace):
                if inode in shared:
                    copyfile(file, staged)
                    staged.replace(file)
    return workspace


PORTFOLIO_RESULTS = {
    ("portfolio-strategy-lab", "execution-series-descriptors"): (
        "portfolio_execution_series",
        "series",
    ),
    ("portfolio-strategy-lab", "report-holdings-descriptors"): (
        "portfolio_report_holding",
        "holding",
    ),
    ("portfolio-benchmark", "surfaces"): ("portfolio_benchmark", "series"),
}
"""Each Portfolio result a workspace names by a file, by the owner folder and folder the file
lies in, as the Portfolio upgrade families find them: its catalog kind and table. A descriptor is
written only beside its catalog rows; a benchmark surface may be a 0.2.7 file the upgrade still
moves into the catalog."""


def _portfolio_results(copy: Path) -> list[tuple[str, str, str]]:
    """The Portfolio results `copy`'s files name: kind, digest and table, in a fixed order."""
    from alphalattice.foundation.data_layer.upgrade import workspace_files  # lazy: a copy's only

    found = set()
    for path in workspace_files(copy):
        depth = 2 if path.parent.name.endswith("-descriptors") else 1  # below a namespace
        named = PORTFOLIO_RESULTS.get((path.parents[depth].name, path.parent.name))
        if path.suffix == ".json" and named is not None:
            kind, table = named
            found.add((kind, path.stem, table))
    return sorted(found)


def _carry_recorded_review(copy: Path, source: Path, target: Path) -> None:
    """Keep the nested recorded-review originals at the root that will serve the copy."""
    from alphalattice.evidence.alternative_evidence.storage.world import (
        keep_original,
        original_locator_bodies,
    )

    evidence = copy / "recorded-review/evidence"
    for _, body in original_locator_bodies(
        evidence / "artifacts", evidence / "workspace", root=source
    ):
        keep_original(target, body)


def carry_served_copy(workspace: Path, copy: Path) -> None:
    """Carry a copied Host workspace and the originals its recorded-review fixture reads."""
    from alphalattice.control.product_host.composition.saved_object_readback import carry_copy

    source, target = served_root(workspace), served_root(copy)
    with owning(source, target):
        carry_copy(workspace, copy, root=source)
        _carry_recorded_review(copy, source, target)


def _with_its_market_selections(copy: Path, source: Path, target: Path) -> None:
    """`copy` moved from `source` to `target` as a Host serving it there needs: the workspace
    re-admitted (`world.carry_workspace`) and each binding it carries keeping its versions, carried
    into `target`'s catalog (`world.carry_selection`)."""
    from alphalattice.control.product_host.composition.research_workspace import (
        named_workspace_id,  # lazy: a golden copy's only
    )
    from alphalattice.foundation.data_layer import history
    from alphalattice.foundation.data_layer.inputs import CARRIED
    from alphalattice.foundation.market_data_ops.storage import world

    if (copy / "market-data.duckdb").is_file():
        world.carry_workspace(source, copy / "market-data.duckdb")
    carried = copy / CARRIED
    bindings = [
        carried / digest.partition(":")[2]
        for manifest in sorted(carried.glob("*.json"))
        for relative, digest in json.loads(manifest.read_bytes())
        if relative.endswith("market-data.duckdb")
    ]
    if not bindings:
        return
    with history.own_root(target):
        history.open_history(str(named_workspace_id(copy)), copy.name, copy.resolve())
        try:
            for database in bindings:
                world.carry_selection(source, database)
        finally:
            history.close_history()


def with_its_outcomes(copy: Path, source: Path, root: Path | None = None) -> Path:
    """`copy`, a copy of a workspace served on the data root `source`, its execution outcomes
    and catalog-held Risk and Portfolio results carried into `root`, the copy's own served root
    unless named, and the world rows its market selections name
    (`_with_its_market_selections`). A Host on a fresh root needs those catalog rows to read or back
    up the copied workspace. The recorded review shares that root, so its nested Evidence
    locators need their exact original bodies there too. An in-process Host names its data root
    explicitly."""
    from alphalattice.control.product_host.composition.research_workspace import (
        named_workspace_id,  # lazy: a golden copy's only
    )
    from alphalattice.foundation.causal_outcomes.execution.readers import (
        committed_outcome_chunks,
    )
    from alphalattice.foundation.data_layer import history
    from alphalattice.foundation.market_data_ops.storage.world import admit_universe
    from alphalattice.investment.risk_research.experiments.development_artifacts import (
        covariance_chunks,
    )
    from alphalattice.investment.risk_research.surfaces.artifacts import RiskArtifactError

    target = served_root(copy) if root is None else root
    if Path(os.path.realpath(target)) != Path(os.path.realpath(source)):
        with owning(source, target):
            _with_its_market_selections(copy, source, target)
    held = sorted(
        {c.content_hash for _store, c, file in committed_outcome_chunks(copy) if not file.exists()}
    )
    results = _portfolio_results(copy)
    risk = {
        (chunk.content_hash, axis): (store, chunk, axis)
        for store, chunk, axis, file in covariance_chunks(copy)
        if not file.is_file()
    }
    evidence = copy / "recorded-review/evidence"
    if not held and not results and not risk and not evidence.is_dir():
        return copy
    with owning(source, target):
        _carry_recorded_review(copy, source, target)
        with history.own_root(source):
            bodies = [(digest, history.read_execution_outcome(digest)) for digest in held]
            workspace = str(named_workspace_id(copy))
            rows = [  # a file-only benchmark is left to the upgrade, as a 0.2.7 workspace's is
                (
                    kind,
                    digest,
                    table,
                    history.read_portfolio_result(kind, digest, result_table=table),
                )
                for kind, digest, table in results
                if kind != "portfolio_benchmark"
                or history.portfolio_result_held(kind, digest, workspace)
            ]
        if not bodies and not rows and not risk:
            return copy
        with history.own_root(target):
            history.open_history(workspace, copy.name, copy.resolve())
            for digest, body in bodies:
                history.write_execution_outcome(digest, body)
            for kind, digest, table, (profile, values) in rows:
                history.write_portfolio_result(kind, digest, profile, values, result_table=table)
            if risk:
                admit_universe(copy / "market-data.duckdb", workspace)
                for store, chunk, axis in risk.values():
                    packed = store.catalog_covariance(chunk, axis, root=source)
                    if packed is None:
                        raise RiskArtifactError("risk_research.covariance_chunk_incomplete")
                    store.admit_covariance(chunk, axis, packed, workspace_id=workspace)
    return copy


def served_here(copy: Path) -> Path:
    """`copy`, a golden's copy a Host in this process is about to serve, its execution outcomes
    carried from the copy's own served root to this process's data root, the root that Host
    opens. Called where the Host is composed, before it opens: there the root is known to serve
    the copy, which the shared copy path cannot know."""
    from tests._support import require_isolated_roots

    require_isolated_roots()
    from alphalattice.foundation.data_layer import history

    return with_its_outcomes(copy, served_root(copy), history.data_root())


def upgraded(workspace: Path) -> Path:
    """`workspace`, an old workspace's copy, as its acknowledged first start leaves it on this
    test's data root: every family run on it (its Strategy Lab lanes moved through the array
    door among them) in its own Host's history, its backups the copy's source, so a Host serves
    it at once. The history closes after, as that Host's does when it stops."""
    from tests._support import require_isolated_roots

    require_isolated_roots()
    from alphalattice.control.product_host.composition.application_session import (
        carried_result_checks,
    )
    from alphalattice.control.product_host.composition.first_start import FAMILIES
    from alphalattice.control.product_host.composition.research_workspace import (
        named_workspace_id,
    )
    from alphalattice.foundation.data_layer import history
    from alphalattice.foundation.data_layer.upgrade import UpgradeRunner

    copy = workspace.resolve()
    checks = carried_result_checks()  # the results it carries, as its Host admits them
    history.open_history(str(named_workspace_id(copy)), copy.name, copy, carried_checks=checks)
    try:
        backups = {"workspaces": [{"workspace": str(copy)}]}
        families = tuple(make() for make in FAMILIES)
        log = history.held()
        UpgradeRunner(history.data_root(), log, (copy,), families, lambda *_: backups).run()
    finally:
        history.close_history()
    return workspace


def closure_on_log(workspace: Path) -> Path:
    """`workspace`, built by test code outside a Host, with the Panel closure its builders wrote
    in folder form (a folder no Host had put on the log) moved onto this process's data root as a
    first start moves it: a seed then carries it with its arrays, and the Host serving a copy
    opens with no upgrade to run. A workspace holding no such closure is left as it is."""
    from alphalattice.control.product_host.composition.research_workspace import (
        named_workspace_id,
    )
    from alphalattice.foundation.data_layer import history
    from alphalattice.foundation.data_layer.upgrade import UpgradeRunner
    from alphalattice.foundation.feature_engine.panels.closure_files import closure_upgrade_family

    copy, family = workspace.resolve(), closure_upgrade_family(named_workspace_id)
    if not family.holds(copy):
        return workspace
    opened = history.holding() is None
    if opened:
        history.open_history(str(named_workspace_id(copy)), copy.name, copy)
    try:
        backups = {"workspaces": [{"workspace": str(copy)}]}
        UpgradeRunner(
            history.data_root(), history.held(), (copy,), (family,), lambda *_: backups
        ).run()
    finally:
        if opened:
            history.close_history()
    return workspace


def admitted(workspace: Path) -> Path:
    """`workspace`, a plain copy of a folder a Host served, admitted as a Host's open admits it
    (`history.move_carried`): what it carries goes onto this test's log, so a read outside a Host
    sees the closure that Host kept."""
    from alphalattice.control.product_host.composition.research_workspace import (
        named_workspace_id,
    )
    from alphalattice.control.product_host.research_authoring.factor_inputs import (
        carried_binding_check,
    )
    from alphalattice.foundation.data_layer import history

    named = str(named_workspace_id(workspace))
    history.move_carried(workspace.resolve(), named, input_check=carried_binding_check)
    return workspace


@contextmanager
def project_scope(
    root: Path, temporary_root: Path | None = None, *, host_reads: bool = True
) -> Iterator[None]:
    """Confine discovery; a guarded unit needs the context, never a Host read adapter."""
    allowed = (temporary_root or PROJECT_ROOT.get() or root).resolve()
    assert root.resolve().is_relative_to(allowed), "project_outside_test_root"
    token = PROJECT_ROOT.set(allowed)
    try:
        if not host_reads:
            yield
            return
        from alphalattice.control.product_host.composition import first_use_road

        project, read = (
            first_use_road.session_project,
            first_use_road.read_research_workspace_manifest,
        )

        def scoped_project(start: Path, host: str | None) -> Path:
            if start == Path.cwd() and not start.resolve().is_relative_to(
                PROJECT_ROOT.get() or allowed
            ):
                start = root
            found = Path(project(start, host))
            assert found.resolve().is_relative_to(PROJECT_ROOT.get() or allowed), (
                f"project_outside_test_root: {found}"
            )
            return found

        def scoped_read(workspace: Path) -> ResearchWorkspaceManifest:
            assert workspace.resolve().is_relative_to(PROJECT_ROOT.get() or allowed), (
                f"project_outside_test_root: {workspace}"
            )
            return cast("ResearchWorkspaceManifest", read(workspace))

        with (
            patch.object(first_use_road, "session_project", scoped_project),
            patch.object(first_use_road, "read_research_workspace_manifest", scoped_read),
        ):
            yield
    finally:
        PROJECT_ROOT.reset(token)


# A fixture narrower than the session takes the temporary roots it made with it. A module's private
# workspace copy lived to the session's end: at a full run's peak, 31 of one fixture's held 3.4 GB
# (mac1, 11-07). Each root `tmp_path_factory.mktemp` makes is named after the fixture being set up,
# and goes once that fixture's own teardown is done, unless a test failed while it lived: its inputs
# are that failure's evidence. A session fixture keeps its roots to the end, and tmp_path keeps its
# own removal (tests/conftest.py), which waits for the test's verdict.
_ROOTS: list[list[Path] | None] = []
"""The roots made for each fixture being set up, innermost last; None for one that keeps them."""
_TORN: list[tuple[list[Path], Any, int]] = []
"""Roots whose fixture's teardown ran, held for its report: one that raised keeps them."""


def _record(factory: Any) -> None:
    if getattr(factory, "_fixture_roots_recorded", False):
        return
    made = factory.mktemp

    def mktemp(basename: str, numbered: bool = True) -> Path:
        path: Path = made(basename, numbered)
        if _ROOTS and _ROOTS[-1] is not None:
            _ROOTS[-1].append(path)
        return path

    factory.mktemp, factory._fixture_roots_recorded = mktemp, True


def _remove(roots: list[Path], request: Any, failed: int) -> None:
    factory = request.config._tmp_path_factory
    if request.session.testsfailed != failed or factory._retention_policy != "failed":
        return
    for root in roots:
        if os.path.lexists(root) and (left := remove_tree(root, factory.getbasetemp())):
            print(
                f"test_tmp.fixture_scratch_left: {root}: {'; '.join(left)[:500]}", file=sys.stderr
            )


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """A teardown is reported once all of it ran: a fixture whose own teardown raised is a failure
    and its roots stay with the rest of its evidence."""
    if report.when != "teardown" or not _TORN:
        return
    torn, _TORN[:] = list(_TORN), []
    if not report.failed:
        for roots, request, failed in torn:
            _remove(roots, request, failed)


def pytest_sessionfinish(session: pytest.Session) -> None:
    """A fixture finalized outside a test's teardown goes only if nothing failed while it lived."""
    torn, _TORN[:] = list(_TORN), []
    for roots, request, failed in torn:
        _remove(roots, request, failed)


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(fixturedef: Any, request: Any) -> Iterator[Any]:
    roots: list[Path] | None = None
    if fixturedef.scope != "session" and fixturedef.argname != "tmp_path":
        _record(request.config._tmp_path_factory)
        roots, failed = [], request.session.testsfailed
        request.addfinalizer(
            lambda: _TORN.append((roots, request, failed))
        )  # after its own teardown
    _ROOTS.append(roots)
    try:
        return (yield)
    finally:
        _ROOTS.pop()
