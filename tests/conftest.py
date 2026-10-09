"""Shared fixtures for every case directory: the real-evidence lane's root resolver, and the
order a parallel run takes; and every operation's answer a test receives, held to the model
`alphalattice schema show` publishes for it (V410).

Deliberately tiny. Directory-level ``conftest.py`` files keep their own session
fixtures; this one only exposes the typed evidence-root declaration so a test
that needs a dogfood artifact asks for it by name and skips with the reason the
declaration gives, instead of spelling a path of its own. A run records each
test's seconds, and an xdist run starts its longest modules first (TT, V289).
"""

from __future__ import annotations

import getpass
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import pytest

from alphalattice.interface.local_application.cli_contract import AGENT_SESSION_VARIABLES
from devtools.architecture.evidence_roots import EvidenceRoots

# Every test runs outside an agent session: a vendor's session variable would otherwise ride
# into each request the tests send, and into its activity (LAWS OP13). A test that needs one
# sets it itself. Popped at import, so each worker and the commands it starts inherit none.
for _vendor, _variable in AGENT_SESSION_VARIABLES:
    os.environ.pop(_variable, None)


class EvidenceRootRequests:
    """``require`` skips with the declared reason; ``path`` merely answers."""

    def __init__(self, roots: EvidenceRoots) -> None:
        self._roots = roots

    def path(self, name: str) -> Path | None:
        return self._roots.path(name)

    def require(self, name: str) -> Path:
        path = self._roots.path(name)
        if path is None:
            pytest.skip(self._roots.describe(name))
        return path


@pytest.fixture(scope="session")
def evidence_roots() -> EvidenceRootRequests:
    return EvidenceRootRequests(EvidenceRoots.load())


_unheld: list[str] = []
"""Each answer this test received that its published model does not hold (V410)."""
_untyped: list[str] = []
"""Each failure without a product code an operation gave this test, answered or raised (V449)."""
_unworded: list[str] = []
"""Each refusal an operation raised whose code the Host answers without words (V456)."""


def plan_admission_problem(operation, body, replans, routes):
    """Hold observed re-PLAN answers to the composed owners' admission declarations."""
    from uuid import UUID

    from alphalattice.control.task_control.contracts import TaskLifecycle
    from alphalattice.interface.local_application.cli_contract import (
        choices,
        offered_requests,
        refused,
        request_problem,
    )
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchRequestDocument,
    )

    expected = {
        item.admitting
        for item in replans.values()
        if item.preview == operation and item.preview != item.admitting
    }
    if not expected or not isinstance(body, dict):
        return None
    status = body.get("status")
    if operation == "DATA_UPDATE_PLAN" and status == "CONFIRMATION_REQUIRED":
        expected = {"DATA_CHANGE_CONFIRM"}  # Human consent precedes the existing run.
    offers = [q for q in offered_requests(body).values() if q.get("operation") in expected]
    if not offers:
        existing_work = False
        if operation in {"RESEARCH_INPUT_PLAN", "WORKSPACE_PREPARE_PLAN"}:
            try:
                UUID(str(body.get("task_id")))
                existing_work = status in (
                    {state.value for state in TaskLifecycle}
                    if operation == "WORKSPACE_PREPARE_PLAN"
                    else {
                        "QUEUED",
                        "RUNNING",
                        "DEFERRED",
                        "REVIEW_PENDING",
                        "CANCEL_REQUESTED",
                        "RECOVERY_REQUIRED",
                    }
                )
            except ValueError:
                pass
        # These public states have no new runnable plan. Unknown states fail closed;
        # deleting both a PLANNED answer's hash and offer cannot escape this check.
        if (
            (
                refused(body)
                and status
                not in {
                    "PLANNED",
                    "CONFIRMATION_REQUIRED",
                    "DEFINITION_PLANNED",
                    "AVAILABLE",
                }
            )
            or (operation, status)
            in {
                ("RESEARCH_INPUT_PLAN", "REUSED_EXACT"),
                ("WORKSPACE_PREPARE_PLAN", "ALREADY_PREPARED"),
                ("RESEARCH_UPDATE_PLAN", "OBSERVATIONS_PENDING"),
                ("PORTFOLIO_UPDATE_PLAN", "OBSERVATIONS_PENDING"),
            }
            or (
                operation == "WORKSPACE_PREPARE_PLAN"
                and status == "CONFIRMATION_REQUIRED"
                and body.get("confirmation_available") is False
                and body.get("source_access_failure")
            )
            or (operation == "EVIDENCE_PREVIEW" and status == "EVIDENCE_PREREQUISITES_MISSING")
            or existing_work
        ):
            return None
        return f"{operation} offers no filled admission among {sorted(expected)}"
    for offer in offers:
        problem = request_problem(offer)
        if problem or choices(offer):
            return f"{operation} admission is unfilled or invalid: {problem or choices(offer)}"
        PortfolioResearchRequestDocument.model_validate(offer)
        route = routes.get(offer["operation"])
        if not route or route["method"] != "POST" or not route["path"]:
            return f"{operation} admission has no session POST route: {offer['operation']}"
    return None


@pytest.fixture
def plan_admission_check():
    """The same whole-class observer used by every executed operation."""
    return plan_admission_problem


@pytest.fixture(autouse=True)
def _memory_checks_read_no_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The early memory refusal reads unknown memory, and skips, unless a test sets it.

    The refusal compares a plan's estimate with the machine's free memory. A fixture's work
    passes or fails by its own code, never by what else the host runs (PERF-1 item 3: a
    preparation fixture was refused while another heavy set held commit).
    """
    # Found by its class among the loaded modules, as below: any test that can reach the gate
    # loaded it at collection through the Host's composition.
    for module in list(sys.modules.values()):
        gate = getattr(module, "ResourceGate", None)
        if isinstance(gate, type) and gate.__module__ == getattr(module, "__name__", None):
            monkeypatch.setattr(module, "available_work_memory_bytes", lambda: None)
            return


@pytest.fixture(autouse=True)
def _answers_hold_their_models(request: pytest.FixtureRequest) -> Iterator[None]:
    """Every answer an operation gives a test holds the model `schema show` publishes for it.

    Each owner is held to its published reading by every test that runs the operation (V410,
    SC3), at the one door every caller's request passes, `PortfolioResearchOperations.execute`;
    a test that never loaded the Host's operations pays nothing.
    """
    # Found by its class among the loaded modules, never by a quoted module path, which the
    # impact walk reads as every test's dependency: the wrapper reaches only the tests that run
    # an operation, and they select themselves by their own imports.
    owner = next(
        (
            found
            for module in list(sys.modules.values())
            if isinstance(found := getattr(module, "PortfolioResearchOperations", None), type)
            and found.__module__ == getattr(module, "__name__", None)
        ),
        None,
    )
    if owner is not None and not getattr(owner.execute, "held", 0):
        from alphalattice.interface.local_application.answers import answer_problem
        from alphalattice.interface.local_application.cli_contract import refusal_words
        from alphalattice.interface.local_application.failure_codes import (
            public_failure,
            untyped_failure,
        )

        original = owner.execute

        def execute(self: Any, request: Any, **kwargs: Any) -> Any:
            try:
                body = original(self, request, **kwargs)
            except Exception as error:
                # A raised failure without a product code is a crash the caller meets as a
                # bare fingerprint (V449: AX15's review IndexError).
                code = public_failure(error, "operation.raised")
                if untyped_failure(code):
                    _untyped.append(f"{request.operation} raised {code}")
                elif refusal_words(code) is None:
                    # The Host answers a raised refusal with its code's words from the table
                    # (OP4); one without them reaches the caller as a bare code (V456).
                    _unworded.append(f"{request.operation} raised {code}")
                raise
            given = (
                {item.name: getattr(request, item.name) for item in fields(request)}
                if is_dataclass(request)
                else {}
            )
            problem = answer_problem(request.operation, body, given)
            if problem is not None:
                _unheld.append(problem)
            replans = self.replans()
            if any(
                item.preview == request.operation and item.preview != item.admitting
                for item in replans.values()
            ):
                # Load the composer beside the observed Host only for its previews.
                # A static route-module import here would make the impact walk
                # select every test for any owner that Local Web composes (TE7).
                import importlib

                session_module = importlib.import_module(
                    owner.__module__.rsplit(".", 1)[0] + ".local_web_session"
                )
                routes = {
                    op: {"method": method, "path": path}
                    for method, path, op in (
                        *session_module.OPERATION_ROUTES,
                        *session_module.HANDLED_OPERATION_ROUTES,
                    )
                }
                if problem := plan_admission_problem(request.operation, body, replans, routes):
                    _unheld.append(problem)
            if isinstance(body, dict):
                code = str(body.get("failure_code") or body.get("refused") or "")
                if untyped_failure(code):
                    _untyped.append(f"{request.operation} answered {code}")
            return body

        execute.held = 1  # type: ignore[attr-defined]
        owner.execute = execute
    yield
    untyped = list(dict.fromkeys(_untyped))
    _untyped.clear()
    if untyped and request.node.get_closest_marker("untyped_failure") is None:
        _unheld.clear()
        pytest.fail(
            "an operation failed without a product code, a crash or a failure naming no rule "
            "(V449; a test that causes one on purpose is marked `untyped_failure`): "
            + "; ".join(untyped[:5]),
            pytrace=False,
        )
    unworded = list(dict.fromkeys(_unworded))
    _unworded.clear()
    report = os.environ.get("ALPHALATTICE_UNWORDED_REPORT")
    if unworded and report:
        # Discovery: every such code a run meets, written down rather than failed (V456).
        with open(report, "a", encoding="utf-8") as kept:
            kept.writelines(f"{line}\t{request.node.nodeid}\n" for line in unworded)
    elif unworded:
        _unheld.extend(f"{line} without words or a way on" for line in unworded)
    if _unheld:
        found = list(dict.fromkeys(_unheld))
        _unheld.clear()
        pytest.fail(
            "an answer does not hold its published model, offers a request the Host does not "
            "accept as it stands, or refuses without words and a way on (V410, V449, OP4): "
            + "; ".join(found[:5]),
            pytrace=False,
        )


_DURATIONS = Path(__file__).resolve().parents[1] / "tmp" / "test-durations.json"
"""Each test's last recorded seconds, setup to teardown, by node id. A hint: a lost or stale file
costs only the order of the next parallel run."""
_WORKER = "PYTEST_XDIST_WORKER"
_measured: dict[str, float] = {}


def _recorded_durations() -> dict[str, float]:
    try:
        payload = json.loads(_DURATIONS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): float(v) for k, v in payload.items() if isinstance(v, (int, float))}


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Refuse checkout temp before collection; keep the workers' longest-first queue."""
    checkout = Path(__file__).resolve().parents[1]
    basetemp = config.getoption("basetemp")
    if basetemp is not None:
        roots = {"--basetemp": Path(os.path.abspath(str(basetemp)))}
    else:
        # pytest 9.1.1 uses this root, then pytest-of-<user>/pytest-<number>.
        # Resolve its existing parents without getbasetemp(): that method deletes
        # an explicit basetemp, and xdist calls it before workers collect.
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
        resolved = path.resolve()
        if resolved.is_relative_to(checkout):
            raise pytest.UsageError(
                f"basetemp_inside_checkout: {source} resolves to {resolved}, inside checkout "
                f"{checkout}. Refused before collection: checkout artifacts and hooks would "
                "contaminate test results. Use --basetemp with an absolute directory outside "
                "the checkout, or move PYTEST_DEBUG_TEMPROOT / TMPDIR / TEMP / TMP outside it."
            )
    # A data update backs the held state up outside the workspace (V209): every test, and
    # each command it starts, uses a temporary root, never the person's application data.
    backup_parent = Path(tempfile.gettempdir()).resolve()
    if backup_parent.is_relative_to(checkout):
        # An explicit external basetemp can override a checkout TEMP. Keep backup
        # roots beside that safe base, outside the directory pytest clears.
        backup_parent = (
            Path(basetemp).resolve().parent if basetemp is not None else temproot.resolve()
        )
    os.environ["ALPHALATTICE_BACKUP_ROOT"] = tempfile.mkdtemp(
        prefix="alphalattice-test-backups-", dir=backup_parent
    )
    if (
        config.getoption("dist", "no") == "load"
        and config.getoption("maxschedchunk", None) is None
        and _recorded_durations()
    ):
        # xdist otherwise reserves a large consecutive batch on each worker.
        # Two is its minimum prefetch; explicit operator settings stay intact.
        config.option.maxschedchunk = 2


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: object) -> None:
    """xdist, in the main process: every worker sorts by the one reading taken here, so their
    collections agree even when another run rewrites the file meanwhile."""
    node.workerinput["recorded_durations"] = _recorded_durations()  # type: ignore[attr-defined]
    # xdist sets config.option.dist to "no" inside a worker. Preserve the
    # controller's mode alongside the snapshot its collection uses.
    node.workerinput["recorded_distribution"] = node.config.getoption("dist", "no")  # type: ignore[attr-defined]
    workers = node.config.getoption("numprocesses", 0)  # type: ignore[attr-defined]
    batch = node.config.getoption("maxschedchunk", None)  # type: ignore[attr-defined]
    node.workerinput.update(recorded_workers=workers, recorded_batch=batch)  # type: ignore[attr-defined]


def pytest_sessionstart(session: pytest.Session) -> None:
    """Every Local Web reader shares current assets before workers collect or start a Host.

    The tests' root is their narrowest common ancestor: other case directories also start
    the Host. Only the controller runs this prerequisite, so xdist workers never race a
    build. A read-only check decides freshness, including a removed output; no old marker
    or manifest alone can stand in for the actual files.
    """
    if hasattr(session.config, "workerinput") or session.config.option.collectonly:
        return
    checkout = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(checkout / "scripts/build_local_web_ui.py"), "--product"]

    def run(*flags: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [*command, *flags],
            cwd=checkout,
            capture_output=True,
            encoding="utf-8",
            check=False,
        )
        if result.returncode not in {0, 1} or (result.returncode == 1 and not flags):
            raise pytest.UsageError(
                "The Local Web prerequisite failed:\n" + result.stdout + result.stderr
            )
        return result

    checked = run("--check")
    if checked.returncode == 1:
        run()
        checked = run("--check")
        if checked.returncode:
            raise pytest.UsageError(
                "The Local Web build did not produce current outputs:\n"
                + checked.stdout
                + checked.stderr
            )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Start the longest work first, at the scheduler's own unit of dispatch.

    Default ``--dist load`` dispatches individual tests. File/scope schedulers
    retain each file's order and start its aggregate work first.

    Every worker receives one timing snapshot before collection. The timing
    sort retains equal-cost order, then balances load's first two-case batches;
    absent timings retain the scheduler's default.
    Shared immutable prerequisites belong in the owning test package's session
    support, including when load dispatches their consumers across workers.
    """
    workerinput = getattr(config, "workerinput", {})
    recorded = workerinput.get("recorded_durations")
    distribution = workerinput.get("recorded_distribution", config.getoption("dist", "no"))
    if not recorded or distribution not in {"load", "loadfile", "loadscope"}:
        return
    if distribution == "load":
        items.sort(key=lambda item: -recorded.get(item.nodeid, 0.0))
        workers = workerinput.get("recorded_workers", 0)
        if (
            workerinput.get("recorded_batch") == 2
            and isinstance(workers, int)
            and workers > 1
            and len(items) >= 2 * workers
        ):
            # load reserves two consecutive cases on each worker initially.
            # Pair the longest with the shortest of that first wave, rather
            # than trapping both longest cases behind the same worker.
            wave = items[: 2 * workers]
            items[: 2 * workers] = [
                item for index in range(workers) for item in (wave[index], wave[-1 - index])
            ]
        return
    modules: dict[str, list[pytest.Item]] = {}
    for item in items:
        modules.setdefault(item.nodeid.split("::", 1)[0], []).append(item)
    cost = {name: sum(recorded.get(i.nodeid, 0.0) for i in its) for name, its in modules.items()}
    items[:] = [item for name in sorted(modules, key=lambda n: -cost[n]) for item in modules[name]]


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """The main process adds each phase's seconds; a worker's reports reach it too."""
    if not os.environ.get(_WORKER):
        _measured[report.nodeid] = _measured.get(report.nodeid, 0.0) + report.duration


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """The main process merges what this run measured into the recorded seconds."""
    if os.environ.get(_WORKER) or not _measured:
        return
    merged = _recorded_durations()
    merged.update({name: round(seconds, 3) for name, seconds in _measured.items()})
    staged = _DURATIONS.with_name(f"{_DURATIONS.name}.{os.getpid()}")
    try:
        _DURATIONS.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text(json.dumps(dict(sorted(merged.items())), indent=0) + "\n", "utf-8")
        os.replace(staged, _DURATIONS)
    except OSError:
        return
