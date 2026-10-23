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

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import pytest

from alphalattice.interface.local_application.cli_contract import AGENT_SESSION_VARIABLES
from devtools import evaluation
from devtools.architecture.evidence_roots import EvidenceRoots
from devtools.evaluation.round_trip import (
    AdmissionPolicy,
    RoundTripObserver,
    TruthFacts,
)
from devtools.evaluation.round_trip import (
    plan_admission_problem as observed_plan_admission_problem,
)
from tests.pytest_environment import configure_temporary_roots, project_scope

pytest_plugins = ["devtools.architecture.registries"]


@pytest.fixture(autouse=True)
def project_discovery(tmp_path, golden_cache):
    """Every test scopes project discovery and rejects outside reads without changing cwd."""
    with project_scope(tmp_path, golden_cache):
        yield


@pytest.fixture(scope="session")
def golden_cache(tmp_path_factory):
    """Workers bind acquisition and discovery to the same invocation, never another run."""
    # Old-source counterfactuals retain their original per-worker discovery scope.
    with getattr(evaluation, "golden_run", nullcontext)(tmp_path_factory.getbasetemp()) as root:
        yield root


@pytest.fixture
def setup_done(tmp_path, monkeypatch):
    """A checkout whose first-use setup is done (`setup_support.setup_done`), in this process
    and in the environment its subprocesses inherit."""
    from tests.setup_support import setup_done as done

    with done(tmp_path) as env:
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        yield env


@pytest.fixture
def host_local_clock(monkeypatch):
    """A Host in New York after local evening has crossed midnight in UTC."""
    from datetime import UTC, datetime
    from zoneinfo import ZoneInfo

    from alphalattice.protocols.actor_execution import bundles

    class HostDatetime(datetime):
        def astimezone(self, tz=None):
            return super().astimezone(tz or ZoneInfo("America/New_York"))

    monkeypatch.setattr(bundles, "datetime", HostDatetime, raising=False)
    return HostDatetime(2026, 10, 18, 2, 44, tzinfo=UTC)


# Every test runs outside an agent session: a vendor's session variable would otherwise ride
# into each request the tests send, and into its activity (LAWS OP13). A test that needs one
# sets it itself. Popped at import, so each worker and the commands it starts inherit none.
_MAINTAINER_THREAD = os.environ.get("CODEX_THREAD_ID") or os.environ.get("CLAUDE_CODE_SESSION_ID")
for _vendor, _variable in AGENT_SESSION_VARIABLES:
    os.environ.pop(_variable, None)


@pytest.fixture
def maintainer_thread(request):
    """Managed test runs retain their caller's owner without attributing product operations."""
    worker = getattr(request.config, "workerinput", {})
    return worker.get("maintainer_thread", _MAINTAINER_THREAD)


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


_ROUND_TRIP_REPORT = os.environ.get("ALPHALATTICE_ROUND_TRIP_REPORT")
"""Discovery: each offer is sent back whole, a preview's body included, and what it finds is
written to this file. A sent preview records its link, so this never runs by default."""
_observer = RoundTripObserver(report=bool(_ROUND_TRIP_REPORT))


def truth_problems(host, operation, body, admits):
    """What an answer says against its Task's record or its own offers (ANSWER-TRUTH).

    A stopped Task's answer, by its record, offers no wait on it; each recovery-bound preview it
    offers, sent back as it stands, passes its owner's admission check (`admits`, which writes
    nothing), never refused as not offered or invalid; and an update offered under a dated first
    use keeps that date.
    """
    from uuid import UUID

    from alphalattice.interface.local_application.answers import stopped_problem
    from alphalattice.interface.local_application.cli_contract import offered_requests
    from alphalattice.interface.local_application.failure_codes import public_failure
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchRequestDocument,
    )

    named = body.get("task_id") or body.get("blocking_task_id")
    try:
        actual = host.workspace_session.task_control_registry.task(UUID(str(named))).lifecycle
    except (AttributeError, LookupError, TypeError, ValueError):
        actual = None
    # A preview that is its own admission writes its link while checked: it is left out.
    previews = {item.preview for item in host.replans().values() if item.preview != item.admitting}
    yield from _observer.truth(
        operation,
        body,
        TruthFacts(lifecycle=None if actual is None else actual.value, previews=previews),
        offered_requests=offered_requests,
        stopped_problem=stopped_problem,
        make_request=lambda offer: PortfolioResearchRequestDocument.model_validate(
            offer
        ).to_operation_request(),
        admits=admits,
        failure_code=lambda error: public_failure(error, "operation.raised"),
        dated_facts=lambda: _dated_truth_facts(host),
    )


def _dated_truth_facts(host):
    """Read delegation after admissions, which report mode can execute."""
    from alphalattice.interface.local_application.goals import target_sessions

    first_use = getattr(getattr(host, "goals", None), "first_use", lambda: None)()
    date = None if first_use is None else first_use.declaration.target_date
    if date is not None and host.goals.first_use_delegation(first_use)["active"]:
        try:
            formation = str(target_sessions(date)["formation_session"])
        except ValueError:
            formation = None
        return "RESEARCH_UPDATE_PLAN", formation
    return None, None


def plan_admission_problem(operation, body, replans, routes):
    """Hold observed re-PLAN answers to the composed owners' admission declarations."""
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

    return observed_plan_admission_problem(
        operation,
        body,
        ((item.preview, item.admitting) for item in replans.values()),
        routes,
        AdmissionPolicy(
            confirmation_admissions={
                ("DATA_UPDATE_PLAN", "CONFIRMATION_REQUIRED"): {"DATA_CHANGE_CONFIRM"}
            },
            existing_work_states={
                "WORKSPACE_PREPARE_PLAN": {state.value for state in TaskLifecycle},
                "RESEARCH_INPUT_PLAN": {
                    "QUEUED",
                    "RUNNING",
                    "DEFERRED",
                    "REVIEW_PENDING",
                    "CANCEL_REQUESTED",
                    "RECOVERY_REQUIRED",
                },
            },
            no_plan_states={
                ("RESEARCH_INPUT_PLAN", "REUSED_EXACT"),
                ("WORKSPACE_PREPARE_PLAN", "ALREADY_PREPARED"),
                ("RESEARCH_UPDATE_PLAN", "OBSERVATIONS_PENDING"),
                ("PORTFOLIO_UPDATE_PLAN", "OBSERVATIONS_PENDING"),
                ("EVIDENCE_PREVIEW", "EVIDENCE_PREREQUISITES_MISSING"),
            },
            runnable_states={"PLANNED", "CONFIRMATION_REQUIRED", "DEFINITION_PLANNED", "AVAILABLE"},
            inaccessible_confirmation=("WORKSPACE_PREPARE_PLAN", "CONFIRMATION_REQUIRED"),
        ),
        offered_requests=offered_requests,
        request_problem=request_problem,
        choices=choices,
        validate_request=PortfolioResearchRequestDocument.model_validate,
        refused=refused,
    )


@pytest.fixture
def plan_admission_check():
    """The same whole-class observer used by every executed operation."""
    return plan_admission_problem


@pytest.fixture
def truth_check():
    """The answer-truth observer every executed operation meets, for its planted cases."""
    return truth_problems


def _host_plan_admission(host, owner, operation, body):
    """Project the observed composer's public declarations and session routes."""
    replans = host.replans()
    if not any(
        item.preview == operation and item.preview != item.admitting for item in replans.values()
    ):
        return None
    # Load the composer beside the observed Host only for its previews. A static route-module
    # import would select every test for any owner Local Web composes (TE7).
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
    return plan_admission_problem(operation, body, replans, routes)


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
                _observer.raised(
                    request.operation,
                    code,
                    untyped_failure=untyped_failure,
                    refusal_words=refusal_words,
                )
                raise
            given = (
                {item.name: getattr(request, item.name) for item in fields(request)}
                if is_dataclass(request)
                else {}
            )

            def truth():
                admits = self._recovery_context
                if _ROUND_TRIP_REPORT:
                    admits = lambda sent: original(self, sent)  # noqa: E731
                return truth_problems(self, request.operation, body, admits)

            _observer.answered(
                request.operation,
                body,
                given,
                answer_problem=answer_problem,
                untyped_failure=untyped_failure,
                truth=truth,
                plan_admission=lambda: _host_plan_admission(self, owner, request.operation, body),
            )
            return body

        execute.held = 1  # type: ignore[attr-defined]
        owner.execute = execute
    yield
    report = os.environ.get("ALPHALATTICE_UNWORDED_REPORT")
    observed = _observer.finish(
        allow_untyped=request.node.get_closest_marker("untyped_failure") is not None,
        report_unworded=bool(report),
    )
    if _ROUND_TRIP_REPORT and observed.trips:
        with open(_ROUND_TRIP_REPORT, "a", encoding="utf-8") as kept:
            kept.writelines(
                f"{issue['reason']}\t{request.node.nodeid}\n" for issue in observed.trips
            )
    if observed.failure_kind == "untyped":
        pytest.fail(
            "an operation failed without a product code, a crash or a failure naming no rule "
            "(V449; a test that causes one on purpose is marked `untyped_failure`): "
            + "; ".join(issue["reason"] for issue in observed.failures[:5]),
            pytrace=False,
        )
    if observed.unworded and report:
        # Discovery: every such code a run meets, written down rather than failed (V456).
        with open(report, "a", encoding="utf-8") as kept:
            kept.writelines(
                f"{issue['reason']}\t{request.node.nodeid}\n" for issue in observed.unworded
            )
    if observed.failure_kind == "answer":
        pytest.fail(
            "an answer does not hold its published model, offers a request the Host does not "
            "accept as it stands, or refuses without words and a way on (V410, V449, OP4): "
            + "; ".join(issue["reason"] for issue in observed.failures[:5]),
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
    try:
        configure_temporary_roots(checkout, config.getoption("basetemp"))
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
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
    node.workerinput["maintainer_thread"] = _MAINTAINER_THREAD  # type: ignore[attr-defined]
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
