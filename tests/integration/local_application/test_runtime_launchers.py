"""The installed entry composes product owners without checkout script imports."""

from __future__ import annotations

import builtins
import contextlib
import ctypes
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from io import BytesIO, StringIO, TextIOWrapper
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphalattice.control.product_host.composition import saved_object_readback
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
    publish_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.web_launcher import OPEN_THE_WORKBENCH
from alphalattice.interface.local_application import client as local_client
from alphalattice.kernel.shared_kernel.clocks import ReplayClock
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root, runtime_json

WINDOWS = pytest.mark.skipif(sys.platform != "win32", reason="Windows process control")
_FLAGS = 0x08004000 if sys.platform == "win32" else 0  # no window, below normal
_PYTHON = str(Path(sys.base_prefix) / "python.exe") if sys.platform == "win32" else sys.executable


def _private_bytes(process):
    """The process's private commit on Windows, else its resident set size.

    psutil gives private commit only on Windows. Resident set size is a different measure (it
    counts shared pages and leaves out paged-out ones), so off Windows the bound is a proxy.
    """
    info = process.memory_info()
    return info.private if hasattr(info, "private") else info.rss


def _job(flags):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = ctypes.c_void_p
    job = ctypes.c_void_p(kernel.CreateJobObjectW(None, None))
    limits = (ctypes.c_byte * 144)()
    ctypes.c_uint32.from_buffer(limits, 16).value = flags
    assert job and kernel.SetInformationJobObject(job, 9, limits, 144)
    assert kernel.QueryInformationJobObject(job, 9, limits, 144, None)
    assert ctypes.c_uint32.from_buffer(limits, 16).value == flags
    current, inside = ctypes.c_void_p(-1), ctypes.c_int()
    assert kernel.AssignProcessToJobObject(job, current)
    assert kernel.IsProcessInJob(current, job, ctypes.byref(inside)) and inside.value
    return kernel, job


@pytest.fixture(scope="module", autouse=True)
def runner_job():
    """The same nodes also run inside a Job that denies every breakaway attempt."""
    held = _job(0) if os.name == "nt" and os.environ.get("ALPHALATTICE_TEST_NO_BREAKAWAY") else None
    yield
    if held:
        assert held[0].CloseHandle(held[1])


def _can_escape():
    if sys.platform != "win32":
        return True
    try:
        with subprocess.Popen(
            [_PYTHON, "-c", "import sys; sys.stdin.read()"],
            stdin=subprocess.PIPE,
            creationflags=_FLAGS | subprocess.CREATE_BREAKAWAY_FROM_JOB,
        ) as child:
            inside = ctypes.c_int()
            handle = ctypes.c_void_p(int(child._handle))
            proven = ctypes.windll.kernel32.IsProcessInJob(handle, None, ctypes.byref(inside))
            return bool(proven) and not inside.value
    except PermissionError:
        return False


def test_product_probe_reopens_a_workspace_and_reads_saved_object_indexes(tmp_path: Path) -> None:
    """Requirement: sandbox readback starts actual product sessions with no harness."""
    harvest = tmp_path / "harvest"
    workspace = harvest / "workspace"
    publish_research_workspace_manifest(workspace, ResearchWorkspaceManifest.research_only("probe"))
    out = tmp_path / "readback.json"
    work = tmp_path / "copies"
    saved_object_readback.probe(harvest, work, out)
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["workspaces"] == 1
    by_op = {row["op"]: row for row in receipt["rows"]}
    assert set(by_op) == {"OPEN", "TASKS", "RESULTS", "FEATURE_TRIALS"}
    assert {op: row["verdict"] for op, row in by_op.items()} == {
        "OPEN": "OPENS",
        "TASKS": "OPENS",
        "FEATURE_TRIALS": "OPENS",
        "RESULTS": "REFUSES research_workspace.strategy_not_installed",
    }
    assert not work.exists()
    assert (workspace / "research-workspace.json").is_file()


_WATCHED_STDIN = """
import subprocess, sys, threading
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/src"]
from scripts.run_local_portfolio_web import _wait_for_stop_on_stdin
watcher = threading.Thread(target=_wait_for_stop_on_stdin)
watcher.start()
# A process started while the Host waits for its stop inherits the stdin pipe (W10's worker).
subprocess.run([sys.executable, "-c", "pass"], check=True, timeout=60)
print("started", flush=True)
watcher.join()
"""


@pytest.mark.parametrize("during", [False, True])  # the stop after the start, or during it
@pytest.mark.parametrize("lines", ["stop\n", "ignored\nstop\n", ""])
def test_attached_launcher_stops_on_explicit_stdin_or_eof(
    tmp_path, monkeypatch, capsys, lines, during
):

    from scripts import run_local_portfolio_web

    from alphalattice.control.product_host.composition import web_launcher
    from alphalattice.interface.local_application.cli import main

    calls, launched = [], threading.Event()

    class Input(StringIO):  # the stop arrives once the launch lines are out, or during the start
        def __next__(self):
            launched.wait(0 if during else 30)
            return super().__next__()

    def shown_then_stoppable(*text, **options):
        builtins.print(*text, **options)
        if text == (OPEN_THE_WORKBENCH,):
            launched.set()

    monkeypatch.setattr(web_launcher, "print", shown_then_stoppable, raising=False)

    class Session:
        launch_url = "http://127.0.0.1:12345/launch?key=k"  # printed and opened (HB)
        operations = None
        dispatcher = None
        stop_requested = staticmethod(bool)

        def start(self):
            if during:  # a slow start: the stop, read from the start, is seen inside it
                deadline = time.monotonic() + 30
                while not self.stop_requested() and time.monotonic() < deadline:
                    time.sleep(0.01)
                calls.append("stopped" if self.stop_requested() else "missed")
            root = resolve_playpen_root(Path(__file__))
            host = subprocess.Popen(
                [sys.executable, "-c", _WATCHED_STDIN, str(root)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            assert host.stdout is not None and host.stdin is not None
            assert host.stdout.readline().strip() == "started"
            host.stdin.write("noise\n" + lines)
            host.stdin.close()
            assert host.wait(timeout=30) == 0
            calls.append("start")
            return "http://127.0.0.1:12345/"

        def stop(self):
            calls.append("stop")

    launcher = run_local_portfolio_web.LocalPortfolioWebSession
    monkeypatch.setattr(launcher, "from_workspace", lambda *a, **k: Session())
    monkeypatch.setattr(sys, "stdin", Input(lines))
    args = ["--workspace", str(tmp_path), "serve", "--no-browser", "--stop-on-stdin"]
    assert main(args, serve=run_local_portfolio_web.main) == 0
    assert calls == (["stopped", "start", "stop"] if during else ["start", "stop"])
    shown = capsys.readouterr().out.splitlines()
    if during:  # stopped while starting: no launch line, no browser, and still stopped
        assert shown == []
    else:
        assert shown[0].endswith(Session.launch_url) and shown[2] == OPEN_THE_WORKBENCH


@WINDOWS
def test_attached_retrieval_host_ends_when_its_cli_parent_is_terminated(tmp_path):
    """A forced retrieval-forwarded Host and its children die with their CLI parent."""
    import threading

    import psutil

    from alphalattice.interface.local_application.client import LocalResearchClient
    from alphalattice.interface.local_application.native_bridge import (
        PROJECT_DECLARATION_NAME,
        PROJECT_DECLARATION_SCHEMA,
    )

    workspace = tmp_path / "workspace"
    project = tmp_path / ".codex"
    project.mkdir()
    (project / PROJECT_DECLARATION_NAME).write_text(
        json.dumps({"schema": PROJECT_DECLARATION_SCHEMA, "host": "codex"}), encoding="utf-8"
    )
    root = resolve_playpen_root(Path(__file__))
    script = """
import importlib.util, sys
from __SELF_TEST_MODULE__ import _job
kernel, job = _job(0)
original = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a, **k: (
    None if name == 'fastembed' else original(name, *a, **k)
)
from pathlib import Path
from alphalattice.control.product_host.composition import entry
entry.interpreter_path = lambda: Path(sys.executable)
raise SystemExit(entry.main(sys.argv[1:]))
""".replace("__SELF_TEST_MODULE__", __spec__.name if __spec__ else __name__)
    sources = (str(root), str(root / "src"), str(Path(sys.prefix) / "Lib/site-packages"))
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sources))
    arguments = ["--workspace", str(workspace), "serve", "--no-browser", "--stop-on-stdin"]
    parent = subprocess.Popen(
        [_PYTHON, "-c", script, *arguments],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        cwd=tmp_path,
        creationflags=_FLAGS,
    )
    assert parent.stdout is not None and parent.stdin is not None
    ready, lines = threading.Event(), []

    def read_ready():
        for line in parent.stdout:
            lines.append(line)
            if OPEN_THE_WORKBENCH in line:
                ready.set()

    reader = threading.Thread(target=read_ready, daemon=True)
    reader.start()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    owned = []

    def capture():
        for child in psutil.Process(parent.pid).children(recursive=True):
            handle = ctypes.c_void_p(kernel.OpenProcess(0x100001, False, child.pid))
            assert handle
            owned.append((child.pid, handle))

    try:
        assert ready.wait(45), "".join(lines)
        host = LocalResearchClient(workspace).activity(limit=1)["observer"]["host"]
        capture()
        assert host["pid"] in {pid for pid, _ in owned}
        parent.terminate()
        parent.wait(timeout=5)
        assert not parent.stdin.closed
        for _, handle in owned:
            assert kernel.WaitForSingleObject(handle, 5000) == 0
    finally:
        if not owned:
            with contextlib.suppress(psutil.NoSuchProcess):
                capture()
        with contextlib.suppress(OSError):
            parent.terminate()
        parent.wait(timeout=5)
        for _, handle in owned:
            kernel.TerminateProcess(handle, 1)
            kernel.WaitForSingleObject(handle, 5000)
            kernel.CloseHandle(handle)
        parent.stdin.close()
        reader.join(timeout=5)
        parent.stdout.close()


@WINDOWS
@pytest.mark.parametrize("method", ["legacy", "attached"])
@pytest.mark.parametrize("noninheritable", [False, True, "collision"])
def test_attached_child_preserves_streams_and_refuses_unrelated_handles(
    tmp_path, method, noninheritable
):
    stray = tmp_path / "unrelated"
    stray.write_bytes(b"unrelated")
    child = """
import _winapi, ctypes, json, msvcrt, os, sys
from ctypes import wintypes

current = _winapi.GetCurrentProcess()
handle = msvcrt.get_osfhandle(sys.stdin.fileno()) if sys.argv[2] == "stdin" else int(sys.argv[2])
try:
    candidate = _winapi.DuplicateHandle(
        current, handle, current, 0, False, _winapi.DUPLICATE_SAME_ACCESS
    )
except OSError as error:
    if error.winerror not in (6, 50):  # absent or an object that cannot be duplicated
        raise
    candidate = None
# Pin the candidate before opening the parent: opening new handles can reuse its number.
parent = _winapi.OpenProcess(_winapi.PROCESS_DUP_HANDLE, False, int(sys.argv[3]))
try:
    original = _winapi.DuplicateHandle(
        parent, int(sys.argv[1]), current, 0, False, _winapi.DUPLICATE_SAME_ACCESS
    )
finally:
    _winapi.CloseHandle(parent)
try:
    compare = ctypes.WinDLL('kernelbase', use_last_error=True).CompareObjectHandles
    compare.argtypes, compare.restype = (wintypes.HANDLE, wintypes.HANDLE), wintypes.BOOL
    control = _winapi.DuplicateHandle(
        current, original, current, 0, False, _winapi.DUPLICATE_SAME_ACCESS
    )
    try:
        assert compare(original, control)  # duplicated original is detected
    finally:
        _winapi.CloseHandle(control)
    value = bool(candidate is not None and compare(candidate, original))
finally:
    _winapi.CloseHandle(original)
    if candidate is not None:
        _winapi.CloseHandle(candidate)
print(json.dumps(dict(stdin=sys.stdin.read(), stray=value)))
print('stderr-ok', file=sys.stderr)
"""
    program = f"""
import _winapi, msvcrt, os, subprocess, sys
from alphalattice.control.product_host.composition.web_launcher import attached_child
if sys.argv[3] in ('True', 'collision'):
    for kind in (-10, -11, -12):
        os.set_handle_inheritable(_winapi.GetStdHandle(kind), False)
call = subprocess.call if sys.argv[2] == 'legacy' else attached_child
with open(sys.argv[1], 'rb') as stray:
    handle = msvcrt.get_osfhandle(stray.fileno())
    os.set_handle_inheritable(handle, True)
    probe = 'stdin' if sys.argv[3] == 'collision' else str(handle)
    command = [sys.executable, '-c', {child!r}, str(handle), probe, str(os.getpid())]
    raise SystemExit(call(command, env=dict(os.environ)))
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(stray), method, str(noninheritable)],
        input="owned-input\n",
        capture_output=True,
        text=True,
        timeout=20,
        creationflags=_FLAGS,
    )
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout)
    assert seen["stdin"] == "owned-input\n"
    assert seen["stray"] is False
    assert result.stderr == "stderr-ok\n"


IDLE_TIME = SimpleNamespace(ticks=0.0, wall=datetime(2026, 10, 9, tzinfo=UTC))
IDLE_CLOCK = ReplayClock(IDLE_TIME.wall, ticks=lambda: IDLE_TIME.ticks, wall=lambda: IDLE_TIME.wall)


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("live", [IDLE_CLOCK], indirect=True)
def test_a_served_detached_host_expires_only_after_its_held_work_finishes(
    live, monkeypatch, capsys, held, language
):
    from alphalattice.control.product_host.composition import web_launcher
    from alphalattice.interface.local_application import cli_contract, dispatcher
    from tests.portfolio_strategy_lab.local_web_support import _json

    origin = live.dispatcher._last_needed
    monkeypatch.setattr(IDLE_TIME, "ticks", 0.0)
    monkeypatch.setattr(IDLE_TIME, "wall", datetime.now(UTC) + timedelta(days=30))
    monkeypatch.setattr(dispatcher, "monotonic", lambda: origin + IDLE_TIME.ticks)
    owner, ran = live.operations.automation, []
    if held:
        owner.hold(live.clock() + timedelta(hours=7), lambda: ran.append(1))

    class Timer:
        stopped, turn = False, 0

        def set(self):
            self.stopped = True

        def is_set(self):
            return self.stopped

        def wait(self, seconds):
            assert seconds == 1
            if self.stopped:
                return True
            self.turn += 1
            assert self.turn <= (3 if held else 1), "idle.expiry_not_observed"
            IDLE_TIME.ticks += 7 * 3600 if held and self.turn == 2 else 901
            assert _json(live, "/api/session")["workspace_id"] == "qa-local-web"
            if held and self.turn == 2:
                owner.held_host.run_once()
                assert ran == [1]
            return False

    monkeypatch.setenv(web_launcher.LAUNCH_KEY_VARIABLE, "controlled-child-key")
    monkeypatch.setattr(web_launcher, "threading", SimpleNamespace(Event=Timer))
    monkeypatch.setattr(type(live), "from_workspace", lambda *a, **k: live)
    monkeypatch.setattr(type(live), "start", lambda self: live.url)
    token = cli_contract.ANSWER_LANGUAGE.set(language)
    try:
        assert web_launcher.main(["--workspace", str(live.workspace), "--no-browser"]) == 0
    finally:
        cli_contract.ANSWER_LANGUAGE.reset(token)
    words = runtime_json("config/registries/messages.json")["entries"]["host.idle"]["words"]
    sentence = words["zh-CN" if language == "zh" else "en"].format(seconds=900)
    assert sentence in capsys.readouterr().out
    if language == "zh":
        assert any("\u4e00" <= ch <= "\u9fff" for ch in sentence)
    assert live.clock.phase_metadata()["clock_drift"]
    assert ran == ([1] if held else [])
    assert live.session is None and live.web is None


_TURN = """
import ctypes, os, sys
if os.name == "nt":
    from __SELF_TEST_MODULE__ import _job
    k, job = _job(0x2800 if os.environ["ALPHALATTICE_TEST_JOB_MODE"] == "escaped" else 0x2000)
from alphalattice.control.product_host.composition.entry import main
result = main(sys.argv[1:])  # this CLI is inside the verified finite Job; only its Host escapes
if os.name == "nt" and result == 2:
    accounting = (ctypes.c_byte * 48)()
    assert k.QueryInformationJobObject(job, 1, accounting, 48, None)
    assert ctypes.c_uint32.from_buffer(accounting, 40).value == 1  # no held child before Job close
sys.exit(result)
""".replace("__SELF_TEST_MODULE__", __spec__.name if __spec__ else __name__)


def _end_turn_after(command: list[str], root: Path, temp: Path) -> str:
    """A finite real CLI parent owns a verified Job, or a POSIX group; it ends before Host work."""
    escaped = _can_escape()
    env = {
        **os.environ,
        **{n: str(temp) for n in ("TEMP", "TMP", "TMPDIR")},
        "PYTHONPATH": os.pathsep.join(
            (str(root), str(root / "src"), str(Path(sys.prefix) / "Lib/site-packages"))
        ),
        "ALPHALATTICE_TEST_JOB_MODE": "escaped" if escaped else "denied",
    }
    turn = subprocess.Popen(
        [_PYTHON, "-c", _TURN, *command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=env,
        start_new_session=sys.platform != "win32",
        creationflags=_FLAGS
        | (subprocess.CREATE_BREAKAWAY_FROM_JOB if escaped and os.name == "nt" else 0),
    )
    try:
        output, error = turn.communicate(timeout=180)
        assert turn.returncode == (0 if escaped else 2), error or output
        assert output, error
        return output
    finally:
        if sys.platform != "win32":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(turn.pid, signal.SIGKILL)


@pytest.mark.parametrize("workspace_state", ["cp1252"])
def test_a_detached_host_outlives_the_turn_that_started_it(
    tmp_path: Path, workspace_state: str, request, monkeypatch
) -> None:
    """A turn's end kills everything its launch started; the Host `serve --detach` started, on
    a new workspace or an initialized one, serves a later client until `host stop`, with its
    key in no log; a second launch is refused by name and leaves no log."""
    if workspace_state == "cp1252":
        from alphalattice.control.product_host.composition import web_launcher

        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("output spawned"))
        data = BytesIO()
        args = SimpleNamespace(workspace=tmp_path, no_browser=True, pid=1)
        token = web_launcher.ANSWER_LANGUAGE.set("zh")
        try:
            with TextIOWrapper(data, encoding="cp1252") as stream, redirect_stdout(stream):
                assert web_launcher._served(args, "http://127.0.0.1/", args, tmp_path / "h") == 0
                expected = str(web_launcher.worded("host.idle")).format(seconds=900)
                assert expected in data.getvalue().decode("utf-8")
        finally:
            web_launcher.ANSWER_LANGUAGE.reset(token)
        return
    import fnmatch

    import psutil

    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    from alphalattice.control.product_host.composition.entry import main
    from alphalattice.interface.local_application import client

    root = resolve_playpen_root(Path(__file__))
    budgets = runtime_json("config/registries/process-budgets.json")
    host_budget = budgets["kinds"]["host"]
    workspace = tmp_path / "workspace"
    if workspace_state == "initialized":
        manifest = ResearchWorkspaceManifest.research_only("turn")
        publish_research_workspace_manifest(workspace, manifest)

    connection = local_client.LocalResearchConnection.path(workspace)

    def stop_owned():
        if not connection.exists():
            return
        assert main(["--workspace", str(workspace), "host", "stop"]) == 0
        deadline = time.monotonic() + 60
        while connection.exists() and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not connection.exists()

    request.addfinalizer(stop_owned)
    temp = tmp_path / "temp"
    temp.mkdir()
    lang = "zh" if workspace_state == "initialized" else "en"
    launch = ["--lang", lang, "--workspace", str(workspace), "serve", "--no-browser", "--detach"]
    launch[5:5] = ["--port", "0"]
    out = _end_turn_after(launch, root, temp)
    if out.startswith("{"):
        refusal = json.loads(out)
        assert refusal["failure_code"] == "local_web.detached_breakaway_denied"
        assert refusal["launch_context"] == {"win32_error": 5, "job_limit_flags": 0x2000}
        assert not connection.exists() and not list(temp.iterdir())
        return
    assert "Portfolio Research workspace:" in out and "outlives it" in out
    words = runtime_json("config/registries/messages.json")["entries"]["host.idle"]["words"]
    assert words["zh-CN" if lang == "zh" else "en"].format(seconds=900) in out
    started = out.splitlines()[0]  # plainly STARTED, with the workspace and the address
    assert started.startswith("STARTED") and str(workspace.resolve()) in started
    assert out.rstrip().endswith(OPEN_THE_WORKBENCH)  # the step right after the launch
    key = out.split("key=", 1)[1].split("#", 1)[0]
    logs = list(temp.glob("alphalattice-host-*.log"))
    assert len(logs) == 1 and key not in logs[0].read_text(encoding="utf-8")
    reader = client.LocalResearchClient(workspace)
    address = reader.connection.url
    query = {"operation": "OPERATION_LIST", "schema_operation": "HOST_STOP"}
    assert "HOST_STOP" in reader.request(query).get("operation_fields", {})
    assert address in started
    second = json.loads(_end_turn_after(launch, root, temp))  # plainly ALREADY_SERVING
    assert second["status"] == "ALREADY_SERVING" and second["workspace"] in started
    assert (second["address"], second["next_action"]) == (address, "USE_THE_SERVING_HOST")
    assert list(temp.glob("alphalattice-host-*.log")) == logs
    tree = psutil.Process(int(out.split("(process ", 1)[1].split(")", 1)[0]))
    tree = [tree, *tree.children(recursive=True)]
    for process in tree:
        process.cpu_percent(None)
    time.sleep(0.9)  # an idle sample
    commit = sum(_private_bytes(process) for process in tree) / 2**30
    cpu = sum(process.cpu_percent(None) for process in tree)
    assert len(tree) <= host_budget["idle_processes"], [p.name() for p in tree]
    assert commit <= host_budget["idle_commit_gib"] and cpu <= host_budget["idle_cpu_percent"]
    assert _visible_windows({process.pid for process in tree}) == budgets["console_windows"]
    declared = [entry["pattern"] for entry in budgets["declared_temporary_files"]]
    written = [path.name for path in temp.iterdir()]
    assert all(any(fnmatch.fnmatch(name, d) for d in declared) for name in written), written
    assert sorted(path.name for path in tmp_path.iterdir()) == ["temp", "workspace"]
    stop_owned()
    _gone, alive = psutil.wait_procs(tree, timeout=host_budget["stopped_within_seconds"])
    assert not alive, [p.name() for p in alive]


@pytest.mark.parametrize("escape", ["escaped", "denied", "unproven"])
@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("panel_host", [None, "claude-code", "codex"])
def test_a_detached_launch_refuses_what_it_cannot_serve(
    tmp_path, monkeypatch, capsys, escape, lang, panel_host
):
    """Detached refusals keep ownership checks and offer the agent panel before a person act."""
    from alphalattice.control.product_host.composition import web_launcher
    from alphalattice.interface.local_application.cli import main

    monkeypatch.setenv("ALPHALATTICE_TERMINAL_PANEL_HOST", panel_host or "")
    for name in ("TEMP", "TMP", "TMPDIR"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.setattr(web_launcher.tempfile, "tempdir", None)
    monkeypatch.delenv("PYTHONFAULTHANDLER", raising=False)
    native, spawned, flags, host_fault_handlers = subprocess.Popen, [], [], []

    def spawn(*args, **kwargs):
        if "alphalattice.control.product_host.composition.web_launcher" in args[0]:
            host_fault_handlers.append(kwargs["env"].get("PYTHONFAULTHANDLER"))
        flags.append(kwargs.get("creationflags", 0))
        kwargs["creationflags"] = flags[-1] & ~getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
        spawned.append(native(*args, **kwargs))
        return spawned[-1]

    def proof(_child, _job, inside):
        inside._obj.value = (
            escape == "denied" or not flags[-1] & subprocess.CREATE_BREAKAWAY_FROM_JOB
        )
        return escape != "unproven"

    monkeypatch.setattr(subprocess, "Popen", spawn)
    if os.name == "nt":

        def limits(_job, _kind, output, _size, _length):
            output._obj.BasicLimitInformation.LimitFlags = 0x2000
            return escape != "unproven"

        kernel = SimpleNamespace(IsProcessInJob=proof, QueryInformationJobObject=limits)
        monkeypatch.setattr(ctypes, "WinDLL", lambda *_a, **_k: kernel)
        monkeypatch.setattr(ctypes, "get_last_error", lambda: 87 if escape == "unproven" else 0)
    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / "stray.txt").write_text("not a workspace", encoding="utf-8")
    beside = taken / "workspaces" / "my-research"
    beside.mkdir(parents=True)
    (beside / "research-workspace.json").write_text("{}", encoding="utf-8")
    launch = ["--lang", lang, "--workspace", str(taken), "serve"]
    launch.extend(["--port", "0", "--no-browser", "--detach"])
    # An unreadable preflight refuses before launch and names how to recover.
    with monkeypatch.context() as unreadable:
        unreadable.setattr(
            web_launcher,
            "holds_foreign_files",
            lambda _: (_ for _ in ()).throw(PermissionError("denied")),
        )
        assert main(launch, serve=web_launcher.main) == 2
        refused = json.loads(capsys.readouterr().out)
        assert refused["failure_code"] == "local_web.service_not_started"
        assert refused["next_action"] == "RESTORE_WORKSPACE_ACCESS"
        assert str(taken) in refused["detail"] and not flags
    # A folder can become unreadable after the foreign-file check, before discovery.
    with monkeypatch.context() as unreadable:
        unreadable.setattr(web_launcher, "holds_foreign_files", lambda _: True)
        unreadable.setattr(
            Path, "resolve", lambda *_a, **_k: (_ for _ in ()).throw(PermissionError("denied"))
        )
        assert web_launcher._workspace_refusal(taken) == 2
        refused = json.loads(capsys.readouterr().out)
        assert refused["failure_code"] == "local_web.service_not_started"
        assert refused["next_action"] == "RESTORE_WORKSPACE_ACCESS" and not flags
    # A folder with a Host's connection record is the launch's to read: that Host may serve it.
    record = local_client.LocalResearchConnection.path(taken)
    record.parent.mkdir()
    record.write_text("{}", encoding="utf-8")
    assert web_launcher._workspace_refusal(taken) is None
    shutil.rmtree(record.parent)
    with monkeypatch.context() as escaping:
        escaping.setattr(
            local_client.LocalResearchConnection,
            "path",
            lambda _: (_ for _ in ()).throw(local_client.LocalResearchClientError("escapes")),
        )
        assert web_launcher._workspace_refusal(taken) is None
    # A folder no Host can serve is named before any launch, with no command that could not work.
    assert main(launch, serve=web_launcher.main) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["failure_code"] == "research_workspace.nonempty_manifest_absent"
    assert refused["workspaces"] == [str(beside.resolve())]
    assert "terminal_command" not in refused and not flags
    (taken / "research-workspace.json").write_text("not a manifest", encoding="utf-8")
    assert main(launch, serve=web_launcher.main) == 2
    exited = json.loads(capsys.readouterr().out)
    if escape == "escaped" or os.name != "nt":
        assert exited["failure_code"] == "local_web.detached_host_exited"
        assert "research_workspace." in Path(exited["detail"].rsplit(": ", 1)[1]).read_text()
        if os.name == "nt":
            probe = "import ctypes; print(bool(ctypes.windll.kernel32.GetConsoleWindow()))"
            host = f"import subprocess, sys; subprocess.run([sys.executable, '-c', {probe!r}])"
            log = tmp_path / "probe.log"
            with log.open("wb") as written:
                child = web_launcher._spawn_detached([_PYTHON, "-c", host], written, "k")
            assert child.wait(timeout=60) == 0
            assert log.read_text(encoding="utf-8").split() == ["False"]
    else:
        assert exited["failure_code"] == "local_web.detached_breakaway_denied"
        assert exited["launch_context"] == {
            "win32_error": 87 if escape == "unproven" else 0,
            "job_limit_flags": None if escape == "unproven" else 0x2000,
        }
        context_schema = runtime_json("config/registries/refusals.json")["entries"][
            exited["failure_code"]
        ]["launch_context_schema"]
        assert sorted(exited["launch_context"]) == sorted(context_schema["required"])
        panel = panel_host == "claude-code"
        assert exited["next_action"] == (
            "SERVE_FROM_THE_HOST_TERMINAL_PANEL" if panel else "SERVE_FROM_A_PERSON_OWNED_TERMINAL"
        )
        message_id = "host.start.breakaway_denied" + (".panel" if panel else "")
        assert exited["message_id"] == message_id
        words = runtime_json("config/registries/messages.json")["entries"][message_id]["words"]
        assert exited["detail"] == words["zh-CN" if lang == "zh" else "en"]
        routes = exited.get("terminal_routes", [])
        assert [r["who_decides"] for r in routes] == ["AGENT", "PERSON"]
        assert routes[0]["next_action"] == "SERVE_FROM_THE_HOST_TERMINAL_PANEL"
        assert routes[1]["next_action"] == "SERVE_FROM_A_PERSON_OWNED_TERMINAL"
        assert all(r["command_field"] == "terminal_command" for r in routes)
        command = exited["terminal_command"]
        assert command[1:] == launch
        assert Path(command[0]).stem.casefold() == "alphalattice"
        resolved = shutil.which(command[0])
        assert resolved and Path(resolved).is_file()
        if escape == "denied" and lang == "en" and panel_host == "codex":
            from alphalattice.interface.local_application import native_setup

            marker = native_setup.ROOT / ".codex" / native_setup.PROJECT_DECLARATION_NAME

            def deny_declaration(*_args):
                raise PermissionError(13, "sandbox fixture", str(marker))

            monkeypatch.setattr(native_setup, "declare_project", deny_declaration)
            configured = native_setup._configure("codex", taken)
            assert configured["next_commands"]["serve"] == exited["terminal_command"]
        assert not list(tmp_path.glob("alphalattice-host-*.log"))
    assert host_fault_handlers and all(value == "1" for value in host_fault_handlers)
    assert spawned and all(child.poll() is not None for child in spawned)
    assert web_launcher.main(["--workspace", str(tmp_path), "--detach", "--stop-on-stdin"]) == 2
    tied = json.loads(capsys.readouterr().out)
    assert tied["failure_code"] == "local_web.detach_with_stop_on_stdin"
    if os.name == "nt":

        def denied():
            raise OSError("private operating-system detail")

        monkeypatch.setattr(web_launcher, "ChildJob", denied, raising=False)
        token = web_launcher.ANSWER_LANGUAGE.set(lang)
        command = [_PYTHON, "-c", "pass"]
        try:
            assert web_launcher.attached_child(command, env=dict(os.environ)) == 2
        finally:
            web_launcher.ANSWER_LANGUAGE.reset(token)
        answer = json.loads(capsys.readouterr().out)
        assert answer["failure_code"] == "local_web.child_ownership_unavailable"
        assert answer["terminal_command"] == command
        assert answer["message_id"] == "host.child_ownership.unavailable"
        words = runtime_json("config/registries/messages.json")["entries"][answer["message_id"]]
        assert answer["detail"] == words["words"]["zh-CN" if lang == "zh" else "en"]
        assert "private" not in json.dumps(answer)


def test_the_launchers_module_loads_no_numerical_library() -> None:
    """A Host run as `-m web_launcher` has each spawned worker re-import it before the worker's
    one-thread setting: imported, it loads no numerical library, so BLAS starts on one thread."""
    root = resolve_playpen_root(Path(__file__))
    probe = (
        "import sys; import alphalattice.control.product_host.composition.web_launcher; "
        "print(sorted({'numpy', 'scipy', 'lightgbm', 'sklearn'} & set(sys.modules)))"
    )
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True
    )
    assert loaded.stdout.strip() == "[]"


def _visible_windows(pids: set[int]) -> int:
    """How many visible top-level windows these processes own (Windows; elsewhere none)."""
    if sys.platform != "win32":
        return 0
    user32, owner, found = ctypes.windll.user32, ctypes.c_ulong(), []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def each(window, _extra):
        user32.GetWindowThreadProcessId(ctypes.c_void_p(window), ctypes.byref(owner))
        if owner.value in pids and user32.IsWindowVisible(ctypes.c_void_p(window)):
            found.append(window)
        return True

    user32.EnumWindows(each, None)
    return len(found)


def test_a_token_count_needs_no_network_and_writes_no_temporary_file(tmp_path: Path) -> None:
    """The proxy token count reads the encoding the package ships: with the network unreachable
    and an empty temporary directory it counts, and writes nothing there."""
    root = resolve_playpen_root(Path(__file__))
    temp = tmp_path / "temp"
    temp.mkdir()
    dead = "http://127.0.0.1:9"  # no proxy answers there: any download fails at once
    env = {
        **{
            k: v
            for k, v in os.environ.items()
            if k not in {"TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR"}
        },
        **{name: str(temp) for name in ("TEMP", "TMP", "TMPDIR")},
        **{name: dead for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")},
        "PYTHONPATH": str(root / "src"),
    }
    probe = (
        "from alphalattice.protocols.actor_execution.projections import _cl100k; "
        "print(_cl100k().encode('hello world'))"
    )
    counted = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, timeout=120
    )
    assert counted.returncode == 0, counted.stderr[-600:]
    # cl100k's own ids for the phrase: the shipped ranks are tiktoken's, bit for bit
    assert counted.stdout.strip() == "[15339, 1917]" and not list(temp.iterdir())


def _access(path: Path) -> list[str]:
    """A file's access entries as Windows lists them, or its POSIX mode."""
    if sys.platform != "win32":
        return [oct(path.stat().st_mode & 0o777)]
    listed = subprocess.run(["icacls", str(path)], capture_output=True, text=True, check=True)
    lines = listed.stdout.splitlines()
    entries = [lines[0][len(str(path)) :].strip()]
    entries += [line.strip() for line in lines[1:] if line.startswith(" ") and line.strip()]
    return entries


def test_a_staged_publication_keeps_the_workspace_access(tmp_path: Path) -> None:
    """What the product stages and renames into a workspace has the access of a file written
    there directly: never a folder private to the writing account (another account's Host)."""
    from alphalattice.control.workspace_runtime.content_store import staging_directory

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if sys.platform == "win32":  # the workspace grants another principal, as the person's own
        grant = ["icacls", str(workspace), "/grant", "*S-1-5-32-545:(OI)(CI)(RX)"]
        subprocess.run(grant, capture_output=True, check=True)
    (workspace / "direct.json").write_text("{}", encoding="utf-8")
    with staging_directory(workspace, ".binding-") as staged:
        (staged / "manifest.json").write_text("{}", encoding="utf-8")
        (staged / "manifest.json").rename(workspace / "manifest.json")
    assert sorted(path.name for path in workspace.iterdir()) == ["direct.json", "manifest.json"]
    assert _access(workspace / "manifest.json") == _access(workspace / "direct.json")


def test_cli_missing_dependency_names_locked_setup_but_does_not_hide_missing_product(
    monkeypatch, capsys
):

    from scripts import run_alphalattice

    original_import = builtins.__import__
    missing = "pydantic"

    def unavailable(name, *args, **kwargs):
        if name == "alphalattice.interface.local_application.cli":
            raise ModuleNotFoundError(name=missing)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    assert run_alphalattice.main(["--help"]) == 2
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["missing_module"] == "pydantic"
    assert refusal["setup_command"][-2:] == ["--locked", "--all-extras"]
    assert refusal["claim_limit"] == "NO_INSTALLATION_OR_WORKSPACE_WORK_PERFORMED"
    missing = "alphalattice.missing_implementation"
    with pytest.raises(ModuleNotFoundError):
        run_alphalattice.main(["--help"])
