"""The installed entry composes product owners without checkout script imports."""

from __future__ import annotations

import builtins
import contextlib
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphalattice.control.product_host.composition import saved_object_readback
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
    publish_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.web_launcher import OPEN_THE_WORKBENCH
from alphalattice.kernel.shared_kernel.project_layout import runtime_json

WINDOWS = pytest.mark.skipif(sys.platform != "win32", reason="Windows process control")
_FLAGS = 0x08004000 if sys.platform == "win32" else 0  # no window, below normal
_PYTHON = str(Path(sys.base_prefix) / "python.exe") if sys.platform == "win32" else sys.executable


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


@pytest.mark.parametrize("lines", ["stop\n", "ignored\nstop\n", ""])
def test_attached_launcher_stops_on_explicit_stdin_or_eof(tmp_path, monkeypatch, capsys, lines):

    from scripts import run_local_portfolio_web

    from alphalattice.interface.local_application.cli import main

    calls = []

    class Session:
        launch_url = "http://127.0.0.1:12345/launch?key=k"  # printed and opened (HB)
        operations = None

        def start(self):
            root = Path(__file__).resolve().parents[2]
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
    monkeypatch.setattr(sys, "stdin", StringIO(lines))
    assert (
        main(
            ["--workspace", str(tmp_path), "serve", "--no-browser", "--stop-on-stdin"],
            serve=run_local_portfolio_web.main,
        )
        == 0
    )
    assert calls == ["start", "stop"]
    shown = capsys.readouterr().out.splitlines()
    assert shown[0].endswith(Session.launch_url) and shown[2] == OPEN_THE_WORKBENCH


_TURN = """
import ctypes, os, sys
if os.name == "nt":
    from tests.portfolio_strategy_lab.test_runtime_launchers import _job
    k, job = _job(0x2800 if os.environ["ALPHALATTICE_TEST_JOB_MODE"] == "escaped" else 0x2000)
from alphalattice.control.product_host.composition.entry import main
result = main(sys.argv[1:])  # this CLI is inside the verified finite Job; only its Host escapes
if os.name == "nt" and result == 2:
    accounting = (ctypes.c_byte * 48)()
    assert k.QueryInformationJobObject(job, 1, accounting, 48, None)
    assert ctypes.c_uint32.from_buffer(accounting, 40).value == 1  # no held child before Job close
sys.exit(result)
"""


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
        env=env,
        start_new_session=sys.platform != "win32",
        creationflags=_FLAGS
        | (subprocess.CREATE_BREAKAWAY_FROM_JOB if escaped and os.name == "nt" else 0),
    )
    try:
        output, error = turn.communicate(timeout=180)
        assert turn.returncode == (0 if escaped else 2), error
        assert output, error
        return output
    finally:
        if sys.platform != "win32":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(turn.pid, signal.SIGKILL)


@pytest.mark.parametrize("workspace_state", ["fresh", "initialized"])
def test_a_detached_host_outlives_the_turn_that_started_it(
    tmp_path: Path, workspace_state: str, request
) -> None:
    """A turn's end kills everything its launch started; the Host `serve --detach` started, on
    a new workspace or an initialized one, serves a later client until `host stop`, with its
    key in no log; a second launch is refused by name and leaves no log."""
    import fnmatch

    import psutil

    from alphalattice.control.product_host.composition.entry import main
    from alphalattice.interface.local_application import client

    root = Path(__file__).resolve().parents[2]
    budgets = runtime_json("config/registries/process-budgets.json")
    host_budget = budgets["kinds"]["host"]
    workspace = tmp_path / "workspace"
    if workspace_state == "initialized":
        manifest = ResearchWorkspaceManifest.research_only("turn")
        publish_research_workspace_manifest(workspace, manifest)

    connection = client.LocalResearchConnection.path(workspace)

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
        assert not connection.exists() and not list(temp.iterdir())
        return
    assert "Portfolio Research workspace:" in out and "outlives it" in out
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
    commit = sum(process.memory_info().private for process in tree) / 2**30
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
def test_a_detached_launch_refuses_what_it_cannot_serve(
    tmp_path, monkeypatch, capsys, escape, lang
):
    """A detached Host that exits before serving answers its code and log, and a detached launch
    tied to standard input is refused by name, before anything starts."""
    from alphalattice.control.product_host.composition import web_launcher
    from alphalattice.interface.local_application.cli import main

    for name in ("TEMP", "TMP", "TMPDIR"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.setattr(web_launcher.tempfile, "tempdir", None)
    native, spawned, flags = subprocess.Popen, [], []

    def spawn(*args, **kwargs):
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
        kernel = SimpleNamespace(IsProcessInJob=proof)
        monkeypatch.setattr(ctypes, "WinDLL", lambda *_a, **_k: kernel)
    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / "stray.txt").write_text("not a workspace", encoding="utf-8")
    launch = ["--lang", lang, "--workspace", str(taken), "serve"]
    launch.extend(["--port", "0", "--no-browser", "--detach"])
    assert main(launch, serve=web_launcher.main) == 2
    exited = json.loads(capsys.readouterr().out)
    if escape == "escaped" or os.name != "nt":
        assert exited["failure_code"] == "local_web.detached_host_exited"
        assert "nonempty_manifest_absent" in Path(exited["detail"].rsplit(": ", 1)[1]).read_text()
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
        assert exited["next_action"] == "SERVE_FROM_A_PERSON_OWNED_TERMINAL"
        message_id = "host.start.breakaway_denied"
        assert exited["message_id"] == message_id
        words = runtime_json("config/registries/messages.json")["entries"][message_id]["words"]
        assert exited["detail"] == words["zh-CN" if lang == "zh" else "en"]
        assert exited["terminal_command"] == ["alphalattice", *launch]
        assert not list(tmp_path.glob("alphalattice-host-*.log"))
    assert spawned and all(child.poll() is not None for child in spawned)
    assert web_launcher.main(["--workspace", str(tmp_path), "--detach", "--stop-on-stdin"]) == 2
    tied = json.loads(capsys.readouterr().out)
    assert tied["failure_code"] == "local_web.detach_with_stop_on_stdin"


def test_the_launchers_module_loads_no_numerical_library() -> None:
    """A Host run as `-m web_launcher` has each spawned worker re-import it before the worker's
    one-thread setting: imported, it loads no numerical library, so BLAS starts on one thread."""
    root = Path(__file__).resolve().parents[2]
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
    root = Path(__file__).resolve().parents[2]
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
