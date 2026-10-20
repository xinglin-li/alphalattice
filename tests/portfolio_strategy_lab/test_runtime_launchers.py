"""The installed entry composes product owners without checkout script imports."""

from __future__ import annotations

import builtins
import contextlib
import ctypes
import json
import os
import signal
import struct
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path

import pytest

from alphalattice.control.product_host.composition import saved_object_readback
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
    publish_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.web_launcher import OPEN_THE_WORKBENCH


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
            calls.append("start")
            return "http://127.0.0.1:12345/"

        def stop(self):
            calls.append("stop")

    monkeypatch.setattr(
        run_local_portfolio_web.LocalPortfolioWebSession,
        "from_workspace",
        lambda *a, **k: Session(),
    )
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


@pytest.mark.parametrize("ending", ["stop\n", ""])
def test_a_host_waiting_for_its_stop_lets_the_processes_it_starts_begin(ending: str) -> None:
    "a synchronous read pending on the stdin pipe held every process the"

    root = Path(__file__).resolve().parents[2]
    host = subprocess.Popen(
        [sys.executable, "-c", _WATCHED_STDIN, str(root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert host.stdout is not None and host.stdin is not None
    assert host.stdout.readline().strip() == "started"
    host.stdin.write("noise\n" + ending)
    host.stdin.close()
    assert host.wait(timeout=30) == 0


_TURN = """
import subprocess, sys
sys.stdin.readline()  # released once the turn holds this process: its Job, or its group
from alphalattice.control.product_host.composition.entry import main
sys.exit(main(sys.argv[1:]))
"""


def _end_turn_after(command: list[str], root: Path, temp: Path) -> str:
    """Run a launch as a Codex turn does and end the turn: its whole Job (Windows) or process
    group is killed once the launch returns, children and all. Its logs go to ``temp``."""
    folders = {name: str(temp) for name in ("TEMP", "TMP", "TMPDIR")}
    env = {**os.environ, **folders, "PYTHONPATH": str(root / "src")}
    if sys.platform != "win32":
        turn = subprocess.Popen(
            [sys.executable, "-c", _TURN, *command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        out = turn.communicate("go\n", timeout=180)[0]
        with contextlib.suppress(ProcessLookupError):  # the group may be empty already
            os.killpg(turn.pid, signal.SIGKILL)
        return out
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    job = ctypes.c_void_p(kernel32.CreateJobObjectW(None, None))
    # JOBOBJECT_EXTENDED_LIMIT_INFORMATION (144 bytes on x64), the only form that takes these
    # flags: LimitFlags at offset 16 kills on close and allows breakaway, as a turn's Job may.
    limits = (ctypes.c_ubyte * 144)()
    struct.pack_into("<I", limits, 16, 0x2000 | 0x800)
    assert kernel32.SetInformationJobObject(job, 9, limits, ctypes.sizeof(limits))
    turn = subprocess.Popen(
        [sys.executable, "-c", _TURN, *command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert kernel32.AssignProcessToJobObject(job, ctypes.c_void_p(int(turn._handle)))
        return turn.communicate("go\n", timeout=180)[0]
    finally:
        kernel32.TerminateJobObject(job, 1)
        kernel32.CloseHandle(job)


@pytest.mark.parametrize("workspace_state", ["fresh", "initialized"])
def test_a_detached_host_outlives_the_turn_that_started_it(
    tmp_path: Path, workspace_state: str
) -> None:
    """A turn's end kills everything its launch started; the Host `serve --detach` started, on
    a new workspace or an initialized one, serves a later client until `host stop`, with its
    key in no log; a second launch is refused by name and leaves no log."""
    from alphalattice.control.product_host.composition.entry import main
    from alphalattice.interface.local_application.client import LocalResearchConnection

    root = Path(__file__).resolve().parents[2]
    workspace = tmp_path / "workspace"
    if workspace_state == "initialized":
        manifest = ResearchWorkspaceManifest.research_only("turn")
        publish_research_workspace_manifest(workspace, manifest)
    temp = tmp_path / "temp"
    temp.mkdir()
    launch = ["--workspace", str(workspace), "serve", "--no-browser", "--detach"]
    out = _end_turn_after(launch, root, temp)
    assert "Portfolio Research workspace:" in out and "outlives it" in out
    started = out.splitlines()[0]  # plainly STARTED, with the workspace and the address
    assert started.startswith("STARTED") and str(workspace.resolve()) in started
    assert out.rstrip().endswith(OPEN_THE_WORKBENCH)  # the step right after the launch
    key = out.split("key=", 1)[1].split("#", 1)[0]
    logs = list(temp.glob("alphalattice-host-*.log"))
    assert len(logs) == 1 and key not in logs[0].read_text(encoding="utf-8")
    address = LocalResearchConnection.read(workspace).url  # a later client finds the Host
    assert address in started
    second = json.loads(_end_turn_after(launch, root, temp))  # plainly ALREADY_SERVING
    assert second["status"] == "ALREADY_SERVING" and second["workspace"] in started
    assert (second["address"], second["next_action"]) == (address, "USE_THE_SERVING_HOST")
    assert list(temp.glob("alphalattice-host-*.log")) == logs
    assert main(["--workspace", str(workspace), "host", "stop"]) == 0
    connection = LocalResearchConnection.path(workspace)
    deadline = time.monotonic() + 60
    while connection.exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not connection.exists()


def test_a_detached_launch_refuses_what_it_cannot_serve(tmp_path, monkeypatch, capsys) -> None:
    """A detached Host that exits before serving answers its code and log, and a detached launch
    tied to standard input is refused by name, before anything starts."""
    from alphalattice.control.product_host.composition import web_launcher

    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setenv("TMP", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(web_launcher.tempfile, "tempdir", None)
    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / "stray.txt").write_text("not a workspace", encoding="utf-8")
    assert web_launcher.main(["--workspace", str(taken), "--no-browser", "--detach"]) == 2
    exited = json.loads(capsys.readouterr().out)
    assert exited["failure_code"] == "local_web.detached_host_exited"
    assert "nonempty_manifest_absent" in Path(exited["detail"].rsplit(": ", 1)[1]).read_text()
    assert web_launcher.main(["--workspace", str(tmp_path), "--detach", "--stop-on-stdin"]) == 2
    tied = json.loads(capsys.readouterr().out)
    assert tied["failure_code"] == "local_web.detach_with_stop_on_stdin"


@pytest.mark.skipif(sys.platform != "win32", reason="a console window is Windows'")
def test_a_detached_hosts_workers_open_no_console_window(tmp_path) -> None:
    """A process the detached Host starts, as its Task workers are, opens no console window on
    the person's desktop (a window there, closed, would kill the worker)."""
    from alphalattice.control.product_host.composition import web_launcher

    probe = "import ctypes; print(bool(ctypes.windll.kernel32.GetConsoleWindow()))"
    host = f"import subprocess, sys; subprocess.run([sys.executable, '-c', {probe!r}])"
    log = tmp_path / "probe.log"
    with log.open("wb") as written:
        child, _held = web_launcher._spawn_detached([sys.executable, "-c", host], written, "k")
    assert child.wait(timeout=60) == 0
    assert log.read_text(encoding="utf-8").split() == ["False"]


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


def test_a_served_host_keeps_the_persons_machine_clean(tmp_path: Path) -> None:
    """A served Host holds its declared budget of processes, commit and CPU while idle, opens
    no window, writes nothing outside the workspace but declared files, and leaves nothing."""
    import fnmatch

    import psutil

    from alphalattice.control.product_host.composition.entry import main

    root = Path(__file__).resolve().parents[2]
    budgets = json.loads((root / "config/registries/process-budgets.json").read_text("utf-8"))
    host_budget = budgets["kinds"]["host"]
    workspace, temp = tmp_path / "workspace", tmp_path / "temp"
    temp.mkdir()
    out = _end_turn_after(
        ["--workspace", str(workspace), "serve", "--no-browser", "--detach"], root, temp
    )
    tree = psutil.Process(int(out.split("(process ", 1)[1].split(")", 1)[0]))
    tree = [tree, *tree.children(recursive=True)]
    try:
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
    finally:
        assert main(["--workspace", str(workspace), "host", "stop"]) == 0
    _gone, alive = psutil.wait_procs(tree, timeout=host_budget["stopped_within_seconds"])
    assert not alive, [p.name() for p in alive]


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
