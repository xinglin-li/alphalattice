"""The process owner gives real children bounded, isolated streams and lifetimes."""

import importlib.util
import logging
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import suppress
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import psutil
import pytest

from alphalattice.kernel.execution import (
    CANCELLED,
    ExecutionStopped,
    cancel_execution,
    execution_cancellation,
    execution_port,
    linux_scope,
    process_execution,
)
from alphalattice.kernel.execution import spawn as execution_spawn

TEST_SOURCE = "src/alphalattice/capabilities/domain_contracts/spawn.py"


@pytest.mark.proves("execution.registered_pools_keep_their_workers")
def test_registered_pools_keep_their_workers() -> None:
    """Register both pools and close the first without stopping the second's worker."""
    owner = process_execution()
    with (
        owner.register_pool(ThreadPoolExecutor(1), name="first-test-pool") as first,
        owner.register_pool(ThreadPoolExecutor(1), name="second-test-pool") as second,
    ):
        assert first is not second
        assert {"first-test-pool", "second-test-pool"} <= set(owner.registered_pool_names())
        first.shutdown()
        assert second.submit(lambda: 1).result(timeout=1) == 1


@pytest.mark.proves("domain_contracts.child_processes_are_bounded_and_isolated")
def test_ambient_job_probe_adds_no_job(monkeypatch, tmp_path):
    """The ambient probe measures the caller's Job without adding its own."""
    from alphalattice.capabilities.domain_contracts import spawn

    calls = []
    monkeypatch.setenv("ALPHALATTICE_DATA_ROOT", str(tmp_path / "person-root"))
    monkeypatch.setattr(execution_spawn.subprocess, "CREATE_NO_WINDOW", 8, raising=False)
    monkeypatch.setattr(
        execution_spawn.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr(execution_spawn, "ChildJob", lambda: pytest.fail("probe added a Job"))
    spawn.ambient_job_probe([sys.executable, "probe.py", "--child"], creationflags=16)
    args, options = calls[0]
    assert args[1:] == ["probe.py", "--child"]
    assert options["creationflags"] == 24
    assert options["timeout"] == 60
    assert options["stdin"] == subprocess.DEVNULL
    assert options["capture_output"] is True
    assert options["text"] is True and options["encoding"] == "utf-8"
    assert options["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert "ALPHALATTICE_DATA_ROOT" not in options["env"]


@pytest.mark.proves("domain_contracts.child_processes_are_bounded_and_isolated")
def test_a_posix_group_is_killed_before_waiting_for_its_members(monkeypatch):
    """The group owner waits for its remaining member after sending the stop signal."""
    from alphalattice.capabilities.domain_contracts import spawn

    events = []

    class Member:
        pid = 456

        def status(self):
            return "running"

    member = Member()
    monkeypatch.setattr(
        execution_spawn.os, "killpg", lambda pid, sig: events.append(("kill", pid)), raising=False
    )
    monkeypatch.setattr(execution_spawn.os, "getpgid", lambda pid: 123, raising=False)
    monkeypatch.setattr(execution_spawn.signal, "SIGKILL", 9, raising=False)

    def members():
        events.append(("scan",))
        return [] if ("kill", 123) in events else [member]

    monkeypatch.setattr(psutil, "process_iter", members)

    def wait(members, *, timeout):
        events.append(("wait", members[0].pid))
        return ([], members) if events.count(("wait", 456)) == 1 else (members, [])

    monkeypatch.setattr(psutil, "wait_procs", wait)
    execution_spawn._stop_group(123, processes=psutil)
    assert events == [("scan",), ("kill", 123), ("scan",), ("wait", 456), ("wait", 456)]

    # A timeout stops the owned group once, even when final cleanup follows.
    stops = []

    class Child:
        pid = 123
        stdin = stdout = stderr = None
        returncode = -9

        def communicate(self, input=None, timeout=None):
            if not stops:
                raise subprocess.TimeoutExpired("child", timeout)
            return b"", b""

        def poll(self):
            return self.returncode

    def stop(pid, sig):
        stops.append(pid)
        if len(stops) > 1:
            raise PermissionError("group no longer belongs to this child")

    monkeypatch.setattr(
        execution_spawn, "sys", SimpleNamespace(platform="linux", executable=sys.executable)
    )
    monkeypatch.setattr(execution_spawn.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(execution_spawn.os, "killpg", stop)
    monkeypatch.setattr(psutil, "process_iter", lambda: [])
    with pytest.raises(subprocess.TimeoutExpired):
        spawn.run([sys.executable, "-c", "pass"], timeout=1)
    assert stops == [123]


def _assert_cancelled_tree_exits(*, parent_returns=False):
    from alphalattice.capabilities.domain_contracts import spawn

    parent = """
import subprocess
import os
import sys
import threading

child = subprocess.Popen(
    [sys.executable, "-c", "import threading; threading.Event().wait(20)"],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=os.name == "posix",
    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
)
print(child.pid, flush=True)
if sys.argv[1] == "live":
    threading.Event().wait(20)
"""
    execution_id = uuid4()
    with (
        execution_cancellation(execution_id),
        spawn.InteractiveSession(timeout=10, stop_scope="tree") as session,
    ):
        child = session.start(
            [sys.executable, "-c", parent, "return" if parent_returns else "live"],
            stdout=subprocess.PIPE,
        )
        assert child.stdout is not None
        descendant = psutil.Process(int(child.stdout.readline()))
        try:
            if os.name == "posix":
                assert os.getsid(descendant.pid) == descendant.pid
            if parent_returns:
                assert child.wait(timeout=5) == 0
            cancel_execution(execution_id)
            assert session._finished.wait(timeout=5)  # The cancellation, not the test, stops it.
            session.stop()
            with pytest.raises(ExecutionStopped, match=CANCELLED.replace(".", r"\.")):
                session.check()
            with suppress(psutil.NoSuchProcess):
                assert descendant.status() == psutil.STATUS_ZOMBIE
        finally:
            with suppress(psutil.NoSuchProcess):
                descendant.kill()
                descendant.wait(timeout=5)


def _leave_pool_descendant(ready):
    context = multiprocessing.get_context("spawn")
    received, published = context.Pipe(duplex=False)
    with received, published:
        child = context.Process(target=_escaped_session, args=(published,))
        child.start()
        published.close()
        if not received.poll(5):
            child.kill()
            child.join(timeout=5)
            raise TimeoutError("fixture descendant did not enter its new session")
        ready.write_text(str(received.recv()), encoding="ascii")
    os._exit(91)


def _escaped_session(published):
    vars(os)["setsid"]()
    with published:
        published.send(os.getpid())
    threading.Event().wait(25)


def _assert_pool_retains_escaped_child(ready):
    from alphalattice.capabilities.domain_contracts import spawn

    execution_id = uuid4()
    with execution_cancellation(execution_id):
        pool = execution_port().register_pool(
            ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")),
            name="causal-derivation",
        )
        descendant = None
        try:
            future = pool.submit(_leave_pool_descendant, ready)
            with pytest.raises(BrokenProcessPool):
                future.result(timeout=10)
            pid = int(ready.read_text(encoding="ascii"))
            with suppress(psutil.NoSuchProcess):
                descendant = psutil.Process(pid)
            cancel_execution(execution_id)
            spawn.stop_pool(pool)
            if descendant is not None:
                with suppress(psutil.NoSuchProcess):
                    assert descendant.status() == psutil.STATUS_ZOMBIE
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            if descendant is not None:
                with suppress(psutil.NoSuchProcess):
                    descendant.kill()
                    descendant.wait(timeout=5)


@pytest.mark.proves("domain_contracts.child_processes_are_bounded_and_isolated")
@pytest.mark.parametrize(
    ("budget", "mode", "members", "limit_event", "stops", "report"),
    [
        (100, "observe", (120,), False, False, "spawn.tree_memory_limit"),
        (100, "enforce", (120,), False, True, "spawn.tree_memory_limit"),
        (100, "enforce", (70, 60), False, True, "spawn.tree_memory_limit"),
        (100, "enforce", (70,), False, False, ""),
        (100, "enforce", (97,), True, True, "spawn.tree_memory_limit"),
        (None, "enforce", (120,), False, False, "budget_bytes=undeclared"),
        (None, "cancel", (), False, True, ""),
        (None, "cancel-pool-admission", (), False, True, ""),
        (100, "darwin-aggregate", (70, 60), False, True, ""),
        pytest.param(
            100,
            "host-observer-observe",
            (),
            False,
            False,
            "spawn.host_observer_unavailable",
            marks=pytest.mark.skipif(sys.platform != "linux", reason="Linux Host cgroup observer"),
        ),
        pytest.param(
            None,
            "cancel-retired-pool",
            (),
            False,
            True,
            "",
            marks=pytest.mark.skipif(
                sys.platform not in {"darwin", "linux"}, reason="POSIX retired pool ownership"
            ),
        ),
    ],
    ids=(
        "observe",
        "enforce",
        "grandchild",
        "under",
        "native-limit",
        "undeclared",
        "cancel",
        "cancel-pool-admission",
        "darwin-aggregate",
        "host-observer-observe",
        "cancel-retired-pool",
    ),
)
def test_spawn_judges_one_budget_for_the_whole_tree(
    budget, mode, members, limit_event, stops, report, caplog, tmp_path, monkeypatch
):
    """A tree charge includes all members, and an OS limit event can deny an allocation."""
    if mode == "cancel":
        _assert_cancelled_tree_exits()
        if sys.platform in {"win32", "darwin"}:
            _assert_cancelled_tree_exits(parent_returns=True)
        return
    if mode == "cancel-pool-admission":
        from alphalattice.capabilities.domain_contracts import spawn

        execution_id = uuid4()
        with execution_cancellation(execution_id):
            pool = execution_port().register_pool(
                ProcessPoolExecutor(max_workers=1), name="causal-derivation"
            )
            try:
                cancel_execution(execution_id)
                with pytest.raises(ExecutionStopped, match=r"execution\.cancelled"):
                    pool.submit(os.getpid)
                assert not pool._processes
            finally:
                spawn.stop_pool(pool)
                pool.shutdown(wait=True, cancel_futures=True)
        return
    if mode == "darwin-aggregate":
        ancestry = SimpleNamespace(
            remember=lambda _members: None,
            members=lambda _candidates: (
                False,
                [
                    SimpleNamespace(memory_info=lambda amount=amount: SimpleNamespace(rss=amount))
                    for amount in members
                ],
            ),
        )
        watch = object.__new__(execution_spawn._PosixMemoryWatch)
        watch._lock = threading.RLock()
        watch._members = []
        watch.lineage = ancestry
        watch.processes = SimpleNamespace(
            process_iter=lambda: (), NoSuchProcess=psutil.NoSuchProcess
        )
        watch.memory = execution_spawn.TreeMemory("host", budget, "enforce")
        assert watch._sample() == sum(members)
        return
    if mode == "cancel-retired-pool":
        if sys.platform == "linux":

            def unavailable_scope():
                pytest.fail("observe consulted the optional Linux cgroup manager")

            monkeypatch.setattr(execution_spawn, "linux_scope_available", unavailable_scope)
            monkeypatch.setattr(psutil.Process, "children", lambda self, recursive=False: [])
        _assert_pool_retains_escaped_child(tmp_path / "descendant")
        return
    if mode == "host-observer-observe":
        observer = MagicMock()
        context = SimpleNamespace(
            Pipe=lambda: (MagicMock(), MagicMock()), Process=lambda **_: observer
        )
        monkeypatch.setattr(execution_spawn, "linux_scope_available", lambda: True)
        monkeypatch.setattr(execution_spawn.multiprocessing, "get_context", lambda _: context)
        memory = execution_spawn.TreeMemory("host", budget, "observe")

        def unavailable(_thread: threading.Thread) -> None:
            raise RuntimeError("fixture process-tree sampler is unavailable")

        monkeypatch.setattr(threading.Thread, "start", unavailable)
        with execution_spawn.current_tree(psutil, memory):
            observer.start.assert_not_called()
            assert f"{report}: kind=host phase=start" in caplog.text
        return
    with caplog.at_level(logging.INFO):
        memory = execution_spawn.TreeMemory("host", budget, mode)
        assert memory.observe(sum(members), limit_event=limit_event) is stops
        memory.finish()
    if report:
        assert report in caplog.text
        assert "kind=host" in caplog.text
        assert f"peak_bytes={sum(members)}" in caplog.text
    else:
        assert "spawn.tree_memory_limit" not in caplog.text
    if stops:
        with pytest.raises(execution_spawn.TreeMemoryLimit, match=r"spawn\.tree_memory_limit"):
            memory.check()
    else:
        memory.check()


@pytest.mark.proves("domain_contracts.child_processes_are_bounded_and_isolated")
@pytest.mark.parametrize(
    ("form", "mode", "reason"),
    [
        ("proved", "enforce", None),
        ("cgroup unavailable", "observe", "no user cgroup manager"),
        ("observer late", "observe", "Host observer not ready in 11 s"),
        ("observer dead", "observe", "Host observer exited with code 1"),
        ("observer refused", "observe", "CalledProcessError: busctl exited 1"),
    ],
    ids=str,
)
def test_a_linux_host_enforces_only_a_proved_admission(form, mode, reason, monkeypatch, tmp_path):
    """An unproved admission starts the Host observing its budget, never refused, and says why;
    an unready observer's own cap is cleared before the Host observes; an inherited one stays."""
    owner, observer = MagicMock(), MagicMock()
    owner.poll.return_value = form != "observer late"
    owner.recv_bytes.side_effect = {
        "proved": [b"ready"],
        "observer dead": EOFError,
        "observer refused": [b"refused CalledProcessError: busctl exited 1"],
    }.get(form)
    observer.is_alive.side_effect = [True, False] if form == "observer late" else None
    observer.is_alive.return_value = False
    observer.exitcode = 1 if form == "observer dead" else 0
    context = SimpleNamespace(Pipe=lambda: (owner, MagicMock()), Process=lambda **_: observer)
    monkeypatch.setattr(
        execution_spawn, "linux_scope_available", lambda: form != "cgroup unavailable"
    )
    monkeypatch.setattr(execution_spawn.multiprocessing, "get_context", lambda _: context)
    # The Host starts in another live Host's capped serving cgroup it inherited. Its observer's
    # attach moves it into this admission's own, capped, before the late and refused rows fail;
    # the dead observer exits before any move.
    record = tmp_path / "cgroup"
    caps = {}
    for scope in ("inherited", "admitted"):
        caps[scope] = tmp_path / "app.slice" / f"alphalattice-{scope}.scope" / "serving"
        caps[scope].mkdir(parents=True)
        for name, value in (("memory.max", "100"), ("memory.oom.group", "1")):
            (caps[scope] / name).write_text(value)
    record.write_text("0::/app.slice/alphalattice-inherited.scope/serving\n")
    moves = form in {"proved", "observer late", "observer refused"}
    moved = "0::/app.slice/alphalattice-admitted.scope/serving\n"
    observer.start.side_effect = (lambda: record.write_text(moved)) if moves else None
    monkeypatch.setattr(
        execution_spawn, "linux_scope_group", partial(linux_scope.own_group, record)
    )
    release = partial(linux_scope.release_limit, root=tmp_path, record=record)
    monkeypatch.setattr(execution_spawn, "release_linux_limit", release)
    memory = execution_spawn.TreeMemory("host", 100, "enforce")
    with execution_spawn._supervised_linux_host(psutil, memory):
        assert memory.mode == mode
        cap = {
            scope: [(path / name).read_text() for name in ("memory.max", "memory.oom.group")]
            for scope, path in caps.items()
        }
        cleared = moves and form != "proved"
        assert cap == {
            "inherited": ["100", "1"],
            "admitted": ["max", "0"] if cleared else ["100", "1"],
        }
    assert memory.unproved == (reason and f"spawn.cgroup_admission_unproved: {reason}")
    assert observer.kill.called is (form == "observer late")


@pytest.mark.proves("domain_contracts.child_processes_are_bounded_and_isolated")
def test_a_child_gets_closed_input_utf8_and_no_inherited_data_root(monkeypatch, tmp_path):
    """A real interpreter sees only the caller's chosen process environment."""
    name = "alphalattice.capabilities.domain_contracts.spawn"
    assert importlib.util.find_spec(name) is not None
    from alphalattice.capabilities.domain_contracts import spawn

    monkeypatch.setenv("ALPHALATTICE_DATA_ROOT", str(tmp_path / "person-root"))
    monkeypatch.setattr(
        psutil.Process, "threads", lambda _: pytest.fail("child startup listed its threads")
    )
    script = (
        "import os,sys; "
        "sys.stdout.buffer.write((str(len(sys.stdin.read()))+'|'+"
        "os.getenv('GIT_TERMINAL_PROMPT','')+'|'+"
        "str('ALPHALATTICE_DATA_ROOT' in os.environ)+'|café').encode('utf-8'))"
    )
    result = spawn.run([sys.executable, "-c", script], timeout=5, capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout == "0|0|False|café"
    env = {**os.environ, "ALPHALATTICE_DATA_ROOT": str(tmp_path / "test-root")}
    selected = spawn.run(
        [sys.executable, "-c", "import os; print(os.environ['ALPHALATTICE_DATA_ROOT'])"],
        timeout=5,
        capture_output=True,
        text=True,
        env=env,
    )
    assert selected.stdout.strip() == str(tmp_path / "test-root")
    with pytest.raises(ValueError, match=r"spawn\.stdin_unfed"):
        spawn.run([sys.executable, "-c", "pass"], timeout=5, stdin=subprocess.PIPE)
    fed = spawn.run(
        [sys.executable, "-c", "import sys; print(sys.stdin.read())"],
        timeout=5,
        input="fed",
        capture_output=True,
        text=True,
    )
    assert fed.stdout.strip() == "fed"
    interpreter = spawn.run(
        ["python", "-c", "import sys; print(sys.executable)"],
        timeout=5,
        capture_output=True,
        text=True,
    )
    assert interpreter.args[0] == sys.executable
    assert os.path.samefile(interpreter.stdout.strip(), sys.executable)
    with pytest.raises(ValueError, match=r"spawn\.encoding_invalid"):
        spawn.run([sys.executable, "-c", "pass"], timeout=5, encoding="cp1252")


@pytest.mark.proves("domain_contracts.child_processes_are_bounded_and_isolated")
@pytest.mark.parametrize("parent", ["live", "exited", "returned"])
def test_a_timed_out_child_and_its_descendant_are_stopped(parent, tmp_path):
    """A real timed-out process cannot leave its child behind."""
    from alphalattice.capabilities.domain_contracts import spawn

    if parent in {"exited", "returned"}:
        pid_file = tmp_path / "descendant.pid"
        script = (
            "import pathlib,subprocess,sys; "
            "p=subprocess.Popen([sys.executable,'-c','import threading; "
            "threading.Event().wait(20)'],stdin=subprocess.DEVNULL, "
            + (
                "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL, "
                if parent == "returned"
                else ""
            )
            + "creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)); "
            "pathlib.Path(sys.argv[1]).write_text(str(p.pid))"
        )
        try:
            began = time.monotonic()
            if parent == "exited":
                with pytest.raises(subprocess.TimeoutExpired):
                    spawn.run(
                        [sys.executable, "-c", script, str(pid_file)],
                        timeout=2,
                        capture_output=True,
                        text=True,
                    )
            else:
                assert (
                    spawn.run(
                        [sys.executable, "-c", script, str(pid_file)],
                        timeout=2,
                        capture_output=True,
                        text=True,
                    ).returncode
                    == 0
                )
            assert time.monotonic() - began < 3.5
            assert pid_file.exists() and not psutil.pid_exists(int(pid_file.read_text()))
        finally:
            if pid_file.exists() and psutil.pid_exists(pid := int(pid_file.read_text())):
                survivor = psutil.Process(pid)
                survivor.kill()
                survivor.wait(timeout=5)
        return

    script = (
        "import subprocess,sys,threading; "
        "p=subprocess.Popen([sys.executable,'-c','import threading; threading.Event().wait(20)'], "
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL, "
        "creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)); "
        "print(p.pid,flush=True); threading.Event().wait(20)"
    )
    with pytest.raises(subprocess.TimeoutExpired) as error:
        spawn.run([sys.executable, "-c", script], timeout=3, capture_output=True, text=True)
    assert error.value.output
    assert not psutil.pid_exists(int(error.value.output.strip()))


@pytest.mark.proves("domain_contracts.child_processes_are_bounded_and_isolated")
def test_an_interactive_session_feeds_turns_and_captures_streams(monkeypatch):
    """A real child consumes two turns before EOF and the owner closes both pipes."""
    from alphalattice.capabilities.domain_contracts import spawn

    monkeypatch.setattr(
        psutil.Process, "threads", lambda _: pytest.fail("child startup listed its threads")
    )
    code = (
        "import sys; a=sys.stdin.readline(); b=sys.stdin.readline(); "
        "print(a.strip()+'|'+b.strip()); print('error',file=sys.stderr)"
    )
    with spawn.InteractiveSession(timeout=10, stop_scope="tree") as session:
        child = session.start(
            [sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        session.send(b"one\n")
        session.send(b"two\n")
        session.close_input()
        assert child.stdout.read().strip() == b"one|two"
        assert child.stderr.read().strip() == b"error"
        assert child.wait(timeout=5) == 0
        session.check()
    assert child.stdout.closed and child.stderr.closed


@pytest.mark.proves("domain_contracts.child_processes_are_bounded_and_isolated")
def test_an_interactive_session_stops_at_its_deadline():
    """The deadline terminates the real child and the owner reports timeout."""
    from alphalattice.capabilities.domain_contracts import spawn

    with spawn.InteractiveSession(timeout=0.5, stop_scope="tree") as session:
        child = session.start(
            [sys.executable, "-c", "import threading; threading.Event().wait(20)"]
        )
        assert child.wait(timeout=8) != 0
        with pytest.raises(subprocess.TimeoutExpired):
            session.check()


@pytest.mark.proves("domain_contracts.child_processes_are_bounded_and_isolated")
def test_an_interactive_session_names_its_stop_scope(monkeypatch):
    """Root scope stops only the child; tree scope selects the POSIX group barrier."""
    from alphalattice.capabilities.domain_contracts import spawn

    with pytest.raises(ValueError, match="stop_scope"):
        spawn.InteractiveSession(timeout=1, stop_scope="unknown")

    calls = []

    class Child:
        pid = 123
        stdin = stdout = stderr = None
        stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def kill(self):
            calls.append("root")
            self.stopped = True

        def wait(self, timeout):
            calls.append("wait")
            self.stopped = True

    monkeypatch.setattr(
        execution_spawn.os, "killpg", lambda pid, sig: calls.append(("group", pid)), raising=False
    )
    monkeypatch.setattr(execution_spawn.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(psutil, "process_iter", lambda: [])
    root = spawn.InteractiveSession(timeout=1, stop_scope="root")
    root.child = Child()
    root.stop()
    assert calls == ["root"]
    calls.clear()
    tree = spawn.InteractiveSession(timeout=1, stop_scope="tree")
    tree.child = Child()
    tree.stop()
    assert calls == [("group", 123), "wait"]


@pytest.mark.proves("domain_contracts.child_processes_are_bounded_and_isolated")
@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job accounting")
def test_a_job_stop_is_proved_when_a_member_joined_and_exited_after_its_capture():
    """A member that joins after the stop's first capture and exits before the next is counted
    in the Job's TotalProcesses but never captured; the Job's active count proves the stop."""
    job = execution_spawn.ChildJob()
    kernel = job.kernel
    wait = "import threading; threading.Event().wait({})"
    quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    quiet["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    held = subprocess.Popen(
        [sys.executable, "-c", wait.format(60)], stdin=subprocess.DEVNULL, **quiet
    )
    late = []

    class LateMember:  # the real kernel; a member joins between the first capture and the stop
        def __getattr__(self, name):
            return getattr(kernel, name)

        def SetInformationJobObject(self, *arguments):
            # The member waits on its stdin, so it is alive when the checked assignment admits
            # it, and exits only once released.
            member = subprocess.Popen(
                [sys.executable, "-c", "import sys; sys.stdin.read()"],
                stdin=subprocess.PIPE,
                **quiet,
            )
            try:
                job.assign(int(member._handle))
            finally:
                member.communicate(input=b"", timeout=10)
            late.append(member.returncode)
            return kernel.SetInformationJobObject(*arguments)

    try:
        job.assign(int(held._handle))
        job.kernel = LateMember()
        job.terminate()
        assert late == [0] and held.wait(timeout=10) is not None
    finally:
        job.kernel = kernel
        if held.poll() is None:
            held.kill()
        execution_spawn.ChildJob.close(job)


@pytest.mark.proves("domain_contracts.child_processes_are_bounded_and_isolated")
def test_concurrent_stops_wait_for_the_same_termination_barrier():
    """Concurrent callers return only after the owned child has stopped."""
    from alphalattice.capabilities.domain_contracts import spawn

    waiting, release, second_started, second_finished = (threading.Event() for _ in range(4))
    failures = []

    class Child:
        pid = 123
        stdin = stdout = stderr = None

        def poll(self):
            return 0 if release.is_set() else None

        def kill(self):
            pass

        def wait(self, timeout):
            waiting.set()
            if not release.wait(timeout):
                raise TimeoutError("child did not stop")

    owner = spawn.InteractiveSession(timeout=1, stop_scope="root")
    owner.child = Child()

    def stop(second=False):
        if second:
            second_started.set()
        try:
            owner.stop()
        except BaseException as error:
            failures.append(error)
        if second:
            second_finished.set()

    first = threading.Thread(target=stop)
    second = threading.Thread(target=stop, kwargs={"second": True})
    first.start()
    assert waiting.wait(timeout=2)
    second.start()
    assert second_started.wait(timeout=2)
    assert not second_finished.wait(timeout=0.05)
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert second_finished.is_set() and not failures


LAUNCHER = """
import subprocess, sys, psutil
from alphalattice.capabilities.domain_contracts import spawn, waits

with spawn.reap_detached_sessions() as keep:
    own = subprocess.Popen(
        [sys.executable, "-c", "import threading; threading.Event().wait(30)"],
        stdin=subprocess.DEVNULL,
    )
    keep(own.pid)
    parent = subprocess.run(
        [sys.executable, "-c", "import subprocess, sys; "
         "print(subprocess.Popen([sys.executable, '-c', "
         "'import threading; threading.Event().wait(30)'], stdin=subprocess.DEVNULL, "
         "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).pid)"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=True)
    orphan = psutil.Process(int(parent.stdout))
    adopted = orphan.ppid() == psutil.Process().pid
    orphan.kill()
    try:
        waits.wait_until(
            lambda _left: not psutil.pid_exists(orphan.pid), timeout=5, what="the orphan reaped"
        )
        print("adopted" if adopted else "not-adopted", "gone")
    except waits.WaitTimeout:
        state = "zombie" if orphan.status() == psutil.STATUS_ZOMBIE else "alive"
        print("adopted" if adopted else "not-adopted", state)
    own.kill()
    own.wait(timeout=5)
"""


@pytest.mark.proves("domain_contracts.child_processes_are_bounded_and_isolated")
@pytest.mark.skipif(sys.platform != "linux", reason="a child subreaper is Linux's alone")
def test_a_subreaper_launcher_reaps_an_adopted_orphan_as_it_ends():
    """Under a subreaper launcher, an adopted orphan is gone to every reader once it ends."""
    done = subprocess.run(
        [sys.executable, "-c", LAUNCHER],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        timeout=60,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert (done.returncode, done.stdout.strip()) == (0, "adopted gone"), done.stderr
