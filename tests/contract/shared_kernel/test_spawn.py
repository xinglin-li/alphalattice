"""The process owner gives real children bounded, isolated streams and lifetimes."""

import importlib.util
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import psutil
import pytest

TEST_SOURCE = "src/alphalattice/kernel/shared_kernel/spawn.py"


@pytest.mark.proves("shared_kernel.child_processes_are_bounded_and_isolated")
def test_ambient_job_probe_adds_no_job(monkeypatch, tmp_path):
    """The ambient probe measures the caller's Job without adding its own."""
    from alphalattice.kernel.shared_kernel import spawn

    calls = []
    monkeypatch.setenv("ALPHALATTICE_DATA_ROOT", str(tmp_path / "person-root"))
    monkeypatch.setattr(spawn.subprocess, "CREATE_NO_WINDOW", 8, raising=False)
    monkeypatch.setattr(
        spawn.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr(spawn, "ChildJob", lambda: pytest.fail("probe added a Job"))
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


@pytest.mark.proves("shared_kernel.child_processes_are_bounded_and_isolated")
def test_a_posix_group_is_killed_before_waiting_for_its_members(monkeypatch):
    """The group owner waits for its remaining member after sending the stop signal."""
    from alphalattice.kernel.shared_kernel import spawn

    events = []

    class Member:
        pid = 456

        def status(self):
            return "running"

    member = Member()
    monkeypatch.setattr(
        spawn.os, "killpg", lambda pid, sig: events.append(("kill", pid)), raising=False
    )
    monkeypatch.setattr(spawn.os, "getpgid", lambda pid: 123, raising=False)
    monkeypatch.setattr(spawn.signal, "SIGKILL", 9, raising=False)

    def members():
        events.append(("scan",))
        return [] if ("kill", 123) in events else [member]

    monkeypatch.setattr(spawn.psutil, "process_iter", members)

    def wait(members, *, timeout):
        events.append(("wait", members[0].pid))
        return ([], members) if events.count(("wait", 456)) == 1 else (members, [])

    monkeypatch.setattr(spawn.psutil, "wait_procs", wait)
    spawn._stop_group(123)
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

    monkeypatch.setattr(spawn, "sys", SimpleNamespace(platform="linux", executable=sys.executable))
    monkeypatch.setattr(spawn.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(spawn.os, "killpg", stop)
    monkeypatch.setattr(spawn.psutil, "process_iter", lambda: [])
    with pytest.raises(subprocess.TimeoutExpired):
        spawn.run([sys.executable, "-c", "pass"], timeout=1)
    assert stops == [123]


@pytest.mark.proves("shared_kernel.child_processes_are_bounded_and_isolated")
def test_a_child_gets_closed_input_utf8_and_no_inherited_data_root(monkeypatch, tmp_path):
    """A real interpreter sees only the caller's chosen process environment."""
    name = "alphalattice.kernel.shared_kernel.spawn"
    assert importlib.util.find_spec(name) is not None
    from alphalattice.kernel.shared_kernel import spawn

    monkeypatch.setenv("ALPHALATTICE_DATA_ROOT", str(tmp_path / "person-root"))
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


@pytest.mark.proves("shared_kernel.child_processes_are_bounded_and_isolated")
@pytest.mark.parametrize("parent", ["live", "exited", "returned"])
def test_a_timed_out_child_and_its_descendant_are_stopped(parent, tmp_path):
    """A real timed-out process cannot leave its child behind."""
    from alphalattice.kernel.shared_kernel import spawn

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


@pytest.mark.proves("shared_kernel.child_processes_are_bounded_and_isolated")
def test_an_interactive_session_feeds_turns_and_captures_streams():
    """A real child consumes two turns before EOF and the owner closes both pipes."""
    from alphalattice.kernel.shared_kernel import spawn

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


@pytest.mark.proves("shared_kernel.child_processes_are_bounded_and_isolated")
def test_an_interactive_session_stops_at_its_deadline():
    """The deadline terminates the real child and the owner reports timeout."""
    from alphalattice.kernel.shared_kernel import spawn

    with spawn.InteractiveSession(timeout=0.5, stop_scope="tree") as session:
        child = session.start(
            [sys.executable, "-c", "import threading; threading.Event().wait(20)"]
        )
        assert child.wait(timeout=8) != 0
        with pytest.raises(subprocess.TimeoutExpired):
            session.check()


@pytest.mark.proves("shared_kernel.child_processes_are_bounded_and_isolated")
def test_an_interactive_session_names_its_stop_scope(monkeypatch):
    """Root scope stops only the child; tree scope selects the POSIX group barrier."""
    from alphalattice.kernel.shared_kernel import spawn

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
        spawn.os, "killpg", lambda pid, sig: calls.append(("group", pid)), raising=False
    )
    monkeypatch.setattr(spawn.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(spawn.psutil, "process_iter", lambda: [])
    root = spawn.InteractiveSession(timeout=1, stop_scope="root")
    root.child = Child()
    root.stop()
    assert calls == ["root"]
    calls.clear()
    tree = spawn.InteractiveSession(timeout=1, stop_scope="tree")
    tree.child = Child()
    tree.stop()
    assert calls == [("group", 123), "wait"]


@pytest.mark.proves("shared_kernel.child_processes_are_bounded_and_isolated")
def test_concurrent_stops_wait_for_the_same_termination_barrier():
    """Concurrent callers return only after the owned child has stopped."""
    from alphalattice.kernel.shared_kernel import spawn

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
from alphalattice.kernel.shared_kernel import spawn, waits

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


@pytest.mark.proves("shared_kernel.child_processes_are_bounded_and_isolated")
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
