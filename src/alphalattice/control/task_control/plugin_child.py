"""A user's or an agent's plugin run in its own guarded child, as a staging run's executor.

`GuardedPlugin` is the `Plugin` a SANDBOX_SUBPROCESS capability's staging run calls. It starts one
child (`plugin_guard`) under the capability's lease: the lease's timeout ends the tree, its memory
is the tree's budget (the Job's limit on Windows, the cgroup scope on Linux, polled resident memory
on macOS), its threads bound the native libraries, and on Windows the Job holds one process. The
child's working directory and HOME, USERPROFILE, TMP, TEMP, TMPDIR and XDG_CACHE_HOME point into
the run's staging folder, and it inherits no other environment than the system's own names. The
request is JSON-RPC on the child's standard input. Its one answer goes to an unnamed file the
Host opened, read up to `ANSWER_BYTES`, and its error stream, the plugin's prints, goes nowhere,
so no plugin output fills the Host's memory or reaches a refusal. A result counts only from a
child that exited 0. The data stays in the staging folder. Every refusal is by name with its next
step, and whatever the child left in its home folder is removed before the staging run checks
what was staged.

Given `started`, the run is held: the child makes its started mark once its hook has heard its
probe, runs the plugin and stays alive, and `started` is called once that mark is seen, so a stop
sent from it ends a running guarded child, never a launch bootstrap. The stopped run settles as
`ExecutionStopped`, raised only once its tree has ended.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alphalattice.capabilities.domain_contracts import spawn
from alphalattice.capabilities.domain_contracts.waits import WaitStopped, WaitTimeout, wait_until
from alphalattice.control.task_control import plugin_guard
from alphalattice.control.task_control.plugin_staging import StagingRefused
from alphalattice.kernel.config import current as process_config
from alphalattice.kernel.resources import NATIVE_THREAD_VARIABLES
from alphalattice.protocols.plugin_contract import (
    ActionInvocation,
    ActionReceipt,
    ExecutionContext,
)

ANSWER_BYTES = 1 << 16
"""The most of the child's answer the Host reads; a longer one is no answer."""
_DETAIL_CHARACTERS = 300
_STARTED_POLL_SECONDS = 0.025
"""How often a held run looks for its child's started mark: the kernel's stop-poll interval."""
_SYSTEM = ("SYSTEMROOT", "WINDIR", "PATH", "LANG", "LC_ALL")
_NEXT = {
    "task_control.plugin_network_refused": (
        "remove the network call, or declare network in the plugin's lease"
    ),
    "task_control.plugin_subprocess_refused": "do the work in the plugin's own process",
    "task_control.plugin_write_outside_staging": (
        "write only inside the run's staging folder, ExecutionContext.staging_dir, never "
        "through a link"
    ),
    "task_control.plugin_import_outside_environment": (
        "import only what the locked environment or the plugin's own files hold"
    ),
    "task_control.plugin_guard_unavailable": "run the plugin on a standard CPython",
    "task_control.plugin_confinement_failed": "rerun; report the system if it recurs",
    "task_control.plugin_failed": "repair the plugin's error and rerun",
    "task_control.plugin_child_unanswered": "rerun; report the system if it recurs",
}


@dataclass(frozen=True)
class GuardedPlugin:
    """One invocation of a plugin's verified entry and source bytes, run in a guarded child."""

    entry: str
    sources: Mapping[str, bytes]
    invocation: ActionInvocation

    def __call__(
        self, context: ExecutionContext, started: Callable[[], None] | None = None
    ) -> ActionReceipt:
        """Run the plugin in `context`'s staging folder under its lease and receipt its run;
        given `started`, hold the run and call `started` once its guarded child is running."""
        home = context.staging_dir / plugin_guard.HOME
        home.mkdir()
        lease = context.lease
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "plugin.run",
            "params": {
                "entry": self.entry,
                "sources": {
                    path: base64.b64encode(body).decode() for path, body in self.sources.items()
                },
                "params": self.invocation.model_dump(mode="json")["params"],
                "staging_dir": str(context.staging_dir),
                "context": context.model_dump(mode="json"),
                "hold": started is not None,
            },
        }
        ended = threading.Event()
        signal = threading.Thread(
            target=signal_started,
            args=(home / plugin_guard.STARTED, started, ended, lease.timeout_ms / 1000),
        )
        try:
            if started is not None:
                signal.start()
            with tempfile.TemporaryFile() as answer:  # opened here, so the child cannot name it
                completed = spawn.run(
                    _command(),
                    input=json.dumps(request).encode(),
                    stdout=answer,
                    stderr=subprocess.DEVNULL,
                    timeout=lease.timeout_ms / 1000,
                    cwd=home,
                    env=_environment(home, lease.threads),
                    process_kind="plugin_child",
                    leased_memory=lease.memory_bytes,
                    active_processes=1,
                )
                answer.seek(0)
                frame = answer.read(ANSWER_BYTES + 1)
        except subprocess.TimeoutExpired as error:
            raise StagingRefused(
                "task_control.plugin_lease_timeout",
                "make the plugin faster or declare a longer timeout in its lease",
            ) from error
        except spawn.TreeMemoryLimit as error:
            raise StagingRefused(
                "task_control.plugin_lease_memory",
                "use less memory or declare a larger memory lease",
            ) from error
        finally:
            ended.set()
            if signal.is_alive():
                signal.join(_STARTED_POLL_SECONDS * 4)  # its wait ends at once on `ended`
            if sys.platform == "win32":
                plugin_guard.clear_labels(str(context.staging_dir))
            shutil.rmtree(home)
        result = child_answer(completed.returncode, frame)
        return self.invocation.staged_receipt(
            context.invocation_id, wall_ms=result["wall_ms"], peak_bytes=result["peak_bytes"]
        )


def signal_started(
    mark: Path, started: Callable[[], None] | None, ended: threading.Event, seconds: float
) -> None:
    """Call `started` once the guarded child has made its started mark, unless its run ends
    first or its lease's timeout passes."""
    if started is None:
        return
    try:
        wait_until(
            lambda _: mark.exists(),
            timeout=seconds,
            what="the guarded child's started mark",
            interval=_STARTED_POLL_SECONDS,
            max_interval=_STARTED_POLL_SECONDS,
            stop=ended,
        )
    except (WaitStopped, WaitTimeout):  # the run ended, or its lease's timeout ends it
        return
    started()


def _command() -> list[str]:
    """The interpreter itself, so the Job's one process is the plugin's: a Windows venv's
    launcher would start the interpreter as a second process."""
    python = sys.executable
    if sys.platform == "win32" and sys.prefix != sys.base_prefix:
        python = getattr(sys, "_base_executable", python)
    return [python, "-I", "-B", "-m", plugin_guard.__name__]


def _environment(home: Path, threads: int) -> dict[str, str]:
    inherited = process_config().child_environment()
    environment = {name: inherited[name] for name in _SYSTEM if name in inherited}
    environment.update(dict.fromkeys(plugin_guard.REDIRECTED, str(home)))
    environment.update(dict.fromkeys(NATIVE_THREAD_VARIABLES, str(threads)))
    if sys.platform == "win32" and sys.prefix != sys.base_prefix:
        environment["__PYVENV_LAUNCHER__"] = sys.executable  # the venv the interpreter reads
    return environment


def child_answer(returncode: int, frame: bytes) -> Mapping[str, Any]:
    """The result of a child that exited 0 with its one JSON-RPC answer, or the refusal by name
    its answer carries, with the hook's record; anything else is no answer, refused with the
    exit status alone."""
    unanswered = StagingRefused(
        "task_control.plugin_child_unanswered",
        f"exit {returncode}; {_NEXT['task_control.plugin_child_unanswered']}",
    )
    named: StagingRefused | None = None
    try:
        if len(frame) > ANSWER_BYTES:
            raise ValueError("the answer is longer than its bound")
        (line,) = frame.decode("utf-8").splitlines()
        answer = json.loads(line)
        if answer.get("jsonrpc") != "2.0" or answer.get("id") != 1:
            raise ValueError("not this request's answer")
        if "error" in answer:
            data = answer["error"]["data"]
            code, detail = data["code"], str(data["detail"])[:_DETAIL_CHARACTERS]
            refusals = tuple(
                {key: str(value)[:_DETAIL_CHARACTERS] for key, value in item.items()}
                for item in data["refusals"]
            )
            if code not in _NEXT:
                raise ValueError("an unregistered refusal")
            named = StagingRefused(code, f"{detail}; {_NEXT[code]}", refusals)
        else:
            result: Mapping[str, Any] = answer["result"]
            if returncode != 0 or not all(
                type(result.get(name)) is int and result[name] >= 0
                for name in ("wall_ms", "peak_bytes")
            ):
                raise ValueError("a result without a clean exit")
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise unanswered from error
    if named is not None:
        raise named
    return result
