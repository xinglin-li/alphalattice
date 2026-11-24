"""Observe the process starts spawn makes, the same way on every operating system."""

from __future__ import annotations

import subprocess
import sys
from functools import partial
from typing import Any, Protocol

from alphalattice.kernel.execution import spawn


class _Patch(Protocol):
    def setattr(self, target: Any, name: str, value: Any) -> None: ...


def observe_starts(patch: _Patch, observe: Any) -> None:
    """Route every start through ``observe(start, command, **kwargs)``, which returns the process.

    ``start(command, **kwargs)`` starts it for real: ``subprocess.Popen``, or on Windows the
    Job-admitted class spawn starts its owned children with, which patching Popen alone misses.
    """
    popen = subprocess.Popen
    patch.setattr(subprocess, "Popen", lambda *args, **kwargs: observe(popen, *args, **kwargs))
    if sys.platform == "win32":
        job_process = spawn.JobProcess

        def admitted(job: Any, environment: Any, command: Any, **kwargs: Any) -> Any:
            return observe(partial(job_process, job, environment), command, **kwargs)

        patch.setattr(spawn, "JobProcess", admitted)
