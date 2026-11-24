"""Resolve product resources and discover its command in the caller's installation."""

from __future__ import annotations

import shutil
import subprocess
from contextlib import suppress
from functools import cache
from pathlib import Path

from alphalattice.capabilities.domain_contracts.runtime_resources import (
    REFUSAL_WORDS_RESOURCE as REFUSAL_WORDS_RESOURCE,
)
from alphalattice.capabilities.domain_contracts.runtime_resources import (
    json_file as json_file,
)
from alphalattice.capabilities.domain_contracts.runtime_resources import (
    resolve_playpen_root as resolve_playpen_root,
)
from alphalattice.capabilities.domain_contracts.runtime_resources import (
    resource_path as resource_path,
)
from alphalattice.capabilities.domain_contracts.runtime_resources import (
    runtime_json as runtime_json,
)
from alphalattice.capabilities.domain_contracts.runtime_resources import (
    source_root as source_root,
)
from alphalattice.capabilities.domain_contracts.runtime_resources import (
    tracked_path as tracked_path,
)
from alphalattice.kernel.config import current as process_config


def command_prefix(*, portable: bool = False) -> tuple[str, ...]:
    """Return the tool-installed command, available in every host's fresh shell."""
    if portable or (
        "UV_RUN_RECURSION_DEPTH" not in process_config().public and shutil.which("alphalattice")
    ):
        return ("alphalattice",)
    uv = shutil.which("uv")
    folder = None
    if uv:
        # Catch outside @cache so a failed probe cannot become a retained value.
        with suppress(OSError, subprocess.SubprocessError, UnicodeError):
            folder = _tool_bin(uv)
    command = shutil.which("alphalattice", path=folder) if folder else None
    if command:
        return (str(Path(command).absolute()),)
    return ("alphalattice",)


@cache
def _tool_bin(uv: str) -> str:
    """Cache successful directory discovery; failed probes remain retriable."""
    from alphalattice.capabilities.domain_contracts import spawn  # lazy: tool directory discovery

    found = spawn.run(
        [uv, "tool", "dir", "--bin"],
        process_kind="uv_tool_bin_probe",
        capture_output=True,
        encoding="utf-8",
        timeout=10,
        check=True,
    ).stdout.strip()
    if not found:
        raise OSError
    return found
