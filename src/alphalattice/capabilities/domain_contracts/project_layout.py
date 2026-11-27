"""Resolve product resources and discover its command in the caller's installation."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
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


def command_prefix(
    *,
    portable: bool = False,
    environment: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> tuple[str, ...]:
    """Return the tool-installed command, available in every host's fresh shell, as this
    process's shell finds it, or the shell whose `environment` and `cwd` a forwarded command
    carries."""
    values = process_config().public if environment is None else environment
    search = None if environment is None else environment.get("PATH")
    if portable or (
        "UV_RUN_RECURSION_DEPTH" not in values and shutil.which("alphalattice", path=search)
    ):
        return ("alphalattice",)
    if shutil.which("uv", path=search) is None:
        return ("alphalattice",)
    command = shutil.which("alphalattice", path=str(tool_bin(values, cwd or Path.cwd())))
    if command:
        return (str(Path(command).absolute()),)
    return ("alphalattice",)


def tool_bin(environment: Mapping[str, str], cwd: Path) -> Path:
    """uv's tool executable directory, the one `uv tool dir --bin` names, read without starting
    uv: UV_TOOL_BIN_DIR, then XDG_BIN_HOME, then the bin beside XDG_DATA_HOME, then the home's
    .local/bin (uv's storage reference, "Tool executables" and "Executable directory"). As uv
    reads them, a nonempty UV_TOOL_BIN_DIR may be relative, to the caller's `cwd`, while an XDG
    setting counts only when absolute."""
    override = environment.get("UV_TOOL_BIN_DIR")
    if override:
        return Path(os.path.abspath(cwd / override))
    for name in ("XDG_BIN_HOME", "XDG_DATA_HOME"):
        value = environment.get(name)
        if value and Path(value).is_absolute():
            return Path(value).parent / "bin" if name == "XDG_DATA_HOME" else Path(value)
    home = environment.get("USERPROFILE" if os.name == "nt" else "HOME")
    return Path(home if home else Path.home(), ".local", "bin")
