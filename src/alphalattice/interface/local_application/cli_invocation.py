"""Caller presentation and captured streams for an admitted one-response CLI invocation."""

from __future__ import annotations

import base64
import builtins
import codecs
import io
import sys
import tempfile
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from pathlib import Path
from typing import NamedTuple, TextIO

from alphalattice.interface.local_application.thin_cli import CLI_ENVIRONMENT, CLI_OPERATION, select
from alphalattice.kernel.config import current as process_config
from alphalattice.kernel.shared_kernel.project_layout import command_prefix as layout_command_prefix


class Call(NamedTuple):
    """State specific to the forwarded command, isolated from other requests."""

    cwd: Path
    environment: dict[str, str]
    stdout: TextIO
    stderr: TextIO


CLI_CALL: ContextVar[Call | None] = ContextVar("cli_invocation", default=None)


def environment() -> Mapping[str, str]:
    """Read the command's presentation environment."""
    call = CLI_CALL.get()
    return process_config().public if call is None else call.environment


def command_prefix(*, portable: bool = False) -> tuple[str, ...]:
    """Keep the proven portable launcher of a forwarded command."""
    return layout_command_prefix(portable=portable or CLI_CALL.get() is not None)


def cwd() -> Path:
    """Read the command's own directory without changing the process directory."""
    call = CLI_CALL.get()
    return Path.cwd() if call is None else call.cwd


def search_path() -> str | None:
    """Read the command's PATH: a forwarded command's caller's; None, this process's own."""
    call = CLI_CALL.get()
    return None if call is None else call.environment.get("PATH")


def temporary() -> Path:
    """Read the command's temporary folder: the one a forwarded command's caller chose."""
    call = CLI_CALL.get()
    folder = None if call is None else call.environment.get("TMPDIR")
    return Path(folder) if folder else Path(tempfile.gettempdir())


def workspace(path: Path | None) -> Path | None:
    """Anchor a forwarded workspace argument in the caller's directory."""
    call = CLI_CALL.get()
    return path if call is None or path is None else call.cwd / path


def stream(file: TextIO | None = None, *, error: bool = False) -> TextIO:
    """Map the two standard output streams to the current call's captures."""
    stdout, stderr = sys.stdout, sys.stderr  # noqa: TID251  # process streams on the full path
    chosen = (stderr if error else stdout) if file is None else file
    call = CLI_CALL.get()
    if call is not None:
        if chosen is stdout:
            return call.stdout
        if chosen is stderr:
            return call.stderr
    return chosen


def cli_print(
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n",
    file: TextIO | None = None,
    flush: bool = False,
) -> None:
    """Print using the same semantics for process stdout and forwarded bytes."""
    builtins.print(*values, sep=sep, end=end, file=stream(file), flush=flush)


def _styles(value: object) -> list[tuple[str, str]]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("invalid CLI stream descriptions")
    styles = []
    for pair in value:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(item, str) and 0 < len(item) <= 64 for item in pair)
        ):
            raise ValueError("invalid CLI stream description")
        encoding, errors = pair
        codecs.lookup(encoding)
        codecs.lookup_error(errors)
        styles.append((encoding, errors))
    return styles


def _settings(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or value.keys() - set(CLI_ENVIRONMENT):
        raise ValueError("unexpected CLI environment")
    if not all(isinstance(item, str) for item in value.values()):
        raise ValueError("invalid CLI environment")
    return value


def _stdout(buffer: io.BytesIO, style: tuple[str, str]) -> io.TextIOWrapper:
    return io.TextIOWrapper(buffer, encoding=style[0], errors=style[1], write_through=True)


def invoke(
    payload: dict[str, object],
    served_workspace: Path,
    dispatch: Callable[[list[str]], int],
    bound: Callable[[], Path | None],
) -> dict[str, object]:
    """Admit the same forms as the client, then run the ordinary CLI with local streams; a form
    that names no workspace runs only where its agent session's binding (`bound`) names this
    one."""
    if payload.keys() != {"operation", "argv", "cwd", "stdin", "streams", "environment"}:
        return {"cli": "local"}
    argv, directory = payload["argv"], payload["cwd"]
    if (
        payload["operation"] != CLI_OPERATION
        or payload["stdin"] is not None
        or not isinstance(argv, list)
        or not all(isinstance(value, str) and "\x00" not in value for value in argv)
        or not isinstance(directory, str)
        or "\x00" in directory
    ):
        return {"cli": "local"}
    caller = Path(directory)
    selected = select(argv, caller)
    if not caller.is_absolute() or selected is None:
        return {"cli": "local"}
    if selected.workspace not in {None, served_workspace}:
        return {"cli": "local"}
    try:
        out_style, err_style = _styles(payload["streams"])
        settings = _settings(payload["environment"])
        out_buffer, err_buffer = io.BytesIO(), io.BytesIO()
        stdout, stderr = _stdout(out_buffer, out_style), _stdout(err_buffer, err_style)
    except (LookupError, ValueError):
        return {"cli": "local"}

    with stdout, stderr:
        scope = CLI_CALL.set(Call(caller, settings, stdout, stderr))
        try:
            if selected.workspace is None and bound() != served_workspace:
                return {"cli": "local"}
            try:
                code = dispatch(argv)
            except SystemExit as exited:
                if not isinstance(exited.code, int):
                    raise
                code = exited.code
            return {
                "cli": "complete",
                "exit": code,
                "stdout": base64.b64encode(out_buffer.getvalue()).decode("ascii"),
                "stderr": base64.b64encode(err_buffer.getvalue()).decode("ascii"),
            }
        finally:
            CLI_CALL.reset(scope)


class ForwardedCli(NamedTuple):
    """The admitted CLI transport request and the Host workspace it reached."""

    document: dict[str, object]
    workspace: Path
