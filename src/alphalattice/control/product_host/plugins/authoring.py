"""`plugin new` and `plugin verify`: a workspace plugin written from a template, then checked as
the Host would run it, with no Host, nothing enabled and no confirmation asked.

`plugin new <name>` writes `plugins/<name>/`: its manifest (`plugin.json`), its entry module, a
sample slice (`sample.json`) and a test, so the author writes only the logic. The template is a
factor. Its slice is a few sessions of a few symbols: `dates` (one ISO date per session, oldest
first), `symbols`, and `close` and `volume` (one row per session, one value per symbol); it
answers `dates` and `values`, one row per session.

`plugin verify <name>` runs the plugin in its guarded child under its own lease with the network
off whatever the lease grants, on its sample slice, and stops at the first fault, naming it with
what is wrong and what to change: the manifest or a value outside its schema; a network reach; a
write outside its staging folder (the guard's refusal, the system's own where it has one, and on
disk the plugin's folder before and after each run); a rerun whose answer differs by a byte; its
memory lease, on every run; a cancel, sent at the guarded child's started signal once the plugin
runs, that takes more than a second to settle with its tree gone; and look-ahead, a value at a
session T that changes when the sessions after T are added. An answer has one row per session of
its slice: its `dates` are the slice's, and every list as long as `dates` is a per-session field,
cut with it. Both commands read and write the workspace only through `plugins/<name>/`, refusing
a link that leaves the workspace.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from alphalattice.control.product_host.plugins.registry import (
    PluginRefused,
    VerifiedPlugin,
    confined_file,
    plugin_name_valid,
    workspace_plugin,
)
from alphalattice.protocols.plugin_contract import ActionInvocation, CatalogRead, ExecutionContext

if TYPE_CHECKING:
    from alphalattice.control.task_control.plugin_staging import StagingRefused

CANCEL_SECONDS = 1.0
"""The longest a cancelled run may take to end."""
LATER_SESSIONS = 3
"""Look-ahead: the slice is cut this many sessions before its end, and the cut's values must be
the whole slice's at each session they share."""
_SAMPLE = "sample.json"
CHECKS = (
    "schema",
    "network",
    "write_outside_staging",
    "deterministic",
    "memory_cap",
    "cancel",
    "look_ahead",
)
"""Verify's checks, in the order it runs them."""

_LEASE = {"memory_bytes": 536870912, "threads": 1, "timeout_ms": 30000}
_ROWS = {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
_DATES = {"type": "array", "items": {"type": "string", "format": "date"}}
_INPUT_SCHEMA = {
    "type": "object",
    "required": ["dates", "symbols", "close", "volume"],
    "properties": {
        "dates": _DATES,
        "symbols": {"type": "array", "items": {"type": "string"}},
        "close": _ROWS,
        "volume": _ROWS,
    },
}
_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["dates", "values"],
    "properties": {
        "dates": _DATES,
        "values": {
            "type": "array",
            "items": {"type": "array", "items": {"type": ["number", "null"]}},
        },
    },
}
_ENTRY_TEXT = '''"""{name}: a factor plugin, one value per session and symbol of its slice.

`alphalattice plugin verify {name}` runs it in its guarded child on sample.json and names each
fault with what to change; test_{module}.py runs it here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def run(params: Mapping[str, Any], context: object) -> dict[str, Any]:
    """One value per session and symbol, from that session and earlier ones only.

    Args:
        params: The slice plugin.json's input_schema declares: `dates` (one ISO date per
            session, oldest first), `symbols`, and `close` and `volume` (one row per session,
            one value per symbol).
        context: The run's ExecutionContext; `context.staging_dir` is the one folder it may
            write.

    Returns:
        `dates` and `values`: one row per session, one value per symbol, or None where the
        value is not defined yet, as plugin.json's output_schema declares.
    """
    del context
    close = params["close"]
    values: list[list[float | None]] = [[None] * len(params["symbols"])]
    for earlier, today in zip(close, close[1:], strict=False):
        values.append(
            [now / before - 1.0 if before else None for before, now in zip(earlier, today)]
        )
    return {{"dates": list(params["dates"]), "values": values}}
'''
_TEST_TEXT = '''"""{name}'s own test: `alphalattice plugin verify {name}` passes it.

Verify runs `run` in its guarded child on sample.json, as the Host would, and names each fault
with what to change; this test never runs the plugin in its own process.
"""

import json
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
VERIFY = "import sys; from alphalattice.control.product_host.composition.entry import main; "
VERIFY += "sys.exit(main())"


def test_{module}_passes_plugin_verify() -> None:
    """Verify passes every check; a failure names the check, what is wrong and the change."""
    line = ["--workspace", str(WORKSPACE), "plugin", "verify", "{name}"]
    done = subprocess.run(  # isolated, so no module beside this test shadows the installed one
        [sys.executable, "-I", "-c", VERIFY, *line],
        capture_output=True,
        text=True,
        timeout=300,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # no console window on Windows
    )
    answer = json.loads(done.stdout)["data"]
    assert answer["status"] == "PASSED", answer.get("failed", done.stderr)
'''
# Each guard refusal's check, verify's code, what is wrong (its detail filled in) and the change.
_GUARD = {
    "task_control.plugin_network_refused": (
        "network",
        "plugin_verify.network",
        "it reached for the network ({detail}) under a lease without network",
        "remove the network call and compute from the inputs run() is given; or declare "
        '"network": true in plugin.json\'s lease, which a person then confirms',
    ),
    "task_control.plugin_write_outside_staging": (
        "write_outside_staging",
        "plugin_verify.write_outside_staging",
        "it wrote outside its staging folder ({detail})",
        "return the answer from run(), and write any file only under context.staging_dir, "
        "never through a link",
    ),
    "task_control.plugin_subprocess_refused": (
        "network",
        "plugin_verify.subprocess",
        "it started a process ({detail})",
        "do the work in run() itself",
    ),
    "task_control.plugin_import_outside_environment": (
        "schema",
        "plugin_verify.import_outside_environment",
        "it imported {detail}, which neither the installed environment nor its own files hold",
        "import only what the installed environment holds, or list the module's file in "
        "plugin.json's files",
    ),
    "task_control.plugin_lease_memory": (
        "memory_cap",
        "plugin_verify.memory_cap",
        "it used more memory than its lease's memory_bytes",
        "use less memory, or raise lease.memory_bytes in plugin.json",
    ),
    "task_control.plugin_lease_timeout": (
        "memory_cap",
        "plugin_verify.timeout",
        "it did not answer within its lease's timeout_ms",
        "make run() faster, or raise lease.timeout_ms in plugin.json",
    ),
    "task_control.plugin_failed": (
        "schema",
        "plugin_verify.failed",
        "run() raised {detail}",
        "fix the error; its test (pytest test_<module>.py in its folder) shows it",
    ),
    "task_control.plugin_input_invalid": (
        "schema",
        "plugin_verify.sample_invalid",
        "sample.json does not match plugin.json's input_schema",
        "make sample.json match input_schema, or correct input_schema",
    ),
    "task_control.plugin_output_invalid": (
        "schema",
        "plugin_verify.output_invalid",
        "its answer does not match plugin.json's output_schema",
        "return a JSON object that matches output_schema, or correct output_schema",
    ),
}


@dataclass(frozen=True, slots=True)
class Run:
    """One run of a plugin: its checked answer's bytes and its peak memory, or, for a cancelled
    run, the seconds from the cancel to its child's end (None: no running child was cancelled)."""

    answer: bytes | None
    peak_bytes: int
    stop_seconds: float | None = None


Runner = Callable[[VerifiedPlugin, Mapping[str, Any], bool], Run]
"""Run a verified plugin on a slice, cancelling it once its child runs when told to; a refusal is
raised as `StagingRefused`."""


class _Fault(Exception):
    def __init__(self, check: str, code: str, what: str, change: str) -> None:
        super().__init__(code)
        self.check, self.code, self.what, self.change = check, code, what, change


def new_plugin(workspace: Path, name: str) -> dict[str, Any]:
    """`plugin new`: write a factor plugin's manifest, entry module, sample slice and test.

    Returns:
        `WRITTEN`, the files written, and the next step.

    Raises:
        ValueError: `plugin_new.name_invalid` for a name no plugin may have;
            `plugin_new.exists:<name>` when its folder or a file of it is already there;
            `plugin_new.outside_workspace:<name>` when a link would take a write outside the
            workspace.
    """
    if not plugin_name_valid(name):
        raise ValueError(f"plugin_new.name_invalid:{name}")
    module = name.replace("-", "_").replace(".", "_")
    root, relative = workspace.resolve(), Path("plugins") / name
    outside = f"plugin_new.outside_workspace:{name}"
    folder = _confined(root, relative, outside)
    if folder.exists():
        raise ValueError(f"plugin_new.exists:{name}")
    manifest = {
        "effect_mode": "READ_ONLY",
        "lease": _LEASE,
        "entry": f"{module}:run",
        "files": [f"{module}.py"],
        "input_schema": _INPUT_SCHEMA,
        "output_schema": _OUTPUT_SCHEMA,
    }
    files = {
        "plugin.json": json.dumps(manifest, indent=2) + "\n",
        f"{module}.py": _ENTRY_TEXT.format(name=name, module=module),
        _SAMPLE: json.dumps(_sample(), indent=1) + "\n",
        f"test_{module}.py": _TEST_TEXT.format(name=name, module=module),
    }
    folder.mkdir(parents=True)
    for file, text in files.items():
        try:
            with _confined(root, relative / file, outside).open(
                "x", encoding="utf-8", newline="\n"
            ) as written:
                written.write(text)
        except FileExistsError as error:
            raise ValueError(f"plugin_new.exists:{name}") from error
    return {
        "status": "WRITTEN",
        "plugin": name,
        "files": [f"plugins/{name}/{file}" for file in files],
        "next_action": "WRITE_RUN_THEN_PLUGIN_VERIFY",
        "next": f"write run() in plugins/{name}/{module}.py, then `plugin verify {name}`",
    }


def _sample() -> dict[str, Any]:
    """Twelve weekday sessions of three symbols, the same on every machine."""
    dates: list[str] = []
    day = date(2024, 1, 2)
    while len(dates) < 12:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day += timedelta(days=1)
    symbols = ["AAA", "BBB", "CCC"]
    return {
        "dates": dates,
        "symbols": symbols,
        "close": [
            [round(50.0 + 10 * j + i * (1 + j) - (i % 3) * 0.5, 2) for j in range(3)]
            for i in range(12)
        ],
        "volume": [[float(1000 * (j + 1) + 37 * i) for j in range(3)] for i in range(12)],
    }


def verify_plugin(workspace: Path, name: str, *, run: Runner | None = None) -> dict[str, Any]:
    """`plugin verify`: run a workspace plugin's checks in order, stopping at the first fault.

    Args:
        workspace: The workspace whose `plugins/<name>/` holds the plugin.
        name: The plugin's name.
        run: Runs the plugin; its guarded child when omitted.

    Returns:
        `PASSED` with every check and the fingerprint plugins.toml enables it by, or `FAILED`
        with the checks that passed, the failing one's code, what is wrong and what to change.
    """
    passed: list[str] = []
    fingerprint = None
    try:
        verified = _verified(workspace, name)
        fingerprint = verified.spec.code_role_hash
        sample = _sample_of(workspace, name)
        _check_input(verified, sample)
        for check in _runs(workspace / "plugins" / name, verified, sample, run or guarded_run):
            passed.append(check)
    except _Fault as fault:
        return {
            "status": "FAILED",
            "plugin": name,
            "fingerprint": fingerprint,
            "passed": passed,
            "failed": {
                "check": fault.check,
                "code": fault.code,
                "what": fault.what,
                "change": fault.change,
            },
            "next_action": "CHANGE_THE_PLUGIN_THEN_VERIFY_AGAIN",
        }
    return {
        "status": "PASSED",
        "plugin": name,
        "fingerprint": fingerprint,
        "passed": passed,
        "next_action": "ENABLE_IT_IN_PLUGINS_TOML",
        "next": f'add "{name}" = "{fingerprint}" under [enabled] in the workspace\'s plugins.toml',
    }


def _verified(workspace: Path, name: str) -> VerifiedPlugin:
    if not plugin_name_valid(name):
        raise _Fault(
            "schema",
            "plugin_verify.name_invalid",
            f"{name!r} is not a plugin's name",
            "name it in lowercase letters, digits, '-', '_' and '.', starting with a letter",
        )
    try:
        return workspace_plugin(workspace, name)
    except PluginRefused as error:
        raise _Fault(
            "schema",
            "plugin_verify.manifest_invalid",
            f"plugins/{name}/plugin.json is not a plugin's manifest ({error.code})",
            "give plugin.json effect_mode, lease, input_schema, output_schema, an entry "
            '"<module>:run" and files listing that module, each file inside its folder; '
            "`plugin new` writes one",
        ) from error


def _confined(root: Path, relative: Path, code: str) -> Path:
    """The registry's confined path; a link or `..` leaving the workspace is refused by `code`."""
    try:
        path: Path = confined_file(root, relative)
    except ValueError as error:
        raise ValueError(code) from error
    return path


def _sample_of(workspace: Path, name: str) -> dict[str, Any]:
    try:
        path = confined_file(workspace.resolve(), Path("plugins") / name / _SAMPLE)
        sample = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # a link leaving the workspace is a ValueError too
        sample = None
    if not isinstance(sample, dict):
        raise _Fault(
            "schema",
            "plugin_verify.sample_unreadable",
            f"plugins/{name}/{_SAMPLE} is missing, not a JSON object, or reached through a link "
            "out of the workspace",
            "write a sample slice of the inputs input_schema declares; `plugin new` writes one",
        )
    return sample


def _check_input(verified: VerifiedPlugin, sample: Mapping[str, Any]) -> None:
    from alphalattice.control.task_control.plugin_staging import (  # lazy: verify only
        StagingRefused,
        check_plugin_input,
    )

    try:
        check_plugin_input(_invocation(verified, sample), verified.spec)
    except StagingRefused as error:
        raise _guard_fault(error) from error


def _runs(
    folder: Path, verified: VerifiedPlugin, sample: Mapping[str, Any], run: Runner
) -> Iterator[str]:
    """Yield each check's name as it passes; raise the first fault."""
    from alphalattice.control.task_control.plugin_staging import (  # lazy: verify only
        StagingRefused,
    )

    def once(params: Mapping[str, Any], *, cancel: bool = False) -> Run:
        before = _tree(folder)
        try:
            answer = run(verified, params, cancel)
        except StagingRefused as error:
            raise _guard_fault(error) from error
        changed = sorted({path for path, _, _ in _tree(folder) ^ before})
        if changed:
            raise _Fault(
                "write_outside_staging",
                "plugin_verify.write_outside_staging",
                "its own folder changed during the run: " + ", ".join(changed),
                "return the answer from run(), and write any file only under context.staging_dir",
            )
        if answer.peak_bytes > verified.spec.lease.memory_bytes:
            raise _guard_fault(None, "task_control.plugin_lease_memory")
        return answer

    first = once(sample)
    yield from ("schema", "network", "write_outside_staging")
    if once(sample).answer != first.answer:
        raise _Fault(
            "deterministic",
            "plugin_verify.nondeterministic",
            "two runs on the same sample.json answered differently",
            "derive every value from the inputs alone, in one type: no clock, no unseeded "
            "random draw, no order taken from a set or from the file system",
        )
    yield from ("deterministic", "memory_cap")
    stopped = once(sample, cancel=True).stop_seconds
    if stopped is None:
        raise _Fault(
            "cancel",
            "plugin_verify.cancel_unproved",
            "no cancel reached its running guarded child: the run ended or stopped before its "
            "started signal",
            "rerun `plugin verify`; if it recurs, report the system",
        )
    if stopped > CANCEL_SECONDS:
        raise _Fault(
            "cancel",
            "plugin_verify.cancel_slow",
            f"a cancelled run took {stopped:.2f} s to end, more than {CANCEL_SECONDS:.0f} s",
            "keep no work running past a cancel: no thread, process or native call that "
            "outlives run()",
        )
    yield "cancel"
    cut = _cut(sample)
    whole = _answered(first.answer, sample)
    if changed := _looked_ahead(whole, _answered(once(cut).answer, cut)):
        raise _Fault(
            "look_ahead",
            "plugin_verify.look_ahead",
            f"its value at {changed} changed when the {LATER_SESSIONS} sessions after it "
            "were added",
            "compute each session's value from that session and earlier ones only: no "
            "centred window, no shift back from a later session, no normalising by the "
            "whole slice",
        )
    yield "look_ahead"


def _guard_fault(error: StagingRefused | None, code: str | None = None) -> _Fault:
    code = code or (error.code if error is not None else "")
    refusals = error.refusals if error is not None else ()
    if refusals:
        detail = refusals[0].get("detail", "")
    else:  # a child's own error carries its detail before its next step
        detail = error.next_step.rsplit("; ", 1)[0] if error is not None else ""
    if code in _GUARD:
        check, verify_code, what, change = _GUARD[code]
        return _Fault(check, verify_code, what.format(detail=detail), change)
    return _Fault(
        "schema",
        code,
        str(error) if error is not None else code,
        error.next_step if error is not None else "rerun `plugin verify`",
    )


def _tree(folder: Path) -> set[tuple[str, int, int]]:
    """Each file under the plugin's folder: its path, size and modification time."""
    found: set[tuple[str, int, int]] = set()

    def unreadable(error: OSError) -> None:
        raise error

    for top, folders, files in os.walk(folder, onerror=unreadable):
        folders[:] = [name for name in folders if name != "__pycache__"]
        for name in files:
            stat = os.stat(os.path.join(top, name))
            relative = Path(top, name).relative_to(folder).as_posix()
            found.add((relative, stat.st_size, stat.st_mtime_ns))
    return found


def _unproved(what: str) -> _Fault:
    return _Fault(
        "look_ahead",
        "plugin_verify.look_ahead_unproved",
        f"look-ahead could not be checked: {what}",
        f"give sample.json more than {LATER_SESSIONS} sessions in `dates`, and answer the "
        "slice's `dates` with each per-session field one row per session",
    )


def _cut(sample: Mapping[str, Any]) -> dict[str, Any]:
    """The slice without its last LATER_SESSIONS sessions."""
    dates = sample.get("dates")
    if not isinstance(dates, list) or len(dates) <= LATER_SESSIONS:
        raise _unproved(f"sample.json has {LATER_SESSIONS} sessions or fewer in `dates`")
    keep = len(dates) - LATER_SESSIONS
    return {
        field: value[:keep] if isinstance(value, list) and len(value) == len(dates) else value
        for field, value in sample.items()
    }


def _answered(answer: bytes | None, params: Mapping[str, Any]) -> dict[str, list[Any]]:
    """The answer's per-session fields, its `dates` the slice's own: one row per session."""
    parsed = json.loads(answer) if answer is not None else None
    if not isinstance(parsed, dict) or parsed.get("dates") != params["dates"]:
        raise _unproved("its answer's `dates` are not its slice's")
    rows = {
        field: value
        for field, value in parsed.items()
        if isinstance(value, list) and len(value) == len(params["dates"])
    }
    if len(rows) < 2:
        raise _unproved("its answer has no per-session field beside `dates`")
    return rows


def _looked_ahead(whole: Mapping[str, list[Any]], cut: Mapping[str, list[Any]]) -> str | None:
    """The first shared session whose values the cut answers differently from the whole."""
    if cut.keys() != whole.keys():
        raise _unproved("its answers on the slice and on the cut slice have different fields")
    for at, session in enumerate(cut["dates"]):
        if any(cut[field][at] != whole[field][at] for field in whole):
            return str(session)
    return None


def _invocation(verified: VerifiedPlugin, params: Mapping[str, Any]) -> ActionInvocation:
    read = CatalogRead(
        snapshot_id="plugin-verify",
        input_versions=dict.fromkeys(verified.spec.inputs, 0),
    )
    return verified.spec.invoke(params, read)


def guarded_run(verified: VerifiedPlugin, params: Mapping[str, Any], cancel: bool) -> Run:
    """Run the plugin in its guarded child with the network off, in a staging folder of its own
    that is removed after; with `cancel`, cancel it at its guarded child's started signal and time
    how long the stopped run takes to settle."""
    from alphalattice.control.task_control.plugin_child import (  # lazy: verify only
        GuardedPlugin,
    )
    from alphalattice.control.task_control.plugin_staging import (  # lazy: verify only
        OUTPUT,
        checked_plugin_output,
    )

    invocation = _invocation(verified, params)
    with tempfile.TemporaryDirectory(prefix="plugin-verify-") as folder:
        staging, data = Path(folder) / "staging", Path(folder) / "data"
        staging.mkdir()
        data.mkdir()
        context = ExecutionContext(
            invocation_id=uuid4(),
            catalog_snapshot_id=invocation.catalog_snapshot_id,
            readonly_data_root=data,
            staging_dir=staging,
            lease=verified.spec.lease.model_copy(update={"network": False}),
        )
        plugin = GuardedPlugin(verified.entry, verified.sources, invocation)
        if not cancel:
            receipt = plugin(context)
            checked_plugin_output(staging / OUTPUT, verified.spec)
            return Run((staging / OUTPUT).read_bytes(), receipt.peak_bytes)
        return Run(None, 0, cancel_seconds(plugin, context))


def cancel_seconds(plugin: Callable[..., object], context: ExecutionContext) -> float | None:
    """Seconds from a cancel sent at `plugin`'s started signal (GuardedPlugin's, once its guarded
    child is running) until the stopped run settles, its tree gone; None when no cancel was sent
    at that signal, so a stop during the launch bootstrap is never credited."""
    from alphalattice.control.task_control.plugin_staging import (  # lazy: verify only
        StagingRefused,
    )
    from alphalattice.kernel.execution import (  # lazy: verify only
        ExecutionStopped,
        cancel_execution,
        execution_cancellation,
    )

    execution, sent = uuid4(), list[float]()

    def started() -> None:
        sent.append(time.perf_counter())
        cancel_execution(execution)

    with execution_cancellation(execution):
        try:
            plugin(context, started=started)
        except ExecutionStopped:
            if sent:
                return time.perf_counter() - sent[0]
        except StagingRefused:
            if sent:
                raise
    return None


__all__ = [
    "CHECKS",
    "Run",
    "Runner",
    "cancel_seconds",
    "guarded_run",
    "new_plugin",
    "verify_plugin",
]
