"""A stage run: each row resolved to one provider on its host, run in order, into one receipt.

The product provides what every host shares: a command row runs its ``run``, and a pytest row its
public selection with its plugin rows as ``-p``. A host adds its own adapters with ``--plugin``, a
module whose ``PROVIDES`` maps row ids to adapters, in the form of pytest's ``-p``. This host skips
a row registered for another, and a CI job runs as its own job. A row with no provider, or two, is
refused before anything runs. Every row runs, so a receipt names every red at once.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import psutil  # type: ignore[import-untyped]

from devtools.architecture.git_inputs import read_blobs, staged_changes, unique_moves
from devtools.pipeline.definition import PATH, Definition, DefinitionError, Row, Stage, load
from devtools.pipeline.receipt import Writer, _now, verify
from devtools.testing.requirements import edited_families, file_holders, selection


def _changed_test_files(pairs: list[tuple[list[str], str]]) -> list[tuple[list[str], str]]:
    """Select regular changed test files from the exact Git diff."""
    changed = [
        (meta, path)
        for meta, path in pairs
        if path.startswith("tests/")
        and path.endswith(".py")
        and (Path(path).name.startswith("test_") or Path(path).name.endswith("_test.py"))
    ]
    if any(
        meta[0][1:] not in {"000000", "100644", "100755"}
        or meta[1] not in {"000000", "100644", "100755"}
        for meta, _ in changed
    ):
        raise ValueError("changed tests must be regular files")
    return changed


def changed_families(root: Path, base: str, head: str) -> list[str]:
    """Compare the stage's exact Git endpoints in two batched reads, only for changed test files."""

    try:
        if not all(isinstance(x, str) and re.fullmatch(r"[0-9a-f]{40}", x) for x in (base, head)):
            raise ValueError("base and head must be full Git commit IDs")
        pairs = staged_changes(root, tree=head, base=base, timeout=60)
        if any(len(meta) != 5 or not path for meta, path in pairs):
            raise ValueError("malformed Git diff")
        moves = unique_moves(pairs)
        changed = _changed_test_files(pairs)
        ids = sorted({oid for meta, _ in changed for oid in meta[2:4] if set(oid) != {"0"}})
        blobs = read_blobs(ids, root, timeout=60)
        if set(ids) != set(blobs):
            raise ValueError("a changed test blob is missing")
        before, after = [], []
        for meta, path in changed:
            if meta[2] in blobs:
                before.extend(file_holders(path, blobs[meta[2]], size_scope=True))
            if meta[3] in blobs:
                after.extend(file_holders(path, blobs[meta[3]], size_scope=True))
        return edited_families(before, after, moves)
    except (OSError, ValueError, SyntaxError, UnicodeError, subprocess.SubprocessError) as error:
        raise ValueError(
            f"test_size.changed_unmeasured: restore the exact test source and Git history: {error}"
        ) from error


@dataclass(frozen=True)
class Options:
    """The result cache's store and mode, handed to the stage's gate; the gate owns reuse."""

    root: Path | None = None
    mode: str = "off"
    cap: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"off", "record", "shadow", "on"} or (
            self.cap is not None and (type(self.cap) is not int or self.cap <= 0)
        ):
            raise ValueError("result_cache.request_invalid: supply a mode and a positive byte cap")

    @classmethod
    def read(
        cls, root: str | None = None, mode: str | None = None, cap: str | None = None
    ) -> Options:
        location = root if root is not None else os.environ.get("ALPHALATTICE_RESULT_CACHE")
        selected = mode or os.environ.get("ALPHALATTICE_RESULT_CACHE_MODE", "off")
        raw = cap if cap is not None else os.environ.get("ALPHALATTICE_RESULT_CACHE_BYTES")
        try:
            return cls(
                Path(location).resolve() if location else None,
                selected,
                int(raw) if raw else None,
            )
        except (ValueError, TypeError):
            raise ValueError(
                "result_cache.request_invalid: supply a mode and a positive byte cap"
            ) from None

    def environment(self) -> dict[str, str]:
        return {
            "ALPHALATTICE_RESULT_CACHE": str(self.root or ""),
            "ALPHALATTICE_RESULT_CACHE_MODE": self.mode,
            "ALPHALATTICE_RESULT_CACHE_BYTES": str(self.cap) if self.cap is not None else "",
        }


SKIPPED = "skipped"
"""The provider of a row this host does not run: registered for another host."""


@dataclass(frozen=True)
class Context:
    """What an adapter may use: the checkout and its merge base, the run's folder, its workers and
    its deadline."""

    root: Path
    base: str
    out: Path
    host: str
    workers: int
    deadline: float
    definition: Definition
    stage: str = ""
    required_tests: tuple[str, ...] = ()
    cache_options: Options = field(default_factory=Options)
    cache_writer: dict[str, str] = field(default_factory=dict)


Outcome = tuple[dict[str, Any], Path | None]
"""A step's status, exit and seconds, and the node results it wrote, if it ran tests."""
Adapter = Callable[[Row, Context], Outcome]


def plugins(paths: Sequence[Path]) -> dict[str, Mapping[str, Adapter]]:
    """Each ``--plugin`` module's name and the rows its PROVIDES claims, refused unless loadable."""
    found: dict[str, Mapping[str, Adapter]] = {}
    for path in paths:
        try:
            spec = importlib.util.spec_from_file_location(f"pipeline_plugin_{path.stem}", path)
            if spec is None or spec.loader is None:
                raise ImportError("not a Python module")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            provides, name = module.PROVIDES, str(getattr(module, "NAME", path.stem))
            if not isinstance(provides, dict) or not all(map(callable, provides.values())):
                raise TypeError("its PROVIDES is no mapping of rows to adapters")
            if name in found:
                raise ImportError(f"a second plugin is named {name}")
        except Exception as error:  # a plugin's own code may raise anything: it refuses by name
            raise DefinitionError(
                "pipeline.plugin_unloadable",
                f"{path}: {error}; a plugin is a Python file whose PROVIDES maps rows to adapters",
            ) from None
        found[name] = provides
    return found


def resolve(
    definition: Definition, stage: Stage, host: str, provides: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    """Each of the stage's rows and the one provider that runs it on this host."""
    if unknown := [
        (n, r) for n, rows in provides.items() for r in rows if r not in definition.rows
    ]:
        raise DefinitionError(
            "pipeline.row_unregistered",
            f"plugin {unknown[0][0]} provides row {unknown[0][1]}, which {PATH} does not register",
        )
    providers = {}
    for row in (definition.rows[name] for name in stage.rows):
        claims = [name for name, rows in provides.items() if row.id in rows]
        if len(claims) > 1:
            raise DefinitionError(
                "pipeline.row_provided_twice",
                f"row {row.id} is provided by {' and '.join(claims)}; one plugin may provide it",
            )
        product = row.kind == "command" or (row.kind == "pytest" and bool(row.public))
        job = f"ci.yml:{row.job}" if row.kind == "ci-job" else ""
        providers[row.id] = (
            SKIPPED
            if host not in row.hosts
            else (claims[0] if claims else "product" if product else job)
        )
        if not providers[row.id]:
            raise DefinitionError(
                "pipeline.row_unprovided",
                f"row {row.id} ({row.kind}) has no provider on the {host} host; pass the "
                "plugin that provides it with --plugin",
            )
    return providers


def _cost_configuration(root: Path, name: str, host: str) -> dict[str, str]:
    """Bind the locked environment for a workshop commit receipt."""
    cost_configuration: dict[str, str] = {}
    if name == "commit" and host == "workshop":
        lock = root / "uv.lock"
        try:
            if lock.is_symlink() or not lock.is_file():
                raise OSError("the lock is missing or not a regular file")
            cost_configuration = {
                "lane": "merge",
                "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
            }
        except OSError as error:
            raise DefinitionError(
                "pipeline.receipt_cost_unmeasured",
                f"{lock}: restore the readable locked environment: {error}",
            ) from None
    return cost_configuration


def _size_selection(
    name: str, workspace: dict[str, Any], required: tuple[str, ...], root: Path
) -> tuple[tuple[str, ...], dict[str, Any] | None]:
    """Resolve changed families once from this stage's pinned workspace."""
    if name != "commit":
        return required, None
    started = perf_counter()
    try:
        edited = selection(changed_families(root, workspace["base"], workspace["head"]))
    except ValueError as error:
        raise DefinitionError("pipeline.size_selection_unmeasured", str(error)) from error
    record = {
        "base": workspace["base"],
        "head": workspace["head"],
        "families": list(edited),
        "seconds": perf_counter() - started,
    }
    return tuple(dict.fromkeys((*required, *edited))), record


def _write_size_selection(out: Path, record: dict[str, Any] | None) -> None:
    """Retain the exact selection with its measured discovery cost."""
    if record is None:
        return
    try:
        (out / "size-stage-selection.json").write_text(json.dumps(record) + "\n", encoding="utf-8")
    except OSError as error:
        raise DefinitionError(
            "pipeline.size_overhead_unmeasured", f"restore the stage timing output: {error}"
        ) from error


def run(
    root: Path,
    name: str,
    *,
    host: str,
    out: Path,
    base: str,
    workers: int,
    command: Sequence[str],
    paths: Sequence[Path] = (),
    required_tests: Sequence[str] = (),
    cache_options: Options | None = None,
    line: str = "pipeline",
) -> dict[str, Any]:
    """Run one stage into ``out`` and return its receipt's summary; ``out`` holds one run."""
    observed_start = perf_counter()
    started, definition = time.monotonic(), load(root)
    stage, provides = definition.stage(name), plugins(paths)
    providers = resolve(definition, stage, host, provides)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise DefinitionError(
            "pipeline.out_used", f"{out} already holds a run; give each stage run its own --out"
        )
    try:
        required = selection(required_tests)
    except ValueError as error:
        raise DefinitionError("pipeline.stage_invalid", str(error)) from None
    workspace = _workspace(root, base, required)
    selected, size_record = _size_selection(name, workspace, required, root)
    cost_configuration = _cost_configuration(root, name, host)
    if required and name != "commit":
        raise DefinitionError("pipeline.stage_invalid", "required cost checks belong to commit")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    _write_size_selection(out, size_record)
    steps = [{"step": {"row": row}} for row in stage.rows]
    children: list[dict[str, Any]] = [
        {"workspace": {}},
        {"configuration": {}},
        *steps,
        {"metrics": {}},
    ]
    writer = Writer(
        out / "receipt.jsonl", {"stage": name, "host": host, "command": [*command]}, children
    )
    writer.write({"workspace": {}}, workspace)
    configuration: dict[str, Any] = {"definition": PATH, "definition_sha256": definition.sha256}
    configuration |= {"rows": list(stage.rows), "providers": providers, "workers": workers}
    configuration |= cost_configuration
    stop = stage.bound_minutes * 60
    writer.write({"configuration": {}}, configuration | {"bound_seconds": stop})
    deadline = started + stop
    context = Context(
        root,
        workspace["base"],
        out,
        host,
        workers,
        deadline,
        definition,
        name,
        selected,
        cache_options or Options(),
        {"line": line, "head": workspace["head"], "invocation": writer.invocation, "at": _now()},
    )
    row_seconds: dict[str, float] = {}
    complete = False
    try:
        try:  # an interrupted run leaves its receipt without finished, which counts as none
            statuses, red = _rows(writer, stage, providers, provides, context, row_seconds)
            elapsed = time.monotonic() - started
            writer.write({"metrics": {}}, {"wall_seconds": round(elapsed, 3)})
            if elapsed > stop:
                statuses.append("ERROR")
            failed = "FAIL" in statuses or red
            result = "ERROR" if "ERROR" in statuses else "FAIL" if failed else "PASS"
            over = round(elapsed, 3) > stop
            classed = (
                {"red_classes": {}, "red_class": None}
                if result == "PASS"
                else writer.red_summary(over_bound=over)
            )
            writer.finish(
                {"result": result, "exit": {"PASS": 0, "FAIL": 1}.get(result, 2), "red_nodes": red}
                | classed
            )
        finally:
            writer.stream.close()
        summary = verify(
            out / "receipt.jsonl", required_tests=required, within_limits=bool(required)
        )
        complete = True
        return summary
    finally:
        try:
            wall = round(perf_counter() - observed_start, 3)
            reported = sum(row_seconds.values())
            timing = {
                "scope": "stage invocation through receipt verification; sidecar write excluded",
                "complete": complete,
                "wall_seconds": wall,
                "rows": row_seconds,
                "unattributed_seconds": round(max(0.0, wall - reported), 3),
                "overlap_seconds": round(max(0.0, reported - wall), 3),
            }
            (out / "wall-timing.json").write_text(json.dumps(timing) + "\n", encoding="utf-8")
        except Exception as error:
            with contextlib.suppress(Exception):
                print(f"stage timing unavailable: {error}", file=sys.stderr)


def _rows(
    writer: Writer,
    stage: Stage,
    providers: dict[str, str],
    provides: Mapping[str, Mapping[str, Adapter]],
    context: Context,
    row_seconds: dict[str, float],
) -> tuple[list[str], list[str]]:
    """Run every row by its provider and write its step; the statuses and red nodes."""
    statuses: list[str] = []
    red: list[str] = []
    for row in (context.definition.rows[r] for r in stage.rows):
        provider = providers[row.id]
        if provider == SKIPPED or provider.startswith("ci.yml:"):
            step: dict[str, Any] = {"status": "SKIPPED_BY_HOST", "exit": None, "seconds": 0.0}
            events: Path | None = None
        else:
            adapter = _product if provider == "product" else provides[provider][row.id]
            step, events = adapter(row, context)
        step |= {"provider": provider}
        row_seconds[row.id] = step["seconds"]
        statuses.append(step["status"])
        line = f"{row.id:<16} {step['status']:<16} {step['seconds']:>7.1f} s  {provider}"
        print(line, file=sys.stderr, flush=True)
        if events is None:
            writer.write({"step": {"row": row.id}}, step)
        else:
            red += writer.tests(row.id, step, events)
    return statuses, red


def _product(row: Row, context: Context) -> Outcome:
    """The product's adapter: a command row's ``run``, or a pytest row's public selection."""
    log = context.out / "logs" / f"{row.id}.log"
    if row.kind == "command":
        argv = [sys.executable if word == "{python}" else word for word in row.run]
        return _call(argv, log, context, {}), None
    events = context.out / f"{row.id}-events.jsonl"
    events.touch()
    argv = [sys.executable, "-m", "pytest", *row.public, "-n", str(context.workers)]
    required = context.required_tests if row.id == "tests" else ()
    argv += [
        n
        for n in required
        if not any(
            "::" not in p and Path(n.split("::", 1)[0]).is_relative_to(p) for p in row.public
        )
    ]
    argv += ["-m", "not real_evidence", "--basetemp", str(context.out / "basetemp")]
    for plugin in row.plugins:
        argv += ["-p", context.definition.rows[plugin].module]
    source = os.pathsep.join([str(context.root / "src"), os.environ.get("PYTHONPATH", "")])
    env = {
        "ALPHALATTICE_PIPELINE_EVENTS": str(events),
        "PYTHONPATH": source,
        "ALPHALATTICE_PIPELINE_REQUIRED_TESTS": json.dumps(required),
    }
    return _call(argv, log, context, env), events


def _call(argv: list[str], log: Path, context: Context, env: dict[str, str]) -> dict[str, Any]:
    """Run one row's process within the stage's deadline; its output goes to the row's log."""
    began = time.monotonic()
    stopped = None
    with log.open("w", encoding="utf-8") as stream:
        try:
            process = subprocess.Popen(
                argv,
                cwd=context.root,
                env={**os.environ, **env, "ALPHALATTICE_PIPELINE_STAGE": context.stage},
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            code: int | None = process.wait(timeout=max(1.0, context.deadline - began))
            status = "PASS" if code == 0 else "FAIL"
        except subprocess.TimeoutExpired:  # past the stage's bound: the row's whole tree goes
            _kill(process.pid)
            process.wait()
            stream.write(f"stopped at the stage's bound, {context.deadline - began:.0f} s on\n")
            status, code, stopped = "ERROR", None, "bound"
        except OSError as error:
            stream.write(f"{argv[0]}: {error}\n")
            status, code, stopped = "ERROR", None, "launch"
    if status != "PASS":
        tail = log.read_text("utf-8", errors="replace").splitlines()[-40:]
        print("\n".join([f"--- {log.name} ({status}), last lines:", *tail]), file=sys.stderr)
    step = {"status": status, "exit": code, "seconds": round(time.monotonic() - began, 3)}
    return step | ({"stopped": stopped} if stopped else {})


def _kill(pid: int) -> None:
    """A row's process and everything it started, children first."""
    with contextlib.suppress(psutil.NoSuchProcess):
        parent = psutil.Process(pid)
        for process in [*parent.children(recursive=True), parent]:
            with contextlib.suppress(psutil.NoSuchProcess):
                process.kill()


def _workspace(root: Path, base: str, required_tests: Sequence[str] = ()) -> dict[str, Any]:
    """The checkout's head, the merge base with ``base``, and whether its tree is clean."""

    def git(*words: str) -> str:
        try:
            done = subprocess.run(["git", *words], cwd=root, capture_output=True, text=True)
            why = done.stderr.strip() if done.returncode else ""
        except OSError as error:
            why = str(error)
        if why or done.returncode:
            raise DefinitionError(
                "pipeline.workspace_unreadable",
                f"git {' '.join(words)}: {why}; run it in a Git checkout, with --base a commit "
                "or branch it has (HEAD in a shallow clone)",
            )
        return done.stdout.strip()

    head = git("rev-parse", "HEAD")
    if required_tests and git(
        "log", "-1", "--format=%(trailers:key=Change-Kind,valueonly)", head
    ).split() != ["cost"]:
        raise DefinitionError(
            "pipeline.stage_invalid", "required named tests need Change-Kind: cost"
        )
    merged = git("merge-base", git("rev-parse", "--verify", f"{base}^{{commit}}"), head)
    return {
        "repo": "product",
        "base": merged,
        "head": head,
        "clean": not git("status", "--porcelain"),
    }
