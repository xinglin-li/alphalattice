"""A stage run: each row resolved to one provider on its host, run in order, into one receipt.

The product provides what every host shares: a command row runs its ``run``, and a pytest row its
public selection with its plugin rows as ``-p``. A host adds its own adapters with ``--plugin``, a
module whose ``PROVIDES`` maps row ids to adapters, in the form of pytest's ``-p``. This host skips
a row registered for another, and a CI job runs as its own job. A row with no provider, or two, is
refused before anything runs. Every row runs, so a receipt names every red at once.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil  # type: ignore[import-untyped]

from devtools.pipeline.definition import PATH, Definition, DefinitionError, Row, Stage, load
from devtools.pipeline.receipt import Writer, verify

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
) -> dict[str, Any]:
    """Run one stage into ``out`` and return its receipt's summary; ``out`` holds one run."""
    started, definition = time.monotonic(), load(root)
    stage, provides = definition.stage(name), plugins(paths)
    providers = resolve(definition, stage, host, provides)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise DefinitionError(
            "pipeline.out_used", f"{out} already holds a run; give each stage run its own --out"
        )
    workspace = _workspace(root, base)
    (out / "logs").mkdir(parents=True, exist_ok=True)
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
    writer.write({"configuration": {}}, configuration | {"bound_seconds": stage.bound_minutes * 60})
    deadline = started + stage.bound_minutes * 60
    context = Context(root, workspace["base"], out, host, workers, deadline, definition)
    try:  # an interrupted run leaves its receipt without finished, which counts as none
        statuses, red = _rows(writer, stage, providers, provides, context)
        writer.write({"metrics": {}}, {"wall_seconds": round(time.monotonic() - started, 3)})
        failed = "FAIL" in statuses or red
        result = "ERROR" if "ERROR" in statuses else "FAIL" if failed else "PASS"
        writer.finish(
            {"result": result, "exit": {"PASS": 0, "FAIL": 1}.get(result, 2), "red_nodes": red}
        )
    finally:
        writer.stream.close()
    return verify(out / "receipt.jsonl")


def _rows(
    writer: Writer,
    stage: Stage,
    providers: dict[str, str],
    provides: Mapping[str, Mapping[str, Adapter]],
    context: Context,
) -> tuple[list[str], list[str]]:
    """Run every row by its provider and write its step; the statuses and the red nodes."""
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
    argv += ["-m", "not real_evidence", "--basetemp", str(context.out / "basetemp")]
    for plugin in row.plugins:
        argv += ["-p", context.definition.rows[plugin].module]
    source = os.pathsep.join([str(context.root / "src"), os.environ.get("PYTHONPATH", "")])
    env = {"ALPHALATTICE_PIPELINE_EVENTS": str(events), "PYTHONPATH": source}
    return _call(argv, log, context, env), events


def _call(argv: list[str], log: Path, context: Context, env: dict[str, str]) -> dict[str, Any]:
    """Run one row's process within the stage's deadline; its output goes to the row's log."""
    began = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        try:
            process = subprocess.Popen(
                argv,
                cwd=context.root,
                env={**os.environ, **env},
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
            status, code = "ERROR", None
        except OSError as error:
            stream.write(f"{argv[0]}: {error}\n")
            status, code = "ERROR", None
    if status != "PASS":
        tail = log.read_text("utf-8", errors="replace").splitlines()[-40:]
        print("\n".join([f"--- {log.name} ({status}), last lines:", *tail]), file=sys.stderr)
    return {"status": status, "exit": code, "seconds": round(time.monotonic() - began, 3)}


def _kill(pid: int) -> None:
    """A row's process and everything it started, children first."""
    with contextlib.suppress(psutil.NoSuchProcess):
        parent = psutil.Process(pid)
        for process in [*parent.children(recursive=True), parent]:
            with contextlib.suppress(psutil.NoSuchProcess):
                process.kill()


def _workspace(root: Path, base: str) -> dict[str, Any]:
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
    merged = git("merge-base", git("rev-parse", "--verify", f"{base}^{{commit}}"), head)
    return {
        "repo": "product",
        "base": merged,
        "head": head,
        "clean": not git("status", "--porcelain"),
    }
