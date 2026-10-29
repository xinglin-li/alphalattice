"""Stage receipts, in the form of Bazel's Build Event Protocol: one JSON event per line.

The first event, ``started``, announces its children; ``finished`` ends the stream with
``last_message``. A stream that misses either, or an announced child, is incomplete and counts as no
receipt. Readers (the READY check, the merge queue, the test manager) read it through ``verify``.
Loaded as a pytest plugin, it writes each node's result to the stream ALPHALATTICE_PIPELINE_EVENTS
names, which the stage folds into its receipt.
"""

from __future__ import annotations

import functools
import hashlib
import json
import operator
import os
import re
import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, TextIO

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    ValidationError,
    create_model,
    model_validator,
)
from pydantic_core import PydanticCustomError

from devtools.pipeline.definition import DefinitionError

if TYPE_CHECKING:
    import pytest

SCHEMA = "alphalattice.pipeline-receipt.v1"
EVENTS = "ALPHALATTICE_PIPELINE_EVENTS"
MASKED = re.compile(r"[A-Za-z]:[\\/]\S*|/\S+|\b[0-9a-f]{7,64}\b|\d+(?:\.\d+)?")
"""Paths, hashes and numbers, masked so that one failure clusters under one signature."""
TOOL_VERSION = "1"
COUNTED = {"PASSED": "passed", "FAILED": "failed", "ERROR": "errors", "SKIPPED": "skipped"}


class ReceiptError(DefinitionError):
    """A receipt that does not verify: its code, then what is missing or different."""


class Writer:
    """One stage run's receipt, written event by event; a stage run never overwrites one."""

    def __init__(self, path: Path, started: dict[str, Any], children: Sequence[dict[str, Any]]):
        self.stream = path.open("x", encoding="utf-8", newline="\n")
        started = {
            "schema": SCHEMA,
            "invocation": str(uuid.uuid4()),
            "tool_version": TOOL_VERSION,
            **started,
            "started_at": _now(),
        }
        self.write({"started": {}}, started, [*children, {"finished": {}}])

    def write(
        self, key: dict[str, Any], payload: dict[str, Any], children: Sequence[Any] = ()
    ) -> None:
        line = {
            "id": key,
            **({"children": list(children)} if children else {}),
            next(iter(key)): payload,
        }
        self.stream.write(json.dumps(line) + "\n")
        self.stream.flush()

    def tests(self, row: str, step: dict[str, Any], events: Path) -> None:
        """The tests step, which announces each node's result and size summary, then writes them."""
        results = [json.loads(line) for line in events.read_text("utf-8").splitlines()]
        sizes: dict[str, dict[str, Any]] = {}
        for result in results:
            empty = {"total": 0, **dict.fromkeys(COUNTED.values(), 0), "test_seconds": 0.0}
            summary = sizes.setdefault(result.get("size") or "unsized", empty)
            summary["total"] += 1
            summary[COUNTED[result["status"]]] += 1
            summary["test_seconds"] += result["seconds"]
        keys = [{"test_result": {"node": r["node"], "attempt": r["attempt"]}} for r in results]
        sums = [{"test_summary": {"size": size}} for size in sizes]
        self.write({"step": {"row": row}}, step, [*keys, *sums])
        for key, payload in [
            *zip(keys, results, strict=True),
            *zip(sums, sizes.values(), strict=True),
        ]:
            self.write(key, payload)

    def finish(self, payload: dict[str, Any]) -> None:
        line = {
            "id": {"finished": {}},
            "last_message": True,
            "finished": {**payload, "finished_at": _now()},
        }
        self.stream.write(json.dumps(line) + "\n")
        self.stream.close()


class _Strict(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False)


Text = Annotated[str, Field(min_length=1)]
Sha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Seconds = Annotated[float, Field(ge=0)]


class Started(_Strict):
    schema_: Literal["alphalattice.pipeline-receipt.v1"] = Field(alias="schema")
    invocation: Text
    stage: Text
    host: Literal["workshop", "public"]
    started_at: Text
    command: Annotated[list[Text], Field(min_length=1)]
    tool_version: Text


class Workspace(_Strict):
    repo: Text
    base: Sha
    head: Sha
    clean: bool


class Configuration(_Strict):
    definition: Text
    definition_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    rows: list[Text]
    providers: dict[str, Text]
    workers: Annotated[int, Field(ge=1)]
    bound_seconds: Annotated[float, Field(gt=0)]


class Step(_Strict):
    status: Literal["PASS", "FAIL", "ERROR", "SKIPPED_BY_HOST"]
    exit: int | None
    seconds: Seconds
    provider: Text

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _exit_agrees(self) -> Step:
        legal = {"PASS": self.exit == 0, "SKIPPED_BY_HOST": self.exit is None}
        if not legal.get(self.status, self.exit != 0):
            raise PydanticCustomError("receipt_inconsistent", "a step's status and exit disagree")
        return self


class TestResult(_Strict):
    node: Text
    attempt: Annotated[int, Field(ge=1)]
    solo: bool
    status: Literal["PASSED", "FAILED", "ERROR", "SKIPPED"]
    seconds: Seconds
    size: str | None = None
    owner: str | None = None
    timeout_seconds: float | None = None
    worker: str = ""
    requirements: list[str] = []
    requirement_layer: str | None = None
    requirement_owner: str | None = None
    class_: str | None = Field(None, alias="class")
    phase: str | None = None
    failure: dict[str, str] | None = None


class Finished(_Strict):
    result: Literal["PASS", "FAIL", "ERROR"]
    exit: int
    finished_at: Text
    red_nodes: list[str] = []


class Metrics(_Strict):
    wall_seconds: Seconds | None = None
    queue_wait_seconds: Seconds | None = None
    test_seconds: Seconds | None = None
    card: dict[str, int | float | list[str]] = {}


class _Canonical(_Strict):
    """A singleton's id holds nothing, so a second one repeats the first."""


SINGLE = ("started", "workspace", "configuration", "metrics", "finished")
PAYLOADS: dict[str, Any] = {
    "started": Started,
    "workspace": Workspace,
    "configuration": Configuration,
    "step": Step,
    "test_result": TestResult,
    "test_summary": dict[str, int | float],
    "metrics": Metrics,
    "finished": Finished,
}


def _event(kind: str, payload: Any) -> Any:
    """One kind's event model: its one-key id, the ids it announces, its payload."""
    key = _Canonical if kind in SINGLE else dict[str, str | int]
    fields: dict[str, Any] = {"id": (dict[Literal[kind], key], ...)}  # type: ignore[valid-type]
    fields |= {kind: (payload, ...), "children": (list[dict[str, dict[str, str | int]]], [])}
    if kind == "finished":
        fields["last_message"] = (Literal[True], ...)
    return Annotated[create_model(f"{kind}_event", __base__=_Strict, **fields), Tag(kind)]


def _kind(value: Any) -> str | None:
    key = value.get("id") if isinstance(value, dict) else None
    return next(iter(key)) if isinstance(key, dict) and len(key) == 1 else None


Event = Annotated[  # type: ignore[valid-type]
    functools.reduce(operator.or_, (_event(kind, model) for kind, model in PAYLOADS.items())),
    Discriminator(_kind),
]


class Receipt(_Strict):
    """A stage run's whole receipt; its rules beyond each event's shape are its validator."""

    events: list[Event]  # type: ignore[valid-type]

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _whole(self) -> Receipt:
        """It opens with started, ends with its one finished, and every announced event came."""
        events: list[Any] = self.events
        kinds = [next(iter(event.id)) for event in events]
        ids = [json.dumps(e.model_dump(mode="json")["id"], sort_keys=True) for e in events]
        announced = {json.dumps(child, sort_keys=True) for e in events for child in e.children}
        if kinds[:1] != ["started"]:
            raise PydanticCustomError("receipt_unknown", "the stream does not open with started")
        if kinds.count("finished") != 1 or kinds[-1] != "finished" or announced - set(ids):
            raise PydanticCustomError("receipt_incomplete", "an announced or last event is missing")
        if len(set(ids)) != len(ids) or not {"workspace", "configuration"} <= set(kinds):
            raise PydanticCustomError("receipt_malformed", "an event repeats, or one is missing")
        return self

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _consistent(self) -> Receipt:
        """Its result stands over every exit, step and node's latest attempt; a rerun is solo."""
        events: list[Any] = self.events
        latest: dict[str, TestResult] = {}
        for event in (e for e in events if "test_result" in e.id):
            result = event.test_result
            if event.id["test_result"] != {"node": result.node, "attempt": result.attempt} or (
                (result.attempt > 1) != result.solo
            ):
                raise PydanticCustomError("receipt_malformed", "a result's id and attempt disagree")
            if result.attempt >= latest.get(result.node, result).attempt:
                latest[result.node] = result
        red = [r.node for r in latest.values() if r.status in ("FAILED", "ERROR")]
        red += [
            e.step.provider for e in events if "step" in e.id and e.step.status in ("FAIL", "ERROR")
        ]
        finished = events[-1].finished
        if (finished.result == "PASS") != (finished.exit == 0) or (
            finished.result == "PASS" and red
        ):
            raise PydanticCustomError("receipt_inconsistent", "its result denies its parts")
        return self


def verify(
    path: Path, *, stage: str = "", host: str = "", head: str = "", passing: bool = False
) -> dict[str, Any]:
    """The receipt's summary, refused unless it is whole, consistent and what the reader asks."""
    try:
        data = path.read_bytes()
        lines = [json.loads(line) for line in data.decode("utf-8").splitlines()]
        events: list[Any] = Receipt.model_validate({"events": lines}).events
    except OSError as error:
        raise ReceiptError("pipeline.receipt_missing", f"{path}: {error.strerror}") from None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReceiptError("pipeline.receipt_unreadable", f"{path}: {error}") from None
    except ValidationError as error:
        first = error.errors()[0]
        code = first["type"] if first["type"].startswith("receipt_") else "receipt_malformed"
        where = ".".join(map(str, first["loc"]))
        raise ReceiptError("pipeline." + code, f"{path}: {where}: {first['msg']}") from None
    found = {next(iter(event.id)): getattr(event, next(iter(event.id))) for event in events}
    started, workspace, finished = found["started"], found["workspace"], found["finished"]
    summary = {"schema": SCHEMA, "stage": started.stage, "host": started.host}
    summary |= {"repo": workspace.repo, "base": workspace.base, "head": workspace.head}
    summary |= {"result": finished.result}
    summary |= {"finished_at": finished.finished_at, "red_nodes": finished.red_nodes}
    summary["card"] = found["metrics"].card if "metrics" in found else {}
    summary["sha256"] = hashlib.sha256(data).hexdigest()
    asked = {"stage": stage, "host": host, "head": head, "result": "PASS" * passing}
    if wrong := [key for key, value in asked.items() if value and summary[key] != value]:
        raise ReceiptError(
            "pipeline.receipt_mismatch",
            f"{path}: its {wrong[0]} is {summary[wrong[0]]!r}, not {asked[wrong[0]]!r}",
        )
    return summary


class _Events:
    """The plugin's half on pytest's controller: one result per node, after its last phase."""

    def __init__(self, stream: TextIO) -> None:
        self.stream, self.open = stream, dict[str, dict[str, Any]]()

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        empty = {
            "node": report.nodeid,
            "attempt": 1,
            "solo": False,
            "status": "PASSED",
            "seconds": 0.0,
        }
        node = self.open.setdefault(report.nodeid, empty)
        node["seconds"] += report.duration
        if report.failed and node["status"] == "PASSED":
            status = "FAILED" if report.when == "call" else "ERROR"
            node.update(status=status, phase=report.when, failure=_failure(report))
        elif report.skipped and node["status"] == "PASSED":
            node["status"] = "SKIPPED"
        if report.when not in ("setup", "call"):  # its teardown, or a crashed worker's "???"
            found = dict(report.user_properties)
            node.update(size=found.get("test_size"), owner=found.get("test_size_owner"))
            node.update({key: found.get(key) for key in ("requirement_layer", "requirement_owner")})
            node.update(requirements=found.get("requirements", []), **{"class": None})
            node.update(timeout_seconds=found.get("test_timeout_seconds"), worker=_worker(report))
            self.stream.write(json.dumps(self.open.pop(report.nodeid)) + "\n")

    def close(self) -> None:
        """A node that never reached its last phase is an error, never a silent absence."""
        for node in self.open.values():
            failure = {
                "type": "Incomplete",
                "message": "no last phase was reported",
                "signature": "",
            }
            node.update(status="ERROR", phase="incomplete", failure=failure, **{"class": None})
            self.stream.write(json.dumps(node) + "\n")
        self.stream.close()


def register(config: pytest.Config, path: Path) -> None:
    """Register the result writer on this session, appending to ``path``."""
    config.pluginmanager.register(_Events(path.open("a", encoding="utf-8")), "pipeline-events")


def pytest_configure(config: pytest.Config) -> None:
    if os.environ.get(EVENTS) and not hasattr(config, "workerinput"):  # workers report to it
        register(config, Path(os.environ[EVENTS]))


def pytest_unconfigure(config: pytest.Config) -> None:
    """The disposer: the registration goes and its stream closes."""
    if plugin := config.pluginmanager.get_plugin("pipeline-events"):
        config.pluginmanager.unregister(plugin)
        plugin.close()


def _failure(report: pytest.TestReport) -> dict[str, str]:
    crash = getattr(report.longrepr, "reprcrash", None)
    message = ((crash.message if crash else str(report.longrepr)).strip().splitlines() or [""])[0][
        :300
    ]
    kind = message.split(":", 1)[0] if re.match(r"^[\w.]+:", message) else "Failure"
    signature = hashlib.sha256(f"{kind}\n{MASKED.sub('#', message)}".encode()).hexdigest()
    return {"type": kind, "message": message, "signature": signature}


def _worker(report: pytest.TestReport) -> str:
    return str(getattr(getattr(getattr(report, "node", None), "gateway", None), "id", ""))


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
