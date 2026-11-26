"""Stage receipts, in the form of Bazel's Build Event Protocol: one JSON event per line.

The first event, ``started``, announces its children; ``finished`` ends the stream with
``last_message``. A stream that misses either, or an announced child, is incomplete and counts as no
receipt. Readers (the check before review, the merge queue, the test manager) read it through
``verify``. Loaded as a pytest plugin, it writes each node's result to the stream
ALPHALATTICE_PIPELINE_EVENTS names, which the stage folds into its receipt.
"""

from __future__ import annotations

import functools
import hashlib
import json
import operator
import os
import re
import subprocess
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from math import isfinite
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
from devtools.testing.requirements import node_matches, selection

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


def read_test_events(path: Path) -> list[dict[str, Any]]:
    """Read the event stream once through the receipt owner, refusing unreadable rows by name."""
    try:
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("an event is not an object")
    except (OSError, UnicodeError, ValueError) as error:
        raise ReceiptError(
            "pipeline.receipt_unreadable", f"{path}: restore the test event stream: {error}"
        ) from None
    return rows


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
        self.invocation = started["invocation"]
        self.steps: list[tuple[str, dict[str, Any]]] = []
        self.classes: dict[str, str] = {}
        self.tested = False
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
        if "step" in key:
            self.steps.append((key["step"]["row"], payload))

    def tests(self, row: str, step: dict[str, Any], events: Path) -> list[str]:
        """The tests step, which announces each node's result and size summary, then writes them;
        each red carries its class; it returns the nodes ``classed`` holds red."""
        raw = read_test_events(events)
        decisions = [r["cache"] for r in raw if "cache" in r]
        results = [r for r in raw if "cache" not in r]
        red, classes = classed(results)[0], red_classes(results)
        for result in results:
            if result["status"] in RED:
                result["class"] = classes[result["node"]]
        self.classes |= classes
        self.tested = self.tested or bool(results)
        sizes: dict[str, dict[str, Any]] = {}
        for result in results:
            empty = {"total": 0, **dict.fromkeys(COUNTED.values(), 0), "test_seconds": 0.0}
            summary = sizes.setdefault(result.get("size") or "unsized", empty)
            summary["total"] += 1
            summary[COUNTED[result["status"]]] += 1
            summary["test_seconds"] += result["seconds"]
            summary["cached"] = summary.get("cached", 0) + bool(result.get("cached_from"))
        keys = [{"test_result": {"node": r["node"], "attempt": r["attempt"]}} for r in results]
        sums = [{"test_summary": {"size": size}} for size in sizes]
        caches = [{"cache": {"group": r["group"]}} for r in decisions]
        self.write({"step": {"row": row}}, step, [*keys, *sums, *caches])
        for key, payload in [
            *zip(keys, results, strict=True),
            *zip(sums, sizes.values(), strict=True),
            *zip(caches, decisions, strict=True),
        ]:
            self.write(key, payload)
        return red

    def red_summary(self, over_bound: bool) -> dict[str, Any]:
        """A non-passing stage's causes by class and its one class, from what it wrote."""
        counts, label = red_class(
            self.steps, self.classes, over_bound=over_bound, tested=self.tested
        )
        return {"red_classes": counts, "red_class": label}

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


def pending_workspace(path: Path) -> Workspace:
    """Read only a running commit stage's flushed identity; this is not a completed receipt."""
    try:
        with path.open(encoding="utf-8") as stream:
            first, second = json.loads(stream.readline()), json.loads(stream.readline())
        if (
            not isinstance(first, dict)
            or first.get("id") != {"started": {}}
            or not isinstance(second, dict)
            or second.get("id") != {"workspace": {}}
        ):
            raise ValueError("missing ordered started/workspace events")
        started = Started.model_validate(first["started"])
        workspace = Workspace.model_validate(second["workspace"])
        if started.stage != "commit" or workspace.repo != "product":
            raise ValueError("not a product commit stage")
        return workspace  # type: ignore[no-any-return]  # Pydantic validates Workspace above.
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, ValidationError) as error:
        raise ReceiptError(
            "pipeline.size_workspace_unmeasured",
            f"{path}: restore the running stage identity: {error}",
        ) from error


class Configuration(_Strict):
    definition: Text
    definition_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    rows: list[Text]
    providers: dict[str, Text]
    workers: Annotated[int, Field(ge=1)]
    bound_seconds: Annotated[float, Field(gt=0)]
    lock_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    lane: Literal["merge", "candidate", "nightly"] | None = None


class Step(_Strict):
    status: Literal["PASS", "FAIL", "ERROR", "SKIPPED_BY_HOST"]
    exit: int | None
    seconds: Seconds
    provider: Text
    stopped: Literal["bound", "launch"] | None = (
        None  # why an exit-less row ended: its runner knows
    )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _exit_agrees(self) -> Step:
        legal = {"PASS": self.exit == 0, "SKIPPED_BY_HOST": self.exit is None}
        if not legal.get(self.status, self.exit != 0):
            raise PydanticCustomError("receipt_inconsistent", "a step's status and exit disagree")
        return self


RED = ("FAILED", "ERROR")


def classed(
    results: Sequence[dict[str, Any]], *, legacy: bool = False
) -> tuple[list[str], dict[str, str]]:
    """The nodes red by their highest attempt, and the class of each node whose red passed alone
    (the protocol's rule 7): within its time a flake, which PROTOCOL GUARDS quarantines, and over
    it a slow test, a test defect that stays red."""
    latest: dict[str, dict[str, Any]] = {}
    for result in results:  # by attempt: workers write in the order they finish
        if result["attempt"] >= latest.get(result["node"], result)["attempt"]:
            latest[result["node"]] = result
    observed_red = {r["node"] for r in results if r["status"] in RED}
    classes = {}
    for node in observed_red:
        last = latest[node]
        bound = last.get("timeout_seconds")
        if legacy and last["status"] not in RED:
            classes[node] = "slow" if last["seconds"] > (bound or float("inf")) else "flake"
        elif not legacy and last["status"] == "PASSED" and last.get("solo") and _valid_bound(bound):
            classes[node] = "slow" if last["seconds"] > bound else "flake"
    red = [n for n in latest if n in observed_red and classes.get(n) != "flake"]
    return red, classes


MASS = 20
"""This many red nodes sharing one failure signature in a stage: the environment, not the tests."""
CRASH = re.compile(r"worker 'gw[0-9]+' crashed while running")
"""A node killed with its worker: the timeout's kill, or the runner's."""
CASCADE = "incomplete session workspace:"
"""A node failed by the workspace a crashed worker left half-built: a cause only beside a crash."""
DEFECT_ROWS = frozenset({"fast-guard", "public-scan", "card-budget"})
"""Rows whose exit 1 is the change's own lint, type, guard, content or budget refusal."""
TOOLING_EXITS = {2: "interrupted", 3: "internal error", 4: "usage error", 5: "no tests collected"}
"""pytest's exits that are its runner's, never a test's verdict."""


def red_classes(
    results: Sequence[dict[str, Any]], *, admission_defect: bool = False
) -> dict[str, str]:
    """Each red node's class from its own attempts, never a guess (the protocol's rule 7).

    A shared setup ERROR is tooling only with a known infrastructure signature. Only a solo PASS
    within a positive recorded bound is a flake, or load when the parallel
    attempt's worker was killed; over the bound it is slow. A solo FAILED is a defect, and a solo
    killed again is tooling. Anything else (a red never rerun, a skipped solo, a solo with no
    recorded bound, a solo setup error) is unclassified, never a defect."""
    runs: dict[str, list[dict[str, Any]]] = {}
    for result in sorted(results, key=lambda r: r["attempt"]):
        runs.setdefault(result["node"], []).append(result)
    red = {node: tries for node, tries in runs.items() if any(r["status"] in RED for r in tries)}
    shared = Counter(_signature(t[0]) for t in red.values() if _setup_error(t[0]))
    crashed = any(_killed(r) for tries in red.values() for r in tries)
    return {
        node: _node_class(tries, shared, crashed, admission_defect=admission_defect)
        for node, tries in red.items()
    }


def _signature(result: dict[str, Any]) -> str:
    return json.dumps(result.get("failure"), sort_keys=True)


def _message(result: dict[str, Any]) -> str:
    return str((result.get("failure") or {}).get("message", ""))


def _killed(result: dict[str, Any]) -> bool:
    return bool(CRASH.search(_message(result)))


def _setup_error(result: dict[str, Any]) -> bool:
    return result["status"] == "ERROR" and result.get("phase") == "setup"


def _valid_bound(value: Any) -> bool:
    return type(value) in (int, float) and isfinite(value) and value > 0


def _node_class(
    tries: list[dict[str, Any]], shared: Counter[str], crashed: bool, *, admission_defect: bool
) -> str:
    """One red node's class from its first and last attempts (see ``red_classes``)."""
    first, last = tries[0], tries[-1]
    if (
        admission_defect
        and first["status"] == "ERROR"
        and first.get("phase") == "collection"
        and (first.get("failure") or {}).get("type") == "test_size.large_contract"
    ):
        return "defect"
    solo = len(tries) > 1 and last.get("solo")
    if solo and last["status"] == "FAILED":
        return "defect"
    if not solo or last["status"] != "PASSED":
        infrastructure = (
            _setup_error(first)
            and shared[_signature(first)] >= MASS
            and bool(
                re.search(r"\bno venv\b|\bno module named ['\"]?pytest\b", _message(first), re.I)
            )
        )
        return "tooling" if (solo and _killed(last)) or infrastructure else "unclassified"
    bound = last.get("timeout_seconds")
    if not _valid_bound(bound):
        return "unclassified"
    if last["seconds"] > bound:
        return "slow"
    cut = _killed(first) or (crashed and CASCADE in _message(first))
    return "load" if cut else "flake"


def red_class(
    steps: Sequence[tuple[str, dict[str, Any]]],
    classes: Mapping[str, str],
    *,
    over_bound: bool,
    tested: bool,
) -> tuple[dict[str, int], str]:
    """A non-passing stage's causes by class, and its one class.

    The class is a defect if any cause is one, else every class present joined, unclassified
    included, so an unknown part is never hidden behind a known one. A row stopped at the bound is
    load, one that could not launch is tooling, and an exit-less row after a failed one is its
    cascade, not a cause. A guard row's exit 1 is the change's own refusal; any other exit is
    unclassified."""
    causes = Counter(classes.values())
    causes["load"] += over_bound
    failed = over_bound
    for row, step in steps:
        status, code = step["status"], step["exit"]
        if status in ("PASS", "SKIPPED_BY_HOST"):
            continue
        if code is None:
            stopped = step.get("stopped")
            if stopped:
                causes["load" if stopped == "bound" else "tooling"] += 1
            elif not failed:
                causes["unclassified"] += 1
        elif row == "tests" and (code in TOOLING_EXITS or not tested):
            causes["tooling"] += 1
        elif row == "tests" and not classes:
            causes["unclassified"] += 1
        elif row == "environment":
            causes["tooling"] += 1
        elif row in DEFECT_ROWS:
            causes["defect" if code == 1 else "unclassified"] += 1
        elif row not in (
            "tests",
            "load-timeouts",
        ):  # those two rows' reds are their nodes, classed above
            causes["unclassified"] += 1
        failed = True
    counts = {name: count for name, count in sorted(causes.items()) if count}
    return counts, "defect" if "defect" in counts else "+".join(counts) or "unclassified"


TOOLING_REDS = frozenset({"flake", "tooling", "load"})
"""The causes 0.2.5's third number counts: a stage red from flakes, tooling or load, no defect."""


class CacheOrigin(_Strict):
    key: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    head: Sha


class CacheDecision(_Strict):
    group: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    mode: Literal["off", "record", "shadow", "on"]
    decision: Literal["HIT", "MISS", "UNCACHEABLE"]
    entry: CacheOrigin | None
    reason: Text | None
    executed: bool
    expected: dict[str, Literal["PASSED", "SKIPPED"]]


class CostEvidence(_Strict):
    """A cost observation and its original identity; inherited is never a fresh measurement."""

    measured: Seconds
    cap: Seconds
    source_head: Sha
    source_node: Text
    lane: Text
    lock_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    provenance: Literal["fresh", "inherited"]
    source_receipt_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    clean: Literal[True]
    other_holders_before: list[Any]
    other_holders_after: list[Any]

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _cost_shape(self) -> CostEvidence:
        if (
            self.measured > self.cap
            or self.other_holders_before
            or self.other_holders_after
            or (self.provenance == "inherited") != bool(self.source_receipt_sha256)
        ):
            raise PydanticCustomError(
                "receipt_malformed", "cost needs a finite cap and an uncontended observation"
            )
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
    cached_from: CacheOrigin | None = None
    cost_evidence: CostEvidence | None = Field(default=None, exclude_if=lambda value: value is None)
    phase: str | None = None
    failure: dict[str, str] | None = None


class Finished(_Strict):
    result: Literal["PASS", "FAIL", "ERROR"]
    exit: int
    finished_at: Text
    red_nodes: list[str] = []
    # a non-passing stage's causes by class and its one class; older receipts omit it
    red_classes: dict[
        Literal["defect", "flake", "slow", "load", "tooling", "unclassified"],
        Annotated[int, Field(ge=1)],
    ] = {}
    red_class: str | None = None


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
    "cache": CacheDecision,
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
        for event in (e for e in events if "test_result" in e.id):
            result = event.test_result
            if event.id["test_result"] != {"node": result.node, "attempt": result.attempt} or (
                (result.attempt > 1) != result.solo
            ):
                raise PydanticCustomError("receipt_malformed", "a result's id and attempt disagree")
        results = [e.test_result for e in events if "test_result" in e.id]
        _cost_consistent(events, results)
        caches = [e.cache for e in events if "cache" in e.id]
        _cache_consistent(caches, results)
        finished = events[-1].finished
        legacy = not ({"red_class", "red_classes"} & finished.model_fields_set)
        red = classed(
            [e.test_result.model_dump() for e in events if "test_result" in e.id], legacy=legacy
        )[0]
        red += [
            e.step.provider for e in events if "step" in e.id and e.step.status in ("FAIL", "ERROR")
        ]
        if (finished.result == "PASS") != (finished.exit == 0) or (
            finished.result == "PASS" and red
        ):
            raise PydanticCustomError("receipt_inconsistent", "its result denies its parts")
        _red_consistent(events, results, finished)
        return self


def _red_consistent(
    events: Sequence[Any], results: Sequence[TestResult], finished: Finished
) -> None:
    """A classed receipt's red classes are what its own steps, nodes and bound make them; a passing
    receipt carries none, and one written before the classing carries neither field."""
    if not finished.red_classes and finished.red_class is None:
        return
    config = next(e.configuration for e in events if "configuration" in e.id)
    wall = next((e.metrics.wall_seconds for e in events if "metrics" in e.id), None)
    steps = [(e.id["step"]["row"], e.step.model_dump()) for e in events if "step" in e.id]
    dumped = [r.model_dump() for r in results]
    over = wall is not None and wall > config.bound_seconds
    # Writers keep the old class until this reader is deployed on develop. Accept both
    # independently recomputed versions, so already-sealed unclassified receipts remain valid.
    found = [
        red_class(
            steps,
            red_classes(dumped, admission_defect=version),
            over_bound=over,
            tested=bool(dumped),
        )
        for version in (False, True)
    ]
    if finished.result == "PASS" or (dict(finished.red_classes), finished.red_class) not in found:
        raise PydanticCustomError(
            "receipt_inconsistent", "its red classes deny its own steps and nodes"
        )


def _fresh_cost_matches(
    result: TestResult, workspace: Workspace, configuration: Configuration
) -> bool:
    cost = result.cost_evidence
    assert cost is not None
    return bool(
        (result.solo or configuration.workers == 1)
        and result.cached_from is None
        and workspace.clean
        and cost.source_head == workspace.head
        and cost.source_node == result.node
        and cost.measured == result.seconds
        and cost.lock_sha256 == configuration.lock_sha256
        and cost.lane == configuration.lane
    )


def _cost_consistent(events: Sequence[Any], results: Sequence[TestResult]) -> None:
    workspace = next(e.workspace for e in events if "workspace" in e.id)
    configuration = next(e.configuration for e in events if "configuration" in e.id)
    for result in results:
        cost = result.cost_evidence
        if cost is None:
            continue
        if result.status != "PASSED" or (
            cost.provenance == "fresh" and not _fresh_cost_matches(result, workspace, configuration)
        ):
            raise PydanticCustomError(
                "receipt_malformed", "cost source does not match a passing clean solo result"
            )


def _cache_consistent(caches: Sequence[CacheDecision], results: Sequence[TestResult]) -> None:
    for decision in caches:
        hit = decision.decision == "HIT"
        if hit != bool(decision.entry and decision.expected) or decision.executed != (
            not hit or decision.mode != "on"
        ):
            raise PydanticCustomError(
                "receipt_inconsistent", "cache decision and execution disagree"
            )
        if hit and decision.mode == "shadow":
            actual = {r.node: r.status for r in results if r.node in decision.expected}
            if actual != decision.expected and decision.reason != "result_cache.not_passing":
                raise PydanticCustomError(
                    "receipt_inconsistent", "a shadow hit disagrees with execution"
                )

    for result in results:
        if result.cached_from and not any(
            c.decision == "HIT"
            and c.mode == "on"
            and c.entry == result.cached_from
            and c.expected.get(result.node) == result.status
            for c in caches
        ):
            raise PydanticCustomError("receipt_inconsistent", "a cached result has no matching hit")


# 4478e213 was authored 2026-10-25 04:44:42 -04 and merged at 07:21:13 -04.
# Retire this compatibility after 2026-11-01 07:21:13 America/New_York (seven local days).
_LEGACY_TARGET_WRITTEN_BEFORE = datetime.fromisoformat("2026-10-25T04:44:42-04:00")
_LEGACY_TARGET_RETIRES = datetime.fromisoformat("2026-11-01T07:21:13-05:00")


def _legacy_target_allowed(lines: list[Any], now: datetime) -> bool:
    try:
        start = datetime.fromisoformat(lines[0]["started"]["started_at"])
        finish = datetime.fromisoformat(lines[-1]["finished"]["finished_at"])
        return bool(
            now.utcoffset() is not None
            and start.utcoffset() is not None
            and finish.utcoffset() is not None
            and now < _LEGACY_TARGET_RETIRES
            and start <= finish < _LEGACY_TARGET_WRITTEN_BEFORE
            and finish <= now
        )
    except (IndexError, KeyError, TypeError, ValueError, AttributeError):
        return False


def _legacy_target_metrics(lines: list[Any], now: datetime) -> None:
    """Discard only the two correctly typed, time-bounded historical metric fields."""
    if not _legacy_target_allowed(lines, now):
        return
    for event in lines:
        payload = event.get("metrics") if isinstance(event, dict) else None
        if not isinstance(payload, dict):
            continue
        value = payload.get("target_seconds")
        flag = payload.get("target_breached")
        try:
            if (
                "target_seconds" in payload
                and value is not None
                and (type(value) not in (int, float) or not isfinite(value) or value < 0)
            ):
                continue
        except OverflowError:
            continue
        if "target_breached" in payload and type(flag) is not bool:
            continue
        event["metrics"] = {
            key: val
            for key, val in payload.items()
            if key not in {"target_seconds", "target_breached"}
        }


def _legacy_now() -> datetime:
    return datetime.now(UTC)


def read(path: Path) -> tuple[bytes, list[Any]]:
    """The receipt's bytes and its validated events, read once, refused unless whole."""
    try:
        data = path.read_bytes()
        lines = [json.loads(line) for line in data.decode("utf-8").splitlines()]
        _legacy_target_metrics(lines, _legacy_now())
        return data, Receipt.model_validate({"events": lines}).events
    except OSError as error:
        raise ReceiptError("pipeline.receipt_missing", f"{path}: {error.strerror}") from None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReceiptError("pipeline.receipt_unreadable", f"{path}: {error}") from None
    except ValidationError as error:
        first = error.errors()[0]
        code = first["type"] if first["type"].startswith("receipt_") else "receipt_malformed"
        where = ".".join(map(str, first["loc"]))
        raise ReceiptError("pipeline." + code, f"{path}: {where}: {first['msg']}") from None


def verify(
    path: Path,
    *,
    stage: str = "",
    host: str = "",
    head: str = "",
    passing: bool = False,
    required_tests: Sequence[str] = (),
    within_limits: bool = False,
    cost_standing: bool = False,
    cost_if_present: bool = False,
    cost_nodes: Sequence[str] = (),
    cost_sources: Mapping[str, Path] | None = None,
    cost_root: Path | None = None,
) -> dict[str, Any]:
    """The receipt's summary, refused unless it is whole, consistent and what the reader asks."""
    data, events = read(path)
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
    _verify_tests(events, required_tests, within_limits)
    if type(cost_if_present) is not bool:
        raise ReceiptError("pipeline.receipt_cost_unmeasured", "cost-if-present must be a boolean")
    present = any(e.test_result.cost_evidence is not None for e in events if "test_result" in e.id)
    if _cost_requested(cost_standing, cost_nodes) or (cost_if_present and present):
        summary["cost_standing"] = _cost_standing(events, cost_nodes, cost_sources, cost_root)
    return summary


def _cost_requested(flag: bool, nodes: Sequence[str]) -> bool:
    if (
        type(flag) is not bool
        or not isinstance(nodes, (tuple, list))
        or any(not isinstance(n, str) or "::" not in n for n in nodes)
        or len(set(nodes)) != len(nodes)
    ):
        raise ReceiptError("pipeline.receipt_cost_unmeasured", "name distinct exact cost nodes")
    return bool(flag or nodes)


def _read_cost_source(
    node: str, digest: str, supplied: Mapping[str, Path] | None
) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(supplied, Mapping) or digest not in supplied:
        raise ReceiptError(
            "pipeline.receipt_cost_unmeasured", f"{node}: restore the source receipt {digest}"
        )
    try:
        data, events = read(Path(supplied[digest]))
    except (ReceiptError, TypeError, ValueError) as error:
        raise ReceiptError(
            "pipeline.receipt_cost_unmeasured", f"{node}: restore its source receipt: {error}"
        ) from None
    if hashlib.sha256(data).hexdigest() != digest:
        raise ReceiptError(
            "pipeline.receipt_cost_unmeasured", f"{node}: source receipt digest differs"
        )
    workspace = next(e.workspace for e in events if "workspace" in e.id)
    if events[-1].finished.result != "PASS":
        raise ReceiptError(
            "pipeline.receipt_cost_unmeasured", f"{node}: source receipt did not pass"
        )
    rows = [e.test_result for e in events if "test_result" in e.id]
    return (
        workspace.head,
        workspace.repo,
        {r.node: r for r in sorted(rows, key=lambda r: r.attempt)},
    )


def _cost_source_matches(
    cost: dict[str, Any], head: str, repo: str, results: dict[str, Any]
) -> bool:
    original = results.get(cost["source_node"])
    evidence = original.cost_evidence if original else None
    return bool(
        repo == "product"
        and head == cost["source_head"]
        and evidence is not None
        and evidence.provenance == "fresh"
        and evidence.source_head == cost["source_head"]
        and evidence.source_node == cost["source_node"]
        and evidence.lane == cost["lane"]
        and evidence.lock_sha256 == cost["lock_sha256"]
        and evidence.measured == cost["measured"]
        and cost["cap"] <= evidence.cap
    )


def _cost_standing(
    events: Sequence[Any],
    nodes: Sequence[str],
    supplied: Mapping[str, Path] | None,
    root: Path | None,
) -> dict[str, Any]:
    if events[-1].finished.result != "PASS":
        raise ReceiptError(
            "pipeline.receipt_cost_unmeasured", "cost standing needs a passing receipt"
        )
    results = [e.test_result for e in events if "test_result" in e.id]
    latest = {r.node: r for r in sorted(results, key=lambda r: r.attempt)}
    standing = {
        node: result.cost_evidence.model_dump(mode="json")
        for node, result in latest.items()
        if result.cost_evidence is not None and (not nodes or node in nodes)
    }
    if not standing:
        raise ReceiptError(
            "pipeline.receipt_cost_unmeasured", "receipt has no measured cost standing"
        )
    if missing := set(nodes) - standing.keys():
        raise ReceiptError(
            "pipeline.receipt_cost_unmeasured",
            f"missing cost evidence for {sorted(missing)!r}; restore its clean source",
        )
    sources: dict[str, tuple[str, str, dict[str, Any]]] = {}
    identities: dict[tuple[str, str], Any] = {}
    workspace = next(e.workspace for e in events if "workspace" in e.id)
    configuration = next(e.configuration for e in events if "configuration" in e.id)
    for node, cost in standing.items():
        if cost["provenance"] == "inherited":
            _verify_inherited_cost(
                node, cost, workspace, configuration, root, supplied, sources, identities
            )
    return standing


def _verify_inherited_cost(
    node: str,
    cost: dict[str, Any],
    workspace: Workspace,
    configuration: Configuration,
    root: Path | None,
    supplied: Mapping[str, Path] | None,
    sources: dict[str, tuple[str, str, dict[str, Any]]],
    identities: dict[tuple[str, str], Any],
) -> None:
    """Bind a source receipt to this target's exact node, Git history, lane and lock."""
    digest = cost["source_receipt_sha256"]
    if digest not in sources:
        sources[digest] = _read_cost_source(node, digest, supplied)
    if (
        not _cost_source_matches(cost, *sources[digest])
        or not workspace.clean
        or workspace.repo != "product"
        or configuration.lane != cost["lane"]
        or configuration.lock_sha256 != cost["lock_sha256"]
    ):
        raise ReceiptError(
            "pipeline.receipt_cost_unmeasured",
            f"{node}: source, target lane, lock or workspace does not match",
        )
    if not isinstance(root, Path):
        raise ReceiptError(
            "pipeline.receipt_cost_unmeasured",
            f"{node}: provide the Git checkout holding the source and target commits",
        )
    try:
        from devtools.architecture.git_inputs import CostIdentity  # lazy: cost-only Git proof

        key = (cost["source_head"], workspace.head)
        if key not in identities:
            identities[key] = CostIdentity(root, *key)
        if not identities[key].matches(cost["source_node"], node):
            raise ValueError("target node is not the same node or a unique identical move")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise ReceiptError(
            "pipeline.receipt_cost_unmeasured",
            f"{node}: restore matching source and target Git identity: {error}",
        ) from None


def _verify_tests(
    events: Sequence[Any], required_tests: Sequence[str], within_limits: bool
) -> None:
    """Check requested outcomes and observed limits without another read or parse."""
    if not (required_tests or within_limits):
        return
    results = [e.test_result for e in events if "test_result" in e.id]
    latest = {r.node: r for r in sorted(results, key=lambda r: r.attempt)}
    missing = set(required_tests)
    for node, result in latest.items():
        matched = {n for n in required_tests if node_matches(node, n)}
        if required_tests and not matched:
            continue
        missing -= matched
        why = "" if result.status == "PASSED" else result.status.lower()
        limit = result.timeout_seconds
        if within_limits and not why:
            why = "unbounded" if not limit else "over-limit" if result.seconds > limit else ""
        if why:
            raise ReceiptError(
                "pipeline.receipt_mismatch",
                f"test {node!r}: {why}; record a passing bounded outcome",
            )
    if missing or not latest:
        raise ReceiptError(
            "pipeline.receipt_mismatch",
            f"missing required tests or outcomes: {sorted(missing)!r}; record them",
        )


class _Events:
    """The plugin's half on pytest's controller: one result per node, after its last phase."""

    def __init__(self, stream: TextIO) -> None:
        self.stream, self.open = stream, dict[str, dict[str, Any]]()
        self.phases: dict[str, dict[str, float]] = {}
        self.collection_errors: set[str] = set()

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if not report.failed or not str(report.longrepr).startswith("test_size.large_contract:"):
            return
        if report.nodeid in self.collection_errors:
            return  # xdist reports the same collection error from each worker
        self.collection_errors.add(report.nodeid)
        self.stream.write(
            json.dumps(
                {
                    "node": report.nodeid,
                    "attempt": 1,
                    "solo": False,
                    "status": "ERROR",
                    "seconds": 0.0,
                    "phase": "collection",
                    "failure": _failure(report),
                    "class": None,
                }
            )
            + "\n"
        )

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
        if report.when in {"setup", "call", "teardown"}:
            self.phases.setdefault(report.nodeid, {})[report.when] = report.duration
        if report.failed and node["status"] == "PASSED":
            status = "FAILED" if report.when == "call" else "ERROR"
            node.update(status=status, phase=report.when, failure=_failure(report))
        elif report.skipped and node["status"] == "PASSED":
            node["status"] = "SKIPPED"
        if report.when not in ("setup", "call"):  # its teardown, or a crashed worker's "???"
            found = dict(report.user_properties)
            phases = self.phases.pop(report.nodeid, {})
            complete = set(phases) == {"setup", "call", "teardown"} and all(
                type(value) in (int, float) and isfinite(value) and value >= 0
                for value in phases.values()
            )
            if complete:
                node["seconds"] = found.get("unit_own_seconds", sum(phases.values()))
            elif node["status"] == "PASSED":
                node.update(
                    status="ERROR",
                    phase="incomplete",
                    failure={
                        "type": "Incomplete",
                        "message": "own test phases are incomplete or invalid",
                        "signature": "",
                    },
                )
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


def size_families(events: Path) -> tuple[str, ...]:
    """Bind the stage's test families to its running workspace through the receipt owner."""
    try:
        workspace = pending_workspace(events.parent / "receipt.jsonl")
        record = json.loads((events.parent / "size-stage-selection.json").read_text("utf-8"))
        if record["base"] != workspace.base or record["head"] != workspace.head:
            raise ValueError("selection and workspace commits differ")
        return selection(record["families"])
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError(
            f"test_size.changed_unmeasured: restore the stage selection: {error}"
        ) from error


def pytest_configure(config: pytest.Config) -> None:
    if os.environ.get(EVENTS) and not hasattr(config, "workerinput"):  # workers report to it
        register(config, Path(os.environ[EVENTS]))


def pytest_unconfigure(config: pytest.Config) -> None:
    """The disposer: the registration goes and its stream closes."""
    if plugin := config.pluginmanager.get_plugin("pipeline-events"):
        config.pluginmanager.unregister(plugin)
        plugin.close()


def _failure(report: pytest.TestReport | pytest.CollectReport) -> dict[str, str]:
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
