"""Portable scenario, trace and grading contracts for deterministic evaluation."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import MISSING, asdict, dataclass, fields
from datetime import date, datetime
from math import isfinite
from typing import Any, Literal, Protocol, TypedDict, cast

import yaml

Environment = Literal["harness", "simulated", "real"]
Decider = Literal["agent", "person", "not_offered"]
DECIDERS = {"agent", "person", "not_offered"}
GEN_AI_TEXT = (
    "gen_ai.operation.name",
    "gen_ai.provider.name",
    "gen_ai.request.model",
    "gen_ai.response.model",
)
GEN_AI_COUNTS = ("gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens")


class GoldenAcquisition(TypedDict):
    """A fixture acquisition's build and copy observations, separate from its research."""

    schema: Literal["alphalattice.golden-acquisition.v1"]
    recipe: str
    cache: str
    cold: bool
    elapsed_clock: bool
    require_cold: bool
    initial_fill_seconds: float
    fill_started_at: float | None
    fill_ended_at: float | None
    copy_started_at: float
    copy_ended_at: float
    copy_seconds: float


def decode_golden_acquisition(value: Any) -> GoldenAcquisition:
    """Decode the complete protocol shape; a cached copy has no initial fill."""
    code = "replay.acquisition_invalid"
    if not isinstance(value, dict) or value.keys() != GoldenAcquisition.__required_keys__:
        raise ValueError(code)
    if (
        value["schema"] != "alphalattice.golden-acquisition.v1"
        or any(not isinstance(value[k], str) or not value[k].strip() for k in ("recipe", "cache"))
        or any(type(value[k]) is not bool for k in ("cold", "elapsed_clock", "require_cold"))
    ):
        raise ValueError(code)
    keys = ["initial_fill_seconds", "copy_seconds", "copy_started_at", "copy_ended_at"]
    if value["cold"]:
        keys += ["fill_started_at", "fill_ended_at"]
    elif (
        value["require_cold"]
        or value["initial_fill_seconds"] != 0
        or any(value[k] is not None for k in ("fill_started_at", "fill_ended_at"))
    ):
        raise ValueError(code)
    try:
        invalid = any(type(value[k]) not in (int, float) or not isfinite(value[k]) for k in keys)
    except OverflowError:
        invalid = True
    if invalid:
        raise ValueError(code)
    if any(value[k] < 0 for k in ("initial_fill_seconds", "copy_seconds")):
        raise ValueError(code)
    return cast(GoldenAcquisition, dict(value))


def admit_replay_measurement(value: Any, source: str) -> dict[str, Any]:
    """Admit the source-bound, complete owner profiles and simultaneous Host peaks."""
    if not isinstance(value, dict):
        raise ValueError("replay.receipt_incomplete")
    try:
        binding = (value["schema"], value["workload"], value["source_sha"], value["status"])
        if binding != (
            "alphalattice.replay-measurement.v1",
            "first-use-and-daily-120",
            source,
            "COMPLETE",
        ) or not re.fullmatch(r"[0-9a-f]{40}", source):
            raise ValueError("replay.receipt_binding")
        if not {"first_use", "daily"}.issubset(value["legs"]):
            raise ValueError("replay.receipt_legs")
        for leg in value["legs"].values():
            profile, meter = leg["profile"], leg["meter"]
            if (
                profile["schema"],
                profile["status"],
                profile["coverage"]["status"],
                profile["gaps"],
            ) != ("alphalattice.goal-profile.v1", "GOAL_PROFILE", "COMPLETE", []):
                raise ValueError("replay.receipt_profile")
            if (
                not profile["stages"]
                or meter["host_tree_complete"] is not True
                or meter["host_tree_observed"] is not True
            ):
                raise ValueError("replay.receipt_coverage")
            peak = meter["peak_host_private_commit_bytes"]
            if type(peak) is not int or peak <= 0:
                raise ValueError("replay.receipt_peak")
            numbers = [
                leg["wall_seconds"],
                profile["coverage"]["host_seconds"],
                *(
                    stage[phase]
                    for stage in profile["stages"]
                    for phase in ("execute_seconds", "verify_seconds")
                ),
            ]
            if any(type(n) not in (int, float) or not isfinite(n) or n < 0 for n in numbers):
                raise ValueError("replay.receipt_time")
        return cast(dict[str, Any], value)
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("replay.receipt_incomplete") from error


def _record(cls: type[Any], value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("record_must_be_object")
    names = {field.name for field in fields(cls)}
    required = {
        field.name
        for field in fields(cls)
        if field.default is MISSING and field.default_factory is MISSING
    }
    if set(value) - names or required - set(value):
        raise ValueError("unknown_or_missing_fields")
    return value


def _version(value: object) -> None:
    if type(value) is not int or value != 1:
        raise ValueError("unsupported_version")


def _identifier(value: object) -> None:
    if type(value) is int and value > 0:
        return
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("invalid_id")


def _text(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("invalid_text")


def _choice(value: object, options: set[str], reason: str) -> None:
    if not isinstance(value, str) or value not in options:
        raise ValueError(reason)


def _samples(value: object) -> None:
    if type(value) is not int or value < 1:
        raise ValueError("invalid_sample_count")


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("text_list_required")
    for item in value:
        _text(item)
    return tuple(value)


def _feedback(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != {"state", "answer", "expected"}:
        raise ValueError("invalid_feedback_context")
    if any(not isinstance(part, dict) for part in value.values()):
        raise ValueError("invalid_feedback_context")
    if value["state"].get("answer_file") != "answer.json":
        raise ValueError("invalid_feedback_answer_file")
    expected = value["expected"]
    if set(expected) - {"decision", "unusable_offers"}:
        raise ValueError("invalid_feedback_expectation")
    _choice(expected.get("decision"), {"send", "ask", "stop"}, "invalid_feedback_decision")
    unusable = expected.get("unusable_offers", {})
    if not isinstance(unusable, dict):
        raise ValueError("invalid_unusable_offers")
    _strings((*unusable, *unusable.values()))
    _attributes({"alphalattice.feedback": value})


@dataclass(frozen=True)
class ReplayInput:
    """A replay recipe and its closed sessions, independent of an agent runner."""

    schema: str
    golden: str
    sessions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema != "alphalattice.replay-input.v1":
            raise ValueError("invalid_replay_schema")
        _text(self.golden)
        _identifier(self.golden)
        sessions = _strings(self.sessions)
        days = tuple(date.fromisoformat(session) for session in sessions)
        if not days or tuple(sorted(set(days))) != days:
            raise ValueError("invalid_replay_sessions")
        if tuple(day.isoformat() for day in days) != sessions:
            raise ValueError("invalid_replay_sessions")
        object.__setattr__(self, "sessions", sessions)


@dataclass(frozen=True)
class Scenario:
    """Each route group requires one of its alternatives; all groups are required."""

    id: int | str
    sentence: str
    now: str
    injected_faults: tuple[str, ...]
    expected_route: tuple[tuple[str, ...], ...]
    decider: Decider
    forbidden_acts: tuple[str, ...]
    version: int = 1
    samples: int = 1
    ask_required: bool = True
    max_asks: int | None = None
    capability: Literal["reached", "missing"] = "reached"
    owner: str | None = None
    evidence: tuple[str, ...] = ()
    starting_state: str = ""
    feedback: dict[str, Any] | None = None
    committee: dict[str, Any] | None = None
    replay: ReplayInput | None = None

    def __post_init__(self) -> None:
        _version(self.version)
        _identifier(self.id)
        _text(self.sentence)
        if not isinstance(self.starting_state, str) or (
            self.starting_state and not self.starting_state.strip()
        ):
            raise ValueError("invalid_starting_state")
        _timestamp(self.now)
        _samples(self.samples)
        _feedback(self.feedback)
        if not isinstance(self.replay, (ReplayInput, type(None))):
            raise ValueError("invalid_replay_input")
        if self.committee is not None:
            if self.feedback is not None or not isinstance(self.committee, dict):
                raise ValueError("invalid_committee_context")
            _choice(self.committee.get("role"), {"ALPHA", "RISK", "CRO", "PM"}, "committee_role")
            _choice(self.committee.get("phase"), {"stance", "decision"}, "committee_phase")
            _attributes({"alphalattice.committee": self.committee})
        _choice(self.decider, DECIDERS, "invalid_decision")
        _choice(self.capability, {"reached", "missing"}, "invalid_capability")
        if type(self.ask_required) is not bool or (
            self.max_asks is not None and (type(self.max_asks) is not int or self.max_asks < 0)
        ):
            raise ValueError("invalid_ask_bounds")
        if self.owner is not None:
            _text(self.owner)
        for name in ("injected_faults", "forbidden_acts", "evidence"):
            object.__setattr__(self, name, _strings(getattr(self, name)))
        for fault in self.injected_faults:
            _identifier(fault)
        if any(
            not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", act) for act in self.forbidden_acts
        ):
            raise ValueError("invalid_act_id")
        if not isinstance(self.expected_route, (list, tuple)):
            raise ValueError("route_groups_required")
        groups = tuple(_strings(group) for group in self.expected_route)
        if any(not group for group in groups):
            raise ValueError("empty_route_group")
        object.__setattr__(self, "expected_route", groups)

    @classmethod
    def from_dict(cls, value: object) -> Scenario:
        body = _record(cls, value)
        if body.get("replay") is not None:
            body = {**body, "replay": ReplayInput(**_record(ReplayInput, body["replay"]))}
        return cls(**body)

    def to_dict(self) -> dict[str, Any]:
        body = cast(dict[str, Any], json.loads(json.dumps(asdict(self), ensure_ascii=False)))
        if self.replay is None:
            body.pop("replay")
        return body


@dataclass(frozen=True)
class RunnerModel:
    model: str
    host: str
    effort: str

    def __post_init__(self) -> None:
        for value in (self.model, self.host, self.effort):
            _text(value)


@dataclass(frozen=True)
class RunnerConfig:
    models: tuple[RunnerModel, ...]
    samples: int

    def __post_init__(self) -> None:
        _samples(self.samples)
        if (
            not isinstance(self.models, (list, tuple))
            or not self.models
            or any(not isinstance(model, RunnerModel) for model in self.models)
        ):
            raise ValueError("runner_models_required")
        object.__setattr__(self, "models", tuple(self.models))

    @classmethod
    def from_dict(cls, value: object) -> RunnerConfig:
        body = _record(cls, value)
        if not isinstance(body["models"], list):
            raise ValueError("runner_models_required")
        return cls(
            tuple(RunnerModel(**_record(RunnerModel, row)) for row in body["models"]),
            body["samples"],
        )


@dataclass(frozen=True)
class EvaluationDataset:
    """A versioned collection retains distinct scenario identities."""

    scenarios: tuple[Scenario, ...]
    version: int = 1
    runner: RunnerConfig | None = None

    def __post_init__(self) -> None:
        _version(self.version)
        if self.runner is not None and not isinstance(self.runner, RunnerConfig):
            raise ValueError("invalid_runner_config")
        if not isinstance(self.scenarios, (list, tuple)) or any(
            not isinstance(scenario, Scenario) for scenario in self.scenarios
        ):
            raise ValueError("scenarios_required")
        if len({str(scenario.id) for scenario in self.scenarios}) != len(self.scenarios):
            raise ValueError("duplicate_scenario_id")
        object.__setattr__(self, "scenarios", tuple(self.scenarios))

    @classmethod
    def loads_yaml(cls, text: str) -> EvaluationDataset:
        body = _record(cls, yaml.safe_load(text))
        if not isinstance(body["scenarios"], list):
            raise ValueError("scenarios_required")
        return cls(
            tuple(Scenario.from_dict(row) for row in body["scenarios"]),
            body.get("version", 1),
            RunnerConfig.from_dict(body["runner"]) if body.get("runner") is not None else None,
        )

    def dumps_yaml(self) -> str:
        body: dict[str, Any] = {
            "version": self.version,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }
        if self.runner is not None:
            body["runner"] = asdict(self.runner)
        return yaml.safe_dump(body, allow_unicode=True, sort_keys=False)


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("invalid_timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("invalid_timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp_timezone_required")
    return parsed


def _attributes(values: object) -> None:
    if not isinstance(values, dict):
        raise ValueError("attributes_must_be_object")
    for key, value in values.items():
        if not isinstance(key, str) or (
            key not in (*GEN_AI_TEXT, *GEN_AI_COUNTS)
            and not re.fullmatch(r"alphalattice\.[a-z][a-z0-9_.]*", key)
        ):
            raise ValueError("unknown_attribute")
        if key in GEN_AI_TEXT:
            _text(value)
        if key in GEN_AI_COUNTS and (type(value) is not int or value < 0):
            raise ValueError("invalid_token_count")
    if (
        set(values).intersection((*GEN_AI_TEXT, *GEN_AI_COUNTS))
        and "gen_ai.operation.name" not in values
    ):
        raise ValueError("gen_ai_operation_required")
    if "alphalattice.answer" in values:
        answer = values["alphalattice.answer"]
        if not isinstance(answer, dict):
            raise ValueError("invalid_answer_decision")
        if "decider" in answer:
            _choice(answer["decider"], DECIDERS, "invalid_answer_decision")
    try:
        json.dumps(values, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("attributes_must_be_json") from error


@dataclass(frozen=True)
class TraceEvent:
    """One completed step retains observed AI facts and namespaced domain data."""

    start_time: str
    end_time: str
    attributes: dict[str, Any]
    version: int

    def __post_init__(self) -> None:
        _version(self.version)
        if _timestamp(self.end_time) < _timestamp(self.start_time):
            raise ValueError("time_reversal")
        _attributes(self.attributes)

    @classmethod
    def from_dict(cls, value: object) -> TraceEvent:
        return cls(**_record(cls, value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Trace:
    events: tuple[TraceEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.events, (list, tuple)) or any(
            not isinstance(event, TraceEvent) for event in self.events
        ):
            raise ValueError("trace_events_required")
        object.__setattr__(self, "events", tuple(self.events))

    @classmethod
    def loads_jsonl(cls, text: str) -> Trace:
        return cls(
            tuple(
                TraceEvent.from_dict(json.loads(line)) for line in text.splitlines() if line.strip()
            )
        )

    def dumps_jsonl(self) -> str:
        return "".join(
            json.dumps(event.to_dict(), ensure_ascii=False, allow_nan=False) + "\n"
            for event in self.events
        )

    @property
    def answer(self) -> dict[str, Any]:
        for event in reversed(self.events):
            if "alphalattice.answer" in event.attributes:
                return cast(dict[str, Any], event.attributes["alphalattice.answer"])
        return {}


IssuePart = Literal["harness", "cli", "system", "feedback", "agent", "host", "evaluator"]


class GradingIssue(TypedDict):
    reason: str
    loop_part: IssuePart


class Grade(TypedDict):
    passed: bool
    reasons: list[GradingIssue]
    finding: str | None


class Runner(Protocol):
    def run(self, scenario: Scenario, environment: Environment) -> Iterable[TraceEvent]: ...


class Grader(Protocol):
    def grade(self, scenario: Scenario, trace: Trace) -> Grade: ...
