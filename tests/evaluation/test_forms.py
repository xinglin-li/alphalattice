"""Portable evaluation forms reject malformed scenarios and step records."""

from __future__ import annotations

import json

import pytest

from devtools.evaluation import (
    EvaluationDataset,
    RunnerConfig,
    RunnerModel,
    Scenario,
    Trace,
    TraceEvent,
    grade,
)


def scenario(**changes):
    return {
        "id": "sample.route",
        "sentence": "Read the current status.",
        "now": "2026-01-02T10:00:00+00:00",
        "injected_faults": [],
        "expected_route": [["status show", "STATUS"]],
        "decider": "agent",
        "forbidden_acts": ["ui_click"],
        **changes,
    }


def answer(**changes):
    return {
        "commands": ["alphalattice status show --id <id>"],
        "fields": [],
        "steps": [],
        "acts": [],
        "decider": "agent",
        "ask": [],
        "refuse": [],
        **changes,
    }


def event(**changes):
    return {
        "version": 1,
        "start_time": "2026-01-02T10:00:00+00:00",
        "end_time": "2026-01-02T10:00:01+00:00",
        "attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "example",
            "gen_ai.request.model": "sample",
            "gen_ai.response.model": "sample",
            "gen_ai.usage.input_tokens": 3,
            "gen_ai.usage.output_tokens": 2,
            "alphalattice.environment": "harness",
            "alphalattice.scenario_id": "sample.route",
            "alphalattice.answer": answer(),
        },
        **changes,
    }


def test_scenario_yaml_round_trips_route_alternatives_and_metadata():
    """YAML preserves required route groups and optional decision evidence."""
    row = Scenario.from_dict(
        scenario(
            id=1,
            injected_faults=["offline"],
            ask_required=False,
            max_asks=0,
            samples=3,
            owner="Status reader",
            evidence=["guide:1"],
            starting_state="Selected retained result.",
        )
    )
    runner = RunnerConfig(
        (
            RunnerModel("example-a", "text-host", "medium"),
            RunnerModel("example-b", "code-host", "low"),
        ),
        3,
    )
    original = EvaluationDataset((row,), runner=runner)
    rebuilt = EvaluationDataset.loads_yaml(original.dumps_yaml())
    assert rebuilt == original
    assert rebuilt.scenarios[0].expected_route == (("status show", "STATUS"),)
    assert row.to_dict()["version"] == 1
    assert rebuilt.scenarios[0].samples == 3
    assert Scenario.from_dict(scenario()).samples == 1
    assert Scenario.from_dict(scenario()).starting_state == ""
    assert rebuilt.scenarios[0].starting_state == row.starting_state
    assert rebuilt.runner == runner
    assert EvaluationDataset((row,)).runner is None


@pytest.mark.parametrize(
    "config",
    [
        {"models": [], "samples": 1},
        {"models": "example", "samples": 1},
        {"models": [{}], "samples": 1},
        {
            "models": [{"model": "example", "host": "text-host", "effort": "low", "extra": True}],
            "samples": 1,
        },
        {"models": [{"model": "", "host": "text-host", "effort": "low"}], "samples": 1},
        {"models": [{"model": "example", "host": False, "effort": "low"}], "samples": 1},
        {"models": [{"model": "example", "host": "text-host", "effort": None}], "samples": 1},
        {"models": [{"model": "example", "host": "text-host", "effort": "low"}], "samples": True},
        {"models": [{"model": "example", "host": "text-host", "effort": "low"}], "samples": 0},
        {"models": [{"model": "example", "host": "text-host", "effort": "low"}]},
        {
            "models": [{"model": "example", "host": "text-host", "effort": "low"}],
            "samples": 1,
            "extra": True,
        },
    ],
)
def test_runner_configuration_requires_model_specs_and_positive_sample_counts(config):
    """Runner configuration rejects missing model facts and invalid sample counts."""
    with pytest.raises(ValueError):
        EvaluationDataset.loads_yaml(
            json.dumps({"version": 1, "scenarios": [scenario()], "runner": config})
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"unexpected": 1},
        {"version": 2},
        {"version": True},
        {"id": True},
        {"id": 0},
        {"id": "bad id"},
        {"sentence": ""},
        {"starting_state": " "},
        {"starting_state": None},
        {"starting_state": False},
        {"now": "2026-01-02T10:00:00"},
        {"now": "invalid"},
        {"now": None},
        {"samples": 0},
        {"samples": -1},
        {"samples": True},
        {"samples": 1.5},
        {"injected_faults": "offline"},
        {"injected_faults": ["bad id"]},
        {"expected_route": ["status show"]},
        {"expected_route": [[]]},
        {"decider": "system"},
        {"decider": []},
        {"forbidden_acts": [1]},
        {"forbidden_acts": ["bad(act"]},
        {"ask_required": 1},
        {"max_asks": -1},
        {"max_asks": True},
        {"capability": "unknown"},
        {"evidence": [False]},
        {"owner": 1},
    ],
)
def test_scenario_rejects_invalid_fields(changes):
    """Malformed declarations fail before a runner can consume them."""
    with pytest.raises(ValueError):
        Scenario.from_dict(scenario(**changes))


@pytest.mark.parametrize(
    "body",
    [
        {"version": 1, "scenarios": [], "extra": True},
        {"version": True, "scenarios": []},
        {"version": 1, "scenarios": {}},
        {"version": 1, "scenarios": [scenario(id=1), scenario(id="1")]},
        {"version": 1, "scenarios": [{"id": "incomplete"}]},
        {
            "version": 1,
            "scenarios": [{key: value for key, value in scenario().items() if key != "now"}],
        },
    ],
)
def test_dataset_requires_complete_distinct_scenarios(body):
    """Dataset identities cannot collide or hide malformed records."""
    with pytest.raises(ValueError):
        EvaluationDataset.loads_yaml(json.dumps(body))


def test_trace_jsonl_round_trips_observed_attributes_and_product_answers():
    """Completed steps retain only observed model facts and product answers in JSONL."""
    model = TraceEvent.from_dict(event())
    product = TraceEvent.from_dict(
        event(
            attributes={
                "alphalattice.environment": "real",
                "alphalattice.command": "status show",
                "alphalattice.answer": {"status": "OK"},
                "alphalattice.outcome": "completed",
            },
        )
    )
    partial = TraceEvent.from_dict(event(attributes={"gen_ai.operation.name": "chat"}))
    trace = Trace((model, product, partial))
    rebuilt = Trace.loads_jsonl(trace.dumps_jsonl())
    assert rebuilt == trace
    assert rebuilt.answer == {"status": "OK"}
    assert rebuilt.events[0].attributes["gen_ai.usage.input_tokens"] == 3
    assert rebuilt.events[1].attributes == product.attributes
    assert rebuilt.events[2].attributes == {"gen_ai.operation.name": "chat"}


@pytest.mark.parametrize(
    "changes",
    [
        {"extra": True},
        {"version": True},
        {"version": 2},
        {"start_time": "2026-01-02T10:00:00"},
        {"end_time": "invalid"},
        {"end_time": "2026-01-02T09:59:59+00:00"},
    ],
)
def test_trace_rejects_invalid_times_and_fields(changes):
    """Trace records require known fields and ordered timezone aware timestamps."""
    with pytest.raises(ValueError):
        TraceEvent.from_dict(event(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"gen_ai.usage.input_tokens": -1},
        {"gen_ai.usage.output_tokens": True},
        {"gen_ai.request.model": ""},
        {"gen_ai.provider.name": 1},
        {"custom.command": "status show"},
        {"gen_ai.extra": "unknown"},
        {"alphalattice.answer": answer(decider="system")},
        {"alphalattice.answer": ["OK"]},
        {"alphalattice.measurement": float("nan")},
    ],
)
def test_trace_attributes_require_typed_usage_and_domain_namespaces(changes):
    """Invalid usage, decisions and unnamespaced attributes cannot enter a trace."""
    attributes = {**event()["attributes"], **changes}
    with pytest.raises(ValueError):
        TraceEvent.from_dict(event(attributes=attributes))
    attributes = event()["attributes"]
    attributes.pop("gen_ai.operation.name")
    with pytest.raises(ValueError):
        TraceEvent.from_dict(event(attributes=attributes))


def test_intent_grading_consumes_validated_forms_and_an_explicit_command_catalog():
    """The grader consumes portable forms and rejects flags absent from its catalog."""
    row = Scenario.from_dict(scenario())
    trace = Trace((TraceEvent.from_dict(event()),))
    catalog = {("status", "show"): {"--id"}}
    assert grade(row, trace, catalog)["passed"]
    planted = answer(commands=["alphalattice status show --invented value"])
    result = grade(row, planted, catalog)
    assert not result["passed"]
    assert result["reasons"] == [
        {"reason": "unknown_flag:status show --invented", "loop_part": "cli"}
    ]
    assert result["finding"] == "harness_hint"
