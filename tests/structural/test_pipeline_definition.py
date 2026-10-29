import copy
from pathlib import Path

import pytest

from devtools.architecture.test_sizes import load_policy
from devtools.pipeline import definition

ROOT = Path(__file__).resolve().parents[2]
ROW = {"kind": "command", "owner": "devtools.pipeline", "reason": "r", "hosts": ["public"]}
STAGE = dict(order=1, reason="r", trigger="line", range="card", bound_minutes=10, lane="merge")
VALID = {
    "schema": definition.SCHEMA,
    "owner": "devtools.pipeline",
    "stages": {"commit": {**STAGE, "sizes": ["small"], "rows": ["guard", "tests"]}},
    "rows": {
        "guard": {**ROW, "run": ["x"], "escape": "e", "review": "2026-12-01"},
        "tests": {**ROW, "kind": "pytest", "plugins": ["events"]},
        "events": {**ROW, "kind": "pytest-plugin", "module": "m"},
    },
}
GONE = object()
CASES = {  # where, key, value (GONE deletes it), the refusal's code
    "an unknown key": ((), "extra", 1, "schema_unknown"),
    "an unregistered row": (("stages", "commit"), "rows", ["x"], "row_unregistered"),
    "an unused row": (("rows",), "spare", {**ROW, "run": ["x"]}, "row_unused"),
    "an unknown kind": (("rows", "guard"), "kind", "script", "row_invalid"),
    "an unknown host": (("rows", "guard"), "hosts", ["cloud"], "row_invalid"),
    "an unregistered owner": (("rows", "guard"), "owner", "x", "row_invalid"),
    "an escape without its review": (("rows", "guard"), "review", GONE, "row_invalid"),
    "an unknown trigger": (("stages", "commit"), "trigger", "x", "stage_invalid"),
    "a zero bound": (("stages", "commit"), "bound_minutes", 0, "stage_invalid"),
    "an unregistered lane": (("stages", "commit"), "lane", "x", "stage_invalid"),
    "a plugin that is no plugin": (("rows", "tests"), "plugins", ["guard"], "plugin_unknown"),
    "a host that is no list": (("rows", "guard"), "hosts", 1, "row_invalid"),
    "a run that is one string": (("rows", "guard"), "run", "uv sync", "row_invalid"),
    "an empty run argument": (("rows", "guard"), "run", [""], "row_invalid"),
    "a ci-job off public": (
        ("rows",),
        "guard",
        {**ROW, "kind": "ci-job", "job": "w", "hosts": ["workshop"]},
        "row_invalid",
    ),
    "a nan bound": (("stages", "commit"), "bound_minutes", float("nan"), "stage_invalid"),
    "the shipped definition": None,
}


@pytest.fixture(scope="module")
def registries():  # the checkout's test sizes and owners, read once
    return load_policy(ROOT), definition.owners(ROOT)


@pytest.mark.parametrize("case", CASES)
def test_a_malformed_definition_is_refused_by_code_and_the_shipped_one_loads(case, registries):
    """The pipeline runs only from config/pipeline.toml: a malformed stage or row stops on load."""
    if CASES[case] is None:
        assert definition.load(ROOT).stage("commit").place == "gate"
        return
    policy, named = registries
    assert definition.check(copy.deepcopy(VALID), "", policy, named).stage("commit").rows
    where, key, value, code = CASES[case]
    document = copy.deepcopy(VALID)
    parent = document
    for step in where:
        parent = parent[step]
    _ = parent.pop(key) if value is GONE else parent.update({key: value})
    with pytest.raises(definition.DefinitionError) as refused:
        definition.check(document, "", policy, named)
    assert refused.value.code == "pipeline." + code
    assert case != "a ci-job off public" or "hosts = ['public']" in str(refused.value)  # its way on
