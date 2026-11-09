import json
from types import SimpleNamespace

import pytest

from devtools.testing.requirements import lower_layer, pyramid

TEST_SOURCE = "src/devtools/testing/requirements.py"


def holders(**counts: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(node=f"tests/{layer}/{area}/test_x.py::test_{n}")
        for key, count in counts.items()
        for layer, area in [key.split("__")]
        for n in range(count)
    ]


@pytest.mark.proves("testing.integration_share_only_falls")
def test_an_areas_integration_share_may_only_fall():
    """A change raising an area's integration share is refused by name; falling or new is kept."""
    before = holders(integration__risk=3, unit__risk=1, integration__feature=2)
    (refused,) = pyramid(holders(integration__risk=4, unit__risk=1, integration__feature=2), before)
    assert refused.startswith("Test pyramid risk for risk:") and "3/4 to 4/5" in refused
    assert pyramid(holders(integration__risk=4, unit__risk=2, integration__feature=3), before) == []
    assert pyramid(holders(integration__risk=3, unit__risk=1, integration__new=5), before) == []
    retired = holders(integration__retired=3, unit__retired=2)
    remaining = holders(integration__retired=3, unit__retired=1)
    removed_test = "tests/unit/retired/test_x.py"
    removed_source = "scripts/retired/example.py"
    old_text = "from scripts.retired.example import run\ndef test_1(): run()\n"
    proof = {removed_test: old_text}
    # Only a whole test file for a deleted source is exempt; a partial file is not.
    remaining[-1].node = "tests/unit/retired/test_other.py::test_0"
    retired[-2].node = remaining[-1].node
    assert pyramid(remaining, retired, retired_tests=proof, removed_sources={removed_source}) == []
    assert pyramid(remaining, retired, retired_tests=proof, removed_sources=set())
    assert pyramid(remaining, retired, retired_tests=proof, removed_sources={"scripts/other.py"})
    assert pyramid(remaining, retired, retired_tests={}, removed_sources={removed_source})
    assert pyramid(
        remaining,
        retired,
        retired_tests={removed_test: old_text + "from scripts.live import run\n"},
        removed_sources={removed_source},
    )
    assert pyramid(
        remaining,
        retired,
        retired_tests={removed_test: old_text + "import importlib\nimportlib.import_module('x')\n"},
        removed_sources={removed_source},
    )
    assert pyramid(
        remaining,
        retired,
        retired_tests={
            removed_test: "from scripts.retired.example import run\ndef test_1(): pass\n"
        },
        removed_sources={removed_source},
    )
    assert pyramid(
        remaining + holders(integration__other=1),
        retired,
        retired_tests=proof,
        removed_sources={removed_source},
    )
    assert pyramid(
        [*remaining, SimpleNamespace(node="tests/integration/retired/test_added.py::test_new")],
        retired,
        retired_tests=proof,
        removed_sources={removed_source},
    )
    structural = [SimpleNamespace(node="tests/structural/test_tree.py::test_walk")]
    assert pyramid(before + structural * 3, before) == []
    old = SimpleNamespace(
        node="tests/unit/risk/test_x.py::test_x", ids=("risk.example",), digest="same"
    )
    moved = SimpleNamespace(
        node=old.node.replace("/unit/", "/integration/"), ids=old.ids, digest=old.digest
    )
    old_rows = {"risk.example": {"layer": "unit"}}
    rows = {
        "risk.example": {
            "layer": "integration",
            "description": "A result.",
            "lower_layer": "The real Host dependency cannot run in unit.",
        }
    }
    assert pyramid([moved], [old], rows, old_rows) == []
    assert pyramid(
        [moved], [old], {"risk.example": rows["risk.example"] | {"lower_layer": ""}}, old_rows
    )
    changed = SimpleNamespace(node=moved.node, ids=moved.ids, digest="changed")
    assert pyramid([changed], [old], rows, old_rows)
    added = SimpleNamespace(node="tests/integration/risk/test_y.py::test_y", ids=(), digest="other")
    unit = SimpleNamespace(node="tests/unit/risk/test_y.py::test_y", ids=(), digest="unit")
    assert pyramid([moved, unit, added], [old, unit], rows, old_rows)


@pytest.mark.proves("testing.integration_share_only_falls")
def test_a_new_row_above_unit_says_why_a_lower_layer_cannot_prove_it(tmp_path, monkeypatch):
    """A row added or raised above unit is refused without its reason; unit needs none."""
    row = {"description": "A Host restarts on its last port.", "layer": "integration"}
    assert "lower_layer" in lower_layer("a.b", row, None)[0]
    why = "It needs the served Host's real loopback socket and database."
    assert lower_layer("a.b", row | {"lower_layer": why}, None) == []
    assert lower_layer("a.b", row | {"lower_layer": row["description"]}, None)
    assert lower_layer("a.b", row | {"layer": "unit"}, None) == []
    assert lower_layer("a.b", row, {"layer": "integration"}) == []  # an old row keeps its standing
    assert lower_layer("a.b", row | {"layer": "e2e"}, {"layer": "integration"})  # raised

    # A new integration function on an old row is also new work.
    from devtools.testing import requirements as r

    monkeypatch.setattr(r, "load_policy", lambda *_: {"limits": {}})
    existing = row | {"owner": "devtools.testing", "seconds": 5}
    document = {"schema": "alphalattice.requirements.v1", "entries": {"testing.example": existing}}
    text = json.dumps(document)
    code = "@pytest.mark.proves('testing.example')\ndef test_old(): pass\n"
    path = "tests/integration/testing/test_example.py"
    old = r.file_holders(path, code)
    fresh = r.file_holders(path, code.replace("test_old", "test_new"))
    assert "lower_layer" in " ".join(
        r.guard(tmp_path, fresh, text, text, tests=old + fresh, before_tests=old)
    )
    existing["lower_layer"] = why
    admitted = json.dumps(document)
    assert r.guard(tmp_path, fresh, admitted, admitted, tests=old + fresh, before_tests=old) == []
