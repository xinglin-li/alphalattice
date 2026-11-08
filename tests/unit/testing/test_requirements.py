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
    structural = [SimpleNamespace(node="tests/structural/test_tree.py::test_walk")]
    assert pyramid(before + structural * 3, before) == []


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
