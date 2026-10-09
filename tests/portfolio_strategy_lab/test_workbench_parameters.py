"""The Local Web workbench keeps every design value in design/parameters.json (laws 105-108).

The gate passes on the checkout, and it refuses what a new chapter would otherwise slip in: a
literal the ratchet has not seen, a sheet re-declaring a parameter the source owns, a custom
property the source does not know, and a parameter nothing reads.
"""

from __future__ import annotations

import json
import re
import shutil
from copy import deepcopy

import pytest
from scripts import check_ui_parameters as gate


def test_the_sheets_keep_every_value_in_the_parameter_source():
    assert gate.problems() == []


@pytest.fixture
def copy_of_the_source(tmp_path, monkeypatch):
    src = tmp_path / "workbench-source"
    shutil.copytree(gate.SRC / "css", src / "css")
    shutil.copytree(gate.SRC / "design", src / "design")
    shutil.copytree(gate.SRC / "js", src / "js")
    monkeypatch.setattr(gate, "SRC", src)
    monkeypatch.setattr(gate, "PARAMETERS", src / "design/parameters.json")
    monkeypatch.setattr(gate, "BASELINE", src / "design/literals-baseline.json")
    return src


def test_the_gate_refuses_a_new_literal_a_stray_parameter_and_an_unknown_property(
    copy_of_the_source,
):
    sheet = copy_of_the_source / "css/06-pages.css"
    sheet.write_text(
        sheet.read_text(encoding="utf-8")
        + ".new-chapter-row { min-height: 31px; }\n"
        + ".new-chapter-card { --space-4: 17px; }\n"
        + ".new-chapter-gap { --chapter-gap: var(--space-3); }\n",
        encoding="utf-8",
    )
    found = "\n".join(gate.problems())
    assert "`.new-chapter-row` { min-height: … } writes 31px" in found
    assert "declares --space-4, which design/parameters.json owns" in found
    assert "declares --chapter-gap, which design/parameters.json does not know" in found


def test_the_gate_refuses_a_glass_value_without_its_opaque_one(copy_of_the_source):
    # (the user, 2026-09-24): the product wears the glass; Opaque controls switches to the
    # configuration without it, which the source records for every value of the glass
    path = copy_of_the_source / "design/parameters.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    glass = source["groups"]["material.glass"]["tokens"]
    del glass["glass-fill"]["opaque"]
    glass["glass-control"]["base"] = "rgba(0,0,0,.04)"  # a tint only a glass rule reads needs none
    path.write_text(json.dumps(source), encoding="utf-8")
    found = gate.problems()
    assert any(p.startswith("parameters.json: --glass-fill has no `opaque` value") for p in found)
    assert not any("--glass-control has no" in p for p in found)


def test_the_gate_refuses_a_parameter_nothing_reads(copy_of_the_source):
    path = copy_of_the_source / "design/parameters.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    source["groups"]["scale.space"]["tokens"]["space-7"] = {"base": "28px"}
    path.write_text(json.dumps(source), encoding="utf-8")
    assert "parameters.json: nothing reads --space-7" in gate.problems()


def test_the_gate_refuses_an_unregistered_breakpoint_and_the_new_literal_kinds(copy_of_the_source):
    sheet = copy_of_the_source / "css/06-pages.css"
    sheet.write_text(
        sheet.read_text(encoding="utf-8")
        + "@media (width < 1000px) { .new-chapter-fold { display: none; } }\n"
        + "@media (max-width: 640px) { .new-chapter-phone { display: none; } }\n"
        + "@container (width >= 700px) { .new-chapter-wide { display: flex; } }\n"
        + ".new-chapter-dim { opacity: .4; z-index: 7; "
        + "color: color-mix(in srgb, var(--text) 18%, transparent); }\n",
        encoding="utf-8",
    )
    found = "\n".join(gate.problems())
    assert "`(width < 1000px)` names 1000px, which is not a registered window breakpoint" in found
    assert "`(max-width: 640px)` -- write a breakpoint in range syntax" in found
    assert "`(width >= 700px)` names 700px, which is not a registered container breakpoint" in found
    assert (
        "writes opacity .4" in found and "writes z 7" in found and "writes tint color-mix" in found
    )


PLANTED = ".qa-planted-mark"
PLANTED_RULE = f"{PLANTED} {{ right: 3px; }}\n"
PLANTED_KEY = f"02-shell.css |  | {PLANTED} | right | 3px"
PLANTED_TABLET = f"@media (width < 768px) {{\n{PLANTED} {{ min-height: 75px; }}\n}}\n"
PLANTED_TABLET_KEY = f"06-pages.css | (width < 768px) | {PLANTED} | min-height | 75px"


@pytest.fixture
def planted(copy_of_the_source):
    """The copy with two literals the ratchet knows and the exceptions name -- one in a sheet's
    plain rules, one in its tablet block -- for the tests that strip and move a literal: never
    one of the product's own, which the ratchet exists to remove."""
    shell = copy_of_the_source / "css/02-shell.css"
    shell.write_text(shell.read_text(encoding="utf-8") + PLANTED_RULE, encoding="utf-8")
    pages = copy_of_the_source / "css/06-pages.css"
    pages.write_text(pages.read_text(encoding="utf-8") + PLANTED_TABLET, encoding="utf-8")
    baseline = copy_of_the_source / "design/literals-baseline.json"
    known = json.loads(baseline.read_text(encoding="utf-8"))
    known["literals"].update({PLANTED_KEY: 1, PLANTED_TABLET_KEY: 1})
    baseline.write_text(json.dumps(known), encoding="utf-8")
    exceptions = copy_of_the_source / "design/literal-exceptions.json"
    named = json.loads(exceptions.read_text(encoding="utf-8"))
    named["reasons"].update({PLANTED_KEY: "a planted mark", PLANTED_TABLET_KEY: "a planted mark"})
    exceptions.write_text(json.dumps(named), encoding="utf-8")
    assert gate.problems() == []
    return copy_of_the_source


def test_every_literal_left_is_a_named_exception(planted):
    path = planted / "design/literal-exceptions.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    assert source["reasons"].pop(PLANTED_KEY)
    source["reasons"]["06-pages.css |  | .gone-row | width | 3px"] = (
        "a reason for a literal no sheet writes"
    )
    path.write_text(json.dumps(source), encoding="utf-8")
    found = "\n".join(gate.problems())
    assert f"{PLANTED_KEY} is an exception without a reason" in found
    assert (
        "names 06-pages.css |  | .gone-row | width | 3px, which no sheet writes any more" in found
    )


def test_a_literal_moved_to_another_sheet_is_not_new_but_a_copy_of_it_is(planted):
    shell, material = planted / "css/02-shell.css", planted / "css/12-material.css"
    shell.write_text(shell.read_text(encoding="utf-8").replace(PLANTED_RULE, ""), encoding="utf-8")
    material.write_text(material.read_text(encoding="utf-8") + PLANTED_RULE, encoding="utf-8")
    assert (
        gate.problems() == []
    )  # one owner per selector folds rules across sheets: the value moved, its reason with it
    copied = PLANTED_RULE.replace(PLANTED, ".new-chapter-badge")
    material.write_text(material.read_text(encoding="utf-8") + copied, encoding="utf-8")
    assert any("`.new-chapter-badge` { right: … } writes 3px" in p for p in gate.problems())


def test_a_literal_whose_rule_moved_to_another_breakpoint_is_not_new(planted):
    sheet = planted / "css/06-pages.css"
    moved = PLANTED_TABLET.replace("(width < 768px)", "(width < 640px)")
    sheet.write_text(
        sheet.read_text(encoding="utf-8").replace(PLANTED_TABLET, moved), encoding="utf-8"
    )
    assert gate.problems() == []


def test_red_is_failure_alone():
    """Red indicates failure and no other standing."""

    p = json.loads(gate.PARAMETERS.read_text(encoding="utf-8"))
    meanings = p["groups"]["role.meaning"]["tokens"]
    assert set(p["tones"]) == set(meanings)
    assert [name for name, v in meanings.items() if v["base"] == "danger"] == ["meaning-failure"]
    status = (gate.SRC / "js/app/status.js").read_text(encoding="utf-8")
    start = status.index("const STATES = {")
    table = status[start : status.index("\n};", start)]
    failures = set(re.findall(r"\n  (\w+): \{[^\n]*?tone: TONE\.failure", table))
    assert failures == {"failed", "error", "unavailable", "missing_or_tampered"}
    for held in ("blocked", "refused"):
        assert re.search(rf"\n  {held}: \{{[^\n]*?tone: TONE\.attention", table), held
    assert not re.search(r"tone: '", table), "a state names its meaning, not a tone"
    danger = r"""['"]danger['"]|class="[^"]*\bdanger\b"""
    named = {
        f.name: len(re.findall(danger, f.read_text(encoding="utf-8")))
        for f in (gate.SRC / "js/app").glob("*.js")
    }
    assert {k: v for k, v in named.items() if v} == {"pages-kit.js": 1}


def missing_edge_contrast_modes(design):
    """Inventory boundaries first, then require a contrast mode for each theme."""
    owners = {
        name: values
        for group, body in design["groups"].items()
        for name, values in body["tokens"].items()
        if (group == "role.line" and name != "rule-width")
        or (
            name.endswith("-edge")
            and any(
                key != "about" and not re.fullmatch(r"[\d.]+%", value)
                for key, value in values.items()
            )
        )
        or (group == "material.glass" and name.endswith("-rim"))
    }
    return [
        (name, theme)
        for name, values in owners.items()
        for theme in ("light", "dark")
        if "contrast" not in values and f"{theme}-contrast" not in values
    ]


def test_every_edge_owner_declares_increased_contrast():
    """the sweep cannot silently omit an undeclared glass mode."""
    design = json.loads(gate.PARAMETERS.read_text(encoding="utf-8"))
    assert missing_edge_contrast_modes(design) == []
    broken = deepcopy(design)
    control = broken["groups"]["material.glass"]["tokens"]["glass-control-edge"]
    control.pop("contrast")
    assert missing_edge_contrast_modes(broken) == [
        ("glass-control-edge", "light"),
        ("glass-control-edge", "dark"),
    ]
    # One theme alone cannot conceal the absent mode in the other theme.
    control["light-contrast"] = "var(--edge-card)"
    assert missing_edge_contrast_modes(broken) == [("glass-control-edge", "dark")]
