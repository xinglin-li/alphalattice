"""The append boundary's form checks, case by case, on in-memory events."""

from __future__ import annotations

import json

import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.admit import admit
from alphalattice.kernel.data_layer.chain import digest

TEST_SOURCE = "src/alphalattice/kernel/data_layer/admit.py"
OPTIONAL = {"required": [], "text": "optional"}
KEY = "sk-" + "a" * 24


def _row(type_="note", attrs=None, text=None, **more):
    """An event of `type_` with `attrs`, naming `text`'s digest when it has one."""
    named = None if text is None else digest(text.encode())
    row = {"type": type_, "type_version": 1, "attrs": json.dumps(attrs or {})}
    return row | {"text_digest": named} | more


def _never(*_):
    """No definition is superseded."""
    return False


REFUSED = {
    "a required attribute missing": (
        _row("goal.set", {"note": "no title"}),
        None,
        {"required": ["title"]},
        "event_attrs_missing",
    ),
    "a text that is not the one its digest names": (
        _row(text="original"),
        "different",
        OPTIONAL,
        "event_text_unbound",
    ),
    "a required text absent": (_row(), None, {"text": "required"}, "event_text_missing"),
    "a live digest without its text": (_row(text="stated"), None, OPTIONAL, "event_text_missing"),
    "a text on a type that carries none": (
        _row(text="x"),
        "x",
        {"text": "none"},
        "text_unexpected",
    ),
    "a token count that is not a number": (
        _row(attrs={"gen_ai.usage.input_tokens": "unavailable"}),
        None,
        {},
        "event_attrs_form",
    ),
    "a boolean cost": (_row(attrs={"cost_usd": True}), None, {}, "event_attrs_form"),
    "an admission part without its knowledge times": (
        _row(attrs={"parts": ["2024"]}),
        None,
        {},
        "event_attrs_form",
    ),
    "an admission part whose range runs backwards": (
        _row(attrs={"parts": [{"part": "p", "min_known_asof_us": 2, "max_known_asof_us": 1}]}),
        None,
        {},
        "event_attrs_form",
    ),
    "a secret-named attribute with a value": (
        _row(attrs={"api_key": "x" * 12}),
        None,
        {},
        "event_secret",
    ),
    "a key-shaped value under any name": (
        _row(attrs={"tags": {"note": [KEY]}}),
        None,
        {},
        "event_secret",
    ),
    "a key in the text": (_row(text=f"key {KEY}"), f"key {KEY}", OPTIONAL, "event_secret"),
    "a credential setting's value": (
        _row("setting.changed", {"scope": "machine", "key": "provider_token", "value": 1}),
        None,
        {},
        "event_secret",
    ),
}


@pytest.mark.proves("data_layer.append_admits_only_a_keepable_event")
@pytest.mark.parametrize(("row", "text", "spec", "code"), REFUSED.values(), ids=REFUSED)
def test_an_event_the_log_cannot_keep_is_refused_by_name(row, text, spec, code):
    """Each form a permanent record must not take is refused with its code."""
    with pytest.raises(DataError) as refused:
        admit(row, text, spec, _never)
    assert refused.value.failure.code.endswith(code)


@pytest.mark.proves("data_layer.append_admits_only_a_keepable_event")
def test_an_inline_body_holding_a_key_is_refused():
    """An inline body that is text is kept forever, so it is scanned like a text."""
    with pytest.raises(DataError, match="body"):
        admit(_row(), None, {}, _never, body=f"key: {KEY}".encode())
    admit(_row(), None, {}, _never, body=b"\xff" + KEY.encode())  # not text: no scan


@pytest.mark.proves("data_layer.append_admits_only_a_keepable_event")
def test_a_live_write_under_a_superseded_definition_is_refused_and_replay_keeps_it():
    """New work never writes under a superseded metric; replay re-admits what was admitted once,
    a forgotten text included, and a keychain reference passes."""
    row = _row("result.recorded", {"measures": [["probe", 1, "", 1.0]]})
    superseded = {("metric", "probe", 1)}
    with pytest.raises(DataError, match="probe v1"):
        admit(row, None, {}, lambda *use: use in superseded)
    admit(row, None, {}, lambda *use: use in superseded, live=False)
    admit(_row(text="forgotten"), None, OPTIONAL, _never, live=False)
    admit(_row(attrs={"key": "api_key", "secret_ref": "alphalattice/provider"}), None, {}, _never)
