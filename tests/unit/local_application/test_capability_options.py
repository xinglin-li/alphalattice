"""A command that runs a capability takes its options from that capability's input schema."""

from __future__ import annotations

import pytest

from alphalattice.interface.local_application.capability_options import capability_parser

TEST_SOURCE = "src/alphalattice/interface/local_application/capability_options.py"

SCHEMA = {
    "type": "object",
    "required": ["window"],
    "properties": {
        "window": {"type": "integer"},
        "prices": {"type": "array", "items": {"type": "number"}},
        "side": {"type": "string", "enum": ["long", "short"]},
        "adjusted": {"type": "boolean"},
        "shape": {"type": "object"},
    },
}


@pytest.mark.proves("local_application.a_capability_command_takes_its_schemas_options")
@pytest.mark.parametrize("case", ["options", "help"])
def test_a_capability_command_takes_its_options_from_its_input_schema(
    capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """Each scalar property, or array of scalars, is an option of its own with its type; any
    other property is named for `--params`; `--help` lists them, the required one marked."""
    parser = capability_parser("fixture.window", SCHEMA)
    if case == "help":
        with pytest.raises(SystemExit) as stopped:
            parser.parse_args(["--help"])
        shown = capsys.readouterr().out
        assert stopped.value.code == 0
        assert "--prices NUMBER [NUMBER ...]" in shown and "--adjusted, --no-adjusted" in shown
        assert "(required)" in shown and "Through --params: shape." in shown
        return
    given = parser.parse_args(["--window", "20", "--prices", "1.5", "2", "--side", "short"])
    assert vars(given) == {
        "window": 20,
        "prices": [1.5, 2.0],
        "side": "short",
        "adjusted": None,
    }
    with pytest.raises(SystemExit):
        parser.parse_args(["--side", "flat"])
