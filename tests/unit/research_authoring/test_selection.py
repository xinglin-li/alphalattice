import pytest

from alphalattice.protocols.research_authoring.selection import allowed_value_notes

TEST_SOURCE = "src/alphalattice/protocols/research_authoring/selection.py"


@pytest.mark.proves("research_authoring.schema_notes_end_on_a_ref_cycle")
def test_a_self_referencing_definition_ends_its_notes_instead_of_hanging():
    """A `$ref` chain that names a definition again is a cycle: it lists nothing and ends, while
    the schema's other enumerated fields are still listed."""
    schema = {
        "$defs": {
            "Loop": {"$ref": "#/$defs/Back"},
            "Back": {"$ref": "#/$defs/Loop"},
            "Side": {"enum": ["long", "short"]},
        },
        "properties": {"loop": {"$ref": "#/$defs/Loop"}, "side": {"$ref": "#/$defs/Side"}},
    }
    assert allowed_value_notes(schema) == ["side: long | short"]
