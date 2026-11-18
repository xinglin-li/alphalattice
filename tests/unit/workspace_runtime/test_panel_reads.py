"""The Panel read switch keys a chunk from its own metadata and refuses what it cannot serve."""

import pytest

from alphalattice.control.workspace_runtime import panel_reads

TEST_SOURCE = "src/alphalattice/control/workspace_runtime/panel_reads.py"
BINDING = {b"alphalattice.panel_binding_hash": b"b" * 64, b"alphalattice.calendar_year": b"2025"}


@pytest.mark.proves("workspace_runtime.panel_reads_key_a_chunk_by_its_binding_year_and_part")
@pytest.mark.parametrize(
    ("setting", "part", "expected"),
    [
        ("stored", b"2", ("b" * 64, "2025/2")),
        ("stored", None, ("b" * 64, "2025")),
        ("derived", b"2", RuntimeError),
        ("neither", b"2", ValueError),
    ],
    ids=["a part", "a whole year", "derived and unserved", "an unknown setting"],
)
def test_a_chunk_is_keyed_by_its_binding_year_and_part_and_unserved_reads_are_refused(
    monkeypatch: pytest.MonkeyPatch, setting: str, part: bytes | None, expected: object
) -> None:
    """A chunk's key is the binding, year and part it records; with no data layer registered a
    derived read is refused by name, a stored one reads nothing, and an unknown setting is
    refused by name."""
    held = panel_reads.registered()
    panel_reads.register(None)
    monkeypatch.setenv(panel_reads.SETTING, setting)
    try:
        if isinstance(expected, type):
            with pytest.raises(expected, match="feature_panel"):
                panel_reads.body("c" * 64)
            return
        assert panel_reads.body("c" * 64) is None
    finally:
        panel_reads.register(*(held or (None,)))
    metadata = BINDING if part is None else BINDING | {b"alphalattice.calendar_part": part}
    assert panel_reads.key(metadata) == expected
