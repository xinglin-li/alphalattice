"""Which store answers the evidence readers, as the setting names it."""

from __future__ import annotations

import pytest

from alphalattice.evidence.alternative_evidence.storage import world_reads
from alphalattice.kernel.data.errors import DataError

TEST_SOURCE = "src/alphalattice/evidence/alternative_evidence/storage/world_reads.py"


@pytest.mark.proves("alternative_evidence.evidence_reads_setting_is_named")
def test_the_reads_setting_defaults_to_world_and_refuses_another_name(monkeypatch):
    """Unset, the setting reads world; it takes workspace; any other value is refused by name."""
    monkeypatch.delenv(world_reads.SETTING, raising=False)
    chosen = [world_reads.chosen()]
    monkeypatch.setenv(world_reads.SETTING, "workspace")
    chosen.append(world_reads.chosen())
    monkeypatch.setenv(world_reads.SETTING, "both")
    with pytest.raises(DataError) as unknown:
        world_reads.chosen()
    assert chosen == ["world", "workspace"]
    assert unknown.value.failure.code == "alternative_evidence.evidence_reads_unknown"
