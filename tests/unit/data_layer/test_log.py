"""An event's id, minted without a database."""

from __future__ import annotations

import uuid

import pytest

from alphalattice.foundation.data_layer.log import uuid7

TEST_SOURCE = "src/alphalattice/foundation/data_layer/log.py"
NOW_NS = 1_794_728_593_622_519_000


@pytest.mark.proves("data_layer.event_ids_are_uuidv7")
def test_an_event_id_is_a_uuidv7_led_by_its_milliseconds():
    """The id is version 7 of the RFC 9562 variant, its first 48 bits the Unix milliseconds, so
    ids sort by the millisecond they were minted in, and two in one millisecond differ."""
    first, second, later = uuid7(NOW_NS), uuid7(NOW_NS), uuid7(NOW_NS + 1_000_000)
    parsed = uuid.UUID(first)
    assert (parsed.version, parsed.variant) == (7, uuid.RFC_4122)
    assert parsed.int >> 80 == NOW_NS // 1_000_000
    assert first != second
    assert max(first, second) < later
