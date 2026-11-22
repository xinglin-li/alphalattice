import pytest

from alphalattice.control.observation_runtime.contracts import (
    ObservationReadCursor,
    ObservationReadPage,
)

TEST_SOURCE = "src/alphalattice/control/observation_runtime/contracts.py"


@pytest.mark.parametrize(("head", "advances"), [(10, True), (3, False)])
@pytest.mark.proves("observation_runtime.a_page_that_does_not_advance_is_named")
def test_a_page_says_whether_it_moves_the_reader_past_its_cursor(head, advances):
    """A page whose next cursor is no further on than the one it answered does not advance, so a
    reader following it stops by name instead of reading it again forever."""
    epoch = "0" * 32
    page = ObservationReadPage(
        disposition="CONTINUED", store_epoch=epoch, head_ordinal=head, items=(), more=True
    )
    assert page.advances(ObservationReadCursor(store_epoch=epoch, ordinal=3)) is advances
