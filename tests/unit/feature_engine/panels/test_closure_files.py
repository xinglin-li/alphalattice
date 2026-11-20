"""The closure store's switch: a value it does not know is refused by name before any store is
read."""

from __future__ import annotations

import pytest

from alphalattice.foundation.feature_engine.panels.closure_files import (
    CLOSURE_STORE_ENV,
    closure_store_mode,
)

TEST_SOURCE = "src/alphalattice/foundation/feature_engine/panels/closure_files.py"


@pytest.mark.proves("feature_engine.closure_store_switch_refuses_an_unknown_value")
def test_an_unknown_closure_store_switch_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the log and the legacy folder are switch values; unset reads by what the log holds,
    and any other value refuses by name."""
    monkeypatch.delenv(CLOSURE_STORE_ENV, raising=False)
    assert closure_store_mode() is None
    for mode in ("log", "legacy"):
        monkeypatch.setenv(CLOSURE_STORE_ENV, mode)
        assert closure_store_mode() == mode
    monkeypatch.setenv(CLOSURE_STORE_ENV, "parquet")
    with pytest.raises(ValueError, match=r"^feature_closure\.closure_store_invalid$"):
        closure_store_mode()
