"""Protected-validation cases of the data root's catalog contract."""

from contextlib import closing
from pathlib import Path

import pytest

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer.log import HostLog
from alphalattice.foundation.data_layer.root import duckdb_session
from alphalattice.foundation.data_layer.store import StorePort


def assert_old_receipt_catalog_is_refused(root: Path) -> None:
    """The Host refuses a catalog that predates validation receipts."""
    with closing(duckdb_session(root)) as previous:
        path = (root / "machine" / "catalog.sqlite").as_posix()
        previous.execute(
            f"ATTACH 'ducklake:sqlite:{path}' AS lake (DATA_PATH '{root.as_posix()}/data/')"
        )
        previous.execute("DROP TABLE lake.results.validation_receipt")
    with (
        HostLog.open(root) as log,
        pytest.raises(
            DataError, match=r"results\.validation_receipt differ.*start a fresh data root"
        ) as stale,
    ):
        StorePort.open(root, log)
    assert stale.value.failure.code == "data_layer.schema_mismatch"
