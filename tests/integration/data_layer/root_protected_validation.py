"""Protected-validation cases of the data root's catalog contract."""

from contextlib import closing
from pathlib import Path

from alphalattice.kernel.data_layer.log import HostLog
from alphalattice.kernel.data_layer.root import duckdb_session
from alphalattice.kernel.data_layer.store import StorePort


def assert_old_receipt_catalog_opens(root: Path) -> None:
    """The Host's writer extends a catalog that predates validation receipts."""
    with closing(duckdb_session(root)) as previous:
        path = (root / "machine" / "catalog.sqlite").as_posix()
        previous.execute(
            f"ATTACH 'ducklake:sqlite:{path}' AS lake (DATA_PATH '{root.as_posix()}/data/')"
        )
        previous.execute("DROP TABLE lake.results.validation_receipt")
    with HostLog.open(root) as log:
        port = StorePort.open(root, log)
        assert port.lake.execute(
            "SELECT count(*) FROM lake.results.validation_receipt"
        ).fetchone() == (0,)
        port.close()
