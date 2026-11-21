"""First-start migration of legacy market tables into World."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from alphalattice.control.workspace_runtime.database import hold_workspace_database
from alphalattice.foundation.market_data_ops.storage import world, world_reads
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.data_layer.upgrade import Family, UpgradeRunner


def market_upgrade_family(identify: Callable[[Path], str | None]) -> Family:
    """Admit older workspace tables through the shared, backed-up first-start protocol."""

    def source(workspace: Path) -> MarketDataRepository | None:
        repository = MarketDataRepository(workspace, universe_reader="workspace")
        if not repository.path.is_file():
            return None
        with (
            hold_workspace_database(repository.path),
            repository.database.read_transaction() as connection,
        ):
            created, tables = connection.execute(
                "SELECT max(comment) FILTER (WHERE table_name = 'market_profile'),"
                " list(table_name) FROM duckdb_tables()"
                " WHERE database_name = current_database() AND schema_name = 'main'"
            ).fetchone()
            if created == world.WORKSPACE_MARKET_FORMAT:
                # A current workspace copied to another machine is published whole by its
                # first publish there (world.publish).
                return None
            if not set(tables or ()) >= set(world.TABLES):
                raise DataError(
                    "Restore the older workspace's complete market tables before upgrading.",
                    code="upgrade.refused",
                )
            rows = connection.execute(
                "SELECT (SELECT count(*) FROM raw_daily_bar_current)"
                " + (SELECT count(*) FROM corporate_action_current)"
                " + (SELECT count(*) FROM provider_symbol_mapping)"
            ).fetchone()[0]
        return repository if rows else None

    def holds(workspace: Path) -> bool:
        identity = identify(workspace)
        return source(workspace) is not None and (identity is None or not world.published(identity))

    def step(runner: UpgradeRunner) -> None:
        for workspace in runner.workspaces:
            old = source(workspace)
            if old is None:
                continue
            identity = identify(workspace)
            if identity is None:
                raise DataError(
                    "Restore the market workspace's manifest identity.", code="upgrade.refused"
                )
            history.workspace_on_log(workspace, workspace.name, identity, root=runner.root)
            world.admit(old.path, identity)
            port = history.store()
            if port is None:
                raise DataError(
                    "Open the Host's catalog before upgrading markets.", code="upgrade.refused"
                )
            for mic, market_id in port.lake.execute(
                "SELECT mic, CAST(market_id AS VARCHAR) FROM market_asof(?)", [world_reads.LATEST]
            ).fetchall():
                runner.map_id("market-data", mic, market_id)

    def verify(runner: UpgradeRunner) -> bool:
        for workspace in runner.workspaces:
            old = source(workspace)
            if old is not None and not world_reads.holds(
                old.path,
                world_reads.LATEST,
                listings=old._listings("SELECT listing_id FROM listing", ()),
            ):
                return False
        return True

    return Family("market-data", step, verify, holds)
