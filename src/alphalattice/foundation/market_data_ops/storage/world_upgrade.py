"""First-start migration of legacy market tables into World."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from alphalattice.foundation.market_data_ops.storage import world, world_reads
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.data_layer.upgrade import Family, UpgradeRunner


def market_upgrade_family(identify: Callable[[Path], str | None]) -> Family:
    """Admit older workspace tables through the shared, backed-up first-start protocol."""
    wanted = os.environ.get(world_reads.SETTING)

    def source(workspace: Path) -> tuple[MarketDataRepository, dict[str, list[int]]] | None:
        repository = MarketDataRepository(workspace, universe_reader="workspace")
        if not repository.path.is_file():
            return None
        with repository.database.read_transaction() as connection:
            created = connection.execute(
                "SELECT comment FROM duckdb_tables() WHERE database_name = current_database()"
                " AND schema_name = 'main' AND table_name = 'market_profile'"
            ).fetchone()
        if created == (world.WORKSPACE_MARKET_FORMAT,):
            # A current workspace copied to another machine still uses ordinary market admission.
            return None
        marks = repository._market_marks()
        if not marks:
            raise DataError(
                "Restore the older workspace's complete market tables before upgrading.",
                code="upgrade.refused",
            )
        if not any(
            marks[table][0]
            for table in (
                "raw_daily_bar_current",
                "corporate_action_current",
                "provider_symbol_mapping",
            )
        ):
            return None
        return repository, marks

    def holds(workspace: Path) -> bool:
        old = source(workspace)
        return old is not None and not world_reads._admitted(
            old[1], workspace_id=identify(workspace)
        )

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
            world.admit(old[0].path, identity)
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
                old[0].path,
                world_reads.LATEST,
                listings=old[0]._listings("SELECT listing_id FROM listing", ()),
            ):
                return False
        return True

    def switch(mode: str) -> None:
        if mode == "old":
            os.environ[world_reads.SETTING] = "workspace"
        elif wanted is None:
            os.environ.pop(world_reads.SETTING, None)
        else:
            os.environ[world_reads.SETTING] = wanted

    return Family("market-data", step, verify, switch, holds)
