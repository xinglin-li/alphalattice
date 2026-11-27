"""A connection whose owner installs the digest at every instance's open asks no catalog for it."""

from typing import Any, cast

import pytest

from alphalattice.control.workspace_runtime import reader_threads

TEST_SOURCE = "src/alphalattice/control/workspace_runtime/reader_threads.py"


class _Untouchable:
    """A connection any statement on which fails the test."""

    BLAKE3_SQL_INSTALLED = True

    def execute(self, *_: object) -> None:
        raise AssertionError("the digest's catalog was asked")


@pytest.mark.proves("workspace_runtime.a_workspace_connection_asks_no_catalog_for_the_digest")
def test_a_connection_that_holds_the_digest_runs_no_statement() -> None:
    """Installing the digest on a connection whose owner declares it installed (as the workspace
    database's connections do) runs no statement at all."""
    reader_threads.install_blake3_sql(cast(Any, _Untouchable()))
