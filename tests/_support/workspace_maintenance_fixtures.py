"""The qualified workspace the local data update suites start from."""

from __future__ import annotations

import pytest

from alphalattice.control.product_host.composition.research_workspace import (
    publish_research_workspace_manifest,
)
from tests.portfolio_strategy_lab.local_web_support import _manifest
from tests.researcher_methodology_surface.real_workspace import build_real_risk_workspace
from tests.researcher_methodology_surface.session_workspace import copy_workspace, session_workspace
from tests.workspace_maintenance.data_update_support import (
    _seed_foundation,
    set_parallel_test_budget,
)


@pytest.fixture(scope="session")
def qualified_seed(tmp_path_factory):
    """One product-written prerequisite; consumers never acquire this seed directly."""

    def build(root):
        built = build_real_risk_workspace(tmp_path_factory.mktemp("data-update-base"))
        publish_research_workspace_manifest(built.workspace, _manifest("data-update-qa"))
        _seed_foundation(built.workspace, built.panel_snapshot_hash)
        set_parallel_test_budget(built.workspace)
        copy_workspace(built.workspace, root)
        return {}

    return session_workspace(tmp_path_factory, "maintenance_qualified", build)[0]


@pytest.fixture(scope="module")
def qualified(qualified_seed, tmp_path_factory):
    # Keep the original per-module mutable boundary, backed by one session build.
    return copy_workspace(qualified_seed, tmp_path_factory.mktemp("qualified") / "workspace")
