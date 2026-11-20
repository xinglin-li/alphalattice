"""The Strategy Lab's lanes and the array door: the upgrade moves a workspace's old lane files into
the door all equal; the old files stay as the upgrade's input."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from alphalattice.control.workspace_runtime.content_store import ContentAddressedStore
from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
    PortfolioResearchArtifactStore,
    lane_upgrade_family,
)
from alphalattice.kernel.data_layer import history

TEST_SOURCE = "src/alphalattice/investment/portfolio_strategy_lab/publication/artifacts.py"
LANE = np.array([[0.5, -0.0], [np.nan, 2.0]])
PACKED = b"\x01legacy packed lane"


def _old_workspace(workspace: Path) -> dict[str, Path]:
    """A workspace as the Strategy Lab store left it before the door: a Parquet lane, a legacy
    packed lane, and files that are no lane of this store."""
    lab = workspace / "research-experiments" / "portfolio-1" / "portfolio-strategy-lab"
    old = ContentAddressedStore(lab, uri_prefix="playpen://portfolio-strategy-lab")
    digest = old.publish_columns(category="development/authored-lanes", columns={"value": LANE})
    files = {"parquet": lab / "development" / "authored-lanes" / f"{digest}.parquet"}
    packed = hashlib.sha256(PACKED).hexdigest()
    files["bin"] = lab / "calibration-arrays" / f"{packed}.bin"
    document = lab / "development" / "paired-alpha-portfolio" / "objective-audits"
    other = hashlib.sha256(b"x").hexdigest()
    for path in (
        files["bin"],
        document / f"{other}.bin",
        lab / "public" / "lanes" / "x" / f"{other}.parquet",
        lab / "carried" / f"{other}.parquet",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PACKED)
    return files


@pytest.mark.proves("portfolio_strategy_lab.lane_files_move_into_the_door_all_equal")
def test_the_upgrade_moves_a_workspaces_lane_files_into_the_door_all_equal(tmp_path: Path) -> None:
    """The family keeps each lane file with its workspace's hold, leaves the ledger's, carried and
    document files out, reads every lane back equal, and the workspace then holds no old lane."""
    workspace = tmp_path / "workspace"
    files = _old_workspace(workspace)
    family = lane_upgrade_family(lambda _workspace: "w1")
    log = history.held()
    log.record("workspace.created", ("workspace", "w1"), {"folder": str(workspace), "name": "w"})
    runner = SimpleNamespace(root=history.data_root(), log=log, workspaces=(workspace,))
    assert family.holds(workspace)
    family.step(runner)
    assert family.verify(runner)
    assert not family.holds(workspace)
    holds = log.read("SELECT root_kind, root_id FROM live_ref")
    assert [tuple(row) for row in holds] == [("workspace", "w1")] * 2
    store = PortfolioResearchArtifactStore(workspace / "research-experiments" / "portfolio-1")
    back = store.load_columns(category="any", content_hash=files["parquet"].stem)
    assert back["value"].tobytes() == LANE.tobytes()
    assert store.load_packed_bytes(category="any", content_hash=files["bin"].stem) == PACKED


@pytest.mark.proves("portfolio_strategy_lab.unmoved_workspace_reads_its_own_lane_files")
def test_a_workspace_whose_lanes_the_door_lacks_reads_its_own_files(tmp_path: Path) -> None:
    """With the switch unset, a workspace holding lane files the door lacks reads them, a lane it
    carries in place among them, until the upgrade moves them; the door holds none of them yet."""
    from alphalattice.investment.portfolio_strategy_lab.publication import artifacts

    files = _old_workspace(tmp_path)
    (tmp_path / artifacts.MANIFEST).write_text("{}", encoding="utf-8")
    carried = np.array([3.0, -0.0])
    digest = hashlib.sha256(carried.tobytes()).hexdigest()
    ContentAddressedStore(
        tmp_path / "artifacts" / "portfolio-strategy-lab", uri_prefix="playpen://x"
    ).publish_columns(category="carried", columns={"value": carried})
    store = PortfolioResearchArtifactStore(tmp_path / "research-experiments" / "portfolio-1")
    category = "development/authored-lanes"
    assert store.load_packed_bytes(category=category, content_hash=files["parquet"].stem) == (
        LANE.tobytes()
    )
    assert store.load_packed_bytes(category=category, content_hash=digest) == carried.tobytes()
    lanes = [files["parquet"].stem, digest]
    assert history.arrays_lacking(lanes) == lanes
