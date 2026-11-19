"""The array door on a real data root and log: a digest names one array, a lane is held once per
machine by each workspace citing it, and a workspace copy carries its lanes onto its own log."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import arrays, history
from alphalattice.kernel.data_layer.store import StorePort

TEST_SOURCE = "src/alphalattice/kernel/data_layer/arrays.py"


def _workspace(log: Any, name: str) -> str:
    log.record("workspace.created", ("workspace", name), {"folder": f"/{name}", "name": name})
    return name


@pytest.mark.proves("data_layer.array_digest_names_one_array")
def test_a_digest_kept_again_with_another_dtype_is_refused_by_name() -> None:
    """Equal bytes are one array only with one dtype and shape: the log's profile refuses more."""
    lane = np.array([1.0, 2.0])
    history.keep_array({"value": lane})
    with pytest.raises(DataError) as refused:
        history.keep_array({"value": lane.view("<i8")})
    assert refused.value.failure.code == "data_layer.array_digest_reused"
    assert history.read_array(arrays.digest({"value": lane}), packed=True) == lane.tobytes()


@pytest.mark.proves("data_layer.array_door_holds_one_body_per_machine")
def test_two_workspaces_citing_one_lane_hold_one_body() -> None:
    """Each workspace holds the lane by one live_ref however often it keeps it; the body is one."""
    log, root = history.held(), history.data_root()
    lane = {"value": np.arange(5.0)}
    for workspace in (_workspace(log, "w1"), _workspace(log, "w2"), "w1", "w2"):
        arrays.keep(root, log, lane, workspace=workspace)
    holds = log.read("SELECT root_kind, root_id FROM live_ref ORDER BY root_id")
    assert [tuple(row) for row in holds] == [("workspace", "w1"), ("workspace", "w2")]
    assert [path.name for path in (root / "blobs").iterdir()] == [arrays.describe(lane).artifact_id]
    # An Alpha part's typed relation and its body share the same workspace holders.
    workspaces = ("w1", "w2")
    for workspace in (*workspaces, *workspaces):
        digest = history.keep_alpha_part(
            "a" * 64, "layout", b"body", upgrade_owner=(log, root, workspace)
        )
    link = history._alpha_part_link(log.read, "a" * 64, "layout")
    assert link is not None and digest == link[2]
    for artifact in (link[0], link[1]):
        rows = log.read(
            "SELECT root_id FROM live_ref WHERE artifact_id = ?"
            " AND root_kind = 'workspace' ORDER BY root_id",
            (artifact,),
        )
        assert [row[0] for row in rows] == ["w1", "w2"]
    assert len(list((root / "blobs").iterdir())) == 2


@pytest.mark.proves("data_layer.workspace_copy_carries_its_lanes")
def test_a_workspace_copy_reads_the_lanes_it_carries_on_its_own_root(tmp_path: Path) -> None:
    """A copy carries its source's lanes in the old form; its Host, on a root of its own, moves
    them onto its log when it opens, held by the copy's workspace."""
    source, copy = tmp_path / "source", tmp_path / "copy"
    source.mkdir()
    log, root = history.held(), history.data_root()
    held = history.workspace_on_log(source, "source")
    lane = {"value": np.array([np.nan, -0.0])}
    columns = arrays.keep(root, log, lane, workspace=held)
    packed = arrays.keep_packed(root, log, b"legacy", workspace=held)
    alpha = history.keep_alpha_part(
        "a" * 64, "alpha-layout", b"alpha body", upgrade_owner=(log, root, held)
    )
    assert history.carry_arrays(source, copy) == 3
    with history.own_root(tmp_path / "copy-root"):
        history.open_history("copy", "copy", copy)
        try:
            assert history.read_array(columns, packed=True) == lane["value"].tobytes()
            assert history.read_array(packed, packed=True) == b"legacy"
            assert history.alpha_part_body_digest("a" * 64, "alpha-layout") == alpha
            assert history.read_alpha_part("a" * 64, "alpha-layout") == b"alpha body"
            holds = history._open["log"].read("SELECT DISTINCT root_id FROM live_ref")
            assert [row[0] for row in holds] == ["copy"]
        finally:
            history.close_history()


class _Refused(Exception):
    """What ends the unit below, rolling its events back."""


def _kept_then_refused(log: Any, root: Path, lane: dict[str, Any]) -> None:
    with log.unit():
        arrays.keep(root, log, lane, workspace=None, durable=False)
        raise _Refused


def _rolled_back(log: Any, root: Path, lanes: list[dict[str, Any]], monkeypatch: Any) -> None:
    with pytest.raises(_Refused):
        _kept_then_refused(log, root, lanes[0])


def _first_rename_refused(
    log: Any, root: Path, lanes: list[dict[str, Any]], monkeypatch: Any
) -> None:
    for lane in lanes:
        arrays.keep(root, log, lane, workspace=None, durable=False)
    first, rename = arrays.describe(lanes[0]).artifact_id, os.replace

    def refuse_first(source: Any, target: Any) -> None:
        if Path(target).name == first:
            raise PermissionError(f"held: {target}")
        rename(source, target)

    monkeypatch.setattr(os, "replace", refuse_first)
    with pytest.raises(PermissionError):
        log.flush()
    monkeypatch.setattr(os, "replace", rename)


@pytest.mark.proves("data_layer.array_bodies_settle_after_a_refused_placement_or_rollback")
@pytest.mark.parametrize(
    ("happen", "placed", "kept"),
    [
        pytest.param(_rolled_back, False, [], id="a unit rolled back"),
        pytest.param(_first_rename_refused, True, [0, 1], id="a commit's first rename refused"),
    ],
)
def test_a_staged_body_left_in_this_process_is_settled_at_its_next_start(
    monkeypatch: Any, happen: Any, placed: bool, kept: list[int]
) -> None:
    """A body whose event rolled back, or whose rename after its commit was refused, is left to the
    next start in this very process: collected without its profile, placed with it, and every later
    lane of the commit placed meanwhile."""
    log, root = history.held(), history.data_root()
    lanes = [{"value": np.array([1.0, -0.0])}, {"value": np.array([2.0])}]
    happen(log, root, lanes, monkeypatch)
    port = StorePort.open(root, log)
    port.close()
    first = arrays.describe(lanes[0]).artifact_id
    assert port.started["placed"] == ([f"blobs/{first}"] if placed else [])
    assert len(port.started["collected"]) == int(not placed)
    held = sorted(path.name for path in (root / "blobs").iterdir())
    assert held == sorted(arrays.describe(lanes[i]).artifact_id for i in kept)
    for i in kept:
        assert (
            history.read_array(arrays.digest(lanes[i]), packed=True) == lanes[i]["value"].tobytes()
        )
