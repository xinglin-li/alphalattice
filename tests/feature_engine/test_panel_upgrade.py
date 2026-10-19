"""A Panel the released source wrote, continued under this one: the sub-year parts' upgrade."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from tests.feature_engine.panel_upgrade import build_days
from tests.feature_engine.panel_workspace import SESSIONS

RELEASED = "fc8212efe"
"""0.2.0's source: every Panel chunk whole-year (its Panel code is 546c772f7's)."""
ROOT = Path(__file__).resolve().parents[2]


def _released_source() -> Path:
    """The released source's src and tests, frozen once per machine."""
    dest = ROOT / "tmp" / "released-sources" / RELEASED
    if not (dest / "source.json").is_file():
        shutil.rmtree(dest, ignore_errors=True)
        archive = ["git", "-C", str(ROOT), "archive", RELEASED, "src", "tests"]
        tar = subprocess.run(archive, capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(tar)) as files:
            files.extractall(dest, filter="data")
        (dest / "source.json").write_text(json.dumps({"commit": RELEASED}), encoding="utf-8")
    return dest


def _released_days(root: Path, steps: list[list[str]]) -> list[dict]:
    """The same days built by the released source, in its own process."""
    source = _released_source()
    path = os.pathsep.join([str(source / "src"), str(source)])
    done = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("panel_upgrade.py")),
            str(root),
            json.dumps(steps),
        ],
        env={**os.environ, "PYTHONPATH": path, "ALPHALATTICE_NETWORK_DISABLED": "1"},
        cwd=source,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout.splitlines()[-1])


def test_a_store_written_whole_year_upgrades_by_splitting_its_open_year_once(tmp_path) -> None:
    """requirement (PANEL-SUB-YEAR-CHUNKS upgrade): 0.2.0's store reruns, splits its open year
    once, reruns again and then writes one quarter, its logical Panel unchanged. About 40 s: the
    released source builds its days in its own process."""
    at = next(index for index, day in enumerate(SESSIONS) if day.isoformat() >= "2019-08-01")
    one, two, three = (SESSIONS[at + n].isoformat() for n in range(3))
    released = _released_days(tmp_path / "released", [[one, "a"], [two, "b"], [three, "c"]])
    assert all(part == 0 for day in released for _year, part, _hash in day["chunks"])
    _released_days(tmp_path / "upgrade", [[one, "a"]])

    rerun, daily, again, second = build_days(
        tmp_path / "upgrade", [[one, "a"], [two, "b"], [two, "b"], [three, "c"]], reopen=True
    )
    assert rerun["snapshot_hash"] == released[0]["snapshot_hash"]  # nothing was written
    assert [chunk[:2] for chunk in daily["chunks"]] == [[2018, 0], [2019, 1], [2019, 2], [2019, 3]]
    assert daily["chunks"][0] == released[1]["chunks"][0]  # a closed year keeps its chunk
    assert daily["written"] == 3  # the open year's one split
    assert again["snapshot_hash"] == daily["snapshot_hash"]
    assert second["written"] == 1 and second["chunks"][:3] == daily["chunks"][:3]
    for upgraded, before in ((daily, released[1]), (second, released[2])):
        assert upgraded["logical_panel_hash"] == before["logical_panel_hash"] is not None
        assert upgraded["panel_content_hash"] == before["panel_content_hash"]
