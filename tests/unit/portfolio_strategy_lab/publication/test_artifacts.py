"""Which files are the Strategy Lab's old lanes, on files alone."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas  # noqa: F401  # Arrow's NumPy conversion loads it: at collection, not in a test
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
    lane_files,
)

TEST_SOURCE = "src/alphalattice/investment/portfolio_strategy_lab/publication/artifacts.py"
LANE = np.array([0.5, -0.0, np.nan, 2.0])
DIGEST = hashlib.sha256(LANE.tobytes()).hexdigest()
OTHER = hashlib.sha256(b"x").hexdigest()


def _lab(workspace: Path) -> Path:
    """A Strategy Lab folder as its store left it before the door: a Parquet lane, a legacy packed
    lane, and files that are no lane of this store."""
    lab = workspace / "research-experiments" / "portfolio-1" / "portfolio-strategy-lab"
    lanes = lab / "development" / "authored-lanes"
    lanes.mkdir(parents=True)
    table = pa.table({"value": LANE}).replace_schema_metadata({"shape": json.dumps([4])})
    pq.write_table(table, lanes / f"{DIGEST}.parquet", use_dictionary=False)
    for relative in (
        f"calibration-arrays/{OTHER}.bin",
        f"development/paired-alpha-portfolio/objective-audits/{OTHER}.bin",
        f"public/lanes/x/{OTHER}.parquet",
        f"pending-finalization/lanes/x/{OTHER}.parquet",
        f"carried/{OTHER}.parquet",
        "development/authored-lanes/not-a-digest.parquet",
    ):
        (lab / relative).parent.mkdir(parents=True, exist_ok=True)
        (lab / relative).write_bytes(b"x")
    return lab


@pytest.mark.proves("portfolio_strategy_lab.lane_files_are_the_stores_lanes_alone")
def test_lane_files_are_the_stores_lanes_alone(tmp_path: Path) -> None:
    """A Parquet lane and a legacy packed lane are lanes; the ledger's namespaces, the carried
    lanes, the documents' category and a file not named by a digest are not."""
    lab = _lab(tmp_path)
    assert lane_files(tmp_path) == [
        lab / "calibration-arrays" / f"{OTHER}.bin",
        lab / "development" / "authored-lanes" / f"{DIGEST}.parquet",
    ]
