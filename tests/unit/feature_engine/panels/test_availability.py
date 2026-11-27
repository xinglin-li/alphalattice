"""The availability stage holds, column for column, what a row-by-row stage of its cells would."""

import json

import pyarrow as pa
import pytest

from alphalattice.foundation.feature_engine.panels import availability

TEST_SOURCE = "src/alphalattice/foundation/feature_engine/panels/availability.py"

CROSS = {"2026-07-27": "x1", "2026-07-28": "x2"}


def _cell(session: str, factor: str, **change: object) -> dict[str, object]:
    return {
        "session_date": session,
        "factor_id": factor,
        "universe_size": 120,
        "computed_count": 118,
        "coverage": 0.983,
        "sector_counts": {"tech": 9, "energy": 3},
        "winsor_lower": None,
        "winsor_upper": 2.5,
        "residual_median": 0.01,
        "residual_mad": None,
        "status": "available",
        "reason": None,
        "small_sector_warning": False,
        "small_sector_names": ("tech", "energy"),
        "panel_binding_hash": "b" * 64,
        **change,
    }


def _row_by_row(cells: list[dict[str, object]]) -> pa.Table:
    """The stage as one normalized dict per cell would build it."""
    return pa.Table.from_pylist(
        [
            {
                "cross_section_identity": CROSS[str(cell["session_date"])],
                "session_date": str(cell["session_date"]),
                "factor_id": str(cell["factor_id"]),
                "universe_size": int(str(cell["universe_size"])),
                "computed_count": int(str(cell["computed_count"])),
                "coverage": float(str(cell["coverage"])),
                "sector_counts_json": json.dumps(
                    cell["sector_counts"], sort_keys=True, separators=(",", ":")
                ),
                "winsor_lower": cell.get("winsor_lower"),
                "winsor_upper": cell.get("winsor_upper"),
                "residual_median": cell.get("residual_median"),
                "residual_mad": cell.get("residual_mad"),
                "status": str(cell["status"]),
                "reason": cell.get("reason"),
                "panel_binding_hash": str(cell["panel_binding_hash"]),
                "small_sector_warning": bool(cell.get("small_sector_warning")),
                "small_sector_names_json": json.dumps(
                    sorted(str(name) for name in cell["small_sector_names"]),
                    separators=(",", ":"),
                ),
                "materialization_receipt_hash": "r",
            }
            for cell in cells
        ]
    )


@pytest.mark.proves("feature_engine.the_availability_stage_holds_its_cells_column_for_column")
def test_the_stage_equals_a_row_by_row_stage_with_each_shared_encoding_reused() -> None:
    """Cells sharing a session's sector counts and names, a column all null and one mixed,
    counts in another key order and an unavailable cell stage exactly as row by row."""
    cells = [
        _cell("2026-07-27", "mom_21"),
        _cell("2026-07-27", "rev_5", sector_counts={"energy": 3, "tech": 9}),
        _cell("2026-07-28", "mom_21", status="unavailable", reason="coverage_below_98_percent"),
        _cell("2026-07-28", "rev_5", small_sector_names=(), winsor_upper=None),
    ]
    staged = availability._availability_stage(
        cells, cross_section_by_session=CROSS, materialization_receipt_hash="r"
    )
    expected = _row_by_row(cells)
    assert staged.schema.equals(expected.schema)
    assert staged.equals(expected)


@pytest.mark.proves("feature_engine.the_availability_stage_holds_its_cells_column_for_column")
@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        ({"panel_binding_hash": ""}, "missing its binding hash"),
        ({"small_sector_names": "tech"}, "small-sector names are invalid"),
        ({"session_date": "2026-07-29"}, "has no cross-section"),
    ],
    ids=["no-binding", "names-a-string", "unknown-session"],
)
def test_a_cell_the_stage_cannot_normalize_is_refused_by_name(
    change: dict[str, object], refusal: str
) -> None:
    """A cell with no binding, names given as one string or a session with no cross-section is
    refused as before, naming what is wrong."""
    with pytest.raises(ValueError, match=refusal):
        availability._availability_stage(
            [_cell("2026-07-27", "mom_21"), _cell("2026-07-27", "rev_5", **change)],
            cross_section_by_session=CROSS,
            materialization_receipt_hash="r",
        )
