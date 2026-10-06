"""Program-level fixture for the deterministic one-session Factor Research path."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from tests.factor_research_pipeline_correctness.case_program import (
    _execute_case_program,
    _FeatureReader,
    _OutcomeReader,
)

PLAYPEN_ROOT = Path(__file__).resolve().parents[2]


"""The Desk method binding this case's dossiers are built from.

Named once because two cases now build a dossier from the same checkpoint, and a
dossier built under a different binding is a different dossier -- which is
exactly what the Host curation owner refuses.
"""


def test_program_uses_dated_members_not_all_covered_outcomes_for_sample_coverage() -> None:
    from alphalattice.foundation.factor_research.programs.program import (
        FactorResearchProgramBoundaryError,
        execute_factor_research_program,
    )

    case = _execute_case_program()
    boundary = case.sessions[8]
    early_members = set(case.listings[:100])
    rows = [
        {**row, "unused_volume_feature": None}
        for row in case.feature_reader._table.to_pylist()
        if row["session_date"] >= boundary or row["listing_id"] in early_members
    ]
    manifest = {
        "snapshot_hash": case.result.program.feature_panel_snapshot_hash,
        "safe_summary": {
            "factor_catalog_summary": {name: {} for name in case.factor_ids},
            "membership": {
                "epochs": [
                    {
                        "first_session": case.sessions[0].isoformat(),
                        "last_session": case.sessions[7].isoformat(),
                        "member_count": 100,
                    },
                    {
                        "first_session": boundary.isoformat(),
                        "last_session": case.sessions[-1].isoformat(),
                        "member_count": 120,
                    },
                ]
            },
        },
    }
    outcome = _OutcomeReader(case.outcome_reader._manifest, case.outcome_reader._table)
    result = execute_factor_research_program(
        program=case.result.program,
        panel_manifest=manifest,
        feature_reader=_FeatureReader(case.sessions, pa.Table.from_pylist(rows)),
        outcome_reader=outcome,
    )
    assert result.feature_row_count < result.target_row_count
    assert all(item.validation_pair_coverage_mean == 1.0 for item in result.evidence_report.items)
    with pytest.raises(FactorResearchProgramBoundaryError, match="program_member_rows_missing"):
        execute_factor_research_program(
            program=case.result.program,
            panel_manifest=manifest,
            feature_reader=_FeatureReader(case.sessions, pa.Table.from_pylist(rows[1:])),
            outcome_reader=outcome,
        )
