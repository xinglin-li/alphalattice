"""The deterministic Factor program run once over synthetic tables, and the
sealed checkpoint the curation cases of this package curate.

Spelled once because a dossier built from the same checkpoint under a
different binding is a different dossier, which is exactly what the Host
curation owner refuses.
Test support beside its owner; nothing here is product authority or evidence.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import cache

import pyarrow as pa

from alphalattice.foundation.causal_outcomes.execution.methods import (
    ExecutionOutcomeMethodSeal,
)
from alphalattice.foundation.factor_research.evaluation.oos_evidence import (
    build_factor_evidence_policy,
)
from alphalattice.foundation.factor_research.evaluation.redundancy import (
    build_factor_redundancy_policy,
)
from alphalattice.foundation.factor_research.inputs.execution_target import (
    build_factor_target_policy,
)
from alphalattice.foundation.factor_research.programs.program import (
    FactorResearchDeterministicEvidence,
    FactorResearchDeterministicResult,
    build_factor_research_program_spec,
    execute_factor_research_program,
    seal_factor_research_deterministic_evidence,
)
from alphalattice.foundation.factor_research.programs.walk_forward import (
    build_factor_walk_forward_policy,
)
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReadRequest
from tests.factor_research_pipeline_correctness.walk_forward_source import (
    _manifest,
    _source_table,
)


class _FeatureReader:
    def __init__(self, sessions: tuple[date, ...], table: pa.Table) -> None:
        self._sessions = sessions
        self._table = table
        self.requests: list[FeaturePanelReadRequest] = []

    def available_sessions(self, manifest_ref: str) -> tuple[date, ...]:
        assert manifest_ref == "playpen://feature-panel/manifest"
        return self._sessions

    def batches(self, request: FeaturePanelReadRequest) -> Iterator[pa.RecordBatch]:
        self.requests.append(request)
        assert request.exact_sessions is not None
        requested = set(request.exact_sessions)
        rows = [row for row in self._table.to_pylist() if row["session_date"] in requested]
        yield from pa.Table.from_pylist(rows).to_batches(max_chunksize=512)


class _OutcomeReader:
    def __init__(self, manifest, table: pa.Table) -> None:
        self._manifest = manifest
        self._table = table
        self.requests: list[tuple[date, ...]] = []

    def load_manifest(self, snapshot_hash: str):
        assert snapshot_hash == self._manifest.snapshot_hash
        return self._manifest

    def read_development_sessions(self, manifest_ref: str, sessions: tuple[date, ...]) -> pa.Table:
        assert manifest_ref == "playpen://outcomes/manifest"
        self.requests.append(sessions)
        requested = set(sessions)
        return pa.Table.from_pylist(
            [row for row in self._table.to_pylist() if row["formation_session"] in requested]
        )

    def resolve_method_seal(self, snapshot_hash: str) -> ExecutionOutcomeMethodSeal:
        """Answer as a snapshot published before the outcome method seam existed.

        The program asks every outcome reader this, and a stub that could not
        answer failed as an ``AttributeError`` from inside the program rather
        than as anything about the case. ``LEGACY_READBACK_ONLY`` is the honest
        answer for a hand-built manifest with no seal marker anywhere: the rows
        are readable and the method authority is genuinely absent, which is
        exactly the state the program's legacy route exists to handle.

        Deliberately not a fabricated ``METHOD_BOUND`` seal. Inventing a binding
        would make this case assert that the program threads a method identity it
        was never handed.
        """

        assert snapshot_hash == self._manifest.snapshot_hash
        return ExecutionOutcomeMethodSeal(disposition="LEGACY_READBACK_ONLY")


CASE_REVIEW_BINDING_HASH = "b" * 64


@dataclass(frozen=True, slots=True)
class _CaseProgramRun:
    """One executed deterministic program plus the readers that served it."""

    result: FactorResearchDeterministicResult
    feature_reader: _FeatureReader
    outcome_reader: _OutcomeReader
    sessions: tuple[date, ...]
    listings: tuple[str, ...]
    factor_ids: tuple[str, ...]


@cache
def _execute_case_program() -> _CaseProgramRun:
    """Run the deterministic Factor program once over synthetic tables.

    Cached because it is the expensive part of this directory and two cases now
    need the same checkpoint. Building a second, hand-written checkpoint for the
    curation case would let the two drift: the whole point of curating a
    checkpoint is that the checkpoint is the authority.
    """

    sessions = tuple(date(2026, 1, 2) + timedelta(days=index) for index in range(14))
    listings = tuple(f"listing-{index:03d}" for index in range(120))
    factor_ids = ("factor_a", "factor_b")
    manifest = _manifest(listings)
    feature_rows: list[dict[str, object]] = []
    for session_index, session in enumerate(sessions):
        for listing_index, listing_id in enumerate(listings):
            feature_rows.append(
                {
                    "session_date": session,
                    "listing_id": listing_id,
                    "factor_a": float(listing_index),
                    "factor_b": float((listing_index * 37 + session_index * 11) % 120),
                }
            )
    feature_reader = _FeatureReader(sessions, pa.Table.from_pylist(feature_rows))
    outcome_reader = _OutcomeReader(manifest, _source_table(sessions, listings))
    program = build_factor_research_program_spec(
        feature_panel_snapshot_hash="9" * 64,
        feature_panel_manifest_ref="playpen://feature-panel/manifest",
        causal_outcome_snapshot_hash=manifest.snapshot_hash,
        causal_outcome_manifest_ref="playpen://outcomes/manifest",
        factor_ids=factor_ids,
        frozen_at=datetime(2027, 1, 1, tzinfo=UTC),
        target_policy=build_factor_target_policy(),
        walk_forward_policy=build_factor_walk_forward_policy(
            train_sessions=4,
            validation_sessions=6,
            step_sessions=6,
            sealed_holdout_sessions=2,
            minimum_folds=1,
        ),
        evidence_policy=build_factor_evidence_policy(minimum_cross_section_observations=100),
        redundancy_policy=build_factor_redundancy_policy(
            minimum_common_listings=100,
            minimum_formal_periods=3,
            distance_cut=0.1,
        ),
    )
    panel_manifest = {
        "snapshot_hash": "9" * 64,
        "safe_summary": {"factor_catalog_summary": {factor_id: {} for factor_id in factor_ids}},
    }

    result = execute_factor_research_program(
        program=program,
        panel_manifest=panel_manifest,
        feature_reader=feature_reader,
        outcome_reader=outcome_reader,
    )
    return _CaseProgramRun(
        result=result,
        feature_reader=feature_reader,
        outcome_reader=outcome_reader,
        sessions=sessions,
        listings=listings,
        factor_ids=factor_ids,
    )


@cache
def build_case_deterministic_checkpoint() -> FactorResearchDeterministicEvidence:
    """The sealed checkpoint the curation cases in this directory curate."""

    return seal_factor_research_deterministic_evidence(
        result=_execute_case_program().result,
        execution_binding_hash="a" * 64,
    )
