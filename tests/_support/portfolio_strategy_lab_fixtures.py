"""Package fixtures for the Local Web and public Portfolio application suites."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from alphalattice.interface.local_application import client
from alphalattice.investment.alpha_research.scores.product_recipe import (
    INSTALLED_ALPHA_PRODUCT_RECIPE,
)
from alphalattice.investment.alpha_research.scores.product_replay import EVIDENCE_PACKAGE_ID
from alphalattice.kernel.identity import integrity_blake3_path
from tests.portfolio_strategy_lab.local_web_support import _manifest, _resolved, _Resolver


@pytest.fixture
def live(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[LocalPortfolioWebSession]:
    """One booted product: session, application, dispatcher and loopback socket. Its workspace
    is its own, by its id too: a Host a wider fixture serves shares this process's machine."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest(f"qa-local-web-{uuid4().hex[:12]}"),
        resolver=_Resolver(_resolved()),
        **({"clock": request.param} if hasattr(request, "param") else {}),
    )
    session.start()
    try:
        yield session
    finally:
        session.stop()


@pytest.fixture
def fake_host(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Record CLI exchanges while returning the answer selected by the test."""
    state = SimpleNamespace(answer={"status": "OK"}, raw=None, sent=[], workspaces=[])

    class Host:
        def __init__(self, workspace: Path, **kwargs: Any) -> None:
            self.workspace, self.goal = workspace, kwargs.get("goal")
            state.workspaces.append(workspace)

        def exchange(self, document: Any) -> tuple[dict[str, Any], bytes]:
            state.sent.append(dict(document))
            answer = state.answer(document) if callable(state.answer) else state.answer
            raw = state.raw if state.raw is not None else json.dumps(answer).encode("utf-8")
            return answer, raw

        def selected_url(self, *_args: Any) -> None:
            return None

        def navigation(self, *_args: Any) -> dict[str, str]:
            return {}

    monkeypatch.setattr(client, "LocalResearchClient", Host)
    return state


@pytest.fixture
def read_only_live(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LocalPortfolioWebSession]:
    """Keep the synthetic Host open only while its test owns the data root."""
    workspace = tmp_path_factory.mktemp("cli-read")
    host = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest(f"qa-cli-read-{uuid4().hex[:12]}"),
        resolver=_Resolver(_resolved()),
    )
    host.start()
    try:
        yield host
        assert host.session is not None and not host.session.task_control_registry.tasks()
    finally:
        host.stop()


@pytest.fixture
def completed_host(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LocalPortfolioWebSession]:
    """Keep one completed default Portfolio run in this test's own Host."""
    from tests.portfolio_strategy_lab.local_web_support import _run_to_completion

    workspace = tmp_path_factory.mktemp("local-web-completed")
    host = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=_manifest(f"qa-local-web-{uuid4().hex[:12]}"),
        resolver=_Resolver(_resolved()),
    )
    host.start()
    try:
        _run_to_completion(host)
        assert host.session is not None
        tasks = host.session.task_control_registry.tasks()
        assert len(tasks) == 1, [(t.task_kind, t.lifecycle, str(t.task_id)) for t in tasks]
        yield host
    finally:
        host.stop()


_SCORE_FIXTURE_RECIPE = INSTALLED_ALPHA_PRODUCT_RECIPE
_SCORE_FIXTURE_SESSION = date(2024, 8, 1)
_SCORE_FIXTURE_NAMES = ("A", "B", "C", "D")
_SCORE_FIXTURE_VINTAGES = ("2024-07", "2024-04", "2024-01", "2023-10")


def _write_package(
    root: Path,
    *,
    scores_by: dict[tuple[str, int], list[float]] | None = None,
    sessions: tuple[date, ...] = (_SCORE_FIXTURE_SESSION,),
    names: tuple[str, ...] = _SCORE_FIXTURE_NAMES,
) -> Path:
    """Write a minimal package with the same shape as the admitted one."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "prediction-axes").mkdir(exist_ok=True)
    (root / "model-predictions").mkdir(exist_ok=True)
    rows = [(session, name) for session in sessions for name in names]
    session_months = tuple(dict.fromkeys(session.strftime("%Y-%m") for session in sessions))
    transform_session_axis_hash = canonical_hash(
        [session.isoformat() for session in dict.fromkeys(sessions)]
    )
    model_lineage: dict[str, object] = {}

    for vintage in _SCORE_FIXTURE_VINTAGES:
        axis_npz = root / "prediction-axes" / f"{vintage}.npz"
        np.savez(
            axis_npz,
            row_sessions=np.array([value for value, _ in rows], dtype="datetime64[D]"),
            row_listing_ids=np.array([name for _, name in rows], dtype=np.dtype((np.str_, 36))),
        )
        axis_sha = integrity_blake3_path(axis_npz)
        (root / "prediction-axes" / f"{vintage}.json").write_text(
            json.dumps(
                {
                    "kind": "IW184QuarterlyPredictionAxis",
                    "quarter_start": vintage,
                    "artifact_digest": axis_sha,
                    "row_count": len(rows),
                    "transform_session_axis_hash": transform_session_axis_hash,
                    "operational_months": list(session_months),
                    "prediction_months": list(session_months),
                }
            ),
            encoding="utf-8",
        )
        vintage_dir = root / "model-predictions" / vintage
        vintage_dir.mkdir(exist_ok=True)
        seed_lineage: dict[str, object] = {}
        for seed in _SCORE_FIXTURE_RECIPE.seeds:
            values = (
                scores_by[(vintage, seed)]
                if scores_by and (vintage, seed) in scores_by
                else [float(index) for index in range(len(rows))]
            )
            payload = vintage_dir / f"seed-{seed}.npz"
            identity = {
                "kind": "IW184Fixed1260QuarterlySeedModel",
                "candidate": _SCORE_FIXTURE_RECIPE.feature_axis_id,
                "candidate_axis_hash": _SCORE_FIXTURE_RECIPE.feature_axis_hash,
                "feature_view_id": "SPARSE_SESSION_AMPLITUDE",
                "annual_root_hash": "blake3:" + "3" * 64,
                "source_resolution_hash": "blake3:" + "4" * 64,
                "recipe_hash": canonical_hash({"seed": seed}),
                "seed": seed,
                "quarter_start": vintage,
                "training_window_sessions": _SCORE_FIXTURE_RECIPE.training_window_sessions,
                "training_first": "2019-06-26",
                "training_last": "2024-06-27",
                "training_session_axis_hash": "blake3:" + "5" * 64,
                "purge_session": "2024-06-28",
                "purge_session_count": _SCORE_FIXTURE_RECIPE.purge_sessions,
                "prediction_months": list(session_months),
                "operational_months": list(session_months),
                "transform_session_axis_hash": transform_session_axis_hash,
                "score_aggregation": _SCORE_FIXTURE_RECIPE.score_aggregation,
            }
            identity_hash = canonical_hash(identity)
            np.savez(
                payload,
                scores=np.asarray(values, dtype=np.float64),
                identity_hash=np.asarray(identity_hash),
            )
            artifact_digest = integrity_blake3_path(payload)
            estimator_content_hash = canonical_hash({"estimator": seed, "vintage": vintage})
            (vintage_dir / f"seed-{seed}.json").write_text(
                json.dumps(
                    {
                        **identity,
                        "ordered_feature_ids_hash": _SCORE_FIXTURE_RECIPE.feature_axis_hash,
                        "identity_hash": identity_hash,
                        "estimator_content_hash": estimator_content_hash,
                        "artifact_digest": artifact_digest,
                        "prediction_axis_digest": axis_sha,
                        "prediction_row_count": len(rows),
                        "training_row_count": 100,
                    }
                ),
                encoding="utf-8",
            )
            seed_lineage[str(seed)] = {
                "artifact_digest": artifact_digest,
                "estimator_content_hash": estimator_content_hash,
                "identity_hash": identity_hash,
                "training_first": identity["training_first"],
                "training_last": identity["training_last"],
                "training_row_count": 100,
            }
        model_lineage[vintage] = {
            "prediction_axis_digest": axis_sha,
            "row_count": len(rows),
            "seeds": seed_lineage,
        }
    daily = root / "iw184-quarterly-5y-ensemble-daily.parquet"
    daily.write_bytes(b"synthetic daily score evidence")
    (root / f"{EVIDENCE_PACKAGE_ID}.json").write_text(
        json.dumps(
            {
                "kind": "IW184Fixed1260QuarterlySeed3Vintage4WeightSensitivityDiagnostic",
                "disposition": _SCORE_FIXTURE_RECIPE.evidence_disposition,
                "schema_version": 2,
                "recipe_hash": _SCORE_FIXTURE_RECIPE.recipe_hash,
                "definition": {
                    "feature_axis": _SCORE_FIXTURE_RECIPE.feature_axis_id,
                    "feature_count": _SCORE_FIXTURE_RECIPE.feature_count,
                    "live_model_count": _SCORE_FIXTURE_RECIPE.live_model_count,
                    "training_window_sessions": _SCORE_FIXTURE_RECIPE.training_window_sessions,
                    "purge_session_count": _SCORE_FIXTURE_RECIPE.purge_sessions,
                    "seeds": list(_SCORE_FIXTURE_RECIPE.seeds),
                    "vintage_count": _SCORE_FIXTURE_RECIPE.vintage_count,
                    "vintage_weights": list(_SCORE_FIXTURE_RECIPE.vintage_weights),
                    "seed_aggregation": _SCORE_FIXTURE_RECIPE.seed_aggregation,
                    "score_aggregation": _SCORE_FIXTURE_RECIPE.score_aggregation,
                },
                "daily_evidence": {
                    "name": "iw184-quarterly-5y-ensemble-daily.parquet",
                    "row_count": _SCORE_FIXTURE_RECIPE.evidence_support_session_count,
                    "digest": integrity_blake3_path(daily),
                },
                "model_lineage": model_lineage,
            }
        ),
        encoding="utf-8",
    )
    return root
