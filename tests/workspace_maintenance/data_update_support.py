"""Helpers the local data update suites share: the seeded foundation, the
recording provider and the active panel manifest."""

from __future__ import annotations

from pathlib import Path

from alphalattice.evidence.alternative_evidence.runtime.execution import CpuBudgetStore
from alphalattice.foundation.factor_research.publication.artifacts import (
    FactorResearchArtifactStore,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.foundation.research_foundation.contracts import (
    PreResearchDeskSafeProjection,
    ResearchFoundationBinding,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from tests.researcher_methodology_surface.real_workspace import AS_OF, OBSERVED_AT
from tests.workspace_maintenance.local_data_provider import (
    NOW,
    recording_provider,
)


def set_parallel_test_budget(workspace: Path) -> None:
    """The real operator setting: two cores, with the original data and work units.

    Copies retain this setting, so four pytest workers do not each request the
    machine's automatic CPU allowance when running independent update Tasks.
    """

    budget = CpuBudgetStore(workspace / "runtime").write(
        "2", chosen_by="EXTERNAL_AUTOMATION", chosen_at=OBSERVED_AT
    )
    assert budget.cpu_budget == 2


def _seed_foundation(workspace: Path, panel_hash: str) -> None:
    """A supplied historical Foundation contract, not new Factor research.

    Its child hashes are fixture authority. Data/Feature below use real writers;
    this fixture proves immutable Foundation readback, not scientific closure.
    """
    digest = canonical_hash("fixture research evidence")
    values = ResearchFoundationBinding.model_construct(
        research_cadence="DAILY",
        feature_panel_snapshot_hash=panel_hash,
        factor_training_outcome_snapshot_hash=digest,
        factor_screening_result_hash=digest,
        factor_candidate_slate_hash=digest,
        research_desk_factor_input_hash=digest,
        ordered_factor_ids=("mom_21",),
        execution_outcome={
            "research_cadence": "DAILY",
            "snapshot_hash": digest,
            "schedule_hash": digest,
            "development_content_hash": digest,
            "sealed_holdout_content_hash": digest,
            "marker_hash": digest,
            "market_as_of": AS_OF.isoformat(),
            "data_validity_class": "CURRENT_UNIVERSE_RESEARCH_ONLY",
        },
    ).model_dump(mode="json", exclude={"foundation_hash"}, exclude_none=True)
    binding = ResearchFoundationBinding(**values, foundation_hash=canonical_hash(values))
    store = FactorResearchArtifactStore(workspace / "artifacts")
    store.publish_research_foundation(
        payload=binding.model_dump(mode="json", exclude_none=True),
        foundation_hash=binding.foundation_hash,
    )
    payload = PreResearchDeskSafeProjection.model_construct(
        research_cadence="DAILY",
        foundation_hash=binding.foundation_hash,
        foundation_marker_hash=digest,
        factor_screening_result_hash=digest,
        factor_candidate_slate_hash=digest,
        research_desk_factor_input_hash=digest,
        execution_outcome_snapshot_hash=digest,
        candidate_count=1,
        market_as_of=AS_OF.isoformat(),
        updated_at=NOW,
    ).model_dump(mode="json", exclude={"projection_hash"}, exclude_none=True)
    projection = PreResearchDeskSafeProjection(**payload, projection_hash=canonical_hash(payload))
    store.publish_pre_research_desk_projection(projection.model_dump(mode="json"))


def _provider():
    return recording_provider()


def _active_panel_manifest(workspace: Path) -> dict:
    from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
    from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository

    market = MarketDataRepository(workspace)
    panel_state = PanelStateRepository(market.database, market_data=market)
    active = panel_state.feature_panel_snapshot_for_active("us-current-index-research")
    assert active is not None
    return ArtifactResolver(workspace / "artifacts").load_feature_panel_manifest(
        str(active["manifest_uri"])
    )


def one_sector_seed(tmp_path_factory, name: str, members: tuple[str, ...]) -> Path:  # type: ignore[no-untyped-def]
    """A qualified workspace of ``members`` in one Sector: real history, catalog and Panel."""
    from alphalattice.control.product_host.composition.research_workspace import (
        publish_research_workspace_manifest,
    )
    from tests.portfolio_strategy_lab.local_web_support import _manifest
    from tests.researcher_methodology_surface.real_workspace import build_real_risk_workspace
    from tests.researcher_methodology_surface.session_workspace import (
        copy_workspace,
        session_workspace,
    )

    def build(root):  # type: ignore[no-untyped-def]
        built = build_real_risk_workspace(
            tmp_path_factory.mktemp(f"{name}-base"), symbols=members, sector_size=len(members)
        )
        publish_research_workspace_manifest(built.workspace, _manifest(f"data-update-{name}"))
        _seed_foundation(built.workspace, built.panel_snapshot_hash)
        set_parallel_test_budget(built.workspace)
        copy_workspace(built.workspace, root)
        return {}

    return session_workspace(tmp_path_factory, f"maintenance_{name}", build)[0]
