"""The Portfolio benchmark value surface and the source authority it admits."""

from __future__ import annotations

import shutil
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphalattice.capabilities.portfolio_inputs import benchmark as benchmark_owner
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root

PLAYPEN_ROOT = resolve_playpen_root(Path(__file__))
PLAYPEN_SRC = PLAYPEN_ROOT / "src"


def test_benchmark_value_surface_refuses_a_source_authority_substitution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A current Program cannot reuse SPY values from another raw/action lineage."""

    sessions = (date(2024, 1, 2),)
    parent = SimpleNamespace(revision_sha256="1" * 64)
    reference = SimpleNamespace(manifest=parent, listing_id="SPY")
    rows = [
        {"session_date": date(2024, 1, 2), "open_split_adjusted": 100.0, "cash_dividend": 0.0},
        {"session_date": date(2024, 1, 3), "open_split_adjusted": 101.0, "cash_dividend": 0.0},
        {"session_date": date(2024, 1, 4), "open_split_adjusted": 102.0, "cash_dividend": 0.0},
    ]
    market = SimpleNamespace(
        current_quality_filtered_research_manifest=lambda **kwargs: parent,
    )
    feature = SimpleNamespace(
        projected_feature_frame=lambda *args, **kwargs: (rows, "2" * 64, "3" * 64),
    )
    monkeypatch.setattr(benchmark_owner, "MarketDataRepository", lambda workspace: market)
    monkeypatch.setattr(
        benchmark_owner, "FeatureStateRepository", lambda workspace, market_data: feature
    )
    monkeypatch.setattr(benchmark_owner.MarketReference, "spy", lambda manifest: reference)
    expected = benchmark_owner.PortfolioBenchmarkSourceAuthority.create(
        formation_sessions=sessions,
        universe_manifest_revision="1" * 64,
        raw_input_hash="2" * 64,
        action_set_hash="4" * 64,
    )
    with pytest.raises(
        benchmark_owner.PortfolioBenchmarkBoundaryError,
        match="source_authority_not_admitted",
    ):
        benchmark_owner.build_portfolio_benchmark_surface(
            workspace=tmp_path,
            formation_sessions=sessions,
            expected_source_authority=expected,
        )

    # The upgrade moves the 0.2.7 files; after it the same reads come from the catalog.
    import json

    import numpy as np

    from alphalattice.control.workspace_runtime.content_store import ContentAddressedStoreError
    from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
        LEDGER_LEGACY_OPTIONAL_FIELDS,
        PortfolioExecutionLedger,
    )
    from alphalattice.investment.portfolio_strategy_lab.publication.portfolio_ledger import (
        PortfolioLedgerStore,
    )
    from alphalattice.investment.portfolio_strategy_lab.reporting.risk import (
        PortfolioRiskAttributionFacts,
    )
    from alphalattice.kernel.data_layer import history
    from alphalattice.kernel.data_layer.upgrade import UpgradeRunner
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    fields = {
        "formation_sessions": (*sessions, date(2024, 1, 3)),
        "simple_returns": (0.1, -0.0),
        "log_returns": (float(np.log1p(0.1)), -0.0),
        "universe_manifest_revision": "1" * 64,
        "raw_input_hash": "2" * 64,
        "action_set_hash": "3" * 64,
    }
    cls = benchmark_owner.PortfolioBenchmarkSurface
    draft = cls.model_construct(**fields, surface_hash="")
    surface = cls(
        **fields,
        surface_hash=canonical_hash(draft.model_dump(mode="json", exclude={"surface_hash"})),
    )

    from alphalattice.control.product_host.composition.research_workspace import (
        initialize_research_workspace,
        named_workspace_id,
    )

    workspace, root = tmp_path / "workspace", tmp_path / "machine-root"
    initialize_research_workspace(workspace)
    artifacts = workspace / "artifacts"
    old = artifacts / "portfolio-benchmark" / "surfaces" / f"{surface.surface_hash}.json"
    old.parent.mkdir(parents=True)
    old.write_text(
        json.dumps(surface.model_dump(mode="json"), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    risk_fields = {
        "formation_session": sessions[0],
        "surface_hash": "a" * 64,
        "risk_projection_hash": "b" * 64,
        "ordered_factor_ids": ("factor",),
        "industry_exposure": (0.0,),
        "factor_variance_contribution": (0.0,),
        "systematic_variance": 0.0,
        "idiosyncratic_variance": 0.0,
        "total_predicted_volatility": 0.0,
        "classification_authority": "fixture",
    }
    risk_draft = PortfolioRiskAttributionFacts.model_construct(**risk_fields, facts_hash="")
    risk = PortfolioRiskAttributionFacts(
        **risk_fields,
        facts_hash=canonical_hash(risk_draft.model_dump(mode="json", exclude={"facts_hash"})),
    )
    ledger_fields = {
        "program_hash": "c" * 64,
        "formation_sessions": sessions,
        "ordered_listing_ids": ("listing",),
        "executed_weights_hash": "d" * 64,
        "executed_weights_shape": (1, 1),
        "pre_cap_weights_hash": "e" * 64,
        "pre_cap_weights_shape": (1, 1),
        "final_weights_hash": "f" * 64,
        "gross_simple_returns": (-0.0,),
        "one_way_turnovers": (0.0,),
        "decision_modes": ("hold",),
        "consumed_score_projection_hashes": (),
        "consumed_risk_projection_hashes": (),
        "aggregate_cap_binding_counts": (0,),
        "missed_execution_count": 0,
        "anchor_simple_returns": (0.1,),
        "risk_facts": (risk,),
        "median_holding_adv20_by_formation": (None,),
    }
    ledger_draft = PortfolioExecutionLedger.model_construct(**ledger_fields, ledger_hash="")
    ledger_identity = ledger_draft.model_dump(mode="json", exclude={"ledger_hash"})
    for field in LEDGER_LEGACY_OPTIONAL_FIELDS:
        if getattr(ledger_draft, field) is None:
            ledger_identity.pop(field)
    ledger = PortfolioExecutionLedger(**ledger_fields, ledger_hash=canonical_hash(ledger_identity))
    ledger_store = PortfolioLedgerStore(artifacts)
    ledger_store.content.publish_model(
        category="execution-ledgers", value=ledger, identity_field="ledger_hash"
    )
    store = benchmark_owner.PortfolioBenchmarkStore(artifacts)
    cases = [
        ("benchmark surface", lambda: store.load(surface.surface_hash)),
        ("execution ledger", lambda: ledger_store.load_execution(ledger.ledger_hash)),
    ]
    # A 0.2.7 workspace's files are read only by its upgrade, never before it.
    with pytest.raises(benchmark_owner.PortfolioBenchmarkBoundaryError):
        store.load(surface.surface_hash)
    before = [("benchmark surface", surface), ("execution ledger", ledger)]
    probe_workspace = tmp_path / "catalog-publication"
    probe_workspace.mkdir()
    probe_root = tmp_path / "catalog-publication-root"
    monkeypatch.setenv("ALPHALATTICE_DATA_ROOT", str(probe_root))
    history.open_history("portfolio-publication", "portfolio", probe_workspace, probe_root)
    try:
        probe = benchmark_owner.PortfolioBenchmarkStore(probe_workspace / "artifacts")
        probe.publish(surface)
        assert probe.load(surface.surface_hash) == surface
    finally:
        history.close_history()

    from alphalattice.kernel.data.errors import DataError

    unavailable = tmp_path / "unavailable-artifacts"
    for publish in (
        lambda: benchmark_owner.PortfolioBenchmarkStore(unavailable).publish(surface),
        lambda: PortfolioLedgerStore(unavailable).publish_execution(ledger),
    ):
        with pytest.raises(DataError) as refusal:
            publish()
        assert refusal.value.failure.code == "data_layer.history_unavailable"
    assert not list(unavailable.rglob("*.json"))

    from alphalattice.control.product_host.composition import first_start

    monkeypatch.setenv("ALPHALATTICE_DATA_ROOT", str(root))
    workspace_id = named_workspace_id(workspace)
    assert workspace_id is not None
    history.open_history(workspace_id, "portfolio", workspace, root)
    try:
        names = ["portfolio-benchmark", "portfolio-results"]
        registered = (make() for make in first_start.FAMILIES)
        families = tuple(family for family in registered if family.name in names)
        assert [family.name for family in families] == names
        family = families[-1]
        empty = tmp_path / "no-portfolio-results"
        empty.mkdir()
        assert all(
            candidate.holds(workspace) and not candidate.holds(empty) for candidate in families
        )
        upgrade = UpgradeRunner(
            root,
            history.holding(),
            (workspace,),
            families,
            lambda *_: {"fixture": "verified"},
        )
        assert upgrade.run() is True
        assert all(not candidate.holds(workspace) for candidate in families)
        copy = tmp_path / "portfolio-copy"
        initialize_research_workspace(copy)
        copy_id = named_workspace_id(copy)
        assert copy_id is not None and copy_id != workspace_id
        copy_benchmark = copy / old.relative_to(workspace)
        copy_benchmark.parent.mkdir(parents=True)
        copy_benchmark.write_bytes(old.read_bytes())
        copy_ledger = PortfolioLedgerStore(copy / "artifacts")
        copy_ledger.content.publish_model(
            category="execution-ledgers", value=ledger, identity_field="ledger_hash"
        )
        assert family.holds(copy)  # rows present, local descriptor absent
        assert families[0].holds(copy)  # rows present, workspace reference absent
        assert UpgradeRunner(
            root,
            history.holding(),
            (copy,),
            families,
            lambda *_: {"fixture": "verified"},
            subject=f"0.2.8/{copy_id}",
        ).run()
        assert all(not candidate.holds(copy) for candidate in families)
        log, port = history.holding(), history.store()
        horizon = log.clock() + timedelta(days=8) // timedelta(microseconds=1)
        with monkeypatch.context() as clock:
            clock.setattr(log, "clock", lambda: horizon)
            assert port.collect_unheld() == []
        assert copy_ledger.load_execution(ledger.ledger_hash) == ledger
        assert [(name, read()) for name, read in cases] == before
        assert store.load(surface.surface_hash).simple_returns[-1].hex() == (-0.0).hex()
        assert store.load(surface.surface_hash).log_returns[-1].hex() == (-0.0).hex()
        assert (
            ledger_store.load_execution(ledger.ledger_hash).gross_simple_returns[0].hex()
            == (-0.0).hex()
        )
        assert ledger_store.load_execution(
            ledger.ledger_hash
        ).median_holding_adv20_by_formation == (None,)
        missing = "f" * 64
        for read in (
            lambda: ledger_store.load_execution(missing),
            lambda: ledger_store.load_report(missing),
            lambda: ledger_store.load_execution_and_report(ledger.ledger_hash, missing),
        ):
            with pytest.raises(
                ContentAddressedStoreError, match=f"content_store.artifact_missing:{missing}"
            ):
                read()
        store.publish(surface)
        assert old.is_file()
    finally:
        history.close_history()

    # A copy of the upgraded workspace served on a fresh data root, as a golden's copy is, owes
    # no upgrade once the copy's Portfolio results are carried into that root.
    from tests.pytest_environment import with_its_outcomes

    served, fresh = tmp_path / "served-copy", tmp_path / "served-copy-root"
    shutil.copytree(workspace, served)
    with_its_outcomes(served, root, fresh)
    with history.own_root(fresh):
        assert [candidate.holds(served) for candidate in families] == [False, False]
    # A benchmark only in its 0.2.7 file, which the source's catalog holds for no copy of this
    # workspace, is left to the upgrade: nothing is carried and the copy's root is not written.
    legacy, legacy_root = tmp_path / "legacy-copy", tmp_path / "legacy-copy-root"
    initialize_research_workspace(legacy)
    (legacy / old.relative_to(workspace)).parent.mkdir(parents=True)
    (legacy / old.relative_to(workspace)).write_bytes(old.read_bytes())
    with_its_outcomes(legacy, root, legacy_root)
    assert not legacy_root.exists()
