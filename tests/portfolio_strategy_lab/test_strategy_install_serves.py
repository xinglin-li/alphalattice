"""An installed research strategy is served by the running Host, with no restart."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from alphalattice.control.product_host.composition import research_workspace
from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.data_preparation import research_strategy
from alphalattice.control.product_host.data_preparation.research_strategy import (
    ARTIFACT_KEY,
    CATEGORY,
)
from alphalattice.interface.local_application.cli import main
from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (
    BROAD_FEATURE_PACKAGE,
    PRODUCT_EVIDENCE_ROOT_KEY,
    install_frozen_strategies,
)


def _cli(live: LocalPortfolioWebSession, capsys, *arguments: str) -> dict:  # type: ignore[no-untyped-def,type-arg]
    main(["--workspace", str(live.workspace), "--view", "full", *arguments], serve=lambda _: 99)
    return json.loads(capsys.readouterr().out)


def test_an_installed_strategy_is_served_by_the_running_host_without_a_restart(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """regression (an offline first-use rehearsal): a Host started with nothing installed serves a
    strategy installed through it, its controls and plan, with no restart. About 13 s: one real
    Host composed with the shipped strategy catalog."""
    shipped = install_frozen_strategies(artifacts={PRODUCT_EVIDENCE_ROOT_KEY: tmp_path})
    # The installed artifact's registry reads name the shipped package: a real preparation
    # runs whole Alpha and Risk studies.
    for module in (research_strategy, research_workspace):
        monkeypatch.setattr(module, "install_frozen_strategies", lambda artifacts: shipped)
    live = LocalPortfolioWebSession.from_workspace(tmp_path / "workspace")
    package_id = BROAD_FEATURE_PACKAGE.strategy_id
    with live:
        assert not live.operations.installed()
        before = _cli(live, capsys, "strategy-book", "controls", "--package", package_id)
        assert before["failure_code"] == "research_workspace.strategy_not_installed", before
        relative = f"artifacts/research-strategy-inputs/p/portfolio-strategy-lab/{CATEGORY}/a.json"
        (live.workspace / relative).parent.mkdir(parents=True)
        (live.workspace / relative).write_text("{}", encoding="utf-8")
        binding = research_workspace.ResearchWorkspaceArtifact(
            artifact_key=ARTIFACT_KEY, relative_path=relative
        )

        def pointed(now: research_workspace.ResearchWorkspaceManifest):  # type: ignore[no-untyped-def]
            values = {
                name: getattr(now, name)
                for name in type(now).model_fields
                if name not in {"kind", "manifest_schema", "manifest_hash"}
            }
            values.update(
                strategy_artifacts=(binding,), strategy_installation="NON_DEFAULT_RESEARCH"
            )
            return research_workspace.ResearchWorkspaceManifest.create(**values)

        def install(_task_id):  # type: ignore[no-untyped-def]
            # The install owner's one write, the pointer the Host reads; its own preparation
            # reads need a whole preparation.
            gate = live.session.mutation_gate
            research_workspace.update_research_workspace_manifest(
                live.workspace, pointed, gate=gate
            )
            return {"status": "INSTALLED_NON_DEFAULT_RESEARCH", "authority_hash": "a"}

        monkeypatch.setattr(live.operations.research_strategies, "install", install)
        answer = _cli(live, capsys, "strategy", "install", "--task", str(uuid4()))["data"]
        assert answer["status"] == "INSTALLED_NON_DEFAULT_RESEARCH", answer
        assert live.operations.installed()

        controls = _cli(live, capsys, "strategy-book", "controls", "--package", package_id)["data"]
        assert controls.get("template"), controls
        plan_file = tmp_path / "plan.json"
        plan_file.write_text(
            json.dumps({"operation": "PLAN", "spec": controls["template"]}), encoding="utf-8"
        )
        planned = _cli(live, capsys, "request", "--file", str(plan_file))
    # The plan reaches the installed package's own planning, which asks for the score history
    # this empty fixture lacks, never for an installation or a restart.
    assert planned["failure_code"] == "alpha_research.replay_manifest_absent", planned
