"""An installed research strategy is served by the running Host, with no restart."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from alphalattice.control.product_host.composition import research_workspace
from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.data_preparation import research_strategy
from alphalattice.control.product_host.data_preparation.research_strategy import (
    ARTIFACT_KEY,
    CATEGORY,
)
from alphalattice.interface.local_application.cli import main
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioResearchSpec,
)
from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (
    BROAD_FEATURE_PACKAGE,
    PRODUCT_EVIDENCE_ROOT_KEY,
    install_frozen_strategies,
)
from tests.portfolio_strategy_lab.local_web_support import _harness, _run


def _cli(live: LocalPortfolioWebSession, capsys, *arguments: str) -> dict:  # type: ignore[no-untyped-def,type-arg]
    main(["--workspace", str(live.workspace), "--view", "full", *arguments], serve=lambda _: 99)
    return json.loads(capsys.readouterr().out)


def test_an_installed_strategy_is_served_by_the_running_host_without_a_restart(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """An installed strategy reaches controls, planning and report disclosure without a restart.

    About 13 s: one real Host and a three-formation report."""
    shipped = install_frozen_strategies(artifacts={PRODUCT_EVIDENCE_ROOT_KEY: tmp_path})
    # The installed artifact's registry reads name the shipped package: a real preparation
    # runs whole Alpha and Risk studies.
    for module in (research_strategy, research_workspace):
        monkeypatch.setattr(module, "install_frozen_strategies", lambda artifacts: shipped)
    live = LocalPortfolioWebSession.from_workspace(tmp_path / "workspace")
    package_id = BROAD_FEATURE_PACKAGE.strategy_id
    with live:
        assert not live.operations.installed()
        file = tmp_path / "book.json"
        file.write_text("{}", encoding="utf-8")
        for action in ("controls", "preview", "run"):
            code = main(
                [
                    "--workspace",
                    str(live.workspace),
                    "--view",
                    "full",
                    "strategy-book",
                    action,
                    *([] if action == "controls" else ["--file", str(file)]),
                ],
                serve=lambda _: 99,
            )
            before = json.loads(capsys.readouterr().out)
            assert (code, before["failure_code"]) == (
                2,
                "research_workspace.strategy_not_installed",
            )
            assert before["failure_code"] == "research_workspace.strategy_not_installed", before
            assert before["detail"]
            assert before["data"]["next_action"] == "PREPARE_AND_INSTALL_A_RESEARCH_STRATEGY"
            assert before["next_requests"]["strategies"] == {
                "operation": "RESEARCH_STRATEGY_CONTROLS"
            }
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

        packages = live.application.resolver.installed_packages()
        assert live.application.executor.packages == dict(packages)
        proof_root = tmp_path / "disclosure"
        proof_root.mkdir()
        with _harness(proof_root) as harness:
            result = _run(harness, PortfolioResearchSpec.default())
            program = harness.application.program(result.result_hash)
            report = harness.application.report(result.result_hash)
        assert program.strategy_package_hash not in packages
        with pytest.raises(
            ValueError, match=r"^portfolio_application\.program_package_not_installed$"
        ):
            live.application.executor.disclosure(report=report, program=program)

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
