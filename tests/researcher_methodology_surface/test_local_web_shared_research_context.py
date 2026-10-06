"""Card 32: an exact research context shared across the CLI and Local Web.

The real Factor owner behind the real Host: competing previews stay addressable
by their own hash, a shared PLAN reads back exactly through both entries,
expiry and restart answer with an explicit re-PLAN path, every RUN branch admits
the caller first, and nothing here runs, confirms or moves a default pointer by
itself. One factor screening is executed to prove the admitted-Task answers; no
model, Provider or data download. The registry's concurrency regression needs no
workspace at all.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.composition.research_experiment_plan import ExperimentPlan
from alphalattice.control.product_host.composition.research_experiments import (
    PREVIEWS_DIRECTORY,
    AuthoringError,
)
from alphalattice.control.product_host.composition.research_workspace import (
    publish_research_workspace_manifest,
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import bind_factor_inputs
from alphalattice.control.product_host.storage.plan_previews import (
    PREVIEW_CAPACITY,
    PREVIEW_TTL,
    PreviewRegistry,
)
from tests.portfolio_strategy_lab.local_web_support import _json, _manifest, _resolved, _Resolver
from tests.researcher_methodology_surface.real_workspace import (
    build_real_risk_workspace,
    publish_causal_outcomes,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_alphalattice.py"


class _Clock:
    """Real time plus an offset the tests move to expire a preview."""

    def __init__(self) -> None:
        self.offset = timedelta()

    def __call__(self) -> datetime:
        return datetime.now(UTC) + self.offset


@pytest.fixture(scope="module")
def research_root(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str, str]:
    """A real Factor research workspace: root, admitted input id and binding hash."""

    source = build_real_risk_workspace(
        tmp_path_factory.mktemp("context-source"),
        symbols=tuple(f"F{i:03d}" for i in range(120)),
    )
    root = tmp_path_factory.mktemp("context-web") / "workspace"
    root.mkdir()
    publish_research_workspace_manifest(root, _manifest("shared-context"))
    outcome, _ref = publish_causal_outcomes(source, at=datetime(2026, 8, 2, tzinfo=UTC))
    binding = bind_factor_inputs(
        workspace=root, source=source.workspace, outcome_snapshot_hash=outcome
    )
    return root, binding.input_id, binding.binding_hash


@pytest.fixture(scope="module")
def clock() -> _Clock:
    return _Clock()


@pytest.fixture(scope="module")
def host(research_root: tuple[Path, str, str], clock: _Clock):  # type: ignore[no-untyped-def]
    root, _input_id, _binding = research_root
    session = LocalPortfolioWebSession(
        workspace=root,
        workspace_manifest=read_research_workspace_manifest(root),
        resolver=_Resolver(_resolved()),
        clock=clock,
    )
    session.start()
    try:
        yield session
    finally:
        session.stop()


def _cli(host: LocalPortfolioWebSession, *arguments: str) -> tuple[int, dict[str, Any]]:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--workspace",
            str(host.workspace),
            "--view",
            "full",
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert host.web.application.external_token not in result.stdout  # type: ignore[union-attr]
    return result.returncode, json.loads(result.stdout)


def _binding(host: LocalPortfolioWebSession) -> tuple[str, str]:
    """The one admitted research input, as the running Host names it."""

    inputs = _json(host, "/api/research-inputs")["inputs"]
    assert len(inputs) == 1 and len(inputs[0]["versions"]) == 1, inputs
    return inputs[0]["input_id"], inputs[0]["versions"][0]["binding_hash"]


def _declaration(host: LocalPortfolioWebSession, path: Path, seed: int) -> Path:
    input_id, binding = _binding(host)
    controls = _json(
        host,
        "/api/experiments/controls?"
        f"research_input_id={input_id}&input_binding_hash={binding}"
        "&experiment_kind=factor.screening-development",
    )
    text = controls["yaml"].replace("seed: 20260816", f"seed: {seed}")
    assert f"seed: {seed}" in text
    path.write_text(text, encoding="utf-8")
    return path


def _plan(host: LocalPortfolioWebSession, declaration: Path) -> dict[str, Any]:
    input_id, binding = _binding(host)
    code, body = _cli(
        host,
        "study",
        "plan",
        "--input",
        input_id,
        "--binding",
        binding,
        "--file",
        str(declaration),
    )
    assert code == 0 and body["data"]["status"] == "PLANNED", body
    return body


def _preview(host: LocalPortfolioWebSession, plan_hash: str) -> dict[str, Any]:
    code, body = _cli(host, "study", "show", "--plan", plan_hash)
    assert code == 0, body
    return body["data"]


def test_competing_previews_expiry_and_restart_keep_every_plan_addressable(
    host: LocalPortfolioWebSession, clock: _Clock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiments = host.operations.experiments  # type: ignore[union-attr]
    manifest_before = read_research_workspace_manifest(host.workspace).manifest_hash
    tasks_before = len(host.session.task_control_registry.tasks())  # type: ignore[union-attr]
    head_before = _json(host, "/api/activity")["head"]

    # Two actors' previews side by side: the second PLAN no longer displaces the first.
    plan_a = _plan(host, _declaration(host, tmp_path / "a.yaml", 20260816))["data"]
    plan_b = _plan(host, _declaration(host, tmp_path / "b.yaml", 20260817))["data"]
    hash_a, hash_b = plan_a["plan_hash"], plan_b["plan_hash"]
    assert hash_a != hash_b
    assert plan_a["preview"] == {"retention": "ON_DISK_UNTIL_EXPIRY", "expires_after_minutes": 60}
    # Sealed by its hash (V105): a restart or an eviction finds it, a changed file does not.
    sealed_root = host.workspace / "runtime" / PREVIEWS_DIRECTORY
    restarted = PreviewRegistry(
        model=ExperimentPlan, clock=lambda: datetime.now(UTC), root=sealed_root
    )
    found = restarted.get(hash_a)
    assert found is not None and found.plan.plan_hash == hash_a
    assert found.plan.document == plan_a["document"] and found.caller == "EXTERNAL_AUTOMATION"
    sealed = sealed_root / f"{hash_a}.json"
    kept = sealed.read_text(encoding="utf-8")
    changed = json.loads(kept)
    changed["plan"]["execution_preview"] = {**changed["plan"]["execution_preview"], "moved": 1}
    sealed.write_text(json.dumps(changed), encoding="utf-8")
    assert (
        PreviewRegistry(
            model=ExperimentPlan, clock=lambda: datetime.now(UTC), root=sealed_root
        ).get(hash_a)
        is None
    )
    sealed.write_text(kept, encoding="utf-8")
    assert plan_a["next_requests"]["inspect"] == {
        "operation": "EXPERIMENT_PREVIEW_READBACK",
        "experiment_plan_hash": hash_a,
    }
    # An installed Agent without its execution binding is not admitted to RUN a
    # retained preview; the preview stays readable and unadmitted.
    with pytest.raises(AuthoringError, match="agent_execution_not_admitted"):
        experiments.run(hash_a, caller="INSTALLED_AGENT", agent_execution=None)
    for plan_hash, planned in ((hash_a, plan_a), (hash_b, plan_b)):
        preview = _preview(host, plan_hash)
        assert preview["status"] == "AVAILABLE" and preview["plan_hash"] == plan_hash
        assert preview["document"] == planned["document"]  # exact, not the newest
        assert preview["caller"] == "EXTERNAL_AUTOMATION"
        assert preview["next_requests"]["run"] == {
            "operation": "EXPERIMENT_RUN",
            "experiment_plan_hash": plan_hash,
        }
        assert preview["next_requests"]["replan"]["experiment_document"] == planned["document"]
        web = _json(host, f"/api/experiments/preview?experiment_plan_hash={plan_hash}")
        assert web["document"] == preview["document"] and web["yaml"] == preview["yaml"]
    assert _preview(host, hash_a)["retained_previews"] == 2

    # Reading previews admits nothing, records no activity and moves no pointer.
    assert len(host.session.task_control_registry.tasks()) == tasks_before  # type: ignore[union-attr]
    assert read_research_workspace_manifest(host.workspace).manifest_hash == manifest_before
    head_after_reads = _json(host, "/api/activity")["head"]
    assert head_after_reads == head_before + 4  # two PLAN pairs; the preview reads added nothing

    # An unknown hash is MISSING and never resolved to the latest preview.
    missing = _preview(host, "f" * 64)
    assert missing["status"] == "MISSING" and "document" not in missing
    assert missing["next_action"] == "EXPERIMENT_PLAN"
    code, refused = _cli(host, "study", "run", "--plan", "f" * 64)
    assert code == 2 and refused["data"]["failure_code"] == "research_experiment.preview_required"

    # Expiry: readable, not runnable; the refusal carries the exact re-PLAN request.
    clock.offset = PREVIEW_TTL + timedelta(minutes=1)
    expired = _preview(host, hash_a)
    assert expired["status"] == "EXPIRED" and expired["document"] == plan_a["document"]
    assert "run" not in expired["next_requests"] and "replan" in expired["next_requests"]
    code, refused = _cli(host, "study", "run", "--plan", hash_a)
    assert code == 2 and refused["data"]["failure_code"] == "research_experiment.preview_expired"
    assert refused["data"]["next_requests"]["replan"]["operation"] == "EXPERIMENT_PLAN"
    with pytest.raises(AuthoringError, match="agent_execution_not_admitted"):
        experiments.run(hash_a, caller="INSTALLED_AGENT", agent_execution=None)  # admission first
    assert len(host.session.task_control_registry.tasks()) == tasks_before  # type: ignore[union-attr]
    saved = tmp_path / "expired-a.json"
    code, _body = _cli(host, "study", "show", "--plan", hash_a, "--output", str(saved))
    assert code == 0 and saved.is_file()
    code, replanned = _cli(host, "request", "--from", str(saved), "--action", "replan")
    assert code == 0 and replanned["data"]["status"] == "PLANNED"
    assert replanned["data"]["plan_hash"] == hash_a  # the same exact plan, re-created explicitly
    assert _preview(host, hash_a)["status"] == "AVAILABLE"

    # Memory is bounded; B, expired meanwhile, left the disk when the next preview was sealed,
    # and A stays addressable.
    for index in range(PREVIEW_CAPACITY - 1):
        _plan(host, _declaration(host, tmp_path / f"fill-{index}.yaml", 20270000 + index))
    assert _preview(host, hash_b)["status"] == "MISSING"
    survivor = _preview(host, hash_a)
    assert survivor["status"] == "AVAILABLE", survivor.get("invalidated_by")
    assert survivor["retained_previews"] == PREVIEW_CAPACITY

    # A workspace that moved beneath a preview makes it INVALID, with the same re-PLAN path.
    def moved(_plan: object, *, task_id: object = None) -> None:
        raise AuthoringError("research_experiment.execution_binding_changed")

    monkeypatch.setattr(experiments, "_current", moved)
    invalid = _preview(host, hash_a)
    assert invalid["status"] == "INVALID"
    assert invalid["invalidated_by"] == "research_experiment.execution_binding_changed"
    assert "failure_code" not in invalid  # the read succeeded; the plan is what cannot run
    assert "run" not in invalid["next_requests"] and "replan" in invalid["next_requests"]
    code, refused = _cli(host, "study", "run", "--plan", hash_a)
    assert code == 2 and refused["data"]["failure_code"] == (
        "research_experiment.execution_binding_changed"
    )
    assert refused["data"]["next_requests"]["replan"]["experiment_document"] == plan_a["document"]
    monkeypatch.undo()
    assert _preview(host, hash_a)["status"] == "AVAILABLE"

    # Explicit RUN of the chosen preview admits exactly that plan; the preview then reads ADMITTED.
    code, sent = _cli(host, "study", "run", "--plan", hash_a)
    assert code == 3 and sent["data"]["status"] == "ADMITTED", sent
    task_id = sent["data"]["task_id"]
    host.dispatcher.drain_for_tests()  # type: ignore[union-attr]
    assert _json(host, f"/api/status?task_id={task_id}")["lifecycle"] == "SUCCEEDED"
    admitted = _preview(host, hash_a)
    assert admitted["status"] == "AVAILABLE"  # still retained here...
    assert admitted["existing_task"]["task_id"] == task_id  # ...and its Task is named beside it
    assert admitted["existing_task"]["lifecycle"] == "SUCCEEDED"

    # A restart finds the sealed preview beside its durable Task (V105); nothing resolves to
    # the latest.
    count = len(host.session.task_control_registry.tasks())  # type: ignore[union-attr]
    host.stop()
    host.start()
    after_restart = _preview(host, hash_a)
    assert after_restart["status"] == "AVAILABLE"
    assert after_restart["document"] == plan_a["document"]
    assert after_restart["existing_task"]["task_id"] == task_id
    assert after_restart["existing_task"]["readback"] == {
        "operation": "EXPERIMENT_READBACK",
        "task_id": task_id,
    }
    assert _preview(host, hash_b)["status"] == "MISSING"
    code, refused = _cli(host, "study", "run", "--plan", hash_b)
    assert code == 2 and refused["data"]["failure_code"] == "research_experiment.preview_required"
    # The caller is admitted before the answer names the existing Task; reading that Task
    # needs no binding at all.
    restarted = host.operations.experiments  # type: ignore[union-attr]
    with pytest.raises(AuthoringError, match="agent_execution_not_admitted"):
        restarted.run(hash_a, caller="INSTALLED_AGENT", agent_execution=None)
    code, read = _cli(host, "study", "show", task_id)
    assert code == 0 and read["data"]["status"] == "EXPERIMENT_PUBLISHED"
    code, reused = _cli(host, "study", "run", "--plan", hash_a)
    assert code == 0 and reused["data"]["status"] == "REUSED_EXACT"
    assert reused["data"]["publication_task_id"] == task_id
    assert len(host.session.task_control_registry.tasks()) == count  # type: ignore[union-attr]
    assert read_research_workspace_manifest(host.workspace).manifest_hash == manifest_before


def test_shared_plan_is_navigable_and_visible_as_activity_without_execution(
    host: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    from alphalattice.interface.local_application.client import LocalResearchClient

    tasks_before = len(host.session.task_control_registry.tasks())  # type: ignore[union-attr]
    cursor = _json(host, "/api/activity")["cursor"]
    planned = _plan(host, _declaration(host, tmp_path / "shared.yaml", 20280101))
    plan_hash = planned["data"]["plan_hash"]
    assert planned["local_web_url"].endswith(f"/?plan={plan_hash}")
    client = LocalResearchClient(host.workspace)
    navigation = client.navigation({"operation": "EXPERIMENT_PLAN"}, planned["data"])
    assert navigation["kind"] == "shared_plan" and navigation["claim"] == "NAVIGATION_NOT_AUTHORITY"
    assert client.navigation(
        {"operation": "EXPERIMENT_PREVIEW_READBACK", "experiment_plan_hash": plan_hash}, {}
    )["url"].endswith(f"/?plan={plan_hash}")
    page = _json(host, f"/api/activity?after={cursor}")
    returned = [
        item
        for item in page["items"]
        if item["payload"].get("operation") == "EXPERIMENT_PLAN"
        and item["payload"].get("phase") == "RETURNED"
    ]
    assert returned and returned[-1]["payload"]["subject"]["plan_hash"] == plan_hash
    assert returned[-1]["payload"]["caller"] == "EXTERNAL_AUTOMATION"
    assert returned[-1]["task_id"] is None
    head = page["head"]
    _preview(host, plan_hash)
    _json(host, f"/api/experiments/preview?experiment_plan_hash={plan_hash}")
    assert _json(host, "/api/activity")["head"] == head  # inspecting is a read
    assert len(host.session.task_control_registry.tasks()) == tasks_before  # type: ignore[union-attr]


@dataclass(frozen=True)
class _Hashed:
    plan_hash: str


def test_preview_registry_is_coherent_under_concurrent_plans() -> None:
    """The service answers PLANs on several threads: capacity eviction, refresh and reads
    must neither corrupt the table nor raise. The reviewed registry raised KeyError and
    RuntimeError under this exact load (capacity 2, four threads, 8000 remembers)."""

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # force interleavings inside the tiny critical sections
    try:
        registry = PreviewRegistry(
            model=ExperimentPlan, clock=lambda: datetime.now(UTC), capacity=2
        )
        failures: list[BaseException] = []
        seen: list[int] = []

        def worker(seed: int) -> None:
            for index in range(2000):
                try:
                    plan = cast(ExperimentPlan, _Hashed(f"{seed}-{index}"))
                    registry.remember(plan, caller="HUMAN")
                    registry.get(f"{(seed + 1) % 4}-{index}")
                    seen.append(len(registry))
                except BaseException as error:  # every failure kind is the finding
                    failures.append(error)

        threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(previous)
    assert failures == []
    assert max(seen) <= 2 and len(registry) == 2
    for plan_hash in list(registry._entries):  # the table itself is the claim
        entry = registry.get(plan_hash)
        assert entry is not None and entry.plan.plan_hash == plan_hash
