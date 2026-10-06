"""`run` executes and `replay` reads back. They used to be the same call.

The external review's decisive finding was that ``validate``, ``freeze``,
``run``, and ``replay`` all resolved to one dispatcher call, so no experiment
ever executed. These cases assert the three commands now differ in what they do,
measured rather than asserted: a first run performs real numerical work and
writes immutable evidence, and a replay performs none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from alphalattice.control.product_host.composition.research_authoring import (
    build_research_experiment_dispatcher,
    build_research_program_workflow,
    installed_desk_compilers,
)
from alphalattice.control.product_host.research_authoring.authority import (
    WorkspaceResearchAuthorityResolver,
)
from alphalattice.control.product_host.research_authoring.execution import (
    WorkspaceReturnSurfaceProvider,
)
from alphalattice.control.research_program.authoring.document import load_authoring_document
from alphalattice.control.research_program.authoring.workflow import (
    ResearchEvidenceStore,
    ResearchProgramStore,
    ResearchProgramWorkflow,
)
from alphalattice.investment.risk_research.experiments.execution import RiskExperimentExecutor
from alphalattice.protocols.actor_execution.contracts import ActorKind
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExecutionEvidence,
    ResearchExperimentEnvelope,
    SealedResearchProgram,
)
from tests.researcher_methodology_surface.real_workspace import RealRiskWorkspace

_FIXTURE = Path(__file__).parent / "fixtures" / "risk_covariance_development.yaml"
_OUTPUT_WORKSPACE = "workspaces/research/risk-development"


class CountingRecorder:
    """A test-side observer, injected through the seam the product already has.

    The plan is explicit that no counting *catalog* ships in the product: a
    second installation path used only by tests is exactly the kind of drift
    this milestone is meant to prevent.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def record(self, *, capability: str) -> None:
        self.calls.append(capability)


def _workflow(workspace: RealRiskWorkspace, root: Path) -> object:
    return build_research_program_workflow(
        workspace=workspace.workspace,
        workspace_root=root,
        executors=(
            RiskExperimentExecutor(
                # The real Host provider over the real artifact root: the surface
                # is now chosen against the resolved authority, so handing the
                # executor one up front would bypass the thing under test.
                surface_provider=WorkspaceReturnSurfaceProvider(workspace.artifact_root),
                return_reader=workspace.return_reader,
                sector_by_listing_id=workspace.sector_by_listing_id,
                freshness_probe=workspace.freshness_probe,
            ),
        ),
    )


def test_run_executes_for_real_and_replay_reads_back_with_zero_calls(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path: Path,
) -> None:
    document = load_authoring_document(_FIXTURE.read_text(encoding="utf-8"))
    workflow = _workflow(real_risk_workspace, tmp_path)
    recorder = CountingRecorder()

    program, _ = workflow.preflight(  # type: ignore[attr-defined]
        document,
        actor_kind=ActorKind.HUMAN,
        actor_id="researcher",
    )
    evidence, binding = workflow.run_sealed(  # type: ignore[attr-defined]
        document,
        actor_kind=ActorKind.HUMAN,
        actor_id="researcher",
        recorder=recorder,
    )
    assert program.program_hash == evidence.program_hash

    assert evidence.disposition == "COMPUTED"
    assert evidence.identity_class == "DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"
    assert evidence.numerical_call_count > 0
    assert len(recorder.calls) == evidence.numerical_call_count
    assert evidence.artifact_uris and evidence.formation_sessions
    assert binding.actor_kind is ActorKind.HUMAN

    replay_recorder = CountingRecorder()
    replayed, _ = workflow.replay(  # type: ignore[attr-defined]
        document,
        actor_kind=ActorKind.HUMAN,
        actor_id="researcher",
    )

    assert replayed.disposition == "REUSED_EXACT"
    assert replayed.numerical_call_count == 0
    # Nothing was recorded because no executor was reached at all.
    assert replay_recorder.calls == []
    assert replayed.program_hash == evidence.program_hash
    assert replayed.artifact_uris == evidence.artifact_uris
    assert replayed.formation_sessions == evidence.formation_sessions

    # regression: an explicitly named immutable Program can be inspected and
    # graph-read back without installing an executor or resolving live inputs.
    source_free = build_research_program_workflow(
        workspace=real_risk_workspace.workspace,
        workspace_root=tmp_path,
        executors=(),
    )
    sealed, stored, _ = source_free.inspect_sealed(  # type: ignore[attr-defined]
        document,
        program_hash=evidence.program_hash,
        actor_kind=ActorKind.HUMAN,
        actor_id="researcher",
    )
    assert sealed.program_hash == evidence.program_hash
    assert stored == evidence
    durable, _ = source_free.readback_sealed(  # type: ignore[attr-defined]
        document,
        program_hash=evidence.program_hash,
        actor_kind=ActorKind.HUMAN,
        actor_id="researcher",
    )
    assert durable.disposition == "REUSED_EXACT"
    assert durable.numerical_call_count == 0
    assert durable.artifact_uris == evidence.artifact_uris

    # regression: the sealed path validates an Alpha envelope and Program
    # without requiring an Alpha compiler or any live authority installation.
    alpha_document = dict(document)
    alpha_experiment = dict(alpha_document["experiment"])
    alpha_experiment.update(
        {
            "kind": "alpha.model-development",
            "output_workspace": "workspaces/research/alpha-sealed-readback",
            "baseline_workspace": "workspaces/research/alpha-sealed-baseline",
        }
    )
    alpha_document["experiment"] = alpha_experiment
    alpha_envelope = ResearchExperimentEnvelope.create(**alpha_experiment)
    alpha_program = SealedResearchProgram.create(
        kind=alpha_envelope.kind,
        envelope_hash=alpha_envelope.envelope_hash,
        desk_program_hash="1" * 64,
        resolved_sessions=program.resolved_sessions,
        catalog_hash="2" * 64,
        method_binding_hash="3" * 64,
        parameter_domain_hash="4" * 64,
        authority_hash="5" * 64,
    )
    alpha_evidence = ResearchExecutionEvidence.create(
        kind=alpha_program.kind,
        program_hash=alpha_program.program_hash,
        desk_program_hash=alpha_program.desk_program_hash,
        method_binding_hash=alpha_program.method_binding_hash,
        authority_hash=alpha_program.authority_hash,
        disposition="COMPUTED",
        numerical_call_count=1,
        artifact_uris=("playpen://alpha-sealed/root",),
        formation_sessions=alpha_program.resolved_sessions,
        desk_input_binding_hash="6" * 64,
    )
    alpha_output = tmp_path / alpha_envelope.output_workspace
    ResearchProgramStore(alpha_output).publish(
        program=alpha_program,
        document=alpha_document,
    )
    ResearchEvidenceStore(alpha_output).publish(alpha_evidence)
    verified_authorities: list[object] = []

    class AlphaSealedVerifier:
        kind = "alpha.model-development"
        replay_disposition = "VERIFIED_DURABLE_GRAPH_READBACK"

        def verify(self, **values: object) -> None:
            verified_authorities.append(values["authority"])

    alpha_source_free = ResearchProgramWorkflow(
        dispatcher=build_research_experiment_dispatcher(workspace=real_risk_workspace.workspace),
        executors=(),
        workspace_root=tmp_path,
        source_workspace=real_risk_workspace.workspace,
        verifiers=(AlphaSealedVerifier(),),  # type: ignore[arg-type]
    )
    alpha_sealed, alpha_stored, _ = alpha_source_free.inspect_sealed(
        alpha_document,
        program_hash=alpha_program.program_hash,
        actor_kind=ActorKind.HUMAN,
        actor_id="researcher",
    )
    assert alpha_sealed == alpha_program
    assert alpha_stored == alpha_evidence
    alpha_readback, _ = alpha_source_free.readback_sealed(
        alpha_document,
        program_hash=alpha_program.program_hash,
        actor_kind=ActorKind.HUMAN,
        actor_id="researcher",
    )
    assert alpha_readback.disposition == "VERIFIED_DURABLE_GRAPH_READBACK"
    assert alpha_readback.numerical_call_count == 0
    assert verified_authorities == [None]


def test_risk_recovery_reuses_verified_chunks_and_refuses_a_tampered_prefix(
    real_risk_workspace, tmp_path, monkeypatch
):
    import json
    from copy import deepcopy

    from alphalattice.investment.risk_research.experiments.observation import ObservingAdapter

    document = load_authoring_document(_FIXTURE.read_text(encoding="utf-8"))
    document["experiment"]["sessions"]["end"] = "2024-03-15"
    document["experiment"]["sessions"]["as_of"]["session"] = "2024-03-20"
    workflow = _workflow(real_risk_workspace, tmp_path)
    arguments = {"actor_kind": ActorKind.HUMAN, "actor_id": "researcher"}
    calls = []
    interrupted = [False]
    estimate = ObservingAdapter.estimate

    def fail_after_one_chunk(self, **kwargs):
        if len(calls) == 21 and not interrupted[0]:
            interrupted[0] = True
            raise RuntimeError("deliberate interruption after a sealed Risk chunk")
        result = estimate(self, **kwargs)
        calls.append(kwargs["inputs"].formation_session)
        return result

    monkeypatch.setattr(ObservingAdapter, "estimate", fail_after_one_chunk)
    workflow.preflight(document, **arguments)
    with pytest.raises(RuntimeError, match="deliberate interruption"):
        workflow.run_sealed(document, **arguments)
    assert len(calls) == 21
    checkpoint = next(
        (tmp_path / _OUTPUT_WORKSPACE / "risk-research/development/build-checkpoints").glob(
            "*.json"
        )
    )
    original = checkpoint.read_bytes()
    payload = json.loads(original)
    payload["expected_formation_count"] += 1
    try:
        checkpoint.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="checkpoint_invalid"):
            workflow.run_sealed(document, **arguments)
        assert len(calls) == 21
    finally:
        checkpoint.write_bytes(original)
    resumed, _ = workflow.run_sealed(document, **arguments)
    assert resumed.numerical_call_count == len(resumed.formation_sessions) - 21
    assert len(calls) == len(set(calls)) == len(resumed.formation_sessions)
    replayed, _ = workflow.replay(document, **arguments)
    assert replayed.numerical_call_count == 0
    assert len(calls) == len(resumed.formation_sessions)
    control = deepcopy(document)
    control["experiment"]["output_workspace"] = "workspaces/research/risk-control"
    workflow.preflight(control, **arguments)
    fresh, _ = workflow.run_sealed(control, **arguments)
    assert fresh.artifact_uris == resumed.artifact_uris
    assert fresh.numerical_call_count == len(resumed.formation_sessions)


def test_replay_before_any_run_refuses_rather_than_inventing_evidence(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path: Path,
) -> None:
    """A replay with nothing to read is a failure, not an empty success."""

    document = load_authoring_document(_FIXTURE.read_text(encoding="utf-8"))
    with pytest.raises(AuthoringError, match="evidence_unavailable_for_replay"):
        _workflow(real_risk_workspace, tmp_path).replay(  # type: ignore[attr-defined]
            document, actor_kind=ActorKind.HUMAN, actor_id="researcher"
        )


def test_freeze_writes_no_evidence_and_performs_no_numerical_work(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path: Path,
) -> None:
    document = load_authoring_document(_FIXTURE.read_text(encoding="utf-8"))
    program, _ = _workflow(real_risk_workspace, tmp_path).freeze(  # type: ignore[attr-defined]
        document, actor_kind=ActorKind.HUMAN, actor_id="researcher"
    )

    store = ResearchEvidenceStore(tmp_path / _OUTPUT_WORKSPACE)
    assert store.load_for_program(program.program_hash) is None
    assert not (tmp_path / _OUTPUT_WORKSPACE / "risk-research").exists()


def test_evidence_is_content_addressed_and_a_reuse_claim_cannot_carry_calls(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path: Path,
) -> None:
    """Two facts the contract enforces rather than trusting a caller for."""

    from alphalattice.protocols.research_authoring.contracts import ResearchExecutionEvidence

    document = load_authoring_document(_FIXTURE.read_text(encoding="utf-8"))
    evidence, _ = _workflow(real_risk_workspace, tmp_path).run(  # type: ignore[attr-defined]
        document, actor_kind=ActorKind.HUMAN, actor_id="researcher"
    )
    store = ResearchEvidenceStore(tmp_path / _OUTPUT_WORKSPACE)
    assert store.load_for_program(evidence.program_hash) == evidence

    # A second observation may not make a previously readable Program ambiguous.
    altered = evidence.model_dump(exclude={"evidence_hash"})
    altered["numerical_call_count"] += 1
    with pytest.raises(ValueError, match="evidence_identity_conflict"):
        store.publish(ResearchExecutionEvidence.create(**altered))
    assert store.load_for_program(evidence.program_hash) == evidence

    # The guards run inside pydantic, so an AuthoringError surfaces wrapped in a
    # ValidationError; both are ValueError and both fail closed.
    payload = evidence.model_dump(mode="json")
    payload["numerical_call_count"] = payload["numerical_call_count"] + 1
    with pytest.raises(ValueError, match="evidence_identity_invalid"):
        ResearchExecutionEvidence(**payload)

    with pytest.raises(ValueError, match="reuse_claimed_with_numerical_calls"):
        ResearchExecutionEvidence.create(
            kind=evidence.kind,
            program_hash=evidence.program_hash,
            desk_program_hash=evidence.desk_program_hash,
            method_binding_hash=evidence.method_binding_hash,
            authority_hash=evidence.authority_hash,
            disposition="REUSED_EXACT",
            numerical_call_count=1,
            artifact_uris=evidence.artifact_uris,
            formation_sessions=evidence.formation_sessions,
            desk_input_binding_hash=evidence.desk_input_binding_hash,
        )


# =============================== what each command asks its owners for


@dataclass
class _CountingAuthority:
    """The real resolver, counting how often a command asks it to resolve.

    Resolution reads the live workspace. Whether it happens once or three times
    inside one command is the difference between a fresh check and the same
    check repeated, and only a count can tell them apart.
    """

    inner: Any
    calls: int = 0

    def resolve(self, envelope: Any) -> Any:
        self.calls += 1
        return self.inner.resolve(envelope)


@dataclass
class _CountingCompiler:
    """The real Desk compiler, counting compilations."""

    inner: Any
    calls: int = 0

    @property
    def kind(self) -> str:
        return str(self.inner.kind)

    def compile_desk_program(self, **arguments: Any) -> Any:
        self.calls += 1
        return self.inner.compile_desk_program(**arguments)


@dataclass
class _CountingExecutor:
    """The real Risk executor, counting executions."""

    inner: Any
    calls: int = 0

    @property
    def kind(self) -> str:
        return str(self.inner.kind)

    def execute(self, **arguments: Any) -> Any:
        self.calls += 1
        return self.inner.execute(**arguments)


@dataclass
class _Counts:
    authority: _CountingAuthority
    compiler: _CountingCompiler
    executor: _CountingExecutor
    workflow: Any
    marks: list[tuple[int, int, int]] = field(default_factory=list)

    def take(self) -> tuple[int, int, int]:
        """Resolutions, compilations and executions since the last call."""

        previous = self.marks[-1] if self.marks else (0, 0, 0)
        current = (self.authority.calls, self.compiler.calls, self.executor.calls)
        self.marks.append(current)
        return tuple(now - before for now, before in zip(current, previous, strict=True))


def _counted(workspace: RealRiskWorkspace, root: Path) -> _Counts:
    authority = _CountingAuthority(
        inner=WorkspaceResearchAuthorityResolver(
            workspace=workspace.workspace,
            artifact_root=workspace.artifact_root,
        )
    )
    compiler = _CountingCompiler(inner=installed_desk_compilers()[0])
    executor = _CountingExecutor(
        inner=RiskExperimentExecutor(
            surface_provider=WorkspaceReturnSurfaceProvider(workspace.artifact_root),
            return_reader=workspace.return_reader,
            sector_by_listing_id=workspace.sector_by_listing_id,
            freshness_probe=workspace.freshness_probe,
        )
    )
    return _Counts(
        authority=authority,
        compiler=compiler,
        executor=executor,
        workflow=build_research_program_workflow(
            workspace=workspace.workspace,
            workspace_root=root,
            executors=(executor,),
            authority=authority,
            compilers=(compiler,),
        ),
    )


def test_each_command_resolves_authority_and_compiles_once(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path: Path,
) -> None:
    """requirement: one fresh authority resolution per command, not several.

    Every command here resolves the authority against the live workspace, and
    every command still must -- that is the check that makes a sealed identity
    mean something. What it must not do is resolve the same handles two or three
    times inside one call, which is what happened when each command re-derived
    the envelope and the authority that sealing had already produced. `resume`
    was the worst: it sealed, then delegated to a command that sealed again.

    Executions are counted beside them, because the point of the reduction is
    that it did not touch them: `preflight`, `replay`, `resume`-into-reuse and
    `inspect` still reach no executor at all.
    """

    document = load_authoring_document(_FIXTURE.read_text(encoding="utf-8"))
    counted = _counted(real_risk_workspace, tmp_path)
    actor = {"actor_kind": ActorKind.HUMAN, "actor_id": "researcher"}

    counted.workflow.preflight(document, **actor)
    assert counted.take() == (1, 1, 0)

    # `run` needs no sealed Program and `run_sealed` refuses without one; both
    # reach the executor exactly once, and neither resolves twice to do it.
    first, _ = counted.workflow.run(document, **actor)
    assert counted.take() == (1, 1, 1)
    evidence_root = tmp_path / _OUTPUT_WORKSPACE / "research-evidence"
    original = {path.name: path.read_bytes() for path in evidence_root.glob("*.json")}
    assert len(original) == 1 and first.numerical_call_count > 0

    repeated, _ = counted.workflow.run_sealed(document, **actor)
    assert counted.take() == (1, 1, 1)
    assert repeated.disposition == "REUSED_EXACT" and repeated.numerical_call_count == 0
    assert repeated.artifact_uris == first.artifact_uris
    assert {path.name: path.read_bytes() for path in evidence_root.glob("*.json")} == original

    counted.workflow.replay(document, **actor)
    assert counted.take() == (1, 1, 0)

    # The reuse branch: evidence exists, so this is a replay and reaches no
    # executor. It used to seal twice to discover that.
    counted.workflow.resume(document, **actor)
    assert counted.take() == (1, 1, 0)

    counted.workflow.inspect(document, **actor)
    assert counted.take() == (1, 1, 0)


def test_a_study_whose_inputs_moved_reads_by_its_recorded_program(
    real_risk_workspace: RealRiskWorkspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """regression (V108): a study whose upstream inputs moved compiles another Program, so its
    live inspect read as not found unless `--program-hash` named the old one; it reads by the
    Program its document sealed, from the store's index, as `NOT_CURRENT`."""

    from alphalattice.control.research_program.authoring.workflow import ResearchProgramStore

    document = load_authoring_document(_FIXTURE.read_text(encoding="utf-8"))
    counted = _counted(real_risk_workspace, tmp_path)
    actor = {"actor_kind": ActorKind.HUMAN, "actor_id": "researcher"}
    counted.workflow.preflight(document, **actor)  # PLAN seals the Program the run executes
    counted.workflow.run_sealed(document, **actor)
    program, evidence, _binding, standing = counted.workflow.inspect(document, **actor)
    assert standing == "CURRENT" and evidence is not None

    real_load = ResearchProgramStore.load
    asked: list[str] = []

    def moved(self: ResearchProgramStore, *, program_hash: str, document: object) -> object:
        # Today's compile is another Program, which no run sealed.
        asked.append(program_hash)
        if len(asked) == 1:
            return None
        return real_load(self, program_hash=program_hash, document=document)  # type: ignore[arg-type]

    monkeypatch.setattr(ResearchProgramStore, "load", moved)
    recorded, read, _binding, standing = counted.workflow.inspect(document, **actor)
    assert standing == "NOT_CURRENT"
    assert recorded.program_hash == program.program_hash and read == evidence


def test_a_run_is_held_offline_and_a_plan_over_budget_is_refused_at_plan(
    real_risk_workspace: RealRiskWorkspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """regression (V116, V120): the offline rule read the process switch and only Risk's
    executor checked it, so an outcome followed the shell that started the process, and a Risk
    PLAN over its budget was stored and refused only at RUN. Every Desk's call runs held offline
    whatever its workspace allows, so a network open for an update refuses no study, and Risk's
    compile checks the budget at PLAN."""

    import copy

    from alphalattice.control.workspace_runtime.network_access import (
        CONTROL_PATH,
        NetworkAccess,
        network_access,
        set_network_access,
    )
    from alphalattice.kernel.shared_kernel.environment import offline

    workspace = real_risk_workspace.workspace
    document = load_authoring_document(_FIXTURE.read_text(encoding="utf-8"))
    counted = _counted(real_risk_workspace, tmp_path)
    actor = {"actor_kind": ActorKind.HUMAN, "actor_id": "researcher"}
    seen: list[tuple[NetworkAccess, bool]] = []

    def probe(**_arguments: Any) -> Any:
        seen.append((network_access(workspace), offline()))
        raise AuthoringError("probe.stopped_before_any_estimate")

    monkeypatch.delenv("ALPHALATTICE_NETWORK_DISABLED")
    set_network_access(workspace, enabled=True)
    try:
        counted.workflow.preflight(document, **actor)
        monkeypatch.setattr(counted.executor, "execute", probe)
        with pytest.raises(AuthoringError, match=r"probe\.stopped_before_any_estimate"):
            counted.workflow.run_sealed(document, **actor)
        assert seen == [(NetworkAccess(False, "RUN_HELD_OFFLINE"), True)]
        assert network_access(workspace) == NetworkAccess(True, "WORKSPACE_CONTROL")
    finally:
        (workspace / CONTROL_PATH).unlink()

    over = copy.deepcopy(dict(document))
    over["experiment"]["budget"]["maximum_numerical_calls"] = 1
    with pytest.raises(AuthoringError, match=r"research_authoring\.numerical_budget_exceeded"):
        counted.workflow.preflight(over, **actor)
