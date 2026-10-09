"""The workspace manifest's one write (WM): no writer replaces what another wrote.

The manifest is one file its writers replace whole. A model-training publication read it,
built its change and published outside the workspace's lock, so a strategy installation
between the two was lost; a preparation's recovery lost one the same way; and a
plan that held the whole manifest's hash refused after any other owner's publication.
"""

from __future__ import annotations

import json
import re
from argparse import Namespace
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

from alphalattice.control.product_host.composition import (
    evidence_authority_setup,
    portfolio_application,
)
from alphalattice.control.product_host.composition.research_workspace import (
    RESEARCH_WORKSPACE_MANIFEST_NAME,
    ResearchWorkspaceError,
    ResearchWorkspaceEvidenceReview,
    ResearchWorkspaceExperimentInput,
    ResearchWorkspaceManifest,
    ResearchWorkspaceModelTrainingInput,
    create_research_workspace_manifest,
    manifest_fields_hash,
    read_research_workspace_manifest,
    update_research_workspace_manifest,
)
from alphalattice.control.product_host.data_preparation import input_capture, research_strategy
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.interface.local_application.cli_contract import refusal_words
from alphalattice.kernel.shared_kernel.retired_spellings import RETIRED_SPELLINGS


def _input(input_id: str, digit: str) -> ResearchWorkspaceExperimentInput:
    return ResearchWorkspaceExperimentInput(input_id=input_id, binding_hash=digit * 64)


def _adding(value: ResearchWorkspaceExperimentInput):  # type: ignore[no-untyped-def]
    def change(current: ResearchWorkspaceManifest) -> ResearchWorkspaceManifest:
        return current.with_bindings(experiment_inputs=(*(current.experiment_inputs or ()), value))

    return change


def test_two_writers_through_the_one_write_keep_both_changes(tmp_path: Path) -> None:
    """the second writer waits for the first and changes what the first wrote."""

    create_research_workspace_manifest(tmp_path, ResearchWorkspaceManifest.research_only("ws"))
    gate = WorkspaceMutationGate()
    read, go_on = Event(), Event()
    first = _adding(_input("first", "a"))

    def slow(current: ResearchWorkspaceManifest) -> ResearchWorkspaceManifest:
        read.set()
        assert go_on.wait(10)
        return first(current)

    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(update_research_workspace_manifest, tmp_path, slow, gate=gate)
        assert read.wait(10)
        two = pool.submit(
            update_research_workspace_manifest, tmp_path, _adding(_input("second", "b")), gate=gate
        )
        with pytest.raises(FutureTimeout):
            two.result(timeout=0.3)  # it waits at the gate, not on a manifest of its own
        go_on.set()
        one.result(timeout=10)
        before, after = two.result(timeout=10)
    assert [v.input_id for v in before.experiment_inputs or ()] == ["first"]
    assert [v.input_id for v in after.experiment_inputs or ()] == ["first", "second"]
    assert read_research_workspace_manifest(tmp_path) == after


def test_a_new_workspace_is_created_once(tmp_path: Path) -> None:
    create_research_workspace_manifest(tmp_path, ResearchWorkspaceManifest.research_only("ws"))
    with pytest.raises(
        ResearchWorkspaceError, match=re.escape("research_workspace.manifest_exists")
    ):
        create_research_workspace_manifest(tmp_path, ResearchWorkspaceManifest.research_only("x"))


def _recorded_setup_source(root: Path) -> tuple[Path, str, Path]:
    """A small synthetic recorded source, sealed by its public acquisition and byte owners."""
    from alphalattice.evidence.alternative_evidence.contracts import (
        AlternativeEvidenceClass,
        AlternativeEvidenceMode,
        AlternativeEvidenceRequest,
        AlternativeEvidenceSourcePolicy,
        SecIssuerRegistryEntry,
        SecIssuerRegistrySnapshot,
        seal_contract,
    )
    from alphalattice.evidence.alternative_evidence.documents.workspace import (
        AlternativeEvidenceDocumentPublisher,
    )
    from alphalattice.evidence.alternative_evidence.sources.acquisition import (
        AlternativeEvidenceAcquisitionService,
    )
    from alphalattice.evidence.alternative_evidence.sources.contracts import (
        AcquiredEvidenceSourceReferenceSet,
    )
    from alphalattice.evidence.alternative_evidence.sources.recorded import RecordedEvidenceDocument

    now = datetime(2026, 8, 12, 16, tzinfo=UTC)
    registry = seal_contract(
        SecIssuerRegistrySnapshot,
        "registry_hash",
        captured_at=now - timedelta(days=1),
        entries=(
            SecIssuerRegistryEntry(
                entity_id="ACME", ticker="ACME", cik="0000000001", legal_name="Synthetic Acme"
            ),
        ),
        source_content_hash="1" * 64,
    )
    request = seal_contract(
        AlternativeEvidenceRequest,
        "request_hash",
        ordered_entity_ids=("ACME",),
        evidence_as_of=now,
        acquisition_deadline=now + timedelta(hours=1),
        evidence_classes=(AlternativeEvidenceClass.ISSUER_OFFICIAL_RECORDED,),
        source_policy=AlternativeEvidenceSourcePolicy(),
        ttl_seconds=86_400,
        mode=AlternativeEvidenceMode.RECORDED,
    )
    acquisition = AlternativeEvidenceAcquisitionService(artifact_root=root / "artifacts")
    snapshot, documents = acquisition.build_recorded_evidence(
        request=request,
        registry=registry,
        documents=(
            RecordedEvidenceDocument(
                entity_id="ACME",
                source_right="USER_PROVIDED_FOR_LOCAL_RESEARCH",
                evidence_class=AlternativeEvidenceClass.ISSUER_OFFICIAL_RECORDED,
                document_type="8-K",
                revision="synthetic-release",
                published_at=now - timedelta(days=2),
                captured_at=now - timedelta(days=1),
                available_at=now - timedelta(days=1),
                text="# Synthetic release\n\nA supplier interruption reduced capacity.",
                immutable_source=True,
            ),
        ),
        published_at=now,
    )
    publisher = AlternativeEvidenceDocumentPublisher(root / "knowledge")
    source_set = seal_contract(
        AcquiredEvidenceSourceReferenceSet,
        "source_set_hash",
        request_hash=request.request_hash,
        source_snapshot_hash=snapshot.snapshot_hash,
        documents=tuple(publisher.publish_source_object(document) for document in documents),
        acquired_at=now,
    )
    acquisition.artifacts.publish("source-document-sets", source_set.source_set_hash, source_set)
    return acquisition.artifacts.root, source_set.source_set_hash, publisher.workspace.root


@pytest.mark.parametrize("rebind", (False, True), ids=("recorded-install", "installed-rebind"))
def test_evidence_setup_waits_for_the_manifest_owner_and_keeps_another_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rebind: bool
) -> None:
    """both setup entries change the current manifest under the session's gate."""
    from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
        PanelClosureArtifactStore,
    )
    from alphalattice.investment.risk_research.surfaces.returns import RiskReturnArtifactStore

    source_root, source_hash, knowledge_root = _recorded_setup_source(tmp_path / "source")
    workspace = tmp_path / "target"
    original = ResearchWorkspaceManifest.create(
        workspace_id="evidence-setup",
        default_strategy_package_id="fixture-package",
        default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
        strategy_artifacts=(),
        experiment_inputs=(_input("initial", "a"),),
    )
    create_research_workspace_manifest(workspace, original)
    models = workspace / "models"
    models.mkdir()
    artifact_root = workspace / "runtime" / "artifacts"
    # These public source readers stand in for the unrelated numerical universe build.
    for directory in (
        RiskReturnArtifactStore(artifact_root).root / "manifests",
        artifact_root / "feature-panel" / "closure" / "sector-maps",
    ):
        directory.mkdir(parents=True)
        (directory / ("b" * 64 + ".json")).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        RiskReturnArtifactStore,
        "load_manifest",
        lambda *_: SimpleNamespace(epoch=SimpleNamespace(universe_manifest_revision="fixture")),
    )
    monkeypatch.setattr(
        PanelClosureArtifactStore,
        "load_model",
        lambda *_, **__: SimpleNamespace(
            manifest_revision="fixture",
            entries=(SimpleNamespace(provider_symbol="ACME", listing_id="US-ACME"),),
        ),
    )
    monkeypatch.setattr(
        evidence_authority_setup,
        "probe_hybrid_retrieval_capabilities",
        lambda *_: SimpleNamespace(status="READY", logical_hash="c" * 64),
    )
    entered, parent_thread = Event(), get_ident()

    class ObservedGate(WorkspaceMutationGate):
        @contextmanager
        def hold(self) -> Iterator[None]:
            if get_ident() != parent_thread:
                entered.set()
            with super().hold():
                yield

    gate = ObservedGate()
    monkeypatch.setattr(
        evidence_authority_setup.WorkspaceApplicationSession,
        "acquire",
        lambda _workspace: nullcontext(SimpleNamespace(mutation_gate=gate)),
    )
    arguments = Namespace(
        workspace=workspace,
        model_name=None,
        deepseek_base_url=None,
        minimum_entity_coverage=1.0,
        source_artifact_root=source_root,
        source_set_hash=source_hash,
        source_knowledge_root=knowledge_root,
        acquire_sec=False,
        rebind_installed=False,
        recipe=None,
        semantic_model=models,
        model_store=None,
        entities=["ACME"],
        authority_id="initial-authority",
        install=True,
        matter_selection="INTEGRATED_TOPIC_ROUTING",
    )
    first = evidence_authority_setup.materialize(arguments)
    first_path = workspace / first["binding"]["relative_path"]
    first_bytes = first_path.read_bytes()
    arguments.authority_id, arguments.install = "intervening-authority", False
    intervening = evidence_authority_setup.materialize(arguments)
    intervening_binding = ResearchWorkspaceEvidenceReview.model_validate(intervening["binding"])
    arguments.authority_id, arguments.install = "final-authority", True
    arguments.rebind_installed = rebind
    if rebind:
        arguments.source_artifact_root = arguments.source_set_hash = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        with gate.hold():
            installation = pool.submit(evidence_authority_setup.materialize, arguments)
            assert entered.wait(10), "Evidence setup did not enter the manifest owner's gate"
            with pytest.raises(FutureTimeout):
                installation.result(timeout=0.1)
            update_research_workspace_manifest(
                workspace,
                lambda current: current.with_bindings(
                    evidence_review=intervening_binding,
                    experiment_inputs=(*(current.experiment_inputs or ()), _input("another", "d")),
                ),
                gate=gate,
            )
        result = installation.result(timeout=10)
    published = read_research_workspace_manifest(workspace)
    assert [value.input_id for value in published.experiment_inputs or ()] == ["initial", "another"]
    assert published.evidence_review == ResearchWorkspaceEvidenceReview.model_validate(
        result["binding"]
    )
    assert result["installed"] and result["replaced"] == intervening_binding.relative_path
    assert (
        published.with_bindings(evidence_review=None, experiment_inputs=original.experiment_inputs)
        == original
    )
    assert first_path.read_bytes() == first_bytes


@pytest.mark.parametrize("spelling", sorted(RETIRED_SPELLINGS))
def test_a_manifest_holding_a_spelling_the_renames_retired_is_refused_by_name(
    tmp_path: Path, spelling: str
) -> None:
    """A manifest holding a spelling the renames retired is refused by name."""

    payload = ResearchWorkspaceManifest.research_only("before-renames").model_dump(mode="json")
    held = spelling + ("component" if spelling.endswith(":") else "")
    payload["strategy_artifacts"] = [{"artifact_key": held}]
    manifest = tmp_path / RESEARCH_WORKSPACE_MANIFEST_NAME
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    before = manifest.read_bytes()
    with pytest.raises(ResearchWorkspaceError) as caught:
        read_research_workspace_manifest(tmp_path)
    code = str(caught.value)
    assert code == f"research_workspace.prepared_before_renames:{spelling}"
    words = refusal_words(code)
    assert words["next_action"] == "START_A_NEW_WORKSPACE"
    assert f"`{spelling}`" in words["detail"] and "new workspace" in words["detail"]
    assert manifest.read_bytes() == before


def test_a_plan_binds_only_the_fields_it_reads() -> None:
    """a model-training publication leaves a strategy plan and a capture plan applicable;
    a change of what each reads does not."""

    before = ResearchWorkspaceManifest.research_only("ws").with_bindings(
        experiment_inputs=(_input("factor-development", "a"),)
    )
    trained = before.with_bindings(
        model_training_inputs=(
            ResearchWorkspaceModelTrainingInput(
                component_id="G2_RETURN",
                input_binding_hash="a" * 64,
                source_identity_hash="b" * 64,
                authority_relative_path="artifacts/model-training/authority.json",
                authority_hash="c" * 64,
            ),
        )
    )
    captured = before.with_bindings(experiment_inputs=(_input("factor-development", "d"),))
    for fields in (research_strategy.PLAN_FIELDS, input_capture.PLAN_FIELDS):
        assert manifest_fields_hash(trained, fields) == manifest_fields_hash(before, fields)
        assert manifest_fields_hash(captured, fields) != manifest_fields_hash(before, fields)
    with pytest.raises(
        ResearchWorkspaceError, match=re.escape("research_workspace.manifest_field_unknown")
    ):
        manifest_fields_hash(before, ("no_such_field",))


def test_a_portfolio_research_plan_binds_the_strategy_it_serves() -> None:
    """a capture publication leaves an admitted Portfolio research Task recoverable; a
    change of the strategy installation it reads does not."""

    fields = portfolio_application.PLAN_FIELDS
    before = ResearchWorkspaceManifest.research_only("ws")
    captured = before.with_bindings(experiment_inputs=(_input("factor-development", "a"),))
    # Only the field the binding reads differs; the binding is under test, not the manifest.
    reinstalled = before.model_copy(update={"strategy_installation": "NON_DEFAULT_RESEARCH"})
    assert manifest_fields_hash(captured, fields) == manifest_fields_hash(before, fields)
    assert manifest_fields_hash(reinstalled, fields) != manifest_fields_hash(before, fields)


def test_a_research_installation_holds_a_persons_activation() -> None:
    """requirement (LS1): an installed research strategy has no default, and a person's
    activation of one of its books binds the daily chain's inputs beside it; a research
    installation with a default is still refused."""

    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceArtifact,
        ResearchWorkspaceDecisionUpdate,
    )

    artifact = ResearchWorkspaceArtifact(
        artifact_key="LIFECYCLE_RESEARCH_STRATEGY_AUTHORITY",
        relative_path="artifacts/research-strategy-inputs/a/portfolio-strategy-lab/x.json",
    )
    installed = ResearchWorkspaceManifest.create(
        workspace_id="ws",
        default_strategy_package_id=None,
        default_score_source_mode=None,
        strategy_artifacts=(artifact,),
        strategy_installation="NON_DEFAULT_RESEARCH",
    )
    active = installed.with_bindings(
        decision_updates=(
            ResearchWorkspaceDecisionUpdate(
                strategy_package_id="RETURN_G6_MU_ONLY",
                strategy_package_hash="1" * 64,
                checkpoint_hash="2" * 64,
            ),
        )
    )
    assert active.decision_updates and active.strategy_installation == "NON_DEFAULT_RESEARCH"
    with pytest.raises(
        ValueError, match=re.escape("research_workspace.non_default_research_conflict")
    ):
        ResearchWorkspaceManifest.create(
            workspace_id="ws",
            default_strategy_package_id="RETURN_G6_MU_ONLY",
            default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
            strategy_artifacts=(artifact,),
            strategy_installation="NON_DEFAULT_RESEARCH",
        )
