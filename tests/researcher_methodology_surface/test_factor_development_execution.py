"""An authored Factor document reaches the real deterministic screening path.

The Factor Desk could already `freeze`: a document compiled to a typed Program,
and the dispatcher sealed it. It could not `run`. The deterministic primitives
that compute screening evidence existed and were tested, but nothing connected an
authored document to them, so the Desk could describe an experiment it had no way
to execute.

These cases drive the real workflow over the real workspace: a published Feature
panel, real causal execution outcomes published by the product's own publisher,
and the three deterministic primitives called directly. Nothing is routed through
``FactorResearchProductionRuntime``, which owns the current pointer and a writer
lease that a development run has no business holding.

Replay reports exact reuse, and it earns the claim: the installed Factor verifier
walks the receipt, re-derives the admitted ordered axis and the Desk Program
identity through the compiler's own functions, reads the deterministic child, and
checks the whole graph against the authority the Host resolves fresh. This case
asserted a refusal for as long as no verifier existed, which was the honest
answer then and would be a hidden regression now.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from alphalattice.control.product_host.composition.research_authoring import (
    build_research_program_workflow,
    host_resolved_factor_inventory,
)
from alphalattice.control.product_host.research_authoring.execution import (
    build_installed_desk_executors,
)
from alphalattice.control.research_program.authoring.workflow import ResearchEvidenceStore
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.factor_research.experiments.authoring import (
    FactorExperimentCompiler,
    FactorInventoryEntry,
    factor_inventory_from_panel_manifest,
)
from alphalattice.foundation.factor_research.experiments.development_evidence import (
    FACTOR_DEVELOPMENT_EVIDENCE_CATEGORY,
    FactorDevelopmentFactorIdentity,
    FactorDevelopmentInputBinding,
    FactorDevelopmentReceipt,
    FactorDevelopmentReceiptReader,
    factor_development_evidence_handle,
    verify_factor_development_receipt,
)
from alphalattice.foundation.factor_research.programs.program import (
    FactorResearchDeterministicEvidence,
)
from alphalattice.foundation.feature_engine.producers.factors.open_intraday import (
    overnight_return_factor_spec,
)
from alphalattice.foundation.feature_engine.producers.factors.registry import (
    factor_methodology_hash,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution.contracts import ActorKind
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExperimentEnvelope,
)
from tests.researcher_methodology_surface.factor_evidence_support import (
    DEVELOPMENT_EVIDENCE_POLICY,
    DEVELOPMENT_REDUNDANCY_POLICY,
    factor_development_checkpoint,
    factor_document,
)
from tests.researcher_methodology_surface.real_workspace import (
    MARKET_PROFILE_ID,
    RealRiskWorkspace,
    publish_causal_outcomes,
)

CASE_ROOT = Path(__file__).resolve().parent
PLAYPEN_ROOT = CASE_ROOT.parents[1]
_FROZEN_AT = datetime(2026, 8, 2, tzinfo=UTC)


def _document(inventory: tuple[Any, ...]) -> dict[str, Any]:
    return factor_document(inventory)


@pytest.fixture(scope="module")
def factor_outcomes(real_risk_workspace: RealRiskWorkspace) -> tuple[str, str]:
    """The causal outcomes a Factor program reads, published by the real writer.

    Published here rather than in the shared builder: Risk never reads execution
    outcomes, so every Risk case would otherwise pay for an artifact it ignores.
    """

    return publish_causal_outcomes(real_risk_workspace, at=_FROZEN_AT)


def _workflow(
    workspace: RealRiskWorkspace, outcomes: tuple[str, str], root: Path
) -> tuple[Any, tuple[Any, ...]]:
    """Compose exactly as the canonical CLI does, through Product Host.

    The executor used to be constructed here, which meant the case study was the
    only thing that could run a Factor experiment: ``build_installed_desk_executors``
    installed Risk and nothing else, so ``freeze`` worked and ``run`` had no
    executor to reach. Now the Host builds it, resolves the inventory off the
    published Panel, and finds the causal outcomes by matching listing set -- so
    what this exercises is the product path rather than a private assembly of it.
    """

    del outcomes
    inventory = host_resolved_factor_inventory(
        workspace=workspace.workspace, market_profile_id=MARKET_PROFILE_ID
    )
    document = factor_document(inventory)
    envelope = ResearchExperimentEnvelope.create(**document["experiment"])
    executors = build_installed_desk_executors(
        envelope=envelope,
        workspace=workspace.workspace,
        workspace_root=root,
        document=document,
        factor_evidence_policy=DEVELOPMENT_EVIDENCE_POLICY,
        factor_redundancy_policy=DEVELOPMENT_REDUNDANCY_POLICY,
    )
    workflow = build_research_program_workflow(
        workspace=workspace.workspace,
        workspace_root=root,
        executors=executors,
        factor_inventory=inventory,
    )
    return workflow, inventory


def test_an_authored_factor_document_reaches_real_deterministic_evidence(
    real_risk_workspace: RealRiskWorkspace,
    factor_outcomes: tuple[str, str],
    tmp_path: Path,
) -> None:
    """An authored factor document reaches real deterministic evidence."""

    workspace = real_risk_workspace
    workflow, inventory = _workflow(workspace, factor_outcomes, tmp_path)
    document = _document(inventory)

    # The recorded gap, closed: the axis carries each Factor's *method* identity,
    # read off the published panel rather than restated here, so the Desk
    # consumes what the Feature capability already computed instead of forming a
    # second opinion about what a Factor is. Not all one value -- a constant
    # would move the catalog hash when the axis changed while still being blind
    # to a Factor rewritten under a stable id.
    assert inventory
    assert all(len(entry.methodology_hash) == 64 for entry in inventory)
    assert len({entry.methodology_hash for entry in inventory}) > 1

    evidence, binding = workflow.run(document, actor_kind=ActorKind.HUMAN, actor_id="researcher")

    assert evidence.disposition == "COMPUTED"
    assert binding.actor_kind is ActorKind.HUMAN
    assert len(evidence.artifact_uris) == 1
    assert evidence.formation_sessions

    # The generic evidence points at the *receipt*, not the bare deterministic
    # child. The child says what was computed and is self-identifying; only the
    # receipt says which authored Program computed it, so only the receipt can
    # authorize a downstream Desk.
    root = tmp_path / _output_workspace(document)
    receipt, child = FactorDevelopmentReceiptReader(root).load(evidence.artifact_uris[0])
    assert receipt.program_hash == evidence.program_hash
    assert receipt.desk_program_hash == evidence.desk_program_hash
    assert receipt.method_binding_hash == evidence.method_binding_hash

    # Two axes, and the receipt keeps them apart. The context axis is the whole
    # published Factor axis, which is what the deterministic program requires:
    # redundancy and multiple-testing are properties of the full cross-section.
    # The selected axis is the subset this document asked about.
    assert receipt.context_factor_ids == tuple(entry.factor_id for entry in inventory)
    assert receipt.selected_factor_ids == tuple(document["factor"]["factor_ids"])
    assert set(receipt.selected_factor_ids) < set(receipt.context_factor_ids)

    # The child is on disk beside it, content-addressed, and was sealed under the
    # development method binding so it can never collide with a record produced
    # under current authority.
    assert child.program.factor_ids == receipt.context_factor_ids
    assert child.execution_binding_hash != "0" * 64
    stored = root / FACTOR_DEVELOPMENT_EVIDENCE_CATEGORY / f"{child.checkpoint_hash}.json"
    payload = json.loads(stored.read_text(encoding="utf-8"))
    assert payload["kind"] == "FactorResearchDeterministicEvidence"


def test_replay_walks_the_graph_and_reports_exact_reuse(
    real_risk_workspace: RealRiskWorkspace,
    factor_outcomes: tuple[str, str],
    tmp_path: Path,
) -> None:
    """Replay walks the graph and reports exact reuse."""

    workflow, inventory = _workflow(real_risk_workspace, factor_outcomes, tmp_path)
    document = _document(inventory)
    computed, _binding = workflow.run(document, actor_kind=ActorKind.HUMAN, actor_id="researcher")

    replayed, _binding = workflow.replay(
        document, actor_kind=ActorKind.HUMAN, actor_id="researcher"
    )
    assert replayed.disposition == "REUSED_EXACT"
    assert replayed.numerical_call_count == 0
    assert replayed.artifact_uris == computed.artifact_uris
    assert replayed.desk_input_binding_hash == computed.desk_input_binding_hash


def _output_workspace(document: dict[str, Any]) -> str:
    return str(document["experiment"]["output_workspace"])


def _entry(factor_id: str, *, methodology_hash: str) -> FactorInventoryEntry:
    return FactorInventoryEntry(
        factor_id=factor_id,
        implementation_hash="f" * 64,
        methodology_hash=methodology_hash,
    )


def test_a_changed_factor_recipe_moves_its_methodology_identity() -> None:
    """A changed factor recipe moves its methodology identity."""

    spec = overnight_return_factor_spec()
    kernel = "a" * 64
    baseline = factor_methodology_hash(spec, implementation_hash=kernel)

    # Same recipe, same kernel, recompiled: identity is stable, or nothing
    # downstream could ever report exact reuse.
    assert factor_methodology_hash(spec, implementation_hash=kernel) == baseline

    for changed in (
        spec.model_copy(update={"window_sessions": 42}),
        spec.model_copy(update={"lag_sessions": 2}),
        spec.model_copy(update={"return_convention": "split_adjusted_close_to_close"}),
        spec.model_copy(update={"minimum_observations": 30}),
    ):
        assert factor_methodology_hash(changed, implementation_hash=kernel) != baseline

    # And a changed kernel under an unchanged recipe still moves it, so the new
    # identity strictly contains what the old one detected.
    assert factor_methodology_hash(spec, implementation_hash="b" * 64) != baseline


def test_the_factor_catalog_hash_consumes_methodology_identity() -> None:
    """The Desk binds the method, not merely the id or the code."""

    from alphalattice.protocols.research_authoring.contracts import (
        ResolvedResearchAuthority,
    )

    document = {
        "factor": {
            "factor_ids": ["alpha"],
            "screening_policy": "BENJAMINI_YEKUTIELI_FDR",
            "redundancy_policy": "ABSOLUTE_CORRELATION_CLUSTER",
        }
    }
    envelope = ResearchExperimentEnvelope.create(
        kind="factor.screening-development",
        schema_id="research-experiment-envelope",
        data_snapshot_handle="current",
        universe_handle="us-current-index-research",
        sessions={
            "start": "2024-02-01",
            "end": "2024-03-01",
            "as_of": {"session": "2024-03-15", "phase": "OFFICIAL_CLOSE"},
        },
        budget={"maximum_candidates": 8, "maximum_numerical_calls": 5000},
        determinism={"seed": 1, "thread_limit": 1, "network_disabled": True},
        output_workspace="workspaces/x",
    )
    authority = ResolvedResearchAuthority.create(
        data_snapshot_handle="current",
        universe_handle="us-current-index-research",
        panel_snapshot_hash="c" * 64,
        panel_manifest_ref="playpen://panel/x",
        universe_revision_sha256="d" * 64,
        ordered_listing_ids=("listing-a",),
        sessions=(datetime(2024, 2, 1, tzinfo=UTC).date(),),
        source_watermark_hash="e" * 64,
    )

    def _catalog_hash(methodology: str) -> str:
        compiled = FactorExperimentCompiler(
            (_entry("alpha", methodology_hash=methodology),)
        ).compile_desk_program(envelope=envelope, document=document, authority=authority)
        return compiled.catalog_hash

    assert _catalog_hash("1" * 64) != _catalog_hash("2" * 64)


def test_the_selection_scopes_the_receipt_without_moving_the_statistics(
    real_risk_workspace: RealRiskWorkspace,
    factor_outcomes: tuple[str, str],
    tmp_path: Path,
) -> None:
    """The selection scopes the receipt without moving the statistics."""

    del factor_outcomes
    inventory = host_resolved_factor_inventory(
        workspace=real_risk_workspace.workspace, market_profile_id=MARKET_PROFILE_ID
    )
    ids = tuple(entry.factor_id for entry in inventory)
    first_handle, first_selected, first_root = factor_development_checkpoint(
        real_risk_workspace, root=tmp_path / "run-a", selected=ids[:3]
    )
    second_handle, second_selected, second_root = factor_development_checkpoint(
        real_risk_workspace, root=tmp_path / "run-b", selected=ids[3:6]
    )

    assert first_selected != second_selected
    assert first_handle != second_handle

    first, first_child = FactorDevelopmentReceiptReader(first_root).load(first_handle)
    second, second_child = FactorDevelopmentReceiptReader(second_root).load(second_handle)

    assert first.selected_factor_ids == ids[:3]
    assert second.selected_factor_ids == ids[3:6]
    # The statistical universe is untouched: same context axis, same hypothesis
    # count, same deterministic checkpoint.
    assert first.context_factor_ids == second.context_factor_ids == tuple(sorted(ids))
    assert (
        first_child.evidence_report.hypothesis_count
        == second_child.evidence_report.hypothesis_count
    )
    assert first.checkpoint_hash == second.checkpoint_hash

    # A factor nobody selected is never reported as this run's selected evidence,
    # even though the deterministic program corrected over it.
    for unselected in ids[3:6]:
        assert unselected not in first.selected_factor_ids
        assert unselected in first.context_factor_ids


def test_a_receipt_is_verified_against_re_derived_authority(
    real_risk_workspace: RealRiskWorkspace,
    factor_outcomes: tuple[str, str],
    tmp_path: Path,
) -> None:
    """A receipt is verified against re derived authority."""

    del factor_outcomes
    handle, _selected, root = factor_development_checkpoint(
        real_risk_workspace, root=tmp_path / "run"
    )
    receipt, child = FactorDevelopmentReceiptReader(root).load(handle)
    resolver = ArtifactResolver(real_risk_workspace.artifact_root)
    manifest = dict(resolver.load_feature_panel_manifest(real_risk_workspace.panel_manifest_ref))
    # The ordered typed inventory, exactly as the Host reads it off the Panel.
    # Order is authority here: rebuilding it from a mapping and re-sorting would
    # discard the Panel's published axis order.
    inventory = factor_inventory_from_panel_manifest(manifest)
    evidence = ResearchEvidenceStore(root).load_for_program(receipt.program_hash)
    assert evidence is not None

    verify_factor_development_receipt(
        receipt=receipt,
        child=child,
        evidence=evidence,
        panel_inventory=inventory,
        panel_snapshot_hash=real_risk_workspace.panel_snapshot_hash,
    )
    # The generic layer sealed the same input binding this receipt carries, so
    # "these were the inputs" is one claim rather than two.
    assert receipt.desk_input_binding_hash == evidence.desk_input_binding_hash
    assert receipt.input_binding.binding_hash == receipt.desk_input_binding_hash

    def _resealed(**overrides: object) -> FactorDevelopmentReceipt:
        """A tamper that is internally valid: everything downstream is resealed."""

        binding_fields = receipt.input_binding.model_dump(
            mode="json", exclude={"binding_hash", "kind"}
        )
        binding_fields.update(overrides)
        binding_fields["context_factors"] = tuple(
            FactorDevelopmentFactorIdentity(**value) for value in binding_fields["context_factors"]
        )
        binding_fields["selected_factor_ids"] = tuple(binding_fields["selected_factor_ids"])
        return FactorDevelopmentReceipt.create(
            program=receipt.program,
            input_binding=FactorDevelopmentInputBinding.create(**binding_fields),  # type: ignore[arg-type]
            checkpoint_hash=receipt.checkpoint_hash,
        )

    # A selection rewritten to a narrower axis, with the binding resealed around
    # it so every hash the receipt owns agrees. Only the admitted Program can
    # refuse it: it was compiled before this run and binds the selected axis into
    # its own identity.
    narrowed = _resealed(selected_factor_ids=list(receipt.selected_factor_ids[:1]))
    assert narrowed.receipt_hash != receipt.receipt_hash
    assert narrowed.desk_input_binding_hash != receipt.desk_input_binding_hash
    with pytest.raises(AuthoringError, match="selection_not_admitted"):
        verify_factor_development_receipt(
            receipt=narrowed,
            child=child,
            evidence=evidence,
            panel_inventory=inventory,
            panel_snapshot_hash=real_risk_workspace.panel_snapshot_hash,
        )

    # A forged input binding that leaves the selection alone still contradicts
    # the binding hash the generic evidence sealed.
    relabelled = _resealed(screening_policy_hash="9" * 64)
    with pytest.raises(AuthoringError, match="receipt_input_binding_mismatch"):
        verify_factor_development_receipt(
            receipt=relabelled,
            child=child,
            evidence=evidence,
            panel_inventory=inventory,
            panel_snapshot_hash=real_risk_workspace.panel_snapshot_hash,
        )

    # A child produced by another method binding, over the same Panel and the
    # same outcomes: identical statistics, identical program spec, resealed so its
    # own checkpoint hash is valid. Every identity inside it is correct. Only
    # comparing its execution binding with the receipt's method binding refuses it.
    payload = json.loads(child.model_dump_json())
    payload["execution_binding_hash"] = "7" * 64
    payload.pop("checkpoint_hash")
    # Re-parsed from JSON rather than from the dumped mapping, because the nested
    # split contracts hold enums, dates and tuples that only decode in JSON mode.
    mixed = FactorResearchDeterministicEvidence.model_validate_json(
        json.dumps({**payload, "checkpoint_hash": canonical_hash(payload)})
    )
    assert mixed.checkpoint_hash != child.checkpoint_hash
    assert mixed.program.program_hash == child.program.program_hash
    mixed_receipt = FactorDevelopmentReceipt.create(
        program=receipt.program,
        input_binding=receipt.input_binding,
        checkpoint_hash=mixed.checkpoint_hash,
    )
    with pytest.raises(AuthoringError, match="receipt_child_method_mismatch"):
        verify_factor_development_receipt(
            receipt=mixed_receipt,
            child=mixed,
            evidence=evidence,
            panel_inventory=inventory,
            panel_snapshot_hash=real_risk_workspace.panel_snapshot_hash,
        )

    # A Panel whose methodology identity has moved no longer answers for it.
    drifted = tuple(replace(entry, methodology_hash="8" * 64) for entry in inventory)
    with pytest.raises(AuthoringError, match="receipt_methodology_mismatch"):
        verify_factor_development_receipt(
            receipt=receipt,
            child=child,
            evidence=evidence,
            panel_inventory=drifted,
            panel_snapshot_hash=real_risk_workspace.panel_snapshot_hash,
        )


def test_malformed_handles_and_catalog_entries_fail_as_domain_errors() -> None:
    """Stable Factor-owned refusals, never a leaked attribute or path error."""

    for bad in ("../secrets", "", "not-a-hash", "A" * 64, "0" * 63):
        with pytest.raises(AuthoringError, match="handle_invalid"):
            factor_development_evidence_handle(bad)
    assert factor_development_evidence_handle("0" * 64) == "0" * 64

    # A summary entry that is not a mapping is a malformed panel, and reaching
    # ``.get`` on it would surface an AttributeError from inside a reader.
    for entry in ("not-a-mapping", ["also", "not"], 7):
        with pytest.raises(AuthoringError, match="panel_factor_entry_invalid"):
            factor_inventory_from_panel_manifest(
                {"safe_summary": {"factor_catalog_summary": {"alpha": entry}}}
            )


def test_a_legacy_panel_without_methodology_identity_is_readback_only() -> None:
    """A legacy panel without methodology identity is readback only."""

    legacy = {
        "safe_summary": {
            "factor_catalog_summary": {"alpha": {"implementation_hash": "f" * 64}},
        }
    }
    with pytest.raises(AuthoringError, match="methodology_identity_missing"):
        factor_inventory_from_panel_manifest(legacy)

    # A panel published by the current writer answers the question.
    current = {
        "safe_summary": {
            "factor_catalog_summary": {
                "alpha": {"implementation_hash": "f" * 64, "methodology_hash": "9" * 64}
            },
        }
    }
    assert factor_inventory_from_panel_manifest(current)[0].methodology_hash == "9" * 64
