"""The matter selection is a policy the admitted authority carries and the
request binds into its identity. The integrated selection is the one current
method (first-release integration T5); the contract keeps the retired
values -- the production reading plan (an omitted selection, absent from the
identity so every earlier request hash stands) and the candidate needs
allocation -- so what was sealed under them reads back. Through the one
application owner, a workspace still installed under a retired selection
prepares, refreshes and continues nothing: the preview names the retirement
and the re-install step, a prepare is refused by name and no Task is
admitted, never run under the integrated selection instead."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from alphalattice.control.product_host.composition.evidence_review_workspace import (
    EvidenceReviewArtifactBinding,
    EvidenceReviewWorkspaceManifest,
    live_evidence_policy,
)
from alphalattice.evidence.alternative_evidence.analysis.matters import (
    TOPIC_LANES_ALLOCATION_ID,
)
from alphalattice.evidence.alternative_evidence.analysis.packet import (
    RESIDUAL_SELECTION_RULES_ID,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    MATTER_FAMILY_CORPORATE_EVENT,
    MATTER_FAMILY_FINANCING,
    MATTER_FAMILY_LITIGATION,
    MATTER_SELECTION_CANDIDATE,
    MATTER_SELECTION_INTEGRATED,
    MATTER_SELECTION_PRODUCTION,
    AlternativeEvidenceRequest,
    MatterSelectionPolicy,
)
from alphalattice.evidence.alternative_evidence.runtime.policy import (
    REINSTALL_INTEGRATED_SELECTION,
    AdmittedEvidencePolicy,
)
from tests.alternative_evidence_desk.matter_selection_support import (
    _filings,
)
from tests.alternative_evidence_desk.review_http_support import (
    build_authority,
    build_workspace,
    start_service,
)

CANDIDATE = MatterSelectionPolicy(
    method=MATTER_SELECTION_CANDIDATE,
    families=(MATTER_FAMILY_LITIGATION, MATTER_FAMILY_CORPORATE_EVENT),
)
THREE_FAMILIES = MatterSelectionPolicy(
    method=MATTER_SELECTION_CANDIDATE,
    families=(MATTER_FAMILY_LITIGATION, MATTER_FAMILY_CORPORATE_EVENT, MATTER_FAMILY_FINANCING),
)


def _authority(
    tmp_path: Path,
    report: Any,
    policy: MatterSelectionPolicy | None,
    filings: Any = _filings,
) -> Any:
    authority = build_authority(
        tmp_path=tmp_path, report=report, extra_documents=filings, model_authority_admitted=False
    )
    return authority.__class__(
        **{
            **{f: getattr(authority, f) for f in authority.__dataclass_fields__},
            "evidence_policy": AdmittedEvidencePolicy(
                admit_model_review=False, matter_selection=policy
            ),
        }
    )


def test_the_policy_is_a_contract_of_the_request_and_the_authority() -> None:
    """requirement (4): the selection validates its families, the production
    method reads the litigation family alone, a request carries it in its
    identity (absent at the default so every earlier request hash stands),
    the workspace manifest binds it into its authority hash and the live
    policy carries it over."""

    with pytest.raises(ValueError, match="matter_families_unknown"):
        MatterSelectionPolicy(method=MATTER_SELECTION_CANDIDATE, families=("EVENTS",))
    with pytest.raises(ValueError, match="matter_selection_families_invalid"):
        MatterSelectionPolicy(
            method=MATTER_SELECTION_PRODUCTION,
            families=(MATTER_FAMILY_LITIGATION, MATTER_FAMILY_CORPORATE_EVENT),
        )
    with pytest.raises(ValueError, match="matter_families_unordered"):
        MatterSelectionPolicy(
            method=MATTER_SELECTION_CANDIDATE,
            families=(MATTER_FAMILY_FINANCING, MATTER_FAMILY_LITIGATION),
        )
    assert CANDIDATE.selection_id == "CANDIDATE_UNMET_NEEDS:LITIGATION,CORPORATE_EVENT"
    # The residual policies of section V (Gates A and C), retired in
    # section X: absent from the identity at their defaults so every
    # earlier request hash stands; a sealed request that carries one still
    # reads back and names it in the selection id, and is `retired` -- the
    # runtime refuses it by name, never dealing it under the default.
    integrated = MatterSelectionPolicy(
        method=MATTER_SELECTION_INTEGRATED, families=THREE_FAMILIES.families
    )
    assert "allocation" not in integrated.model_dump(mode="json")
    assert "residual_search" not in integrated.model_dump(mode="json")
    assert "#" not in integrated.selection_id and not integrated.retired
    context = MatterSelectionPolicy(
        method=MATTER_SELECTION_INTEGRATED,
        families=THREE_FAMILIES.families,
        allocation="CONTEXT_COMPLETE",
    )
    assert context.selection_id == integrated.selection_id + "#allocation=CONTEXT_COMPLETE"
    assert context.retired
    both = MatterSelectionPolicy(
        method=MATTER_SELECTION_INTEGRATED,
        families=THREE_FAMILIES.families,
        allocation="CONTEXT_COMPLETE",
        residual_search="GAP_DIRECTED",
    )
    assert both.selection_id.endswith("#allocation=CONTEXT_COMPLETE;residual_search=GAP_DIRECTED")
    assert MatterSelectionPolicy.model_validate_json(both.model_dump_json()) == both
    assert both.retired
    with pytest.raises(ValueError, match="matter_selection_policy_invalid"):
        MatterSelectionPolicy(
            method=MATTER_SELECTION_CANDIDATE,
            families=THREE_FAMILIES.families,
            allocation="CONTEXT_COMPLETE",
        )
    with pytest.raises(ValueError, match="matter_selection_policy_invalid"):
        MatterSelectionPolicy(
            method=MATTER_SELECTION_CANDIDATE,
            families=THREE_FAMILIES.families,
            residual_search="GAP_DIRECTED",
        )
    assert THREE_FAMILIES.selection_id == (
        "CANDIDATE_UNMET_NEEDS:LITIGATION,CORPORATE_EVENT,FINANCING"
    )
    from datetime import UTC, datetime

    cutoff = datetime(2026, 9, 1, tzinfo=UTC)
    production = AdmittedEvidencePolicy(matter_selection=None).request(
        ordered_entity_ids=("AAPL",), evidence_as_of=cutoff
    )
    candidate = AdmittedEvidencePolicy(matter_selection=CANDIDATE).request(
        ordered_entity_ids=("AAPL",), evidence_as_of=cutoff
    )
    current = AdmittedEvidencePolicy().request(ordered_entity_ids=("AAPL",), evidence_as_of=cutoff)
    # The retired values stay spellable for readback; the host's default is
    # the integrated selection.
    assert production.matter_selection is None
    assert "matter_selection" not in production.model_dump(mode="json")
    assert production.matter_selection_id == "PRODUCTION_PLAN_PREFIX:LITIGATION"
    assert candidate.matter_selection == CANDIDATE
    assert current.matter_selection == integrated
    assert len({production.request_hash, candidate.request_hash, current.request_hash}) == 3
    assert AlternativeEvidenceRequest.model_validate_json(candidate.model_dump_json()) == candidate
    binding = EvidenceReviewArtifactBinding(
        relative_path="authority/x.json", file_sha256="1" * 64, content_hash="2" * 64
    )
    plain = EvidenceReviewWorkspaceManifest.create(
        authority_id="qa",
        issuer_registry=binding,
        listing_authority=binding,
        recorded_documents=binding,
        semantic_model_relative_path="models",
        semantic_capability_hash="3" * 64,
    )
    opted = EvidenceReviewWorkspaceManifest.create(
        authority_id="qa",
        issuer_registry=binding,
        listing_authority=binding,
        recorded_documents=binding,
        semantic_model_relative_path="models",
        semantic_capability_hash="3" * 64,
        matter_selection=integrated,
    )
    assert "matter_selection" not in plain.model_dump(mode="json")
    assert opted.authority_hash != plain.authority_hash
    assert EvidenceReviewWorkspaceManifest.model_validate_json(opted.model_dump_json()) == opted
    live = live_evidence_policy(AdmittedEvidencePolicy(matter_selection=integrated))
    assert live.matter_selection == integrated
    # A retired selection is named as retired, with the re-install step, and
    # described no further.
    for retired in (None, CANDIDATE, THREE_FAMILIES, both):
        view = AdmittedEvidencePolicy(matter_selection=retired).matter_selection_view()
        assert view["retired"] is True
        assert view["refusal"] == "alternative_evidence.matter_selection_policy_retired"
        assert view["setup_help"] == REINSTALL_INTEGRATED_SELECTION
        assert "allocation_rules_id" not in view and "per_session_allowance" not in view
    view = AdmittedEvidencePolicy().matter_selection_view()
    assert view["method"] == MATTER_SELECTION_INTEGRATED and "retired" not in view
    assert view["allocation_rules_id"] == TOPIC_LANES_ALLOCATION_ID
    assert (
        AdmittedEvidencePolicy(matter_selection=THREE_FAMILIES).matter_selection_id
        == THREE_FAMILIES.selection_id
    )
    assert (
        AdmittedEvidencePolicy(matter_selection=both)
        .request(ordered_entity_ids=("AAPL",), evidence_as_of=cutoff)
        .request_hash
        != AdmittedEvidencePolicy(matter_selection=integrated)
        .request(ordered_entity_ids=("AAPL",), evidence_as_of=cutoff)
        .request_hash
    )
    # The preview discloses the one residual allocation the runtime deals
    # by its rules id alone; the retired policy fields are not reported.
    routing = AdmittedEvidencePolicy(matter_selection=integrated).matter_selection_view()["routing"]
    assert isinstance(routing, dict)
    assert routing["residual_allocation"] == {"rules_id": RESIDUAL_SELECTION_RULES_ID}
    assert "residual_search_policy" not in routing
    assert "region-scoped" not in str(routing["residual_search"])


@pytest.mark.parametrize("retired", (None, CANDIDATE), ids=("production-plan", "candidate"))
def test_a_workspace_under_a_retired_selection_is_refused_by_name(
    tmp_path: Path, retired: MatterSelectionPolicy | None
) -> None:
    """requirement (T5): a workspace still installed under the production
    plan or the candidate allocation prepares, refreshes and continues
    nothing. The preview names the retirement and the re-install step;
    a prepare and a refresh are refused by name; no Task is admitted, and
    nothing is run under the integrated selection in its place."""

    workspace, report = build_workspace(tmp_path)
    service = start_service(workspace, _authority(tmp_path, report, retired), tmp_path)
    try:
        selected = {"result_hash": service.result_hash()}
        query = "&".join(f"{k}={v}" for k, v in selected.items())
        count = len(service.registry.tasks())
        preview = service.get("/api/evidence/preview?" + query)
        assert preview["status"] == "EVIDENCE_PREREQUISITES_MISSING"
        assert preview["failure_code"] == "alternative_evidence.matter_selection_policy_retired"
        assert preview["next_action"] == "REINSTALL_THE_AUTHORITY_UNDER_THE_INTEGRATED_SELECTION"
        assert "setup_help" not in preview  # the page reads `setup` since U68
        # The rebind as it is typed, its workspace filled in (V405).
        rebind = preview["setup"]["authority"]["install"]
        assert rebind.endswith(
            "--rebind-installed --matter-selection INTEGRATED_TOPIC_ROUTING --install"
        )
        assert "materialize_evidence_cro_authority.py --workspace " in rebind
        assert preview["matter_selection"]["retired"] is True
        # Its one offered request reads the workspace where the rebind is set up; nothing is
        # prepared or run under the retired selection (f72430809, verified recovery routes).
        assert preview["next_requests"] == {"workspace": {"operation": "WORKSPACE_SHOW"}}
        prepared = service.post("/api/evidence/prepare", selected)
        assert prepared["disposition"] == "REFUSED_MATTER_SELECTION_RETIRED", prepared
        assert prepared["failure_code"] == "alternative_evidence.matter_selection_policy_retired"
        refreshed = service.post("/api/evidence-refresh", selected)
        assert refreshed["disposition"] in {
            "REFUSED_MATTER_SELECTION_RETIRED",
            "REFUSED_MODEL_AUTHORITY_NOT_ADMITTED",
        }
        assert len(service.registry.tasks()) == count
    finally:
        service.session.stop()
