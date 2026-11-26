"""Foundation names preserve sealed identity checks without a parent study read."""

import json
from collections import Counter
from pathlib import Path

import pytest

from alphalattice.control.product_host.composition.foundation_summary import read_foundation_summary
from alphalattice.foundation.factor_research.publication.artifacts import (
    FactorResearchArtifactStore,
)
from alphalattice.foundation.research_foundation.contracts import (
    ResearchDeskExecutionOutcomeRef,
    ResearchFoundationAdmission,
    ResearchFoundationBinding,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import AuthoringError


def _sealed_metadata(workspace):
    outcome = ResearchDeskExecutionOutcomeRef(
        research_cadence="DAILY",
        snapshot_hash="1" * 64,
        schedule_hash="2" * 64,
        development_content_hash="3" * 64,
        sealed_holdout_content_hash="4" * 64,
        marker_hash="5" * 64,
        market_as_of="2024-08-12",
        data_validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
    )
    foundation_values = {
        "research_cadence": outcome.research_cadence,
        "feature_panel_snapshot_hash": "6" * 64,
        "factor_training_outcome_snapshot_hash": "7" * 64,
        "factor_screening_result_hash": "8" * 64,
        "factor_candidate_slate_hash": "9" * 64,
        "research_desk_factor_input_hash": "a" * 64,
        "execution_outcome": outcome,
        "ordered_factor_ids": ("fixture-factor",),
    }
    draft = ResearchFoundationBinding.model_construct(**foundation_values, foundation_hash="")
    foundation = ResearchFoundationBinding(
        **foundation_values,
        foundation_hash=canonical_hash(
            draft.model_dump(mode="json", exclude={"foundation_hash"}, exclude_none=True)
        ),
    )
    admission_values = {
        "foundation": foundation,
        "factor_task_id": "11111111-2222-4333-8444-555555555555",
        "factor_receipt_hash": "b" * 64,
        "curation_receipt_hash": "c" * 64,
        "input_id": "fixture-factor-input",
        "input_binding_hash": "d" * 64,
        "factor_input_binding_hash": "e" * 64,
    }
    draft = ResearchFoundationAdmission.model_construct(**admission_values, admission_hash="")
    admission = ResearchFoundationAdmission(
        **admission_values,
        admission_hash=canonical_hash(draft.model_dump(mode="json", exclude={"admission_hash"})),
    )
    store = FactorResearchArtifactStore(workspace / "artifacts")
    store.publish_research_foundation(
        payload=foundation.model_dump(mode="json", exclude_none=True),
        foundation_hash=foundation.foundation_hash,
    )
    store.publish_foundation_admission(
        payload=admission.model_dump(mode="json"), admission_hash=admission.admission_hash
    )
    return admission, (
        store.root / "research-desk/foundation-admissions" / (admission.admission_hash + ".json"),
        store.root / "research-desk/foundations" / (foundation.foundation_hash + ".json"),
    )


def test_summary_opens_only_exact_sealed_metadata_without_parent_evidence(tmp_path, monkeypatch):
    admission, paths = _sealed_metadata(tmp_path)
    allowed = {p.resolve() for p in paths}
    opened = Counter()
    actual_open = Path.open

    def only_metadata(self, *args, **kwargs):
        exact = self.resolve()
        assert exact in allowed, "summary opened a parent study, input or other descendant"
        opened[exact] += 1
        return actual_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", only_metadata)
    # The workspace contains no Task registry, parent evidence or captured input.
    body = read_foundation_summary(tmp_path, admission.admission_hash)
    assert opened == Counter({p: 1 for p in allowed})
    assert body["status"] == "FOUNDATION_SUMMARY"
    assert body["foundation_admission_hash"] == admission.admission_hash
    assert body["foundation_hash"] == admission.foundation.foundation_hash
    assert body["ordered_factor_ids"] == ["fixture-factor"]
    assert body["input_id"] == admission.input_id
    assert body["input_binding_hash"] == admission.input_binding_hash
    assert body["market_as_of"] == "2024-08-12"
    assert body["evidence_verification"] == "METADATA_ONLY_SOURCE_GRAPH_NOT_CHECKED"
    assert body["next_requests"] == {
        "verify": {
            "operation": "EXPERIMENT_FOUNDATION_READBACK",
            "foundation_admission_hash": admission.admission_hash,
        }
    }
    assert "standing" not in body and "admission" not in body


@pytest.mark.parametrize("record", [0, 1], ids=["admission-seal", "foundation-binding"])
def test_summary_refuses_tampered_sealed_metadata(tmp_path, record):
    admission, paths = _sealed_metadata(tmp_path)
    payload = json.loads(paths[record].read_text(encoding="utf-8"))
    if record == 0:
        payload["input_id"] = "changed-without-resealing"
    else:
        payload["ordered_factor_ids"] = ["changed-without-resealing"]
    paths[record].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        AuthoringError, match=r"^research_foundation\.admission_identity_invalid(?::|$)"
    ):
        read_foundation_summary(tmp_path, admission.admission_hash)


@pytest.mark.parametrize("record", [0, 1], ids=["admission-missing", "binding-missing"])
def test_summary_refuses_missing_metadata_without_selecting_another_admission(tmp_path, record):
    admission, paths = _sealed_metadata(tmp_path)
    paths[record].unlink()
    with pytest.raises(
        AuthoringError, match=r"^research_foundation\.admission_artifact_unavailable$"
    ):
        read_foundation_summary(tmp_path, admission.admission_hash)


def test_summary_keeps_publication_owners_binding_contradiction_refusal(tmp_path, monkeypatch):
    from alphalattice.control.product_host.composition.plain_refusals import refused
    from alphalattice.interface.local_application.cli_contract import worded_refusal

    admission, _paths = _sealed_metadata(tmp_path)
    actual_load = FactorResearchArtifactStore.load_research_foundation

    def contradictory_binding(self, uri):
        body = actual_load(self, uri)
        # A fault at the binding port must still meet publication's equality check.
        return {**body, "ordered_factor_ids": ["contradictory-binding"]}

    monkeypatch.setattr(
        FactorResearchArtifactStore, "load_research_foundation", contradictory_binding
    )
    with pytest.raises(
        AuthoringError, match=r"^research_foundation\.admission_binding_mismatch$"
    ) as caught:
        read_foundation_summary(tmp_path, admission.admission_hash)

    answer = worded_refusal(refused(str(caught.value)))
    assert answer["status"] == "REFUSED"
    assert answer["failure_code"] == str(caught.value)
    assert answer["next_action"] == "RESTORE_THE_WORKSPACE_FROM_A_BACKUP"
    detail = answer["detail"]
    assert "backup list --view full" in detail
    assert (
        "alphalattice backup restore --dir <new directory> "
        "--generation <verified generation hash> --workspace-id <workspace id> "
        "--root <backup root>"
    ) in detail
    assert "new or empty directory outside the backup root and the original workspace" in detail
    assert "open the restored workspace and read the admission there" in detail


@pytest.mark.parametrize("identity", [None, "", "a" * 63, "A" * 64, "../metadata"])
def test_summary_requires_an_exact_admission_hash_before_any_store_read(
    tmp_path, monkeypatch, identity
):
    from alphalattice.control.product_host.composition import foundation_summary

    def forbidden_store(*_args, **_kwargs):
        pytest.fail("an invalid selector reached the sealed store")

    monkeypatch.setattr(foundation_summary, "read_foundation_admission", forbidden_store)
    with pytest.raises(
        AuthoringError, match=r"^research_foundation\.admission_identity_invalid(?::|$)"
    ):
        read_foundation_summary(tmp_path, identity)


def test_summary_unknown_exact_hash_never_falls_back_to_existing_admission(tmp_path):
    _sealed_metadata(tmp_path)
    with pytest.raises(
        AuthoringError, match=r"^research_foundation\.admission_artifact_unavailable$"
    ):
        read_foundation_summary(tmp_path, "f" * 64)
