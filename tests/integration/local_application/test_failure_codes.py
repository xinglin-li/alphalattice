"""Owner failures retain codes and causes without private diagnostics."""

from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from alphalattice.control.data_platform.maintenance.reconciliation import (
    MaintenanceReconciliationError,
)
from alphalattice.control.product_host.storage.inventory import StorageInventoryError
from alphalattice.foundation.feature_engine.catalog.research import ResearchFeatureChange
from alphalattice.interface.local_application.cli_contract import (
    ANSWER_LANGUAGE,
    refusal_words,
    worded_refusal,
)
from alphalattice.interface.local_application.failure_codes import (
    located_failure,
    owner_failure_code,
    public_failure,
    setup_failure,
    untyped_failure,
)
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument,
)


def test_a_committee_document_refusal_keeps_its_members_own_read() -> None:
    """An invalid member message offers that member's keyed floor read."""
    member = dict(
        update_task_id="00000000-0000-0000-0000-000000000001",
        committee_role="ALPHA",
        committee_key="a" * 32,
    )
    request = dict(
        operation="COMMITTEE_SUBMIT",
        **member,
        committee_message=dict(
            kind="STANCE",
            text="Support.",
            positions=dict.fromkeys((f"T{i}" for i in range(1, 34)), "SUPPORT"),
        ),
    )
    with pytest.raises(ValidationError) as caught:
        PortfolioResearchRequestDocument.model_validate(request)
    answer = located_failure(caught.value, "local_client.request_invalid", request=request)
    assert answer["next_requests"]["valid_values"] == {"operation": "COMMITTEE_READ", **member}


def test_an_install_missing_its_setup_offers_its_own_form() -> None:
    """Model-level required fields stay named in metadata and their original setup offer."""
    request = {"operation": "EVIDENCE_INSTALL"}
    with pytest.raises(ValidationError) as caught:
        PortfolioResearchRequestDocument.model_validate(request)
    answer = located_failure(
        caught.value,
        "local_client.request_invalid",
        request=request,
        schema=PortfolioResearchRequestDocument.model_json_schema(),
    )
    assert answer["fields"] == [["evidence_setup"]]
    assert [item["field"] for item in answer["error_info"]["field_violations"]] == [
        ["evidence_setup"]
    ]
    read = answer["next_requests"]["valid_values"]
    assert read == {"operation": "OPERATION_LIST", "schema_operation": "EVIDENCE_INSTALL"}
    PortfolioResearchRequestDocument.model_validate(read)


def test_a_contract_failure_answers_one_located_shape() -> None:
    """Structured failures keep typed paths and bilingual words without private inputs."""

    class Document(BaseModel):
        model_config = ConfigDict(extra="forbid")
        top_k: int
        name: str

        @field_validator("name")
        @classmethod
        def _known(cls, value: str) -> str:
            if value != "known":
                raise ValueError("portfolio_research.name_unknown")
            return value

    with pytest.raises(ValidationError) as caught:
        Document.model_validate({"top_k": "private-input-marker", "name": "x", "extra": 1})
    answer = located_failure(caught.value, "portfolio_research.refused")
    assert answer["failure_code"] == "portfolio_research.refused"
    assert answer["fields"] == [["top_k"], ["name"], ["extra"]]
    assert answer["reasons"] == {
        "top_k": "int_parsing",
        "name": "portfolio_research.name_unknown",
        "extra": "extra_forbidden",
    }
    assert "name: portfolio_research.name_unknown" in str(answer["message"])
    assert "private-input-marker" not in json.dumps(answer)
    assert "error_info" not in answer  # Unregistered legacy reasons keep their old shape.
    located = located_failure(
        caught.value, "local_client.request_invalid", schema=Document.model_json_schema()
    )
    for language in ("en", "zh"):
        token = ANSWER_LANGUAGE.set(language)
        try:
            info = worded_refusal(located)["error_info"]
            assert (info["category"], info["reason"], info["domain"]) == (
                "INVALID_ARGUMENT",
                "local_client.request_invalid",
                "local_client",
            )
            assert [item["field"] for item in info["field_violations"]] == [
                ["top_k"],
                ["name"],
                ["<unrecognized>"],
            ]
            assert "private-input-marker" not in json.dumps(info) and "extra" not in json.dumps(
                info
            )
            assert any("一" <= char <= "鿿" for char in info["message"]) == (language == "zh")
            job = worded_refusal({"failure_code": "local_web.detached_breakaway_denied"})[
                "error_info"
            ]
            assert (job["domain"], job["message_id"]) == (
                "alphalattice.control.product_host.composition",
                "host.start.breakaway_denied",
            )
        finally:
            ANSWER_LANGUAGE.reset(token)

    class Value(BaseModel):
        number: int

    class NestedDocument(Document):
        items: list[dict[str | int, Value]]

    with pytest.raises(ValidationError) as nested:
        NestedDocument.model_validate(
            {
                "top_k": 1,
                "name": "known",
                "items": [
                    {
                        "private-map-key": {"number": "private-value"},
                        42: {"number": "private-value"},
                    }
                ],
            }
        )
    safe = located_failure(
        nested.value, "local_client.request_invalid", schema=NestedDocument.model_json_schema()
    )["error_info"]
    assert safe["field_violations"][0]["field"] == ["items", 0, "<unrecognized>", "number"]
    assert safe["field_violations"][1]["field"] == ["items", 0, "<unrecognized>", "number"]
    assert "private-map-key" not in json.dumps(safe) and "private-value" not in json.dumps(safe)
    for notice in (
        "native_bridge.bound_in_workspace",
        "risk_research.date_risk_unavailable",
        "risk_research.positions_outside_the_risk_axis",
        "risk_research.return_surface_short_of_the_date",
        "sector.recoverable_exclusion_applied",
    ):
        assert "error_info" not in worded_refusal({"failure_code": notice})
    assert located_failure(ValueError("portfolio_research.top_k_outside"), "x.refused") == {
        "failure_code": "portfolio_research.top_k_outside"
    }
    specification = {
        "factor_id": "example_factor",
        "family": "liquidity",
        "formula_ref": "factor.example_factor.v1",
        "formula": "an example",
        "window_sessions": 5,
        "lag_sessions": 0,
        "return_convention": "dimensionless",
        "required_fields": ["volume_raw", "close_raw"],
        "literature_sources": ["urn:example"],
        "minimum_observations": 5,
        "absolute_tolerance": 0,
        "relative_tolerance": 0,
        "track": "model",
    }
    document = {
        "input_binding_hash": "0" * 64,
        "base_revision_hash": "0" * 64,
        "edits": [
            {"operation": "CREATE", "factor_id": "example_factor", "specification": specification}
        ],
        "reason": "an example",
    }
    with pytest.raises(ValidationError) as caught:
        ResearchFeatureChange.model_validate_json(json.dumps(document))
    answer = located_failure(caught.value, "feature_research.document_invalid")
    assert answer["reasons"] == {
        "edits.0.specification.required_fields": "factor_spec.values_sorted_unique"
    }
    assert "sorted and without repeats" in str(answer["message"])
    assert "volume_raw" not in json.dumps(answer)
    crash = public_failure(
        IndexError("single positional indexer is out-of-bounds"), "local_web.handler_failed"
    )
    assert crash == "local_web.handler_failed:IndexError:503ea14e" and untyped_failure(crash)
    assert not untyped_failure("goal.not_found")
    assert not untyped_failure("feature_extension.extensions_page_out_of_range:2 of 1")
    words = refusal_words(crash)
    assert "503ea14e" not in words["next_action"] and "IndexError:503ea14e" in words["detail"]
    code = "storage.managed_capacity_exceeded"
    marker = "private synthetic payload /never/serve with token=not-a-secret"
    error = StorageInventoryError(code, marker)
    assert owner_failure_code(error) == public_failure(error, "local_web.handler_failed") == code
    assert setup_failure(error)["causes"] == [
        {"kind": "PRODUCT", "code": code, "failure_code": code}
    ]

    class Document(BaseModel):
        selected: int

        @field_validator("selected")
        @classmethod
        def admitted(cls, selected: int) -> int:
            raise MaintenanceReconciliationError(marker, failure_code=code)

    with pytest.raises(ValidationError) as seen:
        Document(selected=1)
    located = located_failure(seen.value, "local_application.request_refused")
    assert located["fields"] == [["selected"]]
    assert located["reasons"] == {"selected": code}
    assert marker not in json.dumps(located)
    assert (
        public_failure(seen.value, "local_application.request_refused")
        == f"local_application.request_refused:selected={code}"
    )
    for head in (
        "task_control.queue_full",
        "model_store.download_interrupted",
        "alternative_evidence.table_view_correspondence_unproved",
        "alternative_evidence.topic_coverage_absent",
    ):
        cause = RuntimeError(f"{head}: {marker}; unsafe diagnostic")
        assert (
            owner_failure_code(cause) == public_failure(cause, "local_web.handler_failed") == head
        )
    for fault in (
        RuntimeError("ordinary"),
        RuntimeError(marker),
        RuntimeError(code, marker),
        StorageInventoryError(marker, marker),
        HTTPError(None, 503, marker, None, None),
    ):
        assert owner_failure_code(fault) is None
        shown = public_failure(fault, "local_web.handler_failed")
        assert untyped_failure(shown) and marker not in shown
