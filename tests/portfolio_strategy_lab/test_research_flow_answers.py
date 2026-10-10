"""Research flow answers through their public owners and Host."""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from alphalattice.control.product_host.composition.plain_refusals import refused
from alphalattice.control.task_control.queue import write_queue_setting
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument as PortfolioResearchAgentRequest,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioResearchSpec,
)
from tests.portfolio_strategy_lab.local_web_support import (
    _agent,
    _json,
    _request,
    _resolved,
    _Resolver,
)

ALPHA, BETA = ("ALPHA_BOOK", "BETA_BOOK")


def test_a_trial_the_owner_refuses_to_compare_claims_no_change(tmp_path):
    "a trial whose two Alpha studies the owner refused to compare (their"
    from alphalattice.control.product_host.composition.feature_trials import (
        FeatureTrial,
        FeatureTrials,
    )

    factor, baseline, tried = (uuid4(), uuid4(), uuid4())
    metrics = {baseline: 0.01, tried: 0.014}

    class Owner:
        def readback(self, task_id):
            if task_id == factor:
                return {
                    "result": {
                        "evidence_report": {"items": [{"factor_id": "formula_x"}]},
                        "redundancy_structure": {"pair_evidence": []},
                    }
                }
            return {"receipt": {"child_lineage": [{"candidate_id": "c1"}]}}

        def summary(self, task_id):
            return {
                "result": {"candidates": [{"candidate_id": "c1", "mean_rank_ic": metrics[task_id]}]}
            }

        def operate(self, request, *, caller):
            raise ValueError("alpha_research.saved_comparison_score_support_mismatch")

    trials = FeatureTrials(
        workspace=tmp_path,
        experiments=Owner(),
        feature_builds=None,
        dispatcher=None,
        clock=lambda: datetime(2026, 9, 30, tzinfo=UTC),
    )
    trial = FeatureTrial(
        trial_id="a" * 64,
        feature_plan_hash="b" * 64,
        baseline_task_id=str(baseline),
        baseline_alpha_task_id=str(baseline),
        caller="HUMAN",
        requested_at=datetime(2026, 9, 30, tzinfo=UTC),
        state="COMPLETED",
        outcome="COMPARED",
        steps={"FACTOR_STUDY": str(factor), "ALPHA_STUDY": str(tried)},
        feature_factor_ids=("formula_x",),
    )
    alpha = trials._comparison(trial)["alpha_without_and_with"]
    assert alpha["standing"] == "NOT_COMPARED" and "change" not in alpha
    refused = alpha["owner_comparison"]
    assert refused["failure_code"] == "alpha_research.saved_comparison_score_support_mismatch"
    assert "not compared" in refused["detail"]
    assert (alpha["without"]["mean_rank_ic"], alpha["with"]["mean_rank_ic"]) == (0.01, 0.014)


def test_trial_and_feature_listings_keep_readable_siblings_with_damaged_records(
    tmp_path, monkeypatch
):
    "one corrupt Feature trial hid its readable siblings, and"
    from alphalattice.control.product_host.composition.feature_trials import (
        FeatureTrial,
        FeatureTrials,
    )
    from alphalattice.control.product_host.research_authoring.feature_extensions import (
        FeatureExtensions,
    )

    class Builds:
        def definitions(self):
            return SimpleNamespace(store=SimpleNamespace(root=tmp_path / "plans"))

    trials = FeatureTrials(
        workspace=tmp_path,
        experiments=None,
        feature_builds=Builds(),
        dispatcher=None,
        clock=lambda: datetime(2026, 10, 4, tzinfo=UTC),
    )
    trials.root.mkdir(parents=True)
    healthy = FeatureTrial(
        trial_id="a" * 64,
        feature_plan_hash="d" * 64,
        baseline_task_id="task-1",
        baseline_alpha_task_id="task-1",
        caller="HUMAN",
        requested_at=datetime(2026, 10, 1, tzinfo=UTC),
        state="RUNNING",
        feature_factor_ids=("formula_x",),
    )
    healthy_path = trials.root / f"{healthy.trial_id}.json"
    healthy_path.write_text(healthy.model_dump_json(), encoding="utf-8")
    malformed_path = trials.root / f"{'b' * 64}.json"
    malformed_path.write_text("{", encoding="utf-8")
    unreadable_path = trials.root / f"{'c' * 64}.json"
    unreadable_path.write_text("{}", encoding="utf-8")
    read_bytes = Path.read_bytes

    def read_one_fails(path):
        if path == unreadable_path:
            raise OSError("the local record could not be read")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_one_fails)
    listed = trials.listing()
    assert [row["feature_trial_id"] for row in listed["trials"]] == [healthy.trial_id]
    assert [row["feature_trial_id"] for row in listed["damaged"]] == [
        malformed_path.stem,
        unreadable_path.stem,
    ]
    for row in listed["damaged"]:
        assert row["status"] == "REFUSED"
        assert row["failure_code"] == "feature_trial.record_damaged"
        assert row["detail"]
        assert row["next_requests"]["trials"]["operation"] == "FEATURE_TRIALS"
        assert row["next_requests"]["storage"]["operation"] == "STORAGE_READBACK"
    refusal = refused("feature_trial.record_damaged")
    for row in listed["damaged"]:
        assert row == {
            **refusal,
            "feature_trial_id": row["feature_trial_id"],
            "next_requests": {
                **refusal.get("next_requests", {}),
                "storage": {"operation": "STORAGE_READBACK"},
            },
        }
    extensions = FeatureExtensions(
        tmp_path, lambda: datetime(2026, 10, 4, tzinfo=UTC), trials=trials, goals=None
    )
    extension_listing = extensions.listing()
    assert extension_listing["factors"] == []
    assert extension_listing["damaged"] == listed["damaged"]


def test_a_trial_offers_its_own_read_and_once_completed_each_review():
    "Each trial offers its own bound read and its completed review."
    from alphalattice.control.product_host.composition.feature_trials import (
        FeatureTrial,
        trial_requests,
    )

    trial = FeatureTrial(
        trial_id="a" * 64,
        feature_plan_hash="b" * 64,
        baseline_task_id="c",
        baseline_alpha_task_id="c",
        caller="HUMAN",
        requested_at=datetime(2026, 10, 1, tzinfo=UTC),
        feature_factor_ids=("formula_x",),
    )
    show = {"operation": "FEATURE_TRIAL_READBACK", "feature_trial_id": "a" * 64}
    assert trial_requests(trial) == {"show": show}
    done = trial.model_copy(update={"state": "COMPLETED"})
    assert trial_requests(done) == {
        "show": show,
        "review": {
            "operation": "FEATURE_REVIEW",
            "feature_plan_hash": "b" * 64,
            "feature_factor_id": "formula_x",
        },
    }
    two = done.model_copy(update={"feature_factor_ids": ("formula_x", "formula_y")})
    assert set(trial_requests(two)) == {"show", "review:formula_x", "review:formula_y"}


def test_a_review_packets_trial_says_whether_its_alpha_studies_were_compared(tmp_path):
    "requirement (, ): the packet a person activates from carries each trial's Alpha"
    from alphalattice.control.product_host.composition.feature_trials import FeatureTrial
    from alphalattice.control.product_host.research_authoring.feature_extensions import (
        FeatureExtensions,
    )

    refused = {
        "status": "REFUSED",
        "failure_code": "alpha_research.saved_comparison_score_support_mismatch",
        "detail": "The two Alpha studies scored different sessions, names or available rows.",
    }

    class Ledger:
        def __init__(self, alpha):
            self.alpha = alpha

        def readback(self, trial_id):
            feature = {"out_of_sample_evidence": [], "correlation_with_present_factors": []}
            return {"comparison": {"feature": feature, "alpha_without_and_with": self.alpha}}

    trial = FeatureTrial(
        trial_id="a" * 64,
        feature_plan_hash="b" * 64,
        baseline_task_id="t-1",
        baseline_alpha_task_id="t-1",
        caller="HUMAN",
        requested_at=datetime(2026, 9, 30, tzinfo=UTC),
        state="COMPLETED",
        outcome="COMPARED",
        feature_factor_ids=("formula_x",),
    )

    def packet_row(alpha):
        extensions = FeatureExtensions(
            tmp_path, lambda: datetime(2026, 9, 30, tzinfo=UTC), trials=Ledger(alpha), goals=None
        )
        return extensions._trial(trial, "formula_x")

    not_compared = packet_row({"standing": "NOT_COMPARED", "owner_comparison": refused})
    assert (not_compared["alpha_standing"], not_compared["alpha_change"]) == ("NOT_COMPARED", None)
    assert not_compared["alpha_refusal"] == {
        "failure_code": refused["failure_code"],
        "detail": refused["detail"],
    }
    compared = packet_row(
        {"standing": "COMPARED", "owner_comparison": {"status": "COMPARABLE"}, "change": {"x": 1.0}}
    )
    assert (compared["alpha_standing"], compared["alpha_change"]) == ("COMPARED", {"x": 1.0})
    assert "alpha_refusal" not in compared


def test_every_result_states_one_standing_from_its_owners_marks():
    "Every result states one standing from its owners' marks."
    from alphalattice.control.product_host.composition.result_standing import (
        ResultStanding,
        packet_standing,
        study_standing,
        trial_standing,
    )

    def marks(standing):
        return (
            standing.comparison,
            standing.execution,
            standing.contract,
            standing.evidence,
            standing.activation,
        )

    published = {"status": "EXPERIMENT_PUBLISHED", "research_lane": "PROMOTION"}
    study = study_standing(published)
    assert marks(study) == (
        "NOT_APPLICABLE",
        "SUCCEEDED",
        "PASSED",
        "DEVELOPMENT",
        "NOT_ACTIVATABLE",
    )
    assert study.reasons == {} and len(study.statements) == 5
    superseded = "research_lane.sample_scheme_superseded"
    explored = study_standing(
        {
            **published,
            "research_lane": "EXPLORATION",
            "method_standing": "NOT_CURRENT",
            "method_refusal": superseded,
        }
    )
    assert (explored.contract, explored.evidence) == ("NOT_CURRENT", "EXPLORATION")
    assert explored.reasons == {"contract": superseded} and superseded in explored.statements[2]
    for disposition, evidence in (
        ("CURRENT_ALPHA_CANDIDATE_SET_READY", "QUALIFIED"),
        ("NO_STABLE_CURRENT_ALPHA_MODEL", "NOT_QUALIFIED"),
    ):
        qualified = {**published, "alpha_qualification": {"disposition": disposition}}
        assert study_standing(qualified).evidence == evidence
    assert study_standing({"status": "RUNNING"}).execution == "RUNNING"
    blocked = study_standing({"status": "BLOCKED", "failure_code": "data.input_changed"})
    assert (blocked.execution, blocked.evidence) == ("STOPPED", "NONE")
    assert blocked.reasons == {"execution": "data.input_changed"}
    refused = "alpha_research.saved_comparison_score_support_mismatch"
    not_compared = trial_standing(
        state="COMPLETED",
        outcome="COMPARED",
        stopped=None,
        comparison={"standing": "NOT_COMPARED", "owner_comparison": {"failure_code": refused}},
    )
    assert marks(not_compared)[:2] == ("NOT_COMPARED", "SUCCEEDED")
    assert not_compared.reasons == {"comparison": refused}
    assert (
        not_compared.statements[0].startswith("Not compared")
        and "no change" in not_compared.statements[0]
    )
    compared = trial_standing(
        state="COMPLETED", outcome="COMPARED", stopped=None, comparison={"standing": "COMPARED"}
    )
    assert (compared.comparison, compared.evidence) == ("COMPARED", "DEVELOPMENT")
    screened = trial_standing(
        state="COMPLETED",
        outcome="FEATURE_NOT_ADMITTED_BY_SCREENING",
        stopped=None,
        comparison=None,
    )
    assert (screened.comparison, screened.evidence) == ("NOT_APPLICABLE", "SCREENED_OUT")
    stopped = trial_standing(
        state="STOPPED",
        outcome=None,
        stopped={"step": "ALPHA_STUDY", "failure_code": "task_blocked"},
        comparison=None,
    )
    assert (stopped.execution, stopped.reasons) == ("STOPPED", {"execution": "task_blocked"})
    rows = [
        {
            "state": "COMPLETED",
            "outcome": "COMPARED",
            "alpha_standing": "NOT_COMPARED",
            "alpha_refusal": {"failure_code": refused, "detail": "..."},
        },
        {"state": "STOPPED", "stopped": {"step": "FACTOR_STUDY", "failure_code": "x"}},
    ]
    offered = packet_standing(contract={"status": "PASSED"}, trials=rows, active=False, held=None)
    assert marks(offered) == (
        "NOT_COMPARED",
        "SUCCEEDED",
        "PASSED",
        "DEVELOPMENT",
        "A_PERSON_MAY_ACTIVATE",
    )
    failed = packet_standing(
        contract={"status": "FAILED", "failure_code": "goldens_outside_tolerance"},
        trials=[],
        active=False,
        held="feature_extension.contract_failed:goldens_outside_tolerance",
    )
    assert marks(failed) == ("NOT_APPLICABLE", "NOT_STARTED", "FAILED", "NONE", "HELD")
    assert failed.reasons == {
        "contract": "goldens_outside_tolerance",
        "activation": "feature_extension.contract_failed:goldens_outside_tolerance",
    }
    active = packet_standing(contract={"status": "PASSED"}, trials=rows, active=True, held=None)
    assert active.activation == "ACTIVE" and "activation" not in active.reasons
    for reasons in ({}, {"comparison": refused, "evidence": "x"}):
        with pytest.raises(ValueError, match=r"result_standing.reasons_mismatch"):
            ResultStanding.of(
                comparison="NOT_COMPARED",
                execution="SUCCEEDED",
                contract="NOT_CHECKED",
                evidence="DEVELOPMENT",
                activation="NOT_ACTIVATABLE",
                reasons=reasons,
            )


def test_each_flow_names_its_prerequisites_and_the_way_on(tmp_path):
    "Each flow names its prerequisites and the way on."
    from alphalattice.control.product_host.composition.research_prerequisites import (
        holdings,
        prerequisites,
    )
    from alphalattice.foundation.factor_research.experiments.development_evidence import (
        FACTOR_DEVELOPMENT_CURATION_CATEGORY,
    )

    binding = "b" * 64
    selector = {"research_input_id": "in-1", "input_binding_hash": binding}

    def of(flow, **held):
        return prerequisites(flow, held, input_id="in-1", binding_hash=binding)

    factor = of("FACTOR_STUDY")
    assert (factor["needs"], factor["missing"]) == ([], [])
    from alphalattice.interface.local_application.cli_contract import choices

    for flow, kind in (
        ("FACTOR_STUDY", "factor.screening-development"),
        ("RISK_STUDY", "risk.covariance-development"),
    ):
        offered = of(flow)["next_requests"]
        assert offered == {
            "controls": {"operation": "EXPERIMENT_CONTROLS", **selector, "experiment_kind": kind},
            "plan": {"operation": "EXPERIMENT_PLAN", **selector, "experiment_document": None},
        }
        assert choices(offered["plan"]) == ["experiment_document"]
    nothing = of("ALPHA_STUDY")
    assert nothing["missing"] == ["CURATED_FACTOR_STUDY"]
    assert nothing["next_requests"] == {
        "factor": {
            "operation": "EXPERIMENT_CONTROLS",
            **selector,
            "experiment_kind": "factor.screening-development",
        }
    }
    assert nothing["flow"] == "ALPHA_STUDY"
    assert nothing["needs"] == ["CURATED_FACTOR_STUDY"] and nothing["detail"]
    fresh = {"task_id": "f-2", "curation_receipt_hashes": []}
    uncurated = of("ALPHA_STUDY", FACTOR_STUDY=[fresh])
    assert uncurated["next_requests"] == {
        "curation": {"operation": "EXPERIMENT_CURATION", "task_id": "f-2"}
    }
    once = {"task_id": "f-1", "curation_receipt_hashes": ["r1"]}
    curated = of("ALPHA_STUDY", FACTOR_STUDY=[fresh, once], CURATED_FACTOR_STUDY=[once])
    assert curated["missing"] == []
    assert curated["next_requests"] == {
        "handoff": {
            "operation": "EXPERIMENT_HANDOFF_PREVIEW",
            "task_id": "f-1",
            "curation_receipt_hash": "r1",
        }
    }
    two = {"task_id": "f-1", "curation_receipt_hashes": ["r1", "r2"]}
    twice = of("ALPHA_STUDY", FACTOR_STUDY=[two], CURATED_FACTOR_STUDY=[two])
    assert twice["next_requests"]["handoff"]["curation_receipt_hash"] is None
    alpha = [{"task_id": "a-1", "factor_task_id": "f-1"}]
    book = of("BOOK", ALPHA_STUDY=alpha)
    assert book["missing"] == []
    assert book["next_requests"] == {
        "portfolio-draft": {
            "operation": "EXPERIMENT_PORTFOLIO_DRAFT",
            "task_id": "a-1",
            "candidate_id": None,
        },
        "risk": {
            "operation": "EXPERIMENT_CONTROLS",
            **selector,
            "experiment_kind": "risk.covariance-development",
        },
    }
    assert (
        "risk"
        not in of("BOOK", ALPHA_STUDY=alpha, RISK_STUDY=[{"task_id": "r-1"}])["next_requests"]
    )
    trial = of("FEATURE_TRIAL", FACTOR_STUDY=[fresh])
    assert trial["missing"] == ["ALPHA_STUDY"] and set(trial["next_requests"]) == {"curation"}
    assert of("FEATURE_TRIAL", ALPHA_STUDY=alpha)["next_requests"] == {}
    review = of("BOOK_REVIEW", BOOK=[{"task_id": "p-1", "alpha_task_id": "a-1"}])
    assert review["next_requests"] == {
        "book": {"operation": "EXPERIMENT_READBACK", "task_id": "p-1"}
    }
    start = datetime(2026, 9, 30, tzinfo=UTC)
    folder = "research-experiments/factor-one"
    receipts = tmp_path / folder / FACTOR_DEVELOPMENT_CURATION_CATEGORY / "checkpoint"
    receipts.mkdir(parents=True)
    (receipts / ("c" * 64 + ".json")).write_text("{}", encoding="utf-8")

    def study(
        n,
        kind,
        *,
        input_hash=binding,
        alpha_source=None,
        portfolio_source=None,
        output_workspace=folder,
    ):
        task = SimpleNamespace(task_id=f"t-{n}", admitted_at=start + timedelta(minutes=n))
        plan = SimpleNamespace(
            binding=SimpleNamespace(binding_hash=input_hash),
            program=SimpleNamespace(kind=kind),
            alpha_source=alpha_source,
            portfolio_source=portfolio_source,
            document={"experiment": {"output_workspace": output_workspace}},
        )
        return (task, plan)

    studies = [
        study(1, "factor.screening-development"),
        study(2, "factor.screening-development", input_hash="d" * 64),
        study(3, "alpha.model-development", alpha_source=SimpleNamespace(factor_task_id="t-1")),
        study(4, "alpha.model-development"),
        *(study(10 + n, "risk.covariance-development") for n in range(7)),
        study(
            30,
            "portfolio.policy-development",
            portfolio_source=SimpleNamespace(alpha_task_id="t-3"),
        ),
    ]
    plans = {task.task_id: plan for task, plan in studies}
    held = holdings(
        [task for task, _plan in studies],
        lambda task: plans[task.task_id],
        binding_hash=binding,
        workspace=tmp_path,
    )
    assert held["FACTOR_STUDY"] == [{"task_id": "t-1", "curation_receipt_hashes": ["c" * 64]}]
    assert held["ALPHA_STUDY"] == [{"task_id": "t-3", "factor_task_id": "t-1"}]
    assert [row["task_id"] for row in held["RISK_STUDY"]] == [f"t-{n}" for n in range(16, 11, -1)]
    assert held["BOOK"] == [{"task_id": "t-30", "alpha_task_id": "t-3"}]
    receipts = tmp_path / "factor-0" / FACTOR_DEVELOPMENT_CURATION_CATEGORY / "checkpoint"
    receipts.mkdir(parents=True)
    (receipts / ("c" * 64 + ".json")).write_text("{}", encoding="utf-8")
    studies = [
        study(n, "factor.screening-development", output_workspace=f"factor-{n}") for n in range(7)
    ]
    plans = {task.task_id: plan for task, plan in studies}
    held = holdings(
        [task for task, _plan in studies],
        lambda task: plans[task.task_id],
        binding_hash=binding,
        workspace=tmp_path,
    )
    assert [row["task_id"] for row in held["FACTOR_STUDY"]] == [f"t-{n}" for n in range(6, 1, -1)]
    answer = prerequisites("ALPHA_STUDY", held, input_id="in-1", binding_hash=binding)
    assert answer["missing"] == []
    assert answer["next_requests"] == {
        "handoff": {
            "operation": "EXPERIMENT_HANDOFF_PREVIEW",
            "task_id": "t-0",
            "curation_receipt_hash": "c" * 64,
        }
    }


def test_a_draft_onto_another_input_takes_that_inputs_revision():
    "A draft onto another input takes that input's revision."
    from alphalattice.control.product_host.composition.research_experiments import draft_revision

    prior = ("input-a", "a" * 64)
    assert draft_revision(prior, None, None) == ("input-a", "a" * 64)
    assert draft_revision(prior, "input-a", None) == ("input-a", "a" * 64)
    assert draft_revision(prior, "input-b", None) == ("input-b", None)
    assert draft_revision(prior, "input-b", "b" * 64) == ("input-b", "b" * 64)


def test_a_history_row_offers_its_reads_bound_to_it():
    "A history row offers its reads bound to it."
    from alphalattice.control.product_host.composition.research_history import HistoryEntry
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import BookSelector

    task = uuid4()
    book = BookSelector(
        update_task_id=task, update_publication_hash="p" * 64, position_basis="CONDITIONAL_ESTIMATE"
    )
    entry = HistoryEntry(
        "review:x",
        "CRO_REVIEW",
        datetime(2026, 10, 2, tzinfo=UTC),
        task,
        "SUCCEEDED",
        book=book,
        review_publication_hash="r" * 64,
    )
    offered = entry.body()["next_requests"]
    fields = book.request_fields()
    assert offered["task"] == {"operation": "STATUS", "task_id": str(task)}
    assert offered["review"] == {"operation": "EVIDENCE_CRO", **fields}
    assert offered["export"] == {
        "operation": "EVIDENCE_CRO_EXPORT",
        **fields,
        "review_publication_hash": "r" * 64,
    }


@pytest.mark.parametrize(
    "fields",
    [
        {"result_hash": "a" * 64},
        {"handoff_hash": "b" * 64},
        {
            "update_task_id": UUID("00000000-0000-0000-0000-000000000001"),
            "update_publication_hash": "c" * 64,
            "position_basis": "CONDITIONAL_ESTIMATE",
        },
        {
            "update_task_id": UUID("00000000-0000-0000-0000-000000000001"),
            "update_publication_hash": "c" * 64,
            "position_basis": "OBSERVED_RESEARCH_ENTRY",
        },
        {
            "experiment_task_id": UUID("00000000-0000-0000-0000-000000000002"),
            "experiment_receipt_hash": "d" * 64,
            "portfolio_session": "2026-10-02",
        },
    ],
    ids=["result", "handoff", "conditional-update", "observed-update", "experiment"],
)
def test_history_book_discovery_offers_exact_reads_without_a_full_projection(fields, monkeypatch):
    "/EV: discovery offers the complete selector but makes no current-standing claim."
    from alphalattice.control.product_host.composition.evidence_review_projection import (
        EvidenceCroProjector,
    )
    from alphalattice.control.product_host.composition.research_history import HistoryEntry
    from alphalattice.interface.local_application.answers import EvidenceBookSummary
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import BookSelector

    def no_projection(*_args, **_kwargs):
        raise AssertionError("Book discovery opened the full Evidence projection")

    monkeypatch.setattr(EvidenceCroProjector, "projection", no_projection)
    book = BookSelector(**fields)
    row = HistoryEntry(
        "metadata:book",
        "CRO_REVIEW",
        datetime(2026, 10, 2, tzinfo=UTC),
        None,
        "HISTORICAL_REVIEW",
        book=book,
    ).body()
    summary = row["book_summary"]
    EvidenceBookSummary.model_validate(summary)
    assert summary["state"] == "NOT_READ"
    assert summary["verification"] == "METADATA_ONLY_SELECTED_READBACK_VERIFIES_DESCENDANTS"
    assert summary["next_requests"]["review"] == {
        "operation": "EVIDENCE_CRO",
        **book.request_fields(),
    }
    assert summary["next_requests"]["review"] == row["next_requests"]["review"]
    assert "evidence_as_of" not in summary and "review_standing" not in summary


def test_a_provider_that_limits_or_changes_its_interface_is_worded_with_the_way_on():
    "A provider that limits or changes its interface is worded with the way on."
    from alphalattice.control.product_host.composition.plain_refusals import deferral, explain

    for code in (
        "data.rate_limited",
        "data.provider_session_unstable",
        "workspace_preparation.retry_not_due",
        "workspace_data_update.retry_not_due",
    ):
        words = explain(code)["detail"]
        assert "provider" in words and "`retry_after_at`" in words, code
    timeout = refused("data.provider_timeout")
    assert timeout["status"] == "REFUSED" and timeout["failure_code"] == "data.provider_timeout"
    assert timeout["detail"] and "retry_after_at" not in timeout["detail"], timeout
    assert "retry_after_at" not in timeout and "next_requests" not in timeout, timeout
    for code in (
        "data.provider_fetch_failed",
        "data.current_universe_no_feature_ready_listings",
        "data.listing_updates_incomplete",
    ):
        assert "interface" in explain(code)["detail"], code
    resume = {"operation": "WORKSPACE_PREPARE_CONFIRM", "preparation_plan_hash": "a" * 64}
    first = deferral(
        {"status": "DEFERRED", "next_requests": {}},
        failure_code="data.rate_limited",
        retry_after_at="2026-10-01T12:00:00+00:00",
        resume=resume,
        held=False,
    )
    assert first["next_requests"] == {"resume": resume}
    assert first["retry_after_at"] == "2026-10-01T12:00:00+00:00"
    assert "inputs are unchanged" not in first["detail"]
    update = deferral(
        {"status": "PUBLISHED"},
        failure_code="data.remediation_wait",
        retry_after_at=None,
        resume=None,
        held=True,
    )
    assert update["detail"].startswith("The work was deferred")
    assert update["detail"].endswith("every study keeps reading them.")
    assert update["next_requests"] == {}


def test_automation_settings_use_the_same_http_agent_permission_path(live):
    path = "/api/research-update/automation"
    before = len(live.session.task_control_registry.tasks())
    first = _json(live, path)
    assert first["status"] == "DISABLED"
    assert (first["runs_forward"], first["next_requests"]) == ([], {})
    assert _agent(
        live, PortfolioResearchAgentRequest(operation="RESEARCH_UPDATE_AUTOMATION_READBACK")
    ) == _json(live, path)
    saved = _json(
        live,
        path,
        method="POST",
        payload={"automation_enabled": False, "automation_package_ids": []},
    )
    assert saved["settings"]["chosen_by"] == "HUMAN"
    assert (
        _agent(
            live,
            PortfolioResearchAgentRequest(
                operation="RESEARCH_UPDATE_AUTOMATION_CONFIGURE",
                automation_enabled=False,
                automation_package_ids=(),
            ),
        )
        == saved
    )
    assert (
        _request(
            live,
            path,
            method="POST",
            payload={"automation_enabled": "false", "automation_package_ids": []},
        )[0]
        == 400
    )
    assert (
        _request(
            live,
            path,
            method="POST",
            payload={"automation_enabled": True, "automation_package_ids": ["not-installed"]},
        )[0]
        == 400
    )
    assert len(live.session.task_control_registry.tasks()) == before
    assert not saved["source_or_fit_authority_granted"]
    workspace, manifest = (live.workspace, live.workspace_manifest)
    live.stop()
    with LocalPortfolioWebSession(
        workspace=workspace, workspace_manifest=manifest, resolver=_Resolver(_resolved())
    ) as reopened:
        assert _json(reopened, path)["settings"] == saved["settings"]
        assert len(reopened.session.task_control_registry.tasks()) == before


def test_automation_uses_durable_settings_and_sequential_wakes(tmp_path):
    from alphalattice.control.product_host.composition.research_update_automation import (
        ResearchUpdateAutomation,
    )
    from alphalattice.control.workspace_runtime.content_store import (
        CommittedIndex,
        ContentAddressedStore,
    )

    calls = []
    reached = threading.Event()

    def execute(request):
        calls.append(request.operation)
        if request.operation == "RESEARCH_UPDATE_PLAN":
            return {"status": "PLANNED", "update_plan_hash": "a" * 64}
        if calls.count("RESEARCH_UPDATE_RUN") % 2 == 0:
            reached.set()
        return {"status": "REUSED_EXACT", "task_id": None}

    index = CommittedIndex(
        tmp_path, ContentAddressedStore(tmp_path, uri_prefix="test://automation")
    )

    def owner():
        return ResearchUpdateAutomation(
            index=index,
            workspace_manifest_hash="b" * 64,
            installed_package_ids=("one", "two"),
            clock=lambda: datetime.now(UTC),
            execute=execute,
        )

    first = owner()
    first.start()
    try:
        assert not reached.wait(0.02) and (not calls)
        saved = first.configure(enabled=True, package_ids=("one", "two"), chosen_by="HUMAN")
        assert reached.wait(5)
        assert calls == ["RESEARCH_UPDATE_PLAN", "RESEARCH_UPDATE_RUN"] * 2
    finally:
        assert first.close(timeout=None)
    reached.clear()
    second = owner()
    second.start()
    try:
        assert second.readback()["settings"] == saved["settings"]
        assert reached.wait(5)
        assert len(calls) == 8
        second.configure(enabled=False, package_ids=(), chosen_by="INSTALLED_AGENT")
        assert second.readback()["status"] == "DISABLED"
    finally:
        assert second.close(timeout=None)


def test_automation_attends_what_it_admitted_and_never_replans_a_stopped_update(tmp_path):
    "Automation attends what it admitted and never replans a stopped update."
    from alphalattice.control.product_host.composition.research_update_automation import (
        ResearchUpdateAutomation,
    )
    from alphalattice.control.workspace_runtime.content_store import (
        CommittedIndex,
        ContentAddressedStore,
    )

    clock = [datetime(2026, 9, 11, 22, 0, 30, tzinfo=UTC)]
    task = "11111111-1111-4111-8111-111111111111"
    read: dict[str, object] = {}
    calls: list[str] = []
    seen = threading.Condition()

    def execute(request):
        with seen:
            calls.append(request.operation)
            seen.notify_all()
        if request.operation == "RESEARCH_UPDATE_PLAN":
            return {"status": "PLANNED", "update_plan_hash": "a" * 64}
        if request.operation == "RESEARCH_UPDATE_RUN":
            return {"status": "ADMITTED", "task_id": task, "lifecycle": "QUEUED"}
        assert str(request.task_id) == task
        return dict(read)

    def until(count: int, operation: str) -> None:
        with seen:
            assert seen.wait_for(lambda: calls.count(operation) >= count, timeout=5), calls

    automation = ResearchUpdateAutomation(
        index=CommittedIndex(tmp_path, ContentAddressedStore(tmp_path, uri_prefix="test://a")),
        workspace_manifest_hash="b" * 64,
        installed_package_ids=("one",),
        clock=lambda: clock[0],
        execute=execute,
    )
    automation.start()
    try:
        automation.configure(enabled=True, package_ids=("one",), chosen_by="HUMAN")
        until(1, "RESEARCH_UPDATE_RUN")
        retry = datetime(2026, 9, 11, 22, 5, tzinfo=UTC)
        read.update(status="DEFERRED", retry_after_at=retry.isoformat())
        automation.command_completed()
        until(1, "RESEARCH_UPDATE_READBACK")
        end = time.monotonic() + 5
        while automation.readback()["next_due_at"] != retry.isoformat():
            assert time.monotonic() < end, automation.readback()
            time.sleep(0.01)
        assert calls.count("RESEARCH_UPDATE_PLAN") == 1
        clock[0] = retry + timedelta(minutes=1)
        assert automation.wake.signal_due_work(clock[0])
        until(2, "RESEARCH_UPDATE_RUN")
        clock[0] = datetime(2026, 9, 14, 22, 30, tzinfo=UTC)
        read.clear()
        read.update(status="PROPOSAL_PUBLISHED", update={"target_session": "2026-09-11"})
        automation.command_completed()
        until(3, "RESEARCH_UPDATE_RUN")
        read.clear()
        read.update(
            status="BLOCKED",
            failure_code="research_update.binding_changed",
            update={"target_session": "2026-09-11"},
        )
        automation.command_completed()
        until(3, "RESEARCH_UPDATE_READBACK")
        for _ in range(3):
            automation.command_completed()
        time.sleep(0.5)
        assert calls.count("RESEARCH_UPDATE_RUN") == 3, calls
        assert calls.count("RESEARCH_UPDATE_READBACK") == 3, calls
    finally:
        assert automation.close(timeout=None)


def test_a_books_risk_window_and_its_bound_section_are_refused_with_the_way_on():
    "A book's risk window and its bound section are refused with the way on."
    from alphalattice.control.product_host.composition.plain_refusals import explain
    from alphalattice.interface.local_application import cli_contract

    source = {"task_id": "a-1", "candidate_id": "c-1"}
    short = explain("portfolio_research.risk_session_axis_incomplete", portfolio=source)
    assert "scored formations" in short["detail"] and "lookback" in short["detail"]
    assert short["next_requests"]["alpha"] == {"operation": "EXPERIMENT_READBACK", "task_id": "a-1"}
    bound = explain("portfolio_research.authority_field_mismatch", portfolio=source)
    assert bound["fields"] == [["experiment"]] and "bound to its draft" in bound["detail"]
    assert bound["next_requests"]["portfolio_draft"]["task_id"] == "a-1"
    words = json.loads(
        Path(cli_contract.__file__).with_name("refusal_words.json").read_text("utf-8")
    )
    unavailable = words.get("risk_research.formation_selection_requested_axis_unavailable")
    assert unavailable and unavailable["next_action"] == "START_THE_WINDOW_AFTER_THE_LOOKBACK"


def test_a_support_refusal_is_worded_with_its_cause_and_the_books_way_on(live, monkeypatch):
    "A support refusal is worded with its cause and the book's way on."
    from alphalattice.control.product_host.composition.plain_refusals import explain
    from alphalattice.control.product_host.composition.task_recovery import stop_detail

    code = "portfolio_research.benchmark_support_absent"
    book = {"task_id": "alpha-1", "candidate_id": "alpha-candidate-1"}
    facts = "2 names,3 sessions,2020-03-03..2020-03-05"
    unavailable = explain(f"{code}:returns_unavailable:{facts}", portfolio=book)
    assert f"({facts})" in unavailable["detail"] and "quarantine_listings" in unavailable["detail"]
    assert unavailable["next_requests"]["portfolio_draft"] == {
        "operation": "EXPERIMENT_PORTFOLIO_DRAFT",
        "task_id": "alpha-1",
        "candidate_id": "alpha-candidate-1",
    }
    empty = explain(f"{code}:no_eligible_name:1 sessions,2020-03-06..2020-03-06", portfolio=book)
    assert "(1 sessions,2020-03-06..2020-03-06)" in empty["detail"]
    assert "either unavailable-return policy" in empty["detail"]
    assert empty["next_requests"] == {
        "alpha_draft": {"operation": "EXPERIMENT_DRAFT", "task_id": "alpha-1"}
    }
    assert explain(code)["detail"].startswith(
        "The book's walk met a formation session with no eligible name, or an eligible name"
    )
    worded = explain(code, portfolio={"task_id": "a-1", "candidate_id": "c-1"})
    assert "require_complete" in worded["detail"] and "first-use" in worded["detail"]
    assert worded["fields"] == [["portfolio", "unavailable_return_policy"]]
    assert worded["next_requests"]["portfolio_draft"]["task_id"] == "a-1"
    assert stop_detail("research_experiment", code, "TASK_CONTROL") == explain(code)["detail"]
    unworded = stop_detail("research_experiment", "an_owner.code_without_words", "TASK_CONTROL")
    assert unworded.startswith("The Task stopped at a governed boundary")
    host = live.operations
    assert host is not None and live.application is not None
    asked: list[object] = []
    monkeypatch.setattr(host.experiments, "book_source", lambda task: asked.append(task) or book)
    task = live.application.admit(spec=PortfolioResearchSpec.create()).task_id
    way = host._stopped_way(task, f"{code}:returns_unavailable:{facts}", "BLOCKED")
    assert way["next_requests"]["portfolio_draft"]["task_id"] == "alpha-1" and asked == [task]
    host._stopped_way(task, "research_experiment.execution_binding_changed", "BLOCKED")
    assert asked == [task]


def test_prepared_training_inputs_offer_their_components_study(tmp_path):
    "Prepared training inputs offer their component's study."
    from alphalattice.control.product_host.data_preparation.model_training import (
        ModelTrainingInputApplication,
        ModelTrainingInputPlan,
    )
    from alphalattice.control.task_control.contracts import TaskLifecycle
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )

    plan = ModelTrainingInputPlan.create(
        workspace_id="w",
        workspace_manifest_hash="m" * 64,
        input_id="factor-development",
        input_binding_hash="b" * 64,
        component_ids=("G2_R0_TREND",),
        listing_count=466,
        source_session_count=1264,
        implementation_hash="i" * 64,
    )
    task = SimpleNamespace(lifecycle=TaskLifecycle.SUCCEEDED, failure_code=None)
    session = SimpleNamespace(
        workspace=tmp_path, task_control_registry=SimpleNamespace(task=lambda _t: task)
    )
    owner = ModelTrainingInputApplication(session, clock=lambda: datetime(2026, 10, 2, tzinfo=UTC))
    owner._of = lambda _task: plan
    owner._publication = lambda _hash: SimpleNamespace(preparation_receipt_hash="r")
    receipt = SimpleNamespace(bindings=(), model_dump=lambda mode: {})
    owner.store = SimpleNamespace(_load=lambda *_args: receipt)
    offered = owner.readback(uuid4())["next_requests"]
    controls = offered["controls:G2_R0_TREND"]
    assert controls == {
        "operation": "EXPERIMENT_CONTROLS",
        "research_input_id": "factor-development",
        "input_binding_hash": "b" * 64,
        "experiment_kind": "alpha.model-development",
        "component_id": "G2_R0_TREND",
    }
    PortfolioResearchOperationRequest(**controls)
    task.lifecycle = TaskLifecycle.RUNNING
    assert set(owner.readback(uuid4())["next_requests"]) == {"readback"}


@pytest.mark.parametrize(
    ("code", "next_action"),
    [
        (
            "research_strategy.risk_parent_support_incomplete:needed 2019-10-07..2024-09-30, "
            "Risk study 2026-09-01..2026-09-30",
            "STUDY_RISK_OVER_THE_NEEDED_WINDOW",
        ),
        (
            "research_strategy.risk_history_insufficient:formation 2022-10-03 has 64 Risk "
            "return sessions before it, 567 needed, a Risk study starting on or before 2020-08-11",
            "STUDY_RISK_FROM_THE_NEEDED_START",
        ),
    ],
)
def test_a_strategy_short_of_its_risk_window_names_it_and_offers_the_risk_study(code, next_action):
    "Risk support and history refusals name the cause and recovery."
    from alphalattice.interface.local_application.cli_contract import refusal_words

    words = refusal_words(code)
    assert code.partition(":")[2] in words["detail"]
    assert words["next_action"] == next_action


def test_strategy_risk_recovery_keeps_history_ranges_and_selected_input(tmp_path, monkeypatch):
    "Risk recovery carries its selected input and exact history, skipping absent formations."
    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        PortfolioResearchOperations,
    )
    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceExperimentInput,
        ResearchWorkspaceManifest,
        publish_research_workspace_manifest,
    )
    from alphalattice.control.product_host.data_preparation import research_strategy as module
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )

    binding, first, last = ("b" * 64, date(2022, 10, 3), date(2022, 10, 7))
    manifest = ResearchWorkspaceManifest.research_only("risk").with_bindings(
        experiment_inputs=(
            ResearchWorkspaceExperimentInput(input_id="first", binding_hash="a" * 64),
            ResearchWorkspaceExperimentInput(input_id="selected", binding_hash=binding),
        )
    )
    publish_research_workspace_manifest(tmp_path, manifest)
    calendar = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(1300))
    monkeypatch.setattr(module, "read_factor_bundle", lambda *_: SimpleNamespace(sessions=calendar))
    calibrated = module.FROZEN_RESEARCH_BOOK_RECIPES[0].components[0].component_id
    common = {
        "lifecycle": "SUCCEEDED",
        "input_binding_hash": binding,
        "sessions": {"start": str(first), "end": str(last)},
    }
    studies = [
        {
            **common,
            "kind": "alpha.model-development",
            "component_recipe_id": calibrated,
            "task_id": "alpha",
        },
        *(
            {**common, "kind": "risk.covariance-development", "task_id": str(uuid4())}
            for _ in range(2)
        ),
    ]
    owner = module.ResearchStrategyPreparation(
        SimpleNamespace(workspace=tmp_path),
        clock=lambda: datetime(2026, 10, 2, tzinfo=UTC),
        read_experiment=lambda _: {},
        list_experiments=lambda: {"experiments": studies},
    )
    calls, schedule = ([], module.local_research_economic_schedule)

    def counted_schedule(**arguments):
        calls.append(arguments)
        return schedule(**arguments)

    monkeypatch.setattr(module, "local_research_economic_schedule", counted_schedule)
    axis = calendar[calendar.index(first) - module.RISK_HISTORY_SESSIONS + 1 :]
    monkeypatch.setattr(module, "risk_return_surface", lambda *_: (tmp_path, axis))
    monkeypatch.setattr(
        module.CausalRiskReturnReader, "available_sessions", lambda _self, surface: surface
    )
    recovery = owner.controls(binding)
    assert len(calls) == 1
    (window,) = recovery["risk_windows"]
    assert (window["start"], window["end"]) == (str(first), str(last))
    history_start = (
        calendar.index(first) - module.RISK_HISTORY_SESSIONS + module.REQUIRED_LOOKBACK_SESSIONS
    )
    assert window["formation_history"] == {
        "start": str(calendar[history_start]),
        "end": str(calendar[calendar.index(first) - 1]),
        "required_return_sessions": module.RISK_HISTORY_SESSIONS,
    }
    assert recovery["next_requests"]["risk"] == {
        "operation": "EXPERIMENT_CONTROLS",
        "research_input_id": "selected",
        "input_binding_hash": binding,
        "experiment_kind": "risk.covariance-development",
    }
    axis = calendar[calendar.index(first) - module.RISK_HISTORY_SESSIONS :]
    assert "risk" not in owner.controls(binding)["next_requests"]
    calendar = tuple(day for day in calendar if day != first)
    recovery = owner.controls(binding)
    assert recovery["risk_windows"] == []

    def refused_plan(_document):
        raise ValueError("research_strategy.risk_history_insufficient")

    monkeypatch.setattr(owner, "plan", refused_plan)
    host = PortfolioResearchOperations.__new__(PortfolioResearchOperations)
    host.research_strategies = owner
    answer = host._workspace_operation(
        PortfolioResearchOperationRequest(
            operation="RESEARCH_STRATEGY_PLAN", experiment_document={"input_binding_hash": binding}
        ),
        caller="HUMAN",
    )
    assert answer["risk_windows"] == recovery["risk_windows"]
    assert answer["next_requests"] == recovery["next_requests"]


def test_the_first_intent_is_the_way_forward_and_names_it_in_words() -> None:
    "First-use intent offers strategy preparation or the installed strategy's book."
    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        PortfolioResearchOperations,
    )
    from alphalattice.control.product_host.composition.strategy_activation import (
        INSTALLED_BOOK_WORDS,
        RUN_FORWARD_WORDS,
    )

    forward = PortfolioResearchOperations._forward_intents
    shut = SimpleNamespace(first_use=lambda: None)
    empty = SimpleNamespace(_packages={}, activations=None, installed=lambda: False, goals=shut)
    (none,) = forward(empty)
    assert none["status"] == "NO_RESEARCH_STRATEGY_INSTALLED"
    assert none["detail"] == RUN_FORWARD_WORDS[False] and "need no Factor study" in none["detail"]
    assert none["next_requests"] == {
        "strategy_controls": {"operation": "RESEARCH_STRATEGY_CONTROLS"}
    }
    installed = SimpleNamespace(
        _packages={"PKG": object()},
        installed=lambda: True,
        activations=object(),
        _activation_offer=lambda _package: {"status": "INACTIVE"},
        goals=shut,
    )
    (bookless,) = forward(installed)
    assert (bookless["status"], bookless["detail"]) == ("NO_BOOK_YET", INSTALLED_BOOK_WORDS[False])
    # Under an open first use, the book is its numerical check, never a review before activation.
    road = SimpleNamespace(first_use=object, first_use_delegation=lambda _goal: {"active": True})
    installed.goals = road
    assert forward(installed)[0]["detail"] == INSTALLED_BOOK_WORDS[True]
    assert bookless["next_requests"] == {
        "books": {"operation": "CONTROLS", "strategy_package_id": "PKG"}
    }


def test_a_strategys_risk_history_is_judged_by_one_owner_for_its_plan_and_preparation():
    "The plan and preparation share one exact Risk history requirement."
    from alphalattice.control.product_host.research_authoring import frozen_portfolio
    from alphalattice.investment.risk_research.surfaces.decomposition import (
        INSTALLED_RISK_DECOMPOSITION_RECIPE,
    )

    rule = (
        INSTALLED_RISK_DECOMPOSITION_RECIPE.conditional_volatility_initialization_sessions
        + INSTALLED_RISK_DECOMPOSITION_RECIPE.factor_fit_sessions
    )
    assert frozen_portfolio.RISK_HISTORY_SESSIONS == rule == 567
    sessions = [date(2018, 1, 1) + timedelta(days=day) for day in range(rule + 10)]
    short = frozen_portfolio.risk_history_shortfall(sessions, [sessions[rule - 1], sessions[-1]])
    assert short == (sessions[rule - 1], rule - 1)
    assert frozen_portfolio.risk_history_shortfall(sessions, [sessions[rule]]) is None


def _research_update_plan(package: str, target: Any) -> Any:
    """A sealed research update plan of one strategy, each of its parts well formed ()."""
    from alphalattice.control.data_platform.maintenance.contracts import (
        MaintenanceTrigger,
        WorkspaceDataUpdateBinding,
        WorkspaceDataUpdatePlan,
        WorkspaceInputStatus,
        WorkspaceMaintenanceRequest,
    )
    from alphalattice.control.product_host.composition.decision_advancement import (
        DecisionAdvancementPlan,
    )
    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceScoreInput,
    )
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    fixture = "a" * 64
    binding = WorkspaceDataUpdateBinding.seal(
        profile_file_hash=fixture, data_policy_hash=fixture, feature_catalog_hash=fixture
    )
    before = WorkspaceInputStatus.seal(
        manifest_revision="r1",
        data_revision_hash=fixture,
        data_through=target,
        adjusted_through=target,
        panel_hash=fixture,
        panel_through=target,
        readiness_status="READY",
        sources_checked_at=None,
        foundation_hash=None,
        foundation_panel_hash=None,
        foundation_disposition="MISSING",
    )
    request = WorkspaceMaintenanceRequest.create(
        market_profile_id=binding.market_profile_id,
        target_market_session=target,
        knowledge_cutoff_at=None,
        trigger=MaintenanceTrigger.USER_REQUEST,
        membership_revision="r1",
        data_policy_hash=fixture,
        feature_policy_hash=canonical_hash({"catalog": fixture, "invalidation": "domain-topology"}),
        requested_at=datetime(2026, 10, 1, tzinfo=UTC),
    )
    return DecisionAdvancementPlan.create(
        workspace_manifest_hash=fixture,
        catalog_hash=fixture,
        package_id=package,
        score_binding=ResearchWorkspaceScoreInput(
            strategy_package_id=package,
            strategy_package_hash=fixture,
            authority_relative_path="authority/fixture.json",
            authority_hash=fixture,
            source_kind="WORKSPACE_DATA_FEATURE",
        ),
        checkpoint_hash=fixture,
        parent_hash=None,
        target=target,
        decision_sessions=(target,),
        score_sessions=(target,),
        history_score_hashes=(),
        data_plan=WorkspaceDataUpdatePlan.seal(
            workspace_id="fixture",
            workspace_manifest_hash=fixture,
            binding=binding,
            before=before,
            request=request,
        ),
        implementation_hash=fixture,
        score_implementation_hash=fixture,
        calibration_implementation_hash=fixture,
        decision_implementation_hash=fixture,
    )


def _admitted(registry: Any, envelope: Any, goal: Any, plan: Any) -> str:
    return str(
        registry.admit(
            input_envelope=envelope, goal=goal, plan=plan, observed_at=datetime.now(UTC)
        ).record.task_id
    )


def test_a_cancelled_research_update_keeps_its_stop_and_offers_its_task_record(live, capsys):
    "A cancelled research update keeps its stop and offers its task record."
    from alphalattice.control.product_host.composition.decision_advancement import task_contract
    from alphalattice.control.task_control.contracts import TaskExecutionCompatibility
    from alphalattice.interface.local_application.cli import main
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    registry = live.session.task_control_registry
    envelope, goal, workflow = task_contract(_research_update_plan(ALPHA, date(2026, 9, 29)))
    task_id = UUID(_admitted(registry, envelope, goal, workflow))
    started = registry.start_next(
        compatibility=TaskExecutionCompatibility.create(
            task_contract_hash=canonical_hash("CANCELLED_READBACK_FIXTURE"),
            workflow_definition_hash=workflow.workflow_definition_hash,
            input_schema_id=envelope.input_schema_id,
            domain_policy_hash=canonical_hash("CANCELLED_READBACK_FIXTURE"),
            framework_identity_hash=canonical_hash("CANCELLED_READBACK_FIXTURE"),
        ),
        worker_instance_id=uuid4(),
        observed_at=live.clock(),
        expected_task_id=task_id,
    )
    assert started is not None
    requested, _command = registry.request_cancel(
        task_id=task_id, expected_task_hash=started[0].record_hash, observed_at=live.clock()
    )
    cancelled = registry.finalize_cancel_after_writer_stopped(
        task_id=task_id, expected_task_hash=requested.record_hash, observed_at=live.clock()
    )
    assert live.operations.research_updates.readback(task_id) == {
        "status": "CANCELLED",
        "task_id": str(task_id),
        "failure_code": "TASK_CANCELLED_AFTER_WRITER_STOPPED",
    }
    saved = live.workspace.parent / "cancelled.json"
    assert (
        main(
            [
                "--workspace",
                str(live.workspace),
                "--view",
                "full",
                "research-update",
                "show",
                "--task",
                str(task_id),
                "--output",
                str(saved),
            ],
            serve=lambda _: 99,
        )
        == 2
    )
    answer = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert answer["data"]["status"] == "CANCELLED"
    assert answer["failure_code"] == cancelled.failure_code
    assert "Task will not continue" in answer["detail"]
    assert answer["next_requests"] == {"task": {"operation": "STATUS", "task_id": str(task_id)}}
    assert (
        main(
            [
                "--workspace",
                str(live.workspace),
                "--view",
                "full",
                "request",
                "--from",
                str(saved),
                "--action",
                "task",
            ],
            serve=lambda _: 99,
        )
        == 2
    )
    followed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["data"]
    assert followed["task_id"] == str(task_id) and followed["lifecycle"] == "CANCELLED"
    assert followed["latest_failure_code"] == cancelled.failure_code
    assert registry.task(task_id).record_hash == cancelled.record_hash


def _planned_task(registry: Any, kind: str, plan: dict[str, object]) -> str:
    """A Task of `kind` whose sealed plan is `plan`: what a reader picking among Tasks reads."""
    from alphalattice.control.task_control.contracts import (
        ResearchGoal,
        ResearchPlan,
        TaskInputEnvelope,
        WorkItemDefinition,
    )
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    envelope = TaskInputEnvelope.create(
        task_kind=kind, input_schema_id=f"{kind}.fixture", payload={"plan": plan}
    )
    goal = ResearchGoal.create(
        goal_kind="FIXTURE",
        input_hash=envelope.input_hash,
        deliverable_kind="Fixture",
        summary="A Task its reader picks by the strategy its plan names.",
    )
    work = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=canonical_hash("fixture"),
        verifier_catalog_hash=canonical_hash("fixture"),
        work_items=(
            WorkItemDefinition.create(
                stage_id="only", dependency_ids=(), verifier_id="fixture.only"
            ),
        ),
    )
    return _admitted(registry, envelope, goal, work)


def test_completed_update_metadata_keeps_exact_target_claim_and_sealed_lookup(live, monkeypatch):
    "Completed update metadata keeps exact target claim and sealed lookup."
    from alphalattice.control.product_host.composition.decision_advancement import (
        STAGES,
        task_contract,
    )
    from alphalattice.control.task_control.contracts import TaskLifecycle, TaskRecord
    from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
        PortfolioUpdatePublication,
        advance_decision_state,
    )
    from tests.portfolio_strategy_lab.synthetic_numerical import (
        HASH,
        build_numerical,
        prepared_for,
        snapshot_for,
    )

    assert live.operations is not None
    operations = live.operations
    assert operations.research_updates is not None
    assert operations.activations is not None and operations.automation is not None
    owner = operations.research_updates
    numerical = build_numerical()
    package = numerical.checkpoint.package.strategy_id
    plan = _research_update_plan(package, numerical.sessions[numerical.first])
    publication = advance_decision_state(
        checkpoint=numerical.checkpoint,
        previous=None,
        prepared=prepared_for(numerical, numerical.first),
        observed=snapshot_for(numerical, numerical.first),
        plan_hash=HASH,
        published_at=datetime(2026, 10, 2, tzinfo=UTC),
    )
    assert isinstance(publication, PortfolioUpdatePublication)
    owner.ledger.content.publish_model(
        category="decision-candidates", value=publication, identity_field="content_hash"
    )
    owner._commit(plan, STAGES[5], (publication.content_hash,), HASH)
    envelope, goal, workflow = task_contract(plan)
    recorded_at = datetime(2026, 10, 2, tzinfo=UTC)
    task = TaskRecord.from_identity(
        task_id=uuid4(),
        task_kind=envelope.task_kind,
        input=envelope,
        goal=goal,
        plan=workflow,
        lifecycle=TaskLifecycle.SUCCEEDED,
        active_work_item_id=None,
        latest_execution_id=None,
        admitted_at=recorded_at,
        started_at=None,
        updated_at=recorded_at,
        failure_code=None,
        version=1,
    )
    monkeypatch.setattr(live.session.task_control_registry, "tasks", lambda: (task,))
    monkeypatch.setattr(operations.automation, "installed", (package,))

    def no_full_read(*_args, **_kwargs):
        raise AssertionError("Automation metadata opened full selected verification")

    monkeypatch.setattr(owner, "readback", no_full_read)
    monkeypatch.setattr(operations.activations, "state", no_full_read)
    monkeypatch.setattr(operations.activations, "read_review", no_full_read)
    answer = _json(live, "/api/research-update/automation")
    (running,) = answer["runs_forward"]
    latest = running["latest_update"]
    assert latest["task_id"] == str(task.task_id) and latest["lifecycle"] == "SUCCEEDED"
    assert latest["target_session"] == plan.target.isoformat()
    assert latest["claim"] == publication.claim
    assert latest["verification"] == "METADATA_ONLY_SELECTED_READBACK_VERIFIES_DESCENDANTS"
    assert latest["review_selector"] == {
        "update_task_id": str(task.task_id),
        "update_publication_hash": publication.content_hash,
        "position_basis": "CONDITIONAL_ESTIMATE",
    }
    assert latest["next_requests"]["readback"] == {
        "operation": "RESEARCH_UPDATE_READBACK",
        "task_id": str(task.task_id),
    }
    assert "review_standing" not in running
    stored = owner.ledger.content.root / "decision-candidates" / f"{publication.content_hash}.json"
    payload = json.loads(stored.read_text(encoding="utf-8"))
    payload["published_at"] = "2026-10-03T00:00:00+00:00"
    stored.write_text(json.dumps(payload), encoding="utf-8")
    damaged = _json(live, "/api/research-update/automation")["runs_forward"][0]["latest_update"]
    assert damaged["status"] == "REFUSED" and damaged["failure_code"] and damaged["detail"]
    assert damaged["next_requests"]["task"] == {"operation": "STATUS", "task_id": str(task.task_id)}
    assert "review_selector" not in damaged and "claim" not in damaged


def test_a_strategys_latest_work_is_read_by_its_package_and_never_anothers(
    live, monkeypatch, capsys, tmp_path
):
    "A strategy's latest work is read by its package and never another's."
    from alphalattice.control.product_host.composition.decision_advancement import task_contract
    from alphalattice.control.product_host.composition.portfolio_updates import (
        TASK_KIND as PORTFOLIO_UPDATE,
    )
    from alphalattice.control.product_host.composition.strategy_calibration import (
        TASK_KIND as CALIBRATION,
    )
    from alphalattice.control.product_host.composition.strategy_scoring import TASK_KIND as SCORE
    from alphalattice.interface.local_application.cli import main

    registry = live.session.task_control_registry
    write_queue_setting(
        live.workspace / "runtime", 16, chosen_by="HUMAN", chosen_at=datetime.now(UTC)
    )

    def update(package: str, target: date) -> str:
        return _admitted(registry, *task_contract(_research_update_plan(package, target)))

    def cli(*argv: str) -> tuple[int, dict[str, Any]]:
        code = main(
            ["--workspace", str(live.workspace), "--view", "full", *argv], serve=lambda _: 99
        )
        return (code, json.loads(capsys.readouterr().out.strip().splitlines()[-1]))

    first = update(ALPHA, date(2026, 9, 29))
    _code, alone = cli("research-update", "show")
    assert alone["data"]["task_id"] == first, alone
    beta = update(BETA, date(2026, 9, 29))
    second = update(ALPHA, date(2026, 9, 30))
    code, refused = cli("research-update", "show")
    assert code == 2 and refused["failure_code"] == "research_update.strategy_package_required"
    assert refused["data"]["strategy_package_ids"] == [ALPHA, BETA]
    assert refused["data"]["next_requests"] == {
        f"readback:{package}": {
            "operation": "RESEARCH_UPDATE_READBACK",
            "strategy_package_id": package,
        }
        for package in (ALPHA, BETA)
    }
    for package, task in ((ALPHA, second), (BETA, beta)):
        _code, read = cli("research-update", "show", "--package", package)
        assert (read["data"]["task_id"], read["data"]["status"]) == (task, "QUEUED"), read
    code, crossed = cli("research-update", "show", "--task", beta, "--package", ALPHA)
    assert code == 2
    assert crossed["failure_code"] == f"research_update.task_of_another_strategy:{beta}"
    saved = tmp_path / "refused.json"
    assert cli("research-update", "show", "--output", str(saved))[0] == 2
    _code, followed = cli("request", "--from", str(saved), "--action", f"readback:{BETA}")
    assert followed["data"]["task_id"] == beta, followed
    monkeypatch.setattr(live.operations.automation, "installed", (ALPHA, BETA))
    monkeypatch.setattr(live.operations.activations, "state", lambda _package: {"status": "ACTIVE"})
    rows = _json(live, "/api/research-update/automation")["runs_forward"]
    assert {row["strategy_package_id"]: row["latest_update"] for row in rows} == {
        package: {
            "task_id": task,
            "lifecycle": "QUEUED",
            "target_session": target,
            "next_requests": {
                "readback": {"operation": "RESEARCH_UPDATE_READBACK", "task_id": task}
            },
        }
        for package, task, target in ((ALPHA, second, "2026-09-30"), (BETA, beta, "2026-09-29"))
    }
    for kind, command, owner, nested, none in (
        (PORTFOLIO_UPDATE, "portfolio-update", "portfolio_update", False, "NO_PORTFOLIO_UPDATE"),
        (SCORE, "score", "strategy_score", True, "NO_SCORE_PUBLICATION"),
        (CALIBRATION, "calibration", "strategy_calibration", True, "NO_CALIBRATION_PUBLICATION"),
    ):
        _code, empty = cli(command, "show", "--package", ALPHA)
        assert empty["data"] == {
            "status": none,
            "task_id": None,
            "read_request": {
                "operation": f"{owner.upper()}_READBACK",
                "strategy_package_id": ALPHA,
            },
        }, empty
        for package in (ALPHA, BETA):
            named: dict[str, object] = {"strategy_package_id": package}
            _planned_task(registry, kind, {"binding": named} if nested else named)
        other = str(registry.tasks()[-1].task_id)
        code, refused = cli(command, "show")
        assert code == 2 and refused["failure_code"] == f"{owner}.strategy_package_required"
        assert sorted(refused["data"]["next_requests"]) == [f"readback:{ALPHA}", f"readback:{BETA}"]
        code, crossed = cli(command, "show", "--task", other, "--package", ALPHA)
        assert code == 2 and crossed["failure_code"] == f"{owner}.task_of_another_strategy:{other}"
