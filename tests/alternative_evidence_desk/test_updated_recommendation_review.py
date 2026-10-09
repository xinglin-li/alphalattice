"""CU review seam over real numerical publications and the booted operations.

Only upstream Task completion is seeded (through Task Control's public stage
API). No CU readback, selector, scope, Evidence/CRO Task or publication is stubbed.
The numerical fixture uses frozen Return/Balanced recipes with synthetic inputs;
it is not another model fit or a claim about the admitted research corpus.
"""

from dataclasses import replace
from types import SimpleNamespace
from urllib.parse import urlencode
from uuid import UUID, uuid4

import numpy as np
import pytest

from alphalattice.control.product_host.composition.portfolio_updates import (
    PortfolioUpdatePlan,
    _implementation_hash,
    _task_contract,
)
from alphalattice.control.task_control.contracts import (
    TaskExecutionCompatibility,
    TaskLifecycle,
    TaskStageReceipt,
)
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument as PortfolioResearchAgentRequest,
)
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    PortfolioObservedSettlement,
    PortfolioUpdatePublication,
    advance_decision_state,
    portfolio_update_positions,
)
from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (
    REBOUND_RETURN_BOOK_RECIPE,
    TREND_REBOUND_BOOK_RECIPE,
    _post_observed_package,
)
from alphalattice.investment.portfolio_strategy_lab.reporting.static import render_decision_update
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    PortfolioReviewPublication,
)
from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
    PortfolioReviewTaskAdapter,
)
from tests.alternative_evidence_desk.planted_corpus import _NOW
from tests.alternative_evidence_desk.review_http_support import (
    _raise_interruption,
    build_authority,
    build_workspace,
    start_service,
)
from tests.portfolio_strategy_lab.synthetic_numerical import (
    HASH,
    build_numerical,
    prepared_for,
    snapshot_for,
)


@pytest.fixture(
    scope="module",
    params=(REBOUND_RETURN_BOOK_RECIPE, TREND_REBOUND_BOOK_RECIPE),
    ids=("Return", "Balanced"),
)
def numerical(request):
    recipe = request.param
    package = _post_observed_package(authority=SimpleNamespace(authority_hash=HASH), recipe=recipe)
    return build_numerical(
        installed_package=package,
        book_recipe=recipe,
        listing_ids=tuple(f"review-{i:03d}" for i in range(130)),
    )


def seed_update(service, n, *, previous=None, observed=False, legacy_settlement=False):
    """Publish through existing owners, capture exact Task input and stage evidence."""
    owner, registry = service.session.operations.updates, service.registry
    store = owner.store
    index = n.first + int(observed)
    market = snapshot_for(n, index)
    prepared = None if observed else prepared_for(n, index)
    plan = PortfolioUpdatePlan.create(
        strategy_package_id=n.checkpoint.package.strategy_id,
        workspace_manifest_hash=service.session.workspace_manifest.manifest_hash,
        catalog_hash=HASH,
        checkpoint_hash=n.checkpoint.content_hash,
        parent_hash=None if previous is None else previous.content_hash,
        prepared_input_hash=None if prepared is None else prepared.content_hash,
        requested_input_hash=None if prepared is None else prepared.content_hash,
        observed_through=market.through,
        source_hash=HASH,
        market_hash=market.content_hash,
        implementation_hash=_implementation_hash(),
    )
    store.publish_decision_checkpoint(n.checkpoint)
    store.content.publish_model(category="decision-plans", value=plan, identity_field="plan_hash")
    store.content.publish_model(
        category="decision-observations", value=market, identity_field="content_hash"
    )
    value = advance_decision_state(
        checkpoint=n.checkpoint,
        previous=previous,
        prepared=prepared,
        observed=market,
        plan_hash=plan.plan_hash,
        published_at=_NOW,
    )
    if legacy_settlement:

        def legacy(entry):
            if entry is None:
                return None
            payload = entry.model_dump(mode="json", exclude={"content_hash", "pretrade_weights"})
            return PortfolioObservedSettlement.model_validate(
                {**payload, "content_hash": canonical_hash(payload)}
            )

        value = PortfolioUpdatePublication.create(
            **{
                **{k: getattr(value, k) for k in type(value).model_fields if k != "content_hash"},
                "events": tuple(legacy(e) for e in value.events),
                "active_entry": legacy(value.active_entry),
            }
        )
    html_hash, _ = store.publish_html(
        render_decision_update(
            n.checkpoint, (*store.decision_history(n.checkpoint.history_hash), value)
        )
    )
    value = PortfolioUpdatePublication.create(
        **{
            **{k: getattr(value, k) for k in type(value).model_fields if k != "content_hash"},
            "html_hash": html_hash,
        }
    )
    store.publish_decision_update(value)
    envelope, goal, workflow = _task_contract(plan)
    task = registry.admit(
        input_envelope=envelope, goal=goal, plan=workflow, observed_at=_NOW
    ).record
    compatibility = TaskExecutionCompatibility.create(
        task_contract_hash=canonical_hash(schema_structure(PortfolioUpdatePlan)),
        workflow_definition_hash=workflow.workflow_definition_hash,
        input_schema_id=envelope.input_schema_id,
        domain_policy_hash=plan.checkpoint_hash,
        framework_identity_hash=plan.implementation_hash,
    )
    _, execution = registry.start_next(
        compatibility=compatibility,
        worker_instance_id=uuid4(),
        observed_at=_NOW,
        expected_task_id=task.task_id,
    )
    for definition, identity in zip(
        workflow.work_items, (plan.plan_hash, market.content_hash, value.content_hash), strict=True
    ):
        item = registry.begin_work_item(
            task_id=task.task_id,
            execution_id=execution.execution_id,
            stage_id=definition.stage_id,
            observed_at=_NOW,
        )
        evidence = owner._evidence(definition.stage_id, identity)
        registry.mark_ready(
            task_id=task.task_id,
            execution_id=execution.execution_id,
            stage_id=definition.stage_id,
            evidence=evidence,
            observed_at=_NOW,
        )
        registry.verify_work_item(
            TaskStageReceipt.from_identity(
                receipt_id=uuid4(),
                task_id=task.task_id,
                execution_id=execution.execution_id,
                stage_id=definition.stage_id,
                work_item_definition_hash=item.definition_hash,
                verifier_id=definition.verifier_id,
                evidence=evidence,
                status="VERIFIED",
                failure_code=None,
                observed_at=_NOW,
            )
        )
    assert registry.task(task.task_id).lifecycle is TaskLifecycle.SUCCEEDED
    body = service.get(f"/api/portfolio-update?task_id={task.task_id}")
    assert body["publication"]["content_hash"] == value.content_hash
    return value, body["review_selector"]


@pytest.fixture
def updated(tmp_path, numerical):
    workspace, _ = build_workspace(tmp_path)
    positions = tuple(
        SimpleNamespace(listing_id=v) for v in numerical.checkpoint.ordered_listing_ids
    )
    authority = build_authority(
        tmp_path=tmp_path,
        report=SimpleNamespace(window_end_book=SimpleNamespace(positions=positions)),
    )
    service = start_service(workspace, authority, tmp_path)
    value, selector = seed_update(service, numerical)
    context = SimpleNamespace(
        service=service,
        authority=authority,
        workspace=workspace,
        value=value,
        selector=selector,
        n=numerical,
        path=tmp_path,
    )
    try:
        yield context
    finally:
        context.service.session.stop()


def test_updated_book_uses_exact_source_not_the_default_result(updated):
    c = updated
    service = c.service
    # The update's positions offer their own review, bound to the update, so an agent's
    # `evidence preview --from <update answer>` reads them, never the default book.
    from alphalattice.interface.local_application.client import continued

    readback = service.get(f"/api/portfolio-update?task_id={c.selector['update_task_id']}")
    # An exact reuse names the Task that ran the plan it reused, which its read follows, so
    # `portfolio-update show --from <the reuse>` reads that update, never another.
    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        reused_read,
    )

    reused = reused_read(service.registry, c.value.plan_hash, "PORTFOLIO_UPDATE_READBACK")
    assert reused["publication_task_id"] == c.selector["update_task_id"]
    assert continued("PORTFOLIO_UPDATE_READBACK", reused, {}, frozenset({"task_id"})) == {
        "operation": "PORTFOLIO_UPDATE_READBACK",
        "task_id": c.selector["update_task_id"],
    }
    assert reused_read(service.registry, "0" * 64, "READ") == {}
    # A study's read of a Task that is not a study names its kind and the read that takes it,
    # not a bare code (daily scene).
    refused = service.agent(
        PortfolioResearchAgentRequest(
            operation="EXPERIMENT_READBACK", task_id=UUID(c.selector["update_task_id"])
        )
    )
    assert refused["failure_code"] == "research_experiment.task_kind_mismatch", refused
    assert refused["detail"] and refused["task_kind"], refused
    assert refused["next_requests"] == {
        "task": {"operation": "STATUS", "task_id": c.selector["update_task_id"]}
    }
    allowed = frozenset(c.selector)
    for operation in ("EVIDENCE_PREVIEW", "EVIDENCE_CRO", "CRO_REVIEW_DOSSIER"):
        assert continued(operation, readback, {}, allowed) == {
            "operation": operation,
            **c.selector,
        }
    section = service.get("/api/evidence-cro?" + urlencode(c.selector))
    assert section["book"]["authority"] == "CONDITIONAL_RESEARCH_PROPOSAL"
    subject = section["book"]["update_subject"]
    assert subject["strategy_package_id"] == c.n.checkpoint.package.strategy_id
    assert subject["position_hash"] == c.value.pending_proposal.content_hash
    assert section["book"]["result_hash"] is None
    initial = len(service.registry.tasks())
    for replacement, refused in (
        ({"update_publication_hash": "0" * 64}, "portfolio_update.publication_not_bound_to_task"),
        (
            {"position_basis": "OBSERVED_RESEARCH_ENTRY"},
            "product_host.evidence_review_update_basis_mismatch",
        ),
    ):
        code, body = service.request(
            "/api/evidence-cro?" + urlencode({**c.selector, **replacement})
        )
        # Refused by name and worded with the way on.
        assert (code, body["failure_code"]) == (400, refused), body
        assert body["detail"] and body["next_action"], body
    # A second book beside the update's is refused in words, with the history to choose one
    # from.
    code, body = service.request(
        "/api/evidence-cro?" + urlencode({**c.selector, "result_hash": service.result_hash()})
    )
    assert (code, body["status"], body["failure_code"]) == (
        200,
        "REFUSED",
        "product_host.evidence_review_selector_ambiguous",
    ), body
    assert body["next_action"] == "NAME_ONE_BOOK" and body["detail"]
    assert len(service.registry.tasks()) == initial
    assert service.post("/api/evidence-refresh", c.selector)["disposition"] == "ADMITTED"
    service.drain()
    assert service.post("/api/cro-review", c.selector)["disposition"] == "ADMITTED"
    service.drain()
    published = service.get("/api/evidence-cro?" + urlencode(c.selector))
    assert published["state"] == "REVIEW_PUBLISHED", published
    assert published["review_publication_hash"]
    export_selector = {
        **c.selector,
        "review_publication_hash": published["review_publication_hash"],
    }
    exported = service.get("/api/evidence-cro/export?" + urlencode(export_selector))
    assert exported["portfolio"]["publication"] == c.value.model_dump(mode="json")
    assert exported["review"]["dossier"]["report_hash"] is None
    assert exported["review"]["dossier"]["update_subject"] == subject
    assert exported["evidence"]["verified_spans"]
    assert "Evidence &amp; CRO" in exported["html"]
    assert exported["export_hash"] == canonical_hash(
        {k: v for k, v in exported.items() if k != "export_hash"}
    )
    assert (
        service.agent(
            PortfolioResearchAgentRequest(operation="EVIDENCE_CRO_EXPORT", **export_selector)
        )
        == exported
    )
    calls, tasks = (
        len(service.review.review_actor.observed_deadlines),
        len(service.registry.tasks()),
    )
    assert service.post("/api/cro-review", c.selector)["disposition"] == "REUSED_EXACT"
    assert (len(service.review.review_actor.observed_deadlines), len(service.registry.tasks())) == (
        calls,
        tasks,
    )
    observed, observed_selector = seed_update(service, c.n, previous=c.value, observed=True)
    parent_alias = {**c.selector, "update_task_id": observed_selector["update_task_id"]}
    assert (
        service.get("/api/evidence-cro?" + urlencode(parent_alias))["review_publication_hash"]
        == published["review_publication_hash"]
    )
    assert service.post("/api/cro-review", parent_alias)["disposition"] == "REUSED_EXACT"
    facts = portfolio_update_positions(observed)
    assert facts.preceding is not None
    np.testing.assert_array_equal(observed.book.weights, c.n.reference.executed_weights[c.n.first])
    after = service.get("/api/evidence-cro?" + urlencode(observed_selector))
    assert after["state"] != "REVIEW_PUBLISHED", "a different basis cannot inherit the verdict"
    assert after["book"]["authority"] == "OBSERVED_RESEARCH_ENTRY"
    # New observed changes reuse the issuer evidence when its ordered scope matches.
    if after["state"] == "AWAITING_ALTERNATIVE_EVIDENCE":
        service.post("/api/evidence-refresh", observed_selector)
        service.drain()
    service.post("/api/cro-review", observed_selector)
    service.drain()
    assert (
        service.get("/api/evidence-cro?" + urlencode(observed_selector))["state"]
        == "REVIEW_PUBLISHED"
    )
    code, refused = service.request(
        "/api/evidence-cro/export?"
        + urlencode(
            {**observed_selector, "review_publication_hash": published["review_publication_hash"]}
        )
    )
    assert code == 400 and "subject_mismatch" in refused["refused"]
    service.session.stop()
    c.service = start_service(
        c.workspace, replace(c.authority, model_authority_admitted=False, review_actor=None), c.path
    )
    tasks = len(c.service.registry.tasks())
    assert c.service.get("/api/evidence-cro/export?" + urlencode(export_selector)) == exported
    assert (
        c.service.post("/api/cro-review", c.selector)["disposition"]
        == "REFUSED_MODEL_AUTHORITY_NOT_ADMITTED"
    )
    assert len(c.service.registry.tasks()) == tasks
    other_recipe = (
        TREND_REBOUND_BOOK_RECIPE
        if c.n.checkpoint.package.strategy_id == "RETURN_G6_MU_ONLY"
        else REBOUND_RETURN_BOOK_RECIPE
    )
    other_package = _post_observed_package(
        authority=SimpleNamespace(authority_hash=HASH), recipe=other_recipe
    )
    other = build_numerical(
        installed_package=other_package,
        book_recipe=other_recipe,
        listing_ids=tuple(f"review-{i:03d}" for i in range(130)),
    )
    other_value, _ = seed_update(c.service, other)
    tasks = len(c.service.registry.tasks())
    all_history = c.service.get("/api/research-history?history_limit=50")
    assert any(
        r["book"] and r["book"].get("update_publication_hash") == other_value.content_hash
        for r in all_history["entries"]
    )
    history = c.service.get(
        "/api/research-history?"
        + urlencode(
            {
                "strategy_package_id": c.n.checkpoint.package.strategy_id,
                "history_limit": 50,
            }
        )
    )
    assert history["status"] == "AVAILABLE", history
    updates = [r for r in history["entries"] if r["kind"] == "CONTINUOUS_UPDATE"]
    assert {r["book"]["update_publication_hash"] for r in updates} == {
        c.value.content_hash,
        observed.content_hash,
    }
    # Each publication is dated by its own publication time, not its Task's
    # admission, so a newer result never sorts after an older one.
    dated = {value.content_hash: value.published_at.isoformat() for value in (c.value, observed)}
    assert {r["book"]["update_publication_hash"]: r["recorded_at"] for r in updates} == dated
    assert all(
        r["strategy_package_id"] == c.n.checkpoint.package.strategy_id for r in history["entries"]
    )
    reviews = [r for r in history["entries"] if r["kind"] == "CRO_REVIEW"]
    historical = next(
        r for r in reviews if r["review_publication_hash"] == published["review_publication_hash"]
    )
    assert historical["book"] == {k: str(v) for k, v in c.selector.items()}
    from datetime import timedelta

    c.service.review.clock = lambda: _NOW + timedelta(days=180)
    assert c.service.get("/api/evidence-cro/export?" + urlencode(export_selector)) == exported
    assert len(c.service.registry.tasks()) == tasks


def test_legacy_settlement_omits_new_field_and_never_invents_changes(numerical):
    n = numerical
    first = advance_decision_state(
        checkpoint=n.checkpoint,
        previous=None,
        prepared=prepared_for(n, n.first),
        observed=snapshot_for(n, n.first),
        plan_hash=HASH,
        published_at=_NOW,
    )
    second = advance_decision_state(
        checkpoint=n.checkpoint,
        previous=first,
        prepared=None,
        observed=snapshot_for(n, n.first + 1),
        plan_hash=HASH,
        published_at=_NOW,
    )
    original = second.events[-1]
    payload = original.model_dump(mode="json", exclude={"pretrade_weights", "content_hash"})
    payload["content_hash"] = canonical_hash(payload)
    legacy = PortfolioObservedSettlement.model_validate(payload)
    assert legacy.model_dump(mode="json") == payload
    assert legacy.pretrade_weights is None
    assert original.entry == legacy.entry and original.turnover == legacy.turnover
    payload["pretrade_weights"] = list(original.pretrade_weights)
    with pytest.raises(ValueError, match="identity_invalid"):
        PortfolioObservedSettlement.model_validate(payload)


def test_legacy_entry_stays_readable_but_cannot_invent_a_review(updated):
    c = updated
    value, selector = seed_update(
        c.service, c.n, previous=c.value, observed=True, legacy_settlement=True
    )
    tasks = len(c.service.registry.tasks())
    body = c.service.get("/api/evidence-cro?" + urlencode(selector))
    assert body["state"] == "REVIEW_INPUT_INCOMPLETE"
    assert not body["available_actions"]
    exported = c.service.get("/api/evidence-cro/export?" + urlencode(selector))
    assert exported["review"] is None and exported["portfolio"]["publication"] == value.model_dump(
        mode="json"
    )
    status, refused = c.service.request("/api/cro-review", method="POST", payload=selector)
    assert status == 400 and "input_incomplete" in refused["refused"]
    assert len(c.service.registry.tasks()) == tasks
    assert not c.service.review.review_actor.observed_deadlines


@pytest.mark.parametrize("tamper", (False, True), ids=("resume", "refuse-tampered-source"))
def test_update_review_recovery_revalidates_its_exact_source(updated, monkeypatch, tamper):
    c = updated
    service = c.service
    service.post("/api/evidence-refresh", c.selector)
    service.drain()
    with monkeypatch.context() as patch:
        patch.setattr(PortfolioReviewTaskAdapter, "verify_stage", _raise_interruption)
        admitted = service.post("/api/cro-review", c.selector)
        service.drain()
    task_id = UUID(admitted["task_id"])
    task = service.registry.task(task_id)
    assert task.lifecycle is TaskLifecycle.RECOVERY_REQUIRED
    path = (
        service.session.operations.updates.store.root
        / "decision-updates"
        / f"{c.value.content_hash}.json"
    )
    original = path.read_bytes()
    calls = len(service.review.review_actor.observed_deadlines)
    service.session.stop()
    if tamper:
        path.write_bytes(b"tampered")
    try:
        c.service = start_service(c.workspace, c.authority, c.path)
        assert c.service.session.resumed_task_ids == (task_id,)
        c.service.drain()
        recovered = c.service.registry.task(task_id)
        assert recovered.input == task.input
        values = c.service.review.artifacts.values(
            "cro-review-publications", PortfolioReviewPublication
        )
        if tamper:
            assert recovered.lifecycle is TaskLifecycle.BLOCKED
            assert not values
            assert len(c.service.review.review_actor.observed_deadlines) == calls
        else:
            assert recovered.lifecycle is TaskLifecycle.SUCCEEDED
            assert len(values) == 1
            assert values[0].update_subject.update_publication_hash == c.value.content_hash
            assert len(c.service.review.review_actor.observed_deadlines) == calls + 1
    finally:
        path.write_bytes(original)
