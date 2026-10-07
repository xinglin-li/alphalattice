"""Cutoff intent, entry-only carry and immutable observed settlement."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest

from alphalattice.foundation.causal_outcomes.execution.contracts import LocalQAMarketSnapshot
from alphalattice.foundation.causal_outcomes.execution.readers import planned_local_qa_schedule
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
)
from alphalattice.investment.portfolio_strategy_lab.application.calibration import (
    FrozenRankCalibrationRule,
    PreparedPortfolioBookInput,
    PreparedPortfolioComponentInput,
)
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    PortfolioDecisionCheckpoint,
    PortfolioUpdatePublication,
    advance_decision_state,
    continue_issued_observations,
    make_proposal,
    require_proposal_position,
    source_revision_impact,
    transition_decision_checkpoint,
)
from alphalattice.investment.portfolio_strategy_lab.policies.post_observed_authority import (
    FrozenHistoricalBookRecipe,
    FrozenHistoricalComponentRecipe,
)
from alphalattice.investment.portfolio_strategy_lab.publication.portfolio_ledger import (
    PortfolioLedgerStore,
)
from alphalattice.investment.portfolio_strategy_lab.reporting.static import render_decision_update
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from tests.portfolio_strategy_lab.local_web_support import TEST_PACKAGE
from tests.portfolio_strategy_lab.synthetic_numerical import (
    HASH,
    build_numerical,
    prepared_for,
    snapshot_for,
)

NOW = datetime(2026, 9, 7, 23, tzinfo=UTC)


@pytest.fixture(scope="module")
def numerical():
    return build_numerical()


def test_qa_price_coverage_requires_explicit_retention_and_resolved_identifiers():
    from alphalattice.foundation.causal_outcomes.execution.readers import _qa_source_manifest

    current = SimpleNamespace(listings=(SimpleNamespace(listing_id="A"),))
    resolved = []

    def scope(manifest, *, listing_ids):
        assert manifest is current
        if not set(listing_ids) <= {"A", "B"}:
            raise ValueError("data.covered_listing_identity_unavailable")
        resolved.append(listing_ids)

    store = SimpleNamespace(
        current_quality_filtered_research_manifest=lambda **_: current, listing_scope=scope
    )
    assert _qa_source_manifest(store, ("A",), ()) is current
    with pytest.raises(ValueError, match="qa_listing_epoch_mismatch"):
        _qa_source_manifest(store, ("A", "B"), ())
    assert resolved == []
    assert _qa_source_manifest(store, ("A", "B"), ("B",)) is current
    assert resolved == [("A", "B")]
    with pytest.raises(ValueError, match="covered_listing_identity_unavailable"):
        _qa_source_manifest(store, ("A", "UNKNOWN"), ("UNKNOWN",))


def test_conditional_proposal_has_no_future_and_entry_outcome_match_the_frozen_path(numerical):
    n = numerical
    prepared, snapshot = prepared_for(n, n.first), snapshot_for(n, n.first)
    first = advance_decision_state(
        checkpoint=n.checkpoint,
        previous=None,
        prepared=prepared,
        observed=snapshot,
        plan_hash=HASH,
        published_at=NOW,
    )
    assert first.pending_proposal is not None and not first.events
    original = first.pending_proposal.model_dump_json()
    changed = n.prices.copy()
    changed[n.first + 1 :] *= np.linspace(0.7, 1.3, 80)
    assert snapshot_for(n, n.first, prices=changed) == snapshot
    assert (
        make_proposal(
            checkpoint=n.checkpoint,
            book=n.checkpoint.initial_book,
            prepared=prepared,
            observed=snapshot,
        ).model_dump_json()
        == original
    )
    previous = first
    all_events = []
    for offset in range(1, 5):
        index = n.first + offset
        previous = advance_decision_state(
            checkpoint=n.checkpoint,
            previous=previous,
            prepared=prepared_for(n, index) if offset < 4 else None,
            observed=snapshot_for(n, index),
            plan_hash=canonical_hash(offset),
            published_at=NOW,
        )
        all_events.extend(previous.events)
    entries = [e for e in all_events if e.phase == "ENTRY_SETTLED"]
    outcomes = [e for e in all_events if e.phase == "OUTCOME_SETTLED"]
    assert len(entries) == 4 and len(outcomes) == 3
    for i, entry in enumerate(entries):
        np.testing.assert_array_equal(
            entry.entry.weights, n.reference.executed_weights[n.first + i]
        )
        assert entry.turnover == n.reference.one_way_turnovers[n.first + i]
        assert entry.gross_return is None
        assert entry.entry.next_position == n.first + i + 1
    for i, outcome in enumerate(outcomes):
        assert outcome.gross_return == n.reference.gross_simple_returns[n.first + i]
    assert first.pending_proposal.model_dump_json() == original
    alternative = advance_decision_state(
        checkpoint=n.checkpoint,
        previous=first,
        prepared=None,
        observed=snapshot_for(n, n.first + 1, prices=changed),
        plan_hash=HASH,
        published_at=NOW,
    )
    assert alternative.events[-1].entry.weights != entries[0].entry.weights


def test_two_component_conditional_book_matches_the_existing_merged_walk():
    recipe = FrozenHistoricalBookRecipe(
        strategy_id=TEST_PACKAGE.strategy_id,
        components=(
            FrozenHistoricalComponentRecipe("trend", 5000, "ew", -1.0),
            FrozenHistoricalComponentRecipe("rebound", 5000, "ew"),
        ),
    )
    n = build_numerical(book_recipe=recipe, listing_ids=tuple(f"qa-{i:03d}" for i in range(130)))
    from alphalattice.control.product_host.composition.portfolio_updates import (
        PortfolioUpdateApplication,
    )
    from alphalattice.control.product_host.composition.research_workspace import held

    binding_owner = object.__new__(PortfolioUpdateApplication)
    bindings = tuple(
        SimpleNamespace(
            strategy_package_id=n.checkpoint.package.strategy_id,
            strategy_package_hash=n.checkpoint.package.package_hash,
            component_id=component_id,
            authority_hash=authority_hash,
            source_kind="WORKSPACE_DATA_FEATURE",
        )
        for component_id, authority_hash in zip(
            n.checkpoint.package.component_ids,
            n.checkpoint.model_authority_hashes,
            strict=True,
        )
    )
    binding_owner._manifests = held(SimpleNamespace(score_inputs=bindings))  # type: ignore[arg-type]
    assert binding_owner.model_bindings_match(n.checkpoint)
    binding_owner._manifests.current = SimpleNamespace(score_inputs=bindings[:1])  # type: ignore[assignment]
    assert not binding_owner.model_bindings_match(n.checkpoint)
    previous = None
    events = []
    original = None
    for offset in range(5):
        index = n.first + offset
        previous = advance_decision_state(
            checkpoint=n.checkpoint,
            previous=previous,
            prepared=prepared_for(n, index) if offset < 4 else None,
            observed=snapshot_for(n, index),
            plan_hash=canonical_hash(offset),
            published_at=NOW,
        )
        if offset == 0:
            original = previous
        events.extend(previous.events)
    entries = [event for event in events if event.phase == "ENTRY_SETTLED"]
    outcomes = [event for event in events if event.phase == "OUTCOME_SETTLED"]
    assert len(entries) == 4 and len(outcomes) == 3
    for i, event in enumerate(entries):
        np.testing.assert_array_equal(
            event.entry.weights, n.reference.executed_weights[n.first + i]
        )
        assert event.turnover == n.reference.one_way_turnovers[n.first + i]
        assert event.entry.component_sleeve_counts == (3, 3)
        np.testing.assert_allclose(
            np.asarray(event.entry.sleeves).sum(axis=1).reshape(2, 3).sum(axis=1),
            (1, 1),
            atol=1e-15,
            rtol=0,
        )
    for i, event in enumerate(outcomes):
        assert event.gross_return == n.reference.gross_simple_returns[n.first + i]
    assert isinstance(original.pending_proposal.input, PreparedPortfolioBookInput)
    assert all(
        item.rule is None and not item.bucket_means
        for item in original.pending_proposal.input.components
    )


def test_corrections_keep_issued_prices_and_verify_split_conversion(numerical):
    n = numerical
    old = snapshot_for(n, n.first)
    raw = snapshot_for(n, n.first + 1)
    original = old.model_dump_json()
    changed = LocalQAMarketSnapshot.create(
        **{
            **raw.model_dump(exclude={"content_hash"}),
            "bars": tuple(
                replace(value, close=value.close * 1.2)
                if value.session_date <= old.through
                else value
                for value in raw.bars
            ),
        }
    )
    bridge = continue_issued_observations(old, changed)
    assert tuple(value for value in bridge.bars if value.session_date <= old.through) == old.bars
    assert tuple(value for value in bridge.bars if value.session_date > old.through) == tuple(
        value for value in raw.bars if value.session_date > old.through
    )
    listing = old.ordered_listing_ids[0]
    split = CorporateActionEvent(listing, "SYNTHETIC_QA", raw.through, "SPLIT", 2.0)
    adjusted = LocalQAMarketSnapshot.create(
        **{
            **raw.model_dump(exclude={"content_hash"}),
            "bars": tuple(
                replace(
                    value,
                    open=value.open / 2,
                    high=value.high / 2,
                    low=value.low / 2,
                    close=value.close / 2,
                )
                if value.listing_id == listing
                else value
                for value in raw.bars
            ),
            "actions": (split,),
        }
    )
    converted = continue_issued_observations(old, adjusted)
    assert converted.bars == adjusted.bars
    factual = LocalQAMarketSnapshot.create(
        source_hash=HASH,
        through=old.through,
        ordered_listing_ids=old.ordered_listing_ids,
        schedule=old.schedule,
        bars=tuple(
            replace(
                value,
                open=value.open * 1.02,
                high=value.high * 1.02,
                low=value.low * 1.02,
                close=value.close * 1.02,
            )
            if value.listing_id == listing
            else value
            for value in old.bars
        ),
        actions=(),
    )
    carried = continue_issued_observations(old, factual)
    revised_split = LocalQAMarketSnapshot.create(
        source_hash=HASH,
        through=raw.through,
        ordered_listing_ids=raw.ordered_listing_ids,
        schedule=raw.schedule,
        actions=(split,),
        bars=tuple(
            replace(
                value,
                open=value.open * 1.02,
                high=value.high * 1.02,
                low=value.low * 1.02,
                close=value.close * 1.02,
            )
            if value.listing_id == listing
            else value
            for value in adjusted.bars
        ),
    )
    after = continue_issued_observations(carried, revised_split, raw_previous=factual)
    assert tuple(value for value in after.bars if value.session_date <= old.through) == tuple(
        value for value in adjusted.bars if value.session_date <= old.through
    )
    assert old.model_dump_json() == original
    invalid = LocalQAMarketSnapshot.create(
        **{**raw.model_dump(exclude={"content_hash"}), "actions": (split,)}
    )
    with pytest.raises(ValueError, match="split_conversion_unverified"):
        continue_issued_observations(old, invalid)
    revised_past = LocalQAMarketSnapshot.create(
        **{
            **raw.model_dump(exclude={"content_hash"}),
            "actions": (replace(split, effective_date=old.through),),
        }
    )
    with pytest.raises(ValueError, match="revised_split_basis_requires_review"):
        continue_issued_observations(old, revised_past)
    assert old.model_dump_json() == original

    first = advance_decision_state(
        checkpoint=n.checkpoint,
        previous=None,
        prepared=prepared_for(n, n.first),
        observed=old,
        plan_hash=HASH,
        published_at=NOW,
    )
    revised_close = LocalQAMarketSnapshot.create(
        source_hash=HASH,
        through=old.through,
        ordered_listing_ids=old.ordered_listing_ids,
        schedule=old.schedule,
        bars=tuple(replace(value, close=value.close * 1.2) for value in old.bars),
        actions=(),
    )
    comparison = source_revision_impact(old, revised_close)
    continued = advance_decision_state(
        checkpoint=n.checkpoint,
        previous=first,
        prepared=None,
        observed=continue_issued_observations(old, revised_close),
        plan_hash=HASH,
        published_at=NOW,
        source_revision=comparison,
    )
    assert not continued.events
    assert continued.book == first.book
    assert continued.pending_proposal == first.pending_proposal
    assert comparison.changed_bar_count == len(old.bars)
    assert "Revised replay was not performed or adopted" in render_decision_update(
        n.checkpoint, (first, continued)
    )


def test_axis_transition_settles_sealed_intent_before_reindexing(numerical):
    n = numerical
    old = advance_decision_state(
        checkpoint=n.checkpoint,
        previous=None,
        prepared=prepared_for(n, n.first),
        observed=snapshot_for(n, n.first),
        plan_hash=HASH,
        published_at=NOW,
    )
    assert old.pending_proposal is not None
    original = old.model_dump_json()
    removed = n.checkpoint.ordered_listing_ids[0]
    candidates = {value: value for value in n.checkpoint.ordered_listing_ids if value != removed}
    candidates["zz-new"] = "NEW"
    epoch = transition_decision_checkpoint(
        n.checkpoint, candidate_labels=candidates, source_hash=HASH
    )
    assert removed in epoch.ordered_listing_ids
    assert epoch.history_hash == n.checkpoint.content_hash
    raw = snapshot_for(n, n.first + 1)
    expanded = LocalQAMarketSnapshot.create(
        **{
            **raw.model_dump(exclude={"content_hash"}),
            "ordered_listing_ids": epoch.ordered_listing_ids,
            "bars": tuple(
                sorted(
                    (
                        *raw.bars,
                        *(
                            replace(value, listing_id="zz-new")
                            for value in raw.bars
                            if value.listing_id == removed
                        ),
                    ),
                    key=lambda value: (value.session_date, value.listing_id),
                )
            ),
        }
    )
    prepared = prepared_for(n, n.first + 1)
    axis = tuple(sorted(candidates))
    column = {value: i for i, value in enumerate(prepared.ordered_listing_ids)}
    updated = PreparedPortfolioComponentInput.create(
        **{
            **prepared.model_dump(exclude={"content_hash"}),
            "ordered_listing_ids": axis,
            "scores": tuple(
                prepared.scores[column[value]] if value in column else 0.1 for value in axis
            ),
            "decision_eligible": (True,) * len(axis),
        }
    )
    result = advance_decision_state(
        checkpoint=epoch,
        previous_checkpoint=n.checkpoint,
        previous=old,
        prepared=updated,
        observed=expanded,
        plan_hash=HASH,
        published_at=NOW,
    )
    np.testing.assert_array_equal(result.book.weights[:-1], n.reference.executed_weights[n.first])
    assert result.book.weights[-1] == 0
    assert result.events[0].proposal_hash == old.pending_proposal.content_hash
    assert result.input_checkpoint_hash == epoch.content_hash
    assert result.book.next_position == old.book.next_position + 1
    assert result.pending_proposal.input == updated
    assert old.model_dump_json() == original


def test_unheld_ineligible_coverage_needs_no_price_but_held_exposure_still_does(numerical):
    from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
        settle_entry,
    )

    n = numerical
    book = n.checkpoint.initial_book
    index = next(
        i
        for i in range(len(book.weights))
        if all(row[i] == 0 for row in (book.weights, *book.sleeves))
    )
    listing = n.checkpoint.ordered_listing_ids[index]
    base = prepared_for(n, n.first)
    prepared = PreparedPortfolioComponentInput.create(
        **{
            **{key: getattr(base, key) for key in type(base).model_fields if key != "content_hash"},
            "decision_eligible": tuple(
                value and i != index for i, value in enumerate(base.decision_eligible)
            ),
        }
    )

    def without(snapshot, missing):
        return LocalQAMarketSnapshot.create(
            source_hash=snapshot.source_hash,
            through=snapshot.through,
            ordered_listing_ids=snapshot.ordered_listing_ids,
            schedule=snapshot.schedule,
            actions=snapshot.actions,
            bars=tuple(v for v in snapshot.bars if v.listing_id != missing),
        )

    full_close = snapshot_for(n, n.first)
    kwargs = dict(checkpoint=n.checkpoint, book=book, prepared=prepared)
    full = make_proposal(**kwargs, observed=full_close)
    absent = make_proposal(**kwargs, observed=without(full_close, listing))
    assert absent.estimated_weights == full.estimated_weights
    full_entry = snapshot_for(n, n.first + 1)
    expected = settle_entry(checkpoint=n.checkpoint, proposal=full, observed=full_entry)
    actual = settle_entry(
        checkpoint=n.checkpoint, proposal=absent, observed=without(full_entry, listing)
    )
    assert actual.entry.weights == expected.entry.weights
    assert actual.turnover == expected.turnover
    assert actual.missed_executions == expected.missed_executions
    assert actual.entry.weights[index] == 0
    held = n.checkpoint.ordered_listing_ids[next(i for i, v in enumerate(book.weights) if v > 0)]
    with pytest.raises(ValueError, match="close_mark_unavailable"):
        make_proposal(**kwargs, observed=without(full_close, held))


def test_an_update_too_short_for_a_rebalance_is_refused_before_it_proposes(numerical):
    """requirement (V519, V500's class): a research update whose session leaves fewer
    tradable, scored names than the active book selects per rebalance is refused by that
    session before it proposes, in the door's words with its way on, never inside the
    decision."""

    from alphalattice.interface.local_application.cli_contract import refusal_words

    n = numerical
    base = prepared_for(n, n.first)
    selected = n.checkpoint.recipe.top_k
    eligible = [i for i, value in enumerate(base.decision_eligible) if value]
    assert len(eligible) >= selected
    kept = set(eligible[: selected - 1])
    prepared = PreparedPortfolioComponentInput.create(
        **{
            **{key: getattr(base, key) for key in type(base).model_fields if key != "content_hash"},
            "decision_eligible": tuple(i in kept for i in range(len(base.decision_eligible))),
        }
    )
    with pytest.raises(ValueError) as refused:
        make_proposal(
            checkpoint=n.checkpoint,
            book=n.checkpoint.initial_book,
            prepared=prepared,
            observed=snapshot_for(n, n.first),
        )
    code = str(refused.value)
    assert code == f"portfolio_update.eligible_pool_short:{base.formation_session}"
    words = refusal_words(code)
    assert str(base.formation_session) in words["detail"] and words["next_action"]


def test_a_persons_activation_names_its_time_and_book_and_a_qa_admission_neither(numerical):
    """requirement (LS1, OW12): a checkpoint that runs a strategy forward names the person's
    activation, its time and the book it continues; an operator's QA admission names neither,
    and its sealed form, so its identity, is the one it had before (the fields are left out)."""

    checkpoint = numerical.checkpoint
    assert checkpoint.purpose == "POST_OBSERVED_QA_NOT_FORWARD_ADMISSION"
    sealed = checkpoint.model_dump(mode="json")
    assert "activated_at" not in sealed and "book_task_id" not in sealed
    raw = {k: getattr(checkpoint, k) for k in type(checkpoint).model_fields if k != "content_hash"}
    book = UUID("4d9c3ad6-3c4b-4d1c-9a55-8e1f2b5f0c11")
    forward = PortfolioDecisionCheckpoint.create(
        **{
            **raw,
            "purpose": "PERSON_ACTIVATED_FORWARD_RESEARCH",
            "activated_at": NOW,
            "book_task_id": book,
        }
    )
    assert forward.content_hash != checkpoint.content_hash
    for broken in (
        {"purpose": "PERSON_ACTIVATED_FORWARD_RESEARCH"},
        {"purpose": "PERSON_ACTIVATED_FORWARD_RESEARCH", "activated_at": NOW},
        {"activated_at": NOW, "book_task_id": book},
        {
            "purpose": "PERSON_ACTIVATED_FORWARD_RESEARCH",
            "activated_at": NOW.replace(tzinfo=None),
            "book_task_id": book,
        },
    ):
        with pytest.raises(ValueError, match="checkpoint_activation_invalid"):
            PortfolioDecisionCheckpoint.create(**{**raw, **broken})


def test_a_calibrated_input_is_admitted_only_with_its_axis_activation(numerical):
    """requirement (LS1): the calibration rule's activation is the book's 521st formation on
    the axis it serves, held where the axis is known: a checkpoint admits a prepared input only
    when its formations name that rule's session at that position."""

    n = numerical
    prepared = prepared_for(n, n.first)
    assert n.checkpoint.formation_sessions[521] == prepared.rule.activation_session
    require_proposal_position(
        checkpoint=n.checkpoint,
        prepared=prepared,
        position=n.first,
        expected_session=n.sessions[n.first],
    )
    moved = PreparedPortfolioComponentInput.create(
        **{
            **prepared.model_dump(exclude={"content_hash", "rule"}),
            "rule": FrozenRankCalibrationRule.create(
                activation_session=n.checkpoint.formation_sessions[520]
            ),
        }
    )
    with pytest.raises(ValueError, match="component_input_policy_invalid"):
        require_proposal_position(
            checkpoint=n.checkpoint,
            prepared=moved,
            position=n.first,
            expected_session=n.sessions[n.first],
        )


def test_checkpoint_input_epoch_and_pending_guards(numerical):
    n = numerical
    checkpoint = n.checkpoint
    raw = checkpoint.model_dump(mode="json")
    raw["initial_book"]["cash"] = 0.5
    with pytest.raises(ValueError):
        PortfolioDecisionCheckpoint.model_validate(raw)
    prepared = prepared_for(n, n.first)
    require_proposal_position(
        checkpoint=checkpoint,
        prepared=prepared,
        position=n.first,
        expected_session=n.sessions[n.first],
    )
    with pytest.raises(ValueError, match="proposal_input_or_position"):
        require_proposal_position(
            checkpoint=checkpoint,
            prepared=prepared_for(n, n.first + 2),
            position=n.first + 1,
            expected_session=n.sessions[n.first + 1],
        )
    first = advance_decision_state(
        checkpoint=checkpoint,
        previous=None,
        prepared=prepared,
        observed=snapshot_for(n, n.first),
        plan_hash=HASH,
        published_at=NOW,
    )
    with pytest.raises(ValueError, match="previous_entry_pending"):
        advance_decision_state(
            checkpoint=checkpoint,
            previous=first,
            prepared=prepared,
            observed=snapshot_for(n, n.first),
            plan_hash=HASH,
            published_at=NOW,
        )
    with pytest.raises(ValueError, match="proposal_input_or_position"):
        make_proposal(
            checkpoint=checkpoint,
            book=checkpoint.initial_book,
            prepared=prepared_for(n, n.first + 1),
            observed=snapshot_for(n, n.first + 1),
        )
    raw = first.model_dump(mode="json")
    raw["pending_proposal"]["estimated_weights"][0] += 0.01
    with pytest.raises(ValueError):
        PortfolioUpdatePublication.model_validate(raw)


def test_publication_commit_recovers_an_orphan_and_retains_exact_html(numerical, tmp_path):
    n = numerical
    store = PortfolioLedgerStore(tmp_path)
    store.publish_decision_checkpoint(n.checkpoint)
    value = advance_decision_state(
        checkpoint=n.checkpoint,
        previous=None,
        prepared=prepared_for(n, n.first),
        observed=snapshot_for(n, n.first),
        plan_hash=HASH,
        published_at=NOW,
    )
    page, _ = store.publish_html(render_decision_update(n.checkpoint, (value,)))
    fields = {
        name: getattr(value, name) for name in type(value).model_fields if name != "content_hash"
    }
    value = PortfolioUpdatePublication.create(**{**fields, "html_hash": page})
    store.publish_decision_update(value)
    path = store.root / "decision-updates" / f"{value.content_hash}.json"
    original = path.read_bytes()
    path.unlink()  # Only this test's materialized content; commit remains durable.
    assert store.decision_history(n.checkpoint.content_hash) == (value,)
    assert path.read_bytes() == original
    assert "CLOSE_MARKED_ESTIMATE" in store.load_html(page)
    payload = json.loads(original)
    payload["observed_through"] = "2026-07-24"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity_reused"):
        store.decision_history(n.checkpoint.content_hash)


def test_exact_reuse_revalidates_observation_and_html_children(numerical, tmp_path, monkeypatch):
    """Task admission is supplied here; both artifact readers are the real store."""
    from alphalattice.control.product_host.composition.portfolio_updates import (
        TASK_KIND,
        PortfolioUpdateApplication,
    )
    from alphalattice.control.task_control.contracts import TaskLifecycle

    n = numerical
    store = PortfolioLedgerStore(tmp_path)
    observed = snapshot_for(n, n.first)
    store.content.publish_model(
        category="decision-observations", value=observed, identity_field="content_hash"
    )
    html_hash, _ = store.publish_html("<p>verified QA report</p>")
    plan = SimpleNamespace(
        market_hash=observed.content_hash,
        observed_through=observed.through,
        raw_market_hash=None,
    )
    publication = SimpleNamespace(html_hash=html_hash)
    task = SimpleNamespace(task_kind=TASK_KIND, lifecycle=TaskLifecycle.SUCCEEDED)
    app = object.__new__(PortfolioUpdateApplication)
    app.store = store
    app.session = SimpleNamespace(task_control_registry=SimpleNamespace(tasks=lambda: (task,)))
    monkeypatch.setattr(app, "_require", lambda plan: None)
    monkeypatch.setattr(app, "_publication", lambda plan: publication)
    monkeypatch.setattr(app, "_plan_of", lambda task, current=False: plan)
    assert app.reusable(plan) is publication
    for path in (
        store.root / "html" / f"{html_hash}.html",
        store.root / "decision-observations" / f"{observed.content_hash}.json",
    ):
        original = path.read_bytes()
        path.write_bytes(b"tampered")
        try:
            with pytest.raises(ValueError, match="artifact_tampered"):
                app.reusable(plan)
        finally:
            path.write_bytes(original)
    assert app.reusable(plan) is publication


def test_captured_prefix_and_tail_first_publication(numerical, tmp_path):
    from alphalattice.foundation.causal_outcomes.execution.readers import (
        local_qa_prefix,
        local_qa_snapshot_rows,
    )

    n = numerical
    latest = snapshot_for(n, n.first + 2)
    earlier = local_qa_prefix(latest, through=n.sessions[n.first])
    assert earlier.bars == snapshot_for(n, n.first).bars
    bounded = local_qa_prefix(latest, start=n.sessions[n.first - 1], through=n.sessions[n.first])
    assert (
        bounded.schedule[:2]
        == planned_local_qa_schedule(n.sessions[n.first - 1], n.sessions[n.first])[:2]
    )
    rows, bars, axis = local_qa_snapshot_rows(
        latest, sessions=n.sessions[n.first - 1 : n.first + 1], through=n.sessions[n.first]
    )
    assert rows.num_rows == 0  # Neither of these formation labels has matured.
    assert max(d for d, _ in bars) == n.sessions[n.first]
    assert axis[-1] == n.sessions[n.first]
    store = PortfolioLedgerStore(tmp_path)
    previous = None
    branch = []
    for i in range(n.first, n.first + 3):
        value = advance_decision_state(
            checkpoint=n.checkpoint,
            previous=previous,
            prepared=prepared_for(n, i),
            observed=snapshot_for(n, i),
            plan_hash=HASH,
            published_at=NOW,
        )
        html, _ = store.publish_html(render_decision_update(n.checkpoint, (*branch, value)))
        value = PortfolioUpdatePublication.create(
            **{
                **{k: getattr(value, k) for k in type(value).model_fields if k != "content_hash"},
                "html_hash": html,
            }
        )
        branch.append(value)
        previous = value
    for value in reversed(branch[1:]):
        store.publish_decision_update(value)
        assert store.decision_history(n.checkpoint.content_hash) == ()
    store.publish_decision_update(branch[0])
    assert store.decision_history(n.checkpoint.content_hash) == tuple(branch)


def _captured_rows_snapshot(*, quote="normal", actions=()):
    from alphalattice.foundation.market_data_ops.sources.contracts import RawDailyBar

    schedule = planned_local_qa_schedule(date(2026, 9, 1), date(2026, 9, 11))
    days = tuple(v.formation_session for v in schedule if v.formation_session <= date(2026, 9, 11))
    bars = tuple(
        RawDailyBar(listing, "synthetic", day, 100.0 + i, 102.0 + i, 99.0 + i, 101.0 + i, 1000)
        for i, day in enumerate(days)
        for listing in ("A", "B")
    )
    affected = (days[2], "A")
    if quote == "missing":
        bars = tuple(v for v in bars if (v.session_date, v.listing_id) != affected)
    elif quote != "normal":
        value = {"zero": 0.0, "nan": float("nan"), "infinity": float("inf")}[quote]
        bars = tuple(
            replace(v, open=value) if (v.session_date, v.listing_id) == affected else v
            for v in bars
        )
    return LocalQAMarketSnapshot.create(
        source_hash=HASH,
        through=days[-1],
        ordered_listing_ids=("A", "B"),
        schedule=schedule,
        bars=bars,
        actions=actions,
    )


@pytest.mark.parametrize("quote", ["normal", "missing", "zero", "nan", "infinity"])
def test_prepared_captured_rows_keep_exact_bytes_bounds_and_independent_results(quote):
    import pyarrow as pa

    from alphalattice.foundation.causal_outcomes.execution.readers import (
        PreparedLocalQASnapshotRows,
        local_qa_snapshot_rows,
    )

    source = _captured_rows_snapshot(
        quote=quote,
        actions=(
            CorporateActionEvent(
                "A", "synthetic", date(2026, 9, 3), "CASH_DIVIDEND", cash_amount=0.25
            ),
            CorporateActionEvent(
                "B", "synthetic", date(2026, 9, 4), "SPLIT", new_shares_per_old_share=2.0
            ),
        ),
    )
    prepared = PreparedLocalQASnapshotRows(source)

    def encoded(table):
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        return sink.getvalue().to_pybytes()

    days = tuple(
        v.formation_session for v in source.schedule if v.formation_session <= source.through
    )
    for cutoff in days:
        sessions = tuple(d for d in days if d <= cutoff)
        expected, expected_bars, expected_axis = local_qa_snapshot_rows(
            source, sessions=sessions, through=cutoff
        )
        actual, bars, axis = local_qa_snapshot_rows(prepared, sessions=sessions, through=cutoff)
        assert encoded(actual) == encoded(expected)
        assert actual.schema == expected.schema
        for name in ("entry_source_row_hash", "holding_end_source_row_hash", "row_hash"):
            assert actual[name].to_pylist() == expected[name].to_pylist()
        assert all(column.num_chunks == 1 for column in actual.columns)
        assert all(
            not buffer.is_mutable
            for column in actual.columns
            for buffer in column.chunk(0).buffers()
            if buffer is not None
        )
        assert axis == expected_axis
        assert tuple((key, canonical_hash(asdict(value))) for key, value in bars.items()) == tuple(
            (key, canonical_hash(asdict(value))) for key, value in expected_bars.items()
        )
        assert max(day for day, _ in bars) == cutoff
        assert all(day <= cutoff for day in actual["holding_end_session"].to_pylist())
        bars.clear()
        again, fresh_bars, fresh_axis = prepared.rows(sessions=sessions, through=cutoff)
        assert encoded(again) == encoded(expected) and fresh_bars and fresh_axis == axis
        assert again is not actual
    assert prepared.rows(sessions=days[-2:], through=days[-1])[0].num_rows == 0

    # Smaller and changed request axes fall back, then can themselves grow.
    for sessions, cutoff in (
        (days, days[-1]),
        (days[:4], days[3]),
        (days[1:], days[-1]),
        (days[1::2], days[-1]),
        (days[:-1], days[-2]),
        (days, days[-1]),
    ):
        expected, expected_bars, expected_axis = local_qa_snapshot_rows(
            source, sessions=sessions, through=cutoff
        )
        actual, bars, axis = prepared.rows(sessions=sessions, through=cutoff)
        assert encoded(actual) == encoded(expected)
        assert axis == expected_axis
        assert tuple((key, canonical_hash(asdict(value))) for key, value in bars.items()) == tuple(
            (key, canonical_hash(asdict(value))) for key, value in expected_bars.items()
        )

    # Returned frozen observations are independent of the caller's source values.
    full_bars = prepared.rows(sessions=days, through=days[-1])[1]
    assert all(full_bars[bar.session_date, bar.listing_id] is not bar for bar in source.bars)


def test_prepared_captured_rows_keep_refusals_and_reject_forged_sources():
    from alphalattice.foundation.causal_outcomes.execution.readers import (
        PreparedLocalQASnapshotRows,
        local_qa_snapshot_rows,
    )

    source = _captured_rows_snapshot(
        actions=(CorporateActionEvent("A", "synthetic", date(2026, 9, 10), "CAPITAL_GAIN"),)
    )
    prepared = PreparedLocalQASnapshotRows(source)
    sessions = (date(2026, 9, 1), date(2026, 9, 2))
    assert prepared.rows(sessions=sessions, through=date(2026, 9, 9))[0].num_rows == 4
    for captured in (source, prepared):
        for cutoff in (date(2026, 9, 10), source.through):
            with pytest.raises(ValueError, match="unsupported corporate action"):
                local_qa_snapshot_rows(captured, sessions=sessions, through=cutoff)
        for axis in ((), sessions[::-1], (sessions[0], sessions[0]), (date(2026, 8, 31),)):
            with pytest.raises(ValueError, match="qa_session_axis_invalid"):
                local_qa_snapshot_rows(captured, sessions=axis, through=date(2026, 9, 9))
        for cutoff in (date(2026, 9, 12), date(2026, 9, 7)):
            with pytest.raises(ValueError, match="qa_calendar_support_absent"):
                local_qa_snapshot_rows(captured, sessions=sessions, through=cutoff)
    assert prepared.rows(sessions=sessions, through=date(2026, 9, 9))[0].num_rows == 4
    for changed in (
        {"content_hash": "0" * 64},
        {"ordered_listing_ids": ("B", "A")},
        {"bars": (*source.bars, source.bars[0])},
        {"bars": source.bars[::-1]},
        {"bars": (replace(source.bars[0], listing_id="UNKNOWN"), *source.bars[1:])},
        {"bars": (replace(source.bars[0], session_date=date(2026, 9, 12)), *source.bars[1:])},
        {"schedule": source.schedule[::-1]},
        {"actions": (replace(source.actions[0], effective_date=date(2026, 9, 12)),)},
        {"bars": (replace(source.bars[0], open=123.0), *source.bars[1:])},
    ):
        with pytest.raises(ValueError, match="qa_market_snapshot_invalid"):
            PreparedLocalQASnapshotRows(source.model_copy(update=changed))


def test_frozen_input_extension_refuses_past_corrections_not_future_rows():
    from dataclasses import replace

    from alphalattice.investment.alpha_research.inputs.frozen_price_volume import (
        FrozenPriceVolumeInputs,
    )

    def source(days):
        return FrozenPriceVolumeInputs(
            formation_sessions=days,
            ordered_listing_ids=("a", "b"),
            sector_by_listing_id={"a": "s", "b": "s"},
            **{
                name: np.ones((len(days), 2)) for name in ("open", "high", "low", "close", "volume")
            },
            market_context_values=np.ones((len(days), 3)),
            sector_trend_values=np.ones((len(days), 1)),
            source_binding_hash=HASH,
        )

    prior = source((date(2026, 8, 3), date(2026, 8, 4)))
    current = source((*prior.formation_sessions, date(2026, 8, 5)))
    values = current.close.copy()
    values[-1] = 1000
    replace(current, close=values).require_unchanged_prefix(prior)
    values[0, 0] = 2
    with pytest.raises(ValueError, match="frozen_input_prefix_changed"):
        replace(current, close=values).require_unchanged_prefix(prior)


def _performance_projection(n, tmp_path, prefix):
    from alphalattice.control.product_host.composition.portfolio_updates import (
        PortfolioUpdateApplication,
    )
    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceManifest,
    )

    owner = PortfolioUpdateApplication(
        application=SimpleNamespace(session=None, ledger=PortfolioLedgerStore(tmp_path)),
        calibration=SimpleNamespace(),
        manifest=ResearchWorkspaceManifest.create(
            workspace_id="synthetic-performance",
            default_strategy_package_id=None,
            default_score_source_mode=None,
            strategy_artifacts=(),
            strategy_installation="NOT_INSTALLED",
        ),
        clock=lambda: NOW,
    )
    return owner.publication_body(n.checkpoint, prefix[-1], tuple(prefix), task_id=UUID(int=1))[
        "realized_performance"
    ]


@pytest.mark.parametrize("quote", ("5", "10"))
def test_daily_performance_reads_only_realized_outcomes_and_pins_the_issued_prefix(
    numerical, tmp_path, quote
):
    """BEHAVIOUR: each sealed daily outcome extends its own strategy/cost window."""
    n = numerical
    prefix, previous = [], None
    for offset in range(5):
        index = n.first + offset
        previous = advance_decision_state(
            checkpoint=n.checkpoint,
            previous=previous,
            prepared=prepared_for(n, index),
            observed=snapshot_for(n, index),
            plan_hash=canonical_hash(offset),
            published_at=NOW,
        )
        prefix.append(previous)
    zero = _performance_projection(n, tmp_path, prefix[:2])["cost_lanes"][quote]
    assert zero["series"] == []  # proposal and ENTRY_SETTLED carry no realized returns
    assert zero["selected_window_metrics"] == {"cost_bps": float(quote) * 2}
    assert all(
        row["reason"] == "INSUFFICIENT_REALIZED_OBSERVATIONS"
        for row in zero["selected_window_metric_absences"].values()
    )
    one = _performance_projection(n, tmp_path, prefix[:3])["cost_lanes"][quote]
    assert len(one["series"]) == 1
    assert "sharpe" not in one["selected_window_metrics"]
    assert one["selected_window_metric_absences"]["sharpe"]["reason"] == (
        "INSUFFICIENT_REALIZED_OBSERVATIONS"
    )
    pinned = _performance_projection(n, tmp_path, prefix[:4])
    latest = _performance_projection(n, tmp_path, prefix)
    first, second = pinned["cost_lanes"][quote], latest["cost_lanes"][quote]
    assert len(first["series"]) == 2 and len(second["series"]) == 3
    returns = np.asarray([row["net_simple_return"] for row in second["series"]])
    assert second["selected_window_metrics"]["annualized_return"] == pytest.approx(
        np.expm1(np.log1p(returns).sum() * 252 / len(returns))
    )
    assert first["selected_window_metrics"] != second["selected_window_metrics"]
    provenance = second["selected_window_metric_provenance"]
    assert provenance["publication_hash"] == prefix[-1].content_hash
    assert provenance["strategy_package_hash"] == n.checkpoint.package.package_hash
    assert provenance["selected_start"] == second["series"][0]["entry_session"]
    assert provenance["selected_end"] == second["series"][-1]["holding_end_session"]
    assert provenance["annualization_sessions_per_year"] == 252
    assert provenance["sharpe_cash_return_per_session"] == 0
    assert provenance["sortino_downside_threshold"] == 0
    assert pinned == _performance_projection(n, tmp_path, prefix[:4])
    alien = PortfolioUpdatePublication.create(
        **{
            **{
                name: getattr(prefix[-1], name)
                for name in type(prefix[-1]).model_fields
                if name != "content_hash"
            },
            "checkpoint_hash": canonical_hash("another strategy's checkpoint"),
        }
    )
    with pytest.raises(ValueError, match="performance_history_binding_invalid"):
        _performance_projection(n, tmp_path, [*prefix[:-1], alien])


@pytest.mark.parametrize("condition", ("no_downside", "nonfinite_metric", "overlap", "long_span"))
def test_daily_performance_names_absence_instead_of_annualizing_an_invalid_axis(
    numerical, tmp_path, condition
):
    """BEHAVIOUR: absent/nonfinite ratios and overlapping/non-daily paths stay absent."""
    n = numerical
    prefix, previous = [], None
    for offset in range(4):
        index = n.first + offset
        value = advance_decision_state(
            checkpoint=n.checkpoint,
            previous=previous,
            prepared=prepared_for(n, index),
            observed=snapshot_for(n, index),
            plan_hash=canonical_hash(offset),
            published_at=NOW,
        )
        events = []
        for event in value.events:
            if event.phase == "OUTCOME_SETTLED":
                values = {
                    name: getattr(event, name)
                    for name in type(event).model_fields
                    if name != "content_hash"
                }
                if condition in {"no_downside", "nonfinite_metric"}:
                    amount = (0.1 if condition == "no_downside" else 1e6) * (offset - 1)
                    values.update(
                        gross_return=amount, net_return_5bps=amount, net_return_10bps=amount
                    )
                if condition == "long_span":
                    entry = {
                        name: getattr(event.entry, name)
                        for name in type(event.entry).model_fields
                        if name != "content_hash"
                    }
                    entry["schedule"] = type(event.entry.schedule).model_validate(
                        {**event.entry.schedule.model_dump(), "actual_session_span": 3}
                    )
                    values["entry"] = type(event.entry).create(**entry)
                if condition == "overlap" and offset == 3:
                    repeated = prefix[-1].events[0]
                    values = {
                        name: getattr(repeated, name)
                        for name in type(repeated).model_fields
                        if name != "content_hash"
                    }
                event = type(event).create(**values)
            events.append(event)
        value = PortfolioUpdatePublication.create(
            **{
                **{
                    name: getattr(value, name)
                    for name in type(value).model_fields
                    if name != "content_hash"
                },
                "events": tuple(events),
            }
        )
        prefix.append(value)
        previous = value
    with np.errstate(over="ignore"):
        lane = _performance_projection(n, tmp_path, prefix)["cost_lanes"]["5"]
    if condition in {"overlap", "long_span"}:
        assert lane["selected_window_metrics"] == {"cost_bps": 10.0}
        assert all(
            row["reason"] == "NONCONTIGUOUS_OR_NONDAILY_REALIZED_RETURN_AXIS"
            for row in lane["selected_window_metric_absences"].values()
        )
        assert lane["selected_window_metric_provenance"]["annualization_sessions_per_year"] is None
    else:
        field = "sortino" if condition == "no_downside" else "annualized_return"
        assert field not in lane["selected_window_metrics"]
        assert lane["selected_window_metric_absences"][field]["reason"] == (
            "ZERO_DOWNSIDE_DEVIATION" if condition == "no_downside" else "NONFINITE_DERIVED_METRIC"
        )
