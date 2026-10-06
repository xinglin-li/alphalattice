from __future__ import annotations

# The ignored case adds playpen/src explicitly.
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.market_data_ops.returns.semantic_revisions import (
    AdjustedReturnSemanticDelta,
    AdjustedReturnSemanticRevision,
)
from alphalattice.foundation.research_foundation.runtime.delta import (
    ListingDataDeltaKind,
    ObservedPreResearchCandidateSnapshot,
    PanelDeltaDisposition,
    PreResearchCasError,
    PreResearchHeadStore,
    PreResearchListingObservation,
    PreResearchQuarantine,
    PreResearchRevisionDisposition,
    PreResearchStorageEvidence,
    QuarantineReason,
    build_candidate_snapshot,
    build_verified_head,
    compile_update_plan,
    compute_observed_delta,
    confirm_update_plan,
    stage_qualified_delta,
)
from alphalattice.foundation.research_foundation.runtime.delta_service import (
    PRE_RESEARCH_DELTA_POLICY_HASHES,
    PreResearchDeltaService,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

PLAYPEN_ROOT = Path(__file__).resolve().parents[2]

NOW = datetime(2026, 8, 6, 15, 0, tzinfo=UTC)


def _head(
    *,
    listings: tuple[str, ...] = ("a", "b"),
    source_listings: tuple[str, ...] | None = None,
    marker: str = "m",
):
    source_listings = source_listings or listings
    return build_verified_head(
        kind="VerifiedPreResearchHead",
        market_profile_id="us-current",
        manifest_revision=canonical_hash(["manifest", marker]),
        source_revision=canonical_hash(["source", marker]),
        membership_fingerprint=canonical_hash(source_listings),
        ordered_source_candidate_listing_ids=source_listings,
        ordered_listing_ids=listings,
        listing_set_hash=canonical_hash(listings),
        research_history_start=date(2016, 8, 1),
        target_market_session=date(2026, 7, 31),
        sector_revision=canonical_hash(["sector", marker]),
        catalog_hash=canonical_hash("catalog"),
        panel_snapshot_hash=canonical_hash(["panel", marker]),
        factor_screening_result_hash=canonical_hash(["screening", marker]),
        factor_candidate_slate_hash=canonical_hash(["slate", marker]),
        research_desk_factor_input_hash=canonical_hash(["input", marker]),
        causal_execution_outcome_snapshot_hash=canonical_hash(["causal", marker]),
        research_foundation_hash=canonical_hash(["foundation", marker]),
        research_foundation_marker_hash=canonical_hash(["foundation-marker", marker]),
        ordered_factor_ids=("factor_a", "factor_b"),
        data_state_hash=canonical_hash(["data", marker]),
        validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
        is_point_in_time_historical=False,
        verified_at=NOW,
    )


def _candidate(
    head,
    *,
    listings: tuple[str, ...] | None = None,
    observations: tuple[PreResearchListingObservation, ...] | None = None,
    source_revision: str | None = None,
    policy_hashes: tuple[str, ...] = (canonical_hash("policy"),),
) -> ObservedPreResearchCandidateSnapshot:
    members = listings or head.ordered_listing_ids
    if observations is None:
        observations = tuple(
            PreResearchListingObservation(
                listing_id=value,
                source_member=True,
                local_history_start=head.research_history_start,
                local_history_end=date(2026, 7, 31),
                data_state_hash=canonical_hash([value, "data"]),
            )
            for value in sorted(members)
        )
    return build_candidate_snapshot(
        kind="ObservedPreResearchCandidateSnapshot",
        market_profile_id=head.market_profile_id,
        base_head_hash=head.head_hash,
        source_revision=source_revision or head.source_revision,
        source_membership_fingerprint=canonical_hash(members),
        target_market_session=date(2026, 7, 31),
        research_history_start=head.research_history_start,
        ordered_candidate_listing_ids=members,
        listing_observations=observations,
        sector_revision=head.sector_revision,
        policy_hashes=policy_hashes,
        observed_at=NOW,
    )


def _storage() -> PreResearchStorageEvidence:
    return PreResearchStorageEvidence(
        compressed_bytes=100,
        uncompressed_bytes=1000,
        chunk_count=2,
        newly_written_chunks=1,
        content_reused_chunks=1,
        cumulative_reachable_panel_bytes=5000,
        reachability_class="HEAD",
    )


def test_empty_and_noop_delta_are_explicit() -> None:
    head = _head()
    noop = compute_observed_delta(head=head, candidate=_candidate(head))
    assert noop.revision_disposition is PreResearchRevisionDisposition.NOOP
    assert noop.panel_disposition is PanelDeltaDisposition.NOOP
    assert {value.kind for value in noop.listing_deltas} == {ListingDataDeltaKind.UNCHANGED}
    plan = compile_update_plan(delta=noop, policy_hashes=(canonical_hash("policy"),))
    assert plan.operations == ()
    assert not plan.user_confirmation_required

    initial = build_candidate_snapshot(
        kind="ObservedPreResearchCandidateSnapshot",
        market_profile_id="us-current",
        base_head_hash=None,
        source_revision=canonical_hash("initial-source"),
        source_membership_fingerprint=canonical_hash(("a",)),
        target_market_session=date(2026, 7, 31),
        research_history_start=date(2016, 8, 1),
        ordered_candidate_listing_ids=("a",),
        listing_observations=(PreResearchListingObservation(listing_id="a", source_member=True),),
        sector_revision=None,
        policy_hashes=(canonical_hash("policy"),),
        observed_at=NOW,
    )
    delta = compute_observed_delta(head=None, candidate=initial)
    assert delta.revision_disposition is PreResearchRevisionDisposition.INITIALIZATION_REQUIRED
    assert compile_update_plan(
        delta=delta, policy_hashes=(canonical_hash("policy"),)
    ).user_confirmation_required


def test_membership_plan_scopes_provider_work_to_additions() -> None:
    head = _head()
    observations = (
        PreResearchListingObservation(
            listing_id="b",
            source_member=True,
            local_history_start=date(2016, 8, 1),
            local_history_end=date(2026, 7, 31),
        ),
        PreResearchListingObservation(listing_id="c", source_member=True),
    )
    candidate = _candidate(
        head, listings=("b", "c"), observations=observations, source_revision=canonical_hash("new")
    )
    delta = compute_observed_delta(head=head, candidate=candidate)
    assert delta.additions == ("c",)
    assert delta.removals == ("a",)
    assert delta.retained == ("b",)
    assert delta.revision_disposition is PreResearchRevisionDisposition.MEMBERSHIP_REVISION
    plan = compile_update_plan(delta=delta, policy_hashes=(canonical_hash("policy"),))
    assert plan.full_history_addition_listing_ids == ("c",)
    assert plan.bounded_retained_listing_ids == ()
    assert plan.removed_listing_ids == ("a",)
    assert plan.require_full_panel_rebuild
    assert plan.require_fixed_factor_screening


def test_short_history_addition_is_quarantined_without_moving_history_start() -> None:
    head = _head()
    observations = tuple(
        sorted(
            (
                *(_candidate(head).listing_observations),
                PreResearchListingObservation(
                    listing_id="c",
                    source_member=True,
                    local_history_start=date(2021, 1, 4),
                    local_history_end=date(2026, 7, 31),
                ),
            ),
            key=lambda value: value.listing_id,
        )
    )
    candidate = _candidate(head, listings=("a", "b", "c"), observations=observations)
    delta = compute_observed_delta(head=head, candidate=candidate)
    assert delta.research_history_start == head.research_history_start
    assert next(value for value in delta.listing_deltas if value.listing_id == "c").kind is (
        ListingDataDeltaKind.INSUFFICIENT_RESEARCH_HISTORY
    )
    staged = stage_qualified_delta(
        delta=delta,
        ordered_qualified_listing_ids=head.ordered_listing_ids,
        quarantines=(
            PreResearchQuarantine(
                listing_id="c",
                reason=QuarantineReason.INSUFFICIENT_RESEARCH_HISTORY,
                evidence_hash=canonical_hash("c-short"),
            ),
        ),
    )
    assert not staged.active_set_changed
    assert staged.revision_disposition is (
        PreResearchRevisionDisposition.SOURCE_CHANGED_ACTIVE_SET_UNCHANGED
    )


def test_removing_prior_quarantine_does_not_plan_active_universe_rebuild() -> None:
    head = _head(source_listings=("a", "b", "q"))
    candidate = _candidate(head, listings=("a", "b"), source_revision=canonical_hash("new"))

    delta = compute_observed_delta(head=head, candidate=candidate)
    plan = compile_update_plan(delta=delta, policy_hashes=(canonical_hash("policy"),))

    assert delta.removals == ("q",)
    assert delta.revision_disposition is (
        PreResearchRevisionDisposition.SOURCE_CHANGED_ACTIVE_SET_UNCHANGED
    )
    assert delta.panel_disposition is PanelDeltaDisposition.NOOP
    assert plan.removed_listing_ids == ()
    assert not plan.require_full_panel_rebuild
    assert not plan.require_fixed_factor_screening
    assert not plan.user_confirmation_required


def test_correction_and_evidence_only_have_exact_dispositions() -> None:
    head = _head()
    correction_observations = (
        PreResearchListingObservation(
            listing_id="a",
            source_member=True,
            local_history_start=date(2016, 8, 1),
            local_history_end=date(2026, 7, 31),
            adjusted_return_semantic_delta=True,
        ),
        _candidate(head).listing_observations[1],
    )
    correction = compute_observed_delta(
        head=head, candidate=_candidate(head, observations=correction_observations)
    )
    assert correction.panel_disposition is PanelDeltaDisposition.SPARSE_REBUILD
    assert compile_update_plan(
        delta=correction, policy_hashes=(canonical_hash("policy"),)
    ).bounded_retained_listing_ids == ("a",)

    evidence_observations = (
        correction_observations[0].model_copy(
            update={
                "adjusted_return_semantic_delta": False,
                "evidence_only_delta": True,
            }
        ),
        correction_observations[1],
    )
    evidence = compute_observed_delta(
        head=head, candidate=_candidate(head, observations=evidence_observations)
    )
    assert evidence.revision_disposition is (
        PreResearchRevisionDisposition.SOURCE_CHANGED_ACTIVE_SET_UNCHANGED
    )
    assert evidence.panel_disposition is PanelDeltaDisposition.NOOP
    assert not compile_update_plan(
        delta=evidence, policy_hashes=(canonical_hash("policy"),)
    ).require_foundation_publication


def test_adjusted_revision_chain_drives_semantic_delta_without_manual_classification() -> None:
    head = _head().model_copy(
        update={
            "adjusted_return_revision_cursor": 0,
            "adjusted_return_revision_chain_hash": canonical_hash([]),
        }
    )
    head = build_verified_head(
        **head.model_dump(mode="python", exclude={"head_hash"}),
    )
    identities = (
        {
            "listing_id": "a",
            "changed_return_sessions": (date(2026, 7, 30),),
            "session_set_changed": False,
            "uniform_rescale": False,
        },
        {
            "listing_id": "b",
            "changed_return_sessions": (date(2026, 8, 3),),
            "session_set_changed": True,
            "uniform_rescale": False,
        },
    )
    deltas = tuple(
        AdjustedReturnSemanticDelta(**identity, semantic_hash=canonical_hash(identity))
        for identity in identities
    )
    revision = AdjustedReturnSemanticRevision(
        cursor=len(deltas),
        chain_hash=canonical_hash([value.semantic_hash for value in deltas]),
        deltas=deltas,
    )
    assert PreResearchDeltaService._adjusted_revision_delta_kinds(head=head, current=revision) == {
        "a": ListingDataDeltaKind.ADJUSTED_RETURN_CORRECTION,
        "b": ListingDataDeltaKind.TAIL_APPEND_REQUIRED,
    }
    assert PreResearchDeltaService._adjusted_revision_impacts(head=head, current=revision) == {
        "a": (ListingDataDeltaKind.ADJUSTED_RETURN_CORRECTION, (date(2026, 7, 30),)),
        "b": (ListingDataDeltaKind.TAIL_APPEND_REQUIRED, (date(2026, 8, 3),)),
    }
    candidate = build_candidate_snapshot(
        **{
            **_candidate(head).model_dump(
                mode="python", exclude={"candidate_snapshot_hash", "observed_at"}
            ),
            "listing_observations": (
                _candidate(head)
                .listing_observations[0]
                .model_copy(
                    update={
                        "adjusted_return_semantic_delta": True,
                        "semantic_delta_kind": ListingDataDeltaKind.ADJUSTED_RETURN_CORRECTION,
                        "affected_factor_ids": ("momentum_21",),
                        "affected_sessions": (date(2026, 7, 30),),
                    }
                ),
                _candidate(head)
                .listing_observations[1]
                .model_copy(
                    update={
                        "semantic_delta_kind": ListingDataDeltaKind.TAIL_APPEND_REQUIRED,
                        "affected_factor_ids": ("momentum_21",),
                        "affected_sessions": (date(2026, 8, 3),),
                    }
                ),
            ),
            "adjusted_return_revision_cursor": revision.cursor,
            "adjusted_return_revision_chain_hash": revision.chain_hash,
            "observed_at": NOW,
        }
    )
    observed = compute_observed_delta(head=head, candidate=candidate)
    assert observed.base_adjusted_return_revision_cursor == 0
    assert observed.observed_adjusted_return_revision_cursor == 2
    assert observed.observed_adjusted_return_revision_chain_hash == revision.chain_hash
    assert observed.listing_deltas[0].affected_sessions == (date(2026, 7, 30),)
    assert observed.listing_deltas[1].kind is ListingDataDeltaKind.TAIL_APPEND_REQUIRED
    assert observed.listing_deltas[1].affected_sessions == (date(2026, 8, 3),)


def test_snapshot_identity_is_order_stable_for_sorted_observations() -> None:
    head = _head()
    candidate = _candidate(head)
    same = ObservedPreResearchCandidateSnapshot.model_validate(candidate.model_dump(mode="json"))
    assert same.candidate_snapshot_hash == candidate.candidate_snapshot_hash
    with pytest.raises(ValueError, match="sorted"):
        build_candidate_snapshot(
            **{
                **candidate.model_dump(mode="python", exclude={"candidate_snapshot_hash"}),
                "listing_observations": tuple(reversed(candidate.listing_observations)),
            }
        )


def test_semantic_replay_reuses_artifact_when_only_observation_clock_changes(tmp_path) -> None:
    head = _head()
    first = _candidate(head)
    values = {
        name: getattr(first, name)
        for name in type(first).model_fields
        if name not in {"candidate_snapshot_hash", "observed_at"}
    }
    replay = build_candidate_snapshot(
        **values,
        observed_at=NOW.replace(minute=NOW.minute + 1),
    )
    assert replay.candidate_snapshot_hash == first.candidate_snapshot_hash
    store = PreResearchHeadStore(artifact_root=tmp_path, mutation_gate=WorkspaceMutationGate())
    assert store.publish_candidate(first) == store.publish_candidate(replay)


def test_cas_does_not_overwrite_newer_head_and_freezes_confirmed_source(tmp_path) -> None:
    gate = WorkspaceMutationGate()
    store = PreResearchHeadStore(artifact_root=tmp_path, mutation_gate=gate)
    base = _head(marker="base")
    candidate = _candidate(base)
    delta = compute_observed_delta(head=base, candidate=candidate)
    plan = compile_update_plan(delta=delta, policy_hashes=(canonical_hash("policy"),))
    staged = stage_qualified_delta(
        delta=delta,
        ordered_qualified_listing_ids=base.ordered_listing_ids,
        quarantines=(),
    )
    store.publish_candidate(candidate)
    store.publish_delta(delta)
    store.publish_plan(plan)
    store.publish_staged_delta(staged)
    restarted = PreResearchHeadStore(artifact_root=tmp_path, mutation_gate=gate)
    assert restarted.load_candidate(candidate.candidate_snapshot_hash) == candidate
    assert restarted.load_delta(delta.delta_hash) == delta
    assert restarted.load_plan(plan.update_plan_hash) == plan
    assert restarted.load_staged_delta(staged.staged_delta_hash) == staged
    first_marker = store.advance_head(
        expected_base_head_hash=None,
        next_head=base,
        candidate_snapshot_hash=candidate.candidate_snapshot_hash,
        delta_hash=delta.delta_hash,
        staged_delta_hash=staged.staged_delta_hash,
        update_plan_hash=plan.update_plan_hash,
        mandate_hash=None,
        child_hashes=(base.research_foundation_hash,),
        storage_evidence=_storage(),
        published_at=NOW,
    )
    assert store.current_head() == base

    newer = _head(marker="newer")
    next_marker = store.advance_head(
        expected_base_head_hash=base.head_hash,
        next_head=newer,
        candidate_snapshot_hash=candidate.candidate_snapshot_hash,
        delta_hash=delta.delta_hash,
        staged_delta_hash=staged.staged_delta_hash,
        update_plan_hash=plan.update_plan_hash,
        mandate_hash=None,
        child_hashes=(newer.research_foundation_hash,),
        storage_evidence=_storage(),
        published_at=NOW,
    )
    assert first_marker.marker_hash != next_marker.marker_hash
    with pytest.raises(PreResearchCasError, match="HEAD_COMPARE_AND_SWAP_FAILED"):
        store.advance_head(
            expected_base_head_hash=base.head_hash,
            next_head=_head(marker="stale-writer"),
            candidate_snapshot_hash=candidate.candidate_snapshot_hash,
            delta_hash=delta.delta_hash,
            staged_delta_hash=staged.staged_delta_hash,
            update_plan_hash=plan.update_plan_hash,
            mandate_hash=None,
            child_hashes=(),
            storage_evidence=_storage(),
            published_at=NOW,
        )
    assert store.current_head() == newer

    def fail_activation(_marker) -> None:
        raise RuntimeError("fixture activation failure")

    with pytest.raises(RuntimeError, match="fixture activation failure"):
        store.advance_head(
            expected_base_head_hash=newer.head_hash,
            next_head=_head(marker="callback-failure"),
            candidate_snapshot_hash=candidate.candidate_snapshot_hash,
            delta_hash=delta.delta_hash,
            staged_delta_hash=staged.staged_delta_hash,
            update_plan_hash=plan.update_plan_hash,
            mandate_hash=None,
            child_hashes=(),
            storage_evidence=_storage(),
            published_at=NOW,
            before_pointer_commit=fail_activation,
        )
    assert store.current_head() == newer

    later_source = _candidate(newer, source_revision=canonical_hash("later-source"))
    assert later_source.candidate_snapshot_hash != candidate.candidate_snapshot_hash
    assert candidate.candidate_snapshot_hash == delta.candidate_snapshot_hash


def test_confirmed_pending_transition_is_not_replaced_by_later_source(tmp_path) -> None:
    store = PreResearchHeadStore(artifact_root=tmp_path, mutation_gate=WorkspaceMutationGate())
    head = _head()
    candidate = _candidate(head, listings=("a", "b", "c"))
    delta = compute_observed_delta(head=head, candidate=candidate)
    plan = compile_update_plan(delta=delta, policy_hashes=(canonical_hash("policy"),))
    store.publish_candidate(candidate)
    store.publish_delta(delta)
    store.publish_plan(plan)
    first = store.set_pending_transition(
        candidate=candidate,
        delta=delta,
        plan=plan,
        frozen_at=NOW,
    )
    assert store.clear_unconfirmed_pending_transition(base_head_hash="wrong-base") is False
    assert store.clear_unconfirmed_pending_transition(base_head_hash=head.head_hash) is True
    first = store.set_pending_transition(
        candidate=candidate,
        delta=delta,
        plan=plan,
        frozen_at=NOW,
    )
    mandate = confirm_update_plan(
        plan=plan,
        confirmed_at=NOW,
        confirmation_token="confirmed-frozen-transition",
    )
    store.publish_mandate(mandate)
    confirmed = store.set_pending_transition(
        candidate=candidate,
        delta=delta,
        plan=plan,
        mandate=mandate,
        frozen_at=NOW,
    )
    assert confirmed.pending_hash != first.pending_hash
    assert store.pending_transition() == confirmed
    assert store.clear_unconfirmed_pending_transition(base_head_hash=head.head_hash) is False

    later = _candidate(
        head,
        listings=("a", "b", "d"),
        source_revision=canonical_hash("later-source"),
    )
    later_delta = compute_observed_delta(head=head, candidate=later)
    later_plan = compile_update_plan(delta=later_delta, policy_hashes=(canonical_hash("policy"),))
    store.publish_candidate(later)
    store.publish_delta(later_delta)
    store.publish_plan(later_plan)
    assert (
        store.set_pending_transition(
            candidate=later,
            delta=later_delta,
            plan=later_plan,
            frozen_at=NOW,
        )
        == confirmed
    )
    service = object.__new__(PreResearchDeltaService)
    service.writer_lease = SimpleNamespace(held=True)
    service.heads = store
    service.resolve_verified_head = lambda: head
    frozen_candidate, frozen_delta, frozen_plan = service.assess(
        candidate_document={"content_hash": canonical_hash("newer-external-source")},
        target_market_session=date(2026, 8, 3),
        observed_at=NOW,
    )
    assert frozen_candidate == candidate
    assert frozen_delta == delta
    assert frozen_plan == plan
    store.clear_pending_transition(expected_pending_hash=confirmed.pending_hash)
    assert store.pending_transition() is None


def test_disposable_membership_revision_advances_complete_frozen_tree_and_replays_noop(
    tmp_path,
) -> None:
    store = PreResearchHeadStore(artifact_root=tmp_path, mutation_gate=WorkspaceMutationGate())
    base = _head(marker="base")
    base_candidate = _candidate(base)
    base_delta = compute_observed_delta(head=base, candidate=base_candidate)
    base_plan = compile_update_plan(delta=base_delta, policy_hashes=(canonical_hash("policy"),))
    base_staged = stage_qualified_delta(
        delta=base_delta,
        ordered_qualified_listing_ids=base.ordered_listing_ids,
        quarantines=(),
    )
    store.publish_candidate(base_candidate)
    store.publish_delta(base_delta)
    store.publish_plan(base_plan)
    store.publish_staged_delta(base_staged)
    store.advance_head(
        expected_base_head_hash=None,
        next_head=base,
        candidate_snapshot_hash=base_candidate.candidate_snapshot_hash,
        delta_hash=base_delta.delta_hash,
        staged_delta_hash=base_staged.staged_delta_hash,
        update_plan_hash=base_plan.update_plan_hash,
        mandate_hash=None,
        child_hashes=(base.research_foundation_hash,),
        storage_evidence=_storage(),
        published_at=NOW,
    )

    observations = (
        PreResearchListingObservation(
            listing_id="b",
            source_member=True,
            local_history_start=base.research_history_start,
            local_history_end=base.target_market_session,
        ),
        PreResearchListingObservation(
            listing_id="c",
            source_member=True,
            local_history_start=base.research_history_start,
            local_history_end=base.target_market_session,
        ),
    )
    candidate = _candidate(
        base,
        listings=("b", "c"),
        observations=observations,
        source_revision=canonical_hash("membership-revision"),
    )
    delta = compute_observed_delta(head=base, candidate=candidate)
    plan = compile_update_plan(delta=delta, policy_hashes=(canonical_hash("policy"),))
    mandate = confirm_update_plan(
        plan=plan,
        confirmed_at=NOW,
        confirmation_token="confirmed-complete-membership-revision",
    )
    child_hashes = tuple(canonical_hash(("revision-child", index)) for index in range(7))
    staged = stage_qualified_delta(
        delta=delta,
        ordered_qualified_listing_ids=("b", "c"),
        quarantines=(),
        staged_child_hashes=child_hashes,
    )
    next_head = _head(listings=("b", "c"), marker="membership-revision")
    next_head = build_verified_head(
        **{
            **next_head.model_dump(mode="python", exclude={"head_hash", "verified_at"}),
            "source_revision": candidate.source_revision,
            "membership_fingerprint": candidate.source_membership_fingerprint,
            "ordered_source_candidate_listing_ids": candidate.ordered_candidate_listing_ids,
            "verified_at": NOW,
        }
    )
    for value, publish in (
        (candidate, store.publish_candidate),
        (delta, store.publish_delta),
        (plan, store.publish_plan),
        (staged, store.publish_staged_delta),
        (mandate, store.publish_mandate),
    ):
        publish(value)
    marker = store.advance_head(
        expected_base_head_hash=base.head_hash,
        next_head=next_head,
        candidate_snapshot_hash=candidate.candidate_snapshot_hash,
        delta_hash=delta.delta_hash,
        staged_delta_hash=staged.staged_delta_hash,
        update_plan_hash=plan.update_plan_hash,
        mandate_hash=mandate.mandate_hash,
        child_hashes=child_hashes,
        storage_evidence=_storage(),
        published_at=NOW,
    )
    assert marker.base_head_hash == base.head_hash
    assert store.current_head() == next_head
    assert store.load_candidate(base_candidate.candidate_snapshot_hash) == base_candidate

    replay_candidate = _candidate(
        next_head,
        observations=observations,
        source_revision=candidate.source_revision,
    )
    replay_delta = compute_observed_delta(head=next_head, candidate=replay_candidate)
    replay_plan = compile_update_plan(delta=replay_delta, policy_hashes=(canonical_hash("policy"),))
    assert replay_delta.revision_disposition is PreResearchRevisionDisposition.NOOP
    assert replay_plan.operations == ()


def test_post_hydration_active_set_fact_can_downgrade_fixed_screening() -> None:
    head = _head()
    candidate = _candidate(head, listings=("a", "b", "short-history"))
    delta = compute_observed_delta(head=head, candidate=candidate)
    plan = compile_update_plan(delta=delta, policy_hashes=(canonical_hash("policy"),))
    pending = SimpleNamespace(
        base_head_hash=head.head_hash,
        candidate_snapshot_hash=candidate.candidate_snapshot_hash,
    )

    class Heads:
        @staticmethod
        def pending_transition():
            return pending

        @staticmethod
        def load_candidate(_content_hash):
            return candidate

    class Store:
        qualified = ("a", "b")

        readiness = SimpleNamespace(
            load=lambda _market_profile_id: SimpleNamespace(active_manifest_id="active")
        )

        @classmethod
        def load_universe_manifest(cls, _manifest_id):
            return SimpleNamespace(
                listings=tuple(SimpleNamespace(listing_id=value) for value in cls.qualified)
            )

    service = object.__new__(PreResearchDeltaService)
    service.heads = Heads()
    service.market_data = Store()
    service.resolve_verified_head = lambda: head
    assert plan.require_fixed_factor_screening
    assert service.pending_qualified_active_set_changed() is False
    Store.qualified = ("a", "b", "eligible")
    assert service.pending_qualified_active_set_changed() is True


def test_restart_repairs_lifecycle_after_head_advanced_before_fulfillment(tmp_path) -> None:
    heads = PreResearchHeadStore(artifact_root=tmp_path, mutation_gate=WorkspaceMutationGate())
    base = _head(marker="crash-base")
    base_candidate = _candidate(base, policy_hashes=PRE_RESEARCH_DELTA_POLICY_HASHES)
    base_delta = compute_observed_delta(head=base, candidate=base_candidate)
    base_plan = compile_update_plan(
        delta=base_delta, policy_hashes=PRE_RESEARCH_DELTA_POLICY_HASHES
    )
    base_staged = stage_qualified_delta(
        delta=base_delta,
        ordered_qualified_listing_ids=base.ordered_listing_ids,
        quarantines=(),
    )
    for value, publish in (
        (base_candidate, heads.publish_candidate),
        (base_delta, heads.publish_delta),
        (base_plan, heads.publish_plan),
        (base_staged, heads.publish_staged_delta),
    ):
        publish(value)
    heads.advance_head(
        expected_base_head_hash=None,
        next_head=base,
        candidate_snapshot_hash=base_candidate.candidate_snapshot_hash,
        delta_hash=base_delta.delta_hash,
        staged_delta_hash=base_staged.staged_delta_hash,
        update_plan_hash=base_plan.update_plan_hash,
        mandate_hash=None,
        child_hashes=(base.research_foundation_hash,),
        storage_evidence=_storage(),
        published_at=NOW,
    )

    candidate = _candidate(
        base,
        listings=("a", "b", "c"),
        source_revision=canonical_hash("crash-source"),
        policy_hashes=PRE_RESEARCH_DELTA_POLICY_HASHES,
    )
    delta = compute_observed_delta(head=base, candidate=candidate)
    plan = compile_update_plan(delta=delta, policy_hashes=PRE_RESEARCH_DELTA_POLICY_HASHES)
    mandate = confirm_update_plan(
        plan=plan,
        confirmed_at=NOW,
        confirmation_token="confirmed-crash-window-revision",
    )
    next_head = _head(listings=("a", "b", "c"), marker="crash-next")
    next_head = build_verified_head(
        **{
            **next_head.model_dump(mode="python", exclude={"head_hash", "verified_at"}),
            "source_revision": candidate.source_revision,
            "membership_fingerprint": candidate.source_membership_fingerprint,
            "ordered_source_candidate_listing_ids": candidate.ordered_candidate_listing_ids,
            "verified_at": NOW,
        }
    )
    child_hashes = (
        next_head.panel_snapshot_hash,
        next_head.factor_screening_result_hash,
        next_head.factor_candidate_slate_hash,
        next_head.research_desk_factor_input_hash,
        next_head.causal_execution_outcome_snapshot_hash,
        next_head.research_foundation_hash,
        next_head.research_foundation_marker_hash,
    )
    staged = stage_qualified_delta(
        delta=delta,
        ordered_qualified_listing_ids=next_head.ordered_listing_ids,
        quarantines=(),
        staged_child_hashes=child_hashes,
    )
    for value, publish in (
        (candidate, heads.publish_candidate),
        (delta, heads.publish_delta),
        (plan, heads.publish_plan),
        (staged, heads.publish_staged_delta),
        (mandate, heads.publish_mandate),
    ):
        publish(value)
    pending = heads.set_pending_transition(
        candidate=candidate,
        delta=delta,
        plan=plan,
        mandate=mandate,
        frozen_at=NOW,
    )

    class CrashOnceStore:
        def __init__(self) -> None:
            self.fulfill_calls = 0
            self.block_calls = 0
            self.requirement = SimpleNamespace(
                transition_id="transition-1",
                lifecycle="REQUIRED",
                panel_snapshot_hash=None,
                factor_result_hash=None,
                factor_slate_hash=None,
                execution_outcome_hash=None,
                foundation_hash=None,
                revision_marker_hash=None,
            )

        def start_feature_universe_rebuild(self, _transition_id, *, observed_at):
            del observed_at
            self.requirement.lifecycle = "RUNNING"
            return self.requirement

        def fulfill_feature_universe_rebuild(self, _transition_id, **values):
            self.fulfill_calls += 1
            if self.fulfill_calls == 1:
                raise RuntimeError("crash after HEAD activation")
            self.requirement.lifecycle = "FULFILLED"
            for name, value in values.items():
                if name != "observed_at":
                    setattr(self.requirement, name, value)
            return self.requirement

        def block_feature_universe_rebuild(self, *_args, **_kwargs):
            self.block_calls += 1

        def feature_universe_rebuild_requirement(self, _transition_id):
            return self.requirement

        def feature_universe_rebuild_for_manifest(self, _manifest_revision):
            return self.requirement

    store = CrashOnceStore()
    service = object.__new__(PreResearchDeltaService)
    service.writer_lease = SimpleNamespace(held=True)
    service.heads = heads
    service.market_data = store
    service.foundation_state = store
    with pytest.raises(RuntimeError, match="crash after HEAD activation"):
        service.activate_staged_revision(
            candidate=candidate,
            delta=delta,
            plan=plan,
            staged=staged,
            next_head=next_head,
            storage_evidence=_storage(),
            published_at=NOW,
            mandate=mandate,
            transition_id="transition-1",
        )
    assert heads.current_head() == next_head
    assert heads.pending_transition() == pending
    assert store.block_calls == 0

    marker = service.finalize_pending_transition(published_at=NOW)
    assert marker is not None
    assert marker.next_head_hash == next_head.head_hash
    assert store.requirement.lifecycle == "FULFILLED"
    assert heads.pending_transition() is None


def test_tampered_marker_child_fails_closed(tmp_path) -> None:
    store = PreResearchHeadStore(artifact_root=tmp_path, mutation_gate=WorkspaceMutationGate())
    head = _head()
    candidate = _candidate(head)
    delta = compute_observed_delta(head=head, candidate=candidate)
    plan = compile_update_plan(delta=delta, policy_hashes=(canonical_hash("policy"),))
    staged = stage_qualified_delta(
        delta=delta,
        ordered_qualified_listing_ids=head.ordered_listing_ids,
        quarantines=(),
    )
    store.publish_candidate(candidate)
    store.publish_delta(delta)
    store.publish_plan(plan)
    store.publish_staged_delta(staged)
    marker = store.advance_head(
        expected_base_head_hash=None,
        next_head=head,
        candidate_snapshot_hash=candidate.candidate_snapshot_hash,
        delta_hash=delta.delta_hash,
        staged_delta_hash=staged.staged_delta_hash,
        update_plan_hash=plan.update_plan_hash,
        mandate_hash=None,
        child_hashes=(),
        storage_evidence=_storage(),
        published_at=NOW,
    )
    marker_path = tmp_path / "pre-research-revisions" / "markers" / f"{marker.marker_hash}.json"
    marker_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        store.current_head()
