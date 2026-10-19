"""The Host's declared data-decision rule over the gateway's own cases (OP19)."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date

import pytest

from alphalattice.control.data_platform.data_decision_rule import (
    ESCALATE,
    FULL_HISTORY,
    QUARANTINE,
    RETAIN,
    DataDecisionRule,
    attach,
    audit,
    held_decisions,
    kept_listings,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
)
from alphalattice.foundation.feature_engine.inputs.gateway import FeatureInputGateway
from alphalattice.foundation.market_data_ops.runtime.remediation import canonical_hash
from alphalattice.protocols.actor_execution import ActorKind
from tests.feature_input_gateway.gateway_support import (
    NOW,
    _evidence,
    _manifest,
    _panel_impact,
    _sectors,
    _temporal,
)


@pytest.mark.parametrize(
    ("failed", "code", "sessions", "chosen"),
    [
        ({0}, "data.unexplained_raw_move", 1, RETAIN),
        ({0}, "data.unexplained_raw_move", 2, QUARANTINE),
        ({0}, "data.provider_timeout", 0, QUARANTINE),
        ({0, 1, 2}, "data.unexplained_raw_move", 1, ESCALATE),
    ],
)
def test_the_rule_retains_an_isolated_move_quarantines_the_rest_and_escalates_past_its_cap(
    failed: set[int], code: str, sessions: int, chosen: str
) -> None:
    """requirement: one verified move on one session is kept with its caveat, any other case
    set aside, and more than 2% of the names at once left to the agent, each by evidence only."""
    manifest = _manifest(100)
    moved = tuple(date(2026, 6, day) for day in range(1, sessions + 1))
    evidence = tuple(
        replace(item, unexplained_sessions=moved, anomaly_signature_hash="a" * 64)
        if item.failure_code
        else item
        for item in _evidence(
            manifest, failed=failed, failure_code=code, extreme=code.endswith("raw_move")
        )
    )
    sectors = _sectors(manifest)
    cases = (
        FeatureInputGateway()
        .assess(
            candidate_manifest=manifest,
            evidence=evidence,
            sector_by_listing_id=sectors,
            temporal_boundary=_temporal(),
            observed_at=NOW,
            panel_impact=_panel_impact(manifest, sectors),
        )
        .agent_cases
    )
    rule = DataDecisionRule()
    capped = rule.capped(len(cases), len(manifest.listings), affected_listings=len(failed))
    assert [ESCALATE if capped else rule.choose(case, evidence) for case in cases] == [
        chosen
    ] * len(failed)
    # Audits keep their own bound: two approved earlier never push these cases past theirs.
    audited = {
        "operation": FULL_HISTORY,
        "decided_by": "RULE",
        "status": "DECIDED_BY_RULE",
        "rule_hash": rule.rule_hash,
        "listing_ids": ["A1", "A2"],
    }
    assert rule.scope(cases, len(manifest.listings), [audited])["capped"] == capped
    assert not rule.audits_capped(2, 100) and rule.audits_capped(3, 100)
    # Re-raised, each case this rule already decided on the same anomaly holds that decision
    # and counts toward no bound, however many there are.
    decided = {
        (item.listing_id, item.anomaly_signature_hash): rule.choose(case, evidence)
        for case in cases
        for item in case.evidence
        if item.listing_id in case.listing_ids
    }
    held = rule.scope(cases, len(manifest.listings), (), decided)
    assert held["case_count"] == 0 and not held["capped"]
    assert [rule.ruling(case, evidence, held, decided) for case in cases] == [
        (rule.choose(case, evidence), "HELD_BY_RULE") for case in cases
    ]
    if chosen != RETAIN:
        return
    # A provider correction since: the rule would quarantine now, so the held retain gives way
    # to a new decision that counts toward the bound and reads as changed.
    case = cases[0]
    corrected = replace(
        case,
        evidence=tuple(replace(item, provider_correction_observed=True) for item in case.evidence),
    )
    fresh = rule.scope((corrected,), len(manifest.listings), (), decided)
    assert fresh["case_count"] == 1
    assert rule.ruling(corrected, corrected.evidence, fresh, decided) == (
        QUARANTINE,
        "CHANGED_BY_RULE",
    )
    # The newest decision on an anomaly wins: the person's later choice ends the rule's hold,
    # and a quarantine since released holds nothing.
    document, key = json.dumps(case.document()), (case.listing_ids[0], "a" * 64)

    def receipt(option: str, kind: str, actor: str) -> dict:
        proposal = {
            "option_id": option,
            "rationale": f"workspace-rule-goal:g; rule:{rule.rule_hash}",
        }
        return {
            "receipt": {
                "actor_submission": {"actor_kind": kind, "actor_id": actor},
                "submission": {"proposal": proposal},
            },
            "effect": {"status": "APPLIED"},
        }

    ruled = receipt(RETAIN, "HOST_FALLBACK", f"host-rule:{rule.rule_hash[:16]}")
    person = receipt(QUARANTINE, "HUMAN", "person")
    assert held_decisions([(document, ruled)], rule.rule_hash) == {key: RETAIN}
    assert held_decisions([(document, ruled), (document, person)], rule.rule_hash) == {}
    set_aside = receipt(QUARANTINE, "HOST_FALLBACK", f"host-rule:{rule.rule_hash[:16]}")
    assert held_decisions([(document, set_aside)], rule.rule_hash) == {}
    assert held_decisions([(document, set_aside)], rule.rule_hash, {key[0]}) == {key: QUARANTINE}
    signed = rule.submission(cases[0], chosen, "goal-1")
    assert signed.actor_kind is ActorKind.HOST_FALLBACK
    assert signed.actor_id == f"host-rule:{rule.rule_hash[:16]}"
    assert signed.submission.proposal.rationale == f"first-use-goal:goal-1; rule:{rule.rule_hash}"
    # The cap is the rule's identity: a changed number is another rule, with its own hash.
    assert DataDecisionRule(max_cases=3).rule_hash != rule.rule_hash


def test_the_case_count_and_listing_share_have_independent_bounds() -> None:
    """Grouped cases count once while their distinct listings decide the population share."""
    rule = DataDecisionRule()
    assert not rule.capped(10, 1000, affected_listings=20)
    assert rule.capped(11, 1000, affected_listings=1)
    assert rule.capped(1, 100, affected_listings=3)
    assert not rule.capped(1, 100, affected_listings=2)


def test_a_rule_declared_before_the_precision_keeps_its_identity_decisions_and_stops() -> None:
    """requirement: upgrading keeps a rule's identity, so its manifest loads, its held
    decisions stay held and an escalated audit stop still stands; only a precision other than
    the shipped one is a new rule."""
    # The rule as the release before the precision recorded it: its fields, hashed.
    declared = {"max_cases": 3, "max_population_share": 0.02, "version": "data-decision-rule/1"}
    rule = DataDecisionRule(max_cases=3)
    assert rule.rule_hash == canonical_hash(declared)
    assert DataDecisionRule(max_cases=3, adjusted_precision=1e-5).rule_hash != rule.rule_hash
    manifest = ResearchWorkspaceManifest.create(
        workspace_id="w",
        default_strategy_package_id=None,
        default_score_source_mode=None,
        strategy_artifacts=(),
        data_decision_rule=rule,
        strategy_installation="NOT_INSTALLED",
    )
    stored = manifest.model_dump(mode="json")
    assert stored["data_decision_rule"] == declared
    assert ResearchWorkspaceManifest.model_validate(stored) == manifest
    # Its escalated audit stop, recorded under that hash, still stands: nothing is approved.
    stop = {
        "task_id": "t",
        "operation": FULL_HISTORY,
        "decided_by": "RULE",
        "status": "ESCALATED",
        "rule_hash": canonical_hash(declared),
        "listing_ids": ["A1", "A2", "A3"],
    }
    written: list[dict] = []
    assert (
        audit(rule, lambda: object(), lambda: [stop], written.append, "t", {"A4": "D"}, 100, str)
        is None
    )
    assert written == []


def test_the_rule_keeps_stored_closes_only_while_its_goal_holds_and_records_each_keep() -> None:
    """requirement: the precision applies only under the rule's live authority, and every
    listing it kept is recorded on the goal that granted it; without that authority, none."""
    goal = type("Goal", (), {"goal_hash": "g" * 64})()
    holding: list[object] = [goal]
    recorded: list[dict] = []

    def decide(*_: object) -> None:
        return None

    attach(decide, DataDecisionRule(), lambda: holding[0], tuple, recorded.append, "t")
    assert decide.precision() == 1e-6  # type: ignore[attr-defined]
    listing = type("Listing", (), {"listing_id": "L1", "symbol": "AAA"})()
    holding[0] = None  # the goal closes before the cycle records its change set
    parts = [["2026-07-27", "2026-07-29", 0.99], ["2026-07-30", "2026-07-31", 1.0]]
    split = {"ex_date": "2026-07-30", "first_session": "2026-07-27"}
    split |= {"last_session": "2026-07-29", "price_factor": 0.5, "volume_factor": 2.0}
    split |= {"receipt_hash": "h", "largest_difference": 0.0}
    changes = [{"listing_id": "L1", "adjusted_precision_kept": 5e-7, "adjustment_segments": parts}]
    decide.kept([changes[0] | {"raw_rebases": [split]}], [listing])  # type: ignore[attr-defined]
    assert [(r["listing_ids"], r["goal_hash"]) for r in recorded] == [(["L1"], "g" * 64)]
    entry = json.loads(json.dumps(recorded[0])) | {"rule_hash": "r"}  # as the Goal keeps it
    assert kept_listings([entry]) == {
        "L1": (
            "r",
            5e-7,
            tuple(map(tuple, parts)),
            (("2026-07-30", "2026-07-27", "2026-07-29", 0.5, 2.0, "h"),),
        )
    }
    plain = entry | {"raw_rebases": [[]], "largest_relative_differences": [4e-7]}  # a later keep
    assert kept_listings([entry, plain])["L1"][3] == kept_listings([entry])["L1"][3]
    assert decide.precision() == 0.0  # type: ignore[attr-defined]
    from alphalattice.control.data_platform.data_decision_rule import kept_observations

    rows = (type("Rebase", (), {"listing_id": "L1", "receipt_hash": "h", "kept_within": 3})(),)
    rows[0].kept_largest = 4e-7  # type: ignore[attr-defined]
    snapshot = type(
        "Snapshot", (), {"rebases": rows, "kept_blocks": (0, 2), "kept_agreement": None}
    )
    snapshot.snapshot_hash = "5" * 64  # type: ignore[attr-defined]
    decide.retained(snapshot)  # type: ignore[attr-defined]  # the goal closed: none recorded
    holding[0] = goal
    decide.retained(snapshot)  # type: ignore[attr-defined]
    assert kept_observations(json.loads(json.dumps(item)) for item in recorded) == (
        ("5" * 64, (("L1", "h", 3, 4e-7),), (0, 2), (1e-6, 1e-12)),
    )
