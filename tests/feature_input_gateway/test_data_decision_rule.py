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
    held_decisions,
)
from alphalattice.foundation.feature_engine.inputs.gateway import FeatureInputGateway
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
