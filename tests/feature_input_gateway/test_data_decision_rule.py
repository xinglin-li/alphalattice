"""The Host's declared data-decision rule over the gateway's own cases (OP19)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from alphalattice.control.data_platform.data_decision_rule import (
    ESCALATE,
    QUARANTINE,
    RETAIN,
    DataDecisionRule,
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
        replace(item, unexplained_sessions=moved) if item.failure_code else item
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
