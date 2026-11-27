"""Exploration, completed: samples keep each Sector's share, a Factor study
reads a sample without moving a whole-Panel identity, a study sampled before that reads back as
not current, and a feature trial's Alpha comparison is the Alpha owner's.

For the UI line's suite: tests/integration/product_host/test_exploration_by_sector.py.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

import pytest

from alphalattice.control.product_host.composition.research_experiment_projection import (
    SAMPLE_SCHEME_SUPERSEDED,
    sample_standing,
)
from alphalattice.control.product_host.research_authoring.authority import exploration_sample
from alphalattice.foundation.factor_research.evaluation.oos_evidence import (
    build_factor_evidence_policy,
)
from alphalattice.foundation.factor_research.evaluation.redundancy import (
    build_factor_redundancy_policy,
)
from alphalattice.foundation.factor_research.inputs.execution_target import (
    build_factor_target_policy,
)
from alphalattice.foundation.factor_research.programs.program import (
    FactorResearchProgramSpec,
    build_factor_research_program_spec,
)
from alphalattice.foundation.factor_research.programs.walk_forward import (
    build_factor_walk_forward_policy,
)
from alphalattice.investment.alpha_research.experiments.comparison import _relation
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResolvedResearchAuthority,
)
from tests._support.identity_fixtures import fixture_digest

_LISTINGS = tuple(f"L{i:03d}" for i in range(300))
_SECTORS = {
    listing: ("Energy", "Health", "Tech", "Utilities")[
        0 if i < 30 else 1 if i < 120 else 2 if i < 297 else 3
    ]
    for i, listing in enumerate(_LISTINGS)
}


def test_a_sample_takes_each_sectors_share_and_the_same_names_each_time():
    sample = exploration_sample(_LISTINGS, 110, _SECTORS)

    assert len(sample) == 110
    assert sample == exploration_sample(_LISTINGS, 110, _SECTORS)
    assert sample == tuple(v for v in _LISTINGS if v in set(sample))
    counts = Counter(_SECTORS[v] for v in sample)
    # The shares of 110 are 11.0, 33.0, 64.9 and 1.1; one name at a time by Sainte-Lague.
    assert counts == {"Energy": 11, "Health": 33, "Tech": 65, "Utilities": 1}
    first = exploration_sample(_LISTINGS, 4, _SECTORS)
    assert {_SECTORS[v] for v in first} == set(_SECTORS.values())
    assert set(first) <= set(sample)
    for smaller, larger in ((100, 101), (101, 150), (150, 299)):
        assert set(exploration_sample(_LISTINGS, smaller, _SECTORS)) < set(
            exploration_sample(_LISTINGS, larger, _SECTORS)
        )


def _spec(**extra: object) -> FactorResearchProgramSpec:
    return build_factor_research_program_spec(
        feature_panel_snapshot_hash=fixture_digest("a"),
        feature_panel_manifest_ref="artifacts/panel.json",
        causal_outcome_snapshot_hash=fixture_digest("b"),
        causal_outcome_manifest_ref="artifacts/outcome.json",
        factor_ids=("f1", "f2"),
        frozen_at=datetime(2026, 7, 31, tzinfo=UTC),
        target_policy=build_factor_target_policy(),
        walk_forward_policy=build_factor_walk_forward_policy(),
        evidence_policy=build_factor_evidence_policy(),
        redundancy_policy=build_factor_redundancy_policy(),
        **extra,
    )


def test_a_whole_panel_factor_program_keeps_its_identity_and_a_sample_enters_it():
    whole = _spec()
    dumped = whole.model_dump(mode="json")

    assert "listing_ids" not in dumped
    assert FactorResearchProgramSpec.model_validate_json(whole.model_dump_json()) == whole
    sampled = _spec(listing_ids=("L001", "L002"))
    assert sampled.program_hash != whole.program_hash
    assert sampled.model_dump(mode="json")["listing_ids"] == ["L001", "L002"]
    with pytest.raises(ValueError):
        _spec(listing_ids=("L001", "L001"))


def _authority(**extra: object) -> ResolvedResearchAuthority:
    return ResolvedResearchAuthority.create(
        data_snapshot_handle=fixture_digest("a"),
        universe_handle="us-current-index-research.sample-110",
        panel_snapshot_hash=fixture_digest("a"),
        panel_manifest_ref="artifacts/panel.json",
        universe_revision_sha256=fixture_digest("c"),
        ordered_listing_ids=("L001", "L002"),
        sessions=(datetime(2026, 7, 31).date(),),
        source_watermark_hash=fixture_digest("d"),
        **extra,
    )


def test_a_study_sampled_before_sector_shares_reads_back_not_current():
    explored = {"experiment": {"universe_handle": "us-current-index-research.sample-110"}}
    whole = {"experiment": {"universe_handle": "us-current-index-research"}}
    earlier, current = _authority(), _authority(listing_sample="SECTOR_SHARES")

    assert "listing_sample" not in earlier.model_dump(mode="json")
    assert current.authority_hash != earlier.authority_hash
    assert sample_standing(explored, earlier) == {
        "method_standing": "NOT_CURRENT",
        "method_refusal": SAMPLE_SCHEME_SUPERSEDED,
    }
    assert sample_standing(explored, current) == {}
    assert sample_standing(whole, earlier) == {}


def _subject(features: list[str], values: str, materialization: str) -> dict[str, object]:
    return {
        "input_binding_hash": fixture_digest("i"),
        "target_recipe_binding_hash": fixture_digest("r"),
        "target_materialization_binding_hash": materialization,
        "target_values": {"transformed_target_value_hash": values},
        "ordered_feature_ids": features,
        "metric_policy_hash": fixture_digest("m"),
        "split_policy_hash": fixture_digest("s"),
    }


def test_the_alpha_owner_compares_a_feature_addition_over_the_same_target_values():
    base = _subject(["a", "b"], "v", fixture_digest("x"))

    assert _relation(base, _subject(["a", "b"], "v", fixture_digest("x"))) == "PARAMETERS"
    assert (
        _relation(base, _subject(["a", "b", "c"], "v", fixture_digest("y"))) == "FEATURE_ADDITION"
    )
    assert (
        _relation(_subject(["a", "b", "c"], "v", fixture_digest("y")), base) == "FEATURE_ADDITION"
    )
    for other, code in (
        (_subject(["a", "b"], "v", fixture_digest("y")), "target_materialization_binding_hash"),
        (_subject(["a", "c"], "v", fixture_digest("y")), "ordered_feature_ids"),
        (_subject(["a", "b", "c"], "w", fixture_digest("y")), "target_values"),
    ):
        with pytest.raises(AuthoringError, match=f"saved_comparison_{code}_mismatch"):
            _relation(base, other)
