import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from alphalattice.foundation.factor_research.experiments.execution import (
    factor_policies_at_scale,
)
from alphalattice.foundation.market_data_ops.runtime.universe_onboarding import history_floor
from alphalattice.foundation.market_data_ops.sources.manifest import (
    build_current_index_acquisition_manifest,
    build_quality_filtered_research_manifest,
    default_qualification_policy_hash,
)
from alphalattice.foundation.market_data_ops.sources.universe import (
    ORIGINAL_RESEARCH_WHITELIST_STANDARD,
    CandidateMembershipEvidence,
    CurrentUniverseCandidate,
    CurrentUniverseCandidateManifest,
    bootstrap_from_candidate_manifest_document,
    candidate_manifest_document,
)

TEST_SOURCE = "src/alphalattice/foundation/market_data_ops/sources/universe.py"
PROFILE = """schema: alphalattice.playpen.market-profile
market_profile_id: us-current-index-research
display_name: US Current Index Research
market: US
currency: USD
calendar_id: XNYS_XNAS
provider: yfinance
daily_price_basis: split_adjusted
data_validity_class: CURRENT_UNIVERSE_RESEARCH_ONLY
"""
"""The shipped market profile's fields, as config/market-profiles holds them."""
AT = datetime(2026, 8, 2, tzinfo=UTC)


def _profile(root: Path) -> Path:
    (path := root / "us-current-index-research.yaml").write_text(PROFILE, encoding="utf-8")
    return path


def _document(years: int, listings: int = 100) -> dict[str, object]:
    """Two candidates' persisted document, its content hash sealed as a source writes it."""
    standard = ORIGINAL_RESEARCH_WHITELIST_STANDARD
    candidates = tuple(
        CurrentUniverseCandidate(
            symbol=symbol,
            provider_symbol=symbol,
            source_memberships=(CandidateMembershipEvidence("NASDAQ100", symbol),),
        )
        for symbol in ("AAA", "BBB")
    )
    manifest = CurrentUniverseCandidateManifest(
        AT,
        AT,
        standard.construction_rule,
        standard.data_validity_class,
        (),
        candidates,
        "",
        years,
        minimum_factor_listings=listings,
    )
    document = candidate_manifest_document(manifest)
    document.pop("content_hash")
    sealed = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return document | {"content_hash": hashlib.sha256(sealed.encode()).hexdigest()}


@pytest.mark.proves("market_data_ops.a_standard_names_its_history_floor")
def test_the_original_standard_keeps_every_identity_it_recorded(tmp_path: Path) -> None:
    """The original's ten-year floor, its document and its manifests' identities are the ones
    develop recorded before the floor became the standard's (base 9567e1f95)."""
    document = _document(10)
    assert "minimum_history_calendar_years" not in document
    assert document["content_hash"] == (
        "4a53139ab50646a4f76575549d49d64d65283d2add31dd933554992681d7f801"
    )
    bootstrap = bootstrap_from_candidate_manifest_document(document)
    assert bootstrap.standard == ORIGINAL_RESEARCH_WHITELIST_STANDARD
    acquisition = build_current_index_acquisition_manifest(_profile(tmp_path), bootstrap)
    assert acquisition.revision_sha256 == (
        "600f527a33ca40f946e2de61c05cfa9caad13b2e9853068530d33c0777139880"
    )
    assert acquisition.qualification_obligations == ()
    assert (
        acquisition.qualification_policy_hash
        == default_qualification_policy_hash()
        == ("44137559f37df492a0dea326ec5cbcf85e850813ba8295f8253a0ca14a2e493e")
    )
    assert history_floor(date(2026, 9, 10), bootstrap.standard) == date(2016, 9, 10)
    assert history_floor(date(2028, 2, 29)) == date(2018, 2, 28)
    assert "minimum_factor_listings" not in document
    assert (
        factor_policies_at_scale(
            minimum_listings=bootstrap.standard.minimum_factor_listings,
            history_years=bootstrap.standard.minimum_history_calendar_years,
        )
        == {}
    )  # the shipped Factor defaults


@pytest.mark.proves("market_data_ops.a_standard_names_its_history_floor")
def test_a_shorter_standard_is_persisted_applied_and_named_by_its_manifests(
    tmp_path: Path,
) -> None:
    """A two-year standard survives its document, floors history two years back and names its
    own floor in each manifest derived under it, never the original's. Its Factor floor of ten
    listings survives too and sets the Factor policies, the walk-forward fitted to two years."""
    bootstrap = bootstrap_from_candidate_manifest_document(_document(2, listings=10))
    assert bootstrap.standard == replace(
        ORIGINAL_RESEARCH_WHITELIST_STANDARD,
        minimum_history_calendar_years=2,
        minimum_factor_listings=10,
    )
    policies = factor_policies_at_scale(
        minimum_listings=bootstrap.standard.minimum_factor_listings,
        history_years=bootstrap.standard.minimum_history_calendar_years,
    )
    assert policies["evidence_policy"].minimum_cross_section_observations == 10
    assert policies["redundancy_policy"].minimum_common_listings == 10
    walk = policies["walk_forward_policy"]
    # The outcomes' sealed 252 stay; three complete folds fit the 238 development sessions left.
    assert (walk.train_sessions, walk.validation_sessions, walk.step_sessions) == (119, 39, 39)
    assert (walk.sealed_holdout_sessions, walk.minimum_folds) == (252, 3)
    assert history_floor(date(2026, 9, 10), bootstrap.standard) == date(2024, 9, 10)
    acquisition = build_current_index_acquisition_manifest(_profile(tmp_path), bootstrap)
    floor = f"history_quality:{default_qualification_policy_hash(bootstrap.standard)}"
    assert acquisition.qualification_obligations == (floor,)
    research = build_quality_filtered_research_manifest(
        acquisition,
        eligible_listing_ids=(acquisition.listings[0].listing_id,),
        qualification_obligations=acquisition.qualification_obligations,
    )
    assert research.obligations_for_derivation() == (floor,)
    original = build_current_index_acquisition_manifest(
        _profile(tmp_path), bootstrap_from_candidate_manifest_document(_document(10))
    )
    assert acquisition.revision_sha256 != original.revision_sha256


@pytest.mark.proves("market_data_ops.a_standard_names_its_history_floor")
@pytest.mark.parametrize(
    ("years", "change"),
    [
        pytest.param(2, {"minimum_history_calendar_years": 3}, id="floor-changed-after-sealing"),
        pytest.param(11, {}, id="floor-above-the-original"),
        pytest.param(2, {"minimum_history_calendar_years": "2"}, id="floor-not-whole-years"),
        pytest.param(2, {"minimum_factor_listings": 20}, id="factor-floor-changed-after-sealing"),
        pytest.param(2, {"minimum_factor_listings": "10"}, id="factor-floor-not-whole-listings"),
        pytest.param(2, {"minimum_factor_listings": 101}, id="factor-floor-above-the-original"),
        pytest.param(2, {"minimum_factor_listings": 1}, id="factor-floor-below-a-pair"),
    ],
)
def test_a_persisted_floor_is_refused_unless_sealed_and_within_the_original(
    years: int, change: dict[str, object]
) -> None:
    """A document's floor is trusted only as sealed, whole and no longer than ten years."""
    with pytest.raises(ValueError, match="persisted candidate manifest"):
        bootstrap_from_candidate_manifest_document(_document(years) | change)
