"""The source identity of the one installed fixed-Panel score producer, as the Host binds it."""

from __future__ import annotations

from typing import cast

from alphalattice.investment.alpha_research.scores.temporal_aggregation import (
    AlphaPanelSourceIdentity,
)


def fixed_panel_alpha_source_identity() -> AlphaPanelSourceIdentity:
    """Return the immutable source identity consumed by the sealed score graph."""
    return cast(
        AlphaPanelSourceIdentity,
        AlphaPanelSourceIdentity.model_validate(
            {
                "feature_catalog_hash": (
                    "9a69019d9742f2c3c3629992ba8f9c897cf657d4d66bb622941f32e26040fc31"
                ),
                "formula_observation_policy_hash": (
                    "f954f0c4672880c36e1d5f99b8a6ef5bd6f462110b3534b92a40c68ca06a97f0"
                ),
                "governing_availability_delay_sessions": 0,
                "governing_availability_phase": "OFFICIAL_CLOSE",
                "governing_availability_policy_hash": (
                    "7f969d3efb32d5d3aaf381b66d1eb74968a16198e5e0cc478cb76e0437f139a9"
                ),
                "governing_availability_policy_id": (
                    "feature-availability.daily-provider-session-close"
                ),
                "identity_hash": (
                    "387001a9ee7b78a7b14a75ce487cb8e2032d7c8e2c4ccc602b8b79e50261a46e"
                ),
                "ordered_factor_axis_hash": (
                    "90e032fec9de4e23930569f292a8bb6fa076357de3ae4ff22b1b1213d5a302ce"
                ),
                "ordered_listing_axis_hash": (
                    "dda7c9d7f38dd1d5199dd51d8ac0d2c5e05d240ee979932f4d4feae1b5760e16"
                ),
                "ordered_session_axis_hash": (
                    "2026939e885f6fc3ede551f88135451d31a23b579631a9c5432c0a4a9544edd2"
                ),
                "panel_snapshot_hash": (
                    "5c902b329c34dfd09934906ead549afab95c1df4d8248e3647fdebcd19787f65"
                ),
                "source_authority_binding_hash": (
                    "d0704cd0f5832db51daeb1880468f3d51097ec4fc074ca4a014bcc3a240f98bc"
                ),
                "source_availability_catalog_hash": (
                    "c837502d85fbe82585929fc9e6d9c5172d740c532cce4cacbc710b56d0db5d60"
                ),
                "source_resolution_hash": (
                    "bb06155d2282cd5992988d134c2d108005f9b7f23e9410f1a1af9ac537a89392"
                ),
            }
        ),
    )


__all__ = ["fixed_panel_alpha_source_identity"]
