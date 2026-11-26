"""Lagged Sector leader propagation with explicit membership-as-of input."""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil
from typing import Final

import numpy as np
import pandas as pd

from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    NO_ECONOMIC_SKIP,
)
from alphalattice.kernel.quant.factor_contracts import FactorFamily, FactorSpec, FactorTrack

SECTOR_LEADER_LAG_METHOD_FAMILY: Final = "SECTOR_LEADER_LAG"
SECTOR_LEADER_LAG_5_ID: Final = "factor.desktop.experimental.sector_leader_lag_5.v1"
SECTOR_LEADER_LAG_REQUIRED_FIELDS: Final = (
    "close_raw",
    "provider_adjusted_close",
    "sector_membership_asof",
    "volume_raw",
)
SECTOR_LEADER_LAG_DECLARATION: Final[Mapping[str, str]] = {
    "implementation_id": SECTOR_LEADER_LAG_5_ID,
    "leader_formation": "ell=t-5",
    "adv21_interval": "[t-25,t-5] inclusive, mean(close_raw*volume_raw)",
    "membership": "SectorMembershipAsOf(ell)",
    "leader_count": "max(2,ceil(0.20*N)), sector N>=10",
    "leader_tie_break": "ADV descending, listing_id ascending",
    "return_interval": "ln(A[t]/A[t-5]) = (t-5,t]",
    "output": "leave-one-out leader mean minus own return",
}


def leader_lag_factor_specs() -> tuple[FactorSpec, ...]:
    """Describe the as-of sector-leader lag recipe and its formation history.

    Returns:
        The leader-minus-own five-session return specification. Recipes retain their declared source
        fields, economic skips,
        minimum ordered observations, tolerances, and implementation references.
    """
    return (
        FactorSpec(
            factor_id="sector_leader_lag_5",
            family=FactorFamily.MOMENTUM,
            formula_ref=SECTOR_LEADER_LAG_5_ID,
            formula=(
                "LOO mean leader ln(A[t]/A[t-5])-own return; leaders from ADV21 "
                "[t-25,t-5] and SectorMembershipAsOf(t-5)"
            ),
            window_sessions=25,
            # The five-session leader return is the Formula's economics and it
            # ends at the observation session. The leader-formation offset ell is
            # part of the leader/laggard construction, not a session of safety.
            lag_sessions=NO_ECONOMIC_SKIP,
            return_convention="raw_dollar_volume_and_provider_adjusted_log_return",
            required_fields=SECTOR_LEADER_LAG_REQUIRED_FIELDS,
            literature_sources=("https://doi.org/10.1093/rfs/hhn108",),
            minimum_observations=26,
            absolute_tolerance=1e-10,
            relative_tolerance=1e-10,
            track=FactorTrack.MODEL,
            core_anchor=False,
        ),
    )


def sector_leader_lag_5(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Calculate the cross-sectional method when historical membership is supplied."""
    frame = source.assign(_position=np.arange(len(source), dtype=int)).copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    frame = frame.sort_values(["session_date", "listing_id"], kind="mergesort")
    sessions = tuple(sorted(frame["session_date"].unique()))
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for session_position in range(25, len(sessions)):
        t = sessions[session_position]
        leader_formation = sessions[session_position - 5]
        adv_sessions = set(sessions[session_position - 25 : session_position - 4])
        current = frame.loc[frame["session_date"] == t]
        formation = frame.loc[frame["session_date"] == leader_formation]
        memberships = formation.set_index("listing_id")["sector_membership_asof"].astype(str)
        history = frame.loc[frame["session_date"].isin(adv_sessions)].copy()
        history["dollar_volume"] = pd.to_numeric(
            history["close_raw"], errors="coerce"
        ) * pd.to_numeric(history["volume_raw"], errors="coerce")
        adv = history.groupby("listing_id", sort=False)["dollar_volume"].agg(
            lambda values: (
                float(np.mean(values))
                if len(values) == 21 and np.isfinite(values.to_numpy(float)).all()
                else np.nan
            )
        )
        prices = frame.pivot(
            index="session_date", columns="listing_id", values="provider_adjusted_close"
        )
        start = pd.to_numeric(prices.loc[leader_formation], errors="coerce")
        end = pd.to_numeric(prices.loc[t], errors="coerce")
        returns = np.log((end / start).where((start > 0.0) & (end > 0.0)))
        for sector in sorted(set(memberships.dropna())):
            members = sorted(
                listing
                for listing, value in memberships.items()
                if value == sector and listing in adv
            )
            if len(members) < 10:
                continue
            leader_count = max(2, ceil(0.20 * len(members)))
            ranked = sorted(members, key=lambda listing: (-float(adv[listing]), str(listing)))
            leaders = tuple(listing for listing in ranked if np.isfinite(adv[listing]))[
                :leader_count
            ]
            if len(leaders) < 2 or not np.isfinite(returns.reindex(leaders).to_numpy(float)).all():
                continue
            for index, row in current.loc[current["listing_id"].isin(members)].iterrows():
                listing = str(row["listing_id"])
                own = float(returns.get(listing, np.nan))
                peers = tuple(item for item in leaders if item != listing)
                if not peers or not np.isfinite(own):
                    continue
                output.loc[index] = float(np.mean(returns.reindex(peers).to_numpy(float)) - own)
    frame["_value"] = output
    restored = frame.sort_values("_position", kind="mergesort")
    return pd.Series(restored["_value"].to_numpy(dtype=float), index=source.index)


__all__ = [
    "SECTOR_LEADER_LAG_5_ID",
    "SECTOR_LEADER_LAG_DECLARATION",
    "SECTOR_LEADER_LAG_METHOD_FAMILY",
    "SECTOR_LEADER_LAG_REQUIRED_FIELDS",
    "leader_lag_factor_specs",
    "sector_leader_lag_5",
]
