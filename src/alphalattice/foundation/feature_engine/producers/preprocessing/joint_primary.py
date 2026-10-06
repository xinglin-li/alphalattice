"""Development-only Joint Primary cross-sectional preprocessing methods."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from datetime import date
from functools import lru_cache
from hashlib import sha256
from typing import Final, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import ndtri  # type: ignore[import-untyped]

from alphalattice.foundation.feature_engine.contracts import (
    FeaturePanelBinding,
    PanelAdmissionSummary,
)
from alphalattice.foundation.feature_engine.producers.arithmetic_identity import (
    PREPROCESSING_WALK_EXCLUDED,
    feature_component_identity,
)
from alphalattice.kernel.quant.cross_section import (
    MAD_SCALE,
    MIN_COVERAGE,
    MIN_SECTOR_SAMPLE,
    equal_sector_demean,
    median_mad_winsor,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .adapters import require_uniform_membership
from .contracts import PanelPreprocessingImplementationBinding
from .robust_cross_section import (
    SMALL_SECTOR_WARNING_BELOW,
    PanelFactorClipObservation,
    PanelMaterialization,
)

JOINT_PRIMARY_RELATIVE_FACTOR_STD_Z_IMPLEMENTATION = (
    "feature_engine.producers.preprocessing.joint_primary.JointPrimaryCrossSectionKernel"
)
JOINT_PRIMARY_STATE_INTERACTION_BLOCK_IMPLEMENTATION = (
    "feature_engine.producers.preprocessing.joint_primary.JointPrimaryStateInteractionBlockAdapter"
)
INTERACTION_CLIP = 5.0
FORMULA_PRETRANSFORM_METHOD_ID = "feature_engine.formula-pretransform"
type FormulaPretransformRuleId = Literal[
    "IDENTITY",
    "POSITIVE_NATURAL_LOG",
    "PER_SESSION_RANK_GAUSSIAN",
]
FORMULA_PRETRANSFORM_RULES: Final[Mapping[str, FormulaPretransformRuleId]] = {
    "dollar_volume_21": "POSITIVE_NATURAL_LOG",
    "dollar_volume_252": "POSITIVE_NATURAL_LOG",
    "session_span": "PER_SESSION_RANK_GAUSSIAN",
    "previous_close_excursion_scaled_span_square": "PER_SESSION_RANK_GAUSSIAN",
    "previous_close_excursion_intraday_cross": "PER_SESSION_RANK_GAUSSIAN",
    "previous_close_excursion_intraday_adjusted_square": "PER_SESSION_RANK_GAUSSIAN",
    "gap_amplitude": "PER_SESSION_RANK_GAUSSIAN",
    "intraday_amplitude": "PER_SESSION_RANK_GAUSSIAN",
    "kurt_63": "PER_SESSION_RANK_GAUSSIAN",
}


def _array_digest(values: np.ndarray) -> str:
    finite = np.isfinite(values)
    return str(
        canonical_hash(
            {
                "kind": "PanelArrayDigest",
                "dtype": str(values.dtype),
                "shape": list(values.shape),
                "finite_mask": sha256(np.ascontiguousarray(finite).tobytes()).hexdigest(),
                "finite_values": sha256(
                    np.ascontiguousarray(np.where(finite, values, 0.0), dtype=np.float64).tobytes()
                ).hexdigest(),
            }
        )
    )


def _rank_gauss_rows(
    values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    result: npt.NDArray[np.float64] = np.full(values.shape, np.nan, dtype=np.float64)
    for row_index, row in enumerate(values):
        finite = np.isfinite(row)
        source = row[finite]
        count = int(source.size)
        if count < 2:
            continue
        order = np.argsort(source, kind="mergesort")
        sorted_values = source[order]
        starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
        stops = np.r_[starts[1:], count]
        average = (starts + stops - 1) / 2.0 + 1.0
        ranks: npt.NDArray[np.float64] = np.empty(count, dtype=np.float64)
        ranks[order] = np.repeat(average, stops - starts)
        result[row_index, finite] = ndtri((ranks - 0.5) / count)
    return result


_FORMULA_PRETRANSFORM_RENDER_IDS: Final[Mapping[FormulaPretransformRuleId, str]] = {
    "IDENTITY": "identity",
    "POSITIVE_NATURAL_LOG": "positive_log",
    "PER_SESSION_RANK_GAUSSIAN": "rank_gauss",
}
# One declaration drives arithmetic, rendering, and durable descriptors. The
# former lowercase arithmetic map and uppercase descriptor map drifted apart:
# session-amplitude rules executed rank-Gaussian while receipts said identity.
#
# The installed defaults were selected under the 3.5-MAD band that follows.
# Session-amplitude columns use rank-Gaussian because their exact-zero mass made
# positive-log create absence across full causal windows. `session_dollar_volume`
# deliberately remains identity for the fixed 195-column recipe. Positive-log
# changes its retained nonzero geometry; its zero-volume rows are already absent
# from the final common model axis because other required inputs are unavailable.


def render_formula_pretransform_step() -> str:
    """The recipe sequence step, rendered rather than transcribed."""
    body = ",".join(
        f"{key}={_FORMULA_PRETRANSFORM_RENDER_IDS[FORMULA_PRETRANSFORM_RULES[key]]}"
        for key in sorted(FORMULA_PRETRANSFORM_RULES)
    )
    return f"factor_pretransform[{body},default=identity]"


def formula_pretransform(
    factor_id: str,
    values: npt.NDArray[np.float64],
    *,
    rule_id: FormulaPretransformRuleId | None = None,
) -> npt.NDArray[np.float64]:
    """Apply the Formula-owned pretransform without taking later scale authority.

    Feature views consume this public seam before their causal temporal
    transforms. Keeping it here prevents Alpha from copying the dollar-volume
    log or kurtosis rank-Gaussian rules and becoming a second numerical owner.

    Args:
        factor_id: Formula whose installed default rule is resolved when no rule is supplied.
        values: Source values, with session rows and listing columns for rank-Gaussian rules.
        rule_id: Explicit rule authority, or the Formula's installed default.

    Returns:
        Read-only contiguous values after identity, positive natural log, or per-session
        rank-Gaussian transformation. Nonpositive log inputs and insufficient ranks remain missing.

    Raises:
        ValueError: Rank-Gaussian inputs do not have session and listing dimensions.
    """
    source = np.asarray(values, dtype=np.float64)
    selected_rule = rule_id or formula_pretransform_rule_id(factor_id)
    if selected_rule == "POSITIVE_NATURAL_LOG":
        output = np.full(source.shape, np.nan, dtype=np.float64)
        positive = np.isfinite(source) & (source > 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            output[positive] = np.log(source[positive])
    elif selected_rule == "PER_SESSION_RANK_GAUSSIAN":
        if source.ndim != 2:
            raise ValueError("formula pretransform rank-Gaussian requires session rows")
        output = _rank_gauss_rows(source)
    else:
        output = np.array(source, copy=True, dtype=np.float64)
    output = np.ascontiguousarray(output, dtype=np.float64)
    output.setflags(write=False)
    return output


def formula_pretransform_rule_id(factor_id: str) -> FormulaPretransformRuleId:
    """Resolve the Formula's installed default without applying it.

    Args:
        factor_id: Formula whose installed pretransform rule is requested.

    Returns:
        Installed rule, defaulting to IDENTITY for a Formula without a special rule.
    """
    return FORMULA_PRETRANSFORM_RULES.get(factor_id, "IDENTITY")


def formula_pretransform_descriptor(
    factor_id: str, *, rule_id: FormulaPretransformRuleId | None = None
) -> dict[str, str]:
    """Return the stable Formula descriptor consumed by Feature identities.

    Args:
        factor_id: Formula named by the durable descriptor.
        rule_id: Explicit rule authority, or the Formula's installed default.

    Returns:
        Method, Formula, rule, and owner names used by Feature identities.
    """
    return {
        "method_id": FORMULA_PRETRANSFORM_METHOD_ID,
        "factor_id": factor_id,
        "rule_id": rule_id or formula_pretransform_rule_id(factor_id),
        "owner": "feature_engine.formula",
    }


class JointPrimaryCrossSectionKernel:
    """3.5 normalized-MAD, one Sector demean, then sample-std Z."""

    def materialize(
        self,
        *,
        feature_rows: list[dict[str, object]] | pd.DataFrame,
        active_listing_ids: tuple[str, ...],
        sector_by_listing_id: dict[str, str],
        factor_ids: tuple[str, ...],
        binding: FeaturePanelBinding,
        members_by_session: Mapping[date, Sequence[str]] | None = None,
    ) -> PanelMaterialization:
        """Pretransform, winsorize, demean once by sector, and scale by sample standard deviation.

        Args:
            feature_rows: Listing-session source feature values.
            active_listing_ids: Declared calculation axis in execution order.
            sector_by_listing_id: Sector authority for every axis listing.
            factor_ids: Ordered factor scope to transform.
            binding: Panel lineage and recorded numerical policy authority.
            members_by_session: Nominal session members, or the full uniform axis when omitted.

        Returns:
            Transformed rows, availability and admission, measured clipping observations,
                and ordered input/output identities bound to the supplied Panel lineage.

        Raises:
            KeyError: A required listing, session, or factor source column is absent.
            ValueError: Rows, sorted unique factor scope, sectors, or required source fields
                are invalid.
            PanelPreprocessingError: Declared membership differs from the full calculation axis.
        """
        require_uniform_membership(members_by_session, active_listing_ids)
        if (isinstance(feature_rows, pd.DataFrame) and feature_rows.empty) or not len(feature_rows):
            raise ValueError("sector panel requires feature rows")
        if set(active_listing_ids) - set(sector_by_listing_id):
            raise ValueError("sector panel has no complete active sector revision")
        if not factor_ids or factor_ids != tuple(sorted(set(factor_ids))):
            raise ValueError("sector panel factor IDs must be sorted and unique")
        frame = (
            feature_rows.copy()
            if isinstance(feature_rows, pd.DataFrame)
            else pd.DataFrame(feature_rows)
        )
        frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date
        frame = frame.loc[frame["listing_id"].isin(active_listing_ids)].copy()
        sessions = tuple(sorted(frame["session_date"].unique()))
        rows = pd.DataFrame(
            {
                "listing_id": np.tile(np.asarray(active_listing_ids, dtype=object), len(sessions)),
                "session_date": np.repeat(
                    np.asarray([session.isoformat() for session in sessions], dtype=object),
                    len(active_listing_ids),
                ),
            }
        )
        listing_positions = {
            listing_id: position for position, listing_id in enumerate(active_listing_ids)
        }
        sector_positions: dict[str, npt.NDArray[np.int64]] = {
            sector: np.asarray(
                [
                    listing_positions[listing_id]
                    for listing_id in active_listing_ids
                    if sector_by_listing_id[listing_id] == sector
                ],
                dtype=np.int64,
            )
            for sector in sorted(set(sector_by_listing_id.values()))
        }
        grid = pd.MultiIndex.from_product(
            (sessions, active_listing_ids), names=("session_date", "listing_id")
        )
        indexed = frame.set_index(["session_date", "listing_id"])
        if not indexed.index.is_unique:
            raise ValueError("sector panel feature rows contain duplicate listing sessions")
        aligned = indexed.reindex(grid)
        universe_size = len(active_listing_ids)
        present = (
            np.asarray(grid.isin(indexed.index), dtype=bool)
            .reshape(len(sessions), universe_size)
            .all(axis=1)
        )
        factor_cube = (
            aligned.loc[:, list(factor_ids)]
            .to_numpy(dtype=float)
            .reshape(len(sessions), universe_size, len(factor_ids))
        )
        # A non-finite raw value is missing, as NaN is, in every session statistic -- the
        # median, MAD, winsor and coverage alike -- so it never moves another name's value
        # (V517). The producers screen their own; the kernel does not rely on them.
        factor_cube = np.where(np.isfinite(factor_cube), factor_cube, np.nan)
        availability: list[dict[str, object]] = []
        clips: list[PanelFactorClipObservation] = []
        raw_digests: list[tuple[str, str]] = []
        transformed_digests: list[tuple[str, str]] = []
        for factor_position, factor_id in enumerate(factor_ids):
            raw_matrix = cast(npt.NDArray[np.float64], factor_cube[:, :, factor_position])
            matrix = formula_pretransform(factor_id, raw_matrix)
            valid = np.isfinite(matrix)
            computed = valid.sum(axis=1)
            coverage = computed / universe_size
            sector_counts = {
                sector: valid[:, positions].sum(axis=1)
                for sector, positions in sector_positions.items()
            }
            small_sector: npt.NDArray[np.bool_] = np.zeros(len(sessions), dtype=bool)
            small_sector_warning: npt.NDArray[np.bool_] = np.zeros(len(sessions), dtype=bool)
            for count in sector_counts.values():
                small_sector |= count < MIN_SECTOR_SAMPLE
                small_sector_warning |= (count >= MIN_SECTOR_SAMPLE) & (
                    count < SMALL_SECTOR_WARNING_BELOW
                )
            winsor, _median, mad, lower, upper = median_mad_winsor(
                matrix, multiplier=3.5, mad_scale=MAD_SCALE
            )
            clipped = valid & ((matrix < lower[:, None]) | (matrix > upper[:, None]))
            clips.append(
                PanelFactorClipObservation(
                    factor_id=factor_id,
                    finite_input_count=int(valid.sum()),
                    per_session_finite_counts=tuple(int(value) for value in valid.sum(axis=1)),
                    per_session_clipped_counts=tuple(int(value) for value in clipped.sum(axis=1)),
                    boundary_identity=str(
                        canonical_hash(
                            {
                                "kind": "PanelFactorClipBoundaries",
                                "factor_id": factor_id,
                                "bounds": [
                                    (
                                        float(lower[index]) if np.isfinite(lower[index]) else None,
                                        float(upper[index]) if np.isfinite(upper[index]) else None,
                                    )
                                    for index in range(len(sessions))
                                ],
                            }
                        )
                    ),
                )
            )
            residual = equal_sector_demean(
                winsor, sector_positions=tuple(sector_positions.values())
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                residual_median = np.nanmedian(residual, axis=1)
                residual_mad = np.nanmedian(np.abs(residual - residual_median[:, None]), axis=1)
                residual_scale = np.nanstd(residual, axis=1, ddof=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                score = residual / residual_scale[:, None]
            available = (
                present
                & (coverage >= MIN_COVERAGE)
                & ~small_sector
                & np.isfinite(mad)
                & (mad != 0.0)
                & np.isfinite(residual_scale)
                & (residual_scale != 0.0)
            )
            score[~available, :] = np.nan
            for index, session in enumerate(sessions):
                if not present[index]:
                    reason = "active_manifest_base_row_missing"
                elif coverage[index] < MIN_COVERAGE:
                    reason = "coverage_below_98_percent"
                elif small_sector[index]:
                    reason = "sector_sample_below_5"
                elif not np.isfinite(mad[index]) or mad[index] == 0.0:
                    reason = "feature_mad_zero"
                elif not np.isfinite(residual_scale[index]) or residual_scale[index] == 0.0:
                    reason = "residual_scale_zero"
                else:
                    reason = None
                availability.append(
                    {
                        "session_date": session.isoformat(),
                        "factor_id": factor_id,
                        "universe_size": universe_size,
                        "computed_count": int(computed[index]),
                        "coverage": float(coverage[index]),
                        "sector_counts": {
                            sector: int(count[index]) for sector, count in sector_counts.items()
                        },
                        "winsor_lower": (
                            float(lower[index]) if np.isfinite(lower[index]) else None
                        ),
                        "winsor_upper": (
                            float(upper[index]) if np.isfinite(upper[index]) else None
                        ),
                        "residual_median": (
                            float(residual_median[index])
                            if np.isfinite(residual_median[index])
                            else None
                        ),
                        "residual_mad": (
                            float(residual_mad[index]) if np.isfinite(residual_mad[index]) else None
                        ),
                        "status": "available" if available[index] else "unavailable",
                        "reason": reason,
                        "small_sector_warning": bool(
                            available[index] and small_sector_warning[index]
                        ),
                        "small_sector_names": tuple(
                            sector
                            for sector, count in sector_counts.items()
                            if MIN_SECTOR_SAMPLE <= int(count[index]) < SMALL_SECTOR_WARNING_BELOW
                        ),
                        "panel_binding_hash": binding.panel_binding_hash,
                    }
                )
            raw_digests.append((factor_id, _array_digest(raw_matrix)))
            transformed_digests.append((factor_id, _array_digest(score)))
            flat_score = score.reshape(-1)
            rows[factor_id] = pd.Series(flat_score).where(np.isfinite(flat_score), np.nan)
        latest_session = max(cast(str, item["session_date"]) for item in availability)
        sector_distribution = {
            sector: len(positions) for sector, positions in sector_positions.items()
        }
        admission = PanelAdmissionSummary.evaluate(
            as_of_session=pd.Timestamp(latest_session).date(),
            factor_ids=factor_ids,
            availability=availability,
            sector_distribution=sector_distribution,
        )
        receipt_hash = canonical_hash(
            {
                "binding": binding.panel_binding_hash,
                "row_count": len(rows),
                "availability": [
                    {
                        key: value
                        for key, value in item.items()
                        if key not in {"sector_counts", "small_sector_names"}
                    }
                    for item in availability
                ],
                "admission": admission.summary_hash(),
            }
        )
        return PanelMaterialization(
            rows=rows,
            availability=availability,
            binding=binding,
            receipt_hash=receipt_hash,
            complete=admission.research_admissible,
            admission=admission,
            clip_observations=tuple(clips),
            raw_input_identity=str(
                canonical_hash({"kind": "PanelRawInput", "factors": raw_digests})
            ),
            transformed_identity=str(
                canonical_hash({"kind": "PanelTransformed", "factors": transformed_digests})
            ),
            ordered_sessions_hash=str(
                canonical_hash([session.isoformat() for session in sessions])
            ),
            ordered_listing_ids_hash=str(canonical_hash(list(active_listing_ids))),
        )


_JOINT_PRIMARY_OWNERS = (
    "alphalattice.foundation.feature_engine.producers.preprocessing.joint_primary",
    "alphalattice.foundation.feature_engine.producers.preprocessing.robust_cross_section",
    "alphalattice.kernel.quant.cross_section",
)


@lru_cache(maxsize=1)
def _relative_content_hash() -> str:
    return feature_component_identity(
        "PANEL_PREPROCESSING:joint_primary_relative",
        owners=_JOINT_PRIMARY_OWNERS,
        excluded=PREPROCESSING_WALK_EXCLUDED,
        semantic_owner="feature_engine.producers.preprocessing.joint_primary",
        numerical_role="PANEL_PREPROCESSING",
    )


@lru_cache(maxsize=1)
def _interaction_content_hash() -> str:
    return feature_component_identity(
        "DEVELOPMENT_PANEL_PREPROCESSING:joint_primary_interaction",
        owners=_JOINT_PRIMARY_OWNERS,
        excluded=PREPROCESSING_WALK_EXCLUDED,
        semantic_owner="feature_engine.producers.preprocessing.joint_primary_interaction",
        numerical_role="DEVELOPMENT_PANEL_PREPROCESSING",
    )


class JointPrimaryRelativeFactorStdZAdapter:
    """Adapt Joint Primary relative-factor sample-standard-deviation scaling for development."""

    def __init__(self) -> None:
        """Create the Joint Primary numerical cross-section kernel."""
        self._kernel = JointPrimaryCrossSectionKernel()

    @property
    def implementation_id(self) -> str:
        """Return this Joint Primary adapter's stable executable handle."""
        return JOINT_PRIMARY_RELATIVE_FACTOR_STD_Z_IMPLEMENTATION

    def describe_implementation_binding(self) -> PanelPreprocessingImplementationBinding:
        """Describe the measured Joint Primary executable and its numerical owners.

        Returns:
            Canonical handle, owner, and content binding of this development transformation.
        """
        return PanelPreprocessingImplementationBinding.create(
            implementation_id=self.implementation_id,
            implementation_owners=(
                "alphalattice.foundation.feature_engine.producers.preprocessing.joint_primary",
                "alphalattice.foundation.feature_engine.producers.preprocessing.robust_cross_section",
                "alphalattice.kernel.quant.cross_section",
            ),
            implementation_content_hash=_relative_content_hash(),
        )

    def materialize(
        self,
        *,
        feature_rows: list[dict[str, object]] | pd.DataFrame,
        active_listing_ids: tuple[str, ...],
        sector_by_listing_id: dict[str, str],
        factor_ids: tuple[str, ...],
        binding: FeaturePanelBinding,
        members_by_session: Mapping[date, Sequence[str]] | None = None,
    ) -> PanelMaterialization:
        """Transform a uniform Panel with the Joint Primary relative-factor kernel.

        Args:
            feature_rows: Listing-session source feature values.
            active_listing_ids: Declared calculation axis in execution order.
            sector_by_listing_id: Sector authority for every axis listing.
            factor_ids: Ordered factor scope to transform.
            binding: Panel lineage and recorded numerical policy authority.
            members_by_session: Nominal session members, or the full uniform axis when omitted.

        Returns:
            Transformed rows, availability and admission, measured clipping observations,
                and ordered input/output identities bound to the supplied Panel lineage.

        Raises:
            KeyError: A required listing, session, or factor source column is absent.
            ValueError: Source axes or sectors are invalid, rows are duplicated, or a
                required factor or state child is absent.
            PanelPreprocessingError: Membership differs from the full calculation axis.
        """
        return self._kernel.materialize(
            feature_rows=feature_rows,
            active_listing_ids=active_listing_ids,
            sector_by_listing_id=sector_by_listing_id,
            factor_ids=factor_ids,
            binding=binding,
            members_by_session=members_by_session,
        )


class JointPrimaryStateInteractionBlockAdapter:
    """Multiply Joint Primary stock scores by their state children and bound the interaction."""

    @property
    def implementation_id(self) -> str:
        """Return this Joint Primary adapter's stable executable handle."""
        return JOINT_PRIMARY_STATE_INTERACTION_BLOCK_IMPLEMENTATION

    def describe_implementation_binding(self) -> PanelPreprocessingImplementationBinding:
        """Describe the measured Joint Primary executable and its numerical owners.

        Returns:
            Canonical handle, owner, and content binding of this development transformation.
        """
        return PanelPreprocessingImplementationBinding.create(
            implementation_id=self.implementation_id,
            implementation_owners=(
                "alphalattice.foundation.feature_engine.producers.preprocessing.joint_primary",
                "alphalattice.foundation.feature_engine.producers.preprocessing.robust_cross_section",
                "alphalattice.kernel.quant.cross_section",
            ),
            implementation_content_hash=_interaction_content_hash(),
        )

    def materialize(
        self,
        *,
        feature_rows: list[dict[str, object]] | pd.DataFrame,
        active_listing_ids: tuple[str, ...],
        sector_by_listing_id: dict[str, str],
        factor_ids: tuple[str, ...],
        binding: FeaturePanelBinding,
        members_by_session: Mapping[date, Sequence[str]] | None = None,
    ) -> PanelMaterialization:
        """Multiply Joint Primary stock scores by state values and clip to [-5, 5].

        Args:
            feature_rows: Listing-session source feature values.
            active_listing_ids: Declared calculation axis in execution order.
            sector_by_listing_id: Sector authority for every axis listing.
            factor_ids: Ordered factor scope to transform.
            binding: Panel lineage and recorded numerical policy authority.
            members_by_session: Nominal session members, or the full uniform axis when omitted.

        Returns:
            Transformed rows, availability and admission, measured clipping observations,
                and ordered input/output identities bound to the supplied Panel lineage.

        Raises:
            KeyError: A required listing, session, or factor source column is absent.
            ValueError: Source axes or sectors are invalid, rows are duplicated, or a
                required factor or state child is absent.
            PanelPreprocessingError: Membership differs from the full calculation axis.
        """
        frame = (
            feature_rows.copy()
            if isinstance(feature_rows, pd.DataFrame)
            else pd.DataFrame(feature_rows)
        )
        child = JointPrimaryCrossSectionKernel().materialize(
            feature_rows=feature_rows,
            active_listing_ids=active_listing_ids,
            sector_by_listing_id=sector_by_listing_id,
            factor_ids=factor_ids,
            binding=binding,
            members_by_session=members_by_session,
        )
        frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date
        frame = frame.loc[frame["listing_id"].isin(active_listing_ids)]
        if frame.duplicated(["session_date", "listing_id"]).any():
            raise ValueError("development preprocessing rows contain duplicate listing sessions")
        sessions = tuple(cast(date, item) for item in sorted(frame["session_date"].unique()))
        grid = pd.MultiIndex.from_product(
            (sessions, active_listing_ids), names=("session_date", "listing_id")
        )
        aligned = frame.set_index(["session_date", "listing_id"]).reindex(grid)
        rows = child.rows.copy()
        observations: list[PanelFactorClipObservation] = []
        availability: list[dict[str, object]] = []
        raw_pairs: list[tuple[str, str]] = []
        transformed_pairs: list[tuple[str, str]] = []
        for factor_id in factor_ids:
            state_column = f"{factor_id}__state"
            if state_column not in aligned:
                raise ValueError("STATE_INTERACTION_MARKET_CHILD_MISSING")
            stock = pd.to_numeric(rows[factor_id], errors="coerce").to_numpy(float)
            state = pd.to_numeric(aligned[state_column], errors="coerce").to_numpy(float)
            product = stock * state
            bounded = np.clip(product, -INTERACTION_CLIP, INTERACTION_CLIP)
            bounded[~np.isfinite(product)] = np.nan
            clipped = np.isfinite(product) & (bounded != product)
            rows[factor_id] = bounded
            matrix = bounded.reshape(len(sessions), len(active_listing_ids))
            for session_position, session in enumerate(sessions):
                finite = np.isfinite(matrix[session_position])
                sector_counts = {
                    sector: int(
                        sum(
                            finite[position]
                            for position, listing in enumerate(active_listing_ids)
                            if sector_by_listing_id[listing] == sector
                        )
                    )
                    for sector in sorted(set(sector_by_listing_id.values()))
                }
                coverage = float(finite.mean())
                available = coverage >= 0.98 and all(
                    value >= MIN_SECTOR_SAMPLE for value in sector_counts.values()
                )
                availability.append(
                    {
                        "session_date": session.isoformat(),
                        "factor_id": factor_id,
                        "universe_size": len(active_listing_ids),
                        "computed_count": int(finite.sum()),
                        "coverage": coverage,
                        "sector_counts": sector_counts,
                        "status": "available" if available else "unavailable",
                        "reason": (None if available else "interaction_child_or_state_unavailable"),
                        "small_sector_warning": False,
                        "small_sector_names": (),
                        "panel_binding_hash": binding.panel_binding_hash,
                    }
                )
            observations.append(
                PanelFactorClipObservation(
                    factor_id=factor_id,
                    finite_input_count=int(np.isfinite(product).sum()),
                    per_session_finite_counts=tuple(
                        int(value)
                        for value in np.isfinite(product).reshape(len(sessions), -1).sum(axis=1)
                    ),
                    per_session_clipped_counts=tuple(
                        int(value) for value in clipped.reshape(len(sessions), -1).sum(axis=1)
                    ),
                    boundary_identity=str(
                        canonical_hash({"lower": -INTERACTION_CLIP, "upper": INTERACTION_CLIP})
                    ),
                )
            )
            raw_pairs.append((factor_id, _array_digest(product)))
            transformed_pairs.append((factor_id, _array_digest(bounded)))
        sector_distribution = {
            sector: sum(value == sector for value in sector_by_listing_id.values())
            for sector in sorted(set(sector_by_listing_id.values()))
        }
        admission = PanelAdmissionSummary.evaluate(
            as_of_session=pd.Timestamp(sessions[-1]).date(),
            factor_ids=factor_ids,
            availability=availability,
            sector_distribution=sector_distribution,
        )
        return PanelMaterialization(
            rows=rows,
            availability=availability,
            binding=binding,
            receipt_hash=str(
                canonical_hash({"child": child.receipt_hash, "clip": INTERACTION_CLIP})
            ),
            complete=admission.research_admissible,
            admission=admission,
            clip_observations=tuple(observations),
            raw_input_identity=str(canonical_hash(raw_pairs)),
            transformed_identity=str(canonical_hash(transformed_pairs)),
            ordered_sessions_hash=child.ordered_sessions_hash,
            ordered_listing_ids_hash=child.ordered_listing_ids_hash,
        )


__all__ = [
    "FORMULA_PRETRANSFORM_METHOD_ID",
    "FORMULA_PRETRANSFORM_RULES",
    "JOINT_PRIMARY_RELATIVE_FACTOR_STD_Z_IMPLEMENTATION",
    "JOINT_PRIMARY_STATE_INTERACTION_BLOCK_IMPLEMENTATION",
    "FormulaPretransformRuleId",
    "JointPrimaryCrossSectionKernel",
    "JointPrimaryRelativeFactorStdZAdapter",
    "JointPrimaryStateInteractionBlockAdapter",
    "formula_pretransform",
    "formula_pretransform_descriptor",
    "formula_pretransform_rule_id",
    "render_formula_pretransform_step",
]
