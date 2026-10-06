"""Fail-closed current-sector neutral cross-sectional feature panel.

Composition only. The numerical owner is
``producers/preprocessing/robust_cross_section``; the installed method is
resolved from the preprocessing catalog here, and what the transformation
actually did is sealed into durable clipping evidence bound to that method.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import cast

import pandas as pd

from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.contracts import FeaturePanelBinding
from alphalattice.foundation.feature_engine.producers.preprocessing.catalog import (
    ROBUST_SECTOR_NEUTRAL_Z,
    PanelPreprocessingCatalog,
    build_installed_panel_preprocessing_catalog,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.contracts import (
    PanelClipObservationRecord,
    PanelClippingEvidence,
    PanelFactorClipObservationRecord,
    PanelFactorClippingRecord,
    PanelPreprocessingBinding,
    PanelPreprocessingError,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.robust_cross_section import (
    SMALL_SECTOR_WARNING_BELOW,
    PanelFactorClipObservation,
    PanelMaterialization,
    cross_section_policy_hash,
)
from alphalattice.kernel.quant.cross_section import (
    MAD_SCALE,
    MIN_COVERAGE,
    MIN_SECTOR_SAMPLE,
    WINSOR_MULTIPLIER,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

CellOwnership = Mapping[tuple[str, str], frozenset[str]]
"""``(receipt_hash, factor_id) -> sessions`` a Panel attributes to that batch."""


def clip_observation_record(materialization: PanelMaterialization) -> PanelClipObservationRecord:
    """Seal one batch's measured clipping facts under the receipt its rows carry.

    Args:
        materialization: Batch output carrying measured clipping observations and axes.

    Returns:
        Sealed batch receipt binding per-factor clipping counts and boundaries to
        its Panel, session axis, and raw and transformed input identities.

    Raises:
        PanelPreprocessingError: The materialization carries no clipping observations.
    """
    if not materialization.clip_observations:
        raise PanelPreprocessingError("PANEL_CLIPPING_EVIDENCE_OBSERVATIONS_MISSING")
    sessions: list[str] = []
    for item in materialization.availability:
        session = str(item["session_date"])
        if session not in sessions:
            sessions.append(session)
    return PanelClipObservationRecord.seal(
        {
            "receipt_hash": materialization.receipt_hash,
            "panel_binding_hash": materialization.binding.panel_binding_hash,
            "sessions": tuple(sessions),
            "ordered_sessions_hash": materialization.ordered_sessions_hash,
            "ordered_listing_ids_hash": materialization.ordered_listing_ids_hash,
            "factor_ids": tuple(item.factor_id for item in materialization.clip_observations),
            "raw_input_identity": materialization.raw_input_identity,
            "transformed_identity": materialization.transformed_identity,
            "observations": tuple(
                PanelFactorClipObservationRecord(
                    factor_id=item.factor_id,
                    finite_input_count=item.finite_input_count,
                    per_session_finite_counts=item.per_session_finite_counts,
                    per_session_clipped_counts=item.per_session_clipped_counts,
                    boundary_identity=item.boundary_identity,
                )
                for item in materialization.clip_observations
            ),
        }
    )


def _merge_clip_records(
    records: Sequence[PanelClipObservationRecord],
    owned: CellOwnership,
    sessions: Sequence[str],
) -> tuple[tuple[PanelFactorClipObservation, ...], tuple[PanelClipObservationRecord, ...]]:
    """Join per-batch observations into one per-factor view of the Panel.

    ``sessions`` is the Panel's session axis and ``owned`` says which receipt
    each ``(session, factor)`` cell belongs to. Every cell of the axis must
    be owned by exactly one receipt whose record observed that very cell;
    a cell owned twice, owned by no one, outside the axis, or owned by a
    receipt whose record does not cover it is a contradiction between the
    availability rows and the records, and the fold refuses it rather than
    seal counts that describe some other Panel.

    The per-session arrays follow the axis, in calendar order, whichever
    batch measured each cell; the records that contributed are ordered by
    the first session they contributed, which for one build is batch order.
    """
    axis = tuple(sessions)
    if axis != tuple(sorted(set(axis))):
        raise PanelPreprocessingError("PANEL_CLIPPING_EVIDENCE_SESSION_AXIS_INVALID")
    on_axis = set(axis)
    by_receipt: dict[str, PanelClipObservationRecord] = {}
    for record in records:
        if by_receipt.setdefault(record.receipt_hash, record) is not record:
            raise PanelPreprocessingError("PANEL_CLIPPING_EVIDENCE_RECORD_DUPLICATED")
    # (factor, session) -> receipt: each cell owned once, inside the axis.
    owners: dict[str, dict[str, str]] = {}
    for (receipt, factor_id), cells in owned.items():
        owners.setdefault(factor_id, {})
        for session in cells:
            if session not in on_axis:
                raise PanelPreprocessingError("PANEL_CLIPPING_EVIDENCE_CELL_OUTSIDE_AXIS")
            if session in owners[factor_id]:
                raise PanelPreprocessingError("PANEL_CLIPPING_EVIDENCE_CELL_OWNED_TWICE")
            owners[factor_id][session] = receipt
    # The factor axis is lexical, as every batch of a build lists its factors.
    factor_order = sorted(owners)
    record_sessions = {
        receipt: {session: index for index, session in enumerate(record.sessions)}
        for receipt, record in by_receipt.items()
    }
    record_observations = {
        receipt: {item.factor_id: item for item in record.observations}
        for receipt, record in by_receipt.items()
    }
    first_contribution: dict[str, int] = {}
    boundaries: dict[str, list[str]] = {}
    finite_by_factor: dict[str, list[int]] = {}
    clipped_by_factor: dict[str, list[int]] = {}
    for factor_id in factor_order:
        finite_by_factor[factor_id] = []
        clipped_by_factor[factor_id] = []
        boundaries[factor_id] = []
        cell_owners = owners[factor_id]
        if len(cell_owners) != len(axis):
            raise PanelPreprocessingError("PANEL_CLIPPING_EVIDENCE_CELL_UNOWNED")
        contributed: set[str] = set()
        for position, session in enumerate(axis):
            receipt = cell_owners[session]
            if receipt not in by_receipt:
                raise PanelPreprocessingError("PANEL_CLIPPING_EVIDENCE_OBSERVATIONS_MISSING")
            observation = record_observations[receipt].get(factor_id)
            index = record_sessions[receipt].get(session)
            if observation is None or index is None:
                raise PanelPreprocessingError("PANEL_CLIPPING_EVIDENCE_CELL_NOT_OBSERVED")
            finite_by_factor[factor_id].append(observation.per_session_finite_counts[index])
            clipped_by_factor[factor_id].append(observation.per_session_clipped_counts[index])
            if receipt not in contributed:
                contributed.add(receipt)
                boundaries[factor_id].append(observation.boundary_identity)
            first_contribution[receipt] = min(first_contribution.get(receipt, position), position)
    ordered_receipts = sorted(first_contribution, key=lambda key: (first_contribution[key], key))
    contributing = tuple(by_receipt[receipt] for receipt in ordered_receipts)
    merged = tuple(
        PanelFactorClipObservation(
            factor_id=factor_id,
            finite_input_count=sum(finite_by_factor[factor_id]),
            per_session_finite_counts=tuple(finite_by_factor[factor_id]),
            per_session_clipped_counts=tuple(clipped_by_factor[factor_id]),
            # The batch boundaries behind this factor's cells, in the order
            # the batches first appear on the axis for this factor.
            boundary_identity=str(canonical_hash(boundaries[factor_id])),
        )
        for factor_id in factor_order
    )
    return merged, contributing


class SectorNeutralPanelMaterializer:
    """Catalog-validating adapter over the explicit numerical kernel."""

    def __init__(
        self,
        catalog: FeatureCatalog,
        *,
        preprocessing: PanelPreprocessingCatalog | None = None,
        recipe_id: str = ROBUST_SECTOR_NEUTRAL_Z,
    ) -> None:
        """Resolve an admitted preprocessing method before transforming a Panel.

        Args:
            catalog: Qualified feature catalog whose factor axis may be materialized.
            preprocessing: Preprocessing catalog; the installed catalog when omitted.
            recipe_id: Installed method whose recipe, executable, and implementation bind
                the transformed output.

        Raises:
            PanelPreprocessingError: The recipe is absent, not admitted for an active
                Panel, or cannot resolve its executable and implementation binding.
        """
        self.catalog = catalog
        self.policy_hash = cross_section_policy_hash()
        self.preprocessing = preprocessing or build_installed_panel_preprocessing_catalog()
        # Resolved before anything runs, so an uninstalled or unadmitted method
        # fails here rather than after a Panel has already been transformed by
        # it. The adapter comes back with the recipe: the method that was named
        # is the code that runs, instead of a fixed kernel that would happily
        # compute under whatever identity the recipe happened to carry.
        self.capability = self.preprocessing.admit_for_active_panel(recipe_id)
        self.recipe, self._adapter = self.preprocessing.resolve_executable(recipe_id)
        # Re-derived from the installed catalog rather than accepted from a
        # caller or copied off the capability: the writer must record the
        # implementation it is about to run, and the only authority for that is
        # the catalog that resolved it.
        self.implementation = self.preprocessing.implementation_binding(recipe_id)

    def preprocessing_binding(self, *, panel_binding_hash: str) -> PanelPreprocessingBinding:
        """Which installed method one Panel materialization was produced by.

        Args:
            panel_binding_hash: Lineage identity of the Panel whose transformation is bound.

        Returns:
            Canonically hashed binding of recipe, executable implementation, catalog,
            policy, and the supplied Panel identity.
        """
        values: dict[str, object] = {
            "kind": "PanelPreprocessingBinding",
            "recipe_id": self.recipe.recipe_id,
            "recipe_hash": self.recipe.recipe_hash,
            "implementation_id": self.capability.implementation_id,
            "implementation_binding_hash": self.implementation.implementation_binding_hash,
            "implementation": self.implementation.model_dump(mode="json"),
            "catalog_hash": self.preprocessing.catalog_hash,
            "policy_hash": self.policy_hash,
            "panel_binding_hash": panel_binding_hash,
        }
        return PanelPreprocessingBinding(
            **{**values, "implementation": self.implementation},
            binding_hash=canonical_hash(values),
        )

    def clipping_evidence_from_records(
        self,
        records: Sequence[PanelClipObservationRecord],
        *,
        panel_binding_hash: str,
        owned: CellOwnership,
        sessions: Sequence[str],
    ) -> PanelClippingEvidence:
        """Seal what the transformation changed, bound to the arrays it changed.

        Built from observations the kernel measured while transforming, not
        recomputed from published rows: after the fact a clipped value and a
        value that always sat at the boundary are indistinguishable. A Panel
        is materialized in session batches, and once a build reuses
        partitions another build wrote, the batches that produced its cells
        come from several builds; each batch's record still describes the
        batch it measured. ``sessions`` is the Panel's session axis and
        ``owned`` (from its availability rows) says which receipt each cell
        belongs to. The evidence describes the Panel bound to
        ``panel_binding_hash``: per-session counts on its axis, batch
        identities folded in the order the batches first appear on it.

        Args:
            records: Measured batch receipts, including any receipts reused by this build.
            panel_binding_hash: Identity of the Panel described by the evidence.
            owned: Receipt authority for each published cell.
            sessions: Ordered session axis of that Panel.

        Returns:
            Sealed per-factor and per-session clipping observations, with contributing
            input and transformed identities folded in first-use batch order.

        Raises:
            PanelPreprocessingError: The session axis or receipt set is invalid, a cell
                lies outside that axis or lacks its unique measured receipt, or no
                observations contribute.
        """
        merged, contributing = _merge_clip_records(tuple(records), owned, sessions)
        if not merged or not contributing:
            raise PanelPreprocessingError("PANEL_CLIPPING_EVIDENCE_OBSERVATIONS_MISSING")
        ordered = contributing
        binding = self.preprocessing_binding(panel_binding_hash=panel_binding_hash)
        records_out: list[PanelFactorClippingRecord] = []
        for observation in merged:
            clipped = sum(observation.per_session_clipped_counts)
            record_values: dict[str, object] = {
                "kind": "PanelFactorClippingRecord",
                "factor_id": observation.factor_id,
                "finite_input_count": observation.finite_input_count,
                "clipped_count": clipped,
                "clipped_fraction": (
                    clipped / observation.finite_input_count
                    if observation.finite_input_count
                    else 0.0
                ),
                "per_session_clipped_counts": tuple(observation.per_session_clipped_counts),
                "per_session_clipped_fractions": tuple(
                    (count / finite) if finite else 0.0
                    for count, finite in zip(
                        observation.per_session_clipped_counts,
                        observation.per_session_finite_counts,
                        strict=True,
                    )
                ),
                "boundary_identity": observation.boundary_identity,
            }
            records_out.append(
                PanelFactorClippingRecord(
                    **record_values, record_hash=canonical_hash(record_values)
                )
            )
        evidence_values: dict[str, object] = {
            "kind": "PanelClippingEvidence",
            "preprocessing_binding_hash": binding.binding_hash,
            "recipe_id": self.recipe.recipe_id,
            "recipe_hash": self.recipe.recipe_hash,
            "panel_binding_hash": panel_binding_hash,
            "ordered_sessions_hash": canonical_hash(
                [value.ordered_sessions_hash for value in ordered]
            ),
            "ordered_listing_ids_hash": canonical_hash(
                [value.ordered_listing_ids_hash for value in ordered]
            ),
            "ordered_factor_ids": tuple(value.factor_id for value in merged),
            "raw_input_identity": canonical_hash([value.raw_input_identity for value in ordered]),
            "transformed_identity": canonical_hash(
                [value.transformed_identity for value in ordered]
            ),
            "factor_records": tuple(value.model_dump(mode="json") for value in records_out),
            "total_clipped_count": sum(value.clipped_count for value in records_out),
        }
        return PanelClippingEvidence(
            **{**evidence_values, "factor_records": tuple(records_out)},
            evidence_hash=canonical_hash(evidence_values),
        )

    def materialize(
        self,
        *,
        feature_rows: list[dict[str, object]] | pd.DataFrame,
        active_listing_ids: tuple[str, ...],
        manifest_revision: str,
        sector_revision: str,
        sector_by_listing_id: dict[str, str],
        spy_revision: str,
        factor_ids: tuple[str, ...] | None = None,
        members_by_session: Mapping[date, Sequence[str]] | None = None,
        source_exclusions_by_session: Mapping[date, Sequence[str]] | None = None,
    ) -> PanelMaterialization:
        """Transform an admitted feature scope through the resolved preprocessing adapter.

        Args:
            feature_rows: Raw listing-session feature values.
            active_listing_ids: Historical calculation axis in its declared order.
            manifest_revision: Source Data manifest authority.
            sector_revision: Sector classification authority.
            sector_by_listing_id: Sector mapping used by the cross-section transformation.
            spy_revision: Market-reference revision.
            factor_ids: Nonempty subset of catalog factors; the full axis when omitted.
            members_by_session: Nominal per-session members, or a uniform axis when omitted.
            source_exclusions_by_session: Evidenced source exclusions by session, if present.

        Returns:
            Transformed Panel with lineage and preprocessing bindings, availability,
            and measured clipping observations produced by the selected adapter.

        Raises:
            ValueError: The requested factor scope is empty or contains an unknown factor.
            PanelPreprocessingError: Inputs fail the resolved adapter's materialization contract.
        """
        factors = tuple(factor_ids) if factor_ids is not None else self.catalog.factor_ids
        if not factors or not set(factors).issubset(self.catalog.factor_ids):
            raise ValueError("sector panel contains an unknown factor scope")
        binding = FeaturePanelBinding.create(
            manifest_revision=manifest_revision,
            sector_revision=sector_revision,
            catalog_hash=self.catalog.binding.catalog_hash,
            spy_revision=spy_revision,
            policy_hash=self.policy_hash,
        )
        return cast(
            PanelMaterialization,
            self._adapter.materialize(
                feature_rows=feature_rows,
                active_listing_ids=active_listing_ids,
                sector_by_listing_id=sector_by_listing_id,
                factor_ids=factors,
                binding=binding,
                members_by_session=members_by_session,
                **(
                    {"source_exclusions_by_session": source_exclusions_by_session}
                    if source_exclusions_by_session
                    else {}
                ),
            ),
        )


__all__ = [
    "MAD_SCALE",
    "MIN_COVERAGE",
    "MIN_SECTOR_SAMPLE",
    "SMALL_SECTOR_WARNING_BELOW",
    "WINSOR_MULTIPLIER",
    "CellOwnership",
    "PanelClippingEvidence",
    "PanelMaterialization",
    "PanelPreprocessingBinding",
    "SectorNeutralPanelMaterializer",
    "clip_observation_record",
    "cross_section_policy_hash",
]
