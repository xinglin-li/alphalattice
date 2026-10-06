"""Current Alpha publication artifacts and authoritative marker-last readback."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from alphalattice.control.workspace_runtime.content_store import (
    verified_npz_arrays,
    verified_request_proof,
    verified_request_value,
    verified_source_value,
)
from alphalattice.investment.alpha_research.inputs.frozen_price_volume import (
    FrozenPriceVolumeInputs,
)
from alphalattice.investment.alpha_research.scores.product_replay import (
    HeterogeneousVintageFeatureSurface,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..experiments.contracts import AlphaCandidateRole, AlphaCandidateStatus
from ..experiments.development_artifacts import (
    AlphaDevelopmentArtifactReadbackError,
    AlphaDevelopmentArtifactStore,
)
from ..experiments.development_contracts import (
    AlphaCandidateDevelopmentReport,
    AlphaCandidateFoldEvidence,
    AlphaCandidateInferenceEvidence,
    AlphaCandidateNumericalFoldResult,
    AlphaDecisionReproducibilityReport,
    AlphaDevelopmentEstimatorState,
    AlphaDevelopmentFoldSurface,
    AlphaDevelopmentSurfaceBinding,
    AlphaDevelopmentSurfaceManifest,
    AlphaEstimatorState,
    AlphaModelSelectionDecision,
    AlphaModelViabilityAssessment,
    AlphaNumericalDevelopmentScoreChunkRef,
    LegacyAlphaCandidateFoldEvidence,
)
from .contracts import (
    AlphaCurrentCandidateScoreChild,
    AlphaCurrentCandidateScoreChunkRef,
    AlphaCurrentRuntimeMarker,
    AlphaCurrentRuntimeReceipt,
    AlphaCurrentSafeProjection,
    AlphaCurrentStabilitySurface,
    CurrentFormationScoreChunkRef,
    CurrentFormationScoreSnapshot,
    CurrentRefitStabilityAssessment,
    FrozenComponentScoreSnapshot,
    FrozenFeaturePreparation,
    FrozenScoreObservationSnapshot,
    SectorHistoryRecord,
    seal_current_contract,
)


class AlphaCurrentArtifactReadbackError(AlphaDevelopmentArtifactReadbackError):
    """A current Alpha publication chain is missing, stale, or tampered."""


@dataclass(frozen=True, slots=True)
class PublishedAlphaCurrentRuntime:
    """Retain a read-back current publication and its complete development/decision lineage."""

    marker: AlphaCurrentRuntimeMarker
    marker_ref: str
    surface: AlphaDevelopmentSurfaceManifest
    fold_surfaces: tuple[AlphaDevelopmentFoldSurface, ...]
    candidate_reports: tuple[AlphaCandidateDevelopmentReport, ...]
    candidate_inference_evidence: tuple[AlphaCandidateInferenceEvidence, ...]
    viability: AlphaModelViabilityAssessment
    current_stability: AlphaCurrentStabilitySurface | None
    current_candidate_scores: tuple[AlphaCurrentCandidateScoreChild, ...]
    decision: AlphaModelSelectionDecision | None
    reproducibility: AlphaDecisionReproducibilityReport | None
    current_score: CurrentFormationScoreSnapshot | None


class AlphaCurrentArtifactStore(AlphaDevelopmentArtifactStore):
    """Current publication owner composed over immutable development evidence."""

    _packed_readback_error = AlphaCurrentArtifactReadbackError

    def publish_frozen_observations(
        self,
        source: FrozenPriceVolumeInputs,
        *,
        disposition: Literal["RECORDED_INPUT_QA", "SYNTHETIC_INPUT_QA"],
    ) -> FrozenScoreObservationSnapshot:
        """Persist declared price/volume/context arrays and seal their frozen observation record.

        Packed arrays also retain named formula values and optional reference/population
        declarations. The disposition distinguishes recorded versus synthetic input-QA evidence
        without granting research admission.

        Args:
            source: Dated input axes, classification, source identity and numerical arrays.
            disposition: Declared recorded-input or synthetic-input QA disposition.

        Returns:
            Sealed observation snapshot selecting the packed array payload.

        Raises:
            ValueError: Existing identity-addressed payload or snapshot conflicts with this
                publication.
        """
        payload = BytesIO()
        np.savez(
            payload,
            **{
                name: getattr(source, name)
                for name in (
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "market_context_values",
                    "sector_trend_values",
                )
            },
            **{
                f"formula::{name}": source.formula_values[name]
                for name in sorted(source.formula_values)
            },
            **(
                {"reference_eligible": source.reference_eligible}
                if source.reference_eligible is not None
                else {}
            ),
            **(
                {"nominal_member_count": source.nominal_member_count}
                if source.nominal_member_count is not None
                else {}
            ),
        )
        content_hash = self._publish_packed_bytes(
            category="current/frozen-observation-arrays",
            payload=payload.getvalue(),
        )
        snapshot = seal_current_contract(
            FrozenScoreObservationSnapshot,
            {
                "disposition": disposition,
                "formation_sessions": source.formation_sessions,
                "ordered_listing_ids": source.ordered_listing_ids,
                "sector_by_listing_id": dict(source.sector_by_listing_id),
                **(
                    {"sector_history": record}
                    if (record := SectorHistoryRecord.of(source.sector_by_listing_id)) is not None
                    else {}
                ),
                "array_content_hash": content_hash,
                "source_binding_hash": source.source_binding_hash,
                "reference_eligibility_recorded": source.reference_eligible is not None,
                "nominal_population_recorded": source.nominal_member_count is not None,
                **(
                    {"ordered_formula_ids": tuple(sorted(source.formula_values))}
                    if source.formula_values
                    else {}
                ),
            },
            "snapshot_hash",
        )
        self._publish("frozen-observation-snapshots", snapshot, "snapshot_hash")
        return snapshot

    def load_frozen_observation_snapshot(self, content_hash: str) -> FrozenScoreObservationSnapshot:
        """Load and validate a frozen observation snapshot by its declared identity.

        Args:
            content_hash: Observation snapshot identity selecting its JSON record.

        Returns:
            Validated observation snapshot with declared packed-array and source commitments.
        """
        snapshot, _ = self._load_identity_json_with_identity(
            uri=self.uri("current/frozen-observation-snapshots", content_hash),
            category="current/frozen-observation-snapshots",
            identity_field="snapshot_hash",
        )
        return cast(
            FrozenScoreObservationSnapshot, FrozenScoreObservationSnapshot.model_validate(snapshot)
        )

    def load_frozen_observations(self, content_hash: str) -> FrozenPriceVolumeInputs:
        """Reconstruct immutable input arrays from the declared frozen observation payload.

        Pickle loading is disabled. Exact archive columns and float64/bool/int64 dtypes are checked;
        arrays use immutable byte buffers. Reconstructed source_binding_hash is the observation
        snapshot identity.

        Args:
            content_hash: Frozen observation snapshot identity to read back.

        Returns:
            Validated dated price/volume/context/formula inputs with immutable reconstructed arrays.

        Raises:
            AlphaCurrentArtifactReadbackError: Packed payload, expected archive columns or exact
                dtypes differ.
        """
        snapshot, arrays, _ = self._frozen_observation_source(content_hash)
        return self._frozen_observation_values(snapshot, arrays)

    def verify_frozen_observations(self, content_hash: str) -> None:
        """Verify unchanged observation bytes without retaining arrays only used for admission.

        A cold or invalidated read fully hashes the snapshot and packed payload, and checks
        every archive member, column, dtype and input axis. An opted-in read may retain only
        that complete proof behind leases for both sources.
        Ordinary execution revalidates; no input values or mutable mapping are retained here.
        """
        snapshot = self.load_frozen_observation_snapshot(content_hash)

        def check() -> None:
            current, arrays, identity = self._frozen_observation_source(content_hash)

            def reconcile() -> None:
                self._frozen_observation_values(current, arrays)

            verified_request_proof(
                ("FrozenPriceVolumeInputs.proof.v1", *identity),
                reconcile,
            )

        verified_source_value(
            ("alpha-current-observation-proof", str(self.root), content_hash),
            (
                self._path("current/frozen-observation-snapshots", content_hash, "json"),
                self._path("current/frozen-observation-arrays", snapshot.array_content_hash, "bin"),
            ),
            check,
            nbytes=0,
        )

    def _frozen_observation_source(
        self, content_hash: str
    ) -> tuple[FrozenScoreObservationSnapshot, dict[str, npt.NDArray[Any]], tuple[object, ...]]:
        snapshot, snapshot_file_identity = self._read_identity_json_with_identity(
            FrozenScoreObservationSnapshot,
            category="current/frozen-observation-snapshots",
            content_hash=content_hash,
            identity_field="snapshot_hash",
        )
        arrays, file_identity = self._frozen_arrays_with_identity(
            "frozen-observation-arrays", snapshot.array_content_hash
        )
        identity = (
            "alpha-current",
            str(self.root),
            "frozen-observation-arrays",
            snapshot.snapshot_hash,
            "FrozenPriceVolumeInputs.npz_arrays.v1",
            *snapshot_file_identity,
            snapshot.array_content_hash,
            *file_identity,
        )
        return snapshot, arrays, identity

    @staticmethod
    def _frozen_observation_values(
        snapshot: FrozenScoreObservationSnapshot, arrays: dict[str, npt.NDArray[Any]]
    ) -> FrozenPriceVolumeInputs:
        names = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "market_context_values",
            "sector_trend_values",
        }
        names.update(f"formula::{name}" for name in snapshot.ordered_formula_ids)
        if snapshot.reference_eligibility_recorded:
            names.add("reference_eligible")
        if snapshot.nominal_population_recorded:
            names.add("nominal_member_count")
        if set(arrays) != names:
            raise AlphaCurrentArtifactReadbackError(
                "alpha_research.frozen_observation_columns_invalid"
            )
        for name in names:
            values = arrays[name]
            expected_dtype = np.dtype(
                {"reference_eligible": "bool", "nominal_member_count": "int64"}.get(name, "float64")
            )
            if values.dtype != expected_dtype:
                raise AlphaCurrentArtifactReadbackError(
                    "alpha_research.frozen_observation_dtype_invalid"
                )
        return FrozenPriceVolumeInputs(
            formation_sessions=snapshot.formation_sessions,
            ordered_listing_ids=snapshot.ordered_listing_ids,
            sector_by_listing_id=(
                dict(snapshot.sector_by_listing_id)
                if snapshot.sector_history is None
                else snapshot.sector_history.history(dict(snapshot.sector_by_listing_id))
            ),
            source_binding_hash=snapshot.snapshot_hash,
            formula_values={
                name: arrays.pop(f"formula::{name}") for name in snapshot.ordered_formula_ids
            },
            **arrays,
        )

    def publish_frozen_component_score(self, value: FrozenComponentScoreSnapshot) -> str:
        """Publish the sealed frozen component-score snapshot by identity.

        Args:
            value: Validated frozen component-score snapshot selecting its snapshot_hash
                destination.

        Returns:
            Published immutable record URI.

        Raises:
            ValueError: Existing content at that identity conflicts with the supplied record.
        """
        return self._publish("frozen-component-scores", value, "snapshot_hash")

    def publish_frozen_feature_preparation(
        self,
        *,
        observation: FrozenScoreObservationSnapshot,
        authority_hash: str,
        formation: date,
        surfaces: tuple[HeterogeneousVintageFeatureSurface, ...],
    ) -> FrozenFeaturePreparation:
        """Persist dated vintage feature matrices and seal their inference preparation.

        Args:
            observation: Frozen observation snapshot supplying source/listing authority.
            authority_hash: Exact inference authority identity.
            formation: Formation associated with the prepared vintages.
            surfaces: Nonempty ordered vintage feature surfaces sharing listing and feature axes.

        Returns:
            Sealed preparation binding vintages, source/value identities and the packed feature
            payload.
        """
        payload = BytesIO()
        np.savez(payload, features=np.stack([value.features for value in surfaces]))
        content = self._publish_packed_bytes(
            category="current/frozen-feature-arrays", payload=payload.getvalue()
        )
        result = seal_current_contract(
            FrozenFeaturePreparation,
            {
                "observation_snapshot_hash": observation.snapshot_hash,
                "inference_authority_hash": authority_hash,
                "formation_session": formation,
                "ordered_listing_ids": observation.ordered_listing_ids,
                "ordered_feature_ids": surfaces[0].ordered_feature_ids,
                "vintages": tuple(value.vintage for value in surfaces),
                "source_binding_hashes": tuple(value.source_binding_hash for value in surfaces),
                "feature_values_hashes": tuple(value.feature_values_hash for value in surfaces),
                "array_content_hash": content,
            },
            "preparation_hash",
        )
        self._publish("frozen-feature-preparations", result, "preparation_hash")
        return result

    def load_frozen_feature_preparation(
        self,
        content_hash: str,
    ) -> tuple[FrozenFeaturePreparation, tuple[HeterogeneousVintageFeatureSurface, ...]]:
        """Read back packed vintage features and reconcile every declared value identity.

        Args:
            content_hash: Frozen feature-preparation identity.

        Returns:
            Validated preparation and reconstructed vintage surfaces in declared order.

        Raises:
            AlphaCurrentArtifactReadbackError: Packed axes or reconstructed feature-value identities
                disagree.
        """
        result, _ = self._read_identity_json_with_identity(
            FrozenFeaturePreparation,
            category="current/frozen-feature-preparations",
            content_hash=content_hash,
            identity_field="preparation_hash",
        )

        def build_surfaces() -> tuple[HeterogeneousVintageFeatureSurface, ...]:
            current, current_file_identity = self._read_identity_json_with_identity(
                FrozenFeaturePreparation,
                category="current/frozen-feature-preparations",
                content_hash=content_hash,
                identity_field="preparation_hash",
            )
            payload, file_identity = self._frozen_payload_with_identity(
                "frozen-feature-arrays", current.array_content_hash
            )

            def reconstruct() -> tuple[HeterogeneousVintageFeatureSurface, ...]:
                return self._frozen_feature_surfaces(
                    current, payload, current_file_identity, file_identity
                )

            return verified_request_value(
                (
                    "alpha-current-feature-surfaces",
                    str(self.root),
                    current.preparation_hash,
                    *current_file_identity,
                    current.array_content_hash,
                    *file_identity,
                ),
                reconstruct,
                nbytes=lambda values: sum(surface.features.nbytes for surface in values),
            )

        surfaces = verified_source_value(
            ("alpha-current-feature-surfaces", str(self.root), content_hash),
            (
                self._path("current/frozen-feature-preparations", content_hash, "json"),
                self._path("current/frozen-feature-arrays", result.array_content_hash, "bin"),
            ),
            build_surfaces,
            nbytes=lambda values: sum(surface.features.nbytes for surface in values),
        )
        return result, surfaces

    def _frozen_feature_surfaces(
        self,
        result: FrozenFeaturePreparation,
        payload: bytes,
        result_file_identity: tuple[object, ...],
        file_identity: tuple[object, ...],
    ) -> tuple[HeterogeneousVintageFeatureSurface, ...]:
        arrays = verified_npz_arrays(
            payload,
            identity=(
                "alpha-current",
                str(self.root),
                "frozen-feature-arrays",
                result.preparation_hash,
                "HeterogeneousVintageFeatureSurface.create.v1",
                *result_file_identity,
                result.array_content_hash,
                *file_identity,
            ),
        )
        if set(arrays) != {"features"}:
            raise AlphaCurrentArtifactReadbackError("alpha_research.frozen_feature_columns_invalid")
        values = arrays["features"]
        if values.shape != (
            len(result.vintages),
            len(result.ordered_listing_ids),
            len(result.ordered_feature_ids),
        ):
            raise AlphaCurrentArtifactReadbackError("alpha_research.frozen_feature_axis_invalid")
        found = tuple(
            HeterogeneousVintageFeatureSurface.create(
                vintage=vintage,
                ordered_listing_ids=result.ordered_listing_ids,
                ordered_feature_ids=result.ordered_feature_ids,
                features=values[index],
                source_binding_hash=result.source_binding_hashes[index],
            )
            for index, vintage in enumerate(result.vintages)
        )
        if tuple(value.feature_values_hash for value in found) != result.feature_values_hashes:
            raise AlphaCurrentArtifactReadbackError(
                "alpha_research.frozen_feature_value_identity_invalid"
            )
        return found

    def load_frozen_component_score(self, content_hash: str) -> FrozenComponentScoreSnapshot:
        """Load and validate the exact frozen component-score snapshot.

        Args:
            content_hash: Identity selecting this record category and stored JSON.

        Returns:
            Validated frozen component-score snapshot from the selected stored record.
        """
        return self._load(
            "frozen-component-scores", content_hash, "snapshot_hash", FrozenComponentScoreSnapshot
        )

    def publish_refit_assessment(self, value: CurrentRefitStabilityAssessment) -> str:
        """Publish the sealed current-refit assessment by identity.

        Args:
            value: Validated current-refit assessment selecting its assessment_hash destination.

        Returns:
            Published immutable record URI.

        Raises:
            ValueError: Existing content at that identity conflicts with the supplied record.
        """
        return str(self._publish("refit-assessments", value, "assessment_hash"))

    def load_refit_assessment(self, value: str) -> CurrentRefitStabilityAssessment:
        """Load and validate the exact current-refit assessment.

        Args:
            value: Identity selecting this record category and stored JSON.

        Returns:
            Validated current-refit assessment from the selected stored record.
        """
        return cast(
            CurrentRefitStabilityAssessment,
            CurrentRefitStabilityAssessment.model_validate(
                self._load(
                    "refit-assessments", value, "assessment_hash", CurrentRefitStabilityAssessment
                )
            ),
        )

    def publish_current_stability(self, value: AlphaCurrentStabilitySurface) -> str:
        """Publish the sealed current-stability surface by identity.

        Args:
            value: Validated current-stability surface selecting its surface_hash destination.

        Returns:
            Published immutable record URI.

        Raises:
            ValueError: Existing content at that identity conflicts with the supplied record.
        """
        return str(self._publish("current-stability-surfaces", value, "surface_hash"))

    def load_current_stability(self, value: str) -> AlphaCurrentStabilitySurface:
        """Load and validate the exact current-stability surface.

        Args:
            value: Identity selecting this record category and stored JSON.

        Returns:
            Validated current-stability surface from the selected stored record.
        """
        return cast(
            AlphaCurrentStabilitySurface,
            AlphaCurrentStabilitySurface.model_validate(
                self._load(
                    "current-stability-surfaces",
                    value,
                    "surface_hash",
                    AlphaCurrentStabilitySurface,
                )
            ),
        )

    def publish_current_candidate_score_chunk(
        self,
        table: pa.Table,
        *,
        request_hash: str,
        foundation_hash: str,
        logical_panel_hash: str,
        candidate_id: str,
        estimator_state_hash: str,
        stability_assessment_hash: str,
        formation_session: date,
        ordered_listing_ids: tuple[str, ...],
    ) -> AlphaCurrentCandidateScoreChunkRef:
        """Publish canonical candidate current scores and verify their content readback.

        The writer requires the exact score column inventory and nonempty sorted unique
        formation/listing keys. Content identity binds declared lineage, Arrow schema and rows;
        metadata binds the typed reference.

        Args:
            table: Canonical score rows with the installed column inventory.
            request_hash: Exact numerical request identity.
            foundation_hash: Admitted Foundation identity.
            logical_panel_hash: Logical source Panel identity.
            estimator_state_hash: Retained fitted estimator identity.
            stability_assessment_hash: Admitted refit-stability identity.
            formation_session: Formation associated with the published score rows.
            ordered_listing_ids: Exact ordered listing axis committed by the reference.
            candidate_id: Candidate identity owning these formation scores.

        Returns:
            Chunk reference after persisted Parquet URI/metadata/content verification.

        Raises:
            ValueError: Column inventory or row-key order/uniqueness violates publication.
            AlphaCurrentArtifactReadbackError: Persisted chunk readback differs from its
                declaration.
        """
        if tuple(table.schema.names) != self._CURRENT_SCORE_COLUMNS or table.num_rows < 1:
            raise ValueError("Alpha current candidate score chunk schema is invalid")
        rows = table.to_pylist()
        keys = tuple((row["formation_session"], row["listing_id"]) for row in rows)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("Alpha current candidate score rows are not canonical")
        identity = {
            "request_hash": request_hash,
            "foundation_hash": foundation_hash,
            "logical_panel_hash": logical_panel_hash,
            "candidate_id": candidate_id,
            "estimator_state_hash": estimator_state_hash,
            "stability_assessment_hash": stability_assessment_hash,
            "formation_session": formation_session,
            "ordered_listing_ids_hash": canonical_hash(ordered_listing_ids),
        }
        content_hash = canonical_hash(
            {"identity": identity, "schema": str(table.schema), "rows": rows}
        )
        category = "current-candidate-score-chunks"
        uri = self.uri(f"current/{category}", content_hash)
        provisional = AlphaCurrentCandidateScoreChunkRef.model_construct(
            **identity,
            content_hash=content_hash,
            metadata_hash="0" * 64,
            uri=uri,
            row_count=table.num_rows,
        )
        metadata_hash = canonical_hash(
            provisional.model_dump(mode="json", exclude={"metadata_hash", "uri"})
        )
        reference = AlphaCurrentCandidateScoreChunkRef(
            **provisional.model_dump(mode="python", exclude={"metadata_hash"}),
            metadata_hash=metadata_hash,
        )
        target = self._path(f"current/{category}", content_hash, "parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            b"alphalattice.snapshot_kind": b"AlphaCurrentCandidateScoreChunk",
            b"alphalattice.content_hash": content_hash.encode(),
            b"alphalattice.metadata_hash": metadata_hash.encode(),
        }
        staged = target.with_name(f".{content_hash}.{os.getpid()}.tmp")
        if not target.exists():
            pq.write_table(table.replace_schema_metadata(metadata), staged, compression="zstd")
            os.replace(staged, target)
        staged.unlink(missing_ok=True)
        self.resolve_current_candidate_score_chunk(reference)
        return reference

    def resolve_current_candidate_score_chunk(
        self,
        value: AlphaCurrentCandidateScoreChunkRef,
    ) -> pa.Table:
        """Resolve and verify the exact candidate current-score Parquet content.

        Args:
            value: Declared chunk reference with lineage, URI, content/metadata identities and row
                count.

        Returns:
            Arrow score table with schema metadata removed after declared content checks.

        Raises:
            AlphaCurrentArtifactReadbackError: URI, metadata commitments, columns, row count or
                lineage/schema/row content identity differs.
        """
        category = "current-candidate-score-chunks"
        expected_uri = self.uri(f"current/{category}", value.content_hash)
        if value.uri != expected_uri:
            raise AlphaCurrentArtifactReadbackError("Alpha current candidate score URI changed")
        target = self._path(f"current/{category}", value.content_hash, "parquet")
        parquet = pq.ParquetFile(target)
        metadata = parquet.schema_arrow.metadata or {}
        if (
            metadata.get(b"alphalattice.snapshot_kind") != b"AlphaCurrentCandidateScoreChunk"
            or metadata.get(b"alphalattice.content_hash") != value.content_hash.encode()
            or metadata.get(b"alphalattice.metadata_hash") != value.metadata_hash.encode()
        ):
            raise AlphaCurrentArtifactReadbackError(
                "Alpha current candidate score metadata changed"
            )
        table = pq.read_table(target).replace_schema_metadata(None)
        identity = {
            "request_hash": value.request_hash,
            "foundation_hash": value.foundation_hash,
            "logical_panel_hash": value.logical_panel_hash,
            "candidate_id": value.candidate_id,
            "estimator_state_hash": value.estimator_state_hash,
            "stability_assessment_hash": value.stability_assessment_hash,
            "formation_session": value.formation_session,
            "ordered_listing_ids_hash": value.ordered_listing_ids_hash,
        }
        if (
            tuple(table.schema.names) != self._CURRENT_SCORE_COLUMNS
            or table.num_rows != value.row_count
            or canonical_hash(
                {"identity": identity, "schema": str(table.schema), "rows": table.to_pylist()}
            )
            != value.content_hash
        ):
            raise AlphaCurrentArtifactReadbackError("Alpha current candidate score content changed")
        return table

    def publish_current_candidate_score(self, value: AlphaCurrentCandidateScoreChild) -> str:
        """Publish the sealed current candidate-score child by identity.

        Args:
            value: Validated current candidate-score child selecting its child_hash destination.

        Returns:
            Published immutable record URI.

        Raises:
            ValueError: Existing content at that identity conflicts with the supplied record.
        """
        return str(self._publish("current-candidate-scores", value, "child_hash"))

    def load_current_candidate_score(self, value: str) -> AlphaCurrentCandidateScoreChild:
        """Load a candidate score child and verify its declared Parquet chunk.

        Args:
            value: Exact candidate score-child identity.

        Returns:
            Model-validated child after its chunk URI, metadata and content resolve.

        Raises:
            AlphaCurrentArtifactReadbackError: The child or declared chunk fails readback.
        """
        result = cast(
            AlphaCurrentCandidateScoreChild,
            AlphaCurrentCandidateScoreChild.model_validate(
                self._load(
                    "current-candidate-scores",
                    value,
                    "child_hash",
                    AlphaCurrentCandidateScoreChild,
                )
            ),
        )
        self.resolve_current_candidate_score_chunk(result.chunk)
        return result

    def publish_current_score_chunk(
        self,
        table: pa.Table,
        *,
        request_hash: str,
        foundation_hash: str,
        logical_panel_hash: str,
        selected_candidate_id: str,
        decision_hash: str,
        estimator_state_hash: str,
        stability_assessment_hash: str,
        formation_session: date,
        ordered_listing_ids: tuple[str, ...],
    ) -> CurrentFormationScoreChunkRef:
        """Publish canonical selected current scores and verify their content readback.

        The writer requires the exact score column inventory and nonempty sorted unique
        formation/listing keys. Content identity binds declared lineage, Arrow schema and rows;
        metadata binds the typed reference.

        Args:
            table: Canonical score rows with the installed column inventory.
            request_hash: Exact numerical request identity.
            foundation_hash: Admitted Foundation identity.
            logical_panel_hash: Logical source Panel identity.
            estimator_state_hash: Retained fitted estimator identity.
            stability_assessment_hash: Admitted refit-stability identity.
            formation_session: Formation associated with the published score rows.
            ordered_listing_ids: Exact ordered listing axis committed by the reference.
            selected_candidate_id: Decision-selected candidate identity.
            decision_hash: Exact selection decision identity.

        Returns:
            Chunk reference after persisted Parquet URI/metadata/content verification.

        Raises:
            ValueError: Column inventory or row-key order/uniqueness violates publication.
            AlphaCurrentArtifactReadbackError: Persisted chunk readback differs from its
                declaration.
        """
        if tuple(table.schema.names) != self._CURRENT_SCORE_COLUMNS or table.num_rows < 1:
            raise ValueError("current Alpha score chunk schema is invalid")
        rows = table.to_pylist()
        keys = tuple((row["formation_session"], row["listing_id"]) for row in rows)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("current Alpha score rows are not canonical")
        identity = {
            "request_hash": request_hash,
            "foundation_hash": foundation_hash,
            "logical_panel_hash": logical_panel_hash,
            "selected_candidate_id": selected_candidate_id,
            "decision_hash": decision_hash,
            "estimator_state_hash": estimator_state_hash,
            "stability_assessment_hash": stability_assessment_hash,
            "formation_session": formation_session,
            "ordered_listing_ids_hash": canonical_hash(ordered_listing_ids),
        }
        content_hash = canonical_hash(
            {"identity": identity, "schema": str(table.schema), "rows": rows}
        )
        uri = self.uri("current/current-score-chunks", content_hash)
        provisional = CurrentFormationScoreChunkRef.model_construct(
            **identity,
            content_hash=content_hash,
            metadata_hash="0" * 64,
            uri=uri,
            row_count=table.num_rows,
        )
        metadata_hash = canonical_hash(
            provisional.model_dump(mode="json", exclude={"metadata_hash", "uri"})
        )
        reference = CurrentFormationScoreChunkRef(
            **provisional.model_dump(mode="python", exclude={"metadata_hash"}),
            metadata_hash=metadata_hash,
        )
        target = self._path("current/current-score-chunks", content_hash, "parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            b"alphalattice.snapshot_kind": b"CurrentFormationScoreChunk",
            b"alphalattice.content_hash": content_hash.encode(),
            b"alphalattice.metadata_hash": metadata_hash.encode(),
        }
        staged = target.with_name(f".{content_hash}.{os.getpid()}.tmp")
        if not target.exists():
            pq.write_table(table.replace_schema_metadata(metadata), staged, compression="zstd")
            os.replace(staged, target)
        staged.unlink(missing_ok=True)
        self.resolve_current_score_chunk(reference)
        return reference

    def resolve_current_score_chunk(self, value: CurrentFormationScoreChunkRef) -> pa.Table:
        """Resolve and verify the exact selected current-score Parquet content.

        Args:
            value: Declared chunk reference with lineage, URI, content/metadata identities and row
                count.

        Returns:
            Arrow score table with schema metadata removed after declared content checks.

        Raises:
            AlphaCurrentArtifactReadbackError: URI, metadata commitments, columns, row count or
                lineage/schema/row content identity differs.
        """
        expected_uri = self.uri("current/current-score-chunks", value.content_hash)
        if value.uri != expected_uri:
            raise AlphaCurrentArtifactReadbackError("current Alpha score URI is invalid")
        target = self._path("current/current-score-chunks", value.content_hash, "parquet")
        parquet = pq.ParquetFile(target)
        metadata = parquet.schema_arrow.metadata or {}
        if (
            metadata.get(b"alphalattice.content_hash") != value.content_hash.encode()
            or metadata.get(b"alphalattice.metadata_hash") != value.metadata_hash.encode()
        ):
            raise AlphaCurrentArtifactReadbackError("current Alpha score metadata changed")
        table = pq.read_table(target).replace_schema_metadata(None)
        if (
            tuple(table.schema.names) != self._CURRENT_SCORE_COLUMNS
            or table.num_rows != value.row_count
            or canonical_hash(
                {
                    "identity": {
                        "request_hash": value.request_hash,
                        "foundation_hash": value.foundation_hash,
                        "logical_panel_hash": value.logical_panel_hash,
                        "selected_candidate_id": value.selected_candidate_id,
                        "decision_hash": value.decision_hash,
                        "estimator_state_hash": value.estimator_state_hash,
                        "stability_assessment_hash": value.stability_assessment_hash,
                        "formation_session": value.formation_session,
                        "ordered_listing_ids_hash": value.ordered_listing_ids_hash,
                    },
                    "schema": str(table.schema),
                    "rows": table.to_pylist(),
                }
            )
            != value.content_hash
        ):
            raise AlphaCurrentArtifactReadbackError("current Alpha score content changed")
        return table

    def publish_current_score(self, value: CurrentFormationScoreSnapshot) -> str:
        """Publish the sealed selected current-score snapshot by identity.

        Args:
            value: Validated selected current-score snapshot selecting its snapshot_hash
                destination.

        Returns:
            Published immutable record URI.

        Raises:
            ValueError: Existing content at that identity conflicts with the supplied record.
        """
        return str(self._publish("current-score-snapshots", value, "snapshot_hash"))

    def load_current_score(self, value: str) -> CurrentFormationScoreSnapshot:
        """Load the selected formation score and verify its declared Parquet chunk.

        Args:
            value: Exact selected score-snapshot identity.

        Returns:
            Model-validated snapshot after its chunk URI, metadata and content resolve.

        Raises:
            AlphaCurrentArtifactReadbackError: The snapshot or declared chunk fails readback.
        """
        result = cast(
            CurrentFormationScoreSnapshot,
            CurrentFormationScoreSnapshot.model_validate(
                self._load(
                    "current-score-snapshots",
                    value,
                    "snapshot_hash",
                    CurrentFormationScoreSnapshot,
                )
            ),
        )
        self.resolve_current_score_chunk(result.chunk)
        return result

    def publish_marker(self, value: AlphaCurrentRuntimeMarker) -> str:
        """Publish the sealed current runtime marker by identity.

        Args:
            value: Validated current runtime marker selecting its marker_hash destination.

        Returns:
            Published immutable record URI.

        Raises:
            ValueError: Existing content at that identity conflicts with the supplied record.
        """
        return str(self._publish("markers", value, "marker_hash"))

    def load_marker(self, value: str) -> AlphaCurrentRuntimeMarker:
        """Load and validate the exact current runtime marker.

        Args:
            value: Identity selecting this record category and stored JSON.

        Returns:
            Validated current runtime marker from the selected stored record.
        """
        return cast(
            AlphaCurrentRuntimeMarker,
            AlphaCurrentRuntimeMarker.model_validate(
                self._load("markers", value, "marker_hash", AlphaCurrentRuntimeMarker)
            ),
        )

    def publish_receipt(self, value: AlphaCurrentRuntimeReceipt) -> str:
        """Publish the sealed current runtime receipt by identity.

        Args:
            value: Validated current runtime receipt selecting its receipt_hash destination.

        Returns:
            Published immutable record URI.

        Raises:
            ValueError: Existing content at that identity conflicts with the supplied record.
        """
        return str(self._publish("receipts", value, "receipt_hash"))

    def publish_projection(self, value: AlphaCurrentSafeProjection) -> str:
        """Publish a safe projection and atomically replace the active projection pointer.

        Args:
            value: Sealed current projection selecting its immutable identity destination.

        Returns:
            Immutable projection URI also retained by the active lookup.

        Raises:
            ValueError: Immutable projection publication conflicts with existing content.
        """
        uri = self._publish("safe-projections", value, "projection_hash")
        pointer = self.root / "current" / "active-safe-projection.json"
        self._atomic_write(
            pointer,
            self._json_bytes({"projection_hash": value.projection_hash, "projection_ref": uri}),
        )
        return str(uri)

    def load_projection(self, value: str) -> AlphaCurrentSafeProjection:
        """Load and validate the exact safe current projection.

        Args:
            value: Identity selecting this record category and stored JSON.

        Returns:
            Validated safe current projection from the selected stored record.
        """
        return cast(
            AlphaCurrentSafeProjection,
            AlphaCurrentSafeProjection.model_validate(
                self._load("safe-projections", value, "projection_hash", AlphaCurrentSafeProjection)
            ),
        )

    def load_active_projection(self) -> AlphaCurrentSafeProjection:
        """Read the active projection only through its exact declared hash-to-URI route.

        Returns:
            Model-validated active projection selected by the checked pointer.

        Raises:
            AlphaCurrentArtifactReadbackError: The active pointer is absent or its reference differs
                from the identity-derived URI.
        """
        pointer = self.root / "current" / "active-safe-projection.json"
        if not pointer.is_file():
            raise AlphaCurrentArtifactReadbackError("current Alpha safe projection is unavailable")
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        projection_hash = str(payload.get("projection_hash", ""))
        projection_ref = str(payload.get("projection_ref", ""))
        expected_ref = self.uri("current/safe-projections", projection_hash)
        if projection_ref != expected_ref:
            raise AlphaCurrentArtifactReadbackError("current Alpha projection pointer changed")
        return self.load_projection(projection_hash)

    def find_exact(self, request_hash: str) -> PublishedAlphaCurrentRuntime | None:
        """Find and authoritatively read back a publication for one exact numerical request.

        Args:
            request_hash: Exact request selecting a retained marker.

        Returns:
            Verified publication, or None when no marker is registered for that request.
        """
        marker_hash = self.marker_hash_for_request(request_hash)
        return None if marker_hash is None else self.authoritative_readback(marker_hash)

    def marker_hash_for_request(self, request_hash: str) -> str | None:
        """Resolve an exact marker cheaply; authoritative readback remains separate."""
        marker_root = self.root / "current" / "markers"
        matches: list[str] = []
        for path in sorted(marker_root.glob("*.json")) if marker_root.is_dir() else ():
            marker = self.load_marker(path.stem)
            if marker.request_hash == request_hash:
                matches.append(marker.marker_hash)
        if len(set(matches)) > 1:
            raise AlphaCurrentArtifactReadbackError(
                "multiple current Alpha markers claim a request"
            )
        return matches[0] if matches else None

    @staticmethod
    def _require_binding(condition: bool, detail: str) -> None:
        if not condition:
            raise AlphaCurrentArtifactReadbackError(detail)

    def authoritative_readback(self, marker_hash: str) -> PublishedAlphaCurrentRuntime:
        """Read back the full current publication and reconcile every required parent/child lane.

        Readback checks development folds, candidate report/numerical/score/state/metric lineage,
        viability, optional current qualification, actor decision/proposal/reproducibility and
        selected score availability. Terminal status controls which downstream evidence may be
        exposed; registered non-admission and legacy numerical records follow their explicit
        compatibility rules.

        Args:
            marker_hash: Exact runtime marker selecting the complete retained publication.

        Returns:
            Publication aggregate only after required evidence, row counts/axes and disposition
            rules reconcile.

        Raises:
            AlphaCurrentArtifactReadbackError: Required artifact content or declared
                parent/child/decision/score lineage fails readback.
        """
        marker = self.load_marker(marker_hash)
        fold_surfaces = tuple(
            self.load_fold_surface(value) for value in marker.development_fold_surface_hashes
        )
        candidate_reports = tuple(
            self.load_candidate_report(value) for value in marker.candidate_report_hashes
        )
        candidate_inference_evidence = tuple(
            self.load_candidate_inference_evidence(value)
            for value in marker.candidate_inference_evidence_hashes
        )
        surface = self.load_surface(marker.development_surface_hash)
        viability = self.load_viability(marker.viability_assessment_hash)
        current_stability = (
            self.load_current_stability(marker.current_stability_surface_hash)
            if marker.current_stability_surface_hash
            else None
        )
        current_candidate_scores = tuple(
            self.load_current_candidate_score(value)
            for value in marker.current_candidate_score_child_hashes
        )
        decision = self.load_decision(marker.decision_hash) if marker.decision_hash else None
        reproducibility = (
            self.load_reproducibility(marker.reproducibility_report_hash)
            if marker.reproducibility_report_hash
            else None
        )
        current_score = (
            self.load_current_score(marker.current_score_snapshot_hash)
            if marker.current_score_snapshot_hash
            else None
        )
        proposal = (
            self.load_proposal(decision.authoritative_proposal_hash)
            if decision is not None
            else None
        )
        current_state = (
            self.load_estimator_state(marker.current_estimator_state_hash)
            if marker.current_estimator_state_hash
            else None
        )
        refit_assessment = (
            self.load_refit_assessment(marker.current_refit_assessment_hash)
            if marker.current_refit_assessment_hash
            else None
        )
        fold_surface_by_hash = {value.surface_fold_hash: value for value in fold_surfaces}
        self._require_binding(
            len(fold_surface_by_hash) == len(fold_surfaces),
            "current Alpha fold surface axis is not unique",
        )
        self._require_binding(
            surface.request_hash == marker.request_hash
            and surface.foundation_hash == marker.foundation_hash
            and tuple(value.surface_fold_hash for value in fold_surfaces)
            == marker.development_fold_surface_hashes
            and len(fold_surfaces) == len(surface.fold_commitment_hashes)
            and tuple(value.candidate_id for value in candidate_reports) == surface.candidate_ids
            and tuple(value.candidate_id for value in candidate_inference_evidence)
            == surface.candidate_ids
            and tuple(value.report_hash for value in candidate_reports)
            == marker.candidate_report_hashes
            and tuple(value.evidence_hash for value in candidate_inference_evidence)
            == marker.candidate_inference_evidence_hashes
            and viability.request_hash == marker.request_hash
            and viability.surface_hash == surface.surface_hash,
            "current Alpha parent/child binding changed",
        )
        expected_numerical_surface = seal_current_contract(
            AlphaDevelopmentSurfaceBinding,
            {
                "kind": "AlphaDevelopmentSurfaceBinding",
                "foundation_hash": surface.foundation_hash,
                "logical_panel_hash": surface.logical_panel_hash,
                "logical_semantic_index_hash": surface.logical_semantic_index_hash,
                "causal_outcome_snapshot_hash": surface.causal_outcome_snapshot_hash,
                "listing_set_hash": surface.listing_set_hash,
                "ordered_listing_ids_hash": surface.ordered_listing_ids_hash,
                "ordered_factor_ids_hash": surface.ordered_factor_ids_hash,
                "split_hash": surface.split_hash,
                "fold_commitment_hashes": surface.fold_commitment_hashes,
            },
            "binding_hash",
        )
        for fold_index, fold_surface in enumerate(fold_surfaces):
            self._require_binding(
                fold_surface.request_hash == marker.request_hash
                and fold_surface.development_surface_hash == surface.surface_hash
                and fold_surface.fold_index == fold_index
                and fold_surface.fold_commitment_hash == surface.fold_commitment_hashes[fold_index]
                and fold_surface.validation_chunk.request_hash == marker.request_hash
                and fold_surface.validation_chunk.development_surface_hash == surface.surface_hash
                and fold_surface.validation_chunk.fold_index == fold_index
                and fold_surface.validation_chunk.fold_commitment_hash
                == fold_surface.fold_commitment_hash,
                "current Alpha development fold surface binding changed",
            )

        reports_by_id = {value.candidate_id: value for value in candidate_reports}
        self._require_binding(
            len(reports_by_id) == len(candidate_reports),
            "current Alpha candidate report axis is not unique",
        )
        for candidate_index, (report, inference) in enumerate(
            zip(candidate_reports, candidate_inference_evidence, strict=True)
        ):
            fold_evidence_complete = len(report.fold_evidence_hashes) == len(fold_surfaces)
            registered_non_admission = bool(
                report.status is AlphaCandidateStatus.FAILED
                and not report.fold_evidence_hashes
                and report.failure_codes == ("alpha_research.ols3_required_factors_unavailable",)
            )
            self._require_binding(
                report.request_hash == marker.request_hash
                and report.development_surface_hash == surface.surface_hash
                and report.foundation_hash == marker.foundation_hash
                and report.candidate_card_hash == surface.candidate_card_hashes[candidate_index]
                and report.admitted_fold_count == len(fold_surfaces)
                and (fold_evidence_complete or registered_non_admission)
                and inference.request_hash == marker.request_hash
                and inference.candidate_id == report.candidate_id
                and inference.report_hash == report.report_hash
                and inference.admitted_fold_count == len(fold_surfaces),
                "current Alpha candidate report binding changed",
            )
            folds = tuple(
                self.load_candidate_fold_evidence(value) for value in report.fold_evidence_hashes
            )
            score_hashes: list[str] = []
            estimator_hashes: list[str] = []
            resolved_metrics = []
            successful_fold_count = 0
            scored_row_count = 0
            fold_pairs = (
                () if registered_non_admission else tuple(zip(folds, fold_surfaces, strict=True))
            )
            for fold_index, (fold, fold_surface) in enumerate(fold_pairs):
                self._require_binding(
                    fold.request_hash == marker.request_hash
                    and fold.development_surface_hash == surface.surface_hash
                    and fold.surface_fold_hash == fold_surface.surface_fold_hash
                    and fold.candidate_id == report.candidate_id
                    and fold.candidate_card_hash == report.candidate_card_hash
                    and fold.fold_index == fold_index
                    and fold.fold_commitment_hash == fold_surface.fold_commitment_hash,
                    "current Alpha fold evidence binding changed",
                )
                if isinstance(fold, AlphaCandidateFoldEvidence):
                    numerical = self.load_candidate_numerical_fold_result(
                        fold.numerical_result_hash
                    )
                    self._require_binding(
                        numerical.execution_binding_hash == fold.execution_binding_hash
                        and numerical.development_surface_binding_hash
                        == expected_numerical_surface.binding_hash
                        and numerical.candidate_id == fold.candidate_id
                        and numerical.candidate_card_hash == fold.candidate_card_hash
                        and numerical.fold_index == fold.fold_index
                        and numerical.fold_commitment_hash == fold.fold_commitment_hash
                        and numerical.role is report.role,
                        "current Alpha numerical result parent binding changed",
                    )
                    authority: (
                        AlphaCandidateNumericalFoldResult | LegacyAlphaCandidateFoldEvidence
                    ) = numerical
                elif fold.numerical_result_hash is not None:
                    numerical = self.load_candidate_numerical_fold_result(
                        fold.numerical_result_hash
                    )
                    self._require_binding(
                        numerical.execution_binding_hash == fold.execution_binding_hash
                        and numerical.development_surface_binding_hash
                        == expected_numerical_surface.binding_hash
                        and numerical.candidate_id == fold.candidate_id
                        and numerical.candidate_card_hash == fold.candidate_card_hash
                        and numerical.fold_index == fold.fold_index
                        and numerical.fold_commitment_hash == fold.fold_commitment_hash
                        and fold.restates(numerical),
                        "legacy Alpha numerical result parent binding changed",
                    )
                    authority = numerical
                else:
                    self._require_binding(
                        fold.role is report.role,
                        "legacy Alpha fold role binding changed",
                    )
                    authority = fold
                resolved_metrics.append(authority.metrics)
                if authority.metrics is not None:
                    self._require_binding(
                        authority.metrics.fold_index == fold_index,
                        "current Alpha fold metrics binding changed",
                    )
                if authority.fit_ledger is not None:
                    self._require_binding(
                        authority.fit_ledger.candidate_id == report.candidate_id
                        and authority.fit_ledger.fold_index == fold_index,
                        "current Alpha fit ledger binding changed",
                    )
                if authority.status is AlphaCandidateStatus.SUCCEEDED:
                    successful_fold_count += 1
                    assert authority.score_chunk is not None
                    if isinstance(authority.score_chunk, AlphaNumericalDevelopmentScoreChunkRef):
                        self._require_binding(
                            authority.score_chunk.candidate_id == report.candidate_id
                            and authority.score_chunk.candidate_card_hash
                            == report.candidate_card_hash
                            and authority.score_chunk.fold_index == fold_index
                            and authority.score_chunk.fold_commitment_hash
                            == fold_surface.fold_commitment_hash
                            and authority.score_chunk.row_count
                            == fold_surface.validation_chunk.row_count,
                            "current Alpha numerical score/validation surface rows changed",
                        )
                        score_table = self.resolve_numerical_score_chunk(authority.score_chunk)
                    else:
                        self._require_binding(
                            authority.score_chunk.request_hash == marker.request_hash
                            and authority.score_chunk.development_surface_hash
                            == surface.surface_hash
                            and authority.score_chunk.candidate_id == report.candidate_id
                            and authority.score_chunk.candidate_card_hash
                            == report.candidate_card_hash
                            and authority.score_chunk.fold_index == fold_index
                            and authority.score_chunk.fold_commitment_hash
                            == fold_surface.fold_commitment_hash
                            and authority.score_chunk.row_count
                            == fold_surface.validation_chunk.row_count,
                            "current Alpha score/validation surface rows changed",
                        )
                        score_table = self.resolve_candidate_score_chunk(authority.score_chunk)
                    scored_row_count += sum(
                        str(value) == "SCORED" for value in score_table["availability"].to_pylist()
                    )
                    score_hashes.append(authority.score_chunk.content_hash)
                if authority.estimator_state_hash is not None:
                    state = self.load_any_development_estimator_state(
                        authority.estimator_state_hash
                    )
                    if isinstance(state, AlphaDevelopmentEstimatorState):
                        self._require_binding(
                            state.candidate_id == report.candidate_id
                            and state.candidate_card_hash == report.candidate_card_hash
                            and state.fold_index == fold_index
                            and state.fold_commitment_hash == fold_surface.fold_commitment_hash
                            and canonical_hash(state.ordered_factor_ids)
                            == surface.ordered_factor_ids_hash,
                            "current Alpha numerical estimator binding changed",
                        )
                    else:
                        self._require_binding(
                            state.request_hash == marker.request_hash
                            and state.candidate_id == report.candidate_id
                            and state.scope == "DEVELOPMENT_FOLD"
                            and state.fold_index == fold_index
                            and canonical_hash(state.ordered_factor_ids)
                            == surface.ordered_factor_ids_hash,
                            "current Alpha development estimator binding changed",
                        )
                    estimator_hashes.append(state.state_hash)
            self._require_binding(
                tuple(score_hashes) == inference.score_chunk_hashes
                and tuple(estimator_hashes) == report.estimator_state_hashes
                and tuple(estimator_hashes) == inference.estimator_state_hashes
                and successful_fold_count == inference.successful_fold_count
                and scored_row_count == inference.scored_row_count,
                "current Alpha candidate inference lineage changed",
            )
            if report.metrics is not None:
                self._require_binding(
                    report.metrics.candidate_id == report.candidate_id
                    and tuple(resolved_metrics) == report.metrics.fold_metrics
                    and report.metrics.scored_row_count == inference.scored_row_count
                    and report.metrics.common_surface_row_count
                    == inference.common_surface_row_count,
                    "current Alpha candidate metrics lineage changed",
                )
            else:
                self._require_binding(
                    inference.common_surface_row_count == 0,
                    "failed Alpha candidate exposes successful inference evidence",
                )

        regularized_ids = tuple(
            value.candidate_id
            for value in candidate_reports
            if value.role in {AlphaCandidateRole.REGULARIZED_ALPHA, AlphaCandidateRole.MODEL_ALPHA}
        )
        self._require_binding(
            tuple(value.candidate_id for value in viability.candidates) == regularized_ids
            and viability.benchmark_candidate_id in reports_by_id
            and reports_by_id[viability.benchmark_candidate_id].role
            is AlphaCandidateRole.BENCHMARK,
            "current Alpha viability candidate surface changed",
        )

        if current_stability is not None:
            qualifications_by_id = {
                value.candidate_id: value for value in current_stability.candidates
            }
            viability_by_id = {value.candidate_id: value for value in viability.candidates}
            scores_by_id = {value.candidate_id: value for value in current_candidate_scores}
            self._require_binding(
                current_stability.request_hash == marker.request_hash
                and current_stability.viability_assessment_hash == viability.assessment_hash
                and tuple(qualifications_by_id) == viability.admissible_candidate_ids
                and tuple(scores_by_id) == current_stability.dual_qualified_candidate_ids,
                "current Alpha stability surface binding changed",
            )
            for candidate_id, qualification in qualifications_by_id.items():
                state = self.load_estimator_state(qualification.current_state_hash)
                assessment = self.load_refit_assessment(qualification.stability_assessment_hash)
                child = scores_by_id.get(candidate_id)
                current_sessions = tuple(
                    value.formation_session for value in state.validation_session_statistics
                )
                child_rows_valid = True
                if child is not None:
                    child_table = self.resolve_current_candidate_score_chunk(child.chunk)
                    child_rows_valid = (
                        tuple(child_table["formation_session"].to_pylist())
                        == (child.formation_session,) * len(child.ordered_listing_ids)
                        and tuple(str(value) for value in child_table["listing_id"].to_pylist())
                        == child.ordered_listing_ids
                    )
                self._require_binding(
                    qualification.development_candidate_hash
                    == viability_by_id[candidate_id].candidate_hash
                    and state.request_hash == marker.request_hash
                    and state.candidate_id == candidate_id
                    and state.scope == "CURRENT_REFIT"
                    and canonical_hash(state.ordered_factor_ids) == surface.ordered_factor_ids_hash
                    and current_sessions == (current_stability.formation_session,)
                    and assessment.request_hash == marker.request_hash
                    and assessment.selected_candidate_id == candidate_id
                    and assessment.current_state_hash == state.state_hash
                    and assessment.development_state_hashes
                    == reports_by_id[candidate_id].estimator_state_hashes
                    and assessment.passed == qualification.stable
                    and (
                        (child is None and not qualification.stable)
                        or (
                            child is not None
                            and child.child_hash == qualification.current_score_child_hash
                            and child.request_hash == marker.request_hash
                            and child.foundation_hash == marker.foundation_hash
                            and child.logical_panel_hash == surface.logical_panel_hash
                            and child.estimator_state_hash == state.state_hash
                            and child.stability_assessment_hash == assessment.assessment_hash
                            and child.formation_session == current_stability.formation_session
                            and child.training_cutoff == current_stability.training_cutoff
                            and canonical_hash(child.ordered_listing_ids)
                            == surface.ordered_listing_ids_hash
                            and child_rows_valid
                        )
                    ),
                    "current Alpha candidate stability lineage changed",
                )
        else:
            self._require_binding(
                not current_candidate_scores,
                "legacy Alpha runtime exposes unbound current candidate scores",
            )

        if decision is None:
            self._require_binding(
                proposal is None and reproducibility is None,
                "current Alpha decision sidecars are incomplete",
            )
        else:
            assert proposal is not None
            self._require_binding(
                decision.request_hash == marker.request_hash
                and decision.viability_assessment_hash == viability.assessment_hash
                and proposal.proposal_hash == decision.authoritative_proposal_hash
                and proposal.viability_assessment_hash == viability.assessment_hash
                and proposal.action is decision.action
                and proposal.selected_candidate_id == decision.selected_candidate_id
                and proposal.reason_code == decision.reason_code
                and proposal.observed_candidate_ids
                == (
                    current_stability.dual_qualified_candidate_ids
                    if current_stability is not None
                    else tuple(value.candidate_id for value in viability.candidates)
                )
                and reproducibility is not None
                and reproducibility.decision_hash == decision.decision_hash
                and reproducibility.viability_assessment_hash == viability.assessment_hash
                and reproducibility.samples[0].proposal_hash == proposal.proposal_hash
                and reproducibility.samples[0].decision_surface_hash
                == canonical_hash(
                    proposal.model_dump(mode="json", exclude={"invocation_token", "proposal_hash"})
                ),
                "current Alpha decision/proposal lineage changed",
            )
            if decision.selected_candidate_id is not None:
                self._require_binding(
                    decision.selected_candidate_id
                    in (
                        current_stability.dual_qualified_candidate_ids
                        if current_stability is not None
                        else viability.admissible_candidate_ids
                    ),
                    "current Alpha decision selected an inadmissible candidate",
                )

        self._require_binding(
            (current_state is None) == (refit_assessment is None),
            "current Alpha refit state/assessment pair is incomplete",
        )
        if current_state is not None and refit_assessment is not None:
            self._require_binding(
                decision is not None
                and decision.selected_candidate_id is not None
                and current_state.request_hash == marker.request_hash
                and current_state.candidate_id == decision.selected_candidate_id
                and current_state.scope == "CURRENT_REFIT"
                and current_state.fold_index is None
                and canonical_hash(current_state.ordered_factor_ids)
                == surface.ordered_factor_ids_hash
                and refit_assessment.request_hash == marker.request_hash
                and refit_assessment.selected_candidate_id == decision.selected_candidate_id
                and refit_assessment.current_state_hash == current_state.state_hash
                and refit_assessment.development_state_hashes
                == reports_by_id[decision.selected_candidate_id].estimator_state_hashes,
                "current Alpha refit lineage changed",
            )

        if current_score is not None:
            score_table = self.resolve_current_score_chunk(current_score.chunk)
            score_listing_ids = tuple(str(value) for value in score_table["listing_id"].to_pylist())
            score_sessions = tuple(score_table["formation_session"].to_pylist())
            score_availability = tuple(
                str(value) for value in score_table["availability"].to_pylist()
            )
            availability_counts = {
                key: score_availability.count(key) for key in current_score.availability_counts
            }
            self._require_binding(
                decision is not None
                and decision.selected_candidate_id is not None
                and current_state is not None
                and refit_assessment is not None
                and refit_assessment.passed
                and current_score.request_hash == marker.request_hash
                and current_score.foundation_hash == marker.foundation_hash
                and current_score.logical_panel_hash == surface.logical_panel_hash
                and current_score.selected_candidate_id == decision.selected_candidate_id
                and current_score.decision_hash == decision.decision_hash
                and current_score.estimator_state_hash == current_state.state_hash
                and current_score.stability_assessment_hash == refit_assessment.assessment_hash
                and canonical_hash(current_score.ordered_listing_ids)
                == surface.ordered_listing_ids_hash
                and score_listing_ids == current_score.ordered_listing_ids
                and all(value == current_score.formation_session for value in score_sessions)
                and availability_counts == current_score.availability_counts,
                "current Alpha score lineage changed",
            )

        if marker.disposition == "CURRENT_SCORE_PUBLISHED":
            self._require_binding(
                current_score is not None,
                "current Alpha ready marker lacks a verified score",
            )
        elif marker.disposition == "NO_ADMISSIBLE_ALPHA_MODEL":
            self._require_binding(
                viability.disposition == "NO_ADMISSIBLE_ALPHA_MODEL"
                and decision is None
                and current_state is None
                and current_score is None,
                "current Alpha no-model marker contains downstream decisions",
            )
        elif marker.disposition == "NO_STABLE_CURRENT_ALPHA_MODEL":
            self._require_binding(
                current_stability is not None
                and not current_stability.dual_qualified_candidate_ids
                and decision is None
                and current_score is None,
                "current Alpha no-stable-model marker contains downstream decisions",
            )
        else:
            self._require_binding(
                current_score is None,
                "blocked current Alpha marker exposes a score",
            )
        return PublishedAlphaCurrentRuntime(
            marker=marker,
            marker_ref=self.uri("current/markers", marker.marker_hash),
            surface=surface,
            fold_surfaces=fold_surfaces,
            candidate_reports=candidate_reports,
            candidate_inference_evidence=candidate_inference_evidence,
            viability=viability,
            current_stability=current_stability,
            current_candidate_scores=current_candidate_scores,
            decision=decision,
            reproducibility=reproducibility,
            current_score=current_score,
        )


def build_current_candidate_score_child(
    *,
    request_hash: str,
    foundation_hash: str,
    logical_panel_hash: str,
    candidate_id: str,
    estimator_state: AlphaEstimatorState,
    assessment: CurrentRefitStabilityAssessment,
    formation_session: date,
    training_cutoff: date,
    ordered_listing_ids: tuple[str, ...],
    table: pa.Table,
    store: AlphaCurrentArtifactStore,
) -> AlphaCurrentCandidateScoreChild:
    """Publish a candidate formation chunk and seal its exact score-child lineage.

    Args:
        request_hash: Exact request identity.
        foundation_hash: Admitted Foundation identity.
        logical_panel_hash: Logical source Panel identity.
        candidate_id: Candidate owning these scores.
        estimator_state: Retained current estimator whose state_hash is bound.
        assessment: Refit-stability assessment whose identity is bound.
        formation_session: Published score formation.
        training_cutoff: Causal fitted-history cutoff no later than formation.
        ordered_listing_ids: Exact listing axis.
        table: Canonical score table to publish and verify.
        store: Current artifact publication/readback owner.

    Returns:
        Sealed current candidate-score child with its persisted chunk reference.

    Raises:
        ValueError: Score rows or sealed child consistency violate the declared boundary.
        AlphaCurrentArtifactReadbackError: Persisted score chunk fails verification.
    """
    chunk = store.publish_current_candidate_score_chunk(
        table,
        request_hash=request_hash,
        foundation_hash=foundation_hash,
        logical_panel_hash=logical_panel_hash,
        candidate_id=candidate_id,
        estimator_state_hash=estimator_state.state_hash,
        stability_assessment_hash=assessment.assessment_hash,
        formation_session=formation_session,
        ordered_listing_ids=ordered_listing_ids,
    )
    return seal_current_contract(
        AlphaCurrentCandidateScoreChild,
        {
            "kind": "AlphaCurrentCandidateScoreChild",
            "request_hash": request_hash,
            "foundation_hash": foundation_hash,
            "logical_panel_hash": logical_panel_hash,
            "candidate_id": candidate_id,
            "estimator_state_hash": estimator_state.state_hash,
            "stability_assessment_hash": assessment.assessment_hash,
            "formation_session": formation_session,
            "training_cutoff": training_cutoff,
            "ordered_listing_ids": ordered_listing_ids,
            "chunk": chunk,
        },
        "child_hash",
    )


__all__ = [
    "AlphaCurrentArtifactReadbackError",
    "AlphaCurrentArtifactStore",
    "PublishedAlphaCurrentRuntime",
    "build_current_candidate_score_child",
]
