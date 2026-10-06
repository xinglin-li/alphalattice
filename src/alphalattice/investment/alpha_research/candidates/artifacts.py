"""Content-addressed store of Alpha's candidate registry and the qualifications that seal it."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypeVar, cast
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from alphalattice.control.observation_runtime.contracts import (
    ObservationAuthorityReadback,
    ObservationAuthorityReference,
    build_observation_authority_readback,
    build_observation_authority_reference,
)
from alphalattice.investment.sector_research.models.legacy_excess_ema import (
    SectorEmaAlphaManifest,
    SectorEmaAlphaSurface,
    sector_ema_table_content_hash,
)

from .contracts import (
    AlphaCandidateRegistrySnapshot,
    AlphaGoalProgress,
    AlphaGoalResearchMarker,
    AlphaGoalResearchSafeProjection,
    AlphaOosEvidenceAssessment,
    AlphaQualificationSnapshot,
    AlphaResearchProgram,
    AlphaResearchScientificStop,
    CurrentAlphaCandidateSetSnapshot,
)
from .runtime_contracts import (
    AlphaGoalTaskBoard,
)

ContractT = TypeVar("ContractT", bound=BaseModel)


class AlphaGoalResearchArtifactStore:
    """Marker-last store; it owns bytes but never numerical or research decisions."""

    authority_owner_kind = "ALPHA_GOAL_RESEARCH_ARTIFACT_STORE"

    def __init__(self, artifact_root: Path) -> None:
        """Bind goal-driven Alpha artifacts to the caller artifact root.

        Args:
            artifact_root: Caller root containing alpha-research/goal-driven storage.
        """
        self.root = artifact_root / "alpha-research" / "goal-driven"

    def publish_program(self, value: AlphaResearchProgram) -> str:
        """Publish the sealed research program with exact replay admission.

        Args:
            value: Validated research program selecting its identity-addressed JSON destination.

        Returns:
            Local file URI of the published immutable record.

        Raises:
            ValueError: Existing JSON at that identity differs from the declared payload.
        """
        return self._publish("programs", value.program_hash, value)

    def publish_sector_ema_surface(self, value: SectorEmaAlphaSurface) -> str:
        """Publish the compact downstream sector component before terminal linkage."""
        manifest = SectorEmaAlphaManifest.model_validate(value.manifest)
        table = value.table.combine_chunks()
        if sector_ema_table_content_hash(table) != manifest.table_content_hash:
            raise ValueError("alpha_research.sector_ema_table_hash_mismatch")
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink, compression="zstd")
        target = self.root / "sector-ema-surfaces" / f"{manifest.manifest_hash}.parquet"
        self._atomic_bytes(target, sink.getvalue().to_pybytes())
        durable = pq.read_table(target).combine_chunks()
        if sector_ema_table_content_hash(durable) != manifest.table_content_hash:
            raise ValueError("alpha_research.sector_ema_surface_readback_mismatch")
        return self._publish("sector-ema-manifests", manifest.manifest_hash, manifest)

    def load_sector_ema_manifest(self, value_hash: str) -> SectorEmaAlphaManifest:
        """Load the Sector EMA manifest and reconcile its retained Parquet surface.

        Args:
            value_hash: Identity selecting the manifest and Sector EMA table.

        Returns:
            Validated manifest after table identity, row-count and sector-axis checks.

        Raises:
            ValueError: Table content identity, session-by-sector row count or sector identifiers
                disagree.
        """
        manifest = self._load("sector-ema-manifests", value_hash, SectorEmaAlphaManifest)
        table_path = self.root / "sector-ema-surfaces" / f"{value_hash}.parquet"
        table = pq.read_table(table_path).combine_chunks()
        if (
            sector_ema_table_content_hash(table) != manifest.table_content_hash
            or table.num_rows != manifest.session_count * len(manifest.sector_ids)
            or tuple(sorted(set(str(value) for value in table["sector_id"].to_pylist())))
            != manifest.sector_ids
        ):
            raise ValueError("alpha_research.sector_ema_surface_readback_mismatch")
        return manifest

    def publish_registry(self, value: AlphaCandidateRegistrySnapshot) -> str:
        """Publish the sealed candidate registry with exact replay admission.

        Args:
            value: Validated candidate registry selecting its identity-addressed JSON destination.

        Returns:
            Local file URI of the published immutable record.

        Raises:
            ValueError: Existing JSON at that identity differs from the declared payload.
        """
        return self._publish("candidate-registries", value.registry_hash, value)

    def publish_qualification(self, value: AlphaQualificationSnapshot) -> str:
        """Publish the sealed qualification snapshot with exact replay admission.

        Args:
            value: Validated qualification snapshot selecting its identity-addressed JSON
                destination.

        Returns:
            Local file URI of the published immutable record.

        Raises:
            ValueError: Existing JSON at that identity differs from the declared payload.
        """
        return self._publish("qualifications", value.qualification_hash, value)

    def publish_oos_evidence(self, value: AlphaOosEvidenceAssessment) -> str:
        """Publish the sealed out-of-sample assessment with exact replay admission.

        Args:
            value: Validated out-of-sample assessment selecting its identity-addressed JSON
                destination.

        Returns:
            Local file URI of the published immutable record.

        Raises:
            ValueError: Existing JSON at that identity differs from the declared payload.
        """
        return self._publish("oos-evidence", value.assessment_hash, value)

    def publish_goal_progress(self, value: AlphaGoalProgress) -> str:
        """Publish the sealed bounded goal progress with exact replay admission.

        Args:
            value: Validated bounded goal progress selecting its identity-addressed JSON
                destination.

        Returns:
            Local file URI of the published immutable record.

        Raises:
            ValueError: Existing JSON at that identity differs from the declared payload.
        """
        return self._publish("goal-progress", value.progress_hash, value)

    def publish_candidate_set(self, value: CurrentAlphaCandidateSetSnapshot) -> str:
        """Publish the sealed current candidate set with exact replay admission.

        Args:
            value: Validated current candidate set selecting its identity-addressed JSON
                destination.

        Returns:
            Local file URI of the published immutable record.

        Raises:
            ValueError: Existing JSON at that identity differs from the declared payload.
        """
        return self._publish("candidate-sets", value.snapshot_hash, value)

    def publish_scientific_stop(self, value: AlphaResearchScientificStop) -> str:
        """Publish the sealed scientific stop with exact replay admission.

        Args:
            value: Validated scientific stop selecting its identity-addressed JSON destination.

        Returns:
            Local file URI of the published immutable record.

        Raises:
            ValueError: Existing JSON at that identity differs from the declared payload.
        """
        return self._publish("scientific-stops", value.stop_hash, value)

    def publish_projection(self, value: AlphaGoalResearchSafeProjection) -> str:
        """Publish a safe projection and its exact marker-to-projection lookup.

        Args:
            value: Sealed projection binding a goal terminal marker.

        Returns:
            Immutable projection file URI.

        Raises:
            ValueError: Existing projection content or marker lookup conflicts with this
                publication.
        """
        uri = self._publish("safe-projections", value.projection_hash, value)
        pointer = self.root / "projection-by-marker" / f"{value.marker_hash}.json"
        payload = {
            "marker_hash": value.marker_hash,
            "projection_hash": value.projection_hash,
        }
        self._publish_pointer(pointer, payload, "alpha_research.projection_marker_reused")
        return uri

    def find_projection_for_marker(
        self, marker_hash: str
    ) -> AlphaGoalResearchSafeProjection | None:
        """Find a marker projection while treating only a missing file as absence.

        Args:
            marker_hash: Terminal marker selecting the projection lookup.

        Returns:
            Validated projection, or None when its lookup/record is absent.
        """
        try:
            return self.load_projection_for_marker(marker_hash)
        except FileNotFoundError:
            return None

    def publish_terminal_marker(
        self,
        value: AlphaGoalResearchMarker,
        *,
        expected_active_marker_hash: str | None = None,
    ) -> str:
        """Read back terminal children and publish the active marker with compare-and-swap.

        The immutable marker is retained before active-pointer replacement. Failed active
        publication/readback restores the previous pointer bytes or removes a newly created pointer;
        immutable children are retained.

        Args:
            value: Sealed terminal marker whose required child records must read back.
            expected_active_marker_hash: Expected preceding active identity, or None for no active
                marker.

        Returns:
            Immutable terminal-marker file URI after active readback.

        Raises:
            ValueError: Terminal child evidence, expected active identity or active-marker readback
                is inconsistent.
        """
        self._readback_terminal_children(value)
        uri = self._publish("markers", value.marker_hash, value)
        pointer = self.root / "active.json"
        current = self.find_active_marker()
        current_hash = current.marker_hash if current is not None else None
        if current_hash != expected_active_marker_hash:
            raise ValueError("alpha_research.active_marker_compare_and_swap_failed")
        previous = pointer.read_bytes() if pointer.exists() else None
        try:
            self._atomic_json(
                pointer,
                {"marker_hash": value.marker_hash, "program_hash": value.program_hash},
            )
            if self.load_active_marker() != value:
                raise ValueError("alpha_research.goal_marker_readback_failed")
        except Exception:
            if previous is None:
                pointer.unlink(missing_ok=True)
            else:
                self._atomic_bytes(pointer, previous)
            raise
        return uri

    def load_registry(self, value_hash: str) -> AlphaCandidateRegistrySnapshot:
        """Load and model-validate the stored candidate registry.

        Args:
            value_hash: Identity selecting this record category and JSON filename.

        Returns:
            Validated candidate registry from stored JSON.

        Raises:
            FileNotFoundError: The selected record is absent.
            pydantic.ValidationError: Stored fields or self identity violate the model.
        """
        return self._load("candidate-registries", value_hash, AlphaCandidateRegistrySnapshot)

    def load_board(self, value_hash: str) -> AlphaGoalTaskBoard:
        """Load and model-validate the stored goal task board.

        Args:
            value_hash: Identity selecting this record category and JSON filename.

        Returns:
            Validated goal task board from stored JSON.

        Raises:
            FileNotFoundError: The selected record is absent.
            pydantic.ValidationError: Stored fields or self identity violate the model.
        """
        return self._load("task-boards", value_hash, AlphaGoalTaskBoard)

    def load_program(self, value_hash: str) -> AlphaResearchProgram:
        """Load and model-validate the stored research program.

        Args:
            value_hash: Identity selecting this record category and JSON filename.

        Returns:
            Validated research program from stored JSON.

        Raises:
            FileNotFoundError: The selected record is absent.
            pydantic.ValidationError: Stored fields or self identity violate the model.
        """
        return self._load("programs", value_hash, AlphaResearchProgram)

    def load_qualification(self, value_hash: str) -> AlphaQualificationSnapshot:
        """Load and model-validate the stored qualification snapshot.

        Args:
            value_hash: Identity selecting this record category and JSON filename.

        Returns:
            Validated qualification snapshot from stored JSON.

        Raises:
            FileNotFoundError: The selected record is absent.
            pydantic.ValidationError: Stored fields or self identity violate the model.
        """
        return self._load("qualifications", value_hash, AlphaQualificationSnapshot)

    def load_oos_evidence(self, value_hash: str) -> AlphaOosEvidenceAssessment:
        """Load and model-validate the stored out-of-sample assessment.

        Args:
            value_hash: Identity selecting this record category and JSON filename.

        Returns:
            Validated out-of-sample assessment from stored JSON.

        Raises:
            FileNotFoundError: The selected record is absent.
            pydantic.ValidationError: Stored fields or self identity violate the model.
        """
        return self._load("oos-evidence", value_hash, AlphaOosEvidenceAssessment)

    def load_goal_progress(self, value_hash: str) -> AlphaGoalProgress:
        """Load and model-validate the stored bounded goal progress.

        Args:
            value_hash: Identity selecting this record category and JSON filename.

        Returns:
            Validated bounded goal progress from stored JSON.

        Raises:
            FileNotFoundError: The selected record is absent.
            pydantic.ValidationError: Stored fields or self identity violate the model.
        """
        return self._load("goal-progress", value_hash, AlphaGoalProgress)

    def load_projection(self, value_hash: str) -> AlphaGoalResearchSafeProjection:
        """Load and model-validate the stored safe projection.

        Args:
            value_hash: Identity selecting this record category and JSON filename.

        Returns:
            Validated safe projection from stored JSON.

        Raises:
            FileNotFoundError: The selected record is absent.
            pydantic.ValidationError: Stored fields or self identity violate the model.
        """
        return self._load("safe-projections", value_hash, AlphaGoalResearchSafeProjection)

    def load_projection_for_marker(self, marker_hash: str) -> AlphaGoalResearchSafeProjection:
        """Resolve an exact marker lookup and load its model-validated safe projection.

        Args:
            marker_hash: Terminal marker identity that the lookup must name.

        Returns:
            Stored projection selected by the verified marker lookup.

        Raises:
            ValueError: The lookup marker_hash differs from the requested marker.
        """
        payload = self._read_json(self.root / "projection-by-marker" / f"{marker_hash}.json")
        if payload.get("marker_hash") != marker_hash:
            raise ValueError("alpha_research.projection_pointer_tampered")
        return self.load_projection(str(payload["projection_hash"]))

    def load_candidate_set(self, value_hash: str) -> CurrentAlphaCandidateSetSnapshot:
        """Load and model-validate the stored current candidate set.

        Args:
            value_hash: Identity selecting this record category and JSON filename.

        Returns:
            Validated current candidate set from stored JSON.

        Raises:
            FileNotFoundError: The selected record is absent.
            pydantic.ValidationError: Stored fields or self identity violate the model.
        """
        return self._load("candidate-sets", value_hash, CurrentAlphaCandidateSetSnapshot)

    def load_scientific_stop(self, value_hash: str) -> AlphaResearchScientificStop:
        """Load and model-validate the stored scientific stop.

        Args:
            value_hash: Identity selecting this record category and JSON filename.

        Returns:
            Validated scientific stop from stored JSON.

        Raises:
            FileNotFoundError: The selected record is absent.
            pydantic.ValidationError: Stored fields or self identity violate the model.
        """
        return self._load("scientific-stops", value_hash, AlphaResearchScientificStop)

    def load_marker(self, marker_hash: str) -> AlphaGoalResearchMarker:
        """One sealed marker by its hash, active or superseded."""
        return self._load("markers", marker_hash, AlphaGoalResearchMarker)

    def load_active_marker(self) -> AlphaGoalResearchMarker:
        """Load the model-validated terminal marker selected by active.json.

        Returns:
            Stored active terminal marker.

        Raises:
            FileNotFoundError: The active pointer or selected marker file is absent.
        """
        pointer = self._read_json(self.root / "active.json")
        return self._load("markers", str(pointer["marker_hash"]), AlphaGoalResearchMarker)

    def validate_terminal_lineage(self, marker: AlphaGoalResearchMarker) -> None:
        """Read every terminal child and verify its exact Program lineage."""
        self._readback_terminal_children(marker)
        program = self.load_program(marker.program_hash)
        board = None if marker.board_hash is None else self.load_board(marker.board_hash)
        registry = self.load_registry(marker.registry_hash)
        qualification = self.load_qualification(marker.qualification_hash)
        progress = self.load_goal_progress(marker.goal_progress_hash)
        projection = self.load_projection_for_marker(marker.marker_hash)
        if (
            (board is not None and board.research_goal_hash != program.research_goal_hash)
            or progress.criteria_hash != program.goal_criteria_hash
            or progress.registry_hash != registry.registry_hash
            or progress.qualification_hash != qualification.qualification_hash
            or projection.program_hash != program.program_hash
            or projection.marker_hash != marker.marker_hash
            or projection.status != marker.disposition
        ):
            raise ValueError("alpha_research.goal_terminal_lineage_mismatch")

    def find_active_marker(self) -> AlphaGoalResearchMarker | None:
        """Find the active terminal marker when its pointer exists.

        Returns:
            Model-validated active marker, or None when active.json is absent.
        """
        if not (self.root / "active.json").is_file():
            return None
        return self.load_active_marker()

    def read_sealed_marker(self, marker_hash: str) -> AlphaGoalResearchMarker:
        """Read one marker by exact handle and verify its complete terminal lineage.

        The active pointer is never opened.  A frozen consumer that recorded
        ``marker_hash`` keeps reading after Alpha activates a newer marker.
        """
        if not (self.root / "markers" / f"{marker_hash}.json").is_file():
            raise FileNotFoundError("alpha_research.sealed_marker_missing")
        marker = self._load("markers", marker_hash, AlphaGoalResearchMarker)
        if marker.marker_hash != marker_hash:
            raise ValueError("alpha_research.sealed_marker_tampered")
        self.validate_terminal_lineage(marker)
        return marker

    def observation_authority_reference(
        self, *, marker_hash: str, task_id: str, run_id: str
    ) -> ObservationAuthorityReference:
        """Build and read back a goal-marker observation authority reference.

        Args:
            marker_hash: Exact terminal marker record identity.
            task_id: Observing task identifier bound into the reference.
            run_id: Observing run identifier bound into the reference.

        Returns:
            Observation reference checked against the terminal marker and required child evidence.

        Raises:
            ValueError: Declared goal-marker authority or terminal child readback fails.
        """
        marker = self._load("markers", marker_hash, AlphaGoalResearchMarker)
        reference = build_observation_authority_reference(
            owner_kind=self.authority_owner_kind,
            record_kind="AlphaGoalResearchMarker",
            record_hash=marker.marker_hash,
            task_id=task_id,
            run_id=run_id,
            subject_id=marker.program_hash,
        )
        self.read_observation_authority(reference)
        return reference

    def read_observation_authority(
        self, reference: ObservationAuthorityReference
    ) -> ObservationAuthorityReadback:
        """Read exact goal-marker authority and expose bounded publication claims.

        Args:
            reference: Owner/kind/record/subject reference selecting this store terminal marker.

        Returns:
            Authority readback with counts, result identities, disposition and declared Risk
            admission.

        Raises:
            ValueError: Owner/kind/subject differs from the marker or required terminal evidence
                fails readback.
        """
        if (
            reference.owner_kind != self.authority_owner_kind
            or reference.record_kind != "AlphaGoalResearchMarker"
        ):
            raise ValueError("observation.required_decision_missing")
        marker = self._load("markers", reference.record_hash, AlphaGoalResearchMarker)
        self._readback_terminal_children(marker)
        if reference.subject_id != marker.program_hash:
            raise ValueError("observation.required_decision_missing")
        progress = self.load_goal_progress(marker.goal_progress_hash)
        projection = self.load_projection_for_marker(marker.marker_hash)
        return build_observation_authority_readback(
            reference,
            safe_claims={
                "attempted_spec_count": progress.proposed_count,
                "candidate_registry_hash": marker.registry_hash,
                "completed_batch_count": progress.completed_batch_count,
                "current_qualified_count": progress.current_qualified_count,
                "goal_target": projection.goal_target,
                "marker_hash": marker.marker_hash,
                "qualification_hash": marker.qualification_hash,
                "remaining_batch_count": progress.remaining_batch_count,
                "remaining_spec_count": progress.remaining_spec_count,
                "result_ref_hash": marker.marker_hash,
                "risk_admitted": marker.risk_admitted,
                "scientific_stop_preserved": (
                    marker.disposition == "NO_STABLE_CURRENT_ALPHA_MODEL"
                ),
                "status": marker.disposition,
            },
        )

    def _readback_terminal_children(self, marker: AlphaGoalResearchMarker) -> None:
        registry = self.load_registry(marker.registry_hash)
        qualification = self.load_qualification(marker.qualification_hash)
        progress = self.load_goal_progress(marker.goal_progress_hash)
        board = None if marker.board_hash is None else self.load_board(marker.board_hash)
        if (
            registry.program_hash != marker.program_hash
            or qualification.program_hash != marker.program_hash
            or progress.registry_hash != registry.registry_hash
            or qualification.registry_hash != registry.registry_hash
            or (
                board is not None
                and (
                    board.program_hash != marker.program_hash
                    or board.current_index != len(board.tasks)
                )
            )
        ):
            raise ValueError("alpha_research.goal_marker_child_mismatch")
        if marker.candidate_set_snapshot_hash is not None:
            snapshot = self.load_candidate_set(marker.candidate_set_snapshot_hash)
            if (
                snapshot.program_hash != marker.program_hash
                or snapshot.registry_hash != registry.registry_hash
                or snapshot.qualification_hash != qualification.qualification_hash
            ):
                raise ValueError("alpha_research.candidate_set_readback_failed")
            if snapshot.sector_ema_manifest_hash is not None:
                self.load_sector_ema_manifest(snapshot.sector_ema_manifest_hash)
        if marker.scientific_stop_hash is not None:
            stop = self.load_scientific_stop(marker.scientific_stop_hash)
            if (
                stop.program_hash != marker.program_hash
                or stop.registry_hash != registry.registry_hash
                or stop.qualification_hash != qualification.qualification_hash
                or stop.goal_progress_hash != progress.progress_hash
            ):
                raise ValueError("alpha_research.scientific_stop_readback_failed")

    def _publish_pointer(self, pointer: Path, payload: dict[str, Any], conflict_code: str) -> None:
        if pointer.exists():
            if self._read_json(pointer) != payload:
                raise ValueError(conflict_code)
        else:
            self._atomic_json(pointer, payload)

    def _publish(self, kind: str, value_hash: str, value: BaseModel) -> str:
        path = self.root / kind / f"{value_hash}.json"
        payload = value.model_dump(mode="json")
        if path.exists():
            if self._read_json(path) != payload:
                raise ValueError("alpha_research.goal_artifact_hash_collision")
        else:
            self._atomic_json(path, payload)
        return path.as_uri()

    def _load(self, kind: str, value_hash: str, model: type[ContractT]) -> ContractT:
        path = self.root / kind / f"{value_hash}.json"
        return cast(ContractT, model.model_validate(self._read_json(path)))

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        AlphaGoalResearchArtifactStore._atomic_bytes(
            path,
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            ),
        )

    @staticmethod
    def _atomic_bytes(path: Path, value: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            staging.write_bytes(value)
            os.replace(staging, path)
        finally:
            staging.unlink(missing_ok=True)


__all__ = ["AlphaGoalResearchArtifactStore"]
