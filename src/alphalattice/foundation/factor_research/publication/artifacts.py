"""Content-addressed Factor Research artifacts with marker-last publication."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from alphalattice.control.workspace_runtime.artifacts import ArtifactDescriptor
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FactorPipelineArtifactKind = Literal[
    "deterministic-evidence",
    "program",
    "target-quality",
    "target-surface",
    "walk-forward-plan",
    "oos-evidence",
    "redundancy",
    "decision-dossier",
    "agent-review",
    "review-decision",
    "proposal",
    "research-input",
    "campaign-dossier",
    "curated-checkpoint",
    "campaign-replay",
    "current-projection",
    "current-receipt",
    "current-marker",
]


_PIPELINE_ARTIFACTS: dict[FactorPipelineArtifactKind, tuple[str, str]] = {
    "deterministic-evidence": (
        "pipeline/deterministic-evidence",
        "checkpoint_hash",
    ),
    "program": ("pipeline/programs", "program_hash"),
    "target-quality": ("pipeline/target-quality", "quality_hash"),
    "target-surface": ("pipeline/target-surfaces", "surface_hash"),
    "walk-forward-plan": ("pipeline/walk-forward-plans", "plan_hash"),
    "oos-evidence": ("pipeline/oos-evidence", "report_hash"),
    "redundancy": ("pipeline/redundancy", "structure_hash"),
    "decision-dossier": ("pipeline/decision-dossiers", "dossier_hash"),
    "agent-review": ("pipeline/agent-reviews", "receipt_hash"),
    "review-decision": ("pipeline/review-decisions", "receipt_hash"),
    "proposal": ("pipeline/proposals", "proposal_hash"),
    "research-input": ("pipeline/research-inputs", "input_hash"),
    "campaign-dossier": ("campaign/dossiers", "dossier_hash"),
    "curated-checkpoint": ("campaign/curated-checkpoints", "checkpoint_hash"),
    "campaign-replay": ("campaign/replays", "replay_hash"),
    "current-projection": ("current/projections", "projection_hash"),
    "current-receipt": ("current/receipts", "receipt_hash"),
    "current-marker": ("current/markers", "marker_hash"),
}


class FactorResearchArtifactStore:
    """Own immutable Factor Research children and one safe history projection."""

    _PREFIX = "playpen://factor-research/"

    def __init__(self, artifact_root: Path) -> None:
        """Bind this store to the workspace's Factor Research artifacts.

        Args:
            artifact_root: Workspace root holding Factor Research artifacts.

        """
        self.root = artifact_root.resolve() / "factor-research"

    @staticmethod
    def _require_hash(value: str) -> None:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("Factor Research content identity is not SHA-256")

    def _path(self, category: str, content_hash: str, suffix: str) -> Path:
        self._require_hash(content_hash)
        return self.root / category / f"{content_hash}.{suffix}"

    @classmethod
    def uri(cls, category: str, content_hash: str) -> str:
        """Build the stable URI for a Factor Research artifact.

        Args:
            category: Artifact category within the Factor Research store.
            content_hash: Expected content identity of the artifact.

        Returns:
            Stable URI for the artifact category and identity.

        """
        return f"{cls._PREFIX}{category}/{content_hash}"

    @classmethod
    def _hash_from_uri(cls, uri: str, category: str) -> str:
        prefix = f"{cls._PREFIX}{category}/"
        if not uri.startswith(prefix):
            raise ValueError("unsupported Factor Research artifact URI")
        value = uri[len(prefix) :]
        cls._require_hash(value)
        return value

    @staticmethod
    def _json_bytes(payload: Mapping[str, object]) -> bytes:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")

    @staticmethod
    def _atomic_write(target: Path, content: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        staged.write_bytes(content)
        os.replace(staged, target)
        staged.unlink(missing_ok=True)

    def _publish_identity_json(
        self,
        *,
        category: str,
        payload: Mapping[str, object],
        content_hash: str,
        identity_field: str,
    ) -> ArtifactDescriptor:
        self._require_hash(content_hash)
        if payload.get(identity_field) != content_hash:
            raise ValueError("Factor Research JSON identity field is invalid")
        identity = dict(payload)
        identity.pop(identity_field, None)
        if canonical_hash(identity) != content_hash:
            raise ValueError("Factor Research JSON content hash is invalid")
        serialized = self._json_bytes(payload)
        target = self._path(category, content_hash, "json")
        if target.exists():
            if target.read_bytes() != serialized:
                raise ValueError("Factor Research identity was reused with new content")
        else:
            self._atomic_write(target, serialized)
        return ArtifactDescriptor(
            kind=f"factor-research-{category.replace('/', '-')}",
            content_hash=content_hash,
            metadata_hash=hashlib.sha256(serialized).hexdigest(),
            uri=self.uri(category, content_hash),
        )

    def _load_identity_json(
        self, *, uri: str, category: str, identity_field: str
    ) -> dict[str, object]:
        content_hash = self._hash_from_uri(uri, category)
        target = self._path(category, content_hash, "json")
        if not target.is_file():
            raise FileNotFoundError("Factor Research JSON artifact is missing")
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get(identity_field) != content_hash:
            raise ValueError("Factor Research JSON requested identity is invalid")
        identity = dict(payload)
        identity.pop(identity_field, None)
        if canonical_hash(identity) != content_hash:
            raise ValueError("Factor Research JSON payload hash is invalid")
        return payload

    def publish_pipeline_contract(
        self,
        *,
        artifact_kind: FactorPipelineArtifactKind,
        payload: Mapping[str, object],
        content_hash: str,
    ) -> ArtifactDescriptor:
        """Publish one admitted contract from the correctness pipeline.

        Args:
            artifact_kind: Registered contract category and identity field.
            payload: Contract body including its declared identity.
            content_hash: Expected content identity.

        Returns:
            Descriptor for the verified immutable contract.

        Raises:
            ValueError: If the body or stored artifact has a different identity.

        """
        category, identity_field = _PIPELINE_ARTIFACTS[artifact_kind]
        return self._publish_identity_json(
            category=category,
            payload=payload,
            content_hash=content_hash,
            identity_field=identity_field,
        )

    def load_pipeline_contract(
        self,
        *,
        artifact_kind: FactorPipelineArtifactKind,
        uri: str,
    ) -> dict[str, object]:
        """Read back and verify one correctness-pipeline contract.

        Args:
            artifact_kind: Registered contract category and identity field.
            uri: URI of the contract to load.

        Returns:
            Verified contract body.

        Raises:
            FileNotFoundError: If the contract is missing.
            ValueError: If its category or content identity is invalid.

        """
        category, identity_field = _PIPELINE_ARTIFACTS[artifact_kind]
        return self._load_identity_json(
            uri=uri,
            category=category,
            identity_field=identity_field,
        )

    def find_deterministic_evidence(
        self, *, program_hash: str, execution_binding_hash: str
    ) -> dict[str, object] | None:
        """Find one immutable statistics checkpoint without activating it.

        Args:
            program_hash: Program identity to match.
            execution_binding_hash: Execution binding identity to match.

        Returns:
            Verified checkpoint, or ``None`` when no matching checkpoint exists.

        Raises:
            ValueError: If either identity or a scanned checkpoint is invalid.

        """
        self._require_hash(program_hash)
        self._require_hash(execution_binding_hash)
        category, identity_field = _PIPELINE_ARTIFACTS["deterministic-evidence"]
        root = self.root / category
        for path in sorted(root.glob("*.json")) if root.is_dir() else ():
            payload = self._load_identity_json(
                uri=self.uri(category, path.stem),
                category=category,
                identity_field=identity_field,
            )
            raw_program = payload.get("program")
            bound_program_hash = (
                raw_program.get("program_hash") if isinstance(raw_program, dict) else None
            )
            if (
                bound_program_hash == program_hash
                and payload.get("execution_binding_hash") == execution_binding_hash
            ):
                return payload
        return None

    def find_authoritative_review(self, *, dossier_hash: str) -> dict[str, object] | None:
        """Find an authoritative review after an interrupted publication.

        Args:
            dossier_hash: Decision dossier identity to match.

        Returns:
            Verified review, or ``None`` when none matches.

        Raises:
            ValueError: If the identity or a scanned review is invalid.

        """
        self._require_hash(dossier_hash)
        category, identity_field = _PIPELINE_ARTIFACTS["agent-review"]
        root = self.root / category
        for path in sorted(root.glob("*.json")) if root.is_dir() else ():
            payload = self._load_identity_json(
                uri=self.uri(category, path.stem),
                category=category,
                identity_field=identity_field,
            )
            if (
                payload.get("sample_role") == "AUTHORITATIVE"
                and payload.get("dossier_hash") == dossier_hash
            ):
                return payload
        return None

    def find_review_decision(
        self,
        *,
        dossier_hash: str,
        decision_policy_hash: str,
    ) -> dict[str, object] | None:
        """Find a neutral Host decision after an interrupted publication.

        Args:
            dossier_hash: Decision dossier identity to match.
            decision_policy_hash: Installed decision policy identity to match.

        Returns:
            Verified decision, or ``None`` when none matches.

        Raises:
            ValueError: If an identity or a scanned decision is invalid.

        """
        self._require_hash(dossier_hash)
        self._require_hash(decision_policy_hash)
        category, identity_field = _PIPELINE_ARTIFACTS["review-decision"]
        root = self.root / category
        for path in sorted(root.glob("*.json")) if root.is_dir() else ():
            payload = self._load_identity_json(
                uri=self.uri(category, path.stem),
                category=category,
                identity_field=identity_field,
            )
            submission = payload.get("submission")
            if (
                isinstance(submission, dict)
                and submission.get("dossier_hash") == dossier_hash
                and payload.get("decision_policy_hash") == decision_policy_hash
            ):
                return payload
        return None

    def load_outcome_manifest(self, uri: str) -> dict[str, object]:
        """Load and verify a training outcome manifest.

        Args:
            uri: URI of the artifact to load and verify.

        Returns:
            Verified artifact payload.

        Raises:
            FileNotFoundError: If a requested artifact is missing.
            ValueError: If its identity or content hash is invalid.

        """
        return self._load_identity_json(
            uri=uri, category="outcomes/manifests", identity_field="snapshot_hash"
        )

    def outcome_manifests(
        self,
    ) -> tuple[tuple[dict[str, object], ArtifactDescriptor], ...]:
        """Enumerate immutable outcome manifests for Data Operations reuse.

        Returns:
            Verified manifests paired with their artifact descriptors.

        Raises:
            ValueError: If a stored manifest fails identity verification.

        """
        root = self.root / "outcomes" / "manifests"
        values: list[tuple[dict[str, object], ArtifactDescriptor]] = []
        for path in sorted(root.glob("*.json")) if root.is_dir() else ():
            uri = self.uri("outcomes/manifests", path.stem)
            payload = self.load_outcome_manifest(uri)
            serialized = self._json_bytes(payload)
            values.append(
                (
                    payload,
                    ArtifactDescriptor(
                        kind="factor-research-outcomes-manifests",
                        content_hash=path.stem,
                        metadata_hash=hashlib.sha256(serialized).hexdigest(),
                        uri=uri,
                    ),
                )
            )
        return tuple(values)

    def outcome_reuse_receipts(self) -> tuple[dict[str, object], ...]:
        """Enumerate verified training outcome reuse receipts.

        Returns:
            Verified reuse receipts in artifact-name order.

        Raises:
            FileNotFoundError: If a requested artifact is missing.
            ValueError: If its identity or content hash is invalid.

        """
        root = self.root / "outcomes" / "reuse-receipts"
        return tuple(
            self._load_identity_json(
                uri=self.uri("outcomes/reuse-receipts", path.stem),
                category="outcomes/reuse-receipts",
                identity_field="receipt_hash",
            )
            for path in (sorted(root.glob("*.json")) if root.is_dir() else ())
        )

    def prerequisite_projection(self) -> dict[str, object]:
        """Read the available Factor Research prerequisite projection.

        Returns:
            Available prerequisite projection.

        Raises:
            FileNotFoundError: If no prerequisite projection exists.
            ValueError: If the projection shape is invalid.

        """
        target = self.root / "prerequisites" / "active-projection.json"
        if not target.is_file():
            raise FileNotFoundError("Factor Research prerequisite projection is unavailable")
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Factor Research prerequisite projection is invalid")
        return payload

    def publish_research_desk_input(
        self, *, payload: Mapping[str, object], input_hash: str
    ) -> ArtifactDescriptor:
        """Publish an immutable Research Desk input.

        Args:
            payload: Artifact body containing its declared identity field.
            input_hash: Expected identity of the Research Desk input.

        Returns:
            Descriptor for the verified immutable artifact.

        Raises:
            ValueError: If the payload or stored identity is invalid.

        """
        return self._publish_identity_json(
            category="research-desk/inputs",
            payload=payload,
            content_hash=input_hash,
            identity_field="input_hash",
        )

    def load_research_desk_input(self, uri: str) -> dict[str, object]:
        """Load and verify a Research Desk input.

        Args:
            uri: URI of the artifact to load and verify.

        Returns:
            Verified artifact payload.

        Raises:
            FileNotFoundError: If a requested artifact is missing.
            ValueError: If its identity or content hash is invalid.

        """
        return self._load_identity_json(
            uri=uri, category="research-desk/inputs", identity_field="input_hash"
        )

    def publish_research_foundation(
        self, *, payload: Mapping[str, object], foundation_hash: str
    ) -> ArtifactDescriptor:
        """Publish an immutable Research Foundation contract.

        Args:
            payload: Artifact body containing its declared identity field.
            foundation_hash: Expected identity of the Research Foundation contract.

        Returns:
            Descriptor for the verified immutable artifact.

        Raises:
            ValueError: If the payload or stored identity is invalid.

        """
        return self._publish_identity_json(
            category="research-desk/foundations",
            payload=payload,
            content_hash=foundation_hash,
            identity_field="foundation_hash",
        )

    def load_research_foundation(self, uri: str) -> dict[str, object]:
        """Load and verify a Research Foundation contract.

        Args:
            uri: URI of the artifact to load and verify.

        Returns:
            Verified artifact payload.

        Raises:
            FileNotFoundError: If a requested artifact is missing.
            ValueError: If its identity or content hash is invalid.

        """
        return self._load_identity_json(
            uri=uri,
            category="research-desk/foundations",
            identity_field="foundation_hash",
        )

    def publish_research_foundation_marker(
        self, *, payload: Mapping[str, object], marker_hash: str
    ) -> ArtifactDescriptor:
        """Publish an immutable Research Foundation marker.

        Args:
            payload: Artifact body containing its declared identity field.
            marker_hash: Expected identity of the marker.

        Returns:
            Descriptor for the verified immutable artifact.

        Raises:
            ValueError: If the payload or stored identity is invalid.

        """
        return self._publish_identity_json(
            category="research-desk/foundation-markers",
            payload=payload,
            content_hash=marker_hash,
            identity_field="marker_hash",
        )

    def load_research_foundation_marker(self, uri: str) -> dict[str, object]:
        """Load and verify a Research Foundation marker.

        Args:
            uri: URI of the artifact to load and verify.

        Returns:
            Verified artifact payload.

        Raises:
            FileNotFoundError: If a requested artifact is missing.
            ValueError: If its identity or content hash is invalid.

        """
        return self._load_identity_json(
            uri=uri,
            category="research-desk/foundation-markers",
            identity_field="marker_hash",
        )

    def publish_pre_research_desk_projection(self, payload: Mapping[str, object]) -> None:
        """Atomically write the safe pre-Research Desk projection.

        Args:
            payload: Safe projection body to persist.

        """
        self._atomic_write(
            self.root / "research-desk" / "pre-research-projection.json",
            self._json_bytes(payload),
        )

    def publish_foundation_admission(
        self, *, payload: Mapping[str, object], admission_hash: str
    ) -> ArtifactDescriptor:
        """Publish an immutable Research Foundation admission.

        Args:
            payload: Artifact body containing its declared identity field.
            admission_hash: Expected identity of the Foundation admission.

        Returns:
            Descriptor for the verified immutable artifact.

        Raises:
            ValueError: If the payload or stored identity is invalid.

        """
        return self._publish_identity_json(
            category="research-desk/foundation-admissions",
            payload=payload,
            content_hash=admission_hash,
            identity_field="admission_hash",
        )

    def load_foundation_admission(self, admission_hash: str) -> dict[str, object]:
        """Load and verify a Research Foundation admission.

        Args:
            admission_hash: Expected identity of the Foundation admission.

        Returns:
            Verified admission payload.

        Raises:
            FileNotFoundError: If a requested artifact is missing.
            ValueError: If its identity or content hash is invalid.

        """
        return self._load_identity_json(
            uri=self.uri("research-desk/foundation-admissions", admission_hash),
            category="research-desk/foundation-admissions",
            identity_field="admission_hash",
        )

    def record_foundation_confirmation(self, payload: Mapping[str, object]) -> None:
        """Keep caller provenance separately from the actor-neutral admission.

        Args:
            payload: Confirmation body binding a submission and admission.

        Raises:
            ValueError: If publication or readback changes the binding.

        """
        admission_hash = str(payload["submission_hash"])
        self._require_hash(admission_hash)
        category = f"research-desk/foundation-confirmations/{admission_hash}"
        descriptor = self._publish_identity_json(
            category=category,
            payload=payload,
            content_hash=str(payload["binding_hash"]),
            identity_field="binding_hash",
        )
        if (
            self._load_identity_json(
                uri=descriptor.uri, category=category, identity_field="binding_hash"
            )
            != payload
        ):
            raise ValueError("research_foundation.confirmation_readback_mismatch")

    def pre_research_desk_projection(self) -> dict[str, object]:
        """Read the available safe pre-Research Desk projection.

        Returns:
            Available safe projection.

        Raises:
            FileNotFoundError: If no safe projection exists.
            ValueError: If the projection shape is invalid.

        """
        target = self.root / "research-desk" / "pre-research-projection.json"
        if not target.is_file():
            raise FileNotFoundError("pre-Research Desk safe projection is unavailable")
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("pre-Research Desk safe projection is invalid")
        return payload


__all__ = ["FactorResearchArtifactStore"]
