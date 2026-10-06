"""Read one workspace-owned Alternative Evidence and CRO authority package.

The ordinary Local Web launcher receives only a workspace. This owner turns the
optional, content-addressed Evidence/CRO package named by that workspace into
the existing runtime, actors and typed resources. It does not acquire evidence,
contact an external Provider or create another application path. An official
SEC source is admitted into the same composition only by explicit consent
(`admit_official_source`): the default remains the recorded package, offline.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, Self, cast
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceError,
    ResearchWorkspaceEvidenceReview,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    WHOLE_FILING_BYTES,
    AlternativeEvidenceClass,
    AlternativeEvidenceMode,
    AlternativeEvidenceSourcePolicy,
    MatterSelectionPolicy,
    SecIssuerRegistrySnapshot,
)
from alphalattice.evidence.alternative_evidence.runtime.policy import AdmittedEvidencePolicy
from alphalattice.evidence.alternative_evidence.runtime.service import (
    AlternativeEvidenceDocumentIntelligenceRuntime,
)
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    AlternativeEvidenceDocumentTaskResources,
)
from alphalattice.evidence.alternative_evidence.sources.recorded import (
    RecordedEvidenceDocument,
)
from alphalattice.kernel.knowledge.hybrid_contracts import (
    RECIPE_MINILM_CPU,
    SUPPORTED_RECIPES,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import (
    AdmittedListingTickerAuthority,
)

if TYPE_CHECKING:
    from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
        PortfolioReviewActor,
    )
from alphalattice.evidence.alternative_evidence.sources.admission import OfficialSourceAdmission

EVIDENCE_REVIEW_MANIFEST_SCHEMA = "evidence-review-workspace-authority"
_HASH = r"^[0-9a-f]{64}$"


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceReviewArtifactBinding(_Contract):
    """One confined JSON child and both its byte and contract identities."""

    relative_path: str = Field(min_length=1, max_length=1024)
    file_sha256: str = Field(pattern=_HASH)
    content_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_path(self) -> Self:
        """Require a confined relative evidence artifact path.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Path is absolute, contains backslashes or has dot traversal.
        """
        _relative(self.relative_path, "evidence_review.artifact_path_invalid")
        return self


class EvidenceReviewModelProfile(_Contract):
    """One non-secret DeepSeek profile shared by both thin actors."""

    provider: Literal["deepseek"] = "deepseek"
    model_name: str = Field(min_length=1, max_length=160)
    base_url: str = Field(min_length=1, max_length=240)
    model_revision_disposition: Literal["PROVIDER_ALIAS_UNPINNED"] = "PROVIDER_ALIAS_UNPINNED"
    native_reasoning: Literal["disabled"] = "disabled"
    analyst_timeout_seconds: float = Field(gt=0.0, le=300.0, allow_inf_nan=False)
    # Ten seconds was measured to be unreachable: one real review of a
    # three-issuer, nineteen-finding dossier took 26.4 s against the same
    # endpoint that answers a trivial call in 1.15 s. The reviewer is the
    # analyst's model doing comparable work, so it gets the analyst's bound.
    review_timeout_seconds: float = Field(gt=0.0, le=300.0, allow_inf_nan=False)
    profile_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(
        cls,
        *,
        model_name: str,
        base_url: str,
        analyst_timeout_seconds: float = 90.0,
        review_timeout_seconds: float = 90.0,
    ) -> Self:
        """Seal normalized evidence model/endpoint and declared bounded call timeouts.

        Args:
            model_name: Explicit provider model alias normalized by stripping whitespace.
            base_url: HTTPS provider endpoint without credentials/query/fragment.
            analyst_timeout_seconds: Declared bounded Analyst timeout.
            review_timeout_seconds: Declared bounded review timeout.

        Returns:
            Validated profile with canonical identity, unpinned alias disposition and native
            reasoning disabled.

        Raises:
            ValueError: Provider endpoint or concrete profile fields are not admitted.
        """
        values: dict[str, object] = {
            "provider": "deepseek",
            "model_name": model_name.strip(),
            "base_url": _normalize_deepseek_base_url(base_url),
            "model_revision_disposition": "PROVIDER_ALIAS_UNPINNED",
            "native_reasoning": "disabled",
            "analyst_timeout_seconds": analyst_timeout_seconds,
            "review_timeout_seconds": review_timeout_seconds,
        }
        return cls(**values, profile_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require normalized model/endpoint fields and exact profile identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Model whitespace, endpoint normalization or profile_hash differs.
        """
        if self.model_name != self.model_name.strip():
            raise ValueError("evidence_review.model_name_not_normalized")
        if self.base_url != _normalize_deepseek_base_url(self.base_url):
            raise ValueError("evidence_review.model_base_url_not_normalized")
        identity = self.model_dump(mode="json", exclude={"profile_hash"})
        if self.profile_hash != canonical_hash(identity):
            raise ValueError("evidence_review.model_profile_identity_invalid")
        return self


class RecordedEvidenceDocumentBundle(_Contract):
    """The bounded local documents an offline Evidence Task may acquire."""

    kind: Literal["RecordedEvidenceDocumentBundle"] = "RecordedEvidenceDocumentBundle"
    documents: tuple[RecordedEvidenceDocument, ...] = Field(min_length=1, max_length=96)
    bundle_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, documents: tuple[RecordedEvidenceDocument, ...]) -> Self:
        """Seal recorded documents in stable entity/revision order.

        Args:
            documents: Explicit already-declared recorded document contracts.

        Returns:
            Validated canonical bundle ordered by entity identity then revision.
        """
        ordered = tuple(sorted(documents, key=lambda value: (value.entity_id, value.revision)))
        values: dict[str, object] = {
            "kind": "RecordedEvidenceDocumentBundle",
            "documents": ordered,
        }
        identity = cls.model_construct(**values, bundle_hash="0" * 64).model_dump(
            mode="json", exclude={"bundle_hash"}
        )
        return cls(**identity, bundle_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require sorted unique entity/revision document keys and exact bundle identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Document keys are unordered/duplicate or bundle_hash differs.
        """
        keys = tuple((value.entity_id, value.revision) for value in self.documents)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("evidence_review.recorded_document_axis_invalid")
        identity = self.model_dump(mode="json", exclude={"bundle_hash"})
        if self.bundle_hash != canonical_hash(identity):
            raise ValueError("evidence_review.recorded_document_bundle_identity_invalid")
        return self


class EvidenceReviewWorkspaceManifest(_Contract):
    """All local authority needed by the existing Evidence and CRO route."""

    kind: Literal["EvidenceReviewWorkspaceManifest"] = "EvidenceReviewWorkspaceManifest"
    manifest_schema: Literal["evidence-review-workspace-authority"] = (
        "evidence-review-workspace-authority"
    )
    authority_id: str = Field(min_length=1, max_length=160)
    issuer_registry: EvidenceReviewArtifactBinding
    listing_authority: EvidenceReviewArtifactBinding
    recorded_documents: EvidenceReviewArtifactBinding
    semantic_model_relative_path: str = Field(min_length=1, max_length=1024)
    semantic_capability_hash: str = Field(pattern=_HASH)
    model_profile: EvidenceReviewModelProfile | None = None
    minimum_entity_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    retrieval_recipe: str = Field(
        default=RECIPE_MINILM_CPU,
        min_length=1,
        max_length=80,
        exclude_if=lambda value: value == RECIPE_MINILM_CPU,
    )
    """The retrieval recipe the packs under `semantic_model_relative_path`
    serve and the capability hash was proven for; absent from the identity at
    the retained default, so every installed manifest keeps its hash."""
    matter_selection: MatterSelectionPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The matter selection this authority asks every request for: the
    production reading plan (None, absent from the identity so every
    installed manifest keeps its hash), or the candidate needs allocation
    over named families -- a QA opt-in bound into the authority hash, the
    research workspace's file identity and each request."""
    authority_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(
        cls,
        *,
        authority_id: str,
        issuer_registry: EvidenceReviewArtifactBinding,
        listing_authority: EvidenceReviewArtifactBinding,
        recorded_documents: EvidenceReviewArtifactBinding,
        semantic_model_relative_path: str,
        semantic_capability_hash: str,
        model_profile: EvidenceReviewModelProfile | None = None,
        minimum_entity_coverage: float = 0.60,
        retrieval_recipe: str = RECIPE_MINILM_CPU,
        matter_selection: MatterSelectionPolicy | None = None,
    ) -> Self:
        """Seal exact recorded evidence, semantic capability and admitted review controls.

        Args:
            authority_id: Explicit evidence workspace authority.
            issuer_registry: Bound issuer registry artifact.
            listing_authority: Bound listing authority artifact.
            recorded_documents: Bound recorded document bundle.
            semantic_model_relative_path: Confined semantic model path.
            semantic_capability_hash: Exact installed semantic capability identity.
            model_profile: Optional declared live review model profile.
            minimum_entity_coverage: Declared minimum issuer coverage.
            retrieval_recipe: Installed retrieval recipe.
            matter_selection: Optional explicit matter selection policy.

        Returns:
            Validated workspace manifest with canonical authority_hash.
        """
        values: dict[str, object] = {
            "kind": "EvidenceReviewWorkspaceManifest",
            "manifest_schema": EVIDENCE_REVIEW_MANIFEST_SCHEMA,
            "authority_id": authority_id,
            "issuer_registry": issuer_registry,
            "listing_authority": listing_authority,
            "recorded_documents": recorded_documents,
            "semantic_model_relative_path": semantic_model_relative_path,
            "semantic_capability_hash": semantic_capability_hash,
            "model_profile": model_profile,
            "minimum_entity_coverage": minimum_entity_coverage,
            "retrieval_recipe": retrieval_recipe,
            "matter_selection": matter_selection,
        }
        identity = cls.model_construct(**values, authority_hash="0" * 64).model_dump(
            mode="json", exclude={"authority_hash"}
        )
        return cls(**identity, authority_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require safe semantic model path, installed retrieval recipe and exact authority.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Relative path, supported retrieval recipe or authority_hash differs.
        """
        _relative(
            self.semantic_model_relative_path,
            "evidence_review.semantic_model_path_invalid",
        )
        if self.retrieval_recipe not in SUPPORTED_RECIPES:
            raise ValueError("evidence_review.retrieval_recipe_unsupported")
        identity = self.model_dump(mode="json", exclude={"authority_hash"})
        if self.authority_hash != canonical_hash(identity):
            raise ValueError("evidence_review.authority_identity_invalid")
        return self


@dataclass(frozen=True, slots=True)
class VerifiedEvidenceReviewWorkspace:
    """Verified local children; no Provider authority has been claimed yet."""

    manifest: EvidenceReviewWorkspaceManifest
    registry: SecIssuerRegistrySnapshot
    listing_authority: AdmittedListingTickerAuthority
    documents: RecordedEvidenceDocumentBundle
    runtime: AlternativeEvidenceDocumentIntelligenceRuntime


@dataclass(frozen=True, slots=True)
class AdmittedEvidenceReviewWorkspace:
    """Verified session-independent resources for the existing composition."""

    authority_hash: str
    registry: SecIssuerRegistrySnapshot
    listing_authority: AdmittedListingTickerAuthority
    runtime: AlternativeEvidenceDocumentIntelligenceRuntime
    resources: AlternativeEvidenceDocumentTaskResources
    evidence_policy: AdmittedEvidencePolicy
    review_actor: PortfolioReviewActor | None
    """Always absent since AG2: a review's answer comes from an agent through the seam."""

    @property
    def model_authority_admitted(self) -> bool:
        """Whether an actor exists that could read evidence or review a book."""
        return self.review_actor is not None and self.resources.analysis_actor is not None


SemanticCapabilityReader = Callable[[AlternativeEvidenceDocumentIntelligenceRuntime], str]


def live_evidence_policy(
    recorded: AdmittedEvidencePolicy,
    *,
    maximum_document_bytes: int | None = None,
    acquisition_window_seconds: int | None = None,
) -> AdmittedEvidencePolicy:
    """Bind explicit official-source acquisition controls to the recorded evidence policy.

    The recorded policy widened to the official source: the same document
    budget, the SEC filing class, live acquisition under recorded consent,
    the per-document cap the operator declared (the recorded policy's when
    none was) and the acquisition window the operator declared (likewise),
    a short unit delivered whole (W4); TTL and temporal eligibility are the
    recorded policy's.
    """
    # The official source delivers a short unit whole (W4): the bundle's file
    # bound is the policy's, so a unit of a few current reports builds no index.
    source_policy = AlternativeEvidenceSourcePolicy(
        **{
            **recorded.source_policy.model_dump(mode="python"),
            "whole_filing_bytes": WHOLE_FILING_BYTES,
            **(
                {}
                if maximum_document_bytes is None
                else {"maximum_document_bytes": maximum_document_bytes}
            ),
        }
    )
    return AdmittedEvidencePolicy(
        evidence_classes=(AlternativeEvidenceClass.SEC_FILING,),
        approved_source_families=recorded.approved_source_families,
        source_policy=source_policy,
        mode=AlternativeEvidenceMode.LIVE_OFFICIAL,
        ttl_seconds=recorded.ttl_seconds,
        acquisition_window_seconds=(
            recorded.acquisition_window_seconds
            if acquisition_window_seconds is None
            else acquisition_window_seconds
        ),
        network_consent=True,
        admit_live_official=True,
        admit_model_review=recorded.admit_model_review,
        matter_selection=recorded.matter_selection,
    )


def admit_evidence_review_workspace(
    *,
    workspace: Path,
    binding: ResearchWorkspaceEvidenceReview,
    semantic_capability_reader: SemanticCapabilityReader | None = None,
    official_source: OfficialSourceAdmission | None = None,
) -> AdmittedEvidenceReviewWorkspace:
    """Compose one verified local Evidence review package.

    Verification, the evidence runtime, the recorded documents and the admitted policy are
    deterministic. The Analyst's and the CRO's answers come from agents through the one seam
    (`protocols/actor_execution`), so no model is admitted here and `model_authority_admitted`
    is false: the Evidence and CRO section offers each agent its bundle (AG2).

    An admitted official source (`official_source.admitted`) puts the same composition on the
    live branch: the source becomes the Task resources' live source and the policy asks for SEC
    filings under LIVE_OFFICIAL with the consent recorded. Without one, or with a refused
    admission, the recorded package and its offline policy are what is composed.
    """
    verified = verify_evidence_review_workspace(
        workspace=workspace,
        binding=binding,
        semantic_capability_reader=semantic_capability_reader,
    )
    manifest = verified.manifest
    runtime = verified.runtime
    live_source = None if official_source is None else official_source.source
    policy = AdmittedEvidencePolicy(
        admit_model_review=False, matter_selection=manifest.matter_selection
    )
    return AdmittedEvidenceReviewWorkspace(
        authority_hash=manifest.authority_hash,
        registry=verified.registry,
        listing_authority=verified.listing_authority,
        runtime=runtime,
        resources=AlternativeEvidenceDocumentTaskResources(
            recorded_registry=verified.registry,
            recorded_documents=verified.documents.documents,
            live_source=live_source,
            analysis_actor=None,
            minimum_entity_coverage=manifest.minimum_entity_coverage,
            admitted_authority_hash=manifest.authority_hash,
            recorded_document_bundle_hash=verified.documents.bundle_hash,
        ),
        evidence_policy=(
            policy
            if official_source is None or live_source is None
            else live_evidence_policy(
                policy,
                maximum_document_bytes=official_source.maximum_document_bytes,
                acquisition_window_seconds=official_source.acquisition_window_seconds,
            )
        ),
        review_actor=None,
    )


def verify_evidence_review_workspace(
    *,
    workspace: Path,
    binding: ResearchWorkspaceEvidenceReview,
    semantic_capability_reader: SemanticCapabilityReader | None = None,
) -> VerifiedEvidenceReviewWorkspace:
    """Open and verify local authority children without claiming a Provider."""
    root = workspace.resolve()
    manifest_path = _confined(root, binding.relative_path)
    if _file_sha256(manifest_path) != binding.file_sha256:
        raise ResearchWorkspaceError("evidence_review.manifest_file_identity_invalid")
    manifest = _read_model(manifest_path, EvidenceReviewWorkspaceManifest)
    registry = _read_bound(
        root, manifest.issuer_registry, SecIssuerRegistrySnapshot, "registry_hash"
    )
    listing = _read_bound(
        root, manifest.listing_authority, AdmittedListingTickerAuthority, "authority_hash"
    )
    documents = _read_bound(
        root, manifest.recorded_documents, RecordedEvidenceDocumentBundle, "bundle_hash"
    )
    semantic_root = _confined(root, manifest.semantic_model_relative_path)
    runtime = AlternativeEvidenceDocumentIntelligenceRuntime(
        artifact_root=root / "runtime" / "artifacts",
        workspace_root=root / "runtime" / "evidence-knowledge",
        model_root=semantic_root,
        retrieval_recipe=manifest.retrieval_recipe,
    )
    try:
        capability_hash = (
            semantic_capability_reader(runtime)
            if semantic_capability_reader is not None
            else _verified_semantic_capability_hash(runtime)
        )
        if capability_hash != manifest.semantic_capability_hash:
            raise ResearchWorkspaceError("evidence_review.semantic_capability_invalid")
        return VerifiedEvidenceReviewWorkspace(
            manifest=manifest,
            registry=registry,
            listing_authority=listing,
            documents=documents,
            runtime=runtime,
        )
    except BaseException:
        runtime.close()
        raise


def _read_bound[T: BaseModel](
    root: Path,
    binding: EvidenceReviewArtifactBinding,
    model: type[T],
    identity_field: str,
) -> T:
    path = _confined(root, binding.relative_path)
    if _file_sha256(path) != binding.file_sha256:
        raise ResearchWorkspaceError("evidence_review.child_file_identity_invalid")
    value = _read_model(path, model)
    if getattr(value, identity_field, None) != binding.content_hash:
        raise ResearchWorkspaceError("evidence_review.child_contract_identity_invalid")
    return value


def _read_model[T: BaseModel](path: Path, model: type[T]) -> T:
    try:
        return cast(T, model.model_validate(json.loads(path.read_text(encoding="utf-8"))))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ResearchWorkspaceError("evidence_review.artifact_unreadable") from error


def _relative(value: str, code: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(code)
    return path


def _confined(root: Path, relative_path: str) -> Path:
    relative = _relative(relative_path, "evidence_review.artifact_path_invalid")
    path = (root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ResearchWorkspaceError("evidence_review.artifact_escapes_workspace") from error
    if not path.exists():
        raise ResearchWorkspaceError("evidence_review.artifact_unreadable")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ResearchWorkspaceError("evidence_review.artifact_unreadable") from error
    return digest.hexdigest()


def _normalize_deepseek_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("evidence_review.deepseek_base_url_invalid")
    if parsed.params or parsed.query or parsed.fragment:
        raise ValueError("evidence_review.deepseek_base_url_invalid")
    return normalized


def _verified_semantic_capability_hash(
    runtime: AlternativeEvidenceDocumentIntelligenceRuntime,
) -> str:
    capability = runtime.retrieval.capability()
    if capability.status != "READY":
        cause = next((d for d in capability.details if d.startswith("cause:")), None)
        raise ResearchWorkspaceError(
            "evidence_review.semantic_capability_invalid"
            + ("" if cause is None else f" ({capability.status}; {cause[6:]})")
        )
    return str(capability.logical_hash)


__all__ = [
    "AdmittedEvidenceReviewWorkspace",
    "EvidenceReviewArtifactBinding",
    "EvidenceReviewModelProfile",
    "EvidenceReviewWorkspaceManifest",
    "RecordedEvidenceDocumentBundle",
    "VerifiedEvidenceReviewWorkspace",
    "admit_evidence_review_workspace",
    "live_evidence_policy",
    "verify_evidence_review_workspace",
]
