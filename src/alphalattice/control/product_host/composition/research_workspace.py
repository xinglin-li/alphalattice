"""One identity-bound Research Workspace manifest and its readback.

The launcher receives a directory. This owner is the only code that turns that
directory into installed strategy artifacts, a catalog and a default selection.
Paths in the manifest are relative to the workspace and confined to it, so a
copied or edited manifest cannot silently point a local product at another
workspace.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from alphalattice.control.data_platform.maintenance.contracts import WorkspaceDataUpdateBinding
from alphalattice.control.workspace_runtime.content_store import replace_shared_file
from alphalattice.interface.local_application.failure_codes import public_failure
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    ScoreSourceMode,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    InstalledFrozenStrategyCatalog,
)
from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (
    STRATEGY_ARTIFACT_KEYS,
    install_frozen_strategies,
)
from alphalattice.investment.portfolio_strategy_lab.policies.lifecycle_research import (
    ARTIFACT_KEY as LIFECYCLE_RESEARCH_ARTIFACT_KEY,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.retired_spellings import (
    PREPARED_BEFORE_RENAMES,
    retired_spelling_in,
)

if TYPE_CHECKING:
    from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
    from alphalattice.investment.alpha_research.publication.artifacts import (
        AlphaCurrentArtifactStore,
    )
    from alphalattice.investment.alpha_research.scores.model_renewal import (
        AlphaModelLifecycleAdmission,
    )

RESEARCH_WORKSPACE_MANIFEST_NAME = "research-workspace.json"


class ResearchWorkspaceError(ValueError):
    """Stable refusal for workspace configuration and admission."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResearchWorkspaceArtifact(_Contract):
    """One installed-strategy artifact rooted inside the Research Workspace."""

    artifact_key: str = Field(min_length=1, max_length=160)
    relative_path: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_path(self) -> Self:
        """Require a confined relative artifact path.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ResearchWorkspaceError: Path is absolute, contains backslashes or has dot traversal.
        """
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or "\\" in self.relative_path
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            raise ResearchWorkspaceError("research_workspace.artifact_path_invalid")
        return self


class ResearchWorkspaceEvidenceReview(_Contract):
    """The optional Evidence/CRO authority package installed in this workspace."""

    relative_path: str = Field(min_length=1, max_length=1024)
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_path(self) -> Self:
        """Require a confined relative evidence review path.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ResearchWorkspaceError: Path is absolute, contains backslashes or has dot traversal.
        """
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or "\\" in self.relative_path
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            raise ResearchWorkspaceError("research_workspace.evidence_review_path_invalid")
        return self


class ResearchWorkspaceScoreInput(_Contract):
    """A component preparation capability, not another executable Portfolio mode."""

    strategy_package_id: str = Field(min_length=1)
    strategy_package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    component_id: str | None = Field(default=None, exclude_if=lambda v: v is None)
    authority_relative_path: str = Field(min_length=1)
    authority_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_kind: Literal["WORKSPACE_DATA_FEATURE", "RECORDED_INPUT_SNAPSHOT"]
    observation_snapshot_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_source(self) -> Self:
        ResearchWorkspaceArtifact(
            artifact_key="SCORING_AUTHORITY", relative_path=self.authority_relative_path
        )
        if (self.source_kind == "RECORDED_INPUT_SNAPSHOT") != (
            self.observation_snapshot_hash is not None
        ):
            raise ResearchWorkspaceError("research_workspace.scoring_source_invalid")
        return self


class ResearchWorkspaceCalibrationInput(_Contract):
    """A package's calibration seed: an operator's QA epoch, or a person's activation (LS1)."""

    strategy_package_id: str = Field(min_length=1)
    strategy_package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_kind: Literal["RECORDED_INPUT_SNAPSHOT", "WORKSPACE_DATA_FEATURE"]


class ResearchWorkspaceDecisionUpdate(_Contract):
    """The checkpoint a package's decisions continue, a QA state or a person's activation.

    A person's activation of a reviewed research book runs its strategy forward (LS1, OW12).
    """

    strategy_package_id: str = Field(min_length=1)
    strategy_package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchWorkspaceExperimentInput(_Contract):
    """An immutable local research input closure, never a caller path grant."""

    input_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,95}$")
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchWorkspaceModelTrainingInput(_Contract):
    """A research input/component capability, not a strategy installation."""

    component_id: str = Field(pattern=r"^[A-Z][A-Z0-9_]+$")
    input_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    authority_relative_path: str = Field(min_length=1)
    authority_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def source_handle(self) -> str:
        """Name this component training source with its exact authority identity.

        Returns:
            Normalized component-training handle bound to authority_hash.
        """
        return f"component-training.{self.component_id.lower()}@{self.authority_hash}"

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_path(self) -> Self:
        """Require the declared model training authority path to be confined.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ResearchWorkspaceError: Authority path is not a valid relative workspace artifact path.
        """
        ResearchWorkspaceArtifact(
            artifact_key="RESEARCH_MODEL_TRAINING", relative_path=self.authority_relative_path
        )
        return self


def component_training_selection(handle: str) -> tuple[str, str | None]:
    """Read legacy component handles and explicit research-input selectors."""
    if not handle.startswith("component-training."):
        raise ResearchWorkspaceError("research_workspace.training_handle_invalid")
    component, separator, binding = handle.removeprefix("component-training.").partition("@")
    if not component or (
        separator and (len(binding) != 64 or any(c not in "0123456789abcdef" for c in binding))
    ):
        raise ResearchWorkspaceError("research_workspace.training_handle_invalid")
    return component.upper(), binding if separator else None


class ResearchWorkspaceManifest(_Contract):
    """The complete local configuration the ordinary launcher may consume."""

    kind: Literal["ResearchWorkspaceManifest"] = "ResearchWorkspaceManifest"
    manifest_schema: Literal["research-workspace-manifest"] = "research-workspace-manifest"
    workspace_id: str = Field(min_length=1, max_length=160)
    default_strategy_package_id: str | None = Field(default=None, min_length=1, max_length=160)
    default_score_source_mode: ScoreSourceMode | None = None
    strategy_installation: Literal["NOT_INSTALLED", "NON_DEFAULT_RESEARCH"] | None = None
    strategy_artifacts: tuple[ResearchWorkspaceArtifact, ...]
    evidence_review: ResearchWorkspaceEvidenceReview | None = None
    data_update: WorkspaceDataUpdateBinding | None = None
    score_inputs: tuple[ResearchWorkspaceScoreInput, ...] | None = None
    calibration_inputs: tuple[ResearchWorkspaceCalibrationInput, ...] | None = None
    decision_updates: tuple[ResearchWorkspaceDecisionUpdate, ...] | None = None
    experiment_inputs: tuple[ResearchWorkspaceExperimentInput, ...] | None = None
    model_training_inputs: tuple[ResearchWorkspaceModelTrainingInput, ...] | None = None
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        workspace_id: str,
        default_strategy_package_id: str | None,
        default_score_source_mode: ScoreSourceMode | None,
        strategy_artifacts: tuple[ResearchWorkspaceArtifact, ...],
        evidence_review: ResearchWorkspaceEvidenceReview | None = None,
        data_update: WorkspaceDataUpdateBinding | None = None,
        score_inputs: tuple[ResearchWorkspaceScoreInput, ...] | None = None,
        calibration_inputs: tuple[ResearchWorkspaceCalibrationInput, ...] | None = None,
        decision_updates: tuple[ResearchWorkspaceDecisionUpdate, ...] | None = None,
        experiment_inputs: tuple[ResearchWorkspaceExperimentInput, ...] | None = None,
        model_training_inputs: tuple[ResearchWorkspaceModelTrainingInput, ...] | None = None,
        strategy_installation: Literal["NOT_INSTALLED", "NON_DEFAULT_RESEARCH"] | None = None,
    ) -> Self:
        """Seal normalized workspace declarations with absent optional fields excluded.

        Args:
            workspace_id: Explicit workspace identity.
            default_strategy_package_id: Optional declared default strategy.
            default_score_source_mode: Optional declared default score source.
            strategy_artifacts: Explicit artifacts ordered by artifact key.
            evidence_review: Optional admitted review declaration.
            data_update: Optional admitted data update declaration.
            score_inputs: Optional score source declarations.
            calibration_inputs: Optional calibration source declarations.
            decision_updates: Optional decision checkpoint declarations.
            experiment_inputs: Optional experiment input declarations.
            model_training_inputs: Optional component authority declarations, sorted by
                component/authority.
            strategy_installation: Optional explicit installation mode.

        Returns:
            Validated manifest with canonical identity over its non-null fields.
        """
        ordered = tuple(sorted(strategy_artifacts, key=lambda value: value.artifact_key))
        values: dict[str, object] = {
            "kind": "ResearchWorkspaceManifest",
            "manifest_schema": "research-workspace-manifest",
            "workspace_id": workspace_id,
            "default_strategy_package_id": default_strategy_package_id,
            "default_score_source_mode": default_score_source_mode,
            "strategy_artifacts": ordered,
        }
        if evidence_review is not None:
            values["evidence_review"] = evidence_review
        if data_update is not None:
            values["data_update"] = data_update
        if score_inputs is not None:
            values["score_inputs"] = score_inputs
        if calibration_inputs is not None:
            values["calibration_inputs"] = calibration_inputs
        if decision_updates is not None:
            values["decision_updates"] = decision_updates
        if experiment_inputs is not None:
            values["experiment_inputs"] = experiment_inputs
        if model_training_inputs is not None:
            values["model_training_inputs"] = tuple(
                sorted(model_training_inputs, key=lambda v: (v.component_id, v.authority_hash))
            )
        if strategy_installation is not None:
            values["strategy_installation"] = strategy_installation
        identity = cls.model_construct(**values, manifest_hash="0" * 64).model_dump(
            mode="json", exclude={"manifest_hash"}, exclude_none=True
        )
        return cls(**identity, manifest_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact manifest identity and consistent unique installed bindings.

        Require installation-mode consistency, unique sorted bindings and exact manifest identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ResearchWorkspaceError: Default strategy, installation/artifact keys, input uniqueness
                or canonical manifest binding differs.
        """
        if self.strategy_installation == "NOT_INSTALLED":
            if (
                self.default_strategy_package_id is not None
                or self.default_score_source_mode is not None
                or self.strategy_artifacts
                or self.score_inputs
                or self.calibration_inputs
                or self.decision_updates
            ):
                raise ResearchWorkspaceError("research_workspace.uninstalled_strategy_conflict")
        elif self.strategy_installation == "NON_DEFAULT_RESEARCH":
            # A research installation has no default; its live bindings are a person's
            # activation of one of its books (LS1), so they may stand beside it.
            if (
                self.default_strategy_package_id is not None
                or self.default_score_source_mode is not None
                or tuple(v.artifact_key for v in self.strategy_artifacts)
                != (LIFECYCLE_RESEARCH_ARTIFACT_KEY,)
            ):
                raise ResearchWorkspaceError("research_workspace.non_default_research_conflict")
        elif self.default_strategy_package_id is None or self.default_score_source_mode is None:
            raise ResearchWorkspaceError("research_workspace.strategy_default_required")
        ids = tuple(v.input_id for v in self.experiment_inputs or ())
        if len(set(ids)) != len(ids):
            raise ResearchWorkspaceError("research_workspace.experiment_input_duplicate")
        training_keys = tuple(
            (v.component_id, v.authority_hash) for v in self.model_training_inputs or ()
        )
        if training_keys != tuple(sorted(set(training_keys))):
            raise ResearchWorkspaceError("research_workspace.model_training_input_duplicate")
        keys = tuple(value.artifact_key for value in self.strategy_artifacts)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ResearchWorkspaceError("research_workspace.artifact_keys_invalid")
        unknown = tuple(sorted(set(keys) - set(STRATEGY_ARTIFACT_KEYS)))
        if unknown:
            raise ResearchWorkspaceError(
                "research_workspace.artifact_key_unknown:" + ",".join(unknown)
            )
        identity = self.model_dump(mode="json", exclude={"manifest_hash"}, exclude_none=True)
        if self.manifest_hash != canonical_hash(identity):
            raise ResearchWorkspaceError("research_workspace.manifest_identity_invalid")
        return self

    @classmethod
    def research_only(cls, workspace_id: str) -> Self:
        """Declare a workspace with strategy installation explicitly absent.

        Args:
            workspace_id: Explicit workspace identity.

        Returns:
            Sealed NOT_INSTALLED workspace manifest with no strategy defaults or artifacts.
        """
        return cls.create(
            workspace_id=workspace_id,
            default_strategy_package_id=None,
            default_score_source_mode=None,
            strategy_artifacts=(),
            strategy_installation="NOT_INSTALLED",
        )

    def with_bindings(self, **changes: Any) -> Self:
        """Preserve typed configuration when one owner replaces its bindings."""
        allowed = {
            "evidence_review",
            "data_update",
            "score_inputs",
            "calibration_inputs",
            "decision_updates",
            "experiment_inputs",
            "model_training_inputs",
        }
        if set(changes) - allowed:
            raise ResearchWorkspaceError("research_workspace.binding_field_unknown")
        values = {
            name: getattr(self, name)
            for name in type(self).model_fields
            if name not in {"kind", "manifest_schema", "manifest_hash"}
        }
        return self.create(**{**values, **changes})


@dataclass(frozen=True, slots=True)
class AdmittedResearchWorkspace:
    """Verified manifest plus its installed generic strategy catalog."""

    root: Path
    manifest: ResearchWorkspaceManifest
    catalog: InstalledFrozenStrategyCatalog | None

    def require_catalog(self) -> InstalledFrozenStrategyCatalog:
        """Require the installed frozen strategy catalog before strategy operations.

        Returns:
            Exact admitted strategy catalog.

        Raises:
            ResearchWorkspaceError: Strategy installation is absent.
        """
        if self.catalog is None:
            raise ResearchWorkspaceError("research_workspace.strategy_not_installed")
        return self.catalog


def publish_research_workspace_manifest(
    workspace: Path, manifest: ResearchWorkspaceManifest
) -> Path:
    """Atomically publish a whole manifest file, replacing any there.

    The product never calls it beside its writes: a new workspace's first manifest is
    `create_research_workspace_manifest`, and every change of an existing one is
    `update_research_workspace_manifest` (WM). What still calls it holds the
    workspace alone: a test's fixture, or a script under the workspace's writer lease.
    """
    root = workspace.resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / RESEARCH_WORKSPACE_MANIFEST_NAME
    payload = manifest.model_dump_json(indent=2).encode("utf-8") + b"\n"
    with NamedTemporaryFile(dir=root, prefix=".research-workspace-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        replace_shared_file(temporary, destination)  # every request reads it
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def create_research_workspace_manifest(
    workspace: Path, manifest: ResearchWorkspaceManifest
) -> ResearchWorkspaceManifest:
    """Publish the first manifest of a workspace that has none.

    Args:
        workspace: The workspace, held by its writer lease.
        manifest: Its first manifest.

    Returns:
        The manifest published.

    Raises:
        ResearchWorkspaceError: `research_workspace.manifest_exists` when one is there:
            a change of it goes through `update_research_workspace_manifest`.
    """
    if (workspace.resolve() / RESEARCH_WORKSPACE_MANIFEST_NAME).exists():
        raise ResearchWorkspaceError("research_workspace.manifest_exists")
    publish_research_workspace_manifest(workspace, manifest)
    return manifest


def update_research_workspace_manifest(
    workspace: Path,
    change: Callable[[ResearchWorkspaceManifest], ResearchWorkspaceManifest],
    *,
    gate: WorkspaceMutationGate,
) -> tuple[ResearchWorkspaceManifest, ResearchWorkspaceManifest]:
    """Change the workspace manifest as it stands, under the workspace's mutation gate.

    The manifest is one file its writers replace whole. A writer that read it, built its
    change and published later lost whatever another wrote in between: a strategy
    installation beside a model-training publication, an installation beside a
    preparation's recovery. Here each writer's change is applied to the manifest
    read under the gate, so none replaces what another wrote; a change refuses by
    raising when a field it planned on has moved. The one write of an existing
    manifest (WM).

    Args:
        workspace: The workspace whose manifest changes.
        change: The writer's change, given the current manifest; it returns the new
            one, or the current one when it has nothing to write.
        gate: The workspace's mutation gate: the Host session's, or a writer's own
            while it holds the workspace's writer lease.

    Returns:
        The manifest before the change and after it.

    Raises:
        ResearchWorkspaceError: `research_workspace.publication_readback_failed` when
            the file does not read back as the manifest published.
    """
    with gate.hold():
        current = read_research_workspace_manifest(workspace)
        updated = change(current)
        if updated != current:
            publish_research_workspace_manifest(workspace, updated)
            if read_research_workspace_manifest(workspace) != updated:
                raise ResearchWorkspaceError("research_workspace.publication_readback_failed")
        return current, updated


class ResearchWorkspaceManifestHolder:
    """The one copy of the workspace manifest the Host's applications read.

    The Host refreshes it once when a verified publication moves the manifest, and every
    application reads its `current`, so none keeps a copy of its own to fall behind.
    """

    __slots__ = ("current",)

    def __init__(self, manifest: ResearchWorkspaceManifest) -> None:
        self.current = manifest


def held(
    manifest: ResearchWorkspaceManifest | ResearchWorkspaceManifestHolder,
) -> ResearchWorkspaceManifestHolder:
    """The holder an application reads: the Host's own, or one for a manifest given alone."""
    if isinstance(manifest, ResearchWorkspaceManifestHolder):
        return manifest
    return ResearchWorkspaceManifestHolder(manifest)


def manifest_fields_hash(manifest: ResearchWorkspaceManifest, fields: Iterable[str]) -> str:
    """Name the fields of a manifest a plan reads and writes, by their values.

    A plan binds these, never the whole manifest, so another owner's publication of
    fields the plan does not read leaves it applicable: a model-training publication
    no longer stops a strategy installation or an input capture.

    Args:
        manifest: The manifest as the plan read it, or as it stands when it applies.
        fields: The manifest's field names the plan reads or writes.

    Returns:
        A digest equal whenever those fields are.

    Raises:
        ResearchWorkspaceError: `research_workspace.manifest_field_unknown` for a name
            the manifest does not have.
    """
    names = sorted(set(fields))
    if not names or set(names) - set(type(manifest).model_fields):
        raise ResearchWorkspaceError("research_workspace.manifest_field_unknown")
    values = manifest.model_dump(mode="json")
    return str(canonical_hash({name: values[name] for name in names}))


def initialize_research_workspace(workspace: Path) -> None:
    """Create configuration only for a genuinely empty workspace, under its lease."""
    from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease

    root = workspace.resolve()
    if (root / RESEARCH_WORKSPACE_MANIFEST_NAME).exists():
        return
    lease = WorkspaceWriterLease.acquire(root)
    try:
        if (root / RESEARCH_WORKSPACE_MANIFEST_NAME).exists():
            return
        if any(p.name != ".alphalattice-writer.lock" for p in root.iterdir()):
            raise ResearchWorkspaceError("research_workspace.nonempty_manifest_absent")
        from uuid import uuid4

        create_research_workspace_manifest(
            root, ResearchWorkspaceManifest.research_only(f"research-{uuid4()}")
        )
    finally:
        lease.close()


def read_research_workspace_manifest(workspace: Path) -> ResearchWorkspaceManifest:
    """Open and recursively verify the manifest at the workspace boundary.

    The format grows by optional fields, so every earlier manifest reads (all 449
    saved by September 2026 did). A manifest carrying a field or schema this build
    does not know was written by a newer build; it is refused by that name, with
    the fields, so the next step -- open it with that build or a newer one -- is
    plain. A change that is not additive brings its own schema name and an
    upgrader from the previous one, here. A manifest holding a spelling NM2 retired
    was prepared before the 2026-10-02 renames, which this build does not read; it is
    refused by that name, the spelling its subject, and left as it is.
    """
    path = workspace.resolve() / RESEARCH_WORKSPACE_MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise ResearchWorkspaceError("research_workspace.manifest_unreadable") from error
    retired = retired_spelling_in(payload)
    if retired is not None:
        raise ResearchWorkspaceError(f"{PREPARED_BEFORE_RENAMES}:{retired}")
    try:
        return cast(
            ResearchWorkspaceManifest,
            ResearchWorkspaceManifest.model_validate(payload),
        )
    except ValidationError as error:
        newer = sorted(
            {
                ".".join(str(part) for part in item["loc"])
                for item in error.errors()
                if item["type"] == "extra_forbidden"
                or (
                    item["type"] == "literal_error"
                    and item["loc"] in (("kind",), ("manifest_schema",))
                )
            }
        )
        if newer:
            raise ResearchWorkspaceError(
                "research_workspace.manifest_from_newer_build:" + ",".join(newer)
            ) from error
        # Each field refused and its rule, never the contract's text or a value: a caller
        # reads the code (the text reached an answer as an untyped failure).
        raise ResearchWorkspaceError(
            public_failure(error, "research_workspace.manifest_refused")
        ) from error
    except ValueError as error:
        raise ResearchWorkspaceError("research_workspace.manifest_refused:manifest") from error


def resolve_workspace_model_lifecycle(
    workspace: Path,
    *,
    component_id: str,
    artifact_root: Path | None = None,
    training_authority_hash: str | None = None,
) -> tuple[AlphaCurrentArtifactStore, AlphaModelLifecycleAdmission]:
    """Resolve one installed training source for compiler and authority consumers."""
    from alphalattice.investment.alpha_research.publication.artifacts import (
        AlphaCurrentArtifactStore,
    )
    from alphalattice.investment.alpha_research.scores.model_renewal import (
        AlphaTrainingObservations,
        admit_component_inference,
        read_lifecycle_admission,
    )

    root = workspace.resolve()
    artifacts = (artifact_root or root / "artifacts").resolve()
    if not artifacts.is_relative_to(root):
        raise ResearchWorkspaceError("research_workspace.training_artifacts_outside_workspace")
    matches: dict[str, AlphaModelLifecycleAdmission] = {}
    store = AlphaCurrentArtifactStore(artifacts)
    manifest = read_research_workspace_manifest(root)
    bindings = (
        tuple(
            v
            for v in manifest.model_training_inputs or ()
            if v.component_id == component_id and v.authority_hash == training_authority_hash
        )
        if training_authority_hash is not None
        else manifest.score_inputs or ()
    )
    for binding in bindings:
        directory = (root / binding.authority_relative_path).resolve()
        if not directory.is_relative_to(root):
            raise ResearchWorkspaceError("research_workspace.training_source_outside_workspace")
        if directory.is_file() or (directory / "model-lifecycle-admission.json").is_file():
            value = read_lifecycle_admission(directory, expected_hash=binding.authority_hash)
            if value.component.component_id == component_id:
                if isinstance(binding, ResearchWorkspaceModelTrainingInput):
                    admit_component_inference(
                        directory, store=store, expected_hash=binding.authority_hash
                    )
                    observations = store._load(
                        "lifecycle-training-observations",
                        value.observations_hash,
                        "content_hash",
                        AlphaTrainingObservations,
                    )
                    snapshot = store.load_frozen_observation_snapshot(observations.observation_hash)
                    if snapshot.source_binding_hash != binding.source_identity_hash:
                        raise ResearchWorkspaceError(
                            "research_workspace.training_source_identity_mismatch"
                        )
                matches[value.content_hash] = value
    if len(matches) != 1:
        raise ResearchWorkspaceError("research_workspace.training_source_unresolved")
    return store, next(iter(matches.values()))


def admit_research_workspace(workspace: Path) -> AdmittedResearchWorkspace:
    """Read the manifest, confine its paths and install its declared catalog."""
    root = workspace.resolve()
    manifest = read_research_workspace_manifest(root)
    if manifest.strategy_installation == "NOT_INSTALLED":
        return AdmittedResearchWorkspace(root=root, manifest=manifest, catalog=None)
    artifacts: dict[str, Path] = {}
    for binding in manifest.strategy_artifacts:
        resolved = (root / Path(*PurePosixPath(binding.relative_path).parts)).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ResearchWorkspaceError("research_workspace.artifact_escapes_workspace") from error
        if not resolved.exists():
            raise ResearchWorkspaceError(
                f"research_workspace.artifact_unreadable:{binding.artifact_key}"
            )
        artifacts[binding.artifact_key] = resolved
    try:
        bindings = install_frozen_strategies(artifacts=artifacts)
        catalog = InstalledFrozenStrategyCatalog(
            bindings, default_strategy_id=manifest.default_strategy_package_id
        )
        if manifest.default_strategy_package_id is not None:
            catalog.select(
                strategy_id=manifest.default_strategy_package_id,
                score_source_mode=manifest.default_score_source_mode,
            )
    except ValueError as error:
        raise ResearchWorkspaceError(f"research_workspace.catalog_refused:{error}") from error
    return AdmittedResearchWorkspace(root=root, manifest=manifest, catalog=catalog)


__all__ = [
    "RESEARCH_WORKSPACE_MANIFEST_NAME",
    "AdmittedResearchWorkspace",
    "ResearchWorkspaceArtifact",
    "ResearchWorkspaceError",
    "ResearchWorkspaceEvidenceReview",
    "ResearchWorkspaceExperimentInput",
    "ResearchWorkspaceManifest",
    "ResearchWorkspaceModelTrainingInput",
    "admit_research_workspace",
    "component_training_selection",
    "create_research_workspace_manifest",
    "manifest_fields_hash",
    "publish_research_workspace_manifest",
    "read_research_workspace_manifest",
    "update_research_workspace_manifest",
]
