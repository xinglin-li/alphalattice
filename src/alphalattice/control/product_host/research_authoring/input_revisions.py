"""Verified input publications, extending the existing bundle publication directory."""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.data_platform.maintenance.contracts import WorkspaceDataUpdateBinding
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceExperimentInput,
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.maintenance.data_update import read_workspace_inputs
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    confined,
    read_factor_bundle,
)
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.causal_outcomes.execution.methods import (
    ONE_SESSION_RECIPE_ID,
    build_installed_execution_outcome_method_catalog,
    build_installed_execution_outcome_publication_policy_catalog,
)
from alphalattice.foundation.feature_engine.contracts import panel_source_manifest_revision
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.identity_successors import recorded_origin
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.sealing import seal_model
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash

CAPTURE_TASK_KIND = "research_input_capture"

_PUBLICATION_FAILURES = frozenset(
    {
        "research_input.publication_invalid",
        "research_input.publication_task_mismatch",
        "research_input.publication_source_mismatch",
        "research_input.publication_evidence_mismatch",
        "research_input.publication_unavailable",
        "research_input.publication_ambiguous",
        "research_input.publication_parent_mismatch",
        "research_input.publication_ancestor_missing",
    }
)


MATERIALIZER_ROLE = "product_host.research_input_materializer"
"""The role whose recorded moves the capture's materializer identity follows."""


def materializer_identity() -> str:
    """The capture's materializer identity, held at its recorded origin.

    A move recorded as keeping the capture leaves a plan sealed before it current.

    Returns:
        The identity a captured source binds.
    """
    return recorded_origin(MATERIALIZER_ROLE, materializer_rule_value())


def materializer_rule_value() -> str:
    """The capture's materializer closure as its files hash now: the role's readout value."""
    return source_rule_closure_hash(
        root=resolve_playpen_root(Path(__file__)),
        semantic_owner="product_host",
        numerical_role="RESEARCH_INPUT_CAPTURE",
        tracked_paths=tuple(
            "src/alphalattice/" + p
            for p in (
                "control/product_host/research_authoring/factor_inputs.py",
                "control/product_host/research_authoring/input_revisions.py",
                "foundation/causal_outcomes/execution/publication.py",
                "foundation/causal_outcomes/execution/methods.py",
                "foundation/feature_engine/panels/logical_identity.py",
            )
        ),
    )


class ResearchInputSource(BaseModel):  # type: ignore[misc]
    """Retain exact local Data, Panel, outcome and materializer source identities."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    binding_hash: str
    manifest_revision: str
    data_revision_hash: str
    panel_snapshot_hash: str
    panel_through: date
    outcome_watermark_hash: str
    outcome_recipe_hash: str
    outcome_policy_hash: str
    materializer_hash: str

    @property
    def source_key(self) -> str:
        """Bind the complete declared local input source projection.

        Returns:
            Canonical source-key identity over all serialized source fields.
        """
        return str(canonical_hash(self.model_dump(mode="json")))


def describe_input_source(root: Path, binding: WorkspaceDataUpdateBinding) -> ResearchInputSource:
    """Use governed source journals, not task timestamps or physical DB layout."""
    value = read_workspace_inputs(root, binding)
    if value.readiness_status != "RESEARCH_READY":
        raise ValueError("research_input.qualified_data_update_required")
    market = MarketDataRepository(root)
    resolver = ArtifactResolver(root / "artifacts")
    panel_ref = resolver.feature_panel_manifest_uri(value.panel_hash)
    panel = resolver.load_feature_panel_manifest(panel_ref)
    manifest = market.load_universe_manifest_revision(panel_source_manifest_revision(panel))
    watermark = market.execution_source_watermark(
        manifest,
        through=value.panel_through,
        listing_ids=FeaturePanelReader(resolver).listing_axis(panel_ref),
    )
    recipe = build_installed_execution_outcome_method_catalog().resolve(ONE_SESSION_RECIPE_ID)
    policy = build_installed_execution_outcome_publication_policy_catalog().resolve(
        recipe.recipe_id
    )
    policy.admit(recipe)
    return ResearchInputSource(
        binding_hash=binding.content_hash,
        manifest_revision=value.manifest_revision,
        data_revision_hash=value.data_revision_hash,
        panel_snapshot_hash=value.panel_hash,
        panel_through=value.panel_through,
        outcome_watermark_hash=str(watermark["watermark_hash"]),
        outcome_recipe_hash=recipe.recipe_hash,
        outcome_policy_hash=policy.policy_hash,
        materializer_hash=materializer_identity(),
    )


class ResearchInputRevision(BaseModel):  # type: ignore[misc]
    """Seal one human-selected input revision with exact parent, capture task and source lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["ResearchInputRevision"] = "ResearchInputRevision"
    input_id: str
    anchor_binding_hash: str
    prior_binding_hash: str
    binding_hash: str
    previous_publication_hash: str | None
    source: ResearchInputSource
    task_id: UUID
    task_input_hash: str
    chosen_by: Literal["HUMAN"] = "HUMAN"
    published_at: datetime
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal an explicit captured research input revision.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical receipt_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return seal_model(cls, values, field="receipt_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def verify(self) -> Self:
        """Require aware publication time and exact revision receipt identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Publication timestamp is naive or canonical receipt identity differs.
        """
        if self.published_at.tzinfo is None or self.receipt_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        ):
            raise ValueError("research_input.publication_invalid")
        return self


class ResearchInputRevisions:
    """Read/write one publication family; filesystem presence alone admits nothing."""

    def __init__(self, session: WorkspaceApplicationSession):
        """Wire retained task authority and confined input revision publication storage.

        Args:
            session: Retained workspace writer/task session.
        """
        self.session = session
        self.workspace = session.workspace
        self.root = confined(self.workspace, "research-inputs/publications")

    def publications(self) -> tuple[dict[str, Any], ...]:
        """Read every retained publication and require its exact capture-task/source binding.

        Returns:
            Ordered validated publication metadata; selected bundle byte verification remains
            separate.

        Raises:
            ValueError: Publication path/identity, task/plan/source or succeeded-stage receipt
                evidence differs.
        """
        return tuple(self._publication(path) for path in sorted(self.root.glob("*.json")))

    def _publication(self, path: Path) -> dict[str, Any]:
        """Read and verify one publication with the same strict admission checks.

        Args:
            path: One candidate publication file in the confined publication directory.

        Returns:
            Validated publication mapping.

        Raises:
            OSError: The publication or one of its exact dependencies could not be read.
            ValueError: Its identity, task, source or succeeded-stage evidence differs.
        """
        body = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(body, dict):
            raise ValueError("research_input.publication_invalid")
        identity = body.get("receipt_hash")
        if (
            path.stem != identity
            or canonical_hash({k: v for k, v in body.items() if k != "receipt_hash"}) != identity
        ):
            raise ValueError("research_input.publication_invalid")
        if body.get("kind") is not None:
            revision = ResearchInputRevision.model_validate(body)
            try:
                task = self.session.task_control_registry.task(revision.task_id)
            except KeyError:
                raise ValueError("research_input.publication_task_mismatch") from None
            plan = task.input.payload.get("plan", {})
            if (
                task.task_kind != CAPTURE_TASK_KIND
                or task.input.input_hash != revision.task_input_hash
                or plan.get("input_id") != revision.input_id
                or plan.get("anchor_binding_hash") != revision.anchor_binding_hash
                or plan.get("prior_binding_hash") != revision.prior_binding_hash
                or plan.get("previous_publication_hash") != revision.previous_publication_hash
                or plan.get("source") != revision.source.model_dump(mode="json")
            ):
                raise ValueError("research_input.publication_task_mismatch")
            bundle = read_factor_bundle(self.workspace, revision.binding_hash, verify=False)
            if bundle.panel_snapshot_hash != revision.source.panel_snapshot_hash:
                raise ValueError("research_input.publication_source_mismatch")
            if task.lifecycle is TaskLifecycle.SUCCEEDED and not any(
                evidence.content_hash == revision.receipt_hash
                and evidence.evidence_kind == "research_input.publication"
                for receipt in self.session.task_control_registry.stage_receipts(task.task_id)
                if receipt.status == "VERIFIED"
                for evidence in receipt.evidence
            ):
                raise ValueError("research_input.publication_evidence_mismatch")
        return body

    def _discover_publications(
        self,
    ) -> tuple[tuple[dict[str, Any], ...], list[dict[str, object]]]:
        """Keep one unreadable retained file from hiding other verified publications.

        Scientific selection continues to use :meth:`publications`, which remains strict.
        This per-record isolation is only for the descriptive versions collection.
        """
        values: list[dict[str, Any]] = []
        refusals: list[dict[str, object]] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                values.append(self._publication(path))
            except OSError:
                failure_code = "research_input.publication_unavailable"
            except (TypeError, ValueError, KeyError) as error:
                candidate = str(error).split(":", 1)[0]
                failure_code = (
                    candidate
                    if candidate in _PUBLICATION_FAILURES
                    else "research_input.publication_invalid"
                )
            else:
                continue
            publication_hash = path.stem
            refusals.append(
                {
                    "status": "REFUSED",
                    "publication_hash": publication_hash,
                    "failure_code": failure_code,
                    "detail": (
                        f"Retained research input publication {publication_hash} could not be "
                        f"verified ({failure_code}). Its declared input and whether its version "
                        "can be chosen remain unknown; this list does not establish absence. "
                        "Read RESEARCH_INPUTS again, then inspect the workspace, retained "
                        "backups, and stored files before choosing an input version."
                    ),
                    "next_requests": self._publication_next_requests(),
                }
            )
        return tuple(values), refusals

    @staticmethod
    def _publication_next_requests() -> dict[str, dict[str, str]]:
        """Return the accepted discovery and recovery reads for a damaged publication."""
        return {
            "inputs": {"operation": "RESEARCH_INPUTS"},
            "workspace": {"operation": "WORKSPACE_SHOW"},
            "backups": {"operation": "WORKSPACE_BACKUPS"},
            "storage": {"operation": "STORAGE_READBACK"},
        }

    @staticmethod
    def _lineage_failure_code(error: Exception) -> str:
        candidate = str(error).split(":", 1)[0]
        return (
            candidate
            if candidate in _PUBLICATION_FAILURES
            else "research_input.publication_invalid"
        )

    def anchor(self, input_id: str) -> ResearchWorkspaceExperimentInput:
        """Resolve the unique input anchor explicitly declared by the workspace manifest.

        Args:
            input_id: Exact declared research input.

        Returns:
            Declared anchor binding.

        Raises:
            ValueError: Input is unregistered or not uniquely declared.
        """
        manifest = read_research_workspace_manifest(self.workspace)
        matches = [v for v in manifest.experiment_inputs or () if v.input_id == input_id]
        if len(matches) != 1:
            raise ValueError("research_input.unregistered_input")
        return matches[0]

    def lineage(self, input_id: str) -> tuple[ResearchInputRevision, ...]:
        """Read exact admitted revision lineage from a declared input anchor.

        Args:
            input_id: Exact declared research input.

        Returns:
            Ordered validated revision lineage.
        """
        return self._lineage_from_verified(self.anchor(input_id), self.publications())

    @staticmethod
    def _lineage_from_verified(
        anchor: ResearchWorkspaceExperimentInput,
        publications: tuple[dict[str, Any], ...],
    ) -> tuple[ResearchInputRevision, ...]:
        """Resolve a lineage from publications verified once for this operation."""
        rows = tuple(
            ResearchInputRevision.model_validate(v)
            for v in publications
            if v.get("kind") == "ResearchInputRevision"
            and v["input_id"] == anchor.input_id
            and v["anchor_binding_hash"] == anchor.binding_hash
        )
        ordered = []
        previous, binding = None, anchor.binding_hash
        while True:
            children = [v for v in rows if v.previous_publication_hash == previous]
            if not children:
                break
            if len(children) != 1 or children[0] in ordered:
                raise ValueError("research_input.publication_ambiguous")
            child = children[0]
            if child.prior_binding_hash != binding:
                raise ValueError("research_input.publication_parent_mismatch")
            ordered.append(child)
            previous, binding = child.receipt_hash, child.binding_hash
        if len(ordered) != len(rows):
            raise ValueError("research_input.publication_ancestor_missing")
        return tuple(ordered)

    def select(
        self,
        input_id: str,
        binding_hash: str | None = None,
        *,
        verify: bool = True,
    ) -> ResearchWorkspaceExperimentInput:
        """Select only an explicit declared anchor or succeeded admitted revision.

        Args:
            input_id: Exact declared research input.
            binding_hash: Optional exact revision; defaults to the declared anchor.
            verify: Whether to verify the selected bundle's complete byte set.

        Returns:
            Explicit selected input binding; no implicit latest revision is substituted.

        Raises:
            ValueError: Revision is not admitted or its publication is not completed.
        """
        anchor = self.anchor(input_id)
        requested = binding_hash or anchor.binding_hash
        if requested != anchor.binding_hash:
            matched = [v for v in self.lineage(input_id) if v.binding_hash == requested]
            if not matched:
                raise ValueError("research_input.version_not_admitted")
            if not any(
                self.session.task_control_registry.task(v.task_id).lifecycle
                is TaskLifecycle.SUCCEEDED
                for v in matched
            ):
                raise ValueError("research_input.publication_completion_required")
        read_factor_bundle(self.workspace, requested, verify=verify)
        return ResearchWorkspaceExperimentInput(input_id=input_id, binding_hash=requested)

    def for_task(self, task_id: UUID) -> ResearchInputRevision | None:
        """Resolve the unique revision publication belonging to one exact capture task.

        Args:
            task_id: Exact retained capture task.

        Returns:
            Validated revision or None without a publication.

        Raises:
            ValueError: Multiple publications claim the same task.
        """
        values = [
            ResearchInputRevision.model_validate(v)
            for v in self.publications()
            if v.get("kind") == "ResearchInputRevision" and v["task_id"] == str(task_id)
        ]
        if len(values) > 1:
            raise ValueError("research_input.publication_ambiguous")
        return values[0] if values else None

    def publish(self, revision: ResearchInputRevision) -> None:
        """Record one publication of a bundle read whole, here, first.

        The bundle the revision names is verified -- every file present and
        matching its digest, the sealed database matching its snapshot hash
        -- before the receipt is written; a bundle that fails leaves no
        receipt and moves no lineage. There is no way to record a revision
        without this read.
        """
        with self.session.mutation_gate.hold():
            previous = self.for_task(revision.task_id)
            if previous is not None:
                if previous != revision:
                    raise ValueError("research_input.publication_conflict")
                return
            rows = self.lineage(revision.input_id)
            parent = rows[-1] if rows else None
            if revision.previous_publication_hash != (parent.receipt_hash if parent else None):
                raise ValueError("research_input.publication_plan_stale")
            if revision.prior_binding_hash != (
                parent.binding_hash if parent else revision.anchor_binding_hash
            ):
                raise ValueError("research_input.publication_parent_mismatch")
            if revision.anchor_binding_hash != self.anchor(revision.input_id).binding_hash:
                raise ValueError("research_input.publication_plan_stale")
            read_factor_bundle(self.workspace, revision.binding_hash)
            self.root.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=self.root, delete=False) as stream:
                stream.write(revision.model_dump_json().encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
                temporary = Path(stream.name)
            os.replace(temporary, self.root / f"{revision.receipt_hash}.json")
            if self.for_task(revision.task_id) != revision:
                raise ValueError("research_input.publication_readback_failed")

    def versions(self) -> dict[str, object]:
        """List explicit input versions and report each unreadable revision with its reason.

        Every input's versions. One that cannot be read is listed as such, with what is
        missing and where to look, and the others still list (CLI-3).
        """
        groups = []
        refusals: list[dict[str, object]] = []
        anchors = read_research_workspace_manifest(self.workspace).experiment_inputs or ()
        # One Task Control connection for every revision's lifecycle (binding plan, N8).
        with self.session.task_control_registry.retain(read_only=True):
            publications, refusals = self._discover_publications()
            publication_refusals = list(refusals)
            for anchor in anchors:
                try:
                    rows = self._lineage_from_verified(anchor, publications)
                except (TypeError, ValueError, KeyError) as error:
                    failure_code = self._lineage_failure_code(error)
                    refusals.append(
                        {
                            "status": "REFUSED",
                            "input_id": anchor.input_id,
                            "failure_code": failure_code,
                            "detail": (
                                f"The retained versions for declared input {anchor.input_id} "
                                f"could not be verified ({failure_code}). Only its declared "
                                "default is shown; no retained revision is inferred absent or "
                                "selectable. Inspect the workspace, retained backups, and "
                                "stored files before choosing a version."
                            ),
                            "next_requests": self._publication_next_requests(),
                        }
                    )
                    anchor_version = self._version(anchor.binding_hash, None)
                    values = [
                        {
                            **anchor_version,
                            "available": False,
                            "unreadable": failure_code,
                            "detail": (
                                f"The retained versions for declared input {anchor.input_id} "
                                f"could not be verified ({failure_code}); this default cannot be "
                                "chosen until the input's version history is read successfully."
                            ),
                            "next_requests": self._publication_next_requests(),
                        }
                    ]
                else:
                    versions = [(anchor.binding_hash, None), *((v.binding_hash, v) for v in rows)]
                    values = [self._version(binding, revision) for binding, revision in versions]
                    if publication_refusals and not rows:
                        failure_code = str(publication_refusals[0]["failure_code"])
                        values[0] = {
                            **values[0],
                            "available": False,
                            "unreadable": failure_code,
                            "detail": (
                                f"The retained versions for declared input {anchor.input_id} "
                                f"could not be verified ({failure_code}); this default cannot be "
                                "chosen until the input's version history is read successfully."
                            ),
                            "next_requests": self._publication_next_requests(),
                        }
                        refusals.append(
                            {
                                "status": "REFUSED",
                                "input_id": anchor.input_id,
                                "failure_code": failure_code,
                                "detail": (
                                    f"The retained versions for declared input {anchor.input_id} "
                                    f"could not be verified ({failure_code}). Only its declared "
                                    "default is shown; no retained revision is inferred absent "
                                    "or selectable. Inspect the workspace, retained backups, "
                                    "and stored files before choosing a version."
                                ),
                                "next_requests": self._publication_next_requests(),
                            }
                        )
                    elif publication_refusals:
                        # Exact selection of a captured revision remains strict over the whole
                        # publication set, even if this validated prefix is still listable.
                        failure_code = str(publication_refusals[0]["failure_code"])
                        values = [
                            values[0],
                            *[
                                {
                                    **value,
                                    "available": False,
                                    "unreadable": failure_code,
                                    "detail": (
                                        f"This retained version cannot be selected because a "
                                        f"research input publication could not be verified "
                                        f"({failure_code}). The declared default remains a "
                                        "separate exact choice; inspect the workspace, retained "
                                        "backups, and stored files before choosing this version."
                                    ),
                                    "next_requests": self._publication_next_requests(),
                                }
                                for value in values[1:]
                            ],
                        ]
                groups.append({"input_id": anchor.input_id, "versions": values})
        result: dict[str, object] = {"status": "AVAILABLE", "inputs": groups}
        if refusals:
            result["refusals"] = refusals
        return result

    def _version(self, binding: str, revision: Any) -> dict[str, object]:
        head: dict[str, object] = {
            "binding_hash": binding,
            "configured_default": revision is None,
            "publication_hash": revision.receipt_hash if revision else None,
        }
        try:
            bundle = read_factor_bundle(self.workspace, binding, verify=False)
            lifecycle = (
                self.session.task_control_registry.task(revision.task_id).lifecycle.value
                if revision
                else "REGISTERED"
            )
        except (OSError, ValueError, KeyError) as error:
            return {
                **head,
                "available": False,
                "unreadable": "research_input.manifest_missing"
                if isinstance(error, FileNotFoundError)
                else "research_input.bundle_unreadable",
                "detail": (
                    "This version's prepared input cannot be read in this workspace (its "
                    "manifest or one of its files is missing or changed), so it cannot be "
                    "chosen; the other versions are unaffected. Storage names what it holds."
                ),
                "next_requests": {"storage": {"operation": "STORAGE_READBACK"}},
            }
        source = self.workspace / "research-inputs" / binding / "source"
        return {
            **head,
            "start": str(bundle.sessions[0]),
            "end": str(bundle.sessions[-1]),
            "lifecycle": lifecycle,
            "available": all((source / name).is_file() for name, _digest in bundle.files),
        }
