"""Project real research-input references into the existing retention owner."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from alphalattice.control.data_platform.maintenance.registry import (
    DuckDbWorkspaceMaintenanceRegistry,
)
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.evidence_review_workspace import (
    EvidenceReviewWorkspaceManifest,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceError,
    ResearchWorkspaceManifest,
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.strategy_scoring import (
    workspace_observation_history_scope,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    confined,
    factor_input_paths,
    read_factor_bundle,
)
from alphalattice.control.product_host.research_authoring.input_revisions import (
    ResearchInputRevisions,
)
from alphalattice.control.product_host.storage.contracts import (
    PhysicalAvailability,
    storage_input_estimate,
)
from alphalattice.control.product_host.storage.evidence_references import EvidenceStorageReferences
from alphalattice.control.product_host.storage.inventory import (
    MANAGED_ROOTS,
    RECOVERY_HEADROOM_BYTES,
    managed_file_inventory,
    require_storage_capacity,
    unique_managed_bytes,
    workspace_storage_capacity,
)
from alphalattice.control.product_host.storage.retention import (
    BoundedStorageRetentionOwner,
    StorageRetentionError,
    file_content_hash,
    pending_input_cleanups,
    require_no_pending_cleanup,
)
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.storage.capacity import StorageCapStore
from alphalattice.evidence.alternative_evidence.retrieval.contracts import (
    RETRIEVAL_GENERATION_FORMAT,
)
from alphalattice.evidence.alternative_evidence.storage.inventory import EvidenceStorageBinding
from alphalattice.foundation.causal_outcomes.execution.artifacts import (
    local_qa_preparation_retention_inventory,
)
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.foundation.feature_engine.panels.retention_residue import (
    PanelRetentionResidueOwner,
)
from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.foundation.research_foundation.runtime.delta import VerifiedPreResearchHead
from alphalattice.investment.alpha_research.publication.artifacts import AlphaCurrentArtifactStore
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.knowledge import model_store
from alphalattice.kernel.knowledge.hybrid_contracts import RECIPE_MINILM_CPU
from alphalattice.kernel.shared_kernel.identity import canonical_hash


def _strings(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set().union(*(_strings(v) for v in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(_strings(v) for v in value)) if value else set()
    return set()


def bytes_by_root(files: tuple[tuple[str, int, str], ...]) -> dict[str, dict[str, int]]:
    """Count physical and logical bytes by managed role from one exact inventory.

    Physical (each filesystem object once) and logical (each reference) bytes and the
    file count under every managed role, from one inventory walk. An object shared by two
    roots is counted physically under the first root that names it, so the roles sum to
    the managed total; a role's roots are summed.
    """
    seen: set[str] = set()
    values: dict[str, dict[str, int]] = {}
    for prefix, role in MANAGED_ROOTS:
        held = [v for v in files if v[0] == prefix or v[0].startswith(prefix + "/")]
        physical = 0
        for _name, size, identity in held:
            if identity in seen:
                continue
            seen.add(identity)
            physical += size
        value = values.setdefault(role, {"physical_bytes": 0, "logical_bytes": 0, "files": 0})
        value["physical_bytes"] += physical
        value["logical_bytes"] += sum(v[1] for v in held)
        value["files"] += len(held)
    return values


def _display_bytes(value: int) -> str:
    for unit, size in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if value >= size:
            return f"{value / size:,.2f} {unit} ({value:,} bytes)"
    return f"{value:,} bytes"


EVIDENCE_PIN_PREFIX = "evidence-index:"


class ResearchInputStorage:
    """An application projection; policy and deletion remain with retention.

    Evidence retrieval generations are accounted and released through the
    same owner when the workspace has an evidence store: their index payloads
    are candidates, their vector payloads, source blobs and sealed artifacts
    are retained, and a generation an unfinished Task, an open reader or a
    person still needs is protected.
    """

    def __init__(
        self,
        session: WorkspaceApplicationSession,
        *,
        evidence: EvidenceStorageBinding | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        """Wire bounded retention, exact evidence storage and observed-time ownership.

        Wire bounded retention ownership, exact optional evidence storage and observed-time source.

        Args:
            session: Retained workspace writer/task session.
            evidence: Optional exact evidence storage binding.
            clock: Optional explicit observed-time source.
        """
        self.session = session
        self.workspace = session.workspace
        self.owner = BoundedStorageRetentionOwner(self.workspace)
        self.evidence = evidence
        self.clock = clock or (lambda: datetime.now(UTC))
        self._capacity_sampled = False

    def evidence_index_pinned(self, index_id: str) -> bool:
        """Say whether the person pinned a generation's index (`storage pin`).

        A pinned index is not retired after its use; an unreadable pin record keeps it and
        never reads as unpinned.

        Args:
            index_id: The generation's index identity.

        Returns:
            True when the index is pinned or the pin record cannot be read.
        """
        try:
            return f"{EVIDENCE_PIN_PREFIX}{index_id}" in self._all_pins()
        except (OSError, ValueError, KeyError, TypeError, StorageRetentionError):
            return True

    def retire_evidence_index(self, index_id: str, retire: Callable[[], object]) -> bool:
        """Retire a generation's index unless the person pinned it, under the mutation gate.

        `pin` checks and writes under the same gate, so a pin never lands between this check
        and the removal.

        Args:
            index_id: The generation's index identity.
            retire: The index owner's removal, run only when the index is not pinned.

        Returns:
            False when the index is pinned or its pin record cannot be read; True once retired.

        Raises:
            GateBusy: A foreground caller found the gate held throughout its wait.
        """
        with self.session.mutation_gate.hold():
            if self.evidence_index_pinned(index_id):
                return False
            retire()
            return True

    def _all_pins(self) -> dict[str, int]:
        path = self.owner.governance_root / "input-pins.json"
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        receipt = self.owner.governance_root / "input-pin-receipts" / f"{data['receipt_hash']}.json"
        if (
            json.loads(receipt.read_text(encoding="utf-8")) != data["pins"]
            or canonical_hash(data["pins"]) != data["receipt_hash"]
        ):
            raise StorageRetentionError(
                "storage.retention_root_mismatch", "Input pin receipt is invalid."
            )
        return dict(data["pins"])

    def _evidence_pins(self) -> frozenset[str]:
        return frozenset(
            key[len(EVIDENCE_PIN_PREFIX) :]
            for key in self._all_pins()
            if key.startswith(EVIDENCE_PIN_PREFIX)
        )

    def _evidence_references(self) -> EvidenceStorageReferences | None:
        if self.evidence is None:
            return None
        return EvidenceStorageReferences(
            workspace=self.workspace,
            binding=self.evidence,
            registry=self.session.task_control_registry,
            pinned_index_ids=self._evidence_pins(),
        )

    def _references(self) -> dict[str, Any]:
        """Every fact a cleanup plan depends on, whose hash stales a plan when one changes. The
        research inputs are not among them: they are kept once on the data root behind the
        door (kernel.data_layer.inputs), held while a binding names them, outside this plan."""
        manifest = read_research_workspace_manifest(self.workspace)
        tasks = [
            (str(task.task_id), task.input.input_hash, task.lifecycle.value)
            for task in self.session.task_control_registry.tasks()
            if task.lifecycle not in {TaskLifecycle.SUCCEEDED, TaskLifecycle.CANCELLED}
        ]
        evidence: dict[str, Any] = {
            "manifest": manifest.manifest_hash,
            "tasks": tasks,
            "source": self._source_references(busy=bool(tasks)),
        }
        preparation = self._preparation_references(manifest, busy=bool(tasks))
        evidence["source"]["targets"].update(preparation["targets"])
        evidence["source"]["preparation"] = preparation
        references = self._evidence_references()
        if references is not None:
            # Every fact a plan over evidence indexes depends on: a new pin, an
            # open reader or an unfinished Task changes the hash and stales it.
            evidence["evidence_indexes"] = references.evidence()
        return evidence

    def _preparation_references(
        self, manifest: ResearchWorkspaceManifest, *, busy: bool
    ) -> dict[str, Any]:
        """Project owner-verified preparation roots into the governed cleanup.

        Scientific observation/score/report files stay outside these narrow
        namespaces. Current, previous and explicitly referenced preparation
        heads keep their parts. An unfinished Task protects all candidates,
        including parts whose marker has not yet been committed.
        """
        artifacts = self.workspace / "artifacts"
        store = AlphaCurrentArtifactStore(artifacts)
        scopes = tuple(
            sorted(
                workspace_observation_history_scope(binding)
                for binding in manifest.score_inputs or ()
                if binding.source_kind == "WORKSPACE_DATA_FEATURE"
            )
        )
        alpha = store.workspace_observation_history_retention(active_scope_hashes=scopes)
        causal = local_qa_preparation_retention_inventory(artifacts)
        causal_heads = {
            p.stem
            for p in (
                artifacts / "data-operations/execution-outcomes/local-qa-preparation/heads"
            ).glob("*.json")
        }
        known = set(alpha.head_hashes) | causal_heads
        # Preserve explicit head references if a retained manifest or pin names
        # one. The scientific current products do not need their input cache.
        references = _strings(manifest.model_dump(mode="json")) | set(self._all_pins())
        heads = tuple(sorted(known & references))
        if heads:
            alpha = store.workspace_observation_history_retention(
                active_scope_hashes=scopes,
                referenced_heads=tuple(h for h in heads if h in alpha.head_hashes),
            )
            causal = local_qa_preparation_retention_inventory(
                artifacts,
                referenced_heads=tuple(h for h in heads if h in causal_heads),
            )
        retained = (
            *((f.path, f.sha256, f.bytes) for f in alpha.roots),
            *(
                (artifacts / f.relative_path, f.file_hash, f.byte_count)
                for f in causal.protected_files
            ),
        )
        candidates = (
            *((f.path, f.sha256) for f in alpha.targets),
            *((artifacts / f.relative_path, f.file_hash) for f in causal.candidate_files),
        )
        return {
            "roots": tuple(
                sorted(
                    (path.relative_to(self.workspace).as_posix(), digest, size)
                    for path, digest, size in retained
                )
            ),
            "heads": heads,
            "active_scopes": scopes,
            "busy": busy,
            "targets": {}
            if busy
            else {
                path.relative_to(self.workspace).as_posix(): digest for path, digest in candidates
            },
        }

    def _source_references(self, *, busy: bool) -> dict[str, Any]:
        """Release classified completed snapshots, never raw history or recovery patches.

        A generation no research input captured is kept while it is the current one or the
        previous one; past the previous it is superseded, and the chunks no kept generation
        names are released under the confirmed plan, a composition's that no generation
        published among them.
        """
        artifacts = self.workspace / "artifacts"
        panel_root = artifacts / "feature-panel"
        if not tuple((panel_root / "manifests").glob("*.json")):
            return {"targets": {}, "roots": []}
        # A research input keeps its own copy of what it captured on the data root, so no
        # workspace generation is known to an input: none is released for one.
        protected_panels: set[str] = set()
        protected_outcomes: set[str] = set()
        panel_hashes = {p.stem for p in (panel_root / "manifests").glob("*.json")}
        market = MarketDataRepository(self.workspace)
        active = PanelStateRepository(
            market.database, market_data=market
        ).feature_panel_snapshot_for_active("us-current-index-research")
        if active is None:
            if busy:
                return {"targets": {}, "roots": sorted(panel_hashes), "status": "PREPARING"}
            raise ValueError("storage.current_panel_root_unavailable")
        current = str(active["snapshot_hash"])
        protected_panels.add(current)
        receipt = DuckDbWorkspaceMaintenanceRegistry.read_data_update_receipt(market.path)
        if receipt is None:
            # Without a data update's receipt no generation is known to precede the current.
            protected_panels.update(panel_hashes)
        elif receipt.after.panel_hash == current:
            protected_panels.add(receipt.before.panel_hash)
        else:
            protected_panels.update(panel_hashes)
        # Formal Foundation/PM history is retained: this task grants no retirement of it.
        heads = []
        for path in sorted((artifacts / "pre-research-revisions/heads").glob("*.json")):
            head = VerifiedPreResearchHead.model_validate_json(path.read_bytes())
            if head.head_hash != path.stem:
                raise ValueError("storage.retention_root_mismatch")
            protected_panels.add(head.panel_snapshot_hash)
            protected_outcomes.add(head.causal_execution_outcome_snapshot_hash)
            heads.append(head.head_hash)
        if busy:
            protected_panels.update(panel_hashes)
        resolver = ArtifactResolver(artifacts)
        protected: set[Path] = set()
        for panel_hash in protected_panels:
            panel = resolver.load_feature_panel_manifest(
                resolver.feature_panel_manifest_uri(panel_hash)
            )
            protected.update(
                panel_root / "chunks" / f"{chunk['chunk_hash']}.parquet"
                for chunk in panel["chunks"]
            )
        outcomes = CausalExecutionOutcomeDevelopmentReader(artifacts).artifacts
        for outcome in (*outcomes.manifests(), *outcomes.development_only_manifests()):
            protected.update(
                outcomes._parquet_path(c.split, c.content_hash) for c in outcome.development_chunks
            )
        targets = {}
        if not busy:
            # The chunks no kept generation names: a superseded generation's own, and those a
            # composition wrote that no generation published (11 of `dh-append`'s 24).
            for path in sorted((panel_root / "chunks").glob("*.parquet")):
                if path not in protected:
                    targets[path.relative_to(self.workspace).as_posix()] = file_content_hash(path)
        return {
            "targets": dict(sorted(targets.items())),
            "roots": sorted(protected_panels),
            "foundation_heads": heads,
            "outcome_roots": sorted(protected_outcomes),
            "data_receipt": receipt.content_hash if receipt else None,
        }

    def readback(self) -> dict[str, Any]:
        """Read exact input/evidence roots, physical/logical bytes and pending cleanup state.

        Returns:
            Input availability, protected evidence indexes, budget/capacity/free disk and
            recovery-required cleanup facts; no cleanup is applied.
        """
        evidence = self._references()
        files = managed_file_inventory(self.workspace)
        used = unique_managed_bytes(files)
        capacity = StorageCapStore(self.workspace).capacity(measured_data_bytes=used)
        pending = pending_input_cleanups(self.workspace)
        free = shutil.disk_usage(self.workspace).free
        references = self._evidence_references()
        return {
            "status": "RECOVERY_REQUIRED" if pending else "AVAILABLE",
            "inputs": [],  # deprecated in 0.2.8: inputs are kept on the data root
            "evidence": None
            if references is None
            else {
                **references.summary(),
                "indexes": [
                    {
                        **body,
                        "available_actions": self._index_actions(body, references, pending),
                        "rebuild_limit": None
                        if body["generation_format"] == RETRIEVAL_GENERATION_FORMAT
                        else (
                            "This legacy index has no committed vector payload "
                            "and cannot be rebuilt. "
                            "Start a new Evidence preparation."
                        ),
                    }
                    for body in (row.body() for row in references.rows())
                ],
                "unreferenced": references.unreferenced(),
                "pinned_index_ids": sorted(references.pinned),
            },
            "managed_bytes": used,
            "logical_bytes": sum(v[1] for v in files),
            "by_root": bytes_by_root(files),
            "free_disk_bytes": free,
            "capacity_status": (
                "CAP_EXCEEDED"
                if used > capacity.cap_bytes
                else "CLEANUP_RECOMMENDED"
                if used >= int(capacity.cap_bytes * 0.9)
                else "WITHIN_BUDGET"
            ),
            "display": {
                "managed": _display_bytes(used),
                "free": _display_bytes(free),
                "logical": _display_bytes(sum(v[1] for v in files)),
                "cap": _display_bytes(capacity.cap_bytes),
            },
            "budget": {**self._input_scope(), **capacity.model_dump(mode="json")},
            "references_hash": canonical_hash(evidence),
            "pending_cleanup": pending,
        }

    def _index_actions(
        self, row: dict[str, object], references: EvidenceStorageReferences, pending: object
    ) -> list[str]:
        """The actions the owner would take on one evidence index now.

        Nothing while an approved cleanup waits to be recovered, which the owner
        requires first; a rebuild only of an evicted index, when the evidence runtime
        rebuilds; a pin of an available index, an unpin of a pinned one.

        Args:
            row: The index's row as the storage view shows it.
            references: The evidence references the row came from.
            pending: The cleanups waiting to be recovered.

        Returns:
            The actions, in the order a page offers them.
        """
        if pending:
            return []
        actions = []
        if (
            row["availability"] == PhysicalAvailability.EVICTED_BY_RETENTION.value
            and row["generation_format"] == RETRIEVAL_GENERATION_FORMAT
            and self.evidence is not None
            and self.evidence.rebuild is not None
        ):
            actions.append("REBUILD")
        if row["index_id"] in references.pinned:
            actions.append("UNPIN")
        elif row["availability"] == PhysicalAvailability.AVAILABLE.value:
            actions.append("PIN")
        return actions

    def _input_scope(self) -> dict[str, int]:
        """Keep the current input owner's listing/session facts separate from execution capacity:
        each input the workspace makes current, and its newest published version, read through
        the door."""
        manifest = read_research_workspace_manifest(self.workspace)
        revisions = ResearchInputRevisions(self.session)
        verified = revisions.publications() if manifest.experiment_inputs else ()
        current = set()
        for item in manifest.experiment_inputs or ():
            current.add(item.binding_hash)
            lineage = revisions._lineage_from_verified(item, verified)
            if lineage:
                current.add(lineage[-1].binding_hash)
        estimates = []
        for binding_hash in sorted(current):
            bundle = read_factor_bundle(self.workspace, binding_hash, verify=False)
            artifacts = factor_input_paths(self.workspace, binding_hash)[1]
            manifests = artifacts / "feature-panel/manifests"
            panel = json.loads(
                (manifests / f"{bundle.panel_snapshot_hash}.json").read_text(encoding="utf-8")
            )
            estimates.append(
                storage_input_estimate(
                    active_listing_count=int(panel["active_listing_count"]),
                    research_session_count=len(bundle.sessions),
                )
            )
        return max(estimates, key=lambda row: row["estimated_core_bytes"], default={})

    def cap_readback(self) -> dict[str, Any]:
        """Read operator capacity without opening a research input or changing any record."""
        capacity = workspace_storage_capacity(self.workspace)
        return {
            "status": "AVAILABLE",
            "capacity": capacity.model_dump(mode="json"),
            "recovery_headroom_bytes": RECOVERY_HEADROOM_BYTES,
            "display": {
                "cap": _display_bytes(capacity.cap_bytes),
                "automatic": _display_bytes(capacity.automatic_cap_bytes),
                "managed": _display_bytes(capacity.measured_data_bytes),
                "free": _display_bytes(capacity.free_disk_bytes),
            },
            "next_requests": {
                "set": {"operation": "STORAGE_CAP_SET", "storage_cap_bytes": None},
            },
        }

    def set_cap(self, value: str, *, caller: object) -> dict[str, Any]:
        """Set execution capacity under mutation ownership; preserve every retained object."""
        try:
            self.session.mutation_gate.run(
                StorageCapStore(self.workspace).write,
                value,
                chosen_by=str(caller),
                chosen_at=self.clock(),
            )
        except ValueError:
            return {"status": "REFUSED", "failure_code": "storage.cap_setting_invalid"}
        return {**self.cap_readback(), "status": "CONFIGURED"}

    def capacity_cap_bytes(self) -> int:
        """Resolve the same current operator setting for every write and observation retention."""
        # A new workspace session samples its own retained data even when this process
        # previously opened the same path. Later observation appends reuse the sample
        # refreshed by managed admissions and readbacks, without a whole-tree walk.
        capacity = workspace_storage_capacity(self.workspace, last_sample=self._capacity_sampled)
        self._capacity_sampled = True
        return int(capacity.cap_bytes)

    def admit_evidence_bytes(self, additional_bytes: int) -> None:
        """Admit an evidence write against the workspace budget it shares.

        The same owner and the same rule as every other managed write: the
        unique physical bytes already held plus what this write adds must fit
        under the cap, and the disk must keep its recovery headroom. A refusal
        names the budget (`storage.managed_capacity_exceeded`) and the next
        steps are the ones the storage readback already offers: a cleanup
        plan or a pin review. Nothing is evicted on the write's behalf.
        """
        require_storage_capacity(
            self.workspace,
            additional_bytes=max(0, int(additional_bytes)),
        )

    def plan(self) -> dict[str, Any]:
        """Publish an exact human-confirmation cleanup plan under mutation ownership.

        Current/rollback, user-pinned and in-flight inputs are retained. Raw history, base Features,
        reports, decisions and recovery patches stay governed by their owners; replay requires
        retained inputs.

        Returns:
            Sealed references/targets, rooted retention, reclaimable bytes and explicit
            vector-cleanup refusals.
        """
        with self.session.mutation_gate.hold():
            require_no_pending_cleanup(self.workspace)
            evidence = self._references()
            files = managed_file_inventory(self.workspace)
            targets = {
                p: h
                for p, h in (evidence["source"]["targets"] | self._evidence_targets()).items()
                if confined(self.workspace, p).is_file()
            }
            retained = tuple(v for v in files if v[0] not in targets)
            relinks = self._model_relinks()
            values: dict[str, object] = {
                "references_hash": canonical_hash(evidence),
                "targets": dict(sorted(targets.items())),
                "bindings": [],  # deprecated in 0.2.8: no input is a cleanup target
                "retained_bytes": unique_managed_bytes(retained) - sum(relinks.values()),
                "reclaimable_bytes": unique_managed_bytes(files)
                - unique_managed_bytes(retained)
                + sum(relinks.values()),
                **({"relinks": relinks} if relinks else {}),
            }
            identity = self.owner.publish_input_plan(values)
            references = self._evidence_references()
            refusal = None if references is None else references.vector_cleanup_refusal()
            return {
                "status": "CONFIRMATION_REQUIRED",
                "plan_hash": identity,
                **values,
                "next_requests": {
                    "confirm": {"operation": "STORAGE_CONFIRM", "storage_plan_hash": identity}
                },
                "ask_now": (
                    "May I apply this exact storage cleanup plan, deleting its listed targets "
                    "and applying its listed model relinks?"
                ),
                # Not part of the plan's identity: what this plan could not
                # offer and why, with the step that would make it offerable.
                "refusals": [] if refusal is None else [refusal],
                "limitations": [
                    "CLASSIFIED_INPUT_AND_PANEL_OUTCOME_SNAPSHOTS_ONLY",
                    "RAW_HISTORY_BASE_FEATURES_AND_RECOVERY_PATCHES_RETAINED",
                    "PANEL_GENERATIONS_PAST_THE_PREVIOUS_AND_UNPUBLISHED_CHUNKS_RELEASED",
                    "REPORTS_AND_DECISIONS_RETAINED",
                    "REPLAY_REQUIRES_RETAINED_INPUTS",
                    "EVIDENCE_INDEX_PAYLOADS_ONLY_VECTORS_BLOBS_AND_ARTIFACTS_RETAINED",
                    "EVICTED_EVIDENCE_INDEX_REBUILDS_FROM_ITS_COMMITTED_VECTORS",
                    "A_RETAINED_RETRIEVAL_MODEL_COPY_LINKS_TO_THE_MODEL_STORE_IT_MATCHES",
                ],
            }

    def _model_relinks(self) -> dict[str, int]:
        """The retained retrieval recipe's own model copy, when the machine's store holds its
        verified packs: offered to be linked to them, never deleted. Its bytes, which
        the capability hash binds, stay what sealed indexes read."""
        try:
            binding = read_research_workspace_manifest(self.workspace).evidence_review
        except ResearchWorkspaceError:
            return {}
        if binding is None:
            return {}
        payload = confined(self.workspace, binding.relative_path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != binding.file_sha256:
            return {}
        manifest = EvidenceReviewWorkspaceManifest.model_validate_json(payload)
        root = confined(self.workspace, manifest.semantic_model_relative_path)
        if manifest.retrieval_recipe != RECIPE_MINILM_CPU or not model_store.retained_copy(root):
            return {}
        statuses = model_store.recipe_pack_statuses(
            model_store.default_store_root(), RECIPE_MINILM_CPU
        )
        if any(status.status != model_store.PACK_INSTALLED for status in statuses):
            return {}
        held = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        return {manifest.semantic_model_relative_path: held}

    def _evidence_targets(self) -> dict[str, str]:
        references = self._evidence_references()
        return {} if references is None else references.targets()

    def confirm(self, plan_hash: str, *, caller: str) -> dict[str, object]:
        """Require human confirmation and current references before applying the exact cleanup plan.

        Panel physical availability is reconciled after eviction. The model store proves
        retained-copy bytes before relinking and owns interrupted-swap recovery.

        Args:
            plan_hash: Exact retained cleanup plan.
            caller: Explicit caller required to be HUMAN.

        Returns:
            Completed cleanup path/count metadata and any exact retained-model relinks.

        Raises:
            ValueError: Caller is not human or exact reference/target admission differs.
            StorageRetentionError: Retained model packs cannot be verified/relinked; the same plan
                remains confirmable after repair.
        """
        if caller != "HUMAN":
            raise ValueError("storage.human_confirmation_required")
        with self.session.mutation_gate.hold():
            evidence = self._references()
            deleted = self.owner.apply_input_plan(
                plan_hash,
                references_hash=canonical_hash(evidence),
                eligible_targets=evidence["source"]["targets"] | self._evidence_targets(),
                evidence_evictor=self._evict_evidence_index,
            )
            if any(
                relative.startswith("artifacts/feature-panel/")
                for relative in evidence["source"]["targets"]
            ):
                PanelRetentionResidueOwner(
                    workspace=self.workspace,
                    resolver=ArtifactResolver(self.workspace / "artifacts"),
                ).reconcile_physical_availability()
            # The confirmed plan's relinks, through the model's owner: it proves the copy holds
            # the packs' bytes before it swaps, and finishes an interrupted swap.
            relinked: dict[str, int] = {}
            for relative in self.owner.input_plan(plan_hash).get("relinks") or {}:
                try:
                    relinked[relative] = model_store.relink_retained_copy(
                        model_store.default_store_root(), confined(self.workspace, relative)
                    )
                except model_store.ModelStoreError as error:
                    raise StorageRetentionError(
                        str(error).split(":")[0],
                        "The retrieval model's copy was not linked; the plan can be confirmed "
                        "again once the store's packs are verified and no reader holds the copy.",
                    ) from error
        return {
            "status": "COMPLETED",
            "deleted_paths": deleted,
            "plan_hash": plan_hash,
            **({"relinked": relinked} if relinked else {}),
        }

    def _evict_evidence_index(self, relative: str, plan_hash: str) -> int:
        """Release one evidence index through its owner, under the approved plan.

        The owner refuses while a build, a waiting lease or an open session
        holds the generation; the cleanup then stops where it is, journalled,
        and resumes once the hold has ended.
        """

        assert self.evidence is not None and self.evidence.evict is not None
        knowledge_relative = (
            Path(relative)
            .relative_to(
                Path(self.evidence.knowledge_root.resolve()).relative_to(self.workspace.resolve())
            )
            .as_posix()
        )
        try:
            return int(self.evidence.evict(knowledge_relative, plan_hash, self.clock()))
        except OSError as error:
            # The file system would not release a file: the cleanup stops where it is,
            # journalled, and resumes on the same plan (it reached the caller as a
            # bare fingerprint).
            raise StorageRetentionError(
                "storage.cleanup_interrupted",
                "Cleanup stopped at a file the system would not release; resume the approved "
                "cleanup once the file is free.",
            ) from error
        except ValueError as error:
            if str(error) != "alternative_evidence.retrieval_index_in_use":
                raise
            raise StorageRetentionError(
                "storage.cleanup_target_in_use",
                "An evidence index is held by a build or an open reader; resume the "
                "approved cleanup once it is released.",
            ) from error

    def rebuild_evidence_index(self, index_id: str, *, caller: str) -> dict[str, object]:
        """Restore one evicted evidence index from its committed vectors, receipted.

        Explicit, never a side effect of a read: the index owner re-derives the
        corpus and rebuilds the index from the retained payload -- the pinned
        model runs only when the payload itself is lost, and then its output
        must reproduce the committed digest or the rebuild refuses.
        """
        if caller == "SERVICE_AUTOMATION":
            raise ValueError("storage.evidence_rebuild_caller_not_admitted")
        if self.evidence is None or self.evidence.rebuild is None:
            raise ValueError("storage.evidence_runtime_not_admitted")
        with self.session.mutation_gate.hold():
            require_no_pending_cleanup(self.workspace)
            facts = self.evidence.rebuild(index_id)
            receipt = {
                "kind": "EvidenceIndexRebuildReceipt",
                "index_id": index_id,
                "caller": caller,
                "rebuilt_at": self.clock().isoformat(),
                **facts,
            }
            identity = str(canonical_hash(receipt))
            self.owner._publish("evidence-rebuilds", identity, receipt)
        return {"status": "REBUILT", "receipt_hash": identity, **receipt}

    def pin(self, binding_hash: str, *, pinned: bool, caller: str) -> dict[str, object]:
        """Require human approval before atomically pinning or unpinning an evidence index.

        A research input needs no pin: it is kept on the data root while a binding names it.

        Args:
            binding_hash: Exact evidence index identity.
            pinned: Whether to retain this explicit storage root.
            caller: Explicit caller required to be HUMAN.

        Returns:
            Exact pinned/unpinned storage selection.

        Raises:
            ValueError: Caller is not human, cleanup is pending, the identity is a research
                input, or the selected index is unavailable.
        """
        if caller != "HUMAN":
            raise ValueError("storage.human_confirmation_required")
        with self.session.mutation_gate.hold():
            view = self.readback()
            require_no_pending_cleanup(self.workspace)
            evidence_row = (
                None
                if view["evidence"] is None
                else next(
                    (v for v in view["evidence"]["indexes"] if v["index_id"] == binding_hash),
                    None,
                )
            )
            if evidence_row is not None:
                return self._pin_evidence_index(evidence_row, pinned=pinned)
            if history.research_input_kept(binding_hash):
                raise ValueError("storage.input_pin_retired")
            raise ValueError("storage.input_pin_unavailable")

    def _pin_evidence_index(self, row: dict[str, Any], *, pinned: bool) -> dict[str, object]:
        """A pin on a generation's index, kept in the same receipted pin record."""

        if pinned and row["availability"] != "AVAILABLE":
            raise ValueError("storage.input_pin_unavailable")
        pins = self._all_pins()
        key = f"{EVIDENCE_PIN_PREFIX}{row['index_id']}"
        if pinned:
            pins[key] = int(row["database_bytes"])
        else:
            pins.pop(key, None)
        self._write_pins(pins)
        return {"status": "PINNED" if pinned else "UNPINNED", "index_id": row["index_id"]}

    def _write_pins(self, pins: dict[str, int]) -> None:
        identity = canonical_hash(pins)
        self.owner._publish("input-pin-receipts", identity, {k: v for k, v in pins.items()})
        target = self.owner.governance_root / "input-pins.json"
        with NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            handle.write(
                json.dumps({"pins": pins, "receipt_hash": identity}, sort_keys=True).encode()
            )
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, target)
