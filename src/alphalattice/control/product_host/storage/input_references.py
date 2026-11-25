"""Project real research-input references into the existing retention owner."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.capabilities.knowledge import model_store
from alphalattice.capabilities.knowledge.hybrid_contracts import RECIPE_MINILM_CPU
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
    storage_input_estimate,
)
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
from alphalattice.control.workspace_runtime.verified_facts import verified_bytes
from alphalattice.evidence.alternative_evidence.storage.inventory import (
    EvidenceStorageBinding,
    EvidenceStorageInventory,
)
from alphalattice.foundation.causal_outcomes.execution.artifacts import (
    local_qa_preparation_retention_inventory,
)
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.foundation.data_layer import history
from alphalattice.foundation.feature_engine.panels.retention_residue import (
    PanelRetentionResidueOwner,
)
from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.foundation.research_foundation.runtime.delta import VerifiedPreResearchHead
from alphalattice.investment.alpha_research.publication.artifacts import AlphaCurrentArtifactStore
from alphalattice.kernel.clock import utc_now


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


class ResearchInputStorage:
    """An application projection; policy and deletion remain with retention.

    A workspace's evidence store is reported beside its inputs, its bytes by kind and its
    unreferenced source objects; none of it is a cleanup target.
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
        self.clock = clock or (utc_now)
        self._capacity_sampled = False

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
        inventory = (
            None
            if self.evidence is None
            else EvidenceStorageInventory(workspace=self.workspace, binding=self.evidence)
        )
        return {
            "status": "RECOVERY_REQUIRED" if pending else "AVAILABLE",
            "inputs": [],  # deprecated in 0.2.8: inputs are kept on the data root
            "evidence": None
            if inventory is None
            else {**inventory.summary(), "unreferenced": inventory.unreferenced()},
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
            values = self._cleanup_values()
            identity = self.owner.publish_input_plan(values)
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
                "refusals": [],
                "limitations": [
                    "CLASSIFIED_INPUT_AND_PANEL_OUTCOME_SNAPSHOTS_ONLY",
                    "RAW_HISTORY_BASE_FEATURES_AND_RECOVERY_PATCHES_RETAINED",
                    "PANEL_GENERATIONS_PAST_THE_PREVIOUS_AND_UNPUBLISHED_CHUNKS_RELEASED",
                    "REPORTS_AND_DECISIONS_RETAINED",
                    "REPLAY_REQUIRES_RETAINED_INPUTS",
                    "A_RETAINED_RETRIEVAL_MODEL_COPY_LINKS_TO_THE_MODEL_STORE_IT_MATCHES",
                ],
            }

    def status(self) -> dict[str, object]:
        """Measure the data root and estimate the current plan without publishing it."""
        from alphalattice.control.workspace_runtime.storage.footprint import (  # lazy: status
            allocated_by_category,
        )

        unavailable: dict[str, str] = {}
        try:
            categories = allocated_by_category(history.data_root())
        except OSError:
            categories = None
            unavailable["bytes_by_category"] = "storage.footprint_unreadable"
        try:
            reclaimable = self._cleanup_values()["reclaimable_bytes"]
        except (OSError, ValueError, StorageRetentionError):
            reclaimable = None
            unavailable["reclaimable_bytes"] = "storage.cleanup_estimate_unavailable"
        return {
            "bytes_by_category": categories,
            "reclaimable_bytes": reclaimable,
            "scratch_cleanup": self.session.scratch_cleanup,
            **({"unavailable": unavailable} if unavailable else {}),
        }

    def _cleanup_values(self) -> dict[str, object]:
        evidence = self._references()
        files = managed_file_inventory(self.workspace)
        targets = {
            path: digest
            for path, digest in evidence["source"]["targets"].items()
            if confined(self.workspace, path).is_file()
        }
        retained = tuple(row for row in files if row[0] not in targets)
        relinks = self._model_relinks()
        retained_bytes = unique_managed_bytes(retained) - sum(relinks.values())
        return {
            "references_hash": canonical_hash(evidence),
            "targets": dict(sorted(targets.items())),
            "bindings": [],  # deprecated in 0.2.8: no input is a cleanup target
            "retained_bytes": retained_bytes,
            "reclaimable_bytes": unique_managed_bytes(files) - retained_bytes,
            **({"relinks": relinks} if relinks else {}),
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
        payload, digest = verified_bytes(confined(self.workspace, binding.relative_path))
        if digest != binding.file_sha256:
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
                eligible_targets=evidence["source"]["targets"],
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

    def rebuild_evidence_index(self, index_id: str, *, caller: str) -> dict[str, object]:
        """Deprecated in 0.2.8, removed in 0.3.0: the Host's retrieval store keeps no
        per-generation evidence index, so there is none to rebuild; it refuses as it has since.

        Raises:
            ValueError: `storage.evidence_rebuild_caller_not_admitted` for a service, else
                `storage.evidence_runtime_not_admitted`.
        """
        del index_id
        if caller == "SERVICE_AUTOMATION":
            raise ValueError("storage.evidence_rebuild_caller_not_admitted")
        raise ValueError("storage.evidence_runtime_not_admitted")

    def pin(self, binding_hash: str, *, pinned: bool, caller: str) -> dict[str, object]:
        """Require human approval; nothing is pinned any more.

        A research input needs no pin: it is kept on the data root while a binding names it. The
        Host's retrieval store keeps no per-generation evidence index to pin.

        Args:
            binding_hash: The identity the person named.
            pinned: Whether to retain this explicit storage root.
            caller: Explicit caller required to be HUMAN.

        Returns:
            Never: every pin is refused by name.

        Raises:
            ValueError: Caller is not human, cleanup is pending, the identity is a kept research
                input (`storage.input_pin_retired`), or nothing it names can be pinned.
        """
        del pinned
        if caller != "HUMAN":
            raise ValueError("storage.human_confirmation_required")
        with self.session.mutation_gate.hold():
            require_no_pending_cleanup(self.workspace)
            if history.research_input_kept(binding_hash):
                raise ValueError("storage.input_pin_retired")
            raise ValueError("storage.input_pin_unavailable")
