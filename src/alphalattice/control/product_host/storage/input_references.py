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
    FactorInputBundle,
    confined,
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
    the managed total; a role's roots are summed (V208).
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


def _missing_input_binding(binding_hash: str, detail: str) -> StorageRetentionError:
    """Name a missing retained binding distinctly from a present but invalid record."""
    return StorageRetentionError(
        f"storage.input_binding_missing:research-inputs/{binding_hash}", detail
    )


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

    def _pins(self) -> dict[str, int]:
        """Input pins only; evidence index pins carry their own prefix."""

        return {
            key: value
            for key, value in self._all_pins().items()
            if not key.startswith(EVIDENCE_PIN_PREFIX)
        }

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

    def _bundles(self) -> dict[str, FactorInputBundle]:
        """Read each retained input manifest once for this operation."""
        bundles = {}
        for path in sorted((self.workspace / "research-inputs").glob("*/manifest.json")):
            value = FactorInputBundle.model_validate_json(path.read_bytes())
            if value.binding_hash != path.parent.name:
                raise StorageRetentionError(
                    "storage.retention_root_mismatch", "Input manifest is misfiled."
                )
            bundles[value.binding_hash] = value
        return bundles

    def _references(
        self,
    ) -> tuple[dict[str, FactorInputBundle], dict[str, set[str]], dict[str, Any]]:
        manifest = read_research_workspace_manifest(self.workspace)
        bundles = self._bundles()
        roots: dict[str, set[str]] = {h: set() for h in bundles}
        for item in manifest.experiment_inputs or ():
            if item.binding_hash not in roots:
                raise _missing_input_binding(
                    item.binding_hash, "An installed input manifest is missing."
                )
            roots[item.binding_hash].add("CURRENT_ACTIVE")
        for item in manifest.model_training_inputs or ():
            if item.input_binding_hash not in roots:
                raise _missing_input_binding(
                    item.input_binding_hash, "A retained model training input is missing."
                )
            roots[item.input_binding_hash].add("MODEL_TRAINING_SOURCE")
        revisions = ResearchInputRevisions(self.session)
        try:
            verified_publications = revisions.publications()
        except FileNotFoundError as error:
            missing = Path(error.filename) if error.filename is not None else None
            if (
                missing is not None
                and missing.name == "manifest.json"
                and missing.parent.parent == self.workspace / "research-inputs"
                and len(missing.parent.name) == 64
                and all(value in "0123456789abcdef" for value in missing.parent.name)
            ):
                raise _missing_input_binding(
                    missing.parent.name, "A published input manifest is missing."
                ) from error
            raise
        except ValueError as error:
            raise StorageRetentionError(
                "storage.retention_root_mismatch", "Input publication is invalid."
            ) from error
        publications = [
            {k: v for k, v in value.items() if k != "receipt_hash"}
            for value in verified_publications
        ]
        known = {str(v["binding_hash"]) for v in publications}
        for item in manifest.experiment_inputs or ():
            parents = {
                v["prior_binding_hash"]
                for v in publications
                if v["input_id"] == item.input_id and v["binding_hash"] == item.binding_hash
            }
            for parent in parents:
                if parent in roots:
                    roots[parent].add("PREVIOUS_ROLLBACK")
            lineage = revisions._lineage_from_verified(item, verified_publications)
            if lineage:
                roots[lineage[-1].binding_hash].add("CURRENT_RESEARCH_VERSION")
                roots[lineage[-1].prior_binding_hash].add("PREVIOUS_ROLLBACK")
        for h in roots:
            if h not in known:
                roots[h].add("UNCLASSIFIED_LEGACY")
        tasks = []
        for task in self.session.task_control_registry.tasks():
            portfolio_source = (
                task.input.payload.get("plan", {}).get("portfolio_source")
                if isinstance(task.input.payload.get("plan"), dict)
                else None
            )
            if portfolio_source is not None:
                from alphalattice.investment.portfolio_strategy_lab.application import (
                    research_experiment,
                )

                source = research_experiment.PortfolioExperimentSource.model_validate(
                    portfolio_source
                )
                if source.input_binding_hash not in roots:
                    raise _missing_input_binding(
                        source.input_binding_hash, "Portfolio research input is missing."
                    )
                roots[source.input_binding_hash].add("PORTFOLIO_RESEARCH")
            if task.lifecycle not in {TaskLifecycle.SUCCEEDED, TaskLifecycle.CANCELLED}:
                tasks.append((str(task.task_id), task.input.input_hash, task.lifecycle.value))
                referenced = _strings(task.input.payload).intersection(roots)
                # An older/custom task may carry a source path rather than a
                # binding field. Unknown does not mean safe to evict.
                for h in referenced or roots.keys():
                    roots[h].add("IN_FLIGHT_RECOVERY")
        pins = self._pins()
        from alphalattice.foundation.research_foundation.publication.sponsorship import (
            read_foundation_admission,
        )

        foundations = []
        admission_root = (
            self.workspace / "artifacts/factor-research/research-desk/foundation-admissions"
        )
        for path in sorted(admission_root.glob("*.json")):
            admission = read_foundation_admission(self.workspace / "artifacts", path.stem)
            foundations.append(admission.admission_hash)
            for h in (admission.input_binding_hash, admission.factor_input_binding_hash):
                if h not in roots:
                    raise _missing_input_binding(h, "Foundation input is missing.")
                roots[h].add("RESEARCH_FOUNDATION")
        for h in pins:
            if h not in roots:
                raise _missing_input_binding(h, "Pinned input manifest is missing.")
            roots[h].add("USER_PINNED")
        evidence: dict[str, Any] = {
            "manifest": manifest.manifest_hash,
            "publications": publications,
            "roots": {h: sorted(v) for h, v in sorted(roots.items())},
            "tasks": tasks,
            "pins": pins,
            "foundations": foundations,
            "source": self._source_references(bundles, roots, busy=bool(tasks)),
        }
        preparation = self._preparation_references(manifest, busy=bool(tasks))
        evidence["source"]["targets"].update(preparation["targets"])
        evidence["source"]["preparation"] = preparation
        references = self._evidence_references()
        if references is not None:
            # Every fact a plan over evidence indexes depends on: a new pin, an
            # open reader or an unfinished Task changes the hash and stales it.
            evidence["evidence_indexes"] = references.evidence()
        return bundles, roots, evidence

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

    def _source_references(
        self,
        bundles: dict[str, FactorInputBundle],
        roots: dict[str, set[str]],
        *,
        busy: bool,
    ) -> dict[str, Any]:
        """Release classified completed snapshots, never raw history or recovery patches.

        A generation no research input captured is kept while it is the current one or the
        previous one; past the previous it is superseded, and the chunks no kept generation
        names are released under the confirmed plan, a composition's that no generation
        published among them (V207).
        """
        artifacts = self.workspace / "artifacts"
        panel_root = artifacts / "feature-panel"
        if not tuple((panel_root / "manifests").glob("*.json")):
            return {"targets": {}, "roots": []}
        known_panels = {b.panel_snapshot_hash for b in bundles.values()}
        known_outcomes = {b.outcome_snapshot_hash for b in bundles.values()}
        protected_panels = {b.panel_snapshot_hash for h, b in bundles.items() if roots[h]}
        protected_outcomes = {b.outcome_snapshot_hash for h, b in bundles.items() if roots[h]}
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
            protected_panels.update(panel_hashes - known_panels)
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
            protected_outcomes.update(known_outcomes)
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
            if (
                outcome.snapshot_hash in protected_outcomes
                or outcome.snapshot_hash not in known_outcomes
            ):
                protected.update(
                    outcomes._parquet_path(c.split, c.content_hash)
                    for c in outcome.development_chunks
                )
        targets = {}
        for h, bundle in bundles.items():
            if roots[h]:
                continue
            for name, digest in bundle.files:
                if not name.startswith(
                    (
                        "artifacts/feature-panel/chunks/",
                        "artifacts/data-operations/execution-outcomes/development/chunks/",
                    )
                ) or not name.endswith(".parquet"):
                    continue
                if confined(self.workspace, name) not in protected:
                    targets[name] = digest
        if not busy:
            # The chunks no kept generation names: a superseded generation's own, and those a
            # composition wrote that no generation published (V207; 11 of `dh-append`'s 24).
            captured = {name for h, b in bundles.items() if roots[h] for name, _ in b.files}
            for path in sorted((panel_root / "chunks").glob("*.parquet")):
                name = path.relative_to(self.workspace).as_posix()
                if path not in protected and name not in captured and name not in targets:
                    targets[name] = file_content_hash(path)
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
        bundles, roots, evidence = self._references()
        files = managed_file_inventory(self.workspace)
        used = unique_managed_bytes(files)
        rows = []
        for h, bundle in bundles.items():
            names = {f"research-inputs/{h}/source/{name}" for name, _digest in bundle.files}
            held = tuple(v for v in files if v[0] in names)
            rows.append(
                {
                    "binding_hash": h,
                    "panel_snapshot_hash": bundle.panel_snapshot_hash,
                    "roots": sorted(roots[h]),
                    "logical_bytes": sum(v[1] for v in held),
                    "available": len(held) == len(names),
                }
            )
        capacity = StorageCapStore(self.workspace).capacity(measured_data_bytes=used)
        pending = pending_input_cleanups(self.workspace)
        free = shutil.disk_usage(self.workspace).free
        references = self._evidence_references()
        return {
            "status": "RECOVERY_REQUIRED" if pending else "AVAILABLE",
            "inputs": rows,
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
            "budget": {**self._input_scope(bundles, roots), **capacity.model_dump(mode="json")},
            "references_hash": canonical_hash(evidence),
            "pending_cleanup": pending,
        }

    def _index_actions(
        self, row: dict[str, object], references: EvidenceStorageReferences, pending: object
    ) -> list[str]:
        """The actions the owner would take on one evidence index now (V199).

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

    def _input_scope(
        self, bundles: dict[str, FactorInputBundle], roots: dict[str, set[str]]
    ) -> dict[str, int]:
        """Keep the current input owner's listing/session facts separate from execution capacity."""
        estimates = []
        for key, current in bundles.items():
            if not roots[key].intersection({"CURRENT_ACTIVE", "CURRENT_RESEARCH_VERSION"}):
                continue
            panel_path = (
                self.workspace
                / "research-inputs"
                / current.binding_hash
                / "source"
                / "artifacts/feature-panel/manifests"
                / f"{current.panel_snapshot_hash}.json"
            )
            panel = json.loads(panel_path.read_text(encoding="utf-8"))
            estimates.append(
                storage_input_estimate(
                    active_listing_count=int(panel["active_listing_count"]),
                    research_session_count=len(current.sessions),
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
            bundles, roots, evidence = self._references()
            files = managed_file_inventory(self.workspace)
            targets = {
                p: h
                for p, h in (
                    self._eligible_targets(bundles, roots)
                    | evidence["source"]["targets"]
                    | self._evidence_targets()
                ).items()
                if confined(self.workspace, p).is_file()
            }
            unrooted = sorted(h for h in bundles if not roots[h])
            retained = tuple(v for v in files if v[0] not in targets)
            relinks = self._model_relinks()
            values: dict[str, object] = {
                "references_hash": canonical_hash(evidence),
                "targets": dict(sorted(targets.items())),
                "bindings": unrooted,
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
        verified packs: offered to be linked to them, never deleted (V208). Its bytes, which
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

    @staticmethod
    def _eligible_targets(
        bundles: dict[str, FactorInputBundle], roots: dict[str, set[str]]
    ) -> dict[str, str]:
        protected = {digest for h, b in bundles.items() if roots[h] for _, digest in b.files}
        targets = {}
        for h, bundle in bundles.items():
            if roots[h]:
                continue
            for name, digest in bundle.files:
                if not name.endswith((".parquet", ".duckdb")):
                    continue
                targets[f"research-inputs/{h}/source/{name}"] = digest
                if digest not in protected:
                    if name.endswith(".parquet"):
                        targets[f"research-inputs/parquet/{digest}.parquet"] = digest
                    elif bundle.database_snapshot_hash is not None:
                        targets[f"research-inputs/databases/{digest}/market-data.duckdb"] = digest
        return targets

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
            bundles, roots, evidence = self._references()
            deleted = self.owner.apply_input_plan(
                plan_hash,
                references_hash=canonical_hash(evidence),
                eligible_targets=self._eligible_targets(bundles, roots)
                | evidence["source"]["targets"]
                | self._evidence_targets(),
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
            # the packs' bytes before it swaps, and finishes an interrupted swap (V208).
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
            # journalled, and resumes on the same plan (V449: it reached the caller as a
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
        """Require human approval before atomically pinning or unpinning exact available storage.

        Args:
            binding_hash: Exact admitted input or evidence index identity.
            pinned: Whether to retain this explicit storage root.
            caller: Explicit caller required to be HUMAN.

        Returns:
            Exact pinned/unpinned storage selection.

        Raises:
            ValueError: Caller is not human, cleanup is pending or selected storage is unavailable.
        """
        if caller != "HUMAN":
            raise ValueError("storage.human_confirmation_required")
        with self.session.mutation_gate.hold():
            view = self.readback()
            require_no_pending_cleanup(self.workspace)
            row = next((v for v in view["inputs"] if v["binding_hash"] == binding_hash), None)
            evidence_row = None
            if row is None and view["evidence"] is not None:
                evidence_row = next(
                    (v for v in view["evidence"]["indexes"] if v["index_id"] == binding_hash),
                    None,
                )
            if evidence_row is not None:
                return self._pin_evidence_index(evidence_row, pinned=pinned)
            if row is None or (pinned and not row["available"]):
                raise ValueError("storage.input_pin_unavailable")
            pins = self._pins()
            if pinned:
                pins[binding_hash] = row["logical_bytes"]
            else:
                pins.pop(binding_hash, None)
            pins = {**{k: v for k, v in self._all_pins().items() if k not in self._pins()}, **pins}
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
        return {"status": "PINNED" if pinned else "UNPINNED", "binding_hash": binding_hash}

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
