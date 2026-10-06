"""Execute one compiled Factor development Program against real inputs.

The Factor Desk already owned deterministic primitives that compute screening
evidence and seal it: ``build_factor_research_program_spec``,
``execute_factor_research_program`` and
``seal_factor_research_deterministic_evidence``. What it did not own was a way
for an *authored document* to reach them. ``freeze`` worked and ``run`` did not,
so the Desk could describe an experiment it could never execute.

This module is that path and nothing more. It calls the three primitives
directly, in the order production calls them, and adds no second implementation
of anything they do.

Deliberately **not** routed through ``FactorResearchProductionRuntime``: that
owner short-circuits on the current pointer, takes a writer lease, and reads and
writes current lineage. A development run has none of those authorities, and
borrowing an owner that does is how a development path acquires them by
accident.

This module must not import ``factor_research.publication``. Development
execution cannot move a current or admitted pointer, and the import graph is
where that is enforced rather than left to convention.

The evidence verifier lives in ``experiments/verification.py`` and this module
must not import it, for the same reason it must not be importable from there:
replay resolves a verifier and must not be able to reach a numerical path
through it. The two meet only at the Host, which installs both.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa

from alphalattice.foundation.factor_research.evaluation.oos_evidence import (
    build_factor_evidence_policy,
)
from alphalattice.foundation.factor_research.evaluation.redundancy import (
    build_factor_redundancy_policy,
)
from alphalattice.foundation.factor_research.experiments.authoring import (
    FACTOR_EXPERIMENT_KIND,
    FactorExperimentCompiler,
    FactorInventoryEntry,
    factor_execution_input_hash,
    factor_inventory_from_panel_manifest,
)
from alphalattice.foundation.factor_research.experiments.development_evidence import (
    FACTOR_DEVELOPMENT_EVIDENCE_CATEGORY,
    FactorDevelopmentFactorIdentity,
    FactorDevelopmentInputBinding,
    FactorDevelopmentReceipt,
    publish_factor_development_evidence,
    publish_factor_development_receipt,
)
from alphalattice.foundation.factor_research.inputs.execution_target import (
    build_factor_target_policy,
)
from alphalattice.foundation.factor_research.programs.program import (
    build_factor_research_program_spec,
    execute_factor_research_program,
    seal_factor_research_deterministic_evidence,
)
from alphalattice.foundation.factor_research.programs.walk_forward import (
    build_factor_walk_forward_policy,
    factor_walk_forward_development_sessions,
)
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReadRequest
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    DeskExecutionResult,
    NumericalCallRecorder,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)


class FactorFeatureReader(Protocol):
    """The panel read seam the deterministic program already expects."""

    def available_sessions(self, manifest_ref: str) -> Sequence[date]: ...

    def batches(self, request: FeaturePanelReadRequest) -> Any: ...


class FactorOutcomeReader(Protocol):
    def read_development_schedule(self, manifest_ref: str) -> pa.Table: ...

    """The causal outcome read seam the deterministic program already expects."""

    def load_manifest(self, snapshot_hash: str) -> Any: ...

    def read_development_sessions(
        self, manifest_ref: str, sessions: tuple[date, ...]
    ) -> pa.Table: ...


class FactorExperimentExecutor:
    """Adapt the deterministic Factor path to the Desk executor Protocol.

    The readers and the outcome manifest are constructor state supplied by Host
    composition, for the same reason Risk takes its return surface that way:
    knowing where a workspace keeps its panel and its execution outcomes is
    workspace knowledge, and a Desk module that acquired it would stop being
    composable.
    """

    kind = FACTOR_EXPERIMENT_KIND

    def __init__(
        self,
        *,
        panel_manifest: Mapping[str, Any],
        feature_reader: FactorFeatureReader,
        outcome_reader: FactorOutcomeReader,
        outcome_snapshot_hash: str,
        outcome_manifest_ref: str,
        frozen_at: datetime,
        inventory: tuple[FactorInventoryEntry, ...] | None = None,
        target_policy: Any | None = None,
        walk_forward_policy: Any | None = None,
        evidence_policy: Any | None = None,
        redundancy_policy: Any | None = None,
    ) -> None:
        self._panel_manifest = dict(panel_manifest)
        self._feature_reader = feature_reader
        self._outcome_reader = outcome_reader
        self._outcome_snapshot_hash = outcome_snapshot_hash
        self._outcome_manifest_ref = outcome_manifest_ref
        self._frozen_at = frozen_at
        # Production defaults unless the composer states otherwise. Supplied
        # rather than hardcoded because these policies encode *cross-section
        # scale* -- the redundancy policy's default requires a hundred common
        # listings -- and a bounded development workspace with ten listings
        # cannot meet that. Hardcoding a relaxed policy here would lower the bar
        # for every caller including production; making it composition state
        # keeps the default exactly what production uses and puts the choice, and
        # the reason for it, at the site that knows the workspace scale.
        self._target_policy = target_policy or build_factor_target_policy()
        self._walk_forward_policy = walk_forward_policy or build_factor_walk_forward_policy()
        self._evidence_policy = evidence_policy or build_factor_evidence_policy()
        self._redundancy_policy = redundancy_policy or build_factor_redundancy_policy()
        # Read off the panel rather than accepted as a bare list, so the axis and
        # the implementation identities come from one artifact and cannot
        # disagree about the same run.
        self._inventory = inventory or factor_inventory_from_panel_manifest(self._panel_manifest)
        self._compiler = FactorExperimentCompiler(self._inventory)

    @property
    def minimum_required_listings(self) -> int:
        return max(
            self._evidence_policy.minimum_cross_section_observations,
            self._redundancy_policy.minimum_common_listings,
        )

    def describe_execution_window(self, panel_manifest_ref: str) -> dict[str, Any]:
        """Expose the existing policy-derived statistical window, not new statistics."""
        panel_sessions = tuple(self._feature_reader.available_sessions(panel_manifest_ref))
        sessions = factor_walk_forward_development_sessions(
            panel_sessions=panel_sessions,
            frozen_at=self._frozen_at,
            policy=self._walk_forward_policy,
        )
        if not sessions:
            raise AuthoringError("factor_research.development_sessions_absent")
        selected = set(sessions)
        schedule = [
            row
            for row in self._outcome_reader.read_development_schedule(
                self._outcome_manifest_ref
            ).to_pylist()
            if row["formation_session"] in selected
        ]
        if {row["formation_session"] for row in schedule} != selected:
            raise AuthoringError("factor_research.development_outcome_support_incomplete")
        available_through = max(row["holding_end_session"] for row in schedule)
        if available_through > self._frozen_at.date():
            raise AuthoringError(
                f"research_experiment.outcomes_after_declared_cutoff:"
                f"{available_through}>{self._frozen_at.date()}"
            )
        return {
            "statistical_start": str(sessions[0]),
            "statistical_end": str(sessions[-1]),
            "statistical_session_count": len(sessions),
            "context_factor_count": len(self._inventory),
            "minimum_listings": self.minimum_required_listings,
            "latest_outcome_session": str(available_through),
            "window_policy": self._walk_forward_policy.model_dump(mode="json"),
            "expected_numerical_calls": 1,
            "interval_semantics": (
                "Requested authority interval does not reslice the policy-derived statistics."
            ),
        }

    @property
    def execution_input_binding_hash(self) -> str:
        """Inputs/policies known before inference, bound by successor Programs."""
        return factor_execution_input_hash(
            str(self._panel_manifest["snapshot_hash"]),
            self._outcome_snapshot_hash,
            tuple(
                p.policy_hash
                for p in (
                    self._target_policy,
                    self._walk_forward_policy,
                    self._evidence_policy,
                    self._redundancy_policy,
                )
            ),
        )

    @staticmethod
    def _selected_factor_ids(
        document: Mapping[str, Any], context_factor_ids: tuple[str, ...]
    ) -> tuple[str, ...]:
        """The axis this document asked about, admitted against the context axis.

        Read here rather than taken from the compiler because the receipt has to
        report what was selected, and a selection the executor never saw is one
        that could only ever have moved an outer hash.
        """

        section = document.get("factor")
        requested = section.get("factor_ids") if isinstance(section, dict) else None
        if not isinstance(requested, list) or not requested:
            raise AuthoringError("factor_research.authoring_factor_ids_invalid")
        selected = tuple(str(value) for value in requested)
        if selected != tuple(dict.fromkeys(selected)):
            raise AuthoringError("factor_research.authoring_factor_ids_duplicated")
        if not set(selected).issubset(set(context_factor_ids)):
            raise AuthoringError("factor_research.authoring_factor_id_not_in_inventory")
        return selected

    def execute(
        self,
        *,
        program: SealedResearchProgram,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
        output_workspace: Path,
        recorder: NumericalCallRecorder | None = None,
    ) -> DeskExecutionResult:
        if (
            authority.execution_input_binding_hash is not None
            and authority.execution_input_binding_hash != self.execution_input_binding_hash
        ):
            raise AuthoringError("factor_research.execution_input_binding_mismatch")
        envelope_document = document.get("experiment")
        if not isinstance(envelope_document, dict):
            raise AuthoringError("research_authoring.experiment_section_missing")
        envelope = ResearchExperimentEnvelope.create(**envelope_document)
        compiled = self._compiler.compile_desk_program(
            envelope=envelope,
            document=document,
            authority=authority,
        )
        if compiled.desk_program_hash != program.desk_program_hash:
            # The sealed Program and the document must still agree, or the run
            # produces evidence for identity nobody admitted.
            raise AuthoringError("research_authoring.replay_identity_mismatch")
        if authority.panel_snapshot_hash != str(self._panel_manifest.get("snapshot_hash")):
            raise AuthoringError("factor_research.development_panel_not_this_authority")

        # Two axes, and they are not interchangeable.
        #
        # ``context_factor_ids`` is the panel's whole factor axis and is the
        # numerical and statistical universe: ``_validate_panel_manifest``
        # requires exact equality with the published summary, and
        # ``FactorOosEvidenceReport.hypothesis_count`` is the FDR denominator, so
        # this is what multiple-testing corrects over.
        #
        # ``selected_factor_ids`` is what this document asked about. It scopes
        # what the receipt reports and what a downstream Desk may consume. It
        # deliberately does **not** change the hypothesis universe -- narrowing
        # the FDR denominator from the full axis to a chosen handful would move
        # every significance verdict and make this run's evidence incomparable
        # with every other Factor result.
        #
        # They are named rather than both called ``factor_ids`` because a reader
        # seeing a configured selection next to a bare ``factor_ids`` would
        # reasonably assume the selection narrowed the statistics, and it does not.
        context_factor_ids = tuple(entry.factor_id for entry in self._inventory)
        selected_factor_ids = self._selected_factor_ids(document, context_factor_ids)
        spec = build_factor_research_program_spec(
            feature_panel_snapshot_hash=authority.panel_snapshot_hash,
            feature_panel_manifest_ref=authority.panel_manifest_ref,
            causal_outcome_snapshot_hash=self._outcome_snapshot_hash,
            causal_outcome_manifest_ref=self._outcome_manifest_ref,
            factor_ids=context_factor_ids,
            frozen_at=self._frozen_at,
            target_policy=self._target_policy,
            walk_forward_policy=self._walk_forward_policy,
            evidence_policy=self._evidence_policy,
            redundancy_policy=self._redundancy_policy,
            # An exploration sample: the evidence is computed over the names the Host
            # sampled, instead of the whole Panel.
            listing_ids=None if authority.listing_sample is None else authority.ordered_listing_ids,
        )
        result = execute_factor_research_program(
            program=spec,
            panel_manifest=self._panel_manifest,
            feature_reader=self._feature_reader,
            outcome_reader=self._outcome_reader,
        )
        # The development binding, not a production one: it folds in the Desk
        # method identity the Program was sealed with, so this evidence can never
        # collide with a record produced under current authority.
        evidence = seal_factor_research_deterministic_evidence(
            result=result,
            execution_binding_hash=program.method_binding_hash,
        )
        # What this run was actually allowed to read, sealed as a contract rather
        # than hashed as an anonymous dict. Its content stays Desk methodology --
        # the generic layer carries only ``binding_hash`` and learns nothing about
        # what a Factor input is -- but a verifier can now open it and ask which
        # policies and which axis it describes, instead of comparing one copy of
        # an opaque hash with another.
        input_binding = FactorDevelopmentInputBinding.create(
            selected_factor_ids=selected_factor_ids,
            context_factors=tuple(
                FactorDevelopmentFactorIdentity(
                    factor_id=entry.factor_id,
                    implementation_hash=entry.implementation_hash,
                    # The same methodology identity the compiler binds. A binding
                    # that recorded only the kernel would be a weaker claim than
                    # the catalog hash it travels with, and the weaker of two
                    # identities is the one that decides what a reader can prove.
                    methodology_hash=entry.methodology_hash,
                )
                for entry in self._inventory
            ),
            feature_panel_snapshot_hash=authority.panel_snapshot_hash,
            feature_panel_manifest_ref=authority.panel_manifest_ref,
            causal_outcome_snapshot_hash=self._outcome_snapshot_hash,
            causal_outcome_manifest_ref=self._outcome_manifest_ref,
            # The four policies the deterministic program was actually run under,
            # taken from the spec that ran rather than from this executor's
            # constructor state, so the binding cannot describe a configuration
            # the program did not use.
            target_policy_hash=spec.target_policy.policy_hash,
            walk_forward_policy_hash=spec.walk_forward_policy.policy_hash,
            screening_policy_hash=spec.evidence_policy.policy_hash,
            redundancy_policy_hash=spec.redundancy_policy.policy_hash,
            authority_hash=authority.authority_hash,
        )
        input_binding_hash = input_binding.binding_hash
        publish_factor_development_evidence(
            output_workspace,
            evidence.model_dump(mode="json"),
            evidence.checkpoint_hash,
        )
        # The lineage record. Published alongside the child rather than instead of
        # it: the child is the deterministic result and is self-identifying, and
        # this says which authored Program produced it, under which method
        # binding, over which selected and context axes. A consumer holding only
        # the child could establish none of that.
        #
        # The Program is embedded whole. Three copied hashes could only ever be
        # compared with copies of themselves; the sealed Program was compiled
        # before this run and binds the selected axis into its own identity, so a
        # verifier can recompute that identity and refuse a rewritten selection.
        receipt = FactorDevelopmentReceipt.create(
            program=program,
            input_binding=input_binding,
            checkpoint_hash=evidence.checkpoint_hash,
        )
        artifact_uri = publish_factor_development_receipt(output_workspace, receipt)
        # The formation axis the program computed over: the development
        # sessions it derived from the admitted calendar under the sealed
        # policy, carried out of the result rather than derived again from a
        # second read of the same calendar.
        sessions = tuple(result.development_sessions)
        if not sessions:
            raise AuthoringError("factor_research.development_sessions_empty")
        if recorder is not None:
            recorder.record(capability=spec.program_hash)
        return DeskExecutionResult(
            disposition="COMPUTED",
            artifact_uris=(artifact_uri,),
            formation_sessions=sessions,
            # One sealed deterministic program execution. Factor's numerical work
            # is a single pass over the admitted development surface rather than
            # a call per formation, so counting anything else here would invent a
            # unit this Desk does not have.
            numerical_call_count=1,
            desk_input_binding_hash=input_binding_hash,
        )


__all__ = [
    "FACTOR_DEVELOPMENT_EVIDENCE_CATEGORY",
    "FactorExperimentExecutor",
    "FactorFeatureReader",
    "FactorOutcomeReader",
]
