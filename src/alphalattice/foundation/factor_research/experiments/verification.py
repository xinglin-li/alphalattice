"""Read one Factor development evidence graph back by hash, and cross-check it.

Replay used to refuse this Desk outright. There was no verifier, so
``replay`` reported ``evidence_verifier_not_installed`` rather than
``REUSED_EXACT`` -- an honest refusal, and the right answer while nothing walked
the artifacts. It is the wrong answer now: the graph exists, its lineage is
recoverable, and refusing to look at it means the one Desk that produces
screening evidence is also the one Desk whose evidence can never be reused.

Reading each artifact back is necessary and nowhere near sufficient. Every
artifact here is content-addressed, so each one *individually* verifies against
its own name no matter which run produced it, and the receipt re-validates its
own ``receipt_hash`` over its own contents. Self-consistency is not authority: a
graph re-sealed end to end against a different Panel, a rewritten selected axis
or an uninstalled method is internally perfect. So this walks the graph and then
contradicts it with sides it did not produce.

There are three such sides, and they answer different questions.

The **sealed Program** answers what was admitted. It was compiled at freeze time,
before anything executed, and it binds the ordered context axis through
``catalog_hash`` and the selected axis through ``desk_program_hash`` -- both
recomputed here through the single shared functions the compiler used, never
compared against a copy of themselves.

The **resolved authority** answers what this run was entitled to read. It is
resolved from the live workspace during replay, so it is the only side that can
refuse a graph whose every internal hash agrees about the wrong Panel.

The **installed policy enumerations** answer what this build offers. A method
binding that no installed screening/redundancy pair reproduces describes a
configuration this build cannot run, so replay refuses it. A recorded readback
is not a run: it opens the sealed graph and states what a run would be told --
``method_standing`` and the refusal it would get in ``method_refusal``. A build
that adds or retires a policy must not make the studies saved under the earlier
ones unreadable.

Nothing here can compute. This module must not import
``factor_research.experiments.execution`` or ``factor_research.publication``:
replay resolves a verifier and must not be able to reach a numerical path or a
current pointer through it, and the import graph is where that is enforced
rather than left to convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from alphalattice.foundation.factor_research.experiments.authoring import (
    FACTOR_EXPERIMENT_KIND,
    INSTALLED_REDUNDANCY_POLICIES,
    INSTALLED_SCREENING_POLICIES,
    FactorInventoryEntry,
    factor_catalog_hash,
    factor_execution_input_hash,
    factor_method_binding_hash,
    factor_parameter_domain_hash,
)
from alphalattice.foundation.factor_research.experiments.development_evidence import (
    FACTOR_DEVELOPMENT_RECEIPT_CATEGORY,
    FactorDevelopmentCurationReader,
    FactorDevelopmentReceipt,
    FactorDevelopmentReceiptReader,
    verify_factor_development_receipt,
)
from alphalattice.foundation.factor_research.programs.program import (
    FactorResearchDeterministicEvidence,
)
from alphalattice.foundation.factor_research.research_loop.decisions import (
    FactorResearchDecisionAuthorityError,
    verify_factor_research_curation,
)
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExecutionEvidence,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)

_RECEIPT_PREFIX = f"playpen://factor-research/{FACTOR_DEVELOPMENT_RECEIPT_CATEGORY}/"


class FactorEvidenceVerifier:
    """Verify the receipt, its deterministic child, and any curation beside them."""

    kind = FACTOR_EXPERIMENT_KIND

    def __init__(self) -> None:
        self.verified_readback: (
            tuple[FactorDevelopmentReceipt, FactorResearchDeterministicEvidence] | None
        ) = None
        self.method_standing: Literal["INSTALLED", "NOT_CURRENT"] | None = None
        """Whether this build still offers the method the last verified graph was
        sealed under; replay refuses one it does not, a recorded readback reports it."""
        self.method_refusal: str | None = None
        """The refusal a run would get for that method, or None when it is offered."""

    def verify(
        self,
        *,
        program: SealedResearchProgram,
        evidence: ResearchExecutionEvidence,
        authority: ResolvedResearchAuthority | None,
        output_workspace: Path,
    ) -> None:
        self._verify(
            program=program,
            evidence=evidence,
            authority=authority,
            output_workspace=output_workspace,
            recorded_readback=False,
        )

    def verify_recorded(
        self,
        *,
        program: SealedResearchProgram,
        evidence: ResearchExecutionEvidence,
        authority: ResolvedResearchAuthority | None,
        output_workspace: Path,
    ) -> None:
        """Verify sealed contents, not current admission of later curation policies."""
        self._verify(
            program=program,
            evidence=evidence,
            authority=authority,
            output_workspace=output_workspace,
            recorded_readback=True,
        )

    def _verify(
        self,
        *,
        program: SealedResearchProgram,
        evidence: ResearchExecutionEvidence,
        authority: ResolvedResearchAuthority | None,
        output_workspace: Path,
        recorded_readback: bool,
    ) -> None:
        self.verified_readback = None
        self.method_standing = None
        self.method_refusal = None
        if authority is None:
            raise AuthoringError("research_authoring.live_authority_required_for_replay")
        if evidence.kind != self.kind or program.kind != self.kind:
            raise AuthoringError("factor_research.evidence_kind_mismatch")
        root = Path(output_workspace)
        receipt, child = self._load(root, evidence.artifact_uris)

        # The admitted axis, recomputed from the receipt through the compiler's
        # own formula and contradicted by an artifact sealed before the run.
        inventory = self._admitted_inventory(receipt=receipt, program=program)
        self._verify_program_lineage(receipt=receipt, program=program)
        self._verify_authority(receipt=receipt, program=program, authority=authority)
        method_refusal = self._method_binding_refusal(program=program)
        if method_refusal is not None and not recorded_readback:
            raise AuthoringError(method_refusal)
        # The deep receipt/child cross-check the Desk already owns: selected axis
        # admission, policy hashes, input identities, the child's method binding
        # and the generic record. Passing the inventory is sound only because the
        # step above proved it is the axis the Program was sealed with -- an
        # inventory taken from the receipt alone would agree with itself.
        verify_factor_development_receipt(
            receipt=receipt,
            child=child,
            evidence=evidence,
            panel_inventory=inventory,
            panel_snapshot_hash=authority.panel_snapshot_hash,
        )
        self._verify_curation(
            child=child,
            # The Desk method binding this run was sealed under, which is what
            # the dossier a curation answers was built from.
            review_binding_hash=receipt.method_binding_hash,
            output_workspace=root,
            recorded_readback=recorded_readback,
        )
        self.method_standing = "INSTALLED" if method_refusal is None else "NOT_CURRENT"
        self.method_refusal = method_refusal
        self.verified_readback = receipt, child

    @staticmethod
    def _load(
        output_workspace: Path, artifact_uris: tuple[str, ...]
    ) -> tuple[FactorDevelopmentReceipt, FactorResearchDeterministicEvidence]:
        """Resolve the single receipt this evidence names, and its child.

        Exactly one receipt is the root of exactly one run. Zero means the record
        names nothing this Desk owns; more than one means the graph has two roots
        and no rule says which is authoritative. An unrecognised URI is an
        artifact nobody verified travelling inside a record that claims
        everything was verified.
        """

        unknown = tuple(uri for uri in artifact_uris if not uri.startswith(_RECEIPT_PREFIX))
        if unknown:
            raise AuthoringError("research_authoring.evidence_artifact_uri_unknown")
        if len(artifact_uris) != 1:
            raise AuthoringError("factor_research.evidence_receipt_not_unique")
        try:
            return FactorDevelopmentReceiptReader(output_workspace).load(artifact_uris[0])
        except (AuthoringError, OSError, ValueError) as error:
            # A deleted or tampered artifact is a replay failure, not a Factor
            # storage error leaking through the generic workflow. ``OSError`` is
            # listed because a *removed* child surfaces as a bare
            # ``FileNotFoundError``, and that is the more likely case in practice.
            raise AuthoringError(
                f"research_authoring.evidence_artifact_unverifiable:{error}"
            ) from error

    @staticmethod
    def _admitted_inventory(
        *, receipt: FactorDevelopmentReceipt, program: SealedResearchProgram
    ) -> tuple[FactorInventoryEntry, ...]:
        """Prove the receipt's ordered context axis is the one the Program admitted.

        This is the check that refuses an altered ordered axis and an uninstalled
        method at the same time, and it does so without reading the Panel: the
        compiler computed ``catalog_hash`` from the Host-resolved published
        inventory before the run, so reproducing it from the receipt's own
        entries is a statement about the Panel even though the Panel is not here.

        Order is included because the axis is ordered. Two runs over the same
        Factors in a different order are two different catalogs, and the FDR
        denominator they share does not make them one.
        """

        entries = tuple(
            FactorInventoryEntry(
                factor_id=value.factor_id,
                implementation_hash=value.implementation_hash,
                methodology_hash=value.methodology_hash,
            )
            for value in receipt.input_binding.context_factors
        )
        if factor_catalog_hash(entries) != program.catalog_hash:
            raise AuthoringError("factor_research.evidence_context_axis_not_admitted")
        return entries

    @staticmethod
    def _verify_program_lineage(
        *, receipt: FactorDevelopmentReceipt, program: SealedResearchProgram
    ) -> None:
        """The Program embedded in the receipt must be the Program being replayed.

        The receipt carries the whole sealed Program rather than three copied
        hashes, so this is an equality between two independently sealed objects
        rather than between two copies of one value.
        """

        if receipt.program.program_hash != program.program_hash:
            raise AuthoringError("factor_research.evidence_program_mismatch")
        if receipt.program != program:
            # Same identity, different content is a contradiction rather than a
            # near miss: ``program_hash`` covers every field, so reaching here
            # means one of the two was constructed without its own validator.
            raise AuthoringError("factor_research.evidence_program_content_mismatch")

    @staticmethod
    def _verify_authority(
        *,
        receipt: FactorDevelopmentReceipt,
        program: SealedResearchProgram,
        authority: ResolvedResearchAuthority,
    ) -> None:
        """Contradict the graph with what the Host resolved, not with itself.

        A run re-sealed against a different Panel produces a receipt, a child and
        a Program that all agree; the only thing that disagrees is the workspace.
        """

        if program.authority_hash != authority.authority_hash:
            raise AuthoringError("research_authoring.evidence_authority_not_this_program")
        binding = receipt.input_binding
        if authority.execution_input_binding_hash is not None and (
            authority.execution_input_binding_hash
            != factor_execution_input_hash(
                binding.feature_panel_snapshot_hash,
                binding.causal_outcome_snapshot_hash,
                (
                    binding.target_policy_hash,
                    binding.walk_forward_policy_hash,
                    binding.screening_policy_hash,
                    binding.redundancy_policy_hash,
                ),
            )
        ):
            raise AuthoringError("factor_research.execution_input_binding_mismatch")
        if binding.authority_hash != authority.authority_hash:
            raise AuthoringError("research_authoring.evidence_authority_not_this_program")
        if binding.feature_panel_snapshot_hash != authority.panel_snapshot_hash:
            raise AuthoringError("factor_research.evidence_panel_not_this_authority")
        if binding.feature_panel_manifest_ref != authority.panel_manifest_ref:
            raise AuthoringError("factor_research.evidence_panel_not_this_authority")

    @staticmethod
    def _method_binding_refusal(*, program: SealedResearchProgram) -> str | None:
        """Why this build does not install the sealed method, or None when it does.

        The method binding and its parameter domain are reproduced by searching the
        installed enumerations for the pair that yields both. The search is exhaustive
        rather than clever because the enumerations are closed and tiny, and finding no
        pair means the Program was compiled against policies this build does not offer.
        A policy installed beside the chosen pair is not compared.
        """

        installed = any(
            factor_method_binding_hash(
                catalog_hash=program.catalog_hash,
                screening_policy=screening,
                redundancy_policy=redundancy,
            )
            == program.method_binding_hash
            and factor_parameter_domain_hash(
                screening_policy=screening, redundancy_policy=redundancy
            )
            == program.parameter_domain_hash
            for screening in INSTALLED_SCREENING_POLICIES
            for redundancy in INSTALLED_REDUNDANCY_POLICIES
        )
        return None if installed else "factor_research.evidence_method_binding_not_installed"

    @staticmethod
    def _verify_curation(
        *,
        child: FactorResearchDeterministicEvidence,
        review_binding_hash: str,
        output_workspace: Path,
        recorded_readback: bool = False,
    ) -> None:
        """Re-derive every curation decision filed against this checkpoint, if any.

        Optional by design and checked exactly when present. Curation is a
        separate act by a person or an automation *about* finished evidence, so a
        graph with none is complete and its absence proves nothing against the
        deterministic evidence.

        Re-derived rather than inspected. Comparing the decision's evidence and
        redundancy references against the checkpoint -- which is what this did --
        catches a decision about *other* evidence and misses a decision about
        this evidence that nobody ever admitted: an altered research input, an
        unacknowledged limitation, a decision policy the caller made up. The
        decision owner recompiles all of it from the checkpoint and the installed
        source, so a receipt replays exactly when it would have been persisted.

        Actor provenance is deliberately not evaluated. Which actor submitted is
        recorded, never weighed: ``ActorSubmissionBinding`` already refuses Agent
        process evidence on a non-Agent submission, and a verifier that started
        preferring one actor's decision would be the exact coupling the
        actor-neutral route exists to prevent.
        """

        for decision in FactorDevelopmentCurationReader(output_workspace).submissions(
            child.checkpoint_hash
        ):
            try:
                verify_factor_research_curation(
                    decision=decision,
                    checkpoint=child,
                    review_binding_hash=review_binding_hash,
                    recorded_readback=recorded_readback,
                )
            except FactorResearchDecisionAuthorityError as error:
                raise AuthoringError(
                    f"factor_research.evidence_curation_not_host_admitted:{error}"
                ) from error


__all__ = ["FactorEvidenceVerifier"]
