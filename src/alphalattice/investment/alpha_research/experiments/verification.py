"""Walk one published Alpha development graph back to its authority, by hash.

Until this existed, ``replay`` reported ``evidence_verifier_not_installed`` for
Alpha -- honest, and the reason Alpha could not claim exact reuse. Risk was the
only Desk that could.

What makes this a verifier rather than a reader: it never trusts a document to
describe itself. Every hash it compares has two independent sides. The receipt's
generic identities are compared with the authoring-layer evidence that sealed
them. Its children are compared with what the store returns for their own
addresses. Its target method is compared with the method the installed catalog
would build from the same declaration. Its split geometry is compared with the
embargo the outcome method's maturity clock implies.

A fully self-consistent forgery survives none of those, because self-consistency
is the one property a forger controls.

The last two compare against this build, so they answer what it admits. Replay
refuses a graph that fails them. A recorded readback is not a run: it opens the
graph and reports what a run would be told, in ``method_standing`` and
``method_refusal``. A forged receipt still reads as not current, and a study
saved before an upgrade retired or changed its method still opens.

Deliberately importable without the executor. The verifier must be able to run in
a process that resolves nothing that could compute, or "verified" would mean
"recomputed and agreed with itself".
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from alphalattice.foundation.causal_outcomes.execution.methods import (
    build_installed_execution_outcome_method_catalog,
)
from alphalattice.investment.alpha_research.targets.authority import (
    WholeUniverseAlphaTargetMethodBinding,
    resolve_installed_alpha_target_method,
)
from alphalattice.investment.alpha_research.targets.canonical import (
    CANONICAL_ALPHA_TARGET_RECIPE_ID,
)
from alphalattice.kernel.shared_kernel.sector_treatment import (
    SECTOR_HISTORY_BACKFILLED,
    SECTOR_HISTORY_FORWARD,
)
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExecutionEvidence,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)

from .authoring import ALPHA_EXPERIMENT_KIND
from .development_artifacts import (
    ALPHA_DEVELOPMENT_EXECUTION_RECEIPT_CATEGORY,
    CANONICAL_SCORE_BINDING_CATEGORY,
    CANONICAL_SCORE_SURFACE_CATEGORY,
    AlphaDevelopmentArtifactStore,
)
from .development_contracts import (
    AlphaDevelopmentExecutionReceipt,
    CanonicalAlphaScoreSurface,
)
from .development_evidence import (
    AlphaDevelopmentReceiptReader,
    AlphaDevelopmentVerifiedGraph,
    verify_alpha_development_execution_receipt,
)
from .family_qualification import QUALIFICATION_CATEGORY, read_qualification_receipt
from .lifecycle_authoring import CATEGORY as LIFECYCLE_CATEGORY
from .lifecycle_authoring import EARLIER_SCHEME_REFUSAL, verify_lifecycle_research
from .policies import load_alpha_split_policy
from .target_preprocessing import (
    reconcile_target_preprocessing_receipt,
    verify_comparison_evidence,
)


class AlphaMethodologyEvidenceVerifier(Protocol):
    """Injected recursive owner for an installed cross-Desk Alpha methodology."""

    def applies(self, evidence: ResearchExecutionEvidence) -> bool: ...

    def verify(
        self,
        *,
        program: SealedResearchProgram,
        evidence: ResearchExecutionEvidence,
        authority: ResolvedResearchAuthority | None,
        output_workspace: Path,
    ) -> None: ...


class AlphaEvidenceVerifier:
    """Desk-owned recursive verification of one Alpha development evidence graph."""

    kind = ALPHA_EXPERIMENT_KIND

    def __init__(
        self, *, methodology_verifier: AlphaMethodologyEvidenceVerifier | None = None
    ) -> None:
        self._methodology_verifier = methodology_verifier
        self.verified_graph: AlphaDevelopmentVerifiedGraph | None = None
        """The graph the most recent ``verify`` walked and proved, for the caller
        of that verification to project inside the same request. A value the
        walk produced, not a substitute for it: the next ``verify`` replaces it,
        and it says nothing about the artifacts after the walk."""
        self.method_standing: Literal["INSTALLED", "NOT_CURRENT"] | None = None
        """Whether this build admits the target method and split the last walked
        receipt was sealed under; replay refuses one it does not, a recorded
        readback reports it. None when the graph carries no such receipt."""
        self.method_refusal: str | None = None
        """The refusal a run would get for that method, or None when it is admitted."""

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
        """Verify sealed contents, not this build's admission of their method."""
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
        self.verified_graph = None
        self.method_standing = None
        self.method_refusal = None
        if evidence.kind != self.kind or program.kind != self.kind:
            raise AuthoringError("alpha_research.evidence_kind_mismatch")
        if any(LIFECYCLE_CATEGORY in uri for uri in evidence.artifact_uris):
            scheme = verify_lifecycle_research(
                program=program, evidence=evidence, output_workspace=output_workspace
            )
            # A Program sealed before P bound the implementation as well: every child still
            # verifies, it reads back as recorded, and a run of it is refused as historical.
            refusal = None if scheme == "CURRENT" else EARLIER_SCHEME_REFUSAL
            if refusal is not None and not recorded_readback:
                raise AuthoringError(refusal)
            self.method_standing = "INSTALLED" if refusal is None else "NOT_CURRENT"
            self.method_refusal = refusal
            return
        if any(f"/{QUALIFICATION_CATEGORY}/" in uri for uri in evidence.artifact_uris):
            # A qualification's receipt binds its Program; its terminal children are read at
            # the Alpha owner's store by the readback, which holds the workspace (GR3).
            read_qualification_receipt(
                program=program, evidence=evidence, output_workspace=output_workspace
            )
            self.method_standing = "INSTALLED"
            return
        if self._methodology_verifier is not None and self._methodology_verifier.applies(evidence):
            self._methodology_verifier.verify(
                program=program,
                evidence=evidence,
                authority=authority,
                output_workspace=output_workspace,
            )
            return
        # Accepted and deliberately unused. This receipt names no input surface
        # of its own: it binds the run through ``program_hash``, which is checked
        # below against the Program the workflow sealed from this very
        # ``authority``. Re-deriving the same fact by a second route would read as
        # a stronger claim than it is.
        del authority
        store = AlphaDevelopmentArtifactStore(Path(output_workspace) / "alpha-development")
        reader = AlphaDevelopmentReceiptReader(Path(output_workspace) / "alpha-development")

        handles = tuple(
            uri
            for uri in evidence.artifact_uris
            if ALPHA_DEVELOPMENT_EXECUTION_RECEIPT_CATEGORY in uri
        )
        if len(handles) != 1:
            # One receipt is the root of one run. Zero means the evidence names
            # nothing this Desk owns; more than one means the graph has two roots
            # and no rule says which is authoritative.
            raise AuthoringError("alpha_research.evidence_receipt_not_unique")

        # The reader walks every child, proves each belongs to its parent, and
        # proves the listed children are all of them. Not re-implemented here.
        graph = reader.read(handles[0])
        receipt = graph.receipt
        verify_alpha_development_execution_receipt(receipt=receipt, evidence=evidence)
        if receipt.program_hash != program.program_hash:
            raise AuthoringError("alpha_research.evidence_program_mismatch")

        method_refusal = self._target_method_refusal(receipt) or self._embargo_refusal(receipt)
        if method_refusal is not None and not recorded_readback:
            raise AuthoringError(method_refusal)
        # Surfaces are named by the evidence rather than by the receipt, because
        # a surface points *at* a receipt: a receipt listing its surfaces could
        # not be sealed until they existed, and they cannot exist until it is.
        surface_hashes = tuple(
            uri.rsplit("/", 1)[-1]
            for uri in evidence.artifact_uris
            if f"/{CANONICAL_SCORE_SURFACE_CATEGORY}/" in uri
        )
        if (
            receipt.target_recipe_binding.target_recipe_id == CANONICAL_ALPHA_TARGET_RECIPE_ID
            and not surface_hashes
        ):
            # Omission is the cheapest forgery. A run that fitted the canonical
            # target and then published evidence naming no score surface would
            # otherwise reach an empty loop and pass, so absence is refused
            # rather than iterated over.
            raise AuthoringError("alpha_research.evidence_canonical_score_surface_absent")
        self._verify_score_surfaces(store=store, receipt=receipt, surface_hashes=surface_hashes)
        self._verify_study_evidence(
            store=store,
            evidence_uris=evidence.artifact_uris,
            receipt=receipt,
        )
        self._verify_score_bindings(
            store=store,
            surface_hashes=surface_hashes,
            binding_hashes=tuple(
                uri.rsplit("/", 1)[-1]
                for uri in evidence.artifact_uris
                if f"/{CANONICAL_SCORE_BINDING_CATEGORY}/" in uri
            ),
        )
        self.method_standing = "INSTALLED" if method_refusal is None else "NOT_CURRENT"
        self.method_refusal = method_refusal
        self.verified_graph = graph

    @staticmethod
    def _target_method_refusal(receipt: AlphaDevelopmentExecutionReceipt) -> str | None:
        """Rebuild the declared target method from the installed catalog.

        The receipt records which method it used. That is a claim about
        methodology, and a re-sealed receipt naming an uninstalled or altered
        composition validates its own hash perfectly. The comparison has to be
        against what this build would construct from the same declaration.

        Only the two per-run facts are taken from the evidence -- the Sector
        revision the workspace neutralized against and the outcome recipe the
        snapshot was published under. Everything else is the installed method.
        Returns the refusal a run would get, or None when the method is installed.
        """

        binding = receipt.target_recipe_binding
        # Either treatment is an installed construction; the Panel decides which a run read,
        # and the binding's hash says which it used (V346).
        for treatment in (SECTOR_HISTORY_BACKFILLED, SECTOR_HISTORY_FORWARD):
            try:
                method = resolve_installed_alpha_target_method(
                    target_recipe_id=binding.target_recipe_id,
                    execution_outcome_recipe_id=binding.execution_outcome_recipe_id,
                    sector_revision=(
                        None
                        if isinstance(binding, WholeUniverseAlphaTargetMethodBinding)
                        else binding.sector_revision
                    ),
                    sector_history_treatment=treatment,
                )
            except ValueError:
                return "alpha_research.evidence_target_method_not_installed"
            if (
                method.target_method_hash == binding.target_method_hash
                and method.standardization_id == binding.standardization_id
            ):
                return None
        return "alpha_research.evidence_target_method_not_installed"

    @staticmethod
    def _embargo_refusal(receipt: AlphaDevelopmentExecutionReceipt) -> str | None:
        """Require the split the run used to be the one this method's clock implies.

        An embargo recorded but never re-derived is the defect class this program
        keeps closing: a re-sealed receipt claiming ``embargo_sessions: 0`` under
        a six-session method would train on labels that overlap its own
        validation window, and every hash in the graph would still agree.
        Returns the refusal a run would get, or None when the split is the one
        this build derives. The receipt's own seal over its split is integrity,
        not admission, and refuses in every mode.
        """

        policy = receipt.split_policy
        if receipt.split_policy_hash != policy.policy_hash:
            raise AuthoringError("alpha_research.evidence_embargo_not_installed")
        method_catalog = build_installed_execution_outcome_method_catalog()
        try:
            recipe = method_catalog.resolve(
                receipt.target_recipe_binding.execution_outcome_recipe_id
            )
        except ValueError:
            return "alpha_research.evidence_outcome_method_not_installed"
        geometry = load_alpha_split_policy()
        expected_embargo = recipe.maturity_lag_sessions - 1
        expected_step = geometry.validation_sessions + expected_embargo
        # Re-derived from the installed geometry and the method's own clock, then
        # compared field by field against the policy the run sealed. The fold
        # count is deliberately not re-derived here: it depends on the length of
        # an axis this verifier does not read, and the executor's prediction was
        # already cross-checked against the splitter's output when the plan was
        # built. What must not drift is the embargo and the step that carries it.
        if (
            policy.embargo_sessions != expected_embargo
            or policy.step_sessions != expected_step
            or policy.train_sessions != geometry.train_sessions
            or policy.purge_sessions != geometry.purge_sessions
            or policy.validation_sessions != geometry.validation_sessions
            or policy.sealed_holdout_sessions != geometry.sealed_holdout_sessions
            or policy.minimum_folds != geometry.minimum_folds
        ):
            return "alpha_research.evidence_embargo_not_installed"
        return None

    @staticmethod
    def _verify_score_surfaces(
        *,
        store: AlphaDevelopmentArtifactStore,
        receipt: AlphaDevelopmentExecutionReceipt,
        surface_hashes: tuple[str, ...],
    ) -> None:
        """Re-read every score surface this receipt authored and re-join it to the folds.

        A surface is only meaningful if it aggregates *this* receipt's children.
        One that names another run's receipt, or a fold this receipt never
        produced, is a well-formed document about somebody else's evidence.
        """

        for surface_hash in surface_hashes:
            try:
                surface: CanonicalAlphaScoreSurface = store.load_canonical_score_surface(
                    surface_hash
                )
            except FileNotFoundError as error:
                raise AuthoringError("alpha_research.evidence_score_surface_unavailable") from error
            except ValueError as error:
                raise AuthoringError("alpha_research.evidence_score_surface_invalid") from error
            if surface.development_receipt_hash != receipt.receipt_hash:
                raise AuthoringError("alpha_research.evidence_score_surface_not_this_receipt")
            if surface.target_recipe_binding_hash != receipt.target_recipe_binding.binding_hash:
                raise AuthoringError("alpha_research.evidence_score_surface_target_mismatch")
            lineage = {
                (entry.candidate_id, entry.fold_index): entry for entry in receipt.child_lineage
            }
            for ref in surface.ordered_fold_refs:
                entry = lineage.get((surface.candidate_id, ref.fold_index))
                if entry is None:
                    raise AuthoringError("alpha_research.evidence_score_surface_fold_unbound")
                if (
                    entry.numerical_result_hash != ref.numerical_result_hash
                    or entry.fold_commitment_hash != ref.fold_commitment_hash
                ):
                    raise AuthoringError("alpha_research.evidence_score_surface_fold_mismatch")

    @staticmethod
    def _verify_score_bindings(
        *,
        store: AlphaDevelopmentArtifactStore,
        surface_hashes: tuple[str, ...],
        binding_hashes: tuple[str, ...],
    ) -> None:
        """Every published binding must name a surface this same evidence carries.

        A binding is what G4 consumes, and it is a claim about which scores were
        fitted to which target. One naming a surface that is not part of this
        graph would be a well-formed pointer into somebody else's run, and the
        Portfolio side has no way to notice.
        """

        if surface_hashes and not binding_hashes:
            raise AuthoringError("alpha_research.evidence_score_binding_absent")
        for binding_hash in binding_hashes:
            try:
                binding = store.load_canonical_score_binding(binding_hash)
            except FileNotFoundError as error:
                raise AuthoringError("alpha_research.evidence_score_binding_unavailable") from error
            except ValueError as error:
                raise AuthoringError("alpha_research.evidence_score_binding_invalid") from error
            if binding.alpha_score_surface_hash not in surface_hashes:
                raise AuthoringError("alpha_research.evidence_score_binding_surface_unbound")
            surface = store.load_canonical_score_surface(binding.alpha_score_surface_hash)
            if (
                binding.candidate_id != surface.candidate_id
                or binding.target_recipe_binding_hash != surface.target_recipe_binding_hash
                or binding.target_evidence_hash != surface.target_evidence_hash
            ):
                raise AuthoringError("alpha_research.evidence_score_binding_surface_mismatch")

    @staticmethod
    def _verify_study_evidence(
        *,
        store: AlphaDevelopmentArtifactStore,
        evidence_uris: tuple[str, ...],
        receipt: AlphaDevelopmentExecutionReceipt,
    ) -> None:
        """Walk the study documents this evidence names, when it names any.

        A comparison Program is a claim that three runs were one study, and its
        Evidence is a claim that these three receipts are those runs. Neither is
        checkable from inside itself: the Program validates its own arm order and
        pairing, and a forged Evidence naming three unrelated receipts satisfies
        every one of its own rules. So the Program is re-read by the hash the
        Evidence names, the two are re-joined, and this receipt is required to be
        one of the arms.

        A preprocessing receipt is re-read with its detail child and reconciled,
        which is the same comparison the producer made -- run again here against
        what is durably on disk rather than against what was in memory.
        """

        program_hashes = tuple(
            uri.rsplit("/", 1)[-1] for uri in evidence_uris if "/comparison-programs/" in uri
        )
        study_hashes = tuple(
            uri.rsplit("/", 1)[-1] for uri in evidence_uris if "/comparison-evidence/" in uri
        )
        if bool(program_hashes) != bool(study_hashes):
            # One without the other is a study with no design, or a design that
            # nothing executed. Either way the pair is what carries the claim.
            raise AuthoringError("alpha_research.evidence_study_pair_incomplete")
        for study_hash in study_hashes:
            try:
                study = store.load_comparison_evidence(study_hash)
                program = store.load_comparison_program(study.program_hash)
            except FileNotFoundError as error:
                raise AuthoringError("alpha_research.evidence_study_unavailable") from error
            except ValueError as error:
                raise AuthoringError("alpha_research.evidence_study_invalid") from error
            if study.program_hash not in program_hashes:
                raise AuthoringError("alpha_research.evidence_study_program_unbound")
            verify_comparison_evidence(program=program, evidence=study)
            if receipt.receipt_hash not in {
                result.development_receipt_hash for result in study.ordered_results
            }:
                # The study would otherwise be able to name three receipts none
                # of which is the run this evidence is about.
                raise AuthoringError("alpha_research.evidence_study_receipt_not_an_arm")

        for receipt_hash in (
            uri.rsplit("/", 1)[-1]
            for uri in evidence_uris
            if "/target-preprocessing-receipts/" in uri
        ):
            try:
                preprocessing = store.load_target_preprocessing_receipt(receipt_hash)
                detail = store.load_target_preprocessing_detail(preprocessing.clipping_detail_hash)
            except FileNotFoundError as error:
                raise AuthoringError("alpha_research.evidence_preprocessing_unavailable") from error
            except ValueError as error:
                raise AuthoringError("alpha_research.evidence_preprocessing_invalid") from error
            try:
                reconcile_target_preprocessing_receipt(receipt=preprocessing, detail=detail)
            except ValueError as error:
                raise AuthoringError(
                    "alpha_research.evidence_preprocessing_not_reconciled"
                ) from error


__all__ = ["AlphaEvidenceVerifier", "AlphaMethodologyEvidenceVerifier"]
