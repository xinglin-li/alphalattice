"""Read one Alpha development execution receipt and prove its children.

The receipt is written by the Desk executor through
``AlphaDevelopmentArtifactStore``. This is the other half: resolving one by exact
identity and checking that the children it names are the children that exist.

Reader and handle spelling live together for the same reason the Factor pair do:
split across two modules they drift, and the failure is silent -- a reader that
resolves a slightly different path finds nothing and reports "no receipt" rather
than "wrong root".

The receipt validates its own ``receipt_hash`` on parse, which proves it has not
been edited and proves nothing about whether the artifacts it names agree with
it. So every lineage entry is resolved against the store here, and the generic
Program identities are compared against the sealed
``ResearchExecutionEvidence`` in ``verify_alpha_development_execution_receipt``.

Deliberately **not** importable from ``alpha_research.publication``: a development
receipt is not a published Alpha result and must never be reachable from the owner
that moves a current pointer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from alphalattice.investment.alpha_research.experiments.contracts import (
    AlphaCandidateStatus,
    AlphaExperimentBatchResult,
    AlphaExperimentCandidateResult,
)
from alphalattice.investment.alpha_research.experiments.development_artifacts import (
    ALPHA_DEVELOPMENT_EXECUTION_RECEIPT_CATEGORY,
    AlphaDevelopmentArtifactReadbackError,
    AlphaDevelopmentArtifactStore,
)
from alphalattice.investment.alpha_research.experiments.development_contracts import (
    AlphaCandidateDevelopmentReport,
    AlphaCandidateFoldEvidence,
    AlphaCandidateInferenceEvidence,
    AlphaCandidateNumericalFoldResult,
    AlphaDevelopmentChildLineage,
    AlphaDevelopmentExecutionReceipt,
    AlphaDevelopmentFoldSurface,
    AlphaDevelopmentSurfaceManifest,
    LegacyAlphaCandidateFoldEvidence,
)
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExecutionEvidence,
)

_RECEIPT_URI_PREFIX = AlphaDevelopmentArtifactStore.uri(
    f"current/{ALPHA_DEVELOPMENT_EXECUTION_RECEIPT_CATEGORY}", "0" * 64
)[:-64]
"""The one URI spelling, taken from the store rather than restated beside it."""

_HASH_CHARACTERS = frozenset("0123456789abcdef")


def alpha_development_receipt_uri(receipt_hash: str) -> str:
    return f"{_RECEIPT_URI_PREFIX}{receipt_hash}"


def alpha_development_receipt_handle(uri: str) -> str:
    """Recover the receipt hash from either the canonical URI or the bare hash.

    Both, because the generic ``ResearchExecutionEvidence.artifact_uris`` publishes
    the URI and that is the value a consumer actually holds, while a researcher
    copying an identity out of a run report has the hash. A reader accepting only
    one of the two would refuse the form its own producer emits.

    Admitted as **exactly** 64 lowercase hex and nothing else: a content address
    that is not a content address should fail at the boundary that owns the
    spelling, rather than further in as a missing file.
    """

    handle = uri[len(_RECEIPT_URI_PREFIX) :] if uri.startswith(_RECEIPT_URI_PREFIX) else uri
    if len(handle) != 64 or not set(handle).issubset(_HASH_CHARACTERS):
        raise AuthoringError("alpha_research.development_receipt_handle_invalid")
    return handle


@dataclass(frozen=True, slots=True)
class AlphaDevelopmentVerifiedGraph:
    """One receipt's graph as one walk of the reader proved it, in the walk's order.

    A value, not a cache: it is what ``AlphaDevelopmentReceiptReader.read``
    answered for one handle at one moment, and a caller that still holds it
    inside the request that made it may project it without walking again. It
    proves nothing about the artifacts after that moment.
    """

    receipt: AlphaDevelopmentExecutionReceipt
    batch: AlphaExperimentBatchResult
    folds: tuple[AlphaCandidateNumericalFoldResult, ...]
    candidate_reports: tuple[AlphaCandidateDevelopmentReport, ...]
    """One per batch candidate, in the batch's order: the development report the
    candidate result names, proved to belong to this execution."""

    inference_evidence: tuple[AlphaCandidateInferenceEvidence, ...]
    """One per batch candidate, in the batch's order, proved against the report
    and the numerical children the walk resolved."""

    score_tables: Mapping[str, pa.Table]
    """By numerical result hash: the score chunk rows the walk proved while loading
    that child (absent when the result carries no score chunk). A consumer that
    joins scores inside the request that made this graph reads these rows and
    proves no chunk a second time; the next request walks again."""

    fold_surfaces: Mapping[str, AlphaDevelopmentFoldSurface]
    """By fold surface hash: every fold surface the completeness check loaded."""

    validation_tables: Mapping[str, pa.Table]
    """By fold surface hash: the validation chunk rows proved with that surface."""


class AlphaDevelopmentReceiptReader:
    """Load one Alpha development receipt by exact hash and walk its children.

    Read-only. It owns no pointer, admits nothing for publication, and has no
    "latest", no listing and no search: selecting the newest file in a directory
    is how a run silently binds to evidence nobody chose.
    """

    def __init__(self, artifact_root: Path) -> None:
        self._store = AlphaDevelopmentArtifactStore(Path(artifact_root))

    @property
    def root(self) -> Path:
        return self._store.root

    def projection(self, handle: str) -> dict[str, Any]:
        """Exact verified report facts, not another metric calculation."""
        return self.projection_of(self.read(handle))

    @staticmethod
    def projection_of(graph: AlphaDevelopmentVerifiedGraph) -> dict[str, Any]:
        """The report facts of a graph this reader verified, as the walk listed them."""
        return {
            "receipt": graph.receipt.model_dump(mode="json"),
            "result": graph.batch.model_dump(mode="json"),
            "fold_results": [fold.model_dump(mode="json") for fold in graph.folds],
            "candidate_reports": [
                report.model_dump(mode="json") for report in graph.candidate_reports
            ],
            "inference_evidence": [
                evidence.model_dump(mode="json") for evidence in graph.inference_evidence
            ],
        }

    def load(self, handle: str) -> AlphaDevelopmentExecutionReceipt:
        """Verify the complete graph afresh; retain nothing on this reader."""
        return self.read(handle).receipt

    def summary(self, evidence: ResearchExecutionEvidence) -> dict[str, Any]:
        """Identity-bound saved metrics only; no estimator/score/Feature graph proof.

        A fast first reading, not an alternative authority or a computation input.
        The Host also binds the evidence to its completed Task's publication.
        """
        if len(evidence.artifact_uris) != 1:
            raise AuthoringError("alpha_research.development_receipt_handle_invalid")
        receipt = self._store.load_development_execution_receipt(
            alpha_development_receipt_handle(evidence.artifact_uris[0])
        )
        batch = self._store.load_batch_result(receipt.batch_result_hash)
        if (
            receipt.program_hash != evidence.program_hash
            or receipt.desk_program_hash != evidence.desk_program_hash
            or receipt.method_binding_hash != evidence.method_binding_hash
            or receipt.desk_input_binding_hash != evidence.desk_input_binding_hash
            or batch.result_hash != receipt.batch_result_hash
            or batch.batch_hash != receipt.batch_hash
        ):
            raise AuthoringError("alpha_research.summary_publication_binding_mismatch")
        return {
            "receipt": {"receipt_hash": receipt.receipt_hash},
            "result": batch.model_dump(mode="json"),
            "fold_results": [],
        }

    def read(self, handle: str) -> AlphaDevelopmentVerifiedGraph:
        """Verify the complete graph afresh and hand it back whole; retain nothing."""
        receipt, batch, folds, reports, inference, tables = self._read_verified(handle)
        score_tables, fold_surfaces, validation_tables = tables
        return AlphaDevelopmentVerifiedGraph(
            receipt=receipt,
            batch=batch,
            folds=folds,
            candidate_reports=reports,
            inference_evidence=inference,
            score_tables=score_tables,
            fold_surfaces=fold_surfaces,
            validation_tables=validation_tables,
        )

    def _read_verified(
        self, handle: str
    ) -> tuple[
        AlphaDevelopmentExecutionReceipt,
        AlphaExperimentBatchResult,
        tuple[AlphaCandidateNumericalFoldResult, ...],
        tuple[AlphaCandidateDevelopmentReport, ...],
        tuple[AlphaCandidateInferenceEvidence, ...],
        tuple[
            dict[str, pa.Table],
            dict[str, AlphaDevelopmentFoldSurface],
            dict[str, pa.Table],
        ],
    ]:
        """Read exactly the named receipt, prove every child, and prove they are all of them.

        The walk is entry by entry rather than a comparison of flat tuples. A
        receipt carrying one list of estimator hashes and another of fit hashes
        can be internally consistent while no estimator in it belongs to the fold
        the fit was for, because nothing joins the two lists.

        Resolving the listed children only proves the listed ones are real. It
        says nothing about the ones that are not listed, so a receipt with a child
        removed and resealed used to pass. Completeness is checked against
        artifacts sealed at other moments -- the surface manifest before any fit,
        the batch result after the run -- in ``_assert_child_set_complete``.

        The batch result names, per candidate, a development report and an
        inference evidence record, and the report names one thin fold evidence
        per fold. They are the run's own account of its children and the
        artifacts a downstream reader consumes, so the walk proves each of
        them too: it exists under its exact hash, it belongs to this program,
        surface, candidate and card, and every child it lists is the child the
        lineage just resolved, at the same fold position. A report that reads
        back but binds another candidate's estimators, or a fold evidence list
        with a fold missing, repeated or out of order, is refused here rather
        than discovered by whichever consumer reads it next.
        """

        receipt_hash = alpha_development_receipt_handle(handle)
        try:
            receipt = self._store.load_development_execution_receipt(receipt_hash)
        except FileNotFoundError as error:
            raise AuthoringError("alpha_research.development_receipt_unavailable") from error
        except (ValueError, AlphaDevelopmentArtifactReadbackError) as error:
            raise AuthoringError("alpha_research.development_receipt_invalid") from error
        if receipt.receipt_hash != receipt_hash:
            raise AuthoringError("alpha_research.development_receipt_identity_conflict")

        materialization = receipt.target_materialization_binding
        records = {value.fold_commitment_hash: value for value in materialization.fold_records}
        batch, surfaces, fold_surfaces, validation_tables = self._assert_child_set_complete(receipt)
        folds = []
        score_tables: dict[str, pa.Table] = {}
        results_by_candidate: dict[str, list[AlphaCandidateNumericalFoldResult]] = {}
        entries_by_candidate: dict[str, list[AlphaDevelopmentChildLineage]] = {}
        for entry in receipt.child_lineage:
            try:
                result, scores = self._store.read_candidate_numerical_fold_result(
                    entry.numerical_result_hash
                )
                state = self._store.load_development_estimator_state(entry.estimator_state_hash)
                evidence = self._store.load_development_fit_evidence(entry.fit_evidence_hash)
            except (
                FileNotFoundError,
                ValueError,
                AlphaDevelopmentArtifactReadbackError,
            ) as error:
                raise AuthoringError("alpha_research.development_receipt_child_unavailable") from (
                    error
                )
            if (
                result.candidate_id != entry.candidate_id
                or result.fold_index != entry.fold_index
                or result.fold_commitment_hash != entry.fold_commitment_hash
                or result.estimator_state_hash != entry.estimator_state_hash
                or result.development_surface_binding_hash
                != receipt.development_surface_binding_hash
            ):
                raise AuthoringError("alpha_research.development_receipt_child_result_mismatch")
            if (
                state.candidate_id != entry.candidate_id
                or state.fold_index != entry.fold_index
                or state.fold_commitment_hash != entry.fold_commitment_hash
                or state.fit_evidence_hash != entry.fit_evidence_hash
                or state.ordered_factor_ids != entry.ordered_factor_ids
            ):
                raise AuthoringError("alpha_research.development_receipt_child_state_mismatch")
            if (
                evidence.candidate_id != entry.candidate_id
                or evidence.fold_index != entry.fold_index
                or evidence.training_binding_hash != entry.training_input_binding_hash
                or evidence.estimator_content_hash != entry.estimator_content_hash
                or evidence.fit_provenance_hash != entry.fit_provenance_hash
                or evidence.score_evidence_hash != entry.score_evidence_hash
                or evidence.adapter_id != receipt.model_adapter_id
                or evidence.recipe_hash != receipt.model_adapter_recipe_hash
            ):
                raise AuthoringError("alpha_research.development_receipt_child_fit_mismatch")
            if entry.fold_commitment_hash not in records:
                # Already refused at parse, and checked again against the values
                # actually read back rather than against the receipt's own copy.
                raise AuthoringError("alpha_research.development_receipt_child_fold_unbound")
            folds.append(result)
            if scores is not None:
                score_tables[entry.numerical_result_hash] = scores
            results_by_candidate.setdefault(entry.candidate_id, []).append(result)
            entries_by_candidate.setdefault(entry.candidate_id, []).append(entry)
        reports = []
        inference = []
        for candidate in batch.candidates:
            report, evidence = self._verify_candidate_evidence(
                candidate,
                batch=batch,
                surface=surfaces[str(candidate.development_surface_hash)],
                entries=tuple(entries_by_candidate.get(candidate.candidate_id, ())),
                results=tuple(results_by_candidate.get(candidate.candidate_id, ())),
            )
            reports.append(report)
            inference.append(evidence)
        return (
            receipt,
            batch,
            tuple(folds),
            tuple(reports),
            tuple(inference),
            (score_tables, fold_surfaces, validation_tables),
        )

    def _verify_candidate_evidence(
        self,
        candidate: AlphaExperimentCandidateResult,
        *,
        batch: AlphaExperimentBatchResult,
        surface: AlphaDevelopmentSurfaceManifest,
        entries: tuple[AlphaDevelopmentChildLineage, ...],
        results: tuple[AlphaCandidateNumericalFoldResult, ...],
    ) -> tuple[AlphaCandidateDevelopmentReport, AlphaCandidateInferenceEvidence]:
        """Prove the report, the fold evidence and the inference evidence of one candidate.

        ``entries`` and ``results`` are this candidate's lineage entries and the
        numerical results the walk already resolved for them, in fold order;
        every reference the three artifacts make to a child is checked against
        those, never by loading a child again. Bindings are equalities against
        the batch result, the surface manifest and the resolved children, in
        the terms each contract actually carries: a legacy fold evidence is
        held to the fields it has, and to no field it does not.
        """

        fold_count = len(surface.fold_commitment_hashes)
        try:
            report = self._store.load_candidate_report(candidate.development_report_hash)
        except (FileNotFoundError, ValueError, AlphaDevelopmentArtifactReadbackError) as error:
            raise AuthoringError("alpha_research.development_receipt_report_unavailable") from (
                error
            )
        estimator_hashes = tuple(entry.estimator_state_hash for entry in entries)
        if (
            report.request_hash != batch.program_hash
            or report.development_surface_hash != candidate.development_surface_hash
            or report.foundation_hash != surface.foundation_hash
            or report.candidate_id != candidate.candidate_id
            or report.candidate_card_hash != candidate.spec_hash
            # A receipt exists only for a fully successful batch; a report that
            # says otherwise is not this run's report.
            or report.status is not AlphaCandidateStatus.SUCCEEDED
            or report.failure_codes
            or report.admitted_fold_count != fold_count
            or len(report.fold_evidence_hashes) != fold_count
            or len(set(report.fold_evidence_hashes)) != fold_count
            or report.estimator_state_hashes != estimator_hashes
            or candidate.estimator_state_hashes != estimator_hashes
            or any(result.role is not report.role for result in results)
        ):
            raise AuthoringError("alpha_research.development_receipt_report_mismatch")
        # The aggregate is the report's; what is checked is that it aggregates
        # exactly the fold metrics the resolved children carry, and that the
        # batch's summary of it is a copy of it.
        metrics = report.metrics
        if (
            metrics is None
            or metrics.candidate_id != candidate.candidate_id
            or metrics.fold_metrics != tuple(result.metrics for result in results)
            or candidate.pooled_oos_r2 != metrics.zero_relative_oos_r2.value
            or candidate.mean_rank_ic != metrics.rank_ic_mean.value
            or candidate.mean_gross_decile_spread != metrics.gross_decile_spread_mean.value
            or candidate.fold_coverage_mean != metrics.fold_coverage_mean
        ):
            raise AuthoringError("alpha_research.development_receipt_report_mismatch")

        fold_surface_hashes = candidate.fold_surface_hashes or ()
        for index, fold_evidence_hash in enumerate(report.fold_evidence_hashes):
            try:
                fold = self._store.load_candidate_fold_evidence(fold_evidence_hash)
            except (
                FileNotFoundError,
                ValueError,
                AlphaDevelopmentArtifactReadbackError,
            ) as error:
                raise AuthoringError(
                    "alpha_research.development_receipt_fold_evidence_unavailable"
                ) from error
            entry, result = entries[index], results[index]
            if (
                fold.request_hash != batch.program_hash
                or fold.development_surface_hash != candidate.development_surface_hash
                or fold.surface_fold_hash != fold_surface_hashes[index]
                or fold.candidate_id != candidate.candidate_id
                or fold.candidate_card_hash != candidate.spec_hash
                or fold.fold_index != index
                or fold.fold_commitment_hash != entry.fold_commitment_hash
                or fold.execution_binding_hash != result.execution_binding_hash
                or (
                    fold.numerical_result_hash is not None
                    and fold.numerical_result_hash != entry.numerical_result_hash
                )
            ):
                raise AuthoringError("alpha_research.development_receipt_fold_evidence_mismatch")
            if isinstance(fold, AlphaCandidateFoldEvidence):
                # The thin contract names its numerical child; it must be the one
                # the lineage resolved at this fold.
                if fold.numerical_result_hash != entry.numerical_result_hash:
                    raise AuthoringError(
                        "alpha_research.development_receipt_fold_evidence_mismatch"
                    )
            elif isinstance(fold, LegacyAlphaCandidateFoldEvidence) and not fold.restates(result):
                # The facts a legacy parent repeated beside the numerical result
                # -- role, status, metrics, session series, fit ledger, failure,
                # the score chunk by its shape and the estimator -- are that
                # result's or the evidence is not this fold's.
                raise AuthoringError("alpha_research.development_receipt_fold_evidence_mismatch")

        try:
            inference = self._store.load_candidate_inference_evidence(
                candidate.inference_evidence_hash
            )
        except (FileNotFoundError, ValueError, AlphaDevelopmentArtifactReadbackError) as error:
            raise AuthoringError(
                "alpha_research.development_receipt_inference_unavailable"
            ) from error
        if (
            inference.request_hash != batch.program_hash
            or inference.candidate_id != candidate.candidate_id
            or inference.report_hash != report.report_hash
            or inference.admitted_fold_count != fold_count
            or inference.successful_fold_count != fold_count
            or inference.score_chunk_hashes
            != tuple(
                None if result.score_chunk is None else result.score_chunk.content_hash
                for result in results
            )
            or inference.estimator_state_hashes != estimator_hashes
            or inference.scored_row_count != metrics.scored_row_count
            or inference.common_surface_row_count != metrics.common_surface_row_count
            or inference.scored_row_count
            != sum(
                0 if result.metrics is None else result.metrics.scored_comparison_row_count
                for result in results
            )
            or inference.common_surface_row_count
            != sum(
                0 if result.metrics is None else result.metrics.comparison_row_count
                for result in results
            )
        ):
            raise AuthoringError("alpha_research.development_receipt_inference_mismatch")
        return report, inference

    def _assert_child_set_complete(
        self, receipt: AlphaDevelopmentExecutionReceipt
    ) -> tuple[
        AlphaExperimentBatchResult,
        dict[str, AlphaDevelopmentSurfaceManifest],
        dict[str, AlphaDevelopmentFoldSurface],
        dict[str, pa.Table],
    ]:
        """Prove the lineage is the whole child set, not merely a truthful subset.

        Two artifacts sealed at other moments answer this. The surface manifest is
        written **before any fit** and names every candidate, every candidate card
        and every fold commitment, so it states the axis the run was supposed to
        cover. The batch result is sealed **after** the run and names every
        numerical child that was produced. Neither is an independent producer --
        one execution chain writes all of them -- but each is separately
        content-addressed and the store refuses reusing an identity with different
        content.

        The manifest is what makes a *consistent* forgery detectable. Comparing
        the receipt only with the batch result catches a receipt that dropped a
        child on its own, and not one where the batch and the receipt dropped the
        same child together. So the counts are stated against the pre-fit axis --
        candidates, candidate cards, folds -- rather than left to follow from the
        two post-run records agreeing with each other.

        A receipt exists only for a fully successful batch: ``_child_lineage``
        refuses a fold without an estimator. That is what makes these equalities
        rather than subset checks.
        """

        try:
            batch = self._store.load_batch_result(receipt.batch_result_hash)
        except (FileNotFoundError, ValueError, AlphaDevelopmentArtifactReadbackError) as error:
            raise AuthoringError("alpha_research.development_receipt_batch_unavailable") from error
        if batch.result_hash != receipt.batch_result_hash or batch.batch_hash != receipt.batch_hash:
            raise AuthoringError("alpha_research.development_receipt_batch_mismatch")
        if (
            batch.program_hash != receipt.development_program_hash
            or batch.development_surface_hash != receipt.development_surface_hash
            or batch.development_surface_binding_hash != receipt.development_surface_binding_hash
            or batch.fold_surface_hashes != receipt.fold_surface_hashes
        ):
            # The receipt copied these from the batch, so a batch re-pointed at
            # another surface -- the way a forgery would reach a fold axis it
            # controls -- no longer resolves under the receipt anybody was handed.
            raise AuthoringError("alpha_research.development_receipt_batch_mismatch")

        records = receipt.target_materialization_binding.fold_records
        commitments = tuple(value.fold_commitment_hash for value in records)
        lineage: dict[str, list[AlphaDevelopmentChildLineage]] = {
            entry.candidate_id: [] for entry in receipt.child_lineage
        }
        for entry in receipt.child_lineage:
            lineage[entry.candidate_id].append(entry)

        # Grouped by the surface each candidate names, never through
        # ``batch.development_surface_hash``: that field is a synthetic aggregate
        # whenever a batch spans target lanes, naming no artifact at all. This
        # Desk admits one recipe today, which is a property of the caller and not
        # something a reader may assume.
        grouped: dict[str, list[AlphaExperimentCandidateResult]] = {}
        for candidate in batch.candidates:
            if candidate.development_surface_hash is None or candidate.fold_surface_hashes is None:
                raise AuthoringError(
                    "alpha_research.development_receipt_candidate_surface_unavailable"
                )
            grouped.setdefault(candidate.development_surface_hash, []).append(candidate)

        expected: list[tuple[str, str]] = []
        surfaces: dict[str, AlphaDevelopmentSurfaceManifest] = {}
        fold_surfaces: dict[str, AlphaDevelopmentFoldSurface] = {}
        validation_tables: dict[str, pa.Table] = {}
        for surface_hash, candidates in grouped.items():
            try:
                surface = self._store.load_surface(surface_hash)
            except (FileNotFoundError, ValueError, AlphaDevelopmentArtifactReadbackError) as error:
                raise AuthoringError(
                    "alpha_research.development_receipt_surface_unavailable"
                ) from error
            surfaces[surface_hash] = surface

            # Equality, not membership. The manifest is sealed before any fit, so
            # it states the candidate axis the run was supposed to cover; asking
            # only whether a candidate appears in it would admit a batch that
            # dropped one of the others entirely.
            if surface.candidate_ids != tuple(value.candidate_id for value in candidates):
                raise AuthoringError("alpha_research.development_receipt_surface_mismatch")
            if surface.candidate_card_hashes != tuple(value.spec_hash for value in candidates):
                raise AuthoringError("alpha_research.development_receipt_surface_mismatch")
            # The pre-execution fold axis, positionally against the axis the
            # materialization says it transformed.
            if surface.fold_commitment_hashes != commitments:
                raise AuthoringError("alpha_research.development_receipt_surface_mismatch")

            for candidate in candidates:
                fold_surface_hashes = candidate.fold_surface_hashes or ()
                # Stated against the pre-fit fold axis directly rather than left
                # to follow from the lineage comparison below: a fold dropped from
                # the batch *and* the receipt together is only visible against an
                # artifact neither of them can reseal.
                if len(candidate.numerical_result_hashes) != len(records):
                    raise AuthoringError("alpha_research.development_receipt_child_set_incomplete")
                if len(fold_surface_hashes) != len(records):
                    raise AuthoringError("alpha_research.development_receipt_fold_surface_mismatch")
                try:
                    # Read with their validation rows: the surface is proved once
                    # here and the rows it proved travel on the graph.
                    read = tuple(
                        self._store.read_fold_surface(value) for value in fold_surface_hashes
                    )
                except (
                    FileNotFoundError,
                    ValueError,
                    AlphaDevelopmentArtifactReadbackError,
                ) as error:
                    raise AuthoringError(
                        "alpha_research.development_receipt_surface_unavailable"
                    ) from error
                for index, (fold_surface, validation) in enumerate(read):
                    # Enumerated position, so a reordered or repeated fold surface
                    # is refused rather than passing because every hash it names
                    # exists somewhere in the split.
                    if (
                        fold_surface.development_surface_hash != surface_hash
                        or fold_surface.fold_index != index
                        or fold_surface.fold_commitment_hash != commitments[index]
                    ):
                        raise AuthoringError(
                            "alpha_research.development_receipt_fold_surface_mismatch"
                        )
                    fold_surfaces[fold_surface_hashes[index]] = fold_surface
                    validation_tables[fold_surface_hashes[index]] = validation

                expected.extend(
                    (candidate.candidate_id, value) for value in candidate.numerical_result_hashes
                )

        # Exact ordered correspondence, not set equality: a set comparison loses
        # the order the batch produced and collapses repeats, which is precisely
        # what a tampered lineage would rely on.
        observed = tuple(
            (entry.candidate_id, entry.numerical_result_hash) for entry in receipt.child_lineage
        )
        if observed != tuple(expected):
            raise AuthoringError("alpha_research.development_receipt_child_set_incomplete")
        for entries in lineage.values():
            if tuple(value.fold_index for value in entries) != tuple(range(len(records))):
                raise AuthoringError("alpha_research.development_receipt_child_set_incomplete")
        return batch, surfaces, fold_surfaces, validation_tables


def verify_alpha_development_execution_receipt(
    *,
    receipt: AlphaDevelopmentExecutionReceipt,
    evidence: ResearchExecutionEvidence,
) -> None:
    """Compare the receipt's generic identities with the evidence that sealed them.

    The receipt's own hash proves only that nobody edited it. These four
    identities belong to the generic authoring layer, so they are read from it.
    """

    if (
        receipt.program_hash != evidence.program_hash
        or receipt.desk_program_hash != evidence.desk_program_hash
        or receipt.method_binding_hash != evidence.method_binding_hash
        or receipt.desk_input_binding_hash != evidence.desk_input_binding_hash
    ):
        raise AuthoringError("alpha_research.development_receipt_program_mismatch")
    if alpha_development_receipt_uri(receipt.receipt_hash) not in evidence.artifact_uris:
        # The receipt must be reachable from the evidence, or a downstream reader
        # holding the evidence has no way to arrive at it.
        raise AuthoringError("alpha_research.development_receipt_not_published")


__all__ = [
    "AlphaDevelopmentReceiptReader",
    "AlphaDevelopmentVerifiedGraph",
    "alpha_development_receipt_handle",
    "alpha_development_receipt_uri",
    "verify_alpha_development_execution_receipt",
]
