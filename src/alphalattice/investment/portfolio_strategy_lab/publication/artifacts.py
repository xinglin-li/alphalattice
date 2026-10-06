"""Content-addressed Strategy Lab evidence and marker-last current publication."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy.typing as npt
from pydantic import BaseModel

from alphalattice.control.workspace_runtime.content_store import (
    ContentAddressedStore,
    ContentAddressedStoreError,
)
from alphalattice.investment.portfolio_strategy_lab.contracts import (
    CurrentPortfolioResearchMarker,
    CurrentPortfolioResearchPointer,
    PortfolioBaselineSlate,
    PortfolioEvidenceDossier,
    PortfolioExperimentProgram,
    PortfolioPolicyCandidateSet,
    PortfolioResearchProjectionBundle,
    PortfolioResearchReview,
    PortfolioResearchRFC,
    PortfolioScientificStop,
    PortfolioSolverRuntimeEvidence,
    PortfolioTrialEvidence,
    PortfolioTrialLedger,
    seal_contract,
)
from alphalattice.investment.portfolio_strategy_lab.evaluation.evidence import (
    validate_terminal_evidence,
)
from alphalattice.investment.portfolio_strategy_lab.publication.current import (
    CurrentPortfolioStrategyResearchMarker,
    CurrentPortfolioStrategyResearchPointer,
    PortfolioStrategyResearchPublicationBundle,
)
from alphalattice.investment.portfolio_strategy_lab.regularization.attribution_contracts import (
    CurrentPortfolioFold3AttributionMarker,
    PortfolioAttributionAgentReview,
    PortfolioAttributionEvidenceDelta,
    PortfolioFold3AttributionDiagnostic,
    PortfolioFold3AttributionPublicationBundle,
)
from alphalattice.investment.portfolio_strategy_lab.regularization.contracts import (
    PortfolioAgentValueReport,
    PortfolioCrossFoldEvidenceDossier,
    PortfolioResearchAlphaRecipe,
    PortfolioResearchBoard,
    PortfolioStrategyResearchMandate,
)
from alphalattice.investment.portfolio_strategy_lab.regularization.recovery_agent_contracts import (
    CurrentPortfolioStrategyRecoveryMarker,
    HistoricalPortfolioRecoveryAgentReview,
    HistoricalPortfolioRecoveryResearchBoard,
    PortfolioConditionalPolicyCandidate,
    PortfolioConditionalPolicyCandidateSet,
    PortfolioRecoveryAgentReview,
    PortfolioRecoveryDecisionDossier,
    PortfolioRecoveryResearchBoard,
    PortfolioStrategyRecoveryPublicationBundle,
)
from alphalattice.investment.portfolio_strategy_lab.regularization.recovery_contracts import (
    PortfolioEvidenceClassificationAudit,
    PortfolioSharedConfigEvidenceDossier,
    PortfolioSharedConfigTrialEvidence,
    PortfolioStrategyRecoveryMandate,
)
from alphalattice.investment.portfolio_strategy_lab.research_loop.contracts import (
    PortfolioResearchScientistReview,
)
from alphalattice.investment.risk_research.calibration.diagnostic_contracts import (
    CurrentPortfolioRiskCalibrationMarker,
    PortfolioRiskCalibrationDiagnostic,
    PortfolioRiskCalibrationPublicationBundle,
)
from alphalattice.investment.risk_research.contracts import HistoricalCovarianceSurface
from alphalattice.investment.risk_research.surfaces.artifacts import RiskArtifactStore
from alphalattice.investment.risk_research.surfaces.returns import RiskReturnArtifactStore


class PortfolioPublicationError(ValueError):
    """Stable fail-closed evidence and current-pointer boundary."""


def _load_refusal(error: ContentAddressedStoreError) -> str:
    code = str(error).partition(":")[0]
    if code == "content_store.artifact_missing":
        # A sealed study's child is lost work, with the existing backup recovery route.
        return "portfolio_strategy_lab.artifact_absent"
    if code == "content_store.identity_invalid":
        return str(error)
    return "portfolio_strategy_lab.artifact_tampered"


@dataclass(frozen=True, slots=True)
class CurrentPortfolioResearchPublication:
    """Retain reopened trial-ledger research publication artifacts and their committed pointer."""

    action: str
    marker: CurrentPortfolioResearchMarker
    bundle: PortfolioResearchProjectionBundle
    ledger: PortfolioTrialLedger
    terminal: PortfolioPolicyCandidateSet | PortfolioScientificStop
    pointer: CurrentPortfolioResearchPointer


@dataclass(frozen=True, slots=True)
class CurrentPortfolioStrategyResearchPublication:
    """Retain reopened strategy research publication artifacts and their committed pointer."""

    action: str
    marker: CurrentPortfolioStrategyResearchMarker
    bundle: PortfolioStrategyResearchPublicationBundle
    pointer: CurrentPortfolioStrategyResearchPointer


@dataclass(frozen=True, slots=True)
class CurrentPortfolioStrategyRecoveryPublication:
    action: str
    marker: CurrentPortfolioStrategyRecoveryMarker
    bundle: PortfolioStrategyRecoveryPublicationBundle
    pointer: CurrentPortfolioStrategyResearchPointer


@dataclass(frozen=True, slots=True)
class CurrentPortfolioFold3AttributionPublication:
    """Retain reopened fold attribution publication artifacts and their committed pointer."""

    action: str
    marker: CurrentPortfolioFold3AttributionMarker
    bundle: PortfolioFold3AttributionPublicationBundle
    pointer: CurrentPortfolioStrategyResearchPointer


@dataclass(frozen=True, slots=True)
class CurrentPortfolioRiskCalibrationPublication:
    """Retain reopened Risk calibration publication artifacts and their committed pointer."""

    action: str
    marker: CurrentPortfolioRiskCalibrationMarker
    bundle: PortfolioRiskCalibrationPublicationBundle
    pointer: CurrentPortfolioStrategyResearchPointer


type CurrentPortfolioStrategyPublication = (
    CurrentPortfolioStrategyResearchPublication
    | CurrentPortfolioStrategyRecoveryPublication
    | CurrentPortfolioFold3AttributionPublication
    | CurrentPortfolioRiskCalibrationPublication
)


class PortfolioResearchArtifactStore:
    """Content-addressed Strategy Lab evidence: JSON documents and packed lanes.

    The rule used to be "never trial weight matrices", and it was the right rule
    for the wrong reason. What must never happen is a weight matrix spelled out
    inside a JSON document -- tens of megabytes of decimal digits that no longer
    round-trip to the values the arithmetic ran on, inside a contract whose
    identity then covers all of them. What a Campaign replay *needs* is the exact
    bytes, and those go in their own file named by their own digest, referenced
    from a document that stays compact.
    """

    def __init__(self, artifact_root: Path) -> None:
        """Bind research artifacts and current-pointer locations to one caller-owned root.

        Args:
            artifact_root: Caller-owned workspace artifact directory.
        """
        self.root = artifact_root.resolve() / "portfolio-strategy-lab"
        self._content = ContentAddressedStore(
            self.root,
            uri_prefix="playpen://portfolio-strategy-lab",
        )
        self.pointer_path = self.root / "current" / "active.json"
        self.regularization_pointer_path = self.root / "current" / "strategy-research-active.json"

    @staticmethod
    def _require_hash(value: str) -> None:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise PortfolioPublicationError("portfolio_strategy_lab.artifact_identity_invalid")

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        staged.write_bytes(content)
        os.replace(staged, path)
        staged.unlink(missing_ok=True)

    def publish(
        self,
        *,
        category: str,
        value: BaseModel,
        identity_field: str,
    ) -> None:
        """Publish a validated model under its explicit category and declared identity.

        Args:
            category: Registered artifact category.
            value: Sealed concrete model.
            identity_field: Name of the concrete self identity.

        Raises:
            PortfolioPublicationError: Content-addressed storage refuses identity reuse.
        """
        try:
            self._content.publish_model(
                category=category,
                value=value,
                identity_field=identity_field,
            )
        except ContentAddressedStoreError as error:
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.artifact_identity_reused"
            ) from error

    def load[ContractT: BaseModel](
        self,
        *,
        category: str,
        content_hash: str,
        model: type[ContractT],
        identity_field: str,
    ) -> ContractT:
        """Reopen and validate one concrete content-addressed research model.

        Args:
            category: Registered artifact category.
            content_hash: Exact artifact identity.
            model: Concrete contract class.
            identity_field: Concrete self-identity field.

        Returns:
            Validated exact concrete artifact.

        Raises:
            PortfolioPublicationError: Missing sealed work and corrupt readback are distinct.
        """
        try:
            return self._content.load_model(
                category=category,
                content_hash=content_hash,
                model=model,
                identity_field=identity_field,
            )
        except ContentAddressedStoreError as error:
            raise PortfolioPublicationError(_load_refusal(error)) from error

    def publish_array(self, *, category: str, values: npt.NDArray[Any]) -> str:
        """Write one numerical lane as Parquet, named by the digest of its packed bytes.

        No contract argument and no identity field: the values *are* the identity,
        so there is nothing here a caller could name the file after other than
        what it contains. Returns the digest.
        """
        return self.publish_columns(category=category, columns={"value": values})

    def publish_columns(self, *, category: str, columns: Mapping[str, npt.NDArray[Any]]) -> str:
        """Write numerical columns of one shape as one table, named by their digest."""
        try:
            return self._content.publish_columns(category=category, columns=columns)
        except ContentAddressedStoreError as error:
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.artifact_identity_reused"
            ) from error

    def publish_document(self, *, category: str, payload: bytes) -> str:
        """Write one canonical JSON document, named by the digest of its bytes."""
        try:
            return self._content.publish_document(
                category=category, payload=payload, extension="json"
            )
        except ContentAddressedStoreError as error:
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.artifact_identity_reused"
            ) from error

    def holds_array(self, *, category: str, content_hash: str) -> bool:
        """Whether this store holds the lane a digest names, as Parquet."""
        return self._content.holds_columns(category=category, content_hash=content_hash)

    def load_columns(self, *, category: str, content_hash: str) -> dict[str, npt.NDArray[Any]]:
        """Read one table's flat columns and prove they are the ones that were asked for."""
        try:
            return self._content.load_columns(category=category, content_hash=content_hash)
        except ContentAddressedStoreError as error:
            raise PortfolioPublicationError(_load_refusal(error)) from error

    def load_packed_bytes(self, *, category: str, content_hash: str) -> bytes:
        """Read one lane's packed bytes and prove they are the ones that were asked for."""
        try:
            return self._content.load_packed_bytes(category=category, content_hash=content_hash)
        except ContentAddressedStoreError as error:
            raise PortfolioPublicationError(_load_refusal(error)) from error

    def load_document(self, *, category: str, content_hash: str) -> bytes:
        """Read one JSON document and prove it is the one that was asked for."""
        try:
            return self._content.load_document(
                category=category, content_hash=content_hash, extension="json"
            )
        except ContentAddressedStoreError as error:
            raise PortfolioPublicationError(_load_refusal(error)) from error

    def publish_trial_evidence(self, value: PortfolioTrialEvidence) -> None:
        """Publish exact sealed trial evidence in its registered category.

        Args:
            value: Validated trial evidence.
        """
        self.publish(category="trial-evidence", value=value, identity_field="evidence_hash")

    def load_trial_evidence(self, evidence_hash: str) -> PortfolioTrialEvidence:
        """Reopen exact sealed trial evidence.

        Args:
            evidence_hash: Exact trial evidence identity.

        Returns:
            Validated trial evidence.
        """
        return self.load(
            category="trial-evidence",
            content_hash=evidence_hash,
            model=PortfolioTrialEvidence,
            identity_field="evidence_hash",
        )

    def find_trial_evidence(
        self,
        *,
        program_hash: str,
        stratum_id: str,
        trial_number: int,
    ) -> PortfolioTrialEvidence | None:
        """Find baseline evidence for one exact program, stratum and trial.

        Args:
            program_hash: Exact experiment program.
            stratum_id: Declared experiment stratum.
            trial_number: Exact trial position.

        Returns:
            Unique matching trial with empty search parameters, or None.

        Raises:
            PortfolioPublicationError: Any examined trial is invalid or matching baseline evidence
                is ambiguous.
        """
        matches: list[PortfolioTrialEvidence] = []
        for path in sorted((self.root / "trial-evidence").glob("*.json")):
            try:
                value = PortfolioTrialEvidence.model_validate_json(path.read_bytes())
            except Exception as error:
                raise PortfolioPublicationError(
                    "portfolio_strategy_lab.artifact_tampered"
                ) from error
            if (
                value.program_hash == program_hash
                and value.stratum_id == stratum_id
                and value.trial_number == trial_number
                and not value.search_parameters
            ):
                matches.append(value)
        if len(matches) > 1:
            raise PortfolioPublicationError("portfolio_strategy_lab.baseline_evidence_ambiguous")
        return matches[0] if matches else None

    def publish_baseline_slate(self, value: PortfolioBaselineSlate) -> None:
        """Publish exact baseline slate and atomically record its by-program index.

        Args:
            value: Validated sealed baseline slate.
        """
        self.publish(category="baseline-slates", value=value, identity_field="slate_hash")
        self._atomic_write(
            self.root / "baseline-slates" / "by-program" / f"{value.program_hash}.json",
            json.dumps(
                {"program_hash": value.program_hash, "slate_hash": value.slate_hash},
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        )

    def load_baseline_slate(self, program_hash: str) -> PortfolioBaselineSlate | None:
        """Reopen the exact baseline slate selected by a program index.

        Args:
            program_hash: Exact experiment program identity.

        Returns:
            Validated indexed slate, or None when no index exists.

        Raises:
            PortfolioPublicationError: Index binding/shape or referenced artifact is invalid.
        """
        self._require_hash(program_hash)
        index = self.root / "baseline-slates" / "by-program" / f"{program_hash}.json"
        if not index.is_file():
            return None
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
            if payload.get("program_hash") != program_hash:
                raise PortfolioPublicationError("portfolio_strategy_lab.baseline_index_invalid")
            return self.load(
                category="baseline-slates",
                content_hash=str(payload["slate_hash"]),
                model=PortfolioBaselineSlate,
                identity_field="slate_hash",
            )
        except PortfolioPublicationError:
            raise
        except Exception as error:
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.baseline_index_invalid"
            ) from error

    def load_solver_runtime(self, program_hash: str) -> PortfolioSolverRuntimeEvidence | None:
        """Resolve a prior admitted probe without rerunning its timed computation."""
        self._require_hash(program_hash)
        matches: list[PortfolioSolverRuntimeEvidence] = []
        for path in sorted((self.root / "solver-runtime").glob("*.json")):
            try:
                value = PortfolioSolverRuntimeEvidence.model_validate_json(path.read_bytes())
            except Exception as error:
                raise PortfolioPublicationError(
                    "portfolio_strategy_lab.solver_runtime_tampered"
                ) from error
            if value.program_hash == program_hash and value.admitted:
                matches.append(value)
        return min(matches, key=lambda value: value.evidence_hash) if matches else None

    def publish_ledger(self, value: PortfolioTrialLedger) -> None:
        """Publish the sealed trial ledger in its registered category.

        Args:
            value: Validated trial ledger.
        """
        self.publish(category="trial-ledgers", value=value, identity_field="ledger_hash")

    def write_ledger_checkpoint(self, *, program_hash: str, phase: str, ledger_hash: str) -> None:
        """Atomically record exact program/phase trial-ledger checkpoint identities.

        Args:
            program_hash: Exact experiment program identity.
            phase: Explicit checkpoint phase.
            ledger_hash: Exact sealed trial-ledger identity.
        """
        self._require_hash(program_hash)
        self._require_hash(ledger_hash)
        self._atomic_write(
            self.root / "checkpoints" / f"{program_hash}-{phase}.json",
            json.dumps(
                {"program_hash": program_hash, "phase": phase, "ledger_hash": ledger_hash},
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        )

    def load_ledger_checkpoint(
        self, *, program_hash: str, phase: str
    ) -> PortfolioTrialLedger | None:
        """Reopen a program/phase checkpoint and its exact sealed ledger.

        Args:
            program_hash: Exact experiment program identity.
            phase: Explicit checkpoint phase.

        Returns:
            Validated indexed ledger, or None without a checkpoint.

        Raises:
            PortfolioPublicationError: Checkpoint shape/binding or referenced ledger differs.
        """
        self._require_hash(program_hash)
        target = self.root / "checkpoints" / f"{program_hash}-{phase}.json"
        if not target.is_file():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            if payload != {
                "program_hash": program_hash,
                "phase": phase,
                "ledger_hash": payload.get("ledger_hash"),
            }:
                raise PortfolioPublicationError("portfolio_strategy_lab.ledger_checkpoint_invalid")
            return self.load(
                category="trial-ledgers",
                content_hash=str(payload["ledger_hash"]),
                model=PortfolioTrialLedger,
                identity_field="ledger_hash",
            )
        except PortfolioPublicationError:
            raise
        except Exception as error:
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.ledger_checkpoint_invalid"
            ) from error

    def write_review_checkpoint(
        self,
        *,
        program_hash: str,
        ledger_hash: str,
        dossier_hash: str,
        rfc_hash: str,
        review_hash: str,
    ) -> None:
        """Atomically record exact program, ledger, dossier, RFC and review identities.

        Args:
            program_hash: Exact experiment program.
            ledger_hash: Exact trial ledger.
            dossier_hash: Exact evidence dossier.
            rfc_hash: Exact research RFC.
            review_hash: Exact research review.
        """
        for value in (program_hash, ledger_hash, dossier_hash, rfc_hash, review_hash):
            self._require_hash(value)
        self._atomic_write(
            self.root / "checkpoints" / f"{program_hash}-review.json",
            json.dumps(
                {
                    "program_hash": program_hash,
                    "ledger_hash": ledger_hash,
                    "dossier_hash": dossier_hash,
                    "rfc_hash": rfc_hash,
                    "review_hash": review_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        )

    def load_review_checkpoint(
        self, program_hash: str
    ) -> (
        tuple[
            PortfolioTrialLedger,
            PortfolioEvidenceDossier,
            PortfolioResearchRFC,
            PortfolioResearchReview,
        ]
        | None
    ):
        """Reopen checkpoint artifacts and reconcile their program and review lineage.

        Args:
            program_hash: Exact experiment program identity.

        Returns:
            Ledger, dossier, RFC and review tuple, or None without a checkpoint.

        Raises:
            PortfolioPublicationError: Checkpoint/artifact validity or program/RFC/dossier linkage
                differs.
        """
        self._require_hash(program_hash)
        target = self.root / "checkpoints" / f"{program_hash}-review.json"
        if not target.is_file():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            if payload.get("program_hash") != program_hash:
                raise PortfolioPublicationError("portfolio_strategy_lab.review_checkpoint_invalid")
            ledger = self.load(
                category="trial-ledgers",
                content_hash=str(payload["ledger_hash"]),
                model=PortfolioTrialLedger,
                identity_field="ledger_hash",
            )
            dossier = self.load(
                category="dossiers",
                content_hash=str(payload["dossier_hash"]),
                model=PortfolioEvidenceDossier,
                identity_field="dossier_hash",
            )
            rfc = self.load(
                category="research-rfcs",
                content_hash=str(payload["rfc_hash"]),
                model=PortfolioResearchRFC,
                identity_field="rfc_hash",
            )
            review = self.load(
                category="reviews",
                content_hash=str(payload["review_hash"]),
                model=PortfolioResearchReview,
                identity_field="review_hash",
            )
        except PortfolioPublicationError:
            raise
        except Exception as error:
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.review_checkpoint_invalid"
            ) from error
        if (
            ledger.program_hash != program_hash
            or dossier.program_hash != program_hash
            or review.rfc_hash != rfc.rfc_hash
            or review.dossier_hash != dossier.dossier_hash
        ):
            raise PortfolioPublicationError("portfolio_strategy_lab.review_checkpoint_invalid")
        return ledger, dossier, rfc, review

    def read_current(self) -> CurrentPortfolioResearchPublication | None:
        """Reopen committed current research and validate all terminal evidence lineage.

        Returns:
            Current marker, bundle, ledger, terminal and pointer, or None without a pointer.

        Raises:
            PortfolioPublicationError: Pointer/artifact validity, common lineage or terminal
                evidence verification fails.
        """
        if not self.pointer_path.is_file():
            return None
        try:
            pointer = CurrentPortfolioResearchPointer.model_validate_json(
                self.pointer_path.read_bytes()
            )
            marker = self.load(
                category="current/markers",
                content_hash=pointer.marker_hash,
                model=CurrentPortfolioResearchMarker,
                identity_field="marker_hash",
            )
            bundle = self.load(
                category="projection-bundles",
                content_hash=marker.bundle_hash,
                model=PortfolioResearchProjectionBundle,
                identity_field="bundle_hash",
            )
            ledger = self.load(
                category="trial-ledgers",
                content_hash=marker.ledger_hash,
                model=PortfolioTrialLedger,
                identity_field="ledger_hash",
            )
            if bundle.terminal_kind == "CANDIDATE_SET":
                terminal: PortfolioPolicyCandidateSet | PortfolioScientificStop = self.load(
                    category="candidate-sets",
                    content_hash=bundle.terminal_artifact_hash,
                    model=PortfolioPolicyCandidateSet,
                    identity_field="candidate_set_hash",
                )
            else:
                terminal = self.load(
                    category="scientific-stops",
                    content_hash=bundle.terminal_artifact_hash,
                    model=PortfolioScientificStop,
                    identity_field="stop_hash",
                )
            dossier = self.load(
                category="dossiers",
                content_hash=bundle.dossier_hash,
                model=PortfolioEvidenceDossier,
                identity_field="dossier_hash",
            )
            review = self.load(
                category="reviews",
                content_hash=bundle.review_hash,
                model=PortfolioResearchReview,
                identity_field="review_hash",
            )
            rfc = self.load(
                category="research-rfcs",
                content_hash=terminal.rfc_hash,
                model=PortfolioResearchRFC,
                identity_field="rfc_hash",
            )
            program = self.load(
                category="programs",
                content_hash=bundle.program_hash,
                model=PortfolioExperimentProgram,
                identity_field="program_hash",
            )
        except PortfolioPublicationError:
            raise
        except Exception as error:
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.current_pointer_tampered"
            ) from error
        if (
            marker.marker_hash != pointer.marker_hash
            or marker.program_hash != bundle.program_hash
            or marker.program_hash != ledger.program_hash
            or marker.ledger_hash != bundle.ledger_hash
            or marker.ledger_hash != ledger.ledger_hash
            or marker.bundle_hash != bundle.bundle_hash
            or marker.status != bundle.status
            or bundle.attempted_count != len(ledger.entries)
            or terminal.program_hash != bundle.program_hash
            or terminal.dossier_hash != bundle.dossier_hash
            or terminal.review_hash != bundle.review_hash
            or dossier.program_hash != bundle.program_hash
            or review.dossier_hash != dossier.dossier_hash
            or review.rfc_hash != rfc.rfc_hash
            or terminal.rfc_hash != rfc.rfc_hash
        ):
            raise PortfolioPublicationError("portfolio_strategy_lab.current_lineage_invalid")
        try:
            validate_terminal_evidence(
                program=program,
                ledger=ledger,
                dossier=dossier,
                review=review,
            )
        except ValueError as error:
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.current_lineage_invalid"
            ) from error
        return CurrentPortfolioResearchPublication(
            action="READ_CURRENT",
            marker=marker,
            bundle=bundle,
            ledger=ledger,
            terminal=terminal,
            pointer=pointer,
        )

    def publish_regularization_current(
        self,
        *,
        bundle: PortfolioStrategyResearchPublicationBundle,
        published_at: datetime,
    ) -> CurrentPortfolioStrategyResearchPublication:
        """Promote one fully bound Strategy research result with marker-last semantics."""
        self._validate_regularization_bundle(bundle)
        current = self.read_regularization_current()
        if isinstance(current, CurrentPortfolioStrategyRecoveryPublication):
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.regularization_legacy_route_retired"
            )
        if current is not None and current.bundle == bundle:
            return CurrentPortfolioStrategyResearchPublication(
                action="REUSED_EXACT",
                marker=current.marker,
                bundle=current.bundle,
                pointer=current.pointer,
            )
        self.publish(
            category="regularization/projection-bundles",
            value=bundle,
            identity_field="bundle_hash",
        )
        marker = seal_contract(
            CurrentPortfolioStrategyResearchMarker,
            "marker_hash",
            mandate_hash=bundle.mandate_hash,
            dossier_hash=bundle.dossier_hash,
            bundle_hash=bundle.bundle_hash,
            status=bundle.status,
            published_at=published_at,
        )
        self.publish(
            category="regularization/current-markers",
            value=marker,
            identity_field="marker_hash",
        )
        pointer = seal_contract(
            CurrentPortfolioStrategyResearchPointer,
            "pointer_hash",
            marker_hash=marker.marker_hash,
        )
        self._atomic_write(
            self.regularization_pointer_path,
            json.dumps(
                pointer.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode(),
        )
        durable = self.read_regularization_current()
        if durable is None or durable.pointer != pointer:
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.regularization_current_readback_failed"
            )
        return CurrentPortfolioStrategyResearchPublication(
            action="PUBLISHED" if current is None else "REPLACED_CURRENT",
            marker=marker,
            bundle=bundle,
            pointer=pointer,
        )

    def read_regularization_current(
        self,
    ) -> CurrentPortfolioStrategyPublication | None:
        """Read only a current-schema Strategy publication from the active pointer."""
        if not self.regularization_pointer_path.is_file():
            return None
        try:
            pointer = CurrentPortfolioStrategyResearchPointer.model_validate_json(
                self.regularization_pointer_path.read_bytes()
            )
            risk_calibration_marker_path = (
                self.root
                / "regularization"
                / "risk-calibration-current-markers"
                / f"{pointer.marker_hash}.json"
            )
            if risk_calibration_marker_path.is_file():
                return self._read_risk_calibration_current(pointer)
            attribution_marker_path = (
                self.root
                / "regularization"
                / "attribution-current-markers"
                / f"{pointer.marker_hash}.json"
            )
            if attribution_marker_path.is_file():
                return self._read_attribution_current(pointer)
            recovery_marker_path = (
                self.root
                / "regularization"
                / "recovery-current-markers"
                / f"{pointer.marker_hash}.json"
            )
            if recovery_marker_path.is_file():
                return self._read_recovery_current(pointer)
            marker = self.load(
                category="regularization/current-markers",
                content_hash=pointer.marker_hash,
                model=CurrentPortfolioStrategyResearchMarker,
                identity_field="marker_hash",
            )
            bundle = self.load(
                category="regularization/projection-bundles",
                content_hash=marker.bundle_hash,
                model=PortfolioStrategyResearchPublicationBundle,
                identity_field="bundle_hash",
            )
        except PortfolioPublicationError:
            raise
        except Exception as error:
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.regularization_current_pointer_tampered"
            ) from error
        if (
            pointer.marker_hash != marker.marker_hash
            or marker.mandate_hash != bundle.mandate_hash
            or marker.dossier_hash != bundle.dossier_hash
            or marker.bundle_hash != bundle.bundle_hash
            or marker.status != bundle.status
        ):
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.regularization_current_lineage_invalid"
            )
        self._validate_regularization_bundle(bundle)
        return CurrentPortfolioStrategyResearchPublication(
            action="READ_CURRENT",
            marker=marker,
            bundle=bundle,
            pointer=pointer,
        )

    def publish_recovery_current(
        self,
        *,
        bundle: PortfolioStrategyRecoveryPublicationBundle,
        published_at: datetime,
    ) -> CurrentPortfolioStrategyRecoveryPublication:
        """Promote one recovery result through the existing Strategy current pointer."""
        self._validate_recovery_bundle(bundle, allow_historical_review=False)
        current = self.read_regularization_current()
        if isinstance(current, CurrentPortfolioStrategyRecoveryPublication) and (
            current.bundle == bundle
        ):
            return CurrentPortfolioStrategyRecoveryPublication(
                action="REUSED_EXACT",
                marker=current.marker,
                bundle=current.bundle,
                pointer=current.pointer,
            )
        self.publish(
            category="regularization/recovery-projection-bundles",
            value=bundle,
            identity_field="bundle_hash",
        )
        marker = seal_contract(
            CurrentPortfolioStrategyRecoveryMarker,
            "marker_hash",
            recovery_mandate_hash=bundle.recovery_mandate_hash,
            decision_dossier_hash=bundle.decision_dossier_hash,
            bundle_hash=bundle.bundle_hash,
            status=bundle.status,
            published_at=published_at,
        )
        self.publish(
            category="regularization/recovery-current-markers",
            value=marker,
            identity_field="marker_hash",
        )
        pointer = seal_contract(
            CurrentPortfolioStrategyResearchPointer,
            "pointer_hash",
            marker_hash=marker.marker_hash,
        )
        self._atomic_write(
            self.regularization_pointer_path,
            json.dumps(
                pointer.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode(),
        )
        durable = self.read_regularization_current()
        if not isinstance(durable, CurrentPortfolioStrategyRecoveryPublication) or (
            durable.pointer != pointer
        ):
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.recovery_current_readback_failed"
            )
        return CurrentPortfolioStrategyRecoveryPublication(
            action="PUBLISHED" if current is None else "REPLACED_CURRENT",
            marker=marker,
            bundle=bundle,
            pointer=pointer,
        )

    def publish_attribution_current(
        self,
        *,
        bundle: PortfolioFold3AttributionPublicationBundle,
        published_at: datetime,
    ) -> CurrentPortfolioFold3AttributionPublication:
        """Promote one Fold-3 evidence update through the single Strategy pointer."""
        self._validate_attribution_bundle(bundle)
        current = self.read_regularization_current()
        if isinstance(current, CurrentPortfolioFold3AttributionPublication) and (
            current.bundle == bundle
        ):
            return CurrentPortfolioFold3AttributionPublication(
                action="REUSED_EXACT",
                marker=current.marker,
                bundle=current.bundle,
                pointer=current.pointer,
            )
        self.publish(
            category="regularization/attribution-projection-bundles",
            value=bundle,
            identity_field="bundle_hash",
        )
        marker = seal_contract(
            CurrentPortfolioFold3AttributionMarker,
            "marker_hash",
            diagnostic_hash=bundle.diagnostic_hash,
            evidence_delta_hash=bundle.evidence_delta_hash,
            bundle_hash=bundle.bundle_hash,
            status=bundle.status,
            published_at=published_at,
        )
        self.publish(
            category="regularization/attribution-current-markers",
            value=marker,
            identity_field="marker_hash",
        )
        pointer = seal_contract(
            CurrentPortfolioStrategyResearchPointer,
            "pointer_hash",
            marker_hash=marker.marker_hash,
        )
        self._atomic_write(
            self.regularization_pointer_path,
            json.dumps(
                pointer.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode(),
        )
        durable = self.read_regularization_current()
        if not isinstance(durable, CurrentPortfolioFold3AttributionPublication) or (
            durable.pointer != pointer
        ):
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.attribution_current_readback_failed"
            )
        return CurrentPortfolioFold3AttributionPublication(
            action="PUBLISHED" if current is None else "REPLACED_CURRENT",
            marker=marker,
            bundle=bundle,
            pointer=pointer,
        )

    def _read_attribution_current(
        self,
        pointer: CurrentPortfolioStrategyResearchPointer,
    ) -> CurrentPortfolioFold3AttributionPublication:
        marker = self.load(
            category="regularization/attribution-current-markers",
            content_hash=pointer.marker_hash,
            model=CurrentPortfolioFold3AttributionMarker,
            identity_field="marker_hash",
        )
        bundle = self.load(
            category="regularization/attribution-projection-bundles",
            content_hash=marker.bundle_hash,
            model=PortfolioFold3AttributionPublicationBundle,
            identity_field="bundle_hash",
        )
        if (
            marker.diagnostic_hash != bundle.diagnostic_hash
            or marker.evidence_delta_hash != bundle.evidence_delta_hash
            or marker.bundle_hash != bundle.bundle_hash
            or marker.status != bundle.status
        ):
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.attribution_current_lineage_invalid"
            )
        self._validate_attribution_bundle(bundle)
        return CurrentPortfolioFold3AttributionPublication(
            action="READ_CURRENT",
            marker=marker,
            bundle=bundle,
            pointer=pointer,
        )

    def _validate_attribution_bundle(
        self,
        bundle: PortfolioFold3AttributionPublicationBundle,
    ) -> None:
        try:
            recovery = self.load(
                category="regularization/recovery-projection-bundles",
                content_hash=bundle.recovery_bundle_hash,
                model=PortfolioStrategyRecoveryPublicationBundle,
                identity_field="bundle_hash",
            )
            self._validate_recovery_bundle(recovery, allow_historical_review=False)
            candidate_set = self.load(
                category="regularization/conditional-candidate-sets",
                content_hash=bundle.candidate_set_hash,
                model=PortfolioConditionalPolicyCandidateSet,
                identity_field="candidate_set_hash",
            )
            diagnostic = self.load(
                category="regularization/attribution-diagnostics",
                content_hash=bundle.diagnostic_hash,
                model=PortfolioFold3AttributionDiagnostic,
                identity_field="diagnostic_hash",
            )
            delta = self.load(
                category="regularization/attribution-evidence-deltas",
                content_hash=bundle.evidence_delta_hash,
                model=PortfolioAttributionEvidenceDelta,
                identity_field="delta_hash",
            )
            review = self.load(
                category="regularization/attribution-agent-reviews",
                content_hash=bundle.agent_review_hash,
                model=PortfolioAttributionAgentReview,
                identity_field="review_hash",
            )
        except PortfolioPublicationError:
            raise
        except Exception as error:
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.attribution_publication_tampered"
            ) from error
        expected_status = (
            "PORTFOLIO_FOLD3_ATTRIBUTION_EVIDENCE_UPDATED"
            if diagnostic.evidence_status == "COMPLETE_FROM_FROZEN_EVIDENCE"
            else "PORTFOLIO_FOLD3_ATTRIBUTION_INSUFFICIENT"
        )
        decision = review.decision
        if (
            recovery.candidate_set_hash != candidate_set.candidate_set_hash
            or bundle.candidate_set_hash != candidate_set.candidate_set_hash
            or len(candidate_set.candidates) != bundle.conditional_candidate_count
            or diagnostic.recovery_bundle_hash != recovery.bundle_hash
            or diagnostic.candidate_set_hash != candidate_set.candidate_set_hash
            or delta.prior_recovery_review_hash != recovery.agent_review_hash
            or delta.diagnostic_hash != diagnostic.diagnostic_hash
            or review.evidence_delta_hash != delta.delta_hash
            or bundle.status != expected_status
            or bundle.decision_owner != review.decision_owner
            or bundle.primary_hypothesis != decision.primary_hypothesis
            or bundle.decision_route != decision.route
            or bundle.attribution_conclusion != decision.attribution_conclusion
            or bundle.recommended_next_mandate != decision.recommended_next_mandate
            or bundle.model_call_count != review.model_call_count
            or bundle.failure_count != review.failure_count
            or bundle.typed_repair_count != review.typed_repair_count
        ):
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.attribution_current_lineage_invalid"
            )

    def publish_risk_calibration_current(
        self,
        *,
        bundle: PortfolioRiskCalibrationPublicationBundle,
        published_at: datetime,
    ) -> CurrentPortfolioRiskCalibrationPublication:
        """Promote one bounded calibration result through the single Strategy pointer."""
        self._validate_risk_calibration_bundle(bundle)
        current = self.read_regularization_current()
        if isinstance(current, CurrentPortfolioRiskCalibrationPublication) and (
            current.bundle == bundle
        ):
            return CurrentPortfolioRiskCalibrationPublication(
                action="REUSED_EXACT",
                marker=current.marker,
                bundle=current.bundle,
                pointer=current.pointer,
            )
        self.publish(
            category="regularization/risk-calibration-projection-bundles",
            value=bundle,
            identity_field="bundle_hash",
        )
        marker = seal_contract(
            CurrentPortfolioRiskCalibrationMarker,
            "marker_hash",
            prior_attribution_marker_hash=bundle.prior_attribution_marker_hash,
            diagnostic_hash=bundle.diagnostic_hash,
            bundle_hash=bundle.bundle_hash,
            published_at=published_at,
        )
        self.publish(
            category="regularization/risk-calibration-current-markers",
            value=marker,
            identity_field="marker_hash",
        )
        pointer = seal_contract(
            CurrentPortfolioStrategyResearchPointer,
            "pointer_hash",
            marker_hash=marker.marker_hash,
        )
        self._atomic_write(
            self.regularization_pointer_path,
            json.dumps(
                pointer.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode(),
        )
        durable = self.read_regularization_current()
        if not isinstance(durable, CurrentPortfolioRiskCalibrationPublication) or (
            durable.pointer != pointer
        ):
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.risk_calibration_current_readback_failed"
            )
        return CurrentPortfolioRiskCalibrationPublication(
            action="PUBLISHED" if current is None else "REPLACED_CURRENT",
            marker=marker,
            bundle=bundle,
            pointer=pointer,
        )

    def _read_risk_calibration_current(
        self,
        pointer: CurrentPortfolioStrategyResearchPointer,
    ) -> CurrentPortfolioRiskCalibrationPublication:
        marker = self.load(
            category="regularization/risk-calibration-current-markers",
            content_hash=pointer.marker_hash,
            model=CurrentPortfolioRiskCalibrationMarker,
            identity_field="marker_hash",
        )
        bundle = self.load(
            category="regularization/risk-calibration-projection-bundles",
            content_hash=marker.bundle_hash,
            model=PortfolioRiskCalibrationPublicationBundle,
            identity_field="bundle_hash",
        )
        if (
            marker.prior_attribution_marker_hash != bundle.prior_attribution_marker_hash
            or marker.diagnostic_hash != bundle.diagnostic_hash
            or marker.bundle_hash != bundle.bundle_hash
            or marker.status != bundle.status
        ):
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.risk_calibration_current_lineage_invalid"
            )
        self._validate_risk_calibration_bundle(bundle)
        return CurrentPortfolioRiskCalibrationPublication(
            action="READ_CURRENT",
            marker=marker,
            bundle=bundle,
            pointer=pointer,
        )

    def _validate_risk_calibration_bundle(
        self,
        bundle: PortfolioRiskCalibrationPublicationBundle,
    ) -> tuple[
        PortfolioConditionalPolicyCandidateSet,
        PortfolioConditionalPolicyCandidate,
        PortfolioSharedConfigTrialEvidence,
        PortfolioStrategyRecoveryPublicationBundle,
        PortfolioStrategyRecoveryMandate,
        PortfolioStrategyResearchMandate,
        PortfolioResearchAlphaRecipe,
    ]:
        try:
            prior_marker = self.load(
                category="regularization/attribution-current-markers",
                content_hash=bundle.prior_attribution_marker_hash,
                model=CurrentPortfolioFold3AttributionMarker,
                identity_field="marker_hash",
            )
            prior_bundle = self.load(
                category="regularization/attribution-projection-bundles",
                content_hash=bundle.prior_attribution_bundle_hash,
                model=PortfolioFold3AttributionPublicationBundle,
                identity_field="bundle_hash",
            )
            self._validate_attribution_bundle(prior_bundle)
            diagnostic = self.load(
                category="regularization/risk-calibration-diagnostics",
                content_hash=bundle.diagnostic_hash,
                model=PortfolioRiskCalibrationDiagnostic,
                identity_field="diagnostic_hash",
            )
            candidate_set = self.load(
                category="regularization/conditional-candidate-sets",
                content_hash=prior_bundle.candidate_set_hash,
                model=PortfolioConditionalPolicyCandidateSet,
                identity_field="candidate_set_hash",
            )
            selected_trial = self.load(
                category="regularization/shared-config-trials",
                content_hash=candidate_set.candidates[0].selected_trial_evidence_hash,
                model=PortfolioSharedConfigTrialEvidence,
                identity_field="evidence_hash",
            )
            recovery = self.load(
                category="regularization/recovery-projection-bundles",
                content_hash=prior_bundle.recovery_bundle_hash,
                model=PortfolioStrategyRecoveryPublicationBundle,
                identity_field="bundle_hash",
            )
            recovery_mandate = self.load(
                category="regularization/recovery-mandates",
                content_hash=recovery.recovery_mandate_hash,
                model=PortfolioStrategyRecoveryMandate,
                identity_field="mandate_hash",
            )
            research_mandate = self.load(
                category="regularization/mandates",
                content_hash=recovery_mandate.prior_mandate_hash,
                model=PortfolioStrategyResearchMandate,
                identity_field="mandate_hash",
            )
            risk_store = RiskArtifactStore(self.root.parent)
            covariance_surface = HistoricalCovarianceSurface.model_validate(
                risk_store.load_json(
                    category="covariance/surfaces",
                    uri=risk_store.uri("covariance/surfaces", research_mandate.risk_surface_hash),
                    identity_field="surface_hash",
                )
            )
            return_surface = RiskReturnArtifactStore(self.root.parent).load_manifest(
                covariance_surface.return_surface_hash
            )
        except PortfolioPublicationError:
            raise
        except Exception as error:
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.risk_calibration_publication_tampered"
            ) from error
        if len(candidate_set.candidates) != 1:
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.current_conditional_candidate_count_invalid"
            )
        candidate = candidate_set.candidates[0]
        recipes = tuple(
            value
            for value in research_mandate.recipes
            if value.source == candidate.source and value.candidate_id == candidate.candidate_id
        )
        if len(recipes) != 1:
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.current_conditional_alpha_recipe_invalid"
            )
        alpha_recipe = recipes[0]
        if (
            prior_marker.diagnostic_hash != prior_bundle.diagnostic_hash
            or prior_marker.evidence_delta_hash != prior_bundle.evidence_delta_hash
            or prior_marker.bundle_hash != prior_bundle.bundle_hash
            or prior_marker.status != prior_bundle.status
            or prior_marker.marker_hash != diagnostic.prior_attribution_marker_hash
            or prior_bundle.bundle_hash != diagnostic.prior_attribution_bundle_hash
            or prior_bundle.agent_review_hash != diagnostic.prior_agent_review_hash
            or prior_bundle.candidate_set_hash != diagnostic.candidate_set_hash
            or candidate.selected_trial_evidence_hash != diagnostic.selected_trial_evidence_hash
            or selected_trial.evidence_hash != diagnostic.selected_trial_evidence_hash
            or selected_trial.mandate_hash != recovery_mandate.mandate_hash
            or selected_trial.status != "COMPLETED"
            or selected_trial.source != candidate.source
            or selected_trial.candidate_id != candidate.candidate_id
            or selected_trial.policy != candidate.policy
            or candidate_set.recovery_mandate_hash != recovery_mandate.mandate_hash
            or candidate_set.decision_dossier_hash != recovery.decision_dossier_hash
            or candidate_set.agent_review_hash != recovery.agent_review_hash
            or recovery.candidate_set_hash != candidate_set.candidate_set_hash
            or candidate_set.policy_holdout_state != "SEALED"
            or candidate_set.system_holdout_state != "UNREAD"
            or recovery.policy_holdout_state != "SEALED"
            or recovery.system_holdout_state != "UNREAD"
            or research_mandate.mandate_hash != diagnostic.research_mandate_hash
            or research_mandate.policy_holdout_state != "SEALED"
            or research_mandate.system_holdout_state != "UNREAD"
            or covariance_surface.surface_hash != diagnostic.risk_surface_hash
            or return_surface.surface_hash != diagnostic.return_surface_hash
            or bundle.prior_attribution_marker_hash != prior_marker.marker_hash
            or bundle.prior_attribution_bundle_hash != prior_bundle.bundle_hash
            or bundle.classification != diagnostic.classification
            or bundle.conclusion != diagnostic.conclusion
            or bundle.recommended_next_mandate != diagnostic.recommended_next_mandate
            or bundle.policy_holdout_state != diagnostic.policy_holdout_state
            or bundle.system_holdout_state != diagnostic.system_holdout_state
        ):
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.risk_calibration_current_lineage_invalid"
            )
        return (
            candidate_set,
            candidate,
            selected_trial,
            recovery,
            recovery_mandate,
            research_mandate,
            alpha_recipe,
        )

    def _read_recovery_current(
        self,
        pointer: CurrentPortfolioStrategyResearchPointer,
    ) -> CurrentPortfolioStrategyRecoveryPublication:
        marker, bundle = self._read_recovery_marker(
            marker_hash=pointer.marker_hash,
            allow_historical_review=False,
        )
        return CurrentPortfolioStrategyRecoveryPublication(
            action="READ_CURRENT",
            marker=marker,
            bundle=bundle,
            pointer=pointer,
        )

    def read_recovery_history(
        self,
        marker_hash: str,
    ) -> PortfolioStrategyRecoveryPublicationBundle:
        """Read an immutable old Recovery publication without admitting it as current."""
        _marker, bundle = self._read_recovery_marker(
            marker_hash=marker_hash,
            allow_historical_review=True,
        )
        return bundle

    def _read_recovery_marker(
        self,
        *,
        marker_hash: str,
        allow_historical_review: bool,
    ) -> tuple[CurrentPortfolioStrategyRecoveryMarker, PortfolioStrategyRecoveryPublicationBundle]:
        marker = self.load(
            category="regularization/recovery-current-markers",
            content_hash=marker_hash,
            model=CurrentPortfolioStrategyRecoveryMarker,
            identity_field="marker_hash",
        )
        bundle = self.load(
            category="regularization/recovery-projection-bundles",
            content_hash=marker.bundle_hash,
            model=PortfolioStrategyRecoveryPublicationBundle,
            identity_field="bundle_hash",
        )
        if (
            marker.recovery_mandate_hash != bundle.recovery_mandate_hash
            or marker.decision_dossier_hash != bundle.decision_dossier_hash
            or marker.bundle_hash != bundle.bundle_hash
            or marker.status != bundle.status
        ):
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.recovery_current_lineage_invalid"
            )
        self._validate_recovery_bundle(
            bundle,
            allow_historical_review=allow_historical_review,
        )
        return marker, bundle

    def _validate_recovery_bundle(
        self,
        bundle: PortfolioStrategyRecoveryPublicationBundle,
        *,
        allow_historical_review: bool,
    ) -> None:
        try:
            mandate = self.load(
                category="regularization/recovery-mandates",
                content_hash=bundle.recovery_mandate_hash,
                model=PortfolioStrategyRecoveryMandate,
                identity_field="mandate_hash",
            )
            audit = self.load(
                category="regularization/evidence-classification-audits",
                content_hash=bundle.evidence_audit_hash,
                model=PortfolioEvidenceClassificationAudit,
                identity_field="audit_hash",
            )
            shared = self.load(
                category="regularization/shared-config-dossiers",
                content_hash=bundle.shared_config_dossier_hash,
                model=PortfolioSharedConfigEvidenceDossier,
                identity_field="dossier_hash",
            )
            dossier = self.load(
                category="regularization/recovery-decision-dossiers",
                content_hash=bundle.decision_dossier_hash,
                model=PortfolioRecoveryDecisionDossier,
                identity_field="dossier_hash",
            )
            try:
                review: PortfolioRecoveryAgentReview | HistoricalPortfolioRecoveryAgentReview = (
                    self.load(
                        category="regularization/recovery-agent-reviews",
                        content_hash=bundle.agent_review_hash,
                        model=PortfolioRecoveryAgentReview,
                        identity_field="review_hash",
                    )
                )
            except PortfolioPublicationError:
                if not allow_historical_review:
                    raise
                review = self.load(
                    category="regularization/recovery-agent-reviews",
                    content_hash=bundle.agent_review_hash,
                    model=HistoricalPortfolioRecoveryAgentReview,
                    identity_field="review_hash",
                )
            if isinstance(review, PortfolioRecoveryAgentReview):
                board: PortfolioRecoveryResearchBoard | HistoricalPortfolioRecoveryResearchBoard = (
                    self.load(
                        category="regularization/recovery-research-boards",
                        content_hash=review.board_hash,
                        model=PortfolioRecoveryResearchBoard,
                        identity_field="board_hash",
                    )
                )
            else:
                if not allow_historical_review:
                    raise PortfolioPublicationError(
                        "portfolio_strategy_lab.recovery_current_lineage_invalid"
                    )
                board = self.load(
                    category="regularization/recovery-research-boards",
                    content_hash=review.board_hash,
                    model=HistoricalPortfolioRecoveryResearchBoard,
                    identity_field="board_hash",
                )
            candidates = (
                self.load(
                    category="regularization/conditional-candidate-sets",
                    content_hash=bundle.candidate_set_hash,
                    model=PortfolioConditionalPolicyCandidateSet,
                    identity_field="candidate_set_hash",
                )
                if bundle.candidate_set_hash is not None
                else None
            )
        except PortfolioPublicationError:
            raise
        except Exception as error:
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.recovery_current_lineage_invalid"
            ) from error
        candidate_count = len(candidates.candidates) if candidates is not None else 0
        shared_by_handle = {value.semantic_region_handle: value for value in shared.candidates}
        evidence_by_handle = {value.semantic_handle: value for value in dossier.candidate_evidence}
        selected_handles = review.bounded_decision.selected_candidate_handles
        candidate_handles = (
            tuple(value.semantic_handle for value in candidates.candidates)
            if candidates is not None
            else ()
        )
        candidate_lineage_valid = all(
            value.semantic_handle in shared_by_handle
            and value.semantic_handle in evidence_by_handle
            and value.source == shared_by_handle[value.semantic_handle].source
            and value.candidate_id == shared_by_handle[value.semantic_handle].candidate_id
            and value.policy == shared_by_handle[value.semantic_handle].policy
            and value.classification == shared_by_handle[value.semantic_handle].classification
            and value.selected_trial_evidence_hash
            == shared_by_handle[value.semantic_handle].selected_trial_evidence_hash
            and value.source == evidence_by_handle[value.semantic_handle].source
            and value.classification == evidence_by_handle[value.semantic_handle].classification
            for value in (candidates.candidates if candidates is not None else ())
        )
        if (
            mandate.evidence_audit_hash != audit.audit_hash
            or shared.evidence_audit_hash != audit.audit_hash
            or dossier.evidence_audit_hash != audit.audit_hash
            or dossier.shared_config_dossier_hash != shared.dossier_hash
            or review.dossier_hash != dossier.dossier_hash
            or board.dossier_hash != dossier.dossier_hash
            or review.board_hash != board.board_hash
            or (
                isinstance(review, PortfolioRecoveryAgentReview)
                and (
                    not isinstance(board, PortfolioRecoveryResearchBoard)
                    or board.review_binding_hash != review.review_binding_hash
                )
            )
            or review.decision_owner != bundle.decision_owner
            or review.model_calls != bundle.model_call_count
            or review.agent_failure_count != bundle.agent_failure_count
            or review.typed_repair_count != bundle.typed_repair_count
            or bundle.open_analysis != review.open_analysis
            or bundle.constrained_decision != review.bounded_decision
            or bundle.conditional_candidate_count != candidate_count
            or candidate_handles != selected_handles
            or not candidate_lineage_valid
            or (
                candidates is not None
                and (
                    candidates.recovery_mandate_hash != mandate.mandate_hash
                    or candidates.decision_dossier_hash != dossier.dossier_hash
                    or candidates.agent_review_hash != review.review_hash
                    or candidates.selection_owner != review.decision_owner
                )
            )
        ):
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.recovery_current_lineage_invalid"
            )

    def _validate_regularization_bundle(
        self,
        bundle: PortfolioStrategyResearchPublicationBundle,
    ) -> None:
        try:
            mandate = self.load(
                category="regularization/mandates",
                content_hash=bundle.mandate_hash,
                model=PortfolioStrategyResearchMandate,
                identity_field="mandate_hash",
            )
            dossier = self.load(
                category="regularization/dossiers",
                content_hash=bundle.dossier_hash,
                model=PortfolioCrossFoldEvidenceDossier,
                identity_field="dossier_hash",
            )
            review = self.load(
                category="regularization/research-reviews",
                content_hash=bundle.review_hash,
                model=PortfolioResearchScientistReview,
                identity_field="review_hash",
            )
            report = self.load(
                category="regularization/agent-value-reports",
                content_hash=bundle.agent_value_report_hash,
                model=PortfolioAgentValueReport,
                identity_field="report_hash",
            )
            board = self.load(
                category="regularization/research-boards",
                content_hash=review.board_hash,
                model=PortfolioResearchBoard,
                identity_field="board_hash",
            )
            if bundle.validation_slate_hash is not None:
                raise PortfolioPublicationError(
                    "portfolio_strategy_lab.regularization_validation_slate_route_retired"
                )
        except PortfolioPublicationError:
            raise
        except Exception as error:
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.regularization_current_lineage_invalid"
            ) from error
        accepted_keys = review.terminal.accepted_candidate_keys
        dossier_keys = {value.stable_policy_key for value in dossier.stable_candidates}
        candidate_count = 0
        if (
            dossier.mandate_hash != mandate.mandate_hash
            or dossier.mandate_hash != bundle.mandate_hash
            or review.dossier_hash != dossier.dossier_hash
            or report.dossier_hash != dossier.dossier_hash
            or board.dossier_hash != dossier.dossier_hash
            or review.value_report_hash != report.report_hash
            or review.board_hash != report.board_hash
            or review.board_hash != board.board_hash
            or review.terminal.route is not report.agent_route
            or review.terminal.route is not bundle.terminal_route
            or report.classification is not bundle.agent_value_classification
            or report.compute_attempts_used != bundle.experiment_attempt_count
            or report.model_calls != bundle.model_call_count
            or sum(value.attempt_count for value in board.evidence_deltas)
            != bundle.experiment_attempt_count
            or set(accepted_keys) - dossier_keys
            or len(accepted_keys) != candidate_count
            or bundle.validation_candidate_count != candidate_count
        ):
            raise PortfolioPublicationError(
                "portfolio_strategy_lab.regularization_current_lineage_invalid"
            )


__all__ = [
    "CurrentPortfolioFold3AttributionPublication",
    "CurrentPortfolioResearchPublication",
    "CurrentPortfolioRiskCalibrationPublication",
    "CurrentPortfolioStrategyPublication",
    "CurrentPortfolioStrategyResearchPublication",
    "PortfolioPublicationError",
    "PortfolioResearchArtifactStore",
]
