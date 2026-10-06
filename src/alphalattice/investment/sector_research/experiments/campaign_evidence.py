"""The Campaign root: a comparable dossier, an actor-neutral decision, a replay.

One Campaign publishes a dossier naming every per-method experiment it ran, the
calibration each earned, and the comparison that ranked them. A submitter --
Human, Installed Agent or external automation, equivalently -- proposes which
method the evidence supports; the Host validates that proposal against the
frozen rules and seals the selection. No submitter can seal, and a submission
that disagrees with what the published comparison actually decided is refused
rather than sealed with a note.

The replay re-admits rather than re-compares. It rebuilds the installed catalog,
holds the dossier's catalog identity against it, walks every experiment through
the evidence verifier down to the outcome method seal, recomputes each
calibration and the comparison from the published children, and re-derives the
recommendation through the same frozen rule. Nothing is taken on the dossier's
word.

Call counting is explicit about what a call is. ``forecast_call_count`` counts
invocations of a method adapter and ``fit_call_count`` counts estimator fits;
recomputing a pure function over already-published bytes is verification, not a
numerical call, which is why a replay that re-derives every evaluation and
calibration still reports zero. Provider, Holdout and pointer counters are zero
by construction: no code path here can reach any of them.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.kernel.shared_kernel.source_identity import (
    switched_source_identity,
)
from alphalattice.protocols.actor_execution import (
    ActorKind,
    ActorSubmissionBinding,
    seal_actor_submission,
)

from ..contracts import SectorResearchError
from ..evaluation.calibration import (
    SectorShrinkCalibrationEvidence,
    calibrate_sector_forecast_surface,
)
from ..evaluation.comparison import (
    SectorMethodComparisonEvidence,
    derive_sector_recommendation,
)
from ..models.catalog import SECTOR_CATALOG_ROLE
from ..models.contracts import SectorForecastCatalogBinding
from .development_artifacts import (
    SECTOR_CAMPAIGN_DECISION_CATEGORY,
    SECTOR_CAMPAIGN_DOSSIER_CATEGORY,
    SECTOR_CAMPAIGN_REPLAY_CATEGORY,
    SECTOR_METHOD_COMPARISON_CATEGORY,
    SECTOR_SHRINK_CALIBRATION_CATEGORY,
    SectorArtifactReadbackError,
    SectorDevelopmentArtifactStore,
    sector_artifact_uri,
)
from .resolution import SectorForecastSelection
from .verification import SectorEvidenceVerifier


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


def sector_campaign_decision_policy_hash() -> str:
    """Byte identity of the rules that turn evidence into a selection.

    The comparison module owns the selection rule and the joint-dynamics
    trigger, and this module owns admission. A decision sealed under different
    rule code is a different decision even when every input hash agrees, so the
    bytes of both enter the receipt and the replay refuses a policy that is no
    longer the installed one.
    """

    here = Path(__file__)
    package = here.parents[1]
    return switched_source_identity(
        {
            "sector_research.evaluation.comparison": package / "evaluation" / "comparison.py",
            "sector_research.evaluation.calibration": package / "evaluation" / "calibration.py",
            "sector_research.experiments.campaign_evidence": here,
        },
        semantic_owner="sector_research",
        numerical_role="SECTOR_CAMPAIGN_DECISION_POLICY",
    )


class SectorCampaignDossier(_Contract):
    """The root of one Campaign: every method it compared and what that produced."""

    kind: Literal["SectorCampaignDossier"] = "SectorCampaignDossier"
    identity_class: Literal["DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"] = (
        "DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"
    )
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    causal_outcome_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sector_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_start: date
    training_end: date
    forecast_start: date
    forecast_end: date
    calibration_fold_count: int = Field(ge=2)
    ordered_experiment_hashes: tuple[str, ...] = Field(min_length=2)
    ordered_calibration_hashes: tuple[str, ...] = Field(min_length=2)
    comparison_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    conditional_dispositions: dict[str, Literal["NOT_TRIGGERED", "EVALUATED"]]
    forecast_call_count: int = Field(ge=1)
    """Adapter invocations the Host made. An estimator fit happens inside one of
    these and is the adapter's own business, so it is deliberately not counted
    here as though the Host had performed it."""

    metric_call_count: int = Field(ge=1)
    provider_call_count: Literal[0] = 0
    holdout_access_count: Literal[0] = 0
    pointer_mutation_count: Literal[0] = 0
    limitations: tuple[str, ...] = Field(min_length=1)
    ordered_child_uris: tuple[str, ...] = Field(min_length=3)
    dossier_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if len(self.ordered_calibration_hashes) != len(self.ordered_experiment_hashes):
            raise SectorResearchError("sector_research.campaign_dossier_child_axis_invalid")
        for values in (self.ordered_experiment_hashes, self.ordered_calibration_hashes):
            if len(set(values)) != len(values):
                raise SectorResearchError("sector_research.campaign_dossier_child_duplicated")
        expected = (
            *(
                sector_artifact_uri(SECTOR_SHRINK_CALIBRATION_CATEGORY, value)
                for value in self.ordered_calibration_hashes
            ),
            sector_artifact_uri(SECTOR_METHOD_COMPARISON_CATEGORY, self.comparison_hash),
        )
        if self.ordered_child_uris != expected:
            raise SectorResearchError("sector_research.campaign_dossier_children_invalid")
        if self.training_start > self.training_end or self.forecast_start > self.forecast_end:
            raise SectorResearchError("sector_research.campaign_dossier_range_invalid")
        if self.dossier_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"dossier_hash"})
        ):
            raise SectorResearchError("sector_research.campaign_dossier_identity_invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        draft = dict(values)
        draft.pop("dossier_hash", None)
        provisional = cls.model_construct(**draft, dossier_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"dossier_hash"})
        return cast(
            Self, cls.model_validate({**identity, "dossier_hash": canonical_hash(identity)})
        )


class SectorForecastSelectionSubmission(_Contract):
    """A proposal about which method the published evidence supports.

    Carries no authority and no identity the Host would otherwise derive: the
    submitter names the dossier it read and the method it believes that dossier
    chose, and the Host checks that belief against the comparison.
    """

    kind: Literal["SectorForecastSelectionSubmission"] = "SectorForecastSelectionSubmission"
    dossier_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_method_id: str = Field(min_length=1, max_length=96)
    rationale: str = Field(min_length=1, max_length=2000)
    submission_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        payload = {"kind": "SectorForecastSelectionSubmission", **values}
        identity = cls.model_construct(**payload, submission_hash="0" * 64).model_dump(
            mode="json", exclude={"submission_hash"}
        )
        return cls(**payload, submission_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if self.submission_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"submission_hash"})
        ):
            raise SectorResearchError("sector_research.campaign_submission_identity_invalid")
        return self


class SectorForecastSelectionReceipt(_Contract):
    """The Host-sealed selection: who proposed it, under which policy, and what it is."""

    kind: Literal["SectorForecastSelectionReceipt"] = "SectorForecastSelectionReceipt"
    submission: SectorForecastSelectionSubmission
    actor_binding: ActorSubmissionBinding
    decision_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    comparison_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection: SectorForecastSelection
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if self.actor_binding.submission_hash != self.submission.submission_hash:
            raise SectorResearchError("sector_research.campaign_receipt_actor_route_invalid")
        if self.selection.method_id != self.submission.selected_method_id:
            raise SectorResearchError("sector_research.campaign_receipt_selection_route_invalid")
        if self.receipt_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        ):
            raise SectorResearchError("sector_research.campaign_receipt_identity_invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        draft = dict(values)
        draft.pop("receipt_hash", None)
        provisional = cls.model_construct(**draft, receipt_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"receipt_hash"})
        return cast(
            Self, cls.model_validate({**identity, "receipt_hash": canonical_hash(identity)})
        )


class SectorCampaignReplayReceipt(_Contract):
    """A completed zero-call replay of one Campaign graph."""

    kind: Literal["SectorCampaignReplayReceipt"] = "SectorCampaignReplayReceipt"
    dossier_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    verified_child_count: int = Field(ge=1)
    disposition: Literal["REUSED_EXACT"] = "REUSED_EXACT"
    forecast_call_count: Literal[0] = 0
    fit_call_count: Literal[0] = 0
    metric_call_count: Literal[0] = 0
    provider_call_count: Literal[0] = 0
    holdout_access_count: Literal[0] = 0
    pointer_mutation_count: Literal[0] = 0
    replay_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if self.replay_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"replay_hash"})
        ):
            raise SectorResearchError("sector_research.campaign_replay_identity_invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        draft = dict(values)
        draft.pop("replay_hash", None)
        provisional = cls.model_construct(**draft, replay_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"replay_hash"})
        return cast(Self, cls.model_validate({**identity, "replay_hash": canonical_hash(identity)}))


def admit_sector_forecast_selection(
    *,
    dossier: SectorCampaignDossier,
    comparison: SectorMethodComparisonEvidence,
    submission: SectorForecastSelectionSubmission,
    actor_kind: ActorKind,
    actor_id: str,
) -> SectorForecastSelectionReceipt:
    """Validate a proposal against the frozen rules, then seal it.

    The submitter's claim is checked against what the published comparison
    actually decided, re-derived here from its own rows rather than read off its
    recommendation field -- a comparison whose stored recommendation disagrees
    with its rows is exactly the forgery this check exists for.
    """

    if submission.dossier_hash != dossier.dossier_hash:
        raise SectorResearchError("sector_research.campaign_submission_not_this_dossier")
    if comparison.comparison_hash != dossier.comparison_hash:
        raise SectorResearchError("sector_research.campaign_comparison_not_this_dossier")
    recommended, _ = derive_sector_recommendation(comparison.rows)
    if recommended != comparison.recommended_method_id:
        raise SectorResearchError("sector_research.campaign_comparison_recommendation_forged")
    if submission.selected_method_id != recommended:
        raise SectorResearchError("sector_research.campaign_submission_contradicts_evidence")

    row = {value.method_id: value for value in comparison.rows}[recommended]
    selection = (
        SectorForecastSelection.control_zero()
        if row.experiment_hash is None
        or recommended == SectorForecastSelection.control_zero().method_id
        else SectorForecastSelection.from_experiment(
            method_id=recommended, sector_experiment_hash=row.experiment_hash
        )
    )
    return SectorForecastSelectionReceipt.create(
        kind="SectorForecastSelectionReceipt",
        submission=submission,
        actor_binding=seal_actor_submission(
            actor_kind=actor_kind,
            actor_id=actor_id,
            submission_hash=submission.submission_hash,
        ),
        decision_policy_hash=sector_campaign_decision_policy_hash(),
        comparison_hash=comparison.comparison_hash,
        selection=selection,
    )


def verify_sector_campaign_recursive_replay(
    *,
    store: SectorDevelopmentArtifactStore,
    outcome_reader: CausalExecutionOutcomeDevelopmentReader,
    catalog_binding: SectorForecastCatalogBinding,
    dossier_hash: str,
    decision_receipt_hash: str,
) -> SectorCampaignReplayReceipt:
    """Walk the whole Campaign graph back to its authority, computing nothing new."""

    dossier = store.load_contract(
        category=SECTOR_CAMPAIGN_DOSSIER_CATEGORY,
        content_hash=dossier_hash,
        identity_field="dossier_hash",
        model=SectorCampaignDossier,
    )
    if not is_current(SECTOR_CATALOG_ROLE, dossier.catalog_hash, catalog_binding.catalog_hash):
        raise SectorResearchError("sector_research.campaign_replay_catalog_not_installed")

    verifier = SectorEvidenceVerifier(
        store=store, outcome_reader=outcome_reader, catalog_binding=catalog_binding
    )
    verified = 0
    calibrations: list[SectorShrinkCalibrationEvidence] = []
    for experiment_hash, calibration_hash in zip(
        dossier.ordered_experiment_hashes, dossier.ordered_calibration_hashes, strict=True
    ):
        lineage = verifier.verify(experiment_hash=experiment_hash)
        verified += 4  # experiment root, target evidence, surface, evaluation
        if lineage.target_evidence.evidence_hash != dossier.target_evidence_hash:
            raise SectorResearchError("sector_research.campaign_replay_target_not_shared")
        try:
            calibration = store.load_contract(
                category=SECTOR_SHRINK_CALIBRATION_CATEGORY,
                content_hash=calibration_hash,
                identity_field="calibration_hash",
                model=SectorShrinkCalibrationEvidence,
            )
        except FileNotFoundError as error:
            raise SectorResearchError(
                f"sector_research.campaign_replay_child_missing:{error}"
            ) from error
        except (SectorArtifactReadbackError, ValueError) as error:
            raise SectorResearchError(
                f"sector_research.campaign_replay_child_unverifiable:{error}"
            ) from error
        derived = calibrate_sector_forecast_surface(
            evidence=lineage.target_evidence,
            surface=lineage.surface,
            fold_count=dossier.calibration_fold_count,
        )
        if derived.calibration_hash != calibration.calibration_hash:
            raise SectorResearchError("sector_research.campaign_replay_calibration_not_rederivable")
        if calibration.surface_hash != lineage.surface.surface_hash:
            raise SectorResearchError(
                "sector_research.campaign_replay_calibration_not_this_surface"
            )
        calibrations.append(calibration)
        verified += 1

    comparison = store.load_contract(
        category=SECTOR_METHOD_COMPARISON_CATEGORY,
        content_hash=dossier.comparison_hash,
        identity_field="comparison_hash",
        model=SectorMethodComparisonEvidence,
    )
    verified += 1
    if not is_current(SECTOR_CATALOG_ROLE, comparison.catalog_hash, catalog_binding.catalog_hash):
        raise SectorResearchError("sector_research.campaign_replay_comparison_catalog_invalid")
    if comparison.target_evidence_hash != dossier.target_evidence_hash:
        raise SectorResearchError("sector_research.campaign_replay_comparison_target_invalid")
    published = {value.calibration_hash for value in comparison.rows if value.calibration_hash}
    if published != {value.calibration_hash for value in calibrations}:
        raise SectorResearchError("sector_research.campaign_replay_comparison_children_invalid")
    recommended, triggered = derive_sector_recommendation(comparison.rows)
    if (
        recommended != comparison.recommended_method_id
        or triggered != comparison.joint_dynamics_triggered
    ):
        raise SectorResearchError("sector_research.campaign_replay_recommendation_not_rederivable")
    if triggered != any(
        value == "EVALUATED" for value in dossier.conditional_dispositions.values()
    ):
        raise SectorResearchError("sector_research.campaign_replay_conditional_disposition_invalid")

    receipt = store.load_contract(
        category=SECTOR_CAMPAIGN_DECISION_CATEGORY,
        content_hash=decision_receipt_hash,
        identity_field="receipt_hash",
        model=SectorForecastSelectionReceipt,
    )
    verified += 1
    if receipt.submission.dossier_hash != dossier.dossier_hash:
        raise SectorResearchError("sector_research.campaign_replay_decision_not_this_dossier")
    if receipt.comparison_hash != comparison.comparison_hash:
        raise SectorResearchError("sector_research.campaign_replay_decision_comparison_invalid")
    if receipt.decision_policy_hash != sector_campaign_decision_policy_hash():
        raise SectorResearchError("sector_research.campaign_replay_decision_policy_drift")
    if receipt.selection.method_id != recommended:
        raise SectorResearchError("sector_research.campaign_replay_decision_contradicts_evidence")

    return SectorCampaignReplayReceipt.create(
        kind="SectorCampaignReplayReceipt",
        dossier_hash=dossier.dossier_hash,
        decision_receipt_hash=receipt.receipt_hash,
        verified_child_count=verified,
    )


__all__ = [
    "SECTOR_CAMPAIGN_DECISION_CATEGORY",
    "SECTOR_CAMPAIGN_DOSSIER_CATEGORY",
    "SECTOR_CAMPAIGN_REPLAY_CATEGORY",
    "SectorCampaignDossier",
    "SectorCampaignReplayReceipt",
    "SectorForecastSelectionReceipt",
    "SectorForecastSelectionSubmission",
    "admit_sector_forecast_selection",
    "sector_campaign_decision_policy_hash",
    "verify_sector_campaign_recursive_replay",
]
