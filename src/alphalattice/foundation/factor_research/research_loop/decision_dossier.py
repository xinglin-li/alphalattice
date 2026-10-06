"""Compile one compact model-facing dossier for Factor Research judgment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.factor_research.programs.program import (
    FactorResearchDeterministicEvidence,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution.projections import (
    ModelFacingProjection,
    build_model_facing_projection,
)


class FactorResearchDecisionDossier(BaseModel):  # type: ignore[misc]
    """One content-addressed projection injected directly into an Agent turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["FactorResearchDecisionDossier"] = "FactorResearchDecisionDossier"
    review_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection: ModelFacingProjection
    dossier_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_dossier(self) -> FactorResearchDecisionDossier:
        """Verify the immutable review projection against its canonical dossier identity.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: The recorded dossier hash differs from its contents.
        """
        expected = canonical_hash(self.model_dump(mode="json", exclude={"dossier_hash"}))
        if self.dossier_hash != expected:
            raise ValueError("Factor Research decision dossier hash is invalid")
        return self


def build_factor_research_decision_dossier(
    result: FactorResearchDeterministicEvidence,
    *,
    review_binding_hash: str,
) -> FactorResearchDecisionDossier:
    """Project complete verified evidence directly into one Agent context.

    Args:
        result: Verified deterministic checkpoint carrying the complete evidence family.
        review_binding_hash: Exact review binding answered by this projection.

    Returns:
        Content-addressed model-facing dossier with complete statistics, cluster handles,
        Host authority boundaries, required limitations, and the action protocol.
    """
    date_text = result.program.frozen_at.date().isoformat()
    ordered_clusters = tuple(
        sorted(result.redundancy_structure.clusters, key=lambda value: value.member_factor_ids)
    )
    cluster_rows = tuple(
        (f"C{index:02d}", cluster) for index, cluster in enumerate(ordered_clusters, 1)
    )
    cluster_label_by_factor = {
        factor_id: label
        for label, cluster in cluster_rows
        for factor_id in cluster.member_factor_ids
    }
    lines = [
        "# Factor Research Decision Dossier",
        f"Date: {date_text}",
        "",
        "## Mandate and authority",
        "",
        "- Objective: daily close formation, next-common-session Open entry, and "
        "following-common-session Open exit.",
        "- Fit target: dividend-aware LOG_EXECUTION_RETURN; economic evaluation: "
        "SIMPLE_EXECUTION_RETURN.",
        "- The Host owns target construction, splits, statistics, clustering, validation, "
        "publication, and activation.",
        "- Current Alpha scientific stop remains active; Risk Research is not admitted.",
        "- Five- and twenty-one-session horizons, residualization, Rank-Gauss, Optuna, "
        "Alpha, and Risk are deferred.",
        "",
        "## Frozen evidence quality",
        "",
        f"- Registered factors: {len(result.program.factor_ids)}",
        f"- Complete development folds: {len(result.walk_forward_plan.formal_split.windows)}",
        f"- Valid target rows: {result.target_quality.valid_target_count}",
        f"- Typed missing target rows: {result.target_quality.typed_missing_count}",
        "- Sealed holdout is committed but unread.",
        "",
        "## Complete one-session evidence family",
        "",
        "All registered factors are shown. BY correction covers this complete family. "
        "Your research task is relative curation for downstream Alpha modeling, not a second "
        "significance gate. POSITIVE_OOS_EVIDENCE means favorable directional research "
        "evidence; its reason code states whether BY confirms it. CORE requires "
        "POSITIVE_OOS_EVIDENCE. CONDITIONAL requires "
        "MIXED_OOS_EVIDENCE and a concrete rationale. Multiple factors from one redundancy "
        "cluster are allowed. When eligible alternatives span multiple clusters, a "
        "multi-factor slate must not be concentrated entirely in one cluster. "
        "NO_DETECTABLE_EFFECT, NEGATIVE_OOS_EVIDENCE, and "
        "INSUFFICIENT_EVIDENCE cannot be selected.",
        "",
        "|Factor|Classification|Mean rank IC|BY q-value|Raw simple spread|"
        "Coverage|Cluster|Reasons|",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for item in result.evidence_report.items:
        lines.append(
            f"|`{item.factor_id}`|{item.classification.value}|"
            f"{_number(item.mean_oriented_rank_ic)}|{item.rank_ic_by_q_value:.5f}|"
            f"{_number(item.mean_oriented_simple_spread)}|"
            f"{item.validation_pair_coverage_mean:.5f}|"
            f"{cluster_label_by_factor[item.factor_id]}|"
            f"{', '.join(item.reason_codes) or 'none'}|"
        )
    lines.extend(
        (
            "",
            "## Redundancy clusters",
            "",
            "Cluster labels are semantic handles for this dossier; the Host retains opaque "
            "cluster identities.",
            "",
            "|Cluster|Members|",
            "|---|---|",
            *(
                f"|{label}|{', '.join(f'`{value}`' for value in cluster.member_factor_ids)}|"
                for label, cluster in cluster_rows
            ),
            "",
            "## Required limitations",
            "",
            "- `CURRENT_UNIVERSE_RESEARCH_ONLY`",
            "- `SEALED_HOLDOUT_UNREAD`",
            "- `ALPHA_SCIENTIFIC_STOP_PRESERVED`",
            "- `RISK_RESEARCH_NOT_ADMITTED`",
            "",
            "## Action protocol",
            "",
            "Stage the complete compact slate once with `stage_factor_research_choices`. "
            "When any positive or mixed evidence exists, you MUST curate a non-empty, compact "
            "slate of the relatively strongest factors for Alpha to test jointly. Factors in "
            "one cluster may coexist, but the full multi-factor slate must span more than one "
            "cluster when alternatives exist. Lack of BY-positive evidence does not justify "
            "an empty slate. Then "
            "call `submit_factor_research_proposal` once. Do not demand a fixed factor count.",
        )
    )
    projection = build_model_facing_projection(
        projection_id="factor-research-decision-dossier",
        semantics={
            "contract": "complete Host-verified one-session Factor Research decision context",
            "ordering": "factor_id",
            "hypothesis_family": "all_registered_factors",
            "authority": "model_interpretation_only",
        },
        content="\n".join(lines),
        source_hashes=(
            result.checkpoint_hash,
            result.program.program_hash,
            result.evidence_report.report_hash,
            result.redundancy_structure.structure_hash,
            review_binding_hash,
        ),
    )
    values = {
        "kind": "FactorResearchDecisionDossier",
        "review_binding_hash": review_binding_hash,
        "projection": projection.model_dump(mode="json"),
    }
    return FactorResearchDecisionDossier(
        **values,
        dossier_hash=canonical_hash(values),
    )


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


__all__ = [
    "FactorResearchDecisionDossier",
    "build_factor_research_decision_dossier",
]
