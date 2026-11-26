"""Responsibility-bound source and numerical-environment identities."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from alphalattice.kernel.shared_kernel.environment import recorded_environment
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.project_layout import source_root
from alphalattice.kernel.shared_kernel.source_identity import (
    switched_source_identity,
)

from .impact import AlphaVerificationDomain

_FULL_PARITY_DOMAINS = frozenset({"FOUNDATION_INPUT", "ARRAY_VALUES", "MODEL_KERNEL", "UNKNOWN"})


def _domain_source_paths(
    playpen_root: Path,
) -> dict[AlphaVerificationDomain, dict[str, Path]]:
    return {
        "FOUNDATION_INPUT": {
            "feature_engine.producers.base_materializer": source_root(playpen_root)
            / "alphalattice"
            / "foundation"
            / "feature_engine"
            / "producers"
            / "base_materializer.py",
            "feature_engine.catalog.contracts": source_root(playpen_root)
            / "alphalattice"
            / "foundation"
            / "feature_engine"
            / "catalog"
            / "contracts.py",
            "feature_engine.producers.cross_section": source_root(playpen_root)
            / "alphalattice"
            / "foundation"
            / "feature_engine"
            / "producers"
            / "cross_section.py",
            "feature_engine.producers.preprocessing.robust_cross_section": source_root(playpen_root)
            / "alphalattice"
            / "foundation"
            / "feature_engine"
            / "producers"
            / "preprocessing"
            / "robust_cross_section.py",
            "kernel.quant.cross_section": source_root(playpen_root)
            / "alphalattice"
            / "kernel"
            / "quant"
            / "cross_section.py",
            "feature_engine.resources.desktop_feature_catalog": source_root(playpen_root)
            / "alphalattice"
            / "foundation"
            / "feature_engine"
            / "resources"
            / "desktop-feature-catalog.json",
            "feature_engine.runtime.service": source_root(playpen_root)
            / "alphalattice"
            / "foundation"
            / "feature_engine"
            / "runtime"
            / "service.py",
        },
        "ARRAY_VALUES": {
            "alpha_research.inputs.frozen_price_volume": source_root(playpen_root)
            / "alphalattice/investment/alpha_research/inputs/frozen_price_volume.py",
            "alpha_research.scaling.contracts": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "scaling"
            / "contracts.py",
            "alpha_research.scaling.adapters": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "scaling"
            / "adapters.py",
            "alpha_research.scaling.catalog": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "scaling"
            / "catalog.py",
            "alpha_research.scaling.execution": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "scaling"
            / "execution.py",
            "alpha_research.targets.canonical": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "targets"
            / "canonical.py",
            "alpha_research.targets.standardization": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "targets"
            / "standardization.py",
            "alpha_research.scores.score_filters": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "scores"
            / "score_filters.py",
            "alpha_research.scores.temporal_aggregation": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "scores"
            / "temporal_aggregation.py",
            "alpha_research.inputs.folds": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "inputs"
            / "folds.py",
            "alpha_research.inputs.surfaces": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "inputs"
            / "surfaces.py",
            "feature_engine.panels.reader": source_root(playpen_root)
            / "alphalattice"
            / "foundation"
            / "feature_engine"
            / "panels"
            / "reader.py",
            "causal_outcomes.execution.contracts": source_root(playpen_root)
            / "alphalattice"
            / "foundation"
            / "causal_outcomes"
            / "execution"
            / "contracts.py",
            "causal_outcomes.execution.compile": source_root(playpen_root)
            / "alphalattice"
            / "foundation"
            / "causal_outcomes"
            / "execution"
            / "compile.py",
            "causal_outcomes.execution.artifacts": source_root(playpen_root)
            / "alphalattice"
            / "foundation"
            / "causal_outcomes"
            / "execution"
            / "artifacts.py",
            "causal_outcomes.execution.publication": source_root(playpen_root)
            / "alphalattice"
            / "foundation"
            / "causal_outcomes"
            / "execution"
            / "publication.py",
            "causal_outcomes.execution.readers": source_root(playpen_root)
            / "alphalattice"
            / "foundation"
            / "causal_outcomes"
            / "execution"
            / "readers.py",
            "causal_outcomes.execution.methods": source_root(playpen_root)
            / "alphalattice"
            / "foundation"
            / "causal_outcomes"
            / "execution"
            / "methods.py",
            "alphalattice.kernel.validation.splitting": source_root(playpen_root)
            / "alphalattice"
            / "kernel"
            / "validation"
            / "splitting.py",
        },
        "MODEL_KERNEL": {
            "alpha_research.targets.component_training": source_root(playpen_root)
            / "alphalattice/investment/alpha_research/targets/component_training.py",
            "alpha_research.scores.model_renewal": source_root(playpen_root)
            / "alphalattice/investment/alpha_research/scores/model_renewal.py",
            "alpha_research.scores.lifecycle_preparation": source_root(playpen_root)
            / "alphalattice/investment/alpha_research/scores/lifecycle_preparation.py",
            "alpha_research.scores.product_lifecycle": source_root(playpen_root)
            / "alphalattice/investment/alpha_research/scores/product_lifecycle.py",
            "alpha_modeling": source_root(playpen_root)
            / "alphalattice"
            / "capabilities"
            / "alpha_modeling"
            / "__init__.py",
            "alpha_modeling.adapters.regularized_linear": source_root(playpen_root)
            / "alphalattice"
            / "capabilities"
            / "alpha_modeling"
            / "adapters"
            / "regularized_linear.py",
            "alpha_modeling.runtime.service": source_root(playpen_root)
            / "alphalattice"
            / "capabilities"
            / "alpha_modeling"
            / "runtime"
            / "service.py",
            "alpha_research.scores.refit": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "scores"
            / "refit.py",
            "alpha_research.scores.product_recipe": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "scores"
            / "product_recipe.py",
            "alpha_research.experiments.execution": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "experiments"
            / "execution.py",
        },
        "METRICS": {
            "alpha_research.evaluation.metrics": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "evaluation"
            / "metrics.py",
            "alpha_research.evaluation.viability": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "evaluation"
            / "viability.py",
            "alphalattice.kernel.validation.return_model_statistics": source_root(playpen_root)
            / "alphalattice"
            / "kernel"
            / "validation"
            / "return_model_statistics.py",
        },
        "QUALIFICATION_POLICY": {
            "alpha_research.evaluation.stability": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "evaluation"
            / "stability.py",
            "alpha_research.experiments.contracts": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "experiments"
            / "contracts.py",
            "alpha_research.experiments.mandate": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "experiments"
            / "mandate.py",
            "alpha_research.candidates.qualification": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "candidates"
            / "qualification.py",
            "alpha_research.candidates.control": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "candidates"
            / "control.py",
            "alpha_research.experiments.family_qualification": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "experiments"
            / "family_qualification.py",
            "alpha_research.candidates.qualification_task": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "candidates"
            / "qualification_task.py",
        },
        "ARTIFACT_RUNTIME": {
            "alpha_research.experiments.lifecycle_authoring": source_root(playpen_root)
            / "alphalattice/investment/alpha_research/experiments/lifecycle_authoring.py",
            "alpha_research.scores.frozen_inference": source_root(playpen_root)
            / "alphalattice/investment/alpha_research/scores/frozen_inference.py",
            "alpha_research.verification.identity": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "verification"
            / "identity.py",
            "alpha_research.verification.impact": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "verification"
            / "impact.py",
            "alpha_research.inputs.observations": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "inputs"
            / "observations.py",
            "alpha_research.publication.artifacts": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "publication"
            / "artifacts.py",
            "alpha_research.candidates.artifacts": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "candidates"
            / "artifacts.py",
            "alpha_research.candidates.committer": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "candidates"
            / "committer.py",
            "alpha_research.scores.heterogeneous_product": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "scores"
            / "heterogeneous_product.py",
            "alpha_research.scores.heterogeneous_replay": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "scores"
            / "heterogeneous_replay.py",
            "alpha_research.scores.product_replay": source_root(playpen_root)
            / "alphalattice"
            / "investment"
            / "alpha_research"
            / "scores"
            / "product_replay.py",
            "playpen.scripts.check_alpha_verification_impact": playpen_root
            / "scripts"
            / "check_alpha_verification_impact.py",
            "playpen.scripts.check_playpen": playpen_root / "scripts" / "check_playpen.py",
            # The live-closure runners and the ported research closure they execute.
            "playpen.scripts.broad_ensemble_research_closure": playpen_root
            / "scripts"
            / "broad_ensemble_research_closure.py",
            "playpen.scripts.run_broad_ensemble_live_model_closure": playpen_root
            / "scripts"
            / "run_broad_ensemble_live_model_closure.py",
            "playpen.scripts.run_heterogeneous_live_score_closure": playpen_root
            / "scripts"
            / "run_heterogeneous_live_score_closure.py",
            "playpen.scripts.run_monthly_alpha_refit_research": playpen_root
            / "scripts"
            / "run_monthly_alpha_refit_research.py",
        },
        "TEST_ONLY": {},
        "DOCUMENTATION": {},
        "OUT_OF_SCOPE": {},
        "UNKNOWN": {},
    }


def alpha_execution_domain_hashes(playpen_root: Path) -> dict[str, str]:
    """Hash installed Alpha verification domains and their required numerical environment.

    Args:
        playpen_root: Repository root resolving the current domain source closures.

    Returns:
        Nonempty source-domain identities using switched source identity; only full-parity domains
        include numerical environment authority.
    """
    environment = alpha_numerical_environment_identity()
    result: dict[str, str] = {}
    for domain, sources in _domain_source_paths(playpen_root).items():
        if not sources:
            continue
        result[domain] = cast(
            str,
            canonical_hash(
                {
                    "sources": switched_source_identity(
                        sources,
                        semantic_owner="alpha_research.verification",
                        numerical_role=domain,
                    ),
                    "environment": environment if domain in _FULL_PARITY_DOMAINS else None,
                }
            ),
        )
    return result


def alpha_numerical_environment_identity() -> dict[str, object]:
    """The environment Alpha's numbers run in: provenance recorded beside a result."""
    return recorded_environment(
        ("numpy", "pyarrow", "scikit-learn", "scipy", "statsmodels", "lightgbm")
    )


__all__ = [
    "alpha_execution_domain_hashes",
    "alpha_numerical_environment_identity",
]
