"""Changed-path classification and proportionate Alpha verification gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from alphalattice.kernel.shared_kernel.identity import canonical_hash

type AlphaVerificationDomain = Literal[
    "FOUNDATION_INPUT",
    "ARRAY_VALUES",
    "MODEL_KERNEL",
    "METRICS",
    "QUALIFICATION_POLICY",
    "ARTIFACT_RUNTIME",
    "TEST_ONLY",
    "DOCUMENTATION",
    "OUT_OF_SCOPE",
    "UNKNOWN",
]

_FULL_PARITY_DOMAINS = frozenset({"FOUNDATION_INPUT", "ARRAY_VALUES", "MODEL_KERNEL", "UNKNOWN"})

_DOMAIN_TESTS: dict[AlphaVerificationDomain, tuple[str, ...]] = {
    "FOUNDATION_INPUT": (
        "tests/feature_engine/test_desktop_feature_economics.py",
        "tests/feature_engine/test_feature_catalog_crud.py",
        "tests/feature_engine/test_feature_engine.py",
    ),
    "ARRAY_VALUES": (
        "tests/alpha_research/test_array_firewall.py",
        "tests/alpha_research/test_array_surface.py",
        # The two score owners routed into this domain are not reached by the
        # array-surface cases at all. These two are: the first drives
        # `apply_alpha_score_filter`, the second is the only caller of
        # `aggregate_trailing_mean` outside `src`. Without them the routing
        # would select tests that cannot fail on the change that selected them.
        "tests/alpha_research/test_dynamic_panel_portfolio_campaign.py",
        "tests/alpha_research/test_factor_alpha_campaign.py",
        "tests/portfolio_strategy_lab/test_strategy_scoring.py",
    ),
    "MODEL_KERNEL": (
        "tests/alpha_research/test_model_lifecycle.py",
        "tests/alpha_research/test_numerical_goldens.py",
        "tests/alpha_research/test_current_viability_and_refit.py",
        # `test_full_inventory_execution.py` stood here until `75225911` deleted
        # it. A selected path that is not on disk makes pytest exit 4, so every
        # MODEL_KERNEL change since has been failing the gate on a missing file
        # rather than on its own merits. These two cover what the domain owns
        # now: the installed adapter and the frozen product recipe.
        "tests/alpha_research/test_lightgbm_development_capability.py",
        "tests/alpha_research/test_alpha_product_recipe.py",
    ),
    "METRICS": (
        "tests/alpha_research/test_metrics.py",
        "tests/alpha_research/test_current_viability_and_refit.py",
    ),
    "QUALIFICATION_POLICY": (
        "tests/alpha_research/test_alpha_qualification.py",
        "tests/alpha_research/test_current_viability_and_refit.py",
    ),
    "ARTIFACT_RUNTIME": (
        "tests/alpha_research/test_alpha_qualification.py",
        # `impact.py` and `identity.py` are themselves ARTIFACT_RUNTIME owners,
        # so the domain should run the case that reads them. It is also the only
        # place the two routing maps are checked against each other.
        "tests/alpha_research/test_verification_impact.py",
        # Covers the replay reader's refusals and its composition.
        "tests/alpha_research/test_alpha_product_replay.py",
        "tests/portfolio_strategy_lab/test_heterogeneous_live_scoring.py",
        "tests/portfolio_strategy_lab/test_heterogeneous_strategy_product.py",
        "tests/portfolio_strategy_lab/test_frozen_strategy_packages.py",
        "tests/portfolio_strategy_lab/test_strategy_scoring.py",
        # Starts both closure runners under the workspace alone and proves their
        # preflight hashes research sources without executing them.
        "tests/alpha_research/test_live_closure_entry_points.py",
    ),
    "TEST_ONLY": (),
    "DOCUMENTATION": (),
    "OUT_OF_SCOPE": (),
    "UNKNOWN": (
        "tests/alpha_research/test_array_firewall.py",
        "tests/alpha_research/test_array_surface.py",
        "tests/alpha_research/test_numerical_goldens.py",
        "tests/alpha_research/test_metrics.py",
        "tests/alpha_research/test_alpha_qualification.py",
    ),
}


@dataclass(frozen=True, slots=True)
class AlphaVerificationImpact:
    """Describe changed-path verification domains, required tests and model reuse authority."""

    changed_paths: tuple[str, ...]
    domains: tuple[AlphaVerificationDomain, ...]
    required_test_paths: tuple[str, ...]
    full_numerical_parity_required: bool
    fit_children_reusable: bool
    current_refit_children_reusable: bool

    def __post_init__(self) -> None:
        """Require nonempty sorted unique changed paths/domains and a canonical test axis.

        Raises:
            ValueError: Changed paths or domains are absent/repeated/unordered, or required tests
                are repeated/unordered.
        """
        if (
            not self.changed_paths
            or not self.domains
            or tuple(sorted(set(self.changed_paths))) != self.changed_paths
            or tuple(sorted(set(self.domains))) != self.domains
            or tuple(sorted(set(self.required_test_paths))) != self.required_test_paths
        ):
            raise ValueError("alpha_research.verification_impact_invalid")

    @property
    def impact_hash(self) -> str:
        """Hash the complete declared verification-impact decision.

        Returns:
            Canonical identity of changed paths, domains, required tests and parity/reuse flags.
        """
        return cast(
            str,
            canonical_hash(
                {
                    "changed_paths": self.changed_paths,
                    "domains": self.domains,
                    "required_test_paths": self.required_test_paths,
                    "full_numerical_parity_required": self.full_numerical_parity_required,
                    "fit_children_reusable": self.fit_children_reusable,
                    "current_refit_children_reusable": self.current_refit_children_reusable,
                }
            ),
        )


def _normalized_path(value: str) -> str:
    path = value.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if not path or path.startswith("/") or ".." in Path(path).parts:
        raise ValueError("alpha_research.verification_path_invalid")
    return path


def alpha_verification_domain(path: str) -> AlphaVerificationDomain:
    """Route a normalized changed path to its declared Alpha verification domain.

    Known source owners use explicit domains. Unclassified paths within Alpha route UNKNOWN;
    unrelated paths route OUT_OF_SCOPE. This function classifies paths, not AST-equivalent prose
    changes.

    Args:
        path: Changed repository-relative path to normalize and classify.

    Returns:
        Declared domain for documentation, tests, mathematical owners or artifact/runtime changes.
    """
    value = _normalized_path(path)
    if value.endswith((".md", ".rst", ".txt")) or value.startswith(
        ("implemented-plans/", "experience/", "docs/")
    ):
        return "DOCUMENTATION"
    if value.startswith("case-study/") or value.startswith("tests/"):
        return "TEST_ONLY"
    if value in {
        "src/alphalattice/foundation/feature_engine/producers/base_materializer.py",
        "src/alphalattice/foundation/feature_engine/catalog/contracts.py",
        "src/alphalattice/foundation/feature_engine/producers/cross_section.py",
        "src/alphalattice/foundation/feature_engine/producers/preprocessing/robust_cross_section.py",
        "src/alphalattice/kernel/quant/cross_section.py",
        "src/alphalattice/foundation/feature_engine/runtime/service.py",
    } or value.startswith("src/alphalattice/foundation/feature_engine/resources/"):
        return "FOUNDATION_INPUT"
    if (
        value
        in {
            "src/alphalattice/investment/alpha_research/inputs/folds.py",
            "src/alphalattice/investment/alpha_research/inputs/surfaces.py",
            "src/alphalattice/foundation/feature_engine/panels/reader.py",
            "workspace/src/alphalattice/kernel/validation/splitting.py",
        }
        or value.startswith(
            (
                "src/alphalattice/foundation/causal_outcomes/execution/",
                # The scale owner and the canonical target both produce numbers that
                # reach the array surface. Routed explicitly, because the catch-all
                # below would classify them UNKNOWN, and UNKNOWN demands full
                # numerical parity across every domain -- a wider gate than either
                # earns, which trains people to ignore it.
                "src/alphalattice/investment/alpha_research/scaling/",
                # Sector Research produces numbers the same surface consumes, and
                # its paths are outside `alpha_research/`, so the catch-all below
                # cannot reach them: without this they would route OUT_OF_SCOPE
                # and the parity gate would quietly weaken after the move.
                "src/alphalattice/investment/sector_research/",
            )
        )
        or value
        in {
            "src/alphalattice/investment/alpha_research/targets/canonical.py",
            "src/alphalattice/investment/alpha_research/targets/standardization.py",
            "src/alphalattice/investment/alpha_research/inputs/frozen_price_volume.py",
            # Both score owners emit values that a declared method is compared
            # on. `temporal_aggregation` owns `aggregate_trailing_mean` and
            # `score_filters` owns the standardize/EWMA pair and calls that
            # aggregate; a change to either moves the numbers an evaluation
            # ranks. Routed here rather than left to the catch-all for the
            # reason stated above: UNKNOWN demands parity across every domain,
            # and a gate wider than it earns gets ignored.
            "src/alphalattice/investment/alpha_research/scores/score_filters.py",
            "src/alphalattice/investment/alpha_research/scores/temporal_aggregation.py",
        }
    ):
        return "ARRAY_VALUES"
    if value.startswith("src/alphalattice/capabilities/alpha_modeling/") or value in {
        "src/alphalattice/investment/alpha_research/scores/refit.py",
        "src/alphalattice/investment/alpha_research/scores/model_renewal.py",
        # One component's lifecycle admission, beside the renewal it prepares (V325).
        "src/alphalattice/investment/alpha_research/scores/lifecycle_preparation.py",
        "src/alphalattice/investment/alpha_research/targets/component_training.py",
        "src/alphalattice/investment/alpha_research/scores/product_lifecycle.py",
        "src/alphalattice/investment/alpha_research/experiments/execution.py",
        # The frozen product recipe pins the estimator point, seed set, vintage
        # set and training window, so changing it changes every fit it names --
        # the same blast radius as the refit owner beside it. Routed explicitly
        # for the reason stated above the array cases: the catch-all would call
        # it UNKNOWN, which demands full parity across every domain rather than
        # this one, and a gate that is wider than it earns gets ignored.
        "src/alphalattice/investment/alpha_research/scores/product_recipe.py",
    }:
        return "MODEL_KERNEL"
    if value in {
        "src/alphalattice/investment/alpha_research/evaluation/metrics.py",
        "src/alphalattice/investment/alpha_research/evaluation/viability.py",
    } or value.startswith("workspace/src/alphalattice/kernel/validation/"):
        return "METRICS"
    if value in {
        "src/alphalattice/investment/alpha_research/evaluation/stability.py",
        "src/alphalattice/investment/alpha_research/experiments/contracts.py",
        "src/alphalattice/investment/alpha_research/experiments/mandate.py",
        "src/alphalattice/investment/alpha_research/candidates/qualification.py",
        "src/alphalattice/investment/alpha_research/candidates/control.py",
        "src/alphalattice/investment/alpha_research/experiments/family_qualification.py",
        "src/alphalattice/investment/alpha_research/candidates/qualification_task.py",
    }:
        return "QUALIFICATION_POLICY"
    if value in {
        "src/alphalattice/investment/alpha_research/verification/identity.py",
        "src/alphalattice/investment/alpha_research/verification/impact.py",
        "src/alphalattice/investment/alpha_research/inputs/observations.py",
        "src/alphalattice/investment/alpha_research/publication/artifacts.py",
        "src/alphalattice/investment/alpha_research/candidates/artifacts.py",
        "src/alphalattice/investment/alpha_research/candidates/committer.py",
        # Readers and product contracts over sealed artifacts. `product_replay.py`
        # verifies the admitted IW184 package and composes it; `product_lifecycle.py`
        # decides fit-or-score from the frozen recipe and proves the live set is
        # whole. The heterogeneous pair seals and reopens the successor recipe
        # and historical package. None fits a model, and none of their values
        # reaches the Alpha array surface -- their scores reach the Portfolio
        # book -- so artifact/identity and book-consumer coverage exercises them.
        "src/alphalattice/investment/alpha_research/experiments/lifecycle_authoring.py",
        "src/alphalattice/investment/alpha_research/scores/heterogeneous_product.py",
        "src/alphalattice/investment/alpha_research/scores/heterogeneous_replay.py",
        "src/alphalattice/investment/alpha_research/scores/frozen_inference.py",
        "src/alphalattice/investment/alpha_research/scores/product_replay.py",
        "scripts/check_playpen.py",
        "scripts/check_alpha_verification_impact.py",
        # The live-closure runners produce the sealed Gate L manifest and Gate M
        # receipts, and the ported research closure plus this tree's monthly refit
        # driver are the code they execute in place of sibling-worktree modules.
        "scripts/broad_ensemble_research_closure.py",
        "scripts/run_broad_ensemble_live_model_closure.py",
        "scripts/run_heterogeneous_live_score_closure.py",
        "scripts/run_monthly_alpha_refit_research.py",
    }:
        return "ARTIFACT_RUNTIME"
    if value.startswith("src/alphalattice/investment/alpha_research/candidates/"):
        # The candidate registry's contracts, store and inputs (GR4); its qualification and
        # committer are routed above.
        return "ARTIFACT_RUNTIME"
    if value in {"pyproject.toml", "uv.lock"}:
        return "MODEL_KERNEL"
    if value.startswith(
        (
            "src/alphalattice/investment/alpha_research/",
            "src/alphalattice/capabilities/alpha_modeling/",
        )
    ):
        return "UNKNOWN"
    return "OUT_OF_SCOPE"


def alpha_verification_impact(changed_paths: tuple[str, ...]) -> AlphaVerificationImpact:
    """Derive required tests and parity/model-reuse flags from declared changed paths.

    Args:
        changed_paths: Explicit changed paths normalized, deduplicated and sorted.

    Returns:
        Validated impact; mathematical/unknown domains disallow model-child reuse and full-parity
        domains request numerical parity.

    Raises:
        ValueError: No normalized changed path remains.
    """
    normalized = tuple(sorted({_normalized_path(value) for value in changed_paths}))
    if not normalized:
        raise ValueError("alpha_research.verification_impact_empty")
    domains = tuple(sorted({alpha_verification_domain(value) for value in normalized}))
    tests = {test for domain in domains for test in _DOMAIN_TESTS[domain]}
    mathematical_change = bool(
        set(domains) & {"FOUNDATION_INPUT", "ARRAY_VALUES", "MODEL_KERNEL", "UNKNOWN"}
    )
    return AlphaVerificationImpact(
        changed_paths=normalized,
        domains=domains,
        required_test_paths=tuple(sorted(tests)),
        full_numerical_parity_required=bool(set(domains) & _FULL_PARITY_DOMAINS),
        fit_children_reusable=not mathematical_change,
        current_refit_children_reusable=not mathematical_change,
    )


__all__ = [
    "AlphaVerificationDomain",
    "AlphaVerificationImpact",
    "alpha_verification_domain",
    "alpha_verification_impact",
]
