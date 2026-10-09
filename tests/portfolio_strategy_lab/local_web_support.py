"""Shared harness of the Local Web and public Portfolio application suites.

The installed test package, the host-side resolver stand-ins, the HTTP helper and
the workspace manifest builder that eight suites drove through two test modules.
Test support beside its owner: not a framework, and nothing here is product
authority.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from alphalattice.control.product_host.composition.portfolio_application import (
    PortfolioResearchApplication,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
)
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument as PortfolioResearchAgentRequest,
)
from alphalattice.interface.local_application.web import (
    SESSION_COOKIE,
    SESSION_HEADER,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    OwnerCoverage,
    PortfolioResearchResult,
    PortfolioResearchSpec,
    PortfolioSupportCoverage,
)
from alphalattice.investment.portfolio_strategy_lab.application.executor import (
    ResolvedPortfolioExecution,
)
from alphalattice.investment.portfolio_strategy_lab.application.resolution import (
    PublicPortfolioReplayWorkspace,
    ResolvedPortfolioAuthorities,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    ComponentPlanEntry,
    FrozenStrategyPackage,
    PackageControlSurface,
    PackagePolicyIdentity,
    ScoreSourceCapability,
)
from alphalattice.investment.portfolio_strategy_lab.application.tranche_book_execution import (
    TrancheBookComponent,
    TrancheFormationInputs,
)
from alphalattice.investment.portfolio_strategy_lab.policies.catalog import (
    build_public_portfolio_policy_catalog,
)
from alphalattice.investment.portfolio_strategy_lab.policies.contracts import (
    CausalRankReturnCurveSlice,
)
from alphalattice.investment.portfolio_strategy_lab.policies.tranche_book import (
    TrancheBookRecipe,
)
from alphalattice.investment.risk_research.surfaces.decomposition import (
    INSTALLED_RISK_DECOMPOSITION_RECIPE,
    FactorIdiosyncraticRiskSurface,
    RiskAllocationProjection,
    RiskAttributionProjection,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

_HASH = "a" * 64

TEST_PACKAGE = FrozenStrategyPackage.create(
    strategy_id="SYNTHETIC_SINGLE_BOOK",
    alpha_recipe_hash="b" * 64,
    policy_binding="REQUEST_SELECTED",
    frozen_policy=None,
    component_plan=(
        ComponentPlanEntry(
            component_id="SYNTHETIC_SCORE",
            allocation_basis_points=10_000,
            family_id="SYNTHETIC",
            recipe_hash="b" * 64,
            target_recipe="SYNTHETIC_FORWARD_EXCESS",
            objective="ONE_SCORE_PER_FORMATION",
        ),
    ),
    merge_semantics="SINGLE_COMPONENT_BOOK",
    score_sources=(
        ScoreSourceCapability(
            mode="HISTORICAL_ARRAY_REPLAY",
            evidence_identity_hash="c" * 64,
            description="Synthetic recorded scores",
        ),
    ),
    default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
    required_shared_input_lanes=("MARKET_REALIZED_RETURN",),
    risk_disposition="POLICY_CONSUMED_RISK_ALLOCATION_PLUS_REPORT_ONLY_DFD_PRIME_PLUS_E",
    controls=PackageControlSurface(
        frozen=(),
        admitted=("top_k", "tranches", "exit_rank", "weight_rule"),
        shared=("cost_bps_per_side", "report_unit", "study_start", "study_end"),
    ),
    claim_limits=("NO_INDEPENDENT_HOLDOUT_CLAIM",),
    provenance=("SYNTHETIC_FIXTURE_SCORES",),
)


@dataclass(frozen=True, slots=True)
class InstalledAgent:
    """An operation as an installed Agent calls it: the caller stated by the Host's grant.

    The built-in agent's bridge went with AG2; the seam's caller is what these suites exercise.
    """

    operations: Any
    agent_execution: Any = None

    def invoke(self, request: Any) -> str:
        from alphalattice.interface.local_application.failure_codes import public_failure

        try:
            answer = self.operations.execute(
                request.to_operation_request(),
                caller="INSTALLED_AGENT",
                **(
                    {"agent_execution": self.agent_execution}
                    if request.operation.startswith("EXPERIMENT")
                    else {}
                ),
            )
        except (ValueError, KeyError) as error:
            answer = {"refused": public_failure(error, "local_application.request_refused")}
        return json.dumps(answer, default=str, sort_keys=True, separators=(",", ":"))


def _policy_identity(recipe: TrancheBookRecipe) -> tuple[str, str, str]:
    """The three policy identities the fixture's request compiles to."""

    catalog = build_public_portfolio_policy_catalog()
    binding = catalog.resolve(recipe).describe_adapter_binding()
    return recipe.recipe_hash, catalog.binding.catalog_hash, binding.binding_hash


@dataclass
class _Resolver:
    """A synthetic Host resolver that counts what each phase actually did.

    The counters are the point. `PLAN` claiming to perform no numerical work is
    a field on a contract; these make it a measurement, because the numerical
    half of resolution is only reachable through `resolve`.
    """

    resolved: ResolvedPortfolioExecution
    authority_calls: int = 0
    numerical_calls: int = 0
    score_rotation: int = 0
    """The cross-section's rotation the resolution was built with (`_resolved`), carried
    into the rebuild so the numerical read keeps the shape the admitted one has."""
    numerical: ResolvedPortfolioExecution | None = None
    """A prepared numerical resolution to hand back verbatim, instead of rebuilding.

    The default path rebuilds through `_resolved` so the double reflects the
    *selected* weight rule, which is what caught `iv1` against `ew`. But a caller
    comparing two runs over different axes needs the same session to produce the
    same numbers in both, and a rebuild re-derives index-keyed values -- so the
    comparison would be about the fixture rather than about the path.
    """

    @property
    def strategy_catalog_hash(self) -> str:
        return canonical_hash(
            {
                "kind": "QaInstalledStrategyCatalog",
                "packages": (TEST_PACKAGE.model_dump(mode="json"),),
            }
        )

    def resolve_authorities(
        self, *, workspace: Path, spec: PortfolioResearchSpec
    ) -> ResolvedPortfolioAuthorities:
        del workspace
        self.authority_calls += 1
        sessions = tuple(self.resolved.workspace.formation_sessions)
        listings = tuple(self.resolved.workspace.ordered_listing_ids)
        coverage = PortfolioSupportCoverage.create(
            owners=(
                OwnerCoverage.of(
                    owner_id="alpha_product_replay",
                    lane="BROAD_ENSEMBLE_DEVELOPMENT_REPLAY_SCORE",
                    sessions=sessions,
                    identity_hash=self.resolved.alpha_evidence_manifest_hash,
                ),
                OwnerCoverage.of(
                    owner_id="risk_return_surface",
                    lane="CAUSAL_OPEN_TO_OPEN_RETURN",
                    sessions=sessions,
                    identity_hash=self.resolved.risk_return_surface_hash,
                ),
            ),
            common_watermark_start=sessions[0],
            common_watermark_end=sessions[-1],
            common_session_count=len(sessions),
        )
        policy_recipe, policy_catalog, policy_binding = _policy_identity(
            TrancheBookRecipe.create(
                top_k=spec.top_k,
                tranches=spec.tranches,
                exit_rank=spec.exit_rank,
                weight_rule=spec.weight_rule,
            )
        )
        return ResolvedPortfolioAuthorities(
            coverage=coverage,
            candidate_sessions=sessions,
            ordered_listing_ids=listings,
            eligible_count=len(listings),
            strategy_package_id=TEST_PACKAGE.strategy_id,
            strategy_package_hash=TEST_PACKAGE.package_hash,
            score_source_mode="HISTORICAL_ARRAY_REPLAY",
            policy_identity=PackagePolicyIdentity(
                policy_recipe_hash=policy_recipe,
                policy_catalog_hash=policy_catalog,
                policy_adapter_binding_hash=policy_binding,
            ),
            alpha_recipe_hash=self.resolved.alpha_recipe_hash,
            alpha_evidence_manifest_hash=self.resolved.alpha_evidence_manifest_hash,
            risk_recipe_hash=self.resolved.risk_recipe_hash,
            risk_return_surface_hash=self.resolved.risk_return_surface_hash,
            sector_map_hash=self.resolved.sector_map_hash,
            tradability_decision_hash=self.resolved.tradability_decision_hash,
            execution_outcome_manifest_hash=self.resolved.execution_outcome_manifest_hash,
        )

    def installed_packages(self) -> dict[str, FrozenStrategyPackage]:
        return {TEST_PACKAGE.package_hash: TEST_PACKAGE}

    def resolve(
        self, *, workspace: Path, spec: PortfolioResearchSpec
    ) -> ResolvedPortfolioExecution:
        del workspace
        self.numerical_calls += 1
        if self.numerical is not None:
            if spec.weight_rule != "mu.iv1" or spec.secondary_benchmark_view != "anchor_only":
                raise AssertionError(
                    "a prepared resolution was supplied for a rule it was not built for"
                )
            return self.numerical
        # The real Host resolves lanes against the selected rule, so the double
        # has to as well: a fixture that always supplied every lane would hide
        # exactly the defect that `iv1` and `ew` exposed.
        return _resolved(
            spec.weight_rule,
            spec.secondary_benchmark_view,
            self.resolved.tradability_decision_hash,
            self.resolved.execution_outcome_manifest_hash,
            top_k=spec.top_k,
            tranches=spec.tranches,
            exit_rank=spec.exit_rank,
            # Both axes come from the resolution this double was built with,
            # not from the fixture defaults: the Host checks that the numerical
            # read matches the admitted one, so a double that always widened to
            # 80 names on the development calendar could only ever be used at
            # one shape.
            listing_count=len(self.resolved.workspace.ordered_listing_ids),
            sessions=tuple(self.resolved.workspace.formation_sessions),
            score_rotation=self.score_rotation,
        )


@dataclass
class _InterruptsOnce:
    """A resolver that fails the first execution, the way a crash would.

    Recovery cannot be tested against a task that already succeeded -- resuming
    a finished workspace resumes nothing, and the assertions pass without the
    code under test ever running. So the first numerical resolution raises, Task
    Control marks the task `RECOVERY_REQUIRED` from its own failure path, and the
    restart has something real to pick up.
    """

    inner: _Resolver
    failures: int = 0
    interruptions: int = 1
    """How many resolutions raise before one succeeds: two lets a recovery be interrupted
    again, which is how a Task moves to a second interrupted version."""

    def resolve_authorities(self, **kwargs: Any) -> Any:
        return self.inner.resolve_authorities(**kwargs)

    @property
    def strategy_catalog_hash(self) -> str:
        return self.inner.strategy_catalog_hash

    def installed_packages(self) -> Any:
        return self.inner.installed_packages()

    def resolve(self, **kwargs: Any) -> Any:
        if self.failures < self.interruptions:
            self.failures += 1
            raise RuntimeError("qa.interrupted_mid_execution")
        return self.inner.resolve(**kwargs)


def _formation_scores(count: int, formation_index: int, block: int) -> np.ndarray:
    """The cross-section of one formation: static (block 0), or rotated so that block
    ``formation_index`` ranks first and every earlier block ranks one block lower."""

    base = np.linspace(1.0, 0.0, count, dtype=np.float64)
    if block <= 0:
        return base
    positions = np.arange(count)
    rank = ((formation_index - positions // block) * block + positions % block) % count
    return base[rank]


def _resolved(
    weight_rule: str = "mu.iv1",
    benchmark_view: str = "anchor_only",
    tradability_hash: str = "f" * 64,
    outcome_hash: str = "1" * 64,
    listing_count: int = 80,
    sessions: tuple[date, ...] | None = None,
    *,
    top_k: int = 35,
    tranches: int = 3,
    exit_rank: int | None = None,
    score_rotation: int = 0,
) -> ResolvedPortfolioExecution:
    """Mirror the Host: supply exactly the lanes the selected request reads.

    ``score_rotation`` (a block of listings) moves the cross-section's top block one block
    forward at every formation session, the earlier blocks ranking just under it in
    formation order: each sleeve then holds a fresh block and the book at the window end is
    the union of the live sleeves, as a real tranche book is. Zero keeps the static
    cross-section every existing caller reads.

    ``listing_count`` and ``sessions`` are parameters rather than constants
    because both axes are facts about a workspace rather than properties of the
    path: a fixture that could only build one shape would let a hard-coded
    cardinality survive anywhere downstream of it, and a protected continuation
    needs an axis that starts after the development boundary.
    """

    recipe = TrancheBookRecipe.create(
        top_k=top_k,
        tranches=tranches,
        exit_rank=exit_rank,
        weight_rule=weight_rule,  # type: ignore[arg-type]
    )
    # The whole holdings request, not just the weight rule: the resolution now
    # carries the policy identity the Program binds, so a fixture that always
    # built the default book would refuse every non-default control.
    # Two calendar months by default, on purpose: a single-month axis would
    # collapse every report unit to one row and hide the bucketing the units are
    # for.
    sessions = sessions or (date(2024, 1, 2), date(2024, 1, 3), date(2024, 2, 1))
    listings = tuple(f"listing-{index:04d}" for index in range(listing_count))
    exposures = np.ones((len(listings), 1), dtype=np.float64)
    formations: list[TrancheFormationInputs] = []
    for index, session in enumerate(sessions):
        surface = FactorIdiosyncraticRiskSurface.create(
            recipe=INSTALLED_RISK_DECOMPOSITION_RECIPE,
            formation_session=session,
            ordered_listing_ids=listings,
            ordered_factor_ids=("industry",),
            exposures=exposures,
            factor_covariance=np.array([[0.01]], dtype=np.float64),
            idiosyncratic_variance=np.full(len(listings), 0.01, dtype=np.float64),
            conditional_volatility=np.full(len(listings), 0.20, dtype=np.float64),
            producer_identity=f"synthetic-{index}",
        )
        formations.append(
            TrancheFormationInputs(
                formation_session=session,
                scores=_formation_scores(len(listings), index, score_rotation),
                decision_eligible=np.ones(len(listings), dtype=np.bool_),
                risk_allocation=(
                    RiskAllocationProjection.of(surface) if recipe.consumes_risk else None
                ),
                risk_attribution=RiskAttributionProjection.of(
                    surface,
                    classification_authority="TEST_FOUNDATION_CLASSIFICATION",
                ),
                causal_rank_return_curve=None
                if not recipe.consumes_mu
                else CausalRankReturnCurveSlice(
                    formation_index=index,
                    formation_session=session,
                    bucket_means=np.linspace(0.02, -0.01, 20, dtype=np.float64),
                    bucket_support_counts=(252,) * 20,
                    admitted_formation_count=252,
                    disposition="AVAILABLE",
                    curve_hash=_HASH,
                ),
            )
        )
    # A moving cross-section, because a flat one makes the equal-weight anchor
    # constant and the Backtesting beta owner rightly refuses it -- which would
    # exercise the degenerate branch instead of the real one.
    realized = np.linspace(-0.02, 0.02, len(sessions) * len(listings), dtype=np.float64).reshape(
        len(sessions), len(listings)
    )
    offsets = np.array([[0.004], [-0.006], [0.009]], dtype=np.float64)[: len(sessions)]
    if len(sessions) > 3:
        offsets = np.resize(offsets, (len(sessions), 1))
    realized = realized + offsets
    workspace = PublicPortfolioReplayWorkspace(
        formation_sessions=sessions,
        ordered_listing_ids=listings,
        execution_available=np.ones(realized.shape, dtype=np.bool_),
        realized_simple_returns=realized,
        causal_adv20=np.full(realized.shape, 1_000_000.0, dtype=np.float64),
        sector_exposure_matrix=np.ones((1, len(listings)), dtype=np.float64),
        equal_weight_sector_exposure=np.ones(1, dtype=np.float64),
        passive_returns_by_session={
            session: realized[index] for index, session in enumerate(sessions)
        },
    )
    wants_spy = benchmark_view == "anchor_plus_spy"
    policy_recipe, policy_catalog, policy_binding = _policy_identity(recipe)
    return ResolvedPortfolioExecution(
        secondary_benchmark_id="SPY_TOTAL_RETURN" if wants_spy else None,
        secondary_benchmark_returns=tuple(0.0 for _ in sessions) if wants_spy else None,
        secondary_benchmark_disposition=(
            "VISIBLY_CONTAMINATED_SECONDARY_COMPARATOR_NEVER_A_REPLACEMENT"
            if wants_spy
            else "SECONDARY_BENCHMARK_NOT_REQUESTED"
        ),
        workspace=workspace,
        components=(
            TrancheBookComponent(
                component_id="SYNTHETIC_SCORE",
                allocation_basis_points=10_000,
                recipe=recipe,
                formations=tuple(formations),
                ordered_listing_ids=listings,
                sector_exposure_matrix=workspace.sector_exposure_matrix,
                equal_weight_sector_exposure=workspace.equal_weight_sector_exposure,
            ),
        ),
        score_receipt_hashes=tuple(
            canonical_hash({"formation": value.isoformat(), "fixture": "synthetic"})
            for value in sessions
        ),
        report_risk_attributions=tuple(
            value.risk_attribution for value in formations if value.risk_attribution is not None
        ),
        strategy_package_hash=TEST_PACKAGE.package_hash,
        score_source_mode="HISTORICAL_ARRAY_REPLAY",
        policy_recipe_hash=policy_recipe,
        policy_catalog_hash=policy_catalog,
        policy_adapter_binding_hash=policy_binding,
        alpha_recipe_hash="b" * 64,
        alpha_evidence_manifest_hash="c" * 64,
        risk_recipe_hash=INSTALLED_RISK_DECOMPOSITION_RECIPE.recipe_hash,
        risk_return_surface_hash="d" * 64,
        sector_map_hash="e" * 64,
        tradability_decision_hash=tradability_hash,
        execution_outcome_manifest_hash=outcome_hash,
    )


def _session_keyed_resolved(
    sessions: tuple[date, ...], *, listing_count: int = 80
) -> ResolvedPortfolioExecution:
    """The same fixture, with every value a function of the session, not its index.

    `_resolved` above builds scores, returns and producer identities from a
    formation's *position* on the axis it happens to be on. That is fine for a
    single path and useless for comparing two: split the same calendar into a
    prefix and a suffix and the suffix gets different numbers purely because it
    now starts at index zero, so "the same inputs" cannot be said at all.

    Here a session determines its own row. A path over
    `(A, B, C, D)` and a continuation over `(C, D)` see identical scores,
    identical realised returns and identical Risk projections on C and D, which
    is what makes an exact suffix comparison a statement about the *policy*
    rather than about the fixture.
    """

    recipe = TrancheBookRecipe.create(weight_rule="mu.iv1")
    listings = tuple(f"listing-{index:04d}" for index in range(listing_count))
    exposures = np.ones((len(listings), 1), dtype=np.float64)
    formations: list[TrancheFormationInputs] = []
    rows: list[npt.NDArray[np.float64]] = []
    for index, session in enumerate(sessions):
        # One deterministic seed per session, so the same date is the same
        # cross-section wherever it appears.
        seed = session.toordinal()
        rotation = seed % len(listings)
        scores = np.roll(np.linspace(1.0, 0.0, len(listings), dtype=np.float64), rotation)
        surface = FactorIdiosyncraticRiskSurface.create(
            recipe=INSTALLED_RISK_DECOMPOSITION_RECIPE,
            formation_session=session,
            ordered_listing_ids=listings,
            ordered_factor_ids=("industry",),
            exposures=exposures,
            factor_covariance=np.array([[0.01]], dtype=np.float64),
            idiosyncratic_variance=np.full(len(listings), 0.01, dtype=np.float64),
            conditional_volatility=np.full(len(listings), 0.20, dtype=np.float64),
            producer_identity=f"synthetic-{session.isoformat()}",
        )
        formations.append(
            TrancheFormationInputs(
                formation_session=session,
                scores=scores,
                decision_eligible=np.ones(len(listings), dtype=np.bool_),
                risk_allocation=RiskAllocationProjection.of(surface),
                risk_attribution=RiskAttributionProjection.of(
                    surface,
                    classification_authority="TEST_FOUNDATION_CLASSIFICATION",
                ),
                # The curve slice is checked against the *local* index, because
                # it is one row of this segment's own arrays.
                causal_rank_return_curve=CausalRankReturnCurveSlice(
                    formation_index=index,
                    formation_session=session,
                    bucket_means=np.linspace(0.02, -0.01, 20, dtype=np.float64),
                    bucket_support_counts=(252,) * 20,
                    admitted_formation_count=252,
                    disposition="AVAILABLE",
                    curve_hash=_HASH,
                ),
            )
        )
        base = np.roll(
            np.linspace(-0.02, 0.02, len(listings), dtype=np.float64), (seed * 7) % len(listings)
        )
        rows.append(base + ((seed % 11) - 5) * 0.001)
    assert recipe.consumes_mu and recipe.consumes_risk
    realized = np.ascontiguousarray(np.vstack(rows), dtype=np.float64)
    workspace = PublicPortfolioReplayWorkspace(
        formation_sessions=sessions,
        ordered_listing_ids=listings,
        execution_available=np.ones(realized.shape, dtype=np.bool_),
        realized_simple_returns=realized,
        causal_adv20=np.full(realized.shape, 1_000_000.0, dtype=np.float64),
        sector_exposure_matrix=np.ones((1, len(listings)), dtype=np.float64),
        equal_weight_sector_exposure=np.ones(1, dtype=np.float64),
        passive_returns_by_session={
            session: realized[index] for index, session in enumerate(sessions)
        },
    )
    policy_recipe, policy_catalog, policy_binding = _policy_identity(recipe)
    return ResolvedPortfolioExecution(
        secondary_benchmark_id=None,
        secondary_benchmark_returns=None,
        secondary_benchmark_disposition="SECONDARY_BENCHMARK_NOT_REQUESTED",
        workspace=workspace,
        components=(
            TrancheBookComponent(
                component_id="SYNTHETIC_SCORE",
                allocation_basis_points=10_000,
                recipe=recipe,
                formations=tuple(formations),
                ordered_listing_ids=listings,
                sector_exposure_matrix=workspace.sector_exposure_matrix,
                equal_weight_sector_exposure=workspace.equal_weight_sector_exposure,
            ),
        ),
        score_receipt_hashes=tuple(
            canonical_hash({"formation": value.isoformat(), "fixture": "session-keyed"})
            for value in sessions
        ),
        report_risk_attributions=tuple(
            value.risk_attribution for value in formations if value.risk_attribution is not None
        ),
        strategy_package_hash=TEST_PACKAGE.package_hash,
        score_source_mode="HISTORICAL_ARRAY_REPLAY",
        policy_recipe_hash=policy_recipe,
        policy_catalog_hash=policy_catalog,
        policy_adapter_binding_hash=policy_binding,
        alpha_recipe_hash="b" * 64,
        alpha_evidence_manifest_hash="c" * 64,
        risk_recipe_hash=INSTALLED_RISK_DECOMPOSITION_RECIPE.recipe_hash,
        risk_return_surface_hash="d" * 64,
        sector_map_hash="e" * 64,
        tradability_decision_hash="f" * 64,
        execution_outcome_manifest_hash="1" * 64,
    )


@dataclass
class _Harness:
    """One workspace, one session, one application, and the resolver's counters."""

    workspace: Path
    session: WorkspaceApplicationSession
    application: PortfolioResearchApplication
    resolver: _Resolver


@contextmanager
def _harness(tmp_path: Path) -> Iterator[_Harness]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    resolver = _Resolver(_resolved())
    with WorkspaceApplicationSession.acquire(workspace) as session:
        yield _Harness(
            workspace=workspace,
            session=session,
            application=PortfolioResearchApplication(
                workspace_id="synthetic",
                workspace=workspace,
                manifest_binding=lambda: _HASH,
                session=session,
                resolver=resolver,
            ),
            resolver=resolver,
        )


def _run(harness: _Harness, spec: PortfolioResearchSpec) -> PortfolioResearchResult:
    return harness.application.run(spec=spec).result


def _manifest(workspace_id: str) -> ResearchWorkspaceManifest:
    return ResearchWorkspaceManifest.create(
        workspace_id=workspace_id,
        default_strategy_package_id=TEST_PACKAGE.strategy_id,
        default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
        strategy_artifacts=(),
    )


class _ReadRedirects(urllib.request.HTTPRedirectHandler):
    """The launch URL's 303 is read, not followed: its cookie is the evidence."""

    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_ReadRedirects)


def _request(
    session: LocalPortfolioWebSession,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    host: str | None = None,
    origin: str | None = None,
    omit_origin: bool = False,
    content_type: str = "application/json",
    token: str | None = "valid",
    cookie: str | None = "valid",
    timeout: float = 30.0,
) -> tuple[int, dict[str, str], bytes]:
    """One raw HTTP round trip, with every header this service checks exposed."""

    application = session.web.application  # type: ignore[union-attr]
    base = session.url.rstrip("/")
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(f"{base}{path}", data=body, method=method)
    request.add_header("Host", host or f"127.0.0.1:{session.web.bound_port}")  # type: ignore[union-attr]
    if origin is not None:
        request.add_header("Origin", origin)
    elif method == "POST" and not omit_origin:
        request.add_header("Origin", f"http://127.0.0.1:{session.web.bound_port}")  # type: ignore[union-attr]
    if body is not None:
        request.add_header("Content-Type", content_type)
    resolved_token = application.session_token if token == "valid" else token
    resolved_cookie = application.session_token if cookie == "valid" else cookie
    if resolved_token is not None:
        request.add_header(SESSION_HEADER, resolved_token)
    if resolved_cookie is not None:
        request.add_header("Cookie", f"{SESSION_COOKIE}={resolved_cookie}")
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def _json(session: LocalPortfolioWebSession, path: str, **kwargs: Any) -> dict[str, Any]:
    status, _headers, body = _request(session, path, **kwargs)
    assert status == 200, (status, body[:400])
    return json.loads(body)


def run_node(
    arguments: list[str] | None,
    *,
    missing: str = "Node.js development runtime required",
    required: bool = False,
    **kwargs: Any,
) -> subprocess.CompletedProcess[Any] | None:
    """Run one Node invocation with its caller's availability policy and process options."""
    node = shutil.which("node")
    if node is None:
        if required:
            raise AssertionError(missing)
        pytest.skip(missing)
    if arguments is None:
        return None
    return subprocess.run([node, *arguments], **kwargs)


def _run_badge_browser(
    live: LocalPortfolioWebSession, out: Path, mode: str, ids: dict[str, str]
) -> None:
    """The BADGE owner records through one live built Workbench at 900 px."""
    root = Path(__file__).resolve().parents[2]
    run_node(
        None,
        missing="BADGE requires Node.js and the pinned Playwright browser runtime",
        required=True,
    )
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / "badge-records.json"
    receipt.write_text(json.dumps({**ids, "out": str(out)}), encoding="utf-8", newline="\n")
    completed = run_node(
        [
            str(Path(__file__).with_name("workbench_badge.cjs")),
            live.url,
            live.launch_url,
            f"--mode={mode}",
            str(receipt),
        ],
        missing="BADGE requires Node.js and the pinned Playwright browser runtime",
        required=True,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        env={
            **os.environ,
            "ALPHALATTICE_NETWORK_DISABLED": "1",
            "NODE_PATH": str(root / "third_party/playwright/node_modules"),
            "PLAYWRIGHT_BROWSERS_PATH": str(root / "third_party/playwright/.browsers"),
        },
    )
    diagnostics = (completed.stdout + completed.stderr).replace(live.launch_url, "<launch URL>")
    (out / "browser.log").write_text(diagnostics, encoding="utf-8", newline="\n")
    assert completed.returncode == 0, diagnostics
    assert f"workbench badge {mode} complete" in completed.stdout, diagnostics


def _walk_badge_records(
    workspace: Path, out: Path, source_task_id: str | None, successor_task_id: str | None
) -> None:
    """Walk an existing prepared workspace after its producer Host has stopped."""
    assert source_task_id and successor_task_id
    live = LocalPortfolioWebSession.from_workspace(workspace)
    live.start()
    try:
        _run_badge_browser(
            live, out, "pair", {"source": source_task_id, "successor": successor_task_id}
        )
    finally:
        live.stop()


def _agent(
    session: LocalPortfolioWebSession, request: PortfolioResearchAgentRequest
) -> dict[str, Any]:
    assert session.operations is not None
    bridge = InstalledAgent(session.operations)
    return json.loads(bridge.invoke(request))


def _raw(session: LocalPortfolioWebSession, request: bytes) -> tuple[int, bytes]:
    """One handcrafted request. `urllib` cannot lie about `Content-Length`."""
    port = session.web.bound_port
    connection = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    try:
        connection.sendall(request)
        chunks: list[bytes] = []
        while True:
            try:
                received = connection.recv(4096)
            except TimeoutError:
                break
            if not received:
                break
            chunks.append(received)
        payload = b"".join(chunks)
    finally:
        connection.close()
    status = int(payload.split(b" ", 2)[1]) if payload.startswith(b"HTTP/") else 0
    return (status, payload)


def _run_to_completion(session: LocalPortfolioWebSession) -> str:
    """Admit one background run and wait for the task, not for the request."""
    admitted = _json(session, "/api/run", method="POST", payload={})
    if admitted["disposition"] == "REUSED_EXACT":
        assert admitted["task_id"] is None
        return str(admitted["result_hash"])
    assert admitted["disposition"] == "ADMITTED"
    session.dispatcher.drain_for_tests()
    status = _json(session, f"/api/status?task_id={admitted['task_id']}")
    assert status["lifecycle"] == "SUCCEEDED", status
    results = _json(session, "/api/results")["results"]
    assert results
    return str(results[0]["result_hash"])
