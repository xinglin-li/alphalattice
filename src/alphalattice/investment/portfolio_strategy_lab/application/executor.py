"""One simplified public Portfolio coordinator over shared Backtesting."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import numpy.typing as npt

from alphalattice.capabilities.portfolio_backtesting.active_metrics import (
    ActiveMetricsError,
    active_path_metrics,
)
from alphalattice.capabilities.portfolio_backtesting.contracts import (
    PortfolioBacktestWorkspace,
    PortfolioCostPolicy,
    PortfolioWalkForwardState,
)
from alphalattice.capabilities.portfolio_backtesting.reference_marks import ReferenceMarkLane
from alphalattice.capabilities.portfolio_backtesting.segments import (
    run_portfolio_walk_forward_segment,
)
from alphalattice.capabilities.portfolio_backtesting.state import (
    initial_portfolio_walk_forward_state,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioBenchmarkComparison,
    PortfolioDeclaredPathReport,
    PortfolioEconomicLedger,
    PortfolioExecutionLedger,
    PortfolioExecutionProgram,
    PortfolioResearchResult,
    PortfolioResearchSpec,
    PortfolioSupportCoverage,
    SealedPortfolioBoundaryState,
)
from alphalattice.investment.portfolio_strategy_lab.application.report_projection import (
    project_declared_path_report,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    FrozenStrategyPackage,
    ScoreSourceMode,
    StrategyDisclosure,
)
from alphalattice.investment.portfolio_strategy_lab.application.tranche_book_execution import (
    ComponentBookFactory,
    open_component_book,
)
from alphalattice.investment.portfolio_strategy_lab.publication.portfolio_ledger import (
    PortfolioLedgerStore,
)
from alphalattice.investment.portfolio_strategy_lab.reporting.risk import (
    PortfolioRiskAttributionFacts,
)
from alphalattice.investment.portfolio_strategy_lab.reporting.static import (
    PortfolioReportContext,
    render_portfolio_research_html,
)
from alphalattice.investment.risk_research.surfaces.decomposition import (
    RiskAttributionProjection,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


@dataclass(frozen=True, slots=True)
class ResolvedPortfolioExecution:
    """All numerical inputs resolved before the executor begins.

    The secondary benchmark is resolved by the Host, not here: it is an authority
    read against the workspace, and the executor's job is to consume resolved
    lanes rather than to know where any of them live.
    """

    workspace: PortfolioBacktestWorkspace
    components: tuple[ComponentBookFactory, ...]
    """The selected package's component books, in the order it declared them.

    One entry for a single-book strategy, four for the successor. The executor
    opens them and merges what the plan says to merge; it never asks which
    strategy produced them.
    """

    score_receipt_hashes: tuple[str, ...]
    """One receipt per formation over every component row the scores came from."""

    report_risk_attributions: tuple[RiskAttributionProjection, ...]
    strategy_package_hash: str
    score_source_mode: ScoreSourceMode
    policy_recipe_hash: str
    policy_catalog_hash: str
    policy_adapter_binding_hash: str
    alpha_recipe_hash: str
    alpha_evidence_manifest_hash: str
    risk_recipe_hash: str
    risk_return_surface_hash: str
    sector_map_hash: str
    tradability_decision_hash: str
    execution_outcome_manifest_hash: str
    """Required, not defaulted. A resolution that cannot name the authorities it
    consumed cannot compile a program that binds them."""

    secondary_benchmark_id: str | None = None
    secondary_benchmark_returns: tuple[float, ...] | None = None
    secondary_benchmark_disposition: str = "SECONDARY_BENCHMARK_NOT_REQUESTED"
    reference_mark: ReferenceMarkLane | None = None
    sector_history_hash: str | None = None
    """The Sector history's identity while a reclassification is in force."""


@dataclass(frozen=True, slots=True)
class PortfolioContinuationCarry:
    """A sealed boundary rehydrated into the state the walk actually needs.

    Two halves that have to travel together: the identity the Program binds, and
    the arrays that identity names. Passing the arrays alone would let a caller
    continue from a book no Program can be shown to have declared.
    """

    boundary: SealedPortfolioBoundaryState
    state: PortfolioWalkForwardState
    sleeve_weights: npt.NDArray[np.float64] | None


class PortfolioResearchExecutor:
    """Run/reuse one program; owns coordination, no formula or task lifecycle."""

    def __init__(
        self,
        store: PortfolioLedgerStore,
        *,
        packages: Mapping[str, FrozenStrategyPackage] | None = None,
        report_context: Callable[[PortfolioExecutionProgram, date], PortfolioReportContext]
        | None = None,
    ) -> None:
        """Bind execution/readback to the ledger owner and installed package disclosures.

        Stored package disclosures remain available to cost-only readback, which rebuilds a report
        without resolving a new numerical program.

        Args:
            store: Current portfolio ledger/economic/result persistence owner.
            packages: Optional installed package declarations keyed by package hash.
            report_context: Optional deterministic program/window-end reporting context resolver.
        """
        self.store = store
        self.packages = dict(packages or {})
        self.report_context = report_context
        """Installed package declarations, keyed by package hash.

        The executor looks a disclosure up rather than being told one, because a
        cost-only reopen rebuilds the report from a stored ledger without
        resolving anything -- and that report has to say the same thing about the
        strategy as the run that produced it.
        """

    def disclosure(
        self, *, report: PortfolioDeclaredPathReport, program: PortfolioExecutionProgram
    ) -> StrategyDisclosure | None:
        """What this report says about the strategy behind it, or nothing.

        `None` exactly for a Program sealed before packages were installed. An
        unknown package hash is a refusal rather than a blank section: a report
        that quietly omitted its strategy would read as if it had none.
        """
        if program.strategy_package_hash is None or program.score_source_mode is None:
            return None
        package = self.packages.get(program.strategy_package_hash)
        if package is None:
            raise ValueError("portfolio_application.program_package_not_installed")
        return StrategyDisclosure.compile(
            package=package,
            mode=program.score_source_mode,
            report_hash=report.report_hash,
            program_hash=program.program_hash,
        )

    def execute(
        self,
        *,
        workspace_id: str,
        spec: PortfolioResearchSpec,
        program: PortfolioExecutionProgram,
        resolved: Callable[[], ResolvedPortfolioExecution],
        coverage: PortfolioSupportCoverage,
        authorities_hash: str,
        continuation: PortfolioContinuationCarry | None = None,
    ) -> PortfolioResearchResult:
        """Run or reopen one request, optionally continuing a sealed book.

        ``resolved`` is a thunk rather than a value so that an exact hit costs no
        numerical work. Evaluating it at the call site would resolve every score
        and Risk surface and then discard them, which is the opposite of what
        exact reuse is for.

        ``continuation`` is the opening state, and the Program has to agree with
        it: the boundary identity is one of the Program's bindings, so a carry
        that does not match the Program the caller compiled is refused here
        rather than quietly producing a path nobody can name.
        """
        if program.authorities_hash != authorities_hash:
            raise ValueError("portfolio_application.program_authorities_mismatch")
        if not program.replayable:
            # Refused here, at the reuse owner, and before either index is
            # queried or the resolution thunk is evaluated. A Program that cannot
            # name the inputs its numbers came from cannot authorize *reuse* of
            # those numbers: reuse is the claim that a stored answer is the
            # answer to this question, and there is nothing to check it against.
            # Reading such a path back stays available; standing on it does not.
            raise ValueError("portfolio_application.program_assembly_absent_no_reuse")
        carried = None if continuation is None else continuation.boundary.boundary_hash
        if program.continued_from_state_hash != carried:
            raise ValueError("portfolio_application.program_continuation_mismatch")
        existing = self.store.find_result_for(
            program_hash=program.program_hash, spec_hash=spec.spec_hash
        )
        if existing is not None:
            return cast(
                PortfolioResearchResult,
                PortfolioResearchResult.model_validate(
                    existing.model_copy(update={"action": "REUSED_EXACT"})
                ),
            )

        ledger = self.store.find_execution_for_program(program.program_hash)
        if ledger is None:
            ledger = self._materialize(
                spec=spec, program=program, resolved=resolved, continuation=continuation
            )
        return self._publish_descendants(
            workspace_id=workspace_id,
            spec=spec,
            program=program,
            ledger=ledger,
            resolved=resolved,
            coverage=coverage,
            authorities_hash=authorities_hash,
        )

    def _materialize(
        self,
        *,
        spec: PortfolioResearchSpec,
        program: PortfolioExecutionProgram,
        resolved: Callable[[], ResolvedPortfolioExecution],
        continuation: PortfolioContinuationCarry | None = None,
    ) -> PortfolioExecutionLedger:
        """Walk the book forward once and publish the continuous path.

        Reached only when this exact holdings configuration has never been run.
        Everything it stores is program-determined, so every descendant control
        can be answered later without entering this method again.
        """

        resolution = resolved()
        if (
            resolution.alpha_recipe_hash != program.alpha_recipe_hash
            or resolution.alpha_evidence_manifest_hash != program.alpha_evidence_manifest_hash
            or resolution.risk_recipe_hash != program.risk_recipe_hash
            or resolution.risk_return_surface_hash != program.risk_return_surface_hash
            or resolution.sector_map_hash != program.sector_map_hash
            or resolution.tradability_decision_hash != program.tradability_decision_hash
            or resolution.execution_outcome_manifest_hash != program.execution_outcome_manifest_hash
            or resolution.strategy_package_hash != program.strategy_package_hash
            or resolution.score_source_mode != program.score_source_mode
        ):
            raise ValueError("portfolio_application.program_resolved_authority_mismatch")
        if (
            resolution.policy_recipe_hash != program.policy_recipe_hash
            or resolution.policy_catalog_hash != program.policy_catalog_hash
            or resolution.policy_adapter_binding_hash != program.policy_adapter_binding_hash
        ):
            # One check for every package, instead of re-deriving the predecessor's
            # recipe here and exempting anything that did not look like it.
            raise ValueError("portfolio_application.program_policy_mismatch")
        workspace = resolution.workspace
        sessions = tuple(workspace.formation_sessions)
        listings = tuple(workspace.ordered_listing_ids)
        if (
            len(sessions) != program.formation_count
            or len(listings) != program.listing_count
            or sessions[0] != program.formation_start
            or sessions[-1] != program.formation_end
            or canonical_hash(tuple(value.isoformat() for value in sessions))
            != program.formation_sessions_hash
            or canonical_hash(listings) != program.ordered_listing_ids_hash
        ):
            raise ValueError("portfolio_application.program_workspace_axis_mismatch")
        if len(resolution.score_receipt_hashes) != len(sessions):
            raise ValueError("portfolio_application.score_receipt_axis_mismatch")
        if continuation is not None and continuation.boundary.listing_count != len(listings):
            # The state vectors are indexed by the listing axis. Carrying them
            # onto a different axis would silently re-assign every position.
            raise ValueError("portfolio_application.continuation_listing_axis_mismatch")
        schedule_offset = (
            0 if continuation is None else continuation.boundary.decided_formation_count
        )
        # The schedule resumes where the sealed boundary left it. Passing the
        # local start index here instead would restage the whole book on the
        # first continued formation.
        provider = open_component_book(
            resolution.components,
            initial_sleeve_weights=(None if continuation is None else continuation.sleeve_weights),
            schedule_offset=schedule_offset,
            formation_sessions=sessions,
        )
        opening = (
            initial_portfolio_walk_forward_state(len(listings))
            if continuation is None
            else continuation.state
        )
        segment = run_portfolio_walk_forward_segment(
            workspace=workspace,
            decision_provider=provider,
            start_index=0,
            stop_index=len(sessions),
            initial_state=opening,
            reference_mark=resolution.reference_mark,
        )
        weights = np.ascontiguousarray(segment.executed_weights, dtype="<f8")
        uncapped_weights = np.ascontiguousarray(
            np.vstack(provider.uncapped_targets_by_formation), dtype="<f8"
        )
        if uncapped_weights.shape != weights.shape:
            raise ValueError("portfolio_application.cap_replay_axis_mismatch")
        risk_attributions = resolution.report_risk_attributions
        if len(risk_attributions) != weights.shape[0]:
            raise ValueError("portfolio_application.risk_report_axis_mismatch")
        risk_facts = tuple(
            PortfolioRiskAttributionFacts.create(
                projection=projection,
                weights=weights[index],
            )
            for index, projection in enumerate(risk_attributions)
        )
        final_weights = np.ascontiguousarray(segment.final_state.optimizer_reference, dtype="<f8")
        adv_rows: npt.NDArray[np.float64] = np.asarray(workspace.causal_adv20, dtype=np.float64)
        target_weights = np.ascontiguousarray(
            np.vstack(provider.target_weights_by_formation), dtype="<f8"
        )
        if target_weights.shape != weights.shape:
            raise ValueError("portfolio_application.target_replay_axis_mismatch")
        weights_hash = self.store.publish_lane(category="executed-weights", values=weights)
        final_hash = self.store.publish_lane(category="final-weights", values=final_weights)
        pre_cap_hash = self.store.publish_lane(category="pre-cap-weights", values=uncapped_weights)
        # The three inputs a rerun of the transition needs, sealed beside the
        # outputs they produced. Without them "replay" can only reread what was
        # written down and report that it agrees with itself.
        target_hash = self.store.publish_lane(category="target-weights", values=target_weights)
        realized_hash = self.store.publish_lane(
            category="realized-returns",
            values=np.ascontiguousarray(workspace.realized_simple_returns, dtype="<f8"),
        )
        available_hash = self.store.publish_lane(
            category="execution-available",
            values=np.ascontiguousarray(
                np.asarray(workspace.execution_available, dtype=bool), dtype=np.uint8
            ),
        )
        # A continuation opens on the boundary it was handed, verbatim. Re-sealing
        # it here would produce a second boundary identity for one state -- and
        # the Program binds the carried one, so the ledger would then disagree
        # with the Program about where the path started.
        if continuation is not None:
            # Copy the carried boundary's lanes into *this* store. The bytes came
            # from the development namespace, and a sealed path has to be
            # readable from its own root -- otherwise a promoted protected result
            # would carry a boundary whose arrays live somewhere else.
            self.store.publish_lane(
                category="boundary-weights",
                values=np.ascontiguousarray(continuation.state.pretrade_weights, dtype="<f8"),
            )
            self.store.publish_lane(
                category="boundary-weights",
                values=np.ascontiguousarray(continuation.state.optimizer_reference, dtype="<f8"),
            )
            if continuation.sleeve_weights is not None:
                self.store.publish_lane(
                    category="boundary-sleeves",
                    values=np.ascontiguousarray(continuation.sleeve_weights, dtype="<f8"),
                )
        initial_boundary = (
            continuation.boundary
            if continuation is not None
            else self._seal_boundary(
                state=opening,
                sleeves=None,
                listing_count=len(listings),
                formation_session=sessions[0],
                decided_formation_count=0,
            )
        )
        final_boundary = self._seal_boundary(
            state=segment.final_state,
            sleeves=provider.sleeve_state,
            listing_count=len(listings),
            formation_session=sessions[-1],
            decided_formation_count=provider.schedule_offset + len(sessions),
        )
        ledger_identity: dict[str, object] = {
            "kind": "PortfolioExecutionLedger",
            "program_hash": program.program_hash,
            "formation_sessions": sessions,
            "ordered_listing_ids": listings,
            "executed_weights_hash": weights_hash,
            "executed_weights_shape": weights.shape,
            "pre_cap_weights_hash": pre_cap_hash,
            "pre_cap_weights_shape": uncapped_weights.shape,
            "final_weights_hash": final_hash,
            "gross_simple_returns": segment.gross_simple_returns,
            "one_way_turnovers": segment.one_way_turnovers,
            "decision_modes": segment.decision_modes,
            "consumed_score_projection_hashes": provider.consumed_score_projection_hashes,
            "consumed_score_receipt_hashes": resolution.score_receipt_hashes,
            "consumed_risk_projection_hashes": provider.consumed_risk_projection_hashes,
            "aggregate_cap_binding_counts": provider.aggregate_cap_binding_counts,
            "missed_execution_count": segment.missed_execution_count,
            # The anchor is the eligible-universe equal weight the path is
            # already measured against, so it is derived from the same executed
            # workspace rather than resolved as a second input.
            "anchor_simple_returns": tuple(
                float(np.mean(workspace.passive_returns_by_session[session]))
                for session in sessions
            ),
            "risk_facts": risk_facts,
            "median_holding_adv20_by_formation": tuple(
                _median_holding_adv20(adv20=adv_rows[index], final_weights=weights[index])
                for index in range(len(sessions))
            ),
            "target_weights_hash": target_hash,
            "realized_simple_returns_hash": realized_hash,
            "execution_available_hash": available_hash,
            "initial_boundary": initial_boundary,
            "final_boundary": final_boundary,
        }
        # `risk_facts` are models, so the hash has to be taken over the same
        # serialised form the validator later dumps -- not over the objects.
        ledger_payload = PortfolioExecutionLedger.model_construct(
            **ledger_identity, ledger_hash="0" * 64
        ).model_dump(mode="json", exclude={"ledger_hash"})
        ledger = PortfolioExecutionLedger(
            **ledger_payload,
            ledger_hash=canonical_hash(ledger_payload),
        )
        self.store.publish_program(program)
        self.store.publish_execution(ledger)
        return ledger

    def _seal_boundary(
        self,
        *,
        state: PortfolioWalkForwardState,
        sleeves: npt.NDArray[np.float64] | None,
        listing_count: int,
        formation_session: date,
        decided_formation_count: int,
    ) -> SealedPortfolioBoundaryState:
        """Publish every component of one path boundary and name them together.

        All five, not just the reference. A boundary that seals one book and
        lets the other be re-derived is not a boundary -- the two differ by
        exactly one drift, and a continuation restored from the wrong one
        prices its opening trade against a book it never held.
        """

        pretrade_hash = self.store.publish_lane(
            category="boundary-weights",
            values=np.ascontiguousarray(state.pretrade_weights, dtype="<f8"),
        )
        # Published into this category even when the same bytes already sit in
        # `final-weights`: content addressing dedupes on the digest, and a
        # boundary whose lanes are all in one place is one that can be restored
        # without knowing which path produced it.
        reference_hash = self.store.publish_lane(
            category="boundary-weights",
            values=np.ascontiguousarray(state.optimizer_reference, dtype="<f8"),
        )
        sleeve_hash = None
        sleeve_count = 0
        if sleeves is not None:
            sleeve_rows = np.ascontiguousarray(sleeves, dtype="<f8")
            sleeve_hash = self.store.publish_lane(category="boundary-sleeves", values=sleeve_rows)
            sleeve_count = int(sleeve_rows.shape[0])
        return SealedPortfolioBoundaryState.create(
            listing_count=listing_count,
            formation_session=formation_session,
            pretrade_weights_hash=pretrade_hash,
            pretrade_cash=float(state.pretrade_cash),
            optimizer_reference_hash=reference_hash,
            optimizer_reference_cash=float(state.optimizer_reference_cash),
            sleeve_weights_hash=sleeve_hash,
            sleeve_count=sleeve_count,
            decided_formation_count=decided_formation_count,
        )

    def _publish_descendants(
        self,
        *,
        workspace_id: str,
        spec: PortfolioResearchSpec,
        program: PortfolioExecutionProgram,
        ledger: PortfolioExecutionLedger,
        resolved: Callable[[], ResolvedPortfolioExecution],
        coverage: PortfolioSupportCoverage,
        authorities_hash: str,
    ) -> PortfolioResearchResult:
        """Everything a descendant control rebuilds, read from the stored path.

        No walk, no policy, no Risk surface. The only reason this can still need
        the resolution is the SPY comparator, which is a spec-level choice rather
        than a property of the path -- and the default view does not ask for it.
        """

        turnover: npt.NDArray[np.float64] = np.asarray(ledger.one_way_turnovers, dtype=np.float64)
        gross: npt.NDArray[np.float64] = np.asarray(ledger.gross_simple_returns, dtype=np.float64)
        cost_policy = PortfolioCostPolicy(
            reporting_bps=(int(spec.cost.platform_cost_bps),),
            selection_bps=int(spec.cost.platform_cost_bps),
        )
        net = cost_policy.net_simple_returns(
            gross_simple_returns=gross,
            one_way_turnovers=turnover,
            cost_bps=float(spec.cost.platform_cost_bps),
        )
        wealth = float(np.prod(1.0 + net))
        anchor: npt.NDArray[np.float64] = np.asarray(ledger.anchor_simple_returns, dtype=np.float64)
        # The Backtesting owner, not a local expression. `active_path_metrics`
        # is the same `cov / var` the economic metric set uses, and calling it
        # is what keeps one beta semantics in the product.
        try:
            beta: float | None = active_path_metrics(
                portfolio_simple=net, benchmark_simple=anchor
            ).beta
        except ActiveMetricsError:
            # The owner refuses a benchmark that did not move. That is a fact
            # about the path, not an error to paper over with a zero.
            beta = None
        economic_identity: dict[str, object] = {
            "kind": "PortfolioEconomicLedger",
            "execution_ledger_hash": ledger.ledger_hash,
            "cost_assumption_hash": spec.cost.assumption_hash,
            "cost_bps_per_side": f"{spec.cost.cost_bps_per_side}",
            "platform_one_way_cost_bps": f"{spec.cost.platform_one_way_cost_bps}",
            "net_simple_returns": tuple(float(value) for value in net),
            "cumulative_net_wealth": wealth,
            "benchmark_beta": None if beta is None else float(beta),
            "benchmark_beta_disposition": (
                "ANCHOR_DEGENERATE_BETA_NOT_ESTIMABLE"
                if beta is None
                else "BACKTESTING_ACTIVE_PATH_BETA_NET_VS_ANCHOR_FULL_PATH"
            ),
        }
        economics = PortfolioEconomicLedger(
            **economic_identity,
            economic_ledger_hash=canonical_hash(economic_identity),
        )
        # Only the SPY comparator is a spec-level choice rather than a property
        # of the path, so only it can still require the resolution.
        secondary = resolved() if spec.secondary_benchmark_view == "anchor_plus_spy" else None
        comparison = PortfolioBenchmarkComparison.create(
            execution_ledger_hash=ledger.ledger_hash,
            economic_ledger_hash=economics.economic_ledger_hash,
            secondary_benchmark_view=spec.secondary_benchmark_view,
            primary_simple_returns=tuple(float(value) for value in anchor),
            secondary_benchmark_id=None if secondary is None else secondary.secondary_benchmark_id,
            secondary_simple_returns=(
                None if secondary is None else secondary.secondary_benchmark_returns
            ),
            secondary_disposition=(
                "SECONDARY_BENCHMARK_NOT_REQUESTED"
                if secondary is None
                else secondary.secondary_benchmark_disposition
            ),
        )

        executed: npt.NDArray[np.float64] = np.frombuffer(
            self.store.load_lane(
                category="executed-weights", content_hash=ledger.executed_weights_hash
            ),
            dtype="<f8",
        ).reshape(ledger.executed_weights_shape)
        # One owner for the report projection, shared with strong replay. Two
        # implementations of this would make a replay match mean only that the
        # two agreed, and a drift between them look like a failed path.
        report = project_declared_path_report(
            spec=spec,
            program=program,
            ledger=ledger,
            economics=economics,
            comparison=comparison,
            coverage=coverage,
            net_simple_returns=np.asarray(net, dtype=np.float64),
            one_way_turnovers=turnover,
            executed_weights=executed,
            # Read back from the ledger's own sealed boundary lane rather than
            # from the walk that just ran: this same projection also serves a
            # cost-only reopen of an existing ledger, where no walk happened and
            # the opening book exists only as sealed bytes.
            opening_reference_weights=self.store.load_opening_reference(ledger),
        )
        self.store.publish_economics(economics)
        self.store.publish_comparison(comparison)
        report_uri = self.store.publish_report(report)
        html_payload = render_portfolio_research_html(
            spec=spec,
            report=report,
            economics=economics,
            comparison=comparison,
            coverage=coverage,
            disclosure=self.disclosure(report=report, program=program),
            package=self.packages.get(program.strategy_package_hash or ""),
            context=None
            if self.report_context is None
            else self.report_context(program, report.window_end_book.formation_session),
        )
        _html_hash, html_uri = self.store.publish_html(html_payload)
        result_identity: dict[str, object] = {
            "kind": "PortfolioResearchResult",
            "spec_hash": spec.spec_hash,
            "program_hash": program.program_hash,
            "execution_ledger_hash": ledger.ledger_hash,
            "economic_ledger_hash": economics.economic_ledger_hash,
            "report_hash": report.report_hash,
            "report_uri": report_uri,
            "html_uri": html_uri,
        }
        result = PortfolioResearchResult(
            action="PUBLISHED",
            **result_identity,
            result_hash=canonical_hash(result_identity),
        )
        self.store.publish_result(result)
        if self.store.is_public:
            # The two request-shaped indices are reuse indices: they answer "has
            # this holdings configuration, over these authorities, already been
            # compiled and run". A pending protected path is never an answer to
            # that -- nothing may reuse it, and it is deleted or promoted by
            # identity. Writing them in the pending namespace also imposed a
            # limit nobody asked for: two finalizations continuing *different*
            # books across the same fixture share a holdings spec and an
            # authority receipt, and the second one collided with the first.
            self.store.publish_program_index(
                holdings_spec_hash=program.holdings_spec_hash,
                authorities_hash=authorities_hash,
                program_hash=program.program_hash,
            )
            self.store.publish_plan_index(
                spec_hash=spec.spec_hash,
                authorities_hash=authorities_hash,
                result_hash=result.result_hash,
            )
        return result


def _median_holding_adv20(
    *, adv20: npt.NDArray[np.float64], final_weights: npt.NDArray[np.float64]
) -> float | None:
    """Median 20-session average dollar volume across held names, or `None`."""

    held = final_weights > 1e-12
    finite = adv20[held & np.isfinite(adv20)]
    return float(np.median(finite)) if finite.size else None


__all__ = ["PortfolioResearchExecutor", "ResolvedPortfolioExecution"]
