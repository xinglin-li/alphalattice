"""Two receipts with different jobs: reopen everything, and rederive everything.

`EXACT_READBACK` proves a published path can be reopened layer by layer with no
numerical work at all. `STRONG_REPLAY` proves the numbers follow from the sealed
children by recomputing them through the same owners and comparing identities.
They are separate because they answer separate doubts, and a single receipt
claiming both would have to permit the work one of them forbids.

Three rules shape the module.

**Existence is not verification.** A layer is `VERIFIED` only when its artifact
reopened *and* its own identity check passed *and* the parent's reference to it
matched. An artifact that is merely present, or present under a hash nobody
cites, is reported as what it is.

**What cannot be proved is named.** Readback runs over one workspace, so an
evidence package living outside it is unreachable from here -- and the receipt
says `PARTIAL` with the reason rather than quietly omitting the row. A receipt
whose silence means "fine" is worse than no receipt.

**Replay may compute, readback may not.** Readback's work counters are all zero
by contract. Replay's Backtesting and metric counters are expected to be
positive; its Alpha, Risk, Provider, protected-read and pointer counters are
still zero, because rederiving a path must not become a way to refit a model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.active_metrics import (
    ActiveMetricsError,
    active_path_metrics,
)
from alphalattice.capabilities.portfolio_backtesting.contracts import (
    PortfolioCostPolicy,
    PortfolioWalkForwardState,
)
from alphalattice.capabilities.portfolio_backtesting.execution import (
    TOLERANCE,
    drift_holdings,
    execute_orders,
)
from alphalattice.investment.portfolio_strategy_lab.application.advancement import (
    PortfolioLedgerCoverage,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioApplicationError,
    PortfolioBenchmarkComparison,
    PortfolioControlReceipt,
    PortfolioDeclaredPathReport,
    PortfolioEconomicLedger,
    PortfolioExecutionLedger,
    PortfolioExecutionProgram,
    PortfolioResearchResult,
    PortfolioResearchSpec,
    PortfolioSupportCoverage,
    SealedPortfolioBoundaryState,
)
from alphalattice.investment.portfolio_strategy_lab.application.finalization import (
    FinalPortfolioEvaluationPackage,
    PortfolioValidationReceipt,
    ReleasedPortfolioArtifacts,
    ValidatedPortfolioHandoff,
)
from alphalattice.investment.portfolio_strategy_lab.application.report_projection import (
    project_declared_path_report,
)
from alphalattice.investment.portfolio_strategy_lab.publication.portfolio_ledger import (
    PortfolioLedgerStore,
    PortfolioLedgerStoreError,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sealing import seal_model_from_dump

ReadbackDisposition = Literal[
    "VERIFIED",
    "PARTIAL",
    "MISSING",
    "LEGACY",
    "FAILED",
    "NOT_APPLICABLE",
]

READBACK_LAYERS: tuple[str, ...] = (
    "alpha_recipe",
    "alpha_evidence_package",
    "risk_recipe",
    "risk_return_surface",
    "risk_attribution_projections",
    "portfolio_program",
    "numerical_input_assembly",
    "execution_ledger",
    "executed_weight_lanes",
    "economic_ledger",
    "benchmark_comparison",
    "controls_receipt",
    "schedule_and_window_guards",
    "report_facts",
    "rendered_artifact",
    "final_evaluation_package",
    "validation_receipt",
    "validated_handoff",
    "released_artifacts",
)
"""Every layer a reader could ask about, including the ones often absent.

Fixed and complete on purpose: a receipt that only listed the layers it happened
to find would report a development path and a finalized one identically.
"""

REPLAY_LAYERS: tuple[str, ...] = (
    "portfolio_state",
    "drifted_pretrade_state",
    "fills_and_missed_fills",
    "one_way_turnover",
    "terminal_boundary",
    "cost_and_net_returns",
    "benchmark_alignment",
    "economic_metrics",
    "report_metrics",
)
"""Every quantity the rerun derives, each compared against its sealed original.

Ordered as the transition produces them, so a reader can see where a mismatch
first appears rather than which check happened to be written first.
"""


class ReadbackError(ValueError):
    """Stable refusal for a readback or replay identity failure."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReadbackLayer(_Contract):
    """One layer, its disposition, and the identity that carries the claim."""

    kind: Literal["ReadbackLayer"] = "ReadbackLayer"
    layer_id: str = Field(min_length=1, max_length=64)
    disposition: ReadbackDisposition
    identity_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    detail: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_layer(self) -> Self:
        """Require verified-layer identity and explicit absence for missing/inapplicable layers.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ReadbackError: VERIFIED lacks an identity or MISSING/NOT_APPLICABLE carries one.
        """
        if self.disposition == "VERIFIED" and self.identity_hash is None:
            # A verified layer without an identity is a claim with nothing behind
            # it -- exactly the "artifact exists" shortcut this receipt exists to
            # prevent.
            raise ReadbackError("portfolio_readback.verified_layer_without_identity")
        if self.disposition in {"MISSING", "NOT_APPLICABLE"} and self.identity_hash is not None:
            raise ReadbackError("portfolio_readback.absent_layer_carries_identity")
        return self


class ReadbackWork(_Contract):
    """What was actually done while producing a receipt."""

    kind: Literal["ReadbackWork"] = "ReadbackWork"
    alpha_fits: int = Field(default=0, ge=0)
    alpha_scores: int = Field(default=0, ge=0)
    risk_estimations: int = Field(default=0, ge=0)
    policy_executions: int = Field(default=0, ge=0)
    backtesting_transitions: int = Field(default=0, ge=0)
    metric_recomputations: int = Field(default=0, ge=0)
    provider_calls: int = Field(default=0, ge=0)
    protected_reads: int = Field(default=0, ge=0)
    pointer_mutations: int = Field(default=0, ge=0)
    publications: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        """Sum every declared readback work counter excluding the record kind.

        Returns:
            Total declared work units; no work is performed by this projection.
        """
        return sum(getattr(self, name) for name in type(self).model_fields if name != "kind")

    @property
    def forbidden_in_replay(self) -> int:
        """The work a rederivation may never do, however much it recomputes."""
        return (
            self.alpha_fits
            + self.alpha_scores
            + self.risk_estimations
            + self.provider_calls
            + self.protected_reads
            + self.pointer_mutations
        )


class ExactReadbackReceipt(_Contract):
    """Every layer reopened, and the measured proof that nothing was computed."""

    kind: Literal["ExactReadbackReceipt"] = "ExactReadbackReceipt"
    workspace_id: str = Field(min_length=1, max_length=120)
    result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    layers: tuple[ReadbackLayer, ...] = Field(min_length=1)
    work: ReadbackWork = ReadbackWork()
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one exact zero-work readback receipt.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical receipt_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return seal_model_from_dump(cls, values, field="receipt_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require zero work, the complete ordered layer set and exact receipt identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ReadbackError: Any work was performed, the retained layer IDs differ from
                READBACK_LAYERS or receipt_hash differs.
        """
        if self.work.total != 0:
            # The one invariant the whole receipt exists for.
            raise ReadbackError("portfolio_readback.exact_readback_performed_work")
        if tuple(value.layer_id for value in self.layers) != READBACK_LAYERS:
            raise ReadbackError("portfolio_readback.layer_set_incomplete")
        if self.receipt_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        ):
            raise ReadbackError("portfolio_readback.receipt_identity_invalid")
        return self

    def layer(self, layer_id: str) -> ReadbackLayer:
        """Resolve one explicitly retained proof layer.

        Args:
            layer_id: Exact layer identity.

        Returns:
            Matching retained layer.

        Raises:
            ReadbackError: The requested layer is absent.
        """
        for value in self.layers:
            if value.layer_id == layer_id:
                return value
        raise ReadbackError("portfolio_readback.layer_absent:" + layer_id)

    @property
    def proved(self) -> tuple[str, ...]:
        """Read retained layer identities whose disposition is verified.

        Returns:
            Verified layer IDs in receipt order.
        """
        return tuple(v.layer_id for v in self.layers if v.disposition == "VERIFIED")

    @property
    def unproved(self) -> tuple[str, ...]:
        """Read retained layer identities whose disposition is not verified.

        Returns:
            Unproved layer IDs in receipt order, including missing or inapplicable layers.
        """
        return tuple(v.layer_id for v in self.layers if v.disposition != "VERIFIED")


class StrongReplayReceipt(_Contract):
    """Every rederived quantity compared to its sealed original, by identity."""

    kind: Literal["StrongReplayReceipt"] = "StrongReplayReceipt"
    workspace_id: str = Field(min_length=1, max_length=120)
    result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_input_assembly_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    layers: tuple[ReadbackLayer, ...] = Field(min_length=1)
    rederived_economic_ledger_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sealed_economic_ledger_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    work: ReadbackWork = ReadbackWork()
    disposition: Literal[
        "REPLAYED_EXACT",
        "REFUSED_LEGACY_ASSEMBLY",
        "REFUSED_UNSEALED_CHILDREN",
        "REFUSED_SPEC_MISMATCH",
        "FAILED_MISMATCH",
    ]
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one declared strong replay receipt.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical receipt_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return seal_model_from_dump(cls, values, field="receipt_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require admitted replay work and exact claims only for completely proved layers.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ReadbackError: Forbidden work is nonzero, legacy refusal carries assembly/work, a
                refusal performed work, exact replay lacks matching economic identity or fully
                verified ordered layers, or receipt_hash differs.
        """
        if self.work.forbidden_in_replay != 0:
            raise ReadbackError("portfolio_readback.strong_replay_performed_forbidden_work")
        if (
            self.disposition == "REFUSED_LEGACY_ASSEMBLY"
            and self.numerical_input_assembly_hash is not None
        ):
            raise ReadbackError("portfolio_readback.legacy_refusal_carries_an_assembly")
        if self.disposition in {
            "REFUSED_LEGACY_ASSEMBLY",
            "REFUSED_UNSEALED_CHILDREN",
            "REFUSED_SPEC_MISMATCH",
        } and (self.work.total != 0):
            # A path that cannot name its numerical inputs, or did not seal the
            # children a rerun consumes, must be refused *before* anything is
            # recomputed, or the refusal is decorative.
            raise ReadbackError("portfolio_readback.legacy_refusal_performed_work")
        if self.disposition == "REPLAYED_EXACT":
            if (
                self.rederived_economic_ledger_hash is None
                or self.rederived_economic_ledger_hash != self.sealed_economic_ledger_hash
            ):
                raise ReadbackError("portfolio_readback.replay_claimed_exact_without_a_match")
            if any(value.disposition != "VERIFIED" for value in self.layers):
                # Exact means every layer was rederived and matched. A `PARTIAL`
                # layer is a hole in the proof, and calling the receipt exact
                # with a hole in it is precisely the claim this contract exists
                # to make unstateable.
                raise ReadbackError(
                    "portfolio_readback.replay_claimed_exact_with_an_unproved_layer"
                )
            if tuple(value.layer_id for value in self.layers) != REPLAY_LAYERS:
                raise ReadbackError("portfolio_readback.replay_layer_set_incomplete")
        if self.receipt_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        ):
            raise ReadbackError("portfolio_readback.replay_receipt_identity_invalid")
        return self

    def layer(self, layer_id: str) -> ReadbackLayer:
        """Resolve one explicitly retained proof layer.

        Args:
            layer_id: Exact layer identity.

        Returns:
            Matching retained layer.

        Raises:
            ReadbackError: The requested layer is absent.
        """
        for value in self.layers:
            if value.layer_id == layer_id:
                return value
        raise ReadbackError("portfolio_readback.layer_absent:" + layer_id)


def _layer(
    layer_id: str,
    disposition: ReadbackDisposition,
    detail: str,
    identity_hash: str | None = None,
) -> ReadbackLayer:
    return ReadbackLayer(
        layer_id=layer_id,
        disposition=disposition,
        identity_hash=identity_hash,
        detail=detail,
    )


class OpenedUpstreamArtifact(_Contract):
    """One upstream artifact, opened, carrying the identity it was asked for.

    Typed on purpose. An opener that returned a bare object could hand back
    anything at all and readback would report the layer proved -- so the one
    thing that crosses this boundary is a record that *names* its own identity,
    and readback compares that name against the identity the Program bound.
    """

    kind: Literal["OpenedUpstreamArtifact"] = "OpenedUpstreamArtifact"
    role: Literal[
        "ALPHA_RECIPE",
        "ALPHA_EVIDENCE_MANIFEST",
        "RISK_RECIPE",
        "RISK_RETURN_SURFACE",
    ]
    identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    closure_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    child_count: int = Field(ge=0)
    admitted_at: datetime
    record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one declared opened upstream artifact record.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical record_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, record_hash="0" * 64).model_dump(
            mode="json", exclude={"record_hash"}
        )
        return cls(**identity, record_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require timezone-aware upstream admission and exact retained record identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ReadbackError: Admission time lacks timezone authority or record_hash differs.
        """
        if self.admitted_at.tzinfo is None or self.admitted_at.utcoffset() is None:
            raise ReadbackError("portfolio_readback.upstream_clock_invalid")
        if self.record_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"record_hash"})
        ):
            raise ReadbackError("portfolio_readback.upstream_identity_invalid")
        return self


class UpstreamEvidenceOpener(Protocol):
    """Read-only access to the Alpha and Risk artifacts a Program merely names.

    Injected rather than assumed, because those artifacts do not live in the
    Portfolio workspace. Without an opener the layers are `PARTIAL` and say so;
    with one they are proved by *opening*, which is the difference between
    "the Program has a field" and "the thing that field names exists".
    """

    def open_upstream(self, *, role: str, identity_hash: str) -> OpenedUpstreamArtifact | None:
        """The admitted artifact for this role and identity, if it is reachable."""
        ...


class FinalizationArtifactOpener(Protocol):
    """Read-only access to the three artifacts a finalization published.

    The package and handoff are Portfolio's; the validation receipt is the
    Gate's. Readback consumes one port over both rather than importing either
    store, so proving a finalization does not make the readback owner depend on
    the authority whose answer it is checking.

    Typed returns, because readback has to check how the three refer to *each
    other*. Three hashes and a promise that they opened proves nothing about
    whether they belong together.
    """

    def open_package(self, package_hash: str) -> FinalPortfolioEvaluationPackage | None:
        """Open the exact retained finalization artifact through its deterministic owner.

        Args:
            package_hash: Exact requested artifact identity.

        Returns:
            FinalPortfolioEvaluationPackage when retained, otherwise None.
        """
        ...

    def open_validation_receipt(self, receipt_hash: str) -> PortfolioValidationReceipt | None:
        """Open the exact retained finalization artifact through its deterministic owner.

        Args:
            receipt_hash: Exact requested artifact identity.

        Returns:
            PortfolioValidationReceipt when retained, otherwise None.
        """
        ...

    def open_handoff(self, handoff_hash: str) -> ValidatedPortfolioHandoff | None:
        """Open the exact retained finalization artifact through its deterministic owner.

        Args:
            handoff_hash: Exact requested artifact identity.

        Returns:
            ValidatedPortfolioHandoff when retained, otherwise None.
        """
        ...

    def open_release(self, package_hash: str) -> ReleasedPortfolioArtifacts | None:
        """The release marker for this package, if the release ever committed.

        The last of the four and the one that decides the other three. Copied
        artifacts are preparation: after adoption and before the marker, the
        public store holds bytes that were never released, and a reader holding
        the hashes could otherwise open all three finalization artifacts and call
        the closure proved.
        """
        ...


def _upstream_layer(
    layer_id: str,
    *,
    role: str,
    identity: str,
    opener: UpstreamEvidenceOpener | None,
    proved_detail: str,
    unreachable_detail: str,
) -> ReadbackLayer:
    """Open one upstream artifact and check it is the one that was asked for."""

    if opener is None:
        return _layer(layer_id, "PARTIAL", unreachable_detail, identity)
    try:
        opened = opener.open_upstream(role=role, identity_hash=identity)
    except Exception as error:
        return _layer(layer_id, "FAILED", f"{layer_id} did not reopen: {error}"[:240])
    if opened is None:
        return _layer(layer_id, "FAILED", f"{layer_id} is bound by the Program but not admitted")
    if opened.identity_hash != identity or opened.role != role:
        # The opener answered with something. Whether it answered with the thing
        # that was asked for is a separate question, and this is it.
        return _layer(
            layer_id,
            "FAILED",
            f"{layer_id} opened under a different identity than the Program binds",
        )
    return _layer(layer_id, "VERIFIED", proved_detail, identity)


def exact_readback(
    store: PortfolioLedgerStore,
    *,
    workspace_id: str,
    result_hash: str,
    package_hash: str | None = None,
    validation_receipt_hash: str | None = None,
    handoff_hash: str | None = None,
    upstream: UpstreamEvidenceOpener | None = None,
    finalization: FinalizationArtifactOpener | None = None,
) -> ExactReadbackReceipt:
    """Reopen one published path layer by layer, computing nothing.

    Every layer marked `VERIFIED` here was *opened*: the artifact was loaded, its
    own identity check passed on load, and the parent's reference to it matched
    what came back. A hash the caller supplied, or one copied off the Program, is
    a binding and is reported as one.

    When finalization identities are supplied, `result_hash` must be the
    *released protected* result. Reading back a development result while quoting
    a finalization's package and handoff describes two different paths and calls
    the pair proved, which is the shape of claim this receipt exists to refuse.
    """
    layers: list[ReadbackLayer] = []
    try:
        result = store.load_result(result_hash)
        program = store.load_program(result.program_hash)
        ledger = store.load_execution(result.execution_ledger_hash)
        economics = store.load_economics(result.economic_ledger_hash)
        report = store.load_report(result.report_hash)
    except (PortfolioLedgerStoreError, PortfolioApplicationError, ValueError) as error:
        raise ReadbackError("portfolio_readback.result_not_reopenable:" + str(error)) from error

    comparison: PortfolioBenchmarkComparison | None = None
    comparison_failure: str | None = None
    try:
        comparison = store.load_comparison(report.benchmark_comparison_hash)
    except (PortfolioLedgerStoreError, PortfolioApplicationError, ValueError) as error:
        comparison_failure = f"the report's benchmark comparison did not reopen: {error}"[:240]

    ledger_failure = (
        None
        if ledger.program_hash == program.program_hash
        else "the execution ledger names a different Program than the result"
    )
    economic_failure = (
        None
        if economics.execution_ledger_hash == ledger.ledger_hash
        else "the economic ledger names a different execution ledger than the result"
    )
    expected_coverage = PortfolioLedgerCoverage.of(
        program_hash=program.program_hash,
        ledger_hash=ledger.ledger_hash,
        formation_sessions=tuple(ledger.formation_sessions),
        ordered_listing_ids=tuple(ledger.ordered_listing_ids),
        source_coverage_hash=report.ledger_coverage.source_coverage_hash,
    )
    report_failure: str | None = None
    if (
        report.program_hash != program.program_hash
        or report.execution_ledger_hash != ledger.ledger_hash
        or report.economic_ledger_hash != economics.economic_ledger_hash
    ):
        report_failure = "the report names a different Program or ledger than the result"
    elif report.ledger_coverage != expected_coverage:
        report_failure = "the report coverage does not describe the opened Program and ledger axes"
    elif report.window_guard.coverage_hash != report.ledger_coverage.source_coverage_hash:
        report_failure = "the report window guard is bound to a different source coverage"
    elif comparison is not None and report.benchmark_comparison_hash != comparison.comparison_hash:
        report_failure = "the report names a different benchmark comparison than the one opened"

    if comparison is not None and comparison_failure is None:
        if (
            comparison.execution_ledger_hash != ledger.ledger_hash
            or comparison.economic_ledger_hash != economics.economic_ledger_hash
        ):
            comparison_failure = "the benchmark comparison names a different Portfolio path"
        elif comparison.primary_simple_returns != ledger.anchor_simple_returns:
            comparison_failure = "the benchmark comparison anchor differs from the execution ledger"

    layers.append(
        _upstream_layer(
            "alpha_recipe",
            role="ALPHA_RECIPE",
            identity=program.alpha_recipe_hash,
            opener=upstream,
            proved_detail="the Alpha recipe the Program binds was opened at that identity",
            unreachable_detail=(
                "recipe identity bound by the Program; no upstream opener was supplied, so the "
                "artifact itself was not read"
            ),
        )
    )
    layers.append(
        _upstream_layer(
            "alpha_evidence_package",
            role="ALPHA_EVIDENCE_MANIFEST",
            identity=program.alpha_evidence_manifest_hash,
            opener=upstream,
            proved_detail="the evidence manifest the Program binds was opened at that identity",
            unreachable_detail=(
                "manifest identity bound; the package lives outside this workspace and was "
                "not opened"
            ),
        )
    )
    layers.append(
        _upstream_layer(
            "risk_recipe",
            role="RISK_RECIPE",
            identity=program.risk_recipe_hash,
            opener=upstream,
            proved_detail="the Risk recipe the Program binds was opened at that identity",
            unreachable_detail=(
                "recipe identity bound by the Program; no upstream opener was supplied"
            ),
        )
    )
    layers.append(
        _upstream_layer(
            "risk_return_surface",
            role="RISK_RETURN_SURFACE",
            identity=program.risk_return_surface_hash,
            opener=upstream,
            proved_detail="the Risk return surface the Program binds was opened at that identity",
            unreachable_detail=(
                "surface identity bound by the Program; no upstream opener was supplied"
            ),
        )
    )

    # These *are* in the workspace: they were reopened as part of the ledger, and
    # each fact re-verifies its own identity on load.
    attribution_identity = (
        str(canonical_hash(tuple(v.model_dump(mode="json") for v in ledger.risk_facts)))
        if ledger.risk_facts
        else None
    )
    layers.append(
        _layer(
            "risk_attribution_projections",
            "VERIFIED" if ledger.risk_facts else "NOT_APPLICABLE",
            (
                f"{len(ledger.risk_facts)} attribution facts reopened from the ledger, each "
                "self-verified on load"
                if ledger.risk_facts
                else "this path consumed no attribution projection"
            ),
            attribution_identity,
        )
    )
    layers.append(
        _layer(
            "portfolio_program",
            "VERIFIED",
            "program reopened, self-verified, and is the Program the result names",
            program.program_hash,
        )
    )
    assembly = program.numerical_input_assembly_hash
    layers.append(
        _layer(
            "numerical_input_assembly",
            "VERIFIED" if assembly is not None else "LEGACY",
            (
                "the Program stores one digest over every numerical input it bound, and it "
                "recomputes from those bindings"
                if assembly is not None
                else "pre-binding Program: it cannot name the inputs its numbers came from"
            ),
            assembly,
        )
    )
    layers.append(
        _layer(
            "execution_ledger",
            "VERIFIED" if ledger_failure is None else "FAILED",
            "ledger reopened and is bound to the result's Program"
            if ledger_failure is None
            else ledger_failure,
            ledger.ledger_hash if ledger_failure is None else None,
        )
    )

    lane_state: ReadbackDisposition = "VERIFIED"
    sealed_lanes = [
        ("executed-weights", ledger.executed_weights_hash),
        ("pre-cap-weights", ledger.pre_cap_weights_hash),
        ("final-weights", ledger.final_weights_hash),
        ("target-weights", ledger.target_weights_hash),
        ("realized-returns", ledger.realized_simple_returns_hash),
        ("execution-available", ledger.execution_available_hash),
    ]
    for boundary in (ledger.initial_boundary, ledger.final_boundary):
        if boundary is not None:
            sealed_lanes.append(("boundary-weights", boundary.pretrade_weights_hash))
            sealed_lanes.append(("boundary-weights", boundary.optimizer_reference_hash))
            sealed_lanes.append(("boundary-sleeves", boundary.sleeve_weights_hash))
    present = [(category, value) for category, value in sealed_lanes if value is not None]
    lane_detail = f"{len(present)} sealed lanes reopened by content hash"
    try:
        rows, listings = ledger.executed_weights_shape
        expected_bytes: list[tuple[str, str | None, int]] = [
            ("executed-weights", ledger.executed_weights_hash, rows * listings * 8),
            ("pre-cap-weights", ledger.pre_cap_weights_hash, rows * listings * 8),
            ("final-weights", ledger.final_weights_hash, listings * 8),
            ("target-weights", ledger.target_weights_hash, rows * listings * 8),
            ("realized-returns", ledger.realized_simple_returns_hash, rows * listings * 8),
            ("execution-available", ledger.execution_available_hash, rows * listings),
        ]
        for boundary in (ledger.initial_boundary, ledger.final_boundary):
            if boundary is not None:
                expected_bytes.extend(
                    (
                        ("boundary-weights", boundary.pretrade_weights_hash, listings * 8),
                        (
                            "boundary-weights",
                            boundary.optimizer_reference_hash,
                            listings * 8,
                        ),
                        (
                            "boundary-sleeves",
                            boundary.sleeve_weights_hash,
                            boundary.sleeve_count * listings * 8,
                        ),
                    )
                )
        for category, content_hash, size in expected_bytes:
            if content_hash is None:
                continue
            payload = store.load_lane(category=category, content_hash=content_hash)
            if len(payload) != size:
                raise ReadbackError(
                    f"portfolio_readback.sealed_lane_shape_invalid:{category}:{len(payload)}:{size}"
                )
    except Exception as error:
        lane_state = "FAILED"
        lane_detail = f"a sealed lane did not reopen: {error}"[:240]
    if lane_state == "VERIFIED" and not ledger.replayable:
        lane_state = "PARTIAL"
        lane_detail = (
            f"{len(present)} sealed lanes reopened; this ledger predates the replay children, "
            "so the inputs a rerun consumes are not among them"
        )
    layers.append(
        _layer(
            "executed_weight_lanes",
            lane_state,
            lane_detail,
            ledger.executed_weights_hash if lane_state != "FAILED" else None,
        )
    )
    layers.append(
        _layer(
            "economic_ledger",
            "VERIFIED" if economic_failure is None else "FAILED",
            "cost overlay reopened and bound to the execution ledger"
            if economic_failure is None
            else economic_failure,
            economics.economic_ledger_hash if economic_failure is None else None,
        )
    )
    layers.append(
        _layer(
            "benchmark_comparison",
            "VERIFIED" if comparison_failure is None else "FAILED",
            "benchmark comparison reopened and is bound to the execution and economic ledgers"
            if comparison_failure is None
            else comparison_failure,
            (
                None
                if comparison is None or comparison_failure is not None
                else comparison.comparison_hash
            ),
        )
    )
    layers.append(
        _layer(
            "controls_receipt",
            "VERIFIED",
            "every installed control value reopened beside the report",
            report.control_receipt.receipt_hash,
        )
    )
    layers.append(
        _layer(
            "schedule_and_window_guards",
            "VERIFIED",
            "sleeve schedule and study-window guards reopened",
            str(
                canonical_hash(
                    {
                        "schedule": report.schedule_guard.guard_hash,
                        "window": report.window_guard.guard_hash,
                    }
                )
            ),
        )
    )
    layers.append(
        _layer(
            "report_facts",
            "VERIFIED" if report_failure is None and comparison_failure is None else "FAILED",
            "typed report reopened and all Portfolio child bindings agree"
            if report_failure is None and comparison_failure is None
            else (report_failure or comparison_failure or "report closure invalid"),
            report.report_hash if report_failure is None and comparison_failure is None else None,
        )
    )

    rendered: ReadbackDisposition = "VERIFIED"
    rendered_detail = "self-contained page reopened from the ledger by its own URI"
    try:
        store.load_html_by_uri(result.html_uri)
    except Exception as error:
        rendered = "FAILED"
        rendered_detail = f"the rendered artifact did not reopen: {error}"[:240]
    layers.append(
        _layer(
            rendered_layer_id := "rendered_artifact",
            rendered,
            rendered_detail,
            result.result_hash if rendered == "VERIFIED" else None,
        )
    )
    del rendered_layer_id

    layers.extend(
        _finalization_layers(
            result_hash=result_hash,
            report_hash=result.report_hash,
            package_hash=package_hash,
            validation_receipt_hash=validation_receipt_hash,
            handoff_hash=handoff_hash,
            finalization=finalization,
        )
    )

    return ExactReadbackReceipt.create(
        workspace_id=workspace_id,
        result_hash=result_hash,
        layers=tuple(layers),
        work=ReadbackWork(),
    )


def _finalization_layers(
    *,
    result_hash: str,
    report_hash: str,
    package_hash: str | None,
    validation_receipt_hash: str | None,
    handoff_hash: str | None,
    finalization: FinalizationArtifactOpener | None,
) -> list[ReadbackLayer]:
    """The three finalization layers, opened and checked against each other.

    A development path has no finalization and says `NOT_APPLICABLE`; that is a
    real answer, not a gap. A finalized one is `VERIFIED` only where the artifact
    came back out of its store *and* refers to the others the way it must: the
    package must be about the result being read back, the receipt must close that
    package, and the handoff must name both plus the released report. Any of
    those three opened alone would pass while describing a different run.
    """

    requested = (
        ("final_evaluation_package", package_hash, "protected package"),
        ("validation_receipt", validation_receipt_hash, "metric-free validation receipt"),
        ("validated_handoff", handoff_hash, "handoff"),
        ("released_artifacts", package_hash, "release marker"),
    )
    if all(value is None for _id, value, _noun in requested):
        return [
            _layer(layer_id, "NOT_APPLICABLE", "this is a development path; no finalization exists")
            for layer_id, _value, _noun in requested
        ]
    if finalization is None:
        return [
            _layer(
                layer_id,
                "NOT_APPLICABLE" if value is None else "PARTIAL",
                "this is a development path; no finalization exists"
                if value is None
                else (
                    f"{noun} identity supplied by the caller; no finalization opener was "
                    "supplied, so the artifact itself was not read"
                ),
                value,
            )
            for layer_id, value, noun in requested
        ]

    identities_complete = all(
        value is not None for value in (package_hash, validation_receipt_hash, handoff_hash)
    )
    closure_failure = (
        None
        if identities_complete
        else (
            "the finalization identity set is incomplete; package, receipt and handoff are required"
        )
    )
    package = None if package_hash is None else _safely(finalization.open_package, package_hash)
    receipt = (
        None
        if validation_receipt_hash is None
        else _safely(finalization.open_validation_receipt, validation_receipt_hash)
    )
    handoff = None if handoff_hash is None else _safely(finalization.open_handoff, handoff_hash)
    marker = None if package_hash is None else _safely(finalization.open_release, package_hash)

    package_failure = _package_failure(
        package_hash=package_hash, package=package, result_hash=result_hash, report_hash=report_hash
    )
    receipt_failure = _receipt_failure(
        receipt_hash=validation_receipt_hash, receipt=receipt, package=package
    )
    handoff_failure = _handoff_failure(
        handoff_hash=handoff_hash, handoff=handoff, package=package, receipt=receipt
    )
    release_failure = closure_failure or _release_failure(
        package_hash=package_hash,
        marker=marker,
        package=package,
        receipt=receipt,
        handoff=handoff,
        result_hash=result_hash,
        report_hash=report_hash,
    )
    if release_failure is not None:
        # An unreleased finalization is not a partly proved one. The three
        # artifacts may all open -- adoption puts them where they will be needed
        # -- and none of that makes the closure readable, so every layer of it
        # carries the same answer rather than three encouraging ones and a
        # quiet fourth.
        package_failure = package_failure or release_failure
        receipt_failure = receipt_failure or release_failure
        handoff_failure = handoff_failure or release_failure
    return [
        _finalization_layer(
            "final_evaluation_package", package_hash, package_failure, "protected package"
        ),
        _finalization_layer(
            "validation_receipt",
            validation_receipt_hash,
            receipt_failure,
            "metric-free validation receipt",
        ),
        _finalization_layer("validated_handoff", handoff_hash, handoff_failure, "handoff"),
        _release_layer(marker=marker, package_hash=package_hash, failure=release_failure),
    ]


def _release_failure(
    *,
    package_hash: str | None,
    marker: ReleasedPortfolioArtifacts | None,
    package: FinalPortfolioEvaluationPackage | None,
    receipt: PortfolioValidationReceipt | None,
    handoff: ValidatedPortfolioHandoff | None,
    result_hash: str,
    report_hash: str,
) -> str | None:
    """Whether this finalization was released, and released as *this* one.

    Six identities have to agree, because the marker is the only thing that binds
    them into a single fact: the package that was evaluated, the receipt that
    closed it, the handoff minted for it, and the exact result and report those
    refer to. A marker that named five of the six would be a release of
    something else.
    """

    if package_hash is None:
        return "the finalization package identity is absent"
    if marker is None:
        return (
            "no release marker: the artifacts may be adopted, but this finalization "
            "has not been released"
        )
    if marker.package_hash != package_hash:
        return "the release marker is for a different package"
    if marker.released_result_hash != result_hash:
        return "the release marker released a different result"
    if marker.released_report_hash != report_hash:
        return "the release marker released a different report"
    if package is None:
        return "the release marker cannot be verified because its package is absent"
    if receipt is None:
        return "the release marker cannot be verified because its receipt is absent"
    if handoff is None:
        return "the release marker cannot be verified because its handoff is absent"
    if marker.released_result_hash != package.protected_result_hash:
        return "the release marker and the package disagree about the released result"
    if marker.validation_receipt_hash != receipt.receipt_hash:
        return "the release marker cites a different validation receipt"
    if marker.handoff_hash != handoff.handoff_hash:
        return "the release marker cites a different handoff"
    if receipt.package_hash != package.package_hash:
        return "the release receipt closes a different package"
    if (
        handoff.package_hash != package.package_hash
        or handoff.validation_receipt_hash != receipt.receipt_hash
        or handoff.released_report_hash != package.protected_report_hash
    ):
        return "the release handoff does not close the opened package and receipt"
    return None


def _finalization_layer(
    layer_id: str, identity: str | None, failure: str | None, noun: str
) -> ReadbackLayer:
    if failure is not None:
        return _layer(layer_id, "FAILED", failure[:240])
    if identity is None:
        return _layer(
            layer_id, "NOT_APPLICABLE", "this is a development path; no finalization exists"
        )
    return _layer(layer_id, "VERIFIED", f"the {noun} was opened and its bindings agree", identity)


def _release_layer(
    *,
    marker: ReleasedPortfolioArtifacts | None,
    package_hash: str | None,
    failure: str | None,
) -> ReadbackLayer:
    if failure is not None:
        return _layer("released_artifacts", "FAILED", failure[:240])
    if package_hash is None:
        return _layer(
            "released_artifacts",
            "NOT_APPLICABLE",
            "this is a development path; no finalization exists",
        )
    if marker is None:
        return _layer("released_artifacts", "FAILED", "the release marker is absent")
    return _layer(
        "released_artifacts",
        "VERIFIED",
        "the release marker was opened and closes the package, receipt and handoff",
        marker.marker_hash,
    )


def _safely[ArtifactT](
    open_call: Callable[[str], ArtifactT | None], identity: str
) -> ArtifactT | None:
    try:
        return open_call(identity)
    except Exception:
        return None


def _package_failure(
    *,
    package_hash: str | None,
    package: FinalPortfolioEvaluationPackage | None,
    result_hash: str,
    report_hash: str,
) -> str | None:
    if package_hash is None:
        return None
    if package is None:
        return "the protected package is bound but absent from its store"
    if package.package_hash != package_hash:
        return "the store answered with a different package than the one requested"
    if package.protected_result_hash != result_hash:
        return (
            "this package is about another result: readback was asked for "
            f"{result_hash[:16]} and the package released {package.protected_result_hash[:16]}"
        )
    if package.protected_report_hash != report_hash:
        return "the package's released report is not the report this result names"
    return None


def _receipt_failure(
    *,
    receipt_hash: str | None,
    receipt: PortfolioValidationReceipt | None,
    package: FinalPortfolioEvaluationPackage | None,
) -> str | None:
    if receipt_hash is None:
        return None
    if receipt is None:
        return "the validation receipt is bound but absent from its store"
    if receipt.receipt_hash != receipt_hash:
        return "the store answered with a different receipt than the one requested"
    if package is not None and receipt.package_hash != package.package_hash:
        return "the validation receipt closes a different package"
    return None


def _handoff_failure(
    *,
    handoff_hash: str | None,
    handoff: ValidatedPortfolioHandoff | None,
    package: FinalPortfolioEvaluationPackage | None,
    receipt: PortfolioValidationReceipt | None,
) -> str | None:
    if handoff_hash is None:
        return None
    if handoff is None:
        return "the handoff is bound but absent from its store"
    if handoff.handoff_hash != handoff_hash:
        return "the store answered with a different handoff than the one requested"
    if package is not None:
        if handoff.package_hash != package.package_hash:
            return "the handoff releases a different package"
        if handoff.released_report_hash != package.protected_report_hash:
            return "the handoff released a different report than the package sealed"
    if receipt is not None and handoff.validation_receipt_hash != receipt.receipt_hash:
        return "the handoff cites a different validation receipt"
    return None


def strong_replay(
    store: PortfolioLedgerStore,
    *,
    workspace_id: str,
    result_hash: str,
    spec: PortfolioResearchSpec,
    coverage: PortfolioSupportCoverage,
) -> StrongReplayReceipt:
    """Rerun the transition from the sealed inputs and compare every output.

    The rerun consumes only *inputs*: the opening boundary, the capped decision
    targets, the per-name realized outcomes and the availability matrix. It
    drives the same `execute_orders` and `drift_holdings` the walk used, and the
    executed books, cash, drifted state, fills, misses, turnover and gross
    returns all fall out of that. Everything the ledger sealed is then a thing to
    compare against rather than a thing to read.

    Which is the whole distinction. Reading `one_way_turnovers` back and
    reporting that it equals itself is not a replay; neither is counting
    `decision_modes` and calling the count a proof of fills. Nothing sealed as an
    *output* is used as an input here -- not the turnovers, not the modes, not
    the miss count, not the net returns, not one report value.

    The replay is bound to the exact request. A spec that hashes to something
    other than the one the result was sealed under describes a different
    question, and answering it would produce a mismatch that looks like a broken
    path -- so it is refused, by name, before any Backtesting work.

    Nothing refits a model, re-estimates Risk, opens protected evidence or
    publishes a pointer.
    """
    result = store.load_result(result_hash)
    program = store.load_program(result.program_hash)
    assembly = program.numerical_input_assembly_hash
    if assembly is None:
        # Refused before any recomputation: a Program that cannot name its
        # numerical inputs has nothing to compare a rederivation against.
        return _refused_replay(
            workspace_id=workspace_id,
            result_hash=result_hash,
            program=program,
            assembly=None,
            disposition="REFUSED_LEGACY_ASSEMBLY",
            detail="pre-binding Program cannot authorize unit reuse or strong replay",
        )

    ledger = store.load_execution(result.execution_ledger_hash)
    if not ledger.replayable:
        return _refused_replay(
            workspace_id=workspace_id,
            result_hash=result_hash,
            program=program,
            assembly=assembly,
            disposition="REFUSED_UNSEALED_CHILDREN",
            detail="this ledger sealed outputs but not the inputs a rerun consumes",
        )

    sealed = store.load_economics(result.economic_ledger_hash)
    report = store.load_report(result.report_hash)
    sealed_comparison: PortfolioBenchmarkComparison | None = None
    if spec.secondary_benchmark_view != "anchor_only":
        try:
            sealed_comparison = store.load_comparison(report.benchmark_comparison_hash)
        except (PortfolioLedgerStoreError, PortfolioApplicationError, ValueError):
            # A path sealed before comparisons were published cannot supply the
            # secondary series a rebuild needs. That is an unsealed child, and it
            # is named as one rather than reported as a numerical disagreement.
            return _refused_replay(
                workspace_id=workspace_id,
                result_hash=result_hash,
                program=program,
                assembly=assembly,
                disposition="REFUSED_UNSEALED_CHILDREN",
                detail=(
                    "this path requested a secondary comparator and did not seal the "
                    "series a rebuild needs"
                ),
            )
    spec_failure = _spec_failure(spec=spec, result=result, report=report)
    if spec_failure is not None:
        return _refused_replay(
            workspace_id=workspace_id,
            result_hash=result_hash,
            program=program,
            assembly=assembly,
            disposition="REFUSED_SPEC_MISMATCH",
            detail=spec_failure,
        )
    rows, listings = ledger.executed_weights_shape
    executed_lane = _lane(store, "executed-weights", ledger.executed_weights_hash, (rows, listings))
    targets = _lane(store, "target-weights", ledger.target_weights_hash, (rows, listings))
    realized = _lane(
        store, "realized-returns", ledger.realized_simple_returns_hash, (rows, listings)
    )
    available = _bool_lane(store, ledger.execution_available_hash, (rows, listings))
    opening = _opening_state(store, ledger)

    rerun = _rerun_transition(
        opening=opening, targets=targets, realized=realized, available=available
    )
    layers: list[ReadbackLayer] = []
    recomputations = 3

    executed_matches = bool(
        np.array_equal(
            np.ascontiguousarray(rerun.executed, dtype="<f8"),
            np.ascontiguousarray(executed_lane, dtype="<f8"),
        )
    )
    layers.append(
        _layer(
            "portfolio_state",
            "VERIFIED" if executed_matches else "FAILED",
            (
                f"{rows} executed books rerun from the opening boundary and the sealed "
                "targets reproduce the sealed lane bit for bit"
                if executed_matches
                else "the rerun executed books differ from the sealed weight lane"
            ),
            _digest(rerun.executed) if executed_matches else None,
        )
    )

    # Cash is not sealed anywhere, and it does not have to be: the execution
    # owner holds the book and its cash to a unit budget, so a rerun that
    # reproduces the weights and satisfies that identity has reproduced the cash.
    budget_holds = bool(
        np.allclose(rerun.executed.sum(axis=1) + rerun.executed_cash, 1.0, rtol=0.0, atol=1e-9)
    )
    drift_matches = bool(
        np.allclose(rerun.gross, np.asarray(ledger.gross_simple_returns), rtol=0.0, atol=0.0)
    )
    layers.append(
        _layer(
            "drifted_pretrade_state",
            "VERIFIED" if drift_matches and budget_holds else "FAILED",
            (
                "the drifted pre-trade book each turnover is charged against is rerun "
                "through the execution owner, and its gross returns match exactly"
                if drift_matches and budget_holds
                else "the rerun drifted state does not reproduce the sealed gross returns"
            ),
            _digest(rerun.pretrade) if drift_matches and budget_holds else None,
        )
    )

    missed_matches = rerun.missed == ledger.missed_execution_count
    layers.append(
        _layer(
            "fills_and_missed_fills",
            "VERIFIED" if missed_matches else "FAILED",
            (
                f"{int(rerun.filled)} filled and {rerun.missed} missed orders derived from the "
                "availability matrix reproduce the sealed miss count"
                if missed_matches
                else f"the rerun counted {rerun.missed} missed fills against the sealed "
                f"{ledger.missed_execution_count}"
            ),
            str(canonical_hash({"missed": rerun.missed, "filled": int(rerun.filled)}))
            if missed_matches
            else None,
        )
    )

    sealed_turnover: npt.NDArray[np.float64] = np.asarray(
        ledger.one_way_turnovers, dtype=np.float64
    )
    turnover_matches = bool(np.allclose(rerun.turnover, sealed_turnover, rtol=0.0, atol=0.0))
    layers.append(
        _layer(
            "one_way_turnover",
            "VERIFIED" if turnover_matches else "FAILED",
            (
                f"all {rows} turnovers rerun against the drifted book -- not the executed "
                "one -- reproduce the sealed series exactly"
                if turnover_matches
                else "the rerun turnovers differ from the sealed series"
            ),
            _digest(rerun.turnover) if turnover_matches else None,
        )
    )

    boundary = ledger.final_boundary
    boundary_matches = boundary is not None and _boundary_matches(store, boundary, rerun)
    layers.append(
        _layer(
            "terminal_boundary",
            "VERIFIED" if boundary_matches else "FAILED",
            (
                "the state the path ends at -- both books and both cash balances -- is "
                "rerun and matches the sealed boundary"
                if boundary_matches
                else "the rerun terminal state does not match the sealed boundary"
            ),
            boundary.boundary_hash if boundary_matches and boundary is not None else None,
        )
    )

    cost_policy = PortfolioCostPolicy(
        reporting_bps=(int(spec.cost.platform_cost_bps),),
        selection_bps=int(spec.cost.platform_cost_bps),
    )
    net = cost_policy.net_simple_returns(
        gross_simple_returns=rerun.gross,
        one_way_turnovers=rerun.turnover,
        cost_bps=float(spec.cost.platform_cost_bps),
    )
    net_matches = bool(np.allclose(net, np.asarray(sealed.net_simple_returns), rtol=0.0, atol=0.0))
    layers.append(
        _layer(
            "cost_and_net_returns",
            "VERIFIED" if net_matches else "FAILED",
            (
                "net returns from the rerun gross and rerun turnover, through the "
                "Backtesting cost policy, match the sealed ledger bit for bit"
                if net_matches
                else "net returns from the rerun differ from the sealed ledger"
            ),
            _digest(net) if net_matches else None,
        )
    )

    anchor: npt.NDArray[np.float64] = np.asarray(ledger.anchor_simple_returns, dtype=np.float64)
    try:
        beta: float | None = active_path_metrics(portfolio_simple=net, benchmark_simple=anchor).beta
    except ActiveMetricsError:
        beta = None
    beta_matches = beta == sealed.benchmark_beta
    layers.append(
        _layer(
            "benchmark_alignment",
            "VERIFIED" if beta_matches else "FAILED",
            (
                f"anchor beta rederived through the Backtesting owner from the rerun net "
                f"series: {beta}"
                if beta_matches
                else "anchor beta rederived from the rerun net series differs from the ledger"
            ),
            str(canonical_hash({"beta": beta, "anchor": tuple(float(v) for v in anchor)}))
            if beta_matches
            else None,
        )
    )

    wealth = float(np.prod(1.0 + net))
    rederived_identity: dict[str, object] = {
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
    rederived_economic_hash = str(canonical_hash(rederived_identity))
    economic_matches = rederived_economic_hash == sealed.economic_ledger_hash
    layers.append(
        _layer(
            "economic_metrics",
            "VERIFIED" if economic_matches else "FAILED",
            (
                "the economic ledger rebuilt from the rerun hashes to the sealed one"
                if economic_matches
                else "the economic ledger rebuilt from the rerun does not hash to the sealed one"
            ),
            rederived_economic_hash if economic_matches else None,
        )
    )

    # The *whole* declared report, rebuilt through the owner the executor uses,
    # from the rerun series and the spec -- never from the sealed report's own
    # window or its own metric fields. Comparing one number would prove one
    # number; comparing the report identity proves the window, the guards, the
    # end-of-window book, the cap counts, the Risk facts, the liquidity
    # descriptor, the unit rows and the limitations, all at once.
    rederived_report: str | None = None
    report_detail = "the report rebuilt from the rerun hashes to the sealed one"
    try:
        rebuilt = project_declared_path_report(
            spec=spec,
            program=program,
            ledger=ledger,
            economics=sealed,
            comparison=_rebuilt_comparison(
                spec=spec, ledger=ledger, economics=sealed, sealed=sealed_comparison
            ),
            coverage=coverage,
            net_simple_returns=net,
            one_way_turnovers=rerun.turnover,
            executed_weights=rerun.executed,
            # The same sealed boundary the transitions were rerun from. If it
            # could not be rehydrated the replay has already failed closed above,
            # so the report is never rebuilt against an assumed opening.
            opening_reference_weights=opening.optimizer_reference,
        )
        rederived_report = rebuilt.report_hash
    except (PortfolioApplicationError, ValueError) as error:
        report_detail = f"the report could not be rebuilt from the rerun: {error}"[:240]
    recomputations += 1
    report_matches = rederived_report == report.report_hash
    if rederived_report is not None and not report_matches:
        report_detail = "the report rebuilt from the rerun does not hash to the sealed one"
    layers.append(
        _layer(
            "report_metrics",
            "VERIFIED" if report_matches else "FAILED",
            report_detail,
            rederived_report if report_matches else None,
        )
    )

    # Exact means every layer was rerun and matched, and the economic identity
    # landed on the sealed hash. There is no "nothing failed" shortcut: an
    # unproved layer is a hole, and the receipt refuses to be called exact with
    # one in it.
    exact = economic_matches and all(value.disposition == "VERIFIED" for value in layers)
    return StrongReplayReceipt.create(
        workspace_id=workspace_id,
        result_hash=result_hash,
        program_hash=program.program_hash,
        numerical_input_assembly_hash=assembly,
        layers=tuple(layers),
        rederived_economic_ledger_hash=rederived_economic_hash,
        sealed_economic_ledger_hash=sealed.economic_ledger_hash,
        work=ReadbackWork(backtesting_transitions=rows, metric_recomputations=recomputations),
        disposition="REPLAYED_EXACT" if exact else "FAILED_MISMATCH",
    )


@dataclass(frozen=True, slots=True)
class _RerunPath:
    """Everything one rerun of the transition produced, and nothing read back."""

    executed: npt.NDArray[np.float64]
    executed_cash: npt.NDArray[np.float64]
    pretrade: npt.NDArray[np.float64]
    gross: npt.NDArray[np.float64]
    turnover: npt.NDArray[np.float64]
    missed: int
    filled: int
    final_pretrade: npt.NDArray[np.float64]
    final_pretrade_cash: float
    final_reference: npt.NDArray[np.float64]
    final_reference_cash: float


def _rerun_transition(
    *,
    opening: PortfolioWalkForwardState,
    targets: npt.NDArray[np.float64],
    realized: npt.NDArray[np.float64],
    available: npt.NDArray[np.bool_],
) -> _RerunPath:
    """Drive the shared execution owner formation by formation.

    The same two functions the walk-forward segment calls, in the same order,
    against the same inputs. Not a second implementation of the mechanics: if
    this rederived turnover with its own arithmetic, a match would only say the
    two agreed, and a mismatch would not say which one was wrong.
    """

    rows, listings = targets.shape
    weights = np.array(opening.pretrade_weights, dtype=np.float64)
    cash = float(opening.pretrade_cash)
    executed_rows: list[npt.NDArray[np.float64]] = []
    pretrade_rows: list[npt.NDArray[np.float64]] = []
    executed_cash = np.empty(rows, dtype=np.float64)
    gross = np.empty(rows, dtype=np.float64)
    turnover = np.empty(rows, dtype=np.float64)
    missed = 0
    filled = 0
    reference_cash = float(opening.optimizer_reference_cash)
    reference = np.array(opening.optimizer_reference, dtype=np.float64)
    for index in range(rows):
        pretrade_rows.append(np.array(weights, copy=True))
        executed, cash, one_way, missed_here = execute_orders(
            pretrade_weights=weights,
            pretrade_cash=cash,
            target_weights=targets[index],
            execution_available=available[index],
        )
        filled += int(
            np.count_nonzero((np.abs(targets[index] - weights) > TOLERANCE) & available[index])
        )
        executed_rows.append(np.array(executed, copy=True))
        executed_cash[index] = cash
        turnover[index] = one_way
        missed += missed_here
        reference = np.array(executed, copy=True)
        reference_cash = cash
        weights, cash, gross[index] = drift_holdings(
            weights=executed, cash=cash, returns=realized[index]
        )
    del listings
    return _RerunPath(
        executed=np.vstack(executed_rows),
        executed_cash=executed_cash,
        pretrade=np.vstack(pretrade_rows),
        gross=gross,
        turnover=turnover,
        missed=missed,
        filled=filled,
        final_pretrade=weights,
        final_pretrade_cash=cash,
        final_reference=reference,
        final_reference_cash=reference_cash,
    )


def _boundary_matches(
    store: PortfolioLedgerStore, boundary: SealedPortfolioBoundaryState, rerun: _RerunPath
) -> bool:
    """Whether the rerun ends where the ledger says the path ended.

    Both books and both cash balances. The reference alone would pass on a path
    whose *held* book had drifted somewhere else entirely, and that book is the
    one a continuation prices its opening trade against.
    """

    try:
        sealed_pretrade = _lane(
            store, "boundary-weights", boundary.pretrade_weights_hash, (rerun.final_pretrade.size,)
        )
        sealed_reference = _lane(
            store,
            "boundary-weights",
            boundary.optimizer_reference_hash,
            (rerun.final_reference.size,),
        )
    except (PortfolioLedgerStoreError, ReadbackError):
        return False
    return (
        bool(np.array_equal(sealed_pretrade, rerun.final_pretrade))
        and bool(np.array_equal(sealed_reference, rerun.final_reference))
        and boundary.pretrade_cash == rerun.final_pretrade_cash
        and boundary.optimizer_reference_cash == rerun.final_reference_cash
    )


def _opening_state(
    store: PortfolioLedgerStore, ledger: PortfolioExecutionLedger
) -> PortfolioWalkForwardState:
    """Rehydrate the boundary the path started from, whatever it was.

    A development path opens flat and a continuation opens on a frozen book, and
    a replay that assumed the first would silently rederive a different path for
    every one of the second.
    """

    boundary = ledger.initial_boundary
    if boundary is None:
        raise ReadbackError("portfolio_readback.ledger_sealed_no_opening_boundary")
    size = (len(ledger.ordered_listing_ids),)
    try:
        # Through the store's own reader, so the book a replay reruns from and
        # the book a report compares that same replay against cannot be two
        # different arrays.
        reference = store.load_opening_reference(ledger)
    except PortfolioLedgerStoreError as error:
        # Fail closed. Without the sealed opening book there is no honest way to
        # rerun the first transition or to say what the first formation changed,
        # and assuming a flat one is the misreport this Gate closed.
        raise ReadbackError("portfolio_readback.sealed_opening_boundary_unreadable") from error
    return PortfolioWalkForwardState(
        pretrade_weights=_lane(store, "boundary-weights", boundary.pretrade_weights_hash, size),
        pretrade_cash=boundary.pretrade_cash,
        optimizer_reference=reference,
        optimizer_reference_cash=boundary.optimizer_reference_cash,
    )


def _lane(
    store: PortfolioLedgerStore,
    category: str,
    content_hash: str | None,
    shape: tuple[int, ...],
) -> npt.NDArray[np.float64]:
    if content_hash is None:
        raise ReadbackError("portfolio_readback.sealed_lane_absent:" + category)
    payload = store.load_lane(category=category, content_hash=content_hash)
    values: npt.NDArray[np.float64] = np.frombuffer(payload, dtype="<f8")
    if values.size != int(np.prod(shape)):
        raise ReadbackError("portfolio_readback.sealed_lane_axis_invalid:" + category)
    return values.reshape(shape).astype(np.float64, copy=True)


def _bool_lane(
    store: PortfolioLedgerStore, content_hash: str | None, shape: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    if content_hash is None:
        raise ReadbackError("portfolio_readback.sealed_lane_absent:execution-available")
    payload = store.load_lane(category="execution-available", content_hash=content_hash)
    values: npt.NDArray[np.uint8] = np.frombuffer(payload, dtype=np.uint8)
    if values.size != shape[0] * shape[1]:
        raise ReadbackError("portfolio_readback.sealed_lane_axis_invalid:execution-available")
    return values.reshape(shape).astype(bool)


def _digest(values: npt.NDArray[np.float64]) -> str:
    return str(canonical_hash(np.ascontiguousarray(values, dtype="<f8").tobytes().hex()))


def _spec_failure(
    *,
    spec: PortfolioResearchSpec,
    result: PortfolioResearchResult,
    report: PortfolioDeclaredPathReport,
) -> str | None:
    """Whether the spec in hand is the one this result was sealed under.

    Two checks rather than one. The spec hash says the whole request matches;
    the control receipt says the *installed control values* match, and it is
    sealed inside the report by the run itself. A caller who reconstructed a spec
    that happens to hash correctly but resolves different controls would pass the
    first and fail the second, which is the case worth naming.
    """

    if spec.spec_hash != result.spec_hash:
        return (
            "this replay was handed a different request than the result was sealed under: "
            f"{spec.spec_hash[:16]} against {result.spec_hash[:16]}"
        )
    if PortfolioControlReceipt.of(spec) != report.control_receipt:
        return "the spec's control values differ from the receipt sealed beside the report"
    return None


def _rebuilt_comparison(
    *,
    spec: PortfolioResearchSpec,
    ledger: PortfolioExecutionLedger,
    economics: PortfolioEconomicLedger,
    sealed: PortfolioBenchmarkComparison | None,
) -> PortfolioBenchmarkComparison:
    """Rebuild the comparison the report binds, from inputs rather than outputs.

    The anchor comes off the execution ledger. The secondary comparator is a
    series the run resolved from outside the path -- it cannot be rederived from
    fills any more than the anchor can, and it is reopened from its own sealed
    artifact for exactly the same reason.

    What is taken from that artifact is only the *inputs*: which comparator, its
    series, and the disposition the resolver reported. Everything that makes the
    comparison an identity -- the ledger it belongs to, the overlay it was
    computed against -- is supplied from the rerun, so the rebuilt hash is a
    rederivation and not a copy.
    """

    if spec.secondary_benchmark_view == "anchor_only":
        return PortfolioBenchmarkComparison.create(
            execution_ledger_hash=ledger.ledger_hash,
            economic_ledger_hash=economics.economic_ledger_hash,
            secondary_benchmark_view=spec.secondary_benchmark_view,
            primary_simple_returns=tuple(float(value) for value in ledger.anchor_simple_returns),
            secondary_benchmark_id=None,
            secondary_simple_returns=None,
            secondary_disposition="SECONDARY_BENCHMARK_NOT_REQUESTED",
        )
    if sealed is None:
        raise ValueError("portfolio_readback.secondary_benchmark_series_is_not_a_sealed_child")
    return PortfolioBenchmarkComparison.create(
        execution_ledger_hash=ledger.ledger_hash,
        economic_ledger_hash=economics.economic_ledger_hash,
        secondary_benchmark_view=spec.secondary_benchmark_view,
        primary_simple_returns=tuple(float(value) for value in ledger.anchor_simple_returns),
        secondary_benchmark_id=sealed.secondary_benchmark_id,
        secondary_simple_returns=sealed.secondary_simple_returns,
        secondary_disposition=sealed.secondary_disposition,
    )


def _refused_replay(
    *,
    workspace_id: str,
    result_hash: str,
    program: PortfolioExecutionProgram,
    assembly: str | None,
    disposition: Literal[
        "REFUSED_LEGACY_ASSEMBLY", "REFUSED_UNSEALED_CHILDREN", "REFUSED_SPEC_MISMATCH"
    ],
    detail: str,
) -> StrongReplayReceipt:
    """Refuse before computing anything, and say which of the two reasons it was."""

    return StrongReplayReceipt.create(
        workspace_id=workspace_id,
        result_hash=result_hash,
        program_hash=program.program_hash,
        numerical_input_assembly_hash=assembly,
        layers=tuple(_layer(layer_id, "LEGACY", detail) for layer_id in REPLAY_LAYERS),
        work=ReadbackWork(),
        disposition=disposition,
    )


__all__ = [
    "READBACK_LAYERS",
    "REPLAY_LAYERS",
    "ExactReadbackReceipt",
    "FinalizationArtifactOpener",
    "OpenedUpstreamArtifact",
    "ReadbackDisposition",
    "ReadbackError",
    "ReadbackLayer",
    "ReadbackWork",
    "StrongReplayReceipt",
    "UpstreamEvidenceOpener",
    "exact_readback",
    "strong_replay",
]
