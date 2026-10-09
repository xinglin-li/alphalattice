"""Sole active public Portfolio ledger and exact readback store."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from alphalattice.control.workspace_runtime.content_store import (
    CommittedIndex,
    CommittedKind,
    ContentAddressedStore,
    ContentAddressedStoreError,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioBenchmarkComparison,
    PortfolioDeclaredPathReport,
    PortfolioEconomicLedger,
    PortfolioExecutionLedger,
    PortfolioExecutionProgram,
    PortfolioResearchResult,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

if TYPE_CHECKING:
    from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
        PortfolioDecisionCheckpoint,
        PortfolioUpdatePublication,
    )
    from alphalattice.investment.portfolio_strategy_lab.publication.rolling_history import (
        RollingReportHead,
    )


class PortfolioLedgerStoreError(ValueError):
    """Stable public ledger storage refusal."""


PUBLIC_NAMESPACE = "public"
PENDING_FINALIZATION_NAMESPACE = "pending-finalization"
"""Where a protected continuation publishes before anyone has closed it.

A separate physical root, not a flag on a record. Invisibility enforced by a
field would mean every ordinary lookup had to remember to check it, and the one
that forgot would be the one that leaked a protected report.
"""

# The dtype a lane category is packed in; every other lane is little-endian float64.
_LANE_DTYPES: dict[str, str] = {"execution-available": "u1"}


def _require_reusable(program: PortfolioExecutionProgram) -> PortfolioExecutionProgram:
    """A Program that cannot name its numerical inputs may not be reused.

    Enforced at every reuse *lookup*, not only at the executor, because a lookup
    is what a reuse decision is made of: the answer "yes, this already exists"
    is the whole authorization. A caller who wants to read such a path back opens
    it by identity through `load_program` and friends, which stay open.

    Refused rather than reported absent. Absence would say "run it again", and a
    predecessor path is not something this build knows how to reproduce; the
    honest answer names the missing binding.
    """

    if not program.replayable:
        raise PortfolioLedgerStoreError("portfolio_application.program_assembly_absent_no_reuse")
    return program


class PortfolioLedgerStore:
    """Content-addressed descendants plus one by-program reuse index.

    The namespace chooses which root the artifacts land in. Both roots speak the
    same URI space on purpose: a URI names the artifact, not the drawer it is
    waiting in, so the artifacts a release promotes into `public` are the exact
    bytes that were sealed -- promotion copies, it does not re-seal. Rewriting
    URIs at promotion would change the result identity and make the released
    artifact a different artifact from the validated one.
    """

    def __init__(
        self,
        artifact_root: Path,
        *,
        namespace: str = PUBLIC_NAMESPACE,
        capacity: Callable[[int], None] | None = None,
    ) -> None:
        """Bind explicit public or pending-finalization ledger namespace and storage capacity.

        Args:
            artifact_root: Caller-owned artifact directory.
            namespace: Admitted public or pending-finalization namespace.
            capacity: Optional byte-count admission callback supplied to storage.

        Raises:
            PortfolioLedgerStoreError: Namespace is not declared.
        """
        if namespace not in {PUBLIC_NAMESPACE, PENDING_FINALIZATION_NAMESPACE}:
            raise PortfolioLedgerStoreError("portfolio_application.ledger_namespace_unknown")
        self.artifact_root = artifact_root.resolve()
        self.namespace = namespace
        self.root = self.artifact_root / "portfolio-strategy-lab" / namespace
        self.content = ContentAddressedStore(
            self.root,
            uri_prefix="playpen://portfolio-strategy-lab/public",
            capacity=capacity,
        )

    @classmethod
    def for_workspace(
        cls, workspace: Path, *, capacity: Callable[[int], None] | None = None
    ) -> PortfolioLedgerStore:
        """The public product root, shared by its composition and QA admission."""
        return cls(workspace.resolve() / "runtime" / "artifacts", capacity=capacity)

    @property
    def is_public(self) -> bool:
        """Read whether this ledger uses the public namespace.

        Returns:
            True exactly for PUBLIC_NAMESPACE.
        """
        return self.namespace == PUBLIC_NAMESPACE

    def publish_program(self, value: PortfolioExecutionProgram) -> str:
        """Publish exact sealed program evidence.

        Args:
            value: Validated sealed concrete evidence.

        Returns:
            Content-addressed evidence URI.
        """
        return self.content.publish_model(
            category="programs", value=value, identity_field="program_hash"
        )

    def publish_decision_checkpoint(self, value: PortfolioDecisionCheckpoint) -> str:
        """Publish exact sealed decision checkpoint evidence.

        Args:
            value: Validated sealed concrete evidence.

        Returns:
            Content-addressed evidence URI.
        """
        return self.content.publish_model(
            category="decision-checkpoints", value=value, identity_field="content_hash"
        )

    def load_decision_checkpoint(self, content_hash: str) -> PortfolioDecisionCheckpoint:
        """Reopen exact sealed decision checkpoint evidence.

        Args:
            content_hash: Exact retained content identity.

        Returns:
            Validated concrete artifact reopened through content-addressed storage.
        """
        from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
            PortfolioDecisionCheckpoint,
        )

        return self.content.load_model(
            category="decision-checkpoints",
            content_hash=content_hash,
            model=PortfolioDecisionCheckpoint,
            identity_field="content_hash",
        )

    def _update_key(self, checkpoint_hash: str, parent_hash: str | None) -> str:
        self.content.require_hash(checkpoint_hash)
        if parent_hash is not None:
            self.content.require_hash(parent_hash)
        return str(canonical_hash({"checkpoint": checkpoint_hash, "parent": parent_hash}))

    def _successors(self) -> CommittedKind[PortfolioUpdatePublication]:
        from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
            PortfolioUpdatePublication,
        )

        return CommittedKind(
            "decision-successors", "decision-updates", PortfolioUpdatePublication, "content_hash"
        )

    def decision_successor(
        self, checkpoint_hash: str, parent_hash: str | None
    ) -> PortfolioUpdatePublication | None:
        """Reopen the committed successor for an exact checkpoint and parent.

        Args:
            checkpoint_hash: Exact retained decision checkpoint.
            parent_hash: Exact predecessor publication or None at history origin.

        Returns:
            Validated committed successor, or None.

        Raises:
            PortfolioLedgerStoreError: Reopened checkpoint/parent binding differs.
        """
        value = CommittedIndex(self.root, self.content).open(
            self._successors(), self._update_key(checkpoint_hash, parent_hash)
        )
        if value is not None and (
            value.checkpoint_hash != checkpoint_hash or value.parent_hash != parent_hash
        ):
            raise PortfolioLedgerStoreError("portfolio_update.successor_binding_invalid")
        return value

    def publish_decision_update(self, value: PortfolioUpdatePublication) -> str:
        """Commit a public decision update after reopening its exact HTML artifact.

        Args:
            value: Sealed update publication with declared report identity.

        Returns:
            Committed update URI under exact checkpoint/parent key.

        Raises:
            PortfolioLedgerStoreError: Namespace is not public or HTML/report is absent; storage
                also refuses conflicting committed keys.
        """
        if not self.is_public:
            raise PortfolioLedgerStoreError("portfolio_update.qa_namespace_invalid")
        if value.html_hash is None:
            raise PortfolioLedgerStoreError("portfolio_update.report_missing")
        self.load_html(value.html_hash)
        return CommittedIndex(self.root, self.content).commit(
            self._successors(), self._update_key(value.checkpoint_hash, value.parent_hash), value
        )

    def decision_history(self, checkpoint_hash: str) -> tuple[PortfolioUpdatePublication, ...]:
        """Follow committed decision successors from history origin and refuse cycles.

        Args:
            checkpoint_hash: Exact retained decision checkpoint.

        Returns:
            Ordered successor publication tuple.

        Raises:
            PortfolioLedgerStoreError: A publication repeats or a successor binding differs.
        """
        found: list[PortfolioUpdatePublication] = []
        seen: set[str] = set()
        parent = None
        while (value := self.decision_successor(checkpoint_hash, parent)) is not None:
            if value.content_hash in seen:
                raise PortfolioLedgerStoreError("portfolio_update.history_cycle")
            seen.add(value.content_hash)
            found.append(value)
            parent = value.content_hash
        return tuple(found)

    def publish_rolling_report_head(self, value: RollingReportHead) -> str:
        """Persist one compact link to an existing REPORT and update publication.

        The head contains only references and incremental outcome metadata. Reopening
        its children here prevents a durable chain from naming an artifact this store
        does not actually retain.
        """
        from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
            PortfolioUpdatePublication,
        )
        from alphalattice.investment.portfolio_strategy_lab.publication.rolling_history import (
            RollingReportHead as Head,
        )

        if not self.is_public:
            raise PortfolioLedgerStoreError("portfolio_rolling.namespace_not_public")
        if not isinstance(value, Head):
            raise PortfolioLedgerStoreError("portfolio_rolling.head_contract_invalid")
        report = self.load_report(value.report_hash)
        result = self.load_result(value.result_hash)
        publication = self.content.load_model(
            category="decision-updates",
            content_hash=value.increment_publication_hash,
            model=PortfolioUpdatePublication,
            identity_field="content_hash",
        )
        outcomes = tuple(
            event
            for event in publication.events
            if event.phase == "OUTCOME_SETTLED" and event.formation_session > value.formation_end
        )
        if (
            result.report_hash != report.report_hash
            or result.program_hash != value.program_hash
            or report.program_hash != value.program_hash
            or publication.checkpoint_hash != value.checkpoint_hash
            or publication.parent_hash != value.previous_publication_hash
            or canonical_hash(tuple(event.content_hash for event in outcomes))
            != value.increment_outcome_hash
            or len(outcomes) != value.increment_outcome_count
            or (None if not outcomes else outcomes[0].formation_session)
            != value.first_increment_formation
            or (None if not outcomes else outcomes[-1].formation_session)
            != value.last_increment_formation
        ):
            raise PortfolioLedgerStoreError("portfolio_rolling.head_reference_binding_invalid")
        if value.previous_publication_hash is not None:
            prior_publication = self.content.load_model(
                category="decision-updates",
                content_hash=value.previous_publication_hash,
                model=PortfolioUpdatePublication,
                identity_field="content_hash",
            )
            if prior_publication.checkpoint_hash != value.checkpoint_hash:
                raise PortfolioLedgerStoreError(
                    "portfolio_rolling.previous_publication_binding_invalid"
                )
        if value.previous_head_hash is not None:
            previous = self.load_rolling_report_head(value.previous_head_hash)
            if previous.increment_publication_hash != value.previous_publication_hash or any(
                getattr(previous, field) != getattr(value, field)
                for field in (
                    "report_hash",
                    "result_hash",
                    "program_hash",
                    "checkpoint_hash",
                    "strategy_package_id",
                    "strategy_package_hash",
                    "cost_bps_per_side",
                    "formation_start",
                    "formation_end",
                    "baseline_observed_through",
                    "baseline_claim",
                )
            ):
                raise PortfolioLedgerStoreError("portfolio_rolling.previous_head_binding_invalid")
        return self.content.publish_model(
            category="rolling-report-heads", value=value, identity_field="head_hash"
        )

    def load_rolling_report_head(self, head_hash: str) -> RollingReportHead:
        """Reopen one compact, self-hashed rolling history link by identity."""
        from alphalattice.investment.portfolio_strategy_lab.publication.rolling_history import (
            RollingReportHead,
        )

        return self.content.load_model(
            category="rolling-report-heads",
            content_hash=head_hash,
            model=RollingReportHead,
            identity_field="head_hash",
        )

    def has_rolling_report_head(self, head_hash: str) -> bool:
        """Check for a head file without opening it; `load` proves its contents."""
        self.content.require_hash(head_hash)
        return (self.root / "rolling-report-heads" / f"{head_hash}.json").is_file()

    # ----------------------------------------------------------- index entries
    #
    # Four indices answer four different questions, and the two methods below
    # are everything they share. An entry is a small JSON object filed under a
    # key it must restate: written once and never rewritten, and read back only
    # if it still says what it is filed under. Each caller keeps its own key
    # derivation, its own entry fields, its own refusal names and its own
    # post-read validation, because those are the parts that differ.

    def _write_index_entry(self, path: Path, entry: dict[str, str], *, conflict: str) -> None:
        """File one entry immutably; a different entry under the key is refused.

        The serialization belongs here rather than at each caller: these bytes
        are the stored artifact, so `sort_keys` and the separators are part of
        the identity, not a formatting preference.
        """

        payload = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
        if path.is_file() and path.read_bytes() != payload:
            raise PortfolioLedgerStoreError(conflict)
        if not path.is_file():
            self.content.atomic_write(path, payload)

    def _read_index_entry(
        self, path: Path, *, declared: dict[str, str], tampered: str
    ) -> dict[str, str] | None:
        """Open one entry and require it to restate the key it is filed under.

        Absent is not a refusal: nothing is indexed and the caller is about to
        do the work. Present but inconsistent is, because an entry that cannot
        name its own key cannot be shown to be about the question being asked.
        The artifact the entry points at is loaded by the caller, which is where
        the per-index cross-checks live.
        """

        if not path.is_file():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            if any(entry[field] != value for field, value in declared.items()):
                raise ValueError
            return {str(field): str(value) for field, value in entry.items()}
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            raise PortfolioLedgerStoreError(tampered) from error

    def _reuse_guard(self, program_hash: str) -> None:
        """Refuse to reuse anything published under a Program with no assembly.

        A Program that is not published at all is not a refusal: there is simply
        nothing to reuse, and the caller is about to run. The refusal is for the
        case that matters -- a stored path exists, and it cannot say which inputs
        produced it, so "already done" is a claim nobody can check.
        """

        try:
            program = self.load_program(program_hash)
        except ContentAddressedStoreError as error:
            if str(error).partition(":")[0] == "content_store.artifact_missing":
                return
            raise
        _require_reusable(program)

    def find_execution_for_program(self, program_hash: str) -> PortfolioExecutionLedger | None:
        """The materialized path for one program, if it has already been run.

        A reuse lookup, so it refuses a Program with no numerical-input assembly
        before it opens the index.

        This is what makes "cost reuses the execution ledger" a statement about
        work rather than only about a hash: a descendant change reads the stored
        path instead of walking forward to recompute an identical one.
        """
        tampered = "portfolio_application.execution_index_tampered"
        self._reuse_guard(program_hash)
        entry = self._read_index_entry(
            self._execution_index(program_hash),
            declared={"program_hash": program_hash},
            tampered=tampered,
        )
        if entry is None:
            return None
        try:
            return self.load_execution(entry["ledger_hash"])
        except (ContentAddressedStoreError, KeyError) as error:
            if (
                isinstance(error, ContentAddressedStoreError)
                and str(error).partition(":")[0] == "content_store.artifact_missing"
            ):
                raise
            raise PortfolioLedgerStoreError(tampered) from error

    def publish_program_index(
        self, *, holdings_spec_hash: str, authorities_hash: str, program_hash: str
    ) -> None:
        """The program a holdings configuration compiles to over these authorities.

        Compiling needs the resolved formation axis, which is numerical work. A
        descendant control changes nothing the program binds, so it should not
        have to resolve just to rediscover a program that already exists.
        """
        program = self.load_program(program_hash)
        if (
            program.holdings_spec_hash != holdings_spec_hash
            or program.authorities_hash != authorities_hash
        ):
            raise PortfolioLedgerStoreError("portfolio_application.program_index_binding_invalid")
        self._write_index_entry(
            self._program_index(
                holdings_spec_hash=holdings_spec_hash, authorities_hash=authorities_hash
            ),
            {
                "holdings_spec_hash": holdings_spec_hash,
                "authorities_hash": authorities_hash,
                "program_hash": program_hash,
            },
            conflict="portfolio_application.program_identity_reused",
        )

    def _program_index(self, *, holdings_spec_hash: str, authorities_hash: str) -> Path:
        self.content.require_hash(holdings_spec_hash)
        self.content.require_hash(authorities_hash)
        key = canonical_hash(
            {"holdings_spec_hash": holdings_spec_hash, "authorities_hash": authorities_hash}
        )
        return self.root / "by-holdings" / f"{key}.json"

    def find_program_for(
        self, *, holdings_spec_hash: str, authorities_hash: str
    ) -> PortfolioExecutionProgram | None:
        """Reopen a reusable program selected by exact holdings-spec and authority bindings.

        Args:
            holdings_spec_hash: Exact admitted holdings controls.
            authorities_hash: Exact shared authority binding.

        Returns:
            Reusable indexed program, or None without an index.

        Raises:
            PortfolioLedgerStoreError: Index/artifact bindings are invalid or program reuse is
                refused.
        """
        tampered = "portfolio_application.program_index_tampered"
        entry = self._read_index_entry(
            self._program_index(
                holdings_spec_hash=holdings_spec_hash, authorities_hash=authorities_hash
            ),
            declared={
                "holdings_spec_hash": holdings_spec_hash,
                "authorities_hash": authorities_hash,
            },
            tampered=tampered,
        )
        if entry is None:
            return None
        try:
            program = self.load_program(entry["program_hash"])
            if (
                program.holdings_spec_hash != holdings_spec_hash
                or program.authorities_hash != authorities_hash
            ):
                raise ValueError
        except (ContentAddressedStoreError, KeyError, ValueError) as error:
            if (
                isinstance(error, ContentAddressedStoreError)
                and str(error).partition(":")[0] == "content_store.artifact_missing"
            ):
                raise
            raise PortfolioLedgerStoreError(tampered) from error
        # Outside the `except`, so the refusal keeps its own name rather than
        # being re-reported as a tampered index.
        return _require_reusable(program)

    def _execution_index(self, program_hash: str) -> Path:
        self.content.require_hash(program_hash)
        return self.root / "by-program-execution" / f"{program_hash}.json"

    def publish_execution(self, value: PortfolioExecutionLedger) -> str:
        """Publish exact sealed execution evidence.

        Args:
            value: Validated sealed concrete evidence.

        Returns:
            Content-addressed evidence URI. A program-to-execution index refuses contradictory
            materialized paths.
        """
        uri = self.content.publish_model(
            category="execution-ledgers", value=value, identity_field="ledger_hash"
        )
        # One program, one materialized path. Two different ledgers under one
        # program would mean the program did not determine the fills.
        self._write_index_entry(
            self._execution_index(value.program_hash),
            {"program_hash": value.program_hash, "ledger_hash": value.ledger_hash},
            conflict="portfolio_application.execution_identity_reused",
        )
        return uri

    def publish_economics(self, value: PortfolioEconomicLedger) -> str:
        """Publish exact sealed economics evidence.

        Args:
            value: Validated sealed concrete evidence.

        Returns:
            Content-addressed evidence URI.
        """
        return self.content.publish_model(
            category="economic-ledgers",
            value=value,
            identity_field="economic_ledger_hash",
        )

    def publish_comparison(self, value: PortfolioBenchmarkComparison) -> str:
        """Seal the benchmark comparison the report binds.

        Published rather than left as a hash on the report, because the
        secondary series is an *input* the run resolved from outside the path.
        It cannot be rederived from fills, so a replay that has to reproduce the
        report identity has to be able to reopen it -- exactly the standing the
        eligible-universe anchor already has on the execution ledger.
        """
        return self.content.publish_model(
            category="benchmark-comparisons", value=value, identity_field="comparison_hash"
        )

    def load_comparison(self, comparison_hash: str) -> PortfolioBenchmarkComparison:
        """Reopen an exact sealed benchmark comparison.

        Args:
            comparison_hash: Exact benchmark comparison identity.

        Returns:
            Validated benchmark comparison.
        """
        return self.content.load_model(
            category="benchmark-comparisons",
            content_hash=comparison_hash,
            model=PortfolioBenchmarkComparison,
            identity_field="comparison_hash",
        )

    def publish_report(self, value: PortfolioDeclaredPathReport) -> str:
        """Publish exact sealed report evidence.

        Args:
            value: Validated sealed concrete evidence.

        Returns:
            Content-addressed evidence URI.
        """
        return self.content.publish_model(
            category="reports", value=value, identity_field="report_hash"
        )

    def publish_result(self, value: PortfolioResearchResult) -> str:
        """Publish exact sealed result evidence.

        Args:
            value: Validated sealed concrete evidence.

        Returns:
            Content-addressed evidence URI. A program/spec-to-result index refuses contradictory
            request identities.
        """
        uri = self.content.publish_model(
            category="results", value=value, identity_field="result_hash"
        )
        self._write_index_entry(
            self._request_index(program_hash=value.program_hash, spec_hash=value.spec_hash),
            {
                "program_hash": value.program_hash,
                "spec_hash": value.spec_hash,
                "result_hash": value.result_hash,
            },
            conflict="portfolio_application.request_identity_reused",
        )
        return uri

    def publish_plan_index(
        self, *, spec_hash: str, authorities_hash: str, result_hash: str
    ) -> None:
        """The one lookup `PLAN` can answer without resolving anything numerical.

        `PLAN` knows the request and the authorities it resolved; it cannot know
        the program hash, because the formation axis depends on the rank-mu
        warm-up and that is numerical work `PLAN` must not do. This index closes
        exactly that gap: same request, same authorities, already published.
        """
        result = self.load_result(result_hash)
        program = self.load_program(result.program_hash)
        if result.spec_hash != spec_hash or program.authorities_hash != authorities_hash:
            raise PortfolioLedgerStoreError("portfolio_application.plan_index_binding_invalid")
        self._write_index_entry(
            self._plan_index(spec_hash=spec_hash, authorities_hash=authorities_hash),
            {
                "spec_hash": spec_hash,
                "authorities_hash": authorities_hash,
                "result_hash": result_hash,
            },
            conflict="portfolio_application.plan_identity_reused",
        )

    def _plan_index(self, *, spec_hash: str, authorities_hash: str) -> Path:
        self.content.require_hash(spec_hash)
        self.content.require_hash(authorities_hash)
        key = canonical_hash({"spec_hash": spec_hash, "authorities_hash": authorities_hash})
        return self.root / "by-plan" / f"{key}.json"

    def find_planned_result(
        self, *, spec_hash: str, authorities_hash: str
    ) -> PortfolioResearchResult | None:
        """Reopen a planned result and require its program's exact reusable authority.

        Args:
            spec_hash: Exact admitted research spec.
            authorities_hash: Exact shared authority binding.

        Returns:
            Reusable indexed result, or None without an index.

        Raises:
            PortfolioLedgerStoreError: Result/spec/program authority differs or program reuse is
                refused.
        """
        tampered = "portfolio_application.result_index_tampered"
        entry = self._read_index_entry(
            self._plan_index(spec_hash=spec_hash, authorities_hash=authorities_hash),
            declared={"spec_hash": spec_hash, "authorities_hash": authorities_hash},
            tampered=tampered,
        )
        if entry is None:
            return None
        try:
            result = self.load_result(entry["result_hash"])
            program = self.load_program(result.program_hash)
            if result.spec_hash != spec_hash or program.authorities_hash != authorities_hash:
                raise ValueError
        except (ContentAddressedStoreError, KeyError, ValueError) as error:
            if (
                isinstance(error, ContentAddressedStoreError)
                and str(error).partition(":")[0] == "content_store.artifact_missing"
            ):
                raise
            raise PortfolioLedgerStoreError(tampered) from error
        # Outside the `except`, so the refusal keeps its own name instead of
        # being re-reported as a tampered index.
        _require_reusable(program)
        return result

    def _request_index(self, *, program_hash: str, spec_hash: str) -> Path:
        """One index entry per request, not per program.

        Two specs differing only in cost, benchmark view, report unit or study
        window share a program and an execution ledger, so a program-keyed index
        would hand the second request the first one's report.
        """

        self.content.require_hash(program_hash)
        self.content.require_hash(spec_hash)
        key = canonical_hash({"program_hash": program_hash, "spec_hash": spec_hash})
        return self.root / "by-request" / f"{key}.json"

    def find_result_for(
        self, *, program_hash: str, spec_hash: str
    ) -> PortfolioResearchResult | None:
        """REUSE LOOKUP: the exact answer to this request, if it already exists.

        Refuses a Program with no numerical-input assembly before it opens the
        index, for the same reason as every other reuse route: saying "already
        done" is the authorization, and a path that cannot name its inputs cannot
        be shown to be the answer to anything.
        """
        tampered = "portfolio_application.result_index_tampered"
        self._reuse_guard(program_hash)
        entry = self._read_index_entry(
            self._request_index(program_hash=program_hash, spec_hash=spec_hash),
            declared={"program_hash": program_hash, "spec_hash": spec_hash},
            tampered=tampered,
        )
        if entry is None:
            return None
        try:
            return self.load_result(entry["result_hash"])
        except (ContentAddressedStoreError, KeyError) as error:
            if (
                isinstance(error, ContentAddressedStoreError)
                and str(error).partition(":")[0] == "content_store.artifact_missing"
            ):
                raise
            raise PortfolioLedgerStoreError(tampered) from error

    def publish_lane(self, *, category: str, values: npt.NDArray[Any]) -> str:
        """Publish one packed numerical value column in its declared lane category.

        Args:
            category: Registered lane category.
            values: Exact numerical array.

        Returns:
            Packed-column content identity.
        """
        return self.content.publish_columns(category=f"lanes/{category}", columns={"value": values})

    def load_lane(self, *, category: str, content_hash: str) -> bytes:
        """Reopen exact packed numerical lane bytes.

        Args:
            category: Registered lane category.
            content_hash: Exact packed-lane identity.

        Returns:
            Verified packed bytes; this method does not reinterpret the dtype.
        """
        return self.content.load_packed_bytes(
            category=f"lanes/{category}", content_hash=content_hash
        )

    def load_opening_reference(self, ledger: PortfolioExecutionLedger) -> npt.NDArray[np.float64]:
        """The exact book this path opened on, from its own sealed boundary lane.

        One owner, because two readers need it and they must not disagree: the
        executor reports the first formation's change against it, and a strong
        replay reruns the walk from it. Inferring it from the row position was
        the defect -- row zero is not evidence of an empty book, and a
        continuation opens on a frozen one.

        Fails closed. A ledger with no sealed opening boundary, or a lane whose
        length is not this path's listing axis, has nothing this can honestly
        return; refusing is the only answer that does not invent a book.
        """
        boundary = ledger.initial_boundary
        if boundary is None:
            raise PortfolioLedgerStoreError("portfolio_application.ledger_opening_boundary_absent")
        try:
            payload = self.load_lane(
                category="boundary-weights", content_hash=boundary.optimizer_reference_hash
            )
        except ContentAddressedStoreError as error:
            # One type at this boundary. A caller deciding whether it can report
            # or replay should not have to know which store layer went missing.
            raise PortfolioLedgerStoreError(
                "portfolio_application.ledger_opening_boundary_unreadable"
            ) from error
        values: npt.NDArray[np.float64] = np.frombuffer(payload, dtype="<f8")
        if values.size != len(ledger.ordered_listing_ids):
            raise PortfolioLedgerStoreError(
                "portfolio_application.ledger_opening_boundary_axis_invalid"
            )
        return np.ascontiguousarray(values, dtype=np.float64)

    def publish_html(self, payload: str) -> tuple[str, str]:
        """Publish UTF-8 report HTML as a content-addressed document.

        Args:
            payload: Exact HTML report text.

        Returns:
            Document content hash and registered URI.
        """
        content_hash = self.content.publish_document(
            category="html", payload=payload.encode("utf-8"), extension="html"
        )
        return content_hash, self.content.uri("html", content_hash, extension="html")

    def load_execution(self, ledger_hash: str) -> PortfolioExecutionLedger:
        """Reopen exact sealed execution evidence.

        Args:
            ledger_hash: Exact retained content identity.

        Returns:
            Validated concrete artifact reopened through content-addressed storage.
        """
        return self.content.load_model(
            category="execution-ledgers",
            content_hash=ledger_hash,
            model=PortfolioExecutionLedger,
            identity_field="ledger_hash",
        )

    def load_program(self, program_hash: str) -> PortfolioExecutionProgram:
        """Reopen exact sealed program evidence.

        Args:
            program_hash: Exact retained content identity.

        Returns:
            Validated concrete artifact reopened through content-addressed storage.
        """
        return self.content.load_model(
            category="programs",
            content_hash=program_hash,
            model=PortfolioExecutionProgram,
            identity_field="program_hash",
        )

    def load_economics(self, economic_hash: str) -> PortfolioEconomicLedger:
        """Reopen exact sealed economics evidence.

        Args:
            economic_hash: Exact retained content identity.

        Returns:
            Validated concrete artifact reopened through content-addressed storage.
        """
        return self.content.load_model(
            category="economic-ledgers",
            content_hash=economic_hash,
            model=PortfolioEconomicLedger,
            identity_field="economic_ledger_hash",
        )

    def load_report(self, report_hash: str) -> PortfolioDeclaredPathReport:
        """Reopen exact sealed report evidence.

        Args:
            report_hash: Exact retained content identity.

        Returns:
            Validated concrete artifact reopened through content-addressed storage.
        """
        return self.content.load_model(
            category="reports",
            content_hash=report_hash,
            model=PortfolioDeclaredPathReport,
            identity_field="report_hash",
        )

    def load_result(self, result_hash: str) -> PortfolioResearchResult:
        """Reopen exact sealed result evidence.

        Args:
            result_hash: Exact retained content identity.

        Returns:
            Validated concrete artifact reopened through content-addressed storage.
        """
        return self.content.load_model(
            category="results",
            content_hash=result_hash,
            model=PortfolioResearchResult,
            identity_field="result_hash",
        )

    def load_html_by_uri(self, uri: str) -> str:
        """Read back a published page from the URI the result carries."""
        tail = uri.rsplit("/", maxsplit=1)[-1]
        content_hash, _, extension = tail.rpartition(".")
        # A legacy result names its page `.bin`.
        if extension not in {"html", "bin"} or not content_hash:
            raise PortfolioLedgerStoreError("portfolio_application.html_uri_invalid")
        return self.load_html(content_hash)

    def load_html(self, content_hash: str) -> str:
        """Reopen exact content-addressed report HTML as UTF-8.

        Args:
            content_hash: Exact HTML artifact identity.

        Returns:
            Verified decoded HTML text.
        """
        return self.content.load_document(
            category="html", content_hash=content_hash, extension="html"
        ).decode("utf-8")

    # --------------------------------------------------------------- release

    def adopt_result_from(self, source: PortfolioLedgerStore, result_hash: str) -> str:
        """Copy one sealed result and its whole child set into this namespace.

        The release step of a protected finalization, and the only way an
        artifact leaves the pending root. Every child is copied by its own
        content hash, so what becomes public is byte-identical to what was
        validated -- an adoption that re-derived anything would publish a
        different artifact than the receipt closed over.

        No index is written at all, which is stronger than the previous rule and
        for a better reason. It is not only that the request-shaped indices would
        make a protected path a cache hit for somebody's next question; it is
        that *any* index is a discovery route, and a discovery route that exists
        before the release marker does would make a half-copied tree look
        released. After the marker, a reader reaches this result through the
        handoff, by identity.
        """
        if not self.is_public:
            raise PortfolioLedgerStoreError("portfolio_application.adoption_target_not_public")
        if source.is_public:
            raise PortfolioLedgerStoreError("portfolio_application.adoption_source_not_pending")
        result = source.load_result(result_hash)
        program = source.load_program(result.program_hash)
        ledger = source.load_execution(result.execution_ledger_hash)
        economics = source.load_economics(result.economic_ledger_hash)
        report = source.load_report(result.report_hash)
        comparison = source.load_comparison(report.benchmark_comparison_hash)
        lanes: list[tuple[str, str | None]] = [
            ("executed-weights", ledger.executed_weights_hash),
            ("pre-cap-weights", ledger.pre_cap_weights_hash),
            ("final-weights", ledger.final_weights_hash),
            ("target-weights", ledger.target_weights_hash),
            ("realized-returns", ledger.realized_simple_returns_hash),
            ("execution-available", ledger.execution_available_hash),
        ]
        for boundary in (ledger.initial_boundary, ledger.final_boundary):
            if boundary is not None:
                lanes.append(("boundary-weights", boundary.pretrade_weights_hash))
                lanes.append(("boundary-weights", boundary.optimizer_reference_hash))
                lanes.append(("boundary-sleeves", boundary.sleeve_weights_hash))
        for category, content_hash in lanes:
            if content_hash is not None:
                self.publish_lane(
                    category=category,
                    values=np.frombuffer(
                        source.load_lane(category=category, content_hash=content_hash),
                        dtype=_LANE_DTYPES.get(category, "<f8"),
                    ),
                )
        self.publish_html(source.load_html_by_uri(result.html_uri))
        # Content only. Not one index is written -- not the by-request index
        # `publish_result` keeps, nor the by-program index `publish_execution`
        # keeps. Adoption is *preparation*: it puts the exact validated bytes
        # where a released reader will need them, and releases nothing. What
        # makes them observable is the release marker, committed afterwards, and
        # a crash anywhere in here leaves a public store that discovers nothing.
        for category, value, identity_field in (
            ("programs", program, "program_hash"),
            ("execution-ledgers", ledger, "ledger_hash"),
            ("economic-ledgers", economics, "economic_ledger_hash"),
            ("benchmark-comparisons", comparison, "comparison_hash"),
            ("reports", report, "report_hash"),
            ("results", result, "result_hash"),
        ):
            self.content.publish_model(
                category=category, value=value, identity_field=identity_field
            )
        return result.result_hash


__all__ = [
    "PENDING_FINALIZATION_NAMESPACE",
    "PUBLIC_NAMESPACE",
    "PortfolioLedgerStore",
    "PortfolioLedgerStoreError",
]
