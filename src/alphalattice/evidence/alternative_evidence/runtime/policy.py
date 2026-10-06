"""What a workspace may ask Alternative Evidence for: its admitted evidence policy.

The policy decides the source families, the classes, the mode, the matter selection
and the three consent flags; a route may not widen any of them. It seals each request
and admission it asks for, and says whether a recorded request, or the one a coverage
run is read from, is the one it asks for now (W1: moved from the Host).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from alphalattice.evidence.alternative_evidence.analysis.matters import TOPIC_LANES_ALLOCATION_ID
from alphalattice.evidence.alternative_evidence.analysis.packet import (
    MATTER_WINDOW_BYTES,
    RESIDUAL_RERANK_PAIR_BUDGET,
    RESIDUAL_SELECTION_RULES_ID,
)
from alphalattice.evidence.alternative_evidence.analysis.routing import ROUTING_RULES_ID, TOPICS
from alphalattice.evidence.alternative_evidence.contracts import (
    INTEGRATED_FAMILY_SPELLING,
    MATTER_FAMILY_LITIGATION,
    MATTER_SELECTION_INTEGRATED,
    MATTER_SELECTION_PRODUCTION,
    AlternativeEvidenceAdmission,
    AlternativeEvidenceClass,
    AlternativeEvidenceMode,
    AlternativeEvidenceReadFiling,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSourcePolicy,
    MatterSelectionPolicy,
    matter_selection_identity,
    matter_selection_retired,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.retrieval.session import (
    MAXIMUM_ISSUED_MATTER_WINDOWS,
    MAXIMUM_MATTER_READS,
)
from alphalattice.evidence.alternative_evidence.runtime.coverage import (
    AlternativeEvidenceCoverageRun,
)

REINSTALL_INTEGRATED_SELECTION = (
    "scripts/materialize_evidence_cro_authority.py --workspace <workspace> "
    "--rebind-installed --matter-selection INTEGRATED_TOPIC_ROUTING --install"
)
"""The step that migrates a workspace installed under a retired matter
selection: its authority re-sealed under the integrated selection, its
listing authority, floor and model profile kept."""


RETIRED_SELECTION_EXPLANATION = (
    "This workspace's evidence authority asks for a matter selection that is "
    "retired: the production reading plan or the candidate needs allocation. "
    "Nothing new is prepared, refreshed or continued under it, and nothing is "
    "run under the integrated selection in its place; what it sealed stays "
    "readable. Re-install the authority under the integrated selection, then "
    "preview again."
)


def _integrated_selection() -> MatterSelectionPolicy:
    return MatterSelectionPolicy(
        method=MATTER_SELECTION_INTEGRATED, families=INTEGRATED_FAMILY_SPELLING
    )


@dataclass(frozen=True, slots=True)
class AdmittedEvidencePolicy:
    """What this host is permitted to ask Alternative Evidence for.

    An *authority*, not a preference: it decides the source families, the
    classes, the mode and the three consent flags, and the route may not widen
    any of them.
    """

    evidence_classes: tuple[AlternativeEvidenceClass, ...] = (
        AlternativeEvidenceClass.ISSUER_OFFICIAL_RECORDED,
    )
    approved_source_families: tuple[str, ...] = ("SEC_EDGAR_OFFICIAL",)
    source_policy: AlternativeEvidenceSourcePolicy = field(
        default_factory=AlternativeEvidenceSourcePolicy
    )
    mode: AlternativeEvidenceMode = AlternativeEvidenceMode.RECORDED
    ttl_seconds: int = 86_400
    acquisition_window_seconds: int = 3_600
    network_consent: bool = False
    admit_live_official: bool = False
    admit_model_review: bool = False
    matter_selection: MatterSelectionPolicy | None = field(default_factory=_integrated_selection)
    """The matter selection every request of this host asks for, as the
    workspace manifest binds it into its authority hash and this policy
    binds it into each request's identity: the integrated selection (the
    default and the one current method). None -- a workspace still
    installed under the production plan -- and the candidate needs
    allocation are retired: every new request is refused by name
    (`alternative_evidence.matter_selection_policy_retired`) with
    `REINSTALL_INTEGRATED_SELECTION` as the next step, and never run under
    the integrated selection instead. Never widened by a route."""

    def request(
        self,
        *,
        ordered_entity_ids: tuple[str, ...],
        evidence_as_of: datetime,
        read_filings: tuple[AlternativeEvidenceReadFiling, ...] = (),
    ) -> AlternativeEvidenceRequest:
        """Seal the request this policy asks for one unit's issuers at one cutoff.

        Args:
            ordered_entity_ids: The unit's issuers, in the order they are read.
            evidence_as_of: The cutoff; the acquisition deadline is this policy's window
                after it.
            read_filings: What an earlier analysis read of these issuers, so only what
                is new is read.

        Returns:
            The sealed request, under this policy's classes, source policy, TTL, mode
            and matter selection.
        """
        return seal_contract(
            AlternativeEvidenceRequest,
            "request_hash",
            ordered_entity_ids=ordered_entity_ids,
            evidence_as_of=evidence_as_of,
            acquisition_deadline=evidence_as_of
            + timedelta(seconds=self.acquisition_window_seconds),
            evidence_classes=self.evidence_classes,
            source_policy=self.source_policy,
            ttl_seconds=self.ttl_seconds,
            mode=self.mode,
            matter_selection=self.matter_selection,
            read_filings=read_filings,
        )

    @property
    def matter_selection_id(self) -> str:
        """Name the selection this policy asks every request for.

        Returns:
            The one identity requests, receipts and the reuse index compare it by.
        """
        return matter_selection_identity(self.matter_selection)

    def matter_selection_view(self) -> dict[str, object]:
        """Disclose the matter selection as a preview shows it.

        The method, the families it inventories, the allocation, the routing, the one
        per-session allowance and the reading protocol: the first response inventories the
        supported scope and states its pending work; necessary evidence beyond it is read
        through explicitly bounded continuation. A retired selection is named as retired,
        with the re-install step, and described no further.

        Returns:
            The selection's disclosure, as a preview document carries it.
        """
        selection = self.matter_selection
        if selection is None or matter_selection_retired(selection):
            return {
                "method": MATTER_SELECTION_PRODUCTION if selection is None else selection.method,
                "families": [MATTER_FAMILY_LITIGATION]
                if selection is None
                else list(selection.families),
                "retired": True,
                "refusal": "alternative_evidence.matter_selection_policy_retired",
                "setup_help": REINSTALL_INTEGRATED_SELECTION,
            }
        return {
            "method": selection.method,
            "families": list(selection.families),
            "allocation_rules_id": TOPIC_LANES_ALLOCATION_ID,
            "allocation": (
                "one sealed service order over the plan's needs: the issuers in turn, each "
                "issuer's topics in turn (the eight evidence topics, in their declared "
                "order) under the one shared allowance less a share of at most eight table "
                "views, each topic alternating an opening of an unread disclosure -- latest "
                "filing first -- with an extension of one already opened; an earlier "
                "filing's unit an exact repeat restates is served by the later reading; a "
                "continuation resumes the same order"
            ),
            "inventories": (
                "the litigation regions of the 10-K/10-Q filings; the acquisition, "
                "divestiture and subsequent-events notes of those filings and the items of "
                "the 8-K filings; the debt, borrowings and financing notes of the 10-K/10-Q "
                "filings; the restructuring, impairment and operating-charge notes of the "
                "10-K/10-Q filings and the dated operations statements of their business, "
                "risk-factor and MD&A items"
            ),
            "routing": {
                "rules_id": ROUTING_RULES_ID,
                "topics": [str(topic) for topic in TOPICS],
                "residual_search": (
                    "the question bank runs only over each topic's residual scope "
                    "(the routed ranges no inventory addressed) under one "
                    f"cross-encoder pair budget of {RESIDUAL_RERANK_PAIR_BUDGET} per "
                    "session; a question with no scope or beyond the budget is "
                    "recorded as skipped, never run narrower"
                ),
                # The one residual allocation the runtime deals; a renderer
                # shows the rules id a receipt names, and a historical
                # receipt may name a retired one.
                "residual_allocation": {"rules_id": RESIDUAL_SELECTION_RULES_ID},
                "tables": (
                    "tables inside routed regions are delivered as verified views of "
                    "the retained original inside the matter allowance; a table "
                    "without a retained original is a representation gap"
                ),
            },
            "per_session_allowance": {
                "windows": MAXIMUM_ISSUED_MATTER_WINDOWS,
                "reads": MAXIMUM_MATTER_READS,
                "window_bytes": MATTER_WINDOW_BYTES,
            },
            "reading_protocol": (
                "The first response inventories the supported scope under one "
                "per-session allowance and states its pending work exactly; necessary "
                "evidence beyond it is read only through explicitly bounded continuation "
                "(EVIDENCE_CONTINUE under declared cumulative limits). A first response "
                "is not a completed review."
            ),
        }

    def matches(self, request: AlternativeEvidenceRequest) -> bool:
        """Tell whether a recorded request is the one this policy asks for now.

        The classes, source policy, TTL, mode, acquisition window and the matter selection
        (its method and families, by the request's own identity) must all agree. An
        analysis, run or packet prepared under another selection is history, never the
        current answer; it stays readable.

        Args:
            request: A recorded request.

        Returns:
            Whether this policy would ask the same request now.
        """
        return (
            tuple(request.evidence_classes) == self.evidence_classes
            and request.source_policy == self.source_policy
            and request.ttl_seconds == self.ttl_seconds
            and request.mode is self.mode
            and request.acquisition_deadline - request.evidence_as_of
            == timedelta(seconds=self.acquisition_window_seconds)
            and request.matter_selection_id == self.matter_selection_id
        )

    def run_request(self, run: AlternativeEvidenceCoverageRun) -> AlternativeEvidenceRequest:
        """Choose the request a coverage run's policy is read from.

        Args:
            run: A sealed coverage run.

        Returns:
            Its first unit's request, or for a run with nothing left to read, the one this
            policy asks at its cutoff.
        """
        if run.units:
            return run.units[0].request
        return self.request(
            ordered_entity_ids=((*run.carried, *run.nothing_filed)[0],),
            evidence_as_of=run.evidence_as_of,
        )

    def admission(
        self, *, request: AlternativeEvidenceRequest, admitted_at: datetime
    ) -> AlternativeEvidenceAdmission:
        """Seal the admission of one request under this policy's three consent flags.

        Args:
            request: The request admitted.
            admitted_at: The moment of the admission.

        Returns:
            The sealed admission.
        """
        return seal_contract(
            AlternativeEvidenceAdmission,
            "admission_hash",
            request_hash=request.request_hash,
            network_consent=self.network_consent,
            admit_live_official=self.admit_live_official,
            admit_model_review=self.admit_model_review,
            admitted_at=admitted_at,
        )
