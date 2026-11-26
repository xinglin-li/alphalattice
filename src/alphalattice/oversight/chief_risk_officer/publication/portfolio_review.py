"""Pointer-free publication and exact reuse for the Portfolio evidence review.

One immutable publication per review key, first writer wins. Reuse is keyed on
the Task input envelope, so asking the same question twice is free and asking a
genuinely new one requires a changed input rather than a changed clock.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict

from alphalattice.evidence.alternative_evidence.publication.artifacts import (
    AlternativeEvidenceArtifactStore,
    AlternativeEvidencePublicationError,
)
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    OpenIssueState,
    PortfolioReviewDossier,
    PortfolioReviewMaterialIssue,
    PortfolioReviewOpenIssue,
    PortfolioReviewPublication,
    PortfolioReviewReceipt,
    PortfolioReviewRecommendation,
    PortfolioReviewRoute,
    RequiredActionKind,
    finding_sources,
    raised_issue_handle,
)
from alphalattice.oversight.chief_risk_officer.decision.submissions import (
    PORTFOLIO_REVIEW_POLICY_ROLE,
    build_portfolio_review_policy_binding,
)

_REVIEW_KEY_CATEGORY = "cro-review-keys"
_REGISTER_CATEGORY = "cro-review-register"
_PUBLICATION_CATEGORY = "cro-review-publications"
_REGISTER_DAYS = "days"
"""The register's index beside its records (K2): `days/<cutoff day>.<tag>.<publication>`,
empty files, the tag a fresh review's basis (its first sixteen), `unbased` for a
fresh review sealed before a basis was stated, or `carried`."""
_CARRIED = "carried"
_EARLIEST = datetime.min.replace(tzinfo=UTC)
_PERSON_ROUTES = frozenset(
    {PortfolioReviewRoute.HUMAN_REVIEW_REQUIRED, PortfolioReviewRoute.MATERIAL_OBJECTION}
)
_PERSON_ACTIONS = frozenset(
    {
        RequiredActionKind.HUMAN_REVIEW,
        RequiredActionKind.RECONSIDER_CANDIDATE,
        RequiredActionKind.DO_NOT_ACTIVATE,
    }
)


@dataclass(frozen=True, slots=True)
class PortfolioReviewView:
    """One verified publication and everything it rests on.

    `under_installed_policy` is whether the review was sealed under the installed CRO
    review policy or one its recorded moves lead to, read from the receipt and
    never recompiled (OP6): the policy binds the compiler that chose the
    route. One sealed under an earlier policy reads back exactly as recorded;
    it is never reused as a current answer.
    """

    action: str
    publication: PortfolioReviewPublication
    dossier: PortfolioReviewDossier
    receipt: PortfolioReviewReceipt
    recommendation: PortfolioReviewRecommendation
    under_installed_policy: bool = True

    @property
    def person_action(self) -> dict[str, object] | None:
        """What this review's recommendation asks a person to do, or None.

        Read from the sealed recommendation, never decided here: its route when that route
        needs a person, and each required action only a person takes (a human review, a
        candidate to reconsider, one not to activate).
        """
        actions = [
            {
                "action": item.action.value,
                "entity_id": item.entity_id,
                "reason": item.reason,
                "blocking": item.blocking,
            }
            for item in self.recommendation.required_actions
            if item.action in _PERSON_ACTIONS
        ]
        route = self.recommendation.route
        if route not in _PERSON_ROUTES and not actions:
            return None
        return {"route": route.value, "actions": actions}


class _RegisterIssue(BaseModel):  # type: ignore[misc]
    """One open issue as the register holds it (`OpenIssueState`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    open_issue_handle: str
    raised_on: date
    assessed_on: date
    issue: PortfolioReviewMaterialIssue
    findings: tuple[tuple[str, str, date], ...]


class _Register(BaseModel):  # type: ignore[misc]
    """What one review is to the register (X2): its cutoff, whether it is the
    reviewer's own assessment, its basis, and -- for a fresh review -- the
    issues open as it left them, extending the state of the fresh review
    before it in the register's order (`prior`). A later read takes one state
    instead of folding every review since the first."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    publication_hash: str
    cutoff: datetime
    published_at: datetime
    fresh: bool
    review_basis: str | None = None
    prior: str | None = None
    open_issues: tuple[_RegisterIssue, ...] = ()
    digest: str = ""

    def order(self) -> tuple[datetime, datetime, str]:
        return (self.cutoff, self.published_at, self.publication_hash)

    def sealed(self) -> _Register:
        return cast(_Register, self.model_copy(update={"digest": _digest(self)}))

    def indexed(self) -> tuple[date, str]:
        if not self.fresh:
            return _day(self.cutoff), _CARRIED
        return _day(self.cutoff), (self.review_basis or "unbased")[:16]


def _day(moment: datetime) -> date:
    return moment.astimezone(UTC).date()


def _digest(entry: _Register) -> str:
    content = json.dumps(entry.model_dump(mode="json", exclude={"digest"}), sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _fold(
    state: tuple[_RegisterIssue, ...],
    *,
    receipt: PortfolioReviewReceipt,
    dossier: PortfolioReviewDossier,
    cutoff: datetime,
) -> tuple[_RegisterIssue, ...]:
    """The register after one fresh review: an issue is raised by a material
    issue that states no open one, stated again by one that names it, resolved
    by a resolution; one the Host carried as last assessed states nothing."""
    held = {value.open_issue_handle: value for value in state}
    sources = finding_sources(dossier)
    for issue in receipt.submission.material_issues:
        if issue.carried_on is not None:
            continue
        handle = issue.open_issue_handle or raised_issue_handle(
            receipt.receipt_hash, issue.issue_handle
        )
        before = held.get(handle)
        held[handle] = _RegisterIssue(
            open_issue_handle=handle,
            raised_on=cutoff.date() if before is None else before.raised_on,
            assessed_on=cutoff.date(),
            issue=issue,
            findings=tuple(
                sources[cited] for cited in issue.cited_finding_handles if cited in sources
            ),
        )
    for resolution in receipt.submission.resolved_issues:
        held.pop(resolution.open_issue_handle, None)
    return tuple(
        sorted(held.values(), key=lambda value: (value.raised_on, value.open_issue_handle))
    )


class PortfolioReviewPublicationService:
    """Publish and reopen one review. It owns no pointer.

    The receipts the publications name answer two questions more (W3): the
    last fresh review of a basis -- what a later dossier on the same basis
    carries forward -- and the issuers' register: every issue a fresh review
    raised, open until one resolves it. Both are read from the register kept
    beside the publications, one record per review, written as each is
    published (X2); a review published before it was kept is read from its
    receipt once and written with the next publication.

    A read takes only the records it needs (K2): the register's index names
    each review's cutoff day and basis, so the state before a cutoff is the
    newest fresh review's at or before it, and a basis's reviews are those its
    name lists. The records before a place in the register's order are as the
    last publication settled them -- the Host is the one publisher, and a
    publication writes its own record last, so one a crash lost is read from
    its receipt -- and are not read again.
    """

    def __init__(
        self,
        store: AlternativeEvidenceArtifactStore,
        *,
        installed_policy: Callable[[], str] | None = None,
    ) -> None:
        """Bind immutable review publication and register readback to an artifact store.

        Args:
            store: Store owning admitted CRO dossiers, receipts, recommendations and publications.
            installed_policy: Optional provider of the installed review-policy identity; evaluated
                lazily.
        """
        self.store = store
        self._installed_policy = installed_policy or _installed_review_policy
        self._policy: str | None = None
        self._index: dict[str, tuple[date, str]] | None = None
        self._held: dict[str, _Register] = {}
        self._unwritten: set[str] = set()
        self.records_read = 0
        self.learned = 0
        """Count register reads and records read solely to learn their index placement.

        Register records read, and those read only to learn their place in
        the index (a record written before it was kept)."""
        self.noted: Callable[[str, datetime], None] | None = None
        """Observe a new publication identity and timestamp when the Host keeps its day.

        Keeps each publication's day as it is published, where the Host
        chooses a book's review from the records of their days (Z2); unbound,
        nothing is kept and a reader learns the day by reading the record."""

    def publish(self, *, publication: PortfolioReviewPublication) -> PortfolioReviewView:
        """Publish or exactly replay one review key and settle its issuer-register records.

        The owner reads the complete lineage before indexing the publication. Fresh reviews settle
        affected later register states; the new review record is written last. An exact existing
        installed-policy publication is reused.

        Args:
            publication: Sealed recommendation/receipt/source lineage with its exact review key.

        Returns:
            Verified view of the published or exactly replayed review.

        Raises:
            AlternativeEvidencePublicationError: The review key is ambiguous or publication lineage
                fails verification.
        """
        existing = self.find_for_review_key(publication.review_key)
        if existing is not None:
            if existing.publication != publication:
                raise AlternativeEvidencePublicationError("chief_risk_officer.review_key_ambiguous")
            return existing
        self.store.publish(_PUBLICATION_CATEGORY, publication.publication_hash, publication)
        if self.noted is not None:
            self.noted(publication.publication_hash, publication.published_at)
        self._write_review_key(publication)
        view = self.read(publication.publication_hash)
        entry = _Register(
            publication_hash=publication.publication_hash,
            cutoff=view.recommendation.evidence_as_of,
            published_at=publication.published_at,
            fresh=not view.receipt.carried_forward,
            review_basis=view.receipt.review_basis,
        )
        self._indexed()[entry.publication_hash] = entry.indexed()
        self._held[entry.publication_hash] = entry
        self._unwritten.add(entry.publication_hash)
        if entry.fresh:
            self._settle(entry.order())
        for publication_hash in sorted(self._unwritten - {entry.publication_hash}):
            self._write_register(self._held[publication_hash])
        self._write_register(self._held[entry.publication_hash])
        self._unwritten.clear()
        return view

    def _indexed(self) -> dict[str, tuple[date, str]]:
        """Every review the register holds, by publication: its cutoff's day and
        its tag, from the index's names. A record the index does not name --
        written before it was kept -- is read once to learn it; a publication
        without its record is read from its receipt (H9), and the register is
        settled from the first such fresh review."""
        if self._index is None:
            names = {path.stem for path in (self.store.root / _PUBLICATION_CATEGORY).glob("*.json")}
            directory = self.store.root / _REGISTER_CATEGORY
            recorded = (
                {path.stem for path in directory.glob("*.json")} if directory.is_dir() else set()
            )
            index: dict[str, tuple[date, str]] = {}
            days = directory / _REGISTER_DAYS
            for path in days.iterdir() if days.is_dir() else ():
                day, _, rest = path.name.partition(".")
                tag, _, publication_hash = rest.partition(".")
                try:
                    index[publication_hash] = (date.fromisoformat(day), tag)
                except ValueError:
                    continue
            self._index = {name: index[name] for name in names & recorded & set(index)}
            for name in sorted(names & recorded - set(index)):
                self.learned += 1
                learned = self._load(name)
                self._mark(learned)
                self._index[name] = learned.indexed()
            start = None
            for name in sorted(names - recorded):
                entry = self._from_receipt(name)
                if entry is not None:
                    self._held[name] = entry
                    self._index[name] = entry.indexed()
                    self._unwritten.add(name)
                    if entry.fresh and (start is None or entry.order() < start):
                        start = entry.order()
            if start is not None:
                self._settle(start)
        return self._index

    def _load(self, publication_hash: str) -> _Register:
        entry = self._held.get(publication_hash)
        if entry is None:
            path = self.store.root / _REGISTER_CATEGORY / f"{publication_hash}.json"
            entry = _Register.model_validate_json(path.read_bytes())
            if entry.publication_hash != publication_hash or entry.digest != _digest(entry):
                raise AlternativeEvidencePublicationError(
                    "chief_risk_officer.review_register_tampered"
                )
            self.records_read += 1
            self._held[publication_hash] = entry
        return entry

    def _fresh(self, tag: str | None = None) -> list[tuple[date, list[str]]]:
        """The fresh reviews by cutoff day, newest first -- those of one tag
        when named -- by the index's names alone."""
        days: dict[date, list[str]] = {}
        for name, (day, value) in self._indexed().items():
            if value != _CARRIED and (tag is None or value == tag):
                days.setdefault(day, []).append(name)
        return sorted(days.items(), reverse=True)

    def _before(self, place: tuple[datetime, datetime, str]) -> _Register | None:
        """The fresh review last before this place in the register's order:
        the newest cutoff day at or before it that holds one is read, and no
        day before it."""
        for day, names in self._fresh():
            if day > _day(place[0]):
                continue
            earlier = [entry for entry in map(self._load, names) if entry.order() < place]
            if earlier:
                return max(earlier, key=_Register.order)
        return None

    def _from_receipt(self, publication_hash: str) -> _Register | None:
        """A review the register does not hold yet, read from its receipt."""
        try:
            publication = self.store.load(
                _PUBLICATION_CATEGORY, publication_hash, PortfolioReviewPublication
            )
            receipt = self.store.load(
                "cro-review-receipts", publication.decision_receipt_hash, PortfolioReviewReceipt
            )
            recommendation = self.store.load(
                "cro-review-recommendations",
                publication.recommendation_hash,
                PortfolioReviewRecommendation,
            )
        except (ValueError, FileNotFoundError):
            return None
        return _Register(
            publication_hash=publication_hash,
            cutoff=recommendation.evidence_as_of,
            published_at=publication.published_at,
            fresh=not receipt.carried_forward,
            review_basis=receipt.review_basis,
        )

    def _settle(self, start: tuple[datetime, datetime, str]) -> None:
        """Give every fresh review from this place in the register's order on
        the state it leaves, folded from its receipt onto the one before it: a
        review not yet held, and those after a review published later at an
        earlier cutoff. The days before the place are not read."""
        prior = self._before(start)
        later = sorted(
            (
                entry
                for day, names in self._fresh()
                if day >= _day(start[0])
                for entry in map(self._load, names)
                if entry.order() >= start
            ),
            key=_Register.order,
        )
        for entry in later:
            publication = self.store.load(
                _PUBLICATION_CATEGORY, entry.publication_hash, PortfolioReviewPublication
            )
            entry = entry.model_copy(
                update={
                    "prior": None if prior is None else prior.publication_hash,
                    "open_issues": _fold(
                        () if prior is None else prior.open_issues,
                        receipt=self.store.load(
                            "cro-review-receipts",
                            publication.decision_receipt_hash,
                            PortfolioReviewReceipt,
                        ),
                        dossier=self.store.load(
                            "cro-review-dossiers",
                            publication.dossier_hash,
                            PortfolioReviewDossier,
                        ),
                        cutoff=entry.cutoff,
                    ),
                }
            ).sealed()
            self._held[entry.publication_hash] = entry
            self._unwritten.add(entry.publication_hash)
            prior = entry

    def _write_register(self, entry: _Register) -> None:
        directory = self.store.root / _REGISTER_CATEGORY
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{entry.publication_hash}.json"
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        staged.write_bytes(entry.sealed().model_dump_json().encode("utf-8"))
        os.replace(staged, target)
        self._mark(entry)

    def _mark(self, entry: _Register) -> None:
        """Name a record in the index; derived, rebuilt by reading the record."""
        day, tag = entry.indexed()
        days = self.store.root / _REGISTER_CATEGORY / _REGISTER_DAYS
        days.mkdir(parents=True, exist_ok=True)
        (days / f"{day.isoformat()}.{tag}.{entry.publication_hash}").touch()

    def find_for_basis(
        self, basis: str, *, open_issues: tuple[PortfolioReviewOpenIssue, ...]
    ) -> PortfolioReviewView | None:
        """Find the last fresh review only while its exact open-issue state still applies.

        The last fresh review whose assessment rests on this basis, when the
        issuers' register is as it left it -- every open issue one it left open
        or raised, and none stated since -- or nothing.

        Args:
            basis: Exact review-basis identity.
            open_issues: Current admitted issuer-register state to compare with the prior
                assessment.

        Returns:
            Last fresh matching review, or None when no review or unchanged issue state is
            available.
        """
        fresh = [
            value
            for _, names in self._fresh(basis[:16])
            for value in map(self._load, names)
            if value.review_basis == basis
        ]
        if not fresh:
            return None
        last = max(fresh, key=lambda value: (value.published_at, value.publication_hash))
        view = self.read(last.publication_hash)
        submission = view.receipt.submission
        resolved = {value.open_issue_handle for value in submission.resolved_issues}
        left = {
            *(
                value.open_issue_handle
                for value in view.dossier.open_issues
                if value.open_issue_handle not in resolved
            ),
            *(
                raised_issue_handle(view.receipt.receipt_hash, value.issue_handle)
                for value in submission.material_issues
                if value.open_issue_handle is None
            ),
        }
        if {value.open_issue_handle for value in open_issues} != left or any(
            value.assessed_on > last.cutoff.date() for value in open_issues
        ):
            return None
        return view

    def open_issues(
        self, entity_ids: frozenset[str], *, as_of: datetime
    ) -> tuple[OpenIssueState, ...]:
        """Read issuer issues open strictly before the admitted cutoff.

        The issues open before this cutoff that name any of these issuers, in
        the order they were raised: the register as the last fresh review at an
        earlier cutoff left it, whatever book it reviewed and whatever CRO
        decision policy sealed it -- the register is the issuers', never a
        book's or a policy's, and a raised risk stays open until a review
        resolves it (X4) -- and a review at this cutoff is the one being read,
        whose own issues are its answer, not its question.

        Args:
            entity_ids: Admitted issuer identifiers whose open issues are requested.
            as_of: Cutoff whose own review must not contribute to its question.

        Returns:
            Open issue states in raised order, independent of book or historical review policy.
        """
        last = self._before((as_of, _EARLIEST, ""))
        if last is None:
            return ()
        return tuple(
            OpenIssueState(
                open_issue_handle=value.open_issue_handle,
                raised_on=value.raised_on,
                assessed_on=value.assessed_on,
                issue=value.issue,
                findings=value.findings,
            )
            for value in last.open_issues
            if set(value.issue.affected_entities) & entity_ids
        )

    def read(self, publication_hash: str) -> PortfolioReviewView:
        """Reopen the whole lineage exactly. No clock is consulted.

        The records' agreement with one another is integrity, and a mismatch
        refuses. Whether the review answers under the installed policy is read
        from its receipt against the installed policy and the recorded moves: a
        policy changed after the review was published leaves the review
        readable, with `under_installed_policy` false.
        """
        publication = self.store.load(
            "cro-review-publications", publication_hash, PortfolioReviewPublication
        )
        dossier = self.store.load(
            "cro-review-dossiers", publication.dossier_hash, PortfolioReviewDossier
        )
        receipt = self.store.load(
            "cro-review-receipts", publication.decision_receipt_hash, PortfolioReviewReceipt
        )
        recommendation = self.store.load(
            "cro-review-recommendations",
            publication.recommendation_hash,
            PortfolioReviewRecommendation,
        )
        if (
            receipt.dossier_hash != dossier.dossier_hash
            or recommendation.decision_receipt_hash != receipt.receipt_hash
            or publication.report_hash != dossier.report_hash
            or publication.update_subject != dossier.update_subject
            or publication.experiment_subject != dossier.experiment_subject
            or publication.result_hash != dossier.result_hash
            or publication.book_authority is not dossier.book_authority
            or publication.issuer_scope_hash != dossier.issuer_scope_hash
            or publication.analysis_publication_hash != dossier.analysis_publication_hash
            or publication.decision_policy_hash != receipt.decision_policy_hash
        ):
            raise AlternativeEvidencePublicationError("chief_risk_officer.review_lineage_mismatch")
        return PortfolioReviewView(
            action="READBACK",
            publication=publication,
            dossier=dossier,
            receipt=receipt,
            recommendation=recommendation,
            under_installed_policy=self.under_installed_policy(receipt.decision_policy_hash),
        )

    def under_installed_policy(self, recorded: str) -> bool:
        """Whether a receipt's policy is the installed one or a recorded move leads to it."""
        if self._policy is None:
            self._policy = self._installed_policy()
        return is_current(PORTFOLIO_REVIEW_POLICY_ROLE, recorded, self._policy)

    def publication_hashes(self) -> tuple[str, ...]:
        """Every published review's handle, by its file's name.

        Nothing is parsed here, so a damaged record is named by the read that opens it, not by
        this listing.
        """
        directory = self.store.root / _PUBLICATION_CATEGORY
        if not directory.is_dir():
            return ()
        return tuple(sorted(path.stem for path in directory.glob("*.json")))

    def find_for_review_key(self, review_key: str) -> PortfolioReviewView | None:
        """The exact prior answer to this question, or nothing."""
        path = self.store.root / _REVIEW_KEY_CATEGORY / f"{review_key}.json"
        if not path.is_file():
            return None
        recorded = json.loads(path.read_text(encoding="utf-8"))
        view = self.read(str(recorded["publication_hash"]))
        if view.publication.review_key != review_key:
            raise AlternativeEvidencePublicationError("chief_risk_officer.review_key_tampered")
        # An answer sealed under an earlier policy is historical: the question
        # is answered again, and the earlier review stays readable by its hash.
        return view if view.under_installed_policy else None

    def _write_review_key(self, publication: PortfolioReviewPublication) -> None:
        path = self.store.root / _REVIEW_KEY_CATEGORY / f"{publication.review_key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "review_key": publication.review_key,
                "publication_hash": publication.publication_hash,
                "published_at": publication.published_at.isoformat(),
            },
            sort_keys=True,
        ).encode("utf-8")
        if path.is_file():
            if path.read_bytes() != payload:
                raise AlternativeEvidencePublicationError("chief_risk_officer.review_key_ambiguous")
            return
        path.write_bytes(payload)


def _installed_review_policy() -> str:
    """The installed CRO review policy of the checkout this module is in."""
    return build_portfolio_review_policy_binding(resolve_playpen_root(Path(__file__))).binding_hash


__all__ = ["PortfolioReviewPublicationService", "PortfolioReviewView"]
