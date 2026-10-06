"""The Portfolio-to-Evidence binding for one sealed book.

Portfolio Evidence owns one thing: how a sealed book's holdings and changes are
projected onto admitted issuers, which issuers are selected for review, and how
much of the book that selection covers. It never recomputes a Portfolio number
and it never reads a document.

The book may be any sealed book the Portfolio owner published: a development
result's window-end book, a frozen candidate's, or a validated handoff's. The
projection binds which of the three it is, because what a review of it may
claim depends on that authority and on nothing the reviewer says.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Final, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sealing import seal_model_validated

_HASH = r"^[0-9a-f]{64}$"


class PortfolioEvidenceContract(BaseModel):  # type: ignore[misc]
    """Provide frozen extra-forbidden contracts for the Portfolio-to-Evidence binding.

    Each concrete contract defines its own exact source and identity checks. This base does not
    compute Portfolio numbers or admit an evidence source.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


def seal_portfolio_evidence_contract[ContractT: PortfolioEvidenceContract](
    model: type[ContractT], identity_field: str, /, **values: object
) -> ContractT:
    """Seal cross-Desk evidence contracts without importing Strategy Lab runtime."""
    return seal_model_validated(model, identity_field, **values)


class BookAuthority(StrEnum):
    """What kind of sealed book a review is about. It decides the claim, not the route.

    A development result and a frozen candidate are pre-freeze books: an
    objection asks Portfolio to reconsider the candidate. A validated handoff is
    post-Validation: an objection can only decline activation. The same review,
    the same route grammar, a different required action.
    """

    DEVELOPMENT_RESULT = "DEVELOPMENT_RESULT"
    FROZEN_CANDIDATE = "FROZEN_CANDIDATE"
    VALIDATED_HANDOFF = "VALIDATED_HANDOFF"
    CONDITIONAL_RESEARCH_PROPOSAL = "CONDITIONAL_RESEARCH_PROPOSAL"
    OBSERVED_RESEARCH_ENTRY = "OBSERVED_RESEARCH_ENTRY"


class PortfolioUpdateReviewSubject(PortfolioEvidenceContract):
    """Exact CU origin; embedded in review identities, never a report alias."""

    update_task_id: UUID
    update_publication_hash: str = Field(pattern=_HASH)
    position_basis: Literal["CONDITIONAL_ESTIMATE", "OBSERVED_RESEARCH_ENTRY"]
    position_hash: str = Field(pattern=_HASH)
    checkpoint_hash: str = Field(pattern=_HASH)
    strategy_package_id: str = Field(min_length=1)
    observed_through: str = Field(min_length=10, max_length=10)
    formation_session: str = Field(min_length=10, max_length=10)
    entry_session: str = Field(min_length=10, max_length=10)

    @property
    def book_authority(self) -> BookAuthority:
        """Derive the review authority from the update position basis.

        Returns:
            CONDITIONAL_RESEARCH_PROPOSAL for a conditional estimate; OBSERVED_RESEARCH_ENTRY
            otherwise.
        """
        return (
            BookAuthority.CONDITIONAL_RESEARCH_PROPOSAL
            if self.position_basis == "CONDITIONAL_ESTIMATE"
            else BookAuthority.OBSERVED_RESEARCH_ENTRY
        )


class PortfolioExperimentReviewSubject(PortfolioEvidenceContract):
    """Exact authored replay/date, not a frozen-strategy result alias."""

    experiment_task_id: UUID
    experiment_receipt_hash: str = Field(pattern=_HASH)
    program_hash: str = Field(pattern=_HASH)
    input_binding_hash: str = Field(pattern=_HASH)
    portfolio_session: date
    preceding_session: date | None
    research_as_of_session: date
    research_as_of_phase: Literal["OPEN", "OFFICIAL_CLOSE"]
    position_basis: Literal["EXECUTED_RESEARCH_HOLDINGS"] = "EXECUTED_RESEARCH_HOLDINGS"
    position_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_clock(self) -> Self:
        """Require an observed portfolio session no later than the research cutoff.

        Returns:
            This contract after checking its declared consistency rules.

        Raises:
            ValueError: portfolio_session follows research_as_of_session or preceding_session is
                equal to or later than portfolio_session.
        """
        if self.portfolio_session > self.research_as_of_session or (
            self.preceding_session is not None and self.preceding_session >= self.portfolio_session
        ):
            raise ValueError("portfolio_evidence.experiment_clock_invalid")
        return self


def validate_review_book_authority(
    *,
    authority: BookAuthority,
    report_hash: str | None,
    result_hash: str | None,
    candidate_hash: str | None,
    handoff_hash: str | None,
    update_subject: PortfolioUpdateReviewSubject | None,
    experiment_subject: PortfolioExperimentReviewSubject | None = None,
) -> None:
    """Require one exact experiment, update or declared-book authority shape.

    Args:
        authority: Authority under which the Portfolio owner sealed this book.
        report_hash: Declared-path report identity, absent for experiment/update subjects.
        result_hash: Development/frozen result identity, absent for validated handoff.
        candidate_hash: Required for frozen-candidate or validated-handoff authority.
        handoff_hash: Required exactly for validated-handoff authority.
        update_subject: Exact Portfolio update origin, mutually exclusive with the other subject
            lane.
        experiment_subject: Exact authored development replay origin, when present.

    Raises:
        ValueError: Authority and source identities do not describe one admitted book lane.
    """
    if experiment_subject is not None:
        valid = authority is BookAuthority.DEVELOPMENT_RESULT and all(
            v is None
            for v in (report_hash, result_hash, candidate_hash, handoff_hash, update_subject)
        )
    elif update_subject is not None:
        valid = authority is update_subject.book_authority and all(
            v is None for v in (report_hash, result_hash, candidate_hash, handoff_hash)
        )
    else:
        valid = (
            authority
            in {
                BookAuthority.DEVELOPMENT_RESULT,
                BookAuthority.FROZEN_CANDIDATE,
                BookAuthority.VALIDATED_HANDOFF,
            }
            and report_hash is not None
            and (candidate_hash is not None)
            == (authority in {BookAuthority.FROZEN_CANDIDATE, BookAuthority.VALIDATED_HANDOFF})
            and (handoff_hash is not None) == (authority is BookAuthority.VALIDATED_HANDOFF)
            and (result_hash is None) == (authority is BookAuthority.VALIDATED_HANDOFF)
        )
    if not valid:
        raise ValueError("portfolio_evidence.review_book_authority_invalid")


class ExposureBand(StrEnum):
    """Identify the four ordinal book-exposure bands used in issuer review.

    Bands describe position size/change under the installed exposure policy. They do not score
    finding severity, propose weights or grant activation authority.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


EXPOSURE_BAND_POLICY: Final = "portfolio-evidence.exposure-band.v1"
"""The versioned rule that turns a position into a band. Changing it is a policy change."""


def exposure_band(*, ending_weight: float, signed_change: float, weight_rank: int) -> ExposureBand:
    """Four ordinal bands from the book's own numbers.

    The thresholds are declared for a long-only book of a few dozen names,
    where one name is a few percent: a tenth of the book or a fresh
    five-point build is `CRITICAL`; a top-three name, five percent, or a
    two-point build is `HIGH`; two percent or a half-point change is `MEDIUM`.
    Reductions count by their size too -- a large exit still leaves the
    residual position to review -- but only through the weight and change
    magnitudes, never through a return.
    """
    change = abs(signed_change)
    if ending_weight >= 0.10 or (signed_change > 0.0 and change >= 0.05):
        return ExposureBand.CRITICAL
    if ending_weight >= 0.05 or change >= 0.02 or weight_rank <= 3:
        return ExposureBand.HIGH
    if ending_weight >= 0.02 or change >= 0.005:
        return ExposureBand.MEDIUM
    return ExposureBand.LOW


class PortfolioIssuerMappingFailure(PortfolioEvidenceContract):
    """One holding that could not be resolved to an admitted issuer.

    `ticker` is optional because the first way this fails is having no ticker at
    all: a listing the admitted authority does not carry cannot be given one, and
    guessing a symbol out of a listing id is how a review ends up reading filings
    for the wrong company.
    """

    listing_id: str = Field(min_length=1, max_length=120)
    ticker: str | None = Field(default=None, max_length=16)
    reason: Literal[
        "LISTING_NOT_IN_ADMITTED_AUTHORITY",
        "TICKER_NOT_IN_SEC_REGISTRY",
        "TICKER_CIK_AMBIGUOUS",
    ]
    ending_weight: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    absolute_change: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_failure(self) -> Self:
        """Require ticker absence exactly for a missing admitted listing authority.

        Returns:
            This contract after checking its declared consistency rules.

        Raises:
            ValueError: Ticker presence disagrees with the declared mapping-failure reason.
        """
        if (self.ticker is None) != (self.reason == "LISTING_NOT_IN_ADMITTED_AUTHORITY"):
            raise ValueError("portfolio_evidence.mapping_failure_invalid")
        return self


class AdmittedEvidenceSelection(PortfolioEvidenceContract):
    """Which Alternative Evidence analysis one review scope is to be read against.

    Scoped by issuer scope, not by workspace: the scope names this book's
    issuers and the authority they were mapped through, so a selection cannot
    leak onto another book. It is deliberately not keyed by obligation, which
    carries the cutoff and therefore changes on every refresh -- a person
    chooses for a book, not for one instant. Records are append-only and the
    newest one for a scope is the answer in force, which keeps the store
    immutable and leaves the whole history readable.
    """

    kind: Literal["AdmittedEvidenceSelection"] = "AdmittedEvidenceSelection"
    issuer_scope_hash: str = Field(pattern=_HASH)
    analysis_publication_hash: str = Field(pattern=_HASH)
    evidence_as_of: datetime
    chosen_at: datetime
    chosen_by: Literal["HUMAN", "INSTALLED_AGENT", "EXTERNAL_AUTOMATION"]
    selection_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_selection(self) -> Self:
        for value in (self.evidence_as_of, self.chosen_at):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("portfolio_evidence.selection_clock_invalid")
        _identity(self, "selection_hash")
        return self


ADMITTED_LISTING_AUTHORITY: Final = "PORTFOLIO_EVIDENCE_ADMITTED_LISTING_TICKER_AUTHORITY"


class AdmittedListingTicker(PortfolioEvidenceContract):
    """Bind one admitted listing identifier to its normalized trading symbol.

    ticker is nonempty, stripped and uppercase. Issuer resolution uses this admitted symbol rather
    than guessing one from the listing identifier.
    """

    listing_id: str = Field(min_length=1, max_length=120)
    ticker: str = Field(min_length=1, max_length=16)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_entry(self) -> Self:
        """Require a stripped uppercase admitted trading symbol.

        Returns:
            This contract after checking its declared consistency rules.

        Raises:
            ValueError: ticker contains surrounding whitespace or differs from its uppercase form.
        """
        if self.ticker != self.ticker.strip().upper():
            raise ValueError("portfolio_evidence.ticker_invalid")
        return self


class AdmittedListingTickerAuthority(PortfolioEvidenceContract):
    """Which listings have an admitted trading symbol, and what it is.

    A separate, identity-bearing authority rather than a dict passed around,
    because the mapping is an input to an evidence request and every input to an
    evidence request has to be nameable afterwards.
    """

    kind: Literal["AdmittedListingTickerAuthority"] = "AdmittedListingTickerAuthority"
    authority: Literal["PORTFOLIO_EVIDENCE_ADMITTED_LISTING_TICKER_AUTHORITY"] = (
        ADMITTED_LISTING_AUTHORITY
    )
    entries: tuple[AdmittedListingTicker, ...] = Field(max_length=2048)
    authority_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_authority(self) -> Self:
        """Require sorted unique listings and the exact admitted-authority identity.

        Returns:
            This contract after checking its declared consistency rules.

        Raises:
            ValueError: Listing identifiers repeat/are unordered or authority_hash is inconsistent.
        """
        listings = tuple(value.listing_id for value in self.entries)
        if listings != tuple(sorted(set(listings))):
            raise ValueError("portfolio_evidence.listing_authority_invalid")
        _identity(self, "authority_hash")
        return self

    @property
    def ticker_by_listing(self) -> dict[str, str]:
        """Project admitted trading symbols by exact listing identifier.

        Returns:
            New listing-to-ticker mapping from the validated authority entries.
        """
        return {value.listing_id: value.ticker for value in self.entries}


POSITION_EPSILON: float = 1e-12
"""Below this a weight is not a position; it is the residue of a subtraction."""

MAXIMUM_SCOPE_ISSUERS: Final = 512
"""The scope names every mapped issuer of the book. Its bound is the book
contract's position bound, not an execution budget: eight is how many issuers
one evidence unit prepares together, and that is the unit owner's constant."""

PortfolioTransition = Literal["OPENED", "INCREASED", "HELD", "REDUCED", "EXITED"]
"""The Portfolio report's own change vocabulary, mirrored rather than imported."""

IssuerSelectionReason = Literal[
    "OPENED_OR_INCREASED_BY_POSITIVE_CHANGE",
    "HELD_BY_ENDING_WEIGHT",
    "REDUCED_OR_EXITED_BY_ABSOLUTE_CHANGE",
]
"""Why an issuer is in scope, as a reason rather than a score."""


def transition_of(*, ending: float, preceding: float) -> PortfolioTransition:
    """The one derivation of a transition from two formation-end weights."""
    held_now = ending > POSITION_EPSILON
    held_before = preceding > POSITION_EPSILON
    if not held_before:
        return "OPENED"
    if not held_now:
        return "EXITED"
    change = ending - preceding
    if change > POSITION_EPSILON:
        return "INCREASED"
    if change < -POSITION_EPSILON:
        return "REDUCED"
    return "HELD"


class PortfolioListingPosition(PortfolioEvidenceContract):
    """One listing of the book, as the report already described it.

    An unmapped row stays in the projection with `ticker=None`. Dropping it
    would remove it from every coverage denominator too.
    """

    listing_id: str = Field(min_length=1, max_length=120)
    ticker: str | None = Field(default=None, max_length=16)
    ending_weight: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    preceding_weight: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    signed_change: float = Field(allow_inf_nan=False)
    transition: PortfolioTransition

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_position(self) -> Self:
        """Require normalized ticker, exact weight change and the derived transition.

        Returns:
            This contract after checking its declared consistency rules.

        Raises:
            ValueError: Ticker normalization, signed_change or transition disagree with the source
                weights.
        """
        if self.ticker is not None and self.ticker != self.ticker.strip().upper():
            raise ValueError("portfolio_evidence.ticker_invalid")
        if self.signed_change != self.ending_weight - self.preceding_weight:
            raise ValueError("portfolio_evidence.position_change_invalid")
        if self.transition != transition_of(
            ending=self.ending_weight, preceding=self.preceding_weight
        ):
            raise ValueError("portfolio_evidence.position_transition_invalid")
        return self


class PortfolioExposureProjection(PortfolioEvidenceContract):
    """The book's holdings and changes, bound to the sealed report that owns them.

    Source-bound rather than recomputed: every row here is read from the
    report's window-end book, and the identities above it are what let a reader
    check that claim. `book_authority` names which kind of sealed book it is; a
    candidate or handoff identity is present exactly when the authority says so.
    """

    kind: Literal["PortfolioExposureProjection"] = "PortfolioExposureProjection"
    book_authority: BookAuthority
    report_hash: str | None = Field(pattern=_HASH)
    update_subject: PortfolioUpdateReviewSubject | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    experiment_subject: PortfolioExperimentReviewSubject | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    result_hash: str | None = Field(default=None, pattern=_HASH)
    window_end_book_hash: str = Field(pattern=_HASH)
    listing_authority_hash: str = Field(pattern=_HASH)
    candidate_hash: str | None = Field(default=None, pattern=_HASH)
    handoff_hash: str | None = Field(default=None, pattern=_HASH)
    formation_session: str = Field(min_length=10, max_length=10)
    change_boundary: str = Field(min_length=1, max_length=64)
    held_count: int = Field(ge=0)
    window_end_effective_n: float = Field(ge=0.0, allow_inf_nan=False)
    positions: tuple[PortfolioListingPosition, ...] = Field(min_length=1, max_length=512)
    projection_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_projection(self) -> Self:
        """Require unique listings, an admitted book authority and exact projection identity.

        Returns:
            This contract after checking its declared consistency rules.

        Raises:
            ValueError: A listing repeats, authority/source identities are inconsistent or
                projection_hash is invalid.
        """
        listing_ids = tuple(value.listing_id for value in self.positions)
        if len(listing_ids) != len(set(listing_ids)):
            raise ValueError("portfolio_evidence.projection_listing_duplicate")
        try:
            validate_review_book_authority(
                authority=self.book_authority,
                report_hash=self.report_hash,
                result_hash=self.result_hash,
                candidate_hash=self.candidate_hash,
                handoff_hash=self.handoff_hash,
                update_subject=self.update_subject,
                experiment_subject=self.experiment_subject,
            )
        except ValueError as error:
            raise ValueError("portfolio_evidence.projection_authority_invalid") from error
        _identity(self, "projection_hash")
        return self


class PortfolioIssuerPosition(PortfolioEvidenceContract):
    """One issuer in scope: its aggregate book position, band and why it was selected."""

    entity_id: str = Field(min_length=1, max_length=32)
    cik: str = Field(pattern=r"^[0-9]{10}$")
    tickers: tuple[str, ...] = Field(min_length=1)
    listing_ids: tuple[str, ...] = Field(min_length=1)
    ending_weight: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    preceding_weight: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    signed_change: float = Field(allow_inf_nan=False)
    transition: PortfolioTransition
    weight_rank: int = Field(ge=1)
    exposure_band: ExposureBand
    selection_reason: IssuerSelectionReason
    selection_rank: int = Field(ge=1, le=MAXIMUM_SCOPE_ISSUERS)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_issuer_position(self) -> Self:
        """Require ordered issuer identifiers and exact change, transition and exposure band.

        Returns:
            This contract after checking its declared consistency rules.

        Raises:
            ValueError: Ticker/listing ordering, derived change/transition or installed exposure
                band are inconsistent.
        """
        if (
            self.tickers != tuple(sorted(set(self.tickers)))
            or self.listing_ids != tuple(sorted(set(self.listing_ids)))
            or self.signed_change != self.ending_weight - self.preceding_weight
            or self.transition
            != transition_of(ending=self.ending_weight, preceding=self.preceding_weight)
            or self.exposure_band
            is not exposure_band(
                ending_weight=self.ending_weight,
                signed_change=self.signed_change,
                weight_rank=self.weight_rank,
            )
        ):
            raise ValueError("portfolio_evidence.issuer_position_invalid")
        return self


class PortfolioIssuerScope(PortfolioEvidenceContract):
    """The sole Portfolio-to-Evidence binding for one sealed book.

    Everything downstream evidence work is allowed to know about this book is
    here, once, with the lineage that proves where it came from. Four coverage
    components and their reasons, not one green/red score.
    """

    kind: Literal["PortfolioIssuerScope"] = "PortfolioIssuerScope"
    book_authority: BookAuthority
    exposure_projection_hash: str = Field(pattern=_HASH)
    registry_hash: str = Field(pattern=_HASH)
    exposure_band_policy: Literal["portfolio-evidence.exposure-band.v1"] = EXPOSURE_BAND_POLICY
    selected_issuers: tuple[PortfolioIssuerPosition, ...] = Field(max_length=MAXIMUM_SCOPE_ISSUERS)
    mapping_failures: tuple[PortfolioIssuerMappingFailure, ...]
    reviewed_ending_weight_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reviewed_absolute_change_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mapping_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    selected_issuer_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mapped_issuer_count: int = Field(ge=0)
    unmapped_ending_weight: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    unmapped_absolute_change: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    unavailable_reasons: tuple[str, ...] = ()
    scope_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_scope(self) -> Self:
        """Require unique selected issuers, consecutive ranks and exact scope identity.

        Returns:
            This contract after checking its declared consistency rules.

        Raises:
            ValueError: An issuer repeats, selection ranks or mapped coverage are invalid, or
                scope_hash is inconsistent.
        """
        entities = tuple(value.entity_id for value in self.selected_issuers)
        ranks = tuple(value.selection_rank for value in self.selected_issuers)
        if len(entities) != len(set(entities)) or ranks != tuple(range(1, len(entities) + 1)):
            raise ValueError("portfolio_evidence.scope_selection_invalid")
        if self.selected_issuers and self.mapped_issuer_count < len(self.selected_issuers):
            raise ValueError("portfolio_evidence.scope_coverage_invalid")
        _identity(self, "scope_hash")
        return self

    @property
    def ordered_entity_ids(self) -> tuple[str, ...]:
        """The evidence axis, in selection order. The only thing Evidence sees."""
        return tuple(value.entity_id for value in self.selected_issuers)

    @property
    def priority_rank(self) -> dict[str, int]:
        """Selection rank by issuer: what is prepared first, never what is prepared."""
        return {value.entity_id: value.selection_rank for value in self.selected_issuers}


def _identity(value: PortfolioEvidenceContract, field: str) -> None:
    expected = canonical_hash(value.model_dump(mode="json", exclude={field}))
    if getattr(value, field) != expected:
        raise ValueError("portfolio_evidence.identity_invalid")


def book_key(
    report_hash: str | None,
    update_subject: BaseModel | None,
    experiment_subject: BaseModel | None,
    authority: BookAuthority,
) -> str:
    """Name one book a CRO review may be about, as the CRO matches a review to its book.

    Args:
        report_hash: The book's declared-path report, when it has one.
        update_subject: The Portfolio update the book is, when it is one.
        experiment_subject: The experiment the book is, when it is one.
        authority: The authority that sealed the book.

    Returns:
        A digest equal for every review of the same book (V187).
    """
    return str(
        canonical_hash(
            {
                "report": report_hash,
                "update": None
                if update_subject is None
                else update_subject.model_dump(mode="json"),
                "experiment": (
                    None
                    if experiment_subject is None
                    else experiment_subject.model_dump(mode="json")
                ),
                "authority": authority.value,
            }
        )
    )


__all__ = [
    "ADMITTED_LISTING_AUTHORITY",
    "EXPOSURE_BAND_POLICY",
    "MAXIMUM_SCOPE_ISSUERS",
    "POSITION_EPSILON",
    "AdmittedListingTicker",
    "AdmittedListingTickerAuthority",
    "BookAuthority",
    "ExposureBand",
    "IssuerSelectionReason",
    "PortfolioEvidenceContract",
    "PortfolioExperimentReviewSubject",
    "PortfolioExposureProjection",
    "PortfolioIssuerMappingFailure",
    "PortfolioIssuerPosition",
    "PortfolioIssuerScope",
    "PortfolioListingPosition",
    "PortfolioTransition",
    "PortfolioUpdateReviewSubject",
    "book_key",
    "exposure_band",
    "seal_portfolio_evidence_contract",
    "transition_of",
    "validate_review_book_authority",
]
