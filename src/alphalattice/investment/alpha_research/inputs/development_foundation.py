"""The input authority a *development* Alpha experiment actually has.

``ResearchFoundationBinding`` is the frozen post-Factor-Research constitution. It
names four separate pieces of admission evidence -- the factor training outcome
snapshot, the strict screening result, the candidate slate, and the Research Desk
factor input -- each assembled from a different source by
``research_foundation/mandate/foundation.py`` and each read with that meaning by
its own consumers.

A development Alpha run has none of them. It has a published Panel, published
causal outcomes, and one Factor *development* checkpoint. The tempting move is to
fill those four fields with the one hash a development run does have, which
produces a binding that validates, hashes, and is false: four different questions
answered with the same evidence. A downstream reader comparing a screening result
to a candidate slate would find them equal and conclude something that was never
computed.

So development gets its own binding, with fields that say what they are, and the
frozen constitution is left alone. The current and Goal paths keep requiring
``ResearchFoundationBinding`` by name.

The fold path accepts either through ``AlphaFoundationAuthority``, a structural
Protocol covering exactly the attributes ``alpha_research`` reads off a
foundation. ``ResearchFoundationBinding`` satisfies it without modification, so
nothing current changes shape.
"""

from __future__ import annotations

from typing import Literal, Protocol, Self, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from alphalattice.foundation.research_foundation.contracts import ResearchFoundationBinding
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.validation.enums import RebalanceFrequency


@runtime_checkable
class AlphaExecutionOutcomeAuthority(Protocol):
    """The one thing the fold path asks of an execution outcome reference."""

    @property
    def snapshot_hash(self) -> str:
        """Return the admitted execution-outcome snapshot identity.

        Returns:
            Exact source outcome snapshot identity.
        """
        ...


@runtime_checkable
class AlphaFoundationAuthority(Protocol):
    """Every foundation attribute ``alpha_research`` reads, and nothing more.

    Derived by enumerating the reads rather than by copying the frozen contract's
    field list: a Protocol wider than its consumers forces a development binding
    to invent values for questions nobody asks, which is how the falsified
    lineage this replaces came about in the first place.

    Read-only properties throughout, so a plain pydantic field satisfies a member
    without any variance argument.
    """

    @property
    def research_cadence(self) -> RebalanceFrequency:
        """Return the admitted Alpha research formation cadence.

        Returns:
            Rebalance cadence shared by Foundation and execution outcomes.
        """
        ...

    @property
    def feature_panel_snapshot_hash(self) -> str:
        """Return the admitted source Feature Panel snapshot identity.

        Returns:
            Physical source Panel snapshot identity.
        """
        ...

    @property
    def logical_panel_hash(self) -> str | None:
        """Return the optional admitted logical Panel identity.

        Returns:
            Logical Panel identity, or None for an unbound historical record.
        """
        ...

    @property
    def logical_semantic_index_hash(self) -> str | None:
        """Return the optional logical Panel semantic-index identity.

        Returns:
            Logical semantic-index identity, or None when undeclared.
        """
        ...

    @property
    def ordered_factor_ids(self) -> tuple[str, ...]:
        """Return the admitted feature axis in Foundation order.

        Returns:
            Ordered factor identifiers supplied by the Foundation.
        """
        ...

    @property
    def sector_revision(self) -> str | None:
        """Return the optional admitted classification revision.

        Returns:
            Sector revision identity, or None when absent.
        """
        ...

    @property
    def execution_outcome(self) -> AlphaExecutionOutcomeAuthority:
        """Return the admitted execution-outcome authority.

        Returns:
            Outcome authority sharing the declared research cadence.
        """
        ...

    @property
    def foundation_hash(self) -> str:
        """Return the exact admitted Alpha Foundation binding identity.

        Returns:
            Sealed Foundation identity.
        """
        ...


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class AlphaDevelopmentExecutionOutcomeRef(_Contract):
    """The published causal outcome a development run is authorized to read."""

    kind: Literal["AlphaDevelopmentExecutionOutcomeRef"] = "AlphaDevelopmentExecutionOutcomeRef"
    research_cadence: Literal[RebalanceFrequency.DAILY]  # type: ignore[valid-type]
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_ref: str = Field(min_length=1, max_length=512)
    listing_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class AlphaDevelopmentFoundationBinding(_Contract):
    """Development-only input authority, with every field named for what it is.

    ``factor_development_checkpoint_hash`` is the Factor development evidence
    this run was authorized by -- one checkpoint, named once. It is deliberately
    *not* spread across four differently-named fields, and it is deliberately not
    called a screening result or a candidate slate, because a development
    checkpoint is neither: nothing admitted it and no slate was published.
    """

    kind: Literal["AlphaDevelopmentFoundationBinding"] = "AlphaDevelopmentFoundationBinding"
    research_cadence: Literal[RebalanceFrequency.DAILY]  # type: ignore[valid-type]
    feature_panel_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_panel_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_semantic_index_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_outcome: AlphaDevelopmentExecutionOutcomeRef
    ordered_factor_ids: tuple[str, ...] = Field(min_length=1, max_length=128)
    """The declared development feature axis, in the order the Host validated.

    Bounded independently from the frozen constitution because a development
    Campaign may consume current base activations plus admitted extension methods.
    An axis over the bound is rejected by the Host rather than sliced -- a
    truncated axis is a different experiment that still looks like the one
    somebody authored.
    """

    ordered_context_ids: tuple[str, ...] | None = None
    """Day-constant market state columns this run is additionally authorized for.

    Held apart from the Factor axis because the authority is different: no
    Factor run reported on a market regime, and none could -- a session constant
    has no cross-section to screen. What answers for these columns is the raw
    Formula surface the Feature methodology surface already names, so they are
    admitted as context rather than smuggled in as Factors nobody evaluated.
    """

    selected_factor_ids: tuple[str, ...] = Field(min_length=1, max_length=128)
    """The factors the Factor run reported on, and the only ones Alpha may use.

    Separate from ``ordered_factor_ids`` above, which is the statistical universe
    the deterministic program ran over. Collapsing the two would let a document
    draw features the Factor run never reported on, on the strength of a number
    computed over a different question.
    """

    factor_development_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    factor_development_checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    factor_development_program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    factor_development_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sector_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sector_coverage_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """Exact Sector labels over a wider historical input axis, when needed.

    This is input coverage, not a replacement for the Panel's Sector revision.
    Absent on earlier or fixed-roster bindings, whose serialized identity stays
    unchanged under the existing omit-unpopulated rule.
    """
    feature_authority_outcome_snapshot_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    """The outcome clock the Factor evidence answered on, when it is not this run's.

    ``None`` for every one-session run: the Factor checkpoint and the run read
    the same snapshot and the equality is asserted, so there is nothing to state
    twice. For a development-only clock the Factor screening ran on the
    workspace's one-session snapshot while this run reads the longer-horizon
    snapshot, and that fact is carried openly in the binding rather than being
    silently accepted -- a reader deciding whether the feature axis is evidence
    about this run's question needs the clock the axis was actually screened on.
    """

    development_only: Literal[True] = True
    """Stated in the content, so the artifact says so even out of context."""

    research_foundation: ResearchFoundationBinding | None = None
    """Optional admitted research snapshot; never grants current publication."""

    foundation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal Alpha development Foundation feature, context and outcome authority.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical foundation_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        draft = dict(values)
        draft.pop("foundation_hash", None)
        provisional = cls.model_construct(**draft, foundation_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"foundation_hash"})
        return cls(**draft, foundation_hash=canonical_hash(identity))

    @model_serializer(mode="wrap")  # type: ignore[untyped-decorator]
    def _serialize_populated(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Seal what the binding asserts, so a later member cannot move it."""

        serialized: dict[str, object] = handler(self)
        return {key: value for key, value in serialized.items() if value is not None}

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require disjoint feature/context axes and exact Foundation/outcome lineage.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Axes repeat/overlap, selected factors leave context, cadence differs,
                admitted Panel/index/outcome/selection lineage disagrees or foundation_hash is
                inconsistent.
        """
        if self.ordered_factor_ids != tuple(dict.fromkeys(self.ordered_factor_ids)):
            raise ValueError("alpha_research.development_foundation_feature_axis_duplicated")
        context = self.ordered_context_ids or ()
        if context != tuple(dict.fromkeys(context)) or set(context) & set(self.ordered_factor_ids):
            raise ValueError("alpha_research.development_foundation_context_axis_invalid")
        if not set(self.selected_factor_ids).issubset(set(self.ordered_factor_ids)):
            raise ValueError("alpha_research.development_foundation_selection_not_in_context")
        if self.research_cadence is not self.execution_outcome.research_cadence:
            raise ValueError("alpha_research.development_foundation_cadence_mismatch")
        if self.research_foundation is not None:
            admitted = self.research_foundation
            if (
                admitted.feature_panel_snapshot_hash != self.feature_panel_snapshot_hash
                or admitted.logical_panel_hash != self.logical_panel_hash
                or admitted.logical_semantic_index_hash != self.logical_semantic_index_hash
                or admitted.execution_outcome.snapshot_hash != self.execution_outcome.snapshot_hash
                or not set(admitted.ordered_factor_ids).issubset(self.selected_factor_ids)
            ):
                raise ValueError("alpha_research.admitted_foundation_mismatch")
        if self.foundation_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"foundation_hash"})
        ):
            raise ValueError("alpha_research.development_foundation_identity_invalid")
        return self


__all__ = [
    "AlphaDevelopmentExecutionOutcomeRef",
    "AlphaDevelopmentFoundationBinding",
    "AlphaExecutionOutcomeAuthority",
    "AlphaFoundationAuthority",
]
