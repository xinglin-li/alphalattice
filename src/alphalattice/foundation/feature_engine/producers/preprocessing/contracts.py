"""Contracts for installed Panel preprocessing methods and their clipping evidence.

Panel preprocessing used to be an engine invariant: the kernel winsorized,
neutralized and standardized because that is what the code did, and the only
record of it was a policy hash derived from constants. That is enough to detect
a changed constant and not enough to answer the question a researcher actually
asks -- *what did this transformation do to my data* -- because the clipping was
never evidence, only behaviour.

These contracts make the method an installed capability with a durable receipt.
The recipe says which transformation ran; the binding says which recipe one
Panel was built by; the clipping evidence says what the transformation actually
changed, bound to the exact arrays it changed.
"""

from __future__ import annotations

from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash


class PanelPreprocessingError(ValueError):
    """Stable failure raised before a Panel transformation is admitted."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class PanelPreprocessingRecipe(_Contract):
    """One installed cross-sectional transformation, identified by its content.

    The ordered ``sequence`` is the transformation itself, not a label: two
    recipes that clip and neutralize in different orders are different methods
    even when every constant matches.
    """

    kind: Literal["PanelPreprocessingRecipe"] = "PanelPreprocessingRecipe"
    recipe_id: str = Field(min_length=1, max_length=96)
    sequence: tuple[str, ...] = Field(min_length=1)
    winsor_multiplier: float = Field(gt=0.0)
    mad_scale: float = Field(gt=0.0)
    minimum_coverage: float = Field(gt=0.0, le=1.0)
    minimum_sector_sample: int = Field(ge=0)
    neutralization: Literal["EQUAL_SECTOR_DEMEAN", "NONE"]
    standardization: Literal[
        "GLOBAL_ROBUST_Z",
        "CROSS_SECTIONAL_STD_Z",
        "TRAILING_LISTING_ROBUST_Z",
        "NONE",
    ]
    lookback_sessions: int | None = Field(default=None, ge=1)
    minimum_finite_observations: int | None = Field(default=None, ge=1)
    interaction_clip: float | None = Field(default=None, gt=0.0)
    partial_universe: Literal["fail_closed"] = "fail_closed"
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify the recipe identity against its declared transformation content.

        Returns:
            This validated immutable contract.

        Raises:
            PanelPreprocessingError: The recipe hash differs from its content digest.
        """
        if self.recipe_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"recipe_hash"})
        ):
            raise PanelPreprocessingError("PANEL_PREPROCESSING_RECIPE_IDENTITY_INVALID")
        return self


class PanelPreprocessingImplementationBinding(_Contract):
    """The executable identity behind one installed recipe.

    A recipe says what should happen; this says which code does it, by content.
    The distinction is the whole point: ``implementation_id`` is a name a class
    chooses for itself, and a name survives a rewrite of everything underneath
    it. A build could therefore seal a recipe, run arithmetic that had silently
    changed, and produce evidence that is internally consistent and describes a
    computation nobody performed.

    ``implementation_content_hash`` closes that by hashing the rule closure of the
    modules that actually compute the transformation. Third-party numerics are
    the environment, provenance recorded beside the Panel and never part of this
    identity (LAWS.md ID6): a binding sealed before E0 folded them in as
    ``numerical_environment_hash`` and reads back as it was sealed; recorded
    moves name the binding it became (``catalog.implementation_role``).
    """

    kind: Literal["PanelPreprocessingImplementationBinding"] = (
        "PanelPreprocessingImplementationBinding"
    )
    implementation_id: str = Field(min_length=1, max_length=128)
    """The stable handle. Kept beside the content hash, never instead of it."""

    implementation_owners: tuple[str, ...] = Field(min_length=1)
    """Module names whose syntax composes the rule closure, for readability."""

    implementation_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_environment_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda v: v is None
    )
    """Only on a binding sealed before E0, which folded the environment in."""
    implementation_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        implementation_id: str,
        implementation_owners: tuple[str, ...],
        implementation_content_hash: str,
    ) -> PanelPreprocessingImplementationBinding:
        """Bind an executable handle to its code owners and implementation content.

        Args:
            implementation_id: Stable handle of the installed executable.
            implementation_owners: Unique module names that compute its transformation.
            implementation_content_hash: Measured implementation identity from the rule closure.

        Returns:
            Qualified canonical binding, with no numerical environment added to its identity.

        Raises:
            ValidationError: Fields violate the schema, owners are duplicated, or the
                resulting implementation identity is invalid.
        """
        values: dict[str, object] = {
            "kind": "PanelPreprocessingImplementationBinding",
            "implementation_id": implementation_id,
            "implementation_owners": list(implementation_owners),
            "implementation_content_hash": implementation_content_hash,
        }
        return cls(**values, implementation_binding_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify unique implementation owners and the executable binding identity.

        Returns:
            This validated immutable contract.

        Raises:
            PanelPreprocessingError: Owners are duplicated or the binding hash
                differs from its content.
        """
        if len(set(self.implementation_owners)) != len(self.implementation_owners):
            raise PanelPreprocessingError("PANEL_PREPROCESSING_IMPLEMENTATION_OWNERS_DUPLICATED")
        if self.implementation_binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"implementation_binding_hash"})
        ):
            raise PanelPreprocessingError("PANEL_PREPROCESSING_IMPLEMENTATION_IDENTITY_INVALID")
        return self


class PanelPreprocessingCapability(_Contract):
    """What one installed recipe is, what runs it, and what it may be used for.

    Separate from the recipe because installing a transformation and admitting
    it as the Panel's active method are different decisions; collapsing them
    would make adding a candidate method the same act as activating it.

    The implementation identity is what makes the catalog more than metadata. A
    catalog that names recipes but cannot say which code computes them lets a
    build seal a new recipe's identity while the old kernel does the arithmetic
    -- evidence that is internally consistent and describes a computation that
    never happened. ``implementation_binding_hash`` is the part that survives a
    rename: two adapters may declare the same ``implementation_id`` and still be
    different capabilities when their executable syntax differs.
    """

    kind: Literal["PanelPreprocessingCapability"] = "PanelPreprocessingCapability"
    recipe_id: str = Field(min_length=1, max_length=96)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_id: str = Field(min_length=1, max_length=128)
    """The stable handle for the implementation bound to this recipe."""

    implementation_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Content identity of that implementation. The handle can be reused; this
    cannot, so a silent rewrite behind the same name is visible here."""

    admitted_for_active_panel: bool
    admitted_for_development_overlay: bool = False


class PanelPreprocessingBinding(_Contract):
    """Which installed recipe one Panel materialization was produced by."""

    kind: Literal["PanelPreprocessingBinding"] = "PanelPreprocessingBinding"
    recipe_id: str = Field(min_length=1, max_length=96)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_id: str = Field(min_length=1, max_length=128)
    """The implementation that actually ran, recorded beside the recipe it claims
    to be. The two travelling together is what makes a mismatch detectable."""

    implementation_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Content identity of that implementation, so the record names executable
    content rather than a class's chosen name for itself."""

    implementation: PanelPreprocessingImplementationBinding
    """The implementation binding itself, embedded rather than referenced.

    A hash alone is an opaque leaf: a reader can confirm two documents quote the
    same string and still cannot say which modules were hashed, which versions
    were pinned, or whether the closure it names has anything to do with this
    Panel. Carrying the whole child means the lineage terminates in facts --
    owners, content digest, numerical environment -- rather than in a digest of
    facts nobody holds.

    Embedded rather than published separately because it is small, immutable and
    meaningless apart from the binding that cites it; a second artifact category
    would add a store round-trip and a new way for the pair to go missing.
    """

    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The frozen ``cross_section_policy_hash``. Carried alongside the recipe
    identity rather than replaced by it, so existing Panel identity keeps its
    meaning while the method gains one of its own."""

    panel_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        # The flat hash and the embedded child must agree. Keeping both is what
        # makes a swapped child detectable: the marker quotes the hash, this
        # object holds the content, and a mismatch between them is a tamper
        # rather than a mystery.
        """Verify that the embedded implementation agrees with the flat binding.

        Returns:
            This validated immutable contract.

        Raises:
            PanelPreprocessingError: Implementation name or identity contradicts its embedded child,
                or the Panel preprocessing binding hash is invalid.
        """
        if self.implementation.implementation_binding_hash != self.implementation_binding_hash:
            raise PanelPreprocessingError("PANEL_PREPROCESSING_IMPLEMENTATION_CHILD_MISMATCH")
        if self.implementation.implementation_id != self.implementation_id:
            raise PanelPreprocessingError("PANEL_PREPROCESSING_IMPLEMENTATION_NAME_MISMATCH")
        if self.binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"binding_hash"})
        ):
            raise PanelPreprocessingError("PANEL_PREPROCESSING_BINDING_IDENTITY_INVALID")
        return self


class PanelFactorClippingRecord(_Contract):
    """What the clip did to one factor, per session and in total.

    Per-factor and per-session counts and fractions, with the boundaries that
    produced them. Deliberately not per-observation: a row-level artifact has no
    consumer in this Gate, and a schema without a consumer is a liability rather
    than evidence.
    """

    kind: Literal["PanelFactorClippingRecord"] = "PanelFactorClippingRecord"
    factor_id: str = Field(min_length=1, max_length=128)
    finite_input_count: int = Field(ge=0)
    clipped_count: int = Field(ge=0)
    clipped_fraction: float = Field(ge=0.0, le=1.0)
    per_session_clipped_counts: tuple[int, ...]
    per_session_clipped_fractions: tuple[float, ...]
    boundary_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Content identity of the ordered per-session (lower, upper) bounds."""

    record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify clipping axes, totals, input population, and record identity.

        Returns:
            This validated immutable contract.

        Raises:
            PanelPreprocessingError: Session count and fraction axes differ, totals disagree,
                clipping exceeds finite input, or the record hash is invalid.
        """
        if len(self.per_session_clipped_counts) != len(self.per_session_clipped_fractions):
            raise PanelPreprocessingError("PANEL_CLIPPING_RECORD_SESSION_AXIS_MISMATCH")
        if self.clipped_count != sum(self.per_session_clipped_counts):
            raise PanelPreprocessingError("PANEL_CLIPPING_RECORD_TOTAL_MISMATCH")
        if self.clipped_count > self.finite_input_count:
            raise PanelPreprocessingError("PANEL_CLIPPING_RECORD_COUNT_EXCEEDS_INPUT")
        if self.record_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"record_hash"})
        ):
            raise PanelPreprocessingError("PANEL_CLIPPING_RECORD_IDENTITY_INVALID")
        return self


class PanelFactorClipObservationRecord(_Contract):
    """One factor's clipping facts for one materialization batch, as measured."""

    kind: Literal["PanelFactorClipObservationRecord"] = "PanelFactorClipObservationRecord"
    factor_id: str = Field(min_length=1, max_length=128)
    finite_input_count: int = Field(ge=0)
    per_session_finite_counts: tuple[int, ...]
    per_session_clipped_counts: tuple[int, ...]
    boundary_identity: str = Field(pattern=r"^[0-9a-f]{64}$")


class PanelClipObservationRecord(_Contract):
    """What one materialization batch clipped, keyed by the batch's receipt.

    A Panel is built in session batches and, once a build reuses partitions
    another build wrote, the batches that produced a Panel's cells come from
    several builds. The clipping evidence must still describe the whole Panel
    from measurements taken while the transformation ran, so each batch's
    observations are persisted under the receipt its rows and availability
    already carry. The evidence document is folded from these records; a
    later partial recompute supersedes cells, and the fold takes from each
    record only the cells the Panel still attributes to its receipt.
    """

    kind: Literal["PanelClipObservationRecord"] = "PanelClipObservationRecord"
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    panel_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sessions: tuple[str, ...] = Field(min_length=1)
    ordered_sessions_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_listing_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    factor_ids: tuple[str, ...] = Field(min_length=1)
    raw_input_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    transformed_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    observations: tuple[PanelFactorClipObservationRecord, ...] = Field(min_length=1)
    record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify the measured factor and session axes of one batch receipt.

        Returns:
            This validated immutable contract.

        Raises:
            PanelPreprocessingError: Factor or session order, identities, per-session counts, finite
                totals, or the sealed record identity disagree.
        """
        if tuple(value.factor_id for value in self.observations) != self.factor_ids:
            raise PanelPreprocessingError("PANEL_CLIP_OBSERVATION_FACTOR_AXIS_MISMATCH")
        if self.sessions != tuple(sorted(set(self.sessions))):
            raise PanelPreprocessingError("PANEL_CLIP_OBSERVATION_SESSIONS_INVALID")
        if any(
            len(value.per_session_finite_counts) != len(self.sessions)
            or len(value.per_session_clipped_counts) != len(self.sessions)
            or value.finite_input_count != sum(value.per_session_finite_counts)
            for value in self.observations
        ):
            raise PanelPreprocessingError("PANEL_CLIP_OBSERVATION_SESSION_AXIS_MISMATCH")
        if self.ordered_sessions_hash != canonical_hash(list(self.sessions)):
            raise PanelPreprocessingError("PANEL_CLIP_OBSERVATION_SESSIONS_INVALID")
        identity = self.model_dump(mode="json", exclude={"record_hash"})
        if self.record_hash != canonical_hash(identity):
            raise PanelPreprocessingError("PANEL_CLIP_OBSERVATION_IDENTITY_INVALID")
        return self

    @classmethod
    def seal(cls, values: dict[str, object]) -> PanelClipObservationRecord:
        """Seal measured batch observations and validate their canonical identity.

        Args:
            values: Receipt, Panel binding, ordered axes, array identities,
                and clipping observations.

        Returns:
            Immutable observation record whose identity is derived from the supplied fields.

        Raises:
            ValidationError: Supplied fields violate the observation schema or factor and
                session axes, counts, or derived identity disagree.
        """
        provisional = cls.model_construct(**values, record_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"record_hash"})
        return cast(
            PanelClipObservationRecord,
            cls.model_validate({**values, "record_hash": canonical_hash(identity)}),
        )


class PanelClippingEvidence(_Contract):
    """The durable receipt for one Panel's preprocessing.

    Binds the transformation to the exact arrays on both sides of it. The raw
    input identity is what makes the guarantee checkable: a later reader can
    confirm that preprocessing consumed the inputs it claims and that those
    inputs were not themselves rewritten.
    """

    kind: Literal["PanelClippingEvidence"] = "PanelClippingEvidence"
    preprocessing_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    recipe_id: str = Field(min_length=1, max_length=96)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    panel_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_sessions_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_listing_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_factor_ids: tuple[str, ...] = Field(min_length=1)
    raw_input_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    transformed_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    factor_records: tuple[PanelFactorClippingRecord, ...] = Field(min_length=1)
    total_clipped_count: int = Field(ge=0)
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify the observed clipping totals and bound transformation identity.

        Returns:
            This validated immutable contract.

        Raises:
            PanelPreprocessingError: Factor order or totals disagree, clipping claims an unchanged
                array identity, or the evidence hash is invalid.
        """
        factors = tuple(value.factor_id for value in self.factor_records)
        if factors != self.ordered_factor_ids:
            raise PanelPreprocessingError("PANEL_CLIPPING_EVIDENCE_FACTOR_AXIS_MISMATCH")
        if self.total_clipped_count != sum(value.clipped_count for value in self.factor_records):
            raise PanelPreprocessingError("PANEL_CLIPPING_EVIDENCE_TOTAL_MISMATCH")
        if self.raw_input_identity == self.transformed_identity and self.total_clipped_count:
            # Values were clipped, so the two sides cannot be the same array.
            raise PanelPreprocessingError("PANEL_CLIPPING_EVIDENCE_TRANSFORM_NOT_OBSERVED")
        if self.evidence_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"evidence_hash"})
        ):
            raise PanelPreprocessingError("PANEL_CLIPPING_EVIDENCE_IDENTITY_INVALID")
        return self


class PanelPreprocessingSealMarker(_Contract):
    """Terminal authority tying one published Panel to the method that built it.

    The clipping receipt on its own is an orphan: a well-formed file that
    mentions a Panel proves only that somebody wrote it. This marker is the edge
    that makes the receipt reachable *from* the Panel. It is written last, after
    the binding and the evidence are both durable:

        panel -> marker -> preprocessing binding -> clipping evidence

    Its presence is not by itself proof that the graph is complete, and this
    contract does not claim otherwise. A marker is a set of hashes; only
    resolving them -- loading the binding and the evidence and re-deriving each
    identity from its own fields -- establishes that the method it names exists.
    That resolution lives in ``resolve_panel_preprocessing_lineage``, which
    refuses a marker whose binding is missing or contradicts it.

    Stored under ``panel_content_hash`` rather than ``panel_binding_hash``: one
    binding can be rebuilt over a different session coverage, so keying by the
    binding would let write order decide which method a Panel is said to have
    used. Self-validating, so a tampered marker is refused rather than followed.
    """

    kind: Literal["PanelPreprocessingSealMarker"] = "PanelPreprocessingSealMarker"
    panel_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    panel_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    recipe_id: str = Field(min_length=1, max_length=96)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_id: str = Field(min_length=1, max_length=128)
    implementation_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    clipping_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    marker_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify the terminal marker identity from its own recorded fields.

        Returns:
            This validated immutable contract.

        Raises:
            PanelPreprocessingError: The marker hash differs from its content digest.
        """
        if self.marker_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"marker_hash"})
        ):
            raise PanelPreprocessingError("PANEL_PREPROCESSING_MARKER_IDENTITY_INVALID")
        return self

    @property
    def preprocessing_identity(self) -> str:
        """What a Panel snapshot binds so its identity depends on methodology.

        Without this in the snapshot manifest, two Panels whose numbers happen to
        match are indistinguishable even when different methods produced them.
        """
        return str(
            canonical_hash(
                {
                    "kind": "PanelPreprocessingIdentity",
                    "recipe_id": self.recipe_id,
                    "recipe_hash": self.recipe_hash,
                    "implementation_id": self.implementation_id,
                    "implementation_binding_hash": self.implementation_binding_hash,
                    "catalog_hash": self.catalog_hash,
                    "preprocessing_binding_hash": self.preprocessing_binding_hash,
                    "clipping_evidence_hash": self.clipping_evidence_hash,
                }
            )
        )


def build_panel_preprocessing_seal_marker(
    *,
    binding: PanelPreprocessingBinding,
    evidence: PanelClippingEvidence,
    panel_content_hash: str,
) -> PanelPreprocessingSealMarker:
    """Seal the terminal marker from the graph it names, never from claims.

    Args:
        binding: Qualified method and executable binding for the Panel.
        evidence: Clipping evidence already bound to that method and Panel.
        panel_content_hash: Content identity of the materialized Panel snapshot.

    Returns:
        Canonically hashed terminal marker naming the complete supplied lineage graph.

    Raises:
        PanelPreprocessingError: Evidence contradicts the preprocessing or Panel binding.
        ValidationError: Marker fields violate the schema.
    """
    if evidence.preprocessing_binding_hash != binding.binding_hash:
        raise PanelPreprocessingError("PANEL_PREPROCESSING_MARKER_EVIDENCE_MISMATCH")
    if evidence.panel_binding_hash != binding.panel_binding_hash:
        raise PanelPreprocessingError("PANEL_PREPROCESSING_MARKER_PANEL_MISMATCH")
    values: dict[str, object] = {
        "kind": "PanelPreprocessingSealMarker",
        "panel_binding_hash": binding.panel_binding_hash,
        "panel_content_hash": panel_content_hash,
        "preprocessing_binding_hash": binding.binding_hash,
        "recipe_id": binding.recipe_id,
        "recipe_hash": binding.recipe_hash,
        "implementation_id": binding.implementation_id,
        "implementation_binding_hash": binding.implementation_binding_hash,
        "catalog_hash": binding.catalog_hash,
        "clipping_evidence_hash": evidence.evidence_hash,
    }
    return PanelPreprocessingSealMarker(**values, marker_hash=canonical_hash(values))


__all__ = [
    "PanelClippingEvidence",
    "PanelFactorClippingRecord",
    "PanelPreprocessingBinding",
    "PanelPreprocessingCapability",
    "PanelPreprocessingError",
    "PanelPreprocessingImplementationBinding",
    "PanelPreprocessingRecipe",
    "PanelPreprocessingSealMarker",
    "build_panel_preprocessing_seal_marker",
]
