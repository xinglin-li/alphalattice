"""The durable binding for one simple signed score.

A score is only usable as a fixed instrument if a later reader can rebuild it and
get the same numbers. So the binding records everything the pure re-derivation
needs and nothing a caller could substitute for it.

Two fields carry more weight than they look like they do.

``ordered_listing_ids`` is load-bearing, not descriptive. A z is defined relative
to the cross-section it was standardized over: standardizing across 466 names and
then projecting onto 464 gives different numbers than standardizing across 464.
Without the axis in the identity, "the same score" is not a checkable claim.

``implementation_closure_hash`` covers the module whose bytes decide the numbers.
Without it the standardization could be rewritten while every binding stood
still, which is the "same id, changed code" defect the numerical bindings
elsewhere in this tree exist to catch.

The four ``Literal`` fields at the end assert what was *not* done. They look
redundant against the docstrings until you consider what this package replaced: a
frozen-score module that computed ``clip(slope, 0, 2) * dispersion * z`` and had
to be deleted before merge. A reader of a published binding should be able to see
that this one is target-free without reading any source at all.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .provenance import resolve_feature_clock
from .standardize import (
    MINIMUM_FINITE_LISTINGS,
    STANDARDIZATION_ID,
    SimpleSignalError,
)

_HASH = r"^[0-9a-f]{64}$"

SIMPLE_SCORE_CATEGORY = "current/simple-signed-scores"
"""Where a published binding lives inside the Alpha development store."""

SIMPLE_SCORE_VALUE_ARTIFACT_CATEGORY = "development/simple-signed-score-value-artifacts"
SIMPLE_SCORE_VALUE_BINDING_CATEGORY = "development/simple-signed-score-value-bindings"
SIMPLE_SCORE_VALUE_PAYLOAD_CATEGORY = "development/simple-signed-score-value-payloads"

INSTALLED_SIMPLE_SCORE_FEATURES: tuple[str, ...] = ("mom_252_21",)
"""The Panel base columns a simple score may be built from.

One, deliberately. The point of this score is to be fixed and uninteresting while
the Risk and Portfolio arms move; a menu invites choosing the one that made the
study come out well, which is the confound the whole design exists to remove.
Adding a second is a one-line append here plus a new published binding, and it is
a different experiment.
"""

type SimpleSignedScoreMethodId = Literal["MOMENTUM_252_21_SIGNED_CROSS_SECTION"]
FIXED_STUDY_SIMPLE_SCORE_METHOD_ID: SimpleSignedScoreMethodId = (
    "MOMENTUM_252_21_SIGNED_CROSS_SECTION"
)

_INSTALLED_SIMPLE_SCORE_METHODS: dict[SimpleSignedScoreMethodId, str] = {
    "MOMENTUM_252_21_SIGNED_CROSS_SECTION": "mom_252_21",
}


def feature_id_for_simple_score_method(method_id: str) -> str:
    """Resolve one installed target-free score method without a Host branch."""
    try:
        return _INSTALLED_SIMPLE_SCORE_METHODS[method_id]  # type: ignore[index]
    except KeyError as error:
        raise SimpleSignalError(
            "alpha_research.simple_signal_method_not_installed:" + method_id
        ) from error


class SimpleSignedScoreBinding(BaseModel):  # type: ignore[misc]
    """Everything a pure re-derivation of one published score needs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["SimpleSignedScoreBinding"] = "SimpleSignedScoreBinding"
    identity_class: Literal["DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"] = (
        "DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"
    )

    feature_id: str = Field(min_length=1, max_length=96)
    panel_manifest_ref: str = Field(min_length=1, max_length=128)
    panel_snapshot_identity: str = Field(pattern=_HASH)
    """The snapshot's own identity, beside the ref that names it.

    A manifest ref is a name and names are reused. The identity is what makes
    "the same panel" checkable.
    """

    ordered_formation_sessions: tuple[date, ...] = Field(min_length=1)
    ordered_listing_ids: tuple[str, ...] = Field(min_length=2)
    standardization_id: Literal["CROSS_SECTIONAL_MEAN_ZERO_UNIT_SAMPLE_STD_MIN_2_FINITE"] = (
        "CROSS_SECTIONAL_MEAN_ZERO_UNIT_SAMPLE_STD_MIN_2_FINITE"
    )
    """Spelled out rather than referencing ``STANDARDIZATION_ID``.

    The two must agree, and the assertion below is what checks it. Writing the
    constant here instead would make the field's type follow the constant, so a
    change to the rule would silently widen what a published binding may claim.
    """
    minimum_finite_listings: int = Field(ge=2)
    unresolved_policy: Literal["EXCLUDED_FROM_MOMENTS_AND_LEFT_UNRESOLVED"] = (
        "EXCLUDED_FROM_MOMENTS_AND_LEFT_UNRESOLVED"
    )

    # --- observation and availability, and nothing about execution ----------
    #
    # Every field here is derived by the Feature owner's
    # ``observation_clock_for`` from the recipe that already carries it. There is
    # no decision cutoff, no entry offset and no entry timing: an earlier version
    # of this contract had all three, filled with constants copied out of the
    # execution method, which is a producer deciding when its own values are
    # tradable.
    feature_formula_ref: str = Field(min_length=1, max_length=128)
    feature_formula: str = Field(min_length=1, max_length=400)
    feature_window_sessions: int = Field(ge=1, le=2048)
    feature_source_interval: str = Field(min_length=1, max_length=96)
    """The Formula's ordered source rows relative to ``t``, in the owner's words.

    ``[t-252,t-21]`` for this one: the window *ends* at the observation session
    less the Formula's own economic skip. Carried because it is the sentence a
    reader checks the offset below against.
    """

    observation_session_offset_sessions: int = Field(ge=0, le=512)
    availability_delay_sessions: int = Field(ge=0, le=512)
    availability_policy_id: str = Field(min_length=1, max_length=96)
    availability_policy_hash: str = Field(pattern=_HASH)
    observation_clock_hash: str = Field(pattern=_HASH)

    feature_methodology_identity: str = Field(pattern=_HASH)
    feature_catalog_binding_hash: str = Field(pattern=_HASH)

    strategy_scope: Literal["CONDITIONAL_FIXED_SCORE_NOT_A_STRATEGY_SIGNAL"] = (
        "CONDITIONAL_FIXED_SCORE_NOT_A_STRATEGY_SIGNAL"
    )
    """What a result measured against this score may conclude.

    Not modesty, and not a placeholder for a better word later. This score's
    21-session offset is the Formula's own economic skip -- momentum skips the
    most recent month by construction -- and not the strategy's information
    cutoff. It is causally admissible under both the pre-successor Feature clock
    and the successor one, because 21 sessions exceeds the one-session ambiguity
    between them by twenty. A campaign built on it can say what the Risk and
    Portfolio choices were worth *against this score* and nothing else.
    """

    values_identity: str = Field(pattern=_HASH)
    implementation_closure_hash: str = Field(pattern=_HASH)

    resolved_cell_count: int = Field(ge=1)
    minimum_finite_listings_observed: int = Field(ge=0)
    """The worst formation's resolved-name count, published rather than assumed.

    A campaign selecting a top-K larger than this on some formation would refuse
    mid-run; carrying the number lets a preflight say so before anything solves.
    """

    clipping: Literal["NONE"] = "NONE"
    calibration: Literal["NONE_DIMENSIONLESS"] = "NONE_DIMENSIONLESS"
    dispersion_reconstruction: Literal["NONE"] = "NONE"
    target: Literal["NONE_TARGET_FREE"] = "NONE_TARGET_FREE"

    binding_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one declared simple signed-score binding.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical binding_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        draft = dict(values)
        draft.pop("binding_hash", None)
        provisional = cls.model_construct(**draft, binding_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"binding_hash"})
        return cls(**draft, binding_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        # The literal above and the rule the code implements have to be the same
        # string. They are declared twice on purpose -- once as a type, once as
        # the installed constant -- and this is the line that keeps them honest.
        """Reconcile simple-score clock, axes and identity with installed Feature authority.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            SimpleSignalError: Standardization/minimum support, installed Feature
                formula/clock/catalog binding, feature admission, session/listing axes, resolved
                count or binding hash differs.
        """
        if self.standardization_id != STANDARDIZATION_ID:
            raise SimpleSignalError("alpha_research.simple_signal_standardization_not_installed")
        if self.minimum_finite_listings != MINIMUM_FINITE_LISTINGS:
            raise SimpleSignalError("alpha_research.simple_signal_minimum_finite_not_installed")
        # Re-resolved from the Feature owner rather than trusted from the
        # payload. A binding whose declared clock disagrees with the installed
        # catalog is either stale or forged, and both are refusals.
        declared = resolve_feature_clock(self.feature_id)
        if (
            declared.formula_ref != self.feature_formula_ref
            or declared.formula != self.feature_formula
            or declared.window_sessions != self.feature_window_sessions
            or declared.source_interval != self.feature_source_interval
            or declared.observation_session_offset_sessions
            != self.observation_session_offset_sessions
            or declared.availability_delay_sessions != self.availability_delay_sessions
            or declared.availability_policy_id != self.availability_policy_id
            or declared.availability_policy_hash != self.availability_policy_hash
            or declared.observation_clock_hash != self.observation_clock_hash
            or declared.methodology_identity != self.feature_methodology_identity
            or declared.catalog_binding_hash != self.feature_catalog_binding_hash
        ):
            raise SimpleSignalError("alpha_research.simple_signal_feature_clock_drift")
        if self.feature_id not in INSTALLED_SIMPLE_SCORE_FEATURES:
            raise SimpleSignalError(
                "alpha_research.simple_signal_feature_not_installed:" + self.feature_id
            )
        sessions = self.ordered_formation_sessions
        if sessions != tuple(sorted(set(sessions))):
            raise SimpleSignalError("alpha_research.simple_signal_session_axis_unordered")
        listings = self.ordered_listing_ids
        if len(set(listings)) != len(listings):
            raise SimpleSignalError("alpha_research.simple_signal_listing_duplicated")
        if self.resolved_cell_count > len(sessions) * len(listings):
            raise SimpleSignalError("alpha_research.simple_signal_resolved_count_invalid")
        if self.binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"binding_hash"})
        ):
            raise SimpleSignalError("alpha_research.simple_signal_binding_invalid")
        return self


class SimpleSignedScoreValueArtifact(BaseModel):  # type: ignore[misc]
    """Exact packed values for one binding, owned by Alpha rather than a consumer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["SimpleSignedScoreValueArtifact"] = "SimpleSignedScoreValueArtifact"
    identity_class: Literal["DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"] = (
        "DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"
    )
    binding_hash: str = Field(pattern=_HASH)
    values_identity: str = Field(pattern=_HASH)
    payload_hash: str = Field(pattern=_HASH)
    dtype: Literal["<f8"] = "<f8"
    shape: tuple[int, int]
    artifact_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one declared simple-score value artifact.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical artifact_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        draft = dict(values)
        draft.pop("artifact_hash", None)
        provisional = cls.model_construct(**draft, artifact_hash="0" * 64)
        return cls(
            **draft,
            artifact_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"artifact_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require a positive two-dimensional simple-score artifact and exact identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            SimpleSignalError: The declared shape is not positive two-dimensional or its canonical
                artifact hash differs.
        """
        if (
            len(self.shape) != 2
            or any(value < 1 for value in self.shape)
            or self.artifact_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"artifact_hash"}))
        ):
            raise SimpleSignalError("alpha_research.simple_signal_value_artifact_invalid")
        return self


__all__ = [
    "FIXED_STUDY_SIMPLE_SCORE_METHOD_ID",
    "INSTALLED_SIMPLE_SCORE_FEATURES",
    "SIMPLE_SCORE_CATEGORY",
    "SIMPLE_SCORE_VALUE_ARTIFACT_CATEGORY",
    "SIMPLE_SCORE_VALUE_BINDING_CATEGORY",
    "SIMPLE_SCORE_VALUE_PAYLOAD_CATEGORY",
    "SimpleSignedScoreBinding",
    "SimpleSignedScoreMethodId",
    "SimpleSignedScoreValueArtifact",
    "feature_id_for_simple_score_method",
]
