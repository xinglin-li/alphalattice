"""What makes one published return-unit calibration an *installed* result.

``AlphaReturnUnitCalibrationEvidence`` proves that a document agrees with
itself. Every field is inside ``evidence_hash``, so a tampered artifact fails --
but a *re-sealed* one does not. Anyone able to write the file can change a slope,
change the applied lane beside it, recompute ``evidence_hash`` over the changed
payload, and publish something that validates perfectly and says whatever they
wanted it to say. Nothing in the artifact establishes that the installed method
would produce those numbers, because nothing in the artifact runs the method.

So the authority is here rather than in the document. Two pieces:

* an **installed capability identity** -- the method and recipe that are allowed
  to produce a calibration, the exact implementation bytes that compute it, and
  the numerical environment it was computed in. Host-derived from installed
  source, never supplied by a caller, and deliberately narrow: it names the four
  Alpha modules that decide the number and nothing else. It is not a cross-Desk
  catalog and does not describe any other capability.

* a **resolver** that re-derives the whole calibration from the canonical target
  materialization, the fold-selected score projection and the Stage 3 row axis,
  and admits the published artifact only when the re-derivation reproduces its
  slopes, its applied lane and its identity.

A consumer takes the resolver's output. It never decodes the artifact's own
``transformed_values_hex``, because reading a number out of the document under
test is not a check on it.

An artifact this build cannot re-derive -- a different recipe, a different method
composition, a different numerical environment -- stays readable and is returned
with ``READBACK_ONLY_NO_INSTALLED_AUTHORITY``. That is not a refusal and not an
upgrade: it is the statement that the document can be read and that nothing here
vouches for its numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.investment.alpha_research.experiments.selected_scores import (
    AlphaSelectedRowAxis,
    AlphaSelectedScoreProjection,
)
from alphalattice.investment.alpha_research.targets.materialization import (
    CanonicalTargetMaterialization,
)
from alphalattice.investment.alpha_research.verification.identity import (
    alpha_numerical_environment_identity,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.kernel.shared_kernel.source_identity import (
    source_component_id,
    switched_source_identity,
)

from .return_unit import (
    RETURN_UNIT_CALIBRATION_METHOD_ID,
    RETURN_UNIT_COMPOSITION,
    AlphaReturnUnitCalibration,
    AlphaReturnUnitCalibrationError,
    AlphaReturnUnitCalibrationEvidence,
    calibrate_alpha_return_unit_signal,
)
from .stock_returns import StockCalibrationEvidence, StockCalibrationRecipe

type FloatArray = npt.NDArray[np.float64]

_HASH = r"^[0-9a-f]{64}$"

type CalibrationDisposition = Literal[
    "REDERIVED_UNDER_INSTALLED_CAPABILITY",
    "READBACK_ONLY_NO_INSTALLED_AUTHORITY",
    "READBACK_ONLY_UNVERIFIED_CAPABILITY_MISMATCH",
]
"""Three readings of a published calibration, and the middle two are not the same.

``READBACK_ONLY_NO_INSTALLED_AUTHORITY`` is a document that never recorded what
produced it -- published before the capability was durable, so there is nothing to
compare and nothing was ever claimed.

``READBACK_ONLY_UNVERIFIED_CAPABILITY_MISMATCH`` is a document that *did* record
a capability and recorded one this build is not. An earlier name for this called
it *drift*, which asserted more than two unequal hashes can support: a capability
binding is not resolvable from its hash, so nothing here can tell an honest
artifact from another build apart from one whose capability field was simply
typed in. The name now says what is known -- the capability does not match, and
this build did not verify why -- and the outcome is the same either way, which is
that nothing is admitted.
"""


def _implementation_sources() -> dict[str, Path]:
    """The four modules that decide the number, keyed by semantic component id.

    Narrow on purpose. A closure that reached the whole Desk would rotate this
    capability's identity every time an unrelated Alpha module changed, and an
    identity that moves for reasons nobody can point at stops being read.
    """

    here = Path(__file__)
    experiments = here.parent.parent / "experiments"
    return {
        source_component_id(package_id="alpha_research", source_path=path): path
        for path in (
            here,
            here.with_name("return_unit.py"),
            here.with_name("stock_returns.py"),
            experiments / "selected_scores.py",
        )
    }


class AlphaReturnUnitCapabilityBinding(BaseModel):  # type: ignore[misc]
    """The installed authority a published calibration has to have come from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["AlphaReturnUnitCapabilityBinding"] = "AlphaReturnUnitCapabilityBinding"
    schema_version: Literal["alpha-return-unit-capability@1"] = "alpha-return-unit-capability@1"

    # --- what the method is -------------------------------------------------
    method_id: str = Field(min_length=1, max_length=128)
    composition: str = Field(min_length=1, max_length=200)
    slope_units: Literal["DIMENSIONLESS_RETURN_PER_RETURN"] = "DIMENSIONLESS_RETURN_PER_RETURN"
    recipe_hash: str = Field(pattern=_HASH)
    """Identity of the installed bounded nonnegative calibration policy.

    Carried separately from the implementation closure because the two answer
    different questions: the recipe says which bounds and prior strength were
    applied, the closure says which code applied them.
    """

    # --- what computes it ---------------------------------------------------
    implementation_closure_hash: str = Field(pattern=_HASH)
    numerical_environment_hash: str | None = Field(
        default=None, pattern=_HASH, exclude_if=lambda v: v is None
    )
    """Only on a capability sealed before E0: the environment is provenance (LAWS.md ID6),
    recorded on each calibration instead."""

    capability_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal the declared return-unit capability and implementation authority.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical capability_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        draft = dict(values)
        draft.pop("capability_hash", None)
        provisional = cls.model_construct(**draft, capability_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"capability_hash"})
        return cls(**draft, capability_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the exact installed return-unit capability binding identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaReturnUnitCalibrationError: capability_hash differs from the complete declared
                binding payload.
        """
        if self.capability_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"capability_hash"})
        ):
            raise AlphaReturnUnitCalibrationError(
                "alpha_research.return_unit_capability_identity_invalid"
            )
        return self


def installed_alpha_return_unit_capability() -> AlphaReturnUnitCapabilityBinding:
    """Derive the installed capability identity from source. No parameters.

    There is nothing to pass, which is the point: a function that accepted an
    expected hash would let whoever supplies it also seal it, and the check would
    then establish only that the caller agreed with themselves.
    """
    return AlphaReturnUnitCapabilityBinding.create(
        method_id=RETURN_UNIT_CALIBRATION_METHOD_ID,
        composition=RETURN_UNIT_COMPOSITION,
        recipe_hash=StockCalibrationRecipe.create().recipe_hash,
        implementation_closure_hash=switched_source_identity(
            _implementation_sources(),
            semantic_owner="alpha_research.calibration",
            numerical_role="return-unit-calibration",
        ),
    )


RETURN_UNIT_CAPABILITY_ROLE = "alpha_research.return_unit_capability"
"""The identity role of the installed return-unit capability (`config/identity-roles.json`)."""


@dataclass(frozen=True, slots=True)
class ResolvedAlphaReturnUnitCalibration:
    """One published calibration, and what this build is willing to say about it."""

    evidence: AlphaReturnUnitCalibrationEvidence
    capability: AlphaReturnUnitCapabilityBinding
    disposition: CalibrationDisposition
    applied_expected_returns: FloatArray | None
    """The out-of-fold ``slope * dispersion * predicted_z`` lane, **recomputed**.

    ``None`` under readback. Never decoded from the artifact: the artifact is the
    thing under test, and a consumer reading its published lane would be trusting
    exactly the field a re-sealed forgery moves.
    """

    scaled_scores: FloatArray | None
    economic_returns: FloatArray | None

    row_axis: AlphaSelectedRowAxis | None = None
    """The exact ordered rows the applied lane sits on. ``None`` under readback.

    Carried because the applied lane is a row vector and every consumer needs it
    on a formation-by-listing axis. A consumer that rebuilt the row axis itself
    would be reconstructing the pairing the calibration already fixed, and a
    reconstruction that disagreed would pair signal with the wrong listing while
    still producing a full matrix.
    """

    target_recipe_id: str | None = None
    """Which canonical target method the rebuilt surface was compiled under.

    Read off the durable recipe binding rather than assumed. With more than one
    installed target method, a consumer that hardcoded the recipe id would record
    a lineage it did not verify.
    """

    @property
    def rederived(self) -> bool:
        """Report whether calibration was rederived under the installed capability.

        Returns:
            True only for REDERIVED_UNDER_INSTALLED_CAPABILITY disposition.
        """
        return self.disposition == "REDERIVED_UNDER_INSTALLED_CAPABILITY"


def readback_alpha_return_unit_calibration(
    evidence: AlphaReturnUnitCalibrationEvidence,
    *,
    disposition: CalibrationDisposition = "READBACK_ONLY_NO_INSTALLED_AUTHORITY",
) -> ResolvedAlphaReturnUnitCalibration:
    """Read one calibration without claiming anything about its numbers.

    The path for an artifact whose inputs are not available here, or that names a
    capability this build is not. It carries no applied lane, so a consumer cannot
    accidentally use it and then discover later that nothing checked it.

    The installed capability is derived here and cannot be passed in. A parameter
    for it would let a caller state what "installed" means and then be told its
    artifact matches, which is the check inverted.
    """
    if disposition == "REDERIVED_UNDER_INSTALLED_CAPABILITY":
        raise AlphaReturnUnitCalibrationError(
            "alpha_research.return_unit_calibration_readback_cannot_claim_rederivation"
        )
    return ResolvedAlphaReturnUnitCalibration(
        evidence=evidence,
        capability=installed_alpha_return_unit_capability(),
        disposition=disposition,
        applied_expected_returns=None,
        scaled_scores=None,
        economic_returns=None,
    )


def seal_installed_return_unit_calibration(
    *,
    projection: AlphaSelectedScoreProjection,
    row_axis: AlphaSelectedRowAxis,
    materialization: CanonicalTargetMaterialization,
    superseded: StockCalibrationEvidence,
) -> AlphaReturnUnitCalibration:
    """Fit one calibration and write the installed authority into it.

    The only writer. It resolves the installed capability first, hands the
    numerical function the environment that capability names, and then seals the
    capability's own identity into the evidence -- so a published calibration says
    which method, which code and which environment was allowed to produce it, and
    a verifier has something to compare against other than the document itself.

    The numerical function below is deliberately left ignorant of all of this. A
    primitive that filled in its own capability hash would be asserting its own
    authority, which is the defect this exists to close rather than a fix for it.

    There is no capability parameter, and there must not be. A writer that
    accepted one would let a caller name the authority its own output is then
    sealed under, so the artifact would record whatever the caller preferred to
    have been true.
    """
    installed = installed_alpha_return_unit_capability()
    fitted = calibrate_alpha_return_unit_signal(
        projection=projection,
        row_axis=row_axis,
        materialization=materialization,
        superseded=superseded,
        # The environment this calibration ran in, recorded beside it (LAWS.md ID6).
        numerical_environment_hash=str(canonical_hash(alpha_numerical_environment_identity())),
    )
    sealed = AlphaReturnUnitCalibrationEvidence.create(
        **{
            **fitted.evidence.model_dump(exclude={"evidence_hash", "calibration"}),
            "calibration": fitted.evidence.calibration,
            "capability_binding_hash": installed.capability_hash,
            "implementation_closure_hash": installed.implementation_closure_hash,
        }
    )
    return AlphaReturnUnitCalibration(
        evidence=sealed,
        scaled_scores=fitted.scaled_scores,
        economic_returns=fitted.economic_returns,
        applied_expected_returns=fitted.applied_expected_returns,
    )


def resolve_alpha_return_unit_calibration(
    *,
    evidence: AlphaReturnUnitCalibrationEvidence,
    projection: AlphaSelectedScoreProjection,
    row_axis: AlphaSelectedRowAxis,
    materialization: CanonicalTargetMaterialization,
    superseded: StockCalibrationEvidence,
) -> ResolvedAlphaReturnUnitCalibration:
    """Re-run the installed method on the real inputs and hold the artifact to it.

    Everything the published document asserts is recomputed: the scaled score,
    the economic-return row lane, the cross-fitting folds, the normalized
    moments, the slopes, the applied values and the evidence identity. A document
    that survives is one the installed capability actually produces; a document
    that was re-sealed with a better slope fails here, before any consumer sees a
    number and therefore before any optimizer call.

    Four outcomes, and the ordering between them is the point. What the document
    *records about its own authority* is read before any number is recomputed, so
    a calibration from another build is reported as being from another build
    rather than tried under today's method and reported as wrong.

    * carries no capability at all -- published before that was durable -- is
      legacy readback: nothing to compare, and nothing ever claimed;
    * carries one that is not this build's is an unverified capability mismatch,
      which is all two unequal hashes can support: a capability binding is not
      resolvable from its hash, so nothing here can tell an honest other build
      from a value that was typed in. Not admitted either way;
    * carries this build's and reproduces is admitted;
    * carries this build's and does not reproduce is an authority failure, which
      is the only one of the four that is a refusal.

    The installed capability is derived here. There is no parameter for it,
    because a caller able to supply one could supply the artifact's own value and
    be told they match.
    """
    installed = installed_alpha_return_unit_capability()
    if evidence.capability_binding_hash is None:
        return readback_alpha_return_unit_calibration(
            evidence, disposition="READBACK_ONLY_NO_INSTALLED_AUTHORITY"
        )
    if (
        not is_current(
            RETURN_UNIT_CAPABILITY_ROLE, evidence.capability_binding_hash, installed.capability_hash
        )
        or evidence.implementation_closure_hash != installed.implementation_closure_hash
    ):
        return readback_alpha_return_unit_calibration(
            evidence, disposition="READBACK_ONLY_UNVERIFIED_CAPABILITY_MISMATCH"
        )

    rebuilt = seal_installed_return_unit_calibration(
        projection=projection,
        row_axis=row_axis,
        materialization=materialization,
        superseded=superseded,
    )
    if rebuilt.evidence.slope_by_fold != evidence.slope_by_fold:
        raise AlphaReturnUnitCalibrationError(
            "alpha_research.return_unit_calibration_slope_not_rederivable"
        )
    if (
        rebuilt.evidence.calibration.transformed_value_hash
        != evidence.calibration.transformed_value_hash
    ):
        raise AlphaReturnUnitCalibrationError(
            "alpha_research.return_unit_calibration_applied_values_not_rederivable"
        )
    if rebuilt.evidence.model_dump(mode="json") != evidence.model_dump(mode="json"):
        # Slopes and applied lane agree and the document still differs: something
        # outside the fitted numbers was changed -- a lineage hash, an axis
        # digest, a bound identity. Compared whole rather than by identity hash so
        # the refusal does not depend on the sealing convention being the only way
        # two documents can differ.
        raise AlphaReturnUnitCalibrationError(
            "alpha_research.return_unit_calibration_identity_not_rederivable"
        )
    return ResolvedAlphaReturnUnitCalibration(
        evidence=rebuilt.evidence,
        capability=installed,
        disposition="REDERIVED_UNDER_INSTALLED_CAPABILITY",
        applied_expected_returns=rebuilt.applied_expected_returns,
        scaled_scores=rebuilt.scaled_scores,
        economic_returns=rebuilt.economic_returns,
        row_axis=row_axis,
        target_recipe_id=materialization.recipe_binding.target_recipe_id,
    )


__all__ = [
    "AlphaReturnUnitCapabilityBinding",
    "CalibrationDisposition",
    "ResolvedAlphaReturnUnitCalibration",
    "installed_alpha_return_unit_capability",
    "readback_alpha_return_unit_calibration",
    "resolve_alpha_return_unit_calibration",
    "seal_installed_return_unit_calibration",
]
