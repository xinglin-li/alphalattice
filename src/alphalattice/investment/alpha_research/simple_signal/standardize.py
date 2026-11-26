"""The signed cross-sectional standardization, and nothing else.

Pure by construction: no store, no reader, no clock, no I/O. This module's bytes
are the whole of what decides the numbers, which is why they are hashed into the
score's durable binding and why the replay verifier can call it directly.

What it deliberately does not do, stated because each one has been done here
before and had to be withdrawn:

**No clipping at zero.** A negatively-scored name is a view, not an absence. The
policy that filtered on ``score > 0`` made every negative name indistinguishable
and was deleted for it; the ordering below is complete and signed.

**No calibration.** There is no slope, no intercept, no fit of any kind. The
non-negative-slope route -- ``clip(slope, 0, 2)`` against a realized target -- is
superseded, and a score that carried a fitted scalar would not be target-free
whatever the docstring said.

**No dispersion reconstruction.** ``sigma_XS * z`` restores a relative score to a
return scale it was never measured on. The output here stays dimensionless and
the Portfolio side prices it with an explicit, declared preference coefficient.

**No target.** Nothing in this file reads a realized return, an outcome, or an
evaluation. That is what makes the score usable as a fixed instrument while the
Risk and Portfolio arms vary around it.
"""

from __future__ import annotations

from hashlib import sha256

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]

MINIMUM_FINITE_LISTINGS = 2
"""Fewer than two resolved names is not a cross-section.

A single name standardizes to nothing -- the mean is the value and the deviation
is zero -- so the row is left unresolved rather than filled with a zero that
would read as a real neutral view.
"""

STANDARDIZATION_ID = "CROSS_SECTIONAL_MEAN_ZERO_UNIT_SAMPLE_STD_MIN_2_FINITE"
"""The rule, named including its degrees of freedom.

``ddof=1``. The choice matters at small cross-sections and is invisible at large
ones, which is exactly the kind of thing that must be in the identity rather than
in a reader's assumption.
"""


class SimpleSignalError(ValueError):
    """Stable fail-closed boundary for an unusable simple score."""


def standardize_cross_section(values: FloatArray) -> FloatArray:
    """Standardize each formation row to zero mean and unit sample deviation.

    Unresolved cells stay unresolved and take no part in the moments: a NaN is a
    name this formation has no view on, and treating it as a zero would pull the
    mean toward a view nobody expressed. A row whose resolved names carry no
    dispersion is left entirely unresolved, for the reason above.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise SimpleSignalError("alpha_research.simple_signal_axis_invalid")
    standardized: FloatArray = np.full(array.shape, np.nan, dtype=np.float64)
    for row in range(array.shape[0]):
        finite = np.isfinite(array[row])
        if int(finite.sum()) < MINIMUM_FINITE_LISTINGS:
            continue
        observed = array[row][finite]
        deviation = float(np.std(observed, ddof=1))
        if not np.isfinite(deviation) or deviation <= 0.0:
            continue
        standardized[row][finite] = (observed - float(np.mean(observed))) / deviation
    return standardized


def simple_score_matrix(
    *,
    standardized_values: FloatArray,
    score_sessions: tuple[object, ...],
    score_listing_ids: tuple[str, ...],
    formation_sessions: tuple[object, ...],
    ordered_listing_ids: tuple[str, ...],
) -> FloatArray:
    """Project a published score onto a campaign's own axis.

    Mirrors ``applied_expected_return_matrix``: cells the score does not cover
    come back ``NaN`` rather than zero, because a zero is a neutral view and an
    absence is not one. A campaign axis naming a session or a listing the score
    never carried is a refusal, not a gap to fill -- the score's own cross-section
    is what its z values were measured against, and quietly widening it would
    change what every number means.
    """
    array = np.asarray(standardized_values, dtype=np.float64)
    if array.shape != (len(score_sessions), len(score_listing_ids)):
        raise SimpleSignalError("alpha_research.simple_signal_axis_invalid")
    session_at = {value: index for index, value in enumerate(score_sessions)}
    listing_at = {value: index for index, value in enumerate(score_listing_ids)}
    missing_sessions = [value for value in formation_sessions if value not in session_at]
    if missing_sessions:
        raise SimpleSignalError(
            "alpha_research.simple_signal_session_not_covered:" + str(missing_sessions[0])
        )
    missing_listings = [value for value in ordered_listing_ids if value not in listing_at]
    if missing_listings:
        raise SimpleSignalError(
            "alpha_research.simple_signal_listing_not_covered:" + missing_listings[0]
        )
    rows: npt.NDArray[np.int64] = np.asarray(
        [session_at[value] for value in formation_sessions], dtype=np.int64
    )
    columns: npt.NDArray[np.int64] = np.asarray(
        [listing_at[value] for value in ordered_listing_ids], dtype=np.int64
    )
    projected: FloatArray = np.ascontiguousarray(array[np.ix_(rows, columns)], dtype=np.float64)
    projected.setflags(write=False)
    return projected


def score_values_identity(values: FloatArray) -> str:
    """Byte identity of a score surface, in one construction.

    The same shape/dtype/content digest the Risk chunk store uses, so a rebuilt
    surface is held against the published digest rather than against a second
    convention that happens to agree today.
    """
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    digest = sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(b"float64")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def finite_listing_counts(values: FloatArray) -> npt.NDArray[np.int64]:
    """Resolved names per formation, for a coverage check before anything runs."""
    return np.asarray(np.isfinite(np.asarray(values, dtype=np.float64)).sum(axis=1), dtype=np.int64)


__all__ = [
    "MINIMUM_FINITE_LISTINGS",
    "STANDARDIZATION_ID",
    "FloatArray",
    "SimpleSignalError",
    "finite_listing_counts",
    "score_values_identity",
    "simple_score_matrix",
    "standardize_cross_section",
]
