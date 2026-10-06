"""The stored spellings NM2 renamed, which this release no longer reads (V451, V481).

NM2 (2026-10-02) gave meaningful spellings to the ids, kinds, keys and codes that no sealed root
this tree must read held. A workspace prepared before then still holds the old spellings in what
it stored: its manifest's artifact keys, its component recipes' disposition, its books' kinds.
Read here, such an object fails its contract. Its refusal says why by name: the workspace was
prepared before the renames, and a new workspace is the way on; the old one is left as it is.

The ids a must-read root holds are not here: they keep their stored spellings (V469). A spelling
ending in `:` is a prefix its stored keys begin with.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import ValidationError

PREPARED_BEFORE_RENAMES: Final = "research_workspace.prepared_before_renames"
"""The refusal of a stored object holding a retired spelling; its subject names the spelling."""

RETIRED_SPELLINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "STATIC_PRIMARY_EQUAL_EW": "FOUR_COMPONENT_BOOK",
        "IW184_PUBLIC_TRANCHE_BOOK": "BROAD_FEATURE_BOOK",
        "IW184_PRODUCT_SCORE": "BROAD_ENSEMBLE",
        "G2_R0_ORDERED_18": "TREND_FEATURE_AXIS",
        "G6_R0_ORDERED_20": "REBOUND_CONTEXT_AXIS",
        "G7_R1_ORDERED_44": "MOMENTUM_CONTEXT_AXIS",
        "IW184_FORWARD_EXCESS": "HISTORICAL_ENSEMBLE_TARGET",
        "WHOLE_UNIVERSE_CANONICAL_G0": "UNIVERSE_RETURN_NORMALIZATION",
        "EXACT_INSTALLED_IW184_L2": "BROAD_SQUARED_ERROR",
        "C1_WHOLE_BOOK_HYSTERESIS_EQUAL_WEIGHT": "WHOLE_BOOK_HYSTERESIS_EQUAL_WEIGHT",
        "C2_WHOLE_BOOK_HYSTERESIS_INVERSE_VOLATILITY": "WHOLE_BOOK_HYSTERESIS_INVERSE_VOLATILITY",
        "C6_WHOLE_BOOK_HYSTERESIS_CAUSAL_RANK_MU_DIAGONAL_TILT": (
            "WHOLE_BOOK_HYSTERESIS_CAUSAL_RANK_MU_DIAGONAL_TILT"
        ),
        "GLOBAL_IW184": "BROAD_FEATURE_MODEL",
        "IW184": "BROAD_ENSEMBLE_MODEL",
        "GateIComponentBookRecipe": "ComponentBookRecipe",
        "C1WholeBookHysteresisEqualWeightRecipe": "WholeBookHysteresisEqualWeightRecipe",
        "C2WholeBookHysteresisInverseVolatilityRecipe": (
            "WholeBookHysteresisInverseVolatilityRecipe"
        ),
        "C6WholeBookHysteresisCausalRankMuRecipe": "WholeBookHysteresisCausalRankMuRecipe",
        "GATE_I_FROZEN_RESEARCH_PACKAGE_ADMITTED_AS_SUCCESSOR": (
            "FROZEN_RESEARCH_PACKAGE_ADMITTED_AS_SUCCESSOR"
        ),
        "IW184_EVIDENCE_ROOT": "BROAD_ENSEMBLE_EVIDENCE_ROOT",
        "HETEROGENEOUS_GATE_I_PACKAGE_ROOT": "HETEROGENEOUS_PACKAGE_ROOT",
        "HETEROGENEOUS_GATE_I_SCORE:": "HETEROGENEOUS_COMPONENT_SCORE:",
        "IW184_DEVELOPMENT_REPLAY_SCORE": "BROAD_ENSEMBLE_DEVELOPMENT_REPLAY_SCORE",
        "GATE_I_COMPONENT_SCORE_ARRAY": "COMPONENT_SCORE_ARRAY",
        "ALPHA_SCORES_REPLAYED_FROM_THE_ADMITTED_IW184_EVIDENCE_MANIFEST": (
            "ALPHA_SCORES_REPLAYED_FROM_THE_ADMITTED_ENSEMBLE_EVIDENCE_MANIFEST"
        ),
        "GATE_I_WEIGHT_PARITY_ASSUMES_FULL_EXECUTION_AVAILABILITY": (
            "WEIGHT_PARITY_ASSUMES_FULL_EXECUTION_AVAILABILITY"
        ),
        "GATE_V_WEIGHTS_ARE_PARITY_ORACLES_NOT_RUNTIME_INPUTS": (
            "WEIGHTS_ARE_PARITY_ORACLES_NOT_RUNTIME_INPUTS"
        ),
        "LOCAL_FROZEN_RECIPE_RECONSTRUCTION_NOT_ORIGINAL_GATE_RESULTS": (
            "LOCAL_FROZEN_RECIPE_RECONSTRUCTION_NOT_ORIGINAL_RESEARCH_RESULTS"
        ),
        "LOCAL_MODEL_LIFECYCLE_RESEARCH_NOT_ORIGINAL_GATE_RESULTS": (
            "LOCAL_MODEL_LIFECYCLE_RESEARCH_NOT_ORIGINAL_RESEARCH_RESULTS"
        ),
        "INSTALLED_R0_R1_CONTENT_ADDRESSED_COVARIANCE_SURFACES": (
            "INSTALLED_RISK_METHOD_CONTENT_ADDRESSED_COVARIANCE_SURFACES"
        ),
        "alpha_research.heterogeneous_live_g2_candidate_support_insufficient": (
            "alpha_research.heterogeneous_live_trend_candidate_support_insufficient"
        ),
        "alpha_research.heterogeneous_live_g7_candidate_support_insufficient": (
            "alpha_research.heterogeneous_live_contextual_momentum_candidate_support_insufficient"
        ),
    }
)

_PREFIXES: Final = tuple(spelling for spelling in RETIRED_SPELLINGS if spelling.endswith(":"))
_WHOLE: Final = frozenset(spelling for spelling in RETIRED_SPELLINGS if not spelling.endswith(":"))
_SUBJECT_PART: Final = re.compile(r"[,:=]")


def retired_spelling_in(value: object) -> str | None:
    """The first retired spelling a stored value holds whole, as a key or a value, at any depth.

    Args:
        value: A stored object as JSON reads it, or any part of one.

    Returns:
        The retired spelling, or None when the value holds none.
    """
    if isinstance(value, str):
        if value in _WHOLE:
            return value
        return next((prefix for prefix in _PREFIXES if value.startswith(prefix)), None)
    if isinstance(value, Mapping):
        for key, item in value.items():
            found = retired_spelling_in(key) or retired_spelling_in(item)
            if found is not None:
                return found
        return None
    if isinstance(value, list | tuple):
        for item in value:
            found = retired_spelling_in(item)
            if found is not None:
                return found
    return None


def retired_spelling_of(error: BaseException) -> str | None:
    """The retired spelling a stored object's failure names, where it names one.

    A contract's failure names it as an input it refused, or as the subject of an owner's code
    its validator raised (`research_workspace.artifact_key_unknown:IW184_EVIDENCE_ROOT`);
    any other failure names it as the subject of the code it carries.

    Args:
        error: The failure of a stored object's read.

    Returns:
        The retired spelling, or None when the failure names none.
    """
    if isinstance(error, ValidationError):
        for item in error.errors(include_url=False):
            found = retired_spelling_in(item.get("input"))
            cause = (item.get("ctx") or {}).get("error")
            if found is None and cause is not None:
                found = _in_code(str(cause))
            if found is not None:
                return found
        return None
    return _in_code(str(error)) if isinstance(error, ValueError) else None


def _in_code(code: str) -> str | None:
    """A retired spelling a code's subject names: one of its `,`, `:` or `=` separated parts."""
    _base, _colon, subject = code.partition(":")
    if not subject or any(character.isspace() for character in subject):
        return None
    whole = retired_spelling_in(subject)
    if whole is not None:
        return whole
    return next(
        (
            found
            for part in _SUBJECT_PART.split(subject)
            if (found := retired_spelling_in(part)) is not None
        ),
        None,
    )


__all__ = [
    "PREPARED_BEFORE_RENAMES",
    "RETIRED_SPELLINGS",
    "retired_spelling_in",
    "retired_spelling_of",
]
