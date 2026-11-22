"""A trial's headline return series, kept with its profile, and the effective number of trials a
class's series stand for.

The series is an inline artifact of the Host's `return_series` type: a JSON body of monthly simple
returns, `{"months": ["YYYY-MM", ...], "returns": [r, ...]}`, months strictly increasing, at most
`POINTS` of them, each return finite. It is kept forever, so a class read after its trial's bodies
are collected still reads it. The class read's effective N correlates the members' series month by
month in DuckDB (`reading.experience_class`); `effective_n` is the method's last step.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from itertools import pairwise
from typing import Any, Final

from alphalattice.kernel.data.errors import DataError

TYPE: Final = "return_series"
"""The Host's artifact type of a trial's headline monthly return series."""
POINTS: Final = 600
"""The most months a series holds: fifty years, far under the 64 KiB an inline body may take."""
MIN_OVERLAP: Final = 12
"""The fewest common months a pair of series is correlated over; a pair with fewer is left out."""
SAMPLE: Final = 200
"""The most series, the newest registered, whose pairwise correlations estimate the class's mean
correlation; every member with a series still counts in M."""
METHOD: Final = "Bailey and Lopez de Prado (2014), effective number of independent trials"
_MONTH = re.compile(r"\d{4}-(0[1-9]|1[0-2])")


def encode(months: Sequence[str], returns: Sequence[float]) -> bytes:
    """The series' inline body; refused by name unless its months are distinct calendar months in
    order, one finite return each, at most `POINTS` of them."""
    if not 1 <= len(months) <= POINTS or len(returns) != len(months):
        raise _invalid(f"a series holds 1 to {POINTS} months, one return each")
    if not all(isinstance(m, str) and _MONTH.fullmatch(m) for m in months):
        raise _invalid("a month is written YYYY-MM")
    if any(b <= a for a, b in pairwise(months)):
        raise _invalid("months are strictly increasing")
    values = [float(r) for r in returns]
    if not all(math.isfinite(r) for r in values):
        raise _invalid("every return is a finite number")
    return json.dumps({"months": list(months), "returns": values}).encode()


def produced(
    trial: str, recipe_key: str, months: Sequence[str], returns: Sequence[float], type_version: int
) -> tuple[dict[str, Any], bytes]:
    """The `artifact.produced` attributes and inline body that keep `trial`'s series: identified
    by its trial's recipe, covering its first to last month."""
    body = encode(months, returns)
    attrs = {
        "type": TYPE,
        "type_version": type_version,
        "identity_kind": "recipe",
        "recipe_key": recipe_key,
        "storage": "inline",
        "meaning": f"Trial {trial}'s headline monthly returns, {months[0]} to {months[-1]}.",
        "trial_id": trial,
        "valid_from": months[0],
        "valid_to": months[-1],
        "row_count": len(months),
        "size_bytes": len(body),
    }
    return attrs, body


def effective_n(mean_correlation: float | None, members: int) -> float | None:
    """N = rho + (1 - rho) * M (Bailey and Lopez de Prado 2014): M series whose mean pairwise
    correlation is rho stand for N independent trials. rho is held to [0, 1], so N runs from 1
    (every series the same) to M (none related). None below two series, or with no pair
    correlated."""
    if members < 2 or mean_correlation is None:
        return None
    rho = min(max(mean_correlation, 0.0), 1.0)
    return rho + (1.0 - rho) * members


def _invalid(why: str) -> DataError:
    return DataError(f"a return series is refused: {why}", code="data_layer.return_series_invalid")
