"""The released versions an upgrade check may start from, each with the ref its check-out uses.

A fixed record names the earlier releases; each later one is registered by the `release.json`
written beside its archived wheel. The caller supplies the record and the field to read.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path


def registered(
    fixed: Mapping[str, str], archives: Callable[[], Path], field: str
) -> dict[str, str]:
    """The fixed record, then each archive folder's `release.json`; a disagreement is refused.

    ``archives`` finds the archive, raising ValueError where none exists (a checkout without one),
    which leaves the fixed record alone.
    """
    refs = dict(fixed)
    try:
        paths = sorted(archives().glob("alphalattice-*/release.json"))
    except ValueError:
        return refs
    for path in paths:
        try:
            record = json.loads(path.read_bytes())
            version, ref = record["version"], record[field]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"release.registration_unreadable:{path}") from error
        named = version == path.parent.name.removeprefix("alphalattice-")
        if (
            not named
            or not re.fullmatch("[0-9a-f]{40}", str(ref))
            or refs.setdefault(version, ref) != ref
        ):
            raise ValueError(f"release.registration_mismatch:{path}")
    return refs
