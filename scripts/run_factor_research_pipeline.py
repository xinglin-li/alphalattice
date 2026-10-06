"""Report the Factor authoring surface this build installs.

``--inspect-authoring`` answers the question a researcher has before any study:
which extension Factors this build installs, each one's implementation identity
and frozen Formula Specification hash, the method mechanics the code already
fixes, the exact scientific choices still unsettled, whether the method is
admitted, and who must act next. It resolves no workspace, produces no evidence
and settles nothing -- an unconfirmed specification is reported as unconfirmed
rather than given a default.

Settled mechanics are reported beside the open choices deliberately. The list
once held four pending items of which two were not decisions at all -- a
numerical defect and an arithmetic identity in two units -- and a reader had no
way to tell those from the two questions that genuinely need a person.

The one-session Factor runtime's ``--replay`` retired with that runtime (RT R35);
Factor evidence is made and replayed by the product's Factor study.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PLAYPEN_ROOT = Path(__file__).resolve().parents[1]
PLAYPEN_SRC = PLAYPEN_ROOT / "src"
if str(PLAYPEN_SRC) not in sys.path:
    sys.path.insert(0, str(PLAYPEN_SRC))

from alphalattice.foundation.feature_engine.catalog.authoring_surface import (
    describe_factor_authoring_surface,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inspect-authoring", action="store_true")
    args = parser.parse_args()
    if not args.inspect_authoring:
        parser.error("give --inspect-authoring; Factor studies run as `alphalattice experiment`")
    surface = describe_factor_authoring_surface()
    print(json.dumps(surface.model_dump(mode="json"), indent=2, sort_keys=True))
    # Reporting a pending method is the correct outcome, not a failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
