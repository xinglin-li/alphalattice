"""Bind existing development inputs for workspace-only Local Web research."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PLAYPEN_ROOT = Path(__file__).resolve().parents[1]
if str(PLAYPEN_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PLAYPEN_ROOT / "src"))

from alphalattice.control.product_host.composition.research_workspace import (  # noqa: E402
    initialize_research_workspace,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import (  # noqa: E402
    bind_factor_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-workspace", type=Path, required=True)
    parser.add_argument("--input-id", default="factor-development")
    parser.add_argument("--panel-snapshot")
    parser.add_argument("--outcome-snapshot")
    parser.add_argument("--include-alpha-handoff", action="store_true")
    args = parser.parse_args()
    os.environ["ALPHALATTICE_NETWORK_DISABLED"] = "1"
    try:
        initialize_research_workspace(args.workspace)
        result = bind_factor_inputs(
            workspace=args.workspace,
            source=args.source_workspace,
            input_id=args.input_id,
            panel_snapshot_hash=args.panel_snapshot,
            outcome_snapshot_hash=args.outcome_snapshot,
            include_alpha_handoff=args.include_alpha_handoff,
        )
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "REFUSED", "failure_code": str(error)[:300]}))
        return 2
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
