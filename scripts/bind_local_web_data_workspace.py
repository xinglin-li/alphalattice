"""Bind an existing qualified data workspace; never download or initialize data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alphalattice.control.product_host.maintenance.data_update import (  # noqa: E402
    bind_existing_data_workspace,
    inspect_existing_data_workspace,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Read input qualification without binding, a writer lease, or data changes.",
    )
    args = parser.parse_args()
    if args.inspect:
        result = inspect_existing_data_workspace(args.workspace)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "QUALIFIED_LOCAL_INPUTS" else 2
    try:
        manifest = bind_existing_data_workspace(args.workspace)
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "REFUSED", "failure_code": str(error)[:300]}))
        return 2
    print(
        json.dumps(
            {
                "status": "BOUND_EXISTING_WORKSPACE",
                "workspace_id": manifest.workspace_id,
                "manifest_hash": manifest.manifest_hash,
                "next_action": "BIND_ADMITTED_RESEARCH_INPUTS",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
