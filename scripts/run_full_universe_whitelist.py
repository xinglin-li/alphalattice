"""Print the original project's current-US bootstrap candidates.

This is an explicit, networked bootstrap/rebuild operation.  It is not run at
ordinary application startup: a workspace must persist and use the resulting
frozen manifest, then maintain its data incrementally.

Example:
    .venv/Scripts/python.exe scripts/run_full_universe_whitelist.py --refresh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PLAYPEN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLAYPEN_ROOT / "src"))

from alphalattice.foundation.market_data_ops.sources.universe import (  # noqa: E402
    discover_current_universe_candidates,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="authorize the explicit fetch of the three frozen current-index sources",
    )
    parser.add_argument("--format", choices=("lines", "json"), default="lines")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.refresh:
        print(
            "Refusing implicit network discovery. Re-run with --refresh to build "
            "the current-universe bootstrap candidates.",
            file=sys.stderr,
        )
        return 2
    try:
        bootstrap = discover_current_universe_candidates()
    except Exception as exc:
        print(f"Current-universe discovery failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.format == "lines":
        print("\n".join(bootstrap.candidate_symbols))
        return 0

    manifest = bootstrap.source_manifest
    print(
        json.dumps(
            {
                "candidate_symbols": bootstrap.candidate_symbols,
                "candidate_count": len(bootstrap.candidate_symbols),
                "source_manifest_content_hash": manifest.content_hash,
                "as_of_timestamp": manifest.as_of_timestamp.isoformat(),
                "sources": [
                    {
                        "index": source.index,
                        "source_uri": source.source_uri,
                        "retrieved_at": source.retrieved_at.isoformat(),
                        "response_hash": source.response_hash,
                        "license_class": source.license_class,
                    }
                    for source in manifest.sources
                ],
                "research_whitelist_standard": bootstrap.standard.__dict__,
                "membership_status": "BOOTSTRAP_CANDIDATES_PENDING_QUALITY",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
