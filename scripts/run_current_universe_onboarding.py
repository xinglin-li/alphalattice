"""Run or resume full current-US data onboarding in one local workspace.

New workspace bootstrap (explicit network and durable writes):
    .venv/Scripts/python.exe scripts/run_current_universe_onboarding.py \
        --live-data --refresh-universe

Resume after a rate limit or process stop without re-reading index pages:
    .venv/Scripts/python.exe scripts/run_current_universe_onboarding.py \
        --live-data

Refresh the completed quality-filtered universe on a later session:
    .venv/Scripts/python.exe scripts/run_current_universe_onboarding.py \
        --live-data --maintain-universe
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

PLAYPEN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLAYPEN_ROOT / "src"))

from alphalattice.control.data_platform.preflight import (  # noqa: E402
    resolve_trading_session_authority,
)
from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease  # noqa: E402
from alphalattice.foundation.market_data_ops.runtime.universe_maintenance import (  # noqa: E402
    CurrentUniverseMaintenance,
)
from alphalattice.foundation.market_data_ops.runtime.universe_onboarding import (  # noqa: E402
    CurrentUniverseOnboarding,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (  # noqa: E402
    current_index_profile_id,
)
from alphalattice.foundation.market_data_ops.sources.providers import (  # noqa: E402
    YFinanceMarketDataProvider,
)
from alphalattice.foundation.market_data_ops.sources.universe import (  # noqa: E402
    bootstrap_from_candidate_manifest_document,
    discover_current_universe_candidates,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import (  # noqa: E402
    MarketDataRepository,
)
from alphalattice.kernel.data.calendar import materialize_calendar_schedule  # noqa: E402

DEFAULT_WORKSPACE = PLAYPEN_ROOT / "experiments" / "current-universe-onboarding" / "workspace"
DEFAULT_PROFILE = PLAYPEN_ROOT / "config" / "market-profiles" / "us-current-index-research.yaml"


def _latest_common_us_session(*, on_or_before: date, observed_at: datetime) -> date:
    """Resolve an interactive calendar date to the latest shared US session.

    The durable onboarding contract names an ``as_of_session``. A desktop
    launch on a weekend or US holiday must not freeze a non-session date as if
    it were an observed market bar.
    """

    schedule = materialize_calendar_schedule(
        ("XNAS", "XNYS"),
        start=on_or_before - timedelta(days=14),
        end=on_or_before,
        as_of_timestamp=observed_at,
    )
    calendar_count_by_session: dict[date, int] = {}
    for row in schedule.to_pylist():
        session_close = row["session_close_timestamp"]
        if session_close.tzinfo is None:
            session_close = session_close.replace(tzinfo=UTC)
        if session_close > observed_at:
            continue
        session = row["session_date"]
        calendar_count_by_session[session] = calendar_count_by_session.get(session, 0) + 1
    common = [session for session, count in calendar_count_by_session.items() if count == 2]
    if not common:
        raise ValueError("could not resolve a common XNAS/XNYS as-of session")
    return max(common)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--market-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument(
        "--live-data",
        action="store_true",
        help="authorize yfinance calls and workspace DuckDB mutation",
    )
    parser.add_argument(
        "--refresh-universe",
        action="store_true",
        help="discover a new explicit S&P 500 / NASDAQ-100 / DJIA candidate manifest",
    )
    parser.add_argument(
        "--maintain-universe",
        action="store_true",
        help="advance the latest completed quality-filtered manifest with normal overlap refreshes",
    )
    parser.add_argument(
        "--work-budget",
        type=int,
        help="optional number of complete listing units to advance before returning safely",
    )
    parser.add_argument("--hydration-chunk-size", type=int, default=25)
    parser.add_argument("--hydration-workers", type=int, default=2)
    return parser


def _print_outcome(outcome) -> None:
    manifest = outcome.research_manifest
    print(
        json.dumps(
            {
                "onboarding_id": outcome.onboarding_id,
                "status": outcome.status.value,
                "candidates": outcome.candidates,
                "raw_ready": outcome.raw_ready,
                "quality_eligible": outcome.quality_eligible,
                "feature_ready": outcome.feature_ready,
                "failed": outcome.failed,
                "failure_code": outcome.failure_code,
                "deferred_retry_id": outcome.deferred_retry_id,
                "retry_after_at": (
                    outcome.retry_after_at.isoformat() if outcome.retry_after_at else None
                ),
                "next_workers": outcome.next_workers,
                "research_manifest": (
                    {
                        "manifest_id": manifest.manifest_id,
                        "revision_sha256": manifest.revision_sha256,
                        "listing_count": len(manifest.listings),
                        "data_validity_class": manifest.profile.data_validity_class,
                    }
                    if manifest is not None
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _print_maintenance_outcome(outcome) -> None:
    print(
        json.dumps(
            {
                "maintenance_id": outcome.maintenance_id,
                "status": outcome.status.value,
                "listings": outcome.listings,
                "updated": outcome.updated,
                "failed": outcome.failed,
                "failure_code": outcome.failure_code,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def main() -> int:
    args = _parser().parse_args()
    if not args.live_data:
        print("Refusing database/provider work without --live-data.", file=sys.stderr)
        return 2
    if args.refresh_universe and args.maintain_universe:
        print("Choose exactly one of --refresh-universe or --maintain-universe.", file=sys.stderr)
        return 2
    if args.work_budget is not None and args.work_budget < 1:
        print("--work-budget must be positive.", file=sys.stderr)
        return 2
    if not 1 <= args.hydration_chunk_size <= 50:
        print("--hydration-chunk-size must be between 1 and 50.", file=sys.stderr)
        return 2
    if not 1 <= args.hydration_workers <= 8:
        print("--hydration-workers must be between 1 and 8.", file=sys.stderr)
        return 2
    workspace = args.workspace.resolve()
    profile_path = args.market_profile.resolve()
    if not profile_path.is_file():
        print(f"Market profile does not exist: {profile_path}", file=sys.stderr)
        return 2
    profile_id = current_index_profile_id(profile_path)
    lease = None
    try:
        lease = WorkspaceWriterLease.acquire(workspace)
        store = MarketDataRepository(workspace)
        now = datetime.now(UTC)
        if args.maintain_universe:
            manifest = store.current_quality_filtered_research_manifest(
                market_profile_id=profile_id
            )
            if manifest is None:
                print(
                    "No completed quality-filtered manifest exists. Finish current-universe "
                    "onboarding before routine maintenance.",
                    file=sys.stderr,
                )
                return 2
            resolved_session = _latest_common_us_session(
                on_or_before=args.as_of or now.date(), observed_at=now
            )
            if args.as_of is not None and args.as_of != resolved_session:
                print(
                    "--as-of must be a completed common XNAS/XNYS session; "
                    f"latest eligible session is {resolved_session.isoformat()}.",
                    file=sys.stderr,
                )
                return 2
            qualified_range = store.manifest_raw_range(manifest)
            outcome = CurrentUniverseMaintenance(
                store=store,
                manifest=manifest,
                provider=YFinanceMarketDataProvider(workspace / "yfinance-cache"),
                as_of_session=resolved_session,
                trading_session_authority=resolve_trading_session_authority(
                    start=qualified_range[0] if qualified_range else resolved_session,
                    end=resolved_session,
                    as_of_timestamp=now,
                ),
            ).run(observed_at=now, work_budget=args.work_budget)
            _print_maintenance_outcome(outcome)
            return 0
        if args.refresh_universe:
            active = (
                store.resumable_current_universe_onboarding_input(market_profile_id=profile_id)
                if store.path.exists()
                else None
            )
            if active is not None:
                print(
                    "A current-universe onboarding task is already resumable. "
                    "Resume it before creating a new candidate manifest.",
                    file=sys.stderr,
                )
                return 2
            bootstrap = discover_current_universe_candidates(observed_at=now)
            resolved_session = _latest_common_us_session(
                on_or_before=args.as_of or now.date(), observed_at=now
            )
            if args.as_of is not None and args.as_of != resolved_session:
                print(
                    "--as-of must be a common XNAS/XNYS session; "
                    f"latest eligible session is {resolved_session.isoformat()}.",
                    file=sys.stderr,
                )
                return 2
            as_of_session = resolved_session
        else:
            if not store.path.exists():
                print(
                    "No resumable workspace exists. Start a new explicit bootstrap with "
                    "--refresh-universe.",
                    file=sys.stderr,
                )
                return 2
            resumed = store.resumable_current_universe_onboarding_input(
                market_profile_id=profile_id
            )
            if resumed is None:
                print(
                    "No resumable current-universe onboarding task exists. Use "
                    "--refresh-universe to start a new explicit manifest.",
                    file=sys.stderr,
                )
                return 2
            _onboarding_id, document, history_start, as_of_session = resumed
            bootstrap = bootstrap_from_candidate_manifest_document(document)
            if args.as_of is not None and args.as_of != as_of_session:
                print(
                    "--as-of must match the persisted onboarding task when resuming.",
                    file=sys.stderr,
                )
                return 2
        runner = CurrentUniverseOnboarding(
            store=store,
            bootstrap=bootstrap,
            profile_path=profile_path,
            provider=YFinanceMarketDataProvider(workspace / "yfinance-cache"),
            as_of_session=as_of_session,
            history_start=(history_start if not args.refresh_universe else None),
            hydration_chunk_size=args.hydration_chunk_size,
            hydration_workers=args.hydration_workers,
        )
        outcome = runner.run(observed_at=now, work_budget=args.work_budget)
        _print_outcome(outcome)
        return 0
    except Exception as exc:
        print(f"Current-universe onboarding failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
