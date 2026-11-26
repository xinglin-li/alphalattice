"""The governed pre-Factor product workspace's paths and its Data-truth preflight.

This is application composition, not another workflow framework: the preflight resolves its
scope and every authority itself and submits only a request.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from alphalattice.control.data_platform.preflight import (
    DataTruthPreflightError,
    DataTruthPreflightRequest,
    DataTruthPreflightResult,
    DataTruthScopeUnavailable,
    blocked_preflight_result,
    routine_preflight_formation_scope,
    run_data_truth_preflight,
)
from alphalattice.interface.local_application.failure_codes import public_failure


@dataclass(frozen=True)
class ProductWorkspacePaths:
    """Name the confined local workspace database, artifacts, reports and source cache."""

    root: Path

    @classmethod
    def at(cls, root: Path) -> ProductWorkspacePaths:
        """Resolve an explicit caller-owned local workspace root.

        Args:
            root: Explicit workspace path.

        Returns:
            Resolved workspace path declaration.
        """
        return cls(Path(root).resolve())

    @property
    def database(self) -> Path:
        """Resolve the market-data.duckdb database beneath this workspace.

        Returns:
            Declared workspace-local path.
        """
        return self.root / "market-data.duckdb"

    @property
    def reports(self) -> Path:
        """Resolve the reports directory beneath this workspace.

        Returns:
            Declared workspace-local path.
        """
        return self.root / "reports"

    @property
    def artifacts(self) -> Path:
        """Resolve the artifacts directory beneath this workspace.

        Returns:
            Declared workspace-local path.
        """
        return self.root / "artifacts"

    @property
    def cache(self) -> Path:
        """Resolve the yfinance cache directory beneath this workspace.

        Returns:
            Declared workspace-local path.
        """
        return self.root / "yfinance-cache"

    def ensure(self) -> None:
        """Create the declared local workspace, report, artifact, cache and staging directories."""
        for path in (
            self.root,
            self.reports,
            self.artifacts,
            self.cache,
            self.root / "staging",
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class PreFactorWorkspaceHost:
    """The workspace's pre-Factor checks: its Data-truth preflight."""

    paths: ProductWorkspacePaths
    profile_path: Path

    def _run_data_truth_preflight(
        self,
        *,
        market_profile_id: str,
        as_of_session: date,
        now: datetime,
    ) -> DataTruthPreflightResult | None:
        """Run the Data-truth preflight over a small matured formation scope.

        The Host resolves the scope and every authority itself and submits only
        a request; nothing here accepts a caller-computed hash or verdict.

        Two failures that look alike are kept apart on purpose. A workspace that
        cannot yet express a matured formation axis has genuinely nothing to
        compare and yields no verdict. Every other failure -- a missing research
        manifest, a session with no qualified members, an uninstalled policy --
        means the evidence should have existed and its authority could not be
        resolved. Collapsing those into "no verdict" would let a resolution
        failure read exactly like a clean run, so they seal a typed blocking
        result instead.
        """

        try:
            formation_sessions = routine_preflight_formation_scope(
                as_of_session=as_of_session,
                as_of_timestamp=now,
            )
        except DataTruthScopeUnavailable:
            return None
        request = DataTruthPreflightRequest(
            workspace=self.paths.root,
            market_profile_id=market_profile_id,
            formation_sessions=formation_sessions,
        )
        try:
            return run_data_truth_preflight(request, as_of_timestamp=now)
        except DataTruthScopeUnavailable:
            return None
        except DataTruthPreflightError as error:
            return blocked_preflight_result(
                request=request,
                failure_code=public_failure(error, "data_truth.preflight_refused"),
            )
