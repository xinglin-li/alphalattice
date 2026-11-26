"""Local-data construction and storage for Portfolio benchmark inputs."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import date
from math import log
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.capabilities.portfolio_backtesting.benchmark import (
    PortfolioBenchmarkBoundaryError,
    PortfolioBenchmarkSurface,
)
from alphalattice.foundation.data_layer import history
from alphalattice.foundation.feature_engine.producers.reference_data import MarketReference
from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository
from alphalattice.foundation.market_data_ops.publication.projection import (
    action_set_hash as actions_hash,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository

if TYPE_CHECKING:
    from alphalattice.foundation.data_layer.upgrade import Family, UpgradeRunner

_HASH = r"^[0-9a-f]{64}$"
_BENCHMARK_SERIES = ("log_returns", "simple_returns")


class PortfolioBenchmarkSourceAuthority(BaseModel):  # type: ignore[misc]
    """The value-free Market/Feature authority for one SPY return surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["PortfolioBenchmarkSourceAuthority"] = "PortfolioBenchmarkSourceAuthority"
    reference: Literal["SPY"] = "SPY"
    formation_sessions: tuple[date, ...]
    universe_manifest_revision: str = Field(pattern=_HASH)
    raw_input_hash: str = Field(pattern=_HASH)
    action_set_hash: str = Field(pattern=_HASH)
    authority_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(
        cls,
        *,
        formation_sessions: tuple[date, ...],
        universe_manifest_revision: str,
        raw_input_hash: str,
        action_set_hash: str,
    ) -> PortfolioBenchmarkSourceAuthority:
        """Seal the value-free Market/Feature source commitments for one SPY axis.

        Args:
            formation_sessions: Sorted unique formation dates represented by the benchmark source.
            universe_manifest_revision: Universe manifest identity qualified for the source read.
            raw_input_hash: Raw benchmark input commitment.
            action_set_hash: Corporate-action set commitment.

        Returns:
            The validated source authority and its canonical identity.

        Raises:
            PortfolioBenchmarkBoundaryError: The source axis or authority identity is invalid.
        """
        values = cls._identity_values(
            formation_sessions=formation_sessions,
            universe_manifest_revision=universe_manifest_revision,
            raw_input_hash=raw_input_hash,
            action_set_hash=action_set_hash,
        )
        return cls(
            formation_sessions=formation_sessions,
            universe_manifest_revision=universe_manifest_revision,
            raw_input_hash=raw_input_hash,
            action_set_hash=action_set_hash,
            authority_hash=str(canonical_hash(values)),
        )

    @staticmethod
    def _identity_values(
        *,
        formation_sessions: tuple[date, ...],
        universe_manifest_revision: str,
        raw_input_hash: str,
        action_set_hash: str,
    ) -> dict[str, object]:
        return {
            "kind": "PortfolioBenchmarkSourceAuthority",
            "reference": "SPY",
            "formation_sessions": formation_sessions,
            "universe_manifest_revision": universe_manifest_revision,
            "raw_input_hash": raw_input_hash,
            "action_set_hash": action_set_hash,
        }

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require canonical formation order and the value-free source-authority identity.

        Returns:
            This validated benchmark source authority.

        Raises:
            PortfolioBenchmarkBoundaryError: Formation order or the authority hash differs.
        """
        if self.formation_sessions != tuple(sorted(set(self.formation_sessions))):
            raise PortfolioBenchmarkBoundaryError(
                "portfolio_benchmark.source_authority_axis_invalid"
            )
        expected = canonical_hash(
            self._identity_values(
                formation_sessions=self.formation_sessions,
                universe_manifest_revision=self.universe_manifest_revision,
                raw_input_hash=self.raw_input_hash,
                action_set_hash=self.action_set_hash,
            )
        )
        if self.authority_hash != expected:
            raise PortfolioBenchmarkBoundaryError("portfolio_benchmark.source_authority_invalid")
        return self


class PortfolioBenchmarkSourceBinding(BaseModel):  # type: ignore[misc]
    """The campaign and exposure benchmark inputs admitted before values open."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["PortfolioBenchmarkSourceBinding"] = "PortfolioBenchmarkSourceBinding"
    campaign: PortfolioBenchmarkSourceAuthority
    exposure: PortfolioBenchmarkSourceAuthority
    binding_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(
        cls,
        *,
        campaign: PortfolioBenchmarkSourceAuthority,
        exposure: PortfolioBenchmarkSourceAuthority,
    ) -> PortfolioBenchmarkSourceBinding:
        """Bind campaign and exposure benchmark authorities before opening their values.

        Args:
            campaign: Authority for campaign-aligned SPY inputs.
            exposure: Authority for exposure-aligned SPY inputs.

        Returns:
            The source binding sealed from both authority identities.
        """
        values = cls._identity_values(campaign=campaign, exposure=exposure)
        return cls(campaign=campaign, exposure=exposure, binding_hash=str(canonical_hash(values)))

    @staticmethod
    def _identity_values(
        *, campaign: PortfolioBenchmarkSourceAuthority, exposure: PortfolioBenchmarkSourceAuthority
    ) -> dict[str, object]:
        return {
            "kind": "PortfolioBenchmarkSourceBinding",
            "campaign_authority_hash": campaign.authority_hash,
            "exposure_authority_hash": exposure.authority_hash,
        }

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the canonical pair of campaign and exposure authority identities.

        Returns:
            This validated source binding.

        Raises:
            PortfolioBenchmarkBoundaryError: The binding hash differs from its authority pair.
        """
        if self.binding_hash != canonical_hash(
            self._identity_values(campaign=self.campaign, exposure=self.exposure)
        ):
            raise PortfolioBenchmarkBoundaryError("portfolio_benchmark.source_binding_invalid")
        return self


def resolve_portfolio_benchmark_source_authority(
    *, workspace: Path, formation_sessions: tuple[date, ...]
) -> PortfolioBenchmarkSourceAuthority:
    """Reopen only the exact SPY source metadata that fixes one value lane."""
    if formation_sessions != tuple(sorted(set(formation_sessions))):
        raise PortfolioBenchmarkBoundaryError("portfolio_benchmark.axis_invalid")
    market = MarketDataRepository(workspace.resolve())
    parent = market.current_quality_filtered_research_manifest(
        market_profile_id="us-current-index-research"
    )
    if parent is None:
        raise PortfolioBenchmarkBoundaryError("portfolio_benchmark.manifest_missing")
    reference = MarketReference.spy(parent)
    raw_hash, action_hash = _benchmark_source_lineage(
        market=market,
        reference=reference,
        formation_sessions=formation_sessions,
    )
    return PortfolioBenchmarkSourceAuthority.create(
        formation_sessions=formation_sessions,
        universe_manifest_revision=parent.revision_sha256,
        raw_input_hash=raw_hash,
        action_set_hash=action_hash,
    )


def _benchmark_source_lineage(
    *,
    market: MarketDataRepository,
    reference: MarketReference,
    formation_sessions: tuple[date, ...],
) -> tuple[str, str]:
    """Read the owner metadata that ``projected_feature_frame`` later seals.

    This deliberately asks Market Data for a first available date, payload/revision
    hashes, and corporate-action identities only.  It neither opens a price
    field nor projects a Feature/return array during metadata admission.
    """
    market._assert_manifest_scope(reference.manifest, (reference.listing_id,))
    through = max(formation_sessions) + date.resolution * 10
    with market.market_reads(listings=(reference.listing_id,)) as (reading, tables):
        provider, actions = market._provider_actions(reading, (reference.listing_id,), tables)[
            reference.listing_id
        ]
        row = reading.execute(
            f"""
            SELECT min(session_date)
            FROM {tables["bars"]}
            WHERE listing_id = ? AND provider = ?
              AND session_date BETWEEN ? AND ?
            """,
            [reference.listing_id, provider, min(formation_sessions), through],
        ).fetchone()
        if row is None or row[0] is None:
            raise PortfolioBenchmarkBoundaryError("portfolio_benchmark.axis_unavailable")
    connection = market._connect(read_only=True)
    try:  # the raw-evidence hash is a receipt of the workspace's rows
        raw_hash = market._raw_evidence_hash(
            connection,
            listing_id=reference.listing_id,
            provider=provider,
            history_start=cast(date, row[0]),
            history_end=through,
        )
    finally:
        connection.close()
    return raw_hash, actions_hash(actions)


def resolve_portfolio_benchmark_source_binding(
    *,
    workspace: Path,
    campaign_sessions: tuple[date, ...],
    exposure_sessions: tuple[date, ...],
) -> PortfolioBenchmarkSourceBinding:
    """Bind both benchmark axes used by a Portfolio market composition."""
    return PortfolioBenchmarkSourceBinding.create(
        campaign=resolve_portfolio_benchmark_source_authority(
            workspace=workspace, formation_sessions=campaign_sessions
        ),
        exposure=resolve_portfolio_benchmark_source_authority(
            workspace=workspace, formation_sessions=exposure_sessions
        ),
    )


def build_portfolio_benchmark_surface(
    *,
    workspace: Path,
    formation_sessions: tuple[date, ...],
    expected_source_authority: PortfolioBenchmarkSourceAuthority | None = None,
) -> PortfolioBenchmarkSurface:
    """Build the same-clock SPY return path from already verified local data."""
    if formation_sessions != tuple(sorted(set(formation_sessions))):
        raise PortfolioBenchmarkBoundaryError("portfolio_benchmark.axis_invalid")
    market = MarketDataRepository(workspace.resolve())
    parent = market.current_quality_filtered_research_manifest(
        market_profile_id="us-current-index-research"
    )
    if parent is None:
        raise PortfolioBenchmarkBoundaryError("portfolio_benchmark.manifest_missing")
    reference = MarketReference.spy(parent)
    # The manifest is Market Data's; the projected frame is the Feature state's.
    # This read the second from the first, which no caller had exercised, so the
    # attribute error had been sitting in a live import path unfired.
    store = FeatureStateRepository(workspace.resolve(), market_data=market)
    all_rows, raw_hash, action_hash = store.projected_feature_frame(
        reference.manifest,
        listing_id=reference.listing_id,
        through=max(formation_sessions) + date.resolution * 10,
        start=min(formation_sessions),
    )
    ordered_sessions = tuple(cast(date, row["session_date"]) for row in all_rows)
    position = {value: index for index, value in enumerate(ordered_sessions)}
    simple_returns: list[float] = []
    for formation in formation_sessions:
        index = position.get(formation)
        if index is None or index + 2 >= len(all_rows):
            raise PortfolioBenchmarkBoundaryError("portfolio_benchmark.axis_unavailable")
        entry = all_rows[index + 1]
        exit_row = all_rows[index + 2]
        entry_open = float(entry["open_split_adjusted"])
        exit_open = float(exit_row["open_split_adjusted"])
        dividend = float(exit_row["cash_dividend"])
        if not all(np.isfinite((entry_open, exit_open, dividend))) or entry_open <= 0.0:
            raise PortfolioBenchmarkBoundaryError("portfolio_benchmark.price_invalid")
        simple = (exit_open + dividend) / entry_open - 1.0
        if not np.isfinite(simple) or simple <= -1.0:
            raise PortfolioBenchmarkBoundaryError("portfolio_benchmark.return_invalid")
        simple_returns.append(float(simple))
    values = {
        "formation_sessions": formation_sessions,
        "simple_returns": tuple(simple_returns),
        "log_returns": tuple(log(1.0 + value) for value in simple_returns),
        "universe_manifest_revision": parent.revision_sha256,
        "raw_input_hash": raw_hash,
        "action_set_hash": action_hash,
    }
    actual_authority = PortfolioBenchmarkSourceAuthority.create(
        formation_sessions=formation_sessions,
        universe_manifest_revision=parent.revision_sha256,
        raw_input_hash=raw_hash,
        action_set_hash=action_hash,
    )
    if expected_source_authority is not None and actual_authority != expected_source_authority:
        raise PortfolioBenchmarkBoundaryError("portfolio_benchmark.source_authority_not_admitted")
    return PortfolioBenchmarkSurface(
        **values,
        surface_hash=canonical_hash(
            PortfolioBenchmarkSurface.model_construct(
                **values,
                surface_hash="0" * 64,
            ).model_dump(mode="json", exclude={"surface_hash"})
        ),
    )


class PortfolioBenchmarkStore:
    """Content-addressed storage for Portfolio benchmark input surfaces."""

    def __init__(self, artifact_root: Path) -> None:
        """Choose the retained benchmark artifact root for rollback and upgrade.

        Args:
            artifact_root: Workspace artifact directory resolved before choosing the benchmark
                subtree.
        """
        self.root = artifact_root.resolve() / "portfolio-benchmark"

    def publish(self, surface: PortfolioBenchmarkSurface) -> None:
        """Admit the sealed benchmark's dated series through the Host's catalog.

        Args:
            surface: Validated benchmark surface whose identity names its artifact.

        Raises:
            DataError: The Host cannot admit the series or its stored values differ.
        """
        history.require_portfolio_result_store()
        metadata, table = _benchmark_body(surface)
        history.write_portfolio_result(
            "portfolio_benchmark", surface.surface_hash, metadata, table, result_table="series"
        )

    def load(self, surface_hash: str) -> PortfolioBenchmarkSurface:
        """Reopen and validate the benchmark surface named by an exact identity.

        Args:
            surface_hash: Requested content-addressed benchmark surface identity.

        Returns:
            The typed surface whose stored and requested identities agree.

        Raises:
            PortfolioBenchmarkBoundaryError: The artifact is absent, unreadable, invalid or names
                different lineage.
        """
        try:
            return _catalog_benchmark(surface_hash)
        except Exception as error:
            raise PortfolioBenchmarkBoundaryError("portfolio_benchmark.readback_failed") from error


def _benchmark_body(surface: PortfolioBenchmarkSurface) -> tuple[dict[str, Any], Any]:
    import pyarrow as pa  # lazy: only catalog result admission needs the data extra

    metadata = surface.model_dump(
        mode="json", exclude={"formation_sessions", "simple_returns", "log_returns"}
    )
    dates = surface.formation_sessions
    table = pa.table(
        {
            "name": ["log_returns"] * len(dates) + ["simple_returns"] * len(dates),
            "session_date": list(dates) * len(_BENCHMARK_SERIES),
            "value": list(surface.log_returns) + list(surface.simple_returns),
        }
    )
    return metadata, table


def _catalog_benchmark(
    digest: str, data: tuple[dict[str, Any], Any] | None = None
) -> PortfolioBenchmarkSurface:
    import pyarrow.compute as pc  # lazy: only catalog result reads need the data extra

    metadata, table = data or history.read_portfolio_result(
        "portfolio_benchmark", digest, result_table="series"
    )
    simple = table.filter(pc.equal(table["name"], "simple_returns"))
    logarithmic = table.filter(pc.equal(table["name"], "log_returns"))
    if (
        not simple.num_rows
        or not simple["session_date"].equals(logarithmic["session_date"])
        or table.num_rows != simple.num_rows * len(_BENCHMARK_SERIES)
    ):
        raise ValueError("portfolio_benchmark.catalog_invalid")
    surface = cast(
        PortfolioBenchmarkSurface,
        PortfolioBenchmarkSurface.model_validate(
            metadata
            | {
                "formation_sessions": tuple(simple["session_date"].to_pylist()),
                "simple_returns": tuple(simple["value"].to_pylist()),
                "log_returns": tuple(logarithmic["value"].to_pylist()),
            }
        ),
    )
    if surface.surface_hash != digest:
        raise ValueError("portfolio_benchmark.lineage_invalid")
    return surface


def carried_benchmark_check(_workspace: Path, digest: str, profile: Any, table: Any) -> None:
    """Refuse a carried benchmark whose surface, rebuilt from its profile and rows as the catalog
    reader rebuilds it, is not the one `digest` names, or whose profile is not the one that
    surface's publication writes."""
    surface = _catalog_benchmark(digest, (profile, table))
    if _benchmark_body(surface)[0] != profile:
        raise ValueError("portfolio_benchmark.lineage_invalid")


def portfolio_benchmark_upgrade_family(holder: Callable[[Path], str | None]) -> Family:
    """Move retained benchmark surfaces after the runner verifies its workspace backups."""
    from alphalattice.foundation.data_layer.upgrade import (  # lazy: first-start migration
        Family,
        workspace_files,
    )

    def legacy(runner: UpgradeRunner) -> Iterator[tuple[str | None, PortfolioBenchmarkSurface]]:
        for workspace in runner.workspaces:
            workspace = workspace.resolve()
            held = holder(workspace)
            for path in workspace_files(workspace):
                if (
                    path.suffix != ".json"
                    or path.parent.name != "surfaces"
                    or path.parent.parent.name != "portfolio-benchmark"
                ):
                    continue
                if not path.resolve().is_relative_to(workspace):
                    raise ValueError("portfolio_benchmark.upgrade_path_invalid")
                surface = PortfolioBenchmarkSurface.model_validate_json(path.read_bytes())
                if surface.surface_hash != path.stem:
                    raise ValueError("portfolio_benchmark.lineage_invalid")
                yield held, surface

    def step(runner: UpgradeRunner) -> None:
        for workspace, surface in legacy(runner):
            metadata, table = _benchmark_body(surface)
            artifact = history.write_portfolio_result(
                "portfolio_benchmark",
                surface.surface_hash,
                metadata,
                table,
                result_table="series",
                upgrade_owner=(runner.log, runner.root),
                upgrade_workspace=workspace,
            )
            if runner.map_id("portfolio-benchmark", surface.surface_hash, artifact) != artifact:
                raise ValueError("portfolio_benchmark.lineage_invalid")

    def verify(runner: UpgradeRunner) -> bool:
        return all(
            _catalog_benchmark(old.surface_hash) == old
            and history.portfolio_result_held("portfolio_benchmark", old.surface_hash, workspace)
            for workspace, old in legacy(runner)
        )

    return Family("portfolio-benchmark", step, verify)


__all__ = [
    "PortfolioBenchmarkSourceAuthority",
    "PortfolioBenchmarkSourceBinding",
    "PortfolioBenchmarkStore",
    "build_portfolio_benchmark_surface",
    "resolve_portfolio_benchmark_source_authority",
    "resolve_portfolio_benchmark_source_binding",
]
