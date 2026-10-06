"""Publish one simple signed score, or verify one already published.

    materialize  read the Panel, standardize, publish the binding
    verify       re-derive a published binding and report the disposition

The axis is an operator input, not a document: which sessions and which listings
a score covers is a property of the run that consumed it, and a committed file
stating them would go stale the first time the covariance moved. The *rule* --
which feature, which standardization, what happens to an unresolved cell -- is
installed source and is hashed into the binding.

One binding per evidence root. ``UpstreamHandleRegistry`` resolves the ``SIGNAL``
handle by requiring exactly one, so two in a root is an ambiguity nobody can
resolve for the author.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

# Only when run as a script, before the NumPy-bearing imports below: a test importing this
# module keeps its own threads and network (W11).
if __name__ == "__main__":
    os.environ.setdefault("ALPHALATTICE_NETWORK_DISABLED", "1")
    for _variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(_variable, "1")

PLAYPEN_ROOT = Path(__file__).resolve().parents[1]
if str(PLAYPEN_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PLAYPEN_ROOT / "src"))

from alphalattice.control.observation_runtime.telemetry.process_metrics import (  # noqa: E402
    peak_rss_bytes,
)
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver  # noqa: E402
from alphalattice.investment.alpha_research.experiments.development_artifacts import (  # noqa: E402
    AlphaDevelopmentArtifactStore,
)
from alphalattice.investment.alpha_research.simple_signal.authority import (  # noqa: E402
    implementation_closure_hash,
    resolve_simple_signed_score,
)
from alphalattice.investment.alpha_research.simple_signal.contracts import (  # noqa: E402
    SimpleSignedScoreBinding,
)
from alphalattice.investment.alpha_research.simple_signal.materialization import (  # noqa: E402
    materialize_simple_score,
)
from alphalattice.investment.alpha_research.simple_signal.provenance import (  # noqa: E402
    resolve_feature_clock,
)
from alphalattice.investment.alpha_research.simple_signal.standardize import (  # noqa: E402
    MINIMUM_FINITE_LISTINGS,
    STANDARDIZATION_ID,
)
from alphalattice.investment.risk_research.experiments.development_artifacts import (  # noqa: E402
    DEVELOPMENT_SURFACE_CATEGORY,
    RiskDevelopmentCovarianceSurface,
)
from alphalattice.investment.risk_research.surfaces.artifacts import (  # noqa: E402
    RiskArtifactStore,
)


def _emit(stage: str, started: float, **values: object) -> None:
    print(
        "SIMPLE_SIGNED_SCORE "
        + json.dumps(
            {"stage": stage, "elapsed_seconds": round(time.perf_counter() - started, 3), **values},
            sort_keys=True,
            default=str,
        )
    )


def _axis_from_covariance(evidence_root: Path) -> tuple[tuple[date, ...], tuple[str, ...]]:
    """Take the axis from the covariance surface the campaign will consume.

    Read rather than typed. The score has to cover exactly the formations and
    listings the campaign decides on, and a hand-entered range that drifted from
    the surface by one session would fail late, inside the axis gate, with a
    count instead of a cause.
    """

    store = RiskArtifactStore(evidence_root)
    directory = store.root / DEVELOPMENT_SURFACE_CATEGORY
    identities = (
        sorted(value.stem for value in directory.glob("*.json") if not value.name.startswith("."))
        if directory.is_dir()
        else []
    )
    if len(identities) != 1:
        raise SystemExit(
            f"expected one covariance surface under {evidence_root}, found {len(identities)}"
        )
    surface = RiskDevelopmentCovarianceSurface(
        **store.load_json(
            category=DEVELOPMENT_SURFACE_CATEGORY,
            uri=store.uri(DEVELOPMENT_SURFACE_CATEGORY, identities[0]),
            identity_field="surface_hash",
        )
    )
    return surface.formation_sessions, surface.ordered_listing_ids


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verb", choices=("materialize", "verify"))
    parser.add_argument("--panel-artifacts", type=Path, required=True)
    parser.add_argument("--panel-manifest-ref", default=None)
    parser.add_argument(
        "--axis-from-covariance",
        type=Path,
        default=None,
        help="Risk evidence root whose covariance surface defines the score axis.",
    )
    parser.add_argument("--output-workspace", type=Path, required=True)
    parser.add_argument("--feature-id", default="mom_252_21")
    parser.add_argument("--binding-hash", default="")
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    started = time.perf_counter()

    store = AlphaDevelopmentArtifactStore(arguments.output_workspace.resolve())

    if arguments.verb == "verify":
        if not arguments.binding_hash:
            raise SystemExit("--binding-hash is required for verify")
        resolved = resolve_simple_signed_score(
            binding=store.load_simple_signed_score(arguments.binding_hash),
            panel_artifacts_root=arguments.panel_artifacts.resolve(),
        )
        _emit(
            "verify_complete",
            started,
            binding_hash=arguments.binding_hash,
            disposition=resolved.disposition,
            rederived=resolved.rederived,
            peak_rss_bytes=peak_rss_bytes(),
        )
        return 0 if resolved.rederived else 5

    if arguments.panel_manifest_ref is None or arguments.axis_from_covariance is None:
        raise SystemExit("--panel-manifest-ref and --axis-from-covariance are required")
    sessions, listings = _axis_from_covariance(arguments.axis_from_covariance.resolve())
    panel_root = arguments.panel_artifacts.resolve()
    materialized = materialize_simple_score(
        resolver=ArtifactResolver(panel_root),
        panel_manifest_ref=arguments.panel_manifest_ref,
        feature_id=arguments.feature_id,
        ordered_formation_sessions=sessions,
        ordered_listing_ids=listings,
    )
    # The clock, read out of the Feature Desk's own shipped catalog. Not typed
    # here and not inferred from the numbers: a consumer that guessed a lag would
    # be asserting causality nobody checked, and re-deriving one from the values
    # would be reimplementing a Feature formula inside Alpha.
    clock = resolve_feature_clock(materialized.feature_id)
    binding = SimpleSignedScoreBinding.create(
        feature_id=materialized.feature_id,
        feature_formula_ref=clock.formula_ref,
        feature_formula=clock.formula,
        feature_window_sessions=clock.window_sessions,
        feature_source_interval=clock.source_interval,
        observation_session_offset_sessions=clock.observation_session_offset_sessions,
        availability_delay_sessions=clock.availability_delay_sessions,
        availability_policy_id=clock.availability_policy_id,
        availability_policy_hash=clock.availability_policy_hash,
        observation_clock_hash=clock.observation_clock_hash,
        feature_methodology_identity=clock.methodology_identity,
        feature_catalog_binding_hash=clock.catalog_binding_hash,
        panel_manifest_ref=arguments.panel_manifest_ref,
        panel_snapshot_identity=arguments.panel_manifest_ref.rsplit("/", maxsplit=1)[-1],
        ordered_formation_sessions=materialized.ordered_formation_sessions,
        ordered_listing_ids=materialized.ordered_listing_ids,
        standardization_id=STANDARDIZATION_ID,
        minimum_finite_listings=MINIMUM_FINITE_LISTINGS,
        values_identity=materialized.values_identity,
        implementation_closure_hash=implementation_closure_hash(),
        resolved_cell_count=materialized.resolved_cell_count,
        minimum_finite_listings_observed=materialized.minimum_finite_listings_observed,
    )
    store.publish_simple_signed_score(binding)
    _emit(
        "materialize_complete",
        started,
        binding_hash=binding.binding_hash,
        feature_id=binding.feature_id,
        formation_count=len(binding.ordered_formation_sessions),
        first_formation=binding.ordered_formation_sessions[0],
        last_formation=binding.ordered_formation_sessions[-1],
        listing_count=len(binding.ordered_listing_ids),
        resolved_cell_count=binding.resolved_cell_count,
        resolved_fraction=round(
            binding.resolved_cell_count
            / (len(binding.ordered_formation_sessions) * len(binding.ordered_listing_ids)),
            6,
        ),
        minimum_finite_listings_observed=binding.minimum_finite_listings_observed,
        # The three clocks, printed because they are what a reader must check
        # before believing any number the score later produces.
        observation_session_offset_sessions=binding.observation_session_offset_sessions,
        availability_delay_sessions=binding.availability_delay_sessions,
        availability_policy_id=binding.availability_policy_id,
        observation_clock_hash=binding.observation_clock_hash,
        feature_source_interval=binding.feature_source_interval,
        feature_formula=binding.feature_formula,
        feature_methodology_identity=binding.feature_methodology_identity,
        strategy_scope=binding.strategy_scope,
        peak_rss_bytes=peak_rss_bytes(),
        provider_calls=0,
        holdout_access_count=0,
        pointer_mutation_count=0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
