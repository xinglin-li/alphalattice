"""Thin local command adapter for the Product Host Portfolio application.

Thin means thin. This file parses flags and prints JSON; it validates nothing,
defaults nothing and refuses nothing on its own authority. Every bound comes from
``PortfolioResearchSpec`` and the installed control catalog, which is why a
researcher cannot reach a configuration here that the Desktop would refuse.

It replaces the thirty-argument development CLI as the ordinary entry point. The
advanced adapter it exposes takes a workspace id and a spec -- never an artifact
path, a pointer, or an identity used as authority.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import cast

PLAYPEN_ROOT = Path(__file__).resolve().parents[1]
PLAYPEN_SRC = PLAYPEN_ROOT / "src"
if str(PLAYPEN_SRC) not in sys.path:
    sys.path.insert(0, str(PLAYPEN_SRC))

from alphalattice.control.product_host.composition.application_session import (  # noqa: E402
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.portfolio_application import (  # noqa: E402
    PLAN_FIELDS,
    PortfolioResearchApplication,
)
from alphalattice.control.product_host.composition.research_workspace import (  # noqa: E402
    ResearchWorkspaceManifest,
    admit_research_workspace,
    manifest_fields_hash,
)
from alphalattice.interface.local_application.portfolio_research import (  # noqa: E402
    LocalPortfolioResearchService,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (  # noqa: E402
    REQUEST_WEIGHT_RULES,
    WORKSPACE_DEFAULT_STRATEGY_PACKAGE_ID,
    PortfolioResearchSpec,
    ScoreSourceMode,
)
from alphalattice.investment.portfolio_strategy_lab.application.controls import (  # noqa: E402
    ADMITTED_BENCHMARK_VIEWS,
    ADMITTED_REPORT_UNITS,
    DEFAULT_BENCHMARK_VIEW,
    DEFAULT_COST_BPS_PER_SIDE,
    DEFAULT_REPORT_UNIT,
    INSTALLED_PUBLIC_CONTROL_CATALOG,
)
from alphalattice.investment.portfolio_strategy_lab.application.resolution import (  # noqa: E402
    SharedPortfolioInputResolver,
    StrategyPortfolioResolver,
)
from alphalattice.investment.portfolio_strategy_lab.policies.tranche_book import (  # noqa: E402
    DEFAULT_TOP_K,
    DEFAULT_TRANCHES,
    DEFAULT_WEIGHT_RULE,
)


def _configured_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"portfolio_application.configuration_absent:{name}")
    return Path(value).resolve()


def _add_controls(parser: argparse.ArgumentParser) -> None:
    """One control, one flag, and every bound owned elsewhere.

    Ranges are deliberately not declared to argparse. If they were, this file
    would become a second validator that could be laxer or stricter than the
    spec, which is the exact failure the generated contract exists to prevent.
    `choices` is used only where the value set is the enum itself.
    """

    parser.add_argument("--strategy-package-id", default=None)
    parser.add_argument(
        "--score-source-mode",
        choices=("HISTORICAL_ARRAY_REPLAY", "CURRENT_MODEL_SCORING"),
        default=None,
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--tranches", type=int, default=DEFAULT_TRANCHES)
    parser.add_argument("--exit-rank", type=int, default=None)
    parser.add_argument("--weight-rule", default=DEFAULT_WEIGHT_RULE, choices=REQUEST_WEIGHT_RULES)
    parser.add_argument("--cost-bps-per-side", default=DEFAULT_COST_BPS_PER_SIDE)
    parser.add_argument(
        "--secondary-benchmark-view",
        default=DEFAULT_BENCHMARK_VIEW,
        choices=ADMITTED_BENCHMARK_VIEWS,
    )
    parser.add_argument("--report-unit", default=DEFAULT_REPORT_UNIT, choices=ADMITTED_REPORT_UNITS)
    parser.add_argument("--study-start", type=date.fromisoformat, default=None)
    parser.add_argument("--study-end", type=date.fromisoformat, default=None)
    parser.add_argument(
        "--spec-hash",
        help="Optional identity check from an exported command; never an input.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alphalattice-portfolio")
    parser.add_argument("--workspace", required=True, help="Saved local workspace id")
    subcommands = parser.add_subparsers(dest="operation", required=True)
    for operation in ("plan", "run"):
        _add_controls(subcommands.add_parser(operation))
    for operation in ("report", "export"):
        child = subcommands.add_parser(operation)
        child.add_argument("--result-hash", required=True)
    subcommands.add_parser("controls")
    return parser


def spec_from_args(
    args: argparse.Namespace, *, manifest: ResearchWorkspaceManifest | None = None
) -> PortfolioResearchSpec:
    """Build the request. Every refusal below this line belongs to the spec."""

    strategy_package_id = getattr(args, "strategy_package_id", None) or (
        WORKSPACE_DEFAULT_STRATEGY_PACKAGE_ID
        if manifest is None
        else manifest.default_strategy_package_id
    )
    if strategy_package_id is None:
        raise ValueError("research_workspace.strategy_not_installed")
    score_source_mode = cast(
        ScoreSourceMode,
        getattr(args, "score_source_mode", None)
        or ("HISTORICAL_ARRAY_REPLAY" if manifest is None else manifest.default_score_source_mode),
    )
    spec = PortfolioResearchSpec.create(
        strategy_package_id=strategy_package_id,
        score_source_mode=score_source_mode,
        top_k=args.top_k,
        tranches=args.tranches,
        exit_rank=args.exit_rank,
        weight_rule=args.weight_rule,
        cost_bps_per_side=args.cost_bps_per_side,
        secondary_benchmark_view=args.secondary_benchmark_view,
        report_unit=args.report_unit,
        study_start=args.study_start,
        study_end=args.study_end,
    )
    supplied = getattr(args, "spec_hash", None)
    if supplied is not None and supplied != spec.spec_hash:
        # An exported command carries its identity so a replay can prove it
        # rebuilt the same request from the same flags. It is a check, never a
        # way to name a configuration the compiler never saw.
        raise ValueError("portfolio_application.exported_spec_identity_invalid")
    return spec


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.operation == "controls":
        print(json.dumps(INSTALLED_PUBLIC_CONTROL_CATALOG.model_dump(mode="json"), indent=2))
        return 0
    workspaces_root = _configured_path("ALPHALATTICE_WORKSPACES_ROOT")
    workspace = (workspaces_root / str(args.workspace)).resolve()
    if workspace.parent != workspaces_root or not workspace.is_dir():
        raise ValueError("portfolio_application.workspace_id_invalid")
    admitted_workspace = admit_research_workspace(workspace)
    manifest = admitted_workspace.manifest
    with WorkspaceApplicationSession.acquire(workspace) as session:
        # The CLI is a consumer of the Local Application Service, not a second
        # path beside it. Wiring the Host in here -- rather than inside the
        # service -- is what keeps Product Host a composition root that nothing
        # in `src/` imports.
        host = PortfolioResearchApplication(
            workspace_id=manifest.workspace_id,
            workspace=workspace,
            manifest_binding=lambda: manifest_fields_hash(manifest, PLAN_FIELDS),
            session=session,
            resolver=StrategyPortfolioResolver(
                shared=SharedPortfolioInputResolver(), catalog=admitted_workspace.require_catalog()
            ),
        )
        service = LocalPortfolioResearchService(application=host, lineage=host)
        if args.operation == "plan":
            output = service.plan(spec_from_args(args, manifest=manifest)).model_dump(mode="json")
        elif args.operation == "run":
            result = service.run(spec_from_args(args, manifest=manifest))
            output = {
                "result": result.model_dump(mode="json"),
                "readouts": service.readouts(result.result_hash).model_dump(mode="json"),
            }
        elif args.operation == "report":
            output = service.report(str(args.result_hash)).model_dump(mode="json")
        else:
            output = {"export_command": service.export(str(args.result_hash))}
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
