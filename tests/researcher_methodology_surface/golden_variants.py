"""The golden workspaces a test session reads, declared once with the readers that ask for each.

A session warms every variant its collected tests read before the first test's timer starts
(tests/conftest.py, `pytest_collection_finish`), so no golden builds inside a test's own bound. A
reader parameter that no variant declares is refused at collection, and a cached build of a
variant this table does not declare is refused at its build, both by name
(`golden.variant_undeclared`): a new golden is declared here, with its readers, or it does not
build.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from alphalattice.foundation.market_data_ops.sources.universe import ResearchWhitelistStandard
from tests._support import development_universe, sufficient_universe

SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ")
"""Two sectors of five, the smallest shape that satisfies the five-member sector floor on both sides
of the sector-balanced weighting: the default Risk workspace's universe."""
DEVELOPMENT_SYMBOLS = tuple(item.symbol for item in development_universe.LISTINGS)
"""The development universe's listings (testing.md, the optimal development data set): passed as
`symbols`, the build serves that universe's provider, source and two-year calendar instead of the
seeded walk, and the golden key, which binds `symbols`, keeps its workspace apart."""
SUFFICIENT_SYMBOLS = sufficient_universe.SYMBOLS
FACTOR_SCALE = sufficient_universe.STANDARD
"""The walk a Factor or Alpha study needs: seven calendar years of Factor observations (756
training, three 252-session folds, a 252-session holdout) after the catalog's 252-session lookback,
so eight years of history; at seven the walk cannot fill its folds. The sufficient set's standard,
never the full ten years; its tests are large, nightly only."""
F120_SYMBOLS = tuple(f"F{index:03d}" for index in range(120))
READINESS_UNIVERSE = (F120_SYMBOLS, 6)
"""The workspace-readiness tests' listings and sector size, a golden of its own."""
READINESS_STANDARD = FACTOR_SCALE
"""Its history: its fixture binds Factor inputs and runs a Factor study on them."""
QUARANTINE_LISTINGS = (
    development_universe.LISTINGS[0],
    *sufficient_universe.LISTINGS[len(development_universe.LISTINGS) :][:59],
)
"""59/60 clears Panel coverage. One audit fits the shipped 2% bound and two exceed it."""
QUARANTINE_SYMBOLS = tuple(item.symbol for item in QUARANTINE_LISTINGS)
MAINTENANCE_SYMBOLS = tuple(item.symbol for item in development_universe.MAINTENANCE_LISTINGS)
TEN_SECTOR_SYMBOLS = tuple(f"QD{index:03d}" for index in range(60))
"""The UI QA issue scene's sixty listings in ten sectors of six."""


@dataclass(frozen=True)
class GoldenVariant:
    """One golden: the universe, sector size and standard its build is keyed by, whether it is
    built with the outside method's catalog and kernels, and its readers. A reader is a shared
    fixture's name, a `fixture[param]` of a parametrized one, or `<test file>::<name>`, a test
    function of that file or a fixture defined in it (a module's own override of a shared name)."""

    symbols: tuple[str, ...]
    readers: tuple[str, ...]
    sector_size: int = 5
    standard: ResearchWhitelistStandard = development_universe.STANDARD
    external: bool = False

    def keyed_by(
        self, symbols: Sequence[str], sector_size: int, standard: ResearchWhitelistStandard
    ) -> bool:
        """Whether a build of these inputs is this variant's, before its catalog and kernels."""
        return (self.symbols, self.sector_size, asdict(self.standard)) == (
            tuple(symbols),
            sector_size,
            asdict(standard),
        )


_FILE = "tests/integration/"
GOLDEN_VARIANTS: dict[str, GoldenVariant] = {
    "default": GoldenVariant(SYMBOLS, ()),
    "external": GoldenVariant(SYMBOLS, ("external_factor_workspace",), external=True),
    "development": GoldenVariant(
        DEVELOPMENT_SYMBOLS,
        (
            "development_risk_workspace",
            "input_seed[development]",
            _FILE + "feature_engine/test_real_workspace.py::"
            "test_a_second_consumer_restores_the_golden_workspace_with_the_same_identities",
            _FILE + "feature_engine/test_real_workspace.py::"
            "test_a_golden_that_disagrees_with_its_sidecar_is_refused_not_repaired",
        ),
    ),
    "factor_scale": GoldenVariant(
        SUFFICIENT_SYMBOLS,
        (
            "input_seed",
            _FILE + "alpha_research/test_alpha_development_execution.py::real_risk_workspace",
            _FILE + "factor_research/test_factor_development_execution.py::real_risk_workspace",
            _FILE + "sector_research/test_sector_research_development.py::campaign_workspace",
        ),
        standard=FACTOR_SCALE,
    ),
    "external_factor_scale": GoldenVariant(
        SUFFICIENT_SYMBOLS,
        (_FILE + "factor_research/test_external_factor_platform_acceptance.py::external_outcomes",),
        standard=FACTOR_SCALE,
        external=True,
    ),
    "f120": GoldenVariant(F120_SYMBOLS, ("input_seed[f120]",), standard=FACTOR_SCALE),
    "readiness": GoldenVariant(
        READINESS_UNIVERSE[0],
        ("prepared", "revision_seed"),
        sector_size=READINESS_UNIVERSE[1],
        standard=READINESS_STANDARD,
    ),
    "f120_development": GoldenVariant(
        F120_SYMBOLS,
        (
            _FILE + "product_host/test_workspace_preparation.py::"
            "test_explicit_qualified_data_binding_creates_its_manifest_without_manual_json",
        ),
    ),
    "five": GoldenVariant(
        SYMBOLS[:5], (_FILE + "product_host/test_daily_input_history.py::split_history",)
    ),
    "membership": GoldenVariant(
        SYMBOLS[:7],
        (_FILE + "product_host/test_membership_change_recovery.py::stopped_change_seed",),
        sector_size=7,
    ),
    "maintenance": GoldenVariant(
        MAINTENANCE_SYMBOLS, (_FILE + "feature_engine/test_local_data_update.py::entrant_seed",)
    ),
    "quarantine": GoldenVariant(
        QUARANTINE_SYMBOLS,
        (_FILE + "feature_engine/test_local_data_update_runs.py::quarantine_scale_seed",),
    ),
    # Read outside pytest: the UI QA issue scene builds it.
    "ui_qa_issue": GoldenVariant(TEN_SECTOR_SYMBOLS, (), sector_size=6, standard=FACTOR_SCALE),
}


class GoldenVariantUndeclared(LookupError):
    """A golden this table does not declare was asked for (`golden.variant_undeclared`)."""


def undeclared_build(symbols: Sequence[str], sector_size: int, why: str) -> GoldenVariantUndeclared:
    """The refusal of a cached build that no declared variant warms, by name."""
    return GoldenVariantUndeclared(
        f"golden.variant_undeclared: {len(symbols)} symbols from {tuple(symbols)[:1]}, sector size "
        f"{sector_size}: {why}; declare it in golden_variants.GOLDEN_VARIANTS with the readers "
        "that ask for it"
    )


def variants_read(items: Iterable[Any]) -> tuple[str, ...]:
    """Each declared variant the collected items read, in table order, and the default for any
    item that reads none. A parametrized reader's parameter that no variant declares is refused
    by name, before any test runs."""
    readers = {
        reader: name for name, variant in GOLDEN_VARIANTS.items() for reader in variant.readers
    }
    parametrized = {reader.partition("[")[0] for reader in readers if "[" in reader}
    found: set[str] = set()
    for item in items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        path = Path(str(item.path)).as_posix()
        names = {*item.fixturenames, getattr(item, "originalname", item.nodeid)}
        asked = set()
        for fixture in item.fixturenames:
            if fixture in parametrized and fixture in params:
                key = f"{fixture}[{params[fixture]}]"
                if key not in readers:
                    raise GoldenVariantUndeclared(
                        f"golden.variant_undeclared: {item.nodeid} asks {key}; declare it in "
                        "golden_variants.GOLDEN_VARIANTS with its variant"
                    )
                asked.add(readers[key])
            elif fixture in readers:
                asked.add(readers[fixture])
        for reader, name in readers.items():
            file, _, local = reader.partition("::")
            if local in names and path.endswith("/" + file):
                asked.add(name)
        found |= asked or {"default"}
    return tuple(name for name in GOLDEN_VARIANTS if name in found)
