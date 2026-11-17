"""The data family census: every writer the product persists through is listed, and no more."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.architecture.structural import (
    authority_groups,
    data_family_violations,
    discover_persistence_sites,
    load_artifact_authority_registry,
    load_storage_authority_registry,
)

TEST_SCOPE = "whole-tree"
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.proves("architecture.every_persisted_family_is_listed")
def test_every_persisting_writer_and_authority_group_is_in_one_data_family() -> None:
    """Every writer the product persists through is listed by a data family, and every storage
    and artifact authority group is covered by exactly one."""

    sources = tuple(
        path
        for path in sorted((ROOT / "src" / "alphalattice").rglob("*.py"))
        if "__pycache__" not in path.parts
    )
    groups = authority_groups(
        load_storage_authority_registry(ROOT / "config" / "storage-authorities.json"),
        load_artifact_authority_registry(ROOT / "config" / "artifact-authorities.json"),
    )
    registry = json.loads(
        (ROOT / "config" / "registries" / "data-families.json").read_text(encoding="utf-8")
    )
    sites = discover_persistence_sites(sources, source_root=ROOT / "src")
    assert data_family_violations(sites, registry, groups=groups) == ()
    families = registry["families"]
    assert {
        name: families[name].get("persistence_owner")
        for name in ("protected-validation", "universe-manifests", "universe-membership-journal")
    } == {
        "protected-validation": "model_validation.protected_gate:ProtectedValidationStore",
        "universe-manifests": "market_data_ops.storage.duckdb:MarketDataRepository",
        "universe-membership-journal": "market_data_ops.storage.duckdb:MarketDataRepository",
    }
    for name in (
        "session-marks",
        "portfolio-research-pipeline",
        "pre-research-head",
        "research-programs",
        "research-experiments",
        "factor-development",
    ):
        assert families[name].get("home") and families[name].get("writers")
        assert families[name].get("store_step") and families[name].get("placement_reason")
    # 0.2.8 rows 8a and 8b: A24-A32, A40-A53, A56, A60, A69 and A77-A80.
    for name in (
        "portfolio-score-evidence",
        "portfolio-scores",
        "research-migrations",
        "pre-research-summaries",
        "sector-context",
        "execution-outcomes",
        "research-current-pointers",
        "factor-candidate-evidence",
        "session-marks",
        "protected-validation",
        "guardian-views",
        "portfolio-benchmark",
        "tradability",
        "portfolio-oos-consumption",
        "portfolio-inputs",
        "portfolio-signal-comparisons",
        "portfolio-advancement",
        "portfolio-finalization",
        "portfolio-public-ledger",
        "strategy-lab-evidence",
        "research-experiments",
        "portfolio-research-pipeline",
        "pre-research-head",
        "research-programs",
        "feature-trials",
        "factor-development",
    ):
        family = families[name]
        assert family.get("home"), name
        if family.get("writer_status") == "retired; no current writer":
            assert not family["writers"] and not family.get("persistence_owner"), name
            assert family.get("historical_writers") and family.get("writer_retirement_commits"), (
                name
            )
        else:
            assert family.get("writers") or family.get("persistence_owner"), name


@pytest.mark.proves("architecture.an_unlisted_persisting_writer_is_refused")
def test_a_planted_unlisted_writer_is_refused_and_a_listed_one_passes(tmp_path: Path) -> None:
    """A new store's writer is refused until a family lists it, whether it writes through a
    method, a mode chosen at run time or os.open's flags; a read and an in-memory database are
    not sites; a listed helper and a listed non-file open pass; a group is covered by one family."""

    module = tmp_path / "src" / "alphalattice" / "control" / "demo" / "store.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "import duckdb, os\n"
        "class DemoStore:\n"
        "    def put(self, path, data):\n"
        "        path.write_bytes(data)\n"
        "def read(path):\n"
        "    with path.open('rb') as stream:\n"
        "        return stream.read(), duckdb.connect(':memory:')\n"
        "def log(path):\n"
        "    with open(path, 'a') as stream:\n"
        "        stream.write('x')\n"
        "def chosen(path, mode):\n"
        "    return open(path, mode)\n"
        "def resume(path, again):\n"
        "    return path.open('ab' if again else 'wb')\n"
        "def reopen(path, mode):\n"
        "    return path.open(mode)\n"
        "def count(path):\n"
        "    os.open(path, os.O_RDONLY | os.O_BINARY)\n"
        "    return os.open(path, os.O_WRONLY | os.O_CREAT)\n"
        "def page(key):\n"
        "    return PAGES.open(key)\n",
        encoding="utf-8",
    )
    sites = discover_persistence_sites((module,), source_root=tmp_path / "src")
    assert sites == {
        "demo.store:DemoStore": ["write_bytes"],
        "demo.store:chosen": ["open"],
        "demo.store:count": ["os.open"],
        "demo.store:log": ["open"],
        "demo.store:page": ["open"],
        "demo.store:reopen": ["open"],
        "demo.store:resume": ["open"],
    }
    writers = [f"demo.store:{name}" for name in ("DemoStore", "chosen", "count", "reopen")]
    writers.append("demo.store:resume")
    family = {"writers": writers, "covers": ["table:demo"]}
    page = {"demo.store:page": "opens a page by its key"}
    unlisted = {"families": {"demo": family}, "not_persisting": page}
    (refused,) = data_family_violations(sites, unlisted, whole_tree=False)
    assert refused.startswith("demo.store:log persists (open) outside every data family")
    listed = {**unlisted, "helpers": {"demo.store:log": "a helper"}}
    assert data_family_violations(sites, listed, groups=["table:demo"]) == ()
    (uncovered,) = data_family_violations(sites, listed, groups=["table:demo", "table:other"])
    assert uncovered.startswith("authority group table:other is covered by 0 data families")
