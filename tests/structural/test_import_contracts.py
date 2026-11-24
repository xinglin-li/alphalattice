"""Import rules from the package registry: allowed and refused examples."""

from __future__ import annotations

import subprocess
from pathlib import Path
from runpy import run_path
from typing import Protocol

import pytest

from alphalattice.capabilities.domain_contracts import spawn
from devtools.architecture import import_contracts
from devtools.architecture.structural import (
    PackageArchitectureRegistry,
    discover_module_dependencies,
    discover_package_dependencies,
    import_contract_violations,
    load_package_architecture_registry,
)


@pytest.mark.proves("architecture.import_law_refuses_higher_levels_and_retired_names")
@pytest.mark.parametrize(
    ("module", "body", "found"),
    [  # the import law's must-refuse and must-pass specimens, direct and relative
        ("low/layer.py", "from alphalattice.control.high import down\n", "lint-imports"),
        ("low/relative_layer.py", "from ...control.high import down\n", "lint-imports"),
        ("low/retired.py", "import alphalattice.kernel.low.gone.mod\n", "TID251"),
        ("low/relative_retired.py", "from .gone import thing\n", "TID251"),
        ("low/lookalike.py", "import alphalattice.kernel.low.gonester\n", None),
        # the old law read every tracked file and honored no suppression
        ("low/ignored.py", "import alphalattice.kernel.low.gone\n", "TID251"),
        ("low/noqa_line.py", "import alphalattice.kernel.low.gone  # noqa: TID251\n", "TID251"),
        ("low/noqa_file.py", "# ruff: noqa\nimport alphalattice.kernel.low.gone\n", "TID251"),
        # and every untracked file Git does not ignore, and every file under src on disk
        ("tests/untracked.py", "import alphalattice.kernel.low.gone\n", "TID251"),
        ("low/untracked_ignored.py", "import alphalattice.kernel.low.gone\n", "TID251"),
        ("high/down.py", "from alphalattice.kernel.low import keep\n", None),
        pytest.param(
            "low/direct_clock.py",
            "from datetime import datetime as stamp\nobserved = stamp.now()\n",
            "TID251",
            id="direct-wall-clock",
        ),
        pytest.param(
            "low/clock_port.py",
            "from alphalattice.kernel.clock import utc_now\nobserved = utc_now()\n",
            None,
            id="clock-port",
        ),
        pytest.param(
            "low/duration_clock.py",
            "from time import perf_counter\n"
            "started = perf_counter()\nelapsed = perf_counter() - started\n",
            None,
            id="duration-clock",
        ),
        pytest.param(
            "low/row_clock.py",
            "import time as system\nstamps = [system.time_ns() for row in range(3)]\n",
            "TID251",
            id="per-row-wall-clock",
        ),
        pytest.param(
            "low/anchored_clock.py",
            "from concurrent.futures import ThreadPoolExecutor\n"
            "from datetime import UTC, datetime\n"
            "from alphalattice.kernel.clock import ReplayClock, carry_observation\n"
            "from alphalattice.kernel.clock import observation_time, task_observation\n"
            "from alphalattice.kernel.resources import worker_width\n"
            "anchor = datetime(2026, 1, 5, tzinfo=UTC)\n"
            "ticks = iter((1.0, 1.0, 1.5, 2.75))\n"
            "clock = ReplayClock(anchor, ticks=lambda: next(ticks), wall=lambda: anchor)\n"
            "with task_observation(clock()):\n"
            "    read = carry_observation(lambda: tuple(observation_time().isoformat()\n"
            "                                           for row in range(3)))\n"
            "with ThreadPoolExecutor(max_workers=worker_width(1)) as pool:\n"
            "    stamps = pool.submit(read).result()\n"
            "later = clock().isoformat()\n",
            None,
            id="anchored-task-clock",
        ),
        (
            "high/hashlib.py",
            "from hashlib import sha256\nsha256(b'file')\n",
            "hashlib is owned by low",
        ),
        ("high/blake3.py", "import blake3\n", "blake3 is owned by low"),
        (
            "high/xxhash.py",
            "import importlib\nimportlib.import_module('xxhash')\n",
            "xxhash is owned by low",
        ),
        ("low/hash_backend.py", "from hashlib import sha256\nsha256(b'file')\n", None),
    ],
)
def test_the_import_law_refuses_a_higher_level_and_a_retired_name(
    tmp_path, module, body, found
) -> None:
    """A level-0 package importing level 1, or any file importing a retired owner, is refused by the
    contracts built from the registry. Downward imports and names that only look retired pass.
    The registered owner alone can import its digest backends."""
    tree = _tree(tmp_path)
    group = "kernel" if module.startswith("low/") else "control"
    target = (
        tree / module if module.startswith("tests/") else tree / "src/alphalattice" / group / module
    )
    if "untracked" not in module:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    spawn.run(["git", "add", "-A"], cwd=tree, check=True)
    if "untracked" in module:  # written after the index, so Git does not track it
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    if "ignored" in module:  # tracked or not, an ignored file is still the law's
        (tree / ".gitignore").write_text(target.name + "\n", encoding="utf-8")
    worker_specimens = {}
    if module == "low/retired.py":
        worker_specimens = {
            "cpu.py": ("from os import cpu_count as count\nworkers = count()\n", "TID251"),
            "keyword_pool.py": (
                "from concurrent.futures import ThreadPoolExecutor\n"
                "ThreadPoolExecutor(max_workers=2)\n",
                "resources.literal_max_workers",
            ),
            "positional_pool.py": (
                "from concurrent.futures import ProcessPoolExecutor as Pool\nPool(2)\n",
                "resources.literal_max_workers",
            ),
            "qualified_pool.py": (
                "import concurrent.futures as futures\nfutures.ThreadPoolExecutor(1)\n",
                "resources.literal_max_workers",
            ),
            "leased_pool.py": (
                "from concurrent.futures import ThreadPoolExecutor\n"
                "from alphalattice.kernel.low.keep import width\nThreadPoolExecutor(width)\n",
                None,
            ),
        }
        for name, (source, _) in worker_specimens.items():
            (target.parent / name).write_text(source, encoding="utf-8")
    registry = _registry()
    violations = import_contracts.check(tree, tmp_path / "evidence", registry)
    if target.is_relative_to(tree / "src"):
        qualified = ".".join(target.relative_to(tree / "src").with_suffix("").parts)
        violations += import_contract_violations({qualified: body}, registry=registry)
    for name, (_, expected) in worker_specimens.items():
        hits = [finding for finding in violations if f"/{name}:" in finding.replace(chr(92), "/")]
        if expected:
            assert len(hits) == 1 and expected in hits[0], (name, hits)
        else:
            assert hits == [], (name, hits)
        violations = tuple(finding for finding in violations if finding not in hits)
    if found is None:
        assert violations == ()
        if module == "low/anchored_clock.py":
            observed = run_path(str(target))
            assert observed["stamps"] == ("2026-01-05T00:00:00.500000+00:00",) * 3
            assert observed["later"] == "2026-01-05T00:00:01.750000+00:00"
    else:
        assert violations and all(found in line for line in violations), violations
    assert (tmp_path / "evidence/importlinter.ini").is_file() and (
        tmp_path / "evidence/ruff-retired.toml"
    ).is_file()


@pytest.mark.proves("architecture.kernel_ports_enforce_import_boundaries")
def test_the_kernel_import_law_exposes_only_downward_ports(tmp_path) -> None:
    """Prove all kernel boundaries in one graph from the generated Import Linter contracts."""
    from alphalattice.kernel.clock import ClockPort
    from alphalattice.kernel.config import ConfigPort
    from alphalattice.kernel.door import DoorPort
    from alphalattice.kernel.egress import EgressPort
    from alphalattice.kernel.execution import ExecutionPort
    from alphalattice.kernel.identity import IdentityPort
    from alphalattice.kernel.layout import LayoutPort
    from alphalattice.kernel.ledger import LedgerPort
    from alphalattice.kernel.lifecycle import LifecyclePort
    from alphalattice.kernel.locks import LocksPort
    from alphalattice.kernel.resources import ResourcesPort
    from alphalattice.kernel.storage import StoragePort
    from alphalattice.kernel.telemetry import TelemetryPort
    from alphalattice.kernel.types import TypesPort

    ports = (
        ClockPort,
        ConfigPort,
        DoorPort,
        EgressPort,
        ExecutionPort,
        IdentityPort,
        LayoutPort,
        LedgerPort,
        LifecyclePort,
        LocksPort,
        ResourcesPort,
        StoragePort,
        TelemetryPort,
        TypesPort,
    )
    assert all(Protocol in port.__bases__ and port.__module__.endswith(".port") for port in ports)

    tree = _tree(tmp_path)
    src = tree / "src/alphalattice"
    levels = {
        "clock": -5,
        "layout": -4,
        "identity": -4,
        "ledger": -3,
        "storage": -2,
        "execution": -2,
        "egress": -2,
        "door": -1,
        "types": -5,
    }
    base = _registry()
    kernel = tuple(
        base.by_package["low"].model_copy(
            update={
                "package_id": name,
                "dependency_level": level,
                "physical_namespace_target": f"alphalattice.kernel.{name}",
            }
        )
        for name, level in levels.items()
    )
    registry = PackageArchitectureRegistry.model_validate(
        base.model_dump(by_alias=True)
        | {
            "packages": (*base.packages, *kernel),
            "owned_product_imports": dict.fromkeys(base.owned_product_imports, "identity"),
            "deferred_imports": {
                "alphalattice.kernel.identity._digests -> blake3": "Load BLAKE3 on first use.",
                "alphalattice.kernel.identity._digests -> xxhash": "Load xxHash on first use.",
                "alphalattice.kernel.identity.eager -> blake3": "Load BLAKE3 on first use.",
                "alphalattice.kernel.egress.lazy -> httpx": "fixture backend",
            },
            "owned_imports": {
                "alphalattice.kernel.egress": ("httpx", "requests", "urllib.request"),
                "alphalattice.kernel.low": ("os.getenv",),
            },
        }
    )
    for row in kernel:
        directory = src / "kernel" / row.package_id
        directory.mkdir()
        (directory / "__init__.py").write_text("", encoding="utf-8")
    (src / "kernel/storage/impl.py").write_text("", encoding="utf-8")

    lazy = "def digest():\n    from blake3 import blake3\n    import xxhash\n"
    cases = (
        ("kernel/clock/up.py", "from ..layout import LayoutPort", "kernel.layout"),
        ("kernel/storage/peer.py", "from ..execution import ExecutionPort", "kernel.execution"),
        ("kernel/execution/peer.py", "from ..storage import StoragePort", "kernel.storage"),
        ("kernel/storage/ext.py", "import duckdb", "duckdb"),
        (
            "control/high/impl.py",
            "from alphalattice.kernel.storage import impl",
            "kernel.storage.impl",
        ),
        ("control/high/port.py", "from alphalattice.kernel.storage import StoragePort", None),
        ("kernel/door/down.py", "from ..ledger import LedgerPort", None),
        ("kernel/storage/own.py", "from datetime import datetime\nfrom . import impl", None),
        ("kernel/identity/_digests.py", lazy, None),
        ("kernel/identity/eager.py", "import blake3\ndef digest():\n    import blake3", "blake3"),
        ("kernel/identity/foreign.py", "def digest():\n    import xxhash", "xxhash"),
        ("kernel/low/stdlib.py", "from types import MappingProxyType", None),
        ("control/high/types.py", "from alphalattice.kernel import types", None),
        ("kernel/low/environment.py", "import os\nvalue = os.getenv('OMP_NUM_THREADS')", None),
        ("control/high/environment.py", "import os\nvalue = os.getenv('OMP_NUM_THREADS')", None),
        ("control/high/config_read.py", "from alphalattice.kernel.low import environment", None),
    )
    expected = set()
    for module, body, banned in cases:
        (src / module).write_text(body + "\n", encoding="utf-8")
        if banned:
            destination = f"alphalattice.{banned}" if banned.startswith("kernel.") else banned
            expected.add(f"alphalattice.{module[:-3].replace('/', '.')} -> {destination}")
    outbound = (
        (
            "external_httpx",
            "import httpx\nfrom time import time\nclient = httpx.Client()\nstamp = time()",
            ("httpx", "time.time"),
        ),
        ("external_requests", "from requests import get", ("requests",)),
        ("external_urllib", "from urllib.request import urlopen", ("urllib.request",)),
        ("local_host", "from http.client import HTTPConnection", ()),
        ("egress_port", "from alphalattice.kernel.egress import EgressPort", ()),
    )
    for module, body, _ in outbound:
        (src / "control/high" / f"{module}.py").write_text(body + "\n", encoding="utf-8")
    found = import_contracts.check(tree, tmp_path / "evidence", registry)
    owned = [
        line.replace("\\", "/")
        for line in found
        if line.startswith("ruff:") and ("environment.py" in line or "config_read.py" in line)
    ]
    assert len(owned) == 1 and all(
        value in owned[0] for value in ("control/high/environment.py", "TID251", "os.getenv")
    ), owned
    actual = {
        line.removeprefix("lint-imports: - ").strip().split(" (l.", 1)[0]
        for line in found
        if not line.startswith("ruff:")
    }
    assert actual == expected, {"missing": expected - actual, "unexpected": actual - expected}
    assert (
        import_contract_violations(
            {"alphalattice.kernel.identity._digests": lazy}, registry=registry
        )
        == ()
    )
    for module, _, refusals in outbound:
        findings = [line for line in found if f"{module}.py" in line or f".{module} ->" in line]
        if not refusals:
            assert not findings, findings
        else:
            for refusal in refusals:
                assert any(refusal in line for line in findings), (module, found)
    lazy = (
        "from importlib import import_module\ndef read():\n"
        '    return import_module("httpx")  # lazy: the source is read on demand.\n'
    )
    assert import_contract_violations(
        {
            "alphalattice.__init__": lazy,
            "alphalattice.control.high.lazy": lazy,
            "alphalattice.kernel.egress.lazy": lazy,
        },
        registry=registry,
    ) == (
        "alphalattice.control.high.lazy:3: owned-import httpx; use alphalattice.kernel.egress",
        "alphalattice:3: owned-import httpx; use alphalattice.kernel.egress",
    )
    definition = (
        "from importlib import import_module\n"
        '@import_module("blake3")\n'
        'def digest(backend=import_module("xxhash")): pass\n'
    )
    assert import_contract_violations(
        {"alphalattice.kernel.identity._digests": definition}, registry=registry
    ) == (
        "alphalattice.kernel.identity._digests:2: undeclared-kernel-backend blake3",
        "alphalattice.kernel.identity._digests:3: undeclared-kernel-backend xxhash",
    )
    modules = tuple(sorted(src.rglob("*.py")))
    assert {
        edge
        for edge in discover_module_dependencies(modules, source_root=tree / "src")
        if edge[1] == "types"
    } == {("high.types", "types")}
    assert {
        edge
        for edge in discover_package_dependencies(modules, source_root=tree / "src")
        if edge[1] == "types"
    } == {("high", "types")}


@pytest.mark.proves("architecture.import_law_refuses_higher_levels_and_retired_names")
def test_the_import_law_names_a_partial_listing_and_launches_a_long_tree_in_bounded_batches(
    tmp_path, monkeypatch
) -> None:
    """A Git listing that fails is no population, even with output; a tree whose paths exceed one
    Windows command line is checked in several launches, the ban found in the last; a path too
    long to launch alone is named."""
    tree = _tree(tmp_path)
    names = [f"tests/{index:03d}_{'x' * 96}.py" for index in range(320)]  # about 33,000 characters
    (tree / "tests").mkdir()
    for name in names:
        (tree / name).write_text("", encoding="utf-8")
    (tree / names[-1]).write_text("import alphalattice.kernel.low.gone\n", encoding="utf-8")
    spawn.run(["git", "add", "-A"], cwd=tree, check=True)
    run = import_contracts.run
    scanned: list[str] = []

    def traced(argv, root, **env):
        if argv[1:4] == ["-m", "ruff", "check"]:
            scanned.extend(str(path) for path in argv if str(path).endswith(".py"))
        return run(argv, root, **env)

    monkeypatch.setattr(import_contracts, "run", traced)
    registry = _registry().model_copy(
        update={"owned_imports": {"alphalattice.kernel.egress": ("httpx",)}}
    )
    violations = import_contracts.check(tree, tmp_path / "long", registry)
    assert sorted(scanned) == sorted(
        path.relative_to(tree).as_posix() for path in tree.rglob("*.py")
    )
    assert (
        len(violations) == 1 and "TID251" in violations[0] and Path(names[-1]).name in violations[0]
    )
    launches = import_contracts.launch_batches(["ruff"], names)[0]
    assert len(launches) > 1 and sum(map(len, launches)) == len(names)
    # Supplementary names exceed one Windows command line while each component fits POSIX NAME_MAX.
    wide = _tree(tmp_path / "wide")
    names = [f"tests/{index:03d}_{chr(0x1D4B3) * 60}.py" for index in range(270)]
    (wide / "tests").mkdir()
    for name in names:
        (wide / name).write_text("", encoding="utf-8")
    (wide / names[-1]).write_text("import alphalattice.kernel.low.gone\n", encoding="utf-8")
    spawn.run(["git", "-c", "core.longpaths=true", "add", "-A"], cwd=wide, check=True)
    violations = import_contracts.check(wide, tmp_path / "wide-evidence", _registry())
    assert len(violations) == 1 and "TID251" in violations[0], violations
    wide_batches = import_contracts.launch_batches(["ruff"], names)[0]
    assert len(wide_batches) > 1
    for batch in wide_batches:
        assert len(subprocess.list2cmdline(["ruff", *batch]).encode("utf-16-le")) // 2 < 32_767
    monkeypatch.setattr(import_contracts, "COMMAND_LINE", 60)
    assert import_contracts.launch_batches(["ruff"], ["a.py", names[0]]) == ([["a.py"]], [names[0]])
    monkeypatch.setattr(import_contracts, "COMMAND_LINE", 32_000)

    def partial(argv, root, **env):
        if "ls-files" in map(str, argv):  # a listing that fails part way, its output cut short
            return subprocess.CompletedProcess(argv, 1, names[0] + "\0", "fatal: index unreadable")
        return run(argv, root, **env)

    monkeypatch.setattr(import_contracts, "run", partial)
    violations = import_contracts.check(tree, tmp_path / "partial", _registry())
    assert any(
        "checked files are unknown" in line and "index unreadable" in line for line in violations
    )


def _registry() -> PackageArchitectureRegistry:
    """Two fixture packages, low at level 0 and high at level 1, and one retired owner."""
    product = load_package_architecture_registry(
        Path(__file__).resolve().parents[2] / "config/package-architecture.json"
    )
    return PackageArchitectureRegistry.model_validate(
        {
            "worker_count_reads": product.worker_count_reads,
            "wall_clock_reads": product.wall_clock_reads,
            "schema": "alphalattice.playpen.package-architecture",
            "packages": [
                dict(
                    package_id=name,
                    product_area="fixture",
                    capability_owner="Fixture",
                    role="DESK",
                    dependency_level=level,
                    authority_kind="FIXTURE",
                    allowed_peer_dependencies=[],
                    artifact_writer=False,
                    physical_namespace_target=f"alphalattice.{group}.{name}",
                )
                for name, level, group in (("low", 0, "kernel"), ("high", 1, "control"))
            ],
            "owned_product_imports": dict.fromkeys(("hashlib", "blake3", "xxhash"), "low"),
            "retired_imports": {"low.gone": "retired by a fixture cutover"},
        }
    )


def _tree(tmp_path):
    """A Git checkout holding the two fixture packages, nothing indexed yet."""
    tree = tmp_path / "repo"
    for package in (
        "alphalattice",
        "alphalattice/kernel",
        "alphalattice/control",
        "alphalattice/kernel/low",
        "alphalattice/control/high",
        "devtools",
    ):
        (tree / "src" / package).mkdir(parents=True, exist_ok=True)
        (tree / "src" / package / "__init__.py").write_text("", encoding="utf-8")
    (tree / "src/alphalattice/kernel/low/keep.py").write_text("", encoding="utf-8")
    spawn.run(["git", "init", "-q"], cwd=tree, check=True)
    return tree
