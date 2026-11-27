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
    OwnerBypass,
    PackageArchitectureRegistry,
    _owner_sites,
    discover_module_dependencies,
    discover_package_dependencies,
    grown_bypasses,
    import_contract_violations,
    load_package_architecture_registry,
    owner_bypass_refusals,
    vacated_bypasses,
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
    identity = "alphalattice.kernel.identity"
    caller = "alphalattice.control.high.file_verification"
    assert import_contract_violations(
        {
            caller: (
                f"from {identity} import file_stamp as stamp\n"
                f"from {identity}.file_facts import handle_stamp\n"
                f"from {identity} import integrity_blake3_path\n"
                f"import {identity} as facts\n"
                "facts.file_stamp(path)\n"
                f"from {identity} import canonical_hash\n"
            ),
            f"{identity}.file_facts": f"from {identity} import file_stamp\n",
        },
        registry=registry,
    ) == (
        f"{caller}:1: file-verification-owner-required {identity}.file_stamp",
        f"{caller}:2: file-verification-owner-required {identity}.file_facts.handle_stamp",
        f"{caller}:3: file-verification-owner-required {identity}.integrity_blake3_path",
        f"{caller}:5: file-verification-owner-required {identity}.file_stamp",
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


_STORE = "alphalattice.control.high.store"
_RECORDS = "from alphalattice.kernel.identity import blake3_digest, record_sealed\n"
OWNER_BYPASS_CASES = {  # case: (module, its source, the refusals, the units the registry names)
    "a-new-sqlite-connect-is-refused": (
        _STORE,
        "import sqlite3\nsqlite3.connect(path)\n",
        [f"{_STORE}:2: owner-bypass connect <module>; open the database through its store's owner"],
        {},
    ),
    "a-duckdb-connect-under-another-name-is-refused": (
        _STORE,
        "from duckdb import connect as open_db\ndef read():\n    return open_db(path)\n",
        [f"{_STORE}:3: owner-bypass connect read; open the database through its store's owner"],
        {},
    ),
    "the-log-opens-its-own-database": (
        "alphalattice.foundation.data_layer.log",
        "import sqlite3\nsqlite3.connect(path)\n",
        [],
        {},
    ),
    "a-digest-tag-written-in-place-is-refused": (
        _STORE,
        'def tag(digest):\n    return "blake3:" + digest.hexdigest()\n',
        [
            f"{_STORE}:2: owner-bypass digest-tag tag; tag or match a digest through "
            "kernel.identity (tagged_blake3, is_blake3)"
        ],
        {},
    ),
    "a-docstring-naming-the-tag-is-none": (_STORE, '"""Digests read blake3:<hex>."""\n', [], {}),
    "a-second-canonical-json-is-refused": (
        _STORE,
        "import json\nfrom alphalattice.kernel.identity import identity_blake3\ndef key(v):\n"
        "    text = json.dumps(v, sort_keys=True)\n    return identity_blake3(text.encode())\n",
        [
            f"{_STORE}:5: owner-bypass canonical-json key; hash a value through "
            "kernel.identity's canonical_hash"
        ],
        {},
    ),
    "the-package-root-is-the-products": (
        "alphalattice.__init__",
        "import sqlite3\nsqlite3.connect(path)\n",
        [
            "alphalattice:2: owner-bypass connect <module>; open the database through its "
            "store's owner"
        ],
        {},
    ),
    "a-devtools-module-is-not-the-products": (
        "devtools.probe",
        "import sqlite3\nsqlite3.connect(path)\n",
        [],
        {},
    ),
    "a-canonical-json-under-an-alias-is-refused": (
        _STORE,
        "import json as j\nfrom alphalattice.kernel.identity import identity_blake3\ndef key(v):\n"
        "    return identity_blake3(j.dumps(v).encode())\n",
        [
            f"{_STORE}:4: owner-bypass canonical-json key; hash a value through "
            "kernel.identity's canonical_hash"
        ],
        {},
    ),
    "an-encoded-canonical-json-assigned-first-is-refused": (
        _STORE,
        "from json import dumps\nfrom alphalattice.kernel.identity import identity_blake3\n"
        "def key(v):\n    payload = dumps(v).encode()\n    return identity_blake3(payload)\n",
        [
            f"{_STORE}:5: owner-bypass canonical-json key; hash a value through "
            "kernel.identity's canonical_hash"
        ],
        {},
    ),
    "a-file-digest-compared-in-place-is-refused-once": (
        _STORE,
        _RECORDS + "def held(path, seal):\n    return blake3_digest(path.read_bytes()) == seal\n",
        [
            f"{_STORE}:3: owner-bypass file-verification held; verify the file through "
            "verified_facts.verified_file_digest"
        ],
        {},
    ),
    "a-file-verification-split-across-names-is-refused-once": (
        _STORE,
        _RECORDS + "def held(path, seal):\n    data = path.read_bytes()\n"
        "    actual = blake3_digest(data)\n    return actual == seal\n",
        [
            f"{_STORE}:4: owner-bypass file-verification held; verify the file through "
            "verified_facts.verified_file_digest"
        ],
        {},
    ),
    "a-digest-that-only-names-a-path-is-no-verification": (
        _STORE,
        _RECORDS + "def place(root, body, held):\n    name = blake3_digest(body)\n"
        "    target = root / name\n    return target == held\n",
        [],
        {},
    ),
    "an-mmap-read-is-refused": (
        _STORE,
        "def seal(digest, path):\n    return digest.update_mmap(path)\n",
        [
            f"{_STORE}:2: owner-bypass file-verification seal; verify the file through "
            "verified_facts.verified_file_digest"
        ],
        {},
    ),
    "a-placement-that-records-no-seal-is-refused": (
        _STORE,
        _RECORDS + "import os\ndef put(staged, path, body):\n"
        "    os.replace(staged, path)\n    return blake3_digest(body)\n",
        [
            f"{_STORE}:4: owner-bypass unrecorded-seal put; record the placed file's digest "
            "with record_sealed"
        ],
        {},
    ),
    "a-placement-that-records-its-seal-passes": (
        _STORE,
        _RECORDS + "import os\ndef put(staged, path, body):\n    os.replace(staged, path)\n"
        "    record_sealed(path, blake3_digest(body))\n",
        [],
        {},
    ),
    "one-seal-covers-only-its-own-placement": (
        _STORE,
        _RECORDS + "import os\ndef put(a, b, first, second, body):\n    os.replace(a, first)\n"
        "    os.replace(b, second)\n    record_sealed(first, blake3_digest(body))\n",
        [
            f"{_STORE}:5: owner-bypass unrecorded-seal put; record the placed file's digest "
            "with record_sealed"
        ],
        {},
    ),
    "a-path-replaced-after-its-record-needs-its-own-seal": (
        _STORE,
        _RECORDS + "import os\ndef put(a, b, path, body):\n    os.replace(a, path)\n"
        "    record_sealed(path, blake3_digest(body))\n    os.replace(b, path)\n",
        [
            f"{_STORE}:6: owner-bypass unrecorded-seal put; record the placed file's digest "
            "with record_sealed"
        ],
        {},
    ),
    "a-fact-not-written-records-no-seal": (
        _STORE,
        _RECORDS + "import os\nfrom alphalattice.kernel.identity import verified_file_digest\n"
        "def put(staged, path, body):\n    os.replace(staged, path)\n"
        "    verified_file_digest(path, written=False)\n    return blake3_digest(body)\n",
        [
            f"{_STORE}:5: owner-bypass unrecorded-seal put; record the placed file's digest "
            "with record_sealed"
        ],
        {},
    ),
    "a-lowercase-catalog-update-is-refused": (
        _STORE,
        'def keep(lake):\n    lake.execute("update lake.world.bars set close = 1")\n',
        [
            f"{_STORE}:2: owner-bypass catalog-write keep; write the catalog through "
            "StorePort's commit, publish or carry"
        ],
        {},
    ),
    "a-catalog-target-held-in-a-local-name-is-refused": (
        _STORE,
        "def keep(lake, table):\n    target = f'lake.derived.{table}'\n"
        "    lake.execute(f'INSERT INTO {target} BY NAME SELECT 1')\n",
        [
            f"{_STORE}:3: owner-bypass catalog-write keep; write the catalog through "
            "StorePort's commit, publish or carry"
        ],
        {},
    ),
    "a-catalog-write-outside-the-order-is-refused": (
        _STORE,
        'def keep(lake):\n    lake.execute("INSERT INTO lake.world.bars SELECT * FROM rows")\n',
        [
            f"{_STORE}:2: owner-bypass catalog-write keep; write the catalog through "
            "StorePort's commit, publish or carry"
        ],
        {},
    ),
    "the-ordered-writer-writes-the-catalog": (
        "alphalattice.foundation.data_layer.store",
        "class StorePort:\n    def _statements(self, lake):\n"
        '        lake.execute(f"DELETE FROM lake.world.{table} WHERE 1")\n',
        [],
        {},
    ),
    "a-named-unit-passes-and-its-surplus-is-stale": (
        _STORE,
        "import sqlite3\nsqlite3.connect(path)\n",
        ["stale-owner-bypass src/alphalattice/control/high/store.py::<module>::connect"],
        {"src/alphalattice/control/high/store.py::<module>::connect": 2},
    ),
}


@pytest.mark.proves("architecture.converged_owners_refuse_a_new_bypass")
@pytest.mark.parametrize(
    ("module", "source", "refusals", "named"),
    OWNER_BYPASS_CASES.values(),
    ids=list(OWNER_BYPASS_CASES),
)
def test_each_converged_owner_refuses_a_new_bypass_by_name(module, source, refusals, named):
    """A connection, digest tag, canonical JSON, file verification, placement or catalog write
    outside its owner is refused by kind, scope and repair; its owner's own form and a named unit
    pass, and a named unit with no site left is stale."""
    rows = {key: OwnerBypass(count=count, reason="fixture") for key, count in named.items()}
    found = owner_bypass_refusals({module: _owner_sites(source.encode("utf-8"))}, rows)
    assert sorted(found) == refusals


@pytest.mark.proves("architecture.converged_owners_refuse_a_new_bypass")
def test_a_named_bypass_only_shrinks() -> None:
    """A named unit above its parent's count, or one the parent did not name, is a new bypass;
    a lower count passes, and the first list is held when the parent names none."""
    unit = "src/alphalattice/control/high/store.py::<module>::connect"
    row = {"count": 1, "reason": "fixture"}
    grown = [f"owner-bypass: a named bypass grew {unit}"]
    assert grown_bypasses({unit: row}, None) == []
    assert grown_bypasses({unit: row}, {unit: 2}) == []
    assert grown_bypasses({unit: {**row, "count": 2}}, {unit: 1}) == grown
    assert grown_bypasses({unit: row}, {}) == grown


@pytest.mark.proves("architecture.converged_owners_refuse_a_new_bypass")
def test_an_owners_own_write_is_seen_before_it_is_credited() -> None:
    """The ordered writer's insert into a catalog table its local name holds is a catalog write
    the scan sees, and the owner's unit credits it; it passes because it is credited, not unseen."""
    source = (
        "class StorePort:\n    def _derived(self, lake, table):\n"
        "        target = f'lake.derived.{table}'\n"
        "        return lake.execute(f'INSERT INTO {target} BY NAME SELECT 1')\n"
    )
    sites = _owner_sites(source.encode("utf-8"))
    assert sites == [[4, "catalog-write", "StorePort._derived"]]
    assert owner_bypass_refusals({"alphalattice.foundation.data_layer.store": sites}, {}) == []


@pytest.mark.proves("architecture.converged_owners_refuse_a_new_bypass")
def test_a_deleted_files_named_unit_is_stale() -> None:
    """A named unit whose file the tree no longer holds is stale, so its count is never kept for
    a later file at that path; a unit whose file is present passes."""
    unit = "src/alphalattice/control/high/store.py::<module>::connect"
    assert vacated_bypasses([unit], {"src/alphalattice/control/high/store.py"}) == []
    assert vacated_bypasses([unit], set()) == [f"stale-owner-bypass {unit}"]
