"""Test-size fixture data and registry setup shared by the structural holder."""

import json
from pathlib import Path

import pytest

from tests.structural.pytest_support import AUDIT_HOOK_SAMPLE

# fmt: off
_CASES = {
    "legacy": ({}, 1),
    "date": ({"until": "2026-99-99"}, 0),
    "first": ({"renewals": 0}, 1),
    "renewed": ({"renewals": 1}, 1),
    "twice": ({"renewals": 2}, 0),
    "bool": ({"renewals": True}, 0),
    "string": ({"renewals": "1"}, 0),
    "no-why": ({"reason": ""}, 0),
    "multiline": ({"renewals": 1, "reason": "a\nb"}, 0),
    "platform": ({"class": "platform"}, 1),
    "class": ({"class": "flaky"}, 0),
    "late": ({"since": "2026-11-04"}, 0),
    "bad-day": ({"since": "2026-11-31"}, 0),
    "seen": ({"seen_on": "mac1"}, 0),
    "private": ({"reason": "mac1"}, 0),
}


def _registry_tree(tmp_path):
    """A checkout with one requirement, its holder, an unregistered twin and an unrelated test."""
    root, registries = Path(__file__).resolve().parents[2], tmp_path / "config/registries"
    registries.mkdir(parents=True)
    for name in ("test-sizes.json", "owners-map.json"):
        (registries / name).write_bytes((root / "config/registries" / name).read_bytes())
    registry = json.loads((root / "config/registries/requirements.json").read_bytes())
    row = {"owner": "alphalattice.control.product_host", "layer": "integration", "seconds": 20}
    registry["entries"] = {HELD: row | {"description": "A held update runs after its session."}}
    (registries / "requirements.json").write_text(json.dumps(registry), "utf-8")
    (tmp_path / "tests/a").mkdir(parents=True)
    (tmp_path / "tests/a/test_host.py").write_text(
        "import pytest\n"
        f'@pytest.mark.proves("{HELD}")\n'
        "def test_breakaway_launch_survives(): pass\n"
        'def test_idle_expiry_keeps_owed_work():\n    """A detached Host keeps owed work."""\n'
        "def test_parquet_schema_round_trip(): pass\n",
        "utf-8",
    )
    return registry


HELD = "product_host.detached_start_outlives_the_session"


LAYER_CASES = (
[
        ("fd-provenance", "unit", "", "", False),
        ("module-inventory", "unit", "", "import sys; sys.modules.get('duckdb')", True),
        ("root-pure", "unit", "", "", False),
        ("root-session-golden", "unit", "", "", False),
        ("root-session-golden-error", "unit", "", "", True),
        ("root-session-seed", "unit", "", "", False),
        ("root-session-seed-error", "unit", "", "", True),
        ("root-lookalike", "unit", "", "", True),
        ("root-runtime", "unit", "", "", True),
        ("root-cached", "unit", "", "", True),
        ("database", "unit", "", "import duckdb; duckdb.connect(':memory:').close()", True),
        (
            "bound-database",
            "unit",
            "from duckdb import connect\n",
            "connect(':memory:').close()",
            True,
        ),
        ("preloaded-pure", "unit", "import duckdb\n", "assert 2 + 2 == 4", False),
        (
            "installed-metadata",
            "unit",
            "",
            "import importlib.metadata as m; m.entry_points()",
            False,
        ),
        (
            "installed-timezone",
            "unit",
            "",
            "import zoneinfo as z; z.ZoneInfo.no_cache('UTC')",
            False,
        ),
        ("project-read", "unit", "", f"open({__file__!r}, 'rb').read()", True),
        (
            "integration",
            "integration",
            "",
            "import duckdb; duckdb.connect(':memory:').close()",
            False,
        ),
        ("legacy", "old_area", "", "import duckdb; duckdb.connect(':memory:').close()", False),
        (
            "process",
            "unit",
            "",
            "import subprocess,sys; subprocess.run([sys.executable,'-c','pass'],check=True)",
            True,
        ),
        ("network", "unit", "", "import socket; socket.socket().close()", True),
        (
            "network-caught",
            "unit",
            "",
            "",
            False,
        ),
        pytest.param(
            "size-live-over",
            "unit",
            "import pytest\npytestmark = pytest.mark.small\n",
            "(tmp_path / 'payload').write_bytes(b'x' * 10_000_001)",
            True,
            id="size-live-over",
        ),
        pytest.param(
            "size-live-factory",
            "unit",
            "import pytest\npytestmark = pytest.mark.small\n",
            "(tmp_path_factory.mktemp('large') / 'payload').write_bytes(b'x' * 10_000_001)",
            True,
            id="size-live-factory",
        ),
        pytest.param(
            "size-live-factory-base",
            "unit",
            "import pytest\npytestmark = pytest.mark.small\n",
            "(tmp_path_factory.getbasetemp() / 'payload').write_bytes(b'x' * 10_000_001)",
            True,
            id="size-live-factory-base",
        ),
        pytest.param(
            "size-live-factory-mixed",
            "unit",
            "import pytest\npytestmark = pytest.mark.small\n",
            "(tmp_path / 'first').write_bytes(b'x' * 5_000_000); "
            "(tmp_path_factory.mktemp('more') / 'second').write_bytes(b'x' * 5_000_001)",
            True,
            id="size-live-factory-mixed",
        ),
        pytest.param(
            "size-live-factory-under",
            "unit",
            "import pytest\npytestmark = pytest.mark.small\n",
            "(tmp_path / 'first').write_bytes(b'x' * 6_000_000); "
            "(tmp_path_factory.mktemp('more') / 'second').write_bytes(b'x' * 2_000_000)",
            False,
            id="size-live-factory-under",
        ),
        pytest.param(
            "size-live-factory-teardown",
            "unit",
            """import pytest
pytestmark = pytest.mark.small
@pytest.fixture
"""
            "def tail(tmp_path_factory):\n    yield\n"
            "    (tmp_path_factory.mktemp('tail') / 'data').write_bytes(b'x' * 10_000_001)\n",
            "assert True",
            True,
            id="size-live-factory-teardown",
        ),
        pytest.param(
            "size-live-xdist",
            "unit",
            "import pytest\npytestmark = pytest.mark.small\n",
            "(tmp_path / 'payload').write_bytes(b'x' * 10_000_001)",
            True,
            id="size-live-xdist",
        ),
        pytest.param(
            "size-live-under",
            "unit",
            "import pytest\npytestmark = pytest.mark.small\n",
            "(tmp_path / 'payload').write_bytes(b'x' * 8)",
            False,
            id="size-live-under",
        ),
        pytest.param(
            "size-live-time-under",
            "unit",
            "import pytest\npytestmark = pytest.mark.small\n",
            "assert True",
            False,
            id="size-live-time-under",
        ),
        pytest.param(
            "size-live-time-over",
            "unit",
            "import pytest\npytestmark = pytest.mark.small\n",
            "assert True",
            True,
            id="size-live-time-over",
        ),
        pytest.param(
            "size-live-legacy",
            "unit",
            "import pytest\npytestmark = pytest.mark.small\n",
            "(tmp_path / 'payload').write_bytes(b'x' * 10_000_001)",
            False,
            id="size-live-legacy",
        ),
        pytest.param(
            "size-live-process",
            "old_area",
            "import pytest\npytestmark = pytest.mark.small\n",
            "import subprocess,sys; subprocess.run([sys.executable,'-c','pass'],check=True)",
            True,
            id="size-live-process",
        ),
        pytest.param(
            "size-live-network",
            "old_area",
            "import pytest\npytestmark = pytest.mark.small\n",
            "import socket; socket.socket().close()",
            True,
            id="size-live-network",
        ),
        pytest.param(
            "size-live-skip",
            "unit",
            "import pytest\npytestmark = pytest.mark.small\n",
            "pytest.skip('not measured')",
            True,
            id="size-live-skip",
        ),
        pytest.param(
            "size-live-large-merge",
            "old_area",
            "import pytest\npytestmark = pytest.mark.large\n",
            "print('NIGHTLY_EXECUTED')",
            False,
            id="size-live-large-merge",
        ),
        pytest.param(
            "size-live-large-missing",
            "old_area",
            "import pytest\npytestmark = pytest.mark.large\n",
            "open('unexpected-test-run', 'w').write('ran')",
            True,
            id="size-live-large-missing",
        ),
        pytest.param(
            "size-live-large-nightly",
            "old_area",
            "import pytest\npytestmark = pytest.mark.large\n",
            "print('NIGHTLY_EXECUTED')",
            False,
            id="size-live-large-nightly",
        ),
        (
            "integration-network",
            "integration",
            "",
            "import socket; socket.socket().connect(('192.0.2.1', 80))",
            True,
        ),
        (
            "integration-loopback",
            "integration",
            "",
            "import socket\nwith socket.socket() as s: s.bind(('127.0.0.1', 0))",
            False,
        ),
        (
            "numerical-import",
            "integration",
            "",
            "from scipy.special import ndtri; assert ndtri(0.5) == 0",
            False,
        ),
        (
            "integration-hostname",
            "integration",
            "",
            "import socket; socket.getaddrinfo('example.invalid', 80)",
            True,
        ),
        ("integration-import-hostname", "integration", "", "import hostname_import", True),
        (
            "outside-write",
            "unit",
            "from pathlib import Path\n",
            "Path('outside.txt').write_text('x')",
            True,
        ),
        (
            "outside-report-root",
            "unit",
            "from pathlib import Path\n",
            "import tempfile; Path(tempfile.gettempdir(),'user-output').write_text('x')",
            True,
        ),
        (
            "temporary",
            "unit",
            "",
            "p=tmp_path/'x'; p.write_text('x'); assert p.read_text() == 'x'",
            False,
        ),
        (
            "stdlib",
            "unit",
            "",
            "import math,_json,fractions; assert math.sqrt(4) == fractions.Fraction(2)",
            False,
        ),
        (
            "host",
            "unit",
            "",
            "from alphalattice.control.product_host.composition.application_session "
            "import WorkspaceApplicationSession",
            True,
        ),
        ("slow", "unit", "", "import threading; threading.Event().wait(1.2)", True),
        ("instruction-slow", "unit", "", "import threading; threading.Event().wait(1.2)", False),
        ("instruction-over", "unit", "", "assert True", False),
        ("instruction-failure", "unit", "", "assert False", True),
        ("instruction-gate", "unit", "", "import threading; threading.Event().wait(1.2)", True),
        ("instruction-walltime", "unit", "", "import threading; threading.Event().wait(1.2)", True),
        ("own-over", "unit", "", "assert True", True),
        ("own-under", "unit", "", "assert True", False),
        ("own-boundary", "unit", "", "assert True", False),
        ("own-bad", "unit", "", "assert True", True),
        (
            "own-subtests",
            "unit",
            "",
            "for i in range(3):\n    with subtests.test(i=i): assert i < 3",
            False,
        ),
        (
            "cached-connection",
            "unit",
            "import duckdb\nconnection=duckdb.connect(':memory:')\n",
            "connection.sql('select 1').fetchall()",
            True,
        ),
        ("primary-agree", "unit", "", "", False),
        ("primary-mismatch", "integration", "", "", True),
        ("primary-legacy", "old_area", "", "", False),
        ("primary-mixed", "unit", "", "", True),
        (
            "bound-helper",
            "unit",
            "from duckdb import connect\ndef helper(): connect(':memory:').close()\n",
            "helper()",
            True,
        ),
        (
            "bound-native",
            "unit",
            "from numpy import arange\n",
            "assert list(arange(2)) == [0, 1]",
            False,
        ),
        ("native-file", "unit", "from numpy import fromfile\n", "fromfile('outside.bin')", True),
        ("native-local", "unit", "", "import numpy as np; np.fromfile('outside.bin')", True),
        (
            "native-lookalike",
            "unit",
            "class Compute:\n    @staticmethod\n    def fromfile(): return 2\n",
            "import numpy as np\nassert Compute.fromfile() == 2 and list(np.arange(2)) == [0, 1]",
            False,
        ),
        ("native-memmap", "unit", "import numpy as np\n", "np.memmap('outside.bin')", True),
        (
            "numpy-save",
            "unit",
            "import numpy as np\n",
            "np.save('outside.npy', np.arange(2))",
            True,
        ),
        ("dynamic-import", "unit", "import importlib\n", "importlib.import_module('duckdb')", True),
        (
            "dynamic-stdlib",
            "unit",
            "import importlib\n",
            "assert importlib.import_module('math').sqrt(4) == 2",
            False,
        ),
        (
            "bound-loader",
            "unit",
            "from importlib import import_module\n",
            "import_module('duckdb')",
            True,
        ),
        (
            "outside-read",
            "unit",
            "from pathlib import Path\nPath('outside.txt').write_text('x')\n",
            "Path('outside.txt').read_text()",
            True,
        ),
        (
            "fixture-setup",
            "unit",
            """import pytest
@pytest.fixture(autouse=True)
def held():
 import duckdb; duckdb.connect(':memory:').close()
""",
            "pass",
            True,
        ),
        (
            "fixture-teardown",
            "unit",
            """import pytest
@pytest.fixture(autouse=True)
def held():
 yield
 import duckdb; duckdb.connect(':memory:').close()
""",
            "pass",
            True,
        ),
        pytest.param(
            "observe",
            "old_area",
            "import subprocess,sys,threading\n"
            "def child(): subprocess.run([sys.executable,'-c','pass'],check=True)\n",
            "(tmp_path/'written').write_text('x'); "
            "t=threading.Thread(target=child); t.start(); t.join(); child()",
            False,
            marks=pytest.mark.proves("testing.observed_test_resources"),
        ),
        pytest.param(
            "observe-unavailable",
            "old_area",
            "",
            "assert True",
            False,
            marks=pytest.mark.proves("testing.observed_test_resources"),
        ),
        pytest.param(
            "observe-sidecar-unavailable",
            "old_area",
            "",
            "assert True",
            False,
            marks=pytest.mark.proves("testing.observed_test_resources"),
        ),
        pytest.param(
            "observe-unit",
            "unit",
            "",
            "assert 2+2 == 4",
            False,
            marks=pytest.mark.proves("testing.observed_test_resources"),
        ),
        pytest.param(
            "observe-db",
            "old_area",
            "",
            "import sqlite3; sqlite3.connect(':memory:').close()",
            False,
            marks=pytest.mark.proves("testing.observed_test_resources"),
        ),
        pytest.param(
            "observe-network",
            "old_area",
            "",
            "import socket; socket.socket().close()",
            False,
            marks=pytest.mark.proves("testing.observed_test_resources"),
        ),
        ("shape", "old_area", "", "", False),
        (
            "hook",
            "old_area",
            "",
            AUDIT_HOOK_SAMPLE,
            False,
        ),
    ]
)


LAYER_IDS = (
[
        "fd-provenance",
        "module-inventory",
        "root-pure",
        "root-session-golden",
        "root-session-golden-error",
        "root-session-seed",
        "root-session-seed-error",
        "root-lookalike",
        "root-runtime",
        "root-cached",
        "database",
        "bound-database",
        "preloaded-pure",
        "installed-metadata",
        "installed-timezone",
        "project-read",
        "integration",
        "legacy",
        "process",
        "network",
        "network-caught",
        "size-live-over",
        "size-live-factory",
        "size-live-factory-base",
        "size-live-factory-mixed",
        "size-live-factory-under",
        "size-live-factory-teardown",
        "size-live-xdist",
        "size-live-under",
        "size-live-time-under",
        "size-live-time-over",
        "size-live-legacy",
        "size-live-process",
        "size-live-network",
        "size-live-skip",
        "size-live-large-merge",
        "size-live-large-nightly",
        "size-live-large-missing",
        "integration-network",
        "integration-loopback",
        "numerical-import",
        "integration-hostname",
        "integration-import-hostname",
        "outside-write",
        "outside-report-root",
        "temporary",
        "stdlib",
        "host",
        "slow",
        "instruction-slow",
        "instruction-over",
        "instruction-failure",
        "instruction-gate",
        "instruction-walltime",
        "own-over",
        "own-under",
        "own-boundary",
        "own-bad",
        "own-subtests",
        "cached-connection",
        "primary-agree",
        "primary-mismatch",
        "primary-legacy",
        "primary-mixed",
        "bound-helper",
        "bound-native",
        "native-file",
        "native-local",
        "native-lookalike",
        "native-memmap",
        "numpy-save",
        "dynamic-import",
        "dynamic-stdlib",
        "bound-loader",
        "outside-read",
        "fixture-setup",
        "fixture-teardown",
        "observe",
        "observe-unavailable",
        "observe-sidecar-unavailable",
        "observe-unit",
        "observe-db",
        "observe-network",
        "shape",
        "hook",
    ]
)


DECLARED_CASES = (
[(), ("small",), ("large",), ("medium", "medium"), ("small", "large"), ("registered",)]
    + [(name,) for name in ("timeout", "bad-reason", "bad-limit", "duplicate", "owner-large")]
    + [pytest.param(("null-size-contract",), id="null-size-contract")]
    + [("strict",)]
    + [
        pytest.param(("cost-" + case,), id="cost-" + case)
        for case in (
            "medium-new",
            "medium-pass",
            "skipped-selected",
            "skipped-untouched",
            "small-new",
            "small-pass",
            "grown",
            "within",
            "untouched",
            "loaded",
            "bad-measurement",
            "wrong-confirm-node",
            "missing-confirm-node",
            "generator",
            "zero-debt",
            "base-grown",
            "base-within",
            "missing-base",
            "loaded-base",
            "wrong-base",
            "failed-confirm",
            "failed-row",
            "cached-row",
            "cached-untouched",
            "loaded-required",
            "clean-required-under",
            "loaded-required-under",
            "base-clean-within",
            "base-clean-grown",
            "base-clean-fast",
        )
    ]
    + [("post_" + stage,) for stage in ("commit", "post-merge", "candidate", "nightly")]
    + [("post-kind",), ("post-repeat",)]
    + [("lane_" + lane,) for lane in ("merge", "candidate", "nightly")]
    + [("framework-" + case,) for case in ("entry", "reason", "pattern", "module")]
    + [("backend-" + case,) for case in ("empty", "kind", "type", "reason", "duplicate")]
    + [pytest.param(("sibling-race",), id="sibling-race")]
    + [
        pytest.param(("admission-" + case,), id="admission-" + case)
        for case in (
            "small-in",
            "small-time",
            "small-bytes",
            "medium-in",
            "medium-time",
            "medium-bytes",
            "large",
            "identity",
            "receipt",
            "footprint",
        )
    ]
    + [
        pytest.param(("axis-" + case,), id="axis-" + case)
        for case in ("measured", "fallback", "large")
    ]
)


ADMISSION_CASES = (
[
        ("unregistered", {}, "proves nothing"),
        ("schema-preserved-time", {}, ""),
        ("seed-junction", {}, ""),
        ("unknown", {}, "unregistered or retired"),
        ("product_host.port_reuse", {}, ""),
        ("host.port_reuse", {}, "its id is product_host.<words>"),
        ("product_host.port_reuse", {"description": "One holds. Two hold."}, "one 5-40 word"),
        ("nowhere.port_reuse", {"owner": "nowhere"}, "owns a noun"),
        ("product_host.port_reuse", {"layer": "e2e", "seconds": 600}, "(journey, large)"),
        ("product_host.owed_work", {"description": "A detached Host keeps owed work."}, "near"),
        ("product_host.port_reuse", {"layer": "e2e", "exception": "decided"}, ""),
        ("product_host.port_reuse", {"layer": "e2e", "exception": {"decided": True}}, "decision"),
        ("product_host.port_reuse", {"layer": "e2e", "exception": {"kinds": 1}}, "decision"),
        ("product_host.port_reuse", {"lower_layer": ""}, "above unit"),
        ("product_host.port_reuse", {"layer": "unit", "seconds": 1, "lower_layer": ""}, ""),
        (HELD, None, "was removed"),
        ("schema-unreadable-directory", {}, ""),
        ("schema-unreadable-entry", {}, ""),
        ("entry-order", {}, ""),
    ]
)


HOLDER_CASES = (
[
        "second",
        "same-file",
        "split-file",
        "whole-file-move",
        "same-file-class",
        "legacy-files",
        "legacy-extension",
        "legacy-renamed-function",
        "legacy-extra-file",
        "different-layer",
        "parameter",
        "move",
        "edited-copy",
        "edited-move",
        "legacy",
        "read-once",
        "edited-trivia-move",
        "unchanged",
        "inherited-proof",
        "marker-addition",
        "new-marker",
    ]
)
