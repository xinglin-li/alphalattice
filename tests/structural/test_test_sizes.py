"""Collection resolves one owned size and refuses contradictory declarations."""

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.mark.proves("testing.layer_contract")
@pytest.mark.parametrize(
    "case,layer,header,body,refused",
    [
        ("module-inventory", "unit", "", "import sys; sys.modules.get('duckdb')", True),
        ("root-pure", "unit", "", "", False),
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
        pytest.param(
            "size-live-over",
            "unit",
            "import pytest\npytestmark = pytest.mark.small\n",
            "(tmp_path / 'payload').write_bytes(b'x' * 10_000_001)",
            True,
            id="size-live-over",
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
        ("integration-network", "integration", "", "import socket; socket.socket().close()", True),
        (
            "numerical-import",
            "integration",
            "",
            "from scipy.special import ndtri; assert ndtri(0.5) == 0",
            False,
        ),
        ("integration-hostname", "integration", "", "import socket; socket.gethostname()", True),
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
            """
from devtools.testing import audit as observe
seen=[]
with observe.subscribe(lambda event,args: seen.append(('a',event))):
    with observe.subscribe(lambda event,args: seen.append(('b',event))):
        __import__('sys').audit('test.contract.first')
with observe.subscribe(lambda event,args: seen.append(('c',event))):
    __import__('sys').audit('test.contract.second')
assert seen.count(('a','test.contract.first')) == 1
assert seen.count(('b','test.contract.first')) == 1
assert seen.count(('c','test.contract.second')) == 1
assert ('a','test.contract.second') not in seen
import pytest
calls=[]
def stopped(event,args):
    if event == 'test.contract.failure':
        calls.append(event)
        raise ValueError('stopped')
with pytest.raises(ValueError):
    with observe.subscribe(stopped):
        __import__('sys').audit('test.contract.failure')
__import__('sys').audit('test.contract.failure')
assert calls == ['test.contract.failure']
""",
            False,
        ),
    ],
    ids=[
        "module-inventory",
        "root-pure",
        "root-lookalike",
        "root-runtime",
        "root-cached",
        "database",
        "bound-database",
        "preloaded-pure",
        "integration",
        "legacy",
        "process",
        "network",
        "size-live-over",
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
        "numerical-import",
        "integration-hostname",
        "integration-import-hostname",
        "outside-write",
        "outside-report-root",
        "temporary",
        "stdlib",
        "host",
        "slow",
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
    ],
)
def test_explicit_layers_refuse_resources_and_keep_their_execution_bound(
    tmp_path, monkeypatch, case, layer, header, body, refused
):
    if case in {"temporary", "root-pure"}:
        monkeypatch.setenv(
            "ALPHALATTICE_PIPELINE_REQUIRED_TESTS", json.dumps(["tests/outer.py::test_outer"])
        )
    root = Path(__file__).resolve().parents[2]
    if case.startswith("root-"):
        from tests.structural.pytest_support import pytest_checkout

        checkout, environment = pytest_checkout(tmp_path, root)
        marker = Path(environment["TEMP"]) / "given" / "framework-entered"
        if case == "root-lookalike":
            target = checkout / "tests/unit/test_collection_probe.py"
            target.write_text(
                target.read_text("utf-8")
                + """
from duckdb import connect as cached_backend
import pytest
@pytest.fixture(autouse=True)
def _memory_checks_read_no_host():
    cached_backend(':memory:').close()
""",
                encoding="utf-8",
            )
        elif case in {"root-runtime", "root-cached"}:
            target = checkout / "tests/conftest.py"
            text = target.read_text("utf-8")
            if case == "root-cached":
                text = text.replace(
                    "import pytest\n",
                    "import pytest\nfrom duckdb import connect as cached_backend\n",
                    1,
                )
            function = next(
                node
                for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef) and node.name == "_memory_checks_read_no_host"
            )
            lines = text.splitlines(keepends=True)
            offset = function.body[1].lineno - 1  # after its docstring, before the unit no-op
            fault = [
                "    request.getfixturevalue('tmp_path_factory').getbasetemp()"
                ".joinpath('framework-entered').write_text('entered')\n"
            ]
            fault += (
                ["    cached_backend(':memory:').close()\n"]
                if case == "root-cached"
                else ["    import duckdb\n", "    duckdb.connect(':memory:').close()\n"]
            )
            lines[offset:offset] = fault
            target.write_text("".join(lines), encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "pytest",
                "-q",
                "--basetemp",
                str(marker.parent),
                str(checkout / "tests" / "unit" / "test_collection_probe.py"),
            ],
            cwd=checkout,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
        )
        output = result.stdout + result.stderr
        assert result.returncode == (1 if refused else 0), output
        if refused:
            assert "testing.layer_contract" in output
        if case == "root-runtime":
            assert marker.is_file() and "unit dependency duckdb" in output
        else:
            assert not marker.exists()
        if case in {"root-cached", "root-lookalike"}:
            assert "_duckdb" in output  # selected cached binding, before the fixture's body
        return
    if case == "shape":
        from scripts.check_test_shape import problems

        assert any(
            "testing.layer_contract" in row
            for row in problems("tests/unit/test_x.py", "", "import duckdb\n")
        )
        assert not problems("tests/integration/test_x.py", "", "import duckdb\n")
        assert not problems("tests/unit/test_x.py", "", "import math, _json\n")
        return
    if case.startswith("primary-"):
        from types import SimpleNamespace

        from devtools.testing.collection import _proven
        from devtools.testing.sizes import load_policy

        rows = {"testing.a": {"layer": "unit"}, "testing.b": {"layer": "integration"}}
        ids = ["testing.a", "testing.b"] if case == "primary-mixed" else ["testing.a"]
        item = SimpleNamespace(
            nodeid=f"tests/{layer}/test_sample.py::test_sample",
            iter_markers=lambda *args: [SimpleNamespace(name="proves", args=ids)],
        )
        if refused:
            with pytest.raises(ValueError, match=r"testing\.layer_contract"):
                _proven(item, rows, load_policy(root))
        else:
            assert _proven(item, rows, load_policy(root))[1]["layer"] == "unit"
        return
    target = tmp_path / "tests" / layer / "test_sample.py"
    target.parent.mkdir(parents=True)
    if case == "integration-import-hostname":
        (tmp_path / "hostname_import.py").write_text(
            "import socket\nsocket.gethostname()\n", "utf-8"
        )
    target.write_text(
        header
        + (
            "def test_sample(tmp_path, subtests):\n"
            if case == "own-subtests"
            else "def test_sample(tmp_path):\n"
        )
        + textwrap.indent(textwrap.dedent(body).strip(), "    ")
        + "\n",
        encoding="utf-8",
    )
    phase_args = (
        ["-p", "devtools.pipeline.receipt", "-p", "xdist.plugin", "-n", "2"]
        if case.startswith("observe")
        else []
    )
    if case.startswith("size-live-"):
        from devtools.pipeline import receipt

        writer = receipt.Writer(
            tmp_path / "receipt.jsonl",
            {"stage": "commit", "host": "workshop", "command": ["fixture"]},
            [{"workspace": {}}],
        )
        writer.write(
            {"workspace": {}},
            {"repo": "product", "base": "a" * 40, "head": "b" * 40, "clean": True},
        )
        writer.stream.close()
        families = (
            [] if case == "size-live-legacy" else [f"tests/{layer}/test_sample.py::test_sample"]
        )
        extra = ""
        if case.startswith("size-live-large-"):
            node = f"tests/{layer}/test_sample.py::test_sample"
            extra = (
                "from devtools.testing import collection\n"
                "original = collection.load_policy\n"
                "def sized(root):\n"
                " p = original(root)\n"
                f" p['large'].append({{'pattern': {node!r}, 'owner': 'tests', "
                + (
                    "'reason': 'A synthetic journey.'})\n"
                    if case == "size-live-large-missing"
                    else "'reason': 'A synthetic journey.', 'size_contract': 'nightly_only'})\n"
                )
                + f" p['mandatory'].append({node!r})\n"
                + " return p\ncollection.load_policy = sized\n"
            )
        (tmp_path / "size-stage-selection.json").write_text(
            json.dumps({"base": "a" * 40, "head": "b" * 40, "families": families}),
            encoding="utf-8",
        )
        (tmp_path / "size_fixture.py").write_text(extra, encoding="utf-8")
        phase_args = (
            ["-p", "size_fixture", "-p", "devtools.pipeline.receipt"]
            + (["-s"] if case.startswith("size-live-large-") else [])
            + (
                ["-p", "xdist.plugin", "-n", "2", "--max-worker-restart=0"]
                if case in {"size-live-xdist", "size-live-large-missing"}
                else []
            )
        )
        if case.startswith("size-live-time-"):
            call_seconds = 0.4 if case.endswith("under") else 0.5
            (tmp_path / "size_phase.py").write_text(
                "import pytest\n"
                "@pytest.hookimpl(wrapper=True, tryfirst=True)\n"
                "def pytest_runtest_makereport(item, call):\n"
                + f" call.duration = {{'setup': 0.4, 'call': {call_seconds}, "
                "'teardown': 0.2}[call.when]\n return (yield)\n",
                encoding="utf-8",
            )
            phase_args += ["-p", "size_phase"]
    if case == "observe-sidecar-unavailable":
        for worker in ("main", "gw0", "gw1"):
            (tmp_path / f"node-observations-{worker}.jsonl").mkdir()
    if case == "observe-unavailable":
        (tmp_path / "disabled_audit.py").write_text(
            "from devtools.testing import audit\n"
            "def unavailable(callback): raise OSError('audit unavailable')\n"
            "audit.subscribe = unavailable\n",
            encoding="utf-8",
        )
        phase_args[:0] = ["-p", "disabled_audit"]
    if case.startswith("own-"):
        call_seconds = {"own-bad": "float('nan')", "own-over": "0.7", "own-boundary": "0.6"}.get(
            case, "0.4"
        )
        (tmp_path / "phase_probe.py").write_text(
            "import pytest\n"
            "@pytest.hookimpl(wrapper=True, tryfirst=True)\n"
            "def pytest_runtest_makereport(item, call):\n"
            f"    call.duration = {{'setup': 0.2, 'call': {call_seconds}, "
            "'teardown': 0.2}[call.when]\n"
            "    return (yield)\n",
            encoding="utf-8",
        )
        phase_args = ["-p", "phase_probe", "--log-file", str(tmp_path / "events.log")]
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-p",
            "devtools.testing.collection",
            "-p",
            "pytest_timeout",
            *phase_args,
            "--junitxml",
            str(tmp_path / "child.xml"),
            "-c",
            str(root / "pyproject.toml"),
            "--rootdir",
            str(tmp_path),
            "--confcutdir",
            str(tmp_path),
            "--noconftest",
            "--basetemp",
            str(tmp_path / "child-temp"),
            "-o",
            "addopts=",
            "-q",
            str(target),
        ],
        cwd=tmp_path,
        env=dict(
            {
                key: value
                for key, value in os.environ.items()
                if key
                not in {
                    "ALPHALATTICE_PIPELINE_REQUIRED_TESTS",
                    "PYTEST_CURRENT_TEST",
                    "ALPHALATTICE_REACH_FLOOR",
                }
                if key != "ALPHALATTICE_CHANGED_PATHS"
                or not os.environ.get("ALPHALATTICE_REACH_FLOOR")
            },
            PYTHONPATH=os.pathsep.join([str(tmp_path), str(root / "src")]),
            ALPHALATTICE_NETWORK_DISABLED="1",
            ALPHALATTICE_TEST_LANE="merge"
            if case in {"size-live-large-merge", "size-live-large-missing"}
            else "nightly"
            if case == "size-live-large-nightly"
            else "",
            ALPHALATTICE_PIPELINE_REQUIRED_TESTS=json.dumps(
                [f"tests/{layer}/test_sample.py::test_sample"]
            )
            if case == "size-live-large-merge"
            else "[]",
            ALPHALATTICE_PIPELINE_STAGE="nightly"
            if case == "size-live-large-nightly"
            else "commit"
            if case.startswith("size-live-")
            else "",
            ALPHALATTICE_PIPELINE_EVENTS=(
                str(tmp_path / "test-events.jsonl")
                if case.startswith("observe") or case.startswith("size-live-")
                else ""
            ),
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            PYTEST_PLUGINS="",
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        | getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    output = result.stdout + result.stderr
    if case == "size-live-large-missing":
        assert (
            result.returncode != 0
            and "test_size.large_contract" in output
            and "size_contract=nightly_only" in output
        ), output
        assert "INTERNALERROR" not in output and "WorkerController" not in output, output
        events = receipt.read_test_events(tmp_path / "test-events.jsonl")
        assert len(events) == 1
        event = events[0]
        assert event["phase"] == "collection"
        assert event["failure"]["type"] == "test_size.large_contract"
        assert "size_contract=nightly_only" in event["failure"]["message"]
        classes = receipt.red_classes(events)
        assert classes == {event["node"]: "unclassified"}
        other = event | {"failure": event["failure"] | {"type": "other"}}
        assert receipt.red_classes([other]) == {event["node"]: "unclassified"}
        path = tmp_path / "sealed.jsonl"
        sealed = receipt.Writer(
            path,
            {"stage": "commit", "host": "workshop", "command": ["fixture"]},
            [{"workspace": {}}, {"configuration": {}}],
        )
        sealed.write(
            {"workspace": {}},
            {"repo": "product", "base": "a" * 40, "head": "b" * 40, "clean": True},
        )
        sealed.write(
            {"configuration": {}},
            {
                "definition": "d",
                "definition_sha256": "d" * 64,
                "rows": ["tests"],
                "providers": {"tests": "product"},
                "workers": 2,
                "bound_seconds": 900,
            },
        )
        red = sealed.tests(
            "tests",
            {"status": "FAIL", "exit": result.returncode, "seconds": 1, "provider": "product"},
            tmp_path / "test-events.jsonl",
        )
        sealed.write({"metrics": {}}, {"wall_seconds": 1})
        sealed.finish(
            {"result": "FAIL", "exit": 1, "red_nodes": red} | sealed.red_summary(over_bound=False)
        )
        assert receipt.verify(path)["result"] == "FAIL"
        lines = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        assert lines[-1]["finished"]["red_class"] == "unclassified"
        future = tmp_path / "future.jsonl"
        for line in lines:
            if "test_result" in line:
                line["test_result"]["class"] = "defect"
        lines[-1]["finished"].update(red_class="defect", red_classes={"defect": 1})
        future.write_text("\n".join(map(json.dumps, lines)) + "\n", encoding="utf-8")
        assert receipt.verify(future)["result"] == "FAIL"
        for line in lines:
            if "test_result" in line:
                line["test_result"]["failure"]["type"] = "other"
        future.write_text("\n".join(map(json.dumps, lines)) + "\n", encoding="utf-8")
        with pytest.raises(receipt.ReceiptError, match="receipt_inconsistent"):
            receipt.verify(future)
        assert event["failure"]["message"] in path.read_text("utf-8")
        assert not (tmp_path / "unexpected-test-run").exists()
        return
    if case == "size-live-large-merge":
        assert result.returncode == 5 and "test_size.nightly_only" in output, output
        assert "NIGHTLY_EXECUTED" not in output
        return
    if case == "size-live-large-nightly":
        assert result.returncode == 0 and "NIGHTLY_EXECUTED" in output, output
        return
    if refused:
        expected = (
            "test_size.admission_limit"
            if case in {"size-live-over", "size-live-xdist", "size-live-time-over"}
            else "test_size.admission_unmeasured"
            if case == "size-live-skip"
            else "Timeout"
            if case == "slow"
            else "testing.layer_contract"
        )
        assert result.returncode == 1 and expected in output, output
        if case in {"size-live-over", "size-live-xdist"}:
            assert "10000001" in output and "medium" in output
    else:
        assert result.returncode == 0, output
    if case in {
        "size-live-over",
        "size-live-under",
        "size-live-legacy",
        "size-live-xdist",
        "size-live-time-under",
        "size-live-time-over",
    }:
        paths = list(tmp_path.glob("node-observations-*.jsonl"))
        observed = [
            json.loads(line) for path in paths for line in path.read_text("utf-8").splitlines()
        ]
        assert len(observed) == 1
        if case == "size-live-legacy":
            assert "tmp_footprint_bytes" not in observed[0]
        else:
            assert observed[0]["tmp_footprint_bytes"] == (
                8
                if case == "size-live-under"
                else 0
                if case.startswith("size-live-time-")
                else 10_000_001
            )
            assert observed[0]["tmp_footprint_seconds"] >= 0
        timings = [
            json.loads(path.read_text("utf-8")) for path in tmp_path.glob("size-overhead-*.json")
        ]
        if case == "size-live-legacy":
            assert not timings
        else:
            assert timings and all(all(value >= 0 for value in row.values()) for row in timings)
    if case == "observe":
        observed = list(tmp_path.glob("node-observations-*.jsonl"))
        assert observed, "the test run must retain its observed resources beside JUnit"
        rows = [
            json.loads(line) for path in observed for line in path.read_text("utf-8").splitlines()
        ]
        assert len(rows) == 1 and rows[0]["node"].endswith("::test_sample")
        assert set(rows[0]["phases"]) == {"setup", "call", "teardown"}
        assert all(value >= 0 for value in rows[0]["phases"].values())
        python = Path(sys.executable).name.lower()
        assert rows[0]["subprocesses"][python]["count"] == 2
        assert rows[0]["subprocesses"][python]["background_count"] == 1
        sites = rows[0]["subprocesses"][python]["callsites"]
        assert len(sites) == 1 and sites[0]["function"] == "child" and sites[0]["count"] == 2
        assert sites[0]["file"] == "test_sample.py" and sites[0]["line"] > 0
        assert rows[0]["subprocesses"][python]["sampled_seconds"] is None
        assert "background-thread attribution" in rows[0]["unknown"]
        assert "descendant subprocesses" in rows[0]["unknown"]
        assert rows[0]["observed_layer"] == "integration"
        assert "duckdb" in rows[0]["unknown"]
    if case == "observe-unavailable":
        paths = list(tmp_path.glob("node-observations-*.jsonl"))
        rows = [json.loads(line) for path in paths for line in path.read_text("utf-8").splitlines()]
        assert len(rows) == 1 and rows[0]["observed_layer"] == "unknown"
        assert "audit unavailable" in rows[0]["unknown"]
    if case == "observe-sidecar-unavailable":
        assert all(
            (tmp_path / f"node-observations-{worker}.jsonl").is_dir()
            for worker in ("main", "gw0", "gw1")
        )
    if case in {"observe-unit", "observe-db", "observe-network"}:
        paths = list(tmp_path.glob("node-observations-*.jsonl"))
        rows = [json.loads(line) for path in paths for line in path.read_text("utf-8").splitlines()]
        expected = {"observe-unit": "unit", "observe-db": "integration", "observe-network": "e2e"}
        assert len(rows) == 1 and rows[0]["observed_layer"] == expected[case]
        if case == "observe-db":
            assert rows[0]["database"] is True
        if case == "observe-network":
            assert rows[0]["network"] is True
    if case in {"database", "fixture-setup", "fixture-teardown", "numpy-save"}:
        from xml.etree import ElementTree as ET

        cases = list(ET.parse(tmp_path / "child.xml").iter("testcase"))
        assert len(cases) == 1
        assert sum(len(c.findall("error")) + len(c.findall("failure")) for c in cases) == 1
    if case == "own-over":
        assert '"event": "TIMEOUT"' in (tmp_path / "events.log").read_text("utf-8")
    if case == "numpy-save":
        assert "file outside temporary root" in output
    if case in {"native-file", "native-local", "native-memmap"}:
        assert "dependency numpy." in output


@pytest.mark.parametrize(
    "declared",
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
    + [("post-kind",), ("post-structural",), ("post-repeat",)]
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
    ],
)
@pytest.mark.proves("testing.layer_contract")
def test_collection_resolves_owned_size_or_names_the_conflict(tmp_path, monkeypatch, declared):
    """Owned size declarations enforce their bounds and provenance, refusing invalid catalogs."""
    if not declared:
        monkeypatch.setenv(
            "ALPHALATTICE_PIPELINE_REQUIRED_TESTS", json.dumps(["tests/outer.py::test_outer"])
        )
    root = Path(__file__).resolve().parents[2]
    if declared and declared[0].startswith("admission-"):
        from devtools.testing import sizes

        decide = getattr(sizes, "admission_limit", None)
        assert callable(decide), "new and edited tests need one size-admission owner"
        if declared[0] == "admission-footprint":
            footprint = getattr(sizes, "tmp_footprint", None)
            assert callable(footprint), "the size owner must measure the retained temporary tree"
            folder = tmp_path / "footprint"
            (folder / "nested").mkdir(parents=True)
            (folder / "a").write_bytes(b"abc")
            (folder / "nested/b").write_bytes(b"12345")
            assert footprint(folder) == 8 and footprint(folder / "missing") == 0
            (folder / "nested/b").unlink()
            assert footprint(folder) == 3
            with monkeypatch.context() as broken:
                broken.setattr(
                    sizes.os,
                    "scandir",
                    lambda path: (_ for _ in ()).throw(PermissionError("denied")),
                )
                with pytest.raises(ValueError, match="footprint_unmeasured"):
                    footprint(folder)
            return
        if declared[0] == "admission-receipt":
            from types import SimpleNamespace

            from devtools.pipeline import receipt
            from devtools.testing import collection

            config = SimpleNamespace(pluginmanager=SimpleNamespace(get_plugin=lambda name: None))
            monkeypatch.setenv("ALPHALATTICE_PIPELINE_STAGE", "commit")
            monkeypatch.setenv("ALPHALATTICE_PIPELINE_EVENTS", str(tmp_path / "tests-events.jsonl"))
            monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_outer.py::test_outer (call)")
            assert collection._size_families(config) is None  # a nested controller
            assert collection._size_families(SimpleNamespace(workerinput={})) is None
            monkeypatch.delenv("PYTEST_CURRENT_TEST")
            assert (
                collection._size_families(
                    SimpleNamespace(workerinput={"alphalattice_size_scope": False})
                )
                is None
            )  # nested xdist workers inherit their own controller's scope
            with pytest.raises(pytest.UsageError, match="changed_unmeasured"):
                collection._size_families(SimpleNamespace(workerinput={}))
            monkeypatch.delenv("ALPHALATTICE_PIPELINE_REQUIRED_TESTS", raising=False)
            assert collection._size_families(config) is None  # nested fast-guard pytest
            monkeypatch.setenv("ALPHALATTICE_PIPELINE_REQUIRED_TESTS", "[]")
            with pytest.raises(pytest.UsageError, match="changed_unmeasured"):
                collection._size_families(config)  # a test row must have its receipt owner
            pending = getattr(receipt, "pending_workspace", None)
            assert callable(pending), "commit selection needs the receipt owner's pinned workspace"
            path = tmp_path / "receipt.jsonl"
            writer = receipt.Writer(
                path,
                {"stage": "commit", "host": "workshop", "command": ["fixture"]},
                [{"workspace": {}}],
            )
            writer.write(
                {"workspace": {}},
                {"repo": "product", "base": "a" * 40, "head": "b" * 40, "clean": True},
            )
            result = pending(path)
            assert (result.base, result.head) == ("a" * 40, "b" * 40)
            record = tmp_path / "size-stage-selection.json"
            events = tmp_path / "tests-events.jsonl"
            chosen = ["tests/test_a.py::test_one"]
            record.write_text(
                json.dumps({"base": "a" * 40, "head": "b" * 40, "families": chosen}),
                encoding="utf-8",
            )
            assert receipt.size_families(events) == tuple(chosen)
            monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_outer.py::test_outer (call)")
            other = "tests/test_b.py::test_other"
            assert collection._nested_required((*chosen, other)) == (other,)
            monkeypatch.setenv("ALPHALATTICE_PIPELINE_STAGE", "nightly")
            assert collection._nested_required((*chosen, other)) == (*chosen, other)
            monkeypatch.setenv("ALPHALATTICE_PIPELINE_STAGE", "commit")
            monkeypatch.delenv("PYTEST_CURRENT_TEST")
            record.write_text(
                json.dumps({"base": "a" * 40, "head": "c" * 40, "families": chosen}),
                encoding="utf-8",
            )
            with pytest.raises(ValueError, match="changed_unmeasured"):
                receipt.size_families(events)
            writer.stream.close()
            path.write_text("{", encoding="utf-8")
            with pytest.raises(receipt.ReceiptError, match="size_workspace_unmeasured"):
                pending(path)
            return
        if declared[0] == "admission-identity":
            from devtools.testing import requirements

            edited = getattr(requirements, "edited_families", None)
            assert callable(edited), "edited test families need an identity owner"
            old = (
                "def test_same(): pass\n"
                "@pytest.mark.parametrize('x', [1, 2])\n"
                "def test_param(x): pass\n"
            )
            new = old.replace("[1, 2]", "[1, 2, 3]") + "def test_new(): pass\n"
            before = requirements.file_holders("tests/test_a.py", old, size_scope=True)
            after = requirements.file_holders("tests/test_a.py", new, size_scope=True)
            assert edited(before, after) == [
                "tests/test_a.py::test_new",
                "tests/test_a.py::test_param",
            ]
            moved = requirements.file_holders("tests/test_b.py", old, size_scope=True)
            assert edited(before, moved, {"tests/test_b.py": "tests/test_a.py"}) == []
            marked_before = requirements.file_holders(
                "tests/test_a.py", "pytestmark = pytest.mark.medium\n" + old, size_scope=True
            )
            marked_after = requirements.file_holders(
                "tests/test_a.py", "pytestmark = pytest.mark.small\n" + old, size_scope=True
            )
            assert edited(marked_before, marked_after) == [
                "tests/test_a.py::test_param",
                "tests/test_a.py::test_same",
            ]
            from devtools.pipeline import stage

            changed = getattr(stage, "changed_families", None)
            assert callable(changed), (
                "the stage needs changed families from its exact Git endpoints"
            )
            repo = tmp_path / "repo"
            (repo / "tests").mkdir(parents=True)

            def git(*words):
                return subprocess.run(
                    ["git", "-C", str(repo), *words],
                    check=True,
                    capture_output=True,
                    text=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                ).stdout.strip()

            git("init", "-q")
            (repo / "tests/test_a.py").write_text(old, encoding="utf-8")
            git("add", ".")
            git(
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "base",
            )
            base = git("rev-parse", "HEAD")
            (repo / "tests/test_a.py").write_text(new, encoding="utf-8")
            (repo / "tests/extra_test.py").write_text(
                "def testSuffix(): pass\nclass TestOuter:\n"
                " class TestInner:\n  def test_nested(self): pass\n",
                encoding="utf-8",
            )
            git("add", ".")
            git(
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "head",
            )
            head = git("rev-parse", "HEAD")
            assert changed(repo, base, head) == [
                "tests/extra_test.py::TestOuter::TestInner::test_nested",
                "tests/extra_test.py::testSuffix",
                "tests/test_a.py::test_new",
                "tests/test_a.py::test_param",
            ]
            return
        node = "tests/structural/test_example.py::test_one"
        cases = {
            "small-in": ("small", 1.0, 10_000_000, None),
            "small-time": ("small", 1.01, 0, "medium"),
            "small-bytes": ("small", 0.1, 10_000_001, "medium"),
            "medium-in": ("medium", 30.0, 500_000_000, None),
            "medium-time": ("medium", 30.01, 0, "large"),
            "medium-bytes": ("medium", 0.1, 500_000_001, "large"),
            "large": ("large", 99999.0, 999_999_999, None),
        }
        size, seconds, footprint, next_size = cases[declared[0][10:]]
        policy = sizes.load_policy(root)
        if next_size:
            with pytest.raises(ValueError) as refused:
                decide(node, size, seconds, footprint, policy)
            assert all(
                x in str(refused.value)
                for x in (
                    node,
                    str(seconds),
                    str(footprint),
                    next_size,
                    "tmp footprint at teardown",
                )
            )
        else:
            assert decide(node, size, seconds, footprint, policy) is None
        if size == "large":
            policy["large"].append(
                {
                    "pattern": node,
                    "owner": "tests.structural",
                    "reason": "Exercises the registered journey.",
                    "size_contract": "nightly_only",
                }
            )
            assert not sizes.included(node, "large", "merge", policy, None)
            assert sizes.included(node, "large", "nightly", policy, None)
        return
    if declared and declared[0].startswith("axis-"):
        from types import SimpleNamespace

        from devtools.testing import collection
        from devtools.testing.sizes import load_policy

        node = "tests/workspace_readiness/test_workspace_preparation.py::test_measured"
        policy = load_policy(root)
        case = declared[0][5:]
        if case == "measured":
            policy["defaults"].append(
                {"pattern": node, "owner": "tests.workspace_readiness", "size": "small"}
            )
        if case == "large":
            policy["large"].append(
                {
                    "pattern": node,
                    "owner": "tests.workspace_readiness",
                    "reason": "Exercises a prepared workspace.",
                }
            )

        class Item:
            nodeid = node

            def iter_markers(self, name=None):
                marks = [SimpleNamespace(name="proves", args=("example.integration",))]
                return iter(m for m in marks if name is None or m.name == name)

        rows = {"example.integration": {"layer": "integration"}}
        ids, row, _, (size, owner, _, _) = collection._proven(Item(), rows, policy)
        assert ids == ["example.integration"] and row["layer"] == "integration"
        assert (size, owner) == (
            {"measured": "small", "fallback": "medium", "large": "large"}[case],
            "tests.workspace_readiness",
        )
        return
    if declared and declared[0].startswith("cost-"):
        from devtools.testing import sizes

        decide = getattr(sizes, "measured_costs", None)
        assert callable(decide), "test_size.cost requires one shared measured decision"
        mode = declared[0][5:]
        if mode in {
            "base-grown",
            "base-within",
            "missing-base",
            "loaded-base",
            "wrong-base",
            "base-clean-within",
            "base-clean-grown",
            "base-clean-fast",
        }:
            import inspect

            assert "baseline" in inspect.signature(decide).parameters
            node = "tests/example/test_sample.py::test_one"
            row = {
                "node": node,
                "seconds": 9
                if mode in {"base-clean-within", "base-clean-grown"}
                else 36
                if mode == "base-within"
                else 38,
                "large": False,
                "outcome": "passed",
                "existing": True,
                "limit_seconds": 10,
            }
            original = {
                "node": "wrong" if mode == "wrong-base" else node,
                "solo_seconds": 30,
                "clean": mode != "loaded-base",
                "other_holders_before": [] if mode != "loaded-base" else [{"line": "busy"}],
                "other_holders_after": [],
                "state": "MEASURED",
            }

            def confirm(*args):
                return {
                    **original,
                    "node": node,
                    "solo_seconds": 36
                    if mode == "base-clean-within"
                    else 38
                    if mode == "base-clean-grown"
                    else 9
                    if mode == "base-clean-fast"
                    else row["seconds"],
                    "clean": True,
                    "other_holders_before": [],
                }

            def baseline(*args):
                assert mode != "base-clean-fast", "fast clean result needs no baseline"
                return original

            kwargs = {"confirm": confirm, "baseline": None if mode == "missing-base" else baseline}
            policy = {
                "cost_margin": 1.25,
                "slow_baseline": {},
                "require_clean": mode.startswith("base-clean-"),
            }
            if mode in {"base-within", "base-clean-within"}:
                _, debt = decide([row], policy, **kwargs)
                assert debt == {node: {"measured": 30, "cap": 37.5}}
            elif mode == "base-clean-fast":
                _, debt = decide([row], policy, **kwargs)
                assert not debt
            else:
                with pytest.raises((ValueError, RuntimeError), match=r"test_size.cost"):
                    decide([row], policy, **kwargs)
            return
        node = "tests/example/test_sample.py::test_one"
        existing = mode in {"grown", "within", "cached-untouched"}
        bound = 1 if mode.startswith("small-") else 10
        seconds = {
            "grown": 38,
            "within": 36,
            "untouched": 30,
            "cached-untouched": 30,
            "clean-required-under": 9,
            "loaded-required-under": 9,
            "medium-pass": 9,
            "small-new": 1.2,
            "small-pass": 0.9,
        }.get(mode, 12)
        policy = {
            "cost_margin": 1.25,
            "slow_baseline": {node: {"measured": 30, "cap": 37.5}} if existing else {},
            "record_untouched": mode == "untouched",
            "require_clean": mode
            in {"loaded-required", "clean-required-under", "loaded-required-under"},
        }
        row = {
            "node": node,
            "seconds": seconds,
            "large": False,
            "size": "small" if bound == 1 else "medium",
            "outcome": "passed",
            "limit_seconds": bound,
        }
        measurement = {
            "node": node,
            "solo_seconds": seconds,
            "clean": mode not in {"loaded", "loaded-required", "loaded-required-under"},
            "other_holders_before": []
            if mode not in {"loaded", "loaded-required", "loaded-required-under"}
            else [{"line": "busy"}],
            "other_holders_after": [],
            "state": "MEASURED",
        }
        if mode == "bad-measurement":
            measurement["solo_seconds"] = float("nan")
        if mode in {"wrong-confirm-node", "missing-confirm-node"}:
            measurement["solo_seconds"] = 0.5
            if mode == "wrong-confirm-node":
                measurement["node"] = node + "_other"
            else:
                measurement.pop("node")
        if mode == "failed-confirm":
            measurement.update(solo_seconds=9, state="FAIL")
        if mode == "failed-row":
            row.update(seconds=9, outcome="failure")
        if mode == "cached-row":
            row.update(seconds=9, cached=True)
        if mode == "cached-untouched":
            row["cached"] = True

        def confirm(*args):
            return {**measurement, "node": args[0]} if mode == "generator" else measurement

        if mode in {"skipped-selected", "skipped-untouched"}:
            policy["require_clean"] = True
            other = node.replace("test_one", "test_two")
            rows = [{**row, "outcome": "skipped"}, {**row, "node": other, "seconds": 9}]
            if mode == "skipped-selected":
                with pytest.raises(RuntimeError, match="cost_unmeasured"):
                    decide(rows, policy, confirm=confirm, judged={node})
            else:
                accepted, debt = decide(
                    rows,
                    policy,
                    confirm=lambda *args: {**measurement, "node": args[0], "solo_seconds": 9},
                    judged={other},
                )
                assert [item["node"] for item in accepted] == [other] and not debt
            return
        if mode == "generator":
            other = node.replace("test_one", "test_two")
            with pytest.raises(ValueError, match=r"test_size.cost"):
                decide(
                    [{**row, "seconds": 9}, {**row, "node": other}],
                    policy,
                    confirm=confirm,
                    judged=(n for n in (other, node)),
                )
            return
        if mode == "zero-debt":
            policy["slow_baseline"] = {node: {"measured": 0, "cap": 0}}
            try:
                result, debt = decide([{**row, "seconds": 0}], policy)
            except ValueError:
                result, debt = None, {}
            assert result is not None and debt == policy["slow_baseline"]
            return
        if mode in {"wrong-confirm-node", "missing-confirm-node"}:
            refused = False
            try:
                decide([row], policy, confirm=confirm)
            except (ValueError, RuntimeError) as error:
                assert "test_size.cost" in str(error)
                refused = True
            assert refused, "a different or missing solo node cannot establish cost standing"
            return
        if mode in {
            "medium-new",
            "small-new",
            "grown",
            "bad-measurement",
            "failed-confirm",
            "failed-row",
            "cached-row",
            "loaded-required",
            "loaded-required-under",
        }:
            with pytest.raises((ValueError, RuntimeError), match=r"test_size.cost"):
                decide([row], policy, confirm=confirm)
        else:
            result, debt = decide(
                [row],
                policy,
                confirm=confirm,
                judged=set() if mode in {"untouched", "cached-untouched"} else None,
            )
            assert result[0]["size"] == row["size"]
            if mode == "untouched":
                assert debt == {node: {"measured": 30, "cap": 37.5}}
            if mode == "loaded":
                assert result[0]["unconfirmed_load"] and not debt
            if mode == "cached-untouched":
                assert result[0]["cached"] and debt == policy["slow_baseline"]
            if mode == "clean-required-under":
                assert result[0].get("solo_seconds") == 9
        return
    if declared and declared[0].startswith("backend-"):
        from devtools.testing.sizes import validate_policy

        policy = json.loads((root / "config/registries/test-sizes.json").read_text("utf-8"))
        policy.setdefault(
            "unit_backends",
            [{"kind": "module", "name": "duckdb", "reason": "Database resource."}],
        )
        row = policy["unit_backends"][0]
        faults = {
            "backend-empty": [],
            "backend-kind": [row | {"kind": "native"}],
            "backend-type": [row | {"kind": ["module"]}],
            "backend-reason": [row | {"reason": ""}],
            "backend-duplicate": [row, row],
        }
        assert validate_policy(policy) is policy
        with pytest.raises(ValueError, match=r"testing\.layer_contract_unmeasured"):
            validate_policy(policy | {"unit_backends": faults[declared[0]]})
        return
    if declared and declared[0].startswith("framework-"):
        from devtools.testing.sizes import validate_policy

        policy = json.loads((root / "config/registries/test-sizes.json").read_text("utf-8"))
        policy.setdefault(
            "framework_fixtures",
            [
                {
                    "node": "tests/conftest.py::project_isolation",
                    "reason": "Confines the test invocation.",
                }
            ],
        )
        row = policy["framework_fixtures"][0]
        faults = {
            "framework-entry": [row["node"]],
            "framework-reason": [{"node": row["node"]}],
            "framework-pattern": [row | {"node": "tests/conftest.py::*"}],
            "framework-module": [row | {"node": "tests.conftest"}],
        }
        assert validate_policy(policy) is policy
        with pytest.raises(ValueError, match=r"test_size\.framework_fixtures_invalid"):
            validate_policy(policy | {"framework_fixtures": faults[declared[0]]})
        return
    lane = declared[0][5:] if declared and declared[0].startswith("lane_") else ""
    public = tmp_path / "public-only"
    policy_file = public / "config/registries/test-sizes.json"
    policy_file.parent.mkdir(parents=True)
    policy = json.loads((root / "config/registries/test-sizes.json").read_bytes())
    policy["limits"] = dict(small=1, medium=60, large=900)
    large = policy["large"]
    faults = {
        "bad-reason": (large[0], "reason", "slow", "large_reason"),
        "bad-limit": (policy["limits"], "medium", 0, "limits"),
        "duplicate": (policy, "large", large + large[:1], "large_duplicate"),
        "owner-large": (policy["defaults"][-1], "size", "large", "default_size"),
        "null-size-contract": (large[0], "size_contract", None, "large_contract"),
    }
    faults |= {
        "post-kind": (policy, "post_merge_only", "tests/x.py", "rule"),
        "post-structural": (policy, "post_merge_only", ["tests/structural/test_x.py"], "rule"),
        "post-repeat": (policy, "post_merge_only", ["tests/a.py", "tests/a.py"], "rule"),
    }
    case = declared[0] if declared else ""
    post_stage = case[5:] if case.startswith("post_") else ""
    if post_stage:
        lane = post_stage if post_stage in {"candidate", "nightly"} else "merge"
        contract = "tests/portfolio_strategy_lab/test_cli_contract.py"
        policy["mandatory"] = list(dict.fromkeys([*policy["mandatory"], contract + "*"]))
        policy["post_merge_only"] = [contract]
    if case in faults:
        record, key, value, refusal = faults[case]
        record[key] = value
    if case == "timeout":
        policy["limits"]["medium"] = 0.05
    policy_file.write_text(json.dumps(policy), encoding="utf-8")
    target = tmp_path / "test_example.py"
    target.write_text(
        "import pytest\n"
        + "".join(
            "@pytest.mark." + name + "\n"
            for name in declared
            if name in {"small", "medium", "large"}
        )
        + ("@pytest.mark.timeout(0.05)\n" if case == "strict" else "")
        + 'def test_example():\n    """Exercise the resolved execution limit."""\n'
        + ("    import time; time.sleep(0.2)\n" if case == "timeout" else "    pass\n"),
        encoding="utf-8",
    )
    registered = declared == ("registered",)
    if post_stage:
        target = tmp_path / contract
        target.parent.mkdir(parents=True, exist_ok=True)
    if lane:
        header = "import pytest\n"
        target.write_text(
            header + "@pytest.mark.small\ndef test_small(): pass\n"
            "def test_medium(): pass\n"
            "@pytest.mark.large\ndef test_large(): pass\n"
        )
        mandatory = tmp_path / "tests/structural/test_mandatory.py"
        mandatory.parent.mkdir(parents=True)
        mandatory.write_text(header + "@pytest.mark.large\ndef test_mandatory(): pass\n")
    if registered:
        target = str(root / "tests/portfolio_strategy_lab/test_strategy_activation.py")
        target += "::test_the_daily_update_resumes_its_deferral_then_goes_on_to_the_next_session"
    script = (
        "import pytest,json,sys\nfrom pathlib import Path\n"
        "from devtools.testing import sizes as test_sizes\nread=test_sizes.load_policy\n"
        f"test_sizes.load_policy=lambda _:read(Path({str(public)!r}))\n"
        "class Reader:\n"
        " def pytest_collection_finish(self,session):\n"
        "  rows=[]\n  for i in session.items:\n"
        "   m=i.get_closest_marker('timeout')\n"
        "   rows.append(dict(i.user_properties, timeout_marker=m.args[0] if m else None,\n"
        "     markers=sorted({m.name for m in i.iter_markers()\n"
        "       if m.name in {'small','medium','large'}})))\n"
        "  print('SIZE_ITEMS='+json.dumps(rows))\n"
        "raise SystemExit(pytest.main(sys.argv[1:],plugins=[Reader()]))\n"
    )
    if case == "sibling-race":
        sibling = tmp_path.parent / (tmp_path.name + "-sibling")
        sibling.mkdir()
        script = (
            "import os,threading\nfrom pathlib import Path\nimport _pytest.main\n"
            f"sibling=Path({str(sibling)!r})\n"
            f"owned=Path({str(tmp_path)!r})\n"
            "def move(): os.replace(sibling,sibling.with_name(sibling.name+'-moved'))\n"
            "samefile=_pytest.main.samefile_nofollow\n"
            "def move_then_stat(a,b):\n"
            " if a==sibling and sibling.exists():\n"
            "  move()\n"
            " return samefile(a,b)\n"
            "_pytest.main.samefile_nofollow=move_then_stat\n"
            "collect=_pytest.main.Session._collect_path\n"
            "def collect_owned(self,path,cache):\n"
            " if path==owned and sibling.exists():\n"
            "  other=threading.Thread(target=move); other.start(); other.join()\n"
            " return collect(self,path,cache)\n"
            "_pytest.main.Session._collect_path=collect_owned\n"
        ) + script
    command = [sys.executable, "-c", script]
    if not registered:
        command += [
            "-p",
            "tests.conftest",
            "-o",
            "markers=small\nmedium\nlarge",
            "--rootdir",
            str(tmp_path),
            "--confcutdir",
            str(tmp_path),
        ]
    command += ["-c", str(root / "pyproject.toml")]
    command += ["--basetemp", str(tmp_path / "child-temp")]
    command += ["-vv"] if case == "timeout" else ["--collect-only", "-q"]
    command += ["--no-cov", str(target)]
    if lane:
        command += ["--keep-duplicates", str(target), str(mandatory)]
    result = subprocess.run(
        command,
        cwd=root,
        env=dict(
            {
                key: value
                for key, value in os.environ.items()
                if key
                not in {
                    "ALPHALATTICE_PIPELINE_REQUIRED_TESTS",
                    "PYTEST_CURRENT_TEST",
                    "ALPHALATTICE_REACH_FLOOR",
                }
                if key != "ALPHALATTICE_CHANGED_PATHS"
                or not os.environ.get("ALPHALATTICE_REACH_FLOOR")
            },
            ALPHALATTICE_NETWORK_DISABLED="1",
            PYTHONPATH=str(root / "src"),
            ALPHALATTICE_TEST_LANE=lane,
            ALPHALATTICE_PIPELINE_STAGE=post_stage,
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=45,
        creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
    )
    if case == "sibling-race":
        assert not sibling.exists() and sibling.with_name(sibling.name + "-moved").is_dir()
    if case in faults:
        assert result.returncode != 0 and "test_size." + refusal in result.stderr
    elif case == "timeout":
        assert result.returncode != 0 and "Timeout" in result.stdout + result.stderr
        assert "test_example" in result.stdout + result.stderr
    elif len(set(declared)) > 1:
        assert result.returncode != 0 and "test_size.conflicting" in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        rows = json.loads(
            next(
                x.removeprefix("SIZE_ITEMS=")
                for x in result.stdout.splitlines()
                if x.startswith("SIZE_ITEMS=")
            )
        )
        assert rows
        if case == "strict":
            assert rows[0]["timeout_marker"] == 0.05
            return
        if post_stage:
            expected = (
                ["large"] if post_stage == "commit" else ["large", "large", "medium", "small"]
            )
            assert sorted(r["test_size"] for r in rows) == expected
            return
        if lane:
            expected = ["large", "medium", "small"] if lane == "merge" else ["large", "large"]
            assert sorted(r["test_size"] for r in rows) == expected
            assert sum("tests/structural/" in r["test_size_declaration"] for r in rows) == 1
            return
        expected_size = "large" if registered else declared[0] if declared else "medium"
        if case == "sibling-race":
            expected_size = "medium"
        expected_owner = "tests.portfolio_strategy_lab" if registered else "tests"
        for row in rows:
            assert row["markers"] == [row.get("test_size")]
            assert row.get("test_size") == expected_size
            assert row.get("test_size_owner") == expected_owner
            assert row.get("test_size_declaration")
            limit = 5400 if registered else policy["limits"][expected_size]
            assert row.get("test_timeout_seconds") == limit
        if not declared:
            from devtools.testing.sizes import classify, load_policy

            assert not (public / "tests").exists()
            policy = load_policy(public)
            policy["defaults"].reverse()
            scope = "tests/portfolio_strategy_lab"
            node = scope + "/test_public_control_surface.py"
            node += "::test_coverage_names_the_owner_a_request_escapes"
            expected = ("medium", scope.replace("/", "."), scope + "/*", 60)
            assert classify(node, (), policy) == expected


def test_a_declared_reach_scopes_the_merge_and_fast_lanes(tmp_path, monkeypatch, request):
    """A per_file scan runs scoped unless its own file or input changed; a cross-file check runs
    when its inputs change; an undeclared one leaves the commit stage unless a fence law keeps it
    or its own file changed, and post-merge and nightly run every one whole."""
    from devtools.testing.collection import CHANGED
    from devtools.testing.sizes import (
        changed_paths,
        floor_mode,
        included,
        load_policy,
        scoped,
        validate_policy,
    )

    lane = os.environ.get("ALPHALATTICE_TEST_LANE", "")
    scoped_run = lane in {"merge", "fast"}  # read once
    assert request.config.stash.get(CHANGED, None) == (
        changed_paths(floor=floor_mode(lane)) if scoped_run else None
    )
    scan, closure = "tests/structural/test_scan.py::test_a", "tests/structural/test_closure.py::t"
    reason = {"owner": "tests.structural", "reason": "declared"}
    policy = {
        "mandatory": ["tests/structural/*"],
        "reach": [
            {"pattern": scan, "per_file": True, "inputs": ["src/rules.py"], **reason},
            {"pattern": closure, "inputs": ["config/release/*", "+*"], **reason},
        ],
        "structural_fence": [
            {"pattern": "tests/structural/test_fenced.py*", "law": "identity", "reason": "kept"}
        ],
    }
    listing = tmp_path / "changed.log"
    listing.write_text("M\tsrc/a.py\nA\tsrc/new.py\n", encoding="utf-8")
    monkeypatch.setenv("ALPHALATTICE_CHANGED_PATHS", str(listing))
    changed = changed_paths()
    assert changed == {"src/a.py", "src/new.py", "+src/new.py"}
    assert scoped(scan, policy, changed) and included(scan, "medium", "fast", policy, changed)
    assert not scoped(scan, policy, changed | {"src/rules.py"})
    assert included(closure, "large", "merge", policy, changed)
    assert not included(closure, "large", "merge", policy, {"src/a.py"})
    assert not included(closure, "medium", "fast", policy, changed)
    other = "tests/structural/test_other.py::t"
    floor_change = frozenset({"src/session.py"})
    fenced = "tests/structural/test_fenced.py::t"
    policy["structural_fence"].append(dict(pattern=scan, law="identity", reason="kept"))
    assert scoped(scan, policy, floor_change)
    assert not included(closure, "medium", "merge", policy, floor_change)
    try:

        def choose(node, size="medium", changed=floor_change, rules=policy):
            return included(node, size, "merge", rules, changed, floor=True)

        floor_view = (
            *(choose(n) for n in (scan, closure, other, fenced)),
            choose(other, changed=floor_change | {other.split("::")[0]}),
            choose(closure, "large", rules=policy | {"mandatory": []}),
            choose(closure, "large"),
            scoped(scan, policy, floor_change, floor=True),
        )
    except TypeError:
        floor_view = None  # an older reader cannot honor the floor
    assert floor_view == (True, True, False, True, True, False, True, False)
    nightly_only = policy | {
        "large": [
            {"pattern": closure, "owner": "tests.structural", "size_contract": "nightly_only"}
        ]
    }
    assert not included(closure, "large", "merge", nightly_only, floor_change, floor=True)
    assert included(closure, "large", "nightly", nightly_only, None)
    from types import SimpleNamespace

    from devtools.testing.collection import _size_included

    item = SimpleNamespace(nodeid=closure)
    permitted = {closure.split("::")[0]}
    assert _size_included(
        item, "medium", "merge", policy, floor_change, (), (), permitted, floor=True
    )
    assert not _size_included(item, "medium", "merge", policy, floor_change, (), (), permitted)
    for damaged in ("", "M", "M\t", "R100\told\tnew", 'M\t"quoted"', "M\tbad\ufffd"):
        listing.write_text(damaged, encoding="utf-8")
        assert changed_paths(floor=True) is None
    listing.write_text("M\tsrc/a.py\nA\tsrc/new.py\n", encoding="utf-8")
    assert not included(other, "medium", "merge", policy, changed)  # moved: post-merge runs it
    assert included(other, "medium", "merge", policy, changed | {"tests/structural/test_other.py"})
    assert included("tests/structural/test_fenced.py::t", "medium", "merge", policy, changed)
    assert included(
        other, "medium", "merge", policy, None
    )  # post-merge, unscoped: whole, catching it
    from devtools.testing.collection import _record_moved, pytest_collection_finish

    config = SimpleNamespace(
        stash={},
        option=SimpleNamespace(xmlpath=str(tmp_path / "junit.xml")),
        pluginmanager=SimpleNamespace(get_plugin=lambda name: None),
    )
    nodes = [SimpleNamespace(nodeid=n) for n in (other, "tests/structural/test_fenced.py::t")]
    _record_moved(config, "merge", nodes, policy, changed)
    recorded = json.loads((tmp_path / "structural-moved.json").read_text(encoding="utf-8"))
    assert recorded == {"tier": "commit stage", "moved": [other], "ran": []}
    _record_moved(config, "merge", nodes, policy, changed, floor=True)
    pytest_collection_finish(SimpleNamespace(config=config, items=[nodes[0]]))
    recorded = json.loads((tmp_path / "structural-moved.json").read_text(encoding="utf-8"))
    assert recorded == {"tier": "commit stage", "moved": [other], "ran": [other]}
    _record_moved(config, "merge", nodes, policy, None, omitted=(nodes[0],))
    recorded = json.loads((tmp_path / "structural-moved.json").read_text(encoding="utf-8"))
    assert recorded == {"tier": "post-merge", "moved": [other], "ran": []}
    assert included(closure, "large", "nightly", policy, None) and not scoped(scan, policy, None)

    from copy import deepcopy

    registered = load_policy(Path(__file__).resolve().parents[2])
    for stem, changed in (
        ("test_entry_point_startup", "src/alphalattice/control/product_host/composition/entry.py"),
        ("test_local_web_build", "tests/pytest_environment.py"),
        ("test_pipeline_receipt", "src/devtools/pipeline/receipt.py"),
    ):
        assert included(
            f"tests/structural/{stem}.py::test_guard",
            "medium",
            "merge",
            registered,
            frozenset({changed}),
        ), changed
    node = "tests/structural/test_pipeline_receipt.py::test_guard"
    outside = "src/alphalattice/investment/risk_research/contracts.py"
    assert not included(node, "medium", "merge", registered, frozenset({outside})), outside
    web = (
        "tests/structural/test_local_web_build.py::test_check_reads_every_output_without_rebuilding"
    )
    assert included(
        web, "medium", "merge", registered, frozenset({"scripts/build_local_web_ui.py"})
    )
    assert not included(web, "medium", "merge", registered, frozenset({outside})), outside
    # These source and ledger identity/authority checks are fences even without declared reach.
    for node in (
        "tests/structural/test_task_record_collections.py::test_canonical_task_parsers_cannot_bypass_stored_identity_validation",
        "tests/structural/test_task_record_collections.py::test_every_whole_task_reader_keeps_its_reviewed_authority_requirement",
        "tests/structural/test_plan_retention.py::test_every_plan_is_sealed_on_disk_until_it_expires",
        "tests/structural/test_answer_continuations.py::test_every_compact_locator_restores_by_the_same_rule",
        "tests/structural/test_owner_source_shape.py::test_publication_refuses_rows_from_another_catalog_before_composing",
        "tests/structural/test_documented_commands.py::test_kept_facts_follow_the_bytes_read_and_serve_only_sealed_rows",
        "tests/structural/test_owner_source_shape.py::test_publication_refuses_to_return_before_the_index_resolves",
    ):
        assert included(node, "medium", "merge", registered, frozenset({"docs/x.md"})), node
    # every registered reach: any one declared input still selects its guard, an unrelated change
    # leaves it to the post-merge tier, which runs it whole ("+x": an added path)
    for row in (r for r in registered["reach"] if r.get("inputs") and not r.get("per_file")):
        node = row["pattern"].replace("*", "x")
        for seeded in (p.replace("*", "x") for p in row["inputs"]):
            assert included(node, "medium", "merge", registered, frozenset({seeded})), seeded
        assert not included(node, "medium", "merge", registered, frozenset({"docs/x.md"})), node
    for row in registered["structural_fence"]:  # a fence law keeps its guard in every stage
        node = row["pattern"].replace("*", "x")
        assert included(node, "medium", "merge", registered, frozenset({"docs/x.md"})), node
    unlawful = deepcopy(registered)
    unlawful["structural_fence"][0]["law"] = "convenience"
    with pytest.raises(ValueError, match=r"test_size.fence_invalid"):
        validate_policy(unlawful)
    legacy = deepcopy(registered) | {"reach": [policy["reach"][0]]}
    assert validate_policy(legacy) is legacy
    path = "tests/integration/architecture/test_scan.py"
    moved = path + "::test_a"
    default = {"pattern": path + "*", "owner": "tests.structural", "size": "medium"}
    mapped = deepcopy(registered) | {
        "defaults": [*registered["defaults"], default],
        "reach": [policy["reach"][0] | {"pattern": moved}],
    }
    refused = None
    try:
        validate_policy(mapped)
    except ValueError as error:
        refused = str(error)
    assert refused is None, refused
    assert scoped(moved, mapped, changed) == scoped(scan, legacy, changed)
    disjoint = {
        "pattern": "tests/integration/architecture/*test_different_function.py::*",
        "owner": "tests.other",
        "size": "medium",
    }
    refused = None
    try:
        validate_policy(mapped | {"defaults": [*mapped["defaults"], disjoint]})
    except ValueError as error:
        refused = str(error)
    assert refused is None, refused
    unmapped = "tests/integration/architecture/test_unmapped.py::test_a"
    for altered in (
        mapped | {"defaults": registered["defaults"]},
        mapped | {"defaults": [*registered["defaults"], default | {"owner": "tests.other"}]},
        mapped
        | {
            "defaults": [
                *registered["defaults"],
                default,
                {"pattern": moved, "owner": "tests.other", "size": "medium"},
            ]
        },
        mapped
        | {
            "defaults": [
                *registered["defaults"],
                default,
                {"pattern": path + "::test_*", "owner": "tests.other", "size": "medium"},
            ]
        },
        mapped
        | {
            "defaults": [
                *registered["defaults"],
                default | {"pattern": "tests/integration/architecture/*"},
            ]
        },
        mapped
        | {"defaults": [*registered["defaults"], default, default | {"owner": "tests.other"}]},
        mapped | {"reach": [mapped["reach"][0] | {"pattern": unmapped}]},
        mapped | {"reach": [mapped["reach"][0] | {"owner": "tests.other"}]},
        mapped | {"reach": [mapped["reach"][0] | {"reason": ""}]},
        mapped | {"reach": [{"pattern": moved, "reason": "declared"}]},
    ):
        with pytest.raises(ValueError, match=r"test_size\.reach_invalid") as caught:
            validate_policy(altered)
        assert str(caught.value) == "test_size.reach_invalid:" + altered["reach"][0]["pattern"]
    with pytest.raises(ValueError, match=r"test_size\.rule"):
        validate_policy(mapped | {"post_merge_only": [path]})

    # Registered Large standing survives only the layer's inferred Medium.
    from devtools.testing.collection import _proven

    node = "tests/integration/example/test_carried.py::test_case"
    carried = load_policy(Path(__file__).resolve().parents[2])
    carried["defaults"].append({"pattern": node, "owner": "tests", "size": "medium"})
    large = {
        "pattern": node,
        "owner": "tests",
        "reason": "carried long integration",
        "timeout_seconds": 5400,
    }
    carried["large"].append(large)
    item = SimpleNamespace(nodeid=node, iter_markers=lambda name=None: ())
    error = None
    try:
        classified = _proven(item, {}, carried)[3]
    except ValueError as caught:
        error = str(caught)
    assert error is None, error
    assert classified[:2] == ("large", "tests") and classified[3] == 5400
    carried["large"].remove(large)
    assert _proven(item, {}, carried)[3][0] == "medium"
    carried["large"].append(large)
    item.iter_markers = lambda name=None: (pytest.mark.medium.mark,) if name is None else ()
    with pytest.raises(ValueError, match=r"test_size\.conflicting"):
        _proven(item, {}, carried)


@pytest.mark.parametrize(
    "case",
    ["date", "legacy", "first", "renewed", "twice", "bool", "string", "no-reason", "multiline"],
)
def test_a_quarantined_node_needs_its_owner_reason_and_date(case):
    """A flaky node leaves the blocking tier only with an owner, a reason and an until date."""
    from devtools.testing.sizes import load_policy, validate_policy

    row = {"node": "tests/a.py::t", "owner": "tests.a", "reason": "flaky", "until": "2026-11-03"}
    if case == "date":
        row["until"] = "2026-99-99"
    if case not in {"date", "legacy"}:
        row["renewals"] = {
            "first": 0,
            "renewed": 1,
            "twice": 2,
            "bool": True,
            "string": "1",
            "no-reason": 1,
            "multiline": 1,
        }[case]
    if case == "no-reason":
        row["reason"] = ""
    if case == "multiline":
        row["reason"] = "fix in flight\nmore"
    policy = load_policy(Path(__file__).resolve().parents[2]) | {"quarantine": [row]}
    if case in {"date", "twice", "bool", "string", "no-reason", "multiline"}:
        with pytest.raises(ValueError, match="quarantine_invalid"):
            validate_policy(policy)
    else:
        assert validate_policy(policy)["quarantine"] == [row]


HELD = "product_host.detached_start_outlives_the_session"


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


@pytest.mark.proves("testing.find_matches_requirements_and_unregistered_tests")
def test_find_returns_requirements_with_holders_then_unregistered_tests(tmp_path, capsys):
    from scripts.tests_registry import main

    from devtools.testing.requirements import find, load, matches

    registry = _registry_tree(tmp_path)
    found = find(tmp_path, "detached host survives the session")
    assert found[0].startswith("1 requirements and 1") and "active" in found[1]
    assert "  held by tests/a/test_host.py::test_breakaway_launch_survives" in found
    assert "unregistered  tests/a/test_host.py::test_idle_expiry_keeps_owed_work" in found
    assert not any("parquet" in line for line in found)
    assert find(tmp_path, "detached host keeps work")[1].startswith(HELD)  # rows lead
    assert main(["--root", str(tmp_path), "find", "dividend ledger"]) == 0
    assert capsys.readouterr().out.startswith("Nothing matches") and matches("x", {}) == []
    registries, node = tmp_path / "config/registries", "tests/a/test_host.py::test_breakaway_launch"
    policy = json.loads((registries / "test-sizes.json").read_bytes())
    for until, state in (("2000-01-01", "active"), ("2999-01-01", "quarantined")):
        held = {"node": node + "_survives", "owner": "tests", "reason": "r", "until": until}
        (registries / "test-sizes.json").write_text(json.dumps(policy | {"quarantine": [held]}))
        assert state in find(tmp_path, "detached host survives the session")[1]
    bad = {HELD: registry["entries"][HELD] | {"layer": "system"}}
    for document in ([], registry | {"entries": []}, registry | {"entries": bad}):
        (registries / "requirements.json").write_text(json.dumps(document), "utf-8")
        with pytest.raises(ValueError, match=r"requirements\.(schema|row_invalid:product_host\.)"):
            load(tmp_path)


@pytest.mark.proves(
    "testing.unregistered_test_refused_with_its_find",
    "testing.row_admitted_or_refused_by_its_check",
)
@pytest.mark.parametrize(
    ("case", "edit", "refused"),
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
        (HELD, None, "was removed"),
        ("schema-unreadable-directory", {}, ""),
        ("schema-unreadable-entry", {}, ""),
        ("entry-order", {}, ""),
    ],
)
def test_a_new_test_or_row_is_admitted_or_refused_by_the_check_it_fails(
    tmp_path, case, edit, refused, monkeypatch
):
    from devtools.testing import requirements as r
    from devtools.testing.collection import gate

    if case == "entry-order":
        from scripts.tests_registry import main

        registry = _registry_tree(tmp_path)
        row = registry["entries"][HELD]
        registry["entries"] = {"testing.z": row, "testing.a": row}
        before = json.dumps(registry)
        path = tmp_path / r.REGISTRY
        path.write_text(before, "utf-8")
        try:
            r.load(tmp_path)
        except ValueError as error:
            assert "requirements.entries_out_of_order" in str(error)
            assert "python scripts/tests_registry.py sort" in str(error)
        else:
            raise AssertionError("an unsorted requirements registry must be refused")
        assert main(["--root", str(tmp_path), "sort"]) == 0
        ordered = path.read_text("utf-8")
        assert list(r.load(tmp_path)) == ["testing.a", "testing.z"]
        assert json.loads(ordered)["entries"] == json.loads(before)["entries"]
        # The historical base may be unsorted.
        assert r.guard(tmp_path, [], ordered, before, tests=[]) == []
        assert main(["--root", str(tmp_path), "sort"]) == 0 and path.read_text("utf-8") == ordered
        path.write_text(before, "utf-8")
        before_bytes = path.read_bytes()
        write = Path.write_text

        def full_disk(target, text, *args, **kwargs):
            if target == path or target.name.startswith(".requirements.json."):
                with target.open("w", encoding="utf-8") as stream:
                    stream.write(text[:16])
                raise OSError("fixture disk full")
            return write(target, text, *args, **kwargs)

        with monkeypatch.context() as fault:
            fault.setattr(Path, "write_text", full_disk)
            assert main(["--root", str(tmp_path), "sort"]) == 1
        assert path.read_bytes() == before_bytes
        assert not list(path.parent.glob(".requirements.json.*"))
        with monkeypatch.context() as fault:
            fault.setattr(
                os, "replace", lambda *a: (_ for _ in ()).throw(OSError("fixture replace denied"))
            )
            assert main(["--root", str(tmp_path), "sort"]) == 1
        assert path.read_bytes() == before_bytes and not list(
            path.parent.glob(".requirements.json.*")
        )
        path.write_text('{"entries": {"x": 1, "x": 2}}', "utf-8")
        broken = path.read_bytes()
        assert main(["--root", str(tmp_path), "sort"]) == 1 and path.read_bytes() == broken
        return

    if case in {"schema-unreadable-directory", "schema-unreadable-entry"}:
        root = tmp_path / "schema-hidden"
        hidden = root / "config" / "restricted"
        hidden.mkdir(parents=True)
        (hidden / "manifest.json").write_text(json.dumps({"schema": r.FACTS_SCHEMA}), "utf-8")
        with pytest.raises(ValueError, match="legacy_missing"):
            r.legacy(root)
        scandir = os.scandir

        def unreadable(path):
            if case == "schema-unreadable-entry" and Path(path) == root / "config":
                entry = MagicMock(name="entry")
                entry.name, entry.path = "restricted", str(hidden)
                entry.is_dir.side_effect = PermissionError("fixture entry unavailable")
                scan = MagicMock(name="scandir")
                scan.__iter__.return_value = iter([entry])
                scan.__enter__.return_value = scan
                return scan
            if case == "schema-unreadable-directory" and Path(path) == hidden:
                raise PermissionError("fixture directory unavailable")
            return scandir(path)

        with monkeypatch.context() as fault:
            fault.setattr(os, "scandir", unreadable)
            try:
                r.legacy(root)
            except ValueError as error:
                assert "schema_candidate_unreadable" in str(error)
            else:
                raise AssertionError("unreadable schema population must not prove public absence")
        not_dir = tmp_path / "schema-file"
        not_dir.mkdir()
        (not_dir / "config").write_text("not a directory", "utf-8")
        with pytest.raises(ValueError, match="schema_candidate_unreadable"):
            r.legacy(not_dir)
        return

    if case == "schema-preserved-time":
        root = tmp_path / "schema"
        (root / "config").mkdir(parents=True)
        candidate = root / "config/one.json"
        schema = r.LEGACY_SCHEMA
        candidate.write_text(json.dumps({"schema": schema}), "utf-8")
        assert r.schema_path(root, schema) == candidate
        stat = candidate.stat()
        raw = candidate.read_bytes()
        altered = raw.replace(b"tests-unregistered", b"tests-UNregistered")
        assert altered != raw and len(altered) == len(raw)
        candidate.write_bytes(altered)
        os.utime(candidate, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert r.schema_path(root, schema) is None
        candidate.write_bytes(raw)
        os.utime(candidate, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert r.schema_path(root, schema) == candidate
        return
    if case == "seed-junction":
        (tmp_path / "config").mkdir()
        monkeypatch.setattr(Path, "is_junction", lambda path: path == tmp_path / "config")
        with pytest.raises(ValueError, match="legacy_seed_target_invalid"):
            r.shrink(tmp_path, seed=True, to=Path("config/seed.json"))
        assert not (tmp_path / "config/seed.json").exists()
        return
    registry, node = _registry_tree(tmp_path), "tests/a/test_host.py::test_host_outlives_session"
    if case in {"unregistered", "unknown"}:
        test = r.Holder(node, ("x.y",) * (case == "unknown"), node.split("::")[1])
        found = r.registration(tmp_path, [test], registry["entries"])
        assert refused in found[0] and found[-1].lstrip().startswith("Next:")
        assert any(line.strip().startswith(HELD) for line in found)
        if case == "unknown":
            return
        kept, other = (r.Holder(f"t.py::{c}::test_anchor", (), "", "d") for c in ("TestA", "TestB"))
        assert r.fresh([kept], [kept, other]) == [other]  # a basename never matches
        moved, copy = (kept._replace(node=f"{f}::TestA::test_anchor") for f in ("u.py", "v.py"))
        assert r.fresh([kept], [moved]) == [] and r.fresh([kept], [moved, copy]) == [copy]
        suite, outside = tmp_path / "tests", [(node, [], tmp_path / "child/test_host.py")]
        gate("merge", outside, {}, suite)  # a child session's own tests are not the suite's
        with pytest.raises(ValueError, match=r"unregistered:tests/a/test_host\.py"):
            gate("merge", [(node, [], suite / "a/test_host.py")], {}, suite)
        gate("merge", [(node, [], suite / "a/test_host.py")], r.legacy(tmp_path), suite)  # public
        ledger = Path("config/registries/ledger.json")  # private records are found by schema
        r.shrink(tmp_path, seed=True, to=ledger)
        with pytest.raises(ValueError, match="legacy_seeded"):  # seeded once, then only shrinks
            r.shrink(tmp_path, seed=True, to=ledger)
        (tmp_path / ledger).unlink()
        listing = tmp_path / "config/release/listing.json"
        listing.parent.mkdir(parents=True)
        listing.write_text(json.dumps({"schema": r.FACTS_SCHEMA}), "utf-8")
        with pytest.raises(ValueError, match="legacy_missing"):  # a private checkout needs it
            r.legacy(tmp_path)
        outside = (Path("../out.json"), tmp_path / "config/x.json", Path("tests/x.json"))
        for bad in (listing.relative_to(tmp_path), *outside):  # a new file inside config only
            with pytest.raises(ValueError, match="legacy_seed_target_invalid"):
                r.shrink(tmp_path, seed=True, to=bad)
        assert not (tmp_path.parent / "out.json").exists()
        torn = tmp_path / "config/release/torn.json"
        torn.write_text("{", "utf-8")
        with pytest.raises(ValueError, match="schema_candidate_unreadable:config/release/torn"):
            r.legacy(tmp_path)  # an unreadable candidate is unknown, never a public tree
        torn.unlink()
        r.shrink(tmp_path, seed=True, to=ledger)
        assert r.legacy(tmp_path) is not None
        twin = tmp_path / "config/twin.json"
        twin.write_bytes((tmp_path / ledger).read_bytes())
        with pytest.raises(ValueError, match="schema_ambiguous"):  # the memo sees a new carrier
            r.legacy(tmp_path)
        twin.unlink()
        (tmp_path / ledger).write_text(json.dumps({"schema": "other.v1"}), "utf-8")
        with pytest.raises(ValueError, match="legacy_missing"):  # and never keeps a changed file
            r.legacy(tmp_path)
        return
    base, entries = json.dumps(registry), registry["entries"]
    row = entries[HELD] | {"description": "A restarted Host reuses its last free port."}
    decided = {"exception": {"kinds": ["journey"], "why": "w", "decided": "2026-10-24: w"}}
    if edit is None:
        del entries[case]
    else:
        entries[case] = row | edit | (decided if edit.get("exception") == "decided" else {})
    found = r.admission(tmp_path, entries, json.loads(base)["entries"])
    assert (refused in " ".join(found)) if refused else found == []


@pytest.mark.proves("testing.one_holder_per_layer")
@pytest.mark.parametrize(
    "case",
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
    ],
)
def test_a_new_holder_is_unique_within_its_requirement_layer(tmp_path, monkeypatch, case):
    from devtools.testing import requirements as r

    key = "testing.example"
    rows = {
        key: {
            "description": "One recorded expectation has one local owner.",
            "owner": "devtools.testing",
            "layer": "unit",
            "seconds": 1,
        }
    }
    policy = tmp_path / "config/registries/test-sizes.json"
    policy.parent.mkdir(parents=True)
    policy.write_bytes(
        (Path(__file__).resolve().parents[2] / r.REGISTRY.parent / policy.name).read_bytes()
    )
    source = """import pytest
@pytest.mark.proves('testing.example')
def test_anchor(): assert True
"""
    anchor = tmp_path / "tests/unit/test_first.py"
    anchor.parent.mkdir(parents=True)
    anchor.write_text(source)
    before = r.file_holders("tests/unit/test_first.py", source)
    if case in {"legacy", "split-file", "whole-file-move"}:
        source += """
@pytest.mark.proves('testing.example')
def test_old(): assert True
"""
        anchor.write_text(source)
        before = r.file_holders("tests/unit/test_first.py", source)
    if case == "split-file":
        anchor.write_text(
            source[source.index("import pytest") :].replace(
                "@pytest.mark.proves('testing.example')\ndef test_anchor(): assert True\n", ""
            )
        )
        (tmp_path / "tests/unit/test_split.py").write_text(
            """import pytest
@pytest.mark.proves('testing.example')
def test_anchor(): assert True
"""
        )
    if case in {"legacy-files", "legacy-extension", "legacy-renamed-function", "legacy-extra-file"}:
        previous = tmp_path / "tests/unit/test_legacy.py"
        previous.write_text(source.replace("test_anchor", "test_old_file"))
        before += r.file_holders("tests/unit/test_legacy.py", previous.read_text(encoding="utf-8"))
    if case in {"same-file", "legacy-extension"}:
        anchor.write_text(
            source
            + """
@pytest.mark.proves('testing.example')
def test_focused(): assert True
"""
        )
    if case == "legacy-renamed-function":
        anchor.write_text(source.replace("test_anchor", "test_renamed"))
    if case == "legacy-extra-file":
        (tmp_path / "tests/unit/test_third.py").write_text(
            source.replace("test_anchor", "test_third")
        )
    if case == "same-file-class":
        anchor.write_text(
            source
            + """
@pytest.mark.proves('testing.example')
class TestFocused:
    def test_one(self): assert True
    def test_two(self): assert True
"""
        )
    if case == "parameter":
        anchor.write_text(
            source.replace(
                "def test_anchor()", "@pytest.mark.parametrize('x', [1, 2])\ndef test_anchor(x)"
            )
        )
    elif case not in {
        "legacy",
        "unchanged",
        "new-marker",
        "same-file",
        "same-file-class",
        "legacy-files",
        "legacy-extension",
        "legacy-renamed-function",
        "legacy-extra-file",
        "split-file",
    }:
        other = tmp_path / (
            "tests/integration/test_second.py"
            if case == "different-layer"
            else "tests/unit/test_second.py"
        )
        other.parent.mkdir(parents=True, exist_ok=True)
        other.write_text(
            source
            if case
            in {"move", "whole-file-move", "edited-copy", "edited-move", "edited-trivia-move"}
            else source.replace("test_anchor", "test_added")
        )
        if case in {"move", "whole-file-move", "edited-move", "edited-trivia-move"}:
            anchor.unlink()
        if case == "edited-trivia-move":
            other.write_text(source.replace("assert True", "assert True  # changed definition"))
        if case in {"edited-copy", "edited-move"}:
            other.write_text(other.read_text().replace("assert True", "assert 1 == 1"))
    if case == "inherited-proof":
        rows["testing.other"] = rows[key]
        inherited = (
            "import pytest\npytestmark = pytest.mark.proves('testing.other')\n"
            "def test_moved(): assert True\n"
        )
        before += r.file_holders("tests/unit/test_removed.py", inherited)
        other.write_text(inherited.replace("testing.other", key))
    if case == "marker-addition":
        rows["testing.other"] = rows[key]
        previous = source.replace("testing.example", "testing.other").replace(
            "test_anchor", "test_added"
        )
        before += r.file_holders("tests/unit/test_second.py", previous)
        other.write_text(
            previous.replace(
                "proves('testing.other')", "proves('testing.other', 'testing.example')"
            )
        )
    if case == "new-marker":
        before = r.file_holders(
            "tests/unit/test_first.py",
            source.replace("@pytest.mark.proves('testing.example')\n", ""),
        )
    tests = r.holders(tmp_path)
    added = r.fresh(before, tests)
    calls = []
    read = r.holders
    monkeypatch.setattr(r, "holders", lambda root: calls.append(root) or read(root))
    document = json.dumps(
        {"schema": "alphalattice.requirements.v1", "owner": "devtools.testing", "entries": rows}
    )
    found = r.guard(tmp_path, added, document, document, tests=tests, before_tests=before)
    refused = case in {
        "different-layer",
        "split-file",
        "second",
        "edited-copy",
        "read-once",
        "inherited-proof",
        "marker-addition",
        "legacy-extra-file",
    }
    assert bool(found) == refused
    if case == "different-layer":
        assert key in " ".join(found) and "testing.layer_contract" in " ".join(found)
        assert "register its true layer" in " ".join(found)
        return
    if refused:
        assert (
            key in " ".join(found)
            and "tests/unit/test_first.py" in " ".join(found)
            and "extend" in " ".join(found)
        )
    if case in {"same-file", "same-file-class"}:
        found_cases = r.find(tmp_path, "recorded expectation", r.index(tmp_path, rows))
        assert sum(line.startswith("  held by ") for line in found_cases) == 1
        assert any(line.startswith("    case ") for line in found_cases)
    if case == "read-once":
        assert not calls
    if case == "unchanged":
        assert not calls
    if case == "legacy":
        assert "test_old" in " ".join(
            r.find(tmp_path, "recorded expectation", r.index(tmp_path, rows))
        )
    if case in {"edited-move", "edited-trivia-move", "new-marker", "marker-addition"}:
        assert len(added) == 1
    if case in {"new-marker", "marker-addition"}:
        assert added[0].ids == (key,)
