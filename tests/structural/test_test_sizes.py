"""Collection resolves one owned size and refuses contradictory declarations."""

import ast
import json
import os
import subprocess
import sys
import textwrap
from contextlib import suppress
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.structural.pytest_support import child_environment, descriptor_provenance
from tests.structural.size_fixture_support import (
    _CASES,
    ADMISSION_CASES,
    DECLARED_CASES,
    HELD,
    HOLDER_CASES,
    LAYER_CASES,
    LAYER_IDS,
    _registry_tree,
)


@pytest.mark.proves("testing.layer_contract")
@pytest.mark.parametrize(
    "case,layer,header,body,refused",
    LAYER_CASES,
    ids=LAYER_IDS,
)
def test_explicit_layers_refuse_resources_and_keep_their_execution_bound(
    tmp_path, monkeypatch, case, layer, header, body, refused
):
    if case == "network-caught":
        import socket

        from devtools.testing.contracts import LayerContract

        guard = LayerContract("unit", tmp_path, ({"kind": "module", "name": "sqlite3"},))
        original = socket.socket, socket.getaddrinfo, socket.gethostbyname
        with guard.active():
            for call, args in (
                (socket.socket, ()),
                (socket.getaddrinfo, ("localhost", 80)),
                (socket.gethostbyname, ("localhost",)),
            ):
                with pytest.raises(AssertionError, match=r"testing\.layer_contract"):
                    call(*args)
        assert (socket.socket, socket.getaddrinfo, socket.gethostbyname) == original
        return
    if case == "fd-provenance":
        descriptor_provenance(tmp_path)
        return
    if case in {"temporary", "root-pure"}:
        monkeypatch.setenv(
            "ALPHALATTICE_PIPELINE_REQUIRED_TESTS", json.dumps(["tests/outer.py::test_outer"])
        )
    root = Path(__file__).resolve().parents[2]
    if case.startswith("root-"):
        from tests.structural.pytest_support import pytest_checkout

        checkout, environment = pytest_checkout(tmp_path, root)
        marker = Path(environment["TEMP"]) / "given" / "framework-entered"
        if case in {"root-session-golden", "root-session-golden-error"}:
            (checkout / "tests/integration/test_golden_probe.py").write_text(
                "from devtools.evaluation.golden_workspaces import hold_generation\n"
                "def test_hold(golden_cache):\n"
                "    generation = golden_cache / 'generation'\n"
                "    generation.mkdir()\n"
                + (
                    "    def fail(): raise AssertionError('release-test')\n"
                    if case.endswith("-error")
                    else "    def fail(): pass\n"
                )
                + "    assert hold_generation(generation, fail)\n",
                encoding="utf-8",
            )
        if case in {"root-session-seed", "root-session-seed-error"}:
            (checkout / "tests/integration/test_golden_probe.py").write_text(
                "import sys, types\n"
                "from pathlib import Path\n"
                "def test_seed():\n"
                "    module = types.ModuleType('fixture_seed')\n"
                "    def changed():\n"
                "        Path(__file__).read_text()\n"
                + (
                    "        return ['changed seed']\n"
                    if case.endswith("-error")
                    else "        return []\n"
                )
                + "    module.seeds_changed = changed\n"
                "    sys.modules['tests.researcher_methodology_surface."
                "session_workspace'] = module\n",
                encoding="utf-8",
            )
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
                *(["-p", "xdist.plugin", "-n", "1"] if case.endswith("-error") else []),
                "--basetemp",
                str(marker.parent),
                *(
                    [str(checkout / "tests" / "integration" / "test_golden_probe.py")]
                    if case.startswith(("root-session-golden", "root-session-seed"))
                    else []
                ),
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
            expected = (
                "session fixture cleanup failed: AssertionError: release-test"
                if case == "root-session-golden-error"
                else "session fixture cleanup failed: AssertionError"
                if case == "root-session-seed-error"
                else "testing.layer_contract"
            )
            assert expected in output
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
            "import socket\nsocket.getaddrinfo('example.invalid', 80)\n", "utf-8"
        )
    target.write_text(
        header
        + (
            "def test_sample(tmp_path, subtests):\n"
            if case == "own-subtests"
            else "def test_sample(tmp_path_factory):\n"
            if case in {"size-live-factory", "size-live-factory-base"}
            else "def test_sample(tmp_path, tmp_path_factory):\n"
            if case in {"size-live-factory-mixed", "size-live-factory-under"}
            else "def test_sample(tail):\n"
            if case == "size-live-factory-teardown"
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
                + (
                    f"collection._size_families = lambda config: ({node!r},)\n"
                    if case == "size-live-large-missing"
                    else ""
                )
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
    if case.startswith("own-") or case == "instruction-over":
        call_seconds = {
            "own-bad": "float('nan')",
            "own-over": "0.7",
            "instruction-over": "0.7",
            "own-boundary": "0.6",
        }.get(case, "0.4")
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
    if case.startswith("instruction-"):
        mode = "walltime" if case == "instruction-walltime" else "simulation"
        (tmp_path / "codspeed_probe.py").write_text(
            "from types import SimpleNamespace\n"
            "def pytest_addoption(parser): parser.addoption('--codspeed', action='store_true')\n"
            "def pytest_configure(config):\n"
            " config.addinivalue_line('markers', 'benchmark: simulated CodSpeed')\n"
            " config.pluginmanager.register(SimpleNamespace(is_codspeed_enabled=True, "
            f"mode=SimpleNamespace(value={mode!r})), 'codspeed_plugin')\n",
            encoding="utf-8",
        )
        phase_args += ["-p", "codspeed_probe", "--codspeed"]
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
        env=child_environment(
            PYTHONPATH=os.pathsep.join([str(tmp_path), str(root / "src")]),
            ALPHALATTICE_TEST_LANE="merge"
            if case in {"size-live-large-merge", "size-live-large-missing", "instruction-gate"}
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
            else "post-merge"
            if case == "size-live-large-missing"
            else "commit"
            if case.startswith("size-live-")
            else "",
            ALPHALATTICE_PIPELINE_EVENTS=(
                str(tmp_path / "test-events.jsonl")
                if case.startswith("observe") or case.startswith("size-live-")
                else ""
            ),
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            ALPHALATTICE_CODSPEED_NO_TIME_BOUNDS="1" if case.startswith("instruction-") else "",
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
        assert result.returncode == 5 and "1 deselected" in output, output
        assert "NIGHTLY_EXECUTED" not in output
        return
    if case == "size-live-large-nightly":
        assert result.returncode == 0 and "NIGHTLY_EXECUTED" in output, output
        return
    if refused:
        expected = (
            "test_size.admission_limit"
            if case
            in {
                "size-live-over",
                "size-live-factory",
                "size-live-factory-base",
                "size-live-factory-mixed",
                "size-live-factory-teardown",
                "size-live-xdist",
                "size-live-time-over",
            }
            else "test_size.admission_unmeasured"
            if case == "size-live-skip"
            else "Timeout"
            if case in {"slow", "instruction-gate", "instruction-walltime"}
            else "assert False"
            if case == "instruction-failure"
            else "testing.layer_contract"
        )
        assert result.returncode == 1 and expected in output, output
        if case in {
            "size-live-over",
            "size-live-factory",
            "size-live-factory-base",
            "size-live-factory-mixed",
            "size-live-factory-teardown",
            "size-live-xdist",
        }:
            assert "10000001" in output and "medium" in output
    else:
        assert result.returncode == 0, output
    if case in {"instruction-slow", "instruction-over"}:
        from xml.etree import ElementTree as ET

        case_xml = next(ET.parse(tmp_path / "child.xml").iter("testcase"))
        properties = {row.get("name"): row.get("value") for row in case_xml.iter("property")}
        assert float(properties["test_timeout_seconds"]) == 1
        assert properties["test_timeout_mode"] == "off:codspeed-instruction-simulation"
    if case in {
        "size-live-over",
        "size-live-factory",
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
    [
        *DECLARED_CASES,
        pytest.param(("required-file",), id="required-file"),
        pytest.param(("commit-tier",), id="commit-tier"),
    ],
)
@pytest.mark.proves("testing.layer_contract")
def test_collection_resolves_owned_size_or_names_the_conflict(tmp_path, monkeypatch, declared):
    """Owned size declarations enforce their bounds and provenance, refusing invalid catalogs."""
    if declared == ("required-file",):
        from types import SimpleNamespace

        from devtools.testing import collection

        nodes = [f"tests/test_wide.py::test_case_{i}" for i in range(1400)]
        path = tmp_path / "required.json"
        data = json.dumps(nodes)
        assert len(data) > 32767
        path.write_text(data, encoding="utf-8")
        monkeypatch.delenv("ALPHALATTICE_PIPELINE_EVENTS", raising=False)
        monkeypatch.delenv("WK_SUITE_PLUGIN", raising=False)
        monkeypatch.setenv("ALPHALATTICE_PIPELINE_REQUIRED_TESTS", "@" + str(path))
        config = SimpleNamespace(option=SimpleNamespace(deselect=["tests/test_wide.py"]), stash={})
        with suppress(pytest.UsageError):
            collection.pytest_configure(config)
        assert config.stash.get(collection.REQUIRED) == (tuple(nodes), ("tests/test_wide.py",))
        assert config.option.deselect == []
        path.unlink()
        with pytest.raises(pytest.UsageError, match="required_selection_unavailable") as missing:
            collection.pytest_configure(config)
        assert str(path) in str(missing.value)
        path.write_bytes(b"\xff")
        with pytest.raises(pytest.UsageError, match="required_selection_unavailable"):
            collection.pytest_configure(config)
        path.write_text("{", encoding="utf-8")
        with pytest.raises(pytest.UsageError, match="required_selection_invalid"):
            collection.pytest_configure(config)
        monkeypatch.setenv("ALPHALATTICE_PIPELINE_REQUIRED_TESTS", json.dumps(nodes[:1]))
        collection.pytest_configure(config)
        assert config.stash[collection.REQUIRED][0] == tuple(nodes[:1])
        return
    if declared == ("commit-tier",):
        from devtools.testing import collection

        _stage_admits = getattr(collection, "_stage_admits", None)
        assert callable(_stage_admits), "commit needs a selection boundary for slow tests"
        for layer, size, expected in (
            ("unit", "small", True),
            ("integration", "medium", True),
            ("integration", "large", False),
            ("e2e", "medium", False),
            ("native", "medium", False),
        ):
            assert _stage_admits("commit", size, layer) is expected
        assert _stage_admits("nightly", "large", "e2e")
        assert _stage_admits("post-merge", "large", "integration")
        return
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
            commented = old.replace("def test_same(): pass", "def test_same(): pass  # explanation")
            assert (
                requirements.edited_families(
                    requirements.file_holders("tests/test_a.py", old, size_scope=True),
                    requirements.file_holders("tests/test_a.py", commented, size_scope=True),
                )
                == []
            )
            assert requirements.edited_families(
                requirements.file_holders("tests/test_a.py", old, size_scope=True),
                requirements.file_holders(
                    "tests/test_a.py",
                    old.replace("def test_same(): pass", "def test_same(): assert True"),
                    size_scope=True,
                ),
            ) == ["tests/test_a.py::test_same"]
            literal = (
                "def test_literal():\n    value = \"@pytest.mark.proves('old')\"\n"
                "    assert value\n"
            )
            assert requirements.edited_families(
                requirements.file_holders("tests/test_a.py", literal, size_scope=True),
                requirements.file_holders(
                    "tests/test_a.py", literal.replace("old", "new"), size_scope=True
                ),
            ) == ["tests/test_a.py::test_literal"]
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

            def commit(message):
                git("add", ".")
                git(
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "commit",
                    "-qm",
                    message,
                )
                return git("rev-parse", "HEAD")

            # an edited move with no row in the move's table is new; with one, the same test
            (repo / "tests/test_a.py").unlink()
            moved = repo / "tests/unit/x/test_a.py"
            moved.parent.mkdir(parents=True)
            moved.write_text(new, encoding="utf-8")
            assert changed(repo, base, commit("moved"))
            (repo / "config").mkdir()
            row = {"tests/unit/x/test_a.py": {"old": "tests/test_a.py"}}
            (repo / "config/test-layout.json").write_text(
                json.dumps({"schema": "alphalattice.test-layout.v1", "files": row}),
                encoding="utf-8",
            )
            assert changed(repo, base, commit("paired")) == []
            git("reset", "-q", "--hard", base)
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

        node = "tests/integration/product_host/test_workspace_preparation.py::test_measured"
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
        target = str(root / "tests/integration/product_host/test_strategy_activation.py")
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
        env=child_environment(
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
    if post_stage == "commit":
        assert result.returncode == 5 and "SIZE_ITEMS=[]" in result.stdout
        return
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
            expected = ["large", "large", "medium", "small"]
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
            path = "tests/integration/portfolio_strategy_lab/test_public_control_surface.py"
            node = path + "::test_coverage_names_the_owner_a_request_escapes"
            expected = ("medium", "tests.portfolio_strategy_lab", path + "*", 60)
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
    for rule, want in (("nightly_only", {"nightly"}), ("release_only", {"nightly", "release"})):
        rules = policy | {"large": [{"pattern": closure, "owner": "o", "size_contract": rule}]}
        assert {x for x in want | {"merge"} if included(closure, "large", x, rules, None)} == want
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
    pytest_collection_finish(SimpleNamespace(config=config, items=[]))
    recorded = json.loads((tmp_path / "structural-moved.json").read_text(encoding="utf-8"))
    assert recorded == {"tier": "commit stage", "moved": [other], "ran": []}
    _record_moved(config, "merge", nodes, policy, changed)
    pytest_collection_finish(SimpleNamespace(config=config, items=[nodes[0]]))
    recorded = json.loads((tmp_path / "structural-moved.json").read_text(encoding="utf-8"))
    assert recorded == {"tier": "commit stage", "moved": [other], "ran": [other]}
    _record_moved(config, "merge", nodes, policy, changed, floor=True)
    pytest_collection_finish(SimpleNamespace(config=config, items=[nodes[0]]))
    recorded = json.loads((tmp_path / "structural-moved.json").read_text(encoding="utf-8"))
    assert recorded == {"tier": "commit stage", "moved": [other], "ran": [other]}
    _record_moved(config, "merge", nodes, policy, None)
    pytest_collection_finish(SimpleNamespace(config=config, items=[nodes[1]]))
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


@pytest.mark.parametrize("case", _CASES)
def test_a_quarantined_node_needs_its_owner_reason_and_date(case):
    """A quarantine needs an owner, reason and date."""
    from devtools.testing.sizes import load_policy, validate_policy

    row = {"node": "tests/a.py::t", "owner": "tests.a", "reason": "flaky", "until": "2026-11-03"}
    row |= _CASES[case][0]
    policy = load_policy(Path(__file__).resolve().parents[2]) | {"quarantine": [row]}
    if _CASES[case][1]:
        assert validate_policy(policy)["quarantine"] == [row]
    else:
        with pytest.raises(ValueError, match="quarantine_invalid"):
            validate_policy(policy)


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
    ADMISSION_CASES,
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
        edited = moved._replace(digest="edited")  # the move's table pairs the file: the same test
        assert r.fresh([kept], [edited]) == [edited]
        assert r.fresh([kept], [edited], {"u.py": "t.py"}) == []
        proved = edited._replace(ids=("x.y",))  # whose newly proved ids stay judged
        assert r.fresh([kept], [proved], {"u.py": "t.py"}) == [proved]
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
    row = entries[HELD] | {
        "description": "A restarted Host reuses its last free port.",
        "lower_layer": "Only a served Host on a real loopback port shows the reuse.",
    }
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
    HOLDER_CASES,
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


@pytest.mark.proves("testing.codspeed_marks_every_selected_test")
def test_codspeed_measures_every_selected_test_and_a_plain_run_marks_none(
    monkeypatch, pytestconfig
):
    """CodSpeed deselects unmarked tests, so the cost job marks each; a plain run marks none."""
    from types import SimpleNamespace

    from devtools.testing import collection

    marks: list = []
    config = SimpleNamespace(getoption=lambda name, default=None: default)
    item = SimpleNamespace(config=config, add_marker=marks.append)
    monkeypatch.delenv("CODSPEED_ENV", raising=False)
    collection.pytest_itemcollected(item)
    monkeypatch.setenv("CODSPEED_ENV", "1")  # the CodSpeed action's runner sets it
    pytestconfig.addinivalue_line("markers", "benchmark: registered by pytest-codspeed when on")
    collection.pytest_itemcollected(item)
    assert [mark.name for mark in marks] == ["benchmark"]

    import yaml

    workflow = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml").read_text()
    )
    switch = "ALPHALATTICE_CODSPEED_NO_TIME_BOUNDS"
    assert switch not in workflow.get("env", {})
    for name, job in workflow["jobs"].items():
        assert switch not in job.get("env", {})
        for step in job.get("steps", []):
            value = step.get("env", {}).get(switch)
            if name == "test-cost" and step.get("name") == "Measure each test of the pilot slice":
                assert value == "1" and step["with"]["mode"] == "simulation"
                assert job["continue-on-error"] is True and job["timeout-minutes"] == 90
                assert step["timeout-minutes"] == "${{ fromJSON(steps.budget.outputs.minutes) }}"
            else:
                assert value is None


@pytest.mark.proves("testing.fast_whole_tree_guard_runs_when_its_tree_changed")
def test_a_fast_whole_tree_guard_runs_at_commit_when_the_tree_it_reads_changed() -> None:
    """A deferred guard follows what it reads: source selects it, and a slow one still waits."""
    from devtools.testing import sizes

    guard = "tests/structural/test_a_guard.py"
    fast, slow = f"{guard}::test_fast", f"{guard}::test_slow"
    rule = {"reason": "why", "reads": ["src/*"], "slow": [f"{slow}*"]}
    policy = {**sizes.load_policy(Path(__file__).resolve().parents[2]), "whole_tree_guards": rule}
    source = frozenset({"src/alphalattice/kernel/data/calendar.py"})
    assert sizes.moved(fast, policy) and sizes.included(fast, "medium", "merge", policy, source)
    assert not sizes.included(fast, "medium", "merge", policy, frozenset({"tests/README.md"}))
    assert not sizes.included(slow, "medium", "merge", policy, source)
    assert not sizes.included(
        fast, "medium", "merge", {**policy, "whole_tree_guards": {}}, source
    )  # the escape
    with pytest.raises(ValueError, match=r"test_size\.whole_tree_guards_invalid"):
        sizes.validate_policy({**policy, "whole_tree_guards": {**rule, "reads": []}})


@pytest.mark.proves("testing.fast_whole_tree_guard_runs_when_its_tree_changed")
def test_the_published_policy_keeps_the_whole_tree_rule_without_its_private_nodes() -> None:
    """Every public session loads this policy: its projection passes the rule, no private node."""
    from scripts.release.public_manifest import project_public

    from devtools.testing import sizes

    path = "config/registries/test-sizes.json"
    policy = json.loads((Path(__file__).resolve().parents[2] / path).read_text(encoding="utf-8"))
    policy["whole_tree_guards"]["slow"] = ["tests/kept.py::test_a*", "tests/held.py::test_b*"]
    published, _ = project_public(
        path,
        json.dumps(policy).encode(),
        {"tests/held.py"},
        {path, "tests/kept.py", "tests/held.py"},
    )
    sizes.validate_policy(json.loads(published))
    assert json.loads(published)["whole_tree_guards"]["slow"] == ["tests/kept.py::test_a*"]
    assert b"tests/held.py" not in published
