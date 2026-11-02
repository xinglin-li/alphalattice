import contextlib
import json
from types import SimpleNamespace

import pytest
from scripts import pipeline as cli

from devtools.pipeline import receipt

HEAD = "a" * 40
RESULT = {"node": "t::alpha[x::y]", "attempt": 1, "solo": False, "status": "PASSED", "seconds": 1}
CONFIGURATION = {"definition": "d", "definition_sha256": "d" * 64, "rows": ["tests"]}
CONFIGURATION |= {"providers": {"tests": "product"}, "workers": 1, "bound_seconds": 900}
TAIL = '{"id": {"metrics": {}}, "metrics": {}}'
TWIN = TAIL.replace("metrics", "workspace")
END = '{"id": {"finished": {"n": 1}}, "last_message": true, "finished": {"result": "FAIL"}}'
FIRST = {**RESULT, "status": "FAILED", "timeout_seconds": 1}
ALONE = {**FIRST, "attempt": 2, "solo": True, "status": "PASSED"}
RERUNS = {  # a case's node results in file order, its red nodes and the classes it writes
    "a red node its solo rerun passed": ([ALONE, FIRST], [], ["flake"]),  # workers finish unordered
    "a solo rerun past its time": ([FIRST, {**ALONE, "seconds": 2}], [RESULT["node"]], ["slow"]),
    "a node still red alone": ([FIRST, {**ALONE, "status": "FAILED"}], [RESULT["node"]], []),
    "required skipped": ([RESULT | {"status": "SKIPPED"}], [], []),
    "required over": ([RESULT | {"timeout_seconds": 0.5}], [], []),
}
RERUNS |= {"required missing": ([ALONE], [], []), "required empty": ([], [], [])}
SIBLING = ALONE | {"node": RESULT["node"] + "[z]"}
RERUNS |= {
    "exact missing": ([SIBLING], [], []),
    "exact skip": ([ALONE, SIBLING | {"status": "SKIPPED"}], [], []),
}
NEEDED = {"required_tests": ["t::alpha"], "within_limits": True}
CASES = {  # line i's (old, new): ("", new) puts a line before it, (old, None) drops it
    "complete": ({}, {"stage": "commit", "head": HEAD, "passing": True}, None),
    "a red node its solo rerun passed": ({}, NEEDED | {"passing": True}, None),
    "a solo rerun past its time": ({}, {}, None),
    "a node still red alone": ({}, {}, None),
    "a host-skipped step": (
        {3: ('"PASS", "exit": 0', '"SKIPPED_BY_HOST", "exit": null')},
        {},
        None,
    ),
    "a second started": (
        {1: ("", '{"id": {"started": {"n": 1}}, "started": {}}')},
        {},
        "receipt_malformed",
    ),
    "a second metrics": ({5: ("", TAIL), 6: ("", TAIL)}, {}, "receipt_malformed"),
    "a card that is no mapping": (
        {6: ("", TAIL.replace("{}}", '{"card": []}}'))},
        {},
        "receipt_malformed",
    ),
    "infinite seconds": ({3: ('"seconds": 1,', '"seconds": Infinity,')}, {}, "receipt_malformed"),
    "a child missing": ({1: ("-", None)}, {}, "receipt_incomplete"),
    "a tail after finished": ({7: ("", TAIL)}, {}, "receipt_incomplete"),
    "another head": ({}, {"head": "b" * 40}, "receipt_mismatch"),
    "not a receipt": ({0: ("-", None)}, {}, "receipt_unknown"),
    "not JSON": ({0: ("{", "[")}, {}, "receipt_unreadable"),
    "no file": (None, {}, "receipt_missing"),
    "a JSON list": ({0: ("", "[1, 2]")}, {}, "receipt_malformed"),
    "an event with no id": ({1: ("", '{"workspace": {}}')}, {}, "receipt_malformed"),
    "a repeated workspace": ({2: ("", TWIN)}, {}, "receipt_malformed"),
    "a short head": ({1: (HEAD, "a" * 9)}, {}, "receipt_malformed"),
    "no command": ({0: ('["x"]', "[]")}, {}, "receipt_malformed"),
    "PASS over a failed step": ({3: ('"PASS"', '"FAIL"')}, {}, "receipt_inconsistent"),
    "PASS over a red node": ({4: ("PASSED", "FAILED")}, {}, "receipt_inconsistent"),
    "PASS over exit 1": ({6: ('"exit": 0', '"exit": 1')}, {}, "receipt_inconsistent"),
    "a step PASS with exit 1": ({3: ('"exit": 0', '"exit": 1')}, {}, "receipt_inconsistent"),
    "a second finished": ({6: ("", END)}, {}, "receipt_malformed"),
    "no tool version": ({0: ("tool_version", "tool")}, {}, "receipt_malformed"),
    "a short digest": ({2: ("d" * 64, "d")}, {}, "receipt_malformed"),
    "a list payload": ({6: ("", TAIL.replace("{}}", "[1]}"))}, {}, "receipt_malformed"),
    "children that are no list": (
        {6: ("", TAIL.replace("{}}", '{}, "children": 1}'))},
        {},
        "receipt_malformed",
    ),
    "required missing": ({}, NEEDED | {"required_tests": ["t::alpha", "t::a"]}, "receipt_mismatch"),
    "required skipped": ({}, NEEDED, "receipt_mismatch"),
    "required unbounded": ({}, NEEDED, "receipt_mismatch"),
    "required over": ({}, NEEDED, "receipt_mismatch"),
    "required empty": ({}, {"within_limits": True}, "receipt_mismatch"),
}
CASES |= {
    name: ({}, NEEDED | {"required_tests": [RESULT["node"]]}, code)
    for name, code in (("exact missing", "receipt_mismatch"), ("exact skip", None))
}


@pytest.mark.proves("pipeline.receipt_verifies_only_complete_and_as_asked")
@pytest.mark.parametrize("case", CASES)
def test_a_receipt_verifies_only_complete_and_as_asked(tmp_path, capsys, case):
    """A receipt counts only whole, with its claim standing over its events, and as asked."""
    spoil, asked, code = CASES[case]
    events, path = tmp_path / "events.jsonl", tmp_path / "receipt.jsonl"
    results, red, classes = RERUNS.get(case, ([RESULT], [], []))
    events.write_text("\n".join(map(json.dumps, results)))
    started = {"stage": "commit", "host": "workshop", "command": ["x"]}
    writer = receipt.Writer(path, started, [{"workspace": {}}, {"configuration": {}}])
    writer.write({"workspace": {}}, {"repo": "r", "base": "b" * 40, "head": HEAD, "clean": True})
    writer.write({"configuration": {}}, CONFIGURATION)
    step = {"status": "PASS", "exit": 0, "seconds": 1, "provider": "product"}
    assert writer.tests("tests", step, events) == red  # by its highest attempt, and classed
    writer.finish({"result": "FAIL" if red else "PASS", "exit": int(bool(red)), "red_nodes": red})
    s = path.read_text("utf-8").splitlines()
    assert [c for c in ("flake", "slow") if f'"class": "{c}"' in "".join(s)] == classes
    for i, (old, new) in sorted((spoil or {}).items(), reverse=True):
        s[i : i + bool(old)] = [] if new is None else [s[i].replace(old, new) if old else new]
    _ = path.unlink() if spoil is None else path.write_text("\n".join(s))
    if asked.get("required_tests"):  # CLI and the unchanged default summary
        names = asked["required_tests"]
        argv = ["verify", str(path), "--passing", "--head", HEAD, "--within-limits"]
        argv += ["--required-tests", *names]
        try:
            status = cli.main(argv)
        except SystemExit as error:
            status = error.code
        out = capsys.readouterr()
        assert status == (2 if code else 0) and (not code or code in out.err)
        assert code or json.loads(out.out) == receipt.verify(path)
    with pytest.raises(receipt.ReceiptError) if code else contextlib.nullcontext() as refused:
        assert receipt.verify(path, **asked)["head"] == HEAD
    assert not code or refused.value.code == "pipeline." + code


@pytest.mark.proves("pipeline.receipt_plugin_writes_each_node_once")
def test_the_plugin_writes_each_node_once_with_its_failure_and_disposes_of_itself(tmp_path):
    """A node's result follows its teardown; a failure's signature ignores paths and numbers."""
    config = SimpleNamespace(pluginmanager=pytest.PytestPluginManager())
    receipt.register(config, tmp_path / "events.jsonl")
    plugin = config.pluginmanager.get_plugin("pipeline-events")
    for node, line in (
        ("t::a", r"AssertionError: C:\x\a.py:12 holds 3 rows at abcdef1"),
        ("t::b", "AssertionError: /y/b.py:9 holds 70 rows at 0123fed"),
    ):
        for when, outcome in (("setup", "passed"), ("call", "failed"), ("teardown", "passed")):
            report = pytest.TestReport(node, ("t", 1, node), {}, outcome, line, when, duration=0.5)
            report.user_properties = [("test_size", "small")]
            plugin.pytest_runtest_logreport(report)
    for node, when in (("t::c", "setup"), ("t::c", "???"), ("t::d", "setup")):  # crash; open
        report = pytest.TestReport(node, ("t", 1, node), {}, "passed", "gw1 crashed", when)
        report.outcome = "failed" if when == "???" else "passed"
        plugin.pytest_runtest_logreport(report)
    receipt.pytest_unconfigure(config)
    assert config.pluginmanager.get_plugin("pipeline-events") is None and plugin.stream.closed
    a, b, c, d = map(json.loads, (tmp_path / "events.jsonl").read_text("utf-8").splitlines())
    assert (a["status"], a["phase"], a["seconds"], a["size"]) == ("FAILED", "call", 1.5, "small")
    assert a["failure"]["signature"] == b["failure"]["signature"] != ""
    assert c["status"] == d["status"] == "ERROR"
    assert (c["phase"], d["phase"]) == ("???", "incomplete")
