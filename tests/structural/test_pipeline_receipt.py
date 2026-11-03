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


# The existing receipt owner carries cost standing without changing the default summary.
COST = dict(
    measured=1.0,
    cap=1.25,
    source_head=HEAD,
    source_node=RESULT["node"],
    lane="merge",
    lock_sha256="c" * 64,
    provenance="fresh",
    clean=True,
    other_holders_before=[],
    other_holders_after=[],
)
for _cost_case in (
    "cost fresh",
    "cost inherited",
    "cost bad-head",
    "cost bad-node",
    "cost bad-lock",
    "cost over-cap",
    "cost loaded",
    "cost busy",
    "cost foreign-fresh",
    "cost foreign-node",
    "cost wrong-seconds",
    "cost lock-mismatch",
    "cost dirty-source",
    "cost wrong-lane",
    "cost inherited-no-source",
    "cost fresh-foreign-receipt",
    "cost failing-receipt",
    "cost first-clean",
    "cost first-loaded",
):
    CASES[_cost_case] = (
        {},
        {},
        None
        if _cost_case
        in {"cost fresh", "cost inherited", "cost failing-receipt", "cost first-clean"}
        else "receipt_malformed",
    )


for _case in (
    "binding same",
    "binding move",
    "binding wrong-node",
    "binding wrong-lane",
    "binding wrong-lock",
    "binding nonancestor",
    "binding edited-move",
    "binding duplicate-move",
    "binding duplicate-source",
    "binding missing-target",
    "binding copy",
    "binding missing-git",
    "binding source-mutations",
    "binding git-error",
    "binding interruption",
    "binding foreign-source",
    "binding foreign-target",
    "binding dirty-target",
):
    CASES[_case] = ({}, {}, None)


for _case in (
    "cache hit",
    "cache orphan",
    "cache shadow",
    "cache mismatch",
    "cache mismatch unexplained",
    "cache wrong mode",
):
    CASES[_case] = (
        {},
        {},
        "receipt_inconsistent"
        if _case in {"cache orphan", "cache mismatch unexplained", "cache wrong mode"}
        else None,
    )


@pytest.mark.proves("pipeline.receipt_verifies_only_complete_and_as_asked")
@pytest.mark.parametrize("case", CASES)
def test_a_receipt_verifies_only_complete_and_as_asked(tmp_path, capsys, monkeypatch, case):
    """A receipt counts only whole, with its claim standing over its events, and as asked."""
    if case.startswith("binding "):
        import hashlib
        import inspect
        import subprocess

        supports_root = "cost_root" in inspect.signature(receipt.verify).parameters
        repo = tmp_path / "repo"
        (repo / "tests").mkdir(parents=True)

        def git(*args):
            return subprocess.run(
                ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
            ).stdout.strip()

        git("init", "-q", "-b", "develop")
        git("config", "user.name", "Fixture")
        git("config", "user.email", "fixture@example.invalid")
        old_path, new_path = "tests/test_old.py", "tests/test_new.py"
        old_node = old_path + "::test_a[param]"
        (repo / old_path).write_text("def test_a(): pass\n", encoding="utf-8")
        if case == "binding duplicate-source":
            (repo / "tests/test_duplicate.py").write_bytes((repo / old_path).read_bytes())
        git("add", "."), git("commit", "-qm", "source")
        source_head = git("rev-parse", "HEAD")
        moved = case in {
            "binding move",
            "binding edited-move",
            "binding duplicate-move",
            "binding duplicate-source",
            "binding copy",
        }
        if moved:
            if case == "binding copy":
                (repo / new_path).write_bytes((repo / old_path).read_bytes())
            else:
                git("mv", old_path, new_path)
            if case == "binding edited-move":
                (repo / new_path).write_text("def test_a(): pass  # changed\n", encoding="utf-8")
            if case == "binding duplicate-move":
                (repo / "tests/test_duplicate.py").write_bytes((repo / new_path).read_bytes())
            if case == "binding duplicate-source":
                git("rm", "-q", "tests/test_duplicate.py")
        elif case == "binding missing-target":
            git("rm", "-q", old_path)
        else:
            (repo / old_path).write_text("def test_a(): pass  # target\n", encoding="utf-8")
        git("add", "."), git("commit", "-qm", "target")
        target_head = git("rev-parse", "HEAD")
        if case == "binding nonancestor":
            target_head = git("commit-tree", git("rev-parse", "HEAD^{tree}"), "-m", "unrelated")
        target_node = (new_path if moved else old_path) + (
            "::test_b[param]" if case == "binding wrong-node" else "::test_a[param]"
        )
        cost = dict(COST, source_head=source_head, source_node=old_node)
        source_events, source_receipt = tmp_path / "source-events.jsonl", tmp_path / "source.jsonl"
        source_events.write_text(
            "\n".join(
                map(
                    json.dumps,
                    [
                        {**FIRST, "node": old_node},
                        {**ALONE, "node": old_node, "cost_evidence": cost},
                    ],
                )
            ),
            encoding="utf-8",
        )
        writer = receipt.Writer(
            source_receipt,
            {"stage": "commit", "host": "workshop", "command": ["x"]},
            [{"workspace": {}}, {"configuration": {}}],
        )
        writer.write(
            {"workspace": {}},
            {
                "repo": "harness" if case == "binding foreign-source" else "product",
                "base": source_head,
                "head": source_head,
                "clean": True,
            },
        )
        writer.write(
            {"configuration": {}}, CONFIGURATION | {"lock_sha256": "c" * 64, "lane": "merge"}
        )
        assert not writer.tests(
            "tests",
            {"status": "PASS", "exit": 0, "seconds": 1, "provider": "product"},
            source_events,
        )
        writer.finish({"result": "PASS", "exit": 0, "red_nodes": []})
        digest = hashlib.sha256(source_receipt.read_bytes()).hexdigest()
        inherited = dict(cost, provenance="inherited", source_receipt_sha256=digest)
        target_events, target_receipt = tmp_path / "target-events.jsonl", tmp_path / "target.jsonl"
        target_events.write_text(
            json.dumps({**RESULT, "node": target_node, "cost_evidence": inherited}),
            encoding="utf-8",
        )
        writer = receipt.Writer(
            target_receipt,
            {"stage": "commit", "host": "workshop", "command": ["x"]},
            [{"workspace": {}}, {"configuration": {}}],
        )
        writer.write(
            {"workspace": {}},
            {
                "repo": "harness" if case == "binding foreign-target" else "product",
                "base": source_head,
                "head": target_head,
                "clean": case != "binding dirty-target",
            },
        )
        writer.write(
            {"configuration": {}},
            CONFIGURATION
            | {
                "lock_sha256": ("d" if case == "binding wrong-lock" else "c") * 64,
                "lane": "nightly" if case == "binding wrong-lane" else "merge",
            },
        )
        assert not writer.tests(
            "tests",
            {"status": "PASS", "exit": 0, "seconds": 1, "provider": "product"},
            target_events,
        )
        writer.finish({"result": "PASS", "exit": 0, "red_nodes": []})
        args = dict(
            cost_standing=True,
            cost_nodes=[target_node],
            cost_sources={digest: source_receipt},
        )
        if supports_root:
            args["cost_root"] = tmp_path / "missing" if case == "binding missing-git" else repo
        if case in {"binding git-error", "binding interruption"}:
            original_source, original_target = (
                source_receipt.read_bytes(),
                target_receipt.read_bytes(),
            )
            failure = (
                subprocess.TimeoutExpired("git", 1)
                if case == "binding git-error"
                else KeyboardInterrupt()
            )

            def interrupted(*args, **kwargs):
                raise failure

            with monkeypatch.context() as scoped:
                scoped.setattr(subprocess, "run", interrupted)
                if case == "binding git-error":
                    with pytest.raises(receipt.ReceiptError, match="receipt_cost_unmeasured"):
                        receipt.verify(target_receipt, **args)
                else:
                    with pytest.raises(KeyboardInterrupt):
                        receipt.verify(target_receipt, **args)
            assert (
                source_receipt.read_bytes() == original_source
                and target_receipt.read_bytes() == original_target
            )
        elif case in {"binding same", "binding move"}:
            assert (
                receipt.verify(target_receipt, **args)["cost_standing"][target_node]["source_node"]
                == old_node
            )
        elif case == "binding source-mutations":
            original = target_receipt.read_text("utf-8")
            for field, value in (
                ("source_head", "d" * 40),
                ("source_node", old_node + "_other"),
                ("lane", "candidate"),
                ("lock_sha256", "d" * 64),
                ("measured", 0.5),
                ("cap", 2.0),
            ):
                lines = [json.loads(line) for line in original.splitlines()]
                event = next(e for e in lines if e.get("test_result", {}).get("cost_evidence"))
                event["test_result"]["cost_evidence"][field] = value
                target_receipt.write_text("\n".join(map(json.dumps, lines)), encoding="utf-8")
                with pytest.raises(receipt.ReceiptError, match="receipt_cost_unmeasured"):
                    receipt.verify(target_receipt, **args)
            target_receipt.write_text(original, encoding="utf-8")
        else:
            accepted = True
            try:
                receipt.verify(target_receipt, **args)
            except receipt.ReceiptError as error:
                assert error.code == "pipeline.receipt_cost_unmeasured"
                accepted = False
            assert not accepted, "foreign or unmeasured target identity must not establish standing"
        return
    spoil, asked, code = CASES[case]
    events, path = tmp_path / "events.jsonl", tmp_path / "receipt.jsonl"
    results, red, classes = RERUNS.get(case, ([RESULT], [], []))
    if case.startswith("cost "):
        cost = COST.copy()
        if case in {"cost inherited", "cost inherited-no-source"}:
            cost.update(source_head="b" * 40, provenance="inherited")
            if case == "cost inherited":
                import hashlib

                source, source_events = tmp_path / "source.jsonl", tmp_path / "source-events.jsonl"
                original = {**cost, "provenance": "fresh"}
                source_events.write_text(
                    "\n".join(map(json.dumps, [FIRST, ALONE | {"cost_evidence": original}]))
                )
                source_writer = receipt.Writer(
                    source,
                    {"stage": "commit", "host": "workshop", "command": ["x"]},
                    [{"workspace": {}}, {"configuration": {}}],
                )
                source_writer.write(
                    {"workspace": {}},
                    {"repo": "r", "base": "a" * 40, "head": "b" * 40, "clean": True},
                )
                source_writer.write(
                    {"configuration": {}},
                    CONFIGURATION | {"lock_sha256": "c" * 64, "lane": "merge"},
                )
                assert not source_writer.tests(
                    "tests",
                    {"status": "PASS", "exit": 0, "seconds": 1, "provider": "product"},
                    source_events,
                )
                source_writer.finish({"result": "PASS", "exit": 0, "red_nodes": []})
                cost["source_receipt_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        elif case == "cost bad-head":
            cost["source_head"] = "short"
        elif case == "cost bad-node":
            cost["source_node"] = ""
        elif case == "cost bad-lock":
            cost["lock_sha256"] = "bad"
        elif case == "cost over-cap":
            cost["cap"] = 0.5
        elif case == "cost loaded":
            cost["clean"] = False
        elif case == "cost busy":
            cost["other_holders_before"] = [{"line": "other"}]
        elif case == "cost foreign-fresh":
            cost["source_head"] = "b" * 40
        elif case == "cost foreign-node":
            cost["source_node"] = "t::other"
        elif case == "cost wrong-seconds":
            cost["measured"] = 0.5
        elif case == "cost lock-mismatch":
            cost["lock_sha256"] = "d" * 64
        elif case == "cost wrong-lane":
            cost["lane"] = "nightly"
        elif case == "cost fresh-foreign-receipt":
            cost["source_receipt_sha256"] = "e" * 64
        if case in {"cost first-clean", "cost first-loaded"}:
            results, red, classes = [RESULT | {"cost_evidence": cost}], [], []
        else:
            results, red, classes = [FIRST, ALONE | {"cost_evidence": cost}], [], ["flake"]
    if case.startswith("cache "):
        origin = dict(key="a" * 64, digest="d" * 64, head=HEAD)
        mode = "shadow" if case == "cache shadow" or case.startswith("cache mismatch") else "on"
        results = [RESULT | ({"cached_from": origin} if mode == "on" else {})]
        if case != "cache orphan":
            decision = dict(
                group="f" * 64,
                mode=mode,
                decision="HIT",
                entry=origin,
                reason=None,
                executed=mode != "on",
                expected={
                    RESULT["node"]: "SKIPPED" if case.startswith("cache mismatch") else "PASSED"
                },
            )
            if case == "cache mismatch":
                decision["reason"] = "result_cache.not_passing"
            if case == "cache wrong mode":
                decision["mode"] = "record"
            results = [{"cache": decision}, *results]
    events.write_text("\n".join(map(json.dumps, results)))
    started = {"stage": "commit", "host": "workshop", "command": ["x"]}
    writer = receipt.Writer(path, started, [{"workspace": {}}, {"configuration": {}}])
    workspace = {"repo": "r", "base": "b" * 40, "head": HEAD, "clean": case != "cost dirty-source"}
    writer.write({"workspace": {}}, workspace)
    config = CONFIGURATION | (
        {"lock_sha256": "c" * 64, "lane": "merge"} if case.startswith("cost ") else {}
    )
    if case == "cost first-loaded":
        config["workers"] = 2
    writer.write({"configuration": {}}, config)
    step = {"status": "PASS", "exit": 0, "seconds": 1, "provider": "product"}
    assert writer.tests("tests", step, events) == red  # by its highest attempt, and classed
    failed_receipt = bool(red) or case == "cost failing-receipt"
    writer.finish(
        {
            "result": "FAIL" if failed_receipt else "PASS",
            "exit": int(failed_receipt),
            "red_nodes": red,
        }
    )
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
        try:
            found = receipt.verify(path, **asked)
        except receipt.ReceiptError as error:
            assert code is not None, f"unexpected receipt refusal: {error.code}"
            raise
        assert found["head"] == HEAD
        if case == "complete":
            with pytest.raises(receipt.ReceiptError, match="receipt_cost_unmeasured"):
                receipt.verify(path, cost_standing=True)
            try:
                receipt.verify(path, cost_nodes=[RESULT["node"]])
            except receipt.ReceiptError as missing:
                assert missing.code == "pipeline.receipt_cost_unmeasured"
            except TypeError:
                pytest.fail("required cost evidence cannot be requested")
            else:
                pytest.fail("missing cost evidence must be unmeasured")
        if case == "cost failing-receipt":
            try:
                receipt.verify(path, cost_standing=True)
            except receipt.ReceiptError as failed:
                assert failed.code == "pipeline.receipt_cost_unmeasured"
            else:
                pytest.fail("a failed receipt cannot seed cost standing")
        if case in {"cost fresh", "cost inherited", "cost first-clean"}:
            assert "cost_standing" not in found
            if case == "cost inherited":
                with pytest.raises(receipt.ReceiptError, match="receipt_cost_unmeasured"):
                    receipt.verify(path, cost_standing=True)
                sources = {cost["source_receipt_sha256"]: source}
                for invalid in (path, tmp_path / "missing-source.jsonl"):
                    with pytest.raises(receipt.ReceiptError, match="receipt_cost_unmeasured"):
                        receipt.verify(
                            path,
                            cost_standing=True,
                            cost_sources={cost["source_receipt_sha256"]: invalid},
                        )
            else:
                sources = {}
            if case == "cost inherited":
                with pytest.raises(receipt.ReceiptError, match="receipt_cost_unmeasured"):
                    receipt.verify(path, cost_standing=True, cost_sources=sources)
                return  # synthetic heads without a Git root cannot prove target identity
            standing = receipt.verify(path, cost_standing=True, cost_sources=sources)[
                "cost_standing"
            ]
            assert {k: v for k, v in standing[RESULT["node"]].items() if v is not None} == cost
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
    for when in ("setup", "call", "call", "call", "teardown"):
        report = pytest.TestReport("t::u", ("t", 1, "u"), {}, "passed", "", when, duration=0.4)
        report.user_properties = [("unit_own_seconds", 0.8)] if when == "teardown" else []
        plugin.pytest_runtest_logreport(report)
    for node, when in (("t::c", "setup"), ("t::c", "???"), ("t::d", "setup")):  # crash; open
        report = pytest.TestReport(node, ("t", 1, node), {}, "passed", "gw1 crashed", when)
        report.outcome = "failed" if when == "???" else "passed"
        plugin.pytest_runtest_logreport(report)
    receipt.pytest_unconfigure(config)
    assert config.pluginmanager.get_plugin("pipeline-events") is None and plugin.stream.closed
    a, b, unit, c, d = map(json.loads, (tmp_path / "events.jsonl").read_text("utf-8").splitlines())
    assert (a["status"], a["phase"], a["seconds"], a["size"]) == ("FAILED", "call", 1.5, "small")
    assert a["failure"]["signature"] == b["failure"]["signature"] != ""
    assert unit["seconds"] == 0.8 and unit["status"] == "PASSED"
    assert c["status"] == d["status"] == "ERROR"
    assert (c["phase"], d["phase"]) == ("???", "incomplete")


@pytest.mark.parametrize("fault", ["duplicate", "clock", "tree", "holder-files", "holder-layers"])
@pytest.mark.proves("pipeline.health_report_counts_the_test_metrics")
def test_the_health_report_counts_the_tree_and_the_stages(tmp_path, synthetic_history, fault):
    """The report counts what the tree and the receipts hold; an unreadable one gives no verdict."""
    from datetime import date
    from pathlib import Path

    from devtools.pipeline.definition import DefinitionError
    from devtools.pipeline.health import report
    from devtools.pipeline.health import stages as stage_health

    _, commit = synthetic_history
    root = Path(__file__).resolve().parents[2]
    policy = json.loads((root / "config/registries/test-sizes.json").read_bytes())
    held = {"node": "tests/a/test_x.py::test_c", "owner": "tests.a", "until": "2026-10-28"}
    sizes = tmp_path / "config/registries/test-sizes.json"
    sizes.parent.mkdir(parents=True)
    sizes.write_text(json.dumps(policy | {"quarantine": [held | {"reason": "r"}]}))
    (tmp_path / "config/registries/requirements.json").write_text(
        json.dumps(
            {
                "schema": "alphalattice.requirements.v1",
                "entries": {
                    "a.b": {
                        "description": "One place owns the recorded requirement.",
                        "owner": "devtools.testing",
                        "layer": "integration",
                        "seconds": 1,
                    }
                },
            }
        )
    )
    (tmp_path / "tests/a").mkdir(parents=True)
    mark = '@pytest.mark.proves("a.b")\n'
    body = (
        f"import pytest\n{mark}def test_a(): pass\n{mark}def test_b(): pass\ndef test_c(): pass\n"
    )
    (tmp_path / "tests/a/test_x.py").write_text(body + "#" * 102400 + "\n")
    if fault in {"holder-files", "holder-layers"}:
        extra = tmp_path / (
            "tests/unit/test_y.py" if fault == "holder-layers" else "tests/a/test_y.py"
        )
        extra.parent.mkdir(exist_ok=True)
        extra.write_text(f"import pytest\n{mark}def test_more(): pass\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/m.py").write_text("x = 1\n" * 14)
    commit("A tree to report on")
    flaky = {"node": "t::f", "attempt": 1, "solo": False, "status": "FAILED", "seconds": 1}
    results = [RESULT, RESULT | {"node": "t::slow", "seconds": 12}, flaky]
    results.append(flaky | {"attempt": 2, "solo": True, "status": "PASSED"})
    paths = []
    for name, result, tested in (("pass", "PASS", results), ("error", "ERROR", [])):
        events, path = tmp_path / f"{name}.jsonl", tmp_path / f"{name}-receipt.jsonl"
        events.write_text("\n".join(map(json.dumps, tested)))
        started = {"stage": "commit", "host": "workshop", "command": ["x"]}
        children = [{"workspace": {}}, {"configuration": {}}]
        writer = receipt.Writer(path, started, children)
        writer.write(
            {"workspace": {}}, {"repo": "r", "base": "b" * 40, "head": HEAD, "clean": True}
        )
        writer.write({"configuration": {}}, CONFIGURATION)
        if tested:
            writer.tests(
                "tests", {"status": "PASS", "exit": 0, "seconds": 1, "provider": "p"}, events
            )
        writer.finish({"result": result, "exit": int(result != "PASS"), "red_nodes": []})
        paths.append(path)
    if fault == "holder-layers":
        with pytest.raises(DefinitionError, match=r"testing\.layer_contract") as refused:
            report(tmp_path, paths, date(2026, 10, 25))
        assert "registered layer integration" in str(refused.value)
        assert "register its true layer" in str(refused.value)
        return
    found = report(tmp_path, [*paths, tmp_path / "missing.jsonl"], date(2026, 10, 25))
    tree, stages = found["tree"], found["stages"]
    additional = fault in {"holder-files", "holder-layers"}
    assert (tree["test_lines"], tree["source_lines"], tree["test_to_source"]) == (
        (10, 14, 0.714) if additional else (7, 14, 0.5)
    )
    assert (tree["test_functions"], tree["registered_functions"]) == (
        (4, 3) if additional else (3, 2)
    )
    assert tree["requirements_with_several_holders"] == (["a.b"] if fault == "holder-files" else [])
    assert tree["files_over_100kb"] == ["tests/a/test_x.py"]
    assert tree["quarantine"] == [held | {"days_left": 3}]
    assert stages["by_result"] == {"PASS": 1, "ERROR": 1, "UNREADABLE": 1}
    assert stages["error_rate"] == 0.667 and stages["unreadable"][0]["code"].endswith("missing")
    assert (stages["flaked_nodes"], stages["flake_rate"]) == (["t::f"], 0.3333)
    assert (stages["slow_tests"], stages["slow_share"]) == (1, 0.8)
    assert report(tmp_path, [], date(2026, 10, 25))["stages"]["error_rate"] is None

    if fault == "duplicate":
        copied = tmp_path / "copied.jsonl"
        copied.write_bytes(paths[0].read_bytes())
        assert (
            stage_health([paths[0], paths[0], copied, *paths[1:], tmp_path / "missing.jsonl"])
            == stages
        )
    elif fault == "clock":
        clocks = []
        for name, stamp in (
            ("newer", "2026-10-25T12:00:00+00:00"),
            ("older", "2026-10-25T13:00:00+02:00"),
        ):
            data = [json.loads(line) for line in paths[0].read_text().splitlines()]
            data[0]["started"]["invocation"] = name
            data[-1]["finished"]["finished_at"] = stamp
            path = tmp_path / (name + ".jsonl")
            path.write_text("\n".join(map(json.dumps, data)) + "\n")
            clocks.append(path)
        assert stage_health(clocks)["slow_from"] == str(clocks[0])
        for stamp in ("not-a-time", "2026-10-25T12:00:00"):
            data[-1]["finished"]["finished_at"] = stamp
            clocks[1].write_text("\n".join(map(json.dumps, data)) + "\n")
            invalid = stage_health([clocks[1]])
            assert (
                invalid["by_result"] == {"UNREADABLE": 1}
                and invalid["unreadable"][0]["code"] == "pipeline.receipt_malformed"
            )
    elif fault == "tree":
        bad = tmp_path / "tests/a/test_x.py"
        bad.write_bytes(b"\xff")
        with pytest.raises(Exception) as caught:
            report(tmp_path, [], date(2026, 10, 25))
        assert (
            isinstance(caught.value, DefinitionError)
            and caught.value.code == "pipeline.workspace_unreadable"
            and str(bad) in str(caught.value)
        )
