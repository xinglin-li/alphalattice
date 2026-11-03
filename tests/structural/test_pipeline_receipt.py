import contextlib
import json
import subprocess
import sys
import time
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import pipeline as cli

from devtools.pipeline import receipt

ROOT = Path(__file__).resolve().parents[2]

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
    "binding cli",
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
    "legacy target valid",
    "legacy target unknown",
    "legacy target type",
    "legacy target late",
    "legacy target expired",
    "legacy target current",
    "legacy target one-field",
    "legacy target overflow",
    "legacy target bool",
    "legacy target future",
    "legacy target cutoff",
):
    CASES[_case] = ({}, {}, None)


RED_CLASS_ROWS = {  # the classed writer's fields, read before it ships: (finished, step, admitted)
    "red classes on a pass": ({"red_classes": {}, "red_class": None}, {}, True),
    "red classes counted": (
        {"red_classes": {"tooling": 1, "load": 2}, "red_class": "load+tooling"},
        {},
        True,
    ),
    "red classes stopped at the bound": ({}, {"stopped": "bound"}, True),
    "red classes negative count": ({"red_classes": {"tooling": -1}}, {}, False),
    "red classes unknown class": ({"red_classes": {"cosmic": 1}}, {}, False),
    "red classes text count": ({"red_classes": {"tooling": "1"}}, {}, False),
    "red classes unknown stop": ({}, {"stopped": "later"}, False),
    "red classes other field": ({"red_classification": "tooling"}, {}, False),
}
for _case in RED_CLASS_ROWS:
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
    if case.startswith("legacy target "):
        from datetime import datetime

        path = tmp_path / "legacy.jsonl"
        writer = receipt.Writer(
            path,
            {"stage": "commit", "host": "workshop", "command": ["x"]},
            [{"workspace": {}}, {"configuration": {}}, {"metrics": {}}],
        )
        writer.write(
            {"workspace": {}}, {"repo": "product", "base": "b" * 40, "head": HEAD, "clean": True}
        )
        writer.write({"configuration": {}}, CONFIGURATION)
        metrics = {"wall_seconds": 1.0}
        if case != "legacy target current":
            metrics.update(target_seconds=600, target_breached=True)
        if case == "legacy target one-field":
            metrics.pop("target_breached")
        if case == "legacy target overflow":
            metrics["target_seconds"] = 10**400
        if case == "legacy target bool":
            metrics["target_breached"] = "true"
        if case == "legacy target unknown":
            metrics["unknown"] = 1
        if case == "legacy target type":
            metrics["target_seconds"] = "600"
        writer.write({"metrics": {}}, metrics)
        writer.finish({"result": "PASS", "exit": 0, "red_nodes": []})
        lines = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        lines[0]["started"]["started_at"] = "2026-10-25T04:25:00-04:00"
        lines[-1]["finished"]["finished_at"] = (
            "2026-10-25T04:44:42-04:00"
            if case == "legacy target cutoff"
            else "2026-10-25T04:45:00-04:00"
            if case == "legacy target late"
            else "2026-10-25T04:39:41-04:00"
        )
        path.write_text("\n".join(map(json.dumps, lines)) + "\n", encoding="utf-8")
        before = path.read_bytes()
        now = datetime.fromisoformat(
            "2026-11-01T07:21:14-05:00"
            if case == "legacy target expired"
            else "2026-10-25T04:30:00-04:00"
            if case == "legacy target future"
            else "2026-10-31T12:00:00-04:00"
        )
        with monkeypatch.context() as scoped:

            class FrozenDatetime(datetime):
                @classmethod
                def now(cls, tz=None):
                    return now

            scoped.setattr(receipt, "datetime", FrozenDatetime)
            try:
                data, events = receipt.read(path)
            except receipt.ReceiptError as error:
                assert error.code == "pipeline.receipt_malformed"
                accepted = False
            else:
                assert data == before and path.read_bytes() == before
                metrics_event = next(event.metrics for event in events if "metrics" in event.id)
                assert not hasattr(metrics_event, "target_seconds") and not hasattr(
                    metrics_event, "target_breached"
                )
                accepted = True
        assert accepted is (
            case in {"legacy target valid", "legacy target current", "legacy target one-field"}
        )
        return
    if (
        case in RED_CLASS_ROWS
    ):  # a reader that knows the classed fields, strictly, before any writer
        finished, step, admitted = RED_CLASS_ROWS[case]
        result = "PASS" if case == "red classes on a pass" else "FAIL"
        path = tmp_path / "classed.jsonl"
        writer = receipt.Writer(
            path,
            {"stage": "commit", "host": "workshop", "command": ["x"]},
            [{"workspace": {}}, {"configuration": {}}, {"step": {"row": "tests"}}, {"metrics": {}}],
        )
        writer.write(
            {"workspace": {}}, {"repo": "product", "base": "b" * 40, "head": HEAD, "clean": True}
        )
        writer.write({"configuration": {}}, CONFIGURATION)
        ran = (
            {"status": "PASS", "exit": 0} if result == "PASS" else {"status": "ERROR", "exit": None}
        )
        writer.write(
            {"step": {"row": "tests"}}, ran | {"seconds": 1.0, "provider": "workshop"} | step
        )
        writer.write({"metrics": {}}, {"wall_seconds": 1.0})
        writer.finish({"result": result, "exit": 0 if result == "PASS" else 1, "red_nodes": []})
        lines = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        lines[-1]["finished"] |= finished
        path.write_text("\n".join(map(json.dumps, lines)) + "\n", encoding="utf-8")
        try:
            receipt.read(path)
        except receipt.ReceiptError as error:
            assert error.code == "pipeline.receipt_malformed", error.code
            accepted = False
        else:
            accepted = True
        assert accepted is admitted, case
        return
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
        elif case in {"binding same", "binding move", "binding cli"}:
            assert (
                receipt.verify(target_receipt, **args)["cost_standing"][target_node]["source_node"]
                == old_node
            )
            if case == "binding cli":
                command = [
                    sys.executable,
                    str(ROOT / "scripts/pipeline.py"),
                    "verify",
                    str(target_receipt),
                    "--cost-standing",
                    "--cost-root",
                    str(repo),
                    "--cost-source",
                    digest,
                    str(source_receipt),
                ]
                completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
                assert completed.returncode == 0, completed.stderr
                assert (
                    json.loads(completed.stdout)["cost_standing"][target_node]["source_node"]
                    == old_node
                )
                duplicate = subprocess.run(
                    [*command, "--cost-source", digest, str(source_receipt)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert duplicate.returncode != 0 and "duplicate" in duplicate.stderr.lower()
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
            assert receipt.verify(path, cost_if_present=True) == found
            assert cli.main(["verify", str(path), "--cost-if-present"]) == 0
            assert json.loads(capsys.readouterr().out) == found
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
            if case != "cost inherited":
                assert (
                    receipt.verify(path, cost_if_present=True)["cost_standing"][RESULT["node"]][
                        "cap"
                    ]
                    == 1.25
                )
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
    for when, seconds in (
        ("setup", 0.2),
        ("call", 0.1),
        ("call", 0.2),
        ("call", 0.6),
        ("teardown", 0.1),
    ):
        report = pytest.TestReport("t::m", ("t", 1, "m"), {}, "passed", "", when, duration=seconds)
        report.user_properties = [("test_size", "medium")]
        plugin.pytest_runtest_logreport(report)
    for when in ("setup", "teardown"):
        report = pytest.TestReport(
            "t::missing", ("t", 1, "missing"), {}, "passed", "", when, duration=0.1
        )
        plugin.pytest_runtest_logreport(report)
    for node, when in (("t::c", "setup"), ("t::c", "???"), ("t::d", "setup")):  # crash; open
        report = pytest.TestReport(node, ("t", 1, node), {}, "passed", "gw1 crashed", when)
        report.outcome = "failed" if when == "???" else "passed"
        plugin.pytest_runtest_logreport(report)
    receipt.pytest_unconfigure(config)
    assert config.pluginmanager.get_plugin("pipeline-events") is None and plugin.stream.closed
    a, b, unit, medium, missing, c, d = receipt.read_test_events(tmp_path / "events.jsonl")
    bad_events = tmp_path / "broken-events.jsonl"
    bad_events.write_text("[]\n", encoding="utf-8")
    with pytest.raises(receipt.ReceiptError, match="receipt_unreadable"):
        receipt.read_test_events(bad_events)
    assert (a["status"], a["phase"], a["seconds"], a["size"]) == ("FAILED", "call", 1.5, "small")
    assert a["failure"]["signature"] == b["failure"]["signature"] != ""
    assert unit["seconds"] == 0.8 and unit["status"] == "PASSED"
    assert medium["seconds"] == pytest.approx(0.9) and medium["status"] == "PASSED"
    assert missing["status"] == "ERROR" and missing["phase"] == "incomplete"
    assert c["status"] == d["status"] == "ERROR"
    assert (c["phase"], d["phase"]) == ("???", "incomplete")


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "clock",
        "tree",
        "holder-files",
        "holder-layers",
        "ranked",
        "machine",
        "rank-gap",
        "rank-spike",
        "rank-threshold",
    ],
)
@pytest.mark.proves("pipeline.health_report_counts_the_test_metrics")
def test_the_health_report_counts_the_tree_and_the_stages(tmp_path, synthetic_history, fault):
    """The report counts what the tree and the receipts hold; an unreadable one gives no verdict."""
    from datetime import date
    from pathlib import Path

    from devtools.pipeline import health
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
    ranked = getattr(health, "ranked", None)
    assert callable(ranked), "the report must rank each test's seconds across stages"
    assert (stages["tests"]["runs"], stages["tests"]["unmeasured"]) == (0, 0)  # finished after
    written = stage_health(paths, date.today())["tests"]  # in its window: under 50 nodes each
    for day in ([], [tmp_path / "missing.jsonl"]):  # an empty day and an unreadable one
        quiet = stage_health(day, date(2026, 10, 25))["tests"]
        assert (quiet["runs"], quiet["slowest"], quiet["durations"]) == (0, [], {})
    assert (written["runs"], written["unmeasured"], written["durations"]) == (0, 2, {})
    assert report(tmp_path, [], date(2026, 10, 25))["stages"]["error_rate"] is None

    if fault == "duplicate":
        copied = tmp_path / "copied.jsonl"
        copied.write_bytes(paths[0].read_bytes())
        assert (
            stage_health(
                [paths[0], paths[0], copied, *paths[1:], tmp_path / "missing.jsonl"],
                date(2026, 10, 25),
            )
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
    elif fault in {"ranked", "machine", "rank-gap", "rank-spike", "rank-threshold"}:
        import inspect
        from datetime import datetime, timedelta

        assert "merged" in inspect.signature(ranked).parameters, "a jump says if it is merged"

        today, base = date(2026, 10, 25), {f"t::n{i}": ("PASSED", 1.0) for i in range(50)}

        def run(days_ago, head, nodes=base, **seen):  # a number is a pass; a tuple says itself
            at = datetime(2026, 10, 25, 12, tzinfo=UTC) - timedelta(days=days_ago)
            extra = {
                f"t::{n}": s if isinstance(s, tuple) else ("PASSED", s) for n, s in seen.items()
            }
            return (at, head, nodes | extra)

        if fault == "rank-gap":
            runs = [run(3 - i, f"g{i}", **({"gap": 12} if i != 2 else {})) for i in range(4)]
            assert "t::gap" not in ranked(runs, today)["slow_three_running"]
            return
        if fault == "rank-spike":
            rows = ranked([run(1, "a", spike=1), run(0, "b", spike=4)], today)["slowest"]
            assert next(row for row in rows if row["node"] == "t::spike")["jumped_in"] is None
            return
        if fault == "rank-threshold":
            slower = {n: ("PASSED", 1.996) for n in base}
            state = ranked([run(1, "a"), run(0, "b", nodes=slower)], today)
            assert state["runs"] == 2 and state["degraded"] == []
            # Exactly 2x is degraded; 3x, above it, is the machine case.
            at = {n: ("PASSED", 2.0) for n in base}
            state = ranked([run(1, "a"), run(0, "b", nodes=at)], today)
            assert [(row["head"], row["factor"]) for row in state["degraded"]] == [("b", 2.0)]
            return

        fail = ("FAILED", 0.1)
        rows = [(9, 1, 12, 2), (12, 1, 12, 2), (11, 1, 12, 2), (13, 1, 12, 8)]
        rows += [(12, 4, 12, 8), (12, 4, fail, 2), (12, 4, 12, 2)]
        grow = [
            run(
                6 - i,
                f"h{i}",
                slow=s,
                grow=g,
                broken=b,
                rebound=r,
                **({"once": 7} if i == 6 else {}),
            )
            for i, (s, g, b, r) in enumerate(rows)
        ]
        table = ranked(grow, today, merged=lambda head: head == "h4")
        row = {r["node"]: r for r in table["slowest"]}
        slow, grown, rebound = row["t::slow"], row["t::grow"], row["t::rebound"]
        assert (slow["p50"], slow["p90"], slow["runs"], slow["first_slow"][:10]) == (
            12,
            13,
            7,
            "2026-10-20",
        )
        assert table["slow_three_running"] == ["t::slow"]  # a fast failure breaks t::broken's
        assert grown["jumped_in"] == {
            "at": "2026-10-23T12:00:00+00:00",
            "head": "h4",
            "merged": True,
        }
        assert rebound["jumped_in"] is None and slow["jumped_in"] is None  # 8, 8 came back down
        assert [(r["node"], r["trend"]) for r in table["growers"]] == [
            ("t::grow", 3),
            ("t::slow", 1),
        ]
        assert table["durations"]["t::once"] == row["t::once"]["p90"] == 7  # one sample
        assert table["durations"]["t::slow"] == 12 and len(table["durations"]) == 55
        unknown = {r["node"]: r for r in ranked(grow, today)["slowest"]}["t::grow"]["jumped_in"]
        assert unknown["head"] == "h4" and unknown["merged"] is None  # no history, no claim
        for count, p90 in ((10, 9), (20, 18)):  # nearest rank: the ceil(0.9 n)-th
            series = [run(count - 1 - i, f"s{i}", spread=float(i + 1)) for i in range(count)]
            spread = {r["node"]: r for r in ranked(series, today)["slowest"]}["t::spread"]
            assert spread["p90"] == p90, count
        if fault == "machine":  # a 3x run, a thin one and an old one never enter a baseline
            others = [
                run(1, "slow-machine", nodes={n: ("PASSED", 3.0) for n in base}, slow=40, grow=12),
                (run(1, "thin")[0], "thin", {"t::slow": ("PASSED", 90.0)}),
                run(30, "old", slow=90, grow=90),
            ]
            machine = ranked([*grow, *others], today)
            assert machine["degraded"] == [
                {"at": "2026-10-24T12:00:00+00:00", "head": "slow-machine", "factor": 3.0}
            ]
            assert (machine["runs"], machine["unmeasured"]) == (7, 1)
            assert machine["slowest"] == ranked(grow, today)["slowest"]
            alone = ranked([run(0, "only")], today)  # no other run to compare with: unmeasured
            assert (alone["runs"], alone["unmeasured"], alone["durations"]) == (0, 1, {})
            other = {f"t::m{i}": ("PASSED", 1.0) for i in range(50)}
            disjoint = ranked([run(1, "a"), run(0, "b", nodes=other)], today)
            assert (disjoint["runs"], disjoint["unmeasured"]) == (0, 2)
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


@pytest.mark.parametrize(
    "case",
    [
        "two-stage",
        "digest-unchanged",
        "digest-edited",
        "digest-move",
        "digest-unknown-move",
        "digest-expired-unknown",
        "digest-future-unknown",
        "mixed-unknown",
        "incomplete-mixed",
        "malformed-mixed",
        "unreadable-window",
        "mixed-owner-unknown",
        "post-unknown",
        "blocked-same",
        "own-setup",
        "worker-crash",
        "workspace-cascade",
        "two-mixed",
        "same-head",
        "duplicate",
        "outside",
        "no-owner",
        "unreadable",
        "missing-failure",
        "post-invalid",
        "different-failure",
        "already-held",
        "held-repeat",
        "held-invalid",
        "held-family",
        "held-sibling",
        "held-pattern",
        "skipped-solo",
        "unbounded-solo",
        "future-post",
        "stage-fail-other",
        "stage-error",
        "stage-error-per-node",
        "noncommit-stage",
        "foreign-repo",
        "nonassert",
        "post-nonassert",
        "policy-missing",
        "policy-malformed",
        "no-owner-policy",
        "today-invalid",
        "old-second",
        "post-proof-missing",
        "post-proof-present",
        "missing-requirement",
        "unknown-requirement",
        "ambiguous-owner",
        "same-owner-multiple",
        "requirements-missing",
        "requirements-malformed",
        "legacy-owned",
        "legacy-parameter",
        "legacy-over-count",
        "legacy-missing-file",
        "legacy-bad-owner",
        "legacy-malformed",
        "legacy-requirement-first",
    ],
)
@pytest.mark.proves("pipeline.flake_quarantine_candidates")
def test_flake_candidates_use_distinct_verified_heads_and_registered_owners(
    tmp_path, monkeypatch, case
):
    """Two distinct recent flake observations identify one owned quarantine candidate."""
    from datetime import date
    from pathlib import Path

    from devtools.pipeline import health
    from devtools.pipeline.definition import DefinitionError

    select = getattr(health, "flake_candidates", None)
    assert callable(select), "test health must derive automatic quarantine candidates"
    root = Path(__file__).resolve().parents[2]
    policy = json.loads((root / "config/registries/test-sizes.json").read_text("utf-8"))
    policy["defaults"].append({"pattern": "tests/a/*", "owner": "tests.a", "size": "medium"})
    if case == "no-owner-policy":
        policy["defaults"] = [{"pattern": "tests/b/*", "owner": "tests.b", "size": "medium"}]
    if case in {"already-held", "held-repeat", "held-family", "held-sibling", "held-pattern"}:
        selector = "tests/a/test_x.py::test_c" + (
            "[0]" if case == "held-sibling" else "*" if case == "held-pattern" else ""
        )
        policy["quarantine"].append(
            {
                "node": selector,
                "owner": "tests.a",
                "until": "2026-10-30",
                "reason": "owned review",
            }
        )
    owner_path = tmp_path / "config/registries/test-sizes.json"
    owner_path.parent.mkdir(parents=True)
    owner_path.write_text(json.dumps(policy), encoding="utf-8")
    if case == "policy-missing":
        owner_path.unlink()
    if case == "policy-malformed":
        owner_path.write_text("{", encoding="utf-8")
    requirements = tmp_path / "config/registries/requirements.json"
    rows = {
        "example.flake": {
            "description": "A flake is owned.",
            "owner": "devtools.example",
            "layer": "integration",
            "seconds": 5,
        },
        "example.other": {
            "description": "A second behavior is owned.",
            "owner": "devtools.other" if case == "ambiguous-owner" else "devtools.example",
            "layer": "integration",
            "seconds": 5,
        },
    }
    requirements.write_text(
        json.dumps({"schema": "alphalattice.requirements.v1", "entries": rows}), encoding="utf-8"
    )
    if case == "requirements-missing":
        requirements.unlink()
    if case == "requirements-malformed":
        requirements.write_text("{", encoding="utf-8")
    (tmp_path / "tests/a").mkdir(parents=True)
    ids = (
        []
        if case == "missing-requirement"
        or (case.startswith("legacy-") and case != "legacy-requirement-first")
        else ["example.unknown"]
        if case == "unknown-requirement"
        else ["example.flake", "example.other"]
        if case in {"ambiguous-owner", "same-owner-multiple"}
        else ["example.flake"]
    )
    mark = "" if not ids else "@pytest.mark.proves(" + ", ".join(map(repr, ids)) + ")\n"
    (tmp_path / "tests/a/test_x.py").write_text(
        "import pytest\n" + mark + "def test_c(): pass\n", encoding="utf-8"
    )
    if case.startswith("legacy-"):
        ledger = tmp_path / "config/registries/ledger.json"  # found by its schema
        document = {
            "schema": "alphalattice.tests-unregistered.v1",
            "owner": "" if case == "legacy-bad-owner" else "devtools.testing",
            "files": {} if case == "legacy-missing-file" else {"tests/a/test_x.py": 1},
        }
        ledger.write_text(
            "{" if case == "legacy-malformed" else json.dumps(document), encoding="utf-8"
        )
        if case == "legacy-over-count":
            with (tmp_path / "tests/a/test_x.py").open("a", encoding="utf-8") as source:
                source.write("def test_extra(): pass\n")
    node = "tests/a/test_x.py::test_c" if case != "no-owner" else "foreign/test_x.py::test_c"
    if case == "legacy-parameter":
        node += "[one]"
    if case in {"held-family", "held-sibling", "held-pattern"}:
        node += "[1]"
    heads = ("a" * 40, "b" * 40)
    observed_nodes = (node, node)
    import subprocess

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    (
        git("init", "-q", "-b", "develop"),
        git("config", "user.name", "Fixture"),
        git("config", "user.email", "fixture@example.invalid"),
    )
    git("add", "."), git("commit", "-qm", "first holder")
    source_head = git("rev-parse", "HEAD")
    if case in {"digest-move", "digest-unknown-move"}:
        git("mv", "tests/a/test_x.py", "tests/a/test_moved.py")
        observed_nodes = (node, node.replace("test_x.py", "test_moved.py"))
    else:
        (tmp_path / "unrelated.txt").write_text("other change", encoding="utf-8")
    git("add", "."), git("commit", "-qm", "second observation")
    heads = (source_head, git("rev-parse", "HEAD"))
    if case == "digest-edited":
        path = tmp_path / "tests/a/test_x.py"
        path.write_text(
            path.read_text("utf-8").replace("def test_c(): pass", "def test_c(): assert True"),
            encoding="utf-8",
        )
        git("add", "."), git("commit", "-qm", "holder fix")
    if case in {"digest-move", "digest-unknown-move"}:
        node = observed_nodes[1]

    def stage(name, head, day, failure):
        stage_node = observed_nodes[0] if name == "first" else observed_nodes[1]
        events, path = tmp_path / (name + "-events.jsonl"), tmp_path / (name + ".jsonl")
        first = {
            "node": stage_node,
            "attempt": 1,
            "solo": False,
            "status": "FAILED",
            "seconds": 1,
            "timeout_seconds": 10,
            "failure": {"type": "AssertionError", "message": failure, "signature": "f" * 64},
        }
        if case == "own-setup":
            first.update(
                status="ERROR",
                phase="setup",
                failure={
                    "type": "AssertionError",
                    "message": "AssertionError: setup guard",
                    "signature": "f" * 64,
                },
            )
        if case == "worker-crash":
            first.update(
                status="ERROR",
                phase="???",
                failure={
                    "type": "Failure",
                    "message": f"worker 'gw8' crashed while running '{stage_node}'",
                    "signature": "f" * 64,
                },
            )
        if case == "workspace-cascade":
            first.update(
                status="ERROR",
                phase="setup",
                failure={
                    "type": "RuntimeError",
                    "message": "incomplete session workspace: fixture",
                    "signature": "f" * 64,
                },
            )
        second = {
            "node": stage_node,
            "attempt": 2,
            "solo": True,
            "status": "PASSED",
            "seconds": 1,
            "timeout_seconds": 10,
        }
        results = [first, second]
        if (case == "mixed-unknown" and name == "second") or case == "mixed-owner-unknown":
            unknown = "tests/a/test_x.py::test_unknown"
            results += [
                {
                    **first,
                    "node": unknown,
                    "failure": {
                        "type": "AssertionError" if case == "mixed-owner-unknown" else "ValueError",
                        "message": "assert 1 == 2"
                        if case == "mixed-owner-unknown"
                        else "ValueError: unavailable",
                        "signature": "e" * 64,
                    },
                },
                {**second, "node": unknown},
            ]
        if case == "workspace-cascade":
            crashed = "tests/a/test_x.py::test_crashed"
            results += [
                {
                    **first,
                    "node": crashed,
                    "phase": "???",
                    "failure": {
                        "type": "Failure",
                        "message": f"worker 'gw1' crashed while running '{crashed}'",
                        "signature": "e" * 64,
                    },
                },
                {**second, "node": crashed},
            ]
        events.write_text("\n".join(map(json.dumps, results)), encoding="utf-8")
        writer = receipt.Writer(
            path,
            {"stage": "commit", "host": "workshop", "command": ["x"]},
            [{"workspace": {}}, {"configuration": {}}],
        )
        writer.write(
            {"workspace": {}}, {"repo": "product", "base": "b" * 40, "head": head, "clean": True}
        )
        writer.write({"configuration": {}}, CONFIGURATION)
        assert not writer.tests(
            "tests", {"status": "PASS", "exit": 0, "seconds": 1, "provider": "p"}, events
        )
        writer.finish({"result": "PASS", "exit": 0, "red_nodes": []})
        lines = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        lines[-1]["finished"]["finished_at"] = day + "T12:00:00-04:00"
        path.write_text("\n".join(map(json.dumps, lines)) + "\n", encoding="utf-8")
        return path

    first = stage(
        "first",
        heads[0],
        "2026-10-20" if case == "outside" else "2026-10-22",
        ""
        if case == "missing-failure"
        else "ValueError: bad"
        if case == "nonassert"
        else "assert 4 == 16",
    )
    second = stage(
        "second",
        heads[1],
        "2026-10-26" if case == "old-second" else "2026-10-27",
        "assert 1 == 2" if case == "different-failure" else "assert 4 == 16",
    )
    post = {
        "source": "post-merge",
        "head": heads[0] if case == "same-head" else heads[1],
        "node": node,
        "at": "2026-10-27T13:00:00-04:00",
        "run": "post-run",
        "first_failure": "assert 4 == 16",
    }
    if case == "post-invalid":
        post["head"] = "invalid"
    if case == "post-nonassert":
        post["first_failure"] = "ValueError: bad"
    if case in {"post-proof-missing", "post-proof-present"}:
        post["failure_evidence"] = "failure.xml"
        if case == "post-proof-present":
            post["solo_evidence"] = "solo.xml"
    if case == "future-post":
        post["at"] = "2026-10-28T13:00:00-04:00"
    if case in {"stage-fail-other", "stage-error", "stage-error-per-node"}:
        lines = [json.loads(line) for line in second.read_text("utf-8").splitlines()]
        lines[-1]["finished"]["result"] = "FAIL" if case == "stage-fail-other" else "ERROR"
        lines[-1]["finished"]["exit"] = 1 if case == "stage-fail-other" else 2
        second.write_text("\n".join(map(json.dumps, lines)) + "\n", encoding="utf-8")
    if case in {"noncommit-stage", "foreign-repo"}:
        lines = [json.loads(line) for line in second.read_text("utf-8").splitlines()]
        if case == "noncommit-stage":
            lines[0]["started"]["stage"] = "nightly"
        else:
            next(line["workspace"] for line in lines if "workspace" in line)["repo"] = "harness"
        second.write_text("\n".join(map(json.dumps, lines)) + "\n", encoding="utf-8")
    if case in {"skipped-solo", "unbounded-solo"}:
        lines = [json.loads(line) for line in second.read_text("utf-8").splitlines()]
        latest = next(
            line["test_result"] for line in lines if line.get("test_result", {}).get("attempt") == 2
        )
        latest["status" if case == "skipped-solo" else "timeout_seconds"] = (
            "SKIPPED" if case == "skipped-solo" else None
        )
        second.write_text("\n".join(map(json.dumps, lines)) + "\n", encoding="utf-8")
    receipts = [first, second]
    posts = []
    if case in {
        "two-mixed",
        "same-head",
        "post-invalid",
        "future-post",
        "post-nonassert",
        "post-proof-missing",
        "post-proof-present",
    }:
        receipts, posts = [first], [post]
    if case == "duplicate":
        copy = tmp_path / "copy.jsonl"
        copy.write_bytes(first.read_bytes())
        receipts = [first, copy]
    if case == "unreadable":
        receipts = [first, tmp_path / "missing.jsonl"]
    if case == "incomplete-mixed":
        incomplete = tmp_path / "interrupted.jsonl"
        incomplete.write_text(
            "\n".join(first.read_text("utf-8").splitlines()[:-1]) + "\n", encoding="utf-8"
        )
        receipts = [first, second, incomplete]
    if case == "malformed-mixed":
        incomplete = tmp_path / "malformed.jsonl"
        lines = [json.loads(line) for line in first.read_text("utf-8").splitlines()]
        lines[0]["started"]["unknown_field"] = True
        incomplete.write_text("\n".join(map(json.dumps, lines)) + "\n", encoding="utf-8")
        receipts = [first, second, incomplete]
    if case == "unreadable-window":
        receipts = [first, second, tmp_path / "missing.jsonl"]
    if case in {
        "unreadable",
        "no-owner",
        "missing-failure",
        "post-invalid",
        "unbounded-solo",
        "future-post",
        "stage-error",
        "nonassert",
        "post-nonassert",
        "policy-missing",
        "policy-malformed",
        "today-invalid",
        "post-proof-missing",
        "missing-requirement",
        "unknown-requirement",
        "ambiguous-owner",
        "requirements-missing",
        "requirements-malformed",
        "legacy-over-count",
        "legacy-missing-file",
        "legacy-bad-owner",
        "legacy-malformed",
    }:
        with pytest.raises(DefinitionError, match="flake"):
            select(
                tmp_path,
                receipts,
                posts,
                "bad date" if case == "today-invalid" else date(2026, 10, 27),
            )
        return
    if case.startswith("digest-"):
        assess = getattr(health, "flake_assessment", None)
        assert callable(assess), "holder identity needs the product assessment"
        prior_unknown = (
            [
                dict(
                    source="post-merge",
                    run="post-run",
                    node=observed_nodes[0],
                    head=heads[0],
                    kind="unknown",
                    detail="assertion unavailable",
                )
            ]
            if case in {"digest-unknown-move", "digest-expired-unknown", "digest-future-unknown"}
            else []
        )
        if case in {"digest-expired-unknown", "digest-future-unknown"}:
            prior_unknown[0]["at"] = (
                "2026-10-19T12:00:00+00:00"
                if case == "digest-expired-unknown"
                else "2026-10-28T12:00:00+00:00"
            )
        if case == "digest-future-unknown":
            with pytest.raises(DefinitionError, match="issue date"):
                assess(tmp_path, receipts, posts, date(2026, 10, 27), post_issues=prior_unknown)
            return
        result = assess(tmp_path, receipts, posts, date(2026, 10, 27), post_issues=prior_unknown)
        if case == "digest-unknown-move":
            assert not result["candidates"] and any(
                issue["kind"] == "unknown" and issue["node"] == node for issue in result["issues"]
            )
        elif case == "digest-edited":
            assert not result["candidates"] and any(
                issue["kind"] == "superseded" for issue in result["issues"]
            )
        else:
            assert [row["node"] for row in result["candidates"]] == [node]
            observations = result["candidates"][0]["observations"]
            assert len({row["holder_digest"] for row in observations}) == 1
            assert [row["source_node"] for row in observations] == list(observed_nodes)
        return
    if case in {
        "mixed-unknown",
        "incomplete-mixed",
        "malformed-mixed",
        "unreadable-window",
        "mixed-owner-unknown",
        "post-unknown",
        "blocked-same",
        "own-setup",
        "worker-crash",
        "workspace-cascade",
        "stage-error-per-node",
    }:
        assess = getattr(health, "flake_assessment", None)
        assert callable(assess), "independent node outcomes need a measured flake assessment"
        reported = []
        if case in {"post-unknown", "blocked-same"}:
            reported = [
                dict(
                    source="post-merge",
                    run="post-run",
                    node=node if case == "blocked-same" else "tests/a/test_x.py::test_unknown",
                    kind="unknown",
                    detail="assertion unavailable",
                )
            ]
        if case == "unreadable-window":
            with pytest.raises(DefinitionError, match="receipt_missing"):
                assess(tmp_path, receipts, posts, date(2026, 10, 27), post_issues=reported)
            return
        result = assess(tmp_path, receipts, posts, date(2026, 10, 27), post_issues=reported)
        kinds = {item["kind"] for item in result["issues"]}
        if case == "malformed-mixed":
            assert [row["node"] for row in result["candidates"]] == [node]
            assert any(
                item["kind"] == "unusable-stage" and item["run"] == str(incomplete)
                for item in result["issues"]
            )
        elif case == "incomplete-mixed":
            assert [row["node"] for row in result["candidates"]] == [node]
            assert any(
                item["kind"] == "incomplete-stage" and item["run"] == str(incomplete)
                for item in result["issues"]
            )
        elif case in {"mixed-unknown", "mixed-owner-unknown", "post-unknown"}:
            assert [row["node"] for row in result["candidates"]] == [node]
            assert any(
                item["node"].endswith("::test_unknown") and item["kind"] == "unknown"
                for item in result["issues"]
            )
        elif case == "stage-error-per-node":
            assert [row["node"] for row in result["candidates"]] == [node]
            assert kinds == {"stage-unmeasured"}
        elif case == "blocked-same":
            assert not result["candidates"] and kinds == {"unknown"}
        elif case == "own-setup":
            assert [row["node"] for row in result["candidates"]] == [node] and not result["issues"]
        else:
            assert not result["candidates"]
            assert "worker-crash" in kinds
            if case == "workspace-cascade":
                assert "workspace-cascade" in kinds
        return
    if case == "held-invalid":
        with pytest.raises(DefinitionError, match="flake"):
            select(tmp_path, receipts, posts, date(2026, 10, 27), include_held="yes")
        return
    options = {"include_held": True} if case == "held-repeat" else {}
    if options:
        import inspect

        assert "include_held" in inspect.signature(select).parameters, (
            "repeat assessment needs held-source visibility"
        )
    candidates = select(tmp_path, receipts, posts, date(2026, 10, 27), **options)
    if case in {
        "same-head",
        "duplicate",
        "outside",
        "already-held",
        "held-family",
        "held-pattern",
        "skipped-solo",
        "noncommit-stage",
        "foreign-repo",
    }:
        assert candidates == []
    else:
        assert len(candidates) == 1
        row = candidates[0]
        expected_owner = (
            "devtools.testing"
            if case in {"legacy-owned", "legacy-parameter"}
            else "devtools.example"
        )
        assert (row["node"], row["owner"], row["until"]) == (node, expected_owner, "2026-11-03")
        assert row["owner_kind"] == (
            "unregistered legacy" if case in {"legacy-owned", "legacy-parameter"} else "requirement"
        )
        if case == "held-repeat":
            assert row["held"] is True
        assert row["reason"].startswith("Automatic: flake twice in 7 days")
        assert str(tmp_path) not in row["reason"]
        assert all(observation["head"] in row["reason"] for observation in row["observations"])
        assert len(row["observations"]) == 2
        assert row["same_failure"] is (case != "different-failure")


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@localhost", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return done.stdout.strip()


def _code_tree(root: Path) -> str:
    """Three registered packages in two areas, high importing low and gone holding no source,
    committed; returns the day's start."""
    packages = [
        {
            "package_id": name,
            "physical_namespace_target": f"alphalattice.{group}.{name}",
            "product_area": group,
        }
        for name, group in (("low", "kernel"), ("high", "control"), ("gone", "control"))
    ]
    (root / "config").mkdir(parents=True)
    (root / "config/package-architecture.json").write_text(json.dumps({"packages": packages}))
    for package in (
        "alphalattice",
        "alphalattice/kernel",
        "alphalattice/kernel/low",
        "alphalattice/control",
        "alphalattice/control/high",
        "devtools",
    ):
        (root / "src" / package).mkdir(parents=True, exist_ok=True)
        (root / "src" / package / "__init__.py").write_text("")
    (root / "src/alphalattice/kernel/low/core.py").write_text("value = 1\n")
    (root / "src/alphalattice/control/high/view.py").write_text(
        "from alphalattice.kernel.low import core\n"
    )
    _git(root, "init", "-q", "-b", "develop")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "start")
    return _git(root, "rev-parse", "HEAD")


CODE_CASES = {  # case -> (file under low, its text, landed by a card's merge or directly)
    "growing": ("core.py", "value = 1\nvalue = 2\nvalue = 3\nvalue = 4\n", "merge"),
    "cycle": ("core.py", "from alphalattice.control.high import view\nvalue = 1\n", "merge"),
    "unreadable graph": ("broken.py", "def (:\n", "merge"),
    "unicode path": ("测 试.py", "a = 1\nb = 2\n", "merge"),
    "direct commit": ("core.py", "value = 1\nvalue = 2\nvalue = 3\n", "direct"),
}


@pytest.mark.proves("pipeline.code_health_reports_every_row")
@pytest.mark.parametrize(
    "case",
    [
        *CODE_CASES,
        "reused",
        "dirty",
        "unmeasured measured again",
        "malformed store",
        "unwritable store",
        "quiet day",
        "removed source",
        "ignored source",
        "ignored registry",
        "empty registry",
        "staged source",
        "absent ignore rules",
    ],
)
def test_the_code_report_reads_the_tree_by_its_tools_and_a_failed_row_as_unmeasured(
    tmp_path, monkeypatch, case
):
    """The code rows are the tree's own, for every registered package, path and day, kept by the
    bytes they read; a row, input or store that cannot be trusted is measured anew or named."""
    from devtools.pipeline import health

    root, store = tmp_path / "repo", tmp_path / "store"
    root.mkdir()
    start = _code_tree(root)
    registry = root / "config/package-architecture.json"
    if case in CODE_CASES:
        name, text, how = CODE_CASES[case]
        if how == "merge":
            _git(root, "checkout", "-q", "-b", "card")
        (root / "src/alphalattice/kernel/low" / name).write_text(text, encoding="utf-8")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "grow directly" if how == "direct" else "card")
        if how == "merge":
            _git(root, "checkout", "-q", "develop")
            _git(root, "merge", "-q", "--no-ff", "card", "-m", "Merge GROW-CARD (c2): the change")
    if case == "unmeasured measured again":  # the graph's tool is down for the first report only
        monkeypatch.setattr(health, "GRAPH", "raise SystemExit(3)")
    if case == "unwritable store":
        store.write_text("a file, not a folder")
    if case == "removed source":  # the day removes the last unowned sources, two lines of them
        (root / "src/loose.py").write_text("a = 1\nb = 2\n")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "a loose module")
        start = _git(root, "rev-parse", "HEAD")
        for unowned in (
            "loose.py",
            "alphalattice/__init__.py",
            "alphalattice/kernel/__init__.py",
            "alphalattice/control/__init__.py",
            "devtools/__init__.py",
        ):
            (root / "src" / unowned).unlink()
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "remove them")
    if case == "ignored source":
        (root / ".gitignore").write_text("src/alphalattice/secret.py\n")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "ignore one module")
    if case == "ignored registry":  # the registry the rows read is ignored and untracked
        (root / ".gitignore").write_text("config/package-architecture.json\n")
        _git(root, "rm", "-q", "--cached", "config/package-architecture.json")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "keep the registry local")
    if case == "staged source":  # untracked at the first report; staged later, its bytes unchanged
        (root / "src/alphalattice/kernel/low/extra.py").write_text("a = 1\nb = 2\n")
    if case == "empty registry":
        registry.write_text(json.dumps({"packages": []}))
        _git(root, "commit", "-q", "-am", "no packages")
    found = health.code(root, store, day_start=start)
    rows = {"sizes", "boundaries", "growers", "dead_code", "dependencies", "complexity"}
    assert set(found["seconds"]) == rows  # every row measured on a first report
    kept = next(iter(sorted(store.glob("*.json"))), None) if store.is_dir() else None
    measured = sorted(name for name in rows if "unmeasured" not in found[name])
    if case == "growing":
        assert found["sizes"]["packages"]["low"]["day"] == 3 and found["sizes"]["day_net"] == 3
        assert found["sizes"]["packages"]["gone"] == {"lines": 0, "day": 0}  # registered, empty
        assert found["sizes"]["areas"] == {
            "kernel": {"lines": 4, "day": 3},
            "control": {"lines": 1, "day": 0},
            "unowned": {"lines": 0, "day": 0},
        }
        assert found["growers"][0] == {
            "path": "src/alphalattice/kernel/low/core.py",
            "net": 3,
            "merges": ["GROW-CARD (c2)"],
        }
        assert found["boundaries"]["cycles"] == []
        assert found["boundaries"]["packages"] == {
            "high": {"statements": 1, "share": 1.0, "fan_out": 1, "fan_in": 0},
            "low": {"statements": 0, "share": 0.0, "fan_out": 0, "fan_in": 1},  # a sink
            "gone": {"statements": 0, "share": 0.0, "fan_out": 0, "fan_in": 0},
        }
        assert found["store"] == "kept" and kept is not None
    elif case == "cycle":
        assert found["boundaries"]["cycles"] == [["high", "low"]]
    elif case == "unicode path":  # Git's exact path, not its quoted escape
        assert found["sizes"]["packages"]["low"]["day"] == 2
        assert found["growers"][0]["path"] == "src/alphalattice/kernel/low/测 试.py"
    elif case == "direct commit":  # growth no merge brought is still named, by its commit
        assert found["growers"][0]["merges"] == ["direct: grow directly"]
        assert found["sizes"]["day_net"] == 2
    elif case in {"reused", "empty registry"}:  # unchanged inputs are read from the store
        row = json.loads(kept.read_text("utf-8"))
        kept.write_text(json.dumps(row | {"sizes": row["sizes"] | {"source_lines": 999}}))
        again = health.code(root, store, day_start=start)
        assert again["reused"] == measured and set(again["seconds"]) == rows - {*measured}
        assert again["sizes"]["source_lines"] == 999  # read from the store, not measured
        assert case != "empty registry" or again["boundaries"]["packages"] == {}  # complete: none
        assert health.code(root, store, day_start="")["reused"] == []  # other bases: measured anew
    elif case == "dirty":  # what the rows read is not the commit: its own inputs, measured anew
        before = kept.read_text("utf-8")
        core = root / "src/alphalattice/kernel/low/core.py"
        core.write_text("value = 1\nvalue = 2\n")
        (root / "src/alphalattice/kernel/low/new.py").write_text("fresh = 1\n")  # untracked too
        again = health.code(root, store, day_start=start)
        assert again["reused"] == [] and again["sizes"]["packages"]["low"]["lines"] == 2
        assert kept.read_text("utf-8") == before  # the committed tree's entry untouched
        core.write_text("value = 1\n")
        (root / "src/alphalattice/kernel/low/new.py").unlink()  # back to the commit: its rows again
        assert health.code(root, store, day_start=start)["reused"] == measured
    elif case == "unmeasured measured again":  # a tool's outage is never kept as the tree's answer
        assert "unmeasured" in found["boundaries"]
        monkeypatch.undo()
        again = health.code(root, store, day_start=start)
        assert again["reused"] == measured and "boundaries" not in measured
        assert again["boundaries"]["cycles"] == [] and "boundaries" in again["seconds"]
    elif (
        case == "malformed store"
    ):  # malformed, wrong-shaped, truncated or incomplete: measured anew
        for body in (
            "{}",
            "[1]",
            '{"inputs": ',
            json.dumps({"inputs": found["inputs"], "sizes": 3}),
        ):
            kept.write_text(body)
            again = health.code(root, store, day_start=start)
            assert again["sizes"] == found["sizes"] and again["reused"] == []
        row = json.loads(kept.read_text("utf-8"))
        for broken in (  # the right inputs and keys, the wrong or incomplete rows inside
            {
                "sizes": {},
                "boundaries": [],
                "growers": [{"path": 1}],
                "complexity": {**row["complexity"], "packages": {"x": "y"}},
            },
            {
                "sizes": {**row["sizes"], "packages": {}, "areas": {}},
                "boundaries": {**row["boundaries"], "packages": {}},
            },
            {"sizes": {**row["sizes"], "budget": {}}},
            {"sizes": {**row["sizes"], "areas": {"unowned": {"lines": 0, "day": 0}}}},
        ):
            kept.write_text(json.dumps(row | broken))
            again = health.code(root, store, day_start=start)
            assert again["reused"] == sorted(set(measured) - set(broken))  # valid rows kept
            assert again["sizes"] == found["sizes"] and again["boundaries"] == found["boundaries"]
    elif case == "unwritable store":
        assert found["store"].startswith("not kept: ") and found["sizes"]["source_lines"] > 0
    elif case == "removed source":  # a package whose last source went today still reports its day
        assert found["sizes"]["day_net"] == -2 and "unowned" in found["sizes"]["packages"]
        assert found["sizes"]["packages"]["unowned"] == {"lines": 0, "day": -2}
        assert found["sizes"]["areas"]["unowned"] == {"lines": 0, "day": -2}
    elif case == "ignored source":  # an ignored module the tools still walk is an input too
        secret = root / "src/alphalattice/secret.py"
        secret.write_text("hidden = 1\n")
        assert health.code(root, store, day_start=start)["reused"] == []
        secret.unlink()  # gone again: the tree's own rows are reused
        assert health.code(root, store, day_start=start)["reused"] == measured
    elif case == "ignored registry":  # an ignored, untracked registry edit is a new input
        assert set(found["sizes"]["areas"]) == {"kernel", "control", "unowned"}
        packages = json.loads(registry.read_text())["packages"]
        registry.write_text(
            json.dumps({"packages": [{**p, "product_area": "moved"} for p in packages]})
        )
        again = health.code(root, store, day_start=start)
        assert again["reused"] == [] and set(again["sizes"]["areas"]) == {"moved", "unowned"}
    elif case == "staged source":  # sizes counts Git's index: staging or unstaging is a new input
        _git(root, "add", "src/alphalattice/kernel/low/extra.py")
        again = health.code(root, store, day_start=start)
        assert "sizes" not in again["reused"]
        assert again["sizes"]["source_lines"] == found["sizes"]["source_lines"] + 2
        _git(root, "rm", "-q", "--cached", "src/alphalattice/kernel/low/extra.py")
        assert health.code(root, store, day_start=start)["reused"] == measured  # the first index
    elif case == "absent ignore rules":  # absence is its own mark, never the bytes b"absent"
        assert not (root / ".gitignore").exists()
        (root / ".gitignore").write_bytes(b"absent")
        assert health.code(root, store, day_start=start)["reused"] == []
        (root / ".gitignore").unlink()
        assert health.code(root, store, day_start=start)["reused"] == measured
    elif case == "quiet day":  # the day is the 24 hours before the report, not before HEAD
        later = health.code(root, store, now=time.time() + 3 * 86_400)
        assert later["inputs"]["day_start"] == _git(root, "rev-parse", "HEAD")
        assert later["sizes"]["day_net"] == 0 and later["growers"] == []
        assert "unmeasured" in health.code(root, None)["growers"]  # no commit before the day
    else:
        assert "unmeasured" in found["boundaries"] and "cycles" not in found["boundaries"]
        assert found["sizes"]["source_lines"] > 0  # the other rows still report
