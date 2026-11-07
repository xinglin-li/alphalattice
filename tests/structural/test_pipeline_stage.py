import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from devtools.pipeline import receipt, stage
from devtools.pipeline.definition import DefinitionError
from tests.structural.pytest_support import child_environment

ROOT = Path(__file__).resolve().parents[2]
ROW = 'owner = "devtools.pipeline"\nreason = "r"\nhosts = '
BOTH = ROW + '["workshop", "public"]\nkind = "command"'
DEFINITION = f"""schema = "alphalattice.pipeline.v1"\nowner = "devtools.pipeline"
[stages.commit]\norder = 1\nreason = "r"\ntrigger = "line"\nrange = "card"\nbound_minutes = 1
rows = ["ok", "red", "local", "job"]
[rows.ok]\n{BOTH}\nrun = ["{{python}}", "-c", "pass"]
[rows.red]\n{BOTH}\nrun = ["{{python}}", "-c", "raise SystemExit(3)"]
[rows.local]\n{ROW}["workshop"]\nkind = "workshop"
[rows.job]\n{ROW}["public"]\nkind = "ci-job"\njob = "w"
"""
SHOP = (  # the adapter records the merge base its context carries
    "import json\n"
    'PROVIDES = {"local": lambda row, at: (at.out / "base").write_text(at.base)'
    ' and ((at.out / "stage").write_text(getattr(at, "stage", "")) >= 0)'
    ' and ((at.out / "size-required").write_text(json.dumps(at.required_tests)) >= 0)'
    ' and ({"status": "PASS", "exit": 0, "seconds": 0}, None)}'
)
CASES = {  # host, plugin bodies, the providers or the refusal's code
    "public": ("public", [], {"ok": "product", "local": "skipped", "job": "ci.yml:w"}),
    "cost receipt": ("public", [], {"ok": "product", "local": "skipped", "job": "ci.yml:w"}),
    "a workshop row with no plugin": ("workshop", [], "row_unprovided"),
    "a plugin's row": ("workshop", [SHOP], {"local": "shop0", "job": "skipped"}),
    "size-required": ("workshop", [SHOP], {"local": "shop0", "job": "skipped"}),
    "cost identity": ("workshop", [SHOP], {"local": "shop0", "job": "skipped"}),
    "cost lock missing": ("workshop", [SHOP], "receipt_cost_unmeasured"),
    "two plugins for a row": ("workshop", [SHOP, SHOP], "row_provided_twice"),
    "a plugin for no row": ("workshop", [SHOP.replace('"local"', '"spare"')], "row_unregistered"),
    "a plugin with no PROVIDES": ("workshop", ["X = 1"], "plugin_unloadable"),
    "two plugins of one name": ("workshop", [SHOP + "\nNAME = 'x'"] * 2, "plugin_unloadable"),
}


@pytest.mark.proves("pipeline.stage_runs_each_row_by_one_provider")
@pytest.mark.proves("pipeline.stage_target_is_separate_from_stop")
@pytest.mark.proves("pipeline.cost_receipt_contains_required_nodes")
@pytest.mark.parametrize("case", CASES)
def test_a_stage_runs_each_row_by_its_one_provider_into_a_receipt_that_verifies(
    tmp_path, case, monkeypatch
):
    host, bodies, wanted = CASES[case]
    if case == "size-required":
        assert hasattr(stage, "derived_holders"), "the stage must select declared owner proofs"
        census_path = "scripts/fake_census.py"
        reader_node = "tests/test_registered.py::test_sentence"
        rows = stage.selection_readers(
            {
                "selection_readers": {
                    reader_node: {
                        "inputs": ["src/product/words.json"],
                        "census": census_path,
                        "sample": "tests/fake_sample.py",
                        "conditional_inputs": ["config/registries/messages.json"],
                    }
                }
            }
        )
        census_source = 'OWNERS = {"control/product_host/new_owner.py": (set(), set())}\n'
        assert stage.registered_holders(
            {"src/alphalattice/control/product_host/new_owner.py"},
            {census_path: census_source},
            rows,
        ) == (reader_node,)
        assert stage.registered_holders({"src/product/words.json"}, {}, rows) == (reader_node,)
        assert stage.registered_holders({"src/product/words.json.bak"}, {}, rows) == ()
        assert (
            stage.registered_holders(
                {"config/registries/messages.json"}, {census_path: census_source}, rows
            )
            == ()
        )
        assert stage.registered_holders(
            {"config/registries/messages.json"},
            {census_path: census_source + '\nCATALOG = "config/registries/messages.json"\n'},
            rows,
        ) == (reader_node,)
        literal = {reader_node: {"inputs": [], "literal_root": "src/product/"}}
        own = {"tests/test_registered.py": 'tree("words.py")'}
        named = stage.selection_readers({"selection_readers": literal})
        assert stage.registered_holders({"src/product/words.py"}, own, named) == (reader_node,)
        assert stage.registered_holders({"src/product/other.py"}, own, named) == ()
        with pytest.raises(ValueError):
            stage.selection_readers(
                {"selection_readers": {reader_node: {"inputs": ["../outside"]}}}
            )
        graph = tmp_path / "graph"
        helper = graph / "tests/support/second.py"
        helper.parent.mkdir(parents=True)
        helper.write_text(
            "from product.owner import run\nVALUE = run()\n"
            "class Runner:\n def call(self): return run()\n"
            "def invoke(): return run()\n",
            encoding="utf-8",
        )
        (helper.parent / "first.py").write_text(
            "from tests.support.second import invoke\ndef forward(): return invoke()\n",
            encoding="utf-8",
        )
        script = graph / "scripts/support.py"
        script.parent.mkdir()
        script.write_text(
            "from product.owner import run\ndef via_script(): return run()\n", encoding="utf-8"
        )
        reader = graph / "tests/test_reader.py"
        reader.write_text(
            "def test_reader():\n from tests.support.first import forward\n forward()\n"
            "def test_alias():\n import tests.support.second\n tests.support.second.invoke()\n"
            "def test_script():\n from scripts.support import via_script\n via_script()\n"
            "def test_value():\n from tests.support.second import VALUE\n assert VALUE\n"
            "def test_import_only():\n from tests.support.second import VALUE\n"
            "def test_class():\n from tests.support.second import Runner\n Runner().call()\n"
            "def test_other():\n pass\n",
            encoding="utf-8",
        )
        assert hasattr(stage, "helper_holders")
        assert stage.helper_holders(graph, {"src/product/owner.py"}) == (
            "tests/test_reader.py::test_alias",
            "tests/test_reader.py::test_class",
            "tests/test_reader.py::test_import_only",
            "tests/test_reader.py::test_reader",
            "tests/test_reader.py::test_script",
            "tests/test_reader.py::test_value",
        )
        assert stage.helper_holders(graph, {"src/product/other.py"}) == ()
        assert (
            "tests/integration/product_host/test_workspace_activity.py::"
            "test_cursor_reconnect_restart_and_typed_read_refusals"
            in stage.helper_holders(
                ROOT, {"src/alphalattice/control/product_host/composition/web_launcher.py"}
            )
        )
        old, new = "a" * 40, "b" * 40

        def pair(path):
            return ([":100644", "100644", old, new, "M"], path)

        materializer = (
            "tests/integration/local_application/test_claude_host_materialization.py::"
            "test_committed_derivatives_match_their_owners"
        )
        host_holder = (
            "tests/structural/test_wait_caps.py::"
            "test_every_host_deadline_owner_has_its_wait_or_nonwait_reviewed"
        )
        client = (
            "tests/structural/test_wait_caps.py::"
            "test_every_capped_client_waiter_has_its_cap_rule_reviewed"
        )
        assert stage.derived_holders(
            [pair(".agents/skills/alphalattice-research/references/specialist-handoff.md")], {}
        ) == (materializer,)
        assert stage.derived_holders([pair(".agents/skills-lookalike/no.md")], {}) == ()
        assert stage.derived_holders([pair(".codex/agents/alphalattice_risk.toml")], {}) == (
            materializer,
        )
        assert stage.derived_holders([pair("config/registries/messages.json")], {}) == (
            materializer,
        )
        assert stage.derived_holders([pair("config/registries/messages-backup.json")], {}) == ()
        cli = "src/alphalattice/interface/local_application/cli.py"
        assert stage.derived_holders([pair(cli)], {old: "pass", new: "pass"}) == (materializer,)
        owner = "src/alphalattice/control/product_host/composition/answer_turns.py"
        assert stage.derived_holders(
            [pair(owner)], {old: "def f(deadline): pass", new: "def f(): pass"}
        ) == (host_holder,)
        assert (
            stage.derived_holders([pair(owner)], {old: "def f(): pass", new: "def f(): pass"}) == ()
        )
        with pytest.raises(SyntaxError):
            stage.derived_holders([pair(owner)], {old: "def f(deadline): pass", new: "def ("})
        assert stage.derived_holders(
            [pair(owner)], {old: "def f(): pass", new: "def f(max_wait): pass"}
        ) == (host_holder,)
        client_path = "src/alphalattice/interface/local_application/client.py"
        assert set(
            stage.derived_holders(
                [pair(client_path)], {old: "def f(deadline): pass", new: "def f(): pass"}
            )
        ) == {client, host_holder}
        monkeypatch.setattr(
            stage,
            "changed_families",
            lambda root, base, head, **kwargs: [
                "tests/test_new.py::test_one",
                "tests/e2e/test_journey.py::test_journey",
                "tests/test_heavy.py::test_heavy",
            ],
            raising=False,
        )
    (tmp_path / "config/registries").mkdir(parents=True)
    for name in ("owners-map.json", "test-sizes.json"):
        shutil.copy(ROOT / "config/registries" / name, tmp_path / "config/registries" / name)
    if case == "size-required":
        size_file = tmp_path / "config/registries/test-sizes.json"
        size_policy = json.loads(size_file.read_text(encoding="utf-8"))
        size_policy["selection_readers"] = rows
        size_policy["large"].append(
            {
                "pattern": "tests/test_heavy.py::test_heavy*",
                "owner": "tests",
                "reason": "Exercises the long observed computation.",
            }
        )
        size_file.write_text(json.dumps(size_policy), encoding="utf-8")
    config = tmp_path / "config/pipeline.toml"
    config.write_text(DEFINITION, "utf-8")
    (tmp_path / "uv.lock").write_text("locked-test-environment\n", encoding="utf-8")
    if case == "size-required":
        helper = tmp_path / "tests/support/host.py"
        helper.parent.mkdir(parents=True)
        helper.write_text(
            "from alphalattice.control.product_host.new_owner import wait\n"
            "def call(): return wait()\n",
            encoding="utf-8",
        )
        script = tmp_path / "scripts/support.py"
        script.parent.mkdir(parents=True)
        script.write_text(helper.read_text(encoding="utf-8"), encoding="utf-8")
        census = tmp_path / census_path
        census.parent.mkdir(parents=True, exist_ok=True)
        census.write_text(
            'OWNERS = {"control/product_host/new_owner.py": (set(), set())}\n', encoding="utf-8"
        )
        journey = tmp_path / "tests/e2e/test_journey.py"
        journey.parent.mkdir(parents=True)
        journey.write_text(
            "def test_journey():\n from tests.support.host import call\n call()\n",
            encoding="utf-8",
        )
        test = tmp_path / "tests/test_host.py"
        test.write_text(
            "def test_host():\n from tests.support.host import call\n call()\n"
            "def test_script():\n from scripts.support import call\n call()\n",
            encoding="utf-8",
        )
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(tmp_path)]
    for words in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "c"]):
        subprocess.run([*git, *words], check=True)
    if case == "size-required":
        before = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        skill = tmp_path / ".agents/skills/example/references/handoff.md"
        skill.parent.mkdir(parents=True)
        skill.write_text("owner", encoding="utf-8")
        deadline = tmp_path / "src/alphalattice/control/product_host/new_owner.py"
        deadline.parent.mkdir(parents=True)
        deadline.write_text("def wait(deadline): pass\n", encoding="utf-8")
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-qm", "derived owners"], check=True)
        after = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        original_test = test.read_text(encoding="utf-8")
        test.write_text("def test_host(): pass\n", encoding="utf-8")
        selected, _ = stage._size_selection("commit", {"base": before, "head": after}, (), tmp_path)
        test.write_text(original_test, encoding="utf-8")
        with pytest.raises(DefinitionError, match=r"pipeline\.stage_invalid"):
            stage._size_selection(
                "commit",
                {"base": before, "head": after},
                ("tests/e2e/test_journey.py::test_journey",),
                tmp_path,
            )
        assert set(selected) == {
            materializer,
            host_holder,
            "tests/test_new.py::test_one",
            "tests/test_host.py::test_host",
            "tests/test_host.py::test_script",
            reader_node,
        }
        from scripts import check_alpha_verification_impact as impact_runner

        with monkeypatch.context() as own:
            own.setattr(impact_runner, "PLAYPEN_ROOT", tmp_path)
            changed = (
                skill.relative_to(tmp_path).as_posix(),
                deadline.relative_to(tmp_path).as_posix(),
            )
            assert set(impact_runner._declared_holders(changed, base=before, staged=False)) == {
                materializer,
                host_holder,
                reader_node,
            }
            deadline.unlink()
            subprocess.run([*git, "add", "-u"], check=True)
            assert set(impact_runner._declared_holders((changed[1],), base=None, staged=True)) == {
                host_holder,
                reader_node,
            }
            deadline.write_text("def wait(deadline): pass\n", encoding="utf-8")
            subprocess.run([*git, "add", changed[1]], check=True)
            outside = tmp_path / "docs/moved-handoff.md"
            outside.parent.mkdir(exist_ok=True)
            subprocess.run(
                [*git, "mv", changed[0], outside.relative_to(tmp_path).as_posix()], check=True
            )
            assert impact_runner._declared_holders(
                (outside.relative_to(tmp_path).as_posix(),), base=None, staged=True
            ) == (materializer,)
            subprocess.run(
                [*git, "mv", outside.relative_to(tmp_path).as_posix(), changed[0]], check=True
            )
        unrelated = tmp_path / "docs/unrelated.md"
        unrelated.parent.mkdir(exist_ok=True)
        unrelated.write_text("not an owner", encoding="utf-8")
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-qm", "unrelated"], check=True)
        next_head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        selected, _ = stage._size_selection(
            "commit", {"base": after, "head": next_head}, (), tmp_path
        )
        assert materializer not in selected and host_holder not in selected
    if case == "cost lock missing":
        (tmp_path / "uv.lock").unlink()
    paths = [tmp_path / f"shop{i}.py" for i in range(len(bodies))]
    for path, body in zip(paths, bodies, strict=True):
        path.write_text(body, "utf-8")
    out = tmp_path / "out"
    asked = dict(
        host=host,
        base=before if case == "size-required" else "HEAD",
        workers=1,
        command=["t"],
        paths=paths,
    )
    with pytest.raises(DefinitionError) as refused:  # a run ends on its rerun into its --out
        summary = stage.run(tmp_path, "commit", out=out, **asked)
        assert (summary["result"], summary["red_nodes"]) == ("FAIL", [])
        lines = (out / "receipt.jsonl").read_text("utf-8")
        assert all(f'"{row}": "{provider}"' in lines for row, provider in wanted.items())
        assert receipt.verify(out / "receipt.jsonl", host=host)["sha256"] == summary["sha256"]
        if case == "public":
            timing = out / "wall-timing.json"
            assert timing.is_file(), "the stage must name time not attributed to receipt rows"
            observed = json.loads(timing.read_text("utf-8"))
            assert set(observed["rows"]) == {"ok", "red", "local", "job"}
            assert observed["unattributed_seconds"] >= 0
            assert observed["wall_seconds"] == pytest.approx(
                sum(observed["rows"].values())
                + observed["unattributed_seconds"]
                - observed["overlap_seconds"],
                abs=0.004,
            )
        if case == "cost identity":
            _, events = receipt.read(out / "receipt.jsonl")
            identity = next(event.configuration for event in events if "configuration" in event.id)
            assert identity.lane == "merge"
            assert (
                identity.lock_sha256
                == hashlib.sha256((tmp_path / "uv.lock").read_bytes()).hexdigest()
            )
        assert not bodies or (out / "base").read_text("utf-8") == summary["base"]  # the merge base
        assert not bodies or (out / "stage").read_text("utf-8") == "commit"
        if case == "size-required":
            assert set(json.loads((out / "size-required").read_text("utf-8"))) == {
                "tests/test_new.py::test_one",
                "tests/test_host.py::test_host",
                "tests/test_host.py::test_script",
                reader_node,
                materializer,
                host_holder,
            }
            assert (
                json.loads((out / "size-stage-selection.json").read_text("utf-8"))["seconds"] >= 0
            )
        stage.run(tmp_path, "commit", out=out, **asked)
    assert refused.value.code == "pipeline." + (wanted if isinstance(wanted, str) else "out_used")
    if case == "cost receipt":
        relative = "tests/portfolio_strategy_lab/test_cost_receipt.py"
        source = tmp_path / relative
        source.parent.mkdir(parents=True)
        source.write_text(
            """import pytest
@pytest.mark.medium
@pytest.mark.parametrize('value',[1,2],ids=['one','two'])
def test_cost_family(value):
    assert value > 0
@pytest.mark.large
def test_slow():
    assert True
def test_control():
    assert True
""",
            "utf-8",
        )
        script = tmp_path / "scripts/pipeline.py"
        script.parent.mkdir()
        shutil.copy(ROOT / "scripts/pipeline.py", script)
        (tmp_path / "pytest.ini").write_text(
            "[pytest]\naddopts=-p devtools.testing.collection\n", "utf-8"
        )
        shape = DEFINITION.split("[rows.ok]", 1)[0].replace(
            'rows = ["ok", "red", "local", "job"]',
            'rows = ["tests"]\nlane="merge"\nsizes=["small","medium"]',
        )
        config.write_text(
            shape + f'\n[rows.tests]\n{ROW}["public"]\nkind="pytest"\n'
            f'public=["{relative}"]\nplugins=["events"]\n'
            f'[rows.events]\n{ROW}["public"]\nkind="pytest-plugin"\n'
            'module="devtools.pipeline.receipt"\n',
            "utf-8",
        )
        argv = [
            sys.executable,
            str(script),
            "run",
            "commit",
            "--host",
            "public",
            "--base",
            "HEAD",
            "--workers",
            "1",
        ]
        family = source.relative_to(tmp_path).as_posix() + "::test_cost_family"
        env = child_environment(
            ALPHALATTICE_TEST_LANE="merge",
            PYTHONPATH=str(ROOT / "src"),
            PYTEST_ADDOPTS="--deselect=" + family,
        )
        ordinary = tmp_path / "ordinary"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0
        )
        done = subprocess.run(
            [*argv, "--out", str(ordinary)],
            env=env,
            capture_output=True,
            timeout=30,
            creationflags=flags,
        )
        assert done.returncode == 0, done.stderr
        with pytest.raises(receipt.ReceiptError, match="missing required tests"):
            receipt.verify(ordinary / "receipt.jsonl", required_tests=[family], within_limits=True)
        cache = source.parent / "__pycache__"
        cache.mkdir(exist_ok=True)
        (cache / "fixture.pyc").write_bytes(b"bytecode is not source")
        for words in (
            ["add", "config/pipeline.toml", "scripts/pipeline.py", relative, "pytest.ini"],
            ["commit", "-qm", "cost fixture", "-m", "Change-Kind: cost"],
        ):
            subprocess.run([*git, *words], check=True)
        tracked = subprocess.check_output([*git, "ls-files", "-z"]).split(b"\0")
        assert not any(name.endswith(b".pyc") for name in tracked)
        cost = tmp_path / "cost"
        done = subprocess.run(
            [*argv, "--out", str(cost), "--required-tests", family],
            env=env,
            capture_output=True,
            timeout=30,
            creationflags=flags,
        )
        assert done.returncode == 0, done.stderr
        head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        assert (
            receipt.verify(
                cost / "receipt.jsonl",
                head=head,
                passing=True,
                required_tests=[family],
                within_limits=True,
            )["result"]
            == "PASS"
        )
        slow = subprocess.run(
            [
                *argv,
                "--out",
                str(tmp_path / "slow"),
                "--required-tests",
                family.replace("test_cost_family", "test_slow"),
            ],
            env=env,
            capture_output=True,
            timeout=30,
            creationflags=flags,
        )
        assert slow.returncode != 0 and b"missing required tests" in slow.stderr
        for label, selected in (
            ("missing", family + "[missing]"),
            ("outside", "../test_x.py::test_x"),
        ):
            bad = subprocess.run(
                [*argv, "--out", str(tmp_path / label), "--required-tests", selected],
                env=env,
                capture_output=True,
                timeout=30,
                creationflags=flags,
            )
            assert bad.returncode != 0
        subprocess.run([*git, "commit", "--allow-empty", "-qm", "ordinary"], check=True)
        refused = subprocess.run(
            [*argv, "--out", str(tmp_path / "not-cost"), "--required-tests", family],
            env=env,
            capture_output=True,
            timeout=30,
            creationflags=flags,
        )
        assert refused.returncode == 2 and b"pipeline.stage_invalid" in refused.stderr
    if case == "public":
        config.write_text(DEFINITION.replace("raise SystemExit(3)", "pass"), "utf-8")
        original_write, words = Path.write_text, io.StringIO()

        def fail_timing(path, *args, **kwargs):
            if path.name == "wall-timing.json":
                raise OSError("fixture timing unavailable")
            return original_write(path, *args, **kwargs)

        with monkeypatch.context() as broken, contextlib.redirect_stderr(words):
            broken.setattr(Path, "write_text", fail_timing)
            unchanged = stage.run(tmp_path, "commit", out=tmp_path / "timing-unavailable", **asked)
        assert unchanged["result"] == "PASS" and "fixture timing unavailable" in words.getvalue()
        verified = receipt.verify(tmp_path / "timing-unavailable/receipt.jsonl", passing=True)
        assert verified["result"] == "PASS"
        config.write_text(DEFINITION, "utf-8")
        interrupted = tmp_path / "interrupt.py"
        interrupted.write_text(
            'def run(row, at):\n    raise RuntimeError("interrupted fixture")\n'
            'PROVIDES = {"local": run}\n',
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="interrupted fixture"):
            stage.run(
                tmp_path,
                "commit",
                out=tmp_path / "interrupted",
                **(asked | {"host": "workshop", "paths": [interrupted]}),
            )
        partial = json.loads((tmp_path / "interrupted/wall-timing.json").read_text("utf-8"))
        assert not partial["complete"] and set(partial["rows"]) == {"ok", "red"}
        paths.append(tmp_path / "budget.py")
        paths[0].write_text(SHOP.replace('"local"', '"red"'), "utf-8")
        for stop, wall in ((2, 59), (2, 61), (2, 121), (3, 121)):
            result = "PASS" if wall <= stop * 60 else "ERROR"
            budget = f"target_minutes = 1\nstop_minutes = {stop}"
            config.write_text(DEFINITION.replace("bound_minutes = 1", budget), "utf-8")
            ticks = iter([0, 0, 0, wall])
            monkeypatch.setattr(stage, "time", SimpleNamespace(monotonic=ticks.__next__))
            target = tmp_path / f"budget-{stop}-{wall}"
            assert stage.run(tmp_path, "commit", out=target, **asked)["result"] == result
            records = list(
                map(json.loads, (target / "receipt.jsonl").read_text("utf-8").splitlines())
            )
            if result == "PASS":
                finished = next(row["finished"] for row in records if "finished" in row["id"])
                assert (finished["red_classes"], finished["red_class"]) == ({}, None)
            seen = next(row["configuration"] for row in records if "configuration" in row["id"])
            metrics = next(row["metrics"] for row in records if row["id"] == {"metrics": {}})
            assert (seen["bound_seconds"], metrics["wall_seconds"] > 60) == (stop * 60, wall > 60)
