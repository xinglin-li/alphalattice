"""Find usages before a rename: the recorded escapes of this class, each refused and fixed."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from devtools.architecture.moved_names import moved_name_problems, problems
from devtools.testing.layout import apply, inverse, plan, snapshot

LOCAL = "alphalattice.interface.local_application"
HOST = "alphalattice.control.product_host.composition"
MODELS = "alphalattice.capabilities.alpha_modeling.extensions"
LINEAR = f"src/{MODELS.replace('.', '/')}/regularized_linear.py"
LAZY = "def compare():\n    from .comparison import compare_experiment_reports\n"
HOISTED = "from .comparison import compare_experiment_reports\n"
SETATTR = "monkeypatch.setattr({}, '{}', 1)\n"
HOSTED = "from .client import LocalResearchClient\n"
CLIENT = "class LocalResearchClient: ...\ndef connect():\n    return LocalResearchClient()\n"
CASES = {
    # retrieval_environment.load hoisted into native_bridge, here aliased: its source patch misses
    "hoisted load": (
        f"src/{LOCAL.replace('.', '/')}/native_bridge.py",
        f"from {LOCAL} import retrieval_environment\n",
        f"from {LOCAL}.retrieval_environment import load as fetch\n",
        f"from {LOCAL} import native_bridge, retrieval_environment\n"
        + SETATTR.format("retrieval_environment", "load"),
        f"from {LOCAL} import native_bridge\n" + SETATTR.format("native_bridge", "fetch"),
        f"patch {LOCAL}.native_bridge.fetch",
    ),
    # compare_experiment_reports hoisted to research_experiments' top: patch where it is used
    "hoisted comparison": (
        f"src/{HOST.replace('.', '/')}/research_experiments.py",
        LAZY,
        HOISTED,
        f"patch('{HOST}.comparison.compare_experiment_reports')  # research_experiments\n",
        f"patch('{HOST}.research_experiments.compare_experiment_reports')\n",
        f"patch {HOST}.research_experiments.compare_experiment_reports",
    ),
    # cli._parser renamed command_parser; a script outside the product still imported it
    "renamed": (
        f"src/{LOCAL.replace('.', '/')}/cli.py",
        "def _parser(): ...\n",
        "def command_parser(): ...\n",
        f"from {LOCAL}.cli import _parser\n",
        f"from {LOCAL}.cli import command_parser\n",
        f"it is now {LOCAL}.cli.command_parser",
    ),
    # a moved module's consumer, found by string
    "moved module": (
        "tests/environment.py",
        "def stub(): ...\n",
        None,
        "copied = ['tests.environment']\nfrom tests.environment import stub\n",
        "copied = ['tests.pytest_environment']\n",
        "it is now tests.pytest_environment",
    ),
    # a re-export inside a top-level try (direct script invocation), removed with consumers left
    "re-export in try": (
        "scripts/agent_eval/environments/x5_live.py",
        "try:\n    from .x7_cli import ROOT\nexcept ImportError:\n    from x7_cli import ROOT\n",
        "",
        "from scripts.agent_eval.environments.x5_live import ROOT\n",
        "from scripts.agent_eval.environments.x7_cli import ROOT\n",
        "x5_live.ROOT moved",
    ),
    # Lasso imported lazily: tests still patched it on the module, which no longer binds it
    "lazy import": (
        LINEAR,
        "from sklearn.linear_model import Lasso\n",
        "def fit():\n    from sklearn.linear_model import Lasso\n",
        f"from {MODELS} import regularized_linear\n"
        + SETATTR.format("regularized_linear", "Lasso"),
        "monkeypatch.setattr('sklearn.linear_model.Lasso', 1)\n",
        "regularized_linear.Lasso moved",
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_a_reference_left_at_a_moved_name_is_refused_and_its_fix_passes(case: str) -> None:
    """Each recorded escape names its file and line and the name's new home; its fix passes."""
    path, before, after, bad, good, said = CASES[case]
    moved = "tests/pytest_environment.py" if after is None else None
    changes = [(path, before, after, moved)]
    refused = moved_name_problems(changes, {"tests/test_consumer.py": bad})
    assert refused and refused[0].startswith("tests/test_consumer.py:") and said in refused[0]
    assert moved_name_problems(changes, {"tests/test_consumer.py": good}) == []


@pytest.mark.parametrize("alias", ["client", "client_module"])
def test_a_patch_at_a_module_that_uses_the_name_itself_passes_a_hoist_elsewhere(alias: str) -> None:
    """The native setup tests patch the client where its own code resolves it: never refused."""
    path = f"src/{LOCAL.replace('.', '/')}/"
    changes = [(f"{path}native_bridge.py", "from . import client\n", HOSTED, None)]
    test = f"from {LOCAL} import native_bridge, client as {alias}\n"
    sources = {
        f"{path}client.py": CLIENT,
        "tests/test_native.py": test + SETATTR.format(alias, "LocalResearchClient"),
    }
    assert moved_name_problems(changes, sources) == []


def test_a_staged_check_reads_the_index_not_an_unstaged_repair(synthetic_history, tmp_path: Path):
    """A stale consumer still staged is refused, though the working tree already repairs it."""
    git, commit = synthetic_history
    for path, text in {
        "src/where.py": "def old(): ...\n",
        "tests/use.py": "from where import old\n",
        "scripts/marker.py": "",
        "config/registries/test-sizes.json": "{}",
        "README.md": "Source locators\n",
        ".gitattributes": "* text eol=lf\n",
    }.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8", newline="\n")
    commit("Consumer")
    fact_path = "config/eval/observed.json"  # private facts are found by schema, not by path
    facts = '{"schema": "alphalattice.agent-eval.host-facts", "fakes": []}\n'
    assert fact_path not in snapshot(tmp_path, "HEAD")[1]
    (tmp_path / fact_path).parent.mkdir(parents=True)
    (tmp_path / fact_path).write_text(facts, encoding="utf-8", newline="\n")
    commit("Optional facts")
    captured = snapshot(tmp_path, "HEAD")[1]
    assert fact_path in captured and captured[fact_path] == facts
    (tmp_path / "config/icon.bin").write_bytes(b"\xff\x00")  # a binary asset is never decoded
    commit("A binary config asset")
    assert "config/icon.bin" not in snapshot(tmp_path, "HEAD")[1]
    (tmp_path / "config/eval/torn.json").write_bytes(b"\xff")  # a JSON candidate is named
    commit("An undecodable config JSON")
    with pytest.raises(ValueError, match=r"input_unreadable:config/eval/torn.json"):
        snapshot(tmp_path, "HEAD")
    (tmp_path / "config/eval/torn.json").unlink()
    commit("Repaired")
    assert (
        snapshot(tmp_path, "HEAD", reference_paths=["README.md"])[1]["README.md"]
        == "Source locators\n"
    )
    for address in ("missing.md", "../README.md"):
        with pytest.raises(ValueError, match="registry_invalid"):
            snapshot(tmp_path, "HEAD", reference_paths=[address])
    (tmp_path / "src/where.py").write_text("def new(): ...\n", encoding="utf-8", newline="\n")
    git("add", "src/where.py")
    (tmp_path / "tests/use.py").write_text(
        "from where import new\n", encoding="utf-8", newline="\n"
    )
    assert problems(tmp_path, staged=True)[0].startswith("tests/use.py:1: where.old")
    assert problems(tmp_path, staged=False) == []


def _model_sources(
    root: Path, names=("test_model_contract.py", "test_model_extension.py")
) -> dict[str, Path]:
    found: dict[str, list[Path]] = {name: [] for name in names}
    for path in (root / "tests").rglob("*.py"):
        if path.name in found:
            found[path.name].append(path)
    for name, paths in found.items():
        if len(paths) != 1:
            state = "absent" if not paths else "ambiguous"
            raise ValueError(
                f"test_layout.subject_source_{state}: {name}; keep one defining source under tests/"
            )
    return {name: paths[0] for name, paths in found.items()}


@pytest.mark.proves("testing.whole_file_relocation_restores_bytes")
def test_whole_file_test_layout_restores_bytes_and_refuses_clashes(tmp_path):
    """Whole-file layout restores bytes and refuses path, export and fixture-name clashes."""
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "config").mkdir()
    (tmp_path / "config/package-architecture.json").touch()
    origin = resolve_playpen_root(Path(__file__))
    with pytest.raises(ValueError, match="subject_source_absent"):
        _model_sources(tmp_path)
    for folder in ("a", "b"):
        directory = tmp_path / "tests" / folder
        directory.mkdir(parents=True)
        (directory / "test_model_contract.py").write_text("# duplicate subject\n", encoding="utf-8")
    (tmp_path / "tests/a/test_model_extension.py").write_text(
        "# unique subject\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="subject_source_ambiguous"):
        _model_sources(tmp_path)
    resources = {
        "test_team_scene_evidence_correction.py": {
            "MODULES": tmp_path
            / "src/alphalattice/interface/local_application/assets/workbench-source/js/app",
            "RUNNER": tmp_path / "tests/portfolio_strategy_lab/workbench_team_readback.cjs",
        },
        "test_portfolio_walk_forward.py": {"PLAYPEN_ROOT": tmp_path},
        "test_platform_seams.py": {"ROOT": tmp_path},
    }
    models = ("test_model_contract.py", "test_model_extension.py")
    subjects = _model_sources(origin, (*models, *resources))
    for name in models:
        module = ast.parse(subjects[name].read_text("utf-8"))
        assignments = (n for n in module.body if isinstance(n, ast.Assign))
        resource = next(n for n in assignments if getattr(n.targets[0], "id", None) == "ROOT")
        target = tmp_path / "tests/integration/alpha_modeling" / name
        scope = {**globals(), "resolve_playpen_root": resolve_playpen_root, "__file__": str(target)}
        exec(compile(ast.Module(body=[resource], type_ignores=[]), str(target), "exec"), scope)
        assert scope["ROOT"] == tmp_path, name
    for name, wanted in resources.items():
        subject_path = subjects[name]
        target = tmp_path / "tests/integration/owner" / name
        scope = {**globals(), "__file__": str(target)}
        for node in ast.parse(subject_path.read_text("utf-8")).body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in wanted:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(target), "exec"), scope)
        assert {key: scope[key] for key in wanted} == wanted, name
    body, fixture = "pass\n", "import pytest\n@pytest.fixture\ndef zone(): pass\n"
    fixture += (
        "from tests._support.project_discovery import project_discovery as project_discovery\n"
    )
    files = dict.fromkeys(("tests/a/test_t.py", "tests/b/test_t.py"), body)
    files.update(dict.fromkeys(("tests/a/conftest.py", "tests/b/conftest.py"), fixture))
    catalogs = {
        "test-sizes.json": {
            "defaults": [{"pattern": "tests/a/*", "owner": "a"}],
            "mandatory": ["tests/a/test_t.py*"],
        },
        "test-private.json": {"entries": {"tests/a/test_t.py::patch rule": 1}},
        "ledger.json": {
            "schema": "alphalattice.tests-unregistered.v1",
            "files": {"tests/a/test_t.py": 1},
        },
    }
    catalogs["test-sizes.json"]["defaults"] += [{"pattern": "tests/*", "owner": "fallback"}]
    for kind, key in (("large", "pattern"), ("reach", "pattern"), ("quarantine", "node")):
        catalogs["test-sizes.json"][kind] = [
            {key: "tests/a/test_t.py::case*", "inputs": ["tests/a/test_t.py"]}
        ]
    files |= {f"config/registries/{n}": json.dumps(v) for n, v in catalogs.items()}
    fact_path = "config/eval/observed.json"
    facts = {
        "schema": "alphalattice.agent-eval.host-facts",
        "observations": [{"value": True}],
        "fakes": [
            {"fake": "tests.a.test_t:FakeRunner", "assumes": {"x": {"value": 1}}},
            {"fake": "tests.b.test_t:FakeOwner", "assumes": {}},
        ],
    }
    files[fact_path] = json.dumps(facts)
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes(text.encode("utf-8"))
    choice = ["unit", "a", "t"]
    move = plan("r", files, {"tests/a/test_t.py": choice})
    assert "project_discovery" not in move["files"]["tests/a/conftest.py"][1]
    apply(tmp_path, move)
    assert json.loads((tmp_path / fact_path).read_text()) == {
        **facts,
        "fakes": [
            {**facts["fakes"][0], "fake": "tests.unit.a.test_t:FakeRunner"},
            facts["fakes"][1],
        ],
    }
    assert (tmp_path / "tests/unit/a/test_t.py").read_text() == body
    for name, values in catalogs.items():
        wanted = json.dumps(values).replace("tests/a/test_t.py", "tests/unit/a/test_t.py")
        wanted = wanted.replace("tests/a/*", "tests/unit/a/test_t.py*")
        wanted = json.loads(wanted)
        assert json.loads((tmp_path / f"config/registries/{name}").read_text()) == wanted
    apply(tmp_path, inverse(move))
    assert {p: (tmp_path / p).read_text() for p in files} == files
    for bad, code in (
        ({"fakes": [{"fake": "bad"}]}, "registry_invalid"),
        (
            {
                "fakes": [
                    {"fake": "tests.a.test_t:FakeRunner"},
                    {"fake": "tests.unit.a.test_t:FakeRunner"},
                ]
            },
            "registry_collision",
        ),
    ):
        with pytest.raises(ValueError, match=code):
            plan(
                "r",
                files | {fact_path: json.dumps({"schema": facts["schema"], **bad})},
                {"tests/a/test_t.py": choice},
            )
    for stem in ("t", "other", "../escape"):
        with pytest.raises(ValueError, match=r"collision|fixture|path"):
            plan(
                "r", files, {"tests/a/test_t.py": choice, "tests/b/test_t.py": [*choice[:2], stem]}
            )
    paired = {"tests/a/test_t.py": choice, "tests/a/test_other.py": ["unit", "b", "other"]}
    divided = files | {
        "tests/a/test_other.py": body,
        "tests/a/conftest.py": fixture.replace(
            "@pytest.fixture", "@pytest.fixture(scope='session', name='zone_alias')"
        ),
    }
    for used in (True, False):
        closures = {path: ["zone_alias"] if used else [] for path in paired}
        try:
            moved = plan("r", divided, paired, fixture_closures=closures)
        except ValueError as error:
            assert used and "fixture_scope" in str(error)
        else:
            assert not used and moved["fixture_closures"] == closures
    with pytest.raises(ValueError, match="fixture_scope"):
        plan("r", divided, paired)
    collision = dict.fromkeys(("tests/a/test_t.py", "tests/unit/a/test_t.py"), 1)
    files["config/registries/test-private.json"] = json.dumps({"entries": collision})
    with pytest.raises(ValueError, match="registry_collision"):
        plan("r", files, {"tests/a/test_t.py": choice})
    files["config/registries/test-private.json"] = json.dumps(catalogs["test-private.json"])
    old, new = "tests/a/test_t.py", "tests/unit/a/test_t.py"
    document = {
        "reader": old + "::case",
        "keys": {"before": 17, old: {"window": 126}, "after": 19},
        "unchanged": old + "extra",
    }
    extra = {
        "config/locators.json": (
            '{"owner":"fixture","kind":"fixture","description":"earlier",'
            + json.dumps(document)[1:-1]
            + ',"description":"later"}\n'
        ),
        "README.md": f"Use `{old}::case`.\n",
        "src/reader.py": f'REFERENCE = "{old}"\n',
    }
    references = [
        {"path": "config/locators.json", "target": old, "pointer": "/reader"},
        {
            "path": "config/locators.json",
            "target": old,
            "pointer": "/keys/tests~1a~1test_t.py",
            "key": True,
        },
        {"path": "README.md", "target": old, "line": 1},
        {"path": "src/reader.py", "target": old, "line": 1},
    ]
    files.update(extra)
    for path, text in extra.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8", newline="\n")
    referenced = plan("r", files, {old: choice}, references=references)
    assert len(referenced["source_references"]) == len(references)
    assert "config/__init__.py" not in referenced["files"]
    with pytest.raises(ValueError, match="path_invalid"):
        apply(tmp_path, referenced | {"source_references": []})
    altered = json.loads(referenced["files"]["config/locators.json"][1])
    altered["keys"][new]["window"] = 127
    footprint = referenced["files"] | {
        "config/locators.json": [extra["config/locators.json"], json.dumps(altered)]
    }
    with pytest.raises(ValueError, match="registry_invalid"):
        apply(tmp_path, referenced | {"files": footprint})
    canonical = json.loads(referenced["files"]["config/locators.json"][1])
    for changed in (
        canonical | {"description": "earlier"},
        {"description": "later"} | canonical,
    ):
        with pytest.raises(ValueError, match="registry_invalid"):
            apply(
                tmp_path,
                referenced
                | {
                    "files": referenced["files"]
                    | {"config/locators.json": [extra["config/locators.json"], json.dumps(changed)]}
                },
            )
    apply(tmp_path, referenced)
    loaded = json.loads((tmp_path / "config/locators.json").read_text("utf-8"))
    assert loaded == {"owner": "fixture", "kind": "fixture", "description": "later"} | document | {
        "reader": new + "::case",
        "keys": {"before": 17, new: {"window": 126}, "after": 19},
    }
    assert list(loaded) == ["owner", "kind", "description", "reader", "keys", "unchanged"]
    assert list(loaded["keys"]) == ["before", new, "after"]
    assert (tmp_path / "src/reader.py").read_text("utf-8") == f'REFERENCE = "{new}"\n'
    assert (tmp_path / "README.md").read_text("utf-8") == f"Use `{new}::case`.\n"
    apply(tmp_path, inverse(referenced))
    assert {p: (tmp_path / p).read_text("utf-8") for p in files} == files
    for patch in (
        {"pointer": "/missing"},
        {"pointer": "/bad~2pointer"},
        {"path": "../outside.json"},
        {"target": "tests/b/test_t.py"},
        {"key": 1},
        {"unexpected": 1},
    ):
        with pytest.raises(ValueError, match="registry_invalid"):
            plan("r", files, {old: choice}, references=[references[0] | patch])
    for changed in (
        {"README.md": f"{old} and {old}\n"},
        {"src/reader.py": f'SAFE = "other"; bad = {old}\n'},
    ):
        with pytest.raises(ValueError, match="registry_invalid"):
            plan(
                "r",
                files | changed,
                {old: choice},
                references=[next(r for r in references if r["path"] in changed)],
            )
    with pytest.raises(ValueError, match="registry_invalid"):
        plan("r", files, {old: choice}, references=[{"path": old, "target": old, "line": 1}])
    with pytest.raises(ValueError, match="registry_collision"):
        plan("r", files, {old: choice}, references=[references[0], references[0]])
    with pytest.raises(ValueError, match="addressed member is duplicated"):
        plan(
            "r",
            files
            | {
                "config/locators.json": json.dumps(document)[:-1]
                + ',"reader":'
                + json.dumps(old)
                + "}"
            },
            {old: choice},
            references=[references[0]],
        )
    with pytest.raises(ValueError, match="registry_collision"):
        plan(
            "r",
            files | {"config/locators.json": json.dumps(document | {"keys": {old: 1, new: 2}})},
            {old: choice},
            references=[references[1]],
        )

    module_path = "config/module-locator.json"
    module_text = json.dumps({"module": "tests.a.test_t:case"})
    files[module_path] = module_text
    (tmp_path / module_path).write_text(module_text, encoding="utf-8")
    module_plan = plan(
        "r",
        files,
        {old: choice},
        references=[{"path": module_path, "target": old, "pointer": "/module"}],
    )
    for mutation in ("redirect", "extra"):
        forged = json.loads(json.dumps(module_plan))
        if mutation == "redirect":
            forged["modules"]["tests.a.test_t"] = "tests.unrelated.test_other"
            forged["source_references"][0]["successor"] = "tests.unrelated.test_other:case"
            forged["files"][module_path][1] = (
                json.dumps({"module": "tests.unrelated.test_other:case"}, indent=2) + "\n"
            )
        else:
            forged["modules"]["unrelated.module"] = "another.module"
        words = ""
        try:
            apply(tmp_path, forged)
        except ValueError as error:
            words = str(error)
        assert "reference module map" in words
        assert {p: (tmp_path / p).read_text("utf-8") for p in files} == files
    apply(tmp_path, module_plan)
    assert json.loads((tmp_path / module_path).read_text("utf-8")) == {
        "module": "tests.unit.a.test_t:case"
    }
    apply(tmp_path, inverse(module_plan))
    assert {p: (tmp_path / p).read_text("utf-8") for p in files} == files

    for mutation in (
        "move-image",
        "inverse-key",
        "missing-key",
        "collision",
        "extension",
        "dotted",
        "source-alias",
        "trailing-space",
        "reserved",
        "invalid-name",
        "extra-image",
        "extra-init",
        "extra-provider",
        "ref-alias",
        "dot-image-alias",
    ):
        root = tmp_path / mutation
        root.mkdir()
        for path, text in files.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode("utf-8"))
        forged = json.loads(json.dumps(module_plan))
        if mutation == "move-image":
            forged["moves"][old] = "tests/unrelated/test_other.py"
            forged["modules"]["tests.a.test_t"] = "tests.unrelated.test_other"
            forged["source_references"][0]["successor"] = "tests.unrelated.test_other:case"
            forged["files"][module_path][1] = (
                json.dumps({"module": "tests.unrelated.test_other:case"}, indent=2) + "\n"
            )
        elif mutation == "inverse-key":
            apply(root, module_plan)
            forged = inverse(module_plan)
            forged["modules"]["unrelated.module"] = forged["modules"].pop(
                "tests._support.a_fixtures"
            )
        elif mutation == "missing-key":
            forged["modules"].pop("tests.a.conftest")
        elif mutation in {"extension", "dotted", "trailing-space", "reserved", "invalid-name"}:
            target = {
                "extension": "tests/unit/a/test_other.json",
                "dotted": "tests/unit/a.b/test_other.py",
                "trailing-space": "tests/unit/a /test_other.py",
                "reserved": "tests/unit/con/test_other.py",
                "invalid-name": "tests/unit/a?/test_other.py",
            }[mutation]
            forged["files"][target] = forged["files"].pop(new)
            forged["moves"][old] = target
            module = target[:-3].replace("/", ".")
            forged["modules"]["tests.a.test_t"] = module
            forged["source_references"][0]["successor"] = module + ":case"
            forged["files"][module_path][1] = (
                json.dumps({"module": module + ":case"}, indent=2) + "\n"
            )
        elif mutation in {"extra-image", "extra-init", "extra-provider"}:
            target = {
                "extra-image": "src/extra /__init__.py",
                "extra-init": "tests/unit/extra /__init__.py",
                "extra-provider": "tests/_support/extra /provider.py",
            }[mutation]
            forged["files"][target] = [None, ""]
        elif mutation == "dot-image-alias":
            forged["files"]["tests/a/./test_t.py"] = [files[old], None]
        elif mutation == "ref-alias":
            alias = "config/MODULE-locator.json"
            forged["files"][alias] = list(forged["files"][module_path])
            forged["source_references"].append(forged["source_references"][0] | {"path": alias})
        elif mutation == "source-alias":
            alias, other = "tests/a/TEST_T.py", "tests/unit/a/test_upper.py"
            forged["moves"][alias] = other
            forged["modules"]["tests.a.TEST_T"] = "tests.unit.a.test_upper"
            forged["files"][alias] = [files[old], None]
            forged["files"][other] = [None, files[old]]
        else:
            forged["moves"]["tests/b/test_t.py"] = new
            forged["modules"]["tests.b.test_t"] = "tests.unit.a.test_t"
            forged["files"]["tests/b/test_t.py"] = [files["tests/b/test_t.py"], None]
        before = {
            p: (root / p).read_bytes() if (root / p).is_file() else None for p in forged["files"]
        }
        words = ""
        try:
            apply(root, forged)
        except (ValueError, OSError) as error:
            words = str(error)
        assert (
            "reference source path"
            if mutation == "dot-image-alias"
            else "reference module map"
            if mutation in {"inverse-key", "missing-key"}
            else "reference move images"
        ) in words
        assert {
            p: (root / p).read_bytes() if (root / p).is_file() else None for p in forged["files"]
        } == before
