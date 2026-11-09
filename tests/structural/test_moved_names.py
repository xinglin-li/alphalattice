"""Find usages before a rename: the recorded escapes of this class, each refused and fixed."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from devtools.architecture.moved_names import moved_name_problems, problems

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


@pytest.mark.proves("testing.test_placement_and_structural_scope")
def test_test_placement_and_structural_scope(tmp_path):
    """New tests keep their placement and structural-scope checks."""
    from copy import deepcopy
    from types import SimpleNamespace

    from devtools.testing import layout
    from devtools.testing.collection import _size_included
    from devtools.testing.sizes import (
        load_policy,
        stage_files,
        validate_policy,
        validate_post_merge_scopes,
    )

    registered = load_policy(Path(__file__).resolve().parents[2])
    floor_change = frozenset({"src/example.py"})
    post = "tests/structural/test_whole.py"
    declared_file = tmp_path / post
    declared_file.parent.mkdir(parents=True, exist_ok=True)
    declared_file.write_text("TEST_SCOPE = 'whole-tree'\n", encoding="utf-8")
    candidate = deepcopy(registered) | {"post_merge_only": [post]}
    try:
        validate_policy(candidate)
        admitted = True
    except ValueError:
        admitted = False
    assert admitted, "a structural file may be deferred when its whole-tree declaration is verified"
    partial = tmp_path / "partial-policy" / "config/registries/test-sizes.json"
    partial.parent.mkdir(parents=True)
    partial.write_text(json.dumps(candidate), encoding="utf-8")
    assert load_policy(partial.parents[2])["post_merge_only"] == [
        post
    ]  # policy-only readers need no test tree

    validate_post_merge_scopes(candidate, lambda name: (tmp_path / name).read_text("utf-8"))
    with pytest.raises(ValueError, match=r"test_size\.rule"):
        stage_files(
            candidate, "commit", [post], lane="merge"
        )  # the default reader refuses the absent declared file
    assert (
        stage_files(
            candidate,
            "commit",
            [post],
            lane="merge",
            read=lambda name: (tmp_path / name).read_text("utf-8"),
        )
        == []
    )
    deferred = SimpleNamespace(nodeid=post + "::test_graph")
    assert not _size_included(
        deferred, "medium", "merge", candidate, floor_change, (), (), set(), floor=True
    )
    assert _size_included(
        deferred,
        "medium",
        "merge",
        candidate,
        floor_change,
        (deferred.nodeid,),
        (),
        set(),
        floor=True,
    )  # explicit proof still runs
    assert stage_files(candidate, "post-merge", [post], lane="merge") == [post]
    assert stage_files(candidate, "commit", [post], lane="candidate") == [post]
    assert stage_files(candidate, "commit", [post], lane="nightly") == [post]
    for bad in (
        "",
        "TEST_SCOPE = 'per-file'",
        "TEST_SCOPE = 'whole-tree'\nTEST_SCOPE = 'whole-tree'",
        "not python !",
    ):
        with pytest.raises(ValueError, match=r"test_size\.rule"):
            validate_post_merge_scopes(candidate, lambda _name, bad=bad: bad)
    with pytest.raises(ValueError, match=r"test_size\.rule"):
        validate_post_merge_scopes(
            candidate, lambda _name: (_ for _ in ()).throw(FileNotFoundError(post))
        )

    placement = getattr(layout, "new_test_file_problems", None)
    assert callable(placement), "new test-file placement owner is missing"
    rows = {"testing.example": {"owner": "devtools.testing", "layer": "unit"}}
    source = "src/devtools/testing/layout.py"
    body = """import pytest
@pytest.mark.proves('testing.example')
def test_one(): pass
"""
    declared = f"TEST_SOURCE = {source!r}\n" + body
    expected = "tests/unit/testing/test_layout.py"
    assert not placement(expected, declared, rows, {source})
    for path, text, reason in (
        ("tests/unit/testing/test_other.py", declared, expected),
        ("tests/integration/testing/test_layout.py", declared, expected),
        ("tests/unit/testing/test_layout.py", body, "TEST_SOURCE"),
        ("tests/unit/testing/test_layout.py", declared + f"TEST_SOURCE = {source!r}\n", "one"),
        (
            "tests/unit/testing/test_layout.py",
            declared + f"if True:\n    TEST_SOURCE = {source!r}\n",
            "one",
        ),
        ("tests/unit/testing/test_layout.py", declared.replace(source, "src/other.py"), "source"),
        ("tests/structural/test_tree.py", body, "TEST_SCOPE"),
    ):
        assert reason in " ".join(placement(path, text, rows, {source})), path
    assert not placement(
        "tests/structural/test_tree.py", "TEST_SCOPE = 'whole-tree'\n" + body, rows, {source}
    )
    nested = "src/alphalattice/capabilities/portfolio_inputs/tradability/surface.py"
    portfolio = {
        "testing.example": {
            "owner": "alphalattice.capabilities.portfolio_inputs",
            "layer": "integration",
        }
    }
    target = "tests/integration/portfolio_inputs/tradability/test_surface.py"
    assert not placement(target, declared.replace(source, nested), portfolio, {nested})
    script = "scripts/release/check_example.py"
    script_target = "tests/unit/testing/release/test_check_example.py"
    assert not placement(script_target, declared.replace(source, script), rows, {script})
    assert script_target in " ".join(
        placement(
            "tests/unit/testing/test_check_example.py",
            declared.replace(source, script),
            rows,
            {script},
        )
    )
    assert "source" in " ".join(
        placement(script_target, declared.replace(source, script), rows, set())
    )
    package = "src/devtools/testing/__init__.py"
    assert not placement(
        "tests/unit/testing/test_testing.py", declared.replace(source, package), rows, {package}
    )
    assert "mixed owners" in " ".join(
        placement(
            expected,
            declared.replace("testing.example", "testing.example', 'testing.other"),
            rows | {"testing.other": {"owner": "devtools.architecture", "layer": "unit"}},
            {source},
        )
    )
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
