"""Find usages before a rename: the recorded escapes of this class, each refused and fixed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.architecture.moved_names import moved_name_problems, problems
from devtools.testing.layout import apply, inverse, plan

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
    }.items():
        (tmp_path / path).parent.mkdir(exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8", newline="\n")
    commit("Consumer")
    (tmp_path / "src/where.py").write_text("def new(): ...\n", encoding="utf-8", newline="\n")
    git("add", "src/where.py")
    (tmp_path / "tests/use.py").write_text(
        "from where import new\n", encoding="utf-8", newline="\n"
    )
    assert problems(tmp_path, staged=True)[0].startswith("tests/use.py:1: where.old")
    assert problems(tmp_path, staged=False) == []


@pytest.mark.proves("testing.whole_file_relocation_restores_bytes")
def test_whole_file_test_layout_restores_bytes_and_refuses_clashes(tmp_path):
    """Whole-file layout restores bytes and refuses path, export and fixture-name clashes."""
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
        "tests-unregistered.json": {"files": {"tests/a/test_t.py": 1}},
    }
    catalogs["test-sizes.json"]["defaults"] += [{"pattern": "tests/*", "owner": "fallback"}]
    for kind, key in (("large", "pattern"), ("reach", "pattern"), ("quarantine", "node")):
        catalogs["test-sizes.json"][kind] = [
            {key: "tests/a/test_t.py::case*", "inputs": ["tests/a/test_t.py"]}
        ]
    files |= {f"config/registries/{n}": json.dumps(v) for n, v in catalogs.items()}
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes(text.encode("utf-8"))
    choice = ["unit", "a", "t"]
    move = plan("r", files, {"tests/a/test_t.py": choice})
    assert "project_discovery" not in move["files"]["tests/a/conftest.py"][1]
    apply(tmp_path, move)
    assert (tmp_path / "tests/unit/a/test_t.py").read_text() == body
    for name, values in catalogs.items():
        wanted = json.dumps(values).replace("tests/a/test_t.py", "tests/unit/a/test_t.py")
        wanted = wanted.replace("tests/a/*", "tests/unit/a/test_t.py*")
        wanted = json.loads(wanted)
        assert json.loads((tmp_path / f"config/registries/{name}").read_text()) == wanted
    apply(tmp_path, inverse(move))
    assert {p: (tmp_path / p).read_text() for p in files} == files
    for stem in ("t", "other", "../escape"):
        with pytest.raises(ValueError, match=r"collision|fixture|path"):
            plan(
                "r", files, {"tests/a/test_t.py": choice, "tests/b/test_t.py": [*choice[:2], stem]}
            )
    collision = dict.fromkeys(("tests/a/test_t.py", "tests/unit/a/test_t.py"), 1)
    files["config/registries/test-private.json"] = json.dumps({"entries": collision})
    with pytest.raises(ValueError, match="registry_collision"):
        plan("r", files, {"tests/a/test_t.py": choice})
