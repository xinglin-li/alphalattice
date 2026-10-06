"""The registries the gate holds, and the lookup that reads them (binding plan G0)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from devtools.architecture.registries import private_test_growth, problems, shrink, tracked
from devtools.architecture.where import main, where

ROOT = Path(__file__).resolve().parents[2]


def test_the_tree_adds_nothing_to_a_registry_without_registering_it() -> None:
    """requirement (binding plan G0; LAWS.md PA1, DA6, TE5, OW1): the whole tree against the
    registries' baselines: no unregistered parameter, failure code, format or Task kind, no new
    private import or patch in a test, no semantic owner that is not a package, and no package
    whose product area is off its namespace, beyond what the baselines hold."""

    assert problems(ROOT, tracked(ROOT)) == []


def test_each_kind_of_new_entry_is_refused_by_its_registry(tmp_path: Path) -> None:
    """requirement (G0): a new literal, code, schema, private test import and unregistered owner are
    each refused with the registry that takes it; registering them clears the refusal, and a
    baseline never grows by shrinking."""

    (tmp_path / "config" / "registries").mkdir(parents=True)
    shutil.copy(ROOT / "config" / "package-architecture.json", tmp_path / "config")
    module = tmp_path / "src" / "alphalattice" / "investment" / "desk" / "model.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        'WINDOW = 21\nSCHEMA = {"schema": "alphalattice.desk.model"}\n\n\n'
        "def fit():\n"
        '    raise ValueError("desk.window_invalid")\n\n\n'
        'IDENTITY = identity(semantic_owner="no_such_package")\n',
        encoding="utf-8",
    )
    test = tmp_path / "tests" / "test_desk.py"
    test.parent.mkdir()
    test.write_text("from alphalattice.investment.desk.model import _private\n", encoding="utf-8")
    files = ["src/alphalattice/investment/desk/model.py", "tests/test_desk.py"]

    refused = problems(tmp_path, files)
    for expected in (
        "parameters: src/alphalattice/investment/desk/model.py::WINDOW::21",
        "refusals: desk.window_invalid",
        "formats: schema:alphalattice.desk.model",
        "test-private: tests/test_desk.py::import _private",
        "semantic_owner 'no_such_package'",
    ):
        assert any(expected in line for line in refused), (expected, refused)

    assert shrink(tmp_path, files)["parameters"] == 0  # shrinking never registers
    shrink(tmp_path, files, seed=True)
    assert [line for line in problems(tmp_path, files) if "semantic_owner" not in line] == []
    registered = json.loads((tmp_path / "config/registries/parameters.json").read_text("utf-8"))
    assert registered["entries"]["src/alphalattice/investment/desk/model.py::WINDOW::21"] == {
        "class": "METHOD",
        "count": 1,
    }


def test_the_tests_ratchet_refuses_an_entry_its_base_did_not_hold() -> None:
    """regression (V16, LAWS.md TE5): five private imports passed the gate on 2026-09-30, each
    registered by hand beside its test; against the registry a change starts from, a new entry
    or a raised count is refused, and an entry kept or lowered is not."""

    base = {"tests/test_a.py::import _kept": 2, "tests/test_b.py::patch _gone": 1}
    assert private_test_growth(base, {"tests/test_a.py::import _kept": 1}) == []
    grown = private_test_growth(
        base, {"tests/test_a.py::import _kept": 3, "tests/test_c.py::import _new": 1}
    )
    assert [line.split(" is registered")[0] for line in grown] == [
        "test-private: tests/test_a.py::import _kept",
        "test-private: tests/test_c.py::import _new",
    ]


def test_a_format_that_names_no_reader_or_a_missing_one_is_refused(tmp_path: Path) -> None:
    """requirement (LAWS.md DA6, V267): a persisted format names its version, its readers and
    its upgraders; an entry that names none, or a reader that is not a file, is refused."""

    registries = tmp_path / "config" / "registries"
    registries.mkdir(parents=True)
    shutil.copy(ROOT / "config" / "package-architecture.json", tmp_path / "config")
    reader = tmp_path / "src" / "alphalattice" / "desk.py"
    reader.parent.mkdir(parents=True)
    reader.write_text("", encoding="utf-8")
    entries = {
        "task_kind:named": {
            "owner": "task_kind:named",
            "version": ["named-input"],
            "readers": ["src/alphalattice/desk.py"],
            "upgraders": [],
        },
        "task_kind:bare": {"owner": "task_kind:bare"},
        "task_kind:gone": {
            "owner": "task_kind:gone",
            "version": ["gone-input"],
            "readers": ["src/alphalattice/gone.py"],
            "upgraders": [],
        },
    }
    (registries / "formats.json").write_text(json.dumps({"entries": entries}), encoding="utf-8")
    refused = problems(tmp_path, [])
    assert not any("task_kind:named" in line for line in refused), refused
    assert "formats: task_kind:bare names no version or upgraders" in refused
    assert "formats: task_kind:bare names no reader" in refused
    assert "formats: task_kind:gone names src/alphalattice/gone.py, which is not a file" in refused


def test_where_answers_a_noun_from_the_registries(tmp_path: Path) -> None:
    """requirement (G0): `devtools where <noun>` answers from the registries, before code is
    written: the package that owns the noun, its owner in G1's map with what other packages still
    define of it and the map's note on what stays elsewhere, the operations and identity roles
    that name it."""

    lines = where(ROOT, "task")
    assert any(line.startswith("package task_control:") for line in lines), lines
    assert any(line.startswith("operation ") for line in lines), lines
    assert any(
        line.startswith("owner of task: alphalattice.control.task_control") for line in lines
    ), lines
    assert any(line.startswith("  note: ") for line in lines), lines
    assert where(ROOT, "no-such-noun-anywhere") == [
        "nothing registered names no-such-noun-anywhere"
    ]
    assert main(["task"]) == 0

    # The live map folds nothing in since O3; what another package still defines is shown when
    # a map names it.
    (tmp_path / "config" / "registries").mkdir(parents=True)
    for name in ("package-architecture.json", "identity-roles.json"):
        shutil.copy(ROOT / "config" / name, tmp_path / "config" / name)
    document = json.loads((ROOT / "config/registries/owners-map.json").read_text(encoding="utf-8"))
    document["nouns"]["task"]["folds_in"] = [
        {"from": "alphalattice.control.product_host.example", "what": "a decision", "card": "X"}
    ]
    (tmp_path / "config/registries/owners-map.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    assert "  still defined in alphalattice.control.product_host.example: a decision (X)" in where(
        tmp_path, "task"
    )


def test_the_owner_map_names_what_the_tree_has() -> None:
    """requirement (G1): the owner map names only what exists -- each owner inside a registered
    package, or created by the card it names, and each module it lists -- so a move that leaves
    the map behind fails here rather than in G2's gate."""

    document = json.loads((ROOT / "config/registries/owners-map.json").read_text(encoding="utf-8"))
    namespaces = {
        package["physical_namespace_target"]
        for package in json.loads(
            (ROOT / "config/package-architecture.json").read_text(encoding="utf-8")
        )["packages"]
    }

    def exists(dotted: str) -> bool:
        base = ROOT.joinpath("src", *dotted.split("."))
        return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()

    stale = []
    for noun, row in document["nouns"].items():
        owner = row["owner"]
        if "created_by" not in row and not (
            exists(owner) and any(owner == n or owner.startswith(n + ".") for n in namespaces)
        ):
            stale.append(f"{noun}: owner {owner}")
        named = [
            *row["defined_by"],
            *row.get("with", ()),
            *row.get("retiring", ()),
            *(item["from"] for item in row["folds_in"]),
        ]
        stale += [f"{noun}: {module}" for module in named if not exists(module)]
    assert not stale, stale


def _owner_tree(root: Path) -> None:
    """A tree with two owners: a lab that decides admissions, a Risk desk, and a Host."""

    (root / "config" / "registries").mkdir(parents=True)
    shutil.copy(ROOT / "config" / "package-architecture.json", root / "config")
    contracts = root / "src" / "alphalattice" / "investment" / "lab" / "contracts.py"
    contracts.parent.mkdir(parents=True)
    contracts.write_text("class StrategyAdmission:\n    pass\n", encoding="utf-8")
    document = {
        "schema": "alphalattice.playpen.registry.owners-map",
        "version": 1,
        "note": "",
        "nouns": {
            "portfolio research": {
                "covers": "strategies",
                "owner": "alphalattice.investment.lab",
                "defined_by": ["alphalattice.investment.lab.contracts"],
                "folds_in": [],
            }
        },
    }
    (root / "config" / "registries" / "owners-map.json").write_text(
        json.dumps(document), encoding="utf-8"
    )


def test_a_decision_is_refused_outside_its_owner_naming_the_owner(tmp_path: Path) -> None:
    """requirement (G2, V178; LAWS.md OW1, OP12): a Portfolio admission added to an existing Risk
    module, its imports otherwise unchanged, is refused by the decision it makes, naming the
    owner to write it in; in the owner it passes, and reading the type anywhere is a use."""

    _owner_tree(tmp_path)
    risk = tmp_path / "src" / "alphalattice" / "investment" / "risk" / "model.py"
    risk.parent.mkdir(parents=True)
    risk.write_text(
        "from alphalattice.investment.lab.contracts import StrategyAdmission\n\n\n"
        "def admit_strategy():\n"
        "    return StrategyAdmission()\n",
        encoding="utf-8",
    )
    files = ["src/alphalattice/investment/risk/model.py"]
    refused = problems(tmp_path, files)
    assert any(
        "makes StrategyAdmission, a decision of portfolio research; write it in "
        "alphalattice.investment.lab" in line
        for line in refused
    ), refused
    risk.write_text(
        "from alphalattice.investment.lab.contracts import StrategyAdmission\n\n\n"
        "def read(value: StrategyAdmission) -> str:\n"
        "    return str(value)\n",
        encoding="utf-8",
    )
    assert not any("StrategyAdmission" in line for line in problems(tmp_path, files))
    lab = tmp_path / "src" / "alphalattice" / "investment" / "lab" / "admission.py"
    lab.write_text(
        "from alphalattice.investment.lab.contracts import StrategyAdmission\n\n\n"
        "def admit_strategy():\n"
        "    return StrategyAdmission()\n",
        encoding="utf-8",
    )
    lab_file = ["src/alphalattice/investment/lab/admission.py"]
    assert not any("StrategyAdmission" in line for line in problems(tmp_path, lab_file))


def test_a_new_host_module_is_refused_unless_registered_as_composition(tmp_path: Path) -> None:
    """requirement (G2; LAWS.md OW6): the Host keeps composition only, so a module added to it is
    refused until it is registered as the Host's composition."""

    _owner_tree(tmp_path)
    host = tmp_path / "src" / "alphalattice" / "control" / "product_host" / "composition"
    host.mkdir(parents=True)
    (host / "new_rule.py").write_text("RULE = True\n", encoding="utf-8")
    files = ["src/alphalattice/control/product_host/composition/new_rule.py"]
    assert any("a new module in the Host" in line for line in problems(tmp_path, files))
    registry = tmp_path / "config" / "registries" / "host-modules.json"
    registry.write_text(json.dumps({"entries": {files[0]: "COMPOSITION"}}), encoding="utf-8")
    assert not any("a new module in the Host" in line for line in problems(tmp_path, files))
