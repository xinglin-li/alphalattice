"""The registries the gate holds, and the lookup that reads them (binding plan G0)."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from devtools.architecture.registries import main as registry_main
from devtools.architecture.registries import private_test_growth, problems, shrink, tracked
from devtools.architecture.where import main, where

ROOT = Path(__file__).resolve().parents[2]


def test_the_tree_adds_nothing_to_a_registry_without_registering_it() -> None:
    """The tree adds nothing to a registry without registering it."""

    assert problems(ROOT, tracked(ROOT)) == []


def test_each_kind_of_new_entry_is_refused_by_its_registry(tmp_path: Path) -> None:
    """Registering clears named refusals; shrinking cannot admit new entries."""

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


def test_staged_registry_census_refuses_new_unregistered_items(
    tmp_path, synthetic_history, capsys, monkeypatch
):
    monkeypatch.delenv("ALPHALATTICE_REGISTRY_BASELINE", raising=False)
    assert registry_main(["--staged"]) == 0 and "not configured" in capsys.readouterr().out
    git, commit = synthetic_history

    def write(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")

    def stage():
        git("-c", "core.longpaths=true", "add", "--all")

    def check(expected, *extra):
        try:
            code = registry_main(["--root", str(tmp_path), "--staged", *extra])
        except SystemExit as error:
            code = error.code
        captured = capsys.readouterr()
        assert code == expected, captured.out + captured.err
        return captured.out

    namespace = "alphalattice.interface.local_application"
    architecture = {
        "packages": [{"package_id": "local_application", "physical_namespace_target": namespace}]
    }
    owners = {"nouns": {namespace: {"owner": namespace, "defined_by": [], "folds_in": []}}}
    owners["nouns"]["census_registered_alias"] = owners["nouns"][namespace]
    client = "src/alphalattice/interface/local_application"
    operations = client + "/operations.json"
    refusals = "config/registries/refusals.json"
    legacy = client + "/legacy.py"
    baseline = "fixture-baseline.json"
    monkeypatch.setenv("ALPHALATTICE_REGISTRY_BASELINE", str(tmp_path / baseline))
    write("config/package-architecture.json", architecture)
    write("config/registries/owners-map.json", owners)
    write(refusals, {"entries": {}})
    write(operations, {"fields": {}, "purposes": {}, "commands": {}})
    write(legacy, "raise ValueError('census.legacy')\n")
    facts = {legacy: {"message-kinds|census.legacy": 1}}
    allowed = {"message-kinds|census.legacy": legacy + ":1"}
    declared = sorted(owners["nouns"])
    write(baseline, dict(by_file=facts, automatic_gaps=allowed, registered_nouns=declared))
    commit("Registered fixture baseline with one declared gap")
    check(0, "--baseline", baseline)
    fresh = "alphalattice.investment.census_noun"
    architecture["packages"].append(
        {"package_id": "census_noun", "physical_namespace_target": fresh}
    )
    write("config/package-architecture.json", architecture)
    module = "src/alphalattice/investment/census_noun/__init__.py"
    request = client + "/portfolio_research.py"
    sources = {
        module: "class CensusNoun: pass\ndef refuse(): raise ValueError('census.unregistered')\n",
        request: "type PortfolioResearchOperation = Literal['CENSUS_OPERATION']\n",
    }
    for path, text in sources.items():
        write(path, text)
    stage()
    for path in sources:
        write(path, "")
    owners["nouns"][fresh] = {"owner": fresh, "defined_by": [], "folds_in": []}
    registrations = {
        "config/registries/owners-map.json": owners,
        refusals: {"entries": {"census.unregistered": {"owner": fresh}}},
        operations: {"fields": {"CENSUS_OPERATION": {}}, "purposes": {}, "commands": {}},
    }
    for path, value in registrations.items():
        write(path, value)
    output = check(1)
    for dimension, name, target, form in (
        ("nouns", "census_noun", "config/registries/owners-map.json", "nouns"),
        ("operations", "CENSUS_OPERATION", "operations.json", "fields"),
        ("message-kinds", "census.unregistered", refusals, "entries"),
    ):
        assert any(
            all(part in line for part in (dimension, name, target, form))
            and re.search(r"(?:src|config)/[^:\s]+\.(?:py|json):\d+", line)
            for line in output.splitlines()
        ), (dimension, name, output)
    for path, text in sources.items():
        write(path, text)
    stage()
    check(0, "--update")
    stage()
    write(refusals, {"entries": {}})
    write(module, "raise ValueError('census.physical')\n")
    check(0)
    for path, text in sources.items():
        write(path, text)
    write(refusals, registrations[refusals])
    stage()
    git("-c", "core.longpaths=true", "commit", "-qm", "Registered source facts")
    owners["nouns"].pop("census_registered_alias")
    write("config/registries/owners-map.json", owners)
    stage()
    assert "census_registered_alias" in check(1)
    write(refusals, {"entries": {}})
    stage()
    assert "census.unregistered" in check(1)
    write(refusals, registrations[refusals])
    write(legacy, "raise ValueError('census.swap')\n")
    stage()
    assert "census.swap" in check(1)
    write(legacy, "")
    stage()
    assert baseline in check(1)
    check(0, "--update")
    stage()
    check(0)
    architecture["packages"].pop()
    write("config/package-architecture.json", architecture)
    stage()
    assert "packages[] with package_id and physical_namespace_target=" in check(1)


def test_the_tests_ratchet_refuses_an_entry_its_base_did_not_hold() -> None:
    """Private-access counts can fall or move, while new or duplicated accesses fail."""

    base = {"tests/test_a.py::import _kept": 2, "tests/test_b.py::patch _gone": 1}
    assert private_test_growth(base, {"tests/test_a.py::import _kept": 1}) == []
    grown = private_test_growth(
        base, {"tests/test_a.py::import _kept": 3, "tests/test_c.py::import _new": 1}
    )
    assert [line.split(" is registered")[0] for line in grown] == [
        "test-private: tests/test_a.py::import _kept",
        "test-private: tests/test_c.py::import _new",
    ]
    single = {"tests/test_a.py::import _kept": 1}
    assert private_test_growth(single, {"tests/test_c.py::import _kept": 1}) == []
    duplicated = private_test_growth(single, {**single, "tests/test_c.py::import _kept": 1})
    assert duplicated and all("::import _kept is registered 2" in line for line in duplicated)
    assert private_test_growth(single, {"tests/test_c.py::import _new": 1})


def test_a_format_that_names_no_reader_or_a_missing_one_is_refused(tmp_path: Path) -> None:
    """Formats name versions, readers and upgraders; missing contracts or files fail."""

    registries = tmp_path / "config" / "registries"
    registries.mkdir(parents=True)
    shutil.copy(ROOT / "config" / "package-architecture.json", tmp_path / "config")
    reader = tmp_path / "src" / "alphalattice" / "desk.py"
    reader.parent.mkdir(parents=True)
    reader.write_text("", encoding="utf-8")
    entries = {
        f"task_kind:{name}": {
            "owner": f"task_kind:{name}",
            "version": [f"{name}-input"],
            "readers": [f"src/alphalattice/{file}"],
            "upgraders": [],
        }
        for name, file in (("named", "desk.py"), ("gone", "gone.py"))
    }
    entries["task_kind:bare"] = {"owner": "task_kind:bare"}
    (registries / "formats.json").write_text(json.dumps({"entries": entries}), encoding="utf-8")
    refused = problems(tmp_path, [])
    assert not any("task_kind:named" in line for line in refused), refused
    assert "formats: task_kind:bare names no version or upgraders" in refused
    assert "formats: task_kind:bare names no reader" in refused
    assert "formats: task_kind:gone names src/alphalattice/gone.py, which is not a file" in refused


def test_where_answers_a_noun_from_the_registries(tmp_path: Path) -> None:
    """The where command resolves a requested noun from the owner registries."""

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
    """The owner map names existing registered owner packages and defining modules."""

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
    contracts.write_bytes(b"class StrategyAdmission: pass\nclass InputAdmission: pass\n")
    document = {
        "schema": "alphalattice.playpen.registry.owners-map",
        "version": 1,
        "note": "",
        "nouns": {
            "portfolio research": {
                "covers": "strategies",
                "owner": "alphalattice.investment.lab",
                "defined_by": ["alphalattice.investment.lab.contracts"],
                "input_contracts": ["InputAdmission"],
                "folds_in": [],
            }
        },
    }
    (root / "config" / "registries" / "owners-map.json").write_text(
        json.dumps(document), encoding="utf-8"
    )


def test_a_decision_is_refused_outside_its_owner_naming_the_owner(tmp_path: Path) -> None:
    """Refuse foreign decisions and seals; permit caller inputs, owner work and reads."""

    _owner_tree(tmp_path)
    risk = tmp_path / "src" / "alphalattice" / "investment" / "risk" / "model.py"
    risk.parent.mkdir(parents=True)
    producer = (
        "from alphalattice.investment.lab.contracts import StrategyAdmission\n\n\n"
        "def admit_strategy():\n"
        "    return StrategyAdmission()\n"
    )
    risk.write_text(producer, encoding="utf-8")
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
    lab.write_text(producer, encoding="utf-8")
    lab_file = ["src/alphalattice/investment/lab/admission.py"]
    assert not any("StrategyAdmission" in line for line in problems(tmp_path, lab_file))
    uses = "from alphalattice.investment.lab.contracts import InputAdmission\n"
    for statement, refused in (
        ("value = InputAdmission()", False),
        ("value = InputAdmission.create()", True),
        ("value = seal(InputAdmission)", True),
        ("class Foreign(InputAdmission): pass", True),
    ):
        risk.write_text(uses + statement + "\n", encoding="utf-8")
        assert any("makes InputAdmission" in x for x in problems(tmp_path, files)) is refused


def test_a_new_host_module_is_refused_unless_registered_as_composition(tmp_path: Path) -> None:
    """The Host admits a new module only when registered as composition."""

    _owner_tree(tmp_path)
    host = tmp_path / "src" / "alphalattice" / "control" / "product_host" / "composition"
    host.mkdir(parents=True)
    (host / "new_rule.py").write_text("RULE = True\n", encoding="utf-8")
    files = ["src/alphalattice/control/product_host/composition/new_rule.py"]
    assert any("a new module in the Host" in line for line in problems(tmp_path, files))
    registry = tmp_path / "config" / "registries" / "host-modules.json"
    registry.write_text(json.dumps({"entries": {files[0]: "COMPOSITION"}}), encoding="utf-8")
    assert not any("a new module in the Host" in line for line in problems(tmp_path, files))
