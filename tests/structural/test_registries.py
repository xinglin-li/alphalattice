"""The registries the gate holds, and the lookup that reads them (binding plan G0)."""

from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path

import pytest

from devtools.architecture.registries import main as registry_main
from devtools.architecture.registries import private_test_growth, problems, shrink, tracked
from devtools.architecture.where import main, where

ROOT = Path(__file__).resolve().parents[2]


def refusal_registry(code, owner):
    """One synthetic code family, independent of the installed catalog."""
    category = "INVALID_ARGUMENT"
    return {
        "categories": [category, "FAILED_PRECONDITION"],
        "classification": {
            "rules": [
                {
                    "id": "fixture",
                    "owners": [owner],
                    "pattern": "^" + re.escape(code) + "$",
                    "category": category,
                }
            ],
            "exceptions": {},
        },
        "entries": {code: {"owner": owner, "category": category}},
    }


def test_person_questions_require_bilingual_words_reason_scope_and_way_on(tmp_path: Path) -> None:
    """The catalog refuses a person question with an incomplete language form."""
    path = tmp_path / "config/registries/messages.json"
    path.parent.mkdir(parents=True)
    shutil.copy(ROOT / "config/package-architecture.json", tmp_path / "config")
    for missing in ("words", "reason", "scope", "way_on"):
        entry = {
            name: {"en": "English", "zh-CN": "中文"}
            for name in ("words", "reason", "scope", "way_on")
        }
        entry.update(
            owner="alphalattice.interface.local_application",
            operations=["NETWORK_ACCESS_SET"],
            kind="PERSON_QUESTION",
        )
        entry[missing] = {"en": "English"}
        path.write_text(
            json.dumps({"entries": {"network.workspace_access": entry}}), encoding="utf-8"
        )
        assert any(
            "messages: network.workspace_access." + missing in problem
            for problem in problems(tmp_path, ["config/registries/messages.json"])
        )


def test_the_tree_adds_nothing_to_a_registry_without_registering_it() -> None:
    """The tree adds nothing to a registry without registering it."""

    assert problems(ROOT, tracked(ROOT)) == []


def test_each_kind_of_new_entry_is_refused_by_its_registry(tmp_path: Path) -> None:
    """Categories and disjoint word selectors require their declared owner shapes."""

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
    path = tmp_path / "config/registries/refusals.json"
    registry = refusal_registry("desk.window_invalid", "desk")
    path.write_text(json.dumps(registry), encoding="utf-8")
    for value in ({}, {"category": "UNDECLARED_CATEGORY"}):
        invalid = json.loads(json.dumps(registry))
        invalid["entries"]["desk.window_invalid"] = {"owner": "desk", **value}
        path.write_text(json.dumps(invalid), encoding="utf-8")
        assert any("refusals:" in p and "category" in p for p in problems(tmp_path, files))
    invalid = json.loads(json.dumps(registry))
    invalid["entries"]["desk.window_invalid"]["category"] = "FAILED_PRECONDITION"
    invalid["classification"]["exceptions"]["desk.window_invalid"] = {
        "owner": "desk",
        "category": "FAILED_PRECONDITION",
        "reason": "Reviewed exact override",
    }
    path.write_text(json.dumps(invalid), encoding="utf-8")
    shrink(tmp_path, files, seed=True)
    assert json.loads(path.read_text("utf-8")) == {
        **invalid,
        "schema": "alphalattice.playpen.registry.refusals",
        "note": "",
    }
    notice = json.loads(json.dumps(registry))
    notice["entries"]["desk.window_invalid"] = {"owner": "desk", "kind": "notice"}
    path.write_text(json.dumps(notice), encoding="utf-8")
    assert any("notice exemption" in p for p in problems(tmp_path, files))
    notice["classification"]["notices"] = {
        "desk.window_invalid": {"owner": "desk", "reason": "Named synthetic notice"}
    }
    path.write_text(json.dumps(notice), encoding="utf-8")
    assert not any("refusals:" in p for p in problems(tmp_path, files))
    selector = {"owner": "desk", "reason": "Word lookup only"}
    notice["legacy_unregistered_words"] = ["legacy"]
    for selectors in (
        None,
        {"word": {}},
        {"word": {**selector, "category": "INTERNAL"}},
        {"desk.window_invalid": selector},
        {"legacy": selector},
    ):
        notice["classification"]["word_selectors"] = selectors
        path.write_text(json.dumps(notice), encoding="utf-8")
        assert any("word selector" in p for p in problems(tmp_path, files))
    notice["classification"].pop("word_selectors")
    notice["entries"] = {}
    path.write_text(json.dumps(notice), encoding="utf-8")
    with pytest.raises(ValueError, match="register owner and category"):
        shrink(tmp_path, files, seed=True)
    path.write_text(json.dumps(invalid), encoding="utf-8")
    assert [line for line in problems(tmp_path, files) if "semantic_owner" not in line] == []
    registered = json.loads((tmp_path / "config/registries/parameters.json").read_text("utf-8"))
    assert registered["entries"]["src/alphalattice/investment/desk/model.py::WINDOW::21"] == {
        "class": "METHOD",
        "count": 1,
    }


def test_staged_registry_census_refuses_new_unregistered_items(
    tmp_path, synthetic_history, capsys, monkeypatch
):
    """Staged facts need registration and a purpose sentence; selectors never become reasons."""
    monkeypatch.delenv("ALPHALATTICE_REGISTRY_BASELINE", raising=False)
    assert registry_main(["--staged"]) == 0 and "not configured" in capsys.readouterr().out
    git, commit = synthetic_history

    def write(relative, value):
        if relative == "config/package-architecture.json":
            value = {"schema": "alphalattice.playpen.package-architecture", **value}
            value["packages"] = [
                {
                    "product_area": "kernel",
                    "capability_owner": "Census fixture",
                    "role": "DETERMINISTIC_CAPABILITY",
                    "dependency_level": 0,
                    "authority_kind": "TEST",
                    "allowed_peer_dependencies": [],
                    "artifact_writer": False,
                    **package,
                }
                for package in value["packages"]
            ]
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
    owners = {
        "nouns": {
            namespace: {
                "owner": namespace,
                "defined_by": [],
                "folds_in": [],
                "description": "Own the fixture's application contracts.",
            }
        }
    }
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
    caches = "config/registries/caches.json"
    debt = caches + "::entries:legacy"
    cache_rows = {"entries": {"legacy": {}}}
    write(caches, cache_rows)
    write(legacy, "raise ValueError('census.legacy')\n")
    facts = {legacy: {"message-kinds|census.legacy": 1}}
    allowed = {"message-kinds|census.legacy": legacy + ":1"}
    declared = sorted(owners["nouns"])
    write(
        baseline,
        dict(
            by_file=facts,
            automatic_gaps=allowed,
            registered_nouns=declared,
            undescribed_registry_entries=[debt],
        ),
    )
    commit("Registered fixture baseline with one declared gap")
    check(0, "--baseline", baseline)
    cache_rows["entries"]["new"] = {}
    write(caches, cache_rows)
    stage()
    assert "description" in check(1)
    current = json.loads((tmp_path / baseline).read_text("utf-8"))
    current["undescribed_registry_entries"].append(caches + "::entries:new")
    write(baseline, current)
    stage()
    assert "description" in check(1, "--update")
    for invalid in (None, "", "   ", "Two sentences. More text.", "Two\nlines."):
        cache_rows["entries"]["new"]["description"] = invalid
        write(caches, cache_rows)
        stage()
        assert "description" in check(1)
    cache_rows["entries"]["new"]["description"] = "Cache fixture values until the input changes."
    cache_rows["entries"]["legacy"]["description"] = "Retain the fixture lookup for its caller."
    write(caches, cache_rows)
    stage()
    check(0, "--update")
    stage()
    git("commit", "-qm", "Explain registry entries and retire description debt")
    cache_rows["entries"]["legacy"] = {}
    write(caches, cache_rows)
    stage()
    assert "description" in check(1)
    cache_rows["entries"]["legacy"]["description"] = "Retain the fixture lookup for its caller."
    write(caches, cache_rows)
    stage()
    deferred = client + "/deferred.py"
    write(deferred, f"def load():\n    from {namespace} import legacy\n")
    stage()
    write(deferred, "pass\n")
    assert "unlisted-function-import" in check(1, "--baseline", baseline)
    stage()
    check(0, "--baseline", baseline, "--update")
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
    owners["nouns"][fresh] = {
        "owner": fresh,
        "defined_by": [],
        "folds_in": [],
        "description": "Own the fixture's newly registered contracts.",
    }
    registrations = {
        "config/registries/owners-map.json": owners,
        refusals: refusal_registry("census.unregistered", fresh),
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
    words = "src/alphalattice/interface/local_application/refusal_words.json"
    declared_refusals = registrations[refusals]
    declared_refusals["legacy_unregistered_words"] = ["census.legacy_word"]
    selectors = {
        key: {"owner": fresh, "reason": "Word lookup only"}
        for key in (
            "census.selector",
            "local_client.problem_report_ready",
            "alternative_evidence.unit_failed",
        )
    }
    declared_refusals["classification"]["word_selectors"] = selectors

    def word_check(expected, document=declared_refusals, keys=()):
        write(refusals, document)
        write(words, {key: {} for key in ["census.legacy_word", *keys]})
        stage()
        return check(expected)

    word_check(0, keys=selectors)
    git("-c", "core.longpaths=true", "commit", "-qm", "Reviewed word declarations")
    for key in ("local_client.problem_report_ready", "alternative_evidence.unit_failed"):
        promoted = json.loads(json.dumps(declared_refusals))
        promoted["classification"]["word_selectors"].pop(key)
        promoted["entries"][key] = {"owner": fresh, "category": "INVALID_ARGUMENT"}
        assert key in word_check(1, promoted, selectors)
    assert "census.new_word" in word_check(1, keys=["census.new_word"])
    for target, key in (
        (declared_refusals, "legacy_unregistered_words"),
        (declared_refusals["classification"], "word_selectors"),
    ):
        before = target[key]
        if isinstance(before, list):
            target[key] = [*before, "census.new_word"]
        else:
            target[key] = {**before, "census.new_word": selectors["census.selector"]}
        assert "census.new_word" in word_check(1, keys=["census.new_word"])
        target.pop(key)
        assert "declaration" in word_check(1)
        target[key] = before
    registered = json.loads(json.dumps(declared_refusals))
    registered["entries"]["census.new_word"] = {"owner": fresh, "category": "INVALID_ARGUMENT"}
    word_check(0, registered, ["census.new_word"])
    registered["entries"]["census.legacy_word"] = registered["entries"]["census.new_word"]
    registered["legacy_unregistered_words"] = []
    registered["classification"]["word_selectors"] = {}
    word_check(0, registered)
    word_check(0)
    for value in ({}, {"category": "UNDECLARED_CATEGORY"}):
        invalid = json.loads(json.dumps(registrations[refusals]))
        invalid["entries"]["census.unregistered"] = {"owner": fresh, **value}
        write(refusals, invalid)
        stage()
        assert "refusals:" in check(1)
    write(refusals, registrations[refusals])
    stage()
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
    job_owners = {
        path.relative_to(ROOT).as_posix()
        for folder in ("src", "scripts/agent_eval/runners")
        for path in (ROOT / folder).rglob("*.py")
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "CreateJobObjectW"
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        )
    }
    assert job_owners == {"src/alphalattice/control/product_host/composition/process_job.py"}


def test_message_ids_keep_chinese_when_english_changes_and_refuse_invalid_forms(
    tmp_path, synthetic_history, monkeypatch, capsys
):
    """Stable ids retain Chinese; locale, argument, alias and duplicate defects refuse export."""
    monkeypatch.delenv("ALPHALATTICE_REGISTRY_BASELINE", raising=False)
    git, _ = synthetic_history
    path = tmp_path / "config/registries/messages.json"
    path.parent.mkdir(parents=True)
    entry = dict(
        owner="alphalattice.interface.local_application",
        kind="MESSAGE",
        arguments=["item"],
        words={"en": "Ready {item}", "zh-CN": "就绪 {item}"},
    )

    def check(entries, expected, *, raw=None, diagnostic=""):
        path.write_text(raw or json.dumps({"entries": entries}), encoding="utf-8")
        git("add", "--all")
        assert registry_main(["--root", str(tmp_path), "--staged"]) == expected
        assert diagnostic in capsys.readouterr().out

    check({"job.ready": entry}, 0)
    entry["words"]["en"] = "Prepared {item}"
    check({"job.ready": entry}, 0)  # Copy changes do not change message identity.
    for locale in ("en", "zh-CN"):
        for value in (None, "", " ", "{other}", "{}", "{item!r}", "{item:04}"):
            words = entry["words"] | {locale: value}
            check({"job.ready": entry | {"words": words}}, 1, diagnostic="job.ready")
    for field, value in (
        ("owner", ""),
        ("kind", "OTHER"),
        ("kind", []),
        ("arguments", []),
        ("arguments", ["item", "item"]),
        ("legacy_keys", ["", "old"]),
        ("legacy_keys", ["old", "old"]),
        ("operations", ["ANSWER"]),
    ):
        check({"job.ready": entry | {field: value}}, 1, diagnostic="job.ready")
    alias = entry | {"legacy_keys": ["panel|Ready"]}
    waiting = entry | {"words": {"en": "Waiting {item}", "zh-CN": "等待 {item}"}}
    check({"job.ready": alias, "job.waiting": waiting | {"legacy_keys": ["task|Ready"]}}, 0)
    for words in (entry["words"], entry["words"] | {"zh-CN": "等待 {item}"}):
        check({"job.ready": entry, "job.waiting": entry | {"words": words}}, 1)
    check({"job.ready": alias, "job.waiting": alias}, 1, diagnostic="panel|Ready")
    check({"job.ready": entry | {"legacy_keys": ["job.waiting"]}, "job.waiting": entry}, 1)
    member = json.dumps(entry)
    check(
        {},
        1,
        raw='{"entries":{"job.ready":' + member + ',"job.ready":' + member + "}}",
        diagnostic="job.ready",
    )
    check({}, 1, raw='{"entries":{},"entries":{}}', diagnostic="entries")
    check({}, 1, raw='{"entries":[]}', diagnostic="entries")
