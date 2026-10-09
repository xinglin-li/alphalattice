from __future__ import annotations

import ast
import importlib
import json
from collections import Counter
from enum import StrEnum
from pathlib import Path

import pytest

from alphalattice.kernel.shared_kernel.identity_successors import (
    IdentitySuccessorError,
    is_current,
    latest,
    predecessors,
    recorded_moves,
    recorded_origin,
)
from alphalattice.kernel.shared_kernel.source_identity import (
    source_component_id_from_tracked_path,
)

ROOT = Path(__file__).resolve().parents[2]


def test_old_and_namespaced_paths_resolve_the_same_semantic_component() -> None:
    assert source_component_id_from_tracked_path(
        "src/alpha_modeling/runtime/service.py"
    ) == source_component_id_from_tracked_path(
        "src/alphalattice/capabilities/alpha_modeling/runtime/service.py"
    )
    assert (
        source_component_id_from_tracked_path(
            "src/alphalattice/capabilities/alpha_modeling/runtime/service.py"
        )
        == "alpha_modeling.runtime.service"
    )


_A, _B, _C, _D = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)


def _successors(root: Path, moves: list[dict[str, str]], *, version: int = 1) -> Path:
    path = root / "config" / "identity-successors.json"
    path.parent.mkdir(parents=True)
    document = {"schema": "identity-successors", "version": version, "moves": moves}
    path.write_text(json.dumps(document), encoding="utf-8")
    return root


def _move(role: str, predecessor: str, successor: str) -> dict[str, str]:
    return {
        "role": role,
        "predecessor": predecessor,
        "successor": successor,
        "change": "test",
        "reason": "test",
    }


def test_a_recorded_move_names_the_installed_authority_for_its_role_only(
    tmp_path: Path,
) -> None:
    root = _successors(
        tmp_path,
        [
            _move("role", _A, _B),
            _move("role", _B, _C),
            _move("other", _C, _D),
            _move("loop", _A, _B),
            _move("loop", _B, _A),
        ],
    )

    assert is_current("role", _C, _C, root=root)
    assert is_current("role", _A, _C, root=root), "two recorded moves lead to the installed value"
    assert not is_current("role", _C, _A, root=root), "a move is one way"
    assert not is_current("role", _C, _D, root=root), "another role's move is not this one's"
    assert not is_current("role", _D, _C, root=root), "an unrecorded value is historical"
    assert not is_current("loop", _A, _C, root=root), "a recorded cycle ends, not current"


def test_a_bound_value_is_held_at_the_origin_its_recorded_moves_lead_from(tmp_path: Path) -> None:
    """requirement (LAWS.md ID1, V324): a value bound where it is compared by equality is held
    at the value its recorded moves lead from, so a recorded move leaves it where it was and an
    unrecorded one is a new value; another role's moves and a recorded cycle end the walk."""

    root = _successors(
        tmp_path,
        [
            _move("role", _A, _B),
            _move("role", _B, _C),
            _move("other", _C, _D),
            _move("loop", _A, _B),
            _move("loop", _B, _A),
        ],
    )

    assert recorded_origin("role", _C, root=root) == _A
    assert recorded_origin("role", _A, root=root) == _A
    assert recorded_origin("role", _D, root=root) == _D, "an unrecorded value is its own"
    assert recorded_origin("other", _D, root=root) == _C, "each role follows its own moves"
    assert recorded_origin("loop", _A, root=root) in {_A, _B}, "a recorded cycle ends"


def test_a_keyed_store_finds_what_was_sealed_before_a_recorded_move(tmp_path: Path) -> None:
    """requirement (LAWS.md ID1, W9a): a store keyed by a recorded value finds the entries
    sealed under a value recorded moves lead from -- through ``latest`` on both sides of a
    key, or ``predecessors`` of the installed value -- and no other role's."""

    root = _successors(
        tmp_path,
        [_move("role", _A, _B), _move("role", _B, _C), _move("other", _D, _C)],
    )

    assert latest("role", _A, root=root) == latest("role", _C, root=root) == _C
    assert latest("role", _D, root=root) == _D, "another role's move is not this one's"
    assert predecessors("role", _C, root=root) == (_C, _B, _A), "nearest first"
    assert predecessors("role", _A, root=root) == (_A,), "a move is one way"
    assert all(
        is_current("role", value, _C, root=root) for value in predecessors("role", _C, root=root)
    )


@pytest.mark.parametrize(
    ("moves", "version", "code"),
    [
        ([_move("role", _A, _B)], 2, "shared_kernel.identity_successors_from_another_build"),
        ([_move("role", _A, _A)], 1, "shared_kernel.identity_successor_invalid"),
        ([_move("role", _A, "not-a-hash")], 1, "shared_kernel.identity_successor_invalid"),
        ([{"role": "role", "predecessor": _A}], 1, "shared_kernel.identity_successor_invalid"),
    ],
)
def test_moves_this_build_cannot_read_are_refused_not_ignored(
    tmp_path: Path, moves: list[dict[str, str]], version: int, code: str
) -> None:
    root = _successors(tmp_path, moves, version=version)

    with pytest.raises(IdentitySuccessorError, match=code):
        is_current("role", _A, _B, root=root)


def test_the_committed_moves_read_and_each_predecessor_moves_once_per_role() -> None:
    moves = recorded_moves(ROOT)

    assert moves
    twice = [key for key, n in Counter((m.role, m.predecessor) for m in moves).items() if n > 1]
    assert not twice, f"a predecessor recorded with two successors: {twice}"


def test_study_identities_track_no_platform_or_build_file() -> None:
    """Binding plan B5: a study's replay compares its kind's implementation closure; a
    composition, storage, Task or build file there made an edit that decides no number
    refuse every saved study of the kind."""

    from devtools.architecture.identity_closures import closure_sites, path_class, report

    study = {
        site.numerical_role: site
        for site in closure_sites(ROOT)
        if site.numerical_role
        in {
            "FACTOR_DEVELOPMENT_EXPERIMENT | ALPHA_DEVELOPMENT_EXPERIMENT",
            "RISK_DEVELOPMENT",
            "RISK_DEVELOPMENT_HANDOFF",
        }
    }

    assert len(study) == 3 and all(site.tracked_paths for site in study.values())
    platform = sorted(
        f"{role}: {path}"
        for role, site in study.items()
        for path in site.tracked_paths or ()
        if path_class(path) != "DOMAIN"
    )
    assert not platform
    assert report(ROOT, ["src/alphalattice/foundation/factor_research/experiments/execution.py"])


def test_every_recorded_role_is_in_the_roles_table_and_resolves() -> None:
    """Every recorded role is in the roles table and resolves."""

    table = json.loads((ROOT / "config" / "identity-roles.json").read_text(encoding="utf-8"))
    roles = {entry["role"]: entry for entry in table["roles"]}
    assert len(roles) == len(table["roles"]), "a role listed twice"
    retired = {entry["role"]: entry for entry in table["retired"]}
    assert len(retired) == len(table["retired"]), "a role retired twice"
    assert not set(roles) & set(retired), "a retired role is still installed"
    for role, entry in retired.items():
        assert set(entry) == {"role", "retired_by", "reason"} and all(entry.values()), role
    assert {move.role for move in recorded_moves(ROOT)} <= set(roles) | set(retired)
    for role, entry in roles.items():
        assert entry["comparison"] in {"IS_CURRENT", "EXACT"}, role
        assert entry["closure"] in {"LISTED", "BY_RULE"}, role
        module_name, _, qualname = entry["installed"].partition(":")
        target = importlib.import_module(module_name)
        for part in qualname.split("."):
            target = getattr(target, part)
        assert callable(target), role
    architecture = ROOT / "config" / "package-architecture.json"
    registry = json.loads(architecture.read_text(encoding="utf-8"))
    packages = {package["package_id"] for package in registry["packages"]}
    assert set(table["number_deciding_packages"]) <= packages
    # The rule leaves out what B5's scanner classes as platform, and nothing else.
    from devtools.architecture.identity_closures import path_class

    for segment in table["platform_segments"]:
        assert path_class(f"src/alphalattice/investment/desk/{segment}/module.py") != "DOMAIN"
    assert path_class("src/alphalattice/investment/desk/models/module.py") == "DOMAIN"


def test_every_factor_value_has_a_readout_role_and_binds_its_recorded_origin() -> None:
    """Every factor value has a readout role and binds its recorded origin."""

    from alphalattice.foundation.feature_engine.catalog.contracts import (
        SOURCE_AVAILABILITY_ROLE,
        FeatureCatalog,
    )
    from alphalattice.foundation.feature_engine.catalog.observation_clock import (
        installed_source_availability_catalog,
    )
    from alphalattice.foundation.feature_engine.producers.arithmetic_identity import (
        FACTOR_METHOD_FAMILY_OWNERS,
        FACTOR_VALUE_ROLE,
        control_arithmetic_content_hash,
        control_arithmetic_rule_identity,
        method_family_content_hash,
        method_family_rule_identity,
    )

    table = json.loads((ROOT / "config" / "identity-roles.json").read_text(encoding="utf-8"))
    values = {
        entry["role"]: entry
        for entry in table["roles"]
        if entry["role"].startswith(f"{FACTOR_VALUE_ROLE}.")
    }
    assert set(values) == {
        f"{FACTOR_VALUE_ROLE}.{name}" for name in (*FACTOR_METHOD_FAMILY_OWNERS, "controls")
    }
    assert {(v["comparison"], v["closure"]) for v in values.values()} == {("IS_CURRENT", "BY_RULE")}
    assert control_arithmetic_content_hash() == recorded_origin(
        f"{FACTOR_VALUE_ROLE}.controls", control_arithmetic_rule_identity()
    )
    for family, owners in FACTOR_METHOD_FAMILY_OWNERS.items():
        assert method_family_content_hash(family, owners) == recorded_origin(
            f"{FACTOR_VALUE_ROLE}.{family}", method_family_rule_identity(family, owners)
        )
    assert FeatureCatalog.load().binding.source_availability_policy_hash == recorded_origin(
        SOURCE_AVAILABILITY_ROLE, installed_source_availability_catalog().catalog_hash
    )


def test_a_rule_closure_follows_imports_inside_the_deciding_packages_and_ignores_prose(
    tmp_path: Path,
) -> None:
    """A rule closure follows imports inside the deciding packages and ignores prose."""

    from alphalattice.kernel.shared_kernel.source_identity import (
        NumberDecidingRule,
        number_deciding_closure,
        source_rule_closure_hash,
    )

    def write(relative: str, text: str) -> Path:
        path = tmp_path / "src" / "alphalattice" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    for package in ("investment", "investment/desk", "investment/desk/storage", "control"):
        write(f"{package}/__init__.py", "")
    write(
        "control/host/entry.py",
        "from typing import TYPE_CHECKING\n"
        "from alphalattice.investment.desk.model import fit\n"
        "from alphalattice.control.host import other\n"
        "if TYPE_CHECKING:\n    from alphalattice.investment.desk import typing_only\n",
    )
    write("control/host/other.py", "X = 1\n")
    model = write(
        "investment/desk/model.py",
        '"""Fit."""\nfrom . import helper\nfrom .storage import reader\n\n\n'
        "def fit():\n    return 1\n",
    )
    write("investment/desk/helper.py", "Y = 2\n")
    write("investment/desk/storage/reader.py", "W = 4\n")
    write("investment/desk/typing_only.py", "Z = 3\n")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "identity-roles.json").write_text(
        json.dumps(
            {"number_deciding_packages": ["desk"], "platform_segments": ["storage"], "roles": []}
        ),
        encoding="utf-8",
    )
    entry = ("alphalattice.control.host.entry",)
    rule = NumberDecidingRule(frozenset({"desk"}), frozenset({"storage"}))
    closure = number_deciding_closure(entry, root=tmp_path, rule=rule)
    assert sorted(closure) == [
        "alphalattice.control.host.entry",
        "alphalattice.investment.desk",
        "alphalattice.investment.desk.helper",
        "alphalattice.investment.desk.model",
    ]

    def identity() -> str:
        return source_rule_closure_hash(
            root=tmp_path,
            tracked_paths=("src/alphalattice/control/host/entry.py",),
            semantic_owner="host",
            numerical_role="EXAMPLE",
        )

    before = identity()
    original = model.read_text(encoding="utf-8")
    model.write_text(
        original.replace('"""Fit."""', '"""Fit, in other words."""') + "# a comment\n",
        encoding="utf-8",
    )
    assert identity() == before
    model.write_text(original.replace("return 1", "return 2"), encoding="utf-8")
    assert identity() != before


def test_a_span_moves_no_identity_and_any_other_form_reads_as_written() -> None:
    """A span moves no identity and any other form reads as written."""

    from alphalattice.kernel.shared_kernel.source_identity import (
        source_bytes_syntax_sha256 as digest,
    )

    plain = (
        "import numpy as np\n\n\n"
        "def fit(x):\n    total = np.sum(x)\n    return total * 2\n\n\n"
        "class Model:\n    def score(self):\n        return 1\n"
    )
    measured = (
        "import numpy as np\n"
        "from alphalattice.kernel.shared_kernel.spans import span, spanned\n\n\n"
        "def fit(x):\n"
        '    with span("compute", detail="sum"):\n        total = np.sum(x)\n'
        '    with span("score"):\n        return total * 2\n\n\n'
        'class Model:\n    @spanned("predict", None)\n    def score(self):\n        return 1\n'
    )
    assert digest(measured.encode()) == digest(plain.encode())
    for written in (
        measured.replace("import span, spanned", "import span as timed, spanned").replace(
            'with span("score")', 'with timed("score")'
        ),
        measured.replace('span("score")', "span(stage)"),
        measured + "\nspan = None\n",
    ):
        assert digest(written.encode()) != digest(plain.encode())


def test_every_span_in_the_source_is_literal_and_names_a_category_its_ledger_counts() -> None:
    """requirement (A4): every span in the product names a category the ledger counts under a
    class, with literal arguments, so it lands in its ledger class and reads as its body in every
    identity that holds its module."""

    from alphalattice.kernel.shared_kernel.source_identity import SPAN_MODULE
    from alphalattice.kernel.shared_kernel.spans import SPAN_CLASSES

    found: list[str] = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if SPAN_MODULE not in text:
            continue
        tree = ast.parse(text)
        names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == SPAN_MODULE
            for alias in node.names
            if alias.name in {"span", "spanned"} and alias.asname is None
        }
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in names
            ):
                continue
            where = f"{path.relative_to(ROOT).as_posix()}:{node.lineno}"
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            assert all(
                isinstance(value, ast.Constant)
                and (value.value is None or isinstance(value.value, str))
                for value in arguments
            ), f"{where}: a span's arguments are literal text"
            assert node.args and node.args[0].value in SPAN_CLASSES, (  # type: ignore[attr-defined]
                f"{where}: unknown span category"
            )
            found.append(where)
    assert found


def test_an_area_reaching_a_role_outside_its_row_is_named_with_its_import() -> None:
    """requirement (GB, LAWS.md ID8): the reach table only shrinks, and a growth names the
    import that caused it."""

    import identity_readout

    recorded = {"a.b": ["role.one"], "a.c": ["role.one"]}
    areas = {"a.b": {"role.one", "role.two"}, "a.c": {"role.one"}}
    edges = {
        ("a.b", "role.one"): "a.b.m is an entry of the role",
        ("a.b", "role.two"): "a.d imports a.b.m",
        ("a.c", "role.one"): "a.b.m imports a.c.n",
    }
    assert identity_readout.reach_growth(areas, edges, recorded) == [
        "a.b now reaches role.two: a.d imports a.b.m"
    ]
    assert identity_readout.reach_growth({"a.c": {"role.one"}}, edges, recorded) == []


def test_the_recorded_reach_names_real_areas_and_roles() -> None:
    """requirement (GB): each row is a package of the tree, each role one the readout computes."""

    root = Path(__file__).resolve().parents[2]
    table = json.loads((root / "config/registries/identity-reach.json").read_text(encoding="utf-8"))
    roles = {
        str(entry["role"])
        for entry in json.loads((root / "config/identity-roles.json").read_text(encoding="utf-8"))[
            "roles"
        ]
    }
    assert table["areas"]
    for area, area_roles in table["areas"].items():
        assert (root / "src" / Path(*area.split("."))).is_dir(), area
        assert set(area_roles) <= roles, area


_ENVIRONMENT_READERS = {
    # The one helper (LAWS.md ID6).
    "src/alphalattice/kernel/shared_kernel/environment.py": "the helper",
    # Rebuild keys of caches whose content the environment decides, and packs it gates.
    "src/alphalattice/kernel/knowledge/_embeddings.py": ("an embedding index's rebuild key"),
    "src/alphalattice/kernel/knowledge/_reranking.py": "a reranker pack's rebuild key",
    "src/alphalattice/kernel/knowledge/_torch.py": "a torch pack's rebuild key",
    "src/alphalattice/kernel/knowledge/model_store.py": "a stored model's rebuild key",
    "src/alphalattice/kernel/knowledge/catalog.py": "a knowledge pack's installed distributions",
    "src/alphalattice/capabilities/alpha_modeling/runtime/lightgbm_threads.py": (
        "the LightGBM thread canary is sealed per installed version (PA3)"
    ),
    "src/alphalattice/foundation/market_data_ops/sources/providers.py": (
        "the library that fetched the observations, recorded with them"
    ),
    "src/alphalattice/evidence/alternative_evidence/runtime/execution.py": (
        "the machine a Task ran on, recorded beside its reuse (binding plan, N4)"
    ),
}
"""Every module that reads the interpreter, the platform or an installed version, and why.

The environment is provenance (LAWS.md ID6): the helper reads it and a result records it
beside itself. A module outside this table that reads it is refused by name; one that stops
reading it leaves the table in the same change (sets, not counts)."""


def _environment_readers(source: Path) -> set[str]:
    found: set[str] = set()
    platform_calls = {"python_version", "platform", "python_implementation", "python_version_tuple"}
    for path in sorted(source.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        version_names = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "importlib.metadata"
            for alias in node.names
            if alias.name == "version"
        }
        for node in ast.walk(tree):
            reads = False
            if isinstance(node, ast.Attribute):
                text = ast.unparse(node)
                reads = (
                    node.attr == "__version__"
                    or text in {"sys.version", "sys.version_info"}
                    or (text.startswith("platform.") and node.attr in platform_calls)
                    or text.endswith("metadata.version")
                )
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                reads = node.func.id in version_names
            if reads:
                found.add(path.relative_to(ROOT).as_posix())
                break
    return found


def test_u0_records_an_opened_reads_numbers_and_names_the_sections_that_moved() -> None:
    """regression (V277): U0 compared each read's verdict, so a result whose `rank_ic` moved from
    0.01 to 0.9 read OPENS both times; an opened read records its numbers by section, what the
    clock decides left out, and the comparison names each section that moved."""

    from scripts.u0_probe import moved_numbers, numbers

    def answer(rank_ic: float, seconds: float) -> bytes:
        data = {
            "status": "SUCCEEDED",
            "result": {"rank_ic": rank_ic, "reused": True},
            "elapsed_seconds": seconds,
            "timing": {"wall": seconds},
        }
        return json.dumps({"data": data}).encode("utf-8")

    before = numbers(200, answer(0.01, 1.0))
    assert list(before) == ["result.rank_ic"]
    assert moved_numbers(before, numbers(200, answer(0.01, 9.0))) == []
    assert moved_numbers(before, numbers(200, answer(0.9, 1.0))) == ["result.rank_ic"]
    assert numbers(409, answer(0.01, 1.0)) == {}
    assert moved_numbers(None, before) is None


def test_the_environment_is_read_by_one_helper_and_the_named_keys() -> None:
    """The environment is read by one helper and the named keys."""

    assert _environment_readers(ROOT / "src" / "alphalattice") == set(_ENVIRONMENT_READERS)


def test_a_schema_hash_binds_the_structure_and_never_the_prose() -> None:
    """requirement (SH, LAWS.md SC3, V98, V247): a schema's words -- a model's docstring, a
    field's description, an enum's docstring, a title, examples -- move no schema hash; a type,
    a constraint, a default or a required field does. A property named like prose keeps its name."""

    from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

    from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure

    def contract(*, words: bool, bound: int = 3, default: str = "A", required: bool = True):
        class Colour(StrEnum):
            """Words about the colours."""

            RED = "RED"

        if not words:
            Colour.__doc__ = "Other words."

        class Part(BaseModel):
            model_config = ConfigDict(title="A part" if words else "Another")
            title: str = Field(description="a filing's own title", examples=["10-K"])

        class Whole(BaseModel):
            colour: Colour = Field(description="the colour" if words else "its hue")
            parts: list[Part] = Field(max_length=bound)
            mode: str = default
            note: str | None = None if not required else Field(...)

        Whole.__doc__ = "One." if words else "Two, and longer."
        Whole.model_rebuild(force=True)
        return Whole

    base = schema_structure(contract(words=True))
    # The words differ in the schema a reader sees, and in no structure.
    assert contract(words=False).model_json_schema() != contract(words=True).model_json_schema()
    assert schema_structure(contract(words=False)) == base
    assert "title" in base["$defs"]["Part"]["properties"]  # a property, not a keyword
    assert "description" not in json.dumps(base) and '"examples"' not in json.dumps(base)
    for changed in (
        contract(words=True, bound=4),
        contract(words=True, default="B"),
        contract(words=True, required=False),
    ):
        assert canonical_hash(schema_structure(changed)) != canonical_hash(base)
    whole = contract(words=True)
    assert schema_structure(whole.model_json_schema()) == base
    assert schema_structure(TypeAdapter(whole)) == base


# Where a raw JSON schema is left: shown to a reader, never hashed (SH). A hash takes
# `schema_structure`; a schema both shown and hashed hashes its structure beside the display.
_SCHEMA_DISPLAYS = frozenset(
    {
        ("src/alphalattice/control/product_host/composition/goals.py", "model"),
        (
            "src/alphalattice/control/product_host/composition/portfolio_research_operations.py",
            "PortfolioResearchRequestDocument",
        ),
        (
            "src/alphalattice/control/product_host/data_preparation/research_strategy.py",
            "FrozenPortfolioPreparationRequest",
        ),
        (
            "src/alphalattice/evidence/alternative_evidence/runtime/task_adapter.py",
            "AlternativeEvidenceAnalystAnswer",
        ),
        ("src/alphalattice/foundation/feature_engine/catalog/research.py", "ResearchFeatureChange"),
        ("src/alphalattice/interface/local_application/answers.py", "model"),
        ("src/alphalattice/interface/local_application/cli.py", "AlphaDevelopmentSection"),
        ("src/alphalattice/interface/local_application/cli.py", "AlphaLifecycleSection"),
        ("src/alphalattice/interface/local_application/cli.py", "FactorSection"),
        ("src/alphalattice/interface/local_application/cli.py", "PortfolioExperimentSpec"),
        ("src/alphalattice/interface/local_application/cli.py", "PortfolioResearchRequestDocument"),
        ("src/alphalattice/interface/local_application/cli.py", "RiskSection"),
        ("src/alphalattice/interface/local_application/cli.py", "answer"),
        (
            "src/alphalattice/interface/local_application/operations.py",
            "PortfolioResearchRequestDocument",
        ),
        ("src/alphalattice/kernel/shared_kernel/identity.py", "cast"),
        ("src/alphalattice/kernel/shared_kernel/identity.py", "schema"),
        (
            "src/alphalattice/oversight/chief_risk_officer/decision/book_evidence.py",
            "PortfolioReviewAnswer",
        ),
    }
)


def test_every_hashed_schema_goes_through_the_kernel() -> None:
    """requirement (SH, LAWS.md SC3): a raw `model_json_schema()` or `json_schema()` call stands
    only where a schema is shown to a reader; every schema a hash binds is the kernel's
    `schema_structure`, so no schema's prose is identity. The set is pinned, not counted."""

    found: set[tuple[str, str]] = set()
    for base in ("src", "scripts"):
        for path in sorted((ROOT / base).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "json_schema(" not in text:
                continue
            for node in ast.walk(ast.parse(text)):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"model_json_schema", "json_schema"}
                ):
                    receiver = node.func.value
                    while isinstance(receiver, ast.Call):
                        receiver = receiver.func
                    name = (
                        receiver.attr
                        if isinstance(receiver, ast.Attribute)
                        else getattr(receiver, "id", type(receiver).__name__)
                    )
                    found.add((path.relative_to(ROOT).as_posix(), name))
    assert found == _SCHEMA_DISPLAYS
