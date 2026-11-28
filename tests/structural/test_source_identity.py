from __future__ import annotations

import ast
import importlib
import json
import sys
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

import pytest

from alphalattice.capabilities.domain_contracts.identity_successors import (
    IdentitySuccessorError,
    is_current,
    latest,
    predecessors,
    recorded_moves,
    recorded_origin,
)
from alphalattice.capabilities.domain_contracts.source_identity import (
    current_module_name,
    number_deciding_rule,
    source_component_id_from_tracked_path,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    VALUE_ROLE_PREFIXES,
    recipe_hashes_match,
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


_A, _B, _C, _D = ("blake3:" + digit * 64 for digit in "abcd")


def _successors(root: Path, moves: list[dict[str, str]], *, version: int = 1) -> Path:
    directory = root / "config" / "identity-successors"
    directory.mkdir(parents=True)
    header = {"schema": "identity-successors", "version": version}
    directory.with_suffix(".json").write_text(
        json.dumps(dict(header, moves=moves[:1])), encoding="utf-8"
    )
    for index, move in reversed(list(enumerate(moves[1:]))):
        (directory / f"{index:02d}.json").write_text(
            json.dumps(dict(header, moves=[move])), encoding="utf-8"
        )
    return root


def _move(role: str, predecessor: str, successor: str) -> dict[str, str]:
    return {
        "role": role,
        "predecessor": predecessor,
        "successor": successor,
        "change": "test",
        "reason": "test",
    }


@pytest.fixture
def successor_chain(tmp_path: Path) -> Path:
    return _successors(
        tmp_path,
        [
            _move("role", _A, _B),
            _move("role", _B, _C),
            _move("other", _C, _D),
            _move("loop", _A, _B),
            _move("loop", _B, _A),
        ],
    )


def test_a_recorded_move_names_the_installed_authority_for_its_role_only(
    successor_chain: Path,
) -> None:
    """Control: Recorded move names installed authority for its role only."""
    root = successor_chain

    assert is_current("role", _C, _C, root=root)
    assert is_current("role", _A, _C, root=root), "two recorded moves lead to the installed value"
    assert not is_current("role", _C, _A, root=root), "a move is one way"
    assert not is_current("role", _C, _D, root=root), "another role's move is not this one's"
    assert not is_current("role", _D, _C, root=root), "an unrecorded value is historical"
    assert not is_current("loop", _A, _C, root=root), "a recorded cycle ends, not current"


def test_a_bound_value_is_held_at_the_origin_its_recorded_moves_lead_from(
    successor_chain: Path,
) -> None:
    """Control: Bound value is held at origin its recorded moves lead from."""

    root = successor_chain

    assert recorded_origin("role", _C, root=root) == _A
    assert recorded_origin("role", _A, root=root) == _A
    assert recorded_origin("role", _D, root=root) == _D, "an unrecorded value is its own"
    assert recorded_origin("other", _D, root=root) == _C, "each role follows its own moves"
    assert recorded_origin("loop", _A, root=root) in {_A, _B}, "a recorded cycle ends"

    tied = _successors(root.parent / "tied", [_move("role", _A, _C), _move("role", _B, _C)])
    assert recorded_origin("role", _C, root=tied) == _A
    assert predecessors("role", _C, root=tied) == (_C, _A, _B)

    # A historical audit record cannot make a pre-BLAKE3 identity current.
    retired = "a" * 64
    root = _successors(root.parent / "retired", [_move("role", retired, _B), _move("role", _B, _C)])

    assert [move.predecessor for move in recorded_moves(root)] == [retired, _B]
    assert not is_current("role", retired, _C, root=root)
    assert predecessors("role", _C, root=root) == (_C, _B)
    assert recorded_origin("role", _C, root=root) == _B


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

    # A value moved back to one held before (A to B, B to C, C back to B) moves again (B to D): the
    # later move from the revisited B supersedes the earlier one, and A, B and C all lead to D.
    revisited = _successors(
        tmp_path / "revisited",
        [
            _move("role", _A, _B),
            _move("role", _B, _C),
            _move("role", _C, _B),
            _move("role", _B, _D),
        ],
    )
    for value in (_A, _B, _C):
        assert latest("role", value, root=revisited) == _D, value
        assert is_current("role", value, _D, root=revisited), value
        assert value in predecessors("role", _D, root=revisited), value
    for value in (_A, _B, _C, _D):  # backward and forward agree: the superseded B to C is not read
        assert all(
            is_current("role", earlier, value, root=revisited)
            for earlier in predecessors("role", value, root=revisited)
        ), value
    assert predecessors("role", _C, root=revisited) == (_C,)
    assert recorded_origin("role", _D, root=revisited) == _A
    returned = _successors(
        tmp_path / "returned",
        [_move("role", _A, _B), _move("role", _B, _A), _move("role", _A, _D)],
    )
    assert recorded_origin("role", _D, root=returned) == _A, "an origin bound before a revisit"
    recipe = f"{VALUE_ROLE_PREFIXES[-1]}book"  # a recipe hash moved back and on matches from each
    recipes = _successors(
        tmp_path / "recipes",
        [
            _move(recipe, _A, _B),
            _move(recipe, _B, _C),
            _move(recipe, _C, _B),
            _move(recipe, _B, _D),
        ],
    )
    assert all(recipe_hashes_match(value, _D, root=recipes) for value in (_A, _B, _C))
    assert not recipe_hashes_match(_A, _D, root=revisited), "only a value role's moves join"


@pytest.mark.parametrize(
    ("moves", "version", "code"),
    [
        pytest.param(
            [_move("role", _A, _B)],
            2,
            "shared_kernel.identity_successors_from_another_build",
            id="foreign-version",
        ),
        pytest.param(
            [_move("role", _A, _A)],
            1,
            "shared_kernel.identity_successor_invalid",
            id="self-successor",
        ),
        pytest.param(
            [_move("role", _A, "not-a-hash")],
            1,
            "shared_kernel.identity_successor_invalid",
            id="invalid-hash",
        ),
        pytest.param(
            [{"role": "role", "predecessor": _A}],
            1,
            "shared_kernel.identity_successor_invalid",
            id="incomplete-move",
        ),
        pytest.param(
            [_move("role", _A, _B), _move("role", _A, _B)],
            1,
            "shared_kernel.identity_successor_invalid: duplicate role " + _A,
            id="duplicate-predecessor",
        ),
        pytest.param(
            [_move("role", _A, _B), _move("role", _A, _C)],
            1,
            "shared_kernel.identity_successor_invalid: duplicate role " + _A,
            id="conflicting-successor",
        ),
        pytest.param(
            [_move("role", _A, _B), _move("role", _B, _C), _move("role", _B, _D)],
            1,
            "shared_kernel.identity_successor_invalid: duplicate role " + _B,
            id="repeated-without-a-cycle",
        ),
    ],
)
def test_moves_this_build_cannot_read_are_refused_not_ignored(
    tmp_path: Path, moves: list[dict[str, str]], version: int, code: str
) -> None:
    root = _successors(tmp_path, moves, version=version)

    with pytest.raises(IdentitySuccessorError, match=code):
        is_current("role", _A, _B, root=root)


def test_the_committed_moves_read_and_each_predecessor_moves_once_per_role(
    tmp_path: Path,
) -> None:
    # Once per link: a landing's next move from the value a role's last move reached, a value
    # it moved back to included, reads and leads that chain to the new value.
    moves = recorded_moves(ROOT)
    reached = {move.role: move.successor for move in moves}
    composed = tmp_path / "composed" / "config" / "identity-successors"
    composed.mkdir(parents=True)
    ledger = ROOT / "config" / "identity-successors"
    composed.with_suffix(".json").write_bytes(ledger.with_suffix(".json").read_bytes())
    for path in ledger.glob("*.json"):
        (composed / path.name).write_bytes(path.read_bytes())
    (composed / "9-landing.json").write_text(
        json.dumps(
            {
                "schema": "identity-successors",
                "version": 1,
                "moves": [_move(role, value, _D) for role, value in reached.items()],
            }
        ),
        encoding="utf-8",
    )

    for move in moves:
        if latest(move.role, move.predecessor, root=ROOT) == reached[move.role]:
            assert latest(move.role, move.predecessor, root=composed.parents[1]) == _D, move.role


def test_study_identities_track_no_platform_or_build_file() -> None:
    """A study's replay compares its kind's implementation closure; a
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


MODEL = (
    '"""Fit."""\n'
    "from typing import TYPE_CHECKING\n\n"
    "from . import helper\n"
    "from .storage import reader\n"
    "from .storage import writer\n"
    "from .helper import offset\n"
    "from .helper import offset as step\n"
    "from .helper import scale as rate\n\n"
    "if TYPE_CHECKING:\n    from .helper import Y\n\n"
    "SHIFT = 1\n\n\n"
    "def fit():\n"
    "    total: int = SHIFT\n"
    "    return rate() + total + helper.Y + offset()\n\n\n"
    "def twice():\n"
    "    helper.scale()\n"
    "    return offset() + step()\n\n\n"
    "READY = twice()\n"
)


class Seen(StrEnum):
    SAME = "same"
    ROTATES = "rotates"


def _replaced(relative: str, *pairs: tuple[str, str]) -> Callable[[Path], None]:
    def edit(source: Path) -> None:
        path = source / relative
        text = path.read_text(encoding="utf-8")
        for old, new in pairs:
            assert old in text
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")

    return edit


def _moved(source: Path) -> None:
    desk = source / "investment" / "desk"
    (desk / "model.py").rename(desk / "fitting.py")
    for relative in ("control/host/entry.py", "investment/desk/__init__.py"):
        _replaced(relative, (".model import", ".fitting import"))(source)


MODEL_PATH = "investment/desk/model.py"
EDITS: dict[str, tuple[Callable[[Path], None], Seen]] = {
    "unchanged": (lambda _source: None, Seen.SAME),
    "module shared by two roles": (lambda _source: None, Seen.ROTATES),
    "same-tick rewrite": (_replaced(MODEL_PATH, ("SHIFT = 1", "SHIFT = 2")), Seen.ROTATES),
    "rewrite with restored mtime": (
        _replaced(MODEL_PATH, ("SHIFT = 1", "SHIFT = 2")),
        Seen.ROTATES,
    ),
    "comment": (_replaced(MODEL_PATH, ("SHIFT = 1\n", "SHIFT = 1  # a shift\n")), Seen.SAME),
    "docstring": (_replaced(MODEL_PATH, ('"""Fit."""', '"""Fit, in other words."""')), Seen.SAME),
    "import moved into a function": (
        _replaced(
            MODEL_PATH,
            ("from .helper import scale as rate\n", ""),
            ("def fit():\n", "def fit():\n    from .helper import scale as rate\n\n"),
        ),
        Seen.SAME,
    ),
    "import alias renamed": (
        _replaced(MODEL_PATH, ("scale as rate", "scale as speed"), ("rate()", "speed()")),
        Seen.SAME,
    ),
    "import through a re-export": (
        _replaced("control/host/entry.py", ("desk.model import fit", "desk import fit")),
        Seen.ROTATES,
    ),
    "module moved, same body": (_moved, Seen.SAME),
    "constant changed": (_replaced(MODEL_PATH, ("SHIFT = 1", "SHIFT = 2")), Seen.ROTATES),
    "called body changed": (
        _replaced("investment/desk/helper.py", ("return 2", "return 3")),
        Seen.ROTATES,
    ),
    "operator changed": (_replaced(MODEL_PATH, ("rate() + total", "rate() - total")), Seen.ROTATES),
    "import moved into one of its two readers": (
        _replaced(
            MODEL_PATH,
            ("from .helper import offset\n", ""),
            ("def fit():\n", "def fit():\n    from .helper import offset\n\n"),
        ),
        Seen.ROTATES,
    ),
    "imports kept for their effect reordered": (
        _replaced(
            MODEL_PATH,
            (
                "from .storage import reader\nfrom .storage import writer\n",
                "from .storage import writer\nfrom .storage import reader\n",
            ),
        ),
        Seen.ROTATES,
    ),
    "import moved below a guard": (
        _replaced(
            MODEL_PATH,
            ("from .helper import offset\n", ""),
            ("import Y\n\n", "import Y\nfrom .helper import offset\n\n"),
        ),
        Seen.ROTATES,
    ),
    "import moved below an annotation": (
        _replaced(
            MODEL_PATH,
            ("from .helper import scale as rate\n", ""),
            ("total: int = SHIFT\n", "total: int = SHIFT\n    from .helper import scale as rate\n"),
        ),
        Seen.ROTATES,
    ),
    "import moved below a call": (
        _replaced(
            MODEL_PATH,
            ("from .helper import offset as step\n", ""),
            ("helper.scale()\n", "helper.scale()\n    from .helper import offset as step\n"),
        ),
        Seen.ROTATES,
    ),
}


@pytest.mark.parametrize("edit", EDITS, ids=list(EDITS))
def test_a_rule_closure_follows_imports_inside_the_deciding_packages_and_ignores_prose(
    tmp_path: Path, edit: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rule closure follows the deciding imports and moves only with the code that runs."""

    from alphalattice.capabilities.domain_contracts.source_identity import (
        NumberDecidingRule,
        number_deciding_closure,
        source_rule_closure_hash,
    )

    source = tmp_path / "src" / "alphalattice"

    def write(relative: str, text: str) -> None:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    for package in ("investment", "investment/desk/storage", "control"):
        write(f"{package}/__init__.py", "")
    write("investment/desk/__init__.py", "from .model import fit\n")
    write(
        "control/host/entry.py",
        "from typing import TYPE_CHECKING\n"
        "from alphalattice.investment.desk.model import fit\n"
        "from alphalattice.control.host import other\n"
        "if TYPE_CHECKING:\n    from alphalattice.investment.desk import typing_only\n\n\n"
        "def run():\n    return fit()\n",
    )
    write("control/host/other.py", "X = 1\n")
    write(MODEL_PATH, MODEL)
    write(
        "investment/desk/helper.py",
        "Y = 2\n\n\ndef scale():\n    return 2\n\n\ndef offset():\n    return 1\n",
    )
    write("investment/desk/storage/reader.py", "W = 4\n")
    write("investment/desk/storage/writer.py", "V = 5\n")
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
    modules = [
        "alphalattice.control.host.entry",
        "alphalattice.investment.desk",
        "alphalattice.investment.desk.helper",
        "alphalattice.investment.desk.model",
    ]
    assert sorted(number_deciding_closure(entry, root=tmp_path, rule=rule)) == modules

    from alphalattice.capabilities.domain_contracts import source_identity
    from alphalattice.kernel.clock import wall_time_ns

    clock = wall_time_ns() + 10_000_000_000  # a settled scratch tree without waiting on its clock
    monkeypatch.setattr(source_identity, "wall_time_ns", lambda: clock, raising=False)
    if edit == "same-tick rewrite":
        model = source / MODEL_PATH
        # A same-size rewrite sees the exact same stamp on a coarse filesystem within its tick.
        trust = source_identity.FileDigestTrust
        version = trust.version
        racy = (*trust.stamp(model)[:3], clock, clock)
        monkeypatch.setattr(
            trust,
            "version",
            staticmethod(lambda path: (True, racy) if path == model else version(path)),
        )

    closures: list[object] = []
    source_reads: list[Path] = []
    hash_closure = source_identity.canonical_hash
    read_file = Path.read_bytes

    def hashed(payload):
        closures.append(payload)
        return hash_closure(payload)

    def read(path):
        if path.is_relative_to(source):
            source_reads.append(path)
        return read_file(path)

    monkeypatch.setattr(source_identity, "canonical_hash", hashed)
    monkeypatch.setattr(Path, "read_bytes", read)

    def identity(role="EXAMPLE") -> str:
        return source_rule_closure_hash(
            root=tmp_path,
            tracked_paths=("src/alphalattice/control/host/entry.py",),
            semantic_owner="host",
            numerical_role=role,
        )

    if edit == "unchanged":
        model = source / MODEL_PATH
        observations = []
        read_version = source_identity.FileDigestTrust.version

        def observe_version(path):
            if path == model:
                observations.append(path)
            return read_version(path)

        with monkeypatch.context() as probe:
            probe.setattr(source_identity.FileDigestTrust, "version", staticmethod(observe_version))
            before = identity()
        assert len(observations) == 2  # opening and final confirmation for one acquired source
    else:
        before = identity()
    if edit != "same-tick rewrite":
        assert identity() == before
        assert len(closures) == 1
    initial_reads = source_reads.copy()
    change, expected = EDITS[edit]
    if edit == "rewrite with restored mtime":
        from os import utime

        model = source / MODEL_PATH
        before_edit = model.stat()
        change(source)
        utime(model, ns=(before_edit.st_atime_ns, before_edit.st_mtime_ns))
    else:
        change(source)
    role = "ALSO_EXAMPLE" if edit == "module shared by two roles" else "EXAMPLE"
    assert (Seen.SAME if identity(role) == before else Seen.ROTATES) is expected
    if edit == "unchanged":
        assert len(closures) == 1 and source_reads == initial_reads
        reached: list[dict[str, str]] = []
        with source_identity.observe_source_closures(reached.append):
            assert identity() == before
        assert [sorted(closure) for closure in reached] == [modules]
    elif edit == "module shared by two roles":
        assert len(closures) == 2 and source_reads == initial_reads
        _assert_shared_advancement_source_stamp(source / MODEL_PATH, identity, monkeypatch)
    elif edit in ("constant changed", "same-tick rewrite", "rewrite with restored mtime"):
        assert len(closures) == 2 and source_reads[len(initial_reads) :] == [source / MODEL_PATH]


def _assert_shared_advancement_source_stamp(model, identity, monkeypatch):
    from os import utime

    from alphalattice.capabilities.domain_contracts import source_identity
    from alphalattice.control.product_host.composition import decision_advancement

    monkeypatch.setattr(
        decision_advancement, "source_rule_closure_hash", lambda **_: identity("ADVANCEMENT")
    )
    for getter in ("scoring_identity", "calibration_identity", "decision_identity"):
        monkeypatch.setattr(decision_advancement, getter, lambda role=getter: identity(role))
    implementation = decision_advancement.DecisionAdvancementApplication.implementation
    before = implementation(None)
    versions = []
    trust = source_identity.FileDigestTrust
    read_version = trust.version

    def counted(path):
        versions.append(path)
        return read_version(path)

    monkeypatch.setattr(trust, "version", staticmethod(counted))
    stamp = model.stat()
    original = model.read_text(encoding="utf-8")

    def replace_source(contents):
        staged = model.with_suffix(".next")
        staged.write_text(contents, encoding="utf-8", newline="\n")
        utime(staged, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        staged.replace(model)

    with source_identity.source_read_admission(lambda: None):
        assert implementation(None) == before
        assert versions.count(model) == 1
        versions.clear()
        with source_identity.source_role_acquisition():
            assert implementation(None) == before
            assert implementation(None) == before
        assert versions.count(model) == 1
        versions.clear()
        with source_identity.source_role_acquisition():
            assert identity("COLD_FIRST") != identity("COLD_SECOND")
        assert versions.count(model) == 2  # one opening and one stability check
        with source_identity.source_role_acquisition():
            assert implementation(None) == before
            replace_source(original.replace("SHIFT = 1", "SHIFT = 2"))
            with source_identity.source_role_acquisition(fresh=True):
                assert implementation(None) != before
            replace_source(original)
            with source_identity.source_role_acquisition(fresh=True):
                assert implementation(None) == before


@pytest.mark.proves("domain_contracts.source_identity_survives_relocation")
@pytest.mark.parametrize("numerical", [False, True], ids=["external", "numerical"])
def test_a_rule_closure_carries_the_semantic_binding_of_an_import(
    tmp_path: Path, numerical: bool
) -> None:
    """Requirement (LAWS.md ID3): both closure forms preserve resolved source identity on a move."""
    from alphalattice.capabilities.domain_contracts.source_identity import (
        number_deciding_closure,
        number_deciding_rule,
        source_rule_closure_hash,
    )

    entry = "src/alphalattice/control/host/entry.py"
    entry_module = "alphalattice.control.host.entry"
    binding = "alphalattice.kernel.storage"
    layouts = (
        (binding, binding),
        ("alphalattice.foundation.moved_storage", binding),
        ("alphalattice.foundation.other_storage", "alphalattice.foundation.other_storage"),
    )
    values = []
    for index, (package, semantic_name) in enumerate(layouts):
        root = tmp_path / ("numerical" if numerical else "external") / str(index)
        paths = {
            entry: "from .bridge import value\n",
            "src/alphalattice/control/host/bridge.py": (
                f"from {package} import reader\nvalue = reader.load()\n"
            ),
            "src/" + package.replace(".", "/") + "/reader.py": "def load():\n    return 7\n",
            "config/identity-roles.json": json.dumps(
                {
                    "number_deciding_packages": ["host", "storage"] if numerical else ["host"],
                    "platform_segments": [],
                    "module_bindings": {semantic_name: package},
                }
            ),
        }
        for name, text in paths.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        bound_entry = "src/" + semantic_name.replace(".", "/") + "/reader.py"
        form_one = tuple(
            number_deciding_closure(entries, root=root, rule=number_deciding_rule(root))
            for entries in ((entry_module,), (entry_module, semantic_name + ".reader"))
        )
        form_two = tuple(
            source_rule_closure_hash(
                root=root, tracked_paths=entries, semantic_owner="host", numerical_role="EXAMPLE"
            )
            for entries in ((entry,), (entry, bound_entry))
        )
        values.append((*form_one, *form_two))
    before, moved, different = values
    assert (binding + ".reader" in before[0]) is numerical
    assert before == moved
    assert before[0] != different[0]
    assert before[1] != different[1]
    assert before[2] != different[2]


_CACHED_TREE = {
    "investment/__init__.py": "",
    "investment/desk/__init__.py": "",
    "investment/desk/model.py": (
        "from alphalattice.investment.desk.helper import scale\n\n\ndef fit():\n"
        "    return scale() + 1\n"
    ),
    "investment/desk/helper.py": "def scale():\n    return 2\n",
}


def _resealed(entry: Path, old: str, new: str) -> None:
    """The entry rewritten and named after its new bytes, as a writer that sealed it would."""
    from alphalattice.kernel.identity import FileDigestTrust, blake3_hex

    text = entry.read_text(encoding="utf-8").replace(old, new, 1)
    entry.unlink()
    partial = entry.with_name("rewritten")
    partial.write_text(text, encoding="utf-8")
    own = blake3_hex(FileDigestTrust.fresh_digest(partial))
    partial.replace(entry.with_name(f"{entry.name.rsplit('.', 2)[0]}.{own}.json"))


def _altered(entry: Path) -> None:
    """The entry's template changed in place, its name kept: still a form the walk reads."""
    text = entry.read_text(encoding="utf-8")
    entry.write_text(text.replace('"template": ["', '"template": ["x', 1), encoding="utf-8")


_DAMAGED = {  # the file's own digest no longer matches its name
    "corrupt entry": lambda entry: entry.write_bytes(b"not json"),
    "partial entry": lambda entry: entry.write_bytes(
        entry.read_bytes()[: entry.stat().st_size // 2]
    ),
    "altered form": _altered,
    # resealed: its digest matches, its key or its form does not
    "another key": lambda entry: _resealed(entry, '"package": "', '"package": "other.'),
    "damaged form, key kept": lambda entry: _resealed(
        entry, '"imports": [', '"imports": [["bogus"], '
    ),
}
_RESEALED = ("another key", "damaged form, key kept")
CACHE_CASES = (
    "miss",
    "hit",
    *_DAMAGED,
    "changed between reads",
    "changed canonicaliser",
    "changed bytes",
)


@pytest.mark.proves("domain_contracts.source_identity_cache_keyed_by_what_decides")
@pytest.mark.parametrize(
    "case", CACHE_CASES, ids=[case.replace(",", "").replace(" ", "-") for case in CACHE_CASES]
)
def test_a_cached_canonical_form_hashes_as_the_parsed_one_and_a_bad_entry_is_parsed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """requirement (LAWS.md ID1): a cached canonical form gives a cold parse's digests, any entry
    it cannot prove from the one read it decodes is parsed again, and no older file is removed."""
    from itertools import count

    from alphalattice.capabilities.domain_contracts import source_identity
    from alphalattice.capabilities.domain_contracts.source_identity import (
        NumberDecidingRule,
        canonical_closure,
    )
    from alphalattice.kernel.config import CACHE_ROOT_ENV
    from alphalattice.kernel.identity import file_facts

    tree = tmp_path / "tree"
    for relative, text in _CACHED_TREE.items():
        path = tree / "src" / "alphalattice" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    entry, rule = (
        ("alphalattice.investment.desk.model",),
        NumberDecidingRule(frozenset({"desk"}), frozenset()),
    )
    fresh = count()  # each call keys its process memo anew, as a new process would
    monkeypatch.setattr(
        source_identity, "change_xxh3_128_hex", lambda _: f"{tmp_path.name}-{next(fresh)}"
    )
    parses: list[str] = []
    parse = ast.parse
    monkeypatch.setattr(
        ast,
        "parse",
        lambda source, *args, **kwargs: parses.append("") or parse(source, *args, **kwargs),
    )

    def digests(cache: Path) -> tuple[dict[str, str], int]:
        monkeypatch.setenv(CACHE_ROOT_ENV, str(cache))
        parses.clear()
        return canonical_closure(entry, root=tree, rule=rule), len(parses)

    cold, modules = digests(tmp_path / "cold")
    (folder,) = (tmp_path / "cold" / "source-identity").iterdir()  # <cache tag>-<canonicaliser>
    assert folder.name.startswith(f"{sys.implementation.cache_tag}-")
    cache = tmp_path / "cache"
    folder = cache / "source-identity" / folder.name
    older = folder.with_name(f"{sys.implementation.cache_tag}-{'0' * 64}")
    expected, parsed, left = cold, modules, set()
    if case == "changed canonicaliser":
        older.mkdir(parents=True)
        (older / f"{'1' * 64}-alphalattice.{'2' * 64}.json").write_text("{}", encoding="utf-8")
        (older / "notes.txt").write_text("not the cache's", encoding="utf-8")
        left = {kept.name for kept in older.iterdir()}
    elif case != "miss":
        digests(cache)
    for kept in folder.glob("*.json") if case in _DAMAGED else ():
        _DAMAGED[case](kept)
    hashed = file_facts.whole_digest
    if case == "changed between reads":

        def changing(handle: Any, kept: list[bytes] | None = None) -> str:
            digest = hashed(handle, kept)
            _altered(Path(handle.name))  # after its one read, before any decode
            return digest

        monkeypatch.setattr(file_facts, "whole_digest", changing)
    if case in ("hit", "changed between reads"):
        parsed = 0  # the bytes decoded are the bytes proved, so the form is the one written
    if case == "changed bytes":
        helper = tree / "src" / "alphalattice" / "investment" / "desk" / "helper.py"
        helper.write_text("def scale():\n    return 3\n", encoding="utf-8")
        expected, parsed = digests(tmp_path / "cold-changed")[0], 1
        assert expected != cold
    assert digests(cache) == (expected, parsed)
    if case == "changed between reads":  # each entry, changed in place, is now a miss
        monkeypatch.setattr(file_facts, "whole_digest", hashed)
        assert digests(cache) == (expected, modules)
    assert digests(cache) == (expected, 0)  # every entry now reads back
    assert {kept.name for kept in older.iterdir()} == left if older.exists() else not left
    stale = modules if case in _RESEALED else 0  # a resealed entry stays beside the new one
    assert len(list(folder.glob("*.json"))) == modules + stale + (case == "changed bytes")


def test_a_span_moves_no_identity_and_any_other_form_reads_as_written() -> None:
    """A span moves no identity and any other form reads as written."""

    from alphalattice.capabilities.domain_contracts.source_identity import (
        source_bytes_syntax_digest as digest,
    )

    plain = (
        "import numpy as np\n\n\n"
        "def fit(x):\n    total = np.sum(x)\n    return total * 2\n\n\n"
        "class Model:\n    def score(self):\n        return 1\n"
    )
    measured = (
        "import numpy as np\n"
        "from alphalattice.capabilities.domain_contracts.spans import span, spanned\n\n\n"
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

    from alphalattice.capabilities.domain_contracts.source_identity import SPAN_MODULE
    from alphalattice.capabilities.domain_contracts.spans import SPAN_CLASSES

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
    "src/alphalattice/capabilities/domain_contracts/environment.py": "the helper",
    # Rebuild keys of caches whose content the environment decides, and packs it gates.
    "src/alphalattice/capabilities/knowledge/model_store.py": "a stored model's rebuild key",
    "src/alphalattice/capabilities/knowledge/catalog.py": (
        "a knowledge pack's installed distributions"
    ),
    "src/alphalattice/capabilities/alpha_modeling/runtime/lightgbm_threads.py": (
        "the LightGBM thread canary is sealed per installed version (PA3)"
    ),
    "src/alphalattice/foundation/market_data_ops/sources/providers.py": (
        "the library that fetched the observations, recorded with them"
    ),
    "src/alphalattice/evidence/alternative_evidence/runtime/execution.py": (
        "the machine a Task ran on, recorded beside its reuse"
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
    """regression: U0 compared each read's verdict, so a result whose `rank_ic` moved from
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


@pytest.mark.proves("domain_contracts.schema_hash_ignores_prose_and_binds_structure")
def test_a_schema_hash_binds_the_structure_and_never_the_prose() -> None:
    """requirement (SH, LAWS.md SC3, V98): a schema's words -- a model's docstring, a
    field's description, an enum's docstring, a title, examples -- move no schema hash; a type,
    a constraint, a default or a required field does. A property named like prose keeps its name."""

    from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

    from alphalattice.capabilities.domain_contracts.identity import canonical_hash, schema_structure

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

    schema = {
        "title": "Original words",
        "type": "object",
        "properties": {"exposure": {"type": "number", "default": 1.0}},
    }
    same = "blake3:0ec3e68bc13c78ea0f15224c610d0dca2f85ecc17994aa6fac186cf9dabd8c64"
    golden = [
        (schema_structure(schema), same),
        (schema_structure(schema), same),
        (schema_structure({**schema, "title": "Other words"}), same),
        (
            schema_structure(
                {
                    **schema,
                    "properties": {"exposure": {"type": "number", "default": 2.0}},
                }
            ),
            "blake3:8a653ee96783d88ca5b143112b30a5c15cdb194562a8ad1cbf32f211b1f0fbad",
        ),
        (
            {"fees": [float("nan"), -0.0]},
            "blake3:93c1bea9c9fef87315b8341c258a5f9a092b03485bd572613142f265c107bfcc",
        ),
    ]
    for value, expected in golden:
        assert canonical_hash(value) == expected


# Raw schemas below are display-only; hashes use the kernel schema_structure (SH).
_SCHEMA_DISPLAYS = frozenset(
    {
        *(
            ("src/alphalattice/control/product_host/composition/goal_check.py", m)
            for m in ("GoalDeclaration", "GoalReferenceRequest", "GoalStatement", "GoalSubmission")
        ),
        (
            "src/alphalattice/control/product_host/composition/portfolio_research_operations.py",
            "full_request_document",
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
        ("src/alphalattice/interface/local_application/cli.py", "full_request_document"),
        ("src/alphalattice/interface/local_application/cli.py", "RiskSection"),
        ("src/alphalattice/interface/local_application/cli.py", "answer"),
        (
            "src/alphalattice/interface/local_application/operations.py",
            "full_request_document",
        ),
        ("src/alphalattice/interface/local_application/portfolio_research.py", "_request_document"),
        ("src/alphalattice/capabilities/domain_contracts/identity.py", "cast"),
        ("src/alphalattice/capabilities/domain_contracts/identity.py", "schema"),
        (
            "src/alphalattice/oversight/chief_risk_officer/decision/book_evidence.py",
            "PortfolioReviewAnswer",
        ),
        # The interface inventory displays the plugin contract schemas and has no identity readers.
        ("src/devtools/architecture/interface_registry.py", "TypeAdapter"),
        ("src/devtools/architecture/interface_registry.py", "contract"),
        # The compatibility manifest displays typed contracts; neither schema feeds a hash.
        ("src/devtools/architecture/operation_registry.py", "PortfolioResearchRequestDocument"),
        ("src/devtools/architecture/operation_registry.py", "model"),
    }
)


def test_every_hashed_schema_goes_through_the_kernel() -> None:
    """Pin raw schemas to displays; hashed schemas use kernel schema_structure (SH/SC3)."""

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


def test_preparation_identity_tracks_the_data_rule_and_not_grant_mechanics(tmp_path, monkeypatch):
    """Grant mechanics leave preparation identity unchanged; data decision logic rotates it."""
    from alphalattice.control.product_host.data_preparation import application

    tree = ast.parse(Path(application.__file__).read_text(encoding="utf-8"))
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_implementation"
    )
    paths = next(
        ast.literal_eval(k.value)
        for n in ast.walk(function)
        if isinstance(n, ast.Call)
        for k in n.keywords
        if k.arg == "tracked_paths"
    )
    (tmp_path / "config").mkdir()
    (tmp_path / "config/identity-roles.json").write_bytes(
        (ROOT / "config/identity-roles.json").read_bytes()
    )
    bindings = number_deciding_rule(ROOT)  # a stable path's file sits where its binding puts it
    for relative in paths:
        module = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
        file = tmp_path / "src" / (current_module_name(module, bindings).replace(".", "/") + ".py")
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("VALUE = 1\n", encoding="utf-8")
    grant = tmp_path / "src/alphalattice/control/data_platform/delegation.py"
    grant.write_text("GRANT = 1\n", encoding="utf-8")
    monkeypatch.setattr(application, "resolve_playpen_root", lambda _: tmp_path)
    before = application._implementation()
    grant.write_text("GRANT = 2\n", encoding="utf-8")
    assert application._implementation() == before
    rule = tmp_path / "src/alphalattice/control/data_platform/data_decision_rule.py"
    rule.write_text("VALUE = 2\n", encoding="utf-8")
    assert application._implementation() != before


@pytest.mark.proves("domain_contracts.source_identity_reads_alike_on_every_python")
def test_a_module_reads_the_same_syntax_on_every_supported_python() -> None:
    """A module's identity text is the one Python 3.12 wrote, so moving the interpreter moves no
    identity: newer interpreters' empty fields stay written, and a type parameter's absent default
    (PEP 696) writes nothing."""
    from alphalattice.capabilities.domain_contracts import source_identity

    text = "def f[T](x: T) -> T:\n    return x\n"
    assert source_identity._dumped(ast.parse(text)) == (
        "Module(body=[FunctionDef(name='f', args=arguments(posonlyargs=[], args=[arg(arg='x', "
        "annotation=Name(id='T', ctx=Load()))], kwonlyargs=[], kw_defaults=[], defaults=[]), "
        "body=[Return(value=Name(id='x', ctx=Load()))], decorator_list=[], "
        "returns=Name(id='T', ctx=Load()), type_params=[TypeVar(name='T')])], type_ignores=[])"
    )
    template = source_identity._canonical_tree(ast.parse(text), ("module", "alphalattice")).template
    assert template == (
        "Module([FunctionDef('f',arguments([],[arg('x',Name('T',Load),None,),],None,[],[],None,[],),"
        "[Return(Name('x',Load),),],[],Name('T',Load),None,[TypeVar('T',None,),],),],[],)",
    )
