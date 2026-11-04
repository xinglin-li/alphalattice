"""Import rules from the package registry: allowed and refused examples."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from devtools.architecture import import_contracts
from devtools.architecture.structural import PackageArchitectureRegistry


@pytest.mark.proves("architecture.import_law_refuses_higher_levels_and_retired_names")
@pytest.mark.parametrize(
    ("module", "body", "found"),
    [  # the import law's must-refuse and must-pass specimens, direct and relative
        ("low/layer.py", "from alphalattice.control.high import down\n", "lint-imports"),
        ("low/relative_layer.py", "from ...control.high import down\n", "lint-imports"),
        ("low/retired.py", "import alphalattice.kernel.low.gone.mod\n", "TID251"),
        ("low/relative_retired.py", "from .gone import thing\n", "TID251"),
        ("low/lookalike.py", "import alphalattice.kernel.low.gonester\n", None),
        # the old law read every tracked file and honored no suppression
        ("low/ignored.py", "import alphalattice.kernel.low.gone\n", "TID251"),
        ("low/noqa_line.py", "import alphalattice.kernel.low.gone  # noqa: TID251\n", "TID251"),
        ("low/noqa_file.py", "# ruff: noqa\nimport alphalattice.kernel.low.gone\n", "TID251"),
        # and every untracked file Git does not ignore, and every file under src on disk
        ("tests/untracked.py", "import alphalattice.kernel.low.gone\n", "TID251"),
        ("low/untracked_ignored.py", "import alphalattice.kernel.low.gone\n", "TID251"),
        ("high/down.py", "from alphalattice.kernel.low import keep\n", None),
    ],
)
def test_the_import_law_refuses_a_higher_level_and_a_retired_name(
    tmp_path, module, body, found
) -> None:
    """A level-0 package importing level 1, or any file importing a retired owner, is refused by the
    contracts built from the registry; an import downward, or of a name that only looks retired,
    passes."""
    tree = _tree(tmp_path)
    group = "kernel" if module.startswith("low/") else "control"
    target = (
        tree / module if module.startswith("tests/") else tree / "src/alphalattice" / group / module
    )
    if "untracked" not in module:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tree, check=True)
    if "untracked" in module:  # written after the index, so Git does not track it
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    if "ignored" in module:  # tracked or not, an ignored file is still the law's
        (tree / ".gitignore").write_text(target.name + "\n", encoding="utf-8")
    violations = import_contracts.check(tree, tmp_path / "evidence", _registry())
    if found is None:
        assert violations == ()
    else:
        assert violations and all(found in line for line in violations), violations
    assert (tmp_path / "evidence/importlinter.ini").is_file() and (
        tmp_path / "evidence/ruff-retired.toml"
    ).is_file()


@pytest.mark.proves("architecture.import_law_refuses_higher_levels_and_retired_names")
def test_the_import_law_names_a_partial_listing_and_launches_a_long_tree_in_bounded_batches(
    tmp_path, monkeypatch
) -> None:
    """A Git listing that fails is no population, even with output; a tree whose paths exceed one
    Windows command line is checked in several launches, the ban found in the last; a path too
    long to launch alone is named."""
    tree = _tree(tmp_path)
    names = [f"tests/{index:03d}_{'x' * 96}.py" for index in range(320)]  # about 33,000 characters
    (tree / "tests").mkdir()
    for name in names:
        (tree / name).write_text("", encoding="utf-8")
    (tree / names[-1]).write_text("import alphalattice.kernel.low.gone\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tree, check=True)
    violations = import_contracts.check(tree, tmp_path / "long", _registry())
    assert (
        len(violations) == 1 and "TID251" in violations[0] and Path(names[-1]).name in violations[0]
    )
    launches = import_contracts.launch_batches(["ruff"], names)[0]
    assert len(launches) > 1 and sum(map(len, launches)) == len(names)
    # Supplementary names exceed one Windows command line while each component fits POSIX NAME_MAX.
    wide = _tree(tmp_path / "wide")
    names = [f"tests/{index:03d}_{chr(0x1D4B3) * 60}.py" for index in range(270)]
    (wide / "tests").mkdir()
    for name in names:
        (wide / name).write_text("", encoding="utf-8")
    (wide / names[-1]).write_text("import alphalattice.kernel.low.gone\n", encoding="utf-8")
    subprocess.run(["git", "-c", "core.longpaths=true", "add", "-A"], cwd=wide, check=True)
    violations = import_contracts.check(wide, tmp_path / "wide-evidence", _registry())
    assert len(violations) == 1 and "TID251" in violations[0], violations
    wide_batches = import_contracts.launch_batches(["ruff"], names)[0]
    assert len(wide_batches) > 1
    for batch in wide_batches:
        assert len(subprocess.list2cmdline(["ruff", *batch]).encode("utf-16-le")) // 2 < 32_767
    monkeypatch.setattr(import_contracts, "COMMAND_LINE", 60)
    assert import_contracts.launch_batches(["ruff"], ["a.py", names[0]]) == ([["a.py"]], [names[0]])
    monkeypatch.setattr(import_contracts, "COMMAND_LINE", 32_000)
    run = import_contracts.run

    def partial(argv, root, **env):
        if "ls-files" in map(str, argv):  # a listing that fails part way, its output cut short
            return subprocess.CompletedProcess(argv, 1, names[0] + "\0", "fatal: index unreadable")
        return run(argv, root, **env)

    monkeypatch.setattr(import_contracts, "run", partial)
    violations = import_contracts.check(tree, tmp_path / "partial", _registry())
    assert any(
        "checked files are unknown" in line and "index unreadable" in line for line in violations
    )


def _registry() -> PackageArchitectureRegistry:
    """Two fixture packages, low at level 0 and high at level 1, and one retired owner."""
    return PackageArchitectureRegistry.model_validate(
        {
            "schema": "alphalattice.playpen.package-architecture",
            "packages": [
                dict(
                    package_id=name,
                    product_area="fixture",
                    capability_owner="Fixture",
                    role="DESK",
                    dependency_level=level,
                    authority_kind="FIXTURE",
                    allowed_peer_dependencies=[],
                    artifact_writer=False,
                    physical_namespace_target=f"alphalattice.{group}.{name}",
                )
                for name, level, group in (("low", 0, "kernel"), ("high", 1, "control"))
            ],
            "retired_imports": {"low.gone": "retired by a fixture cutover"},
        }
    )


def _tree(tmp_path):
    """A Git checkout holding the two fixture packages, nothing indexed yet."""
    tree = tmp_path / "repo"
    for package in (
        "alphalattice",
        "alphalattice/kernel",
        "alphalattice/control",
        "alphalattice/kernel/low",
        "alphalattice/control/high",
        "devtools",
    ):
        (tree / "src" / package).mkdir(parents=True, exist_ok=True)
        (tree / "src" / package / "__init__.py").write_text("", encoding="utf-8")
    (tree / "src/alphalattice/kernel/low/keep.py").write_text("", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tree, check=True)
    return tree
