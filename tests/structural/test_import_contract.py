"""Package deferrals and private fences are explicit owner contracts."""

import json
from pathlib import Path

import pytest

from devtools.architecture.structural import (
    PackageArchitectureRegistry,
    physical_namespace_violations,
)

SOURCE = "alphalattice.capabilities.alpha_modeling"
TARGET = "alphalattice.kernel.shared_kernel"

MODULE = TARGET + ".sample"
READ = "def read():\n    "
IMPORTS = {
    "statement": f"{READ}from {MODULE} import Marker",
    "importlib": f"import importlib as loader\n{READ}return loader.import_module('{MODULE}')",
    "alias": f"from importlib import import_module as load\n{READ}return load('{MODULE}')",
    "package": "",
    "keyword": f"import importlib\n{READ}return importlib.import_module(name='{MODULE}')",
    "annotated": "from typing import Final\nimport importlib\n"
    f"MODULE: Final = '{MODULE}'\n{READ}return importlib.import_module(MODULE)",
    "relative-positional": "import importlib\n"
    f"{READ}return importlib.import_module('.sample', '{TARGET}')",
    "relative-keyword": f"import importlib\nPACKAGE: str = '{TARGET}'\n{READ}"
    "return importlib.import_module(name='.sample', package=PACKAGE)",
    "builtin": f"{READ}return __import__(name='{MODULE}')",
    "repeated": f"from {MODULE} import Marker\n{READ}from {MODULE} import Marker",
    "external": f"{READ}import yaml",
}


def import_fixture(tmp_path, statement):
    raw = json.loads((Path(__file__).parents[2] / "config/package-architecture.json").read_text())
    packages = [
        p for p in raw["packages"] if p["package_id"] in {"alpha_modeling", "shared_kernel"}
    ]
    registry = PackageArchitectureRegistry.model_validate(
        {"schema": raw["schema"], "packages": packages}
    )
    src = tmp_path / "src"
    for package in packages:
        directory = src.joinpath(*package["physical_namespace_target"].split("."))
        directory.mkdir(parents=True)
        for name in ("__init__", "sample", "_private"):
            (directory / f"{name}.py").write_text("class Marker: pass\n_Marker = Marker\n")
    reader = src.joinpath(*SOURCE.split(".")) / "reader.py"
    reader.write_text(statement + "\n")
    return src, registry, reader


def fenced(registry, scope, target):
    key = SOURCE + ".reader::" + scope
    return registry.model_copy(update={"private_import_fences": {key: {target: "Fixture fence"}}})


@pytest.mark.parametrize("allowed", [False, True])
@pytest.mark.parametrize("load", IMPORTS)
def test_a_function_import_requires_its_module_declaration(tmp_path, allowed, load):
    """Deferrals need a declaration and reason; repeating an eager import still refuses."""
    module = SOURCE + ".nested.sample" if load == "package" else MODULE
    reason = "  # lazy: optional fixture" if allowed else ""
    src, registry, _ = import_fixture(tmp_path, IMPORTS[load] + reason)
    if load == "package":
        directory = src.joinpath(*SOURCE.split(".")) / "nested"
        directory.mkdir()
        (directory / "sample.py").write_text("class Marker: pass\n")
        (directory / "__init__.py").write_text(
            "def read():\n    from .sample import Marker" + reason + "\n"
        )
    if allowed:
        registry = registry.model_copy(
            update={"deferred_imports": {module: "Declared optional fixture"}}
        )
    else:
        registry = fenced(registry, "read", module + ".Marker")
    found = physical_namespace_violations(source_root=src, registry=registry)
    assert any("unlisted-function-import" in issue for issue in found) is (
        not allowed and load != "external"
    )
    assert any("unexplained-function-import" in issue for issue in found) is (not allowed)
    assert any("repeated-function-import" in issue for issue in found) is (load == "repeated")


@pytest.mark.parametrize("allowed", [False, True])
@pytest.mark.parametrize(
    "statement,target,cross_owner",
    [
        (f"from {TARGET}._private import Marker", TARGET + "._private.Marker", True),
        (f"from {TARGET}.sample import _Marker as Marker", TARGET + ".sample._Marker", True),
        (f"import {TARGET}._private as public", TARGET + "._private", True),
        (f"from {SOURCE}._private import Marker", SOURCE + "._private.Marker", False),
        (f"from {TARGET}.sample import Marker as _local", TARGET + ".sample.Marker", False),
    ],
)
def test_a_private_import_keeps_its_exact_owner_fence(
    tmp_path, allowed, statement, target, cross_owner
):
    """Private fences permit one existing scope; internal and public aliases stay legal."""
    src, registry, reader = import_fixture(tmp_path, statement)
    if allowed:
        registry = fenced(registry, "<module>", target)
    found = physical_namespace_violations(source_root=src, registry=registry)
    assert any("private-import" in issue for issue in found) is (cross_owner and not allowed)
    if cross_owner and allowed:
        existing = fenced(registry, "existing", target)
        reader.write_text("def existing():\n    " + statement + "  # lazy: fenced fixture\n")
        assert physical_namespace_violations(source_root=src, registry=existing) == ()
        for changed in (statement + "\n" + statement, "def other():\n    " + statement):
            reader.write_text(changed + "\n")
            assert any(
                "private-import" in issue
                for issue in physical_namespace_violations(source_root=src, registry=registry)
            )
