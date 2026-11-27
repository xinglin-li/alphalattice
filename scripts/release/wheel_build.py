"""Setuptools build command carrying explicit runtime data without private checkout files."""

from __future__ import annotations

import json
import shutil
import sys
import tomllib
from pathlib import Path

from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from alphalattice.capabilities.domain_contracts import spawn  # noqa: E402
from alphalattice.kernel.identity import integrity_blake3_path  # noqa: E402

MANIFEST = ROOT / "config/release/runtime-resources.json"


def runtime_requirements(root: Path = ROOT) -> str:
    """Export the accepted runtime lock for installing either command leg."""
    result = spawn.run(
        [
            "uv",
            "export",
            "--locked",
            "--offline",
            "--all-extras",
            "--no-dev",
            "--no-emit-project",
            "--no-hashes",
            "--no-annotate",
            "--no-header",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def runtime_overrides(root: Path = ROOT) -> str:
    """Carry the project's resolver overrides into the tool install."""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return "".join(override + "\n" for override in project["tool"]["uv"]["override-dependencies"])


def resource_files(root: Path = ROOT) -> tuple[str, ...]:
    """Enumerate the distribution's exact external resource inputs deterministically."""
    document = json.loads((root / MANIFEST.relative_to(ROOT)).read_text(encoding="utf-8"))
    paths = set(document["files"]) | set(document["host_declarations"])
    for directory in document["directories"]:
        paths.update(
            path.relative_to(root).as_posix()
            for path in (root / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    for path in sorted(paths):
        if not (root / path).is_file():
            raise FileNotFoundError(f"wheel.runtime_resource_missing:{path}")
    return tuple(sorted(paths))


class BuildPy(build_py):
    """Build product modules and their declared data for an installed wheel."""

    def run(self) -> None:
        if not self.editable_mode:
            staging = Path(self.build_lib).resolve() / "alphalattice"
            if not staging.is_relative_to((ROOT / "build").resolve()):
                raise ValueError("wheel.staging_path_outside_build")
            if staging.exists():
                shutil.rmtree(staging)
        super().run()
        if self.editable_mode:
            return
        target = Path(self.build_lib) / "alphalattice" / "_runtime"
        for relative in resource_files():
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        document = json.loads(MANIFEST.read_text(encoding="utf-8"))
        hashes = {
            path: integrity_blake3_path(ROOT / path)
            for path in sorted(document["development_source_hashes"])
        }
        (target / "config/release/source-hashes.json").write_text(
            json.dumps(hashes, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        (target / "config/release/runtime-requirements.txt").write_text(
            runtime_requirements(), encoding="utf-8", newline="\n"
        )
        (target / "config/release/runtime-overrides.txt").write_text(
            runtime_overrides(), encoding="utf-8", newline="\n"
        )
