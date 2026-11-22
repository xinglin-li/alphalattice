"""Provision a pinned development kit and verify the telemetry registry off-line."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from zipfile import ZipFile

from alphalattice.kernel.shared_kernel.domain.serialization import sha256_hex

ROOT = Path(__file__).resolve().parents[3]
REGISTRY_INPUTS = (
    "config/telemetry/manifest.yaml",
    "config/telemetry/signals.yaml",
)
TEMPLATE_INPUTS = (
    "config/telemetry-templates/reference/weaver.yaml",
    "config/telemetry-templates/reference/telemetry_names.py.j2",
    "config/telemetry-templates/reference/telemetry.md.j2",
)
GENERATED = {
    "telemetry_names.py": ROOT / "src/alphalattice/kernel/shared_kernel/telemetry_names.py",
    "telemetry.md": ROOT / "docs/public-source/telemetry.md",
}


def _publish(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.close()
            os.replace(temporary, path)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)


def _checked_archive(kit: Path, name: str, pin: dict[str, str]) -> Path:
    path = kit / name
    cached = path.is_file()
    if cached:
        payload = path.read_bytes()
    else:
        with urlopen(pin["url"], timeout=30) as response:
            payload = response.read()
    if sha256_hex(payload) != pin["sha256"]:
        raise ValueError(f"Weaver development input digest differs: {name}")
    if not cached:
        _publish(path, payload)
    return path


def _extract_program(archive: Path, binary: Path) -> None:
    if archive.suffix == ".zip":
        with ZipFile(archive) as compressed:
            (entry,) = (name for name in compressed.namelist() if Path(name).name == binary.name)
            binary.write_bytes(compressed.read(entry))
    else:
        with tarfile.open(archive) as compressed:
            (member,) = (
                item for item in compressed if item.isfile() and Path(item.name).name == binary.name
            )
            stream = compressed.extractfile(member)
            assert stream is not None
            with stream:
                binary.write_bytes(stream.read())
    binary.chmod(0o755)


def _extract_model(archive: Path, directory: Path) -> None:
    with tarfile.open(archive) as compressed:
        for item in compressed:
            relative = item.name.partition("/")[2]
            if not item.isfile() or not relative.startswith("model/"):
                continue
            path = directory / relative.removeprefix("model/")
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError(f"Weaver model path escapes its directory: {item.name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = compressed.extractfile(item)
            assert stream is not None
            with stream:
                path.write_bytes(stream.read())


def _local_dependencies(manifest: Path, directories: Sequence[Path]) -> None:
    contents = manifest.read_text(encoding="utf-8")
    paths = re.compile(r"^([ \t]*registry_path: )[^\n]+$", re.MULTILINE)
    if len(paths.findall(contents)) != len(directories):
        raise ValueError(f"Weaver dependency count differs: {manifest}")
    local = iter(directories)
    relocated = paths.sub(lambda match: match[1] + json.dumps(next(local).as_posix()), contents)
    manifest.write_text(relocated, encoding="utf-8")


def _binary(kit: Path) -> Path:
    return kit / ("weaver.exe" if platform.system() == "Windows" else "weaver")


def prepare(kit: Path) -> None:
    """Download verified development inputs before a test starts."""
    kit.mkdir(parents=True, exist_ok=True)
    lock = json.loads((ROOT / "config/weaver-lock.json").read_bytes())
    platform_name = f"{platform.system()}-{platform.machine()}"
    pin = lock["archives"][platform_name]
    archive = _checked_archive(kit, pin["url"].rsplit("/", 1)[1], pin)
    _extract_program(archive, _binary(kit))
    if (
        subprocess.check_output(
            [str(_binary(kit)), "--version"],
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).strip()
        != f"weaver {lock['version']}"
    ):
        raise ValueError("Weaver development tool version differs")
    directories = [kit / name for name in lock["registries"]]
    for directory in directories:
        pin = lock["registries"][directory.name]
        _extract_model(_checked_archive(kit, f"{directory.name}.tar.gz", pin), directory)
    _local_dependencies(kit / "genai/manifest.yaml", (kit / "opentelemetry",))
    registry = kit / "registry"
    registry.mkdir(exist_ok=True)
    for source in REGISTRY_INPUTS:
        shutil.copyfile(ROOT / source, registry / Path(source).name)
    _local_dependencies(registry / "manifest.yaml", directories)
    shutil.copyfile(ROOT / "config/telemetry/live-check.toml", kit / ".weaver.toml")


def _weaver(
    kit: Path, action: str, *args: str, payload: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(_binary(kit)),
            "registry",
            action,
            "--quiet",
            "--v2",
            "-r",
            str(kit / "registry"),
            *args,
        ],
        cwd=kit,
        input=payload,
        capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        encoding="utf-8",
        timeout=60,
        check=False,
    )


def live_check(kit: Path, samples: list[dict[str, Any]]) -> subprocess.CompletedProcess[str]:
    """Send samples to the preinstalled Weaver without fetching code or dependencies."""
    return _weaver(
        kit.resolve(),
        "live-check",
        "--input-source",
        "stdin",
        "--input-format",
        "json",
        "--format",
        "json",
        "--fail-on",
        "violation",
        payload=json.dumps(samples) + "\n",
    )


def generate(kit: Path, *, update: bool = False) -> None:
    """Check registry resolution and prove the committed artifacts are generated."""
    templates = kit / "templates" / "reference"
    templates.mkdir(parents=True, exist_ok=True)
    for source in TEMPLATE_INPUTS:
        shutil.copyfile(ROOT / source, templates / Path(source).name)
    for action, args in (
        ("check", ()),
        (
            "generate",
            ("-t", str(templates.parent), "reference", str(kit / "generated")),
        ),
    ):
        done = _weaver(kit, action, *args)
        if done.returncode:
            raise RuntimeError(done.stderr)
    for name, target in GENERATED.items():
        content = (kit / "generated" / name).read_text(encoding="utf-8").rstrip("\n") + "\n"
        if update:
            _publish(target, content.encode("utf-8"))
        elif target.read_text(encoding="utf-8") != content:
            raise ValueError(f"Generated telemetry artifact differs: {target.relative_to(ROOT)}")


def main() -> None:
    """Run the development preparation or the off-line registry check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "generate"))
    parser.add_argument("kit", type=Path)
    args = parser.parse_args()
    kit = args.kit.resolve()
    if args.action == "prepare":
        prepare(kit)
    else:
        generate(kit, update=args.action == "generate")


if __name__ == "__main__":
    main()
