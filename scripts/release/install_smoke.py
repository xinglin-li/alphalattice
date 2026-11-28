r"""Prepare release-wheel sidecars and prove the public install commands without an agent turn.

For the 0.2.9 release, extract the sidecars from the cut wheel with this command.

    uv run --locked python -m scripts.release.install_smoke \
        --wheel <release-wheel> --assets-only --out <asset-directory>

Attach `alphalattice-0.2.9-py3-none-any.whl`, `alphalattice-runtime-requirements.txt`
and `alphalattice-runtime-overrides.txt` to the GitHub release. Take all three
files from the same cut wheel.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import http.server
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from packaging.requirements import Requirement
from scripts.release import wheel_smoke

from alphalattice.capabilities.domain_contracts import spawn
from alphalattice.interface.local_application.thin_cli import AGENT_SESSION_VARIABLES

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = "alphalattice/_runtime/config/release/"
SIDECARS = ("runtime-requirements.txt", "runtime-overrides.txt")
UV_INSTALL = "https://docs.astral.sh/uv/getting-started/installation/"
PLUGIN = "alphalattice@alphalattice-tools"
FIRST_USE_WALL_SECONDS = 600
FIRST_USE_NODE = (
    "tests/integration/product_host/test_first_use_goal.py::"
    "test_a_first_use_decides_its_own_preparations_data_issue_and_its_preparation_goes_on"
)


def write_sidecars(wheel: Path, output: Path) -> None:
    """Extract the wheel's own exact lock and resolver overrides for its release."""
    with ZipFile(wheel) as package:
        for name in SIDECARS:
            (output / f"alphalattice-{name}").write_bytes(package.read(RUNTIME + name))


def run(
    command: list[str],
    folder: Path,
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run one bounded command in a run-owned folder."""
    return spawn.run(
        command,
        cwd=folder,
        env=env,
        check=check,
        text=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=300,
    )


def documented_launcher() -> tuple[list[str], dict[str, str]]:
    """Use the public command itself and check that both READMEs agree with it."""
    guide = (ROOT / "docs/public-source/install.md").read_text(encoding="utf-8")
    blocks = dict(re.findall(r"^\x60\x60\x60(powershell|bash)\n([^\n]+)\n", guide, re.MULTILINE))
    commands = [
        match.group()
        for block in blocks.values()
        if (match := re.search(r"uvx --isolated .*? --no-browser", block))
    ]
    assert len(commands) == 2 and commands[0] == commands[1]
    command = commands[0]
    for name in ("README.md", "README.zh-CN.md"):
        assert blocks["powershell"] in (ROOT / name).read_text(encoding="utf-8"), name
    return shlex.split(command), blocks


def documented_doctor() -> list[str]:
    """Use the doctor command the plugin and shared guide give to a fresh project."""
    guide = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    match = re.search(
        r"^python -m \S+ --project \. doctor --host claude-code$", guide, re.MULTILINE
    )
    assert match is not None
    return shlex.split(match.group())


def wheel_launcher(command: list[str], wheel: str, requirements: str, overrides: str) -> list[str]:
    """Use the documented launcher flags with sidecars from this build."""
    prefix = command[:-4]
    for flag, asset in zip(
        ("--from", "--with-requirements", "--overrides"),
        (wheel, requirements, overrides),
        strict=True,
    ):
        prefix[prefix.index(flag) + 1] = asset
    return prefix


def first_use(wheel: Path, output: Path, *, prepare_only: bool = False) -> None:
    """Warm the pinned wheel's test requirements or time its offline labelled trial."""
    bound = FIRST_USE_WALL_SECONDS
    command, _ = documented_launcher()
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="alphalattice-first-use-") as tmp:
        folder = Path(tmp)
        write_sidecars(wheel, folder)
        requirements, overrides = (folder / f"alphalattice-{name}" for name in SIDECARS)
        test_requirements = folder / "test-requirements.txt"
        # Test tooling adds packages to the environment. The wheel supplies its own runtime pins.
        runtime_packages = set()
        for line in requirements.read_text(encoding="utf-8").splitlines():
            if line.strip():
                requirement = Requirement(line)
                if requirement.marker is None or requirement.marker.evaluate():
                    runtime_packages.add(requirement.name)
        omit_runtime = [
            word for package in sorted(runtime_packages) for word in ("--no-emit-package", package)
        ]
        run(
            [
                "uv",
                "export",
                "--locked",
                "--offline",
                "--only-group",
                "dev",
                "--no-hashes",
                "--no-emit-project",
                *omit_runtime,
                "--output-file",
                str(test_requirements),
            ],
            ROOT,
        )
        argv = wheel_launcher(command, wheel.as_uri(), requirements.as_uri(), overrides.as_uri())
        argv[1:1] = [
            "--constraints",
            test_requirements.as_uri(),
            "--with",
            "pytest",
            "--with",
            "pytest-socket",
            "--with",
            "pytest-timeout",
            "--with",
            "pytest-xdist",
            "--with",
            "portalocker",
        ]
        env = os.environ | {"PYTHONPATH": ""}
        for key in ("VIRTUAL_ENV", "CONDA_PREFIX", "UV_RUN_RECURSION_DEPTH"):
            env.pop(key, None)
        if prepare_only:
            # Resolve the same isolated tool environment while the index is reachable.
            run([*argv[:-1], "python", "-I", "-c", "pass"], folder, env=env)
            return
        argv.insert(1, "--offline")
        env.update(ALPHALATTICE_NETWORK_DISABLED="1", ALPHALATTICE_TEST_FIRST_USE_WHEEL="1")
        junit = folder / "first-use.xml"
        # Load the installed package before exposing the checkout's test and devtools modules.
        pytest_entry = """
import sys
import sysconfig
from pathlib import Path

import alphalattice

assert Path(alphalattice.__file__).resolve().is_relative_to(
    Path(sysconfig.get_path("purelib")).resolve()
), "fresh_first_use.wheel_import_required"
root = Path(sys.argv[1])
sys.path.extend(map(str, (root, root / "src", root / "scripts")))

import pytest

raise SystemExit(pytest.main(sys.argv[2:]))
"""
        argv[-1:] = [
            "python",
            "-I",
            "-c",
            pytest_entry,
            str(ROOT),
            "-c",
            str(ROOT / "pyproject.toml"),
            "-o",
            "pythonpath=",
            "-m",
            "not real_evidence",
            "--disable-socket",
            "--allow-hosts=127.0.0.1,::1",
            "--basetemp",
            str(folder / "test"),
            "--junitxml",
            str(junit),
            str(ROOT / FIRST_USE_NODE),
        ]
        trial = label = None
        start = time.monotonic()
        try:
            result = spawn.run(argv, cwd=folder, env=env, timeout=bound, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            failure = "fresh_first_use.wall_limit_exceeded"
        else:
            cases = ET.parse(junit).findall(".//testcase") if junit.is_file() else []
            passed = (
                result.returncode == 0
                and len(cases) == 1
                and all(cases[0].find(tag) is None for tag in ("failure", "error", "skipped"))
            )
            properties = (
                {
                    item.get("name"): item.get("value")
                    for item in cases[0].findall("properties/property")
                }
                if passed
                else {}
            )
            trial = properties.get("first_use_trial")
            label = properties.get("first_use_result")
            failure = (
                "fresh_first_use.scenario_failed"
                if not passed
                else "fresh_first_use.trial_result_missing"
                if label != "DEVELOPMENT" or not trial
                else None
            )
        elapsed = time.monotonic() - start
        if elapsed > bound:
            failure = "fresh_first_use.wall_limit_exceeded"
        result_code = failure or label
        reading = failure or f"{label} trial {trial}"
        summary = f"Offline wheel first use: {reading} in {elapsed:.3f} s (limit {bound} s)."
        output.write_text(
            json.dumps(
                {
                    "result": result_code,
                    "trial_id": trial,
                    "wall_seconds": elapsed,
                    "limit_seconds": bound,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        print(summary, flush=True)
        if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(summary_path).open("a", encoding="utf-8") as report:
                report.write(summary + "\n")
        if failure:
            raise SystemExit(1)


def plugin(folder: Path, claude: str) -> dict[str, object]:
    """Install the tracked marketplace tree into isolated Claude Code configuration."""
    archive = folder / "marketplace.zip"
    run(["git", "archive", "--format=zip", "-o", str(archive), "HEAD"], ROOT)
    source = folder / "marketplace"
    with ZipFile(archive) as repository:
        repository.extractall(source)

    home = folder / "claude home"
    home.mkdir()
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("ANTHROPIC_", "CLAUDE_CODE_OAUTH"))
    } | {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "CLAUDE_CONFIG_DIR": str(home / "config"),
        "DISABLE_TELEMETRY": "1",
        "DISABLE_AUTOUPDATER": "1",
    }
    valid = run([claude, "plugin", "validate", str(source), "--json"], folder, env=env)
    report = json.loads(valid.stdout)
    components = report["provides"]
    assert set(components["skills"]) == {"alphalattice:alphalattice-research", "alphalattice:start"}
    agents = {
        f"alphalattice:{path.stem}"
        for path in (source / ".claude/agents").iterdir()
        if path.is_file() and path.name.startswith("alphalattice_") and path.suffix == ".md"
    }
    assert set(components["agents"]) == agents and not components["mcpServers"]
    run(
        [claude, "plugin", "install", "alphalattice", "--marketplace", str(source)],
        folder,
        env=env,
    )
    installed = run([claude, "plugin", "list", "--json"], folder, env=env)
    assert any(item["id"] == PLUGIN and item["enabled"] for item in json.loads(installed.stdout))
    return {"plugin": PLUGIN, "skills": components["skills"], "agents": sorted(agents)}


def absent_uv(block: str, shell: str, folder: Path) -> None:
    """Check the user-facing command with an empty search path."""
    if shell == "powershell":
        program = shutil.which("powershell.exe")
        args = ["-NoProfile", "-NonInteractive", "-Command", block]
    else:
        program = wheel_smoke.git_bash()
        args = ["--noprofile", "--norc", "-c", block]
    assert program
    result = run([program, *args], folder, env=os.environ | {"PATH": ""}, check=False)
    assert result.returncode and UV_INSTALL in result.stderr, result


def verify(wheel: Path, output: Path, claude: str) -> dict[str, object]:
    """Exercise both install paths against fresh workspaces, with no paid agent run."""
    command, blocks = documented_launcher()
    shell = "powershell" if os.name == "nt" else "bash"
    launcher = blocks[shell].replace(shlex.join(command), "{launcher}")
    assert "{launcher}" in launcher
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="AlphaLattice install ") as tmp:
        folder = Path(tmp)
        assets = folder / "release"
        assets.mkdir()
        shutil.copyfile(wheel, assets / wheel.name)
        write_sidecars(wheel, assets)
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=assets)
        with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                prefix = wheel_launcher(
                    command,
                    f"{base}/{wheel.name}",
                    *(f"{base}/alphalattice-{name}" for name in SIDECARS),
                )

                def read_doctor(project: Path, install: dict[str, str]) -> dict[str, object]:
                    doctor_env = (
                        os.environ
                        | install
                        | {
                            "ALPHALATTICE_NETWORK_DISABLED": "1",
                            "ALPHALATTICE_MODEL_STORE": str(folder / "models"),
                            "PYTHONPATH": "",
                        }
                    )
                    project_variables = ("VIRTUAL_ENV", "CONDA_PREFIX", "UV_RUN_RECURSION_DEPTH")
                    for variable in (
                        *project_variables,
                        *(name for _, name in AGENT_SESSION_VARIABLES),
                    ):
                        doctor_env.pop(variable, None)
                    doctor = run(
                        wheel_smoke.shell_command(
                            [*prefix[:-1], *documented_doctor()], shell, launcher=launcher
                        ),
                        project,
                        env=doctor_env,
                        check=False,
                    )
                    report = json.loads(doctor.stdout)
                    assert doctor.returncode == 2 and report["research_nonblocking"]
                    assert report["failure_code"] == "native_bridge.project_declaration_missing"
                    readiness = {item["key"]: item for item in report["host_readiness"]}
                    installed_leg = readiness["installed_leg"]["facts"]
                    assert installed_leg["path_owner"] == "ACTIVE_ENVIRONMENT"
                    interpreter = Path(readiness["python"]["facts"]["interpreter"])
                    assert not interpreter.is_relative_to(Path(install["UV_TOOL_DIR"]))
                    return {
                        "exit": doctor.returncode,
                        "attachment": report["failure_code"],
                        "installed_leg": installed_leg["path_owner"],
                        "version": installed_leg["version"],
                    }

                fresh = folder / "New workspace with spaces"
                fresh.mkdir()
                absent_uv(blocks[shell], shell, fresh)
                fresh_env = {
                    "UV_TOOL_DIR": str(folder / "empty tools"),
                    "UV_TOOL_BIN_DIR": str(folder / "empty bin"),
                }
                launched = wheel_smoke.smoke(
                    prefix,
                    cwd=fresh,
                    fresh_shell=shell,
                    install_environment=fresh_env,
                    launch_workspace=command[-3],
                    launcher=launcher,
                )
                empty_project = folder / "Empty agent project"
                empty_project.mkdir()
                launched["pinned_doctor"] = read_doctor(empty_project, fresh_env)
                previous = folder / "Workspace with older install"
                previous.mkdir()
                bins = previous / "bin"
                prior_env = {
                    "UV_TOOL_DIR": str(previous / "tools"),
                    "UV_TOOL_BIN_DIR": str(bins),
                    "ALPHALATTICE_DATA_ROOT": str(folder / "older-machine-data"),
                    "PATH": os.pathsep.join((str(bins), os.environ["PATH"])),
                }
                run(
                    [
                        "uv",
                        "tool",
                        "install",
                        "--python",
                        "3.14",
                        wheel.as_uri(),
                        "--with-requirements",
                        (assets / "alphalattice-runtime-requirements.txt").as_uri(),
                        "--overrides",
                        (assets / "alphalattice-runtime-overrides.txt").as_uri(),
                    ],
                    previous,
                    env=os.environ | prior_env,
                )
                old_cli = bins / ("alphalattice.exe" if os.name == "nt" else "alphalattice")
                assert old_cli.is_file()
                with_older: dict[str, object] = {}

                def launch_beside_old_host() -> None:
                    with_older.update(
                        wheel_smoke.smoke(
                            prefix,
                            cwd=previous,
                            fresh_shell=shell,
                            install_environment=prior_env,
                            launch_workspace=command[-3],
                            launcher=launcher,
                        )
                    )
                    assert (previous / "alphalattice-data").is_dir()
                    with_older["pinned_doctor"] = read_doctor(previous, prior_env)
                    older = run(
                        [str(old_cli), "--workspace", "./older-workspace", "workspace", "show"],
                        previous,
                        env=os.environ | prior_env,
                    )
                    assert json.loads(older.stdout)["outcome"] == "OK"

                legacy = wheel_smoke.smoke(
                    [str(old_cli)],
                    cwd=previous,
                    fresh_shell=shell,
                    install_environment=prior_env,
                    launch_workspace="./older-workspace",
                    while_serving=launch_beside_old_host,
                )
            finally:
                server.shutdown()
                thread.join(timeout=5)
        repo_link = folder / "Repo link workspace"
        repo_link.mkdir()
        plugin_folder = folder / "Plugin workspace"
        plugin_folder.mkdir()
        installed = plugin(plugin_folder, claude)
        checkout = wheel_smoke.smoke(
            ["uv", "run", "--project", str(ROOT), "--locked", "alphalattice"],
            cwd=repo_link,
            fresh_shell=shell,
            launch_workspace=command[-3],
        )
    return {
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "uv_missing": UV_INSTALL,
        "older_install": legacy,
        "plugin": installed,
        "pinned_wheel": launched,
        "pinned_wheel_with_older": with_older,
        "repo_link": checkout,
    }


def main() -> None:
    """Run the selected release preparation or install check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assets-only", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--first-use", action="store_true")
    modes.add_argument(
        "--prepare-first-use", action="store_true", help="Cache the trial's locked dependencies."
    )
    parser.add_argument("--claude")
    args = parser.parse_args()
    if args.assets_only:
        write_sidecars(args.wheel, args.out)
        return
    try:
        if args.first_use or args.prepare_first_use:
            first_use(args.wheel.resolve(), args.out.resolve(), prepare_only=args.prepare_first_use)
            return
        if not args.claude:
            parser.error("--claude is required for the install check")
        result = verify(args.wheel.resolve(), args.out.resolve(), args.claude)
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"install_smoke: {error.cmd} exited {error.returncode}\n"
            f"{error.stdout or ''}{error.stderr or ''}"
        ) from error
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
