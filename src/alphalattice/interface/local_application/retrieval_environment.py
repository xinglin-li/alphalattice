"""Create the declared Playpen retrieval environment, reproducibly and locally.

The product's retrieval path needs FastEmbed and its closure on top of what this
workspace's committed `uv.lock` provides. The lock is not edited to add them:
this builds a second worktree-local environment from that lock and then installs
the pinned additions from `config/requirements-retrieval.lock`, so the default
`.venv` stays exactly as locked and this environment is reproducible from two
committed files in this tree. No parent project, lock, or interpreter is
consulted; a missing lock is a stop.

    .venv/Scripts/python.exe scripts/create_retrieval_environment.py --offline

Acceptance that needs a real model runs under the interpreter this prints. No
interpreter is ever guessed: a caller either uses this path or does not. A setup
that fails answers in words with its way on (V539): uv's own exit, a cache that
lacks a pinned package offline, uv missing, or a checkout it cannot write.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from alphalattice.interface.local_application.failure_codes import setup_failure
from alphalattice.kernel.shared_kernel.environment import recorded_environment
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root

PLAYPEN: Path = resolve_playpen_root(Path(__file__))
LOCK = PLAYPEN / "uv.lock"
ENVIRONMENT = PLAYPEN / ".venv-retrieval"
ADDITIONS = PLAYPEN / "config" / "requirements-retrieval.lock"


def _uv() -> str:
    executable = os.environ.get("UV", "uv")
    return executable


def interpreter_path() -> Path:
    """Return the interpreter of the declared retrieval environment."""
    if not (PLAYPEN / "pyproject.toml").is_file():
        return Path(sys.executable)
    return ENVIRONMENT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def fill_command(*, offline: bool = True) -> list[str]:
    """The agent's fill of this product's retrieval runtime, typed from any folder.

    A checkout builds the declared environment beside its lock; an installed distribution takes
    the pinned additions into its own environment. Either is the lock's own dependency; offline
    reads the local cache only, and the network form needs the download permission.
    """
    module = "alphalattice.interface.local_application.retrieval_environment"
    return [sys.executable, "-m", module, *(["--offline"] if offline else [])]


def load() -> bool:
    """Whether FastEmbed imports in this process, reading the declared environment it lacks.

    A Host started before the environment was filled reads it with no restart: its site
    packages follow this interpreter's own, and only while it serves this Python version.
    """
    importlib.invalidate_caches()
    running = str(recorded_environment()["python"]).split(".")
    config, version = ENVIRONMENT / "pyvenv.cfg", ".".join(running[:2])
    site = ENVIRONMENT / (
        "Lib/site-packages" if os.name == "nt" else f"lib/python{version}/site-packages"
    )
    if importlib.util.find_spec("fastembed") is None and site.is_dir() and config.is_file():
        lines = config.read_text(encoding="utf-8").splitlines()
        cfg = {
            key.strip(): value.strip() for key, _, value in (line.partition("=") for line in lines)
        }
        served = cfg.get("version_info") or cfg.get("version") or ""
        if served.startswith(version + ".") and str(site) not in sys.path:
            sys.path.append(str(site))
            importlib.invalidate_caches()
    return importlib.util.find_spec("fastembed") is not None


def create(*, offline: bool) -> Path:
    """Build the environment; return the interpreter that owns it."""
    if not (PLAYPEN / "pyproject.toml").is_file():
        # An installed distribution already owns its environment; pinned additions
        # enter that environment, never the package or a workspace.
        subprocess.run(
            (
                _uv(),
                "pip",
                "install",
                "--python",
                sys.executable,
                "-r",
                str(ADDITIONS),
                *(("--offline",) if offline else ()),
            ),
            check=True,
        )
        return Path(sys.executable)
    if not LOCK.is_file():
        raise FileNotFoundError(LOCK)
    interpreter = interpreter_path()
    environment = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(ENVIRONMENT)}
    offline_flag = ("--offline",) if offline else ()
    subprocess.run(
        (
            _uv(),
            "sync",
            "--project",
            str(PLAYPEN),
            "--locked",
            "--all-extras",
            "--group",
            "dev",
            *offline_flag,
        ),
        check=True,
        cwd=PLAYPEN,
        env=environment,
    )
    subprocess.run(
        (
            _uv(),
            "pip",
            "install",
            "--python",
            str(interpreter),
            "-r",
            str(ADDITIONS),
            *offline_flag,
        ),
        check=True,
        cwd=PLAYPEN,
        env=environment,
    )
    return interpreter


def refusal(
    error: Exception,
    *,
    offline: bool,
    kind: str = "retrieval_environment",
    environment: Path = ENVIRONMENT,
    locks: tuple[Path, ...] = (LOCK,),
) -> dict[str, object]:
    """A setup's failure in words, with its way on (V539).

    The retrieval environment's and the GPU environment's alike: uv's own exit (offline, a
    pinned package the cache lacks), a committed lock or the checkout missing, uv not found,
    or a directory this process cannot write.

    Args:
        error: What stopped the setup.
        offline: Whether the setup read the local cache only.
        kind: The code's owner, `retrieval_environment` or `gpu_environment`.
        environment: The environment being built.
        locks: The committed files it is built from.

    Returns:
        The refusal: its code, words, way on and the environment it was building.
    """
    place = environment.name
    missing = [path.name for path in locks if not path.is_file()]
    if isinstance(error, subprocess.CalledProcessError):
        step = "uv pip install" if error.cmd[1] == "pip" else f"uv {error.cmd[1]}"
        code = f"{kind}.setup_failed:{step} exited {error.returncode}"
        detail = f"uv could not build {place}; its own output above names why. " + (
            "With --offline it reads only the local cache, and a pinned package the cache lacks "
            "stops it: run the same setup without --offline, with the person's permission to "
            "download, which fetches it."
            if offline
            else "Run it again once what uv names is resolved; a checkout this process cannot "
            "write cannot hold the environment."
        )
        next_action = (
            "RUN_THE_NETWORK_SETUP_WITH_DOWNLOAD_PERMISSION" if offline else "RESOLVE_AND_RUN_AGAIN"
        )
    elif missing:
        code, next_action = f"{kind}.lock_missing", "RESTORE_THE_CHECKOUT"
        detail = (
            f"The checkout lacks {', '.join(missing)}, which {place} is built from; restore it."
        )
    elif isinstance(error, FileNotFoundError) and not environment.parent.is_dir():
        code, next_action = f"{kind}.checkout_missing", "RESTORE_THE_CHECKOUT"
        detail = f"The checkout {place} belongs in is not there; restore it."
    elif isinstance(error, FileNotFoundError):
        code, next_action = f"{kind}.uv_unavailable", "INSTALL_UV"
        detail = (
            "uv, which builds the environment, is not on this process's path (or UV names "
            "nothing); install uv, or name it in UV, and run the setup again."
        )
    elif isinstance(error, OSError):
        code = f"{kind}.unwritable:{type(error).__name__}"
        next_action = "RUN_WHERE_THE_CHECKOUT_CAN_BE_WRITTEN"
        detail = (
            f"The environment is built inside the checkout, at {place}, and this process could "
            "not write there; run the setup where the checkout can be written."
        )
    else:
        return {
            **setup_failure(error),
            "environment": str(environment),
            "next_commands": {"help": _help_command(kind)},
        }
    return {
        **setup_failure(error),
        "status": "REFUSED",
        "failure_code": code,
        "detail": detail,
        "next_action": next_action,
        "environment": str(environment),
        "next_commands": {"help": _help_command(kind)},
    }


def _help_command(kind: str) -> list[str]:
    """The setup's own help, its only read-only way on, as the person runs it (V590)."""
    if kind == "gpu_environment":
        return [sys.executable, "scripts/create_gpu_environment.py", "--help"]
    if (PLAYPEN / "pyproject.toml").is_file():
        return [sys.executable, "scripts/create_retrieval_environment.py", "--help"]
    return [
        sys.executable,
        "-m",
        "alphalattice.interface.local_application.retrieval_environment",
        "--help",
    ]


def main() -> int:
    """Run the declared command and return its exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="resolve the lock and the additions from the local uv cache only",
    )
    arguments = parser.parse_args()
    try:
        interpreter = create(offline=arguments.offline)
    except Exception as error:
        print(json.dumps(refusal(error, offline=arguments.offline), indent=1))
        return 2
    print(interpreter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
