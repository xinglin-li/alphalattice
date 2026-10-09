"""Compose the root CLI, Local Web and model sandbox for both runtime legs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root

ROOT: Path = resolve_playpen_root(Path(__file__))


def serve(arguments: list[str]) -> int:
    """Start the Local Web Host with its declared retrieval environment."""
    import argparse
    import importlib.util
    import os
    import subprocess

    from alphalattice.control.product_host.composition.research_workspace import (
        RESEARCH_WORKSPACE_MANIFEST_NAME,
        read_research_workspace_manifest,
    )
    from alphalattice.interface.local_application.retrieval_environment import interpreter_path

    from .web_launcher import main as start

    entry = argparse.ArgumentParser(add_help=False)
    entry.add_argument("--workspace", type=Path, required=True)
    selected, _rest = entry.parse_known_args(arguments)
    manifest = (
        read_research_workspace_manifest(selected.workspace)
        if (selected.workspace / RESEARCH_WORKSPACE_MANIFEST_NAME).is_file()
        else None
    )
    interpreter = interpreter_path()
    checkout = (ROOT / "pyproject.toml").is_file()
    packaged = manifest is not None and manifest.evidence_review is not None
    # The Host runs in the declared retrieval environment whenever it exists, so an Evidence
    # package installed while it runs is served by it; a package with none to run in refuses.
    if importlib.util.find_spec("fastembed") is None and (
        packaged or (checkout and interpreter.is_file())
    ):
        if not checkout or not interpreter.is_file():
            print(
                json.dumps(
                    {
                        "status": "REFUSED",
                        "failure_code": "local_client.retrieval_environment_required",
                        "next_action": "CREATE_DECLARED_RETRIEVAL_ENVIRONMENT",
                        "setup_command": [
                            sys.executable,
                            "-m",
                            "alphalattice.interface.local_application.retrieval_environment",
                            "--offline",
                        ],
                        "network_setup_command": [
                            sys.executable,
                            "-m",
                            "alphalattice.interface.local_application.retrieval_environment",
                        ],
                        "explanation": "Offline setup requires cached pinned packages. "
                        "Use network_setup_command only with dependency-download permission; "
                        "neither command supplies model or issuer authority.",
                    }
                )
            )
            return 2
        # Inherit cwd/stdin/env. The workspace and input file arguments retain
        # their caller-relative meaning; no source is loaded from a sibling tree.
        # The checkout shim's sys.path bootstrap does not reach a new interpreter.
        # Give that child the same checkout source, even when its environment has
        # dependencies installed without an editable package installation.
        environment = dict(os.environ)
        source = str(ROOT / "src")
        inherited_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source + os.pathsep + inherited_path if inherited_path else source
        )
        return subprocess.call(
            [
                str(interpreter),
                "-m",
                "alphalattice.control.product_host.composition.web_launcher",
                *arguments,
            ],
            env=environment,
        )
    return start(arguments)


def restore(
    workspace: Path,
    target: Path,
    *,
    workspace_id: str | None,
    generation_hash: str | None,
    root: Path | None,
) -> dict[str, object]:
    """Restore one backup generation from the backup root alone, with no Host (V328)."""
    from alphalattice.control.product_host.storage.backup import restore_from_root

    return restore_from_root(
        workspace, target, workspace_id=workspace_id, generation_hash=generation_hash, root=root
    )


def sandbox(workspace: Path, model_id: str, *, study: Path | None, keep: bool) -> dict[str, object]:
    """Try an agent's model on a copy of the workspace, at rest, and record the trial (EX).

    Its saved-object readback is shared with the checkout verification probe.
    """
    from .model_sandbox import run_sandbox

    answer: dict[str, object] = run_sandbox(workspace, model_id, study=study, keep=keep)
    return answer


def main(argv: list[str] | None = None) -> int:
    """Run the declared command and return its exit status."""
    arguments = sys.argv[1:] if argv is None else argv
    try:
        from alphalattice.interface.local_application.cli import main as root

        return int(root(arguments, serve=serve, restore=restore, sandbox=sandbox))
    except ModuleNotFoundError as error:
        # A runtime dependency this interpreter lacks, named with the locked setup that
        # provides it; a missing product module is a broken tree and is raised as one.
        if error.name is None or error.name.split(".")[0] in {"alphalattice", "scripts"}:
            raise
        print(
            json.dumps(
                {
                    "status": "REFUSED",
                    "failure_code": "local_runtime.dependency_unavailable",
                    "missing_module": error.name,
                    "next_action": "SYNC_LOCKED_ENVIRONMENT_AND_USE_ITS_INTERPRETER",
                    "setup_command": (
                        [
                            "uv",
                            "sync",
                            "--project",
                            str(ROOT),
                            "--locked",
                            "--all-extras",
                        ]
                        if (ROOT / "pyproject.toml").is_file()
                        else [
                            "uv",
                            "pip",
                            "install",
                            "--python",
                            sys.executable,
                            "alphalattice[data,data-live,live-evidence,quant,semantic]==0.1.3",
                        ]
                    ),
                    "interpreter": sys.executable
                    if not (ROOT / "pyproject.toml").is_file()
                    else str(
                        ROOT
                        / ".venv"
                        / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
                    ),
                    "claim_limit": "NO_INSTALLATION_OR_WORKSPACE_WORK_PERFORMED",
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
