"""Compose the root CLI, Local Web and model sandbox for both runtime legs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from alphalattice.interface.local_application.retrieval_environment import (
    LOCKED_SETUP_DISCLOSURE,
    LOCKED_SETUP_WORDS,
    fill_command,
    interpreter_path,
)
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root

from .model_sandbox import run_sandbox
from .web_launcher import attached_child
from .web_launcher import main as start

ROOT: Path = resolve_playpen_root(Path(__file__))


def serve(arguments: list[str]) -> int:
    """Start the Local Web Host with its declared retrieval environment."""
    from alphalattice.control.product_host.composition.research_workspace import (
        RESEARCH_WORKSPACE_MANIFEST_NAME,
        read_research_workspace_manifest,
    )

    from .first_use_road import serve_road

    entry = argparse.ArgumentParser(add_help=False)
    entry.add_argument("--workspace", type=Path, required=True)
    entry.add_argument("--new-research", action="store_true")
    selected, _rest = entry.parse_known_args(arguments)
    # Before anything starts: the project's completed first use in another workspace is its
    # daily road, unless the person said yes to separate new research.
    if not selected.new_research and (road := serve_road(selected.workspace)) is not None:
        print(json.dumps(road))  # noqa: T201  # bootstrap/lifecycle
        return 2
    arguments = [a for a in arguments if a != "--new-research"]
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
            print(  # noqa: T201  # bootstrap/lifecycle
                json.dumps(
                    {
                        "status": "REFUSED",
                        "failure_code": "local_client.retrieval_environment_required",
                        "next_action": "CREATE_DECLARED_RETRIEVAL_ENVIRONMENT",
                        "setup_command": fill_command(),
                        "network_setup_command": fill_command(offline=False),
                        "offline_setup_command": fill_command(offline=True),
                        "explanation": LOCKED_SETUP_WORDS,
                        "disclosure": LOCKED_SETUP_DISCLOSURE,
                    }
                )
            )
            return 2
        # Detach before a venv trampoline adds its own Job; that Job can mask the caller's.
        if "--detach" in arguments:
            return start(
                arguments, host_interpreter=interpreter, new_research=selected.new_research
            )
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
        return attached_child(
            [
                str(interpreter),
                "-m",
                "alphalattice.control.product_host.composition.web_launcher",
                *arguments,
            ],
            env=environment,
        )
    return (
        start(arguments, new_research=selected.new_research)
        if "--detach" in arguments
        else start(arguments)
    )


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
        print(  # noqa: T201  # bootstrap/lifecycle
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
                            "alphalattice[data,data-live,live-evidence,quant,semantic]==0.2.3",
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
