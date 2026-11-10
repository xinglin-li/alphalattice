"""Compose the root CLI, Local Web and model sandbox for both runtime legs."""

from __future__ import annotations

import argparse
import errno
import importlib.util
import json
import os
import sys
from contextlib import suppress
from importlib.machinery import ModuleSpec
from pathlib import Path

from alphalattice.interface.local_application.retrieval_environment import (
    GENERATION_ENV,
    LOCKED_SETUP_DISCLOSURE,
    LOCKED_SETUP_WORDS,
    current_generation,
    environment_lock,
    fill_command,
    interpreter_path,
)
from alphalattice.interface.local_application.retrieval_environment import (
    stale_reason as retrieval_environment_stale_reason,
)
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root

from .model_sandbox import run_sandbox
from .web_launcher import attached_child
from .web_launcher import main as start

ROOT: Path = resolve_playpen_root(Path(__file__))


def _declared_retrieval_used(
    checkout: bool, interpreter: Path, fastembed: ModuleSpec | None
) -> bool:
    if not checkout or not interpreter.is_file():
        return False
    if fastembed is None or Path(sys.executable).resolve() == interpreter.resolve():
        return True
    return fastembed.origin is not None and Path(fastembed.origin).resolve().is_relative_to(
        interpreter.parent.parent.resolve()
    )


def _retrieval_refusal(detail: str, *, busy: bool = False) -> int:
    print(  # noqa: T201  # bootstrap/lifecycle
        json.dumps(
            {
                "status": "REFUSED",
                "failure_code": "local_client.retrieval_environment_in_use"
                if busy
                else "local_client.retrieval_environment_stale",
                "worktree": str(ROOT),
                "detail": detail,
                "next_action": "WAIT_FOR_RETRIEVAL_SETUP_OR_STOP_HOST"
                if busy
                else "REBUILD_DECLARED_RETRIEVAL_ENVIRONMENT",
                "setup_command": [
                    sys.executable,
                    str(ROOT / "scripts/create_retrieval_environment.py"),
                ],
            }
        )
    )
    return 2


def _serve_with_retrieval(arguments: list[str], interpreter: Path, new_research: bool) -> int:
    lease = environment_lock(exclusive=False)
    entered = False
    try:
        lease.__enter__()
        entered = True
        generation = current_generation()
    except (OSError, RuntimeError, ValueError) as error:
        if entered:
            lease.__exit__(*sys.exc_info())
        return _retrieval_refusal(
            str(error), busy=str(error).startswith("retrieval_environment.in_use:")
        )
    try:
        # Keep setup excluded until the child has acquired its own lease or exited.
        if "--detach" in arguments:
            return start(
                arguments,
                host_interpreter=interpreter,
                new_research=new_research,
                retrieval_generation=generation,
            )
        environment = dict(os.environ)
        source, inherited_path = str(ROOT / "src"), environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source + os.pathsep + inherited_path if inherited_path else source
        )
        environment[GENERATION_ENV] = generation
        return attached_child(
            [
                str(interpreter),
                "-m",
                "alphalattice.control.product_host.composition.web_launcher",
                *arguments,
            ],
            env=environment,
        )
    finally:
        lease.__exit__(None, None, None)


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
    fastembed = importlib.util.find_spec("fastembed")
    if (
        checkout
        and interpreter.is_file()
        and Path(sys.executable).resolve() == interpreter.resolve()
    ):
        return _serve_with_retrieval(arguments, interpreter, selected.new_research)
    if (
        fastembed is not None
        and _declared_retrieval_used(checkout, interpreter, fastembed)
        and (reason := retrieval_environment_stale_reason())
    ):
        return _retrieval_refusal(reason)
    # The Host runs in the declared retrieval environment whenever it exists, so an Evidence
    # package installed while it runs is served by it; a package with none to run in refuses.
    if fastembed is None and (packaged or (checkout and interpreter.is_file())):
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
        return _serve_with_retrieval(arguments, interpreter, selected.new_research)
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
    """Restore one backup generation from the backup root alone, with no Host."""
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


def _closed_output_pipe(error: OSError) -> bool:
    """Windows may report EINVAL when flushing a pipe whose reader has closed."""
    if isinstance(error, BrokenPipeError):
        return True
    if sys.platform != "win32" or error.errno != errno.EINVAL:
        return False
    import ctypes  # lazy: only Windows needs pipe-handle inspection.
    import msvcrt  # lazy: only Windows has CRT handles.

    try:
        handle = msvcrt.get_osfhandle(sys.stdout.fileno())  # noqa: TID251  # inspect CLI pipe
        return bool(
            ctypes.windll.kernel32.GetFileType(ctypes.c_void_p(handle)) == 3
        )  # FILE_TYPE_PIPE
    except (OSError, ValueError, AttributeError):
        return False


def main(argv: list[str] | None = None) -> int:
    """Run the declared command and return its exit status."""
    arguments = sys.argv[1:] if argv is None else argv
    try:
        from alphalattice.interface.local_application.cli import main as root

        result = int(root(arguments, serve=serve, restore=restore, sandbox=sandbox))
        sys.stdout.flush()  # noqa: TID251  # detect a closed CLI output pipe
        return result
    except OSError as error:
        if not _closed_output_pipe(error):
            raise
        # Avoid a second flush of buffered stdout during interpreter shutdown.
        with suppress(OSError, ValueError, AttributeError), open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), sys.stdout.fileno())  # noqa: TID251  # stop shutdown flush
        from alphalattice.interface.local_application import cli_contract  # lazy: pipe refusal.

        print(  # noqa: T201  # closed-pipe diagnostic
            cli_contract.worded("cli.output_pipe_closed"),
            file=sys.stderr,  # noqa: TID251  # CLI stderr
        )
        return 1
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
                            "alphalattice[data,data-live,live-evidence,quant,semantic]==0.2.6",
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
