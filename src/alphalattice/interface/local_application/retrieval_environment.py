"""Create the declared Playpen retrieval environment, reproducibly and locally.

The product's retrieval path needs FastEmbed and its closure on top of what this
workspace's committed `uv.lock` provides. The lock is not edited to add them:
this builds a second worktree-local environment from that lock and then installs
the pinned additions from `config/requirements-retrieval.lock`, so the default
`.venv` stays exactly as locked and this environment is reproducible from two
committed files in this tree. No parent project, lock, or interpreter is
consulted; a missing lock is a stop.

    uv run python scripts/create_retrieval_environment.py --offline

Acceptance that needs a real model runs under the interpreter this prints. No
interpreter is ever guessed: a caller either uses this path or does not. A setup
that fails answers in words with its way on: uv's own exit, a cache that
lacks a pinned package offline, uv missing, or a checkout it cannot write.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from alphalattice.interface.local_application.failure_codes import setup_failure
from alphalattice.kernel.shared_kernel.environment import recorded_environment
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root

PLAYPEN: Path = resolve_playpen_root(Path(__file__))
LOCK = PLAYPEN / "uv.lock"
ENVIRONMENT = PLAYPEN / ".venv-retrieval"
ADDITIONS = PLAYPEN / "config" / "requirements-retrieval.lock"
RECORD = "retrieval-locks.json"
GENERATION_ENV = "ALPHALATTICE_RETRIEVAL_GENERATION"
_lease_guard = threading.RLock()
_lease = None
_generation: str | None = None
_lease_stamps: tuple[tuple[int, int, int, int], ...] | None = None
LOCKED_SETUP_DISCLOSURE = (
    "I will install the retrieval packages pinned in config/requirements-retrieval.lock, "
    "downloading any missing packages as part of this setup."
)
LOCKED_SETUP_WORDS = (
    "Run the locked retrieval setup online as the agent's act, disclosing it in one line. "
    "An empty cache needs no second person question or renewed first-use network consent. "
    "Only software outside the lock needs the person's permission."
)


def _uv() -> str:
    executable = os.environ.get("UV", "uv")
    return executable


def interpreter_path() -> Path:
    """Return the interpreter of the declared retrieval environment."""
    if not (PLAYPEN / "pyproject.toml").is_file():
        return Path(sys.executable)
    return ENVIRONMENT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def lock_record() -> dict[str, str]:
    """Bind this environment to both committed dependency inputs."""
    return {
        "schema": "alphalattice.retrieval-locks.v1",
        "uv_lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
        "requirements_sha256": hashlib.sha256(ADDITIONS.read_bytes()).hexdigest(),
    }


def stale_reason() -> str | None:
    """Name why a recorded environment cannot be shown current."""
    try:
        recorded = (ENVIRONMENT / RECORD).read_text(encoding="utf-8")
    except FileNotFoundError:
        return "the environment has no lock record"
    except (OSError, UnicodeError):
        return "the environment lock record is unreadable"
    try:
        record = json.loads(recorded)
    except ValueError:
        return "the environment lock record is malformed"
    try:
        expected = lock_record()
    except OSError:
        return "the checkout's dependency locks are missing or unreadable"
    if not isinstance(record, dict) or any(
        record.get(key) != value for key, value in expected.items()
    ):
        return "the dependency locks differ from the environment"
    generation = record.get("generation")
    if (
        not isinstance(generation, str)
        or len(generation) != 32
        or any(c not in "0123456789abcdef" for c in generation)
    ):
        return "the environment has no valid installation generation"
    return None


@contextmanager
def environment_lock(*, exclusive: bool) -> Iterator[None]:
    """Exclude installers from each other and from processes using the environment."""
    path = PLAYPEN / ".retrieval-environment.lock"
    with path.open("a+b") as handle:
        if os.name == "nt":
            import ctypes  # lazy: Windows locking API is only available on Windows
            import msvcrt  # lazy: Windows locking API is only available on Windows
            from ctypes import wintypes  # lazy: Windows locking API is only available on Windows

            class Overlapped(ctypes.Structure):
                _fields_ = [
                    ("internal", ctypes.c_size_t),
                    ("high", ctypes.c_size_t),
                    ("offset", wintypes.DWORD),
                    ("offset_high", wintypes.DWORD),
                    ("event", wintypes.HANDLE),
                ]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.LockFileEx.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.POINTER(Overlapped),
            ]
            kernel.UnlockFileEx.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.POINTER(Overlapped),
            ]
            native = wintypes.HANDLE(msvcrt.get_osfhandle(handle.fileno()))
            overlap = Overlapped()
            if not kernel.LockFileEx(
                native, 1 | (2 if exclusive else 0), 0, 1, 0, ctypes.byref(overlap)
            ):
                code = ctypes.get_last_error()
                if code == 33:
                    raise RuntimeError(
                        "retrieval_environment.in_use: stop the Host or wait for setup"
                    )
                raise ctypes.WinError(code)
            try:
                yield
            finally:
                if not kernel.UnlockFileEx(native, 0, 1, 0, ctypes.byref(overlap)):
                    raise ctypes.WinError(ctypes.get_last_error())
        else:
            import fcntl  # lazy: POSIX locking API is unavailable on Windows

            try:
                fcntl.flock(  # type: ignore[attr-defined]  # POSIX API is absent in Windows stubs
                    handle.fileno(),
                    (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB,  # type: ignore[attr-defined]
                )
            except BlockingIOError as error:
                raise RuntimeError(
                    "retrieval_environment.in_use: stop the Host or wait for setup"
                ) from error
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]


def current_generation() -> str:
    """Read the validated installation while holding its environment lock."""
    reason = stale_reason()
    if reason:
        raise ValueError(
            f"retrieval_environment.stale: {reason}; run scripts/create_retrieval_environment.py"
        )
    generation = json.loads((ENVIRONMENT / RECORD).read_text(encoding="utf-8"))["generation"]
    if not isinstance(generation, str):
        raise ValueError("retrieval_environment.stale: invalid installation generation")
    return generation


def _loaded_from_environment() -> bool:
    root = ENVIRONMENT.resolve()
    return any(
        isinstance(source := getattr(module, "__file__", None), str)
        and Path(source).resolve().is_relative_to(root)
        for module in tuple(sys.modules.values())
    )


def check_host_dependencies() -> None:
    """Refuse an ambient unleased environment before importing the Host session."""
    with _lease_guard:
        root = ENVIRONMENT.resolve()
        exposed = any(path and Path(path).resolve().is_relative_to(root) for path in sys.path)
        if _lease is None and (exposed or _loaded_from_environment()):
            raise ValueError(
                "retrieval_environment.unprotected: restart the Host from the checkout environment"
            )
        if _lease is not None:
            retain_for_process()


def _proof_stamps() -> tuple[tuple[int, int, int, int], ...]:
    """Detect changes to the proved inputs without hashing them on every load."""
    values = (LOCK, ADDITIONS, ENVIRONMENT / RECORD)
    try:
        return tuple(
            (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) for s in (p.stat() for p in values)
        )
    except OSError as error:
        raise ValueError(
            "retrieval_environment.stale: a dependency input is unavailable; "
            "run scripts/create_retrieval_environment.py"
        ) from error


def retain_for_process(expected: str | None = None) -> str:
    """Keep a shared lease until process exit; loaded native modules cannot be unloaded safely."""
    global _lease, _generation, _lease_stamps
    with _lease_guard:
        if _lease is None:
            lease = environment_lock(exclusive=False)
            lease.__enter__()
            try:
                stamps = _proof_stamps()
                generation = current_generation()
                if _proof_stamps() != stamps:
                    raise ValueError(
                        "retrieval_environment.changed: dependency locks changed during validation"
                    )
                if expected is not None and generation != expected:
                    raise ValueError(
                        "retrieval_environment.changed: restart the Host after rebuilding"
                    )
                if expected is None and _loaded_from_environment():
                    raise ValueError(
                        "retrieval_environment.unprotected: restart the Host before using "
                        "already imported retrieval dependencies"
                    )
            except BaseException:
                lease.__exit__(*sys.exc_info())
                raise
            _lease, _generation, _lease_stamps = lease, generation, stamps
            atexit.register(lease.__exit__, None, None, None)
        elif _proof_stamps() != _lease_stamps or (expected is not None and expected != _generation):
            raise ValueError("retrieval_environment.changed: restart the Host after rebuilding")
        return _generation


def fill_command(*, offline: bool = False) -> list[str]:
    """The agent's fill of this product's retrieval runtime, typed from any folder.

    A checkout builds the declared environment beside its lock; an installed distribution takes
    the pinned additions into its own environment. Either is the lock's own dependency; offline
    reads the local cache only. Locked downloads are the agent's disclosed setup act.
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
    spec = importlib.util.find_spec("fastembed")
    origin = getattr(spec, "origin", None)
    from_declared = origin is not None and Path(origin).resolve().is_relative_to(
        ENVIRONMENT.resolve()
    )
    if (
        from_declared
        or Path(sys.prefix).resolve() == ENVIRONMENT.resolve()
        or str(site) in sys.path
    ):
        retain_for_process()
    if spec is None and site.is_dir() and config.is_file():
        lines = config.read_text(encoding="utf-8").splitlines()
        cfg = {
            key.strip(): value.strip() for key, _, value in (line.partition("=") for line in lines)
        }
        served = cfg.get("version_info") or cfg.get("version") or ""
        if (served == version or served.startswith(version + ".")) and str(site) not in sys.path:
            retain_for_process()
            sys.path.append(str(site))
            importlib.invalidate_caches()
    return importlib.util.find_spec("fastembed") is not None


@contextmanager
def setup_environment(project: Path) -> Iterator[dict[str, str]]:
    """Give setup children writable project scratch without changing the caller's environment."""
    temporary = (
        TemporaryDirectory(prefix=".retrieval-setup-", dir=project)
        if os.name == "nt"
        else nullcontext(None)
    )
    with temporary as scratch:
        environment = dict(os.environ)
        if scratch is not None:
            environment.update(TEMP=scratch, TMP=scratch, TMPDIR=scratch)
        yield environment


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
    with environment_lock(exclusive=True):
        return _create_locked(offline=offline)


def _create_locked(*, offline: bool) -> Path:
    expected = lock_record()
    record = ENVIRONMENT / RECORD
    if ENVIRONMENT.is_dir():
        record.write_text("{}", encoding="utf-8")  # a failed rebuild cannot retain an old proof
    interpreter = interpreter_path()
    offline_flag = ("--offline",) if offline else ()
    with setup_environment(PLAYPEN) as environment:
        environment["UV_PROJECT_ENVIRONMENT"] = str(ENVIRONMENT)
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
    if lock_record() != expected:
        raise ValueError("retrieval_environment.locks_changed: rebuild using the current locks")
    record.write_text(
        json.dumps({**expected, "generation": uuid4().hex}, sort_keys=True) + "\n", encoding="utf-8"
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
    """A setup's failure in words, with its way on.

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
    if str(error).startswith("retrieval_environment.in_use:"):
        return {
            **setup_failure(error),
            "status": "REFUSED",
            "failure_code": "retrieval_environment.in_use",
            "detail": "Wait for the other setup to finish, or stop the Host "
            "using this environment.",
            "next_action": "WAIT_FOR_RETRIEVAL_SETUP_OR_STOP_HOST",
            "environment": str(environment),
            "next_commands": {"setup": fill_command()},
        }
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
        if kind == "retrieval_environment" and offline:
            detail = (
                f"uv could not build {place} from its local cache. Run the same setup "
                "without --offline. " + LOCKED_SETUP_WORDS
            )
            next_action = "RUN_LOCKED_DEPENDENCY_SETUP"
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
        "next_commands": {
            "help": _help_command(kind),
            **(
                {"setup": fill_command()}
                if kind == "retrieval_environment" and next_action == "RUN_LOCKED_DEPENDENCY_SETUP"
                else {}
            ),
        },
        **(
            {"who_decides": "AGENT", "disclosure": LOCKED_SETUP_DISCLOSURE}
            if kind == "retrieval_environment" and isinstance(error, subprocess.CalledProcessError)
            else {}
        ),
    }


def _help_command(kind: str) -> list[str]:
    """The setup's own help, its only read-only way on, as the person runs it."""
    if kind == "gpu_environment":
        return [sys.executable, "scripts/create_gpu_environment.py", "--help"]
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
    if not arguments.offline:
        print(LOCKED_SETUP_DISCLOSURE, file=sys.stderr)  # noqa: T201, TID251  # CLI stderr
    try:
        interpreter = create(offline=arguments.offline)
    except Exception as error:
        print(json.dumps(refusal(error, offline=arguments.offline), indent=1))  # noqa: T201  # setup command
        return 2
    print(interpreter)  # noqa: T201  # setup command
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
