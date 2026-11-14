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
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qsl, unquote, urlsplit

from alphalattice.interface.local_application.failure_codes import setup_failure
from alphalattice.kernel.shared_kernel.environment import recorded_environment
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root

PLAYPEN: Path = resolve_playpen_root(Path(__file__))
LOCK = PLAYPEN / "uv.lock"
ENVIRONMENT = PLAYPEN / ".venv-retrieval"
ADDITIONS = PLAYPEN / "config" / "requirements-retrieval.lock"
NETWORK = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "UV_HTTP_TIMEOUT",
    "UV_REQUEST_TIMEOUT",
    "UV_HTTP_RETRIES",
    "UV_CONCURRENT_DOWNLOADS",
    "UV_INDEX_URL",
    "UV_DEFAULT_INDEX",
    "UV_EXTRA_INDEX_URL",
    "UV_NATIVE_TLS",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "UV_OFFLINE",
)
"""The settings uv's downloads read: proxies, indexes, TLS roots and timeouts."""
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


def trace_path() -> Path:
    """Where the setup keeps each install step's verbose uv trace, beside what it builds."""
    if not (PLAYPEN / "pyproject.toml").is_file():
        return (
            Path(sys.executable).parent.parent / "retrieval-setup.log"
        )  # the filled environment's root
    return ENVIRONMENT.with_name(ENVIRONMENT.name + "-setup.log")


def network_environment(environment: dict[str, str] | None = None) -> dict[str, str]:
    """The network settings a uv step ran with, a credential in a URL masked."""
    found = os.environ if environment is None else environment
    return {
        key.upper(): _redact(value, found)
        for key, value in sorted(found.items())
        if key.upper() in NETWORK
    }


SHORT = 4
"""Characters below which a configured credential cannot be masked by value alone."""


def _redact(text: str, environment: Mapping[str, str]) -> str:
    """Remove URL credentials, query values and known environment secrets from diagnostics.

    A line that still holds a configured credential too short to mask by value is withheld
    whole, its variable named, so no configured credential reaches the trace.
    """
    text = re.sub(r"//[^/@\s]+@", "//***@", text)
    text = re.sub(r"(?i)(https?://[^\s\"'?#]+)\?[^\s\"']+", r"\1?***", text)
    text = re.sub(
        r"(?i)\b(authorization|proxy-authorization)\s*[:=]\s*(?:bearer|basic)\s+[^\s,]+",
        r"\1: ***",
        text,
    )
    text = re.sub(
        r"(?i)\b(password|passwd|secret|token|api[-_]?key)([\"']?\s*[:=]\s*[\"']?)[^\s,\"'}]+",
        r"\1\2***",
        text,
    )
    secrets = _secrets(environment)
    for value in sorted((value for value in secrets if len(value) >= SHORT), key=len, reverse=True):
        text = text.replace(value, "***")
    short = {value: key for value, key in secrets.items() if len(value) < SHORT}
    return "".join(_withheld(line, short) for line in text.splitlines(keepends=True))


def _withheld(line: str, short: Mapping[str, str]) -> str:
    """The line, or a marker in its place when it holds a short configured credential."""
    found = sorted({key for value, key in short.items() if value in line})
    if not found:
        return line
    ending = line[len(line.rstrip("\r\n")) :]
    return (
        f"[uv line withheld: it holds a credential from {', '.join(found)} too short to mask;"
        f" lengthen or unset it to see this line]{ending}"
    )


def _secrets(environment: Mapping[str, str]) -> dict[str, str]:
    """The credential values the environment configures, each with its variable's name.

    A value is secret when its variable's name says so, or when it is a configured URL's user,
    password or query value (encoded and decoded).
    """
    found: dict[str, str] = {}
    for key, value in environment.items():
        values: set[str] = set()
        if re.search(r"(?i)(password|passwd|secret|token|api.?key)", key):
            values.add(value)
        for url in re.findall(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s\"']+", value):
            try:
                parts = urlsplit(url)
            except ValueError:  # an unparsable URL (a bad IPv6 host) must not hide the failure
                continue  # it is named; its userinfo is still masked by _redact's own pattern
            if parts.password:
                values.update(part for part in (parts.username, parts.password) if part)
            for pair in parts.query.split("&"):
                values.add(pair.partition("=")[2])
            values.update(query for _, query in parse_qsl(parts.query))
        for secret in values | {unquote(secret) for secret in values}:
            if secret:
                found.setdefault(secret, key.upper())
    return found


def _run_uv(
    arguments: tuple[str, ...],
    trace: Path,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run one uv step verbosely, its whole trace and network settings appended to `trace`.

    A failed step raises CalledProcessError, as before; its trace stays for the refusal to name.
    """
    command = (_uv(), "-v", *arguments)
    effective = dict(os.environ) if env is None else env
    with trace.open("a", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "step": " ".join(arguments[:2]),
                    "command": [_redact(part, effective) for part in command],
                    "network": network_environment(env),
                }
            )
            + "\n"
        )
        log.flush()
        try:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=cwd,
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.CalledProcessError as error:
            log.write(_redact(error.stdout or "", effective))
            raise subprocess.CalledProcessError(error.returncode, command) from None
        log.write(_redact(result.stdout or "", effective))


def interpreter_path() -> Path:
    """Return the interpreter of the declared retrieval environment."""
    if not (PLAYPEN / "pyproject.toml").is_file():
        return Path(sys.executable)
    return ENVIRONMENT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


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
    if importlib.util.find_spec("fastembed") is None and site.is_dir() and config.is_file():
        lines = config.read_text(encoding="utf-8").splitlines()
        cfg = {
            key.strip(): value.strip() for key, _, value in (line.partition("=") for line in lines)
        }
        served = cfg.get("version_info") or cfg.get("version") or ""
        if (served == version or served.startswith(version + ".")) and str(site) not in sys.path:
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
        trace = trace_path()
        trace.write_text("", encoding="utf-8")
        _run_uv(
            (
                "pip",
                "install",
                "--python",
                sys.executable,
                "-r",
                str(ADDITIONS),
                *(("--offline",) if offline else ()),
            ),
            trace,
        )
        return Path(sys.executable)
    if not LOCK.is_file():
        raise FileNotFoundError(LOCK)
    interpreter = interpreter_path()
    offline_flag = ("--offline",) if offline else ()
    trace = trace_path()
    trace.write_text("", encoding="utf-8")
    with setup_environment(PLAYPEN) as environment:
        environment["UV_PROJECT_ENVIRONMENT"] = str(ENVIRONMENT)
        _run_uv(
            (
                "sync",
                "--project",
                str(PLAYPEN),
                "--locked",
                "--all-extras",
                "--group",
                "dev",
                *offline_flag,
            ),
            trace,
            cwd=PLAYPEN,
            env=environment,
        )
        _run_uv(
            (
                "pip",
                "install",
                "--python",
                str(interpreter),
                "-r",
                str(ADDITIONS),
                *offline_flag,
            ),
            trace,
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
    if isinstance(error, subprocess.CalledProcessError):
        verb = next(part for part in error.cmd[1:] if part != "-v")
        step = "uv pip install" if verb == "pip" else f"uv {verb}"
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
        "detail": detail
        + (
            f" uv's verbose trace of each step, with its network settings, is kept at "
            f"{trace_path()}."
            if _traced(kind, error)
            else ""
        ),
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
        **_traced(kind, error),
    }


def _traced(kind: str, error: Exception) -> dict[str, object]:
    """A failed retrieval install step's kept trace: its path, its tail and its network settings."""
    if kind != "retrieval_environment" or not isinstance(error, subprocess.CalledProcessError):
        return {}
    return {
        "trace": str(trace_path()),
        "trace_tail": _tail(trace_path()),
        "network": network_environment(),
    }


def _tail(trace: Path, lines: int = 20) -> list[str]:
    """The trace's last lines, where uv names a failed step's cause; none when it is unreadable."""
    try:
        return trace.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


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
