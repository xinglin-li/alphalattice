"""Check an installed or editable command on a new workspace, offline and without a checkout cwd."""

from __future__ import annotations

import argparse
import contextlib
import http.cookiejar
import json
import os
import queue
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

import psutil

from alphalattice.kernel.shared_kernel import spawn

HOST_MODULE = "alphalattice.control.product_host.composition.web_launcher"
"""The module a detached Host runs as (web_launcher._detach's command)."""


def shell_command(
    command: list[str], shell: str | None, *, launcher: str | None = None
) -> list[str]:
    """Start a new unprofiled host shell with no prior variables or activation."""
    if shell == "powershell":
        words = " ".join("'" + part.replace("'", "''") + "'" for part in command)
        invocation = "& " + words
        script = launcher.replace("{launcher}", invocation) if launcher else invocation
        return [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script + "; exit $LASTEXITCODE",
        ]
    if shell == "bash":
        invocation = "exec " + shlex.join(command)
        script = launcher.replace("{launcher}", invocation) if launcher else invocation
        return [git_bash(), "--noprofile", "--norc", "-c", script]
    return command


def git_bash() -> str:
    """The bash that PATH names, as the person's own shell starts it.

    Windows process creation searches System32 before PATH, where WSL's launcher bash.exe would
    stand in for Git Bash; a PATH that still reaches only that launcher is refused by name.
    """
    bash = shutil.which("bash")
    system = Path(os.environ.get("SYSTEMROOT", "C:/Windows")) / "System32"
    if bash is None or Path(bash).parent.resolve() == system.resolve():
        raise SystemExit(
            f"wheel_smoke: no Git Bash on PATH (found {bash or 'none'}); run the bash smoke from "
            "Git Bash, or put Git Bash's bin first on PATH"
        )
    return bash


def read_launch(lines: queue.Queue[str | None], *, timeout: float = 60) -> str:
    """Read the launch announcement within one overall wall-time bound."""
    deadline = time.monotonic() + timeout
    last_line = ""
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=max(0.001, deadline - time.monotonic()))
        except queue.Empty:
            break
        if line is None:
            raise ValueError(
                f"wheel_smoke.launch_exit: Host ended before its launch URL\n{last_line}"
            )
        last_line = line.rstrip() or last_line
        if line.startswith("Portfolio Research workspace: "):
            launch = line.removeprefix("Portfolio Research workspace: ").strip()
            if launch:
                return launch
    raise ValueError(
        f"wheel_smoke.launch_timeout: Host did not announce a launch URL within {timeout:g} seconds"
    )


def smoke(
    command: list[str],
    *,
    cwd: Path,
    fresh_shell: str | None = None,
    install_environment: dict[str, str] | None = None,
    launch_workspace: str | None = None,
    launcher: str | None = None,
    while_serving: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Exercise help, a fresh-workspace refusal, Host session and served workspace read."""
    environment = {
        **os.environ,
        **(install_environment or {}),
        "ALPHALATTICE_NETWORK_DISABLED": "1",
        "PYTHONPATH": "",
    }
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("CONDA_PREFIX", None)
    with (
        tempfile.TemporaryDirectory(
            prefix="alphalattice-wheel-smoke-", dir=cwd.parent if launch_workspace else None
        ) as temporary,
        spawn.InteractiveSession(timeout=300, stop_scope="tree") as session,
    ):
        workspace = Path(temporary) / "workspace"
        environment["ALPHALATTICE_DATA_ROOT"] = (install_environment or {}).get(
            "ALPHALATTICE_DATA_ROOT", str(Path(temporary) / "data-root")
        )
        prefix = [*command, "--workspace", launch_workspace or str(workspace)]

        def read_help() -> int:
            help_run = spawn.run(
                shell_command([*command, "--help"], fresh_shell, launcher=launcher),
                cwd=cwd,
                env=environment,
                text=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=30,
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            assert "workspace" in help_run.stdout and "serve" in help_run.stdout
            return help_run.returncode

        def read_without_host() -> tuple[int, str]:
            before = spawn.run(
                shell_command([*prefix, "workspace", "show"], fresh_shell, launcher=launcher),
                cwd=cwd,
                env=environment,
                text=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=30,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            response = json.loads(before.stdout)
            assert (
                before.returncode == 4
                and response["failure_code"] == "local_client.service_not_running"
            )
            assert response["detail"] and response["next_action"]
            return before.returncode, response["failure_code"]

        def hosted_workspace(process: psutil.Process, words: list[str]) -> Path | None:
            """The workspace a detached Host's command line names; None for any other process,
            and for one that ended or hides its folder while it is read."""
            if HOST_MODULE not in words or "--workspace" not in words[:-1]:
                return None
            named = Path(words[words.index("--workspace") + 1])
            try:
                return (named if named.is_absolute() else Path(process.cwd()) / named).resolve()
            except psutil.Error:
                return None

        def read_after_stop() -> tuple[int, str]:
            """The stopped workspace's next read starts its Host detached and answers. Where the
            platform refuses a detached Host (a Windows job that denies breakaway), the read
            answers NO_HOST and `serve --detach` names the refusal. Whatever the answers, a Host
            either started is stopped and its process ended before the smoke leaves its root."""

            def line(*words: str, check: bool = False) -> subprocess.CompletedProcess[str]:
                return spawn.run(
                    shell_command([*prefix, *words], fresh_shell, launcher=launcher),
                    cwd=cwd,
                    env=environment,
                    text=True,
                    capture_output=True,
                    stdin=subprocess.DEVNULL,
                    timeout=360,  # a Host start's own ready wait
                    check=check,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )

            def owned_hosts() -> list[psutil.Process]:
                """The detached Hosts serving this workspace, found by their command line, so
                on whatever data root a launcher sets and whatever they answer."""
                target = (cwd / (launch_workspace or ".")).resolve()
                return [
                    process
                    for process in psutil.process_iter(["cmdline"])
                    if hosted_workspace(process, process.info["cmdline"] or []) == target
                ]

            def stop_detached() -> None:
                """Stop each detached Host the reads started and wait for its process to end:
                `host stop` answers before the Host joins its workers and releases the
                workspace. No such process proves none started. The stop's own failure is kept,
                a Host the stop itself started (one ended between the scan and the stop) is
                waited for as well, and every survivor is ended; the smoke fails by name unless
                the stop answered and a final scan finds no Host left on the workspace."""
                hosts = owned_hosts()
                if not hosts:
                    return
                try:
                    stopped = line("host", "stop")
                    refusal = None if stopped.returncode == 0 else stopped.stdout
                except (OSError, subprocess.SubprocessError) as error:
                    refusal = repr(error)
                _, alive = psutil.wait_procs(hosts, timeout=120)
                started = [process for process in owned_hosts() if process not in alive]
                _, late = psutil.wait_procs(started, timeout=120)
                survivors = [*alive, *late]
                for process in survivors:
                    process.terminate()
                psutil.wait_procs(survivors, timeout=30)
                left = owned_hosts()
                for process in left:
                    process.kill()
                psutil.wait_procs(left, timeout=30)
                assert refusal is None and not survivors and not left, (
                    refusal,
                    [process.pid for process in survivors],
                    [process.pid for process in left],
                )

            try:
                read = line("workspace", "show")
                answer = json.loads(read.stdout)
                if read.returncode == 0:
                    assert answer["outcome"] == "OK"
                    return read.returncode, answer["outcome"]
                assert read.returncode == 4
                assert answer["failure_code"] == "local_client.service_not_running"
                refused = json.loads(line("serve", "--detach", "--no-browser").stdout)
                assert refused["failure_code"] == "local_web.detached_breakaway_denied"
                return read.returncode, refused["failure_code"]
            finally:
                stop_detached()

        if not launch_workspace:
            refusal_exit, refusal_code = read_without_host()
        start = [*prefix, "serve", "--no-browser", "--stop-on-stdin"]
        process = session.start(
            shell_command(start, fresh_shell, launcher=launcher) if launcher else start,
            cwd=cwd,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        lines: queue.Queue[str | None] = queue.Queue()

        def collect() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                lines.put(line)
            lines.put(None)

        reader = threading.Thread(target=collect, daemon=True)
        reader.start()
        try:
            launch = read_launch(lines, timeout=240) if launch_workspace else read_launch(lines)
            cookies = http.cookiejar.CookieJar()
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(cookies)
            )
            with opener.open(launch, timeout=20) as opened:
                assert opened.status == 200
            address = urllib.parse.urlsplit(launch)
            base = f"{address.scheme}://{address.netloc}"
            with opener.open(base + "/api/session", timeout=20) as session_response:
                assert session_response.status == 200
                payload = json.loads(session_response.read())
                assert isinstance(payload, dict) and payload
            shown = spawn.run(
                shell_command([*prefix, "workspace", "show"], fresh_shell, launcher=launcher),
                cwd=cwd,
                env=environment,
                text=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=30,
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            answer = json.loads(shown.stdout)
            assert answer["outcome"] == "OK"
            if while_serving:
                while_serving()
        finally:
            assert process.stdin is not None
            with contextlib.suppress(OSError):
                session.send("stop\n")
            with contextlib.suppress(OSError):
                session.close_input()
            try:
                exit_code = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                session.stop()
                raise
            reader.join(timeout=10)
        assert exit_code == 0
        help_exit = read_help()
        if launch_workspace:
            stopped_exit, stopped_answer = read_after_stop()
        return {
            "help_exit": help_exit,
            **(
                {
                    "stopped_workspace_show_exit": stopped_exit,
                    "stopped_workspace_show": stopped_answer,
                }
                if launch_workspace
                else {
                    "fresh_workspace_show_exit": refusal_exit,
                    "fresh_workspace_show": refusal_code,
                }
            ),
            "session_route_status": 200,
            "served_workspace_show_exit": shown.returncode,
            "host_stop_exit": exit_code,
            "network_disabled": True,
            "fresh_shell": fresh_shell,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fresh-shell", choices=["powershell", "bash"])
    args = parser.parse_args()
    try:
        with tempfile.TemporaryDirectory(prefix="alphalattice-outside-checkout-") as directory:
            result = smoke(args.command, cwd=Path(directory), fresh_shell=args.fresh_shell)
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"wheel_smoke: {error.cmd} exited {error.returncode}\n"
            f"{error.stdout or ''}{error.stderr or ''}"
        ) from error
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
