"""Check an installed or editable command on a new workspace, offline and without a checkout cwd."""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import queue
import shlex
import shutil
import subprocess
import tempfile
import threading
import urllib.parse
import urllib.request
from pathlib import Path


def shell_command(command: list[str], shell: str | None) -> list[str]:
    """Start a new unprofiled host shell with no prior variables or activation."""
    if shell == "powershell":
        words = " ".join("'" + part.replace("'", "''") + "'" for part in command)
        return [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "& " + words + "; exit $LASTEXITCODE",
        ]
    if shell == "bash":
        return [git_bash(), "--noprofile", "--norc", "-c", "exec " + shlex.join(command)]
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


def smoke(command: list[str], *, cwd: Path, fresh_shell: str | None = None) -> dict[str, object]:
    """Exercise help, a fresh-workspace refusal, Host session and served workspace read."""
    environment = {**os.environ, "ALPHALATTICE_NETWORK_DISABLED": "1", "PYTHONPATH": ""}
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("CONDA_PREFIX", None)
    with tempfile.TemporaryDirectory(prefix="alphalattice-wheel-smoke-") as temporary:
        workspace = Path(temporary) / "workspace"
        help_run = subprocess.run(
            shell_command([*command, "--help"], fresh_shell),
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
        assert "workspace" in help_run.stdout and "serve" in help_run.stdout
        prefix = [*command, "--workspace", str(workspace)]
        before = subprocess.run(
            shell_command([*prefix, "workspace", "show"], fresh_shell),
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        response = json.loads(before.stdout)
        # A read never creates a Host: this named refusal is the same on both legs.
        assert (
            before.returncode == 4
            and response["failure_code"] == "local_client.service_not_running"
        )
        assert response["detail"] and response["next_action"]
        process = subprocess.Popen(
            [*prefix, "serve", "--no-browser", "--stop-on-stdin"],
            cwd=cwd,
            env=environment,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        lines: queue.Queue[str] = queue.Queue()

        def collect() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                lines.put(line)

        reader = threading.Thread(target=collect, daemon=True)
        reader.start()
        try:
            launch = ""
            while not launch:
                line = lines.get(timeout=60)
                if line.startswith("Portfolio Research workspace: "):
                    launch = line.removeprefix("Portfolio Research workspace: ").strip()
            cookies = http.cookiejar.CookieJar()
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(cookies)
            )
            with opener.open(launch, timeout=20) as opened:
                assert opened.status == 200
            address = urllib.parse.urlsplit(launch)
            base = f"{address.scheme}://{address.netloc}"
            with opener.open(base + "/api/session", timeout=20) as session:
                assert session.status == 200
                payload = json.loads(session.read())
                assert isinstance(payload, dict) and payload
            shown = subprocess.run(
                shell_command([*prefix, "workspace", "show"], fresh_shell),
                cwd=cwd,
                env=environment,
                text=True,
                capture_output=True,
                timeout=30,
                check=True,
            )
            answer = json.loads(shown.stdout)
            assert answer["outcome"] == "OK"
        finally:
            assert process.stdin is not None
            process.stdin.write("stop\n")
            process.stdin.flush()
            process.stdin.close()
            try:
                exit_code = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise
            reader.join(timeout=10)
        assert exit_code == 0
        return {
            "help_exit": 0,
            "fresh_workspace_show_exit": before.returncode,
            "fresh_workspace_show": response["failure_code"],
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
