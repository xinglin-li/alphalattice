"""The installed leg keeps the checkout's entry, resources, and declared exceptions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import tomllib
import zipfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import pytest
from markdown_it import MarkdownIt
from scripts.release import install_smoke, wheel_smoke
from scripts.release.wheel_build import resource_files, runtime_requirements

from alphalattice.capabilities.domain_contracts import project_layout, spawn
from alphalattice.control.product_host.composition import (
    entry,
    evidence_authority_setup,
    retrieval_pack_setup,
)
from alphalattice.interface.local_application import (
    cli,
    cli_contract,
    cli_invocation,
    client,
    native_bridge,
    native_setup,
    thin_cli,
)
from alphalattice.kernel.config import DATA_ROOT_ENV, Change, override
from tests.portfolio_strategy_lab.cli_support import _agent_project

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "config/release/runtime-resources.json"
REACHED_BY = (entry,)
"""The console entry the repo-link case runs in a child, imported so a change to it reaches here."""
CHECKOUT_ONLY = {
    "scripts/build_local_web_ui.py",
    "scripts/create_gpu_environment.py",
    "scripts/materialize_claude_host.py",
    "scripts/u0_probe.py:private-harness",
    "model scaffold:adapter-source",
}
SOURCE_HASHES = {
    "scripts/check_alpha_verification_impact.py",
    "scripts/check_playpen.py",
    "scripts/broad_ensemble_research_closure.py",
    "scripts/run_broad_ensemble_live_model_closure.py",
    "scripts/run_heterogeneous_live_score_closure.py",
    "scripts/run_monthly_alpha_refit_research.py",
}


def _assert_guidance_links_resolve(
    root: Path, names: list[str], *, public_pages: bool = False
) -> None:
    """TE12: parse every shipped Skill/reference/card link in the consumer's layout."""
    parser = MarkdownIt("commonmark").enable("table")
    for name in names:
        source = root / name
        text = source.read_text(encoding="utf-8")
        if source.suffix == ".toml" and source.parent.name == "agents":
            text = tomllib.loads(text)["developer_instructions"]
        if public_pages:
            for imported in re.findall(r"^@([^\s]+)$", text, re.M):
                target = (source.parent / imported).resolve()
                assert target.is_relative_to(root.resolve()), name
                # The person's own layer is imported when present; no release ships it.
                assert target.is_file() or imported.startswith(".alphalattice/user/"), name
        tokens = parser.parse(text)
        while tokens:
            token = tokens.pop()
            tokens.extend(token.children or [])
            attribute = {"link_open": "href", "image": "src"}.get(token.type)
            if attribute is None:
                continue
            href = token.attrGet(attribute)
            assert href is not None, name
            url = urlsplit(href)
            if public_pages and (url.scheme or url.netloc):
                continue  # External destinations are outside this offline file/anchor check.
            assert not (url.scheme or url.netloc or url.query) and (
                public_pages or not url.fragment
            ), f"{name}: {href!r} needs validation beyond the offline local-file audit"
            target = (source.parent / unquote(url.path)).resolve() if url.path else source.resolve()
            assert target.is_relative_to(root.resolve()) and target.is_file(), (
                f"{name}: {href!r} does not resolve to a shipped file: {target}"
            )
            if url.fragment:
                anchors, repetitions = set(), {}
                headings = parser.parse(target.read_text(encoding="utf-8"))
                for index, heading in enumerate(headings):
                    if heading.type != "heading_open":
                        continue
                    title = headings[index + 1].content
                    slug = re.sub(r"[^\w -]", "", title.lower()).replace(" ", "-")
                    repeated = repetitions.get(slug, 0)
                    anchors.add(f"{slug}-{repeated}" if repeated else slug)
                    repetitions[slug] = repeated + 1
                assert unquote(url.fragment) in anchors, f"{name}: {href!r} has no heading"


@pytest.mark.proves("pipeline.shipped_instruction_links_reach_their_targets")
def test_every_shipped_instruction_link_reaches_its_checkout_target() -> None:
    """Every shipped instruction link reaches its checkout target."""
    from scripts.release.public_manifest import classify

    roots = {"AGENTS.md", "CLAUDE.md", "README.md"}
    scopes = ("docs/public-source/", ".agents/", ".claude/", ".codex/")
    tracked = spawn.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    names = sorted(
        name
        for name in tracked
        if name
        and (name in roots or name.startswith(scopes))
        and classify(name, set(), set(), set())[0] == "PUBLIC"
    )
    assert roots <= set(names)
    assert all(any(name.startswith(scope) for name in names) for scope in scopes)
    _assert_guidance_links_resolve(ROOT, names, public_pages=True)
    config = ROOT / ".codex/config.toml"
    for role in tomllib.loads(config.read_text(encoding="utf-8"))["agents"].values():
        assert (config.parent / role["config_file"]).is_file()


@pytest.mark.parametrize("owner", [evidence_authority_setup, retrieval_pack_setup])
def test_runtime_setup_owners_expose_the_checkout_flags(owner, capsys):
    """requirement: setup is a public src entry as well as a checkout shim."""
    with pytest.raises(SystemExit) as stopped:
        owner.main(["--help"])
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out


class _Loopback:
    """The loopback ports a Host answers on, and the connections made to them: no socket."""

    def __init__(self) -> None:
        self.live: set[int] = set()
        self.holders: dict[Path, int] = {}  # each data root's lock: the port of its Host
        self.next_port = 41000
        self.foreign: set[int] = set()  # a crashed Host's port another service answers on
        self.closing: set[int] = set()  # a Host that closes each connection after its answer
        self.sent: list[dict[str, object]] = []  # each request body, as it reached a port

    def connection(self, host: str, port: int, timeout: float) -> SimpleNamespace:
        ended: list[bool] = []  # this connection, once a closing Host has answered on it

        def connect() -> None:
            if port not in self.live:
                raise ConnectionRefusedError(port)

        def response() -> SimpleNamespace:
            answer = {"cli": "complete", "exit": 0, "stdout": "", "stderr": ""}
            answer = answer if "argv" in self.sent[-1] else {"cli": "local"}  # its handshake
            body = b"<html>another service</html>" if port in self.foreign else json.dumps(answer)
            ended.extend([True] * (port in self.closing))
            return SimpleNamespace(status=200, read=lambda: body)

        def request(_method: str, _path: str, body: bytes, headers: object) -> None:
            if ended:  # as a connection that may not reopen refuses before sending
                raise thin_cli.http.client.NotConnected
            self.sent.append(json.loads(body))

        return SimpleNamespace(
            connect=connect,
            close=lambda: None,
            port=port,
            request=request,
            getresponse=response,
        )


def _record(root: Path, workspace: Path, port: int) -> None:
    path = thin_cli.endpoint_path(root, workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_id": "local-research-connection",
                "workspace": str(workspace),
                "workspace_id": workspace.name,
                "url": f"http://127.0.0.1:{port}",
                "instance": "i" * 16,
                "token": "t" * 32,
            }
        ),
        encoding="utf-8",
    )


def _workspace(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / thin_cli.MANIFEST_NAME).write_text("{}", encoding="utf-8")
    return path.resolve()


RESIDENT = {
    "no-host-running": (1, ["thin:a"]),
    "a-host-running": (0, ["thin:a"]),
    "two-data-roots": (2, ["thin:a", "thin:b"]),
    "one-data-root-twice": (1, ["thin:a", "thin:a"]),
    "idle-past-its-bound": (1, ["thin:a"]),
    "a-crashed-hosts-record": (1, ["thin:a"]),
    "the-root-serves-another-workspace": (1, [None]),
    "help-starts-nothing": (0, [None]),
    "a-bound-sessions-first-command": (1, ["cli:a"]),
    "an-agent-sessions-first-command": (1, ["thin:a"]),
    "a-bound-command-meets-another-service": (0, [None]),
    "a-bound-command-after-the-host-closed": (0, [None]),
}
"""Each resident-Host case: the Host starts its commands asked for, and the path and workspace
each reached. A Host past its idle bound stays while a Task runs
(test_local_web_service.py:464-471), so it is the running Host's case here."""


def _command(
    argv: list[str], cwd: Path, *, thin: bool = True, hosted: bool = True, entry: bool = False
) -> str | None:
    """The path and workspace a command reached, as the console entry decides: the thin path,
    which starts the Host when none answers, else (a bound session goes there first) the full
    CLI's Host-backed operation, which takes the same start; `--help` starts none. `entry` sends
    it through the console's own entry, which answers from the Host it reached."""
    if entry:
        return None if thin_cli.invoke(argv) is None else "thin:" + cwd.name
    selected = thin_cli.select(argv, cwd) if thin else None
    serving = None if selected is None else thin_cli._serving(selected)
    if serving is not None:
        return "thin:" + Path(serving[0]["workspace"]).name
    if not hosted:
        return None
    try:
        return "cli:" + Path(client.attached(cwd).connection.workspace).name
    except client.LocalResearchClientError:  # the full CLI's NO_HOST answer
        return None


def _resident(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    """A command none answers starts its data root's Host once, through the CLI's own detached
    `serve`, and later commands on that root reach it; each root has its own Host."""
    starts, served = RESIDENT[case]
    loopback, started = _Loopback(), []
    workspaces = {name: _workspace(tmp_path / name) for name in ("a", "b", "other")}
    roots = {"a": tmp_path / "root-1", "b": tmp_path / "root-2", "other": tmp_path / "root-1"}

    def serve_detached(command: list[str], **options: object) -> SimpleNamespace:
        assert command[2:4] == ["alphalattice.control.product_host.composition.entry", "serve"]
        assert command[-2:] == ["--detach", "--no-browser"]
        workspace = Path(command[command.index("--workspace") + 1])
        started.append(workspace.name)
        if loopback.holders.get(thin_cli.data_root()) in loopback.live:
            return SimpleNamespace(returncode=2)  # the root's lock is held: the launcher refuses
        loopback.next_port += 1
        loopback.live.add(loopback.next_port)
        loopback.holders[thin_cli.data_root()] = loopback.next_port
        _record(thin_cli.data_root(), workspace, loopback.next_port)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(thin_cli.http.client, "HTTPConnection", loopback.connection)
    monkeypatch.setattr(thin_cli.subprocess, "run", serve_detached)
    if case == "a-host-running":
        _record(roots["a"], workspaces["a"], 40001)
        loopback.live.add(40001)
        loopback.holders[roots["a"]] = 40001
    elif case == "idle-past-its-bound":
        _record(roots["a"], workspaces["a"], 40001)  # it retired; its port no longer answers
    elif case == "a-crashed-hosts-record":
        _record(roots["a"], workspaces["other"], 40001)  # its lock went with its process
    elif case == "the-root-serves-another-workspace":
        _record(roots["a"], workspaces["other"], 40001)
        loopback.live.add(40001)
        loopback.holders[roots["a"]] = 40001
    elif case == "a-bound-command-meets-another-service":
        _record(roots["a"], workspaces["other"], 40001)  # its Host crashed; its port was taken
        loopback.live.add(40001)
        loopback.foreign.add(40001)
    elif case == "a-bound-command-after-the-host-closed":
        _record(roots["a"], workspaces["other"], 40001)  # it proves itself, then closes
        loopback.live.add(40001)
        loopback.closing.add(40001)
    names = {
        "two-data-roots": ["a", "b"],
        "one-data-root-twice": ["a", "a"],
    }.get(case, ["a"])
    bound = case in {
        "a-bound-command-meets-another-service",
        "a-bound-command-after-the-host-closed",
    }
    agent = bound or case == "an-agent-sessions-first-command"
    session = {"CLAUDE_CODE_SESSION_ID": "claude-1", "UV_RUN_RECURSION_DEPTH": None}
    session = session if agent else {}
    if agent:  # an installed console on the caller's PATH, as an agent's shell finds it
        console = tmp_path / "bin"
        console.mkdir()
        for launcher in ("alphalattice", "alphalattice.cmd"):
            (console / launcher).write_text("", encoding="utf-8")
            (console / launcher).chmod(0o755)
        monkeypatch.setenv("PATH", str(console) + os.pathsep + os.environ.get("PATH", ""))
        monkeypatch.setattr(tempfile, "tempdir", str(console))
    seen = []
    for name in names:
        with override(Change("data-root", {DATA_ROOT_ENV: str(roots[name]), **session})):
            help_only = case == "help-starts-nothing"
            argv = (
                ["--help"] if help_only else ["--workspace", str(workspaces[name]), "study", "list"]
            )
            argv = argv[2:] if bound else argv
            with thin_cli.one_command():  # one console command, as the entry holds it
                seen.append(
                    _command(
                        argv,
                        workspaces[name],
                        thin=case != "a-bound-sessions-first-command",
                        hosted=not help_only,
                        entry=agent,
                    )
                )
    assert (len(started), seen) == (starts, served)
    if agent:  # the Host counts the launch it dispatches; the console counts none of its own
        assert not list(console.rglob("*.count"))
    if bound:  # only the handshake, no caller data, on any connection
        assert loopback.sent == [{"operation": thin_cli.CLI_OPERATION}]


AGENT_SESSIONS = {
    "claude-code": {"CLAUDE_CODE_SESSION_ID": "claude-1"},
    "codex": {"CODEX_THREAD_ID": "codex-1"},
    "both": {"CLAUDE_CODE_SESSION_ID": "claude-1", "CODEX_THREAD_ID": "codex-1"},
    "neither": {},
    "malformed": {"CLAUDE_CODE_SESSION_ID": "not one token"},
}
"""Each case's session variables; both at once, or a value not one token, names no session."""


def _agent_session(case: str, request: pytest.FixtureRequest, monkeypatch, tmp_path) -> None:
    """An agent's CLI call, named or left to its binding, takes the thin path to its running Host,
    which counts the launch in the folder the caller's `tempfile` chose (TMPDIR=. here), anchored
    in the caller, and answers as the full CLI does; a call no binding names is the full CLI's."""
    live, capsys = request.getfixturevalue("live"), request.getfixturevalue("capsys")
    caller, hosts, console = _agent_project(tmp_path), tmp_path / "host-temp", tmp_path / "bin"
    for folder in (hosts, console):
        folder.mkdir()
    for name in ("alphalattice", "alphalattice.cmd"):  # an installed console on the caller's PATH
        (console / name).write_text("", encoding="utf-8")
        (console / name).chmod(0o755)
    monkeypatch.setenv("PATH", str(console) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    monkeypatch.chdir(caller)
    monkeypatch.setenv("TMPDIR", ".")
    for _, variable in thin_cli.AGENT_SESSION_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    for variable, value in AGENT_SESSIONS[case].items():
        monkeypatch.setenv(variable, value)
    chosen = tempfile.gettempdir
    monkeypatch.setattr(tempfile, "tempdir", None)  # the caller's own choice, from its TMPDIR
    monkeypatch.setattr(  # the Host's request thread stands for the Host's own process
        tempfile,
        "gettempdir",
        lambda: str(hosts) if threading.current_thread().name == "local-web-request" else chosen(),
    )
    anchored, temporary = [], cli_invocation.temporary

    def recorded() -> Path:  # the folder the Host counts a forwarded launch in
        folder = temporary()
        if threading.current_thread().name == "local-web-request":
            anchored.append(folder.is_absolute())
        return folder

    monkeypatch.setattr(cli_invocation, "temporary", recorded)

    def answer(printed: str) -> dict[str, object]:
        body = json.loads(printed.strip().splitlines()[-1])
        body.pop("timing")
        return body

    named = {"claude-code": "claude-code--claude-1", "codex": "codex--codex-1"}.get(case)
    launches, workspace = 1, ["--workspace", str(live.workspace)]
    cli.main([*workspace, "task", "list"], serve=lambda _: 99)
    capsys.readouterr()  # a session's first answer alone carries the Workbench step
    lines = [[*workspace, "--view", "full", "task", "list"], ["--view", "full", "task", "list"]]
    if case == "claude-code":  # a bound session's line leaves its workspace to the binding
        assert cli.main([*workspace, "session", "bind"], serve=lambda _: 99) == 0
        launches += 1
        capsys.readouterr()
    elif named is not None:
        lines.pop()
    for argv in lines:
        full_code, full = cli.main(argv, serve=lambda _: 99), answer(capsys.readouterr().out)
        thin_code = thin_cli.invoke(argv)
        if named is None and argv[0] != "--workspace":
            assert thin_code is None  # no session's binding names it: the full CLI answers
            continue
        launches += 2
        thin = answer(capsys.readouterr().out)
        assert (thin_code, thin.pop("session_launches", None)) == (
            full_code,
            None if full.pop("session_launches", None) is None else launches,
        )
        assert thin == full
    counts = {path.name: path.read_bytes() for path in caller.rglob("*.count")}
    assert list(hosts.iterdir()) == [] and all(anchored) and (named is None or anchored)
    assert counts == ({} if named is None else {f"{named}.count": b"1\n" * launches})


@pytest.mark.parametrize(
    "case",
    [
        "metadata",
        "build-child",
        pytest.param(
            "client-imports",
            marks=pytest.mark.proves(
                "local_application.console_entry_imports_only_the_standard_library"
            ),
        ),
        *(
            pytest.param(
                case, marks=pytest.mark.proves("local_application.one_host_serves_its_data_root")
            )
            for case in RESIDENT
        ),
        *(
            pytest.param(
                f"agent-session:{case}",
                marks=pytest.mark.proves("local_application.agent_calls_count_as_the_full_cli"),
            )
            for case in AGENT_SESSIONS
        ),
        pytest.param(
            "repo-link-first-use",
            marks=pytest.mark.proves(
                "local_application.repo_link_first_use_passes_the_install_smoke"
            ),
        ),
        pytest.param(
            "smoke-diagnostics",
            marks=pytest.mark.proves("local_application.smoke_diagnostics_keep_no_command_output"),
        ),
    ],
    ids=str,
)
def test_the_console_entry_and_runtime_exceptions_are_explicit(
    case, monkeypatch, tmp_path, request
) -> None:
    """requirement: one console entry and backend, named development exceptions; one Host per data
    root, which a command none answers starts and an agent session reaches as the full CLI would;
    the install smoke's repo-link leg passes wherever product tests run, keeping no output."""
    if case in RESIDENT:
        _resident(tmp_path, monkeypatch, case)
        return
    if case.startswith("agent-session:"):
        _agent_session(case.removeprefix("agent-session:"), request, monkeypatch, tmp_path)
        return
    if case == "smoke-diagnostics":
        key = "workbench-session-key"
        said = {
            "started": (
                0,
                f"STARTED\nPortfolio Research workspace: http://127.0.0.1:1/#key={key}",
                "",
            ),
            "refused": (2, json.dumps({"failure_code": "local_web.x", "detail": key}), ""),
            "crashed": (1, "", f"Traceback (most recent call last):\nFileNotFoundError: {key}"),
        }
        assert {
            name: wheel_smoke.outcome(subprocess.CompletedProcess([], *answer))
            for name, answer in said.items()
        } == {
            "started": (0, None, None),
            "refused": (2, "local_web.x", None),
            "crashed": (1, None, "FileNotFoundError"),
        }
        return
    if case == "repo-link-first-use":
        folder = tmp_path / "Repo link workspace"
        folder.mkdir()
        wheel_smoke.smoke(
            ["uv", "run", "--project", str(ROOT), "--locked", "alphalattice"],
            cwd=folder,
            launch_workspace=install_smoke.documented_launcher()[0][-3],
        )
        return
    if case == "client-imports":
        probe = """
import json
import sys
before = set(sys.modules)
from alphalattice.control.product_host.composition import entry

names = {name.split(".", 1)[0] for name in set(sys.modules) - before}
print(json.dumps({
    "third_party": sorted(names - sys.stdlib_module_names - {"__main__", "alphalattice"}),
    "full_cli": "alphalattice.interface.local_application.cli" in sys.modules,
}))
"""
        loaded = spawn.run(
            [sys.executable, "-I", "-B", "-c", probe],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
            timeout=15,
        )
        assert json.loads(loaded.stdout) == {"third_party": [], "full_cli": False}
        return
    if case == "build-child":
        from scripts.release import wheel_build

        calls = []
        monkeypatch.setattr(
            wheel_build.spawn,
            "run",
            lambda *a, **kw: (
                calls.append((a, kw)) or SimpleNamespace(stdout="locked requirements\n")
            ),
        )
        assert runtime_requirements() == "locked requirements\n"
        assert calls[0][0][0][:2] == ["uv", "export"]
        assert calls[0][1]["check"] and calls[0][1]["text"]
        return
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["tool"]["uv"]["package"] is True
    assert project["project"]["scripts"] == {
        "alphalattice": "alphalattice.control.product_host.composition.entry:main"
    }
    assert project["build-system"] == {
        "requires": ["setuptools==83.0.0", "psutil==7.2.2"],
        "build-backend": "setuptools.build_meta",
    }
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(manifest["checkout_only"]) == CHECKOUT_ONLY
    assert set(manifest["development_source_hashes"]) == SOURCE_HASHES
    assert all(manifest["checkout_only"].values())
    assert all(manifest["development_source_hashes"].values())
    named = {
        name
        for path in (ROOT / "src").rglob("*.py")
        for name in re.findall(r'runtime_json\(\s*"([^"]+)"', path.read_text(encoding="utf-8"))
    }
    assert {name for name in named if not name.startswith("src/")} <= set(resource_files())


def test_both_guides_start_with_the_editable_leg_and_name_the_installed_leg() -> None:
    """Control: Launch guides retain usable editable and installed command forms."""
    owner = ".agents/skills/alphalattice-research/references/operating.md"
    for name in ("AGENTS.md", "README.md"):
        assert f"({owner})" in (ROOT / name).read_text(encoding="utf-8")
    guide = (ROOT / owner).read_text(encoding="utf-8")
    assert guide.index("uv sync --locked") < guide.index("wheel")
    assert "function alphalattice" not in guide
    assert all(line in guide for line in native_bridge.TOOL_INSTALL)  # a first use offers it
    version = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]["version"]
    assert all(line.format(version=version) in guide for line in native_bridge.WHEEL_INSTALL)
    for name in (
        ".claude/skills/alphalattice-research/references/operating.md",
        "docs/public-source/upgrading.md",
    ):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert text.count(f"alphalattice-{version}-py3-none-any.whl") == 2
    assert "absolute prefix" in guide and "no shell update" in guide
    assert all(value in guide for value in ("uv tool dir", "fresh shell", "continue_here"))
    assert "Activate.ps1" not in guide and "$ws" not in guide
    assert "uv run alphalattice" in guide and "python scripts/run_alphalattice.py" in guide
    assert "editable" in guide and "no checkout" in guide


def test_a_built_wheel_holds_the_exact_runtime_content_set(tmp_path: Path) -> None:
    """requirement: actual wheel entries and bytes hold every declared runtime resource."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = project["tool"]["setuptools"]["package-data"]["alphalattice"]
    package = ROOT / "src/alphalattice"
    wheel_value = os.environ.get("ALPHALATTICE_TEST_WHEEL")
    if wheel_value:
        wheel = Path(wheel_value)
    else:
        source = tmp_path / "source"
        tracked = spawn.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        names = (
            set(filter(None, tracked))
            | {path.relative_to(ROOT).as_posix() for path in package.rglob("*.py")}
            | {
                path.relative_to(ROOT).as_posix()
                for pattern in patterns
                for path in package.glob(pattern)
                if path.is_file()
            }
        )
        for name in sorted(names):
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        stale = source / "build/lib/alphalattice/retired_v429_module.py"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("raise RuntimeError('retired staging file')\n", encoding="utf-8")
        spawn.run(
            ["uv", "build", "--wheel", "--offline", "--out-dir", str(tmp_path)],
            cwd=source,
            check=True,
            capture_output=True,
        )
        wheel = next(tmp_path.glob("*.whl"))
    expected = {path.relative_to(ROOT / "src").as_posix() for path in package.rglob("*.py")}
    expected.update(
        path.relative_to(ROOT / "src").as_posix()
        for pattern in patterns
        for path in package.glob(pattern)
        if path.is_file()
    )
    inputs = resource_files()
    assert "config/registries/messages.json" in inputs
    expected.update("alphalattice/_runtime/" + name for name in inputs)
    expected.add("alphalattice/_runtime/config/release/source-hashes.json")
    expected.add("alphalattice/_runtime/config/release/runtime-requirements.txt")
    expected.add("alphalattice/_runtime/config/release/runtime-overrides.txt")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert {name for name in names if ".dist-info/" not in name} == expected
        entry = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        assert archive.read(entry).decode().strip() == (
            "[console_scripts]\nalphalattice = "
            "alphalattice.control.product_host.composition.entry:main"
        )
        for name in inputs:
            assert archive.read("alphalattice/_runtime/" + name) == (ROOT / name).read_bytes()
        hashes = json.loads(archive.read("alphalattice/_runtime/config/release/source-hashes.json"))
        assert hashes == {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_HASHES
        }
        requirements = archive.read("alphalattice/_runtime/config/release/runtime-requirements.txt")
        assert requirements.decode() == runtime_requirements()
        overrides = archive.read("alphalattice/_runtime/config/release/runtime-overrides.txt")
        assert overrides.decode().splitlines() == project["tool"]["uv"]["override-dependencies"]
        assert "-e ." not in requirements.decode()
        assert not any(
            name.startswith(("scripts/", "tests/", "devtools/", "workspaces/")) for name in names
        )
        for name in ("workbench.html", "workbench-manifest.json"):
            assert f"alphalattice/interface/local_application/assets/{name}" in names


def test_resource_names_and_printed_commands_survive_the_installed_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: Installed layouts retain canonical resources and usable printed commands."""
    installed = tmp_path / "site-packages/alphalattice/_runtime"
    source = installed.parent / "capabilities/domain_contracts/source_identity.py"
    assert project_layout.source_root(installed) == tmp_path / "site-packages"
    assert (
        project_layout.resource_path(
            installed, "src/alphalattice/capabilities/domain_contracts/source_identity.py"
        )
        == source
    )
    assert (
        project_layout.tracked_path(installed, source)
        == "src/alphalattice/capabilities/domain_contracts/source_identity.py"
    )
    assert project_layout.tracked_path(installed, installed / "uv.lock") == "uv.lock"
    for name in ("", "../outside", str(tmp_path.resolve())):
        with pytest.raises(ValueError, match="resource_path_invalid"):
            project_layout.resource_path(installed, name)
    for name in resource_files():
        if name.startswith("config/registries/"):
            destination = installed / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setattr(project_layout, "resolve_playpen_root", lambda _: installed)
    assert project_layout.command_prefix() == ("alphalattice",)
    assert cli_contract.entry(tmp_path) == ("alphalattice", "--workspace", str(tmp_path.resolve()))
    from alphalattice.interface.local_application import client

    answer = client._write_bundle(
        tmp_path / "bundle",
        {"files": [{"name": "answer.json", "text": "{}"}]},
        prefix=cli_contract.entry(tmp_path),
    )
    assert answer["submit_arguments"][:2] == ["--workspace", str(tmp_path.resolve())]
    assert answer["submit_command"].lstrip("& ").startswith("alphalattice")


@pytest.mark.parametrize("host", ["codex", "claude-code"])
@pytest.mark.parametrize("off_path", [False, True, "source"])
def test_installed_configure_copies_guidance_unchanged_and_refuses_an_overwrite(
    host: str,
    off_path: bool | str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Configured cards resolve off-PATH tools and preserve other guidance and personal edits."""
    if off_path:
        uv = shutil.which("uv")
        assert uv
        bare, binary = tmp_path / "uv-only", tmp_path / "tool bin 雪"
        bare.mkdir()
        binary.mkdir()
        shutil.copy2(uv, bare / Path(uv).name)
        path = binary / ("alphalattice.exe" if os.name == "nt" else "alphalattice")
        path.write_bytes(b"")
        path.chmod(0o755)
        monkeypatch.setenv("PATH", str(bare))
        monkeypatch.setenv("UV_TOOL_BIN_DIR", str(binary))
        launcher = shutil.which("alphalattice", path=str(binary))
        assert launcher
        assert shutil.which("alphalattice") is None and project_layout.command_prefix() == (
            launcher,
        )

    def unchanged(name):
        shown, original = (project / name).read_bytes(), (ROOT / name).read_bytes()
        if not off_path or Path(name).suffix not in {".md", ".toml"}:
            assert shown == original
            return
        shown, original = shown.decode(), original.decode()
        prefixes = (
            "& '" + launcher.replace("'", "''") + "'",
            shlex.quote(Path(launcher).as_posix()),
        )
        shared = name == "AGENTS.md" or name.startswith(".agents/")
        if shared and re.search(r"(?:`|Bash\(|\n)alphalattice ", original):
            assert launcher in shown or Path(launcher).as_posix() in shown
            words = project_layout.runtime_json("config/registries/messages.json")["entries"][
                "setup.tool.prefixes"
            ]["words"]["en"].format(powershell=prefixes[0], posix=prefixes[1])
            block = (
                f"<!-- alphalattice-command-prefix -->\n{words}\n"
                "<!-- /alphalattice-command-prefix -->\n\n"
            )
            assert shown.count(block) == 1
            shown = shown.replace(block, "", 1)

        def restore(text):
            for prefix in prefixes:
                text = text.replace(prefix, "alphalattice")
            return text

        if name.endswith(".toml"):
            data = tomllib.loads(shown)
            shown = data["developer_instructions"]
            data["developer_instructions"] = restore(shown)
            assert data == tomllib.loads(original)
        else:
            assert restore(shown) == original
        if not shared and re.search(r"(?:`|Bash\(|\n)alphalattice ", original):
            assert launcher in shown or Path(launcher).as_posix() in shown
        if name.startswith(".claude/agents/"):
            rules = re.findall(r"Bash\(([^)]*):\*\)", shown.split("\n---", 1)[0])
            assert all(rule.startswith(prefixes[1] + " ") for rule in rules)

    resources = list(resource_files())
    if off_path == "source":
        resources.append(".venv/" + ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    guidance = [
        name
        for name in resources
        if name.endswith((".md", ".toml")) and ("skills" in Path(name).parts or "/agents/" in name)
    ]
    _assert_guidance_links_resolve(ROOT, guidance)
    installed = tmp_path / "site-packages/alphalattice/_runtime"
    for name in resources:
        destination = installed / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = Path(sys.executable) if name.startswith(".venv/") else ROOT / name
        destination.write_bytes(source.read_bytes())
    project = installed if off_path == "source" else tmp_path / "agent-project"
    monkeypatch.setattr(native_setup, "INSTALLED", off_path != "source")
    monkeypatch.setattr(native_setup, "RESOURCE_ROOT", installed)
    monkeypatch.setattr(native_setup, "ROOT", project)
    monkeypatch.setattr(native_setup.sys, "argv", ["native", "configure", "--host", host])
    assert native_setup.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "LOCAL_DECLARATIONS_VALIDATED"
    unchanged("AGENTS.md")
    host_directory = ".codex/" if host == "codex" else ".claude/"
    directories = [
        name
        for name in json.loads(MANIFEST.read_text(encoding="utf-8"))["directories"]
        if name.startswith((".agents/", host_directory))
    ]
    copied_resources = [
        name
        for name in resources
        if any(name.startswith(directory + "/") for directory in directories)
    ]
    for name in copied_resources:
        unchanged(name)
    if off_path:
        from scripts import materialize_claude_host as cards

        assert "`alphalattice workspace show`" in cards.command_block("alphalattice_risk")
    configured_guidance = [name for name in guidance if name in copied_resources]
    _assert_guidance_links_resolve(project, configured_guidance)
    _assert_guidance_links_resolve(project, ["AGENTS.md"], public_pages=True)
    if host == "codex":
        hooks = tomllib.loads((project / ".codex/config.toml").read_text(encoding="utf-8")).get(
            "hooks", {}
        )
    else:
        assert (project / "CLAUDE.md").read_bytes() == (ROOT / "CLAUDE.md").read_bytes()
        hooks = json.loads((project / ".claude/settings.json").read_text(encoding="utf-8")).get(
            "hooks", {}
        )
    assert not hooks
    assert native_setup.session_project(project, host) == project
    marker = project / host_directory / native_setup.PROJECT_DECLARATION_NAME
    assert json.loads(marker.read_bytes()) == {
        "schema": native_setup.PROJECT_DECLARATION_SCHEMA,
        "host": host,
    }
    native_setup.sys.argv[-1] = "claude-code" if host == "codex" else "codex"
    assert native_setup.main() == 0
    capsys.readouterr()
    native_setup.sys.argv[-1] = host
    assert native_setup.main() == 0
    capsys.readouterr()
    declaration = project / (".codex/config.toml" if host == "codex" else ".claude/settings.json")
    foreign = {"matcher": "Explore", "hooks": [{"type": "command", "command": "foreign-hook"}]}
    if host == "claude-code":
        declaration.write_text(
            json.dumps({"permissions": {"allow": ["Read"]}, "hooks": {"SubagentStart": [foreign]}}),
            encoding="utf-8",
        )
    else:
        with declaration.open("a", encoding="utf-8") as stream:
            stream.write(
                '\n[[hooks.SubagentStart]]\nmatcher = "Explore"\n'
                '[[hooks.SubagentStart.hooks]]\ntype = "command"\ncommand = "foreign-hook"\n'
            )
    foreign_bytes = declaration.read_bytes()
    assert native_setup.main() == 0
    assert json.loads(capsys.readouterr().out)["retired_hook_groups_removed"] == 0
    assert declaration.read_bytes() == foreign_bytes
    (project / "AGENTS.md").write_text("A person's own guide", encoding="utf-8")
    assert native_setup.main() == (0 if off_path == "source" else 2)
    refused = json.loads(capsys.readouterr().out)
    if off_path != "source":
        assert refused["reason"] == "native_bridge.existing_configuration_differs"
        assert refused["detail"] and refused["next_action"]
    assert (project / "AGENTS.md").read_text(encoding="utf-8") == "A person's own guide"
    if off_path == "source":
        return
    doctor = [
        native_setup.sys.executable,
        "-m",
        "alphalattice.interface.local_application.native_setup",
        "--project",
        str(project),
        "doctor",
    ]
    assert refused["next_commands"]["doctor"] == doctor
    assert declaration.read_bytes() == foreign_bytes

    binding_path = project / ".codex/native-research.local.json"
    binding_path.parent.mkdir(exist_ok=True)
    binding_path.write_text("{", encoding="utf-8")
    monkeypatch.setattr(native_setup.sys, "argv", ["native", "doctor", "--host", host])
    assert native_setup.main() == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["reason"] == "native_bridge.binding_invalid"
    assert refused["next_commands"]["doctor"] == doctor
    assert binding_path.read_text(encoding="utf-8") == "{"


def test_installed_scaffolding_refuses_before_writing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement: a developer command never writes inside an installed distribution."""
    from alphalattice.capabilities.alpha_modeling import model_scaffold

    installed = tmp_path / "alphalattice/_runtime"
    monkeypatch.setattr(model_scaffold, "resolve_playpen_root", lambda _: installed)
    with pytest.raises(ValueError, match=r"model_extension\.editable_checkout_required"):
        model_scaffold.scaffold_model(tmp_path / "not-read.yaml")
    assert not installed.exists()
    words = cli_contract.client_refusal("model_extension.editable_checkout_required")
    assert "editable checkout" in words.detail and words.next_action


def test_the_bash_smoke_starts_the_git_bash_that_path_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """regression: process creation finds System32's WSL bash.exe before PATH (the hosted Windows
    runner has one); the smoke starts the bash PATH names and refuses the WSL launcher by name."""
    ticks = iter((0.0, 1.0, 2.0, 3.0))
    monkeypatch.setattr(
        wheel_smoke, "time", SimpleNamespace(monotonic=lambda: next(ticks)), raising=False
    )
    with pytest.raises(ValueError, match=r"wheel_smoke\.launch_timeout"):
        wheel_smoke.read_launch(SimpleNamespace(get=lambda **_: "still starting"), timeout=2)
    git_bash = str(tmp_path / "Git/usr/bin/bash.exe")
    monkeypatch.setattr(wheel_smoke.shutil, "which", lambda name: git_bash)
    assert wheel_smoke.shell_command(["alphalattice", "--help"], "bash")[0] == git_bash
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path / "Windows"))
    launcher = str(tmp_path / "Windows/System32/bash.exe")
    monkeypatch.setattr(wheel_smoke.shutil, "which", lambda name: launcher)
    with pytest.raises(SystemExit, match="no Git Bash on PATH"):
        wheel_smoke.shell_command(["alphalattice", "--help"], "bash")

    # The smoke supplies a private data root and uses one bounded Host session.
    import io
    from contextlib import nullcontext

    calls, sessions = [], []

    class Session:
        def __init__(self, *, timeout, stop_scope):
            assert (timeout, stop_scope) == (300, "tree")
            self.sent = []
            sessions.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def start(self, command, **options):
            calls.append((command, options))
            return SimpleNamespace(
                stdin=io.StringIO(), stdout=io.StringIO(), wait=lambda timeout: 0
            )

        def send(self, data):
            self.sent.append(data)

        def close_input(self):
            pass

    def run(command, **options):
        calls.append((command, options))
        if "--help" in command:
            return SimpleNamespace(returncode=0, stdout="workspace serve")
        if "show" in command and not any("serve" in called for called, _ in calls):
            refusal = {  # the fresh workspace's read, before the Host serves
                "failure_code": "local_client.service_not_running",
                "detail": "d",
                "next_action": "n",
            }
            return SimpleNamespace(returncode=4, stdout=json.dumps(refusal))
        return SimpleNamespace(returncode=0, stdout='{"outcome":"OK"}')

    response = SimpleNamespace(status=200, read=lambda: b'{"session":"fixture"}')
    opener = SimpleNamespace(open=lambda *args, **kwargs: nullcontext(response))
    monkeypatch.setattr(wheel_smoke.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(wheel_smoke.spawn, "run", run)
    monkeypatch.setattr(wheel_smoke.spawn, "InteractiveSession", Session)
    monkeypatch.setattr(wheel_smoke, "read_launch", lambda lines: "http://127.0.0.1:1234/launch")
    monkeypatch.setattr(wheel_smoke.urllib.request, "build_opener", lambda *args: opener)
    answer = wheel_smoke.smoke(["fixture"], cwd=tmp_path)
    assert answer["host_stop_exit"] == 0 and sessions[0].sent == ["stop\n"]
    assert all(
        Path(options["env"]["ALPHALATTICE_DATA_ROOT"]).is_relative_to(tmp_path)
        for _, options in calls
    )
