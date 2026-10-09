"""The installed leg keeps the checkout's entry, resources, and declared exceptions (V429)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tomllib
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
from markdown_it import MarkdownIt
from scripts.release import wheel_smoke
from scripts.release.wheel_build import resource_files, runtime_requirements

from alphalattice.control.product_host.composition import (
    evidence_authority_setup,
    retrieval_pack_setup,
)
from alphalattice.interface.local_application import cli_contract, native_setup
from alphalattice.kernel.shared_kernel import project_layout

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "config/release/runtime-resources.json"
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


def test_every_shipped_instruction_link_reaches_its_checkout_target() -> None:
    """Every shipped instruction link reaches its checkout target."""
    from scripts.release.public_manifest import classify

    roots = {"AGENTS.md", "CLAUDE.md", "README.md"}
    scopes = ("docs/public-source/", ".agents/", ".claude/", ".codex/")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
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


def test_the_console_entry_and_runtime_exceptions_are_explicit() -> None:
    """requirement: one packaged console entry, existing backend, named development exceptions."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["tool"]["uv"]["package"] is True
    assert project["project"]["scripts"] == {
        "alphalattice": "alphalattice.control.product_host.composition.entry:main"
    }
    assert project["build-system"] == {
        "requires": ["setuptools==83.0.0"],
        "build-backend": "setuptools.build_meta",
    }
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(manifest["checkout_only"]) == CHECKOUT_ONLY
    assert set(manifest["development_source_hashes"]) == SOURCE_HASHES
    assert all(manifest["checkout_only"].values())
    assert all(manifest["development_source_hashes"].values())


def test_both_guides_start_with_the_editable_leg_and_name_the_installed_leg() -> None:
    """Both guides start with the editable leg and name the installed leg."""
    owner = ".agents/skills/alphalattice-research/references/operating.md"
    for name in ("AGENTS.md", "README.md"):
        assert f"({owner}#setup-and-launch)" in (ROOT / name).read_text(encoding="utf-8")
    guide = (ROOT / owner).read_text(encoding="utf-8")
    assert guide.index("uv sync --locked") < guide.index("wheel")
    assert "function alphalattice" not in guide
    assert "uv tool install" in guide and "--constraints" in guide
    assert "uv tool dir --bin" in guide and "fresh shell" in guide
    assert "Activate.ps1" not in guide and "$ws" not in guide
    assert "uv run alphalattice" in guide
    assert "python scripts/run_alphalattice.py" in guide
    assert "editable" in guide and "no checkout" in guide


def test_a_built_wheel_holds_the_exact_runtime_content_set(tmp_path: Path) -> None:
    """requirement: actual wheel entries and bytes hold every declared runtime resource."""
    wheel_value = os.environ.get("ALPHALATTICE_TEST_WHEEL")
    if wheel_value:
        wheel = Path(wheel_value)
    else:
        stale = ROOT / "build/lib/alphalattice/retired_v429_module.py"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("raise RuntimeError('retired staging file')\n", encoding="utf-8")
        subprocess.run(
            ["uv", "build", "--wheel", "--offline", "--out-dir", str(tmp_path)],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        wheel = next(tmp_path.glob("*.whl"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = project["tool"]["setuptools"]["package-data"]["alphalattice"]
    package = ROOT / "src/alphalattice"
    expected = {path.relative_to(ROOT / "src").as_posix() for path in package.rglob("*.py")}
    expected.update(
        path.relative_to(ROOT / "src").as_posix()
        for pattern in patterns
        for path in package.glob(pattern)
        if path.is_file()
    )
    inputs = resource_files()
    expected.update("alphalattice/_runtime/" + name for name in inputs)
    expected.add("alphalattice/_runtime/config/release/source-hashes.json")
    expected.add("alphalattice/_runtime/config/release/runtime-requirements.txt")
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
        assert "-e ." not in requirements.decode()
        assert not any(
            name.startswith(("scripts/", "tests/", "devtools/", "workspaces/")) for name in names
        )
        for name in ("workbench.html", "workbench-manifest.json"):
            assert f"alphalattice/interface/local_application/assets/{name}" in names


def test_resource_names_and_printed_commands_survive_the_installed_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement: physical installation directories never enter canonical source names."""
    installed = tmp_path / "site-packages/alphalattice/_runtime"
    source = installed.parent / "kernel/shared_kernel/source_identity.py"
    assert project_layout.source_root(installed) == tmp_path / "site-packages"
    assert (
        project_layout.resource_path(
            installed, "src/alphalattice/kernel/shared_kernel/source_identity.py"
        )
        == source
    )
    assert (
        project_layout.tracked_path(installed, source)
        == "src/alphalattice/kernel/shared_kernel/source_identity.py"
    )
    assert project_layout.tracked_path(installed, installed / "uv.lock") == "uv.lock"
    for name in ("", "../outside", str(tmp_path.resolve())):
        with pytest.raises(ValueError, match="resource_path_invalid"):
            project_layout.resource_path(installed, name)
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
def test_installed_configure_copies_guidance_unchanged_and_refuses_an_overwrite(
    host: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """requirement: copied guidance retains its bytes and every link on both legs."""
    resources = resource_files()
    guidance = [
        name
        for name in resources
        if (
            name.endswith(".md")
            and ("skills" in Path(name).parts or name.startswith(".claude/agents/"))
        )
        or (name.startswith(".codex/agents/") and name.endswith(".toml"))
    ]
    _assert_guidance_links_resolve(ROOT, guidance)
    installed = tmp_path / "site-packages/alphalattice/_runtime"
    for name in resources:
        destination = installed / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    project = tmp_path / "agent-project"
    monkeypatch.setattr(native_setup, "INSTALLED", True)
    monkeypatch.setattr(native_setup, "RESOURCE_ROOT", installed)
    monkeypatch.setattr(native_setup, "ROOT", project)
    monkeypatch.setattr(native_setup.sys, "argv", ["native", "configure", "--host", host])
    assert native_setup.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "LOCAL_DECLARATIONS_VALIDATED"
    assert (project / "AGENTS.md").read_bytes() == (ROOT / "AGENTS.md").read_bytes()
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
        assert (project / name).read_bytes() == (ROOT / name).read_bytes()
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
    assert native_setup.main() == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["reason"] == "native_bridge.existing_configuration_differs"
    assert refused["detail"] and refused["next_action"]
    assert (project / "AGENTS.md").read_text(encoding="utf-8") == "A person's own guide"
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
    git_bash = str(tmp_path / "Git/usr/bin/bash.exe")
    monkeypatch.setattr(wheel_smoke.shutil, "which", lambda name: git_bash)
    assert wheel_smoke.shell_command(["alphalattice", "--help"], "bash")[0] == git_bash
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path / "Windows"))
    launcher = str(tmp_path / "Windows/System32/bash.exe")
    monkeypatch.setattr(wheel_smoke.shutil, "which", lambda name: launcher)
    with pytest.raises(SystemExit, match="no Git Bash on PATH"):
        wheel_smoke.shell_command(["alphalattice", "--help"], "bash")
