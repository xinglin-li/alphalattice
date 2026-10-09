"""Every product script that writes a location it chooses answers its failure in words (V539).

RR5d's agent met a PermissionError traceback from the pack installer in a sandbox that could
not write the application-data model store, and found `--store` by reading the script. Each
product script that writes a location its caller did not name -- the model store, the backup
root, the retrieval and GPU environments, the Local Web assets, the Claude Code host files, the
native bridge's files -- answers a write it cannot make with a refusal: its code, its words
and its way on, exit 2, never a traceback. `tests/structural/test_product_scripts_answer_their_
writes.py` holds the set.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from alphalattice.interface.local_application.cli_contract import refusal_words


def _blocked(tmp_path: Path) -> Path:
    """A directory no process can create: its parent is a file."""

    wall = tmp_path / "wall"
    wall.write_text("a file, not a directory", encoding="utf-8")
    return wall / "store"


def _refusal(capsys: pytest.CaptureFixture[str]) -> dict[str, str]:
    refusal: dict[str, str] = json.loads(capsys.readouterr().out)
    assert refusal["status"] == "REFUSED", refusal
    assert refusal["detail"] and refusal["next_action"], refusal
    return refusal


def test_the_pack_installer_refuses_an_unwritable_store_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """regression (V539, RR5d's FINDING 20:06): an unwritable model store is refused by name
    before any download, `--store <directory>` the way on, outside the checkout and any
    workspace; a store inside either is refused too, and nothing is created there."""

    from scripts import install_retrieval_pack as installer

    blocked = _blocked(tmp_path)
    arguments = ["--install", "hybrid-v2-minilm", "--network"]
    assert not blocked.resolve().is_relative_to(installer.PLAYPEN), "pytest's root is outside"
    assert installer.main(["--store", str(blocked), *arguments]) == 2
    refusal = _refusal(capsys)
    assert refusal["failure_code"] == "model_store.store_unwritable"
    assert "`--store <directory>`" in refusal["detail"]
    assert "ALPHALATTICE_MODEL_STORE" in refusal["detail"]
    assert refusal["next_action"] == "INSTALL_INTO_A_WRITABLE_STORE"
    assert refusal["store"] == str(blocked.resolve())

    inside = installer.PLAYPEN / "models-never-here"
    assert installer.main(["--store", str(inside), *arguments]) == 2
    assert _refusal(capsys)["failure_code"] == "model_store.store_inside_checkout"
    assert not inside.exists()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "research-workspace.json").write_text("{}", encoding="utf-8")
    assert installer.main(["--store", str(workspace / "models"), *arguments]) == 2
    assert _refusal(capsys)["failure_code"] == "model_store.store_inside_workspace"
    assert not (workspace / "models").exists()


def test_the_retrieval_environment_setup_answers_its_failure_in_words(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The retrieval environment setup answers its failure in words."""

    from scripts import create_gpu_environment as gpu
    from scripts import create_retrieval_environment as setup

    def exits(command: tuple[str, ...], **_options: object) -> None:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(setup.subprocess, "run", exits)
    monkeypatch.setattr(sys, "argv", ["create_retrieval_environment.py", "--offline"])
    assert setup.main() == 2
    refusal = _refusal(capsys)
    assert refusal["failure_code"] == "retrieval_environment.setup_failed:uv sync exited 1"
    assert refusal["next_action"] == "RUN_THE_NETWORK_SETUP_WITH_DOWNLOAD_PERMISSION"
    assert "without --offline" in refusal["detail"]
    assert refusal["environment"] == str(setup.ENVIRONMENT)
    assert gpu.main() == 2
    assert _refusal(capsys)["failure_code"] == "gpu_environment.setup_failed:uv sync exited 1"

    def missing(command: tuple[str, ...], **_options: object) -> None:
        raise FileNotFoundError(2, "The system cannot find the file specified")

    monkeypatch.setattr(setup.subprocess, "run", missing)
    assert setup.main() == 2
    assert _refusal(capsys)["failure_code"] == "retrieval_environment.uv_unavailable"


def test_the_local_web_build_answers_an_asset_it_cannot_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """requirement (V539): the build writes its assets beside their sources in the checkout; a
    write it cannot make is refused in words, the served assets left as they were."""

    from scripts import build_local_web_ui as build

    for name in build.FONT_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(build.ASSETS / name, target)
    monkeypatch.setattr(build, "ASSETS", tmp_path)

    def denied(*_arguments: object, **_options: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(build, "replace_with_retry", denied)
    assert build.main([]) == 2
    refusal = _refusal(capsys)
    assert refusal["failure_code"] == "local_web.assets_unwritable:PermissionError"
    assert refusal["next_action"] == "BUILD_WHERE_THE_CHECKOUT_CAN_BE_WRITTEN"
    assert not (tmp_path / build.MANIFEST).exists()


def test_the_claude_host_files_answer_a_write_they_cannot_make(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """requirement (V539): the host files are derived into the checkout's .claude directory; a
    write that cannot be made is refused in words, `--check` named for any drift left."""

    from scripts import materialize_claude_host as host

    # The derived files of a checkout whose .claude directory cannot be made.
    monkeypatch.setattr(host, "ROOT", tmp_path)
    monkeypatch.setattr(host, "expected_files", lambda: {_blocked(tmp_path) / "card.md": b"x"})
    assert host.main([]) == 2
    refusal = _refusal(capsys)
    assert refusal["failure_code"].startswith("claude_host.files_unwritable:")
    assert "`--check`" in refusal["detail"]
    assert refusal["next_action"] == "MATERIALIZE_WHERE_THE_CHECKOUT_CAN_BE_WRITTEN"


def test_the_native_bridge_answers_files_it_cannot_reach_in_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """requirement (V539): the bridge reads and writes the checkout's host directories, its
    binding and the workspace; a file it cannot reach is refused in words with its way on,
    never as a bare code."""

    from scripts import native_research as bridge

    # A checkout whose environment is there and whose host declarations are not.
    python = tmp_path / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    monkeypatch.setattr(bridge, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["native_research.py", "configure"])
    assert bridge.main() == 2
    refusal = _refusal(capsys)
    assert refusal["reason"] == "native_bridge.files_unavailable:FileNotFoundError"
    assert refusal["path_category"] == "HOST_DECLARATIONS"
    assert refusal["path"] == str(tmp_path / ".codex/config.toml")
    assert refusal["next_action"] == "USE_THE_WORKSPACE_HOST_OR_CHECK_THE_NAMED_LOCAL_PATH"


def test_a_backup_refuses_an_unwritable_root_by_name_and_words_it(tmp_path: Path) -> None:
    """requirement (V539): the backup root is the application data unless
    ALPHALATTICE_BACKUP_ROOT names another; a root the Host cannot write is refused by name
    with that way on, and the automatic backup after a data update reads the same words."""

    from alphalattice.control.product_host.storage.backup import (
        WorkspaceBackupError,
        WorkspaceBackups,
        backup_answer,
        record_backup_failure,
    )
    from alphalattice.interface.local_application.failure_codes import located_failure

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    backups = WorkspaceBackups(workspace, workspace_id="w1", root=_blocked(tmp_path))
    with pytest.raises(WorkspaceBackupError) as refused:
        backups.create(reason="REQUEST")
    code = "workspace_backup.root_unwritable"
    assert located_failure(refused.value, "workspace_backup.refused") == {"failure_code": code}
    words = refusal_words(code)
    assert "ALPHALATTICE_BACKUP_ROOT" in words["detail"]
    assert words["next_action"] == "SERVE_WITH_A_WRITABLE_BACKUP_ROOT"
    record_backup_failure(workspace, refused.value, at=datetime(2026, 10, 2, tzinfo=UTC))
    attempt = backup_answer(backups, None)["last_automatic_attempt"]
    assert attempt["failure_code"] == code and attempt["detail"] == words["detail"]
    assert attempt["next_action"] == words["next_action"]


@pytest.mark.parametrize(
    ("failure", "code", "kind"),
    [
        (
            ValueError("alternative_evidence.issuer_not_in_registry"),
            "alternative_evidence.issuer_not_in_registry",
            "PRODUCT",
        ),
        (
            ValueError("research_workspace.strategy_not_installed"),
            "research_workspace.strategy_not_installed",
            "PRODUCT",
        ),
        (
            RuntimeError("provider.model_unavailable:fixture-model"),
            "provider.model_unavailable:fixture-model",
            "PRODUCT",
        ),
        (
            ConnectionError("Authorization: Bearer NEVER-PRINT-ME"),
            "setup.network_failed:ConnectionError",
            "NETWORK",
        ),
        (
            FileNotFoundError(2, "NEVER-PRINT-ME", "secret-file"),
            "setup.file_unavailable:FileNotFoundError",
            "OS",
        ),
        (ValueError("invalid binding NEVER-PRINT-ME"), "setup.input_invalid:ValueError", "INPUT"),
        (
            RuntimeError("headers and source body NEVER-PRINT-ME"),
            "setup.unexpected_failure:RuntimeError",
            "UNEXPECTED",
        ),
    ],
)
def test_the_evidence_setup_preserves_its_cause_and_offers_a_read_only_way_on(
    failure, code, kind, monkeypatch, capsys, tmp_path
) -> None:
    """regression: setup keeps any owner's code and sanitized causes, with the exact binding
    and issuer scope for its next preflight; arbitrary text, headers and tokens stay private."""
    from scripts import materialize_evidence_cro_authority as setup

    from alphalattice.interface.local_application.cli_contract import command_table

    def refused(_arguments):
        raise failure

    monkeypatch.setattr(setup, "materialize", refused)
    binding = "a" * 64
    assert (
        setup.main(
            [
                "--workspace",
                str(tmp_path),
                "--acquire-sec",
                "--entities",
                "FIXTURE",
                "--research-input-id",
                "fixture-input",
                "--research-input-hash",
                binding,
                "--network-consent",
                "--install",
            ]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["failure_code"] == code
    assert payload["detail"] and payload["next_action"]
    assert payload["causes"][0]["kind"] == kind
    assert payload["context"]["research_input_hash"] == binding
    assert payload["context"]["issuers"] == ["FIXTURE"]
    assert payload["context"]["network_consent"] is True
    assert payload["next_requests"] == {"network": {"operation": "NETWORK_ACCESS"}}
    assert "NETWORK_ACCESS" in command_table()["fields"]
    command = payload["next_commands"]["preflight"]
    assert "--preflight" in command and "--network-consent" in command
    assert binding in command and "FIXTURE" in command and "--install" not in command
    shown = json.dumps(payload)
    assert "NEVER-PRINT-ME" not in shown and "secret-file" not in shown
    assert "explicit_sec_network_consent_required" not in shown
    if kind == "PRODUCT":
        assert payload["causes"][0]["code"] == code
        words = refusal_words(code) or refusal_words(f"setup.owner_refused:{code}")
        assert payload["detail"] == words["detail"]


def test_the_evidence_setup_names_http_status_without_response_or_request_secrets(
    monkeypatch, capsys, tmp_path
) -> None:
    """regression: a non-product HTTP error retains its status, never its URL, body or headers."""
    import httpx
    from scripts import materialize_evidence_cro_authority as setup

    def refused(_arguments):
        request = httpx.Request(
            "GET",
            "https://example.invalid/?token=NEVER-PRINT-ME",
            headers={"Authorization": "Bearer NEVER-PRINT-ME"},
        )
        response = httpx.Response(403, request=request, text="NEVER-PRINT-ME")
        raise httpx.HTTPStatusError("NEVER-PRINT-ME", request=request, response=response)

    monkeypatch.setattr(setup, "materialize", refused)
    assert setup.main(["--workspace", str(tmp_path)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["failure_code"] == "setup.http_failed:403"
    assert payload["causes"][0]["status"] == 403
    assert "403" in payload["detail"]
    assert payload["next_action"] and payload["next_commands"]["help"].endswith("--help")
    assert "NEVER-PRINT-ME" not in json.dumps(payload)


def test_the_evidence_setup_names_the_network_hold_without_offering_to_override_it(
    monkeypatch, capsys, tmp_path
) -> None:
    """regression: explicit consent and effective permission are different, and a held-offline
    process offers a read of the decision, never a request that would circumvent it."""
    from scripts import materialize_evidence_cro_authority as setup

    def refused(_arguments):
        raise ValueError("evidence_review.workspace_network_not_allowed")

    monkeypatch.setattr(setup, "materialize", refused)
    monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "1")
    # Only an acquisition meets the network hold, and only it is offered the decision (V590).
    acquisition = ["--acquire-sec", "--entities", "AAPL", "--network-consent"]
    assert setup.main(["--workspace", str(tmp_path), *acquisition]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["context"]["network_consent"] is True
    assert payload["network_access"]["network_allowed"] is False
    assert payload["network_access"]["decided_by"] == "OPERATOR_OFFLINE_SWITCH"
    assert payload["next_requests"] == {"network": {"operation": "NETWORK_ACCESS"}}
    assert "command gave SEC consent" in payload["detail"]
    assert payload["next_action"] == "RESTART_WITHOUT_OPERATOR_OFFLINE_SWITCH"
    assert "ALPHALATTICE_NETWORK_DISABLED=1" in payload["detail"]
    assert "restart the idle Host" in payload["detail"]
    assert "network set" not in payload["detail"]

    # V620: with no operator hold, the actual workspace control's way works.
    from alphalattice.control.workspace_runtime.network_access import set_network_access

    monkeypatch.delenv("ALPHALATTICE_NETWORK_DISABLED", raising=False)
    set_network_access(tmp_path, enabled=False)
    assert setup.main(["--workspace", str(tmp_path), *acquisition]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["network_access"]["decided_by"] == "WORKSPACE_CONTROL"
    assert "set the workspace control" in payload["detail"]
    assert payload["next_requests"]["set"]["network_enabled"]


@pytest.mark.parametrize(
    "script",
    [
        "build_local_web_ui",
        "create_gpu_environment",
        "create_retrieval_environment",
        "install_retrieval_pack",
        "materialize_claude_host",
        "materialize_evidence_cro_authority",
        "native_research",
    ],
)
def test_every_product_setup_script_words_an_unexpected_failure(
    script, monkeypatch, capsys, tmp_path
) -> None:
    """requirement: every setup entry in the structural set serves an unexpected failure's
    sanitized cause and way on, never a traceback or private message."""

    setup = importlib.import_module(f"scripts.{script}")

    def failed(*_args, **_kwargs):
        raise RuntimeError("private source payload NEVER-PRINT-ME")

    arguments = []
    if script in {"create_gpu_environment", "create_retrieval_environment"}:
        monkeypatch.setattr(setup, "create", failed)
        arguments = ["--offline"]
    elif script == "build_local_web_ui":
        monkeypatch.setattr(setup, "build", failed)
    elif script == "materialize_claude_host":
        monkeypatch.setattr(setup, "expected_files", failed)
    elif script == "materialize_evidence_cro_authority":
        monkeypatch.setattr(setup, "materialize", failed)
        arguments = ["--workspace", str(tmp_path)]
    elif script == "native_research":
        # Every binding read, by Session or by collection, passes through its entries.
        monkeypatch.setattr(setup.NativeResearchBinding, "binding_entries", failed)
        arguments = ["unbind", "--session-id", "fixture-session"]
    else:
        monkeypatch.setattr(setup.model_store, "recipe_readiness", failed)
        arguments = ["--status", "--store", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", [f"{script}.py", *arguments])
    assert setup.main() == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "REFUSED"
    assert payload["failure_code"] == "setup.unexpected_failure:RuntimeError"
    assert payload["causes"][0]["kind"] == "UNEXPECTED"
    assert payload["detail"] and payload["next_action"]
    assert "NEVER-PRINT-ME" not in json.dumps(payload)


_NETWORK_FLAGS = frozenset({"--acquire-sec", "--network", "--network-consent"})
"""What makes a setup command reach the network: an offline run offers none of them."""

_ANSWER_PATH = re.compile(r"`([a-z_]+(?:\.[a-z_]+)*)`")
"""A place in the answer, as a refusal's words name what they claim it carries."""


def _setup_failure(kind: str) -> Exception:
    """Each failure the setup refusals word (`setup_failure`), its text never to be shown."""

    import httpx

    if kind == "http":
        request = httpx.Request("GET", "https://example.invalid/?token=NEVER-PRINT-ME")
        response = httpx.Response(403, request=request, text="NEVER-PRINT-ME")
        return httpx.HTTPStatusError("NEVER-PRINT-ME", request=request, response=response)
    return {
        "file": lambda: FileNotFoundError(2, "NEVER-PRINT-ME", "secret-file"),
        "input": lambda: ValueError("invalid binding NEVER-PRINT-ME"),
        "network": lambda: ConnectionError("NEVER-PRINT-ME"),
        "command": lambda: subprocess.CalledProcessError(1, ["uv", "sync"]),
        "unexpected": lambda: RuntimeError("NEVER-PRINT-ME"),
        "owner": lambda: ValueError("fixture_owner.refused"),
    }[kind]()


def _setup_runs(setup: Any, script: str, tmp_path: Path) -> list[tuple[str, list[str], Any, str]]:
    """Each mode a setup script runs in: offline or network, its arguments and the public seam
    a failure is raised at."""

    if script in {"create_gpu_environment", "create_retrieval_environment"}:
        return [("offline", ["--offline"], setup, "create"), ("network", [], setup, "create")]
    if script == "install_retrieval_pack":
        store = ["--store", str(tmp_path / "store")]
        return [
            ("offline", ["--status", *store], setup.model_store, "recipe_readiness"),
            (
                "network",
                ["--install", "hybrid-v2-minilm", "--network", *store],
                setup.model_store,
                "hub_fetcher",
            ),
        ]
    if script == "build_local_web_ui":
        return [("offline", [], setup, "build")]
    if script == "materialize_claude_host":
        return [("offline", [], setup, "expected_files")]
    if script == "native_research":
        unbind = ["unbind", "--session-id", "fixture-session"]
        return [("offline", unbind, setup.NativeResearchBinding, "binding_entries")]
    workspace = ["--workspace", str(tmp_path / "workspace")]
    recorded = ["--source-artifact-root", str(tmp_path / "source"), "--source-set-hash", "a" * 64]
    return [
        (
            "network",
            [*workspace, "--acquire-sec", "--entities", "AAPL", "--network-consent"],
            setup,
            "materialize",
        ),
        ("offline", [*workspace, *recorded], setup, "materialize"),
        ("offline", [*workspace, "--rebind-installed"], setup, "materialize"),
    ]


@pytest.mark.parametrize(
    "kind", ["file", "input", "http", "network", "command", "unexpected", "owner"]
)
@pytest.mark.parametrize(
    "script",
    [
        "build_local_web_ui",
        "create_gpu_environment",
        "create_retrieval_environment",
        "install_retrieval_pack",
        "materialize_claude_host",
        "materialize_evidence_cro_authority",
        "native_research",
    ],
)
def test_every_setup_refusal_claims_only_what_it_carries_and_its_way_on_fits_its_mode(
    script, kind, monkeypatch, capsys, tmp_path
) -> None:
    """Every setup refusal claims only what it carries and its way on fits its mode."""

    setup = importlib.import_module(f"scripts.{script}")
    monkeypatch.setenv("ALPHALATTICE_SHELL", "posix")
    for mode, arguments, target, seam in _setup_runs(setup, script, tmp_path):

        def failed(*_args: object, **_kwargs: object) -> None:
            raise _setup_failure(kind)

        monkeypatch.setattr(target, seam, failed)
        monkeypatch.setattr(sys, "argv", [f"{script}.py", *arguments])
        assert setup.main() == 2, (script, mode)
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "REFUSED" and payload["detail"], (script, mode, payload)
        code = str(payload.get("failure_code") or payload.get("reason"))
        quoted = {code, code.partition(":")[2]}
        for place in set(_ANSWER_PATH.findall(payload["detail"])) - quoted:
            found: Any = payload
            for part in place.split("."):
                assert isinstance(found, dict) and part in found, (script, mode, place, payload)
                found = found[part]
            assert found, (script, mode, place, payload)
        if mode == "offline":
            requests = payload.get("next_requests") or {}
            assert all(value.get("operation") != "NETWORK_ACCESS" for value in requests.values())
            for command in (payload.get("next_commands") or {}).values():
                parts = shlex.split(command) if isinstance(command, str) else command
                assert not _NETWORK_FLAGS & set(parts), (script, mode, command)


def test_every_setup_refusal_word_names_its_way_on_and_only_places_every_setup_carries() -> None:
    """Every setup refusal word names its way on and only places every setup carries."""

    from alphalattice.interface import local_application

    table = json.loads(
        (Path(local_application.__file__).parent / "refusal_words.json").read_text("utf-8")
    )
    family = {code: words["detail"] for code, words in table.items() if code.startswith("setup.")}
    assert set(family) == {
        "setup.command_failed",
        "setup.file_unavailable",
        "setup.http_failed",
        "setup.input_invalid",
        "setup.network_failed",
        "setup.owner_refused",
        "setup.unexpected_failure",
    }
    for code, detail in family.items():
        places = set(_ANSWER_PATH.findall(detail))
        assert "next_commands" in places and places <= {"causes", "next_commands"}, code


def test_a_denied_store_reports_the_os_cause_before_any_download(monkeypatch, capsys, tmp_path):
    """regression: the original PermissionError route names the selected external store and
    sanitized errno/WinError, and keeps the writable-store preflight before network reads."""
    from scripts import install_retrieval_pack as setup

    def denied(**_kwargs):
        error = PermissionError(13, "NEVER-PRINT-ME")
        error.winerror = 5
        raise error

    fetched = []

    def fetch(*_args, **_kwargs):
        fetched.append(True)
        raise AssertionError("an unwritable store must not download")

    monkeypatch.setattr(setup.model_store.tempfile, "TemporaryFile", denied)
    monkeypatch.setattr(setup.model_store, "hub_fetcher", lambda **_kwargs: fetch)
    assert setup.main(["--store", str(tmp_path), "--install", "hybrid-v2-minilm", "--network"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["failure_code"] == "model_store.store_unwritable"
    assert payload["store"] == str(tmp_path.resolve())
    os_cause = next(cause for cause in payload["causes"] if cause["kind"] == "OS")
    assert os_cause["errno"] == 13 and os_cause["winerror"] == 5
    assert os_cause["message"] == os.strerror(13)
    assert "--store <directory>" in payload["detail"] and payload["next_action"]
    assert "NEVER-PRINT-ME" not in json.dumps(payload) and fetched == []


def test_setup_validation_and_asset_check_failures_have_words(monkeypatch, capsys):
    """regression: missing host declarations, native binding refusals and a failed node/check
    step are worded too, not just OS writes covered by the original writer pin."""
    from scripts import build_local_web_ui as assets
    from scripts import materialize_claude_host as host
    from scripts import native_research as bridge

    def missing():
        raise host.MaterializationError("skill.missing")

    monkeypatch.setattr(host, "expected_files", missing)
    assert host.main([]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["failure_code"] == "skill.missing"
    assert payload["detail"] and payload["next_action"] and payload["next_commands"]["check"]

    def unbound(*_args):
        raise bridge.NativeBridgeError("native_bridge.not_bound")

    monkeypatch.setattr(bridge.NativeResearchBinding, "binding_entries", unbound)
    monkeypatch.setattr(sys, "argv", ["native_research.py", "unbind", "--session-id", "fixture"])
    assert bridge.main() == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["failure_code"] == "native_bridge.not_bound"
    assert payload["detail"] and payload["next_action"] and payload["next_commands"]["doctor"]

    def failed():
        raise subprocess.CalledProcessError(
            1, ["node", "--check", "NEVER-PRINT-ME"], stderr="NEVER-PRINT-ME"
        )

    monkeypatch.setattr(assets, "build", lambda **_kwargs: {"workbench.html": "synthetic"})
    monkeypatch.setattr(assets, "check", failed)
    assert assets.main(["--check"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["failure_code"] == "setup.command_failed:1"
    assert payload["causes"][0]["exit_code"] == 1
    assert payload["detail"] and payload["next_action"]
    assert "NEVER-PRINT-ME" not in json.dumps(payload)
