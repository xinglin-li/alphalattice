"""The native bridge after FLOW-1: Session bindings and Host-read observations, no hooks."""

import ast
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import tomllib
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from alphalattice.interface.local_application import native_setup as setup
from alphalattice.interface.local_application import retrieval_environment
from alphalattice.interface.local_application.client import (
    LocalResearchClient,
    LocalResearchConnection,
)
from alphalattice.interface.local_application.native_bridge import (
    BINDING_NAME,
    NativeBridgeError,
    NativeResearchBinding,
    file_observation,
    observation_request,
)
from alphalattice.interface.local_application.native_setup import (
    PRODUCT_HOOK_MATCHER,
    declare_project,
    strip_product_hooks,
)
from alphalattice.kernel.knowledge import model_store
from tests.portfolio_strategy_lab.local_web_support import _json

ROOT = Path(__file__).resolve().parents[2]
READINESS_KEYS = (
    "python",
    "uv",
    "installed_leg",
    "retrieval_runtime",
    "background_notices",
    "disk",
    "cpu",
    "network",
    "user_layer",
    "user_backup",
)


def _usage(host: str, **changes: object) -> dict[str, object]:
    return {
        "session_id": "parent",
        "agent_id": "child",
        "role": "alphalattice_cro",
        "host": host,
        "model": "synthetic-model",
        "efforts": ["medium", "high"],
        "pin_differs": ["model", "effort"],
        "last_at": "2026-10-06T12:00:00+00:00",
        "responses": 2,
        "input_tokens": 15,
        "cache_read_tokens": 3,
        "cache_write_tokens": 4,
        "output_tokens": 5,
        **changes,
    }


@pytest.mark.parametrize("host", ("codex", "claude-code"))
def test_complete_usage_snapshot_fits_public_event_subject_and_files_once(live, tmp_path, host):
    """A Host-read snapshot fits the event contract, and the same snapshot files one row: its
    sequence is its own content's, so no stored counter or lock is needed (FLOW-1)."""
    project = tmp_path / "usage-project"
    (project / ".codex").mkdir(parents=True)
    binding = NativeResearchBinding(
        session_id="parent", workspace=live.workspace, roles=("alphalattice_cro",), host=host
    )
    usage = _usage(host)
    event = {"source": "native_usage", "usage": usage}
    document = observation_request(project, binding, event)
    subject = document["subject"]
    assert len(subject) == 15
    assert "native_event_id" not in subject and "source_kind" not in subject
    assert subject["efforts"] == "medium,high"
    assert subject["pin_differs"] == "model,effort"
    assert subject["last_at"] == document["occurred_at"] == usage["last_at"]
    assert subject["sample_time_kind"] == "LATEST_USAGE_RECORD_AT"
    assert subject["input_channel"] == (
        "CODEX_SESSION_FILE" if host == "codex" else "CLAUDE_CODE_SESSION_FILE"
    )
    for count in (
        "responses",
        "input_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "output_tokens",
    ):
        assert subject[count] == str(usage[count])
    assert observation_request(project, binding, event) == document
    changed = observation_request(
        project, binding, {"source": "native_usage", "usage": _usage(host, output_tokens=6)}
    )
    assert changed["producer_sequence"] != document["producer_sequence"]
    goal = observation_request(
        project, binding, event, usage_goal_id="00000000-0000-4000-8000-000000000001"
    )
    assert goal["producer_sequence"] != document["producer_sequence"]

    client = LocalResearchClient(live.workspace)
    refused = client.publish_event(
        {
            **document,
            "subject": {**subject, "unexpected_field": "extra", "another_field": "extra"},
        }
    )
    assert refused["status"] == "REFUSED"
    assert refused["reasons"] == {"": "activity.event_subject_too_large"}
    first = file_observation(project, binding, event, publish=client.publish_event)
    assert first["status"] == "DELIVERED"
    assert file_observation(project, binding, event, publish=client.publish_event) == first
    (stored,) = _json(live, "/api/activity/external")["items"]
    assert stored["observation_id"] == first["observation_id"]
    assert stored["source_sequence"] == document["producer_sequence"]
    assert stored["payload"]["subject"] == subject


def test_a_reading_with_an_unusable_record_time_files_at_its_reading_time(tmp_path):
    """A usage record time the activity contract cannot take never makes the reading invalid."""
    binding = NativeResearchBinding("parent", tmp_path, ("alphalattice_cro",), host="codex")
    for last_at in ("t", "2026-10-06T12:00:00"):
        document = observation_request(
            tmp_path, binding, {"source": "native_usage", "usage": _usage("codex", last_at=last_at)}
        )
        assert document["occurred_at"] != last_at and document["occurred_at"].endswith("+00:00")


def test_a_filing_failure_is_named_and_never_raised(tmp_path):
    """Optional observation: a refusal or an exception of the Host's owner is a named result."""
    binding = NativeResearchBinding("parent", tmp_path, ("alphalattice_cro",), host="codex")
    event = {"source": "native_usage", "usage": _usage("codex")}

    def refusing(_document):
        return {"status": "REFUSED", "failure_code": "activity.storage_unavailable"}

    def raising(_document):
        raise RuntimeError("SECRET-HOST-ERROR")

    assert file_observation(tmp_path, binding, event, publish=refusing) == {
        "status": "UNAVAILABLE",
        "reason": "activity.storage_unavailable",
    }
    failed = file_observation(tmp_path, binding, event, publish=raising)
    assert failed["status"] == "UNAVAILABLE" and "SECRET" not in json.dumps(failed)
    with pytest.raises(NativeBridgeError, match="event_source_invalid"):
        observation_request(tmp_path, binding, {"source": "actor_declared"})


def _entry():
    spec = importlib.util.spec_from_file_location(
        "native_entry", ROOT / "src/alphalattice/interface/local_application/native_setup.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


def _bind(project, workspace):
    (project / ".codex").mkdir(parents=True)
    value = {"session_id": "parent", "workspace": str(workspace), "roles": ["alphalattice_cro"]}
    (project / ".codex" / BINDING_NAME).write_text(json.dumps(value))
    return NativeResearchBinding.read(project)


def test_a_binding_admits_only_its_own_fields(tmp_path):
    binding = _bind(tmp_path, tmp_path / "workspace")
    with pytest.raises(NativeBridgeError):
        NativeResearchBinding.from_document({**asdict(binding), "authority": "HUMAN"})
    with pytest.raises(NativeBridgeError, match="binding_invalid"):
        NativeResearchBinding.from_document(
            {"session_id": "parent", "workspace": "relative", "roles": ["alphalattice_cro"]}
        )


def test_off_binding_admits_only_its_exact_lead_without_native_discovery(tmp_path, monkeypatch):
    from alphalattice.interface.local_application import native_bridge as bridge

    binding = NativeResearchBinding("parent", tmp_path, ("alphalattice_cro",), usage="OFF")

    def no_native_access(*_args):
        pytest.fail("OFF binding discovered a native Session file.")

    monkeypatch.setattr(bridge, "codex_thread_spawn", no_native_access)
    assert binding.serves(("codex", "parent"))
    assert not binding.serves(("codex", "child"))
    assert not binding.serves(("claude-code", "parent"))


def test_configuration_and_detach_preserve_unrelated_settings(tmp_path, monkeypatch, capsys):
    entry = _entry()
    roles = entry._roles()
    assert set(roles) == {
        "alphalattice_data",
        "alphalattice_factor",
        "alphalattice_alpha",
        "alphalattice_risk",
        "alphalattice_portfolio",
        "alphalattice_evidence_analyst",
        "alphalattice_cro",
        "alphalattice_maintainer",
    }
    assert set(entry._roles("claude-code")) == set(roles)
    config = tomllib.loads((ROOT / ".codex/config.toml").read_text())
    # The specialists run on one model at one reasoning effort. Both are the cards' own settings
    # (a change is made in them alone); a card left on another model or effort than the rest
    # fails here.
    models = set()
    efforts = set()
    for role in roles:
        card = tomllib.loads((ROOT / ".codex" / config["agents"][role]["config_file"]).read_text())
        assert card["name"] == role
        assert card["model"] and card["model_reasoning_effort"]
        models.add(card["model"])
        efforts.add(card["model_reasoning_effort"])
        # Every specialist writes: a stage card runs its path and saves the answers it
        # continues from (V384), the two evidence specialists their answer file.
        assert (card["sandbox_mode"], card["approval_policy"]) == ("workspace-write", "never")
    assert len(models) == 1, f"the specialists' cards name different models: {sorted(models)}"
    assert len(efforts) == 1, f"the specialists' cards name different efforts: {sorted(efforts)}"
    assert "hooks" not in config  # the shipped declaration registers no product hook
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    (tmp_path / ".codex").mkdir()
    hooks = tmp_path / ".codex/hooks.json"
    hooks.write_text('{"hooks":{"Stop":[]}}')
    before = hooks.read_bytes()
    with pytest.raises(NativeBridgeError, match="existing_configuration_differs"):
        entry._create_or_match("hooks.json", {"hooks": {}})
    assert hooks.read_bytes() == before
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(entry, "_roles", lambda host="codex", root=None: ["alphalattice_cro"])
    declare_project(tmp_path, "codex")
    monkeypatch.setattr(
        entry.sys,
        "argv",
        ["native", "bind", "--workspace", str(workspace), "--session-id", "parent"],
    )
    assert entry.main() == 0
    assert entry.main() == 0
    monkeypatch.setattr(entry.sys, "argv", ["native", "unbind", "--session-id", "wrong"])
    assert entry.main() == 2
    assert NativeResearchBinding.read(tmp_path).session_id == "parent"
    monkeypatch.setattr(entry.sys, "argv", ["native", "unbind", "--session-id", "parent"])
    assert entry.main() == 0
    assert NativeResearchBinding.read(tmp_path) is None
    assert hooks.read_bytes() == before
    assert "native_bridge.session_mismatch" in capsys.readouterr().out


def _hook_group(event: str, matcher: str) -> dict[str, object]:
    command = {"type": "command", "command": f"run-{event} || exit 0", "timeout": 5}
    return {"matcher": matcher, "hooks": [command]}


def test_configure_removes_only_the_products_retired_hook_groups(tmp_path, monkeypatch, capsys):
    """FLOW-1: an earlier configure's product hooks called a command that no longer exists and
    kept the host's trust prompts; configuring again removes exactly those groups, keeping every
    other hook and setting as written."""
    claude = tmp_path / "claude-project"
    (claude / ".claude").mkdir(parents=True)
    settings = {
        "permissions": {"allow": ["Read"]},
        "hooks": {
            "SubagentStart": [
                _hook_group("SubagentStart", PRODUCT_HOOK_MATCHER),
                _hook_group("SubagentStart", "^mine$"),
            ],
            "SubagentStop": [_hook_group("SubagentStop", PRODUCT_HOOK_MATCHER)],
            "Stop": [_hook_group("Stop", PRODUCT_HOOK_MATCHER)],
        },
    }
    (claude / ".claude/settings.json").write_text(json.dumps(settings), encoding="utf-8")
    assert strip_product_hooks(claude, "claude-code") == 2
    kept = json.loads((claude / ".claude/settings.json").read_text(encoding="utf-8"))
    assert kept == {
        "permissions": {"allow": ["Read"]},
        "hooks": {
            "SubagentStart": [_hook_group("SubagentStart", "^mine$")],
            "Stop": [_hook_group("Stop", PRODUCT_HOOK_MATCHER)],
        },
    }
    assert strip_product_hooks(claude, "claude-code") == 0

    codex = tmp_path / "codex-project"
    (codex / ".codex").mkdir(parents=True)
    product = (
        '[[hooks.{event}]]\nmatcher = "^alphalattice_.*$"\n[[hooks.{event}.hooks]]\n'
        'type = "command"\ncommand = "python -m native_setup hook || exit 0"\ntimeout = 5\n'
    )
    text = (
        '[agents.alphalattice_cro]\nconfig_file = "agents/alphalattice_cro.toml"\n\n'
        + product.format(event="SubagentStart")
        + '\n[[hooks.SubagentStart]]\nmatcher = "^mine$"\n[[hooks.SubagentStart.hooks]]\n'
        'type = "command"\ncommand = "mine"\n\n'
        + product.format(event="SubagentStop")
        + "\n[features]\nflag = true\n"
    )
    (codex / ".codex/config.toml").write_text(text, encoding="utf-8")
    entry = _entry()
    python = codex / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    (codex / ".codex/agents").mkdir()
    shutil.copyfile(
        ROOT / ".codex/agents/alphalattice_cro.toml", codex / ".codex/agents/alphalattice_cro.toml"
    )
    monkeypatch.setattr(entry.sys, "argv", ["native", "--project", str(codex), "configure"])
    assert entry.main() == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["retired_hook_groups_removed"] == 2 and answer["trust_changed"] is False
    result = tomllib.loads((codex / ".codex/config.toml").read_text(encoding="utf-8"))
    assert result["agents"] == {"alphalattice_cro": {"config_file": "agents/alphalattice_cro.toml"}}
    assert result["features"] == {"flag": True}
    assert result["hooks"] == {
        "SubagentStart": [{"matcher": "^mine$", "hooks": [{"type": "command", "command": "mine"}]}]
    }


@pytest.mark.parametrize("git_checkout", (False, True))
def test_ordinary_project_setup_validates_local_declarations_without_trust(
    tmp_path, monkeypatch, capsys, git_checkout
):
    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    config = project / ".codex/config.toml"
    config.write_bytes((ROOT / ".codex/config.toml").read_bytes())
    python = project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    if git_checkout:
        (project / ".git").mkdir()
        (project / ".git/HEAD").write_text("ref: refs/heads/main\n", newline="\n")
    entry = _entry()
    for command, status, code in (
        ("configure", "LOCAL_DECLARATIONS_VALIDATED", 0),
        ("doctor", "REFUSED", 2),
    ):
        monkeypatch.setattr(entry.sys, "argv", ["native", "--project", str(project), command])
        before = {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}
        assert entry.main() == code
        result = json.loads(capsys.readouterr().out)
        assert result["status"] == status
        if command == "configure":
            # The installing session continues: it reads the guide and Skill by path, and a
            # session the host enforces the cards in is an option it discloses (STOPS-1).
            here = result["continue_here"]
            assert here["read"] == [
                str(project.resolve() / "AGENTS.md"),
                str(project.resolve() / ".agents/skills/alphalattice-research/SKILL.md"),
            ]
            assert here["specialists"] == str(project.resolve() / ".codex/agents")
            assert here["open_session"] == f'codex -C "{project.resolve()}"'
            assert here["open_session"] in here["disclosure"]
        if command == "doctor":
            assert result["failure_code"] == "native_bridge.not_bound"
            assert {"alphalattice_evidence_analyst", "alphalattice_cro"} <= set(result["roles"])
            assert result["session_bound"] is False
            assert result["attachment_preflight"]["missing"] == ["native_session_binding"]
            assert result["research_nonblocking"] is True
            readiness = {item["key"]: item for item in result["host_readiness"]}
            assert set(READINESS_KEYS) == readiness.keys()
            assert readiness["disk"]["present"] is None and readiness["network"]["present"] is None
            assert "native_proof" not in result and "host_trust" not in result
            assert {
                path: path.read_bytes() for path in project.rglob("*") if path.is_file()
            } == before
    assert not (project / ".codex" / BINDING_NAME).exists()


def test_claude_host_configures_binds_and_inspects_without_the_codex_files(
    tmp_path, monkeypatch, capsys
):
    # A Claude-only tree ships the bridge, the Claude cards and settings, and no Codex file.
    project = tmp_path / "project"
    (project / ".claude/agents").mkdir(parents=True)
    for card in (ROOT / ".claude/agents").glob("*.md"):
        (project / ".claude/agents" / card.name).write_bytes(card.read_bytes())
    settings = (ROOT / ".claude/settings.json").read_bytes()
    (project / ".claude/settings.json").write_bytes(settings)
    python = project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    (project / ".codex").mkdir()
    workspace = project / "workspace"
    workspace.mkdir()
    entry = _entry()
    monkeypatch.setattr(entry, "ROOT", project)

    def run(*argv):
        monkeypatch.setattr(entry.sys, "argv", ["native", *argv])
        code = entry.main()
        return code, json.loads(capsys.readouterr().out)

    cards = sorted(path.stem for path in (ROOT / ".claude/agents").glob("alphalattice_*.md"))
    assert set(cards) == {
        "alphalattice_data",
        "alphalattice_factor",
        "alphalattice_alpha",
        "alphalattice_risk",
        "alphalattice_portfolio",
        "alphalattice_evidence_analyst",
        "alphalattice_cro",
        "alphalattice_maintainer",
    }
    assert run("configure", "--host", "claude-code")[1]["status"] == "LOCAL_DECLARATIONS_VALIDATED"
    assert (project / ".claude/settings.json").read_bytes() == settings
    code, bound = run(
        "bind", "--host", "claude-code", "--session-id", "parent", "--workspace", str(workspace)
    )
    assert (code, bound["status"]) == (0, "BOUND")
    assert NativeResearchBinding.read(project).roles == tuple(cards)
    binding_before_doctor = (project / ".codex" / BINDING_NAME).read_bytes()
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "parent")
    code, report = run("doctor")
    assert (code, report["status"], report["host"], report["roles"]) == (
        0,
        "READY",
        "claude-code",
        cards,
    )
    assert report["session_bound"] is True and report["session_id"] == "parent"
    assert report["usage_reading"] == "READ"
    notices = next(item for item in report["host_readiness"] if item["key"] == "background_notices")
    assert notices["present"] is True and notices["who_decides"] == "AGENT"
    assert (project / ".codex" / BINDING_NAME).read_bytes() == binding_before_doctor
    assert not (project / ".codex/config.toml").exists()
    # The Codex host still reads its own file, and refuses without it.
    assert run("configure")[1]["status"] == "REFUSED"


@pytest.mark.parametrize("missing", (None, *READINESS_KEYS))
def test_host_readiness_checks_dependencies_against_bound_host(tmp_path, monkeypatch, missing):
    """Each missing dependency has a way to fill it and Host facts override the caller's PATH."""
    python = tmp_path / "bin/python"
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n')
    monkeypatch.setattr(setup, "RESOURCE_ROOT", tmp_path)
    monkeypatch.setattr(setup.sys, "executable", str(python))
    monkeypatch.setattr(setup.sys, "version_info", (3, 11) if missing == "python" else (3, 12))
    monkeypatch.setattr(os, "cpu_count", lambda: 0 if missing == "cpu" else 4)
    monkeypatch.delenv("UV", raising=False)

    def which(name):
        return None if name == missing else str(python.parent / name)

    monkeypatch.setattr(shutil, "which", which)
    distribution = SimpleNamespace(
        version="0" if missing == "installed_leg" else "1.2.3",
        read_text=lambda _: json.dumps({"url": tmp_path.as_uri()}),
    )
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: distribution)
    monkeypatch.setattr(retrieval_environment, "interpreter_path", lambda: python)
    monkeypatch.setattr(model_store, "default_store_root", lambda: tmp_path / "packs")
    monkeypatch.setattr(model_store, "recipe_readiness", lambda *_: {"ready": True})

    def run(argv, **_):
        exit_code = int(missing == "retrieval_runtime" and "-c" in argv)
        return subprocess.CompletedProcess(argv, exit_code)

    monkeypatch.setattr(subprocess, "run", run)
    user = tmp_path / ".alphalattice/user"
    if missing != "user_layer":
        user.mkdir(parents=True)
        if missing != "user_backup":
            (user / "backups/2026-10-09").mkdir(parents=True)
    free = 0 if missing == "disk" else 2 * 1024**3
    capacity = {"measured_data_bytes": 1, "cap_bytes": 2, "free_disk_bytes": free}
    replies = {
        "STORAGE_CAP_SHOW": {"capacity": capacity},
        "CPU_BUDGET_SHOW": {"status": "CPU_BUDGET", "cores": 4},
        "NETWORK_ACCESS": None if missing == "network" else {"network_allowed": False},
    }
    client = LocalResearchClient
    monkeypatch.setattr(client, "request", lambda _self, request: replies[request["operation"]])
    queue = {"present": missing != "background_notices", "reason": None}
    monkeypatch.setattr(client, "activity", lambda _self, **_: {"observer": {"codex_queue": queue}})
    workspace = tmp_path / "workspace"
    connection = LocalResearchConnection(
        str(workspace), "qa-readiness", "http://127.0.0.1:1", "i" * 16, "t" * 32
    )
    descriptor = LocalResearchConnection.path(workspace)
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text(connection.to_json(), encoding="utf-8")
    binding = NativeResearchBinding("parent", workspace, ("alphalattice_cro",))
    items = {item["key"]: item for item in setup.host_readiness(tmp_path, "codex", binding)}
    if missing is None:
        assert all(item["present"] is True for item in items.values())
    else:
        assert items[missing]["present"] is (None if missing == "network" else False)
    assert all(item["what_it_enables"] and item["how_to_fill"] for item in items.values())
    assert items["network"]["facts"]["scope"] == "WORKSPACE_HOST"


@pytest.mark.parametrize(
    "outcome, reason",
    (
        ("ready", None),
        ("missing", "COMMAND_MISSING"),
        ("failed", "COMMAND_FAILED"),
        ("timeout", "COMMAND_TIMED_OUT"),
        ("start_failed", "COMMAND_START_FAILED"),
    ),
)
def test_codex_queue_check_names_its_bounded_probe_result(tmp_path, monkeypatch, outcome, reason):
    """Queue readiness distinguishes absence, exit failure, timeout and failure to start."""
    executable = str(tmp_path / "codex")
    monkeypatch.setattr(shutil, "which", lambda _: None if outcome == "missing" else executable)
    calls = []

    def run(argv, **options):
        calls.append((argv, options))
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(argv, options["timeout"])
        if outcome == "start_failed":
            raise OSError("private launch detail")
        return subprocess.CompletedProcess(argv, 1 if outcome == "failed" else 0)

    monkeypatch.setattr(subprocess, "run", run)
    result = setup.codex_queue_readiness()
    assert (result["present"], result["reason"]) == (outcome == "ready", reason)
    assert len(calls) == (0 if outcome == "missing" else 1)
    if calls:
        argv, options = calls[0]
        assert argv == [executable, "queue", "--help"] and 0 < options["timeout"] <= 30


def test_claude_code_host_binding_is_kept_apart_from_codex(tmp_path):
    """A binding names its host; producer id and channel name the host as source information."""
    workspace = tmp_path / "workspace"
    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    document = {
        "session_id": "parent",
        "workspace": str(workspace),
        "roles": ["alphalattice_cro"],
        "host": "claude-code",
    }
    (project / ".codex" / BINDING_NAME).write_text(json.dumps(document))
    binding = NativeResearchBinding.read(project)
    assert binding.host == "claude-code"
    with pytest.raises(NativeBridgeError, match="binding_invalid"):
        NativeResearchBinding.from_document({**document, "host": "other-host"})
    assert (
        NativeResearchBinding.from_document({k: v for k, v in document.items() if k != "host"}).host
        == "codex"
    )
    filed = observation_request(
        project, binding, {"source": "native_usage", "usage": _usage("claude-code")}
    )
    assert filed["producer_id"] == "claude-code-native"
    assert filed["subject"]["native_host"] == "claude-code"
    assert filed["subject"]["input_channel"] == "CLAUDE_CODE_SESSION_FILE"
    with pytest.raises(NativeBridgeError, match="event_scope_invalid"):
        observation_request(project, binding, {"source": "native_usage", "usage": _usage("codex")})


def test_every_bound_the_bridge_applies_is_the_contract_of_what_it_carries(tmp_path):
    """Every bound the bridge applies is the contract of what it carries."""

    from pydantic import ValidationError

    from alphalattice.interface.local_application import native_bridge as bridge
    from alphalattice.interface.local_application.activity import (
        SUMMARY_RETAINED_CHARACTERS,
        ExternalActivityEventDocument,
    )
    from alphalattice.protocols.actor_execution.bundles import AgentBundleRecord, bundle_slot

    def admitted(subject=None, correlation_ids=()):
        try:
            ExternalActivityEventDocument.model_validate(
                {
                    "event_kind": "NATIVE_AGENT_USAGE",
                    "producer_id": "codex-native",
                    "producer_session": "a" * 64,
                    "producer_sequence": 0,
                    "occurred_at": "2026-10-03T00:00:00+00:00",
                    "summary": "A reading.",
                    "subject": subject or {},
                    "correlation_ids": list(correlation_ids),
                }
            )
        except ValidationError:
            return False
        return True

    def edge(admits):
        """The longest length a contract admits; one more it refuses."""
        longest = max(n for n in range(1, 513) if admits(n))
        assert not admits(longest + 1)
        return longest

    subject_value = edge(lambda n: admitted({"reference": "r" * n}))
    correlation = edge(lambda n: admitted(correlation_ids=["s" * n]))
    subject_keys = edge(lambda n: admitted({f"k{i}": "v" for i in range(n)}))

    binding = NativeResearchBinding(
        session_id="parent", workspace=tmp_path, roles=("alphalattice_cro",), host="claude-code"
    )
    accepted = {
        "source": "product_accepted",
        "session_id": "parent",
        "agent_id": "parent",
        "role": "research_lead",
        "bundle_role": "CRO",
        "message_id": "accepted-" + "a" * 24,
        "message": "Product accepted deliverable (ACCEPTED): text: Read.",
        "reference": "r" * 36,
        "bundle_reference": "b" * 64,
        "answer_reference": "c" * 64,
        "submitted_by": "parent",
        "recipient_id": "parent",
        "occurred_at": "2026-10-03T00:00:00+00:00",
        "source_time_kind": "PRODUCT_ACCEPTED_AT",
    }
    widest = max(
        len(bridge.observation_request(tmp_path, binding, event)["subject"])
        for event in (accepted, {"source": "native_usage", "usage": _usage("claude-code")})
    )
    # The widest binding `session bind` writes, as it writes it (`_create_or_match`).
    document = {
        "session_id": "s" * bridge.CORRELATION_CHARACTERS,
        "workspace": "D:\\" + "\u00e9" * (bridge.TEXT_CHARACTERS - 3),
        "roles": [f"r{index:02d}" + "x" * 61 for index in range(bridge.BINDING_ROLES)],
        "host": "claude-code",
        "usage": "OFF",
    }
    NativeResearchBinding.from_document(document)
    written = len((json.dumps(document, indent=2) + "\n").encode())
    directory = AgentBundleRecord.model_fields["bundle_directory"].metadata
    longest_directory = max(getattr(item, "max_length", 0) or 0 for item in directory)
    entry = _entry()
    shipped = {host: entry._roles(host) for host in ("codex", "claude-code")}
    # name: (the bridge's bound, how it stands to its contract, the contract's, what it bounds)
    table = {
        "SUBJECT_VALUE_CHARACTERS": (
            bridge.SUBJECT_VALUE_CHARACTERS,
            "==",
            subject_value,
            "each id, role and reference an event carries",
        ),
        "CORRELATION_CHARACTERS": (
            bridge.CORRELATION_CHARACTERS,
            "==",
            correlation,
            "the bound session's id, every event's correlation id, at binding",
        ),
        "MESSAGE_KEPT_CHARACTERS": (
            bridge.MESSAGE_KEPT_CHARACTERS,
            "==",
            SUMMARY_RETAINED_CHARACTERS,
            "what the Host keeps of an accepted deliverable's preview",
        ),
        "PIN_CHARACTERS": (
            bridge.PIN_CHARACTERS,
            "<=",
            subject_value,
            "a card's model or effort pin, compared with a reading",
        ),
        "BINDING_BYTES": (
            bridge.BINDING_BYTES,
            ">=",
            written,
            "the binding file: the widest binding `bind` writes",
        ),
        "BINDING_ROLES": (
            bridge.BINDING_ROLES,
            ">=",
            max(len(roles) for roles in shipped.values()),
            "a binding's roles: the cards a host ships",
        ),
        "BINDING_RECORDS": (
            bridge.BINDING_RECORDS,
            "==",
            128,
            "the bounded exact host/Session collection; overflow refuses without replacing records",
        ),
        "TEXT_CHARACTERS": (
            bridge.TEXT_CHARACTERS,
            ">=",
            260,
            "a binding's host, reading and workspace, never carried: a Windows path",
        ),
        "SPAWN_HOPS": (
            bridge.SPAWN_HOPS,
            ">=",
            2,
            "the rollouts a Codex specialist's spawn chain reads, a first record each: its "
            "lead's own and a specialist's specialist's (V568)",
        ),
        "USAGE_READ_BYTES": (
            bridge.USAGE_READ_BYTES,
            "==",
            512 * 1024 * 1024,
            "a complete optional native session scan; overflow publishes no prefix",
        ),
        "USAGE_LINE_BYTES": (
            bridge.USAGE_LINE_BYTES,
            "==",
            8 * 1024 * 1024,
            "one complete native JSONL record; an overbound record publishes no prefix",
        ),
        "USAGE_MODEL_ROWS": (
            bridge.USAGE_MODEL_ROWS,
            "==",
            32,
            "one complete per-participant model projection; overflow publishes no rows",
        ),
    }
    relations = {"==": int.__eq__, "<=": int.__le__, ">=": int.__ge__}
    for name, (bound, relation, contract, _what) in table.items():
        assert relations[relation](bound, contract), (name, bound, relation, contract)
    assert widest <= subject_keys, widest
    # A bundle is named by its key, which any event carries; its directory may not fit one.
    assert len(bundle_slot("D:\\" + "d" * 1020)) <= subject_value < longest_directory
    for roles in shipped.values():
        assert roles and all(bridge._ROLE_NAME.match(role) for role in roles), roles

    # Every length bound is named, and every named one is a row.
    tree = ast.parse(Path(bridge.__file__).read_text(encoding="utf-8"))
    named = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and type(getattr(bridge, target.id)) is int
    }
    assert named <= set(table), named - set(table)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        if any(
            isinstance(item, ast.Call) and getattr(item.func, "id", "") == "len"
            for item in operands
        ):
            literals = [
                item.value
                for item in operands
                if isinstance(item, ast.Constant) and item.value not in (0, 1)
            ]
            assert not literals, (node.lineno, literals)

    # A binding the Host could never correlate, or naming no card, is refused when bound.
    for refused in (
        {**document, "session_id": "s" * (correlation + 1)},
        {**document, "roles": ["Alphalattice-CRO"]},
    ):
        with pytest.raises(NativeBridgeError, match="binding_invalid"):
            NativeResearchBinding.from_document(refused)
    # A carried value past the subject bound is refused by the bridge, before the Host.
    with pytest.raises(NativeBridgeError, match="activity_subject_invalid"):
        bridge.observation_request(
            tmp_path,
            binding,
            {
                "source": "native_usage",
                "usage": _usage("claude-code", model="m" * (subject_value + 1)),
            },
        )


def test_a_binding_is_found_up_from_any_folder_and_serves_its_session_and_specialists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A binding is found up from any folder and serves its session and specialists."""

    from alphalattice.interface.local_application import native_bridge as bridge
    from alphalattice.interface.local_application.native_setup import (
        bind_session,
        session_project,
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    claude = tmp_path / "claude-project"
    (claude / ".claude" / "agents").mkdir(parents=True)
    shutil.copyfile(ROOT / ".claude/settings.json", claude / ".claude/settings.json")
    shutil.copyfile(ROOT / ".claude/agents/alphalattice_cro.md", claude / ".claude/agents/c.md")
    (claude / ".claude/agents/c.md").rename(claude / ".claude/agents/alphalattice_cro.md")
    declare_project(claude, "claude-code")
    below = claude / "deep" / "er"
    below.mkdir(parents=True)
    lead = str(uuid4())
    assert session_project(below, "claude-code") == claude
    with pytest.raises(NativeBridgeError, match="project_declaration_missing"):
        session_project(below, "codex")
    written = bind_session(claude, host="claude-code", session_id=lead, workspace=workspace)
    assert written["workspace"] == str(workspace.resolve()) and (claude / ".codex").is_dir()
    assert bind_session(claude, host="claude-code", session_id=lead, workspace=workspace)
    second = str(uuid4())
    before = (claude / ".codex" / BINDING_NAME).read_bytes()
    assert bind_session(claude, host="claude-code", session_id=second, workspace=workspace)
    assert (claude / ".codex" / BINDING_NAME).read_bytes() == before
    with pytest.raises(NativeBridgeError, match="binding_ambiguous"):
        bridge.NativeResearchBinding.find(below)
    found = bridge.NativeResearchBinding.find(below, session=("claude-code", lead))
    assert found is not None and found[0] == claude
    binding = found[1]
    assert binding.serves(("claude-code", lead)) and not binding.serves(("codex", lead))
    assert not binding.serves(("claude-code", str(uuid4())))
    assert bridge.NativeResearchBinding.find(tmp_path) is None

    # A Codex lead's specialists, by their own rollouts' spawn chains.
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    day = tmp_path / "codex-home" / "sessions" / "2026" / "10" / "03"
    day.mkdir(parents=True)

    def spawned(parent: str) -> str:
        thread = str(uuid4())
        meta = {
            "id": thread,
            "source": {"subagent": {"thread_spawn": {"parent_thread_id": parent}}},
        }
        record = json.dumps({"type": "session_meta", "payload": meta})
        (day / f"rollout-2026-10-03T00-00-00-{thread}.jsonl").write_text(record + "\n", "utf-8")
        return thread

    codex = bridge.NativeResearchBinding(
        session_id=lead, workspace=workspace, roles=("alphalattice_cro",), host="codex"
    )
    chain = [lead]
    for _hop in range(bridge.SPAWN_HOPS + 1):
        chain.append(spawned(chain[-1]))
    served = [codex.serves(("codex", thread)) for thread in chain]
    assert served == [True] * (bridge.SPAWN_HOPS + 1) + [False]
    assert not codex.serves(("codex", spawned(str(uuid4()))))
    assert not codex.serves(("claude-code", chain[1]))


def _declared_binding_project(tmp_path):
    """A labelled fixture project; bindings confer no live native credit or trust."""
    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    (project / ".codex/config.toml").write_bytes((ROOT / ".codex/config.toml").read_bytes())
    (project / ".claude/agents").mkdir(parents=True)
    (project / ".claude/settings.json").write_bytes((ROOT / ".claude/settings.json").read_bytes())
    card = ".claude/agents/alphalattice_cro.md"
    (project / card).write_bytes((ROOT / card).read_bytes())
    for host in ("codex", "claude-code"):
        declare_project(project, host)
    workspace = project / "workspace"
    workspace.mkdir()
    return project, workspace


@pytest.mark.parametrize("problem", ("malformed_json", "wrong_key", "unknown_file", "oversized"))
def test_binding_collection_keeps_exact_readable_slots_and_reports_each_bad_record(
    tmp_path, problem
):
    """TE12: one malformed metadata item cannot erase another Session or select its scope."""
    from alphalattice.interface.local_application.native_bridge import (
        BINDING_BYTES,
        BINDING_DIRECTORY,
    )
    from alphalattice.interface.local_application.native_setup import bind_session

    project, workspace = _declared_binding_project(tmp_path)
    for session in ("fixture-first", "fixture-second"):
        assert bind_session(project, host="codex", session_id=session, workspace=workspace)
    selected = NativeResearchBinding.read(project, session=("codex", "fixture-second"))
    before = selected.record_path(project).read_bytes()
    directory = project / ".codex" / BINDING_DIRECTORY
    bad = directory / ("0" * 64 + ".json")
    if problem == "malformed_json":
        bad.write_bytes(b"{broken")
    elif problem == "oversized":
        bad.write_bytes(b"x" * (BINDING_BYTES + 1))
    elif problem == "unknown_file":
        bad = directory / "unknown.json"
        bad.write_bytes(before)
    else:
        bad.write_bytes(before)
    bindings, refusals = NativeResearchBinding.binding_entries(project)
    assert {(each.host, each.session_id) for each in bindings} == {
        ("codex", "fixture-first"),
        ("codex", "fixture-second"),
    }
    assert len(refusals) == 1
    assert refusals[0]["binding_key"] == bad.name
    assert refusals[0]["failure_code"] == (
        "native_bridge.binding_path_invalid"
        if problem == "unknown_file"
        else "native_bridge.binding_invalid"
    )
    assert NativeResearchBinding.read(project, session=("codex", "fixture-second")) == selected
    assert selected.record_path(project).read_bytes() == before
    with pytest.raises(NativeBridgeError, match=r"binding_(path_)?invalid"):
        NativeResearchBinding.read(project)


@pytest.mark.parametrize("inner_host", ("codex", "claude-code"))
def test_binding_discovery_stops_at_the_nearest_configured_project_even_when_unbound(
    tmp_path, inner_host
):
    """An outer exact Session record grants no implicit workspace to an inner project."""
    from alphalattice.interface.local_application.native_setup import bind_session

    outer, workspace = _declared_binding_project(tmp_path)
    assert bind_session(outer, host="codex", session_id="fixture-parent", workspace=workspace)
    before = (outer / ".codex" / BINDING_NAME).read_bytes()
    inner = outer / "inner"
    inner.mkdir()
    declare_project(inner, inner_host)
    below = inner / "notes"
    below.mkdir()
    if inner_host == "codex":
        assert NativeResearchBinding.find(below, session=("codex", "fixture-parent")) is None
    else:
        with pytest.raises(NativeBridgeError, match="project_mismatch"):
            NativeResearchBinding.find(below, session=("codex", "fixture-parent"))
    assert NativeResearchBinding.find(below) is None
    assert (outer / ".codex" / BINDING_NAME).read_bytes() == before
    assert NativeResearchBinding.find(outer, session=("codex", "fixture-parent"))[0] == outer


def test_an_unrelated_read_binding_cannot_discover_an_off_parents_child(tmp_path, monkeypatch):
    """OFF is a project privacy boundary for unknown children, independent of other slots."""
    from alphalattice.interface.local_application.native_setup import bind_session

    project, workspace = _declared_binding_project(tmp_path)
    assert bind_session(
        project, host="codex", session_id="fixture-off-parent", workspace=workspace, usage="off"
    )
    assert bind_session(
        project, host="codex", session_id="fixture-unrelated-read", workspace=workspace
    )
    home = tmp_path / "synthetic-codex-home"
    home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(home))
    scan = os.scandir
    opening = Path.open

    def no_discovery(path):
        assert not Path(path).is_relative_to(home), "OFF must not discover native files"
        return scan(path)

    def no_native_open(path, mode="r", *args, **kwargs):
        assert not path.is_relative_to(home), "OFF must not open a child or parent native file"
        return opening(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", no_discovery)
    monkeypatch.setattr(Path, "open", no_native_open)
    assert NativeResearchBinding.find(project, session=("codex", "fixture-unbound-child")) is None
    exact = NativeResearchBinding.find(project, session=("codex", "fixture-unrelated-read"))
    assert exact[1].session_id == "fixture-unrelated-read"
    assert (
        NativeResearchBinding.read(project, session=("codex", "fixture-off-parent")).usage == "OFF"
    )
