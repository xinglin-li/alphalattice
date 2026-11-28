"""`plugin new` writes a plugin `plugin verify` passes, and verify refuses each fault by name."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from alphalattice.control.product_host.plugins.authoring import (
    CHECKS,
    Run,
    Runner,
    cancel_seconds,
    new_plugin,
    verify_plugin,
)
from alphalattice.control.product_host.plugins.registry import VerifiedPlugin
from alphalattice.control.task_control.plugin_staging import StagingRefused, checked_plugin_output
from alphalattice.kernel.execution import ExecutionStopped

TEST_SOURCE = "src/alphalattice/control/product_host/plugins/authoring.py"
GUARDED = {
    "network": (
        "task_control.plugin_network_refused",
        "socket.connect",
        "plugin_verify.network",
    ),
    "subprocess": (
        "task_control.plugin_subprocess_refused",
        "subprocess.Popen",
        "plugin_verify.subprocess",
    ),
    "import": (
        "task_control.plugin_import_outside_environment",
        "requests",
        "plugin_verify.import_outside_environment",
    ),
    "guard-write": (
        "task_control.plugin_write_outside_staging",
        "open /etc/x",
        "plugin_verify.write_outside_staging",
    ),
    "raises": ("task_control.plugin_failed", "", "plugin_verify.failed"),
}
# Each case: what it changes, the check it fails, verify's code (None: it passes) and how many
# checks pass before it; a run proves schema, network and write_outside_staging together.
CASES = [
    ("passes", None, None, len(CHECKS)),
    ("manifest", "schema", "plugin_verify.manifest_invalid", 0),
    ("no-sample", "schema", "plugin_verify.sample_unreadable", 0),
    ("linked-sample", "schema", "plugin_verify.sample_unreadable", 0),
    ("sample-shape", "schema", "plugin_verify.sample_invalid", 0),
    ("output-shape", "schema", "plugin_verify.output_invalid", 0),
    ("raises", "schema", "plugin_verify.failed", 0),
    ("import", "schema", "plugin_verify.import_outside_environment", 0),
    ("network", "network", "plugin_verify.network", 0),
    ("subprocess", "network", "plugin_verify.subprocess", 0),
    ("guard-write", "write_outside_staging", "plugin_verify.write_outside_staging", 0),
    ("own-folder", "write_outside_staging", "plugin_verify.write_outside_staging", 0),
    ("random", "deterministic", "plugin_verify.nondeterministic", 3),
    ("one-type", "deterministic", "plugin_verify.nondeterministic", 3),
    ("memory", "memory_cap", "plugin_verify.memory_cap", 0),
    ("later-memory", "memory_cap", "plugin_verify.memory_cap", 3),
    ("slow-cancel", "cancel", "plugin_verify.cancel_slow", 5),
    ("bootstrap-stop", "cancel", "plugin_verify.cancel_unproved", 5),
    ("next-session", "look_ahead", "plugin_verify.look_ahead", 6),
    ("no-sessions", "look_ahead", "plugin_verify.look_ahead_unproved", 6),
    ("own-dates", "look_ahead", "plugin_verify.look_ahead_unproved", 6),
]
ENTRY_CHANGES = {
    "output-shape": "return {'values': 3}\n",
    "random": "import random\n    params = {**params, 'close': "
    "[[c * random.random() for c in row] for row in params['close']]}\n",
    "one-type": "return {'dates': params['dates'], 'values': [[1.0] for _ in params['dates']]}\n",
    "next-session": "params = {**params, 'close': params['close'][1:] + params['close'][-1:]}\n",
    "own-dates": "return {'dates': [], 'values': []}\n",
}


def _stopped_at_launch(context: object, started: object) -> None:
    """A run stopped while its launch bootstrap waits, before any started signal."""
    raise ExecutionStopped("cancelled")


def _in_process(folder: Path, case: str) -> Runner:
    """The plugin's own code run in this process, with each guard's refusal stood in for."""
    runs: list[Mapping[str, Any]] = []

    def run(verified: VerifiedPlugin, params: Mapping[str, Any], cancel: bool) -> Run:
        if case == "bootstrap-stop" and cancel:  # timed by verify's own cancel owner
            return Run(None, 0, cancel_seconds(_stopped_at_launch, cast(Any, None)))
        if cancel:
            return Run(None, 0, 1.5 if case == "slow-cancel" else 0.01)
        runs.append(params)
        if case in GUARDED:
            code, detail, _ = GUARDED[case]
            refusals = ({"code": code, "event": "e", "detail": detail},) if detail else ()
            raise StagingRefused(code, "ZeroDivisionError: x; repair it", refusals)
        if case == "own-folder":
            (folder / "leak.txt").write_text("x", encoding="utf-8")
        module = verified.entry.partition(":")[0]
        spec = importlib.util.spec_from_file_location(module, folder / f"{module}.py")
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        text = json.dumps(loaded.run(dict(params), None), sort_keys=True, allow_nan=False)
        if case == "one-type" and len(runs) == 2:
            text = text.replace("1.0", "1")  # the same number, answered as an integer
        answer = folder.parent / f"{case}-answer.json"
        answer.write_text(text, encoding="utf-8")
        checked_plugin_output(answer, verified.spec)
        lease = verified.spec.lease.memory_bytes
        over = case == "memory" or (case == "later-memory" and len(runs) == 2)
        return Run(answer.read_bytes(), lease + over)

    return run


def _link_directory(link: Path, target: Path) -> None:
    """A directory link made without privilege: a junction on Windows, a symlink elsewhere."""
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
        return
    link.symlink_to(target, target_is_directory=True)


@pytest.mark.proves("product_host.plugin_verify_refuses_each_fault_by_name")
@pytest.mark.parametrize(
    ("case", "check", "code", "passed"), CASES, ids=[case[0] for case in CASES]
)
def test_verify_names_each_fault_with_what_to_change(
    tmp_path: Path, case: str, check: str | None, code: str | None, passed: int
) -> None:
    """A written plugin passes every check, and each fault is refused at its own check by name."""
    written = new_plugin(tmp_path, "my-factor")
    folder = tmp_path / "plugins" / "my-factor"
    entry = folder / "my_factor.py"
    if case in ENTRY_CHANGES:
        text = entry.read_text(encoding="utf-8")
        entry.write_text(text.replace("    del context\n", "    " + ENTRY_CHANGES[case]), "utf-8")
    if case == "manifest":
        (folder / "plugin.json").write_text("{}", encoding="utf-8")
    if case == "no-sample":
        (folder / "sample.json").unlink()
    if case == "linked-sample":  # a file link: Windows needs symlink rights, as CI's runners hold
        (tmp_path / "outside.json").write_bytes((folder / "sample.json").read_bytes())
        (folder / "sample.json").unlink()
        (folder / "sample.json").symlink_to(tmp_path / "outside.json")
    if case == "sample-shape":
        (folder / "sample.json").write_text('{"dates": 1}', encoding="utf-8")
    if case == "no-sessions":
        empty = {"dates": [], "symbols": ["AAA"], "close": [], "volume": []}
        (folder / "sample.json").write_text(json.dumps(empty), encoding="utf-8")

    answer = verify_plugin(tmp_path, "my-factor", run=_in_process(folder, case))

    assert written["files"] == [
        f"plugins/my-factor/{file}"
        for file in ("plugin.json", "my_factor.py", "sample.json", "test_my_factor.py")
    ]
    assert tuple(answer["passed"]) == CHECKS[:passed], answer
    if code is None:
        assert answer["status"] == "PASSED"
        assert answer["fingerprint"] in answer["next"]
        return
    failed = answer["failed"]
    assert (answer["status"], failed["check"], failed["code"]) == ("FAILED", check, code)
    assert failed["what"] and failed["change"]
    if case in GUARDED:
        assert GUARDED[case][1] in failed["what"] or "ZeroDivisionError: x" in failed["what"]


@pytest.mark.proves("product_host.plugin_verify_refuses_each_fault_by_name")
@pytest.mark.parametrize(
    ("name", "code"),
    [
        pytest.param("my-factor", "plugin_new.exists:my-factor", id="exists"),
        pytest.param("My Factor", "plugin_new.name_invalid:My Factor", id="name"),
        pytest.param("linked", "plugin_new.outside_workspace:linked", id="linked-plugins"),
    ],
)
def test_new_refuses_a_taken_invalid_or_linked_name(tmp_path: Path, name: str, code: str) -> None:
    """A plugin's folder is written once, inside the workspace, under a name the registry
    accepts; a linked plugins folder leaves nothing written outside."""
    workspace, outside = tmp_path / "workspace", tmp_path / "outside"
    outside.mkdir()
    new_plugin(workspace, "my-factor")
    if name == "linked":
        (workspace / "plugins").rename(tmp_path / "plugins-before")
        _link_directory(workspace / "plugins", outside)
    with pytest.raises(ValueError, match=f"^{code}$"):
        new_plugin(workspace, name)
    assert list(outside.iterdir()) == []
    if name == "linked":
        assert verify_plugin(workspace, "linked")["failed"]["code"] == (
            "plugin_verify.manifest_invalid"
        )


@pytest.mark.proves("product_host.plugin_new_writes_what_verify_passes_in_its_guarded_child")
def test_a_new_plugin_passes_verify_in_its_guarded_child_and_enables_by_its_fingerprint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new plugin's own test passes isolated verify in its real child, enabling by its
    fingerprint, and verify refuses its network reach under a granted lease."""
    from alphalattice.control.product_host.composition.entry import plugin
    from alphalattice.control.product_host.plugins.registry import PluginRegistry
    from alphalattice.interface.local_application import cli

    line = ["--workspace", str(tmp_path), "plugin", "new", "my-factor"]
    assert cli.main(line, serve=lambda _: 99, plugin=plugin) == 0
    assert json.loads(capsys.readouterr().out)["data"]["status"] == "WRITTEN"
    folder = tmp_path / "plugins" / "my-factor"
    spec = importlib.util.spec_from_file_location("test_my_factor", folder / "test_my_factor.py")
    assert spec is not None and spec.loader is not None
    generated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated)
    (folder / "alphalattice.py").write_text("raise SystemExit('shadowed')\n", encoding="utf-8")
    monkeypatch.chdir(folder)
    launched: list[tuple[list[str], dict[str, Any]]] = []
    real_run = subprocess.run

    def launch(command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        launched.append((command, options))
        return real_run(command, **options)

    monkeypatch.setattr(generated.subprocess, "run", launch)
    generated.test_my_factor_passes_plugin_verify()  # verify, run as the author's test runs it
    monkeypatch.undo()

    answer = verify_plugin(tmp_path, "my-factor")
    (tmp_path / "plugins.toml").write_text(
        f'[enabled]\n"my-factor" = "{answer["fingerprint"]}"\n', encoding="utf-8"
    )
    enabled = PluginRegistry(tmp_path).verify("my-factor").spec.code_role_hash
    entry, manifest = folder / "my_factor.py", folder / "plugin.json"
    reach = "import socket\n    socket.create_connection(('127.0.0.1', 9), timeout=1)\n"
    entry.write_text(
        entry.read_text(encoding="utf-8").replace("    del context\n", "    " + reach), "utf-8"
    )
    granted = json.loads(manifest.read_text(encoding="utf-8"))
    granted["lease"]["network"] = True
    manifest.write_text(json.dumps(granted), encoding="utf-8")

    [(command, options)] = launched
    assert command[1] == "-I"
    assert options["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert (answer["status"], tuple(answer["passed"])) == ("PASSED", CHECKS), answer
    assert enabled == answer["fingerprint"]
    assert verify_plugin(tmp_path, "my-factor")["failed"]["code"] == "plugin_verify.network"
