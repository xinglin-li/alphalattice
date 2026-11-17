"""An upgrade never serves a partly migrated family or forgets a durable completion."""

import json
import os
import subprocess
import sys
from functools import partial

import pytest

from alphalattice.control.product_host.composition.application_session import verify_upgrade_backups
from alphalattice.control.product_host.storage.backup import WorkspaceBackupError, WorkspaceBackups
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import create_data_root
from alphalattice.kernel.data_layer.backup import backup_data_root
from alphalattice.kernel.data_layer.chain import canonical
from alphalattice.kernel.data_layer.log import HostLog

TEST_SOURCE = "src/alphalattice/kernel/data_layer/upgrade.py"
PROGRAM = """
import json, os, sys
from pathlib import Path
from alphalattice.kernel.data_layer.upgrade import Family, UpgradeRunner
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.log import HostLog
from functools import partial
from alphalattice.control.product_host.composition.application_session import verify_upgrade_backups
from alphalattice.control.product_host.storage.backup import WorkspaceBackups
root, workspace = Path(sys.argv[1]), Path(sys.argv[2])
backup_check = partial(
    verify_upgrade_backups, machine_generation=Path(sys.argv[3]),
    workspace_backups=(WorkspaceBackups(
        workspace, workspace_id='fixture', root=Path(sys.argv[4])
    ),),
)
def family(name):
    def step(runner):
        counter = workspace / (name + '.attempts')
        counter.write_text(counter.read_text() + 'x' if counter.exists() else 'x')
        (workspace / (name + '.new')).write_text((workspace / (name + '.old')).read_text())
        manifest = workspace / 'research-workspace.json'
        manifest.write_text(json.dumps({**json.loads(manifest.read_text()), 'migrated': name}))
        runner.map_id(name, 'old-' + name, 'new-' + name)
        if name == 'two' and not (workspace / 'killed').exists():
            (workspace / 'killed').write_text('once')
            os._exit(9)
    def verify(runner):
        return (
            (workspace / (name + '.new')).read_bytes()
            == (workspace / (name + '.old')).read_bytes()
        )
    def switch(mode):
        (workspace / (name + '.mode')).write_text(mode)
    return Family(name, step, verify, switch)
try:
    with HostLog.open(root) as log:
        runner = UpgradeRunner(
            root, log, (workspace,), (family('one'), family('two')), backup_check
        )
        migrated = runner.run()
        before = log.read('SELECT count(*) FROM event')[0][0]
        assert runner.run() is False
        assert log.read('SELECT count(*) FROM event')[0][0] == before
        assert runner.id_map == {('one', 'old-one'): 'new-one', ('two', 'old-two'): 'new-two'}
        print(json.dumps({'state': 'complete', 'migrated': migrated}))
except DataError as error:
    print(json.dumps({'state': 'refused', 'code': error.failure.code, 'message': str(error)}))
    raise SystemExit(2)
"""


@pytest.mark.proves("data_layer.upgrade_verifies_before_switch_and_resumes")
def test_upgrade_finishes_only_verified_families_and_resumes_the_first_undone(tmp_path):
    """A real log and killed child prove durable resume and ID reuse, beyond unit decisions."""
    root, workspace = tmp_path / "data", tmp_path / "workspace"
    create_data_root(root)
    workspace.mkdir()
    for name in ("one", "two"):
        (workspace / (name + ".old")).write_text(name + " unchanged old bytes")
        (workspace / (name + ".mode")).write_text("old")
    (workspace / "research-workspace.json").write_text('{"workspace_id": "fixture"}')
    with HostLog.open(root):
        pass  # seed and flush before the online backup
    generation = backup_data_root(root, tmp_path / "machine-backups")
    backups = WorkspaceBackups(
        workspace, workspace_id="fixture", root=tmp_path / "workspace-backups"
    )
    check = partial(
        verify_upgrade_backups, machine_generation=generation, workspace_backups=(backups,)
    )
    with pytest.raises(DataError, match="machine backup manifest") as missing:
        verify_upgrade_backups(
            root,
            (workspace,),
            machine_generation=tmp_path / "missing-generation",
            workspace_backups=(backups,),
        )
    assert missing.value.failure.code == "data_layer.backup_manifest_invalid"
    with pytest.raises(WorkspaceBackupError, match="generation_not_found"):
        check(root, (workspace,))
    kept = backups.create(reason="REQUEST")
    assert check(root, (workspace,))
    (workspace / "research-workspace.json").write_text('{"workspace_id": "fixture", "newer": true}')
    stale = []
    try:
        check(root, (workspace,))
    except WorkspaceBackupError:
        stale.append("workspace")
    kept = backups.create(reason="REQUEST")
    with HostLog.open(root) as log:
        definition = {"family": "event_type", "name": "fixture.after_backup", "version": 1}
        log.record(
            "definition.registered",
            ("definition", "event_type/fixture.after_backup"),
            definition,
            text=canonical(
                {
                    **definition,
                    "status": "standard",
                    "spec": {
                        "work": False,
                        "subject": "fixture",
                        "required": [],
                        "text": "none",
                    },
                    "text": "A change after the historical backup.",
                }
            ).decode(),
        )
    try:
        check(root, (workspace,))
    except DataError:
        stale.append("machine")
    assert stale == ["workspace", "machine"], "historical backups admitted newer source state"
    generation = backup_data_root(root, tmp_path / "machine-backups")
    check = partial(
        verify_upgrade_backups, machine_generation=generation, workspace_backups=(backups,)
    )
    assert check(root, (workspace,))
    body = backups.objects.path(kept.entries[0].digest)
    saved = body.read_bytes()
    body.write_bytes(b"x" * len(saved))
    with pytest.raises(WorkspaceBackupError, match="object_invalid"):
        check(root, (workspace,))
    body.write_bytes(saved)
    manifest = generation / "manifest.json"
    saved_manifest = manifest.read_bytes()
    broken = json.loads(saved_manifest)
    broken["head"]["head"] = "blake3:" + "0" * 64
    manifest.write_text(json.dumps(broken))
    with pytest.raises(DataError, match="does not match its saved chain head"):
        check(root, (workspace,))
    manifest.write_bytes(saved_manifest)
    other = tmp_path / "other-root"
    create_data_root(other)
    with HostLog.open(other):
        pass
    with pytest.raises(DataError, match="is not a backup of"):
        check(other, (workspace,))
    child = [
        sys.executable,
        "-c",
        PROGRAM,
        str(root),
        str(workspace),
        str(generation),
        str(backups.root),
    ]

    def upgrade(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
            env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"},
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    first = upgrade(child)
    assert first.returncode == 9, first.stderr
    assert (workspace / "one.mode").read_text() == "new"
    assert (workspace / "two.mode").read_text() == "old"
    with HostLog.open(root) as log:
        authority = json.loads(
            log.read(
                "SELECT attrs FROM event WHERE type = 'upgrade.progress' ORDER BY seq LIMIT 1"
            )[0][0]
        )["backups"]
    assert authority["machine"]["head"] == json.loads(manifest.read_bytes())["head"]
    assert authority["workspaces"][0]["generation"] == kept.generation_hash
    newer_workspace = backups.create(reason="REQUEST")
    assert newer_workspace.generation_hash != kept.generation_hash
    newer_machine = backup_data_root(root, tmp_path / "machine-backups")
    wrong = upgrade([*child[:5], str(newer_machine), child[-1]])
    assert wrong.returncode == 2 and "pre-upgrade" in wrong.stdout
    assert (workspace / "two.attempts").read_text() == "x"
    again = upgrade(child)
    assert again.returncode == 0, again.stderr
    assert json.loads(again.stdout.strip().splitlines()[-1]) == {
        "state": "complete",
        "migrated": True,
    }
    assert (workspace / "one.attempts").read_text() == "x"
    assert (workspace / "two.attempts").read_text() == "xx"
    for name in ("one", "two"):
        assert (workspace / (name + ".mode")).read_text() == "new"
        assert (workspace / (name + ".new")).read_bytes() == (
            workspace / (name + ".old")
        ).read_bytes()
    with HostLog.open(root) as log:
        states = [
            json.loads(row[0])
            for row in log.read(
                "SELECT attrs FROM event WHERE type = 'upgrade.progress' ORDER BY seq"
            )
        ]
    assert [row["family"] for row in states if row["kind"] == "family_done"] == ["one", "two"]
    assert len([row for row in states if row["kind"] == "id_mapped"]) == 2
    for name in ("one", "two"):
        assert (workspace / (name + ".old")).read_text() == name + " unchanged old bytes"
