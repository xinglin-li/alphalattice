# Upgrading AlphaLattice
Last checked: 2026-11-23 against 0.2.9

For 0.2.10 and later, start with a fresh data root and research workspace.
These releases open only data created in the new format. Earlier data roots,
workspaces, and backups can still be opened with their original release.

Use this guide to upgrade from 0.2.0 or later to 0.2.9. A workspace from before 0.2.8
has its data moved once, at its first start on 0.2.8 or later
([the 0.2.8 data move](#the-028-data-move)). A workspace last opened by a release before
0.2.7 still holds tables 0.2.7 retired and is refused with
`feature_storage.workspace_before_0_2_7`: open it with 0.2.7 first, then upgrade. From
0.2.8 to 0.2.9 no data moves: 0.2.9 uses a 0.2.8 data root as it is. Keep using your
existing agent project and research workspace. [Getting started](getting-started.md)
and the [operating reference](../../.agents/skills/alphalattice-research/references/operating.md)
cover installation requirements and setup repairs.

## Before you upgrade

1. Back up your agent project's `.alphalattice/user/` folder to a safe location
   outside the installation. It holds your lessons, corrections and local guide
   additions; releases do not overwrite it.
2. Write down your research workspace's absolute path. Keep that workspace and
   its data; a relative path can select a different folder after you change
   installation directories.
3. Stop the Host serving that workspace with `alphalattice host stop`. Name the
   saved workspace path explicitly if your session is not bound to it:

   ```powershell
   alphalattice --workspace "<existing-workspace-path>" host stop
   ```

   The Host stops Tasks at safe points and retains completed work for resumption.
   Task status remains readable while it stops. Wait until it has released the
   workspace before replacing the installation.

## Source checkout

From your existing checkout's root, get the new source:

```powershell
git pull
```

To select a particular release instead, use `git checkout <release-tag>`, replacing
the placeholder with that release's published tag. Then install its locked
dependencies and run the doctor:

```powershell
uv sync --locked --all-extras
uv run python -m alphalattice.interface.local_application.native_setup doctor
```

Follow the doctor's missing setup items and the repair commands it returns.
This can include rebuilding the Workbench and refreshing the installed command's
locked runtime. [Getting started](getting-started.md#first-run-from-a-source-checkout)
explains those repairs and their permissions.

If the doctor calls for agent-project configuration, run it from the same root:

```powershell
uv run python -m alphalattice.interface.local_application.native_setup configure --host claude-code
```

Use `--host codex` for Codex. In a shell outside an agent session,
give the doctor the same host choice; otherwise its fallback is Codex. Read the
files named in `continue_here` to continue in the same session. Configuration
refuses to overwrite different files; follow its answer if it refuses.
Run the doctor again after repairs or configuration.

## Release wheel

Download the new release's wheel to a separate installation directory. Extract
the locked requirements and resolver overrides from that same wheel, as in the
[operating reference](../../.agents/skills/alphalattice-research/references/operating.md#setup-and-launch).
Replace the example filename in both commands with the new wheel you downloaded.
It must contain both runtime files:

```powershell
python -c "from zipfile import ZipFile; from pathlib import Path; wheel = ZipFile('alphalattice-0.2.9-py3-none-any.whl'); Path('alphalattice-runtime-requirements.txt').write_bytes(wheel.read('alphalattice/_runtime/config/release/runtime-requirements.txt')); Path('alphalattice-runtime-overrides.txt').write_bytes(wheel.read('alphalattice/_runtime/config/release/runtime-overrides.txt'))"
uv tool install --force --compile-bytecode --python 3.12 ./alphalattice-0.2.9-py3-none-any.whl --with-requirements alphalattice-runtime-requirements.txt --constraints alphalattice-runtime-requirements.txt --overrides alphalattice-runtime-overrides.txt
```

`--force` replaces the previous tool installation. To switch between the wheel
and source-checkout installation methods, first run `uv tool uninstall alphalattice`,
as the operating reference describes.

Run the new tool's doctor against your existing agent project. Replace
`<agent-project>` with that folder's path:

```powershell
$toolDir = uv tool dir
& "$toolDir/alphalattice/Scripts/python.exe" -m alphalattice.interface.local_application.native_setup --project "<agent-project>" doctor --host claude-code
```

If the doctor calls for configuration, use the same interpreter and project:

```powershell
& "$toolDir/alphalattice/Scripts/python.exe" -m alphalattice.interface.local_application.native_setup --project "<agent-project>" configure --host claude-code
```

Use `--host codex` for Codex. On macOS and Linux the interpreter is
`bin/python` rather than `Scripts/python.exe`; see [Getting started](getting-started.md#release-wheel)
for platform requirements. Follow any setup repairs, read `continue_here` after
configuration and run the doctor again.

## Continue your daily update

Start the Host on the workspace path you saved:

```powershell
alphalattice --workspace "<existing-workspace-path>" serve --no-browser --detach
```

If startup asks for a person-owned terminal, run its offered command once there
and tell the agent to continue. This is separate from the standing network
permission in your existing workspace.

Open its printed Workbench link in your coding app's built-in browser. In the
same agent project, ask for your next daily update as usual, for example
`Positions for <next session>, reviewed.`

The first daily update after an upgrade may need a one-time rebuild of derived
work. Allow it to finish; later updates can reuse that work. Keep your earlier
sealed goals, reports and research history in the existing workspace.

## The 0.2.8 data move

The 0.2.8 Host startup has an ordered upgrade runner. Before moving a family of
stored data, it requires verified backups of the machine data root and every
workspace being upgraded. Each family compares its old and new reads before
switching to the new store. Its old data is retained.

Backup verification reads the saved machine log's chain, the catalog's complete
file set and its sealed file bytes. It also checks each workspace generation and
its retained objects. A manifest's presence alone does not admit an upgrade.
Before the first family starts, the machine backup must match the current log
head and each workspace generation must match its current held state. A valid
older backup is refused; create fresh backups before that first start.

The start record retains those exact backup generations. Resume verifies their
seals and reuses that pre-upgrade authority even after a family has changed the
stores. It does not substitute a newer backup or require the partially migrated
stores to still equal their pre-upgrade copies. Keep the recorded generations
available until the upgrade finishes.

An interrupted start resumes the first unfinished family. A failed comparison
names that family and leaves its reader on the old store; the Host does not
serve a partial upgrade. Repair the named problem before starting again. A
completed family does not migrate again on the next start.

The runner moves these families, in this order: market data, execution
outcomes, Strategy Lab lanes, Alpha arrays, protected validation, risk covariance,
sector results, the portfolio benchmark, portfolio results, panel closures,
research inputs and the machine backup's copies of research inputs.

## If a command refuses

Read its explanation and follow its offered next step. If the problem remains,
generate a report, choosing a new output folder in your workspace:

```powershell
alphalattice problem report --output "<out>/report.json"
```

Review the whole report before sharing it. Then [open a GitHub issue](https://github.com/xinglin-li/alphalattice/issues)
with the command, what you expected, the refusal and the reviewed report. An
agent asks your permission before sending it. The [reporting procedure](../../.agents/skills/alphalattice-research/references/operating.md#report-a-problem)
describes the report and its issue link.
