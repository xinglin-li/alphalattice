# Upgrading AlphaLattice
Date: 2026-10-18

Use this guide to upgrade from 0.2.0 to a later 0.2.x patch. Keep using your
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

   The Host lets running Tasks finish and releases the workspace without
   cancelling them. Wait until it has stopped before replacing the installation.

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
the runtime requirements from that same wheel and use them as both requirements
and constraints, as in the [operating reference](../../.agents/skills/alphalattice-research/references/operating.md#setup-and-launch).
The example below uses the 0.2.1 wheel; replace its filename in both commands
with the wheel you downloaded:

```powershell
python -c "from zipfile import ZipFile; from pathlib import Path; Path('alphalattice-runtime-requirements.txt').write_bytes(ZipFile('alphalattice-0.2.1-py3-none-any.whl').read('alphalattice/_runtime/config/release/runtime-requirements.txt'))"
uv tool install --force --python 3.12 ./alphalattice-0.2.1-py3-none-any.whl --with-requirements alphalattice-runtime-requirements.txt --constraints alphalattice-runtime-requirements.txt
```

`--force` replaces the previous tool installation. To switch between the wheel
and source-checkout installation methods, first run `uv tool uninstall alphalattice`,
as the operating reference describes.

Run the new tool's doctor against your existing agent project. Replace
`<agent-project>` with that folder's path:

```powershell
$toolDir = uv tool dir
& "$toolDir/alphalattice/Scripts/python.exe" -m alphalattice.interface.local_application.native_setup --project "<agent-project>" doctor --host codex
```

If the doctor calls for configuration, use the same interpreter and project:

```powershell
& "$toolDir/alphalattice/Scripts/python.exe" -m alphalattice.interface.local_application.native_setup --project "<agent-project>" configure --host codex
```

Use `--host claude-code` for Claude Code. On macOS and Linux the interpreter is
`bin/python` rather than `Scripts/python.exe`; see [Getting started](getting-started.md#release-wheel)
for platform requirements. Follow any setup repairs, read `continue_here` after
configuration and run the doctor again.

## Continue your daily update

Start the Host on the workspace path you saved:

```powershell
alphalattice --workspace "<existing-workspace-path>" serve --no-browser --detach
```

Open its printed Workbench link in your coding app's built-in browser. In the
same agent project, ask for your next daily update as usual, for example
`Positions for <next session>, reviewed.`

The first daily update after an upgrade may need a one-time rebuild of derived
work. Allow it to finish; later updates can reuse that work. Earlier sealed goals
and reports keep verifying, and your existing research history stays available.

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
