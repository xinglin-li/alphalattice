# Getting started
Date: 2026-10-08

Give your agent the repository link and one research intent. It installs
AlphaLattice and continues in the same session. The
[setup procedure](../../.agents/skills/alphalattice-research/references/operating.md)
covers locked Python 3.12 and uv dependencies, the editable checkout or installed
wheel, and Local Web. The [README](../../README.md) states the scope of platform checks.

## Prerequisites by system

Every system needs Python 3.12, uv and a browser; the lock brings the Node that builds the Workbench.

- **Windows:** nothing more.
- **macOS:** libomp, which LightGBM loads to train and score models: `brew install libomp`.
- **Linux:** libatomic1, which the locked Node loads to build the Workbench, and libgomp1, which LightGBM
  loads to train and score models: `sudo apt install libatomic1 libgomp1`, or your distribution's packages.

These are system packages outside the lock. The doctor (`uv run python -m alphalattice.interface.local_application.native_setup doctor`, after the locked sync below)
names any missing under `system_libraries`, and your agent asks you once before installing them.

## First run from a source checkout

Use Windows PowerShell from the checkout root. Your agent follows these steps;
[operating the CLI](../../.agents/skills/alphalattice-research/references/operating.md)
holds the installation and launch procedure.

1. Install the locked environment before running any product module:

   ```powershell
   uv sync --locked --all-extras
   ```

2. Read the setup requirements with that environment's interpreter:

   ```powershell
   uv run python -m alphalattice.interface.local_application.native_setup doctor
   ```

   For each missing `host_readiness` item, follow its `who_decides` and repair
   instructions. Only missing items carry `who_decides`; it is not an outstanding
   decision on an item already present. Commands are under `facts.commands`, or
   `facts.command` for the Workbench build. Typical checkout repairs are:

   ```powershell
   uv run python scripts/build_local_web_ui.py --product
   uv export --locked --all-extras --no-dev --no-emit-project --no-hashes --output-file .venv/alphalattice-runtime-requirements.txt
   uv tool install --python 3.12 --editable . --with-requirements .venv/alphalattice-runtime-requirements.txt --constraints .venv/alphalattice-runtime-requirements.txt
   ```

   Run the retrieval runtime's offline fill exactly as the doctor prints it.
   If its package cache is cold, follow its returned network form within the
   stated download permission. If the installed command is off PATH, you run
   `uv tool update-shell` once and restart the session or terminal, then the agent
   repeats the doctor. Dependency repairs outside the lock require your permission.

3. Configure the agent project from the same checkout root:

   ```powershell
   uv run python -m alphalattice.interface.local_application.native_setup configure --host claude-code
   ```

   Use `--host codex` for Codex; omitting the flag selects Codex. The installing
   agent continues in the same session by reading `continue_here`'s files. In a shell
   with no agent session, give the doctor the same `--host` choice; otherwise it
   defaults to Codex.

4. Run the doctor again after repairs and configuration. Finish missing required
   setup before the first use. Workspace-dependent readings can remain unavailable
   before its Host starts; the optional user layer and its backup are not research
   prerequisites.

5. Start one Host on an explicit workspace and keep its printed launch link:

   ```powershell
   alphalattice --workspace "workspaces/my-research" serve --no-browser --detach
   ```

   The [agent guide](../../AGENTS.md) says how the agent opens that link visibly
   in its coding app's Workbench pane. Reuse a Host already serving this workspace.

6. Start the first use with your exact sentence and the date you named:

   ```powershell
   alphalattice --workspace "workspaces/my-research" first-use prepare --sentence "Positions for 2026-10-19, reviewed." --date 2026-10-19
   ```

   A bound agent session carries its goal automatically. A shell with no agent
   session must keep naming the workspace and add `--goal <goal_id>` from the
   first-use answer on later steps. The same sentence and date reuse this
   workspace's first use; follow its returned next action.

### Release wheel

The [operating reference](../../.agents/skills/alphalattice-research/references/operating.md)
gives the wheel's locked tool install; no checkout is needed. Run its doctor and
configure with the installed tool's own interpreter, naming your agent folder:

```powershell
$toolDir = uv tool dir
& "$toolDir/alphalattice/Scripts/python.exe" -m alphalattice.interface.local_application.native_setup --project <agent-project> doctor --host claude-code
& "$toolDir/alphalattice/Scripts/python.exe" -m alphalattice.interface.local_application.native_setup --project <agent-project> configure --host claude-code
```

On macOS and Linux that interpreter is `bin/python`. Then run the doctor again,
start the Host with an explicit workspace and follow the same first-use and goal
rules above.

## First preparation

The [README overview](../../README.md#two-ways-to-use-alphalattice) describes
both supported uses:

- **First use:** start in a new folder with your intent, or
  `Positions for <date>, reviewed.` The agent prepares data and the strategy,
  then follows the date's positions through review, committee discussion and a report.
  It stops once to ask whether daily updates may keep the network on for market
  data and SEC filings within the default budget (3 per issuer, 600 documents
  and 3 GB overall). **Work waits until you answer yes or no.** Your yes is a
  continuing setting until you tell it to stop. At the first-use report, the
  agent also asks whether daily updates should convene the investment committee;
  without that setting, later updates get the portfolio manager's view on the
  reviewed positions.
- **Daily update:** open a new session in the same folder and say
  `Positions for <next session>, reviewed.` It uses the installed strategy and
  standing network yes to update data and produce that session's reviewed research
  positions, following your daily committee setting. It asks nothing further unless
  a decision belongs to you.

Your first-use sentence becomes the goal's unchanged objective. Its delegation
lasts 24 hours from opening: the agent handles the admitted preparation and
activates the book, telling you each act. Accepted submission, abandonment or
expiry ends that delegation. The separate continuing permission above survives
it. The [agent guide](../../AGENTS.md) owns the flow and the list of
[person decisions](../../AGENTS.md#what-only-a-person-decides).

Follow Tasks, results and review in the Workbench. Deactivate the book on
**Portfolio**. To stop the goal, use **Stop** in Local Web or ask the agent to
abandon it; cancel a running Task separately.

Each answer names its next step. A refusal can identify a data shortfall, missing
permission or required decision; read its explanation and permitted continuation.
A provider deferral retains fetched listings and names when the existing
preparation can resume. After a disconnect, the agent reads the existing Task.
[Data and care](data-and-care.md) explains recovery.

Workspace network access starts closed, and `ALPHALATTICE_NETWORK_DISABLED=1`
keeps the process offline even when its workspace setting allows access.
Research calculations are held offline. An admitted preparation answer's offered
confirmation is sent; network access becomes a step only when a refusal names it.
A first-use delegation can authorize that step only while its terms hold and
never replaces your own network setting. [Privacy](privacy.md#workspace-network-and-consent)
gives the exact consent words, activity disclosure and controls for stopping access.

## Measured preparation and CPU budget

The 2026-09-30 reference preparation run took 407.1 seconds and peaked
at 3.99 GB on Windows 11 Pro, Intel Core i9-13900K (24 cores, 32 logical
processors), with 128 GB memory. From an empty workspace through its data decision
to completion, it used eight logical processors on four P-cores; a comparison
ran concurrently on disjoint cores. This observation is neither a hardware
minimum nor a runtime guarantee.

```powershell
alphalattice cpu-budget show
alphalattice cpu-budget set --cores 8
alphalattice cpu-budget set --cores auto
```

`show` reports capacity, load and work. `auto` uses processors not already
busy when work starts; a positive count bounds preparation CPU use without
changing the calculation. Queue capacity is a separate `--queue` setting.

After preparation publishes an input, [Research flows](research-flows.md)
explains the research choices. [CLI](cli.md) explains answers and continuations,
and [Third-party data](third-party-and-data.md) describes sources, model packs
and their terms.
