# Getting started on Windows
Date: 2026-10-08

Give your agent the repository link and one research intent. It installs
AlphaLattice and continues in the same session. The
[setup procedure](../../.agents/skills/alphalattice-research/references/operating.md#setup-and-launch)
covers locked Python 3.12 and uv dependencies, the editable checkout or installed
wheel, and Local Web. Windows is verified; macOS and Linux are unverified.

## First preparation

Your sentence becomes the first-use goal's unchanged objective. Its delegation
lasts 24 hours from opening: the agent handles the admitted preparation and
activates the book once reviewed, telling you each act. Submission, abandonment
or expiry ends it. The [agent guide](../../AGENTS.md) owns the flow and sole
list of [person decisions](../../AGENTS.md#what-only-a-person-decides).

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
The agent takes that step under the first-use delegation; outside it, you decide
in **Settings**.

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
