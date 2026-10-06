# Getting started on Windows

Use the [setup and launch procedure](../../.agents/skills/alphalattice-research/references/operating.md#setup-and-launch) for locked Windows/Python 3.12 dependencies, the editable checkout or installed wheel, persistent PATH, native configuration and workspace launch. That reference ships with the Skill in configured projects. macOS/Linux remain unverified.

## First preparation

Give your agent the research intent in one sentence. Before a workspace's first preparation, its agent opens the one `FIRST_USE` goal with that sentence copied word for word as `objective`. This goal opens once per workspace and is never revised. Follow the exact declaration and command sequence in [the agent guide](../../AGENTS.md#run-the-first-use-from-the-persons-sentence).

For 24 hours after opening, the goal delegates only opening preparation network access, confirming the preparation and its resumes, and deciding its data issues. `goal show` records `record.delegated_steps` and the delegation's end/active state in `record.delegation`. Submission, abandonment or expiry ends that authority and closes a network setting still held by the delegation; a later setting made by you remains. You can stop the goal in Local Web or ask the agent to abandon it. Task cancellation is separate. Activation, storage, automation, revocation and paid choices remain yours.

Workspace network access starts closed. Outside this first-use delegation, you open it and confirm the initial preparation preview on Home in Local Web. `ALPHALATTICE_NETWORK_DISABLED=1` keeps workspace acquisition offline even if the control allows it; research calculations are held offline. If that override holds the Host, its operator must remove it from the launch environment and restart the idle service. Setup downloads use their own explicit consent; live Evidence source admission is separate. Planning preparation does not acquire sources. Preview again after source access is admitted, then read the source mode, sources, limits and next action. It distinguishes approved acquisition, qualified local reuse and source revalidation. The candidate count can remain unknown until capture.

Past first use, only the person admits network access in Settings and confirms the initial preparation preview on Home; an automation resume must match its recorded delegation. A new preview alone grants no source access. If a data issue interrupts preparation, inspect its evidence and permitted decisions. After a disconnect, follow the existing Task instead of submitting another preparation. **Stop** or `goal abandon` stops the first-use goal; closing network in Settings is another control. Task cancellation is separate.

If a provider defers preparation, its read gives `failure_code` and `progress.retry_after_at`. Keep fetched listings and resume the same plan through its permitted confirmation path after that time; an early retry is refused as `workspace_preparation.retry_not_due`. A timeout or unreadable response can leave failed listings and a data issue instead. Read its evidence and permitted decision. Published inputs remain unchanged. See [Data and care](data-and-care.md) for maintenance retry and backup details.

## Measured preparation and CPU budget

The 2026-09-30 reference run at checkout `c109a7bc` took 407.1 seconds and peaked at 3.99 GB on Windows 11 Pro with an Intel Core i9-13900K (24 cores, 32 logical processors) and 128 GB memory. It used a fixed budget of eight logical processors on four P-cores, from an empty workspace through the data decision to completion; a comparison ran concurrently on disjoint cores. This measurement is context, not a hardware minimum or runtime guarantee.

```powershell
alphalattice cpu-budget show
alphalattice cpu-budget set --cores 8
alphalattice cpu-budget set --cores auto
```

`show` reports budget, processors, memory, load and recent work. `auto` uses processors not already busy when work starts. A fixed positive count bounds preparation CPU use; the calculation is unchanged. Queue capacity is a separate `--queue` setting; change one setting per request.

Once preparation publishes an input, follow [Research flows](research-flows.md). [CLI](cli.md) explains saved answers and continuations; [Third-party data](third-party-and-data.md) owns model packs, sources and their terms.
