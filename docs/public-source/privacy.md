# Privacy and native usage reading
Last checked: 2026-10-22 against 0.2.3

The workspace Host reads usage facts from this machine's Claude Code or Codex files for Sessions bound to the workspace. It keeps response IDs, models, recorded effort, token counts and timestamps, including first and last times. From a Claude Code subagent's own commands that name a member key, it also keeps bounded attribution facts: the key's SHA-256, the committee thread or a hash of the bundle directory, the role named, whether the command submits, hashed claim and tool-use identifiers, and the floor message IDs or answer references its accepted submits made (at most 64 commands, 8 references each). Conversation text, prompts, tool bodies, raw keys, private reasoning and file paths are discarded.

## Workspace network and consent

At first use, your agent asks once, in the product's exact words:

> May AlphaLattice keep this workspace's network on for your daily updates, to fetch new market data and SEC filings within the default budget (3 per issuer, 600 documents and 3 GB overall), until you tell me to stop? Each use is recorded in the activity log and shown in the Workbench.

Your relayed yes sets the workspace network and the default SEC acquisition consent. It is your own continuing permission, separate from the first-use goal's 24-hour delegation. The delegation never overwrites your network setting, including a setting that keeps it off. A research run's offline hold and the operator's offline switch still take precedence.

An admitted network step reports `network_use`, with `under` naming whose setting allowed it and `words` giving the one-line disclosure. Activity records the same permission source, including for automated updates.

Say "Turn AlphaLattice's network off" to your agent, or turn it off in **Settings**. Your stop also blocks SEC acquisition before its next request; a data update already fetching finishes its current stage unless its Task is cancelled. A delegation cannot reopen your setting after that stop. A later change needs your decision again. Earlier inputs, Evidence and activity records remain available. Source and model terms are in [Downloads and data terms](third-party-and-data.md).

## Files and readings

| Host | Admitted session root |
| --- | --- |
| Claude Code | `~/.claude/projects/` or `CLAUDE_CONFIG_DIR/projects/` |
| Codex | `~/.codex/sessions/` or `CODEX_HOME/sessions/` |

Research requires no product hook. The reader opens the bound Session's file and its directly linked children: Codex records child threads in the lead's rollout; Claude uses `subagents/agent-<id>.jsonl` beneath the lead's session directory. A child's metadata must name that direct parent. On Claude Code, where a subagent's commands carry its lead's Session, a child whose own file holds a member key the Host issued reads as that member (a Codex specialist is told apart by its own thread): the Host checks the key's hash, and each message an accepted submit with it made, named in that submit's own result, shows under the child on the Team page. Messages the lead submitted stay the lead's, and a goal's record keeps every message's declared Session. Other Sessions, symlinks, non-files, ambiguous parents, wrong Session names and paths outside the resolved root are refused.

Reads occur when a goal is taken or submitted, an answer is submitted, a Team or Goal page opens, or `alphalattice session usage` asks. There is no timer; a Session is read at most once in ten seconds. Reading failures are named and do not stop research.

Supported formats were checked against Codex CLI 0.162.0-alpha.2 and Claude Code 2.1.293 on 2026-10-07. Unknown or changed records remain unavailable. Facts include uncached input, cache reads and writes, output, model, source and source time. Repeated Claude response blocks count once using later counts; Codex input subtracts cached tokens. Each participant retains its latest cumulative snapshot per model. Missing or impossible counts stay unknown; parent and child totals are combined only with proof they do not overlap. Observed settings may be compared with role-card settings. Keyed Analyst and CRO answers record the submitting member Session; that record and usage do not independently authenticate its author. Usage establishes neither billing, trust nor research success.

## Privacy controls

Reading starts enabled. Either control disables it:

- **Settings** controls all Sessions bound to this workspace. `alphalattice usage-reading show` reads this setting. Only a person changes it; an unreadable or invalid switch record reads as off.
- `alphalattice session bind --usage off` disables one Session's native file discovery and reading. Another Session cannot enable it. Automatic binding uses `--usage read`.

To change your Session's workspace, usage or roles, run `alphalattice session unbind`, then bind again with the explicit workspace. Unbind removes only your own record and keeps workspace history. See [native sessions](../../.agents/skills/alphalattice-research/references/native-visibility.md) for the binding contract. Earlier readings and product records remain, and neither control changes the native host's storage.

## Separate launch counter

A valid agent-session identifier permits the CLI to append a constant `1` to a temporary counter under `alphalattice-cli-launches/`. Help and schema launches count. This reads no session files and writes nothing to the research workspace. `--usage off` does not disable it; optional `session_launches` includes the current launch and is omitted if the identifier or writable counter is unavailable. See [CLI](cli.md).
