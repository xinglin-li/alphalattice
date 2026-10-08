# Privacy and native usage reading
Date: 2026-10-07

The workspace Host can read usage facts from this machine's Claude Code or Codex session files, for the agent Sessions bound to the workspace it serves. It retains a whitelist of response IDs, models, recorded effort, token counts and timestamps (including recorded first/last times); it discards conversation text and other file content.

## Files and fields

| Host | Admitted session root |
| --- | --- |
| Claude Code | `~/.claude/projects/` or `CLAUDE_CONFIG_DIR/projects/` |
| Codex | `~/.codex/sessions/` or `CODEX_HOME/sessions/` |

Research requires no product hook. For each Session bound to the workspace, the Host opens only that Session's own file and the files of the specialists that file records: a Codex lead's rollout names each specialist thread it started, and a Claude Code lead's specialists are the `subagents/agent-<id>.jsonl` files under its own session directory, each read only when its own metadata names that lead. Another Session's file is never opened. Symlinks, non-files, wrong Session names, ambiguous parents and paths outside the resolved host root are refused. The reader retains neither paths, prompts, assistant text, tool bodies nor private reasoning.

The Host reads only when research asks: when a goal is taken or submitted, when an answer is submitted, when a Team or Goal page opens, or when `alphalattice session usage` asks. Nothing reads on a timer, and one Session is read at most once in ten seconds. A failed read is named beside the others and never stops research.

The reader knows the record formats of Codex CLI 0.162.0-alpha.2 and Claude Code 2.1.293, as checked on 2026-10-07. A record it does not know, from another version or a changed format, reads as unavailable, never as zero or as a partial count.

Recorded facts include uncached input, cache reads/writes, output tokens, model, source and source time. A Claude response repeated across content blocks counts once using later counts; Codex input subtracts cached tokens. Each participant/model shows its latest cumulative snapshot; missing or impossible counts remain unknown. Parent and child readings are not combined without proof that they do not overlap. Observed model/effort may be compared with role-card settings. An accepted answer is recorded as submitted by the lead, its author `NOT_OBSERVED`; attribution never changes judgment identity. Usage is neither a bill nor proof of trust or research success. Turning reading off keeps product records and earlier usage.

## Turn usage reading off

Reading is on by default. Either of two switches turns it off, and then nothing is read.

A person's switch on **Settings** turns reading off for every Session bound to the workspace. `alphalattice usage-reading show` reads it; only a person sets it, and an agent's request to set it is refused. A switch record the product did not write, or cannot read, reads as off.

For one Session, bind the actual native Session with `--usage off`; see [Getting started](getting-started.md). The default is `--usage read`. Each actual host/Session binds independently without removing or replacing another record. To change this Session's usage, workspace or roles, run `alphalattice session unbind` yourself, then bind again with the explicit workspace. Run unbind at the end too; it removes only your own record and keeps workspace history.

`native_research.py doctor` reports `usage_reading` as `OFF`, `READ` or null. OFF skips native file discovery and reads for this Session; another Session's READ setting cannot enable them. A Codex child whose ancestry was not admitted must name `--workspace` explicitly; OFF never grants another Session's binding. Research continues. Neither switch changes the host's own storage.

## Separate launch counter

The CLI can use a valid agent-session identifier from its environment to append a constant `1` line to a temporary per-host/session count under `alphalattice-cli-launches/`. Help and schema launches count. This does not read host session files or write to the research workspace. `--usage off` does not disable it; optional `session_launches` includes the current launch. The field is omitted when no valid session or writable counter is available, without blocking the command. See [CLI](cli.md).
