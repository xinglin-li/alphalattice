# Privacy and native usage reading
Date: 2026-10-03

The native bridge's usage reader can read usage facts from this machine's Claude Code or Codex session files. It retains a whitelist of response IDs, models, recorded effort, token counts and timestamps (including recorded first/last times); it discards conversation text and other file content.

## Files and fields

| Host | Admitted session root |
| --- | --- |
| Claude Code | `~/.claude/projects/` or `CLAUDE_CONFIG_DIR/projects/` |
| Codex | `~/.codex/sessions/` or `CODEX_HOME/sessions/` |

Claude hooks identify lead and subagent files; names and paths must match the admitted session/agent. Codex files are located by thread ID, not hook paths. Symlinks, non-files, wrong session names and paths outside the resolved host root are not admitted. A bound specialist's stop hook may read the admitted lead and specialist usage records and turn context needed for attribution. It does not retain prompts, assistant text, tool bodies or private reasoning.

Before `goal take`, `goal submit` and `bundle submit`, the CLI may read only the native lead session named by the local binding, and only when the command session matches and usage reading is on. Without a matching binding, enabled reading or admitted file, it reads nothing. A failed usage read does not stop the research request.

Recorded facts include uncached input, cache reads/writes and output tokens. A Claude response repeated across content blocks counts once using later counts; Codex input subtracts cached tokens. Incomplete or impossible counts remain incomplete. Reports may group facts by participant/model and compare observed model/effort with role-card settings. Evidence/CRO answers can retain host/session/agent/role/model/effort beside the judgment, based on recorded hook and usage observations; the basis names a hook, role card, session file or `NOT_OBSERVED`. Missing observations are not invented and do not enter judgment identity. Counts and timestamps are usage evidence, not a bill, proof of host trust or proof of research success. An actor's submitted assignment, reply or finding is a separate intentional product record. Turning usage reading off does not remove those records or earlier usage.

## Turn usage reading off

Bind the actual native session from inside it to the existing workspace with `--usage off`; see the binding form in [Getting started](getting-started.md). The default is `--usage read`. A different existing binding must be unbound first with `alphalattice session unbind`. `native_research.py doctor` reports `usage_reading` as `OFF`, `READ` or null. With `off`, stop hooks and lead commands skip usage reading; Codex binding lookup can still read a thread's parent and role from its first record. Hooks, explicit messages and research continue. Unbinding removes the native binding, not workspace history. This control does not change what the host itself stores.

## Separate launch counter

The CLI can use a valid agent-session identifier from its environment to append a constant `1` line to a temporary per-host/session count under `alphalattice-cli-launches/`. Help and schema launches count. This does not read host session files or write to the research workspace. `--usage off` does not disable it; optional `session_launches` includes the current launch. The field is omitted when no valid session or writable counter is available, without blocking the command. See [CLI](cli.md).
