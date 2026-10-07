# Privacy and native usage reading
Date: 2026-10-06

The native bridge's usage reader can read usage facts from this machine's Claude Code or Codex session files. It retains a whitelist of response IDs, models, recorded effort, token counts and timestamps (including recorded first/last times); it discards conversation text and other file content.

## Files and fields

| Host | Admitted session root |
| --- | --- |
| Claude Code | `~/.claude/projects/` or `CLAUDE_CONFIG_DIR/projects/` |
| Codex | `~/.codex/sessions/` or `CODEX_HOME/sessions/` |

Default research requires no product hook. The running workspace Host reads the bound lead and exactly assigned children independently of Stop. Codex uses exact thread ancestry; Claude validates the unique Session file, child header and matching metadata. Symlinks, non-files, wrong Session names, ambiguous parents and paths outside the resolved host root are refused. The reader retains neither paths, prompts, assistant text, tool bodies nor private reasoning. Optional legacy hook readers remain separate.

Before `goal take`, `goal submit` and `bundle submit`, the CLI may read only the native lead session named by the exact host/Session binding, and only when the command session matches and usage reading is on. Another Session's binding grants no read. Without a matching binding, enabled reading or admitted file, it reads nothing. A failed usage read does not stop the research request.

Recorded facts include uncached input, cache reads/writes, output tokens, model, source and source time. A Claude response repeated across content blocks counts once using later counts; Codex input subtracts cached tokens. Each participant/model shows its latest cumulative snapshot; missing or impossible counts remain unknown. Parent and child readings are not combined without proof that they do not overlap. Observed model/effort may be compared with role-card settings. Answer attribution preserves its recorded basis, including `NOT_OBSERVED`, and never changes judgment identity. Usage is neither a bill nor proof of trust or research success. Intentional public messages, product receipts and optional native observations retain separate sources. Turning reading off preserves those records and earlier usage.

## Turn usage reading off

Bind the actual native Session with `--usage off`; see [Getting started](getting-started.md). The default is `--usage read`. Each actual host/Session binds independently without removing or replacing another record. To change this Session's usage, workspace or roles, run `alphalattice session unbind` yourself, then bind again with the explicit workspace. Run unbind at the end too; it removes only your own record and keeps workspace history.

`native_research.py doctor` reports `usage_reading` as `OFF`, `READ` or null. OFF skips native file discovery and reads for this Session; another Session's READ setting cannot enable them. A Codex child whose ancestry was not admitted must name `--workspace` explicitly; OFF never grants another Session's binding. Public messages and research continue. This control does not change the host's own storage.

## Separate launch counter

The CLI can use a valid agent-session identifier from its environment to append a constant `1` line to a temporary per-host/session count under `alphalattice-cli-launches/`. Help and schema launches count. This does not read host session files or write to the research workspace. `--usage off` does not disable it; optional `session_launches` includes the current launch. The field is omitted when no valid session or writable counter is available, without blocking the command. See [CLI](cli.md).
