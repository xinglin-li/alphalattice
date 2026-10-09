# Privacy and native usage reading
Date: 2026-10-08

The workspace Host reads usage facts from this machine's Claude Code or Codex files for Sessions bound to the workspace. It keeps response IDs, models, recorded effort, token counts and timestamps, including first and last times. Conversation text, prompts, tool bodies, private reasoning and file paths are discarded.

## Files and readings

| Host | Admitted session root |
| --- | --- |
| Claude Code | `~/.claude/projects/` or `CLAUDE_CONFIG_DIR/projects/` |
| Codex | `~/.codex/sessions/` or `CODEX_HOME/sessions/` |

Research requires no product hook. The reader opens the bound Session's file and its admitted product specialists: Codex records child threads in the lead's rollout; Claude uses `subagents/agent-<id>.jsonl` beneath the lead's session directory. A child's metadata must name that lead and an admitted specialist role. General helpers, including specialists started from card text in the installing session, are not read. Other Sessions, symlinks, non-files, ambiguous parents, wrong Session names and paths outside the resolved root are refused.

Reads occur when a goal is taken or submitted, an answer is submitted, a Team or Goal page opens, or `alphalattice session usage` asks. There is no timer; a Session is read at most once in ten seconds. Reading failures are named and do not stop research.

Supported formats were checked against Codex CLI 0.162.0-alpha.2 and Claude Code 2.1.293 on 2026-10-07. Unknown or changed records remain unavailable. Facts include uncached input, cache reads and writes, output, model, source and source time. Repeated Claude response blocks count once using later counts; Codex input subtracts cached tokens. Each participant retains its latest cumulative snapshot per model. Missing or impossible counts stay unknown; parent and child totals are combined only with proof they do not overlap. Observed settings may be compared with role-card settings. Accepted answers record the submitting lead; the author remains unobserved. Usage establishes neither billing, trust nor research success.

## Privacy controls

Reading starts enabled. Either control disables it:

- **Settings** controls all Sessions bound to this workspace. `alphalattice usage-reading show` reads this setting. Only a person changes it; an unreadable or invalid switch record reads as off.
- `alphalattice session bind --usage off` disables one Session's native file discovery and reading. Another Session cannot enable it. Automatic binding uses `--usage read`.

To change your Session's workspace, usage or roles, run `alphalattice session unbind`, then bind again with the explicit workspace. Unbind removes only your own record and keeps workspace history. See [native sessions](../../.agents/skills/alphalattice-research/references/native-visibility.md) for the binding contract. Earlier readings and product records remain, and neither control changes the native host's storage.

## Separate launch counter

A valid agent-session identifier permits the CLI to append a constant `1` to a temporary counter under `alphalattice-cli-launches/`. Help and schema launches count. This reads no session files and writes nothing to the research workspace. `--usage off` does not disable it; optional `session_launches` includes the current launch and is omitted if the identifier or writable counter is unavailable. See [CLI](cli.md).
