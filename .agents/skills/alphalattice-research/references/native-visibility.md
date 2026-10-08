# Native sessions and Team
Date: 2026-10-08

Research and Team need no product hook and no hook approval. Binding starts no model, agent or
Task and grants no execution or write rights. Configuration alone proves no live child.

## Configure and bind

Open the host on the intended project; a shell `cd` does not rebind its Session. With the locked
environment (`.venv/bin/python` outside Windows), configure once:

```text
.venv/Scripts/python.exe scripts/native_research.py configure
```

For Claude use `native_research.py configure --host claude-code`. The host-local
`alphalattice-project.local.json` in `.codex/` or `.claude/` declares the project's host.
Existing settings remain; configure removes only the product's own retired lifecycle hook
groups. `native_research.py doctor` checks local readiness. A Session is bound when it first
works on a workspace, in the configured project above the workspace when there is one, else in
the workspace; its first research request opens a goal when it holds none. Bind explicitly only to rebind or to turn reading off:

```text
alphalattice --workspace <explicit-workspace> session bind --usage read
```

Ignored project-local binding records hold host, Session, workspace and roles, never a token
or research authority. Each actual host/Session has its own record. A new Session binds
independently, without cleaning up or replacing another Session's record. Its lead and admitted
children can omit `--workspace` through that binding only.

To change workspace, usage or roles for the same Session, run `alphalattice session unbind`
inside it, then bind again. At the end, the agent runs that command itself; it removes only
its own record. Research, original Goal and answer context, and retained history stay.
A person outside an agent Session may unbind a single unambiguous record; with multiple
records the command refuses to choose. Do not edit or clear another Session's record.

`scripts/materialize_claude_host.py` derives Claude cards and byte-identical Skill files from
Codex cards and this Skill; `--check` reports drift. Neither host declaration registers a
product hook. Analyst and CRO cards retain their restricted read/answer tools.

## Team and the Goal's Conversation

Team and the Goal's Conversation show what the product recorded of a bound Session's work:
its requests, the bundles it prepared, the answers it submitted and the ones the Host accepted,
and the usage read below. There is no message command. Record a decision or conclusion worth
keeping with `goal note` ([goals](goals.md)).

The lead submits every answer, so an accepted answer is filed as the lead's. Which specialist
wrote it is not observed, and its author reads `NOT_OBSERVED`. A prepared bundle with no
accepted answer is listed under `open_assignments` in `goal show` and in the completion
answer. It reminds and never blocks submission. A failed filing is named and never changes
the research result; submitting the same accepted answer again from its original bound
Session files it.

## Usage

With reading on, the Host reads the bound Session's own session file and the specialists that
file records. It reads when a goal is taken or submitted, an answer is submitted, a Team or Goal
page opens, or `alphalattice session usage` asks; never on a timer and never another Session's
file. A Codex lead's rollout names each specialist thread it started. A Claude Code lead's
specialists are the `subagents` files under its own session directory, each read only when its
metadata names that lead. The whitelist keeps models, efforts, token counts and times and
discards conversation and tools. Team and the Goal show each member's latest cumulative reading
by model, source and source time; parent and child counts are not combined. A format the reader
does not know reads as unavailable, never as zero. An unavailable reading is named, and
research continues.

Reading is on by default. `session bind --usage off` keeps one Session's files unread. The
person's switch on **Settings** turns reading off for every Session bound to the workspace;
`usage-reading show` reads it, and only a person sets it. Either way nothing is read, and
earlier readings stay.
