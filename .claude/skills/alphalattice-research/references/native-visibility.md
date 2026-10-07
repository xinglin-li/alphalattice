# Native sessions and Team
Date: 2026-10-06

Default research and Team need no product hook or hook approval. Observation starts no model,
agent or Task and grants no execution or write rights. Configuration alone proves no live child.

## Configure and bind

Open the host on the intended project; a shell `cd` does not rebind its Session. With the locked
environment (`.venv/bin/python` outside Windows), configure once:

```text
.venv/Scripts/python.exe scripts/native_research.py configure
```

For Claude use `native_research.py configure --host claude-code`. The host-local
`alphalattice-project.local.json` in `.codex/` or `.claude/` declares the project's host;
existing settings and hooks remain. `native_research.py doctor` checks default local readiness
without reading hook trust or requiring native attachment. Bind the actual Session inside it:

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
Use `--usage off` to disable this Session's native file discovery and usage reads; see
"Continuous usage and optional native proof" below.

`scripts/materialize_claude_host.py` derives Claude cards and byte-identical Skill files from
Codex cards and this Skill; `--check` reports drift. Neither default host declaration registers
product lifecycle hooks. Analyst and CRO cards retain their restricted read/answer tools.

## Public Conversation and closure

Submit only intentional, safe coordination text on stdin. The lead sends an exact assignment
after preparing its bundle and while holding the [Goal](goals.md):

```text
native_research.py message --kind assignment --to <child-id> --reference <bundle-reference>
```

Use the returned `message_id` in `--reply-to <message-id>`. Kinds are assignment, question,
answer, objection, pm_response, plan, decision, dead_end and surprise. Specialists publish
messages only where their loaded card permits the command; Analyst and CRO remain commandless.
The lead submits their nominated answers. A task name is a locator, not proof of a loaded role.

The Host files each message under its exact bound Goal and Session. Messages are actor-declared,
not verified findings or native speech. Product operations supply receipts and sealed answers
through the same bridge. The accepted answer retains its original submission context on retry;
without strict native proof its author stays `NOT_OBSERVED`.

Only the matching bundle's product acceptance, or the lead's explicit terminal decision, closes
an exact assignment. An ordinary reply leaves it open. For a lead decision use `native_research.py message --kind
decision --reply-to <assignment-message-id> --terminal-decision COMPLETED --terminal-reason
<public-reason>`; WITHDRAWN and DECLINED are also allowed. Closure records its source and reason;
it proves neither child exit nor native authorship. Open assignments block Goal submission.

Text is bounded to 4,000 characters, with 500 retained; identifiers and references to 200.
Longer content cites an exact Task UUID, 64-hex artifact hash, History entry or `case:<hash>`.
Never send raw prompts, conversation, credentials or hidden reasoning. Same content retries
idempotently. Delivery failure is visible and never changes the research result; retry the
same observation or accepted submission from its original bound context, not the research.
Do not delete the bounded local sequence store to retry.

## Continuous usage and optional native proof

With usage enabled, the running workspace Host reads the bound lead and exactly assigned
children independently of Stop. Codex uses exact thread ancestry; Claude uses exact Session,
agent and admitted metadata, never filename, timing or role guesses. The whitelist discards
conversation and tools. Team and Goal show each member's latest cumulative reading by model,
source and source time; missing counts remain absent. Parent and child counts are not combined.
Unavailable, incomplete or failed delivery readings are diagnostic; research continues.

Legacy Start/Stop observation is explicitly optional: `native_research.py configure --native-proof` and `native_research.py doctor
--native-proof`, with `--host claude-code` for Claude. The person reviews actual definitions in
the host's first-use or changed-definition trust UI; never bypass trust or edit its records.
Strict readers verify exact definitions, Start, assignment and accepted-answer attribution;
declared role names and Stop callbacks alone prove none of these. Default readiness does not
claim this proof or request its approval.
