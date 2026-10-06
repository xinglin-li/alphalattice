# Native observation
Date: 2026-10-04

An optional bridge that shows the lead and its specialists on Local Web's Team page. It starts
no model, agent or Task, and it never grants execution or write rights; ordinary research needs
none of it. Do not claim a live team from configuration alone.

## Set up

Open the host on the product checkout (a shell `cd` does not rebind a session) and run, with the
checkout's locked environment (`.venv/bin/python` elsewhere than Windows), then bind the session
once, which reads its host and session id itself and names the workspace for its commands too:

```text
.venv/Scripts/python.exe scripts/native_research.py configure
.venv/Scripts/python.exe scripts/native_research.py doctor
alphalattice --workspace <explicit-workspace> session bind
```

`configure` checks the shipped declarations and the local environment; `doctor` reports local
facts only, never the host's trust or attachment. Review the two project hooks in the host's own
trust UI; never use a trust-bypass flag or edit global trust records. The ignored
`.codex/native-research.local.json` holds the session, workspace and role names, never a token,
and is no research authority. A different binding refuses: when its scope is finished,
`alphalattice session unbind` removes it, the bound session's own or the person's from
their own shell; it stops only the observation and the default workspace.

Claude Code reads the same product through the same bridge: run
`native_research.py configure --host claude-code` and the same `session bind`, which names
the host from the session it runs in.
`scripts/materialize_claude_host.py` derives its files from the Codex cards and this Skill (edit
those, then rerun it; `--check` reports drift): one subagent per card, the two evidence
specialists with only Read and Write and a medium-effort card each, a byte copy of this Skill
and the two lifecycle hooks in `.claude/settings.json`. A binding made before the medium cards
existed does not name them until the session is bound again.

## What it carries

- The start and stop hooks keep bounded parent, turn, child and role metadata, and, once a
  stop's own delivery succeeded and unless usage reading is off, the token counts so far that
  the host's session files hold (never their text or paths); a missing or unreadable record
  gives none. Transcript paths, raw tools and final message bodies are dropped. A stop hook
  proves no finished turn or product result; a role name is the subagent's declared type, not
  proof that its card loaded.
- Explicit, user-safe coordination text goes on stdin. The lead names only the kind and its
  recipient or reply; a specialist adds its child locator and loaded role:
```text
.venv/Scripts/python.exe scripts/native_research.py message --kind assignment --to <child-id>
.venv/Scripts/python.exe scripts/native_research.py message --agent-id <child-id> --role <loaded-role> --kind objection --to <parent-id> --reference <exact-product-reference>
```

  Kinds: assignment, question, answer, objection, pm_response, and the decision notes plan,
  decision, dead_end and surprise, written only at such a turn, never as raw reasoning. A reply
  names `--reply-to <message-id>`, the `message_id` the answer to the message it answers gave;
  an assignment needs `--to`. A message naming no sender is the bound session's, as
  `research_lead`; specialists use their role names and real child locators. A message's id is
  its content's: the same message sent again is the same message, and a changed text a new one.
- The Host files each message under the goal the session holds ([goals](goals.md)). A message is
  actor-declared content, not a host-verified event or a validated finding, and it submits
  nothing: the lead submits every specialist's nominated answer with `bundle submit`.
  The five stage specialists keep their card's ANALYZE/REVIEW reads and authorized EXECUTE
  commands; Analyst and CRO remain commandless. The nominated answer-file write is the child's
  last action, followed by one line: written; the child never submits. An assignment whose
  reference is its prepared bundle's key (its prepare answer's `bundle_reference`) is
  how that bundle's answer is credited to its child (AU3), checked against the child's start
  hook; an answer whose bundle no assignment names is recorded with no author, the session
  alone. An accepted answer with HOOK authorship adds one idempotent child exchange through
  the same bridge, marked `PRODUCT_ACCEPTED_ANSWER`, with the parent submitter and immutable
  answer acceptance time (or Task admission time for the domain answer). It is an accepted
  artifact projection, never reconstructed speech. Never send prompts, raw conversation,
  credentials or hidden reasoning.
- Text is bounded (4,000 characters; 500 kept), and each id or reference to 200 characters; a
  longer message cites an exact product reference (a Task UUID, a 64-hex hash, a History entry
  id or `case:<hash>`). Anything else is a declaration only, never fetched.

Delivery is advisory: a failure or timeout never blocks the host or asks for a retry of
research, and a missing binding is silent. The sender's sequence store is a named, bounded
local file; never delete it to retry. Host version differences are diagnostic, not a pin, and
host or telemetry failure leaves research and the CLI untouched.
