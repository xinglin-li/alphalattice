# AlphaLattice: guide for your research agent
Date: 2026-10-08

**Your first step in every session, before any answer:** run `Get-Content
.alphalattice/user/memory/MEMORY.md, .alphalattice/user/guide.md` (or `cat` them),
then read each memory the request touches. They are the person's own layer (below)
and may change what you do; a missing file means there is none yet. Claude Code
has already imported the two, so it skips the command, not the memories: still
open each memory the request touches.

AlphaLattice computes and records local quantitative research. You state the
question, follow the product's answers and explain the evidence to the person.
The answers lead: each one's `next_action` and `next_requests` decide your next
step, and a nested part's own hint describes only that part.

## Read the person's layer

`.alphalattice/user/` holds this person's tuning of the checkout: memories, a local
guide, and method for each card and Skill. No release contains or writes it. Read
its index and guide first, as above, and `agent-notes/INDEX.md` in a workspace you
continue. They refine this
guide and never widen what it permits; a memory is background, so verify what it
names before acting on it. With the alphalattice-maintenance Skill, start the
experience maintainer at a goal's end, on the person's correction, and after a
critical handoff once the date's positions are published; back the layer up before
upgrading the checkout.

## Change the checkout

This checkout is the person's workspace to change. Fix runtime bugs and add
strategies, models and Features: read the failure's cause, fix its source owner,
add a regression test and run again ([failure and
recovery](.agents/skills/alphalattice-research/references/operating.md)).
Under `src/alphalattice/`, strategies are declared in
`investment/portfolio_strategy_lab/policies/installed_strategies.py`, models in
`capabilities/alpha_modeling/extensions/`, and Feature kernels in
`foundation/feature_engine/producers/factors/`. Follow
`docs/public-source/extending.md` and run the tests that answer for the change.

A changed computation gets a new method identity and new result evidence; earlier
results keep their recorded method and inputs. Fix an identity or evidence-binding
refusal at its cause; never bypass the check or edit a stored record to make it pass.

## Open the workspace

If you installed AlphaLattice in this session, continue here: read this guide and
the research Skill by the paths `configure` printed under `continue_here`, start
each specialist as a general subagent whose prompt is its card's text, and tell the
person its `disclosure` in one line ([setup](.agents/skills/alphalattice-research/references/operating.md)).
Use a new workspace for new research:

```powershell
alphalattice --workspace "workspaces/my-research" serve --no-browser --stop-on-stdin
```

Keep the service attached to stdin; `stop` closes it after its workers join. If
your host has a browser tool, open the printed launch link once so the person can
watch; otherwise print the link and go on. Never navigate or click the Workbench
to show work. Your Session binds itself when it first works on the workspace, and
its first research request opens a goal when it holds none; commands then omit
`--workspace`. Use the **alphalattice-research** Skill for research
([command contract](.agents/skills/alphalattice-research/references/operating.md)).

## Run the first use from the person's sentence

Start it with the person's exact sentence and the date it names for the positions
(today's when it names none), as you read it:

```powershell
alphalattice first-use prepare --sentence "Positions for 2026-10-09, reviewed." --date 2026-10-09 --output "<out>/first-use.json"
```

It opens the workspace's one `FIRST_USE` goal, prepares the data under the goal's
delegation and follows each Task. Tell the person first how the product reads the
date, from `first_use.date`: the positions are entered on its `entry_session`,
decided at the `formation_session` close, and a date that is not a session says
so. Asked during the US session for the next session's positions, the usual case,
`information_available` is false: give the person `reading` in one line (ready after
about 18:00 ET, and their own time where it differs; preparing now, it finishes
itself). Everything before the update runs now; the Host holds the update and runs
it once at that time (`held_until` on the activation's answer), and your goal wait
wakes when it ends. They correct the date with one note. `first_use.road` is the whole first use, one
command per step, each answer names its next action, and `first_use.setup` names
anything this session's setup still lacks, with its way on. Follow `ask_now`'s
returned words: disclose delegated steps in one line, and ask only for a decision
it leaves to the person.

The road: data, the strategy's models and Risk, its whole-support book (the
numerical check), activation with the date's update in the same act, then Evidence
and the CRO on that date's published positions, the committee and its report.
The historical book's own review runs only when someone asks for it. A stop names
its way on: decide each data issue with its offered `confirm`, or take the refused
step under the delegation, then run the same command again. The network setting and the offline switch matter only
when an answer refuses for them. Never restart the Host or serve a second one for
network access, and never ask the person to.

Follow the wait and first-use wake procedure in [operating](.agents/skills/alphalattice-research/references/operating.md).

While the goal is open, for 24 hours from opening, it delegates to you: opening
the preparation's network, confirming the preparation and its resumes, deciding
its data issues, confirming its membership changes, activating its book, and
setting up Evidence with recent SEC filings within the default budget and the
retrieval model download at the size its setup states. Tell
the person each act in one line; they stop the book by telling you, or on
**Portfolio**. Accepted submission, abandonment or expiry ends the delegation and
closes the network it opened. The `FIRST_USE` declaration is never revised.

`strategy build` runs the research strategy's required whole-support Alpha and
Risk studies and installs it, reusing completed studies, and offers each installed
package's book run.
For the model lifecycle, follow [leading research](.agents/skills/alphalattice-research/references/research-lead.md).

## Choose the path by intent

The person's own target comes first: a date or a request in their words outranks
`workspace show`'s first `intents` entry, so a request for a date's positions never
becomes a new preparation. Otherwise take that first intent. A prepared workspace or
an installed strategy goes to `strategy-book controls --package <package>`; open a
Factor study, an Alpha handoff or a Lab book only when the person asks for
exploration. A default (the model, the Risk window, the CPU budget) you take and
say in one line with how to change it; a delegation is the person's step you carry
under the first use; a permission stays theirs. Never change a study's model,
window or bounds to save time.

## Review the date's positions

```powershell
alphalattice strategy-book review --package <package> --dir "<out>/analysts" --update <task>
alphalattice review continue --dir "<out>/analysts" --cro-dir "<out>/cro"
alphalattice review continue --dir "<out>/cro"
```

The first reads that update's published positions, bound by their publication,
prepares their Evidence and writes every Analyst bundle; start one Evidence Analyst
per bundle. Without `--update` it reviews the strategy's whole-support book
instead, when someone asks for that. The second submits their answers and writes
the CRO's bundle; the third publishes the review. Each stops at the first answer
that needs another step; follow its `next_action`. Wait for each as the first use
says.

## Convene the committee

```powershell
alphalattice committee open --update <task>
alphalattice bundle prepare --role ALPHA --task <task> --dir "<out>/committee/alpha"
```

On a date's published positions, you, the PM, convene the investment committee
with Alpha, Risk and the CRO. `committee open` offers each specialist's bundle: its
view of the date's positions and its `submit` and `wait` on the floor. Prepare the
three (ALPHA, RISK, CRO), then in one turn start them as general subagents (the
registered CRO card runs no command, and each bundle's route runs two), each
prompt its card's text and its bundle's path, and submit your own stance after
reading `committee show --update <task>`. The open's answer gives your PM key
(`pm_key`), and your session's later open gives it again: pass it as `--key` with
`--role PM`, and keep it out of every file a specialist reads; each specialist's
key is in its own bundle. Each specialist
moves itself: it submits, then waits for what is addressed to it. Keep `committee
wait --update <task> --role PM --key <pm-key>` running in the background and rule
each challenge as it arrives (ADOPT, REJECT or FOR_THE_PERSON), then give the
verdict. The stances are blind until all four are in, each specialist has three
challenges or replies after its stance, and the floor closes at its time box
whoever is silent. Messages name holdings, points and messages by alias and type
no other digit. The committee changes no number. Ask the person
about each item it hands them in one line, and relay their words as a
PERSON_ANSWER. Then export the report from the closed floor:

```powershell
alphalattice committee show --update <task> --output "<out>/committee/floor.json"
alphalattice request --from "<out>/committee/floor.json" --action report
```

## Run an installed strategy forward

On every fresh session read `strategy-book controls --package <package>` for
`activation`, its book and horizon before a forward plan, and show the person
the offer's `review_holdings`: the book's last sealed holdings, not next positions.
Before activating, read `strategy_dates.information_cutoff` and the conditional
`strategy_dates.first_actionable_session`. Activation admits the first update in
the same act, for the first use's date while its goal is open: follow it at once,
then review its published positions on their own publication. Hold positions only
from the first actionable session; sessions before it are a causal replay, inside
the research window where marked. Report each position with its basis, dates and
the publication's `claim`
([reading the dates](.agents/skills/alphalattice-research/references/research-lead.md)):
research positions, never orders or advice. Past the horizon, about eleven
months, request a newer book. Time a phase from its Goal and Task records, never
from one `--wait`, and resume a stopped Task as itself after a fix.

Outside the first use, activation is the person's: introduce the strategy once,
from the controls with no number of your own, then ask in one line. Name its
components; its holdings count and rules (top-k, exits, tranches, capital split);
the book's results, as a replay and never a forecast; the date whose positions
activation produces and its entry session; and how to stop it.

## Delegate and ask for decisions

### What only a person decides

Everything else is a default you take and disclose in one line. These are the
person's. Name the decision and ask once, in one line; on a clear yes, send it
with their words, `--person-said "<their words>" --asked "<your question>"`.
Never paraphrase or invent the words, and never send the person to a page to
click. Each yes answers one request, once.

- Network access, except preparation and default SEC acquisition under the first
  use's delegation.
- Strategy activation, except under the first use's delegation; deactivation
  always. Model and Factor activation.
- Daily research automation and the usage-reading switch.
- Storage cleanup and pins; data-decision grants and their revocation.
- Preparation, data issues and membership changes outside the first use's
  delegation, and a new research-input version: keep the current input and tell
  the person newer data exists.
- Source scope beyond the default budget; retrieval model downloads outside the
  first use's delegation; a dependency outside the lock.
- Scope: a request authorizes its stated scope only. Updating source data,
  sealing a new input or starting a different experiment is a new decision.
- External sharing or publication. A paid action is never relayed. Isolation the
  host cannot enforce holds the run.

### Specialists

Load the shipped Data, Factor, Alpha, Risk or Portfolio card for its professional
question, the Evidence Analyst and CRO cards for their bundles
([specialists](.agents/skills/alphalattice-research/references/specialist-handoff.md)).
Give an executor the exact workspace, permitted operations, references, goal and
budget. You submit every answer with the preparation's `submit_command`; the
Host checks shape and bindings, never scientific correctness. A wait is one call,
never a poll.

## Interrupt and report

On interruption, stop admitting work and name running Tasks; `task cancel
<task-id>` stops one, and `task show <task-id>` confirms it. After a timeout or
lost connection, read the known Task rather than resubmitting its run. Deliberately
stopped work uses `goal abandon <goal-id> --reason <reason>`.

Complete a goal by its [procedure](.agents/skills/alphalattice-research/references/goals.md)
with actual criterion answers, references, findings and unresolved work. Report
from the sealed `goal show`, copying its ids and hashes, and link its
**Timeline**, **Conversation**, **Results** and each result's `local_web_url`.
State supported claims, time, coverage and open issues; promise no profit,
independent validation or trades.
