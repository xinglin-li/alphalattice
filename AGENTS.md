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

At first use and any refusal naming a missing dependency, run `python -m alphalattice.interface.local_application.native_setup doctor`; follow each `host_readiness` item's `who_decides`, disclose locked repairs, and relay one question before an outside-lock install.

If you installed AlphaLattice in this session, continue here: read this guide and
the research Skill by the paths `configure` printed under `continue_here`, start
each specialist as a general subagent whose prompt is its card's text, and tell the
person its `disclosure` in one line ([setup](.agents/skills/alphalattice-research/references/operating.md)).
Use a new workspace for new research:

```powershell
alphalattice --workspace "workspaces/my-research" serve --no-browser --stop-on-stdin
```

Run it in the background on Claude Code. On Codex, where a turn ends its shell's
children, start it with `serve --no-browser --detach` instead: it returns once the
Host serves, and the Host outlives the turn. `alphalattice host stop` stops it after its workers
join; where your shell holds its standard input open, `stop` on it does the same,
and a background shell that gives it none leaves it serving until then. Once it
serves, its ready lines end with the next action, OPEN_THE_WORKBENCH. The rule: the
Workbench opens in the coding app's built-in browser, visibly, once, never in a separate
system browser, so keep `--no-browser` on every serve you start. In Claude Code's desktop
app, load the browser pane's tools and open the printed link with `preview_start` (url),
which shows the person a card in the chat; never `navigate` or a background tab for this
first open. In Codex, open it in Codex's built-in browser. Then tell the person in one line
that the Workbench is open there, with the link. Only a host with no built-in browser (a
terminal CLI) gives the person the link alone. Never navigate or click the Workbench to
show work. Your Session binds itself when it first works on the workspace, and
its first research request opens a goal when it holds none. A committee member's
bundle, message or review answer counts toward its floor's or thread's open goal
instead, and a committee message never opens one; once that goal has closed, a
member's request opens the Session's own goal, as a first research request does.
Usage notices belong to the selected host Session; an unverified-participant count may exceed the displayed identifiers.
Commands then omit `--workspace`. Where an agent sandbox protects the project's `.codex`
(Codex's default workspace-write), the Host binds the Session in the workspace's own
folder instead and every answer's `session_binding` says so: keep naming `--workspace`
on each command, with nothing to approve. Use the **alphalattice-research** Skill for research
([command contract](.agents/skills/alphalattice-research/references/operating.md)).

## Run the first use from the person's sentence

Start it with the person's exact sentence and the date it names for the positions
(today's when it names none), as you read it:

```powershell
alphalattice first-use prepare --sentence "Positions for 2026-10-09, reviewed." --date 2026-10-09 --output "<out>/first-use.json"
```

It first checks the setup a first use cannot run without, and opens nothing while
it is missing: the bare `alphalattice` command your specialists' routes run in their
own fresh shells, and the retrieval environment. Run each `missing` item's `commands`,
disclosing them (a person's item is asked once, in one line), then the same command
again. It then opens the workspace's one `FIRST_USE` goal, prepares the data under the
goal's delegation and follows each Task. Tell the person first how the product reads the
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
returned words: disclose its delegated steps in one line, then ask the person its one
question in its words; ask nothing else. On their yes, run each command it offers once,
their words verbatim in place of `<their words>`: a `network set` under the delegation
never answers that question. On every step whose answer carries `network_use`, read its
`words` aloud to the person in one line, as returned.

The road: data, the strategy's models and Risk, its whole-support book (the
numerical check), activation with the date's update in the same act, then Evidence
and the CRO on that date's published positions, the committee and its report.
The historical book's own review runs only when someone asks for it. A stop names
its way on: decide each data issue with its offered `confirm`, or take the refused
step under the delegation, then run the same command again. The network setting and the offline switch matter only
when an answer refuses for them. Never restart the Host or serve a second one for
network access, and never ask the person to.

Follow the wait and first-use wake procedure in [operating](.agents/skills/alphalattice-research/references/operating.md).
On Codex, a wake registration that answers `local_client.codex_unreachable` means no wake
will come: wait inside the turn with `alphalattice activity wait --task <task-id>`.

While the goal is open, for 24 hours from opening, it delegates to you: opening
the preparation's network, confirming the preparation and its resumes, deciding
its data issues, confirming its membership changes, activating its book, and
setting up Evidence with recent SEC filings within the default budget and the
retrieval model download at the size its setup states. Tell
the person each act in one line; they stop the book by telling you, or on
**Portfolio**. Accepted submission, abandonment or expiry ends the delegation and
closes the network it opened. The `FIRST_USE` declaration is never revised. So
ask once at the start, in `ask_now`'s words, whether daily updates may keep the
network on; their yes is theirs until they say stop (relay `network set --enabled
false`, which also stops SEC acquisition).

`strategy build` runs the research strategy's required whole-support Alpha and
Risk studies and installs it, reusing completed studies, and offers each installed
package's book run.
For the model lifecycle, follow [leading research](.agents/skills/alphalattice-research/references/research-lead.md).

## Choose the path by intent

For "report a problem", run `alphalattice problem report --output "<out>/report.json"`
and follow the [reporting procedure](.agents/skills/alphalattice-research/references/operating.md#report-a-problem): show the report and ask before sending it.

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

Bundle outputs accept new/empty directories or matching output from the same review;
rerun with that directory after a stop. Unrelated files are refused and never overwritten.
The first reads that update's published positions, bound by their publication,
prepares their Evidence and writes every Analyst bundle; start one Evidence Analyst
per bundle. Where no Evidence is installed it stops with the install offered ready (`install`),
its issuers named by the Host's rule: send it as it stands, once, then run the same command
again; the review acquires the rest. An install whose issuers all filed nothing offers the next
eight as `install`. Where none is offered, the answer says why and the way on.
Without `--update` it reviews the strategy's whole-support book instead, when someone asks for that.
Each Analyst, and then the CRO, submits its own answer with its bundle's key and takes the Host's corrections itself; once
they return their receipts, the second reads their accepted answers and writes the
CRO's bundle, and the third publishes the review. Each stops at the first answer
that needs another step; follow its `next_action`. Wait for each as the first use
says. Once the date's review is published, `research-update show` offers its
committee. So does your continue on an earlier bundle when it is refused as stale or
ambiguous because you prepared again; a passing refusal keeps its own way.

## Convene the committee

```powershell
alphalattice committee open --update <task>
alphalattice bundle prepare --role ALPHA --task <task> --dir "<out>/committee/alpha"
```

On a date's published positions, you, the PM, convene the investment committee
with Alpha, Risk and the CRO. `alphalattice committee open` offers each specialist's bundle: its
view of the date's positions and its `submit` and `wait` on the floor. Prepare the
three (ALPHA, RISK, CRO), then in one turn start them as general subagents (the
registered CRO card runs no command, and each bundle's route runs two), each
prompt its card's text and its bundle's path, and in that same turn submit your own stance,
argued from `alphalattice committee show --update <task>`, which lists each member's questions with the
floor's facts by their S alias: the reveal waits for your stance, and nothing of theirs shows before it. As PM you judge whether the book adds
up: what changed and what it cost, its research envelope, whether it can be carried out,
and what the committee hands on; its dates are in the CRO's process fact. The open's answer gives your PM key
(`pm_key`), and your session's later open gives it again: pass it as `--key` with
`--role PM`, and keep it out of every file a specialist reads; each specialist's
key is in its own bundle. Each specialist
moves itself: it submits its stance, waits, takes the one discussion turn its wait offers
(`TAKE_YOUR_DISCUSSION_TURN`: speak or pass) and waits again; brief it to follow its route to the end. Keep `alphalattice committee
wait --update <task> --role PM --key <key>` running in the background and rule
each challenge as it arrives (ADOPT, REJECT or FOR_THE_PERSON); the wait answers
`settled` once every specialist has spoken or passed its turn and no challenge stands
unruled, or once the floor's discussion window (`discussion_minutes` from the reveal) has passed,
and then you give the verdict at once. Inside the window the floor refuses a verdict while a challenge
stands unruled and offers its ruling to fill; past it, a silent specialist has passed (silent) and
an unruled challenge is reported as unruled. The VERDICT is the committee's discussion record,
not an approval or advice: PROCEED_WITH_NOTES by default, members' reservations carried as notes;
DECLINE endorsement of action on the positions only for a named hard defect (a data error, a Risk
bound breached, key evidence missing), naming it, and it preserves the research record and
positions; FOR_THE_PERSON only for a named decision of the person; PROCEED when nothing needs noting. The stances are blind until all four are in; after the reveal each
specialist takes one discussion turn, a CHALLENGE or REPLY or a PASS that yields it, with three
challenges or replies in all, and the floor closes at its time box
whoever is silent. Messages name holdings, points, components and messages by alias
and type no other digit. A refused message names the rule it broke and, where only its
shape or a typed holding name is wrong, offers the same message corrected: send that as
it stands. The report keeps
each member's last message and every earlier one whose position it has not changed, verbatim.
Read the agenda points before speaking: below the agenda line, no stance is required; the PM may raise one listed point through the positions of a REPLY to a revealed message.

The committee changes no number. Members may suggest avoiding, reducing or adding specific
holdings, with reasons from the materials: the report leads with those suggestions, then the
verdict, and ends with one line that also ends your reply to the person: the committee changed
no number; whether to act on its suggestions is the person's decision. Ask the person
about each item it hands them in one line, and relay their words as a
PERSON_ANSWER. Then export the report from the closed floor:

```powershell
alphalattice committee show --update <task> --output "<out>/committee/floor.json"
alphalattice request --from "<out>/committee/floor.json" --action report
```

The first use's report asks the person once (`ask_now`), in its words, whether each
daily update also convenes the committee: ask it, then relay their answer with the
command it offers. Unanswered, it is off; one sentence from them changes it later
(`committee daily --enabled true|false`, relayed).

## Run an installed strategy forward

On every fresh session read `strategy-book controls --package <package>` for
`activation`, its book and horizon before a forward plan, and show the person
the offer's `review_holdings`: the book's last sealed holdings, not next positions.
Before activating, read `strategy_dates.information_cutoff` and the conditional
`strategy_dates.first_actionable_session`. Activation admits the first update in
the same act, for the first use's date while its goal is open: follow it at once,
then review its published positions on their own publication. A reviewed date runs
through Evidence and the CRO; with the person's daily committee setting on, its
published review convenes the committee and its report, as the first use does. Off,
its next action is `GIVE_THE_PM_VIEW`: as PM, give the person your view of the
reviewed positions from the Host's materials (the positions, the date's Risk, the
CRO's review), with suggestions on holdings from the review's findings, the positions
unchanged and the person deciding, then offer the committee for that date in one
line (`committee open --update <task>`). For a date the
person names, plan `research-update plan --package <package> --date <date>`: as
in the first use, the date is the trade date its positions are entered on, decided
at the close before it, and the plan's `positions_for` names that date. With no
date it plans the latest settled close (about two hours after the close), whose
positions are the next session's, so tomorrow's positions after today's settled
close need no other request. Hold positions only
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
click. Words that say no person gave them are refused
(`person_confirmation.not_the_persons_words`): with no person to ask, leave the
decision undone and say so. Each yes answers one request, once. The daily network question's yes is the one
standing exception: it answers its two requests (network on, SEC filings at the
default budget), and under it you may send the Evidence install the Host offers,
as it stands; the Host takes nothing else for it.

- Network access, except preparation and default SEC acquisition under the first
  use's delegation.
  If an Evidence install loses `evidence_setup`, read its offered form and restore
  the exact setup from the original offer; that read grants no permission to
  change issuers or scope.
- Strategy activation, except under the first use's delegation; deactivation
  always. Model and Factor activation.
- Daily research automation, the daily committee setting and the usage-reading switch.
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
