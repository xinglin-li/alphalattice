# AlphaLattice: guide for your research agent
Last checked: 2026-10-22 against 0.2.3

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
names before acting on it. A sealing goal answer offers `RECORD_THE_EXPERIENCE`:
follow it through the alphalattice-maintenance Skill, using its Goal and Task reads.
The Host supplies the trigger; the agent summarizes the experience. Record the person's
correction in their own words at once, and back the layer up before upgrading the checkout.

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

For an invalid Feature document or strategy declaration, send its offered controls or
inputs request, repair the invalid fields and plan again with the same research choices.

## Open the workspace

At first use and any refusal naming a missing dependency, run the doctor from the checkout's root, after `uv sync --locked --all-extras`: `uv run python -m alphalattice.interface.local_application.native_setup doctor` (on the wheel leg, with the tool environment's interpreter beneath `uv tool dir`). Fill each missing `host_readiness` item by its `who_decides`, disclose locked repairs, and relay one question before an outside-lock install.

If you installed AlphaLattice in this session, continue here: read this guide and
the research Skill by the paths `configure` printed under `continue_here`, start
each specialist as a general subagent whose prompt is its card's text, and tell the
person its `disclosure` in one line ([setup](.agents/skills/alphalattice-research/references/operating.md)).
In released 0.2.3, a Claude Code startup refusal can ask the person to run its
offered `terminal_command` once in a terminal they own. Wait for their word before
continuing; this is separate from the network-permission question.

The next terminal-panel capability is a later-checkout preview; use it only when
that checkout advertises the capability and offers the named route.
For a new Host on Claude Code desktop, use its available `run_in_terminal` tool to
probe the Terminal panel shell as the operating guide describes; only after a successful outside-every-Job result, declare
`ALPHALATTICE_TERMINAL_PANEL_HOST=claude-code` for this serve invocation, then remove it.
Codex's panel capability is undeclared; never assume a panel from a host's name alone.
Use a new workspace for new research:

```powershell
alphalattice --workspace "workspaces/my-research" serve --no-browser --detach
```

Use `--detach` on both Claude Code and Codex. Never keep a long-lived serve in a background
Bash task: its time limit can stop the Host and running Tasks. STARTED requires a ready,
authenticated Host proved outside its parent Job. A readiness timeout can leave the named
process starting: follow its offered log or stop recovery before another launch.
If the refusal offers `SERVE_FROM_THE_HOST_TERMINAL_PANEL`, run
`terminal_command` yourself there through `run_in_terminal`; the app may ask the person to
approve that one line. Otherwise give
the person one copyable `terminal_command` for a terminal they own outside the agent's shell,
then wait for their word before any further command for this workspace.
An explicitly session-only Host still accepts `--stop-on-stdin`; a held update then waits
for the next session. Never bypass Job containment through a broker, service
or scheduler. `alphalattice host stop` joins running work and stops your Host.
In 0.2.3, a detached Host serves until its explicit stop; use the returned startup
and shutdown answers. Later checkouts add the following idle policy.
A detached Host stays while a held update, Task, request or enabled automation needs it;
otherwise it stops after 900 idle seconds (0 disables idle shutdown).

The marked authenticated-reuse procedure below is also a later-checkout preview.
In 0.2.3, follow only the `ALREADY_SERVING` answer actually returned by the launcher.

<!-- message: host.reuse.guide -->
A matching serving Host can be reused even when this shell cannot detach a new process. ALREADY_SERVING reuses only an authenticated Host for this workspace and instance, preserving its authority and idle policy. Activity's observer names its instance, process and start time; the Workbench shows the same Host. On an unverified or stale descriptor, read the named Host log and identify its owning process before retrying the original serve command; leave the descriptor and writer lease unchanged.

Once it serves, its
ready lines end with the next action, OPEN_THE_WORKBENCH. The rule: the
Workbench opens in the coding app's built-in browser, visibly, once, never in a separate
system browser, so keep `--no-browser` on every serve you start. In Claude Code's desktop
app, load the browser pane's tools and open the printed link with `preview_start` (url),
which shows the person a card in the chat; never `navigate` or a background tab for this
first open. In Codex, open it in Codex's built-in browser. Then tell the person in one line
that the Workbench is open there, with the link. Only a host with no built-in browser (a
terminal CLI) gives the person the link alone. Open it so at the start of every session,
first use or daily: on a Host already serving, your Session's first answer carries
`workbench`, the same next action with a launch link of its own; open that link the same
way. Never navigate or click the Workbench to show work. Your Session binds itself when it first works on the workspace, and
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

Version 0.2.3 accepts `compact` (the default) and `full`. Save the complete owner
answer with `--output` when you will continue from it, cite it or hand it on.
Goal-profile displays remain compact; their `--output` contains the full profile.

The following bounded-summary procedure is an unreleased preview for later
checkouts that advertise `summary` in their installed grammar.
On Claude Code and Codex, use `--view summary` for reads: the complete envelope stays
within 32 KiB and gives whole offers and a `full_answer` pointer. Use `--output "<out>/answer.json"`
when you need the complete owner answer; it saves the exact owner bytes, independently of
the summary. Give `--output` before reading stdin or sending a write in summary mode.
Follow the pointer for omitted data; an omission is not an empty value. Legacy `--view full`
stdout remains available with a deprecation notice; prefer the summary and saved answer.

Where a refusal carries `error_info`, its category and owner domain come from the registry,
and `reason` keeps its exact failure code. Read its catalog message and safe field paths;
unknown supplied keys appear as `<unrecognized>`. A category grants no retry or authority:
follow the unchanged offered request or ask the person where required.

A project has one first use. A new session reads the project binding and `workspace show`
first: once the first use is complete with its strategy running forward, a date's sentence is
the daily road in that same workspace. In a project whose first use is already
complete, `serve` on another new workspace refuses before it starts,
and `first-use prepare` refuses a second first use, by name (`local_client.first_use_completed`):
each names the workspace and offers its `research-update plan`. Only when the person asks for
separate new research, asked in one line, run it again with `--new-research`.

Start it in the workspace you serve, with the person's exact sentence and the date it names
for the positions (today's when it names none), as you read it:

```powershell
alphalattice --workspace "workspaces/my-research" first-use prepare --sentence "Positions for 2026-10-09, reviewed." --date 2026-10-09 --output "<out>/first-use.json"
```

It first checks the setup a first use cannot run without, and opens nothing while
it is missing: the bare `alphalattice` command your specialists' routes run in their
own fresh shells, and the retrieval environment. Run each `missing` item's `commands`,
disclosing them (a person's item is asked once, in one line). Download and install
`config/requirements-retrieval.lock` packages as your act, without repeating first-use
network consent, even with an empty cache; say `disclosure` in one line and ask no
second question. Software outside the lock still needs the person's permission. Run the same command
again. It then opens the workspace's one `FIRST_USE` goal, prepares the data under the
goal's delegation and follows each Task. The goal is this session's; the same sentence and
date reuse it from another session, and a shell with no agent session puts
`--goal <goal_id>` before each later step's command. Tell the person first how the product reads the
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
`words` aloud to the person in one line, as returned. Under a relayed yes, its `yes` (and
`network show`'s `set_by`) names the person's words, the question, when and what it covers:
the daily yes stands until they say stop, so name it when asked, and ask again only if they
want to change it.

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
three (ALPHA, RISK, CRO), then in one turn start them as general subagents (each
bundle's route runs two commands, which the registered CRO type's allowlist also covers), each
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
a problem with particular holdings is a note and a suggestion to avoid or reduce them, never a
DECLINE. DECLINE endorsement of action on the positions only for a book-level hard defect (a data
error affecting the book, a breached book Risk bound, or key evidence missing for the book as a
whole), naming it, and it preserves the research record and positions; FOR_THE_PERSON only for a named decision of the person; PROCEED when nothing needs noting. Stances remain blind until all four arrive or the stance window ends, and never reveal before the PM's stance; after the reveal each
specialist takes one discussion turn, a CHALLENGE or REPLY or a PASS that yields it, with three
challenges or replies in all, and the floor closes at its time box
whoever is silent. Messages name holdings, points, components, messages and facts by their
floor alias listed in `aliases` and type no other digit; a member's role is never a target,
so answer its message by M alias. A refused message names the rule it broke and the aliases
it may use, and where only its shape or a typed name is wrong, offers the same message
corrected: send that as it stands; [committee refusals](.agents/skills/alphalattice-research/references/committee.md)
gives each rule with a corrected example. Take your one discussion turn before the wait's
`discussion_closes_at`; never wait again with your turn open. A stance may give a holding its own position (H5: OBJECT)
beside the agenda points; a holding's objection leads the report as a suggestion. The report keeps
each member's last message and every earlier one whose position it has not changed, verbatim.
Read the agenda points before speaking: below the agenda line, no stance is required; the PM may raise one listed point through the positions of a REPLY to a revealed message. A HIGH Evidence negative on a below-line holding reaches the report's appendix in the CRO's words.

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
daily update also convenes the committee, and its next action, COMPLETE_THE_FIRST_USE_GOAL,
offers the goal's read beside the question: ask it, then complete the goal at once, as the
goals procedure says, sending each attach it offers. Never wait for their answer: it may come
at any later time, and until it does the setting stays off. Relay it when it comes with the
command offered; one sentence from them changes it later (`committee daily --enabled
true|false`, relayed).

## Run an installed strategy forward

A new session runs forward in the workspace its project binding names, never a new one. On
every fresh session read `strategy-book controls --package <package>` for
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
line (`committee open --update <task>`). The date's road ends with its report or your
view given to the person: that answer offers the Session's goal's read (`goal`);
complete the goal as the first use's. For a date the
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
research positions, not trade advice or orders. Past the horizon, about eleven
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
budget. For a stage role, submit its answer with the preparation's `submit_command`.
Evidence Analysts, CROs and committee members submit through their own keyed routes.
The Host checks shape and bindings, never scientific correctness. A wait is one call,
never a poll.

## Interrupt and report

On interruption, stop admitting work and name running Tasks; `task cancel
<task-id>` stops one, and `task show <task-id>` confirms it. After a timeout or
lost connection, read the known Task rather than resubmitting its run. Deliberately
stopped work uses `goal abandon <goal-id> --reason <reason>`.
Before following a Task, the client prints its id and executable `read`; if the
tool window ends, send that read. With structured `--output`, it also saves the
admission and names `output_file` before stage progress; without it, no file is saved.

Complete a goal by the versioned [procedure](.agents/skills/alphalattice-research/references/goals.md)
with actual criterion answers, references, findings and unresolved work. In 0.2.3,
fill and submit the saved declaration, follow any offered attaches and resubmit,
then report from the sealed `goal show` with its ids, hashes and result links.
Later checkouts can offer completion with bound inline references and return the
`report`, sealed ids/hashes and `workbench.launch_url` together. Where that offer
exists, fill it and submit once; use its returned artifacts without repeating
Goal reads or exports. Repair a returned `report_error` with its offered export,
without resubmission. Keep the experience-recording action and link the
**Timeline**, **Conversation**, **Results** and each result's `local_web_url`.
State supported claims, time, coverage and open issues; promise no profit,
independent validation or trades.
