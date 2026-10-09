# AlphaLattice: guide for your research agent
Date: 2026-10-08

AlphaLattice computes and records local quantitative research. You state the
question, follow the product's answers and explain the evidence to the person.
The answers lead: each one's `next_action` and `next_requests` decide your next
step, and a nested part's own hint describes only that part.

## Change the checkout

This checkout is the person's workspace to change. Fix runtime bugs and add
strategies, models and Features: read the failure's cause, fix its source owner,
add a regression test and run again ([failure and
recovery](.agents/skills/alphalattice-research/references/operating.md#failure-and-recovery)).
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
person its `disclosure` in one line ([setup](.agents/skills/alphalattice-research/references/operating.md#setup-and-launch)).
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
([command contract](.agents/skills/alphalattice-research/references/operating.md#command-contract)).

## Run the first use from the person's sentence

Start it with the person's exact sentence:

```powershell
alphalattice first-use prepare --sentence "Build me a reviewed book from public data." --output "<out>/first-use.json"
```

It opens the workspace's one `FIRST_USE` goal, prepares the data under the goal's
delegation and follows each Task. Its answer's `first_use.road` is the whole first
use, one command per step, and each answer names its next action. Its
`ask_now` is what only the person decides that the first use will need: ask for it
at once, in one line. A stop names its way on: decide each data issue with its
offered `confirm`, or take the refused step under the delegation, then run the same
command again. The network setting and the offline switch matter only when an
answer refuses for them. Never restart the Host or serve a second one for network
access, and never ask the person to.

On Claude Code, run every `--wait`, `activity wait` and agent verb (`first-use
prepare`, `strategy build`, `strategy-book review`, `review continue`) with the
Bash tool's `run_in_background`, then act on its completion notice; never read its
output or check its Task before the notice. On Codex, add `--notify codex-queue`:
at its first running Task the command registers the Host's wake and returns, and
you end your turn; the wake names the command to run again, which reuses what is
done and goes on.

While the goal is open, for 24 hours from opening, it delegates to you: opening
the preparation's network, confirming the preparation and its resumes, deciding
its data issues, confirming its membership changes, and activating its book once
the book's review standing is `REVIEWED`. Tell the person each act in one line;
they deactivate the book in one click on **Portfolio**. Accepted submission,
abandonment or expiry ends the delegation and closes the network it opened. The
`FIRST_USE` declaration is never revised.

`strategy build` then runs the research strategy's required whole-support Alpha
and Risk studies and installs it, reusing completed studies; the book, its
Evidence and CRO review and its activation offer follow on the road.
Training prepares the light lifecycle by default, one seed of each model
vintage, as its answers' `model_lifecycle` says; tell the person so. When they
ask for the full one, plan it by name with `training plan --input <input-id>
--component <component-id> --lifecycle FULL`, then `study controls --input
<input-id> --component <component-id> --lifecycle FULL`.

## Choose the path by intent

Read `workspace show` and take its first `intents` entry. A prepared workspace or
an installed strategy goes to `strategy-book controls --package <package>`; open a
Factor study, an Alpha handoff or a Lab book only when the person asks for
exploration. Where a step offers a default (the model, the Risk window, the CPU
budget), take it and say in one line what you chose and how to change it; never
change a study's model, window or bounds to save time.

## Review the book

```powershell
alphalattice strategy-book review --package <package> --dir "<out>/analysts"
alphalattice review continue --dir "<out>/analysts" --cro-dir "<out>/cro"
alphalattice review continue --dir "<out>/cro" --package <package>
```

The first runs or reuses the whole-support book and writes every Analyst bundle;
start one Evidence Analyst per bundle. The second submits their answers and
writes the CRO's bundle; the third publishes the review and reads the activation
offer. Each stops at the first answer that needs another step; follow its
`next_action`. Wait for each as the first use says.

## Run an installed strategy forward

On every fresh session read `strategy-book controls --package <package>` for
`activation`, its book and horizon before a forward plan, and show the person
the offer's `review_holdings`: the book's last sealed holdings, not next positions.
Before activating, or asking a person to activate, review the completed historical book and its review standing, and read `strategy_dates.information_cutoff` and the conditional `strategy_dates.first_actionable_session`.
After activation, run the offered update and review its first published forward positions at the first actionable session.
Hold positions only from the first actionable session; sessions before it are a causal replay, inside the research window where marked.
Report each position with its basis, dates and the publication's `claim`
([reading the dates](.agents/skills/alphalattice-research/references/research-lead.md#reading-the-dates)):
research positions, never orders or advice. Past the horizon, about eleven
months, request a newer book.

## Delegate and ask for decisions

### What only a person decides

Everything else is a default you take and disclose in one line. These stay the
person's; name the decision and its page, and wait:

- Network access (**Settings**), except the first use's preparation.
- Strategy activation (**Portfolio**), except the first use's reviewed book;
  deactivation always. Model activation (**Models**) and Factor activation
  (**Features**).
- Daily research automation (**Settings**) and the usage-reading switch.
- Storage cleanup, pins and the cap; data-decision grants and their revocation.
- Preparation, data issues and membership changes outside the first use's
  delegation, and a new research-input version (**Research inputs**): keep the
  current input and tell the person newer data exists.
- Source consent: official SEC acquisition, its contact and retrieval model
  downloads; a dependency outside the lock.
- Scope: a request authorizes its stated scope only. Updating source data,
  sealing a new input or starting a different experiment is a new decision.
- External sharing or publication (**Report & delivery**, **Review**) and any
  paid action. Isolation the host cannot enforce holds the run.

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
