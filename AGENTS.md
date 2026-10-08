# AlphaLattice: guide for your research agent
Date: 2026-10-08

AlphaLattice computes and records local quantitative research. You state the
question, follow the product's answers and explain the evidence to the person.

## Change the checkout

This checkout is the person's workspace to change. Fix runtime bugs and add
strategies, models and Features: read the failure's cause, fix its source owner,
add a regression test and run again. The [failure and recovery
procedure](.agents/skills/alphalattice-research/references/operating.md#failure-and-recovery)
tells a defect from a refusal and returns you to the original Task. Under `src/alphalattice/`, strategies are
declared in `investment/portfolio_strategy_lab/policies/installed_strategies.py`,
models in `capabilities/alpha_modeling/extensions/`, and Feature kernels in
`foundation/feature_engine/producers/factors/`. Follow
`docs/public-source/extending.md` in the editable checkout and run
the tests that answer for the change. The lead model is whatever your host runs; keep
the shipped specialists' inexpensive defaults.

A changed computation gets a new method identity and new result evidence.
Earlier results retain their recorded method and inputs; historical reads still
check their artifacts and bindings. Current reuse or replay may refuse them.
Fix an identity or evidence-binding refusal at its cause; never bypass the check
or edit a stored record to make it pass.

## Open the workspace

If you installed AlphaLattice in this session, continue here: read this guide and the
research Skill by the paths `configure` printed under `continue_here`, start each
specialist as a general subagent whose prompt is its card's text, and tell the person
its `disclosure` in one line. A session opened in the checkout with its `open_session`
command loads them itself and the host enforces the cards' tool limits there.
Follow [setup](.agents/skills/alphalattice-research/references/operating.md#setup-and-launch) for the locked Windows
checkout or installed wheel, download permission, persistent PATH and native
configuration. Use a new workspace for new research. Keep its service attached
to stdin and open its exact launch link once in your browser so the person can
watch; after a restart, open the new link. Never navigate or click the Workbench to
show work; the person reads it there. A bare address grants no browser session.

Your Session is bound when it first works on a workspace, and its first research request opens a goal when it holds none; nothing needs a separate step. Run `session bind` only to rebind or to turn reading off; the command reads its native host and session id:

```powershell
alphalattice --workspace "workspaces/my-research" session bind
```

Each host/Session has its own binding. A new Session binds independently without
cleaning up or replacing earlier records. Then use `alphalattice <object> <action>`:
this Session and its admitted specialists find their workspace from any checkout
folder, and printed commands omit it. An unbound shell adds `--workspace` to each
command. To change your own Session's workspace, usage or roles, run
`alphalattice session unbind` first. Run it yourself at the end; it removes only
your own binding and keeps research and history.
A person outside an agent Session may unbind one unambiguous record; the command
refuses to choose among multiple records.
Enter `stop` on service stdin to close it after its workers join.

Use new `<out>/` files inside the selected workspace for every example below.
For answers, declarations, refusals, reference prefixes, units and waits, use the
[command contract](.agents/skills/alphalattice-research/references/operating.md#command-contract).
Read help or `schema show <object> <action>` only when an answer leaves a field
unresolved. Use the **alphalattice-research** Skill for research, not product code
changes or live trading: Codex under `.agents/skills/alphalattice-research/`,
Claude under `.claude/skills/alphalattice-research/`.
On Claude Code with Bedrock, Vertex or Foundry, set `ANTHROPIC_DEFAULT_SONNET_MODEL` to pin a Sonnet version.

## Run the first use from the person's sentence

Read `alphalattice workspace show` and use its `intents` to select the input and
flow. Before a workspace's first preparation, open its one `FIRST_USE` goal.
Save this YAML in `<out>/first-use.yaml`, replacing the objective with the
person's exact sentence and the criterion with what completes that request:

```yaml
title: First use
kind: FIRST_USE
objective: "Build me a reviewed book from public data."
criteria: [{criterion_id: book, text: "A reviewed book stands for the person."}]
```

```powershell
alphalattice goal open --file "<out>/first-use.yaml" --output "<out>/first-use-open.json"
alphalattice network set --enabled true
alphalattice network show
alphalattice preparation plan --output "<out>/preparation.json"
alphalattice preparation confirm --from "<out>/preparation.json" --wait --output "<out>/preparation-answer.json"
alphalattice cpu-budget show
```

Opening binds an identified session to the goal; otherwise pass its returned id
with `--goal`. Confirm only the plan the preview offers. While the goal is open,
for 24 hours from opening, you may open the preparation's network, confirm its
preparation and resumes, preview/decide its data issues, confirm its membership
changes and activate its book once that book's review standing is `REVIEWED`,
within that first use's scope. Tell the person each in one line; they deactivate
the book in one click on **Portfolio**. Read the decisions in `goal show`'s
`record.delegated_steps`.
The `FIRST_USE` declaration is never revised. Evidence attachments, notes and
an accepted submission may create record revisions; none restarts its original
24-hour delegation window. Accepted submission, abandonment or expiry ends delegation
and closes the network it opened. A normal research goal grants none of it.

The person may interrupt with `goal abandon <goal-id> --reason <reason>` or close
network access in **Settings**. Outside first use, ask for network access there
and preparation confirmation on **Home**. Activation, storage, automation,
revocation, paid actions and external publication remain the person's.
`ALPHALATTICE_NETWORK_DISABLED=1` keeps the process offline despite its workspace
setting; restart only an idle service you own when online acquisition is
authorized. Follow the admitted Task and its continuation after a lost connection.

From preparation on, follow each answer's `next_action` and offered
`next_requests`. They lead through the research strategy's controls and its
required whole-support Alpha and Risk studies, its installation, its whole-support
book, that book's Evidence and CRO review, and its activation offer. Reuse
completed required studies; add an exploratory study only when the person's
question calls for it.

## Keep the goal

Keep first-use research under that goal. Before later multi-step research,
declare its objective, criteria, deliverables, scope and bounds:

```powershell
alphalattice goal schema --save-declaration "<out>/goal.yaml"
alphalattice goal open --file "<out>/goal.yaml" --output "<out>/goal-open.json"
```

Edit the schema's declaration for the question before opening it. Another
identified session continues it with `goal take <goal-id>`. Read `goal show` for
recorded requests and actual Task states, including stopped and failed work;
do not replace that ledger with your command log. Use the
[goal procedure](.agents/skills/alphalattice-research/references/goals.md) for
assignments, waits and checked completion.

## Explore in the Lab

An exploratory Factor, Alpha or Risk study and a Lab book follow the input's
`intents` in `workspace show`; each answer offers its next request (`study
controls`, `study plan`, `study run`, `study show`, `curation show`, `curation
submit`, `handoff preview`, `book draft`). Every placeholder is a selection or a
returned reference. Declare the person's Factor hypothesis and bounds before
planning, and read Factor evidence before filling curation choices and their
rationale. The handoff's target and model are the person's choice, as its answer
says. For risk sizing, set `portfolio.risk_task_id` to a completed compatible Risk
study; equal weights read no Risk study, and `risk-link add` adds report evidence
without changing weights. A Lab book is research only and is never activated.

## Review the book

For an installed strategy, one call takes its whole-support book to its Analyst bundles:
`alphalattice strategy-book review --package <package> --dir "<out>/analysts"`
runs or reuses the book, follows it, prepares its Evidence and writes every Analyst bundle,
answering with each bundle's answer path and submit command. It stops at the first answer
that needs another step and names it. Run it as a wait: in the background, or on Codex as its
own process with `--notify codex-queue --output <file>`, ending your turn. The commands below
are its steps, for when you choose otherwise.

Carry the book readback's Evidence/CRO selectors, never the default book. Admit
the required source access and model setup before preparation; if either is
missing, follow the Skill's [source setup](.agents/skills/alphalattice-research/references/evidence-analysis-handoff.md)
with the declared input, source scope and network consent. Data preparation
alone supplies no issuer package.

```powershell
alphalattice evidence preview --from "<out>/book-study.json" --output "<out>/evidence-preview.json"
alphalattice evidence run --from "<out>/evidence-preview.json" --output "<out>/evidence-run.json"
alphalattice activity wait --task <evidence-task-id> --each-stage
alphalattice evidence show --from "<out>/book-study.json" --output "<out>/evidence-answer.json"
alphalattice request --from "<out>/evidence-answer.json" --action <analyst-bundle-name> --choices "<out>/bundle-choices.yaml" --output "<out>/bundle-answer.json"
alphalattice bundle submit --dir "<out>/analyst-bundle" --file "<out>/analyst-bundle/answer.json" --wait --output "<out>/analyst-receipt.json"
```

Before CRO, continue current Evidence only under an explicitly declared cumulative
allowance with positive sessions and windows remaining. Follow its exact packet
and continuation requests, analyze and publish each successor packet, then reread
Evidence. A null first-reading allowance grants no continuation authority; never
invent limits. An absent or exhausted allowance or `NOTHING_RESUMABLE` preserves
unread ranges and limits for bounded review. `COMPLETE` names only the sealed
reading plan. Read current Evidence's exact `dossier` action, then that dossier
answer's `cro_bundle`; fill `bundle_directory` in `<out>/cro-choices.yaml` with a
new folder:

```powershell
alphalattice evidence show --from "<out>/book-study.json" --output "<out>/evidence-current.json"
alphalattice request --from "<out>/evidence-current.json" --action dossier --output "<out>/cro-dossier-answer.json"
alphalattice request --from "<out>/cro-dossier-answer.json" --action cro_bundle --choices "<out>/cro-choices.yaml" --output "<out>/cro-bundle-answer.json"
alphalattice bundle submit --dir "<out>/cro-bundle" --file "<out>/cro-bundle/answer.json" --wait --output "<out>/cro-receipt.json"
alphalattice evidence show --from "<out>/book-study.json" --output "<out>/book-review.json"
```

After a prepared Evidence unit, fill the offered Analyst directory choice with
a new workspace folder. Load each shipped Analyst/CRO card and give it only the
prepared bundle, file list and answer path. You run the returned `submit_command`;
keep its receipt and await publication before reporting findings. A correction
asks the same specialist to fix named items, never to change judgment for approval.

The other five specialists use the same return path. As the lead, run
`bundle prepare --role <role> --task <task-id> --dir "<out>/bundle"`
for one retained Task. Give the loaded card its stage assignment plus the bundle's
listed files and nominated answer path. Its existing ANALYZE/REVIEW stdout reads and
authorized EXECUTE operations and `<out>` writes remain unchanged. README.md gives
the steps and answer format; the material file lists "Exact references allowed in
the answer". As its last action, the child writes bounded `text`, `references`
copied from that list and its `read` list into that answer file, then returns one
line: written. This nominated write is the sole exception to ANALYZE/REVIEW's
write-nothing rule. The child never runs AGENT_ANSWER_SUBMIT; you submit the file
with the preparation's returned `submit_command`. The Host checks shape and
bindings, never scientific correctness.
A prepared bundle with no accepted answer stays an open assignment in `goal show`;
it reminds and never blocks submission. Accepted answers are filed as yours.

## Run an installed strategy forward

Installation only enables historical replay. Run and review the strategy's book
over its entire supported historical interval before requesting activation of
the completed Task. Activation binds component models, required calibration and
the last sealed book state; it fits nothing. Read
`strategy-book controls --package '<strategy-package-id>'` for `activation`, its
book and horizon on every fresh session before a forward plan. If `INACTIVE`,
take the offered activation under an open first-use goal once its review standing
is `REVIEWED`; otherwise ask the person to take the exact offered Portfolio
activation action. Follow its held reason when none is offered.
Before activating, or asking a person to activate, review the completed historical book and its review standing, and read `strategy_dates.information_cutoff` and the conditional `strategy_dates.first_actionable_session`.
After activation, run the offered update and review its first published forward positions at the first actionable session.
Hold positions only from the first actionable session; sessions before it are a causal replay, inside the research window where marked.
Before activation, show the person the offer's `review_holdings`: the reviewed
book's last sealed holdings and the sessions they were decided and entered, never
the next positions. Activation is reversible and deactivation keeps history; the
first forward update after it publishes the first-day positions. Run, build or
compute nothing to preview them. Report each position with its basis, a close-marked
conditional estimate or an entry observed at the next session's open, and the
publication's `claim`; [reading the dates](.agents/skills/alphalattice-research/references/research-lead.md#reading-the-dates)
works one activation through every timestamp.
After activation, updates advance observed
sessions and publish research positions, not orders or investment advice. The
horizon is about eleven months beyond the latest completed session at activation;
read its end and request a newer book before continuing beyond it. A changed
package needs a new run, review and activation. Deactivation keeps history.
Installation, activation and enabling daily automation on Settings are separate
decisions; first-use delegation covers only its reviewed book's activation, never
automation.

## Delegate and ask for decisions

Load the shipped Data, Factor, Alpha, Risk or Portfolio card for its professional
question; declarations in `.codex/agents/` or `.claude/agents/` do not prove it
ran. Give an executor the exact workspace, permitted operations, result
references, goal and budget; it takes that goal in its own session. Use the
Skill's [native visibility](.agents/skills/alphalattice-research/references/native-visibility.md)
for configuration, binding and permitted usage. Configuration, binding, Team observations
and a subagent stop grant no authority and prove no Task success.

Everything else is a default you take: say in one line what you chose and how to
change it, and go on. Advice in an answer, such as an incomplete Analyst review, a
CRO's request for a person's review or a review sealed under earlier Evidence, goes
into your report and is never a stop. Execution parameters are yours too: before a
heavy Task read `cpu-budget show`, keep `auto` unless its `machine` is busy or
small, set the budget without asking and say so in one line. Never change a
study's model, window or bounds to save time. A wait is one call, never a poll: run
it in the background, or on Codex with `--notify codex-queue` ([waits](.agents/skills/alphalattice-research/references/operating.md#waits-and-return-visits)).

Name each person-only decision, scope and page; wait for their answer. Factor
activation is on **Features**, model activation on **Models**; neither a trial
nor a first-use goal activates either. Read the review's activation standing and
reasons first. Strategy activate/deactivate is on **Portfolio**: name the exact
book Task or package; outside the first-use delegation your CLI request is
refused. Daily automation is on
**Settings** for named packages. External sharing/publication needs permission;
point to **Report & delivery** and **Review**. Internal publication grants no
external sharing or use.

## Interrupt and report

On interruption, stop admitting work and name running Tasks. If asked to stop
one, use `task cancel <task-id>`, then `task show <task-id>` to confirm its state.
Closing the browser does not cancel research. After a timeout or lost connection,
reopen the known Task rather than resubmit its run.

Follow the [goal completion procedure](.agents/skills/alphalattice-research/references/goals.md)
with actual criterion answers, references, findings and unresolved work. Supply
every returned gap: evidence, deliverables, unresolved Tasks or assignments.
Completion checks the record, not your summary's truth or whether the objective
was met. Deliberately stopped work uses `goal abandon <goal-id> --reason <reason>`.

Report from the sealed `goal show`, copying its ids and hashes so repaired
references remain repaired. Link its **Timeline**, **Conversation**, **Results**
and each result's `local_web_url` or `navigation.url`. State supported claims,
time, coverage and open issues; promise no profit, independent validation or trades.
