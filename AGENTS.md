# AlphaLattice: guide for your research agent
Date: 2026-10-07

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
the tests that answer for the change. The person chooses the lead model; keep
the shipped specialists' inexpensive defaults.

A changed computation gets a new method identity and new result evidence.
Earlier results retain their recorded method and inputs; historical reads still
check their artifacts and bindings. Current reuse or replay may refuse them.
Fix an identity or evidence-binding refusal at its cause; never bypass the check
or edit a stored record to make it pass.

## Open the workspace

Follow [setup](.agents/skills/alphalattice-research/references/operating.md#setup-and-launch) for the locked Windows
checkout or installed wheel, download permission, persistent PATH and native
configuration. Use a new workspace for new research. Keep its service attached
to stdin and open its exact launch link in your browser so the person can watch;
after a restart, open the new link. A bare address grants no browser session.

Bind your actual Session from the checkout or configured agent project; the command reads its native host and session id:

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
preparation and resumes, and preview/decide its data issues within that first
use's scope. Read the decisions in `goal show`'s `record.delegated_steps`.
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

To reach a reviewed installed book, follow strategy controls for only its required
whole-support Alpha and Risk studies, prepare and install, run the whole-support
historical book, settle current Evidence scope and publish its Analyst answers,
then read its dossier and take that dossier answer's CRO bundle action; person
activation and the first forward update follow review. Reuse completed required studies and add an
exploratory study only when the person's question calls for it.

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

## Factor, then Alpha; Risk beside them

Every placeholder is a selection or returned reference. Declare the person's
Factor hypothesis and bounds before planning:

```powershell
alphalattice study controls --input <input-id> --save-declaration "<out>/factor.yaml"
alphalattice study plan --input <input-id> --file "<out>/factor.yaml" --output "<out>/factor-plan.json"
alphalattice study run --from "<out>/factor-plan.json" --wait --output "<out>/factor-run.json"
alphalattice study show <factor-task-id> --output "<out>/factor-study.json"
alphalattice curation show --from "<out>/factor-run.json" --output "<out>/curation.json"
alphalattice curation submit --from "<out>/curation.json" --choices "<out>/curation-choices.yaml" --output "<out>/curation-decision.json"
alphalattice handoff preview --from "<out>/curation-decision.json" --save-declaration "<out>/alpha.yaml" --output "<out>/handoff.json"
```

Read Factor evidence before filling curation choices and rationale. Fill the
handoff's Alpha target and model within the person's bounds. Risk needs the same
input and may run beside Factor and Alpha:

```powershell
alphalattice study plan --from "<out>/handoff.json" --file "<out>/alpha.yaml" --output "<out>/alpha-plan.json"
alphalattice study run --from "<out>/alpha-plan.json" --wait --output "<out>/alpha-run.json"
alphalattice study show <alpha-task-id> --output "<out>/alpha-study.json"
alphalattice study controls --input <input-id> --kind <risk-kind> --save-declaration "<out>/risk.yaml"
alphalattice study plan --input <input-id> --file "<out>/risk.yaml" --output "<out>/risk-plan.json"
alphalattice study run --from "<out>/risk-plan.json" --wait --output "<out>/risk-run.json"
alphalattice study show <risk-task-id> --output "<out>/risk-study.json"
```

## Build and review the book

Choose an Alpha candidate, draft its book and fill the Portfolio declaration:

```powershell
alphalattice book draft --from "<out>/alpha-study.json" --candidate <candidate-id> --save-declaration "<out>/portfolio.yaml" --output "<out>/book-draft.json"
alphalattice study plan --from "<out>/book-draft.json" --file "<out>/portfolio.yaml" --output "<out>/book-plan.json"
alphalattice study run --from "<out>/book-plan.json" --wait --output "<out>/book-run.json"
alphalattice study show <book-task-id> --output "<out>/book-study.json"
```

For risk sizing, set `portfolio.risk_task_id` to a completed compatible Risk study
and choose a risk-reading policy. Equal weights read no Risk study; use
`risk-link add --from "<out>/book-run.json" --risk-study <risk-task-id>` for
report evidence without changing weights. Use computational titles such as
**Rebound Return Book** and **Trend Rebound Book**, but select ids from answers.

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
ask the person to take the exact offered Portfolio activation action; follow its
held reason when none is offered.
Before asking a person to activate, review the completed historical book and its review standing, and read `strategy_dates.information_cutoff` and the conditional `strategy_dates.first_actionable_session`.
After activation, run the offered update and review its first published forward positions at the first actionable session.
Hold positions only from the first actionable session; sessions before it are a causal replay, inside the research window where marked.
If the person needs positions before activation and the controls answer offers no
pre-activation preview, report that gap and stop the activation path: run, build
or compute nothing to fill it. Report each position with its basis, a close-marked
conditional estimate or an entry observed at the next session's open, and the
publication's `claim`; [reading the dates](.agents/skills/alphalattice-research/references/research-lead.md#reading-the-dates)
works one activation through every timestamp.
After person activation, updates advance observed
sessions and publish research positions, not orders or investment advice. The
horizon is about eleven months beyond the latest completed session at activation;
read its end and request a newer book before continuing beyond it. A changed
package needs a new run, review and person activation. Deactivation keeps history.
Installation, activation and enabling daily automation on Settings are separate
decisions; first-use delegation grants no activation or automation authority.

## Delegate and ask for decisions

Load the shipped Data, Factor, Alpha, Risk or Portfolio card for its professional
question; declarations in `.codex/agents/` or `.claude/agents/` do not prove it
ran. Give an executor the exact workspace, permitted operations, result
references, goal and budget; it takes that goal in its own session. Use the
Skill's [native visibility](.agents/skills/alphalattice-research/references/native-visibility.md)
for configuration, binding and permitted usage. Configuration, binding, Team observations
and a subagent stop grant no authority and prove no Task success.

Name each person-only decision, scope and page; wait for their answer. Factor
activation is on **Features**, model activation on **Models**; neither a trial
nor a first-use goal activates either. Read the review's activation standing and
reasons first. Strategy activate/deactivate is on **Portfolio**: name the exact
book Task or package; your CLI request is refused. Daily automation is on
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
