---
name: alphalattice-research
description: Conduct quantitative research in an AlphaLattice workspace through its CLI. Use for Factor, Alpha, Risk, Portfolio, feature, strategy, data, Evidence and CRO work; not for changing product code or live trading.
---

# AlphaLattice Research
Date: 2026-10-07

You lead the research. AlphaLattice's owners calculate, validate, record and publish; you choose the question, declare the work, seek judgment and explain what the evidence supports. A specialist gives judgment, never product authority.

## Start and choose a path

- Use the installed `alphalattice` command from the checkout or configured agent project. Follow the [installation guide](../../../AGENTS.md) for that leg first. Name the workspace; announce a new directory under `workspaces/` for new work, and let the launcher initialize it. Do not choose existing research or a default for the person. Keep requests and exports in the workspace; relative paths resolve from your shell's directory.
- Reuse the Host for that workspace. Unless the person asks for terminal-only work, use the host's browser tools to open its exact printed launch and keep Local Web available as research progresses. The [operating guide](references/operating.md) covers launch, binding and safe shutdown. Never read or copy `runtime/local-research-connection.json`.
- Begin at `workspace show` and choose the input's `intents`. Read `standing`; a status alone does not establish completion. See the [command contract](references/operating.md) for all shared command and answer rules.
- One exact read needs no goal. Open a goal before multi-step work; the Host records its Tasks and checks its submission ([goals](references/goals.md)). Delegate a distinct question to the relevant role card ([specialists](references/specialist-handoff.md)).

## Shortest paths

- **Orient**: `workspace show` → choose the input's intent → [workspace and task flow](../../../AGENTS.md).
- **First use from one sentence**: open `FIRST_USE` before preparation → follow strategy controls for only the required whole-support Alpha and Risk studies → prepare and install → run the whole-support historical book → settle current Evidence within any declared allowance and publish its Analyst answers → read Evidence's dossier, then take that dossier answer's CRO bundle action → request person activation → run the offered forward update ([leading research](references/research-lead.md)).
- **A Factor study**: `study controls` → `study plan` → `study run` → `study show`; the full flow is in the [research-agent guide](../../../AGENTS.md).
- **An Alpha study from a Factor study**: `curation show` → `curation submit` → `handoff preview` → `study plan` → `study run`; follow the same [research-agent guide](../../../AGENTS.md).
- **A Risk study**: `study controls --kind risk.covariance-development` → `study plan` → `study run`; follow the same [research-agent guide](../../../AGENTS.md).
- **A book**: `book draft` → `study plan` → `study run` → `study show`; follow the same [research-agent guide](../../../AGENTS.md) and [research contract](references/research-contract.md).
- **The book's Evidence and CRO review**: follow the book readback's selectors, then [prepare the Analyst packet](references/evidence-analysis-handoff.md) and [submit the CRO review](references/cro-handoff.md).
- **Run an installed research strategy forward**: `workspace show` → exact package `strategy-book controls` and `activation` → person activation if `INACTIVE` → offered `research-update plan` → `research-update run` → `research-update show`; read [leading research](references/research-lead.md).
- **A formula factor**: `feature controls` → `feature plan` → `trial run` → `trial show` → `feature review`; see [feature changes](references/operating.md) and the [research-agent guide](../../../AGENTS.md).
- **A model of your own**: `model scaffold` → `model check` → `model sandbox` with the Host stopped; see the model contract in [research contract](references/research-contract.md).
- **A data issue**: `issue list` → its offered `request`; follow [pipeline issues](references/pipeline-issues.md).

Before asking a person to activate, review the completed historical book and its review standing, and read `strategy_dates.information_cutoff` and the conditional `strategy_dates.first_actionable_session`.
After activation, run the offered update and review its first published forward positions at the first actionable session.
Hold positions only from the first actionable session; sessions before it are a causal replay, inside the research window where marked.

## Commands

Follow the [Command contract](references/operating.md) for shared syntax, answers, continuations, files and waits. The catalog below gives each operation's exact form and its own flags; run it only within your scope and budget.
- `workspace show`: Input ids, recent studies and Tasks, and `intents`: each flow's needs, holdings and next requests.
- `strategy-book controls --package <package>`: Reads the exact installed package's activation, book and recorded review standing before planning Forward work; an inactive book's activation is the person's action on Portfolio, separate from installation and automatic scheduling.
  --package: The installed strategy package, by its id.
- `goal schema --save-declaration "<out>/goal.yaml"`: Writes the shortest goal declaration, valid as it stands, to edit.
- `goal open --file "<out>/goal.yaml"`: Opens a goal before multi-step work and binds this session to it.
- `goal show --save-declaration "<out>/submission.yaml"`: Writes the bound goal's completion to fill, bound to its revision: its criteria and deliverable slots, its references listed as the evidence to cite.
- `request --file "<out>/submission.yaml"`: Submits the filled completion to its goal's revision, last ([goals](references/goals.md)).
- `study controls --input <input> --save-declaration "<out>/study.yaml"`: Writes a new study's declaration to edit: Factor, or the kind --kind names (`risk.covariance-development`).
  --input: The research input, by its id (RESEARCH_INPUTS lists them).
  --kind: The study kind whose controls to show, as `study controls` lists them; Factor when omitted.
- `study plan --input <input> --file "<out>/study.yaml" --output "<out>/plan.json"`: Plans a declaration; nothing runs. A handoff, draft or book draft plans `--from <answer> --file <edited>.yaml`.
- `study run --from "<out>/plan.json" --wait --output "<out>/run.json"`: Runs a plan.
- `study show <task> --output "<out>/study.json"`: A study, verified: `standing` first, then its parts (--section).
- `study summary <task>`: A completed Alpha study's recorded model, training facts and metrics; metadata only.
- `task show <task>`: A Task's actual state and permitted next step.
- `curation show --from "<out>/run.json" --output "<out>/curation.json"`: A Factor study's curation choices; `next_templates` names what you choose.
- `curation submit --from "<out>/curation.json" --choices "<out>/choices.yaml" --output "<out>/decision.json"`: Sends `experiment_curation.choices` and `limitations_acknowledged`; the answer offers the Alpha `handoff`.
- `handoff preview --from "<out>/decision.json" --save-declaration "<out>/alpha.yaml" --output "<out>/handoff.json"`: The Alpha declaration on the curated factors, from the decision's `handoff`: fill its target and model.
- `study draft --from "<out>/run.json" --save-declaration "<out>/next.yaml" --output "<out>/draft.json"`: Continues a saved study; plan it `--from "<out>/draft.json"`.
- `book draft --from "<out>/alpha.json" --candidate <candidate_id> --save-declaration "<out>/portfolio.yaml" --output "<out>/draft.json"`: Drafts a book from the Alpha study saved in alpha.json; `portfolio.risk_task_id` with `iv1`/`iv2` or a catalog `portfolio.policy` sizes it by a Risk study on that input.
  --candidate (required): The Alpha candidate a Portfolio draft is built from.
- `risk-link add --from "<out>/book-run.json" --risk-study <risk_task>`: Attaches a Risk report to the book as evidence; weights unchanged.
- `feature controls --binding <binding> --save-declaration "<out>/feature.yaml"`: The formula language, admitted recipes and a feature declaration to edit.
- `feature plan --file "<out>/feature.yaml" --output "<out>/feature-plan.json"`: Plans one `CREATE` (its `formula`, an explicit `preprocessing_recipe`, a `reason`); offers the `trial`, the Alpha study to choose.
- `trial run --from "<out>/feature-plan.json" --task <alpha_task> --output "<out>/trial.json"`: Builds and screens the feature; reruns Alpha if screening admits it. The `feature_trial.study_not_from_factor_evidence` refusal lists eligible studies.
- `trial show --from "<out>/trial.json" --wait --output "<out>/trial-show.json"`: Follows the trial until it ends, stops or needs a decision; completed, it offers each factor's `review`.
- `feature review --from "<out>/trial-show.json" --output "<out>/review.json"`: The review packet, from the trial's `review`.
- `model scaffold --save-declaration "<out>/model.yaml"`: Writes a model's declaration to edit: the contract's fields, an installed model's values as the example.
- `model scaffold --file "<out>/model.yaml"`: Writes the model's adapter, declaration and contract test from the edited declaration.
- `model check <model>`: Runs the model's contract.
- `model sandbox <model>`: Tries the model on a copy at rest, the Host stopped, for a person's activation. Uses the latest completed Alpha model-development study that names a model, or an Alpha study declaration naming one, supplied with --file <study.yaml>.
- `issue list --output "<out>/issues.json"`: Open data cases and their preview requests.
- `request --from "<out>/issues.json" --action <preview_request> --output "<out>/preview.json"`: Previews one offered option; applies nothing.
- `data-update plan --output "<out>/update-plan.json"`: Plans a data update without fetching. An update queued, running, deferred or awaiting recovery returns its own plan; run follows or resumes it.
- `data-update run --from "<out>/update-plan.json" --wait`: Runs the plan saved, as the workspace's network access allows; a provider's limit defers it with a retry time, and one stopped on the network is resumed by this same run once a person allows it.
- `research-update plan --package <package> --output "<out>/research-update-plan.json"`: Plans a strategy's next sessions and offers `run`. An update queued, running, deferred or awaiting recovery returns its own plan to follow or resume. If workspace data is not ready, refuses with `next_requests` to settle it.
  --package (required): The installed strategy package, by its id.
- `research-update run --from "<out>/research-update-plan.json" --wait`: Runs the plan saved, reusing an identical update already made or in flight; one stopped on the network resumes from where it stopped when this run is sent again after a person allows it, and one the provider deferred once its `retry_after_at` has passed, which its read offers as `resume`.
- `research-update show --task <task>`: The update's published positions, their dates and claim, and its review's requests; --package, in its place, reads that strategy's own latest update, never another's.
- `activity wait --task <task>`: Waits, without polling, until the Task ends, needs a decision, is deferred, reports an incident or reaches --max-wait; --goal, in its place, also wakes on the goal's messages and its closing; --each-stage only when a verified stage lets you act before the Task ends ([waits](references/operating.md)).
  --task: The Task to wait for.
  --goal: The goal whose next Task end to wait for.
  --each-stage: With --task: also return as the Task verifies each stage (STAGE_VERIFIED), a coverage run's unit among them.
- `answer show --file "<out>/answer.json" --list-sections`: Names the parts of an answer saved by --output, offline and without a Host; --section, in its place, reads one part whole. A historical snapshot, never reverified, whose saved requests are never sent.
  --list-sections: List the saved answer's root paths.

## Authority and evidence

- A person authorizes workspace/input preparation, sets network access, confirms data or storage changes, controls the daily research automation, and activates or deactivates a strategy, model or feature. Only the narrow first-use delegation in [goals](references/goals.md) delegates its named preparation steps; it authorizes no other person-only act.
- Product answers and sealed artifacts are the evidence. Never invent an id, hash, model, field or number; preserve exported bytes and hashes.
- Keep a development result separate from an installed strategy, prospective validation, current recommendation and trading instruction. A comparison names no winner; unavailable return is not zero. State the evidence limits and keep adverse observations.
- Count D5-protected material before any bundle. Keep research and preparation offline unless the workspace has the person's authority; give specialists only the Host-packed, role-appropriate projection, never original source material, dirty data, credentials, prompts or hidden reasoning.
- This Skill grants no data download, model call, activation, global configuration or destructive permission. Keep the CRO's assessment distinct from your response.
- Browser control grants no authority. Your operations carry your provenance, never a person's.
- Keep raw prompts, model dialogue, private datasets, credentials and hidden reasoning out of Git and shared memory.

## Further reading

- [Field notes](references/field-notes.md): task entry points and recovery from observed research flows.
- [Research contract](references/research-contract.md) and [delivery](references/research-delivery.md)
- [Goals](references/goals.md), [pipeline issues](references/pipeline-issues.md), [leading research](references/research-lead.md)
- [Specialists](references/specialist-handoff.md), [Evidence preparation](references/evidence-analysis-handoff.md), [CRO review](references/cro-handoff.md)
- [Operating the CLI](references/operating.md), [native observation](references/native-visibility.md)
- [Research-agent guide](../../../AGENTS.md): installation, CLI and YAML commands, and first-use steps.
