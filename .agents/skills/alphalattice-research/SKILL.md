---
name: alphalattice-research
description: Conduct quantitative research in an AlphaLattice workspace through its CLI. Use for Factor, Alpha, Risk, Portfolio, feature, strategy, data, Evidence and CRO work; not for changing product code or live trading.
---

# AlphaLattice Research
Date: 2026-10-08

You lead the research. AlphaLattice's owners calculate, validate, record and publish; you choose the question, follow each answer's `next_action` and `next_requests`, seek judgment and explain what the evidence supports. The [research-agent guide](../../../AGENTS.md) covers setup, the first use and what only a person decides; a specialist gives judgment, never product authority. If `.alphalattice/user/skills/alphalattice-research.md` exists, read it too: the person's additions to this Skill.

## Shortest paths

Begin at the person's own target (a date's positions, a named request), else at `workspace show`'s first `intents` entry, and each answer offers the next request ([command contract](references/operating.md)). One exact read needs no goal; multi-step work runs under one ([goals](references/goals.md)).

- **First use from one sentence**: `first-use prepare --sentence` with the person's exact sentence and `--date` the date it names, then each answer's next action along its `first_use.road`: data, models and Risk, the book, activation with the date's update, the review of that date's positions ([guide](../../../AGENTS.md); [where a first use stands](references/research-lead.md)).
- **Orient**: `workspace show` → the first intent's offered request.
- **Run an installed research strategy forward**: `strategy-book controls` → activation, which admits its update → `research-update show` → `strategy-book review --update` → `review continue` ([leading research](references/research-lead.md)).
- **Evidence and CRO review of a date's positions**: `strategy-book review --update`, then `review continue` after the Analysts and again after the CRO; without `--update`, of the whole-support book when asked ([specialists](references/specialist-handoff.md)).
- **The committee on a date's positions**: `committee open --update` → `bundle prepare` for Alpha, Risk and the CRO → start them → your stance with the open's `pm_key` → `committee wait --role PM --key`, rulings and the verdict → `committee show` → its offered report ([guide](../../../AGENTS.md)).
- **A Factor study**: `study controls` → `study plan` → `study run` → `study show`.
- **An Alpha study from a Factor study**: the Factor study's offered curation, then the handoff it offers → `study plan` → `study run`.
- **A Risk study**: `study controls --kind risk.covariance-development` → `study plan` → `study run`.
- **A book**: `book draft` → `study plan` → `study run` → `study show` ([research contract](references/research-contract.md)).
- **A formula factor**: `feature controls` → its offered plan, trial and review.
- **A data issue**: `issue list` → its offered `request` ([pipeline issues](references/pipeline-issues.md)).

Before activating, read `strategy_dates.information_cutoff` and the conditional `strategy_dates.first_actionable_session`.
Activation admits the first update in the same act: follow it at once, then review its published positions on their own publication.
Hold positions only from the first actionable session; sessions before it are a causal replay, inside the research window where marked.

## Commands

Each command's exact form; the [Command contract](references/operating.md) covers answers, continuations and waits.
- `workspace show`: Inputs, recent studies and Tasks, and `intents` with their next requests.
- `first-use prepare --sentence "Positions for 2026-10-09, reviewed." --date 2026-10-09`: Opens the first use from the person's exact sentence and prepares its data; its answer lays out the whole first use.
- `strategy build`: Runs the strategy's required Alpha and Risk studies on their defaults, then prepares and installs it.
- `strategy-book controls --package <package>`: The installed strategy's activation, book, horizon and review standing.
- `strategy-book review --package <package> --dir "<out>/analysts"`: Runs or reuses the whole-support book, prepares its Evidence, writes the Analyst bundles.
- `review continue --dir "<out>/analysts" --cro-dir "<out>/cro"`: Submits the answers in the folder, follows them, then writes the CRO's bundle or, with --package, reads the review and the activation offer.
- `committee open --update <task>`: Opens the committee on a date's positions; offers each specialist's bundle.
- `bundle prepare --role ALPHA --task <task> --dir "<out>/committee/alpha"`: A specialist's bundle; on a date's update, its view of the positions and the floor.
- `committee wait --update <task> --role PM --key <key>`: Waits, as one member, for what the floor addresses to it, or its close.
- `committee show --update <task> --output "<out>/committee/floor.json"`: The floor: its stage, members, tension points, messages and, once closed, the report.
- `research-update plan --package <package> --output "<out>/research-update-plan.json"`: Plans the strategy's next sessions and offers `run`.
- `research-update run --from "<out>/research-update-plan.json" --wait`: Runs the saved plan, or resumes or reuses the same update.
- `research-update show --task <task>`: The update's published positions, their dates and claim.
- `study controls --input <input> --save-declaration "<out>/study.yaml"`: A new study's declaration to edit: Factor, or the kind --kind names.
- `study plan --input <input> --file "<out>/study.yaml" --output "<out>/plan.json"`: Plans a declaration; nothing runs.
- `study run --from "<out>/plan.json" --wait --output "<out>/run.json"`: Runs a plan.
- `study show <task> --output "<out>/study.json"`: A study, verified: `standing` first.
- `book draft --from "<out>/alpha.json" --candidate <candidate_id> --save-declaration "<out>/portfolio.yaml" --output "<out>/draft.json"`: Drafts a book from the Alpha study saved in alpha.json.
- `feature controls --binding <binding> --save-declaration "<out>/feature.yaml"`: The formula language and a feature declaration to edit.
- `issue list --output "<out>/issues.json"`: Open data cases and their preview requests.
- `request --from "<out>/issues.json" --action <preview_request> --output "<out>/preview.json"`: Previews one offered option; applies nothing.
- `task show <task>`: A Task's actual state and permitted next step.
- `activity wait --task <task>`: One wait, never a poll: returns when the Task ends, needs a decision or is deferred.

## Authority and evidence

- The guide lists what only a person decides; everything else is a default you take and state in one line ([goals](references/goals.md)).
- Product answers and sealed artifacts are the evidence: never invent an id, hash, model, field or number.
- A development result is not an installed strategy, prospective validation, recommendation or trading instruction. A comparison names no winner; unavailable return is not zero; keep adverse observations.
- Give specialists only the Host-packed projection; keep original sources, credentials, prompts and hidden reasoning out of bundles, Git and shared memory.
- This Skill and browser control grant no download, activation, configuration or destructive permission; your operations carry your provenance, never a person's.

## Further reading

- [Operating the CLI](references/operating.md), [goals](references/goals.md), [pipeline issues](references/pipeline-issues.md), [leading research](references/research-lead.md)
- [Specialists](references/specialist-handoff.md), [Evidence preparation](references/evidence-analysis-handoff.md), [CRO review](references/cro-handoff.md), [native sessions](references/native-visibility.md)
- [Research contract](references/research-contract.md) and [delivery](references/research-delivery.md)
