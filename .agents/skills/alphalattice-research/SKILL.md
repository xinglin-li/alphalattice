---
name: alphalattice-research
description: Conduct quantitative research in an AlphaLattice workspace through its CLI. Use for Factor, Alpha, Risk, Portfolio, feature, strategy, data, Evidence and CRO work; not for changing product code or live trading.
---

# AlphaLattice Research
Last checked: 2026-10-22 against 0.2.3

You lead the research. AlphaLattice's owners calculate, validate, record and publish; you choose the question, follow each answer's `next_action` and `next_requests`, seek judgment and explain what the evidence supports. The [research-agent guide](../../../AGENTS.md) covers setup, the first use and what only a person decides; a specialist gives judgment, never product authority. If `.alphalattice/user/skills/alphalattice-research.md` exists, read it too: the person's additions to this Skill.

At first use and any refusal naming a missing dependency, run the doctor from the checkout's root, `uv run python -m alphalattice.interface.local_application.native_setup doctor` (the guide's Open the workspace names the wheel leg's); fill each missing item by its `who_decides`, disclose locked repairs, and relay one question before an outside-lock install.

## Shortest paths

Begin at the person's own target (a date's positions, a named request), else at `alphalattice workspace show`'s first `intents` entry, and each answer offers the next request ([command contract](references/operating.md)). One exact read needs no goal; multi-step work runs under one ([goals](references/goals.md)).

- **First use from one sentence**: `alphalattice first-use prepare --sentence "<sentence>" --date <date>` with the person's exact sentence and `--date` the date it names, then each answer's next action along its `first_use.road`: data, models and Risk, the book, activation with the date's update, the review of that date's positions ([guide](../../../AGENTS.md); [where a first use stands](references/research-lead.md)).
- **Orient**: `alphalattice workspace show` → the first intent's offered request.
- **Run an installed research strategy forward**: `alphalattice strategy-book controls` → activation, which admits its update → `alphalattice research-update show` → `alphalattice strategy-book review --update <task> --package <package> --dir "<out>/review"` → `alphalattice review continue --dir "<out>/review"` after the Analysts and again after the CRO → its offered committee: `alphalattice committee open --update <task>` and the committee below → its report, with the person's daily committee setting on; off, your view as PM on the reviewed positions (`GIVE_THE_PM_VIEW`) with the committee offered in one line ([leading research](references/research-lead.md)).
- **Evidence and CRO review of a date's positions**: `alphalattice strategy-book review --update <task> --package <package> --dir "<out>/review"`, then `alphalattice review continue --dir "<out>/review"` after the Analysts and again after the CRO; without `--update`, of the whole-support book when asked ([specialists](references/specialist-handoff.md)).
- **The committee on a date's positions**: `alphalattice committee open --update <task>` → `alphalattice bundle prepare` for Alpha, Risk and the CRO → start them → your stance with the open's `pm_key` → `alphalattice committee wait --update <task> --role PM --key <key>`, a ruling on every challenge, then the verdict once the wait answers `settled` (every specialist has spoken or passed its turn and no challenge stands unruled, or the discussion window has passed) → `alphalattice committee show` → its offered report; a resent message answers with its first admission, and a refusal's offered way back is sent as it stands ([guide](../../../AGENTS.md), [refusals](references/committee.md)).
- **A Factor study**: `alphalattice study controls` → `alphalattice study plan` → `alphalattice study run` → `alphalattice study show`.
- **An Alpha study from a Factor study**: the Factor study's offered curation, then the handoff it offers → `alphalattice study plan` → `alphalattice study run`.
- **A Risk study**: `alphalattice study controls --kind risk.covariance-development` → `alphalattice study plan` → `alphalattice study run`.
- **A book**: `alphalattice book draft` → `alphalattice study plan` → `alphalattice study run` → `alphalattice study show` ([research contract](references/research-contract.md)).
- **A formula factor**: `alphalattice feature controls` → its offered plan, trial and review.
- **A data issue**: `alphalattice issue list` → its offered `request` ([pipeline issues](references/pipeline-issues.md)).
- **Report a problem**: `problem report --output "<out>/report.json"` → show the report → ask before sending (the “Report a problem” section of the [operating contract](references/operating.md)).

For an invalid Feature document or strategy declaration, send its offered controls or
inputs request, repair the invalid fields and plan again with the same research choices.

Before activating, read `strategy_dates.information_cutoff` and the conditional `strategy_dates.first_actionable_session`.
Activation admits the first update in the same act: follow it at once, then review its published positions on their own publication.
Hold positions only from the first actionable session; sessions before it are a causal replay, inside the research window where marked.

## Commands

Each command's exact form; the [Command contract](references/operating.md) covers answers, continuations and waits.
- `alphalattice workspace show`: Inputs, recent studies and Tasks, and `intents` with their next requests.
- `alphalattice problem report --output "<out>/report.json"`: Writes a redacted local problem report; show it and ask before sending it.
- `alphalattice first-use prepare --sentence "Positions for 2026-10-09, reviewed." --date 2026-10-09`: Opens the first use from the person's exact sentence and prepares numerical inputs through the preceding formation session's information cutoff; activation publishes the requested date's positions once that information is available, or holds until then; follow a calendar or source refusal's exact offered recovery. The declared current universe and Sector stay retrospective, with observation times recorded; historical membership is not claimed.
- `alphalattice strategy build`: Runs the strategy's required Alpha and Risk studies on their defaults, then prepares and installs it.
- `alphalattice strategy-book controls --package <package>`: The installed strategy's activation, book, horizon and review standing.
- `alphalattice strategy-book review --package <package> --dir "<out>/analysts"`: Runs or reuses the whole-support book, prepares its Evidence, writes the Analyst bundles.
- `alphalattice review continue --dir "<out>/analysts" --cro-dir "<out>/cro"`: Submits the answers in the folder, follows them, then writes the CRO's bundle or, after the CRO, offers a dated review's committee (its report once closed) and, with --package, the activation offer.
- `alphalattice committee open --update <task>`: Opens the committee on a date's positions; offers each specialist's bundle.
- `alphalattice bundle prepare --role ALPHA --task <task> --dir "<out>/committee/alpha"`: A specialist's bundle; on a date's update, its view of the positions and the floor.
- `alphalattice committee wait --update <task> --role PM --key <key>`: Waits, as one member, for what the floor addresses to it, the settled debate, or its close.
- `alphalattice committee show --update <task> --output "<out>/committee/floor.json"`: The floor: its stage, members, agenda points, below-line points, messages and, once closed, the report.
- `alphalattice research-update plan --package <package> --date <date> --output "<out>/research-update-plan.json"`: Plans the update for a date's positions, entered on that trade date and decided at the close before it; without `--date`, the next session's. Offers `run`.
- `alphalattice research-update run --from "<out>/research-update-plan.json" --wait`: Runs the saved plan, or resumes or reuses the same update.
- `alphalattice research-update show --task <task>`: The update's published positions, their dates and claim.
- `alphalattice study controls --input <input> --save-declaration "<out>/study.yaml"`: A new study's declaration to edit: Factor, or the kind --kind names.
- `alphalattice study plan --input <input> --file "<out>/study.yaml" --output "<out>/plan.json"`: Plans a declaration; nothing runs.
- `alphalattice study run --from "<out>/plan.json" --wait --output "<out>/run.json"`: Runs a plan.
- `alphalattice study show <task> --output "<out>/study.json"`: A study, verified: `standing` first.
- `alphalattice book draft --from "<out>/alpha.json" --candidate <candidate_id> --save-declaration "<out>/portfolio.yaml" --output "<out>/draft.json"`: Drafts a book from the Alpha study saved in alpha.json.
- `alphalattice feature controls --binding <binding> --save-declaration "<out>/feature.yaml"`: The formula language and a feature declaration to edit.
- `alphalattice issue list --output "<out>/issues.json"`: Open data cases and their preview requests.
- `alphalattice request --from "<out>/issues.json" --action <preview_request> --output "<out>/preview.json"`: Previews one offered option; applies nothing.
- `alphalattice task show <task>`: A Task's actual state and permitted next step.
- `alphalattice activity wait --task <task>`: One wait, never a poll: returns when the Task ends, needs a decision or is deferred.

## Authority and evidence

- The guide lists what only a person decides; everything else is a default you take and state in one line ([goals](references/goals.md)).
- Product answers and sealed artifacts are the evidence: never invent an id, hash, model, field or number.
- Where a refusal adds `error_info`, read its registered category, exact reason and safe field paths; follow the original offered request. The category grants no retry or authority.
- If an Evidence install loses `evidence_setup`, read its offered form and restore
  the exact setup from the original offer; that read grants no permission to
  change issuers or scope.
- A development result is not an installed strategy, prospective validation, recommendation or trading instruction. A comparison names no winner; unavailable return is not zero; keep adverse observations.
- Give specialists only the Host-packed projection; keep original sources, credentials, prompts and hidden reasoning out of bundles, Git and shared memory.
- This Skill and browser control grant no download, activation, configuration or destructive permission; your operations carry your provenance, never a person's.

## Further reading

- [Requests and capability routes](references/intents.md): the person's intent, admitted declaration fields, costs and unsupported choices.
- [Operating the CLI](references/operating.md), [goals](references/goals.md), [pipeline issues](references/pipeline-issues.md), [leading research](references/research-lead.md)
- [Specialists](references/specialist-handoff.md), [Evidence preparation](references/evidence-analysis-handoff.md), [CRO review](references/cro-handoff.md), [native sessions](references/native-visibility.md)
- [Research contract](references/research-contract.md) and [delivery](references/research-delivery.md)
