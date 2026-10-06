# Goals
Date: 2026-10-03

A goal is the Host's durable record of a multi-step objective. It records requests and Tasks from the bound session and checks a submission against the declaration and evidence (LAWS OP13). One exact read needs no goal. Goal UUIDs identify goals; content hashes identify revisions.

## Open a goal

Use `goal schema --save-declaration "<out>/goal.yaml"` to get the shortest valid declaration, edit it and open it with `goal open --file "<out>/goal.yaml"`. Include an objective, kind, scope, constraints, criteria, deliverables and, for research, purpose, comparison design and required stages. The schema supplies optional budget and parent-goal fields. Opening binds the session and returns a prompt for another agent; a new session uses `goal take <id>`, otherwise name `--goal <id>`.

A compact example, valid for one formula factor and review:

```yaml
title: One formula factor, tried and reviewed
objective: Add one formula factor with its hypothesis and bring it to a completed trial and a review packet whose contract passed.
kind: RESEARCH
criteria:
  - criterion_id: reviewed
    text: The factor's trial completed and its review packet's contract passed.
deliverables:
  - deliverable_id: packet
    kind: RESULT
    description: The factor's review packet.
research:
  purpose: NEW_RESEARCH
  comparison_design: The trial's Alpha study against its Factor-derived baseline on the same input.
  required_stages: [DATA_FEATURES]
```

## First use

Only a person's one sentence may open the one-per-workspace `FIRST_USE` goal, before the first preparation. It may delegate only opening network access for that preparation, confirming that preparation and its resumes, and deciding its data issues, while the goal is open and within its hours. Record these as the person's delegated acts. The person can stop it at any time; its network grant closes at the delegation's end or earlier submission/abandonment. Activation, storage, automation, revocation and paid actions remain theirs. The book draft's declared unavailable-return quarantine is allowed only under its declared policy; keep original inputs and report the effective population, never select exclusions to improve results.

## Record and submit work

The Host records every request and Task; do not log steps yourself. `goal attach` adds an exact read under DATA_FEATURES, FACTOR_FOUNDATION, ALPHA, RISK, PORTFOLIO or EVIDENCE_CRO (a feature trial/review uses FEATURE_TRIAL_READBACK/FEATURE_REVIEW under DATA_FEATURES), never a RUN, latest-input default or declared completion. `goal note` records a decision or conclusion; SUPPORTS/DOES_NOT_SUPPORT needs evidence. Revising the objective or criteria marks earlier evidence POST_HOC. For existing results declare `EXISTING_RESULTS`; do not claim preregistration.

Prefer `goal show --save-declaration "<out>/submission.yaml"`: fill the owner-provided criterion and deliverable slots and submit that whole declaration with `request --file "<out>/submission.yaml"`. The declaration is bound to that goal revision. A hand-written submission uses `goal submit <id> --file "<out>/submission.yaml"`; it names outcome, summary, each criterion as MET/NOT_MET with evidence or NOT_ASSESSED with a note, deliverable slots, references, files, findings, problems and follow-ups. Cite held evidence by its id; declare only new reads once. Name every Task still waiting on a person in `problems`. `INCOMPLETE` returns missing items and their next requests. `COMPLETE` means the record and evidence are verified, not that the objective succeeded. A NOT_ACHIEVED goal may be complete; ACHIEVED with a NOT_MET criterion is refused. Do not overstate.

## Read and continue

`goal list` finds goals. `goal show <id>` verifies the head's references and shows gaps, sessions, Tasks, request count, messages and open assignments; `--revision` reads a specific revision. `goal narrative` is unverified and cannot be cited. `goal continue --revision <hash> --task <id>` returns the original owner's draft to plan. Export an exact revision as JSON (or select HTML or YAML) with `goal export --revision <hash> --format json --output "<out>/goal.json"`.

The Alpha owner qualifies Alpha question families, not the goal: use `ALPHA_FAMILY_QUALIFICATION` with the question Task, nominated candidate ids and the `alpha.model-development` envelope. It counts every study on that question since the goal opened. Wait for running studies; cite the qualification result.

## Team record

With the bridge bound, Team messages and hooks belong to the goal held by the session. Assignment is a Team message; the assignee answers or the lead closes it with `pm_response`. Evidence Analysts and CROs run no command; the lead submits their prepared bundle. The bundle's exact reference credits its child. An unanswered assignment blocks submission. With no bridge there is no Team record. Notes mark decisions, never raw reasoning; product Task/result facts establish completion.
