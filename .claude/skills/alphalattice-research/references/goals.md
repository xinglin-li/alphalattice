# Goals
Date: 2026-10-07

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

Only a person's one sentence may open the one-per-workspace `FIRST_USE` goal, before the first preparation. It may delegate only opening network access for that preparation, confirming that preparation and its resumes, deciding its scoped data issues, confirming its membership changes and activating its book once that book's review standing is `REVIEWED`, while the goal is open within 24 hours of its original opening. Tell the person each delegated act in one line; they deactivate the book in one click on Portfolio. Record these as the person's delegated acts. Its declaration is immutable; evidence attachments, notes and an accepted submission may create record revisions, retaining that opening time and never restarting delegation. The person can stop it at any time; its network grant closes at the delegation's end or earlier accepted submission/abandonment. Deactivation, model and Feature activation, storage, automation, revocation and paid actions remain theirs. The book draft's declared unavailable-return quarantine is allowed only under its declared policy; keep original inputs and report the effective population, never select exclusions to improve results.

## Record and submit work

The Host records every request and Task; do not log steps yourself. `goal attach` adds an exact read under DATA_FEATURES, FACTOR_FOUNDATION, ALPHA, RISK, PORTFOLIO or EVIDENCE_CRO (a feature trial/review uses FEATURE_TRIAL_READBACK/FEATURE_REVIEW under DATA_FEATURES), never a RUN, latest-input default or declared completion. `goal note` records a decision or conclusion; SUPPORTS/DOES_NOT_SUPPORT needs evidence. For other goals whose declarations can be revised, changing the objective or criteria marks earlier evidence POST_HOC. For existing results declare `EXISTING_RESULTS`; do not claim preregistration.

Prefer `goal show --save-declaration "<out>/submission.yaml"`: fill the owner-provided criterion and deliverable slots and submit that whole declaration with `request --file "<out>/submission.yaml"`. The declaration is bound to that goal revision. A hand-written submission uses `goal submit <id> --file "<out>/submission.yaml"`; it names outcome, summary, each criterion as MET/NOT_MET with evidence or NOT_ASSESSED with a note, deliverable slots, references, files, findings, problems and follow-ups. Cite held evidence by its id; declare only new reads once. Name every Task still waiting on a person in `problems`. `INCOMPLETE` returns missing items and their next requests. `COMPLETE` means the record and evidence are verified, not that the objective succeeded. A NOT_ACHIEVED goal may be complete; ACHIEVED with a NOT_MET criterion is refused. Do not overstate.

## Read and continue

`goal list` finds goals. `goal show <id>` verifies the head's references and shows gaps, sessions, Tasks, request count, Conversation and open assignments; `--revision` reads a specific revision. `goal narrative` is unverified and cannot be cited. `goal continue --revision <hash> --task <id>` returns the original owner's draft to plan. Export an exact revision as JSON (or select HTML or YAML) with `goal export --revision <hash> --format json --output "<out>/goal.json"`.

The Alpha owner qualifies Alpha question families, not the goal: use `ALPHA_FAMILY_QUALIFICATION` with the question Task, nominated candidate ids and the `alpha.model-development` envelope. It counts every study on that question since the goal opened. Wait for running studies; cite the qualification result.

## Team record

With the Session bound, Team and the Goal's Conversation show what the product recorded of its work: requests, prepared bundles, submitted and accepted answers, and usage. There is no message command; `goal note` records a decision. Accepted answers are filed as the lead's, their author `NOT_OBSERVED`. A prepared bundle with no accepted answer is an open assignment: `goal show` and the completion answer list it, and it never blocks submission. Evidence Analysts and CROs remain commandless; the lead submits their nominated answers. See [native sessions](native-visibility.md). Product Task/result facts establish completion; notes record decisions, never hidden reasoning.
