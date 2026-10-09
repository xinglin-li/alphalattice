# Goals
Date: 2026-10-08

A goal is the Host's record of a multi-step objective: it records the bound session's requests and Tasks and checks a submission against the declaration and evidence. One exact read needs no goal. The first use's `FIRST_USE` goal and its delegation are in the [guide](../../../../AGENTS.md).

## Open a goal

`goal schema --save-declaration "<out>/goal.yaml"` writes the shortest valid declaration; edit it and open it with `goal open --file "<out>/goal.yaml"`. Give an objective, kind, scope, criteria and deliverables and, for research, purpose, comparison design and required stages. Opening binds the session; another session continues it with `goal take <id>`. For example:

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

## Record and submit

The Host records every request and Task; do not log steps yourself. `goal attach` adds an exact read to a stage, never a run or a latest-input default; `goal note` records a decision, and SUPPORTS or DOES_NOT_SUPPORT needs evidence. Revising a goal's objective or criteria marks earlier evidence `POST_HOC`; existing results are declared `EXISTING_RESULTS`, never preregistered.

To complete, write `goal show --save-declaration "<out>/submission.yaml"`, fill its criterion and deliverable slots and submit it with `request --file "<out>/submission.yaml"`. Answer each criterion MET or NOT_MET with evidence, or NOT_ASSESSED with a note, and name every Task still waiting on a person under `problems`. `INCOMPLETE` returns what is missing and its next requests. Four words answer four questions: the goal's state, `COMPLETE` once its submission is accepted, verifies the record and its evidence; `submission.outcome` (`ACHIEVED`, `PARTLY_ACHIEVED`, `NOT_ACHIEVED`) says how far the objective was achieved; each criterion's answer (`MET`, `NOT_MET`, `NOT_ASSESSED`) says what its evidence shows; the case's outcome stays `QUESTION_OPEN` until a conclusion is attributed. A `PARTLY_ACHIEVED` goal may be `COMPLETE`. Deliberately stopped work uses `goal abandon <goal-id> --reason <reason>`.

## Read and continue

`goal show <id>` verifies the head's references and lists gaps, sessions, Tasks, the Conversation and open assignments. `goal narrative` is unverified and cannot be cited. `goal export --revision <hash> --format json --output "<out>/goal.json"` exports an exact revision. Alpha question families are qualified by `ALPHA_FAMILY_QUALIFICATION` over every study on that question since the goal opened; wait for running studies and cite the result.
