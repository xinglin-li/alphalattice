# Goals
Last checked: 2026-11-04 against 0.2.4

A goal is the Host's record of a multi-step objective: it records the bound session's requests and Tasks and checks a submission against the declaration and evidence. One exact read needs no goal. The first use's `FIRST_USE` goal and its delegation are in the [guide](../../../../AGENTS.md).

## Open a goal

`alphalattice goal schema --save-declaration "<out>/goal.yaml"` writes the shortest valid declaration; edit it and open it with `alphalattice goal open --file "<out>/goal.yaml"`. Give an objective, kind, scope, criteria and deliverables and, for research, purpose, comparison design and required stages. Opening binds the session; another session continues it with `alphalattice goal take <goal-id>`. For example:

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

The Host records every request and Task; do not log steps yourself. `alphalattice goal attach` adds an exact read to a stage, never a run or a latest-input default; `alphalattice goal note` records a DECISION, CONCLUSION, PM_RESPONSE or HYPOTHESIS, and SUPPORTS or DOES_NOT_SUPPORT needs evidence. A HYPOTHESIS note supplies its text and attribution with disposition OPEN; the Host issues and returns a `hypothesis_id` independently of the statement's name. Give that ID to `alphalattice study run --hypothesis <id>` to record what the trial tests, and read trials with `alphalattice experience list`. Before new work, `alphalattice experience recall --query "<sentence>"` finds the experiences most like the sentence you are about to try, failures included, each with its trial's class; give `--known-at` a backtest's cutoff so it reads nothing known later. Recall needs the retrieval-store model packs: `scripts/install_retrieval_pack.py --install retrieval-store --network`. Each trial row names the Task that ran it in `task_id`; a direct run without a Task returns null. Its bounded `notes` also include goal conclusions and lessons when no trial exists; a hypothesis or method filter matches only notes that record that field. Once a trial or Task has ended, review it: `alphalattice experience record --reviews trial:<id> --verdict success|failure|inconclusive --file note.json`, the note giving its `hypothesis`, `method` and `conclusion` as sentences and its `lessons` as a list; the Host records the run's tokens, cost and seconds with it and renders the note into an experience after the reply. A trial that has ended with no note reads as awaiting review. Revising a goal's objective or criteria marks earlier evidence `POST_HOC`; existing results are declared `EXISTING_RESULTS`, never preregistered.

A goal completes when its road's last answer offers its read; a question waiting for the person, such as the first use's daily committee question, never holds it. To complete, save the offered request with `alphalattice --view summary goal show --save-declaration "<out>/submission.yaml"`. Its revision and reference reads are selected: fill the remaining judgments, then send `alphalattice --view summary request --file "<out>/submission.yaml" --output "<out>/completion.json"` once. Read the returned `report` and open its `workbench.launch_url`; the reply names the sealed Goal hash, so no separate Goal read, export or link discovery is needed. Answer each criterion MET or NOT_MET with evidence, or NOT_ASSESSED with a note, and name every Task still waiting on a person under `problems`; a stop that a later Task continued and finished accounts for itself. Cite references by the ids the goal holds, never a Task id; `goal schema`'s `grammar` gives the id pattern, a problem's fields and each stage's admitted reads. `INCOMPLETE` names what is missing and offers `next_requests.complete`, a corrected request with the current hash and inline references. Fill any remaining judgments and send that request once. The legacy attach and resubmit forms stay valid. If `report_error` is returned after sealing, the Goal is still `COMPLETE`: use its offered export to repair the report, never resubmit or rerun Tasks. Keep the returned experience-recording action.

The outcome follows the criteria: every criterion `MET` is `ACHIEVED`, every one `NOT_MET` is `NOT_ACHIEVED`, `MET` beside `NOT_MET` is `PARTLY_ACHIEVED`; with one `NOT_ASSESSED` you choose `PARTLY_ACHIEVED` or `NOT_ACHIEVED`. A refused document names each field by its path and, for a read's request, the fields it takes (`allowed_fields`). Four words answer four questions: the goal's state, `COMPLETE` once its submission is accepted, verifies the record and its evidence; `submission.outcome` (`ACHIEVED`, `PARTLY_ACHIEVED`, `NOT_ACHIEVED`) says how far the objective was achieved; each criterion's answer (`MET`, `NOT_MET`, `NOT_ASSESSED`) says what its evidence shows; the case's outcome stays `QUESTION_OPEN` until a conclusion is attributed. A `PARTLY_ACHIEVED` goal may be `COMPLETE`. Deliberately stopped work uses `alphalattice goal abandon <goal-id> --reason "<reason>"`.

## Read and continue

`alphalattice goal show <id>` verifies the head's references and lists gaps, sessions, Tasks, the Conversation and open assignments. `alphalattice goal narrative` is unverified and cannot be cited. `alphalattice goal export --revision <hash> --format json --output "<out>/goal.json"` exports an exact revision. Alpha question families are qualified by `ALPHA_FAMILY_QUALIFICATION` over every study on that question since the goal opened; wait for running studies and cite the result.
