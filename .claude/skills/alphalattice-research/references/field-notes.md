# Research field notes
Date: 2026-10-05

Read the task you are doing. These notes connect observed stops to the maintained procedures; use the current answer for its exact choices and permitted continuation.

## Start a workspace and bind its session

Choose the workspace inside the configured project and serve and bind from that project ([launch and bind](operating.md), [native setup](native-visibility.md)). Finish an earlier session scope with `session unbind` before binding another workspace.

Binding checks the configured project above the served workspace. A workspace under another checkout can return `native_bridge.project_mismatch`; follow its request to serve and bind the same project workspace.

For `native_bridge.configuration_path_invalid` in a linked worktree, read `hook_declarations_effective`: Codex can read that worktree's hooks from the main checkout. Use an independent ordinary project, configure there, and have the person review the actual definitions in `/hooks` ([native setup](native-visibility.md)). `recorded_agent.basis: NOT_OBSERVED` means native observation was not established, even when the review Task succeeds ([review receipt](cro-handoff.md)).

If the bound session's public `session unbind` returns `native_bridge.files_unavailable:PermissionError`, the host's write sandbox can block removal of local binding metadata. Use that same session's host permission flow for the authorized unbind command, then retry the public door. Hook trust and research checks remain in force ([native setup](native-visibility.md)).

## Begin the first preparation

Open the workspace's `FIRST_USE` goal before preparation and read the plan's source access ([first use](goals.md), [first-use procedure](../../../../AGENTS.md)). An available plan does not establish permission to acquire its sources.

If source access names `OPERATOR_OFFLINE_SWITCH`, `ALPHALATTICE_NETWORK_DISABLED=1` holds the process offline even when workspace network access is enabled. Once acquisition is authorized, restart only an idle Host you own without that launch switch, then read the preparation preview again ([Host operation](operating.md)).

## Continue preparation after a truth review

The stop `data.truth_review_required` needs current case decisions. Read each complete case and its offered choices through [pipeline issues](pipeline-issues.md); a preview does not apply the choice. Repeating preparation confirmation cannot settle those cases. Follow the first use's current [delegation](goals.md), or the person's confirmation outside it.

After the permitted decisions, follow the owner's preparation continuation and inspect the returned plan's source reuse and retained checkpoint. It may return a successor Task; follow the Task actually admitted and read readiness and input publication before claiming preparation complete ([continuation procedure](pipeline-issues.md), [waits and return visits](operating.md)).

## Continue from a completed Factor study

Take the published study's `curation` edge with `request --from "<out>/factor-study.json" --action curation`. `request` has no `--section` option; the saved answer holds the full owner body even when stdout is compact ([answer reading](operating.md)). Use the returned curation choices and rationale, then follow its handoff within the person's bounds ([study flow](../../../../AGENTS.md)).

Default Factor controls describe that stage. A published Factor study does not establish that the requested first-use book exists. Keep the same goal through its required downstream stages and check `goal show` and held book evidence before claiming completion ([goal completion](goals.md)).

## Complete a Goal with a new Risk study

Choose a returned action name from `workspace show --list-next`; a nested JSON path is not an `--action` name. The workspace's Risk controls can be offered as `risk`. Save their declaration, plan it and follow the admitted Task before reading its exact result ([command contract](operating.md), [study flow](../../../../AGENTS.md)). A successful default Risk study can remain development-only and `NOT_ACTIVATABLE`; report its returned standing and limits.

Attach real results through their exact read requests. An existing Factor result may read `POST_HOC`, while a new Task can read `QUESTION_RECORDED_BEFORE_TASK_ADMISSION`; neither establishes independent preregistration ([intent and evidence](goals.md)). Save the completion template after the last attach or revise. An earlier template refuses `goal.revision_conflict_read_latest`; follow its `goal show --from "<out>/refusal.json" --save-declaration "<out>/submission.yaml"`, fill the refreshed slots and submit that request file ([completion](goals.md)).

Read `state` and the caller's submitted outcome separately. `COMPLETE` verifies the record and evidence; it does not establish scientific success. A Goal's `QUESTION_OPEN` describes the absence of an attributed conclusion, even when its record is complete. `goal narrative` leaves references unverified; use `goal show` or `goal export` for verified evidence ([Goal reads](goals.md)). CLI session attribution alone establishes no native Team messages ([Team record](goals.md)).

## Open a historical book's Evidence review

An installed book's `Review evidence` entry belongs to its sealed report window end. If the entry is absent at an earlier holdings session, select that window end or open the exact sealed book from Books, then follow its [Evidence handoff](evidence-analysis-handoff.md). Keep the review's returned book scope when describing it.

If a CRO Task succeeds but its current review is absent after a dossier carrying earlier evidence changes, follow the exact historical export request or publication handle the owner offers. That read opens the sealed review; it does not establish current coverage ([review readback](cro-handoff.md)).

## Continue a daily update

An update plan with `OBSERVATIONS_PENDING` and `WAIT_FOR_NEXT_COMPLETED_SESSION` waits for a completed session. Revisit its current plan after that condition is met and follow only an offered run request ([forward updates](../../../../AGENTS.md), [waits](operating.md)).

An older `PROPOSAL_PUBLISHED` update is existing evidence, even while the fresh plan waits. A schedule's next due time is not a completed session or a new execution. A person's Settings schedule switch grants no source or fit authority ([leading research](research-lead.md)).

Data maintenance has its own plan and can be available while the strategy waits. If it stops at `workspace_maintenance.derived_manifest_evidence_incomplete`, its derived membership lacks current audit evidence. Read the Task's current recovery and follow its offered replan within scope; do not replace that stop with a provider deferral or claim a new publication ([continuations](operating.md)).

If admission instead refuses `workspace_data_update.panel_binding_mismatch`, follow its current data-update read request and inspect the input state and `current_input_failure`. That read does not repair the binding or grant authority. If it reports no input failure while RUN still refuses, report the differing plan/read/admission answers rather than repeating the same run ([answer reading](operating.md)).

If a data-update run refuses `workspace_data_update.source_access_not_admitted` under `OPERATOR_OFFLINE_SWITCH`, read its offered network-access request. Workspace settings cannot lift the process switch. Restarting without it needs source-acquisition authority; scheduling alone supplies none ([Host operation](operating.md)). Claim a provider deferral or continuation only when an actual Task returns that state; wait for its retry time and follow its offered resume ([waits](operating.md)).

## Resume interrupted or capacity-blocked work

After restarting a Host you own, read the known Task first: interrupted work can already have resumed and completed ([waits and return visits](operating.md)). A stale recovery confirmation requires the current view, not another execution.

For a research Task stopped on `storage.managed_capacity_exceeded`, an authorized cap increase or cleanup permits a fresh `recovery show <task>` and its offered RECOVER. Follow `recovery run --from "<out>/recovery.json" --wait`; the owner rechecks the plan, input and capacity before reopening that same Task. An insufficient cap or changed input still refuses. Use only the action this Task offers ([continuations](operating.md)).

## Restore held state and missing inputs

Read `workspace show` when study controls cannot read an input. Restore the exact missing binding bytes when available; a newly prepared binding cannot stand in for old evidence ([answer reading](operating.md)).

For an authorized backup restore, choose an unused destination and retain the returned workspace id and `backup_root`. An explicit `--root` uses that exact root, not its configured parent container. Read the restored coverage: inputs marked `LISTED` were recorded but not restored. Recover their exact bytes or follow an authorized acquisition before claiming those inputs available ([command contract](operating.md)).
