# Maintain data and the workspace
Date: 2026-10-03

An update can publish a new research input; it never replaces the exact input bound by an earlier study. Read the proposed update and its receipt, because an exact completed update can return `REUSED_EXACT` without making another Task. The [third-party data guide](third-party-and-data.md) owns source terms and acquisition details.

## Update, defer and resume

After the workspace is bound and its Host is available:

```powershell
alphalattice data-update plan --output update-plan.json
alphalattice data-update run --from update-plan.json --output update-submission.json
alphalattice task show --from update-submission.json --wait
alphalattice data-update show --from update-submission.json
```

While a data update is unfinished, `data-update plan` returns that update's own plan. Run follows or resumes it. The plan binds the workspace state and maintenance policy; run that exact plan. Acquisition requires admitted workspace network access. The first-use 24-hour delegation is the only agent network/initial-preparation exception; otherwise the person opens network and confirms in Local Web. Maintenance follows exchange sessions and the settlement policy, not a promise that recent provider bars are final. Source deferral or quality review can stop publication. A completed download alone is not a new research input.

Yahoo Finance can defer an update with its provider code in `cycle.failure_code` and `retry_after_at`; its Task stays `DEFERRED` and fetched listings are kept. After that time, run the same plan to resume the same update. An early attempt is refused as `workspace_data_update.retry_not_due`. Preparation uses `progress.retry_after_at` and its permitted confirmation path. Timeout or unreadable responses may instead leave failed listings and issues. Read `issue list`, the evidence and allowed choices. Do not assume automatic retry or restart acquisition. A deferred or failed acquisition leaves earlier published inputs unchanged.

New Sector observations apply from their effective sessions; published sessions keep their values. Historical source revisions keep their audit evidence; full-history escalation requires authorization. A new temporal statement does not repair an earlier study's point-in-time limitations.

## Data issues

Read an issue's evidence, permitted options and effects before choosing. Its preview binds the case token, evidence hash and exact option; confirm only the offered choice through the authorized path. Agents cannot invent an option or enlarge the scope. The first-use goal may record permitted preparation-issue decisions; it does not authorize revocation, storage changes or activation. A person's `issue delegate` can admit a bounded automation continuation and `issue revoke` withdraws it; the continuation must match its grant. A large price move alone does not establish source corruption. Quality decisions and exclusions remain in the product record. Resume the existing Task when offered; do not edit databases or replace hashes.

## Backup and restore

```powershell
alphalattice backup run
alphalattice backup list
alphalattice backup restore --dir workspaces/restored-research
```

Backups hold market revisions/history, audit tables, Task Control, research programs and evidence, Evidence/CRO records, manifest and Host records outside the workspace. Rebuildable projections are not full duplicates; listed inputs and authority packages carry digests for recapture or reinstall. The Windows default is `%LOCALAPPDATA%/AlphaLattice/backups/<workspace-id>`; `ALPHALATTICE_BACKUP_ROOT` changes the root. Seven generations are kept by default; `backup run --keep <count>` selects the retained count. Protect the backup root as well as the workspace.

Restore to a new or empty directory; do not overwrite the original workspace. Omit `--generation` for the newest generation or use `--generation <hash>` with a whole/unique hash from `backup list`; ambiguous prefixes are refused as `workspace_backup.generation_ambiguous`. If the original manifest is lost, provide `--workspace-id <id>` and the recorded backup root with `--root <path>`. Restore can run without a Host. Read `limitations`: market tables are staged as Parquet for rebuild, and listed inputs or authority packages may need recapturing/reinstalling. `RESTORED` does not prove all derived values or reports can already be read.

## Storage

`storage show` describes storage; `storage plan` estimates retention and bytes without deleting files. Panel retention protects the current head, prior rollback head and explicit pins; only unprotected candidates enter the plan. A person confirms the exact plan in Local Web; CLI and agents cannot confirm it. Inspect scope and recovery status. Use product retention controls rather than deleting files whose bindings determine whether they can be rebuilt.
