# Maintain data and the workspace
Date: 2026-10-08

Updates preserve the exact inputs bound by earlier studies. Local Web shows the proposed change, progress and receipt; reusing an exact completed update can return its receipt without another Task. Source terms belong to the [third-party data guide](third-party-and-data.md). The [CLI guide](cli.md) describes commands and continuations. The [research-agent guide](../../AGENTS.md#what-only-a-person-decides) lists choices that need you.

## Updates and data issues

An unfinished update keeps its own plan. A provider deferral retains fetched listings and records when continuation is due. The offered continuation resumes that update; preparation has its own confirmation path. Timeouts or unreadable responses can instead leave failed listings and issues requiring review. Earlier published inputs remain available.

Issue records show evidence, permitted choices and their effects. A preview binds the exact case, evidence and choice; confirmation cannot enlarge its scope. First-use decisions follow the guide's delegation. A person's `issue delegate` grants a bounded automation continuation, and `issue revoke` withdraws it. The continuation must match its grant. A large price move alone does not establish source corruption.

Maintenance follows exchange sessions and settlement policy; recent provider bars may still be provisional. A completed download alone does not publish a research input. New Sector observations apply from their effective sessions, preserving published values. Historical revisions retain audit evidence, and full-history escalation requires authorization. Later temporal statements do not repair an earlier study's point-in-time limitations.

## Backup and restore

```powershell
alphalattice backup run --keep 7
alphalattice backup list
alphalattice backup restore --dir workspaces/restored-research
```

Backups hold market revisions and history, audits, Tasks, research programs and evidence, Evidence and CRO records, manifest and Host records outside the workspace. Rebuildable projections are not full duplicates; listed inputs and authority packages carry digests for recapture or reinstall.

On Windows, the default root is `%LOCALAPPDATA%/AlphaLattice/backups/<workspace-id>`; `ALPHALATTICE_BACKUP_ROOT` changes it. Seven generations are retained by default; `backup run --keep <count>` selects another count. Protect this root alongside the workspace.

Restore targets a new or empty directory and can run without a Host. It uses the newest generation unless `--generation <hash>` selects a whole or unique hash from `backup list`. If the original manifest is lost, name its recorded identity with `--workspace-id <id>` and backup root with `--root <path>`.

The restore receipt identifies the generation and limitations: market tables are staged as Parquet for rebuild, and listed inputs or packages may need recapture or installation. A restored workspace may still lack derived values or readable reports.

## Storage

`storage show` describes storage; `storage plan` estimates retention and bytes before deletion. Current and rollback Panels and explicit pins are protected. A person confirms the exact plan in Local Web; CLI and agents cannot confirm it. Review its scope and recovery status through the product controls. File bindings determine what can be rebuilt.
