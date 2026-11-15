# Maintain data and the workspace
Last checked: 2026-10-22 against 0.2.3

Updates preserve the exact inputs bound by earlier studies. Local Web shows the proposed change, progress and receipt; reusing an exact completed update can return its receipt without another Task. Source terms belong to the [third-party data guide](third-party-and-data.md). The [CLI guide](cli.md) describes commands and continuations. The [research-agent guide](../../AGENTS.md#what-only-a-person-decides) lists choices that need you.

## Updates and data issues

An unfinished update keeps its own plan. A provider deferral retains fetched listings and records when continuation is due. The offered continuation resumes that update; preparation has its own confirmation path. Timeouts or unreadable responses can instead leave failed listings and issues requiring review. Earlier published inputs remain available.

Issue records show evidence, permitted choices and their effects. A preview binds the exact case, evidence and choice; confirmation cannot enlarge its scope. First-use decisions follow the guide's delegation. For a data or research update attributed to an open goal, the workspace's declared rule also decides the run's data cases and dividend-related full-history audits within their separate bounds. Data cases use a case count and affected-listing share; audits have their own affected-listing share, so the two kinds do not consume one combined allowance. Beyond a bound the case remains escalated. Outside that goal authority, follow the offered confirmation. A person's `issue delegate` grants a bounded automation continuation, and `issue revoke` withdraws it. The continuation must match its grant. A large price move alone does not establish source corruption.

A rule decision holds for the same anomaly while the rule's fresh choice still agrees; a changed fresh choice is recorded as a new decision. A person's later decision on that anomaly ends the rule's hold, and a released quarantine no longer holds it. Issue readbacks and the goal record disclose the rule, its identity and whether the decision was held, changed or escalated. [Privacy](privacy.md#workspace-network-and-consent) explains continuing network permission and its per-step disclosure.

Maintenance follows exchange sessions and settlement policy; recent provider bars may still be provisional. A completed download alone does not publish a research input. New Sector observations apply from their effective sessions, preserving published values. Historical revisions retain audit evidence. A full-history audit uses the recorded rule approval, or the offered authorization when the rule cannot decide. Later temporal statements do not repair an earlier study's point-in-time limitations.

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

## Machine data roots in 0.2.8

Machine-root backups keep the admitted data, results, Evidence bodies and index text.
They exclude raw transport, credentials, keychains and process locks. The existing
workspace backup commands remain available.

The data-layer API creates a generation in a separate backup repository and restores
it into a fresh root:

```python
from pathlib import Path

from alphalattice.kernel.data_layer import backup_data_root, restore_data_root

generation = backup_data_root(Path("data-root"), Path("machine-backups"))
restored = restore_data_root(generation, Path("restored-data-root"))
```

Keep the whole backup repository: each generation contains its three SQLite snapshots,
file manifest and saved log chain head; its immutable files live in the shared `files/`
pool. Later generations copy only new files into that pool.

Each file keeps its owner's recorded content digest when available; otherwise the
generation records a versioned BLAKE3 seal of its bytes. Backup verifies a pool file
before reusing it. Restore verifies the retained bytes and reconciles the file list
against the copied catalog and admitted blob owners before publishing the new root.
A changed file or an incomplete generation is refused by name. Generations without
these file seals are refused; take a new backup to obtain a verified generation.

Online backup copies the index, then the log, then the catalog, without holding the
Host writer for the backup's duration. The catalog may contain commits newer than the
copied log. Restore first verifies the saved chain and its catalog references, then runs
the store's existing startup repair to append owed events. It relocates the restored
catalog to the new root. An unverifiable chain, missing file or existing destination is
refused by name; choose an intact generation and a fresh destination.

## Storage

The data dependencies include engine-matched DuckLake and SQLite scanner binaries. The data-root catalog loads these installed packages without downloading extensions. Workspace connections refuse runtime extension installation or loading. If an installed binary cannot load, restore the locked data dependencies with `uv sync --locked --extra data`.

`storage show` describes storage; `storage plan` estimates retention and bytes before deletion. Current and rollback Panels and explicit pins are protected. A person decides whether to confirm the exact plan. They can confirm it in Local Web or answer the agent's one-line question; the agent then relays their words through `storage confirm` with `person_confirmation`. Review its scope and recovery status through the product controls. File bindings determine what can be rebuilt.
