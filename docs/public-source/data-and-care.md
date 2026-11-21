# Maintain data and the workspace
Last checked: 2026-10-22 against 0.2.3

Updates preserve the exact inputs bound by earlier studies. Local Web shows the proposed change, progress and receipt; reusing an exact completed update can return its receipt without another Task. Source terms belong to the [third-party data guide](third-party-and-data.md). The [CLI guide](cli.md) describes commands and continuations. The [research-agent guide](../../AGENTS.md#what-only-a-person-decides) lists choices that need you.

## Updates and data issues

An unfinished update keeps its own plan. A provider deferral retains fetched listings and records when continuation is due. The offered continuation resumes that update; preparation has its own confirmation path. Timeouts or unreadable responses can instead leave failed listings and issues requiring review. Earlier published inputs remain available.

Issue records show evidence, permitted choices and their effects. A preview binds the exact case, evidence and choice; confirmation cannot enlarge its scope. First-use decisions follow the guide's delegation. For a data or research update attributed to an open goal, the workspace's declared rule also decides the run's data cases and dividend-related full-history audits within their separate bounds. Data cases use a case count and affected-listing share; audits have their own affected-listing share, so the two kinds do not consume one combined allowance. Beyond a bound the case remains escalated. Outside that goal authority, follow the offered confirmation. A person's `issue delegate` grants a bounded automation continuation, and `issue revoke` withdraws it. The continuation must match its grant. A large price move alone does not establish source corruption.

A rule decision holds for the same anomaly while the rule's fresh choice still agrees; a changed fresh choice is recorded as a new decision. A person's later decision on that anomaly ends the rule's hold, and a released quarantine no longer holds it. Issue readbacks and the goal record disclose the rule, its identity and whether the decision was held, changed or escalated. [Privacy](privacy.md#workspace-network-and-consent) explains continuing network permission and its per-step disclosure.

Maintenance follows exchange sessions and settlement policy; recent provider bars may still be provisional. A completed download alone does not publish a research input. New Sector observations apply from their effective sessions, preserving published values. Historical revisions retain audit evidence. A full-history audit uses the recorded rule approval, or the offered authorization when the rule cannot decide. Later temporal statements do not repair an earlier study's point-in-time limitations.

An older workspace's sessions generated from the pinned calendar rules remain reference data for historical reads. Their rule origin and version are retained. Recorded source observations keep their own times, and later calendar corrections become visible from their later knowledge time.

## Backup and restore

```powershell
alphalattice backup run --keep 7
alphalattice backup list
alphalattice backup restore --dir workspaces/restored-research
```

Backups hold market revisions and history, audits, Tasks, research programs and evidence, Evidence and CRO records, manifest and Host records outside the workspace. Rebuildable projections are not full duplicates; listed inputs and authority packages carry digests for recapture or reinstall.

On Windows, the default root is `%LOCALAPPDATA%/AlphaLattice/backups/<workspace-id>`; `ALPHALATTICE_BACKUP_ROOT` changes it. Seven generations are retained by default; `backup run --keep <count>` selects another count. Protect this root alongside the workspace.

Restore targets a new or empty directory and can run without a Host. The restored workspace's Tasks move onto the log when a Host first opens it. It uses the newest generation unless `--generation <hash>` selects a whole or unique hash from `backup list`. If the original manifest is lost, name its recorded identity with `--workspace-id <id>` and backup root with `--root <path>`.

The restore receipt identifies the generation and limitations: market tables are staged as Parquet for rebuild, and listed inputs or packages may need recapture or installation. A restored workspace may still lack derived values or readable reports.

## Machine data roots in 0.2.8

Machine-root backups keep the admitted data, results, Evidence bodies and index text.
They exclude raw transport, credentials, keychains and process locks. The existing
workspace backup commands remain available.

The data-layer API creates a generation in a separate backup repository and restores
it into a fresh root:

```python
from pathlib import Path

from alphalattice.kernel.data_layer.backup import backup_data_root, restore_data_root

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

Markets have stable local identifiers. Their MICs are dated names, so changing a
market code preserves the identities of its listings and sessions.

An older workspace's market tables follow the first-start upgrade plan into a new
data root: verified backups and an equality check precede the reader switch, and
the workspace's tables are retained. Earlier development roots are refused with
`data_layer.schema_mismatch`; they are never migrated in place.

Tasks are written to the machine's log in the data root: `ALPHALATTICE_DATA_ROOT` when set, otherwise `%LOCALAPPDATA%/AlphaLattice` on Windows and `$XDG_DATA_HOME/alphalattice` elsewhere. One Host holds a data root at a time. A second Host on the same data root is refused with `data_layer.host_running`: close the other Host, or start this one with `ALPHALATTICE_DATA_ROOT` set to a folder of its own. A Host whose log does not open does not start (`data_layer.log_unavailable`); restore the data root from its last backup. A 0.2.7 workspace's Tasks move onto the log at its first open, and its old Task store and frozen requests are set aside under `runtime/retired-0.2.7/`, never deleted. A model sandbox and a saved-object readback open their copies on a data root of their own, so the copies' Tasks stay out of the machine's history.

A workspace's market data enters the data root's world when the Host opens it. Its prices, adjusted closes, corporate actions and provider mappings are then read from world, as of the moment of the read. A listing, provider mapping or universe member is known from the start of its valid date in UTC, or from its admission when that date is still ahead, so a first use reads the reference data it has just written; prices and corporate actions are known at their session's close. Provider mappings retain their source's lifecycle state: a delisted interval reads as active before its closing date and delisted from that date on. Universe manifests and membership also enter the data root and are read the same way. World answers only while it holds the workspace's current tables: after a data update, the Host admits the new rows before its next read, and a Task reads the workspace's own tables until then. A research input bound in the Host fixes what its research may see, as a selection recorded in the log: its market values are read as world held them when it was bound, so a later correction never moves a study's inputs. An input bound with no selection, outside a Host or while world did not hold the workspace's tables, reads its own copy. A Panel's sector map is selected the same way at its first read in the Host, when world holds exactly the map's Sectors then; its readers take those Sectors from world as of that moment after, and a map world holds differently reads its own.

For a prepared workspace, put recorded bars in its `market-bars/` folder before starting the Host to select the local-file adapter for that workspace. Store one UTF-8 JSON file per provider symbol, such as `SPY.json`. Each file has `schema` set to `alphalattice.market-bars.v1`, `provider` set to the data source named by the workspace's market profile, and a `bars` array. Each row has a unique ISO `session_date` and numeric or null `open`, `high`, `low`, `close`, `volume`, `split_ratio`, `cash_dividend`, and `capital_gain` fields. Supply `adjusted_close` when the recorded source provided it. Use the profile's price basis. The adapter reads local files for bars, actions and adjusted closes, including both endpoints of the requested session range. A missing or invalid file is reported by the adapter. Daily reads keep the workspace's admission and knowledge-time rules.

Research inputs, the market database and Panel files a study is bound to, are kept once per machine under the data root and read through each workspace's view of them; equal files are kept once whichever workspaces bind them, and a binding's files go only after the last workspace that binds it is deleted. A 0.2.7 workspace's copies under `research-inputs/` are kept there at its first open, read back equal, and removed only after a machine backup holds them and reads them back equal; its publication records stay in the workspace. Storage clean-up neither pins nor removes research inputs; `alphalattice input list` shows them.

Execution outcomes are kept in the machine's data-root catalog and read from it. At the first 0.2.8 start the upgrade copies each workspace's outcome files into the catalog and leaves the files in place; newly published outcomes have catalog bodies only. A workspace copy, a model sandbox's or a saved-object readback's, and a workspace backup carry the workspace's execution outcomes and Portfolio results under `runtime/carried-results/`, and the Host that opens the copy or the restored workspace admits them onto its own data root's catalog before it reads them. Each carried result is admitted only when its rows rebuild the result it names; one that does not is refused by name, stays in the copy, and the Host does not open. A workspace backup also holds the Portfolio ledger's records under `runtime/artifacts/portfolio-strategy-lab/`, which a restored workspace reads its Portfolio results with.

At the first 0.2.8 start, the upgrade copies and verifies Sector development target values, forecast series and evaluation measures in the data-root catalog and leaves their workspace files in place. The retained files are no longer read, and new results are written to the catalog only. Start the workspace's Host before using `scripts/run_sector_research_development.py --publish`. The command sends publication to that Host.

Alpha development publishes its validation and score rows to the data-root catalog
and records candidate summary measures in the log. A workspace on a data root
reads admitted chunks from the catalog. A standalone workspace without a
catalog reads its own files.

The Strategy Lab's numerical lanes are kept once per machine in the data root: each lane's body under `blobs/`, its kind, shape and dtypes in the log, held by every workspace that cites it. Every read checks the body against that profile and proves its bytes; a lane that differs is refused with `data_layer.array_profile_mismatch` or `data_layer.array_tampered`, and a lane whose body is gone with `data_layer.array_missing`: restore the data root from its last backup. At the first 0.2.8 start the upgrade copies each workspace's lane files into the data root and leaves the files in place as its input; until it has, a workspace reads its lanes from its own files. A lane written after the upgrade has no file. A workspace copy, a model sandbox's or a saved-object readback's, and a workspace backup carry the workspace's lanes under `artifacts/portfolio-strategy-lab/carried/`, and the Host that opens the copy or the restored workspace moves them onto its own data root before it reads them.

At the first 0.2.8 start, the upgrade also copies and verifies Alpha's existing array and score bodies in the data root, leaving their workspace files in place as its input. Alpha reads the data root only; the backup is the way back.

Risk models use the catalog's factor covariance, factor loading and specific-risk tables. The catalog keys factors by their registered lineage IDs. The upgrade moves published covariance history into exact lower-triangle coordinates with each formation's original listing order and verifies the original matrix hashes. It leaves the packed files in place, and the catalog is the only covariance store. Workspace backups and workspace copies carry the exact packed bytes of published history and durable checkpoints. A restored workspace's covariance is read from the catalog once the Risk upgrade has admitted it, which waits for its market listings to be registered in the receiving catalog; the next start offers that upgrade once those listings are available.

The data root keeps a result's body (a numerical lane, an execution outcome's rows) while something holds it: each workspace that cites it, or a pin. A body nothing holds is removed 7 days after it was produced, when the Host opens the data root; its profile and measures stay in the log, and a removed body is never held again (`data_layer.artifact_held_collected`): produce it again. The log records each removal before the body goes, so an interrupted one only keeps a body longer, and the next start finishes it. No removal runs while a machine backup copies, and the backup keeps its own copy of every body it names.

A workspace's acquired evidence documents enter world too: when the Host opens the workspace, and when an acquisition ends. Each document is held once per machine, its body under the data root's `blobs/`, and the workspace's own copy is only read, never moved. An evidence read takes the version of a document world knew at its selection's cutoff; a document world knew only later is refused as missing for that cutoff. A document world does not hold is read from the workspace's own store. A body world holds but whose blob is gone is refused with `alternative_evidence.world_body_missing`: restore the data root from its last backup.

The data dependencies include engine-matched DuckLake and SQLite scanner binaries. The data-root catalog loads these installed packages without downloading extensions. Workspace connections refuse runtime extension installation or loading. If an installed binary cannot load, restore the locked data dependencies with `uv sync --locked --extra data`.

`storage show` describes storage; `storage plan` estimates retention and bytes before deletion. Current and rollback Panels and explicit pins are protected. A person decides whether to confirm the exact plan. They can confirm it in Local Web or answer the agent's one-line question; the agent then relays their words through `storage confirm` with `person_confirmation`. Review its scope and recovery status through the product controls. File bindings determine what can be rebuilt.
