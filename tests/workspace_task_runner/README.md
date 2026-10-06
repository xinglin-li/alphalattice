# Workspace Task Runner

This runnable, offline case establishes the durable desktop task plane between
Research Triage and Factor Research. It stops after `FeatureInputSnapshot`.
It is playpen evidence, not a production runtime contract.

```text
Front Desk private session
  -> ResearchHandoff -> ResearchHandoffGateway -> DataOperationsRequest
  -> DuckDB task registry -> one preparation turn -> FeatureInputSnapshot
  -> safe task-status projection -> Front Desk
```

## Ownership

| Owner | Authority | Explicit non-authority |
| --- | --- | --- |
| `research_intake` | host-facing handoff and safe outcome contracts | task execution, DuckDB writes, model transcript storage |
| `research_tasks` | task admission, lifecycle/event authority, execution identity, reconciliation | raw data truth, physical artifact paths, agent memory |
| `workspace_runtime` | one-process mutation ordering, private artifact resolution | business lifecycle or provider policy |
| `market_data_ops` | deterministic fetch, sanitizer, raw/action storage, audit, snapshot production | Front Desk conversation, task admission, model decisions |

DuckDB is the one authority for task status, events, the interrupt an execution
recorded, immutable artifact links, and market data. No graph journal is kept
beside it (W10): a deferral or a review is an interrupt the execution records in
the registry, and any runner resumes it from there.

## Identities and safe state

- `DataOperationsRequest` is stable business input. It has no `run_id`.
- `task_id` is one admitted Data Preparation task.
- `execution_id` is one restartable execution; the interrupt it recorded
  (kind, id, hash) is what a resume must match.
- `review_id` and `deferred_retry_id` bind the only permitted resume commands.
- `TaskArtifact` contains content hash, metadata hash, kind, and a safe URI;
  it never contains a physical path.
- An execution record holds only task/execution identifiers plus the current
  typed interrupt contract. It contains no Front Desk messages, agent VFS,
  tool transcript, SQL, credentials, or prices.

`ExecutionCompatibility` freezes the task contract, workflow definition,
manifest, market profile, provider binding, data policy, and remediation policy
hashes. A compatibility change is not a patchable resume: the old execution is
marked `execution_superseded_by_compatibility` and a newly admitted task is
required.

## Lifecycle and reconciliation

The registry atomically records lifecycle plus append-only safe event. Allowed
states are `admitted`, `running`, `deferred`, `review_pending`, `completed`,
`blocked`, `reconciliation_required`, and
`execution_superseded_by_compatibility`.

| Latest execution in DuckDB | Action |
| --- | --- |
| matching compatibility + a recorded deferral or review | Resume, under any runner, only after the review/deferred contract is revalidated against the recorded interrupt. |
| matching compatibility + no recorded interrupt | Enter `reconciliation_required` (`research_task.execution_abandoned`): a process stopped mid-turn, so do not infer that a refresh or snapshot did or did not happen. A later owner must prove effect receipts before recovery. |
| none | Enter `reconciliation_required` (`research_task.execution_missing`). |
| changed compatibility | Mark execution superseded; do not resume the old execution. |
| malformed/stale/expired review or deferred contract | Fail closed with a safe status; no model is called. |
| terminal task | The DuckDB result is authoritative. |

The runner first validates due time and the recorded interrupt
`(kind, id, hash)`, the hash taken of the interrupt's stored form. It lets the
resumed preparation turn make its idempotent deterministic effect, then records
the accepted resume. A crash before the turn leaves the original interruption
resumable; a crash after a terminal task effect leaves DuckDB terminal.

`WorkspaceMutationGate` serializes every preparation turn and all registry writes in
the one desktop Python process. It uses a synchronous re-entrant lock, not an
async queue: this prototype has one runner, and a queue would create another
durable command/recovery protocol without a real concurrent producer. A second
read-write process is outside the runtime boundary and must fail before it can
claim the workspace writer role. `WorkspaceRuntime` therefore acquires an
OS-level `.alphalattice-writer.lock` lease for its lifetime; DuckDB remains the
second protection at the database boundary.

## Artifacts

`ArtifactResolver` alone maps
`playpen://feature-input/<content-hash>` to
`feature-input/<content-hash>.parquet` under the trusted artifact root. It
validates the Parquet snapshot metadata before first publish and on reuse. A
matching existing hash is verified and reused; it is never overwritten.

Task records contain immutable links rather than fragile reference counts. A
future GC must use reachability plus an explicit retention policy. Completed or
blocked executions are still marked `eligible_for_prune`, a field with nothing
left to prune since W10 that retires with the Task registries' merge.

## Runnable proof

```powershell
$env:ALPHALATTICE_NETWORK_DISABLED = '1'
uv run --extra data pytest --no-cov `
  tests/workspace_task_runner -q
```

The fixture proves duplicate active admission, rate-limit defer/early rejection
/due resume, a recorded deferral resumed under another runner, no repeated review
factory on exact HITL resume, compatibility rejection, content-addressed artifact
reuse without path leakage, abandoned-execution reconciliation, and rejection of
a second local writer lease. It uses no provider credential
or model call. It does not yet claim a scheduler, a second writer process,
artifact GC, automatic effect-receipt repair, Factor Research, or live agent
remediation.

The current product-stage assessment, including desktop-robust evidence and
intentionally deferred production capabilities, does not belong in this case
record. It lived in `workflow-readiness-audit.md` at the Playpen root, which was
removed when desktop entrypoints and host composition were contained; read it
with `git show 7a2c88b3:workflow-readiness-audit.md`. No successor document has
replaced it.
