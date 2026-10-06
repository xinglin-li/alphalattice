# Feature Input Gateway and Recoverable Research Universe

This runnable playpen case closes the boundary between mutable data/feature
maintenance and Factor Research. It does not alter raw OHLCV, corporate
actions, base features, or sector-neutral math.

## Proven shape

```text
candidate manifest
  -> deterministic evidence
  -> provider/sector macro circuit breaker
  -> one token-scoped Data Engineer case when retry is exhausted
  -> host option validation and deterministic execution
  -> recoverable quarantine / next-manifest revision
  -> immutable panel manifest
  -> bounded PyArrow RecordBatch reader
```

- The clean path invokes no Agent and no HITL.
- Ordinary failures aggregate at `max(10, ceil(active * 10%))`.
- Provider macro failures aggregate at `max(25, ceil(active * 15%))`;
  sector cohorts use `max(5, ceil(sector * 35%))`. They defer before any
  listing quarantine or Agent call.
- The live remediation adapter reaches the real `create_deep_agent` harness
  through the confined kernel and has no unsafe built-in capability surface:
  mutation, deletion, execution, Todo, and general-purpose delegation are
  excluded. One no-argument ToolRuntime tool returns the complete admitted
  Markdown case and finite options. The model returns a fresh-token option
  selection; the Host supplies run, case, and evidence identity when creating
  `DataRemediationProposal`.
- The host owns every executable argument. Quarantine and next-manifest
  exclusion are recoverable; raw data and old manifests remain immutable.
- An unexplained isolated move cannot be retained automatically. Market or
  sector co-movement is diagnostic context, not data truth.
- A unique verified alias is a host-built run-scoped option. Repeated evidence
  churn permits one rediagnosis and then becomes a durable five-minute defer.
- Last-known-good binds its original market as-of session and cannot be
  advertised as a current-data recovery.
- `market_as_of_session` and UTC `knowledge_cutoff_at` enter admission, active
  panel, and snapshot identities. `materialized_at` remains operational audit.
- Factor Research reads only gateway-qualified annual ZSTD Parquet chunks via
  a projected, filtered PyArrow Dataset Scanner. It never reads mutable DuckDB
  or calls `to_table()` on the complete panel.

## Run

```powershell
$env:PYTHONPATH='playpen/src'
uv run --extra application --extra data pytest --no-cov `
  --basetemp playpen/experiments/pytest-feature-input-gateway `
  tests/feature_input_gateway/test_feature_input_gateway.py -q
```

The case is deterministic and offline. It includes a fake tool-calling model
that executes the real Deep Agent harness, validates the primary/repair/recovery
modes, proves at most two executions, and verifies context overflow before any
model call. No model provider or network is used.

## Validation evidence

- The dedicated case includes the original gateway portfolio plus focused
  Agent-interface scenarios for complete Markdown, hidden ToolRuntime context,
  host-bound identity, protocol repair, budget recovery, terminal double
  failure, business rejection, and the workspace-maintenance adapter.
- The dedicated Feature Input Gateway portfolio has 23 passing scenarios.
- The combined affected regression portfolio has 96 passing scenarios across
  Feature Input, Front Desk Data Operations, the pre-Factor remediation adapter
  and terminal, workspace maintenance, and the shared Agent profile/protocol
  catalogs.
