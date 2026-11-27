# Privacy and native usage reading
Last checked: 2026-11-24 against 0.2.9

The workspace Host accepts native agent OTLP on its existing authenticated loopback
socket. Before storage it discards prompt and tool text, bodies, opt-in attributes and
unknown fields. DuckLake keeps only numerical usage, times and explicit attribution:
conversation, session, agent, goal and trial identifiers when provided. Unplaced records
are counted; missing exports, counts, parents and member claims remain unmeasured.
Native observation grants no research or membership authority.
Only known native event and metric names are retained; other names become unplaced.
The authenticated workspace binds event identity and scopes session and trial readings.

## Workspace network and consent

At first use, your agent asks once, in the product's exact words:

> May AlphaLattice keep this workspace's network on for your daily updates, to fetch new market data and SEC filings within the default budget (3 per issuer, 600 documents and 3 GB overall), until you tell me to stop? Each use is recorded in the activity log and shown in the Workbench.

Your relayed yes sets the workspace network and the default SEC acquisition consent. It is your own continuing permission, separate from the first-use goal's 24-hour delegation. The delegation never overwrites your network setting, including a setting that keeps it off. A research run's offline hold and the operator's offline switch still take precedence.

An admitted network step reports `network_use`, with `under` naming whose setting allowed it and `words` giving the one-line disclosure. Activity records the same permission source when available, including for automated updates; observation gaps are named.

Say "Turn AlphaLattice's network off" to your agent, or turn it off in **Settings**. Your stop also blocks SEC acquisition before its next request; a data update already fetching finishes its current stage unless its Task is cancelled. A delegation cannot reopen your setting after that stop. A later change needs your decision again. Earlier inputs, Evidence and activity records remain available. Source and model terms are in [Downloads and data terms](third-party-and-data.md).

## Optional trace export

The Host keeps its history locally. To also send traces and metrics to an OTLP/HTTP backend, set `OTEL_EXPORTER_OTLP_ENDPOINT` to the collector's base URL in the Host process environment before starting the Host. The Host appends `/v1/traces` and `/v1/metrics` to that base. To use a specific URL, set `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` or `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` to the full URL. It takes precedence over the base URL. Use an HTTP or HTTPS URL without credentials or a query string. The metrics are the duration of each traced Host request (`http.server.request.duration`, by method and route): a client's step or a write, never a polling read, which opens no trace. A point's exemplars carry the trace and span ids of sampled requests in it, so a slow point opens its trace. With no endpoint, the Host opens no export connection and records no metric. The workspace network control and the operator's offline switch still apply, including when the network is turned off after startup. While offline, the Host sends only to a receiver at a literal loopback address (`127.0.0.1` or another `127.x.x.x` address, or `::1`), and only while the workspace's own network control is on; such a send never uses a proxy, so it does not leave the machine. A host name, even `localhost`, is refused.

Exported span attributes are checked for credentials and redacted. Prompt, message and result content is left out by default. Set `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=span_only` in the Host environment only if you choose to include content in exported spans. Credential redaction still applies when content is enabled. Restart the Host to apply changes to export settings. The local history remains the record of truth.

## Native export and readings

Start the Host, then run `alphalattice session exporters --workspace <explicit-workspace>` to
print Claude Code's exporter environment and the Codex user-configuration lines. Only
after the person's yes, add `--configure-claude` to merge that environment into the
project's local Claude Code settings. Other project keys remain; no user settings are
opened. Codex's configuration is always printed for the person to apply. Run the command
again after a Host restart because its existing loopback port and credentials change.

No receiver component or extension download is needed: the Host decodes OTLP with its
installed protocol dependency and commits each accepted batch through StorePort, the
catalog's one writer. A repeated delivery keeps the same event rows. Vendor usage maps
to one GenAI reading; counter snapshots are not added to their corresponding events.
The same response's later observation replaces its counts. A trial's close records its
observed native sums in the log, or explicitly records UNMEASURED.

Research requires no product hook. Milestones and `alphalattice session usage` publish
snapshots of the exact bound conversation, with only explicit parent links. There is no
transcript token reader or polling timer. Reads are cached for ten seconds; failures
are named and do not stop research. No native tool command is parsed for a member key.
Existing keyed product submissions retain their original recorded authority, but
telemetry alone cannot attribute a Claude subagent to a committee member.

The Codex binding authorizer still reads only the bounded first `session_meta` record
to prove a thread's parent. It never reads turns or token records. This authorization
source is separate from observational lineage in the catalog. Prior file-derived
usage observations remain readable; old derived checkpoints are no longer read or
allocated and remain subject to ordinary storage collection.

Offline developer checks can still read explicitly supplied native evidence files to
verify recorded token counts and member-key evidence. These checks do not run in the
research Host, watch live conversations, or allocate transcript-reading checkpoints.

Counts and model time are observations, not billing or completion proof. Missing or
impossible numerical fields do not become a complete zero total; parent and child
totals are not combined as if their non-overlap had been proved.

## Privacy controls

Reading starts enabled. Either control disables it:

- **Settings** controls all Sessions bound to this workspace. `alphalattice usage-reading show` reads this setting. Only a person changes it; an unreadable or invalid switch record reads as off.
- `alphalattice session bind --usage off` disables ingest matching that Session and its usage reading. Another Session cannot enable it. Automatic binding uses `--usage read`.

To change your Session's workspace, usage or roles, run `alphalattice session unbind`, then bind again with the explicit workspace. Unbind removes only your own record and keeps workspace history. See [native sessions](../../.agents/skills/alphalattice-research/references/native-visibility.md) for the binding contract. Earlier readings and product records remain, and neither control changes the native host's storage.

## Separate launch counter

A valid agent-session identifier permits the CLI to append a constant `1` to a temporary counter under `alphalattice-cli-launches/`. Help and schema launches count. This reads no session files and writes nothing to the research workspace. `--usage off` does not disable it; optional `session_launches` includes the current launch and is omitted if the identifier or writable counter is unavailable. See [CLI](cli.md).
