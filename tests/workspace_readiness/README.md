# Workspace readiness

This case proves the deterministic control plane that runs **before** a
conversational Front Desk receives its effective capability profile for the
full current-US workspace.

```text
workspace missing
  -> host explains initialization and waits for explicit consent
  -> bounded current-universe onboarding
  -> activated frozen research manifest
  -> active manifest replaces the static dogfood US allow-list in the Front
     Desk's in-memory capability profile

active manifest, source not yet checked for the latest settled session
  -> local assess returns SOURCE_CHECK_REQUIRED, without fetching or writing
  -> explicit refresh_sources_if_due performs the deterministic candidate-source check
  -> unchanged: refresh check timestamp and continue
  -> changed: pause new research, require host approval, delta onboard additions
  -> activated revision: require a full downstream feature-universe rebuild
```

It is intentionally neither an Agent nor a generic gate framework. The gate
owns the current-membership fact, user consent, and transition state. Static
`capabilities.yaml` supplies only default market/profile shape for dogfood; it
is never rewritten with live membership facts. At runtime the gate produces an
effective in-memory capability profile: an active quality-qualified manifest
replaces the old US symbols, while every non-ready state exposes an empty US
allow-list and disables candidate handoff creation. The Front Desk may still
converse in that state to explain progress, but it cannot admit research. A
source failure blocks new research but never rewrites the prior manifest or
artifact.

`assess` is a local read, including when an older workspace claims READY without
its required Panel. It describes the missing work without migrating the record.
Existing onboarding/maintenance commands call `refresh_sources_if_due` explicitly;
changed membership still requires the same consent and activation sequence.

The manifest declares `CURRENT_ACTIVE_SURVIVORS` and
`is_point_in_time_historical: false`. It is a current-universe research input,
not a historical point-in-time universe and must not support a claim of
survivorship-bias-free backtesting.

Activation creates a durable `REQUIRED` feature-universe rebuild requirement.
No feature engine exists in this case: that record is a contract for the next
stage, ensuring a future cross-sectional panel cannot silently retain the old
membership revision. Existing artifacts remain immutable and readable; new
handoffs tied to the old manifest receive a structured re-triage request.

Run the offline proof:

```powershell
uv run --extra agent --extra application --extra data pytest --no-cov `
  tests/workspace_readiness/test_workspace_readiness.py -q
```

The fixture verifies first-run consent, bounded resumable onboarding, a
seven-day no-network window, unchanged-source continuation, changed-membership
pause, retained-listing reuse, delta-only hydration, immutable prior manifest,
feature rebuild invalidation, and old-handoff re-triage.
