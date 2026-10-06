# Current-universe onboarding

This case turns the original project's current-US whitelist standard into a
desktop workspace task:

```text
explicit current-index discovery (516 candidates in the 2026-08-02 probe)
  -> frozen candidate/acquisition manifest
  -> bounded chunks: concurrent ten-year provider hydration
  -> main-thread listing units: sanitize, persist, quality, action audit
  -> frozen quality-qualified research manifest
  -> later sessions: durable 45-day raw/action maintenance over that manifest
  -> later user-selected research subset
```

The onboarding runner is deliberately not an Agent and does not publish a
FeatureInputSnapshot. Provider workers only fetch one-response hydration
evidence; they cannot access DuckDB or durable task state. The main thread
consumes each completed future and runs the complete listing unit before
releasing response memory. A crash reads DuckDB state and skips completed
units; a provider-wide rate limit or known shared-session failure yields a
durable `deferred`; a broken listing is excluded without blocking unrelated
candidates. The only final membership is
`FEATURE_READY`: raw-history quality and action/adjustment admission both pass.

Each yfinance full-history unit uses one `Ticker.history(auto_adjust=False,
actions=True)` response for raw OHLCV, corporate actions, and ephemeral
adjusted-close diagnostics. The host starts with bounded `chunk_size=25` and a
qualified worker count. A provider-wide failure stops before the next chunk,
commits successes from the current chunk, and degrades only on a due resume
(`4 -> 2 -> 1`). It never sleeps, retries the same batch immediately, invokes
an Agent, or changes manifest/data identity hashes because of worker count.

The bounded 20-symbol live probe on 2026-08-02 completed with zero failures at
workers 1/2/4. Recorded wall times were 12.63s, 2.92s, and 2.86s. Because four
workers were not at least 10% faster than two, the qualified desktop default is
two workers; four remains a tested candidate, not the default or an SLA.

The original thresholds are unchanged: at least ten calendar years, at most
2% missing expected sessions, and no missing sequence longer than 20 sessions.
The resulting manifest is explicitly `CURRENT_ACTIVE_SURVIVORS` with
`is_point_in_time_historical: false`: it remains current-universe research
input, not a PIT universe.

Run the offline case:

```powershell
.venv\Scripts\python.exe -m pytest --no-cov tests/current_universe_onboarding/test_current_universe_onboarding.py -q
```

The explicit CLI/runtime composition stores a live candidate manifest and
advances this runner under the workspace writer lease. That adapter is
[`run_current_universe_onboarding.py`](../../scripts/run_current_universe_onboarding.py):

```powershell
uv run --extra data --extra data-live python playpen/scripts/run_current_universe_onboarding.py `
  --live-data --refresh-universe --work-budget 10
```

`--work-budget` means complete listing units, not fetch phases. Omit it to
advance until the provider defers or all candidate listing units are terminal.
A later restart omits `--refresh-universe`; it
loads the hash-validated candidate manifest from DuckDB and resumes exactly the
same acquisition scope. It must not be silently invoked by a normal Front Desk
conversation. A default desktop launch resolves a weekend or holiday to the
latest common XNAS/XNYS session; an explicit non-session `--as-of` is rejected.

Once onboarding is complete, routine workspace maintenance does **not** repeat
the ten-year bootstrap or rediscover index members. It advances the frozen
research manifest one persisted listing unit at a time with the normal 45-day
raw overlap, full action audit when the receipt is no longer valid, and no
FeatureInputSnapshot publication:

```powershell
uv run --extra data --extra data-live python playpen/scripts/run_current_universe_onboarding.py `
  --live-data --maintain-universe --work-budget 10
```

For the same manifest and completed `as_of_session`, terminal listing states
are reused. A later completed session creates a new maintenance run and leaves
the original qualification evidence untouched. Provider/API failures are
explicit per-listing states; they are not reclassified as history-quality
membership changes.

After completion, the pre-Factor terminal can opt into the frozen
quality-filtered manifest. Its workspace-readiness gate runs before the Front
Desk is built: first launch asks for explicit host consent rather than silently
starting a full download; after seven days it deterministically checks the
candidate membership source. A changed source pauses new research until an
approved delta onboarding completes. It replaces only the Front Desk's US
eligibility projection: users may request any subset, but never a symbol
outside the workspace-maintained research universe.

```powershell
uv run --extra application --extra data --extra data-live python `
  playpen/scripts/run_pre_factor_research.py `
  --full-universe `
  --market-profile playpen/config/market-profiles/us-current-index-research.yaml `
  --data-workspace playpen/experiments/current-universe-onboarding/workspace `
  --live-data --interactive
```

DuckDB retains the candidate manifest, quality summaries, final revision, and
raw/action authority. The Front Desk receives only the final symbol tuple; it
never receives a database path, raw observations, provider evidence, or an
onboarding transcript.
