# Current-universe bootstrap

This helper moves the local-first prototype from a 64-symbol Front Desk
allowlist to the original project's full current-US bootstrap standard:

```text
current S&P 500 ∪ NASDAQ-100 ∪ DJIA
  -> explicit, frozen source manifest
  -> full historical acquisition
  -> deterministic ten-year quality gate
  -> workspace research whitelist
  -> user-selected research subset
```

The executable intentionally prints **bootstrap candidates**, not a falsely
qualified whitelist. The original quality gate remains authoritative: at least
ten calendar years of data, no more than 2% missing expected sessions, and no
run of more than 20 missing sessions. Candidates that fail remain outside the
frozen research whitelist.

Run the offline invariant:

```powershell
.venv\Scripts\python.exe -m pytest --no-cov tests/current_universe_bootstrap/test_current_universe.py -q
```

Run an explicit live discovery (the current source pages may legitimately fail
closed if their expected schemas change):

```powershell
uv run --extra data --extra data-live python playpen/scripts/run_full_universe_whitelist.py `
  --refresh --format json
```

The application must not perform that web discovery on every startup. It first
persists the response hashes, retrieval timestamp, construction rule, member
decisions, and content hash as one workspace manifest. Subsequent runs use the
frozen manifest and incrementally refresh its members' market data. A later,
separate manifest rebuild is an explicit long-running maintenance task.

The helper deliberately does not reuse the production helper's cross-source
company-display-name equality check. Live index pages use incompatible labels
for one ticker (for example `Amazon` and `AMAZON.COM INC`), so that check makes
the real source union fail before data acquisition. At candidate stage a ticker
only selects a bounded fetch target; each source label is retained as evidence.
It cannot create a durable listing identity or silently resolve issuer identity.
