# Evidence preparation and the Analyst
Date: 2026-10-08

`strategy-book review` prepares a book's Evidence and writes one Analyst bundle per prepared unit; `review continue` submits the answers. Never write source sets, issuer mappings or analysis publications by hand. Give each unit to its own Analyst with only the bundle path, file list and answer file, never your conclusion; its exposure is the one the goal declared or a first use delegated. On Claude, use `alphalattice_evidence_analyst` with its shipped model and effort; pass no model override.

The Host rechecks Task, sources, scope, expiry and policy, and maps the answer's aliases back. On `CORRECT`, the same Analyst fixes only the named problems, at most twice. `ACCEPTED` admits the answer; `DONE` admits acceptable findings and records the rest as dropped. Do not bundle a `FAILED` unit; report its code. Source-exact citations prove traceability, not truth or investment significance.

## Settle the reading scope before the CRO

Read current Evidence's `continuation_scope` and remaining allowance. Continue the selected packet lineage only under an explicitly declared cumulative allowance with sessions and windows remaining: run the bound continuation, have its successor packet analyzed and published, then reread Evidence. A null first-reading allowance grants no continuation; never invent limits. `COMPLETE` means the sealed reading plan is done, not that every source was read. An absent or exhausted allowance or `NOTHING_RESUMABLE` keeps every unread range and `remainders` disclosed for the CRO's bounded review; never enlarge the allowance or claim full coverage.

## First source package

If `evidence preview` reports no suitable package, the running Host installs one as a Task (environment, model, acquisition, index, publish), with an admitted input and the workspace's pinned semantic pack; nothing is stopped:

```text
alphalattice evidence install --setup="--research-input-id <input-id> --research-input-hash <binding> --acquire-sec --entities <tickers> --preflight"
```

Take `--entities` from the preview's `coverage` packs for the book's unit and tell the person which you chose. Preflight acquires nothing and answers at once. To acquire, replace `--preflight` with `--network-consent --install`, with the workspace's network allowed, and follow the Task with `activity wait`; the product names itself to the SEC by its own contact, so never ask the person for a name or an address. Defaults admit at most three filings per issuer, unless the person approves another count, and 10 MB per document, eight issuers and one unit; they describe the setup's own package only, and a later Evidence Task admits under its own budget. Before you acquire, compare the preview's scope and budget with what the person, or the first use's delegation, approved, item by item, and acquire nothing when any is missing or differs. A failed issuer floor offers the same acquisition at one cutoff: run it under the consent already given, and ask only for a wider scope or budget. `--install` changes the workspace's Evidence binding, never a strategy; an install makes packets prepared under the old package stale.
