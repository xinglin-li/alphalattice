# Evidence preparation and the Analyst
Date: 2026-10-06

Use the exact book selector and the Host's requests. Never write source sets, issuer mappings or analysis publications by hand.

## Prepare packets

`evidence preview` shows issuer scope, admitted source mode, candidates and missing prerequisites; it creates no Task. Use its `prepare` request for the same book. Candidate count is not accepted-document count. `evidence run` acquires only through the admitted mode, canonicalizes and seals packets; it produces no findings. Live source access and local model setup have their own permissions.

Evidence uses workspace CPU budget; read `cpu-budget show` and keep `auto` unless its `machine` is busy or small. Then set the cores yourself with `alphalattice cpu-budget set --cores 4`, your count in place of 4, and say so in one line; research meaning is unchanged.

A coverage Task readies units independently. Wait on its Task with `activity wait --task <task> --each-stage`; when a unit becomes `PREPARED`, read `evidence show` and run its exact `analyst_bundle_<unit>` request into a new directory while other units continue. After each wait event, read the Task; wait again until it ends. Do not bundle a `FAILED` unit; report its failure code.

If preparing by hand, use the exact book selector and current prepared Task:
```text
alphalattice bundle prepare --role ANALYST --task <prepared-task> --unit <unit> --result <book> --dir "<out>/analyst-bundle"
```
An experiment book uses `--study`, `--receipt` and `--session` instead of `--result`; a one-unit book uses `u01`. The Host packs an index and aliased Markdown excerpts. Its reply names the answer file, files/bytes and one submit command; the JSON packet remains the audit/UI read.

## Assign, submit, correct

Give each unit to its own Analyst with the bundle path, filenames and answer file, not your conclusion. Its exposure is the one the goal declared, or a first use's delegation; hand it on without asking again. On Claude, use `alphalattice_evidence_analyst` with its shipped model and effort. Codex keeps its own card. Pass no model override.

Run the returned `submit_command`. The Host rechecks Task, sources, scope, expiry and policy; screens findings and maps aliases back. On `CORRECT`, the same Analyst fixes only named problems, never changes judgment to win approval; make at most two corrections and rerun the same command. `ACCEPTED` admits the answer; `DONE` admits acceptable findings and records the rest as dropped. Keep the receipt, not the answer. Submissions use EXTERNAL_AUTOMATION provenance. Wait for the receipt Task to publish, then read the exact book's current Evidence answer. Select among eligible analyses with `evidence select`; never move files. Source-exact citations prove traceability, not truth or investment significance.

## Settle current reading scope before CRO

Use the current Evidence answer's exact packet requests when offered to read `continuation_scope`, `continuation_request` and remaining allowance. A null first-reading cumulative allowance grants no continuation authority; never invent `session_limit` or `window_limit`. While scope is `PENDING` and an explicitly declared allowance has positive sessions and windows remaining, run the bound continuation under those same limits. Await its Task, prepare the successor packet's Analyst bundle, submit and await its publication, then reread current Evidence before deciding the next step. Another delivery part reads sealed excerpts, not more source; neither an old packet nor a ledger's newest Task substitutes for the selected analysis's exact lineage.

`COMPLETE` means the sealed reading plan is exhausted, never that every source was read. An absent or exhausted allowance or `NOTHING_RESUMABLE` leaves every pending or unread range and `remainders` disclosed for bounded review; do not enlarge the allowance or claim full-source coverage. After required continuation within the declared allowance and publication of the current Analyst answers, take current Evidence's exact `dossier` action, then that dossier answer's `cro_bundle` action for the [CRO review](cro-handoff.md).

## First source package

If no suitable package exists, install one with the maintained command while the Host is stopped and its workspace lease released. Never stop someone else's service. Use an admitted input and the pinned semantic pack in the workspace, with the declared retrieval environment:

```text
.venv-retrieval/Scripts/python.exe scripts/materialize_evidence_cro_authority.py --workspace <path> --research-input-id <selected-id> --research-input-hash <exact-binding> --semantic-model <workspace-pack-path> --acquire-sec --entities <tickers> --preflight
```

For `--entities`, take the issuers `evidence preview`'s `coverage` packs into the book's unit, within the package's limits, and tell the person in one line which you chose. Preflight checks input, scope, local capability, contact and consent; it acquires nothing. To acquire, replace `--preflight` with `--network-consent`, set the user's real `SEC_USER_AGENT` contact in the process environment and allow that process network access. Never print the contact. Defaults admit at most three filings per issuer and 2 MB per document; the installed package covers at most eight issuers and one unit. Stop at a budget or source refusal.

Read `issuer_coverage`: it distinguishes issuers with documents, none filed in-window, and acquisition failures. No-filing issuers do not count against the floor. A failed floor offers the same acquisition at one cutoff; run it yourself under the source consent already given, and ask only when it needs a wider scope or budget. The immutable package installs only with `--install`, which changes the workspace Evidence binding, not a strategy. Re-import its `source_artifact_root` and `source_set_hash` in archive mode to reproduce offline. Import preflight checks only; it never opens a network path. The setup answer names the option and required `location` (a path under a root or the knowledge object missing); follow it exactly. Capture date alone does not prove historical filing availability.

An install replaces the package; packets prepared under the old one go stale. Finish or abandon them first. A book's units must share one package and cutoff, or use official SEC acquisition with the person's consent and the Host served with `--sec-network-consent`. The preview's `source_ways` names the available choices.
