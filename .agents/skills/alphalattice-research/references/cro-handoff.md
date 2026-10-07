# The review handoff
Date: 2026-10-06

The CRO assesses one book, one date and one eligible Evidence publication.

## Prepare and assign

Read the exact book's current Evidence answer after required continuation within an explicitly declared allowance and publication of the current Analyst answers ([Evidence handoff](evidence-analysis-handoff.md)). An absent or exhausted allowance grants no further source reading; retain unread limits for bounded review. Follow Evidence's exact `dossier` request, then that dossier answer's `cro_bundle` request, naming a new directory with `bundle_directory` in the choices file:

```text
alphalattice request --from "<out>/evidence-current.json" --action dossier --output "<out>/cro-dossier-answer.json"
alphalattice request --from "<out>/cro-dossier-answer.json" --action cro_bundle --choices "<out>/cro-choices.yaml" --output "<out>/cro-bundle-answer.json"
```

The Host packs a bounded Markdown bundle with index, coverage, holdings and aliased findings, including unread scope after an absent or exhausted allowance or `NOTHING_RESUMABLE`. The answer names the answer file, files/bytes and one submit command. The JSON dossier remains the audit/UI read. If no product bundle is offered, follow the current answer's action or report its held reason; never rebuild it from edited artifacts or an internal owner.

Give the CRO the bundle path, filenames and answer file, never your interpretation. Confirm source exposure, permissions and budget. On Claude, up to 24,017 bytes may use `alphalattice_cro_medium`; larger bundles use `alphalattice_cro`. This is a measured recall boundary, not a product limit. Codex has only the high card. Pass no model override. The card reads the bundle and writes its answer file.

## Submit and read

Run the returned `submit_command`. On `CORRECT`, ask the same reviewer to fix only the named issues, at most twice, then rerun the same command. A correction never permits changing judgment, book or reviewer. `ACCEPTED` admits the answer; after the second correction, `DONE` admits acceptable risks and records the rest as dropped.

Keep the receipt (role, verdict, Task and accepted/dropped counts); do not read or relay the answer, add a caller or installed-Agent binding, or invent token counts. A stale bundle needs a fresh preparation. Wait on the receipt Task, then read or export the published review by the book selector. Keep product actor provenance apart from child/model metadata. Unknown external tokens are unavailable, not zero; a separate role's judgment is not independent validation.
