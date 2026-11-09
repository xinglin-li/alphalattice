# Delivery pipeline policy

## Process policy

- Keep the impact runner's file-per-process default; batch only through its audited safe list and revoke batching on a process-global framework import.
- Defer `post_merge_only` files from commit while retaining whole post-merge, candidate and nightly standing; a structural file must declare whole-tree scope and contain no fenced guard.
- Run a structural guard with no declared `reach` in the commit stage only when its own file changed or a
  `structural_fence` row names its fence law (public-private, identity, evidence, secrets); post-merge, candidate
  and nightly run every structural guard whole, and each run's `structural-moved.json` names the guards that moved.
- Run explicitly required Small and Medium nodes at declared limits, including quarantined nodes, and record them in the same receipt. Measure Large, end-to-end and native tests separately and run them in their scheduled tiers.

## Verified result reuse

- Workshop commit stages currently execute selected tests without result reuse.
- Use `--result-cache <directory>`, `--result-cache-bytes <positive bytes>` and `--cache off|record|shadow|on`; default off, with record and shadow executing every selected group.
- Reuse only a verified passing first attempt in explicit on mode; keep cache decisions, original bounded metadata and provenance in the receipt, and never publish a dirty stage.
- Disable the shared store on a shadow mismatch and repair it before reenabling reuse; require reviewed shadow and recall evidence before promotion.
- Never store Large, quarantined, end-to-end or native results.
