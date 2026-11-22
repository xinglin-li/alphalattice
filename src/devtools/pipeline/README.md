# Delivery pipeline policy

## Process policy

- Run every static check that judges a change at commit, within the hook's 30-second budget, so its red costs seconds rather than a stage; `COLD_AT_STAGE` names the one left to the stage while its cache is cold (strict mypy). The hook records each passing step under the key of what it read (`REUSED_STEPS`: the tree, what the step compares it with, the target, the public ledger and the interpreter), in the Git common folder; a commit stage takes that verdict instead of running the step again, and runs every step with no such record. A merge commit runs no hook, a clone may have none, and the queue runs the composed commit's hook, so nothing else is skipped. Run the guard's steps side by side, one process each, the slowest first, `GUARD_WORKERS` at once.
- Keep the impact runner's file-per-process default; batch only through its audited safe list and revoke batching on a process-global framework import.
- Defer `post_merge_only` files from commit while retaining whole post-merge, candidate and nightly standing; a structural file must declare whole-tree scope and contain no fenced guard.
- Run a structural guard with no declared `reach` in the commit stage only when its own file changed, the tree a
  fast guard reads changed (`whole_tree_guards`), a slow guard's `selection_readers` row says the change touches
  what it reads, or a `structural_fence` row names its fence law (public-private, identity, evidence, secrets).
  Every slow whole-tree guard has its row, or is Large and runs in the nightly; the nightly runs every structural
  guard whole, and each run's `structural-moved.json` names the guards that moved.
- Run explicitly required Small and Medium nodes at declared limits, including quarantined nodes, and record them in the same receipt. Measure Large, end-to-end and native tests separately and run them in their scheduled tiers.

## Verified result reuse

- Workshop commit stages currently execute selected tests without result reuse.
- Use `--result-cache <directory>`, `--result-cache-bytes <positive bytes>` and `--cache off|record|shadow|on`; default off, with record and shadow executing every selected group.
- Reuse only a verified passing first attempt in explicit on mode; keep cache decisions, original bounded metadata and provenance in the receipt, and never publish a dirty stage.
- Disable the shared store on a shadow mismatch and repair it before reenabling reuse; require reviewed shadow and recall evidence before promotion.
- Never store Large, quarantined, end-to-end or native results.
