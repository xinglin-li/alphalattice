# Playpen Tests

Use this protocol for product and workshop tests.
Use it over a skill's generic test-writing advice.
Keep maintainer delivery mechanics in the resolved core and its triggered modules.
Use the existing policy and generators; this document grants no guard waiver.

## Rules for new tests

| Rule | Check |
| --- | --- |
| Declare Small, Medium, or Large with a size marker or registered requirement layer. | `devtools.testing.collection` resolves the declaration and `config/registries/test-sizes.json` owns the bounds. |
| Small: at most 1 s and 10 MB of temporary data; no subprocess or network. | `test_size.admission_limit` checks phases and `tmp_path` and new worker temporary-root entries; `testing.layer_contract` checks process and network use. Growth inside a pre-existing shared directory is not attributed; use a registered dataset. |
| Medium: at most 30 s and 500 MB of temporary data. | `test_size.admission_limit` checks phases and `tmp_path` and new worker temporary-root entries; growth inside a pre-existing shared directory is not attributed. |
| Large runs only in the nightly. | `test_size.large_contract` requires a `nightly_only` size rule for a new or changed Large test. |
| Use registered test datasets in place; do not copy read-only data. | Dataset registry check (in development); request a private writable copy only when the test must write. |
| A new test file names its one source module and goes at its mapped owner path. | `TEST-LAYOUT` in `scripts/check_test_shape.py` checks added files against `devtools.testing.layout`. |
| Keep the median to one new test function per card. | `TEST-GROWTH` reports the count in the fast guard; review judges the median, not a per-card refusal. |
| Whole-tree structural checks run post-merge by default. | `devtools.testing.sizes.moved` defers them; a changed holder or registered fence may still run in the commit stage. |

For a new non-structural file, put `TEST_SOURCE = "src/<owner package>/<module>.py"` or its
`scripts/<subpath>/<module>.py` path at module level and mark each test with its registered requirement. The target is
`tests/<layer>/<owner leaf>/<source subpackages>/test_<module>.py`.
For scripts, the source subpath starts after `scripts/`.
A whole-tree file under `tests/structural/` declares `TEST_SCOPE = "whole-tree"` instead.
Existing files and byte-identical moves keep their standing.

## The test protocol

1. Register each requirement in `config/registries/requirements.json` before its test.
   - Give it an id, a plain description, one package owner, a layer and seconds.
   - Follow the registered schema; record an exception's kinds, reason and decision date.
   - Ask the maintainer once for a new native/e2e journey, a test over 60 s or a near-duplicate. See [contributing](../CONTRIBUTING.md) and [governance](../GOVERNANCE.md).
   - Mark its function, class or module with `pytest.mark.proves("<id>")`; a journey may name several ids.
   - Keep retired rows with their reason; lower legacy counts with `python scripts/tests_registry.py shrink`.
   - Insert rows in id order; run `python scripts/tests_registry.py sort` to order the registry.
   - Refuse unknown/retired ids and excess legacy counts through the registry owner.
2. Run `python scripts/tests_registry.py find "<words>"` before adding a test.
   - Keep one holder file per requirement and layer; focused functions and parameter cases may share it.
   - Extend the named holder; register a missing requirement before adding its holder.
   - Carry a byte-identical move's standing; judge an edited copy as new.
   - Move all remaining holders together when a partial move would create a second file.
3. Choose the lowest layer that proves the requirement. Size is time; layer is what a test touches.
   - Keep the registered primary layer authoritative during migration.
   - Reject a conflicting explicit layer folder or mixed registered layers; separate focused tests.
4. Make the smallest regression change to the existing holder and prove it red on the base.
   - Name the requirement when requesting a regression test.
   - Add a function only for a newly registered requirement or to split an existing chained check.
5. Judge test proportion in review; keep every test tied to its requirement.
   - Use existing outcome metrics; do not invent a per-card allowance or exception trailer.
   - Read observed costs from canonical receipts; leave unavailable summaries unmeasured.
6. Delete redundant tests only with evidence and the admitted removal review.
   - Name covering tests; require no unique covered lines, no unique killed mutant and no recorded caught defect.
   - Inspect uncovered source as a possible dead-code candidate before adding a test.
   - Keep snapshots and frozen transcripts as provenance.
7. Class each red through the execution and receipt owners.
   - Use registered test data in place, with a private writable copy only for a test that must write.
   - Confirm it alone before attribution: passing within its bound is a flake; passing over it is a slow-test defect.
   - Give an admitted quarantine an owner and expiry; use the existing post-merge and recovery route.
   - Keep product defects, test defects, infrastructure and environment failures distinct.
   - Obtain valid evidence after a tool/environment failure; unknown is not a passing result.
8. Keep folder, domain and registered owner consistent.
   - Ask the maintainer about requirement admission, placement, Large tests and test health.

| Layer | Contract |
| --- | --- |
| Unit | Use temporary files only; no database, subprocess, socket, network or Host. |
| Integration | Use files, databases and an in-process Host; no network. |
| Contract | Run one port's suite against every adapter. |
| E2E | Use registered journeys, real composition and external readback; check that untouched files are byte-identical; keep doubles to model, network and clock. |
| Native | Use real host agents in admitted nightly and release lanes. |

- Apply layer contracts during setup, call and teardown, including registered legacy paths.
- Keep unit root isolation; admit only exact registered framework-fixture exceptions.
- Keep selected-binding and runtime checks active inside those exceptions.
- Use the reasoned backend catalog; permit ordinary compiled standard-library and numeric computation.
- Keep file operations under their audit, including native array file readers and writers.
- Share one scoped process audit hook with the recorder; release subscriptions on every exit.
- Treat these as Python contracts, not an operating-system sandbox.
- Keep unregistered legacy folders' existing behavior until migration.

## Boundary rules

1. Keep one lifecycle per file.
   - Mark real-evidence readers with `pytest.mark.real_evidence`.
   - Resolve named roots through `devtools.architecture.evidence_roots`.
   - Use the explicit evidence lane; let routed selection deselect those files and report `REQUIRED`.
2. Import shared setup by package name; never edit `sys.path` or use drive-letter imports.
   - Extract existing test-to-test dependencies instead of adding new ones.
3. Keep helpers beside their owner.
   - Route changed helpers to importers, conftests to their subtree and unparsable files to their directory.
4. Extend the owner's suite; require an independent acceptance boundary for a new package.
   - Put real-workspace builders in owner support modules.
5. Preserve numerical, authority, tamper, recovery, fail-closed, historical-readback, replay and Holdout/OOS assertions and match strings.
   - Change their setup only with equal or stricter proof.
6. Keep automated pytest assets under `tests/`; do not add them under `case-study/`.
   - Move a runnable example's regression or acceptance contract into its owning test package.

## Writing a test

1. State an unmarked test's requirement in one plain sentence of at most three docstring lines.
   - Let a `proves` marker use its registry sentence; keep further reasoning in the change record.
   - Use public behavior words rather than internal labels.
2. Extend or replace the holder returned by the registry search.
3. Drive the real producer's output through its real consumer at the owner's seam.
   - Read source only in `tests/structural`; test behavior elsewhere.
   - Prefer a real component; make a necessary double match the real structure.
   - Assert behavior rather than the mock itself or a private helper's incidental layout.
4. Check declared word keys and facts; avoid snapshots, markup and whole-sentence pins.
   - Check translations by key and their difference from English.
5. Apply the parameter rule in [Tests are cheap by construction](#tests-are-cheap-by-construction).
6. Apply its shared-setup rule; keep builders in owner support.
7. Drive the clock or supervisor step; do not sleep for time to pass.
8. Check Workbench behavior once in the Node harness; reserve a browser for real layout, width, focus, scrolling or motion.
9. Keep files within the existing test-shape size ratchet; an oversized file may only shrink.
10. Judge test/source proportion by the protocol's outcome rule; use registered bounds and Large reasons.
11. Correct or delete obsolete tests; never loosen them.
   - Name the covering test when deleting one.
12. Prove a new regression red on the base; name a base-passing addition as a specific control.
   - Treat a wording/private-layout change detector as a redesign or deletion candidate, not a requirement.

Use `scripts/check_test_shape.py` through the normal fast guard.
Read numeric shape limits from that owner; this protocol does not duplicate them.
Require an agent route or an existing named UI-only exception for UI writes; refuse UI-origin checks that block the person relay.

## Tests are cheap by construction

- Parameterize a dimension only when it is the requirement; multiply only the work that reads it.
- Move parameter-independent checks out of the parametrized function.
- Build an expensive environment, export, history, workspace, Host or model once in a shared module/session fixture.
- Share setup read-only; isolate each case's mutations.
- Give each function one behavior; split chained checks so workers can run them in parallel.
- Follow the Small, Medium and Large bounds and declarations in [Rules for new tests](#rules-for-new-tests).
- Measure own setup, enclosing call and teardown through the receipt producer; never use double-counted subtest time.
- Include cold first-use fixture setup when a node may run first; a later warm-only JUnit time does not establish its size.
- Confirm an overage through the clean-solo/load route; keep missing, malformed or aborted proof unmeasured.
- Run new or changed Large tests only in the nightly.
- Read watchdog settings from `config/registries/test-sizes.json`; the admission bounds are separate from the watchdog.
- Report touched-node durations before and after; explain slowdown without weakening its check.

## Running tests

- Use the locked environment and select touched or failed nodes, or changed test files.
- Keep tests offline with `ALPHALATTICE_NETWORK_DISABLED=1`.
- Let the managed runner own temporary roots, reports, plugins and workers.
- Keep first-use discovery, workspace reads, QA launches and golden caches inside their validated scope.
- Use the owning checkout for source paths and offered source commands.
- Follow applicable maintainer instructions for managed output, heavy admission and READY.
- Reuse unchanged passing proof; rerun changed or failed nodes.
- Use normal stage producers for steps a failed stage never reached.
- Keep mandatory files and declared reach in selection; a size edit alone cannot defer them.
- Use `scripts/check_playpen.py --evidence` for the admitted daily or affected-owner evidence run.
- Declare structural reach in `config/registries/test-sizes.json`; undeclared structural checks stay whole.
- Make `per_file` scans read Python through `scanned()` or `in_scope()` in `tests/structural/source_shape_samples.py`.
- Run a scoped scan whole when its own file or a declared input changes; refuse reads outside the changed scope.
- Declare cross-file inputs; use `+*` for additions or deletions.
- Keep nightly and candidate structural checks whole.
- Take timing from the owning gate's reports; do not repeat a run solely for another timing.

## Packages

- Keep deterministic regression and acceptance assets in the `tests` package tree with `__init__.py` files.
- Keep runnable learning/framework cases in `case-study/`, performance scripts in `benchmark/` and opt-in probes in `probe/`.
- Preserve fixture scope, instances and consumer groups when changing placement.
- Import shared methodology session fixtures by name from `researcher_methodology_surface/session_fixtures.py`.
- Keep `researcher_methodology_extension` external-catalog cases isolated from shipped-catalog cases.
- Leave historical paths in historical records.

For process policy and verified result reuse, see the [pipeline policy](../src/devtools/pipeline/README.md).
