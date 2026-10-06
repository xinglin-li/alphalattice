# Playpen Tests

Date: 2026-09-08

`tests/` holds the deterministic product regression, authority acceptance and
external-consumer acceptance suites that `scripts/check_playpen.py` runs. It is
a package tree: every directory carries an `__init__.py`, a module is named
`tests.<package>.<module>`, and shared setup is imported by that name. The
runnable learning cases and exact-version framework probes stay under
`case-study/`; performance scripts live under `benchmark/` and opt-in live or
real-workspace probes under `probe/`. Paths moved here at the Phase 2 checkpoint
of the maintainers' test asset governance record;
historical records name the old `case-study/<directory>/` locations and are not
rewritten.

## Boundary rules

1. One lifecycle per file. A test that reads a dogfood evidence root carries
   `@pytest.mark.real_evidence`, resolves the root by name through
   `devtools.architecture.evidence_roots` (declared in
   the private evidence-root declaration), and runs in the explicit lane
   (`scripts/check_playpen.py --evidence`). The routed gate deselects it and
   names its file as `REQUIRED`.
2. Tests import shared setup by package name, never by editing `sys.path` and
   never through a drive-letter path. `src/` and `scripts/` are on the pytest
   `pythonpath`; a test module that another test still imports is a Phase 3
   extraction item, not a pattern to extend.
3. A helper module lives beside the owner that consumes it; the runner routes a
   changed helper to its importers by dotted name, a changed `conftest.py` to
   its whole subtree, and an unparsable file to its directory.
4. A new test extends its owner's suite. A new package needs an independent
   acceptance boundary or runnable question; a new real-workspace build belongs
   in the owner's support module, not in a test body.
5. Protected classes (numerical goldens, authority, tamper, recovery,
   fail-closed, historical readback, replay, Holdout/OOS firewalls) change only
   how they are built; their assertions and `match` strings stay equal or
   stricter.
6. Automated pytest assets belong only under `tests/`: do not add `test_*.py`
   or `conftest.py` beneath `case-study/`. A runnable example may have its own
   ordinary assertions, but when it becomes a regression or acceptance
   contract, move that contract into its owning test package here.

## Packages

One package per former case-study directory, named by replacing hyphens with
underscores; `structural-guards` became `structural`, and the two thin loaders
over documented runnable cases sit in `cases/`. `researcher_methodology_extension`
holds the two methodology cases that install a catalog revision: they need the
extension and external workspaces, which nothing else does, so they run in
their own process while the shipped-catalog cases run in parallel. The shared
session fixtures are defined once in
`researcher_methodology_surface/session_fixtures.py` and imported by name into
each package's `conftest.py`.

## Process policy

The impact runner runs one file per process by default, because Agent and model
frameworks are process-global. A package may run as one process only after an
audit shows its one-process result equals its per-file result and it imports no
such framework; the allow-list is `_BATCH_SAFE_TEST_DIRS` in
`scripts/check_alpha_verification_impact.py`, and the runner revokes an entry
by itself the moment a process-global import appears in the package.
