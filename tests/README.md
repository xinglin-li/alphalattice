# Playpen Tests

Date: 2026-10-08

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

## Writing a test

A test is kept only while it holds a requirement nothing else holds. The fast
guard (`scripts/check_test_shape.py`, run by `scripts/check_playpen.py --staged
--fast`) refuses rules 3, 7 and 9, a sentence pin of five or more words (rule 4)
and the docstring bound in rule 1 on what a change adds; review holds the rest.
Source growth past 100 KiB or 120 function lines/20 decisions, copied eight-line blocks and added performance lints are refused by the fast guard; owner count-budget excess is a COUNT STOP, and a higher pin needs maintainer approval.
The Workbench parity guard refuses a write without an agent route, an added named UI-only exception, or a UI-origin check that blocks the person's relay.

1. A test holds one requirement: a behaviour a user or a consumer relies on, a
   regression, a tamper or recovery case, or a boundary nothing else covers. Its
   docstring states that requirement in one sentence, at most three lines, in
   words rather than internal card or ledger ids; the reasoning belongs in the
   change's record.
2. Before writing one, search for a test of the same requirement and extend or
   replace it. A requirement is tested once, in its owner's suite, not again in
   another file or another harness. A fix extends the test that should have
   caught the defect; only a new requirement gets a new test.
3. Test at the owner's seam: drive the producer's real output through the real
   consumer. Only `tests/structural` reads source text; elsewhere a test does not
   call `inspect.getsource` or test a private helper whose owner's public call
   shows the behaviour.
4. Assert the requirement, not the incidental. No snapshot, markup or
   whole-sentence pins outside the declared door words: check words by their key
   and the facts they carry (counts, dates, codes), and a Chinese translation by
   its key and that it differs from the English.
5. A parameter dimension (width, language, theme, class) appears only when that
   dimension is the requirement.
6. Expensive setup (a built workspace, a started Host, a browser) is shared at
   module or session scope unless the test mutates it; a new real-workspace build
   belongs in the owner's support module (boundary rule 4).
7. A test does not sleep for time to pass; it drives the clock or the
   supervisor's step.
8. A Workbench behaviour is checked once, in the Node harness. A real-browser
   class checks only what needs real layout: widths, focus, scrolling and motion.
9. A test file stays under 100 KB, and one already over it only shrinks.
10. A change's test lines do not grow more than its source lines unless its
    record names the requirement each new test holds. A test over 10 s names why
    in its docstring; one over 60 s needs the maintainer's agreement.
11. An outdated test is deleted or changed, never loosened. A deleted test names
    its covering test in the change's record; protected classes (boundary rule 5)
    change only how they are built.

## Running tests

1. While working, run the nodes you changed or that fail, with one or two
   workers, plus a test file you changed. Never a directory or the whole suite.
2. Every affected file runs whole once, in the integration set before a merge;
   its `--durations` report is the test timing. No separate timing run.
3. A passing test is not rerun unchanged. After a fix, rerun the failed node; the
   integration set reruns its file.
4. The real-evidence lane (`scripts/check_playpen.py --evidence`) runs once a day
   and for a change on the data update, deferral or activation owners; the
   integration set deselects it otherwise.
5. Tests run offline (`ALPHALATTICE_NETWORK_DISABLED=1`), and their temporary data
   is released when the work is handed back.

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
