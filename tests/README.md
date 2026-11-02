# Playpen Tests

Date: 2026-10-08; the test protocol added 2026-10-24.

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

## The test protocol

Agents write most of these tests, and agents like writing tests. Without a protocol
a suite only grows, as this one did to 236,583 lines, 57% of source, by 2026-10-23.
This protocol says what a test may be, where it lives, what it may touch, when it
may be added and when it must go. It overrides any other instruction on writing
tests, a skill's included: a skill's "write a failing test first" means rule 4's
smallest red change to the holder. Each rule names what enforces it; a rule marked
*from* a card is held by review until that card lands.

1. **Every test proves a registered requirement.** A requirement is a row in
   the requirements registry, which the registry's cards add: an id, one plain sentence that is its
   reason, the owner package, a layer and a time budget. A test names the ids it
   proves with `@pytest.mark.proves("<id>")`. Its layer, owner, size and time are
   derived from the row, never written by hand.
   - Admission is by exception. A row that passes the checks (owner, sentence,
     layer, budget, no near-duplicate in its owner) is admitted at once.
   - A new end-to-end or native journey, a test over 60 s, a budget overrun or a
     near-duplicate needs the test manager's decision, asked once. The decision is
     copied into the row.
   - Existing tests are counted per file in a list beside that registry, and the
     count only falls.
   - A row in config/registries/requirements.json holds its id
     (`<owner's last segment>.<words>`), a `description` of one sentence of 5 to 40
     plain public words, its `owner` (a package that owns a noun in the owner map),
     its `layer` (unit, integration, contract, e2e or native; a layer's limit is
     test-sizes.json's for its size) and its `seconds`. A decided exception is recorded
     as `"exception": {"kinds": [...], "why": "...", "decided": "<date>: <decision>"}`.
     A row is never removed: it is retired with `"retired": "<date>: <why>"`.
   - The marker goes on a function, a class or a module's `pytestmark`, and a journey
     may name several ids. A marked test needs no docstring sentence: its row holds it.
   - Held by the fast guard's TEST-REGISTRY step (each new test and each added or
     edited row), and in a gate lane by collection, which refuses an unknown or retired
     id and a file past its count in the private legacy count list.
     `python scripts/tests_registry.py shrink` lowers the counts.
2. **Find before you write.** `python scripts/tests_registry.py find "<words>"` returns the
   matching requirements and the tests that hold them, requirements first. One holder per requirement
   per layer: a new case is a parameter or one assertion in that holder.
   - A refused new test shows that search's result for its name and docstring, then
     the next step: add the case to the holder it names (an unregistered holder
     registers its requirement and takes the marker), or register the row first and
     mark the test.
3. **The lowest layer that can prove it.** The folder is the layer, with the
   owning domain below it.
   - **unit:** no file outside tmp, no DuckDB, subprocess, network or Host; 1 s.
   - **integration:** files, DuckDB and an in-process Host; no network; 60 s.
   - **contract:** one port's suite, run on every adapter of that port.
   - **e2e:** only the registered journeys. Each boots the real composition,
     re-reads the world from outside (files, ledger rows, the page) rather than
     trusting the agent's own report, checks that untouched files are
     byte-identical, and keeps test doubles to the model, the network and the
     clock.
   - **native:** real host agents, in the nightly smoke and the release.
   - *From the layout move* (the folders) *and the protocol's guards* (each
     layer's contract enforced while the test runs).
4. **Test-first by this protocol.** The red is the holder's smallest change, and
   it fails on the card's base (writing rule 12).
   - A bug extends the holder of the requirement it broke, or registers the
     requirement nobody had.
   - A reviewer who asks for a test names the requirement.
   - In force now.
5. **The author sees the cost.** A card's net test lines stay within max(20, 30%
   of its net source), with at most three new test functions, each for a new
   requirement named in the change's record.
   - An exception is a `Test-Budget-Exception: <ids>` trailer the test manager agreed.
   - The fast guard holds the lines now.
   - *From the pipeline's receipts:* each card's receipt also carries its
     test-seconds, against each owner's seconds budget, which the test manager
     sets from the inventory.
6. **Deletion is a normal change.** A requirement's holders change or go with it.
   - A test that is deleted on evidence, without anyone's permission, is one whose
     covered lines are a subset of other tests', that kills no mutant another test
     does not, and that has caught nothing. The change names the covering tests.
   - An uncovered source line is first a candidate for deletion as dead code, not
     a missing test.
   - Snapshots and frozen transcripts are provenance, never gates.
   - *The inventory supplies the evidence.*
7. **Deterministic, and every red classed.** No real clock where the product has
   a clock seam, no sleeps and no network; data comes from the support factories.
   A red is one of these:
   - a flake: it passes alone, within its layer's time. It is quarantined with an
     owner and an expiry (`config/registries/test-sizes.json`) and runs post-merge
     only;
   - a slow test: it passes alone, but only over its time. That is a test defect;
   - a test defect;
   - a product defect;
   - infrastructure (a tool error);
   - the environment.
   Only product and test defects block or revert. The quarantine list is in
   force; the classifier comes *from the protocol's guards*.
8. **Owned and budgeted.** Folder, domain and owner match.
   - The test manager, a maintainer role, owns the tests. It admits the
     exceptions, leads deletions, runs the daily flake duty and reports test
     health daily.
   - A question about a test's place or budget goes to the test manager.

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

The protocol above governs; these rules are its detail. A test is kept only while it holds a requirement nothing else holds. The fast
guard (`scripts/check_test_shape.py`, run by `scripts/check_playpen.py --staged
--fast`) refuses rules 1, 3, 7 and 9, a sentence pin of five or more words (rule 4)
and the docstring bound in rule 1 on what a change adds; review holds the rest.
Source growth past 100 KiB or 120 function lines/20 decisions, copied eight-line blocks and added performance lints are refused by the fast guard; owner count-budget excess is a COUNT STOP, and a higher pin needs maintainer approval.
The Workbench parity guard refuses a write without an agent route, an added named UI-only exception, or a UI-origin check that blocks the person's relay.

1. A test holds one requirement: a behaviour a user or a consumer relies on, a
   regression, a tamper or recovery case, or a boundary nothing else covers. An
   unmarked test's docstring states that requirement in one sentence, at most three
   lines, in words rather than internal card or ledger ids; the guard refuses new
   functions without it. A test marked `@pytest.mark.proves("<id>")` takes the
   sentence from its registered row instead (protocol rule 1).
   The change's record carries any further reasoning.
2. Before writing one, run `python scripts/tests_registry.py find "<words>"` (protocol
   rule 2) and extend or replace the holder it names. A requirement is tested once, in its owner's suite, not again in
   another file or another harness. A fix extends the test that should have
   caught the defect; only a new requirement gets a new test.
   The fast guard measures each card from its merge base with develop. Net test lines stay
   within max(20, 30% of net source), and the card adds at most 3 test functions; an exception is a
   `Test-Budget-Exception: <ids>` trailer the test manager agreed (protocol rule 5).
3. Test at the owner's seam: drive the producer's real output through the real
   consumer. Only `tests/structural` reads source text; elsewhere a test does not
   call `inspect.getsource` or test a private helper whose owner's public call
   shows the behaviour. Prefer the real component to a mock. A mock is not itself
   asserted on, and it mirrors the real structure completely where one is needed.
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
10. Across a merge set, test lines do not grow more than source lines; the
    maintainer's merge stops a set that does. A single card is bound by rule 2's
    per-card budget instead. Only the test manager admits an
    exception, such as a small fix whose regression case outweighs it, and its
    reason is recorded. A test over 10 s names why
    in its docstring; one over 60 s is a Large exception the test manager decides
    (protocol rule 1), with a specific registry reason; the declaration ratchet refuses growth.
    Pytest-timeout enforces the registered bound over setup, call and teardown, naming overruns.
11. An outdated test is deleted or changed, never loosened. A deleted test names
    its covering test in the change's record; protected classes (boundary rule 5)
    change only how they are built.
12. A new or changed test fails on the change's base, without its product
    change, and the maintainer's merge checks this. One that passes there is a
    declared control for a named behaviour, or it is removed. A test that only an
    intended change can break (a pinned wording, a private layout) is a change
    detector, not a requirement.

## Running tests

The shared fixture confines first-use discovery without changing the process cwd. Project and
workspace reads outside the declared temporary scope fail by name. QA Host launches and default
golden caches use that scope; consumers read the producer's scoped receipt. Existing managed
producers pass their validated run root explicitly. Source paths and offered source commands
use the owning checkout.

1. While working, run the nodes you changed or that fail, with one or two
   workers, plus a test file you changed. Never a directory or the whole suite.
2. The merge gate has two tiers (`tools/lead_check.py --tier pre|post`).
   - The pre tier blocks; its target is about 15 minutes at four workers, unmeasured until its
     first runs. It always runs the fast guard over the whole set, the registry's mandatory
     files, tests/structural and the changed tests, naming any overrun of the budget. The tests
     that import a changed module then join, rarest import first, within 3,600 estimated
     test-seconds (60 s for a file without a recorded duration).
   - A line runs the same pre tier on its card before READY, so defects are found in parallel.
   - The post tier runs every affected file whole once, after the merge; its `--durations`
     report is the test timing. No separate timing run.
3. A passing test is not rerun unchanged. After a fix, rerun the failed node; the
   integration set reruns its file.
4. The real-evidence lane (`scripts/check_playpen.py --evidence`) runs once a day
   and for a change on the data update, deferral or activation owners; the
   integration set deselects it otherwise.
5. Tests run offline (`ALPHALATTICE_NETWORK_DISABLED=1`), and their temporary data
   is released when the work is handed back.
6. `tests/structural` reads source as text, which import selection cannot see, so
   every structural test runs in every lane unless it declares its reach in
   `config/registries/test-sizes.json`.
   - A `per_file` scan reads repository Python only through `scanned()` or
     `in_scope()` (tests/structural/source_shape_samples.py). The fast guard and the
     merge lane run it over the changed files only. A read outside that scope fails
     the test, and a change to the test's own file or a declared input runs it whole.
   - A cross-file check declares `inputs`, and the merge lane runs it when one of
     them changes. `+*` matches any added or deleted file.
   - Nightly and candidate runs keep every structural test whole, so a wrong
     declaration surfaces within a day.

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
