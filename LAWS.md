# The AlphaLattice Laws
Date: 2026-09-26

The rules for the code outside the Workbench's UI, in one place. From 2026-09-26 this file is their one
owner: a rule is added, changed or retired here, and every card cites the laws it keeps. It gathers the
maintainer's working agreement, the design decisions and the lessons of the project's development. The
Workbench has its own book, indexed as chapter UI at the end and not copied.

Look a law up before writing: the chapter of what you are about to change says where it belongs and what
must hold. Each law says what holds, why where the reason is not obvious, and what holds it:

- `gate: <step>`, a step of `scripts/check_playpen.py` (the hook runs its fast steps on every commit);
- `test: <path>::<name>`, a public test that fails when the law is broken;
- `private-test: <name>`, a named check retained in the private verification suite;
- `config: <path>`, a file whose content enforces it;
- `hook`, the pre-commit hook itself; `U0`, the reopen probe
  (`scripts/u0_probe.py`), run by every card that moves an identity;
- `review`, when no check holds it yet. Those laws are listed at the end; each becomes a check where it
  can, and the list only shrinks.

A holder may check part of a law. What the code does against a law, and what no holder checks yet, is
written in the law as *Not yet held*, with its rows in the convergence list and the card that closes
it: a law states the rule, never the rule as if the code already kept it.

The maintainer is the project's creator, who decides its scope, its risk and its releases. A card is
one unit of development work on its own branch. An id in parentheses, a convergence-list row (V662),
a Workbench change-ledger row (U70) or a card (AX10), points into the maintainers' records, which
are kept privately; each law states its rule without them.

The private law-holder check refuses a law
without a holder, a duplicate id, a holder that names a test, gate step or file that does not exist,
and a review list that is not the book's.

## ID. Identity

**ID1. Identity follows meaning.** A change that keeps what a study computes records its successor in
`config/identity-successors.json`, which `is_current` reads; a change that moves a number is a new
identity, and the studies it moves become historical: they read back, they are not current. A role
whose code retires leaves the roles table for its `retired` list, naming the card and why; its recorded
moves stay as history and nothing installs or compares it again (RT). *Why:* an
identity is the promise that the same inputs and method give the same numbers; a silent move breaks every
reuse, replay and comparison. A value bound where it is compared by equality (a Program's
authorities, a captured source, each Factor value identity and the source-availability catalog a
Feature catalog binds) is held at the value its recorded moves lead from (`recorded_origin`),
as the switch table holds a byte value; its role reads the value before the hold. *Held by:* `test: tests/structural/test_source_identity.py::test_a_recorded_move_names_the_installed_authority_for_its_role_only`,
`test: tests/structural/test_source_identity.py::test_the_committed_moves_read_and_each_predecessor_moves_once_per_role`,
`test: tests/structural/test_source_identity.py::test_every_recorded_role_is_in_the_roles_table_and_resolves`,
`gate: identity-closures`.

**ID2. A move is proved, never assumed.** A successor carries its evidence: the identity readout before
and after, where only the recorded roles move; U0 on both corpora, where no reopened object changes its
verdict; and the controls that must not move. An expected hash, count or baseline is never updated
blindly: its semantic cause is found first. *Not yet held:* the readout shows the identity roles, not
the byte closures that read the 349 files GN's scan lists (V231, V233); and U0's corpora hold few
studies whose replay succeeds (V234), so U0 proves reads more than replays until U0R. *Held by:*
`U0`, review.

**ID3. A closure holds what decides the number, and nothing else.** A study's identity tracks the files
whose bytes can move its numbers, never a platform or build file, and a loader that turns bytes into
numbers is inside it. Since R1 a study's closure is its rule closure: the import closure of its entries inside the
number-deciding packages (`config/identity-roles.json`), with the entry files it lists by name whatever
their package, hashed as syntax without docstrings, comments or execution spans: a span measures
where a run's time goes and decides no number, so a module reads as if each literal `with span(...)`
were its body (`kernel/shared_kernel/spans.py`). *Not yet held:* 50 identity sites
still hash raw source bytes and 4 source text, each with the card IS names for it
(the private identity inventory); the schema prose left with SH (V98, SC3). *Held by:*
`test: tests/structural/test_source_identity.py::test_study_identities_track_no_platform_or_build_file`,
`test: tests/structural/test_source_identity.py::test_moves_this_build_cannot_read_are_refused_not_ignored`,
`test: tests/structural/test_source_identity.py::test_a_span_moves_no_identity_and_any_other_form_reads_as_written`.

**ID4. The Risk recipe and the Risk numerical closure move only by a recorded rotation.** The
default covariance recipe seals to the value Risk studies, and the eight legacy surfaces the original
workspace keeps, rest on; the Risk numerical closure holds the code that can change a covariance number
and never what measures a run. A change to either moves the Risk identities by the rule and takes
recorded successors like any other (ID1, ID2). The frozen pin that held Risk identities before retired
with M01 (RT, 2026-09-29); the legacy surfaces stay on disk as history (DA1). *Held by:*
`test: tests/researcher_methodology_surface/test_risk_development_surface.py::test_the_default_covariance_recipe_seal_does_not_move`,
`test: tests/researcher_methodology_surface/test_risk_development_surface.py::test_runtime_measurement_cannot_move_a_covariance_identity`.

**ID5. Only the maintainer edits identity records.** The successor records and the closure lists
are changed by the maintainer's own cards only, each with ID2's evidence. *Why:* one writer per identity keeps every
move reviewable in one history. *Held by:* review.

**ID6. The environment is provenance, not identity.** The interpreter, the installed library versions,
the dependency declarations and lock, and thread counts are recorded beside a result through one
helper and never hashed into an identity while the product is developed; a cache whose content they
decide (an embedding index) keeps them as a rebuild key only. Whether a change of environment moves a
number is shown by recomputation: the owners' numerical canaries (PA3), and at the release the corpora's
studies recomputed under the release environment, which is then locked and recorded. U0 on both
corpora proves that what was sealed still reads and is reused, and names each read whose numbers
moved (V277): a change in what a read derives, never a recomputation. *Why:* dependencies are still being optimised; binding them made adding
or removing a package move identities and demand successors with no numerical cause (the
maintainer, 2026-09-26: dependencies are not locked while they are still being optimised). *Not yet held:* the holder below checks
who reads the environment, not where its value is hashed. The canaries cover thread counts only,
and the release's recomputation is phase 7's (V277). *Held by:*
`test: tests/structural/test_source_identity.py::test_the_environment_is_read_by_one_helper_and_the_named_keys`,
which admits `kernel/shared_kernel/environment.py` and names every other reader and why.

**ID7. An agent's judgment is identified by its content.** What an agent submits through the seam is
identified by the submission's content and the digest of the bundle it answered; the role card's
digest, the model, the reasoning effort and the host are recorded beside it as provenance, as the
environment is (ID6). No identity binds prose: rewording a role card or a Skill makes no judgment new,
and a different answer does. *Why:* role cards and Skills are rewritten as the agents improve, and
binding them would retire every earlier judgment for a change of words; which model judges better is
measured apart, by the agent evaluation's planted-majors yardstick (the maintainer, 2026-09-27). *Not yet held:* an external
submission binds its answer and what it answers (the dossier, the policy, the schema), not the
bundle's digest; and the role card's digest is not recorded. Each answer records the host and
the lead's session that submitted it, its author `NOT_OBSERVED`: a request names its session and
not its subagent, and the Host reads no hook, message or native start to name which of the
lead's specialists wrote it, so no answer is credited to another specialist and none by a guess
by time (V555). An answer recorded earlier with `HOOK` credit keeps it as recorded; nothing new
writes it (V677). *Held by:*
`test: tests/alternative_evidence_desk/test_evidence_review_http_route.py::test_no_analyst_is_credited_with_another_bundles_answer`,
`test: tests/portfolio_strategy_lab/test_accepted_answer_conversation.py::test_an_answer_of_any_recorded_basis_is_the_leads_fact_without_child_credit`,
`private-test: test_existing_answer_contracts_publish_once_as_the_leads_product_fact`.

**ID8. A consumer binds an upstream result's content, not its code.** A role's closure lists its
owner's code as entries; a result it reads from another owner (a Panel, a Foundation, Alpha scores) is
bound by that result's content hash, which the result's own identity already ties to the code that
made it. Which roles an edit to each area can move is a table the gate holds, and a change that
reaches further is refused, naming the import that did it. *Why:* measured on 2026-09-27, an edit to
one factor family's formula moved 12 of the 20 identity roles, 2 of them the Feature materialization
that computes it; each false reach marks results not current that nothing changed, and exploring
becomes a chain of recomputation. Code that moves to another area is recorded under its new owner,
with its reason. *Not yet held:* byte closures are not in the table (W8, W9b), and the live scoring
family's reach of the factor formulas is a true one, kept (V229); the Evidence analysis policy's
closure misses `analysis/operations.py`, which `packet.py` imports outside the number-deciding
packages (V274, W9b). *Held by:*
`test: tests/structural/test_source_identity.py::test_an_area_reaching_a_role_outside_its_row_is_named_with_its_import`,
and the gate's reach step (GB, `--reach-check`).

**ID9. A new model is its own identity.** The Host loads an agent's model by name, never by a
static import, so its module is its own closure: activating it adds a capability identity and
needs no successor, and a Program binds only the models its mandate admits (V118), so no other
identity moves. The declared adapter, its declaration and its search axes are in the number
closures; the contract suite and the scaffold are not, so improving them moves nothing.
*Held by:* `gate: identity-closures`,
`test: tests/alpha_research/test_model_extension.py::test_the_installed_mandate_is_the_same_value_from_one_source_of_domains`.

## OW. Owners and packages

**OW1. One responsibility, one owner; a noun, one owner package.** Every package is registered in
`config/package-architecture.json` with its owner and area; a noun defined in two packages is a defect
until a merge removes it. Each noun's owner is in `config/registries/owners-map.json` (G1), and a
decision it makes -- a type its owner's contracts declare decided -- made outside the owner is refused
against the splits the tree held, a baseline that only shrinks (G2). *Not yet held:* a decision that
writes a table or a path another package owns (G2's storage and artifact checks). *Held by:*
`private-test: test_package_architecture_registry_is_complete_and_enforced`,
`test: tests/structural/test_registries.py::test_a_decision_is_refused_outside_its_owner_naming_the_owner`.

**OW2. Change the owner, not a parallel path.** A change replaces the owning path and removes what it
displaced in the same change. A compatibility path is kept only for a named current reader, with the
rule that retires it, and is listed in the convergence list. *Not yet held:* the holding test covers the retired owners it
lists; a parallel path elsewhere is found by review (V195: two entries reuse a Portfolio result by
different rules). *Held by:*
`private-test: test_retired_production_owners_are_absent`.

**OW3. No version-shaped names.** Modules, classes, functions, routes and capabilities carry stable
domain names, never `V2`, `Next`, `New`, `foo_v2` or a numeric suffix that keeps two implementations
alive. A `vN` names only a real wire, schema, checkpoint or pack compatibility identity with a current
reader and a retirement path. *Held by:* review.

**OW4. Every module has an owner and a consumer.** A new module or abstraction names its owning domain
and its current consumer; an empty package shell is refused; every product module has a named source of
proof. *Not yet held:* 5 modules have none (`config/internal-ownership-baseline.json`,
`module_proof_reach.no_executable_evidence`), pinned as a set: the Portfolio inputs' preparation, which
goes with W9b's rotation (W6's row), and four that code reads statically (the Factor publication
repository, the model validation contracts, the Risk calibration runtime, the Risk return surface
parity). *Held by:*
`private-test: test_tracked_python_packages_are_not_empty_owner_shells`,
`private-test: test_every_product_module_has_a_named_source_of_proof`.

**OW5. No cycles.** No module imports back into a module that imports it, and no two packages depend on
each other. *Not yet held:* the holding test checks packages (its baseline records no mutual
dependency); a cycle between modules of one package is not checked. *Held by:*
`private-test: test_internal_ownership_baseline_is_exact_and_non_growing`.

**OW6. A composition root composes.** `control/product_host` wires owners together and holds no domain
logic of its own; its domain parts move to their owners (O2). What it does hold is the agent's side of
the product, which makes it agent-native: what an
agent reads, what it may do next, how a refused request goes on and where the loop's state lives
(OP10 to OP13). "Composition only" moves the misplaced domain judgments out and keeps that side. Its modules are a baseline
(`config/registries/host-modules.json`): the DOMAIN ones the owner map moves out only go, and a new one
is registered as COMPOSITION or goes to its owner. *Held by:*
`test: tests/structural/test_registries.py::test_a_new_host_module_is_refused_unless_registered_as_composition`.

**OW7. Agents interpret, deterministic owners decide.** An Agent reads evidence and proposes;
calculation, validation, permissions, provenance, publication, activation and rollback belong to
deterministic owners, and model output is never authority. Statistical and optimisation code stays out
of Agent tools. *Held by:*
`private-test: test_host_decision_policies_name_no_agent_configuration`,
`private-test: test_experiment_compilers_are_agent_free_and_desk_owned`,
`private-test: test_the_agent_seam_is_a_domain_free_import_and_the_built_in_agent_is_gone`.

**OW8. The agent decides on evidence; the maintainer decides scope and risk.** An open question that
"the owner decides" is decided by the agent on its evidence and reported; product scope, risk and
anything irreversible or outward-facing are the maintainer's. *Held by:* review.

**OW9. The tree is its own dependency root.** Its interpreter, imports and hashes resolve inside this
checkout: its own `.venv` from its lock, no parent path, no junction to another tree's environment.
*Held by:*
`private-test: test_workspace_environment_and_identities_never_reach_the_outer_checkout`,
`private-test: test_no_product_source_reads_or_hashes_the_outer_checkout`.

**OW10. One rule, one function.** A rule two places read, a standing, a readiness, a grid of what may be submitted and run, is written once, by its owner, and every reader calls that function; a restatement is a defect even while it agrees. A second gate that guards another entry, permission or person stays, and reads the same rule. *Why:* restated rules drift when one side changes; the reviews of 2026-09-27 found the Evidence standing restated by the Host (V96), the readiness decided twice (V107) and the Portfolio grid written twice (V150), none caught by a check. A static check cannot tell a restated rule from a new one, so W2's near-duplicate comparison and the reviews find them. *Held by:* review.

**OW11. An agent's model enters by its declaration, and a person activates it.** An Alpha model
an agent adds is three files where G1's owner says, written by `model scaffold` from its YAML
declaration: the adapter under `capabilities/alpha_modeling/extensions/`, whose `fit` and
`predict` the agent writes (its route, recipe, search domain and fit protocol are the
declaration's), the declaration beside it, and its contract test. `model check` runs the contract
every adapter passes, the installed ones included: route and numerical binding, every point the
search axes' probes state admitted by the adapter's own refusals, the fit protocol, determinism
under the recipe's seed at one thread (the operator's threads are W10's canary's to prove), a
prediction that reads its row alone, and imports the lock holds. A library outside the lock is a
new dependency a person approves first, and nothing is written until then. `model sandbox` tries
the model on a copy of the workspace at rest, one Alpha study and U0 before and after, and
records the trial; a person activates the model for one workspace after a passed trial of the
same identity (`MODEL_ACTIVATE`, person-only) and deactivates it. The search axes are declared
and held by the contract; they enter no identity until V309 binds them. *Why:* the maintainer,
2026-09-30 (the model extension's design): an agent gets one way in, and a person decides what real research may
use. *Held by:*
`test: tests/alpha_research/test_model_extension.py::test_a_model_is_scaffolded_checked_and_passes_once_it_fits_and_predicts`,
`test: tests/alpha_research/test_model_extension.py::test_a_scaffold_refuses_a_new_dependency_and_an_existing_model`,
`test: tests/alpha_research/test_model_extension.py::test_a_person_activates_a_sandboxed_model_and_the_workspace_catalog_installs_it`,
`test: tests/alpha_research/test_model_extension.py::test_a_sandbox_tries_an_agents_model_on_a_workspace_at_rest`,
`test: tests/alpha_research/test_model_contract.py::test_an_installed_model_passes_its_contract`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_a_model_is_activated_by_a_person_only`.

**OW12. A strategy runs forward by a person's activation of its reviewed book.** An installed
research strategy's book replays history until a person activates a completed run of it over its
whole support (`STRATEGY_ACTIVATE`, person-only; under a first use's delegation its agent activates
the book once that book's review is published, OP19). The activation binds what the daily chain reads,
each built from the strategy's own research: each component's models, a lifecycle grant from the
training input its Alpha study read, so the study's fitted models are reused by their keys and
later quarters fit on the workspace's data through the horizon; the calibration's starting
observations, the installed authority's own lanes anchored at the book's activation formation;
and the book's opening state, its run's sealed last state, in a checkpoint that names the
person's activation. Nothing is fitted, replayed or built from final weights. Before a person activates, the strategy states its information cutoff, its first actionable session (the first planned entry strictly after the activation, decided before entry; earlier forward decisions are a causal replay, never out-of-sample) and the book's review standing (V589, V597, V603, V614). A book whose package
moved since it ran is refused, and activating a new run of it is the re-bind (V458); a person
stops it (`STRATEGY_DEACTIVATE`), and its history stays readable; past its horizon, a newer book
is activated. *Why:* a frozen strategy's live scores are a first-release must (the maintainer,
2026-10-02). *Held by:*
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_a_strategy_runs_forward_by_a_persons_activation_only`,
`test: tests/portfolio_strategy_lab/test_portfolio_decision_updates.py::test_a_persons_activation_names_its_time_and_book_and_a_qa_admission_neither`,
`test: tests/portfolio_strategy_lab/test_portfolio_decision_updates.py::test_a_calibrated_input_is_admitted_only_with_its_axis_activation`,
`test: tests/workspace_readiness/test_workspace_manifest_writes.py::test_a_research_installation_holds_a_persons_activation`,
`test: tests/portfolio_strategy_lab/test_strategy_activation.py::test_a_reviewed_research_book_runs_forward_when_a_person_activates_it`.

## OP. Operations and interfaces

**OP1. Every entry is a registered operation.** A route, a CLI command and an Agent tool are each an
operation of the operation registry; the gate refuses one that is not. The client answers four
commands itself and they are no operation: `serve`, `request` (any operation's document),
`schema show` (read from the registry's table, no Host) and `activity wait` (a waiter over the
feed's operation); the workspace, the operations, the activity feed, a declared event and the CPU
budget are operations like any other (V266). *Held by:*
`gate: operation-registry`.

**OP2. One CLI grammar, declared, on the Unix and GNU conventions, Git's for objects and context and the agent CLIs' for execution and continuation.** `alphalattice [global
options] <object> <action> [<id>] [--flag ...]`: the one instance a command acts on is a
positional operand, what modifies it an option; a document is `--file <path>`, `-` reads stdin
and `--` ends the options; help at every level; one envelope on stdout, progress on stderr, the
exit code the outcome. An action has one meaning in every object (`list`, `show`, `preview`,
`plan`, `run`, `verify`, ...), a flag one meaning in every command (`--file`, `--from`,
`--choices`, `--save-declaration`), and a selector names its object (`--task`, `--plan`); no
aliases. A reference is taken as the compact view shows it; every answer of the Host names its
context (the workspace, the goal counted, the agent session); a command never prompts, and long work answers
with a Task id that `task show`, `activity wait` and `task cancel` take; the top-level help
opens with the common path and lists every object in the research's order. The compact view is
every caller's default (a script asks `--view full`); a printed next command keeps its caller's
context, and a continuation keeps the references it bound. The operation registry declares each command (`GRAMMAR`: its object, action, purpose,
positional instance and document) and each field's flag, and the CLI parses from the table it
writes, `operations.json`, with the standard library, so a command costs a process start and not
the product's models (K1); a stale table is refused, and a text naming a command or flag the CLI
lacks is refused. The older client and its verbs are retired (2026-10-01,
A4). *Held by:* `gate: operation-registry`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_a_command_a_text_names_is_one_the_cli_has`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_the_grammar_reads_files_stdin_and_operands_as_gnu_does`.

**OP3. One answer envelope, exit by outcome.** Every command prints one envelope and exits 0 OK,
1 INVALID_INPUT, 2 REFUSED, 3 PENDING or 4 NO_HOST; `--output` holds the owner's full answer whatever its
status, and stdout is the compact view when saving. The compact view shows each hash and id by its
first twelve characters, links and paths whole, and the CLI reads a value of that shape back as the
one whole value the Host has answered with before anything is sent, refusing one that begins several
(V393); the Host reads only whole values. The compact view is one read whatever the answer
offers: its next requests keep a quarter of it, the owner's first in its order, `next_left`
counting the rest, and `--list-next` lists them a page at a time (V411). `--list-next` answers in
the one envelope (V261), and `--wait --output` saves the admission before it waits (V273).
*Held by:*
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_every_answer_is_one_envelope_and_exits_by_its_outcome`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_a_compact_answer_shows_references_short_and_they_can_be_sent_so`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_a_compact_answer_holds_its_next_requests_to_the_read`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_list_next_answers_in_the_one_envelope`,
`test: tests/portfolio_strategy_lab/test_status_follow.py::test_an_interrupted_wait_keeps_the_admitted_task_in_its_output`,
`test: tests/portfolio_strategy_lab/test_local_web_product.py::test_a_short_reference_is_read_back_as_the_one_value_it_begins`.

**OP4. A refusal is a code with words and a way on.** A refusal carries its owner's stable code, words a
person can act on, and the next request the owner accepts; never exception text or a path. Its words
are its owner's, else its code's own from one table (`refusal_words.json`), filled where a refusal
leaves the Host: at the operation door for a returned refusal, at the web layer for a raised one
(V449). Every web handler preserves a typed owner's refusal through the shared failure-code
reader, whatever its exception class; the web response and an observed operation's record retain
the owner's code. `handler_failed` is reserved for untyped unhandled faults (V679). A way on is a
request the Host accepts (an offered request, which the answer check holds)
or a person's act; never a retry of work the Host would reuse as sealed, which meets the same
refusal (V449: words that offered a rerun of a study whose sealed file was missing). A way on that runs a plan again after a stop resumes the same Task from the stage it stopped in, and is refused by that stop's code, with its retry time where it has one, until its cause clears; no newer plan or admitted Task waits behind an update that has not ended (V600, V601, V604). A failure
without a product code (a crash, or a validator's sentence) never reaches a caller as its own
words: the Host answers it as its fingerprint with words and a way on, and a test that meets one
fails unless it causes it on purpose (`untyped_failure`). *Held where a test reaches it:* the
answer check every test's operations pass refuses a refusal without words or a way on, and an
untyped failure; a registered code no test reaches is worded only where its owner or the table
words it. A document that fails its contract is answered in one located shape at every operation: the owner's code, the fields by path, each field's reason and the contract's words, never a value (`failure_codes.located_failure`, V248). *Held by:*
`test: tests/portfolio_strategy_lab/test_research_delivery.py::test_each_explained_refusal_keeps_its_code_and_asks_only_operations_the_host_accepts`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_a_contract_failure_answers_one_located_shape`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_a_refusal_leaves_with_words_and_a_way_on`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_every_door_whose_way_on_reruns_a_plan_resumes_its_stopped_task`,
`test: tests/portfolio_strategy_lab/test_local_web_product.py::test_every_web_handler_preserves_every_typed_owner_exception`,
`test: tests/portfolio_strategy_lab/test_local_web_product.py::test_activation_keeps_typed_refusals_through_its_owner_and_observer`,
`test: tests/portfolio_strategy_lab/test_local_web_product.py::test_a_raised_owner_refusal_keeps_its_transport_and_observer_code`,
`test: tests/portfolio_strategy_lab/test_local_web_product.py::test_a_genuine_fault_stays_a_fault_at_every_web_owner_entry`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_an_explicit_owner_code_survives_validation_without_its_private_detail`.
V679's observer regression checks the discovered typed cases through one observed operation;
it does not establish every route and observer combination.

**OP5. Offline by default.** Network and providers are off unless a workspace's control allows them
(`control/workspace_runtime/network_access.py`); `ALPHALATTICE_NETWORK_DISABLED=1` forces a process
offline, and every test, probe and product run uses it unless a card admits live access (a test that proves the person's switch clears it for itself, its own guards keeping every read on the machine, V600); one
kernel function reads the switch (`kernel/shared_kernel/environment.py` `offline`), and anything but
the control's two mappings reads as closed: a person's (version 1), and a first-use goal's
delegation's (version 2), which carries that delegation and its end and reads closed after it at
every read, with no write (OP19, V452). A research run is held offline while it runs,
whatever its workspace allows (`held_offline`, V116), so a network open for an update never
refuses a study. *Held by:*
`test: tests/workspace_maintenance/test_network_access.py::test_anything_but_a_version_one_mapping_with_a_boolean_reads_closed`,
`test: tests/workspace_maintenance/test_network_access.py::test_the_written_control_opens_and_closes_the_network`,
`test: tests/workspace_maintenance/test_network_access.py::test_a_delegated_setting_holds_until_its_end_with_no_write`,
`test: tests/researcher_methodology_surface/test_program_execution.py::test_a_run_is_held_offline_and_a_plan_over_budget_is_refused_at_plan`, review.

**OP6. A read never becomes work.** A readback, export or replay reopens what was sealed and refuses
rather than computes. A collection reads its independent items independently: one missing or refused
item leaves the others readable and names its own refusal with OP4's usable way on, never refusing the
whole collection (V633). Display readers of the registry's Task collection isolate an unreadable
canonical record; complete and control scans still refuse it, rather than treating the readable
subset as complete authority (V661). *Held by:* `U0` for sealed readback;
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_each_registered_collection_read_checks_item_refusal_routes`
for every registered collection's answer contract;
`test: tests/researcher_methodology_surface/test_local_web_factor_experiments.py::test_foundation_seal_consumption_readback_and_refusals`
for Foundation's missing-Task, item isolation, identical readback/export and usable controls;
`test: tests/portfolio_strategy_lab/test_local_web_product.py::test_task_record_collections_keep_readable_rows_and_offer_named_authority_routes`
for unreadable Task records in six display readers and strict complete/control scans;
the Result, History, Feature, Goal and Activity owner regressions for representative runtime
isolation, and owner review for the remaining registered readers. The registry-wide check uses
synthetic answers; it does not claim runtime failure injection for every owner.

Collection discovery reads metadata or the owner's summary, never every item's full object;
selection reads only the exact object and revision the person selected (V683, V688). A Task's
Result lookup uses its metadata and one exact Report, a Goal reference uses its exact reference
selector, and Foundation discovery uses its sealed summary. *Held by:*
`private-test: test_the_workbench_harnesses_pass`, whose collection, route and handoff probes
check these selected reads and reject mismatched references. These probes hold the named
Workbench readers, not every owner query. Page and intent lifetimes belong to the Workbench's
ST1, ST5, PG2 and ST10.

**OP7. Product words.** Product text never says "supplementary", "nice-to-have" or "today", and never
places the Evidence review as secondary. *Held by:* review.

**OP8. Configuration names no code and carries no resolved authority.** Committed configuration never
carries a resolved authority, and an authored document never names a Python module or callable. *Held by:*
`private-test: test_committed_configuration_never_carries_resolved_authority`,
`private-test: test_authored_documents_never_name_python_modules_or_callables`.

**OP9. The CLI and YAML operate everything.** The system uses YAML so that people and agents can both
drive it, and it must be easy to operate through the CLI together with YAML (the maintainer,
2026-09-26). Every
operation is reachable from `alphalattice <noun> <verb>`. A scalar selector (a Task id, a session) is a
flag; a structured input is a YAML file, attached to its field with `@<file>` or given as the whole
request with `alphalattice request --file <file>` (`-` reads stdin); a long document is never typed on the
command line. `--output` with `--format yaml` saves the answer as YAML, which `--from` reads, while stdout
stays the JSON envelope (OP3); a template or draft comes back as YAML to edit. A card
that adds or changes an operation keeps all of this, and its help names the `@<file>` form. *Why:* a
recipe holds too much for flags, and a file a person can read, comment and keep beside the command is
the interface both a person and an agent use well. *Held by:*
`test: tests/portfolio_strategy_lab/test_local_web_product.py::test_cli_document_reader_keeps_utf8_and_byte_limits_for_files_and_stdin`,
`test: tests/researcher_methodology_surface/test_experiment_authoring.py::test_yaml_compiles_to_an_immutable_program_with_exact_sessions`.

**OP10. A specialist reads what the Host prepared.** A specialist agent (the Analyst, the CRO, any role
the Host packs a bundle for) is given Markdown bundles the Host packs from verified owners, bounded by
size with one index, holding only what its role needs; never raw or dirty data, a database, an
artifact's JSON, a hash or an internal code. The main agent, who drives the CLI, reads the owners'
answers as the envelope gives them, their identities included (OP3, OP9), a compact view leaving out
what no decision of its own reads (a Task's `timing`, V280). Short aliases (S3, F2) stand for
sources and findings and the Host maps them back; what could not be covered is named, and anything
sealed from agents is checked by count before a bundle is handed over. *Why:* the agent's attention
is the scarce resource; the Analyst's material fell from 55,859 to 11,329 proxy tokens when the Host
packed it. Each shipped specialist card must be satisfiable
by its declared Host's actual tools. Bundle reads stay within listed files, writes within the
nominated answer file; stage cards retain their declared product CLI (V669). *Not yet held:*
the card validator compares instructions with a declared native-tool set and access phrases;
it does not invoke every tool on both Hosts. Live receipts hold only the observed card and Host
(V669, V677). *Held by:*
`private-test: test_shipped_configs_and_all_role_cards_parse_and_match_host_tools`;
the remaining bundle boundaries by review.

**OP11. An agent answers with judgment only.** Its answer schema holds judgment fields, never a hash,
a field the program can compute or a ritual confirmation; a subset is a valid answer and an empty
list is one; the Host binds the answer to what it answers and marks the rest not addressed. Whether a
task is complete is judged against its agreed scope and required evidence, never by the answer's
size: an empty list over materials read whole is a complete answer, and only required work left
undone makes it partial. Every shipped specialist card has a bound answer door. Evidence Analyst
and CRO retain their typed answer operations; Alpha, Data, Factor, Portfolio and Risk retain
their stage CLI and write one bounded generic answer file as their final action. The lead submits
the file through `AGENT_ANSWER_SUBMIT`; the Host checks its shape and bindings, not the truth
of its text (V698). Card and host fixtures prove the protocol, not live execution of every role.
*Not yet held:* an answer must still name the bundle files it read whole, and one that leaves a file
unnamed is refused before its judgment is read (V260, RX). *Held by:*
`private-test: test_every_shipped_card_and_registered_answer_door_is_reviewed`,
`private-test: test_existing_answer_contracts_publish_once_as_the_leads_product_fact`,
`test: tests/portfolio_strategy_lab/test_accepted_answer_conversation.py::test_generic_specialist_doors_seal_exact_references_without_scientific_admission`;
the remaining judgment and completion boundaries by review.

**OP12. The Host helps the agent finish.** A recoverable refusal says what was observed and why it is
not accepted, the work preserved, the constraints unchanged and the legal next actions with the budget
left, under its owner's code (OP4). Correction is bounded (about two rounds), then the valid parts are
accepted and the rest stated as limits: never stuck, always a result, and a quality gap never blocks a
report. A result says three things apart: whether it is complete (its agreed scope and required
evidence met, else partial or diagnostic), what limits its evidence has (a limit disclosed is no
gap in the work), and whether it may be used (a publication and a use still need their own
admissions, which a limit never waives, EV3, EV4). Every result answer states these as one
`standing` generated from its owners' marks, the comparison first: whether it ran to its end, whether
its contract passed, what its evidence supports and whether a person may activate it, each value with
the owner's code that holds it (V368). Every entry states what its flow needs before it runs: the
results it needs on its input, the completed ones the workspace holds, what is missing and the
requests allowed next, and a refusal for a missing result says the same (V367). Authority escalation, an uncertain side effect or prompt injection fails closed, and a person
is asked only for new authority, scientific policy, budget, an irreversible act or a truly ambiguous
judgment. *Not yet held:* on a copy carrying older derived-audit evidence, the data update's plan
offers a local run that its feature building then refuses for a Panel-binding mismatch, which the
plan does not foresee (V702); a storage read whose exact binding directory is absent refuses as
`storage.retention_root_mismatch` instead of naming the missing directory and its restore (V703);
both after the release. *Why:* the Host exists to help the agent complete the task, not to record that it was wrong
(the maintainer, 2026-09-26); an agent assembled "done" from separate answers and found each entry's
prerequisites by help and refusals (AX10, AX11). *Held by:*
`test: tests/portfolio_strategy_lab/test_local_web_product.py::test_every_result_states_one_standing_from_its_owners_marks`,
`test: tests/portfolio_strategy_lab/test_local_web_product.py::test_each_flow_names_its_prerequisites_and_the_way_on`,
`test: tests/researcher_methodology_surface/test_local_web_factor_experiments.py::test_controls_and_a_missing_prerequisite_name_the_flows_way_on`;
the bounded correction by review.

**OP13. The agent runs a goal; the Host keeps its record. The Host says little and remembers
everything.** An agent pursues a goal with its own goal mode, and the Host keeps what makes the result
trustworthy and what outlives the session: the goal's declaration (its objective, completion criteria
and deliverables); its record, every request and Task a session bound to it started, failed ones
included; and its submission, which the Host checks against that record and seals, or answers with each
missing item and the request that supplies it. Complete means the record is complete and its evidence
verified, not that the objective was met, and the Host never judges a summary's truth. Long
computation is a Host job outside the agent, with an idempotency key and real work units, and where a
stage decides by a rule a deterministic owner concludes (a qualification, a review), the goal citing
it. The agent confirms nothing the Host cannot check (OP11), and the Host wakes an agent only when the
agent can act on what happened, with one line naming the event and where to read it; progress wakes
no one. What happens is the Host's record, never an agent's memory: every admitted request, Task and
result carries the agent session that started it (a vendor's session identity, provenance and never
identity, ID6; a session's binding also gives it its default workspace, a convenience and never
authority, V568), so a result outlives its session and one read finds it, the latest results grouped
by session and goal, each with its read command; every Team event keeps the goal its
session held when the Host received it, so Team (by session), the goal (its conversation) and the
recent read (by time) are three reads of one record. *Why:* an agent's goal mode already plans and
persists, and what it lacks is a record the Host can check and a place to manage it; an agent's
tokens go to what it reads and how often it is woken, and a session can stop at any moment; a person
asking for "the result from earlier" gets it back from the record, whichever agent asks. *Held by:* review.

**OP14. The Host records what happened; what an agent read stays in its own context.** An agent's
tools are the Host's operations: each is a registered operation (OP1) whose pydantic request the Host
validates and refuses by name; the CLI is how an agent calls it, finding it when needed (`alphalattice
schema show`); a YAML file is the lasting form of a structured input or a saved answer (OP9); the JSON
envelope is the tool's answer (OP3). No second tool definition is written, and a native adapter (MCP),
if one is ever wanted, is generated from the same registry. The Host records what changes state
durably at its owner (a Task, a receipt, a publication, the goal's record), with the agent session as
provenance (OP13); each observed operation (a preview, a submission, a confirmation, a Task command)
as it happens, in the activity ledger, with its caller, provenance, the fields it named, its outcome
and failure code, bounded by the ledger's size budget (a request made under a goal also joins the
goal's durable record); and every refusal, reads included, counted durably by day, operation, code,
caller kind and agent vendor, never by session, request or text (`alphalattice activity refusals`). It
never records a read's answer, which the reader holds, nor an agent's reasoning, prompt, context or the
text it read: the activity keeps typed codes and references only. *Why:* the system keeps one place that records the calls agents make to it (the
maintainer, 2026-09-28). An agent's runtime keeps what it saw,
privately and for one session; the Host keeps what happened, for every session and agent. A refusal
is the plainest sign of where an agent fails, and it left with the transient rows. *Held by:*
`test: tests/portfolio_strategy_lab/test_workspace_activity.py::test_every_refusal_is_counted_by_operation_code_caller_and_vendor_and_nothing_else`,
`test: tests/portfolio_strategy_lab/test_workspace_activity.py::test_every_operation_in_the_vocabulary_is_classified_once`,
`test: tests/portfolio_strategy_lab/test_workspace_activity.py::test_only_typed_failure_codes_and_exception_classes_are_persisted`.

**OP15. A next request is a closure or a declared template, and a read reads again from itself.**
Every request an answer offers names an operation the Host answers and only that operation's
fields, each of the kind the grammar declares; a field it leaves to its reader is None or absent
(`choices`), never a stand-in (`<candidate_id>`); what the Host knows it binds. The CLI's copy
of a read's answer, printed and saved, keeps the request it sent, its whole selection
(`read_request`, `named_read`), so `--from` reads the same read again from the answer alone, its
target, references and every choice (a day, a page), a flag naming another value winning; the
Host's answer stays its owner's, byte for byte, so an export's record, a measured delivery and a
self-hash hold, and a read answers alike at every entry. *Why:* the outside reviews' continuation findings (V419, V426, V438,
V443) and the sweep's (V449): an Evidence answer offering a unit's id as a request without an
operation, packet requests carrying fields no packet request takes, 22 reads whose answers did
not read again from themselves; named at the operation door first, the references broke an
export's hash, a measured packet's bytes and an export's equality with its read (V449);
`study show --from` a saved read of a past day read the latest one (V454).
*Held by:* the answer check every test's operations pass (`request_problem`, the self-read, V449),
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_an_offered_request_is_one_the_host_accepts_as_it_stands`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_a_read_names_what_it_read_and_reads_again_from_itself`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_every_read_reads_its_whole_selection_again_from_its_answer`.

**OP16. A printed command is read back as the request it came from, in the context it ran in.**
The CLI's reading of a command line (`request_of`) is the printer's inverse for every operation,
field and declared kind, in POSIX shells and in Windows PowerShell: a value the parser would read
otherwise is printed so it is not (a leading `@` doubled, a leading `-` joined to its flag), and a
request holding a double quote goes to PowerShell whole through stdin. Every command the CLI
prints keeps the context its call ran in (its view, its language, its goal: `_kept_options`). An
answer saved as JSON or as YAML reads back alike through `--from`. *Why:* 436 of 1,925 printed
requests misread, Windows PowerShell 5.1 stripping every printed JSON argument's quotes, a
bundle's submit command dropping its `--goal` (V449). *Held by:*
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_every_printed_command_reads_back_as_the_request_it_came_from`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_a_command_printed_for_powershell_sends_the_request_it_came_from`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_a_bundle_submit_command_keeps_the_goal_it_was_prepared_under`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_an_answer_saved_as_json_or_yaml_reads_back_alike`.
Every executable command offered in refusal words or an answer template must also parse with
the real CLI; a printer round trip alone does not hold those independently authored commands
(V687, V692). *Held by:*
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_every_refusal_and_answer_template_command_parses`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_the_canary_door_operator_command_parses_through_the_real_cli`.
The census walks the refusal and answer catalogs, client refusals, schemas and generated offers;
its prose extraction recognizes registered command heads. *Not yet held:* that extraction does
not detect arbitrary unregistered command-like prose (V687).

**OP17. Context resolves from the request, then the file it names, then the session; a target in
conflict is refused.** A request's target (its Task, its goal) is the one its flags name, else
the one the saved answer `--from` names, else, where the operation declares it, the session's own
goal (V391); a file naming no target the operation needs is refused, never read as a default, and
a flag naming another target than the file is refused, never preferred. Any other reference the
file names fills what the flags leave, a flag giving one winning. *Why:* `goal abandon --from
goal-a.json` closed the session's own goal (V449, an outside review). *Held by:*
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_a_command_from_a_saved_answer_acts_on_what_the_file_names`.

**OP18. Every state has its exit.** Every Task state has one CLI outcome (pending, refused or
done), read wherever an answer names it (`lifecycle`, `task_lifecycle` beside an answer's own
`status`, or `status`: `TASK_STATE_FIELDS`), and a composite (a trial, a goal) names the part whose Task waits on a request, so a wait
on it ends where a decision is needed; an answer that reuses, admits or resubmits work names what
holds it (its Task, or the result or record it is, by an id or a hash at its top, or a request it
offers naming one). *Why:* a goal waiter missed BLOCKED (V440); a
trial whose step needed recovery read RUNNING and a wait ran on (V449); a data update's reuse
answered no Task to read (V449); a data update's wait ended at once, its Task's state named
`task_lifecycle` (V455). *Held by:*
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_every_task_state_has_one_cli_outcome`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_a_task_state_is_read_wherever_an_answer_names_it`,
`test: tests/portfolio_strategy_lab/test_status_follow.py::test_a_wait_on_a_trial_ends_when_a_step_needs_its_recovery`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_a_reuse_names_what_holds_its_work`,
the answer check every test's operations pass (`exit_problem`).

**OP19. A first use is the agent's, by the person's one sentence.** A goal of kind `FIRST_USE`,
opened from the person's sentence, once per workspace and before its first preparation, delegates
the first use's person-only steps (`FIRST_USE_STEPS`: opening the network for the preparation,
confirming it and its resumes, deciding its data issues, confirming its membership changes, and
activating its book once that book's Evidence and CRO review is published, which the person
deactivates) to the agent running it, while it is
open and within its hours (`FIRST_USE_HOURS`); each runs as the person's decision carried by the
agent and is recorded in the goal's ledger as delegated. Its book draft's declared
unavailable-return quarantine is a data decision of that kind: the draft's policy kept and the
effective population reported, never chosen to improve a result (V504). The network it opens holds until the
delegation's end by the control itself (OP5), and its earlier end, by submission or abandonment,
closes it; a deactivation, any other activation, a storage decision, an automation, a revocation
and anything paid stay a person's, and the goal is never revised nor offers a revision. Its record names the delegation,
its end and whether it holds, and the person's decisions name it with its Stop while it holds
(U70). *Why:*
the maintainer, 2026-10-01: the person gives one sentence and the agent runs the first use, the person
able to interrupt at any time (V452); the maintainer's hands-off rule, 2026-10-07: a reversible step
is a default the agent takes and discloses, so the reviewed book's activation and the membership its
source names are the first use's (STOPS-1). *Held by:*
`test: tests/portfolio_strategy_lab/test_first_use_goal.py::test_a_first_use_goal_lets_its_agent_take_the_first_steps_and_ends_with_them`,
`test: tests/portfolio_strategy_lab/test_first_use_goal.py::test_a_first_use_delegates_only_its_steps_for_its_hours_and_is_never_revised`,
`test: tests/portfolio_strategy_lab/test_first_use_goal.py::test_a_delegated_activation_takes_only_a_book_with_a_published_review`,
`test: tests/portfolio_strategy_lab/test_first_use_goal.py::test_a_first_use_is_the_one_before_the_first_preparation`.

**OP20. A person's decision carries by its evidence.** A retry that changes only a run or clock
field carries the person's existing decision for the same issue, evidence and scope; changed
evidence requires a new decision, and a different issue never borrows one (V658). A carried raw
retention choice preserves the original sealed receipt and caveat, and matches the retained
listing evidence, option, policy and source membership. *Not yet held:* the holder proves this
raw-retention choice, not an enumeration of every person decision a retry can meet (V658).
*Held by:*
`test: tests/feature_input_gateway/test_feature_input_gateway.py::test_raw_retention_choice_continues_only_for_the_same_case_on_a_derived_manifest`.

**OP21. Browser authority belongs to one Host.** Two Hosts on one machine do not share a browser
credential. Each listening port has its own cookie name and independently generated secret;
a restart refuses its prior secret. A write checks that Host's admitted origin and matching
cookie/header, rather than accepting another Host's token from the same browser jar (V693).
The cookie's path remains `/`; the isolation is by name and token, not browser cookie port
semantics. An explicit client's legacy shared-name cookie still requires that Host's current
token. *Held by:*
`test: tests/portfolio_strategy_lab/test_local_web_product.py::test_browser_session_cookies_are_port_scoped_and_restart_replaces_only_its_own`.

**OP22. What using the product teaches ships where the person's agent reads it.** The research Skill's
references carry what an agent should know to use the product: the operating guide (launch, binding,
safe shutdown) and the field notes (each friction met in real use, its way on and its timings). A
card that learns such a thing by using the product writes it there, not only in a record the person's
agent never reads (V699). Installed guidance links to files the install carries, never to an anchor
or an outside address, so the offline audit of every installed link holds. *Why:* an evaluation agent used the
product twice as its first real user (V699, V701), and the whole suite on the final candidate found
34 field-note links with anchors that no merge's test set had reached. *Held by:*
`test: tests/release/test_wheel_runtime.py::test_installed_configure_copies_guidance_unchanged_and_refuses_an_overwrite`;
the guidance's content by review.

## PA. Parameters

**PA1. Three classes of parameter.** METHOD changes a result and enters an identity (its change records a
successor or is a new recipe); EXECUTION paces work and never enters one; LIMIT bounds the product. The
parameter registry (`config/registries/parameters.json`) holds each literal of the number-deciding
code with its class; a new one is refused until it is registered. A literal is classed by what it
does, never by its name: a budget that decides what a study searches, or a chunk size that shapes a
sealed artifact's hashes, is METHOD; a provider's qualified policy, a timeout or a measured ceiling
is LIMIT (W13). *Held by:* `gate: registries`, `test: tests/structural/test_registries.py::test_the_tree_adds_nothing_to_a_registry_without_registering_it`.

**PA2. Execution parameters are the operator's.** Threads, workers, batch and cache sizes are set through
typed configuration, the CLI and the UI, with a default the program computes from the machine; they
never enter an identity, a sealed spec or an Agent tool's vocabulary. *Not yet held:* the Sector
reference refresh's staging binds its worker count, so fewer workers after a rate limit cannot resume
it (V257, RX). *Held by:*
`private-test: test_installed_numerical_adapters_declare_and_enforce_a_thread_bound`.

**PA3. An execution parameter that could move a number is proved not to, or the result says it was not.**
A sealed canary each session checks it and refuses by name on a mismatch; where no canary exists, the
value used is recorded beside the result as provenance (the working agreement), and the result then
claims no equality with a run under another value. A method's identity leaves execution out either
way, and a result is identified by its content, the environment beside it. What counts as the same
number is the owner's to state, exact bits or a tolerance it names, and a tolerance never relaxes an
artifact's integrity check. The BLAS thread count can move the Alpha metrics' last bits (V68), so
each Alpha run's fit log records the numerical libraries' thread counts beside the result (W10);
a result claims no equality with a run under other counts. *Held by:*
`test: tests/alpha_research/test_lightgbm_development_capability.py::test_lightgbm_fits_on_one_thread_unless_the_sealed_canary_holds`,
`test: tests/workspace_maintenance/test_retrieval_recipes.py::test_a_session_whose_canary_moved_is_refused_by_name_with_its_threads`,
`test: tests/workspace_task_runner/test_task_child.py::test_a_fit_log_records_the_numerical_thread_counts`.

**PA4. No ambient knobs.** A value that steers the product lives in typed configuration, never in a
machine-specific constant or an environment variable read deep in the code; a retired variable's
writers go with it (V6, V61). `ALPHALATTICE_NETWORK_DISABLED` is the named exception (OP5). Scripts
and tests carry no machine-specific absolute paths outside literal test fixtures. The evaluation's QA
store and run tree resolve through one typed declaration with an operator override; a test gives the
allocation its own temporary root (V651). *Held by:*
`private-test: test_scripts_and_tests_have_no_machine_absolute_paths_outside_literal_fixtures`,
`private-test: test_relative_roots_and_operator_override_resolve_from_one_typed_declaration`,
`private-test: test_audit_accepts_the_operator_allocation_and_refuses_an_escape`;
the other configuration boundaries by review.

**PA5. A managed-storage limit is the operator's typed setting.** The storage cap is LIMIT
(PA1), not a research method or identity. Its automatic value uses measured managed data and
machine free space; capacity writers and observation retention use the same workspace setting.
Changing it does not change a fresh research identity, preparation plan, Task or input seal.
An over-cap refusal offers an accepted cap-setting or cleanup action (V680). Historical receipts
retain their sealed budget and exact readback. *Held by:*
`test: tests/workspace_maintenance/test_current_state_storage.py::test_storage_cap_counts_managed_models_panels_and_artifacts_and_follows_the_operator`,
`test: tests/workspace_maintenance/test_current_state_storage.py::test_cleanup_plan_checks_old_cap_receipts_and_new_plans_exclude_execution_capacity`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_storage_cap_is_one_operator_setting_through_the_real_cli_and_http`,
`test: tests/workspace_readiness/test_workspace_preparation.py::test_new_preparation_plan_and_exact_task_reuse_ignore_the_operator_cap`,
`test: tests/unified_observation/test_unified_observation.py::test_an_open_observation_ledger_follows_the_workspace_cap_without_reidentifying_records`.

## DA. Data and formats

**DA1. Real data only on copies.** An original workspace is never opened by a card; copy it from the
maintainers' private QA store to a scratch directory and work on the copy. *Held by:* review.

**DA2. Every table and artifact has a registered authority.** A persistent table is registered in
`config/storage-authorities.json` with its owner, generation model and rebuildability, and an artifact
prefix in `config/artifact-authorities.json`. Whether a table may be discarded is its group's declared
rebuildability there, never a rule for a whole engine: the market store holds several owners' tables,
some rebuildable and some not. A table is registered by its database and name, and its schema and
data are written only by its registered owners: another package's data writes are a baseline that only
shrinks (`config/registries/table-writers.json`, G2). A written path is registered under its root
to the class that writes it, the most specific prefix winning (G2). *Not yet held:* the discovery of
paths is AR's static scan of literal URIs and artifact-root joins, so a path built only at run time is
not seen; the Evidence stores under `runtime/` are not in the registry yet (the storage map), nor the Feature Trial records (`runtime/feature-trials/`),
whose damaged record also blocks its own reopening (V272, RX). *Held by:*
`private-test: test_every_table_is_registered_and_written_by_its_owners`,
`private-test: test_every_written_path_is_registered_to_its_writer`,
`private-test: test_panel_closure_artifacts_have_explicit_non_eviction_authority`.

**DA3. The store surface only shrinks.** A new public store method, table or connection is admitted by
name with its cause. *Held by:*
`private-test: test_store_surface_does_not_grow`.

**DA4. A DuckDB file is opened with its access mode declared.** *Held by:*
`private-test: test_file_backed_duckdb_connections_declare_access_mode`.

**DA5. Writes go through their owners.** Feature writes go through the closure coordinator, and Alpha
fits through the Host-owned adapter. The workspace manifest is created and changed only through its
owner; an update holds the workspace's mutation gate over its read, change, publication and readback,
so concurrent updates keep both changes (V641). *Not yet held:* Market Data's bootstrap sets the
lifecycle in Feature's Panel table by its own SQL (V176) and writes `remediation_execution`, registered to
Feature (V177). The current-state migration copies only the tables it names (`WRITTEN_TABLES`),
each registered as its write (V264). *Held by:*
`private-test: test_production_feature_writes_cannot_bypass_the_closure_coordinator`,
`private-test: test_alpha_model_fits_cannot_bypass_the_host_owned_adapter`,
`private-test: test_every_table_is_registered_and_written_by_its_owners`,
`test: tests/workspace_readiness/test_workspace_manifest_writes.py::test_only_the_owner_writes_the_manifest`,
`test: tests/workspace_readiness/test_workspace_manifest_writes.py::test_two_writers_through_the_one_write_keep_both_changes`,
`test: tests/workspace_readiness/test_workspace_manifest_writes.py::test_evidence_setup_waits_for_the_manifest_owner_and_keeps_another_binding`.

**DA6. A persisted format names its version, its readers and its upgraders** (the formats registry,
`config/registries/formats.json`, with the Task kinds); a new schema name or Task kind is refused
until it is registered, and an entry that names no version, reader or upgrader list, or a reader
that is not a file, is refused (V267). *Held by:* `gate: registries`, `test: tests/structural/test_registries.py::test_the_tree_adds_nothing_to_a_registry_without_registering_it`.

**DA7. Private things stay out of Git.** Never secrets, provider payloads, raw prompts or tool traces,
hidden reasoning, strategy evidence or large numerical artifacts. *Held by:* review.

**DA8. Retired as a standing rule** (the maintainer, 2026-10-04, V647). That an evaluation's
blind review set is read only as counts was the evaluation's design, not a product rule: its
mandatory count, halt and quarantine steps are retired, a run's counted report follows it, and
review accuracy is next measured on a fresh blind set. AXLEAN-2 (2026-10-08) retired the
optional counting, instruction preflight and historical source screen/seal tools. No law
replaces the retired restriction. *Held by:* the dated decision (review).

**DA9. Each file type has one job.** Before writing a file, choose its type by who writes it and who
reads it:

| type | its job | written by | read by |
| --- | --- | --- | --- |
| YAML | what a person or an agent authors for the product to act on: experiment documents and recipes, campaigns, research cases, arms, market and Agent profiles (the YAML front matter of a Markdown profile), a CLI request | people, agents | the product |
| JSON | what the product writes and reads back: sealed records, manifests, receipts and evidence, every answer (the CLI envelope, HTTP bodies), the JSON registries and baselines under `config/` | the product; registries by an agent's card, held by the gate | the product, agents |
| JSONL | an append-only stream: an activity or execution log | the product | the product, agents |
| Parquet | numeric and tabular evidence: chunks, Panels, score and validation rows | the product | the product |
| DuckDB, SQLite | operational state; each table as its group's rebuildability says (DA2); and a research input's sealed source, the market store copied whole and pooled by its digest, which its readers open through the store's own repositories (V204) | the product | the product |
| Markdown | prose for people and agents: laws, plans, records, documentation, Agent profiles | people, agents | people, agents |
| HTML | a rendered report and the Workbench | the product | people |
| TOML | the Python project's build and checker settings (`pyproject.toml`) only | people | the build and its checkers |
| an external program's, a model's or a protocol's own format | what that consumer requires: the Codex and Claude host configuration (`.codex/*.toml`), model files (ONNX), an index's memory-mapped vectors (`.f32`) | the program or its owner | the program |

A new file follows the table; a file that does not is moved to its type in the card that touches it.
The table governs what the product owns; a format an external program, a model or a protocol requires is kept as
that consumer needs it (the table's last row), never moved to the product's types.
A sealed source stays the market store (the maintainer, 2026-09-29): its readers read 26 of its 66 tables,
the feature rows among them, and as Parquet they are 560 MB of the measured workspace's 768 MB (IN1, V204).
*Not yet held:* Risk's covariance chunks keep their numerical evidence as packed `.bin` blobs until
V314's rotation (V294's Risk half), where the Portfolio lab's and Alpha's lanes moved to Parquet and
read a store written earlier from its `.bin` files until the formats registry names their upgrader
(V210, V294, V267).
*Why:* YAML is kind to people and agents writing, and ambiguous for machines storing (types guessed,
comments lost); JSON is exact and hashable but tiring to write by hand. *Held by:* review.

**DA10. Authored YAML stays the author's.** The product reads an authored document, validates it and seals
its normalized JSON with the plan; it never writes into the author's file. What the product offers to be
edited (a template, a draft, a next request) it gives as YAML. *Held by:*
`test: tests/researcher_methodology_surface/test_experiment_authoring.py::test_the_same_document_and_authority_produce_the_same_program_identity`.

**DA11. YAML is read strictly.** One owner parses the product's YAML
(`protocols/research_authoring/selection.py`), each document in a named dialect: an authored document
in the declaration dialect, a `yaml.SafeLoader` that refuses a duplicate key and reads an exponent
number as a number; a document in a historical format decoded in the dialect that wrote it, then
converted (`rewrite_in_declaration_dialect` reads an old export as PyYAML's default dialect wrote it,
so its string `"1e-10"` stays a string). A document parses into its operation's typed contract,
which refuses unknown keys and types dates and numbers itself; a contract cannot undo what the
dialect already decoded, so the dialect is the owner's to name. A document names no Python module or
callable (OP8). *Not yet held:* four Desk
compilers (Alpha, its lifecycle, Factor, Risk) read their section as a mapping; three refuse an
unknown key by hand, each its own way, and by the code the Alpha compiler refuses none (V249, W12);
six sites still call `yaml.safe_load` themselves, the Alpha and Sector campaigns among them, where
a duplicate key is silently overwritten (V276, W12); the market profile is read as a loose mapping
(V275, W12); and the declaration dialect keeps YAML 1.1's implicit booleans and octal numbers, so an
authored `010` reads as 8 and `on` as true (V278, W12). *Held by:*
`test: tests/researcher_methodology_surface/test_experiment_authoring.py::test_documents_cannot_name_python_modules_or_callables`.

**DA12. A published session never moves.** A refresh changes the sessions from its effective session
on and none before it: a membership change from the session its update computes, a Sector
reclassification from the trading day its update observed it, the next session when that day has
none, and never a session a Panel already published (V346); before T0 the first recorded
classification stands in for every earlier session, a disclosed backfill. A history with no change
binds what the backfill bound, so no identity moves until something is reclassified. *Why:* a
refresh that applied the day's classification to every session rewrote the history published
results were computed on, against the approximate point-in-time contract, and rebuilt every Panel
(51.0 s against 37.9 s on the fixture; 25.7 s after). *Held by:*
`test: tests/feature_engine/test_sector_forward_rule.py::test_a_planted_reclassification_is_read_forward_and_never_moves_a_published_session`,
`test: tests/feature_engine/test_sector_forward_rule.py::test_a_published_session_keeps_its_panel_cross_section`,
`test: tests/feature_engine/test_panel_membership_composition.py::test_membership_change_moves_only_the_sessions_from_its_effective_one`.

**DA13. A published file closes over what ships.** No public file requires or names a local dependency
the public manifest omits: the dependency ships too, or the referring file is private (V662).
A generated dependency needs a public producer, its declared output and actual write, and public
required inputs; declaring an output never makes an existing private file public. The check uses
the candidate tree's current release classification, not a previously generated manifest.
*Not yet held (V662):* arbitrary computed paths, ECMAScript module imports/exports, dynamic module loads,
Markdown reference-style links and other generated-output families are outside V662's bounded
static scan. Its passing result establishes only the forms that scan resolves and the enumerated
Workbench-asset and native-binding output families.
*Held by:* `private-test: test_every_public_file_closes_over_the_public_manifest`;
`test: tests/structural/test_public_closure.py::test_python_import_requires_a_public_local_module`,
`test: tests/structural/test_public_closure.py::test_markdown_links_inline_paths_and_fenced_commands_need_public_files`,
`test: tests/structural/test_public_closure.py::test_a_generated_output_requires_its_public_declaration_and_actual_write`,
`test: tests/structural/test_public_closure.py::test_a_generated_output_loses_permission_when_a_required_source_is_unpublished`,
`test: tests/structural/test_public_closure.py::test_a_producer_never_authorizes_an_existing_private_output_path`.

## SC. Contracts

A contract is the typed record at a boundary: what a person or an agent writes (a request, a
declaration, an answer), what an owner seals and reads back (a receipt, a manifest, a Program), and what
one owner hands another. These rules were learned building the in-process agents, whose tools and
answers had to be contracts first, and they outlive them: how to define a contract and how to use
Pydantic well stay the project's standing guidelines (the maintainer, 2026-09-27). The rules are
stated for any project; their *Not yet held* notes are this product's. A contract also keeps ID1 (a
change of meaning records its successor), DA6 (a persisted format names its version), DA10 and DA11
(authored YAML), OP4 (a refusal) and OP11 (an agent answers with judgment only).

**SC1. A record's kind follows its job.** A record that is stored, sealed, hashed or crosses a boundary
is a pydantic contract; inside one process a value takes the kind that fits it (a frozen dataclass, a
`NamedTuple`, a `TypedDict`), which no law chooses. *Why:* a record kept or hashed without its
contract has neither the contract's validation nor its schema: a dataclass hashed through `asdict`
beside a sealed contract is two sets of rules for one job. *Not yet held* (measured 2026-09-27): 17
scopes hash or write a dataclass through `asdict`, each converted when its identity next rotates, and
8 stored dataclasses the pin cannot see, the action audit chain's receipt among them (V258), with 28
more to check by their call paths (SCC, the private record-kind inventory). *Held by:*
`private-test: test_records_take_the_kind_their_job_names`, which
pins those scopes.

**SC2. A contract is strict and says what it admits.** It is frozen and refuses unknown fields. A closed
vocabulary is an enum or a `Literal`; a bound, a pattern or a length is a `Field` constraint (a hash
`^[0-9a-f]{64}$`, a count `ge=0`, a text handed to an agent `max_length`); a rule across its own fields
is the contract's validator, never a check in its callers, while a check that needs another owner's
state (an authority, a Task's owner, whether a reference exists) is that owner's, where the state is;
a default that can change a result is written into the sealed normalized document (DA10). A view over another owner's answer may allow unknown
fields, and a reader of a historical document or of the environment may ignore them, each named.
*Why:* a contract that admits anything moves its validation into every caller, each checking a
different part, and an agent learns the rule only by failing it. *Not yet held:* 5 contracts are
mutable (the Evidence analyst's three submissions, the Factor agent review receipt, the Feature trial)
and 2 progress projections ignore unknown fields
unnamed; four Desk compilers read their section as a mapping (DA11, V249). *Held by:*
`private-test: test_contracts_are_frozen_and_strict_or_named`, which
pins the named exceptions and the rest.

**SC3. A field's meaning is written once, in its contract.** A contract's docstring and its `Field`
descriptions are what a person or an agent reads, and `alphalattice schema` prints them from the
owner's models, never from a copy; a role card, a Skill or a document points to the schema instead of
restating a field. A schema's hash binds its structure (the types, constraints, required fields,
defaults and definitions), never its prose, so a description is free to change. *Why:* an agent reads
the contract before it writes a document, and a meaning written in two places drifts. *Held by:*
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_an_answer_is_read_in_parts`, which reads
an answer's parts from the owner's own models;
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_every_field_schema_prints_is_described`;
`test: tests/structural/test_source_identity.py::test_a_schema_hash_binds_the_structure_and_never_the_prose`,
`test: tests/structural/test_source_identity.py::test_every_hashed_schema_goes_through_the_kernel`.

**SC4. A sealed record proves itself.** A sealed contract carries its content hash, computed by one
helper over its canonical JSON without the hash field and checked by the contract's own validator
whenever it is built or read, so a changed record is refused before use (EV2). A contract changes
compatibly, by a field added as optional, `None` by default, whose check accepts the hash computed
without it while it holds `None`, or by a new version whose readers still read the old one (DA6), so
a record sealed before still verifies; any other change of meaning records its successor (ID1). *Why:* a record that proves itself is reused as sealed (EV1) without trusting where it
was read from. *Not yet held:* the sealing helper and the compatible check are written per owner: 288
sealing definitions, 56 named sealers and 25 identical bodies among them (SEAL, V251, W2,
counted before RT); the compatible hash check is written once, in the kernel
(`shared_kernel.sealing.validate_hash_compatible`, RT `94c55fce`).
*Held by:*
`test: tests/researcher_methodology_surface/test_local_web_factor_experiments.py::test_input_and_result_tamper_refuse_without_losing_historical_readback`.

## EV. Evidence and science

**EV1. Sealed evidence is reused as sealed.** A resumed, recovered or replayed run takes what was sealed
(numerical results with their metrics, receipts, reports) and never recomputes it. *Why:* recomputed on
another host or thread count it can differ in the last bits, and the run refuses its own receipt (V69).
Daily rolling evidence appends only newly completed outcomes and a compact content-addressed
head linking the immutable base and increment; it never re-seals history. Earlier heads remain
readable, and a read neither writes a missing head nor repairs a corrupt one (V690).
*Held by:*
`test: tests/researcher_methodology_surface/test_local_web_factor_experiments.py::test_alpha_process_recovery_keeps_task_and_reuses_sealed_work`,
`test: tests/researcher_methodology_surface/test_local_web_factor_experiments.py::test_hard_process_death_recovers_the_same_task_without_reexecuting_complete_evidence`,
`test: tests/portfolio_strategy_lab/test_rolling_history.py::test_two_daily_heads_append_only_two_refs_each_and_detect_earlier_file_tamper`,
`test: tests/portfolio_strategy_lab/test_rolling_report_composition.py::test_publish_missing_heads_is_idempotent_and_reads_never_write`,
`test: tests/portfolio_strategy_lab/test_rolling_report_composition.py::test_read_refuses_corrupt_persisted_head_without_repairing_it`.

**EV2. Tamper is refused before use, and history still reads.** A changed input, chunk, receipt or result
is refused by name before it is used, and what was published before reads back. An immutable-source
lease reuses a full verification across requests only with a held OS change signal, exact file
identity and read access; size and timestamps alone never grant reuse. A changed or uncertain
signal, restart or missing proof requires full-byte verification before use (V691). Every export,
publication and admission establishes full-content proof, freshly or through a currently valid
lease; the saved-study sweep remains weekly and after an upgrade (VR, decision 5). A prepared
bundle's background verification binds its actual read time,
exact subject and policy, not a later run or clock field; actual publication must refuse expired
evidence (V675). *Not yet held:* a Panel chunk's check binds its row hashes, not its values
(V270, W8). The legacy study shortcut still uses path, size, time and identity; V691's immutable
source lease checks cover the opted-in Windows helper, not that separate shortcut or an
enumeration of every owner read and daily-chain composition that should use a lease (V691).
The review clock holder checks delivery writers and callers; the expired-publication holder
does not fail when background verification uses the wrong read
time, so that verification binding lacks a direct failing holder (V675). *Held by:*
`test: tests/researcher_methodology_surface/test_local_web_factor_experiments.py::test_input_and_result_tamper_refuse_without_losing_historical_readback`,
`test: tests/researcher_methodology_surface/test_local_web_factor_experiments.py::test_binding_refuses_a_corrupt_copy_a_changed_source_and_a_tampered_pool`,
`test: tests/researcher_methodology_surface/test_local_web_factor_experiments.py::test_a_read_reuses_the_study_verification_while_its_files_are_unchanged`
for that legacy shortcut;
`test: tests/workspace_maintenance/test_verified_source_leases.py::test_os_changes_reverify_and_refuse_even_with_restored_size_and_time`,
`test: tests/workspace_maintenance/test_verified_source_leases.py::test_missing_or_failed_os_signal_forces_full_verification`,
`test: tests/workspace_maintenance/test_verified_source_leases.py::test_request_thread_exit_keeps_reuse_and_later_tampering_is_refused`
for the immutable-source lease;
`private-test: test_review_continuation_clocks_bind_the_resolved_dossier_or_sealed_bundle`,
`test: tests/alternative_evidence_desk/test_evidence_review_http_route.py::test_external_review_expiry_before_publication_keeps_the_assessment_but_not_a_publication`
for review delivery clocks and genuine expiry, respectively.

**EV3. A result claims only what its evidence shows.** A claim beyond the sealed evidence carries its
limit, and completion is proved from sealed pages, never assumed beside a gap. Each installed-book
read names its supported performance metrics from the exact sealed window and declared convention,
or the reason a metric is absent. Historical views stay pinned; forward views use only settled
outcomes for the exact book and cost lane, naming their source and as-of; proposals and unsupported
benchmark metrics do not become measured performance (V678). A coverage-floor comparison uses
the review owner's sealed accounted coverage, retaining reviewed, official nothing-filed and
unreached partitions separately. Missing accounting makes the comparison unavailable; reviewed
share never substitutes for accounted coverage (V681). Display of these partitions belongs to
Workbench ST1. *Not yet held:* the performance checks below exercise named owner and installed-book
paths, but do not enumerate every possible installation path (V678). *Held by:*
`test: tests/portfolio_strategy_lab/test_portfolio_decision_updates.py::test_daily_performance_reads_only_realized_outcomes_and_pins_the_issued_prefix`,
`test: tests/portfolio_strategy_lab/test_portfolio_decision_updates.py::test_daily_performance_names_absence_instead_of_annualizing_an_invalid_axis`,
`test: tests/portfolio_strategy_lab/test_strategy_activation.py::test_current_performance_follows_exact_books_daily_publications_and_keeps_history_fixed`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_the_installed_report_names_every_absent_performance_metric`,
`test: tests/alternative_evidence_desk/test_portfolio_review_decision.py::test_a_holding_that_filed_nothing_is_counted_apart_and_never_as_no_risk`;
the other claim limits by review.

**EV4. Science is never weakened silently.** A scientific invariant, threshold, identity, causal firewall
or fail-closed condition changes only as an explicit old/new experiment, recommended, not activated. A
named lifecycle range is measured whole, with development, observed-validation and stitched views; a
diagnostic label forbids retuning on the same observations, not measuring them. *Held by:* review.

**EV5. Unchanged work is reused, not recomputed; a cost that matters is measured.** A verified input is
proved once per request (A5); a cost is timed directly, never inferred; an overrun is fixed at its owner,
never met by weaker science or masked parallelism. The immutable-source helper shares a proof
within its opted-in request scope; cross-request reuse follows EV2's change signal, never size
or time alone (V691). *Not yet held:* the helper checks do not establish daily-chain integration
or enumerate every owner's request composition (V691). *Held by:*
`test: tests/workspace_maintenance/test_verified_source_leases.py::test_unchanged_sources_verify_once_and_disabled_reuse_reads_again`;
the remaining request compositions and measured-cost decisions by review.

## TE. Tests

The repository will be heavily reworked, and old tests must not hold it back; and a whole-suite run
wastes time, so it is never casual (the maintainer, 2026-09-26).

**TE1. Tests are evidence, not a counter.** A test is added only for a real requirement, regression,
tamper, recovery or uncovered boundary, reusing existing fixtures. *Held by:* review.

**TE2. The equivalence net protects a refactor.** A cut or merge proves it changed nothing the product
answers: U0, the golden CLI transcripts (C6), the identity readout, and the contract tests of what it
touched. *Held by:* `U0`, review.

**TE3. Every test is classed by what it pins**: CONTRACT (a public entry), BEHAVIOUR (an owner's rule
through its public functions), MECHANISM (a private name, a patched internal or an environment
variable, pinning how rather than what), DUPLICATE, each by the responsibility it holds, not its form:
a count that proves a reuse without recomputation, a scientific minimum or a persisted layout is
BEHAVIOUR or CONTRACT. A test goes when its responsibility retires, is duplicated or has an equivalent
holder; one in a refactor's way is rewritten at a boundary, never deleted for being in the way. *Held by:* review.

**TE4. Tests go with their owner.** Deleting or merging an owner deletes or merges its tests in the same
card, listed in the private test signal record; a test is never ported to keep exercising code that
is gone. *Held by:* review.

**TE5. New tests enter through public entries**: no new private import or patched private attribute in
`tests/` (the ratchet, `config/registries/test-private.json`, which only shrinks). *Held by:* `gate: registries`, `test: tests/structural/test_registries.py::test_the_tree_adds_nothing_to_a_registry_without_registering_it`, `test: tests/structural/test_registries.py::test_the_tests_ratchet_refuses_an_entry_its_base_did_not_hold` (the gate compares the registry with the one the change started from, V16).

**TE6. A structural pin is a set or an invariant, never a count**; a moved set is re-pinned with its
cause by name. *Held by:* review.

**TE7. Development speed first; the whole suite at the maintainer's word.** Per commit the hook's fast gate
and the smallest tests that answer for what changed; a change that cannot alter behaviour is proved
directly (its syntax without docstrings, the identity readout) rather than by running more. The full
gate (`check_playpen.py --staged` without `--fast`) and the whole suite run only at the maintainer's
word, typically just before a product release, never as a goal's closing step (the maintainer,
2026-09-26: development speed comes first). A merge moving door words or Chinese runs the whole
Workbench readback; a new local require or import in a shipped reader reruns the public release
closure on the combined candidate (V692, V696). Each check runs once, by one runner: a card runs
the nodes it adds or edits, its owner's direct nodes and the census or harness its change touches;
the merger runs the touched files whole, once, at the merge; U0, the golden replay, the UI walk and
the whole suite run once each, at the end, on the final candidate (the maintainer, 2026-10-04: no
work is done twice). *Not yet held:* the fast hook does not select
these merge checks automatically; their selection remains the merger's recorded responsibility
(V692, V696). *Held by:* `hook` for the per-commit fast gate;
`private-test: test_every_chinese_translation_uses_its_owned_ui_vocabulary`,
`private-test: test_every_public_file_closes_over_the_public_manifest`
for the selected word and release-boundary checks, not automatic merge selection;
review for the merger's recorded selection duty.

**TE8. A failure is diagnosed, never masked.** No skip, relaxed assertion or unrelated fixture churn to get
green; the failing cluster is diagnosed, its smallest failing subset rerun, then the consolidated set. A
mechanism is reproduced by a targeted setup, not by rerunning until it shows; a heavy test runs once for
evidence. *Held by:* review.

**TE9. The quarantine names every test with its class, owner and end** (the private test signal
record and private quarantine record); still there when the next wave is planned, it is deleted or
its owner decides. *Held by:* review.

**TE10. Real evidence has its own lane.** A test that reads real evidence is marked `real_evidence`, reads
roots declared in the private evidence-root declaration, leaves the routed lane, and skips by a named reason when
a root is absent or unreadable. *Held by:*
`private-test: test_the_routed_lane_deselects_real_evidence_and_names_it_required`,
`private-test: test_an_evidence_root_that_cannot_be_read_is_a_named_skip`.

**TE11. The tests tree keeps its boundaries.** No test module imports another, no test edits `sys.path` (one that
needs a tool's directory uses `monkeypatch.syspath_prepend`, which restores it),
and case studies never host the automated suite. *Held by:*
`private-test: test_tests_tree_keeps_its_import_and_evidence_boundaries`,
`private-test: test_case_studies_never_host_the_automated_test_suite`.

**TE12. A seam is held by a check of its class, never by its instance.** Where two owners meet (a
printed command and the CLI's reading of it, an offered request and the operation it names, a saved
answer and the `--from` that reads it, a refusal and its words, a Task's state and the waiter that
reads it, a write and the target it acts on), a fix lands with the check that holds the whole class:
a property over the contract's table where the class can be enumerated (every operation, field and
declared kind printed and read back, in POSIX and in Windows PowerShell; every lifecycle state
classed; awkward values saved in both formats and read back), and the answer check every test's
operations pass where it can only be observed (each offered request one the Host accepts as it
stands, each refusal with words and a way on). An observed check holds only what the tests reach:
enumerate where the table allows. *Why:* nine outside reviews in one day each found new instances
of the same seams; the sweep found 436 of 1,925 printed requests misread and Windows PowerShell
5.1 stripping every printed JSON argument's quotes, which no test had met (V449). *Held by:*
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_every_printed_command_reads_back_as_the_request_it_came_from`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_a_command_printed_for_powershell_sends_the_request_it_came_from`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_an_answer_saved_as_json_or_yaml_reads_back_alike`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_an_offered_request_is_one_the_host_accepts_as_it_stands`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_every_task_state_has_one_cli_outcome`,
`test: tests/portfolio_strategy_lab/test_cli_contract.py::test_a_command_from_a_saved_answer_acts_on_what_the_file_names`,
the answer check in `tests/conftest.py` (V410, V449).

**TE13. Test temporary roots stay outside the checkout.** The resolved pytest base temporary root,
explicit or default and through any path alias, is outside the checkout. An inside root is refused
before collection or cleanup, with its reason and an outside command, so checkout stores, source and
hooks cannot contaminate a product result or be deleted as test temporary files (V646). Tests' backup
roots also stay outside the checkout. *Held by:*
`private-test: test_pytest_refuses_checkout_temp_before_collection_or_cleanup`,
`private-test: test_pytest_uses_external_test_and_backup_temp`,
the pre-collection guard in `tests/conftest.py::pytest_configure`.

**TE14. Browser class checks preserve coverage and watchdog margin.** Every collected class in the
split pinned-browser harnesses must run alone within half its declared watchdog and leave both
files runnable whole at eight workers without a load-only timeout. Splitting preserves assertions,
scenes, widths, languages, themes and the original watchdogs; a missing, unknown or repeated class
selection refuses rather than silently changing coverage (V695). *Not yet held:* the roster and
runner holders below enforce class coverage and the full watchdogs, not the half-budget margin
or an eight-worker load bound. The per-class measurements and whole-file run are observations,
not an automated half-budget holder (V695). *Held by:*
`private-test: test_pinned_browser_harness_class_rosters_are_collected_and_require_a_selection`,
`private-test: test_minor_ui_classes_in_the_pinned_browser`,
`private-test: test_workbench_repaints_and_route_identity_in_a_real_browser`.

## PR. Process

**PR1. One development line.** The private development branch holds code, tests, plans and records;
every card's branch enters it only by the maintainer's merge after the gate, and the public
repository is exported from it. *Held by:* review.

**PR2. Commits.** One card a branch, a commit a unit, staged by path; never a bare `git stash`,
`--no-verify`, a push or a history rewrite; the tree left clean. *Held by:* `hook`.

**PR3. Less code first.** Retire before adding: each card reports its net lines by area, an addition
names what it replaced, and a new path beside an old one removes it or lists it with its readers. *Held by:*
review.

**PR4. Living lists.** Every card adds what it found to the convergence list (removable, wrong or kept,
with its decision and resolver) and the CLI list, and records its landing in the development plan. A card
fixes what it finds at its owner, in its own branch, while it is open, and names each finding, fix
and commit in its hand-back; no finding is reassigned to another card, and where the fix lies in
code another card is changing, the finder records it for that card (the maintainer, 2026-10-04). *Held by:* review.

**PR5. Records.** A material code, identity or numerical change keeps one dated record; every new
Markdown document starts with one H1 and a `Date:` line. *Held by:* review.

**PR6. Files are LF**, but for the Windows scripts (`.ps1`, `.cmd`, `.bat`), which stay CRLF. *Held by:*
`config: .gitattributes`.

**PR7. Who edits what.** Only the maintainer edits the identity records, closure lists, the gate
and this book; a card edits only its own scope, and the Workbench's area changes through its own
owner; a change keeps the page working until the UI pass (PR12). *Held by:* review.

**PR8. Report faithfully.** A failing check is reported with its output and a skipped step is said; code
completion is kept apart from scientific or production readiness; a hand-off gives the SHA, what landed,
the evidence, the numerical work run, what stays open and the next lawful action. *Held by:* review.

**PR9. Stop only for a real blocker**: an explicit redirect, new authority, an invalid premise, a
required artifact unavailable, invalid evidence. Then finish the independent work and name the next
lawful action; never interrupt a healthy, authorised numerical run. *Held by:* review.

**PR10. Sources and language.** Uncertain behaviour is resolved from current primary documentation or
exact-version source, keeping observed fact, documented behaviour, inference and decision apart. Files,
identifiers, failure codes and commits are in English; the product's Chinese translates its English. *Held by:* review.

**PR11. Coordination is written down.** Cards are handed out, handed back and decided on a private
coordination log: one page a day, append-only and never edited, every entry signed by its author;
a card ends with its completion entry before its owner's next card starts. A new rule enters the
next card, never one already running (the maintainer, 2026-10-04). *Held by:* review.

**PR12. The Workbench changes last, from one ledger.** No card edits the Workbench before the UI pass,
once the development converges. A card that changes what a page reads (an answer's fields, an offered action,
a word, a route, a paging rule) records the page's side in the private UI change ledger
in the same commit, and keeps the page working until the pass. *Why:* the Host still moves, so a page
fixed now is fixed twice, and scattered notes lose some (the maintainer, 2026-09-27: the UI
changes last, once everything converges, and every change is recorded). *Held by:* review.

**PR13. The ledger is filed by month and day, and code never cites it.** A finished plan or record
goes to the private ledger by year, month and day, the day it first entered the repository. A comment, a
docstring, a constant or an answer in `src`, `scripts` or `tests` names the law, the convergence row or
the card, never a plan's path: a plan is filed and moved as the work goes on, and a path to it goes
stale (the maintainer, 2026-09-28). A sealed record keeps
the provenance path it was sealed with, since that names where an input came from, not where to read
it. *Held by:* `private-test: test_the_ledger_is_filed_by_month_and_day`,
`private-test: test_code_cites_no_plan_path`.

## UI. The Workbench

The Workbench's laws are
[`src/alphalattice/interface/local_application/assets/workbench-source/LAWS.md`](src/alphalattice/interface/local_application/assets/workbench-source/LAWS.md)
(chapters MA, TY, CO, CT, ST, FT, PG, LS, LY, WD, AC, PR), with its parameters in
`workbench-source/design/parameters.json`. They have their own owner; a UI change cites them.

## Held by review

The laws no check holds yet, each to become a check where it can; this list only shrinks:
ID5, OW3, OW8, OW10, OP7, OP13, DA1, DA7, DA8, DA9, EV4, TE1, TE3, TE4, TE6, TE8, TE9,
PR1, PR3, PR4, PR5, PR7, PR8, PR9, PR10, PR11, PR12.
