# Design principles
Date: 2026-10-10

AlphaLattice is a local quantitative research workstation operated by a person's own Claude Code or Codex agent. From one sentence it takes a research intention through recorded inputs, numerical studies, a book, an investment committee and a delivery report. Its positions are research results, never investment advice or orders.

These are the maintainer's choices and the reasons for them. Together they describe one idea: the agent brings judgment, the system makes that judgment safe and reliable, and a measured loop makes both better over time. Each principle points to a mechanism a reader can inspect in this repository. A passing check proves the requirement it states, not that an investment conclusion is right.

## 1. One sentence, and the agent completes it safely

A person states an intention, with the date they care about, and the agent carries the work through. Every operation it needs is within reach through the command interface. An action available only as a Workbench click is a defect, because the agent cannot finish the work without the person. The [operation registry](../../src/devtools/architecture/operation_registry.py) checks that every Workbench action has a command, client or confirmation route.

The person's attention goes only where the decision is theirs: activation outside the first use, a wider scope, network access, outside software, cleanup and publication. The agent asks once and sends the person's own words with the request as `person_confirmation`, bound to that one request and used once.

**Capability is not authority.** Giving the agent every operation is safe because the system enforces authority instead of trusting the agent:
- **The person's decisions are checked by their owners.** A request that needs the person's word is refused without it. The confirmation records declared words; it is not an identity check.
- **Delegation is narrow and expires.** The first use delegates named acts only: preparing data and deciding its issues, activating its book, and setting up Evidence within the default budget. It lasts 24 hours and ends on submission, abandonment or expiry, closing the network it opened.
- **The network is the person's.** Outside that delegation it is asked for. An operator's offline switch holds it closed, and the agent can neither lift it nor restart the Host to get around it.
- **The numbers are out of reach.** Results are sealed to their inputs and method, the agent never types a figure, and no improvement may move a number.
- **Specialists hold least privilege.** They read by default and run only what their assignment names. In the committee each member acts only with its own key.
- **Outside text is data.** Filings, evidence and earlier answers are never instructions.
- **Nothing leaves the machine unasked, and nothing trades.** Publication is the person's, and AlphaLattice places no orders.
- **Every act is recorded,** so an agent's account can be checked against the record.

The evaluation grades safety on every turn: a forbidden act or a skipped question fails an attempt, even if a later turn succeeds (principle 12).

## 2. Build where the host agents are not, and ride where they are

AlphaLattice runs on the person's own host agent, today Claude Code or Codex, and chooses deliberately where to spend its effort. It does not compete with what host agents and model providers already do, and do better every month: the model, the agent loop, coding, tool calling, context management and subagents. Each of their improvements reaches AlphaLattice for free.

Setting that layer aside does not leave little to do. It reveals how much remains: everything a domain needs before an agent's work can be trusted. That space lies outside every harness, whichever host or model runs on top, and it grows as more harnesses appear, including ones built on open-weight models and plain model APIs. It is not a thin layer around a model; it is the hardest part of the system to get right. The command contract is host-agnostic: one source already serves two hosts, and another harness needs an adapter for its guidance, not a new product.

**No model inside.** The product itself calls no language model. The only models in the loop are the person's own host agent and the specialists it starts, and Evidence's retrieval models run on the machine. There is no model API key to configure and no hidden usage to pay for, the person's choice of model applies everywhere, and their research does not pass through a service they did not choose.

The host agent brings judgment: reasoning, reading, writing and running commands. AlphaLattice's Host brings what must be reliable:
- recorded inputs and methods, and sealed numerical studies;
- Tasks that survive a crash, and a watch that notices when one stops;
- Goals that tie requests to their completion;
- the market calendar and its holds;
- authority enforced at admission;
- a committee whose rules are kept and whose dissent is preserved;
- a complete record, and answers that correct an agent's mistakes.

A language model cannot provide these by design, and they are where most of the engineering lives. A capable analyst is not yet an investment firm; the firm's systems for orders, risk, compliance and records make it one.

## 3. The program computes, the agent judges, the person decides

The program owns every number and its binding to inputs and methods. The agent interprets results, names their limits and challenges conclusions. The person keeps the decisions that change authority or put research to work. Keeping these roles apart lets a debate be reviewed without persuasive language moving a result.

In the investment committee, members refer to figures by the aliases the floor shows, and the Host renders the kept value. A member can argue that a weight is too large without copying, rounding or recomputing it. In the [committee admission test](../../tests/portfolio_strategy_lab/test_committee.py), `Hold H1.` is admitted and `Hold 5%.` is refused as a typed figure, with the same message offered using the alias. The committee changes no number.

**Judgment is argued, not averaged.** The committee is built so that disagreement surfaces instead of blending away:
- the first round is blind: each member's stance stays hidden until all four are in, or the stance window ends with the portfolio manager's in, so no one anchors on another;
- after its stance, each specialist takes one turn, speaking or passing, with three challenges or replies in all, so they go to what matters most and the floor settles once every member has spoken or passed;
- the portfolio manager rules on every challenge, adopting it, rejecting it or handing it to the person, whose own words are relayed;
- one short discussion round follows the reveal; a specialist silent past it has passed, and the verdict never waits on silence;
- the chief risk officer's standing dissent goes into the delivery report in its own words, beside the verdict.
- the committee is a showcase of discussion, never a gate: the verdict defaults to proceeding with notes, turns a problem with particular holdings into a suggestion to avoid or reduce them, declines only for a named book-level hard defect, and the report leads with each member's suggestions on holdings for the person to decide.

Members argue from records already kept (principle 5), so the debate is about judgment, never about which numbers are true.

## 4. The research is honest about what it knows

A system that makes research easy also makes it easy to fool oneself. The numerical methods are built so that the common mistakes of quantitative research are refused, not left to discipline:
- **No look-ahead.** Every input is read as it was known at the time. Sessions before the first actionable one are a causal replay, never presented as live history.
- **Held-out evidence stays held out.** Studies validate out of sample in outer folds, and a development span is kept apart from a sealed holdout. A method whose labels would overlap the sealed boundary may be installed but may not publish into it.
- **A comparison may end inconclusive.** Methods are compared paired, on the same sessions, and a comparison reports superior, inferior or inconclusive from its interval. An interval that spans zero is inconclusive, never a narrow win.
- **Data enters on one declared basis.** Prices enter on the provider's split-adjusted basis, and every corporate action that can move a return, splits included, is part of that return's identity. An adjustment cannot be applied twice unnoticed, and a tampered one is visible.

Dates are part of the method. An information cutoff, a first actionable session and a horizon mean different things, and a later date never silently makes earlier information available. The [first-use goal test](../../tests/portfolio_strategy_lab/test_first_use_goal.py) fixes the clock and targets a later session. It checks that the update is held until its data is ready and is reported in market time and the person's time. The records let an agent explain what happened and what remains held without inventing either; the [agent guide](agents.md) states their limits.

These choices make some results less exciting. A large return that comes from a data error, or a winner that comes from noise, is the most expensive result a research system can produce, because it is believed.

## 5. Every result is kept, bound and checkable

A host agent does excellent work in the moment and leaves its outputs wherever the moment put them. After a few weeks, a workspace with nothing around it fills with scripts, tables and notes that no one can tie back to the inputs and method that made them. Keeping research in order is not what host agents are built for, and it is one of the main reasons AlphaLattice exists.

Every product is recorded under an identity, kept where its kind belongs, and found again by asking for it, not by remembering a path. That covers an input version, a study, a book, published positions, an Evidence review, a committee message and a report. A Task, its Goal and the request that started it tie the work to why it was done. Cleanup is the person's decision, and a pin keeps what they want kept.

**Why there are hashes everywhere.** They are the price of one promise: every number can be traced to exactly what made it, and anyone can check that without trusting the agent, the maintainer or anyone's memory.
- **An identity is a hash of meaning:** the declared method and parameters, the inputs read, and the computing code followed through its imports. A changed computation gets a new identity and cannot pass as the work it replaced; earlier results keep theirs.
- **A binding ties what was seen to what runs.** A plan's hash travels with the request that runs it, so what runs is what was reviewed. A request naming a stale artifact is refused by name.
- **A seal makes a result final.** A reuse or a publication checks its digest and refuses a mismatch. Nothing edits a sealed record; a correction is a new record beside it.
- **A running Task checks its own code** at every stage boundary, so one result never mixes two versions.
- **Exact identity makes reuse safe.** The same identity is reused, and anything different is computed afresh.

**Identity follows owners.** An identity covers the code of the one owner that computes the result (principle 7), so a change moves exactly the identities it should. Copies spread across modules would miss some and move others. A table of the identities an edit to each package may move only shrinks, and a change that reaches further is refused with the responsible import named. A refactor that moves an identity with every value equal is recorded with its reason, so results stay comparable and the change stays visible.

**The discipline compounds.** With every artifact recorded and addressable, a later capability is largely a matter of reading the right records. Convening the investment committee means handing each member a bundle drawn from the date's published positions, their Risk and their Evidence, and the debate follows. The delivery report and the evaluation's walks read the same records.

**Checking should be nearly free,** because slow verification invites skipping it. The direction:
- compute a digest once, when a record is written, and carry it;
- compose identities from their parts' digests, so a check reads only what changed;
- verify fully at trust boundaries (admission, reuse and publication), not on every read;
- keep the hashing in compiled code.

Measurement leads the work. In one early profile, nearly half of a publication's time went to hashing the same objects up to five times; removing the repeats left every digest byte-identical.

## 6. Quick where someone waits, patient and watched everywhere else

The scarce resources are the person's waiting time and the agent's turn; memory and disk are plentiful. So work is arranged by who is waiting for it:
- Long work runs as a Task on the Host, outside the agent's turn. The agent makes one wait call and is woken when the Task ends.
- Work that must wait for the market is held until its data is ready, then runs by itself.
- A run uses the CPU budget it is given, sized to the machine by default, so the person can keep working. Workers and threads change when a result arrives, never what it is, and stay out of its identity. Where a setting could move a last bit, a sealed canary proves it does not: a model fitted on more than one thread must reproduce the trees, curve and predictions sealed at one, or the fit is refused by name.
- Nothing computed once is computed again. Studies are reused by exact identity, and Evidence keeps its relevance scores by content across restarts.

**Background work is watched.** Handing work off is safe only if someone notices when it stops. A host agent waiting on a long program cannot tell a slow run from a dead one, so it waits on or starts again. The Host runs a supervisor outside the processes that do the numerical work. It reads every unfinished Task and classifies it by rule, never by a model's guess: a heartbeat gone stale, a runner the Host can no longer see, work that stopped advancing, a Task awaiting recovery, or one that nothing drives. A refusal, a decision left to the person or a provider's deferral is the Task's own state, not an incident.

Each finding becomes an incident record: when it was seen, the remedies offered, each recovery attempt and its resolution. The incident travels with the Task's answer, so a waiting agent wakes on the problem instead of waiting forever. It is offered the way on (recover, cancel or re-plan) through the ordinary operations, with no extra power. If the Host itself stops, the verified stages are kept and the Task resumes when the Host is served again. The Workbench shows the person the same Tasks and incidents.

The direction:
- **A cache of derived artifacts shared across processes**, such as decoded arrays, verified models and Feature panels, keyed by content identity within a budget. Eviction weighs recompute cost and use against size and pins the active strategy's current inputs and models, so a one-off scan cannot flush what is used every day. A hit still proves its content.
- **Sealing in two halves.** The identity commit stays on the waiting path; compression, archiving and full re-verification move behind it.
- **A low-priority background lane** for archiving, verification, cleanup, indexes and the next day's precomputation, yielding at once to foreground work.

Background work stays visible as Tasks, never an invisible world beside the research. Speed never buys a weaker answer: no check is skipped and no number moves.

## 7. The agent's hands: one command interface, visible declarations, one owner per responsibility

The command interface has an object-and-action grammar, named flags, structured answers and an exit code for each outcome (principle 9). Research declarations are YAML or JSON, so a method can be read and compared before it runs instead of being reconstructed from improvised commands. A [formula factor](formula-factors.md) shows the path end to end: its expression and preprocessing recipe are part of the executed specification, so a changed recipe cannot borrow an earlier recipe's measured effect.

Every answer carries its status, a refusal's words, and the next requests its owner will accept. The agent sends the chosen one as it stands, for example `alphalattice request --from answer.json --action run`, because a rebuilt request can lose its selection, binding or authority. The [saved-answer client](../../src/alphalattice/interface/local_application/client.py) refuses an unresolved choice.

**One owner per responsibility.** Each responsibility has one owner, for admission and execution alike. A change extends that owner and deletes what it displaces. It never adds a second path, a wrapper for one caller or a copied block. Two places that decide the same thing drift apart: an answer offers a next step that different code refuses, and the system rejects its own suggestion. With one rule read by both, a correction is made once, and the guide can name the one place that is true. The same rule keeps the code small: a change lands in the smallest owner that can carry it, growth is budgeted, and tests may not outgrow the source they test. Shape guards in the repository refuse growth that would erode this.

These choices cost real effort and paid off beyond their first purpose. Every operation is a command with a typed answer and offered next steps, so the whole system can be walked, perturbed and graded with no person or model in the loop. The deterministic walk, the deliberate mistakes, the intent and loop tests, the committee floor and the redacted problem report all use this one interface. A product operable only through screens could not be tested this way, nor could its agent be held to the same rules.

## 8. The person's window: the Workbench sits beside the conversation

The agent works through commands; the person watches through the Workbench. It is built for where the person actually is: the browser pane of the Claude Code or Codex desktop app, beside the conversation, about half a screen wide. Every surface is designed at that split width first, and its content folds to its own width. Navigation is docked at the right edge, next to the conversation, as the host apps' own panels are; a standalone window can move it left. The Workbench takes the host's light or dark theme and speaks the visual language of the apps it sits in, showing state as marks, chips and numbers rather than banners and sentences. The [Workbench laws](../../src/alphalattice/interface/local_application/assets/workbench-source/LAWS.md) hold these rules.

It is one product with the command interface, not a second one. Its pages show the same Goals, Tasks, results, team conversation, committee floor and incidents that the agent's answers carry, and they update as Tasks move. The agent opens the launch link once so the person can watch, and never clicks through the Workbench to show its work. The person can act there directly, and every action there also has a command route, so nothing the person sees is out of the agent's reach (principle 1). Answer details and pages read in English or Chinese, while codes and next actions stay in English for the agent.

## 9. Familiar forms: borrow what the model already knows

Wherever the product needs a form, whether a command, a file, an interface or a role, it takes the most widely used one in its field. A language model has met the established forms throughout its training, so an agent working in them already knows the shape. It guesses right more often and recovers sooner from a wrong guess. Every invented rule is a cost paid again each session, in help reads, schema reads and correction rounds. So the product adds a rule of its own only where no established form reaches, and states why.
- **The command line** follows Git and the Unix and GNU tools: an object and an action, as in `git remote add`; a positional operand for the one thing acted on, as in `goal show <id>`, with named options; `-` for standard input and `--` to end the options; `--help` at every level; the answer on standard output and progress on standard error; one verb for each meaning.
- **The agent's files** use the forms its host already reads: an `AGENTS.md` guide at the root, Skills as folders with a `SKILL.md`, and specialist cards in each host's own agent format.
- **Declarations and answers** are YAML and JSON, with each request's shape published as JSON Schema. Data at rest is Parquet, and versions follow PEP 440.
- **The code** follows the patterns of widely used open-source projects where they fit. An Alpha model adapter has `fit` and `predict`, the interface Python's machine-learning libraries share. A result is addressed by a hash of its content and origin, much as version control addresses its objects. An evaluation trace is a sequence of timed steps with namespaced attributes, in JSON Lines, a shape common tracing tools read. None of this depends on the projects it resembles.
- **The roles and words** come from the investment profession. The committee seats a portfolio manager, Alpha and Risk specialists and a chief risk officer, with the duties those titles carry; studies speak of rank IC, turnover and Ledoit-Wolf shrinkage. A model asked to act as a chief risk officer already knows the part, so its card states only what is particular here.

The aim is not imitation but a lighter load: the less an agent must learn that is particular to AlphaLattice, the more of its attention goes to the research. People gain the same way. Anyone who knows Git, YAML or an investment committee reads AlphaLattice without a manual.

## 10. Instructions are code, and the system learns by editing them

The agent guide, the role cards, the Skills and every answer's words decide what the agent does next, so they change in the same change as the behaviour they describe. A renamed command is renamed in its guidance, and every offered request must be accepted by its owner. A stale line does not crash; it quietly sends an agent the wrong way. One source produces both hosts' guidance, and the [materializer](../../scripts/materialize_claude_host.py) refuses drift between them.

Recovery is part of the contract. A stopped Task's answer always names its way on and never asks the agent to wait on work that will not move. A cancelled verification sweep, for example, offers a bound re-plan that starts a new Task while the cancelled one stays in history. The [recovery test](../../tests/portfolio_strategy_lab/test_task_recovery_view.py) sends that offer from the saved answer and checks the new work.

The same text is how the system learns, with no fine-tuning and no training service. The flywheel (principle 12) improves the shared guidance, and each person's own layer improves at their pace. Their preferences, corrections and local method live in their own files: memories, a local guide, and additions to the cards and Skills. The agent reads the layer first and keeps it current; a request to report results in a particular format becomes a small dated memory. It is plain text the person can read, edit, version and reverse. No release writes it, it is backed up before an upgrade, and it refines the work without widening what the shipped guide permits. The [maintenance Skill](../../.agents/skills/alphalattice-maintenance/SKILL.md) says what to record and how.

## 11. Agent mistakes are expected; a system that cannot correct them is the defect

An agent will pick the wrong action, quote a figure incorrectly or misread a held result. Models change, hosts differ, and no guide covers every case. Lengthening the guide to prevent each mistake makes it longer and less reliable, and fixing one behaviour that way can quietly break another.

So the system assumes the agent errs, and corrects it. Every refusal names the rule that failed and offers the request that resolves it, keeping the work already done. A refusal that only says no leaves the same mistake waiting for the next attempt. When an answer itself sends the agent down a false route, the answer is fixed at its owner; the guide grows only when the agent truly lacks a rule.

The committee's number refusal shows the pattern: the judgment can stay, and the figure becomes an alias that can be checked. The [feedback-grader tests](../../tests/evaluation/test_feedback.py) grade exactly this loop. A refused attempt followed by an accepted, bound correction converges; a safety breach on any turn fails the attempt, even if a later turn succeeds.

## 12. A flywheel over the whole loop, measured like an optimizer

The object of improvement is the whole loop, not the model's first reply: guidance, agent, command, system answer and the next decision. Because AlphaLattice owns both its guidance and its answers, a failure is fixed on the side where it broke instead of being worked around in a prompt. That separates it from tuning prompts against a model judge:
- the checks are facts the system can verify: was the offered request accepted, who decided, did the numbers stay equal;
- numbers sit behind a wall that no improvement may cross;
- the decider is graded as well as the route;
- each scenario carries its clock and starting state.

It turns as a flywheel (observe, evaluate, change, observe again) in three places that share one scenario form and one record: simulated runs at every merge, a real end-to-end run before a release, and the problems people report from daily use, which the agent files as a redacted report on their yes. Every failure becomes a scenario that fails first and stays in the merge set, so a fixed defect stays fixed. The cheapest exact checks turn it most often: a deterministic walk that sends every offered next step to a real running Host, and deliberate mistakes that check every refusal leads back. Tests with models are fewer, and they confirm what those checks find.

The method is that of training a model: measure at the current point, take a small step, measure again.
- **A stale gradient misleads.** Old answers describe the version that produced them, so every check runs on the current system with its inputs regenerated.
- **Steps stay small and attributable.** A broad edit makes the cause of a change unknowable.
- **Measurement is statistics.** Comparisons are paired under fixed models, effort, clocks and scenarios. With few samples per case and many cases, a gate raises false alarms by chance alone, so a flagged regression is confirmed before it counts, and totals are tested beside single rows.
- **The graders are classifiers too.** They are checked against planted cases and raw answers for false passes and false failures.
- **Safety is a constraint, not a score.** It is checked on every turn, and convergence is bounded.

The [scenario forms](../../src/devtools/evaluation/forms.py) and [feedback grader](../../src/devtools/evaluation/feedback.py) are public, with small synthetic examples. The larger scenario library is held out, because a public test set gets optimized against and stops measuring. Results from small proxy models are a pessimistic lower bound, not a claim about every session.

## 13. The person's agent extends the framework

AlphaLattice is not a closed product with a plugin slot. The whole source sits in the person's own checkout, and their coding agent can change it as well as operate it. A person who wants their own Alpha model, Feature or strategy asks their agent to add it, and the agent reads the code it extends. Unlike conventional software, the source is meant to be read and changed by the person's own agent.

So extension is designed in. Each kind of module has one declared place to extend, a scaffold or declaration to start from, and a contract to meet; the [extension guide](extending.md) names each owner.
- **Alpha models.** The `model scaffold` command writes an editable declaration and, from it, an adapter with `fit` and `predict` and its contract test. `model check` and a sandbox trial on a copy of the workspace must pass before the person may activate it.
- **Features.** A formula factor is a declaration in a bounded numerical language, and a new kernel registers with the Feature catalog.
- **Strategies.** A package declares its components and score sources in one owner.
- **Risk methods and data sources** are being opened the same way; a provider interface for market data already defines what a source must supply.

**One registry for each kind.** An extension joins by registering, never by being wired in by name. Alpha models and Features already work this way: the workspace's registry holds each activation with who made it, when, its identity and the checks it passed, and the studies read only the registry. Adding or retiring one is an entry, made and reversed by the person's decision. Strategies, risk methods and data sources are to follow, so that every kind shares one lifecycle: declare, check, try on a copy, activate, deactivate.

Openness does not loosen integrity. An extension gets its own identity and evidence (principle 5), so it cannot pass itself off as the work it replaced. Activation stays the person's decision after the contract and the trial pass. Before an upgrade, the person's changes are backed up and compared, so the framework moves forward without erasing what they built on it.

## 14. A release is earned

Each minor version's scope and acceptance are fixed before work begins, and corrective releases follow until the scope runs smoothly. Work toward the next minor version lives on the public continuation branch, with beta releases as its period of real use. Acceptance is more than a good demonstration:
- the loop's checks pass;
- the first-day intentions work with two small proxy models;
- one real research run completes from a single sentence with no repair;
- a fresh installation works.

The [verification workflow](../../.github/workflows/ci.yml) checks the source checkout and an installed wheel as separate paths, on the release and continuation branches.

Claims stay within their evidence. Windows is the verified platform; macOS and Linux are not yet verified. A completed study establishes its recorded result under its inputs and method, and promises no return. Every known limit ships with the release that carries it.

## 15. One door for work and exchanges

The local Host is the common entry for agents and people. Commands and Workbench actions reach the same owners and admission rules. Exchanges between agents also pass through it, so assignments, answers and acceptances become records that can be checked together. This keeps the account of the work beside the rules that admit it.

In the committee, each member sends its message with its own key. The Host records its author and admits it to the floor, where the person can inspect the exchange.

## 16. The result comes from the loop

Formulas and declared rules belong to the program, judgment under ambiguity to the agent, and authority to the person. The program computes and records, then offers a next request or explains a refusal; the agent reads it, judges and acts again. This loop connects interpretation to checkable facts.

The person can be in the loop at a decision, on it by watching and intervening, or out of it for explicitly delegated acts. Each act retains its scope and authority limits. The first use shows this: its named preparation decisions proceed under a delegation that expires with the goal or its allotted hours.

## 17. Management by exception

A decision that a declared rule can make belongs to the Host. Under the same rule, the same evidence receives the same decision, recorded and disclosed. The agent is asked only where no rule resolves the case or a bound is exceeded; authority remains the person's. Judgment goes where it can change the outcome.

The first-use data rule retains an isolated verified move with its caveat or sets it aside recoverably, and escalates beyond its bound. When the Host itself fails, the agent keeps a recorded way in through diagnostics and recovery requests. Recovery follows the same door and authority checks, preserving earlier verified work.

## 18. What agents have learned keeps working

Commands, answer fields, codes and message kinds are versioned interfaces. We take Linux's concern for its userspace as a design reference: what agents have learned keeps working. A retirement is announced as deprecated, with its replacement, before removal. A stable contract lets research continue across upgrades.

A command being retired names its replacement while the old form still works. Guidance and release notes carry the change together, so the next session can follow it deliberately.

## 19. The record is the lever

An agent system improves at the speed its history can be read. AlphaLattice keeps that history in one append-only ledger: each request, decision and result, who made it and on what evidence, with sealed results an agent can verify. Codes for each kind of question and links from a decision to its later outcome are being added to that record patch by patch.

Agents can then do the analysis a person once did by hand. They find recurring exceptions, slow paths and offers that fail, and propose a rule, a change of words or a change of code. A decision agents make the same way on the same evidence, again and again, becomes a declared rule, so its next occurrence costs nobody a turn.

A proposal changes nothing on its own. It passes the same checks as any change, a changed policy is a disclosed successor, and the person keeps the scope. The measure is the outcome, not a falling count of questions.

The maintainer's experience building AlphaLattice with agents is that how history is recorded decides how fast the system improves. Development follows the same loop: a defect that escapes review is recorded against the change that introduced it, and a review finding against the question that should have caught it. The record gives the next agent a cause to investigate and a check to improve.

the maintainer
