"""The operation registry: every typed operation, its CLI name and its fields (C1 rule 3).

The operations are the ``PortfolioResearchOperation`` literal; their fields are the request's
own field contract; which only read is the activity ledger's ``READ_OPERATIONS``. This module
reads those owners rather than copying them, and derives from them the one thing nobody kept:
each operation's name in the CLI grammar, ``alphalattice <noun> <verb>``. The gate
(``devtools.architecture.operation_registry``) checks that the owners agree with each other and
with the Host's routes, so a new operation cannot be registered in one place and missed in
another.

The CLI does not import this module: it reads the table this module writes beside itself,
``operations.json`` (each command's operation, each operation's fields, each field's JSON
types), with the standard library, so a command costs a process start and nothing more (K1).
The gate refuses a table that is not ``table()``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, get_args

from alphalattice.interface.local_application.activity import READ_OPERATIONS
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperation,
    PortfolioResearchOperationRequest,
    PortfolioResearchRequestDocument,
)

OPERATIONS: Final[tuple[str, ...]] = tuple(get_args(PortfolioResearchOperation.__value__))


@dataclass(frozen=True, slots=True)
class Command:
    """One command of the CLI: ``alphalattice <noun> <verb>`` (LAWS OP2; the grammar's rules)."""

    noun: str
    """The object: what a person reasons about; a sub-object's reads and acts share its noun."""
    verb: str
    """The action, one meaning in every object (`plan` seals, `run` starts the work, ...)."""
    purpose: str
    """What the command does, in one line: its help, the cards' and the Skill's words."""
    positional: str | None = None
    """The field taken as the positional argument: the one instance of its object it acts on."""
    collection: bool = False
    """Whether this read lists independent records, each with its own read or refusal (OP4).
    Numerical arrays and evidence inside one exact result remain that result's single read."""


GRAMMAR: Final[dict[str, Command]] = {
    "ACTIVITY_LIST": Command(
        "activity",
        "list",
        "Reads one page of the activity feed, with the state of any watched Tasks.",
        collection=True,
    ),
    "ACTIVITY_RECENT": Command(
        "activity",
        "recent",
        "Lists the newest requests grouped by agent session and goal, each with its read.",
        collection=True,
    ),
    "ACTIVITY_REFUSALS": Command(
        "activity",
        "refusals",
        "Counts refusals over the last days (30 by default), most frequent first.",
        collection=True,
    ),
    "AGENT_ANSWER_SUBMIT": Command(
        "bundle",
        "submit",
        "Submits an agent's answer to its bundle; the Host accepts it or names each problem.",
    ),
    "AGENT_BUNDLE_PREPARE": Command(
        "bundle",
        "prepare",
        "Writes a specialist's complete bundle into a new directory; names its answer file.",
    ),
    "CANCEL": Command(
        "task",
        "cancel",
        "Requests cancellation of a Task, optionally against the Task version you read.",
        "task_id",
    ),
    "COMPARE": Command(
        "result",
        "compare",
        "Compares two saved installed-strategy results: which controls differ and how metrics "
        "compare; no winner.",
    ),
    "CONTROLS": Command(
        "strategy-book",
        "controls",
        "Lists the controls of an installed strategy's Portfolio book, with defaults and limits.",
    ),
    "CPU_BUDGET_SET": Command(
        "cpu-budget",
        "set",
        "Sets the CPU budget (auto or a core count) or how many Tasks may wait.",
    ),
    "CPU_BUDGET_SHOW": Command(
        "cpu-budget",
        "show",
        "Shows the CPU budget, the machine's load and how many Tasks may wait.",
    ),
    "CRO_REVIEW": Command(
        "review",
        "request",
        "Requests a CRO review of a book; the product runs no model, so use `bundle prepare`.",
    ),
    "CRO_REVIEW_DOSSIER": Command(
        "review",
        "dossier",
        "Delivers a book's CRO dossier and answer contract, in parts when it is large.",
    ),
    "CRO_REVIEW_FINDING": Command(
        "review",
        "finding",
        "Delivers one CRO finding with its verified cited evidence.",
        "finding_handle",
    ),
    "CRO_REVIEW_SUBMIT": Command(
        "review",
        "submit",
        "Submits a CRO's answer to its dossier; accepted risks publish as a Task.",
    ),
    "DATA_CHANGE_CONFIRM": Command(
        "data-update",
        "confirm",
        "Approves a proposed data change by plan hash, admitting its Task; `run` then applies it.",
    ),
    "DATA_ISSUES": Command(
        "issue",
        "list",
        "Lists data issues with their evidence, options, decisions and grants, a page at a time.",
        collection=True,
    ),
    "DATA_ISSUE_CONFIRM": Command(
        "issue",
        "confirm",
        "Records a data issue's chosen option; from the CLI only under a person's grant.",
        "data_issue_case_token",
    ),
    "DATA_ISSUE_DELEGATE": Command(
        "issue",
        "delegate",
        "Grants automation the right to confirm one option for a preparation blocked on data "
        "review.",
        "data_issue_case_token",
    ),
    "DATA_ISSUE_PREVIEW": Command(
        "issue",
        "preview",
        "Previews one data-issue option's consequences; nothing is applied.",
        "data_issue_case_token",
    ),
    "DATA_ISSUE_REVOKE": Command("issue", "revoke", "Revokes a data-issue grant by its hash."),
    "DATA_UPDATE_PLAN": Command(
        "data-update",
        "plan",
        "Plans a data update: due source checks, Data and Feature upkeep, any membership change; "
        "fetches nothing.",
    ),
    "DATA_UPDATE_READBACK": Command(
        "data-update",
        "show",
        "Shows the workspace's data state and the latest data update Task, or the one named.",
    ),
    "DATA_UPDATE_RUN": Command(
        "data-update",
        "run",
        "Runs a planned data update as a Task, within the workspace's network access.",
    ),
    "EVENT_DECLARE": Command(
        "event",
        "declare",
        "Records an event this client declares about its own work in the activity feed.",
    ),
    "WAKE_REGISTER": Command(
        "activity",
        "notify",
        "Asks the Host to queue one line to a Codex thread, naming the exact read, when the Task "
        "ends, needs a decision or is deferred; returns at once, the wake held in the Task's "
        "journal across turns and restarts.",
    ),
    "SESSION_USAGE_READ": Command(
        "session",
        "usage",
        "Reads the usage of the agent Sessions bound to this workspace now, from their own "
        "files and the children they record; nothing else runs, and reading off reads nothing.",
    ),
    "USAGE_READING": Command(
        "usage-reading",
        "show",
        "Shows whether the Host may read the bound agent Sessions' own files for usage, and "
        "what it reads.",
    ),
    "USAGE_READING_SET": Command(
        "usage-reading",
        "set",
        "Turns reading the bound agent Sessions' usage on or off for this workspace.",
    ),
    "EVIDENCE_ANALYSIS_SUBMIT": Command(
        "evidence",
        "submit",
        "Submits an Analyst's answer to a prepared packet; accepted findings publish as a Task.",
    ),
    "EVIDENCE_CONTINUE": Command(
        "evidence",
        "continue",
        "Admits one more bounded source-reading session over a prepared packet, as a Task.",
    ),
    "EVIDENCE_CRO": Command(
        "evidence",
        "show",
        "Shows a book's Evidence and CRO state, from preparation to published review.",
    ),
    "EVIDENCE_CRO_EXPORT": Command(
        "evidence",
        "export",
        "Exports a published CRO review with its dossier and verified evidence spans.",
    ),
    "EVIDENCE_DOCUMENTS": Command(
        "evidence",
        "documents",
        "Lists the workspace's retained SEC documents, newest first, a page at a time; fetches "
        "nothing.",
        collection=True,
    ),
    "EVIDENCE_LEDGER": Command(
        "evidence",
        "ledger",
        "Reads a book's issuer-topic reading ledger from its sealed packets, twenty groups a page.",
        collection=True,
    ),
    "EVIDENCE_PACKET": Command(
        "evidence",
        "packet",
        "Delivers one prepared evidence packet for an Analyst, whole, in parts or as a chosen "
        "detail.",
    ),
    "EVIDENCE_PREPARE": Command(
        "evidence",
        "run",
        "Prepares a book's evidence packets as a Task: acquires, reads and seals sources; no "
        "findings.",
    ),
    "EVIDENCE_PREVIEW": Command(
        "evidence",
        "preview",
        "Previews a book's evidence preparation: scope, sources and prerequisites; no Task or "
        "acquisition.",
    ),
    "EVIDENCE_REFRESH": Command(
        "evidence",
        "refresh",
        "Requests an evidence refresh; the product runs no model, so use `bundle prepare`.",
    ),
    "EVIDENCE_SELECT": Command(
        "evidence",
        "select",
        "Records which published analysis a book's review reads; refuses one that is not eligible.",
    ),
    "EXPERIMENTS": Command(
        "study",
        "list",
        "Lists every study Task with its kind, input and declared choices.",
        collection=True,
    ),
    "EXPERIMENT_ALPHA_COMPARE": Command(
        "candidate",
        "compare",
        "Compares two saved Alpha candidates from stored evidence; names no winner, recomputes "
        "nothing.",
    ),
    "EXPERIMENT_COMPARE": Command(
        "study",
        "compare",
        "Compares two completed Portfolio books on the same input and support; names no winner.",
    ),
    "EXPERIMENT_CONTINUE": Command(
        "study",
        "continue",
        "Continues a saved study by drafting, planning and running its next version in one "
        "request.",
        "task_id",
    ),
    "EXPERIMENT_CONTROLS": Command(
        "study",
        "controls",
        "Shows the controls and an editable declaration for a new study on a research input.",
    ),
    "EXPERIMENT_CURATE": Command(
        "curation",
        "submit",
        "Records a Factor study's curation decision: which factors go on, and why; no Task.",
        "task_id",
    ),
    "EXPERIMENT_CURATION": Command(
        "curation",
        "show",
        "Shows a completed Factor study's curation choices and what a decision must name.",
        "task_id",
    ),
    "EXPERIMENT_DELIVERY_EXPORT": Command(
        "book",
        "export",
        "Exports one delivery report binding a selected book with its comparison, Risk link and "
        "review.",
    ),
    "EXPERIMENT_DRAFT": Command(
        "study",
        "draft",
        "Copies a completed study's declaration for editing, onto its own input or a chosen one.",
        "task_id",
    ),
    "EXPERIMENT_EXPORT": Command(
        "study",
        "export",
        "Exports a verified study as JSON, YAML and HTML with an export hash.",
        "task_id",
    ),
    "EXPERIMENT_FOUNDATIONS": Command(
        "foundation",
        "list",
        "Lists Foundation admissions with each current, historical or refused read.",
        collection=True,
    ),
    "EXPERIMENT_FOUNDATION_DRAFT": Command(
        "foundation",
        "draft",
        "Drafts the Alpha declaration that builds on a sealed Foundation.",
        "foundation_admission_hash",
    ),
    "EXPERIMENT_FOUNDATION_EXPORT": Command(
        "foundation",
        "export",
        "Exports a sealed Foundation admission as JSON.",
        "foundation_admission_hash",
    ),
    "EXPERIMENT_FOUNDATION_PREVIEW": Command(
        "foundation",
        "preview",
        "Previews the Foundation a curated Factor study would seal; nothing is sealed.",
    ),
    "EXPERIMENT_FOUNDATION_READBACK": Command(
        "foundation",
        "show",
        "Shows one sealed Foundation by its admission hash, read from its recorded graph.",
        "foundation_admission_hash",
    ),
    "EXPERIMENT_FOUNDATION_SUMMARY": Command(
        "foundation",
        "summary",
        "Reads one sealed admission's label metadata without verifying its source graph.",
    ),
    "EXPERIMENT_FOUNDATION_SEAL": Command(
        "foundation",
        "seal",
        "Seals the Foundation admission just previewed, named by its admission hash.",
        "foundation_admission_hash",
    ),
    "EXPERIMENT_HANDOFF_PREVIEW": Command(
        "handoff",
        "preview",
        "Previews the Alpha declaration built on a curated Factor decision; nothing is planned or "
        "run.",
    ),
    "EXPERIMENT_LINK_RISK": Command(
        "risk-link",
        "add",
        "Attaches a completed Risk study's report to a Portfolio book as evidence; weights "
        "unchanged.",
    ),
    "EXPERIMENT_PLAN": Command(
        "study",
        "plan",
        "Plans a study from its declaration and keeps the plan for an hour; nothing runs.",
    ),
    "EXPERIMENT_PORTFOLIO_DRAFT": Command(
        "book",
        "draft",
        "Drafts a Portfolio book declaration from one candidate of a completed Alpha study.",
    ),
    "EXPERIMENT_PREVIEW_READBACK": Command(
        "study",
        "show",
        "Shows a kept plan by its hash: available, expired, invalid or missing, and what follows.",
    ),
    "EXPERIMENT_PROMOTE": Command(
        "study",
        "promote",
        "Runs an explored study's declaration on its whole universe as a Task, or reuses it.",
        "task_id",
    ),
    "EXPERIMENT_READBACK": Command(
        "study",
        "show",
        "Shows a saved study verified against its sealed evidence, its standing first.",
        "task_id",
    ),
    "EXPERIMENT_REPLAY": Command(
        "study",
        "verify",
        "Checks a saved study still matches the installed code and evidence; recomputes nothing, "
        "admits no Task.",
        "task_id",
    ),
    "EXPERIMENT_RISK_EXPORT": Command(
        "risk-link",
        "export",
        "Exports one linked Risk report of a Portfolio book as HTML with its hash.",
    ),
    "EXPERIMENT_RISK_LINKS": Command(
        "risk-link",
        "list",
        "Lists the Risk reports linked to a Portfolio book, each with its export request.",
        collection=True,
    ),
    "EXPERIMENT_RUN": Command(
        "study", "run", "Runs a kept plan as a Task, or reuses the identical completed study."
    ),
    "EXPERIMENT_SUMMARY": Command(
        "study",
        "summary",
        "Shows a completed Alpha study's recorded model, training facts and metrics, without "
        "re-verifying evidence.",
        "task_id",
    ),
    "EXPERIMENT_VERIFY_ALL": Command(
        "study",
        "verify-all",
        "Verifies every saved study's sealed evidence in full, as a Task that yields to waiting "
        "work.",
    ),
    "EXPORT": Command(
        "result",
        "export",
        "Returns the export manifest of a saved result; recomputes nothing.",
        "result_hash",
    ),
    "FEATURE_ACTIVATE": Command(
        "feature",
        "activate",
        "Activates a reviewed formula factor into the workspace's daily feature catalog.",
        "feature_factor_id",
    ),
    "FEATURE_CATALOG_BUILD": Command(
        "feature", "run", "Builds a planned feature's raw or preprocessed values as a Task."
    ),
    "FEATURE_CATALOG_BUILD_READBACK": Command(
        "feature",
        "show",
        "Shows a feature build Task: its verified columns, counts and what they can feed.",
    ),
    "FEATURE_CATALOG_CONTROLS": Command(
        "feature",
        "controls",
        "Shows the formula language, admitted recipes and an editable feature declaration for an "
        "input.",
    ),
    "FEATURE_CATALOG_PLAN": Command(
        "feature",
        "plan",
        "Plans a feature change from its declaration and saves the plan; nothing is built.",
    ),
    "FEATURE_CATALOG_READBACK": Command(
        "feature", "show", "Shows a saved feature plan: its changes, work and next requests."
    ),
    "FEATURE_DEACTIVATE": Command(
        "feature",
        "deactivate",
        "Removes an activated formula factor from the workspace's daily feature catalog.",
        "feature_factor_id",
    ),
    "FEATURE_EXTENSIONS": Command(
        "feature",
        "list",
        "Lists the formula factors the workspace's plans declare, with activation and latest "
        "trial.",
        collection=True,
    ),
    "FEATURE_REVIEW": Command(
        "feature",
        "review",
        "Shows a formula factor's review packet: contract, trials, build coverage and activation.",
        "feature_factor_id",
    ),
    "FEATURE_TRIAL": Command(
        "trial",
        "run",
        "Starts or reopens a trial: builds a planned feature, screens it, reruns a finished study.",
    ),
    "FEATURE_TRIALS": Command(
        "trial",
        "list",
        "Lists feature trials with their state and current step.",
        collection=True,
    ),
    "FEATURE_TRIAL_READBACK": Command(
        "trial",
        "show",
        "Shows a feature trial's steps and, once complete, its comparison; changes nothing.",
        "feature_trial_id",
    ),
    "FINALIZATION": Command(
        "release",
        "show",
        "Shows a frozen candidate's release state and next permitted action; grants nothing.",
        "candidate_hash",
    ),
    "FREEZE": Command(
        "release",
        "freeze",
        "Freezes a saved development result, by its hash, as a candidate for finalization.",
    ),
    "GOAL_ABANDON": Command(
        "goal", "abandon", "Closes an open goal as abandoned, recording why.", "goal_id"
    ),
    "GOAL_ATTACH": Command(
        "goal", "attach", "Attaches an exact read to the goal as evidence under a stage.", "goal_id"
    ),
    "GOAL_CONTINUE": Command(
        "goal",
        "continue",
        "Returns the draft that continues a study linked to the goal; plans and runs nothing.",
    ),
    "GOAL_EXPORT": Command(
        "goal",
        "export",
        "Exports a goal's verified record, with the book deliveries it names, as JSON, YAML or "
        "HTML.",
        "goal_id",
    ),
    "GOAL_LIST": Command(
        "goal",
        "list",
        "Lists goals newest first, a page at a time, or those of one agent session.",
        collection=True,
    ),
    "GOAL_NARRATIVE": Command(
        "goal",
        "narrative",
        "Shows a goal's saved revision as recorded, without re-verifying its evidence.",
        "goal_id",
    ),
    "GOAL_REFERENCE": Command(
        "goal",
        "reference",
        "Verifies one named reference of an exact saved goal revision at its owner.",
    ),
    "GOAL_NOTE": Command(
        "goal",
        "note",
        "Records a decision or conclusion in the goal's record; a conclusion must cite evidence.",
        "goal_id",
    ),
    "GOAL_OPEN": Command(
        "goal",
        "open",
        "Opens a goal from its declaration and binds this agent session to it.",
        "goal_id",
    ),
    "GOAL_REVISE": Command(
        "goal",
        "revise",
        "Revises an open goal's declaration with a reason; a changed objective marks earlier "
        "evidence post hoc.",
        "goal_id",
    ),
    "GOAL_SCHEMA": Command(
        "goal",
        "schema",
        "Prints the goal declaration, reference, note and submission schemas, with a starter "
        "declaration.",
    ),
    "GOAL_SHOW": Command(
        "goal",
        "show",
        "Shows a goal's latest or a chosen revision, re-verifying every reference at its owner.",
        "goal_id",
    ),
    "GOAL_SUBMIT": Command(
        "goal",
        "submit",
        "Submits the completion document; the Host seals the goal complete or lists what is "
        "missing.",
        "goal_id",
    ),
    "GOAL_TAKE": Command(
        "goal",
        "take",
        "Binds this agent session to an open goal, so its requests are recorded under it.",
        "goal_id",
    ),
    "MODEL_ACTIVATE": Command(
        "model",
        "activate",
        "Activates a reviewed Alpha model for this workspace's research.",
        "model_id",
    ),
    "MODEL_DEACTIVATE": Command(
        "model",
        "deactivate",
        "Removes an activated Alpha model from this workspace's catalog.",
        "model_id",
    ),
    "MODEL_EXTENSIONS": Command(
        "model",
        "list",
        "Lists installed and extension Alpha models, each with its review packet.",
        collection=True,
    ),
    "MODEL_TRAINING_INPUT_PLAN": Command(
        "training",
        "plan",
        "Plans model-training inputs for a component on a research input; no model is fit.",
    ),
    "MODEL_TRAINING_INPUT_PREPARE": Command(
        "training",
        "run",
        "Prepares a planned component's training inputs as a Task, or reuses them.",
    ),
    "MODEL_TRAINING_INPUT_READBACK": Command(
        "training",
        "show",
        "Shows a training-input Task and the prepared sources and support it published.",
    ),
    "NETWORK_ACCESS": Command(
        "network", "show", "Shows whether this workspace may reach the network and what decided it."
    ),
    "NETWORK_ACCESS_SET": Command(
        "network", "set", "Turns this workspace's network access on or off."
    ),
    "OPERATION_LIST": Command(
        "operation",
        "list",
        "Lists every operation with its required and allowed fields, and the request schema.",
        collection=True,
    ),
    "PENDING_DECISIONS": Command(
        "decision",
        "list",
        "Lists everything waiting on a person, each with the request that takes it.",
        collection=True,
    ),
    "PLAN": Command(
        "strategy-book",
        "preview",
        "Previews an installed strategy's book from its controls: window, reuse and work; nothing "
        "runs.",
    ),
    "PORTFOLIO_UPDATE_PLAN": Command(
        "portfolio-update",
        "plan",
        "Plans a Portfolio update that settles observed sessions for an installed package; fits "
        "nothing.",
    ),
    "PORTFOLIO_UPDATE_READBACK": Command(
        "portfolio-update",
        "show",
        "Shows a Portfolio update Task, or a strategy's latest, with the book it published.",
    ),
    "PORTFOLIO_UPDATE_RUN": Command(
        "portfolio-update",
        "run",
        "Runs a planned Portfolio update as a Task, or reuses the identical published one.",
    ),
    "RECOVER": Command(
        "recovery",
        "run",
        "Resumes a Task that needs recovery, or retries a blocked study whose artifact was "
        "repaired.",
        "task_id",
    ),
    "REPORT": Command(
        "result",
        "show",
        "Opens a saved result's report: window, book, controls and readouts; reruns nothing.",
        "result_hash",
    ),
    "RESEARCH_HISTORY": Command(
        "history",
        "list",
        "Lists past studies, installed results, updates and CRO reviews, a page at a time.",
        collection=True,
    ),
    "RESEARCH_INPUTS": Command(
        "input",
        "list",
        "Lists each research input and its versions, naming any that cannot be read.",
        collection=True,
    ),
    "RESEARCH_INPUT_CONFIRM": Command(
        "input",
        "confirm",
        "Confirms a planned research input capture, which seals a new input version as a Task.",
    ),
    "RESEARCH_INPUT_PLAN": Command(
        "input",
        "plan",
        "Plans sealing a new version of a research input from the local sources; no network.",
        "research_input_id",
    ),
    "RESEARCH_INPUT_READBACK": Command(
        "input", "show", "Shows an input capture Task and the input version it published."
    ),
    "RESEARCH_STRATEGY_CONTROLS": Command(
        "strategy",
        "controls",
        "Shows the strategy-preparation declaration schema and the completed Alpha and Risk "
        "studies it can use.",
    ),
    "STRATEGY_ACTIVATE": Command(
        "strategy",
        "activate",
        "Runs a reviewed book's installed research strategy forward: the daily update scores, "
        "calibrates and gives its next positions; a person's.",
        "task_id",
    ),
    "STRATEGY_DEACTIVATE": Command(
        "strategy",
        "deactivate",
        "Stops an installed research strategy running forward; its history stays readable; a "
        "person's.",
        "strategy_package_id",
    ),
    "RESEARCH_STRATEGY_INSTALL": Command(
        "strategy",
        "install",
        "Installs a prepared strategy as non-default research in this workspace; the running "
        "service serves its packages at once.",
    ),
    "RESEARCH_STRATEGY_PLAN": Command(
        "strategy",
        "plan",
        "Plans a strategy preparation from promoted Alpha and Risk studies; nothing runs.",
    ),
    "RESEARCH_STRATEGY_PREPARE": Command(
        "strategy",
        "run",
        "Prepares the planned strategy's Portfolio inputs from its Alpha and Risk studies, as a "
        "Task.",
    ),
    "RESEARCH_STRATEGY_READBACK": Command(
        "strategy",
        "show",
        "Shows a strategy preparation Task, what it prepared and what is still missing.",
    ),
    "RESEARCH_UPDATE_AUTOMATION_CONFIGURE": Command(
        "automation",
        "set",
        "Turns the daily research update on or off for named packages.",
    ),
    "RESEARCH_UPDATE_AUTOMATION_READBACK": Command(
        "automation",
        "show",
        "Shows whether the daily research update is on, its packages and next due time.",
    ),
    "RESEARCH_UPDATE_PLAN": Command(
        "research-update",
        "plan",
        "Plans a research update bringing data, scores, calibration and the book to the latest "
        "completed session.",
    ),
    "RESEARCH_UPDATE_READBACK": Command(
        "research-update",
        "show",
        "Shows a research update Task, or a strategy's latest, with the book it published.",
    ),
    "RESEARCH_UPDATE_RUN": Command(
        "research-update",
        "run",
        "Runs a planned research update as one Task, or reuses the identical completed one.",
    ),
    "RESULTS": Command(
        "result",
        "list",
        "Lists result metadata, for one Task when selected; opens no reports.",
        collection=True,
    ),
    "RUN": Command(
        "strategy-book",
        "run",
        "Runs an installed strategy's book from its controls, or reuses an identical saved result.",
    ),
    "STATUS": Command(
        "task",
        "show",
        "Shows one Task's state, stages, timing and incident; may wait for it to move on.",
        "task_id",
    ),
    "STORAGE_CONFIRM": Command(
        "storage", "confirm", "Applies a storage clean-up plan, deleting the input copies it names."
    ),
    "STORAGE_EVIDENCE_REBUILD": Command(
        "storage",
        "rebuild",
        "Rebuilds an evicted evidence index from its committed vectors, with a receipt.",
    ),
    "STORAGE_PIN": Command(
        "storage",
        "pin",
        "Pins or unpins a research input or evidence index against storage clean-up.",
    ),
    "STORAGE_PLAN": Command(
        "storage",
        "plan",
        "Plans a storage clean-up of unpinned input copies and saves it for confirmation.",
    ),
    "STORAGE_CAP_SHOW": Command(
        "storage", "cap", "Shows the operator storage cap, its automatic estimate and free disk."
    ),
    "STORAGE_CAP_SET": Command(
        "storage",
        "set",
        "Sets the workspace storage cap to auto or a positive whole byte count; "
        "preserves retained work.",
    ),
    "STORAGE_READBACK": Command(
        "storage",
        "show",
        "Shows storage use, retained inputs and evidence indexes, pins and pending clean-up.",
    ),
    "STRATEGY_CALIBRATION_PLAN": Command(
        "calibration",
        "plan",
        "Plans calibration of a package's Portfolio input from a published score snapshot; fits "
        "nothing.",
    ),
    "STRATEGY_CALIBRATION_READBACK": Command(
        "calibration",
        "show",
        "Shows a calibration Task, or a strategy's latest, and the Portfolio input it published.",
    ),
    "STRATEGY_CALIBRATION_RUN": Command(
        "calibration",
        "run",
        "Prepares a package's Portfolio input from a calibration plan as a Task, or reuses it.",
    ),
    "STRATEGY_SCORE_PLAN": Command(
        "score",
        "plan",
        "Plans scoring of a package's models for a formation session; nothing is scored yet.",
    ),
    "STRATEGY_SCORE_READBACK": Command(
        "score",
        "show",
        "Shows a scoring Task, or a strategy's latest, and the score snapshot it published.",
    ),
    "STRATEGY_SCORE_RUN": Command(
        "score", "run", "Runs a planned scoring as a Task, or reuses the identical published score."
    ),
    "TASKS": Command(
        "task",
        "list",
        "Lists the workspace's Tasks with their states, or one agent session's, a page at a time.",
        collection=True,
    ),
    "TASK_GUARDIAN": Command(
        "recovery",
        "list",
        "Reads every unfinished Task: liveness, stages, kept work, incidents and permitted "
        "recovery.",
        collection=True,
    ),
    "TASK_INCIDENTS": Command(
        "incident",
        "list",
        "Lists the supervisor's incident records, open first, with what was done about each.",
        collection=True,
    ),
    "TASK_RECOVERY": Command(
        "recovery",
        "show",
        "Shows what stopped a Task, what stays verified and what recovery its owner permits.",
        "task_id",
    ),
    "TASK_REMEDIATE": Command(
        "incident",
        "remediate",
        "Performs one remedy the Host offers for an open incident: cancel, recover or re-plan.",
        "incident_key",
    ),
    "UPGRADE_ACKNOWLEDGE": Command(
        "upgrade",
        "acknowledge",
        "Records the installed identities you saw on the upgrade overview, by their set hash.",
        "upgrade_set_hash",
    ),
    "UPGRADE_OVERVIEW": Command(
        "upgrade",
        "show",
        "Shows what the installed upgrade changed: saved studies, reviews and waiting Tasks.",
    ),
    "WORKSPACE_BACKUP": Command(
        "backup",
        "run",
        "Backs up what the workspace cannot rebuild, outside it, keeping the newest generations.",
    ),
    "WORKSPACE_BACKUPS": Command(
        "backup",
        "list",
        "Lists the kept backup generations and the backup root; makes none.",
        collection=True,
    ),
    "WORKSPACE_PREPARE_CONFIRM": Command(
        "preparation",
        "confirm",
        "Starts a planned workspace preparation: the CLI starts one under the workspace's "
        "first-use goal, or resumes one under a person's grant.",
    ),
    "WORKSPACE_PREPARE_PLAN": Command(
        "preparation",
        "plan",
        "Plans first-use workspace preparation, or recovery of a stopped one; acquires nothing.",
    ),
    "WORKSPACE_PREPARE_READBACK": Command(
        "preparation",
        "show",
        "Shows the latest preparation Task, or the one named, and the inputs it published.",
    ),
    "WORKSPACE_SHOW": Command(
        "workspace",
        "show",
        "Shows the workspace's inputs, recent studies and Tasks, data state, and next steps.",
    ),
}
"""Every operation's command, declared (the user, 2026-10-01: the CLI's grammar). Two operations
share a command only where their required selectors tell them apart (`study show <id>` or
`study show --plan <id>`)."""

PRIMARY: Final[tuple[str, ...]] = (
    "experiment_yaml",
    "experiment_document",
    "feature_document",
    "goal_declaration",
    "goal_submission",
    "goal_reference",
    "goal_statement",
    "experiment_curation",
    "spec",
    "agent_answer",
    "analysis_answer",
    "review_answer",
    "event",
)
"""The document fields `--file <path>` reads, in order: a command reading several takes the
first it allows (a study's declaration as its YAML text)."""

FLAGS: Final[dict[str, str]] = {
    "agent_role": "role",
    "agent_session": "session-id",
    "analysis_context_hash": "context",
    "analysis_publication_hash": "analysis",
    "automation_enabled": "enabled",
    "automation_package_ids": "packages",
    "backup_generations_kept": "keep",
    "bundle_directory": "dir",
    "calibration_plan_hash": "plan",
    "candidate_hash": "candidate",
    "candidate_id": "candidate",
    "change_reason": "reason",
    "citation_entity_id": "citation-entity",
    "citation_page": "citation-page",
    "citation_unit_id": "citation-unit",
    "component_id": "component",
    "model_lifecycle": "lifecycle",
    "continuation_of": "continues",
    "continuation_spans": "spans",
    "storage_cap_bytes": "cap-bytes",
    "cpu_budget": "cores",
    "curation_receipt_hash": "receipt",
    "data_issue_case_token": "case",
    "data_issue_evidence_hash": "evidence-hash",
    "data_issue_grant_hash": "grant",
    "data_issue_option_hash": "option-hash",
    "data_issue_option_id": "option",
    "delivery_budget_bytes": "budget",
    "delivery_commentary": "commentary",
    "delivery_part": "part",
    "delivery_question": "question",
    "documents_page": "page",
    "evidence_as_of": "as-of",
    "evidence_detail": "detail",
    "evidence_index_id": "index",
    "evidence_unit_id": "unit",
    "expected_task_hash": "expected",
    "experiment_kind": "kind",
    "experiment_plan_hash": "plan",
    "experiment_receipt_hash": "receipt",
    "experiment_task_id": "study",
    "extensions_page": "page",
    "factor_task_id": "factor-study",
    "feature_factor_id": "factor",
    "feature_output": "output-dir",
    "feature_plan_hash": "plan",
    "feature_preparation_hash": "preparation",
    "feature_trial_id": "trial",
    "finding_handle": "finding",
    "formation_session": "session",
    "foundation_admission_hash": "admission",
    "goal_hash": "revision",
    "goal_reference_id": "reference",
    "goal_id": "goal",
    "handoff_hash": "handoff",
    "history_cursor": "cursor",
    "history_entry_id": "entry",
    "history_kind": "kind",
    "history_limit": "limit",
    "incident_key": "incident",
    "input_binding_hash": "binding",
    "input_pinned": "pinned",
    "ledger_page": "page",
    "left_candidate_id": "left-candidate",
    "left_result_hash": "left",
    "left_task_id": "left",
    "model_id": "model",
    "network_enabled": "enabled",
    "observed_through": "through",
    "origin_task_id": "origin",
    "portfolio_session": "session",
    "position_basis": "basis",
    "preparation_binding_hash": "preparation",
    "preparation_plan_hash": "plan",
    "prepared_input_hash": "prepared-input",
    "prior_review_publication_hash": "prior-review",
    "research_input_id": "input",
    "research_input_plan_hash": "plan",
    "result_hash": "result",
    "review_dossier_hash": "dossier",
    "review_policy_hash": "policy",
    "review_publication_hash": "review",
    "review_read_at": "read-at",
    "review_schema_hash": "schema-hash",
    "right_candidate_id": "right-candidate",
    "right_result_hash": "right",
    "right_task_id": "right",
    "risk_report_hash": "risk-report",
    "risk_report_scope": "scope",
    "risk_task_id": "risk-study",
    "score_plan_hash": "plan",
    "score_snapshot_hash": "snapshot",
    "session_limit": "session-limit",
    "storage_plan_hash": "plan",
    "strategy_package_id": "package",
    "task_id": "task",
    "tasks_waiting": "queue",
    "update_plan_hash": "plan",
    "update_publication_hash": "update-publication",
    "update_task_id": "update",
    "upgrade_set_hash": "upgrade-set",
    "usage_reading_enabled": "enabled",
    "view_last_days": "days",
    "wake_read": "read",
    "wake_thread": "thread",
    "window_limit": "window-limit",
}
"""A field's flag where it is not the field's own name, hyphenated: a selector names its object
(`--task`, `--plan`, `--input`), one flag per field in every command."""


def flag_name(field: str) -> str:
    """A request field's flag, without its dashes: `file` for a document `--file` reads."""
    if field in PRIMARY:
        return "file"
    return FLAGS.get(field, field.replace("_", "-"))


def cli_name(operation: str) -> tuple[str, str]:
    """``(noun, verb)``: the operation's name in ``alphalattice <noun> <verb>``."""
    command = GRAMMAR[operation]
    return command.noun, command.verb


COMMANDS: Final[dict[tuple[str, str], tuple[str, ...]]] = {}
for _operation in sorted(OPERATIONS):
    COMMANDS.setdefault(cli_name(_operation), ())
    COMMANDS[cli_name(_operation)] += (_operation,)
"""``(noun, verb)`` to its operations: one, or two told apart by their required selectors."""


def fields(operation: str) -> tuple[frozenset[str], frozenset[str]]:
    """``(required, allowed)`` request fields: the request's own contract."""
    return PortfolioResearchOperationRequest.field_contract(operation)  # type: ignore[arg-type]


TABLE: Final = Path(__file__).with_name("operations.json")
"""The CLI's command table, generated by ``table_text()``."""


def _schema_options(
    spec: dict[str, Any], definitions: dict[str, Any], seen: frozenset[str] = frozenset()
) -> Iterator[dict[str, Any]]:
    """Visit a field's alternatives and local schema references without assuming their type."""
    yield spec
    reference = spec.get("$ref")
    if isinstance(reference, str) and reference not in seen:
        if not reference.startswith("#/$defs/"):
            raise ValueError("operations.schema_reference_not_local")
        name = reference.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
        yield from _schema_options(definitions[name], definitions, seen | {reference})
    for keyword in ("anyOf", "oneOf", "allOf"):
        for option in spec.get(keyword, []):
            yield from _schema_options(option, definitions, seen)


def _types(spec: dict[str, Any], definitions: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    for option in _schema_options(spec, definitions):
        if isinstance(option.get("type"), str):
            found.add(option["type"])
        elif "properties" in option:
            found.add("object")
    return sorted(found - {"null"})


def _reference(spec: dict[str, Any]) -> bool:
    """Whether the contract types a field as a hash or an id, or a list of them (V399), or
    names it an id the Host issues around one (`reference: issued`, V561)."""
    if spec.get("reference") == "issued":
        return True
    options = (spec, *spec.get("anyOf", []))
    for option in options:
        if not isinstance(option, dict):
            continue
        for typed in (option, option.get("items")):
            if isinstance(typed, dict) and (
                typed.get("format") == "uuid" or typed.get("pattern") == r"^[0-9a-f]{64}$"
            ):
                return True
    return False


def _first_sentence(text: str) -> str:
    """A field's meaning, its description's first sentence on one line."""
    line = " ".join(text.split())
    end = line.find(". ")
    return line if end < 0 else line[: end + 1]


def _limits(spec: dict[str, Any], definitions: dict[str, Any]) -> str:
    """A field's choices and range as its contract states them, in a few words (V401)."""
    found: list[str] = []
    for option in _schema_options(spec, definitions):
        if option.get("enum"):
            found.append("one of " + ", ".join(str(v) for v in option["enum"]))
        low = option.get("minimum", option.get("exclusiveMinimum"))
        high = option.get("maximum", option.get("exclusiveMaximum"))
        if low is not None or high is not None:
            above = "above" if "exclusiveMinimum" in option else "from"
            below = "below" if "exclusiveMaximum" in option else "to"
            found.append(
                " ".join(
                    part
                    for part in (
                        f"{above} {low}" if low is not None else "",
                        f"{below} {high}" if high is not None else "",
                    )
                    if part
                )
            )
    return "; ".join(dict.fromkeys(found))


def _inner_required(document: dict[str, Any]) -> dict[str, list[str]]:
    """Each request field typed by a model, and the fields that model requires (V373)."""
    models = document.get("$defs", {})
    found: dict[str, list[str]] = {}
    for name, spec in sorted(document["properties"].items()):
        for option in (spec, *spec.get("anyOf", [])):
            reference = option.get("$ref") if isinstance(option, dict) else None
            required = models.get(str(reference).rsplit("/", 1)[-1], {}).get("required")
            if reference and required:
                found[name] = sorted(required)
    return found


HELP_GROUPS: Final[tuple[tuple[str, tuple[tuple[str, str], ...]], ...]] = (
    (
        "Start and follow the work",
        (
            ("workspace", "The workspace: its inputs, recent studies and Tasks, what flows need."),
            ("session", "An agent session's binding to its workspace: bind, unbind, usage."),
            ("goal", "A goal an agent works for: open or take it, note it, submit it."),
            ("task", "A Task, the work a command started: show it, wait on it, cancel it."),
            ("activity", "The activity feed, and a wait on a Task or a goal."),
            ("decision", "Everything waiting on a person, each with its request."),
            ("recovery", "Unfinished Tasks: what stopped one and how it resumes."),
        ),
    ),
    (
        "Studies: Factor, then Alpha; Risk beside them",
        (
            ("study", "A Factor, Alpha or Risk study: controls, plan, run, show, compare."),
            ("curation", "A Factor study's curation: which factors go on."),
            ("foundation", "A curated Factor study sealed as an Alpha's foundation."),
            ("handoff", "The Alpha declaration a curated Factor study hands on."),
            ("feature", "Formula factors an agent declares and a person activates."),
            ("trial", "A formula factor tried against a completed study."),
            ("candidate", "Alpha candidates compared from their stored evidence."),
            ("model", "Alpha models an agent scaffolds, checks and sandboxes."),
            ("training", "A component's model-training inputs."),
        ),
    ),
    (
        "Portfolio",
        (
            ("book", "A Portfolio book: its draft from a candidate, its delivery report."),
            ("risk-link", "The Risk reports linked to a Portfolio book."),
            ("strategy", "A strategy prepared from promoted Alpha and Risk studies."),
            ("strategy-book", "An installed strategy's book: controls, preview, run."),
            ("result", "The saved results of installed strategies."),
            ("release", "A development result frozen as a release candidate."),
        ),
    ),
    (
        "Evidence and CRO",
        (
            ("evidence", "A book's evidence: its preview, sources, ledger and packets."),
            ("review", "A CRO review of a book: request, dossier, findings, answer."),
            ("bundle", "A specialist's prepared bundle and the answer to it."),
        ),
    ),
    (
        "Data and updates",
        (
            ("preparation", "The workspace's first-use preparation."),
            ("input", "Research inputs and their sealed versions."),
            ("data-update", "A data update: due source checks, Data and Feature upkeep."),
            ("issue", "Data issues: their evidence, options and decisions."),
            ("research-update", "A research update: data, scores, calibration and books."),
            ("score", "The scoring of a package's models for a session."),
            ("calibration", "The calibration of a package's Portfolio input."),
            ("portfolio-update", "A Portfolio update settling observed sessions."),
            ("automation", "The daily research update, on or off."),
            ("history", "Past studies, results, updates and reviews."),
        ),
    ),
    (
        "The workspace's care",
        (
            ("storage", "Storage clean-up, pins and rebuilds."),
            ("backup", "Backups of what the workspace cannot rebuild, and restores."),
            ("network", "Whether the workspace may reach the network."),
            ("usage-reading", "Whether the Host may read the bound agent Sessions' usage."),
            ("cpu-budget", "The CPU budget and how many Tasks run at once."),
            ("incident", "The supervisor's incidents and their remedies."),
            ("upgrade", "What an installed upgrade changed."),
        ),
    ),
    (
        "The CLI itself",
        (
            ("answer", "A saved full answer read locally, without a Host or fresh verification."),
            ("serve", "Start the Local Web Host on this workspace."),
            ("request", "Send one whole request, or a saved answer's next request."),
            ("schema", "A command's request schema, its answer's and a template."),
            ("operation", "Every operation with its fields."),
            ("event", "An event this client declares about its own work."),
        ),
    ),
)
"""Every object as the top-level help lists it, grouped in the research's order, each with what
it is: Git's common commands first, then the rest by kind (the user, 2026-10-01; the review's
F3). The gate holds that each object the CLI has is listed once."""

COMMON_PATH: Final[tuple[str, ...]] = (
    "workspace show",
    "goal take",
    "study controls",
    "study plan",
    "study run",
    "study show",
    "goal submit",
)
"""The common path the top-level help opens with: read the workspace, take the goal, declare,
plan, run and read a study, submit the goal."""

ALTERNATIVES: Final[dict[str, tuple[tuple[str, ...], ...]]] = {
    # The plan reads one declaration, its YAML text or its object, and refuses both or none
    # (`research_experiment.one_document_required`).
    "EXPERIMENT_PLAN": (("experiment_yaml",), ("experiment_document",)),
}
"""The requests whose owner takes exactly one of several field groups, which `schema show` states
as its `oneOf` (V409, the review's F4). The gate holds each group to the request's own fields."""

PERSON_ONLY: Final[frozenset[str]] = frozenset(
    {
        "NETWORK_ACCESS_SET",
        "USAGE_READING_SET",
        "STORAGE_CONFIRM",
        "STORAGE_PIN",
        "RESEARCH_INPUT_CONFIRM",
        "DATA_ISSUE_DELEGATE",
        "DATA_ISSUE_REVOKE",
        "MODEL_ACTIVATE",
        "MODEL_DEACTIVATE",
        "FEATURE_ACTIVATE",
        "FEATURE_DEACTIVATE",
        "RESEARCH_UPDATE_AUTOMATION_CONFIGURE",
        "STRATEGY_ACTIVATE",
        "STRATEGY_DEACTIVATE",
    }
)
"""The operations only a person completes, in the Workbench (V143).

Each owner refuses every other caller, a client's and an Agent's included: the network
authority, the storage confirmation and pin, the research input's confirmation, the data
issues' delegation and its revocation, the daily research
update's automation (V407: its owner refused every client while the CLI offered it), and a
research strategy's activation and deactivation (LS1, OW12). The CLI
lists them so that a person knows they exist, and marks each so that nobody else sends one
expecting it to run. A test holds this set to the owners' refusals.

Data change confirmation reads its plan kind: membership requires the person's approval,
while an exactly scoped full-history audit also admits the installed agent. It is not
unconditionally person-only (person-stops row 49).
"""

FIRST_USE_STEPS: Final[frozenset[str]] = frozenset(
    {
        "NETWORK_ACCESS_SET",
        "WORKSPACE_PREPARE_CONFIRM",
        "DATA_ISSUE_CONFIRM",
        "DATA_CHANGE_CONFIRM",
        "STRATEGY_ACTIVATE",
    }
)
"""The person's steps a first-use goal delegates to the agent that runs it (V452, OP19; STOPS-1):
opening the network for the first preparation, confirming that preparation and its resumes,
deciding its data issues, confirming its membership changes, and activating its book once that
book has a published review, which the person deactivates in one click. A deactivation, a model's
or Feature's activation, a storage decision, an automation, a revocation and anything paid stay a
person's."""


def table() -> dict[str, Any]:
    """Describe the CLI command and request-field contract.

    Returns:
        Each command's operation, its required and allowed fields, and each
        request field's JSON types from the request document schema.
    """
    document = PortfolioResearchRequestDocument.model_json_schema()
    properties = document["properties"]
    definitions = document.get("$defs", {})
    contracts = {operation: fields(operation) for operation in OPERATIONS}
    return {
        "schema": "alphalattice.local-application.operations",
        "commands": {f"{noun} {verb}": list(ops) for (noun, verb), ops in sorted(COMMANDS.items())},
        "purposes": {op: GRAMMAR[op].purpose for op in sorted(GRAMMAR)},
        "positional": {
            op: GRAMMAR[op].positional for op in sorted(GRAMMAR) if GRAMMAR[op].positional
        },
        "flags": {name: flag_name(name) for name in sorted(properties)},
        "primary": list(PRIMARY),
        # Each field's meaning and limits, from its contract (SC3), for the help (V401).
        "descriptions": {
            name: _first_sentence(spec.get("description", ""))
            for name, spec in sorted(properties.items())
            if spec.get("description")
        },
        "limits": {
            name: limits
            for name, spec in sorted(properties.items())
            if (limits := _limits(spec, definitions))
        },
        "fields": {
            operation: {"required": sorted(required), "allowed": sorted(allowed)}
            for operation, (required, allowed) in sorted(contracts.items())
        },
        "types": {name: _types(spec, definitions) for name, spec in sorted(properties.items())},
        # Each object field whose owner's model requires fields of its own (V373), so a next
        # request that fills such an object in part reads as a template naming what is left.
        "inner_required": _inner_required(document),
        # Each field the contract types as a hash or an id: the only ones where the client reads
        # a beginning the compact view shows back as the whole value (V393, V399).
        "references": sorted(name for name, spec in properties.items() if _reference(spec)),
        "person_only": sorted(PERSON_ONLY),
        "first_use": sorted(FIRST_USE_STEPS),
        # The operations that change nothing (the activity ledger's): each answer names what it
        # read and reads again from itself through `--from` (V449).
        "reads": sorted(READ_OPERATIONS),
        "collections": sorted(op for op, command in GRAMMAR.items() if command.collection),
        # Each request's exactly-one field groups, as its owner refuses otherwise (V409).
        "alternatives": {
            op: [list(group) for group in groups] for op, groups in sorted(ALTERNATIVES.items())
        },
        # The top-level help's common path and its objects in the research's order (V401).
        "help_groups": [[group, [list(pair) for pair in nouns]] for group, nouns in HELP_GROUPS],
        "common_path": list(COMMON_PATH),
    }


def table_text() -> str:
    """Serialize the CLI operation table as deterministic JSON text.

    Returns:
        The sorted, indented table followed by a newline.
    """
    return json.dumps(table(), indent=1, sort_keys=True) + "\n"


__all__ = [
    "COMMANDS",
    "FIRST_USE_STEPS",
    "OPERATIONS",
    "PERSON_ONLY",
    "TABLE",
    "cli_name",
    "fields",
    "table",
    "table_text",
]
