"""What a refusal says to the person who met it: what happened, what to do, what to ask next.

A refusal code is exact and stable; it is not an explanation. The Host's operations answered
many refusals with the code alone, or with raw exception text, so a person who met one after
an upgrade had to know the code base to act (binding plan, B9). ``explain`` gives each code
the user can meet on the study, report and recovery paths a sentence in plain words and the
lawful next requests, built from what the caller already knows. The code itself is unchanged,
so an Agent and a test read exactly what they read before.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from alphalattice.foundation.factor_research.experiments.authoring import (
    INSTALLED_REDUNDANCY_POLICIES,
    INSTALLED_SCREENING_POLICIES,
)
from alphalattice.interface.local_application.cli_contract import (
    NETWORK_ACCESS_REFUSALS,
    refusal_words,
)

_CHANGED = frozenset(
    {
        "research_experiment.execution_binding_changed",
        "alpha_research.lifecycle_execution_plan_changed",
        "risk_research.execution_plan_changed",
        "alpha_research.execution_plan_changed",
        "portfolio_research.program_changed",
        "portfolio_research.source_changed",
        "alpha_research.parent_evidence_changed",
        "research_experiment.input_authority_changed",
        "alpha_research.lifecycle_program_scheme_superseded",
        "risk_research.program_scheme_superseded",
        "portfolio_research.program_scheme_superseded",
        "research_lane.sample_scheme_superseded",
    }
)
_EARLIER_SCHEME = (
    "its Program was sealed when a Program also bound the code that runs it; the code is now "
    "bound by the study's plan, so the same declaration seals another Program"
)
_WHAT_CHANGED = {
    "research_experiment.execution_binding_changed": (
        "its kind's implementation or its research input changed since it ran"
    ),
    "portfolio_research.source_changed": "the Alpha study it builds on is not the one it sealed",
    "alpha_research.parent_evidence_changed": (
        "the Factor study it builds on is not the one it sealed"
    ),
    "research_experiment.input_authority_changed": (
        "its research input now resolves to other data than it sealed"
    ),
    "alpha_research.lifecycle_program_scheme_superseded": _EARLIER_SCHEME,
    "risk_research.program_scheme_superseded": _EARLIER_SCHEME,
    "portfolio_research.program_scheme_superseded": _EARLIER_SCHEME,
    "research_lane.sample_scheme_superseded": (
        "its names were sampled before a sample kept each Sector's share of the names; the "
        "same size now samples other names"
    ),
}


_KIND_WORDS = {
    "factor.screening-development": "Factor screening",
    "alpha.model-development": "Alpha model",
    "risk.covariance-development": "Risk covariance",
    "portfolio.policy-development": "Portfolio policy",
}
_MOVED_WORDS = {
    "implementation": "the {kind} method installed now is not the one it ran under",
    "research_input": "its research input no longer resolves to the data it sealed",
}


_PROVIDER_WAIT = frozenset(
    {
        "data.rate_limited",
        "data.provider_session_unstable",
        "workspace_preparation.retry_not_due",
        "workspace_data_update.retry_not_due",
    }
)
"""A provider that limited the requests or whose session failed: the owners defer the work, which
resumes after `retry_after_at` (V375). A timeout is not among them (V396)."""
_PROVIDER_UNREADABLE = frozenset(
    {
        "data.provider_fetch_failed",
        "data.current_universe_no_feature_ready_listings",
        "data.listing_updates_incomplete",
    }
)
"""Listings whose provider answer could not be read: a changed interface or a symbol it no
longer serves (V375)."""


_UNREAD_KEYS = {
    "research_authoring.section_key_unknown": (
        "",
        "A method's parameter is written inside `parameters`, and `schema show study plan` "
        "prints each section's keys under `declaration_sections`.",
    ),
}
"""A key a Desk's section contract does not name (V134, V249): where it was written, and where
each section's keys are printed."""

_RISK_LINK_WORDS = {
    "portfolio_research.risk_study_required": (
        "An inverse-volatility rule (`iv1`, `iv2`) weighs each name by a Risk study's per-name "
        "volatility, and a `portfolio.policy` decides on its covariance: name a completed Risk "
        "study on this research input in `portfolio.risk_task_id`, or choose `ew` and no policy."
    ),
    "portfolio_research.risk_study_unread": (
        "Equal weight (`ew`) reads no Risk study: remove `portfolio.risk_task_id`, or choose "
        "`iv1`, `iv2` or a `portfolio.policy`."
    ),
    "portfolio_research.risk_study_invalid": (
        "`portfolio.risk_task_id` names no Risk study in this workspace."
    ),
    "portfolio_research.risk_study_not_completed": (
        "The Risk study has not published its result; run it to completion first."
    ),
    "portfolio_research.risk_study_scoped": (
        "The Risk study's surface is split by scope; link one over the whole universe."
    ),
    "portfolio_research.risk_input_mismatch": (
        "The Risk study read another research input, Panel or universe than the Alpha scores; "
        "run it on the same input."
    ),
    "portfolio_research.risk_session_axis_incomplete": (
        "The Risk study does not cover every formation session the Portfolio study weighs: "
        "those are the Alpha study's scored formations, whose first and last `study show` of "
        "the Alpha study names. Run a Risk study over exactly them, not over the Alpha "
        "study's whole data window, whose first sessions lack the Risk estimator's lookback."
    ),
    "portfolio_research.risk_listing_axis_incomplete": (
        "The Risk study does not cover every listing the Portfolio study weighs."
    ),
}
"""A Portfolio study's linked Risk study (V310): what is wrong with it and what may be named."""


_PORTFOLIO_POLICY_WORDS = {
    "portfolio_research.policy_book_conflict": (
        "A `portfolio.policy` runs in place of the tranche book: leave `top_k`, `tranches`, "
        "`exit_rank` and `weight_rule` unset, and set the policy's own `top_k`."
    ),
    "portfolio_research.top_k_exceeds_universe": (
        "The policy's `top_k` is larger than the listings the Alpha study scored."
    ),
    "portfolio_research.policy_cap_infeasible": (
        "The policy cannot invest fully: `top_k` times `maximum_weight` must be at least 1."
    ),
}
"""A Portfolio study's catalog policy (V310): what is wrong with it."""


_NOT_RUNNING_FORWARD = frozenset(
    {
        "strategy_score.input_preparation_not_installed",
        "portfolio_calibration.not_installed",
        "portfolio_update.not_installed",
    }
)
"""The refusals of a strategy that does not run forward: its activation binds what they ask for."""


_STRATEGY_ACTIVATION_WORDS = {
    "strategy_activation.human_confirmation_required": (
        "Running a strategy forward or stopping it is the person's decision: introduce the "
        "strategy from its book's controls, ask the person in one line and, on a clear yes, "
        "send this again with --person-said and --asked."
    ),
    "strategy_activation.research_strategy_required": (
        "Only an installed research strategy runs forward: prepare one from your Alpha and Risk "
        "studies and install it, then run and review its book."
    ),
    "strategy_activation.book_task_absent": "No Task has this id; `study list` names the books.",
    "strategy_activation.book_task_required": (
        "A strategy runs forward from one of its books: name a completed run of an installed "
        "strategy's book."
    ),
    "strategy_activation.completed_book_required": (
        "This book has not completed: wait for its Task, review the book, then activate it."
    ),
    "strategy_activation.research_book_required": (
        "This book is not an installed research strategy's; only those books run forward."
    ),
    "strategy_activation.book_package_moved": (
        "This book ran under a strategy package that has changed since: run the book again and "
        "activate the new run."
    ),
    "strategy_activation.book_result_absent": (
        "This book's result is not in the workspace's ledger: run the book again."
    ),
    "strategy_activation.book_state_unsealed": (
        "This book's run did not seal its last state, so nothing can continue it: run the book "
        "again."
    ),
    "strategy_activation.book_not_whole": (
        "A strategy runs forward from its book's last formation, decided from its first: run "
        "the book over its whole support ({subject}) and activate that run."
    ),
    "strategy_activation.book_shorter_than_its_activation": (
        "This book has fewer formations than its calibration reads before it sizes positions, "
        "so it cannot run forward."
    ),
    "strategy_activation.schedule_unavailable": (
        "The installed market calendar plans no session after the book's last formation."
    ),
    "strategy_activation.study_unreadable": (
        "The Alpha study behind the strategy's {subject} component does not read back; "
        "`study show` verifies it."
    ),
    "strategy_activation.training_admission_absent": (
        "The training input the {subject} component's Alpha study read is no longer in this "
        "workspace: prepare and install the strategy again from current studies."
    ),
    "strategy_activation.training_admission_mismatch": (
        "The {subject} component's training input is not the one its study read, or ends "
        "before the book's next formation: prepare and install the strategy again."
    ),
    "strategy_activation.models_unavailable": (
        "The {subject} component's models do not reach the book's next formation: prepare and "
        "install the strategy again from current studies."
    ),
    "strategy_activation.already_active": (
        "This strategy already runs forward, and its daily update continues it. To start it "
        "from another book, stop it first."
    ),
    "strategy_activation.configuration_changed": (
        "The workspace's configuration changed while the strategy was being activated: "
        "activate it again."
    ),
    "strategy_activation.not_active": (
        "This strategy does not run forward; there is nothing to stop."
    ),
    "portfolio_update.checkpoint_activation_invalid": (
        "A book's opening state names a person's activation only with its time and its book: "
        "activate the book again."
    ),
}
"""A person's activation of a research strategy's book (LS1): what refused, and the way on."""


_FACTOR_POLICIES = {
    "factor_research.authoring_screening_policy_not_installed": (
        "screening_policy",
        INSTALLED_SCREENING_POLICIES,
    ),
    "factor_research.authoring_redundancy_policy_not_installed": (
        "redundancy_policy",
        INSTALLED_REDUNDANCY_POLICIES,
    ),
}
"""A Factor policy field and what is installed for it (V110)."""


def kind_words(kind: str) -> str:
    """A study kind as a person names it: `alpha.model-development` is an Alpha model."""
    return _KIND_WORDS.get(kind) or kind.partition(".")[0].capitalize()


def explain(
    code: str,
    *,
    task_id: str | None = None,
    kind: str | None = None,
    refresh_calls: object = None,
    lifecycle: str | None = None,
    result_hash: str | None = None,
    moved: list[dict[str, str | None]] | None = None,
    book: Mapping[str, str] | None = None,
    origin: str | None = None,
    portfolio: Mapping[str, str | None] | None = None,
    trial: Mapping[str, Any] | None = None,
    installed_packages: tuple[str, ...] | None = None,
    workspace: Path | None = None,
) -> dict[str, Any]:
    """The words and next requests for one refusal code; empty for a code without them.

    `moved` names which recorded identities differ from what is installed; the words say
    which, and the answer carries them for an Agent. `book` is the book a review request
    named, as its request fields. `origin` is the study a continuation's PLAN named.
    `portfolio` is the Alpha study and candidate a Portfolio declaration named.
    `trial` is a refused trial's way on: the studies it can run against, or the Factor studies
    whose handoff opens one (`FeatureTrials.way_on`). `installed_packages` names the book's
    admitted choices when its selector is missing or unavailable.
    """
    task = {"task_id": task_id} if task_id else {}
    base, _sep, subject = code.partition(":")
    if base == "workspace_data_update.panel_binding_mismatch":
        return {
            **refusal_words(code, workspace=workspace),
            "next_requests": {"show": {"operation": "DATA_UPDATE_READBACK"}},
        }
    if base in {
        "risk_report.completed_portfolio_required",
        "risk_research.covariance_chunk_identity_invalid",
    }:
        return {
            **refusal_words(code),
            **({"selected_task_kind": kind} if kind else {}),
            "next_requests": {"studies": {"operation": "EXPERIMENTS"}},
        }
    if base == "evidence_review.replan_book_selection_required":
        return {
            **refusal_words(code),
            "next_requests": {"books": {"operation": "RESEARCH_HISTORY"}},
        }
    if base in {
        "strategy_book.strategy_package_required",
        "local_application.strategy_package_not_installed",
    }:
        packages = tuple(sorted(installed_packages or ()))
        selection = (
            "The book document did not select a strategy: give `strategy_package_id`."
            if base == "strategy_book.strategy_package_required"
            else f"The selected strategy package `{subject}` is not installed in this workspace."
        )
        return {
            "detail": selection
            + (
                " Installed packages: " + ", ".join(packages) + ". "
                "Read a package's controls and save its flat declaration, then preview that file."
                if packages
                else " Read the installed strategy controls before selecting a package."
            ),
            "next_action": "SELECT_STRATEGY_PACKAGE_FROM_CONTROLS",
            "next_requests": {
                f"controls:{package}": {"operation": "CONTROLS", "strategy_package_id": package}
                for package in packages
            }
            or {"strategies": {"operation": "RESEARCH_STRATEGY_CONTROLS"}},
        }
    if base == "research_update.input_source_access_not_admitted":
        from alphalattice.control.product_host.maintenance.data_update import NETWORK_WORK_WORDS

        target, _sep, kinds = subject.partition(",")
        names = kinds.split(",")
        words = refusal_words(code, workspace=workspace)
        if kinds and all(name in NETWORK_WORK_WORDS for name in names):
            needs = "; ".join(NETWORK_WORK_WORDS[name].format(session=target) for name in names)
            words = refusal_words(f"{base}:{needs}", workspace=workspace)
        return {**words, "next_requests": {"network": {"operation": "NETWORK_ACCESS"}}}
    if base in NETWORK_ACCESS_REFUSALS:
        return {
            **refusal_words(code, workspace=workspace),
            "next_requests": {"network": {"operation": "NETWORK_ACCESS"}},
        }
    if base in _PROVIDER_WAIT:
        return {
            "detail": (
                "The market data provider limited the requests or did not answer, which is the "
                "provider's state, not the workspace's. Every listing already fetched is kept "
                "and the work resumes from them: send the same plan again once "
                "`retry_after_at` has passed."
            ),
        }
    if base == "data.provider_timeout":
        # The owners record a timed-out listing as failed and go on; a timed-out universe source
        # stops the step; neither sets a retry time (V396).
        return {
            "detail": (
                "The market data provider did not answer in time, after its retries. A listing "
                "that timed out is recorded as failed in this run while the others go on; if the "
                "universe source itself timed out, the step stopped. No retry time is set: send "
                "the same plan again once the provider answers."
            ),
        }
    if base == "alternative_evidence.unit_not_prepared":
        # A Task that stops on a unit reads these words whole in its recovery view, so they fit
        # its bound: the coverage's pointer only where it fits after the unit's own (V565).
        unit = "This coverage unit was not prepared. " + unit_failure_words(subject)["detail"]
        pointer = " The book's coverage (`next_requests.coverage`) lists each unit's state."
        return {"detail": unit + pointer if len(unit + pointer) <= STOP_WORDS_BOUND else unit}
    if base == "alternative_evidence.preparation_not_complete":
        return {
            "detail": (
                "This preparation, or this unit of it, has not finished: wait for its Task "
                "(`activity wait`), then read the book's coverage (`next_requests.coverage`), "
                "which offers each prepared unit's packet and bundle."
            ),
        }
    if base == "alternative_evidence.coverage_unit_unknown":
        return {
            "detail": (
                "The preparation has no such unit: the book's coverage (`next_requests.coverage`) "
                "names its units and their states."
            ),
        }
    if base == "goal.closed_open_a_follow_up":
        return {
            "detail": (
                "This goal is closed, complete or abandoned, and its record stays as it was "
                "sealed; nothing was changed. Its show request reads its standing. Work that "
                "goes on is a follow-up goal whose declaration names this one in "
                "`parent_goal_id` (`goal schema --save-declaration` writes one to edit)."
            ),
            "next_action": "READ_THE_GOAL_OR_OPEN_A_FOLLOW_UP",
            "next_requests": {"schema": {"operation": "GOAL_SCHEMA"}},
        }
    if base == "goal.goal_id_required":
        return {
            "detail": (
                "No goal is bound to this session, and the request names none: name it "
                "(`goal show <id>`), or open one (`goal open`) or take one (`goal take`) first."
            ),
        }
    if base == "goal.revision_conflict_read_latest":
        # V527: the refusal kept nothing to go on with, and the session's own goal may be another.
        return {
            "detail": (
                "This goal changed after the revision this request started "
                "from, so nothing was changed. Read its latest revision by the goal's id "
                "(`next_requests.show`), then make the change on that revision."
            ),
        }
    if base == "goal.reference_id_already_used":
        return {
            "detail": (
                "A reference id names one result in a goal. The goal already holds the ids in "
                "`expected.attached_reference_ids`: cite them by id in `criteria`, `deliverables` "
                "and `findings`, and declare only new ones in `references` or `files`, each once. "
                "`fields` names each entry that reused an id. `next_requests.submit`, when "
                "present, is this submission without the entries that only declared again a "
                "held reference, its citations kept; another result needs an id of its own."
            ),
        }
    if base == "feature_research.input_binding_unresolved":
        return {
            "detail": (
                "The `input_binding_hash` names no research input this workspace holds, so "
                "nothing was planned or built. `expected` lists the bindings it holds; each "
                "`next_requests.controls:<binding>` opens that input's feature controls, whose "
                "template carries its binding exactly."
            ),
        }
    if base in _PROVIDER_UNREADABLE:
        return {
            "detail": (
                "The provider's answer could not be read for some listings: a changed provider "
                "interface, or a symbol it no longer serves. Every listing already fetched is "
                "kept, and `issue list` names each failed one with the decisions a person may "
                "take. When every listing fails so, the provider's interface changed, and an "
                "updated product is needed."
            ),
        }
    if base == "research_lane.exploration_not_promoted":
        return {
            "detail": (
                "A strategy is prepared only from studies on the whole universe, and one of "
                "its parents ran on a sample of the input's names. Promote that study; its "
                "promotion runs the same declaration on every name."
            ),
            "next_requests": {"promote": {"operation": "EXPERIMENT_PROMOTE", "task_id": subject}}
            if subject
            else {},
        }
    if base in _NOT_RUNNING_FORWARD:
        # The score, calibration or Portfolio update of a strategy not activated, or
        # one stopped since; the daily update's plan meets them first (OP4, U73).
        return {
            "detail": (
                "This strategy is not active. Read this package's controls for its "
                "activation offer: under the first use's delegation you activate it; otherwise "
                "introduce the strategy and ask the person in one line. Once active, "
                "continue with this package's Forward update. Installation, activation and "
                "the automatic schedule are separate facts; activation does not enable the "
                "automatic schedule."
            ),
            "next_requests": {"books": {"operation": "CONTROLS"}},
        }
    if base in _STRATEGY_ACTIVATION_WORDS:
        return {
            "detail": _STRATEGY_ACTIVATION_WORDS[base].format(
                subject=subject.replace("..", " to ") or "its components"
            ),
            "next_requests": {"books": {"operation": "CONTROLS"}},
        }
    if base == "alpha_research.saved_comparison_score_support_mismatch":
        return {
            "detail": (
                "The two Alpha studies scored different sessions, names or available rows, so "
                "their results are not compared: a difference between them would mix what the "
                "feature adds with what each scored. Each study's saved metrics stand on their "
                "own."
            ),
            "next_requests": {},
        }
    if base == "model_extension.sandbox_required":
        return {
            "detail": (
                "A person activates a model after a sandbox trial of this identity passed: "
                "one Alpha study on a copy of the workspace, read back, and U0 on the copy. "
                f"`model sandbox {subject}` runs one, with the Host stopped."
            ),
            "next_requests": {"models": {"operation": "MODEL_EXTENSIONS"}},
        }
    if base == "model_extension.contract_failed":
        return {
            "detail": (
                f"The model fails its contract ({subject}); `model check` names every check."
            ),
            "next_requests": {"models": {"operation": "MODEL_EXTENSIONS"}},
        }
    if base == "model_extension.human_confirmation_required":
        return {
            "detail": (
                "Activating or deactivating a model is the person's decision: ask the person "
                "in one line and, on a clear yes, send this again with --person-said and "
                "--asked. An agent declares, checks and sandboxes it, and reads its review packet."
            ),
            "next_requests": {"models": {"operation": "MODEL_EXTENSIONS"}},
        }
    if base == "research_update.human_confirmation_required":
        return {
            "detail": (
                "Turning the daily research update on or off is the person's decision: ask the "
                "person in one line and, on a clear yes, send this again with --person-said and "
                "--asked; `automation show` reads its state."
            ),
            "next_requests": {"automation": {"operation": "RESEARCH_UPDATE_AUTOMATION_READBACK"}},
        }
    if base == "research_update.automation_package_not_installed":
        # The daily update accepts only the strategies that run forward (U73).
        return {
            "detail": (
                "Only a strategy that runs forward updates daily, after one of its reviewed "
                "books is activated. `automation show` lists the strategies that run forward."
            ),
            "next_requests": {"automation": {"operation": "RESEARCH_UPDATE_AUTOMATION_READBACK"}},
        }
    if base == "feature_extension.human_confirmation_required":
        return {
            "detail": (
                "Activating or deactivating a formula factor is the person's decision: ask the "
                "person in one line and, on a clear yes, send this again with --person-said and "
                "--asked. An agent declares, builds and tries it, and reads its review packet."
            ),
        }
    if base == "feature_extension.trial_required":
        return {
            "detail": (
                f"A person activates {subject} after one of its trials completed: `trial run` "
                "with its research plan and a finished study."
            ),
        }
    if base == "feature_extension.preprocessing_not_admitted_for_active_panel":
        return {
            "detail": (
                f"The daily Panel preprocesses with ROBUST_SECTOR_NEUTRAL_Z alone; {subject} is "
                "a research recipe. Declare the factor again with that recipe to activate it."
            ),
        }
    if base == "portfolio_calibration.intervening_score_absent":
        count, _sep, span = subject.partition(",")
        first, _sep, last = span.partition("..")
        return {
            "detail": (
                "A calibration reads a published score for every session after its seed "
                "through the score's formation, and "
                + (f"{count} have none ({first} to {last})" if first else "some have none")
                + ". Score them first, the earliest first, then plan the calibration again."
            ),
            "next_requests": {
                "score": {
                    "operation": "STRATEGY_SCORE_PLAN",
                    "strategy_package_id": None,
                    "formation_session": first,
                }
            }
            if first
            else {},
        }
    if base == "strategy_score.component_required":
        return {
            "detail": (
                "This package scores its components one at a time; name one with "
                f"--component ({subject.replace(',', ', ')})."
            ),
            "next_requests": {},
        }
    if base == "strategy_score.formation_or_model_epoch_unavailable":
        said = dict(part.split("=", 1) for part in subject.split(",") if "=" in part)
        epoch = said.get("epoch", "").replace("..", " to ")
        return {
            "detail": (
                f"No score can be planned on {said.get('formation', 'that session')}: the model "
                "lifecycle admits "
                + (f"formations {epoch}" if epoch else "another range")
                + ". Plan a formation inside it with --session, or renew the "
                "lifecycle past it."
            ),
            "next_requests": {},
        }
    if base == "feature_trial.feature_input_not_the_studys":
        return {
            "detail": (
                "The feature was planned on another research input than the study it is tried "
                "against; plan it on the study's input."
            ),
            "next_requests": {},
        }
    if base == "feature_trial.study_not_from_factor_evidence":
        return _trial_way_on(trial)
    if base == "feature_trial.not_found":
        return {
            "detail": "No feature trial has this ID in this workspace.",
            "next_requests": {"trials": {"operation": "FEATURE_TRIALS"}},
        }
    if base == "feature_trial.record_damaged":
        return {
            "detail": (
                "This trial's record in the workspace no longer reads: it was changed or cut "
                "short. Start the trial again with its feature plan and study; the damaged record "
                "is kept aside and every step that did not change is reused."
            ),
            "next_requests": {"trials": {"operation": "FEATURE_TRIALS"}},
        }
    if base == "research_lane.sample_size_invalid":
        return {
            "detail": (
                "An exploration sample needs at least as many names as an Alpha study ranks "
                "in a session (100) and fewer than the input holds; the declared size is "
                "outside that range."
            ),
            "next_requests": {},
        }
    if base == "research_experiment.task_kind_mismatch":
        # A study's read named another owner's Task, an installed strategy's book among them:
        # its kind, and the read that takes any Task (V490).
        return {
            "detail": "This Task is not a study, so the study reads do not take it; `task show` "
            "reads any Task, with the requests its owner offers.",
            "next_action": "READ_THE_TASK_WITH_TASK_SHOW",
            **({"task_kind": kind} if kind else {}),
            **({"next_requests": {"task": {"operation": "STATUS", **task}}} if task else {}),
        }
    if base == "research_experiment.standalone_kind_not_installed":
        return {
            "detail": (
                "`study controls` writes a new Factor, Alpha or Risk study's declaration "
                "(`factor.screening-development`, `alpha.model-development`, "
                "`risk.covariance-development`). A Portfolio study starts from an Alpha "
                "development study: its Portfolio draft (`EXPERIMENT_PORTFOLIO_DRAFT` with that "
                "study's Task) writes the declaration PLAN takes."
            ),
            "fields": [["experiment_kind"]],
            "next_requests": {
                "portfolio_draft": {
                    "operation": "EXPERIMENT_PORTFOLIO_DRAFT",
                    "task_id": None,
                    "candidate_id": None,
                }
            },
        }
    if base == "goal.continuation_requires_parent_goal":
        return {
            "detail": (
                "`research.purpose: CONTINUATION` continues a Goal: name it in `parent_goal_id`, "
                "or open this Goal with `NEW_RESEARCH` or `EXISTING_RESULTS`."
            ),
            "fields": [["research", "purpose"], ["parent_goal_id"]],
            "next_requests": {"goals": {"operation": "GOAL_LIST"}},
        }
    if base in _RISK_LINK_WORDS or base in _PORTFOLIO_POLICY_WORDS:
        # A Portfolio declaration is rewritten from its Alpha study's Portfolio draft,
        # never from `study controls`, which writes no Portfolio study (V322).
        source = portfolio or {}
        return {
            "detail": _RISK_LINK_WORDS.get(base) or _PORTFOLIO_POLICY_WORDS[base],
            "fields": [["portfolio", "risk_task_id" if base in _RISK_LINK_WORDS else "policy"]],
            "next_requests": {
                "portfolio_draft": {
                    "operation": "EXPERIMENT_PORTFOLIO_DRAFT",
                    "task_id": source.get("task_id"),
                    "candidate_id": source.get("candidate_id"),
                },
                # The Alpha study names the formations a Risk study must cover (V501).
                **(
                    {"alpha": {"operation": "EXPERIMENT_READBACK", "task_id": source["task_id"]}}
                    if base == "portfolio_research.risk_session_axis_incomplete"
                    and source.get("task_id")
                    else {}
                ),
            },
        }
    if base == "portfolio_research.benchmark_support_absent":
        # Two causes under one code, its subject naming which and where (V504, V511, RR5): a
        # session with no eligible name, or an eligible name without its realized return.
        source = portfolio or {}
        cause, _sep, facts = subject.partition(":")
        if cause == "no_eligible_name":
            return {
                "detail": (
                    f"The book's walk met formation sessions with no eligible name ({facts}): no "
                    "listing was both tradable and a member of the universe there, so the book "
                    "holds nothing to value against its benchmark, under either unavailable-"
                    "return policy. A book's sessions are its Alpha study's support: draft that "
                    "study again with a window that starts after those sessions and run it, "
                    "then draft the book from it, or choose another research input."
                ),
                "next_requests": {
                    "alpha_draft": {
                        "operation": "EXPERIMENT_DRAFT",
                        "task_id": source.get("task_id"),
                    }
                },
            }
        found = (
            f"The book's walk met eligible names without their realized return ({facts}), "
            "which `require_complete` refuses. "
            if cause == "returns_unavailable" and facts
            else "The book's walk met a formation session with no eligible name, or an eligible "
            "name without its realized return there, which `require_complete` refuses. "
        )
        return {
            "detail": found
            + (
                "The draft's declared `unavailable_return_policy: quarantine_listings` excludes "
                "those names, keeps the original inputs and reports the effective population; "
                "it needs the person's authorization, which a first-use goal's delegation "
                "gives. Plan the draft again with it, or choose another research input."
            ),
            "fields": [["portfolio", "unavailable_return_policy"]],
            "next_requests": {
                "portfolio_draft": {
                    "operation": "EXPERIMENT_PORTFOLIO_DRAFT",
                    "task_id": source.get("task_id"),
                    "candidate_id": source.get("candidate_id"),
                }
            },
        }
    if base == "portfolio_research.authority_field_mismatch":
        # A book's experiment section is its draft's: the Alpha study's support (V502).
        source = portfolio or {}
        return {
            "detail": (
                "A book's `experiment` section is bound to its draft: its input, sessions and "
                "window are the Alpha study's support, and a changed field is refused. Keep "
                "the draft's `experiment` section as written and change the `portfolio` "
                "section; a different window needs an Alpha study over that window."
            ),
            "fields": [["experiment"]],
            "next_requests": {
                "portfolio_draft": {
                    "operation": "EXPERIMENT_PORTFOLIO_DRAFT",
                    "task_id": source.get("task_id"),
                    "candidate_id": source.get("candidate_id"),
                }
            },
        }
    if base in _FACTOR_POLICIES:
        field, installed = _FACTOR_POLICIES[base]
        way = (
            "The study's draft writes its declaration with the study's own policies; change "
            "what the continuation changes there and plan that."
            if origin
            else "`study controls` writes a new study's declaration with them, and "
            "`study draft` a continuation's."
        )
        return {
            "detail": (
                f"`factor.{field}` names no installed policy: it is missing or misspelled. "
                f"Installed: {', '.join(f'`{name}`' for name in installed)}. {way}"
            ),
            "fields": [["factor", field]],
            "next_requests": {"draft": {"operation": "EXPERIMENT_DRAFT", "task_id": origin}}
            if origin
            else {"controls": {"operation": "EXPERIMENT_CONTROLS"}},
        }
    if base in _UNREAD_KEYS:
        place, holds = _UNREAD_KEYS[base]
        return {
            "detail": (
                f"The declaration writes `{place}{subject}`, which nothing in the study reads, "
                f"so it is refused rather than kept unread in the plan's identity. {holds} "
                "`study controls` shows what is installed."
            ),
            "next_requests": {},
        }
    if base == "research_lane.sample_not_supported":
        return {
            "detail": (
                "A Factor study, an Alpha study and the Portfolio study built on it run on a "
                "sample of names. A Risk study estimates on its own axis; declare the whole "
                "universe for this kind."
            ),
            "next_requests": {},
        }
    if code in _CHANGED:
        named = kind_words(kind) if kind else None
        what = (
            " and ".join(
                _MOVED_WORDS[str(item["identity"])].format(kind=named or "study's")
                for item in moved
            )
            if moved
            else _WHAT_CHANGED.get(
                code, "what its plan compiles to under the installed code is not what it sealed"
            )
        )
        study = f"This {named} study" if named else "This study"
        refresh = (
            f" A continuation declares {refresh_calls} numerical "
            f"call{'' if refresh_calls == 1 else 's'}."
            if isinstance(refresh_calls, int)
            else ""
        )
        return {
            **({"moved": moved} if moved else {}),
            "detail": (
                f"{study} cannot be replayed as it ran: {what}. It still reads back exactly as "
                "recorded. A continuation plans the same declaration under what is installed "
                "now; its plan says what it reuses and what it computes before anything runs."
                + refresh
            ),
            "next_requests": {
                **(
                    {
                        "readback": {"operation": "EXPERIMENT_READBACK", **task},
                        "continue": {"operation": "EXPERIMENT_DRAFT", **task},
                    }
                    if task
                    else {}
                ),
                "overview": {"operation": "UPGRADE_OVERVIEW"},
            },
        }
    if code == "product_host.evidence_review_evidence_not_current":
        return {
            "detail": (
                "An analysis this review would read is not current: it expired, or it was sealed "
                "under an earlier version of the Evidence method. It still reads back as "
                "recorded. The book's Evidence & CRO state names each unit's analysis and what "
                "prepares current evidence; ask again once it is prepared."
            ),
            "next_requests": {"evidence": {"operation": "EVIDENCE_CRO", **(book or {})}},
        }
    if code == "local_application.network_access_human_only":
        return {
            "detail": (
                "Network access is the person's decision outside a first use: ask the person "
                "in one line and, on a clear yes, send this again with --person-said and "
                "--asked; `network show` reads what decides it."
            ),
            "next_requests": {"network": {"operation": "NETWORK_ACCESS"}},
        }
    if code == "task_control.task_not_found":
        return {
            "detail": (
                "No Task with this ID is recorded in this workspace; it may belong to another "
                "workspace, or the ID was mistyped."
            ),
            "next_requests": {
                "tasks": {"operation": "TASKS"},
                "history": {"operation": "RESEARCH_HISTORY"},
            },
        }
    if code == "research_experiment.summary_kind_not_installed":
        named = kind_words(kind) if kind else "different"
        return {
            "detail": (
                f"SUMMARY reads Alpha studies; this is a {named} study, and its readback "
                "carries its results."
            ),
            "next_requests": {"readback": {"operation": "EXPERIMENT_READBACK", **task}}
            if task
            else {},
        }
    if code == "research_workspace.strategy_not_installed":
        return {
            "detail": (
                "This workspace has no installed strategy package, so Portfolio runs, results "
                "and reports are not available here. Research studies do not need one."
            ),
            "next_action": "PREPARE_AND_INSTALL_A_RESEARCH_STRATEGY",
            "next_requests": {
                "strategies": {"operation": "RESEARCH_STRATEGY_CONTROLS"},
                "experiments": {"operation": "EXPERIMENTS"},
            },
        }
    if code == "portfolio_research.result_not_found":
        return {
            "detail": "No Portfolio result with this hash is recorded in this workspace.",
            "next_requests": {"results": {"operation": "RESULTS"}},
        }
    if base == "content_store.artifact_missing":
        return {
            "detail": (
                f"The requested artifact {subject or result_hash or 'named by this request'} "
                "is not kept in this workspace. Read workspace show to find a kept input or "
                "result. If a completed record names this missing file, restore a backup "
                "that holds it into a new folder and work there."
            ),
            "next_requests": {"workspace": {"operation": "WORKSPACE_SHOW"}},
        }
    if code == "content_store.artifact_tampered":
        return {
            "detail": (
                "The stored result's bytes no longer match the hash they were sealed under, so "
                "it is not shown; the workspace copy was changed after it was written."
            ),
            "next_requests": {"results": {"operation": "RESULTS"}},
        }
    if code == "portfolio_application.session_outside_report":
        return {
            "detail": "This session is outside the report's window; the report reads without it.",
            "next_requests": {"report": {"operation": "REPORT", "result_hash": result_hash}}
            if result_hash
            else {},
        }
    if code == "local_application.task_recovery_not_configured":
        return {
            "detail": (
                "This service does not resume Tasks; the Task's own record says why it stopped."
            ),
            "next_requests": {"recovery": {"operation": "TASK_RECOVERY", **task}} if task else {},
        }
    if code == "alpha_modeling.lightgbm_thread_canary_mismatch":
        return {
            "detail": (
                "LightGBM on this machine grew other trees at the CPU budget's thread count "
                "than the sealed canary, so the study did not fit. Set the workspace's CPU "
                "budget to 1 (`cpu-budget set --cores 1`) and run it again: on one thread it "
                "fits as the method was sealed."
            ),
            "next_requests": {"status": {"operation": "STATUS", **task}} if task else {},
        }
    if code == "task_not_succeeded":
        return {
            "detail": (
                f"This study's Task is {lifecycle}; its results read back once it has "
                "succeeded, and its recovery view says what stopped it and what may resume it."
            ),
            "next_requests": {
                "status": {"operation": "STATUS", **task},
                "recovery": {"operation": "TASK_RECOVERY", **task},
            }
            if task
            else {},
        }
    if base == "factor_research.handoff_authority_field_mismatch":
        return {**refusal_words(code), "next_requests": {}}
    if base in _PLAN_FIELD_REFUSALS:
        return {
            "detail": _PLAN_FIELD_REFUSALS[base](subject),
            "next_action": "EDIT_DECLARATION_AND_PLAN",
            "next_requests": {},
        }
    if base == "product_host.evidence_review_experiment_selector_invalid" and subject:
        # A study's parts beside another book (V546), as a position basis is (V290).
        parts = subject.split(",")
        one = len(parts) == 1
        return {
            "detail": (
                f"{', '.join(f'`{part}`' for part in parts)} {'names' if one else 'name'} a "
                "study's book, with its Task, its receipt and the session together; beside "
                f"another book {'it does' if one else 'they do'} not apply. A published result, a "
                "public development replay (an installed strategy's historical replay) among "
                "them, is named by `result_hash` alone and a handoff by `handoff_hash`: the same "
                f"request without {'it' if one else 'them'} is offered."
            ),
            "next_action": "SEND_THE_OFFERED_REQUEST",
            "next_requests": {"without_" + "_and_".join(parts): dict(book)} if book else {},
        }
    if base == "product_host.evidence_review_experiment_selector_invalid":
        return {
            "detail": (
                "A study's book is named by three parts together -- its Task "
                "(`experiment_task_id`), its receipt (`experiment_receipt_hash`) and the session "
                "(`portfolio_session`), as the study's readback offers them -- and one is "
                "missing. A published result, a public development replay among them, is named "
                "by `result_hash` alone."
            ),
            "next_action": "NAME_THE_STUDY_TASK_RECEIPT_AND_SESSION",
            "next_requests": {"history": {"operation": "RESEARCH_HISTORY"}},
        }
    if base == "product_host.evidence_review_update_selector_invalid" and not subject:
        return {
            "detail": (
                "An update's book is named by three parts together -- `update_task_id`, "
                "`update_publication_hash` and `position_basis` (`CONDITIONAL_ESTIMATE` or "
                "`OBSERVED_RESEARCH_ENTRY`), as the update's readback offers them -- and one is "
                "missing or the basis is neither."
            ),
            "next_action": "NAME_THE_UPDATE_TASK_PUBLICATION_AND_BASIS",
            "next_requests": {"history": {"operation": "RESEARCH_HISTORY"}},
        }
    if base == "product_host.evidence_review_selector_ambiguous":
        return {
            "detail": (
                "The request names more than one book, and a review reads one: name a result "
                "(`result_hash`), a handoff (`handoff_hash`), an update or a study's book."
            ),
            "next_action": "NAME_ONE_BOOK",
            "next_requests": {"history": {"operation": "RESEARCH_HISTORY"}},
        }
    if base == "product_host.evidence_review_study_selector_not_a_study":
        # A study's selector that named another owner's Task, a replay among them (V546).
        return {
            "detail": (
                f"{f'Task {task_id}' if task_id else 'The Task named'} is a `{subject}` Task, "
                "not a study, so a study's selector does not name its book. "
                + (
                    "Its book is its published result, named by `result_hash` alone, as an "
                    "installed strategy's historical replay is: the same request by the receipt "
                    "it named is offered."
                    if book
                    else "`task show` reads it, with the requests its owner offers."
                )
            ),
            "next_action": "SEND_THE_OFFERED_REQUEST" if book else "READ_THE_TASK_WITH_TASK_SHOW",
            "next_requests": {"result": dict(book)}
            if book
            else ({"task": {"operation": "STATUS", **task}} if task else {}),
        }
    if base == "product_host.evidence_review_update_selector_invalid" and subject:
        return {
            "detail": (
                f"`{subject}` names an update's book, beside `update_task_id` and "
                "`update_publication_hash`. An experiment's book is named by "
                "`experiment_task_id`, `experiment_receipt_hash` and `portfolio_session` alone, "
                "a result's by `result_hash` and a handoff's by `handoff_hash`: the same request "
                f"without `{subject}` is offered."
            ),
            "next_action": "SEND_THE_OFFERED_REQUEST",
            "next_requests": {"without_" + subject: dict(book)} if book else {},
        }
    return {}


def _trial_way_on(trial: Mapping[str, Any] | None) -> dict[str, Any]:
    """A refused trial baseline's words and its way on (V354)."""
    why = (
        "A feature is tried against an Alpha study handed off from a Factor study, or the "
        "Portfolio study built on one: its factors are what the feature joins."
    )
    if trial is None:
        return {"detail": why, "next_requests": {"studies": {"operation": "EXPERIMENTS"}}}
    plan = trial["feature_plan_hash"]
    if trial["studies"]:
        return {
            "detail": why
            + " The trial can run against the completed ones on the feature's input, offered "
            "here newest first.",
            "next_requests": {
                f"trial:{task_id}": {
                    "operation": "FEATURE_TRIAL",
                    "feature_plan_hash": plan,
                    "task_id": task_id,
                }
                for task_id in trial["studies"]
            },
        }
    if trial["factor_studies"]:
        return {
            "detail": why
            + " None is complete on the feature's input. A completed Factor study there opens "
            "one: its curation chooses the factors, its handoff the Alpha declaration, and the "
            "Alpha study's run is the trial's baseline; the feature's build is kept.",
            "next_requests": {
                f"curation:{task_id}": {"operation": "EXPERIMENT_CURATION", "task_id": task_id}
                for task_id in trial["factor_studies"]
            },
        }
    return {
        "detail": why
        + " Neither such a study nor a completed Factor study is on the feature's input: a "
        "Factor study on it comes first, then its handoff.",
        "next_requests": {"studies": {"operation": "EXPERIMENTS"}},
    }


def _nomination(rule: str) -> str:
    why = {
        "empty": "names no candidate",
        "repeated": "names a candidate twice",
        "not_in_family": "names a candidate the goal's family did not attempt",
    }.get(rule, "is not a list of the family's candidates")
    return (
        f"`qualification.nominated_candidate_ids` {why}: it lists distinct candidates from "
        "the family, every one of which `expected` gives. Nominate from those and plan again."
    )


def _qualification_parent(field: str) -> str:
    return (
        f"A qualification's parent is its goal's family, named in its `qualification` "
        f"section, so its plan takes no `{field or 'parent field'}`. Plan again without it."
    )


def _qualification_declaration(field: str) -> str:
    what = {
        "methodology_id": "is not a family qualification's: its `alpha.methodology_id` is the "
        "one `expected` gives",
        "alpha": "holds other fields than a qualification's: its `alpha` section holds exactly "
        "the four `expected` lists, and a development study's fields do not belong there",
        "goal_id": "names no goal: `alpha.goal_id` is the id of the goal whose family it closes",
        "question_task_id": "names no study: `alpha.question_task_id` is the Task id of a "
        "development study that asked the family's question",
    }.get(field, "is not a qualification declaration")
    return f"The declaration {what}. Edit it and plan again."


_PLAN_FIELD_REFUSALS = {
    "alpha_research.qualification_declaration_invalid": _qualification_declaration,
    "alpha_research.qualification_nomination_invalid": _nomination,
    "alpha_research.qualification_unexpected_parent": _qualification_parent,
}
"""Plan refusals that name the declaration field and its requirement (V292): the subject after
the code's colon is the field or the rule, and `expected` beside the code what may be written."""


def deferral(
    answer: dict[str, Any],
    *,
    failure_code: str | None,
    retry_after_at: str | None,
    resume: dict[str, str] | None,
    held: bool,
) -> dict[str, Any]:
    """A deferred Task's answer with its words, its retry time and the request that resumes it.

    A preparation or a data update the provider deferred (V375) resumes when its owner is sent
    the same plan again once `retry_after_at` has passed; before then the owner refuses it
    (`retry_not_due`, worded). The answer says what the provider did, keeps what was fetched,
    and, where the workspace holds published research inputs (`held`), that they stand.

    Args:
        answer: The owner's readback of the deferred Task.
        failure_code: The Task's failure code, the provider's.
        retry_after_at: When the owner accepts the plan again; None when it names no time.
        resume: The request that sends the same plan again; None when the answer names none.
        held: Whether the workspace's published research inputs stand meanwhile.

    Returns:
        The answer with `detail`, `retry_after_at` and `next_requests.resume`.
    """
    words = explain(failure_code or "").get("detail") or (
        "The work was deferred and keeps what it did; send the same plan again once "
        "`retry_after_at` has passed."
    )
    return {
        **answer,
        "detail": words
        + (
            " The published research inputs are unchanged, and every study keeps reading them."
            if held
            else ""
        ),
        "retry_after_at": retry_after_at,
        "next_requests": {
            **(answer.get("next_requests") or {}),
            **({"resume": resume} if resume else {}),
        },
    }


def refused(code: str, **context: Any) -> dict[str, Any]:
    """A typed refusal carrying its explanation."""
    return {"status": "REFUSED", "failure_code": code, **explain(code, **context)}


def task_record_refusal(task_id: str) -> dict[str, Any]:
    """Name unreadable canonical authority without guessing its state or study kind."""
    code = "task_control.database_authority_unreadable"
    return {
        "status": "REFUSED",
        "task_id": task_id,
        "failure_code": code,
        **refusal_words(code),
        "next_requests": {
            "workspace": {"operation": "WORKSPACE_SHOW"},
            "backups": {"operation": "WORKSPACE_BACKUPS"},
        },
    }


EVIDENCE_UNIT_CODES = frozenset(
    {
        "alternative_evidence.unit_not_prepared",
        "alternative_evidence.preparation_not_complete",
        "alternative_evidence.coverage_unit_unknown",
    }
)
"""A coverage unit's state refusing its packet or bundle (V388)."""

SELECTOR_CODES = frozenset(
    {
        "product_host.evidence_review_experiment_selector_invalid",
        "product_host.evidence_review_update_selector_invalid",
        "product_host.evidence_review_selector_ambiguous",
    }
)
"""A review request's book selector refused for its parts, worded with which parts name a book."""

STALE_PACKET = "alternative_evidence.task_resource_authority_mismatch"
"""A packet prepared under authority the Host no longer holds (V547)."""


def stale_packet_refusal(
    code: str, *, role: str, book: Mapping[str, Any], unit_id: object
) -> dict[str, Any]:
    """An answer to a packet that went stale, refused with what moved and the way on (V547).

    Args:
        code: The refusal, `alternative_evidence.task_resource_authority_mismatch:<what moved>`.
        role: The bundle's role.
        book: The book the bundle's submission names.
        unit_id: The packet's coverage unit, when it was one.

    Returns:
        The refusal, its words and the book's preview, which prepares the unit again.
    """
    return {
        "status": "REFUSED",
        "agent_role": role,
        "failure_code": code,
        **refusal_words(code),
        **({"evidence_unit_id": str(unit_id)} if unit_id else {}),
        "next_requests": {"preview": {"operation": "EVIDENCE_PREVIEW", **book}},
    }


STOP_WORDS_BOUND = 500
"""The most a Task's stop is worded with in its recovery view: an owner's words for a
code a Task stops on fit it whole, never cut (V504, V565)."""

SOURCE_SHORT_CODES = frozenset(
    {
        "alternative_evidence.minimum_entity_coverage_not_met",
        "alternative_evidence.document_set_empty",
    }
)
"""A unit's refusals for too few sources: their way on is sources, never a retry (V541)."""


def unit_failure_words(code: str, *, recorded: bool = True) -> dict[str, str]:
    """A failed coverage unit's words and way on, whatever its code (V541).

    Its code's own words where the door table words it, else the owner's code a task failure
    carries where that one is worded, else the words of a refusal of the preparation's own
    records or work, which name the code. A unit short of sources under official acquisition is
    prepared again, its filings retried; under the recorded package that fails the same way.

    Args:
        code: The unit's failure code, as its sealed failure records it.
        recorded: Whether the Host reads the recorded package rather than the official source.

    Returns:
        The unit's ``detail`` and ``next_action``.
    """
    base, _sep, inner = code.partition(":")
    words: dict[str, str] = refusal_words(code)
    if not words and base == "alternative_evidence.task_failed" and inner:
        words = refusal_words(inner)
    if not words:
        words = refusal_words(f"alternative_evidence.unit_failed:{code}")
    if base in SOURCE_SHORT_CODES and not recorded:
        return {**words, "next_action": "PREVIEW_AND_PREPARE_AGAIN"}
    return words


def unit_refusal(code: str, book: Mapping[str, str] | None) -> dict[str, Any] | None:
    """A unit's packet or bundle refused for the unit's state, with its words and way on.

    The way on is the book's coverage read, which names every unit's state and offers each
    prepared unit's packet and Analyst bundle (V388).

    Args:
        code: The refusal's code, its subject after the first colon.
        book: The book's selector fields as the request named them, or None for the
            workspace's default book.

    Returns:
        The refusal, or None when the code is not a unit's state.
    """
    if code.partition(":")[0] not in EVIDENCE_UNIT_CODES:
        return None
    return {
        **refused(code),
        "next_requests": {"coverage": {"operation": "EVIDENCE_CRO", **(book or {})}},
    }
