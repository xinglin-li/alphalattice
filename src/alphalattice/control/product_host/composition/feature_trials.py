"""Try a feature in one step (binding plan, B18).

One request names a planned research feature and a finished study to try it against (an
Alpha study handed off from a Factor study, or the Portfolio study built on one). The Host
then runs the chain the researcher would otherwise ask for one by one: it builds and
prepares the feature, screens it beside the study's factors in a Factor study, curates the
study's own factors and the feature, runs the study's Alpha declaration on that selection
at the study's scale (its exploration sample, or its whole universe), and the Portfolio
study's policy on that Alpha when the study was a Portfolio study. Every step asks its owner
as a person would, so unchanged work answers `REUSED_EXACT` and is not computed again.

A trial is a durable record under `runtime/feature-trials/`, named by the feature plan and
the study, so asking again reopens it. It advances when it is asked for and whenever the
Host's Task worker goes idle; reading it changes nothing. A step that is refused or does not
succeed stops the trial there, by its owner's code. A completed trial's readback carries
the comparison: the feature's out-of-sample rank IC and its correlation with each factor
present, the Alpha study with and without it, and the book with and without it.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from alphalattice.control.product_host.composition.plain_refusals import explain
from alphalattice.control.product_host.composition.research_experiments import (
    ENVELOPE_SCHEMA_ID,
    TASK_KIND,
    ResearchExperimentApplication,
)
from alphalattice.control.product_host.composition.result_standing import trial_standing
from alphalattice.control.product_host.data_preparation.feature_research import (
    ResearchFeatureBuildApplication,
)
from alphalattice.control.product_host.research_authoring.timing import binding_temporal_scope
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.control.workspace_runtime.content_store import replace_shared_file
from alphalattice.foundation.factor_research.experiments.authoring import FACTOR_EXPERIMENT_KIND
from alphalattice.foundation.feature_engine.catalog.research import research_feature_execution_spec
from alphalattice.interface.local_application.cli_contract import ACTION_STATES
from alphalattice.interface.local_application.dispatcher import LocalBackgroundDispatcher
from alphalattice.interface.local_application.failure_codes import located_failure, public_failure
from alphalattice.interface.local_application.portfolio_research import (
    FactorCurationRequest,
    PortfolioResearchOperationRequest,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

TRIALS_DIRECTORY = "feature-trials"
Step = Literal["FEATURE_BUILD", "FACTOR_STUDY", "CURATION", "ALPHA_STUDY", "PORTFOLIO_STUDY"]
_TASK_STEPS: tuple[Step, ...] = ("FEATURE_BUILD", "FACTOR_STUDY", "ALPHA_STUDY", "PORTFOLIO_STUDY")
_ALPHA_METRICS = ("mean_rank_ic", "pooled_oos_r2", "mean_gross_decile_spread", "fold_coverage_mean")
_TERMINAL = {TaskLifecycle.SUCCEEDED, TaskLifecycle.BLOCKED, TaskLifecycle.CANCELLED}
_WAY_ON = 5
"""The most studies a refused baseline's way on offers of each kind, newest first (V354)."""


class FeatureTrialError(ValueError):
    """A trial request the Host refuses by name."""


class FeatureTrial(BaseModel):  # type: ignore[misc]
    """One trial: what it tries against what, and how far it has come."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    trial_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_task_id: str
    baseline_alpha_task_id: str
    baseline_portfolio_task_id: str | None = None
    caller: str
    requested_at: datetime
    state: Literal["RUNNING", "COMPLETED", "STOPPED"] = "RUNNING"
    outcome: Literal["COMPARED", "FEATURE_NOT_ADMITTED_BY_SCREENING"] | None = None
    """How a completed trial ended: compared through the chain, or at the Factor study,
    where screening found no effect to carry into an Alpha study."""
    steps: dict[str, str] = Field(default_factory=dict)
    """Each step's Task ID, or the curation's receipt hash."""
    preparation_hash: str | None = None
    feature_factor_ids: tuple[str, ...] = ()
    stopped: dict[str, str] | None = None
    """The step that stopped the trial and its owner's code."""


def _rationale(trial: FeatureTrial) -> str:
    # The feature plan, not the trial: two trials of one feature on one selection share
    # one curation decision, which answers REUSED_EXACT.
    return (
        f"Feature trial of plan {trial.feature_plan_hash[:12]}: the study's factors and the "
        "tried feature."
    )


def trial_id_for(feature_plan_hash: str, baseline_task_id: UUID) -> str:
    """Bind one feature plan and exact baseline task into a deterministic trial identity.

    Args:
        feature_plan_hash: Exact admitted feature plan.
        baseline_task_id: Explicit baseline task identity.

    Returns:
        Canonical feature-trial identity.
    """
    return str(
        canonical_hash(
            {
                "kind": "feature-trial",
                "feature_plan_hash": feature_plan_hash,
                "baseline_task_id": str(baseline_task_id),
            }
        )
    )


def trial_requests(trial: FeatureTrial) -> dict[str, dict[str, str]]:
    """The trial's own read, and on completion each factor's review, filled (V391).

    The trial id, plan hash and factor id travel by `--from`, so an agent never copies them.

    Args:
        trial: The trial as recorded.

    Returns:
        `show`, and once the trial completed `review` (or `review:<factor>` for each of
        several factors).
    """
    reviews = {
        ("review" if len(trial.feature_factor_ids) == 1 else f"review:{factor}"): {
            "operation": "FEATURE_REVIEW",
            "feature_plan_hash": trial.feature_plan_hash,
            "feature_factor_id": factor,
        }
        for factor in trial.feature_factor_ids
    }
    return {
        "show": {"operation": "FEATURE_TRIAL_READBACK", "feature_trial_id": trial.trial_id},
        **(reviews if trial.state == "COMPLETED" else {}),
    }


def waiting_step(steps: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The trial's step whose Task waits on a request (its recovery, a review), if one does.

    A Task in such a state waits on the trial's caller, never on time (`ACTION_STATES`), so the
    trial names it and a wait on the trial ends there (V449: every state has its exit).

    Args:
        steps: The trial's steps as its readback lists them.

    Returns:
        The first such step, or None.
    """
    return next((step for step in steps if step.get("state") in ACTION_STATES), None)


class FeatureTrials:
    """The Host's feature trials over its research applications."""

    def __init__(
        self,
        *,
        workspace: Path,
        experiments: ResearchExperimentApplication,
        feature_builds: ResearchFeatureBuildApplication,
        dispatcher: LocalBackgroundDispatcher,
        clock: Callable[[], datetime],
    ) -> None:
        """Compose trial storage, baseline/feature build owners and bounded background dispatch.

        Args:
            workspace: Caller-owned workspace root.
            experiments: Deterministic experiment owner.
            feature_builds: Deterministic feature build owner.
            dispatcher: Local background task dispatcher.
            clock: Explicit observed-time source.
        """
        self.root = workspace / "runtime" / TRIALS_DIRECTORY
        self.workspace = workspace
        self.experiments = experiments
        self.feature_builds = feature_builds
        self.dispatcher = dispatcher
        self.clock = clock
        self._lock = Lock()

    # ------------------------------------------------------------------ records

    def _path(self, trial_id: str) -> Path:
        return self.root / f"{trial_id}.json"

    def _read(self, trial_id: str) -> FeatureTrial:
        path = self._path(trial_id)
        if not path.is_file():
            raise FeatureTrialError("feature_trial.not_found")
        try:
            return cast(FeatureTrial, FeatureTrial.model_validate_json(path.read_bytes()))
        except ValueError as error:
            # A record changed or cut short in place is named, never read as a trial (V272).
            raise FeatureTrialError("feature_trial.record_damaged") from error

    def _existing(self, trial_id: str) -> FeatureTrial | None:
        """The trial's record, or None; a damaged one is kept aside, so the trial starts again."""
        path = self._path(trial_id)
        if not path.is_file():
            return None
        try:
            return self._read(trial_id)
        except FeatureTrialError as error:
            if str(error) != "feature_trial.record_damaged":
                raise
        os.replace(path, path.with_name(f"{path.name}.damaged-{self.clock():%Y%m%dT%H%M%S%f}"))
        return None

    def _save(self, trial: FeatureTrial) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(trial.trial_id)
        staged = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.partial")
        staged.write_text(trial.model_dump_json(indent=1), encoding="utf-8")
        replace_shared_file(staged, path)  # concurrent reads of the trial hold it (V477)

    def records(self) -> tuple[FeatureTrial, ...]:
        """Every trial on the ledger that reads, oldest name first."""
        return self._records()[0]

    def _records(self) -> tuple[tuple[FeatureTrial, ...], tuple[str, ...]]:
        """Every trial that reads, and the IDs of the records that do not."""
        if not self.root.is_dir():
            return (), ()
        found, damaged = [], []
        for path in sorted(self.root.glob("*.json")):
            try:
                found.append(FeatureTrial.model_validate_json(path.read_bytes()))
            except (OSError, ValueError):
                damaged.append(path.stem)
        return tuple(found), tuple(damaged)

    # ------------------------------------------------------------------ requests

    def start(
        self, feature_plan_hash: str, baseline_task_id: UUID, *, caller: str
    ) -> dict[str, Any]:
        """Open (or reopen) the trial of a planned feature against a finished study."""
        with self._lock:
            trial_id = trial_id_for(feature_plan_hash, baseline_task_id)
            existing = self._existing(trial_id)
            definitions = self.feature_builds.definitions()
            plan = definitions.read(feature_plan_hash)
            if existing is None:
                executed = research_feature_execution_spec(plan)
                existing = next(
                    (
                        record
                        for record in self.records()
                        if record.baseline_task_id == str(baseline_task_id)
                        and research_feature_execution_spec(
                            definitions.read(record.feature_plan_hash)
                        )
                        == executed
                    ),
                    None,
                )
            if existing is not None:
                trial = existing
                trial_id = trial.trial_id
            else:
                alpha_id, portfolio_id = self._baseline(baseline_task_id)
                alpha_plan = self._plan(alpha_id)
                if plan.request.input_binding_hash != alpha_plan.binding.binding_hash:
                    raise FeatureTrialError("feature_trial.feature_input_not_the_studys")
                trial = FeatureTrial(
                    trial_id=trial_id,
                    feature_plan_hash=feature_plan_hash,
                    baseline_task_id=str(baseline_task_id),
                    baseline_alpha_task_id=str(alpha_id),
                    baseline_portfolio_task_id=None if portfolio_id is None else str(portfolio_id),
                    caller=caller,
                    requested_at=self.clock(),
                )
                self._save(trial)
            self._advance(trial)
        return self.readback(trial_id)

    def advance_all(self) -> None:
        """Move every running trial on; the Host calls this when its Task worker is idle."""
        with self._lock:
            for trial in self.records():
                if trial.state == "RUNNING":
                    self._advance(trial)

    def listing(self) -> dict[str, Any]:
        """List retained trial state and explicitly report damaged records without numerical work.

        Returns:
            Trial metadata/current steps, damaged-record refusals with their detail and
            requests, and zero numerical_call_count.
        """
        trials, damaged = self._records()
        damaged_code = "feature_trial.record_damaged"
        damaged_words = explain(damaged_code)
        damaged_requests = {
            **(damaged_words.get("next_requests") or {}),
            "storage": {"operation": "STORAGE_READBACK"},
        }
        return {
            "status": "FEATURE_TRIALS",
            "trials": [
                {
                    "feature_trial_id": trial.trial_id,
                    "state": trial.state,
                    "feature_plan_hash": trial.feature_plan_hash,
                    "baseline_task_id": trial.baseline_task_id,
                    "requested_at": trial.requested_at.isoformat(),
                    "step": self._current_step(trial),
                }
                for trial in trials
            ],
            # A record that no longer reads is named, never skipped. Its own route reads the
            # kept trials, and Storage lets the reader inspect the retained record (V272).
            "damaged": [
                {
                    "status": "REFUSED",
                    "feature_trial_id": trial_id,
                    "failure_code": damaged_code,
                    "detail": damaged_words["detail"],
                    "next_requests": damaged_requests,
                }
                for trial_id in damaged
            ],
            "numerical_call_count": 0,
        }

    def readback(self, trial_id: str) -> dict[str, Any]:
        """The trial as it stands; a completed trial's comparison. Changes nothing.

        Inside `trial_reads_once`, a completed trial, which nothing changes, is read once per
        request and its later reads answer from it (V461).
        """
        reads = _TRIAL_READS.get()
        if reads is not None and trial_id in reads:
            return copy.deepcopy(reads[trial_id])
        body = self._readback(trial_id)
        if reads is not None and body["state"] == "COMPLETED":
            reads[trial_id] = copy.deepcopy(body)
        return body

    def _readback(self, trial_id: str) -> dict[str, Any]:
        trial = self._read(trial_id)
        registry = self.experiments.session.task_control_registry
        steps = []
        for step in (*_TASK_STEPS[:2], "CURATION", *_TASK_STEPS[2:]):
            value = trial.steps.get(step)
            if step == "PORTFOLIO_STUDY" and trial.baseline_portfolio_task_id is None:
                continue
            entry: dict[str, Any] = {"step": step}
            if value is None:
                entry["state"] = "WAITING"
            elif step == "CURATION":
                entry.update(state="SUCCEEDED", curation_receipt_hash=value)
            else:
                entry.update(task_id=value, state=registry.task(UUID(value)).lifecycle.value)
            steps.append(entry)
        body: dict[str, Any] = {
            "status": "FEATURE_TRIAL",
            "feature_trial_id": trial.trial_id,
            "state": trial.state,
            "feature_plan_hash": trial.feature_plan_hash,
            "baseline_task_id": trial.baseline_task_id,
            "feature_factor_ids": list(trial.feature_factor_ids),
            "steps": steps,
            "stopped": trial.stopped,
            "outcome": trial.outcome,
            # What the trial's input can claim about time, from its Panel (V347).
            "temporal_scope": binding_temporal_scope(
                self.workspace,
                self.feature_builds.definitions()
                .read(trial.feature_plan_hash)
                .request.input_binding_hash,
                window_start=None,
                window_end=None,
            ),
            "numerical_call_count": 0,
        }
        body["next_requests"] = trial_requests(trial)
        # A step whose Task waits on a request (its recovery, a review) is the trial's to
        # answer: the trial names that Task and its lifecycle and offers its recovery, so a
        # wait on the trial ends there (V449: every state has its exit).
        waiting = waiting_step(steps)
        if trial.state == "RUNNING" and waiting is not None:
            body.update(lifecycle=waiting["state"], task_id=waiting["task_id"])
            body["next_requests"]["recovery"] = {
                "operation": "TASK_RECOVERY",
                "task_id": waiting["task_id"],
            }
        if trial.state == "COMPLETED":
            body["comparison"] = self._comparison(trial)
        elif trial.state == "RUNNING":
            body["detail"] = (
                "The trial runs its chain one step after another; each step reuses what did not "
                "change. Read it again to follow it."
            )
        # What the trial can claim, one standing from its marks (V368).
        body["standing"] = trial_standing(
            state=trial.state,
            outcome=trial.outcome,
            stopped=trial.stopped,
            comparison=(body.get("comparison") or {}).get("alpha_without_and_with"),
        ).model_dump(mode="json")
        return body

    def way_on(self, feature_plan_hash: str) -> dict[str, Any] | None:
        """What a trial of this plan can run against, for a refused baseline (V354).

        The completed Alpha studies handed off from a Factor study on the feature's input, and
        the Portfolio studies built on one; when there is none, the completed Factor studies on
        that input, whose curation opens the handoff. Newest first, at most `_WAY_ON` of each.
        Nothing is created or accepted, and a build already made is kept for the trial.

        Args:
            feature_plan_hash: The planned feature the trial tries.

        Returns:
            The plan, its studies and its Factor studies; None when the plan does not read.
        """
        try:
            plan = self.feature_builds.definitions().read(feature_plan_hash)
        except (OSError, ValueError):
            return None
        binding = plan.request.input_binding_hash
        registry = self.experiments.session.task_control_registry
        studies: list[str] = []
        factors: list[str] = []
        for task in sorted(registry.tasks(), key=lambda value: value.admitted_at, reverse=True):
            if task.task_kind != TASK_KIND or task.lifecycle is not TaskLifecycle.SUCCEEDED:
                continue
            study = self._plan(task.task_id)
            if study.program.kind == FACTOR_EXPERIMENT_KIND:
                if study.binding.binding_hash == binding and len(factors) < _WAY_ON:
                    factors.append(str(task.task_id))
                continue
            try:
                alpha_id, _portfolio = self._baseline(task.task_id)
            except FeatureTrialError:
                continue
            if self._plan(alpha_id).binding.binding_hash == binding and len(studies) < _WAY_ON:
                studies.append(str(task.task_id))
        return {
            "feature_plan_hash": feature_plan_hash,
            "studies": studies,
            "factor_studies": [] if studies else factors,
        }

    # ------------------------------------------------------------------ the chain

    def _plan(self, task_id: UUID) -> Any:
        return self.experiments._of(self.experiments.session.task_control_registry.task(task_id))[0]

    def _baseline(self, task_id: UUID) -> tuple[UUID, UUID | None]:
        task = self.experiments.session.task_control_registry.task(task_id)
        if task.lifecycle is not TaskLifecycle.SUCCEEDED:
            raise FeatureTrialError("feature_trial.completed_study_required")
        plan = self._plan(task_id)
        if plan.portfolio_source is not None:
            alpha_id = UUID(plan.portfolio_source.alpha_task_id)
            if self._plan(alpha_id).alpha_source is None:
                raise FeatureTrialError("feature_trial.study_not_from_factor_evidence")
            return alpha_id, task_id
        if plan.alpha_source is None:
            raise FeatureTrialError("feature_trial.study_not_from_factor_evidence")
        return task_id, None

    @staticmethod
    def _current_step(trial: FeatureTrial) -> str | None:
        for step in ("FEATURE_BUILD", "FACTOR_STUDY", "CURATION", "ALPHA_STUDY", "PORTFOLIO_STUDY"):
            if step not in trial.steps:
                return step
        return None

    def _stop(self, trial: FeatureTrial, step: str, code: str) -> None:
        trial.state = "STOPPED"
        trial.stopped = {"step": step, "failure_code": code[:200]}
        self._save(trial)

    def _waiting_on(self, trial: FeatureTrial, step: Step) -> bool:
        """True while the step's Task has not finished; stops the trial when it failed."""

        task = self.experiments.session.task_control_registry.task(UUID(trial.steps[step]))
        if task.lifecycle not in _TERMINAL:
            if trial.state == "STOPPED":
                trial.state, trial.stopped = "RUNNING", None
                self._save(trial)
            return True
        if task.lifecycle is not TaskLifecycle.SUCCEEDED:
            self._stop(trial, step, task.failure_code or f"task_{task.lifecycle.value.lower()}")
            return True
        return False

    def _admitted(self, trial: FeatureTrial, answer: dict[str, Any]) -> str:
        task = answer.get("task_id") or answer.get("publication_task_id")
        if not task:
            raise FeatureTrialError(str(answer.get("failure_code") or "feature_trial.not_admitted"))
        trial.state, trial.stopped = "RUNNING", None
        return str(task)

    def _advance(self, trial: FeatureTrial) -> None:
        try:
            self._advance_steps(trial)
        except ValueError as error:
            self._stop(
                trial,
                self._current_step(trial) or "COMPLETION",
                public_failure(error, "feature_trial.step_failed"),
            )

    def _advance_steps(self, trial: FeatureTrial) -> None:
        alpha_id = UUID(trial.baseline_alpha_task_id)
        alpha_plan = self._plan(alpha_id)
        binding = alpha_plan.binding
        factor_parent = alpha_plan.alpha_source.factor_task_id
        if "FEATURE_BUILD" not in trial.steps:
            answer = self.feature_builds.build(
                trial.feature_plan_hash,
                caller=trial.caller,
                dispatcher=self.dispatcher,
                preprocess=True,
            )
            trial.steps["FEATURE_BUILD"] = self._admitted(trial, answer)
            self._save(trial)
        if self._waiting_on(trial, "FEATURE_BUILD"):
            return
        if trial.preparation_hash is None:
            built = self.feature_builds.readback(UUID(trial.steps["FEATURE_BUILD"]))
            preparation = built.get("preparation") or {}
            trial.preparation_hash = str(preparation["content_hash"])
            self._save(trial)
        if "FACTOR_STUDY" not in trial.steps:
            factor_plan = self._plan(factor_parent)
            controls = self.experiments.controls(
                binding.input_id,
                binding.binding_hash,
                "factor.screening-development",
                None,
                trial.preparation_hash,
            )
            options = [str(v) for v in cast(list[Any], controls["factor_options"])]
            present = set(factor_plan.document["factor"]["factor_ids"])
            missing = present - set(options)
            if missing:
                raise FeatureTrialError("feature_trial.study_factor_not_in_prepared_input")
            # The factors the feature plan declares, not whatever else the prepared input holds.
            declared = {
                edit.factor_id
                for edit in self.feature_builds.definitions()
                .read(trial.feature_plan_hash)
                .request.edits
                if edit.operation != "RETIRE"
            }
            trial.feature_factor_ids = tuple(
                v for v in options if v in declared and v not in present
            )
            if not trial.feature_factor_ids:
                raise FeatureTrialError("feature_trial.feature_adds_no_factor")
            document = json.loads(json.dumps(factor_plan.document))
            document["experiment"].update(
                data_snapshot_handle=f"research-features@{trial.preparation_hash}",
                output_workspace="managed",
                baseline_workspace="managed",
            )
            document["experiment"].pop("envelope_hash", None)
            wanted = present | set(trial.feature_factor_ids)
            document["factor"]["factor_ids"] = [v for v in options if v in wanted]
            planned = self.experiments.plan(
                binding.input_id, document, None, binding.binding_hash, caller=trial.caller
            )
            if planned.get("status") != "PLANNED":
                raise FeatureTrialError(str(planned.get("failure_code")))
            sent = self.experiments.run(
                str(planned["plan_hash"]), caller=trial.caller, agent_execution=None
            )
            trial.steps["FACTOR_STUDY"] = self._admitted(trial, sent)
            self._save(trial)
        if self._waiting_on(trial, "FACTOR_STUDY"):
            return
        factor_task = UUID(trial.steps["FACTOR_STUDY"])
        if "CURATION" not in trial.steps:
            readback = self.experiments.operate(
                PortfolioResearchOperationRequest(
                    operation="EXPERIMENT_CURATION", task_id=factor_task
                ),
                caller=trial.caller,
            )
            roles = {
                str(choice["factor_id"]): list(choice.get("roles") or [])
                for choice in cast(list[dict[str, Any]], readback["choices"])
            }
            study_factors = list(alpha_plan.document["alpha"]["ordered_feature_ids"])
            if any(not roles.get(factor) for factor in study_factors):
                raise FeatureTrialError("feature_trial.study_factor_not_curatable")
            if not any(roles.get(factor) for factor in trial.feature_factor_ids):
                # A real answer, not a failure: screening found no effect to carry on.
                trial.state, trial.outcome = "COMPLETED", "FEATURE_NOT_ADMITTED_BY_SCREENING"
                trial.stopped = None
                self._save(trial)
                return
            selected = [
                *study_factors,
                *(factor for factor in trial.feature_factor_ids if roles.get(factor)),
            ]
            wanted = {(factor, roles[factor][0]) for factor in dict.fromkeys(selected)}
            # A decision with these choices on this Factor study is reused: a decision binds
            # its rationale, so the same choices in other words would be a new one.
            existing = next(
                (
                    str(saved["receipt_hash"])
                    for saved in cast(list[dict[str, Any]], readback.get("decisions") or [])
                    if {
                        (str(choice["factor_id"]), str(choice["role"]))
                        for choice in saved["submission"]["proposal"]["choices"]
                    }
                    == wanted
                ),
                None,
            )
            if existing is not None:
                trial.steps["CURATION"] = existing
                self._save(trial)
            else:
                decision = self.experiments.operate(
                    PortfolioResearchOperationRequest(
                        operation="EXPERIMENT_CURATE",
                        task_id=factor_task,
                        experiment_curation=FactorCurationRequest.model_validate(
                            {
                                "expected_receipt_hash": readback["receipt_hash"],
                                "choices": [
                                    {
                                        "factor_id": factor,
                                        "role": roles[factor][0],
                                        "rationale": _rationale(trial),
                                    }
                                    for factor in dict.fromkeys(selected)
                                ],
                                "limitations_acknowledged": readback["limitations"],
                            }
                        ),
                    ),
                    caller=trial.caller,
                )
                trial.steps["CURATION"] = str(
                    cast(dict[str, Any], decision["decision"])["receipt_hash"]
                )
                self._save(trial)
        if "ALPHA_STUDY" not in trial.steps:
            alpha = alpha_plan.document["alpha"]
            planned = self.experiments.plan(
                binding.input_id,
                {
                    "experiment": {
                        # The installed envelope schema, as every declaration names it (V128);
                        # without it the Alpha step was refused whenever screening admitted the
                        # feature (V351).
                        "schema_id": ENVELOPE_SCHEMA_ID,
                        "kind": "alpha.model-development",
                        "universe_handle": alpha_plan.document["experiment"]["universe_handle"],
                    },
                    "alpha": {
                        key: alpha[key]
                        for key in (
                            "target_recipe_id",
                            "model_capability_handle",
                            "model_parameters",
                        )
                        if key in alpha
                    },
                },
                None,
                binding.binding_hash,
                None,
                factor_task,
                trial.steps["CURATION"],
                caller=trial.caller,
            )
            if (
                planned.get("status") not in {"PLANNED", "DRAFT_INCOMPLETE"}
                or "plan_hash" not in planned
            ):
                raise FeatureTrialError(
                    str(planned.get("failure_code") or "feature_trial.alpha_not_planned")
                )
            sent = self.experiments.run(
                str(planned["plan_hash"]), caller=trial.caller, agent_execution=None
            )
            trial.steps["ALPHA_STUDY"] = self._admitted(trial, sent)
            self._save(trial)
        if self._waiting_on(trial, "ALPHA_STUDY"):
            return
        if trial.baseline_portfolio_task_id is not None:
            if "PORTFOLIO_STUDY" not in trial.steps:
                baseline = self._plan(UUID(trial.baseline_portfolio_task_id))
                trial_alpha = UUID(trial.steps["ALPHA_STUDY"])
                document, portfolio_binding = self.experiments.portfolio_document(
                    trial_alpha,
                    baseline.document["portfolio"],
                    candidate_id=self._candidate(trial_alpha),
                )
                planned = self.experiments.plan(
                    portfolio_binding.input_id,
                    document,
                    None,
                    portfolio_binding.binding_hash,
                    caller=trial.caller,
                )
                if planned.get("status") != "PLANNED":
                    raise FeatureTrialError(str(planned.get("failure_code")))
                sent = self.experiments.run(
                    str(planned["plan_hash"]), caller=trial.caller, agent_execution=None
                )
                trial.steps["PORTFOLIO_STUDY"] = self._admitted(trial, sent)
                self._save(trial)
            if self._waiting_on(trial, "PORTFOLIO_STUDY"):
                return
        trial.state, trial.outcome = "COMPLETED", "COMPARED"
        trial.stopped = None
        self._save(trial)

    def _alpha_metrics(self, alpha_task_id: UUID) -> dict[str, Any]:
        """One Alpha study's saved headline metrics, for its only candidate."""

        candidate = self._candidate(alpha_task_id)
        summary = self.experiments.summary(alpha_task_id)
        saved = next(
            value
            for value in cast(dict[str, Any], summary["result"])["candidates"]
            if value["candidate_id"] == candidate
        )
        return {
            "task_id": str(alpha_task_id),
            "candidate_id": candidate,
            **{key: saved.get(key) for key in (*_ALPHA_METRICS, "target_lane")},
        }

    def _candidate(self, alpha_task_id: UUID) -> str:
        body = self.experiments.readback(alpha_task_id)
        lineage = cast(dict[str, Any], body["receipt"])["child_lineage"]
        candidates = sorted({str(entry["candidate_id"]) for entry in lineage})
        if len(candidates) != 1:
            raise FeatureTrialError("feature_trial.alpha_candidate_ambiguous")
        return candidates[0]

    # ------------------------------------------------------------------ the answer

    def _comparison(self, trial: FeatureTrial) -> dict[str, Any]:
        factor = self.experiments.readback(UUID(trial.steps["FACTOR_STUDY"]))
        result = cast(dict[str, Any], factor["result"])
        tried = set(trial.feature_factor_ids)
        entries = [
            item
            for item in result["evidence_report"]["items"]
            if str(item.get("factor_id")) in tried
        ]
        pairs = [
            pair
            for pair in result["redundancy_structure"]["pair_evidence"]
            if {str(pair.get("left_factor_id")), str(pair.get("right_factor_id"))} & tried
        ]
        feature_body = {
            "factor_ids": list(trial.feature_factor_ids),
            "out_of_sample_evidence": entries,
            "correlation_with_present_factors": pairs,
        }
        if trial.outcome == "FEATURE_NOT_ADMITTED_BY_SCREENING":
            return {
                "feature": feature_body,
                "detail": (
                    "Screening found no detectable out-of-sample effect for the feature, so no "
                    "Alpha study or book was run with it; its evidence and its correlation with "
                    "the factors present are above."
                ),
            }
        baseline_alpha = UUID(trial.baseline_alpha_task_id)
        trial_alpha = UUID(trial.steps["ALPHA_STUDY"])
        alpha: dict[str, Any] = {
            "without": self._alpha_metrics(baseline_alpha),
            "with": self._alpha_metrics(trial_alpha),
        }
        try:
            alpha["owner_comparison"] = self.experiments.operate(
                PortfolioResearchOperationRequest(
                    operation="EXPERIMENT_ALPHA_COMPARE",
                    left_task_id=baseline_alpha,
                    left_candidate_id=self._candidate(baseline_alpha),
                    right_task_id=trial_alpha,
                    right_candidate_id=self._candidate(trial_alpha),
                ),
                caller=trial.caller,
            )
        except ValueError as error:
            # The owner compares a feature addition over the same target values and names
            # what it refuses (a sample of other names, say); its saved metrics stand beside.
            alpha["owner_comparison"] = {
                "status": "REFUSED",
                **located_failure(error, "feature_trial.comparison_refused"),
            }
        owner = alpha["owner_comparison"]
        if owner.get("status") == "REFUSED":
            # Not compared (V363): each study's saved metrics stand on what it scored, and
            # their difference is no increment the feature made; the owner says why.
            alpha["standing"] = "NOT_COMPARED"
            owner.update(explain(str(owner.get("failure_code") or "")))
        else:
            alpha["standing"] = "COMPARED"
            alpha["change"] = {
                key: alpha["with"][key] - alpha["without"][key]
                for key in _ALPHA_METRICS
                if isinstance(alpha["with"].get(key), float)
                and isinstance(alpha["without"].get(key), float)
            }
        body: dict[str, Any] = {"feature": feature_body, "alpha_without_and_with": alpha}
        if trial.baseline_portfolio_task_id is not None:
            body["book_without_and_with"] = self.experiments.operate(
                PortfolioResearchOperationRequest(
                    operation="EXPERIMENT_COMPARE",
                    left_task_id=UUID(trial.baseline_portfolio_task_id),
                    right_task_id=UUID(trial.steps["PORTFOLIO_STUDY"]),
                ),
                caller=trial.caller,
            )
        return body


_TRIAL_READS: ContextVar[dict[str, dict[str, Any]] | None] = ContextVar(
    "feature_trial_reads", default=None
)


@contextmanager
def trial_reads_once() -> Iterator[None]:
    """One request reads each completed trial once (V461).

    A goal's submission and its read resolve the trial a reference names and the review packet
    another names, and the packet reads the same trial again, its comparison included. Inside
    this scope a completed trial answers from its first read in the request. An operation run
    inside another joins the scope already open; the reads are released with it.
    """
    if _TRIAL_READS.get() is not None:
        yield
        return
    token = _TRIAL_READS.set({})
    try:
        yield
    finally:
        _TRIAL_READS.reset(token)


__all__ = [
    "FeatureTrial",
    "FeatureTrialError",
    "FeatureTrials",
    "trial_id_for",
    "trial_reads_once",
]
