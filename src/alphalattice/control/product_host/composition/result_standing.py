"""What a result can claim: one typed standing generated from its owners' marks.

Whether a result ran to its end, whether its contract passed, whether it was compared with its
baseline, what its evidence supports and whether a person may activate it are separate marks,
each recorded by its owner in its own place: a Task's lifecycle or a trial's state, the Desk
verifier's method standing or a formula factor's goldens, the owner's comparison, the
study's lane, its qualification or the trial's screening, the activation registry and its
preconditions. Every result answer -- a study, a book, a trial, a review packet -- carries them
as one `standing`, each field from one mark and each value with one statement, so an agent and a
person read one answer to what the result can claim, and a mark's new value changes the
statement with no carrier edited, as the temporal statement does. The comparison comes first.
The launch budget is an agent session's, which every answer carries as
`session_launches`, never a result's mark.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field

Comparison = Literal["COMPARED", "NOT_COMPARED", "NOT_APPLICABLE"]
Execution = Literal["SUCCEEDED", "RUNNING", "STOPPED", "NOT_STARTED"]
Contract = Literal["PASSED", "FAILED", "NOT_CURRENT", "NOT_CHECKED"]
Evidence = Literal[
    "DEVELOPMENT", "EXPLORATION", "QUALIFIED", "NOT_QUALIFIED", "SCREENED_OUT", "NONE"
]
Activation = Literal["ACTIVE", "A_PERSON_MAY_ACTIVATE", "HELD", "NOT_ACTIVATABLE"]

_STATEMENTS: Final[Mapping[str, Mapping[str, str]]] = {
    "comparison": {
        "COMPARED": (
            "Compared: its owner compared it with its baseline over the same scored values, so "
            "the change it states is its own."
        ),
        "NOT_COMPARED": (
            "Not compared: its owner refused to compare it with its baseline ({reason}); each "
            "side's saved metrics stand on what it scored, and no change is claimed."
        ),
        "NOT_APPLICABLE": (
            "No comparison with a baseline result (the study without the change) is part of it."
        ),
    },
    "execution": {
        "SUCCEEDED": "It ran to its end.",
        "RUNNING": "It is still running; read it again to follow it.",
        "STOPPED": "It stopped before its end ({reason}).",
        "NOT_STARTED": "Nothing has run for it yet.",
    },
    "contract": {
        "PASSED": "Its contract passed: what it rests on was checked again and verifies.",
        "FAILED": "Its contract failed ({reason}); nothing may rest on it until it passes.",
        "NOT_CURRENT": (
            "It reads back as recorded, but its method is no longer the installed one "
            "({reason}); a new run claims the current method."
        ),
        "NOT_CHECKED": "No contract of its own is checked here.",
    },
    "evidence": {
        "DEVELOPMENT": (
            "Development evidence: it supports a research decision, not an independent validation."
        ),
        "EXPLORATION": (
            "Exploration evidence on a sample of its input's names: it cannot become a strategy "
            "until the same declaration runs on the whole universe."
        ),
        "QUALIFIED": (
            "Qualified: its family's qualification found a stable candidate set; the sealed "
            "holdout is unread."
        ),
        "NOT_QUALIFIED": "Not qualified: its family's qualification found no stable current model.",
        "SCREENED_OUT": (
            "Screening found no detectable out-of-sample effect, so nothing was carried further."
        ),
        "NONE": "No evidence yet.",
    },
    "activation": {
        "ACTIVE": "Active: its activation is recorded.",
        "A_PERSON_MAY_ACTIVATE": "Read the exact activation offer for who may activate it.",
        "HELD": "It cannot be activated yet ({reason}).",
        "NOT_ACTIVATABLE": (
            "Nothing activates from it: it is evidence a person's decision rests on."
        ),
    },
}
"""Each mark's statement, by its value; `{reason}` is the owner's code that holds it."""

_FIELDS: Final = ("comparison", "execution", "contract", "evidence", "activation")
_RUNNING: Final = frozenset({"QUEUED", "RUNNING", "DEFERRED", "CANCEL_REQUESTED"})
"""The Task lifecycles still on their way to an end."""


class ResultStanding(BaseModel):  # type: ignore[misc]
    """One result's standing: its marks, the owners' codes that hold them, their statements."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    comparison: Comparison
    """First: whether its owner compared it with its baseline."""
    execution: Execution
    contract: Contract
    evidence: Evidence
    activation: Activation
    reasons: dict[str, str] = Field(default_factory=dict)
    """The owner's code behind each value that names one (not compared, stopped, failed, not
    current, held), by field."""
    statements: tuple[str, ...] = Field(min_length=len(_FIELDS), max_length=len(_FIELDS))

    @classmethod
    def of(
        cls,
        *,
        comparison: Comparison,
        execution: Execution,
        contract: Contract,
        evidence: Evidence,
        activation: Activation,
        reasons: Mapping[str, str] | None = None,
    ) -> Self:
        """State a result's marks, each value with its statement.

        Args:
            comparison: Whether its owner compared it with its baseline.
            execution: How far it ran.
            contract: Whether its owner's checks passed.
            evidence: What its evidence supports.
            activation: Whether a person may activate it.
            reasons: The owner's code behind each value that names one.

        Returns:
            The standing.

        Raises:
            ValueError: `result_standing.reasons_mismatch:<fields>` when a value that names a
                code has none, or a code stands beside a value that names none.
        """
        marks = {
            "comparison": comparison,
            "execution": execution,
            "contract": contract,
            "evidence": evidence,
            "activation": activation,
        }
        given = {key: value for key, value in (reasons or {}).items() if value}
        needed = {key for key, value in marks.items() if "{reason}" in _STATEMENTS[key][value]}
        if set(given) != needed:
            raise ValueError(
                "result_standing.reasons_mismatch:" + ",".join(sorted(set(given) ^ needed))
            )
        return cls(
            **marks,
            reasons=given,
            statements=tuple(
                _STATEMENTS[key][value].format(reason=given.get(key, ""))
                for key, value in marks.items()
            ),
        )


def study_standing(body: Mapping[str, Any]) -> ResultStanding:
    """A study's or a book's standing, from the marks its readback carries.

    Args:
        body: The experiment readback: its `status` (`EXPERIMENT_PUBLISHED` or the Task's
            lifecycle), `failure_code`, `method_standing` and `method_refusal`, `research_lane`
            and, for a family qualification, `alpha_qualification`'s `disposition`.

    Returns:
        The standing; nothing activates from a study.
    """
    status = str(body.get("status"))
    if status != "EXPERIMENT_PUBLISHED":
        running = status in _RUNNING
        return ResultStanding.of(
            comparison="NOT_APPLICABLE",
            execution="RUNNING" if running else "STOPPED",
            contract="NOT_CHECKED",
            evidence="NONE",
            activation="NOT_ACTIVATABLE",
            reasons={}
            if running
            else {"execution": str(body.get("failure_code") or f"task_{status.lower()}")},
        )
    qualification = body.get("alpha_qualification")
    disposition = qualification.get("disposition") if isinstance(qualification, Mapping) else None
    evidence: Evidence = (
        "QUALIFIED"
        if disposition == "CURRENT_ALPHA_CANDIDATE_SET_READY"
        else "NOT_QUALIFIED"
        if disposition == "NO_STABLE_CURRENT_ALPHA_MODEL"
        else "EXPLORATION"
        if body.get("research_lane") == "EXPLORATION"
        else "DEVELOPMENT"
    )
    current = body.get("method_standing") != "NOT_CURRENT"
    return ResultStanding.of(
        comparison="NOT_APPLICABLE",
        execution="SUCCEEDED",
        contract="PASSED" if current else "NOT_CURRENT",
        evidence=evidence,
        activation="NOT_ACTIVATABLE",
        reasons={} if current else {"contract": str(body.get("method_refusal") or "method")},
    )


def trial_standing(
    *,
    state: str,
    outcome: str | None,
    stopped: Mapping[str, Any] | None,
    comparison: Mapping[str, Any] | None,
) -> ResultStanding:
    """A feature trial's standing: its state, its screening and its comparison.

    Args:
        state: `RUNNING`, `COMPLETED` or `STOPPED`.
        outcome: A completed trial's: compared through the chain, or screened out.
        stopped: A stopped trial's step and its owner's `failure_code`.
        comparison: A completed trial's `alpha_without_and_with`, its `standing` and, not
            compared, its `owner_comparison`'s `failure_code`.

    Returns:
        The standing; a person activates a formula factor from its review packet, never from a
        trial.
    """
    if state != "COMPLETED":
        return ResultStanding.of(
            comparison="NOT_APPLICABLE",
            execution="RUNNING" if state == "RUNNING" else "STOPPED",
            contract="NOT_CHECKED",
            evidence="NONE",
            activation="NOT_ACTIVATABLE",
            reasons={
                "execution": str((stopped or {}).get("failure_code") or "feature_trial.stopped")
            }
            if state == "STOPPED"
            else {},
        )
    if outcome == "FEATURE_NOT_ADMITTED_BY_SCREENING" or comparison is None:
        return ResultStanding.of(
            comparison="NOT_APPLICABLE",
            execution="SUCCEEDED",
            contract="NOT_CHECKED",
            evidence="SCREENED_OUT",
            activation="NOT_ACTIVATABLE",
        )
    owner = comparison.get("owner_comparison")
    compared = comparison.get("standing") == "COMPARED"
    return ResultStanding.of(
        comparison="COMPARED" if compared else "NOT_COMPARED",
        execution="SUCCEEDED",
        contract="NOT_CHECKED",
        evidence="DEVELOPMENT",
        activation="NOT_ACTIVATABLE",
        reasons={}
        if compared
        else {
            "comparison": str(
                (owner.get("failure_code") if isinstance(owner, Mapping) else None)
                or "feature_trial.comparison_refused"
            )
        },
    )


def packet_standing(
    *,
    contract: Mapping[str, Any],
    trials: list[Mapping[str, Any]],
    active: bool,
    held: str | None,
) -> ResultStanding:
    """A formula factor's review packet's standing: its contract, its newest trial, activation.

    Args:
        contract: The packet's contract: its `status` and, failed, its `failure_code`.
        trials: The packet's trials, oldest first, each its `state`, `outcome`, `stopped`,
            `alpha_standing` and `alpha_refusal`.
        active: Whether the factor is in this workspace's daily catalog.
        held: The code a person's activation would be refused with now; None when it would not.

    Returns:
        The standing, its trial marks from the newest completed trial, else the newest.
    """
    completed = [trial for trial in trials if trial.get("state") == "COMPLETED"]
    newest = completed[-1] if completed else trials[-1] if trials else None
    passed = contract.get("status") == "PASSED"
    if newest is None:
        trial = None
    else:
        refusal = newest.get("alpha_refusal")
        stopped = newest.get("stopped")
        trial = trial_standing(
            state=str(newest.get("state")),
            outcome=newest.get("outcome"),
            stopped=stopped if isinstance(stopped, Mapping) else None,
            comparison={
                "standing": newest.get("alpha_standing"),
                "owner_comparison": refusal if isinstance(refusal, Mapping) else None,
            }
            if newest.get("alpha_standing")
            else None,
        )
    reasons = {
        **({} if trial is None else {k: v for k, v in trial.reasons.items() if k != "contract"}),
        **(
            {}
            if passed
            else {"contract": str(contract.get("failure_code") or "goldens_outside_tolerance")}
        ),
        **({"activation": held} if held is not None and not active else {}),
    }
    return ResultStanding.of(
        comparison="NOT_APPLICABLE" if trial is None else trial.comparison,
        execution="NOT_STARTED" if trial is None else trial.execution,
        contract="PASSED" if passed else "FAILED",
        evidence="NONE" if trial is None else trial.evidence,
        activation="ACTIVE" if active else "HELD" if held is not None else "A_PERSON_MAY_ACTIVATE",
        reasons=reasons,
    )


def _complete() -> None:
    """Every value of every mark has its statement, and no statement stands for no value."""
    for field, values in zip(
        _FIELDS, (Comparison, Execution, Contract, Evidence, Activation), strict=True
    ):
        assert set(_STATEMENTS[field]) == set(get_args(values)), field


_complete()

__all__ = ["ResultStanding", "packet_standing", "study_standing", "trial_standing"]
