"""What an upgrade touched, said once, before the user meets it piece by piece.

Every saved study, published review and waiting Task is compared with what is installed now
through the identity it recorded when it was made. Nothing is verified, planned or computed:
the records are read from Task Control and the review store, and each installed identity is
its owner's own -- a study kind's implementation hash (the one its REPLAY compares), the
Evidence runtime's binding tuple and its listed history, the CRO review policy a review's
receipt recorded, and a Task owner's resume check. Opening a saved object stays its owner's read.

Each object lands in one state:
- a study is CURRENT (its kind's implementation is the one it ran under) or CHANGED (it reads
  back as recorded; a replay refuses, and a continuation runs its declared numerical calls
  again under the installed code);
- a review is CURRENT, CHANGED (sealed under an earlier Evidence binding, listed or
  observed in this workspace, or sealed under an earlier CRO review policy: it reads back
  as recorded, and a new review prepares evidence again) or UNREADABLE (an analysis that
  does not verify, which refuses);
- a Task that waits is RESUMABLE or CHANGED_SINCE_ADMISSION (its owner refuses it as
  recorded; cancel it and plan the same work again).

`show` is whether the page is owed: the installed identities differ from the ones the user
last acknowledged, and something is not current. Acknowledging records the identities the
user saw, in the workspace's runtime; the page stays reachable afterwards.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import UUID

from alphalattice.control.product_host.composition.evidence_review_application import (
    EvidenceReviewApplication,
)
from alphalattice.control.product_host.composition.plain_refusals import (
    kind_words,
    task_record_refusal,
)
from alphalattice.control.product_host.composition.research_experiment_plan import ExperimentPlan
from alphalattice.control.product_host.composition.research_experiments import (
    TASK_KIND,
    _implementation_hash,
    plan_implementation_role,
)
from alphalattice.control.task_control.contracts import TaskLifecycle, TaskRecord
from alphalattice.control.task_control.registry import DuckDbTaskControlRegistry
from alphalattice.evidence.alternative_evidence.publication.contracts import (
    AlternativeEvidenceAnalysisPublication,
)
from alphalattice.foundation.factor_research.experiments.authoring import FACTOR_EXPERIMENT_KIND
from alphalattice.investment.alpha_research.experiments.authoring import ALPHA_EXPERIMENT_KIND
from alphalattice.investment.portfolio_strategy_lab.application.research_experiment import (
    KIND as PORTFOLIO_EXPERIMENT_KIND,
)
from alphalattice.investment.risk_research.experiments.compiler import RISK_EXPERIMENT_KIND
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.oversight.chief_risk_officer.decision.book_evidence import review_book_key
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    PortfolioReviewPublication,
)

ACKNOWLEDGEMENT_PATH = Path("runtime") / "upgrade-overview.json"
_ACKNOWLEDGEMENT_SCHEMA = "upgrade-overview-acknowledgement"
_ACKNOWLEDGEMENT_VERSION = 1
_STUDY_KINDS = (
    FACTOR_EXPERIMENT_KIND,
    ALPHA_EXPERIMENT_KIND,
    RISK_EXPERIMENT_KIND,
    PORTFOLIO_EXPERIMENT_KIND,
)


def installed_study_identities() -> dict[str, str]:
    """Each study kind's installed implementation, as its REPLAY compares it.

    Computed from the source tree (about 0.6 s for the four kinds), so a Host asks once:
    the code it serves is the code it imported.
    """
    return {f"study:{kind}": _implementation_hash(kind) for kind in _STUDY_KINDS}


def upgrade_overview(
    *,
    workspace: Path,
    registry: DuckDbTaskControlRegistry,
    review: EvidenceReviewApplication | None,
    resume_refusal: Callable[[TaskRecord], str | None],
    command_running: Callable[[UUID], bool],
    tasks: Sequence[TaskRecord] | None = None,
    task_refusals: Sequence[dict[str, Any]] = (),
    study_identities: Mapping[str, str] | None = None,
    failed_studies: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Every saved study, published review and waiting Task, with its standing now.

    `tasks` is the request's one reading of the registry when the caller took it (V119), and
    `study_identities` what `installed_study_identities` answered for this Host.
    """
    identities = dict(
        installed_study_identities() if study_identities is None else study_identities
    )
    analyses = None
    if review is not None and review.evidence_task_adapter is not None:
        analyses = review.evidence_task_adapter.runtime.publications
        identities["evidence_bindings"] = str(
            canonical_hash({"bindings": list(analyses.expected_bindings)})
        )
    if tasks is None:
        batch = registry.record_collection()
        tasks = batch.records
        task_refusals = [task_record_refusal(task_id) for task_id in batch.refused_task_ids]
    failed = dict(failed_studies or {})
    studies = [_study(task, identities, failed) for task in tasks if _is_study(task)]
    reviews = [] if review is None else _reviews(review, analyses)
    waiting = [
        _waiting(task, resume_refusal(task))
        for task in tasks
        if task.lifecycle in {TaskLifecycle.RECOVERY_REQUIRED, TaskLifecycle.QUEUED}
        and not command_running(task.task_id)
    ]
    set_hash = str(canonical_hash(identities))
    acknowledged, unreadable = _acknowledged(workspace)
    settled = {"CURRENT", "RESUMABLE"}
    touched = any(
        row["state"] not in settled for rows in (studies, reviews, waiting) for row in rows
    )
    return {
        "status": "UPGRADE_OVERVIEW",
        "installed": {"set_hash": set_hash, "identities": identities},
        "acknowledged": acknowledged,
        **({"acknowledgement_unreadable": unreadable} if unreadable else {}),
        "moved": sorted(
            name
            for name, value in identities.items()
            if acknowledged is not None and acknowledged["identities"].get(name) != value
        ),
        "show": touched and (acknowledged is None or acknowledged["set_hash"] != set_hash),
        "counts": {
            name: dict(Counter(str(row["state"]) for row in rows))
            for name, rows in (("studies", studies), ("reviews", reviews), ("tasks", waiting))
        },
        **({"refusals": list(task_refusals)} if task_refusals else {}),
        "studies": studies,
        "reviews": reviews,
        "tasks": waiting,
        "numerical_call_count": 0,
        "next_requests": {
            **(
                {}
                if task_refusals
                else {
                    "acknowledge": {
                        "operation": "UPGRADE_ACKNOWLEDGE",
                        "upgrade_set_hash": set_hash,
                    }
                }
            )
        },
    }


def acknowledge_upgrade(
    workspace: Path, overview: dict[str, object], *, confirmed: str, now: datetime
) -> dict[str, object]:
    """Record the installed identities the user saw; a page that moved meanwhile is refused."""
    installed = overview["installed"]
    assert isinstance(installed, dict)
    if confirmed != installed["set_hash"]:
        return {
            "status": "REFUSED",
            "failure_code": "upgrade_overview.confirmation_stale",
            "upgrade_set_hash": installed["set_hash"],
            "detail": (
                "The upgrade this acknowledgement names is not the one installed now; read the "
                "overview again and acknowledge what it shows."
            ),
            "next_requests": {"overview": {"operation": "UPGRADE_OVERVIEW"}},
        }
    record = {
        "schema": _ACKNOWLEDGEMENT_SCHEMA,
        "version": _ACKNOWLEDGEMENT_VERSION,
        "set_hash": installed["set_hash"],
        "identities": installed["identities"],
        "acknowledged_at": now.isoformat(),
    }
    target = workspace.resolve() / ACKNOWLEDGEMENT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, prefix=".upgrade-overview-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(record, sort_keys=True, indent=1).encode("utf-8") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {**overview, "acknowledged": record, "moved": [], "show": False}


def _is_study(task: TaskRecord) -> bool:
    return task.task_kind == TASK_KIND and task.lifecycle is TaskLifecycle.SUCCEEDED


def _study(
    task: TaskRecord, identities: dict[str, str], failed: Mapping[str, str]
) -> dict[str, object]:
    plan = ExperimentPlan.model_validate(task.input.payload["plan"])
    kind = plan.program.kind
    installed = identities.get(f"study:{kind}") or _implementation_hash(kind)
    # The same comparison a replay makes first (binding plan, V28): the plan's role, the
    # recorded moves consulted.
    current = is_current(plan_implementation_role(plan), plan.implementation_hash, installed)
    calls = plan.execution_preview.get("expected_numerical_calls")
    ids = {"task_id": str(task.task_id)}
    code = failed.get(str(task.task_id))
    if code is not None:
        # The sweep could not verify its sealed evidence (V89): named here, refused when read.
        return {
            **ids,
            "kind": kind,
            "state": "INTEGRITY_FAILED",
            "failure_code": code,
            "detail": "Its sealed evidence did not verify when the workspace's studies were last "
            "checked in full. It refuses when read; a continuation runs its declared work again.",
            "next_requests": {
                "readback": {"operation": "EXPERIMENT_READBACK", **ids},
                "draft": {"operation": "EXPERIMENT_DRAFT", **ids},
            },
        }
    return {
        **ids,
        "kind": kind,
        "state": "CURRENT" if current else "CHANGED",
        "recorded_implementation_hash": plan.implementation_hash,
        "installed_implementation_hash": installed,
        "refresh_numerical_calls": calls,
        "detail": (
            "Its kind's implementation is the one it ran under; a replay reuses it exactly."
            if current
            else f"The {kind_words(kind)} implementation changed since this study ran. It reads "
            "back as recorded; a replay refuses, and a continuation runs its declared work again "
            f"under the installed code ({calls} numerical calls declared)."
        ),
        "next_requests": {
            "readback": {"operation": "EXPERIMENT_READBACK", **ids},
            **({} if current else {"draft": {"operation": "EXPERIMENT_DRAFT", **ids}}),
        },
    }


def _reviews(review: EvidenceReviewApplication, analyses: Any) -> list[dict[str, object]]:
    """Each published review, read once (V119); one that does not read back is named in its
    row, never in the way of the others."""

    service = review.review_publications
    rows: list[dict[str, object]] = []
    for handle in service.publication_hashes():
        request: dict[str, object] = {
            "export": {"operation": "EVIDENCE_CRO_EXPORT", "review_publication_hash": handle}
        }
        try:
            view = service.read(handle)
        except (OSError, ValueError) as error:
            recorded = _publication_if_readable(service.store, handle)
            rows.append(
                _review_row(handle, recorded, "UNREADABLE", str(error), {}, None, request, None)
            )
            continue
        publication = view.publication
        # The analyses the review rests on, as its export replays them.
        contracts: dict[str, str] = {}
        for identity in sorted(set(view.dossier.evidence_publication_hashes)):
            if analyses is None:
                contracts[identity] = "NOT_ADMITTED"
                continue
            try:
                record = analyses.store.load(
                    "analysis-publications", identity, AlternativeEvidenceAnalysisPublication
                )
            except (OSError, ValueError):
                contracts[identity] = "UNREADABLE"
                continue
            contracts[identity] = analyses.binding_contract(record) or "UNLISTED"
        refused = sorted(k for k, v in contracts.items() if v in {"UNLISTED", "UNREADABLE"})
        earlier = sorted(k for k, v in contracts.items() if v.startswith("HISTORICAL:"))
        if refused:
            state = "UNREADABLE"
            detail = (
                "Its analyses do not verify under any Evidence binding this build reads "
                f"({len(refused)} of {len(contracts)}), so the review does not read back here."
            )
        elif earlier or not view.under_installed_policy:
            state = "CHANGED"
            detail = " ".join(
                part
                for part in (
                    "Its analyses were sealed under an earlier Evidence binding "
                    f"({len(earlier)} of {len(contracts)})."
                    if earlier
                    else "",
                    "It was sealed under an earlier CRO review policy."
                    if not view.under_installed_policy
                    else "",
                    "It reads back as recorded; a current review prepares evidence again and "
                    "asks for a new review.",
                )
                if part
            )
        else:
            state = "CURRENT"
            detail = "Its evidence and its recommendation are what this build gives."
        rows.append(
            _review_row(
                handle,
                publication,
                state,
                detail,
                contracts,
                view.under_installed_policy,
                request,
                view.person_action,
            )
        )
    return rows


def _publication_if_readable(store: Any, handle: str) -> PortfolioReviewPublication | None:
    """The publication record alone, for a review whose lineage does not read back."""

    try:
        record: PortfolioReviewPublication = store.load(
            "cro-review-publications", handle, PortfolioReviewPublication
        )
    except (OSError, ValueError):
        return None
    return record


def _review_row(
    handle: str,
    publication: PortfolioReviewPublication | None,
    state: str,
    detail: str,
    contracts: dict[str, str],
    under_installed_policy: bool | None,
    next_requests: dict[str, object],
    person_action: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "review_publication_hash": handle,
        "review_key": None if publication is None else publication.review_key,
        "book_key": None if publication is None else review_book_key(publication),
        "published_at": None if publication is None else publication.published_at.isoformat(),
        "person_action": person_action,
        "state": state,
        "evidence_contracts": contracts,
        "review_under_installed_policy": under_installed_policy,
        "detail": detail,
        "next_requests": next_requests,
    }


def _waiting(task: TaskRecord, refusal: str | None) -> dict[str, object]:
    ids = {"task_id": str(task.task_id)}
    return {
        **ids,
        "task_kind": task.task_kind,
        "lifecycle": task.lifecycle.value,
        "state": "RESUMABLE" if refusal is None else "CHANGED_SINCE_ADMISSION",
        "resume_refusal": refusal,
        "next_requests": {
            "recovery": {"operation": "TASK_RECOVERY", **ids},
            **(
                {}
                if refusal is None
                else {
                    "cancel": {
                        "operation": "CANCEL",
                        **ids,
                        "expected_task_hash": task.record_hash,
                    }
                }
            ),
        },
    }


def _acknowledged(workspace: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = workspace.resolve() / ACKNOWLEDGEMENT_PATH
    if not path.is_file():
        return None, None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "upgrade_overview.acknowledgement_unreadable"
    if (
        not isinstance(record, dict)
        or record.get("schema") != _ACKNOWLEDGEMENT_SCHEMA
        or record.get("version") != _ACKNOWLEDGEMENT_VERSION
        or not isinstance(record.get("identities"), dict)
    ):
        return None, "upgrade_overview.acknowledgement_from_another_build"
    return record, None
