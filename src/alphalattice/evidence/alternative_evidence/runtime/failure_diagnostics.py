"""Bounded unit failure diagnostics for the Evidence Task adapter."""

from collections.abc import Callable

from pydantic import ValidationError

from alphalattice.control.task_control.contracts import (
    FAILURE_CODE_MAX_LENGTH,
    StageFailureCause,
    failure_code_from,
)
from alphalattice.kernel.data_layer import recall

from ..documents.contracts import AlternativeEvidenceDocumentSet


def failure_cause(
    error: Exception, stage: str, documents: Callable[[], AlternativeEvidenceDocumentSet]
) -> StageFailureCause:
    """Fit the observed exception and any known citation context to the existing receipt."""
    cause = StageFailureCause.from_facts(
        {"exception_type": type(error).__name__, "detail": failure_detail(error, stage, documents)}
    )
    assert cause is not None
    return cause


def failure_detail(
    error: Exception, stage: str, documents: Callable[[], AlternativeEvidenceDocumentSet]
) -> str:
    """Name a citation failure's actual filing without changing selection or source bytes."""
    if (
        isinstance(error, ValidationError)
        and stage == "select_evidence_spans"
        and (
            error.title == "EvidenceNeedRecord"
            or (
                error.title == "LitigationMatterRecord"
                and failure_code_from(error) == "alternative_evidence.evidence_need_invalid"
            )
        )
    ):
        return _need_failure_detail(error, documents)
    if stage != "build_retrieval_generation":
        return str(error)
    issuer, filing = "UNKNOWN", "UNKNOWN"
    frame = error.__traceback__
    while frame is not None:
        # The store's build takes each filing in turn (`recall.index_references`): the one it
        # held when it failed is named.
        if frame.tb_frame.f_code is recall.index_references.__code__:
            document = frame.tb_frame.f_locals.get("document")
            revision = frame.tb_frame.f_locals.get("revision")
            try:
                matches = [
                    item
                    for item in documents().documents
                    if item.workspace_document_id == document
                    and item.workspace_revision == revision
                ]
            except Exception:
                matches = []  # Preserve the original failure when diagnostics are unavailable.
            if len(matches) == 1:
                ref = matches[0]
                issuer = ref.entity_id
                filing = f"{ref.semantic_handle} ({ref.document_type}, {ref.revision_label})"
            break
        frame = frame.tb_next
    code = getattr(getattr(error, "failure", None), "code", None) or getattr(error, "code", None)
    return f"Issuer {issuer}; filing {filing}; refused {code or type(error).__name__}. {error}"


def _need_failure_detail(
    error: ValidationError, documents: Callable[[], AlternativeEvidenceDocumentSet]
) -> str:
    """Name the failing need and the invariant recorded by its validator."""
    errors = error.errors(include_url=False)
    context = errors[0].get("ctx", {}) if errors else {}
    cause = context.get("error")
    invariant = getattr(cause, "invariant", "UNKNOWN")
    need = getattr(cause, "need", None)
    value = need.model_dump() if need is not None else errors[0].get("input") if errors else None
    facts = value if isinstance(value, dict) else {}
    handle = facts.get("document_handle")
    issuer, filing = "UNKNOWN", str(handle) if isinstance(handle, str) else "UNKNOWN"
    try:
        matches = [item for item in documents().documents if item.semantic_handle == handle]
    except Exception:
        matches = []
    if len(matches) == 1:
        ref = matches[0]
        issuer = ref.entity_id
        filing = f"{ref.semantic_handle} ({ref.document_type}, {ref.revision_label})"
    names = (
        "kind",
        "status",
        "span_handle",
        "matter_handle",
        "character_start",
        "character_end",
        "target_state",
    )
    fields = ", ".join(f"{name}={str(facts.get(name))[:48]}" for name in names)
    return (
        f"Issuer {issuer}; filing {filing}; rejected field {error.title} (model-level): "
        f"invariant={invariant}; {fields}. {error}"
    )


def failure_code(error: Exception) -> str:
    """The code a stage records for what stopped it, naming the owner's own reason.

    A Workspace knowledge failure (an index that fails its check, a snapshot
    that moved, a semantic pack that is not the pinned one) carries its code
    on ``failure``; a contract refusal raised through pydantic carries its
    code as the first message of a validation report. Both used to be
    recorded as the bare ``task_failed``, which told the person who has to
    repair the workspace nothing about what to repair.
    """
    code = getattr(getattr(error, "failure", None), "code", None)
    if not (isinstance(code, str) and code):
        # A Host owner's refusal (the storage budget's, for one) names its
        # reason on `failure_code`; the Task records that reason, not the
        # bare fact that a stage failed.
        code = getattr(error, "failure_code", None)
    if isinstance(code, str) and code:
        return f"alternative_evidence.task_failed:{code}"[:FAILURE_CODE_MAX_LENGTH]
    value = failure_code_from(error)
    return (
        value if value.startswith("alternative_evidence.") else "alternative_evidence.task_failed"
    )
