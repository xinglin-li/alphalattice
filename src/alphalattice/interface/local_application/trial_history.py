"""One research trial's events, recorded beside what the workflow writes today.

A sealed run is a trial: its selection (the universe, the snapshot and the session window up to
the decision cutoff), its recipe (the Desk kind and the sealed method and parameter identities),
its registration before the Desk runs, its start, its execution evidence as its artifact and its
end. Each is a small event through the Host's history; nothing here reads or writes a result.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from alphalattice.control.research_program.authoring.dispatcher import SealedSubmission
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.data_layer.chain import canonical, digest
from alphalattice.kernel.data_layer.log import uuid7
from alphalattice.kernel.data_layer.project import recipe_key
from alphalattice.kernel.shared_kernel.environment import package_versions
from alphalattice.protocols.actor_execution.contracts import ActorKind
from alphalattice.protocols.research_authoring.contracts import ResearchExecutionEvidence

_ACTORS = {ActorKind.HUMAN: "person", ActorKind.HOST_FALLBACK: "host"}


@dataclass(frozen=True)
class Trial:
    """A registered trial: its id, its span, its actor and when it started."""

    trial_id: str
    span: str
    actor: tuple[str, str]
    started_us: int


def started(sealed: SealedSubmission) -> Trial | None:
    """Record the trial's selection, recipe, registration and start; None outside a Host."""
    envelope, program, binding = sealed.envelope, sealed.program, sealed.binding
    actor = (_ACTORS.get(binding.actor_kind, "agent"), binding.actor_id)
    cutoff = envelope.sessions.as_of
    window = {
        "universe": envelope.universe_handle,
        "valid_from": envelope.sessions.start.isoformat(),
        "valid_to": envelope.sessions.end.isoformat(),
        "spec": {
            "snapshot": envelope.data_snapshot_handle,
            "as_of": cutoff.model_dump(mode="json"),
        },
    }
    selection = digest(canonical(window))
    if not history.recording():
        return None
    # Nothing known after the cutoff session's end, UTC; the phase is in the spec.
    end = datetime.combine(cutoff.session + timedelta(days=1), datetime.min.time(), UTC)
    attrs = window | {"cutoff_us": int(end.timestamp()) * 1_000_000}
    exists = "SELECT 1 FROM selection WHERE selection_id = ?"
    history.record_new(
        exists, (selection,), "selection.made", ("selection", selection), attrs, actor=actor
    )
    params = {
        "method_binding": program.method_binding_hash,
        "parameter_domain": program.parameter_domain_hash,
        "desk_program": program.desk_program_hash,
    }
    history.define("method", program.kind, f"The {program.kind} Desk's sealed research program.")
    recipe = recipe_key(program.kind, 1, params)
    defined = {"method": program.kind, "method_version": 1, "params": params}
    history.record("recipe.defined", ("recipe", recipe), defined, actor=actor)
    trial = uuid7()
    registered = {
        "selection_id": selection,
        "recipe_key": recipe,
        "seeds": [envelope.determinism.seed],
        "product_version": _product_version(),
        "context": {"program": program.program_hash, "sessions": len(program.resolved_sessions)},
    }
    cause = history.record("trial.registered", ("trial", trial), registered, actor=actor)
    span = secrets.token_hex(8)
    now = time.time_ns() // 1000
    history.record("trial", ("trial", trial), phase="started", span=span, cause=cause, actor=actor)
    return Trial(trial, span, actor, now)


def ended(trial: object, evidence: ResearchExecutionEvidence | None) -> None:
    """Record the trial's end: completed with its evidence as its artifact, or failed."""
    if not isinstance(trial, Trial):  # no Host: `started` recorded nothing
        return
    subject = ("trial", trial.trial_id)
    ending: dict[str, Any] = {
        "span": trial.span,
        "actor": trial.actor,
        "started_us": trial.started_us,
    }
    if evidence is None:
        history.record("trial", subject, phase="failed", **ending)
        return
    artifact = evidence.evidence_hash
    headline = None
    exists = "SELECT 1 FROM artifact WHERE artifact_id = ?"
    produced = {
        "type": history.EVIDENCE_TYPE,
        "identity_kind": "content",
        "storage": "none",  # its body is the workspace's evidence store's, not the data root's
        "meaning": f"{evidence.kind} evidence of program {evidence.program_hash[:12]}",
        "trial_id": trial.trial_id,
        "content_digest": artifact,
        "row_count": len(evidence.artifact_uris),
    }
    subject_artifact = ("artifact", artifact)
    # A reused run's evidence stays its first trial's: this trial names it only when it produced it.
    new = history.record_new(
        exists, (artifact,), "artifact.produced", subject_artifact, produced, actor=trial.actor
    )
    headline = artifact if new is not None else None
    completed: dict[str, Any] = {"disposition": evidence.disposition}
    completed |= {"numerical_calls": evidence.numerical_call_count}
    if headline is not None:
        completed["result_artifact"] = headline
    history.record("trial", subject, completed, phase="completed", **ending)


def _product_version() -> str:
    ((_, version),) = package_versions(("alphalattice",))
    return str(version)
