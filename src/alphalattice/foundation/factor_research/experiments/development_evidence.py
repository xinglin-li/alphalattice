"""Write and read one sealed Factor development evidence record.

The writer was inline in ``execution.py`` with a comment observing that nothing
consumed what it wrote, so a reader would have been speculative. Alpha consumes
it now, and that comment is out of date: a development Alpha experiment is
authorized by a specific Factor development checkpoint, and it has to be able to
load exactly that one.

Reader and writer live together because they are one format decision. Split
across two modules they drift, and the failure mode is silent -- a reader that
resolves a slightly different path finds nothing and reports "no evidence"
rather than "wrong root".

This is not an artifact store. The record arrives sealed and self-identifying,
so what remains is writing those bytes under their own name and reading them
back by that name. A store would add a schema registry, a category index and a
lifecycle that nothing here needs.

Deliberately **not** importable from ``factor_research.publication``: development
evidence is not a published Factor result and must never be reachable from the
owner that moves a current pointer.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.factor_research.experiments.authoring import (
    FACTOR_EXPERIMENT_KIND,
    FactorInventoryEntry,
    factor_desk_program_hash,
)
from alphalattice.foundation.factor_research.programs.program import (
    FactorResearchDeterministicEvidence,
)
from alphalattice.foundation.factor_research.research_loop.contracts import (
    FactorResearchReviewDecisionReceipt,
)
from alphalattice.foundation.factor_research.research_loop.decisions import (
    verify_factor_research_curation,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExecutionEvidence,
    SealedResearchProgram,
)

FACTOR_DEVELOPMENT_EVIDENCE_CATEGORY = "development/factor-deterministic-evidence"
"""Namespaced under ``development``, so no reader walking published Factor
categories can encounter one of these."""

FACTOR_DEVELOPMENT_RECEIPT_CATEGORY = "development/factor-development-receipt"
"""The lineage record beside the deterministic child, in its own category."""

FACTOR_DEVELOPMENT_CURATION_CATEGORY = "development/factor-curation-receipt"
"""Host-sealed curation decisions about one deterministic checkpoint.

Filed under the checkpoint they judge rather than content-addressed at the top
level, because there is legitimately more than one: a Human, an Installed Agent
and an External Automation submitting the same domain content produce one
submission identity and three distinct actor bindings, and all three belong in
the graph. Grouping by checkpoint is what lets a verifier ask "was this evidence
curated, and by whom" without listing an unbounded directory.
"""

_URI_PREFIX = f"playpen://factor-research/{FACTOR_DEVELOPMENT_EVIDENCE_CATEGORY}/"
_RECEIPT_PREFIX = f"playpen://factor-research/{FACTOR_DEVELOPMENT_RECEIPT_CATEGORY}/"
_CURATION_PREFIX = f"playpen://factor-research/{FACTOR_DEVELOPMENT_CURATION_CATEGORY}/"


class FactorDevelopmentFactorIdentity(BaseModel):  # type: ignore[misc]
    """One context factor, in the Panel's published order, with its identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["FactorDevelopmentFactorIdentity"] = "FactorDevelopmentFactorIdentity"
    factor_id: str = Field(min_length=1, max_length=128)
    implementation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    methodology_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class FactorDevelopmentInputBinding(BaseModel):  # type: ignore[misc]
    """Exactly what one Factor development run was allowed to read, typed.

    This was an inline ``canonical_hash`` over an anonymous dict. The hash was
    real and the content was unrecoverable: a verifier could compare the value
    with another copy of the same value and could not ask what it described, so
    every policy it covered was unauditable and every one it omitted was
    invisible.

    Sealed as a contract instead, and its ``binding_hash`` is what the executor
    reports as ``desk_input_binding_hash`` -- so the generic layer seals this
    object's identity rather than one nobody can open.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["FactorDevelopmentInputBinding"] = "FactorDevelopmentInputBinding"
    selected_factor_ids: tuple[str, ...] = Field(min_length=1)
    context_factors: tuple[FactorDevelopmentFactorIdentity, ...] = Field(min_length=1)
    """The whole axis in published order, each entry carrying its own identity.

    Ordered rather than keyed: the Panel's factor order is authority in its own
    right, and a mapping rebuilt and re-sorted at a boundary silently discards it.
    """

    feature_panel_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_panel_manifest_ref: str = Field(min_length=1)
    causal_outcome_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    causal_outcome_manifest_ref: str = Field(min_length=1)
    target_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    walk_forward_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    screening_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The evidence policy: the multiple-testing correction and its thresholds."""

    redundancy_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    authority_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def context_factor_ids(self) -> tuple[str, ...]:
        return tuple(value.factor_id for value in self.context_factors)

    @classmethod
    def create(
        cls,
        *,
        selected_factor_ids: tuple[str, ...],
        context_factors: tuple[FactorDevelopmentFactorIdentity, ...],
        feature_panel_snapshot_hash: str,
        feature_panel_manifest_ref: str,
        causal_outcome_snapshot_hash: str,
        causal_outcome_manifest_ref: str,
        target_policy_hash: str,
        walk_forward_policy_hash: str,
        screening_policy_hash: str,
        redundancy_policy_hash: str,
        authority_hash: str,
    ) -> Self:
        values = {
            "kind": "FactorDevelopmentInputBinding",
            "selected_factor_ids": list(selected_factor_ids),
            "context_factors": [value.model_dump(mode="json") for value in context_factors],
            "feature_panel_snapshot_hash": feature_panel_snapshot_hash,
            "feature_panel_manifest_ref": feature_panel_manifest_ref,
            "causal_outcome_snapshot_hash": causal_outcome_snapshot_hash,
            "causal_outcome_manifest_ref": causal_outcome_manifest_ref,
            "target_policy_hash": target_policy_hash,
            "walk_forward_policy_hash": walk_forward_policy_hash,
            "screening_policy_hash": screening_policy_hash,
            "redundancy_policy_hash": redundancy_policy_hash,
            "authority_hash": authority_hash,
        }
        return cls(
            selected_factor_ids=tuple(selected_factor_ids),
            context_factors=tuple(context_factors),
            feature_panel_snapshot_hash=feature_panel_snapshot_hash,
            feature_panel_manifest_ref=feature_panel_manifest_ref,
            causal_outcome_snapshot_hash=causal_outcome_snapshot_hash,
            causal_outcome_manifest_ref=causal_outcome_manifest_ref,
            target_policy_hash=target_policy_hash,
            walk_forward_policy_hash=walk_forward_policy_hash,
            screening_policy_hash=screening_policy_hash,
            redundancy_policy_hash=redundancy_policy_hash,
            authority_hash=authority_hash,
            binding_hash=str(canonical_hash(values)),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        context = self.context_factor_ids
        if len(set(context)) != len(context):
            raise ValueError("factor_research.development_input_binding_context_duplicated")
        if not set(self.selected_factor_ids).issubset(set(context)):
            raise ValueError("factor_research.development_input_binding_selection_not_in_context")
        if self.binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"binding_hash"})
        ):
            raise ValueError("factor_research.development_input_binding_invalid")
        return self


class FactorDevelopmentReceipt(BaseModel):  # type: ignore[misc]
    """What authored Program produced one deterministic checkpoint, and over what.

    The child ``FactorResearchDeterministicEvidence`` is self-identifying but
    says nothing about the generic authoring run that caused it: a consumer
    holding one could not tell which Program admitted it, which method binding it
    was sealed under, or which factors the document actually selected. Alpha
    consumed exactly that child, so "authorized by a Factor checkpoint" meant
    only "a Factor checkpoint exists".

    Both axes are carried, and they answer different questions.
    ``context_factor_ids`` is the statistical universe the deterministic program
    ran over -- the FDR denominator. ``selected_factor_ids`` is what this document
    asked about and the only axis a downstream Desk may consume. Reporting one
    under the other's name is the misreading this contract exists to prevent.

    The **admitted Program** is embedded whole rather than reduced to three
    copied hashes. Copied hashes can only ever be compared with another copy of
    themselves; the sealed Program was compiled at freeze time, before anything
    executed, and it binds the selected axis into its own ``desk_program_hash``.
    So carrying it makes a resealed receipt with a rewritten selection
    contradictable by an artifact that predates the run -- which is the whole
    difference between self-consistency and authority.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["FactorDevelopmentReceipt"] = "FactorDevelopmentReceipt"
    program: SealedResearchProgram
    input_binding: FactorDevelopmentInputBinding
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    # Read-through rather than restated. Every one of these used to be a field of
    # its own beside the object that already held it, which is two places for one
    # fact and therefore a place for them to disagree.
    @property
    def program_hash(self) -> str:
        return self.program.program_hash

    @property
    def desk_program_hash(self) -> str:
        return self.program.desk_program_hash

    @property
    def method_binding_hash(self) -> str:
        return self.program.method_binding_hash

    @property
    def desk_input_binding_hash(self) -> str:
        return self.input_binding.binding_hash

    @property
    def selected_factor_ids(self) -> tuple[str, ...]:
        return self.input_binding.selected_factor_ids

    @property
    def context_factor_ids(self) -> tuple[str, ...]:
        return self.input_binding.context_factor_ids

    @property
    def feature_panel_snapshot_hash(self) -> str:
        return self.input_binding.feature_panel_snapshot_hash

    @property
    def causal_outcome_snapshot_hash(self) -> str:
        return self.input_binding.causal_outcome_snapshot_hash

    @classmethod
    def create(
        cls,
        *,
        program: SealedResearchProgram,
        input_binding: FactorDevelopmentInputBinding,
        checkpoint_hash: str,
    ) -> Self:
        values = {
            "kind": "FactorDevelopmentReceipt",
            "program": program.model_dump(mode="json"),
            "input_binding": input_binding.model_dump(mode="json"),
            "checkpoint_hash": checkpoint_hash,
        }
        return cls(
            program=program,
            input_binding=input_binding,
            checkpoint_hash=checkpoint_hash,
            receipt_hash=str(canonical_hash(values)),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if self.selected_factor_ids != tuple(dict.fromkeys(self.selected_factor_ids)):
            raise ValueError("factor_research.development_receipt_selection_duplicated")
        if self.program.kind != FACTOR_EXPERIMENT_KIND:
            raise ValueError("factor_research.development_receipt_program_kind_invalid")
        if self.receipt_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        ):
            raise ValueError("factor_research.development_receipt_identity_invalid")
        return self


def factor_development_receipt_uri(receipt_hash: str) -> str:
    return f"{_RECEIPT_PREFIX}{receipt_hash}"


def factor_development_receipt_handle(uri: str) -> str:
    """Recover the receipt hash, admitted as exactly 64 lowercase hex."""

    handle = uri[len(_RECEIPT_PREFIX) :] if uri.startswith(_RECEIPT_PREFIX) else uri
    if len(handle) != 64 or not set(handle).issubset(_HASH_CHARACTERS):
        raise AuthoringError("factor_research.development_receipt_handle_invalid")
    return handle


def factor_development_evidence_uri(checkpoint_hash: str) -> str:
    """The one URI spelling, so writer and reader cannot disagree about it."""

    return f"{_URI_PREFIX}{checkpoint_hash}"


_HASH_CHARACTERS = frozenset("0123456789abcdef")


def factor_development_evidence_handle(uri: str) -> str:
    """Recover the checkpoint hash an authored document may name as a handle.

    Accepts either the full URI or the bare hash, because a researcher copying a
    value out of a run report has both in front of them and neither is wrong.

    Admitted as **exactly** 64 lowercase hex and nothing else. Rejecting only
    path separators left every other malformed value to fail somewhere further
    in -- as a missing file, or worse as a lookup that happened to succeed -- and
    a content address that is not a content address should fail at the boundary
    that owns the spelling.
    """

    handle = uri[len(_URI_PREFIX) :] if uri.startswith(_URI_PREFIX) else uri
    if len(handle) != 64 or not set(handle).issubset(_HASH_CHARACTERS):
        raise AuthoringError("factor_research.development_evidence_handle_invalid")
    return handle


def _publish_json(root: Path, payload: Mapping[str, Any], identity: str, *, conflict: str) -> None:
    """The shared byte codec and write-once mechanic; callers own admission."""
    root.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    target = root / f"{identity}.json"
    if target.exists():
        if target.read_bytes() != content:
            raise AuthoringError(conflict)
        return
    staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    staged.write_bytes(content)
    os.replace(staged, target)


def publish_factor_development_evidence(
    output_workspace: Path, payload: Mapping[str, Any], identity: str
) -> str:
    """Write one sealed evidence record, content-addressed, exactly once."""
    _publish_json(
        Path(output_workspace) / FACTOR_DEVELOPMENT_EVIDENCE_CATEGORY,
        payload,
        identity,
        conflict="factor_research.development_evidence_identity_conflict",
    )
    return factor_development_evidence_uri(identity)


class FactorDevelopmentEvidenceReader:
    """Load one Factor development checkpoint by its exact hash, read-only.

    The root is constructor state supplied by the Host, never taken from an
    authored document: a document that could name its own evidence root could
    name any directory on the machine and have the Host read it as authority.
    The document names the *handle*; the Host decides where handles resolve.

    There is deliberately no "latest", no listing and no search. Selecting the
    newest file in a directory is how a run silently binds to evidence nobody
    chose -- it succeeds, it is reproducible on the machine that has that
    directory, and it means something different everywhere else.
    """

    def __init__(self, evidence_root: Path) -> None:
        self._root = Path(evidence_root).resolve() / FACTOR_DEVELOPMENT_EVIDENCE_CATEGORY

    @property
    def root(self) -> Path:
        return self._root

    def load(self, handle: str) -> FactorResearchDeterministicEvidence:
        """Read exactly the named checkpoint and prove it is the one asked for."""

        checkpoint = factor_development_evidence_handle(handle)
        path = self._root / f"{checkpoint}.json"
        if not path.is_file():
            raise AuthoringError("factor_research.development_evidence_unavailable")
        try:
            evidence = FactorResearchDeterministicEvidence.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except ValueError as error:
            raise AuthoringError("factor_research.development_evidence_invalid") from error
        if evidence.checkpoint_hash != checkpoint:
            # The contract revalidates its own identity on parse, so this catches
            # the remaining case: a well-formed record stored under someone
            # else's name.
            raise AuthoringError("factor_research.development_evidence_identity_conflict")
        return evidence


def publish_factor_development_receipt(
    output_workspace: Path, receipt: FactorDevelopmentReceipt
) -> str:
    """Write one lineage receipt beside its child, content-addressed, once."""

    _publish_json(
        Path(output_workspace) / FACTOR_DEVELOPMENT_RECEIPT_CATEGORY,
        receipt.model_dump(mode="json"),
        receipt.receipt_hash,
        conflict="factor_research.development_receipt_identity_conflict",
    )
    return factor_development_receipt_uri(receipt.receipt_hash)


class FactorDevelopmentReceiptReader:
    """Load one lineage receipt and the child it names, by exact hash.

    Reads; it does not adjudicate. Whether a receipt's copied identities match
    the artifacts they claim to describe is a question about *other* authorities
    -- the sealed Program, the Panel manifest -- and answering it here would mean
    this reader acquiring opinions about both. The Host re-derives, in
    ``verify_factor_development_receipt``.
    """

    def __init__(self, evidence_root: Path) -> None:
        root = Path(evidence_root).resolve()
        self._receipts = root / FACTOR_DEVELOPMENT_RECEIPT_CATEGORY
        self._children = FactorDevelopmentEvidenceReader(root)

    @property
    def root(self) -> Path:
        return self._receipts

    def load(
        self, handle: str
    ) -> tuple[FactorDevelopmentReceipt, FactorResearchDeterministicEvidence]:
        receipt_hash = factor_development_receipt_handle(handle)
        path = self._receipts / f"{receipt_hash}.json"
        if not path.is_file():
            raise AuthoringError("factor_research.development_receipt_unavailable")
        try:
            receipt = FactorDevelopmentReceipt.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise AuthoringError("factor_research.development_receipt_invalid") from error
        if receipt.receipt_hash != receipt_hash:
            raise AuthoringError("factor_research.development_receipt_identity_conflict")
        child = self._children.load(receipt.checkpoint_hash)
        return receipt, child


def factor_development_curation_uri(*, checkpoint_hash: str, receipt_hash: str) -> str:
    """The one URI spelling for a curation decision about one checkpoint."""

    return f"{_CURATION_PREFIX}{checkpoint_hash}/{receipt_hash}"


def publish_factor_development_curation(
    output_workspace: Path,
    *,
    checkpoint: FactorResearchDeterministicEvidence,
    review_binding_hash: str,
    receipt: FactorResearchReviewDecisionReceipt,
) -> str:
    """Write one Host-sealed curation decision beside the evidence it judges.

    The checkpoint is required, not its hash, and that is the whole difference
    from the previous signature. Given only a hash this was a caller-trusted
    writer: any self-consistent receipt naming any checkpoint was written, so a
    decision that had never passed Host admission became durable evidence simply
    by being well formed. It now re-derives the decision from the checkpoint
    through the same owner the evidence verifier uses, so a receipt is persisted
    exactly when it would also replay.

    Still not a second sealer. It admits nothing this module decides; it refuses
    to write what the decision owner will not vouch for.
    """

    verify_factor_research_curation(
        decision=receipt,
        checkpoint=checkpoint,
        review_binding_hash=review_binding_hash,
    )
    checkpoint_handle = factor_development_evidence_handle(checkpoint.checkpoint_hash)
    root = Path(output_workspace) / FACTOR_DEVELOPMENT_CURATION_CATEGORY / checkpoint_handle
    _publish_json(
        root,
        receipt.model_dump(mode="json"),
        receipt.receipt_hash,
        conflict="factor_research.development_curation_identity_conflict",
    )
    return factor_development_curation_uri(
        checkpoint_hash=checkpoint_handle, receipt_hash=receipt.receipt_hash
    )


class FactorDevelopmentCurationReader:
    """Every curation decision filed against one checkpoint, in receipt order.

    Listing is admissible here and nowhere else in this module, and the
    difference matters. Elsewhere a listing would be a way of *choosing* an
    artifact nobody named, which is how a run silently binds to evidence nobody
    picked. Here the checkpoint is named exactly and the answer is the complete
    set of decisions about it -- a reader that returned one would be choosing
    which actor's submission counted.
    """

    def __init__(self, evidence_root: Path) -> None:
        self._root = Path(evidence_root).resolve() / FACTOR_DEVELOPMENT_CURATION_CATEGORY

    @property
    def root(self) -> Path:
        return self._root

    def submissions(self, checkpoint_hash: str) -> tuple[FactorResearchReviewDecisionReceipt, ...]:
        checkpoint = factor_development_evidence_handle(checkpoint_hash)
        folder = self._root / checkpoint
        if not folder.is_dir():
            return ()
        found = []
        for path in sorted(folder.glob("*.json")):
            try:
                receipt = FactorResearchReviewDecisionReceipt.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except ValueError as error:
                raise AuthoringError("factor_research.development_curation_invalid") from error
            if receipt.receipt_hash != path.stem:
                raise AuthoringError("factor_research.development_curation_identity_conflict")
            found.append(receipt)
        return tuple(found)


def verify_factor_development_receipt(
    *,
    receipt: FactorDevelopmentReceipt,
    child: FactorResearchDeterministicEvidence,
    evidence: ResearchExecutionEvidence,
    panel_inventory: tuple[FactorInventoryEntry, ...],
    panel_snapshot_hash: str,
) -> None:
    """Re-derive the receipt's claims from the artifacts that own them.

    A receipt validates its own ``receipt_hash`` over its own contents, which
    proves it has not been edited and proves nothing about whether the hashes it
    carries are the right ones. Self-consistency is not authority.

    Two independent sides are used, and they answer different questions.

    The **admitted Program** answers what was authorized. It was sealed at freeze
    time, before any execution, and the Factor compiler binds the selected axis
    into its ``desk_program_hash`` -- so recomputing that hash from the receipt's
    own selected axis, through the single shared function the compiler used,
    refuses a receipt whose selection was rewritten and everything downstream of
    it consistently resealed.

    The **executed child** answers what actually ran: its ``execution_binding_hash``
    is the Desk method binding it was sealed under, so a child produced by another
    method over the same Panel and outcome cannot be packed into this receipt.

    ``evidence`` is the generic execution record. It is a consistency check rather
    than a second admission check -- the same executor submitted it and the
    receipt -- but it is the artifact the generic layer will hand a consumer, and
    it must not disagree with the receipt it points at.

    ``panel_inventory`` is the ordered typed inventory read from the published
    manifest. Ordered, because the Panel's published factor order is authority in
    its own right; typed, so the caller cannot pass a summary it derived from the
    receipt and have the comparison agree with itself.
    """

    program = receipt.program
    binding = receipt.input_binding

    # What was admitted, recomputed rather than compared with a copy of itself.
    if (
        factor_desk_program_hash(
            envelope_hash=program.envelope_hash,
            selected_factor_ids=receipt.selected_factor_ids,
            method_binding_hash=program.method_binding_hash,
            authority_hash=program.authority_hash,
        )
        != program.desk_program_hash
    ):
        raise AuthoringError("factor_research.development_receipt_selection_not_admitted")

    # What the generic layer recorded about the same run.
    if (
        receipt.program_hash != evidence.program_hash
        or receipt.desk_program_hash != evidence.desk_program_hash
        or receipt.method_binding_hash != evidence.method_binding_hash
    ):
        raise AuthoringError("factor_research.development_receipt_program_mismatch")
    if receipt.desk_input_binding_hash != evidence.desk_input_binding_hash:
        # The binding the Desk sealed and the binding the generic evidence names
        # must be one object, or "these were the inputs" is two claims.
        raise AuthoringError("factor_research.development_receipt_input_binding_mismatch")
    if program.authority_hash != evidence.authority_hash:
        raise AuthoringError("factor_research.development_receipt_authority_mismatch")

    # What actually executed.
    if receipt.checkpoint_hash != child.checkpoint_hash:
        raise AuthoringError("factor_research.development_receipt_child_mismatch")
    if child.execution_binding_hash != receipt.method_binding_hash:
        # The child was sealed under the Desk method binding of the run that
        # produced it. A child from another method is refused here and nowhere
        # else: every other identity in it would still be internally valid.
        raise AuthoringError("factor_research.development_receipt_child_method_mismatch")

    spec = child.program
    if (
        binding.feature_panel_snapshot_hash != spec.feature_panel_snapshot_hash
        or binding.feature_panel_manifest_ref != spec.feature_panel_manifest_ref
        or binding.causal_outcome_snapshot_hash != spec.causal_outcome_snapshot_hash
        or binding.causal_outcome_manifest_ref != spec.causal_outcome_manifest_ref
    ):
        raise AuthoringError("factor_research.development_receipt_input_mismatch")
    if (
        binding.target_policy_hash != spec.target_policy.policy_hash
        or binding.walk_forward_policy_hash != spec.walk_forward_policy.policy_hash
        or binding.screening_policy_hash != spec.evidence_policy.policy_hash
        or binding.redundancy_policy_hash != spec.redundancy_policy.policy_hash
    ):
        # The policies decide the numbers. A binding naming a screening policy
        # the program did not run under describes a different experiment.
        raise AuthoringError("factor_research.development_receipt_policy_mismatch")
    if binding.feature_panel_snapshot_hash != panel_snapshot_hash:
        raise AuthoringError("factor_research.development_receipt_panel_mismatch")

    # The context axis is the statistical universe, so it must be the whole
    # published axis -- a receipt claiming a narrower universe would be claiming
    # a different FDR denominator than the one that ran.
    context = binding.context_factor_ids
    if context != tuple(entry.factor_id for entry in panel_inventory):
        raise AuthoringError("factor_research.development_receipt_context_axis_mismatch")
    if tuple(sorted(context)) != child.program.factor_ids:
        raise AuthoringError("factor_research.development_receipt_context_axis_mismatch")
    for entry, published in zip(binding.context_factors, panel_inventory, strict=True):
        if (
            entry.factor_id != published.factor_id
            or entry.implementation_hash != published.implementation_hash
            or entry.methodology_hash != published.methodology_hash
        ):
            raise AuthoringError("factor_research.development_receipt_methodology_mismatch")


__all__ = [
    "FACTOR_DEVELOPMENT_CURATION_CATEGORY",
    "FACTOR_DEVELOPMENT_EVIDENCE_CATEGORY",
    "FACTOR_DEVELOPMENT_RECEIPT_CATEGORY",
    "FactorDevelopmentCurationReader",
    "FactorDevelopmentEvidenceReader",
    "FactorDevelopmentFactorIdentity",
    "FactorDevelopmentInputBinding",
    "FactorDevelopmentReceipt",
    "FactorDevelopmentReceiptReader",
    "factor_development_curation_uri",
    "factor_development_evidence_handle",
    "factor_development_evidence_uri",
    "factor_development_receipt_handle",
    "factor_development_receipt_uri",
    "publish_factor_development_curation",
    "publish_factor_development_evidence",
    "publish_factor_development_receipt",
    "verify_factor_development_receipt",
]
