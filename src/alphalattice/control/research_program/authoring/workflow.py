"""Desk-neutral freeze, run, and replay for one authored experiment.

The three commands used to resolve to the same call, so nothing ever executed
and nothing was ever replayed. They are genuinely different here:

``freeze``   compiles and seals a Program. No execution.
``run``      seals, then calls the injected Desk executor, then writes evidence.
``replay``   seals again, proves the identity is unchanged, and reads the stored
             evidence and every artifact it names back **by hash**. It never
             touches an executor, so it cannot recompute and then claim reuse.

Every command that resolves live sources begins the same way and only the same
way: one ``dispatcher.seal``, one ``_outputs``. Sealing already validates the
envelope and resolves the authority, so re-deriving either afterwards was
repetition rather than a second check -- a single ``resume`` used to reach the
authority resolver four times to answer one question. The two readback commands
deliberately never seal: they resolve no live source, which is the property that
makes them readback rather than re-execution.

This module knows no Desk. Executors arrive by injection keyed on the same kind
the compilers are keyed on, so adding a method changes nothing here.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast
from uuid import uuid4

from alphalattice.kernel.shared_kernel.environment import held_offline
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution.contracts import (
    ActorKind,
    ActorSubmissionBinding,
    AgentExecutionBinding,
    seal_actor_submission,
)
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    DeskEvidenceVerifier,
    DeskExperimentExecutor,
    NumericalCallRecorder,
    ResearchExecutionEvidence,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)

from .dispatcher import ResearchExperimentDispatcher, SealedSubmission

_REPLAY_DISPOSITIONS = frozenset({"REUSED_EXACT", "VERIFIED_DURABLE_GRAPH_READBACK"})


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """Encode the canonical byte form used by both stores."""
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode()


def _publish_immutable(root: Path, name: str, content: bytes, *, conflict: str) -> None:
    """Write content-addressed bytes once, atomically, and never rewrite them.

    Content-addressed means an existing file with this name already holds these
    bytes. Rewriting it turns a read of immutable content into a window where
    the file is briefly absent or half-written, for no gain. A file that exists
    with *different* bytes is a name collision, and the caller names the refusal
    because "which identity collided" is the caller's question, not this one's.
    """
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if target.exists():
        if target.read_bytes() != content:
            raise AuthoringError(conflict)
        return
    staged = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    staged.write_bytes(content)
    os.replace(staged, target)


class ResearchEvidenceStore:
    """Content-addressed evidence under one output workspace.

    One Program has one immutable terminal result in this output workspace.
    Identical publication is idempotent; a conflicting observation is refused.
    Zero-work readbacks keep their per-call counters without replacing that run.
    """

    def __init__(self, output_workspace: Path) -> None:
        """Locate the evidence store under an output workspace.

        Args:
            output_workspace: Workspace where research evidence is published.

        """
        self._root = Path(output_workspace) / "research-evidence"

    def uri(self, evidence_hash: str) -> str:
        """Return the workspace URI for a sealed evidence hash."""
        return f"playpen://research-evidence/{evidence_hash}"

    def publish(self, evidence: ResearchExecutionEvidence) -> str:
        """Publish immutable evidence and return its workspace URI.

        Args:
            evidence: Validated evidence to publish.

        Returns:
            The URI of the published evidence.

        Raises:
            AuthoringError: If this Program or evidence name conflicts with stored data.

        """
        # Nothing re-validates identity here: ResearchExecutionEvidence seals and
        # checks its own hash in a model validator, so an instance that exists is
        # already consistent. A second check would only drift from that one.
        previous = self.load_for_program(evidence.program_hash)
        if previous is not None and previous.evidence_hash != evidence.evidence_hash:
            raise AuthoringError("research_authoring.evidence_identity_conflict")
        _publish_immutable(
            self._root,
            f"{evidence.evidence_hash}.json",
            _canonical_bytes(evidence.model_dump(mode="json")),
            conflict="research_authoring.evidence_identity_conflict",
        )
        return self.uri(evidence.evidence_hash)

    def load_for_program(self, program_hash: str) -> ResearchExecutionEvidence | None:
        """Find the single valid evidence record for a sealed Program.

        Args:
            program_hash: Identity of the Program to read.

        Returns:
            Its evidence, or ``None`` when none is published.

        Raises:
            AuthoringError: If a stored record is damaged or identities conflict.

        """
        if not self._root.is_dir():
            return None
        matches = []
        for path in sorted(self._root.glob("*.json")):
            # A stored record that no longer parses as evidence is a damaged
            # artifact, named as such: it is not the caller's declaration,
            # which is what a contract validation error would be read as.
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("research evidence record is not an object")
                if payload.get("program_hash") != program_hash:
                    continue
                evidence = ResearchExecutionEvidence(**payload)
            except (ValueError, TypeError) as error:
                raise AuthoringError(
                    f"research_authoring.evidence_artifact_invalid:{path.name}"
                ) from error
            if evidence.evidence_hash != path.stem:
                raise AuthoringError("research_authoring.evidence_identity_conflict")
            matches.append(evidence)
        if len(matches) > 1:
            raise AuthoringError("research_authoring.evidence_identity_conflict")
        return matches[0] if matches else None


class ResearchProgramStore:
    """Content-addressed sealed Programs for the preflight/run lifecycle."""

    def __init__(self, output_workspace: Path) -> None:
        """Locate the sealed Program store under an output workspace.

        Args:
            output_workspace: Workspace where Programs are published.

        """
        self._root = Path(output_workspace) / "research-programs"

    def publish(self, *, program: SealedResearchProgram, document: Mapping[str, Any]) -> None:
        """Publish one immutable Program with its authored document identity.

        Args:
            program: Sealed Program to publish.
            document: Authored document whose content is bound to the Program.

        Raises:
            AuthoringError: If the Program name already holds different content.

        """
        document_hash = str(canonical_hash(document))
        _publish_immutable(
            self._root,
            f"{program.program_hash}.json",
            _canonical_bytes(
                {
                    "program": program.model_dump(mode="json"),
                    "document_hash": document_hash,
                }
            ),
            conflict="research_authoring.program_identity_conflict",
        )
        # The document's index, written after its Program: a read by the document finds the
        # Programs it sealed by key, never by opening every record.
        known = self.recorded_for(document_hash)
        if program.program_hash not in known:
            index = self._root / "by-document" / f"{document_hash}.json"
            index.parent.mkdir(parents=True, exist_ok=True)
            staged = index.with_name(f".{index.name}.{uuid4().hex}.tmp")
            staged.write_bytes(
                _canonical_bytes(
                    {
                        "document_hash": document_hash,
                        "program_hashes": [*known, program.program_hash],
                    }
                )
            )
            os.replace(staged, index)

    def recorded_for(self, document_hash: str) -> tuple[str, ...]:
        """The Programs a document sealed, oldest first, as its index names them.

        Args:
            document_hash: The authored document's canonical hash.

        Returns:
            The Programs' hashes; none for a document sealed before the index was kept.

        Raises:
            AuthoringError: If the index is damaged.

        """
        index = self._root / "by-document" / f"{document_hash}.json"
        if not index.is_file():
            return ()
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
        except ValueError as error:
            raise AuthoringError("research_authoring.program_index_invalid") from error
        hashes = payload.get("program_hashes") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("document_hash") != document_hash
            or not isinstance(hashes, list)
            or not all(isinstance(value, str) for value in hashes)
        ):
            raise AuthoringError("research_authoring.program_index_invalid")
        return tuple(hashes)

    def load(
        self, *, program_hash: str, document: Mapping[str, Any]
    ) -> SealedResearchProgram | None:
        """Read a Program only when its authored document still agrees.

        Args:
            program_hash: Identity of the Program to read.
            document: Authored document to compare with the stored binding.

        Returns:
            The stored Program, or ``None`` when it is absent.

        Raises:
            AuthoringError: If the document differs or the stored Program is invalid.

        """
        stored = self._stored(program_hash)
        if stored is None:
            return None
        payload, stem = stored
        if payload.get("document_hash") != canonical_hash(document):
            raise AuthoringError("research_authoring.sealed_program_document_mismatch")
        return self._program(payload, stem)

    def load_by_hash(self, program_hash: str) -> SealedResearchProgram | None:
        """Read an explicitly named immutable Program without reauthoring it."""
        stored = self._stored(program_hash)
        return None if stored is None else self._program(*stored)

    def _stored(self, program_hash: str) -> tuple[Mapping[str, Any], str] | None:
        """Read the raw record and the name it was found under, if present.

        Raw on purpose: `load` has to decide the document mismatch before the
        Program is validated, so that a record written against another document
        says so rather than failing as a malformed Program.
        """
        target = self._root / f"{program_hash}.json"
        if not target.is_file():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except ValueError as error:
            raise AuthoringError(
                f"research_authoring.program_artifact_invalid:{target.name}"
            ) from error
        if not isinstance(payload, dict):
            raise AuthoringError(f"research_authoring.program_artifact_invalid:{target.name}")
        return payload, target.stem

    @staticmethod
    def _program(payload: Mapping[str, Any], stem: str) -> SealedResearchProgram:
        try:
            program = cast(
                SealedResearchProgram,
                SealedResearchProgram.model_validate(payload.get("program")),
            )
        except ValueError as error:
            raise AuthoringError(
                f"research_authoring.program_artifact_invalid:{stem}.json"
            ) from error
        if program.program_hash != stem:
            raise AuthoringError("research_authoring.program_identity_conflict")
        return program


@dataclass(frozen=True)
class _OutputStores:
    """One resolved output root and the two stores that live under it.

    Resolved once per command. The resolution is the source/output isolation
    proof, and it is a pure function of the declared path and the two roots, so
    repeating it inside the same command could only ever agree with itself.
    """

    root: Path
    programs: ResearchProgramStore
    evidence: ResearchEvidenceStore


class ResearchProgramWorkflow:
    """Orchestrate the three commands over an injected dispatcher and executors."""

    def __init__(
        self,
        *,
        dispatcher: ResearchExperimentDispatcher,
        executors: tuple[DeskExperimentExecutor, ...],
        workspace_root: Path,
        source_workspace: Path,
        verifiers: tuple[DeskEvidenceVerifier, ...] = (),
    ) -> None:
        """Install Desk execution and verification owners for one workspace.

        Args:
            dispatcher: Authoring dispatcher that seals Programs.
            executors: Explicitly installed Desk executors.
            workspace_root: Root allowed to hold output workspaces.
            source_workspace: Source workspace that outputs must not overlap.
            verifiers: Explicitly installed Desk evidence verifiers.

        Raises:
            AuthoringError: If executor or verifier kinds are duplicated.

        """
        indexed = {value.kind: value for value in executors}
        if len(indexed) != len(executors):
            raise AuthoringError("research_authoring.executor_installation_invalid")
        indexed_verifiers = {value.kind: value for value in verifiers}
        if len(indexed_verifiers) != len(verifiers):
            raise AuthoringError("research_authoring.verifier_installation_invalid")
        self._dispatcher = dispatcher
        self._executors = MappingProxyType(indexed)
        self._verifiers = MappingProxyType(indexed_verifiers)
        self._workspace_root = Path(workspace_root)
        # The real source workspace, not the document's declared baseline: the
        # baseline is caller input, so trusting it to describe the source would
        # let a document authorize writes into the workspace it reads.
        self._source_workspace = Path(source_workspace)

    def validate(self, document: Mapping[str, Any]) -> object:
        """Validate an authored document against the installed Desk kinds."""
        return self._dispatcher.validate(document)

    def prepare(
        self,
        document: Mapping[str, Any],
        *,
        actor_kind: ActorKind,
        actor_id: str,
        agent_execution: AgentExecutionBinding | None = None,
    ) -> SealedSubmission:
        """Prepare one call-scoped submission without publishing or executing."""
        sealed = self._dispatcher.seal(
            document,
            actor_kind=actor_kind,
            actor_id=actor_id,
            agent_execution=agent_execution,
        )
        self._output_workspace(sealed.envelope.output_workspace)
        return sealed

    def execute_prepared(self, sealed: SealedSubmission) -> ResearchExecutionEvidence:
        """Resume the admitted Program, using this call's verified bound inputs."""
        if self._dispatcher.compile(sealed.document) != sealed.program:
            raise AuthoringError("research_authoring.sealed_program_identity_mismatch")
        outputs = self._outputs(sealed.envelope.output_workspace)
        outputs.programs.publish(program=sealed.program, document=sealed.document)
        stored = outputs.evidence.load_for_program(sealed.program.program_hash)
        if stored is not None:
            self._verified_readback(
                program=sealed.program, stored=stored, authority=sealed.authority, outputs=outputs
            )
            return stored
        return self._run_sealed(sealed=sealed, outputs=outputs, recorder=None)[0]

    def freeze(
        self,
        document: Mapping[str, Any],
        *,
        actor_kind: ActorKind,
        actor_id: str,
        agent_execution: AgentExecutionBinding | None = None,
    ) -> tuple[SealedResearchProgram, ActorSubmissionBinding]:
        """Seal a Program and its actor without execution or publication."""
        sealed = self._dispatcher.seal(
            document,
            actor_kind=actor_kind,
            actor_id=actor_id,
            agent_execution=agent_execution,
        )
        return sealed.program, sealed.binding

    def _outputs(self, declared: str) -> _OutputStores:
        """Resolve the proven output root and its stores once per command."""
        root = self._output_workspace(declared)
        return _OutputStores(
            root=root,
            programs=ResearchProgramStore(root),
            evidence=ResearchEvidenceStore(root),
        )

    def preflight(
        self,
        document: Mapping[str, Any],
        *,
        actor_kind: ActorKind,
        actor_id: str,
        agent_execution: AgentExecutionBinding | None = None,
    ) -> tuple[SealedResearchProgram, ActorSubmissionBinding]:
        """Seal and persist a Program without calling any numerical owner."""
        sealed = self._dispatcher.seal(
            document,
            actor_kind=actor_kind,
            actor_id=actor_id,
            agent_execution=agent_execution,
        )
        self._outputs(sealed.envelope.output_workspace).programs.publish(
            program=sealed.program,
            document=document,
        )
        return sealed.program, sealed.binding

    def run(
        self,
        document: Mapping[str, Any],
        *,
        actor_kind: ActorKind,
        actor_id: str,
        agent_execution: AgentExecutionBinding | None = None,
        recorder: NumericalCallRecorder | None = None,
    ) -> tuple[ResearchExecutionEvidence, ActorSubmissionBinding]:
        """Seal, execute for real, and write immutable evidence."""
        sealed = self._dispatcher.seal(
            document,
            actor_kind=actor_kind,
            actor_id=actor_id,
            agent_execution=agent_execution,
        )
        # Executor first, then the output root: an uninstalled Desk is reported
        # as one here, rather than as whatever the declared output path happens
        # to be wrong about. `run_sealed` resolves them the other way round, and
        # both orders are the ones each path already had.
        executor = self._resolve_executor(sealed.program.kind)
        return self._execute(
            sealed=sealed,
            program=sealed.program,
            executor=executor,
            outputs=self._outputs(sealed.envelope.output_workspace),
            recorder=recorder,
        )

    def run_sealed(
        self,
        document: Mapping[str, Any],
        *,
        actor_kind: ActorKind,
        actor_id: str,
        agent_execution: AgentExecutionBinding | None = None,
        recorder: NumericalCallRecorder | None = None,
    ) -> tuple[ResearchExecutionEvidence, ActorSubmissionBinding]:
        """Execute only the exact Program previously sealed by preflight."""
        sealed = self._dispatcher.seal(
            document,
            actor_kind=actor_kind,
            actor_id=actor_id,
            agent_execution=agent_execution,
        )
        return self._run_sealed(
            sealed=sealed,
            outputs=self._outputs(sealed.envelope.output_workspace),
            recorder=recorder,
        )

    def resume(
        self,
        document: Mapping[str, Any],
        *,
        actor_kind: ActorKind,
        actor_id: str,
        agent_execution: AgentExecutionBinding | None = None,
        recorder: NumericalCallRecorder | None = None,
    ) -> tuple[ResearchExecutionEvidence, ActorSubmissionBinding]:
        """Reuse only an exact completed Program; otherwise run that sealed Program."""
        sealed = self._dispatcher.seal(
            document,
            actor_kind=actor_kind,
            actor_id=actor_id,
            agent_execution=agent_execution,
        )
        outputs = self._outputs(sealed.envelope.output_workspace)
        if (
            outputs.programs.load(program_hash=sealed.program.program_hash, document=document)
            is None
        ):
            raise AuthoringError("research_authoring.sealed_program_required")
        stored = outputs.evidence.load_for_program(sealed.program.program_hash)
        if stored is not None:
            # The replay branch, from the seal this command already took. No
            # executor is resolved on this path, which is what keeps the zero
            # call count a fact about the path rather than a claim about a run.
            return (
                self._verified_readback(
                    program=sealed.program,
                    stored=stored,
                    authority=sealed.authority,
                    outputs=outputs,
                ),
                sealed.binding,
            )
        return self._run_sealed(sealed=sealed, outputs=outputs, recorder=recorder)

    def inspect(
        self,
        document: Mapping[str, Any],
        *,
        actor_kind: ActorKind,
        actor_id: str,
        agent_execution: AgentExecutionBinding | None = None,
    ) -> tuple[
        SealedResearchProgram,
        ResearchExecutionEvidence | None,
        ActorSubmissionBinding,
        Literal["CURRENT", "NOT_CURRENT"],
    ]:
        """Read the sealed Program and optional evidence without verification or work.

        A document whose inputs moved since it was sealed compiles another Program now; the
        last Program it sealed is read instead, `NOT_CURRENT`, never as not found.
        """
        sealed = self._dispatcher.seal(
            document,
            actor_kind=actor_kind,
            actor_id=actor_id,
            agent_execution=agent_execution,
        )
        outputs = self._outputs(sealed.envelope.output_workspace)
        stored = outputs.programs.load(
            program_hash=sealed.program.program_hash,
            document=document,
        )
        if stored is not None:
            return (
                stored,
                outputs.evidence.load_for_program(stored.program_hash),
                sealed.binding,
                "CURRENT",
            )
        recorded = outputs.programs.recorded_for(str(canonical_hash(document)))
        program = (
            None
            if not recorded
            else outputs.programs.load(program_hash=recorded[-1], document=document)
        )
        if program is None:
            raise AuthoringError("research_authoring.sealed_program_required")
        return (
            program,
            outputs.evidence.load_for_program(program.program_hash),
            seal_actor_submission(
                actor_kind=actor_kind,
                actor_id=actor_id,
                submission_hash=program.program_hash,
                agent_execution=agent_execution,
            ),
            "NOT_CURRENT",
        )

    def inspect_sealed(
        self,
        document: Mapping[str, Any],
        *,
        program_hash: str,
        actor_kind: ActorKind,
        actor_id: str,
        agent_execution: AgentExecutionBinding | None = None,
    ) -> tuple[
        SealedResearchProgram,
        ResearchExecutionEvidence | None,
        ActorSubmissionBinding,
    ]:
        """Read one explicitly named sealed Program without resolving live inputs.

        Only the Desk-neutral envelope is parsed and no authority is resolved,
        which is the whole point of this route: nothing here consults a live
        source, so a readback cannot quietly become a recompilation.
        """
        envelope = self._sealed_envelope(document)
        outputs = self._outputs(envelope.output_workspace)
        sealed = outputs.programs.load_by_hash(program_hash)
        if sealed is None:
            raise AuthoringError("research_authoring.sealed_program_required")
        if sealed.kind != envelope.kind or sealed.envelope_hash != envelope.envelope_hash:
            raise AuthoringError("research_authoring.sealed_program_document_mismatch")
        return (
            sealed,
            outputs.evidence.load_for_program(sealed.program_hash),
            seal_actor_submission(
                actor_kind=actor_kind,
                actor_id=actor_id,
                submission_hash=sealed.program_hash,
                agent_execution=agent_execution,
            ),
        )

    def readback_sealed(
        self,
        document: Mapping[str, Any],
        *,
        program_hash: str,
        actor_kind: ActorKind,
        actor_id: str,
        agent_execution: AgentExecutionBinding | None = None,
        authority: ResolvedResearchAuthority | None = None,
        recorded_readback: bool = False,
        stored_sink: list[ResearchExecutionEvidence] | None = None,
    ) -> tuple[ResearchExecutionEvidence, ActorSubmissionBinding]:
        """Verify an explicitly sealed durable graph with zero live-source resolution."""
        program, stored, binding = self.inspect_sealed(
            document,
            program_hash=program_hash,
            actor_kind=actor_kind,
            actor_id=actor_id,
            agent_execution=agent_execution,
        )
        if stored is None:
            raise AuthoringError("research_authoring.evidence_unavailable_for_replay")
        # A durable Task may supply its independently admitted authority. Plain
        # historical callers still pass None; verifiers retain their refusal.
        verified = self._verified_readback(
            program=program,
            stored=stored,
            authority=authority,
            outputs=self._outputs(self._sealed_envelope(document).output_workspace),
            recorded_readback=recorded_readback,
        )
        if stored_sink is not None:
            # Bind a Task publication to the record whose graph this call proved.
            stored_sink.append(stored)
        return verified, binding

    def replay(
        self,
        document: Mapping[str, Any],
        *,
        actor_kind: ActorKind,
        actor_id: str,
        agent_execution: AgentExecutionBinding | None = None,
    ) -> tuple[ResearchExecutionEvidence, ActorSubmissionBinding]:
        """Prove the identity is unchanged and read every artifact back by hash.

        No executor is resolved and none is called. That is what makes the zero
        call count a fact about this path rather than a claim about a run that
        happened to skip its work.
        """
        sealed = self._dispatcher.seal(
            document,
            actor_kind=actor_kind,
            actor_id=actor_id,
            agent_execution=agent_execution,
        )
        outputs = self._outputs(sealed.envelope.output_workspace)
        stored = outputs.evidence.load_for_program(sealed.program.program_hash)
        if stored is None:
            raise AuthoringError("research_authoring.evidence_unavailable_for_replay")
        # The authority this command resolved from the live workspace, rather
        # than one read out of the record being checked. A graph that was
        # re-sealed against another Panel is self-consistent by construction;
        # only a side the graph did not produce can contradict it.
        return (
            self._verified_readback(
                program=sealed.program,
                stored=stored,
                authority=sealed.authority,
                outputs=outputs,
            ),
            sealed.binding,
        )

    def _run_sealed(
        self,
        *,
        sealed: SealedSubmission,
        outputs: _OutputStores,
        recorder: NumericalCallRecorder | None,
    ) -> tuple[ResearchExecutionEvidence, ActorSubmissionBinding]:
        """Execute the stored Program this document seals to, and nothing else."""
        stored = outputs.programs.load(
            program_hash=sealed.program.program_hash,
            document=sealed.document,
        )
        if stored is None:
            raise AuthoringError("research_authoring.sealed_program_required")
        if stored != sealed.program:
            raise AuthoringError("research_authoring.sealed_program_identity_mismatch")
        return self._execute(
            sealed=sealed,
            program=stored,
            executor=self._resolve_executor(stored.kind),
            outputs=outputs,
            recorder=recorder,
        )

    def _execute(
        self,
        *,
        sealed: SealedSubmission,
        program: SealedResearchProgram,
        executor: DeskExperimentExecutor,
        outputs: _OutputStores,
        recorder: NumericalCallRecorder | None,
    ) -> tuple[ResearchExecutionEvidence, ActorSubmissionBinding]:
        """Call the Desk, describe what it did, and publish that description once.

        The one place a numerical owner is called on any path. `run` reaches it
        with the Program it just sealed and `run_sealed` with the stored Program
        it proved identical, which is the only difference the two ever had. Every Desk's
        call runs held offline, as its envelope declares, whatever the workspace allows.
        """
        with held_offline():
            result = executor.execute(
                program=program,
                document=sealed.document,
                authority=sealed.authority,
                output_workspace=outputs.root,
                recorder=recorder,
            )
        evidence = ResearchExecutionEvidence.create(
            kind=program.kind,
            program_hash=program.program_hash,
            desk_program_hash=program.desk_program_hash,
            method_binding_hash=program.method_binding_hash,
            authority_hash=program.authority_hash,
            disposition=result.disposition,
            numerical_call_count=result.numerical_call_count,
            artifact_uris=result.artifact_uris,
            formation_sessions=result.formation_sessions,
            desk_input_binding_hash=result.desk_input_binding_hash,
        )
        previous = outputs.evidence.load_for_program(program.program_hash)
        if previous is not None and evidence.numerical_call_count == 0:
            # A completed checkpoint may do no new work. Its per-call count is
            # not a second terminal result. Verify the same graph and return the
            # existing zero-work readback shape without rewriting the first run.
            observation_fields = {"evidence_hash", "disposition", "numerical_call_count"}
            if evidence.model_dump(exclude=observation_fields) != previous.model_dump(
                exclude=observation_fields
            ):
                raise AuthoringError("research_authoring.evidence_identity_conflict")
            return (
                self._verified_readback(
                    program=program, stored=previous, authority=sealed.authority, outputs=outputs
                ),
                sealed.binding,
            )
        outputs.evidence.publish(evidence)
        return evidence, sealed.binding

    def _verified_readback(
        self,
        *,
        program: SealedResearchProgram,
        stored: ResearchExecutionEvidence,
        authority: ResolvedResearchAuthority | None,
        outputs: _OutputStores,
        recorded_readback: bool = False,
    ) -> ResearchExecutionEvidence:
        """Prove stored evidence belongs to this Program, then read its graph back.

        Shared by `replay`, `resume`'s reuse branch and `readback_sealed`, which
        differ only in where the Program came from and whether a freshly
        resolved authority exists to check it against. No executor is resolved
        here and none is called, so the zero call count below is a fact about
        this code path rather than a claim about a run.

        Artifact verification is required, not an optional callback. As a
        callback it was simply never installed by the CLI, so replay validated
        one JSON file and reported ``REUSED_EXACT`` for evidence whose surface,
        diagnostics, and chunks could all have been deleted.
        """
        if (
            stored.desk_program_hash != program.desk_program_hash
            or stored.method_binding_hash != program.method_binding_hash
            or stored.authority_hash != program.authority_hash
        ):
            raise AuthoringError("research_authoring.replay_identity_mismatch")
        # A Desk that can produce artifacts must be able to verify them, so a
        # missing verifier is a refusal rather than a silent skip.
        verifier = self._verifiers.get(stored.kind)
        if verifier is None:
            raise AuthoringError("research_authoring.evidence_verifier_not_installed")
        # The Program is passed, not the stored evidence's own hashes: the
        # verifier's job includes proving the graph belongs to the Program being
        # replayed, and evidence that vouched for itself could not answer that.
        recorded_verifier = (
            getattr(verifier, "verify_recorded", None) if recorded_readback else None
        )
        verify = recorded_verifier or verifier.verify
        verify(
            program=program,
            evidence=stored,
            authority=authority,
            output_workspace=outputs.root,
        )
        disposition = (
            "VERIFIED_DURABLE_GRAPH_READBACK"
            if recorded_verifier is not None
            else getattr(verifier, "replay_disposition", "REUSED_EXACT")
        )
        if disposition not in _REPLAY_DISPOSITIONS:
            raise AuthoringError("research_authoring.replay_disposition_invalid")
        return ResearchExecutionEvidence.create(
            kind=stored.kind,
            program_hash=stored.program_hash,
            desk_program_hash=stored.desk_program_hash,
            method_binding_hash=stored.method_binding_hash,
            authority_hash=stored.authority_hash,
            disposition=disposition,
            numerical_call_count=0,
            artifact_uris=stored.artifact_uris,
            formation_sessions=stored.formation_sessions,
            desk_input_binding_hash=stored.desk_input_binding_hash,
        )

    def verifier(self, kind: str) -> DeskEvidenceVerifier:
        """Return the verifier installed for one Desk kind.

        A readback caller can use the verifier's own walk in this request.
        """
        try:
            return self._verifiers[kind]
        except KeyError as error:
            raise AuthoringError("research_authoring.evidence_verifier_not_installed") from error

    def _resolve_executor(self, kind: str) -> DeskExperimentExecutor:
        try:
            return self._executors[kind]
        except KeyError as error:
            raise AuthoringError("research_authoring.desk_executor_not_installed") from error

    @staticmethod
    def _sealed_envelope(document: Mapping[str, Any]) -> ResearchExperimentEnvelope:
        """Validate only the Desk-neutral envelope for source-free readback."""
        value = document.get("experiment")
        if not isinstance(value, Mapping):
            raise AuthoringError("research_authoring.experiment_section_missing")
        return ResearchExperimentEnvelope.create(**dict(value))

    def _output_workspace(self, declared: str) -> Path:
        """Resolve the output root and prove it cannot touch the source.

        Rejecting absolute paths was never enough: ``../../elsewhere`` is
        relative. And the workflow did not know the real source workspace, so
        even a contained path could have been the source itself. Both are
        resolved and compared here, before any directory, artifact, checkpoint,
        or evidence write.
        """
        candidate = Path(declared.replace("\\", "/"))
        if candidate.is_absolute():
            raise AuthoringError("research_authoring.output_workspace_not_relative")
        root = self._workspace_root.resolve()
        # resolve() follows symlinks and junctions and collapses "..", so an
        # escape shows up as a path that is no longer under the authorized root.
        output = (root / candidate).resolve()
        if not self._contains(root, output) and output != root:
            raise AuthoringError("research_authoring.output_workspace_escapes_root")
        source = self._source_workspace.resolve()
        if self._same_path(output, source):
            raise AuthoringError("research_authoring.output_workspace_is_source")
        if self._contains(source, output) or self._contains(output, source):
            raise AuthoringError("research_authoring.output_workspace_overlaps_source")
        return output

    @staticmethod
    def _normalized(value: Path) -> str:
        # NTFS is case-insensitive, so "Out" and "out" are one directory. A
        # case-sensitive comparison would call them distinct and let an alias
        # of the source through.
        return os.path.normcase(str(value))

    @classmethod
    def _same_path(cls, left: Path, right: Path) -> bool:
        return cls._normalized(left) == cls._normalized(right)

    @classmethod
    def _contains(cls, parent: Path, child: Path) -> bool:
        """Report whether ``child`` is strictly inside ``parent``.

        Compared component-wise rather than by string prefix, so a sibling named
        ``workspace-backup`` is not mistaken for a child of ``workspace``.
        """
        parent_parts = Path(cls._normalized(parent)).parts
        child_parts = Path(cls._normalized(child)).parts
        return len(child_parts) > len(parent_parts) and (
            child_parts[: len(parent_parts)] == parent_parts
        )


__all__ = ["ResearchEvidenceStore", "ResearchProgramStore", "ResearchProgramWorkflow"]
