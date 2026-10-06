"""Explicit dispatch from one authoring document to its Desk compiler.

Compilers are installed by Host composition. There is no filesystem discovery,
no entry-point scan, and no dynamic import: adding a *method* changes nothing
here, and only adding a whole new Desk kind changes the installed set.

This module is Desk-neutral by construction. It imports the authoring vocabulary
and the compiler Protocol, never a Desk package, so it cannot grow a branch on
which methodology it is routing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from alphalattice.protocols.actor_execution.contracts import (
    ActorKind,
    ActorSubmissionBinding,
    AgentExecutionBinding,
    seal_actor_submission,
)
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    DeskExperimentCompiler,
    ResearchAuthorityResolver,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)


@dataclass(frozen=True)
class SealedSubmission:
    """One authored document, sealed once, with everything sealing resolved.

    Sealing already validates the envelope, resolves the authority against the
    live workspace and compiles the Program. Returning only the Program made
    every lifecycle command re-derive the other two, and a single `resume`
    reached the authority resolver four times to answer one question.

    Carrying the result does not weaken the check. Authority is still resolved
    freshly, from the live workspace, once at the start of the command that
    needs it; what is gone is the second and third resolution of the same
    handles inside the same command, which could only ever agree with the first
    or leave the command working from two different answers.

    `document` travels with the seal so a later step cannot execute a different
    document than the one whose identity was sealed.
    """

    document: Mapping[str, Any]
    envelope: ResearchExperimentEnvelope
    authority: ResolvedResearchAuthority
    program: SealedResearchProgram
    binding: ActorSubmissionBinding


class ResearchExperimentDispatcher:
    """Route an authored document to exactly one explicitly installed compiler."""

    def __init__(
        self,
        *,
        compilers: tuple[DeskExperimentCompiler, ...],
        authority: ResearchAuthorityResolver,
    ) -> None:
        """Install unique Desk compilers and an authority resolver.

        Args:
            compilers: Explicitly installed compilers, one per Desk kind.
            authority: Resolver for live workspace authority.

        Raises:
            AuthoringError: If the compiler set is empty or has duplicate kinds.

        """
        indexed = {value.kind: value for value in compilers}
        if not compilers or len(indexed) != len(compilers):
            raise AuthoringError("research_authoring.compiler_installation_invalid")
        self._compilers = MappingProxyType(indexed)
        self._authority = authority

    @property
    def kinds(self) -> tuple[str, ...]:
        """Return the installed Desk kinds in compiler order."""
        return tuple(self._compilers)

    @property
    def authority(self) -> ResearchAuthorityResolver:
        """Return the installed live authority resolver."""
        return self._authority

    def resolve_compiler(self, kind: str) -> DeskExperimentCompiler:
        """Find the compiler installed for a Desk kind.

        Args:
            kind: Desk kind declared by an experiment.

        Returns:
            The compiler installed for that kind.

        Raises:
            AuthoringError: If no compiler is installed for the kind.

        """
        try:
            return self._compilers[kind]
        except KeyError as error:
            raise AuthoringError("research_authoring.desk_kind_not_installed") from error

    def validate(self, document: Mapping[str, Any]) -> ResearchExperimentEnvelope:
        """Build the envelope and prove the Desk kind is installed. No execution."""
        envelope_document = document.get("experiment")
        if not isinstance(envelope_document, dict):
            raise AuthoringError("research_authoring.experiment_section_missing")
        envelope = ResearchExperimentEnvelope.create(**envelope_document)
        self.resolve_compiler(envelope.kind)
        return envelope

    def compile(self, document: Mapping[str, Any]) -> SealedResearchProgram:
        """Seal the actor-neutral Program for one authored document.

        Actor-neutral by construction and still the published entry point for
        callers that want an identity and nothing else: the Program carries no
        actor field, so every actor submitting the same document reaches the
        same ``program_hash``.
        """
        _envelope, _authority, program = self._compiled(document)
        return program

    def _compiled(
        self, document: Mapping[str, Any]
    ) -> tuple[ResearchExperimentEnvelope, ResolvedResearchAuthority, SealedResearchProgram]:
        """Validate, resolve authority, compile -- the three steps, once.

        Authority is resolved *before* the Desk compiler is called, so an
        unknown snapshot or universe handle fails while the request is still a
        request -- never after a Desk has begun compiling against handles that
        turn out to name nothing.

        The envelope and the authority come back with the Program because both
        are already in hand here, and every caller needed them.
        """
        envelope = self.validate(document)
        compiler = self.resolve_compiler(envelope.kind)
        authority = self._authority.resolve(envelope)
        compilation = compiler.compile_desk_program(
            envelope=envelope,
            document=document,
            authority=authority,
        )
        return (
            envelope,
            authority,
            SealedResearchProgram.create(
                kind=envelope.kind,
                envelope_hash=envelope.envelope_hash,
                desk_program_hash=compilation.desk_program_hash,
                resolved_sessions=authority.sessions,
                catalog_hash=compilation.catalog_hash,
                method_binding_hash=compilation.method_binding_hash,
                parameter_domain_hash=compilation.parameter_domain_hash,
                authority_hash=authority.authority_hash,
            ),
        )

    def seal(
        self,
        document: Mapping[str, Any],
        *,
        actor_kind: ActorKind,
        actor_id: str,
        agent_execution: AgentExecutionBinding | None = None,
    ) -> SealedSubmission:
        """Compile the Program, then seal who submitted it.

        The ordering is the point. The Program is sealed first and carries no
        actor field, so every actor submitting the same document reaches the same
        ``program_hash``. The actor binding is then sealed *over* that hash, so
        the signature covers the compiled Desk method -- which ``envelope_hash``
        alone excludes, since the envelope carries no methodology at all.

        `agent_execution` is required exactly when the actor is an installed
        Agent; the existing actor-execution seam rejects any other combination,
        so a plain CLI invocation cannot claim Agent provenance.
        """
        envelope, authority, program = self._compiled(document)
        return SealedSubmission(
            document=document,
            envelope=envelope,
            authority=authority,
            program=program,
            binding=seal_actor_submission(
                actor_kind=actor_kind,
                actor_id=actor_id,
                submission_hash=program.program_hash,
                agent_execution=agent_execution,
            ),
        )

    def freeze(
        self,
        document: Mapping[str, Any],
        *,
        actor_kind: ActorKind,
        actor_id: str,
        agent_execution: AgentExecutionBinding | None = None,
    ) -> tuple[SealedResearchProgram, ActorSubmissionBinding]:
        """Return the sealed Program and its actor binding."""
        sealed = self.seal(
            document,
            actor_kind=actor_kind,
            actor_id=actor_id,
            agent_execution=agent_execution,
        )
        return sealed.program, sealed.binding


__all__ = ["ResearchExperimentDispatcher", "SealedSubmission"]
