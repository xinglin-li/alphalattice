"""Thin actor-neutral authoring envelope shared by every research Desk.

This envelope carries only what every Desk experiment needs to be admitted. It
is deliberately not a mega schema: methodology stays in the Desk-owned
experiment contract that each compiler validates.

The envelope is a *request*, never authority. It carries opaque handles and an
unresolved date range; the Host resolves exact sessions, exact snapshot
identities, and every content hash. It cannot request current activation,
Holdout access, or a publication pointer change: `publication_intent` admits one
development-only value.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from enum import IntEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, Self, cast

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

_HASH = r"^[0-9a-f]{64}$"
_HANDLE = r"^[a-z0-9][a-z0-9._-]{0,95}$"
_SNAPSHOT_HANDLE = r"^[a-z0-9][a-z0-9._-]{0,95}(?:@[0-9a-f]{64})?$"


class AuthoringError(ValueError):
    """Stable failure raised before any Desk compiler or numerical call runs.

    ``expected`` is what the refusing owner knows a declaration field may hold, by the field's
    path, served beside the code so the caller can correct the declaration (LAWS OP12).
    """

    def __init__(self, code: str, *, expected: Mapping[str, object] | None = None) -> None:
        """Refuse with a stable code and, when the owner knows it, what each field may hold.

        Args:
            code: The typed failure code, optionally with its subject after a colon.
            expected: Field path to the value or values it may hold.
        """
        super().__init__(code)
        self.expected: dict[str, object] = dict(expected or {})


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class DeskSection(BaseModel):  # type: ignore[misc]
    """A Desk's section of an authored document: the keys its readers read, each described.

    Frozen and closed (SC2); `schema show EXPERIMENT_PLAN` prints it under
    `declaration_sections` (SC3). The Desk judges the values with its own codes, since most need
    its installed catalogs; the key set is checked by ``refuse_unknown_section_keys``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)


def refuse_unknown_section_keys(
    section: Mapping[str, Any], contract: type[DeskSection], *, place: str
) -> None:
    """Refuse a key a Desk's section contract does not name, with the one code (DA11).

    Args:
        section: The authored section, at ``place`` in the document.
        contract: The Desk's section contract.
        place: The section's path in the document, such as ``risk`` or ``risk.estimator``.

    Raises:
        AuthoringError: ``research_authoring.section_key_unknown:<place>.<key>``, naming the
            first unknown key in sorted order.
    """
    admitted = {field.alias or name for name, field in contract.model_fields.items()}
    unknown = sorted(str(key) for key in section if str(key) not in admitted)
    if unknown:
        raise AuthoringError(f"research_authoring.section_key_unknown:{place}.{unknown[0]}")


class MarketPhase(IntEnum):
    """Ordered points inside one trading session, coarse enough to be decidable.

    A decision cutoff needs more than a date. "2024-03-15" cannot distinguish a
    pre-open decision from a post-close one, and the difference is a whole
    session of information: daily Feature values for a session exist only after
    that session's official close. Ordered rather than a bare label because the
    only question ever asked of it is whether the decision happened at or after
    the point some input became available.

    ``UNSPECIFIED`` is what a date-only request carries. It is deliberately the
    lowest member and is never treated as a phase: the Host refuses it rather
    than guessing, because guessing high is exactly the read that is not earned.
    """

    UNSPECIFIED = 0
    PRE_OPEN = 1
    INTRADAY = 2
    OFFICIAL_CLOSE = 3
    POST_CLOSE = 4

    @classmethod
    def _missing_(cls, value: object) -> MarketPhase | None:
        """Accept the member name, which is what an authored document writes."""

        if isinstance(value, str):
            return cls.__members__.get(value.strip().upper())
        return None


class DecisionCutoff(_Contract):
    """The event a research request makes its decision at.

    Accepts a bare date so an older document still parses; that form resolves to
    ``UNSPECIFIED`` and is refused by the Host rather than silently promoted to a
    post-close read.
    """

    session: date
    phase: MarketPhase = MarketPhase.UNSPECIFIED

    @field_serializer("phase")  # type: ignore[untyped-decorator]
    def serialize_phase(self, value: MarketPhase) -> str:
        """Emit the name: a stored ``3`` is unreadable and invites a wrong guess."""
        return value.name

    @classmethod
    def coerce(cls, value: object) -> DecisionCutoff:
        """Accept the typed form, a bare date, or the ISO date a document carries."""
        if isinstance(value, DecisionCutoff):
            return value
        if isinstance(value, Mapping):
            return cast(DecisionCutoff, cls.model_validate(dict(value)))
        if isinstance(value, date):
            return cls(session=value)
        if isinstance(value, str):
            try:
                return cls(session=date.fromisoformat(value))
            except ValueError as error:
                raise AuthoringError("research_authoring.decision_cutoff_invalid") from error
        raise AuthoringError("research_authoring.decision_cutoff_invalid")

    @property
    def is_resolvable(self) -> bool:
        """Report whether a request names an actual market phase.

        Returns:
            False for a date-only UNSPECIFIED cutoff; True for a declared phase.
        """
        return self.phase is not MarketPhase.UNSPECIFIED


class SessionRequest(_Contract):
    """An unresolved calendar request; the Host compiles it to exact sessions."""

    start: date
    end: date
    as_of: DecisionCutoff

    @model_validator(mode="before")  # type: ignore[untyped-decorator]
    @classmethod
    def coerce_cutoff(cls, value: object) -> object:
        """Normalize a mapping's as_of value through the decision-cutoff contract.

        Args:
            value: Incoming session request, with a typed, mapped, bare-date or ISO cutoff.

        Returns:
            Mapping with a typed cutoff when as_of is present; otherwise the original value.

        Raises:
            AuthoringError: The cutoff cannot be interpreted.
        """
        if isinstance(value, Mapping) and "as_of" in value:
            payload = dict(value)
            payload["as_of"] = DecisionCutoff.coerce(payload["as_of"])
            return payload
        return value

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_range(self) -> Self:
        """Require an ordered requested range ending no later than the cutoff session.

        Returns:
            This request after date-range validation.

        Raises:
            AuthoringError: The start follows the end or the cutoff session precedes the end.
        """
        if self.start > self.end:
            raise AuthoringError("research_authoring.session_range_invalid")
        if self.as_of.session < self.end:
            raise AuthoringError("research_authoring.as_of_precedes_end")
        return self


class ExecutionBudget(_Contract):
    """Bound admitted candidate count and planned numerical work.

    Attributes:
        maximum_candidates: Maximum candidate count declared by the request.
        maximum_numerical_calls: Maximum numerical-call count admitted for the execution plan.
    """

    maximum_candidates: int = Field(ge=1, le=4096)
    maximum_numerical_calls: int = Field(ge=1, le=1_000_000)


class DeterminismPolicy(_Contract):
    """Declare the seed, thread bound and compulsory offline execution policy.

    Attributes:
        seed: Nonnegative reproducibility seed.
        thread_limit: Requested numerical thread limit, subject to installed adapter requirements.
        network_disabled: Required True value; research execution remains held offline.
    """

    seed: int = Field(ge=0, le=2**31 - 1)
    thread_limit: int = Field(ge=1, le=64)
    network_disabled: Literal[True] = True


class ResearchExperimentEnvelope(_Contract):
    """The common authoring boundary; Desk methodology lives outside it."""

    kind: str = Field(pattern=_HANDLE)
    schema_id: str = Field(pattern=_HANDLE)
    data_snapshot_handle: str = Field(pattern=_SNAPSHOT_HANDLE)
    universe_handle: str = Field(pattern=_HANDLE)
    sessions: SessionRequest
    budget: ExecutionBudget
    determinism: DeterminismPolicy
    output_workspace: str = Field(min_length=1, max_length=512)
    baseline_workspace: str | None = Field(default=None, max_length=512)
    publication_intent: Literal["DEVELOPMENT_EVIDENCE_ONLY"] = "DEVELOPMENT_EVIDENCE_ONLY"
    envelope_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal an envelope from an authored mapping.

        Nested sections are validated *before* the identity is computed, so the
        hash covers admitted content rather than whatever shape the document
        happened to use.
        """
        draft = dict(values)
        draft.pop("envelope_hash", None)
        for field, contract in (
            ("sessions", SessionRequest),
            ("budget", ExecutionBudget),
            ("determinism", DeterminismPolicy),
        ):
            if field not in draft:
                raise AuthoringError(f"research_authoring.envelope_section_missing:{field}")
            draft[field] = contract.model_validate(draft[field])
        provisional = cls.model_construct(**draft, envelope_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"envelope_hash"})
        return cls(**draft, envelope_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require nonoverlapping declared workspaces and the exact envelope identity.

        Returns:
            This envelope after workspace-shape and canonical-hash validation.

        Raises:
            AuthoringError: Declared workspaces are identical/nested or the envelope hash is
                invalid.
        """
        _reject_overlapping_workspaces(self.output_workspace, self.baseline_workspace)
        if self.envelope_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"envelope_hash"})
        ):
            raise AuthoringError("research_authoring.envelope_identity_invalid")
        return self


def _reject_overlapping_workspaces(output: str, baseline: str | None) -> None:
    if baseline is None:
        return
    left = PurePosixPath(output.replace("\\", "/"))
    right = PurePosixPath(baseline.replace("\\", "/"))
    if left == right:
        raise AuthoringError("research_authoring.workspace_identical")
    if left.is_relative_to(right) or right.is_relative_to(left):
        raise AuthoringError("research_authoring.workspace_nested")


class ResolvedListingScope(_Contract):
    """A dated calculation axis resolved by the Host, not a caller-supplied roster."""

    first_session: date
    last_session: date
    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_scope(self) -> Self:
        if self.first_session > self.last_session or len(set(self.ordered_listing_ids)) != len(
            self.ordered_listing_ids
        ):
            raise AuthoringError("research_authoring.listing_scope_invalid")
        return self


class ResolvedResearchAuthority(_Contract):
    """What the opaque handles in an envelope actually resolved to.

    The envelope carries a *request*: two handles and an unresolved date range.
    Nothing in it can be trusted as identity. This is the Host's answer, read
    from artifacts that already exist -- a published Panel snapshot, the current
    quality-filtered research manifest, and the Panel's own session axis. A
    handle that resolves to nothing fails here, before any Desk compiler runs
    and long before any numerical call.

    ``sessions`` come from the Panel semantic index rather than a generated
    calendar. A synthetic weekday calendar can name sessions that no Panel ever
    materialized, which makes a Program look admissible and then fail, or worse,
    silently resolve to a different axis than the data it will read.
    """

    data_snapshot_handle: str = Field(pattern=_SNAPSHOT_HANDLE)
    universe_handle: str = Field(pattern=_HANDLE)
    panel_snapshot_hash: str | None = Field(
        default=None, pattern=_HASH, exclude_if=lambda v: v is None
    )
    panel_manifest_ref: str | None = Field(
        default=None, min_length=1, max_length=512, exclude_if=lambda v: v is None
    )
    training_snapshot_hash: str | None = Field(
        default=None, pattern=_HASH, exclude_if=lambda v: v is None
    )
    training_manifest_ref: str | None = Field(
        default=None, min_length=1, max_length=512, exclude_if=lambda v: v is None
    )
    universe_revision_sha256: str = Field(pattern=_HASH)
    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)
    sessions: tuple[date, ...] = Field(min_length=1)
    source_watermark_hash: str = Field(pattern=_HASH)
    execution_input_binding_hash: str | None = Field(
        default=None, pattern=_HASH, exclude_if=lambda v: v is None
    )
    listing_scopes: tuple[ResolvedListingScope, ...] = Field(default=(), exclude_if=lambda v: not v)
    listing_sample: Literal["SECTOR_SHARES"] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    """How ``ordered_listing_ids`` was sampled from the Panel's names, for an exploration
    sample: each Sector's share (binding plan, decision 4). Absent for the whole Panel, and
    from a sample drawn before samples kept each Sector's share."""
    authority_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal source authority from contract values, replacing any supplied self hash.

        Returns:
            Validated contract with authority_hash derived from its JSON content excluding that
            field.

        Raises:
            AuthoringError: The resulting contract violates its identity or consistency checks.
            pydantic.ValidationError: Values violate the contract's structural constraints.
        """
        draft = dict(values)
        draft.pop("authority_hash", None)
        if "listing_scopes" in draft:
            draft["listing_scopes"] = tuple(
                ResolvedListingScope.model_validate(scope) for scope in draft["listing_scopes"]
            )
        provisional = cls.model_construct(**draft, authority_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"authority_hash"})
        return cls(**draft, authority_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Validate one exact source lane, ordered axes, scope coverage and authority hash.

        Returns:
            This authority after source-lane, sorted-session, unique-listing, scope-coverage and
            canonical-hash checks.

        Raises:
            AuthoringError: A source reference pair, axis, listing scope or canonical identity is
                invalid.
        """
        panel = self.panel_snapshot_hash is not None and self.panel_manifest_ref is not None
        training = (
            self.training_snapshot_hash is not None and self.training_manifest_ref is not None
        )
        if (
            panel == training
            or (self.panel_snapshot_hash is None) != (self.panel_manifest_ref is None)
            or (self.training_snapshot_hash is None) != (self.training_manifest_ref is None)
        ):
            raise AuthoringError("research_authoring.source_authority_shape_invalid")
        if self.sessions != tuple(sorted(set(self.sessions))):
            raise AuthoringError("research_authoring.resolved_sessions_unordered")
        if self.ordered_listing_ids != tuple(dict.fromkeys(self.ordered_listing_ids)):
            raise AuthoringError("research_authoring.resolved_listings_duplicated")
        if self.listing_scopes:
            covered: list[date] = []
            for scope in self.listing_scopes:
                wanted = set(scope.ordered_listing_ids)
                if (
                    scope.first_session not in self.sessions
                    or scope.last_session not in self.sessions
                    or tuple(v for v in self.ordered_listing_ids if v in wanted)
                    != scope.ordered_listing_ids
                ):
                    raise AuthoringError("research_authoring.listing_scope_invalid")
                covered.extend(
                    day for day in self.sessions if scope.first_session <= day <= scope.last_session
                )
            if tuple(covered) != self.sessions:
                raise AuthoringError("research_authoring.listing_scope_coverage_invalid")
        if self.authority_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"authority_hash"})
        ):
            raise AuthoringError("research_authoring.authority_identity_invalid")
        return self

    def for_scope(self, scope: ResolvedListingScope) -> Self:
        """Derive a sealed authority for one previously bound listing/session scope.

        Args:
            scope: Exact scope already present in this authority's listing_scopes.

        Returns:
            Resealed authority with the scope's listing axis and inclusive session interval.

        Raises:
            AuthoringError: The scope is not bound by this authority.
        """
        if scope not in self.listing_scopes:
            raise AuthoringError("research_authoring.listing_scope_unbound")
        return type(self).create(
            **{
                **self.model_dump(exclude={"authority_hash", "listing_scopes"}),
                "ordered_listing_ids": scope.ordered_listing_ids,
                "sessions": tuple(
                    day for day in self.sessions if scope.first_session <= day <= scope.last_session
                ),
            }
        )


class DeskProgramCompilation(_Contract):
    """What a Desk compiler returns: identity, never numerical results.

    A bare ``tuple[str, str]`` could not say which of its two hashes was which,
    and had nowhere to carry the implementation and parameter-domain identities
    that stop a changed method reusing old evidence under an unchanged
    identifier. Those are named fields here so the sealed Program consumes them.
    """

    desk_program_hash: str = Field(pattern=_HASH)
    catalog_hash: str = Field(pattern=_HASH)
    method_binding_hash: str = Field(pattern=_HASH)
    parameter_domain_hash: str = Field(pattern=_HASH)


class SealedResearchProgram(_Contract):
    """Immutable, content-addressed admission of one authored experiment.

    `desk_program_hash` is produced by the Desk compiler from resolved
    identities. The caller never supplies it, and the envelope never carries it.

    There is deliberately no ``actor_binding_hash`` field. Who submitted a
    document is provenance, not methodology: folding it into ``program_hash``
    would give the same experiment a different identity per submitter and make
    an Agent-authored program un-replayable by a human. The actor binding is
    sealed *over* ``program_hash`` instead, so the signature covers the Desk
    method content that ``envelope_hash`` alone excludes.
    """

    kind: str = Field(pattern=_HANDLE)
    envelope_hash: str = Field(pattern=_HASH)
    desk_program_hash: str = Field(pattern=_HASH)
    resolved_sessions: tuple[date, ...] = Field(min_length=1)
    catalog_hash: str = Field(pattern=_HASH)
    method_binding_hash: str = Field(pattern=_HASH)
    parameter_domain_hash: str = Field(pattern=_HASH)
    authority_hash: str = Field(pattern=_HASH)
    program_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal admitted program from contract values, replacing any supplied self hash.

        Returns:
            Validated contract with program_hash derived from its JSON content excluding that field.

        Raises:
            AuthoringError: The resulting contract violates its identity or consistency checks.
            pydantic.ValidationError: Values violate the contract's structural constraints.
        """
        draft = dict(values)
        draft.pop("program_hash", None)
        provisional = cls.model_construct(**draft, program_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"program_hash"})
        return cls(**draft, program_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require sorted unique resolved sessions and the program's canonical identity.

        Returns:
            This admitted program after session-axis and hash validation.

        Raises:
            AuthoringError: Sessions are unordered/duplicated or the program identity is invalid.
        """
        if self.resolved_sessions != tuple(sorted(set(self.resolved_sessions))):
            raise AuthoringError("research_authoring.resolved_sessions_unordered")
        if self.program_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"program_hash"})
        ):
            raise AuthoringError("research_authoring.program_identity_invalid")
        return self


class DeskExecutionResult(_Contract):
    """What a Desk executor reports after running one Program.

    ``desk_input_binding_hash`` is deliberately opaque here. Every Desk seals
    *something* describing the exact inputs its numerical path was allowed to
    read, but what belongs in that seal is Desk methodology: Risk binds a return
    surface, an epoch and an ordered listing axis, and another Desk will bind
    something else entirely. Carrying the hash and nothing else keeps the generic
    layer able to require the binding, and require it to match, without acquiring
    any Desk's field semantics -- the Desk's own verifier owns those.
    """

    disposition: Literal["COMPUTED", "REUSED_EXACT"]
    artifact_uris: tuple[str, ...] = Field(min_length=1)
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    numerical_call_count: int = Field(ge=0)
    desk_input_binding_hash: str = Field(pattern=_HASH)


class ResearchExecutionEvidence(_Contract):
    """Immutable record of what one Program actually did.

    ``identity_class`` is stated rather than implied. Development evidence is
    never publication evidence, and a record that does not say so is one careless
    reader away from being treated as an admitted result.
    """

    kind: str = Field(pattern=_HANDLE)
    program_hash: str = Field(pattern=_HASH)
    desk_program_hash: str = Field(pattern=_HASH)
    method_binding_hash: str = Field(pattern=_HASH)
    authority_hash: str = Field(pattern=_HASH)
    identity_class: Literal["DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"] = (
        "DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"
    )
    disposition: Literal["COMPUTED", "REUSED_EXACT", "VERIFIED_DURABLE_GRAPH_READBACK"]
    numerical_call_count: int = Field(ge=0)
    artifact_uris: tuple[str, ...] = Field(min_length=1)
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    desk_input_binding_hash: str = Field(pattern=_HASH)
    evidence_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal development execution evidence, replacing any supplied self hash.

        Returns:
            Validated contract with evidence_hash derived from its JSON content excluding that
            field.

        Raises:
            AuthoringError: The resulting contract violates its identity or consistency checks.
            pydantic.ValidationError: Values violate the contract's structural constraints.
        """
        draft = dict(values)
        draft.pop("evidence_hash", None)
        provisional = cls.model_construct(**draft, evidence_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"evidence_hash"})
        return cls(**draft, evidence_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require zero numerical calls for readback/reuse claims and an exact evidence hash.

        Returns:
            This development evidence after disposition/count and canonical-hash validation.

        Raises:
            AuthoringError: Reuse/readback claims include numerical calls or the evidence hash is
                invalid.
        """
        if (
            self.disposition in {"REUSED_EXACT", "VERIFIED_DURABLE_GRAPH_READBACK"}
            and self.numerical_call_count != 0
        ):
            raise AuthoringError("research_authoring.reuse_claimed_with_numerical_calls")
        if self.evidence_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"evidence_hash"})
        ):
            raise AuthoringError("research_authoring.evidence_identity_invalid")
        return self


class NumericalCallRecorder(Protocol):
    """Narrow observer of numerical work; it can only count, never decide.

    Deliberately not a catalog. A counting *catalog* in the product would be a
    second installation path that only tests use, and the difference between it
    and the real one is exactly the kind of drift this record is meant to catch.
    """

    def record(self, *, capability: str) -> None:
        """Record one numerical call without deciding methodology or admission.

        Args:
            capability: Installed numerical capability identifier reported by the executor.
        """
        ...


class ResearchAuthorityResolver(Protocol):
    """Host-owned resolution of opaque handles into real, existing identities."""

    def resolve(self, envelope: ResearchExperimentEnvelope) -> ResolvedResearchAuthority:
        """Resolve request handles into exact existing data/session authority.

        Args:
            envelope: Validated request whose handles and date range require deterministic
                resolution.

        Returns:
            Exact source authority for subsequent Desk compilation and execution.
        """
        ...


class DeskExperimentCompiler(Protocol):
    """One Desk's compiler from an authored document to a typed Program."""

    kind: str

    def compile_desk_program(
        self,
        *,
        envelope: ResearchExperimentEnvelope,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
    ) -> DeskProgramCompilation:
        """Compile Desk methodology against already resolved authority without numerical work.

        Args:
            envelope: Validated common research request.
            document: Desk-owned authored methodology document.
            authority: Exact Host-resolved data/session authority.

        Returns:
            Desk program, catalog, method and parameter-domain identities for admission.
        """
        ...


class DeskExperimentExecutor(Protocol):
    """One Desk's executor for an already-sealed Program."""

    kind: str

    def execute(
        self,
        *,
        program: SealedResearchProgram,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
        output_workspace: Path,
        recorder: NumericalCallRecorder | None = None,
    ) -> DeskExecutionResult:
        """Execute an admitted Desk program against its exact resolved inputs.

        Args:
            program: Sealed program admitted before execution.
            document: Desk-owned methodology document used for the admitted program.
            authority: Exact source authority bound by admission.
            output_workspace: Caller-owned development workspace for durable evidence artifacts.
            recorder: Optional narrow numerical-call observer.

        Returns:
            Desk execution disposition, artifacts, sessions, numerical count and exact input
            binding.
        """
        ...


class DeskEvidenceVerifier(Protocol):
    """Read one Desk's published artifacts back by hash, recursively.

    Kept separate from the executor on purpose: replay must be able to verify
    without resolving anything that could compute. A Desk whose verifier is not
    installed cannot be replayed, which is the intended outcome -- reporting
    exact reuse over artifacts nobody read is worse than refusing.

    The sealed Program and the whole evidence record are passed, not a bare list
    of URIs. A verifier handed only URIs can confirm that some artifacts exist
    and hash to their own names; it cannot confirm they are *these* artifacts,
    belonging to *this* Program, describing *this* authority. Two valid runs
    would cross-check clean against each other's children. Verification of a
    content-addressed graph is a statement about lineage, so the verifier needs
    the root of that lineage.

    ``authority`` is present for live-source replay. Explicit sealed-Program
    readback passes ``None`` so a verifier can prove a fully self-contained
    durable graph without reopening source data. A Desk whose graph needs an
    independent live Panel/universe/session comparison must refuse ``None``;
    Desks whose published graph already binds those inputs by independently
    verified child identities may read it back source-free.
    """

    kind: str

    def verify(
        self,
        *,
        program: SealedResearchProgram,
        evidence: ResearchExecutionEvidence,
        authority: ResolvedResearchAuthority | None,
        output_workspace: Path,
    ) -> None:
        """Verify durable Desk evidence against its admitted program and optional authority.

        Args:
            program: Exact admitted program whose evidence is being checked.
            evidence: Execution record and durable artifact references to verify.
            authority: Resolved source authority when available for the verification path.
            output_workspace: Workspace containing the durable evidence graph.
        """
        ...


__all__ = [
    "AuthoringError",
    "DecisionCutoff",
    "DeskEvidenceVerifier",
    "DeskExecutionResult",
    "DeskExperimentCompiler",
    "DeskExperimentExecutor",
    "DeskProgramCompilation",
    "DeskSection",
    "DeterminismPolicy",
    "ExecutionBudget",
    "MarketPhase",
    "NumericalCallRecorder",
    "ResearchAuthorityResolver",
    "ResearchExecutionEvidence",
    "ResearchExperimentEnvelope",
    "ResolvedResearchAuthority",
    "SealedResearchProgram",
    "SessionRequest",
    "refuse_unknown_section_keys",
]
