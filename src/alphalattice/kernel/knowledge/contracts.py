"""Strict immutable contracts for system Knowledge resources."""

from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from alphalattice.kernel.knowledge.enums import (
    ArgumentValueKind,
    ResolutionStatus,
    ResourceKind,
    ResourceNamespace,
)
from alphalattice.kernel.shared_kernel.domain.base import DomainModel, Sha256Hex, ShortString
from alphalattice.kernel.shared_kernel.domain.models import TypedFailure
from alphalattice.kernel.shared_kernel.domain.serialization import canonical_json_bytes, sha256_hex

StableId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*$",
    ),
]
Revision = Annotated[
    str,
    StringConstraints(pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"),
]
HttpsUrl = Annotated[str, StringConstraints(pattern=r"^https://[^\s]+$")]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
type JsonAtom = None | bool | int | FiniteFloat | str
type ArgumentValue = JsonAtom | tuple[JsonAtom, ...]


def _validate_relative_posix(value: str, kind: ResourceKind) -> None:
    if "\\" in value or any(ord(char) < 32 for char in value):
        raise ValueError("resource path must use printable POSIX characters")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("resource path must be normalized and relative")
    owner_directory = {
        ResourceKind.KNOWLEDGE: "knowledge",
        ResourceKind.SKILL: "skills",
        ResourceKind.PROMPT: "prompts",
        ResourceKind.PACKAGE_ARG_SPEC: "package-args",
    }[kind]
    if path.parts[0] != owner_directory or path.suffix != ".json":
        raise ValueError("resource path does not match its kind")


class KnowledgeRef(DomainModel):
    """Bind knowledge guidance to an exact namespaced revision and logical identity.

    The resource kind is fixed by this contract. Resolution also checks the installed catalog;
    constructing a reference does not prove availability.
    """

    resource_kind: Literal[ResourceKind.KNOWLEDGE] = ResourceKind.KNOWLEDGE
    namespace: ResourceNamespace
    stable_id: StableId
    revision: Revision
    logical_hash: Sha256Hex


class SkillRef(DomainModel):
    """Bind a research skill to an exact namespaced revision and logical identity.

    The resource kind is fixed by this contract. Resolution also checks the installed catalog;
    constructing a reference does not prove availability.
    """

    resource_kind: Literal[ResourceKind.SKILL] = ResourceKind.SKILL
    namespace: ResourceNamespace
    stable_id: StableId
    revision: Revision
    logical_hash: Sha256Hex


class PromptRef(DomainModel):
    """Bind a research prompt to an exact namespaced revision and logical identity.

    The resource kind is fixed by this contract. Resolution also checks the installed catalog;
    constructing a reference does not prove availability.
    """

    resource_kind: Literal[ResourceKind.PROMPT] = ResourceKind.PROMPT
    namespace: ResourceNamespace
    stable_id: StableId
    revision: Revision
    logical_hash: Sha256Hex


class PackageArgSpecRef(DomainModel):
    """Bind a package argument specification to an exact namespaced revision and logical identity.

    The resource kind is fixed by this contract. Resolution also checks the installed catalog;
    constructing a reference does not prove availability.
    """

    resource_kind: Literal[ResourceKind.PACKAGE_ARG_SPEC] = ResourceKind.PACKAGE_ARG_SPEC
    namespace: ResourceNamespace
    stable_id: StableId
    revision: Revision
    logical_hash: Sha256Hex


type ResourceRef = Annotated[
    KnowledgeRef | SkillRef | PromptRef | PackageArgSpecRef,
    Field(discriminator="resource_kind"),
]


class CatalogEntry(DomainModel):
    """Bind one installed JSON resource to its path, file digest and logical identity.

    The namespace is SYSTEM. The relative path is validated for its resource kind; the file SHA-256
    binds bytes while logical_hash binds the resource payload.
    """

    resource_kind: ResourceKind
    namespace: Literal[ResourceNamespace.SYSTEM] = ResourceNamespace.SYSTEM
    stable_id: StableId
    revision: Revision
    relative_path: ShortString
    media_type: Literal["application/json"] = "application/json"
    logical_hash: Sha256Hex
    file_sha256: Sha256Hex

    @model_validator(mode="after")
    def validate_path(self) -> CatalogEntry:
        """Require a confined relative POSIX resource path for the declared kind.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: The path is absolute, traversing or inconsistent with the resource kind.
        """
        _validate_relative_posix(self.relative_path, self.resource_kind)
        return self


class SystemResourceManifest(DomainModel):
    """List the exact system resources admitted by one catalog revision.

    Entries are nonempty, canonically ordered and unique by kind, stable identifier and revision.
    Each relative path occurs once.
    """

    namespace: Literal[ResourceNamespace.SYSTEM] = ResourceNamespace.SYSTEM
    catalog_id: Literal["alphalattice.system-resources"] = "alphalattice.system-resources"
    revision: Revision
    entries: tuple[CatalogEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entries(self) -> SystemResourceManifest:
        """Require ordered unique resource identities and distinct catalog paths.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: Entry identities are unordered/duplicated or resource paths repeat.
        """
        keys = tuple(
            (entry.resource_kind.value, entry.stable_id, entry.revision) for entry in self.entries
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("catalog entries must be sorted and unique")
        paths = tuple(entry.relative_path for entry in self.entries)
        if len(paths) != len(set(paths)):
            raise ValueError("catalog paths must be unique")
        return self


class KnowledgeResource(DomainModel):
    """Declare system guidance with its scope, owners and official sources.

    Guidance is accompanied by explicit assumptions and warnings; owner references identify the
    deterministic responsibilities to which it applies.
    """

    resource_kind: Literal[ResourceKind.KNOWLEDGE] = ResourceKind.KNOWLEDGE
    namespace: Literal[ResourceNamespace.SYSTEM] = ResourceNamespace.SYSTEM
    stable_id: StableId
    revision: Revision
    title: ShortString
    scope: ShortString
    owner_refs: tuple[ShortString, ...] = Field(min_length=1)
    guidance: tuple[ShortString, ...] = Field(min_length=1)
    assumptions: tuple[ShortString, ...] = ()
    warnings: tuple[ShortString, ...] = ()
    official_sources: tuple[HttpsUrl, ...] = Field(min_length=1)


class SkillResource(DomainModel):
    """Declare a system research skill with admitted tools and evidence rules.

    Required evidence, stop rules and downweight rules accompany its tool allowlist and
    knowledge/prompt references. The declaration does not execute tools or grant publication
    authority.
    """

    resource_kind: Literal[ResourceKind.SKILL] = ResourceKind.SKILL
    namespace: Literal[ResourceNamespace.SYSTEM] = ResourceNamespace.SYSTEM
    stable_id: StableId
    revision: Revision
    title: ShortString
    owner: ShortString
    required_evidence: tuple[ShortString, ...] = Field(min_length=1)
    tool_allowlist: tuple[StableId, ...] = Field(min_length=1)
    stop_rules: tuple[ShortString, ...] = Field(min_length=1)
    downweight_rules: tuple[ShortString, ...] = Field(min_length=1)
    knowledge_refs: tuple[StableId, ...] = Field(min_length=1)
    prompt_refs: tuple[StableId, ...] = Field(min_length=1)


class PromptResource(DomainModel):
    """Declare system prompt instructions, prohibited behaviors and owner references.

    The stable identifier and revision identify installed instructions used by the catalog; the
    payload carries no provider response or research result.
    """

    resource_kind: Literal[ResourceKind.PROMPT] = ResourceKind.PROMPT
    namespace: Literal[ResourceNamespace.SYSTEM] = ResourceNamespace.SYSTEM
    stable_id: StableId
    revision: Revision
    title: ShortString
    instructions: tuple[ShortString, ...] = Field(min_length=1)
    prohibited_behaviors: tuple[ShortString, ...] = Field(min_length=1)
    owner_refs: tuple[ShortString, ...] = Field(min_length=1)


class PackageVersionPin(DomainModel):
    """Name the exact installed distribution version required by an argument specification."""

    distribution: StableId
    version: ShortString


class ArgumentRule(DomainModel):
    """Constrain one immutable package argument by kind, default, choices and numeric bounds.

    Required arguments cannot also have defaults. Bounds may be inclusive or exclusive; defaults and
    every allowed value must satisfy the declared rule.
    """

    name: ShortString
    value_kind: ArgumentValueKind
    required: bool = False
    has_default: bool = False
    default: ArgumentValue = None
    allowed_values: tuple[ArgumentValue, ...] = ()
    minimum: FiniteFloat | None = None
    maximum: FiniteFloat | None = None
    minimum_inclusive: bool = True
    maximum_inclusive: bool = True

    @model_validator(mode="after")
    def validate_rule(self) -> ArgumentRule:
        """Require consistent defaults, ordered bounds and unique admitted values.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: Required/default flags, bound order, defaults or allowed values violate the
                rule.
        """
        if self.required and self.has_default:
            raise ValueError("required argument cannot declare a default")
        if not self.has_default and self.default is not None:
            raise ValueError("argument without a default cannot store a default value")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("argument minimum cannot exceed maximum")
        if self.has_default and not _argument_value_matches_rule(self.default, self):
            raise ValueError("argument default does not satisfy its rule")
        encoded_allowed = tuple(canonical_json_bytes(value) for value in self.allowed_values)
        if len(encoded_allowed) != len(set(encoded_allowed)):
            raise ValueError("allowed argument values must be unique")
        if any(not _argument_value_matches_rule(value, self) for value in self.allowed_values):
            raise ValueError("allowed argument value does not satisfy its rule")
        return self


class PackageArgSpec(DomainModel):
    """Declare the exact package callable, pinned versions and admitted argument rules.

    Argument names and distribution pins are unique. Owner policy and official sources explain the
    callable contract; the resource itself does not execute it.
    """

    resource_kind: Literal[ResourceKind.PACKAGE_ARG_SPEC] = ResourceKind.PACKAGE_ARG_SPEC
    namespace: Literal[ResourceNamespace.SYSTEM] = ResourceNamespace.SYSTEM
    stable_id: StableId
    revision: Revision
    title: ShortString
    primary_distribution: PackageVersionPin
    companion_distributions: tuple[PackageVersionPin, ...] = ()
    callable: ShortString
    arguments: tuple[ArgumentRule, ...]
    interactions: tuple[ShortString, ...] = ()
    assumptions: tuple[ShortString, ...] = ()
    warnings: tuple[ShortString, ...] = ()
    failure_modes: tuple[ShortString, ...] = Field(min_length=1)
    owner_policy_ref: ShortString
    official_source_url: HttpsUrl

    @model_validator(mode="after")
    def validate_arguments(self) -> PackageArgSpec:
        """Require sorted unique argument names and unique distribution pins.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: Arguments are unordered/duplicated or a distribution is pinned more than
                once.
        """
        names = tuple(rule.name for rule in self.arguments)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("package argument rules must be sorted and unique")
        pins = (self.primary_distribution, *self.companion_distributions)
        distributions = tuple(pin.distribution for pin in pins)
        if len(distributions) != len(set(distributions)):
            raise ValueError("package distributions must be unique")
        return self


type ResourcePayload = Annotated[
    KnowledgeResource | SkillResource | PromptResource | PackageArgSpec,
    Field(discriminator="resource_kind"),
]


class ResourceResolution(DomainModel):
    """Represent an exact resource payload or a typed unsupported outcome.

    RESOLVED requires a matching SYSTEM reference, entry and payload without failure. UNSUPPORTED
    contains failure and no entry or payload.
    """

    status: ResolutionStatus
    reference: ResourceRef
    entry: CatalogEntry | None = None
    resource: ResourcePayload | None = None
    failure: TypedFailure | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> ResourceResolution:
        """Require a matching resolved payload or a failure without payload.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: Status, payload, namespace, reference identity or logical hash are
                inconsistent.
        """
        if self.status is ResolutionStatus.RESOLVED:
            if self.entry is None or self.resource is None or self.failure is not None:
                raise ValueError("resolved resource requires entry and payload only")
            expected_identity = (
                self.reference.resource_kind,
                self.reference.stable_id,
                self.reference.revision,
            )
            if (
                self.reference.namespace is not ResourceNamespace.SYSTEM
                or (
                    self.entry.resource_kind,
                    self.entry.stable_id,
                    self.entry.revision,
                )
                != expected_identity
            ):
                raise ValueError("resolved entry does not match its reference")
            if (
                self.resource.resource_kind,
                self.resource.stable_id,
                self.resource.revision,
            ) != expected_identity:
                raise ValueError("resolved resource does not match its reference")
            if self.reference.logical_hash != self.entry.logical_hash:
                raise ValueError("resolved hash does not match its reference")
        elif self.entry is not None or self.resource is not None or self.failure is None:
            raise ValueError("unsupported resource requires failure and no payload")
        return self


class PackageArgumentAudit(DomainModel):
    """Seal one package-argument resolution with installed versions and normalized values.

    Names and version pairs are canonically ordered and unique. Resolved audits contain no failure;
    unsupported audits contain failure and no normalized arguments. audit_hash binds the complete
    outcome.
    """

    status: ResolutionStatus
    reference: PackageArgSpecRef
    submitted_argument_names: tuple[ShortString, ...]
    installed_versions: tuple[tuple[StableId, ShortString], ...]
    normalized_arguments: tuple[tuple[ShortString, ArgumentValue], ...] = ()
    failure: TypedFailure | None = None
    audit_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_outcome_and_hash(self) -> PackageArgumentAudit:
        """Require ordered audit values, a consistent outcome and the exact audit hash.

        Returns:
            This contract after validating the declared consistency rules.

        Raises:
            ValueError: Names, versions, normalized values, failure disposition or audit_hash are
                invalid.
        """
        if self.submitted_argument_names != tuple(sorted(set(self.submitted_argument_names))):
            raise ValueError("submitted argument names must be sorted and unique")
        installed_names = tuple(name for name, _version in self.installed_versions)
        if self.installed_versions != tuple(sorted(self.installed_versions)) or len(
            installed_names
        ) != len(set(installed_names)):
            raise ValueError("installed versions must be sorted and unique")
        if self.status is ResolutionStatus.RESOLVED:
            if self.failure is not None:
                raise ValueError("resolved argument audit cannot contain failure")
        elif self.normalized_arguments or self.failure is None:
            raise ValueError("unsupported argument audit requires failure and no arguments")
        normalized_names = tuple(name for name, _value in self.normalized_arguments)
        if self.normalized_arguments != tuple(sorted(self.normalized_arguments)) or len(
            normalized_names
        ) != len(set(normalized_names)):
            raise ValueError("normalized arguments must be sorted and unique")
        expected = _package_argument_audit_hash(
            status=self.status,
            reference=self.reference,
            submitted_argument_names=self.submitted_argument_names,
            installed_versions=self.installed_versions,
            normalized_arguments=self.normalized_arguments,
            failure=self.failure,
        )
        if self.audit_hash != expected:
            raise ValueError("package argument audit hash is invalid")
        return self


def _package_argument_audit_hash(
    *,
    status: ResolutionStatus,
    reference: PackageArgSpecRef,
    submitted_argument_names: tuple[str, ...],
    installed_versions: tuple[tuple[str, str], ...],
    normalized_arguments: tuple[tuple[str, ArgumentValue], ...],
    failure: TypedFailure | None,
) -> str:
    return sha256_hex(
        canonical_json_bytes(
            {
                "status": status,
                "reference": reference,
                "submitted_argument_names": submitted_argument_names,
                "installed_versions": installed_versions,
                "normalized_arguments": normalized_arguments,
                "failure": failure,
            }
        )
    )


def _matches_argument_kind(value: ArgumentValue, kind: ArgumentValueKind) -> bool:
    if kind is ArgumentValueKind.BOOLEAN:
        return isinstance(value, bool)
    if kind is ArgumentValueKind.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if kind is ArgumentValueKind.NUMBER:
        return isinstance(value, int | float) and not isinstance(value, bool)
    if kind is ArgumentValueKind.STRING:
        return isinstance(value, str)
    if kind is ArgumentValueKind.STRING_OR_NULL:
        return value is None or isinstance(value, str)
    if not isinstance(value, tuple):
        return False
    if kind is ArgumentValueKind.INTEGER_LIST:
        return all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    if kind is ArgumentValueKind.NUMBER_LIST:
        return all(isinstance(item, int | float) and not isinstance(item, bool) for item in value)
    return all(isinstance(item, str) for item in value)


def _argument_numbers(value: ArgumentValue) -> tuple[float, ...]:
    values = value if isinstance(value, tuple) else (value,)
    if not values:
        return ()
    result: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int | float):
            return ()
        number = float(item)
        if not math.isfinite(number):
            return ()
        result.append(number)
    return tuple(result)


def _argument_value_matches_rule(value: ArgumentValue, rule: ArgumentRule) -> bool:
    """Validate one immutable JSON-native argument against its package rule."""

    if not _matches_argument_kind(value, rule.value_kind):
        return False
    if rule.allowed_values:
        encoded = canonical_json_bytes(value)
        if all(canonical_json_bytes(allowed) != encoded for allowed in rule.allowed_values):
            return False
    numbers = _argument_numbers(value)
    if rule.minimum is not None:
        if not numbers:
            return False
        if rule.minimum_inclusive:
            if any(number < rule.minimum for number in numbers):
                return False
        elif any(number <= rule.minimum for number in numbers):
            return False
    if rule.maximum is not None:
        if not numbers:
            return False
        if rule.maximum_inclusive:
            if any(number > rule.maximum for number in numbers):
                return False
        elif any(number >= rule.maximum for number in numbers):
            return False
    return True


__all__ = [
    "ArgumentRule",
    "ArgumentValue",
    "CatalogEntry",
    "KnowledgeRef",
    "KnowledgeResource",
    "PackageArgSpec",
    "PackageArgSpecRef",
    "PackageArgumentAudit",
    "PackageVersionPin",
    "PromptRef",
    "PromptResource",
    "ResourcePayload",
    "ResourceRef",
    "ResourceResolution",
    "SkillRef",
    "SkillResource",
    "SystemResourceManifest",
]
