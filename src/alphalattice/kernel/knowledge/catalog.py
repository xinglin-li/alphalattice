"""Read-only loader and resolver for packaged system Knowledge resources."""

from __future__ import annotations

import math
from collections.abc import Mapping
from importlib import metadata, resources
from importlib.resources.abc import Traversable
from pathlib import PurePosixPath
from types import MappingProxyType

from pydantic import JsonValue

from alphalattice.kernel.knowledge.contracts import (
    ArgumentValue,
    CatalogEntry,
    KnowledgeRef,
    KnowledgeResource,
    PackageArgSpec,
    PackageArgSpecRef,
    PackageArgumentAudit,
    PromptRef,
    PromptResource,
    ResourcePayload,
    ResourceRef,
    ResourceResolution,
    SkillRef,
    SkillResource,
    SystemResourceManifest,
    _argument_value_matches_rule,
    _package_argument_audit_hash,
)
from alphalattice.kernel.knowledge.enums import (
    ResolutionStatus,
    ResourceKind,
    ResourceNamespace,
)
from alphalattice.kernel.knowledge.errors import KnowledgeError, knowledge_failure
from alphalattice.kernel.shared_kernel.domain.models import TypedFailure
from alphalattice.kernel.shared_kernel.domain.serialization import (
    canonical_json_bytes,
    parse_model,
    sha256_hex,
)

type ResourceKey = tuple[ResourceKind, str, str]


def _resource_key(kind: ResourceKind, stable_id: str, revision: str) -> ResourceKey:
    return (kind, stable_id, revision)


def _cause_chain(exc: BaseException) -> tuple[str, ...]:
    return (f"{type(exc).__name__}: {exc}",)


def _parse_resource(entry: CatalogEntry, data: bytes) -> ResourcePayload:
    try:
        if entry.resource_kind is ResourceKind.KNOWLEDGE:
            return parse_model(data, KnowledgeResource)
        if entry.resource_kind is ResourceKind.SKILL:
            return parse_model(data, SkillResource)
        if entry.resource_kind is ResourceKind.PROMPT:
            return parse_model(data, PromptResource)
        return parse_model(data, PackageArgSpec)
    except Exception as exc:
        raise KnowledgeError(
            f"declared resource is not valid: {entry.relative_path}",
            code="knowledge.resource_integrity",
            cause_chain=_cause_chain(exc),
        ) from exc


def _unsupported_resolution(
    reference: ResourceRef,
    failure: TypedFailure,
) -> ResourceResolution:
    return ResourceResolution(
        status=ResolutionStatus.UNSUPPORTED,
        reference=reference,
        failure=failure,
    )


def _to_argument_value(value: JsonValue) -> tuple[bool, ArgumentValue]:
    if isinstance(value, list):
        items: list[None | bool | int | float | str] = []
        for item in value:
            if isinstance(item, dict | list) or (
                item is not None and not isinstance(item, bool | int | float | str)
            ):
                return False, None
            if isinstance(item, float) and not math.isfinite(item):
                return False, None
            items.append(item)
        return True, tuple(items)
    if isinstance(value, dict) or (
        value is not None and not isinstance(value, bool | int | float | str)
    ):
        return False, None
    if isinstance(value, float) and not math.isfinite(value):
        return False, None
    return True, value


class SystemResourceCatalog:
    """Verified immutable view of resources shipped in the AlphaLattice wheel."""

    __slots__ = ("_entries", "_manifest", "_resources")

    def __init__(
        self,
        manifest: SystemResourceManifest,
        entries: Mapping[ResourceKey, CatalogEntry],
        verified_resources: Mapping[ResourceKey, ResourcePayload],
    ) -> None:
        """Hold a verified manifest and immutable copies of its resource mappings.

        Args:
            manifest: Manifest already checked by the catalog loader.
            entries: Catalog entries keyed by resource kind, stable identifier and revision.
            verified_resources: Corresponding parsed payloads whose bytes and logical identities
                were verified.
        """
        self._manifest = manifest
        self._entries = MappingProxyType(dict(entries))
        self._resources = MappingProxyType(dict(verified_resources))

    @property
    def manifest(self) -> SystemResourceManifest:
        """Return the manifest held by this catalog.

        Returns:
            Installed system-resource manifest; this access performs no resource reload.
        """
        return self._manifest

    @classmethod
    def load(cls) -> SystemResourceCatalog:
        """Load and verify the installed system-resource package.

        Returns:
            Catalog whose manifest, resource files and parsed payload identities have been verified.

        Raises:
            KnowledgeError: The installed package is unavailable or its manifest/resources fail
                verification.
        """
        try:
            root = resources.files("alphalattice.kernel.resources")
        except Exception as exc:
            raise KnowledgeError(
                "system resource package is unavailable",
                code="knowledge.dependency_unavailable",
                cause_chain=_cause_chain(exc),
            ) from exc
        return cls._load_from_root(root)

    @classmethod
    def _load_from_root(cls, root: Traversable) -> SystemResourceCatalog:
        catalog_file = root.joinpath("catalog.json")
        try:
            if not catalog_file.is_file():
                raise FileNotFoundError("catalog.json")
            manifest = parse_model(catalog_file.read_bytes(), SystemResourceManifest)
        except Exception as exc:
            raise KnowledgeError(
                "system resource catalog is invalid",
                code="knowledge.catalog_invalid",
                cause_chain=_cause_chain(exc),
            ) from exc

        entries: dict[ResourceKey, CatalogEntry] = {}
        verified: dict[ResourceKey, ResourcePayload] = {}
        for entry in manifest.entries:
            path = PurePosixPath(entry.relative_path)
            child = root.joinpath(*path.parts)
            try:
                if not child.is_file():
                    raise FileNotFoundError(entry.relative_path)
                data = child.read_bytes()
            except Exception as exc:
                raise KnowledgeError(
                    f"declared resource is unavailable: {entry.relative_path}",
                    code="knowledge.resource_integrity",
                    cause_chain=_cause_chain(exc),
                ) from exc
            if sha256_hex(data) != entry.file_sha256:
                raise KnowledgeError(
                    f"declared resource byte hash differs: {entry.relative_path}",
                    code="knowledge.resource_integrity",
                )
            resource = _parse_resource(entry, data)
            identity = (
                resource.resource_kind,
                resource.stable_id,
                resource.revision,
            )
            key = _resource_key(entry.resource_kind, entry.stable_id, entry.revision)
            if identity != key or resource.namespace is not ResourceNamespace.SYSTEM:
                raise KnowledgeError(
                    f"declared resource identity differs: {entry.relative_path}",
                    code="knowledge.resource_integrity",
                )
            if sha256_hex(canonical_json_bytes(resource)) != entry.logical_hash:
                raise KnowledgeError(
                    f"declared resource logical hash differs: {entry.relative_path}",
                    code="knowledge.resource_integrity",
                )
            entries[key] = entry
            verified[key] = resource
        return cls(manifest, entries, verified)

    def _resolve_exact(self, reference: ResourceRef) -> ResourceResolution:
        if reference.namespace is not ResourceNamespace.SYSTEM:
            return _unsupported_resolution(
                reference,
                knowledge_failure(
                    "system catalog cannot access another namespace",
                    code="knowledge.access_denied",
                ),
            )
        key = _resource_key(
            reference.resource_kind,
            reference.stable_id,
            reference.revision,
        )
        entry = self._entries.get(key)
        resource = self._resources.get(key)
        if entry is None or resource is None or reference.logical_hash != entry.logical_hash:
            return _unsupported_resolution(
                reference,
                knowledge_failure(
                    "system resource reference is unsupported",
                    code="knowledge.reference_unsupported",
                ),
            )
        return ResourceResolution(
            status=ResolutionStatus.RESOLVED,
            reference=reference,
            entry=entry,
            resource=resource,
        )

    @staticmethod
    def _installed_versions(
        spec: PackageArgSpec,
    ) -> tuple[tuple[tuple[str, str], ...], TypedFailure | None]:
        installed: list[tuple[str, str]] = []
        for pin in (spec.primary_distribution, *spec.companion_distributions):
            try:
                actual = metadata.version(pin.distribution)
            except metadata.PackageNotFoundError as exc:
                return (
                    tuple(sorted(installed)),
                    knowledge_failure(
                        f"required distribution is unavailable: {pin.distribution}",
                        code="knowledge.dependency_unavailable",
                        cause=exc,
                    ),
                )
            installed.append((pin.distribution, actual))
            if actual != pin.version:
                return (
                    tuple(sorted(installed)),
                    knowledge_failure(
                        f"required distribution version differs: {pin.distribution}",
                        code="knowledge.package_version_mismatch",
                    ),
                )
        return tuple(sorted(installed)), None

    def resolve(
        self,
        ref: KnowledgeRef | SkillRef | PromptRef | PackageArgSpecRef,
    ) -> ResourceResolution:
        """Resolve an exact system resource, including installed package-version checks.

        Args:
            ref: Kind, namespace, stable identifier, revision and logical hash to resolve.

        Returns:
            RESOLVED entry and payload, or UNSUPPORTED with a typed failure and no payload.
        """
        resolution = self._resolve_exact(ref)
        if resolution.status is ResolutionStatus.UNSUPPORTED:
            return resolution
        resource = resolution.resource
        if isinstance(resource, PackageArgSpec):
            _, failure = self._installed_versions(resource)
            if failure is not None:
                return _unsupported_resolution(ref, failure)
        return resolution

    @staticmethod
    def _audit(
        *,
        status: ResolutionStatus,
        reference: PackageArgSpecRef,
        submitted_names: tuple[str, ...],
        installed_versions: tuple[tuple[str, str], ...],
        normalized: tuple[tuple[str, ArgumentValue], ...] = (),
        failure: TypedFailure | None = None,
    ) -> PackageArgumentAudit:
        audit_hash = _package_argument_audit_hash(
            status=status,
            reference=reference,
            submitted_argument_names=submitted_names,
            installed_versions=installed_versions,
            normalized_arguments=normalized,
            failure=failure,
        )
        return PackageArgumentAudit(
            status=status,
            reference=reference,
            submitted_argument_names=submitted_names,
            installed_versions=installed_versions,
            normalized_arguments=normalized,
            failure=failure,
            audit_hash=audit_hash,
        )

    def validate_package_arguments(
        self,
        ref: PackageArgSpecRef,
        arguments: Mapping[str, JsonValue],
    ) -> PackageArgumentAudit:
        """Audit submitted package arguments against an exact installed specification.

        Unknown names, missing required values, incompatible values and version mismatches produce
        an unsupported audit. Declared defaults fill omitted arguments; optional arguments without
        defaults remain absent. No package callable is executed.

        Args:
            ref: Exact PackageArgSpec reference, including its logical identity.
            arguments: Submitted JSON-native values indexed by declared argument name.

        Returns:
            Sealed audit of sorted submitted names, installed versions and normalized arguments or
            failure.
        """
        submitted_names = tuple(sorted(arguments))
        resolution = self._resolve_exact(ref)
        if resolution.status is ResolutionStatus.UNSUPPORTED:
            failure = resolution.failure or knowledge_failure(
                "unsupported resolution omitted its failure",
                code="knowledge.invalid_reference",
            )
            return self._audit(
                status=ResolutionStatus.UNSUPPORTED,
                reference=ref,
                submitted_names=submitted_names,
                installed_versions=(),
                failure=failure,
            )
        spec = resolution.resource
        if not isinstance(spec, PackageArgSpec):
            failure = knowledge_failure(
                "reference does not identify a PackageArgSpec",
                code="knowledge.invalid_reference",
            )
            return self._audit(
                status=ResolutionStatus.UNSUPPORTED,
                reference=ref,
                submitted_names=submitted_names,
                installed_versions=(),
                failure=failure,
            )
        installed, version_failure = self._installed_versions(spec)
        if version_failure is not None:
            return self._audit(
                status=ResolutionStatus.UNSUPPORTED,
                reference=ref,
                submitted_names=submitted_names,
                installed_versions=installed,
                failure=version_failure,
            )

        rules = {rule.name: rule for rule in spec.arguments}
        if any(name not in rules for name in submitted_names):
            failure = knowledge_failure(
                "submitted package argument is not declared",
                code="knowledge.argument_unsupported",
            )
            return self._audit(
                status=ResolutionStatus.UNSUPPORTED,
                reference=ref,
                submitted_names=submitted_names,
                installed_versions=installed,
                failure=failure,
            )

        normalized_values: dict[str, ArgumentValue] = {}
        for name, rule in rules.items():
            if name in arguments:
                valid, value = _to_argument_value(arguments[name])
            elif rule.has_default:
                valid = True
                value = rule.default
            elif rule.required:
                valid = False
                value = None
            else:
                continue
            if not valid or not _argument_value_matches_rule(value, rule):
                failure = knowledge_failure(
                    f"package argument is unsupported: {name}",
                    code="knowledge.argument_unsupported",
                )
                return self._audit(
                    status=ResolutionStatus.UNSUPPORTED,
                    reference=ref,
                    submitted_names=submitted_names,
                    installed_versions=installed,
                    failure=failure,
                )
            normalized_values[name] = value

        normalized = tuple(sorted(normalized_values.items()))
        return self._audit(
            status=ResolutionStatus.RESOLVED,
            reference=ref,
            submitted_names=submitted_names,
            installed_versions=installed,
            normalized=normalized,
        )


__all__ = ["SystemResourceCatalog"]
