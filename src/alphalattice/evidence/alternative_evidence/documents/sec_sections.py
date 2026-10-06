"""Typed, optional boundary for SEC filing section discovery."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Protocol, cast, runtime_checkable

from alphalattice.kernel.live_evidence.errors import LiveEvidenceError


@runtime_checkable
class SecFilingSection(Protocol):
    """Minimum section projection consumed by Alternative Evidence."""

    @property
    def item_label(self) -> str:
        """Return the SEC Item label for this section."""
        ...

    @property
    def canonical_markdown(self) -> bytes:
        """Return the section's canonical Markdown bytes."""
        ...


@runtime_checkable
class SecSectionDiscovery(Protocol):
    """Model-neutral discovery result projected by an installed source capability."""

    @property
    def status(self) -> object:
        """Return the discovery availability status."""
        ...

    @property
    def unavailable_reason(self) -> str | None:
        """Explain why section discovery is unavailable, if applicable."""
        ...

    @property
    def sections(self) -> tuple[SecFilingSection, ...]:
        """Return the discovered filing sections in source order."""
        ...


SecSectionDiscoveryPort = Callable[..., SecSectionDiscovery]


def resolve_sec_section_discovery() -> SecSectionDiscoveryPort:
    """Resolve the optional outer capability lazily and fail closed when absent."""
    module = import_module("alphalattice.kernel.live_evidence.online_sources")
    capability = getattr(module, "discover_sec_filing_sections", None)
    if capability is None or not callable(capability):
        raise LiveEvidenceError(
            "SEC material-section discovery capability is not installed",
            code="live_evidence.source_unavailable",
            retryable=True,
        )
    return cast(SecSectionDiscoveryPort, capability)


def section_discovery_is_available(discovery: SecSectionDiscovery) -> bool:
    """Normalize external enum/string status without importing its concrete type."""
    status = discovery.status
    value = getattr(status, "value", status)
    return str(value).upper() == "AVAILABLE"


__all__ = [
    "SecFilingSection",
    "SecSectionDiscovery",
    "SecSectionDiscoveryPort",
    "resolve_sec_section_discovery",
    "section_discovery_is_available",
]
