"""Bind a development run's inputs, and hold the numerical path to them.

``build_historical_covariance_surface`` takes its session axis from exactly one
place -- ``return_reader.available_sessions(surface)`` -- and then selects the
last up to 1000 eligible formations from it. Nothing in that path ever sees the
Program. So a Program could seal 21 sessions while the estimator ran 1000, and
no type error, no assertion, and no published field would disagree.

That is fixed here rather than there. ``historical.py`` is inside the frozen Risk
source closure: editing it to accept a window would republish every frozen
covariance identity. Instead ``BoundedCausalReturnReader`` presents the reader
interface over exactly the bound window, so the unmodified selection arithmetic
inside the frozen path lands on precisely the requested formations. The frozen
path is constrained from outside without being touched.

The two watermarks are deliberately *not* compared to each other. The authority's
watermark and the surface's watermark are computed at different ``through`` dates
**and by different projections** -- ``canonical_hash(payload)`` against
``payload["watermark_hash"]`` -- so an equality between them would be a type-check
masquerading as a causal proof. Both are bound into the identity, and the causal
property that actually matters is validated through the authoritative reader:
the source state behind the surface must still hash to what the surface recorded.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Literal, Protocol, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.investment.risk_research.contracts import CausalRiskReturnSurface
from alphalattice.investment.risk_research.experiments.formation_selection import (
    DEVELOPMENT_FORMATION_CAPACITY,
)
from alphalattice.investment.risk_research.surfaces.returns import CausalRiskReturnReader
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResolvedResearchAuthority,
)

type FloatArray = NDArray[np.float64]

RISK_INPUT_BINDING_CATEGORY = "development/input-bindings"
"""Artifact category of the persisted development input binding.

Namespaced under ``development`` rather than beside the covariance surfaces:
this artifact exists only for development evidence, and a reader walking the
published covariance categories must not encounter it.
"""

REQUIRED_LOOKBACK_SESSIONS = 314
"""Sessions of history each formation consumes.

``historical.py`` starts eligibility at index 314 of its own axis. It is restated
here because the bounded window must reproduce that arithmetic exactly; the
agreement is asserted by test rather than assumed.
"""

REQUIRED_NEXT_SESSIONS = 1
"""The realized session each formation is evaluated against."""

"""Formation count is authored, not ceilinged.

``MAXIMUM_FORMATION_SESSIONS`` used to live here at 1000, described as the frozen
path own ``eligible_indices[-1000:]``. The development writer never runs that
slice -- ``build_development_covariance_surface`` iterates
``input_binding.formation_sessions`` and asserts the bound axis length -- so the
mirror guarded nothing here while making a wider authored axis unauthorable.
Selection now belongs to ``FormationSelectionPolicy`` and capacity to
``DEVELOPMENT_FORMATION_CAPACITY``, which are different questions.
"""


class ReturnSurfaceFreshnessProbe(Protocol):
    """Recompute the surface-scope source watermark, as the surface computed it.

    Supplied by Host composition: recomputing it needs the market store and the
    universe manifest, which is exactly the knowledge this Desk module must not
    acquire.
    """

    def __call__(self, *, through: date) -> str: ...


class PublishedReturnSurfaceProvider(Protocol):
    """Read-only access to every causal return surface a workspace published.

    The Host implements this because knowing where a workspace keeps manifests is
    workspace knowledge. *Choosing* between them is Risk methodology, so the
    choice is made by ``select_return_surface_for_authority`` below rather than
    by whoever happens to read the directory.
    """

    def published_surfaces(self) -> tuple[CausalRiskReturnSurface, ...]: ...


def select_return_surface_for_authority(
    *,
    authority: ResolvedResearchAuthority,
    surfaces: Sequence[CausalRiskReturnSurface],
) -> CausalRiskReturnSurface:
    """Pick the one surface this authority can be answered from, or fail closed.

    A real workspace accumulates immutable return surfaces as dates advance, so
    "there is exactly one manifest in the directory" was never a property of a
    running system -- it was a property of a fixture. Loading whichever file
    happened to be there produced evidence nobody could reproduce from the
    document alone.

    Selection is by exact agreement on the three axes that decide whether a
    surface can answer for an authority at all: the Panel snapshot, the universe
    revision, and the *ordered* listing axis. Order matters because the
    covariance matrix axis is positional, so a permuted surface silently
    transposes the result.

    The source watermark is deliberately not an equality here. The authority's
    watermark and a surface's watermark are computed at different ``through``
    dates and by different projections, so comparing them would be a type-check
    dressed as a causal proof. Freshness is established instead by
    ``resolve_development_input_binding``, which re-derives the surface-scope
    watermark through the authoritative reader.

    Neither "no match" nor "several matches" has a sensible fallback. Choosing
    the newest would make the answer depend on when the run happened rather than
    on what the document said.
    """

    candidates = tuple(
        surface
        for surface in surfaces
        if surface.epoch.panel_snapshot_hash == authority.panel_snapshot_hash
        and surface.epoch.universe_manifest_revision == authority.universe_revision_sha256
        and tuple(surface.epoch.ordered_listing_ids) == tuple(authority.ordered_listing_ids)
    )
    if not candidates:
        raise AuthoringError("research_authoring.return_surface_unavailable_for_authority")
    if len(candidates) > 1:
        raise AuthoringError("research_authoring.return_surface_ambiguous_for_authority")
    return candidates[0]


class RiskDevelopmentInputBinding(BaseModel):  # type: ignore[misc]
    """Every input the numerical path will read, sealed before it reads any."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["RiskDevelopmentInputBinding"] = "RiskDevelopmentInputBinding"
    # The authority this binding was resolved against. Without it a binding is
    # just a plausible set of hashes: a valid binding from another authority
    # would cross-check clean against a surface that happens to share an epoch.
    authority_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    return_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    return_epoch_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    panel_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    universe_revision_sha256: str = Field(min_length=1)
    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    lookback_sessions: int = Field(ge=0)
    next_sessions: int = Field(ge=0)
    # Both watermarks, at their own scopes, never equated.
    authority_source_watermark_hash: str = Field(min_length=1)
    return_surface_source_watermark_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_binding(self) -> Self:
        sessions = self.formation_sessions
        if sessions != tuple(sorted(set(sessions))):
            raise ValueError("risk_research.development_input_session_axis_invalid")
        if len(sessions) > DEVELOPMENT_FORMATION_CAPACITY:
            raise ValueError("risk_research.development_input_window_too_wide")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"input_binding_hash"}))
        if self.input_binding_hash != expected:
            raise ValueError("risk_research.development_input_binding_identity_invalid")
        return self

    @classmethod
    def create(
        cls,
        *,
        authority_hash: str,
        return_surface_hash: str,
        return_epoch_hash: str,
        panel_snapshot_hash: str,
        universe_revision_sha256: str,
        ordered_listing_ids: tuple[str, ...],
        formation_sessions: tuple[date, ...],
        lookback_sessions: int,
        next_sessions: int,
        authority_source_watermark_hash: str,
        return_surface_source_watermark_hash: str,
    ) -> RiskDevelopmentInputBinding:
        fields: dict[str, object] = {
            "kind": "RiskDevelopmentInputBinding",
            "authority_hash": authority_hash,
            "return_surface_hash": return_surface_hash,
            "return_epoch_hash": return_epoch_hash,
            "panel_snapshot_hash": panel_snapshot_hash,
            "universe_revision_sha256": universe_revision_sha256,
            "ordered_listing_ids": list(ordered_listing_ids),
            "formation_sessions": [value.isoformat() for value in formation_sessions],
            "lookback_sessions": lookback_sessions,
            "next_sessions": next_sessions,
            "authority_source_watermark_hash": authority_source_watermark_hash,
            "return_surface_source_watermark_hash": return_surface_source_watermark_hash,
        }
        return cls(
            authority_hash=authority_hash,
            return_surface_hash=return_surface_hash,
            return_epoch_hash=return_epoch_hash,
            panel_snapshot_hash=panel_snapshot_hash,
            universe_revision_sha256=universe_revision_sha256,
            ordered_listing_ids=ordered_listing_ids,
            formation_sessions=formation_sessions,
            lookback_sessions=lookback_sessions,
            next_sessions=next_sessions,
            authority_source_watermark_hash=authority_source_watermark_hash,
            return_surface_source_watermark_hash=return_surface_source_watermark_hash,
            input_binding_hash=str(canonical_hash(fields)),
        )


def resolve_development_input_binding(
    *,
    authority: ResolvedResearchAuthority,
    return_surface: CausalRiskReturnSurface,
    return_reader: CausalRiskReturnReader,
    freshness_probe: ReturnSurfaceFreshnessProbe,
) -> tuple[RiskDevelopmentInputBinding, tuple[date, ...]]:
    """Prove the surface can answer for this authority, before any estimate.

    Returns the sealed binding and the bounded session axis the numerical path
    will be held to. Every rejection here happens before the first estimator
    call, because a run that discovers its inputs disagree halfway through has
    already produced numbers nobody can attribute.
    """

    epoch = return_surface.epoch
    if authority.panel_snapshot_hash is None:
        raise AuthoringError("research_authoring.return_surface_panel_required")
    if epoch.panel_snapshot_hash != authority.panel_snapshot_hash:
        raise AuthoringError("research_authoring.return_surface_panel_mismatch")
    if epoch.universe_manifest_revision != authority.universe_revision_sha256:
        raise AuthoringError("research_authoring.return_surface_universe_mismatch")
    # Order, not membership: the covariance matrix axis is positional, so a
    # permuted listing axis silently transposes the result.
    if tuple(epoch.ordered_listing_ids) != tuple(authority.ordered_listing_ids):
        raise AuthoringError("research_authoring.return_surface_listing_axis_mismatch")

    # The causal check, through the authoritative reader rather than by
    # comparing two hashes from different scopes: the source state behind this
    # surface must still project to what the surface recorded when published.
    if freshness_probe(through=return_surface.last_formation_session) != (
        return_surface.source_watermark_hash
    ):
        raise AuthoringError("research_authoring.return_surface_source_moved")

    available = return_reader.available_sessions(return_surface)
    requested = tuple(authority.sessions)
    if not requested:
        raise AuthoringError("research_authoring.no_sessions_resolved")
    if len(requested) > DEVELOPMENT_FORMATION_CAPACITY:
        raise AuthoringError("research_authoring.requested_window_too_wide")

    position_by_session = {session: index for index, session in enumerate(available)}
    try:
        positions = tuple(position_by_session[session] for session in requested)
    except KeyError as error:
        raise AuthoringError("research_authoring.requested_session_outside_surface") from error
    # Contiguity is required, not incidental: the bounded axis below is a slice,
    # so a gap would silently pull in sessions the Program never requested.
    if positions != tuple(range(positions[0], positions[0] + len(positions))):
        raise AuthoringError("research_authoring.requested_sessions_not_contiguous")
    if positions[0] < REQUIRED_LOOKBACK_SESSIONS:
        raise AuthoringError("research_authoring.insufficient_lookback_sessions")
    if positions[-1] + REQUIRED_NEXT_SESSIONS > len(available) - 1:
        raise AuthoringError("research_authoring.insufficient_next_sessions")

    bounded = available[
        positions[0] - REQUIRED_LOOKBACK_SESSIONS : positions[-1] + REQUIRED_NEXT_SESSIONS + 1
    ]
    binding = RiskDevelopmentInputBinding.create(
        authority_hash=authority.authority_hash,
        return_surface_hash=return_surface.surface_hash,
        return_epoch_hash=epoch.epoch_hash,
        panel_snapshot_hash=authority.panel_snapshot_hash,
        universe_revision_sha256=authority.universe_revision_sha256,
        ordered_listing_ids=tuple(authority.ordered_listing_ids),
        formation_sessions=requested,
        lookback_sessions=REQUIRED_LOOKBACK_SESSIONS,
        next_sessions=REQUIRED_NEXT_SESSIONS,
        authority_source_watermark_hash=authority.source_watermark_hash,
        return_surface_source_watermark_hash=return_surface.source_watermark_hash,
    )
    return binding, bounded


class BoundedCausalReturnReader:
    """The reader interface, restricted to one bound window.

    The development writer calls exactly ``available_sessions`` and
    ``read_sessions``. Presenting the bounded axis through the first keeps the
    reader from seeing one session outside what was sealed; the second delegates
    unchanged so the numbers come from the real reader.

    This used to say the bounded axis exists so the frozen selection arithmetic
    lands on the bound formations. It no longer does:
    ``build_development_covariance_surface`` iterates the binding own
    ``formation_sessions``, so there is no tail slice left to steer.
    """

    def __init__(
        self,
        *,
        reader: CausalRiskReturnReader,
        bounded_sessions: Sequence[date],
        binding: RiskDevelopmentInputBinding,
    ) -> None:
        self._reader = reader
        self._bounded = tuple(bounded_sessions)
        self._binding = binding
        expected = (
            binding.lookback_sessions + len(binding.formation_sessions) + (binding.next_sessions)
        )
        if len(self._bounded) != expected:
            raise AuthoringError("research_authoring.bounded_window_inconsistent")

    def available_sessions(self, surface: CausalRiskReturnSurface) -> tuple[date, ...]:
        if surface.surface_hash != self._binding.return_surface_hash:
            raise AuthoringError("research_authoring.bounded_window_surface_mismatch")
        return self._bounded

    def read_sessions(
        self,
        surface: CausalRiskReturnSurface,
        formation_sessions: tuple[date, ...],
    ) -> FloatArray:
        if surface.surface_hash != self._binding.return_surface_hash:
            raise AuthoringError("research_authoring.bounded_window_surface_mismatch")
        if not set(formation_sessions).issubset(self._bounded):
            raise AuthoringError("research_authoring.bounded_window_read_escaped")
        return self._reader.read_sessions(surface, formation_sessions)


__all__ = [
    "REQUIRED_LOOKBACK_SESSIONS",
    "REQUIRED_NEXT_SESSIONS",
    "RISK_INPUT_BINDING_CATEGORY",
    "BoundedCausalReturnReader",
    "ReturnSurfaceFreshnessProbe",
    "RiskDevelopmentInputBinding",
    "resolve_development_input_binding",
]
