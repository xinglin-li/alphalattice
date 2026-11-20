"""Method-neutral development artifacts, shared by every Risk capability.

``HistoricalCovarianceSurface.recipe`` is typed ``CovarianceRecipe``, so it can
only ever describe one recipe schema. Pointing a second method at it would mean
either writing the wrong recipe into that field or giving the new method an
artifact of its own -- and the second choice is worse, because then every
subsequent method adds another branch to whatever decides which artifact to
write.

So there is exactly one development surface contract and both capabilities use
it. What it binds instead of a typed recipe is the ``RiskEstimatorRecipeEnvelope``
the adapter actually received, which is schema-neutral by construction: adapter
id, schema id, parameters, and the schema's own recipe hash.

Current and frozen artifact schemas are untouched. These live in their own
categories with their own identities, and nothing here is ever written to a
current or admitted pointer.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.market_data_ops.storage import world
from alphalattice.investment.risk_research.contracts import (
    HistoricalCovarianceChunk,
    RiskFormationEvaluation,
)
from alphalattice.investment.risk_research.estimators.contracts import (
    RiskEstimatorRecipeEnvelope,
)
from alphalattice.investment.risk_research.estimators.matrix_identity import matrix_content_hash
from alphalattice.investment.risk_research.experiments.contracts import (
    RiskDevelopmentProgramBinding,
)
from alphalattice.investment.risk_research.surfaces.artifacts import (
    CovarianceChunkLease,
    RiskArtifactError,
    RiskArtifactStore,
)
from alphalattice.kernel.data_layer import history, risk_history
from alphalattice.kernel.data_layer.upgrade import Family, UpgradeRunner
from alphalattice.kernel.shared_kernel.identity import canonical_hash

_HASH = r"^[0-9a-f]{64}$"

DEVELOPMENT_SURFACE_CATEGORY = "development/covariance-surfaces"
DEVELOPMENT_DIAGNOSTICS_CATEGORY = "development/covariance-diagnostics"
DEVELOPMENT_CHECKPOINT_CATEGORY = "development/build-checkpoints"


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class RiskFormationAxis(_Contract):
    """The listing axis used for one development formation and its excluded names."""

    formation_session: date
    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)
    excluded_listing_ids: tuple[str, ...]


class RiskDevelopmentDiagnostics(_Contract):
    """Per-formation evaluations for one development build, any method.

    ``RiskFormationEvaluation`` is already method-neutral -- eigenvalues, trace,
    condition number, realized-versus-predicted -- so a second method needs no
    new diagnostic shape, only its own values.
    """

    kind: Literal["RiskDevelopmentDiagnostics"] = "RiskDevelopmentDiagnostics"
    input_binding_hash: str = Field(pattern=_HASH)
    recipe_hash: str = Field(pattern=_HASH)
    evaluations: tuple[RiskFormationEvaluation, ...] = Field(min_length=1)
    component_shrinkages: tuple[tuple[str, tuple[float, ...]], ...] = ()
    """Per-formation intensity of each named component of a blended method.

    ``RiskFormationEvaluation.shrinkage`` is one number, which is the whole
    story for a method that fits one estimator and only a summary for a method
    that fits several. It is inside the frozen execution closure and cannot
    grow a field, and the parts are not recoverable afterwards without
    refitting -- so they are recorded here, in the development diagnostics,
    which no published identity depends on.

    Empty for a single-component method. The series is per formation and in
    formation order, so a Campaign can aggregate it any way it needs without
    the writer having chosen a statistic on its behalf.
    """

    diagnostics_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(
        cls,
        *,
        input_binding_hash: str,
        recipe_hash: str,
        evaluations: tuple[RiskFormationEvaluation, ...],
        component_shrinkages: tuple[tuple[str, tuple[float, ...]], ...] = (),
    ) -> Self:
        values = {
            "kind": "RiskDevelopmentDiagnostics",
            "input_binding_hash": input_binding_hash,
            "recipe_hash": recipe_hash,
            "evaluations": [value.model_dump(mode="json") for value in evaluations],
            "component_shrinkages": [[name, list(series)] for name, series in component_shrinkages],
        }
        return cls(
            input_binding_hash=input_binding_hash,
            recipe_hash=recipe_hash,
            evaluations=evaluations,
            component_shrinkages=component_shrinkages,
            diagnostics_hash=str(canonical_hash(values)),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        expected = canonical_hash(self.model_dump(mode="json", exclude={"diagnostics_hash"}))
        if self.diagnostics_hash != expected:
            raise ValueError("risk_research.development_diagnostics_identity_invalid")
        return self


class RiskDevelopmentCheckpoint(_Contract):
    """A mutable working prefix; terminal evidence remains content-addressed."""

    execution_key: str = Field(pattern=_HASH)
    expected_formation_count: int = Field(ge=1)
    chunks: tuple[HistoricalCovarianceChunk, ...]
    evaluations: tuple[RiskFormationEvaluation, ...]
    component_shrinkages: tuple[tuple[str, tuple[float, ...]], ...] = ()
    formation_axes: tuple[RiskFormationAxis, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    ordered_listing_ids: tuple[str, ...] | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )
    checkpoint_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        draft = cls.model_construct(**values, checkpoint_hash="")
        return cls(
            **values,
            checkpoint_hash=str(
                canonical_hash(draft.model_dump(mode="json", exclude={"checkpoint_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_prefix(self) -> Self:
        sessions = tuple(s for chunk in self.chunks for s in chunk.formation_sessions)
        hashes = tuple(h for chunk in self.chunks for h in chunk.matrix_hashes)
        if (
            len(sessions) != len(self.evaluations)
            or len(sessions) > self.expected_formation_count
            or sessions != tuple(sorted(set(sessions)))
            or sessions != tuple(e.formation_session for e in self.evaluations)
            or hashes != tuple(e.matrix_hash for e in self.evaluations)
            or any(len(series) != len(sessions) for _, series in self.component_shrinkages)
            or (
                self.formation_axes is not None
                and tuple(axis.formation_session for axis in self.formation_axes) != sessions
            )
            or self.checkpoint_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"checkpoint_hash"}))
        ):
            raise ValueError("risk_research.development_checkpoint_invalid")
        if self.ordered_listing_ids is not None and self.formation_axes is not None:
            raise ValueError("risk_research.development_checkpoint_axis_mismatch")
        for chunk, axis in self.chunks_with_axes():
            if axis is not None and (len(axis) != chunk.asset_count or len(set(axis)) != len(axis)):
                raise ValueError("risk_research.development_checkpoint_axis_mismatch")
        return self

    def chunks_with_axes(
        self,
    ) -> Iterator[tuple[HistoricalCovarianceChunk, tuple[str, ...] | None]]:
        """Yield each sealed chunk with its declared shared or per-formation listing axis."""
        position = 0
        for chunk in self.chunks:
            axis = self.ordered_listing_ids
            if self.formation_axes is not None:
                formations = self.formation_axes[position : position + chunk.matrix_count]
                axis = formations[0].ordered_listing_ids
                if any(item.ordered_listing_ids != axis for item in formations):
                    raise ValueError("risk_research.development_checkpoint_axis_mismatch")
            yield chunk, axis
            position += chunk.matrix_count


class RiskDevelopmentCovarianceSurface(_Contract):
    """One bounded development build, produced by any installed capability.

    The method is described by ``recipe_envelope`` and
    ``selected_numerical_binding_hash`` rather than by a typed recipe field, so
    this contract does not have to change when a capability with a different
    recipe schema is installed.
    """

    kind: Literal["RiskDevelopmentCovarianceSurface"] = "RiskDevelopmentCovarianceSurface"
    identity_class: Literal["DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"] = (
        "DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"
    )
    input_binding_hash: str = Field(pattern=_HASH)
    capability_handle: str = Field(min_length=1, max_length=128)
    recipe_envelope: RiskEstimatorRecipeEnvelope
    recipe_hash: str = Field(pattern=_HASH)
    """The schema's *own* recipe identity, which is what the Program binds.

    Distinct from ``recipe_envelope.recipe_hash``, which covers the envelope --
    adapter id, schema id and parameters. Both are real identities of different
    things, and conflating them would silently compare unlike values.
    """
    program_binding: RiskDevelopmentProgramBinding
    """The whole Program binding, not a copy of four of its hashes.

    Carrying the hashes flat made the surface unable to answer for itself. Its
    method fields were free-standing strings, so a surface could name a
    ``development_binding_hash`` that no combination of its other fields would
    ever produce, and replay had nothing to check that against.

    Embedded, the binding revalidates on read: parsing the artifact recomputes
    ``selected_method_binding_hash`` and ``development_binding_hash`` from the
    fields underneath them, so an edited method field fails to parse rather than
    verifying against itself. What replay then has to establish is only the part
    the artifact cannot know -- that this binding is the Program being replayed.
    """

    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    chunks: tuple[HistoricalCovarianceChunk, ...] = Field(min_length=1)
    formation_axes: tuple[RiskFormationAxis, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    diagnostics_hash: str = Field(pattern=_HASH)
    limitations: tuple[str, ...] = Field(min_length=1)
    surface_hash: str = Field(pattern=_HASH)

    def chunk_listing_ids(self, chunk: HistoricalCovarianceChunk) -> tuple[str, ...]:
        """Return the shared listing axis validated for this chunk's formations."""
        if self.formation_axes is None:
            return self.ordered_listing_ids
        position = self.formation_sessions.index(chunk.formation_sessions[0])
        return self.formation_axes[position].ordered_listing_ids

    @property
    def selected_numerical_binding_hash(self) -> str:
        return self.program_binding.selected_numerical_binding_hash

    @property
    def selected_method_binding_hash(self) -> str:
        return self.program_binding.selected_method_binding_hash

    @property
    def development_binding_hash(self) -> str:
        return self.program_binding.development_binding_hash

    @property
    def numerical_environment_hash(self) -> str | None:
        """The environment a Program sealed before E0 declared; None since (LAWS.md ID6)."""
        return self.program_binding.numerical_environment_hash

    @classmethod
    def create(cls, **values: object) -> Self:
        draft = dict(values)
        draft.pop("surface_hash", None)
        provisional = cls.model_construct(**draft, surface_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"surface_hash"})
        return cls(**draft, surface_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify the binding, formation axes, chunks and sealed surface identity."""
        # The surface's own method fields must agree with the Program binding it
        # carries. Without this the artifact could describe one method in its
        # envelope and a different one in its binding, and every hash on it would
        # still be internally valid.
        if self.recipe_hash != self.program_binding.recipe_hash:
            raise ValueError("risk_research.development_surface_recipe_not_its_program")
        if self.capability_handle != self.recipe_envelope.recipe_schema_id:
            raise ValueError("risk_research.development_surface_capability_route_invalid")
        if self.recipe_envelope.adapter_id != self.program_binding.selected_adapter_id:
            raise ValueError("risk_research.development_surface_adapter_route_invalid")
        if sum(chunk.matrix_count for chunk in self.chunks) != len(self.formation_sessions):
            raise ValueError("risk_research.development_surface_count_invalid")
        if self.formation_axes is None:
            if any(chunk.asset_count != len(self.ordered_listing_ids) for chunk in self.chunks):
                raise ValueError("risk_research.development_surface_axis_invalid")
        else:
            if (
                tuple(axis.formation_session for axis in self.formation_axes)
                != self.formation_sessions
            ):
                raise ValueError("risk_research.development_surface_axis_invalid")
            for axis in self.formation_axes:
                selected = set(axis.ordered_listing_ids)
                if (
                    tuple(item for item in self.ordered_listing_ids if item in selected)
                    != axis.ordered_listing_ids
                    or tuple(item for item in self.ordered_listing_ids if item not in selected)
                    != axis.excluded_listing_ids
                ):
                    raise ValueError("risk_research.development_surface_axis_invalid")
            position = 0
            for chunk in self.chunks:
                axes = self.formation_axes[position : position + chunk.matrix_count]
                if any(
                    axis.ordered_listing_ids != axes[0].ordered_listing_ids
                    or len(axis.ordered_listing_ids) != chunk.asset_count
                    for axis in axes
                ):
                    raise ValueError("risk_research.development_surface_axis_invalid")
                position += chunk.matrix_count
        computed = tuple(session for chunk in self.chunks for session in chunk.formation_sessions)
        # Order, not membership: the covariance axis is positional.
        if computed != tuple(self.formation_sessions):
            raise ValueError("risk_research.development_surface_session_axis_invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"surface_hash"}))
        if self.surface_hash != expected:
            raise ValueError("risk_research.development_surface_identity_invalid")
        return self


def _raise_walk_error(error: OSError) -> None:
    raise error


def _covariance_chunks(
    workspace: Path,
) -> Iterator[tuple[RiskArtifactStore, HistoricalCovarianceChunk, tuple[str, ...], Path]]:
    """Read the sealed covariance references from published surfaces and durable checkpoints."""
    if not workspace.is_dir():
        return
    for folder, children, _files in os.walk(workspace, onerror=_raise_walk_error):
        if "risk-research" not in children:
            continue
        children.remove("risk-research")
        store = RiskArtifactStore(Path(folder))
        for payload in store.iter_json(
            category=DEVELOPMENT_SURFACE_CATEGORY, identity_field="surface_hash"
        ):
            surface = RiskDevelopmentCovarianceSurface.model_validate(payload)
            for chunk in surface.chunks:
                axis = surface.chunk_listing_ids(chunk)
                yield (
                    store,
                    chunk,
                    axis,
                    store._path("covariance/chunks", chunk.content_hash, "bin"),
                )

        try:
            checkpoints = sorted((store.root / DEVELOPMENT_CHECKPOINT_CATEGORY).iterdir())
        except FileNotFoundError:
            checkpoints = []
        for path in checkpoints:
            if path.suffix != ".json":
                continue
            checkpoint = RiskDevelopmentCheckpoint.model_validate_json(path.read_bytes())
            if path.stem != checkpoint.execution_key:
                raise RiskArtifactError("risk_research.development_checkpoint_scope_mismatch")
            for chunk, axis in checkpoint.chunks_with_axes():
                packed = store._path("covariance/chunks", chunk.content_hash, "bin")
                if axis is not None:
                    yield store, chunk, axis, packed
                elif not packed.is_file():
                    raise RiskArtifactError("risk_research.covariance_chunk_incomplete")
                # A pre-catalog dense checkpoint only has packed bytes, copied by the workspace.


def _legacy_covariance_chunks(
    workspace: Path,
) -> Iterator[tuple[RiskArtifactStore, HistoricalCovarianceChunk, tuple[str, ...]]]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for store, chunk, axis, path in _covariance_chunks(workspace):
        key = (chunk.content_hash, axis)
        if key not in seen and path.is_file():
            seen.add(key)
            yield store, chunk, axis


def carry_risk_covariance(
    workspace: Path, copy: Path, *, source_root: Path | None = None
) -> Iterator[Path]:
    """Carry catalog-only chunks into a backup or copy at their retained-file paths."""
    workspace = workspace.resolve()
    for store, chunk, axis, path in _covariance_chunks(workspace):
        target = copy / path.relative_to(workspace)
        if path.is_file() or target.is_file():
            continue
        packed = store.catalog_covariance(chunk, axis, root=source_root)
        if packed is None:
            raise RiskArtifactError("risk_research.covariance_chunk_incomplete")
        store._atomic_write(target, packed.astype("<f8", copy=False).tobytes())
        yield target


def risk_covariance_upgrade_family(holder: Callable[[Path], str | None]) -> Family:
    """Move each sealed legacy chunk with its owning workspace and prove the original seals."""

    def deferred(workspace: Path) -> bool:
        # A workspace backup restores market tables as exports, not as a live market database.
        # Its packed Risk chunks remain readable until authoritative listings have arrived.
        if (
            not (workspace / "held-state" / "generation.json").is_file()
            or (workspace / "market-data.duckdb").is_file()
        ):
            return False
        return any(
            risk_history.unregistered_risk_listings(store.coordinate_axis(chunk, axis)) != []
            for store, chunk, axis in _legacy_covariance_chunks(workspace)
        )

    def holds(workspace: Path) -> bool:
        if deferred(workspace):
            return False
        identity = holder(workspace)
        return any(
            identity is None
            or risk_history.risk_covariance_held(
                chunk.content_hash,
                store.coordinate_key(store.coordinate_axis(chunk, axis)),
                identity,
            )
            != chunk.matrix_count
            for store, chunk, axis in _legacy_covariance_chunks(workspace)
        )

    def step(runner: UpgradeRunner) -> None:
        for workspace in runner.workspaces:
            identity = holder(workspace)
            for index, (store, chunk, axis) in enumerate(_legacy_covariance_chunks(workspace)):
                if index == 0:
                    identity = history.workspace_on_log(workspace, workspace.name, identity)
                    world.admit_universe(workspace / "market-data.duckdb", identity)
                packed = store.legacy_covariance(chunk)
                store.admit_covariance(chunk, axis, packed, workspace_id=identity)

    def verify(runner: UpgradeRunner) -> bool:
        for workspace in runner.workspaces:
            for store, chunk, axis in _legacy_covariance_chunks(workspace):
                packed = store.catalog_covariance(chunk, axis)
                if packed is None:
                    return False
                lease = CovarianceChunkLease(packed=packed, chunk=chunk)
                if any(
                    matrix_content_hash(lease.matrix(day)) != expected
                    for day, expected in zip(
                        chunk.formation_sessions, chunk.matrix_hashes, strict=True
                    )
                ):
                    return False
        return True

    return Family("risk-covariance", step, verify, holds, deferred=deferred)


__all__ = [
    "DEVELOPMENT_DIAGNOSTICS_CATEGORY",
    "DEVELOPMENT_SURFACE_CATEGORY",
    "RiskDevelopmentCovarianceSurface",
    "RiskDevelopmentDiagnostics",
]
