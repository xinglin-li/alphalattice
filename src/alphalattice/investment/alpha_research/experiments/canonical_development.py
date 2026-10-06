"""Host-owned publication of the canonical Alpha target and its cross-sectional scale.

Until now the canonical target compiler, its evidence sealer and the dispersion
executor were callable functions with no product caller: every one of them was
reachable only from a case study. A method that exists but that nothing in the
product runs is not an installed capability, it is a proposal -- and calling it
complete is the failure this program keeps rediscovering.

This service is the missing writer. It is deliberately narrow: it resolves
authority, compiles, publishes, reads back, and verifies. It is not a pipeline
framework, it owns no schedule, and it decides nothing about which snapshot to
process -- a Human, an Installed Agent or external automation submits a request
naming resolvable handles and gets the same behaviour.

The request boundary is the point. A caller may name *what* to publish from --
an outcome snapshot, a Sector revision, a maturity clock -- and may not state
any of the identities that give the result its authority. The snapshot hash it
was built from, the outcome method binding, the maturity lag and the catalog
identity are all derived here from resolved evidence. A caller that could supply
those would be asserting exactly what publication exists to establish.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.causal_outcomes.execution.contracts import (
    CausalExecutionOutcomeManifest,
)
from alphalattice.foundation.causal_outcomes.execution.methods import (
    ExecutionOutcomeMethodSeal,
)
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalOutcomeDevelopmentRows,
)
from alphalattice.foundation.factor_research.inputs.execution_target import (
    build_factor_target_policy_for_method,
    compile_factor_target_surface,
)
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.closure_contracts import SectorRevisionMap
from alphalattice.investment.alpha_research.inputs.workspace import load_sector_history
from alphalattice.investment.alpha_research.scaling.catalog import (
    CrossSectionalScaleCatalog,
    build_installed_cross_sectional_scale_catalog,
)
from alphalattice.investment.alpha_research.scaling.contracts import (
    CrossSectionalDispersionForecast,
)
from alphalattice.investment.alpha_research.scaling.execution import (
    compile_cross_sectional_dispersion_forecast,
)
from alphalattice.investment.alpha_research.targets.canonical import (
    CANONICAL_ALPHA_TARGET_RECIPE_ID,
    JOINT_PRIMARY_ALPHA_TARGET_RECIPE_ID,
    CanonicalAlphaTargetEvidence,
    CanonicalAlphaTargetRecipeBinding,
    CanonicalAlphaTargetSurface,
    build_canonical_alpha_target_recipe,
    build_joint_primary_alpha_target_recipe,
    compile_canonical_alpha_target_surface,
    seal_canonical_alpha_target_evidence,
)
from alphalattice.investment.alpha_research.targets.catalog import (
    AlphaTargetCatalog,
    build_installed_alpha_target_catalog,
)
from alphalattice.investment.alpha_research.targets.materialization import (
    CanonicalTargetMaterialization,
)

from .development_artifacts import AlphaDevelopmentArtifactStore


class CanonicalDevelopmentError(ValueError):
    """Stable failure raised before any canonical development evidence is written."""


@dataclass(frozen=True, slots=True)
class CanonicalDevelopmentRequest:
    """What a caller is entitled to name.

    Every field is a handle this service resolves, never a conclusion. There is
    deliberately no place to put a method binding hash, a maturity lag or a
    catalog identity: those are results of resolution, and a request carrying
    them would let the caller decide what the evidence says about itself.
    """

    causal_outcome_snapshot_hash: str
    sector_revision: str
    holding_end_through: date
    target_method_id: str = CANONICAL_ALPHA_TARGET_RECIPE_ID
    """The maturity clock. Only formations whose outcome had already finished by
    this session are eligible, so the caller states an observation date rather
    than a session list it might have chosen to flatter the result."""


@dataclass(frozen=True, slots=True)
class CanonicalDevelopmentPublication:
    """Everything one run sealed, with the URIs it can be re-read from."""

    recipe_binding: CanonicalAlphaTargetRecipeBinding
    evidence: CanonicalAlphaTargetEvidence
    dispersion: CrossSectionalDispersionForecast
    surface: CanonicalAlphaTargetSurface
    recipe_binding_uri: str
    evidence_uri: str
    dispersion_uri: str
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CanonicalTargetLineage:
    """A verified walk from a published evidence hash back to its authority."""

    evidence: CanonicalAlphaTargetEvidence
    recipe_binding: CanonicalAlphaTargetRecipeBinding
    outcome_snapshot_hash: str
    outcome_method_binding_hash: str
    maturity_lag_sessions: int
    dispersion: CrossSectionalDispersionForecast | None


class CanonicalAlphaDevelopmentService:
    """Resolve, compile, publish and verify one canonical target and its scale."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        outcome_reader: CausalOutcomeDevelopmentRows,
        resolver: ArtifactResolver,
        target_catalog: AlphaTargetCatalog | None = None,
        scale_catalog: CrossSectionalScaleCatalog | None = None,
    ) -> None:
        self.store = AlphaDevelopmentArtifactStore(artifact_root)
        self._outcome_reader = outcome_reader
        self._resolver = resolver
        self._target_catalog = target_catalog or build_installed_alpha_target_catalog()
        self._scale_catalog = scale_catalog or build_installed_cross_sectional_scale_catalog()

    # ------------------------------------------------------------------ publish
    def publish(self, request: CanonicalDevelopmentRequest) -> CanonicalDevelopmentPublication:
        """Compile and seal the canonical target and its scale for one snapshot."""

        manifest_ref = self._outcome_reader.manifest_uri(request.causal_outcome_snapshot_hash)
        manifest = self._outcome_reader.load_manifest(request.causal_outcome_snapshot_hash)

        seal = self._outcome_reader.resolve_method_seal(request.causal_outcome_snapshot_hash)
        if seal.disposition != "METHOD_BOUND":
            # A pre-seam snapshot stays readable and never becomes the source of
            # new development evidence. This is the boundary that keeps legacy
            # readback from quietly becoming an active writer.
            raise CanonicalDevelopmentError("alpha_research.canonical_outcome_method_unbound")
        method = seal.method_bound
        if method.snapshot_hash != manifest.snapshot_hash:
            raise CanonicalDevelopmentError("alpha_research.canonical_outcome_seal_mismatch")

        sessions = self._outcome_reader.available_development_sessions(
            manifest_ref, holding_end_through=request.holding_end_through
        )
        if len(sessions) < method.maturity_lag_sessions + 1:
            # Fewer formations than the lag needs means the scale surface would
            # be entirely unavailable; refuse rather than publish an empty claim.
            raise CanonicalDevelopmentError("alpha_research.canonical_matured_axis_too_short")

        surface = self._compile(
            manifest=manifest,
            manifest_ref=manifest_ref,
            seal=seal,
            sessions=sessions,
            sector_revision=request.sector_revision,
            target_method_id=request.target_method_id,
        )

        # Every identity below is derived here. Nothing in the request could have
        # supplied one.
        recipe_binding = CanonicalAlphaTargetRecipeBinding.create(
            # The recipe the compiler actually sealed onto the surface, not a
            # second one assembled here: the binding must answer for the object
            # that produced the numbers.
            recipe=surface.recipe,
            target_catalog_hash=self._target_catalog.binding.catalog_hash,
            causal_outcome_snapshot_hash=manifest.snapshot_hash,
            outcome_method_binding_hash=method.binding_hash,
            maturity_lag_sessions=method.maturity_lag_sessions,
        )
        evidence = seal_canonical_alpha_target_evidence(
            surface=surface, recipe_binding=recipe_binding
        )
        dispersion = compile_cross_sectional_dispersion_forecast(
            target_evidence=evidence,
            target_recipe_binding=recipe_binding,
            outcome_method_binding_hash=method.binding_hash,
            maturity_lag_sessions=method.maturity_lag_sessions,
            catalog=self._scale_catalog,
        )

        # Children before parents, so nothing durable ever names a document that
        # does not yet exist.
        recipe_binding_uri = self.store.publish_canonical_target_recipe_binding(recipe_binding)
        evidence_uri = self.store.publish_canonical_target_evidence(evidence)
        dispersion_uri = self.store.publish_canonical_dispersion_forecast(dispersion)

        # Read back through the verifier rather than trusting the objects still
        # in memory: a publisher that verifies its own variables proves only that
        # it can remember them.
        self.verify(evidence_hash=evidence.evidence_hash, dispersion_hash=dispersion.forecast_hash)
        return CanonicalDevelopmentPublication(
            recipe_binding=recipe_binding,
            evidence=evidence,
            dispersion=dispersion,
            surface=surface,
            recipe_binding_uri=recipe_binding_uri,
            evidence_uri=evidence_uri,
            dispersion_uri=dispersion_uri,
            formation_sessions=surface.formation_sessions,
            ordered_listing_ids=surface.ordered_listing_ids,
        )

    # -------------------------------------------------------------- materialize
    def _compile(
        self,
        *,
        manifest: CausalExecutionOutcomeManifest,
        manifest_ref: str,
        seal: ExecutionOutcomeMethodSeal,
        sessions: tuple[date, ...],
        sector_revision: str,
        target_method_id: str,
    ) -> CanonicalAlphaTargetSurface:
        """Read the named sessions and compile the canonical surface over them.

        The one compile path. ``publish`` and ``materialize`` differ only in how
        the session axis and the target method are chosen -- a maturity clock and
        a request in the first, a published axis and a durable recipe binding in
        the second -- and sharing everything below that is what makes a rebuilt
        surface comparable to the sealed one by construction rather than by having
        been written to match it.
        """

        method = seal.method_bound
        source_rows = self._outcome_reader.read_development_sessions(manifest_ref, sessions)
        if source_rows.num_rows == 0:
            raise CanonicalDevelopmentError("alpha_research.canonical_outcome_rows_unavailable")

        # The log/simple lanes come from the Factor target compiler, which is the
        # one owner of that projection, and it is given the resolved seal rather
        # than a described method.
        factor_surface = compile_factor_target_surface(
            source_table=source_rows,
            source_manifest=manifest,
            source_manifest_ref=manifest_ref,
            policy=build_factor_target_policy_for_method(method),
            outcome_method=seal,
        )

        sector_by_listing_id = load_sector_history(
            resolver=self._resolver, sector_revision=sector_revision
        )
        listing_ids = tuple(sorted(set(str(value) for value in manifest.listing_ids)))
        missing = set(listing_ids) - set(sector_by_listing_id)
        if missing:
            # Sector membership is an authority, not a convenience: a listing the
            # revision does not cover cannot be residualized against a Sector.
            raise CanonicalDevelopmentError("alpha_research.canonical_sector_authority_incomplete")

        if target_method_id == CANONICAL_ALPHA_TARGET_RECIPE_ID:
            recipe = build_canonical_alpha_target_recipe(
                execution_outcome_recipe_id=method.recipe_id,
                sector_revision=sector_revision,
            )
        elif target_method_id == JOINT_PRIMARY_ALPHA_TARGET_RECIPE_ID:
            recipe = build_joint_primary_alpha_target_recipe(
                execution_outcome_recipe_id=method.recipe_id,
                sector_revision=sector_revision,
            )
        else:
            raise CanonicalDevelopmentError("alpha_research.target_method_not_installed")
        surface = compile_canonical_alpha_target_surface(
            source_table=factor_surface.table,
            recipe=recipe,
            sector_by_listing_id=sector_by_listing_id,
            standardizations=self._target_catalog,
        )
        if surface.ordered_listing_ids != listing_ids:
            raise CanonicalDevelopmentError("alpha_research.canonical_listing_axis_mismatch")
        if surface.formation_sessions != tuple(sessions):
            raise CanonicalDevelopmentError("alpha_research.canonical_session_axis_mismatch")
        return surface

    def resolved_sector_revision(self, stated: str | None = None) -> str:
        """Read the Sector revision out of the Panel closure, or take the stated one.

        Resolved rather than required, because a source workspace is an input
        *location* and not an authority: the Host reads the lineage out of what
        that location published. A revision the caller states is still honoured
        -- and is checked downstream against the published recipe hash either
        way, so a wrong one is refused before any arithmetic.

        Here rather than in an entry point because two entry points need it --
        the canonical publisher and the Portfolio Campaign's calibration
        resolver -- and a rule spelled out twice is a rule that can drift.
        """

        if stated is not None:
            return stated
        store = PanelClosureArtifactStore(self._resolver)
        revisions = sorted(
            {
                store.load_model(
                    category="sector-maps", content_hash=path.stem, model=SectorRevisionMap
                ).sector_revision
                for path in sorted((store.root / "sector-maps").glob("*.json"))
            }
        )
        if len(revisions) != 1:
            # More than one published map is not an error in the Panel; it is an
            # ambiguity this service must not resolve by picking.
            raise CanonicalDevelopmentError("alpha_research.canonical_sector_revision_ambiguous")
        return str(revisions[0])

    def materialize(
        self, *, evidence_hash: str, sector_revision: str
    ) -> CanonicalTargetMaterialization:
        """Rebuild one published canonical surface from source, and reconcile it.

        Writes nothing. The session axis is taken from the published evidence
        rather than from a clock the caller states, so "which formations" is
        answered by the artifact being reconciled instead of by an argument that
        could be adjusted until two axes happened to line up.

        ``sector_revision`` is a handle, not an assertion: the recipe it produces
        must hash to the ``target_recipe_hash`` the published binding already
        carries, so a wrong revision is refused before any arithmetic rather than
        silently residualising against a different Sector map.
        """

        evidence = self.store.load_canonical_target_evidence(evidence_hash)
        recipe_binding = self.store.load_canonical_target_recipe_binding(
            evidence.recipe_binding_hash
        )
        seal = self._outcome_reader.resolve_method_seal(recipe_binding.causal_outcome_snapshot_hash)
        if seal.disposition != "METHOD_BOUND":
            raise CanonicalDevelopmentError("alpha_research.canonical_outcome_method_unbound")
        method = seal.method_bound
        if recipe_binding.outcome_method_binding_hash != method.binding_hash:
            raise CanonicalDevelopmentError("alpha_research.canonical_outcome_method_mismatch")
        if recipe_binding.maturity_lag_sessions != method.maturity_lag_sessions:
            raise CanonicalDevelopmentError("alpha_research.canonical_maturity_lag_mismatch")
        if recipe_binding.target_catalog_hash != self._target_catalog.binding.catalog_hash:
            raise CanonicalDevelopmentError("alpha_research.canonical_target_catalog_mismatch")

        manifest_ref = self._outcome_reader.manifest_uri(
            recipe_binding.causal_outcome_snapshot_hash
        )
        manifest = self._outcome_reader.load_manifest(recipe_binding.causal_outcome_snapshot_hash)
        sessions = tuple(evidence.formation_sessions)
        # The maturity clock still applies: every published formation must be one
        # this reader offers for development as at the snapshot's own market
        # cutoff. Derived from the manifest rather than taken as an argument --
        # a caller-supplied observation date is exactly the field that could be
        # moved until an axis lined up.
        matured = set(
            self._outcome_reader.available_development_sessions(
                manifest_ref, holding_end_through=manifest.market_as_of
            )
        )
        if not set(sessions).issubset(matured):
            raise CanonicalDevelopmentError("alpha_research.canonical_published_axis_not_matured")

        surface = self._compile(
            manifest=manifest,
            manifest_ref=manifest_ref,
            seal=seal,
            sessions=sessions,
            sector_revision=sector_revision,
            # The durable binding says which target method produced the published
            # surface. Taking it from an argument would let a caller rebuild one
            # recipe's evidence under another and discover the mismatch only if
            # the hash check below happened to catch it.
            target_method_id=recipe_binding.target_recipe_id,
        )
        if surface.recipe.recipe_hash != recipe_binding.target_recipe_hash:
            raise CanonicalDevelopmentError("alpha_research.canonical_rebuilt_recipe_mismatch")
        if tuple(surface.ordered_listing_ids) != tuple(evidence.ordered_listing_ids):
            raise CanonicalDevelopmentError(
                "alpha_research.canonical_rebuilt_listing_axis_mismatch"
            )

        identity = surface.lane_identity
        for field in (
            "lane_identity_hash",
            "ordered_sessions_hash",
            "ordered_listing_ids_hash",
            "cross_sectional_dispersion_identity",
            "redemeaned_residual_identity",
            "raw_log_execution_return_identity",
            "fit_target_identity",
        ):
            if getattr(identity, field) != getattr(evidence, field):
                raise CanonicalDevelopmentError(
                    f"alpha_research.canonical_rebuilt_lane_mismatch:{field}"
                )
        return CanonicalTargetMaterialization(
            evidence=evidence,
            recipe_binding=recipe_binding,
            surface=surface,
            outcome_method_binding_hash=method.binding_hash,
            maturity_lag_sessions=method.maturity_lag_sessions,
            simple_economic_return_identity=identity.simple_economic_return_identity,
        )

    # ------------------------------------------------------------------- verify
    def verify(
        self, *, evidence_hash: str, dispersion_hash: str | None = None
    ) -> CanonicalTargetLineage:
        """Walk a published evidence hash back to the outcome method that authorized it.

        Every edge is re-derived from a document loaded from the store, and the
        terminal edge is re-derived from the *outcome reader* rather than from
        anything this Desk wrote. That last step is what a self-hash check cannot
        do: a caller who forged a recipe binding, sealed evidence against it and
        published both would produce a perfectly self-consistent pair, and it
        fails here because the outcome snapshot it names does not carry the
        method binding it claims.
        """

        evidence = self.store.load_canonical_target_evidence(evidence_hash)
        recipe_binding = self.store.load_canonical_target_recipe_binding(
            evidence.recipe_binding_hash
        )
        if evidence.recipe_binding_hash != recipe_binding.binding_hash:
            raise CanonicalDevelopmentError("alpha_research.canonical_evidence_binding_mismatch")

        seal = self._outcome_reader.resolve_method_seal(recipe_binding.causal_outcome_snapshot_hash)
        if seal.disposition != "METHOD_BOUND":
            raise CanonicalDevelopmentError("alpha_research.canonical_outcome_method_unbound")
        method = seal.method_bound
        if recipe_binding.outcome_method_binding_hash != method.binding_hash:
            raise CanonicalDevelopmentError("alpha_research.canonical_outcome_method_mismatch")
        if recipe_binding.maturity_lag_sessions != method.maturity_lag_sessions:
            raise CanonicalDevelopmentError("alpha_research.canonical_maturity_lag_mismatch")
        if recipe_binding.target_catalog_hash != self._target_catalog.binding.catalog_hash:
            raise CanonicalDevelopmentError("alpha_research.canonical_target_catalog_mismatch")

        dispersion: CrossSectionalDispersionForecast | None = None
        if dispersion_hash is not None:
            dispersion = self.store.load_canonical_dispersion_forecast(dispersion_hash)
            if dispersion.target_evidence_hash != evidence.evidence_hash:
                raise CanonicalDevelopmentError("alpha_research.canonical_scale_evidence_mismatch")
            if dispersion.target_recipe_binding_hash != recipe_binding.binding_hash:
                raise CanonicalDevelopmentError("alpha_research.canonical_scale_binding_mismatch")
            if dispersion.outcome_method_binding_hash != method.binding_hash:
                raise CanonicalDevelopmentError("alpha_research.canonical_scale_method_mismatch")
            if dispersion.maturity_lag_sessions != method.maturity_lag_sessions:
                raise CanonicalDevelopmentError("alpha_research.canonical_scale_lag_mismatch")
            if (
                dispersion.source_dispersion_identity
                != evidence.cross_sectional_dispersion_identity
            ):
                raise CanonicalDevelopmentError("alpha_research.canonical_scale_lane_mismatch")
            # The installed implementation must still be the one that ran, by value or
            # by recorded moves (LAWS.md ID1).
            if not self._scale_catalog.implementation_current(
                dispersion.recipe_id, dispersion.implementation.implementation_binding_hash
            ):
                raise CanonicalDevelopmentError(
                    "alpha_research.canonical_scale_implementation_drift"
                )

        return CanonicalTargetLineage(
            evidence=evidence,
            recipe_binding=recipe_binding,
            outcome_snapshot_hash=recipe_binding.causal_outcome_snapshot_hash,
            outcome_method_binding_hash=method.binding_hash,
            maturity_lag_sessions=method.maturity_lag_sessions,
            dispersion=dispersion,
        )

    # ------------------------------------------------------------------- lookup
    def find_published_evidence(self, *, snapshot_hash: str) -> tuple[str, ...]:
        """Evidence hashes already sealed against one outcome snapshot.

        Used by callers that must decide between running and waiting, so the
        decision is made from what is durable rather than from a flag.
        """

        root = self.store.root / "current" / "canonical-target-evidence"
        if not root.is_dir():
            return ()
        found: list[str] = []
        for path in sorted(root.glob("*.json")):
            evidence = self.store.load_canonical_target_evidence(path.stem)
            binding = self.store.load_canonical_target_recipe_binding(evidence.recipe_binding_hash)
            if binding.causal_outcome_snapshot_hash == snapshot_hash:
                found.append(evidence.evidence_hash)
        return tuple(found)

    def find_matched_canonical_inputs(
        self,
        *,
        candidate_id: str | None = None,
        development_score_store: AlphaDevelopmentArtifactStore | None = None,
    ) -> tuple[tuple[str, str], ...]:
        """Score-binding and dispersion hashes that belong to the same lineage.

        Scanning a directory and taking the lexicographically first file is not a
        selection: once more than one candidate, snapshot or horizon exists, the
        first score binding and the first forecast need not share a target
        binding at all, and the pairing would be decided by hash ordering.

        This pairs them by the identity that actually relates them -- the target
        recipe binding both descend from -- and returns every match, so a caller
        with more than one can refuse rather than pick.
        """

        forecasts_root = self.store.development_category_dir("canonical-dispersion-forecasts")
        if not forecasts_root.is_dir():
            return ()
        # A Stage 1 study seals the score binding in the reference arm's own
        # development store while the dispersion forecast lives beside the
        # target evidence in the study output. Pairing still happens here, by
        # the same lineage rule -- the caller only names where bindings may
        # additionally be enumerated, never which one wins.
        binding_stores = [self.store]
        if development_score_store is not None:
            binding_stores.append(development_score_store)
        forecasts = [
            self.store.load_canonical_dispersion_forecast(path.stem)
            for path in sorted(forecasts_root.glob("*.json"))
        ]
        matched: list[tuple[str, str]] = []
        pairs = [
            (store, path)
            for store in binding_stores
            for path in sorted(
                store.development_category_dir("canonical-score-bindings").glob("*.json")
            )
        ]
        for store, path in pairs:
            score_binding = store.load_canonical_score_binding(path.stem)
            if candidate_id is not None and score_binding.candidate_id != candidate_id:
                continue
            for forecast in forecasts:
                if (
                    forecast.target_recipe_binding_hash == score_binding.target_recipe_binding_hash
                    and forecast.target_evidence_hash == score_binding.target_evidence_hash
                ):
                    matched.append((score_binding.binding_hash, forecast.forecast_hash))
        return tuple(matched)


__all__ = [
    "CanonicalAlphaDevelopmentService",
    "CanonicalDevelopmentError",
    "CanonicalDevelopmentPublication",
    "CanonicalDevelopmentRequest",
    "CanonicalTargetLineage",
    "CanonicalTargetMaterialization",
]
