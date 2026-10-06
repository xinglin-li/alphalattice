"""Read one Risk development evidence graph back by hash, and cross-check it.

Replay previously validated one evidence JSON file and reported ``REUSED_EXACT``.
The surface it named, the diagnostics beside it, and every packed covariance
chunk could all have been deleted or edited, and replay would still have claimed
exact reuse.

Reading each artifact back is necessary but nowhere near sufficient. Every
artifact here is content-addressed, so each one *individually* verifies against
its own name no matter which run produced it. A surface from run A and
diagnostics from run B are both perfectly valid artifacts; what makes the pair
wrong is that they do not belong to each other. So this walks the graph and then
asserts the lineage that ties it to one Program and one authority.

The owner is the Risk Desk, not the Host. The relationships between a covariance
surface, its diagnostics and its chunks are Risk methodology; a Host that knew
them would accumulate every Desk's domain model as each Desk arrives. The Host
installs this verifier and knows only that some verifier exists for a kind.

The artifacts verified here are the *development* ones, which every Risk
capability shares. That is what lets this module stay method-neutral: it never
names a recipe schema, so a second capability needs no branch here either.

Nothing here can compute. This module must not import
``risk_research.experiments.execution``: replay resolves a verifier and must not
be able to reach an estimator through it, and the import graph is where that is
enforced rather than left to convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

from alphalattice.investment.risk_research.experiments.contracts import RISK_EXPERIMENT_KIND
from alphalattice.investment.risk_research.experiments.development_artifacts import (
    DEVELOPMENT_DIAGNOSTICS_CATEGORY,
    DEVELOPMENT_SURFACE_CATEGORY,
    RiskDevelopmentCovarianceSurface,
    RiskDevelopmentDiagnostics,
)
from alphalattice.investment.risk_research.experiments.series import (
    SERIES_CATEGORY,
    RiskDevelopmentSeries,
    combined_diagnostics,
    load_series_member,
)
from alphalattice.investment.risk_research.experiments.window import (
    RISK_INPUT_BINDING_CATEGORY,
    RiskDevelopmentInputBinding,
)
from alphalattice.investment.risk_research.surfaces.artifacts import (
    RiskArtifactError,
    RiskArtifactStore,
)
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExecutionEvidence,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)

_INPUT_BINDINGS = RISK_INPUT_BINDING_CATEGORY
_SURFACES = DEVELOPMENT_SURFACE_CATEGORY
_DIAGNOSTICS = DEVELOPMENT_DIAGNOSTICS_CATEGORY

_IDENTITY_FIELD = {
    _INPUT_BINDINGS: "input_binding_hash",
    _SURFACES: "surface_hash",
    _DIAGNOSTICS: "diagnostics_hash",
}

REQUIRED_CATEGORIES = (_INPUT_BINDINGS, _SURFACES, _DIAGNOSTICS)
"""Exactly one artifact of each, no more and no fewer.

Stated as a required set rather than a minimum. A missing category is an
incomplete graph; a duplicate is two candidate answers to "which surface did this
Program produce"; an unrecognised one is an artifact nobody verified travelling
inside a record that claims everything was verified.
"""


_RiskMetadata = tuple[
    RiskDevelopmentInputBinding, RiskDevelopmentCovarianceSurface, RiskDevelopmentDiagnostics
]
_RiskSeriesMetadata = tuple[
    RiskDevelopmentSeries, tuple[_RiskMetadata, ...], RiskDevelopmentDiagnostics
]


class RiskEvidenceVerifier:
    """Verify the input binding, development surface, diagnostics, and chunks."""

    kind = RISK_EXPERIMENT_KIND

    def __init__(self) -> None:
        self.method_standing: Literal["INSTALLED", "NOT_CURRENT"] | None = None
        self.method_refusal: str | None = None
        self._verified_key: tuple[Path, dict[str, Any]] | None = None
        self._verified_metadata: _RiskMetadata | None = None
        self._verified_series_metadata: _RiskSeriesMetadata | None = None

    def verify(
        self,
        *,
        program: SealedResearchProgram,
        evidence: ResearchExecutionEvidence,
        authority: ResolvedResearchAuthority | None,
        output_workspace: Path,
    ) -> None:
        # Artifact-only readback needs no mutable workspace. A series persists
        # its parent authority so each child scope can be derived and verified.
        self._verified_key = None
        self._verified_metadata = None
        self._verified_series_metadata = None
        self.method_standing = None
        self.method_refusal = None
        store = RiskArtifactStore(Path(output_workspace))
        if self._is_series(evidence):
            self._verified_series_metadata = self._verify_series(
                program=program, evidence=evidence, authority=authority, store=store
            )
            self._verified_key = (Path(output_workspace).resolve(), self._readback_key(evidence))
            return
        binding, surface, diagnostics = self._metadata(evidence, store)
        self._verify_root_lineage(program=program, evidence=evidence, binding=binding)
        self._verify_method_lineage(program=program, evidence=evidence, surface=surface)
        # A binding sealed before P also bound the code: it verifies and reads back as recorded,
        # and is historical -- a run of its Program is refused, a continuation plans anew.
        earlier = surface.program_binding.earlier_scheme
        self.method_standing = "NOT_CURRENT" if earlier else "INSTALLED"
        self.method_refusal = "risk_research.program_scheme_superseded" if earlier else None
        self._verify_input_lineage(binding=binding, surface=surface)
        self._verify_child_lineage(surface=surface, diagnostics=diagnostics)
        self._verify_chunks(store, surface)
        self._verified_metadata = binding, surface, diagnostics
        self._verified_key = (Path(output_workspace).resolve(), self._readback_key(evidence))

    def projection(
        self, *, evidence: ResearchExecutionEvidence, output_workspace: Path
    ) -> dict[str, Any]:
        """Project a just-verified graph once, or reopen metadata for standalone callers."""
        same_read = self._verified_key == (
            Path(output_workspace).resolve(),
            self._readback_key(evidence),
        )
        root = self._verified_metadata if same_read else None
        scoped = self._verified_series_metadata if same_read else None
        self._verified_key = None
        self._verified_metadata = None
        self._verified_series_metadata = None
        store = RiskArtifactStore(output_workspace)
        if self._is_series(evidence):
            series, children, diagnostics = scoped or self._series_metadata(evidence, store)
            surfaces = [surface.model_dump(mode="json") for _, surface, _ in children]
            projected = dict(surfaces[0])
            projected.pop("chunks")
            projected.update(
                kind="RiskDevelopmentSeries",
                surface_hash=series.series_hash,
                input_binding_hash=series.input_binding_hash,
                diagnostics_hash=series.diagnostics_hash,
                ordered_listing_ids=list(series.authority.ordered_listing_ids),
                formation_sessions=[str(day) for day in series.authority.sessions],
                scope_surfaces=surfaces,
            )
            limitations = [
                *projected["limitations"],
                "Each dated scope has its own matrix axis; "
                "stability comparisons restart at a scope boundary.",
            ]
            return {
                "risk_input": {
                    "kind": "RiskDevelopmentInputSeries",
                    "input_binding_hash": series.input_binding_hash,
                    "authority_hash": series.authority.authority_hash,
                    "panel_snapshot_hash": series.authority.panel_snapshot_hash,
                    "universe_revision_sha256": series.authority.universe_revision_sha256,
                    "ordered_listing_ids": list(series.authority.ordered_listing_ids),
                    "formation_sessions": [str(day) for day in series.authority.sessions],
                    "scope_inputs": [binding.model_dump(mode="json") for binding, _, _ in children],
                },
                "risk_surface": projected,
                "result": diagnostics.model_dump(mode="json"),
                "limitations": limitations,
            }
        binding, surface, diagnostics = root or self._metadata(evidence, store)
        return {
            "risk_input": binding.model_dump(mode="json"),
            "risk_surface": surface.model_dump(mode="json"),
            "result": diagnostics.model_dump(mode="json"),
            "limitations": list(surface.limitations),
        }

    @staticmethod
    def _readback_key(evidence: ResearchExecutionEvidence) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            evidence.model_dump(
                mode="json", exclude={"evidence_hash", "disposition", "numerical_call_count"}
            ),
        )

    @staticmethod
    def _is_series(evidence: ResearchExecutionEvidence) -> bool:
        return len(evidence.artifact_uris) == 1 and evidence.artifact_uris[0].startswith(
            f"playpen://risk-research/{SERIES_CATEGORY}/"
        )

    def _series_metadata(
        self, evidence: ResearchExecutionEvidence, store: RiskArtifactStore
    ) -> _RiskSeriesMetadata:
        try:
            series = RiskDevelopmentSeries.model_validate(
                store.load_json(
                    category=SERIES_CATEGORY,
                    uri=evidence.artifact_uris[0],
                    identity_field="series_hash",
                )
            )
            children = tuple(load_series_member(store, member) for member in series.members)
            diagnostics = RiskDevelopmentDiagnostics.model_validate(
                store.load_json(
                    category=_DIAGNOSTICS,
                    uri=store.uri(_DIAGNOSTICS, series.diagnostics_hash),
                    identity_field="diagnostics_hash",
                )
            )
            return series, children, diagnostics
        except (RiskArtifactError, OSError, ValueError) as error:
            raise AuthoringError(
                f"research_authoring.evidence_artifact_unverifiable:{error}"
            ) from error

    def _verify_series(
        self,
        *,
        program: SealedResearchProgram,
        evidence: ResearchExecutionEvidence,
        authority: ResolvedResearchAuthority | None,
        store: RiskArtifactStore,
    ) -> _RiskSeriesMetadata:
        series, children, diagnostics = self._series_metadata(evidence, store)
        if (
            series.program_hash != program.program_hash
            or series.authority.authority_hash != program.authority_hash
            or series.authority.authority_hash != evidence.authority_hash
            or (authority is not None and authority != series.authority)
            or series.input_binding_hash != evidence.desk_input_binding_hash
            or series.authority.sessions != evidence.formation_sessions
        ):
            raise AuthoringError("research_authoring.evidence_series_root_mismatch")
        for scope, (binding, surface, child_diagnostics) in zip(
            series.authority.listing_scopes, children, strict=True
        ):
            selected = series.authority.for_scope(scope)
            if (
                binding.authority_hash != selected.authority_hash
                or binding.ordered_listing_ids != selected.ordered_listing_ids
                or binding.formation_sessions != selected.sessions
                or binding.panel_snapshot_hash != selected.panel_snapshot_hash
                or binding.universe_revision_sha256 != selected.universe_revision_sha256
            ):
                raise AuthoringError("research_authoring.evidence_series_scope_mismatch")
            self._verify_method_lineage(program=program, evidence=evidence, surface=surface)
            self._verify_input_lineage(binding=binding, surface=surface)
            self._verify_child_lineage(surface=surface, diagnostics=child_diagnostics)
            self._verify_chunks(store, surface)
        earlier = any(surface.program_binding.earlier_scheme for _b, surface, _d in children)
        self.method_standing = "NOT_CURRENT" if earlier else "INSTALLED"
        self.method_refusal = "risk_research.program_scheme_superseded" if earlier else None
        expected = combined_diagnostics(
            input_binding_hash=series.input_binding_hash,
            children=tuple(value[2] for value in children),
        )
        if diagnostics != expected:
            raise AuthoringError("research_authoring.evidence_series_diagnostics_mismatch")
        if (
            tuple(row.formation_session for row in diagnostics.evaluations)
            != series.authority.sessions
        ):
            raise AuthoringError("research_authoring.evidence_series_formation_coverage_invalid")
        return series, children, diagnostics

    def _metadata(
        self, evidence: ResearchExecutionEvidence, store: RiskArtifactStore
    ) -> _RiskMetadata:
        by_category = self._partition(evidence.artifact_uris)
        try:
            payloads = {
                category: store.load_json(
                    category=category,
                    uri=uri,
                    identity_field=_IDENTITY_FIELD[category],
                )
                for category, uri in by_category.items()
            }
        except (RiskArtifactError, OSError, ValueError) as error:
            # A missing or tampered artifact is a replay failure, not a Risk
            # storage error leaking through the generic workflow.
            #
            # ``OSError`` is listed because the store only raises
            # ``RiskArtifactError`` for content it managed to read: a *deleted*
            # artifact surfaced as a bare ``FileNotFoundError``, escaping the
            # workflow as an unhandled error rather than a clean refusal. That
            # is the same "artifact is gone" case the rest of this module exists
            # to reject, and it is the more likely one in practice.
            raise AuthoringError(
                f"research_authoring.evidence_artifact_unverifiable:{error}"
            ) from error

        # Re-parsed through the contracts rather than read as dicts: a malformed
        # chunk list would otherwise silently verify zero chunks, and a dossier
        # missing a coverage field would silently skip its own cross-check.
        binding = self._parse(RiskDevelopmentInputBinding, payloads[_INPUT_BINDINGS])
        surface = self._parse(RiskDevelopmentCovarianceSurface, payloads[_SURFACES])
        diagnostics = self._parse(RiskDevelopmentDiagnostics, payloads[_DIAGNOSTICS])

        return binding, surface, diagnostics

    @staticmethod
    def _partition(artifact_uris: tuple[str, ...]) -> dict[str, str]:
        """Map each URI to its category, requiring exactly one of each."""

        found: dict[str, str] = {}
        for uri in artifact_uris:
            category = next(
                (
                    value
                    for value in REQUIRED_CATEGORIES
                    if uri.startswith(f"playpen://risk-research/{value}/")
                ),
                None,
            )
            if category is None:
                raise AuthoringError("research_authoring.evidence_artifact_uri_unknown")
            if category in found:
                raise AuthoringError("research_authoring.evidence_artifact_duplicated")
            found[category] = uri
        missing = tuple(value for value in REQUIRED_CATEGORIES if value not in found)
        if missing:
            raise AuthoringError(f"research_authoring.evidence_artifact_missing:{missing[0]}")
        return found

    @staticmethod
    def _parse[T](contract: type[T], payload: dict[str, Any]) -> T:
        try:
            return contract(**payload)
        except ValueError as error:
            raise AuthoringError(
                f"research_authoring.evidence_artifact_unverifiable:{error}"
            ) from error

    @staticmethod
    def _verify_root_lineage(
        *,
        program: SealedResearchProgram,
        evidence: ResearchExecutionEvidence,
        binding: RiskDevelopmentInputBinding,
    ) -> None:
        """Tie the graph to the Program being replayed, not merely to itself."""

        if binding.authority_hash != evidence.authority_hash:
            raise AuthoringError("research_authoring.evidence_input_binding_authority_mismatch")
        if binding.authority_hash != program.authority_hash:
            raise AuthoringError("research_authoring.evidence_authority_not_this_program")
        if binding.input_binding_hash != evidence.desk_input_binding_hash:
            raise AuthoringError("research_authoring.evidence_input_binding_mismatch")
        # Order included: the covariance axis is positional, so a same-shape
        # reordering is a different computation wearing the same identity.
        if tuple(binding.formation_sessions) != tuple(evidence.formation_sessions):
            raise AuthoringError("research_authoring.evidence_formation_axis_mismatch")

    @staticmethod
    def _verify_method_lineage(
        *,
        program: SealedResearchProgram,
        evidence: ResearchExecutionEvidence,
        surface: RiskDevelopmentCovarianceSurface,
    ) -> None:
        """Tie the surface to the Program's *methodology*, not just its authority.

        The gap this closes: two Programs over the same resolved authority --
        differing only in the method they selected -- produce two graphs whose
        input bindings are byte-identical. Every artifact is content-addressed, so
        each verifies against its own name; every one of them is a real artifact
        of a real run. Root and input lineage both pass. So Program A could claim
        exact reuse of the surface, diagnostics and chunks that Program B
        computed, and report ``REUSED_EXACT`` over numbers it never produced.

        The surface's embedded Program binding is what makes this checkable:
        parsing it already re-derived ``development_binding_hash`` from the
        selected adapter, recipe, domain, source closures and environment
        underneath it, so agreeing here means the numbers came from this
        Program's method rather than from something that merely shares its
        authority.
        """

        if surface.development_binding_hash != program.method_binding_hash:
            raise AuthoringError("research_authoring.evidence_surface_not_this_program")
        # The evidence record is sealed separately from the Program and travels
        # separately, so it is checked rather than assumed to agree.
        if surface.development_binding_hash != evidence.method_binding_hash:
            raise AuthoringError("research_authoring.evidence_method_binding_mismatch")
        if surface.program_binding.parameter_domain_hash != program.parameter_domain_hash:
            raise AuthoringError("research_authoring.evidence_parameter_domain_mismatch")
        # Governance, checked separately from methodology: a graph produced under
        # a different installed set is not this Program's graph even when the
        # selected method is identical.
        if surface.program_binding.catalog_hash != program.catalog_hash:
            raise AuthoringError("research_authoring.evidence_catalog_mismatch")

    @staticmethod
    def _verify_input_lineage(
        *,
        binding: RiskDevelopmentInputBinding,
        surface: RiskDevelopmentCovarianceSurface,
    ) -> None:
        if binding.input_binding_hash != surface.input_binding_hash:
            raise AuthoringError("research_authoring.evidence_surface_not_this_input_binding")
        # Order, not membership: the covariance axis is positional.
        if tuple(binding.ordered_listing_ids) != tuple(surface.ordered_listing_ids):
            raise AuthoringError("research_authoring.evidence_listing_axis_mismatch")
        computed = tuple(
            session for chunk in surface.chunks for session in chunk.formation_sessions
        )
        if computed != tuple(binding.formation_sessions):
            raise AuthoringError("research_authoring.evidence_chunk_session_axis_mismatch")

    @staticmethod
    def _verify_child_lineage(
        *,
        surface: RiskDevelopmentCovarianceSurface,
        diagnostics: RiskDevelopmentDiagnostics,
    ) -> None:
        """Reject a valid surface paired with valid but unrelated children."""

        if surface.diagnostics_hash != diagnostics.diagnostics_hash:
            raise AuthoringError("research_authoring.evidence_diagnostics_not_this_surface")
        if surface.input_binding_hash != diagnostics.input_binding_hash:
            raise AuthoringError("research_authoring.evidence_diagnostics_input_binding_mismatch")
        # The method the diagnostics describe must be the method the surface ran.
        if surface.recipe_hash != diagnostics.recipe_hash:
            raise AuthoringError("research_authoring.evidence_diagnostics_recipe_mismatch")
        if len(diagnostics.evaluations) != len(surface.formation_sessions):
            raise AuthoringError("research_authoring.evidence_diagnostics_coverage_mismatch")
        if tuple(value.formation_session for value in diagnostics.evaluations) != tuple(
            surface.formation_sessions
        ):
            raise AuthoringError("research_authoring.evidence_diagnostics_session_axis_mismatch")

    @staticmethod
    def _verify_chunks(store: RiskArtifactStore, surface: RiskDevelopmentCovarianceSurface) -> None:
        """Descend into the surface's own chunk list.

        ``chunk_path`` re-reads each chunk and checks its byte length and
        SHA-256, so this is where the packed matrices are actually verified
        rather than assumed present.
        """

        if not surface.chunks:
            raise AuthoringError("research_authoring.evidence_surface_has_no_chunks")
        try:
            for chunk in surface.chunks:
                store.chunk_path(chunk)
        except (RiskArtifactError, OSError) as error:
            raise AuthoringError(
                f"research_authoring.evidence_artifact_unverifiable:{error}"
            ) from error


__all__ = ["REQUIRED_CATEGORIES", "RiskEvidenceVerifier"]
