"""Re-derive a published Panel's observation-clock authority from installed owners.

A Panel manifest states which clock produced it. That statement is not evidence:
the artifact wrote it, so reading it back proves authorship and not authority.
This verifier answers the question the manifest cannot answer for itself -- would
*this build*, from its installed catalog, installed Formula recipes and installed
availability policy, produce exactly the identities this Panel carries?

Everything compared here is re-derived. Nothing is accepted from the manifest as
authority, and no hash supplied by a caller is honoured: the entry point takes a
snapshot hash and a resolver, never an expected identity.

Deliberately not a value verifier. Panel *content* is verified by
``ArtifactOnlyPanelRematerializer``, which re-reads the durable base closure. The
layer verified here is the one that used to have no owner at all: the clock, the
catalog binding that carries it, the per-Factor method identities that sit beside
it, and the manifest/semantic-index relations that quote them. Saying which layer
a replay covers is part of the receipt.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.catalog.contracts import (
    FeatureCatalog,
    source_availability_is_current,
)
from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    FEATURE_OBSERVATION_CLOCK_POLICY_ID,
    installed_feature_availability_policy,
)
from alphalattice.foundation.feature_engine.contracts import FeaturePanelBinding
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    catalog_implementation_hashes,
    catalog_methodology_hashes,
    default_extension_kernel_registry,
)
from alphalattice.foundation.feature_engine.producers.factors.registry import FeatureKernelRegistry
from alphalattice.kernel.shared_kernel.identity import canonical_hash

CLOCK_VERIFIED_LAYER = "CLOCK_AUTHORITY_RELATIONS_VERIFIED"
"""Exactly what a ``VERIFIED`` disposition covers, named rather than implied.

Relations, not values. This walks the catalog binding, the per-Formula clock,
method and source-authority identities, the installed availability owners, the
panel binding recomposition, the durable chunk children and the semantic-index
relation, all re-derived from installed owners. It does not re-read a single
Panel value, so it is never a numerical replay -- that is
``ArtifactOnlyPanelRematerializer``'s work, from the durable base closure.
"""


@dataclass(frozen=True)
class PanelObservationClockVerification:
    """Report a Panel clock-authority verification or named refusal."""

    disposition: Literal["VERIFIED", "REFUSED"]
    snapshot_hash: str
    verified_layer: str
    observation_clock_policy_id: str
    formula_observation_policy_hash: str
    source_availability_policy_hash: str
    catalog_hash: str
    factor_count: int
    chunk_count: int
    checks_passed: tuple[str, ...]
    failure_code: str | None = None
    failure_detail: str | None = None

    @property
    def verified(self) -> bool:
        """Report whether all installed clock-authority relations verified."""
        return self.disposition == "VERIFIED"


class FeaturePanelObservationClockVerifier:
    """Walk one published Panel down to the owners that decide its clock."""

    def __init__(
        self,
        *,
        resolver: ArtifactResolver,
        catalog: FeatureCatalog | None = None,
        kernel_registry: FeatureKernelRegistry | None = None,
    ) -> None:
        """Select installed catalog and Formula owners for re-derivation."""
        self._resolver = resolver
        self._catalog = catalog if catalog is not None else FeatureCatalog.load()
        self._registry = (
            kernel_registry if kernel_registry is not None else default_extension_kernel_registry()
        )

    def verify(self, snapshot_hash: str) -> PanelObservationClockVerification:
        """Re-derive clock identities for a sealed Panel without reading values."""
        availability = installed_feature_availability_policy()
        source_authorities = self._catalog.source_authorities
        binding = self._catalog.binding
        clocks = self._catalog.clocks_by_factor
        passed: list[str] = []

        def refuse(
            code: str, detail: str, chunk_count: int = 0
        ) -> PanelObservationClockVerification:
            return PanelObservationClockVerification(
                disposition="REFUSED",
                snapshot_hash=snapshot_hash,
                verified_layer=CLOCK_VERIFIED_LAYER,
                observation_clock_policy_id=FEATURE_OBSERVATION_CLOCK_POLICY_ID,
                formula_observation_policy_hash=binding.formula_observation_policy_hash,
                source_availability_policy_hash=binding.source_availability_policy_hash,
                catalog_hash=binding.catalog_hash,
                factor_count=len(clocks),
                chunk_count=chunk_count,
                checks_passed=tuple(passed),
                failure_code=code,
                failure_detail=detail,
            )

        try:
            manifest = self._resolver.load_feature_panel_manifest(
                self._resolver.feature_panel_manifest_uri(snapshot_hash)
            )
        except (FileNotFoundError, ValueError) as error:
            # A handle naming no readable manifest is unresolved, not unverified.
            # Refusing rather than raising keeps every outcome a disposition the
            # caller can map, which is what lets this run first on the real route.
            return refuse("feature_panel.manifest_unresolved", str(error))
        passed.append("manifest_payload_hash")
        summary = manifest.get("safe_summary")
        if not isinstance(summary, dict):
            return refuse("feature_panel.manifest_summary_absent", "safe_summary is not a mapping")
        lineage = summary.get("lineage")
        if not isinstance(lineage, dict):
            return refuse("feature_panel.manifest_lineage_absent", "lineage is not a mapping")
        factor_summary = summary.get("factor_catalog_summary")
        if not isinstance(factor_summary, dict):
            return refuse(
                "feature_panel.manifest_factor_summary_absent",
                "factor_catalog_summary is not a mapping",
            )

        # A Panel published before the observation-clock successor records no
        # clock identity at all. It stays readable under its stored identity and
        # is refused *here*, which is the whole point: legacy readback must not
        # be promotable into successor authority by passing a verifier.
        without_clock = sorted(
            factor_id
            for factor_id, entry in factor_summary.items()
            if not isinstance(entry, dict) or "observation_clock_hash" not in entry
        )
        if without_clock:
            return refuse(
                "feature_panel.observation_clock_authority_absent",
                f"{len(without_clock)} factors carry no observation clock identity",
            )
        passed.append("observation_clock_authority_present")

        if str(lineage.get("catalog_hash")) != binding.catalog_hash:
            return refuse(
                "feature_panel.catalog_binding_mismatch",
                f"manifest {lineage.get('catalog_hash')} != installed {binding.catalog_hash}",
            )
        passed.append("installed_catalog_binding")

        if set(factor_summary) != set(clocks):
            return refuse(
                "feature_panel.factor_axis_mismatch",
                "manifest factor axis differs from the installed catalog",
            )
        implementations = catalog_implementation_hashes(self._catalog, registry=self._registry)
        methodologies = catalog_methodology_hashes(self._catalog, registry=self._registry)
        authorities = self._catalog.formula_source_authorities
        for factor_id, clock in sorted(clocks.items()):
            declared = cast(Mapping[str, object], factor_summary[factor_id])
            for key, expected in (
                ("observation_clock_hash", clock.clock_hash),
                ("implementation_hash", implementations[factor_id]),
                ("methodology_hash", methodologies[factor_id]),
            ):
                if str(declared.get(key)) != expected:
                    return refuse(
                        "feature_panel.factor_authority_mismatch",
                        f"{factor_id}.{key} differs from the installed owner",
                    )
            # Re-derived from the installed field map and this Formula's declared
            # fields, never read back from the manifest and compared to itself.
            if (
                tuple(cast(list[str], declared.get("source_authority_ids") or ()))
                != (authorities[factor_id])
            ):
                return refuse(
                    "feature_panel.factor_source_authority_mismatch",
                    f"{factor_id} names source authorities the installed owners do not derive",
                )
        passed.append("per_factor_clock_implementation_and_methodology")

        # Availability is one authority for the catalog, so it is checked once
        # rather than seventy times -- and separately from the Formula clocks, so
        # a Provider schedule change is visible as itself.
        declared_availability = lineage.get("source_availability")
        if not isinstance(declared_availability, dict):
            return refuse(
                "feature_panel.availability_authority_absent",
                "lineage carries no source availability policy",
            )
        if str(declared_availability.get("policy_hash")) != availability.policy_hash:
            return refuse(
                "feature_panel.availability_policy_mismatch",
                "source availability policy differs from the installed one",
            )
        declared_owners = lineage.get("source_authorities")
        if not isinstance(declared_owners, dict):
            return refuse(
                "feature_panel.source_authority_catalog_absent",
                "lineage carries no installed source-authority catalog",
            )
        if not source_availability_is_current(
            str(declared_owners.get("catalog_hash")), source_authorities.catalog_hash
        ):
            return refuse(
                "feature_panel.source_authority_catalog_mismatch",
                "installed source-availability owners differ from the Panel's",
            )
        if str(lineage.get("source_authority_binding_hash")) != (
            binding.source_authority_binding_hash
        ):
            return refuse(
                "feature_panel.source_authority_binding_mismatch",
                "per-Formula source-authority binding differs from the installed one",
            )
        if (
            str(lineage.get("formula_observation_policy_hash"))
            != binding.formula_observation_policy_hash
        ):
            return refuse(
                "feature_panel.formula_observation_policy_mismatch",
                "Formula observation policy differs from the installed one",
            )
        passed.append("split_clock_availability_and_source_authority")

        rebuilt = FeaturePanelBinding.create(
            manifest_revision=str(lineage["manifest_revision"]),
            sector_revision=str(lineage["sector_revision"]),
            catalog_hash=binding.catalog_hash,
            spy_revision=str(lineage["spy_revision"]),
            policy_hash=str(lineage["policy_hash"]),
        )
        if rebuilt.panel_binding_hash != str(manifest["panel_binding_hash"]):
            return refuse(
                "feature_panel.panel_binding_mismatch",
                "panel binding does not recompose from its own lineage under this catalog",
            )
        passed.append("panel_binding_recomposition")

        chunks = manifest.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            return refuse("feature_panel.chunk_axis_absent", "manifest carries no chunks")
        for chunk in chunks:
            reference = cast(Mapping[str, object], chunk)
            try:
                path = self._resolver.resolve_feature_panel_chunk_ref(
                    uri=str(reference["uri"]),
                    content_hash=str(reference["chunk_hash"]),
                    metadata_hash=str(reference["metadata_hash"]),
                )
            except ValueError as error:
                return refuse("feature_panel.chunk_unresolved", str(error), chunk_count=len(chunks))
            if not Path(path).is_file():
                return refuse("feature_panel.chunk_absent", str(path), chunk_count=len(chunks))
        passed.append("durable_chunk_children_resolved")

        found = self._resolver.find_feature_panel_semantic_index(panel_snapshot_hash=snapshot_hash)
        if found is None:
            return refuse(
                "feature_panel.semantic_index_absent",
                "no semantic index is published for this snapshot",
                chunk_count=len(chunks),
            )
        index_payload, _uri = found
        index_identity = {key: value for key, value in index_payload.items() if key != "index_hash"}
        if canonical_hash(index_identity) != str(index_payload["index_hash"]):
            return refuse(
                "feature_panel.semantic_index_identity_invalid",
                "semantic index hash does not recompute from its own content",
                chunk_count=len(chunks),
            )
        if (
            str(index_payload["panel_snapshot_hash"]) != snapshot_hash
            or str(index_payload["catalog_hash"]) != binding.catalog_hash
            or str(index_payload["panel_content_hash"]) != str(manifest["panel_content_hash"])
            or str(index_payload["listing_set_hash"]) != str(manifest["listing_set_hash"])
            or int(index_payload["active_listing_count"]) != int(manifest["active_listing_count"])
        ):
            return refuse(
                "feature_panel.semantic_index_relation_mismatch",
                "semantic index does not bind this manifest under this catalog",
                chunk_count=len(chunks),
            )
        passed.append("semantic_index_relation")

        return PanelObservationClockVerification(
            disposition="VERIFIED",
            snapshot_hash=snapshot_hash,
            verified_layer=CLOCK_VERIFIED_LAYER,
            observation_clock_policy_id=FEATURE_OBSERVATION_CLOCK_POLICY_ID,
            formula_observation_policy_hash=binding.formula_observation_policy_hash,
            source_availability_policy_hash=binding.source_availability_policy_hash,
            catalog_hash=binding.catalog_hash,
            factor_count=len(clocks),
            chunk_count=len(chunks),
            checks_passed=tuple(passed),
        )


__all__ = [
    "CLOCK_VERIFIED_LAYER",
    "FeaturePanelObservationClockVerifier",
    "PanelObservationClockVerification",
]
