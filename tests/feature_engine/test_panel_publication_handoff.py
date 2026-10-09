"""Publication ends when research can resolve the Panel, not when a file exists.

A Panel used to be "published" with a manifest, chunks and a lifecycle entry, and
`WorkspaceResearchAuthorityResolver` still refused it with
`snapshot_handle_unresolved` because no semantic index existed. The index was
built lazily by whoever happened to ask, so every Panel this product published
was complete by the publisher's standard and unusable by the research path.

The handoff is bounded by a gate that already existed: a Panel with no Feature
Input Gateway admission is one research must not read, and its index is withheld
on purpose. Completing the handoff unconditionally would have converted that
deliberate refusal into an accident. That gate also creates a durable state
nothing used to leave: admission arrives later, and the published Panel stays
unresolvable forever because no path ever revisits it.

Recovery had the matching gap: a build that failed after the base closure was
indistinguishable from one that never finished it, so the only recovery anybody
could describe was starting over -- which is how 452 completed listings were
discarded once. And "terminal" was asserted rather than derived, so a Panel with
an active snapshot and no semantic index reported the same stage as one research
could actually use.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.feature_engine.contracts import FeatureBuildStage
from alphalattice.foundation.feature_engine.panels.artifacts import (
    PanelArtifactCompositionOwner,
)
from alphalattice.foundation.feature_engine.panels.semantic_index import (
    FeaturePanelSemanticIndex,
    FeaturePanelSemanticIndexService,
)
from alphalattice.foundation.feature_engine.publication.persistence import (
    FeatureClosureDisposition,
)
from alphalattice.foundation.feature_engine.publication.snapshots import (
    FeaturePanelSnapshotPublisher,
)
from alphalattice.foundation.feature_engine.runtime.service import FeatureFoundationService
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

NOW = datetime(2026, 8, 19, 13, tzinfo=UTC)
PROFILE_ID = "panel-handoff-fixture"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


SNAPSHOT_HASH = _hash("snapshot")
MANIFEST_URI = ArtifactResolver.feature_panel_manifest_uri(SNAPSHOT_HASH)


def _manifest() -> UniverseManifest:
    profile = MarketProfile(
        market_profile_id=PROFILE_ID,
        display_name="Panel handoff fixture",
        market="US",
        currency="USD",
        calendar_id="XNYS",
        provider="fixture",
        daily_price_basis="unadjusted",
        manifest_as_of=NOW.date(),
        data_validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
    )
    return UniverseManifest(
        manifest_id=PROFILE_ID,
        profile=profile,
        listings=(
            ManifestListing(
                listing_id="listing-a", symbol="AAA", mic="XNYS", provider_symbol="AAA"
            ),
        ),
        revision_sha256=_hash("manifest"),
        universe_membership_basis="CURRENT_ACTIVE_SURVIVORS",
        is_point_in_time_historical=False,
    )


class _StubPanelState:
    """Only the two durable questions the stage classification actually asks."""

    def __init__(self, *, snapshot: dict[str, object] | None, gateway_qualified: bool) -> None:
        self.snapshot = snapshot
        self.gateway_qualified = gateway_qualified
        self.disclosure_calls = 0

    def feature_panel_snapshot_for_active(self, market_profile_id: str) -> dict[str, object] | None:
        assert market_profile_id == PROFILE_ID
        return self.snapshot

    def feature_input_quality_disclosure(
        self, *, result_manifest_revision: str
    ) -> dict[str, object]:
        self.disclosure_calls += 1
        return {"gateway_qualified": self.gateway_qualified}


class _StubClosure:
    def __init__(self, disposition: FeatureClosureDisposition) -> None:
        self.disposition = disposition
        self.expected_listing_ids: tuple[str, ...] = ()

    def closure_disposition(
        self, catalog_hash: str, *, expected_listing_ids: Any
    ) -> FeatureClosureDisposition:
        self.expected_listing_ids = tuple(expected_listing_ids)
        return self.disposition


class _PoisonedFeatureState:
    """Any Feature-store access at all fails the test that touches it."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"semantic-index recovery read the Feature store: {name}")


class _StubMarketData:
    """A workspace whose Universe journal has no bootstrap yet: the manifest is the axis."""

    @staticmethod
    def universe_bootstrap(market_profile_id: str) -> None:
        return None

    @staticmethod
    def membership_events(market_profile_id: str) -> tuple[object, ...]:
        return ()

    @staticmethod
    def research_listing_sources(manifest: Any) -> dict[str, str]:
        # Without a bootstrap the store's coverage is the manifest itself,
        # every member admitted by this revision.
        return {item.listing_id: manifest.revision_sha256 for item in manifest.listings}


def _service(
    *,
    resolver: ArtifactResolver,
    panel_state: _StubPanelState,
    closure: _StubClosure,
) -> FeatureFoundationService:
    """The real service, given only the collaborators this question needs.

    ``_durable_build_stage`` reads the closure owner, the Panel state, the
    artifact resolver and the Universe journal (for the calculation axis the
    closure must hold) and nothing else, so the rest stay unbuilt rather
    than faked into something that looks like a workspace.
    """

    return FeatureFoundationService(
        market_data=cast(Any, _StubMarketData()),
        feature_state=cast(Any, None),
        panel_state=cast(Any, panel_state),
        manifest=_manifest(),
        provider=cast(Any, None),
        mutation_gate=WorkspaceMutationGate(),
        panel_artifacts=PanelArtifactCompositionOwner(resolver),
        feature_persistence=cast(Any, closure),
        sector_activation=cast(Any, None),
    )


def _publish_index(resolver: ArtifactResolver, *, catalog_hash: str) -> FeaturePanelSemanticIndex:
    identity = {
        "kind": "FeaturePanelSemanticIndex",
        "panel_snapshot_hash": SNAPSHOT_HASH,
        "panel_content_hash": _hash("panel-content"),
        "catalog_hash": catalog_hash,
        "listing_set_hash": _hash("listing-set"),
        "active_listing_count": 1,
        "calendar_hash": canonical_hash([date(2026, 8, 5)]),
        "slice_algorithm_identity": "ordered-session-row-hash-digest",
        "sessions": [
            {
                "session_date": "2026-08-05",
                "row_count": 1,
                "ordered_row_hash_digest": _hash("ordered-rows"),
            }
        ],
    }
    index = FeaturePanelSemanticIndex(**identity, index_hash=str(canonical_hash(identity)))
    resolver.publish_feature_panel_semantic_index(
        payload=index.model_dump(mode="json"), index_hash=index.index_hash
    )
    return index


def test_recovery_stages_name_one_recovery_each() -> None:
    """Recovery stages name one recovery each."""

    assert {stage.value for stage in FeatureBuildStage} == {
        "base_closure_incomplete",
        "base_closure_complete_panel_pending",
        "transition_recovery_pending",
        "closure_authority_unavailable",
        "panel_published_awaiting_gateway_admission",
        "panel_semantic_index_pending",
        "terminal_panel_complete",
    }


@pytest.mark.parametrize(
    ("disposition", "expected"),
    [
        ("CLOSURE_ABSENT", FeatureBuildStage.BASE_CLOSURE_INCOMPLETE),
        ("MEMBERSHIP_INCOMPLETE", FeatureBuildStage.BASE_CLOSURE_INCOMPLETE),
        ("TRANSITION_RECOVERY_PENDING", FeatureBuildStage.TRANSITION_RECOVERY_PENDING),
        ("CLOSURE_AUTHORITY_UNAVAILABLE", FeatureBuildStage.CLOSURE_AUTHORITY_UNAVAILABLE),
    ],
)
def test_the_closure_owner_decides_the_closure_half_of_the_stage(
    tmp_path: Path, disposition: str, expected: FeatureBuildStage
) -> None:
    """The closure owner decides the closure half of the stage."""

    resolver = ArtifactResolver(tmp_path / "artifacts")
    closure = _StubClosure(cast(FeatureClosureDisposition, disposition))
    panel_state = _StubPanelState(
        snapshot={"snapshot_hash": SNAPSHOT_HASH, "manifest_uri": MANIFEST_URI},
        gateway_qualified=True,
    )
    service = _service(resolver=resolver, panel_state=panel_state, closure=closure)
    # A published, gateway-qualified, fully indexed Panel sits underneath, so the
    # only reason any of these is not terminal is the closure itself.
    _publish_index(resolver, catalog_hash=service.catalog.binding.catalog_hash)
    panel_state.snapshot = {
        "snapshot_hash": SNAPSHOT_HASH,
        "manifest_uri": MANIFEST_URI,
        "catalog_hash": service.catalog.binding.catalog_hash,
    }

    assert service._durable_build_stage() == expected
    assert closure.expected_listing_ids == ("listing-a",)


def test_an_active_snapshot_without_a_semantic_index_is_not_terminal(tmp_path: Path) -> None:
    """An active snapshot without a semantic index is not terminal."""

    resolver = ArtifactResolver(tmp_path / "artifacts")
    closure = _StubClosure("MEMBERSHIP_COMPLETE")
    panel_state = _StubPanelState(snapshot=None, gateway_qualified=True)
    service = _service(resolver=resolver, panel_state=panel_state, closure=closure)
    catalog_hash = service.catalog.binding.catalog_hash

    assert service._durable_build_stage() == FeatureBuildStage.BASE_CLOSURE_COMPLETE_PANEL_PENDING

    # A snapshot produced by some other catalog is not this build's Panel.
    panel_state.snapshot = {
        "snapshot_hash": SNAPSHOT_HASH,
        "manifest_uri": MANIFEST_URI,
        "catalog_hash": "0" * 64,
    }
    assert service._durable_build_stage() == FeatureBuildStage.BASE_CLOSURE_COMPLETE_PANEL_PENDING

    panel_state.snapshot = {
        "snapshot_hash": SNAPSHOT_HASH,
        "manifest_uri": MANIFEST_URI,
        "catalog_hash": catalog_hash,
    }
    panel_state.gateway_qualified = False
    assert (
        service._durable_build_stage()
        == FeatureBuildStage.PANEL_PUBLISHED_AWAITING_GATEWAY_ADMISSION
    ), "a Panel whose inputs the Gateway never admitted is gated, not indexed and not terminal"

    panel_state.gateway_qualified = True
    assert service._durable_build_stage() == FeatureBuildStage.PANEL_SEMANTIC_INDEX_PENDING

    _publish_index(resolver, catalog_hash=catalog_hash)
    assert service._durable_build_stage() == FeatureBuildStage.TERMINAL_PANEL_COMPLETE


class _RecordingIndexOwner:
    """Stands in for the one real index writer, and counts what it is asked to do."""

    def __init__(self, resolver: ArtifactResolver, *, catalog_hash: str) -> None:
        self.resolver = resolver
        self.catalog_hash = catalog_hash
        self.calls: list[str] = []

    def obtain(self, panel_manifest_ref: str) -> tuple[FeaturePanelSemanticIndex, str, bool]:
        self.calls.append(panel_manifest_ref)
        index = _publish_index(self.resolver, catalog_hash=self.catalog_hash)
        return index, "playpen://feature-panel/semantic-index/" + index.index_hash, True


def _publisher(
    *, resolver: ArtifactResolver, panel_state: _StubPanelState, index_owner: Any
) -> FeaturePanelSnapshotPublisher:
    return FeaturePanelSnapshotPublisher(
        feature_state=cast(Any, _PoisonedFeatureState()),
        panel_state=cast(Any, panel_state),
        resolver=resolver,
        mutation_gate=WorkspaceMutationGate(),
        recovery_binding=cast(Any, None),
        logical_identity=cast(Any, None),
        semantic_index=cast(Any, index_owner),
    )


def test_semantic_index_recovery_recomputes_no_feature_value(tmp_path: Path) -> None:
    """Semantic index recovery recomputes no feature value."""

    resolver = ArtifactResolver(tmp_path / "artifacts")
    from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog

    catalog_hash = str(FeatureCatalog.load().binding.catalog_hash)
    panel_state = _StubPanelState(
        snapshot={
            "snapshot_hash": SNAPSHOT_HASH,
            "manifest_uri": MANIFEST_URI,
            "catalog_hash": catalog_hash,
        },
        gateway_qualified=True,
    )
    index_owner = _RecordingIndexOwner(resolver, catalog_hash=catalog_hash)
    publisher = _publisher(resolver=resolver, panel_state=panel_state, index_owner=index_owner)

    assert resolver.find_feature_panel_semantic_index(panel_snapshot_hash=SNAPSHOT_HASH) is None
    assert publisher.complete_semantic_index_handoff(manifest=_manifest()) == SNAPSHOT_HASH
    assert index_owner.calls == [MANIFEST_URI]
    assert resolver.find_feature_panel_semantic_index(panel_snapshot_hash=SNAPSHOT_HASH) is not None

    # Idempotent: a second maintenance cycle must not write a second index, and
    # must not ask the owner to derive one it already has.
    assert publisher.complete_semantic_index_handoff(manifest=_manifest()) is None
    assert index_owner.calls == [MANIFEST_URI]


def test_recovery_respects_the_gate_that_withheld_the_index(tmp_path: Path) -> None:
    """Recovery respects the gate that withheld the index."""

    resolver = ArtifactResolver(tmp_path / "artifacts")
    from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog

    catalog_hash = str(FeatureCatalog.load().binding.catalog_hash)
    panel_state = _StubPanelState(
        snapshot={
            "snapshot_hash": SNAPSHOT_HASH,
            "manifest_uri": MANIFEST_URI,
            "catalog_hash": catalog_hash,
        },
        gateway_qualified=False,
    )
    index_owner = _RecordingIndexOwner(resolver, catalog_hash=catalog_hash)
    publisher = _publisher(resolver=resolver, panel_state=panel_state, index_owner=index_owner)

    assert publisher.complete_semantic_index_handoff(manifest=_manifest()) is None
    assert index_owner.calls == []
    assert resolver.find_feature_panel_semantic_index(panel_snapshot_hash=SNAPSHOT_HASH) is None

    # The admission arrives later, and the same path finishes the handoff.
    panel_state.gateway_qualified = True
    assert publisher.complete_semantic_index_handoff(manifest=_manifest()) == SNAPSHOT_HASH
    assert index_owner.calls == [MANIFEST_URI]


def test_the_semantic_index_orders_each_sessions_row_hashes_by_listing_as_before() -> None:
    """The semantic index orders each session's row hashes by listing."""

    import pyarrow as pa

    rows = [
        (date(2026, 1, 6), "Ünion", "c" * 64),
        (date(2026, 1, 5), "zeta", "d" * 64),
        (date(2026, 1, 6), "Alpha", "e" * 64),
        (date(2026, 1, 5), "Ünion", "a" * 64),
        (date(2026, 1, 5), "zeta", "b" * 64),
        (date(2026, 1, 6), "zeta", "f" * 64),
    ]
    batches = [
        pa.RecordBatch.from_pylist(
            [{"session_date": s, "listing_id": n, "row_hash": h} for s, n, h in part],
            schema=pa.schema(
                [
                    ("session_date", pa.date32()),
                    ("listing_id", pa.string()),
                    ("row_hash", pa.string()),
                ]
            ),
        )
        for part in (rows[:4], rows[4:])
    ]
    manifest = {
        "snapshot_hash": _hash("snapshot"),
        "panel_content_hash": _hash("content"),
        "listing_set_hash": _hash("listings"),
        "active_listing_count": 3,
        "safe_summary": {"lineage": {"catalog_hash": _hash("catalog")}},
    }
    published: dict[str, object] = {}

    class _Resolver:
        def load_feature_panel_manifest(self, _ref: str) -> dict[str, object]:
            return manifest

        def find_feature_panel_semantic_index(self, **_: object) -> None:
            return None

        def publish_feature_panel_semantic_index(self, *, payload: object, index_hash: str):
            published[index_hash] = payload
            return type("Descriptor", (), {"uri": index_hash})()

    class _Reader:
        def identity_batches(self, _ref: str):
            return iter(batches)

    service = FeaturePanelSemanticIndexService(cast(ArtifactResolver, _Resolver()))
    service.reader = cast(Any, _Reader())
    index, _uri, built = service.obtain("manifest")
    expected: dict[date, list[tuple[str, str]]] = {}
    for session, listing, row_hash in rows:
        expected.setdefault(session, []).append((listing, row_hash))
    assert built and published
    assert [
        (item.session_date, item.row_count, item.ordered_row_hash_digest) for item in index.sessions
    ] == [
        (session, len(values), canonical_hash([h for _n, h in sorted(values)]))
        for session, values in sorted(expected.items())
    ]
