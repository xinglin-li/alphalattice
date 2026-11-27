"""Panel artifacts are resolved by their stored identities."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

from alphalattice.capabilities.domain_contracts import identity_successors
from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.capabilities.domain_contracts.identity_successors import IdentityMove
from alphalattice.control.workspace_runtime.artifacts import ArtifactDescriptor, ArtifactResolver
from alphalattice.kernel.identity import blake3_filename
from tests._support.identity_fixtures import fixture_digest

TEST_SOURCE = "src/alphalattice/control/workspace_runtime/artifacts.py"

HASH = fixture_digest("a")


@pytest.mark.proves("workspace_runtime.feature_panel_chunk_refuses_another_kind")
def test_a_descriptor_of_another_kind_is_refused_as_not_a_panel_chunk(tmp_path: Path):
    """A feature input's descriptor never resolves as a Panel chunk, whatever its hashes say."""
    descriptor = ArtifactDescriptor(
        kind="feature-input",
        content_hash=HASH,
        metadata_hash=HASH,
        uri=ArtifactResolver.feature_panel_chunk_uri(HASH),
    )
    with pytest.raises(ValueError, match="not a feature panel chunk"):
        ArtifactResolver(tmp_path).resolve_feature_panel_chunk(descriptor)


@pytest.mark.proves("workspace_runtime.feature_panel_chunk_refuses_a_foreign_uri")
def test_a_uri_that_does_not_match_its_content_hash_is_refused(tmp_path: Path):
    """A chunk's address must name its own content hash; another hash's address is refused."""
    descriptor = ArtifactDescriptor(
        kind="feature-panel-chunk",
        content_hash=HASH,
        metadata_hash=HASH,
        uri=ArtifactResolver.feature_panel_chunk_uri(fixture_digest("b")),
    )
    with pytest.raises(ValueError, match="URI does not match its content hash"):
        ArtifactResolver(tmp_path).resolve_feature_panel_chunk(descriptor)


@pytest.mark.proves("workspace_runtime.feature_panel_chunk_resolves_to_its_stored_file")
def test_a_published_chunk_resolves_to_its_stored_parquet_file(tmp_path: Path):
    """A chunk published with its descriptor resolves, by that descriptor, to the Parquet file the
    publisher wrote under its content hash."""
    resolver = ArtifactResolver(tmp_path)
    descriptor = resolver.publish_feature_panel_chunk(
        table=pa.table({"value": [1.0, 2.0]}), content_hash=HASH, metadata={}
    )
    path = resolver.resolve_feature_panel_chunk(descriptor)
    assert path == resolver.root / "feature-panel" / "chunks" / f"{blake3_filename(HASH)}.parquet"
    assert path.is_file()


def _preprocessing_marker(resolver, revision, policy="e", receipt="c"):
    implementation = dict(
        implementation_id="fixture", implementation_content_hash=fixture_digest(revision)
    )
    implementation_hash = canonical_hash(implementation)
    method = {
        "recipe_id": "robust",
        "recipe_hash": HASH,
        "implementation_id": implementation["implementation_id"],
        "implementation_binding_hash": implementation_hash,
        "catalog_hash": fixture_digest(revision),
        "panel_binding_hash": fixture_digest("b"),
    }
    binding = {
        **method,
        "kind": "PanelPreprocessingBinding",
        "policy_hash": fixture_digest(policy),
        "implementation": dict(implementation, implementation_binding_hash=implementation_hash),
    }
    binding_hash = canonical_hash(binding)
    resolver.publish_panel_preprocessing_binding(
        payload=dict(binding, binding_hash=binding_hash), binding_hash=binding_hash
    )
    marker = dict(
        method,
        kind="PanelPreprocessingSealMarker",
        panel_content_hash=HASH,
        preprocessing_binding_hash=binding_hash,
        clipping_evidence_hash=fixture_digest(receipt),
    )
    return dict(marker, marker_hash=canonical_hash(marker))


@pytest.mark.proves("workspace_runtime.panel_preprocessing_marker_follows_recorded_methods")
@pytest.mark.parametrize(
    ("before", "after", "policy", "refusal", "fault"),
    [
        pytest.param("4", "4", "e", None, None, id="current"),
        pytest.param("3", "4", "e", None, None, id="one-predecessor"),
        pytest.param("2", "4", "e", None, None, id="two-predecessors"),
        pytest.param("5", "4", "e", "contradicts the sealed method", None, id="unrecorded"),
        pytest.param("4", "3", "e", "contradicts the sealed method", None, id="reverse"),
        pytest.param("3", "4", "f", "contradicts the sealed method", None, id="different-policy"),
        pytest.param(
            "3", "4", "e", "identity is invalid", "clipping_evidence_hash", id="changed-seal"
        ),
        pytest.param(
            "3", "4", "e", "does not name this panel", "panel_content_hash", id="wrong-panel"
        ),
    ],
)
def test_a_sealed_panel_method_follows_only_its_recorded_successors(
    tmp_path, monkeypatch, before, after, policy, refusal, fault
):
    resolver = ArtifactResolver(tmp_path)
    markers = {key: _preprocessing_marker(resolver, key) for key in "234"}
    roles = {
        "catalog_hash": "feature_engine.preprocessing_catalog",
        "implementation_binding_hash": "feature_engine.preprocessing_implementation.robust",
    }
    moves = tuple(
        IdentityMove(role, markers[old][field], markers[new][field], "fixture", "same method")
        for field, role in roles.items()
        for old, new in (("2", "3"), ("3", "4"))
    )
    monkeypatch.setattr(identity_successors, "recorded_moves", lambda root=None: moves)
    publish = resolver.publish_panel_preprocessing_marker
    original = _preprocessing_marker(resolver, before)
    first = publish(payload=original, panel_content_hash=HASH)
    proposed = _preprocessing_marker(resolver, after, policy, receipt="d")
    path = tmp_path / "feature-panel" / "preprocessing-markers" / f"{blake3_filename(HASH)}.json"
    sealed = path.read_bytes()
    if fault:
        changed = dict(original, **{fault: fixture_digest("f")})
        if fault == "panel_content_hash":
            changed.pop("marker_hash")
            changed["marker_hash"] = canonical_hash(changed)
        sealed = json.dumps(changed).encode("utf-8")
        path.write_bytes(sealed)

    if refusal is None:
        assert publish(payload=proposed, panel_content_hash=HASH) == first
    else:
        with pytest.raises(ValueError, match=f"panel preprocessing marker {refusal}"):
            publish(payload=proposed, panel_content_hash=HASH)
    if fault:
        with pytest.raises(ValueError, match=f"panel preprocessing marker {refusal}"):
            resolver.load_panel_preprocessing_marker(HASH)
    else:
        assert resolver.load_panel_preprocessing_marker(HASH) == original
    assert path.read_bytes() == sealed
