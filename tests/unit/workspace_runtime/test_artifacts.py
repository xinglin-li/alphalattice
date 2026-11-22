"""A feature panel chunk resolves to its stored file only when its descriptor names it exactly."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from alphalattice.control.workspace_runtime.artifacts import ArtifactDescriptor, ArtifactResolver

TEST_SOURCE = "src/alphalattice/control/workspace_runtime/artifacts.py"

HASH = "a" * 64


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
        uri=ArtifactResolver.feature_panel_chunk_uri("b" * 64),
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
    assert path == resolver.root / "feature-panel" / "chunks" / f"{HASH}.parquet"
    assert path.is_file()
