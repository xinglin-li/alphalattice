"""The content store's memo keeps a value only where a file system names a file for its life."""

from pathlib import Path

import pytest

from alphalattice.control.workspace_runtime.content_store import (
    verified_model_read_scope,
    verified_source_value,
)


@pytest.fixture
def kept_here(tmp_path: Path) -> bool:
    """Whether the store keeps values on this test's file system: one that names a file for its
    lifetime. On one that names none (overlayfs, 9p) the store keeps nothing and proves every read,
    so a test of what is kept counts its reads by this rule and keeps every other check."""
    path = tmp_path / "file-system" / "named.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"named")
    builds: list[bytes] = []

    def build() -> bytes:
        builds.append(path.read_bytes())
        return builds[-1]

    for _ in range(2):
        with verified_model_read_scope():
            verified_source_value(("file-system-probe", str(path)), (path,), build, nbytes=len)
    return len(builds) == 1
