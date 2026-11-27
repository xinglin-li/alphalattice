from concurrent.futures import Future
from pathlib import Path

import pytest

from alphalattice.control.workspace_runtime import content_store
from alphalattice.kernel.identity import blake3_from_filename

TEST_SOURCE = "src/alphalattice/control/workspace_runtime/content_store.py"


class _Interrupted(Future):
    """A check still running when the waiting thread is interrupted (Ctrl-C)."""

    def result(self, timeout=None):
        raise KeyboardInterrupt


@pytest.mark.proves("workspace_runtime.an_interrupt_ends_a_source_check_wait")
def test_an_interrupt_while_waiting_on_checks_is_raised_at_once():
    """An interrupt raised while a check still runs ends the wait at once; a finished check's own
    failure is kept, the first in input order."""
    failed: Future = Future()
    failed.set_exception(ValueError("first input"))
    later: Future = Future()
    later.set_exception(ValueError("second input"))
    assert str(content_store._joined([failed, later])) == "first input"
    with pytest.raises(KeyboardInterrupt):
        content_store._joined([failed, _Interrupted()])


@pytest.mark.proves("workspace_runtime.tagged_artifact_paths_are_portable")
@pytest.mark.parametrize(
    "portable",
    [
        pytest.param(False, id="raw digest component"),
        pytest.param(True, id="portable digest component"),
    ],
)
def test_a_tagged_document_has_one_portable_filename(tmp_path: Path, portable: bool) -> None:
    """A document writes and reads at the encoded path or refuses before touching disk."""
    store = content_store.ContentAddressedStore(tmp_path / "store", uri_prefix="artifact://qa")
    digest = "blake3:" + "a" * 64
    if not portable:
        with pytest.raises(
            content_store.ContentAddressedStoreError, match=r"content_store\.path_component_invalid"
        ):
            store.atomic_write(store.root / digest / "record.json", b"discard")
        with pytest.raises(
            content_store.ContentAddressedStoreError, match=r"content_store\.path_component_invalid"
        ):
            store.artifact_path(category=f"records/{digest}", content_hash=digest)
        assert not store.root.exists()
        return

    path = store.artifact_path(category="records", content_hash=digest)
    assert ":" not in path.name and blake3_from_filename(path.stem) == digest
    store.atomic_write(path, b"kept")
    assert path.read_bytes() == b"kept"
