"""A closure folder keeps its files in place and lists them by name or by their last write."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from alphalattice.foundation.feature_engine.panels.closure_files import FolderFiles

TEST_SOURCE = "src/alphalattice/foundation/feature_engine/panels/closure_files.py"


@pytest.mark.proves("feature_engine.closure_folder_files_hold_their_bytes")
def test_a_folder_closure_writes_lists_reads_and_removes_its_files(tmp_path: Path):
    """The folder reader a closure root outside any workspace uses writes each file whole, lists
    its folder's files sorted, reads them back, and removes one without touching the rest."""
    files = FolderFiles(tmp_path / "closure")
    files.write("sector/a.json", b"{}")
    files.write("sector/b.json", b"[]")
    assert files.names("sector") == ["a.json", "b.json"]
    assert files.exists("sector/a.json")
    assert files.read("sector/a.json") == b"{}"
    assert files.read("sector/missing.json") is None
    files.remove("sector/a.json")
    assert files.names("sector") == ["b.json"]
    assert files.read("sector/b.json") == b"[]"


@pytest.mark.proves("feature_engine.closure_folder_lists_files_by_their_last_write")
def test_a_closure_folder_lists_its_files_by_their_last_write(tmp_path: Path):
    """The move reads a folder's files in their write order: the file written last comes last,
    and a staged file is no file of the folder."""
    files = FolderFiles(tmp_path / "closure")
    for name, written in (("a.json", 3), ("b.json", 1), ("c.json", 2)):
        files.write(f"sector/{name}", b"{}")
        os.utime(tmp_path / "closure" / "sector" / name, (written * 1000, written * 1000))
    (tmp_path / "closure" / "sector" / ".d.json.1.tmp").write_bytes(b"{}")
    assert files.written("sector") == ["b.json", "c.json", "a.json"]
    assert files.names("sector") == ["a.json", "b.json", "c.json"]
