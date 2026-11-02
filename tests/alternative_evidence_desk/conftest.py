"""Private HTTP workspaces copied from a completed Portfolio publication."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from tests.alternative_evidence_desk.review_http_support import build_workspace


@pytest.fixture(scope="module")
def built_http_book(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Any]:
    return build_workspace(tmp_path_factory.mktemp("evidence-http-book"))


@pytest.fixture
def http_book(built_http_book, tmp_path: Path) -> tuple[Path, Any]:
    """Keep the original per-test layout and independently writable files."""
    source, report = built_http_book
    workspace = tmp_path / "portfolio" / "workspace"
    shutil.copytree(source, workspace)
    return workspace, report
