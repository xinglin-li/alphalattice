"""Private HTTP workspaces copied from a completed Portfolio publication."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from alphalattice.control.task_control.upgrade import carry_tasks
from alphalattice.kernel.data_layer.history import carry_arrays
from tests._support.project_discovery import project_discovery as project_discovery
from tests.alternative_evidence_desk.planted_corpus import review_store
from tests.alternative_evidence_desk.review_http_support import build_workspace
from tests.pytest_environment import upgraded


@pytest.fixture(scope="module")
def built_http_book(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Any]:
    workspace, report = build_workspace(tmp_path_factory.mktemp("evidence-http-book"))
    carry_tasks(workspace, workspace)  # its Tasks go with it past the machine that made them
    carry_arrays(workspace, workspace)  # and its lanes
    return workspace, report


@pytest.fixture
def http_book(built_http_book, tmp_path: Path) -> tuple[Path, Any]:
    """Keep the original per-test layout and independently writable files."""
    source, report = built_http_book
    workspace = tmp_path / "portfolio" / "workspace"
    shutil.copytree(source, workspace)
    upgraded(workspace)
    return workspace, report


@pytest.fixture
def review_on_the_store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """An evidence review on the Host's store with the planted models, for a module that runs one
    outside the review packages (whose conftests apply it to every test)."""
    with review_store(tmp_path_factory.mktemp("review-store")):
        yield
