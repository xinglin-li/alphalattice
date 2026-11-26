"""Session-scoped real workspaces, shared by the methodology packages' conftests.

Fixture functions, imported by name into each package's `conftest.py`, so that
one builder produces the same workspace wherever a package needs it while each
package still pays only for the workspaces its own cases resolve against.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from alphalattice.foundation.data_layer import history
from tests._support import development_universe
from tests.researcher_methodology_surface.real_workspace import (
    DEVELOPMENT_SYMBOLS,
    FACTOR_SCALE,
    SUFFICIENT_SYMBOLS,
    SYMBOLS,
    RealRiskWorkspace,
    build_real_risk_workspace,
    development_feature_catalog,
)


@contextmanager
def open_outcome_catalog(root: Path) -> Iterator[Path]:
    """Isolate causal outcome recording from the workspace's own catalog."""
    with history.own_root(root):
        history.open_history("methodology-outcomes", "Methodology outcomes", root, root=root)
        yield root


def shared_outcome_catalog(factory: pytest.TempPathFactory) -> Path:
    """The process's one outcome catalog. The session's workspaces are shared across modules,
    and a workspace's publication reads its outcome bodies from the catalog that recorded them,
    so every module publishing on them records here."""
    return factory.getbasetemp() / "methodology-outcome-catalog"


SESSION_WORKSPACES = frozenset(
    {
        "development_risk_workspace",
        "extension_feature_workspace",
        "external_factor_workspace",
        "real_risk_workspace",
    }
)


@pytest.fixture(scope="module")
def outcome_catalog_root(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Path]:
    """Hold the module's isolated catalog for its shared publication fixtures. A session
    workspace builds under a root of its own, which is refused while the catalog records, so the
    workspaces this module's cases use are built before the catalog opens: a later case's first
    use would otherwise build inside the hold an earlier case left open."""
    used = {
        name
        for item in request.session.items
        if getattr(item, "module", None) is request.module
        for name in item.fixturenames
    }
    for name in sorted(used & SESSION_WORKSPACES):
        request.getfixturevalue(name)
    with open_outcome_catalog(shared_outcome_catalog(tmp_path_factory)) as root:
        yield root


@pytest.fixture(scope="session")
def real_risk_workspace(tmp_path_factory: pytest.TempPathFactory) -> RealRiskWorkspace:
    """Build through product writers once per tree; share a golden across processes
    and copy it for each worker. The cache root can be set by ALPHALATTICE_TEST_GOLDEN_ROOT.
    """

    return build_real_risk_workspace(
        tmp_path_factory.mktemp("real-risk-workspace"), factory=tmp_path_factory
    )


@pytest.fixture(scope="session")
def development_risk_workspace(tmp_path_factory: pytest.TempPathFactory) -> RealRiskWorkspace:
    return build_real_risk_workspace(
        tmp_path_factory.mktemp("development-workspace"),
        symbols=DEVELOPMENT_SYMBOLS,
        factory=tmp_path_factory,
    )


@pytest.fixture(scope="session")
def extension_feature_workspace(tmp_path_factory: pytest.TempPathFactory) -> RealRiskWorkspace:
    """A second real workspace, composed with the extension catalog revision.

    Deliberately a separate workspace rather than a re-publication over the
    shared one. The factor axis is fixed when the closure is opened -- genesis,
    the persistence coordinator and the panel publisher are all composed from one
    catalog -- so installing a revision afterwards would not be an installation,
    it would be a rewrite of a workspace that had already committed to a
    different axis.

    Session-scoped, and it costs roughly what the shared workspace costs, which
    is the price of proving materialization through the real service rather than
    asserting that a registry lookup succeeds.
    """

    # Fresh, not restored: this workspace's claim is the materialization itself,
    # and the routed lane keeps the extension branch of the materializer covered
    # only by driving it.
    return build_real_risk_workspace(
        tmp_path_factory.mktemp("extension-feature-workspace"),
        feature_catalog=development_feature_catalog(),
        fresh=True,
    )


@pytest.fixture(scope="session")
def external_factor_workspace(tmp_path_factory: pytest.TempPathFactory) -> RealRiskWorkspace:
    """A real workspace whose Panel carries a Factor the product does not own.

    The catalog revision and the kernel registry travel together, and both come
    from ``external_factor_method`` -- a module outside ``alphalattice`` that this
    build has never heard of. That is the whole claim being made: an outside
    method reaches a published Panel, a sealed Program and durable evidence
    without a line of product source changing.

    A third session-scoped workspace rather than a reuse of either existing one.
    The factor axis is fixed when the closure is opened, so installing a revision
    afterwards would be a rewrite of a workspace that had already committed to a
    different axis.

    A warmed golden's copy: built fresh in setup, it timed out each worker's first case.
    """

    return build_external_factor_workspace(tmp_path_factory.mktemp("external-factor"), fresh=False)


def factor_scale_workspace(factory: pytest.TempPathFactory) -> RealRiskWorkspace:
    """The sufficient test data set at FACTOR_SCALE. A module whose studies run the walk-forward
    policy, which needs that history, overrides ``real_risk_workspace`` with it; every such module
    copies the one golden the nightly builds."""

    return build_real_risk_workspace(
        factory.mktemp("factor-scale-workspace"),
        factory=factory,
        symbols=SUFFICIENT_SYMBOLS,
        standard=FACTOR_SCALE,
    )


def build_external_factor_workspace(
    root: Path, *, fresh: bool, standard=development_universe.STANDARD, symbols=SYMBOLS
) -> RealRiskWorkspace:
    """The outside method's catalog revision and kernel registry, installed together."""

    from tests.researcher_methodology_surface.external_factor_method import (
        external_feature_catalog,
        external_kernel_registry,
    )

    return build_real_risk_workspace(
        root,
        feature_catalog=external_feature_catalog(),
        feature_kernels=external_kernel_registry(),
        fresh=fresh,
        symbols=symbols,
        standard=standard,
    )


__all__ = [
    "extension_feature_workspace",
    "external_factor_workspace",
    "real_risk_workspace",
]
