"""Session-scoped real workspaces, shared by the methodology packages' conftests.

Fixture functions, imported by name into each package's `conftest.py`, so that
one builder produces the same workspace wherever a package needs it while each
package still pays only for the workspaces its own cases resolve against.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alphalattice.kernel.data_layer import history
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


@pytest.fixture(scope="module")
def outcome_catalog_root(tmp_path_factory: pytest.TempPathFactory):
    """Isolate causal outcome recording from the workspace's own catalog."""
    root = tmp_path_factory.getbasetemp() / "methodology-outcome-catalog"
    with history.own_root(root):
        history.open_history("methodology-outcomes", "Methodology outcomes", root, root=root)
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
