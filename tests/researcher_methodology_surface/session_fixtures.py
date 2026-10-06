"""Session-scoped real workspaces, shared by the methodology packages' conftests.

Fixture functions, imported by name into each package's `conftest.py`, so that
one builder produces the same workspace wherever a package needs it while each
package still pays only for the workspaces its own cases resolve against. The
split exists for the gate's wall clock: the extension and external catalog
workspaces were built in the same process as the twenty-seven cases that need
only the shipped one, and that single group set the routed lane's makespan.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from tests.researcher_methodology_surface.real_workspace import (
    RealRiskWorkspace,
    build_real_risk_workspace,
    development_feature_catalog,
)


@pytest.fixture(scope="session", autouse=True)
def offline_execution_environment() -> Iterator[None]:
    """The operator's offline switch, as the product's research entry points set it.

    The authored-execution policy reads no switch: the workflow holds every run offline
    (V116), so a case passes whatever the operator's shell carries. The switch stays on here
    because the data readers these cases reach outside a run keep a process offline by it.
    """

    previous = os.environ.get("ALPHALATTICE_NETWORK_DISABLED")
    os.environ["ALPHALATTICE_NETWORK_DISABLED"] = "1"
    try:
        yield
    finally:
        if previous is None:
            del os.environ["ALPHALATTICE_NETWORK_DISABLED"]
        else:
            os.environ["ALPHALATTICE_NETWORK_DISABLED"] = previous


@pytest.fixture(scope="session")
def real_risk_workspace(tmp_path_factory: pytest.TempPathFactory) -> RealRiskWorkspace:
    """The product's own writers, end to end, once per source tree.

    Session scope shares the copy inside one process; the golden cache under
    ``tmp/golden-workspaces`` shares the eighty-second build across processes,
    so a gate that runs several groups needing this workspace builds it once
    and each group pays a copy of a few seconds.
    """

    return build_real_risk_workspace(tmp_path_factory.mktemp("real-risk-workspace"))


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
    """

    from tests.researcher_methodology_surface.external_factor_method import (
        external_feature_catalog,
        external_kernel_registry,
    )

    # Fresh for the same reason as the extension workspace: the outside method
    # reaching a published Panel is the claim, so the Panel is written, not copied.
    return build_real_risk_workspace(
        tmp_path_factory.mktemp("external-factor-workspace"),
        feature_catalog=external_feature_catalog(),
        feature_kernels=external_kernel_registry(),
        fresh=True,
    )


__all__ = [
    "extension_feature_workspace",
    "external_factor_workspace",
    "offline_execution_environment",
    "real_risk_workspace",
]
