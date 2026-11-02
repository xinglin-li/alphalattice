"""The two methodology cases that install a catalog revision, in their own process.

They need the extension and external workspaces, which nothing else does, and
the shipped one as a control. Running them beside the other methodology cases
made one group build three real workspaces in sequence and set the routed
lane's wall clock; here they build what they need while the shipped-catalog
cases run in parallel. The builders and fixtures are the methodology package's
own, imported by name.
"""

from __future__ import annotations

from tests.researcher_methodology_surface.session_fixtures import (
    extension_feature_workspace,
    external_factor_workspace,
    offline_execution_environment,
    real_risk_workspace,
)

__all__ = [
    "extension_feature_workspace",
    "external_factor_workspace",
    "offline_execution_environment",
    "real_risk_workspace",
]
