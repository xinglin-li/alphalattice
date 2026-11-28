"""Route the installed capabilities' operations to the Host's plugin registry.

`PLUGIN_LIST` answers the registry's listing; `PLUGIN_INVOKE` hands the registry the Host's
catalog read, its clock's date, whether the caller is attended and its one staging owner, and
answers the receipt, or the registry's or staging owner's refusal by name.
"""

from __future__ import annotations

import json
from threading import Lock
from typing import TYPE_CHECKING

from alphalattice.control.product_host.composition.plain_refusals import refused
from alphalattice.control.product_host.plugins.registry import PluginRefused

if TYPE_CHECKING:
    from collections.abc import Collection

    from alphalattice.control.product_host.composition.operation_route_families import (
        WorkspaceOperationCall,
    )
    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        PortfolioResearchOperations,
    )
    from alphalattice.control.task_control.plugin_staging import PluginStaging
    from alphalattice.protocols.plugin_contract import CatalogRead

_staging_lock = Lock()


def route(call: WorkspaceOperationCall[PortfolioResearchOperations]) -> dict[str, object]:
    """List the capabilities, or run one and answer its receipt."""
    from alphalattice.control.task_control.plugin_staging import (  # lazy: a run's refusal
        StagingRefused,
    )

    operations, request = call.operations, call.request
    registry = operations.plugins
    assert registry is not None  # the Host composes its registry before it serves
    if request.operation == "PLUGIN_LIST":
        return {
            "status": "AVAILABLE",
            "capabilities": [
                {
                    "capability": entry.capability,
                    "plugin": entry.plugin,
                    "effect_mode": entry.effect_mode.value,
                    "workspace": entry.workspace,
                    "state": entry.state.value,
                    "input_schema": json.loads(json.dumps(entry.input_schema, default=dict)),
                }
                for entry in registry.listing()
            ],
        }
    name = request.capability or ""
    try:
        receipt = registry.invoke(
            name,
            request.params or {},
            catalog=_catalog,
            as_of=operations.dispatcher.clock().date(),
            attended=call.caller != "SERVICE_AUTOMATION",
            staging=lambda: _staging(operations),
        )
    except PluginRefused as error:
        decision = error.decision
        return {
            **refused(error.code),
            "capability": name,
            **({"decision_id": str(decision.id)} if decision is not None else {}),
        }
    except StagingRefused as error:
        return {**refused(error.code), "capability": name, "next_step": error.next_step}
    return {"status": "COMPLETED", "capability": name, "receipt": receipt.model_dump(mode="json")}


def _catalog(tables: Collection[str], at: str | None = None) -> CatalogRead:
    """The Host catalog's read of `tables`, at the snapshot `at` names or the latest."""
    from alphalattice.foundation.data_layer import history_runtime  # lazy: a run's read

    store = history_runtime.store()
    if store is None:
        raise PluginRefused("plugin_registry.catalog_unavailable", ",".join(sorted(tables)))
    return store.catalog_read(tables, at)


def _staging(operations: PortfolioResearchOperations) -> PluginStaging:
    """The Host's one staging owner, made at its first staging run."""
    with _staging_lock:
        if operations.plugin_staging is None:
            from alphalattice.control.task_control.plugin_staging import (  # lazy: a staging run
                PluginStaging,
            )
            from alphalattice.control.workspace_runtime.artifacts import (  # lazy: a staging run
                ArtifactResolver,
            )
            from alphalattice.foundation.data_layer.root import data_root  # lazy: its read root

            session = operations.workspace_session
            operations.plugin_staging = PluginStaging(
                session.workspace,
                data_root(),
                ArtifactResolver(session.workspace / "runtime" / "artifacts"),
                session.mutation_gate,
                _catalog,
            )
        return operations.plugin_staging
