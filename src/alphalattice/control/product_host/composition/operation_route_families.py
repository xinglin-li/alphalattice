"""Wire the Host to heavier route families without loading them at startup."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphalattice.protocols.actor_execution.answers import AgentRun
    from alphalattice.protocols.actor_execution.bundles import AgentBundleRecord

    from .portfolio_research_operations import WorkspaceOperationCall


class HostRouteFamilies:
    """Resolve the requested route family at its first use."""

    def workspace(self, call: WorkspaceOperationCall) -> dict[str, object] | None:
        """Dispatch the workspace families, leaving unmatched requests to the Host."""
        match call.request.operation.partition("_")[0]:
            case "FEATURE":
                # The Feature readers and their request dependencies belong to this route.
                from . import (  # lazy: on use
                    feature_operations,
                )

                return feature_operations.execute(call)
            case "RESEARCH" | "MODEL" | "WORKSPACE":
                from . import (  # lazy: defer the research route family until it is requested.
                    research_operations,
                )

                return research_operations.execute(call)
            case "DATA" | "STORAGE":
                from . import (  # lazy: defer the data route family until it is requested.
                    data_operations,
                )

                return data_operations.execute(call)
            case "STATUS" | "TASK" | "TASKS" | "RECOVER" | "CANCEL":
                from . import (  # lazy: defer the task route family until it is requested.
                    task_operations,
                )

                return task_operations.execute(call)
        return None

    def portfolio(self, call: WorkspaceOperationCall) -> dict[str, object]:
        """Load and dispatch a Portfolio operation."""
        from . import portfolio_operations  # lazy: Portfolio operation

        return portfolio_operations.execute(call.operations, call.request, caller=call.caller)

    def review(
        self,
        call: WorkspaceOperationCall,
        *,
        read_files: tuple[str, ...] | None,
        agent_run: AgentRun | None,
        accepted_bundle: AgentBundleRecord | None,
    ) -> dict[str, object]:
        """Load and dispatch a review operation."""
        from . import review_operations  # lazy: review operation

        return review_operations.execute(
            call.operations,
            call.request,
            caller=call.caller,
            read_files=read_files,
            agent_run=agent_run,
            accepted_bundle=accepted_bundle,
        )
