"""Workspace request fields and their public descriptions."""

from typing import Any, Literal
from uuid import UUID

from pydantic import ConfigDict, Field

from alphalattice.protocols.actor_execution.bundles import AgentRole

from .contracts import PortfolioResearchOperation, RequestFieldModel


class RequestFields(RequestFieldModel):
    """Workspace and activity fields for the operations that admit them."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    schema_operation: PortfolioResearchOperation | None = None
    """Only this operation's request schema and fields."""
    agent_role: AgentRole | None = None
    """Whose bundle to prepare: `ANALYST`, `CRO`, `ALPHA`, `DATA`, `FACTOR`,
    `PORTFOLIO` or `RISK`. The five research specialists name a retained Task."""
    bundle_directory: str | None = Field(default=None, min_length=1, max_length=1024)
    """Where the agent's bundle is written, and its answer read back."""
    agent_answer: dict[str, Any] | None = None
    """An answer to the named bundle as written. Analyst and CRO retain their
    domain schemas; the other specialists supply bounded `text` and exact
    `references` listed in the bundle. The Host checks shape and bindings,
    never scientific content, and names each problem."""
    network_enabled: bool | None = Field(default=None, strict=True)
    """true lets this workspace reach the network; false keeps it offline."""
    usage_reading_enabled: bool | None = Field(default=None, strict=True)
    """true lets the Host read the bound agent Sessions' own files for usage; false reads
    nothing."""
    analyst_reads_mode: Literal["SUBAGENTS", "IN_SESSION"] | None = None
    """How the lead reads a review's Analyst bundles: each by its own subagent, or itself."""
    analyst_reads_bundles: int | None = Field(default=None, ge=1, le=10_000, strict=True)
    """The Analyst bundles a review hands out, or the count the person's answer covers."""
    upgrade_set_hash: str | None = None
    """The installed identity set an upgrade overview showed, when acknowledging it."""
    after: str | None = Field(default=None, min_length=1, max_length=256)
    """Where the activity feed reads on: the cursor its previous page answered."""
    limit: int | None = Field(default=None, ge=1, le=200, strict=True)
    """The most a page reads: 1 to 200 rows of the activity feed, 1 to 50 recent groups."""
    watch: tuple[UUID, ...] | None = Field(default=None, max_length=16)
    """Tasks, by id, whose current projection the activity page joins; a short id the compact
    display gave reads as its whole id."""
    event: dict[str, Any] | None = None
    """The event this client declares about its own work (`ExternalActivityEventDocument`)."""
    capability: str | None = Field(default=None, min_length=1, max_length=200)
    """The installed capability to run, as `plugin list` names it."""
    params: dict[str, Any] | None = None
    """The capability's parameters as a JSON object, inline or `@` and a file; checked against its
    input schema before it runs."""
    wake_thread: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{8,128}$")
    """The Codex thread to wake (`CODEX_THREAD_ID`)."""
    wake_read: str | None = Field(
        default=None, min_length=1, max_length=4096, pattern=r"^[^\x00-\x1f\x7f]+$"
    )
    """The exact read or rerun command, with its next free output path."""
    wake_codex_path: str | None = Field(
        default=None, max_length=4096, pattern=r"^[^\x00-\x1f\x7f]+$"
    )
    """The client's Codex executable; the Host validates it or uses its PATH."""
    cpu_budget: str | None = Field(default=None, min_length=1, max_length=16)
    """`auto`, or a whole number of cores a book's preparation may use."""
    tasks_waiting: str | None = Field(default=None, min_length=1, max_length=16)
    """`auto`, or how many Tasks may wait behind the running one."""
    tasks_running: str | None = Field(default=None, min_length=1, max_length=16)
    """`auto`, or how many Tasks that may run beside others run at once, within the budget."""
