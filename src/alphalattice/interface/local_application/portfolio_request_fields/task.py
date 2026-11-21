"""Task request fields and their public descriptions."""

from typing import Literal

from pydantic import ConfigDict, Field

from .contracts import RequestFieldModel


class RequestFields(RequestFieldModel):
    """Task and supervised-run fields for their operations."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    expected_task_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The Task's version as last read; a Task that moved since refuses the request."""
    incident_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The Guanyin incident, by the key TASK_INCIDENTS listed."""
    remedy: Literal["CANCEL", "RECOVER", "REPLAN"] | None = None
    """The remedy to perform, one the incident offers: `CANCEL`, `RECOVER` or `REPLAN`."""
    wait_seconds: float | None = Field(default=None, gt=0, le=20)
    """How long a Task's status may wait for the Task to move on; at most 20 seconds."""
