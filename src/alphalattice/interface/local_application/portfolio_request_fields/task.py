"""Task request fields and their public descriptions."""

from typing import Literal

from pydantic import ConfigDict, Field

from alphalattice.kernel.identity import BLAKE3_PATTERN

from .contracts import RequestFieldModel


class RequestFields(RequestFieldModel):
    """Task and supervised-run fields for their operations."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    expected_task_hash: str | None = Field(default=None, pattern=BLAKE3_PATTERN)
    """The Task's version as last read; a Task that moved since refuses the request."""
    incident_key: str | None = Field(default=None, pattern=BLAKE3_PATTERN)
    """The Guanyin incident, by the key TASK_INCIDENTS listed."""
    remedy: Literal["CANCEL", "RECOVER", "REPLAN"] | None = None
    """The remedy to perform, one the incident offers: `CANCEL`, `RECOVER` or `REPLAN`."""
    wait_seconds: float | None = Field(default=None, gt=0, le=20)
    """How long a read may wait for what it reads to move on (a Task's status, a goal's
    narrative, a committee floor addressing its member); at most 20 seconds."""
    status_section: Literal["storage"] | None = None
    """The optional Task status section to compute on an explicit read."""
