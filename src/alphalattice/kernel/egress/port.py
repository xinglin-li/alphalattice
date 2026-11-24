"""Outbound access boundary."""

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Protocol

from .http import HttpResponse


class EgressPort(Protocol):
    """Timed outbound HTTP reads under the selected source's policy."""

    def get(self, url: str) -> HttpResponse:
        """Read the complete resource at the given URL."""
        ...

    def stream(
        self, url: str, *, headers: Mapping[str, str] | None = None
    ) -> AbstractContextManager[HttpResponse]:
        """Open the response as a bounded transport stream."""
        ...

    def close(self) -> None:
        """Release the outbound session."""
        ...
