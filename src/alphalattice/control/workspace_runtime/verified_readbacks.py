"""One verification of a sealed review update, shared by its readers."""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Hashable
from concurrent.futures import Future


class VerifiedReadbacks:
    """Share verified update readbacks by their identity.

    Every Analyst submit and bundle reads the reviewed update; verifying it each time cost
    about 165 s apiece with 15 at once (11-04). Concurrent readers of one key share the one
    verification in flight, its result or its error. Bundle preparation can hold one sealed
    book's projection across requests. Other reads verify afresh after their flight ends.
    Each reader gets its own copy.
    """

    def __init__(self) -> None:
        """Start with no verifications in flight."""
        self._lock = threading.Lock()
        self._flights: dict[Hashable, Future[dict[str, object]]] = {}
        self._book: tuple[Hashable, str, dict[str, object]] | None = None

    def book(
        self,
        key: Hashable,
        publication_hash: str | None,
        verify: Callable[[], dict[str, object]],
    ) -> dict[str, object]:
        """Reuse the last verified final publication for the exact completed Task evidence."""
        with self._lock:
            held = self._book
        if held is not None and held[0] == key and publication_hash in (None, held[1]):
            return copy.deepcopy(held[2])
        if publication_hash is not None:
            return verify()

        def keep() -> dict[str, object]:
            with self._lock:
                current = self._book
            if current is not None and current[0] == key:
                return current[2]
            value = verify()
            publication = value.get("publication")
            if isinstance(publication, dict) and isinstance(
                exact := publication.get("content_hash"), str
            ):
                with self._lock:
                    self._book = (key, exact, value)
            return value

        return self.read(("bundle-book", key), keep)

    def read(self, key: Hashable, verify: Callable[[], dict[str, object]]) -> dict[str, object]:
        """The readback for `key`, shared only while `verify` runs.

        Args:
            key: The identity the verification proves.
            verify: The full verification, run by the first reader alone.

        Returns:
            This reader's own copy of the verified readback.
        """
        with self._lock:
            flight = self._flights.get(key)
            leading = flight is None
            if flight is None:
                flight = self._flights[key] = Future()
        if not leading:
            return copy.deepcopy(flight.result())
        try:
            value = verify()
        except BaseException as error:
            with self._lock:
                flight.set_exception(error)
                del self._flights[key]
            raise
        with self._lock:
            flight.set_result(value)
            del self._flights[key]
        return copy.deepcopy(value)
