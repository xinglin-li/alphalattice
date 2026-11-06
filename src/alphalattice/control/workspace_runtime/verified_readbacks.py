"""One verification of a sealed review update, shared by its concurrent readers."""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Hashable
from concurrent.futures import Future


class VerifiedReadbacks:
    """Share only in-flight verified update readbacks by their identity.

    Every Analyst submit and bundle reads the reviewed update; verifying it each time cost
    about 165 s apiece with 15 at once (11-04). Concurrent readers of one key share the one
    verification in flight, its result or its error. No result outlives its flight,
    so the next reader verifies afresh. Each reader gets its own copy.
    """

    def __init__(self) -> None:
        """Start with no verifications in flight."""
        self._lock = threading.Lock()
        self._flights: dict[Hashable, Future[dict[str, object]]] = {}

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
