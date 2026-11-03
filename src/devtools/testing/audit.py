"""One process audit hook with scoped subscribers and reentrant per-thread dispatch."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

# NumPy's locked import initializes platform metadata. Complete it before either
# a layer contract or a result recorder subscribes to this process's audit hook.
import numpy.testing  # noqa: F401

_INSTALLED: bool = globals().get("_INSTALLED", False)
_LOCK = globals().get("_LOCK") or threading.RLock()
_SUBSCRIBERS: list[tuple[object, Callable[[str, tuple[Any, ...]], None]]] = globals().get(
    "_SUBSCRIBERS", []
)
_DISPATCH = globals().get("_DISPATCH") or threading.local()


def dispatch(event: str, args: tuple[Any, ...]) -> None:
    if not getattr(_DISPATCH, "busy", False):
        _DISPATCH.busy = True
        try:
            for _, callback in tuple(_SUBSCRIBERS):
                callback(event, args)
        finally:
            _DISPATCH.busy = False


def _install_audit() -> None:
    global _INSTALLED
    with _LOCK:
        if not _INSTALLED:
            _INSTALLED = True  # a vetoed installation cannot be retried as a second hook
            sys.addaudithook(dispatch)


@contextmanager
def subscribe(callback: Callable[[str, tuple[Any, ...]], None]) -> Iterator[None]:
    """Own a subscription to the single process hook, verifying actual dispatch."""
    if not callable(callback):
        raise ValueError("testing.layer_contract_unmeasured: subscriber must be callable")
    token, acknowledged = object(), []

    def receive(event: str, args: tuple[Any, ...]) -> None:
        if event == "alphalattice.audit.subscribed" and len(args) == 1 and args[0] is token:
            acknowledged.append(True)
        else:
            callback(event, args)

    with _LOCK:
        _install_audit()
        _SUBSCRIBERS.append((token, receive))
    try:
        sys.audit("alphalattice.audit.subscribed", token)
        if not acknowledged:
            raise RuntimeError(
                "testing.layer_contract_unmeasured: audit dispatch unavailable; "
                "restore the process hook"
            )
        yield
    finally:
        with _LOCK:
            _SUBSCRIBERS[:] = [row for row in _SUBSCRIBERS if row[0] is not token]
