"""The operator and research scopes that can hold outbound reads offline."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar

from alphalattice.kernel.config import current as process_config

OFFLINE_SWITCH = "ALPHALATTICE_NETWORK_DISABLED"
"""The operator's offline switch: ``=1`` keeps the process offline, whatever a workspace allows."""


_HELD_OFFLINE: ContextVar[bool] = ContextVar("alphalattice_held_offline", default=False)


def offline(environment: Mapping[str, str] | None = None) -> bool:
    """Whether the operator's offline switch holds, or a run holds its reads offline (OP5).

    The one reader of `ALPHALATTICE_NETWORK_DISABLED`: the workspace's network control and every
    source guard ask it, so the rule cannot differ between them. Only ``1`` holds it.

    Args:
        environment: The environment to read, the process's own when omitted.

    Returns:
        True when the switch is set to ``1`` or inside ``held_offline``.
    """
    if _HELD_OFFLINE.get():
        return True
    if environment is None:
        return process_config().settings.network_disabled
    return environment.get(OFFLINE_SWITCH) == "1"


def running_held_offline() -> bool:
    """Whether this context runs inside ``held_offline``."""
    return _HELD_OFFLINE.get()


@contextmanager
def held_offline() -> Iterator[None]:
    """Hold this context's reads offline for one run, whatever its workspace allows (OP5).

    A research run declares the network off and reads only sealed inputs; inside this block the
    network control and every source guard read closed, so the declaration holds by
    construction and a person's network, open for an update, never refuses a study. Threads the
    run starts keep their own context, so a guard reached there reads the process's switch.
    """
    token = _HELD_OFFLINE.set(True)
    try:
        yield
    finally:
        _HELD_OFFLINE.reset(token)
