"""The environment a result was computed in: provenance beside it, never its identity.

The interpreter, the platform and the installed library versions are read here (LAWS.md ID6).
A result records them beside itself and no identity folds them, so adding, removing or
upgrading a dependency moves no identity; whether it moves a number is answered by U0 on both
corpora, and the release locks the environment. A cache whose content they decide (an
embedding index) keeps them as its rebuild key.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar

OFFLINE_SWITCH = "ALPHALATTICE_NETWORK_DISABLED"
"""The operator's offline switch: ``=1`` keeps the process offline, whatever a workspace allows."""

_HELD_OFFLINE: ContextVar[bool] = ContextVar("alphalattice_held_offline", default=False)


def offline(environment: Mapping[str, str] | None = None) -> bool:
    """Whether the operator's offline switch holds, or a run holds its reads offline (OP5, V250).

    The one reader of `ALPHALATTICE_NETWORK_DISABLED`: the workspace's network control and every
    source guard ask it, so the rule cannot differ between them. Only ``1`` holds it.

    Args:
        environment: The environment to read, the process's own when omitted.

    Returns:
        True when the switch is set to ``1`` or inside ``held_offline``.
    """
    if _HELD_OFFLINE.get():
        return True
    return (os.environ if environment is None else environment).get(OFFLINE_SWITCH) == "1"


def running_held_offline() -> bool:
    """Whether this context runs inside ``held_offline``."""
    return _HELD_OFFLINE.get()


@contextmanager
def held_offline() -> Iterator[None]:
    """Hold this context's reads offline for one run, whatever its workspace allows (OP5, V116).

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


def package_versions(names: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """The installed version of each named distribution, by name; one not installed says so."""
    found = []
    for name in sorted(set(names)):
        try:
            found.append((name, importlib.metadata.version(name)))
        except importlib.metadata.PackageNotFoundError:
            found.append((name, "absent"))
    return tuple(found)


def recorded_environment(packages: Iterable[str] = ()) -> dict[str, object]:
    """The interpreter, the platform and the named packages, as a result records them."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "byte_order": sys.byteorder,
        "packages": dict(package_versions(packages)),
    }


def numerical_thread_counts() -> tuple[tuple[str, int], ...]:
    """Each loaded numerical library's thread count, by its API (a BLAS, OpenMP), as recorded.

    A sum a BLAS splits by thread count can move a number's last bits, so a result keeps the
    counts it ran at beside it (LAWS PA3, V68); the most any library of one API runs at.

    Returns:
        ``(api, threads)`` pairs, in API order; empty when no such library is loaded.
    """
    from threadpoolctl import threadpool_info  # type: ignore[import-untyped]

    found: dict[str, int] = {}
    for library in threadpool_info():
        api, threads = library.get("internal_api"), library.get("num_threads")
        if isinstance(api, str) and type(threads) is int:
            found[api] = max(found.get(api, 0), threads)
    return tuple(sorted(found.items()))


__all__ = [
    "OFFLINE_SWITCH",
    "held_offline",
    "numerical_thread_counts",
    "offline",
    "package_versions",
    "recorded_environment",
    "running_held_offline",
]
