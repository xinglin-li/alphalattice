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
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from typing import Any

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
    """The installed version of each named distribution, by name; one not installed says so.

    Each distinct name is looked up once a process: the lookup scans the installed
    distributions' metadata, and a running process keeps the versions it imported.
    """
    return tuple((name, _installed_version(name)) for name in sorted(set(names)))


@cache
def _installed_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


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
    found: dict[str, int] = {}
    for library in numerical_thread_pools():
        api, threads = library.get("internal_api"), library.get("num_threads")
        if isinstance(api, str) and type(threads) is int:
            found[api] = max(found.get(api, 0), threads)
    return tuple(sorted(found.items()))


_THREAD_POOLS_LOCK = threading.Lock()
_THREAD_POOLS: tuple[tuple[int, ...], Any] | None = None


@cache
def _module_lister() -> Any:
    """kernel32's module enumeration with its declared signature (pointer-sized handles)."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.argtypes = []
    lister = kernel.K32EnumProcessModulesEx
    lister.restype = wintypes.BOOL
    lister.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.DWORD,
    ]
    return kernel


def _loaded_module_handles() -> tuple[int, ...] | None:
    """Every module the process has loaded, in one Windows call; None where it is not offered."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    kernel = _module_lister()
    process = kernel.GetCurrentProcess()
    width = ctypes.sizeof(wintypes.HMODULE)
    count = 1024
    while True:
        handles = (wintypes.HMODULE * count)()
        needed = wintypes.DWORD()
        if not kernel.K32EnumProcessModulesEx(
            process, handles, count * width, ctypes.byref(needed), 3
        ):
            return None
        if needed.value <= count * width:
            return tuple(int(handle or 0) for handle in handles[: needed.value // width])
        count = needed.value // width + 64


def _thread_pool_controller() -> Any:
    """The threadpoolctl view of the loaded numerical libraries, reused while no module loads.

    Building the view asks every loaded module for its path, about 10 ms on Windows, and a
    model scope builds it twice for each fit or prediction. The view is reused while the
    process's loaded modules are exactly those it was built over, so a library loaded since
    is always seen; elsewhere, or when the modules cannot be listed, it is built each time.
    """
    global _THREAD_POOLS
    from threadpoolctl import ThreadpoolController  # type: ignore[import-untyped]

    loaded = _loaded_module_handles()
    with _THREAD_POOLS_LOCK:
        if loaded is not None and _THREAD_POOLS is not None and _THREAD_POOLS[0] == loaded:
            return _THREAD_POOLS[1]
        controller = ThreadpoolController()
        _THREAD_POOLS = None if loaded is None else (loaded, controller)
        return controller


def numerical_thread_pools() -> list[dict[str, Any]]:
    """Each loaded numerical library, its API and live threads, as ``threadpool_info()`` lists."""
    return list(_thread_pool_controller().info())


_LIMITS: list[int] = []
_UNLIMITED: list[Any] = []
_LIMITS_LOCK = threading.Lock()


@contextmanager
def numerical_thread_limit(limits: int) -> Iterator[None]:
    """Limit the loaded numerical libraries' threads in the block, as ``threadpool_limits`` does.

    The limit is the process's, so the scopes of Tasks running at once share it: the libraries
    run at the fewest threads any open scope asks for, and only the last scope to close
    restores the counts from before the first, so a scope closing early leaves no other
    unlimited.
    """
    with _LIMITS_LOCK:
        if not _LIMITS:
            _UNLIMITED.append(_thread_pool_controller().limit(limits=limits))
        elif limits < min(_LIMITS):
            _thread_pool_controller().limit(limits=limits)
        _LIMITS.append(limits)
    try:
        yield
    finally:
        with _LIMITS_LOCK:
            _LIMITS.remove(limits)
            if not _LIMITS:
                _UNLIMITED.pop().restore_original_limits()
            elif limits < min(_LIMITS):
                _thread_pool_controller().limit(limits=min(_LIMITS))


__all__ = [
    "OFFLINE_SWITCH",
    "held_offline",
    "numerical_thread_counts",
    "numerical_thread_limit",
    "numerical_thread_pools",
    "offline",
    "package_versions",
    "recorded_environment",
    "running_held_offline",
]
