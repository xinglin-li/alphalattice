"""Small atomic content-addressed primitive shared by legacy and public stores.

Each file is named by its content's digest and written in its DA9 type: a contract or a
document as JSON, numerical lanes as Parquet, a rendered page as HTML. A store written
before V210 kept lanes, documents and pages as packed ``.bin`` bytes under the same
digest; those are read, never written, until the formats registry names their upgrader
(V267).
"""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import Context, ContextVar
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from io import BytesIO
from pathlib import Path
from queue import Queue
from threading import BoundedSemaphore, Event, Lock, Thread
from time import sleep
from typing import Any, cast
from uuid import UUID
from weakref import WeakSet

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from alphalattice.kernel.shared_kernel.spans import span

_WINDOWS_REPLACE_ATTEMPTS = 6
_WINDOWS_WAIT_HANDLE_LIMIT = 64
_VERIFIED_MODEL_CACHE_LIMIT = 128
_VERIFIED_MODEL_KEY = tuple[object, ...]
_VERIFIED_MODEL_SCOPE: ContextVar[dict[_VERIFIED_MODEL_KEY, BaseModel] | None] = ContextVar(
    "content_store_verified_models", default=None
)
_REUSE_VERIFIED_MODELS: ContextVar[bool] = ContextVar(
    "content_store_reuse_verified_models", default=False
)
_VERIFIED_MODEL_CACHE: OrderedDict[_VERIFIED_MODEL_KEY, BaseModel] = OrderedDict()
_VERIFIED_MODEL_CACHE_LOCK = Lock()
_VERIFIED_ARRAY_CACHE_LIMIT_BYTES = 512 * 1024 * 1024
_VERIFIED_ARRAY_PROCESS_CACHE_LIMIT_BYTES = 512 * 1024 * 1024
_VERIFIED_ARRAY_KEY = tuple[object, ...]
_VERIFIED_ARRAY_SCOPE: ContextVar[OrderedDict[_VERIFIED_ARRAY_KEY, tuple[object, int]] | None] = (
    ContextVar("content_store_verified_arrays", default=None)
)
_REUSE_VERIFIED_ARRAYS: ContextVar[bool] = ContextVar(
    "content_store_reuse_verified_arrays", default=False
)
_VERIFIED_SOURCE_BUILD: ContextVar[bool] = ContextVar(
    "content_store_verified_source_build", default=False
)
_VERIFIED_ARRAY_CACHE: OrderedDict[_VERIFIED_ARRAY_KEY, tuple[object, int]] = OrderedDict()
_VERIFIED_ARRAY_CACHE_BYTES = 0
_VERIFIED_ARRAY_CACHE_LOCK = Lock()
_VERIFIED_SOURCE_LEASE_LIMIT = 512
_VERIFIED_SOURCE_LEASES: OrderedDict[_VERIFIED_ARRAY_KEY, tuple[_WindowsReadLease, ...]] = (
    OrderedDict()
)
_VERIFIED_SOURCE_BUILDS: dict[_VERIFIED_ARRAY_KEY, Future[None]] = {}
_VERIFIED_SOURCE_CHECK_WORKERS = 2
_VERIFIED_SOURCE_CHECK_POOL: ThreadPoolExecutor | None = None
_VERIFIED_SOURCE_CHECK_POOL_LOCK = Lock()
_VERIFIED_SOURCE_CHECK_SLOTS = BoundedSemaphore(_VERIFIED_SOURCE_CHECK_WORKERS)
_VERIFIED_SOURCE_CHECK_RUNNING: ContextVar[bool] = ContextVar(
    "content_store_verified_source_check_running", default=False
)
_VERIFIED_SOURCE_ISSUER: _WindowsReadLeaseIssuer | None = None
_VERIFIED_SOURCE_ISSUER_LOCK = Lock()
_VERIFIED_ARRAY_PROCESS_CACHE_HITS = 0
_VERIFIED_ARRAY_PROCESS_CACHE_MISSES = 0
_VERIFIED_ARRAY_PROCESS_CACHE_EVICTIONS = 0
_VERIFIED_PROOF_CACHE_LIMIT = 128
_VERIFIED_PROOF_SCOPE: ContextVar[set[tuple[object, ...]] | None] = ContextVar(
    "content_store_verified_proofs", default=None
)
_VERIFIED_PROOF_CACHE: OrderedDict[tuple[object, ...], None] = OrderedDict()
_VERIFIED_PROOF_CACHE_LOCK = Lock()


class _WindowsReadLease:
    """A file-level read-handle oplock, never a timestamp or notification guess.

    Modern Read-Handle oplocks break before incompatible writes, including writes through
    another hard link and writable mappings. Closing a broken handle acknowledges rename,
    replacement and ancestor-directory rename, so retaining proof never obstructs those acts.
    The event is one-shot: a completed or failed request is
    discarded, never reset. A current path identity check additionally detects deletion,
    renames, replacement and changed path resolution. Unavailable leases grant no reuse.
    These leases are for immutable content-addressed sources only; a process-lived issuer
    also closes signalled handles when no reader is running.
    """

    def __init__(self, path: Path) -> None:
        from ctypes import wintypes

        class Overlapped(ctypes.Structure):
            _fields_ = [
                ("Internal", ctypes.c_size_t),
                ("InternalHigh", ctypes.c_size_t),
                ("Offset", wintypes.DWORD),
                ("OffsetHigh", wintypes.DWORD),
                ("hEvent", wintypes.HANDLE),
            ]

        class FileInformation(ctypes.Structure):
            _fields_ = [
                ("attributes", wintypes.DWORD),
                ("creation", wintypes.FILETIME),
                ("access", wintypes.FILETIME),
                ("write", wintypes.FILETIME),
                ("volume", wintypes.DWORD),
                ("size_high", wintypes.DWORD),
                ("size_low", wintypes.DWORD),
                ("links", wintypes.DWORD),
                ("index_high", wintypes.DWORD),
                ("index_low", wintypes.DWORD),
            ]

        class FileIdInformation(ctypes.Structure):
            _fields_ = [("volume", ctypes.c_ulonglong), ("file_id", ctypes.c_ubyte * 16)]

        class OplockInput(ctypes.Structure):
            _fields_ = [
                ("version", wintypes.WORD),
                ("length", wintypes.WORD),
                ("level", wintypes.DWORD),
                ("flags", wintypes.DWORD),
            ]

        class OplockOutput(ctypes.Structure):
            _fields_ = [
                ("version", wintypes.WORD),
                ("length", wintypes.WORD),
                ("original_level", wintypes.DWORD),
                ("new_level", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("access", wintypes.DWORD),
                ("share", wintypes.WORD),
            ]

        self.path = path
        self.lock = Lock()
        self.file_id_type = FileIdInformation
        self.identity = self._path_identity()
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        declarations = {
            "CreateFileW": (
                [
                    wintypes.LPCWSTR,
                    wintypes.DWORD,
                    wintypes.DWORD,
                    ctypes.c_void_p,
                    wintypes.DWORD,
                    wintypes.DWORD,
                    wintypes.HANDLE,
                ],
                wintypes.HANDLE,
            ),
            "CreateEventW": (
                [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR],
                wintypes.HANDLE,
            ),
            "DeviceIoControl": (
                [
                    wintypes.HANDLE,
                    wintypes.DWORD,
                    ctypes.c_void_p,
                    wintypes.DWORD,
                    ctypes.c_void_p,
                    wintypes.DWORD,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ],
                wintypes.BOOL,
            ),
            "GetFileInformationByHandle": ([wintypes.HANDLE, ctypes.c_void_p], wintypes.BOOL),
            "GetFileInformationByHandleEx": (
                [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD],
                wintypes.BOOL,
            ),
            "WaitForSingleObject": ([wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
            "WaitForMultipleObjects": (
                [wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE), wintypes.BOOL, wintypes.DWORD],
                wintypes.DWORD,
            ),
            "CancelIoEx": ([wintypes.HANDLE, ctypes.c_void_p], wintypes.BOOL),
            "GetOverlappedResult": (
                [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, wintypes.BOOL],
                wintypes.BOOL,
            ),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (arguments, result) in declarations.items():
            function = getattr(self.kernel, name)
            function.argtypes = arguments
            function.restype = result
        self.handle = None
        self.event = None
        self.pending = False
        self.overlapped = Overlapped()
        self.oplock_input = OplockInput(1, ctypes.sizeof(OplockInput), 3, 1)
        self.oplock_output = OplockOutput()
        try:
            self.handle = self.kernel.CreateFileW(
                os.fspath(path), 0x80000000, 7, None, 3, 0x40000000, None
            )
            if self.handle in (None, ctypes.c_void_p(-1).value):
                self.handle = None
                raise OSError("OS read lease unavailable")
            observed = FileInformation()
            if not self.kernel.GetFileInformationByHandle(self.handle, ctypes.byref(observed)):
                raise OSError("OS read lease unavailable")
            file_id = FileIdInformation()
            if not self.kernel.GetFileInformationByHandleEx(
                self.handle, 18, ctypes.byref(file_id), ctypes.sizeof(file_id)
            ):
                raise OSError("OS read lease unavailable")
            if (
                file_id.volume,
                int.from_bytes(bytes(file_id.file_id), "little"),
                (observed.size_high << 32) | observed.size_low,
            ) != self.identity[1:4]:
                raise OSError("OS read lease path changed")
            self.event = self.kernel.CreateEventW(None, True, False, None)
            if not self.event:
                raise OSError("OS read lease unavailable")
            self.overlapped.hEvent = self.event
            returned = wintypes.DWORD()
            completed = self.kernel.DeviceIoControl(
                self.handle,
                0x00090240,
                ctypes.byref(self.oplock_input),
                ctypes.sizeof(self.oplock_input),
                ctypes.byref(self.oplock_output),
                ctypes.sizeof(self.oplock_output),
                ctypes.byref(returned),
                ctypes.byref(self.overlapped),
            )
            self.pending = not completed and ctypes.get_last_error() == 997
            if not self.pending:
                raise OSError("OS read lease unavailable")
        except BaseException:
            self.close()
            raise

    def _path_identity(self) -> tuple[object, ...]:
        info = self.path.stat()
        return (
            os.fspath(self.path.resolve(strict=True)),
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    def unchanged(self) -> bool:
        """Fail closed for a signalled event, an API failure or any path identity change."""
        try:
            with self.lock:
                if not (
                    self.pending
                    and self.kernel.WaitForSingleObject(self.event, 0) == 258
                    and self._path_identity() == self.identity
                ):
                    return False
            # A competing open can wait for the RH break to be acknowledged. Never keep
            # the lease lock across that open: the break thread needs it to close us.
            if not self._current_read_access():
                return False
            with self.lock:
                return bool(
                    self.pending
                    and self.kernel.WaitForSingleObject(self.event, 0) == 258
                    and self._path_identity() == self.identity
                )
        except Exception:
            return False

    def _current_read_access(self) -> bool:
        """Prove current read permission and exact file ID without reading its bytes."""
        handle = self.kernel.CreateFileW(
            os.fspath(self.path), 0x80000000, 7, None, 3, 0x40000000, None
        )
        if handle in (None, ctypes.c_void_p(-1).value):
            return False
        try:
            file_id = self.file_id_type()
            return bool(
                self.kernel.GetFileInformationByHandleEx(
                    handle, 18, ctypes.byref(file_id), ctypes.sizeof(file_id)
                )
                and (file_id.volume, int.from_bytes(bytes(file_id.file_id), "little"))
                == self.identity[1:3]
            )
        finally:
            self.kernel.CloseHandle(handle)

    def release_if_broken(self) -> None:
        """Acknowledge an OS break by closing; API failure also forfeits cached proof."""
        try:
            with self.lock:
                broken = self.pending and self.kernel.WaitForSingleObject(self.event, 0) != 258
        except Exception:
            broken = True
        if broken:
            self.close()

    def close(self) -> None:
        """Cancel and drain the IRP before freeing its OVERLAPPED/event storage."""
        with self.lock:
            if self.handle is not None:
                if self.pending:
                    returned = ctypes.c_ulong()
                    self.kernel.CancelIoEx(self.handle, ctypes.byref(self.overlapped))
                    self.kernel.GetOverlappedResult(
                        self.handle, ctypes.byref(self.overlapped), ctypes.byref(returned), True
                    )
                self.kernel.CloseHandle(self.handle)
                self.handle = None
                self.pending = False
            if self.event:
                self.kernel.CloseHandle(self.event)
                self.event = None


class _WindowsReadLeaseIssuer:
    """Issue pending oplock IRPs on one process-lived thread, never a request thread.

    Windows cancels event-based pending I/O when its issuing thread exits. Only lease
    acquisition runs here; builders, cache decisions and current-access checks remain
    on the caller. Serialized acquisition allows at most one queued job.
    """

    def __init__(self) -> None:
        self.jobs: Queue[tuple[Path, Future[_WindowsReadLease]] | None] = Queue(maxsize=1)
        self.stopped = False
        self.issued: WeakSet[_WindowsReadLease] = WeakSet()
        self.issued_lock = Lock()
        self.breaks_stopped = Event()
        self.thread = Thread(target=self._run, name="content-store-read-leases", daemon=True)
        self.break_thread = Thread(
            target=self._release_breaks, name="content-store-lease-breaks", daemon=True
        )
        self.thread.start()
        self.break_thread.start()

    def _release_breaks(self) -> None:
        # Inspect only OS events. Never a stat/timestamp shortcut, and never a second proof
        # cache. A separate thread can acknowledge a rename while acquisition is waiting
        # in Windows; using the issuing thread for both could deadlock that rename.
        while not self.breaks_stopped.wait(0.01):
            with self.issued_lock:
                issued = tuple(self.issued)
            for offset in range(0, len(issued), _WINDOWS_WAIT_HANDLE_LIMIT):
                held = tuple(
                    lease
                    for lease in issued[offset : offset + _WINDOWS_WAIT_HANDLE_LIMIT]
                    if lease.pending and lease.event
                )
                if not held:
                    continue
                events = (ctypes.c_void_p * len(held))(*(lease.event for lease in held))
                try:
                    state = held[0].kernel.WaitForMultipleObjects(len(held), events, False, 0)
                except Exception:
                    state = None
                # Concurrent closes can invalidate a batch handle. A failed/signalled batch
                # falls back to each lease's locked event check; failure never grants reuse.
                if state != 258:
                    for lease in held:
                        lease.release_if_broken()

    def _run(self) -> None:
        while (job := self.jobs.get()) is not None:
            path, result = job
            try:
                lease = _WindowsReadLease(path)
                with self.issued_lock:
                    self.issued.add(lease)
                # Register before the current-access open: a rename can break this new
                # RH lease while that open waits for the break thread to acknowledge it.
                if not lease.unchanged():
                    lease.close()
                    raise OSError("OS read lease unavailable")
                result.set_result(lease)
            except BaseException as error:
                result.set_exception(error)

    def acquire(self, path: Path) -> _WindowsReadLease:
        if self.stopped or not self.thread.is_alive() or not self.break_thread.is_alive():
            raise OSError("OS read lease issuer unavailable")
        result: Future[_WindowsReadLease] = Future()
        self.jobs.put_nowait((path, result))
        return result.result()

    def close(self) -> None:
        self.stopped = True
        if self.thread.is_alive():
            self.jobs.put_nowait(None)
            self.thread.join()
        self.breaks_stopped.set()
        self.break_thread.join()


def _acquire_source_lease(path: Path) -> _WindowsReadLease:
    global _VERIFIED_SOURCE_ISSUER
    with _VERIFIED_SOURCE_ISSUER_LOCK:
        if _VERIFIED_SOURCE_ISSUER is None:
            _VERIFIED_SOURCE_ISSUER = _WindowsReadLeaseIssuer()
        return _VERIFIED_SOURCE_ISSUER.acquire(path)


def _close_source_leases(key: _VERIFIED_ARRAY_KEY) -> None:
    for lease in _VERIFIED_SOURCE_LEASES.pop(key, ()):
        lease.close()


def _close_all_source_leases() -> None:
    global _VERIFIED_SOURCE_ISSUER
    with _VERIFIED_SOURCE_ISSUER_LOCK:
        with _VERIFIED_ARRAY_CACHE_LOCK:
            for key in tuple(_VERIFIED_SOURCE_LEASES):
                _close_source_leases(key)
        if _VERIFIED_SOURCE_ISSUER is not None:
            _VERIFIED_SOURCE_ISSUER.close()
            _VERIFIED_SOURCE_ISSUER = None


atexit.register(_close_all_source_leases)


def verified_source_value[T](
    identity: tuple[object, ...],
    sources: tuple[Path, ...],
    builder: Callable[[], T],
    *,
    nbytes: int | Callable[[T], int],
) -> T:
    """Reuse a full immutable verification result while file read leases remain valid.

    The owner supplies exact content commitments and performs all ordinary byte, hash and
    semantic checks in ``builder``. Leases are acquired before that verification and checked
    afterwards and on every hit. This shares the existing 512-MiB result LRU; at most 512
    source handles are retained, closed on invalidation, eviction or process exit. Only an
    explicitly opted-in read scope uses leases. Unsupported filesystems/platforms and any
    unavailable/failed signal perform the original full verification on every access.
    A builder owns all of its declared sources. Nested helpers still verify in full but
    do not retain a second source result or process copy of its intermediate arrays.
    A single process-lived issuer keeps pending read leases alive after request threads exit;
    process cleanup closes and drains retained leases before stopping that issuer.
    Concurrent readers of one exact key share a producer, then check its retained leases
    normally. Waiting never holds the cache lock and never grants proof by itself.
    """
    global _VERIFIED_ARRAY_PROCESS_CACHE_HITS, _VERIFIED_ARRAY_PROCESS_CACHE_MISSES
    if _VERIFIED_SOURCE_BUILD.get():
        return builder()
    paths = tuple(path.absolute() for path in sources)
    if (
        os.name != "nt"
        or not paths
        or len(paths) > _VERIFIED_SOURCE_LEASE_LIMIT
        or _VERIFIED_ARRAY_SCOPE.get() is None
        or not _REUSE_VERIFIED_ARRAYS.get()
    ):
        with span("verify", "source_value"):
            return builder()
    key = ("file-read-lease", *identity, *(os.fspath(path) for path in paths))
    while True:
        with _VERIFIED_ARRAY_CACHE_LOCK:
            cached = _VERIFIED_ARRAY_CACHE.get(key)
            leases = _VERIFIED_SOURCE_LEASES.get(key)
            if cached is not None and leases and all(lease.unchanged() for lease in leases):
                _VERIFIED_ARRAY_CACHE.move_to_end(key)
                _VERIFIED_SOURCE_LEASES.move_to_end(key)
                _VERIFIED_ARRAY_PROCESS_CACHE_HITS += 1
                return cast(T, cached[0])
            pending = _VERIFIED_SOURCE_BUILDS.get(key)
            if pending is None:
                _VERIFIED_ARRAY_PROCESS_CACHE_MISSES += 1
                _close_source_leases(key)
                if cached is not None:
                    global _VERIFIED_ARRAY_CACHE_BYTES
                    _VERIFIED_ARRAY_CACHE_BYTES -= _VERIFIED_ARRAY_CACHE.pop(key)[1]
                pending = Future()
                _VERIFIED_SOURCE_BUILDS[key] = pending
                break
        pending.result()
    acquired: list[_WindowsReadLease] = []

    def build() -> T:
        token = _VERIFIED_SOURCE_BUILD.set(True)
        try:
            with span("verify", "source_value"):
                return builder()
        finally:
            _VERIFIED_SOURCE_BUILD.reset(token)

    try:
        try:
            for path in paths:
                acquired.append(_acquire_source_lease(path))
        except BaseException as error:
            for lease in acquired:
                lease.close()
            acquired = []
            if isinstance(error, Exception):
                return build()
            raise
        try:
            value = build()
            if not all(lease.unchanged() for lease in acquired):
                for lease in acquired:
                    lease.close()
                acquired = []
                return build()
            size = nbytes(value) if callable(nbytes) else nbytes
            if size < 0:
                raise ValueError("content_store.verified_value_size_invalid")
            if (
                size <= _VERIFIED_ARRAY_PROCESS_CACHE_LIMIT_BYTES
                and _immutable_request_value(value)
                and all(lease.unchanged() for lease in acquired)
            ):
                _remember_process_value(key, value, size, leases=tuple(acquired))
                acquired = []
            return value
        finally:
            for lease in acquired:
                lease.close()
    except BaseException as error:
        pending.set_exception(error)
        raise
    finally:
        with _VERIFIED_ARRAY_CACHE_LOCK:
            _VERIFIED_SOURCE_BUILDS.pop(key)
        if not pending.done():
            pending.set_result(None)


def verify_source_checks(checks: Iterable[Callable[[], None]]) -> None:
    """Complete independent full source-owner validations with a process bound of two.

    Cross-record bindings, dependencies and control checks remain in the caller. Each
    callback performs its owner's complete byte/hash, structure and local semantic checks;
    a structural-only check cannot replace a complete source-owner proof. Callbacks run in
    fresh Contexts and independent model/array/proof scopes, inheriting only the caller's
    respective reuse and declared-source-build policies. At most two callbacks are submitted
    process-wide; nested calls run serially to avoid waiting on their own pool. Every
    submitted check completes before return or refusal, with errors in input order.
    """
    array_reuse = _REUSE_VERIFIED_ARRAYS.get()
    model_reuse = _REUSE_VERIFIED_MODELS.get()
    source_build = _VERIFIED_SOURCE_BUILD.get()

    def run(check: Callable[[], None]) -> None:
        _VERIFIED_SOURCE_CHECK_RUNNING.set(True)
        _VERIFIED_SOURCE_BUILD.set(source_build)
        with (
            verified_array_read_scope(reuse_verified=array_reuse),
            verified_model_read_scope(reuse_verified=model_reuse),
            span("verify", "source_check"),
        ):
            check()

    failure: BaseException | None = None
    if _VERIFIED_SOURCE_CHECK_RUNNING.get():
        try:
            for check in checks:
                try:
                    Context().run(run, check)
                except BaseException as error:
                    if failure is None:
                        failure = error
        except BaseException as error:
            if failure is None:
                failure = error
        if failure is not None:
            raise failure
        return

    global _VERIFIED_SOURCE_CHECK_POOL
    with _VERIFIED_SOURCE_CHECK_POOL_LOCK:
        if _VERIFIED_SOURCE_CHECK_POOL is None:
            _VERIFIED_SOURCE_CHECK_POOL = ThreadPoolExecutor(
                max_workers=_VERIFIED_SOURCE_CHECK_WORKERS,
                thread_name_prefix="content-source-check",
            )
        pool = _VERIFIED_SOURCE_CHECK_POOL
    pending: list[Future[None]] = []
    submission_error: BaseException | None = None
    try:
        for check in checks:
            _VERIFIED_SOURCE_CHECK_SLOTS.acquire()
            try:
                future = pool.submit(Context().run, run, check)
            except BaseException:
                _VERIFIED_SOURCE_CHECK_SLOTS.release()
                raise
            future.add_done_callback(lambda _future: _VERIFIED_SOURCE_CHECK_SLOTS.release())
            pending.append(future)
    except BaseException as error:
        submission_error = error
    for future in pending:
        while True:
            try:
                future.result()
                break
            except BaseException as error:
                if failure is None:
                    failure = error
                if future.done():
                    break
    if failure is not None:
        raise failure
    if submission_error is not None:
        raise submission_error


@contextmanager
def verified_array_read_scope(*, reuse_verified: bool = False) -> Iterator[None]:
    """Share immutable NPZ values in one request, optionally reusing a bounded process LRU.

    Nested owners join the existing scope and cannot widen its cross-request policy. The
    persistent LRU (bounded independently to 512 MiB) is used only when the outer operation
    explicitly opts in; the local request cache always clears when its outer scope exits.
    """
    if _VERIFIED_ARRAY_SCOPE.get() is not None:
        yield
        return
    scope: OrderedDict[_VERIFIED_ARRAY_KEY, tuple[object, int]] = OrderedDict()
    token = _VERIFIED_ARRAY_SCOPE.set(scope)
    reuse_token = _REUSE_VERIFIED_ARRAYS.set(reuse_verified)
    proof_token = _VERIFIED_PROOF_SCOPE.set(set())
    try:
        yield
    finally:
        scope.clear()
        _VERIFIED_PROOF_SCOPE.reset(proof_token)
        _REUSE_VERIFIED_ARRAYS.reset(reuse_token)
        _VERIFIED_ARRAY_SCOPE.reset(token)


def verified_request_proof(identity: tuple[object, ...], check: Callable[[], None]) -> None:
    """Reuse an exact-byte successful proof without retaining the numerical values.

    Owners must fully read/hash the current sources before calling, and bind the key to
    their namespace, resolved paths, same-read file identities and content commitments.
    Failed checks are never remembered. The separate 128-entry proof LRU cannot be evicted
    by large decoded arrays; across-request use still requires the outer read's opt-in.
    """
    scope = _VERIFIED_PROOF_SCOPE.get()
    if scope is not None:
        if identity in scope:
            return
        if _REUSE_VERIFIED_ARRAYS.get():
            with _VERIFIED_PROOF_CACHE_LOCK:
                if identity in _VERIFIED_PROOF_CACHE:
                    _VERIFIED_PROOF_CACHE.move_to_end(identity)
                    scope.add(identity)
                    return
    with span("verify", "request_proof"):
        check()
    if scope is not None:
        scope.add(identity)
        if _REUSE_VERIFIED_ARRAYS.get():
            with _VERIFIED_PROOF_CACHE_LOCK:
                _VERIFIED_PROOF_CACHE[identity] = None
                _VERIFIED_PROOF_CACHE.move_to_end(identity)
                while len(_VERIFIED_PROOF_CACHE) > _VERIFIED_PROOF_CACHE_LIMIT:
                    _VERIFIED_PROOF_CACHE.popitem(last=False)


def verified_request_value[T](
    identity: tuple[object, ...],
    builder: Callable[[], T],
    *,
    nbytes: int | Callable[[T], int],
) -> T:
    """Memoize one immutable, validated value under local and optional process byte bounds.

    Owners must re-read and fully verify every current source artifact before calling
    this helper. The identity must bind the owner namespace, exact content commitments,
    and the source file's resolved path and identity. Builders that raise are never cached.
    Values larger than either cache's byte limit are returned uncached. The two limits apply
    separately; active requests and caller-held references can keep evicted bytes alive.
    """
    key = tuple(identity)
    scope = _VERIFIED_ARRAY_SCOPE.get()
    if scope is not None and key in scope:
        value, _ = scope[key]
        scope.move_to_end(key)
        return cast(T, value)

    if scope is not None and _REUSE_VERIFIED_ARRAYS.get():
        global _VERIFIED_ARRAY_PROCESS_CACHE_HITS, _VERIFIED_ARRAY_PROCESS_CACHE_MISSES
        with _VERIFIED_ARRAY_CACHE_LOCK:
            cached = _VERIFIED_ARRAY_CACHE.get(key)
            if cached is not None:
                _VERIFIED_ARRAY_CACHE.move_to_end(key)
                _VERIFIED_ARRAY_PROCESS_CACHE_HITS += 1
            else:
                _VERIFIED_ARRAY_PROCESS_CACHE_MISSES += 1
        if cached is not None:
            value, size = cached
            if size <= _VERIFIED_ARRAY_CACHE_LIMIT_BYTES:
                _remember_request_value(scope, key, value, size)
            return cast(T, value)

    value = builder()
    size = nbytes(value) if callable(nbytes) else nbytes
    if size < 0:
        raise ValueError("content_store.verified_value_size_invalid")
    if scope is not None and _immutable_request_value(value):
        if size <= _VERIFIED_ARRAY_CACHE_LIMIT_BYTES:
            _remember_request_value(scope, key, value, size)
        if (
            _REUSE_VERIFIED_ARRAYS.get()
            and not _VERIFIED_SOURCE_BUILD.get()
            and size <= _VERIFIED_ARRAY_PROCESS_CACHE_LIMIT_BYTES
        ):
            _remember_process_value(key, value, size)
    return value


def _remember_request_value(
    scope: OrderedDict[_VERIFIED_ARRAY_KEY, tuple[object, int]],
    key: _VERIFIED_ARRAY_KEY,
    value: object,
    size: int,
) -> None:
    retained = sum(entry_size for _, entry_size in scope.values())
    while scope and retained + size > _VERIFIED_ARRAY_CACHE_LIMIT_BYTES:
        _, (_, evicted_size) = scope.popitem(last=False)
        retained -= evicted_size
    scope[key] = (value, size)


def _remember_process_value(
    key: _VERIFIED_ARRAY_KEY,
    value: object,
    size: int,
    *,
    leases: tuple[_WindowsReadLease, ...] = (),
) -> None:
    global _VERIFIED_ARRAY_CACHE_BYTES, _VERIFIED_ARRAY_PROCESS_CACHE_EVICTIONS
    with _VERIFIED_ARRAY_CACHE_LOCK:
        previous = _VERIFIED_ARRAY_CACHE.pop(key, None)
        _close_source_leases(key)
        if previous is not None:
            _VERIFIED_ARRAY_CACHE_BYTES -= previous[1]
        while (
            _VERIFIED_ARRAY_CACHE
            and _VERIFIED_ARRAY_CACHE_BYTES + size > _VERIFIED_ARRAY_PROCESS_CACHE_LIMIT_BYTES
        ):
            evicted_key, (_, evicted_size) = _VERIFIED_ARRAY_CACHE.popitem(last=False)
            _close_source_leases(evicted_key)
            _VERIFIED_ARRAY_CACHE_BYTES -= evicted_size
            _VERIFIED_ARRAY_PROCESS_CACHE_EVICTIONS += 1
        if size <= _VERIFIED_ARRAY_PROCESS_CACHE_LIMIT_BYTES:
            _VERIFIED_ARRAY_CACHE[key] = (value, size)
            _VERIFIED_ARRAY_CACHE_BYTES += size
            if leases:
                while (
                    _VERIFIED_SOURCE_LEASES
                    and sum(len(held) for held in _VERIFIED_SOURCE_LEASES.values()) + len(leases)
                    > _VERIFIED_SOURCE_LEASE_LIMIT
                ):
                    prior = next(iter(_VERIFIED_SOURCE_LEASES))
                    _close_source_leases(prior)
                    _VERIFIED_ARRAY_CACHE_BYTES -= _VERIFIED_ARRAY_CACHE.pop(prior)[1]
                    _VERIFIED_ARRAY_PROCESS_CACHE_EVICTIONS += 1
                _VERIFIED_SOURCE_LEASES[key] = leases


def verified_npz_arrays(
    payload: bytes, *, identity: tuple[object, ...]
) -> dict[str, npt.NDArray[Any]]:
    """Decode an already verified NPZ once per exact request or opted-in process identity.

    Callers must read and fully hash the current file on every invocation before
    calling this function, and include that same-read digest, expected artifact
    hash, resolved path, file identity and owner namespace in ``identity``. This
    function does not grant verification or current-pointer authority. It only
    avoids repeating NPZ inflation after those checks have succeeded. Cross-request
    reuse is available only within ``verified_array_read_scope(reuse_verified=True)``.

    Returned mappings are fresh shells. Their arrays are backed by immutable
    ``bytes`` so neither array flags nor mutation through a shared view can alter
    the request cache.
    """

    def build_arrays() -> tuple[tuple[str, npt.NDArray[Any]], ...]:
        with (
            span("serialize", "npz_decode"),
            np.load(BytesIO(payload), allow_pickle=False) as archive,
        ):
            values: list[tuple[str, npt.NDArray[Any]]] = []
            for name in archive.files:
                array = archive[name]
                values.append(
                    (
                        name,
                        np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
                            array.shape
                        ),
                    )
                )
        return tuple(values)

    cached = verified_request_value(
        ("npz-arrays", *identity),
        build_arrays,
        nbytes=lambda values: sum(array.nbytes for _, array in values),
    )
    return dict(cached)


@contextmanager
def verified_model_read_scope(*, reuse_verified: bool = False) -> Iterator[None]:
    """Share verified immutable JSON models during one request, optionally across reads.

    Nested owner scopes join the outer request and cannot widen its across-request policy.
    A separate bounded process cache is consulted only when the outer scope explicitly opts
    in; ordinary execution and write paths therefore keep validating from disk.
    """
    with verified_array_read_scope(reuse_verified=reuse_verified):
        if _VERIFIED_MODEL_SCOPE.get() is not None:
            yield
            return
        scope: dict[_VERIFIED_MODEL_KEY, BaseModel] = {}
        scope_token = _VERIFIED_MODEL_SCOPE.set(scope)
        reuse_token = _REUSE_VERIFIED_MODELS.set(reuse_verified)
        try:
            yield
        finally:
            scope.clear()
            _REUSE_VERIFIED_MODELS.reset(reuse_token)
            _VERIFIED_MODEL_SCOPE.reset(scope_token)


def _deeply_immutable_model(value: BaseModel) -> bool:
    """Whether a frozen Pydantic model contains only recursively immutable values."""
    if not value.model_config.get("frozen"):
        return False

    def immutable(item: object) -> bool:
        if item is None or isinstance(
            item,
            (str, bytes, int, float, bool, date, datetime, time, timedelta, Decimal, Enum, UUID),
        ):
            return True
        if isinstance(item, BaseModel):
            return (
                item.model_config.get("frozen", False)
                and not getattr(item, "__pydantic_private__", None)
                and getattr(item, "__pydantic_extra__", None) is None
                and all(immutable(getattr(item, name)) for name in item.__class__.model_fields)
            )
        if isinstance(item, (tuple, frozenset)):
            return all(immutable(child) for child in item)
        if (
            not isinstance(item, type)
            and is_dataclass(item)
            and getattr(getattr(item, "__dataclass_params__", None), "frozen", False)
        ):
            return all(immutable(getattr(item, field.name)) for field in fields(item))
        return False

    return (
        not getattr(value, "__pydantic_private__", None)
        and getattr(value, "__pydantic_extra__", None) is None
        and all(immutable(getattr(value, name)) for name in value.__class__.model_fields)
    )


def _immutable_request_value(value: object) -> bool:
    """Refuse to retain objects whose reachable state can be changed by a caller."""
    if value is None or isinstance(
        value,
        (str, bytes, int, float, bool, date, datetime, time, timedelta, Decimal, Enum, UUID),
    ):
        return True
    if isinstance(value, np.ndarray):
        if value.flags.writeable:
            return False
        current: object = value
        while isinstance(current, np.ndarray):
            current = current.base
        return isinstance(current, bytes)
    if isinstance(value, BaseModel):
        return _deeply_immutable_model(value)
    if isinstance(value, (tuple, frozenset)):
        return all(_immutable_request_value(item) for item in value)
    if (
        not isinstance(value, type)
        and is_dataclass(value)
        and getattr(getattr(value, "__dataclass_params__", None), "frozen", False)
    ):
        return all(_immutable_request_value(getattr(value, field.name)) for field in fields(value))
    return False


def _verified_model_file_identity(path: Path, payload: bytes) -> tuple[object, ...] | None:
    """Bind reuse to the resolved file, stat identity and exact bytes read this time."""
    try:
        resolved = path.resolve(strict=False)
        info = path.stat()
    except OSError:
        return None
    return (
        os.fspath(resolved),
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        hashlib.sha256(payload).hexdigest(),
    )


def _verified_model_get(key: _VERIFIED_MODEL_KEY) -> BaseModel | None:
    scope = _VERIFIED_MODEL_SCOPE.get()
    if scope is not None:
        value = scope.get(key)
        if value is not None:
            return value
        if _REUSE_VERIFIED_MODELS.get():
            with _VERIFIED_MODEL_CACHE_LOCK:
                value = _VERIFIED_MODEL_CACHE.get(key)
                if value is not None:
                    _VERIFIED_MODEL_CACHE.move_to_end(key)
            if value is not None:
                _verified_model_put(scope, key, value)
                return value
    return None


def _verified_model_put(
    scope: dict[_VERIFIED_MODEL_KEY, BaseModel],
    key: _VERIFIED_MODEL_KEY,
    value: BaseModel,
) -> None:
    scope[key] = value
    if _REUSE_VERIFIED_MODELS.get():
        with _VERIFIED_MODEL_CACHE_LOCK:
            _VERIFIED_MODEL_CACHE[key] = value
            _VERIFIED_MODEL_CACHE.move_to_end(key)
            while len(_VERIFIED_MODEL_CACHE) > _VERIFIED_MODEL_CACHE_LIMIT:
                _VERIFIED_MODEL_CACHE.popitem(last=False)


def replace_shared_file(staged: Path, destination: Path) -> None:
    """Replace a file other requests may hold open to read it, despite their brief locks.

    Windows refuses to replace a file another handle holds open, and a Host's concurrent reads
    hold one while they read: the replace waits out short backoffs before it gives up (V477,
    AX17's overlapping trial reads). Elsewhere the first replace stands.
    """
    for attempt in range(_WINDOWS_REPLACE_ATTEMPTS):
        try:
            os.replace(staged, destination)
            return
        except PermissionError:
            if os.name != "nt" or attempt + 1 == _WINDOWS_REPLACE_ATTEMPTS:
                raise
            sleep(0.01 * (2**attempt))


class ContentAddressedStoreError(ValueError):
    """Stable refusal for identity reuse, absence, or tamper."""


def columns_digest(columns: Mapping[str, npt.NDArray[Any]]) -> str:
    """The digest numerical columns are named by: SHA-256 over their C-order bytes, in order.

    One column's digest is that of its packed bytes, the name its ``.bin`` lane had, so a
    lane keeps its identity across the type change (V210).
    """
    with span("hash", "columns"):
        digest = hashlib.sha256()
        for values in columns.values():
            digest.update(np.ascontiguousarray(values).tobytes())
        return digest.hexdigest()


class ContentAddressedStore:
    """JSON contracts, Parquet lanes and HTML pages under one caller-owned root."""

    def __init__(
        self, root: Path, *, uri_prefix: str, capacity: Callable[[int], None] | None = None
    ) -> None:
        """Configure a content store and optional write-capacity check.

        Args:
            root: Workspace or store root used by this owner.
            uri_prefix: Public artifact address prefix; its trailing slash is removed.
            capacity: Optional capacity check called with the byte count before each atomic write.
        """
        self.root = root.resolve()
        self.uri_prefix = uri_prefix.rstrip("/")
        self.capacity = capacity

    @staticmethod
    def require_hash(value: str) -> None:
        """Require exactly 64 lowercase hexadecimal identity characters.

        Args:
            value: Identity string to validate.

        Raises:
            ContentAddressedStoreError: The value is not a lowercase SHA-256 representation.
        """
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ContentAddressedStoreError("content_store.identity_invalid")

    def atomic_write(self, path: Path, content: bytes) -> None:
        """Check capacity and atomically replace the target with exact content bytes.

        Args:
            path: Physical file path owned by the caller.
            content: Exact bytes to publish through a same-directory temporary file.
        """
        if self.capacity is not None:
            self.capacity(len(content))
        with span("write", "content_store"):
            path.parent.mkdir(parents=True, exist_ok=True)
            staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            staged.write_bytes(content)
            os.replace(staged, path)
            staged.unlink(missing_ok=True)

    def uri(self, category: str, content_hash: str, *, extension: str = "json") -> str:
        """Construct an artifact URI after validating its content hash.

        Args:
            category: Caller-owned artifact category below the store root.
            content_hash: Exact content identity used in the artifact address.
            extension: Artifact filename extension; defaults to JSON.

        Returns:
            Public category/hash URI with the requested extension.

        Raises:
            ContentAddressedStoreError: The content hash is malformed.
        """
        self.require_hash(content_hash)
        return f"{self.uri_prefix}/{category}/{content_hash}.{extension}"

    def publish_model(self, *, category: str, value: BaseModel, identity_field: str) -> str:
        """Publish deterministic model JSON without reusing an identity for different bytes.

        Args:
            category: Caller-owned artifact category below the store root.
            value: Pydantic contract containing the declared identity field.
            identity_field: Contract field containing the content identity.

        Returns:
            Public URI of the stored JSON contract.

        Raises:
            ContentAddressedStoreError: The hash is invalid or an existing identity names different
                bytes.
        """
        content_hash = str(getattr(value, identity_field))
        self.require_hash(content_hash)
        with span("serialize", "model"):
            content = json.dumps(
                value.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode()
        target = self.root / category / f"{content_hash}.json"
        if target.is_file() and target.read_bytes() != content:
            raise ContentAddressedStoreError("content_store.identity_reused")
        if not target.is_file():
            self.atomic_write(target, content)
        return self.uri(category, content_hash)

    def load_model[ContractT: BaseModel](
        self,
        *,
        category: str,
        content_hash: str,
        model: type[ContractT],
        identity_field: str,
    ) -> ContractT:
        """Load and validate a JSON contract at its exact content identity.

        Args:
            category: Caller-owned artifact category below the store root.
            content_hash: Exact content identity used in the artifact address.
            model: Pydantic contract used to validate the stored JSON.
            identity_field: Contract field containing the content identity.

        Returns:
            Validated instance of the supplied contract.

        Raises:
            ContentAddressedStoreError: The artifact is missing, invalid, or inconsistent with its
                address.
        """
        self.require_hash(content_hash)
        target = self.root / category / f"{content_hash}.json"
        scope = _VERIFIED_MODEL_SCOPE.get()
        cacheable = scope is not None and model.model_config.get("frozen")

        def verify() -> ContractT:
            file_identity = None
            try:
                with span("read", "model"):
                    payload = target.read_bytes()
            except FileNotFoundError as error:
                raise ContentAddressedStoreError(
                    f"content_store.artifact_missing:{content_hash}"
                ) from error
            except Exception as error:
                raise ContentAddressedStoreError("content_store.artifact_tampered") from error
            if cacheable:
                file_identity = _verified_model_file_identity(target, payload)
                if file_identity is not None:
                    key = (*file_identity, content_hash, model, identity_field)
                    kept = _verified_model_get(key)
                    if kept is not None:
                        return cast(ContractT, kept)
            try:
                with span("serialize", "model_validate"):
                    value = model.model_validate_json(payload)
            except Exception as error:
                raise ContentAddressedStoreError("content_store.artifact_tampered") from error
            if getattr(value, identity_field) != content_hash:
                raise ContentAddressedStoreError("content_store.artifact_tampered")
            if cacheable and file_identity is not None and _deeply_immutable_model(value):
                assert scope is not None
                _verified_model_put(scope, key, value)
            return cast(ContractT, value)

        return cast(
            ContractT,
            verified_source_value(
                (
                    "content-store-model",
                    str(self.root),
                    category,
                    content_hash,
                    model,
                    identity_field,
                ),
                (target,),
                verify,
                nbytes=lambda _: target.stat().st_size,
            ),
        )

    def publish_columns(self, *, category: str, columns: Mapping[str, npt.NDArray[Any]]) -> str:
        """Write numerical columns of one shape as one Parquet table, named by their digest.

        Each column is stored flat, the shape in the table's metadata. Plain encoding, never a
        dictionary, keeps every value's bits, NaN payloads and signed zeros included, so the
        digest holds on every read.
        """
        arrays = {name: np.ascontiguousarray(values) for name, values in columns.items()}
        shapes = {array.shape for array in arrays.values()}
        if len(shapes) != 1:
            raise ContentAddressedStoreError("content_store.columns_invalid")
        content_hash = columns_digest(arrays)
        target = self.root / category / f"{content_hash}.parquet"
        if target.is_file():
            if self._packed(target) != b"".join(array.tobytes() for array in arrays.values()):
                raise ContentAddressedStoreError("content_store.identity_reused")
            return content_hash
        with span("serialize", "columns"):
            table = pa.table({name: array.reshape(-1) for name, array in arrays.items()})
            table = table.replace_schema_metadata({"shape": json.dumps(list(shapes.pop()))})
            sink = pa.BufferOutputStream()
            pq.write_table(table, sink, use_dictionary=False)
        self.atomic_write(target, sink.getvalue().to_pybytes())
        return content_hash

    def holds_columns(self, *, category: str, content_hash: str) -> bool:
        """Whether this store holds the Parquet table a digest names."""
        self.require_hash(content_hash)
        return (self.root / category / f"{content_hash}.parquet").is_file()

    def load_columns(self, *, category: str, content_hash: str) -> dict[str, npt.NDArray[Any]]:
        """Read one table's columns, flat, and prove they are the ones its digest names."""
        self.require_hash(content_hash)
        target = self.root / category / f"{content_hash}.parquet"
        columns = self._columns(target)
        if columns_digest(columns) != content_hash:
            raise ContentAddressedStoreError("content_store.artifact_tampered")
        return columns

    def load_packed_bytes(self, *, category: str, content_hash: str) -> bytes:
        """The bytes a lane's digest names: its columns' C-order bytes, in order.

        A store written before V210 holds them as a ``.bin`` file, read as it is.
        """
        self.require_hash(content_hash)
        target = self.root / category / f"{content_hash}.parquet"
        payload = self._packed(target) if target.is_file() else self._legacy(category, content_hash)
        with span("hash", "packed"):
            if hashlib.sha256(payload).hexdigest() != content_hash:
                raise ContentAddressedStoreError("content_store.artifact_tampered")
        return payload

    def publish_document(self, *, category: str, payload: bytes, extension: str) -> str:
        """Write one JSON document or rendered page, named by the digest of its bytes."""
        if extension not in {"json", "html"}:
            raise ContentAddressedStoreError("content_store.document_type_invalid")
        content_hash = hashlib.sha256(payload).hexdigest()
        target = self.root / category / f"{content_hash}.{extension}"
        if target.is_file() and target.read_bytes() != payload:
            raise ContentAddressedStoreError("content_store.identity_reused")
        if not target.is_file():
            self.atomic_write(target, payload)
        return content_hash

    def load_document(self, *, category: str, content_hash: str, extension: str) -> bytes:
        """Read one document and prove it is the one asked for; a pre-V210 store's is ``.bin``."""
        self.require_hash(content_hash)
        target = self.root / category / f"{content_hash}.{extension}"
        try:
            with span("read", "document"):
                payload = target.read_bytes()
        except FileNotFoundError:
            payload = self._legacy(category, content_hash)
        with span("hash", "document"):
            if hashlib.sha256(payload).hexdigest() != content_hash:
                raise ContentAddressedStoreError("content_store.artifact_tampered")
        return payload

    def _columns(self, target: Path) -> dict[str, npt.NDArray[Any]]:
        try:
            # A Parquet file opens and closes with its magic; the reader checks only the end.
            with span("read", "columns"):
                with target.open("rb") as handle:
                    if handle.read(4) != b"PAR1":
                        raise ContentAddressedStoreError("content_store.artifact_tampered")
                table = pq.read_table(target)
                return {
                    name: np.ascontiguousarray(table.column(name).to_numpy())
                    for name in table.column_names
                }
        except FileNotFoundError as error:
            raise ContentAddressedStoreError(
                f"content_store.artifact_missing:{target.stem}"
            ) from error
        except (OSError, pa.ArrowException) as error:
            raise ContentAddressedStoreError("content_store.artifact_tampered") from error

    def _packed(self, target: Path) -> bytes:
        return b"".join(values.tobytes() for values in self._columns(target).values())

    def _legacy(self, category: str, content_hash: str) -> bytes:
        # A store written before V210 packed its lanes, documents and pages as `.bin` bytes;
        # read only, until the formats registry names their upgrader (V267).
        try:
            return (self.root / category / f"{content_hash}.bin").read_bytes()
        except FileNotFoundError as error:
            raise ContentAddressedStoreError(
                f"content_store.artifact_missing:{content_hash}"
            ) from error


@dataclass(frozen=True, slots=True)
class CommittedKind[ContractT: BaseModel]:
    """One kind of one-time record a store commits: where it is indexed and kept.

    Attributes:
        index: The index its entries are written under, one per key.
        category: The content category its artifacts materialize in.
        model: The contract it is read back as.
        identity_field: The field that holds its content identity.
    """

    index: str
    category: str
    model: type[ContractT]
    identity_field: str


class CommittedIndex:
    """One atomic commit point that carries the artifact it points at.

    Lives here, beside the content store it wraps, because two independent
    authorities need it -- Portfolio's finalization store and the Validation
    Gate's -- and the crash semantics have to be stated once. Putting it in
    either owner's module would make the other depend on its peer for a storage
    primitive.

    The rule: a commit writes one index entry carrying the artifact's whole
    canonical payload, with a single atomic replace; the content file is a
    materialization of that entry, written afterwards. Either the entry is
    absent and nothing happened, or it is present and the artifact is fully
    recoverable from it. A reader that finds an entry and no content file heals
    forward.

    Content-first, index-last cannot close these windows, because the artifacts
    that pass through here carry timestamps. Recomputing one after a crash
    produces a different identity, so an orphaned content file could never be
    found again -- and the protocol would issue a second permit, or evaluate a
    second time, for want of a file already on disk.
    """

    def __init__(self, root: Path, content: ContentAddressedStore) -> None:
        """Bind an index root to the content store that validates its keys.

        Args:
            root: Workspace or store root used by this owner.
            content: Shared content store used to validate index keys.
        """
        self.root = root
        self.content = content

    def path(self, index: str, key: str) -> Path:
        """Resolve the physical index entry for a validated content key.

        Args:
            index: Caller-owned index namespace.
            key: Lowercase hexadecimal SHA-256 index key.

        Returns:
            Index namespace/key JSON path.

        Raises:
            ContentAddressedStoreError: The index key is not a lowercase SHA-256 representation.
        """
        self.content.require_hash(key)
        return self.root / "index" / index / f"{key}.json"

    def commit[ContractT: BaseModel](
        self, kind: CommittedKind[ContractT], key: str, value: ContractT
    ) -> str:
        """Write the index entry, then materialize the content. Idempotent.

        The entry is the commit. A second commit of the *same* artifact is a
        no-op that heals any missing content file; a commit of a *different*
        artifact under the same key is refused, because that key names a
        one-time fact -- one permit per candidate, one package per permit, one
        handoff per package.

        Args:
            kind: The record's kind.
            key: The one-time fact its entry is keyed by.
            value: The record.

        Returns:
            Its content identity.

        Raises:
            ContentAddressedStoreError: If another record holds the key.
        """
        identity = str(getattr(value, kind.identity_field))
        self.content.require_hash(identity)
        entry = json.dumps(
            {"key": key, "identity": identity, "payload": value.model_dump(mode="json")},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        target = self.path(kind.index, key)
        if target.is_file() and target.read_bytes() != entry:
            raise ContentAddressedStoreError("content_store.committed_identity_reused")
        if not target.is_file():
            self.content.atomic_write(target, entry)
        self.content.publish_model(
            category=kind.category, value=value, identity_field=kind.identity_field
        )
        return identity

    def open[ContractT: BaseModel](
        self, kind: CommittedKind[ContractT], key: str
    ) -> ContractT | None:
        """Reopen a committed artifact, healing a missing content file.

        Validates the payload the entry carries rather than trusting the content
        file: if the two ever disagreed, the entry is the one that was written
        atomically, and re-publishing from it is what makes the disagreement
        impossible to observe twice.

        Args:
            kind: The record's kind.
            key: The one-time fact its entry is keyed by.

        Returns:
            The record, or None when nothing was committed under the key.

        Raises:
            ContentAddressedStoreError: If the entry does not hold what it names.
        """
        target = self.path(kind.index, key)
        if not target.is_file():
            return None
        try:
            entry = json.loads(target.read_text(encoding="utf-8"))
            if entry["key"] != key:
                raise ValueError
            value = kind.model.model_validate(entry["payload"])
            if str(getattr(value, kind.identity_field)) != entry["identity"]:
                raise ValueError
        except (ContentAddressedStoreError, KeyError, ValueError, json.JSONDecodeError) as error:
            raise ContentAddressedStoreError("content_store.committed_index_tampered") from error
        self.content.publish_model(
            category=kind.category, value=value, identity_field=kind.identity_field
        )
        return cast(ContractT, value)

    def load[ContractT: BaseModel](
        self, kind: CommittedKind[ContractT], content_hash: str
    ) -> ContractT:
        """Load a committed record by its content identity and verify it.

        Args:
            kind: The record's kind.
            content_hash: Its content identity.

        Returns:
            The record.
        """
        return self.content.load_model(
            category=kind.category,
            content_hash=content_hash,
            model=kind.model,
            identity_field=kind.identity_field,
        )


__all__ = [
    "CommittedIndex",
    "CommittedKind",
    "ContentAddressedStore",
    "ContentAddressedStoreError",
    "columns_digest",
    "verified_array_read_scope",
    "verified_model_read_scope",
    "verified_npz_arrays",
    "verified_request_proof",
    "verified_request_value",
    "verified_source_value",
    "verify_source_checks",
]
