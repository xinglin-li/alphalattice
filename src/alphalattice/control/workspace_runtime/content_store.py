"""Small atomic content-addressed primitive shared by legacy and public stores.

Each file is named by its content's digest and written in its DA9 type: a contract or a
document as JSON, numerical lanes as Parquet, a rendered page as HTML. A legacy store
kept lanes, documents and pages as packed ``.bin`` bytes under the same digest; those are
read, never written, until the formats registry names their upgrader.

A sealed file is proved from its bytes when it enters (a write, an import, a stage's check of what
it just wrote), when it leaves (an export) and by the background scrub, never on an ordinary read
(LAWS.md EV2): the Host is its only writer and nothing rewrites a sealed name. A read inside
``full_content_proof()`` proves every sealed byte it reads; any other read takes the bytes at
their name and keeps its structural checks.
"""

from __future__ import annotations

import base64
import copy
import ctypes
import functools
import json
import marshal
import os
import shutil
import sys
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
from threading import BoundedSemaphore, Lock
from time import sleep
from types import CodeType
from typing import Any, cast
from uuid import UUID, uuid4

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from alphalattice.control.workspace_runtime.verified_facts import (
    _FULL_PROOF,
    file_fact,
    full_content_proof,
    full_proof,
    record_file_fact,
)
from alphalattice.kernel.identity import integrity_sha256, integrity_sha256_path
from alphalattice.kernel.resources import worker_width
from alphalattice.kernel.shared_kernel.environment import recorded_environment
from alphalattice.kernel.shared_kernel.spans import span

_WINDOWS_REPLACE_ATTEMPTS = 6
_VERIFIED_MODEL_CACHE_LIMIT = 128
_VERIFIED_MODEL_KEY = tuple[object, ...]
_VERIFIED_MODEL_SCOPE: ContextVar[dict[_VERIFIED_MODEL_KEY, BaseModel] | None] = ContextVar(
    "content_store_verified_models", default=None
)
_VERIFIED_MODEL_CACHE: OrderedDict[_VERIFIED_MODEL_KEY, BaseModel] = OrderedDict()
_VERIFIED_MODEL_CACHE_LOCK = Lock()
_VERIFIED_ARRAY_CACHE_LIMIT_BYTES = 512 * 1024 * 1024
_VERIFIED_ARRAY_PROCESS_CACHE_LIMIT_BYTES = 512 * 1024 * 1024
_VERIFIED_ARRAY_KEY = tuple[object, ...]
_VERIFIED_ARRAY_SCOPE: ContextVar[OrderedDict[_VERIFIED_ARRAY_KEY, tuple[object, int]] | None] = (
    ContextVar("content_store_verified_arrays", default=None)
)
_VERIFIED_SOURCE_BUILD: ContextVar[bool] = ContextVar(
    "content_store_verified_source_build", default=False
)
_VERIFIED_ARRAY_CACHE: OrderedDict[_VERIFIED_ARRAY_KEY, tuple[object, int]] = OrderedDict()
_VERIFIED_ARRAY_CACHE_BYTES = 0
_VERIFIED_ARRAY_CACHE_LOCK = Lock()
_VERIFIED_SOURCE_BUILDS: dict[_VERIFIED_ARRAY_KEY, Future[None]] = {}
_VERIFIED_SOURCE_CHECK_WORKERS = 2
_VERIFIED_SOURCE_CHECK_POOL: ThreadPoolExecutor | None = None
_VERIFIED_SOURCE_CHECK_POOL_LOCK = Lock()
_VERIFIED_SOURCE_CHECK_SLOTS = BoundedSemaphore(_VERIFIED_SOURCE_CHECK_WORKERS)
_VERIFIED_SOURCE_CHECK_RUNNING: ContextVar[bool] = ContextVar(
    "content_store_verified_source_check_running", default=False
)
_VERIFIED_ARRAY_PROCESS_CACHE_HITS = 0
_VERIFIED_ARRAY_PROCESS_CACHE_MISSES = 0
_VERIFIED_ARRAY_PROCESS_CACHE_EVICTIONS = 0
_VERIFIED_PROOF_CACHE_LIMIT = 4096
_VERIFIED_PROOF_SCOPE: ContextVar[set[tuple[object, ...]] | None] = ContextVar(
    "content_store_verified_proofs", default=None
)
_VERIFIED_PROOF_CACHE: OrderedDict[tuple[object, ...], None] = OrderedDict()
_VERIFIED_PROOF_CACHE_LOCK = Lock()


def at_the_edge[**P, R](read: Callable[P, R]) -> Callable[P, R]:
    """`read` as an edge: each call runs inside ``full_content_proof()``, so what it reports as
    verified is proved from the bytes on disk, never from a value an earlier read kept."""

    @functools.wraps(read)
    def proved(*args: P.args, **kwargs: P.kwargs) -> R:
        with full_content_proof():
            return read(*args, **kwargs)

    return proved


@contextmanager
def staging_directory(parent: Path, prefix: str = ".staging-") -> Iterator[Path]:
    """A new staging folder under `parent` that keeps `parent`'s permissions, removed on leaving.

    Never `tempfile.mkdtemp` here: on Windows it restricts its folder to the account that made it
    (CVE-2024-4030), what is staged in it inherits that, and a rename keeps it. A Host running as
    another account (an agent's sandbox) would then publish data the workspace's own account
    cannot read. A plain folder inherits the workspace's access (POSIX: the umask).

    Args:
        parent: The folder the staged content will be renamed into, or beside.
        prefix: The staging folder's name prefix.

    Yields:
        The staging folder, absent again once the block leaves.
    """
    while True:
        folder = parent / f"{prefix}{uuid4().hex}"
        try:
            folder.mkdir()
            break
        except FileExistsError:
            continue
    try:
        yield folder
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def verified_source_value[T](
    identity: tuple[object, ...],
    sources: tuple[Path, ...],
    builder: Callable[[], T],
    *,
    nbytes: int | Callable[[T], int],
) -> T:
    """Build an owner's value from sealed sources once per process, and keep it (EV2).

    ``builder`` reads its declared sources and makes the owner's checks; ``identity`` binds the
    owner's namespace and the sources' content commitments, so a kept value answers every later
    read of the same sealed bytes in this process. Inside ``full_content_proof()`` the builder
    runs once per proof and its value replaces what was kept. Concurrent readers of one key share
    one producer; waiting never holds the cache lock. A builder owns every source it declares:
    helpers nested in it build in full and keep nothing of their own.
    """
    global _VERIFIED_ARRAY_PROCESS_CACHE_HITS, _VERIFIED_ARRAY_PROCESS_CACHE_MISSES
    if _VERIFIED_SOURCE_BUILD.get():
        return builder()
    facts = tuple(_file_identity(path) for path in sources)
    if None in facts:  # a source that cannot be read keeps nothing; its builder names the failure
        token = _VERIFIED_SOURCE_BUILD.set(True)
        try:
            return builder()
        finally:
            _VERIFIED_SOURCE_BUILD.reset(token)
    key = (
        "sealed-source",
        *identity,
        *((os.fspath(path.absolute()), fact) for path, fact in zip(sources, facts, strict=True)),
    )
    proved = _FULL_PROOF.get()
    while True:
        with _VERIFIED_ARRAY_CACHE_LOCK:
            if proved is not None and key in proved:
                return cast(T, proved[key])
            cached = _VERIFIED_ARRAY_CACHE.get(key)
            if cached is not None and proved is None:
                _VERIFIED_ARRAY_CACHE.move_to_end(key)
                _VERIFIED_ARRAY_PROCESS_CACHE_HITS += 1
                return cast(T, cached[0])
            pending = _VERIFIED_SOURCE_BUILDS.get(key)
            if pending is None:
                _VERIFIED_ARRAY_PROCESS_CACHE_MISSES += 1
                pending = Future()
                _VERIFIED_SOURCE_BUILDS[key] = pending
                break
        pending.result()
    token = _VERIFIED_SOURCE_BUILD.set(True)
    try:
        with span("verify", "source_value"):
            value = builder()
        size = nbytes(value) if callable(nbytes) else nbytes
        if size < 0:
            raise ValueError("content_store.verified_value_size_invalid")
        if size <= _VERIFIED_ARRAY_PROCESS_CACHE_LIMIT_BYTES and _immutable_request_value(value):
            _remember_process_value(key, value, size)
            if proved is not None:
                proved[key] = value
        return value
    except BaseException as error:
        pending.set_exception(error)
        raise
    finally:
        _VERIFIED_SOURCE_BUILD.reset(token)
        with _VERIFIED_ARRAY_CACHE_LOCK:
            _VERIFIED_SOURCE_BUILDS.pop(key)
        if not pending.done():
            pending.set_result(None)


def verify_source_checks(checks: Iterable[Callable[[], None]]) -> None:
    """Complete independent full source-owner validations with a process bound of two.

    Cross-record bindings, dependencies and control checks remain in the caller. Each
    callback performs its owner's checks. Callbacks run in fresh Contexts and independent
    model/array/proof scopes, inheriting only the caller's proof mode and declared-source-build
    policy. At most two callbacks are submitted
    process-wide; nested calls run serially to avoid waiting on their own pool. Every
    submitted check completes before return or refusal, with errors in input order.
    """
    full = _FULL_PROOF.get()
    source_build = _VERIFIED_SOURCE_BUILD.get()

    def run(check: Callable[[], None]) -> None:
        _VERIFIED_SOURCE_CHECK_RUNNING.set(True)
        _VERIFIED_SOURCE_BUILD.set(source_build)
        _FULL_PROOF.set(full)
        with (
            verified_array_read_scope(),
            verified_model_read_scope(),
            span("verify", "source_check"),
        ):
            check()

    failure: BaseException | None = None
    if _VERIFIED_SOURCE_CHECK_RUNNING.get() or worker_width(_VERIFIED_SOURCE_CHECK_WORKERS) == 1:
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
    failure = _joined(pending)
    if failure is not None:
        raise failure
    if submission_error is not None:
        raise submission_error


def _joined(pending: Iterable[Future[None]]) -> BaseException | None:
    """Wait for each submitted check in input order: the first check's own failure. An interrupt
    raised while waiting (Ctrl-C) is the caller's, never a check's: it is raised at once."""
    failure: BaseException | None = None
    for future in pending:
        try:
            future.result()
        except BaseException as error:
            if not future.done():
                raise
            if failure is None:
                failure = error
    return failure


@contextmanager
def verified_array_read_scope() -> Iterator[None]:
    """Share immutable values built in one request; nested owners join the outer scope.

    The process memo answers outside full proof whatever the scope; this request memo always
    clears when its outer scope exits.
    """
    if _VERIFIED_ARRAY_SCOPE.get() is not None:
        yield
        return
    scope: OrderedDict[_VERIFIED_ARRAY_KEY, tuple[object, int]] = OrderedDict()
    token = _VERIFIED_ARRAY_SCOPE.set(scope)
    proof_token = _VERIFIED_PROOF_SCOPE.set(set())
    try:
        yield
    finally:
        scope.clear()
        _VERIFIED_PROOF_SCOPE.reset(proof_token)
        _VERIFIED_ARRAY_SCOPE.reset(token)


def verified_request_proof(identity: tuple[object, ...], check: Callable[[], None]) -> None:
    """Keep a successful proof of sealed bytes without retaining the numerical values.

    Owners bind the key to their namespace, resolved paths and content commitments. Failed
    checks are never remembered, and full proof runs the check once per proof. The separate
    4,096-entry proof LRU cannot be evicted by large decoded arrays.
    """
    proved = _FULL_PROOF.get()
    if proved is not None:
        if ("request-proof", identity) in proved:
            return
        with span("verify", "request_proof"):
            check()
        proved["request-proof", identity] = None
        return
    scope = _VERIFIED_PROOF_SCOPE.get()
    if scope is not None:
        if identity in scope:
            return
        with _VERIFIED_PROOF_CACHE_LOCK:
            if identity in _VERIFIED_PROOF_CACHE:
                _VERIFIED_PROOF_CACHE.move_to_end(identity)
                scope.add(identity)
                return
    with span("verify", "request_proof"):
        check()
    if scope is not None:
        scope.add(identity)
        with _VERIFIED_PROOF_CACHE_LOCK:
            _VERIFIED_PROOF_CACHE[identity] = None
            _VERIFIED_PROOF_CACHE.move_to_end(identity)
            while len(_VERIFIED_PROOF_CACHE) > _VERIFIED_PROOF_CACHE_LIMIT:
                _VERIFIED_PROOF_CACHE.popitem(last=False)


class _Copied:
    """A kept value that is not deeply immutable: each caller receives its own deep copy."""

    __slots__ = ("value",)

    def __init__(self, value: object) -> None:
        self.value = value


def _handed(value: object) -> object:
    return copy.deepcopy(value.value) if isinstance(value, _Copied) else value


def verified_request_value[T](
    identity: tuple[object, ...],
    builder: Callable[[], T],
    *,
    nbytes: int | Callable[[T], int],
    copy_mutable: bool = False,
) -> T:
    """Memoize one immutable, validated value under local and optional process byte bounds.

    The identity binds the owner namespace, exact content commitments and the source
    file's resolved path. Builders that raise are never cached; full proof builds once per proof.
    Values larger than either cache's byte limit are returned uncached. The two limits apply
    separately; active requests and caller-held references can keep evicted bytes alive.

    A value that is not deeply immutable is returned uncached, unless the owner passes
    ``copy_mutable``: then it is kept too, and every caller, the first included, receives
    its own deep copy, so no caller can change what another reads. An owner opts in only
    where that copy is cheap against rebuilding the value.
    """
    key = tuple(identity)
    proved = _FULL_PROOF.get()
    if proved is not None:
        return _proved_request_value(proved, key, builder, nbytes, copy_mutable=copy_mutable)
    scope = _VERIFIED_ARRAY_SCOPE.get()
    if scope is not None and key in scope:
        value, _ = scope[key]
        scope.move_to_end(key)
        return cast(T, _handed(value))

    if scope is not None:
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
            return cast(T, _handed(value))

    value = builder()
    size = nbytes(value) if callable(nbytes) else nbytes
    if size < 0:
        raise ValueError("content_store.verified_value_size_invalid")
    if scope is None:
        return value
    if _immutable_request_value(value):
        kept: object = value
    elif copy_mutable:
        kept = _Copied(value)
    else:
        return value
    if size <= _VERIFIED_ARRAY_CACHE_LIMIT_BYTES:
        _remember_request_value(scope, key, kept, size)
    if not _VERIFIED_SOURCE_BUILD.get() and size <= _VERIFIED_ARRAY_PROCESS_CACHE_LIMIT_BYTES:
        _remember_process_value(key, kept, size)
    return cast(T, _handed(kept))


def _proved_request_value[T](
    proved: dict[tuple[object, ...], object],
    key: tuple[object, ...],
    builder: Callable[[], T],
    nbytes: int | Callable[[T], int],
    *,
    copy_mutable: bool,
) -> T:
    """One full proof's value: built once inside it, never answered from what was kept before."""
    if ("request-value", key) in proved:
        return cast(T, _handed(proved["request-value", key]))
    value = builder()
    size = nbytes(value) if callable(nbytes) else nbytes
    if size < 0:
        raise ValueError("content_store.verified_value_size_invalid")
    if size > _VERIFIED_ARRAY_CACHE_LIMIT_BYTES:
        return value
    if _immutable_request_value(value):
        proved["request-value", key] = value
    elif copy_mutable:
        proved["request-value", key] = _Copied(value)
    else:
        return value
    return cast(T, _handed(proved["request-value", key]))


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


def _remember_process_value(key: _VERIFIED_ARRAY_KEY, value: object, size: int) -> None:
    global _VERIFIED_ARRAY_CACHE_BYTES, _VERIFIED_ARRAY_PROCESS_CACHE_EVICTIONS
    with _VERIFIED_ARRAY_CACHE_LOCK:
        previous = _VERIFIED_ARRAY_CACHE.pop(key, None)
        if previous is not None:
            _VERIFIED_ARRAY_CACHE_BYTES -= previous[1]
        while (
            _VERIFIED_ARRAY_CACHE
            and _VERIFIED_ARRAY_CACHE_BYTES + size > _VERIFIED_ARRAY_PROCESS_CACHE_LIMIT_BYTES
        ):
            _evicted, (_, evicted_size) = _VERIFIED_ARRAY_CACHE.popitem(last=False)
            _VERIFIED_ARRAY_CACHE_BYTES -= evicted_size
            _VERIFIED_ARRAY_PROCESS_CACHE_EVICTIONS += 1
        if size <= _VERIFIED_ARRAY_PROCESS_CACHE_LIMIT_BYTES:
            _VERIFIED_ARRAY_CACHE[key] = (value, size)
            _VERIFIED_ARRAY_CACHE_BYTES += size


def verified_npz_arrays(
    payload: bytes, *, identity: tuple[object, ...]
) -> dict[str, npt.NDArray[Any]]:
    """Decode a sealed NPZ once per exact identity, in the request and the process memo.

    ``identity`` names the owner namespace, the artifact's content hash and its resolved
    path. This grants no current-pointer authority; it only avoids repeating NPZ inflation.

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
def verified_model_read_scope() -> Iterator[None]:
    """Share validated immutable JSON models during one request; nested scopes join it.

    A bounded process cache answers outside full proof as well, since a sealed model's bytes
    never change under its name.
    """
    with verified_array_read_scope():
        if _VERIFIED_MODEL_SCOPE.get() is not None:
            yield
            return
        scope: dict[_VERIFIED_MODEL_KEY, BaseModel] = {}
        scope_token = _VERIFIED_MODEL_SCOPE.set(scope)
        try:
            yield
        finally:
            scope.clear()
            _VERIFIED_MODEL_SCOPE.reset(scope_token)


def _file_identity(path: Path) -> tuple[object, ...] | None:
    """The file now at ``path`` for its lifetime, or None when none can be named there.

    A kept value is bound to it, so a file gone from its path, or another file put there, is read
    again rather than answered from memory: a test's numbered tmp_path is reused once pytest
    removes a passing test's folder, and a restored workspace sits at its old path. A sealed file
    rewritten in place is still the same file: an ordinary read keeps the bytes proved when they
    entered, and full proof refuses the change (EV2).

    Linux gives a freed inode number to the next new file at once, so there the file is named by
    its file handle, the inode number with its generation (name_to_handle_at). Elsewhere the file
    id, which NTFS and APFS do not give again at once, is bound to the file's creation time. A
    file system that names no handle (9p, an overlay without one) keeps nothing."""
    try:
        if sys.platform == "linux":
            return _file_handle(path)
        stat = os.stat(path)
        return (stat.st_dev, stat.st_ino, stat.st_birthtime)
    except OSError:
        return None


class _FileHandle(ctypes.Structure):
    """Linux's struct file_handle, with room for the largest handle (MAX_HANDLE_SZ)."""

    _fields_ = [
        ("handle_bytes", ctypes.c_uint),
        ("handle_type", ctypes.c_int),
        ("f_handle", ctypes.c_ubyte * 128),
    ]


_LIBC = ctypes.CDLL(None, use_errno=True) if sys.platform == "linux" else None


def _file_handle(path: Path) -> tuple[object, ...] | None:
    """The mount and file handle of the file at ``path``, following a link as os.stat does."""
    handle, mount = _FileHandle(handle_bytes=128), ctypes.c_int()
    named = getattr(_LIBC, "name_to_handle_at", None)
    at_cwd, follow = -100, 0x400  # AT_FDCWD, AT_SYMLINK_FOLLOW
    pointers = (ctypes.byref(handle), ctypes.byref(mount))
    if named is None or named(at_cwd, os.fsencode(path), *pointers, follow):
        return None
    return (mount.value, handle.handle_type, bytes(handle.f_handle[: handle.handle_bytes]))


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


def _verified_model_get(key: _VERIFIED_MODEL_KEY) -> BaseModel | None:
    scope = _VERIFIED_MODEL_SCOPE.get()
    if scope is None or _FULL_PROOF.get() is not None:
        return None
    value = scope.get(key)
    if value is not None:
        return value
    with _VERIFIED_MODEL_CACHE_LOCK:
        value = _VERIFIED_MODEL_CACHE.get(key)
        if value is not None:
            _VERIFIED_MODEL_CACHE.move_to_end(key)
    if value is not None:
        scope[key] = value
    return value


def _verified_model_put(
    scope: dict[_VERIFIED_MODEL_KEY, BaseModel],
    key: _VERIFIED_MODEL_KEY,
    value: BaseModel,
) -> None:
    scope[key] = value
    with _VERIFIED_MODEL_CACHE_LOCK:
        _VERIFIED_MODEL_CACHE[key] = value
        _VERIFIED_MODEL_CACHE.move_to_end(key)
        while len(_VERIFIED_MODEL_CACHE) > _VERIFIED_MODEL_CACHE_LIMIT:
            _VERIFIED_MODEL_CACHE.popitem(last=False)


def replace_shared_file(staged: Path, destination: Path) -> None:
    """Replace a file other requests may hold open to read it, despite their brief locks.

    Windows refuses to replace a file another handle holds open, and a Host's concurrent reads
    hold one while they read: the replace waits out short backoffs before it gives up.
    Elsewhere the first replace stands.
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
    lane keeps its identity across the type change.
    """
    with span("hash", "columns"):
        digest = integrity_sha256()
        for values in columns.values():
            digest.update(np.ascontiguousarray(values).tobytes())
        return digest.hexdigest()


_INSTALLED_COLUMNS_DIGEST = columns_digest


def _column_fact_kind(codes: tuple[CodeType, ...], toolchain: tuple[object, ...]) -> str:
    """The actual loaded proof code and toolchain, included in a reusable column fact."""

    def normalized(code: CodeType) -> CodeType:
        constants = tuple(
            normalized(value) if isinstance(value, CodeType) else value for value in code.co_consts
        )
        if constants and isinstance(constants[0], str) and not code.co_name.startswith("<"):
            constants = (None, *constants[1:])  # docstrings do not decide a proof
        return code.replace(co_consts=constants, co_filename="", co_firstlineno=1, co_linetable=b"")

    payload = [
        toolchain,
        [base64.b64encode(marshal.dumps(normalized(code))).decode("ascii") for code in codes],
    ]
    return "columns-v2:" + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _column_kind(store: ContentAddressedStore) -> str | None:
    if (
        columns_digest is not _INSTALLED_COLUMNS_DIGEST
        or type(store) is not ContentAddressedStore
        or (
            getattr(store._columns, "__func__", None) is not _INSTALLED_COLUMN_READER
            or getattr(store._stable_columns, "__func__", None)
            is not _INSTALLED_STABLE_COLUMN_READER
        )
    ):
        return None  # an overridden reader has no declared producer closure
    functions = (
        columns_digest,
        store._columns,
        store._stable_columns,
        integrity_sha256_path,
    )
    if any(
        getattr(getattr(function, "__func__", function), "__closure__", None)
        for function in functions
    ):
        return None  # captured runtime state is not described by the code object
    codes = tuple(
        getattr(getattr(function, "__func__", function), "__code__", None) for function in functions
    )
    if not all(isinstance(code, CodeType) for code in codes):
        return None  # an unknown producer cannot reuse a held proof
    environment = recorded_environment(("numpy", "pyarrow"))
    toolchain = (
        environment["python"],
        environment["implementation"],
        environment["platform"],
        environment["architecture"],
        environment["byte_order"],
        tuple(sorted(cast(dict[str, str], environment["packages"]).items())),
    )
    return _column_fact_kind(cast(tuple[CodeType, ...], codes), toolchain)


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
            staged = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
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
        fact = _file_identity(target) if scope is not None else None
        cacheable = fact is not None and model.model_config.get("frozen")

        key = (os.fspath(target.absolute()), fact, content_hash, model, identity_field)

        def verify() -> ContractT:
            if cacheable:
                kept = _verified_model_get(key)
                if kept is not None:
                    return cast(ContractT, kept)
            try:
                with span("read", "model"):
                    payload = target.read_bytes()
            except FileNotFoundError as error:
                raise ContentAddressedStoreError(
                    f"content_store.artifact_missing:{content_hash}"
                ) from error
            except Exception as error:
                raise ContentAddressedStoreError("content_store.artifact_tampered") from error
            try:
                with span("serialize", "model_validate"):
                    value = model.model_validate_json(payload)
            except Exception as error:
                raise ContentAddressedStoreError("content_store.artifact_tampered") from error
            if getattr(value, identity_field) != content_hash:
                raise ContentAddressedStoreError("content_store.artifact_tampered")
            if cacheable and _deeply_immutable_model(value):
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
        kind = _column_kind(self)
        if target.is_file():
            if kind is None or file_fact(target, kind) != content_hash:
                stat = os.stat(target)
                existing, content = self._stable_columns(target)
                if b"".join(v.tobytes() for v in existing.values()) != b"".join(
                    array.tobytes() for array in arrays.values()
                ):
                    raise ContentAddressedStoreError("content_store.identity_reused")
                if kind is not None:
                    record_file_fact(target, kind, content_hash, stat=stat, _content=content)
            return content_hash
        with span("serialize", "columns"):
            table = pa.table({name: array.reshape(-1) for name, array in arrays.items()})
            table = table.replace_schema_metadata({"shape": json.dumps(list(shapes.pop()))})
            sink = pa.BufferOutputStream()
            pq.write_table(table, sink, use_dictionary=False)
        payload = sink.getvalue().to_pybytes()
        self.atomic_write(target, payload)
        if kind is not None:
            record_file_fact(
                target,
                kind,
                content_hash,
                written=True,
                _content=integrity_sha256(payload).hexdigest(),
            )
        return content_hash

    def holds_columns(self, *, category: str, content_hash: str) -> bool:
        """Whether this store holds the Parquet table a digest names."""
        self.require_hash(content_hash)
        return (self.root / category / f"{content_hash}.parquet").is_file()

    def load_columns(self, *, category: str, content_hash: str) -> dict[str, npt.NDArray[Any]]:
        """Read one table's columns, flat; in full proof, prove its digest names them."""
        self.require_hash(content_hash)
        target = self.root / category / f"{content_hash}.parquet"
        if _FULL_PROOF.get() is None:
            return self._columns(target)
        kind = _column_kind(self)
        stat = os.stat(target) if target.is_file() else None
        columns, content = self._stable_columns(target)
        if kind is None or file_fact(target, kind, _content=content) != content_hash:
            if columns_digest(columns) != content_hash:
                raise ContentAddressedStoreError("content_store.artifact_tampered")
            if stat is not None and kind is not None:
                record_file_fact(target, kind, content_hash, stat=stat, _content=content)
        return columns

    def load_packed_bytes(self, *, category: str, content_hash: str) -> bytes:
        """The bytes a lane's digest names: its columns' C-order bytes, in order.

        A legacy store holds them as a ``.bin`` file, read as it is.
        """
        self.require_hash(content_hash)
        target = self.root / category / f"{content_hash}.parquet"
        if _FULL_PROOF.get() is None:
            if not target.is_file():
                return self._legacy(category, content_hash)
            return b"".join(values.tobytes() for values in self._columns(target).values())
        kind = _column_kind(self)
        if not target.is_file():
            payload, stat = self._legacy(category, content_hash), None
        else:
            stat = os.stat(target)
            columns, content = self._stable_columns(target)
            payload = b"".join(values.tobytes() for values in columns.values())
            if kind is not None and file_fact(target, kind, _content=content) == content_hash:
                return payload
        with span("hash", "packed"):
            if integrity_sha256(payload).hexdigest() != content_hash:
                raise ContentAddressedStoreError("content_store.artifact_tampered")
        if stat is not None and kind is not None:
            record_file_fact(target, kind, content_hash, stat=stat, _content=content)
        return payload

    def publish_document(self, *, category: str, payload: bytes, extension: str) -> str:
        """Write one JSON document or rendered page, named by the digest of its bytes."""
        if extension not in {"json", "html"}:
            raise ContentAddressedStoreError("content_store.document_type_invalid")
        content_hash = integrity_sha256(payload).hexdigest()
        target = self.root / category / f"{content_hash}.{extension}"
        if target.is_file() and target.read_bytes() != payload:
            raise ContentAddressedStoreError("content_store.identity_reused")
        if not target.is_file():
            self.atomic_write(target, payload)
        return content_hash

    def load_document(self, *, category: str, content_hash: str, extension: str) -> bytes:
        """Read one document; in full proof, prove it is the one asked for.

        A legacy store's is ``.bin``.
        """
        self.require_hash(content_hash)
        target = self.root / category / f"{content_hash}.{extension}"
        try:
            with span("read", "document"):
                payload = target.read_bytes()
        except FileNotFoundError:
            payload = self._legacy(category, content_hash)
        if _FULL_PROOF.get() is None:
            return payload
        with span("hash", "document"):
            if integrity_sha256(payload).hexdigest() != content_hash:
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

    def _stable_columns(self, target: Path) -> tuple[dict[str, npt.NDArray[Any]], str]:
        """Bind a decoded table to its bytes without keeping a changed read as a proof."""
        try:
            before = integrity_sha256_path(target)
            columns = self._columns(target)
            if integrity_sha256_path(target) != before:
                raise ContentAddressedStoreError("content_store.artifact_tampered")
            return columns, before
        except FileNotFoundError as error:
            raise ContentAddressedStoreError(
                f"content_store.artifact_missing:{target.stem}"
            ) from error
        except OSError as error:
            raise ContentAddressedStoreError("content_store.artifact_tampered") from error

    def _legacy(self, category: str, content_hash: str) -> bytes:
        # A store written in a legacy format packed its lanes, documents and pages as `.bin` bytes;
        # read only, until the formats registry names their upgrader.
        try:
            return (self.root / category / f"{content_hash}.bin").read_bytes()
        except FileNotFoundError as error:
            raise ContentAddressedStoreError(
                f"content_store.artifact_missing:{content_hash}"
            ) from error


_INSTALLED_COLUMN_READER = ContentAddressedStore._columns
_INSTALLED_STABLE_COLUMN_READER = ContentAddressedStore._stable_columns


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
    "full_content_proof",
    "full_proof",
    "verified_array_read_scope",
    "verified_model_read_scope",
    "verified_npz_arrays",
    "verified_request_proof",
    "verified_request_value",
    "verified_source_value",
    "verify_source_checks",
]
