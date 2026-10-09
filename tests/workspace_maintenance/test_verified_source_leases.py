"""BEHAVIOUR: unchanged sealed reads reuse proofs; OS changes always reverify (V691)."""

from __future__ import annotations

import ctypes
import hashlib
import mmap
import os
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from io import BytesIO
from pathlib import Path
from queue import Queue
from threading import BoundedSemaphore, Event, Lock, Thread, current_thread
from typing import ClassVar

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, model_validator

from alphalattice.control.workspace_runtime.content_store import (
    ContentAddressedStore,
    verified_array_read_scope,
    verified_model_read_scope,
    verified_npz_arrays,
    verified_request_proof,
    verified_request_value,
    verified_source_value,
    verify_source_checks,
)

if os.name == "nt":
    import msvcrt
    from ctypes import wintypes


def _reader(path: Path, payload: bytes = b"sealed"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    calls = []

    def verify():
        calls.append(path)
        current = path.read_bytes()
        if hashlib.sha256(current).hexdigest() != expected:
            raise ValueError("sealed bytes changed")
        return current

    def read(*, reuse=True):
        with verified_model_read_scope(reuse_verified=reuse):
            return verified_source_value(
                ("test-sealed-read", expected), (path,), verify, nbytes=lambda value: len(value)
            )

    return read, calls


def test_unchanged_sources_verify_once_and_disabled_reuse_reads_again(tmp_path):
    read, calls = _reader(tmp_path / "sealed.bin")
    first = read()
    second = read()
    assert first == second == b"sealed"
    assert len(calls) == (1 if os.name == "nt" else 2)
    if os.name == "nt":
        assert first is second
    read(reuse=False)
    assert len(calls) == (2 if os.name == "nt" else 3)


@contextmanager
def _concurrent_source_reads(read, entered, release, monkeypatch):
    waiting = Queue()
    result = Future.result

    def wait(future, *args, **kwargs):
        if current_thread().name.startswith("source-waiter-"):
            waiting.put(None)
        return result(future, *args, **kwargs)

    monkeypatch.setattr(Future, "result", wait)
    outcomes = []

    def request():
        try:
            outcomes.append((read(), None))
        except BaseException as error:
            outcomes.append((None, error))

    workers = [Thread(target=request, name="source-producer")]
    try:
        workers[0].start()
        assert entered.wait(timeout=5)
        for index in range(3):
            worker = Thread(target=request, name=f"source-waiter-{index}")
            workers.append(worker)
            worker.start()
        for _ in range(3):
            waiting.get(timeout=5)
        yield outcomes
    finally:
        release.set()
        for worker in workers:
            worker.join(timeout=5)
            assert not worker.is_alive()


@pytest.mark.skipif(os.name != "nt", reason="Windows read leases grant single-flight reuse")
@pytest.mark.parametrize("signal", ("available", "unavailable"))
def test_concurrent_source_reads_share_only_valid_lease_proof_without_blocking_other_keys(
    tmp_path, monkeypatch, signal
):
    path = tmp_path / "sealed.bin"
    path.write_bytes(b"sealed")
    entered, release = Event(), Event()
    calls = []
    if signal == "unavailable":

        def unavailable(*args, **kwargs):
            raise OSError("OS signal unavailable")

        monkeypatch.setattr(ctypes, "WinDLL", unavailable)

    def verify():
        calls.append(True)
        payload = path.read_bytes()
        assert payload == b"sealed"
        entered.set()
        assert release.wait(timeout=5)
        return payload

    def read():
        with verified_model_read_scope(reuse_verified=True):
            return verified_source_value(("single-flight",), (path,), verify, nbytes=6)

    other, other_calls = _reader(tmp_path / "independent.bin")
    with _concurrent_source_reads(read, entered, release, monkeypatch) as outcomes:
        independent = Thread(target=other, name="independent-source")
        independent.start()
        independent.join(timeout=5)
        assert not independent.is_alive()
        assert len(other_calls) == 1
        assert len(calls) == 1
    assert len(outcomes) == 4
    assert all(value == b"sealed" and error is None for value, error in outcomes)
    assert len(calls) == (1 if signal == "available" else 4)
    assert read() == b"sealed"
    assert len(calls) == (1 if signal == "available" else 5)


@pytest.mark.skipif(os.name != "nt", reason="Windows read leases grant single-flight reuse")
@pytest.mark.parametrize("fault", ("builder", "invalid-size"))
def test_concurrent_producer_failure_releases_waiters_and_later_reads_verify_again(
    tmp_path, monkeypatch, fault
):
    path = tmp_path / "sealed.bin"
    path.write_bytes(b"sealed")
    entered, release = Event(), Event()
    calls = []
    failing = True

    def verify():
        calls.append(True)
        entered.set()
        assert release.wait(timeout=5)
        if failing and fault == "builder":
            raise ValueError("sealed verification failed")
        return path.read_bytes()

    def read():
        with verified_model_read_scope(reuse_verified=True):
            return verified_source_value(
                ("single-flight-failure",),
                (path,),
                verify,
                nbytes=lambda value: -1 if failing and fault == "invalid-size" else len(value),
            )

    with _concurrent_source_reads(read, entered, release, monkeypatch) as outcomes:
        assert len(calls) == 1
    assert len(calls) == 1
    assert len(outcomes) == 4
    assert all(value is None and isinstance(error, ValueError) for value, error in outcomes)
    failing = False
    assert read() == b"sealed"
    assert len(calls) == 2


@pytest.mark.skipif(os.name != "nt", reason="Windows read leases grant single-flight reuse")
def test_a_waiter_rechecks_the_lease_after_its_producer_finishes(tmp_path, monkeypatch):
    path = tmp_path / "sealed.bin"
    path.write_bytes(b"sealed")
    entered, release, waiting, resumed, consume = (Event() for _ in range(5))
    result = Future.result
    calls = []
    outcomes = []

    def wait(future, *args, **kwargs):
        if current_thread().name == "source-waiter":
            waiting.set()
            value = result(future, *args, **kwargs)
            resumed.set()
            assert consume.wait(timeout=5)
            return value
        return result(future, *args, **kwargs)

    monkeypatch.setattr(Future, "result", wait)

    def verify():
        calls.append(True)
        payload = path.read_bytes()
        if payload != b"sealed":
            raise ValueError("sealed bytes changed")
        entered.set()
        assert release.wait(timeout=5)
        return payload

    def read():
        try:
            with verified_model_read_scope(reuse_verified=True):
                value = verified_source_value(("delayed-waiter",), (path,), verify, nbytes=6)
            outcomes.append((value, None))
        except BaseException as error:
            outcomes.append((None, error))

    producer = Thread(target=read, name="source-producer")
    waiter = Thread(target=read, name="source-waiter")
    try:
        producer.start()
        assert entered.wait(timeout=5)
        waiter.start()
        assert waiting.wait(timeout=5)
        release.set()
        producer.join(timeout=5)
        assert not producer.is_alive()
        assert resumed.wait(timeout=5)
        assert outcomes == [(b"sealed", None)]
        before = path.stat()
        path.write_bytes(b"broken")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    finally:
        release.set()
        consume.set()
        producer.join(timeout=5)
        if waiter.ident is not None:
            waiter.join(timeout=5)
    assert not waiter.is_alive()
    assert len(calls) == 2
    assert len(outcomes) == 2
    assert outcomes[1][0] is None
    assert isinstance(outcomes[1][1], ValueError)
    assert str(outcomes[1][1]) == "sealed bytes changed"


def test_a_nested_read_of_the_same_source_key_does_not_wait_for_itself(tmp_path):
    path = tmp_path / "sealed.bin"
    path.write_bytes(b"sealed")
    calls = []

    def verify():
        calls.append(True)
        if len(calls) == 1:
            return verified_source_value(("nested-source",), (path,), verify, nbytes=6)
        return path.read_bytes()

    with verified_model_read_scope(reuse_verified=True):
        assert verified_source_value(("nested-source",), (path,), verify, nbytes=6) == b"sealed"
    assert len(calls) == 2


def test_parallel_checks_have_fresh_contexts_and_isolated_request_memos(tmp_path):
    marker = ContextVar("outside-check-context", default="unset")
    marker.set("caller")
    values, proofs = [], []
    identity = ("isolated-check-scopes", str(tmp_path))

    def value():
        values.append(True)
        return b"sealed"

    def check():
        assert marker.get() == "unset"
        for _ in range(2):
            assert verified_request_value(identity, value, nbytes=6) == b"sealed"
            verified_request_proof(identity, lambda: proofs.append(True))

    with verified_model_read_scope():
        verified_request_value(identity, value, nbytes=6)
        verified_request_proof(identity, lambda: proofs.append(True))
        verify_source_checks((check, check))
        verified_request_value(identity, value, nbytes=6)
        verified_request_proof(identity, lambda: proofs.append(True))
    assert len(values) == len(proofs) == 3
    assert marker.get() == "caller"


def test_parallel_checks_bound_running_and_submitted_work_across_requests(monkeypatch):
    release, entered = Event(), Queue()
    attempts = Queue()
    lock = Lock()
    acquire, submit = BoundedSemaphore.acquire, ThreadPoolExecutor.submit
    outstanding = peak = 0
    errors = []

    def acquire_slot(semaphore, *args, **kwargs):
        if current_thread().name.startswith("source-check-caller-"):
            attempts.put(None)
        return acquire(semaphore, *args, **kwargs)

    def submitted(pool, *args, **kwargs):
        nonlocal outstanding, peak
        with lock:
            outstanding += 1
            peak = max(peak, outstanding)
        future = submit(pool, *args, **kwargs)

        def completed(_future):
            nonlocal outstanding
            with lock:
                outstanding -= 1

        future.add_done_callback(completed)
        return future

    monkeypatch.setattr(BoundedSemaphore, "acquire", acquire_slot)
    monkeypatch.setattr(ThreadPoolExecutor, "submit", submitted)

    def check():
        entered.put(None)
        assert release.wait(timeout=5)

    def request():
        try:
            verify_source_checks((check, check))
        except BaseException as error:
            errors.append(error)

    callers = [Thread(target=request, name=f"source-check-caller-{i}") for i in range(2)]
    try:
        for caller in callers:
            caller.start()
        for _ in range(2):
            entered.get(timeout=5)
        for _ in range(3):
            attempts.get(timeout=5)
        with lock:
            assert outstanding == peak == 2
    finally:
        release.set()
        for caller in callers:
            caller.join(timeout=5)
            assert not caller.is_alive()
    assert not errors
    assert outstanding == 0
    assert peak == 2


def test_parallel_checks_join_failures_in_input_order_and_drain_started_checks():
    first_failed, second_started, release = Event(), Event(), Event()
    errors = []

    def first():
        assert second_started.wait(timeout=5)
        first_failed.set()
        raise ValueError("first input")

    def second():
        second_started.set()
        assert release.wait(timeout=5)
        raise ValueError("second input")

    def request():
        try:
            verify_source_checks((first, second))
        except BaseException as error:
            errors.append(error)

    caller = Thread(target=request)
    try:
        caller.start()
        assert first_failed.wait(timeout=5)
        assert caller.is_alive()
        assert not errors
    finally:
        release.set()
        caller.join(timeout=5)
    assert not caller.is_alive()
    assert len(errors) == 1 and str(errors[0]) == "first input"
    # Both submission permits were released despite both callbacks raising.
    verify_source_checks((lambda: None, lambda: None))


def test_nested_parallel_checks_finish_without_waiting_for_their_own_pool():
    started, release = Queue(), Event()
    finished = []

    def outer():
        started.put(None)
        assert release.wait(timeout=5)
        verify_source_checks((lambda: finished.append(True), lambda: finished.append(True)))

    caller = Thread(target=lambda: verify_source_checks((outer, outer)))
    try:
        caller.start()
        for _ in range(2):
            started.get(timeout=5)
    finally:
        release.set()
        caller.join(timeout=5)
    assert not caller.is_alive()
    assert len(finished) == 4


@pytest.mark.parametrize("reuse", (False, True))
def test_parallel_checks_inherit_only_the_outer_reuse_policy(tmp_path, reuse):
    read, calls = _reader(tmp_path / "sealed.bin")

    def check():
        assert read() == read() == b"sealed"

    with verified_model_read_scope(reuse_verified=reuse):
        verify_source_checks((check,))
    assert len(calls) == (1 if reuse and os.name == "nt" else 2)


def test_parallel_array_optin_does_not_widen_the_model_reuse_policy(tmp_path):
    class Document(BaseModel):
        model_config = ConfigDict(frozen=True)
        identity: str
        validations: ClassVar[int] = 0

        @model_validator(mode="after")
        def count_validation(self):
            type(self).validations += 1
            return self

    store = ContentAddressedStore(tmp_path, uri_prefix="artifact://qa")
    identity = "a" * 64
    store.publish_model(
        category="documents", value=Document(identity=identity), identity_field="identity"
    )
    Document.validations = 0
    path = tmp_path / "documents" / f"{identity}.json"

    def check():
        for _ in range(2):
            store.load_model(
                category="documents",
                content_hash=identity,
                model=Document,
                identity_field="identity",
            )

    def proof():
        # Separate callbacks keep their local model memo; array opt-in supplies no
        # permission to consult or publish the process model memo between them.
        verify_source_checks((check,))
        verify_source_checks((check,))

    with verified_array_read_scope(reuse_verified=True):
        verified_source_value(("array-only-model-policy",), (path,), proof, nbytes=0)
    assert Document.validations == 2


@pytest.mark.skipif(os.name != "nt", reason="Windows leases retain the complete source owner")
def test_parallel_checks_inherit_declared_source_ownership_without_nested_retention(tmp_path):
    path = tmp_path / "sealed.bin"
    read, calls = _reader(path)

    def check():
        assert read() == read() == b"sealed"

    def proof():
        verify_source_checks((check,))

    for _ in range(2):
        with verified_model_read_scope(reuse_verified=True):
            verified_source_value(("parallel-complete-proof",), (path,), proof, nbytes=0)
    assert len(calls) == 2
    assert read() == b"sealed"
    assert len(calls) == 3


def _read_in_request_thread(read):
    outcomes = []

    def request():
        try:
            outcomes.append((read(), None))
        except BaseException as error:
            outcomes.append((None, error))

    worker = Thread(target=request, name="lease-request")
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    value, error = outcomes[0]
    if error is not None:
        raise error
    return value


@pytest.mark.skipif(os.name != "nt", reason="Windows pending oplock IRPs need a live issuer")
def test_request_thread_exit_keeps_reuse_and_later_tampering_is_refused(tmp_path):
    path = tmp_path / "sealed.bin"
    read, calls = _reader(path)
    for _ in range(8):
        assert _read_in_request_thread(read) == b"sealed"
    assert len(calls) == 1  # every previous request thread has exited
    before = path.stat()
    path.write_bytes(b"broken")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="sealed bytes changed"):
        _read_in_request_thread(read)
    assert len(calls) == 2


@pytest.mark.skipif(os.name != "nt", reason="Windows pending oplock IRPs need a live issuer")
@pytest.mark.parametrize("worker", ("content-store-read-leases", "content-store-lease-breaks"))
def test_unavailable_process_issuer_forces_full_verification(tmp_path, monkeypatch, worker):
    path = tmp_path / "sealed.bin"
    read, calls = _reader(path)
    original_alive = Thread.is_alive

    def unavailable(thread):
        if thread.name == worker:
            return False
        return original_alive(thread)

    monkeypatch.setattr(Thread, "is_alive", unavailable)
    assert read() == read() == b"sealed"
    assert len(calls) == 2
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="sealed bytes changed"):
        read()
    assert len(calls) == 3


@pytest.mark.skipif(os.name != "nt", reason="Windows RH breaks must close before rename proceeds")
@pytest.mark.parametrize("stage", ("acquisition", "cached-access", "identity-check"))
def test_an_access_check_does_not_obstruct_an_ancestor_rename(tmp_path, monkeypatch, stage):
    """Access checks open files outside the lease lock so an ancestor rename can close the lease."""
    path = tmp_path / "source" / "sealed.bin"
    read, calls = _reader(path)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    original_create = kernel.CreateFileW
    entered, release = Event(), Event()
    outcomes = []
    acquisition_opens = 0

    blocked = []

    def create(*args):
        nonlocal acquisition_opens
        issuer = current_thread().name == "content-store-read-leases"
        reader = current_thread().name == "cached-source-reader"
        identity = args[1] == 0x80  # the path identity's attribute-only open
        if issuer and not identity:
            acquisition_opens += 1
        if not blocked and (
            (stage == "acquisition" and issuer and not identity and acquisition_opens == 2)
            or (stage == "cached-access" and reader and not identity)
            or (stage == "identity-check" and reader and identity)
        ):
            blocked.append(args[0])
            entered.set()
            # Longer than the rename's own timeout: a lease lock held across this open
            # fails the rename rather than racing it.
            assert release.wait(timeout=30)
        return original_create(*args)

    monkeypatch.setattr(kernel, "CreateFileW", create)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: kernel)
    if stage != "acquisition":
        assert read() == b"sealed"
        assert len(calls) == 1

    def cached_read():
        try:
            read()
        except Exception as error:
            outcomes.append(error)

    reader = Thread(target=cached_read, name="cached-source-reader")
    try:
        reader.start()
        assert entered.wait(timeout=5)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import os,sys; os.rename(sys.argv[1],sys.argv[2])",
                str(path.parent),
                str(tmp_path / "prior"),
            ],
            check=True,
            timeout=5,
        )
    finally:
        release.set()
        reader.join(timeout=5)
    assert not reader.is_alive()
    assert len(outcomes) == 1 and isinstance(outcomes[0], FileNotFoundError)
    assert len(calls) == (1 if stage == "acquisition" else 2)
    # Neither an interrupted acquisition nor a hit returns retained proof after rename.


@pytest.mark.skipif(os.name != "nt", reason="Windows read leases own retained source results")
def test_complete_source_owner_does_not_retain_nested_packed_bytes(tmp_path):
    path = tmp_path / "sealed.bin"
    read, calls = _reader(path)
    expected = hashlib.sha256(b"sealed").hexdigest()

    def proof():
        assert read() == b"sealed"

    for _ in range(2):
        with verified_model_read_scope(reuse_verified=True):
            verified_source_value(("complete-proof", expected), (path,), proof, nbytes=0)
    assert len(calls) == 1
    # The nested reader did full verification but retained no duplicate byte cache.
    read()
    assert len(calls) == 2


@pytest.mark.skipif(os.name != "nt", reason="Windows read leases own retained source results")
def test_complete_source_owner_retains_decoded_values_without_a_second_process_copy(
    tmp_path, monkeypatch
):
    stream = BytesIO()
    np.savez(stream, values=np.arange(8, dtype=np.float64))
    payload = stream.getvalue()
    path = tmp_path / "arrays.bin"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    load = np.load
    decodes = 0

    def counted_load(*args, **kwargs):
        nonlocal decodes
        decodes += 1
        return load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counted_load)

    def build():
        current = path.read_bytes()
        assert hashlib.sha256(current).hexdigest() == expected
        arrays = verified_npz_arrays(current, identity=(str(path), expected))
        return tuple(arrays.items())

    def read(key):
        with verified_model_read_scope(reuse_verified=True):
            return verified_source_value(
                ("complete-decoded-result", key, expected),
                (path,),
                build,
                nbytes=lambda values: sum(array.nbytes for _, array in values),
            )

    first = read("one")
    second = read("two")
    assert decodes == 2  # intermediate NPZ cache was not independently retained
    assert first is read("one")
    assert second is read("two")
    assert decodes == 2
    assert np.array_equal(first[0][1], np.arange(8))
    with pytest.raises(ValueError):
        first[0][1].setflags(write=True)


@pytest.mark.parametrize(
    "change",
    (
        "same-size-restored-time",
        "replacement",
        "atomic-replacement",
        "ancestor-rename",
        "rename",
        "recreate",
        "hard-link",
        "mapped-write",
        "suppressed-write-time",
    ),
)
def test_os_changes_reverify_and_refuse_even_with_restored_size_and_time(tmp_path, change):
    path = tmp_path / "source" / "sealed.bin"
    read, calls = _reader(path)
    read()
    before = path.stat()
    if change == "replacement":
        alternate = tmp_path / "replacement.bin"
        alternate.write_bytes(b"broken")
        if os.name == "nt":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.ReplaceFileW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.LPCWSTR,
                wintypes.LPCWSTR,
                wintypes.DWORD,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            kernel.ReplaceFileW.restype = wintypes.BOOL
            assert kernel.ReplaceFileW(str(path), str(alternate), None, 0, None, None)
        else:
            os.replace(alternate, path)
    elif change == "atomic-replacement":
        alternate = tmp_path / "replacement.bin"
        alternate.write_bytes(b"broken")
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import os,sys; os.replace(sys.argv[1],sys.argv[2])",
                str(alternate),
                str(path),
            ],
            check=True,
            timeout=5,
        )
    elif change == "ancestor-rename":
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import os,sys; os.rename(sys.argv[1],sys.argv[2])",
                str(path.parent),
                str(tmp_path / "prior"),
            ],
            check=True,
            timeout=5,
        )
        path.parent.mkdir()
        path.write_bytes(b"broken")
    elif change == "rename":
        path.rename(tmp_path / "prior.bin")
        path.write_bytes(b"broken")
    elif change == "recreate":
        path.unlink()
        path.write_bytes(b"broken")
    elif change == "hard-link":
        other_directory = tmp_path / "other"
        other_directory.mkdir()
        alias = other_directory / "alias.bin"
        os.link(path, alias)
        alias.write_bytes(b"broken")
    elif change == "mapped-write":
        with path.open("r+b") as stream, mmap.mmap(stream.fileno(), 0) as mapping:
            mapping[:] = b"broken"
            mapping.flush()
    elif change == "suppressed-write-time" and os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.SetFileTime.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        kernel.SetFileTime.restype = wintypes.BOOL
        with path.open("r+b") as stream:
            unchanged_time = wintypes.FILETIME(0xFFFFFFFF, 0xFFFFFFFF)
            assert kernel.SetFileTime(
                msvcrt.get_osfhandle(stream.fileno()), None, None, ctypes.byref(unchanged_time)
            )
            stream.write(b"broken")
        assert path.stat().st_mtime_ns == before.st_mtime_ns
    else:
        path.write_bytes(b"broken")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    for _ in range(2):
        with pytest.raises(ValueError, match="sealed bytes changed"):
            read()
    assert len(calls) == 3  # failures are never retained
    path.write_bytes(b"sealed")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert read() == b"sealed"
    assert len(calls) == 4


def _directory_link(link: Path, target: Path) -> None:
    """A junction on Windows (no privilege needed), a directory symlink elsewhere."""
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
            timeout=10,
        )
    else:
        link.symlink_to(target, target_is_directory=True)


def test_a_directory_link_repointed_after_the_first_read_reverifies_and_refuses(tmp_path):
    """regression: the identity check reads the path's resolution and file from one
    handle; a junction (a symlink elsewhere) repointed after the first verified read is a
    changed path even with the same size and time, so the read verifies again and refuses."""
    sealed, other, link = tmp_path / "sealed", tmp_path / "other", tmp_path / "link"
    sealed.mkdir()
    other.mkdir()
    _directory_link(link, sealed)
    read, calls = _reader(link / "sealed.bin")
    read()
    read()
    assert len(calls) == (1 if os.name == "nt" else 2)  # reused through the link while it holds
    before = (sealed / "sealed.bin").stat()
    (other / "sealed.bin").write_bytes(b"broken")
    os.utime(other / "sealed.bin", ns=(before.st_atime_ns, before.st_mtime_ns))
    os.rmdir(link) if os.name == "nt" else link.unlink()
    _directory_link(link, other)
    seen = len(calls)
    for _ in range(2):
        with pytest.raises(ValueError, match="sealed bytes changed"):
            read()
    assert len(calls) == seen + 2  # failures are never retained
    os.rmdir(link) if os.name == "nt" else link.unlink()
    _directory_link(link, sealed)
    assert read() == b"sealed"
    assert len(calls) == seen + 3


@pytest.mark.parametrize("fault", ("unavailable", "wait-failed", "signal-exception", "read-access"))
def test_missing_or_failed_os_signal_forces_full_verification(tmp_path, monkeypatch, fault):
    path = tmp_path / "sealed.bin"
    read, calls = _reader(path)
    if os.name != "nt":
        read()
        read()
        assert len(calls) == 2
        return

    if fault == "unavailable":

        def unavailable(*args, **kwargs):
            raise OSError("OS signal unavailable")

        monkeypatch.setattr(ctypes, "WinDLL", unavailable)
        read()
    else:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        original_create = kernel.CreateFileW

        original_create.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        original_create.restype = wintypes.HANDLE
        original_wait = kernel.WaitForSingleObject
        original_wait.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        original_wait.restype = ctypes.c_ulong
        fail = []

        def wait(*args):
            if fail and fault != "read-access":
                if fault == "signal-exception":
                    raise OSError("OS signal failed")
                return 0xFFFFFFFF
            return original_wait(*args)

        def create(*args):
            if fail and fault == "read-access":
                raise PermissionError("current read access denied")
            return original_create(*args)

        monkeypatch.setattr(kernel, "WaitForSingleObject", wait)
        monkeypatch.setattr(kernel, "CreateFileW", create)
        monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: kernel)
        read()
        assert len(calls) == 1
        fail.append(True)
    assert read() == b"sealed"
    assert len(calls) == 2
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="sealed bytes changed"):
        read()
    assert len(calls) == 3


def test_a_source_change_during_verification_is_reverified_before_return(tmp_path):
    path = tmp_path / "sealed.bin"
    path.write_bytes(b"sealed")
    calls = []

    def racing_verifier():
        calls.append(path.read_bytes())
        if len(calls) == 1:
            path.write_bytes(b"broken")
        if calls[-1] != b"sealed":
            raise ValueError("sealed bytes changed")
        return calls[-1]

    with verified_model_read_scope(reuse_verified=True):
        if os.name == "nt":
            with pytest.raises(ValueError, match="sealed bytes changed"):
                verified_source_value(("racing",), (path,), racing_verifier, nbytes=6)
            assert calls == [b"sealed", b"broken"]
        else:
            assert (
                verified_source_value(("racing",), (path,), racing_verifier, nbytes=6) == b"sealed"
            )
            with pytest.raises(ValueError, match="sealed bytes changed"):
                verified_source_value(("racing",), (path,), racing_verifier, nbytes=6)
            assert calls == [b"sealed", b"broken"]


def test_a_writable_mapping_open_before_verification_never_grants_reuse(tmp_path):
    path = tmp_path / "sealed.bin"
    read, calls = _reader(path)
    before = path.stat()
    with path.open("r+b") as stream, mmap.mmap(stream.fileno(), 0) as mapping:
        assert read() == b"sealed"
        assert read() == b"sealed"
        assert len(calls) == 2
        mapping[:] = b"broken"
        mapping.flush()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with pytest.raises(ValueError, match="sealed bytes changed"):
            read()
        assert len(calls) == 3


def test_every_source_in_a_proof_is_watched_and_mutable_results_are_not_retained(tmp_path):
    paths = tuple(tmp_path / f"source-{index}.bin" for index in range(2))
    for path in paths:
        path.write_bytes(b"sealed")
    calls = []

    def verify():
        calls.append(True)
        if any(path.read_bytes() != b"sealed" for path in paths):
            raise ValueError("sealed bytes changed")
        return b"sealed"

    for _ in range(2):
        with verified_model_read_scope(reuse_verified=True):
            verified_source_value(("two-source-proof",), paths, verify, nbytes=6)
    assert len(calls) == (1 if os.name == "nt" else 2)
    paths[1].write_bytes(b"broken")
    with verified_model_read_scope(reuse_verified=True), pytest.raises(ValueError):
        verified_source_value(("two-source-proof",), paths, verify, nbytes=6)
    mutable_calls = []
    for _ in range(2):
        with verified_model_read_scope(reuse_verified=True):
            result = verified_source_value(
                ("mutable-source",), (paths[0],), lambda: mutable_calls.append(True) or {}, nbytes=1
            )
            result["caller"] = "changed"
    assert len(mutable_calls) == 2


def test_source_leases_are_bounded_and_evicted_sources_verify_again(tmp_path):
    read, calls = _reader(tmp_path / "oldest.bin")
    read()
    readers = []
    for index in range(511):
        newer, newer_calls = _reader(tmp_path / f"source-{index}.bin")
        assert newer() == b"sealed"
        readers.append((newer, newer_calls))
    # All 512 leases fit; reading this source moves it to the newest LRU position.
    assert read() == b"sealed"
    assert len(calls) == (1 if os.name == "nt" else 2)
    extra, _ = _reader(tmp_path / "overflow.bin")
    extra()
    evicted, evicted_calls = readers[0]
    assert evicted() == b"sealed"
    assert len(evicted_calls) == 2  # the 513th lease evicted the least recently used source


def test_restart_verifies_bytes_changed_while_the_host_process_was_down(tmp_path):
    path = tmp_path / "sealed.bin"
    path.write_bytes(b"sealed")
    program = """
import hashlib
import sys
from pathlib import Path
from alphalattice.control.workspace_runtime.content_store import (
    verified_model_read_scope, verified_source_value,
)
path = Path(sys.argv[1])
calls = []
def verify():
    calls.append(True)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != hashlib.sha256(b'sealed').hexdigest():
        raise ValueError('sealed bytes changed')
    return payload
try:
    for _ in range(2):
        with verified_model_read_scope(reuse_verified=True):
            verified_source_value(('restart-proof',), (path,), verify, nbytes=6)
except ValueError as error:
    print(str(error))
    sys.exit(42)
print(len(calls))
"""
    first = subprocess.run(
        [sys.executable, "-c", program, str(path)], capture_output=True, text=True, timeout=30
    )
    assert first.returncode == 0, first.stderr
    assert first.stdout.strip() == ("1" if os.name == "nt" else "2")
    before = path.stat()
    path.write_bytes(b"broken")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    restarted = subprocess.run(
        [sys.executable, "-c", program, str(path)], capture_output=True, text=True, timeout=30
    )
    assert restarted.returncode == 42, restarted.stderr
    assert restarted.stdout.strip() == "sealed bytes changed"
