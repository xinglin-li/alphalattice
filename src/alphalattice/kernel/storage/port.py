"""The one port every DuckDB open of this process goes through.

- **The driver** is injected (`install_driver`): the product's (control.workspace_runtime's
  reader_threads) installs itself when it is imported, and each process's entry installs it (the
  CLI and Host entry, each Task worker). The kernel imports only the standard library and opens
  nothing itself: an open with no driver installed is refused by name.
- **Every instance open is counted** by its target (`opens`): a data root's file or ``:memory:``;
  another connection to an instance the process holds open is opened inside `attaching` and not
  counted. A
  workspace file's owner keeps one instance of the file per process for as long as its session
  lives, and threads take cursors from it (control.workspace_runtime.database); this port is
  where that instance, and every other, is opened.
- **The in-memory instance** (`memory`): one per process, opened at its first use, whose cursors
  serve the short in-memory computations that each opened an instance of their own. A cursor is
  its own connection: what one registers, another does not see.
- **Table schemas** (`schema`): read once per data root and table, and kept until the root's
  writer revision changes; the root's owner names the revision.
- **Opens outside the engine** (`outside`): a reader of a file's bytes, or its replacement, names
  the file first, and the owner holding an instance of it (`on_outside`) lets go of what would
  refuse that access or keep the replaced file's old contents, until the block ends.
- **Teardown** (`on_teardown`, `close`): each owner's own, when the process ends or a caller
  closes the sessions, then the in-memory instance.
"""

from __future__ import annotations

import atexit
import threading
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from os import PathLike
from typing import Any, Protocol, cast

MEMORY = ":memory:"
"""The in-memory target, counted as one target however many instances it names."""


class StoragePort(Protocol):
    """The kernel port for persistent storage sessions and objects."""


class Driver(Protocol):
    """What opens an engine instance: DuckDB's `connect`, with the product's settings."""

    def connect(self, target: str, *, read_only: bool, config: Mapping[str, object]) -> Any:
        """Open `target` (a file path or ``:memory:``) in the access mode with `config`."""


_LOCK = threading.Lock()
_DRIVER: list[Driver] = []
_OPENS: Counter[str] = Counter()
_MEMORY: list[Any] = []
"""The process's in-memory instance's root connection, once opened."""
_SCHEMAS: dict[tuple[str, str | tuple[str, str]], tuple[int, object]] = {}
"""Per data root and table: the writer revision a schema was read at, and the schema."""
_TEARDOWNS: list[Callable[[], None]] = []
_OUTSIDE: list[Callable[[tuple[PathLike[str], ...], bool], Callable[[], None]]] = []
"""The owners that hold instances of files: each lets go of what an access outside the engine
needs, and answers the end it calls when the access is over."""
_ATTACHING: ContextVar[bool] = ContextVar("storage_attaching", default=False)
"""Inside `attaching`: an open is another connection to an instance already open."""


class DriverMissing(RuntimeError):
    """An open in a process whose entry installed no driver."""


def install_driver(driver: Driver | None) -> Driver | None:
    """Serve every later open through `driver` (None: no driver, so the next open is refused);
    the driver it replaces comes back."""
    with _LOCK:
        previous = _DRIVER[0] if _DRIVER else None
        _DRIVER[:] = [] if driver is None else [driver]
    return previous


def _driver() -> Driver:
    with _LOCK:
        if _DRIVER:
            return _DRIVER[0]
    raise DriverMissing(
        "storage.driver_missing: no DuckDB driver is installed in this process; its entry "
        "installs the product's (alphalattice.control.workspace_runtime.reader_threads.DRIVER)"
    )


def connect(
    target: str, *, read_only: bool = False, config: Mapping[str, object] | None = None
) -> Any:
    """Open an engine instance of `target` through the driver, counted by its target unless it
    is another connection to an instance already open (`attaching`)."""
    driver = _driver()
    if not _ATTACHING.get():
        with _LOCK:
            _OPENS[target] += 1
    return driver.connect(target, read_only=read_only, config=dict(config or {}))


@contextmanager
def attaching() -> Iterator[None]:
    """Opens in this block are further connections to an instance this process holds open: the
    engine attaches them to it, and they are not counted as instances."""
    token = _ATTACHING.set(True)
    try:
        yield
    finally:
        _ATTACHING.reset(token)


def opens() -> dict[str, int]:
    """The instances this process has opened so far, by target."""
    with _LOCK:
        return dict(_OPENS)


@contextmanager
def memory(config: Mapping[str, object] | None = None) -> Iterator[Any]:
    """A cursor of the process's in-memory instance, closed when the block ends. The instance is
    opened at the first use with `config`, which every later use shares."""
    with _LOCK:
        root = _MEMORY[0] if _MEMORY else None
    if root is None:
        opened = connect(MEMORY, config=config)
        with _LOCK:
            if not _MEMORY:
                _MEMORY.append(opened)
                opened = None
            root = _MEMORY[0]
        if opened is not None:  # another thread opened it first
            opened.close()
    with _LOCK:
        cursor = root.cursor()
    try:
        yield cursor
    finally:
        cursor.close()


def first(result: Any) -> Any:
    """A read's first row, or None, with its result finished.

    DuckDB holds an execute() result open until a fetch returns no rows, and the connection then
    refuses its next sql(), table() or register() (ResourceInUseException); `fetchone` on a row
    leaves it open. Reading every row ends it, so this is for reads of at most a few rows."""
    rows = result.fetchall()
    return rows[0] if rows else None


def schema[T](
    root: str,
    table: str | tuple[str, str],
    revision: int,
    read: Callable[[], T],
    *,
    installed_revision: Callable[[], int] | None = None,
    recheck: Callable[[T], T] | None = None,
) -> T:
    """Keep the schema at the revision its owner names.

    An installer uses the committed revision after its writes and can recheck the last
    installed shape after an unrelated DDL. Failed checks leave no new entry.
    """
    key = (root, table)
    with _LOCK:
        held = _SCHEMAS.get(key)
    if held is not None and held[0] == revision:
        return cast(T, held[1])
    value = recheck(cast(T, held[1])) if held is not None and recheck is not None else read()
    if installed_revision is not None:
        revision = installed_revision()
    with _LOCK:
        _SCHEMAS[key] = (revision, value)
    return value


def on_outside(
    callback: Callable[[tuple[PathLike[str], ...], bool], Callable[[], None]],
) -> None:
    """Call `callback(paths, replacing)` before each access to `paths` outside the engine
    (`outside`), each owner's once; the end it returns runs when the access is over."""
    with _LOCK:
        if callback not in _OUTSIDE:
            _OUTSIDE.append(callback)


@contextmanager
def outside(*paths: PathLike[str], replacing: bool = False) -> Iterator[None]:
    """A block that reads `paths`' bytes, or replaces them (`replacing`), outside the engine:
    each owner holding an instance of one lets go of what would refuse the access, or would keep
    serving a replaced file's old contents, and keeps none of it until the block ends."""
    with _LOCK:
        owners = list(_OUTSIDE)
    with ExitStack() as ends:
        for owner in owners:
            ends.callback(owner(paths, replacing))
        yield


def on_teardown(callback: Callable[[], None]) -> None:
    """Run `callback` when the sessions close (`close`), each owner's once."""
    with _LOCK:
        if callback not in _TEARDOWNS:
            _TEARDOWNS.append(callback)


def close() -> None:
    """Close the sessions: each owner's teardown, then the in-memory instance. Every teardown
    runs; the first failure is raised after them."""
    with _LOCK:
        teardowns = list(_TEARDOWNS)
        root = _MEMORY.pop() if _MEMORY else None
        _SCHEMAS.clear()
    failure: BaseException | None = None
    for teardown in teardowns:
        try:
            teardown()
        except BaseException as error:  # every owner's teardown runs; the first is raised
            failure = failure or error
    if root is not None:
        root.close()
    if failure is not None:
        raise failure


atexit.register(close)
