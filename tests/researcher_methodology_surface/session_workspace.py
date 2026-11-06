"""Product-written research prerequisites, once per pytest session across workers.

Only the builder sees the seed. Tests receive a private database and record copy;
read-only Portfolio consumers may link the seed's immutable Parquet objects.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import portalocker
import pytest

from devtools.evaluation.golden_workspaces import (
    GoldenInputs,
    GoldenPayloadCorrupt,
    hold_generation,
    payload_path,
    persistent_golden_root,
    refuse_writers,
    remove_idle_reader_lock,
    remove_tree,
    seal_payload,
    verified_payload,
)
from tests.pytest_environment import RUNNING_TEST, golden_runtime


def _files(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        with path.open("rb") as stream:
            result[path.relative_to(root).as_posix()] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return result


@contextmanager
def _closed_seed(workspace: Path) -> Iterator[None]:
    """The golden owner seals only after its Host and writer have stopped."""
    from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease

    if (workspace / "runtime/local-research-connection.json").exists():
        raise RuntimeError("session workspace still names a Host")
    lease = WorkspaceWriterLease.acquire(workspace)
    try:
        yield
    finally:
        lease.close()


def session_workspace(
    factory: pytest.TempPathFactory,
    name: str,
    build: Callable[[Path], dict[str, Any]],
    *,
    inputs: Any = None,
    persistent: GoldenInputs | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Publish immutable session generations; rebuild incomplete seeds and refuse corruption.

    Callers must declare varying builder inputs as JSON values in ``inputs``.
    The session-local name and builder identify the fixed part of the build.
    """
    if not name.isidentifier():
        raise ValueError("session workspace name must be an identifier")
    input_hash = hashlib.sha256(
        json.dumps(inputs, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    builder = f"{build.__module__}:{build.__qualname__}"
    key = hashlib.sha256(f"{builder}\0{input_hash}".encode()).hexdigest()
    if persistent is not None:
        key = persistent.key(
            {"name": name, "builder": builder, "inputs": input_hash, "format": 1}, []
        )
        root = Path(__file__).resolve().parents[2]
        shared = persistent_golden_root(root / "tmp/golden-workspaces")
        if not shared.is_absolute():
            raise ValueError("session workspace persistent root must be absolute")
        cache = payload_path(shared / "session-prerequisites")
    else:
        base = factory.getbasetemp()
        if os.environ.get("PYTEST_XDIST_WORKER"):
            base = base.parent
        cache = base / "research-prerequisites"
    if cache.is_symlink() or cache.is_junction():
        raise RuntimeError("session workspace cache is a link")
    cache.mkdir(parents=True, exist_ok=True)
    index_path = cache / f"{name}.json"
    started = time.perf_counter()
    runtime = golden_runtime()
    with runtime.lock(cache / f"{name}.lock"):
        if persistent is not None:
            persistent.verify()
        if index_path.is_symlink() or index_path.is_junction():
            raise RuntimeError(f"session workspace index is a link: {name}")
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            index = {}
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"session workspace index unreadable: {name}") from error
        if not isinstance(index, dict) or any(
            not isinstance(k, str)
            or re.fullmatch(r"[0-9a-f]{64}", k) is None
            or not isinstance(v, str)
            or re.fullmatch(r"g-[0-9a-f]{32}", v) is None
            for k, v in index.items()
        ):
            raise RuntimeError(f"session workspace index malformed: {name}")

        def save(path: Path, value: dict[str, Any]) -> None:
            pending = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
                    dir=cache,
                    prefix="index-",
                    delete=False,
                ) as stream:
                    pending = Path(stream.name)
                    json.dump(value, stream, sort_keys=True)
                pending.replace(path)
            finally:
                if pending is not None:
                    pending.unlink(missing_ok=True)

        last_use_path = cache / f"{name}.last-use.json"

        def read_usage() -> dict[str, float]:
            if last_use_path.is_symlink() or last_use_path.is_junction():
                raise RuntimeError(f"session workspace usage record is a link: {name}")
            try:
                usage = json.loads(last_use_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return {}
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise RuntimeError(f"session workspace usage unreadable: {name}") from error
            if not isinstance(usage, dict) or any(
                not isinstance(generation, str)
                or re.fullmatch(r"g-[0-9a-f]{32}", generation) is None
                or type(used) not in (int, float)
                or not math.isfinite(used)
                or used < 0
                for generation, used in usage.items()
            ):
                raise RuntimeError(f"session workspace usage malformed: {name}")
            return usage

        last_use: dict[str, float] = {}
        if persistent is not None:
            last_use = read_usage()
            now = time.time()
            # An older record has unknown last-use; start its retention clock now.
            for generation in index.values():
                last_use.setdefault(generation, now)
            keep = index.get(key)
            for generation, used in list(last_use.items()):
                if generation == keep or used >= now - 3 * 86400:
                    continue
                candidate = cache / generation
                lock = cache / f".{generation}.readers.lock"
                try:
                    with runtime.lock(lock, blocking=False):
                        markers = list(cache.glob(f".{generation}.reader-*"))
                        if markers:
                            if any(
                                marker.is_symlink() or marker.is_junction() for marker in markers
                            ):
                                raise RuntimeError("golden reader record is a link")
                            # A crashed reader has no recorded end; start retention now.
                            last_use[generation] = now
                            save(last_use_path, last_use)
                            for marker in markers:
                                marker.unlink()
                            continue
                        # Drop the selection before the file names: interruption cannot point at
                        # a half-removed generation. Its bytes stay while any reader holds a lease.
                        for old_key, old_generation in list(index.items()):
                            if old_generation == generation:
                                del index[old_key]
                        save(index_path, index)
                        if candidate.exists() and (left := remove_tree(candidate, cache)):
                            logging.getLogger(__name__).warning(
                                "golden retention refused: %s", left
                            )
                            continue
                    remove_idle_reader_lock(lock)
                    del last_use[generation]
                except portalocker.AlreadyLocked:
                    logging.getLogger(__name__).info("golden retention busy: %s", generation)
                except Exception as error:
                    # Unreadable ownership is not proof that a reader has released it.
                    logging.getLogger(__name__).warning("golden retention unmeasured: %s", error)
            save(last_use_path, last_use)

        def used(generation: Path) -> None:
            if persistent is not None:
                marker = cache / f".{generation.name}.reader-{uuid.uuid4().hex}"

                def released() -> None:
                    with runtime.lock(cache / f"{name}.lock"):
                        if marker.is_symlink() or marker.is_junction():
                            raise RuntimeError("golden reader record is a link")
                        usage = read_usage()
                        usage[generation.name] = max(usage.get(generation.name, 0), time.time())
                        save(last_use_path, usage)
                        marker.unlink()

                # Record ownership before creating a generation so an interrupted build is
                # discoverable by the retention owner, even if it never publishes an index.
                last_use[generation.name] = max(last_use.get(generation.name, 0), time.time())
                save(last_use_path, last_use)
                if hold_generation(generation, released):
                    marker.touch(exist_ok=False)

        def reject_corrupt(message: str) -> None:
            index.pop(key, None)  # supersede; never mutate a generation linked by another run
            save(index_path, index)
            raise GoldenPayloadCorrupt(message)

        target = cache / index[key] if key in index else None
        if target is not None:
            used(target)
        result = None
        if target is not None:
            record = target / "complete.json"
            if (
                target.is_symlink()
                or target.is_junction()
                or record.is_symlink()
                or record.is_junction()
            ):
                raise RuntimeError(f"session workspace is a link: {name}")
            try:
                if persistent is not None and record.stat().st_mode & 0o222:
                    reject_corrupt(f"session workspace published marker writable: {name}")
                result = json.loads(record.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError) as error:
                if persistent is not None:
                    reject_corrupt(f"session workspace published marker invalid: {name}: {error}")
            except (OSError, UnicodeError) as error:
                raise RuntimeError(f"session workspace marker unreadable: {name}") from error
        built = (
            not isinstance(result, dict)
            or result.get("name") != name
            or result.get("builder") != builder
            or result.get("inputs") != input_hash
            or not isinstance(result.get("metadata"), dict)
            or not isinstance(result.get("files"), dict)
            or (persistent is not None and result.get("persistent") is not True)
        )
        if built:
            if persistent is not None and target is not None:
                reject_corrupt(f"session workspace published marker invalid: {name}")
            target = cache / f"g-{uuid.uuid4().hex}"
            used(target)
            target.mkdir()
            running = dict(RUNNING_TEST)
            RUNNING_TEST.update(id=None, private=False)  # a builder writes what it copies
            try:
                metadata = build(target / "workspace")
            finally:
                RUNNING_TEST.update(running)
            if not isinstance(metadata, dict) or not (target / "workspace").is_dir():
                raise RuntimeError(
                    f"session workspace builder produced no workspace or metadata: {name}"
                )
            if persistent is not None:
                persistent.verify()
                seal_payload(target / "workspace", closed=_closed_seed)
                (target / "payload.json").chmod(stat.S_IREAD)
            result = {
                "name": name,
                "builder": builder,
                "inputs": input_hash,
                "metadata": metadata,
                "persistent": persistent is not None,
                "files": (
                    verified_payload(target / "workspace")
                    if persistent is not None
                    else _files(target / "workspace")
                ),
            }
            pending = target / "complete.pending"
            pending.write_text(
                json.dumps(result) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            pending.replace(target / "complete.json")
            if persistent is not None:
                (target / "complete.json").chmod(stat.S_IREAD)
            index[key] = target.name
            save(index_path, index)
        assert target is not None
        try:
            workspace_stat = (target / "workspace").lstat()
        except FileNotFoundError:
            if persistent is not None:
                reject_corrupt(f"session workspace was changed: {name}")
            raise RuntimeError(f"session workspace was changed: {name}") from None
        except OSError as error:
            raise RuntimeError(f"session workspace is unreadable: {name}") from error
        if not stat.S_ISDIR(workspace_stat.st_mode) or (
            getattr(workspace_stat, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise RuntimeError(f"session workspace is not a plain directory: {name}")
        try:
            actual = (
                verified_payload(target / "workspace")
                if persistent is not None
                else _files(target / "workspace")
            )
        except GoldenPayloadCorrupt as error:
            reject_corrupt(str(error))
        if actual != result["files"]:
            if persistent is not None:
                reject_corrupt(f"session workspace was changed: {name}")
            raise RuntimeError(f"session workspace was changed: {name}")
    line = (
        f"PLAYPEN_SESSION_WORKSPACE {'BUILD' if built else 'REUSE'} "
        f"{name} {time.perf_counter() - started:.3f}s"
    )
    print(line, flush=True)
    logging.getLogger("devtools.testing.session_footprint").info(line)
    return target / "workspace", result["metadata"]


# Each session seed a test read in place, checked against its build record at session end.
LINKED_SEEDS: set[Path] = set()


def seeds_changed() -> list[str]:
    """Each file of a seed read in place that no longer matches the hash its build recorded."""
    changed = []
    for seed in sorted(LINKED_SEEDS):
        marker = json.loads((seed.parent / "complete.json").read_text(encoding="utf-8"))
        record = marker["files"]
        now = verified_payload(seed) if marker.get("persistent") is True else _files(seed)
        changed += [
            f"{seed}: {name}"
            for name in sorted(set(record) | set(now))
            if record.get(name) != now.get(name)
        ]
    return changed


def _shared_data(path: Path) -> bool:
    """Use the product's byte-addressed array categories alongside the existing datasets."""
    if path.name == "market-data.duckdb" or path.suffix == ".parquet":
        return True
    if path.suffix != ".bin" or re.fullmatch(r"[0-9a-f]{64}", path.stem) is None:
        return False
    from alphalattice.control.product_host.storage.backup import BYTE_ADDRESSED

    return path.parent.parts[-3:] in BYTE_ADDRESSED


def copy_workspace(
    source: Path, target: Path, *, link_parquet: bool = False, link_data: bool | None = None
) -> Path:
    """Private writable files; only callers that never edit Parquet opt into links.

    The session seed remains alive for the entire run, including every linked
    consumer. Writable databases, records and model files get independent copies of
    the seed; the product's byte-addressed frozen arrays are shared read-only.
    Paths the product linked to one file inside the seed (a research input's
    pooled database and its source) stay one file inside the copy, as the
    product wrote them, so a copy never holds the same database twice.

    A running test reads the seed's market data, Parquet and byte-addressed arrays in place:
    linked, never copied, the seed's file refusing writers first (refuse_writers) so a test
    that writes one fails loudly, and seeds_changed() names any change at session end. Only a
    test marked private_data, which writes them, gets them copied; any other test
    asking for a copy is refused by name. A seed builder copies (RUNNING_TEST is clear).
    """
    copied: dict[tuple[int, int], str] = {}  # a seed file the product linked -> its copy here
    sealed = (Path(source).parent / "payload.json").is_file()
    test = RUNNING_TEST["id"] if (Path(source).parent / "complete.json").is_file() else None
    if test is not None and not RUNNING_TEST["private"]:
        if link_data is False:
            raise RuntimeError(
                f"test_datasets.private_copy_unmarked: {test} copies the seed {source}; mark "
                "it private_data if it writes the seed's protected data, else read in place"
            )
        link_data = True

    def copy_file(src: str, dst: str) -> str:
        data = _shared_data(Path(src))
        if link_data and data:
            refuse_writers(src)
            try:
                os.link(src, dst)
            except OSError as error:
                raise RuntimeError(
                    f"test_datasets.link_refused: {test} cannot read {src} in place: {error}"
                ) from error
            LINKED_SEEDS.add(Path(source))
            return dst
        if link_parquet and Path(src).suffix == ".parquet":
            os.link(src, dst)
            return dst
        seen = os.stat(src)
        if seen.st_nlink > 1 and (key := (seen.st_dev, seen.st_ino)) in copied:
            os.link(copied[key], dst)
            return dst
        shutil.copy2(src, dst)
        if data or sealed:  # a private copy is writable without changing its sealed source
            os.chmod(dst, os.stat(dst).st_mode | stat.S_IWRITE)
        if seen.st_nlink > 1:
            copied[(seen.st_dev, seen.st_ino)] = dst
        return dst

    shutil.copytree(source, target, copy_function=copy_file)
    return target


def keep_seed_links(source: str, target: str) -> str:
    """Keep a seed's read-only datasets linked in a copy of a test's own copy."""
    data = _shared_data(Path(source))
    if data and os.stat(source).st_nlink > 1:
        os.link(source, target)
        return target
    return str(shutil.copy2(source, target))
