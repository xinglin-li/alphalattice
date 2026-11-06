"""Product-written research prerequisites, once per pytest session across workers.

Only the builder sees the seed. Tests receive a private database and record copy;
read-only Portfolio consumers may link the seed's immutable Parquet objects.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from devtools.evaluation.golden_workspaces import refuse_writers
from tests.pytest_environment import RUNNING_TEST, golden_runtime


def _files(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        with path.open("rb") as stream:
            result[path.relative_to(root).as_posix()] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return result


def session_workspace(
    factory: pytest.TempPathFactory,
    name: str,
    build: Callable[[Path], dict[str, Any]],
    *,
    inputs: Any = None,
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
    base = factory.getbasetemp()
    if os.environ.get("PYTEST_XDIST_WORKER"):
        base = base.parent
    cache = base / "research-prerequisites"
    if cache.is_symlink() or cache.is_junction():
        raise RuntimeError("session workspace cache is a link")
    cache.mkdir(exist_ok=True)
    index_path = cache / f"{name}.json"
    started = time.perf_counter()
    with golden_runtime().lock(cache / f"{name}.lock"):
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
        target = cache / index[key] if key in index else None
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
                result = json.loads(record.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            except (OSError, UnicodeError) as error:
                raise RuntimeError(f"session workspace marker unreadable: {name}") from error
        built = (
            not isinstance(result, dict)
            or result.get("name") != name
            or result.get("builder") != builder
            or result.get("inputs") != input_hash
            or not isinstance(result.get("metadata"), dict)
            or not isinstance(result.get("files"), dict)
        )
        if built:
            target = cache / f"g-{uuid.uuid4().hex}"
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
            result = {
                "name": name,
                "builder": builder,
                "inputs": input_hash,
                "metadata": metadata,
                "files": _files(target / "workspace"),
            }
            pending = target / "complete.pending"
            pending.write_text(
                json.dumps(result) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            pending.replace(target / "complete.json")
            index[key] = target.name
            pending_index = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
                    dir=cache,
                    prefix="index-",
                    delete=False,
                ) as file:
                    pending_index = Path(file.name)
                    json.dump(index, file, sort_keys=True)
                pending_index.replace(index_path)
            finally:
                if pending_index is not None:
                    pending_index.unlink(missing_ok=True)
        assert target is not None
        if not (target / "workspace").is_dir() or _files(target / "workspace") != result["files"]:
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
        record = json.loads((seed.parent / "complete.json").read_text(encoding="utf-8"))["files"]
        now = _files(seed)
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
        if data:  # a private copy stays writable after its seed's file refused writers
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
