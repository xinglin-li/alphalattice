"""Product-written research prerequisites, once per pytest session across workers.

Only the builder sees the seed. Tests receive a private database and record copy;
read-only Portfolio consumers may link the seed's immutable Parquet objects.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.pytest_environment import golden_runtime


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
            metadata = build(target / "workspace")
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
    print(
        f"PLAYPEN_SESSION_WORKSPACE {'BUILD' if built else 'REUSE'} {name} "
        f"{time.perf_counter() - started:.3f}s",
        flush=True,
    )
    return target / "workspace", result["metadata"]


def copy_workspace(source: Path, target: Path, *, link_parquet: bool = False) -> Path:
    """Private writable files; only callers that never edit Parquet opt into links.

    The session seed remains alive for the entire run, including every linked
    consumer. Databases, records and model files always get independent copies.
    """

    def copy_file(src: str, dst: str) -> str:
        if link_parquet and Path(src).suffix == ".parquet":
            os.link(src, dst)
            return dst
        return shutil.copy2(src, dst)

    shutil.copytree(source, target, copy_function=copy_file)
    return target
