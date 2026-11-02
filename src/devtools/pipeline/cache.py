"""The result cache, in the form of Bazel's action cache: a group's result keyed by what it read.

The action key digests what is declared; an entry lists what its group was seen to read, each
checked against the tree on lookup. Any doubt about an entry is a miss, never a refusal.
``Store.open`` and ``Tree.read`` answer a ``result_cache.*`` reason when the run cannot use it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

SCHEMA = "alphalattice.result-cache.v1"
GIT_SECONDS = 10
IDLE_SECONDS = 14 * 86400
CAP_SHARE, CAP_MOST = 100, 4 << 30  # a hundredth of the volume, at most 4 GiB
Relative = Annotated[str, Field(pattern=r"^(?![/\\]|[A-Za-z]:)(?!(?:.*/)?\.\.(?:/|$))[^\\\x00]+$")]
Hex64 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Text = Annotated[str, Field(min_length=1)]
Seconds = Annotated[float, Field(ge=0)]


def digest(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def action_key(**declared: Any) -> str:
    return digest({"schema": SCHEMA, **declared})


class Entry(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(
        strict=True, extra="forbid", frozen=True, allow_inf_nan=False, regex_engine="python-re"
    )
    schema_: Literal["alphalattice.result-cache.v1"] = Field(alias="schema")
    key: Hex64
    files: dict[Relative, Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]]  # by Git blob
    probes: dict[Relative, Literal["file", "dir"] | None]
    listings: dict[Relative, Hex64]
    env: dict[str, Hex64]  # a value's digest; an unset variable's is digest(None)
    results: Annotated[dict[str, tuple[Literal["PASSED", "SKIPPED"], Seconds]], Field(min_length=1)]
    details: dict[str, dict[str, Any]] = Field(default_factory=dict)
    writer: Annotated[dict[Literal["line", "head", "invocation", "at"], Text], Field(min_length=4)]
    digest: Hex64

    @classmethod
    def sealed(cls, **fields: Any) -> Entry:
        unsealed = cls(schema=SCHEMA, **fields, digest="0" * 64)
        return unsealed.model_copy(update={"digest": digest(unsealed.body())})  # type: ignore[no-any-return]

    def body(self) -> dict[str, Any]:
        return self.model_dump(  # type: ignore[no-any-return]
            mode="json", by_alias=True, exclude_defaults=True, exclude={"digest"}
        )


@dataclass
class Tree:
    """A checkout's tracked files by Git blob, read once a stage; probes and listings memoized."""

    root: Path
    blobs: dict[str, str]
    seen: dict[tuple[str, str], str | None] = field(default_factory=dict)

    @classmethod
    def read(cls, root: Path) -> Tree | str:
        try:
            listed = _git(root, "ls-files", "-s", "-z").split("\0")
            rows = (r.partition("\t") for r in listed if r[:6] not in ("", "120000", "160000"))
            blobs = {path: row.split()[1] for row, _, path in rows}  # no links or submodules
            changed = [path for path in _git(root, "diff", "--name-only", "-z").split("\0") if path]
            dirty = [path for path in changed if blobs.pop(path, None) and (root / path).is_file()]
            if dirty:
                hashed = _git(root, "hash-object", "--stdin-paths", stdin="\n".join(dirty))
                blobs.update(zip(dirty, hashed.split(), strict=True))
        except subprocess.TimeoutExpired:
            return "result_cache.git_timeout"
        except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
            return "result_cache.tree_unreadable"
        return cls(Path(os.path.realpath(root)), blobs)

    def holds(self, entry: Entry) -> bool:
        return (
            all(
                self.blobs.get(path) == blob and self._look("probe", path) == "file"
                for path, blob in entry.files.items()
            )
            and all(self._look("probe", path) == kind for path, kind in entry.probes.items())
            and all(self._look("listing", path) == names for path, names in entry.listings.items())
            and all(digest(os.environ.get(name)) == value for name, value in entry.env.items())
        )

    def _look(self, how: str, relative: str) -> str | None:
        if (how, relative) not in self.seen:
            path, found = self.root / relative, None
            if not _owned(path):
                found = "outside"
            elif how == "listing":
                with contextlib.suppress(OSError):
                    found = digest(sorted(set(os.listdir(path)) - {"__pycache__"}))
            else:
                try:  # one stat; only a path known to be absent is None
                    found = "dir" if stat.S_ISDIR(os.stat(path).st_mode) else "file"
                except (FileNotFoundError, NotADirectoryError):
                    found = None
                except OSError:
                    found = "unreadable"
            self.seen[how, relative] = found
        return self.seen[how, relative]


@dataclass(frozen=True)
class Store:
    """The entries under ``root``/ac, each written whole by a clean commit's stage run."""

    root: Path
    cap: int

    @classmethod
    def open(cls, root: Path, cap: int | None = None) -> Store | str:
        root = Path(os.path.realpath(root))
        if (root / "disabled").exists():
            return "result_cache.disabled"
        try:
            (root / "ac").mkdir(parents=True, exist_ok=True)
            if not _owned(root / "ac"):  # a linked ac leads outside the store
                return "result_cache.unwritable"
            tempfile.TemporaryFile(dir=root / "ac").close()  # writable, or the cache is off
            return cls(root, cap or min(shutil.disk_usage(root).total // CAP_SHARE, CAP_MOST))
        except OSError:
            return "result_cache.unwritable"

    def path(self, key: str) -> Path:
        return self.root / "ac" / key[:2] / f"{key}.json"

    def usable(self, path: Path) -> bool:
        """Asked before any read or write: the breaker stops a stage at once, and a linked ac,
        shard or entry leads outside the store."""
        try:
            return not (self.root / "disabled").exists() and _owned(path)
        except OSError:  # a breaker or path that cannot be read declines the cache
            return False

    def lookup(self, key: str, tree: Tree) -> Entry | None:
        path = self.path(key)
        if not self.usable(path):
            return None
        with contextlib.suppress(OSError, ValidationError):
            entry: Entry = Entry.model_validate_json(path.read_bytes())
            if entry.key == key and entry.digest == digest(entry.body()) and tree.holds(entry):
                with contextlib.suppress(OSError):
                    os.utime(path)  # a hit renews its idle time
                return entry
        return None

    def write(self, entry: Entry) -> None:
        """Best effort: a store that cannot take the entry skips it, never failing the stage."""
        path = self.path(entry.key)
        temporary = path.with_name(f".{uuid.uuid4().hex}.tmp")
        if not self.usable(path.parent):
            return
        try:
            path.parent.mkdir(exist_ok=True)
            temporary.write_text(entry.model_dump_json(by_alias=True), "utf-8")
            os.replace(temporary, path)
        except OSError:  # a reader holding it open on Windows, a full disk, an evicted temporary
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)

    def evict(self, now: float | None = None) -> None:
        """Once a stage: entries idle past 14 days go, then the oldest while over the cap."""
        now, rows = now or time.time(), []
        if not _owned(self.root / "ac"):
            return
        with contextlib.suppress(OSError):  # a scan cut short evicts less, never outside the store
            for shard in os.scandir(self.root / "ac"):
                if shard.is_dir(follow_symlinks=False) and not shard.is_junction():
                    for item in os.scandir(shard.path):
                        if item.is_file(follow_symlinks=False):
                            status = item.stat(follow_symlinks=False)
                            rows.append((status.st_mtime, status.st_size, Path(item.path)))
        total = sum(size for _, size, _ in rows)
        for mtime, size, path in sorted(rows):  # oldest first, so the idle ones lead
            if total <= self.cap and now - mtime <= IDLE_SECONDS:
                break
            with contextlib.suppress(OSError):
                path.unlink()
                total -= size


def _owned(path: Path) -> bool:
    """``path`` as itself: not through a junction or link, nor a case NTFS folds."""
    return os.path.realpath(path) == os.path.normpath(path)


def _git(root: Path, *words: str, stdin: str | None = None) -> str:
    command, bound = ["git", "-C", str(root), *words], GIT_SECONDS  # a hang turns the cache off
    return subprocess.run(
        command,
        input=stdin,
        capture_output=True,
        encoding="utf-8",
        check=True,
        timeout=bound,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
    ).stdout
