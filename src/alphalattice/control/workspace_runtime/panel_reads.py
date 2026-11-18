"""Feature Panel chunks as a derived method: the setting that chooses the file a chunk is read from.

`ALPHALATTICE_PANEL_READS` is `stored`, the default and the rollback, or `derived`. Under
`derived` each Panel chunk is a partition of the derived table `feature_panel` that the Host's
StorePort holds as a file (kernel/data_layer/derived_files): its body is the chunk's own Parquet,
byte for byte, placed once per chunk hash under the data root, and keyed in the log's cache_entry
by the binding that wrote it, its year and part, and its chunk hash. Publishing writes the
workspace's chunk as before. Under `derived`, a chunk's first verified read (one published since,
one published before, or a copy the Host stopped before) reads the workspace's chunk, the same
bytes, and hands it to the Host's StorePort, which keeps it on its own worker; later reads take
its body. The workspace's chunk stays written: it is the rollback until row U retires it, and the
source a body is kept from.

The data root is data_layer's alone, so it registers the two calls this module makes on import
(`register`, from kernel/data_layer/derived_reads); a derived read with none registered is
refused by name, never answered from the workspace's file.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

SETTING: Final = "ALPHALATTICE_PANEL_READS"
CHOICES: Final = ("stored", "derived")
"""Where Panel chunks are read from: the workspace's stored files, or the derived method."""
TABLE: Final = "feature_panel"


Body = Callable[[str, str], Path | None]
"""data_layer's read: the body the log names for a file table's partition with a digest, or
None."""
Keep = Callable[..., None]
"""data_layer's write: hand one file to the Host to keep as a partition (table, recipe_key, part,
digest, source); refused by name outside the Host."""
_files: list[tuple[Body, Keep]] = []


def register(bodies: Body | None, keep: Keep | None = None) -> None:
    """Install data_layer's read and write; None leaves the process with none, so a derived read
    is refused."""
    _files[:] = [] if bodies is None or keep is None else [(bodies, keep)]


def registered() -> tuple[Body, Keep] | None:
    """data_layer's read and write as installed, if any."""
    return _files[0] if _files else None


def chosen() -> str:
    """The setting's value; refused by name when it names neither choice."""
    value = os.environ.get(SETTING, CHOICES[0])
    if value not in CHOICES:
        raise ValueError(
            f"feature_panel.panel_reads_unknown: {SETTING} is {value!r}; set it to stored or"
            " derived, or unset it for stored"
        )
    return value


def _registered() -> tuple[Body, Keep]:
    if not _files:
        raise RuntimeError("feature_panel.derived_reads_unserved: no data layer is registered")
    return _files[0]


def body(chunk_hash: str) -> Path | None:
    """Under `derived`, the body the log names for the chunk `chunk_hash`, found by its hash
    alone; None under `stored`, or while no body is kept for it."""
    if chosen() != "derived":
        return None
    return _registered()[0](TABLE, chunk_hash)


def key(metadata: Mapping[bytes, bytes]) -> tuple[str, str]:
    """A chunk's partition key from the metadata its writer recorded: the binding that wrote it,
    and its part, `<year>` or `<year>/<part>`."""
    year = metadata[b"alphalattice.calendar_year"].decode()
    part = metadata.get(b"alphalattice.calendar_part")
    binding = metadata[b"alphalattice.panel_binding_hash"].decode()
    return binding, year if part is None else f"{year}/{part.decode()}"


def keep(chunk: Path, chunk_hash: str, metadata: Mapping[bytes, bytes]) -> None:
    """Under `derived`, hand the verified stored chunk `chunk`, whose recorded metadata is
    `metadata`, to the Host to keep as its partition (`key`); nothing under `stored`."""
    if chosen() != "derived":
        return
    recipe_key, part = key(metadata)
    _registered()[1](TABLE, recipe_key=recipe_key, part=part, digest=chunk_hash, source=chunk)


__all__ = [
    "CHOICES",
    "SETTING",
    "TABLE",
    "Body",
    "Keep",
    "body",
    "chosen",
    "keep",
    "key",
    "register",
]
