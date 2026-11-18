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
source a body is kept from, or refilled from once evicted.

The Host's bodies follow the workspace's Panels: each publication of their lifecycle (`root`)
makes the chunks of every snapshot still physically available the workspace's only roots, so a
chunk a new date or a changed dependency replaced (a new chunk hash) loses its root. It evicts a
body unread for `UNREAD` whose stored chunk the Panel reader's proof still holds (`proven`: its
identity and values, as a read checks them), and collects the bodies no workspace roots.

The data root is data_layer's alone, so it registers the three calls this module makes on import
(`register`, from kernel/data_layer/derived_reads); a derived read with none registered is
refused by name, never answered from the workspace's file.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Collection, Mapping
from datetime import timedelta
from pathlib import Path
from typing import Final

SETTING: Final = "ALPHALATTICE_PANEL_READS"
CHOICES: Final = ("stored", "derived")
"""Where Panel chunks are read from: the workspace's stored files, or the derived method."""
TABLE: Final = "feature_panel"
UNREAD: Final = timedelta(days=30)
"""How long a kept body goes unread before it may be evicted: a month, the slowest research
cadence that reads a Panel; refilling one costs a link and one footer read."""


Body = Callable[[str, str], Path | None]
"""data_layer's read: the body the log names for a file table's partition with a digest, or
None."""
Keep = Callable[..., None]
"""data_layer's write: hand one file to the Host to keep as a partition (table, recipe_key, part,
digest, source); refused by name outside the Host."""
Root = Callable[..., dict[str, int] | None]
"""data_layer's lifecycle: make a file table's partitions with these digests the Host
workspace's only roots, evict the unread ones whose source verifies, collect the unrooted; None
outside the Host."""
_files: list[tuple[Body, Keep, Root]] = []


def register(bodies: Body | None, keep: Keep | None = None, root: Root | None = None) -> None:
    """Install data_layer's read, write and lifecycle; None leaves the process with none, so a
    derived read is refused."""
    _files[:] = [] if bodies is None or keep is None or root is None else [(bodies, keep, root)]


Proof = Callable[[Path, dict, dict], bool]
"""feature_engine's proof that a chunk file holds the content a manifest's entry identifies
(panels/reader.chunk_content_proven), the one a read applies."""
_proofs: list[Proof] = []


def register_proof(proof: Proof | None) -> None:
    """Install the Panel reader's content proof, which it registers on import; None removes it,
    and with none installed no body is evicted."""
    _proofs[:] = [] if proof is None else [proof]


def proven(path: Path, manifest: dict, item: dict) -> bool:
    """Whether the chunk file `path` holds the content `manifest`'s entry `item` identifies, by the
    registered proof; False with none registered, so eviction keeps every body."""
    return bool(_proofs) and _proofs[0](path, manifest, item)


def registered() -> tuple[Body, Keep, Root] | None:
    """data_layer's read, write and lifecycle as installed, if any."""
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


def _registered() -> tuple[Body, Keep, Root]:
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


def root(chunk_hashes: Collection[str], verified: Callable[[str], bool]) -> dict[str, int] | None:
    """Under `derived`, in the Host, make the kept chunks among `chunk_hashes` its workspace's
    only Panel roots, evict a body unread for `UNREAD` whose stored chunk `verified` proves
    byte for byte, and collect the bodies no workspace roots; the counts, or None under
    `stored` or outside the Host, where the Host's next publication roots them."""
    if chosen() != "derived":
        return None
    return _registered()[2](TABLE, chunk_hashes, unread_for=UNREAD, verified=verified)


__all__ = [
    "CHOICES",
    "SETTING",
    "TABLE",
    "UNREAD",
    "Body",
    "Keep",
    "Proof",
    "Root",
    "body",
    "chosen",
    "keep",
    "key",
    "proven",
    "register",
    "register_proof",
    "root",
]
