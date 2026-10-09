"""Keep pinned model packs outside Git and research workspaces.

The application model store: where retrieval packs are installed, verified
and bound into a workspace, outside Git and outside every workspace.

One directory per pack and revision (`<owner>--<repo>/<revision>/`), holding
exactly the files the pack's loader consumes. A pack is installed only once
every file is present with its pinned hash and nothing else is in the
directory; it is staged under a sibling `.staging-<revision>` directory
first and renamed into place last, so an interrupted install is never an
installed pack -- and what it staged is kept, so the next install continues
it instead of fetching 541 MB again. A recipe is bound into a workspace as a small directory of
links (`encoder`, `reranker`; NTFS junctions on Windows, symlinks elsewhere)
pointing at the store's packs, so one physical copy serves every workspace
and the workspace manifest still names a confined path.

The store's location is the user's: `ALPHALATTICE_MODEL_STORE`, else the
platform's application-data directory. Downloading is a separate, admitted
step (`fetch`): this module verifies, it never reaches the network itself.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alphalattice.kernel.knowledge.hybrid_contracts import (
    BGE_MODEL_ARTIFACT_SHA256,
    BGE_MODEL_ID,
    BGE_MODEL_REVISION,
    BGE_TOKENIZER_SHA256,
    MODEL_ARTIFACT_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    QWEN3_EMBEDDING_MODEL_ID,
    QWEN3_EMBEDDING_MODEL_REVISION,
    QWEN3_EMBEDDING_PACK,
    QWEN3_RERANKER_MODEL_ID,
    QWEN3_RERANKER_MODEL_REVISION,
    QWEN3_RERANKER_PACK,
    RECIPE_BGE_SMALL_CPU,
    RECIPE_MINILM_CPU,
    RECIPE_QWEN3_ENCODER_GPU,
    RECIPE_QWEN3_GPU,
    RECIPE_QWEN3_RERANKER_GPU,
    RERANKER_MODEL_ID,
    RERANKER_MODEL_REVISION,
    RERANKER_PACK,
    SUPPORTED_RECIPES,
    TOKENIZER_SHA256,
)


@dataclass(frozen=True, slots=True)
class PackDefinition:
    """Declare an exact model repository revision and required file digests.

    One supported pack: the repository and revision it is fetched from and
    every file its loader reads, with the hash each must have.
    """

    pack_id: str
    repository: str
    revision: str
    files: tuple[tuple[str, str], ...]
    license_id: str
    runtime: str
    approximate_bytes: int

    @property
    def directory_name(self) -> str:
        """Return the filesystem-safe repository directory name.

        Returns:
            Repository identifier with each slash replaced by two hyphens.
        """
        return self.repository.replace("/", "--")


MINILM_ENCODER = PackDefinition(
    "minilm-l12-multilingual",
    MODEL_ID,
    MODEL_REVISION,
    (("onnx/model.onnx", MODEL_ARTIFACT_SHA256), ("sentencepiece.bpe.model", TOKENIZER_SHA256)),
    "Apache-2.0",
    "ONNX_CPU",
    475_370_661,
)
MSMARCO_RERANKER = PackDefinition(
    "ms-marco-minilm-l6",
    RERANKER_MODEL_ID,
    RERANKER_MODEL_REVISION,
    RERANKER_PACK,
    "Apache-2.0",
    "FASTEMBED_ONNX_CPU",
    91_707_303,
)
BGE_ENCODER = PackDefinition(
    "bge-small-en-v1.5",
    BGE_MODEL_ID,
    BGE_MODEL_REVISION,
    # The ONNX adapter reads the graph and `tokenizer.json`; the tokenizer's
    # companion files are fetched with it (the upstream tokenizer
    # distribution is these files together) and pinned, never read.
    (
        ("config.json", "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750"),
        ("onnx/model.onnx", BGE_MODEL_ARTIFACT_SHA256),
        (
            "special_tokens_map.json",
            "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
        ),
        ("tokenizer.json", BGE_TOKENIZER_SHA256),
        (
            "tokenizer_config.json",
            "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3",
        ),
        ("vocab.txt", "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
    ),
    "MIT",
    "ONNX_CPU",
    134_037_628,
)
QWEN3_ENCODER = PackDefinition(
    "qwen3-embedding-0.6b",
    QWEN3_EMBEDDING_MODEL_ID,
    QWEN3_EMBEDDING_MODEL_REVISION,
    QWEN3_EMBEDDING_PACK,
    "Apache-2.0",
    "TORCH_CUDA",
    1_207_469_357,
)
QWEN3_RERANKER = PackDefinition(
    "qwen3-reranker-0.6b",
    QWEN3_RERANKER_MODEL_ID,
    QWEN3_RERANKER_MODEL_REVISION,
    QWEN3_RERANKER_PACK,
    "Apache-2.0",
    "TORCH_CUDA_CAUSAL_LM",
    1_207_471_008,
)
PACKS: tuple[PackDefinition, ...] = (
    MINILM_ENCODER,
    MSMARCO_RERANKER,
    BGE_ENCODER,
    QWEN3_ENCODER,
    QWEN3_RERANKER,
)
RECIPE_PACKS: dict[str, tuple[PackDefinition, PackDefinition]] = {
    RECIPE_MINILM_CPU: (MINILM_ENCODER, MSMARCO_RERANKER),
    RECIPE_BGE_SMALL_CPU: (BGE_ENCODER, MSMARCO_RERANKER),
    RECIPE_QWEN3_ENCODER_GPU: (QWEN3_ENCODER, MSMARCO_RERANKER),
    RECIPE_QWEN3_RERANKER_GPU: (MINILM_ENCODER, QWEN3_RERANKER),
    RECIPE_QWEN3_GPU: (QWEN3_ENCODER, QWEN3_RERANKER),
}
"""Encoder and reranker of every supported recipe; the two presets a user
chooses between are recipes, never a free pairing."""
assert set(RECIPE_PACKS) == set(SUPPORTED_RECIPES)

PACK_ABSENT = "ABSENT"
PACK_STAGED = "STAGED_INCOMPLETE"
PACK_TAMPERED = "TAMPERED"
PACK_INSTALLED = "INSTALLED_VERIFIED"


class ModelStoreError(ValueError):
    """The store refused: the code names what is missing or wrong."""


def writable_store(store: Path) -> Path:
    """The model store, made and shown to take a file before anything is staged in it.

    A store this process cannot create or write -- an application-data directory a sandbox
    denies -- is refused by name before any download, never met as an error halfway through.

    Args:
        store: The model store root.

    Returns:
        The store, which now exists and takes a file.

    Raises:
        ModelStoreError: `model_store.store_unwritable`.
    """
    try:
        store.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=store, prefix=".write-probe-"):
            pass
    except OSError as error:
        raise ModelStoreError("model_store.store_unwritable") from error
    return store


def default_store_root() -> Path:
    """`ALPHALATTICE_MODEL_STORE`, else the platform's application-data directory."""
    override = os.environ.get("ALPHALATTICE_MODEL_STORE", "").strip()
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "AlphaLattice" / "models"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "alphalattice" / "models"


def pack_directory(store: Path, pack: PackDefinition) -> Path:
    """Locate the exact pack revision beneath a supplied model store.

    Args:
        store: Caller-selected local model-store root.
        pack: Exact repository and immutable pinned revision.

    Returns:
        Repository directory followed by the immutable pinned revision.
    """
    return store / pack.directory_name / pack.revision


def _staging_directory(store: Path, pack: PackDefinition) -> Path:
    return store / pack.directory_name / f".staging-{pack.revision}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_directory(directory: Path, pack: PackDefinition) -> str | None:
    """Verify that a pack directory contains exactly its pinned files and digests.

    None when every consumed file is present with its hash and nothing
    else is there; otherwise what is wrong, naming the first file.

    Args:
        directory: Local pack directory to verify.
        pack: Exact expected file names and SHA-256 digests.

    Returns:
        None for an exact verified pack, otherwise the first verification problem.
    """
    if not directory.is_dir():
        return "directory is absent"
    present = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    expected = {name for name, _digest in pack.files}
    missing = sorted(expected - present)
    if missing:
        return f"missing {missing[0]}"
    unexpected = sorted(present - expected)
    if unexpected:
        return f"unadmitted file {unexpected[0]}"
    for name, digest in pack.files:
        if _sha256(directory / name) != digest:
            return f"hash differs for {name}"
    return None


@dataclass(frozen=True, slots=True)
class PackStatus:
    """Describe an installed, absent, staged-incomplete or tampered local model pack.

    pack identifies the pinned definition; status and detail report verification/presence, and
    directory locates its intended installed revision. Staged bytes do not count as an installed
    pack.
    """

    pack: PackDefinition
    status: str
    detail: str | None
    directory: Path


def pack_status(store: Path, pack: PackDefinition) -> PackStatus:
    """Inspect installed and staged bytes for one pinned model pack.

    Args:
        store: Local model-store root.
        pack: Exact repository revision and expected file digests.

    Returns:
        INSTALLED_VERIFIED only after verification; otherwise absent, staged-incomplete or tampered
        details.
    """
    directory = pack_directory(store, pack)
    if not directory.exists():
        staged = _staging_directory(store, pack)
        if staged.exists():
            held = sum(path.stat().st_size for path in staged.rglob("*") if path.is_file())
            return PackStatus(
                pack,
                PACK_STAGED,
                f"a staged install was not completed: {held} of about "
                f"{pack.approximate_bytes} bytes staged; installing again continues it",
                directory,
            )
        return PackStatus(pack, PACK_ABSENT, None, directory)
    problem = verify_directory(directory, pack)
    if problem is None:
        return PackStatus(pack, PACK_INSTALLED, None, directory)
    return PackStatus(pack, PACK_TAMPERED, problem, directory)


Fetcher = Callable[[PackDefinition, str, Path], Path]
"""`(pack, file name, destination directory) -> the fetched file's path`:
the admitted way a file reaches the staging directory (a Hub download, or a
copy from a local directory). The store verifies what arrived."""

PARTIAL_SUFFIX = ".part"
"""A file's bytes while they arrive, beside the name it takes when whole."""

Received = Callable[[str, int, int | None], None]
"""`(file name, bytes held, the file's size when the server says it)`, as bytes arrive."""


def install_pack(
    store: Path,
    pack: PackDefinition,
    fetch: Fetcher,
    *,
    progress: Callable[[str, int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> PackStatus:
    """Stage, verify and publish one pinned pack with resumable arrival handling.

    Stage every consumed file, verify each as it arrives, rename into place
    last. A cancelled or interrupted install keeps what it staged (the pack is
    `STAGED_INCOMPLETE`, never installed): the next install keeps each staged
    file that verifies and continues a partial one where it stopped (the
    fetcher's `.part`); a file that arrives with another hash is removed and
    refused by name. An already installed and verified pack is returned
    without work.

    Args:
        store: Local model-store root.
        pack: Exact pack definition to install.
        fetch: Caller-admitted arrival function; installation verifies every arriving file.
        progress: Optional observer receiving file name, one-based file index and file count.
        cancelled: Optional cancellation predicate checked during installation.

    Returns:
        Status of the verified installed pack; cancellation retains staged bytes and raises.

    Raises:
        ModelStoreError: Installation is cancelled or an arriving file/complete pack fails
            verification.
    """
    current = pack_status(store, pack)
    if current.status == PACK_INSTALLED:
        return current
    staging = _staging_directory(writable_store(store), pack)
    staging.mkdir(parents=True, exist_ok=True)
    expected = {name for name, _digest in pack.files}
    for path in [p for p in staging.rglob("*") if p.is_file()]:
        name = path.relative_to(staging).as_posix()
        if name not in expected and name.removesuffix(PARTIAL_SUFFIX) not in expected:
            path.unlink()
    for index, (name, digest) in enumerate(pack.files, start=1):
        if cancelled is not None and cancelled():
            raise ModelStoreError("model_store.install_cancelled")
        if progress is not None:
            progress(name, index, len(pack.files))
        destination = staging / name
        if destination.is_file() and _sha256(destination) == digest:
            continue
        fetched = fetch(pack, name, staging)
        if fetched.resolve() != destination.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fetched, destination)
        if _sha256(destination) != digest:
            destination.unlink(missing_ok=True)
            raise ModelStoreError(f"model_store.file_identity_invalid:{name}")
    problem = verify_directory(staging, pack)
    if problem is not None:
        raise ModelStoreError(f"model_store.pack_incomplete:{problem}")
    final = pack_directory(store, pack)
    if final.exists():
        shutil.rmtree(final)
    staging.rename(final)
    return pack_status(store, pack)


def local_directory_fetcher(source: Path) -> Fetcher:
    """Import from a directory that already holds the pack's files."""

    def fetch(_pack: PackDefinition, name: str, _staging: Path) -> Path:
        path = source / name
        if not path.is_file():
            raise ModelStoreError(f"model_store.import_source_missing:{name}")
        return path

    return fetch


HUB_ENDPOINT = "https://huggingface.co"
DOWNLOAD_ATTEMPTS = 5
"""A transfer that breaks is asked again for the rest this many times in one install."""


def hub_fetcher(
    *,
    received: Received | None = None,
    client: Any = None,
) -> Fetcher:
    """Create the pinned HTTPS fetcher for a caller-admitted model-pack download.

    Download the pinned revision's file from the Hub over HTTPS, continuing
    a partial one: its bytes arrive in `<name>.part` beside the staged name, a
    later attempt -- in this install after a broken transfer, or in the next
    install -- asks only for the rest (an HTTP range), and the file takes its
    name when whole. `received(name, bytes, total)` reports as bytes arrive.
    The caller admits the network before asking for this; the store verifies
    what arrived by its pinned hash, so a wrong byte is never installed.

    Args:
        received: Optional observer of file name, received bytes and total.
        client: Optional HTTPS client supplied by the caller.

    Returns:
        Pinned-revision fetcher using resumable partial arrivals; network admission is the caller's
        responsibility.
    """
    import httpx

    def fetch(pack: PackDefinition, name: str, staging: Path) -> Path:
        destination = staging / name
        partial = destination.with_name(destination.name + PARTIAL_SUFFIX)
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HUB_ENDPOINT}/{pack.repository}/resolve/{pack.revision}/{name}"
        session = client or httpx.Client(
            follow_redirects=True, timeout=httpx.Timeout(60.0, connect=30.0)
        )
        try:
            for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
                start = partial.stat().st_size if partial.is_file() else 0
                try:
                    _transfer(session, url, name, partial, start, received)
                    break
                except (httpx.TransportError, _RetryableStatus) as error:
                    if attempt == DOWNLOAD_ATTEMPTS:
                        raise ModelStoreError(
                            f"model_store.download_interrupted:{name}; "
                            "installing again continues it"
                        ) from error
                    time.sleep(min(30.0, 2.0**attempt))
        finally:
            if client is None:
                session.close()
        os.replace(partial, destination)
        return destination

    return fetch


class _RetryableStatus(RuntimeError):
    """The Hub answered with a server error: asked again, like a broken transfer."""


def _transfer(
    session: Any,
    url: str,
    name: str,
    partial: Path,
    start: int,
    received: Received | None,
) -> None:
    headers = {"Range": f"bytes={start}-"} if start else {}
    with session.stream("GET", url, headers=headers) as response:
        if response.status_code == 416 and start:
            # The partial already holds every byte the file has: it is whole.
            return
        if response.status_code >= 500:
            raise _RetryableStatus(f"{response.status_code}")
        if response.status_code not in (200, 206):
            raise ModelStoreError(f"model_store.download_refused:{name}:{response.status_code}")
        if response.status_code == 200:
            start = 0  # the server sent the whole file: it replaces the partial
        total = _content_total(response.headers, start)
        done = start
        with partial.open("ab" if start else "wb") as handle:
            # Each network chunk as it arrives: a break loses nothing already received.
            for chunk in response.iter_bytes():
                handle.write(chunk)
                done += len(chunk)
                if received is not None:
                    received(name, done, total)


def _content_total(headers: Any, start: int) -> int | None:
    ranged = headers.get("content-range", "")
    if "/" in ranged and ranged.rsplit("/", 1)[1].isdigit():
        return int(ranged.rsplit("/", 1)[1])
    length = headers.get("content-length", "")
    return start + int(length) if length.isdigit() else None


def _link_directory(link: Path, target: Path) -> None:
    """A directory link the OS follows: a junction on Windows, a symlink elsewhere."""

    if link.exists() or link.is_symlink():
        if link.is_symlink() or os.name == "nt":
            link.unlink() if link.is_symlink() else os.rmdir(link)
        else:
            shutil.rmtree(link)
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
        )
    else:
        link.symlink_to(target, target_is_directory=True)


def recipe_pack_statuses(store: Path, recipe: str) -> tuple[PackStatus, PackStatus]:
    """Inspect the encoder and reranker packs required by an installed recipe.

    Args:
        recipe: Installed recipe identifier.
        store: Local model-store root used for both pack inspections.

    Returns:
        Encoder and reranker status records in that order.

    Raises:
        ModelStoreError: The recipe identifier is unsupported.
    """
    if recipe not in RECIPE_PACKS:
        raise ModelStoreError(f"model_store.recipe_unsupported:{recipe}")
    encoder, reranker = RECIPE_PACKS[recipe]
    return pack_status(store, encoder), pack_status(store, reranker)


def bind_recipe(store: Path, recipe: str, root: Path) -> Path:
    """Link verified local encoder/reranker packs into the recipe binding root.

    Compose `root/{encoder,reranker}` as links to the store's verified packs
    for `recipe`; refused while either pack is not installed and verified.

    Args:
        store: Local store containing verified packs.
        recipe: Installed recipe identifier.
        root: Local binding root for the recipe's encoder/reranker links.

    Returns:
        Root containing verified model-pack bindings.

    Raises:
        ModelStoreError: The recipe is unsupported or a required local pack fails verification.
    """
    encoder, reranker = recipe_pack_statuses(store, recipe)
    for status in (encoder, reranker):
        if status.status != PACK_INSTALLED:
            raise ModelStoreError(
                f"model_store.pack_not_installed:{status.pack.pack_id}:{status.status}"
            )
    root.mkdir(parents=True, exist_ok=True)
    _link_directory(root / "encoder", encoder.directory)
    _link_directory(root / "reranker", reranker.directory)
    return root


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (os.name == "nt" and path.is_junction())


def _unlink_links(root: Path) -> None:
    """Remove a layout's links themselves, never what they point at."""
    for child in root.iterdir() if root.is_dir() else ():
        if _is_link(child):
            child.unlink() if child.is_symlink() else os.rmdir(child)


def _tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0


def bind_retained_recipe(
    store: Path,
    root: Path,
    *,
    packs: tuple[PackDefinition, PackDefinition] | None = None,
) -> Path:
    """Compose the retained recipe's layout from the store's verified packs.

    The retained `hybrid-v2-minilm` reads its encoder at the root and its reranker under
    `reranker/`: each of the encoder's directories and the reranker are links to the store's
    packs, the encoder's loose files (its tokenizer) are copies, since a file takes no
    junction. The bytes the capability hash binds are the packs' own.

    Args:
        store: The application model store.
        root: The layout's directory, created here.
        packs: The encoder and reranker, the retained recipe's unless named.

    Returns:
        The composed root.

    Raises:
        ModelStoreError: A pack is not installed and verified.
    """
    encoder, reranker = (
        recipe_pack_statuses(store, RECIPE_MINILM_CPU)
        if packs is None
        else tuple(pack_status(store, pack) for pack in packs)
    )
    for status in (encoder, reranker):
        if status.status != PACK_INSTALLED:
            raise ModelStoreError(
                f"model_store.pack_not_installed:{status.pack.pack_id}:{status.status}"
            )
    root.mkdir(parents=True, exist_ok=True)
    for top in sorted({name.split("/")[0] for name, _digest in encoder.pack.files}):
        if (encoder.directory / top).is_dir():
            _link_directory(root / top, encoder.directory / top)
        else:
            shutil.copy2(encoder.directory / top, root / top)
    _link_directory(root / "reranker", reranker.directory)
    return root


def retained_copy(root: Path) -> bool:
    """Whether a retained recipe's directory holds its own copy of the encoder, not links."""
    return (root / "onnx").is_dir() and not _is_link(root / "onnx")


def relink_retained_copy(
    store: Path,
    root: Path,
    *,
    packs: tuple[PackDefinition, PackDefinition] | None = None,
) -> int:
    """Replace a retained recipe's own copy with its linked layout, the bytes released.

    Refused unless the store's packs are installed and verified and the copy holds, file by
    file, the bytes they hold, so the capability hash sealed indexes bind does not move. The
    layout is staged beside the copy and swapped in by two renames; an interrupted relink
    leaves the copy or the layout whole, and the next call finishes it.

    Args:
        store: The application model store.
        root: The retained recipe's directory in a workspace.
        packs: The encoder and reranker, the retained recipe's unless named.

    Returns:
        The bytes the copy held, released; 0 when there was no copy.

    Raises:
        ModelStoreError: A pack is missing, the copy's bytes differ, or a reader holds it.
    """
    staging = root.with_name(f"{root.name}.relinking")
    retired = root.with_name(f"{root.name}.retired")
    if retired.exists():
        if not root.exists():
            if not staging.exists():
                os.replace(retired, root)
                return 0
            os.replace(staging, root)
        released = _tree_bytes(retired)
        shutil.rmtree(retired)
        return released
    if staging.exists():
        _unlink_links(staging)
        shutil.rmtree(staging)
    if not retained_copy(root):
        return 0
    encoder, reranker = (
        recipe_pack_statuses(store, RECIPE_MINILM_CPU)
        if packs is None
        else tuple(pack_status(store, pack) for pack in packs)
    )
    for status, held in ((encoder, root), (reranker, root / "reranker")):
        if status.status != PACK_INSTALLED:
            raise ModelStoreError(
                f"model_store.pack_not_installed:{status.pack.pack_id}:{status.status}"
            )
        for name, digest in status.pack.files:
            if not (held / name).is_file() or _sha256(held / name) != digest:
                raise ModelStoreError(f"model_store.retained_copy_differs:{status.pack.pack_id}")
    bind_retained_recipe(store, staging, packs=(encoder.pack, reranker.pack))
    try:
        os.replace(root, retired)
    except OSError as error:
        _unlink_links(staging)
        shutil.rmtree(staging)
        raise ModelStoreError("model_store.retained_copy_in_use") from error
    os.replace(staging, root)
    released = _tree_bytes(retired)
    shutil.rmtree(retired)
    return released


def runtime_availability() -> dict[str, object]:
    """What the GPU runtime looks like on this host, resolved rather than assumed."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return {"torch": None, "cuda_available": False, "device": None}
    device = None
    available = bool(torch.cuda.is_available())
    if available:
        try:
            device = str(torch.cuda.get_device_name(0))
        except Exception:  # a lost device answers nothing usable
            available = False
    return {
        "torch": str(torch.__version__),
        "cuda_available": available,
        "device": device,
        "python": sys.executable,
    }


def recipe_readiness(store: Path, recipe: str) -> dict[str, object]:
    """One recipe's packs and runtime as they stand: what a user sees before choosing."""
    encoder, reranker = recipe_pack_statuses(store, recipe)
    gpu = any(pack.runtime.startswith("TORCH") for pack in (encoder.pack, reranker.pack))
    runtime = (
        runtime_availability() if gpu else {"torch": None, "cuda_available": None, "device": "cpu"}
    )
    ready = (
        encoder.status == PACK_INSTALLED
        and reranker.status == PACK_INSTALLED
        and (not gpu or bool(runtime["cuda_available"]))
    )
    return {
        "recipe": recipe,
        "device": "gpu" if gpu else "cpu",
        "encoder": {
            "pack": encoder.pack.pack_id,
            "repository": encoder.pack.repository,
            "revision": encoder.pack.revision,
            "license": encoder.pack.license_id,
            "approximate_bytes": encoder.pack.approximate_bytes,
            "status": encoder.status,
            "detail": encoder.detail,
        },
        "reranker": {
            "pack": reranker.pack.pack_id,
            "repository": reranker.pack.repository,
            "revision": reranker.pack.revision,
            "license": reranker.pack.license_id,
            "approximate_bytes": reranker.pack.approximate_bytes,
            "status": reranker.status,
            "detail": reranker.detail,
        },
        "runtime": runtime,
        "ready": ready,
    }


__all__ = [
    "BGE_ENCODER",
    "DOWNLOAD_ATTEMPTS",
    "HUB_ENDPOINT",
    "MINILM_ENCODER",
    "MSMARCO_RERANKER",
    "PACKS",
    "PACK_ABSENT",
    "PACK_INSTALLED",
    "PACK_STAGED",
    "PACK_TAMPERED",
    "PARTIAL_SUFFIX",
    "QWEN3_ENCODER",
    "QWEN3_RERANKER",
    "RECIPE_PACKS",
    "Fetcher",
    "ModelStoreError",
    "PackDefinition",
    "PackStatus",
    "Received",
    "bind_recipe",
    "default_store_root",
    "hub_fetcher",
    "install_pack",
    "local_directory_fetcher",
    "pack_directory",
    "pack_status",
    "recipe_pack_statuses",
    "recipe_readiness",
    "runtime_availability",
    "verify_directory",
    "writable_store",
]
