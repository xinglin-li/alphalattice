"""The application model store installs a pack whole or not at all.

A pack is staged file by file, each verified against its pinned hash as it
arrives, and renamed into place only when the whole closure is present and
nothing else is; a cancelled, failed or tampered install is never a partly
installed pack a recipe could be bound to -- what it staged is kept and the
next install continues it, a byte range at a time. A recipe is bound into a
workspace only from installed, verified packs, as links to the store's one
physical copy; the store's location is the user's, never Git.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from alphalattice.kernel.knowledge import model_store
from alphalattice.kernel.knowledge.hybrid_contracts import RECIPE_QWEN3_GPU


def _pack(
    tmp_path: Path, files: dict[str, bytes], name: str = "test-pack"
) -> model_store.PackDefinition:
    manifest = tuple(
        sorted((file, hashlib.sha256(payload).hexdigest()) for file, payload in files.items())
    )
    source = tmp_path / "source" if name == "test-pack" else tmp_path / f"source-{name}"
    for file, payload in files.items():
        path = source / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return model_store.PackDefinition(
        name, f"tests/{name}", "0" * 40, manifest, "MIT", "ONNX_CPU", 42
    )


def test_a_pack_is_installed_whole_or_not_at_all(tmp_path: Path) -> None:
    store = tmp_path / "store"
    pack = _pack(tmp_path, {"onnx/model.onnx": b"graph", "tokenizer.json": b"{}"})
    assert model_store.pack_status(store, pack).status == model_store.PACK_ABSENT
    seen: list[tuple[str, int, int]] = []
    status = model_store.install_pack(
        store,
        pack,
        model_store.local_directory_fetcher(tmp_path / "source"),
        progress=lambda name, index, total: seen.append((name, index, total)),
    )
    assert status.status == model_store.PACK_INSTALLED
    assert seen == [("onnx/model.onnx", 1, 2), ("tokenizer.json", 2, 2)]
    assert sorted(p.name for p in status.directory.rglob("*") if p.is_file()) == [
        "model.onnx",
        "tokenizer.json",
    ]
    # Installed again: no work, the same status.
    again = model_store.install_pack(
        store, pack, lambda *_args: (_ for _ in ()).throw(AssertionError("must not fetch"))
    )
    assert again.status == model_store.PACK_INSTALLED

    # Tampering after the install is seen, and named.
    (status.directory / "tokenizer.json").write_bytes(b'{"model_max_length": 16}')
    tampered = model_store.pack_status(store, pack)
    assert tampered.status == model_store.PACK_TAMPERED
    assert tampered.detail == "hash differs for tokenizer.json"
    (status.directory / "notes.txt").write_bytes(b"x")
    (status.directory / "tokenizer.json").write_bytes(b"{}")
    assert model_store.pack_status(store, pack).detail == "unadmitted file notes.txt"


def test_an_interrupted_install_is_continued_and_a_bad_file_is_refused(tmp_path: Path) -> None:
    """An interrupted install is continued and a bad file is refused."""

    store = tmp_path / "store"
    pack = _pack(tmp_path, {"onnx/model.onnx": b"graph", "tokenizer.json": b"{}"})
    fetch = model_store.local_directory_fetcher(tmp_path / "source")
    with pytest.raises(model_store.ModelStoreError, match="install_cancelled"):
        model_store.install_pack(store, pack, fetch, cancelled=lambda: True)
    assert model_store.pack_status(store, pack).status == model_store.PACK_STAGED

    (tmp_path / "source" / "tokenizer.json").write_bytes(b"other bytes")
    with pytest.raises(model_store.ModelStoreError, match=r"file_identity_invalid:tokenizer\.json"):
        model_store.install_pack(store, pack, fetch)
    staged = model_store.pack_status(store, pack)
    assert staged.status == model_store.PACK_STAGED
    assert staged.detail is not None and "installing again continues it" in staged.detail
    staging = store / pack.directory_name / f".staging-{pack.revision}"
    assert sorted(p.name for p in staging.rglob("*") if p.is_file()) == ["model.onnx"]

    (tmp_path / "source" / "tokenizer.json").write_bytes(b"{}")
    (staging / "notes.txt").write_bytes(b"not the pack's")
    fetched: list[str] = []

    def counting(value: model_store.PackDefinition, name: str, directory: Path) -> Path:
        fetched.append(name)
        return fetch(value, name, directory)

    status = model_store.install_pack(store, pack, counting)
    assert status.status == model_store.PACK_INSTALLED
    assert fetched == ["tokenizer.json"], "the verified staged file was kept, not fetched again"

    missing = _pack(tmp_path / "other", {"model.bin": b"weights"})
    (tmp_path / "other" / "source" / "model.bin").unlink()
    with pytest.raises(model_store.ModelStoreError, match="import_source_missing"):
        model_store.install_pack(
            tmp_path / "store-missing",
            missing,
            model_store.local_directory_fetcher(tmp_path / "other" / "source"),
        )
    assert (
        model_store.pack_status(tmp_path / "store-missing", missing).status
        == model_store.PACK_STAGED
    )


def test_a_broken_download_continues_from_its_last_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken download continues from its last byte."""

    import httpx

    monkeypatch.setattr(model_store.time, "sleep", lambda _seconds: None)
    payload = bytes(range(256)) * 64
    pack = model_store.PackDefinition(
        "test-pack",
        "tests/test-pack",
        "1" * 40,
        (("onnx/model.onnx", hashlib.sha256(payload).hexdigest()),),
        "MIT",
        "ONNX_CPU",
        len(payload),
    )
    ranges: list[str | None] = []
    breaks = {"left": 1}

    class _Broken(httpx.SyncByteStream):
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __iter__(self):  # type: ignore[no-untyped-def]
            yield self.body[: len(self.body) // 3]
            raise httpx.ReadError("the line dropped")

    def serve(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/tests/test-pack/resolve/{'1' * 40}/onnx/model.onnx"
        wanted = request.headers.get("range")
        ranges.append(wanted)
        start = int(wanted.split("=")[1].rstrip("-")) if wanted else 0
        body = payload[start:]
        headers = {"content-length": str(len(body))}
        if start:
            headers["content-range"] = f"bytes {start}-{len(payload) - 1}/{len(payload)}"
        if breaks["left"]:
            breaks["left"] -= 1
            return httpx.Response(206 if start else 200, headers=headers, stream=_Broken(body))
        return httpx.Response(206 if start else 200, headers=headers, content=body)

    received: list[tuple[str, int, int | None]] = []
    client = httpx.Client(transport=httpx.MockTransport(serve))
    fetch = model_store.hub_fetcher(received=lambda *value: received.append(value), client=client)
    status = model_store.install_pack(tmp_path / "store", pack, fetch)
    assert status.status == model_store.PACK_INSTALLED
    third = len(payload) // 3
    assert ranges == [None, f"bytes={third}-"], "the second request asked only for the rest"
    assert received[-1] == ("onnx/model.onnx", len(payload), len(payload))
    assert all(total == len(payload) for _name, _done, total in received)

    # A partial left by an earlier install is continued by the next one.
    other = tmp_path / "later"
    staging = other / pack.directory_name / f".staging-{pack.revision}" / "onnx"
    staging.mkdir(parents=True)
    (staging / "model.onnx.part").write_bytes(payload[:1000])
    ranges.clear()
    model_store.install_pack(other, pack, fetch)
    assert ranges == ["bytes=1000-"]
    assert model_store.pack_status(other, pack).status == model_store.PACK_INSTALLED


def test_a_recipe_binds_only_from_installed_verified_packs(tmp_path: Path) -> None:
    store = tmp_path / "store"
    with pytest.raises(
        model_store.ModelStoreError, match=r"pack_not_installed:qwen3-embedding-0\.6b:ABSENT"
    ):
        model_store.bind_recipe(store, RECIPE_QWEN3_GPU, tmp_path / "workspace" / "semantic-model")
    with pytest.raises(model_store.ModelStoreError, match="recipe_unsupported"):
        model_store.bind_recipe(store, "hybrid-v9-free-pairing", tmp_path / "workspace" / "x")
    readiness = model_store.recipe_readiness(store, RECIPE_QWEN3_GPU)
    assert readiness["ready"] is False and readiness["device"] == "gpu"
    assert readiness["encoder"]["status"] == model_store.PACK_ABSENT  # type: ignore[index]


def test_the_store_root_is_the_users(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHALATTICE_MODEL_STORE", str(tmp_path / "elsewhere"))
    assert model_store.default_store_root() == tmp_path / "elsewhere"
    monkeypatch.delenv("ALPHALATTICE_MODEL_STORE")
    root = model_store.default_store_root()
    assert "AlphaLattice" in str(root) or "alphalattice" in str(root)
    assert not str(root).startswith(str(Path(__file__).resolve().parents[2]))
    assert os.name != "nt" or root.parts[-2:] == ("AlphaLattice", "models")


def test_a_retained_copy_is_linked_to_the_store_only_when_it_holds_the_packs_bytes(
    tmp_path: Path,
) -> None:
    """A retained copy links to the store only when it holds the pack's exact bytes."""

    encoder_files = {"onnx/model.onnx": b"encoder graph", "sentencepiece.bpe.model": b"spm"}
    reranker_files = {"config.json": b"{}", "onnx/model.onnx": b"reranker graph"}
    encoder = _pack(tmp_path, encoder_files, "encoder")
    reranker = _pack(tmp_path, reranker_files, "reranker")
    store = tmp_path / "store"
    for pack, name in ((encoder, "encoder"), (reranker, "reranker")):
        fetcher = model_store.local_directory_fetcher(tmp_path / f"source-{name}")
        assert model_store.install_pack(store, pack, fetcher).status == model_store.PACK_INSTALLED

    def copy(root: Path, *, differ: bool = False) -> Path:
        for file, payload in encoder_files.items():
            (root / file).parent.mkdir(parents=True, exist_ok=True)
            (root / file).write_bytes(payload + (b"!" if differ else b""))
        for file, payload in reranker_files.items():
            (root / "reranker" / file).parent.mkdir(parents=True, exist_ok=True)
            (root / "reranker" / file).write_bytes(payload)
        (root / ".cache" / "note").parent.mkdir(parents=True)
        (root / ".cache" / "note").write_bytes(b"download metadata")
        return root

    root = copy(tmp_path / "workspace" / "semantic-model")
    assert model_store.retained_copy(root)
    released = model_store.relink_retained_copy(store, root, packs=(encoder, reranker))
    assert released == sum(map(len, (*encoder_files.values(), *reranker_files.values()))) + 17
    assert not model_store.retained_copy(root)
    assert (root / "onnx" / "model.onnx").read_bytes() == b"encoder graph"
    assert (root / "reranker" / "onnx" / "model.onnx").read_bytes() == b"reranker graph"
    assert (root / "sentencepiece.bpe.model").read_bytes() == b"spm"
    assert not (root / ".cache").exists()
    assert model_store.pack_status(store, encoder).status == model_store.PACK_INSTALLED
    assert model_store.relink_retained_copy(store, root, packs=(encoder, reranker)) == 0

    differing = copy(tmp_path / "other" / "semantic-model", differ=True)
    with pytest.raises(model_store.ModelStoreError, match="retained_copy_differs:encoder"):
        model_store.relink_retained_copy(store, differing, packs=(encoder, reranker))
    assert model_store.retained_copy(differing)

    # Cut short between the two renames: the copy retired, the layout staged, no root.
    cut = copy(tmp_path / "cut" / "semantic-model")
    staging = cut.with_name("semantic-model.relinking")
    model_store.bind_retained_recipe(store, staging, packs=(encoder, reranker))
    cut.rename(cut.with_name("semantic-model.retired"))
    assert model_store.relink_retained_copy(store, cut, packs=(encoder, reranker)) > 0
    assert (cut / "onnx" / "model.onnx").read_bytes() == b"encoder graph"
    assert not cut.with_name("semantic-model.retired").exists() and not staging.exists()
