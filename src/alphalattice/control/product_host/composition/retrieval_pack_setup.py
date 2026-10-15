"""Install, import or inspect the retrieval model packs of the supported recipes.

    <retrieval python> scripts/install_retrieval_pack.py --status
    <retrieval python> scripts/install_retrieval_pack.py --install hybrid-v3-bge-small-en --network
    <gpu python>       scripts/install_retrieval_pack.py --install hybrid-v3-qwen3-gpu --network
    <retrieval python> scripts/install_retrieval_pack.py --import hybrid-v2-minilm --from <pack-dir>

(`<retrieval python>` is `.venv-retrieval/Scripts/python.exe`, `<gpu python>`
is `.venv-gpu/Scripts/python.exe`.)

Packs live in the application model store (`--store`, else
`ALPHALATTICE_MODEL_STORE`, else the platform's application-data directory),
never in the Git checkout or a workspace: a store there, or one this process
cannot write, is refused by name, `--store <directory>` the way on, and every
other failure is answered in words too (V539). `--install` fetches the pinned
revision's files from the Hub over HTTPS only with `--network` (an
explicitly admitted setup step; the product's `ALPHALATTICE_NETWORK_DISABLED`
stays as it is for everything else), printing the bytes as they arrive;
`--import` copies from a directory already holding the files. Every file is
verified against its pinned hash as it arrives and the pack is renamed into
place only when complete, so a cancelled or failed install is never an
installed pack -- and running the same `--install` again continues it where
it stopped (the files already staged, then the rest of a partial file). Nothing is uploaded,
no repository code runs, and installing changes no workspace: binding a
recipe into a workspace is `materialize_evidence_cro_authority.py --recipe`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alphalattice.interface.local_application.failure_codes import setup_failure
from alphalattice.kernel.knowledge import model_store
from alphalattice.kernel.knowledge.hybrid_contracts import SUPPORTED_RECIPES
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root

PLAYPEN = resolve_playpen_root(Path(__file__))


def _outside_checkout_and_workspaces(store: Path) -> Path:
    """The store, refused by name when it lies in the checkout or in a workspace (V539).

    Packs stay outside both: the checkout is code, a workspace binds a pack by a link to the
    store. A store named by `--store` or `ALPHALATTICE_MODEL_STORE` keeps that boundary as the
    default one does.
    """
    # The workspace's owner names its manifest; imported here, so `--status` stays quick.
    from alphalattice.control.product_host.composition.research_workspace import (
        RESEARCH_WORKSPACE_MANIFEST_NAME,
    )

    if store.is_relative_to(PLAYPEN):
        raise model_store.ModelStoreError("model_store.store_inside_checkout")
    if any((path / RESEARCH_WORKSPACE_MANIFEST_NAME).is_file() for path in (store, *store.parents)):
        raise model_store.ModelStoreError("model_store.store_inside_workspace")
    return store


_ELSEWHERE = (
    "Install into a directory this process can write, outside the checkout and any workspace, "
    "with `--store <directory>`, then give the Host and the Evidence setup that directory "
    "(ALPHALATTICE_MODEL_STORE, or the setup's `--model-store`)."
)
_AGAIN = "Every file already verified is kept, so the same command continues where it stopped."
_WORDS: dict[str, tuple[str, str]] = {
    "model_store.store_unwritable": (
        "This process cannot create or write the model store, so no pack was installed. "
        + _ELSEWHERE,
        "INSTALL_INTO_A_WRITABLE_STORE",
    ),
    "model_store.store_inside_checkout": (
        "Model packs stay outside the checkout, and this store lies inside it, so no pack was "
        "installed. " + _ELSEWHERE,
        "INSTALL_INTO_A_STORE_OUTSIDE_THE_CHECKOUT_AND_WORKSPACES",
    ),
    "model_store.store_inside_workspace": (
        "Model packs stay outside every workspace, which binds a pack by a link to the store, and "
        "this store lies inside one, so no pack was installed. " + _ELSEWHERE,
        "INSTALL_INTO_A_STORE_OUTSIDE_THE_CHECKOUT_AND_WORKSPACES",
    ),
    "model_store.install_interrupted": (
        "The installation stopped on a file or connection error ({subject}). " + _AGAIN + " If it "
        "stops the same way, check the store's free space and the connection.",
        "RUN_THE_SAME_INSTALL_AGAIN",
    ),
    "model_store.download_refused": (
        "The Hub refused a pack file ({subject}), so the pack was not installed. " + _AGAIN,
        "RUN_THE_SAME_INSTALL_AGAIN",
    ),
    "model_store.file_identity_invalid": (
        "A pack file arrived with another hash than its pinned one ({subject}); it was removed "
        "and the pack was not installed. " + _AGAIN + " If it recurs, report the file.",
        "RUN_THE_SAME_INSTALL_AGAIN",
    ),
    "model_store.pack_incomplete": (
        "The staged pack does not hold exactly its pinned files ({subject}), so it was not "
        "installed. " + _AGAIN,
        "RUN_THE_SAME_INSTALL_AGAIN",
    ),
    "model_store.install_cancelled": (
        "The installation was cancelled before the pack was complete. " + _AGAIN,
        "RUN_THE_SAME_INSTALL_AGAIN",
    ),
    "model_store.import_source_missing": (
        "The directory `--from` names lacks a pack file ({subject}), so nothing was imported; "
        "import from a directory holding the pack's files, or install it with "
        "`--install <recipe> --network`.",
        "IMPORT_FROM_A_DIRECTORY_HOLDING_THE_PACK",
    ),
}
"""Each refusal the installer meets, in words with its way on (V539): its own, since it never
reaches the Host."""


def pack_command(*arguments: str) -> list[str]:
    """This installer with these arguments, as the person runs it (V590).

    The checkout's script, or the installed module.
    """
    if (PLAYPEN / "pyproject.toml").is_file():
        return [sys.executable, "scripts/install_retrieval_pack.py", *arguments]
    module = "alphalattice.control.product_host.composition.retrieval_pack_setup"
    return [sys.executable, "-m", module, *arguments]


def _status_command(store: Path) -> list[str]:
    """The installer's read-only check of this store, as the person runs it."""
    return pack_command("--status", "--store", str(store))


def _refused(code: str, store: Path, error: Exception) -> int:
    """Print a refusal with its words and way on, the store it names beside it (V539)."""
    base, _sep, subject = code.partition(":")
    detail, next_action = _WORDS.get(
        base,
        (
            "The installer refused with `{subject}`; nothing unverified was installed. If the "
            "same command refuses the same way again, report the code.",
            "RUN_THE_SAME_INSTALL_AGAIN",
        ),
    )
    refusal = {
        **setup_failure(error),
        "status": "REFUSED",
        "failure_code": code,
        "detail": detail.replace("{subject}", subject if base in _WORDS else code),
        "next_action": next_action,
        "store": str(store),
        "next_commands": {"status": _status_command(store)},
    }
    print(json.dumps(refusal, indent=1))
    return 2


def _status(store: Path) -> dict[str, object]:
    return {
        "store": str(store),
        "recipes": {
            recipe: model_store.recipe_readiness(store, recipe) for recipe in SUPPORTED_RECIPES
        },
        "claim": "Pack presence and hashes; no retrieval quality is implied by a READY pack.",
    }


def _install(store: Path, recipe: str, *, network: bool, source: Path | None) -> dict[str, object]:
    if recipe not in model_store.RECIPE_PACKS:
        raise SystemExit(f"unsupported recipe: {recipe}; one of {', '.join(SUPPORTED_RECIPES)}")
    if source is None and not network:
        raise SystemExit("--install needs --network (an admitted download) or --from <directory>")
    results = {}
    encoder, reranker = model_store.RECIPE_PACKS[recipe]
    for pack in (encoder, reranker):
        if source is not None:
            # A pack directory as a workspace holds it: the encoder at the
            # root (the retained layout) or under `encoder/`, the reranker
            # under `reranker/`. A pack the directory does not hold is left
            # as it stands and reported.
            candidates = (source, source / "encoder") if pack is encoder else (source / "reranker",)
            found = next(
                (c for c in candidates if model_store.verify_directory(c, pack) is None), None
            )
            if found is None:
                results[pack.pack_id] = model_store.pack_status(store, pack).status
                continue
            fetch = model_store.local_directory_fetcher(found)
        else:
            fetch = model_store.hub_fetcher(received=_bytes_printer(pack.pack_id))

        def progress(name: str, index: int, total: int, pack_id: str = pack.pack_id) -> None:
            """Read the captured source document under its admitted contract."""
            print(f"{pack_id}: {index}/{total} {name}", flush=True)

        status = model_store.install_pack(store, pack, fetch, progress=progress)
        results[pack.pack_id] = status.status
    if model_store.PACK_INSTALLED in results.values():
        model_store.remember_store(store)
    return {
        "store": str(store),
        "recipe": recipe,
        "packs": results,
        **model_store.recipe_readiness(store, recipe),
    }


def _bytes_printer(pack_id: str) -> model_store.Received:
    """A line per file at about every 2% of it (and its last byte)."""
    shown: dict[str, int] = {}

    def received(name: str, done: int, total: int | None) -> None:
        """Read the captured source document under its admitted contract."""
        step = max((total or 0) // 50, 4 << 20)
        if done - shown.get(name, -step) >= step or done == total:
            shown[name] = done
            whole = f" of {total / 1e6:.1f}" if total else ""
            print(f"{pack_id}: {name} {done / 1e6:.1f}{whole} MB", flush=True)

    return received


def main(argv: list[str] | None = None) -> int:
    """Run the declared command and return its exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--store", type=Path, help="The application model store (default: the platform's)."
    )
    parser.add_argument(
        "--status", action="store_true", help="Every recipe's packs and runtime as they stand."
    )
    parser.add_argument(
        "--install", metavar="RECIPE", help="Fetch the recipe's packs at their pinned revisions."
    )
    parser.add_argument(
        "--import",
        dest="import_recipe",
        metavar="RECIPE",
        help="Import the recipe's encoder pack from --from.",
    )
    parser.add_argument(
        "--from", dest="source", type=Path, help="A directory holding a pack's files, for --import."
    )
    parser.add_argument(
        "--network", action="store_true", help="Admit the Hub download for --install."
    )
    arguments = parser.parse_args(argv)
    store = (arguments.store or model_store.default_store_root()).resolve()
    try:
        if arguments.install:
            payload = _install(
                _outside_checkout_and_workspaces(store),
                arguments.install,
                network=arguments.network,
                source=None,
            )
        elif arguments.import_recipe:
            if arguments.source is None:
                raise SystemExit("--import needs --from <directory>")
            payload = _install(
                _outside_checkout_and_workspaces(store),
                arguments.import_recipe,
                network=False,
                source=arguments.source,
            )
        else:
            payload = _status(store)
    except model_store.ModelStoreError as error:
        return _refused(str(error), store, error)
    except OSError as error:
        # A file or connection error past the store's own checks: what was verified is kept.
        return _refused(f"model_store.install_interrupted:{type(error).__name__}", store, error)
    except Exception as error:
        print(
            json.dumps(
                {
                    **setup_failure(error),
                    "store": str(store),
                    "next_commands": {"status": _status_command(store)},
                }
            )
        )
        return 2
    print(json.dumps(payload, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
