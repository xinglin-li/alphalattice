"""The lawful ways on when a book's sources fall short, and the short-source action.

Official SEC acquisition, the person's decision or, at the default budget, a first use's delegated
agent's own step; under the person's daily yes, the install the Host offers; and a recorded package
only where one covers every unit's issuers. The review
application answers them; the projection shows them beside a unit that failed for its sources.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from alphalattice.control.workspace_runtime.network_access import network_access, persons_stop
from alphalattice.evidence.alternative_evidence.sources.admission import DEFAULT_SOURCE_CONSENT
from alphalattice.interface.local_application.cli_contract import join, refusal_words, shell
from alphalattice.interface.local_application.retrieval_environment import fill_command
from alphalattice.kernel.knowledge import model_store
from alphalattice.kernel.knowledge.hybrid_contracts import RECIPE_MINILM_CPU
from alphalattice.kernel.shared_kernel.project_layout import command_prefix, resolve_playpen_root

if TYPE_CHECKING:
    from datetime import datetime

    from alphalattice.control.task_control.contracts import TaskInputEnvelope
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
        BookSelector,
        SealedBook,
    )

SHORT_SOURCE_ACTION: Final = {
    False: "ASK_FOR_OFFICIAL_ACQUISITION_OR_A_COVERING_PACKAGE",
    True: "ACQUIRE_UNDER_THE_FIRST_USE_DELEGATION",
}
"""A book short of sources: the person's decision, or the default budget a first use's agent
gives under its delegation (by whether one is active)."""


def short_source_action(ways: Mapping[str, object]) -> str:
    """A short book's way on: the delegated default budget when its source ways offer it."""
    source = ways.get("source_ways")
    return SHORT_SOURCE_ACTION[isinstance(source, Mapping) and "next_requests" in source]


def _authored(book: BookSelector | None) -> bool:
    """Whether a book is one a study authored, whose research input the setup names."""
    return book is not None and book.experiment_task_id is not None


SETUP_OPTIONS: dict[str, str] = {
    "accessions": (
        "up to four exact accession numbers of the one issuer --entities names, in the official "
        "##########-##-###### form its own submissions index holds"
    ),
    "entities": (
        "the issuers to acquire or import, by ticker, at most eight, each in the workspace's "
        "universe and, for an import, in the source's request"
    ),
    "evidence_as_of": (
        "a timezone-aware ISO 8601 cutoff no later than now, the filings selected being those the "
        "official source accepted by then; the run's clock unless given"
    ),
    "maximum_documents_per_issuer": (
        "from 3 to 20 filings each issuer may contribute, at most 24 across the issuers named; 3 "
        "unless given"
    ),
    "minimum_entity_coverage": (
        "the package's floor: the share, from 0 to 1, of its counted issuers that must hold a "
        "document; 0.60 unless given, the installed one on a rebind"
    ),
    "model_name": (
        "an optional managed analyst and reviewer model, named with its --deepseek-base-url, or "
        "neither for native analysis"
    ),
    "research_input_hash": (
        "that input's exact sealed binding, also named by `study show <id>`, never inferred from "
        "the latest data"
    ),
    "research_input_id": (
        "the admitted research input the book's study was built on, as `study show <id>` names it"
    ),
    "semantic_model": (
        "the retained recipe's pack directory inside the workspace, the installed one on a "
        "rebind; without it the recipe is bound from the application model store"
    ),
    "source_artifact_root": (
        "the alternative-evidence artifact store of the workspace that acquired the source, "
        "runtime/artifacts/alternative-evidence under that workspace, which holds "
        "source-document-sets/, snapshots/, registries/ and requests/"
    ),
    "source_knowledge_root": (
        "the evidence knowledge store of that workspace, runtime/evidence-knowledge under it, "
        "which holds the source objects a reference-form source set names; by default the one "
        "beside the artifact store's runtime/artifacts"
    ),
    "source_set_hash": (
        "the 64-character lowercase hexadecimal identity of a source set under that store, as "
        "the acquisition's answer names it"
    ),
}
"""What each option of the Evidence setup must hold, by its name in the answer: the one owner the
setup's offer states and its refusals name."""


def _holds(*options: str) -> str:
    """What the offered options must hold, each from `SETUP_OPTIONS`."""
    return (
        "; ".join(
            f"--{option.replace('_', '-')} names {SETUP_OPTIONS[option]}" for option in options
        )
        + "."
    )


def evidence_setup(
    workspace: Path, *, authored: bool = False, rebind: bool = False
) -> dict[str, object]:
    """The commands that complete the Evidence setup.

    Each is written as a person types it at the product folder, its interpreter named. The
    retrieval pack first (the authority binds the recipe from the application model
    store), then the authority's check and installation in the declared retrieval environment,
    as the Evidence handoff's first-package steps run them. What the Host knows is filled in:
    the workspace, the recipe, the interpreters and whether the retrieval environment exists.
    What only a person decides stays open in ``choose``: the issuers, a recorded package in place
    of the SEC acquisition, and for a book a study authored the research input it was built on.
    A workspace installed under a retired matter selection is rebound (``rebind``).

    Args:
        workspace: The workspace's folder.
        authored: Whether the book is one a study authored.
        rebind: Whether the installed authority is rebound under the integrated selection.

    Returns:
        The setup's ``pack`` and ``authority`` steps.
    """
    windows = os.name == "nt"
    python = ".venv/Scripts/python.exe" if windows else ".venv/bin/python"
    retrieval = ".venv-retrieval/Scripts/python.exe" if windows else ".venv-retrieval/bin/python"
    installed = not (resolve_playpen_root(Path(__file__)) / "pyproject.toml").is_file()
    if installed:
        python = join([sys.executable], shell())
        retrieval = python
    # The running Host installs a package as a Task (its setup script is the path with no Host).
    host = f'{" ".join(command_prefix())} --workspace "{workspace}" evidence install --setup='
    if rebind:
        return {
            "authority": {
                "install": f'{host}"--rebind-installed --matter-selection '
                'INTEGRATED_TOPIC_ROUTING --install"',
                "before": "The running Host installs it as a Task; nothing is stopped.",
            }
        }
    store, packs = model_store.default_store_root(), model_store.RECIPE_PACKS[RECIPE_MINILM_CPU]
    pack: dict[str, object] = {
        "recipe": RECIPE_MINILM_CPU,
        # Where the packs are read from, whether they are there and what a download takes;
        # the setup verifies them by hash.
        "store": str(store),
        "present": all(model_store.pack_directory(store, value).is_dir() for value in packs),
        "download_bytes": sum(value.approximate_bytes for value in packs),
        "check": f"{retrieval} scripts/install_retrieval_pack.py --status",
        "install": f"{retrieval} scripts/install_retrieval_pack.py --install "
        f"{RECIPE_MINILM_CPU} --network",
    }
    if installed:
        from alphalattice.control.product_host.composition.retrieval_pack_setup import pack_command

        pack["check"] = join(pack_command("--status"), shell())
        pack["install"] = join(pack_command("--install", RECIPE_MINILM_CPU, "--network"), shell())
    if installed or not (resolve_playpen_root(Path(__file__)) / retrieval).is_file():
        pack["environment"] = join(fill_command(), shell())  # the one command, on either leg
    scope = " ".join(
        [
            *(["--research-input-id <id> --research-input-hash <hash>"] if authored else []),
            "--acquire-sec --entities <TICKER> ...",
        ]
    )
    # Each choice states what its options must hold, from the table its refusals name.
    choose = [
        {"arg": "--entities", "why": _holds("entities")},
        {
            "arg": "--source-artifact-root <dir> --source-set-hash <hash>",
            "why": "In place of --acquire-sec and --network-consent, SEC artifacts already "
            "captured, read offline: " + _holds("source_artifact_root", "source_set_hash"),
        },
    ]
    if authored:
        choose.append(
            {
                "arg": "--research-input-id <id> --research-input-hash <hash>",
                "why": _holds("research_input_id", "research_input_hash"),
            }
        )
    return {
        "pack": pack,
        "authority": {
            "check": f'{host}"{scope} --preflight"',
            "install": f'{host}"{scope} --network-consent --install"',
            "choose": choose,
            "before": "The running Host installs it as a Task; nothing is stopped. The "
            "acquisition names itself to the SEC by the product's own contact.",
        },
    }


INSTALL_SCOPE: Final = 8
"""An offered install's issuers: the install's own scope, at three filings each 24 documents."""


def _ranked(workspace: Path, book: SealedBook | None, *, authored: bool) -> list[str] | str:
    """The book's holdings by weight, largest first, as tickers the source can serve.

    Or the key of why the Host names none (`NO_OFFER`).

    A holding counts when the workspace's universe carries it and, where an acquisition already
    holds the SEC issuer registry, the registry names it (BRK.B is not BRK-B).
    """
    from alphalattice.control.product_host.composition.portfolio_result_context import (
        universe_listing_ids,
    )
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import holding_weights

    if authored or book is None:
        return "authored" if authored else "no_book"
    try:
        symbols = {listing: symbol for symbol, listing in universe_listing_ids(workspace).items()}
    except (ValueError, OSError):
        return "no_universe"
    registered = _registered(workspace)
    weights = holding_weights(book).items()
    served = {
        listing: weight
        for listing, weight in weights
        if weight > 0
        and listing in symbols
        and (registered is None or symbols[listing] in registered)
    }
    ranked = [symbols[listing] for listing in sorted(served, key=lambda key: (-served[key], key))]
    return ranked or "no_holding"


def _registered(workspace: Path) -> frozenset[str] | None:
    """The SEC issuer registry's tickers, where an acquisition here already holds it."""
    root = workspace / "runtime" / "artifacts" / "alternative-evidence" / "registries"
    held = sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not held:
        return None
    entries = json.loads(held[-1].read_text(encoding="utf-8")).get("entries") or ()
    return frozenset(str(entry.get("ticker")) for entry in entries)


def _install(ranked: list[str], after: Sequence[str]) -> dict[str, object]:
    """The offered install: the next `INSTALL_SCOPE` past the issuers `after` named, quoted."""
    tried = set(after[after.index("--entities") + 1 :]) if "--entities" in after else set()
    start = 1 + max((at for at, symbol in enumerate(ranked) if symbol in tried), default=-1)
    named = " ".join(shlex.quote(symbol) for symbol in ranked[start : start + INSTALL_SCOPE])
    setup = f"--acquire-sec --entities {named} --network-consent --install"
    return {"install": {"operation": "EVIDENCE_INSTALL", "evidence_setup": setup}} if named else {}


NO_OFFER: Final = {
    "authored": "No install is offered: a study authored this book, so its install also names "
    "that study's research input, the agent's to name; fill the setup's `install` from its "
    "`choose`.",
    "no_book": "No install is offered: no sealed book is open to name issuers from; select the "
    "book, then ask again.",
    "no_universe": "No install is offered: the Host cannot read this workspace's universe; "
    "`workspace show` reads its data state.",
    "no_holding": "No install is offered: no holding of this book is a ticker the source serves; "
    "name issuers in the setup's `install` from its `choose`.",
    "tried": "No next install is offered: every holding the source serves was named in an install "
    "whose issuers filed nothing in the window; acquire again at a later cutoff.",
}
"""Why no install is offered, and the way on: said in place of the offer."""


def install_offer(
    workspace: Path, book: SealedBook | None, *, authored: bool, after: Sequence[str] = ()
) -> dict[str, object]:
    """The install a book without its Evidence offers ready, its issuers named by the Host's rule.

    The rule: the book's largest holdings by weight in the workspace's universe, at most
    `INSTALL_SCOPE` at three filings each (24 documents), within the default budget; one install
    suffices, the review acquiring the rest. `after` is the argv of an install whose issuers all
    filed nothing: the next eight past its last are offered. None is offered for the reasons
    `NO_OFFER` words.
    """
    ranked = _ranked(workspace, book, authored=authored)
    return {} if isinstance(ranked, str) else _install(ranked, after)


def _offer_or_why(
    workspace: Path, opened: SealedBook | None, book: BookSelector | None, after: Sequence[str] = ()
) -> tuple[dict[str, object], dict[str, str]]:
    """The install offered, as `next_requests` holds it, or the words saying why none is."""
    ranked = _ranked(workspace, opened, authored=_authored(book))
    offer = {} if isinstance(ranked, str) else _install(ranked, after)
    why = ranked if isinstance(ranked, str) else None if offer else "tried"
    return offer, {} if why is None else {"detail": NO_OFFER[why]}


def offered_installs(
    workspace: Path,
    opened: SealedBook | None,
    book: BookSelector | None,
    quiet: Sequence[Sequence[str]],
) -> list[list[str]]:
    """Every install the Host offers now, as argv, for the selected book.

    The first, and the next past each install whose issuers all filed nothing (`QUIET_INSTALL`):
    the only installs an agent sends under the person's daily yes (`install_refusal`).
    """
    ranked = _ranked(workspace, opened, authored=_authored(book))
    if isinstance(ranked, str):
        return []
    offers = (_install(ranked, after) for after in [[], *quiet])
    lines = (cast(dict[str, str], offer["install"])["evidence_setup"] for offer in offers if offer)
    return [shlex.split(line) for line in lines]


QUIET_INSTALL: Final = "evidence_review.materialization_has_no_admitted_documents"
"""An install stopped because each issuer it named filed nothing in the window."""


SOURCE_MISSING: Final = "REFUSED_NO_ADMITTED_EVIDENCE_AUTHORITY"
"""A review refused for want of its issuer registry and listing authority, which an install
admits."""


def quiet_install(
    way: dict[str, Any],
    workspace: Path,
    opened: SealedBook | None,
    book: BookSelector | None,
    stopped: TaskInputEnvelope,
) -> dict[str, Any]:
    """The way on of an install whose issuers each filed nothing in the window (`QUIET_INSTALL`).

    The next eight by weight past them, ready (`install_offer`).
    """
    offer, why = _offer_or_why(workspace, opened, book, stopped.payload["arguments"])
    return {**way, **why, "next_requests": {**way.get("next_requests", {}), **offer}}


def adapter_absent(
    workspace: Path, book: BookSelector, opened: SealedBook | None
) -> dict[str, object]:
    """A preview with no source package or retrieval runtime: its setup and the install ready.

    Where none is offered, the words say why (`NO_OFFER`).
    """
    offer, why = _offer_or_why(workspace, opened, book)
    return {
        "status": "EVIDENCE_PREREQUISITES_MISSING",
        "failure_code": "product_host.evidence_task_adapter_absent",
        **why,
        "next_action": "ADMIT_SOURCE_PACKAGE_AND_PINNED_RETRIEVAL_RUNTIME",
        "next_requests": offer,
        "setup": evidence_setup(workspace, authored=_authored(book)),
        "claim": "Source and retrieval authority are the workspace's admitted packages.",
    }


def documents_absent(book: BookSelector | None) -> dict[str, object]:
    """The documents read with no Evidence installed: the dated book's preview, offered.

    The preview offers the install, so its issuers are that book's.
    """
    selected = {} if book is None else book.request_fields()
    return {
        "status": "EVIDENCE_PREREQUISITES_MISSING",
        "failure_code": "product_host.evidence_task_adapter_absent",
        "next_action": "ADMIT_SOURCE_PACKAGE_AND_PINNED_RETRIEVAL_RUNTIME",
        "next_requests": {"preview": {"operation": "EVIDENCE_PREVIEW", **selected}},
    }


def prerequisites_missing(
    workspace: Path,
    disposition: str,
    detail: str,
    *,
    next_requests: Mapping[str, dict[str, str]] | None,
    book: BookSelector | None,
    opened: SealedBook | None,
    listings: int | None,
    scope: Callable[[int], object],
) -> dict[str, object]:
    """A preview refused before any source is admitted, with its way on.

    For want of the authority, the setup's commands and the install ready by the Host's rule
    (`install_offer`), with the book's acquisition `scope` over its `listings`; otherwise the
    setup script's help; the page reads `setup`, its fallback `setup_help`.
    """
    missing = disposition == SOURCE_MISSING
    offer, why = _offer_or_why(workspace, opened, book) if missing else ({}, {})
    help_line = (
        "scripts/materialize_evidence_cro_authority.py --help"
        if (resolve_playpen_root(Path(__file__)) / "pyproject.toml").is_file()
        else "python -m alphalattice.control.product_host.composition."
        "evidence_authority_setup --help"
    )
    return {
        "status": "EVIDENCE_PREREQUISITES_MISSING",
        "failure_code": disposition,
        "explanation": detail,
        **why,
        "next_action": "SELECT_A_BOOK_AND_ADMIT_ITS_LOCAL_EVIDENCE_PACKAGE",
        "next_requests": {
            **(next_requests or {}),
            **offer,
            "workspace": {"operation": "WORKSPACE_SHOW"},
        },
        **(
            {"setup": evidence_setup(workspace, authored=_authored(book))}
            if missing
            else {"setup_help": help_line}
        ),
        **({} if listings is None else {"acquisition_scope": scope(listings)}),
        "claim": "No Task, acquisition or model work performed.",
    }


PACKAGE_RULE = (
    "A recorded package covers at most eight issuers, one unit, and an install replaces the "
    "package before it: a packet prepared under that one goes stale, and units prepared under "
    "different packages cannot be reviewed together, so a book's whole review takes one package "
    "covering every unit's issuers, or official acquisition."
)
"""What a recorded package can review: said wherever a package is offered or met."""


def source_ways(
    workspace: Path,
    *,
    entities: tuple[str, ...] | None,
    book: BookSelector | None,
    for_refusal: bool = False,
    refusal: str | None = None,
    delegation: str | None = None,
    now: datetime,
) -> dict[str, object]:
    """The lawful ways on when the recorded package holds too few sources.

    Official SEC acquisition first, the person's decision: with their consent and the network
    open, the running Host reads every holding's filing index at one cutoff, names those that
    filed nothing in the window and fetches the rest, so every unit stands at that cutoff and
    the book is reviewed whole. A recorded package is offered only where one covers every unit's
    issuers, a book of one unit (`PACKAGE_RULE`).

    Args:
        workspace: The workspace's folder.
        entities: Every issuer of the book, when one package covers them all; else None.
        book: The book, whose research input the package names when a study authored it.
        for_refusal: Whether official acquisition has actually been refused for network access.
        refusal: Why the official source is not admitted though consented, said first.
        delegation: Network authority validated by the first-use owner.
        now: The Host's clock, on which the network control is read.

    Returns:
        The ``official`` consent command with what it needs first, the package rule, and the
        ``package`` steps with the issuers filled where one package covers the book.
    """
    # After the person's own stop the way on is their yes, never a delegated set.
    delegation = None if persons_stop(workspace, now) else delegation
    # The person's own consent, in their shell: it names the workspace on either leg.
    ways: dict[str, object] = {
        "official": {
            "command": f'{" ".join(command_prefix())} --workspace "{workspace}" evidence-consent '
            f"set --per-issuer {DEFAULT_SOURCE_CONSENT.documents_per_issuer}",
            "before": str(
                network_access(workspace, now=now).body(
                    for_refusal=for_refusal, delegation=delegation
                )["detail"]
            )
            + " "
            + (
                # A first use's agent gives the default budget under its delegation (F11).
                "Under the first use's delegation the default budget is yours to give: set it "
                "and open the network, telling the person in one line; only a wider budget is "
                "the person's decision"
                if delegation
                else "The person's decision: their consent to acquire SEC filings from the "
                "official endpoints, within the budget it names, and the workspace's network open"
            )
            + "; the product names itself to the SEC by its own contact. Both take effect at "
            "once, with no restart; then preview and prepare again, which reads every unit at "
            "one cutoff.",
        },
        "package_rule": PACKAGE_RULE,
    }
    if delegation:
        # The default budget is the delegated agent's own request (F11).
        ways["next_requests"] = {
            "consent": {
                "operation": "EVIDENCE_CONSENT_SET",
                "evidence_documents_per_issuer": DEFAULT_SOURCE_CONSENT.documents_per_issuer,
            }
        }
    if refusal is not None:
        official = cast(dict[str, object], ways["official"])
        official["refusal_code"] = refusal
        official["before"] = f"{refusal_words(refusal).get('detail', refusal)} {official['before']}"
    if entities is None:
        return ways
    setup = evidence_setup(workspace, authored=_authored(book))
    authority = cast(dict[str, Any], setup["authority"])
    named = "--entities " + " ".join(entities)
    ways["package"] = {
        "entities": list(entities),
        "check": str(authority["check"]).replace("--entities <TICKER> ...", named),
        "install": str(authority["install"]).replace("--entities <TICKER> ...", named),
        "choose": [value for value in authority["choose"] if value["arg"] != "--entities"],
        "before": authority["before"],
        "covers": "Every issuer of this book, in place of the installed package; an issuer that "
        "filed nothing in the window holds no document in a package.",
    }
    return ways
