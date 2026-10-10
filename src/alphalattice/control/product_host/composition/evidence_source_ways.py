"""The lawful ways on when a book's sources fall short, and the short-source action.

Official SEC acquisition, the person's decision or, at the default budget, a first use's delegated
agent's own step; and a recorded package only where one covers every unit's issuers. The review
application answers them; the projection shows them beside a unit that failed for its sources.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from alphalattice.control.workspace_runtime.network_access import network_access
from alphalattice.evidence.alternative_evidence.sources.admission import DEFAULT_SOURCE_CONSENT
from alphalattice.interface.local_application.cli_contract import refusal_words
from alphalattice.kernel.knowledge import model_store
from alphalattice.kernel.knowledge.hybrid_contracts import RECIPE_MINILM_CPU
from alphalattice.kernel.shared_kernel.project_layout import command_prefix, resolve_playpen_root

if TYPE_CHECKING:
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import BookSelector

SHORT_SOURCE_ACTION: Final = {
    False: "ASK_FOR_OFFICIAL_ACQUISITION_OR_A_COVERING_PACKAGE",
    True: "ACQUIRE_UNDER_THE_FIRST_USE_DELEGATION",
}
"""A book short of sources: the person's decision, or the default budget a first use's agent
gives under its delegation (by whether one is active)."""


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
        from alphalattice.interface.local_application.cli_contract import join, shell

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
        from alphalattice.interface.local_application.retrieval_environment import fill_command

        pack["check"] = join(pack_command("--status"), shell())
        pack["install"] = join(pack_command("--install", RECIPE_MINILM_CPU, "--network"), shell())
        pack["environment"] = join(fill_command(), shell())
    elif not (resolve_playpen_root(Path(__file__)) / retrieval).is_file():
        pack["environment"] = f"{python} scripts/create_retrieval_environment.py --offline"
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

    Returns:
        The ``official`` consent command with what it needs first, the package rule, and the
        ``package`` steps with the issuers filled where one package covers the book.
    """
    # The person's own consent, in their shell: it names the workspace on either leg.
    ways: dict[str, object] = {
        "official": {
            "command": f'{" ".join(command_prefix())} --workspace "{workspace}" evidence-consent '
            f"set --per-issuer {DEFAULT_SOURCE_CONSENT.documents_per_issuer}",
            "before": str(
                network_access(workspace).body(for_refusal=for_refusal, delegation=delegation)[
                    "detail"
                ]
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
