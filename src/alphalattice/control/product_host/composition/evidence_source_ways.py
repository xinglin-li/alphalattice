"""The lawful ways on when a book's sources fall short, and the short-source action.

Official SEC acquisition, the person's decision or, at the default budget, a first use's delegated
agent's own step; and a recorded package only where one covers every unit's issuers. The review
application answers them; the projection shows them beside a unit that failed for its sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from alphalattice.control.workspace_runtime.network_access import network_access
from alphalattice.evidence.alternative_evidence.sources.admission import DEFAULT_SOURCE_CONSENT
from alphalattice.interface.local_application.cli_contract import refusal_words
from alphalattice.kernel.shared_kernel.project_layout import command_prefix

if TYPE_CHECKING:
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import BookSelector

SHORT_SOURCE_ACTION: Final = {
    False: "ASK_FOR_OFFICIAL_ACQUISITION_OR_A_COVERING_PACKAGE",
    True: "ACQUIRE_UNDER_THE_FIRST_USE_DELEGATION",
}
"""A book short of sources: the person's decision, or the default budget a first use's agent
gives under its delegation (by whether one is active)."""


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
    from alphalattice.control.product_host.composition.evidence_review_application import (
        PACKAGE_RULE,
        _authored,
        evidence_setup,
    )

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
