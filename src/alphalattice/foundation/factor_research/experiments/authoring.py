"""Compile one authored Factor experiment into a typed development Program.

The Factor ID inventory is a dynamic ordered axis supplied by the Host, not a
literal in this module: adding or removing an ordinary Factor changes the
inventory, never this compiler and never Alpha, Risk, or Portfolio source.

It was dynamic in shape and static in *breadth*, which is a distinction that
only shows up when the axis finally grows. The compiler refused any inventory
longer than the shipped catalog's historical 55, so the first method-family batch
large enough to matter would have been refused by a development compiler quoting
a number from a frozen artifact. The budget here is now a development budget with
a reason of its own, and the frozen contracts keep their literals untouched.

The identity formulas this compiler seals -- the ordered catalog, the parameter
domain, the method binding, the Desk Program -- are module-level functions rather
than inline expressions, because the evidence verifier has to recompute them. A
verifier holding its own copy of an identity formula eventually answers a
different question than the compiler asked.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    DeskProgramCompilation,
    DeskSection,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
    refuse_unknown_section_keys,
)

FACTOR_EXPERIMENT_KIND = "factor.screening-development"
INSTALLED_SCREENING_POLICIES = ("BENJAMINI_YEKUTIELI_FDR",)
INSTALLED_REDUNDANCY_POLICIES = ("ABSOLUTE_CORRELATION_CLUSTER",)

DEVELOPMENT_CONTEXT_AXIS_CEILING = 256
"""How many Factors one development context axis may carry.

Bounded independently of the shipped catalog's historical breadth, so a frozen
screening report's factor count never stands in for the development budget.

The bound that belongs here is statistical rather than historical. The context
axis is the multiple-testing denominator: every Factor on it is a simultaneous
hypothesis the installed Benjamini-Yekutieli correction has to resolve on one
cross-section, and past a few hundred the correction has no power left to give.
So the ceiling is stated once, by the Desk that owns the correction, and it is a
ceiling rather than an expectation -- adding a Factor never edits this module.

Frozen contracts keep their own literal breadth and stay readable. This does not
widen them; it stops the development path from borrowing them.
"""


class FactorSection(DeskSection):
    """A Factor screening study's section: everything an authored Factor document may say.

    Unknown keys were previously ignored. Ignoring is the wrong answer twice over: a
    misspelled policy key silently selected the default, and a document carrying a
    Panel hash, an evidence handle or an implementation identity read as though the
    Host had honoured it. The request layer names *what* to compute over; every
    identity is resolved by the Host and none of it may arrive as document text.
    """

    factor_ids: list[str]
    """The Factors screened, each in the installed inventory, in the order written."""
    screening_policy: str
    """The installed screening policy that judges each Factor."""
    redundancy_policy: str
    """The installed redundancy policy that sets near duplicates aside."""


_IDENTITY_BEARING_TOKENS = ("hash", "uri", "path", "root", "snapshot", "handle", "identity")
"""Substrings that make an unknown key a refused *authority* claim, not a typo.

Reported separately because the two failures need different answers from a
reader: one is a spelling mistake in their document, the other is an attempt --
however well meant -- to supply an identity that is not theirs to supply.
"""


def factor_execution_input_hash(
    panel_snapshot_hash: str, outcome_snapshot_hash: str, policy_hashes: tuple[str, ...]
) -> str:
    """The same pre-execution input commitment at admission and readback."""
    return str(
        canonical_hash(
            {
                "panel_snapshot_hash": panel_snapshot_hash,
                "outcome_snapshot_hash": outcome_snapshot_hash,
                "policy_hashes": policy_hashes,
            }
        )
    )


def factor_desk_program_hash(
    *,
    envelope_hash: str,
    selected_factor_ids: tuple[str, ...],
    method_binding_hash: str,
    authority_hash: str,
) -> str:
    """The admitted Desk Program identity, computed in exactly one place.

    The compiler seals this at freeze time, before anything executes, and the
    receipt verifier recomputes it afterwards from the selected axis it was
    handed. That makes the admitted Program an independent side: a receipt that
    rewrote its selected axis and resealed every hash it owns still contradicts
    an artifact sealed before the run.

    That only holds while the two agree on the formula, so there is one function
    rather than two copies. A verifier with its own copy of an identity formula
    is a verifier that eventually checks a different question than the one the
    compiler answered.

    The axis is used exactly as ordered. Sorting it here would make two documents
    that selected the same factors in different orders one Program, and order is
    what the researcher authored.
    """

    return str(
        canonical_hash(
            {
                "kind": FACTOR_EXPERIMENT_KIND,
                "envelope_hash": envelope_hash,
                "factor_ids": list(selected_factor_ids),
                "method_binding_hash": method_binding_hash,
                "authority_hash": authority_hash,
            }
        )
    )


@dataclass(frozen=True, slots=True)
class FactorInventoryEntry:
    """One installed Factor and the identity of the code that computes it.

    The inventory used to be bare Factor IDs, and an ID is not an identity: the
    same string denotes whichever recipe the Feature catalog currently binds to
    it. So a Factor could be rewritten -- different window, different kernel,
    different numbers -- while the Factor Desk's ``catalog_hash`` stood still and
    evidence produced by the old implementation still looked reusable. That is
    the same "same identifier, changed code" defect the Risk catalog closed, and
    it was recorded as a known gap in this module rather than fixed.

    Both hashes are values the Feature panel manifest publishes per factor, so
    this Desk consumes identities the Feature capability already computed rather
    than inventing a second opinion about what a Factor is.

    ``methodology_hash`` is the one that answers the question this Desk is asking.
    ``implementation_hash`` says only *which code*; two factors can share a kernel
    and differ in window, lag or return convention, and those are different
    methods. Carried alongside rather than instead, because the two answer
    genuinely different questions and a reader that wants the kernel identity
    should not have to unpack a method identity to get it.
    """

    factor_id: str
    implementation_hash: str
    methodology_hash: str

    def __post_init__(self) -> None:
        if not self.factor_id or not self.implementation_hash or not self.methodology_hash:
            raise AuthoringError("factor_research.authoring_inventory_entry_invalid")


def factor_catalog_hash(entries: tuple[FactorInventoryEntry, ...]) -> str:
    """The identity of the ordered installed axis, computed in exactly one place.

    The compiler seals this into the Program before anything runs, and the
    evidence verifier recomputes it afterwards from the axis the receipt reports.
    That is what makes a resealed receipt refusable: a graph that rewrote its
    context axis and consistently resealed everything it owns still contradicts
    an artifact sealed before the run.

    It binds each Factor's *methodology* identity, not its ID. An ID is not an
    identity: the same string denotes whichever recipe the Feature catalog
    currently binds to it, so a Factor rewritten under a stable ID used to leave
    this hash standing still and its old evidence looking reusable. The
    implementation hash alone was not enough either -- it moves when the kernel
    changes and stands still when only the window, lag or return convention does,
    which is a rewrite this Desk would still not have seen.

    One function rather than two copies, for the same reason
    ``factor_desk_program_hash`` is one: a verifier with its own copy of an
    identity formula eventually checks a different question than the one the
    compiler answered.
    """

    return str(
        canonical_hash(
            {
                "ordered_factors": [
                    {
                        "factor_id": entry.factor_id,
                        "implementation_hash": entry.implementation_hash,
                        "methodology_hash": entry.methodology_hash,
                    }
                    for entry in entries
                ]
            }
        )
    )


def factor_parameter_domain_hash(*, screening_policy: str, redundancy_policy: str) -> str:
    """The admissible space of the chosen method: the one screening and one redundancy
    policy the Program names, each a closed method with no free parameter.

    The installed menu is not the domain: a policy no Program chose decides none of its
    numbers, so installing one moves no Program (LAWS.md ID3). The shape is the menu's
    shape, so a Program sealed while each menu held one policy keeps its value.
    """

    return str(
        canonical_hash(
            {
                "screening_policies": [screening_policy],
                "redundancy_policies": [redundancy_policy],
            }
        )
    )


def factor_method_binding_hash(
    *,
    catalog_hash: str,
    screening_policy: str,
    redundancy_policy: str,
) -> str:
    """What method this Program runs: the installed axis and the chosen policies."""

    return str(
        canonical_hash(
            {
                "catalog_hash": catalog_hash,
                "screening_policy": screening_policy,
                "redundancy_policy": redundancy_policy,
                "parameter_domain_hash": factor_parameter_domain_hash(
                    screening_policy=screening_policy, redundancy_policy=redundancy_policy
                ),
            }
        )
    )


def factor_inventory_from_panel_manifest(
    panel_manifest: Mapping[str, Any],
) -> tuple[FactorInventoryEntry, ...]:
    """Read the installed Factor axis and its identities off a published panel.

    The panel is the authority on which Factors exist and which code produced
    them, so the Desk reads them together from one artifact. Taking the axis from
    the panel and the identities from anywhere else would allow the two to
    disagree about the same run.

    Sorted, because the axis is compared for equality against the program's own
    ordered ``factor_ids`` downstream.
    """

    summary = panel_manifest.get("safe_summary")
    factors = summary.get("factor_catalog_summary") if isinstance(summary, dict) else None
    if not isinstance(factors, dict) or not factors:
        raise AuthoringError("factor_research.authoring_panel_factor_catalog_missing")
    entries = []
    for factor_id in sorted(str(value) for value in factors):
        entry = factors[factor_id]
        if not isinstance(entry, dict):
            # A summary whose entries are not mappings is a malformed panel, and
            # it must fail as this Desk's own refusal. Reaching ``.get`` on a
            # string or a list would surface an ``AttributeError`` from inside a
            # reader, which tells a caller nothing about which artifact was bad.
            raise AuthoringError("factor_research.authoring_panel_factor_entry_invalid")
        implementation = entry.get("implementation_hash")
        if not isinstance(implementation, str) or not implementation:
            # Panels published before the Feature catalog summary carried
            # implementation identity read back fine everywhere else, but they
            # cannot answer this question -- and silently substituting the factor
            # id would restore exactly the defect this closes.
            raise AuthoringError("factor_research.authoring_panel_implementation_identity_missing")
        methodology = entry.get("methodology_hash")
        if not isinstance(methodology, str) or not methodology:
            # A panel published before methodology identity existed is
            # **readback-only**: every other reader still resolves it under its
            # own stored schema, but it cannot enter this writer, because the one
            # question this Desk needs answered is the one it cannot answer.
            #
            # Refused rather than defaulted. Substituting the implementation hash
            # would silently reinstate the gap -- a factor rewritten to a new
            # window under an unchanged kernel would look unchanged -- and
            # backfilling the field here would mean this reader inventing
            # identity for an artifact somebody else sealed.
            raise AuthoringError("factor_research.authoring_panel_methodology_identity_missing")
        entries.append(
            FactorInventoryEntry(
                factor_id=factor_id,
                implementation_hash=implementation,
                methodology_hash=methodology,
            )
        )
    return tuple(entries)


class FactorExperimentCompiler:
    """Desk-owned compiler; produces identity, never screening results."""

    kind = FACTOR_EXPERIMENT_KIND

    def __init__(self, factor_inventory: tuple[FactorInventoryEntry, ...]) -> None:
        ordered = tuple(dict.fromkeys(entry.factor_id for entry in factor_inventory))
        if not ordered or len(ordered) != len(factor_inventory):
            raise AuthoringError("factor_research.authoring_inventory_invalid")
        if len(ordered) > DEVELOPMENT_CONTEXT_AXIS_CEILING:
            raise AuthoringError("factor_research.authoring_inventory_exceeds_development_ceiling")
        self._entries = tuple(factor_inventory)
        self._inventory = ordered

    @property
    def factor_inventory(self) -> tuple[str, ...]:
        """The ordered Factor axis an author may select from."""

        return self._inventory

    @property
    def inventory_entries(self) -> tuple[FactorInventoryEntry, ...]:
        return self._entries

    def compile_desk_program(
        self,
        *,
        envelope: ResearchExperimentEnvelope,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
    ) -> DeskProgramCompilation:
        section = document.get("factor")
        if not isinstance(section, dict):
            raise AuthoringError("factor_research.authoring_section_missing")
        _refuse_unadmitted_keys(section)

        requested = section.get("factor_ids")
        if not isinstance(requested, list) or not requested:
            raise AuthoringError("factor_research.authoring_factor_ids_invalid")
        factor_ids = tuple(str(value) for value in requested)
        if factor_ids != tuple(dict.fromkeys(factor_ids)):
            raise AuthoringError("factor_research.authoring_factor_ids_duplicated")
        unknown = tuple(value for value in factor_ids if value not in self._inventory)
        if unknown:
            raise AuthoringError("factor_research.authoring_factor_id_not_in_inventory")

        screening = section.get("screening_policy")
        if screening not in INSTALLED_SCREENING_POLICIES:
            raise AuthoringError("factor_research.authoring_screening_policy_not_installed")
        redundancy = section.get("redundancy_policy")
        if redundancy not in INSTALLED_REDUNDANCY_POLICIES:
            raise AuthoringError("factor_research.authoring_redundancy_policy_not_installed")

        if len(factor_ids) > envelope.budget.maximum_candidates:
            raise AuthoringError("factor_research.authoring_candidate_budget_exceeded")

        # The inventory axis is ordered and content-addressed, so adding an
        # ordinary Factor moves the catalog identity without touching source.
        catalog_hash = factor_catalog_hash(self._entries)
        parameter_domain_hash = factor_parameter_domain_hash(
            screening_policy=str(screening), redundancy_policy=str(redundancy)
        )
        method_binding_hash = factor_method_binding_hash(
            catalog_hash=catalog_hash,
            screening_policy=str(screening),
            redundancy_policy=str(redundancy),
        )
        desk_program_hash = factor_desk_program_hash(
            envelope_hash=envelope.envelope_hash,
            selected_factor_ids=factor_ids,
            method_binding_hash=method_binding_hash,
            authority_hash=authority.authority_hash,
        )
        return DeskProgramCompilation(
            desk_program_hash=desk_program_hash,
            catalog_hash=catalog_hash,
            method_binding_hash=method_binding_hash,
            parameter_domain_hash=parameter_domain_hash,
        )


def _refuse_unadmitted_keys(section: Mapping[str, Any]) -> None:
    """Refuse anything the request layer is not allowed to say.

    Split into two refusals on purpose. An unrecognised ordinary key is a
    document mistake and reads as one; a key that looks like it carries an
    identity -- a hash, a URI, a root, a snapshot, a handle -- is a request to
    supply authority the Host resolves, and a reader deserves to be told that is
    what was refused rather than that they misspelled something.
    """

    unknown = tuple(
        sorted(str(key) for key in section if str(key) not in FactorSection.model_fields)
    )
    claimed = tuple(
        key for key in unknown if any(token in key.lower() for token in _IDENTITY_BEARING_TOKENS)
    )
    if claimed:
        raise AuthoringError(f"factor_research.authoring_request_claims_authority:{claimed[0]}")
    refuse_unknown_section_keys(section, FactorSection, place="factor")


__all__ = [
    "DEVELOPMENT_CONTEXT_AXIS_CEILING",
    "FACTOR_EXPERIMENT_KIND",
    "INSTALLED_REDUNDANCY_POLICIES",
    "INSTALLED_SCREENING_POLICIES",
    "FactorExperimentCompiler",
    "FactorInventoryEntry",
    "FactorSection",
    "factor_catalog_hash",
    "factor_desk_program_hash",
    "factor_inventory_from_panel_manifest",
    "factor_method_binding_hash",
    "factor_parameter_domain_hash",
]
