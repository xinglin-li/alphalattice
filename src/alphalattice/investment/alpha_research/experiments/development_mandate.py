"""The capability mandate a development Alpha experiment searches within (O3).

Moved from the Host's research authoring, beside the installed capability mandate it
narrows and in a module of its own, so the closures that hold `mandate.py` do not move
for a function they never run (ID8).
"""

from __future__ import annotations

from alphalattice.capabilities.alpha_modeling.catalog import AlphaModelCatalog
from alphalattice.investment.alpha_research.experiments.mandate import (
    AlphaResearchModelMandate,
    build_installed_alpha_model_capability_mandate,
)


def build_development_alpha_model_mandate(
    *, catalog: AlphaModelCatalog | None = None
) -> AlphaResearchModelMandate:
    """The capability mandate a development Alpha experiment searches within.

    Separate from ``build_current_alpha_research_model_mandate`` because the
    budgets differ and only the budgets differ: the capability inventory comes
    from ``build_installed_alpha_model_capability_mandate`` -- the one installed
    definition production and the study verifier also derive from -- so a recipe
    admissible here is admissible there. A development run admits one recipe
    because a document names one.

    Args:
        catalog: The catalog it admits; the installed one when omitted (a workspace's
            activated models join it, EX).

    Returns:
        The mandate: one qualified candidate from one recipe in one batch.
    """
    capability = build_installed_alpha_model_capability_mandate(catalog=catalog)
    return AlphaResearchModelMandate.create(
        catalog_binding=capability.catalog_binding,
        ordered_search_domains=capability.ordered_search_domains,
        target_current_qualified_candidates=1,
        initial_batch_size=1,
        refinement_batch_max_size=1,
        max_batch_count=1,
        max_unique_new_recipes=1,
    )


__all__ = ["build_development_alpha_model_mandate"]
