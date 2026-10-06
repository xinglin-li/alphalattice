"""Alpha Research-owned model authority over Host-installed capabilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.alpha_modeling.catalog import (
    AlphaModelCatalog,
    AlphaModelCatalogBinding,
    build_installed_alpha_model_catalog,
    build_installed_alpha_model_search_domains,
)
from alphalattice.capabilities.alpha_modeling.contracts import (
    AlphaModelRecipeEnvelope,
    AlphaModelSearchDomainEnvelope,
)
from alphalattice.investment.alpha_research.inputs.training import AlphaTrainingInputAuthorityError
from alphalattice.investment.alpha_research.targets.execution_outcome import (
    AlphaTargetLane,
    AlphaTargetPolicy,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class AlphaModelRecipeProposal(_Contract):
    """Untrusted Agent proposal; the Host supplies every durable identity."""

    capability_handle: str = Field(pattern=r"^capability-[1-9][0-9]*$")
    parameters: dict[str, Any]
    target_lane: AlphaTargetLane | None = None


class AlphaResearchModelRecipe(_Contract):
    """One Host-admitted model recipe bound to its research target lane."""

    kind: Literal["AlphaResearchModelRecipe"] = "AlphaResearchModelRecipe"
    search_domain_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    recipe: AlphaModelRecipeEnvelope
    target_lane: AlphaTargetLane | None = None
    target_method_binding_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """Which target-bound method admitted this model, for a lane-free target.

    Present, it enters ``research_recipe_hash`` -- and therefore ``spec_hash``
    and ``candidate_id``, both of which derive from it -- so two arms running the
    same estimator with the same parameters against two different target
    compositions produce different candidates rather than one ambiguous id.

    Absent, it is excluded from the identity entirely rather than hashed as
    ``null``, so every frozen-lane recipe keeps the exact
    ``research_recipe_hash`` it already has. The same dual-identity idiom
    ``AlphaTargetPolicy`` uses for its historical hash, and for the same reason:
    a new field must not rotate identities that are already sealed.
    """

    research_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        search_domain_hash: str,
        recipe: AlphaModelRecipeEnvelope,
        target_lane: AlphaTargetLane | None,
        target_method_binding_hash: str | None = None,
    ) -> Self:
        values: dict[str, object] = {
            "kind": "AlphaResearchModelRecipe",
            "search_domain_hash": search_domain_hash,
            "recipe": recipe.model_dump(mode="json"),
            "target_lane": target_lane.value if target_lane is not None else None,
        }
        if target_method_binding_hash is not None:
            values["target_method_binding_hash"] = target_method_binding_hash
        return cls(**values, research_recipe_hash=canonical_hash(values))

    @property
    def spec_hash(self) -> str:
        """Compatibility name for consumers whose durable field remains `spec_hash`."""

        return self.research_recipe_hash

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        excluded = {"research_recipe_hash"}
        if self.target_method_binding_hash is None:
            # Excluded, not hashed as null, so a frozen-lane recipe sealed before
            # this field existed still validates against its own identity.
            excluded.add("target_method_binding_hash")
        if self.research_recipe_hash != canonical_hash(
            self.model_dump(mode="json", exclude=excluded)
        ):
            raise ValueError("ALPHA_RESEARCH_MODEL_RECIPE_IDENTITY_INVALID")
        return self


class AlphaDevelopmentModelMethodBinding(_Contract):
    """One admitted model, bound to the exact target and outcome it was fitted to.

    ``AlphaResearchModelRecipe`` identifies a model by its search domain, its
    parameters and -- for the frozen lanes -- its target lane. That was complete
    while every target *was* a lane. It is not complete now: two arms of a paired
    study run the same estimator with the same parameters against two different
    target compositions, and neither composition has a lane, so both would seal
    the same ``research_recipe_hash`` and produce candidates that are
    indistinguishable at every level a consumer can inspect.

    Passing ``None`` for the lane does not repair that. ``None`` means "not one of
    the frozen lanes", which is a statement about what the model is *not* bound
    to; used as the binding it would become a channel through which any target
    reaches any model with no admission recorded at all.

    This binding is the missing statement. It names the model, the domain that
    admitted it, the target method, the target-to-outcome binding, and the
    outcome method seal -- so the identity of a fitted candidate answers which
    science produced it rather than which estimator ran.
    """

    kind: Literal["AlphaDevelopmentModelMethodBinding"] = "AlphaDevelopmentModelMethodBinding"
    model_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_domain_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_recipe_id: str = Field(min_length=1, max_length=96)
    target_method_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_recipe_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome_method_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        model_recipe_hash: str,
        search_domain_hash: str,
        target_recipe_id: str,
        target_method_hash: str,
        target_recipe_binding_hash: str,
        outcome_method_binding_hash: str,
    ) -> Self:
        """Seal from the *adapter* recipe rather than the research recipe.

        Deliberately, and it is not a detail. The research recipe carries this
        binding's hash, so deriving this binding from the research recipe would
        be circular -- neither could be sealed without the other. The adapter
        envelope is the part that is knowable first: it is the model's
        parameters and route, fixed by admission before anything is bound to a
        target.
        """

        values: dict[str, object] = {
            "kind": "AlphaDevelopmentModelMethodBinding",
            "model_recipe_hash": model_recipe_hash,
            "search_domain_hash": search_domain_hash,
            "target_recipe_id": target_recipe_id,
            "target_method_hash": target_method_hash,
            "target_recipe_binding_hash": target_recipe_binding_hash,
            "outcome_method_binding_hash": outcome_method_binding_hash,
        }
        return cls(**values, binding_hash=str(canonical_hash(values)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if self.binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"binding_hash"})
        ):
            raise ValueError("ALPHA_DEVELOPMENT_MODEL_METHOD_BINDING_IDENTITY_INVALID")
        return self


def bind_admitted_model_to_target(
    *,
    admitted: AlphaResearchModelRecipe,
    target_recipe_id: str,
    target_method_hash: str,
    target_recipe_binding_hash: str,
    outcome_method_binding_hash: str,
) -> tuple[AlphaResearchModelRecipe, AlphaDevelopmentModelMethodBinding]:
    """Re-seal one admitted model with the target it will actually be fitted to.

    Admission stays the mandate's decision -- this runs after it and changes
    nothing about what was admitted, only what the admitted thing is bound to.
    Shared by the compiler and the executor so the two cannot drift: they seal
    the same identity from the same inputs or the run fails its own replay check.
    """

    binding = AlphaDevelopmentModelMethodBinding.create(
        model_recipe_hash=admitted.recipe.recipe_hash,
        search_domain_hash=admitted.search_domain_hash,
        target_recipe_id=target_recipe_id,
        target_method_hash=target_method_hash,
        target_recipe_binding_hash=target_recipe_binding_hash,
        outcome_method_binding_hash=outcome_method_binding_hash,
    )
    bound = AlphaResearchModelRecipe.create(
        search_domain_hash=admitted.search_domain_hash,
        recipe=admitted.recipe,
        target_lane=admitted.target_lane,
        target_method_binding_hash=binding.binding_hash,
    )
    return bound, binding


class AlphaModelCapabilityAuthority(Protocol):
    """Capability/search-domain authority required by development execution."""

    @property
    def catalog_binding(self) -> AlphaModelCatalogBinding: ...

    @property
    def ordered_search_domains(self) -> tuple[AlphaModelSearchDomainEnvelope, ...]: ...

    @property
    def mandate_hash(self) -> str: ...

    def capability_handle(self, index: int) -> str: ...

    def resolve_capability_handle(self, handle: str) -> AlphaModelSearchDomainEnvelope: ...

    def admitting(self, search_domain_hash: str) -> AlphaModelCapabilityAuthority: ...

    def admit_proposal(
        self,
        *,
        proposal: AlphaModelRecipeProposal,
        catalog: AlphaModelCatalog,
        admitted_target_lanes: tuple[AlphaTargetLane, ...] | None,
    ) -> AlphaResearchModelRecipe: ...


def admitted_capabilities_installed(
    authority: AlphaModelCapabilityAuthority, catalog: AlphaModelCatalog
) -> bool:
    """Whether every model the authority admits is installed as the authority recorded it.

    Only the admitted capabilities are compared, never the whole catalog: a model
    installed beside them moves no mandate and refuses nothing (V118).
    """
    admitted = {value.adapter_id for value in authority.ordered_search_domains}
    try:
        installed = catalog.binding.restricted_to(admitted)
    except ValueError:
        return False
    return installed == authority.catalog_binding.restricted_to(admitted)


def _admitted_domain(
    ordered_search_domains: tuple[AlphaModelSearchDomainEnvelope, ...], search_domain_hash: str
) -> AlphaModelSearchDomainEnvelope:
    domain = next(
        (
            value
            for value in ordered_search_domains
            if value.search_domain_hash == search_domain_hash
        ),
        None,
    )
    if domain is None:
        raise ValueError("ALPHA_MODEL_SEARCH_DOMAIN_NOT_MANDATED")
    return domain


def _resolve_capability_handle(
    ordered_search_domains: tuple[AlphaModelSearchDomainEnvelope, ...], handle: str
) -> AlphaModelSearchDomainEnvelope:
    try:
        index = int(handle.removeprefix("capability-")) - 1
    except ValueError as error:
        raise ValueError("ALPHA_MODEL_CAPABILITY_HANDLE_INVALID") from error
    if index < 0 or handle != f"capability-{index + 1}":
        raise ValueError("ALPHA_MODEL_CAPABILITY_HANDLE_INVALID")
    try:
        return ordered_search_domains[index]
    except IndexError as error:
        raise ValueError("ALPHA_MODEL_CAPABILITY_HANDLE_INVALID") from error


def _admit_proposal(
    *,
    authority: AlphaModelCapabilityAuthority,
    proposal: AlphaModelRecipeProposal,
    catalog: AlphaModelCatalog,
    admitted_target_lanes: tuple[AlphaTargetLane, ...] | None,
) -> AlphaResearchModelRecipe:
    if not admitted_capabilities_installed(authority, catalog):
        raise ValueError("ALPHA_MODEL_CATALOG_MANDATE_MISMATCH")
    if admitted_target_lanes is None:
        if proposal.target_lane is not None:
            raise ValueError("ALPHA_MODEL_TARGET_LANE_NOT_ADMITTED")
    elif proposal.target_lane not in admitted_target_lanes:
        raise ValueError("ALPHA_MODEL_TARGET_LANE_NOT_ADMITTED")
    domain = authority.resolve_capability_handle(proposal.capability_handle)
    recipe = AlphaModelRecipeEnvelope.create(
        adapter_id=domain.adapter_id,
        recipe_schema_id=domain.recipe_schema_id,
        parameters=proposal.parameters,
    )
    catalog.admit_recipe(recipe=recipe, domain=domain)
    return AlphaResearchModelRecipe.create(
        search_domain_hash=domain.search_domain_hash,
        recipe=recipe,
        target_lane=proposal.target_lane,
    )


def _assert_capability_inventory(
    *,
    catalog_binding: AlphaModelCatalogBinding,
    ordered_search_domains: tuple[AlphaModelSearchDomainEnvelope, ...],
    failure_code: str,
) -> None:
    domain_hashes = tuple(value.search_domain_hash for value in ordered_search_domains)
    domain_routes = tuple(
        (value.adapter_id, value.recipe_schema_id, value.search_domain_schema_id)
        for value in ordered_search_domains
    )
    installed = {
        (value.adapter_id, value.recipe_schema_id, value.search_domain_schema_id)
        for value in catalog_binding.ordered_capabilities
    }
    if (
        len(set(domain_hashes)) != len(domain_hashes)
        or len(set(domain_routes)) != len(domain_routes)
        or any(value not in installed for value in domain_routes)
    ):
        raise ValueError(failure_code)


class AlphaModelCapabilityMandate(_Contract):
    """Standalone permission over installed adapters and their search domains."""

    kind: Literal["AlphaModelCapabilityMandate"] = "AlphaModelCapabilityMandate"
    catalog_binding: AlphaModelCatalogBinding
    ordered_search_domains: tuple[AlphaModelSearchDomainEnvelope, ...] = Field(min_length=1)
    mandate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        catalog_binding: AlphaModelCatalogBinding,
        ordered_search_domains: tuple[AlphaModelSearchDomainEnvelope, ...],
    ) -> Self:
        values = {
            "kind": "AlphaModelCapabilityMandate",
            "catalog_binding": catalog_binding.model_dump(mode="json"),
            "ordered_search_domains": [
                value.model_dump(mode="json") for value in ordered_search_domains
            ],
        }
        return cls(**values, mandate_hash=canonical_hash(values))

    def capability_handle(self, index: int) -> str:
        if index < 0 or index >= len(self.ordered_search_domains):
            raise ValueError("ALPHA_MODEL_CAPABILITY_HANDLE_INVALID")
        return f"capability-{index + 1}"

    def resolve_capability_handle(self, handle: str) -> AlphaModelSearchDomainEnvelope:
        return _resolve_capability_handle(self.ordered_search_domains, handle)

    def admitting(self, search_domain_hash: str) -> AlphaModelCapabilityMandate:
        """This mandate narrowed to the one domain a study chose and the model it routes to.

        What a development Program binds (V118): the model it runs, never the other
        models the installed catalog or this mandate hold.
        """
        domain = _admitted_domain(self.ordered_search_domains, search_domain_hash)
        return AlphaModelCapabilityMandate.create(
            catalog_binding=self.catalog_binding.restricted_to((domain.adapter_id,)),
            ordered_search_domains=(domain,),
        )

    def admit_proposal(
        self,
        *,
        proposal: AlphaModelRecipeProposal,
        catalog: AlphaModelCatalog,
        admitted_target_lanes: tuple[AlphaTargetLane, ...] | None,
    ) -> AlphaResearchModelRecipe:
        return _admit_proposal(
            authority=self,
            proposal=proposal,
            catalog=catalog,
            admitted_target_lanes=admitted_target_lanes,
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        _assert_capability_inventory(
            catalog_binding=self.catalog_binding,
            ordered_search_domains=self.ordered_search_domains,
            failure_code="ALPHA_MODEL_CAPABILITY_MANDATE_INVALID",
        )
        if self.mandate_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"mandate_hash"})
        ):
            raise ValueError("ALPHA_MODEL_CAPABILITY_MANDATE_INVALID")
        return self


class AlphaResearchModelMandate(_Contract):
    """Frozen Alpha Research permission and budget over installed model capabilities."""

    kind: Literal["AlphaResearchModelMandate"] = "AlphaResearchModelMandate"
    catalog_binding: AlphaModelCatalogBinding
    ordered_search_domains: tuple[AlphaModelSearchDomainEnvelope, ...] = Field(min_length=1)
    target_current_qualified_candidates: int = Field(ge=1)
    initial_batch_size: int = Field(ge=1)
    refinement_batch_max_size: int = Field(ge=1)
    max_batch_count: int = Field(ge=1)
    max_unique_new_recipes: int = Field(ge=1)
    mandate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        catalog_binding: AlphaModelCatalogBinding,
        ordered_search_domains: tuple[AlphaModelSearchDomainEnvelope, ...],
        target_current_qualified_candidates: int,
        initial_batch_size: int,
        refinement_batch_max_size: int,
        max_batch_count: int,
        max_unique_new_recipes: int,
    ) -> Self:
        values = {
            "kind": "AlphaResearchModelMandate",
            "catalog_binding": catalog_binding.model_dump(mode="json"),
            "ordered_search_domains": [
                value.model_dump(mode="json") for value in ordered_search_domains
            ],
            "target_current_qualified_candidates": target_current_qualified_candidates,
            "initial_batch_size": initial_batch_size,
            "refinement_batch_max_size": refinement_batch_max_size,
            "max_batch_count": max_batch_count,
            "max_unique_new_recipes": max_unique_new_recipes,
        }
        return cls(**values, mandate_hash=canonical_hash(values))

    def capability_handle(self, index: int) -> str:
        if index < 0 or index >= len(self.ordered_search_domains):
            raise ValueError("ALPHA_MODEL_CAPABILITY_HANDLE_INVALID")
        return f"capability-{index + 1}"

    def resolve_capability_handle(self, handle: str) -> AlphaModelSearchDomainEnvelope:
        return _resolve_capability_handle(self.ordered_search_domains, handle)

    def admitting(self, search_domain_hash: str) -> AlphaResearchModelMandate:
        """This mandate narrowed to the one domain a study chose and the model it routes to,
        its budgets kept: what a development Program binds (V118)."""
        return self.admitting_domains((search_domain_hash,))

    def admitting_domains(self, search_domain_hashes: Iterable[str]) -> AlphaResearchModelMandate:
        """This mandate narrowed to the domains a question's studies chose and their models.

        What a qualification binds (V118, V299): the models its family ran, in this mandate's
        order, its budgets kept; a model installed or retired beside them moves nothing.

        Args:
            search_domain_hashes: The chosen domains; each must be this mandate's.

        Returns:
            The narrowed mandate.
        """
        chosen = {
            _admitted_domain(self.ordered_search_domains, value).search_domain_hash
            for value in search_domain_hashes
        }
        domains = tuple(
            value for value in self.ordered_search_domains if value.search_domain_hash in chosen
        )
        return AlphaResearchModelMandate.create(
            catalog_binding=self.catalog_binding.restricted_to(
                value.adapter_id for value in domains
            ),
            ordered_search_domains=domains,
            target_current_qualified_candidates=self.target_current_qualified_candidates,
            initial_batch_size=self.initial_batch_size,
            refinement_batch_max_size=self.refinement_batch_max_size,
            max_batch_count=self.max_batch_count,
            max_unique_new_recipes=self.max_unique_new_recipes,
        )

    def admit_proposal(
        self,
        *,
        proposal: AlphaModelRecipeProposal,
        catalog: AlphaModelCatalog,
        admitted_target_lanes: tuple[AlphaTargetLane, ...] | None,
    ) -> AlphaResearchModelRecipe:
        return _admit_proposal(
            authority=self,
            proposal=proposal,
            catalog=catalog,
            admitted_target_lanes=admitted_target_lanes,
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        _assert_capability_inventory(
            catalog_binding=self.catalog_binding,
            ordered_search_domains=self.ordered_search_domains,
            failure_code="ALPHA_RESEARCH_MODEL_MANDATE_INVALID",
        )
        identity = self.model_dump(mode="json", exclude={"mandate_hash"})
        accepted_hashes = {canonical_hash(identity)}
        if all(
            value.numerical_binding_hash is None
            for value in self.catalog_binding.ordered_capabilities
        ):
            legacy_identity = dict(identity)
            legacy_catalog_binding = dict(legacy_identity["catalog_binding"])
            legacy_catalog_binding["ordered_capabilities"] = [
                {key: value for key, value in capability.items() if key != "numerical_binding_hash"}
                for capability in legacy_catalog_binding["ordered_capabilities"]
            ]
            legacy_identity["catalog_binding"] = legacy_catalog_binding
            accepted_hashes.add(canonical_hash(legacy_identity))
        if (
            self.initial_batch_size > self.max_unique_new_recipes
            or self.refinement_batch_max_size > self.max_unique_new_recipes
            or self.target_current_qualified_candidates > self.max_unique_new_recipes
            or self.mandate_hash not in accepted_hashes
        ):
            raise ValueError("ALPHA_RESEARCH_MODEL_MANDATE_INVALID")
        return self


def build_installed_alpha_model_capability_mandate(
    *, catalog: AlphaModelCatalog | None = None
) -> AlphaModelCapabilityMandate:
    """The installed capability inventory, defined exactly once and budget-free.

    Every mandate -- production, development, and the verifier's independent
    side -- derives its capability inventory from this factory, so "installed"
    has one definition. A second capability arrives by adding its domain here,
    never by loosening a handle check somewhere downstream.
    """

    installed = catalog or build_installed_alpha_model_catalog()
    domains = build_installed_alpha_model_search_domains(installed)
    for domain in domains:
        installed.resolve_search_domain(domain)
    return AlphaModelCapabilityMandate.create(
        catalog_binding=installed.binding,
        ordered_search_domains=domains,
    )


def build_current_alpha_research_model_mandate(
    *, catalog: AlphaModelCatalog | None = None
) -> AlphaResearchModelMandate:
    capability = build_installed_alpha_model_capability_mandate(catalog=catalog)
    return AlphaResearchModelMandate.create(
        catalog_binding=capability.catalog_binding,
        ordered_search_domains=capability.ordered_search_domains,
        target_current_qualified_candidates=3,
        initial_batch_size=6,
        refinement_batch_max_size=3,
        max_batch_count=2,
        max_unique_new_recipes=9,
    )


def assert_research_recipe_target_authority(
    *,
    program_target_policy_hashes: tuple[str, ...] | None,
    recipe_target_lanes: tuple[AlphaTargetLane | None, ...],
    default_target_policy: AlphaTargetPolicy | None,
    lane_target_policies: Mapping[AlphaTargetLane, AlphaTargetPolicy | None],
) -> None:
    """Bind every active recipe lane to a Program-authorized target policy."""

    if program_target_policy_hashes is None:
        if (
            any(value is not None for value in recipe_target_lanes)
            or default_target_policy is not None
            or lane_target_policies
        ):
            raise AlphaTrainingInputAuthorityError("TARGET_AUTHORITY_MISMATCH")
        return

    if (
        not recipe_target_lanes
        or any(value is None for value in recipe_target_lanes)
        or default_target_policy is not None
    ):
        raise AlphaTrainingInputAuthorityError("TARGET_AUTHORITY_MISMATCH")
    used_lanes = {value for value in recipe_target_lanes if value is not None}
    if set(lane_target_policies) != used_lanes:
        raise AlphaTrainingInputAuthorityError("TARGET_AUTHORITY_MISMATCH")
    authorized_hashes = set(program_target_policy_hashes)
    for lane, policy in lane_target_policies.items():
        if policy is None or policy.lane is not lane or policy.policy_hash not in authorized_hashes:
            raise AlphaTrainingInputAuthorityError("TARGET_AUTHORITY_MISMATCH")


__all__ = [
    "AlphaDevelopmentModelMethodBinding",
    "AlphaModelCapabilityAuthority",
    "AlphaModelCapabilityMandate",
    "AlphaModelRecipeProposal",
    "AlphaResearchModelMandate",
    "AlphaResearchModelRecipe",
    "admitted_capabilities_installed",
    "assert_research_recipe_target_authority",
    "bind_admitted_model_to_target",
    "build_current_alpha_research_model_mandate",
    "build_installed_alpha_model_capability_mandate",
]
