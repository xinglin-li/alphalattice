"""Adapter and capability identity of the Risk development program.

A changed implementation under an unchanged id changes the program, an adapter
that is not the installed one is refused at admission and at execution, a
separate instance stays installable, executable content moves the selected
numerical binding, the declared implementation hash is the real module bytes,
every bound identity is consumed by the program hash, and tampered bindings and
governance fields fail their own identity checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alphalattice.investment.risk_research.estimators.catalog import (
    RiskEstimatorCatalog,
    build_installed_risk_estimator_catalog,
)
from alphalattice.investment.risk_research.estimators.contracts import (
    RiskEstimatorAdapter,
    RiskEstimatorNumericalBinding,
    RiskEstimatorRecipeEnvelope,
)
from alphalattice.investment.risk_research.estimators.covariance import (
    COVARIANCE_RECIPE_SCHEMA_ID,
    CovarianceCapability,
    CovarianceEstimatorAdapter,
)
from alphalattice.investment.risk_research.estimators.domains import (
    COVARIANCE_PARAMETER_DOMAIN,
    RiskParameterDomain,
)
from alphalattice.investment.risk_research.experiments.contracts import (
    RiskDevelopmentProgramBinding,
)
from alphalattice.investment.risk_research.experiments.execution import (
    RiskDevelopmentExecutor,
)
from alphalattice.investment.risk_research.experiments.observation import NumericalCallCounter
from alphalattice.investment.risk_research.experiments.window import (
    resolve_development_input_binding,
)
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
)
from tests.researcher_methodology_surface.real_workspace import RealRiskWorkspace
from tests.researcher_methodology_surface.risk_development_program_support import (
    _admission,
)
from tests.researcher_methodology_surface.risk_development_support import (
    _binding,
    _bounded_authority,
)


class _RestatedCovarianceAdapter:
    """The installed adapter's identifiers over different numerical behaviour.

    Same ``adapter_id`` and same ``recipe_schema_id`` as the real covariance
    adapter, so every route that keys on an identifier accepts it, and a
    different declared binding, so anything that checks *content* does not. That
    is the whole shape of the "same identifier, changed code" problem, in one
    object.
    """

    adapter_id = CovarianceEstimatorAdapter().adapter_id
    recipe_schema_id = CovarianceEstimatorAdapter().recipe_schema_id

    def __init__(self) -> None:
        self._inner = CovarianceEstimatorAdapter()

    def describe_numerical_binding(self) -> RiskEstimatorNumericalBinding:
        original = self._inner.describe_numerical_binding()
        return RiskEstimatorNumericalBinding.create(
            adapter_id=original.adapter_id,
            estimate_content_format_id=original.estimate_content_format_id,
            implementation_owners=original.implementation_owners,
            implementation_content_hash=original.implementation_content_hash,
            deterministic_policy={**original.deterministic_policy, "revision": "restated"},
            required_runtime_capabilities=original.required_runtime_capabilities,
        )

    def validate_recipe(self, recipe: RiskEstimatorRecipeEnvelope) -> object:
        return self._inner.validate_recipe(recipe)

    def estimate(self, **kwargs: object) -> object:
        raise AssertionError("an adapter that is not the admitted one must never compute")


def test_changed_implementation_under_an_unchanged_id_changes_the_program() -> None:
    """The whole point of binding implementation identity, as an assertion."""

    installed = build_installed_risk_estimator_catalog()
    baseline = _binding()

    restated: RiskEstimatorAdapter = _RestatedCovarianceAdapter()  # type: ignore[assignment]
    changed = RiskEstimatorCatalog((restated,)).binding

    assert restated.adapter_id == installed.adapter_ids[0]
    assert changed.catalog_hash != installed.binding.catalog_hash
    moved = _binding(
        catalog_hash=changed.catalog_hash,
        selected_numerical_binding_hash=changed.ordered_capabilities[0].numerical_binding_hash,
    )
    assert moved.development_binding_hash != baseline.development_binding_hash
    # 3: this one is a *selected method* change, not merely governance -- the
    # adapter that will compute is the one that changed.
    assert moved.selected_method_binding_hash != baseline.selected_method_binding_hash


class _RestatedCovarianceCapability:
    """A capability holding an implementation the catalog does not install."""

    capability_handle = COVARIANCE_RECIPE_SCHEMA_ID
    randomness_policy = "NONE"

    @property
    def adapter(self) -> RiskEstimatorAdapter:
        return _RestatedCovarianceAdapter()  # type: ignore[return-value]

    @property
    def parameter_domain(self) -> RiskParameterDomain:
        return COVARIANCE_PARAMETER_DOMAIN

    def seal(self, admitted: dict[str, object]) -> tuple[object, str]:
        return CovarianceCapability().seal(admitted)


def test_a_capability_whose_adapter_is_not_the_installed_one_is_refused() -> None:
    """A capability whose adapter is not the installed one is refused."""

    with pytest.raises(ValueError, match="RISK_ESTIMATOR_CAPABILITY_BINDING_NOT_INSTALLED"):
        RiskEstimatorCatalog(
            (CovarianceEstimatorAdapter(),),
            capabilities=(_RestatedCovarianceCapability(),),  # type: ignore[arg-type]
        )


def test_a_separate_instance_of_the_same_implementation_stays_installable() -> None:
    """A separate instance of the same implementation stays installable."""

    installed = CovarianceEstimatorAdapter()
    capability = CovarianceCapability()

    assert capability.adapter is not installed
    catalog = RiskEstimatorCatalog((installed,), capabilities=(capability,))
    assert catalog.capability_handles == (COVARIANCE_RECIPE_SCHEMA_ID,)


def test_a_capability_handle_that_is_not_its_adapters_schema_is_refused() -> None:
    """The handle an author writes must be the schema the adapter decodes."""

    class _MisroutedCapability(_RestatedCovarianceCapability):
        capability_handle = "SOME_OTHER_SCHEMA"

        @property
        def adapter(self) -> RiskEstimatorAdapter:
            return CovarianceEstimatorAdapter()

    with pytest.raises(ValueError, match="RISK_ESTIMATOR_CAPABILITY_HANDLE_NOT_ITS_SCHEMA"):
        RiskEstimatorCatalog(
            (CovarianceEstimatorAdapter(),),
            capabilities=(_MisroutedCapability(),),  # type: ignore[arg-type]
        )


class _SwappingCatalog(RiskEstimatorCatalog):
    """Resolution hands back an implementation admission never saw.

    The construction-time guard cannot reach this: a catalog that decides what to
    return at resolve time is consistent when it is built. This is the residual
    the executor's own re-check exists for.
    """

    def resolve(self, recipe: RiskEstimatorRecipeEnvelope) -> RiskEstimatorAdapter:
        super().resolve(recipe)
        return _RestatedCovarianceAdapter()  # type: ignore[return-value]


def test_execution_refuses_an_adapter_that_is_not_the_one_admitted(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path: Path,
) -> None:
    """Execution refuses an adapter that is not the one admitted."""

    authority, _requested = _bounded_authority(real_risk_workspace, count=3)
    input_binding, bounded = resolve_development_input_binding(
        authority=authority,
        return_surface=real_risk_workspace.return_surface,
        return_reader=real_risk_workspace.return_reader,
        freshness_probe=real_risk_workspace.freshness_probe,
    )
    counter = NumericalCallCounter()
    output = tmp_path / "swapped"
    with pytest.raises(AuthoringError, match="resolved_numerical_binding_mismatch"):
        # Built from the installed adapter set so its catalog identity matches
        # the Program's: the point is to reach the executor's own re-check, not
        # the governance guard above it.
        RiskDevelopmentExecutor(
            estimators=_SwappingCatalog(build_installed_risk_estimator_catalog().adapters)
        ).execute(
            binding=_binding(),
            input_binding=input_binding,
            admission=_admission(),
            return_surface=real_risk_workspace.return_surface,
            return_reader=real_risk_workspace.return_reader,
            bounded_sessions=bounded,
            output_workspace=output,
            sector_by_listing_id=real_risk_workspace.sector_by_listing_id,
            counter=counter,
        )

    assert counter.estimate_calls == 0
    # Refused before the input binding is committed, so nothing at all was written.
    assert not output.exists() or not any(output.rglob("*.json"))


class _InertAdapter:
    """A second installed capability that is never selected.

    Deliberately not a covariance adapter: it has its own schema, so installing
    it changes nothing about how the covariance capability computes. That is the
    whole point -- governance moved, methodology did not.
    """

    adapter_id = "case-study.inert"
    recipe_schema_id = "CASE_STUDY_INERT_SCHEMA"

    def describe_numerical_binding(self) -> RiskEstimatorNumericalBinding:
        return RiskEstimatorNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimate_content_format_id="case-study-inert",
            implementation_owners=("case_study.inert",),
            implementation_content_hash="a" * 64,
            deterministic_policy={"kind": "inert"},
            required_runtime_capabilities=(),
        )

    def validate_recipe(self, recipe: RiskEstimatorRecipeEnvelope) -> object:
        raise AssertionError("an unselected adapter must never be asked to validate")

    def estimate(self, **kwargs: object) -> object:
        raise AssertionError("an unselected adapter must never compute")


def test_installing_an_unselected_adapter_moves_governance_but_not_the_method() -> None:
    """Installing an unselected adapter moves governance but not the method."""

    covariance = CovarianceEstimatorAdapter()
    catalog_a = RiskEstimatorCatalog((covariance,))
    catalog_b = RiskEstimatorCatalog((covariance, _InertAdapter()))  # type: ignore[arg-type]

    def _selected(catalog: RiskEstimatorCatalog) -> RiskDevelopmentProgramBinding:
        selected = next(
            value
            for value in catalog.binding.ordered_capabilities
            if value.adapter_id == covariance.adapter_id
        )
        return _binding(
            catalog_hash=catalog.binding.catalog_hash,
            selected_adapter_id=selected.adapter_id,
            selected_numerical_binding_hash=selected.numerical_binding_hash,
        )

    a = _selected(catalog_a)
    b = _selected(catalog_b)

    # Governance changed: the Host installed something new.
    assert a.catalog_hash != b.catalog_hash
    # Methodology did not: the same code will run on the same inputs.
    assert a.selected_method_binding_hash == b.selected_method_binding_hash
    # And the conflation is genuinely gone, rather than the two hashes happening
    # to agree because neither depends on the catalog.
    assert a.development_binding_hash != b.development_binding_hash


def test_changed_executable_content_moves_the_selected_numerical_binding() -> None:
    """Changed executable content moves the selected numerical binding."""

    original = CovarianceEstimatorAdapter().describe_numerical_binding()
    # Same adapter id, same declared owners, same policy -- different bytes.
    rewritten = RiskEstimatorNumericalBinding.create(
        adapter_id=original.adapter_id,
        estimate_content_format_id=original.estimate_content_format_id,
        implementation_owners=original.implementation_owners,
        implementation_content_hash="b" * 64,
        deterministic_policy=original.deterministic_policy,
        required_runtime_capabilities=original.required_runtime_capabilities,
    )

    assert rewritten.adapter_id == original.adapter_id
    assert rewritten.implementation_owners == original.implementation_owners
    assert rewritten.numerical_binding_hash != original.numerical_binding_hash

    baseline = _binding(selected_numerical_binding_hash=original.numerical_binding_hash)
    moved = _binding(selected_numerical_binding_hash=rewritten.numerical_binding_hash)
    assert moved.selected_method_binding_hash != baseline.selected_method_binding_hash


def test_the_declared_implementation_hash_is_the_real_module_bytes() -> None:
    """The declared implementation hash is the real module bytes."""

    from alphalattice.investment.risk_research.estimators import covariance, matrix_identity
    from alphalattice.investment.risk_research.estimators.contracts import (
        implementation_content_hash,
    )

    expected = implementation_content_hash(
        Path(covariance.__file__),
        Path(matrix_identity.__file__),
    )
    assert (
        CovarianceEstimatorAdapter().describe_numerical_binding().implementation_content_hash
        == (expected)
    )


@pytest.mark.parametrize(
    "field",
    [
        "catalog_hash",
        "recipe_hash",
        "parameter_domain_hash",
        "selected_adapter_id",
        "selected_numerical_binding_hash",
    ],
)
def test_every_bound_identity_is_consumed_by_the_program_hash(field: str) -> None:
    """An unread field would not count; each one has to move the identity."""

    baseline = _binding()
    moved = _binding(**{field: "b" * 64})

    assert moved.development_binding_hash != baseline.development_binding_hash


def test_a_tampered_binding_fails_its_own_identity_check() -> None:
    """A tampered binding fails its own identity check."""

    baseline = _binding()
    payload = baseline.model_dump(mode="json")
    payload["recipe_hash"] = "c" * 64

    with pytest.raises(ValueError, match="selected_method_binding_identity_invalid"):
        RiskDevelopmentProgramBinding(**payload)


def test_a_tampered_governance_field_fails_the_development_identity_check() -> None:
    """A catalog edit is not a methodology edit, and the errors say so."""

    payload = _binding().model_dump(mode="json")
    payload["catalog_hash"] = "c" * 64

    # The selected-method identity is untouched -- correctly, since nothing
    # about the computation changed -- so the outer identity is what catches it.
    with pytest.raises(ValueError, match="development_binding_identity_invalid"):
        RiskDevelopmentProgramBinding(**payload)


def test_a_code_closure_no_longer_moves_the_binding_and_an_earlier_one_reads_as_historical() -> (
    None
):
    """A Program binds meaning only (binding plan, P): the code that runs a study is bound by
    its plan's implementation hash, so the binding carries no closure. A binding sealed before
    P carries both closures, still parses with its own identity, and says it is historical."""

    current = _binding()
    assert not current.earlier_scheme
    assert "source_closure_hash" not in current.model_dump(mode="json")
    earlier = current.model_dump(mode="json", exclude={"selected_method_binding_hash"})
    # A binding sealed before P also folded the environment in, as every one did before E0.
    earlier |= {
        "source_closure_hash": "a" * 64,
        "development_source_closure_hash": "b" * 64,
        "numerical_environment_hash": "c" * 64,
    }
    earlier["selected_method_binding_hash"] = (
        RiskDevelopmentProgramBinding.selected_method_identity(
            **{
                k: earlier[k]
                for k in (
                    "selected_adapter_id",
                    "selected_numerical_binding_hash",
                    "recipe_hash",
                    "parameter_domain_hash",
                    "numerical_environment_hash",
                    "source_closure_hash",
                    "development_source_closure_hash",
                )
            }
        )
    )
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    earlier["development_binding_hash"] = canonical_hash(
        {k: v for k, v in earlier.items() if k != "development_binding_hash"}
    )
    parsed = RiskDevelopmentProgramBinding.model_validate(earlier)
    assert parsed.earlier_scheme
    assert parsed.development_binding_hash != current.development_binding_hash
