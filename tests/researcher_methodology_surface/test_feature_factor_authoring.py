"""Feature and Factor acceptance: reuse what exists, prove the dynamic axes.

Feature already owns a kernel registry, so this file proves extensibility rather
than building a second one. Factor gains an experiment compiler so a second Desk
kind reaches the same envelope, dispatcher, and sealing path.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from alphalattice.control.product_host.composition.research_authoring import (
    build_research_experiment_dispatcher,
    installed_desk_compilers,
)
from alphalattice.control.research_program.authoring.document import load_authoring_document
from alphalattice.foundation.factor_research.experiments.authoring import (
    DEVELOPMENT_CONTEXT_AXIS_CEILING,
    FACTOR_EXPERIMENT_KIND,
    FactorExperimentCompiler,
    FactorInventoryEntry,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
)
from alphalattice.foundation.feature_engine.producers.factors.open_intraday import (
    MEAN_ADJUSTED_RETURN_ID,
)
from alphalattice.foundation.feature_engine.producers.factors.registry import (
    FeatureKernelRegistry,
    RegisteredFeatureKernel,
)
from alphalattice.investment.risk_research.experiments.compiler import RISK_EXPERIMENT_KIND
from alphalattice.kernel.quant.factor_contracts import FactorSpec
from alphalattice.protocols.actor_execution.contracts import ActorKind
from alphalattice.protocols.research_authoring.contracts import AuthoringError
from tests.researcher_methodology_surface.real_workspace import RealRiskWorkspace

_FIXTURE = Path(__file__).parent / "fixtures" / "factor_screening_development.yaml"
_FACTOR_IDS = (
    "factor.desktop.momentum_12_1",
    "factor.desktop.short_reversal",
    "factor.desktop.liquidity_amihud",
    "factor.desktop.volatility_60d",
)


def _entries(*factor_ids: str) -> tuple[FactorInventoryEntry, ...]:
    """Inventory entries with distinct, deterministic method identities.

    Distinct per factor on purpose: an inventory that gave every factor the same
    identity would still move the catalog hash when a factor was added, so it
    could not tell "the axis changed" from "one factor was rewritten" -- which is
    the distinction this entry type exists to make.
    """

    return tuple(
        FactorInventoryEntry(
            factor_id=factor_id,
            implementation_hash=sha256(factor_id.encode()).hexdigest(),
            methodology_hash=sha256(f"method:{factor_id}".encode()).hexdigest(),
        )
        for factor_id in factor_ids
    )


_INVENTORY = _entries(*_FACTOR_IDS)


def _dispatcher(
    workspace: RealRiskWorkspace,
    inventory: tuple[FactorInventoryEntry, ...] = _INVENTORY,
) -> object:
    return build_research_experiment_dispatcher(
        workspace=workspace.workspace,
        factor_inventory=inventory,
    )


def _document(**factor_overrides: object) -> dict[str, object]:
    document = dict(load_authoring_document(_FIXTURE.read_text(encoding="utf-8")))
    if factor_overrides:
        section = dict(document["factor"])  # type: ignore[arg-type]
        section.update(factor_overrides)
        document["factor"] = section
    return document


def test_a_second_desk_kind_reaches_the_same_envelope_and_sealing_path(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    program, binding = _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
        _document(), actor_kind=ActorKind.HUMAN, actor_id="researcher"
    )
    assert program.kind == FACTOR_EXPERIMENT_KIND
    assert program.resolved_sessions
    assert binding.actor_kind is ActorKind.HUMAN


def test_both_desk_kinds_are_installed_without_touching_the_dispatcher() -> None:
    kinds = tuple(value.kind for value in installed_desk_compilers(factor_inventory=_INVENTORY))
    assert kinds == (RISK_EXPERIMENT_KIND, FACTOR_EXPERIMENT_KIND)


def test_the_factor_axis_is_dynamic_and_moves_catalog_identity(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    """Adding an ordinary Factor changes identity, never Desk source code."""

    baseline, _ = _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
        _document(), actor_kind=ActorKind.HUMAN, actor_id="researcher"
    )
    extended, _ = _dispatcher(  # type: ignore[attr-defined]
        real_risk_workspace, _entries(*_FACTOR_IDS, "factor.desktop.accruals")
    ).freeze(_document(), actor_kind=ActorKind.HUMAN, actor_id="researcher")
    assert extended.catalog_hash != baseline.catalog_hash

    # And the other half of the same claim: a Factor rewritten under an unchanged
    # ID moves the identity too. Binding bare IDs left this case indistinguishable
    # from no change at all, so old evidence stayed reusable across a rewrite.
    # The rewrite here changes only the *method* -- same id, same kernel -- which
    # is the case an implementation hash alone could not see.
    rewritten = (
        FactorInventoryEntry(
            factor_id=_FACTOR_IDS[0],
            implementation_hash=_INVENTORY[0].implementation_hash,
            methodology_hash="f" * 64,
        ),
        *_INVENTORY[1:],
    )
    restated, _ = _dispatcher(real_risk_workspace, rewritten).freeze(  # type: ignore[attr-defined]
        _document(), actor_kind=ActorKind.HUMAN, actor_id="researcher"
    )
    assert tuple(entry.factor_id for entry in rewritten) == _FACTOR_IDS
    assert restated.catalog_hash != baseline.catalog_hash

    # Selecting a different subset only changes YAML.
    subset, _ = _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
        _document(factor_ids=["factor.desktop.momentum_12_1"]),
        actor_kind=ActorKind.HUMAN,
        actor_id="researcher",
    )
    assert subset.desk_program_hash != baseline.desk_program_hash


def test_a_policy_no_program_chose_moves_no_program_and_refuses_none(
    real_risk_workspace: RealRiskWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (LAWS.md ID3, V91): a Factor Program binds the policies it chose, not
    the installed menu, so installing another screening or redundancy policy leaves every
    Program's identity and keeps every sealed method installed."""

    from alphalattice.foundation.factor_research.experiments import authoring, verification
    from alphalattice.foundation.factor_research.experiments.verification import (
        FactorEvidenceVerifier,
    )

    before, _ = _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
        _document(), actor_kind=ActorKind.HUMAN, actor_id="researcher"
    )
    for module in (authoring, verification):
        monkeypatch.setattr(
            module,
            "INSTALLED_SCREENING_POLICIES",
            (*authoring.INSTALLED_SCREENING_POLICIES, "ANOTHER_SCREENING"),
        )
        monkeypatch.setattr(
            module,
            "INSTALLED_REDUNDANCY_POLICIES",
            (*authoring.INSTALLED_REDUNDANCY_POLICIES, "ANOTHER_REDUNDANCY"),
        )
    after, _ = _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
        _document(), actor_kind=ActorKind.HUMAN, actor_id="researcher"
    )

    assert after == before
    assert FactorEvidenceVerifier._method_binding_refusal(program=before) is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"factor_ids": ["factor.desktop.not_installed"]}, "factor_id_not_in_inventory"),
        ({"screening_policy": "NOT_INSTALLED"}, "screening_policy_not_installed"),
        ({"redundancy_policy": "NOT_INSTALLED"}, "redundancy_policy_not_installed"),
        (
            {"factor_ids": ["factor.desktop.momentum_12_1", "factor.desktop.momentum_12_1"]},
            "factor_ids_duplicated",
        ),
    ],
    ids=["unknown-factor", "screening", "redundancy", "duplicate"],
)
def test_invalid_factor_configuration_fails_before_any_numerical_call(
    overrides: dict[str, object], message: str, real_risk_workspace: RealRiskWorkspace
) -> None:
    with pytest.raises(AuthoringError, match=message):
        _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
            _document(**overrides), actor_kind=ActorKind.HUMAN, actor_id="researcher"
        )


def test_the_candidate_budget_is_enforced_against_the_requested_axis(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    document = _document(factor_ids=list(_FACTOR_IDS))
    experiment = dict(document["experiment"])  # type: ignore[arg-type]
    experiment["budget"] = {"maximum_candidates": 2, "maximum_numerical_calls": 100}
    document["experiment"] = experiment
    with pytest.raises(AuthoringError, match="candidate_budget_exceeded"):
        _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
            document, actor_kind=ActorKind.HUMAN, actor_id="researcher"
        )


def test_the_development_axis_ceiling_is_not_the_frozen_catalog_breadth() -> None:
    """requirement: a development budget, not a historical fact standing in for one.

    ``55`` is how many Factors one frozen screening report happens to contain,
    and this compiler used to refuse any inventory longer than that. Under that
    rule the first method-family batch large enough to matter would have been
    refused by a *development* compiler quoting a published artifact. The frozen
    screening contracts that carried 55 retired with V312 (RT); the development
    path never borrows their breadth.
    """

    frozen_catalog_breadth = 55
    assert frozen_catalog_breadth < DEVELOPMENT_CONTEXT_AXIS_CEILING

    # An axis wider than the frozen catalog compiles, which is the whole change.
    wide = FactorExperimentCompiler(
        _entries(*(f"factor.desktop.f{index:03d}" for index in range(frozen_catalog_breadth + 1)))
    )
    assert len(wide.factor_inventory) == frozen_catalog_breadth + 1

    # Bounded, not unbounded: the ceiling is the multiple-testing denominator the
    # installed correction has to resolve, and it is still a hard refusal.
    with pytest.raises(AuthoringError, match="inventory_exceeds_development_ceiling"):
        FactorExperimentCompiler(
            _entries(
                *(
                    f"factor.desktop.f{index:04d}"
                    for index in range(DEVELOPMENT_CONTEXT_AXIS_CEILING + 1)
                )
            )
        )


def test_a_test_only_feature_kernel_installs_without_touching_the_feature_engine() -> None:
    """Feature already owns a kernel registry; adding a kernel is registration."""

    installed = default_extension_kernel_registry()
    assert isinstance(installed, FeatureKernelRegistry)
    # The installed extension kernel resolves; an unregistered id never does,
    # which is the property that keeps model-supplied Python out of the engine.
    assert installed.resolve(MEAN_ADJUSTED_RETURN_ID).implementation_id == MEAN_ADJUSTED_RETURN_ID
    with pytest.raises(ValueError, match="not code-owned"):
        installed.resolve("factor.case_study.constant")

    def _constant(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
        del specification
        return pd.Series([1.0] * len(source), index=source.index)

    extended = FeatureKernelRegistry(
        (
            installed.resolve(MEAN_ADJUSTED_RETURN_ID),
            RegisteredFeatureKernel(
                implementation_id="factor.case_study.constant",
                # A kernel must say which family's bytes compute it, and the
                # family may live anywhere importable -- here, this case study.
                method_family="CASE_STUDY_CONSTANT",
                method_family_owners=(__name__,),
                required_fields=("close",),
                implementation_hash="c" * 64,
                compute=_constant,
            ),
        )
    )
    assert extended.resolve("factor.case_study.constant").required_fields == ("close",)
