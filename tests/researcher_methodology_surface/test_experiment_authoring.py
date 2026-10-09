"""Prove an experiment can be composed from installed methods without Python.

This covers the second researcher workflow: a declarative document is compiled,
validated, and sealed by the Host into an immutable content-addressed Program.
The document is a request, never authority.

Every case resolves against the workspace the product actually wrote. A
generated weekday calendar can name sessions no Panel ever materialized, which
is exactly how an unresolvable handle used to reach a Desk compiler and how a
Program could look admissible while pointing at data that does not exist.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pytest

from alphalattice.control.product_host.composition.research_authoring import (
    build_research_experiment_dispatcher,
    installed_desk_compilers,
)
from alphalattice.control.research_program.authoring.document import load_authoring_document
from alphalattice.investment.risk_research.experiments.compiler import RISK_EXPERIMENT_KIND
from alphalattice.protocols.actor_execution.contracts import ActorKind
from alphalattice.protocols.research_authoring.contracts import AuthoringError
from tests.researcher_methodology_surface.real_workspace import RealRiskWorkspace

_FIXTURE = Path(__file__).parent / "fixtures" / "risk_covariance_development.yaml"


def _dispatcher(workspace: RealRiskWorkspace) -> object:
    return build_research_experiment_dispatcher(workspace=workspace.workspace)


def _document(**overrides: object) -> dict[str, object]:
    document = dict(load_authoring_document(_FIXTURE.read_text(encoding="utf-8")))
    experiment = dict(document["experiment"])  # type: ignore[arg-type]
    experiment.update(overrides)
    document["experiment"] = experiment
    return document


def test_yaml_compiles_to_an_immutable_program_with_exact_sessions(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    dispatcher = _dispatcher(real_risk_workspace)
    document = load_authoring_document(_FIXTURE.read_text(encoding="utf-8"))
    program, binding = dispatcher.freeze(  # type: ignore[attr-defined]
        document, actor_kind=ActorKind.HUMAN, actor_id="researcher"
    )

    assert program.kind == RISK_EXPERIMENT_KIND
    # The raw date range became ordered exact sessions read off the Panel's own
    # semantic index, so every one of them is a session the Panel materialized.
    assert program.resolved_sessions == tuple(sorted(set(program.resolved_sessions)))
    assert all(date(2024, 2, 1) <= value <= date(2024, 3, 1) for value in program.resolved_sessions)
    assert program.resolved_sessions
    assert binding.actor_kind is ActorKind.HUMAN


def test_resolved_authority_names_the_published_panel_and_universe(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    """The Program is bound to artifacts that exist, not to the handles asked for."""

    from alphalattice.control.product_host.research_authoring.authority import (
        WorkspaceResearchAuthorityResolver,
    )

    dispatcher = _dispatcher(real_risk_workspace)
    envelope = dispatcher.validate(  # type: ignore[attr-defined]
        load_authoring_document(_FIXTURE.read_text(encoding="utf-8"))
    )
    authority = WorkspaceResearchAuthorityResolver(workspace=real_risk_workspace.workspace).resolve(
        envelope
    )

    assert authority.panel_snapshot_hash == real_risk_workspace.panel_snapshot_hash
    assert authority.panel_manifest_ref == real_risk_workspace.panel_manifest_ref
    assert authority.universe_revision_sha256 == real_risk_workspace.manifest.revision_sha256
    assert set(authority.ordered_listing_ids) == set(real_risk_workspace.sector_by_listing_id)
    assert authority.source_watermark_hash
    assert any(
        item.startswith("baseline_features:")
        for item in real_risk_workspace.manifest.qualification_obligations
    )


def test_an_unresolvable_handle_fails_before_any_desk_compilation(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    """An unresolvable handle fails before any desk compilation."""

    dispatcher = _dispatcher(real_risk_workspace)
    with pytest.raises(AuthoringError, match="universe_handle_unresolved"):
        dispatcher.freeze(  # type: ignore[attr-defined]
            _document(universe_handle="no-such-universe"),
            actor_kind=ActorKind.HUMAN,
            actor_id="researcher",
        )
    with pytest.raises(AuthoringError, match="snapshot_handle_unresolved"):
        dispatcher.freeze(  # type: ignore[attr-defined]
            _document(data_snapshot_handle="a" * 64),
            actor_kind=ActorKind.HUMAN,
            actor_id="researcher",
        )
    with pytest.raises(AuthoringError, match="snapshot_handle_unresolved"):
        dispatcher.freeze(  # type: ignore[attr-defined]
            _document(data_snapshot_handle="not-a-snapshot"),
            actor_kind=ActorKind.HUMAN,
            actor_id="researcher",
        )


def test_a_range_outside_the_panel_axis_resolves_to_nothing(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    """A generated calendar would have happily produced sessions here."""

    with pytest.raises(AuthoringError, match="no_sessions_resolved"):
        _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
            _document(
                sessions={
                    "start": "2011-02-01",
                    "end": "2011-03-01",
                    "as_of": {"session": "2011-03-15", "phase": "OFFICIAL_CLOSE"},
                }
            ),
            actor_kind=ActorKind.HUMAN,
            actor_id="researcher",
        )


def test_the_same_document_and_authority_produce_the_same_program_identity(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    text = _FIXTURE.read_text(encoding="utf-8")
    first, _ = _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
        load_authoring_document(text), actor_kind=ActorKind.HUMAN, actor_id="researcher"
    )
    replayed, _ = _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
        load_authoring_document(text), actor_kind=ActorKind.HUMAN, actor_id="researcher"
    )
    assert replayed.program_hash == first.program_hash


def test_every_actor_reaches_the_same_host_compiler_and_program(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    """Every actor reaches the same host compiler and program."""

    text = _FIXTURE.read_text(encoding="utf-8")
    programs = []
    bindings = []
    for actor_kind in (
        ActorKind.HUMAN,
        ActorKind.EXTERNAL_AUTOMATION,
        ActorKind.HOST_FALLBACK,
    ):
        program, binding = _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
            load_authoring_document(text), actor_kind=actor_kind, actor_id="actor"
        )
        programs.append(program)
        bindings.append(binding)
        assert binding.actor_kind is actor_kind
    assert len({value.desk_program_hash for value in programs}) == 1
    assert len({value.program_hash for value in programs}) == 1
    assert len({value.binding_hash for value in bindings}) == len(bindings)


def test_the_actor_seals_over_the_compiled_method_not_only_the_envelope(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    """The actor seals over the compiled method not only the envelope."""

    text = _FIXTURE.read_text(encoding="utf-8")
    program, binding = _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
        load_authoring_document(text), actor_kind=ActorKind.HUMAN, actor_id="researcher"
    )

    assert binding.submission_hash == program.program_hash
    assert binding.submission_hash != program.envelope_hash
    # The compiled method and resolved authority are what the envelope cannot carry.
    assert program.method_binding_hash and program.parameter_domain_hash
    assert program.method_binding_hash != program.envelope_hash
    assert program.authority_hash != program.envelope_hash


def test_a_cli_invocation_cannot_claim_installed_agent_provenance(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    """The actor-execution seam requires real Agent evidence for that claim."""

    with pytest.raises(ValueError, match="Agent execution evidence"):
        _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
            load_authoring_document(_FIXTURE.read_text(encoding="utf-8")),
            actor_kind=ActorKind.INSTALLED_AGENT,
            actor_id="agent",
        )


def test_a_seed_no_risk_method_consumes_cannot_create_a_second_identity(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    """A seed no risk method consumes cannot create a second identity."""

    with pytest.raises(ValueError, match="authoring_seed_not_applicable"):
        _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
            _document(determinism={"seed": 7, "thread_limit": 1, "network_disabled": True}),
            actor_kind=ActorKind.HUMAN,
            actor_id="researcher",
        )


@pytest.mark.parametrize(
    ("overrides", "changes_identity"),
    [
        (
            {
                "sessions": {
                    "start": "2024-02-05",
                    "end": "2024-03-01",
                    "as_of": {"session": "2024-03-15", "phase": "OFFICIAL_CLOSE"},
                }
            },
            True,
        ),
        ({"budget": {"maximum_candidates": 8, "maximum_numerical_calls": 5000}}, True),
    ],
    ids=["dates", "budget"],
)
def test_changing_the_document_changes_the_program_identity(
    overrides: dict[str, object],
    changes_identity: bool,
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    baseline, _ = _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
        _document(), actor_kind=ActorKind.HUMAN, actor_id="researcher"
    )
    changed, _ = _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
        _document(**overrides), actor_kind=ActorKind.HUMAN, actor_id="researcher"
    )
    assert (changed.program_hash != baseline.program_hash) is changes_identity


def test_a_thread_limit_is_bound_and_one_the_estimator_cannot_honour_is_refused_at_seal(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    """Regression: the thread limit is part of the envelope's identity, and a Risk plan
    whose installed estimator runs single-threaded is refused when it is sealed, at PLAN, where
    a sealed Program with two threads was refused only at RUN."""

    from alphalattice.protocols.research_authoring.contracts import ResearchExperimentEnvelope

    two = {"determinism": {"seed": 0, "thread_limit": 2, "network_disabled": True}}
    one = ResearchExperimentEnvelope.create(**_document()["experiment"])  # type: ignore[arg-type]
    moved = ResearchExperimentEnvelope.create(**_document(**two)["experiment"])  # type: ignore[arg-type]
    assert moved.envelope_hash != one.envelope_hash
    with pytest.raises(
        AuthoringError, match=r"research_authoring\.thread_limit_violates_adapter_requirement"
    ):
        _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
            _document(**two), actor_kind=ActorKind.HUMAN, actor_id="researcher"
        )


def test_documents_cannot_name_python_modules_or_callables() -> None:
    for text in (
        "experiment:\n  python_module: alphalattice.evil\n",
        "experiment:\n  kind: alphalattice.investment.risk_research.experiments.compiler:run\n",
        "experiment: !!python/object:os.system {}\n",
    ):
        with pytest.raises(AuthoringError):
            load_authoring_document(text)


def test_configuration_cannot_request_current_holdout_or_publication(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    with pytest.raises((AuthoringError, ValueError)):
        _dispatcher(real_risk_workspace).validate(  # type: ignore[attr-defined]
            _document(publication_intent="CURRENT_ACTIVATION")
        )


def test_uninstalled_desk_kind_and_capability_fail_before_any_numerical_call(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    dispatcher = _dispatcher(real_risk_workspace)
    with pytest.raises(AuthoringError, match="desk_kind_not_installed"):
        dispatcher.validate(_document(kind="factor.not-installed"))  # type: ignore[attr-defined]

    document = _document()
    document["risk"] = {"estimator": {"capability": "NOT_INSTALLED", "parameters": {}}}
    with pytest.raises(AuthoringError, match="authoring_capability_not_installed"):
        dispatcher.freeze(  # type: ignore[attr-defined]
            document, actor_kind=ActorKind.HUMAN, actor_id="researcher"
        )


def test_a_parameter_outside_the_declared_domain_is_rejected(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    """A parameter outside the declared domain is rejected."""

    document = _document()
    document["risk"] = {
        "estimator": {
            "capability": "EWMA_STANDARDIZED_LEDOIT_WOLF_CORRELATION",
            "parameters": {"ewma_decay": 0.5},
        }
    }
    with pytest.raises(AuthoringError, match="parameter_value_not_admitted"):
        _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
            document, actor_kind=ActorKind.HUMAN, actor_id="researcher"
        )

    document["risk"] = {
        "estimator": {
            "capability": "EWMA_STANDARDIZED_LEDOIT_WOLF_CORRELATION",
            "parameters": {"no_such_axis": 1},
        }
    }
    with pytest.raises(AuthoringError, match="parameter_axis_not_declared"):
        _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
            document, actor_kind=ActorKind.HUMAN, actor_id="researcher"
        )


def test_an_unmentioned_axis_takes_its_declared_default(
    real_risk_workspace: RealRiskWorkspace,
) -> None:
    """An author states what they are choosing, not the whole recipe."""

    document = _document()
    document["risk"] = {
        "estimator": {
            "capability": "EWMA_STANDARDIZED_LEDOIT_WOLF_CORRELATION",
            "parameters": {},
        }
    }
    stated, _ = _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
        document, actor_kind=ActorKind.HUMAN, actor_id="researcher"
    )
    document["risk"]["estimator"]["parameters"] = {"ewma_decay": 0.94}  # type: ignore[index]
    restated, _ = _dispatcher(real_risk_workspace).freeze(  # type: ignore[attr-defined]
        document, actor_kind=ActorKind.HUMAN, actor_id="researcher"
    )
    assert restated.program_hash == stated.program_hash


def test_overlapping_workspaces_are_refused(real_risk_workspace: RealRiskWorkspace) -> None:
    # The envelope validator runs inside pydantic, so the AuthoringError surfaces
    # as a ValidationError; both are ValueError and both fail closed.
    with pytest.raises(ValueError, match="workspace_nested"):
        _dispatcher(real_risk_workspace).validate(  # type: ignore[attr-defined]
            _document(baseline_workspace="workspaces/research/risk-development/inner")
        )
    with pytest.raises(ValueError, match="workspace_identical"):
        _dispatcher(real_risk_workspace).validate(  # type: ignore[attr-defined]
            _document(baseline_workspace="workspaces/research/risk-development")
        )


def test_adding_a_method_does_not_change_the_installed_dispatcher() -> None:
    """Only a whole new Desk kind touches Host composition."""

    assert tuple(value.kind for value in installed_desk_compilers()) == (RISK_EXPERIMENT_KIND,)


def test_an_exploration_sample_is_named_by_its_handle_and_draws_the_same_names() -> None:
    """An exploration sample is named by its handle and draws the same names."""

    from alphalattice.control.product_host.composition.research_experiment_projection import (
        research_lane,
    )
    from alphalattice.control.product_host.research_authoring.authority import (
        EXPLORATION_SAMPLE_KINDS,
        AuthoringError,
        exploration_sample,
        exploration_sample_size,
        universe_profile,
    )

    axis = tuple(f"L{n:04d}" for n in range(120))
    sectors = dict.fromkeys(axis, "Sector")
    small, large = exploration_sample(axis, 6, sectors), exploration_sample(axis, 80, sectors)

    assert len(small) == 6 and len(large) == 80 and set(small) < set(large)
    assert small == tuple(sorted(small)), "the sample keeps the axis order"
    assert set(exploration_sample(tuple(reversed(axis)), 6, sectors)) == set(small)
    assert (universe_profile("us.sample-80"), exploration_sample_size("us.sample-80")) == ("us", 80)
    assert (universe_profile("us"), exploration_sample_size("us")) == ("us", None)
    for handle in ("us.sample-080", "us.sample-x", "us.sample-"):
        with pytest.raises(AuthoringError, match=r"research_lane\.sample_handle_invalid"):
            exploration_sample_size(handle)
    assert {
        "factor.screening-development",
        "alpha.model-development",
        "portfolio.policy-development",
    } == EXPLORATION_SAMPLE_KINDS
    assert research_lane({"experiment": {"universe_handle": "us.sample-80"}}) == "EXPLORATION"
    assert research_lane({"experiment": {"universe_handle": "us"}}) == "PROMOTION"


def test_the_universe_is_offered_whole_or_sampled_in_the_sizes_admitted() -> None:
    """requirement (R4): a draft offers its universe whole or as a sample, from the metric
    policy's minimum cross-section to one fewer than the Panel's names, the sizes the authority
    admits."""

    from alphalattice.control.product_host.research_authoring.authority import (
        EXPLORATION_SAMPLE_MARK,
        MINIMUM_EXPLORATION_SAMPLE,
        universe_control,
    )

    control = universe_control("us.sample-80", 120)
    assert (control["path"], control["type"]) == (["experiment", "universe_handle"], "universe")
    assert (control["value"], control["whole"], control["sample_mark"]) == (
        "us.sample-80",
        "us",
        EXPLORATION_SAMPLE_MARK,
    )
    assert (control["sample_min"], control["sample_max"]) == (MINIMUM_EXPLORATION_SAMPLE, 119)
    assert universe_control("us", 120)["whole"] == "us"


def test_an_exported_declaration_reads_back_as_it_was_written() -> None:
    """Requirement: YAML the Host writes reads back through the declaration loader to the
    same values: a string that looks like an exponent stays a string, a number stays a number."""

    import yaml  # type: ignore[import-untyped]

    from alphalattice.protocols.research_authoring.selection import (
        dump_declaration,
        load_safe_yaml_document,
        rewrite_in_declaration_dialect,
    )

    document = {
        "feature": {"reason": "1e-10", "floor": 1e-10, "note": "风险说明"},
        "kept": ["2024-03-15", 3],
    }
    assert load_safe_yaml_document(dump_declaration(document)) == document
    exported = yaml.safe_dump(document, sort_keys=False)
    # PyYAML's own dumper leaves the string bare, and the declaration loader reads a number.
    assert load_safe_yaml_document(exported)["feature"]["reason"] == 1e-10
    assert load_safe_yaml_document(rewrite_in_declaration_dialect(exported)) == document


def test_every_desk_section_refuses_a_key_its_contract_does_not_name() -> None:
    """Regression: the ordinary Alpha section compared no key set, so a misspelled key
    was read by nothing and kept in the plan's identity, and three Desks refused one each their
    own way; each section is a typed contract and one code names the key where it was written."""

    from alphalattice.foundation.factor_research.experiments.authoring import FactorSection
    from alphalattice.investment.alpha_research.experiments.authoring import (
        AlphaExperimentCompiler,
    )
    from alphalattice.investment.alpha_research.experiments.lifecycle_authoring import (
        AlphaLifecycleSection,
    )
    from alphalattice.investment.risk_research.experiments.compiler import (
        RiskEstimatorSection,
        RiskSection,
    )
    from alphalattice.protocols.research_authoring.contracts import (
        DeskSection,
        refuse_unknown_section_keys,
    )

    # The key set is checked before any catalog is asked, so none is built.
    compiler = AlphaExperimentCompiler(
        target_recipes=Mock(),
        model_mandate=Mock(),
        model_catalog=Mock(),
        panel_factor_ids=("a",),
        factor_evidence_factor_ids=("a",),
        factor_evidence_checkpoint_hash="a" * 64,
    )
    section = {
        "target_recipe_id": "t",
        "ordered_feature_ids": ["a"],
        "model_capability_handle": "m",
        "model_parameters": {},
        "factor_evidence_handle": "e",
        "evaluation_policy_id": "p",
    }
    with pytest.raises(
        AuthoringError,
        match=r"^research_authoring\.section_key_unknown:alpha\.evaluation_policy_id$",
    ):
        compiler.compile_desk_program(
            envelope=Mock(),
            document={"alpha": section},
            authority=Mock(panel_snapshot_hash="b" * 64),
        )
    written: tuple[tuple[type[DeskSection], str, dict[str, object]], ...] = (
        (
            FactorSection,
            "factor",
            {"factor_ids": [], "screening_policy": "s", "redundancy_policy": "r", "factorids": []},
        ),
        (RiskSection, "risk", {"estimator": {}, "parameters": {}}),
        (RiskEstimatorSection, "risk.estimator", {"capability": "c", "ewma_decay": 0.9}),
        (
            AlphaLifecycleSection,
            "alpha",
            {"methodology_id": "M", "component_recipe_id": "c", "lifecycle": {}, "seeds": [1]},
        ),
    )
    for contract, place, document in written:
        (extra,) = set(document) - set(contract.model_fields)
        with pytest.raises(AuthoringError) as refused:
            refuse_unknown_section_keys(document, contract, place=place)
        assert str(refused.value) == f"research_authoring.section_key_unknown:{place}.{extra}"
