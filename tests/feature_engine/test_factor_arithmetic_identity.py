"""Factor identity names the code that computes the values, by its rule.

Four identities in this capability all read like implementation identity and none
of them contained any: the catalog's ``formula_implementation_hash`` was a
hand-written description of the engine, the core bundle hashed declared fields, a
core Factor's identity was ``{bundle_hash, factor_id, formula_ref}``, and a
registered extension kernel declared its own algorithm summary. Every one of them
survives an arbitrary rewrite of the arithmetic.

That is not a theoretical gap. A Panel identity successor took equal catalog
hashes as proof that the current implementation produced a historical Panel's
values, and backfilled per-factor identities onto them on that basis. The
inference had no evidence behind it, and these cases exist so it cannot be made
again: the measured source closure is folded into every one of those identities,
so "the code changed" is a question the identity can answer.

What these cases now also pin is the *granularity*, in two steps that were
taken for the same reason and a milestone apart. One closure containing the
shared materializer and every extension kernel answered the question too loudly:
editing an unshipped experimental Factor rotated the implementation identity of
all maintained controls. Splitting control from family fixed that and left a
narrower version of it behind -- the family closures still contained the module
that *lists* the installed kernels, so registering a second family rotated an
unchanged first one. Installing something is not editing it, and the whole point
of a per-family closure is that adding a method does not invalidate unrelated
methods.

Registry mechanics and installed composition are separate modules now. Which
kernels a build installs is a real identity and has its own name --
``FeatureKernelRegistry.installed_capability_hash`` -- and it reaches no factor's
own identity. Every claim below is proved on the walk each identity hashes, on the
switch that kept what was sealed before the rule, or by composing a different registry.
"""

from __future__ import annotations

import json
import math
from datetime import date
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from alphalattice.foundation.feature_engine.catalog.contracts import (
    FeatureCatalog,
    desktop_core_feature_bundle,
)
from alphalattice.foundation.feature_engine.catalog.observation_clock import observation_clock_for
from alphalattice.foundation.feature_engine.producers.arithmetic_identity import (
    EXTENSION_KERNEL_OWNERS,
    FACTOR_METHOD_FAMILY_OWNERS,
    FACTOR_VALUE_ROLE,
    SHARED_FACTOR_ARITHMETIC_OWNERS,
    WALK_EXCLUDED,
    control_arithmetic_content_hash,
    control_arithmetic_rule_identity,
    feature_component_identity,
    installed_method_family_owners,
    method_family_content_hash,
)
from alphalattice.foundation.feature_engine.producers.factors import registry as registry_module
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    catalog_implementation_hashes,
    catalog_methodology_hashes,
    default_extension_kernel_registry,
    extension_factor_specs,
    installed_formula_specs,
)
from alphalattice.foundation.feature_engine.producers.factors.open_intraday import (
    MEAN_ADJUSTED_RETURN_ID,
    OPEN_INTRADAY_METHOD_FAMILY,
    OVERNIGHT_RETURN_ID,
    overnight_return_factor_spec,
)
from alphalattice.foundation.feature_engine.producers.factors.registry import (
    FeatureKernelRegistry,
    RegisteredFeatureKernel,
    factor_methodology_hash,
)
from alphalattice.foundation.feature_engine.producers.factors.session_liquidity import (
    SESSION_DOLLAR_VOLUME_ID,
    raw_session_dollar_volume,
    session_liquidity_factor_specs,
)
from alphalattice.foundation.feature_engine.producers.factors.session_observation import (
    SESSION_OBSERVATION_DECLARATIONS,
    SESSION_OBSERVATION_FACTOR_IDS,
    SESSION_OBSERVATION_IMPLEMENTATION_IDS,
    SESSION_OBSERVATION_KERNELS,
    session_observation_factor_specs,
)
from alphalattice.foundation.feature_engine.producers.factors.specifications import (
    admit_factor_development_capabilities,
)
from alphalattice.kernel.quant.factor_contracts import FactorSpec
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.identity_successors import recorded_origin
from alphalattice.kernel.shared_kernel.source_identity import (
    IDENTITY_SWITCH_PATH,
    number_deciding_closure,
    number_deciding_rule,
    source_syntax_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
_FAMILY_MODULES = frozenset(
    module for owners in FACTOR_METHOD_FAMILY_OWNERS.values() for module in owners
)
_COMPOSITION = "alphalattice.foundation.feature_engine.producers.factors.catalog"
_MECHANICS = "alphalattice.foundation.feature_engine.producers.factors.registry"
_CONTRACTS = "alphalattice.foundation.feature_engine.catalog.contracts"


def _walk(owners: tuple[str, ...], excluded: frozenset[str]) -> frozenset[str]:
    """The modules an identity's rule walk hashes, as ``feature_component_identity`` walks."""

    inside = tuple(owner for owner in owners if owner.startswith("alphalattice."))
    return frozenset(
        number_deciding_closure(
            inside, root=ROOT, rule=number_deciding_rule(ROOT), excluded=excluded
        )
    )


def _control_walk() -> frozenset[str]:
    return _walk(
        SHARED_FACTOR_ARITHMETIC_OWNERS, WALK_EXCLUDED | frozenset(EXTENSION_KERNEL_OWNERS)
    )


def _family_walk(owners: tuple[str, ...]) -> frozenset[str]:
    return _walk(
        (*SHARED_FACTOR_ARITHMETIC_OWNERS, *EXTENSION_KERNEL_OWNERS, *owners), WALK_EXCLUDED
    )


def _switched(component: str) -> dict[str, str]:
    table = json.loads((ROOT / IDENTITY_SWITCH_PATH).read_text(encoding="utf-8"))
    return dict(table["components"][component])


def _installed_family_hash(family: str) -> str:
    return method_family_content_hash(family, installed_method_family_owners(family))


def test_the_control_closure_names_only_the_shared_arithmetic() -> None:
    """requirement: the control closure is the shared arithmetic and what it runs, nothing else.

    The materializer and the parent formula module are its owners, and the rule walks what
    they import (the observation clock that decides the skipped sessions, the factor
    contracts). Catalog contracts stay outside: editing an admission rule must move catalog
    identity, not claim the arithmetic changed. No extension family and no registry mechanics
    are in it, which is the control/extension split.
    """

    assert SHARED_FACTOR_ARITHMETIC_OWNERS == (
        "alphalattice.foundation.feature_engine.producers.base_materializer",
        "alphalattice.kernel.quant.factor_formulas",
    )
    walk = _control_walk()
    assert set(SHARED_FACTOR_ARITHMETIC_OWNERS) <= walk
    assert not walk & _FAMILY_MODULES
    assert _MECHANICS not in walk and _COMPOSITION not in walk and _CONTRACTS not in walk
    # The switch kept the value every Panel was sealed under.
    assert control_arithmetic_content_hash() == _switched("FACTOR_VALUE:controls")["byte"]


def test_a_method_family_closure_contains_its_module_and_the_shared_owners() -> None:
    """requirement: a family binds its own formulas *and* what post-processes them.

    A closure naming only the family module would stand still through a rewrite of the
    materializer that projects its inputs and screens its outputs.
    """

    owners = FACTOR_METHOD_FAMILY_OWNERS[OPEN_INTRADAY_METHOD_FAMILY]
    assert owners == ("alphalattice.foundation.feature_engine.producers.factors.open_intraday",)
    assert {*owners, *SHARED_FACTOR_ARITHMETIC_OWNERS, _MECHANICS} <= _family_walk(owners)
    assert (
        _installed_family_hash(OPEN_INTRADAY_METHOD_FAMILY)
        == (_switched(f"FACTOR_VALUE:{OPEN_INTRADAY_METHOD_FAMILY}")["byte"])
    )
    # The *product* refuses a family it does not install; the identity function
    # itself takes owners, so an external consumer measures the same way.
    with pytest.raises(ValueError, match="factor_method_family_not_installed"):
        installed_method_family_owners("NO_SUCH_FAMILY")
    with pytest.raises(ValueError, match="factor_method_family_owners_required"):
        method_family_content_hash("NO_SUCH_FAMILY", ())


def test_a_family_edit_cannot_reach_a_control_and_a_shared_edit_reaches_both() -> None:
    """requirement: identity isolation where it holds, and reach where the code is shared.

    An edit reaches exactly the closures that hash its module: no family module is in the
    control walk, and the materializer, which projects and screens every extension series,
    is in the controls' walk and in every family's.
    """

    control = _control_walk()
    materializer = SHARED_FACTOR_ARITHMETIC_OWNERS[0]
    assert not control & _FAMILY_MODULES
    assert materializer in control
    for family, owners in FACTOR_METHOD_FAMILY_OWNERS.items():
        assert materializer in _family_walk(owners), family


def test_editing_one_family_leaves_an_unrelated_family_bit_identical() -> None:
    """requirement: family isolation runs both ways, not only against the controls.

    No family's walk hashes another family's own module, so two families installed side by
    side stay apart when one is edited.
    """

    for family, owners in FACTOR_METHOD_FAMILY_OWNERS.items():
        others = _FAMILY_MODULES - set(owners)
        assert not _family_walk(owners) & others, family
    # An external family measures its own module and the shared owners, nothing installed.
    assert not _family_walk(_SECOND_FAMILY_OWNERS) & _FAMILY_MODULES


def test_registry_mechanics_reach_every_family_and_composition_reaches_none() -> None:
    """requirement: the split is exactly where the values are shaped.

    The registry resolves a kernel, validates its inputs, checks the returned shape and
    coerces the numeric domain, so every family binds it. The module that *lists* which
    kernels a build installs decides no value, so no closure walks it, and that is what
    makes installing family B invisible to family A.
    """

    assert (_MECHANICS,) == EXTENSION_KERNEL_OWNERS
    assert _COMPOSITION in WALK_EXCLUDED
    for owners in (*FACTOR_METHOD_FAMILY_OWNERS.values(), _SECOND_FAMILY_OWNERS):
        walk = _family_walk(owners)
        assert _MECHANICS in walk and _COMPOSITION not in walk
    assert _MECHANICS not in _control_walk()


def test_a_comment_moves_no_identity_and_a_changed_rule_is_a_new_one(tmp_path: Path) -> None:
    """requirement: identity follows syntax, and the switch keeps what was sealed (V70, V231).

    A module's rule digest ignores its comments and docstrings, so C8's docstrings move
    nothing. The switch recorded each component's rule value beside its byte value: while the
    walk gives that rule value, the component keeps its byte value; a component whose rule the
    switch did not record carries its rule value, and a changed rule is a new value its role
    reads, which a Panel binds only through a recorded move (LAWS.md ID1, V345).
    """

    spec = find_spec(SHARED_FACTOR_ARITHMETIC_OWNERS[1])
    assert spec is not None and spec.origin is not None
    formulas = Path(spec.origin)
    commented = tmp_path / "factor_formulas.py"
    commented.write_text(
        '"""A new docstring."""\n# a comment\n' + formulas.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assert source_syntax_sha256(commented) == source_syntax_sha256(formulas)

    recorded = _switched("FACTOR_VALUE:controls")
    unrecorded = feature_component_identity(
        "FACTOR_VALUE:a-component-the-switch-never-saw",
        owners=SHARED_FACTOR_ARITHMETIC_OWNERS,
        excluded=WALK_EXCLUDED | frozenset(EXTENSION_KERNEL_OWNERS),
        semantic_owner="feature_engine.producers",
        numerical_role="FACTOR_VALUE",
    )
    assert recorded["rule"] != recorded["byte"]
    assert control_arithmetic_rule_identity() == (
        recorded["byte"] if unrecorded == recorded["rule"] else unrecorded
    )
    assert control_arithmetic_content_hash() == recorded_origin(
        f"{FACTOR_VALUE_ROLE}.controls", control_arithmetic_rule_identity()
    )
    assert control_arithmetic_content_hash() == recorded["byte"]


def test_a_registered_kernel_must_measure_the_module_that_owns_its_callable() -> None:
    """regression: a declared closure cannot omit the code that actually runs."""

    def _local_kernel(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
        del specification
        return pd.Series([0.0] * len(source), index=source.index)

    with pytest.raises(ValueError, match="compute owner is absent"):
        RegisteredFeatureKernel(
            implementation_id="factor.case-study.unmeasured",
            method_family="CASE_STUDY_UNMEASURED",
            method_family_owners=(
                "alphalattice.foundation.feature_engine.producers.factors.open_intraday",
            ),
            required_fields=("close",),
            implementation_hash="a" * 64,
            compute=_local_kernel,
        )


def test_every_factor_identity_folds_the_measured_arithmetic() -> None:
    """requirement: a rewritten materializer cannot keep the identities it had.

    Checked by re-deriving each identity with a different closure value and
    requiring every one of them to move. Declarations alone would leave all base
    core identities and the extension identity untouched, which is exactly the
    state that let a Panel successor claim provenance it could not prove.
    """

    catalog = FeatureCatalog.load()
    registry = default_extension_kernel_registry()
    bundle = desktop_core_feature_bundle()
    implementations = catalog_implementation_hashes(catalog, registry=registry)
    methodologies = catalog_methodology_hashes(catalog, registry=registry)
    assert set(implementations) == set(catalog.factor_ids)
    assert set(methodologies) == set(catalog.factor_ids)
    assert len(set(implementations.values())) == len(catalog.factor_ids)
    assert len(set(methodologies.values())) == len(catalog.factor_ids)
    assert {item.factor_id for item in installed_formula_specs(catalog)} == (
        set(catalog.factor_ids) | {item.factor_id for item in extension_factor_specs()}
    )

    closure = control_arithmetic_content_hash()
    formula_refs = {spec.factor_id: spec.formula_ref for spec in catalog.factors}

    def _core_identity(factor_id: str, source_content: str) -> str:
        return str(
            canonical_hash(
                {
                    "owner": bundle.bundle_hash,
                    "factor_id": factor_id,
                    "formula_ref": formula_refs[factor_id],
                    "source_content": source_content,
                }
            )
        )

    for factor_id in catalog.factor_ids:
        assert implementations[factor_id] == _core_identity(factor_id, closure), factor_id
        assert implementations[factor_id] != _core_identity(factor_id, "0" * 64), factor_id

    extension = overnight_return_factor_spec()
    kernel = registry.resolve(extension.formula_ref)
    assert kernel.method_family == OPEN_INTRADAY_METHOD_FAMILY
    assert registry.implementation_hash(extension, core_bundle=bundle) == canonical_hash(
        {
            "declared": kernel.implementation_hash,
            "method_family": OPEN_INTRADAY_METHOD_FAMILY,
            "source_content": _installed_family_hash(OPEN_INTRADAY_METHOD_FAMILY),
        }
    )
    # The declared summary alone is no longer the identity.
    assert registry.implementation_hash(extension, core_bundle=bundle) != (
        kernel.implementation_hash
    )
    # And the extension does not borrow the controls' closure, or the isolation
    # above would be undone one layer up.
    assert _installed_family_hash(OPEN_INTRADAY_METHOD_FAMILY) != closure


def test_the_catalog_binding_contains_the_arithmetic_it_describes() -> None:
    """requirement: equal catalog hashes must mean equal arithmetic.

    This is the property the withdrawn Panel claim assumed and did not have. The
    binding's formula identity is rebuilt here with a different closure value and
    must produce a different catalog hash; if it did not, "this catalog produced
    that Panel" would remain unfalsifiable.
    """

    catalog = FeatureCatalog.load()
    binding = catalog.binding
    assert binding.catalog_hash == FeatureCatalog.load().binding.catalog_hash

    payload = catalog.to_payload()
    rebuilt = FeatureCatalog.from_payload(payload)
    assert rebuilt.binding.catalog_hash == binding.catalog_hash
    assert rebuilt.binding.formula_implementation_hash == binding.formula_implementation_hash

    # The formula identity is a function of the measured closure: recomputing it
    # without that term yields a different value, so the term is load-bearing.
    without_closure = canonical_hash(
        {"engine": "playpen.feature_engine.desktop_finite_window_materializer"}
    )
    assert binding.formula_implementation_hash != without_closure


@pytest.mark.parametrize("accessor", [control_arithmetic_content_hash])
def test_the_closure_is_stable_within_a_process(accessor: Any) -> None:
    """requirement: identity is cached, not re-read per call, and never varies."""

    assert callable(accessor)
    first = accessor()
    second = accessor()
    assert first == second
    assert len(first) == 64 and set(first) <= set("0123456789abcdef")


_SECOND_FAMILY = "CASE_STUDY_SECOND_FAMILY"
_SECOND_FAMILY_OWNERS = (__name__,)
"""A second method family whose formulas live in this file.

Which is exactly what an external consumer's family looks like from the identity
owner's side: a name and one or more importable modules. Using this module keeps
the perturbation honest -- the bytes really are somewhere else -- without adding
a fixture package for one function.
"""


def _second_family_value(source: Any, specification: Any) -> Any:
    """A deterministic constant. Its numbers are irrelevant; its bytes are not."""

    del specification
    import pandas as pd

    return pd.Series([1.0] * len(source), index=source.index)


def _second_family_kernel() -> RegisteredFeatureKernel:
    return RegisteredFeatureKernel(
        implementation_id="factor.case-study.second-family.v1",
        method_family=_SECOND_FAMILY,
        method_family_owners=_SECOND_FAMILY_OWNERS,
        required_fields=("close_split_adjusted",),
        implementation_hash="c" * 64,
        compute=_second_family_value,
    )


def _registry_with_second_family() -> FeatureKernelRegistry:
    """The installed registry plus one more family, composed the way a consumer would."""

    installed = default_extension_kernel_registry()
    return FeatureKernelRegistry(
        (
            installed.resolve(MEAN_ADJUSTED_RETURN_ID),
            installed.resolve(OVERNIGHT_RETURN_ID),
            _second_family_kernel(),
        )
    )


def test_installing_a_second_family_leaves_the_first_bit_identical() -> None:
    """requirement: adding a method does not invalidate an unrelated method.

    This is the defect the previous split left behind. The family closures
    contained the module that composes the installed registry, so registering
    family B edited that module and rotated family A's implementation identity --
    and A's methodology identity with it, since methodology folds implementation.
    Nothing about A had changed.

    The installed set is still an identity and still moves; it is just not A's.
    A consumer asking "is this the same build" reads
    ``installed_capability_hash``, and a consumer asking "is this the same code
    behind this factor" reads the factor's own identity.
    """

    bundle = desktop_core_feature_bundle()
    a_only = default_extension_kernel_registry()
    a_plus_b = _registry_with_second_family()
    recipe = overnight_return_factor_spec()

    base_catalog_hash = FeatureCatalog.load().binding.catalog_hash
    implementation_a = a_only.implementation_hash(recipe, core_bundle=bundle)
    implementation_ab = a_plus_b.implementation_hash(recipe, core_bundle=bundle)
    assert implementation_a == implementation_ab
    assert factor_methodology_hash(recipe, implementation_hash=implementation_a) == (
        factor_methodology_hash(recipe, implementation_hash=implementation_ab)
    )

    # And installation is not invisible: it moves the identity that is about the
    # installed set, and only that one.
    assert a_only.installed_capability_hash != a_plus_b.installed_capability_hash
    assert FeatureCatalog.load().binding.catalog_hash == base_catalog_hash
    assert len(a_plus_b.installed_capability_hash) == 64
    # Registration order is not identity: the same kernels compose to one build.
    reordered = FeatureKernelRegistry(
        (
            _second_family_kernel(),
            a_only.resolve(OVERNIGHT_RETURN_ID),
            a_only.resolve(MEAN_ADJUSTED_RETURN_ID),
        )
    )
    assert reordered.installed_capability_hash == a_plus_b.installed_capability_hash


def test_installed_capability_identity_binds_measured_family_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """regression: installed code changes even when its declaration does not."""

    installed = default_extension_kernel_registry()
    before = installed.installed_capability_hash
    monkeypatch.setattr(
        registry_module,
        "method_family_content_hash",
        lambda _family, _owners: "f" * 64,
    )
    assert installed.installed_capability_hash != before


def test_previous_close_excursion_formulas_are_named_for_the_arithmetic_they_run() -> None:
    """requirement: gap-session excursion quantities are not classic OHLC estimators."""

    source = pd.DataFrame(
        {
            "session_date": (date(2026, 1, 2), date(2026, 1, 5)),
            "listing_id": ("A", "A"),
            "open_split_adjusted": (100.0, 120.0),
            "high_split_adjusted": (101.0, 125.0),
            "low_split_adjusted": (99.0, 110.0),
            "close_split_adjusted": (100.0, 115.0),
            "close_raw": (100.0, 115.0),
            "volume_raw": (10.0, 12.0),
        }
    )
    specs = {value.factor_id: value for value in session_observation_factor_specs()}
    assert tuple(sorted(specs)) == SESSION_OBSERVATION_FACTOR_IDS
    registry = default_extension_kernel_registry()
    receipts = {
        item.factor_id: item
        for item in admit_factor_development_capabilities()
        if item.factor_id in specs
    }
    assert set(receipts) == set(specs)
    assert all(item.disposition == "ADMITTED" for item in receipts.values())
    for factor_id, spec in specs.items():
        assert spec.formula_ref in SESSION_OBSERVATION_KERNELS
        assert registry.resolve(spec.formula_ref).implementation_id == spec.formula_ref
        assert observation_clock_for(spec).factor_id == factor_id
    high_extension = math.log(125.0 / 100.0)
    low_extension = math.log(110.0 / 100.0)
    intraday = math.log(115.0 / 120.0)
    span = abs(high_extension) + abs(low_extension)
    expected = {
        "previous_close_excursion_scaled_span_square": span**2 / (4.0 * math.log(2.0)),
        "previous_close_excursion_intraday_cross": high_extension * (high_extension + intraday)
        + low_extension * (low_extension + intraday),
        "previous_close_excursion_intraday_adjusted_square": span**2 / 2.0
        - (2.0 * math.log(2.0) - 1.0) * intraday**2,
    }
    observed = {
        factor_id: float(registry.compute(source, specs[factor_id]).iloc[-1])
        for factor_id in expected
    }
    assert observed == pytest.approx(expected, rel=1e-14, abs=1e-14)

    high_low = math.log(125.0 / 110.0)
    parkinson = high_low**2 / (4.0 * math.log(2.0))
    garman_klass = 0.5 * high_low**2 - (2.0 * math.log(2.0) - 1.0) * intraday**2
    rogers_satchell = math.log(125.0 / 115.0) * math.log(125.0 / 120.0) + math.log(
        110.0 / 115.0
    ) * math.log(110.0 / 120.0)
    assert not np.isclose(observed["previous_close_excursion_scaled_span_square"], parkinson)
    assert not np.isclose(
        observed["previous_close_excursion_intraday_adjusted_square"], garman_klass
    )
    assert not np.isclose(observed["previous_close_excursion_intraday_cross"], rogers_satchell)
    dollar_spec = session_liquidity_factor_specs()[0]
    dollar_receipt = next(
        item
        for item in admit_factor_development_capabilities()
        if item.factor_id == dollar_spec.factor_id
    )
    assert dollar_receipt.disposition == "ADMITTED"
    assert dollar_spec.formula_ref == SESSION_DOLLAR_VOLUME_ID
    assert observation_clock_for(dollar_spec).factor_id == "session_dollar_volume"
    expected_dollar_volume = raw_session_dollar_volume(
        np.asarray((100.0, 115.0), dtype=np.float64),
        np.asarray((10.0, 12.0), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        registry.compute(source, dollar_spec).to_numpy(dtype=float),
        expected_dollar_volume,
    )
    assert dollar_spec.return_convention == "raw-dollar-volume"
    for factor_id, _expected_value in expected.items():
        spec = specs[factor_id]
        declaration = SESSION_OBSERVATION_DECLARATIONS[
            SESSION_OBSERVATION_IMPLEMENTATION_IDS[factor_id]
        ]
        assert spec.formula in declaration["algorithm"] or factor_id.endswith("cross")
        assert spec.return_convention == "split-adjusted-previous-close-session-excursion"
        assert all("doi.org" not in source for source in spec.literature_sources)
