"""What every Alpha model adapter passes, the installed ones included (EX, its design's section 2).

`model check` runs it on a declared model; the tests run it on the installed catalog. Each check
passes or names what it found, as a code:

`route`: the declaration, the adapter and the domain name one model, and the adapter declares
  its numerical binding;
`search_axes`: every point the axes' probes state is admitted by the adapter's own refusals;
`fit_protocol`: the reference recipe fits by the protocol the declaration states;
`determinism`: two fits under the recipe's seed, at one thread (the fit's default; the
  operator's threads are W10's canary's to prove), give one estimator and one prediction;
`prediction_rows`: predictions are finite float64, one per row, read only, and a row's
  prediction reads that row alone;
`state`: the fit's state is one a development study seals: a LINEAR state carries a
  coefficient per feature with its diagnostics, a TREE state its trees' hash, iterations and a
  gain per feature, and a state of the model's own kind its payload;
`imports`: every library the adapter's module imports is held by the lock. A library outside
  it is a new dependency, which a person approves first (the user, 2026-09-30).

The environment a fit ran in is recorded beside it (the fit provenance), never compared here.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import re
import sys
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root

from .contracts import (
    AlphaModelAdapter,
    AlphaModelFitResult,
    AlphaModelRecipeEnvelope,
    AlphaModelSearchDomainEnvelope,
    BoundAlphaModelFitInput,
    BoundAlphaPredictionInput,
    BoundAlphaTrainingInput,
    FloatArray,
)
from .declaration import AlphaModelDeclaration
from .extension import DeclaredModelAdapter, extension_adapter

_PACKAGE = "alphalattice.capabilities.alpha_modeling"
STATE_CONTRACT: Mapping[str, Mapping[str, str]] = {
    "LINEAR": {
        "coefficient_hex": "one float64 as float.hex() per feature, in the input's order",
        "intercept_hex": "one float64 as float.hex()",
        "coefficient_l2_norm": "float >= 0: the coefficients' L2 norm",
        "coefficient_max_abs": "float >= 0: the largest absolute coefficient",
        "nonzero_support_count": "int >= 0: the coefficients not zero",
    },
    "TREE": {
        "model_text_hash": "the fitted model's text, sha256 hex",
        "best_iteration": "int >= 0",
        "feature_gain_hex": "one float64 as float.hex() per feature, in the input's order",
        "top_feature_gain_share": "float in [0, 1]: the largest feature's share of the gain",
    },
}
"""The payload a LINEAR or TREE state projection carries: each field and its type and shape,
the development study's own rule (`AlphaDevelopmentEstimatorState`); a model of another kind
seals a generic state."""
_TRAINING_ROWS, _PREDICTION_ROWS, _FEATURES = 400, 100, 6


@dataclass(frozen=True, slots=True)
class ContractFinding:
    """One check: None when it passed, else the code naming what it found."""

    check: str
    code: str | None


def locked_distributions(lock: Path) -> frozenset[str]:
    """The distributions a uv lock holds, by normalized name.

    Args:
        lock: The `uv.lock` file.

    Returns:
        The names.
    """
    packages = tomllib.loads(lock.read_text(encoding="utf-8")).get("package", [])
    return frozenset(_normal(str(value["name"])) for value in packages)


def _normal(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _rows(seed: int) -> tuple[BoundAlphaTrainingInput, BoundAlphaPredictionInput]:
    """Synthetic training and prediction rows: a noisy linear signal on standard normals."""
    generator = np.random.default_rng(seed)
    features = generator.standard_normal((_TRAINING_ROWS + _PREDICTION_ROWS, _FEATURES))
    targets = features @ np.linspace(0.5, -0.5, _FEATURES) + 0.1 * generator.standard_normal(
        len(features)
    )
    training, prediction = features[:_TRAINING_ROWS], features[_TRAINING_ROWS:]
    targets = targets[:_TRAINING_ROWS]
    for array in (training, prediction, targets):
        array.setflags(write=False)
    ids = tuple(f"feature_{n}" for n in range(_FEATURES))
    binding = canonical_hash({"model_contract": "synthetic_rows", "seed": seed})
    return (
        BoundAlphaTrainingInput(binding, ids, training, targets),
        BoundAlphaPredictionInput(binding, ids, prediction),
    )


def _code(error: Exception, fallback: str) -> str:
    text = str(error)
    return text if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:=,<>+\- ]{0,160}", text) else fallback


def _imports(module: str, seen: set[str]) -> set[str]:
    """The top-level libraries a module imports, through its package-relative imports."""
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None or module in seen:
        return set()
    seen.add(module)
    found: set[str] = set()
    package = module.rpartition(".")[0]
    for node in ast.walk(ast.parse(Path(spec.origin).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.partition(".")[0])
                if node.module.startswith(_PACKAGE):
                    found |= _imports(node.module, seen)
            elif node.level:
                name = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                if name.startswith(_PACKAGE):
                    found |= _imports(name, seen)
    return found


def import_findings(adapter: AlphaModelAdapter, locked: frozenset[str]) -> tuple[str, ...]:
    """The libraries an adapter imports that the lock does not hold.

    Args:
        adapter: The adapter.
        locked: The lock's distributions (`locked_distributions`).

    Returns:
        One code per library: `model_contract.import_outside_lock:<name>`, or
        `model_contract.import_not_installed:<name>` for one no distribution provides.
    """
    names = _imports(type(adapter).__module__, set())
    distributions = importlib.metadata.packages_distributions()
    found = []
    for name in sorted(names - set(sys.stdlib_module_names) - {"alphalattice", "__future__"}):
        provided = {_normal(value) for value in distributions.get(name, ())}
        if not provided:
            found.append(f"model_contract.import_not_installed:{name}")
        elif not provided & locked:
            found.append(f"model_contract.import_outside_lock:{name}")
    return tuple(found)


def contract_findings(
    adapter: AlphaModelAdapter,
    declaration: AlphaModelDeclaration,
    domain: AlphaModelSearchDomainEnvelope,
    *,
    locked: frozenset[str],
) -> tuple[ContractFinding, ...]:
    """Run a model's contract.

    Args:
        adapter: The model's adapter.
        declaration: Its declaration.
        domain: The search domain its mandate admits.
        locked: The lock's distributions.

    Returns:
        One finding per check, in the order the module's docstring lists them.
    """

    def recipe(parameters: dict[str, object]) -> AlphaModelRecipeEnvelope:
        return AlphaModelRecipeEnvelope.create(
            adapter_id=adapter.adapter_id,
            recipe_schema_id=adapter.recipe_schema_id,
            parameters=parameters,
        )

    def run(check: str, body: Callable[[], str | None]) -> ContractFinding:
        try:
            return ContractFinding(check, body())
        except (ValueError, TypeError, ArithmeticError) as error:
            return ContractFinding(check, _code(error, f"model_contract.{check}_raised"))

    reference = recipe(dict(declaration.recipe))

    def route() -> str | None:
        if not declaration.model_id == adapter.adapter_id == domain.adapter_id:
            return "model_contract.route_mismatch"
        if adapter.describe_numerical_binding().adapter_id != adapter.adapter_id:
            return "model_contract.numerical_binding_route_invalid"
        return None

    def search_axes() -> str | None:
        for index, point in enumerate(declaration.search.probes()):
            try:
                adapter.validate_recipe_for_domain(
                    recipe=recipe(declaration.recipe_at(point)), domain=domain
                )
            except (ValueError, TypeError) as error:
                return f"model_contract.axis_point_refused:{index}:{_code(error, 'refused')}"
        return None

    def fit_protocol() -> str | None:
        stated = adapter.fit_protocols(reference)[0]
        return None if stated == declaration.fit_protocol else "model_contract.fit_protocol_differs"

    def predictions() -> tuple[AlphaModelFitResult, FloatArray, FloatArray]:
        training, prediction = _rows(0)
        # The plan the runtime hands a direct fit: the training rows' binding and axis.
        plan = BoundAlphaModelFitInput(
            fit_plan_hash=canonical_hash(
                {"model_contract": "direct_fit", "parent": training.training_binding_hash}
            ),
            protocol_id="DIRECT_FIT",
            parent_training_binding_hash=training.training_binding_hash,
            ordered_feature_ids=training.ordered_feature_ids,
        )
        fitted = adapter.fit(recipe=reference, inputs=training, fit_plan=plan)
        values = adapter.predict(estimator=fitted.estimator_content, inputs=prediction)
        half = BoundAlphaPredictionInput(
            prediction.training_binding_hash,
            prediction.ordered_feature_ids,
            prediction.features[: _PREDICTION_ROWS // 2],
        )
        part = adapter.predict(estimator=fitted.estimator_content, inputs=half)
        return fitted, values.predictions, part.predictions

    def determinism() -> str | None:
        if declaration.fit_protocol != "DIRECT_FIT":
            return "model_contract.nested_fit_not_probed"
        first, second = predictions(), predictions()
        if first[0].estimator_content.content_hash != second[0].estimator_content.content_hash:
            return "model_contract.estimator_not_deterministic"
        if first[1].tobytes() != second[1].tobytes():
            return "model_contract.predictions_not_deterministic"
        return None

    def prediction_rows() -> str | None:
        if declaration.fit_protocol != "DIRECT_FIT":
            return "model_contract.nested_fit_not_probed"
        _content, values, part = predictions()
        if (
            values.shape != (_PREDICTION_ROWS,)
            or values.dtype != np.float64
            or values.flags.writeable
            or not np.isfinite(values).all()
        ):
            return "model_contract.predictions_malformed"
        if part.tobytes() != values[: len(part)].tobytes():
            return "model_contract.prediction_reads_other_rows"
        return None

    def state() -> str | None:
        if declaration.fit_protocol != "DIRECT_FIT":
            return "model_contract.nested_fit_not_probed"
        projection = predictions()[0].state_projection
        if projection.adapter_id != adapter.adapter_id:
            return "model_contract.state_route_mismatch"
        payload = projection.payload
        # The development study's own rule (`AlphaDevelopmentEstimatorState`), checked
        # before a study meets it: an incomplete state stops a study as an interruption. The
        # refusal names each field absent or of the wrong length.
        fields = STATE_CONTRACT.get(projection.state_kind)
        if fields is None:
            return None
        per_feature = "coefficient_hex" if projection.state_kind == "LINEAR" else "feature_gain_hex"
        missing = [name for name in fields if name not in payload]
        if per_feature in payload and len(payload[per_feature]) != _FEATURES:
            missing.append(per_feature)
        if missing:
            kind = projection.state_kind.lower()
            return f"model_contract.{kind}_state_incomplete:{','.join(missing)}"
        return None

    def imports() -> str | None:
        found = import_findings(adapter, locked)
        return found[0] if found else None

    return tuple(
        run(check, body)
        for check, body in (
            ("route", route),
            ("search_axes", search_axes),
            ("fit_protocol", fit_protocol),
            ("determinism", determinism),
            ("prediction_rows", prediction_rows),
            ("state", state),
            ("imports", imports),
        )
    )


def _expected(code: str | None) -> dict[str, dict[str, str]]:
    """What an incomplete state lacks: each field with its type and shape."""
    if code is None or "_state_incomplete:" not in code:
        return {}
    kind = code.removeprefix("model_contract.").split("_", 1)[0].upper()
    names = code.split(":", 1)[1].split(",")
    return {"expected": {name: STATE_CONTRACT[kind][name] for name in names}}


def _subject(
    model_id: str,
) -> tuple[AlphaModelAdapter, AlphaModelDeclaration, AlphaModelSearchDomainEnvelope]:
    """A model's adapter, its declaration and the search domain its mandate admits."""
    from .catalog import (
        build_installed_alpha_model_catalog,
        build_installed_alpha_model_search_domains,
    )
    from .declaration import installed_declaration

    installed = build_installed_alpha_model_catalog()
    if model_id in installed.adapter_ids:
        # The installed models pass the same contract (EX).
        return (
            installed.adapter(model_id),
            installed_declaration(model_id),
            build_installed_alpha_model_search_domains(installed)[
                installed.adapter_ids.index(model_id)
            ],
        )
    adapter = extension_adapter(model_id)
    declared = cast(DeclaredModelAdapter, adapter)
    return adapter, declared.declaration, declared.declared_search_domain()


def contract_key(model_id: str, *, root: Path | None = None) -> str:
    """What a model's contract answer depends on, by content: one key per identity.

    The declaration, the numerical binding, the search domain, the source bytes of the adapter's
    module and the lock; reading them costs milliseconds, where the contract fits the model.

    Args:
        model_id: The model.
        root: The checkout; the one this module is in when omitted.

    Returns:
        The key's hash.
    """
    checkout = root or resolve_playpen_root(Path(__file__))
    adapter, declaration, domain = _subject(model_id)
    module = sys.modules[type(adapter).__module__]
    source = Path(str(module.__file__)).read_bytes() if module.__file__ else b""
    key: str = canonical_hash(
        {
            "model_id": model_id,
            "declaration": declaration.model_dump(mode="json", by_alias=True),
            "numerical_binding_hash": adapter.describe_numerical_binding().numerical_binding_hash,
            "domain": domain.model_dump(mode="json"),
            "module_source_sha256": hashlib.sha256(source).hexdigest(),
            "lock_sha256": hashlib.sha256((checkout / "uv.lock").read_bytes()).hexdigest(),
        }
    )
    return key


def check_model(model_id: str, *, root: Path | None = None) -> dict[str, Any]:
    """`model check`: run an extension's contract.

    Args:
        model_id: The model.
        root: The checkout; the one this module is in when omitted.

    Returns:
        `PASSED` or `FAILED`, each check's finding, and the hashes an activation records.
    """
    checkout = root or resolve_playpen_root(Path(__file__))
    adapter, declaration, domain = _subject(model_id)
    findings = contract_findings(
        adapter, declaration, domain, locked=locked_distributions(checkout / "uv.lock")
    )
    declaration_hash = canonical_hash(declaration.model_dump(mode="json", by_alias=True))
    rows = [
        {"check": value.check, "code": value.code, **_expected(value.code)} for value in findings
    ]
    return {
        "status": "PASSED" if all(value.code is None for value in findings) else "FAILED",
        "model_id": model_id,
        "findings": rows,
        "declaration_hash": declaration_hash,
        "numerical_binding_hash": adapter.describe_numerical_binding().numerical_binding_hash,
        "contract_receipt_hash": canonical_hash(
            {"model_id": model_id, "declaration_hash": declaration_hash, "findings": rows}
        ),
    }


__all__ = [
    "ContractFinding",
    "check_model",
    "contract_findings",
    "contract_key",
    "import_findings",
    "locked_distributions",
]
