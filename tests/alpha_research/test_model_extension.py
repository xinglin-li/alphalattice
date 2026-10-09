"""An agent's model, from its declaration to a passing contract (EX, the model point)."""

from __future__ import annotations

import importlib
import json
import re
import shutil
import sys
import types
from datetime import UTC, datetime
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

import pytest

from alphalattice.capabilities.alpha_modeling.extension import EXTENSION_PACKAGE
from alphalattice.capabilities.alpha_modeling.model_contract import STATE_CONTRACT, check_model
from alphalattice.capabilities.alpha_modeling.model_scaffold import scaffold_model
from alphalattice.control.product_host.research_authoring.model_extensions import (
    ModelExtensionRegistry,
    ModelExtensions,
    ModelSandboxRecord,
    activated_models,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL = "numpy_ridge_probe"
DECLARATION = f"""model_id: {MODEL}
family: linear
fit_protocol: DIRECT_FIT
parameters:
  ridge: float
recipe:
  ridge: 1.0
search:
  axes:
    - {{name: ridge, kind: float, low: 0.01, high: 100.0, log: true, default: 1.0}}
"""
FIT_AND_PREDICT = """
    def fit(self, *, recipe, inputs, fit_plan=None):  # type: ignore[no-untyped-def]
        import numpy as np

        from alphalattice.capabilities.alpha_modeling.contracts import (
            AlphaModelFitResult,
            AlphaModelStateProjection,
        )

        ridge = float(self.validate_recipe(recipe)["ridge"])
        x, y = inputs.features, inputs.targets
        coefficients = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
        content = AlphaEstimatorContent.create(
            adapter_id=self.adapter_id,
            content_format_id=self.content_format_id,
            ordered_feature_ids=inputs.ordered_feature_ids,
            payload={"coefficient_hex": [float(v).hex() for v in coefficients]},
        )
        state = AlphaModelStateProjection.create(
            adapter_id=self.adapter_id,
            model_family_id="ridge",
            state_kind="LINEAR",
            state_schema_id=self.content_format_id,
            payload={
                "coefficient_hex": tuple(float(v).hex() for v in coefficients),
                "intercept_hex": (0.0).hex(),
                "coefficient_l2_norm": float(np.linalg.norm(coefficients)),
                "coefficient_max_abs": float(np.max(np.abs(coefficients))),
                "nonzero_support_count": int(np.count_nonzero(coefficients)),
                "model_text_hash": None,
                "best_iteration": None,
                "feature_gain_hex": (),
                "top_feature_gain_share": None,
            },
        )
        mse = float(np.mean(np.square(x @ coefficients - y)))
        return AlphaModelFitResult(
            estimator_content=content, state_projection=state, training_mse=mse,
            iteration_count=None,
        )

    def predict(self, *, estimator, inputs):  # type: ignore[no-untyped-def]
        import numpy as np

        from alphalattice.capabilities.alpha_modeling.contracts import AlphaModelPredictionResult

        coefficients = np.array([float.fromhex(v) for v in estimator.payload["coefficient_hex"]])
        values = np.asarray(inputs.features @ coefficients, dtype=np.float64)
        values.setflags(write=False)
        return AlphaModelPredictionResult(predictions=values)
"""


def _checkout(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    root.mkdir()
    shutil.copyfile(ROOT / "uv.lock", root / "uv.lock")
    (tmp_path / "declaration.yaml").write_text(DECLARATION, encoding="utf-8")
    return root


def _load(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the checkout's extensions as the package, as the Host finds and loads a model."""
    folder = str(root / "src/alphalattice/capabilities/alpha_modeling/extensions")
    package = types.ModuleType(EXTENSION_PACKAGE)
    package.__path__ = [folder]
    package.__spec__ = ModuleSpec(EXTENSION_PACKAGE, None, is_package=True)
    package.__spec__.submodule_search_locations = [folder]
    monkeypatch.setitem(sys.modules, EXTENSION_PACKAGE, package)
    monkeypatch.delitem(sys.modules, f"{EXTENSION_PACKAGE}.{MODEL}", raising=False)
    importlib.invalidate_caches()
    importlib.import_module(f"{EXTENSION_PACKAGE}.{MODEL}")


def _written(root: Path) -> None:
    """The agent's step: `fit` and `predict` in place of the scaffold's refusals."""
    module = root / "src/alphalattice/capabilities/alpha_modeling/extensions" / f"{MODEL}.py"
    text = module.read_text(encoding="utf-8")
    start = text.index("    def fit(")
    module.write_text(
        text[:start] + FIT_AND_PREDICT.lstrip("\n") + "\n\n" + text[text.index("ADAPTER = ") :],
        encoding="utf-8",
    )


def test_a_model_is_scaffolded_checked_and_passes_once_it_fits_and_predicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (EX): `model scaffold` writes the adapter, the declaration and the contract
    test where G1's owner says; the contract names each unwritten step; once the agent writes
    `fit` and `predict`, every check passes, and the route, domain and protocol are the
    declaration's."""

    root = _checkout(tmp_path)
    answer = scaffold_model(tmp_path / "declaration.yaml", root=root)
    assert answer == {
        "status": "SCAFFOLDED",
        "model_id": MODEL,
        "files": [
            f"src/alphalattice/capabilities/alpha_modeling/extensions/{MODEL}.py",
            f"src/alphalattice/capabilities/alpha_modeling/extensions/{MODEL}.model.yaml",
            f"tests/alpha_research/extensions/test_{MODEL}.py",
        ],
        "next_action": "WRITE_FIT_AND_PREDICT_THEN_MODEL_CHECK",
    }
    _load(root, monkeypatch)
    unwritten = check_model(MODEL, root=root)
    assert unwritten["status"] == "FAILED"
    assert {row["check"]: row["code"] for row in unwritten["findings"]} == {
        "route": None,
        "search_axes": None,
        "fit_protocol": None,
        "determinism": f"model_extension.fit_not_written:{MODEL}",
        "prediction_rows": f"model_extension.fit_not_written:{MODEL}",
        "state": f"model_extension.fit_not_written:{MODEL}",
        "imports": None,
    }

    _written(root)
    _load(root, monkeypatch)
    written = check_model(MODEL, root=root)
    assert written["status"] == "PASSED", written["findings"]
    assert len(written["contract_receipt_hash"]) == 64


def test_an_incomplete_state_names_each_field_it_lacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement: a LINEAR state without its summaries is refused by a code that
    names each absent field, and the finding gives each one's type and shape, so an author reads
    no source to find them."""

    root = _checkout(tmp_path)
    scaffold_model(tmp_path / "declaration.yaml", root=root)
    _written(root)
    module = root / "src/alphalattice/capabilities/alpha_modeling/extensions" / f"{MODEL}.py"
    text = module.read_text(encoding="utf-8")
    for line in (
        '                "coefficient_l2_norm": float(np.linalg.norm(coefficients)),\n',
        '                "coefficient_max_abs": float(np.max(np.abs(coefficients))),\n',
        '                "nonzero_support_count": int(np.count_nonzero(coefficients)),\n',
    ):
        assert text.count(line) == 1
        text = text.replace(line, "")
    module.write_text(text, encoding="utf-8")
    _load(root, monkeypatch)
    answer = check_model(MODEL, root=root)
    (state,) = [row for row in answer["findings"] if row["check"] == "state"]
    missing = ("coefficient_l2_norm", "coefficient_max_abs", "nonzero_support_count")
    assert state["code"] == "model_contract.linear_state_incomplete:" + ",".join(missing)
    assert state["expected"] == {name: STATE_CONTRACT["LINEAR"][name] for name in missing}


def test_a_scaffold_refuses_a_new_dependency_and_an_existing_model(
    tmp_path: Path,
) -> None:
    """requirement (EX, the user): a library outside the locked environment is a new
    dependency a person approves first, so the scaffold writes nothing; a model already there
    is not written over."""

    root = _checkout(tmp_path)
    declaration = tmp_path / "declaration.yaml"
    declaration.write_text(DECLARATION + "libraries: [not-a-locked-library]\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match=re.escape("model_extension.library_outside_lock:not-a-locked-library")
    ):
        scaffold_model(declaration, root=root)
    assert not (root / "src").exists()
    declaration.write_text(DECLARATION.replace(MODEL, "regularized_linear"), encoding="utf-8")
    with pytest.raises(
        ValueError, match=re.escape("model_extension.entry_exists:regularized_linear")
    ):
        scaffold_model(declaration, root=root)


def test_the_host_runs_a_models_contract_once_per_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement: the Host keeps each model's contract answer by what it depends on, so
    a Models read runs no contract a second time; a model whose source moves is checked again."""

    root = _checkout(tmp_path)
    scaffold_model(tmp_path / "declaration.yaml", root=root)
    _load(root, monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    contracts: dict[str, dict[str, Any]] = {}
    models = ModelExtensions(
        workspace, lambda: datetime(2026, 9, 30, 12, tzinfo=UTC), contracts=contracts
    )
    first = models.review()
    assert len(contracts) == 3
    assert models.review() == first and len(contracts) == 3
    packets = {value["model_id"]: value for value in first["models"]}
    assert packets[MODEL]["contract"]["status"] == "FAILED"

    _written(root)
    _load(root, monkeypatch)
    again = {value["model_id"]: value for value in models.review()["models"]}
    assert again[MODEL]["contract"]["status"] == "PASSED"
    assert len(contracts) == 4
    assert again["regularized_linear"]["contract"] == packets["regularized_linear"]["contract"]


def test_model_review_names_one_unavailable_extension_and_keeps_healthy_packets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken extension is a row refusal; it does not hide installed or healthy models."""

    root = _checkout(tmp_path)
    scaffold_model(tmp_path / "declaration.yaml", root=root)
    _load(root, monkeypatch)
    extension_folder = root / "src/alphalattice/capabilities/alpha_modeling/extensions"
    (extension_folder / "broken_model.py").write_text(
        "raise RuntimeError('private import detail')\n", encoding="utf-8"
    )
    (extension_folder / "broken_model.model.yaml").write_text(
        "model_id: broken_model\n", encoding="utf-8"
    )
    importlib.invalidate_caches()

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    packets = ModelExtensions(workspace, lambda: datetime(2026, 9, 30, 12, tzinfo=UTC)).review()

    assert packets["status"] == "AVAILABLE"
    rows = {row["model_id"]: row for row in packets["models"]}
    assert rows["regularized_linear"]["state"] == "INSTALLED"
    assert rows[MODEL]["contract"]["status"] == "FAILED"
    assert rows["broken_model"] == {
        "status": "REFUSED",
        "model_id": "broken_model",
        "failure_code": "model_extension.review_unavailable",
        "detail": (
            "This model's review packet could not be read. Repair its declaration, module, "
            "or required runtime, then run the model check command shown here."
        ),
        "next_requests": {"models": {"operation": "MODEL_EXTENSIONS"}},
        "next_commands": {"check": "alphalattice model check broken_model"},
    }
    assert "private import detail" not in str(rows["broken_model"])
    semantic = ModelExtensions._review_refusal(
        "broken_model", ValueError("model_extension.recipe_route_invalid")
    )
    assert semantic["failure_code"] == "model_extension.recipe_route_invalid"


def test_a_person_activates_a_sandboxed_model_and_the_workspace_catalog_installs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (EX): the review packet shows each model, installed or an agent's; an
    activation needs a passing contract and a passed sandbox trial of the same identity, and
    records it; the workspace's catalog and mandate then install the model after the installed
    ones, and a deactivation removes it."""

    root = _checkout(tmp_path)
    scaffold_model(tmp_path / "declaration.yaml", root=root)
    _written(root)
    _load(root, monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    now = datetime(2026, 9, 30, 12, tzinfo=UTC)
    models = ModelExtensions(workspace, lambda: now)
    packets = {value["model_id"]: value for value in models.review()["models"]}
    assert {name: value["state"] for name, value in packets.items()} == {
        "regularized_linear": "INSTALLED",
        "dynamic_panel_lightgbm": "INSTALLED",
        MODEL: "NOT_ACTIVE",
    }
    assert packets[MODEL]["identity"]["adds"] == "CAPABILITY"
    assert packets[MODEL]["next_requests"] == {}, "no sandbox trial yet"
    with pytest.raises(ValueError, match=re.escape(f"model_extension.sandbox_required:{MODEL}")):
        models.activate(MODEL)

    contract = packets[MODEL]["contract"]
    trial = ModelSandboxRecord(
        model_id=MODEL,
        declaration_hash=contract["declaration_hash"],
        numerical_binding_hash=contract["numerical_binding_hash"],
        contract_receipt_hash=contract["contract_receipt_hash"],
        study_task_id="sandbox-study",
        study_model_id=MODEL,
        study_seconds=1.5,
        peak_memory_bytes=2**20,
        u0_reads=12,
        u0_changed=0,
        recorded_at=now,
    )
    unproved = trial.model_copy(update={"study_model_id": None})
    another = trial.model_copy(update={"study_model_id": "regularized_linear"})
    assert trial.passed and not unproved.passed and not another.passed, "V414"
    ModelExtensionRegistry(sandboxes=(unproved, another)).write(workspace)
    with pytest.raises(ValueError, match=re.escape(f"model_extension.sandbox_required:{MODEL}")):
        models.activate(MODEL)
    ModelExtensionRegistry(sandboxes=(trial,)).write(workspace)
    packet = {value["model_id"]: value for value in models.review()["models"]}[MODEL]
    assert packet["next_requests"] == {
        "activate": {"operation": "MODEL_ACTIVATE", "model_id": MODEL}
    }
    assert models.activate(MODEL)["status"] == "ACTIVATED"
    assert activated_models(workspace) == (MODEL,)

    from alphalattice.capabilities.alpha_modeling.catalog import (
        build_installed_alpha_model_catalog,
    )
    from alphalattice.investment.alpha_research.experiments.mandate import (
        build_installed_alpha_model_capability_mandate,
    )

    catalog = build_installed_alpha_model_catalog(activated_models(workspace))
    assert catalog.adapter_ids == ("regularized_linear", "dynamic_panel_lightgbm", MODEL)
    mandate = build_installed_alpha_model_capability_mandate(catalog=catalog)
    assert mandate.ordered_search_domains[-1].adapter_id == MODEL
    assert models.deactivate(MODEL)["status"] == "DEACTIVATED"
    assert activated_models(workspace) == ()
    with pytest.raises(ValueError, match=re.escape("model_extension.installed:regularized_linear")):
        models.activate("regularized_linear")


def test_the_installed_mandate_is_the_same_value_from_one_source_of_domains() -> None:
    """regression (EX part 2b): each adapter's domain now has one source, and the installed
    capability mandate keeps its value: the same catalog binding and the same two domains."""

    from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
        build_dynamic_panel_lightgbm_search_domain,
    )
    from alphalattice.capabilities.alpha_modeling.adapters.regularized_linear import (
        build_regularized_linear_search_domain,
    )
    from alphalattice.capabilities.alpha_modeling.catalog import (
        build_installed_alpha_model_catalog,
    )
    from alphalattice.investment.alpha_research.experiments.mandate import (
        AlphaModelCapabilityMandate,
        build_installed_alpha_model_capability_mandate,
    )

    expected = AlphaModelCapabilityMandate.create(
        catalog_binding=build_installed_alpha_model_catalog().binding,
        ordered_search_domains=(
            build_regularized_linear_search_domain(),
            build_dynamic_panel_lightgbm_search_domain(),
        ),
    )
    assert build_installed_alpha_model_capability_mandate() == expected


def test_a_sandbox_tries_an_agents_model_on_a_workspace_at_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (EX): the sandbox copies the workspace at rest, so it refuses while a Host
    holds the workspace's writer lease, naming the way on; an installed model has no trial."""

    from alphalattice.control.product_host.composition.model_sandbox import run_sandbox
    from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease

    root = _checkout(tmp_path)
    scaffold_model(tmp_path / "declaration.yaml", root=root)
    _written(root)
    _load(root, monkeypatch)
    workspace = tmp_path / "workspace"
    lease = WorkspaceWriterLease.acquire(workspace)
    try:
        with pytest.raises(ValueError, match=re.escape("model_sandbox.host_serving")):
            run_sandbox(workspace, MODEL, root=tmp_path / "sandboxes")
    finally:
        lease.close()
    assert not (tmp_path / "sandboxes").exists(), "nothing is copied"
    with pytest.raises(ValueError, match=re.escape("model_extension.installed:regularized_linear")):
        run_sandbox(workspace, "regularized_linear")


def test_a_model_begins_from_the_declaration_its_contract_writes(
    tmp_path: Path, capsys: Any
) -> None:
    """requirement (an outside review of the Skill and cards): a model's first step was
    to guess a six-field declaration, refused as `model_declaration.invalid` alone. `model
    scaffold --save-declaration` writes one to edit, which the scaffold takes as it stands; a
    refused declaration names its fields; the sandbox's refusal names the flag it has."""

    from alphalattice.capabilities.alpha_modeling.declaration import read_declaration
    from alphalattice.interface.local_application import cli
    from alphalattice.interface.local_application.cli_contract import client_refusal

    written = tmp_path / "model.yaml"
    workspace = ["--workspace", str(tmp_path)]
    assert (
        cli.main(
            [*workspace, "model", "scaffold", "--save-declaration", str(written)],
            serve=lambda _: 99,
        )
        == 0
    )
    answer = json.loads(capsys.readouterr().out)
    assert answer["data"]["status"] == "DECLARATION_WRITTEN", answer
    assert read_declaration(written).model_id == "my_model"
    scaffolded = scaffold_model(written, root=_checkout(tmp_path))
    assert scaffolded["status"] == "SCAFFOLDED", scaffolded
    assert (
        cli.main(
            [*workspace, "model", "scaffold", "--save-declaration", str(written)],
            serve=lambda _: 99,
        )
        != 0
    )
    capsys.readouterr()  # an existing file is never written over

    broken = tmp_path / "broken.yaml"
    broken.write_text("model_id: my_model\nfamily: linear\n", encoding="utf-8")
    assert (
        cli.main([*workspace, "model", "scaffold", "--file", str(broken)], serve=lambda _: 99) == 1
    )
    refused = json.loads(capsys.readouterr().out)
    assert refused["failure_code"] == (
        "model_declaration.invalid:fit_protocol,parameters,recipe,search"
    ), refused
    assert "--save-declaration" in refused["detail"]
    sandbox = client_refusal("model_sandbox.no_alpha_study")
    assert "--file" in sandbox.detail and "--study" not in sandbox.detail


def test_a_sandbox_trial_runs_the_model_it_names_whatever_study_declares_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """regression (an outside review at 3fa785fd): `model sandbox <model> --file <study>`
    ran the file's own model and recorded the trial under <model>. The sandbox puts the model in
    the declared study's place, refuses a study that names none, and records a trial only when
    the study's readback names the model."""

    from alphalattice.capabilities.alpha_modeling.catalog import (
        build_installed_alpha_model_catalog,
    )
    from alphalattice.capabilities.alpha_modeling.extension import extension_adapter
    from alphalattice.control.product_host.composition import model_sandbox
    from alphalattice.control.product_host.composition.local_web_session import (
        LocalPortfolioWebSession,
    )
    from alphalattice.investment.alpha_research.experiments.mandate import (
        build_installed_alpha_model_capability_mandate,
    )

    root = _checkout(tmp_path)
    scaffold_model(tmp_path / "declaration.yaml", root=root)
    _written(root)
    _load(root, monkeypatch)
    copy = tmp_path / "copy"
    copy.mkdir()
    ModelExtensionRegistry(trial=(MODEL,)).write(copy)
    study = tmp_path / "old-alpha.yaml"
    study.write_text(
        "experiment: {window: kept}\n"
        "alpha: {model_capability_handle: another-model, model_parameters: {alpha: 1.0}}\n",
        encoding="utf-8",
    )
    mandate = build_installed_alpha_model_capability_mandate(
        catalog=build_installed_alpha_model_catalog(activated_models(copy))
    )
    ids = [value.adapter_id for value in mandate.ordered_search_domains]
    document = model_sandbox._declared_study(None, copy, MODEL, study)
    assert document == {
        "experiment": {"window": "kept"},
        "alpha": {
            "model_capability_handle": mandate.capability_handle(ids.index(MODEL)),
            "model_parameters": dict(extension_adapter(MODEL).declaration.recipe),
        },
    }
    lifecycle = tmp_path / "lifecycle.yaml"
    lifecycle.write_text("alpha: {component_recipe_id: recipe}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape("model_sandbox.study_names_no_model")):
        model_sandbox._declared_study(None, copy, MODEL, lifecycle)

    task_id = "00000000-0000-4000-8000-000000000001"
    ran = {"task_id": task_id, "model_adapter_id": "regularized_linear"}
    answers: dict[str, dict[str, Any]] = {
        "EXPERIMENT_PLAN": {"status": "PLANNED", "plan_hash": "a" * 64},
        "EXPERIMENT_RUN": {"task_id": task_id},
        "STATUS": {"lifecycle": "SUCCEEDED"},
        "EXPERIMENT_READBACK": {"status": "EXPERIMENT_PUBLISHED"},
        "EXPERIMENTS": {"experiments": [ran]},
    }

    class Operations:
        """The copy's Host, answering each operation the trial sends."""

        def execute(self, request: Any, caller: str) -> dict[str, Any]:
            return answers[request.operation]

    class Session:
        """A session on the copy."""

        operations = Operations()

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    monkeypatch.setattr(
        LocalPortfolioWebSession, "from_workspace", classmethod(lambda _cls, _path: Session())
    )
    refused = "model_sandbox.study_ran_another_model:regularized_linear"
    with pytest.raises(ValueError, match=re.escape(refused)):
        model_sandbox._run_study(copy, MODEL, study)
    ran["model_adapter_id"] = MODEL
    assert model_sandbox._run_study(copy, MODEL, study)[::3] == (task_id, MODEL)
