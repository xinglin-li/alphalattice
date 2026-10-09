"""The Alpha models a workspace admits beyond the installed ones: their review and activation (EX).

An agent declares, checks and sandboxes a model (`model scaffold`, `model check`, `model
sandbox`); a person reads its review packet and activates it for this workspace's
research, or deactivates it. Activation records who and when, the entry's identity, its
contract receipt and the sandbox study it passed; it never edits a sealed record. The
workspace's registry, `runtime/extensions/alpha-models.json`, is what the Host's Alpha
studies read (`activated_models`), so an activated model is installed in this workspace's
catalog alone.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, MutableMapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from alphalattice.capabilities.alpha_modeling.catalog import build_installed_alpha_model_catalog
from alphalattice.capabilities.alpha_modeling.extension import (
    EXTENSION_PACKAGE,
    extension_adapter,
)
from alphalattice.capabilities.alpha_modeling.model_contract import check_model, contract_key
from alphalattice.capabilities.alpha_modeling.runtime.numerical_environment import (
    resolve_alpha_model_numerical_environment,
)

REGISTRY = Path("runtime") / "extensions" / "alpha-models.json"


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSandboxRecord(_Contract):
    """One sandbox trial of a model's identity: its study and saved-object readback on a copy."""

    model_id: str
    declaration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    study_task_id: str = Field(min_length=1)
    study_model_id: str | None = None
    """The model the sandbox study ran, read back from the study; a record without it
    holds no proof of which model ran and passes no activation."""

    study_seconds: float = Field(ge=0)
    """The sandbox study's run time, its fits included."""
    peak_memory_bytes: int = Field(ge=0)
    u0_reads: int = Field(ge=0)
    u0_changed: int = Field(ge=0)
    recorded_at: datetime

    @property
    def passed(self) -> bool:
        """Whether its study ran this model and the copy read as before."""
        return self.study_model_id == self.model_id and self.u0_changed == 0 and self.u0_reads > 0


class ModelActivation(_Contract):
    """A person's activation of one model identity for this workspace."""

    model_id: str
    declaration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sandbox: ModelSandboxRecord
    activated_by: Literal["HUMAN"] = "HUMAN"
    activated_at: datetime


class ModelExtensionRegistry(_Contract):
    """What a workspace admits: its activations, a sandbox copy's trial, and the trials held."""

    active: tuple[ModelActivation, ...] = ()
    trial: tuple[str, ...] = ()
    """The models a sandbox copy installs for its trial, in that copy alone."""
    sandboxes: tuple[ModelSandboxRecord, ...] = ()

    @classmethod
    def read(cls, workspace: Path) -> Self:
        """The workspace's registry; empty when it holds none.

        Args:
            workspace: The workspace.

        Returns:
            The registry.
        """
        path = workspace / REGISTRY
        if not path.is_file():
            return cls()
        registry: Self = cls.model_validate_json(path.read_text(encoding="utf-8"))
        return registry

    def write(self, workspace: Path) -> None:
        """Replace the workspace's registry atomically.

        Args:
            workspace: The workspace.
        """
        path = workspace / REGISTRY
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_suffix(".json.tmp")
        staged.write_text(
            json.dumps(self.model_dump(mode="json"), indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(staged, path)


def activated_models(workspace: Path) -> tuple[str, ...]:
    """The models a person activated in this workspace, in activation order.

    Args:
        workspace: The Host's workspace.

    Returns:
        Their ids, and a sandbox copy's trial, for
        `build_installed_alpha_model_catalog(extensions=...)`.
    """
    registry = ModelExtensionRegistry.read(workspace)
    return tuple(dict.fromkeys((*(value.model_id for value in registry.active), *registry.trial)))


def _extension_ids() -> tuple[str, ...]:
    import importlib.util

    spec = importlib.util.find_spec(EXTENSION_PACKAGE)
    if spec is None or not spec.submodule_search_locations:
        return ()
    folder = Path(next(iter(spec.submodule_search_locations)))
    return tuple(
        sorted(
            path.stem
            for path in folder.glob("*.py")
            if path.stem != "__init__" and path.with_name(f"{path.stem}.model.yaml").is_file()
        )
    )


class ModelExtensions:
    """The Host's model extensions: the review packets, a person's activation and deactivation."""

    def __init__(
        self,
        workspace: Path,
        clock: Callable[[], datetime],
        *,
        contracts: MutableMapping[str, dict[str, Any]] | None = None,
    ) -> None:
        """Bind a workspace.

        Args:
            workspace: The Host's workspace.
            clock: The Host's clock.
            contracts: The Host's contract answers by `contract_key`, kept for its life; none
                runs each model's contract at each read.
        """
        self.workspace, self.clock = workspace, clock
        self.contracts = contracts

    def _contract(self, model_id: str) -> dict[str, Any]:
        # A model's contract fits it (0.2-0.4 s a model): the Host runs it once per
        # identity, not once per read; `model check` runs it each time.
        if self.contracts is None:
            return check_model(model_id)
        key = contract_key(model_id)
        held = self.contracts.get(key)
        if held is None:
            held = self.contracts[key] = check_model(model_id)
        return held

    def review(self) -> dict[str, Any]:
        """Every installed and extension model with its review packet.

        Returns:
            `AVAILABLE` and one packet per model: its declaration, contract, identity,
            environment, latest sandbox trial of that identity, activation, and the next request.
        """
        registry = ModelExtensionRegistry.read(self.workspace)
        active = {value.model_id: value for value in registry.active}
        installed = build_installed_alpha_model_catalog().adapter_ids
        models = []
        for model_id in (*installed, *_extension_ids()):
            try:
                models.append(self._review_model(model_id, installed, active, registry))
            except Exception as error:
                # The workspace registry and installed catalog above are shared authority. Once
                # they validate, a broken extension file, import, contract, or runtime must not
                # hide the other models' review packets.
                models.append(self._review_refusal(model_id, error))
        return {"status": "AVAILABLE", "models": models}

    def _review_model(
        self,
        model_id: str,
        installed: tuple[str, ...],
        active: dict[str, ModelActivation],
        registry: ModelExtensionRegistry,
    ) -> dict[str, Any]:
        """Build one model's packet after the shared registry and catalog have validated."""
        contract = self._contract(model_id)
        adapter = (
            build_installed_alpha_model_catalog().adapter(model_id)
            if model_id in installed
            else extension_adapter(model_id)
        )
        binding = adapter.describe_numerical_binding()
        sandbox = next(
            (
                value
                for value in reversed(registry.sandboxes)
                if value.model_id == model_id
                and value.numerical_binding_hash == contract["numerical_binding_hash"]
                and value.declaration_hash == contract["declaration_hash"]
            ),
            None,
        )
        activation = active.get(model_id)
        state = (
            "INSTALLED"
            if model_id in installed
            else "ACTIVE"
            if activation is not None
            else "NOT_ACTIVE"
        )
        return {
            "model_id": model_id,
            "state": state,
            "contract": contract,
            "identity": {
                "numerical_binding_hash": binding.numerical_binding_hash,
                # An extension's module is its own closure: it adds this identity and
                # moves none (EX, identity by rule).
                "adds": None if model_id in installed else "CAPABILITY",
                "moves": [],
            },
            "environment": resolve_alpha_model_numerical_environment(binding).model_dump(
                mode="json"
            ),
            "sandbox": None if sandbox is None else sandbox.model_dump(mode="json"),
            "activation": None if activation is None else activation.model_dump(mode="json"),
            "next_requests": self._next(model_id, state, contract, sandbox),
        }

    @staticmethod
    def _review_refusal(model_id: str, error: Exception) -> dict[str, Any]:
        """Name one unreadable model without serving exception text or hiding its peers."""
        code = getattr(error, "code", None)
        if not isinstance(code, str):
            code = str(error)
        if code.startswith("model_extension.not_found:") or not re.fullmatch(
            r"model_extension\.[A-Za-z0-9_.]+(?::[A-Za-z0-9_.,<>=+\- ]{1,160})?",
            code,
        ):
            code = "model_extension.review_unavailable"
        refusal: dict[str, Any] = {
            "status": "REFUSED",
            "model_id": model_id,
            "failure_code": code,
            "detail": (
                "This model's review packet could not be read. Repair its declaration, module, "
                "or required runtime, then run the model check command shown here."
            ),
            "next_requests": {"models": {"operation": "MODEL_EXTENSIONS"}},
        }
        if re.fullmatch(r"[a-z][a-z0-9_]*", model_id):
            refusal["next_commands"] = {"check": f"alphalattice model check {model_id}"}
        return refusal

    @staticmethod
    def _next(
        model_id: str,
        state: str,
        contract: dict[str, Any],
        sandbox: ModelSandboxRecord | None,
    ) -> dict[str, dict[str, str]]:
        if state == "INSTALLED" or contract["status"] != "PASSED":
            return {}
        if state == "ACTIVE":
            return {"deactivate": {"operation": "MODEL_DEACTIVATE", "model_id": model_id}}
        if sandbox is None or not sandbox.passed:
            # The sandbox trial is the client's, not an operation.
            return {}
        return {"activate": {"operation": "MODEL_ACTIVATE", "model_id": model_id}}

    def activate(self, model_id: str) -> dict[str, Any]:
        """A person's activation of a model for this workspace's research.

        Args:
            model_id: The model.

        Returns:
            `ACTIVATED` and the record.

        Raises:
            ValueError: `model_extension.installed:<id>` for an installed model;
                `model_extension.contract_failed:<code>`; `model_extension.sandbox_required:<id>`
                when no passed sandbox trial holds this identity.
        """
        if model_id in build_installed_alpha_model_catalog().adapter_ids:
            raise ValueError(f"model_extension.installed:{model_id}")
        contract = self._contract(model_id)
        failed = next((row["code"] for row in contract["findings"] if row["code"]), None)
        if failed is not None:
            raise ValueError(f"model_extension.contract_failed:{failed}")
        registry = ModelExtensionRegistry.read(self.workspace)
        sandbox = next(
            (
                value
                for value in reversed(registry.sandboxes)
                if value.model_id == model_id
                and value.numerical_binding_hash == contract["numerical_binding_hash"]
                and value.declaration_hash == contract["declaration_hash"]
                and value.passed
            ),
            None,
        )
        if sandbox is None:
            raise ValueError(f"model_extension.sandbox_required:{model_id}")
        record = ModelActivation(
            model_id=model_id,
            declaration_hash=contract["declaration_hash"],
            numerical_binding_hash=contract["numerical_binding_hash"],
            contract_receipt_hash=contract["contract_receipt_hash"],
            sandbox=sandbox,
            activated_at=self.clock(),
        )
        others = tuple(value for value in registry.active if value.model_id != model_id)
        registry.model_copy(update={"active": (*others, record)}).write(self.workspace)
        return {"status": "ACTIVATED", "activation": record.model_dump(mode="json")}

    def deactivate(self, model_id: str) -> dict[str, Any]:
        """A person's deactivation: the model leaves this workspace's catalog.

        A kept study that binds it still reads back, and its code stays while a kept record
        binds it.

        Args:
            model_id: The model.

        Returns:
            `DEACTIVATED`.

        Raises:
            ValueError: `model_extension.not_active:<id>`.
        """
        registry = ModelExtensionRegistry.read(self.workspace)
        kept = tuple(value for value in registry.active if value.model_id != model_id)
        if len(kept) == len(registry.active):
            raise ValueError(f"model_extension.not_active:{model_id}")
        registry.model_copy(update={"active": kept}).write(self.workspace)
        return {"status": "DEACTIVATED", "model_id": model_id}


__all__ = [
    "REGISTRY",
    "ModelActivation",
    "ModelExtensionRegistry",
    "ModelExtensions",
    "ModelSandboxRecord",
    "activated_models",
]
