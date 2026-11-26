"""The formula factors a person activated and the daily Feature catalog they make (EX).

The workspace's registry, `runtime/extensions/features.json`, holds each activation; the daily
catalog this workspace computes is the shipped one with its activations appended
(`workspace_feature_catalog`), which its data update composes and binds. Each catalog an
activation or a deactivation makes is kept by its hash (`runtime/extensions/feature-catalogs`), so
a Panel built under it resolves its definition later (`feature_catalog_for`); a catalog change
recomputes the daily Panel at the next data update. The review and a person's acts are
`feature_extensions`'.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.kernel.quant.factor_contracts import FactorSpec

REGISTRY = Path("runtime") / "extensions" / "features.json"
CATALOG_REVISIONS = Path("runtime") / "extensions" / "feature-catalogs"


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class FeatureActivation(_Contract):
    """A person's activation of one formula factor into this workspace's daily catalog."""

    factor_id: str
    specification: FactorSpec
    preprocessing_recipe: str
    feature_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    trial_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    methodology_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    packet_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The review packet the person read, by content."""
    activated_by: Literal["HUMAN"] = "HUMAN"
    activated_at: datetime


class FeatureExtensionRegistry(_Contract):
    """The formula factors a workspace's daily catalog holds beyond the shipped ones."""

    active: tuple[FeatureActivation, ...] = ()

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
        """Replace the workspace's registry atomically, and keep the catalog it makes.

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
        _keep(workspace, workspace_feature_catalog(workspace))


def activated_feature_specs(workspace: Path) -> tuple[FactorSpec, ...]:
    """The formula factors a person activated in this workspace, in activation order.

    Args:
        workspace: The Host's workspace.

    Returns:
        Their specs.
    """
    return tuple(value.specification for value in FeatureExtensionRegistry.read(workspace).active)


def workspace_feature_catalog(workspace: Path | None) -> FeatureCatalog:
    """The daily Feature catalog this workspace computes: the shipped one with its activations.

    Args:
        workspace: The workspace; none reads the shipped catalog.

    Returns:
        The catalog; the shipped one itself while nothing is activated.
    """
    shipped = FeatureCatalog.load()
    specs = () if workspace is None else activated_feature_specs(workspace)
    if not specs:
        return shipped
    payload = shipped.to_payload()
    # The catalog's axis is its factors sorted by id; an activation takes its place in it.
    payload["factors"] = sorted(
        [*payload["factors"], *(json.loads(spec.model_dump_json()) for spec in specs)],
        key=lambda value: str(value["factor_id"]),
    )
    return FeatureCatalog.from_payload(payload)


def feature_catalog_for(workspace: Path, catalog_hash: str) -> FeatureCatalog | None:
    """The catalog a Panel recorded, by its hash: the shipped one, or one this workspace kept.

    Args:
        workspace: The workspace.
        catalog_hash: The catalog hash the Panel's lineage names.

    Returns:
        The catalog, or None when neither the shipped catalog nor a kept one has the hash.
    """
    shipped = FeatureCatalog.load()
    if shipped.binding.catalog_hash == catalog_hash:
        return shipped
    path = workspace / CATALOG_REVISIONS / f"{catalog_hash}.json"
    if not path.is_file():
        return None
    catalog = FeatureCatalog.from_payload(json.loads(path.read_text(encoding="utf-8")))
    return catalog if catalog.binding.catalog_hash == catalog_hash else None


def panel_feature_catalog(workspace: Path, panel_snapshot_hash: str) -> FeatureCatalog:
    """The catalog a workspace Panel was built under; the shipped one when it names no other.

    Args:
        workspace: The workspace, whose `artifacts` hold the Panel's manifest.
        panel_snapshot_hash: The Panel.

    Returns:
        The catalog its lineage names, resolved by `feature_catalog_for`, else the shipped one
        (whose owner then refuses a Panel it does not describe, as before).
    """
    resolver = ArtifactResolver(workspace / "artifacts")
    try:
        panel = resolver.load_feature_panel_manifest(
            resolver.feature_panel_manifest_uri(panel_snapshot_hash)
        )
    except (FileNotFoundError, ValueError):
        return FeatureCatalog.load()
    recorded = str(panel.get("safe_summary", {}).get("lineage", {}).get("catalog_hash", ""))
    return feature_catalog_for(workspace, recorded) or FeatureCatalog.load()


def _keep(workspace: Path, catalog: FeatureCatalog) -> None:
    path = workspace / CATALOG_REVISIONS / f"{catalog.binding.catalog_hash}.json"
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(".json.tmp")
    staged.write_text(
        json.dumps(catalog.to_payload(), sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    os.replace(staged, path)


__all__ = [
    "CATALOG_REVISIONS",
    "REGISTRY",
    "FeatureActivation",
    "FeatureExtensionRegistry",
    "activated_feature_specs",
    "feature_catalog_for",
    "panel_feature_catalog",
    "workspace_feature_catalog",
]
