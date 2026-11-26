"""Campaign-specific economic evidence and curated Factor checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, Self

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]

type FactorCampaignArtifactKind = Literal[
    "curated-checkpoint",
    "campaign-dossier",
    "campaign-replay",
    "deterministic-evidence",
    "review-decision",
    "program",
    "target-quality",
    "target-surface",
    "walk-forward-plan",
    "oos-evidence",
    "redundancy",
    "proposal",
    "research-input",
]
"""Exactly the durable children a Stage 2 campaign replay resolves.

A subset of the store's own artifact kinds rather than a copy of it: this package
reads thirteen of them and has no business naming the current-pointer kinds at
all. Stating the subset is what lets the dependency below be read-only in the
type system as well as in the import graph.
"""


class FactorCampaignArtifactReader(Protocol):
    """The read-only slice of the Factor artifact store this package depends on.

    A Protocol rather than the store's own class, and the reason is structural
    rather than stylistic: a development module that can *name* its publication
    owner is one import away from being able to move a current pointer, and the
    methodology-surface guard enforces the absence of that import for every Desk.
    A ``TYPE_CHECKING`` import is still an import -- it is in the graph the guard
    reads, and it is one edit away from being a runtime one.

    So the dependency is declared as what it actually is: a root to enumerate
    under, a URI convention, and one identity-verified contract load. The
    installed store satisfies this structurally, and nothing reachable from here
    can publish, activate or point at anything.
    """

    root: Path

    def uri(self, category: str, content_hash: str) -> str:
        """The content-addressed URI for one child of one category."""

    def load_pipeline_contract(
        self, *, artifact_kind: FactorCampaignArtifactKind, uri: str
    ) -> dict[str, object]:
        """Read back and verify one correctness-pipeline contract."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class CuratedFactorCheckpoint(_Contract):
    """Development-only, Host-resolved axis handed to Alpha Research."""

    kind: Literal["CuratedFactorCheckpoint"] = "CuratedFactorCheckpoint"
    deterministic_checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    campaign_dossier_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    curation_decision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_factor_ids: tuple[str, ...] = Field(min_length=1)
    curated_candidate_ids: tuple[str, ...]
    ordered_factor_ids: tuple[str, ...] = Field(min_length=1)
    feature_source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_only: Literal[True] = True
    redundancy_structure_hash: str | None = None
    redundancy_pruned_factor_ids: tuple[str, ...] | None = None
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_serializer(mode="wrap")  # type: ignore[untyped-decorator]
    def _serialize_populated(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Seal what the checkpoint asserts, not the shape it was declared with.

        A checkpoint sealed before redundancy pruning existed carries no key for
        it. Hashing the declared shape would move every such identity and break
        readback of the exact artifacts this successor change has to preserve.
        """

        serialized: dict[str, object] = handler(self)
        return {key: value for key, value in serialized.items() if value is not None}

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        pruned = self.redundancy_pruned_factor_ids or ()
        union = tuple(sorted((*self.control_factor_ids, *self.curated_candidate_ids)))
        if (
            self.curated_candidate_ids != tuple(sorted(set(self.curated_candidate_ids)))
            or pruned != tuple(sorted(set(pruned)))
            or not set(pruned) <= set(union)
            # Provenance and effect are one decision: an axis may only be
            # narrowed by naming the structure that justified the narrowing.
            or (self.redundancy_pruned_factor_ids is None)
            != (self.redundancy_structure_hash is None)
            or self.ordered_factor_ids != tuple(item for item in union if item not in set(pruned))
            or self.checkpoint_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"checkpoint_hash"}))
        ):
            raise ValueError("FACTOR_CURATED_CHECKPOINT_INVALID")
        return self


__all__ = ["CuratedFactorCheckpoint"]
