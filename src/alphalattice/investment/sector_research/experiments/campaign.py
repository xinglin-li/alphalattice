"""The typed and YAML request surface for one Sector forecast Campaign.

A Campaign is a comparison, so the request names *which installed methods to
compare* rather than one method to run. Everything a caller may state is a
handle, a range, a restatement of an installed constant, or an intent; nothing
here can carry a conclusion. The catalog identity, every method and numerical
binding, the compiled target, the calibration slopes and the decision policy are
derived by the Host from resolved evidence.

Parameters are restatements, exactly as they are for a single experiment: naming
them is a claim that must equal the installed frozen singleton, and omitting
them selects it. That is also where a refit schedule is stated -- the cadence
belongs to a method's recipe identity, so a request-level schedule would either
duplicate it or contradict it, and neither is a selection.

``development_only`` is required and must be ``true``. It is not decoration: the
field exists so a request that intends anything else fails at the boundary
rather than somewhere downstream, and there is no value of it that reaches a
current pointer.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import AuthoringError
from alphalattice.protocols.research_authoring.selection import (
    load_safe_yaml_document,
    require_selection_only_document,
)

from ..contracts import SectorResearchError
from ..models.distributed_lag_elastic_net import DISTRIBUTED_LAG_ELASTIC_NET_METHOD_ID
from ..models.ewma import EWMA_SECTOR_MEAN_METHOD_ID
from ..models.relative_strength import SECTOR_RELATIVE_STRENGTH_METHOD_ID
from ..models.zero import ZERO_SECTOR_FORECAST_METHOD_ID

SECTOR_CAMPAIGN_METHOD_IDS: tuple[str, ...] = (
    ZERO_SECTOR_FORECAST_METHOD_ID,
    EWMA_SECTOR_MEAN_METHOD_ID,
    SECTOR_RELATIVE_STRENGTH_METHOD_ID,
    DISTRIBUTED_LAG_ELASTIC_NET_METHOD_ID,
)
"""The installed clean-target batch a Campaign may select from."""

CONDITIONAL_SECTOR_METHOD_IDS: tuple[str, ...] = ("REGULARIZED_VAR", "PCA_FACTOR_VAR")
"""Named so a Campaign can publish their disposition without installing them.

They are not in the catalog and no adapter exists. A Campaign records
``NOT_TRIGGERED`` for each unless the pre-registered joint-dynamics trigger
fires, which is what makes "we did not run these" a published fact rather than
an absence a reader has to infer.
"""

DEFAULT_CALIBRATION_FOLD_COUNT = 5


class SectorCampaignRequest(BaseModel):  # type: ignore[misc]
    """What a caller is entitled to name for a whole Campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["SectorCampaignRequest"] = "SectorCampaignRequest"
    causal_outcome_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sector_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    method_ids: tuple[str, ...] = Field(min_length=2)
    method_parameters: dict[str, dict[str, int]] = Field(default_factory=dict)
    training_start: date
    training_end: date
    forecast_start: date
    forecast_end: date
    calibration_fold_count: int = Field(default=DEFAULT_CALIBRATION_FOLD_COUNT, ge=2, le=20)
    development_only: Literal[True]

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_request(self) -> Self:
        if self.method_ids != tuple(dict.fromkeys(self.method_ids)):
            raise SectorResearchError("sector_research.campaign_method_duplicated")
        uninstalled = set(self.method_ids) - set(SECTOR_CAMPAIGN_METHOD_IDS)
        if uninstalled:
            raise SectorResearchError("sector_research.campaign_method_not_installed")
        if ZERO_SECTOR_FORECAST_METHOD_ID not in set(self.method_ids):
            # Without the null control the comparison cannot answer whether a
            # Sector expected-return contribution is worth having at all.
            raise SectorResearchError("sector_research.campaign_zero_control_required")
        if set(self.method_parameters) - set(self.method_ids):
            raise SectorResearchError("sector_research.campaign_parameters_unselected_method")
        if self.training_start > self.training_end or self.forecast_start > self.forecast_end:
            raise SectorResearchError("sector_research.campaign_range_invalid")
        if self.training_start > self.forecast_start:
            raise SectorResearchError("sector_research.campaign_training_after_forecast")
        return self

    @classmethod
    def from_yaml(
        cls, document: str, *, causal_outcome_snapshot_hash: str, sector_revision: str
    ) -> Self:
        """Parse the experiment selections; refuse anything that smuggles authority.

        The refusal is by whitelist rather than by inspecting for known-bad
        keys: ``extra="forbid"`` means a document naming a catalog hash, a
        target identity or a sealed decision fails to parse, which is the only
        version of this check that stays correct as contracts grow.

        The whitelist was not enough on its own, because two of the whitelisted
        fields *were* authority: the committed document carried the causal
        outcome snapshot and the published Sector revision as literal digests, so
        "which evidence did this Campaign consume" was answered by a file under
        version control rather than by the workspace that holds it. Both are now
        operator inputs supplied here by the caller, and a document that states
        either is refused rather than silently overridden -- an override would
        leave a committed value that reads authoritative and is not.
        """

        # The one declaration loader, in its dialect (V276): a key written twice is refused.
        payload = load_safe_yaml_document(document)
        if not isinstance(payload, dict):
            raise SectorResearchError("sector_research.campaign_request_document_invalid")
        try:
            require_selection_only_document(payload)
        except AuthoringError as error:
            raise SectorResearchError(
                f"sector_research.campaign_request_states_authority:{error}"
            ) from error
        draft = dict(payload)
        draft["causal_outcome_snapshot_hash"] = causal_outcome_snapshot_hash
        draft["sector_revision"] = sector_revision
        methods = draft.pop("methods", None)
        if isinstance(methods, list):
            # The YAML shape is a list of {id, parameters} so a reader sees a
            # method and its restated constants together.
            selected: list[str] = []
            parameters: dict[str, dict[str, int]] = {}
            for entry in methods:
                if not isinstance(entry, dict) or "id" not in entry:
                    raise SectorResearchError("sector_research.campaign_method_entry_invalid")
                method_id = str(entry["id"])
                selected.append(method_id)
                stated = entry.get("parameters")
                if stated is not None:
                    if not isinstance(stated, dict):
                        raise SectorResearchError("sector_research.campaign_method_entry_invalid")
                    parameters[method_id] = {str(key): int(value) for key, value in stated.items()}
            draft["method_ids"] = tuple(selected)
            draft["method_parameters"] = parameters
        return cast(Self, cls.model_validate(draft))

    def parameters_for(self, method_id: str) -> dict[str, int] | None:
        stated = self.method_parameters.get(method_id)
        return dict(stated) if stated is not None else None

    @property
    def request_hash(self) -> str:
        """Content identity of the selections, for the Program to bind.

        Derived rather than stored: a caller supplying this field would be
        naming an identity, and the request boundary exists to refuse exactly
        that.
        """

        return str(canonical_hash(self.model_dump(mode="json")))


__all__ = [
    "CONDITIONAL_SECTOR_METHOD_IDS",
    "DEFAULT_CALIBRATION_FOLD_COUNT",
    "SECTOR_CAMPAIGN_METHOD_IDS",
    "SectorCampaignRequest",
]
