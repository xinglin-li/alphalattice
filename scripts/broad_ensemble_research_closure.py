"""The IW184 live-closure research owners, ported from the sealed research sources.

Gate L (``run_broad_ensemble_live_model_closure.py``) and Gate M
(``run_heterogeneous_live_score_closure.py``) reconstruct and reopen the frozen
IW184 models against research evidence that lives in other worktrees. They used
to *execute* four modules of the monthly research worktree and one of the
heterogeneous research worktree, loaded by file path, and one of those modules
put its own worktree's ``src`` at the front of ``sys.path``. Nothing in this
tree owned the functions they called, so the closure ran product code nobody
reviewed here.

This module is the local executable owner of exactly those functions. Each was
ported from the research file whose sha256 the sealed Gate M source readback
names (``LIVE_SOURCE_READBACK_SHA256`` in ``scores/heterogeneous_product``),
with substitutions that reuse owners this tree already has: the estimator point
and the axis identity come from ``scores/product_recipe``, quarter arithmetic
and the seed-model kind from ``scores/product_replay``, the source-array
contract from ``inputs/panel_feature_views``, and the monthly refit driver is
this tree's own copy of the research one. The research files stay
identity-bound inputs: ``verify_ported_research_sources`` refuses when their
bytes no longer match the digests this port was taken from, so a closure can
never record sources the executed code has silently diverged from.

Historical source provenance and local executable identity are two records,
not one. ``MONTHLY_RESEARCH_SOURCES`` digests the files in the research
worktree, which is what this port was taken from and what a closure names as
its input; ``executable_owner_readback`` digests the workspace files that
actually run. The local driver supplies its own import bootstrap and requires
an explicit historical asset root for its operator CLI. It does not assume a
developer filesystem. The ported geometry, fitting and training-binding helpers
remain the numerical owners; operational path changes do not rewrite the sealed
research-source digests or claim byte equality between the two whole files.

Nothing here fits, predicts, scores, or writes. The runners still do that,
through ``monthly_runner`` and the installed Alpha runtime.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parent
PLAYPEN_ROOT = SCRIPTS_ROOT.parent
for _entry in (str(PLAYPEN_ROOT / "src"), str(SCRIPTS_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

# A sibling script by its directory at run time, by the repository namespace for types.
if TYPE_CHECKING:
    from scripts import run_monthly_alpha_refit_research as monthly_runner
else:
    import run_monthly_alpha_refit_research as monthly_runner

from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (  # noqa: E402
    build_dynamic_panel_lightgbm_recipe,
)
from alphalattice.capabilities.alpha_modeling.contracts import (  # noqa: E402
    AlphaModelRecipeEnvelope,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_views import (  # noqa: E402
    PanelFeatureSourceArrays,
)
from alphalattice.investment.alpha_research.scores.product_recipe import (  # noqa: E402
    INSTALLED_ALPHA_PRODUCT_RECIPE,
    PRODUCT_FEATURE_AXIS_HASH,
    PRODUCT_FEATURE_AXIS_ID,
    PRODUCT_FEATURE_COUNT,
    PRODUCT_PURGE_SESSIONS,
    PRODUCT_TRAINING_WINDOW_SESSIONS,
)
from alphalattice.investment.alpha_research.scores.product_replay import (  # noqa: E402
    SEED_MODEL_KIND,
    shift_quarter,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash  # noqa: E402

# ------------------------------------------------------------------ provenance

MONTHLY_RESEARCH_SOURCES: Final[dict[str, str]] = {
    "scripts/run_monthly_alpha_refit_research.py": (
        "7c1fc28fa3bd337210cfc0df78996a9de298abb803182c20c4ea8a632c9be856"
    ),
    "workspaces/monthly-alpha-refit-research/run_iw184_quarterly_5y_ensemble.py": (
        "cc82d07621ac0710ef00873d9f9b483b5c2f89a6384e34abd25bf622d4e5fa76"
    ),
    "workspaces/monthly-alpha-refit-research/feature-recipe-exploration/"
    "run_feature_recipe_exploration.py": (
        "eb596aa417961601df2a104cdc78de40f24bb99a63bea4714d984a5342dcacb2"
    ),
    "workspaces/monthly-alpha-refit-research/frozen-iw184-holdout-smoke/"
    "run_frozen_iw184_holdout_smoke.py": (
        "a678ccb7da9d19b7427a6be226baa0b53c864a25ff22fb71c6dadb32394d880a"
    ),
}
"""The monthly research worktree files this port was taken from, by sha256.

These are the digests the sealed Gate M source readback names, and they
describe the research worktree, never this tree: the local copy of the refit
driver carries the startup block below and is not byte identical. A closure
run still lists the research files in its own readback; it never executes
them.
"""

CANDIDATE_AXIS_RUNNER_SHA256: Final = (
    "5eb5b9264e1da34dbeae4b0d866eee68f52455717c3960b432a2a09e206b60e5"
)
"""``scripts/run_heterogeneous_alpha_strategy_gate_d2.py`` of the heterogeneous
research worktree, the source of the Gate D2 axis loader below."""

CANDIDATE_AXES_RELATIVE_PATH: Final = (
    "workspaces/monthly-alpha-refit-research/feature-recipe-exploration/candidate-axes.json"
)
SOURCE_CACHE_RELATIVE_PATH: Final = (
    "workspaces/monthly-alpha-refit-research/frozen-iw184-holdout-smoke/cache/current-source"
)

EXECUTABLE_OWNER_PATHS: Final[tuple[Path, ...]] = (
    Path(__file__).resolve(),
    SCRIPTS_ROOT / "run_monthly_alpha_refit_research.py",
    PLAYPEN_ROOT / "src/alphalattice/investment/alpha_research/scores/product_recipe.py",
    PLAYPEN_ROOT / "src/alphalattice/investment/alpha_research/scores/product_replay.py",
    PLAYPEN_ROOT
    / "src/alphalattice/capabilities/alpha_modeling/adapters/lightgbm_dynamic_panel.py",
)
"""The workspace files whose code decides what the ported closure computes.

This is the local executable identity, recomputed on every run and recorded
beside the research digests above. The refit driver appears in both records
with different digests, which is the point of keeping them separate.
"""


class ResearchClosureError(RuntimeError):
    """Typed refusal from the ported research closure.

    A ``RuntimeError`` so that the runners' existing ``except RuntimeError``
    around the vintage geometry keeps producing its diagnostic message.
    """


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_ported_research_sources(
    *, monthly_worktree: Path, gate_d2_runner: Path | None = None
) -> dict[str, str]:
    """Refuse unless every research file this port was taken from is byte identical.

    Returns the verified absolute paths and digests so a closure can record them
    beside its own source readback.
    """

    expected: dict[Path, str] = {
        monthly_worktree / relative: digest for relative, digest in MONTHLY_RESEARCH_SOURCES.items()
    }
    if gate_d2_runner is not None:
        expected[gate_d2_runner] = CANDIDATE_AXIS_RUNNER_SHA256
    verified: dict[str, str] = {}
    for path, digest in expected.items():
        if not path.is_file():
            raise ResearchClosureError(f"alpha_research.ported_research_source_absent:{path}")
        actual = sha256_of(path)
        if actual != digest:
            raise ResearchClosureError(
                "alpha_research.ported_research_source_provenance_mismatch:"
                f"{path}:{actual}!={digest}"
            )
        verified[str(path.resolve())] = actual
    return verified


def executable_owner_readback() -> dict[str, dict[str, Any]]:
    """Digests of the workspace files that execute in place of the research modules."""

    return {
        path.relative_to(PLAYPEN_ROOT).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path),
        }
        for path in EXECUTABLE_OWNER_PATHS
    }


# ------------------------------------------------------- IW184 quarterly ensemble

IW184_CANDIDATE: Final = PRODUCT_FEATURE_AXIS_ID
TRAINING_SESSIONS: Final = PRODUCT_TRAINING_WINDOW_SESSIONS
PURGE_SESSIONS: Final = PRODUCT_PURGE_SESSIONS
PREDICTION_MONTHS: Final = 15
OPERATIONAL_MONTHS: Final = 12
FEATURE_VIEW_ID: Final = "SPARSE_SESSION_AMPLITUDE"
REFIT_QUARTER_START_MONTHS: Final = INSTALLED_ALPHA_PRODUCT_RECIPE.refit_quarter_start_months


def quarter_specs(
    *,
    source_sessions: tuple[date, ...],
    validation_sessions: tuple[date, ...],
    first_vintage: str,
    last_vintage: str,
) -> list[dict[str, Any]]:
    """The fixed-1260 quarterly vintage windows over the monthly refit geometry.

    Ported from ``run_iw184_quarterly_5y_ensemble._quarter_specs``; the vintage
    range and validation sessions are parameters here because both runners
    always overrode the research module's constants with the live ones.
    """

    geometry = monthly_runner._monthly_geometry(
        source_sessions=source_sessions,
        validation_sessions=validation_sessions,
    )
    source_lookup = {session: index for index, session in enumerate(source_sessions)}
    selected: list[dict[str, Any]] = []
    for ordinal, row in enumerate(geometry):
        month, base_training, _validation, purge = row
        if (
            int(month[5:7]) not in REFIT_QUARTER_START_MONTHS
            or month < first_vintage
            or month > last_vintage
        ):
            continue
        training_last_position = source_lookup[base_training[-1]]
        start = training_last_position - TRAINING_SESSIONS + 1
        if start < 0:
            raise ResearchClosureError(f"broad_ensemble_quarterly_5y.history_insufficient:{month}")
        training = source_sessions[start : training_last_position + 1]
        if (
            len(training) != TRAINING_SESSIONS
            or source_sessions[training_last_position + 1] != purge
        ):
            raise ResearchClosureError(
                f"broad_ensemble_quarterly_5y.training_clock_invalid:{month}"
            )
        future = geometry[ordinal : ordinal + PREDICTION_MONTHS]
        transform = tuple(session for future_row in future for session in future_row[2])
        if not transform:
            raise ResearchClosureError(f"broad_ensemble_quarterly_5y.transform_axis_empty:{month}")
        selected.append(
            {
                "month": month,
                "ordinal": ordinal,
                "training": training,
                "purge": purge,
                "transform": transform,
                "prediction_months": [future_row[0] for future_row in future],
                "operational_months": [future_row[0] for future_row in future[:OPERATIONAL_MONTHS]],
            }
        )
    expected = []
    month = first_vintage
    while month <= last_vintage:
        expected.append(month)
        month = shift_quarter(month, 1)
    if [row["month"] for row in selected] != expected:
        raise ResearchClosureError("broad_ensemble_quarterly_5y.vintage_axis_invalid")
    return selected


def seed_model_identity(
    *,
    axis_hash: str,
    recipe_hash: str,
    seed: int,
    spec: Mapping[str, Any],
    source_resolution_hash: str,
) -> dict[str, Any]:
    """The identity a Gate L payload is bound to, ported from the ensemble's ``_identity``."""

    identity: dict[str, Any] = {
        "kind": SEED_MODEL_KIND,
        "candidate": IW184_CANDIDATE,
        "candidate_axis_hash": axis_hash,
        "feature_view_id": FEATURE_VIEW_ID,
        "annual_root_hash": monthly_runner.ANNUAL_ROOT_HASH,
        "source_resolution_hash": source_resolution_hash,
        "recipe_hash": recipe_hash,
        "seed": seed,
        "quarter_start": spec["month"],
        "training_window_sessions": TRAINING_SESSIONS,
        "training_first": spec["training"][0].isoformat(),
        "training_last": spec["training"][-1].isoformat(),
        "training_session_axis_hash": str(
            canonical_hash([value.isoformat() for value in spec["training"]])
        ),
        "purge_session": spec["purge"].isoformat(),
        "purge_session_count": PURGE_SESSIONS,
        "prediction_months": spec["prediction_months"],
        "operational_months": spec["operational_months"],
        "transform_session_axis_hash": str(
            canonical_hash([value.isoformat() for value in spec["transform"]])
        ),
        "score_aggregation": INSTALLED_ALPHA_PRODUCT_RECIPE.score_aggregation,
    }
    identity["identity_hash"] = str(canonical_hash(identity))
    return identity


def candidate_axis(monthly_worktree: Path) -> tuple[str, ...]:
    """The ordered IW184 feature axis, read from research data and bound to the product pin.

    The research explorer merged three candidate files; only ``candidate-axes.json``
    carries this candidate, and the installed product recipe already states the
    axis hash and count the file must resolve to.
    """

    path = monthly_worktree / CANDIDATE_AXES_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", {})
    if IW184_CANDIDATE not in candidates:
        raise ResearchClosureError(f"alpha_research.candidate_axis_absent:{path}")
    axis = tuple(str(value) for value in candidates[IW184_CANDIDATE]["ordered_feature_ids"])
    if (
        len(axis) != PRODUCT_FEATURE_COUNT
        or str(canonical_hash(list(axis))) != PRODUCT_FEATURE_AXIS_HASH
    ):
        raise ResearchClosureError("alpha_research.candidate_axis_identity_invalid")
    return axis


def broad_ensemble_recipe(seed: int) -> AlphaModelRecipeEnvelope:
    """The frozen estimator recipe for one seed, from the installed product recipe point."""

    return build_dynamic_panel_lightgbm_recipe(
        INSTALLED_ALPHA_PRODUCT_RECIPE.estimator_point.resolve(seed=seed)
    )


# ------------------------------------------------------------- source cache


def load_cached_panel_source(cache_root: Path) -> PanelFeatureSourceArrays:
    """Open the qualified current-source cache, ported from the smoke runner's ``_inspect_source``.

    Every array is verified by size and digest against the cache manifest, and
    the manifest against its own identity hash, before anything is mapped.
    """

    manifest_path = cache_root / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    if manifest.get("cache_identity_hash") != canonical_hash(
        {key: value for key, value in manifest.items() if key != "cache_identity_hash"}
    ):
        raise ResearchClosureError("holdout_smoke.source_cache_identity_invalid")
    loaded: dict[str, np.ndarray[Any, Any]] = {}
    for name, artifact in manifest["arrays"].items():
        path = cache_root / artifact["path"]
        if (
            not path.is_file()
            or path.stat().st_size != artifact["bytes"]
            or sha256_of(path) != artifact["sha256"]
        ):
            raise ResearchClosureError(f"holdout_smoke.source_cache_array_invalid:{name}")
        loaded[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    return PanelFeatureSourceArrays(
        formation_sessions=tuple(
            date.fromisoformat(value) for value in manifest["formation_sessions"]
        ),
        holding_end_sessions=tuple(
            date.fromisoformat(value) for value in manifest["holding_end_sessions"]
        ),
        ordered_listing_ids=tuple(manifest["ordered_listing_ids"]),
        ordered_factor_ids=tuple(manifest["ordered_factor_ids"]),
        absolute_state_factor_ids=tuple(manifest["absolute_state_factor_ids"]),
        ordered_sector_ids=tuple(manifest["ordered_sector_ids"]),
        sector_by_listing_id=MappingProxyType(dict(manifest["sector_by_listing_id"])),
        raw_formula_values=loaded["raw_formula_values"],
        total_return_target_z=loaded["total_return_target_z"],
        raw_log_execution_returns=loaded["raw_log_execution_returns"],
        raw_simple_execution_returns=loaded["raw_simple_execution_returns"],
        sector_context_values=loaded["sector_context_values"],
        market_context_values=loaded["market_context_values"],
        source_identity_hashes=MappingProxyType(dict(manifest["source_identity_hashes"])),
    )


# ------------------------------------------------------------ Gate D2 axes

CANDIDATE_AXIS_ASSET_IDS: Final = (
    "G0_IW184",
    "M0_RAW_MOMENTUM",
    "G2_R0_TREND",
    "G2_R2_CANDIDATE_TARGET",
    "G7_R2_CONTEXTUAL_MOMENTUM",
    "G7_R1_CONTEXTUAL_MOMENTUM",
    "G1_R2_WITHIN_SECTOR",
    "G1_R1_WITHIN_SECTOR",
    "G6_R0_FAST_REBOUND",
    "G6_R1_FAST_REBOUND_CONTEXT",
    "G5_R1_LIQUIDITY_CONTEXT",
    "G5_R2_LIQUIDITY_WEIGHTED",
)
CANDIDATE_AXIS_BATCH_OWNER_SHA256: Final[dict[str, str]] = {
    "batch61_factor_model_in_the_book.py": (
        "d15a7f95ec9a80ae700deba0ed91384d9989b703fb9fc6e3ec483fc8e403e1a1"
    ),
    "batch81_tranches_replace_cadence.py": (
        "8b66b2a1b6b7d21618a708d66310302009e9b56e46ef567b2045f7b384daafef"
    ),
    "batch169_the_illiquidity_frontier_with_capacity.py": (
        "69a10e68ed4cbf3c0d4440bdd0734c64908b1cc23c95e013adb60752b9e0e266"
    ),
}
"""The Risk lab book owners Gate D2 pins before admitting its axes; digests only."""


class CandidateAxisRefusal(RuntimeError):
    """Typed fail-closed refusal for invalid full-history book evidence."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}:{detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class CandidateAxisRoots:
    """Where the Gate D2 evidence sits, derived exactly as the research runner derived it."""

    score_root: Path
    admission: Path
    risk_diagonal: Path
    current_source: Path
    successor_composite: Path
    risk_panel: Path
    batch_owner_root: Path


@dataclass(frozen=True, slots=True)
class CandidateAxes:
    sessions: tuple[str, ...]
    listings: tuple[str, ...]
    returns: np.ndarray[Any, np.dtype[np.float64]]
    eligible: np.ndarray[Any, np.dtype[np.bool_]]
    variance: np.ndarray[Any, np.dtype[np.float64]]
    sector_codes: np.ndarray[Any, np.dtype[np.int32]]


def candidate_axis_roots(
    *, heterogeneous_worktree: Path, monthly_worktree: Path, risk_lab_root: Path
) -> CandidateAxisRoots:
    research = heterogeneous_worktree / "workspaces" / "heterogeneous-alpha-strategy-research"
    output = research / "gate-d-full-history-evidence"
    smoke = monthly_worktree / "workspaces/monthly-alpha-refit-research/frozen-iw184-holdout-smoke"
    return CandidateAxisRoots(
        score_root=output / "score-surfaces",
        admission=output / "full-history-evidence-admission.json",
        risk_diagonal=output / "full-history-risk-diagonal.npy",
        current_source=smoke / "cache/current-source",
        successor_composite=smoke / "current-successor-tail/composite",
        risk_panel=risk_lab_root / "_panel_shared",
        batch_owner_root=risk_lab_root,
    )


def _d2_sha256(path: Path) -> str:
    if not path.is_file():
        raise CandidateAxisRefusal("SOURCE_EVIDENCE_MISSING", str(path))
    return sha256_of(path)


def _d2_read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CandidateAxisRefusal("SOURCE_EVIDENCE_MISSING", str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CandidateAxisRefusal("JSON_OBJECT_REQUIRED", str(path))
    return cast(dict[str, Any], payload)


def _d2_artifact(path: Path, *, include_mtime: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise CandidateAxisRefusal("SOURCE_EVIDENCE_MISSING", str(path))
    stat = path.stat()
    payload: dict[str, Any] = {
        "absolute_path": str(path.resolve()),
        "bytes": stat.st_size,
        "sha256": _d2_sha256(path),
    }
    if include_mtime:
        payload["mtime_ns"] = stat.st_mtime_ns
    if path.suffix == ".npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        payload.update({"shape": list(values.shape), "dtype": str(values.dtype)})
    return payload


def _d2_validate_seal(payload: Mapping[str, Any], field: str, code: str) -> None:
    if field not in payload:
        raise CandidateAxisRefusal(code, "field_absent")
    expected = str(canonical_hash({key: value for key, value in payload.items() if key != field}))
    if payload[field] != expected:
        raise CandidateAxisRefusal(code, f"{payload[field]}!={expected}")


def validate_candidate_axis_batch_sources(roots: CandidateAxisRoots) -> dict[str, str]:
    """The pinned Risk lab book owners must be byte identical; nothing in them runs here."""

    verified: dict[str, str] = {}
    for name, expected in CANDIDATE_AXIS_BATCH_OWNER_SHA256.items():
        path = roots.batch_owner_root / name
        actual = _d2_sha256(path)
        if actual != expected:
            raise CandidateAxisRefusal(
                "PINNED_BOOK_OWNER_HASH_MISMATCH", f"{path}:{actual}!={expected}"
            )
        verified[str(path.resolve())] = actual
    return verified


def load_candidate_axes(
    roots: CandidateAxisRoots,
) -> tuple[CandidateAxes, dict[str, dict[str, Any]]]:
    """Admit the Gate D0/D1 evidence and assemble the book axes, ported from ``_load_axes``."""

    admission = _d2_read_json(roots.admission)
    _d2_validate_seal(admission, "admission_hash", "D0_ADMISSION_HASH_INVALID")
    if admission.get("asset_count") != len(CANDIDATE_AXIS_ASSET_IDS) or set(
        admission.get("assets", {})
    ) != set(CANDIDATE_AXIS_ASSET_IDS):
        raise CandidateAxisRefusal("D0_ASSET_SET_INVALID")

    score_manifests = {
        asset_id: _d2_read_json(roots.score_root / f"{asset_id}.json")
        for asset_id in CANDIDATE_AXIS_ASSET_IDS
    }
    reference = score_manifests["G0_IW184"]["identity"]
    sessions = tuple(str(value) for value in reference["sessions"])
    listings = tuple(str(value) for value in reference["listings"])
    if (
        len(sessions) != 1751
        or len(listings) != 466
        or sessions[0] != "2019-08-05"
        or sessions[-1] != "2026-07-29"
    ):
        raise CandidateAxisRefusal("D0_AXIS_INVALID")
    for asset_id, manifest in score_manifests.items():
        identity = manifest["identity"]
        expected_hash = str(canonical_hash(identity))
        if manifest.get("identity_hash") != expected_hash:
            raise CandidateAxisRefusal("SCORE_IDENTITY_HASH_INVALID", asset_id)
        if tuple(identity["sessions"]) != sessions or tuple(identity["listings"]) != listings:
            raise CandidateAxisRefusal("SCORE_AXIS_MISMATCH", asset_id)
        artifact = manifest["artifact"]
        if _d2_artifact(roots.score_root / f"{asset_id}.npy") != artifact:
            raise CandidateAxisRefusal("SCORE_ARTIFACT_INVALID", asset_id)
        admitted = admission["assets"][asset_id]
        if admitted["identity_hash"] != manifest["identity_hash"]:
            raise CandidateAxisRefusal("SCORE_ADMISSION_MISMATCH", asset_id)

    composite = _d2_read_json(roots.successor_composite / "manifest.json")
    source_sessions = tuple(str(value) for value in composite["formation_sessions"])
    source_listings = tuple(str(value) for value in composite["ordered_listing_ids"])
    current = _d2_read_json(roots.current_source / "manifest.json")
    if tuple(str(value) for value in current["formation_sessions"]) != source_sessions:
        raise CandidateAxisRefusal("CURRENT_COMPOSITE_SESSION_AXIS_MISMATCH")
    if tuple(str(value) for value in current["ordered_listing_ids"]) != source_listings:
        raise CandidateAxisRefusal("CURRENT_COMPOSITE_LISTING_AXIS_MISMATCH")
    session_lookup = {value: index for index, value in enumerate(source_sessions)}
    listing_lookup = {value: index for index, value in enumerate(source_listings)}
    try:
        session_positions = [session_lookup[value] for value in sessions]
        listing_positions = [listing_lookup[value] for value in listings]
    except KeyError as exc:
        raise CandidateAxisRefusal("BOOK_AXIS_UNRESOLVED", str(exc)) from exc
    deployment_returns = np.load(
        roots.successor_composite / "returns.npy", mmap_mode="r", allow_pickle=False
    )
    deployment_eligible = np.load(
        roots.successor_composite / "eligible.npy", mmap_mode="r", allow_pickle=False
    )
    returns: np.ndarray[Any, np.dtype[np.float64]] = np.full(
        (len(sessions), len(listings)), np.nan, dtype=np.float64
    )
    eligible: np.ndarray[Any, np.dtype[np.bool_]] = np.zeros(
        (len(sessions), len(listings)), dtype=np.bool_
    )

    panel_ready = _d2_read_json(roots.risk_panel / "ready.json")
    panel_sessions = tuple(str(value)[:10] for value in panel_ready["sessions"])
    panel_listings = tuple(str(value) for value in panel_ready["listing_ids"])
    panel_session_lookup = {value: index for index, value in enumerate(panel_sessions)}
    panel_listing_lookup = {value: index for index, value in enumerate(panel_listings)}
    panel_returns = np.load(roots.risk_panel / "returns.npy", mmap_mode="r", allow_pickle=False)
    panel_eligible = np.load(roots.risk_panel / "eligible.npy", mmap_mode="r", allow_pickle=False)
    output_listing_lookup = {value: index for index, value in enumerate(listings)}
    try:
        panel_output_columns = [output_listing_lookup[value] for value in panel_listings]
    except KeyError as exc:
        raise CandidateAxisRefusal("DEVELOPMENT_LISTING_AXIS_UNRESOLVED", str(exc)) from exc
    development_rows = [index for index, value in enumerate(sessions) if value <= "2024-07-29"]
    try:
        panel_rows = [panel_session_lookup[sessions[index]] for index in development_rows]
    except KeyError as exc:
        raise CandidateAxisRefusal("DEVELOPMENT_SESSION_AXIS_UNRESOLVED", str(exc)) from exc
    returns[np.ix_(development_rows, panel_output_columns)] = panel_returns[
        np.ix_(panel_rows, [panel_listing_lookup[value] for value in panel_listings])
    ]
    eligible[np.ix_(development_rows, panel_output_columns)] = panel_eligible[
        np.ix_(panel_rows, [panel_listing_lookup[value] for value in panel_listings])
    ]
    deployment_rows = [index for index, value in enumerate(sessions) if value >= "2024-07-30"]
    deployment_source_rows = [session_positions[index] for index in deployment_rows]
    returns[np.ix_(deployment_rows, range(len(listings)))] = deployment_returns[
        np.ix_(deployment_source_rows, listing_positions)
    ]
    eligible[np.ix_(deployment_rows, range(len(listings)))] = deployment_eligible[
        np.ix_(deployment_source_rows, listing_positions)
    ]
    risk = np.asarray(
        np.load(roots.risk_diagonal, mmap_mode="r", allow_pickle=False)[
            np.ix_(session_positions, listing_positions)
        ],
        dtype=np.float64,
    )
    if risk.shape != returns.shape or np.any(~np.isfinite(risk)) or np.any(risk <= 0.0):
        raise CandidateAxisRefusal("BOOK_RISK_SUPPORT_INVALID")
    if np.any(eligible & ~np.isfinite(returns)):
        raise CandidateAxisRefusal("ELIGIBLE_RETURN_NOT_FINITE")
    sector_by_listing = cast(Mapping[str, str], current["sector_by_listing_id"])
    sector_names = tuple(sorted({sector_by_listing[value] for value in listings}))
    sector_lookup = {value: index for index, value in enumerate(sector_names)}
    sector_codes: np.ndarray[Any, np.dtype[np.int32]] = np.asarray(
        [sector_lookup[sector_by_listing[value]] for value in listings], dtype=np.int32
    )
    return CandidateAxes(sessions, listings, returns, eligible, risk, sector_codes), score_manifests


__all__ = [
    "CANDIDATE_AXES_RELATIVE_PATH",
    "CANDIDATE_AXIS_ASSET_IDS",
    "CANDIDATE_AXIS_BATCH_OWNER_SHA256",
    "CANDIDATE_AXIS_RUNNER_SHA256",
    "EXECUTABLE_OWNER_PATHS",
    "MONTHLY_RESEARCH_SOURCES",
    "SOURCE_CACHE_RELATIVE_PATH",
    "CandidateAxes",
    "CandidateAxisRefusal",
    "CandidateAxisRoots",
    "ResearchClosureError",
    "broad_ensemble_recipe",
    "candidate_axis",
    "candidate_axis_roots",
    "executable_owner_readback",
    "load_cached_panel_source",
    "load_candidate_axes",
    "monthly_runner",
    "quarter_specs",
    "seed_model_identity",
    "sha256_of",
    "validate_candidate_axis_batch_sources",
    "verify_ported_research_sources",
]
