"""One-session Factor Research target projected from causal execution outcomes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.causal_outcomes.execution.contracts import (
    CausalExecutionOutcomeManifest,
    CausalExecutionOutcomeMarker,
    DevelopmentOnlyExecutionOutcomeManifest,
    DevelopmentOnlyExecutionOutcomeMarker,
)
from alphalattice.foundation.causal_outcomes.execution.methods import (
    ExecutionOutcomeMethodBinding,
    ExecutionOutcomeMethodError,
    ExecutionOutcomeMethodSeal,
    ExecutionOutcomeMethodSealMarker,
    build_installed_execution_outcome_method_catalog,
    build_installed_execution_outcome_publication_policy_catalog,
    build_one_session_recipe,
    verify_execution_outcome_method_seal_marker,
)
from alphalattice.foundation.factor_research.programs.sealed import seal_contract
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]

_ELIGIBLE_EXECUTION_STATUSES = frozenset({"VERIFIED_ELIGIBLE", "ASSUMED_ELIGIBLE_FROM_DAILY_BAR"})
_REQUIRED_COLUMNS = frozenset(
    {
        "listing_id",
        "formation_session",
        "formation_close_at",
        "entry_session",
        "entry_open_at",
        "holding_end_session",
        "holding_end_open_at",
        "actual_session_span",
        "entry_status",
        "holding_end_status",
        "entry_open_split_adjusted",
        "holding_end_open_split_adjusted",
        "period_dividend_split_adjusted",
        "simple_return",
        "row_hash",
    }
)


class FactorTargetBoundaryError(ValueError):
    """Stable failure raised before target evidence can be admitted."""


@dataclass(frozen=True, slots=True)
class _ResolvedOutcomeMethod:
    """The outcome method one target compilation is entitled to assume."""

    expected_span: int
    entry_timing: str
    exit_timing: str
    price_basis: str
    corporate_action_identity: str
    sealed: bool


def _legacy_one_session_method() -> _ResolvedOutcomeMethod:
    recipe = build_one_session_recipe()
    return _ResolvedOutcomeMethod(
        expected_span=recipe.actual_session_span,
        entry_timing=recipe.entry_timing,
        exit_timing=recipe.exit_timing,
        price_basis=recipe.price_basis,
        corporate_action_identity=recipe.corporate_action_identity,
        sealed=False,
    )


#: The compatibility route for snapshots published before the method seam
#: existed. It asserts the one-session span and timing and nothing else -- in
#: particular it makes no claim about the source snapshot's method seal, because
#: an unsealed snapshot has no method authority to borrow. Numbers come from the
#: installed recipe so the two cannot drift apart.
LEGACY_ONE_SESSION_TARGET_METHOD = _legacy_one_session_method()


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class FactorTargetPolicy(_Contract):
    """Bind formation and execution timing to the Factor fit and economic targets.

    Attributes:
        kind: Target-policy discriminator.
        formation_frequency: Daily formation schedule.
        prediction_horizon_sessions: Number of common sessions between entry and exit.
        information_cutoff: Official formation close used as the information boundary.
        entry_timing: Next-common-session official open.
        exit_timing: Installed horizon-specific official-open exit.
        fit_target_semantics: Log execution return used by research statistics.
        economic_return_semantics: Simple execution return used for economic interpretation.
        source_price_basis: Split-adjusted open price field.
        source_action_semantics: Provider adjustment and period-dividend convention.
        policy_hash: Canonical identity of the declared target policy.
    """

    kind: Literal["FactorTargetPolicy"] = "FactorTargetPolicy"
    formation_frequency: Literal["DAILY"] = "DAILY"
    prediction_horizon_sessions: int = Field(default=1, ge=1)
    information_cutoff: Literal["FORMATION_OFFICIAL_CLOSE"] = "FORMATION_OFFICIAL_CLOSE"
    entry_timing: Literal["NEXT_COMMON_SESSION_OFFICIAL_OPEN"] = "NEXT_COMMON_SESSION_OFFICIAL_OPEN"
    exit_timing: Literal[
        "FOLLOWING_COMMON_SESSION_OFFICIAL_OPEN",
        "FIFTH_FOLLOWING_COMMON_SESSION_OFFICIAL_OPEN",
    ] = "FOLLOWING_COMMON_SESSION_OFFICIAL_OPEN"
    """Widened to the union of installed exit timings, never respelled: the
    frozen value keeps its exact string and its position as the default, so the
    one-session ``policy_hash`` every published surface carries does not move.
    A declaration outside the installed set still fails at the field."""

    fit_target_semantics: Literal["LOG_EXECUTION_RETURN"] = "LOG_EXECUTION_RETURN"
    economic_return_semantics: Literal["SIMPLE_EXECUTION_RETURN"] = "SIMPLE_EXECUTION_RETURN"
    source_price_basis: Literal["open_split_adjusted"] = "open_split_adjusted"
    source_action_semantics: Literal["provider-split-adjusted-open-and-period-dividend"] = (
        "provider-split-adjusted-open-and-period-dividend"
    )
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> FactorTargetPolicy:
        """Verify the target policy against its canonical contents.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: The recorded policy hash differs from its fields.
        """
        expected = canonical_hash(self.model_dump(mode="json", exclude={"policy_hash"}))
        if self.policy_hash != expected:
            raise ValueError("Factor target policy hash is invalid")
        return self


class FactorTargetMissingReasonCount(_Contract):
    """Count target rows missing for one declared execution or source reason.

    Attributes:
        reason: Entry, exit, or source-return condition which prevents a target.
        row_count: Positive number of rows attributed to that reason.
    """

    reason: Literal[
        "ENTRY_NOT_EXECUTABLE",
        "EXIT_NOT_EXECUTABLE",
        "SOURCE_RETURN_MISSING",
    ]
    row_count: int = Field(ge=1)


class FactorTargetQualityReport(_Contract):
    """Reconcile the projected target population and its numerical quality.

    Attributes:
        kind: Quality-report discriminator.
        source_outcome_snapshot_hash: Outcome snapshot from which targets were projected.
        policy_hash: Target policy applied to those outcomes.
        row_count: Total projected rows, including typed missing values.
        formation_count: Number of formation sessions represented.
        listing_count: Number of listings on the target axis.
        valid_target_count: Rows carrying finite valid target returns.
        typed_missing_count: Rows whose target is absent with a declared reason.
        missing_reasons: Canonical reason counts covering the missing rows.
        minimum_log_return: Minimum finite fit target, or None if none is available.
        maximum_log_return: Maximum finite fit target, or None if none is available.
        maximum_absolute_log_return: Largest finite target magnitude, or None if none is available.
        formula_mismatch_count: Zero: source simple returns were reproduced exactly.
        duplicate_key_count: Zero: formation/listing keys were unique.
        nonpositive_gross_return_count: Zero: admitted gross returns were positive.
        quality_hash: Canonical identity of the quality evidence.
    """

    kind: Literal["FactorTargetQualityReport"] = "FactorTargetQualityReport"
    source_outcome_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(ge=1)
    formation_count: int = Field(ge=1)
    listing_count: int = Field(ge=1)
    valid_target_count: int = Field(ge=0)
    typed_missing_count: int = Field(ge=0)
    missing_reasons: tuple[FactorTargetMissingReasonCount, ...]
    minimum_log_return: float | None
    maximum_log_return: float | None
    maximum_absolute_log_return: float | None
    formula_mismatch_count: Literal[0] = 0
    duplicate_key_count: Literal[0] = 0
    nonpositive_gross_return_count: Literal[0] = 0
    quality_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_report(self) -> FactorTargetQualityReport:
        """Reconcile valid and missing counts, reason order, and quality identity.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: Counts do not cover the surface, reasons are out of order, or the hash is
                invalid.
        """
        if self.valid_target_count + self.typed_missing_count != self.row_count:
            raise ValueError("Factor target quality counts do not cover the surface")
        if sum(item.row_count for item in self.missing_reasons) != self.typed_missing_count:
            raise ValueError("Factor target missing reasons do not reconcile")
        if self.missing_reasons != tuple(
            sorted(self.missing_reasons, key=lambda item: item.reason)
        ):
            raise ValueError("Factor target missing reasons are not canonical")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"quality_hash"}))
        if self.quality_hash != expected:
            raise ValueError("Factor target quality hash is invalid")
        return self


class FactorTargetSurfaceManifest(_Contract):
    """Bind one target table to its source, policy, quality, and formation axes.

    Attributes:
        kind: Target-surface discriminator.
        source_outcome_snapshot_hash: Source execution-outcome identity.
        source_outcome_manifest_ref: Reference used to reopen that outcome manifest.
        policy_hash: Qualified target-policy identity.
        quality_hash: Quality report retained with the target table.
        first_formation_session: Earliest projected formation session.
        last_formation_session: Latest projected formation session.
        formation_count: Number of distinct formation sessions.
        listing_count: Number of listings per complete formation axis.
        row_count: Total table rows.
        table_content_hash: Measured identity of the projected table contents.
        surface_hash: Canonical identity of this manifest.
    """

    kind: Literal["FactorTargetSurfaceManifest"] = "FactorTargetSurfaceManifest"
    source_outcome_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_outcome_manifest_ref: str = Field(min_length=1)
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    first_formation_session: date
    last_formation_session: date
    formation_count: int = Field(ge=1)
    listing_count: int = Field(ge=1)
    row_count: int = Field(ge=1)
    table_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_manifest(self) -> FactorTargetSurfaceManifest:
        """Verify the formation range and canonical target-surface identity.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: The range is reversed or the manifest hash is invalid.
        """
        if self.last_formation_session < self.first_formation_session:
            raise ValueError("Factor target formation range is reversed")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"surface_hash"}))
        if self.surface_hash != expected:
            raise ValueError("Factor target surface hash is invalid")
        return self


@dataclass(frozen=True, slots=True)
class FactorTargetSurface:
    """Carry the immutable target table with its manifest and reconciled quality evidence.

    Attributes:
        table: Formation/listing targets, execution schedule, simple returns, and source-row hashes.
        manifest: Content-addressed lineage and target axes.
        quality: Valid and typed-missing population evidence.
    """

    table: pa.Table
    manifest: FactorTargetSurfaceManifest
    quality: FactorTargetQualityReport


def build_factor_target_policy() -> FactorTargetPolicy:
    """Seal the installed one-session target policy using its canonical defaults.

    Returns:
        Validated official-open, dividend-aware log/simple target policy.
    """
    return seal_contract(FactorTargetPolicy, "policy_hash")


def build_factor_target_policy_for_method(
    binding: ExecutionOutcomeMethodBinding,
) -> FactorTargetPolicy:
    """The projection policy for one resolved outcome method, derived not spelled.

    The one-session builder above stays byte-identical for every frozen caller.
    This variant exists for a development clock: its horizon and exit timing come
    from the *resolved method binding*, so the declaration the compile
    cross-checks against the seal was itself produced from installed authority
    rather than typed beside it -- two routes to the same facts, which is what
    makes the mismatch check able to catch a policy about the wrong method.

    Args:
        binding: Resolved installed outcome method carrying its entry and exit timing.

    Returns:
        Legacy one-session policy for its installed recipe, otherwise a sealed policy
        whose horizon is the difference between exit and entry offsets.

    Raises:
        ValueError: The derived timing or horizon cannot satisfy the installed policy contract.
    """
    if binding.recipe_id == build_one_session_recipe().recipe_id:
        return build_factor_target_policy()
    return seal_contract(
        FactorTargetPolicy,
        "policy_hash",
        prediction_horizon_sessions=binding.exit_offset_sessions - binding.entry_offset_sessions,
        entry_timing=binding.entry_timing,
        exit_timing=binding.exit_timing,
    )


def _as_float64(column: pa.ChunkedArray) -> FloatArray:
    values = np.asarray(column.combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64)
    return cast(FloatArray, values)


def _table_content_hash(table: pa.Table) -> str:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table.combine_chunks())
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _ordered_source(source: pa.Table) -> pa.Table:
    missing = sorted(_REQUIRED_COLUMNS - set(source.schema.names))
    if missing:
        raise FactorTargetBoundaryError(
            f"factor_research.target_required_column_missing:{','.join(missing)}"
        )
    if source.num_rows == 0:
        raise FactorTargetBoundaryError("factor_research.target_surface_empty")
    indices = pc.sort_indices(
        source,
        sort_keys=[("formation_session", "ascending"), ("listing_id", "ascending")],
    )
    ordered = source.take(indices).combine_chunks()
    if ordered.num_rows > 1:
        same_session = pc.equal(
            ordered["formation_session"].slice(1),
            ordered["formation_session"].slice(0, ordered.num_rows - 1),
        )
        same_listing = pc.equal(
            ordered["listing_id"].slice(1),
            ordered["listing_id"].slice(0, ordered.num_rows - 1),
        )
        if bool(pc.any(pc.and_(same_session, same_listing)).as_py()):
            raise FactorTargetBoundaryError("factor_research.target_duplicate_row")
    return ordered


def _validate_schedule(ordered: pa.Table, *, expected_span: int) -> None:
    schedule = ordered.group_by("formation_session", use_threads=False).aggregate(
        [
            ("entry_session", "min"),
            ("entry_session", "max"),
            ("holding_end_session", "min"),
            ("holding_end_session", "max"),
            ("actual_session_span", "min"),
            ("actual_session_span", "max"),
        ]
    )
    for name in ("entry_session", "holding_end_session", "actual_session_span"):
        if bool(pc.any(pc.not_equal(schedule[f"{name}_min"], schedule[f"{name}_max"])).as_py()):
            raise FactorTargetBoundaryError("factor_research.target_schedule_axis_mismatch")
    spans = np.asarray(schedule["actual_session_span_min"].to_numpy(), dtype=np.int64)
    if not bool(np.all(spans == expected_span)):
        raise FactorTargetBoundaryError("factor_research.target_horizon_mismatch")


def _resolve_outcome_method(
    *,
    outcome_method: ExecutionOutcomeMethodSeal | None,
    source_manifest: CausalExecutionOutcomeManifest | DevelopmentOnlyExecutionOutcomeManifest,
    source_manifest_ref: str,
    policy: FactorTargetPolicy,
) -> _ResolvedOutcomeMethod:
    """Decide which method this surface is being compiled under, and check it.

    The method-aware route takes a *resolved seal* -- the terminal marker, the
    outcome marker and the exact binding together -- never a bare binding. A
    binding on its own is a well-formed file; only the terminal marker chain
    says a publication actually carries it. The whole graph is re-verified here
    against the manifest and ref this compiler was handed and against the
    installed catalog and publication policy, so a seal resolved for another
    snapshot, or never resolved by an authoritative reader at all, is refused
    before any numerical work.
    """
    if outcome_method is None:
        resolved = LEGACY_ONE_SESSION_TARGET_METHOD
    else:
        if (
            not isinstance(outcome_method, ExecutionOutcomeMethodSeal)
            or outcome_method.disposition != "METHOD_BOUND"
            or outcome_method.seal_marker is None
            or outcome_method.outcome_marker is None
            or outcome_method.binding is None
        ):
            # A bare binding, a legacy disposition, or anything else that is
            # not a fully method-bound seal has no method authority to assert.
            raise FactorTargetBoundaryError("factor_research.target_outcome_method_unbound")
        # Re-parsed from serialized fields rather than trusted: ``model_validate``
        # on a same-class instance re-runs only the after-validators.
        seal_marker = ExecutionOutcomeMethodSealMarker.model_validate(
            outcome_method.seal_marker.model_dump(mode="json")
        )
        # Re-parsed under the marker's own contract. The development-only
        # successor publishes its own marker kind, and forcing it through the
        # frozen class would refuse every honest five-session seal at the
        # kind-literal rather than at anything about its authority.
        marker_payload = outcome_method.outcome_marker.model_dump(mode="json")
        outcome_marker: CausalExecutionOutcomeMarker | DevelopmentOnlyExecutionOutcomeMarker
        if marker_payload.get("kind") == "DevelopmentOnlyExecutionOutcomeMarker":
            outcome_marker = DevelopmentOnlyExecutionOutcomeMarker.model_validate(marker_payload)
        else:
            outcome_marker = CausalExecutionOutcomeMarker.model_validate(marker_payload)
        binding = ExecutionOutcomeMethodBinding.model_validate(
            outcome_method.binding.model_dump(mode="json")
        )
        catalog = build_installed_execution_outcome_method_catalog()
        try:
            recipe = verify_execution_outcome_method_seal_marker(
                seal_marker=seal_marker,
                outcome_marker=outcome_marker,
                outcome_marker_ref=seal_marker.outcome_marker_ref,
                manifest=source_manifest,
                manifest_ref=source_manifest_ref,
                binding=binding,
                binding_ref=seal_marker.binding_ref,
                catalog=catalog,
                expected_catalog_hash=catalog.binding.catalog_hash,
                policies=build_installed_execution_outcome_publication_policy_catalog(),
            )
        except ExecutionOutcomeMethodError as error:
            raise FactorTargetBoundaryError(
                "factor_research.target_outcome_method_unbound"
            ) from error
        resolved = _ResolvedOutcomeMethod(
            expected_span=recipe.actual_session_span,
            entry_timing=recipe.entry_timing,
            exit_timing=recipe.exit_timing,
            price_basis=recipe.price_basis,
            corporate_action_identity=recipe.corporate_action_identity,
            sealed=True,
        )
    if (
        policy.entry_timing != resolved.entry_timing
        or policy.exit_timing != resolved.exit_timing
        or policy.source_price_basis != resolved.price_basis
        or policy.source_action_semantics != resolved.corporate_action_identity
    ):
        raise FactorTargetBoundaryError("factor_research.target_outcome_method_mismatch")
    return resolved


def _executable(statuses: pa.ChunkedArray) -> npt.NDArray[np.bool_]:
    """Whether each row's status names an eligible execution; a null status does not."""
    eligible = pc.is_in(
        statuses, value_set=pa.array(sorted(_ELIGIBLE_EXECUTION_STATUSES), type=statuses.type)
    )
    return cast(
        npt.NDArray[np.bool_],
        np.asarray(pc.fill_null(eligible, False).to_numpy(zero_copy_only=False), dtype=np.bool_),
    )


def _missing_reason_counts(
    *,
    entry_executable: npt.NDArray[np.bool_],
    exit_executable: npt.NDArray[np.bool_],
    valid: npt.NDArray[np.bool_],
) -> tuple[FactorTargetMissingReasonCount, ...]:
    missing = ~valid
    entry_missing = missing & ~entry_executable
    exit_missing = missing & entry_executable & ~exit_executable
    counts = {
        "ENTRY_NOT_EXECUTABLE": int(entry_missing.sum()),
        "EXIT_NOT_EXECUTABLE": int(exit_missing.sum()),
        "SOURCE_RETURN_MISSING": int((missing & entry_executable & exit_executable).sum()),
    }
    return tuple(
        FactorTargetMissingReasonCount.model_validate({"reason": reason, "row_count": count})
        for reason, count in sorted(counts.items())
        if count
    )


def compile_factor_target_surface(
    *,
    source_table: pa.Table,
    source_manifest: CausalExecutionOutcomeManifest | DevelopmentOnlyExecutionOutcomeManifest,
    source_manifest_ref: str,
    policy: FactorTargetPolicy,
    outcome_method: ExecutionOutcomeMethodSeal | None = None,
) -> FactorTargetSurface:
    """Validate and project one immutable 1D log-return research surface.

    ``outcome_method`` is a resolved method seal -- terminal marker, outcome
    marker and exact binding -- never a bare binding and never a recipe: a
    caller that could describe a method would be asserting the very thing this
    check exists to verify. ``None`` selects
    ``LEGACY_ONE_SESSION_TARGET_METHOD``, the compatibility route for snapshots
    published before the seam existed.

    Args:
        source_table: Complete formation/listing outcomes with execution schedule and return fields.
        source_manifest: Validated causal or development-only outcome manifest.
        source_manifest_ref: Provenance reference retained by the projected manifest.
        policy: Qualified target semantics and execution timing.
        outcome_method: Installed method seal; None admits only the legacy causal route.

    Returns:
        Ordered targets using log1p of reproduced simple returns, with missing values
        typed by entry, exit, or source condition and counts reconciled in the quality report.

    Raises:
        FactorTargetBoundaryError: Method authority, source basis/actions, schedule,
            prices, reproduced returns, positive gross returns, or complete axes fail validation.
        ValueError: A supplied manifest or policy fails its immutable contract.
    """
    # Re-validated under the manifest's own contract, decided by its declared
    # kind rather than by trying one class and falling back: a payload that
    # matches neither must fail as itself, not as whichever guess came last.
    manifest_payload = source_manifest.model_dump(mode="json")
    if manifest_payload.get("kind") == "DevelopmentOnlyExecutionOutcomeSnapshot":
        source_manifest = DevelopmentOnlyExecutionOutcomeManifest.model_validate(manifest_payload)
        if outcome_method is None:
            # There is no legacy era for the successor contract: it postdates the
            # method seam, so an unsealed development-only snapshot was written
            # by something that had no authority to write it.
            raise FactorTargetBoundaryError("factor_research.target_outcome_method_unbound")
    else:
        source_manifest = CausalExecutionOutcomeManifest.model_validate(manifest_payload)
    policy = FactorTargetPolicy.model_validate(policy)
    if source_manifest.price_basis != policy.source_price_basis:
        raise FactorTargetBoundaryError("factor_research.target_price_basis_mismatch")
    if source_manifest.corporate_action_identity != policy.source_action_semantics:
        raise FactorTargetBoundaryError("factor_research.target_action_semantics_mismatch")
    method = _resolve_outcome_method(
        outcome_method=outcome_method,
        source_manifest=source_manifest,
        source_manifest_ref=source_manifest_ref,
        policy=policy,
    )
    ordered = _ordered_source(source_table)
    _validate_schedule(ordered, expected_span=method.expected_span)

    simple = _as_float64(ordered["simple_return"])
    entry = _as_float64(ordered["entry_open_split_adjusted"])
    exit_open = _as_float64(ordered["holding_end_open_split_adjusted"])
    dividend = _as_float64(ordered["period_dividend_split_adjusted"])
    valid = np.isfinite(simple)
    if bool(np.any(valid & (~np.isfinite(entry) | (entry <= 0.0)))):
        raise FactorTargetBoundaryError("factor_research.target_entry_price_invalid")
    if bool(np.any(valid & (~np.isfinite(exit_open) | (exit_open <= 0.0)))):
        raise FactorTargetBoundaryError("factor_research.target_exit_price_invalid")
    if bool(np.any(valid & (~np.isfinite(dividend) | (dividend < 0.0)))):
        raise FactorTargetBoundaryError("factor_research.target_dividend_invalid")
    recomputed: FloatArray = np.full(simple.shape, np.nan, dtype=np.float64)
    recomputed[valid] = (exit_open[valid] + dividend[valid]) / entry[valid] - 1.0
    if not np.array_equal(recomputed[valid].view(np.uint64), simple[valid].view(np.uint64)):
        raise FactorTargetBoundaryError("factor_research.target_formula_mismatch")
    gross = simple[valid] + 1.0
    if bool(np.any(~np.isfinite(gross) | (gross <= 0.0))):
        raise FactorTargetBoundaryError("factor_research.target_gross_return_nonpositive")
    log_target: FloatArray = np.full(simple.shape, np.nan, dtype=np.float64)
    log_target[valid] = np.log1p(simple[valid])
    if bool(np.any(~np.isfinite(log_target[valid]))):
        raise FactorTargetBoundaryError("factor_research.target_log_return_nonfinite")

    missing_reasons = _missing_reason_counts(
        entry_executable=_executable(ordered["entry_status"]),
        exit_executable=_executable(ordered["holding_end_status"]),
        valid=valid,
    )
    columns = {
        "formation_session": ordered["formation_session"],
        "formation_close_at": ordered["formation_close_at"],
        "entry_session": ordered["entry_session"],
        "entry_open_at": ordered["entry_open_at"],
        "holding_end_session": ordered["holding_end_session"],
        "holding_end_open_at": ordered["holding_end_open_at"],
        "listing_id": ordered["listing_id"],
        "fit_target": pa.array(log_target, mask=~valid, type=pa.float64()),
        "simple_economic_return": pa.array(simple, mask=~valid, type=pa.float64()),
        "source_row_hash": ordered["row_hash"],
    }
    target_table = pa.table(columns).combine_chunks()
    formations = tuple(sorted(pc.unique(target_table["formation_session"]).to_pylist()))
    listings = tuple(
        sorted(str(value) for value in pc.unique(target_table["listing_id"]).to_pylist())
    )
    if listings != source_manifest.listing_ids:
        raise FactorTargetBoundaryError("factor_research.target_listing_axis_mismatch")
    if target_table.num_rows != len(formations) * len(listings):
        raise FactorTargetBoundaryError("factor_research.target_surface_incomplete")
    finite_target = log_target[valid]
    quality = seal_contract(
        FactorTargetQualityReport,
        "quality_hash",
        source_outcome_snapshot_hash=source_manifest.snapshot_hash,
        policy_hash=policy.policy_hash,
        row_count=target_table.num_rows,
        formation_count=len(formations),
        listing_count=len(listings),
        valid_target_count=int(valid.sum()),
        typed_missing_count=int((~valid).sum()),
        missing_reasons=missing_reasons,
        minimum_log_return=(float(np.min(finite_target)) if finite_target.size else None),
        maximum_log_return=(float(np.max(finite_target)) if finite_target.size else None),
        maximum_absolute_log_return=(
            float(np.max(np.abs(finite_target))) if finite_target.size else None
        ),
        formula_mismatch_count=0,
        duplicate_key_count=0,
        nonpositive_gross_return_count=0,
    )
    manifest = seal_contract(
        FactorTargetSurfaceManifest,
        "surface_hash",
        source_outcome_snapshot_hash=source_manifest.snapshot_hash,
        source_outcome_manifest_ref=source_manifest_ref,
        policy_hash=policy.policy_hash,
        quality_hash=quality.quality_hash,
        first_formation_session=formations[0],
        last_formation_session=formations[-1],
        formation_count=len(formations),
        listing_count=len(listings),
        row_count=target_table.num_rows,
        table_content_hash=_table_content_hash(target_table),
    )
    return FactorTargetSurface(table=target_table, manifest=manifest, quality=quality)


__all__ = [
    "LEGACY_ONE_SESSION_TARGET_METHOD",
    "FactorTargetBoundaryError",
    "FactorTargetMissingReasonCount",
    "FactorTargetPolicy",
    "FactorTargetQualityReport",
    "FactorTargetSurface",
    "FactorTargetSurfaceManifest",
    "build_factor_target_policy",
    "build_factor_target_policy_for_method",
    "compile_factor_target_surface",
]
