"""Descriptive comparison of two exact saved Alpha development candidates."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from typing import Any

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import AuthoringError

from .development_evidence import AlphaDevelopmentVerifiedGraph
from .score_rows import read_experiment_score_support


def compare_saved_alpha_candidates(
    *,
    left: Mapping[str, Any],
    left_candidate_id: str,
    left_graph: AlphaDevelopmentVerifiedGraph | None,
    right: Mapping[str, Any],
    right_candidate_id: str,
    right_graph: AlphaDevelopmentVerifiedGraph | None,
) -> dict[str, object]:
    """Report selected stored evidence without selecting a model or recomputing metrics.

    Each side is a completed Task's readback body and the graph that readback
    verified in this same request (``AlphaDevelopmentReceiptReader.read``):
    the receipt, its children and the score rows the walk proved. The
    comparison consumes that verification -- it walks nothing again and reads
    nothing the walk did not prove -- and a side whose Task did not publish
    an Alpha graph carries ``None`` and is reported incomplete.
    """

    left_state = _incomplete_subject(left, left_candidate_id)
    right_state = _incomplete_subject(right, right_candidate_id)
    if left_state is not None or right_state is not None:
        return {
            "status": "INCOMPLETE",
            "disposition": "DESCRIPTIVE_ALPHA_COMPARISON_NO_SELECTION",
            "left": left_state or _requested_subject(left, left_candidate_id),
            "right": right_state or _requested_subject(right, right_candidate_id),
            "claim": (
                "An incomplete selected candidate is visible but has no invented comparison "
                "metrics."
            ),
            "limitations": [
                "COMPLETE_RECEIPT_AND_SCORE_SUPPORT_REQUIRED",
                "NO_COMMON_INTERSECTION_OR_ZERO_FILL",
                "NO_STATISTICAL_SIGNIFICANCE_OR_SELECTION_CLAIM",
            ],
        }

    if left_graph is None or right_graph is None:
        raise AuthoringError("alpha_research.saved_comparison_receipt_unavailable")
    left_subject = _verified_subject(left, left_candidate_id, left_graph)
    right_subject = _verified_subject(right, right_candidate_id, right_graph)
    relation = _relation(left_subject, right_subject)
    folds = _pair_folds(left_subject, right_subject)
    support = _assert_score_support(left_subject, right_subject)
    left_features, right_features = (
        left_subject["ordered_feature_ids"],
        right_subject["ordered_feature_ids"],
    )
    return {
        "status": "COMPARABLE",
        "disposition": "DESCRIPTIVE_ALPHA_COMPARISON_NO_SELECTION",
        "left": _public_subject(left_subject),
        "right": _public_subject(right_subject),
        "relation": relation,
        "declared_parameter_difference": {
            "left": left_subject["model_parameters"],
            "right": right_subject["model_parameters"],
        },
        "declared_feature_difference": {
            "added": [v for v in right_features if v not in left_features],
            "removed": [v for v in left_features if v not in right_features],
        },
        "prerequisites": {
            "same_input_binding": left_subject["input_binding_hash"],
            "same_target_recipe_binding": left_subject["target_recipe_binding_hash"],
            **(
                {
                    "same_target_materialization": left_subject[
                        "target_materialization_binding_hash"
                    ],
                    "same_feature_axis": left_features,
                }
                if relation == "PARAMETERS"
                else {"same_target_values": left_subject["target_values"]}
            ),
            "same_metric_policy": left_subject["metric_policy_hash"],
            "same_split_policy": left_subject["split_policy_hash"],
            "paired_fold_commitments": [item["fold_commitment_hash"] for item in folds],
        },
        "score_support": support,
        "folds": folds,
        "claim": (
            "Descriptive stored Alpha evidence only; no significance test, winner, refit, "
            "publication, independent validation, or strategy selection."
        ),
        "limitations": [
            "POST_OBSERVED_DEVELOPMENT_NOT_INDEPENDENT_VALIDATION",
            "NO_COMMON_INTERSECTION_OR_ZERO_FILL",
            "NO_STATISTICAL_SIGNIFICANCE_OR_SELECTION_CLAIM",
        ],
        "next_requests": {
            "reopen": {
                "operation": "EXPERIMENT_ALPHA_COMPARE",
                "left_task_id": left_subject["task_id"],
                "left_candidate_id": left_subject["candidate_id"],
                "right_task_id": right_subject["task_id"],
                "right_candidate_id": right_subject["candidate_id"],
            }
        },
    }


def _incomplete_subject(body: Mapping[str, Any], candidate_id: str) -> dict[str, object] | None:
    if body.get("status") != "EXPERIMENT_PUBLISHED":
        return {
            **_requested_subject(body, candidate_id),
            "status": body.get("status", "UNAVAILABLE"),
            "failure_code": body.get("failure_code"),
        }
    result = body.get("result")
    candidates = result.get("candidates") if isinstance(result, Mapping) else None
    if candidates is None:
        # The comparison operation deliberately receives a minimal completed-Task
        # projection, then verifies the receipt itself.  A full Alpha readback
        # would verify that same graph once here and once again below.
        return None
    if not isinstance(candidates, list):
        raise AuthoringError("alpha_research.saved_comparison_candidate_unavailable")
    candidate = next(
        (
            item
            for item in candidates
            if isinstance(item, Mapping) and item.get("candidate_id") == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise AuthoringError("alpha_research.saved_comparison_candidate_unavailable")
    if candidate.get("status") != "DEVELOPMENT_EVALUATED":
        return {
            **_requested_subject(body, candidate_id),
            "status": candidate.get("status", "INCOMPLETE"),
            "candidate": _candidate_summary(candidate),
            "failure_codes": candidate.get("failure_codes", []),
        }
    return None


def _requested_subject(body: Mapping[str, Any], candidate_id: str) -> dict[str, object]:
    program = body.get("program")
    return {
        "task_id": body.get("task_id"),
        "candidate_id": candidate_id,
        "program_hash": program.get("program_hash") if isinstance(program, Mapping) else None,
    }


def _verified_subject(
    body: Mapping[str, Any], candidate_id: str, graph: AlphaDevelopmentVerifiedGraph
) -> dict[str, Any]:
    receipt_body = body.get("receipt")
    if not isinstance(receipt_body, Mapping) or not isinstance(
        receipt_body.get("receipt_hash"), str
    ):
        raise AuthoringError("alpha_research.saved_comparison_receipt_unavailable")
    receipt, batch, folds = graph.receipt, graph.batch, graph.folds
    if receipt.receipt_hash != receipt_body["receipt_hash"]:
        raise AuthoringError("alpha_research.saved_comparison_receipt_mismatch")
    candidate = next((item for item in batch.candidates if item.candidate_id == candidate_id), None)
    if candidate is None:
        raise AuthoringError("alpha_research.saved_comparison_candidate_unavailable")
    if candidate.status != "DEVELOPMENT_EVALUATED":
        raise AuthoringError("alpha_research.saved_comparison_candidate_incomplete")
    selected_folds = tuple(item for item in folds if item.candidate_id == candidate_id)
    if len(selected_folds) != len(receipt.fold_surface_hashes):
        raise AuthoringError("alpha_research.saved_comparison_fold_set_incomplete")
    document = body.get("document")
    alpha = document.get("alpha") if isinstance(document, Mapping) else None
    program = body.get("program")
    if not isinstance(alpha, Mapping) or not isinstance(program, Mapping):
        raise AuthoringError("alpha_research.saved_comparison_not_alpha")
    support = read_experiment_score_support(graph, candidate_id)
    return {
        "task_id": body.get("task_id"),
        "candidate_id": candidate_id,
        "program_hash": program.get("program_hash"),
        "receipt_hash": receipt.receipt_hash,
        "development_program_hash": receipt.development_program_hash,
        "candidate": candidate.model_dump(mode="json"),
        "model_parameters": alpha.get("model_parameters"),
        "input_binding_hash": body.get("input_binding_hash"),
        "target_recipe_binding_hash": receipt.target_recipe_binding.binding_hash,
        "target_materialization_binding_hash": receipt.target_materialization_binding.binding_hash,
        "target_values": _target_values(receipt.target_materialization_binding),
        "ordered_feature_ids": list(receipt.ordered_base_feature_ids),
        "metric_policy_hash": receipt.metric_policy_hash,
        "split_policy_hash": receipt.split_policy_hash,
        "folds": tuple(selected_folds),
        "score_support": {
            "formation_session_count": len(support.sessions),
            "formation_sessions_hash": canonical_hash(
                [value.isoformat() for value in support.sessions]
            ),
            "listing_count": len(support.listings),
            "ordered_listing_ids_hash": canonical_hash(support.listings),
            "scored_row_count": int(support.available.sum()),
            "common_surface_row_count": int(support.available.size),
            "availability_hash": _availability_hash(support.available),
        },
    }


def _availability_hash(available: Any) -> str:
    """Hash the complete boolean surface without materializing it as JSON rows.

    The dimensions are reported separately in the public support summary, so a
    contiguous uint8 payload has one unambiguous interpretation here.  This
    keeps the exact-support check fail-closed without retaining raw availability
    rows or making a saved-evidence readback proportional to JSON serialization.
    """

    return sha256(available.astype("uint8", copy=False).tobytes(order="C")).hexdigest()


def _target_values(binding: Any) -> dict[str, object]:
    """The target values a materialization sealed, without the feature axis it names."""

    return {
        "standardization_id": binding.standardization_id,
        "fold_record_hashes": [record.record_hash for record in binding.fold_records],
        "training_row_count": binding.training_row_count,
        "transformed_target_value_hash": binding.transformed_target_value_hash,
    }


def _same(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> None:
    if left.get(key) is None or left.get(key) != right.get(key):
        raise AuthoringError(f"alpha_research.saved_comparison_{key}_mismatch")


def _relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    """What two comparable candidates differ in, as this owner defines it.

    ``PARAMETERS``: the declared model parameters alone, over one target
    materialization and one feature axis. ``FEATURE_ADDITION``: one feature axis
    holds every feature of the other and more (a feature trial's without and with),
    over the same target values. The materialization binding names the feature axis
    it was built beside, so it differs there while the values it seals must not.
    Everything else is shared either way.
    """

    _same(left, right, "input_binding_hash")
    _same(left, right, "target_recipe_binding_hash")
    left_features, right_features = left["ordered_feature_ids"], right["ordered_feature_ids"]
    if left_features == right_features:
        _same(left, right, "target_materialization_binding_hash")
        relation = "PARAMETERS"
    else:
        smaller, larger = sorted((set(left_features), set(right_features)), key=len)
        if not smaller < larger:
            raise AuthoringError("alpha_research.saved_comparison_ordered_feature_ids_mismatch")
        _same(left, right, "target_values")
        relation = "FEATURE_ADDITION"
    _same(left, right, "metric_policy_hash")
    _same(left, right, "split_policy_hash")
    return relation


def _pair_folds(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict[str, object]]:
    left_by_commitment = {fold.fold_commitment_hash: fold for fold in left["folds"]}
    right_by_commitment = {fold.fold_commitment_hash: fold for fold in right["folds"]}
    if len(left_by_commitment) != len(left["folds"]) or len(right_by_commitment) != len(
        right["folds"]
    ):
        raise AuthoringError("alpha_research.saved_comparison_fold_commitment_duplicate")
    if set(left_by_commitment) != set(right_by_commitment):
        raise AuthoringError("alpha_research.saved_comparison_fold_commitment_mismatch")
    return [
        {
            "fold_commitment_hash": commitment,
            "left": _fold_summary(left_by_commitment[commitment]),
            "right": _fold_summary(right_by_commitment[commitment]),
        }
        for commitment in sorted(left_by_commitment)
    ]


def _assert_score_support(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, object]:
    left_support, right_support = left["score_support"], right["score_support"]
    if any(
        left_support[key] != right_support[key]
        for key in (
            "formation_sessions_hash",
            "ordered_listing_ids_hash",
            "availability_hash",
        )
    ):
        raise AuthoringError("alpha_research.saved_comparison_score_support_mismatch")
    return {
        key: left_support[key]
        for key in (
            "formation_session_count",
            "formation_sessions_hash",
            "listing_count",
            "ordered_listing_ids_hash",
            "scored_row_count",
            "common_surface_row_count",
            "availability_hash",
        )
    }


def _public_subject(subject: Mapping[str, Any]) -> dict[str, object]:
    return {
        key: subject[key]
        for key in (
            "task_id",
            "candidate_id",
            "program_hash",
            "receipt_hash",
            "development_program_hash",
            "candidate",
        )
    }


def _candidate_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: candidate.get(key)
        for key in (
            "candidate_id",
            "result_hash",
            "status",
            "pooled_oos_r2",
            "mean_rank_ic",
            "mean_gross_decile_spread",
            "fold_coverage_mean",
            "failure_codes",
        )
    }


def _fold_summary(fold: Any) -> dict[str, object]:
    score = fold.score_chunk
    return {
        "candidate_id": fold.candidate_id,
        "fold_index": fold.fold_index,
        "numerical_result_hash": fold.numerical_result_hash,
        "status": fold.status.value,
        "role": fold.role.value,
        "metrics": None if fold.metrics is None else fold.metrics.model_dump(mode="json"),
        "score": None
        if score is None
        else {
            "content_hash": score.content_hash,
            "metadata_hash": score.metadata_hash,
            "row_count": score.row_count,
        },
        "failure": None if fold.failure is None else fold.failure.model_dump(mode="json"),
    }
