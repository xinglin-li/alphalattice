"""Shared identities for Feature persistence and immutable recovery evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np
import pandas as pd


def align_feature_source_sessions(frame: pd.DataFrame, sessions: Sequence[date]) -> pd.DataFrame:
    """Keep absent sessions in a lookback, never fill their numerical observations."""
    expected = tuple(sessions)
    observed = tuple(pd.to_datetime(frame["session_date"]).dt.date)
    if (
        not expected
        or expected != tuple(sorted(set(expected)))
        or observed != tuple(sorted(set(observed)))
        or set(observed) - set(expected)
    ):
        raise ValueError("feature.source_window_calendar_invalid")
    if observed == expected:
        return frame
    return (
        frame.assign(session_date=observed)
        .set_index("session_date")
        .reindex(expected)
        .rename_axis("session_date")
        .reset_index()
    )


def feature_source_values_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Hash exact numerical arguments, independently of batch/projection metadata."""
    dates = tuple(pd.to_datetime(frame["session_date"]).dt.date)
    if not dates or dates != tuple(sorted(set(dates))):
        raise ValueError("feature.source_window_calendar_invalid")
    numeric = tuple(value for value in columns if value != "session_date")
    values = np.ascontiguousarray(frame.loc[:, list(numeric)].to_numpy(dtype="<f8"))
    missing = np.isnan(values)
    return _canonical_hash(
        {
            "columns": numeric,
            "sessions": dates,
            "values": hashlib.sha256(np.where(missing, 0.0, values).tobytes()).hexdigest(),
            "missing": hashlib.sha256(missing.tobytes()).hexdigest(),
        }
    )


def feature_materialization_coverage(
    *,
    rows: int,
    factor_count: int,
    rebuilt_factor_count: int,
    ineligible_count: int,
    source_window: Mapping[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "rows": rows,
        "factor_count": factor_count,
        "rebuilt_factor_count": rebuilt_factor_count,
        "ineligible_count": ineligible_count,
    }
    if source_window is not None:
        result["source_window"] = dict(source_window)
    return result


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_materialization_receipt_hash(
    *,
    listing_id: str,
    range_start: date,
    range_end: date,
    catalog_hash: str,
    raw_input_hash: str,
    action_set_hash: str,
    market_reference_revision: str,
    idempotency_key: str,
    coverage: Mapping[str, object],
) -> str:
    """Return the receipt identity shared by Store and closure coordinator."""
    return _canonical_hash(
        {
            "listing_id": listing_id,
            "range_start": range_start.isoformat(),
            "range_end": range_end.isoformat(),
            "catalog_hash": catalog_hash,
            "raw_input_hash": raw_input_hash,
            "action_set_hash": action_set_hash,
            "market_reference_revision": market_reference_revision,
            "idempotency_key": idempotency_key,
            "coverage": dict(coverage),
        }
    )


def feature_row_content_hash(
    *,
    listing_id: str,
    session_date: date,
    catalog_hash: str,
    input_cutoffs: Mapping[str, object],
    factor_ids: Sequence[str],
    values: Sequence[object],
) -> str:
    """Hash one complete catalog row using the persisted Feature contract."""
    if len(factor_ids) != len(values):
        raise ValueError("Feature row value axis does not match its factor axis")
    return _canonical_hash(
        {
            "listing_id": listing_id,
            "session_date": session_date.isoformat(),
            "catalog_hash": catalog_hash,
            "input_cutoffs": dict(input_cutoffs),
            "values": dict(zip(factor_ids, values, strict=True)),
        }
    )


_ROW_VALUES_ENCODER = json.JSONEncoder(
    ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
)


def feature_row_content_hash_from_canonical(
    *,
    listing_id: str,
    session_date: date,
    catalog_hash: str,
    canonical_cutoffs: str,
    factor_ids: Sequence[str],
    values: Sequence[object],
) -> str:
    """``feature_row_content_hash`` over a cutoff set already in canonical JSON form.

    Byte-identical to the mapping form: the encoder orders the five keys and
    writes a sorted mapping exactly as the mapping's canonical payload reads,
    so the cutoffs are not encoded a second time. The payload is the ASCII
    spelling ``canonical_cutoff_set`` returns; one that carries an escape
    (a non-ASCII or escaped character) is spelled differently by the row
    encoder and is parsed and hashed as a mapping instead.
    """
    if len(factor_ids) != len(values):
        raise ValueError("Feature row value axis does not match its factor axis")
    if "\\" in canonical_cutoffs or not canonical_cutoffs.isascii():
        return feature_row_content_hash(
            listing_id=listing_id,
            session_date=session_date,
            catalog_hash=catalog_hash,
            input_cutoffs=json.loads(canonical_cutoffs),
            factor_ids=factor_ids,
            values=values,
        )
    payload = (
        "{"
        f'"catalog_hash":{json.dumps(catalog_hash, ensure_ascii=False)},'
        f'"input_cutoffs":{canonical_cutoffs},'
        f'"listing_id":{json.dumps(listing_id, ensure_ascii=False)},'
        f'"session_date":{json.dumps(session_date.isoformat())},'
        f'"values":{_ROW_VALUES_ENCODER.encode(dict(zip(factor_ids, values, strict=True)))}'
        "}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_row_is_numerically_identical(
    *,
    existing_cutoffs: str,
    existing_values: Sequence[object],
    next_cutoffs: Mapping[str, object],
    next_values: Sequence[object],
) -> bool:
    """Ignore representation-scale rolling drift, never an economic delta."""
    try:
        if json.loads(existing_cutoffs) != dict(next_cutoffs):
            return False
    except json.JSONDecodeError:
        return False
    if len(existing_values) != len(next_values):
        return False

    def same_scalar(left: object, right: object) -> bool:
        left_missing = left is None or (isinstance(left, float) and math.isnan(left))
        right_missing = right is None or (isinstance(right, float) and math.isnan(right))
        if left_missing or right_missing:
            return left_missing and right_missing
        try:
            if not isinstance(left, (int, float, str)) or not isinstance(right, (int, float, str)):
                return left == right
            left_number = float(left)
            right_number = float(right)
        except (TypeError, ValueError):
            return left == right
        if left_number == right_number:
            return True
        if not math.isfinite(left_number) or not math.isfinite(right_number):
            return False
        ulp = max(math.ulp(left_number), math.ulp(right_number))
        # The observed bounded/full-prefix drift is at most 32 ULPs.  The
        # doubled guard stays at representation scale and cannot hide an
        # economically meaningful correction.
        return abs(left_number - right_number) <= 64.0 * ulp

    return all(
        same_scalar(left, right) for left, right in zip(existing_values, next_values, strict=True)
    )


__all__ = [
    "feature_materialization_receipt_hash",
    "feature_row_content_hash",
    "feature_row_content_hash_from_canonical",
    "feature_row_is_numerically_identical",
]
