"""Catalog-backed native factor, loading and specific-risk surfaces."""

from __future__ import annotations

import json
from contextlib import closing
from dataclasses import replace
from datetime import date
from typing import Any

import numpy as np
from numpy.typing import NDArray

from alphalattice.foundation.market_data_ops.storage.world import world_listing
from alphalattice.kernel.data_layer import history, project, risk_history
from alphalattice.kernel.data_layer.log import attach, uuid7
from alphalattice.kernel.data_layer.root import data_root

from .artifacts import risk_store_mode
from .decomposition import FactorIdiosyncraticRiskSurface, RiskDecompositionError, _frozen

_FACTOR_PROFILE = "risk_factor_surface"
_PROFILE = """
    SELECT shape FROM artifact
    WHERE content_digest = ? AND type_id = (
        SELECT definition_id FROM definition_name WHERE family = 'artifact_type' AND name = ?
    )
"""
_FACTOR_ROWS = """
    WITH coordinates AS (SELECT ? AS recipe_key, ? AS input_digest, ?::DATE AS as_of),
    factors AS (
        SELECT * FROM unnest(?::VARCHAR[]) WITH ORDINALITY AS f(factor, pos)
    ), listings AS (
        SELECT * FROM unnest(?::UUID[]) WITH ORDINALITY AS l(listing_id, pos)
    )
    SELECT 1 AS block, i.pos AS row_index, j.pos AS col_index, t.value
    FROM lake.derived.factor_cov t JOIN coordinates USING (recipe_key, input_digest, as_of)
    JOIN factors i ON t.factor_i = i.factor JOIN factors j ON t.factor_j = j.factor
    UNION ALL
    SELECT 2, i.pos, j.pos, t.value
    FROM lake.derived.factor_loading t JOIN coordinates USING (recipe_key, input_digest, as_of)
    JOIN listings i USING (listing_id) JOIN factors j USING (factor)
    UNION ALL
    SELECT 3, i.pos, 1, t.value
    FROM lake.derived.specific_risk t JOIN coordinates USING (recipe_key, input_digest, as_of)
    JOIN listings i USING (listing_id)
    ORDER BY block, row_index, col_index
"""


def _vector(values: Any) -> NDArray[np.float64]:
    return _frozen(np.asarray(values, dtype=np.float64))


def read_factor_surface(input_digest: str) -> FactorIdiosyncraticRiskSurface | None:
    """Read the previously produced model at the exact causal input identity, if held."""
    log = history.holding()
    args = (input_digest, _FACTOR_PROFILE)
    if log is not None:
        profiles = log.read(_PROFILE, args)
    elif (data_root() / "machine" / "catalog.sqlite").is_file():
        with closing(attach(data_root())) as connection:
            profiles = connection.execute(_PROFILE, args).fetchall()
    else:
        return None
    if not profiles:
        return None
    if len(profiles) != 1:
        raise RiskDecompositionError("risk_research.decomposition_content_hash_mismatch")
    profile = json.loads(profiles[0][0])
    empty = _vector([])
    surface = FactorIdiosyncraticRiskSurface(
        recipe_hash=profile["recipe_hash"],
        formation_session=date.fromisoformat(profile["session"]),
        ordered_listing_ids=tuple(profile["listings"]),
        ordered_factor_ids=tuple(profile["factors"]),
        exposures=empty,
        factor_covariance=empty,
        idiosyncratic_variance=empty,
        conditional_volatility=_vector(profile["conditional_volatility"]),
        rescaling=_vector(profile["rescaling"]),
        producer_identity=input_digest,
        surface_hash=profile["surface_hash"],
    )
    return _project(surface, profile["factor_lineages"])


def _project(
    surface: FactorIdiosyncraticRiskSurface, lineages: list[str]
) -> FactorIdiosyncraticRiskSurface:
    listings = [world_listing(item) for item in surface.ordered_listing_ids]
    with history.catalog_session() as cursor:
        if cursor is None:
            raise RiskDecompositionError("data_layer.history_unavailable")
        table = cursor.sql(
            _FACTOR_ROWS,
            params=(
                surface.recipe_hash,
                surface.producer_identity,
                surface.formation_session,
                lineages,
                listings,
            ),
        ).to_arrow_table()
    n, k = len(listings), len(lineages)
    cov_end, loading_end = k * k, k * (k + n)
    shapes = ((k, k), (n, k), (n, 1))
    if (
        k != len(surface.ordered_factor_ids)
        or len(table) != loading_end + n
        or not np.array_equal(
            table["block"].to_numpy(),
            np.repeat(np.arange(1, len(shapes) + 1), [rows * cols for rows, cols in shapes]),
        )
        or not np.array_equal(
            table["row_index"].to_numpy(),
            np.concatenate([np.repeat(np.arange(1, rows + 1), cols) for rows, cols in shapes]),
        )
        or not np.array_equal(
            table["col_index"].to_numpy(),
            np.concatenate([np.tile(np.arange(1, cols + 1), rows) for rows, cols in shapes]),
        )
    ):
        raise RiskDecompositionError("risk_research.decomposition_shape_invalid")
    values = _vector(table["value"].to_numpy())
    result = replace(
        surface,
        factor_covariance=values[:cov_end].reshape(k, k),
        exposures=values[cov_end:loading_end].reshape(n, k),
        idiosyncratic_variance=values[loading_end:],
    )
    result.verify_content()
    return result


def admit_factor_surface(surface: FactorIdiosyncraticRiskSurface) -> FactorIdiosyncraticRiskSurface:
    """Root all three native factor blocks before publishing the model's sealed profile."""
    import pyarrow as pa  # lazy: construct Arrow rows only at catalog admission

    from alphalattice.kernel.data_layer.store import Derivation  # lazy: catalog writes only

    port = history.store()
    opened = history.opened_log()
    if port is None or opened is None:
        raise RiskDecompositionError("data_layer.history_unavailable")
    surface.verify_content()
    log, workspace = opened
    listings = [world_listing(item) for item in surface.ordered_listing_ids]
    risk_history.require_risk_listings(listings)
    names = [f"risk/sector/{item}" for item in surface.ordered_factor_ids]
    for name in names:
        history.define("factor", name, "A sector factor of the installed Risk decomposition.")
    with log.reading() as con:
        lineages = [project.lineage(con, "factor", name) for name in names]
    if any(value is None for value in lineages):
        raise RiskDecompositionError("data_layer.lineage_unregistered")
    factor_ids = [str(item) for item in lineages]

    parts = pa.table(
        {
            "part": [surface.formation_session.isoformat()],
            "input_digest": [surface.producer_identity],
        }
    )
    derivations: list[Derivation] = []
    n, k = len(listings), len(factor_ids)
    for target, axis, values in (
        (
            "factor_cov",
            {"factor_i": np.repeat(factor_ids, k), "factor_j": np.tile(factor_ids, k)},
            surface.factor_covariance,
        ),
        (
            "factor_loading",
            {"listing_id": np.repeat(listings, k), "factor": np.tile(factor_ids, n)},
            surface.exposures,
        ),
        ("specific_risk", {"listing_id": listings}, surface.idiosyncratic_variance),
    ):
        count = values.size
        rows = pa.table(
            {
                "recipe_key": [surface.recipe_hash] * count,
                "input_digest": [surface.producer_identity] * count,
                "as_of": pa.array([surface.formation_session] * count, type=pa.date32()),
                **axis,
                "value": values.ravel(),
            }
        )
        derivations.append(Derivation(target, rows, surface.recipe_hash, parts, compute_ms=0))
    written = port.derive_set(derivations, rooted_by=workspace)
    # A retry can reuse held rows, so verify those coordinates before publishing.
    if written != [len(item.rows) for item in derivations]:
        surface = _project(surface, factor_ids)
    history.define(
        "artifact_type",
        _FACTOR_PROFILE,
        "A sealed native Risk factor model.",
        {"storage": "none", "measures": []},
    )
    history.record_new(
        _PROFILE,
        (surface.producer_identity, _FACTOR_PROFILE),
        "artifact.produced",
        ("artifact", uuid7()),
        {
            "type": _FACTOR_PROFILE,
            "type_version": history.version("artifact_type", _FACTOR_PROFILE),
            "identity_kind": "content",
            "storage": "none",
            "meaning": "Seal and coordinates of one causal Risk factor model.",
            "content_digest": surface.producer_identity,
            "valid_from": surface.formation_session.isoformat(),
            "valid_to": surface.formation_session.isoformat(),
            "shape": {
                "recipe_hash": surface.recipe_hash,
                "session": surface.formation_session.isoformat(),
                "surface_hash": surface.surface_hash,
                "listings": surface.ordered_listing_ids,
                "factors": surface.ordered_factor_ids,
                "factor_lineages": factor_ids,
                "conditional_volatility": surface.conditional_volatility.tolist(),
                "rescaling": surface.rescaling.tolist(),
            },
        },
        node="host",
    )
    history.flush()
    return surface


class _CatalogFactorSurfaces:
    read = staticmethod(read_factor_surface)
    admit = staticmethod(admit_factor_surface)


_CATALOG_FACTOR_SURFACES = _CatalogFactorSurfaces()


def catalog_factor_surfaces() -> _CatalogFactorSurfaces | None:
    """Bind production to the catalog while a Host records and rollback is off."""
    mode = risk_store_mode()
    if mode == "catalog" or (mode is None and history.recording()):
        return _CATALOG_FACTOR_SURFACES
    return None
