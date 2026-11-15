"""The 0.2.8 data layer: one event log, its registry, the DuckLake catalog DDL, the index."""

from alphalattice.kernel.data_layer.root import (
    SCHEMA_VERSION,
    DataRoot,
    create_data_root,
    schema_text,
    seed_rows,
)

__all__ = ["SCHEMA_VERSION", "DataRoot", "create_data_root", "schema_text", "seed_rows"]
