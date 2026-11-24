"""Knowledge's seed and log table contract."""

from alphalattice.foundation.data_layer.root import seed_rows


def assert_seed_history(_named: dict[str, set[str]]) -> None:
    """The three Knowledge event definitions join the seed at count 123."""
    assert [len(seed_rows(historical_count=n)) for n in (122, 123)] == [120, 123]


def assert_knowledge_tables(tables: set[str]) -> None:
    """The new log contains the Knowledge source, model and snapshot tables."""
    assert tables >= {
        "knowledge_source",
        "knowledge_source_version",
        "knowledge_workspace_source",
        "knowledge_recipe",
        "knowledge_model",
        "knowledge_chunk",
        "knowledge_embedding",
        "knowledge_embedding_batch",
        "knowledge_snapshot",
        "knowledge_snapshot_revision",
    }
