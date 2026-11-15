-- index.sqlite: the search index, the one background projection. It can be deleted and rebuilt from the log and the
-- evidence corpus; it is backed up because rebuilding re-embeds every chunk. Needs FTS5 and sqlite-vec.
-- An embedding is made once per machine for a chunk: its key is the chunk's digest, the chunking recipe and the model.

-- A searchable passage: a conclusion, a lesson, a hypothesis, a decision's reasons, or a chunk of an evidence document.
CREATE TABLE passage (
  passage_id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL CHECK (kind IN ('conclusion', 'lesson', 'hypothesis', 'decision', 'review', 'evidence')),
  source_kind TEXT NOT NULL CHECK (source_kind IN ('event', 'doc')),
  source_ref TEXT NOT NULL,         -- an event_id, or a doc_digest
  chunk_digest TEXT NOT NULL,       -- blake3 of the passage text
  chunk_recipe TEXT NOT NULL,       -- how the source was cut into passages
  workspace_id TEXT,                -- where it was made; every workspace's agents may read it
  text TEXT NOT NULL,
  made_seq INTEGER NOT NULL,        -- the log position when it was indexed
  UNIQUE (source_ref, chunk_recipe, chunk_digest)
) STRICT;
CREATE INDEX passage_by_source ON passage(source_kind, source_ref);  -- forgetting a record removes its passages

CREATE VIRTUAL TABLE passage_fts USING fts5(text, content='passage', content_rowid='passage_id');

-- One vector table per embedding model, named by the model's definition; a new model is a new table.
CREATE VIRTUAL TABLE passage_vec USING vec0(passage_id INTEGER PRIMARY KEY, embedding FLOAT[384]);

-- embedding_model | dims | cursor_seq (how far into the log the index has read)
CREATE TABLE index_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
) STRICT, WITHOUT ROWID;
