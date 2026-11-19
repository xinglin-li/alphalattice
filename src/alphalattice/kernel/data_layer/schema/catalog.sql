-- catalog.sqlite: the DuckLake catalog. The Host attaches it once and keeps it open:
--   ATTACH 'ducklake:sqlite:<data root>/machine/catalog.sqlite' AS lake
--     (DATA_PATH '<data root>/data/', DATA_INLINING_ROW_LIMIT 1000);
--   USE lake;
--
-- The write order, and the one repair (trade-off 8). An admission mints its admission_id (a UUIDv7) first; rows
-- carry it. Then: (1) Parquet files, (2) one catalog commit that carries the event it owes in its commit info
--   CALL lake.set_commit_message('host', 'admission', extra_info => '<the owed event's whole envelope, as JSON>');
-- The envelope keeps its type version, workspace, actor, trace and occurrence time; repair appends it unchanged except
-- for its recording time and the snapshot it names.
-- (3) the log event `admission.committed`, naming the snapshot. A crash leaves at most one gap, and start-up closes
-- it from the catalog alone: a snapshot whose commit info names an event the log lacks gets that event; files no
-- snapshot names are collected. Each snapshot's commit_extra_info is in the catalog's ducklake_snapshot_changes.
--
-- DuckLake enforces no keys: each table's key is stated here and kept by the Host.
-- Idle maintenance, bounded and cancellable: ducklake_flush_inlined_data, ducklake_merge_adjacent_files (small files
-- only), ducklake_expire_snapshots and ducklake_cleanup_old_files (older than 7 days).
CALL lake.set_option('parquet_compression', 'zstd');
CREATE SCHEMA IF NOT EXISTS world;
CREATE SCHEMA IF NOT EXISTS derived;
CREATE SCHEMA IF NOT EXISTS results;
-- Results and derived rows are never inlined: each read and key check scans an inlined table's every row, and an
-- inlined commit costs more per table, while a file per write is pruned by its artifact_id or input_digest range.
-- World's small daily batches stay inlined (the ATTACH's 1000).
CALL lake.set_option('data_inlining_row_limit', 0, schema => 'results');
CALL lake.set_option('data_inlining_row_limit', 0, schema => 'derived');

-- ===== world: what the outside world said, and when we knew it =====
-- Append-only and bitemporal. Every table has its valid-time key and these five columns:
--   version_id     the version's own id, a UUIDv7 the engine mints at its admission. A staged row is a new
--                  version only when its content (its key, its values and retracted, conformed to these
--                  types; a column it leaves out is null) differs from its key's newest version's: a
--                  correction is a new version of the fact, and the old row stays
--   known_at       when we observed it: the admission's own moment. Empty on a row migrated from 0.2.7, whose
--                  knowledge time nobody recorded; nothing is invented here (rule 17)
--   known_asof     the knowledge time a point-in-time read uses, `known_asof <= cutoff`, on every row alike. It
--                  equals known_at where that is known. For a legacy row it is derived: the provider's publication
--                  time, or the valid date plus the source's registered lag rule, counted from the close_utc in
--                  world.session of its market's first session on or after that date, so half days, time zones and
--                  a date between sessions hold. The row names the rule and its version (lag_rule, lag_rule_version,
--                  on the tables a legacy batch fills). Being derived, a legacy row's value depends on that version:
--                  an admission under another version than its key's held rows is refused by name until an upgrade
--                  step re-derives them, so an earlier, wrong value cannot stay readable; an observed row's never
--                  changes
--   retracted      true on a version that withdraws the fact: the provider deleted the bar, or the action never
--                  happened. A read whose newest version is retracted sees no row
--   admission_id   the admission that brought it; the log's `admission` view
-- raw_bar and corporate_action also carry source_observed_at: the source's own last write of the value, on the
-- source's clock (0.2.7's observed_at, and for a bar the later of it and its adjusted close's updated_at). It is not
-- content, so it never decides whether a row is new, and it moves no knowledge time. It orders two sources' values for one key:
-- a held key is superseded only by a row its source wrote after the held value was last written; an older row, or one
-- written at the same time with another value, is dropped, which keeps the held version, and the admission's event
-- counts each by its reason. A later write of the held value itself adds no version: world.rewritten records it, and
-- the value's last write is the later of its version's own source_observed_at and those. A table whose rows carry no
-- source time (listing, provider_symbol) keeps the newest admission's value.
-- A legacy row of a key world already holds is a change observed at its admission: its known_asof is never before
-- that admission, nor before the version it replaces.
-- Rule-generated legacy sessions are reference data: calendar_origin and calendar_version retain their rule
-- engine; their initial known_asof is -infinity and known_at is empty. A recorded source_observed_at wins over
-- that reference origin. A later recorded correction is known from its source observation or admission time.
-- A new source is a definition (family 'source') that names its table here; it brings these columns and no migration.
-- A new world table (fundamentals, exchange rates) comes with its first source and the same five columns.
-- A legacy interval row whose end is set (listing's last_session; valid_to below) is imported as two versions:
-- the open one, known_asof from its start plus the lag, and the closing one, known_asof from its end plus the lag,
-- so a cutoff between them sees the interval open and never its future end.

-- key: (market_id, version_id). The Host mints the market's id once. MIC is its outside name,
-- never its identity. Versions retain every name and its knowledge interval through superseded;
-- valid_from records a source's MIC change date, null when that date was never recorded.
CREATE TABLE world.market (
  market_id UUID NOT NULL,
  version_id UUID NOT NULL,
  mic VARCHAR NOT NULL,
  valid_from DATE,
  known_at TIMESTAMPTZ,
  known_asof TIMESTAMPTZ NOT NULL,
  retracted BOOLEAN NOT NULL,
  admission_id VARCHAR NOT NULL
);

-- key: (listing_id, version_id)
CREATE TABLE world.listing (
  listing_id UUID NOT NULL,       -- our stable id; a symbol is an alias that changes
  version_id UUID NOT NULL,
  market_id UUID NOT NULL,       -- world.market; a MIC change never changes this link
  universe_class VARCHAR,         -- the class 0.2.7 kept as the listing's mic, such as US_INDEX_CANDIDATE
  currency VARCHAR,
  issuer_name VARCHAR,
  first_session DATE,
  last_session DATE,
  known_at TIMESTAMPTZ,
  known_asof TIMESTAMPTZ NOT NULL,
  lag_rule VARCHAR,               -- a legacy row's lag rule, by name
  lag_rule_version INTEGER,       -- and its version, which derived known_asof
  retracted BOOLEAN NOT NULL,
  admission_id VARCHAR NOT NULL
);

-- No key holds a name: a provider, a sector scheme, a universe and a world table are each a lineage in the log's
-- definition registry, named here by its id; its name is the lineage's, which a rename moves and these rows keep.
-- key: (provider_id, symbol, valid_from, version_id). A row is one alias: a changed symbol is a new row, by design.
CREATE TABLE world.provider_symbol (
  provider_id VARCHAR NOT NULL,    -- the provider's lineage
  symbol VARCHAR NOT NULL,
  valid_from DATE NOT NULL,
  version_id UUID NOT NULL,
  valid_to DATE,
  lifecycle_state VARCHAR,        -- the source's state; internal aliases leave it empty
  listing_id UUID NOT NULL,
  known_at TIMESTAMPTZ,
  known_asof TIMESTAMPTZ NOT NULL,
  lag_rule VARCHAR,               -- a legacy row's lag rule, by name
  lag_rule_version INTEGER,       -- and its version, which derived known_asof
  retracted BOOLEAN NOT NULL,
  admission_id VARCHAR NOT NULL
);

-- key: (universe_id, listing_id, valid_from, version_id). Membership by interval: today's members never stand in for the past.
CREATE TABLE world.universe_member (
  universe_id VARCHAR NOT NULL,    -- the universe's lineage
  listing_id UUID NOT NULL,
  valid_from DATE NOT NULL,
  version_id UUID NOT NULL,
  valid_to DATE,
  known_at TIMESTAMPTZ,
  known_asof TIMESTAMPTZ NOT NULL,
  lag_rule VARCHAR,               -- a legacy row's lag rule, by name
  lag_rule_version INTEGER,       -- and its version, which derived known_asof
  retracted BOOLEAN NOT NULL,
  admission_id VARCHAR NOT NULL
);

-- key: (scheme_id, listing_id, valid_from, version_id)
CREATE TABLE world.sector (
  scheme_id VARCHAR NOT NULL,      -- the sector scheme's lineage
  listing_id UUID NOT NULL,
  valid_from DATE NOT NULL,
  version_id UUID NOT NULL,
  valid_to DATE,
  sector_code VARCHAR NOT NULL,
  known_at TIMESTAMPTZ,
  known_asof TIMESTAMPTZ NOT NULL,
  lag_rule VARCHAR,               -- a legacy row's lag rule, by name
  lag_rule_version INTEGER,       -- and its version, which derived known_asof
  retracted BOOLEAN NOT NULL,
  admission_id VARCHAR NOT NULL
);

-- key: (provider_id, listing_id, session_date, version_id). The provider's raw daily bar; adjusted prices are
-- computed, and adjusted_close keeps the provider's own adjusted close beside them, as 0.2.7 kept it to check them.
CREATE TABLE world.raw_bar (
  provider_id VARCHAR NOT NULL,
  listing_id UUID NOT NULL,
  session_date DATE NOT NULL,
  version_id UUID NOT NULL,
  open DOUBLE,
  high DOUBLE,
  low DOUBLE,
  close DOUBLE,
  volume DOUBLE,
  adjusted_close DOUBLE,          -- the provider's adjusted close, when it gives one
  published_at TIMESTAMPTZ,       -- the provider's own publication time, when it gives one
  source_observed_at TIMESTAMPTZ, -- the source's own last write of the value (above)
  source_payload_digest VARCHAR, -- the 0.2.7 payload_hash; source metadata, never part of a version's identity
  known_at TIMESTAMPTZ,
  known_asof TIMESTAMPTZ NOT NULL,
  lag_rule VARCHAR,               -- a legacy row's lag rule, by name
  lag_rule_version INTEGER,       -- and its version, which derived known_asof
  retracted BOOLEAN NOT NULL,
  admission_id VARCHAR NOT NULL
);
ALTER TABLE world.raw_bar SET PARTITIONED BY (year(session_date));
-- Written in key order, listing then date, in row groups of 10,240 rows: a read of one listing's bars
-- touches only the groups that hold them.
CALL lake.set_option('parquet_row_group_size', 10240, schema => 'world', table_name => 'raw_bar');

-- key: (provider_id, listing_id, effective_date, kind, action_id, version_id). An announced action is known from
-- its announcement; an adjustment as of a date applies only actions effective by then. kind is a closed vocabulary;
-- action_id is the provider's own identifier, kept as it gave it.
CREATE TABLE world.corporate_action (
  provider_id VARCHAR NOT NULL,
  listing_id UUID NOT NULL,
  effective_date DATE NOT NULL,
  kind VARCHAR NOT NULL,          -- split | dividend | spinoff | delisting
  action_id VARCHAR NOT NULL,     -- the provider's id, or the digest of its defining fields: two same-day dividends stay two
  version_id UUID NOT NULL,
  ratio DOUBLE,
  amount DOUBLE,                  -- delisting: the cash per share paid at delisting, 0 when worthless
  currency VARCHAR,
  pay_date DATE,                  -- a cash distribution's payment date, when given
  provisional BOOLEAN NOT NULL,   -- announced or estimated, not yet final
  provenance VARCHAR NOT NULL,    -- how it was learned, such as provider-observation
  published_at TIMESTAMPTZ,
  source_observed_at TIMESTAMPTZ, -- the source's own last write of the value (above)
  source_payload_digest VARCHAR, -- the 0.2.7 payload_hash; source metadata, never part of a version's identity
  known_at TIMESTAMPTZ,
  known_asof TIMESTAMPTZ NOT NULL,
  lag_rule VARCHAR,               -- a legacy row's lag rule, by name
  lag_rule_version INTEGER,       -- and its version, which derived known_asof
  retracted BOOLEAN NOT NULL,
  admission_id VARCHAR NOT NULL
);

-- key: (market_id, session_date, version_id). Holidays, half days and time zones are versioned inputs.
CREATE TABLE world.session (
  market_id UUID NOT NULL,
  session_date DATE NOT NULL,
  version_id UUID NOT NULL,
  open_utc TIMESTAMPTZ,
  close_utc TIMESTAMPTZ,
  half_day BOOLEAN NOT NULL,
  calendar_origin VARCHAR,       -- the rule engine, when this is generated reference data
  calendar_version VARCHAR,      -- that engine's pinned version, retained with this version
  source_observed_at TIMESTAMPTZ, -- a recorded source clock takes precedence over reference origin
  known_at TIMESTAMPTZ,
  known_asof TIMESTAMPTZ NOT NULL,
  retracted BOOLEAN NOT NULL,
  admission_id VARCHAR NOT NULL
);

-- key: (doc_id, version_id). An outside evidence document (a filing; later, news): doc_id is the Host's UUIDv7 for one
-- source document, its source's own id (`source_ref`: an SEC accession and its primary document) held as a column. Its
-- body is under blobs/, addressed by doc_digest, the BLAKE3 of the body's bytes: unique among the documents, so one body
-- acquired twice is one document, and never the key. A new version of a doc_id is the same source document acquired
-- again with other bytes, or withdrawn (retracted). An amending filing is its own document, naming the one it amends.
-- Its times are 0.2.7's, each kept apart: publication at its precision (DATE or INSTANT), the source's acceptance, and
-- its availability, what reads are bounded by: the acceptance itself, or a recorded import's capture, later
-- (`availability_basis` says which). A legacy document is known from its availability, never its filing date.
CREATE TABLE world.evidence_doc (
  doc_id UUID NOT NULL,
  version_id UUID NOT NULL,
  provider_id VARCHAR NOT NULL,    -- the source (SEC_EDGAR, ISSUER_RECORDED), as a provider lineage
  source_ref VARCHAR NOT NULL,     -- the source's own id of the document
  doc_digest VARCHAR NOT NULL,     -- the body's BLAKE3, 64 hex digits: its blob's file name
  byte_count BIGINT NOT NULL,
  media_type VARCHAR NOT NULL,
  title VARCHAR NOT NULL,
  document_type VARCHAR NOT NULL,  -- the source's form (10-K, 8-K/A) or a recorded import's kind
  entity_id VARCHAR NOT NULL,      -- the issuer as 0.2.7's evidence names it
  published_at TIMESTAMPTZ,
  published_precision VARCHAR,     -- DATE | INSTANT; null when never recorded
  accepted_at TIMESTAMPTZ,         -- the source's acceptance (EDGAR acceptanceDateTime)
  available_at TIMESTAMPTZ NOT NULL,
  availability_basis VARCHAR NOT NULL, -- OFFICIAL_ACCEPTANCE | RECORDED_IMPORT_CAPTURE[_ACCEPTANCE_UNKNOWN] | SOURCE_STATED
  report_period_end DATE,
  right_of_use VARCHAR NOT NULL,   -- SEC_PUBLIC_OFFICIAL_ACCESS | USER_PROVIDED_FOR_LOCAL_RESEARCH
  amends_doc_id UUID,              -- the document an amending filing amends
  retrieved_at TIMESTAMPTZ,        -- when 0.2.7 obtained the bytes: never a disclosure time
  known_at TIMESTAMPTZ,
  known_asof TIMESTAMPTZ NOT NULL,
  lag_rule VARCHAR,
  lag_rule_version INTEGER,
  retracted BOOLEAN NOT NULL,
  admission_id VARCHAR NOT NULL
);

-- key: (table_id, version_id). When a newer version of a key becomes known, the Host appends one
-- row for the version it replaced, naming the moment its replacement became known, in the same catalog commit as the
-- new row. Point-in-time reads anti-join it instead of ranking every version on read: Step M on mac1, 12.6 million bars
-- with 126,000 corrections, read the newest cutoff in 1.7 s this way against 3.3 s by a window (0.7 s against 1.7 s at
-- half the history). A walk-forward read takes superseded_asof as the version's visible_to. Versions of one key are
-- admitted in their known_asof order; a legacy row's derived known_asof and the superseded_asof its replacement gives
-- move together when the lag rule's version moves.
CREATE TABLE world.superseded (
  table_id VARCHAR NOT NULL,             -- the world table's lineage (family world_table), such as raw_bar's
  version_id UUID NOT NULL,              -- the replaced version
  superseded_asof TIMESTAMPTZ NOT NULL,  -- when its replacement became known
  by_admission_id VARCHAR NOT NULL       -- the admission that replaced it
);

-- key: (table_id, version_id, source_observed_at)
-- A later source write of a version's own value, in raw_bar and corporate_action: no new version, since the value is the
-- same, but its time orders what may supersede it (the source clock above). Append-only; no read of the world sees it.
CREATE TABLE world.rewritten (
  table_id VARCHAR NOT NULL,             -- the world table's lineage, as in world.superseded
  version_id UUID NOT NULL,              -- the written version
  source_observed_at TIMESTAMPTZ NOT NULL,  -- the source's later write of the value, on its clock
  by_admission_id VARCHAR NOT NULL       -- the admission that saw the write
);

-- ===== derived: a cache, empty by default =====
-- Computed on read unless a method measures over its threshold. A cached partition is valid for one input_digest:
-- the digest of the exact world rows and upstream partitions it read. The log's cache_entry says which are valid;
-- these rows are not authoritative and may be dropped at any time.

-- One feature set is one wide table, a column per factor, NULL for missing. Created from its definition
-- (family 'feature_set'), which adds a column when it adds a factor, and named by its lineage's id, never its name,
-- so a renamed feature set keeps every cached partition (store.feature_table: feature_set_<the id's hex>):
--   CREATE TABLE derived.feature_set_<hex> (
--     recipe_key VARCHAR NOT NULL, input_digest VARCHAR NOT NULL,
--     listing_id UUID NOT NULL, session_date DATE NOT NULL, <factor> DOUBLE, ...);
--   ALTER TABLE derived.feature_set_<hex> SET PARTITIONED BY (year(session_date));
--   key: (recipe_key, input_digest, session_date, listing_id), written in that order
-- StorePort.derive adds each column its rows bring, typed as they type it: the factors, and the lineage a build
-- stamps on each row (raw_input_hash, row_hash, ...). Its cache partition is a year and listing: rows of one under
-- several digests sit side by side, a workspace roots the digests it reads (log cache_root) and reads them under its
-- own table's name (workspace_runtime.derived_views), and a digest no workspace roots is collected.

-- key: (recipe_key, input_digest, row_no). x is a list: DuckLake has no fixed-size array; it reads back to NumPy zero-copy.
CREATE TABLE derived.training_sample (
  recipe_key VARCHAR NOT NULL,
  input_digest VARCHAR NOT NULL,
  row_no BIGINT NOT NULL,
  session_date DATE NOT NULL,
  listing_id UUID NOT NULL,
  x FLOAT[] NOT NULL,
  y DOUBLE
);

-- key: (recipe_key, input_digest, as_of, factor_i, factor_j)
CREATE TABLE derived.factor_cov (
  recipe_key VARCHAR NOT NULL,
  input_digest VARCHAR NOT NULL,
  as_of DATE NOT NULL,
  factor_i VARCHAR NOT NULL,
  factor_j VARCHAR NOT NULL,
  value DOUBLE NOT NULL
);

-- key: (recipe_key, input_digest, as_of, listing_id, factor)
CREATE TABLE derived.factor_loading (
  recipe_key VARCHAR NOT NULL,
  input_digest VARCHAR NOT NULL,
  as_of DATE NOT NULL,
  listing_id UUID NOT NULL,
  factor VARCHAR NOT NULL,
  value DOUBLE NOT NULL
);

-- key: (recipe_key, input_digest, as_of, listing_id)
CREATE TABLE derived.specific_risk (
  recipe_key VARCHAR NOT NULL,
  input_digest VARCHAR NOT NULL,
  as_of DATE NOT NULL,
  listing_id UUID NOT NULL,
  value DOUBLE NOT NULL
);

-- ===== results: table-shaped artifact bodies =====
-- Keyed by the log's artifact_id. These are bodies: collection may delete an artifact's rows, and its profile and
-- measures in the log stay.

-- key: (artifact_id, session_date, listing_id)
CREATE TABLE results.alpha_score (
  artifact_id VARCHAR NOT NULL,
  session_date DATE NOT NULL,
  listing_id UUID NOT NULL,
  score DOUBLE NOT NULL
);

-- key: (artifact_id, as_of, listing_id). Target holdings; a deployed book's actual positions are the log's ledger.
CREATE TABLE results.holding (
  artifact_id VARCHAR NOT NULL,
  as_of DATE NOT NULL,
  listing_id UUID NOT NULL,
  weight DOUBLE NOT NULL,
  shares DOUBLE
);

-- key: (artifact_id, name, session_date). A time series a study produced, such as a book's net value.
CREATE TABLE results.series (
  artifact_id VARCHAR NOT NULL,
  name VARCHAR NOT NULL,
  session_date DATE NOT NULL,
  value DOUBLE NOT NULL
);

-- key: (artifact_id, listing_id, formation_session). One execution outcome: a formation close, the actual entry and
-- holding-end opens after it, their statuses, and the return between them; a price that was missing leaves its value
-- and the return null. The hashes name the source rows and the row itself, as 0.2.7 wrote them.
CREATE TABLE results.execution_outcome (
  artifact_id VARCHAR NOT NULL,
  listing_id VARCHAR NOT NULL,
  symbol VARCHAR NOT NULL,
  sequence INTEGER NOT NULL,
  formation_session DATE NOT NULL,
  formation_close_at TIMESTAMPTZ NOT NULL,
  entry_session DATE NOT NULL,
  entry_open_at TIMESTAMPTZ NOT NULL,
  holding_end_session DATE NOT NULL,
  holding_end_open_at TIMESTAMPTZ NOT NULL,
  actual_session_span SMALLINT NOT NULL,
  entry_status VARCHAR NOT NULL,
  holding_end_status VARCHAR NOT NULL,
  entry_open_split_adjusted DOUBLE,
  holding_end_open_split_adjusted DOUBLE,
  period_dividend_split_adjusted DOUBLE,
  simple_return DOUBLE,
  entry_source_row_hash VARCHAR NOT NULL,
  holding_end_source_row_hash VARCHAR NOT NULL,
  row_hash VARCHAR NOT NULL
);

-- key: (artifact_id, metric_id, metric_version, dims). The fine breakdown behind a measure, such as IC by date and
-- factor: too many values for a profile, so it is a body. The summary stays in the log's `measure`.
CREATE TABLE results.breakdown (
  artifact_id VARCHAR NOT NULL,
  metric_id VARCHAR NOT NULL,      -- the metric's lineage
  metric_version INTEGER NOT NULL,
  dims VARCHAR NOT NULL,           -- the measure's form: canonical JSON of typed dimensions (project.dims)
  value DOUBLE
);

-- key: (artifact_id). One metric-free protected-validation closure, preserving its sealed identity.
CREATE TABLE results.validation_receipt (
  artifact_id VARCHAR NOT NULL,
  permit_hash VARCHAR NOT NULL,
  candidate_hash VARCHAR NOT NULL,
  package_hash VARCHAR NOT NULL,
  fixture_hash VARCHAR NOT NULL,
  pre_protected_state_hash VARCHAR NOT NULL,
  closure VARCHAR NOT NULL,
  claim_limits VARCHAR[] NOT NULL,
  reselection_disposition VARCHAR NOT NULL,
  sealed_at TIMESTAMPTZ NOT NULL,
  sealed_offset_microseconds BIGINT NOT NULL,
  receipt_hash VARCHAR NOT NULL
);
