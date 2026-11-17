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
-- snapshot names are collected. Checked on DuckDB 1.5.5: lake.snapshots() returns commit_extra_info.
--
-- DuckLake enforces no keys: each table's key is stated here and kept by the Host.
-- Idle maintenance, bounded and cancellable: ducklake_flush_inlined_data, ducklake_merge_adjacent_files (small files
-- only), ducklake_expire_snapshots and ducklake_cleanup_old_files (older than 7 days).
CALL lake.set_option('parquet_compression', 'zstd');
CREATE SCHEMA IF NOT EXISTS world;
CREATE SCHEMA IF NOT EXISTS derived;
CREATE SCHEMA IF NOT EXISTS results;

-- ===== world: what the outside world said, and when we knew it =====
-- Append-only and bitemporal. Every table has its valid-time key and these five columns:
--   value_digest  blake3 content digest of the canonical row; a correction is a new version of the fact,
--                 identified by its digest, and the old row stays
--   known_at       when we observed it: the admission's own moment. Empty on a row migrated from 0.2.7, whose
--                  knowledge time nobody recorded; nothing is invented here (rule 17)
--   known_asof     the knowledge time a point-in-time read uses, `known_asof <= cutoff`, on every row alike. It
--                  equals known_at where that is known. For a legacy row it is derived: the provider's publication
--                  time, or the valid date plus the source's registered lag rule, counted from the close_utc in
--                  world.session of its market's first session on or after that date, so half days, time zones and
--                  a date between sessions hold. The admission names the rule's version. Being derived, a legacy row's
--                  value is rewritten when that rule changes version, so an earlier, wrong value cannot stay
--                  readable; an observed row's never changes
--   retracted      true on a version that withdraws the fact: the provider deleted the bar, or the action never
--                  happened. A read whose newest version is retracted sees no row
--   admission_id   the admission that brought it; the log's `admission` view
-- raw_bar and corporate_action also carry source_observed_at: the source's own last write of the value, on the
-- source's clock (0.2.7's observed_at, and for a bar the later of it and its adjusted close's updated_at). It is not
-- content, so it stays out of value_digest, and it moves no knowledge time. It orders two sources' values for one key:
-- a held key is superseded only by a row its source wrote after the held value was last written; an older row, or one
-- written at the same time with another value, is dropped, which keeps the held version, and the admission's event
-- counts each by its reason. A later write of the held value itself adds no version: world.rewritten records it, and
-- the value's last write is the later of its version's own source_observed_at and those. A table whose rows carry no
-- source time (listing, provider_symbol) keeps the newest admission's value.
-- A legacy row of a key world already holds is a change observed at its admission: its known_asof is never before
-- that admission, nor before the version it replaces.
-- A new source is a definition (family 'source') that names its table here; it brings these columns and no migration.
-- A new world table (fundamentals, exchange rates) comes with its first source and the same five columns.
-- A legacy interval row whose end is set (listing's last_session; valid_to below) is imported as two versions:
-- the open one, known_asof from its start plus the lag, and the closing one, known_asof from its end plus the lag,
-- so a cutoff between them sees the interval open and never its future end.

-- key: (listing_id, value_digest)
CREATE TABLE world.listing (
  listing_id UUID NOT NULL,       -- our stable id; a symbol is an alias that changes
  value_digest VARCHAR NOT NULL,
  market VARCHAR NOT NULL,       -- the ISO 10383 MIC of its market's calendar
  universe_class VARCHAR,         -- the class 0.2.7 kept as the listing's mic, such as US_INDEX_CANDIDATE
  currency VARCHAR,
  issuer_name VARCHAR,
  first_session DATE,
  last_session DATE,
  known_at TIMESTAMPTZ,
  known_asof TIMESTAMPTZ NOT NULL,
  retracted BOOLEAN NOT NULL,
  admission_id VARCHAR NOT NULL
);

-- No key holds a name: a provider, a sector scheme, a universe and a world table are each a lineage in the log's
-- definition registry, named here by its id; its name is the lineage's, which a rename moves and these rows keep.
-- key: (provider_id, symbol, valid_from, value_digest). A row is one alias: a changed symbol is a new row, by design.
CREATE TABLE world.provider_symbol (
  provider_id VARCHAR NOT NULL,    -- the provider's lineage
  symbol VARCHAR NOT NULL,
  valid_from DATE NOT NULL,
  value_digest VARCHAR NOT NULL,
  valid_to DATE,
  listing_id UUID NOT NULL,
  known_at TIMESTAMPTZ,
  known_asof TIMESTAMPTZ NOT NULL,
  retracted BOOLEAN NOT NULL,
  admission_id VARCHAR NOT NULL
);

-- key: (universe_id, listing_id, valid_from, value_digest). Membership by interval: today's members never stand in for the past.
CREATE TABLE world.universe_member (
  universe_id VARCHAR NOT NULL,    -- the universe's lineage
  listing_id UUID NOT NULL,
  valid_from DATE NOT NULL,
  value_digest VARCHAR NOT NULL,
  valid_to DATE,
  known_at TIMESTAMPTZ,
  known_asof TIMESTAMPTZ NOT NULL,
  retracted BOOLEAN NOT NULL,
  admission_id VARCHAR NOT NULL
);

-- key: (scheme_id, listing_id, valid_from, value_digest)
CREATE TABLE world.sector (
  scheme_id VARCHAR NOT NULL,      -- the sector scheme's lineage
  listing_id UUID NOT NULL,
  valid_from DATE NOT NULL,
  value_digest VARCHAR NOT NULL,
  valid_to DATE,
  sector_code VARCHAR NOT NULL,
  known_at TIMESTAMPTZ,
  known_asof TIMESTAMPTZ NOT NULL,
  retracted BOOLEAN NOT NULL,
  admission_id VARCHAR NOT NULL
);

-- key: (provider_id, listing_id, session_date, value_digest). The provider's raw daily bar; adjusted prices are
-- computed, and adjusted_close keeps the provider's own adjusted close beside them, as 0.2.7 kept it to check them.
CREATE TABLE world.raw_bar (
  provider_id VARCHAR NOT NULL,
  listing_id UUID NOT NULL,
  session_date DATE NOT NULL,
  value_digest VARCHAR NOT NULL,
  open DOUBLE,
  high DOUBLE,
  low DOUBLE,
  close DOUBLE,
  volume DOUBLE,
  adjusted_close DOUBLE,          -- the provider's adjusted close, when it gives one
  published_at TIMESTAMPTZ,       -- the provider's own publication time, when it gives one
  source_observed_at TIMESTAMPTZ, -- the source's own last write of the value (above)
  known_at TIMESTAMPTZ,
  known_asof TIMESTAMPTZ NOT NULL,
  retracted BOOLEAN NOT NULL,
  admission_id VARCHAR NOT NULL
);
ALTER TABLE world.raw_bar SET PARTITIONED BY (year(session_date));

-- key: (provider_id, listing_id, effective_date, kind, action_id, value_digest). An announced action is known from
-- its announcement; an adjustment as of a date applies only actions effective by then. kind is a closed vocabulary;
-- action_id is the provider's own identifier, kept as it gave it.
CREATE TABLE world.corporate_action (
  provider_id VARCHAR NOT NULL,
  listing_id UUID NOT NULL,
  effective_date DATE NOT NULL,
  kind VARCHAR NOT NULL,          -- split | dividend | spinoff | delisting
  action_id VARCHAR NOT NULL,     -- the provider's id, or the digest of its defining fields: two same-day dividends stay two
  value_digest VARCHAR NOT NULL,
  ratio DOUBLE,
  amount DOUBLE,                  -- delisting: the cash per share paid at delisting, 0 when worthless
  currency VARCHAR,
  pay_date DATE,                  -- a cash distribution's payment date, when given
  provisional BOOLEAN NOT NULL,   -- announced or estimated, not yet final
  provenance VARCHAR NOT NULL,    -- how it was learned, such as provider-observation
  published_at TIMESTAMPTZ,
  source_observed_at TIMESTAMPTZ, -- the source's own last write of the value (above)
  known_at TIMESTAMPTZ,
  known_asof TIMESTAMPTZ NOT NULL,
  retracted BOOLEAN NOT NULL,
  admission_id VARCHAR NOT NULL
);

-- key: (market, session_date, value_digest). Holidays, half days and time zones are versioned inputs. market is the
-- ISO 10383 MIC, a standard code designed as an identifier, so it stays a key.
CREATE TABLE world.session (
  market VARCHAR NOT NULL,
  session_date DATE NOT NULL,
  value_digest VARCHAR NOT NULL,
  open_utc TIMESTAMPTZ,
  close_utc TIMESTAMPTZ,
  half_day BOOLEAN NOT NULL,
  known_at TIMESTAMPTZ,
  known_asof TIMESTAMPTZ NOT NULL,
  retracted BOOLEAN NOT NULL,
  admission_id VARCHAR NOT NULL
);

-- key: (doc_digest, value_digest). An outside evidence document (a filing; later, news); its body is under blobs/,
-- addressed by doc_digest. A withdrawal is a new version with retracted, as in every world table. The listings a
-- document names are the index's entities. A legacy document without published_at is known from the time 0.2.7
-- acquired it, never earlier.
CREATE TABLE world.evidence_doc (
  doc_digest VARCHAR NOT NULL,
  value_digest VARCHAR NOT NULL,
  source VARCHAR NOT NULL,
  title VARCHAR,
  published_at TIMESTAMPTZ,
  known_at TIMESTAMPTZ,
  known_asof TIMESTAMPTZ NOT NULL,
  retracted BOOLEAN NOT NULL,
  admission_id VARCHAR NOT NULL
);

-- key: (table_id, value_digest, admission_id). When a newer version of a key becomes known, the Host appends one
-- row for the version it replaced, naming the moment its replacement became known, in the same catalog commit as the
-- new row. Point-in-time reads anti-join it instead of ranking every version on read: Step M on mac1, 12.6 million bars
-- with 126,000 corrections, read the newest cutoff in 1.7 s this way against 3.3 s by a window (0.7 s against 1.7 s at
-- half the history). A walk-forward read takes superseded_asof as the version's visible_to. Versions of one key are
-- admitted in their known_asof order; a legacy row's derived known_asof and the superseded_asof its replacement gives
-- move together when the lag rule's version moves.
CREATE TABLE world.superseded (
  table_id VARCHAR NOT NULL,             -- the world table's lineage (family world_table), such as raw_bar's
  value_digest VARCHAR NOT NULL,        -- the replaced row's digest
  admission_id VARCHAR NOT NULL,         -- the replaced row's admission: one value may return in a later admission
  superseded_asof TIMESTAMPTZ NOT NULL,  -- when its replacement became known
  by_admission_id VARCHAR NOT NULL       -- the admission that replaced it
);

-- A later source write of a version's own value, in raw_bar and corporate_action: no new version, since the value is the
-- same, but its time orders what may supersede it (the source clock above). Append-only; no read of the world sees it.
CREATE TABLE world.rewritten (
  table_id VARCHAR NOT NULL,             -- the world table's lineage, as in world.superseded
  value_digest VARCHAR NOT NULL,         -- the written version's digest
  admission_id VARCHAR NOT NULL,         -- the written version's admission
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
--   key: (recipe_key, input_digest, listing_id, session_date)

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
  dims VARCHAR NOT NULL,
  value DOUBLE
);
