-- log.sqlite: the machine's one event log, and every table projected from it. Only the Host writes.
-- Shape tiers (plan 0.2.8): [1] strict core  [2] versioned definitions  [3] contract content  [4] opaque bodies.
-- Rule: `event` is the only truth. Every other table here is a projection the Host updates in the event's own
-- transaction, and dropping them and replaying the log rebuilds them ALL EQUAL. Two tables are the Host's private
-- working state and are not projections: cache_entry, deletion_queue.
-- Times are integer microseconds since the Unix epoch, UTC (`*_us`). Every `*_seq` is a position in event.seq.
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;  -- FULL for the commits that cannot be obtained again (an answer, a paid call, a download)
PRAGMA foreign_keys = ON;

-- ===== [1] The log =====
-- One row per fact, and one row per finished step: a step is recorded once, at its end, with its start time, as an
-- OpenTelemetry span is. A long step (a task, a trial) may also record `started` so that it is visible while running.
CREATE TABLE event (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,  -- global order, assigned in the transaction
  event_id TEXT NOT NULL UNIQUE,          -- UUIDv7; stable across export
  type TEXT NOT NULL,                     -- definition('event_type', type, type_version), enforced below
  type_version INTEGER NOT NULL,
  phase TEXT,                             -- NULL: a fact. started | completed | failed | cancelled: a step
  occurred_us INTEGER NOT NULL,           -- when it happened; for a finished step, when it ended
  started_us INTEGER,                     -- a finished step's start
  recorded_us INTEGER NOT NULL,           -- when the Host wrote it
  actor_kind TEXT NOT NULL,               -- person | agent | host | node
  actor_id TEXT NOT NULL,
  node TEXT NOT NULL,                     -- the module that emitted it
  workspace_id TEXT,                      -- NULL for a machine-wide event
  subject_kind TEXT,                      -- what it is about; a note's subject is the thing it is about
  subject_id TEXT,
  trace_id TEXT,                          -- W3C, 32 hex; NULL only on a legacy import
  span_id TEXT,                           -- a step's own span, or the span a fact happened in; 16 hex
  parent_span_id TEXT,
  cause_seq INTEGER REFERENCES event(seq),  -- the event that caused this one; it exists, so it precedes
  attrs TEXT NOT NULL DEFAULT '{}',       -- [3] contract content, checked by the Host against the type's schema
  text_digest TEXT,                       -- [4] blake3 of this event's text in event_text; binds it into the chain
  legacy INTEGER NOT NULL DEFAULT 0,      -- imported from 0.2.7; missing fields stay missing
  CHECK (phase IS NULL OR phase IN ('started', 'completed', 'failed', 'cancelled')),
  CHECK (phase IS NULL OR span_id IS NOT NULL),
  CHECK (phase IS NULL OR phase = 'started' OR started_us IS NOT NULL),
  CHECK (started_us IS NULL OR (phase IN ('completed', 'failed', 'cancelled') AND started_us <= occurred_us)),
  CHECK (actor_kind IN ('person', 'agent', 'host', 'node')),
  CHECK (trace_id IS NOT NULL OR legacy = 1),
  CHECK (trace_id IS NULL OR length(trace_id) = 32),
  CHECK (span_id IS NULL OR length(span_id) = 16),
  CHECK (json_valid(attrs) AND length(attrs) <= 4096),
  CHECK (legacy IN (0, 1))
) STRICT;
CREATE INDEX event_trace ON event(trace_id);
CREATE INDEX event_subject ON event(subject_kind, subject_id);
CREATE INDEX event_type_subject ON event(type, subject_id);
CREATE INDEX event_workspace ON event(workspace_id);
CREATE UNIQUE INDEX event_span_end ON event(span_id) WHERE phase IN ('completed', 'failed', 'cancelled');
CREATE UNIQUE INDEX event_span_start ON event(span_id) WHERE phase = 'started';
CREATE TRIGGER event_no_update BEFORE UPDATE ON event BEGIN SELECT RAISE(ABORT, 'event is append-only'); END;
CREATE TRIGGER event_no_delete BEFORE DELETE ON event BEGIN SELECT RAISE(ABORT, 'event is append-only'); END;
-- An event keeps the type name it was written under; a name is never reused in its family, so it names one lineage.
CREATE TRIGGER event_type_registered BEFORE INSERT ON event
WHEN NOT EXISTS (SELECT 1 FROM definition WHERE family = 'event_type' AND name = NEW.type AND version = NEW.type_version)
BEGIN SELECT RAISE(ABORT, 'event type and version are not registered'); END;

-- [4] An event's own text: a hypothesis, a conclusion, a decision's reasons, a lesson, a registered definition.
-- Kept forever and never parsed. A definition's text is the one text a projection reads, so it is never forgotten.
-- Forgetting any other record deletes its row here and writes `record.forgotten`; the event
-- and its digest stay, so the chain stays whole.
-- Attributes are permanent, so nothing private goes in them: the Host's secret scan refuses an event that carries one.
CREATE TABLE event_text (
  seq INTEGER PRIMARY KEY REFERENCES event(seq),
  text TEXT NOT NULL
) STRICT;
CREATE TRIGGER event_text_bound BEFORE INSERT ON event_text
WHEN (SELECT text_digest FROM event WHERE seq = NEW.seq) IS NULL
BEGIN SELECT RAISE(ABORT, 'the event names no text digest'); END;
CREATE TRIGGER event_text_no_update BEFORE UPDATE ON event_text BEGIN SELECT RAISE(ABORT, 'event text is never edited'); END;

-- The hash chain: head(n) = blake3(head(n-1) || canonical bytes of event n), folded in seq order by every commit
-- batch. It covers `event` alone, since every other row derives from events or is bound by a digest in one.
CREATE TABLE chain_head (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  seq INTEGER NOT NULL,            -- the last event folded in
  head TEXT NOT NULL,
  chain_version INTEGER NOT NULL   -- the canonical encoding's version
) STRICT;
-- Heads kept at their moment: monthly, and in every backup and export. A later rewrite of an earlier event
-- disagrees with every head recorded after it.
-- A redaction is the one sanctioned rewrite: a secret that still reached an event's attributes is removed, the chain
-- is folded again from there, and the head it replaced is kept here, so the rewrite stays visible.
CREATE TABLE chain_checkpoint (
  seq INTEGER NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('month', 'backup', 'export', 'redaction')),
  head TEXT NOT NULL,
  at_us INTEGER NOT NULL,
  note TEXT,
  chain_version INTEGER NOT NULL,  -- the encoding the events up to seq were folded with; verify folds each segment by its own
  PRIMARY KEY (seq, kind)
) STRICT, WITHOUT ROWID;
CREATE TRIGGER chain_checkpoint_no_update BEFORE UPDATE ON chain_checkpoint BEGIN SELECT RAISE(ABORT, 'a checkpoint is never edited'); END;
CREATE TRIGGER chain_checkpoint_no_delete BEFORE DELETE ON chain_checkpoint BEGIN SELECT RAISE(ABORT, 'a checkpoint is never deleted'); END;

-- ===== [2] Versioned definitions: one registry, one rule =====
-- Every definition that evolves lives here: event types, artifact types, metrics, methods, feature sets, sources,
-- tag vocabularies, knowledge-lag rules, nodes, providers, sector schemes, universes and world tables. A new family
-- needs no migration. A definition is a lineage with an id assigned once; its name is a column and may be renamed.
-- A version's content is never edited; only its status moves. A version moves only when values move. `spec` is the
-- family's contract content; the typed views below read it. No key holds a name (engineering.md, Keys never change).
CREATE TABLE definition_lineage (
  definition_id TEXT PRIMARY KEY,  -- blake3 of (family, the name it was first registered under); assigned once
  family TEXT NOT NULL,
  name TEXT NOT NULL,              -- its current name: agents and the person read this
  registered_seq INTEGER NOT NULL,
  named_seq INTEGER NOT NULL,      -- the event that gave it its current name
  UNIQUE (family, name)
) STRICT, WITHOUT ROWID;
CREATE TRIGGER definition_lineage_frozen BEFORE UPDATE OF definition_id, family, registered_seq ON definition_lineage
BEGIN SELECT RAISE(ABORT, 'a lineage keeps its id and family'); END;
CREATE TRIGGER definition_lineage_no_delete BEFORE DELETE ON definition_lineage
BEGIN SELECT RAISE(ABORT, 'a lineage is kept forever'); END;
-- Every name a lineage has held, its renames. A name is never reused in its family, so an event written under an old
-- name, and the lineage's first-name digest, still name it.
CREATE TABLE definition_name (
  family TEXT NOT NULL,
  name TEXT NOT NULL,
  definition_id TEXT NOT NULL REFERENCES definition_lineage(definition_id),
  named_seq INTEGER NOT NULL,
  PRIMARY KEY (family, name)
) STRICT, WITHOUT ROWID;
CREATE TRIGGER definition_name_no_update BEFORE UPDATE ON definition_name BEGIN SELECT RAISE(ABORT, 'a held name is never moved'); END;
CREATE TRIGGER definition_name_no_delete BEFORE DELETE ON definition_name BEGIN SELECT RAISE(ABORT, 'a held name is kept forever'); END;
CREATE TABLE definition_version (
  definition_id TEXT NOT NULL REFERENCES definition_lineage(definition_id),
  version INTEGER NOT NULL CHECK (version >= 1),
  status TEXT NOT NULL CHECK (status IN ('experimental', 'standard', 'deprecated', 'superseded')),
  relation TEXT CHECK (relation IN ('equal', 'convertible', 'not_comparable')),  -- to the version before it
  conversion TEXT CHECK (conversion IS NULL OR json_valid(conversion)),  -- convertible: {"scale", "offset"}, new = scale * old + offset
  spec TEXT NOT NULL CHECK (json_valid(spec)),
  text TEXT NOT NULL,             -- the definition in words, for the person and agents
  code_ref TEXT,                  -- module:function that implements it
  golden TEXT CHECK (golden IS NULL OR json_valid(golden)),  -- a small case: code that moves it without a new version fails
  owner TEXT NOT NULL,            -- the node that owns it
  registered_seq INTEGER NOT NULL,
  status_seq INTEGER NOT NULL,    -- the event that set the current status
  status_reason TEXT,             -- superseded: the defect fixed
  PRIMARY KEY (definition_id, version),
  CHECK ((version = 1) = (relation IS NULL)),
  CHECK ((relation IS 'convertible') = (conversion IS NOT NULL))
) STRICT, WITHOUT ROWID;
CREATE TRIGGER definition_frozen BEFORE UPDATE OF definition_id, version, relation, conversion, spec, text, code_ref, golden, owner, registered_seq ON definition_version
BEGIN SELECT RAISE(ABORT, 'a definition version is never edited'); END;
CREATE TRIGGER definition_no_delete BEFORE DELETE ON definition_version BEGIN SELECT RAISE(ABORT, 'a definition is deprecated, never deleted'); END;
CREATE TRIGGER definition_versions_in_order BEFORE INSERT ON definition_version
WHEN NEW.version > 1 AND NOT EXISTS (SELECT 1 FROM definition_version WHERE definition_id = NEW.definition_id AND version = NEW.version - 1)
BEGIN SELECT RAISE(ABORT, 'a version follows the one before it'); END;
-- What a definition's spec names of another: a feature set's method, an artifact type's measures. Rows, so a check
-- guards them; the spec keeps the names as written. ref_version NULL: the lineage, at any version.
CREATE TABLE definition_ref (
  definition_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  role TEXT NOT NULL,             -- the spec field that names it
  ref_id TEXT NOT NULL REFERENCES definition_lineage(definition_id),
  ref_version INTEGER,
  PRIMARY KEY (definition_id, version, role, ref_id),
  FOREIGN KEY (definition_id, version) REFERENCES definition_version(definition_id, version)
) STRICT, WITHOUT ROWID;
CREATE TRIGGER definition_ref_registered BEFORE INSERT ON definition_ref
WHEN NEW.ref_version IS NOT NULL AND NOT EXISTS (SELECT 1 FROM definition_version WHERE definition_id = NEW.ref_id AND version = NEW.ref_version)
BEGIN SELECT RAISE(ABORT, 'a definition names a version that is not registered'); END;
-- Every version under every name its lineage has held: an event's written name, an old name in code, and the
-- current name all find it. `current_name` is the one to show.
CREATE VIEW definition AS
SELECT n.family, n.name, v.version, v.status, v.relation, v.conversion, v.spec, v.text, v.code_ref, v.golden, v.owner,
       v.registered_seq, v.status_seq, v.status_reason, v.definition_id, l.name AS current_name
FROM definition_name n JOIN definition_lineage l ON l.definition_id = n.definition_id
JOIN definition_version v ON v.definition_id = n.definition_id;

-- The version new work uses: the newest that is experimental or standard.
CREATE VIEW definition_current AS
SELECT d.* FROM definition d
WHERE d.name = d.current_name AND d.status IN ('experimental', 'standard')
  AND d.version = (SELECT max(x.version) FROM definition_version x
                   WHERE x.definition_id = d.definition_id AND x.status IN ('experimental', 'standard'));
-- The agent's manual of event types.
CREATE VIEW event_type AS
SELECT definition_id AS type_id, name AS type, version, status, json_extract(spec, '$.work') AS work, json_extract(spec, '$.subject') AS subject_kind,
       json_extract(spec, '$.required') AS required, json_extract(spec, '$.text') AS text_use,
       json_extract(spec, '$.durable') AS durable, text AS description, owner
FROM definition_current WHERE family = 'event_type';
-- Every version of every metric, since a recorded value names its version.
CREATE VIEW metric AS
SELECT definition_id AS metric_id, name AS metric, version, status, relation, conversion, json_extract(spec, '$.unit') AS unit,
       json_extract(spec, '$.direction') AS direction, json_extract(spec, '$.params') AS params,
       text AS description, code_ref, golden
FROM definition WHERE family = 'metric' AND name = current_name;
CREATE VIEW artifact_type AS
SELECT definition_id AS type_id, name AS type, version, status, json_extract(spec, '$.storage') AS storage, json_extract(spec, '$.table') AS table_name,
       json_extract(spec, '$.measures') AS standard_measures, json_extract(spec, '$.keep') AS keep, text AS description
FROM definition_current WHERE family = 'artifact_type';
CREATE VIEW method AS
SELECT definition_id AS method_id, name AS method, version, status, relation, json_extract(spec, '$.materialize') AS materialize,
       json_extract(spec, '$.threshold_ms') AS threshold_ms, json_extract(spec, '$.output') AS output, text AS description, code_ref
FROM definition WHERE family = 'method' AND name = current_name;

-- ===== [1] Work =====
-- A workspace is a tag; its folder holds only workspace.json.
CREATE TABLE workspace (
  workspace_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  folder TEXT,                 -- the last folder seen; the Host's alone, never shown to an agent
  status TEXT NOT NULL CHECK (status IN ('active', 'archived', 'missing', 'deleted')),
  created_seq INTEGER NOT NULL,
  opened_seq INTEGER
) STRICT, WITHOUT ROWID;

CREATE TABLE goal (
  goal_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL REFERENCES workspace(workspace_id),
  title TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('open', 'done', 'dropped')),
  deadline_us INTEGER,
  set_seq INTEGER NOT NULL,
  closed_seq INTEGER,
  CHECK ((state = 'open') = (closed_seq IS NULL))
) STRICT, WITHOUT ROWID;

-- One execution of a task is one trace.
CREATE TABLE task (
  task_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL REFERENCES workspace(workspace_id),
  goal_id TEXT REFERENCES goal(goal_id),
  type_id TEXT NOT NULL REFERENCES definition_lineage(definition_id),  -- its work event type's lineage
  state TEXT NOT NULL CHECK (state IN ('requested', 'running', 'waiting', 'completed', 'failed', 'cancelled')),
  trace_id TEXT NOT NULL,
  product_version TEXT NOT NULL,  -- the code it started under; an upgrade decides by it
  attempts INTEGER NOT NULL DEFAULT 0,
  requested_seq INTEGER NOT NULL,
  started_seq INTEGER,
  ended_seq INTEGER,
  CHECK (started_seq IS NULL OR started_seq > requested_seq),
  CHECK (ended_seq IS NULL OR ended_seq > requested_seq),
  CHECK ((state IN ('completed', 'failed', 'cancelled')) = (ended_seq IS NOT NULL))
) STRICT, WITHOUT ROWID;
CREATE INDEX task_open ON task(state) WHERE state IN ('requested', 'running', 'waiting');

-- A recovery point: a row exists once a slow step, or one with an outside effect, has completed. A restart skips it.
CREATE TABLE task_step (
  task_id TEXT NOT NULL REFERENCES task(task_id),
  step TEXT NOT NULL,
  output TEXT CHECK (output IS NULL OR json_valid(output)),  -- small, or an artifact id
  completed_seq INTEGER NOT NULL,
  PRIMARY KEY (task_id, step)
) STRICT, WITHOUT ROWID;

-- A durable wait for an event (an approval, an answer) or a time (a prediction's horizon). It holds no process.
CREATE TABLE wait (
  wait_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL REFERENCES task(task_id),
  kind TEXT NOT NULL CHECK (kind IN ('event', 'time')),
  event_type_id TEXT REFERENCES definition_lineage(definition_id),  -- the awaited event type's lineage
  match_attrs TEXT CHECK (match_attrs IS NULL OR json_valid(match_attrs)),
  fires_us INTEGER,
  purpose TEXT NOT NULL,
  started_seq INTEGER NOT NULL,
  resolved_seq INTEGER,
  resolved_by_seq INTEGER,
  CHECK ((kind = 'event' AND event_type_id IS NOT NULL) OR (kind = 'time' AND fires_us IS NOT NULL))
) STRICT, WITHOUT ROWID;
CREATE INDEX wait_open ON wait(kind, fires_us) WHERE resolved_seq IS NULL;

-- ===== [1] Research: state, action, effect, new state =====
-- The action's definition. Its identity holds the method's lineage, its version, the parameters and the upstream
-- recipes, and nothing else: no name, no code build and no seed, so one approach keeps one key across releases, trials
-- and renames.
CREATE TABLE recipe (
  recipe_key TEXT PRIMARY KEY,  -- blake3 of the canonical (method_id, method_version, params, sorted upstream keys)
  method_id TEXT NOT NULL,
  method_version INTEGER NOT NULL,
  params TEXT NOT NULL CHECK (json_valid(params)),
  defined_seq INTEGER NOT NULL,
  FOREIGN KEY (method_id, method_version) REFERENCES definition_version(definition_id, version)
) STRICT, WITHOUT ROWID;
CREATE INDEX recipe_method ON recipe(method_id, method_version);
CREATE TRIGGER recipe_no_update BEFORE UPDATE ON recipe BEGIN SELECT RAISE(ABORT, 'a recipe is never edited'); END;
-- The recipes one reads, in the order its key hashes them; each is registered before the recipe that reads it.
CREATE TABLE recipe_upstream (
  recipe_key TEXT NOT NULL REFERENCES recipe(recipe_key),
  position INTEGER NOT NULL CHECK (position >= 0),
  upstream_key TEXT NOT NULL REFERENCES recipe(recipe_key),
  PRIMARY KEY (recipe_key, position),
  UNIQUE (recipe_key, upstream_key)
) STRICT, WITHOUT ROWID;

-- What a piece of research was allowed to see. The knowledge cutoff is what makes it point-in-time; no snapshot is pinned.
CREATE TABLE selection (
  selection_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL REFERENCES workspace(workspace_id),
  cutoff_us INTEGER NOT NULL,   -- nothing known after this moment is read
  universe TEXT,
  valid_from TEXT,              -- the span of market dates read, ISO dates
  valid_to TEXT,
  spec TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(spec)),  -- [3] the rest: sources preferred, filters
  made_seq INTEGER NOT NULL
) STRICT, WITHOUT ROWID;
CREATE TRIGGER selection_no_update BEFORE UPDATE ON selection BEGIN SELECT RAISE(ABORT, 'a selection is never edited'); END;

CREATE TABLE hypothesis (
  hypothesis_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL REFERENCES workspace(workspace_id),
  goal_id TEXT REFERENCES goal(goal_id),
  standing TEXT NOT NULL CHECK (standing IN ('open', 'supported', 'refuted', 'inconclusive', 'dropped')),
  stated_seq INTEGER NOT NULL,   -- its statement is that event's text
  standing_seq INTEGER NOT NULL
) STRICT, WITHOUT ROWID;

-- One trial is one state transition. State: the goal, the hypothesis, the selection, the context. Action: the
-- recipe and seeds. Effect: its result artifact's measures. New state: the hypothesis's standing, the notes on it.
-- Registered before it runs, and counted whether it completes, fails or is cancelled.
CREATE TABLE trial (
  trial_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL REFERENCES workspace(workspace_id),
  goal_id TEXT REFERENCES goal(goal_id),
  hypothesis_id TEXT REFERENCES hypothesis(hypothesis_id),
  selection_id TEXT NOT NULL REFERENCES selection(selection_id),
  recipe_key TEXT NOT NULL REFERENCES recipe(recipe_key),
  seeds TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(seeds)),
  product_version TEXT NOT NULL,
  model TEXT,                   -- the agent's model, when an agent ran it
  context TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(context) AND length(context) <= 2048),  -- [3] conclusions in hand, a few market-condition measures
  task_id TEXT REFERENCES task(task_id),
  state TEXT NOT NULL CHECK (state IN ('registered', 'running', 'completed', 'failed', 'cancelled')),
  result_artifact TEXT,         -- its headline artifact, its own, named by its completion; the spread reads this one alone
  standing_after TEXT CHECK (standing_after IS NULL OR standing_after IN ('open', 'supported', 'refuted', 'inconclusive', 'dropped')),
                                -- the hypothesis's standing this trial left, never moved by a later trial
  reads_legacy INTEGER NOT NULL DEFAULT 0 CHECK (reads_legacy IN (0, 1)),  -- it read rows whose known time is derived
  kept INTEGER NOT NULL DEFAULT 0 CHECK (kept IN (0, 1)),                  -- the person keeps its artifacts
  candidates INTEGER NOT NULL DEFAULT 1 CHECK (candidates >= 1),  -- configurations evaluated inside it; the search size sums them
  registered_seq INTEGER NOT NULL,
  ended_seq INTEGER,
  CHECK (ended_seq IS NULL OR ended_seq > registered_seq)
) STRICT, WITHOUT ROWID;
CREATE INDEX trial_recipe ON trial(recipe_key);
CREATE INDEX trial_hypothesis ON trial(hypothesis_id);
CREATE INDEX trial_goal ON trial(goal_id);

-- Registered first, scored at its horizon.
CREATE TABLE prediction (
  prediction_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL REFERENCES workspace(workspace_id),
  trial_id TEXT REFERENCES trial(trial_id),
  target TEXT NOT NULL CHECK (json_valid(target)),  -- [3] what is predicted and how it is judged
  horizon_us INTEGER NOT NULL,
  confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  registered_seq INTEGER NOT NULL,  -- its statement is that event's text
  outcome_seq INTEGER,
  score REAL,
  score_metric_id TEXT REFERENCES definition_lineage(definition_id),
  score_metric_version INTEGER,
  CHECK (outcome_seq IS NULL OR outcome_seq > registered_seq),
  CHECK ((score IS NULL) = (score_metric_id IS NULL))
) STRICT, WITHOUT ROWID;
CREATE INDEX prediction_due ON prediction(horizon_us) WHERE outcome_seq IS NULL;

-- Facts made once and never changed are views over the log, served by event_type_subject.
CREATE VIEW decision AS
SELECT e.subject_id AS decision_id, e.workspace_id, json_extract(e.attrs, '$.kind') AS kind,
       json_extract(e.attrs, '$.chosen') AS chosen, json_extract(e.attrs, '$.alternatives') AS alternatives,
       t.text AS reasons, e.actor_kind, e.actor_id, e.seq AS made_seq
FROM event e LEFT JOIN event_text t ON t.seq = e.seq WHERE e.type = 'decision.made';
-- Conclusions, lessons, reviews and proposals: each is about its subject, and its body is its text.
CREATE VIEW research_note AS
SELECT e.seq, e.event_id AS note_id, e.type, e.subject_kind, e.subject_id, e.workspace_id, t.text,
       e.actor_kind, e.actor_id, e.occurred_us
FROM event e LEFT JOIN event_text t ON t.seq = e.seq
WHERE e.type IN ('conclusion.drawn', 'lesson.recorded', 'review.completed', 'proposal.made');
-- The person's feedback: adopted or not, read from what they did, and an optional remark.
CREATE VIEW feedback AS
SELECT e.seq, e.subject_kind, e.subject_id, json_extract(e.attrs, '$.adopted') AS adopted, t.text AS remark, e.occurred_us
FROM event e LEFT JOIN event_text t ON t.seq = e.seq WHERE e.type = 'feedback.given';
-- One admission of outside data. World rows carry its admission_id, minted before the write. `parts` lists each
-- partition it touched as {part, min_known_asof_us, max_known_asof_us}, the knowledge times of its rows there.
CREATE VIEW admission AS
SELECT subject_id AS admission_id, seq AS admission_seq, json_extract(attrs, '$.provider_id') AS provider_id,
       json_extract(attrs, '$.dataset') AS dataset, json_extract(attrs, '$.row_count') AS row_count,
       json_extract(attrs, '$.catalog_snapshot') AS catalog_snapshot, json_extract(attrs, '$.known_us') AS known_us,
       json_extract(attrs, '$.parts') AS parts  -- a cache's input digest reads these, never the data
FROM event WHERE type = 'admission.committed';

-- ===== [1] Artifacts: the profile stays, the body may go =====
-- What an artifact is, what it means, and where its body is or was. The profile is kept forever; the body may be
-- collected, or never stored at all. Its measures are in `measure`.
CREATE TABLE artifact (
  artifact_id TEXT PRIMARY KEY,     -- blake3: of the content (small final artifacts), or of the recipe and the producing span
  identity_kind TEXT NOT NULL CHECK (identity_kind IN ('content', 'recipe')),
  type_id TEXT NOT NULL,            -- the artifact type's lineage
  type_version INTEGER NOT NULL,
  meaning TEXT NOT NULL CHECK (length(meaning) BETWEEN 1 AND 1000),  -- the question it answers, in a sentence or two
  recipe_key TEXT REFERENCES recipe(recipe_key),
  selection_id TEXT REFERENCES selection(selection_id),
  trial_id TEXT REFERENCES trial(trial_id),
  shape TEXT CHECK (shape IS NULL OR json_valid(shape)),  -- a table's columns and types
  valid_from TEXT,                  -- the market dates it covers
  valid_to TEXT,
  row_count INTEGER,
  size_bytes INTEGER,
  content_digest TEXT,              -- of the bytes, where one is computed anyway; checked at export and by the scrub
  storage TEXT NOT NULL CHECK (storage IN ('none', 'inline', 'blob', 'table')),
  body_state TEXT NOT NULL CHECK (body_state IN ('never_stored', 'present', 'collected')),
  produced_seq INTEGER NOT NULL,
  collected_seq INTEGER,
  CHECK (identity_kind = 'content' OR recipe_key IS NOT NULL),
  CHECK ((storage = 'none') = (body_state = 'never_stored')),
  CHECK ((body_state = 'collected') = (collected_seq IS NOT NULL)),
  FOREIGN KEY (type_id, type_version) REFERENCES definition_version(definition_id, version)
) STRICT, WITHOUT ROWID;
CREATE INDEX artifact_recipe ON artifact(recipe_key);
CREATE INDEX artifact_trial ON artifact(trial_id);
CREATE TRIGGER artifact_no_delete BEFORE DELETE ON artifact BEGIN SELECT RAISE(ABORT, 'an artifact profile is kept forever'); END;

-- [4] A small body (at most 64 KiB). Larger ones are files under blobs/ or rows in results.*. It is opaque storage, not a
-- projection: the Host writes it in the same transaction as its `artifact.produced`, which refuses an inline artifact
-- without it; replay, backup and restore copy it as they copy blobs. Collection deletes the row.
CREATE TABLE artifact_body (
  artifact_id TEXT PRIMARY KEY REFERENCES artifact(artifact_id),
  body BLOB NOT NULL CHECK (length(body) <= 65536)
) STRICT;

-- Every measure of every artifact, one value per row. Recorded values never move: a changed definition is a new
-- metric version, and a new artifact records every version not superseded, deprecated ones included. Two artifacts
-- compare on the same (metric_id, metric_version, dims), or across versions joined by equal or convertible steps
-- (reading.compare), with neither body present. A renamed metric keeps every recorded value.
CREATE TABLE measure (
  artifact_id TEXT NOT NULL REFERENCES artifact(artifact_id),
  metric_id TEXT NOT NULL,
  metric_version INTEGER NOT NULL,
  dims TEXT NOT NULL DEFAULT '',   -- canonical 'k=v|k=v', such as 'horizon=5d'; '' for the whole artifact
  value REAL,                      -- NULL: computed, and undefined for this artifact
  recorded_seq INTEGER NOT NULL,
  PRIMARY KEY (artifact_id, metric_id, metric_version, dims),
  FOREIGN KEY (metric_id, metric_version) REFERENCES definition_version(definition_id, version)
) STRICT, WITHOUT ROWID;
CREATE INDEX measure_by_metric ON measure(metric_id, metric_version, dims, value);
CREATE TRIGGER measure_no_update BEFORE UPDATE ON measure BEGIN SELECT RAISE(ABORT, 'a recorded measure never moves'); END;
CREATE TRIGGER measure_no_delete BEFORE DELETE ON measure BEGIN SELECT RAISE(ABORT, 'a recorded measure is kept forever'); END;

-- What a step used and produced, by the ids it already held; an artifact's own producing event is its produced_seq.
CREATE TABLE event_artifact (
  seq INTEGER NOT NULL REFERENCES event(seq),
  role TEXT NOT NULL CHECK (role IN ('used', 'produced')),
  artifact_id TEXT NOT NULL,
  PRIMARY KEY (seq, role, artifact_id)
) STRICT, WITHOUT ROWID;
CREATE INDEX event_artifact_by_artifact ON event_artifact(artifact_id, role);

-- The research memory's keys: an instrument, a theme, a method family, anything a tag vocabulary defines.
CREATE TABLE tag (
  subject_kind TEXT NOT NULL,
  subject_id TEXT NOT NULL,
  key_id TEXT NOT NULL,                    -- the tag vocabulary's lineage
  tag_version INTEGER NOT NULL DEFAULT 1,  -- the vocabulary's version: as given, else the key's current one; an unregistered key is refused
  value TEXT NOT NULL,
  seq INTEGER NOT NULL,
  PRIMARY KEY (subject_kind, subject_id, key_id, tag_version, value),
  FOREIGN KEY (key_id, tag_version) REFERENCES definition_version(definition_id, version)
) STRICT, WITHOUT ROWID;
CREATE INDEX tag_lookup ON tag(key_id, value, subject_kind);

-- What keeps an artifact's body from collection: a running task, an active book, a publication, a kept trial, a
-- backup. A body no row holds is collected 7 days after it was produced. A crash may over-retain, never under-retain.
CREATE TABLE live_ref (
  artifact_id TEXT NOT NULL REFERENCES artifact(artifact_id),
  root_kind TEXT NOT NULL CHECK (root_kind IN ('task', 'book', 'publication', 'trial', 'backup')),
  root_id TEXT NOT NULL,
  since_seq INTEGER NOT NULL,
  PRIMARY KEY (artifact_id, root_kind, root_id)
) STRICT, WITHOUT ROWID;

-- Daily totals of finished steps, kept in each step's own transaction (a projection, rebuilt by replay), so the
-- scoreboards read days, never the whole history. Money and time are integers, so sums are exact in any order.
CREATE TABLE daily_span (
  day TEXT NOT NULL,             -- UTC date of the step's end
  actor_kind TEXT NOT NULL,
  actor_id TEXT NOT NULL,
  node_id TEXT NOT NULL,         -- the emitting node's lineage, named on first sight
  type_id TEXT NOT NULL,         -- the event type's lineage
  outcome TEXT NOT NULL,
  n INTEGER NOT NULL,
  micros INTEGER NOT NULL,
  tokens_in INTEGER NOT NULL,
  tokens_out INTEGER NOT NULL,
  cost_microusd INTEGER NOT NULL,
  bytes_read INTEGER NOT NULL,
  bytes_written INTEGER NOT NULL,
  PRIMARY KEY (day, actor_kind, actor_id, node_id, type_id, outcome)
) STRICT, WITHOUT ROWID;

-- ===== [1] Deployed strategies: an exact ledger =====
CREATE TABLE book (
  book_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL REFERENCES workspace(workspace_id),
  mode TEXT NOT NULL CHECK (mode IN ('paper', 'live')),
  state TEXT NOT NULL CHECK (state IN ('active', 'retired')),
  holding_artifact TEXT,   -- the target holdings it follows
  activated_seq INTEGER NOT NULL,
  retired_seq INTEGER
) STRICT, WITHOUT ROWID;

-- Double entry: the lines of one event sum to zero for each asset. Quantities are integers in nano-units (1e-9 of a
-- share or a currency unit), so sums are exact. A fill moves the asset from `market` to `book` and cash the other way.
CREATE TABLE ledger_entry (
  seq INTEGER NOT NULL REFERENCES event(seq),
  line INTEGER NOT NULL,
  book_id TEXT NOT NULL REFERENCES book(book_id),
  account TEXT NOT NULL CHECK (account IN ('book', 'market', 'fees', 'owner')),
  asset TEXT NOT NULL,            -- a listing id, or 'CASH:USD'
  quantity_n INTEGER NOT NULL CHECK (quantity_n <> 0),
  PRIMARY KEY (seq, line)
) STRICT, WITHOUT ROWID;
CREATE INDEX ledger_by_book ON ledger_entry(book_id, account, asset);
CREATE TRIGGER ledger_no_update BEFORE UPDATE ON ledger_entry BEGIN SELECT RAISE(ABORT, 'the ledger is append-only'); END;
CREATE TRIGGER ledger_no_delete BEFORE DELETE ON ledger_entry BEGIN SELECT RAISE(ABORT, 'the ledger is append-only'); END;
CREATE VIEW position AS
SELECT book_id, asset, sum(quantity_n) AS quantity_n FROM ledger_entry
WHERE account = 'book' GROUP BY book_id, asset HAVING sum(quantity_n) <> 0;
-- Must be empty; the Host checks it in the posting transaction, and the scrub checks it whole.
CREATE VIEW ledger_unbalanced AS
SELECT seq, asset, sum(quantity_n) AS off_n FROM ledger_entry GROUP BY seq, asset HAVING sum(quantity_n) <> 0;

-- ===== [1] The person's settings =====
-- The current value; history is in the log. A key's value is in the OS keychain, never here.
CREATE TABLE setting (
  scope TEXT NOT NULL,    -- 'machine' or a workspace id
  key TEXT NOT NULL,
  value TEXT CHECK (value IS NULL OR json_valid(value)),
  secret_ref TEXT,        -- the keychain entry's name
  changed_seq INTEGER NOT NULL,
  PRIMARY KEY (scope, key),
  CHECK (value IS NULL OR secret_ref IS NULL)
) STRICT, WITHOUT ROWID;

-- ===== The Host's working state (not projections; safe to lose) =====
-- Which cached partition of derived.* is valid for which inputs. A cache hit needs the same recipe, partition and
-- digest of the exact rows it depends on; a correction to those rows changes the digest and nothing else. The digest
-- is per (part, cutoff): the admissions whose rows in the part are visible at the cutoff (min_known_asof_us <= cutoff),
-- the cutoff itself when it falls inside one of their ranges, else their largest max_known_asof_us, and the lag-rule
-- versions, so a later admission never reaches an earlier cutoff's cache (project.input_digest).
CREATE TABLE cache_entry (
  recipe_key TEXT NOT NULL,
  part TEXT NOT NULL,             -- a year, or an as-of date
  input_digest TEXT NOT NULL,
  row_count INTEGER NOT NULL,
  size_bytes INTEGER NOT NULL,
  compute_ms INTEGER NOT NULL,
  computed_us INTEGER NOT NULL,
  last_read_us INTEGER NOT NULL,  -- eviction reads it
  PRIMARY KEY (recipe_key, part, input_digest)
) STRICT, WITHOUT ROWID;

-- Files to delete that another process still holds open; retried, never blocking.
CREATE TABLE deletion_queue (
  handle TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('file', 'blob', 'staging')),
  queued_us INTEGER NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT
) STRICT, WITHOUT ROWID;
