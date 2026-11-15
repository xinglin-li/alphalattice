-- The reading surface, defined in DuckDB when the Host starts. Nothing here is stored: every view and macro is
-- computed on read from the log (attached read-only) and the lake. The Host's session runs SET TimeZone = 'UTC';
-- times are TIMESTAMPTZ and a cutoff is integer microseconds, so no local offset moves a cutoff.
--   ATTACH '<data root>/machine/log.sqlite' AS log (TYPE SQLITE, READ_ONLY);

-- ===== Point-in-time reads of the world =====
-- What we knew at `cutoff_us`: among the versions known by then, the one no later version had replaced by then, unless
-- it withdraws the fact. A version is replaced when world.superseded says so, which the Host writes with the replacing
-- row, so no read ranks every version. Every world table gets this macro; a selection's cutoff_us is the argument.
-- A decision at session t reads knowledge no later than the lesser of its selection's cutoff and t's decision time.
CREATE MACRO raw_bar_asof(cutoff_us) AS TABLE
SELECT r.* FROM lake.world.raw_bar r
WHERE r.known_asof <= make_timestamptz(cutoff_us) AND NOT r.retracted
  AND NOT EXISTS (SELECT 1 FROM lake.world.superseded s WHERE s.table_name = 'raw_bar' AND s.value_version = r.value_version
                  AND s.admission_id = r.admission_id AND s.superseded_asof <= make_timestamptz(cutoff_us));

CREATE MACRO corporate_action_asof(cutoff_us) AS TABLE
SELECT r.* FROM lake.world.corporate_action r
WHERE r.known_asof <= make_timestamptz(cutoff_us) AND NOT r.retracted
  AND NOT EXISTS (SELECT 1 FROM lake.world.superseded s WHERE s.table_name = 'corporate_action' AND s.value_version = r.value_version
                  AND s.admission_id = r.admission_id AND s.superseded_asof <= make_timestamptz(cutoff_us));

-- Every version with the moment the next one replaced it (world.superseded), for a walk-forward read over many decision times in one
-- pass: version v is what decision time t saw when known_asof <= t < visible_to (or visible_to is null), with t never
-- past the selection's cutoff and a retracted v showing nothing. A price adjustment as of t applies only actions with
-- effective_date <= t.
CREATE VIEW raw_bar_versions AS
SELECT r.*, s.superseded_asof AS visible_to FROM lake.world.raw_bar r
LEFT JOIN lake.world.superseded s ON s.table_name = 'raw_bar' AND s.value_version = r.value_version AND s.admission_id = r.admission_id;

CREATE VIEW corporate_action_versions AS
SELECT r.*, s.superseded_asof AS visible_to FROM lake.world.corporate_action r
LEFT JOIN lake.world.superseded s ON s.table_name = 'corporate_action' AND s.value_version = r.value_version AND s.admission_id = r.admission_id;

-- What we know now.
CREATE VIEW raw_bar_latest AS
SELECT r.* FROM lake.world.raw_bar r
WHERE NOT r.retracted AND NOT EXISTS (SELECT 1 FROM lake.world.superseded s WHERE s.table_name = 'raw_bar' AND s.value_version = r.value_version
                  AND s.admission_id = r.admission_id);

-- ===== Steps =====
-- One row per finished step: a filter over the log, since a step is one event.
CREATE VIEW span AS
SELECT seq, span_id, trace_id, parent_span_id, type, node, workspace_id, actor_kind, actor_id, subject_kind, subject_id,
       make_timestamptz(started_us) AS started_at, make_timestamptz(occurred_us) AS ended_at,
       (occurred_us - started_us) / 1e6 AS seconds, phase AS outcome,
       CAST(json_extract_string(attrs, '$."gen_ai.usage.input_tokens"') AS BIGINT) AS tokens_in,
       CAST(json_extract_string(attrs, '$."gen_ai.usage.output_tokens"') AS BIGINT) AS tokens_out,
       CAST(json_extract_string(attrs, '$.cost_usd') AS DOUBLE) AS cost_usd,
       CAST(json_extract_string(attrs, '$.bytes_read') AS BIGINT) AS bytes_read,
       CAST(json_extract_string(attrs, '$.bytes_written') AS BIGINT) AS bytes_written
FROM log.event WHERE phase IN ('completed', 'failed', 'cancelled');

-- ===== Experience =====
-- Every measure of every artifact a trial made: its results at once, and its `outcome` artifacts, which record
-- what happened at a horizon after it. A trial has many artifacts; its measures are all of theirs.
CREATE VIEW trial_measure AS
SELECT a.trial_id, a.artifact_id, a.type AS artifact_type, m.metric, m.metric_version, m.dims, m.value, m.recorded_seq
FROM log.artifact a JOIN log.measure m ON m.artifact_id = a.artifact_id
WHERE a.trial_id IS NOT NULL;

-- One row per trial, as state, action, effect and new state: what an agent studies before it tries something.
-- It reads profiles and measures only, so it is whole whether or not any artifact body still exists.
CREATE VIEW experience AS
SELECT t.trial_id, t.workspace_id,
       -- state
       t.goal_id, t.hypothesis_id, ht.text AS hypothesis_text, s.cutoff_us, make_timestamptz(s.cutoff_us) AS knowledge_cutoff,
       s.universe, s.valid_from, s.valid_to,
       t.product_version, t.model, t.context, t.reads_legacy,
       -- action
       r.method, r.method_version, t.recipe_key, r.params, t.seeds,
       -- effect: at once and at each horizon
       t.state AS outcome, sp.seconds, a.artifact_id, a.meaning, a.body_state, m.measures,
       -- new state
       t.standing_after AS hypothesis_standing, h.standing AS hypothesis_standing_now, n.conclusions, n.lessons, n.adopted,
       d.trial_id IS NOT NULL AS deployed, hn.conclusions AS hypothesis_conclusions, hn.lessons AS hypothesis_lessons,
       make_timestamptz(reg.occurred_us) AS registered_at
FROM log.trial t
JOIN log.selection s ON s.selection_id = t.selection_id
JOIN log.recipe r ON r.recipe_key = t.recipe_key
JOIN log.event reg ON reg.seq = t.registered_seq
LEFT JOIN span sp ON sp.seq = t.ended_seq
LEFT JOIN log.artifact a ON a.artifact_id = t.result_artifact
LEFT JOIN (SELECT trial_id,
                  list({'artifact_type': artifact_type, 'metric': metric, 'version': metric_version, 'dims': dims, 'value': value}
                       ORDER BY artifact_type, metric, metric_version, dims) AS measures
           FROM trial_measure GROUP BY trial_id) m ON m.trial_id = t.trial_id
LEFT JOIN log.hypothesis h ON h.hypothesis_id = t.hypothesis_id
LEFT JOIN log.event_text ht ON ht.seq = h.stated_seq
-- adopted is the person's newest word; deployed: a book followed one of the trial's artifacts, at activation or a rebalance
LEFT JOIN (SELECT e.subject_id AS trial_id,
                  list(x.text) FILTER (WHERE e.type = 'conclusion.drawn') AS conclusions,
                  list(x.text) FILTER (WHERE e.type = 'lesson.recorded') AS lessons,
                  arg_max(CAST(json_extract_string(e.attrs, '$.adopted') AS BOOLEAN), e.seq) FILTER (WHERE e.type = 'feedback.given') AS adopted
           FROM log.event e LEFT JOIN log.event_text x ON x.seq = e.seq
           WHERE e.subject_kind = 'trial' AND e.type IN ('conclusion.drawn', 'lesson.recorded', 'feedback.given')
           GROUP BY e.subject_id) n ON n.trial_id = t.trial_id
LEFT JOIN (SELECT e.subject_id AS hypothesis_id,
                  list(x.text) FILTER (WHERE e.type = 'conclusion.drawn') AS conclusions,
                  list(x.text) FILTER (WHERE e.type = 'lesson.recorded') AS lessons
           FROM log.event e LEFT JOIN log.event_text x ON x.seq = e.seq
           WHERE e.subject_kind = 'hypothesis' AND e.type IN ('conclusion.drawn', 'lesson.recorded')
           GROUP BY e.subject_id) hn ON hn.hypothesis_id = t.hypothesis_id
LEFT JOIN (SELECT DISTINCT a.trial_id FROM log.event e
           JOIN log.artifact a ON a.artifact_id = json_extract_string(e.attrs, '$.holding_artifact')
           WHERE e.type IN ('book.activated', 'book.rebalanced') AND a.trial_id IS NOT NULL) d ON d.trial_id = t.trial_id;

-- Code is deterministic; the world is random. One approach's results are draws: this is their spread, so a new
-- result is read against it and a redo outside it stands out. One draw is one trial: its headline artifact alone.
CREATE VIEW approach_spread AS
SELECT t.recipe_key, any_value(r.method) AS method, any_value(r.method_version) AS method_version,
       m.artifact_type, m.metric, m.metric_version, m.dims,
       count(m.value) AS n, avg(m.value) AS mean, stddev_samp(m.value) AS sd,
       min(m.value) AS lo, median(m.value) AS mid, max(m.value) AS hi
FROM log.trial t
JOIN log.recipe r ON r.recipe_key = t.recipe_key
JOIN trial_measure m ON m.trial_id = t.trial_id AND m.artifact_id = t.result_artifact
GROUP BY t.recipe_key, m.artifact_type, m.metric, m.metric_version, m.dims;

-- How wide the search was, over every method: every trial counts, failed and cancelled ones included, and a trial
-- that evaluated many configurations counts each. The deflated Sharpe ratio and the probability of backtest
-- overfitting read `candidates`. By goal, and by selection for trials with no goal.
CREATE VIEW search_size AS
SELECT t.workspace_id, t.goal_id, count(*) AS trials, sum(t.candidates) AS candidates, count(DISTINCT r.method) AS methods,
       count(DISTINCT t.recipe_key) AS recipes, count(*) FILTER (WHERE t.state = 'completed') AS completed
FROM log.trial t JOIN log.recipe r ON r.recipe_key = t.recipe_key
GROUP BY t.workspace_id, t.goal_id;

CREATE VIEW search_size_by_selection AS
SELECT t.workspace_id, t.selection_id, count(*) AS trials, sum(t.candidates) AS candidates, count(DISTINCT r.method) AS methods,
       count(DISTINCT t.recipe_key) AS recipes, count(*) FILTER (WHERE t.state = 'completed') AS completed
FROM log.trial t JOIN log.recipe r ON r.recipe_key = t.recipe_key
GROUP BY t.workspace_id, t.selection_id;

-- Comparing two artifacts is the Host's `compare` (project.py), which reads `measure` through SQLite by its primary
-- key, artifact first: through DuckDB's SQLite scan the filter is not pushed down and every measure is read.

-- ===== Scoreboards =====
CREATE VIEW decision_scorecard AS
SELECT p.prediction_id, p.trial_id, p.workspace_id, e.actor_kind, e.actor_id,
       make_timestamptz(e.occurred_us) AS registered_at, make_timestamptz(p.horizon_us) AS horizon,
       p.confidence, p.score, p.score_metric, p.score_metric_version
FROM log.prediction p JOIN log.event e ON e.seq = p.registered_seq;

-- The scoreboards read the log's daily totals, never the whole history.
CREATE VIEW agent_daily AS
SELECT CAST(day AS DATE) AS day, actor_id,
       sum(n) FILTER (WHERE type = 'agent.turn') AS turns,
       sum(tokens_in) AS tokens_in, sum(tokens_out) AS tokens_out, sum(cost_microusd) / 1e6 AS cost_usd,
       sum(n) FILTER (WHERE type = 'agent.turn' AND outcome = 'completed') AS completed,
       sum(n) FILTER (WHERE type = 'agent.turn' AND outcome = 'failed') AS failed
FROM log.daily_span WHERE actor_kind = 'agent'
GROUP BY ALL;

CREATE VIEW node_io AS
SELECT CAST(day AS DATE) AS day, node, sum(n) AS calls,
       sum(bytes_read) AS bytes_read, sum(bytes_written) AS bytes_written, sum(micros) / 1e6 AS seconds
FROM log.daily_span
GROUP BY ALL;
