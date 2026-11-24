-- Native agent observation: the entire stored shape is this numerical/attribution allowlist.
-- event_id binds the authenticated workspace, sanitized record and its batch ordinal; repeated
-- delivery is idempotent. Bodies, opt-in attributes, tool text and unknown keys have no column.
CREATE SCHEMA IF NOT EXISTS lake.telemetry;
CREATE TABLE IF NOT EXISTS lake.telemetry.event (
  event_id VARCHAR NOT NULL,
  received_us BIGINT NOT NULL,
  record_index BIGINT NOT NULL,
  vendor VARCHAR,
  signal VARCHAR NOT NULL,
  name VARCHAR NOT NULL,
  conversation_id VARCHAR,
  session_id VARCHAR,
  agent_id VARCHAR,
  agent_role VARCHAR,
  parent_agent_id VARCHAR,
  workspace_id VARCHAR,
  goal_id VARCHAR,
  trial_id VARCHAR,
  model VARCHAR,
  effort VARCHAR,
  response_id VARCHAR,
  input_tokens BIGINT,
  cache_read_tokens BIGINT,
  cache_write_tokens BIGINT,
  output_tokens BIGINT,
  trace_id VARCHAR,
  span_id VARCHAR,
  parent_span_id VARCHAR,
  time_ns BIGINT NOT NULL,
  model_seconds DOUBLE,
  metric_value DOUBLE
);
