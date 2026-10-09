/* What a colour says (the user, 2026-09-25: 受阻用琥珀色，红色严格保留给失败): seven meanings, each a
 * tone from design/parameters.json (`role.meaning`); the table and every caller name the meaning,
 * never the colour. Red is failure alone -- work that ran and failed, a read or a render that could
 * not complete, a check whose object did not hold. A stop by decision (blocked, refused) is attention. */
const TONE = {failure: PARAMETER_VALUES['meaning-failure'], attention: PARAMETER_VALUES['meaning-attention'], decision: PARAMETER_VALUES['meaning-decision'], done: PARAMETER_VALUES['meaning-done'], active: PARAMETER_VALUES['meaning-active'], stopped: PARAMETER_VALUES['meaning-stopped'], rest: PARAMETER_VALUES['meaning-rest']};
const STATES = {
  // the owner is working
  queued: {word: 'Queued', tone: TONE.rest, line: 'Accepted for execution; not started.', moving: true},
  running: {word: 'Running', tone: TONE.active, line: 'Executing; submission is not completion.', moving: true},
  cancel_requested: {word: 'Cancelling', tone: TONE.attention, line: 'Cancellation requested; the worker stops at its next safe checkpoint.', moving: true},
  in_progress: {word: 'In progress', tone: TONE.active, moving: true},
  reused_in_flight: {word: 'Reused · in flight', tone: TONE.rest, moving: true},
  preparing: {word: 'Preparing the workspace', tone: TONE.active, line: 'The first preparation runs while the local service runs; closing the page stops nothing.', moving: true},
  settling: {word: 'Settling', tone: TONE.active, line: 'The choice was sent; the owner\'s next readback says where the Task stands.', moving: true},
  recheck_due: {word: 'Recheck due', tone: TONE.attention, line: 'Listings the owner will check again at the next update; nothing is needed from you.'},
  // held: at its owner, or waiting for a person
  review_pending: {word: 'Waiting for a decision', tone: TONE.decision, line: 'Waiting for an authorized decision.', next: 'decide', held: true},
  blocked: {word: 'Blocked', tone: TONE.attention, line: 'The owner stopped the Task and recorded why.', next: 'read the reason, then take the way its Task offers', held: true},
  recovery_required: {word: 'Needs recovery', tone: TONE.attention, line: 'The Task was interrupted; its owner needs a supported recovery.', next: 'resume or cancel', held: true},
  deferred: {word: 'Deferred', tone: TONE.attention, line: 'Waiting for a condition or a scheduled retry at its owner; nothing is needed from you yet.', held: true},
  // ended
  succeeded: {word: 'Completed', tone: TONE.done, line: 'The named Task completed; this grants no trading authority.'},
  cancelled: {word: 'Cancelled', tone: TONE.stopped, line: 'Stopped. Prior saved artifacts remain readable.'}, // W (the user's reading, 2026-09-23): a stop is a fact, not the plain record -- its grey dot is shown where a neutral one is not
  failed: {word: 'Failed', tone: TONE.failure, line: 'A failure occurred; the owner recorded why.', next: 're-plan'},
  refused: {word: 'Refused', tone: TONE.attention, line: 'The owner refused the request and recorded why.', next: 're-plan'},
  error: {word: 'Error', tone: TONE.failure, line: 'A failure occurred. Diagnose before retrying.'},
  // a stage's mark
  verified: {word: 'Verified', tone: TONE.done},
  ready_for_verification: {word: 'Ready for verification', tone: TONE.active},
  pending: {word: 'Not started', tone: TONE.rest},
  // an object's standing
  ready: {word: 'Ready', tone: TONE.done, line: 'Available for its stated scope.'},
  draft: {word: 'Draft', tone: TONE.rest, line: 'An editable declaration; nothing has run.'},
  planned: {word: 'Planned', tone: TONE.active, line: 'A preview only; no computation or execution result.'},
  reused_exact: {word: 'Reused exact', tone: TONE.active, line: 'An exact compatible result reused; not a fresh computation.'},
  partial: {word: 'Partial', tone: TONE.attention, line: 'Only a stated subset was reviewed or covered.'},
  expired: {word: 'Expired', tone: TONE.attention, line: 'A dated item is outside its validity window.'},
  authority_not_admitted: {word: 'Authority not admitted', tone: TONE.decision, line: 'The requested authority has not been admitted.'},
  historical: {word: 'Historical', tone: TONE.rest, line: 'Read-only readback, not current investment advice.'},
  metadata: {word: 'Metadata discovered', tone: TONE.rest, line: 'Identity found; contents not yet verified.'},
  no_book_to_review: {word: 'No book to review', tone: TONE.rest, line: 'No sealed book is selected; nothing is inferred.'},
  evidence_authority_not_admitted: {word: 'Evidence authority not admitted', tone: TONE.attention, line: 'No issuer registry and listing authority are admitted for this workspace; coverage cannot be computed and nothing is invented.'},
  model_authority_not_admitted: {word: 'Answered by an agent', tone: TONE.rest, line: 'The product runs no model of its own: an agent answers the Analyst\'s packet and the CRO\'s dossier through its bundle. Sources are prepared locally, and published work stays readable.'},
  evidence_refresh_in_progress: {word: 'Evidence work in progress', tone: TONE.active, line: 'A Task is preparing or validating evidence for this book; its stages are in the work area. Nothing is inferred from it until it publishes.', moving: true},
  evidence_selection_ambiguous: {word: 'Choose the analysis to review against', tone: TONE.attention, line: 'More than one current analysis is eligible; the workspace authority must select the exact one. Another refresh does not resolve this.', next: 'choose the exact analysis', held: true},
  awaiting_alternative_evidence: {word: 'No admitted analysis yet', tone: TONE.rest, line: 'No current Alternative Evidence analysis exists for this book. Prepare sources, then hand the packet to an Analyst.', next: 'prepare sources'},
  alternative_evidence_ready_for_review: {word: 'Analysis published; no review yet', tone: TONE.rest, line: 'An analysis is current; no CRO review has been published against it. A review is separate work with its own confirmation.', next: 'hand the dossier to the CRO'},
  alternative_evidence_expired: {word: 'The analysis has expired', tone: TONE.attention, line: 'The selected analysis is past its expiry; a current review needs newer evidence. The old publications stay readable.', next: 'prepare newer evidence'},
  review_input_incomplete: {word: 'Review input incomplete', tone: TONE.attention, line: 'This book lacks a sealed preceding session or pretrade weights; it stays readable and missing changes are not zero.'},
  review_published: {word: 'Review published', tone: TONE.done, line: 'A CRO review is published for this exact book and evidence version.'},
  // the book's sources (round E1): an issuer's holding, a unit of a coverage run, a checked resource's outcome, a check's outcome
  held: {word: 'Held', tone: TONE.rest, line: 'The workspace holds this issuer\'s admitted documents; they are verified against their references at use, never assumed current.'},
  no_source: {word: 'No source', tone: TONE.attention, line: 'No admitted document is held for this issuer; nothing is invented for it.', next: 'prepare, or admit a source'},
  prepared: {word: 'Prepared', tone: TONE.done, line: 'The unit\'s packet is prepared and readable; a prepared packet is not an analysis.'},
  analyzed: {word: 'Analyzed', tone: TONE.done, line: 'An analysis is published for the unit; an analysis is not a CRO decision.'},
  reviewed: {word: 'Reviewed', tone: TONE.done, line: 'A published review read this unit\'s analysis.'},
  evidence_reviewed: {word: 'Reviewed', tone: TONE.rest, line: 'The review read passages for this issuer; what it concluded is the report\'s.'}, // B1: the Overview's issuer row
  not_started: {word: 'Not started', tone: TONE.rest, line: 'No preparation has reached this unit.'},
  reused_local: {word: 'Reused', tone: TONE.rest, line: 'The body the workspace held was reused without a request, verified against its reference.'},
  fetched: {word: 'Fetched', tone: TONE.active, line: 'The body was transferred from the official locator in this check.'},
  complete: {word: 'Complete', tone: TONE.done, line: 'Every item the owner expected was delivered; nothing is said about what was not planned.'},
  unavailable: {word: 'Unavailable', tone: TONE.failure, line: 'Nothing expected was available; every failure is named.'},
  empty: {word: 'Empty', tone: TONE.rest, line: 'Nothing was expected and nothing was delivered.'},
  // the ledger's cells (round E2; the read model's delivery_state, contract section 3.1): the words a cell's code takes
  pending_delivery: {word: 'Pending', tone: TONE.rest, line: 'Routed; nothing of the sealed plan delivered yet.'},
  no_route: {word: 'No route', tone: TONE.rest, line: 'No region of the issuer\'s documents routes to the topic and no typed observation serves it; one bounded broader search ran. Not "nothing found".'},
  awaiting: {word: 'Awaiting', tone: TONE.attention, line: 'Waiting for an actor\'s answer; nothing runs.'},
  kept_acceptable: {word: 'Admitted with the acceptable items', tone: TONE.attention, line: 'The Host kept the acceptable items and dropped the rest; no more answers are read for this bundle.'}, // contract 10.4, the last answer read
  returned_for_correction: {word: 'Returned for correction', tone: TONE.attention, line: 'The Host read the answer and named its problems by item; nothing was admitted.', next: 'correct the named items and submit again'}, // contract 10.4
  // the reading's states (round E3): a passage's delivery, a table's progress, a typed statement, a continuation
  delivered: {word: 'Delivered', tone: TONE.done, line: 'The excerpt bytes are in the packet as sealed.'},
  served_by_later_reading: {word: 'Served by a later reading', tone: TONE.rest, line: 'The same text was delivered by a later reading of the same document; this row is context, not recent.'},
  unknown_progress: {word: 'Unknown progress', tone: TONE.rest, line: 'A page sealed before pages carried rows: nothing is proved and nothing is guessed.'},
  ambiguous: {word: 'Ambiguous', tone: TONE.attention, line: 'A sentence to read, with its reason; neither a clean nor a negative statement.'},
  nothing_resumable: {word: 'Nothing resumable', tone: TONE.rest, line: 'No continuation can read more; the remainders name what the plan leaves unread.'},
  // an issuer's review state (round E4; the Analyst's sealed completion per issuer) and a finding's answer
  source_missing: {word: 'No source held', tone: TONE.attention, line: 'No admitted document for the issuer; nothing was checked.'},
  no_spans_delivered: {word: 'No passages delivered', tone: TONE.attention, line: 'Documents held, but the plan delivered no passage to read.'},
  not_reported: {word: 'Not reported', tone: TONE.attention, line: 'Material delivered but no check answered for the issuer; never done.'},
  checks_incomplete: {word: 'Checks incomplete', tone: TONE.attention, line: 'Some required checks answered, some deferred or unreported.'},
  // a delivery fact, not a completed check (contract 10.7, record section AO): what reached the Analyst and what the answer reported
  executed_no_findings: {word: 'Read, no finding reported', tone: TONE.rest, line: 'Its passages reached the Analyst and the answer reported no finding for it; not a completed check and not clearance.'},
  executed_with_findings: {word: 'Read, findings reported', tone: TONE.rest, line: 'Its passages reached the Analyst and the answer reported findings for it.'},
  not_answered: {word: 'Not answered', tone: TONE.attention, line: 'The held assessment gives this finding no disposition yet; it stays open.'},
  // an evidence index's availability (round E5; the storage owner's word)
  available: {word: 'Available', tone: TONE.done, line: 'Present and proved; a prepared packet reads it.'},
  evicted_by_retention: {word: 'Evicted', tone: TONE.rest, line: 'Released by an approved cleanup; rebuilds from its committed vectors.'},
  missing_or_tampered: {word: 'Missing or tampered', tone: TONE.failure, line: 'The index file is not the one the record names; nothing reads it.'},
  admitted: {word: 'Admitted', tone: TONE.rest}, returned: {word: 'Returned', tone: TONE.rest}, requested: {word: 'Requested', tone: TONE.rest},
  recorded: {word: 'Recorded', tone: TONE.rest}, declared: {word: 'Declared', tone: TONE.rest}, payload_not_retained: {word: 'Payload not retained', tone: TONE.rest},
  candidate: {word: 'Candidate', tone: TONE.rest, line: 'A document the recorded package holds; not yet an admitted passage.'},
  live: {word: 'Live sources', tone: TONE.rest, line: 'Official documents acquired over the network under the admitted policy.'},
  published: {word: 'Published', tone: TONE.done, line: 'Sealed and published by its owner.'},
  superseded: {word: 'Superseded', tone: TONE.attention, line: 'The policy or the binding moved; a newer preparation is needed.'},
};
// An unknown state has no declared word; its exact code remains a fact rather than a guessed translation.
const stateOf = (code) => { const key = String(code ?? '').toLowerCase(); const s = STATES[key]; return s ? {key, next: '', line: '', ...s} : {key, word: code ? 'Word not declared' : '', tone: TONE.rest, line: '', next: ''}; };
const stateMoving = (code) => Boolean(stateOf(code).moving);
const stateHeld = (code) => Boolean(stateOf(code).held);
const STOP_WAYS = {'data.truth_review_required': 'decide on Data issues', 'data.remediation_wait': 'continue it when the wait is over'};
const wayOn = (code, state) => STOP_WAYS[code] ? t(STOP_WAYS[code]) : stateOf(state).next ? t(stateOf(state).next) : '';
/* A duration in the reader's units: seconds under a minute, minutes under an hour, then hours
 * and minutes. Never a fraction, never a progress. `coarse` (a run still moving) says minutes
 * only, so the words change once a minute and a poll repaints nothing in between. */
function durationText(ms, coarse = false) {
  const s = Math.max(0, Math.round(Number(ms) / 1000));
  if (!Number.isFinite(s)) return '';
  if (s < 60) return coarse ? t('under 1 min') : `${s} s`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min`;
  return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, '0')}`;
}
/* A moving run's elapsed time is the owner's finite nonnegative running_seconds, as of its
 * read; without that span nothing is inferred from the browser's clock. An ended run keeps
 * the span between the owner's two times (its start and its last activity). */
function durationOf(x) {
  if (stateMoving(x?.lifecycle || x?.state)) {
    const seconds = x?.timing?.running_seconds ?? x?.running_seconds;
    return Number.isFinite(seconds) && seconds >= 0 ? durationText(seconds * 1000, true) : '';
  }
  const start = Date.parse(x?.running_since || x?.started || ''), end = Date.parse(x?.last_activity_at || x?.finished || '');
  if (!Number.isFinite(start)) return '';
  return Number.isFinite(end) && end >= start ? durationText(end - start) : '';
}
const STAGES = {
  build_historical_tradability: {word: 'Build historical tradability'}, pre_research_delta_gate: {word: 'Pre research delta gate'},
  // the workspace preparation: the five steps, each with its note for the unprepared Home
  freeze_sources: {word: 'Freeze sources', note: 'Keep a reproducible copy of the selected source membership.', line: 'Capturing the current index memberships from the declared sources: the exact candidate list this preparation uses.'},
  prepare_data: {word: 'Prepare market data', note: 'Prices and corporate actions enter the working store.', line: 'Acquiring daily bars and corporate actions for every candidate into the working store, chunk by chunk; each chunk is kept on its own.'},
  prepare_features: {word: 'Prepare Features', note: 'Prepare current Features and immutable panel outputs.', line: 'Building the current Features and the immutable panel from the prepared bars: the longest step for a full universe.'},
  publish_inputs: {word: 'Publish research inputs', note: 'Publish an immutable research input, distinct from mutable work data.', line: 'Publishing one immutable research input from the panel; working data stays separate.'},
  verify_inputs: {word: 'Verify inputs', note: 'Admit the input for research only after verification.', line: 'Verifying the published input before it can be selected for research.'},
  validate_update_request: {word: 'Validate the request', note: 'Check the plan against the workspace as it stands now.'},
  maintain_data_feature: {word: 'Maintain data and Features', note: 'Fetch the new sessions for every admitted member, govern quality, maintain the Features and verify the Panel.'},
  publish_update_receipt: {word: 'Publish the receipt', note: 'Seal one receipt of what changed; working data stays separate from published inputs.'},
  // the Feature owner's work steps
  historical_audit_preflight: {word: 'Historical audit preflight'}, base_feature_materialization: {word: 'Feature materialization'}, panel_compute: {word: 'Panel computation'}, panel_year_finalize: {word: 'Panel year finalization'}, causal_execution_sources: {word: 'Outcome sources'}, causal_execution_outcome: {word: 'Outcome derivation'}, causal_execution_publication: {word: 'Outcome publication'}, panel_authority_cutover: {word: 'Panel authority cutover'}, panel_tail_cleanup: {word: 'Panel tail cleanup'}, market_data_increment: {word: 'Market-data increment'},
  // the evidence owner's stages and the CRO review's
  admit_evidence_request: {word: 'Admit the evidence request', line: 'The evidence owner checks the book, its issuer scope and the source policy; nothing is read yet.'},
  resolve_official_sources: {word: 'Resolve official sources'},
  acquire_source_evidence: {word: 'Obtain the source documents'},
  canonicalize_documents: {word: 'Canonicalize the documents', line: 'Documents are canonicalized and sectioned for exact citation.'},
  build_retrieval_generation: {word: 'Build the retrieval index', line: 'The retrieval index is built for the admitted documents; the embedding work of this stage is its cost.'},
  select_evidence_spans: {word: 'Select the admitted passages', line: 'Admitted passages are selected and given the only citable handles.'},
  analyze_evidence: {word: 'Validate the submitted answer', line: 'The submitted answer is validated against the packet: binding, schema and every cited handle. No actor runs; the answer came from outside the product.'},
  publish_evidence_analysis: {word: 'Publish the analysis', line: 'The validated analysis is sealed and published; it is not a CRO review.'},
  admit_portfolio_review: {word: 'Admit the dossier', line: 'The dossier, policy and schema bindings of the assessment are checked.'},
  seal_portfolio_review_assessment: {word: 'Seal the assessment', line: 'The assessment is sealed against the dossier with the decision policy.'},
  publish_portfolio_review: {word: 'Publish the review', line: 'The review is published as a Host-routed recommendation without action authority.'},
  prepare_component_training_inputs: {word: "Prepare component training inputs"},
  bind_research_training_sources: {word: "Bind research training sources"},
  materialize_local_formula_columns: {word: "Materialize local formula columns"},
  prepare_local_formula_overlay: {word: "Prepare local formula overlay"},
  materialize_verified_lifecycle_portfolio_inputs: {word: "Materialize verified lifecycle portfolio inputs"},
  publish_input_version: {word: "Publish input version"},
  execute_sealed_experiment: {word: "Execute the sealed experiment"},
  verify_experiment_evidence: {word: "Verify the experiment evidence"},
  verify_update: {word: "Verify update"},
  seal_market_observations: {word: "Seal market observations"},
  publish_decision_settlement: {word: "Publish decision settlement"},
  seal_observations: {word: "Seal observations"},
  compile_portfolio_input: {word: "Compile portfolio input"},
  publish_portfolio_input: {word: "Publish portfolio input"},
  verify_frozen_inputs: {word: "Verify frozen inputs"},
  materialize_features: {word: "Materialize features"},
  score_models: {word: "Score models"},
  publish_score_receipt: {word: "Publish score receipt"},
  verify_saved_studies: {word: "Verify saved studies"},
  execute_and_publish_declared_path: {word: "Execute and publish declared path"},
  verify_request: {word: "Verify request"},
  update_inputs: {word: "Update inputs"},
  seal_inputs: {word: "Seal inputs"},
  prepare_scores: {word: "Prepare scores"},
  prepare_calibration: {word: "Prepare calibration"},
  advance_book: {word: "Advance book"},
  publish_readback: {word: "Publish readback"},
  claim_protected_permit: {word: "Claim protected permit"},
  continue_protected_path: {word: "Continue protected path"},
  seal_pending_package: {word: "Seal pending package"},
  verify_protected_closure: {word: "Verify protected closure"},
  release_validated_handoff: {word: "Release validated handoff"},
};
// Coverage unit stages are an owner's bounded format, not arbitrary underscore suffixes.
const COVERAGE_STAGES = new Set(['admit_evidence_request', 'resolve_official_sources', 'acquire_source_evidence', 'canonicalize_documents', 'build_retrieval_generation', 'select_evidence_spans', 'analyze_evidence', 'publish_evidence_analysis']);
const stageWord = (id) => { const match = String(id ?? '').match(/^u[0-9]{2,3}_(.+)$/); return STAGES[id] || (match && COVERAGE_STAGES.has(match[1]) ? STAGES[match[1]] : null); };
const stageOf = (id) => stageWord(id) || {word: id ? 'Word not declared' : ''};
/* The lines of a set of stages (every stage with one, by default), for a work area's spec. */
const stageLines = (ids = Object.keys(STAGES)) => Object.fromEntries(ids.filter((id) => STAGES[id]?.line).map((id) => [id, STAGES[id].line]));
