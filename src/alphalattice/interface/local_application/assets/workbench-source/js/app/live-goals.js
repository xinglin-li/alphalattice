/* The goals (U23, U32; GR1, GR2; the user, 2026-09-28: 这个东西可以在UI里单独立一个section): a section of their own,
 * in place of the research case page. A goal is what is set out to be achieved and what completes it, kept by the
 * Host as immutable revisions; an agent takes it, works under it and submits it (through the CLI, never this page).
 * The list groups goals by state or kind. A goal opens in two owner reads -- its saved narrative paints at once,
 * then the verified readback re-reads every reference at its owner and replaces it -- under three folders: its
 * timeline (the Host's facts, the agents' messages and the decision notes), its conversation (the Team exchanges
 * made under it, each linking to its session; assignments a filter) and its results (the submission, its
 * deliverables and references). A person opens, revises, notes, attaches and abandons. */
const LiveGoals = (() => {
  const pages = new Set(['goals', 'goal', 'goal-conversation', 'goal-results', 'cases']); // `cases`: the research case page's old addresses
  const S = {hash: '', key: '', body: null, rows: null, error: '', loading: false, ticket: 0, verifying: false, verifyError: '', dirty: false, activity: null, draft: null, references: [], dialogHash: ''};
  const KINDS = [['RESEARCH', 'Research'], ['DATA', 'Data'], ['OPERATIONS', 'Operations'], ['REVIEW', 'Review']];
  // a first use is opened from the person's sentence, never in the composer (V452, OP19): its word beside the composer's kinds
  const KIND_WORDS = [...KINDS, ['FIRST_USE', 'First use']];
  const kindWord = (kind) => t((KIND_WORDS.find(([k]) => k === kind) || [])[1] || '') || codeWords(kind);
  const PURPOSES = [['NEW_RESEARCH', 'New research'], ['EXISTING_RESULTS', 'Interpret existing results'], ['CONTINUATION', 'Continue prior research']];
  const STAGES = ['DATA_FEATURES', 'FACTOR_FOUNDATION', 'ALPHA', 'RISK', 'PORTFOLIO', 'EVIDENCE_CRO'];
  const passComplete = (b) => Boolean(b) && b.evidence_verification === 'COMPLETE';
  const byId = (value) => /^[0-9a-f]{8}-[0-9a-f]{4}-/i.test(value); // a goal named by its id (a Team link) opens its current revision
  const addressed = () => hashParams().get('goal') || hashParams().get('case') || '';
  const selected = () => pages.has(app.page) && Boolean(addressed());
  const listing = () => (app.page === 'goals' || app.page === 'cases') && !addressed();
  const fieldOf = (id, label, control) => html`<div class="field"><label for="${id}">${t(label)}</label>${control}</div>`;
  // A saved revision's conversation is mutable. Observe the existing feed by its exact Goal
  // UUID, without joining unbound/same-book messages or opening another polling loop.
  // This reader's own operation receipts cannot make its saved record dirty again.
  const GOAL_READS = new Set(['GOAL_SCHEMA', 'GOAL_LIST', 'GOAL_NARRATIVE', 'GOAL_SHOW', 'GOAL_REFERENCE', 'GOAL_EXPORT', 'GOAL_CONTINUE', 'CASE_NARRATIVE', 'CASE_READBACK', 'CASE_EXPORT']);
  function activitySnapshot() {
    if (typeof LiveActivity === 'undefined' || !LiveActivity.state || !LiveActivity.retained) return null;
    const state = LiveActivity.state();
    if (!state.epoch || state.stale || state.error || state.disposition === 'UNAVAILABLE') return null;
    return {epoch: state.epoch, gap: state.disposition === 'RESET' || (state.disposition === 'TAIL' && Boolean(state.notice)), items: LiveActivity.retained().flatMap(g => g.items || []).filter(item => item.payload?.subject?.goal_id && !(item.schema_kind === 'ProductOperationObserved' && GOAL_READS.has(item.payload.operation))).map(item => [item.payload.subject.goal_id, item.ordinal])};
  }
  const activityStamp = (snapshot, id) => snapshot ? {epoch: snapshot.epoch, gap: snapshot.gap, ordinal: Math.max(0, ...snapshot.items.filter(([goal]) => goal === id).map(([, ordinal]) => ordinal))} : null;
  function activityChanged(snapshot) {
    const next = activityStamp(snapshot, S.body?.goal?.goal_id), was = S.activity;
    if (!next) return false;
    const changed = !was || next.epoch !== was.epoch || next.ordinal !== was.ordinal || (next.gap && !was.gap);
    // A continued page rearms the next discontinuity. Cursor movement within one gap,
    // including this reader's own receipts, cannot launch another verification pass.
    if (was && !next.gap) was.gap = false;
    return changed;
  }
  async function ensure() {
    if (app.page === 'cases') { const h = addressed(); app.page = h ? 'goal' : 'goals'; replaceHash({page: app.page, goal: h, case: ''}); } // an old research case address opens its goal
    else if (pages.has(app.page) && hashParams().has('case')) replaceHash({goal: addressed(), case: ''}); // every old Goal folder uses the same canonical object key before its tabs draw
    if (pages.has(app.page) && app.page !== 'goals' && !addressed()) { app.page = 'goals'; replaceHash({page: 'goals'}); }
    if (!selected() && !listing()) return;
    const hash = addressed(), key = hash || 'list';
    if (S.key !== key) { S.key = key; S.ticket++; S.loading = false; S.verifying = false; S.error = ''; S.verifyError = ''; }
    const activity = activitySnapshot();
    if (hash && S.hash === hash && S.body && activityChanged(activity)) { S.dirty = true; S.error = ''; }
    if (S.loading || S.error || (S.verifying && S.dirty)) return;
    if (hash && S.hash === hash && S.body && !S.dirty) {
      // a narrative whose verification was dropped by navigation is verified now; a failed one waits for an explicit retry
      if (!passComplete(S.body) && !S.verifying && !S.verifyError) await verify();
      return;
    }
    if (!hash && S.rows) return;
    const ticket = ++S.ticket; S.loading = true; S.error = ''; S.dirty = false;
    S.activity = activityStamp(activity, S.hash === hash ? S.body?.goal?.goal_id : byId(hash) ? hash : null);
    try {
      const b = await Data.read(hash ? '/api/goals/narrative?' + new URLSearchParams(byId(hash) ? {goal_id: hash} : {goal_hash: hash}) : '/api/goals');
      if (ticket !== S.ticket) return;
      if (hash) { S.hash = hash; S.body = b; S.activity = activityStamp(activity, b.goal?.goal_id); } else S.rows = b;
    } catch (e) { if (ticket === S.ticket) S.error = e.message; }
    finally { if (ticket === S.ticket) { S.loading = false; render(); } }
    if (hash && ticket === S.ticket && S.body) await verify();
    if (hash && ticket === S.ticket && S.dirty) void ensure();
  }
  function observe() {
    if (!selected() || S.hash !== addressed() || !S.body) return;
    if (activityChanged(activitySnapshot())) { S.dirty = true; S.error = ''; }
    if (S.dirty && !S.loading && !S.verifying && !S.error) return ensure();
  }
  /* The owner's strict readback: every reference re-read at its owner; a failure keeps the saved narrative on
   * screen and says so; nothing retries on its own. */
  async function verify() {
    if (!S.hash || S.verifying) return;
    const hash = S.hash, ticket = S.ticket;
    S.verifying = true; S.verifyError = ''; render();
    try {
      const b = await Data.read('/api/goals/show?' + new URLSearchParams(byId(hash) ? {goal_id: hash} : {goal_hash: hash}));
      if (ticket !== S.ticket || S.hash !== hash) return;
      S.body = b;
    } catch (e) { if (ticket === S.ticket) S.verifyError = e.message; }
    finally { if (ticket === S.ticket) { S.verifying = false; render(); } }
  }
  /* The completed pass and the references' actual states are separate owner facts. */
  const evidenceState = (b) => {
    const check = goalReferenceIntegrity(b);
    if (!(b.references || []).length) return stateLine(check.state, {word: check.word, next: ''});
    const state = S.verifying ? 'running' : S.verifyError ? 'failed' : check.state;
    const word = S.verifying ? t('Checking reference integrity') : S.verifyError ? t('Reference integrity not verified') : check.word;
    return hint(stateLine(state, {word, next: ''}), check.limit);
  };
  const GOAL_STATES = {OPEN: ['pending', 'state|Open'], COMPLETE: ['succeeded', 'Record complete'], ABANDONED: ['cancelled', 'Abandoned']};
  const OUTCOMES = {ACHIEVED: 'Achieved', PARTLY_ACHIEVED: 'Partly achieved', NOT_ACHIEVED: 'Not achieved'};
  const outcomeWord = (outcome) => (outcome ? t(OUTCOMES[outcome] || '') || codeWords(outcome) : '');
  // U54: a listed goal's check, as the Team's outputs show it -- the Host's state, Complete once it found the record complete
  const checkLine = (g) => { const [s, w] = GOAL_STATES[g.state] || ['metadata', '']; return stateLine(s, {word: t(w) || codeWords(g.state), next: ''}); };
  /* Goals as a lobby (F2, law 136): grouped by state -- the open ones open, the closed folded -- or by kind; one
   * line a row: the title, its kind, revision and references, the day. The owner's older pages are read from the foot. */
  const RANK = {OPEN: 0, COMPLETE: 1, ABANDONED: 2};
  /* U23: a goal by the agent sessions its row names (the Host's `sessions`, the list its record computes); a goal
   * several sessions worked on is one group of them all, one no session took yet a group of its own. */
  const sessionWords = (x) => `${codeWords(x.vendor)} ${short(x.session_id)}`;
  const sessionGroup = (g) => { const all = g.sessions || []; return all.length ? {key: all.map((x) => x.vendor + ':' + x.session_id).join(','), label: all.map(sessionWords).join(' · '), rank: all.length, open: true} : {key: '', label: t('No session yet'), rank: Infinity, open: false}; };
  function lobby(rows) {
    const kinds = [...new Set(rows.goals.map((g) => g.kind).filter(Boolean))];
    const row = (g, d) => { const on = (k) => d.props[k] !== false; return objectRow({name: g.title, why: g.objective, ref: short(g.goal_hash, SHORT.hash), to: {action: 'goal-open', value: g.goal_hash}}, {key: g.goal_id, columns: ['kind', 'revision', 'references'], props: [on('kind') ? kindWord(g.kind) : '', on('revision') ? t('revision {n}', {n: g.revision}) : '', on('references') ? countText(g.reference_count, '{n} reference', '{n} references') : ''], time: when(g.recorded_at)}); };
    const state = (g) => ({key: g.state, label: t((GOAL_STATES[g.state] || [])[1] || '') || codeWords(g.state), rank: RANK[g.state] ?? 9, open: g.state === 'OPEN'});
    return Lobby.render('goals', {items: rows.goals, row, axes: [{key: 'state', label: t('State'), group: state}, {key: 'kind', label: t('Kind'), group: (g) => ({key: g.kind, label: kindWord(g.kind), rank: kinds.indexOf(g.kind), open: true})}, {key: 'session', label: t('Agent session'), group: sessionGroup}, {key: 'time', label: t('Time'), group: (g) => timeGroup(g.recorded_at)}],
      words: (g) => [g.title, g.objective, g.goal_id, g.goal_hash].join(' '), placeholder: t('Title, objective or reference'),
      properties: [['kind', t('Kind')], ['revision', t('Revision')], ['references', t('References')]],
      cls: 'lines', // one line a row (LS2): the objective cut beside its title, whole on hover
      foot: rows.next_cursor ? Lobby.older(t('Older goals are not read yet'), 'goal-more', rows.next_cursor, Boolean(S.paging)) : ''});
  }
  function refusedGoals(rows) {
    const refused = rows?.refused || [];
    if (!refused.length) return '';
    return panel(t('Unreadable goal records'), '', html`<div class="card-list">${refused.map((g) => html`<div>
      <h3>${t('Goal')} ${hashCell(g.goal_id, SHORT.id)}</h3>
      ${refusal({failure_code: g.failure_code, reason: g.detail}, TONE.attention, {state: 'refused', word: t('Refused'), catalog: true, next: link(t('Storage & retention'), 'storage', 'text-btn')})}
    </div>`)}</div>`);
  }
  function open(hash) { objectEntry('goal:' + hash); S.error = ''; navigate('goal', {goal: hash, case: ''}, {replace: true}); }
  function list() { S.rows = null; S.error = ''; navigate('goals', {goal: ''}); }
  /* The goal's head, the same on each folder: its title, its objective as the lede, its completion criteria, its
   * state and what is missing; the ways a person changes it in the tools. */
  function head(b) {
    const g = b.goal, d = g.declaration, isOpen = g.state === 'OPEN', gaps = passComplete(b) ? (b.gaps || []) : [], firstUse = d.kind === 'FIRST_USE';
    const tools = [...(isOpen ? [...(firstUse ? [] : [{ic: 'edit', action: 'goal-edit', word: t('Revise goal'), why: t('A new revision is recorded')}]), {ic: 'plus', action: 'goal-attach', word: t('Attach exact evidence'), why: t('A saved owner read, verified before it is kept')}, ...(firstUse ? [] : [{ic: 'stop', action: 'goal-abandon', word: t('Abandon goal'), why: t('Closes it; a follow-up opens a new goal')}])] : []),
      {ic: 'file', action: 'goal-export', value: 'json', word: t('Export JSON'), why: t('The exact goal')}, {ic: 'file', action: 'goal-export', value: 'html', word: t('Export HTML'), why: t('The exact goal')}, {ic: 'file', action: 'goal-export', value: 'yaml', word: t('Export YAML'), why: t('Its declaration')}, {ic: 'history', action: 'goal-list', word: t('Goals'), why: t('Every saved goal')}];
    const primary = !passComplete(b) && !S.verifying ? btn(t('Check reference integrity'), 'goal-verify', '', 'button primary') : isOpen ? btn(t('Add a decision note'), 'goal-note', '', 'button primary') : '';
    const facts = [[t('Kind'), kindWord(d.kind)], [t('Revision'), html`${g.revision}${g.parent_hash ? html` · ${btn(t('previous'), 'goal-open', g.parent_hash, 'inline-link')}` : ''}`], [t('Criteria'), count(d.criteria.length)], [t('Evidence'), evidenceState(b)], ...(gaps.length ? [[t('Missing'), count(gaps.length)]] : []), [t('Recorded'), when(g.recorded_at)], [t('Revision recorded by'), actorWords(g.submitted_by)]];
    const criteria = kv(d.criteria.map((c) => [html`<span class="mono">${c.criterion_id}</span>`, html`<span class="owner-text">${c.text}</span>`]), 'kv-columns');
    const historical = g.goal_hash !== b.head_hash ? noteLine(t('Historical revision'), t('This exact saved revision is open; revising needs the current one.'), 'warning', btn(t('Open current revision'), 'goal-open', b.head_hash, 'button compact')) : '';
    const missing = gaps.length ? noteLine(t('Missing'), gaps.map((x) => x.stage ? codeWords(x.stage) : x.reference_id).join(' · '), 'warning') : '';
    return html`${objectHead(d.title, html`<p class="lede owner-text">${d.objective}</p>${criteria}`, primary, checkLine(g), tools, {object: true, id: g.goal_hash, facts, scope: {name: d.title, href: routeUrl('goal', {goal: addressed()}), self: app.page === 'goal'}})}${historical}${missing}`;
  }
  function page() {
    if (!selected() && !listing()) return null;
    const top = S.error ? banner(t('Goal unavailable'), S.error, 'warning') : '';
    if (listing()) {
      const hasGoals = Boolean(S.rows?.goals?.length), hasRefused = Boolean(S.rows?.refused?.length);
      const create = hasGoals || hasRefused ? btn(html`${icon('plus')}${t('New goal')}`, 'goal-new', '', 'button') : '';
      return html`${top}${objectHead(t('Goals'), t('A goal is what is set out to be achieved and what completes it: its objective, its criteria and the deliverables that answer them. An agent takes it and submits it; each change is a new revision.'), create)}${S.rows ? refusedGoals(S.rows) : ''}${hasGoals ? lobby(S.rows) : S.rows && !hasRefused ? emptyState(t('No goal yet'), btn(t('New goal'), 'goal-new', '', 'button primary'), 'page-empty') : ''}${S.rows ? '' : skeleton()}`;
    }
    const b = S.body; if (!b || S.hash !== addressed()) return html`${top}${skeleton()}`;
    const folder = app.page === 'goal-conversation' ? conversation(b) : app.page === 'goal-results' ? results(b) : timeline(b);
    return html`${top}${head(b)}${firstUse(b)}${folder}`;
  }
  /* U70 (V452, OP19): a first use is the person's, run by their agent from their sentence (the lede): each step it took
   * as theirs (`record.delegated_steps`, the goal's ledger), the hours its delegation has left (`record.delegation`),
   * and one Stop that ends it -- the goal abandoned, the network it opened closed by the Host. */
  const STEP_WORDS = {WORKSPACE_PREPARE_CONFIRM: 'Confirmed the preparation', DATA_ISSUE_CONFIRM: 'Decided a data issue'};
  const stepWords = (s) => (s.operation === 'NETWORK_ACCESS_SET' ? t(s.network_enabled ? 'Opened the network' : 'Closed the network') : STEP_WORDS[s.operation] ? t(STEP_WORDS[s.operation]) : codeWords(s.operation));
  function hoursLeft(delegation) {
    if (!delegation?.ends_at) return '';
    const left = Date.parse(delegation.ends_at) - Date.now();
    return delegation.active && left > 0 ? t('{n} h left', {n: count(Math.ceil(left / 3600000))}) : t('its hours have ended');
  }
  function firstUse(b) {
    const g = b.goal, r = b.record || {}, steps = r.delegated_steps || [];
    if (g.declaration.kind !== 'FIRST_USE') return '';
    const hours = g.state === 'OPEN' ? hoursLeft(r.delegation) : '';
    const rows = steps.map((s) => [stepWords(s), html`${when(s.recorded_at)}${s.status && s.operation !== 'NETWORK_ACCESS_SET' ? html` · ${codeWords(s.status)}` : ''}`]); // a network step's words say what it set
    return panel(html`${t('Your first use')}${hours ? html` · ${hours}` : ''}`, t('Opened from your sentence: the agent running it takes the first steps as yours, for its hours; Stop ends them.'),
      steps.length ? kv(rows, 'kv-columns') : emptyState(t('No step taken for you yet.')), g.state === 'OPEN' ? btn(t('Stop'), 'goal-first-use-stop', '', 'button compact') : '', 'data-first-use');
  }
  /* The timeline: the Host's facts (registration, the revision, the Tasks and sessions it recorded), the agents'
   * messages as they arrived and the decision notes; the notes whole in their thread, the sessions' usage (U32). */
  function timeline(b) {
    const g = b.goal, r = b.record || {}, by = actorWords(g.submitted_by);
    const said = (m) => [m.agent_id, m.role ? codeWords(m.role) : ''].filter(Boolean).join(' · ') || t('Agent');
    const lines = [logLine({at: when(g.intent_registered_at || g.recorded_at), by, words: t('Goal registered')}), logLine({at: when(g.recorded_at), by, words: html`${t('Revision {n} recorded', {n: g.revision})}${g.change_reason ? html` · <span class="owner-text">${g.completion ? t(g.change_reason) : g.change_reason}</span>` : ''}`}),
      ...[...(r.conversation || [])].sort((a, c) => String(a.recorded_at).localeCompare(String(c.recorded_at))).map((m) => logLine({at: when(m.recorded_at), by: said(m), words: html`${codeWords(m.message_kind)}${m.summary ? html` · <span class="owner-text">${m.summary}</span>` : ''}`, code: m.message_id})),
      ...g.references.map((x) => logLine({by, words: html`${t('Reference attached')} · ${x.label}`, code: x.reference_id})),
      ...g.statements.map((s) => logLine({by: s.attribution, words: html`${codeWords(s.kind)} · ${codeWords(s.disposition)}`, code: s.statement_id})),
      ...(g.completion ? [logLine({at: when(g.completion.checked_at), by: t('Host'), words: t('Submission checked')})] : []),
      logLine({by: t('This page'), words: passComplete(b) ? html`${goalReferenceIntegrity(b).word}${goalReferenceIntegrity(b).limit ? html` · ${goalReferenceIntegrity(b).limit}` : ''}` : (b.references || []).length ? t(S.verifyError ? 'Verification failed; references shown as saved' : 'References shown as saved; none re-read at its owner') : t('Nothing to check yet')})];
    const tasks = (r.tasks || []).map((x) => objectRow({state: String(x.state).toLowerCase(), name: codeWords(x.kind), to: {action: 'task', value: x.task_id}}, {key: x.task_id, columns: ['id'], props: [html`<span class="mono">${short(x.task_id)}</span>`]}));
    const facts = panel(t('Host facts'), t('What the Host recorded under this goal; an agent\'s words are its own.'), html`${kv([[t('Sessions'), (r.sessions || []).length ? (r.sessions || []).map((s) => html`${codeWords(s.vendor)} <span class="mono">${short(s.session_id)}</span>`).reduce((a, x, i) => html`${a}${i ? ' · ' : ''}${x}`, '') : t('none bound')], [t('Requests'), count(r.request_count ?? 0)], [t('Messages'), count(r.message_count ?? 0)]], 'kv-columns')}${tasks.length ? html`<div class="card-list lines slotted">${tasks}</div>` : ''}`);
    const notes = panel(html`${t('Decision notes')} <span class="num">${count(g.statements.length)}</span>`, t('An attributed decision or conclusion with its evidence standing; a note is not product validation or scientific approval.'), g.statements.length ? html`<div class="case-thread">${g.statements.map((s) => statement(s, b.statement_context?.[s.statement_id]))}</div>` : emptyState(t('No decision note yet.'), g.state === 'OPEN' ? btn(t('Add a decision note'), 'goal-note', '', 'button primary') : ''), g.statements.length && g.state === 'OPEN' ? btn(t('Add a decision note'), 'goal-note', '', 'button compact') : '');
    return html`${notes}${facts}${usage(r.session_usage || [])}${runLog({id: 'goalTimeline', title: t('Timeline'), lines, count: lines.length})}`;
  }
  /* U32: each session's models and tokens under this goal -- its totals by model, each participant's latest
   * reading by model, a mark where a reading differs from the agent's pinned card. A session's totals are its
   * own so far, whatever else it worked on; the Host says so. */
  function usage(sessions) {
    if (!sessions.length) return '';
    const tokens = (x) => t('{i} in · {o} out · {r} cache read · {w} cache written', {i: count(x.input_tokens), o: count(x.output_tokens), r: count(x.cache_read_tokens), w: count(x.cache_write_tokens)});
    const rows = sessions.map((s) => html`<h3>${codeWords(s.vendor)} <span class="mono">${short(s.session_id)}</span></h3>${kv([...s.by_model.map((m) => [html`<span class="mono">${m.model}</span>`, html`${tokens(m)} · ${countText(m.responses, '{n} response', '{n} responses')}`]), ...s.participants.map((p) => [html`${p.agent_id}${p.role ? html` · ${codeWords(p.role)}` : ''}`, html`${p.models.map((m) => html`<span class="mono">${m.model}</span>${m.efforts.length ? html` (${m.efforts.join(', ')})` : ''}`).reduce((a, x, i) => html`${a}${i ? ' · ' : ''}${x}`, '')}${p.pin_differs.length ? html` · ${badge('blocked', t('differs from its card: {what}', {what: p.pin_differs.map((x) => codeWords(x)).join(', ')}))}` : ''}`])], 'kv-columns')}`);
    return panel(t('Sessions and usage'), t('Each session\'s totals so far, whatever else it worked on; tokens are not attributed to a goal or a Task.'), html`${rows}`);
  }
  /* The conversation: the Team exchanges made under the goal, the newest first, each linking to its session;
   * the kind a filter (assignments one of them), the open assignments counted. */
  function conversation(b) {
    const r = b.record || {}, messages = [...(r.conversation || [])].reverse(), openAssignments = (r.open_assignments || []).length;
    if (!messages.length) return emptyState(html`${t(b.goal.state === 'OPEN' ? 'No Team exchange under this goal yet' : 'No recorded conversation under this goal')}${infoMark(t('No Team exchange is retained with this goal\'s explicit binding. Recorded deliverables are in Results; goal changes are in Timeline. Unbound sessions and unobserved native history are not added here.'))}`, html`${btn(t('Results'), 'go', 'goal-results', 'text-btn')}${btn(t('Timeline'), 'go', 'goal', 'text-btn')}`, 'page-empty', 'elsewhere');
    // Bound messages supply authors; recipient locators use Team's recorded exact-session binding.
    const sessions = LiveTeam.scene().sessions;
    const nameOf = (id, m, recipient = false) => {
      const roles = id && m.agent_vendor && m.agent_session ? [...new Set(messages.filter((x) => x.agent_id === id && x.agent_vendor === m.agent_vendor && x.agent_session === m.agent_session).map((x) => x.role).filter(Boolean))] : [];
      const session = recipient && m.agent_vendor === 'codex' ? sessions.find((s) => s.id === m.agent_session) : null;
      return html`<span tabindex="0" data-tip="${id || ''}">${roles.length ? roles.map(LiveTeam.roleName).join('/') : session ? LiveTeam.recipientName(session, id) : t('Unidentified')}</span>`;
    };
    const kinds = [...new Set(messages.map((m) => m.message_kind).filter(Boolean))];
    const acceptedClock = (m) => m.message_kind === 'answer' && m.input_channel === 'PRODUCT_ACCEPTED_ANSWER' && ['TASK_ADMISSION', 'PRODUCT_ACCEPTED_AT'].includes(m.source_time_kind) && m.occurred_at;
    const messageTime = (m) => acceptedClock(m) ? m.occurred_at : m.recorded_at;
    const row = (m) => {
      const accepted = m.message_kind === 'answer' && m.input_channel === 'PRODUCT_ACCEPTED_ANSWER';
      return objectRow({name: m.summary ? html`<span class="owner-text">${m.summary}</span>` : codeWords(m.message_kind), to: m.agent_session ? {page: 'team', extra: {team: m.agent_session, actor: '', event: m.observation_id || ''}} : null}, {key: m.message_id || m.observation_id, columns: ['kind', 'actor', 'recipient', 'submitter', 'reply'], props: [accepted ? hint(t('Accepted answer'), t('Accepted structured answer · product record, not a native spoken turn')) : codeWords(m.message_kind), nameOf(m.agent_id, m), m.recipient_id ? html`${t('to')} ${nameOf(m.recipient_id, m, true)}` : '', accepted ? html`${t('Submitted by')} ${nameOf(m.submitted_by, m)}` : '', m.reply_to ? html`${t('answers')} <span class="mono">${short(m.reply_to)}</span>` : ''], time: acceptedClock(m) ? hint(when(messageTime(m)), `${t(m.source_time_kind === 'PRODUCT_ACCEPTED_AT' ? 'Answer acceptance time' : 'Task admission time')} · ${t('Recorded')} ${whenText(m.recorded_at)}`) : when(messageTime(m))});
    };
    const foot = (r.message_count ?? messages.length) > messages.length ? html`<p class="caption">${t('The newest {n} of {m} messages; the Team session holds them all.', {n: count(messages.length), m: count(r.message_count)})}</p>` : '';
    return html`${openAssignments ? noteLine(t('Open assignments'), countText(openAssignments, '{n} assignment no reply names yet', '{n} assignments no reply names yet'), 'neutral') : ''}${Lobby.render('goal-conversation', {items: messages, row, axes: [{key: 'time', label: t('Time'), group: (m) => timeGroup(messageTime(m))}],
      words: (m) => [m.summary, m.agent_id, m.role, m.message_kind].join(' '), placeholder: t('Words, agent or kind'),
      filters: kinds.length > 1 ? [{field: 'kind', label: t('Kind'), multiple: true, options: kinds.map((k) => [k, codeWords(k)]), test: (m, one) => m.message_kind === one}] : []})}${foot}`;
  }
  /* The results: the submission (its outcome, summary, each criterion's answer, the deliverables and findings,
   * what stayed unresolved), the Host's completion facts, and the references with their verification. */
  function results(b) {
    const g = b.goal, sub = g.submission, d = g.declaration;
    const criterion = (id) => d.criteria.find((c) => c.criterion_id === id)?.text || id;
    const submission = sub ? panel(html`${t('Submission')} · ${t(OUTCOMES[sub.outcome] || '') || codeWords(sub.outcome)}`, t('The agent\'s judgment; the Host checked it against its own record, never the truth of its summary.'), html`<p class="owner-text">${sub.summary}</p>${kv([
      ...sub.criteria.map((a) => [html`<span class="owner-text">${criterion(a.criterion_id)}</span>`, html`${codeWords(a.answer)}${a.evidence.length ? html` · ${a.evidence.map((h) => citePill(h, false))}` : ''}${a.note ? html` · <span class="owner-text">${a.note}</span>` : ''}`]),
      ...sub.deliverables.map((x) => [html`<span class="owner-text">${d.deliverables.find((v) => v.deliverable_id === x.deliverable_id)?.description || x.deliverable_id}</span>`, html`${x.references.map((h) => citePill(h, false))}`]),
      ...sub.findings.map((f) => [t('Finding'), html`<span class="owner-text">${f.text}</span>`]),
      ...sub.problems.map((p) => [t('Unresolved'), html`<span class="owner-text">${p.text}</span>${p.task_ids.length ? html` · ${p.task_ids.map((id) => btn(short(id), 'task', id, 'inline-link'))}` : ''}`]),
      ...sub.follow_ups.map((f) => [t('Follow-up'), html`<span class="owner-text">${f}</span>`]),
      ...sub.files.map((f) => [t('File'), html`${f.name} <span class="sub-cell">${f.media_type}</span>`])], 'kv-columns')}`) : emptyState(t(g.state === 'OPEN' ? 'Not submitted yet: an agent submits it through the CLI.' : 'Closed without a submission.'), btn(t('Timeline'), 'go', 'goal', 'text-btn'), '', 'elsewhere');
    const completion = g.completion ? kv([[t('Checked'), when(g.completion.checked_at)], [t('Sessions'), count(g.completion.sessions.length)], [t('Tasks'), count(g.completion.tasks.length)], [t('Requests'), count(g.completion.request_count)]], 'kv-columns') : '';
    const refs = b.references || [];
    const verification = passComplete(b) || S.verifying ? '' : S.verifyError ? refusal({reason: S.verifyError}, 'warning', {state: 'failed', word: t('Reference integrity not verified'), action: btn(t('Check reference integrity'), 'goal-verify', '', 'button compact')}) : '';
    const references = panel(html`${t('References')} <span class="num">${count(refs.length)}</span>`, t(!refs.length ? 'Nothing to check yet' : passComplete(b) ? 'Reference integrity only; not scientific approval.' : 'As recorded; not re-read at their owners yet.'), html`${verification}${refs.length ? html`<div class="card-list lines">${refs.map((r, i) => evidenceRow('reference', r, {to: {action: 'goal-evidence', value: r.reference.reference_id}, actions: r.reference.request.operation === 'EXPERIMENT_READBACK' && g.state === 'OPEN' ? btn(t('Continue as a new draft'), 'goal-continue', String(i), 'menu-row') : '', columns: ['summary'], props: [r.summary ? codeRef(t('Recorded evidence and limitations'), r.summary) : '']}))}</div>` : emptyState(t('No evidence attached yet'), g.state === 'OPEN' ? btn(t('Attach exact evidence'), 'goal-attach', '', 'button primary') : '')}`, refs.length && g.state === 'OPEN' ? btn(t('Attach exact evidence'), 'goal-attach', '', 'button compact') : '');
    return html`${submission}${completion ? panel(t('Completion'), t('The Host\'s own facts when it sealed the goal.'), completion) : ''}${references}`;
  }
  function statement(s, context) { const body = html`${context?.design === 'PRIOR_DESIGN' ? noteLine(t('Earlier design'), html`<span class="owner-text">${context.objective}</span>`, 'warning') : ''}<p class="owner-text">${s.text}</p><p class="caption">${t('Evidence')}: ${s.evidence.length ? s.evidence.map((h) => citePill(h, false)) : ''}${s.responds_to ? html` · ${t('Response to')}: ${s.responds_to}` : ''}</p>`; return html`<article class="case-comment"><header><strong>${s.attribution}</strong>${stateLine(s.disposition === 'OPEN' ? 'review_pending' : s.disposition === 'SUPPORTS' ? 'verified' : s.disposition === 'REJECTED' || s.disposition === 'DOES_NOT_SUPPORT' ? 'blocked' : 'metadata', {word: codeWords(s.disposition), next: ''})}</header>${body}</article>`; }
  /* The Facts panel (the inspector): the exact revision and the verification counts. */
  const hasFacts = () => Boolean(selected() && S.body && S.hash === addressed());
  function factsSections() {
    const b = S.body, g = b.goal;
    return [{title: t('Exact revision'), body: html`${kv([[t('Goal'), html`<span class="mono">${g.goal_id}</span>`], [t('Previous revision'), g.parent_hash ? mono(g.parent_hash, SHORT.hash) : ''], [t('Registered'), whenText(g.intent_registered_at)], [t('Revision recorded by'), actorWords(g.submitted_by)], [t('Change reason'), g.change_reason || '']])}${codeRef(t('Read the exact revision (JSON)'), g)}`}];
  }
  /* A goal is written in a composer: the objective and its title, the kind as a chip, the criteria one a line,
   * the rest folded. A new goal's answer is the `/goal ` prompt that hands it to Claude or Codex. */
  const example = () => ({title: '', objective: '', kind: 'RESEARCH', scope: '', constraints: [], criteria: [{criterion_id: 'c1', text: ''}], deliverables: [], budget: null, research: {purpose: 'NEW_RESEARCH', comparison_design: '', required_stages: ['FACTOR_FOUNDATION']}, parent_goal_id: null});
  function edit(fresh = false) {
    const g = S.body?.goal;
    S.draft = fresh ? {id: '', hash: null, doc: example()} : {id: g.goal_id, hash: g.goal_hash, doc: JSON.parse(JSON.stringify(g.declaration))};
    const d = S.draft.doc;
    const text = (id, label, value) => fieldOf(id, label, html`<textarea id="${id}" class="ui-field" rows="3">${value}</textarea>`);
    const kinds = KINDS.map(([k, l]) => [k, t(l)]);
    const kindChip = propertyChip(t('Kind'), kinds.find(([k]) => k === d.kind)?.[1] || '', {icon: 'flag', rows: kinds.map(([value, title]) => ({value, title})), context: t(fresh ? 'New goal' : 'Revise goal'), action: 'goal-kind', current: d.kind, id: 'goalKindChip'});
    const research = d.research || {purpose: 'NEW_RESEARCH', comparison_design: '', required_stages: []};
    const more = html`<details class="reveal-details composer-more"><summary>${t('Criteria, scope and research design')}</summary>
      ${text('goalCriteria', 'Completion criteria · one per line', d.criteria.map((c) => c.text).join('\n'))}
      ${text('goalScope', 'Scope', d.scope)}${text('goalConstraints', 'Constraints · one per line', d.constraints.join('\n'))}
      ${fieldOf('goalPurpose', 'Research purpose', picker('goalPurpose', PURPOSES.map(([k, l]) => [k, t(l)]), {selected: research.purpose, label: t('Research purpose')}))}
      ${text('goalComparison', 'Comparison design', research.comparison_design)}
      <div class="composer-stages">${STAGES.map((stage) => html`<label data-tip="${stage}"><input type="checkbox" data-goal-stage="${stage}"${research.required_stages.includes(stage) ? ' checked' : ''}> ${codeWords(stage)}</label>`)}</div>
      ${fresh ? '' : fieldOf('goalReason', 'Change reason', html`<input id="goalReason" class="ui-field" value="">`)}
      </details>`;
    openDialog(t('Goal'), t(fresh ? 'New goal' : 'Revise goal'), html`${composer({context: Data.workspace(), title: {id: 'goalTitle', label: t('Title'), value: d.title}, body: {id: 'goalObjective', label: t('Objective'), value: d.objective}, chips: kindChip, more, note: t('Saves the goal only. No Task or numerical work is started; an agent takes it and submits it.')})}<div id="goalSaveRefusal" hidden></div>`,
      html`<span class="dialog-foot-note">${keycap('Ctrl Enter')} ${t('saves')}</span>${btn(t('Save goal'), 'goal-save', '', 'button primary')}`);
    $('#dialog').classList.add('composer-dialog');
    $('#goalTitle')?.focus();
  }
  function setKind(value) { if (!S.draft) return; S.draft.doc.kind = value; const chip = $('#goalKindChip'); if (chip) { const words = chip.querySelector('.filter-chip-main strong'); const label = KINDS.find(([k]) => k === value)?.[1]; if (words && label) words.textContent = t(label); chip.querySelectorAll('.menu-row').forEach((r) => r.setAttribute('aria-checked', String(r.dataset.value === value))); } }
  async function saveRequest(path, payload) {
    const slot = $('#goalSaveRefusal');
    if (slot) { slot.hidden = true; slot.innerHTML = ''; }
    for (const field of $$('#dialog [aria-invalid]')) { field.removeAttribute('aria-invalid'); field.removeAttribute('aria-describedby'); }
    try { return await Data.post(path, payload); }
    catch (error) {
      const body = error.body || {failure_code: 'Request refused', message: error.message};
      const labels = {'title': ['goalTitle', 'Title'], 'objective': ['goalObjective', 'Objective'], 'criteria': ['goalCriteria', 'Completion criteria · one per line'], 'scope': ['goalScope', 'Scope'], 'constraints': ['goalConstraints', 'Constraints · one per line'], 'research.comparison_design': ['goalComparison', 'Comparison design']};
      const fields = (body.fields || []).map((field) => Array.isArray(field) ? field.join('.') : String(field));
      const named = fields.map((field) => html`<li>${labels[field] ? t(labels[field][1]) : codeWords(field)} ${hashCell(field)}</li>`);
      const words = body.failure_code === 'goal.document_invalid' ? t('Correct the named fields, then save the goal again.') : body.message || body.detail || error.message;
      if (slot) {
        slot.innerHTML = refusal({...body, detail: words, message: words}, 'warning', {more: html`${named.length ? html`<p>${t('Fields to correct')}</p><ul>${named}</ul>` : ''}${codeRef(t('Goal validation details'), body)}`});
        slot.hidden = false;
      }
      for (const name of fields) {
        const field = labels[name] && $('#' + labels[name][0]);
        if (!field) continue;
        if (field.closest('details')) field.closest('details').open = true;
        field.setAttribute('aria-invalid', 'true'); field.setAttribute('aria-describedby', 'goalSaveRefusal');
      }
      const first = fields.map((field) => labels[field] && $('#' + labels[field][0])).find(Boolean);
      const pressed = document.activeElement, dialog = $('#dialog'), notice = slot?.firstElementChild;
      if (first) requestAnimationFrame(async () => {
        const disclosure = first.closest('details');
        const current = () => first.isConnected && dialog.open && slot?.firstElementChild === notice && document.activeElement === pressed;
        if (!current()) return;
        if (disclosure) {
          // A frame callback precedes style/layout: opening the native disclosure has not yet
          // published its ::details-content transition generation. Resolve its CSS layout,
          // then read the generated animations after that rendering step, including reversals.
          void getComputedStyle(disclosure, '::details-content').blockSize;
          await new Promise(requestAnimationFrame);
          while (current() && disclosure.open) {
            const transitions = disclosure.getAnimations({subtree: true}).filter((animation) => animation.playState !== 'finished' && animation.effect?.getTiming().iterations !== Infinity);
            if (!transitions.length) break;
            await Promise.allSettled(transitions.map((animation) => animation.finished));
            await new Promise(requestAnimationFrame);
          }
        }
        if (current() && (!disclosure || disclosure.open)) { first.scrollIntoView({block: 'nearest'}); first.focus({preventScroll: true}); }
      });
      return null;
    }
  }
  async function save() {
    const lines = (id) => $('#' + id).value.split('\n').map((s) => s.trim()).filter(Boolean);
    const d = S.draft.doc, stages = $$('[data-goal-stage]:checked').map((el) => el.dataset.goalStage);
    const doc = {...d, title: $('#goalTitle').value, objective: $('#goalObjective').value, scope: $('#goalScope').value, constraints: lines('goalConstraints'),
      criteria: lines('goalCriteria').map((text, i) => ({criterion_id: d.criteria[i]?.criterion_id || 'c' + (i + 1), text})),
      research: d.kind === 'RESEARCH' ? {purpose: Picker.value('goalPurpose') || 'NEW_RESEARCH', comparison_design: $('#goalComparison').value, required_stages: stages} : null};
    if (!S.draft.hash) {
      const b = await saveRequest('/api/goals/open', {goal_declaration: doc});
      if (!b) return;
      closeDialog(); S.body = null; S.rows = null;
      // the answer's `/goal ` prompt hands the goal to Claude or Codex; it is shown once, to copy
      openDialog(t('Goal'), t('Goal opened'), html`<p>${t('Give this prompt to Claude or Codex; the agent takes the goal, works under it and submits it.')}</p><pre class="code-block code-document">${b.goal_prompt || ''}</pre>`, html`${btn(t('Copy prompt'), 'copy-text', b.goal_prompt || '', 'button')}${btn(t('Open the goal'), 'goal-open', b.goal_hash, 'button primary')}`);
      return;
    }
    const b = await saveRequest('/api/goals/revise', {goal_id: S.draft.id, goal_hash: S.draft.hash, goal_declaration: doc, change_reason: $('#goalReason').value});
    if (!b) return;
    closeDialog(); S.body = null; open(b.goal_hash);
  }
  function referenceChoices() {
    return Data.history().flatMap((row) => {
      const raw = row.raw || {}, base = {reference_id: 'evidence-' + crypto.randomUUID().slice(0, 8), label: row.name + (row.reference ? ' · ' + row.reference : '')};
      if (raw.review_publication_hash && raw.book) return [{...base, stage: 'EVIDENCE_CRO', request: {operation: 'EVIDENCE_CRO_EXPORT', ...raw.book, review_publication_hash: raw.review_publication_hash}}];
      if (raw.book?.result_hash) return [{...base, stage: 'PORTFOLIO', request: {operation: 'REPORT', result_hash: raw.book.result_hash}}];
      const stage = {factor: 'FACTOR_FOUNDATION', alpha: 'ALPHA', risk: 'RISK', portfolio: 'PORTFOLIO'}[String(raw.kind || '').split('.')[0]];
      return stage && raw.task_id ? [{...base, stage, request: {operation: 'EXPERIMENT_READBACK', task_id: raw.task_id, ...(raw.book?.portfolio_session ? {portfolio_session: raw.book.portfolio_session} : {})}}] : [];
    });
  }
  function attach() {
    S.references = referenceChoices(); S.dialogHash = S.body.goal.goal_hash;
    openDialog(t('Exact evidence'), t('Attach exact evidence'), html`<p>${t('Use a saved owner read request, never RUN or a latest-result selector.')}</p>
      ${fieldOf('goalExisting', 'Saved research', html`${picker('goalExisting', [...S.references.map((r, i) => [String(i), r.label]), ['custom', t('Exact request (advanced)')]], {selected: S.references.length ? '0' : 'custom', label: t('Saved research')})}`)}
      <details class="reveal-details"><summary>${t('Exact request (advanced)')}</summary><textarea id="goalReference" class="ui-field" rows="12">${JSON.stringify({reference_id: 'factor', label: 'Factor evidence', stage: 'FACTOR_FOUNDATION', request: {operation: 'EXPERIMENT_READBACK', task_id: ''}}, null, 2)}</textarea></details>`,
      html`${btn(t('Verify and attach'), 'goal-attach-save', '', 'button primary')}`);
  }
  async function attachSave() {
    const chosen = Picker.value('goalExisting');
    const ref = chosen === 'custom' ? JSON.parse($('#goalReference').value) : S.references[Number(chosen)];
    const b = await Data.post('/api/goals/attach', {goal_hash: S.dialogHash, goal_reference: ref, change_reason: 'Explicit evidence association'});
    closeDialog(); S.body = null; open(b.goal_hash);
  }
  function note() {
    S.dialogHash = S.body.goal.goal_hash;
    openDialog(t('Decision note'), t('Add a decision note'), html`<p>${t('A note is attributed, not product validation or scientific approval.')}</p>
      ${fieldOf('goalNoteKind', 'Statement kind', html`${picker('goalNoteKind', [['DECISION', t('Decision')], ['CONCLUSION', t('Conclusion')], ['PM_RESPONSE', t('PM response')]], {selected: 'DECISION', label: t('Statement kind')})}`)}
      ${fieldOf('goalAttribution', 'Attribution', html`<input id="goalAttribution" class="ui-field" value="Researcher">`)}
      ${fieldOf('goalDisposition', 'Evidence judgment', html`${picker('goalDisposition', ['SUPPORTS', 'DOES_NOT_SUPPORT', 'INSUFFICIENT', 'OPEN', 'REJECTED'].map((k) => [k, codeWords(k)]), {selected: 'INSUFFICIENT', label: t('Evidence judgment')})}`)}
      ${fieldOf('goalNoteText', 'The note and its limitations', html`<textarea id="goalNoteText" class="ui-field" rows="8"></textarea>`)}
      <fieldset><legend>${t('Cited evidence')}</legend>${(S.body.references || []).map((r) => html`<label><input type="checkbox" data-goal-reference="${r.reference.reference_id}"${r.state !== 'UNAVAILABLE' ? ' checked' : ''}> ${r.reference.label} · ${codeWords(r.state)}</label>`)}</fieldset>
      ${fieldOf('goalResponse', 'Response to', html`${picker('goalResponse', [['', t('None')], ...S.body.goal.statements.map((s) => [s.statement_id, s.attribution + ' · ' + s.statement_id])], {selected: '', label: t('Response to')})}`)}`,
      html`${btn(t('Save note'), 'goal-note-save', '', 'button primary')}`);
  }
  async function noteSave() {
    const statement = {statement_id: 'statement-' + crypto.randomUUID().slice(0, 8), kind: Picker.value('goalNoteKind'), attribution: $('#goalAttribution').value,
      text: $('#goalNoteText').value, evidence: $$('[data-goal-reference]:checked').map((el) => el.dataset.goalReference),
      disposition: Picker.value('goalDisposition'), ...(Picker.value('goalResponse') ? {responds_to: Picker.value('goalResponse')} : {})};
    const b = await Data.post('/api/goals/note', {goal_hash: S.dialogHash, goal_statement: statement, change_reason: 'Explicit attributed decision note'});
    closeDialog(); S.body = null; open(b.goal_hash);
  }
  /* Abandoning closes the goal; a follow-up opens a new goal. Confirmed with its reason. */
  function abandon() {
    S.dialogHash = S.body.goal.goal_id;
    openDialog(t('Goal · explicit confirmation'), t('Abandon this goal?'), html`<p>${t('The goal closes as abandoned; its record stays readable, and a follow-up opens a new goal.')}</p>${fieldOf('goalAbandonReason', 'Reason', html`<input id="goalAbandonReason" class="ui-field" value="">`)}`, html`${btn(t('Abandon goal'), 'goal-abandon-commit', '', 'button primary', true)}`);
  }
  async function abandonCommit() {
    const b = await Data.post('/api/goals/abandon', {goal_id: S.dialogHash, change_reason: $('#goalAbandonReason').value || 'Abandoned by a person'});
    closeDialog(); S.body = null; S.rows = null; open(b.goal_hash);
  }
  /* U70: Stop ends the first use -- its goal abandoned (the Host closes the network its delegation opened) -- confirmed once. */
  function stopFirstUse() {
    S.dialogHash = S.body.goal.goal_id;
    openDialog(t('First use · explicit confirmation'), t('Stop the first use?'), html`<p>${t('The agent takes no more steps for you, and a network it opened is closed; the goal closes as abandoned, its record readable.')}</p>`, html`${btn(t('Stop'), 'goal-first-use-stop-commit', '', 'button primary', true)}`);
  }
  async function stopFirstUseCommit() {
    const b = await Data.post('/api/goals/abandon', {goal_id: S.dialogHash, change_reason: 'Stopped by the person in the Workbench'});
    closeDialog(); S.body = null; S.rows = null; open(b.goal_hash);
  }
  /* V668 / PG2: all owner-declared reference kinds use the same sealed reference
   * reader. GOAL_REFERENCE reopens the exact request and checks its recorded identity;
   * its summary is the owner's projection, never the Goal's Facts or a latest object. */
  const REPORT_READOUTS = [
    ['distinct_names_held', 'Distinct names held'], ['effective_n', 'Effective number of holdings'],
    ['one_way_turnover_per_trading_session', 'One-way turnover per trading session'],
    ['cumulative_net_wealth', 'Cumulative net wealth'], ['cost_bps_per_side', 'Cost per side (bps)'],
    ['cost_bps_round_trip', 'Round-trip cost (bps)'], ['platform_one_way_cost_bps', 'Platform one-way cost (bps)'],
    ['aggregate_cap_binding_sessions', 'Sessions at the aggregate cap'], ['aggregate_cap_binding_names_total', 'Total names at the aggregate cap'],
    ['median_holding_adv20_dollar_volume', 'Median holding ADV20 dollar volume'], ['industry_attribution_available', 'Industry attribution available'],
  ];
  const referenceMatchesAddress = () => S.reference?.key === addressed() + ':' + hashParams().get('reference');
  function referenceBody(reading) {
    const row = reading.row, ref = row?.reference, summary = row?.summary;
    const identity = kv([[t('Reference'), html`<span class="mono">${reading.id}</span>`],
      [t('Exact revision'), hashCell(reading.hash)], ...(ref ? [[t('Stage'), codeWords(ref.stage)], [t('Request'), codeWords(ref.request.operation)]] : [])]);
    const retry = btn(t('Check reference integrity'), 'goal-reference-retry', reading.id, 'button compact');
    if (reading.loading) return html`<section class="inspector-section">${identity}<p>${t('Reading this exact reference…')}</p></section>`;
    if (reading.error || row?.state === 'UNAVAILABLE' || !summary) return html`<section class="inspector-section">${identity}${refusal({failure_code: row?.failure_code || 'workbench.goal_reference_unavailable', reason: t('Reference {reference} could not be read at its recorded identity.', {reference: reading.id})}, 'warning', {word: t('Reference unavailable'), action: retry})}${reading.error ? html`<p class="caption">${explainCode(reading.error)}</p>` : ''}${ref ? codeRef(t('Exact request'), ref.request) : ''}</section>`;
    if (row.state === 'VERIFIED_REFUSAL') {
      const owner = summary.owner_refusal || {}, hasCause = owner.reason || owner.detail || owner.message || owner.explanation;
      // A code-only owner refusal still carries the owner's recorded claim; read
      // that sentence rather than inventing a cause or exposing an undeclared word.
      const body = hasCause ? owner : {...owner, reason: t(summary.claim)};
      return html`<section class="inspector-section">${identity}${refusal(body, 'warning', {word: codeWords(row.state), catalog: true, action: retry, more: hasCause && summary.claim ? html`<p>${t(summary.claim)}</p>` : ''})}${codeRef(t('Recorded evidence and limitations'), summary)}${codeRef(t('Exact request'), ref.request)}</section>`;
    }
    const readouts = summary.readouts ? kv(REPORT_READOUTS.filter(([key]) => summary.readouts[key] != null).map(([key, word]) => [t(word), typeof summary.readouts[key] === 'boolean' ? t(summary.readouts[key] ? 'Yes' : 'No') : String(summary.readouts[key])]), 'kv-columns') : '';
    const window = summary.window ? kv([[t('From'), summary.window.selected_start], [t('To'), summary.window.selected_end], [t('Sessions'), summary.window.selected_session_count]], 'kv-columns') : '';
    return html`<section class="inspector-section">${identity}${stateLine(row.state === 'VERIFIED_READBACK' ? 'verified' : row.state === 'VERIFIED_REFUSAL' ? 'refused' : 'metadata', {word: codeWords(row.state), next: ''})}<p>${t('The reference owner’s recorded reading and limitations; reference integrity is not scientific approval.')}</p>${summary.result_hash ? kv([[t('Result'), hashCell(summary.result_hash)], [t('Report'), hashCell(summary.report_hash)]]) : ''}${window}${readouts}${summary.limitations?.length ? html`<ul>${summary.limitations.map(word => html`<li>${codeWords(word)}</li>`)}</ul>` : ''}${codeRef(t('Recorded evidence and limitations'), summary)}${codeRef(t('Exact request'), ref.request)}${codeRef(t('Recorded identity'), ref.identity)}</section>`;
  }
  function refreshReference() {
    if (Window.inspectorMode() === 'reference' && referenceMatchesAddress()) Window.setInspectorBody(referenceBody(S.reference));
  }
  async function openReference(id, {push = false, goalHash = addressed(), retry = false} = {}) {
    if (!id || !goalHash) return false;
    const key = goalHash + ':' + id;
    if (!retry && S.reference?.key === key && Window.inspectorMode() === 'reference') return;
    if (push && (hashParams().get('reference') !== id || addressed() !== goalHash)) objectEntry('goal-reference:' + key);
    Inspect.selectAddressMode('reference', {goal: goalHash, case: '', reference: id});
    const reading = {key, id, hash: goalHash, row: null, error: '', loading: true};
    S.reference = reading;
    const readHeader = () => ({title: reading.row?.reference.label || id, kind: reading.row ? codeWords(reading.row.reference.request.operation) : t('Referenced object')});
    Window.openInspector({mode: 'reference', ...readHeader(), readHeader, body: referenceBody(reading), by: ['goal-evidence', id], onClose: () => {
      if (S.reference === reading) S.reference = null;
      if (addressed() === goalHash && hashParams().get('reference') === id) replaceHash({reference: ''});
    }});
    try {
      const body = await Data.read('/api/goals/reference?' + new URLSearchParams({goal_hash: goalHash, goal_reference_id: id}));
      if (S.reference !== reading || !referenceMatchesAddress()) return;
      reading.row = body.goal_hash === goalHash && body.reference?.reference?.reference_id === id ? body.reference : null;
      if (!reading.row) reading.error = 'goal.reference_not_found';
    } catch (error) { if (S.reference === reading) reading.error = error.message; }
    finally { if (S.reference === reading) { reading.loading = false; Window.refreshInspectorHeader(); refreshReference(); } }
  }
  async function evidence(id) {
    const ref = S.body.references.find(row => row.reference.reference_id === id)?.reference;
    if (!ref) return openReference(id);
    return openReference(ref.reference_id, {push: true, goalHash: S.body.goal.goal_hash});
  }
  async function continueFrom(i) { const ref = S.body.references[Number(i)].reference; objectEntry('page:lab'); replaceHash({goal: ''}); await LiveResearch.continueFrom(ref.request.task_id); }
  async function exportGoal(format) { const b = await Data.read('/api/goals/export?' + new URLSearchParams({goal_hash: S.body.goal.goal_hash})); download(format === 'json' ? JSON.stringify(b, null, 2) : b[format], 'goal-' + short(S.body.goal.goal_hash, SHORT.hash) + '.' + format, format === 'html' ? 'text/html' : format === 'yaml' ? 'text/yaml' : 'application/json'); }
  async function more(cursor) {
    const rows = S.rows, navigation = Data.navigationIntent();
    if (!listing() || !cursor || rows?.next_cursor !== cursor || S.paging) return;
    const pending = S.paging = {rows, cursor};
    render();
    try {
      const b = await Data.read('/api/goals?' + new URLSearchParams({history_cursor: cursor}));
      if (S.paging !== pending || S.rows !== rows || !Data.navigationCurrent(navigation) || !listing()) return;
      const consumed = new Set([...(rows.consumedCursors || []), cursor]);
      S.rows = {...b, goals: Data.uniqueRows([...rows.goals, ...(b.goals || [])], row => row.goal_id), refused: Data.uniqueRows([...(rows.refused || []), ...(b.refused || [])], row => row.goal_id || JSON.stringify(row)), next_cursor: consumed.has(b.next_cursor) ? null : b.next_cursor, consumedCursors: [...consumed]};
    } catch (error) { if (S.rows === rows && Data.navigationCurrent(navigation)) S.error = error.message; }
    finally { if (S.paging === pending) { S.paging = null; render(); } }
  }
  function leaveReads() {S.ticket++;S.loading=false;S.verifying=false;S.paging=null;S.error='';S.verifyError='';S.dirty=true;S.rows=null;}
  return {pages, leaveReads, selected, checkLine, outcomeWord, stepWords, hoursLeft, listing, ensure, observe, verify, page, open, list, edit, setKind, save, attach, attachSave, note, noteSave, abandon, abandonCommit, stopFirstUse, stopFirstUseCommit, evidence, openReference, referenceMatchesAddress, refreshReference, continueFrom, exportGoal, more, hasFacts, factsSections};
})();
