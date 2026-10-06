/* The Features page (U56; R5, U28 and V354 joined; EX, the formula point): the formula factors agents declared in
 * this workspace's research plans, the Host's page of them at a time, each with its review packet -- the declaration
 * and its recipe, the contract's golden examples, the identity it adds (none moved), its trials and the study each ran
 * against, its build's coverage, the formulas tried beside it, the daily Panel's admission and its activation. A
 * person activates a formula into this workspace's daily catalog, or deactivates it, by the requests the Host offers;
 * the Host refuses an agent's. A trial record that no longer reads is named beside the trials (U28). Factor
 * screening's second tab. A packet computes its golden examples, so it is read once a visit, never on a repaint. */
const LiveFeatureResearch = (() => {
  const pages = new Set(['feature-research']);
  const S = {trialPrereq: null, list: null, listError: '', listLoading: false, page: 1, packet: null, packetKey: '', packetError: '', packetLoading: false, ticket: 0, pending: null, busy: false, notice: '', refused: ''};
  const keyOf = (x) => x.feature_plan_hash + ':' + x.factor_id;
  // the address names a packet by its plan and its factor: `feature=<plan hash>:<factor id>`
  const addressed = () => { const v = hashParams().get('feature') || '', at = v.indexOf(':'); return at < 0 ? null : {key: v, plan: v.slice(0, at), id: v.slice(at + 1)}; };
  const STATE = {ACTIVE: ['succeeded', 'feature|Active'], NOT_ACTIVE: ['metadata', 'Not active']};
  const TRIAL = {RUNNING: ['running', 'Running'], COMPLETED: ['succeeded', 'Completed'], STOPPED: ['blocked', 'Stopped']};
  const LEDES = {ACTIVE: 'In this workspace\'s daily catalog: its daily updates compute it.', NOT_ACTIVE: 'Declared by an agent in a research plan; not in this workspace\'s daily catalog.'};
  const ALPHA = ['mean_rank_ic', 'pooled_oos_r2', 'mean_gross_decile_spread', 'fold_coverage_mean']; // the trial's comparison, in the owner's order
  const exact = (v) => v == null || !Number.isFinite(Number(v)) ? t('no value') : Number(v) !== 0 && Math.abs(v) < 0.00005 ? Number(v).toExponential(2).replace(/^-/, '−') : fmt(v, 4);
  const share = (v) => v == null ? '' : num(Number(v) * 100, 'percent');
  const idCell = (id, n = SHORT.id) => html`<span class="mono">${short(id, n)}</span>`;
  const completed = (x) => (x.trials || []).filter((r) => r.state === 'COMPLETED');
  /* The Host's reads, once a visit: the listing's page, and the packet the address names. `leave` forgets both. */
  async function readList(page) {
    const ticket = S.ticket; S.listLoading = true;
    try { const b = await Data.read(Data.route('FEATURE_EXTENSIONS') + (page > 1 ? '?' + new URLSearchParams({extensions_page: page}) : '')); if (ticket === S.ticket) { S.list = b; S.page = b.page || page; S.listError = ''; } }
    catch (e) { if (ticket === S.ticket) S.listError = e.message; }
    finally { if (ticket === S.ticket) { S.listLoading = false; render(); } }
  }
  async function readPacket(a) {
    const ticket = S.ticket; S.packetKey = a.key; S.packet = null; S.packetError = ''; S.packetLoading = true;
    try { const b = await Data.read(Data.route('FEATURE_REVIEW') + '?' + new URLSearchParams({feature_plan_hash: a.plan, feature_factor_id: a.id})); if (ticket === S.ticket && S.packetKey === a.key) S.packet = b; }
    catch (e) { if (ticket === S.ticket && S.packetKey === a.key) S.packetError = e.message; }
    finally { if (ticket === S.ticket && S.packetKey === a.key) { S.packetLoading = false; render(); } }
  }
  // the routes are the session's: before it is read (a direct load of the address) nothing is asked, and its render calls again
  function ensure() {
    if (!pages.has(app.page) || Data.workspaceStatus !== 'ready') return;
    const a = addressed();
    if (!S.list && !S.listLoading && !S.listError) void readList(S.page); // the damaged records ride on every page of it
    if (a && S.packetKey !== a.key && !S.packetLoading) void readPacket(a);
  }
  /* U60: with no formula factor declared, what a trial will run against -- the feature controls' block on the newest
   * available input, read once a visit. */
  const newestInput = () => (Data.inputs?.() || []).filter((v) => v.available).sort((a, b) => String(b.date).localeCompare(String(a.date)))[0] || null;
  async function readTrialPrereq() {
    const v = newestInput(), ticket = S.ticket;
    if (!v || S.trialPrereq) return;
    S.trialPrereq = {pending: true, input: v};
    try { const b = await Data.read('/api/features/controls?' + new URLSearchParams({input_binding_hash: v.binding_hash})); if (ticket === S.ticket) { S.trialPrereq = {input: v, block: b.prerequisites || null}; render(); } }
    catch { if (ticket === S.ticket) S.trialPrereq = {input: v, block: null}; }
  }
  function leave() { S.ticket++; Object.assign(S, {trialPrereq: null, list: null, listError: '', listLoading: false, page: 1, packet: null, packetKey: '', packetError: '', packetLoading: false, notice: '', refused: '', pending: null}); }
  function refresh() { leave(); render(); ensure(); }
  /* Only what the Host offers: its `next_requests` name the activation or the deactivation. A packet without one says
   * why from its own facts -- a failed contract, no completed trial, or a Panel that does not admit it. */
  const held = (x) => x.contract?.status !== 'PASSED' ? 'The formula fails its contract' : !completed(x).length ? 'No completed trial of this formula' : x.active_panel && !x.active_panel.admitted ? 'The daily Panel does not admit it' : '';
  function act(x, cls = 'button compact') {
    const next = x.next_requests || {}, busy = S.busy ? 'Waiting for the product owner' : '';
    if (next.deactivate) return typedBtn(t('Deactivate'), 'feature-research-confirm', 'deactivate:' + keyOf(x), cls, busy);
    if (next.activate) return typedBtn(t('Activate'), 'feature-research-confirm', 'activate:' + keyOf(x), cls, busy);
    if (x.state === 'NOT_ACTIVE' && x.contract && held(x)) return typedBtn(t('Activate'), 'feature-research-confirm', 'activate:' + keyOf(x), cls, held(x));
    return '';
  }
  const trialWords = (r) => r.trial ? html`${countText(r.trials, '{n} trial', '{n} trials')} · ${t('latest {state} {when}', {state: t(TRIAL[r.trial.state]?.[1] || r.trial.state).toLowerCase(), when: when(r.trial.requested_at)})}` : t('Not tried');
  const row = (r) => { const [state, word] = STATE[r.state] || ['metadata', r.state]; return objectRow({state, name: html`<span class="mono">${r.factor_id}</span>`, to: {action: 'feature-research-open', value: keyOf(r)}}, {key: keyOf(r), word: t(word), props: [html`<span class="mono" data-tip="${r.formula}">${r.formula}</span>`, r.preprocessing_recipe ? coded(r.preprocessing_recipe) : t('No recipe named'), trialWords(r)], actions: act(r, 'menu-row')}); };
  /* U28: a trial record that no longer reads, named with the owner's words and its way on; outside the boxes (MA2). */
  function damaged() {
    const d = S.list?.damaged || [];
    if (!d.length) return '';
    return banner(countText(d.length, '{n} trial record no longer reads', '{n} trial records no longer read'), html`${explainCode(d[0].failure_code)} ${joinMarkup(d.map((x) => hashCell(x.feature_trial_id, SHORT.hash)))}`, 'warning', link(t('Storage & retention'), 'storage', 'text-btn'), 'warning');
  }
  function refusedPlans() {
    const plans = S.list?.refused_plans || [];
    if (!plans.length) return '';
    return panel(t('Unreadable Feature plans'), '', html`<div class="card-list">${plans.map((plan) => html`<div>
      <h3>${t('Feature plan')} ${hashCell(plan.feature_plan_hash, SHORT.hash)}</h3>
      ${refusal({failure_code: plan.failure_code, reason: plan.detail}, TONE.attention, {state: 'refused', word: t('Refused'), next: html`<span class="flow">${link(t('Research inputs'), 'inputs', 'text-btn')} ${link(t('Storage & retention'), 'storage', 'text-btn')}</span>`})}
    </div>`)}</div>`);
  }
  function top() {
    const a = S.listError ? notRead(t('Features not read'), S.listError, explainCode(S.listError), btn(t('Read again'), 'feature-research-refresh', '', 'button compact')) : '';
    const b = S.packetError ? notRead(t('Review packet not read'), S.packetError, explainCode(S.packetError), btn(t('Read again'), 'feature-research-refresh', '', 'button compact')) : '';
    return html`${a}${b}${S.refused ? notRead(t('Feature action refused'), S.refused, explainCode(S.refused)) : ''}${S.notice ? noteLine(t('Last operation'), S.notice) : ''}`;
  }
  function page() {
    if (!pages.has(app.page)) return null;
    const a = addressed();
    if (!a) {
      const head = objectHead(t('Features'), t('The formula factors agents declared in this workspace\'s research plans. An agent plans, builds and tries a formula; a person activates it into this workspace\'s daily catalog. Reading a packet checks its golden examples and builds nothing.'), '');
      if (!S.list) return html`${head}${top()}${S.listError ? '' : skeleton('rows')}`;
      const all = S.list.factors || [], at = (S.list.page || 1) - 1, n = S.list.page_count || 1;
      const foot = pager({total: S.list.total, one: '{n} formula factor', many: '{n} formula factors', page: at, pages: n, prev: ['feature-research-page', String(at)], next: ['feature-research-page', String(at + 2)]});
      const hasRefusedPlans = (S.list.refused_plans || []).length > 0;
      return html`${head}${top()}${refusedPlans()}${damaged()}${all.length ? html`<div class="card-list lines slotted">${all.map(row)}</div>${foot}` : hasRefusedPlans ? '' : html`${emptyState(t('No formula factor is declared in this workspace.'))}${trialPrereq()}`}`;
    }
    const x = S.packet;
    if (!x || keyOf(x) !== a.key) return html`${top()}${S.packetError ? '' : skeleton('head')}`;
    const [state, word] = STATE[x.state] || ['metadata', x.state];
    return html`${top()}${objectHead(html`<span class="mono">${x.factor_id}</span>`, html`<p class="lede">${t(LEDES[x.state] || '')}</p>`, act(x, 'button'), stateLine(state, {word: t(word), next: ''}), [], {object: true, id: x.factor_id, facts: [[t('Research plan'), idCell(x.feature_plan_hash, SHORT.hash)]]})}${standingPanel(x.standing)}${declaration(x)}${contract(x)}${identity(x)}${damaged()}${trials(x)}${build(x)}${tried(x)}${activation(x)}${environment(x)}${codeRef(t('Exact review packet (JSON)'), x)}`;
  }
  function trialPrereq() {
    if (!S.trialPrereq) { if (typeof queueMicrotask === 'function') queueMicrotask(() => void readTrialPrereq()); return ''; }
    const p = S.trialPrereq;
    return p.block ? prerequisitesPanel(p.block, {id: p.input.id, binding: p.input.binding_hash}) : '';
  }
  function declaration(x) {
    const d = x.declaration || {}, s = d.specification || {};
    return panel(t('Declaration'), t('The formula as its research plan declares it, and the recipe that preprocesses it.'), kv([[t('Formula'), html`<span class="mono">${d.formula}</span>`], [t('Registered computation'), html`<span class="mono">${s.formula_ref}</span>`],
      [t('Preprocessing recipe'), d.preprocessing_recipe ? coded(d.preprocessing_recipe) : t('No recipe named')], [t('Family'), s.family || ''], [t('Window (trading sessions)'), count(s.window_sessions)], [t('Economic lag (trading sessions)'), count(s.lag_sessions)]], 'kv-columns'));
  }
  /* The contract: the reference's golden examples, each computed by the installed kernel within the declared tolerance. */
  function contract(x) {
    const c = x.contract || {}, goldens = c.goldens || [], passed = c.status === 'PASSED';
    const rows = goldens.map((g, i) => tr([count(i + 1), html`<span class="owner-text">${g.label}</span>`, exact(g.expected), exact(g.computed), stateLine(g.within_tolerance ? 'succeeded' : 'failed', {word: t(g.within_tolerance ? 'passes' : 'fails'), next: ''})]));
    const facts = kv([[t('Result'), stateLine(passed ? 'succeeded' : 'failed', {word: t(passed ? 'Contract passed' : 'Contract fails'), next: ''})], ...(c.failure_code ? [[t('Code'), codedSubject(c.failure_code)]] : []), ...(c.specification_hash ? [[t('Specification'), hashCell(c.specification_hash)]] : []), ...(c.preprocessing_role ? [[t('Preprocessing role'), coded(c.preprocessing_role)]] : [])], 'kv-columns');
    const examples = goldens.length ? table([{label: '#', type: 'num', index: true}, {label: t('Example'), type: 'text', absorb: true}, {label: t('Expected'), type: 'num'}, {label: t('Computed'), type: 'num'}, {label: t('Result'), type: 'status', cls: 'col-tight'}], rows, '', {report: true, countLine: false, classes: 'compact'}) : '';
    return panel(t('Contract'), t('The reference\'s golden examples, each computed by the installed kernel and held to the declared tolerance.'), html`${facts}${examples}`, '', goldens.length ? 'data-box="table"' : '');
  }
  function identity(x) {
    const id = x.identity || {};
    return panel(t('Identity'), t('A formula factor is data: activation adds a catalog entry and moves no code identity.'), kv([[t('Implementation'), hashCell(id.implementation_hash)], [t('Methodology'), hashCell(id.methodology_hash)], [t('Numerical specification'), hashCell(id.numerical_spec_hash)],
      [t('Adds'), codeWords(id.adds)], [t('Moves'), (id.moves || []).length ? joinMarkup(id.moves.map((v) => html`<span class="mono">${v}</span>`)) : t('no existing identity')]], 'kv-columns'));
  }
  /* The trials of this methodology, under any plan: each by its state and the study it ran against (V354); the
   * latest completed one's comparison -- the screening's evidence, the correlation with the factors present, and the
   * Alpha study's change with the formula. */
  function trials(x) {
    const list = x.trials || [];
    if (!list.length) return panel(t('Trials'), '', html`<p class="caption">${t('No trial of this formula yet. An agent tries it against a completed study:')}</p><pre class="code-block">trial run --plan ${x.feature_plan_hash} --task &lt;study&gt;</pre>`);
    const rows = list.map((r) => { const [state, word] = TRIAL[r.state] || ['metadata', r.state]; return tr([idCell(r.feature_trial_id, SHORT.hash), stateLine(state, {word: t(word), next: ''}), r.outcome ? coded(r.outcome) : r.stopped ? html`${coded(r.stopped.step)} · ${codedSubject(r.stopped.failure_code)}` : '', idCell(r.baseline_task_id)]); });
    const table1 = table([{label: t('Trial'), type: 'id'}, {label: t('State'), type: 'status'}, {label: t('Outcome'), type: 'text', absorb: true}, {label: t('Baseline study'), type: 'id'}], rows, '', {report: true, countLine: false, classes: 'compact'});
    const last = completed(x).at(-1);
    return html`${panel(t('Trials'), t('Each trial runs against its baseline study: an Alpha study handed off from a Factor study, or the Portfolio study built on one.'), table1, '', 'data-box="table"')}${last ? comparison(last, x.factor_id) : ''}`;
  }
  function comparison(r, factor) {
    const e = (r.out_of_sample_evidence || [])[0], pairs = r.correlation_with_present_factors || [], change = r.alpha_change || null;
    const evidence = e ? [[t('Classification'), coded(e.classification)], [t('Rank IC (mean, oriented)'), exact(e.mean_oriented_rank_ic)], [t('BY q-value'), exact(e.rank_ic_by_q_value)], [t('Pair coverage (fraction)'), exact(e.validation_pair_coverage_mean)]] : [];
    // V363: studies the owner refused to compare stand on what each scored -- no change is claimed, the owner says why
    const refused = r.alpha_standing === 'NOT_COMPARED', moved = change && !refused ? ALPHA.filter((k) => change[k] != null).map((k) => [codeWords(k), signed(change[k], 'ratio', 4)]) : [];
    const standing = refused ? html`<h3>${t('Alpha study, with the formula against without')}</h3>${kv([[t('Comparison'), coded('NOT_COMPARED')], ...(r.alpha_refusal?.failure_code ? [[t('Code'), codedSubject(r.alpha_refusal.failure_code)]] : [])], 'kv-columns')}${r.alpha_refusal?.detail ? html`<p class="caption owner-text">${r.alpha_refusal.detail}</p>` : ''}` : '';
    const other = (p) => p.left_factor_id === factor ? p.right_factor_id : p.left_factor_id; // the pair's other factor
    const corr = pairs.length ? table([{label: t('Present factor'), type: 'id'}, {label: t('Median cross-section Spearman'), type: 'num'}, {label: t('Common periods'), type: 'num'}], pairs.map((p) => tr([html`<span class="mono">${other(p)}</span>`, exact(p.median_cross_section_spearman), count(p.common_formal_period_count)])), '', {report: true, countLine: false, classes: 'compact'}) : '';
    const body = html`${kv([[t('Trial'), idCell(r.feature_trial_id, SHORT.hash)], [t('Outcome'), coded(r.outcome)], ...evidence], 'kv-columns')}${standing}${moved.length ? html`<h3>${t('Alpha study, with the formula against without')}</h3>${kv(moved, 'kv-columns')}` : ''}${corr ? html`<h3>${t('Correlation with the factors present')}</h3>${corr}` : ''}`;
    return panel(t('Latest comparison'), t('The completed trial\'s screening evidence for this formula and, where screening admitted it, the Alpha study\'s change with it.'), body, '', corr ? 'data-box="table"' : '');
  }
  function build(x) {
    const b = x.build;
    if (!b) return panel(t('Build'), '', html`<p class="caption">${t('No trial has built this formula yet.')}</p>`);
    return panel(t('Build'), t('The formula\'s column in the latest trial\'s build.'), kv([[t('Build Task'), idCell(b.task_id)], [t('Trading sessions'), count(b.sessions)], [t('Listings'), count(b.listings)],
      [t('Available values / total cells'), html`${count(b.available)} / ${count(b.cells)}`], [t('Coverage'), share(b.coverage)], [t('Missing share'), share(b.missing_share)]], 'kv-columns'));
  }
  /* The formulas tried beside this one: what a multiple-testing correction reads. */
  function tried(x) {
    const d = x.tried || {}, words = (f, n) => html`${countText(f, '{n} formula', '{n} formulas')} · ${countText(n, '{n} trial', '{n} trials')}`;
    return panel(t('Formulas tried'), t('Every trial is on the ledger; these counts are what a multiple-testing correction reads.'), kv([[t('In this workspace'), words(d.workspace_formulas || 0, d.workspace_trials || 0)],
      ...(d.goals || []).map((g) => [html`${t('Goal')} ${idCell(g.goal_id)}`, words(g.formulas, g.trials)])], 'kv-columns'));
  }
  function activation(x) {
    const p = x.active_panel || {}, a = x.activation;
    const admits = [[t('Daily Panel'), p.admitted ? t('admits it') : codedSubject(p.reason)], [t('On activation'), codeWords(x.activation_effect)]];
    const record = a ? [[t('Activated by'), t('a person')], [t('Activated'), when(a.activated_at)], [t('Trial'), idCell(a.trial_id, SHORT.hash)], [t('Review packet'), hashCell(a.packet_hash)]] : [];
    return panel(t('Activation'), a ? '' : t('Not in this workspace\'s daily catalog.'), html`${kv([...admits, ...record], 'kv-columns')}${!p.admitted && explainCode(p.reason) ? html`<p class="caption">${explainCode(p.reason)}</p>` : ''}`);
  }
  function environment(x) {
    const e = x.environment || {};
    return panel(t('Environment'), t('Recorded beside the packet as provenance; it is not the formula\'s identity.'), kv([[t('Python'), [e.implementation, e.python].filter(Boolean).join(' ')], [t('Platform'), e.platform || ''],
      [t('Packages'), joinMarkup(Object.entries(e.packages || {}).map(([n, v]) => html`<span class="mono">${n} ${v}</span>`)) || t('none recorded')]], 'kv-columns'));
  }
  /* A person's activation or deactivation, confirmed with the facts it records. */
  function confirm(value) {
    const at = value.indexOf(':'), kind = value.slice(0, at), key = value.slice(at + 1);
    const x = S.packet && keyOf(S.packet) === key ? S.packet : (S.list?.factors || []).find((r) => keyOf(r) === key), request = x?.next_requests?.[kind];
    if (!x || !request || !Data.offers(request.operation) || S.busy) return; // U13: the route the session names
    S.pending = {kind, name: x.factor_id, path: Data.route(request.operation), payload: Object.fromEntries(Object.entries(request).filter(([k]) => k !== 'operation'))};
    const trial = x.trial || completed(x).at(-1);
    const facts = [[t('Formula factor'), html`<span class="mono">${x.factor_id}</span>`], [t('Formula'), html`<span class="mono">${x.formula ?? x.declaration?.formula}</span>`], [t('Research plan'), idCell(x.feature_plan_hash, SHORT.hash)],
      ...(trial ? [[t('Trial'), idCell(trial.feature_trial_id, SHORT.hash)]] : []), ...(x.packet_hash ? [[t('Review packet'), hashCell(x.packet_hash)]] : [])];
    const activate = kind === 'activate';
    openDialog(t('Features · explicit confirmation'), t(activate ? 'Activate this formula factor for this workspace?' : 'Deactivate this formula factor?'), html`${kv(facts)}<p class="caption">${t(activate ? 'It joins this workspace\'s daily Feature catalog: the daily Panel is rebuilt at the next data update, every factor from the history\'s start. Activation records you and the time, the formula and its recipe, the trial it passed and the packet read; it edits no sealed record.' : 'It leaves this workspace\'s daily catalog, and the daily Panel is rebuilt at the next data update; a Panel or study that binds it still reads back.')}</p>`,
      html`${btn(t(activate ? 'Activate' : 'Deactivate'), 'feature-research-commit', '', 'button primary', true)}`);
  }
  async function commit() {
    const p = S.pending;
    if (!p || S.busy) return;
    S.pending = null; S.busy = true; S.refused = ''; S.notice = ''; closeDialog(); render();
    try {
      const b = await Data.post(p.path, p.payload);
      S.notice = b.status === 'ACTIVATED' ? t('Activated for this workspace: {factor}. The daily Panel is rebuilt at the next data update.', {factor: p.name}) : t('Deactivated: {factor}. Panels and studies that bind it still read back.', {factor: p.name});
      S.ticket++; Object.assign(S, {list: null, listLoading: false, packet: null, packetKey: '', packetLoading: false});
    } catch (e) { S.refused = e.message; }
    finally { S.busy = false; render(); ensure(); }
  }
  function open(key) { objectEntry('feature-review:' + key); navigate('feature-research', {feature: key}, {replace: true}); }
  const turn = (v) => { const n = Number(v); if (Number.isInteger(n) && n >= 1 && !S.listLoading) void readList(n); };
  return {pages, ensure, leave, refresh, page, open, list: () => navigate('feature-research', {feature: ''}), confirm, commit, turn};
})();
