/* A strategy's activation (U73; LS1, V459, OW12): a completed book of an installed research strategy, run over its
 * whole support, runs the strategy forward by a person's press. The Host's CONTROLS answer for the book's package says
 * whether it runs forward (`activation`: ACTIVE with its book, first forward session and horizon; MOVED when the
 * package changed since; INACTIVE) and names what it offers (`activation.next_requests`: `activate` the newest book it
 * would admit, `deactivate` while it runs or moved) and, for a book it would refuse, why (`held`). The book page shows
 * the state and offers only what the Host named; the activation's own answer -- its next decision session, horizon and
 * models, with no fit run -- is shown once it is bound. A refusal is said in the Host's words. */
const LiveActivation = (() => {
  const S = {pkg: '', value: null, error: '', reading: false, busy: false, refused: null, pending: null, bound: null, ticket: 0, standingDirty: false};
  let observing = false;
  const repaint = () => (typeof patchMain === 'function' ? patchMain : render)();
  // an installed result names its strategy package; an authored book runs nothing forward
  const packageOf = (subject) => (subject?.source_kind === 'INSTALLED_RESULT' && subject.strategy_package_id) || '';
  // a result whose package this Host has not installed runs nothing forward here: it says nothing, offers nothing
  const NOT_INSTALLED = new Set(['local_application.strategy_package_not_installed', 'research_workspace.strategy_not_installed']);
  async function ensure(subject, again = false) {
    const pkg = packageOf(subject);
    if (!pkg || (!again && S.pkg === pkg && (S.value || S.error || S.reading))) return;
    const ticket = ++S.ticket;
    Object.assign(S, {pkg, value: again && S.pkg === pkg ? S.value : null, error: '', reading: true});
    try {
      const b = await Data.readShared('/api/controls?' + new URLSearchParams({strategy_package_id: pkg}));
      if (ticket === S.ticket) {
        const value = b.activation || {status: 'INACTIVE'};
        if (JSON.stringify(S.value) !== JSON.stringify(value)) S.standingDirty = true;
        S.value = value;
        return ticket;
      }
    } catch (e) {
      if (ticket !== S.ticket) return;
      const code = String(e?.body?.failure_code || e?.body?.refused || e?.message || e).split(':')[0].trim();
      if (NOT_INSTALLED.has(code)) S.value = {status: 'NOT_INSTALLED'}; else S.error = String(e?.message || e);
    }
    finally { if (ticket === S.ticket) { S.reading = false; repaint(); } }
  }
  // V676: the existing activity cadence observes owner changes even when no Task moved.
  // Slow reads join one observation; a changed book invalidates its Standing response.
  async function observe() {
    if (observing || S.reading || S.busy || app.page !== 'portfolio' || !Data.ready || !packageOf(Data.subject())) return;
    observing = true;
    try {
      const subject = Data.subject(), ticket = await ensure(subject, true);
      if (!ticket || ticket !== S.ticket || S.reading || S.busy || Data.subject()?.task_id !== subject.task_id) return;
      if (!S.error && S.standingDirty && await Data.refreshStanding() && ticket === S.ticket) S.standingDirty = false;
    } finally { observing = false; }
  }
  const STATES = {ACTIVE: ['verified', 'Runs forward'], MOVED: ['blocked', 'Package changed'], INACTIVE: ['metadata', 'Not running forward']};
  /* V589, V594, V597: a strategy's dates as its owner states them (`strategy_dates`), each with its words on hover --
   * the information cutoff (an undetermined one names its missing sources and never shows its latest dated information
   * as the cutoff); the trading start, the first actionable session positions are held from (a conditional one says so);
   * a forward book decided before it, its sessions to then a causal replay in-sample, never out-of-sample; the next refit,
   * each component's next fit vintage through its grant's renewal. A field the owner does not state reads as not
   * stated; nothing here is computed. */
  const unstated = (word) => hint(t(word), t('Its owner does not state it yet.'));
  function datesOf(d) {
    if (!d) return {start: unstated('Trading start not stated'), cutoff: unstated('Information cutoff not stated'), replay: '', replayRange: '', refit: unstated('Next refit not stated'), refitEach: [], startValue: unstated('Not stated'), cutoffValue: unstated('Not stated')};
    const start = d.first_actionable_session === undefined ? unstated('Trading start not stated')
      : d.first_actionable_session ? hint(t(d.first_actionable_basis === 'IF_ACTIVATED' ? 'Trading from {date} if activated' : 'Trading from {date}', {date: d.first_actionable_session}), t(d.first_actionable_detail || ''))
      : hint(t('Trading start not determined'), t(d.first_actionable_detail || ''));
    const missing = (d.information_sources || []).filter((s) => s.detail).map((s) => t(s.detail));
    const why = [t('The information cutoff cannot be determined from every required sealed record.'), ...missing, d.latest_dated_information ? t('Its latest dated information is {date}; that is not the cutoff.', {date: d.latest_dated_information}) : ''].filter(Boolean).join(' ');
    const cutoff = d.information_cutoff ? hint(t('Information through {date}', {date: d.information_cutoff}), t(d.information_cutoff_detail || '')) : hint(t('Information cutoff not determined'), why);
    // the forward book decided before the trading start: said once, the replayed range in the owner's words
    const decided = d.forward_book_first_decided_session, r = d.replayed_in_sample_forward_sessions;
    const before = Boolean(decided && d.first_actionable_session && decided < d.first_actionable_session);
    const range = r?.count ? t('{first} to {last}', {first: r.first_session, last: r.last_session}) : '';
    const words = [t('The forward book decided from {date}.', {date: decided || ''}), t(r?.detail || ''), t(d.forward_book_start_detail || '')].filter(Boolean).join(' ');
    const replay = !before ? '' : r?.count == null ? hint(t('In-sample replay not determined'), words) : hint(countText(r.count, '{n} session replayed in-sample', '{n} sessions replayed in-sample'), range ? html`${range}. ${words}` : words);
    const replayRange = !before ? '' : html`${t('Decided from {date}', {date: decided})}${r?.count == null ? html` · ${t('In-sample replay not determined')}` : html` · ${countText(r.count, '{n} session replayed in-sample', '{n} sessions replayed in-sample')}${range ? html` (${range})` : ''}`}`;
    // the next refit: each component's next fit vintage through its grant's renewal, its words on hover
    const m = d.model_renewals, parts = m?.components || [];
    const each = (c) => c.next_fit_vintage ? t('{component} {vintage}', {component: codeWords(c.component_id), vintage: c.next_fit_vintage}) : t('{component} none left', {component: codeWords(c.component_id)});
    const said = (c) => [c.renewal_through ? t('Renewal through {date}.', {date: c.renewal_through}) : '', t(c.detail || '')].filter(Boolean).join(' ');
    const refit = m === undefined ? unstated('Next refit not stated')
      : m.status === 'INACTIVE' ? hint(t('No refit while inactive'), t(m.detail || ''))
      : m.status === 'UNAVAILABLE' && !parts.some((c) => c.next_fit_vintage) ? hint(t('Next refit unknown'), [t(m.detail || ''), ...parts.map((c) => t(c.detail || ''))].filter(Boolean).join(' '))
      : parts.length === 1 && parts[0].next_fit_vintage ? hint(t('Next refit {vintage}', {vintage: parts[0].next_fit_vintage}), said(parts[0]))
      : parts.length === 1 ? hint(t('No refit left'), said(parts[0]))
      : hint(html`${t('Next refit')} ${parts.map(each).join(' · ')}`, parts.map((c) => `${codeWords(c.component_id)}: ${said(c)}`).join(' '));
    const refitEach = parts.map((c) => [codeWords(c.component_id), hint(c.next_fit_vintage ? html`${c.next_fit_vintage}${c.renewal_through ? html` · ${t('through {date}', {date: c.renewal_through})}` : ''}` : c.status === 'UNAVAILABLE' ? t('Unknown') : t('None left'), t(c.detail || ''))]);
    // the panel's values beside their labels: the date alone, its words on hover
    const startValue = d.first_actionable_session === undefined ? unstated('Not stated') : d.first_actionable_session ? hint(d.first_actionable_basis === 'IF_ACTIVATED' ? t('{date} if activated', {date: d.first_actionable_session}) : d.first_actionable_session, t(d.first_actionable_detail || '')) : hint(t('Not determined'), t(d.first_actionable_detail || ''));
    const cutoffValue = d.information_cutoff ? hint(d.information_cutoff, t(d.information_cutoff_detail || '')) : hint(t('Not determined'), why);
    return {start, cutoff, replay, replayRange, refit, refitEach, startValue, cutoffValue};
  }
  /* V614 (U84; FIX1's `review_standing`): the book's Evidence and CRO review at the decision, as its owner states it
   * on the activation offer and its answer -- reviewed (its publication date and the CRO's route; on hover the
   * owner's words, the evidence's date and what the preparation coverage counts), not reviewed, or no book named;
   * a standing the owner does not state reads as not stated, never inferred. */
  function reviewOf(r) {
    if (!r) return unstated('Review standing not stated');
    if (r.status !== 'REVIEWED') return hint(t(r.status === 'NOT_REVIEWED' ? 'Not reviewed' : 'Not available'), t(r.detail || ''));
    const prepared = r.coverage?.prepared || {};
    const words = [t(r.detail || ''), r.evidence_as_of ? t('Evidence as of {date}.', {date: String(r.evidence_as_of).slice(0, 10)}) : '',
      Number.isFinite(prepared.packets) ? t('{a} of {n} analysis-linked packets carry a preparation receipt.', {a: count(prepared.with_preparation_receipt ?? 0), n: count(prepared.packets)}) : '',
      prepared.detail ? t(prepared.detail) : ''].filter(Boolean).join(' ');
    return hint(html`${t('Reviewed {date}', {date: String(r.published_at || '').slice(0, 10)})}${r.cro?.route ? html` · ${codeWords(r.cro.route)}` : ''}`, words);
  }
  // the activation's dates, as the confirmation and its answer say them: the cutoff, the trading start, the replay
  const decisionRows = (dates) => [[t('Information cutoff'), dates.cutoffValue], [t('Trading start'), dates.startValue], ...(dates.replayRange ? [[t('Forward book'), dates.replayRange]] : [])];
  const bookRef = (task, subject) => (task === subject.task_id ? t('this book') : html`<span class="mono">${short(task)}</span>`);
  function panelOf(subject) {
    if (!packageOf(subject) || S.pkg !== packageOf(subject)) return '';
    if (S.error) return notRead(t('Activation not read'), S.error, explainCode(String(S.error).split(':')[0]), btn(t('Read again'), 'activation-read', '', 'button compact'));
    const a = S.value;
    if (!a || a.status === 'NOT_INSTALLED') return ''; // reading, or a package this Host does not hold: nothing is said (ST7)
    const next = a.next_requests || {}, mine = next.activate?.task_id === subject.task_id, held = a.held?.task_id === subject.task_id ? a.held : null;
    const [tone, word] = STATES[a.status] || ['metadata', a.status];
    const dates = a.strategy_dates ? datesOf(a.strategy_dates) : null;
    // the dates the owner states, the trading start beside the forward book's own start where that precedes it; the
    // forward book's first session is the dates' when they state it, else the activation's (said once)
    const dated = dates ? [[t('Information cutoff'), dates.cutoffValue], [t('Trading start'), dates.startValue], ...(dates.replayRange ? [[t('Forward book'), dates.replayRange]] : []), ...(a.status !== 'ACTIVE' ? [] : dates.refitEach.length > 1 ? dates.refitEach.map(([c, v]) => [html`${t('Next refit')} · ${c}`, v]) : dates.refitEach.length ? [[t('Next refit'), dates.refitEach[0][1]]] : [[t('Next refit'), dates.refit]])] : [];
    const facts = a.status === 'ACTIVE' ? kv([[t('From the book'), bookRef(a.book_task_id, subject)], ...dated, ...(dates?.replayRange ? [] : [[t('First forward session'), a.first_forward_session || '']]), [t('Horizon'), a.horizon || ''], ...(a.activated_at ? [[t('Activated'), when(a.activated_at)]] : [])], 'kv-columns') : dated.length ? kv(dated, 'kv-columns') : '';
    const line = a.status === 'ACTIVE' ? ''
      : a.status === 'MOVED' ? t('The strategy package changed since its activation: run the book again and activate the new run.')
      : held ? html`<span class="owner-text">${t(held.detail || '')}</span>${infoMark(held.failure_code)}`
      : mine ? t('This book can run its strategy forward: activation binds its models and opening state from it, and fits nothing.')
      : next.activate ? t('A newer run of this strategy\'s book is the one that activates.')
      : a.detail ? t(a.detail) : t('Run the book over its whole support, then activate it.');
    const refused = S.refused ? refusal({code: S.refused.code, reason: S.refused.detail ? t(S.refused.detail) : explainCode(S.refused.code)}, 'warning', {word: t('Not done')}) : '';
    const busy = S.busy ? 'The request is being sent' : '';
    const ways = html`${mine ? typedBtn(t('Activate'), 'activation-confirm', 'activate', 'button primary', busy) : ''}${next.deactivate ? typedBtn(t('Stop'), 'activation-confirm', 'deactivate', 'button compact', busy) : ''}`;
    return panel(html`${t('Activation')} ${stateLine(tone, {word: t(word), next: ''})}`, t('A person runs an installed strategy forward from a reviewed book; its daily research update then continues it. Research, not trading advice.'), html`${refused}${line ? html`<p>${line}</p>` : ''}${facts}`, ways, 'data-activation');
  }
  /* A person's activation or stop, confirmed with what it binds or removes; the request is the one the Host named. */
  function confirm(kind) {
    const a = S.value, request = a?.next_requests?.[kind];
    if (!request || !Data.offers(request.operation) || S.busy) return;
    const activate = kind === 'activate';
    S.pending = {kind, path: Data.route(request.operation), payload: activate ? {task_id: request.task_id} : {strategy_package_id: request.strategy_package_id}};
    const facts = activate ? [[t('Strategy'), codeWords(S.pkg)], [t('Book'), html`<span class="mono">${short(request.task_id)}</span>`], [t('Its last session'), Data.sessions().at(-1) || ''], ...decisionRows(datesOf(a.strategy_dates)), [t('Review'), reviewOf(a.review_standing)]]
      : [[t('Strategy'), codeWords(S.pkg)], ...(a.book_task_id ? [[t('From the book'), html`<span class="mono">${short(a.book_task_id)}</span>`]] : []), ...(a.horizon ? [[t('Horizon'), a.horizon]] : [])];
    openDialog(t('Strategy · explicit confirmation'), t(activate ? 'Run this strategy forward from this book?' : 'Stop running this strategy forward?'),
      html`${kv(facts)}<p class="caption">${t(activate ? 'Activation binds its models, seed and opening state from this book and fits nothing; the daily research update then continues it.' : 'Its bindings are removed and the daily research update no longer continues it; its history stays readable.')}</p>`,
      btn(t(activate ? 'Activate' : 'Stop'), 'activation-commit', '', 'button primary', true));
  }
  async function commit() {
    const p = S.pending;
    if (!p || S.busy) return;
    S.pending = null; S.busy = true; S.refused = null; closeDialog(); render();
    try {
      const b = await Data.post(p.path, p.payload);
      if (b.status === 'ACTIVATED') bound(b);
    } catch (e) { S.refused = {code: e.body?.failure_code || e.body?.refused || String(e.message).split(':')[0], detail: e.body?.detail || ''}; }
    finally { S.busy = false; render(); void ensure(Data.subject(), true); void Settings.rereadUpdate(); }
  }
  /* What the activation bound, as its answer says: the next decision session, the horizon, each component's models
   * (the fits it reuses), and that it fitted nothing. */
  function bound(b) {
    const dates = datesOf(b.strategy_dates);
    const models = (b.models || []).map((m) => [codeWords(m.component_id), countText(Number(m.reused_fits) || 0, '{n} fit reused', '{n} fits reused')]);
    openDialog(t('Strategy'), t('Runs forward'), html`${kv([...decisionRows(dates), ...(dates.replayRange ? [] : [[t('Next decision session'), b.next_decision_session || '']]), [t('Horizon'), b.horizon || ''], ...models, [t('Models fitted'), count(b.fit_calls ?? 0)], [t('Review'), reviewOf(b.review_standing)]])}<p class="caption">${t('Positions are held only from the trading start; its daily research update continues the forward book.')}</p>`, btn(t('Close'), 'close', '', 'button'));
  }
  /* V593 (U81): Home's "Running forward" -- each strategy that runs forward on one line, from the daily update's answer
   * (`runs_forward`, read through Settings): its information cutoff and trading start, its latest update's positions
   * with their recorded date, claim and exact selector (the owner's metadata summary), its next refit as its owner
   * states it, the daily update's state and next run. The row opens the strategy's book (this panel); its ways are the
   * positions' review and the update's switch in Settings. Nothing is computed here. */
  function positionsOf(l) {
    // absent, the owner states nothing yet (before V595); null, it states that the strategy has had no update
    if (l === undefined) return {fact: hint(t('Positions not stated'), t('Its owner does not state it yet.')), way: ''};
    if (!l) return {fact: t('No update yet'), way: ''};
    if (l.lifecycle !== 'SUCCEEDED') return {fact: html`${t('Update for {date}', {date: l.target_session || ''})} · ${stateLine(String(l.lifecycle || '').toLowerCase(), {next: ''})}`, way: ''};
    if(l.status==='REFUSED')return {fact:hint(t('Positions not read'),t(l.detail || '')),way:''};
    if(!l.review_selector)return {fact:t('Positions not stated'),way:''};
    const fact = html`${t('Positions for {date}', {date: l.target_session || ''})}${l.claim ? html` · ${hint(codeWords(l.claim), t('The update is post-observed research, not timely advice.'))}` : ''}`;
    const way = link(t('Positions'), 'evidence', 'text-btn', {review_selector: JSON.stringify(l.review_selector)});
    return {fact, way};
  }
  function forwardRows() {
    const v = Settings.dailyUpdate().value;
    if (!v) return [];
    const covered = new Set(v.settings?.package_ids || []), on = ['ENABLED_SERVICE_LIFETIME', 'AUTOMATION_CHECK_FAILED'].includes(v.status);
    return (v.runs_forward || []).filter((f) => f.status === 'ACTIVE').map((f) => {
      const {start, cutoff, replay, refit} = datesOf(f.strategy_dates), p = positionsOf(f.latest_update);
      const update = on && covered.has(f.strategy_package_id) ? html`${t('Daily update')} · ${Settings.updateState(v)}` : t('Daily update off');
      const briefUpdate = on && covered.has(f.strategy_package_id) ? html`${t('Daily update')} · ${Settings.updateState(v, {brief: true})}` : t('Daily update off');
      // Full owner captions belong to this object's Facts reader; their hints and the next
      // due time remain whole there while the dense row keeps its state and exact book route.
      const facts = factsRef(t('Facts'), kv([[t('Strategy'), codeWords(f.strategy_package_id)], [t('Information cutoff'), cutoff], [t('Trading start'), start], [t('Forward book'), replay], [t('Positions'), p.fact], [t('Next refit'), refit], [t('Daily update'), update]]));
      const ways = html`${p.way}${link(t('Daily update'), 'settings', 'text-btn', {row: 'researchUpdate'})}`;
      return objectRow({lead: tile('activity', 'accent'), name: codeWords(f.strategy_package_id), to: f.book_task_id ? {page: 'portfolio', extra: {book: f.book_task_id}} : null, cls: 'forward-row'}, {key: 'forward:' + f.strategy_package_id, columns: ['state', 'update', 'facts'], props: [stateLine(STATES[f.status][0], {word: t(STATES[f.status][1]), next: ''}), briefUpdate, facts], time: f.activated_at ? when(f.activated_at) : '', actions: ways});
    });
  }
  return {ensure, observe, panel: panelOf, confirm, commit, read: () => ensure(Data.subject(), true), forwardRows};
})();
