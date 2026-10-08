/* Workspace activity: what the shared operation owner, Task Control and the artifact owners
 * recorded, read in bounded pages from one cursor. Nothing here runs, retries or selects work.
 * A new Task or result is announced; the reader's open object is never replaced.
 *
 * Everything retained is bounded: at most MAX_GROUPS groups of MAX_ITEMS observations, one Task
 * projection per retained group, and a cursor. Catch-up after an absence reads at most MAX_PAGES
 * pages per refresh and continues on the next tick. Polling never stops because the document
 * reports itself hidden (an embedded pane does so permanently); it slows down instead. */
const LiveActivity = (() => {
  // The first read is the feed's tail and the only way back in time the read offers, so it asks
  // for the most the read gives (round 24f): the Team page's retained sessions are whatever that
  // tail holds; later reads continue from the cursor.
  const LIMIT = 200, VISIBLE = 50, MAX_GROUPS = 200, MAX_ITEMS = 40, MAX_PAGES = 4;
  const INTERVAL = 3000, HIDDEN_INTERVAL = 15000, BACKOFF = 10000, CATCH_UP = 250;
  const S = {cursor: null, epoch: null, watermark: 0, groups: new Map(), tasks: {}, error: '', observer: null,
    notice: '', unavailable: 0, unseen: 0, primed: false, fetching: null, timer: null, cost: null, disposition: null, stopped: false,
    stale: false, following: null, followGeneration: 0, pendingOpen: null, pendingNoticed: null, opening: null, reconciled: 0};
  // `S.epoch` is the store whose ordinals the retained groups, watermark and cursor belong to.
  /* The declared Team event kinds the feed can name in words; any other kind is shown as declared.
   * The raw kind and payload stay in the row's exact observations. */
  const DECLARED_KINDS = {NATIVE_COORDINATION_MESSAGE: 'Team message', NATIVE_SUBAGENT_START_HOOK: 'Team hook · start', NATIVE_SUBAGENT_STOP_HOOK: 'Team hook · stop'};
  const declaredWords = (p) => { const kind = p?.event_kind; if (!DECLARED_KINDS[kind]) return {operation: kind || 'DECLARED', state: kind || 'DECLARED'}; const s = p.subject || {}; return {operation: t(DECLARED_KINDS[kind]), state: kind === 'NATIVE_COORDINATION_MESSAGE' ? (s.message_kind || t('message kind not declared')) : (s.native_hook_event || t('hook event not declared'))}; };
  const mayChange = (life) => stateMoving(life) || stateHeld(life);
  const shortRef = (v) => short(v, SHORT.id);
  const clock = (iso) => when(iso);
  /* One group per Task when the observation names one, else per operation reference; the group
   * is the row the reader sees, its items are the exact observations behind it. */
  const keyOf = (item) => item.task_id || item.payload?.operation_ref || item.observation_id;
  function absorb(item) {
    if (item.ordinal <= S.watermark) return null; // pages are in commit order: a re-delivered row is not a second event
    S.watermark = item.ordinal;
    const key = keyOf(item), ref = item.payload?.operation_ref;
    let g = S.groups.get(key);
    if (!g) { g = {key, items: [], task_id: item.task_id || null, last: 0}; S.groups.set(key, g); }
    // The REQUESTED half of an operation was recorded before its Task existed; once the RETURNED
    // half names the Task, the operation's earlier rows move under that Task.
    if (item.task_id && ref && ref !== key && S.groups.has(ref)) { g.items.push(...S.groups.get(ref).items); S.groups.delete(ref); }
    g.items.push(item); g.items.sort((a, b) => a.ordinal - b.ordinal);
    if (g.items.length > MAX_ITEMS) g.items.splice(0, g.items.length - MAX_ITEMS);
    g.last = Math.max(g.last, item.ordinal); g.at = item.occurred_at;
    if (item.task_id) g.task_id = item.task_id;
    return g;
  }
  /* Retention: the newest MAX_GROUPS groups by last ordinal; Task state only for retained groups. */
  function prune() {
    if (S.groups.size > MAX_GROUPS) {
      for (const g of [...S.groups.values()].sort((a, b) => a.last - b.last).slice(0, S.groups.size - MAX_GROUPS)) S.groups.delete(g.key);
    }
    const keep = new Set([...S.groups.values()].map((g) => g.task_id).filter(Boolean));
    if (S.following) keep.add(S.following); // the followed Task's projection is referenced by the choice
    for (const id of Object.keys(S.tasks)) if (!keep.has(id)) delete S.tasks[id];
  }
  /* Facts of a group. Three truths stay apart: what the operation returned, what Task Control
   * says *now* (the fresh projection wins over any recorded transition), and whether an owner
   * verified a result. A recorded RECOVERY_REQUIRED never outranks a current RUNNING. */
  function facts(g) {
    const ops = g.items.filter((v) => v.schema_kind === 'ProductOperationObserved');
    const declared = g.items.filter((v) => v.schema_kind === 'ExternalActivityObserved');
    const transition = g.items.filter((v) => v.schema_kind === 'TaskControlTransition').at(-1);
    const artifact = g.items.filter((v) => v.schema_kind === 'ArtifactVerificationObserved').at(-1);
    const returned = ops.filter((v) => v.payload?.phase !== 'REQUESTED').at(-1);
    const requested = ops.find((v) => v.payload?.phase === 'REQUESTED') || ops[0];
    const p = (returned || requested)?.payload || {};
    const live = g.task_id ? S.tasks[g.task_id] || null : null;
    const gap = g.items.some((v) => v.availability !== 'AVAILABLE');
    let state, tone, basis;
    if (live?.status === 'REFUSED') { state = 'REFUSED'; tone = 'refused'; basis = 'unavailable'; }
    else if (live) { state = live.lifecycle; tone = state.toLowerCase(); basis = 'current'; }
    else if (transition?.payload) { state = transition.payload.task_lifecycle; tone = state.toLowerCase(); basis = 'recorded'; }
    else if (returned) { state = p.status || (p.phase === 'FAILED' ? 'FAILED' : 'RETURNED'); tone = state.toLowerCase(); basis = 'returned'; }
    else if (requested) { state = 'REQUESTED'; tone = 'running'; basis = 'returned'; }
    else if (declared.length) { state = declaredWords(declared.at(-1).payload).state; tone = 'metadata'; basis = 'declared'; }
    else { state = gap ? 'PAYLOAD_NOT_RETAINED' : 'RECORDED'; tone = 'metadata'; basis = 'recorded'; }
    return {ops, declared, transition, artifact, returned, requested, payload: p, live, gap, state, tone, basis,
      verified: Boolean(artifact?.payload), artifactKind: artifact?.payload?.artifact_kind || null, artifactHash: artifact?.payload?.artifact_hash || null,
      taskKind: live?.task_kind || transition?.payload?.task_class || null,
      operation: p.operation || (declared.length ? declaredWords(declared.at(-1).payload).operation : '') || (transition ? transition.payload?.task_class : '') || '',
      caller: p.caller || (declared.length ? 'EXTERNAL_CLIENT' : transition ? 'TASK_CONTROL' : '')};
  }
  function subjectLine(f, g, locators = false) {
    const s = f.payload.subject || {};
    const parts = [];
    const identity = (value) => /^[0-9a-f]{16,}$|^[0-9a-f]{8}-[0-9a-f-]{27}$/i.test(String(value));
    const named = (label, value) => locators ? html`${label} ${identity(value) ? hashCell(value, SHORT.id) : locatorCell(value)}` : label + ' ' + (identity(value) ? shortRef(value) : value);
    if (g.task_id) parts.push(locators ? html`${t('Task')} ${hashCell(g.task_id, SHORT.id)}` : t('Task') + ' ' + shortRef(g.task_id));
    for (const [k, label] of [['spec_hash', 'spec'], ['plan_hash', 'plan'], ['experiment_plan_hash', 'plan'], ['result_hash', 'result'], ['cached_result_hash', 'cached result'], ['publication_task_id', 'published task'], ['research_input_id', 'input'], ['strategy_package_id', 'strategy'], ['candidate_hash', 'candidate'], ['review_publication_hash', 'review']]) {
      if (s[k] !== undefined && s[k] !== null && s[k] !== '') parts.push(named(t(label), s[k]));
    }
    if (f.verified) parts.push(locators ? html`${codeWords(f.artifactKind)} ${hashCell(f.artifactHash, SHORT.id)}` : codeWords(f.artifactKind) + ' ' + shortRef(f.artifactHash));
    if (f.declared.length && !parts.length) parts.push(f.declared.at(-1).payload?.summary || '');
    return locators ? joinMarkup(parts) : parts.join(' · ');
  }
  function metaLine(f) {
    return [html`<span>${codeWords(f.state)}</span>`, // the state in words, never the code
      f.basis === 'recorded' && f.transition ? html`<span>${t('recorded at command return · not re-read')}</span>` : '',
      f.live?.status === 'REFUSED' ? html`<span>${explainCode(f.live.failure_code) || t(f.live.detail)}</span>` : f.live ? html`<span>${f.live.verified_stage_count} / ${f.live.total_stage_count} ${t('stages')}${f.live.current_stage ? ' · ' + codeWords(f.live.current_stage) : ''}</span>` : '',
      f.verified ? stateLine('verified', {word: t('Result verified by its owner')}) : f.transition?.payload?.task_lifecycle === 'SUCCEEDED' ? html`<span>${t('completed · result through its readback')}</span>` : '',
      f.payload.failure_code ? html`<span>${codeWords(f.payload.failure_code)}${f.payload.failure_type && f.payload.failure_code === 'activity.failure_detail_withheld' ? html` · ${codeWords(f.payload.failure_type)}` : ''}</span>` : '',
      f.live?.worker_failure_type ? html`<span>${t('worker stop')} · ${codeWords(f.live.worker_failure_type)}</span>` : '',
      f.gap ? html`<span>${t('some payloads no longer retained')}</span>` : ''];
  }
  /* The one read that follows: opening goes by the clicked object's own kind and reference. */
  function nextRead(f, g) {
    const read = f.payload.next_read, out = [];
    if (f.live?.status === 'REFUSED') out.push(prerequisiteWays(f.live.next_requests));
    if (f.verified) out.push(btn(t('Open result'), 'activity-open', g.key, 'menu-row'));
    else if (g.task_id) out.push(btn(t('Inspect task'), 'task', g.task_id, 'menu-row'));
    else if (read?.operation === 'REPORT' && read.result_hash) out.push(LiveViews.savedObjectLink(t('Open this Portfolio result'), 'result:' + read.result_hash, 'menu-row'));
    if (followable(f, g)) out.push(g.task_id && S.following === g.task_id ? btn(t('Following'), 'activity-pin', '', 'menu-row') : btn(t('Follow'), 'activity-follow', g.key, 'menu-row'));
    const session = typeof LiveTeam !== 'undefined' ? g.items.map(LiveTeam.sessionOf).find(Boolean) : null;
    if (session) out.push(btn(t('Team scene'), 'team-open', session, 'menu-row'));
    return out;
  }
  /* Following is a viewer's choice to be moved along: a Task still in flight, or a shared PLAN
   * that can be inspected. Pinning (the default) keeps the current view whatever arrives. */
  const sharedPlanOf = (f) => f.payload.subject?.plan_hash || f.payload.subject?.experiment_plan_hash || null;
  function followable(f, g) {
    if (f.live?.status === 'REFUSED') return false;
    if (g.task_id) return !f.verified && (!f.live || mayChange(f.live.lifecycle));
    return Boolean(sharedPlanOf(f)) && f.payload.operation === 'EXPERIMENT_PLAN' && f.payload.phase === 'RETURNED';
  }
  function follow(key) {
    const g = S.groups.get(key);
    if (!g) throw Error('This activity entry is no longer retained.');
    const f = facts(g);
    if (g.task_id) { setFollowing(g.task_id); notify('Following Task {task} · its verified result will open here', {task: shortRef(g.task_id)}); return; }
    const hash = sharedPlanOf(f);
    if (hash) return LiveResearch.inspectShared(hash);
    throw Error('Nothing to follow in this entry.');
  }
  /* A Task that stopped moving (Data reports the move): said once, as a toast with one press to open
   * it, and -- while the page is in the background and the viewer turned it on in Settings -- as a
   * system notification (the user, 2026-09-25: 任何任务结束都提示 / 切走时也能知道). The followed Task
   * says its own ending (its result opens) and is not said twice. The words are the state's. */
  const ENDED_WORDS = {succeeded: 'Task completed', failed: 'Task failed', cancelled: 'Task cancelled', refused: 'Task refused', blocked: 'Task blocked', review_pending: 'Task needs a decision', recovery_required: 'Task needs recovery', deferred: 'Task deferred'};
  const noticesOffered = () => typeof Notification !== 'undefined';
  const noticesWanted = () => { try { return readPreference('taskNotices') === true; } catch { return false; } };
  function taskSettled(v) {
    if (!v || v.task_id === S.following) return;
    const word = ENDED_WORDS[String(v.lifecycle || '').toLowerCase()];
    if (!word) return;
    const name = (typeof LiveViews !== 'undefined' && LiveViews.nameOf ? LiveViews.nameOf(v)?.name : '') || (v.goal_summary ? t(v.goal_summary) : '') || codeWords(v.task_kind || v.kind || '');
    notify(word, {name}, {action: 'task', value: v.task_id, word: t('Open')});
    if (typeof document === 'undefined' || !document.hidden || !noticesWanted() || !noticesOffered() || Notification.permission !== 'granted') return;
    try {
      const n = new Notification(String(name), {body: t(word), tag: 'alphalattice-task-' + v.task_id});
      n.onclick = () => { try { window.focus(); } catch { /* the host may refuse */ } n.close(); if (typeof ACTIONS !== 'undefined' && ACTIONS.task) ACTIONS.task(v.task_id); };
    } catch { /* a browser that refuses the notification leaves the toast and the title to say it */ }
  }
  /* The viewer's choice (Settings): the browser asks once; a refusal is said on the row, never asked
   * again behind the viewer's back. */
  async function setNotices(on) {
    if (on && noticesOffered() && Notification.permission === 'default') { try { await Notification.requestPermission(); } catch { /* unanswered is not granted */ } }
    try { savePreference('taskNotices', Boolean(on) && noticesOffered() && Notification.permission === 'granted'); } catch { /* per viewer, best effort */ }
    if (app.page === 'settings') render();
  }
  const noticesState = () => !noticesOffered() ? 'unsupported' : Notification.permission === 'denied' ? 'denied' : noticesWanted() && Notification.permission === 'granted' ? 'on' : 'off';
  /* Following is one choice at a time: each change bumps a generation, and an automatic open
   * decided under an older generation, or on another page, is abandoned at the point where it
   * would actually move the reader. The hash records the choice so a reload restores it. */
  function setFollowing(taskId) {
    S.following = taskId || null; S.followGeneration += 1; S.pendingOpen = S.pendingNoticed = null; S.reconciled = 0;
    replaceHash({follow: S.following || ''});
    if (S.following) reconcileFollowed();
    paint();
  }
  function pin() { setFollowing(null); }
  const SETTLED = new Set(['SUCCEEDED', 'FAILED', 'BLOCKED', 'CANCELLED']);
  const busy = () => Boolean((typeof LiveResearch !== 'undefined' && LiveResearch.dirty && LiveResearch.dirty()) || document.querySelector('#dialog')?.open);
  const route = () => app.page + '|' + (app.book || '') + '|' + (typeof location !== 'undefined' ? location.hash.replace(/([#&])follow=[^&]*/, '$1') : '');
  const groupOfTask = (task) => [...S.groups.values()].find((g) => g.task_id === task) || null;
  /* What is known about the followed Task right now: a retained verified result opens; a
   * settled Task whose verification this reader never saw (restored after the event, or the
   * event left retention) is reconciled once against the owner's own readback; a Task that
   * ended without a result ends the follow. A moving Task waits for the feed. */
  function reconcileFollowed() {
    const task = S.following;
    if (!task || S.pendingOpen || S.opening) return;
    const g = groupOfTask(task);
    if (g && facts(g).verified) { S.pendingOpen = g.key; return openFollowed(); }
    const live = S.tasks[task] || Data.tasks().find((v) => v.task_id === task) || null;
    if (!live || !SETTLED.has(live.lifecycle) || S.reconciled === S.followGeneration) return;
    S.reconciled = S.followGeneration;
    if (live.lifecycle !== 'SUCCEEDED') { notify('Followed Task {task} ended {state}; there is no result to open', {task: shortRef(task), state: codeWords(live.lifecycle)}); setFollowing(null); return; }
    S.pendingOpen = 'task:' + task; openFollowed();
  }
  /* A followed Task's verified result opens once the reader is free to be moved: not over an
   * unsaved draft, a confirmation or an open dialog. Until then it waits, announced once. */
  function openFollowed() {
    const key = S.pendingOpen;
    if (!key) return;
    if (!key.startsWith('task:') && !S.groups.has(key)) { S.pendingOpen = null; return; }
    if (S.opening) return; // one automatic open at a time
    const task = key.startsWith('task:') ? key.slice(5) : S.groups.get(key).task_id;
    if (busy()) {
      if (S.pendingNoticed !== key) { S.pendingNoticed = key; notify('Followed Task {task} has a verified result; it opens when you finish editing', {task: shortRef(task)}); }
      return;
    }
    const intent = {key, task, generation: S.followGeneration, route: route()};
    S.pendingOpen = null; S.opening = intent;
    const wanted = () => intent.generation === S.followGeneration && S.following === intent.task && route() === intent.route && !busy();
    const settle = (opened) => {
      if (S.opening === intent) S.opening = null;
      if (intent.generation !== S.followGeneration) return; // the viewer chose again meanwhile
      if (opened === false) {
        // Abandoned at the boundary: a dialog or edit keeps it pending; a navigation cancels it.
        if (busy() && route() === intent.route) { S.pendingOpen = key; return; }
        setFollowing(null);
        notify('Followed Task {task} has a verified result; open it from the activity list', {task: shortRef(task)});
        return;
      }
      setFollowing(null);
    };
    const opening = key.startsWith('task:') ? openSettled(task, wanted) : open(key, wanted);
    Promise.resolve(opening).then(settle, (e) => { if (S.opening === intent) S.opening = null; notify(e.message); if (intent.generation === S.followGeneration) setFollowing(null); });
  }
  /* A settled followed Task without a retained verification: the owner says whether a result
   * exists. An experiment answers through its readback; an installed Portfolio replay through
   * the Portfolio owner's index and the REPORT of that exact result -- the same owners the
   * Host's artifact resolver reads -- and its reader is the saved `result:` entry. History's
   * `INSTALLED_RESULT` row is that entry; it is never searched for an authored study's kind. */
  async function openSettled(task, wanted) {
    const live = S.tasks[task] || Data.tasks().find((v) => v.task_id === task) || null;
    if (live?.task_kind === 'research_experiment') return LiveTasks.openResult(task, wanted);
    if (live && live.task_kind !== 'portfolio_public_development_replay') throw Error('This Task kind has no result reader in the workbench; open it from the original product.');
    const report = await Data.installedResult(task);
    if (!wanted()) return false;
    if (!report) throw Error('The Portfolio owner records no result for the followed Task; History lists it once discovered.');
    return openSavedResult(report.result_hash, wanted);
  }
  /* A saved Portfolio result opens the way History opens it; until History lists the entry, the
   * original product's link to that exact entry is offered. Reading only; nothing recomputes. */
  async function openSavedResult(hash, wanted = () => true) {
    const entry = 'result:' + hash;
    if (!Data.history().some((r) => r.id === entry)) await Data.refreshHistory();
    if (!wanted()) return false;
    if (Data.history().some((r) => r.id === entry)) return Data.openEntry(entry);
    return openDialog(t('Saved object · readback'), shortRef(hash), html`<p class="caption">${t('This result is not in the discovered history yet; History lists it once discovered.')}</p>`, LiveViews.savedObjectLink(t('Find it in History'), entry));
  }
  /* Open the verified result of one group by what it is: a Portfolio result is a saved
   * `result:` object and opens the way History opens it; an experiment result opens through
   * the experiment readback. Neither consults which Task the Task Center has selected. An
   * automatic open passes `wanted`, re-checked after every wait and before the move itself. */
  async function open(key, wanted = () => true) {
    const g = S.groups.get(key);
    if (!g) throw Error('This activity entry is no longer retained. Refresh and open it from History.');
    const f = facts(g);
    if (!f.verified) { if (g.task_id) return LiveTasks.open(g.task_id); throw Error('No verified result is recorded for this entry.'); }
    if (f.artifactKind === 'PortfolioResearchResult') return openSavedResult(f.artifactHash, wanted);
    if (f.artifactKind === 'ResearchExecutionEvidence' && g.task_id) return LiveTasks.openResult(g.task_id, wanted);
    if (g.task_id) return LiveTasks.open(g.task_id);
    throw Error('This result kind has no reader in the workbench yet.');
  }
  /* One recorded group as a row (round 20): the state as a dot, the operation and its caller as
   * the title, the reference and the state's words as properties, the clock at the end; the way
   * is the verified result or the Task; the reads that follow are the row's hover actions and the
   * exact observations and the full state facts a fact opener on the row. The key anchors reading state through a
   * repaint (an open disclosure, focus, scroll), as before. */
  function row(g) {
    const f = facts(g);
    const attention = ['REFUSED', 'FAILED', 'BLOCKED', 'RECOVERY_REQUIRED', 'CANCELLED'].includes(f.state) || f.gap;
    const state = codeWords(f.state);
    const exact = g.items.map((v) => logLine(lineOf(v))), meta = metaLine(f);
    const lead = statusDot(f.tone, state);
    const title = html`${codeWords(f.operation)}<span class="fv-log-by"> · ${actorWords(f.caller, f.payload?.producer_id)}</span>`;
    const way = f.verified ? {action: 'activity-open', value: g.key} : g.task_id ? {action: 'task', value: g.task_id} : null;
    const refusedSubject = f.state === 'REFUSED' ? subjectLine(f, g, true) : '';
    if (refusedSubject) meta[4] = joinMarkup([meta[4], refusedSubject].filter(Boolean));
    const observations = html`${factsRef(html`${t('Exact observations')} · ${g.items.length}`, html`${meta.filter(Boolean).map(bit => html`<p>${bit}</p>`)}${exact}${f.payload.latency_milliseconds !== undefined ? html`<p>${t('Owner latency')}: ${f.payload.latency_milliseconds} ms</p>` : ''}`)}`;
    return objectRow({lead: lead, name: title, why: refusedSubject ? '' : subjectLine(f, g) || t('No reference recorded'), to: way, cls: 'activity-row' + (attention ? ' attention' : '')}, {columns: ['state', 'basis', 'live', 'result', 'failure', 'worker', 'gap', 'observations'], props: [...meta.map((bit, index) => bit && index && !attention && f.basis === 'current' ? ['', bit, 'drop'] : bit), observations], time: clock(g.at), actions: nextRead(f, g), attrs: html`data-activity-key="${g.key}"`});
  }
  /* The record of one object (round 67): the recorded groups that name its Task, as the feed's
   * own rows in recorded order; the exact observations under each. */
  function recordOf(taskId) {
    const groups = [...S.groups.values()].filter((g) => g.task_id === taskId).sort((a, b) => a.last - b.last);
    if (!groups.length) return html`<p class="caption">${t('Nothing recorded for this object in the retained activity window.')}</p>`;
    return html`<div class="card-list lines slotted">${groups.map(row)}</div>`;
  }
  /* The Task's log (round 72): every retained observation of the Task as one line -- the instant,
   * who, the words, the telling reference as code -- in recorded order; `starterOf` is who admitted
   * it (the first operation the record holds for it). */
  const lineOf = (v) => observationLine(v, (p) => { const d = declaredWords(p); return [t('External client · not verified'), html`${codeWords(d.operation)} · ${codeWords(d.state)}`, '']; });
  const logLines = (taskId) => [...S.groups.values()].filter((g) => g.task_id === taskId).sort((a, b) => a.last - b.last).flatMap((g) => g.items).map(lineOf);
  function starterOf(taskId) {
    for (const g of [...S.groups.values()].filter((g) => g.task_id === taskId).sort((a, b) => a.last - b.last)) for (const v of g.items) if (v.schema_kind === 'ProductOperationObserved' && v.payload?.caller) return actorWords(v.payload.caller, v.payload.producer_id);
    return '';
  }
  function ordered() {
    return [...S.groups.values()].sort((a, b) => b.last - a.last).slice(0, VISIBLE);
  }
  function banners() {
    const out = [], o = S.observer;
    if (S.error) out.push(banner(t('Activity feed unreachable'), html`${S.error}<br>${t('A transport failure is not an execution failure: Tasks continue and their owners are re-read when the feed returns.')}`, 'warning'));
    if (o?.status === 'UNAVAILABLE') out.push(banner(t('Activity recording unavailable'), html`<span class="mono">${o.store_failure || ''}</span><br>${t('The product runs unobserved: operations, Tasks and results still work and are read through their own routes.')}${S.stale ? html`<br>${t('Entries shown are retained from before the store became unavailable; they are not current.')}` : ''}`, 'warning'));
    else if (o?.status === 'DEGRADED') out.push(banner(t('Activity recording degraded'), html`${t('{n} observations missing since {since}', {n: o.missing_observations, since: o.first_failure_at || ''})} · <span class="mono">${o.last_failure_code || o.hook_failure_type || o.entry_failure_type || ''}</span><br>${t(o.recording === 'OK' ? 'Recording works again; the gap remains a gap.' : 'Recording is still failing; operations and Tasks are unaffected.')}`, 'warning'));
    if (S.notice) out.push(noteLine(t('Activity feed reset'), t(S.notice)));
    if (S.unavailable) out.push(noteLine(t('Retention gap'), t('{n} recorded entries no longer retain their payload.', {n: S.unavailable})));
    return out;
  }
  /* The read cost of the latest poll. It changes every poll, so it is written into the
   * painted section's own status line (`paintCost`) rather than into the markup a repaint
   * compares: a poll that changed nothing the reader sees replaces nothing. */
  const costLine = () => (S.cost ? t('{n} entries read in {ms} ms · cursor {cursor}', {n: S.cost.observations, ms: S.cost.elapsed_ms, cursor: String(S.cursor || '').slice(-8)}) : t('Not read yet'));
  function paintCost() { if (typeof document === 'undefined' || !document.querySelectorAll) return; for (const el of document.querySelectorAll('[data-activity-cost]')) el.textContent = costLine(); }
  // N6: the followed Task is a state with its control (notes); the log's sentence is the label's (i)
  function section() {
    const rows = ordered();
    const following = S.following ? html`${t('Following Task {task} · its verified result will open here; pin to keep the current view.', {task: shortRef(S.following)})} ${btn(t('Pin current view'), 'activity-pin', '', 'text-btn')}` : '';
    return runLog({id: 'activityLog', title: t('Workspace activity'), caption: t('Recorded operations, Task returns and verified results · newest first · no simulated progress'), notes: html`${following ? html`<p class="run-log-following">${following}</p>` : ''}${banners()}`, lines: rows.map(row), empty: emptyState(t('No recorded activity yet.')), status: html`<span data-activity-cost>${t('Not read yet')}</span>`, cls: 'activity-log', linesCls: 'card-list lines slotted activity-rows'});
  }

  /* Announce, never navigate (round 75: count, never toast): a new Task, a verified result, a
   * refusal or a declared event counts as unseen -- the sidebar's number, the rows' state lines
   * and the record say it; a toast is for the person's own actions. The reader's open object,
   * page and Task Center selection stay exactly where they are. */
  function notice(item) {
    const p = item.payload || {};
    if (item.schema_kind === 'ArtifactVerificationObserved') return ['Result verified · Task {task} · open it from the activity list', {task: shortRef(item.task_id)}];
    if (item.schema_kind === 'ProductOperationObserved' && p.phase === 'RETURNED' && p.task_id && p.status === 'ADMITTED') return ['{caller} admitted {op} · Task {task} {state}', {caller: actorWords(p.caller, p.producer_id), op: codeWords(p.operation), task: shortRef(p.task_id), state: codeWords(p.task_lifecycle) || ''}];
    if (item.schema_kind === 'TaskControlTransition' && ['BLOCKED', 'CANCELLED', 'RECOVERY_REQUIRED'].includes(p.task_lifecycle)) return ['Task {task} · {state}', {task: shortRef(item.task_id), state: codeWords(p.task_lifecycle)}];
    if (item.schema_kind === 'ProductOperationObserved' && p.status === 'REFUSED') return ['{op} refused · {code}', {op: codeWords(p.operation), code: codeWords(p.failure_code) || ''}];
    if (item.schema_kind === 'ExternalActivityObserved') { if (!DECLARED_KINDS[p.event_kind]) return ['External client declared {kind}', {kind: p.event_kind}]; const w = declaredWords(p); return ['External client declared {kind} · {state}', {kind: w.operation, state: String(w.state).replaceAll('_', ' ')}]; }
    return null;
  }
  function announce(item) {
    if (S.following && item.task_id === S.following && item.schema_kind === 'ArtifactVerificationObserved') S.pendingOpen = keyOf(item);
    if (S.primed && notice(item)) S.unseen += 1;
  }
  function rebuild(reason) {
    // Only this module's own state: never the open object, the Task Center selection or a draft.
    S.notice = reason; S.groups.clear(); S.tasks = {}; S.watermark = 0;
  }
  /* One page into state. Retained groups, the watermark and the cursor all belong to one store
   * (`S.epoch`). A page from another store -- RESET, or a TAIL/CONTINUED that names a different
   * epoch -- rebuilds that state from the page; a TAIL on the same store after an outage keeps
   * the retained groups and names the rows it could not read. UNAVAILABLE keeps everything,
   * including the cursor, so the next read continues on the same store or is RESET by a new one. */
  function absorbPage(body) {
    const continuous = S.primed && !S.stale && !S.error && body.epoch === S.epoch && body.disposition === 'CONTINUED' && !body.more;
    if (typeof LiveTeam !== 'undefined' && LiveTeam.arrivals) LiveTeam.arrivals(body, continuous);
    if (body.disposition === 'UNAVAILABLE') {
      S.disposition = body.disposition; S.observer = body.observer || null; S.cost = body.read_cost || null;
      S.stale = S.groups.size > 0;
      mergeFresh(body); S.primed = true;
      return;
    }
    const storeChanged = S.epoch !== null && body.epoch !== S.epoch;
    if (storeChanged) rebuild('The activity store changed; the list was rebuilt from the tail of the new store and Task owners are re-read.');
    else if (body.disposition === 'RESET') rebuild('The cursor was no longer valid for this store; the list was rebuilt from the current tail and Task owners are re-read.');
    else if (body.disposition === 'TAIL' && S.watermark > 0 && body.items?.length && body.items[0].ordinal > S.watermark + 1) {
      S.notice = 'The feed resumed from the tail of the store; entries recorded in between were not read and are not shown.';
    }
    // A rebuilt list (another store, or a cursor the store no longer honours) is read from the
    // tail as a baseline: its rows are retained facts, not events that just happened, so none of
    // them is announced or counted as unseen -- the reset notice above says what happened. Rows
    // that arrive after that baseline, and every row of an ordinary CONTINUED or TAIL page,
    // announce as before. A followed Task's verification in the baseline still reconciles below.
    const baseline = storeChanged || body.disposition === 'RESET';
    S.disposition = body.disposition; S.epoch = body.epoch; S.cursor = body.cursor; S.observer = body.observer || null; S.cost = body.read_cost || null;
    S.stale = false;
    S.unavailable += body.unavailable || 0;
    for (const item of body.items || []) { if (absorb(item) && !baseline) announce(item); }
    mergeFresh(body);
    prune();
    S.primed = true;
    if (S.pendingOpen) openFollowed(); else if (S.following) reconcileFollowed();
  }
  function mergeFresh(body) {
    const fresh = Object.values(body.tasks || {});
    for (const v of fresh) if (S.tasks[v.task_id] || v.task_id === S.following || [...S.groups.values()].some((g) => g.task_id === v.task_id)) S.tasks[v.task_id] = v;
    if (fresh.length) Data.mergeTasks(fresh);
  }
  function watched() {
    const moving = [...S.groups.values()].filter((g) => g.task_id && (!S.tasks[g.task_id] || mayChange(S.tasks[g.task_id].lifecycle))).sort((a, b) => b.last - a.last).map((g) => g.task_id);
    // The followed Task is watched even without a retained group, so a restored follow can reconcile.
    return [...new Set([...(S.following ? [S.following] : []), ...moving])].slice(0, 16);
  }
  const cadence = () => (S.error ? BACKOFF : document.hidden ? HIDDEN_INTERVAL : INTERVAL);
  async function refresh() {
    if (S.fetching) return S.fetching;
    clearTimeout(S.timer);
    let behind = false;
    S.fetching = (async () => {
      try {
        for (let pages = 0; pages < MAX_PAGES; pages++) {
          const query = new URLSearchParams({limit: String(LIMIT)});
          if (S.cursor) query.set('after', S.cursor);
          const watch = watched();
          if (watch.length) query.set('watch', watch.join(','));
          const page = await Data.readShared('/api/activity?' + query, true);
          absorbPage(page);
          behind = Boolean(page.more);
          if (!behind || page.disposition === 'UNAVAILABLE') break;
        }
        S.error = '';
      } catch (e) { S.error = e.message; }
    })();
    try { await S.fetching; }
    finally {
      S.fetching = null; paint();
      if (!S.stopped) S.timer = setTimeout(() => { S.timer = null; refresh(); }, behind && !S.error ? CATCH_UP : cadence());
    }
  }
  function paint() {
    if (typeof LiveTasks !== 'undefined' && LiveTasks.paintActivity) LiveTasks.paintActivity();
    else Window.renderSide(); // the Tasks row's count (round 62)
    paintCost();
    if (typeof LiveTeam !== 'undefined' && LiveTeam.refresh) LiveTeam.refresh();
    if (typeof LiveGoals !== 'undefined' && LiveGoals.observe) void LiveGoals.observe();
    if (typeof LiveWorkspace !== 'undefined' && LiveWorkspace.observe) LiveWorkspace.observe();
    if (typeof LiveResearch !== 'undefined' && LiveResearch.observe) LiveResearch.observe(); // the experiment Task read on the Lab page
    if (typeof LiveReview !== 'undefined' && LiveReview.observe) LiveReview.observe(); // the evidence or review Task read on the review pages
    if (typeof LiveActivation !== 'undefined' && LiveActivation.observe) void LiveActivation.observe();
    if (typeof Settings !== 'undefined' && Settings.observeUpdate) void Settings.observeUpdate();
    if (typeof Settings !== 'undefined' && Settings.observeStorageCap) void Settings.observeStorageCap();
  }
  function markSeen() { S.unseen = 0; S.notice = ''; S.unavailable = 0; paint(); }
  /* Teardown and return. `pagehide` ends the loop for a page that is going away or into the
   * back/forward cache; a `pageshow` that restores that page resumes it. `refresh` joins an
   * in-flight request instead of starting a second loop, and its completion schedules exactly
   * one timer once the page is live again. */
  function stop() { S.stopped = true; clearTimeout(S.timer); S.timer = null; }
  /* FLOW-1: a Team or Goal page asks the Host to read the bound Sessions' usage once as it opens;
   * nothing reads on a timer, and a failed read shows only in the usage state. */
  let usageAskedAt = 0;
  function readSessionUsage() {
    if (!Data.offers('SESSION_USAGE_READ') || Date.now() - usageAskedAt < 10000) return;
    usageAskedAt = Date.now();
    Promise.resolve().then(() => Data.post(Data.route('SESSION_USAGE_READ'), {})).catch(() => {});
  }
  function resume() { if (!S.stopped) return; S.stopped = false; void refresh(); }
  function bind() {
    document.addEventListener('visibilitychange', () => { if (!document.hidden && !S.stopped) refresh(); });
    window.addEventListener('pagehide', stop);
    window.addEventListener('pageshow', (event) => { if (event.persisted) resume(); });
  }
  return {section, logLines, starterOf, recordOf, refresh, markSeen, bind, facts, nextRead, absorbPage, open, openSavedResult, cadence, stop, resume, follow, pin, setFollowing, taskSettled, setNotices, noticesState, readSessionUsage,
    retained: () => [...S.groups.values()],
    nativeUsageState: () => S.observer?.native_usage || null,
    state: () => ({cursor: S.cursor, epoch: S.epoch, groups: S.groups.size, unseen: S.unseen, error: S.error, notice: S.notice, tasks: Object.keys(S.tasks).length, watermark: S.watermark, disposition: S.disposition, stale: S.stale, stopped: S.stopped, fetching: Boolean(S.fetching), timer: S.timer !== null, following: S.following, pendingOpen: S.pendingOpen, opening: Boolean(S.opening), generation: S.followGeneration}),
    groups: ordered, unseen: () => S.unseen};
})();
