/* Native identity and member text are observations, not completion. Exact Task, hash and History references join owner readbacks; one feed merges records and arrivals. */
const LiveTeam = (() => {
  const KINDS = {NATIVE_SUBAGENT_START_HOOK: 'hook', NATIVE_SUBAGENT_STOP_HOOK: 'hook', NATIVE_COORDINATION_MESSAGE: 'message', NATIVE_AGENT_USAGE: 'usage'};
  // U25: the decision notes (决策笔记) a member writes are messages of their own kinds, in the conversation
  const MESSAGE_KINDS = {issue: ['Bundle issued', 'arrow'], correction: ['Correction', 'warning'], acceptance: ['Acceptance', 'check'], assignment: ['Assignment', 'arrow'], question: ['Question', 'info'], answer: ['Answer', 'file'], objection: ['Objection', 'warning'], pm_response: ['PM response', 'user'], plan: ['Plan', 'flag'], decision: ['Decision', 'check'], dead_end: ['Dead end', 'ban'], surprise: ['Surprise', 'info'], session_bound: ['Session bound', 'check'], stance: ['Stance', 'flag'], challenge: ['Challenge', 'warning'], reply: ['Reply', 'arrow'], ruling: ['Ruling', 'user'], person_answer: ['Person answer', 'user'], verdict: ['Verdict', 'file']};
  const NOTE_KINDS = new Set(['plan', 'decision', 'dead_end', 'surprise']);
  const CHANNELS = {CODEX_HOOK: 'Codex hook input · not host-authenticated', CLAUDE_CODE_HOOK: 'Claude Code hook input · not host-authenticated', ACTOR_DECLARED: 'Actor-declared text · not host-verified', PRODUCT_ACCEPTED_ANSWER: 'Accepted structured answer · product record, not a native spoken turn'};
  const PM_ROLE = 'research_lead';
  /* Product identity fields an operation may name; nothing else in a subject is a reference. */
  const PRODUCT_KEYS = ['case_hash', 'task_id', 'experiment_task_id', 'publication_task_id', 'existing_task_id', 'update_task_id', 'left_task_id', 'right_task_id', 'development_task_id', 'experiment_plan_hash', 'plan_hash', 'result_hash', 'cached_result_hash', 'review_publication_hash', 'analysis_publication_hash', 'candidate_hash', 'report_hash', 'export_hash', 'foundation_admission_hash', 'publication_hash', 'answer_reference', 'bundle_reference'];
  const MAX_RESOLVED = 64;
  /* A session's shared head serves its conversation, committee and recorded views. */
  const PAGES_SET = new Set(['team', 'team-committee', 'team-participants', 'team-outputs', 'team-evidence', 'team-sessions']);
  const onTeam = () => PAGES_SET.has(app.page);
  /* One owner feed merges native records and arrivals; epoch/session keys prevent repeat reads. */
  const EXTERNAL_LIMIT = 200, EXTERNAL_KEEP = 400;
  const X = {items: new Map(), product: [], productSession: '', epoch: null, fetching: null, error: '', read: false, more: false, oldest: null, keep: EXTERNAL_KEEP, selectedKey: '', selectedMore: false, selectedOldest: null};
  function ensureExternal() {
    if (X.fetching || Data.workspaceStatus !== 'ready') return;
    const epoch = LiveActivity.state().epoch, session = onTeam() && app.page !== 'team-sessions' ? selection().session : '', key = [epoch, session].join('|');
    if (session ? X.selectedKey === key : X.read && (!epoch || X.epoch === epoch)) return;
    if (session) { X.selectedMore = false; X.selectedOldest = null; }
    const query = new URLSearchParams({limit: EXTERNAL_LIMIT, ...(session ? {native_session_id: session} : {})});
    X.fetching = Data.readShared('/api/activity/external?' + query).then((body) => {
      if (X.epoch && body.epoch && X.epoch !== body.epoch) { X.items.clear(); X.product = []; X.productSession = ''; X.read = false; X.selectedKey = ''; X.keep = EXTERNAL_KEEP; }
      for (const item of [...(body.items || []), ...(body.snapshots || [])]) X.items.set(item.observation_id, item);
      if (session) { X.product = body.product_items || []; X.productSession = session; }
      X.epoch = body.epoch || epoch; X.keep = Math.max(X.keep, X.items.size); X.error = '';
      if (session) { X.selectedMore = Boolean(body.more); X.selectedOldest = body.oldest ?? null; }
      else { X.more = Boolean(body.more); X.oldest = body.oldest ?? null; }
      if (body.snapshots_more) X.error = t('Older records are not read yet');
    }, (e) => { X.error = String(e?.message || e); X.epoch = epoch; }).finally(() => {
      if (session) X.selectedKey = key; else X.read = true;
      X.fetching = null; S.paintKey = '';
      refresh();
      if (onTeam()) paint(); else if (typeof patchMain === 'function') patchMain();
    });
  }
  /* F1, law 136: read one owner page per explicit request; keep it until the store epoch changes. */
  function readOlder() {
    const session = onTeam() && app.page !== 'team-sessions' ? selection().session : '', before = session ? X.selectedOldest : X.oldest, more = session ? X.selectedMore : X.more;
    if (X.fetching || !more || before == null) return;
    X.fetching = Data.readShared('/api/activity/external?' + new URLSearchParams({limit: EXTERNAL_LIMIT, before, ...(session ? {native_session_id: session} : {})})).then((body) => {
      for (const item of [...(body.items || []), ...(body.snapshots || [])]) X.items.set(item.observation_id, item);
      if (session && X.productSession === session) X.product = Data.uniqueRows([...X.product, ...(body.product_items || [])], i => i.observation_id);
      X.keep += (body.items || []).length; X.error = body.snapshots_more ? t('Older records are not read yet') : '';
      if (session) { X.selectedMore = Boolean(body.more); X.selectedOldest = body.oldest ?? before; }
      else { X.more = Boolean(body.more); X.oldest = body.oldest ?? before; }
    }, (e) => { X.error = String(e?.message || e); }).finally(() => { X.fetching = null; S.paintKey = ''; paint(); });
    paint();
  }
  const S = {resolved: new Map(), busy: new Set(), visible: 50, incoming: new Set(), flash: new Set(), factNew: new Set(), factVerified: new Set(), factFlash: new Set(), paintKey: '', retained: null, question: new Map(), unfolded: new Set(), replies: new Set(), words: new Set(), factKind: '', answerDetail: null};
  /* Session labels keep a short head plus the differing suffix across retained peers; short ids stay whole. */
  let peerPrefix = 0; // the retained sessions' common prefix, measured by scene()
  function sessionLabel(id) {
    const s = String(id || '');
    if (!s) return t('no session declared');
    if (s.length <= 22) return s;
    const cut = peerPrefix >= 8 && peerPrefix < s.length - 2 ? peerPrefix : 8;
    const rest = s.slice(cut);
    return s.slice(0, SHORT.id) + '…' + (rest.length > 14 ? rest.slice(0, 14) + '…' : rest);
  }
  // the reading grammar: to the minute (an exchange's exact stamp stays on the element's title)
  const clock = (iso) => when(iso); // N3 (law 133): one clock -- the same instant reads as the Sessions list reads it
  const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i, HASH = /^[0-9a-f]{64}$/i;
  const selection = () => ({session: hashParams().get('team') || '', actor: hashParams().get('actor') || '', event: hashParams().get('event') || '', committee: hashParams().get('committee') || '', point: hashParams().get('committee_point') || ''});
  const ROLE_NAMES = {research_lead: 'Main PM', alphalattice_data: 'Data', alphalattice_factor: 'Factor & Foundation', alphalattice_alpha: 'Alpha Modeling', alphalattice_risk: 'Risk Modeling', alphalattice_portfolio: 'Portfolio', alphalattice_evidence_analyst: 'Alternative Analyst', alternative_analyst: 'Alternative Analyst', alphalattice_cro: 'CRO', independent_cro: 'CRO', [HOST_ROLE]: 'Host'};
  const roleName = (role) => (role ? t(ROLE_NAMES[role] || role) : '');
  /* A qualified product reference, or null: a Task id, a 64-hex hash or a History entry id. */
  function qualify(value) {
    const v = String(value || '').trim();
    if (/^review-thread:[0-9a-f-]+$/i.test(v) && UUID.test(v.slice(14))) return {kind: 'review_thread', ref: v.toLowerCase()};
    if (UUID.test(v)) return {kind: 'task', ref: v.toLowerCase()};
    if (HASH.test(v)) return {kind: 'hash', ref: v.toLowerCase()};
    if (/^case:[0-9a-f]{64}$/i.test(v)) return {kind:'case',ref:v.toLowerCase()};
    const m = v.match(/^(experiment|result|review):([0-9a-f-]{8,64})$/i);
    if (m) return {kind: 'history', ref: m[1].toLowerCase() + ':' + m[2].toLowerCase()};
    return null;
  }
  /* The qualified references one product observation names, from its typed identity fields. */
  function productReferences(item) {
    const p = item.payload || {}, found = [];
    if (item.schema_kind === 'ProductOperationObserved') {
      for (const key of PRODUCT_KEYS) { const q = qualify(p.subject?.[key]); if (q) found.push(q.ref); }
      for (const key of ['task_id', 'publication_task_id']) { const q = qualify(p[key]); if (q) found.push(q.ref); }
    } else if (item.schema_kind === 'ArtifactVerificationObserved') {
      const q = qualify(p.artifact_hash); if (q) found.push(q.ref);
    }
    if (item.schema_kind !== 'ExternalActivityObserved') { const q = qualify(item.task_id); if (q) found.push(q.ref); }
    return found;
  }
  /* ---- the scene, derived from the retained feed ---- */
  const sessionState = id => ({id, participants: new Map(), entries: new Map(), byEvent: new Map(), references: new Set(), producers: new Set(), facts: [], unknownKinds: [], goals: new Set(), last: 0});
  function scene(fullRecord = false, includeSessionMembers = false) {
    const sessions = new Map(), unknown = [];
    ensureExternal();
    const external = TeamMembers.externalItems(X), scopes = new Map(), contributors = new Map();
    for (const item of external) {
      const sub = item.payload?.subject || {}, id = sub.native_session_id;
      if (sub.message_kind === 'session_bound' && sub.input_channel === 'PRODUCT_OPERATION' && sub.native_agent_id === id && UUID.test(sub.goal_id || '')) scopes.set(id, sub.goal_id);
      if (['hook', 'message'].includes(KINDS[item.payload?.event_kind]) && sub.goal_id && sub.native_agent_id) {
        const key = id + ':' + sub.goal_id;
        if (!contributors.has(key)) contributors.set(key, new Set());
        contributors.get(key).add(sub.native_agent_id);
      }
    }
    const session = (id) => { if (!sessions.has(id)) sessions.set(id, sessionState(id)); return sessions.get(id); };
    const participant = (s, id, role) => { if (!s.participants.has(id)) s.participants.set(id, {id, roles: new Set(), messages: [], hooks: [], usage: new Map(), pins: null, last: 0}); const p = s.participants.get(id); if (role) p.roles.add(role); return p; };
    for (const item of external) {
      const p = item.payload || {}, sub = p.subject || {};
      if (!p.event_kind) { unknown.push({item, reason: 'payload not retained'}); continue; }
      const kind = KINDS[p.event_kind], sid = sub.native_session_id || '';
      const scope = scopes.get(sid);
      const outsideGoal = scope && (sub.goal_id !== scope || (kind === 'usage' && !contributors.get(sid + ':' + scope)?.has(sub.native_agent_id)));
      const nativeMember = includeSessionMembers && kind === 'usage' && ['CODEX_SESSION_FILE', 'CLAUDE_CODE_SESSION_FILE'].includes(sub.input_channel);
      if (!fullRecord && outsideGoal && !nativeMember) continue;
      if (!kind) { (sid ? session(sid).unknownKinds : unknown).push({item, reason: 'unknown event kind'}); continue; }
      const s = session(sid);
      s.goalScope = scope || '';
      s.producers.add(p.producer_id + ':' + short(p.producer_session, SHORT.hash));
      if (sub.goal_id && (!outsideGoal || fullRecord)) s.goals.add(sub.goal_id); // A session-wide reading does not add a Goal contribution.
      if (kind === 'usage') { // U31: the latest cumulative reading per model, never a sum.
        if (sub.native_agent_id && sub.model) { const pt = participant(s, sub.native_agent_id, sub.role), held = pt.usage.get(sub.model), n = (k) => usageNumber(sub[k]);
          if (['true','false'].includes(sub.specialist_verified) && (!held || held.ordinal < item.ordinal)) pt.specialistVerified = sub.specialist_verified;
          if (!held || held.ordinal < item.ordinal) pt.usage.set(sub.model, {ordinal: item.ordinal, first: Date.parse(sub.first_at || ''), model: sub.model, efforts: String(sub.efforts || '').split(',').filter(Boolean), pinDiffers: String(sub.pin_differs || '').split(',').filter(Boolean), responses: n('responses'), input: n('input_tokens'), output: n('output_tokens'), cacheRead: n('cache_read_tokens'), cacheWrite: n('cache_write_tokens'), source_kind: sub.source_kind || (sub.input_channel === 'CODEX_SESSION_FILE' || sub.input_channel === 'CLAUDE_CODE_SESSION_FILE' ? sub.input_channel : null), sample_time_kind: sub.sample_time_kind || null, last_at: sub.last_at || null}); }
        s.last = Math.max(s.last, item.ordinal); continue;
      }
      // An observation is an identity; a native event is a grouping claim.
      const entry = {id: item.observation_id, ordinal: item.ordinal, kind, at: item.occurred_at, timeKind: sub.source_time_kind || null, channel: sub.input_channel || null, availability: item.availability, authority: item.authority, source: item.source_id + ':' + item.source_sequence, subject: sub, payload: p, actor: sub.native_agent_id || null, role: sub.role || null, declaredEvent: sub.native_event_id || null, groupKey: item.task_id || p.operation_ref || item.observation_id, taskId: item.task_id || null, replayOf: null, conflictsWith: []};
      if (kind === 'message') {
        entry.messageKind = Object.hasOwn(MESSAGE_KINDS, sub.message_kind) ? sub.message_kind : null;
        entry.declaredKind = sub.message_kind || null;
        entry.text = p.summary || ''; entry.truncated = p.summary_truncated === true;
        entry.bytes = sub.message_bytes || null; entry.sha = sub.message_sha256 || null; entry.messageId = sub.message_id || null;
        entry.recipient = sub.recipient_id || null; entry.reference = sub.reference || null; entry.replyTo = sub.reply_to || null; // U25: what it answers, named since GR2
        if (entry.channel === 'PRODUCT_REVIEW' && UUID.test(entry.reference || '')) entry.reference = 'review-thread:' + entry.reference.toLowerCase();
        entry.qualified = qualify(entry.reference);
        if (entry.qualified) { s.references.add(entry.qualified.ref); if (['history','case'].includes(entry.qualified.kind)) s.references.add(entry.qualified.ref.split(':')[1]); } // a qualified id names its own hash or Task
        if (acceptedArtifact(s, entry)) for (const name of ['answer_reference', 'bundle_reference']) s.references.add(sub[name].toLowerCase());
      } else {
        entry.hookEvent = sub.native_hook_event || null; entry.turn = sub.native_turn_id || null;
        entry.terminal = sub.terminal_state || null; entry.stopActive = sub.stop_hook_active || null;
      }
      if (p.task_verified && qualify(sub.task_id)) s.references.add(qualify(sub.task_id).ref);
      if (entry.actor) { const pt = participant(s, entry.actor, entry.role); (kind === 'message' ? pt.messages : pt.hooks).push(entry); pt.keyed ||= sub.authorship_basis === 'HOST_ACCEPTED'; pt.last = Math.max(pt.last, item.ordinal);
        if (kind === 'hook' && (sub.hook_model || sub.role_model || sub.role_effort)) pt.pins = {host: sub.hook_model || '', model: sub.role_model || '', effort: sub.role_effort || ''}; } // U31: the pins a hook carries
      if (entry.declaredEvent) {
        const same = s.byEvent.get(entry.declaredEvent) || [];
        const twin = same.find((other) => other.payload.summary === p.summary && JSON.stringify(other.subject) === JSON.stringify(sub));
        if (twin) { entry.replayOf = twin.id; twin.replays = [...(twin.replays || []), entry.id]; }
        else for (const other of same) { entry.conflictsWith.push(other.id); other.conflictsWith.push(entry.id); }
        same.push(entry); s.byEvent.set(entry.declaredEvent, same);
      }
      s.entries.set(entry.id, entry);
      s.last = Math.max(s.last, item.ordinal);
    }
    // Requests name their session; this proves no native authorship. Other facts join exact
    // references or one owner's Task return, never an External payload in a mixed group.
    for (const s of sessions.values()) {
      const product = [], held = new Set(), selected = (X.productSession === s.id ? X.product : []).map(item => ({key: item.task_id || item.payload?.operation_ref || item.observation_id, task_id: item.task_id || null, items: [item]}));
      for (const g of [...LiveActivity.retained(), ...selected]) {
        if (g.items.every((v) => v.schema_kind === 'ExternalActivityObserved')) continue;
        for (const item of g.items) if (item.schema_kind !== 'ExternalActivityObserved') {
          if (held.has(item.observation_id)) continue;
          held.add(item.observation_id);
          const p = item.payload || {}, sub = p.subject || {};
          const attributed = item.schema_kind === 'ProductOperationObserved'
            && item.source_kind === 'PRODUCT_OPERATION' && item.authority === 'OPERATIONAL_ASSERTION'
            && Boolean(s.id) && ['codex', 'claude-code'].includes(sub.agent_vendor) && sub.agent_session === s.id;
          product.push({item, group: g, refs: [...new Set(productReferences(item))], attributed});
        }
      }
      const derived = new Map(); // Task id -> the owner return that named it for this session
      for (const fact of product) {
        const p = fact.item.payload || {};
        if (fact.item.schema_kind !== 'ProductOperationObserved' || p.phase !== 'RETURNED') continue;
        const declared = fact.refs.filter((r) => s.references.has(r));
        for (const key of ['task_id', 'publication_task_id']) {
          const q = qualify(p[key]);
          if ((fact.attributed || declared.length) && q && !s.references.has(q.ref) && !derived.has(q.ref)) derived.set(q.ref, {operation: p.operation, status: p.status || null, via: declared[0] || s.id, by: fact.item.observation_id});
        }
      }
      for (const fact of product) {
        const named = fact.refs.filter((r) => s.references.has(r));
        const followed = fact.refs.filter((r) => !s.references.has(r) && derived.has(r) && derived.get(r).by !== fact.item.observation_id).map((r) => ({ref: r, ...derived.get(r)}));
        if (fact.attributed || named.length || followed.length) s.facts.push({...fact, named, followed});
      }
      s.derived = derived;
      s.facts.sort((a, b) => a.item.ordinal - b.item.ordinal);
    }
    const ids = [...sessions.keys()].filter(Boolean);
    peerPrefix = ids.length > 1 ? ids.reduce((n, id) => { let i = 0; while (i < n && i < id.length && id[i] === ids[0][i]) i++; return i; }, ids[0].length) : 0;
    TeamMembers.prune(sessions.values());
    return {sessions: [...sessions.values()].sort((a, b) => b.last - a.last), unknown};
  }
  const participantCount = s => TeamMembers.countOf(s, ROLE_SAME);
  /* A Main PM needs both the parent session id and research_lead role. */
  const isLead = (pt, s) => Boolean(s.id) && pt.id === s.id && pt.roles.has(PM_ROLE);
  /* Sessions and book Facts use the specialist order; equivalent cards share a role.
   * The Main PM leads every session and is no filter. */
  const ROLE_SAME = {alternative_analyst: 'alphalattice_evidence_analyst', independent_cro: 'alphalattice_cro'};
  const ROLE_ORDER = Object.keys(ROLE_NAMES);
  const rolesOf = (s) => [...new Set([...s.participants.values()].filter((pt) => !isLead(pt, s)).flatMap((pt) => [...pt.roles]).map((r) => ROLE_SAME[r] || r))]
    .filter((r) => r !== PM_ROLE && r !== HOST_ROLE).sort((a, b) => (ROLE_ORDER.indexOf(a) + 1 || 99) - (ROLE_ORDER.indexOf(b) + 1 || 99));
  function roleLabel(pt, s) {
    const roles = [...pt.roles];
    if (isLead(pt, s)) return html`${t('Main PM · declared foreground conversation')}${roles.length > 1 ? html` <span class="muted">${t('also declared as')} ${roles.filter((r) => r !== PM_ROLE).join(', ')}</span>` : ''}`;
    if (pt.roles.has(PM_ROLE)) return html`${PM_ROLE} <span class="muted">· ${t('declared research_lead, not the parent session')}</span>`;
    if (Boolean(s.id) && pt.id === s.id) return html`${roles.join(', ') || t('role not declared')} <span class="muted">· ${t('the parent session id without the research_lead role')}</span>`;
    return roles.length ? roles.map(roleName).join(', ') : '';
  }
  const actorName = (s, id) => { if (!id) return t('agent id not declared'); const pt = s.participants.get(id); return pt && pt.roles.size ? [...pt.roles].map(roleName).join('/') : TeamMembers.unique(s, id); };
  /* A path names one recipient through one bound start hook, never its spelling or authorship.
   * Conflicting paths or participants leave the recipient unknown. */
  function recipientParticipant(s, id) {
    if (!id) return null;
    const direct = s.participants.get(id);
    if (direct) return direct;
    const candidates = [...s.participants.values()].map((pt) => {
      const paths = new Set(pt.hooks.filter((h) => h.hookEvent === 'SubagentStart'
        && h.subject.native_host === 'codex' && h.subject.native_session_id === s.id
        && h.subject.native_agent_path_basis === 'CODEX_SESSION_META'
        && (!Object.hasOwn(h.subject, 'native_spawn_role') || h.subject.native_spawn_role === h.role))
        .map((h) => h.subject.native_agent_path).filter(Boolean));
      return {pt, paths};
    }).filter(({paths}) => paths.has(id));
    return candidates.length === 1 && candidates[0].paths.size === 1 ? candidates[0].pt : null;
  }
  const recipientName = (s, id, entry = null) => {
    const bundle = entry?.subject.bundle_reference;
    const roles = new Set(bundle && entry.messageKind === 'issue' ? [...s.entries.values()].filter((e) => e.channel === 'PRODUCT_REVIEW' && e.subject.authorship_basis === 'HOST_ACCEPTED' && e.replyTo === entry.messageId && e.subject.bundle_reference === bundle && e.role && ![PM_ROLE, HOST_ROLE].includes(e.role)).map((e) => ROLE_SAME[e.role] || e.role) : []);
    if (roles.size === 1) return roleName([...roles][0]);
    const pt = recipientParticipant(s, id); return pt?.roles.size ? actorName(s, pt.id) : entry?.messageKind === 'issue' && id ? TeamMembers.unique(s, id) : t('Unidentified');
  };
  /* One outcome per written reference; failed owner verification keeps its discovery and names the failure. */
  async function withOutcome(ref, work) {
    if (!ref || S.busy.has(ref)) return;
    S.busy.add(ref); paint();
    const previous = S.resolved.get(ref) || null;
    let outcome;
    try { outcome = await work(previous); } catch (e) { if(e.name==='AbortError')return;outcome = previous ? {...previous, failure: e.message} : {level: 'unresolved', target: null, failure: e.message}; }
    finally {S.busy.delete(ref);}

    S.resolved.delete(ref); S.resolved.set(outcome.level === 'verified' && outcome.reference ? outcome.reference : ref, outcome);
    while (S.resolved.size > MAX_RESOLVED) S.resolved.delete(S.resolved.keys().next().value);
    paint();
  }
  const historyRow = async (id) => { if (!Data.history().some((r) => r.id === id)) await Data.refreshHistory(); return Data.history().find((r) => r.id === id) || null; };
  /* History resolves result hashes and experiment Tasks; reviews have no owner readback here. */
  function historyTarget(id) {
    const [prefix, value] = id.split(':');
    if (prefix === 'result' && HASH.test(value)) return {kind: 'result', ref: value};
    if (prefix === 'experiment' && UUID.test(value)) return {kind: 'task', ref: value, taskKind: 'research_experiment'};
    return null;
  }
  /* Discovery: which owner's index names this reference. A hit is a candidate, not a verified object. */
  const resolve = (ref) => withOutcome(ref, async () => {
    const q = qualify(ref);
    if (ref.startsWith('goal:') && UUID.test(ref.slice(5))) return {level: 'discovered', owner: 'Goal', target: {kind: 'case', ref: ref.slice(5), byId: true}};
    if(q?.kind==='case')return {level:'discovered',owner:'Goal',target:{kind:'case',ref:q.ref.slice(5)},note:t('Exact goal reference; verify its owner before opening.')};
    if (!q) return {level: 'declaration', note: t('Not a qualified product reference; shown as declared, never opened.'), target: null};
    if (q.kind === 'review_thread') {
      const s = scene().sessions.find(s => [...s.entries.values()].some(e => e.channel === 'PRODUCT_REVIEW' && e.reference === q.ref)), e = s && [...s.entries.values()].find(e => e.reference === q.ref);
      return e ? {level: 'verified', owner: 'Review thread', target: {kind: 'review_thread', ref: q.ref.slice(14)}, route: {team: s.id, actor: '', event: e.id}, note: ''} : {level: 'unresolved', owner: 'Review thread', target: null, note: t('Selected exchange not retained')};
    }
    if (q.kind === 'history') {
      const row = await historyRow(q.ref);
      return row ? {level: 'discovered', owner: 'History', note: `${t(row.name)} · ${codeWords(row.raw.status)}`, open: ['history-open', q.ref], target: historyTarget(q.ref)} : {level: 'unresolved', owner: 'History', note: t('No saved object with this reference is discovered.'), target: null};
    }
    if (q.kind === 'task') {
      const live = Data.tasks().find((v) => v.task_id === q.ref) || (await Data.read('/api/status?' + new URLSearchParams({task_id: q.ref})));
      return {level: 'discovered', owner: 'Task Control', note: `${t('Task')} ${short(q.ref)} · ${codeWords(live.lifecycle)} · ${t('projection, not the result')}`, open: ['task', q.ref], target: {kind: 'task', ref: q.ref, taskKind: live.task_kind || null}};
    }
    for (const prefix of ['result:', 'review:']) {
      const row = await historyRow(prefix + q.ref);
      if (row) return {level: 'discovered', owner: 'History', note: `${t(row.name)} · ${codeWords(row.raw.status)}`, open: ['history-open', prefix + q.ref], target: historyTarget(prefix + q.ref)};
    }
    return {level: 'unresolved', owner: 'History', note: t('No saved result or review carries this hash; a retained PLAN can be asked from the research owner.'), target: {kind: 'plan', ref: q.ref}};
  });
  /* Open History's exact result entry when listed; otherwise offer the product's saved-object link. */
  async function resultReader(hash) {
    const entry = 'result:' + hash;
    if (!Data.history().some((r) => r.id === entry)) await Data.refreshHistory();
    return Data.history().some((r) => r.id === entry) ? {open: ['history-open', entry], link: null} : {open: null, link: entry};
  }
  /* Only exact owner readback verifies a target; a non-verifying answer keeps its discovery and names the limit. */
  const verify = (ref) => withOutcome(ref, async (previous) => {
    const target = previous?.target;
    if (!target) return {...(previous || {level: 'unresolved'}), note: t('Nothing to verify.'), target: null};
    const keep = (level, owner, note, extra = {}) => ({...previous, level, owner, note, failure: null, limit: null, ...extra});
    if (target.kind === 'review_thread') return previous;
    if(target.kind==='case'){
      // Verify the stored revision here; its references are re-verified on opening.
      const body=await Data.read('/api/goals/narrative?'+new URLSearchParams(target.byId ? {goal_id:target.ref} : {goal_hash:target.ref})), hash=body.goal_hash;
      if(!HASH.test(hash || '') || body.goal?.goal_hash!==hash || !UUID.test(body.goal?.goal_id || '') || (target.byId ? body.goal.goal_id!==target.ref : hash!==target.ref))return keep('discovered','Goal',t('The owner answered for another goal.'),{open:null});
      return keep('verified','Goal',t('Goal revision verified; its evidence is re-verified on opening; a note is not scientific approval.'),{open:['goal-open',hash],reference:'case:'+hash,question:body.goal.declaration?.objective || '',title:body.goal.declaration?.title || '',goalId:body.goal.goal_id,revision:body.goal.revision});
    }
    if (target.kind === 'result') {
      const report = await Data.read('/api/report?' + new URLSearchParams({result_hash: target.ref}));
      if (report.result_hash !== target.ref) return keep('discovered', previous.owner, `${t('Portfolio result')}: ${t('the owner answered for another result')}`);
      return keep('verified', 'Portfolio result', `${t('report')} ${short(report.report_hash)}${report.originating_task_id ? ' · ' + t('Task') + ' ' + short(report.originating_task_id) : ''}`, await resultReader(target.ref));
    }
    if (target.kind === 'task') {
      const kind = target.taskKind || Data.tasks().find((v) => v.task_id === target.ref)?.task_kind || null;
      if (kind === 'research_experiment') {
        // Authored experiments read their own exact Task back.
        const body = await Data.read('/api/experiments/readback?' + new URLSearchParams({task_id: target.ref}));
        if (body.task_id !== target.ref) return keep('discovered', previous.owner, `${t('Research experiments')}: ${t('the owner answered for another Task')}`);
        return body.status === 'EXPERIMENT_PUBLISHED' ? keep('verified', 'Research experiments', `${codeWords(body.program?.kind)} · ${codeWords(body.status)}`, {open: ['task-result', target.ref]}) : keep('discovered', previous.owner, `${t('Research experiments')}: ${t('readback')} ${codeWords(body.status)}`);
      }
      if (kind === 'portfolio_public_development_replay') {
        // Installed replay's exact index/report must name this originating Task; experiment readback does not apply.
        let report;
        try { report = await Data.installedResult(target.ref); }
        catch (e) { if (e.message !== 'portfolio_application.result_readback_mismatch') throw e; return keep('discovered', previous.owner, `${t('Portfolio result')}: ${t('the owner answered for another result')}`); }
        if (!report) return keep('discovered', previous.owner, `${t('Portfolio result')}: ${t('no result is recorded for this Task')}`);
        return keep('verified', 'Portfolio result', `${t('result')} ${short(report.result_hash)} · ${t('report')} ${short(report.report_hash)}`, await resultReader(report.result_hash));
      }
      return keep('discovered', previous.owner, previous.note, {limit: t('no owner readback through this reader; open the Task')});
    }
    if (target.kind === 'plan') {
      const preview = await Data.read('/api/experiments/preview?' + new URLSearchParams({experiment_plan_hash: target.ref}));
      if (['AVAILABLE', 'EXPIRED', 'INVALID', 'ADMITTED_AS_TASK'].includes(preview.status)) return keep('preview', 'Research experiments', `PLAN ${short(target.ref)} · ${codeWords(preview.status)}`, {open: ['research-inspect-shared', target.ref]});
      return keep('unresolved', 'Research experiments', t('No saved result, review, Task or retained PLAN carries this reference.'));
    }
    return {...previous, note: t('Nothing to verify.'), target: null};
  });
  const LEVELS = {declaration: ['metadata', 'declaration only'], discovered: ['metadata', 'discovered by {owner} · not verified'], verified: ['verified', 'verified by {owner}'], preview: ['planned', 'preview readback · {owner}'], unresolved: ['metadata', 'unresolved']};
  /* The target is shown beside the reference whenever it is a different object from what was written. */
  const sameObject = (ref, target) => (['result','case','review_thread'].includes(target.kind) ? ref === (target.kind === 'review_thread' ? 'review-thread' : target.kind) + ':' + target.ref : target.ref === ref);
  const targetLabel = (target) => target.kind === 'review_thread' ? t('Review thread') + ' ' + short(target.ref) : target.kind === 'result' ? 'result:' + short(target.ref, SHORT.hash) + '…' : target.kind === 'case' ? t('Research question')+' '+short(target.ref,SHORT.hash)+'…' : target.kind === 'plan' ? 'PLAN ' + short(target.ref, SHORT.hash) + '…' : t('Task') + ' ' + short(target.ref);
  // C4 (item 5): a qualified reference reads as the product's do: its kind, 8 hex, the copy glyph
  const referenceKind = q => q.kind === 'review_thread' ? t('Review thread') : q.kind === 'case' ? t('Goal') : q.kind === 'task' ? t('Task') : q.kind === 'history' ? codeWords(String(q.ref).split(':')[0]) : t('Reference');
  const refChip = (q, ref) => { const kind = referenceKind(q); const id = String(q.ref).split(':').at(-1); return html`<span class="run-ref"><span class="run-ref-kind">${kind}</span><span class="mono">${short(id, SHORT.id)}</span>${btnAttrs(icon('copy'), 'copy-text', ref, 'icon-btn compact', html`aria-label="${t('Copy the reference')}" data-tip="${t('Copy the reference')}"`)}</span>`; };
  function referenceLine(ref) {
    const done = S.resolved.get(ref), busy = S.busy.has(ref), q = qualify(ref);
    const label = q ? refChip(q, ref) : html`${ref} <span class="muted">· ${t('not a qualified product reference')}</span>`;
    if (busy) return html`${label} · <span class="muted">${t('asking the owner')}</span>`;
    if (!done) return html`${label} · <span class="muted">${t('declared · not resolved')}</span> ${q ? btn(t('Resolve'), 'team-resolve', ref, 'text-btn') : ''}`;
    const [tone, text] = LEVELS[done.level] || LEVELS.unresolved;
    const target = done.target && !sameObject(ref, done.target) ? html` <span class="muted">· ${t('asked as')} <span class="mono">${targetLabel(done.target)}</span></span>` : '';
    const notes = html`${done.note || ''}${done.limit ? html` <span class="muted">· ${done.limit}</span>` : ''}${done.failure ? html` <span class="muted">· ${t('owner readback failed')}: <span class="mono">${done.failure}</span></span>` : ''}`;
    return html`${label}${target} · ${stateLine(tone, {word: t(text, {owner: t(done.owner || '')})})} ${notes} ${done.route ? link(t('Open'), 'team', 'button compact', done.route) : ''}${done.open ? btn(t('Open'), done.open[0], done.open[1], 'button compact') : ''}${done.link ? LiveViews.savedObjectLink(t('Open saved object'), done.link) : ''}${done.target && done.level !== 'verified' && !done.limit ? btn(t('Verify with owner'), 'team-verify', ref, 'button compact') : ''}`;
  }
  /* ---- rendering ---- */
  const acceptedAnswer = (e) => e.kind === 'message' && e.messageKind === 'answer' && e.channel === 'PRODUCT_ACCEPTED_ANSWER';
  const acceptedArtifact = (s, e) => acceptedAnswer(e) && e.availability === 'AVAILABLE' && e.actor && e.subject.authorship_basis === 'HOOK' && e.subject.submitted_by === s.id && HASH.test(e.subject.answer_reference || '') && HASH.test(e.subject.bundle_reference || '') && ['TASK_ADMISSION', 'PRODUCT_ACCEPTED_AT'].includes(e.timeKind);
  // A default relay is a candidate; its correlated product receipt verifies it.
  const acceptedRelay = (s, e) => acceptedAnswer(e) && e.availability === 'AVAILABLE' && e.subject.authorship_basis === 'NOT_OBSERVED' && e.actor === s.id && e.role === 'research_lead' && e.subject.submitted_by === s.id && ['codex', 'claude-code'].includes(e.subject.native_host) && ['ALPHA', 'ANALYST', 'CRO', 'DATA', 'FACTOR', 'PORTFOLIO', 'RISK'].includes(e.subject.bundle_role) && HASH.test(e.subject.answer_reference || '') && HASH.test(e.subject.bundle_reference || '') && UUID.test(e.reference || '') && ['TASK_ADMISSION', 'PRODUCT_ACCEPTED_AT'].includes(e.timeKind);
  /* Sealed contributions open independently of native credit; manufacture no author, statement or lifecycle. */
  function productAcceptedAnswer(s, f) {
    const item = f?.item, p = item?.payload, sub = p?.subject;
    // ActivityItem has no stored envelope version; the selected owner checks it before content.
    return Boolean(s && item?.schema_kind === 'ProductOperationObserved' && (item.schema_version == null || item.schema_version === 1) && item.source_kind === 'PRODUCT_OPERATION' && /^local-web:[0-9a-f]{32}$/.test(item.source_id || '') && item.authority === 'OPERATIONAL_ASSERTION' && item.availability === 'AVAILABLE' && p?.operation === 'AGENT_ANSWER_SUBMIT' && p.phase === 'RETURNED' && ['ACCEPTED', 'DONE'].includes(p.status) && sub?.agent_session === s.id && ['codex', 'claude-code'].includes(sub.agent_vendor) && UUID.test(sub.task_id || '') && item.task_id === p.task_id && p.task_id === sub.task_id && HASH.test(sub.answer_reference || '') && HASH.test(sub.bundle_reference || ''));
  }
  const readableAnswer = (s, m) => m?.item ? productAcceptedAnswer(s, m) : m && (acceptedArtifact(s, m) || acceptedRelay(s, m));
  const answerObservation = (m) => m?.item?.observation_id || m?.id;
  const selectedAnswerRecord = (s, id) => s?.entries.get(id) || s?.facts.find((f) => f.item.observation_id === id && productAcceptedAnswer(s, f)) || (s && S.committee?.session === s.id && S.committee.task === committeeTask(s) && S.committee.body ? committeeEntries(S.committee.body).find(e => e.id === id) : null);
  const reviewed = (e) => e.channel === 'PRODUCT_REVIEW' && e.messageKind === 'acceptance' && e.recipient;
  const acceptedArtifacts = (s) => Data.uniqueRows([...s.entries.values()].filter((e) => !e.replayOf && (acceptedArtifact(s, e) || reviewed(e))), (e) => e.subject.answer_reference).sort((a, b) => b.ordinal - a.ordinal);
  const timeKindWords = (e) => t(e.timeKind === 'TASK_ADMISSION' ? 'Task admission time' : e.timeKind === 'PRODUCT_ACCEPTED_AT' ? 'Answer acceptance time' : e.timeKind === 'BRIDGE_RECEIVED' ? 'bridge receipt time' : 'time kind not declared');
  function identityNote(e) {
    const bits = [];
    if (e.replays?.length) bits.push(t('also admitted as {ids} · identical content under the same declared event id', {ids: e.replays.map((id) => short(id)).join(', ')}));
    if (e.conflictsWith.length) bits.push(badge('blocked', t('conflicting content · same declared event id as {ids}', {ids: e.conflictsWith.map((id) => short(id)).join(', ')})));
    return bits.length ? html`<p>${bits}</p>` : '';
  }
  // T4 (the Team review, 2026-09-24): an exchange's record read in place is a grid of what was recorded, not a sentence of dots
  const sourceRows = (e) => e.ownerFloor ? [[t('Recorded'), clock(e.at)], [t('Committee'), referenceLine(e.reference)]] : [[acceptedAnswer(e) && ['TASK_ADMISSION', 'PRODUCT_ACCEPTED_AT'].includes(e.timeKind) ? timeKindWords(e) : t('Received'), html`${clock(e.at)} <span class="muted">· ${timeKindWords(e)}</span>`], [t('Channel'), t(CHANNELS[e.channel] || 'input channel not declared')], [t('Observation'), html`<span class="mono">${short(e.id)}</span>`], [t('Declared event'), e.declaredEvent ? html`<span class="mono">${short(e.declaredEvent)}</span>` : t('not declared')]];
  /* One selected accepted record reads its full sealed contribution; keep no second history cache. */
  const answerBindings = (s, m) => {
    if (m.item) return {observation_id: m.item.observation_id, record_kind: 'PRODUCT_OPERATION', ...Object.fromEntries(['agent_role', 'agent_vendor', 'agent_session', 'goal_id', 'answer_reference', 'bundle_reference', 'task_id'].map((key) => [key, m.item.payload.subject[key] ?? null])), source_kind: m.item.source_kind, source_id: m.item.source_id, authority: m.item.authority};
    if (acceptedRelay(s, m)) return {observation_id: m.id, record_kind: 'PRODUCT_OPERATION', agent_role: m.subject.bundle_role, agent_vendor: m.subject.native_host, agent_session: s.id, goal_id: m.subject.goal_id || null, answer_reference: m.subject.answer_reference, bundle_reference: m.subject.bundle_reference, task_id: m.reference, source_kind: 'PRODUCT_OPERATION', authority: 'OPERATIONAL_ASSERTION'};
    return {observation_id: m.id, native_session_id: s.id, native_agent_id: m.actor, native_host: m.subject.native_host, role: m.role, submitted_by: m.subject.submitted_by, answer_reference: m.subject.answer_reference, bundle_reference: m.subject.bundle_reference, task_id: m.reference};
  };
  const answerEpoch = () => LiveActivity.state().epoch || X.epoch;
  const sameAnswer = (detail, s, m) => s && m && detail.session === s.id && readableAnswer(s, m) && Object.entries(answerBindings(s, m)).every(([key, value]) => detail.bindings[key] === value);
  function selectedAnswer(detail) {
    const sel = selection();
    if (app.page !== 'team' || detail !== S.answerDetail || detail.epoch !== answerEpoch() || sel.session !== detail.session || sel.event !== detail.bindings.observation_id) return false;
    const s = scene().sessions.find((s) => s.id === sel.session);
    return sameAnswer(detail, s, selectedAnswerRecord(s, sel.event));
  }
  function acceptedDetail(s, m) {
    if (!readableAnswer(s, m)) return '';
    if (!S.answerDetail || S.answerDetail.epoch !== answerEpoch() || !sameAnswer(S.answerDetail, s, m)) {
      const detail = {session: s.id, bindings: answerBindings(s, m), epoch: answerEpoch(), pending: true, answer: null, error: ''};
      S.answerDetail = detail;
      Data.read('/api/activity/external?' + new URLSearchParams({observation_id: answerObservation(m)})).then((body) => {
        const answer = body.accepted_answer;
        if (body.observation_id !== answerObservation(m) || body.epoch !== detail.epoch || !['AVAILABLE', 'UNAVAILABLE'].includes(answer?.status) || (answer.status === 'AVAILABLE' && (!Object.entries(detail.bindings).every(([key, value]) => answer[key] === value) || !answer.contribution || typeof answer.contribution !== 'object' || Array.isArray(answer.contribution)))) throw new Error(t('The owner answered for another accepted answer.'));
        detail.answer = answer;
      }).catch((e) => { detail.error = e; }).finally(() => {
        detail.pending = false;
        if (selectedAnswer(detail)) paint();
      });
    }
    const detail = S.answerDetail, title = t('Accepted answer');
    if (detail.pending) return noteLine(title, t('Loading'));
    if (detail.error) return notRead(title, detail.error, t('The full accepted answer is unavailable.'));
    if (detail.answer.status !== 'AVAILABLE') return notRead(title, detail.answer.reason || t('Unavailable'), t('The full accepted answer is unavailable.'));
    const contribution = detail.answer.contribution;
    const attribution = m.item || acceptedRelay(s, m) ? (detail.answer.recorded_agent?.basis === 'HOOK' ? codeRef(t('Stored native attribution'), detail.answer.recorded_agent) : noteLine(t('Native author not observed'))) : '';
    return html`${attribution}${typeof contribution.text === 'string' ? html`${noteLine(title)}<p class="owner-text">${contribution.text}</p>${Array.isArray(contribution.references) && contribution.references.length ? codeRef(t('References'), contribution.references) : ''}` : codeRef(title, contribution)}`;
  }
  function messageRecord(s, m) {
    const words = m.truncated ? html`${stateLine('partial', {word: t('the first 500 characters')})} <span class="muted">· ${t('the full original is not held by this feed')}${m.reference ? '' : ' · ' + t('no reference declared for it')}</span>` : t('complete as declared');
    return html`${identityNote(m)}${kv([[t('Words'), words], [t('Size'), m.bytes ? t('{n} bytes declared', {n: m.bytes}) : t('size not declared')], ...(m.reference ? [[t('Declared reference'), referenceLine(m.reference)]] : []), ...sourceRows(m), ...(acceptedAnswer(m) ? [[t('Submitted by'), html`<span tabindex="0" data-tip="${m.subject.submitted_by || ''}">${recipientName(s, m.subject.submitted_by)}</span>`], ...(HASH.test(m.subject.answer_reference || '') ? [[t('Answer reference'), hashCell(m.subject.answer_reference, SHORT.hash)]] : []), ...(HASH.test(m.subject.bundle_reference || '') ? [[t('Bundle reference'), hashCell(m.subject.bundle_reference, SHORT.hash)]] : [])] : []), [t('Message ID'), m.messageId ? html`<span class="mono">${short(m.messageId, SHORT.id)}</span>` : t('not declared')]], 'kv-columns')}${acceptedDetail(s, m)}`;
  }
  function hookRecord(s, h) {
    return html`<p class="muted">${t('the child may continue; not proof of exit, Task completion or publication')}</p>${identityNote(h)}${kv([[t('Terminal state'), html`<span class="mono">${h.terminal || t('not declared')}</span>`], [t('Stop hook active'), html`<span class="mono">${h.stopActive || t('not declared')}</span>`], ...sourceRows(h), [t('Turn'), h.turn ? html`<span class="mono">${short(h.turn, SHORT.id)}</span>` : t('not declared')]], 'kv-columns')}`;
  }
  /* Only an explicitly compatible artifact verification reads as owner-verified: the schema the
   * artifact owner writes, at its ARTIFACT_ASSERTION authority, retained, naming the artifact. */
  const artifactVerified = (item) => item.schema_kind === 'ArtifactVerificationObserved' && item.authority === 'ARTIFACT_ASSERTION' && item.availability === 'AVAILABLE' && HASH.test(String(item.payload?.artifact_hash || ''));
  /* One product observation as recorded: its own phase, status and references, at its own time.
   * The product operation, the Task Control transition and the compatible artifact verification
   * are read; every other observation that names a declared reference (a worker's progress, a
   * stage verification, an unretained payload) stays an observation, labelled by its schema. */
  function factFacts(fact) {
    const {item} = fact, p = item.payload || {}, retained = item.availability === 'AVAILABLE' && Boolean(item.payload);
    let title, tone = 'metadata', state, ic = 'activity';
    if (!retained) {
      title = `${codeWords(item.schema_kind)} · ${t('observation')}`; state = t('payload not retained');
    } else if (item.schema_kind === 'ProductOperationObserved') {
      const refused = p.status === 'REFUSED' || (p.phase === 'FAILED' && p.failure_code), correct = p.status === 'CORRECT'; // an answer returned for correction is not a refusal (contract 10.4)
      title = `${codeWords(p.operation)} · ${t(refused ? 'product refusal' : correct ? 'returned for correction' : p.phase === 'REQUESTED' ? 'requested' : p.status === 'PLANNED' ? 'product preview' : ['ADMITTED', 'REUSED_IN_FLIGHT'].includes(p.status) ? 'product admission' : p.status === 'REUSED_EXACT' ? 'exact reuse' : 'product return')}`;
      state = correct ? 'returned_for_correction' : p.status || (p.phase === 'FAILED' ? 'FAILED' : p.phase); tone = refused ? 'blocked' : correct ? 'returned_for_correction' : p.status === 'PLANNED' ? 'planned' : p.status === 'ADMITTED' ? 'queued' : p.status === 'REUSED_EXACT' ? 'reused_exact' : 'metadata'; ic = refused ? 'ban' : correct ? 'edit' : 'cube';
    } else if (item.schema_kind === 'TaskControlTransition') {
      title = `${codeWords(p.task_class || 'Task')} · ${t('Task Control fact')}`; state = p.task_lifecycle || ''; tone = String(state).toLowerCase(); ic = 'task';
    } else if (artifactVerified(item)) {
      title = `${codeWords(p.artifact_kind || 'artifact')} · ${t('owner-verified artifact')}`; state = p.availability; tone = 'verified'; ic = 'checkcircle';
    } else {
      title = `${codeWords(item.schema_kind)} · ${t('observation')}`; state = t('recorded');
    }
    return {title, tone, state, ic, p, retained};
  }
  /* The evidence list folds (round 95; GitHub's `23 similar events`, Linear's `3 times`): a
   * request and its return are one unit (`requested 17:23 → prerequisites missing 17:23`); units
   * that repeat one after another -- the same operation, the same return, the same caller and
   * references -- are one row with their count and an opener; a pressed opener shows the units.
   * Recorded order is kept: the fold sits where its first unit was. */
  const isProduct = (f) => f.item.availability === 'AVAILABLE' && Boolean(f.item.payload) && f.item.schema_kind === 'ProductOperationObserved';
  function factUnits(facts) {
    const units = [], pending = new Map();
    for (const f of facts) {
      const p = f.item.payload || {};
      if (isProduct(f) && p.phase === 'REQUESTED') { const u = {request: f, response: null, facts: [f]}; units.push(u); pending.set(p.operation, u); continue; }
      const open = isProduct(f) ? pending.get(p.operation) : null;
      if (open) { open.response = f; open.facts.push(f); pending.delete(p.operation); continue; }
      units.push({request: null, response: f, facts: [f]});
    }
    return units;
  }
  const unitSignature = (u) => { const f = u.response || u.request, {title, state} = factFacts(f), p = f.item.payload || {}; return [u.request ? codeWords(u.request.item.payload?.operation) : '', title, state, p.failure_code || '', p.caller || '', f.named.join(',')].join('|'); };
  function foldUnits(units) {
    const out = [];
    for (const u of units) {
      const sig = unitSignature(u), last = out.at(-1);
      if (last && last.signature === sig) last.units.push(u);
      else out.push({signature: sig, units: [u], key: (u.request || u.response).item.observation_id});
    }
    return out;
  }
  // the second Team review: a withheld detail says why, in the owner's terms (the feed stores a typed failure code)
  const withheldWhy = (p) => p.failure_code === 'activity.failure_detail_withheld' ? html` <span class="muted">· ${t('the feed keeps a typed failure code, never a failure\'s own text')}</span>` : '';
  const callerOf = (x) => { const q = x.item.payload || {}; return q.caller ? actorWords(q.caller, q.producer_id) : x.item.source_kind === 'TASK_CONTROL' ? t('Task Control') : x.item.source_kind === 'PRODUCT_ARTIFACT' ? t('artifact owner') : codeWords(x.item.source_kind || ''); };
  /* What a record says apart from its group's first: its state, its outcome, what it names, who made it. */
  const differences = (f, h) => { const a = factFacts(f), b = factFacts(h); return {state: a.state !== b.state, outcome: (a.p.failure_code || '') !== (b.p.failure_code || ''), named: f.named.join(' ') !== h.named.join(' '), caller: String(callerOf(f)) !== String(callerOf(h)) || f.item.authority !== h.item.authority}; };
  // U53: a group whose records differ from its first only in time opens to nothing its one line does not say
  const sameButTime = (g) => { const h = g.units[0].response || g.units[0].request; return g.units.every((u) => !Object.values(differences(u.response || u.request, h)).some(Boolean)); };
  function unitRow(u, fold = null) {
    const f = u.response || u.request, {item, group, named} = f, {title, tone, state, p} = factFacts(f);
    const caller = callerOf(f);
    const links = [...named.map((r) => html`<span class="mono">${short(r, r.includes(':') ? 20 : 8)}</span>`), ...(f.followed || []).map((x) => html`<span class="mono">${short(x.ref)}</span> <span class="muted">(${t('Task named by {operation} {status} for {ref}', {operation: x.operation, status: x.status || '', ref: short(x.via)})})</span>`)];
    const outcome = html`${p.failure_code ? coded(p.failure_code) : ''}${p.failure_type && p.failure_code === 'activity.failure_detail_withheld' ? html` <span class="mono">${p.failure_type}</span>` : ''}${withheldWhy(p)}`;
    const course = u.request && u.response ? html`${t('requested')} ${clock(u.request.item.occurred_at)} → ${clock(item.occurred_at)} · ` : ''; // the return's kind is the row's title, its meaning the state word
    const why = html`${course}${outcome}${group.task_id ? html` ${t('Task')} ${short(group.task_id)} ·` : ''}${links.length ? html` ${t('names')} ${links}` : ''}`;
    const fresh = u.facts.some((x) => S.factNew.has(x.item.observation_id)), flash = u.facts.some((x) => S.factFlash.has(x.item.observation_id));
    const isNew = fresh ? html` <span class="team-unread">${t('New')}</span>` : '';
    if (fold) {
      // T5 (the Team review, 2026-09-24): under its group an occurrence says its own times and only what differs from its first
      const h = fold.units[0].response || fold.units[0].request, d = differences(f, h);
      const differs = [d.state ? codeWords(state) : '', d.outcome ? outcome : '', d.named ? html`${t('names')} ${links}` : '', d.caller ? html`${caller} · ${codeWords(item.authority)}` : ''].filter(Boolean);
      const asked = u.request && u.response ? clock(u.request.item.occurred_at) : '', at = asked && asked !== clock(item.occurred_at) ? html`${asked} → ${clock(item.occurred_at)}` : clock(item.occurred_at);
      // the dot keeps the group's column; no word -- the group says it once
      return objectRow({lead: statusDot(tone, codeWords(state)), name: html`${at}${isNew}`, why: differs.map((d, i) => html`${i ? ' · ' : ''}${d}`), cls: 'evidence-row' + (flash ? ' just-verified' : '')}, {key: item.observation_id, attrs: html`data-kind="observation" data-observation="${short(item.observation_id)}" data-depth="1"`});
    }
    return evidenceRow({kind: 'Observation', type: 'observation', subject: html`${title}${isNew}`, state: tone, word: codeWords(state), why, id: item.observation_id}, null, {cls: flash ? 'just-verified' : '', columns: ['caller'], props: [html`${caller} · ${codeWords(item.authority)}`], time: clock(item.occurred_at), attrs: html`data-observation="${short(item.observation_id)}"`});
  }
  function toggleFold(key) { if (S.unfolded.has(key)) S.unfolded.delete(key); else S.unfolded.add(key); paint(); }
  /* What arrived new among the product facts, in two honest counts: Task completions (a
   * SUCCEEDED Task Control transition) and owner-verified results (the compatible artifact
   * verification). A completion is not a verification and is never counted as one. */
  function newFacts(s) {
    const fresh = s.facts.filter((f) => S.factNew.has(f.item.observation_id));
    const verified = fresh.filter((f) => S.factVerified.has(f.item.observation_id)).length, completed = fresh.length - verified;
    if (!fresh.length) return '';
    return html` · <span class="team-unread">${t('New')}: ${[completed ? countText(completed, '{n} Task completion', '{n} Task completions') : '', verified ? countText(verified, '{n} owner-verified result', '{n} owner-verified results') : ''].filter(Boolean).join(' · ')}</span>`;
  }
  /* The current projection of each Task the facts touch, kept apart from the recorded history. */
  function currentStates(s) {
    const groups = new Map(s.facts.filter((f) => f.group.task_id).map((f) => [f.group.key, f.group]));
    return [...groups.values()].map((g) => { const f = LiveActivity.facts(g); return html`<div class="record-bundle">${icon('task')}<div><strong>${t('Task')} ${f.live?.status==='REFUSED' ? hashCell(g.task_id,SHORT.id) : html`<span class="mono">${short(g.task_id)}</span>`} · ${t('current state')}</strong><small>${f.live?.status==='REFUSED' ? html`${stateLine('refused')} ${explainCode(f.live.failure_code) || t(f.live.detail)}` : f.live ? html`${stateLine(f.state)} ${f.live.verified_stage_count} / ${f.live.total_stage_count} ${t('stages')} · ${t('Task Control projection, read now')}` : html`${stateLine('metadata', {word: t('no current projection')})} ${t('last recorded')} ${codeWords(f.state)}`} ${f.verified ? stateLine('verified', {word: t('Result verified by its owner')}) : ''}</small></div><div class="flow">${LiveActivity.nextRead(f, g)}</div></div>`; });
  }
  /* Exact replyTo/reference links replies; shared topics do not. A PM response requires the recorded parent research_lead and resolves nothing. */
  function linkExchanges(s, entries) {
    const declared = new Map();
    for (const e of entries) if (e.kind === 'message' && e.messageId) declared.set(e.messageId, [...(declared.get(e.messageId) || []), e]);
    const pm = (e) => { const pt = s.participants.get(e.actor); return Boolean(pt) && isLead(pt, s); };
    for (const e of entries) {
      const named = e.kind === 'message' ? e.replyTo || e.reference : null, same = named ? declared.get(named) || [] : [];
      const others = same.filter((o) => o !== e);
      e.namesExchange = others.length === 1 && same.length === 1 ? others[0] : null;
      e.ambiguousReference = same.length > 1 ? {id: named, count: same.length} : null;
      e.sameReference = e.qualified ? entries.filter((o) => o !== e && o.qualified && o.qualified.ref === e.qualified.ref) : [];
      e.responses = []; e.pmResponses = [];
    }
    for (const e of entries) if (e.messageKind === 'objection') {
      e.responses = entries.filter((r) => r.namesExchange === e);
      e.pmResponses = e.responses.filter((r) => r.messageKind === 'pm_response' && pm(r));
    }
  }
  const kindWords = (e) => acceptedAnswer(e) ? t('Accepted answer') : e.kind === 'message' ? t((e.messageKind ? MESSAGE_KINDS[e.messageKind] : ['Message kind not declared'])[0]) : t('Host input');
  /* The scoped or latest delivered Goal names its question through the exact owner read. */
  const titleOf = (q) => (q.kind === 'none' ? t('No research question is declared') : q.text || q.title || t('Untitled session'));
  function referencesOf(entries) {
    const groups = new Map();
    for (const e of entries.filter(e => e.qualified)) {
      const key = e.qualified.kind === 'case' ? S.resolved.get(e.reference)?.goalId || e.qualified.ref : e.qualified.ref;
      if (!groups.has(key)) groups.set(key, []);
      const group = groups.get(key);
      if (!group.some(was => was.qualified.ref === e.qualified.ref)) group.push(e);
    }
    return [...groups.values()].map(group => group.sort((a, b) => Number(S.resolved.get(b.reference)?.revision || 0) - Number(S.resolved.get(a.reference)?.revision || 0)));
  }
  const revisionRefs = (group, read) => html`${read(group[0])}${group.length > 1 ? factsRef(countText(group.length - 1, '{n} earlier revision', '{n} earlier revisions'), html`${group.slice(1).map(e => html`<p>${read(e)}</p>`)}`) : ''}`;
  function questionOf(s, all) {
    const lead = [...s.participants.values()].find((p) => isLead(p, s)) || null;
    const distinct = (list) => [...new Map(list.map((e) => [e.qualified.ref, e])).values()];
    const cases = all.filter((e) => e.kind === 'message' && e.qualified?.kind === 'case');
    const declared = lead ? distinct(cases.filter((e) => e.actor === lead.id)) : [];
    const groups = referencesOf(declared);
    const related = distinct(cases.filter((e) => !lead || e.actor !== lead.id)).filter((e) => !declared.some((d) => d.qualified.ref === e.qualified.ref)).map((e) => ({ref: e.reference, actor: actorName(s, e.actor), by: e.actor}));
    const chosen = S.question.get(s.id) || '';
    const scope = s.goalScope || [...all].reverse().find(e => acceptedAnswer(e) || reviewed(e) || e.messageKind === 'verdict')?.subject.goal_id;
    const candidate = declared.find(e => e.reference === chosen) || groups.flat().find(e => scope && (e.subject.goal_id === scope || S.resolved.get(e.reference)?.goalId === scope)) || (!scope && groups.length === 1 ? groups[0][0] : null);
    const assignments = lead ? all.filter((e) => e.messageKind === 'assignment' && e.actor === lead.id) : [];
    const base = {related, declared: declared.map((e) => e.reference), groups, lead: Boolean(lead), scope};
    if (candidate) {
      const outcome = S.resolved.get(candidate.reference) || null;
      if (outcome?.level === 'verified') return {...base, kind: 'case-verified', text: outcome.question, title: outcome.title || '', ref: candidate.reference, goalId: outcome.goalId, choice: groups.length > 1};
      return {...base, kind: 'case-declared', text: assignments.length ? assignments[0].text : t('Goal {ref}', {ref: short(candidate.reference.slice(5), SHORT.hash)}), ref: candidate.reference, resolved: Boolean(outcome), choice: groups.length > 1};
    }
    const heldGoal = scope && [...S.resolved].find(([ref, outcome]) => ref.startsWith('case:') && outcome.level === 'verified' && outcome.goalId === scope);
    if (heldGoal) {
      const [ref, outcome] = heldGoal;
      return {...base, kind: 'case-verified', text: outcome.question, title: outcome.title || '', ref: outcome.reference || ref, goalId: outcome.goalId, viaDelivery: !s.goalScope, choice: false};
    }
    if (groups.length > 1) return {...base, kind: 'choice', text: declared.every(e => S.resolved.get(e.reference)?.goalId) ? t('{n} goals declared by Main PM', {n: groups.length}) : t('Choose a Goal reference'), count: groups.length};
    if (assignments.length) return {...base, kind: 'assignment', text: assignments[0].text, count: assignments.length};
    return {...base, kind: 'none', text: ''};
  }
  /* The source line under a question, the same words on the Team page and the Overview. */
  function questionSource(q) {
    if (q.kind === 'case-verified') return html`${stateLine('verified', {word: t('Goal revision verified by its owner')})} <span class="muted">${t(q.viaDelivery ? 'Question supplied for this delivery' : 'association declared by Main PM · not verified')}</span>`;
    if (q.kind === 'case-declared') return html`<span class="muted">${t(q.resolved ? 'declared by Main PM · Goal discovered, not yet verified' : 'declared by Main PM · Goal not yet resolved')} · <span class="mono">${short(String(q.ref).slice(5), SHORT.hash)}</span></span>`;
    if (q.kind === 'choice') return html`<span class="muted">${t('explicit choice required · none chosen by order or recency')}</span>`;
    if (q.kind === 'assignment') return html`<span class="muted">${t(q.count > 1 ? 'first of {n} assignments declared by Main PM · not a bound goal' : 'declared by Main PM · not a bound goal', {n: q.count})}</span>`;
    return '';
  }
  /* The owner question and the declared references, with earlier revisions disclosed. */
  function questionFacts(s, all) {
    const q = questionOf(s, all);
    const choice = q.kind === 'choice' || q.choice ? html`<ul class="team-question-choice">${q.groups.map(group => html`<li>${revisionRefs(group, e => html`${e.reference === q.ref ? stateLine('ready', {word: t('chosen')}) : btn(t('Show this case'), 'team-question', e.reference, 'text-btn')} ${referenceLine(e.reference)}`)}</li>`)}</ul>` : '';
    const related = q.related.length ? html`<p class="caption team-question-related">${t('Related case references declared by members, not the session\'s question')}: ${q.related.map((r) => html`<span>${r.actor} · ${referenceLine(r.ref)}</span>`)}</p>` : '';
    return html`<div class="team-question-card panel-body" data-question="${q.kind}">${q.kind === 'case-verified' && q.text ? html`<p class="team-question-text">${q.text}</p>` : ''}${q.kind !== 'none' ? html`<p class="team-question-source">${questionSource(q)}</p>${q.ref && !q.choice ? html`<p class="team-question-ref flow">${referenceLine(q.ref)}</p>` : ''}${choice}` : html`<p class="team-question-source">${t('A Main PM declaration of a research case or an assignment would name it; nothing is inferred from the messages.')}</p>`}${related}</div>`;
  }
  /* One glyph per role family, so a member is recognised before its words are read. */
  const ROLE_ICONS = {research_lead: 'user', alphalattice_data: 'data', alphalattice_factor: 'grid', alphalattice_alpha: 'lab', alphalattice_risk: 'partial', alphalattice_portfolio: 'portfolio', alphalattice_evidence_analyst: 'file', alternative_analyst: 'file', alphalattice_cro: 'review', independent_cro: 'review'};
  const roleIcon = (s, id) => { const pt = s.participants.get(id); const role = pt ? [...pt.roles][0] : ''; return pt && isLead(pt, s) ? 'user' : ROLE_ICONS[role] || 'fork'; };
  /* A role's identity colour (round 20e): the Main PM in the brand blue, Alpha in cyan, Portfolio in
   * emerald, the CRO in amber, Risk in violet, analysts and data neutral, an undeclared role grey —
   * on the node, the roster's icon and the division marks only, never on a row. */
  const ROLE_TONES = {research_lead: 'lead', alphalattice_alpha: 'alpha', alphalattice_portfolio: 'portfolio', alphalattice_cro: 'cro', independent_cro: 'cro', alphalattice_risk: 'risk', alphalattice_factor: 'factor', alphalattice_data: 'data', alphalattice_evidence_analyst: 'neutral'}; // one tone per domain (round 89, law 76)
  const roleTone = (s, id) => { const pt = s.participants.get(id); if (!pt) return 'none'; if (isLead(pt, s)) return 'lead'; const role = [...pt.roles][0]; return role ? ROLE_TONES[role] || 'neutral' : 'none'; };
  /* N5, law 125: one topic chip per declared reference; a shared topic is never a reply. */
  // Q2: a goal's revision, a result and a review by their hash's 12, an experiment and a Task by their id's 8
  const topicWords = (q) => q.kind === 'review_thread' ? html`${referenceKind(q)} <span class="mono">${short(q.ref.slice(14))}</span>` : q.kind === 'case' ? html`${t('Goal')} <span class="mono">${short(q.ref.slice(5), SHORT.hash)}</span>` : q.kind === 'history' ? (([kind, id]) => html`<span class="mono">${kind}:${short(id, kind === 'experiment' ? SHORT.id : SHORT.hash)}</span>`)(q.ref.split(':')) : q.kind === 'task' ? html`${t('Task')} <span class="mono">${short(q.ref)}</span>` : html`<span class="mono">${short(q.ref, SHORT.hash)}</span>`;
  /* N5, law 125; C1, law 139; law 17: show role, addressee, kind, topic, time and clamped words; pressing opens the full record. Exact replies stay one level deep; PM responses are neutral and never resolve objections. */
  function exchange(s, e, selected, replies = [], chosen = null, reply = false) {
    const accepted = acceptedAnswer(e), kind = accepted ? ['Accepted answer', 'file'] : e.kind === 'message' ? MESSAGE_KINDS[e.messageKind] || ['Message kind not declared', 'info'] : ['Host input', 'activity'];
    const pmOf = (r) => { const pt = s.participants.get(r.actor); return ['pm_response', 'ruling'].includes(r.messageKind) && (r.ownerFloor ? r.role === PM_ROLE && r.actor === s.id : Boolean(pt) && isLead(pt, s)); };
    const person = e.channel === 'PRODUCT_COMMITTEE' && e.messageKind === 'person_answer', author = person ? t('Person') : e.ownerFloor || e.channel === 'PRODUCT_REVIEW' ? roleName(e.role) : actorName(s, e.actor);
    const mention = person ? e.subject.relay_word ? html`<span class="team-kind">· ${t(e.subject.relay_word)}</span>` : '' : e.recipient ? html`<span class="team-mention" tabindex="0" data-tip="${e.recipient}">→ @${e.recipientRole ? roleName(e.recipientRole) : recipientName(s, e.recipient, e)}</span>` : '';
    const declaredPm = e.messageKind === 'pm_response' && !pmOf(e) ? html` <span class="muted">(${t('declared pm_response, not the Main PM')})</span>` : '';
    const kindMark = accepted ? html`<span class="team-kind" tabindex="0" data-tip="${t(CHANNELS.PRODUCT_ACCEPTED_ANSWER)}">· ${t(kind[0])}</span>` : html`<span class="team-kind">· ${t(kind[0])}${declaredPm}</span>`;
    const submitter = accepted ? html`<span class="team-kind">· ${t('Submitted by')} <span tabindex="0" data-tip="${e.subject.submitted_by || ''}">${recipientName(s, e.subject.submitted_by)}</span></span>` : '';
    const topic = e.targetText ? html`<span class="team-topic">${e.targetText}</span>` : !e.ownerFloor && e.qualified ? html`<span class="team-topic" tabindex="0" data-tip="${e.sameReference?.length ? countText(e.sameReference.length, 'Names the same reference as {n} other exchange · related, not a reply', 'Names the same reference as {n} other exchanges · related, not a reply') : t('Declared reference')}">${topicWords(e.qualified)}</span>` : '';
    let status = '';
    if (e.outcome) status = html`<span class="team-kind" data-outcome="${e.outcome}">${codeWords(e.outcome)}</span>`;
    if (e.dissentStands) status = html`${status}<span class="team-objection-status" data-dissent="stands">${stateLine('review_pending', {word: t('Dissent stands'), next: ''})}</span>`;
    if (e.messageKind === 'objection') {
      const answered = e.pmResponses.length > 0, others = e.responses.filter((r) => !e.pmResponses.includes(r));
      const tip = answered ? t('Recorded Main PM response · not a resolution; the objection stays as recorded') : `${t('No recorded Main PM response · unresolved choice')}${others.length ? ' · ' + countText(others.length, '{n} other message names this objection', '{n} other messages name this objection') : ''}`;
      status = html`<span class="team-objection-status" data-answered="${answered}" data-tip="${tip}" tabindex="0">${answered ? btn(stateLine('verified', {word: t('Main PM responded'), next: ''}), 'team-event', e.pmResponses.at(-1).id, 'text-btn') : stateLine('review_pending', {word: t('Awaiting the Main PM'), next: ''})}</span>`;
    }
    // the second Team review: an objection no Main PM response answers says, read-only, what the page waits for and where
    const waiting = e.messageKind === 'objection' && !e.pmResponses.length && !reply ? html`<p class="team-awaiting-note">${t('No Main PM response is recorded. The Main PM answers in its own host session; the feed shows the answer here once it is recorded.')}</p>` : '';
    const ambiguous = e.ambiguousReference ? html`<p class="team-relation team-ambiguous">${t('Names message ID {id}, declared by {n} exchanges · ambiguous, names none', {id: short(e.ambiguousReference.id, SHORT.id), n: e.ambiguousReference.count})}</p>` : '';
    // C2 (law 140): a run of replies keeps its first (the cause) and its last; the middle is one line that opens in place
    const replyRow = (r) => exchange(s, r, chosen?.id === r.id, [], chosen, true), middle = replies.length > 2 ? replies.length - 2 : 0;
    const heldReply = replies.slice(1, -1).find((r) => r.id === chosen?.id), open = S.replies.has(e.id) || Boolean(heldReply) || replies.some(r => r.dissentStands);
    const thread = replies.length ? html`<ol class="team-replies">${middle && !open ? html`${replyRow(replies[0])}<li class="team-replies-more" data-holds="${replies.slice(1, -1).map((r) => r.id).join(' ')}">${btnAttrs(html`··· ${countText(middle, 'Show {n} more', 'Show {n} more')} ···`, 'team-replies', e.id, 'text-btn', html`aria-expanded="false" data-fold-count="${middle}"`)}</li>${replyRow(replies.at(-1))}` : html`${replies.map(replyRow)}${middle ? html`<li class="team-replies-more">${btnAttrs(t('Show less'), heldReply && !S.replies.has(e.id) ? 'team-event' : 'team-replies', heldReply && !S.replies.has(e.id) ? heldReply.id : e.id, 'text-btn', html`aria-expanded="true" data-fold-count="${middle}"`)}</li>` : ''}`}</ol>` : '';
    // the feed keeps a message's first 500 characters (`summary_truncated`): an opened exchange says where the retained words end
    const words = e.kind === 'message' ? html`${e.text}${e.truncated ? html`<span class="muted"> … ${t('the feed keeps its first 500 characters')}</span>` : ''}` : t('Hook observed; child completion is not established.');
    return html`<li class="team-exchange${reply ? ' team-reply' : ''} ${e.messageKind === 'objection' ? 'is-objection' : ''} ${pmOf(e) ? 'is-pm-response' : ''} ${selected ? 'is-selected' : ''} ${S.flash.has(e.id) ? 'just-arrived' : ''}" id="team-event-${e.id}" data-kind="${e.messageKind || e.kind}" data-role="${roleTone(s, e.actor)}" data-read="${e.id}" tabindex="0" aria-label="${author} · ${t(kind[0])}">
      <span class="team-exchange-icon">${icon(roleIcon(s, e.actor))}</span><div class="team-exchange-body"><div class="team-exchange-line"><strong>${author}</strong>${mention}${kindMark}${submitter}${topic}${status}${e.channel !== 'PRODUCT_COMMITTEE' && S.incoming.has(e.id) ? html`<span class="team-unread">${t('Unread')}</span>` : ''}${e.conflictsWith.length ? badge('blocked', t('Conflicting event')) : ''}</div><span class="team-exchange-end"><time data-tip="${whenText(e.at)}${accepted ? ' · ' + timeKindWords(e) : ''}">${clock(e.at)}</time></span>
      <p class="team-words owner-text" data-clamp="${e.dissentStands || S.words.has(e.id) ? 'open' : ''}">${words}</p>${e.dissentStands ? '' : btnAttrs(t(S.words.has(e.id) ? 'Show less' : 'Show more'), 'team-words', e.id, 'text-btn team-words-more', html`aria-expanded="${S.words.has(e.id)}" data-clamp-way ${S.words.has(e.id) ? '' : 'hidden'}`)}${waiting}${selected ? inlineRecord(s, e) : ''}${ambiguous}${thread}
      </div></li>`;
  }
  /* A host input as one line of the thread (a hook is an observation, not a statement). */
  function hookEvent(s, h, selected) {
    return html`<li class="team-exchange team-event ${selected ? 'is-selected' : ''} ${S.flash.has(h.id) ? 'just-arrived' : ''}" id="team-event-${h.id}" data-kind="hook" data-role="${roleTone(s, h.actor)}" data-read="${h.id}" tabindex="0"><span class="team-event-mark">${icon('activity')}</span><span class="team-event-line"><strong>${actorName(s, h.actor)}</strong> ${t('{event} hook observed', {event: h.hookEvent || t('hook event not declared')})}${S.incoming.has(h.id) ? html` <span class="team-unread">${t('Unread')}</span>` : ''}</span><span class="team-exchange-end"><time data-tip="${whenText(h.at)}">${clock(h.at)}</time></span>${selected ? inlineRecord(s, h) : ''}</li>`;
  }
  /* The product's observations as one line of the thread, folded as the Evidence page folds them
   * (a request and its return one unit; units in a row that repeat one line with their count). */
  function productEvent(s, fold) {
    const last = fold.units.find((u) => (u.response || u.request).item.observation_id === selection().event) || fold.units.at(-1), f = last.response || last.request, {title, tone, state} = factFacts(f), n = fold.units.length;
    const fresh = fold.units.some((u) => u.facts.some((x) => S.factNew.has(x.item.observation_id)));
    const accepted = productAcceptedAnswer(s, f), selected = accepted && selection().event === f.item.observation_id;
    return html`<li class="team-event team-product-event" data-fact="${short(f.item.observation_id)}" id="team-event-${f.item.observation_id}"><span class="team-event-mark">${statusDot(tone)}</span><span class="team-event-line">${title} · ${codeWords(state)}${n > 1 ? html` · ${countText(n, '{n} time', '{n} times')}` : ''}${fresh ? html` <span class="team-unread">${t('New')}</span>` : ''}${accepted ? html` ${btn(t('Accepted answer'), 'team-event', f.item.observation_id, 'text-btn')}` : ''}${selected ? html`<div class="team-record-inline">${acceptedDetail(s, f)}</div>` : ''}</span><span class="team-exchange-end">${spanTime(fold.units)}</span></li>`;
  }
  // C4 (item 1): a line's real span -- its first unit to its last, the time alone where they are one
  const spanTime = (units) => { const from = (units[0].request || units[0].response).item.occurred_at, to = (units.at(-1).response || units.at(-1).request).item.occurred_at, a = clock(from), b = clock(to); return html`<time data-tip="${whenText(from)}${from !== to ? ' – ' + whenText(to) : ''}">${a === b ? a : html`${a} – ${b}`}</time>`; };
  /* Consecutive product lines between two exchanges are one counted line that opens in place, as the
   * hooks are (C4 item 1, law 140): what they were, how many, first to last. */
  function productRun(s, folds) {
    const key = 'product:' + folds[0].key, units = folds.flatMap((f) => f.units), held = units.some((u) => (u.response || u.request).item.observation_id === selection().event), open = S.unfolded.has(key) || held;
    const titles = [...new Set(folds.map((f) => factFacts(f.units.at(-1).response || f.units.at(-1).request).title))];
    const line = html`<li class="team-event team-product-event team-product-run"><span class="team-event-mark">${icon('activity')}</span><span class="team-event-line">${btnAttrs(html`<span>${countText(units.length, '{n} product operation', '{n} product operations')} · ${titles.join(', ')}</span>${icon('chevron')}`, 'team-fold', key, 'text-btn team-hook-toggle', html`aria-expanded="${open}" data-fold-count="${units.length}"`)}</span><span class="team-exchange-end">${spanTime(units)}</span></li>`;
    return html`${line}${open ? folds.map((f) => productEvent(s, f)) : ''}`;
  }
  /* N5, law 125: store-ordered exchanges and hooks; exact replies nest one level under visible targets. Product lines appear only without a member filter; order proves nothing. */
  /* Consecutive hook events are one line with their count (C2, law 140): the observations between two
   * exchanges read as one pause, opened in place; a single hook stays its own line. A run that holds
   * the exchange being read opens by itself, as the replies do (its reading is in place, C4 item 7);
   * its toggle then closes the reading with the run. */
  function hookRun(s, run, chosen) {
    const key = 'hooks:' + run[0].id, held = run.find((h) => h.id === chosen?.id), open = S.unfolded.has(key) || Boolean(held), actors = [...new Set(run.map((h) => actorName(s, h.actor)))];
    const [action, value] = held && !S.unfolded.has(key) ? ['team-event', held.id] : ['team-fold', key];
    const line = html`<li class="team-exchange team-event team-hook-run" data-kind="hooks"${open ? '' : html` data-holds="${run.map((h) => h.id).join(' ')}"`}><span class="team-event-mark">${icon('activity')}</span><span class="team-event-line">${btnAttrs(html`<span>${countText(run.length, '{n} hook event', '{n} hook events')} · ${actors.join(', ')}</span>${icon('chevron')}`, action, value, 'text-btn team-hook-toggle', html`aria-expanded="${open}" data-fold-count="${run.length}"`)}</span><span class="team-exchange-end"><time data-tip="${whenText(run.at(-1).at)}">${clock(run.at(-1).at)}</time></span></li>`;
    return html`${line}${open ? run.map((h) => hookEvent(s, h, chosen?.id === h.id)) : ''}`;
  }
  function threadItems(s, entries, actor, facts = s.facts) {
    const inView = new Set(entries), replies = new Map(), items = [];
    const root = (e) => { let r = e; for (let hop = 0; hop < 20 && r.namesExchange && inView.has(r.namesExchange); hop++) r = r.namesExchange; return r; };
    for (const e of entries) {
      const r = e.kind === 'message' ? root(e) : e;
      if (r !== e) { if (!replies.has(r)) replies.set(r, []); replies.get(r).push(e); } else items.push({at: e.ordinal, entry: e});
    }
    if (!actor) for (const fold of foldUnits(factUnits(facts))) items.push({at: (fold.units[0].request || fold.units[0].response).item.ordinal, fold});
    return items.sort((a, b) => a.at - b.at).map((it) => (it.entry ? {...it, replies: replies.get(it.entry) || []} : it));
  }
  /* C4 item 7: pressing toggles the inline excerpt, reference and full provenance; preserve the open row's fill and rail. */
  const inlineRecord = (s, e) => html`<div class="team-record-inline">${e.kind === 'message' ? messageRecord(s, e) : hookRecord(s, e)}</div>`;
  /* C4 item 3: member chips filter exchanges; tips keep their declared composition and latest contribution. */
  const usageNumber = (value) => /^(0|[1-9][0-9]*)$/.test(String(value)) && Number.isSafeInteger(Number(value)) ? Number(value) : null;
  const usageCount = (value) => usageNumber(value) === null ? '' : count(usageNumber(value));
  const usageMetadata = (u) => {
    const source = t(u.source_kind === 'CODEX_SESSION_FILE' ? 'Codex session file' : u.source_kind === 'CLAUDE_CODE_SESSION_FILE' ? 'Claude Code session file' : 'Usage source not observed');
    const at = u.sample_time_kind === 'LATEST_USAGE_RECORD_AT' && u.last_at && Number.isFinite(Date.parse(u.last_at)) ? t('sampled {at}', {at: whenText(u.last_at)}) : t('Sample time not observed');
    return `${source} · ${at}`;
  };
  // ST7: each known count has its clause; unknown counts leave no label or separator.
  const usageTokenWords = (u) => [
    usageNumber(u.input_tokens) === null ? '' : t('{n} in', {n: usageCount(u.input_tokens)}),
    usageNumber(u.output_tokens) === null ? '' : t('{n} out', {n: usageCount(u.output_tokens)}),
    usageNumber(u.cache_read_tokens) === null ? '' : t('{n} cache read', {n: usageCount(u.cache_read_tokens)}),
    usageNumber(u.cache_write_tokens) === null ? '' : t('{n} cache written', {n: usageCount(u.cache_write_tokens)})
  ].filter(Boolean).join(' · ');
  const usageWords = (pt) => [...pt.usage.values()].map((u) => [`${u.model}${u.efforts.length ? ' (' + u.efforts.join(', ') + ')' : ''}`, usageTokenWords({input_tokens: u.input, output_tokens: u.output, cache_read_tokens: u.cacheRead, cache_write_tokens: u.cacheWrite})].filter(Boolean).join(' · ')).join(' · ');
  const pinWords = (pt) => pt.pins ? [pt.pins.model || pt.pins.effort ? `${t('card')} ${[pt.pins.model, pt.pins.effort].filter(Boolean).join(' ')}` : '', pt.pins.host ? `${t('host')} ${pt.pins.host}` : ''].filter(Boolean).join(' · ') : '';
  function memberChips(s, actor, all) {
    const members = TeamMembers.rows(s, ROLE_SAME).sort((a, b) => (isLead(b, s) ? 1 : 0) - (isLead(a, s) ? 1 : 0) || a.last - b.last);
    if (members.length < 2) return '';
    const chip = (pt) => {
      const {made, latest} = TeamMembers.activities(pt, all, ROLE_SAME, MESSAGE_KINDS), pressed = actor === pt.id;
      const declared = String(roleLabel(pt, s)).replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim(); // who it declared itself to be (law 17: the lead is a declaration)
      const differs = [...pt.usage.values()].flatMap((u) => u.pinDiffers);
      const tip = [declared, made || t('No exchange retained'), latest ? `${t('Latest contribution')} ${clock(latest.at)}` : '', pt.sharedUsage ? t('Usage shown with this thread’s first role') : usageWords(pt), pinWords(pt), differs.length ? t('differs from its card: {what}', {what: [...new Set(differs)].map((x) => codeWords(x)).join(', ')}) : ''].filter(Boolean).join(' · ');
      return btnAttrs(html`${icon(ROLE_ICONS[pt.displayRole] || roleIcon(s, pt.id))}<span>${pt.displayRole ? roleName(pt.displayRole) : actorName(s, pt.id)}</span>${differs.length ? icon('warning') : ''}`, 'team-actor', pressed ? '' : pt.id, 'team-chip team-member-chip', html`aria-pressed="${pressed}" data-role="${ROLE_TONES[pt.displayRole] || roleTone(s, pt.id)}" data-declared-role="${[...pt.roles].join(' ')}" data-tip="${tip}"`);
    };
    return html`<span class="team-members" role="group" aria-label="${t('Members')}">${members.map(chip)}</span>`;
  }
  // said once for the session, in the thread's (i), where no member's host events were observed
  const hostWords = (s) => ([...s.participants.values()].every((pt) => !pt.hooks.length) ? ' ' + t('No host event was observed for any member; each is as it declared.') : '');
  /* C4 items 2 and 6: session facts retain the lead, span, question and exact references. */
  function sessionFacts(s, stated, q, shownTask='') {
    const lead = [...s.participants.values()].find((pt) => isLead(pt, s));
    const at = stated.map((e) => e.at).filter(Boolean).sort(), a = at.length ? clock(at[0]) : '', b = at.length ? clock(at.at(-1)) : '';
    const sameDay = a.split(' ').slice(0, -1).join(' ') === b.split(' ').slice(0, -1).join(' ');
    const span = !a ? '' : a === b ? a : `${a} → ${sameDay ? b.split(' ').at(-1) : b}`;
    // Law 17: references lead to their first declaration; the head already names its bound Task.
    const references = referencesOf(stated).filter(group => group[0].qualified.kind !== 'task' || group[0].qualified.ref !== shownTask).map(group => revisionRefs(group, e => {
      const ref = e.qualified.ref, owner = S.resolved.get(e.reference);
      const objected = stated.some((x) => x.messageKind === 'objection' && x.qualified?.ref === ref), verified = S.resolved.get(e.reference)?.level === 'verified';
      const word = [objected ? t('objected') : '', verified ? t('verified by its owner') : ''].filter(Boolean).join(' · ');
      return btn(html`${owner?.title ? html`<span class="owner-text">${owner.title}</span>` : referenceKind(e.qualified)} <span class="mono">${short(String(ref).split(':').at(-1), SHORT.id)}</span>${owner?.revision ? html` · ${t('revision {n}', {n: owner.revision})}` : ''}${word ? html` <span class="muted">· ${word}</span>` : ''}`, 'team-reveal', e.id, 'text-btn team-reference');
    }));
    const goals = [...s.goals].map((id) => link(html`<span class="mono">${short(id, SHORT.id)}</span>`, 'goal', 'text-btn', {goal: id}));
    // U51: the models and tokens are the Participants folder's, by member and model, the session's totals at its foot
    return [[t('Lead'), lead ? actorName(s, lead.id) : html`<span class="muted">${t('no main PM declared')}</span>`], ...(span ? [[t('Span'), span]] : []), ...(goals.length ? [[t('Goals'), html`${goals}`]] : []), ...(s.goalScope ? [[t('Participants'), hint(participantCount(s), t('Only contributions attributed to this Goal count as participation; session-wide usage alone does not.'))]] : []), ...(q.kind !== 'none' ? [[t('Question'), questionSource(q)]] : []), ...(references.length ? [[t('Declared references'), html`${references}`]] : [])];
  }
  /* The thread's product lines (C4 item 1): what the members' work made the product record -- never a
   * person's Local Web reads of the session's references (the owner records `caller HUMAN`), nor what was
   * recorded before its first exchange; both are observations of the references, on Observations, and the
   * thread says them once at its top. */
  // a read, by the operation's recorded name: a preview, a packet, a dossier or a finding read, a readback, an export, a plan
  const READ_OPS = /PREVIEW|PACKET|DOSSIER|FINDING|READBACK|EXPORT|_PLAN$/;
  const byPerson = (f) => f.item.schema_kind === 'ProductOperationObserved' && f.item.payload?.caller === 'HUMAN';
  const humanRead = (f) => byPerson(f) && READ_OPS.test(String(f.item.payload?.operation || ''));
  function threadFacts(s, stated) {
    const first = stated[0]?.ordinal ?? Infinity, human = new Set(s.facts.filter(humanRead).map((f) => f.item.observation_id));
    const followsHuman = (f) => !f.named.length && (f.followed || []).length > 0 && f.followed.every((x) => human.has(x.by));
    const off = (f) => human.has(f.item.observation_id) || followsHuman(f);
    const inThread = s.facts.filter((f) => !off(f) && f.item.ordinal > first);
    return {inThread, earlier: factUnits(s.facts.filter((f) => !off(f) && f.item.ordinal <= first)).length, reads: factUnits(s.facts.filter(off)).length};
  }
  function outsideLine(s, tf) {
    if (!tf.earlier && !tf.reads) return '';
    const parts = [tf.earlier ? countText(tf.earlier, '{n} product operation before its first exchange', '{n} product operations before its first exchange') : '', tf.reads ? countText(tf.reads, '{n} Local Web read of its references', '{n} Local Web reads of its references') : ''].filter(Boolean);
    return html`<p class="team-outside">${icon('activity')}<span>${t('Recorded outside this conversation')}: ${parts.join(' · ')} · ${link(t('on Product record'), 'team-evidence', 'inline-link', {team: s.id})}</span></p>`;
  }
  /* Law 137: disclose the owner question, explicit case choices and related references when the head cannot hold them. */
  function questionCard(s, all, q) {
    return q.kind === 'case-verified' || q.kind === 'choice' || q.choice || q.related.length ? questionFacts(s, all) : '';
  }
  /* The retained entries of one session in receipt order, the replays dropped, linked. */
  function entriesOf(s) { const all = [...s.entries.values()].sort((a, b) => a.ordinal - b.ordinal); linkExchanges(s, all.filter((e) => !e.replayOf)); return all; }
  const latestAt = (s) => { const at = [...s.entries.values()].map((e) => e.at).filter(Boolean).sort().at(-1); return at || ''; };
  /* The record of one session (round 67): every retained exchange in recorded order, as the
   * thread renders them — a run log, never a conversation. */
  function recordOf(sessionId) {
    const s = scene(true).sessions.find((v) => v.id === sessionId);
    if (!s) return html`<p class="caption">${t('Selected session not retained')}</p>`;
    const entries = entriesOf(s).filter((e) => !e.replayOf);
    return html`<ol class="team-record">${entries.map((e) => exchange(s, e, false))}</ol>`;
  }
  /* The thread's Display (C3, law 142; the lobby's component, F1): Show -- everything, the questions
   * and objections, the objections awaiting the Main PM, or one member -- and the density; never an
   * order: the store's time is the thread's only one. Kept per session and viewer; one member is the
   * route's filter (a link carries it), so that choice is read from the route and written there. A
   * kind keeps the comments that ask -- in the comment or among its replies -- with every reply;
   * the hooks and the product's lines are not among them. */
  const SHOWS = {all: ['Everything', ''], asks: ['Questions and objections', 'questions and objections'], awaiting: ['Awaiting the Main PM', 'objections awaiting the Main PM'], assignments: ['Assignments', 'assignments'], notes: ['Decision notes', 'decision notes']};
  const KEEPS = {asks: (e) => e.messageKind === 'question' || e.messageKind === 'objection', awaiting: (e) => e.messageKind === 'objection' && !e.pmResponses.length, assignments: (e) => e.messageKind === 'assignment', notes: (e) => NOTE_KINDS.has(e.messageKind)};
  function threadDisplay(s, actor) {
    const name = 'thread.' + String(s.id || 'undeclared').replace(/[^\w.-]/g, '_');
    const members = [...s.participants.values()].sort((a, b) => (isLead(b, s) ? 1 : 0) - (isLead(a, s) ? 1 : 0) || a.last - b.last);
    const shows = Object.entries(SHOWS).map(([key, [word]]) => [key, t(word)]);
    const saved = displayState(name, {shows}), at = members.findIndex((pt) => pt.id === actor);
    const show = actor ? (at >= 0 ? 'm' + at : '') : saved.show || 'all';
    const apply = (d) => {
      const i = /^m(\d+)$/.exec(d.show)?.[1];
      if (i != null) { setDisplay(name, {show: 'all'}); return showActor(members[Number(i)]?.id || ''); }
      if (actor && d.show !== show) return showActor('');
      paint();
    };
    return {name, show: actor ? '' : show, density: saved.density, spec: {shows, members: members.map((pt, i) => ['m' + i, actorName(s, pt.id)]), current: {show}, density: true, apply}};
  }
  /* The conversation (N5, law 125; C4): the session read as an issue -- its question the page's title,
   * its facts the head's line, the thread its comments and events in one column; the members its filter;
   * an exchange pressed opens whole in the inspector (none is read, none is filled). The thread flows
   * with the page (law 113): no box scrolls on its own. */
  function sessionView(s, actor) {
    const all = entriesOf(s), stated = all.filter((e) => !e.replayOf && e.channel !== 'PRODUCT_COMMITTEE');
    const entries = stated.filter((e) => !actor || e.actor === actor || recipientParticipant(s, e.recipient)?.id === actor);
    const chosenId = selection().event, chosen = chosenId ? entries.find((e) => e.id === chosenId) || null : null;
    const unread = entries.filter((e) => S.incoming.has(e.id)).length;
    const unknownActor = actor && !s.participants.has(actor);
    const display = threadDisplay(s, actor), keep = KEEPS[display.show], tf = threadFacts(s, stated);
    const chosenProduct = !actor && !keep && tf.inThread.some((f) => f.item.observation_id === chosenId && productAcceptedAnswer(s, f));
    const items = threadItems(s, entries, actor, tf.inThread).filter((it) => !keep || (it.entry?.kind === 'message' && [it.entry, ...it.replies].some(keep))), shown = items.slice(-S.visible);
    // the path says Conversation (F3): the head counts what it holds; its tools -- the members, mark seen, the Display (C3)
    const held = keep ? items.reduce((n, it) => n + 1 + it.replies.length, 0) : entries.length;
    const unidentified = stated.filter((e) => !e.actor).length; // said beside the count: an event no member is named for
    const head = sectionHead(html`${countText(held, '{n} exchange', '{n} exchanges')}${unidentified ? html` <span class="muted">· ${countText(unidentified, '{n} event without an agent id', '{n} events without an agent id')}</span>` : ''}`, `${t('Member statements: assignments, questions and replies in receipt order. A statement is what a member said, never a product fact.')}${hostWords(s)}`, html`<span class="team-head-tools">${memberChips(s, actor, stated)}${unread ? btn(t('{n} unread · mark seen', {n: unread}), 'team-seen', '', 'button compact') : ''}${displayOptions(display.name, display.spec)}</span>`);
    const filter = actor ? noteLine(html`${t('Showing')} ${actorName(s, actor)}`, '', 'neutral', btn(t('Show everyone'), 'team-actor', '', 'button compact'))
      : keep ? noteLine(html`${t('Showing')} ${t(SHOWS[display.show][1])}`, '', 'neutral', btn(t('Show everything'), 'display-set', display.name + ':show:all', 'button compact')) : '';
    // arrivals below the reader's place are one pill at the thread's top; a press goes to the newest (C3, law 142) -- a fold that holds it is its place
    const inThread = new Set(shown.flatMap((it) => (it.entry ? [it.entry.id, ...it.replies.map((r) => r.id)] : [])));
    const fresh = entries.filter((e) => S.incoming.has(e.id) && inThread.has(e.id)), newest = fresh.at(-1)?.id;
    const pill = newest ? html`<div class="team-new-bar" data-waypoint="#team-event-${newest}, [data-holds~='${newest}']" hidden>${btnAttrs(html`${icon('arrow')}<span>${countText(fresh.length, '{n} new exchange', '{n} new exchanges')}</span>`, 'team-newest', newest, 'button compact team-new-pill')}</div>` : '';
    // consecutive hooks gather into one run (C2), consecutive product lines into one (C4): a run of one stays its own line
    const runs = [];
    for (const it of shown) {
      const hook = it.entry && it.entry.kind !== 'message', last = runs.at(-1);
      if (hook && last?.hooks) last.hooks.push(it.entry);
      else if (it.fold && last?.folds) last.folds.push(it.fold);
      else runs.push(hook ? {hooks: [it.entry]} : it.fold ? {folds: [it.fold]} : it);
    }
    const line = (it) => it.hooks ? (it.hooks.length > 1 ? hookRun(s, it.hooks, chosen) : hookEvent(s, it.hooks[0], chosen?.id === it.hooks[0].id)) : it.folds ? (it.folds.length > 1 ? productRun(s, it.folds) : productEvent(s, it.folds[0])) : exchange(s, it.entry, chosen?.id === it.entry.id, it.replies, chosen);
    const empty = keep ? t(display.show === 'awaiting' ? 'No objection awaits the Main PM.' : 'No question or objection is recorded.') : t('Nothing retained for this selection.');
    const thread = shown.length ? html`<ol class="team-thread${display.density === 'compact' ? ' density-compact' : ''}" id="teamThread" role="log" aria-live="off">${runs.map(line)}</ol>` : emptyState(empty);
    const missing = chosenId && !chosen && !chosenProduct ? noteLine(t('Selected exchange not retained'), t('Selected exchange is not retained for this selection.'), 'neutral') : '';
    return html`${unknownActor ? noteLine(t('Selected participant not retained'), actor, 'neutral') : ''}${missing}${questionCard(s, stated, questionOf(s, stated))}
      <section class="team-conversation" aria-label="${t('Conversation')}">${head}${filter}${committeeLinks(s)}${actor ? '' : outsideLine(s, tf)}${items.length > S.visible ? btn(t('Show earlier exchanges'), 'team-more', '', 'button compact') : ''}${pill}${thread}</section>
      ${s.unknownKinds.length ? noteLine(t('Unknown event kinds in this session'), s.unknownKinds.map(({item}) => `${item.payload.event_kind} (${item.payload.producer_id})`).join(', '), 'neutral') : ''}`;
  }
  /* The Product record (U53, the user 2026-09-30: 至今都没看懂这个observation是干什么的; C4 items 4 and 9 before it): what the product
   * recorded naming a reference this session declared, one column. Its kinds are the list's filter; the
   * list groups by operation, outcome and caller -- the session's own operations apart from the Local Web
   * reads of its references -- one row a group: its count, its span, the outcome's words once, the
   * references it names and the exchanges that declared them; a group opens in place. The Tasks it names
   * follow, their current state kept apart from the recorded history. */
  const FACT_KINDS = [['refusal', 'Product refusals'], ['correction', 'Returned for correction'], ['preview', 'Product previews'], ['admission', 'Admissions'], ['reuse', 'Exact reuses'], ['requested', 'Requests'], ['returned', 'Product returns'], ['control', 'Task Control facts'], ['artifact', 'Owner-verified artifacts']];
  function factKind(f) {
    const item = f.item, p = item.payload || {};
    if (item.availability !== 'AVAILABLE' || !item.payload) return '';
    if (item.schema_kind === 'ProductOperationObserved') return p.status === 'REFUSED' || (p.phase === 'FAILED' && p.failure_code) ? 'refusal' : p.status === 'CORRECT' ? 'correction' : p.status === 'PLANNED' ? 'preview' : ['ADMITTED', 'REUSED_IN_FLIGHT'].includes(p.status) ? 'admission' : p.status === 'REUSED_EXACT' ? 'reuse' : p.phase === 'REQUESTED' ? 'requested' : 'returned';
    return item.schema_kind === 'TaskControlTransition' ? 'control' : artifactVerified(item) ? 'artifact' : '';
  }
  function factGroups(facts) {
    const groups = new Map();
    for (const u of factUnits(facts)) {
      const f = u.response || u.request, {title, state, p} = factFacts(f), who = byPerson(f) ? 'person' : 'member';
      const key = 'group:' + [title, state, p.failure_code || '', who].join('|');
      if (!groups.has(key)) groups.set(key, {key, title, who, units: []});
      groups.get(key).units.push(u);
    }
    return [...groups.values()];
  }
  function groupRow(s, g, stated) {
    const first = g.units[0], last = g.units.at(-1), from = first.request || first.response, f = last.response || last.request, {tone, state, p} = factFacts(f), n = g.units.length, open = S.unfolded.has(g.key);
    const refs = [...new Set(g.units.flatMap((u) => (u.response || u.request).named))];
    const names = refs.map((r) => { const e = stated.find((x) => x.qualified?.ref === r); return e ? html`${link(html`<span class="mono">${short(String(r).split(':').at(-1), SHORT.id)}</span>`, 'team', 'inline-link', {team: s.id, event: e.id})} <span class="muted">${t('declared by {member}', {member: actorName(s, e.actor)})}</span>` : html`<span class="mono">${short(String(r).split(':').at(-1), SHORT.id)}</span>`; });
    // the clip census (2026-09-24): the title says what, the second line (it wraps) says who, how often, when
    const why = html`${t(g.who === 'person' ? 'a person on the Local Web' : 'the session\'s own operations')} · ${countText(n, '{n} time', '{n} times')} · ${t('first')} ${when(from.item.occurred_at)} · ${t('last')} ${when(f.item.occurred_at)}${p.failure_code ? html` · ${coded(p.failure_code)}${withheldWhy(p)}` : ''}`;
    // LS2/PG2: declared-reference links are controls beside the title, never nested in its clipped main.
    const refsSlot = {columns: ['names'], props: [names.length ? factsRef(t('names'), html`${names.map((name) => html`<p>${name}</p>`)}`) : '']};
    if (sameButTime(g)) return evidenceRow({kind: 'Observation', type: 'observation', subject: g.title, state: tone, word: codeWords(state), why, id: g.key}, null, {...refsSlot, cls: 'team-fold-row'}); // U53: nothing to open
    return evidenceRow({kind: 'Observation', type: 'observation', subject: g.title, state: tone, word: codeWords(state), why, id: g.key}, null, {...refsSlot, to: {action: 'team-fold', value: g.key}, cls: 'team-fold-row', attrs: html`data-fold="${g.key}" data-fold-count="${n}" aria-expanded="${open}"`});
  }
  function evidenceView(s) {
    if (!s.facts.length) return emptyState(html`${t('No product observations are retained in the current activity window.')}${infoMark(t('This view lists retained product operation receipts attributed to this session or naming its exact references. A native Start/Stop or a completion relay does not supply one.'))}`, link(t('Outputs'), 'team-outputs', 'text-btn', {team: s.id, actor: '', event: ''}), '', 'elsewhere');
    const stated = entriesOf(s).filter((e) => !e.replayOf), tally = new Map();
    for (const f of s.facts) { const k = factKind(f); if (k) tally.set(k, (tally.get(k) || 0) + 1); }
    const kind = tally.has(S.factKind) ? S.factKind : '';
    const chips = [['', t('All'), s.facts.length], ...FACT_KINDS.filter(([k]) => tally.has(k)).map(([k, w]) => [k, t(w), tally.get(k)])].map(([k, w, n]) => btnAttrs(html`<span>${w}</span><span class="num">${count(n)}</span>`, 'team-fact-kind', k, 'team-chip', html`aria-pressed="${kind === k}"`));
    const groups = factGroups(kind ? s.facts.filter((f) => factKind(f) === kind) : s.facts);
    const rowsOf = (g) => (g.units.length === 1 ? [unitRow(g.units[0])] : [groupRow(s, g, stated), ...(S.unfolded.has(g.key) && !sameButTime(g) ? g.units.map((u) => unitRow(u, g)) : [])]);
    // U53: the session's own work leads; what a person read or asked on the Local Web is one line, closed until pressed
    const own = groups.filter((g) => g.who !== 'person'), people = groups.filter((g) => g.who === 'person');
    const read = people.flatMap((g) => g.units), times = read.map((u) => (u.response || u.request).item.occurred_at).filter(Boolean).sort();
    const peopleOpen = S.unfolded.has('people');
    // a person's line has no state of its own: its glyph, its count and span, closed until pressed
    const peopleRow = read.length ? objectRow({lead: 'user', name: t('Read or asked by a person on the Local Web'), why: html`${countText(read.length, '{n} record', '{n} records')} · ${t('first')} ${when(times[0])} · ${t('last')} ${when(times.at(-1))} · ${t('not the members\' work')}`, to: {action: 'team-fold', value: 'people'}, cls: 'evidence-row team-fold-row'}, {key: 'people', attrs: html`data-fold="people" data-fold-count="${people.length}" aria-expanded="${peopleOpen}"`}) : '';
    const rows = [...(own.length ? own.flatMap(rowsOf) : [emptyState(t('Nothing the session\'s members ran is recorded for what it cited.'))]), peopleRow, ...(peopleOpen ? people.flatMap(rowsOf) : [])];
    const states = currentStates(s);
    return html`<div class="team-fact-kinds" role="group" aria-label="${t('Observations by kind')}" data-tip="${t('Each count is of the events the product observed: a request and its answer are two.')}">${chips}${newFacts(s) ? html`<span class="team-fact-new">${newFacts(s)}</span>` : ''}</div><section class="panel team-product" id="teamProductFacts" aria-label="${t('Product record')}"><div class="card-list lines">${rows}</div></section>${states.length ? panel(t('Named Tasks · current state'), '', html`<div class="team-current-states">${states}</div>`) : ''}`;
  }
  function setFactKind(key) { S.factKind = key || ''; paint(); }
  /* Sessions: every retained conversation by its own words, the shown one marked. */
  /* Round 73: Session runs show question, lead and first/latest instants; their state is recorded, never a lifecycle. */
  function runs(sessions = scene().sessions) {
    return sessions.map((s) => {
      const all = entriesOf(s), q = questionOf(s, all.filter((e) => !e.replayOf));
      const objections = all.filter((e) => !e.replayOf && e.messageKind === 'objection'), unresolved = objections.filter((e) => !e.pmResponses.length).length;
      const lead = [...s.participants.values()].find((pt) => isLead(pt, s));
      return {id: s.id, kind: 'session', name: titleOf(q), question: q.text || '', state: 'recorded', starter: lead ? actorName(s, lead.id) : '', started: all.map((e) => e.at).filter(Boolean).sort()[0] || '', finished: latestAt(s), exchanges: s.entries.size, participants: participantCount(s), roles: rolesOf(s), unread: [...s.entries.keys()].filter((id) => S.incoming.has(id)).length, objections: objections.length, unresolved, object: s};
    });
  }
  /* F1, law 136; N5, law 134; round 20: Session rows show short id/question, then lead/size. Unanswered objections group open, recorded Sessions folded; time is the other axis. */
  function sessionsView(sessions) { // the lobby marks no session: none is open while it is shown (the user, 2026-09-24: back from a session, its row stayed selected)
    const items = runs(sessions), leads = [...new Set(items.map((r) => r.starter).filter(Boolean))];
    const roles = [...new Set(items.flatMap((r) => r.roles))].sort((a, b) => ROLE_ORDER.indexOf(a) - ROLE_ORDER.indexOf(b));
    const row = (r, d) => { const on = (k) => d.props[k] !== false; return objectRow({lead: 'team', name: r.name, ref: sessionLabel(r.id), to: {page: 'team', extra: {team: r.id, actor: '', event: ''}}, cls: 'retained-session-row'}, {key: r.id, columns: ['lead', 'roles', 'size', 'objections'], props: [on('lead') && r.starter ? ['', r.starter, 'drop'] : '', on('roles') && r.roles.length ? ['', r.roles.map(roleName).join(' · '), 'drop'] : '', on('size') ? ['', html`${countText(r.participants, '{n} participant', '{n} participants')} · ${countText(r.exchanges, '{n} exchange', '{n} exchanges')}${r.unread ? html` <span class="team-unread">${t('{n} unread', {n: r.unread})}</span>` : ''}`, 'drop'] : '', on('objections') && r.objections ? html`<span class="team-objection-status" data-answered="${!r.unresolved}">${countText(r.objections, '{n} objection', '{n} objections')}${r.unresolved ? html` · ${r.unresolved} ${t('unresolved')}` : ''}</span>` : ''], time: r.finished ? when(r.finished) : ''}); };
    const axes = [{key: 'state', label: t('State'), group: (r) => r.unresolved ? {key: 'awaiting', label: t('Awaiting the Main PM'), rank: 0, open: true} : {key: 'recorded', label: t('Recorded'), rank: 1, open: false}},
      {key: 'time', label: t('Time'), group: (r) => timeGroup(r.finished)}];
    // the owner holds sessions before the pages read: the lobby's foot says so and reads the next
    const older = X.more ? Lobby.older(t('Older sessions are not read yet'), 'team-older', '', Boolean(X.fetching)) : '';
    return Lobby.render('sessions', {items, row, axes, foot: older, words: (r) => [r.name, r.question, r.id, r.starter, ...r.roles.map(roleName)].join(' '), placeholder: t('Question, session or lead'),
      filters: [...(leads.length > 1 ? [{field: 'lead', label: t('Lead'), multiple: true, options: leads.map((l) => [l, l]), test: (r, one) => r.starter === one}] : []),
        ...(roles.length ? [{field: 'role', label: t('Participant'), multiple: true, options: roles.map((x) => [x, roleName(x)]), test: (r, one) => r.roles.includes(one)}] : [])],
      properties: [['lead', t('Lead')], ['roles', t('Roles')], ['size', t('Participants and exchanges')], ['objections', t('Objections')]]}); // N3 (law 121): the path says Sessions
  }
  const committeeRefs = s => [...new Set([...s.entries.values()].filter(e => e.channel === 'PRODUCT_COMMITTEE').map(e => e.reference).concat(s.facts.filter(f => f.attributed && f.item.payload?.operation === 'COMMITTEE_OPEN' && f.item.payload?.status === 'COMMITTEE_OPEN').map(f => f.item.payload.subject?.update_task_id || f.item.payload.task_id), S.committee?.session === s.id ? S.committee.body?.update_task_id : null).filter(id => UUID.test(id || '')))];
  const committeeTask = s => selection().committee || (committeeRefs(s).length === 1 ? committeeRefs(s)[0] : '');
  function hasCommittee() { const sel = selection(), s = scene().sessions.find(s => s.id === sel.session); return Boolean(s && (sel.committee || committeeRefs(s).length) || S.committee?.session === sel.session && S.committee?.body?.update_task_id === sel.committee); }
  async function readCommittee(s, force = false) {
    const task = committeeTask(s); if (!task) return;
    const last = [...s.entries.values()].filter(e => e.channel === 'PRODUCT_COMMITTEE' && e.reference === task).at(-1)?.id || '';
    const changed = s.facts.filter(f => ['COMMITTEE_OPEN', 'COMMITTEE_SUBMIT'].includes(f.item.payload?.operation) && (f.item.payload.subject?.update_task_id || f.item.payload.task_id) === task).at(-1)?.item.observation_id || '';
    const epoch = LiveActivity.state().epoch, key = [epoch, task, last, changed].join('|'), was = S.committee;
    if (was?.session === s.id && was.task === task && (was.busy || (!force && was.key === key))) return was.promise;
    const mine = S.committee = {session: s.id, task, key, epoch, busy: true, deadlineRead: was?.deadlineRead, body: was?.epoch === epoch && was?.session === s.id && was.task === task ? was.body : null}, intent = Data.navigationIntent();
    mine.promise = Data.read('/api/committee?' + new URLSearchParams({update_task_id: task})).then(b => {
      if (S.committee !== mine || !Data.navigationCurrent(intent)) { if (S.committee === mine) S.committee = null; return; }
      if (b.status === 'REFUSED') mine.refused = b;
      else if (b.update_task_id !== task) mine.error = t('The owner answered for another Task.');
      else if (b.session?.[1] !== s.id) mine.error = t('The owner answered for another session.');
      else mine.body = b;
    }, e => { if (S.committee === mine) { if (Data.navigationCurrent(intent)) mine.error = e.message; else S.committee = null; } }).finally(() => { if (S.committee === mine) mine.busy = false; paint(); });
    return mine.promise;
  }
  function committeeLinks(s) {
    return html`${committeeRefs(s).map(task => { const b = S.committee?.task === task ? S.committee.body : null; return html`<p class="caption">${link(b ? countText(b.message_count, 'Committee · {n} message', 'Committee · {n} messages') : t('Committee'), 'team-committee', 'text-btn', {team: s.id, actor: '', event: '', committee: task, committee_point: ''})}${b?.first_message_at ? html` · ${when(b.first_message_at)}${b.last_message_at !== b.first_message_at ? html` – ${when(b.last_message_at)}` : ''}` : ''}</p>`; })}`;
  }
  function committeeEntries(b) {
    return b.messages.map(m => {
      const row = m.row, sub = row.subject, recipient = Object.keys(b.member_agents).find(role => b.member_agents[role] === sub.recipient_id);
      return {id: row.producer_session + ':' + m.id, ordinal: row.producer_sequence, kind: 'message', at: row.occurred_at, subject: sub, payload: row, actor: sub.native_agent_id, role: sub.role, channel: sub.input_channel, messageKind: sub.message_kind, messageId: sub.message_id, recipient: sub.recipient_id, recipientRole: b.member_cards[recipient], reference: sub.reference, replyTo: sub.reply_to, qualified: qualify(sub.reference), text: m.text, outcome: m.outcome, targets: m.targets, targetText: m.target_text, dissentStands: b.standing_dissents.includes(m.id), ownerFloor: true, conflictsWith: []};
    });
  }
  function committeeView(s, actor) {
    const task = committeeTask(s), selected = selection().point, o = S.committee?.session === s.id && S.committee.task === task ? S.committee : null;
    if (!task) return committeeRefs(s).length ? Lobby.render('committee-floors', {items: committeeRefs(s), words: String, columns: [], axes: [{key: 'committee', label: t('Committee'), group: () => ({key: 'committee', label: t('Committee'), open: true})}], row: id => objectRow({lead: 'team', name: t('Committee'), ref: id, to: {page: 'team-committee', extra: {team: s.id, committee: id, committee_point: '', actor: '', event: ''}}})}) : emptyState(t('No committee floor is open'));
    if (!o?.body) return o?.refused ? refusal(o.refused, TONE.attention, {catalog: true}) : o?.error ? notRead(t('Committee not read'), o.error, '', btn(t('Read again'), 'team-committee-read', '', 'button compact')) : skeleton('rows');
    const b = o.body, entries = committeeEntries(b);
    linkExchanges(s, entries);
    const members = html`<div class="team-members">${Object.entries(b.members).map(([role, state]) => btnAttrs(html`<span>${roleName(b.member_cards[role])}</span> · ${{IN: t('committee|Received'), NOT_YET: t('Not yet received'), NOT_ADDRESSED: t('Not addressed')}[state] || codeWords(state)}`, 'team-actor', actor === b.member_agents[role] ? '' : b.member_agents[role], 'team-chip team-member-chip', html`aria-pressed="${actor === b.member_agents[role]}" data-member="${role}" data-state="${state}"`))}</div>`;
    const points = Lobby.render('committee-agenda', {items: b.tension_points, tools: false, columns: [{key: 'dissent', label: t('Dissent stands')}], axes: [{key: 'state', label: t('State'), group: p => ({key: p.state, label: codeWords(p.state), open: true})}], row: p => objectRow({lead: 'evidence', name: p.alias, to: {action: 'team-committee-point', value: p.alias}, why: p.word ? t(p.word, {...p.words, ...(p.words?.reason ? {reason: declaredCodeWord(p.words.reason) ? codeWords(p.words.reason) : t(p.words.reason)} : {})}) : html`<span class="owner-text">${p.text}</span>`}, {key: p.alias, selected: selected === p.alias, columns: ['dissent'], props: [p.dissent_stands ? ['', stateLine('review_pending', {word: t('Dissent stands'), next: ''})] : ''], attrs: html`data-point="${p.alias}" data-state="${p.state}"`})});
    const shown = entries.filter(e => (!actor || e.actor === actor || e.recipient === actor) && (!selected || e.targets?.includes(selected) || e.namesExchange?.targets?.includes(selected))), items = threadItems(s, shown, actor, []), chosen = shown.find(e => e.id === selection().event);
    const incoming = new Set([...s.entries.values()].filter(e => e.channel === 'PRODUCT_COMMITTEE' && e.reference === task && S.incoming.has(e.id)).map(e => e.messageId));
    const fresh = shown.filter(e => incoming.has(e.messageId)), newest = fresh.at(-1)?.id;
    const pill = newest ? html`<div class="team-new-bar" data-waypoint="#team-event-${newest}" hidden>${btnAttrs(countText(fresh.length, '{n} new exchange', '{n} new exchanges'), 'team-newest', newest, 'button compact team-new-pill')}</div>` : '';
    const floor = items.length ? html`<ol class="team-thread" id="teamThread" role="log" aria-live="off">${items.map(it => exchange(s, it.entry, chosen?.id === it.entry.id, it.replies, chosen))}</ol>` : b.stage === 'STANCES' ? '' : emptyState(t('The floor’s recorded messages are not retained here yet.'));
    return html`${o.refused ? refusal(o.refused, TONE.attention, {catalog: true}) : o.error ? notRead(t('Committee not read'), o.error) : ''}${stackSlot('committeeProgress', b.stage === 'STANCES' ? stat(t('Stances received'), html`${b.stances_in} / ${b.member_count}`) : '')}${stackSlot('committeeMembers', members)}${stackSlot('committeeAgenda', html`${sectionHead(t('Agenda'))}${points}`)}${b.person_items.length ? panel(t('For you'), t('Asked in your conversation with the agent'), b.person_items.map(p => html`<p class="owner-text" data-person-item="${p.id}">${p.question}</p><p class="caption">${when(p.asked_at)}</p>${p.answer !== null ? html`<p class="owner-text">${p.answer}</p><p class="caption">${p.relay_word ? html`${t(p.relay_word)} · ` : ''}${when(p.answer_at)}</p>` : ''}`)) : ''}${stackSlot('committeeFloor', html`${sectionHead(t('committee|Floor'))}${selected ? noteLine(html`${t('Showing')} ${selected}`, '', 'neutral', btn(t('Show everything'), 'team-committee-point', '', 'button compact')) : ''}${pill}${floor}`)}`;
  }
  const chooseCommitteePoint = point => { replaceHash({committee_point: selection().point === point ? '' : point, event: ''}); paint(); };
  // One lede per page; member statements remain separate from product evidence.
  const LEDES = {team: 'The conversation, who is in it and what they cite.', 'team-participants': 'Who took part, the models they ran and the tokens they spent.', 'team-sessions': 'Retained conversations, read back exactly.', 'team-outputs': 'What this session produced: accepted answers, submitted Tasks and their artifacts, and the goals it took.', 'team-evidence': 'The product\'s records about this session.'};
  // N3: what a section's caption said is the page's (i)
  const INFO = {'team-outputs': 'Accepted answers retain their recorded author, submitter and sealed references. Tasks appear only when this session submitted them; goals appear only when bound to it.', 'team-participants': 'Each member\'s latest reading by model: its own so far, never a sum of readings, and not attributed to a goal or a Task.', 'team-evidence': 'Operations and their answers attributed to this session or naming its exact references, with Task Control facts and verified artifacts; the session\'s own work first, a person\'s reads on the Local Web in one line.'};
  function section() {
    const view = app.page, {sessions, unknown} = scene(false, ['team-sessions', 'team-participants'].includes(view)), sel = selection();
    // law 123: a session's page with no session chosen opens Sessions, the Team's Home
    if (view !== 'team-sessions' && !sel.session && sessions.length) { app.page = 'team-sessions'; replaceHash({page: 'team-sessions'}); return section(); }
    const chosen = sel.session ? sessions.find((s) => s.id === sel.session) || (sel.committee ? sessionState(sel.session) : null) : null;
    if (S.answerDetail && (view !== 'team' || S.answerDetail.epoch !== answerEpoch() || !sameAnswer(S.answerDetail, chosen, selectedAnswerRecord(chosen, sel.event)))) S.answerDetail = null;
    // Global entries restore the retained selection; an absent named session stays unavailable.
    S.retained = {session: chosen?.id || sel.session, actor: sel.actor, event: sel.event, committee: sel.committee, point: sel.point};
    if (chosen && view !== 'team-sessions' && committeeTask(chosen)) void readCommittee(chosen);
    // the two sentences of the workroom (what it is; what a member statement is not) are the head's (i)
    const readback = X.error ? html`<p class="caption">${t('The sessions\' readback is unavailable ({error}); only what the activity feed retains is shown.', {error: X.error})}</p>` : '';
    const body = view === 'team-sessions' ? (sessions.length ? sessionsView(sessions) : emptyState(html`${t('No session yet')}${infoMark(t('No native session has been declared in the retained activity window. The foreground Codex remains the only conversational entry; nothing is simulated here.'))}`,'','page-empty'))
      : !sessions.length && !chosen ? emptyState(html`${t('No session yet')}${infoMark(t('No native session has been declared in the retained activity window. The foreground Codex remains the only conversational entry; nothing is simulated here.'))}`,'','page-empty')
      : !chosen ? noteLine(t('Selected session not retained'), html`<span class="mono">${sessionLabel(sel.session)}</span> ${t('is not in the retained activity window (reset, gap or retention).')}`, 'neutral', btn(t('Choose a session'), 'team-select', '', 'button compact'))
      : view === 'team-committee' ? committeeView(chosen, sel.actor) : view === 'team-evidence' ? evidenceView(chosen) : view === 'team-participants' ? participantsView(chosen) : view === 'team-outputs' ? outputsView(chosen) : sessionView(chosen, sel.actor);
    const state = LiveActivity.state();
    const usage = LiveActivity.nativeUsageState?.(), catching = usage?.status === 'OBSERVING' && usage?.reason === 'native_bridge.usage_catching_up';
    const measured = Number.isSafeInteger(usage?.read_through) && Number.isSafeInteger(usage?.total_bytes);
    const usageFailure = catching ? noteLine(t('Reading usage'), html`<span data-usage-notice="CATCHING_UP">${measured ? t('Usage read through {read} of {total} bytes; the next read continues.', {read: usage.read_through, total: usage.total_bytes}) : ''}</span>${usage.detail && measured ? infoMark(t(usage.detail, {read: usage.read_through, total: usage.total_bytes})) : ''}`, 'neutral') : usage?.status === 'UNAVAILABLE' ? noteLine(t('Usage observation unavailable'), html`<span data-tip="${usage.reason || ''}">${t(usage.detail || 'The usage reading is unavailable; research can continue.')}</span>`, 'neutral') : usage?.rescan ? noteLine(t('Usage rescanned'), t('The session file changed; its usage was read again.'), 'neutral') : '';
    // The feed's own condition, said once at the top: current, not current, or unreachable.
    const feed = state.error ? ['unreachable', t('Feed unreachable')] : state.stale || ['RESET', 'UNAVAILABLE'].includes(state.disposition) ? ['stale', t('Feed not current')] : ['live', t('Live feed')];
    // All session tabs share its state and facts; the floor's standing is a fact.
    const stated = chosen ? entriesOf(chosen).filter((e) => !e.replayOf) : [];
    const conversation = view !== 'team-sessions' && chosen ? questionOf(chosen, stated) : null;
    const floor = chosen && S.committee?.session === chosen.id && S.committee?.task === committeeTask(chosen) ? S.committee?.body : null;
    const floorFacts = floor ? [[t(floor.verdict ? 'Verdict' : 'Committee'), html`<span data-committee-stage="${floor.stage}"${floor.verdict ? html` data-verdict="${floor.verdict.outcome}"` : ''}>${codeWords(floor.verdict?.outcome || floor.stage)}</span>`], ...(labelWords(floor.labels) ? [[t('Strategy'), html`<span class="owner-text">${labelWords(floor.labels)}</span>`]] : []), ...(floor.schedule?.entry_session ? [[t('Positions for'), floor.schedule.entry_session]] : []), [t('Holdings'), floor.holdings_count], [t('Task'), btnAttrs(mono(floor.update_task_id,SHORT.id),'task',floor.update_task_id,'text-btn',html`data-tip="${floor.update_task_id}"`)], [t(floor.stage === 'STANCES' ? 'Stances close' : 'Floor closes'), when(floor.stage === 'STANCES' ? floor.stances_close_at : floor.closes_at)]] : [];
    const floorActions = floor?.stage === 'CLOSED' ? link(t('Read the report'), 'report', 'text-btn', {review_selector: JSON.stringify(floor.review_selector)}) : '';
    const pending = stated.filter((e) => e.messageKind === 'objection' && !e.pmResponses.length), awaiting = pending.length;
    // T2: the awaiting objection is the head's way to it -- the Conversation, the first one read in place
    const marks = html`<span class="team-live" data-feed="${feed[0]}"><i class="live-dot" aria-hidden="true"></i><span class="state-word">${feed[1]}</span></span>${conversation && awaiting ? html`<span class="team-objection-status">${btn(stateLine('review_pending', {word: countText(awaiting, '{n} objection awaiting the Main PM', '{n} objections awaiting the Main PM'), next: ''}), 'team-reveal', pending[0].id, 'text-btn')}</span>` : ''}`;
    const view_ = html`<section class="team-scene" id="teamScene" data-view="${view}">${objectHead(conversation ? titleOf(conversation) : t(ROUTES[view]?.[1] || ROUTES.team[1]), html`<p class="lede">${t(view === 'team-committee' ? 'The committee’s recorded floor on these positions.' : LEDES[view] || LEDES.team)}${INFO[view] ? ' ' + t(INFO[view]) : ''}</p>`, floorActions, marks, [], {cls: 'team-head', headingId: 'teamSceneHeading', object: Boolean(conversation), facts: [...floorFacts, ...(conversation ? sessionFacts(chosen, stated, conversation, floor?.update_task_id) : [])], id: conversation ? chosen.id : '', scope: chosen && view !== 'team-sessions' ? {name: titleOf(questionOf(chosen, stated)), href: routeUrl('team', {team: chosen.id, actor: '', event: ''}), self: view === 'team'} : null})}
      ${state.error || state.notice || state.stale || ['RESET', 'UNAVAILABLE'].includes(state.disposition) ? noteLine(t('Activity visibility limited'), state.error || state.notice || t('Retained observations are not current host state.'), 'warning') : ''}
      ${readback}${usageFailure}${body}${view !== 'team-sessions' && X.selectedMore ? Lobby.older(t('Older records are not read yet'), 'team-older', '', Boolean(X.fetching)) : ''}${unknown.length ? noteLine(t('Events outside any declared session'), countText(unknown.length, '{n} external event without a native session id or with an unretained payload is listed in the activity feed, not here.', '{n} external events without a native session id or with an unretained payload are listed in the activity feed, not here.'), 'neutral') : ''}</section>`;
    // An arrival flashes once, on the view that shows it: the workroom consumes the exchanges'
    // flash, the evidence view the facts'; the other views leave both for their first sight.
    if (view === 'team-evidence') S.factFlash.clear(); else if (view !== 'team-sessions') S.flash.clear();
    return view_;
  }
  /* A deliberate choice: another session, or the newest ("Show the newest", an empty id) --
   * the one action that gives up the retained selection. */
  function select(sessionId) { S.visible = 50; if (!sessionId) S.retained = null; replaceHash({team: sessionId || '', actor: '', event: '', committee: '', committee_point: ''}); paint(); }
  /* The reader's choice among several PM-declared research cases, per session, in this tab. */
  function chooseQuestion(ref) { const s = selection().session || S.retained?.session || ''; if (!s) return; S.question.set(s, ref || ''); paint(); }
  // Route keys restore the retained selection; explicit caller keys override them.
  const routeContext = () => (S.retained ? {team: S.retained.session || '', actor: S.retained.actor || '', event: S.retained.event || '', committee: S.retained.committee || '', committee_point: S.retained.point || ''} : {});
  // Only an explicit exchange entry moves the reader, after the page's frame.
  function revealExchange(eventId) {
    const sid = selection().session;
    if (!sid) return;
    navigate(app.page === 'team-committee' ? app.page : 'team', {team: sid, actor: '', event: eventId});
    requestAnimationFrame(() => { const row = document.getElementById('team-event-' + eventId); if (row) { row.scrollIntoView({block: 'center', behavior: 'instant'}); row.focus({preventScroll: true}); } });
  }
  // Per member and model: latest usage, never summed or Goal/Task-attributed; roles stay declared and hook pins retain their source.
  function participantsView(s) {
    const all = entriesOf(s).filter((e) => !e.replayOf);
    const members = TeamMembers.rows(s, ROLE_SAME).sort((a, b) => (isLead(b, s) ? 1 : 0) - (isLead(a, s) ? 1 : 0) || a.last - b.last);
    if (!members.length) return emptyState(t('No member has taken part yet'));
    const rows = members.flatMap((pt) => {
      const {made, latest} = TeamMembers.activities(pt, all, ROLE_SAME, MESSAGE_KINDS), pins = pinWords(pt);
      const member = html`<span class="team-member-mark" data-role="${ROLE_TONES[pt.displayRole] || roleTone(s, pt.id)}">${icon(ROLE_ICONS[pt.displayRole] || roleIcon(s, pt.id))}</span> ${btnAttrs(pt.displayRole ? roleName(pt.displayRole) : actorName(s, pt.id), 'team-member-open', pt.id, 'text-btn')}${isLead(pt, s) ? html` <span class="sub-cell">${t('Lead')}</span>` : (pt.keyed || pt.specialistVerified) ? html`<span class="sub-cell" data-specialist-verified="${pt.keyed || pt.specialistVerified}">${t(pt.keyed || pt.specialistVerified === 'true' ? 'Specialist identity verified' : 'Specialist identity not verified')}</span>` : ''}`;
      const exchanges = html`${made || t('No exchange retained')}${latest ? html`<span class="sub-cell">${t('latest')} ${clock(latest.at)}</span>` : ''}${s.goalScope && !pt.messages.length && !pt.hooks.length && pt.usage.size ? html`<span class="sub-cell">${t('Session-wide usage only')}</span>` : ''}`;
      const usage = pt.sharedUsage ? [] : [...pt.usage.values()].sort((a, b) => a.model.localeCompare(b.model));
      if (!usage.length) return [tr([member, exchanges, html`<span class="muted">${pt.sharedUsage ? t('Usage shown with this thread’s first role') : pins ? hint(t('no usage read'), pins) : t('no usage read')}</span>`, '', '', '', ''])];
      return usage.map((u, i) => {
        const differs = u.pinDiffers.length ? html` <span class="team-differs" data-tip="${t('differs from its card: {what}', {what: u.pinDiffers.map((x) => codeWords(x)).join(', ')})}">${icon('warning')}</span>` : '';
        const model = html`<span class="mono">${u.model}</span>`;
        return tr([i ? '' : member, i ? '' : exchanges, html`${pins ? hint(model, pins) : model}${u.efforts.length ? html`<span class="sub-cell">${u.efforts.join(', ')}</span>` : ''}${differs}<span class="sub-cell">${usageMetadata(u)}</span>`, usageCount(u.input), usageCount(u.output), usageCount(u.cacheRead), usageCount(u.cacheWrite)]);
      });
    });
    const {shown, page, pages} = pageOf(rows, S.participantsPage);
    return html`${TeamMembers.othersLine(s)}${table([{label: t('Agent'), type: 'text'}, {label: t('Activities'), type: 'text', absorb: true}, {label: t('Model'), type: 'text'}, {label: hint(t('Input'), t('All usage values are tokens. Input excludes cache reads and cache writes.')), type: 'num'}, {label: t('Output'), type: 'num'}, {label: t('Cache read'), type: 'num'}, {label: t('Cache written'), type: 'num'}], shown, '', {report: true, countLine: false, classes: 'compact team-participants'})}${pager({page, pages, prev: ['team-participants-page', 'prev'], next: ['team-participants-page', 'next']})}`;
  }
  function openMember(actorId) { navigate('team', {team: selection().session || '', actor: actorId || '', event: ''}); }
  const turnParticipants = (way) => { S.participantsPage = Math.max(0, (S.participantsPage || 0) + (way === 'next' ? 1 : -1)); paint(); };
  // Outputs use session-filtered owner pages once per visit; accepted answers bind sealed receipts, and source Tasks remain citations.
  const OUTPUT_READS = {tasks: ['/api/tasks', 'tasks', 'refusals'], goals: ['/api/goals', 'goals', 'refused']};
  async function readOutputs(sid, which, cursor = '') {
    if (S.outputs?.session === sid && (S.outputs[which]?.busy || (cursor && S.outputs[which]?.next !== cursor))) return;
    if (S.outputs?.session !== sid) S.outputs = {session: sid, tasks: null, goals: null};
    const was = cursor ? S.outputs[which] : null, [path, key, refusalKey] = OUTPUT_READS[which];
    const mine = S.outputs[which] = {rows: was ? was.rows : [], refusals: was ? was.refusals || [] : [], next: was ? was.next : null, page: was ? was.page : 0, busy: true, error: ''};
    try {
      const b = await Data.read(path + '?' + new URLSearchParams({agent_session: sid, history_limit: LIST_PAGE, ...(cursor ? {history_cursor: cursor} : {})}));
      if (S.outputs?.[which] === mine) {
        const consumed = new Set([...(was?.consumed || []), ...(cursor ? [cursor] : [])]);
        const identity = r => r.task_id || r.goal_id || JSON.stringify(r);
        S.outputs[which] = {...mine, rows: Data.uniqueRows([...mine.rows, ...(b[key] || []).filter(r=>r.status!=='REFUSED')], identity), refusals: Data.uniqueRows([...mine.refusals, ...(b[refusalKey] || []), ...(b[key] || []).filter(r=>r.status==='REFUSED')], identity), next: consumed.has(b.next_cursor) ? null : b.next_cursor || null, consumed: [...consumed], busy: false};
      }
    } catch (e) { if (S.outputs?.[which] === mine) S.outputs[which] = {...mine, busy: false, error: e.message}; }
    paint();
  }
  const olderOf = (which, o, words) => (o.next ? Lobby.older(t(words), 'team-outputs-older', which, o.busy) : '');
  const outputRefusals = (o, which) => (o.refusals || []).map(r=>refusal(r,TONE.attention,{catalog:true,more:html`<p>${t(which==='tasks' ? 'Task' : 'Goal')} ${hashCell(r.task_id || r.goal_id,SHORT.id)}</p>`,next:prerequisiteWays(r.next_requests)}));
  const author = (e) => (reviewed(e) ? e.recipient : e.actor);
  function answersBox(s) {
    const answers = acceptedArtifacts(s);
    if (!answers.length) return '';
    if (S.answers?.session !== s.id) S.answers = {session: s.id, page: 0};
    const rows = answers.map((e) => tr([link(t('Accepted answer'), 'team', 'text-btn', {team: s.id, actor: '', event: e.id}), html`<span tabindex="0" data-tip="${author(e)}">${recipientName(s, author(e))}</span>`, html`<span tabindex="0" data-tip="${e.subject.submitted_by || author(e)}">${recipientName(s, e.subject.submitted_by || author(e))}</span>`, hashCell(e.subject.answer_reference, SHORT.hash), hashCell(e.subject.bundle_reference, SHORT.hash), hint(clock(e.at), timeKindWords(e))]));
    const {shown, page, pages} = pageOf(rows, S.answers.page);
    return panel(t('Accepted answers'), '', html`${table([{label: t('Answer'), type: 'text', absorb: true}, {label: t('Agent'), type: 'text'}, {label: t('Submitted by'), type: 'text'}, {label: t('Answer reference'), type: 'id'}, {label: t('Bundle reference'), type: 'id'}, {label: t('Time'), type: 'date'}], shown, '', {report: true, countLine: false, classes: 'compact team-outputs'})}${pager({page, pages, prev: ['team-outputs-page', 'answers:prev'], next: ['team-outputs-page', 'answers:next']})}`, '', 'data-box="table"');
  }
  function tasksBox(o) {
    const refused = outputRefusals(o, 'tasks');
    if (!o.rows.length) return panel(t('Tasks'), '', refused.length ? html`${refused}${olderOf('tasks',o,'Older Tasks of this session are not read yet')}` : o.busy ? skeleton('rows') : emptyState(t('No Task submitted by this session')));
    const goals = o.rows.some((r) => r.submitted_by?.goal_id);
    const rows = o.rows.map((r) => {
      const a = r.artifact;
      // an artifact only once the Task succeeded and its kind publishes one; unavailable says the owner's code
      const artifact = !a ? '' : a.availability === 'AVAILABLE' ? html`${codeWords(a.artifact_kind)}<span class="sub-cell">${hashCell(a.artifact_hash, SHORT.hash)}</span>` : html`${t('Unavailable')}<span class="sub-cell">${coded(a.failure_code)}</span>`;
      const goal = r.submitted_by?.goal_id ? link(html`${t('Goal')} ${mono(r.submitted_by.goal_id, SHORT.id)}`, 'goal', 'text-btn', {goal: r.submitted_by.goal_id}) : '';
      return tr([html`${btnAttrs(r.goal_summary || codeWords(r.task_kind), 'task', r.task_id, 'text-btn')}<span class="sub-cell">${codeWords(r.task_kind)} · ${mono(r.task_id, SHORT.id)}</span>`, stateLine(r.final_state || r.lifecycle), artifact, ...(goals ? [goal] : []), when(r.last_activity_at)]);
    });
    const {shown, page, pages} = pageOf(rows, o.page);
    return html`${refused}${panel(t('Tasks'), '', html`${table([{label: t('Task'), type: 'text', absorb: true}, {label: t('State'), type: 'status'}, {label: t('Artifact'), type: 'text'}, ...(goals ? [{label: t('Goal'), type: 'id'}] : []), {label: t('Last activity'), type: 'date'}], shown, '', {report: true, countLine: false, classes: 'compact team-outputs'})}${pager({page, pages, prev: ['team-outputs-page', 'tasks:prev'], next: ['team-outputs-page', 'tasks:next']})}${olderOf('tasks', o, 'Older Tasks of this session are not read yet')}`, '', 'data-box="table"')}`;
  }
  /* The goals the session took (the user's rule, 2026-09-30): each with the Host's check -- Complete once it found the
   * record complete -- and the deliverables its submission names, never mixed with the Tasks' artifacts. */
  function goalsBox(o) {
    const refused = outputRefusals(o, 'goals');
    if (!o.rows.length) return panel(t('Goals'), '', refused.length ? html`${refused}${olderOf('goals',o,'Older goals of this session are not read yet')}` : o.busy ? skeleton('rows') : emptyState(t('No goal names this session')));
    const rows = o.rows.map((g) => {
      const delivered = (g.deliverables || []).map((d) => html`<span class="owner-text">${d.description || d.deliverable_id}</span><span class="sub-cell">${codeWords(d.kind)} · ${countText(d.reference_count, '{n} reference', '{n} references')}</span>`);
      return tr([html`${link(html`<span class="owner-text">${g.title}</span>`, 'goal', 'text-btn', {goal: g.goal_id})}${g.outcome ? html`<span class="sub-cell">${LiveGoals.outcomeWord(g.outcome)}</span>` : ''}`, LiveGoals.checkLine(g), html`${delivered}`, g.completion?.checked_at ? when(g.completion.checked_at) : '']);
    });
    const {shown, page, pages} = pageOf(rows, o.page);
    return html`${refused}${panel(t('Goals'), t('The Host checks a goal\'s record, not whether its objective was met; a goal holds deliverables once its record is complete.'), html`${table([{label: t('Goal'), type: 'text', absorb: true}, {label: t('Check'), type: 'status'}, {label: t('Deliverables'), type: 'text'}, {label: t('Checked'), type: 'date'}], shown, '', {report: true, countLine: false, classes: 'compact team-output-goals'})}${pager({page, pages, prev: ['team-outputs-page', 'goals:prev'], next: ['team-outputs-page', 'goals:next']})}${olderOf('goals', o, 'Older goals of this session are not read yet')}`, '', 'data-box="table"')}`;
  }
  function outputsView(s) {
    if (S.outputs?.session !== s.id) { void readOutputs(s.id, 'tasks'); void readOutputs(s.id, 'goals'); }
    const {tasks, goals} = S.outputs, answers = answersBox(s), errors = [tasks.error, goals.error].filter(Boolean);
    const failed = errors.length ? notRead(t('Outputs not read'), errors.join(' · '), '', btn(t('Read again'), 'team-outputs-read', '', 'button compact')) : '';
    if (!tasks.rows.length && !goals.rows.length && !(tasks.refusals || []).length && !(goals.refusals || []).length) return answers ? html`${failed}${answers}${tasks.busy || goals.busy ? skeleton('rows') : ''}` : tasks.busy || goals.busy ? skeleton('rows') : html`${failed}${errors.length ? '' : html`${emptyState(html`${t('This session submitted no Task and took no goal')}${infoMark(t('The Host returned no Task submitted by this session and no Goal bound to it. A written answer file or a cited reference does not create either record.'))}`, link(t('Conversation'), 'team', 'text-btn', {team: s.id, actor: '', event: ''}), '', 'elsewhere')}${s.references.size ? html`<p class="caption">${countText(s.references.size, 'Its members cited {n} reference; a citation is not a submission by this session.', 'Its members cited {n} references; a citation is not a submission by this session.')}</p>` : ''}`}`;
    return html`${failed}${answers}${tasksBox(tasks)}${goalsBox(goals)}`;
  }
  const turnOutputs = (value) => { const [which, way] = String(value).split(':'), o = which === 'answers' ? S.answers : S.outputs?.[which]; if (o) o.page = Math.max(0, (o.page || 0) + (way === 'next' ? 1 : -1)); paint(); };
  const outputsOlder = (which) => { const o = S.outputs?.[which]; if (o?.next && !o.busy) void readOutputs(S.outputs.session, which, o.next); };
  const outputsRead = () => { S.outputs = null; paint(); };
  const leaveOutputs = () => { S.outputs = null; };
  function showActor(actorId) { S.visible = 50; replaceHash({actor: actorId || '', event: ''}); paint(); }
  /* An exchange pressed opens its verification in place (C4 item 7); pressed again, it closes. */
  function showEvent(eventId) {
    const sessions = scene().sessions, sid = selection().session;
    const s = sid ? sessions.find((v) => v.id === sid) || (selection().committee ? sessionState(sid) : null) : sessions[0];
    if (!selectedAnswerRecord(s, eventId)) return;
    S.incoming.delete(eventId); replaceHash({team: s.id, event: selection().event === eventId ? '' : eventId}); paint();
  }
  /* Instant, not the page's smooth scroll: a repaint's own scroll restoration would cancel the
   * animation midway and leave the heading wherever it was at that moment. Then the actual
   * sticky header is measured: where the document's scroll padding is shorter than the header
   * (the narrow layout), the heading is moved out from under it -- for an explicit entry only,
   * by what is covered, never by a fixed number. Focus follows without a second scroll. */
  function reveal(heading) {
    heading.scrollIntoView({block: 'start', behavior: 'instant'});
    const top = document.querySelector('#top');
    const covered = (top && getComputedStyle(top).position === 'sticky' ? top.getBoundingClientRect().bottom : 0) + (parseFloat(getComputedStyle(heading).scrollMarginTop) || 0) - heading.getBoundingClientRect().top;
    if (covered > 0) scrollBy({top: -covered, behavior: 'instant'});
    heading.focus({preventScroll: true});
  }
  function more() { S.visible += 50; paint(); }
  /* A comment's earlier replies, shown or folded again (N5). */
  function toggleReplies(id) { if (S.replies.has(id)) S.replies.delete(id); else S.replies.add(id); paint(); }
  function toggleWords(id) { if (S.words.has(id)) S.words.delete(id); else S.words.add(id); paint(); }
  function markSeen() { const s = scene().sessions.find((v) => v.id === selection().session) || (!selection().session ? scene().sessions[0] : null); if(s) { for(const id of s.entries.keys()) S.incoming.delete(id); for (const f of s.facts) { S.factNew.delete(f.item.observation_id); S.factVerified.delete(f.item.observation_id); } } paint(); }
  /* Arrivals: a continuous page's own new rows -- past the retained watermark, on the same
   * store, never a rebuilt, reconnect or replayed page (the feed decides `continuous`). The
   * store's own commit order is the arrival test; the browser's clock is not compared with the
   * observation stamps, which follow the workspace's research clock. A new message is unread
   * and flashes once; a Task completion (a SUCCEEDED Task Control transition) is marked new;
   * an owner-verified result (the compatible artifact verification) is marked new and flashes
   * once -- the one arrival that reads as verified. Neither moves the reader's selection,
   * scroll or focus. */
  function arrivals(body, continuous) {
    S.flash.clear(); S.factFlash.clear();
    if (!continuous) return;
    const fresh = (e) => e.ordinal > LiveActivity.state().watermark;
    for (const e of body.items || []) {
      if (!fresh(e)) continue;
      if (e.schema_kind === 'ExternalActivityObserved' && KINDS[e.payload?.event_kind]) { if (!S.incoming.has(e.observation_id)) { S.incoming.add(e.observation_id); S.flash.add(e.observation_id); } }
      else if (artifactVerified(e)) { if (!S.factNew.has(e.observation_id)) { S.factNew.add(e.observation_id); S.factVerified.add(e.observation_id); S.factFlash.add(e.observation_id); } }
      else if (e.schema_kind === 'TaskControlTransition' && e.payload?.task_lifecycle === 'SUCCEEDED') S.factNew.add(e.observation_id);
    }
    while(S.incoming.size > 200) S.incoming.delete(S.incoming.values().next().value);
    while(S.factNew.size > 100) { const id = S.factNew.values().next().value; S.factNew.delete(id); S.factVerified.delete(id); }
  }
  function refresh() {
    if (!onTeam()) { S.answerDetail = null; S.usageAsked = false; if (app.page !== 'overview') return; }
    if (onTeam() && !S.usageAsked) { S.usageAsked = true; LiveActivity.readSessionUsage?.(); }
    if (onTeam() && app.page !== 'team-sessions') { const s = scene().sessions.find(s => s.id === selection().session); if (s && committeeTask(s)) { const o = S.committee, deadline = o?.body?.stage === 'STANCES' ? o.body.stances_close_at : o?.body?.stage === 'DEBATE' ? o.body.closes_at : ''; const crossed = deadline && Date.now() >= Date.parse(deadline) && o.deadlineRead !== deadline; if (crossed) o.deadlineRead = deadline; void readCommittee(s, crossed); } }
    for (const s of scene().sessions) {
      const q = questionOf(s, entriesOf(s)), refs = q.ref ? q.declared : UUID.test(q.scope || '') ? ['goal:' + q.scope] : q.declared;
      for (const ref of refs) if (!S.resolved.has(ref) && !S.busy.has(ref)) void resolve(ref).then(() => verify(ref));
    }
    const a = LiveActivity.state(), usage = LiveActivity.nativeUsageState?.();
    const key = [a.epoch, a.cursor, a.watermark, a.error, a.stale, a.disposition, usage?.status, usage?.reason, usage?.read_through, usage?.total_bytes, usage?.rescan].join('|');
    if (key !== S.paintKey) { S.paintKey = key; paint(); }
  }
  /* The explicit entry from an activity row, wherever that row lives: the Task Center drawer
   * closes without handing focus back to the button it was opened from, the Research Team page
   * routes to the session, and the scene's heading is brought into view and focused once the
   * page's own place memory has had its frame. Only this click moves the reader; arriving
   * events repaint in place and never scroll or refocus. */
  function open(sessionId) {
    const session = sessionId || '';
    const settle = () => {
      if (!onTeam() || (hashParams().get('team') || '') !== session) return; // the reader moved on
      const heading = document.getElementById('teamSceneHeading'); // an id: the page's surface memory keeps focus here across repaints
      if (heading) reveal(heading);
    };
    if (typeof LiveTasks !== 'undefined') LiveTasks.close();
    navigate('team', {team: session, actor: '', event: ''}); // explicit: never another session's retained exchange
    requestAnimationFrame(settle);
  }
  /* The page repaints in place: a scene whose markup did not change is left alone, and one
   * that did keeps its thread scroll, open disclosures and focus (the router's surface memory). */
  const paint = () => { if (onTeam()) patchMain(); };
  /* The retained scene in a few facts for the Overview: the question as the lead block says it,
   * the latest contribution, the unresolved objections. Derived, never read. */
  function summary() {
    const sessions = scene().sessions, sel = S.retained?.session || '';
    if (!sessions.length) return {sessions: 0};
    const s = sel ? sessions.find((v) => v.id === sel) || null : sessions[0];
    if (!s) return {sessions: sessions.length, missing: sel}; // the retained selection is not in the window: said, not replaced
    const all = [...s.entries.values()].sort((a, b) => a.ordinal - b.ordinal).filter((e) => !e.replayOf);
    linkExchanges(s, all);
    const latest = all.filter((e) => e.kind === 'message').at(-1) || null;
    const objections = all.filter((e) => e.messageKind === 'objection');
    return {sessions: sessions.length, session: s.id, selected: Boolean(sel), participants: participantCount(s), unread: [...s.entries.keys()].filter((id) => S.incoming.has(id)).length,
      question: questionOf(s, all),
      latest: latest ? {actor: actorName(s, latest.actor), kind: kindWords(latest), text: latest.text, at: latest.at, id: latest.id} : null,
      objections: objections.length, unresolved: objections.filter((e) => !e.pmResponses.length).length};
  }
  /* The retained sessions whose declared references name an exact object (a Task id, a result
   * hash or a History entry naming it): the object's own way to its collaboration. Several
   * matches are several, none is none; nothing is picked by recency. */
  function sessionsNaming(...refs) {
    const wanted = new Set(refs.filter(r => typeof r === 'string' && r).map(r => r.toLowerCase())), context = refs.find(r => r?.native_session_id && r.update_task_id && r.review_selector?.update_task_id === r.update_task_id);
    if (!wanted.size) return [];
    const named = scene().sessions.filter((s) => [...s.references].some((r) => wanted.has(r) || wanted.has(r.split(':').at(-1)))).map((s) => { const all = [...s.entries.values()].sort((a, b) => a.ordinal - b.ordinal).filter((e) => !e.replayOf); linkExchanges(s, all); return {id: s.id, entries: s.entries.size, question: questionOf(s, all).text || '', roles: rolesOf(s)}; });
    if (context && [context.update_task_id, context.review_selector.update_publication_hash].some(ref => wanted.has(ref))) {
      const held = named.find(s => s.id === context.native_session_id);
      if (held) held.committee = context;
      else named.push({id: context.native_session_id, roles: context.specialists, committee: context});
    }
    return named;
  }
  /* The session an activity row belongs to, for the row's own "Team scene" read. */
  const sessionOf = (item) => (item.schema_kind === 'ExternalActivityObserved' && KINDS[item.payload?.event_kind] ? item.payload.subject?.native_session_id || null : null);
  function counts() {
    const s = scene().sessions.find((v) => v.id === selection().session);
    // U54's 产出 carries no count: known only once both its reads answer, it would appear on one tab and move the others
    return s ? {exchanges: entriesOf(s).filter((e) => !e.replayOf && e.channel !== 'PRODUCT_COMMITTEE').length, participants: participantCount(s), observations: s.facts.length, more: X.selectedMore} : {};
  }
  return {pages: PAGES_SET, hasCommittee, readCommittee, chooseCommitteePoint, committeeRead: () => { const sel = selection(), s = scene().sessions.find(s => s.id === sel.session) || (sel.session && sel.committee ? sessionState(sel.session) : null); if (s) void readCommittee(s, true); }, usageCount, usageTokenWords, usageMetadata, sessionLabel, roleName, recipientName, section, scene, select, counts, openMember, turnParticipants, turnOutputs, outputsOlder, outputsRead, leaveOutputs, setFactKind, readOlder, toggleWords, showActor, showEvent, revealExchange, more, toggleReplies, toggleFold, markSeen, arrivals, refresh, open, resolve, verify, sessionOf, qualify, productReferences, summary, recordOf, sessionsNaming, questionSource, chooseQuestion, routeContext, runs, retained: () => S.retained, resolved: () => S.resolved};
})();
for (const page of LiveTeam.pages) PAGES[page] = LiveTeam.section;
