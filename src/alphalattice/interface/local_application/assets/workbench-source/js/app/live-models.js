/* The Models page (U50; EX, the model point): every Alpha model this workspace may fit -- the installed ones and
 * the ones an agent declared, checked and sandboxed -- each with its review packet: the contract's checks by name,
 * the identity it adds (none moved), the environment its fits record, its sandbox trial and its activation. A
 * person activates an agent's model for this workspace once a sandbox trial of its identity passed, or deactivates
 * it, by the requests the Host offers; the Host refuses an agent's. Checking the contract runs each model's fits,
 * so the packets are read once a visit, never on a repaint. */
const LiveModels = (() => {
  const pages = new Set(['models']);
  const S = {body: null, error: '', loading: false, ticket: 0, page: 0, pending: null, busy: false, notice: '', refused: ''};
  const addressed = () => hashParams().get('model') || '';
  const STATE = {INSTALLED: ['verified', 'Installed'], ACTIVE: ['succeeded', 'model|Active'], NOT_ACTIVE: ['metadata', 'Not active']};
  const LEDES = {INSTALLED: 'Installed with the product: every Alpha study may fit it.', ACTIVE: 'Added by an agent and active in this workspace: its Alpha studies may fit it.', NOT_ACTIVE: 'Added by an agent; not active in this workspace.'};
  const CHECKS = {route: 'model|Route', search_axes: 'Search axes', fit_protocol: 'Fit protocol', determinism: 'Determinism', prediction_rows: 'Prediction rows', state: 'Fitted state', imports: 'Imports'}; // the contract's checks, in its order
  const nameOf = (m) => labelOf(m.model_id) || METHOD_WORDS[m.model_id] ? methodWords(m.model_id) : html`<span class="mono">${m.model_id}</span>`;
  const failed = (m) => (m.contract?.findings || []).find((f) => f.code);
  /* The packets, once a visit: `leave` forgets them, so the next visit reads the Host again. */
  async function ensure() {
    if (!pages.has(app.page) || S.body || S.loading || S.error) return;
    const ticket = ++S.ticket; S.loading = true;
    try { const b = await Data.read('/api/models'); if (ticket === S.ticket) S.body = b; }
    catch (e) { if (ticket === S.ticket) S.error = e.message; }
    finally { if (ticket === S.ticket) { S.loading = false; render(); } }
  }
  function leave() { S.ticket++; S.body = null; S.error = ''; S.loading = false; S.notice = ''; S.refused = ''; S.pending = null; }
  function refresh() { leave(); render(); void ensure(); }
  /* Only what the Host offers: its `next_requests` name the activation or the deactivation. An agent's model without
   * one says why from the packet's own facts -- a failed contract, or no passed sandbox trial of this identity. */
  function act(m, cls = 'button compact') {
    const next = m.next_requests || {}, busy = S.busy ? 'Waiting for the product owner' : '';
    if (next.deactivate) return typedBtn(t('Deactivate'), 'model-confirm', 'deactivate:' + m.model_id, cls, busy);
    if (next.activate) return typedBtn(t('Activate'), 'model-confirm', 'activate:' + m.model_id, cls, busy);
    if (m.state === 'NOT_ACTIVE') return typedBtn(t('Activate'), 'model-confirm', 'activate:' + m.model_id, cls, failed(m) ? 'The model fails its contract' : 'No passed sandbox trial of this identity');
    return '';
  }
  const contractWords = (m) => { const f = failed(m); return f ? t('Contract fails: {check}', {check: t(CHECKS[f.check] || f.check)}) : t('Contract passed'); };
  const trialWords = (m) => m.state === 'INSTALLED' ? '' : m.sandbox ? t('Sandbox trial {when}', {when: when(m.sandbox.recorded_at)}) : t('No sandbox trial');
  const row = (m) => {
    const to = {action: 'model-open', value: m.model_id};
    if (m.status === 'REFUSED') return objectRow({state: 'refused', name: nameOf(m), to}, {key: m.model_id, props: [locatorCell(m.model_id), codedSubject(m.failure_code)]});
    const [state, word] = STATE[m.state] || ['metadata', m.state];
    return objectRow({state, name: nameOf(m), to}, {key: m.model_id, word: t(word), props: [contractWords(m), trialWords(m)], actions: act(m, 'menu-row')});
  };
  function refusedPacket(m) {
    const command = m.next_commands?.check;
    return refusal(m, 'warning', {
      catalog: true,
      next: prerequisiteWays(m.next_requests),
      more: html`${command ? codeRef(t('Model check command'), command, 'text') : ''}${codeRef(t('Exact review packet (JSON)'), m)}`,
    });
  }
  function page() {
    if (!pages.has(app.page)) return null;
    const id = addressed();
    const top = html`${S.error ? notRead(t('Models not read'), S.error, explain(String(S.error).split(':')[0]), btn(t('Read again'), 'models-refresh', '', 'button compact')) : ''}${S.refused ? notRead(t('Model action refused'), S.refused, explainCode(S.refused)) : ''}${S.notice ? noteLine(t('Last operation'), S.notice) : ''}`;
    if (!id) {
      const head = objectHead(t('Models'), t('Every Alpha model this workspace may fit: the installed ones and those an agent added. An agent declares, checks and sandboxes a model; a person activates it for this workspace. Reading this page fits nothing.'), '');
      if (!S.body) return html`${head}${top}${S.error ? '' : skeleton('rows')}`;
      const all = S.body.models || [], {shown, page: at, pages: n} = pageOf(all, S.page);
      return html`${head}${top}${all.length ? html`<div class="card-list lines slotted">${shown.map(row)}</div>${pager({page: at, pages: n, prev: ['models-page', 'prev'], next: ['models-page', 'next']})}` : emptyState(t('No Alpha model is installed.'))}`;
    }
    if (!S.body) return html`${top}${S.error ? '' : skeleton('head')}`;
    const m = (S.body.models || []).find((x) => x.model_id === id);
    if (!m) return html`${top}${objectHead(html`<span class="mono">${id}</span>`, '', '')}${emptyState(t('This model is not in the list now.'), btn(t('All models'), 'models-list', '', 'button'))}`;
    if (m.status === 'REFUSED') return html`${top}${objectHead(nameOf(m), '', '', '', [], {object: true, id: m.model_id})}${refusedPacket(m)}`;
    const [state, word] = STATE[m.state] || ['metadata', m.state];
    return html`${top}${objectHead(nameOf(m), html`<p class="lede">${t(LEDES[m.state] || '')}</p>`, act(m, 'button'), stateLine(state, {word: t(word), next: ''}), [], {object: true, id: m.model_id})}${contract(m)}${identity(m)}${trial(m)}${activation(m)}${environment(m)}${codeRef(t('Exact review packet (JSON)'), m)}`;
  }
  /* The contract: each check by name, in its order; a failed one names its code. */
  function contract(m) {
    const all = m.contract?.findings || [], failing = all.some((f) => f.code); // the Code column only where a check failed
    const rows = all.map((f, i) => tr([count(i + 1), t(CHECKS[f.check] || f.check), stateLine(f.code ? 'failed' : 'succeeded', {word: t(f.code ? 'fails' : 'passes'), next: ''}), ...(failing ? [f.code ? codedSubject(f.code) : ''] : [])]));
    return panel(t('Contract'), t('Each check runs the model\'s own fits on a probe panel; a failed check names its code.'), table([{label: '#', type: 'num', index: true}, {label: t('Check'), type: 'text'}, {label: t('Result'), type: 'status', ...(failing ? {cls: 'col-tight'} : {absorb: true})}, ...(failing ? [{label: t('Code'), type: 'text', absorb: true}] : [])], rows, '', {report: true, countLine: false, classes: 'compact model-contract'}), '', 'data-box="table"');
  }
  function identity(m) {
    const id = m.identity || {}, c = m.contract || {};
    return panel(t('Identity'), t('An added model is its own closure: it adds its identity and moves none.'), kv([[t('Numerical binding'), hashCell(id.numerical_binding_hash)], [t('Declaration'), hashCell(c.declaration_hash)], [t('Contract receipt'), hashCell(c.contract_receipt_hash)],
      [t('Adds'), id.adds ? codeWords(id.adds) : t('nothing: installed with the product')], [t('Moves'), (id.moves || []).length ? id.moves.map((v) => html`<span class="mono">${v}</span>`).reduce((a, x, i) => html`${a}${i ? ' · ' : ''}${x}`, '') : t('no existing identity')]], 'kv-columns'));
  }
  /* The sandbox trial: one Alpha study on a copy of the workspace, read back, and U0 on the copy -- the copy's
   * study, so its Task is named, never opened here. */
  function trial(m) {
    if (m.state === 'INSTALLED') return '';
    const s = m.sandbox;
    if (!s) return panel(t('Sandbox trial'), '', html`<p class="caption">${t('No sandbox trial of this identity yet. An agent runs one on a copy of the workspace, with the Host stopped:')}</p><pre class="code-block">model sandbox ${m.model_id}</pre>`);
    return panel(t('Sandbox trial'), t('One Alpha study on a copy of the workspace, read back, and U0 on the copy.'), kv([[t('Study on the copy'), html`<span class="mono">${short(s.study_task_id)}</span>`], [t('Run time'), durationText(Number(s.study_seconds) * 1000)], [t('Peak memory'), bytesWords(s.peak_memory_bytes)],
      [t('U0'), t('{r} saved objects read back · {c} changed', {r: count(s.u0_reads), c: count(s.u0_changed)})], [t('Recorded'), when(s.recorded_at)]], 'kv-columns'));
  }
  function activation(m) {
    if (m.state === 'INSTALLED') return '';
    const a = m.activation;
    return panel(t('Activation'), '', a ? kv([[t('Activated by'), t('a person')], [t('Activated'), when(a.activated_at)], [t('Contract receipt'), hashCell(a.contract_receipt_hash)], [t('Sandbox study'), html`<span class="mono">${short(a.sandbox?.study_task_id)}</span>`]], 'kv-columns') : html`<p class="caption">${t('Not active in this workspace: its Alpha studies cannot fit it.')}</p>`);
  }
  function environment(m) {
    const e = m.environment || {};
    return panel(t('Environment'), t('Recorded beside each fit as provenance; it is not the model\'s identity.'), kv([[t('Packages'), (e.package_versions || []).map(([n, v]) => html`<span class="mono">${n} ${v}</span>`).reduce((a, x, i) => html`${a}${i ? ' · ' : ''}${x}`, '') || t('none recorded')],
      [t('Threads'), e.configured_thread_count == null ? t('not pinned') : count(e.configured_thread_count)], [t('Environment'), hashCell(e.environment_hash)]], 'kv-columns'));
  }
  /* A person's activation or deactivation, confirmed with the facts it records. */
  function confirm(value) {
    const at = value.indexOf(':'), kind = value.slice(0, at), id = value.slice(at + 1);
    const m = (S.body?.models || []).find((x) => x.model_id === id), request = m?.next_requests?.[kind];
    if (!m || !request || !Data.offers(request.operation) || S.busy) return; // U13: the route the session names
    S.pending = {kind, name: nameOf(m), path: Data.route(request.operation), payload: {model_id: request.model_id}};
    const facts = [[t('Model'), nameOf(m)], [t('Numerical binding'), hashCell(m.contract?.numerical_binding_hash)], [t('Contract receipt'), hashCell(m.contract?.contract_receipt_hash)], ...(m.sandbox ? [[t('Sandbox study'), html`<span class="mono">${short(m.sandbox.study_task_id)}</span> · ${when(m.sandbox.recorded_at)}`]] : [])];
    const activate = kind === 'activate';
    openDialog(t('Models · explicit confirmation'), t(activate ? 'Activate this model for this workspace?' : 'Deactivate this model?'), html`${kv(facts)}<p class="caption">${t(activate ? 'Its Alpha studies may then fit it, in this workspace alone. Activation records you and the time, the identity, its contract receipt and the sandbox study it passed; it edits no sealed record and fits nothing.' : 'It leaves this workspace\'s catalog, so a new study cannot fit it; a kept study that binds it still reads back.')}</p>`,
      html`${btn(t(activate ? 'Activate' : 'Deactivate'), 'model-commit', '', 'button primary', true)}`);
  }
  async function commit() {
    const p = S.pending;
    if (!p || S.busy) return;
    S.pending = null; S.busy = true; S.refused = ''; S.notice = ''; closeDialog(); render();
    try {
      const b = await Data.post(p.path, p.payload);
      S.notice = b.status === 'ACTIVATED' ? t('Activated for this workspace: {model}.', {model: p.name}) : t('Deactivated: {model}. Kept studies that bind it still read back.', {model: p.name});
      S.ticket++; S.body = null;
    } catch (e) { S.refused = e.message; }
    finally { S.busy = false; render(); void ensure(); }
  }
  function open(id) { objectEntry('model:' + id); navigate('models', {model: id}, {replace: true}); }
  const turn = (way) => { S.page = Math.max(0, (S.page || 0) + (way === 'next' ? 1 : -1)); render(); };
  return {pages, ensure, leave, refresh, page, open, list: () => navigate('models', {model: ''}), confirm, commit, turn};
})();
