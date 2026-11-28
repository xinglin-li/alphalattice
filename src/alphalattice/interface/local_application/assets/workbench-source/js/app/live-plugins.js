/* The Plugins page: each installed capability as its manifest declares it -- its effect, its state and its
 * parameters -- and a run of one. The run form is the capability's input schema read as fields: a string, integer,
 * number or boolean parameter, or an array of them, is a field of its own; any other parameter goes in one JSON
 * field. A run answers its receipt, or the Host's refusal by name. The page names no capability: the Host's list
 * is the page, read once a visit. */
const LivePlugins = (() => {
  const pages = new Set(['plugins']);
  const S = {body: null, error: '', loading: false, ticket: 0, busy: false, receipt: null, refused: ''};
  const SCALARS = new Set(['string', 'integer', 'number', 'boolean']);
  const STATE = {DISCOVERED: 'metadata', VERIFIED: 'verified', ACTIVE: 'succeeded', EVICTED: 'metadata'};
  const addressed = () => hashParams().get('capability') || '';
  const nameOf = (p) => html`<span class="mono">${p.capability}</span>`;
  async function ensure() {
    if (!pages.has(app.page) || S.body || S.loading || S.error) return;
    const ticket = ++S.ticket; S.loading = true;
    try { const b = await Data.read('/api/plugins'); if (ticket === S.ticket) S.body = b; }
    catch (e) { if (ticket === S.ticket) S.error = e.message; }
    finally { if (ticket === S.ticket) { S.loading = false; render(); } }
  }
  function leave() { S.ticket++; S.body = null; S.error = ''; S.loading = false; S.receipt = null; S.refused = ''; }
  function refresh() { leave(); render(); void ensure(); }
  /* Each parameter with the field it takes: its own for a scalar or an array of scalars, else the JSON field. */
  const kindOf = (spec) => spec.type === 'array' && SCALARS.has(spec.items?.type) ? 'array' : SCALARS.has(spec.type) ? spec.type : '';
  const params = (p) => Object.entries(p.input_schema?.properties || {}).map(([name, spec]) => ({name, spec, kind: kindOf(spec), required: (p.input_schema.required || []).includes(name)}));
  /* Each control carries the schema's names in quoted, escaped slots; a boolean is true, false or not given. */
  function control({name, spec, kind, required}) {
    const id = 'plugin-' + name, item = spec.items?.type || '', label = html`<span class="mono">${name}</span>${required ? ' *' : ''}`;
    const input = kind === 'boolean'
      ? html`<select id="${id}" class="ui-field" data-plugin-param="${name}" data-kind="${kind}" data-item="${item}"><option value=""></option><option value="true">true</option><option value="false">false</option></select>`
      : html`<input id="${id}" class="ui-field" type="${kind === 'integer' || kind === 'number' ? 'number' : 'text'}" data-plugin-param="${name}" data-kind="${kind}" data-item="${item}">`;
    const hint = kind === 'array' ? t('plugins.label.array_hint') : spec.description || '';
    return html`<div class="field"><label for="${id}">${label}</label>${input}${hint ? html`<small>${hint}</small>` : ''}</div>`;
  }
  const typed = (kind, text) => kind === 'integer' ? Number.parseInt(text, 10) : kind === 'number' ? Number(text) : kind === 'boolean' ? text === 'true' : text;
  /* The parameters as the form holds them: each field by its kind, an empty one not given; the JSON field's beside them. */
  function values() {
    const all = {...JSON.parse($('#plugin-params')?.value || '{}')};
    for (const el of $$('[data-plugin-param]')) {
      const {pluginParam: name, kind, item} = el.dataset, text = el.value.trim();
      if (!text) continue;
      all[name] = kind === 'array' ? text.split(',').map((v) => typed(item, v.trim())) : typed(kind, text);
    }
    return all;
  }
  /* The form is read before the page repaints, which replaces its controls. */
  async function run() {
    const p = (S.body?.capabilities || []).find((x) => x.capability === addressed());
    if (!p || S.busy) return;
    let payload;
    try { payload = {capability: p.capability, params: values()}; }
    catch (e) { S.refused = e.message; render(); return; }
    S.busy = true; S.refused = ''; S.receipt = null; render();
    try { S.receipt = (await Data.post('/api/plugins/run', payload)).receipt; }
    catch (e) { S.refused = e.message; }
    finally { S.busy = false; render(); }
  }
  const row = (p) => objectRow({state: STATE[p.state] || 'metadata', name: nameOf(p), to: {action: 'plugin-open', value: p.capability}}, {key: p.capability, word: codeWords(p.state), props: [codeWords(p.effect_mode), t(p.workspace ? 'Workspace' : 'Installed')]});
  function receipt(r) {
    return panel(t('Receipt'), '', html`${kv([[t('Status'), codeWords(r.status)], [t('plugins.label.idempotency_key'), hashCell(r.idempotency_key)], [t('Run time'), durationText(r.wall_ms)], [t('Output'), html`<pre class="code-block">${JSON.stringify(r.output, null, 2)}</pre>`]], 'kv-columns')}${codeRef(t('plugins.label.receipt_json'), r)}`);
  }
  function page() {
    if (!pages.has(app.page)) return null;
    const id = addressed();
    const top = html`${S.error ? notRead(t('plugins.label.not_read'), S.error, explain(String(S.error).split(':')[0]), btn(t('Read again'), 'plugins-refresh', '', 'button compact')) : ''}${S.refused ? notRead(t('plugins.label.run_refused'), S.refused, explainCode(S.refused)) : ''}`;
    if (!id) {
      const head = objectHead(t('plugins.label.title'), t('plugins.label.lede'), '');
      if (!S.body) return html`${head}${top}${S.error ? '' : skeleton('rows')}`;
      const all = S.body.capabilities || [];
      return html`${head}${top}${all.length ? html`<div class="card-list lines slotted">${all.map(row)}</div>` : emptyState(t('plugins.label.none'))}`;
    }
    if (!S.body) return html`${top}${S.error ? '' : skeleton('head')}`;
    const p = (S.body.capabilities || []).find((x) => x.capability === id);
    if (!p) return html`${top}${objectHead(html`<span class="mono">${id}</span>`, '', '')}${emptyState(t('plugins.label.gone'), btn(t('plugins.label.all'), 'plugins-list', '', 'button'))}`;
    const fields = params(p), own = fields.filter((f) => f.kind), others = fields.filter((f) => !f.kind);
    const json = others.length ? html`<div class="field"><label for="plugin-params">${t('plugins.label.other_params')}: <span class="mono">${others.map((f) => f.name).join(', ')}</span></label><textarea id="plugin-params" class="ui-field" rows="4">{}</textarea></div>` : '';
    const form = panel(t('Run'), t('plugins.label.form_note'), html`${own.map(control)}${json}${typedBtn(t('Run'), 'plugin-run', '', 'button primary', S.busy ? 'Waiting for the product owner' : '')}`);
    return html`${top}${objectHead(nameOf(p), html`<p class="lede">${codeWords(p.effect_mode)} · ${codeWords(p.state)}</p>`, '', '', [], {object: true, id: p.capability})}${form}${S.receipt ? receipt(S.receipt) : ''}`;
  }
  function open(id) { S.receipt = null; S.refused = ''; navigate('plugins', {capability: id}, {replace: true}); }
  return {pages, ensure, leave, refresh, page, open, run, list: () => navigate('plugins', {capability: ''})};
})();
