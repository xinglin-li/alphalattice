/* Research history: the workspace's saved records and exact evidence objects, as rows. */
const historyItems = () => Data.history();
const OPERATION_MODES = {readback: 'Readback', replay: 'Replay', reuse: 'Reuse exact', fresh_run: 'Fresh run'}; // a row names its operation only where it is not the ordinary readback
/* Each date sort names the date it orders. Rows without that date stay listed after the dated
 * rows, in product order, rather than being sorted by an unrelated field or dropped. */
const HISTORY_SORTS = {
  'recorded-old': ['recordedAt', 1], 'input-new': ['inputCutoff', -1], 'input-old': ['inputCutoff', 1],
  'holdings-new': ['holdingsSession', -1], 'holdings-old': ['holdingsSession', 1], name: ['label', 1],
};

const History = (() => {
  const label = (x) => t(x.name) + (x.words || x.summary ? ' · ' + (x.words || x.summary) : ''); // the declared facts in words; the exact token stays in the row's sub-cell
  const ORDERS = () => [['source', t('Recorded · newest')], ['recorded-old', t('Recorded · oldest')], ['input-new', t('Input cutoff · newest')], ['input-old', t('Input cutoff · oldest')], ['holdings-new', t('Holdings session · newest')], ['holdings-old', t('Holdings session · oldest')], ['name', t('Object label')]];
  function sorted(rows, key) {
    const spec = HISTORY_SORTS[key];
    if (!spec) return rows;
    const [field, direction] = spec;
    const value = (x) => (field === 'label' ? label(x) : x[field]);
    const dated = rows.filter((x) => value(x)), undated = rows.filter((x) => !value(x));
    dated.sort((a, b) => direction * String(value(a)).localeCompare(String(value(b))) || a.id.localeCompare(b.id));
    return [...dated, ...undated];
  }
  function filtered() {
    const q = app.historyQuery.toLowerCase().trim();
    const kinds = app.historyKind === 'all' ? null : String(app.historyKind).split(','); // round 91: a filter holds several values
    const rows = historyItems().filter((x) => (!kinds || kinds.includes(x.kind)) && (!q || [x.kind, x.name, x.summary, x.words, x.reference, x.id, x.task_id, x.input, x.inputCutoff, x.recordedAt, x.holdingsSession, x.note].join(' ').toLowerCase().includes(q)));
    return sorted(rows, app.historySort);
  }
  const declaredNote = () => Data.experiments() === null ? t('Declared parameters not loaded yet.') : Data.experimentsError ? t('Declared parameters unavailable: {error}', {error: Data.experimentsError}) : t('Labels summarize declared parameters, never results or a winner.');
  /* The route carries the search and the filter (replaced, never pushed): a copied link, a reload
   * and a return by Back show this same list; the order is the viewer's display (round 57). */
  const routeFilters = () => { replaceHash({q: app.historyQuery, kind: app.historyKind === 'all' ? '' : app.historyKind}); };
  /* History as a lobby (F2, law 136): grouped by time -- the reader's day, this week, earlier this
   * month open, each month before folded -- by kind, or by input version (the newest open); one
   * line a row: the kind's mark, the reference, the name, the facts, the day. The owner's older
   * pages are read from the foot. */
  function lobby() {
    const items = historyItems(), kinds = [...new Set(items.map((x) => x.kind))];
    const versions = [...new Set(items.map((x) => x.raw?.input_binding_hash).filter(Boolean))].sort((a, b) => String(items.find((x) => x.raw?.input_binding_hash === b)?.inputCutoff || '').localeCompare(String(items.find((x) => x.raw?.input_binding_hash === a)?.inputCutoff || '')));
    const byInput = (x) => { const h = x.raw?.input_binding_hash; return h ? {key: h, label: html`${x.input} · ${LiveViews.cutoffText(x)}`, rank: versions.indexOf(h), open: versions.indexOf(h) === 0} : {key: 'none', label: t('No input version'), rank: 9e9, open: false}; };
    const row = (x, d) => LiveViews.recordRow(x, {ref: x.ref, columns: ['mode', 'note'], props: [x.mode !== 'readback' ? t(OPERATION_MODES[x.mode]) : '', x.note ? coded(x.note) : ''], show: d.props, actions: html`${x.task_id ? btnAttrs(html`${icon('task')}<span>${t('Open the Task')}</span>`, 'task', x.task_id, 'menu-row', html`role="menuitem"`) : ''}${btnAttrs(html`${icon('copy')}<span>${t('Copy reference')}</span>`, 'history-copy', x.id, 'menu-row', html`role="menuitem"`)}`});
    return Lobby.render('history', {items, row,
      axes: [{key: 'time', label: t('Time'), group: (x) => timeGroup(x.raw?.recorded_at || x.recordedAt)}, {key: 'kind', label: t('Artifact kind'), group: (x) => ({key: x.kind, label: t(x.kind), rank: kinds.indexOf(x.kind), open: true})}, {key: 'input', label: t('Input version'), group: byInput}],
      select: (d) => { app.historySort = d.order; return filtered(); }, orders: ORDERS(), apply: (d) => { app.historySort = d.order; },
      bind: {get: () => ({query: app.historyQuery, filters: app.historyKind === 'all' ? {} : {kind: app.historyKind}}), set: (patch) => { if ('query' in patch) app.historyQuery = patch.query; if ('filters' in patch) app.historyKind = patch.filters.kind || 'all'; routeFilters(); }},
      words: label, placeholder: t('Strategy, input, kind or ID'),
      filters: [{field: 'kind', label: t('Kind'), multiple: true, options: kinds.map((k) => [k, t(k)]), test: (x, one) => x.kind === one}],
      properties: [['origin', t('Continued from'), false], ['interval', t('Interval')], ['input', t('Input version')], ['holdings', t('Holdings session')], ['recorded', t('Recorded')]],
      foot: Data.hasMoreHistory() ? Lobby.older(t('Older records are not read yet'), 'history-more', '', Data.historyLoading) : ''});
  }
  function page() {
    // The head: the collection in two icon facts; what a label is and is not is its (i)
    const latest = historyItems().map((x) => x.recordedAt).filter(Boolean).sort().at(-1);
    const meta = html`<span>${icon('archive')} ${countText(historyItems().length, '{n} saved record', '{n} saved records')}${Data.hasMoreHistory() ? html` · ${t('more available')}` : ''}${latest ? html` · ${t('latest recorded')} ${latest}` : ''}</span><span>${icon('lock')} ${t('Exact readback · nothing recomputed')}${infoMark(`${declaredNote()} ${t('Metadata discovery is not artifact verification. Open retains the saved input and book.')}`)}</span>`;
    const unread = Data.historyError?.();
    const collectionRefusal = unread ? refusal(unread, 'warning', {word: t('History could not be read'), catalog: true, more: unread.task_id ? html`<p>${t('Task')} ${hashCell(unread.task_id)}</p>` : '', next: html`${prerequisiteWays(unread.next_requests)} ${btn(t('Read again'), 'workspace-refresh', '', 'button compact')}`}) : '';
    const refused=(Data.historyRefusals?.() || []).map(r=>refusal(r,'warning',{catalog:true,more:html`<p>${t('Recorded reference')} ${r.task_id ? hashCell(r.task_id) : locatorCell(r.entry_id || r.result_hash || r.index_file || '')}</p>`,next:prerequisiteWays(r.next_requests)}));
    return html`${objectHead(t('History'), html`<p class="lede">${t('Browse saved research, including failed and interrupted tasks.')}</p>${meta}`, '', '', [])}${collectionRefusal}${refused}${historyItems().length ? detailSplit(lobby(), 'object') : refused.length || unread ? '' : emptyState(t('No saved record yet'), link(t('New experiment'), 'lab', 'button primary'), 'page-empty')}`;
  }
  function openItem(id) {
    // N1: the list that showed the row is the one a record steps through (a study's own list, not
    // History's: `3 / 6` stood over a book opened from a list of one); History's own in the order shown
    const from = app.pressed?.isConnected && app.page !== 'history' ? app.pressed.closest('.card-list, [role=list]') : null;
    const shown = from ? [...from.querySelectorAll('[data-row][data-key]')].map((r) => r.dataset.key) : [];
    const listed = Lobby.listed('history').map((x) => x.id);
    const keys = shown.includes(id) ? shown : listed.includes(id) ? listed : filtered().map((x) => x.id);
    app.listContext = keys.includes(id) ? {list: shown.includes(id) ? app.page : 'history', keys, index: keys.indexOf(id), from: routeObjectId(), at: ''} : null; // `from`/`at`: the record it opens (keepListContext)
    return Data.openEntry(id);
  }
  return {page, routeFilters, openItem, filtered};
})();
PAGES.history = History.page;

Object.assign(ACTIONS, {
  'history-open': (id) => History.openItem(id),
  'history-copy': (id) => copyText(id), // the full exact reference; labels are never the route
});
