/* Inspection without leaving the page: Focus mode, the read-only proof lens, Quick Open,
 * declaration differences, chart observations and the research map. */
const Inspect = (() => {
  const S = {baseline: clone(defaults()), observation: Data.series().length - 1, window: null, focus: false, lens: null, lensReturn: null, commandIndex: 0};

  /* ---- declaration difference (initial fixture draft → current declaration) ---- */
  const DIFF_FIELDS = [['kind', 'Experiment type'], ['input', 'Research input'], ['names', 'Names per sleeve'], ['sleeves', 'Sleeve count'], ['exit', 'Exit rank'], ['cost', 'Cost · bps / side'], ['seed', 'Deterministic seed']];
  function diffValues(d) {
    return DIFF_FIELDS.filter(([k]) => d.kind.startsWith('portfolio') || !['names', 'sleeves', 'exit', 'cost'].includes(k)).map(([key, label]) => ({key, label, before: S.baseline[key], after: d[key], changed: String(S.baseline[key]) !== String(d[key])}));
  }
  function declarationDiff(d) {
    const rows = diffValues(d);
    const changes = rows.filter((x) => x.changed);
    const shown = changes.length ? changes : rows.filter((x) => ['input', 'names', 'exit'].includes(x.key));
    return html`<div class="diff-header"><span class="diff-kind">${t('Declaration difference')}</span><span class="diff-count">${I18N.plural(changes.length, '{n} edit', '{n} edits')}</span></div><h3>${changes.length ? t('Parameter changes') : t('No parameter changes')}</h3><p class="diff-reference">${t('Initial fixture draft → current declaration.')}</p><div class="diff-column-labels"><span>${t('Parameter')}</span><span>${t('Starting → Current')}</span></div><div class="diff-rows">${shown.map((x) => html`<div class="diff-row ${x.changed ? 'changed' : ''}"><span>${(t(x.label))}</span><span class="diff-values"><span class="diff-before">${(x.before ?? t('Not set'))}</span><span class="diff-arrow" aria-label="${t('to')}">→</span><strong>${(x.after ?? t('Not set'))}</strong></span></div>`)}</div>${changes.length ? '' : html`<p class="diff-empty">${t('No parameter changes. Confirmation still submits a new Task.')}</p>`}<p class="diff-boundary">${t('Parameter comparison only · no predicted outcome.')}</p>`;
  }

  /* ---- chart window and observation cursor (portfolio) ----
   * The navigator under the chart is the whole published series in miniature with the reader's
   * window on it: two handles, a pannable middle, presets by calendar. The chart above draws the
   * rows in the window; the observation cursor is one row, moved on the chart or by keyboard.
   * Neither changes the holdings date, the book or a value. */
  // round 95 (Apple Stocks' row): the ranges the series is long enough to need, else none
  const PRESETS = [['3m', '3M', 3], ['1y', '1Y', 12], ['3y', '3Y', 36]];
  const MIN_SPAN = 2;
  function presetsFor(rows = Data.series()) {
    const n = rows.length;
    if (n < 2) return [];
    const first = rows[0].date, last = rows[n - 1].date;
    const months = (Number(last.slice(0, 4)) - Number(first.slice(0, 4))) * 12 + (Number(last.slice(5, 7)) - Number(first.slice(5, 7)));
    const ranges = PRESETS.filter(([, , span]) => months > span).map(([key, label]) => [key, label]);
    return ranges.length ? [...ranges, ['all', 'All']] : [];
  }
  function clampWindow(a, b, n) {
    a = Math.max(0, Math.min(n - MIN_SPAN, Math.round(a)));
    b = Math.max(a + MIN_SPAN - 1, Math.min(n - 1, Math.round(b)));
    return [a, b];
  }
  function chartWindow(n = Data.series().length) {
    if (!n) return [0, 0];
    if (!S.window) return [0, n - 1];
    return clampWindow(S.window[0], S.window[1], n);
  }
  /* A preset window by calendar from the last observation: the first row on or after that date. */
  function presetWindow(key, rows = Data.series()) {
    const n = rows.length;
    if (!n || key === 'all') return [0, n - 1];
    const last = rows[n - 1].date, y = Number(last.slice(0, 4)), m = Number(last.slice(5, 7)), d = Number(last.slice(8, 10));
    let start;
    if (key === 'ytd') start = `${y}-01-01`;
    else {
      const months = {'1m': 1, '3m': 3, '6m': 6, '1y': 12, '3y': 36}[key] || 12;
      const date = new Date(Date.UTC(y, m - 1 - months, Math.min(d, 28)));
      start = date.toISOString().slice(0, 10);
    }
    let a = rows.findIndex((r) => r.date >= start);
    if (a < 0) a = 0;
    return clampWindow(a, n - 1, n);
  }
  const presetActive = (key, rows, a, b) => { const [pa, pb] = presetWindow(key, rows); return pa === a && pb === b; };
  /* Readings of the rows in the window: the change of the published index from just before the
   * first row to the last, for the study and the benchmark, and the deepest fall of the index
   * from a peak inside the window. Arithmetic on the owner's rows; the tiles above stay the
   * owner's whole-report values. */
  function windowReadings(rows, a, b) {
    const finite = (z) => z !== null && z !== undefined && Number.isFinite(z);
    const base = (key) => (a > 0 ? rows[a - 1]?.[key] : 100);
    const change = (key) => { const s = base(key), e = rows[b]?.[key]; return finite(s) && finite(e) && s > 0 ? (e / s - 1) * 100 : null; };
    const fall = (key) => { let peak = base(key), worst = 0; if (!finite(peak) || peak <= 0) return null; for (let i = a; i <= b; i++) { const v = rows[i][key]; if (!finite(v)) return null; if (v > peak) peak = v; worst = Math.min(worst, v / peak - 1); } return worst * 100; };
    const study = change('value'), bench = change('benchmark');
    return {study, bench, excess: finite(study) && finite(bench) ? study - bench : null, drawdown: fall('value'), sessions: b - a + 1};
  }
  function readingsMarkup(rows, a, b) {
    const r = windowReadings(rows, a, b);
    const tone = (v) => (v === null ? '' : v > 0 ? 'pos' : v < 0 ? 'neg' : '');
    const cell = (label, value, cls = '') => html`<div class="reading"><small>${label}</small><b class="${cls}">${value}</b></div>`;
    return html`${cell(t('Study'), signed(r.study), tone(r.study))}${cell(t('Benchmark'), signed(r.bench), tone(r.bench))}${cell(t('Excess'), signed(r.excess, 'pp'), tone(r.excess))}${cell(t('Deepest fall'), r.drawdown === null ? '' : num(r.drawdown, 'percent'))}<div class="reading"><small>${t('trading|Sessions')}</small><b>${count(r.sessions)}${infoMark(t(a > 0 || b < rows.length - 1 ? 'Readings of the published rows in this window, {from} — {to}: the index change from just before its first row; the tiles above stay the owner\'s whole-report values.' : 'Readings of the whole published series; the tiles above are the owner\'s whole-report values, not recomputed here.', {from: rows[a].date, to: rows[b].date}))}</b></div>`; // N3: the readings' explanation is their (i)
  }
  /* The whole series in miniature, in a 1000×40 box (stretched to the rail): the study index and
   * the benchmark. Two dozen segments per hundred rows are enough for a strip this small. */
  function miniPath(rows, key) {
    const finite = (z) => z !== null && z !== undefined && Number.isFinite(z);
    const vs = rows.map((r) => r[key]), flat = vs.filter(finite);
    if (!flat.length) return '';
    const lo = Math.min(...flat), hi = Math.max(...flat), span = Math.max(1e-9, hi - lo), n = rows.length;
    const every = Math.max(1, Math.floor(n / 400));
    let d = '', open = false;
    for (let i = 0; i < n; i += every) {
      const v = vs[i];
      if (!finite(v)) { open = false; continue; }
      d += (open ? 'L' : 'M') + ((i / Math.max(1, n - 1)) * 1000).toFixed(1) + ' ' + (36 - ((v - lo) / span) * 32).toFixed(1);
      open = true;
    }
    return d;
  }
  /* The navigator (round 95: the observation row above it retired -- the chart's hero is the
   * readout under the pointer, and a figure's facts live in Facts; the range presets moved to the
   * chart's toolbar, so the foot is the window's two dates). */
  function observationDock() {
    const rows = Data.series();
    if (!rows.length) return '';
    const n = rows.length, [a, b] = chartWindow(n);
    return html`<div class="chart-navigator" data-navigator aria-label="${t('Chart window')}"><div class="navigator-rail" data-rail><svg class="navigator-svg" viewBox="0 0 1000 40" preserveAspectRatio="none" aria-hidden="true"><path class="navigator-benchmark" d="${miniPath(rows, 'benchmark')}"/><path class="navigator-series" d="${miniPath(rows, 'value')}"/></svg><div class="navigator-shade start" data-shade-start></div><div class="navigator-shade end" data-shade-end></div><div class="navigator-window" data-window><button type="button" class="navigator-handle" data-handle="start" role="slider" aria-label="${t('Window start')}" aria-valuemin="0" aria-valuemax="${n - 1}" aria-valuenow="${a}" aria-valuetext="${rows[a].date}"></button><button type="button" class="navigator-handle" data-handle="end" role="slider" aria-label="${t('Window end')}" aria-valuemin="0" aria-valuemax="${n - 1}" aria-valuenow="${b}" aria-valuetext="${rows[b].date}"></button></div><i class="navigator-cursor" data-cursor aria-hidden="true"></i></div><div class="navigator-foot"><span class="mono" data-window-start>${rows[a].date}</span><span class="mono" data-window-end>${rows[b].date}</span></div><div class="window-readings" data-readings>${readingsMarkup(rows, a, b)}</div><p>${t('{n} observations · the window and the cursor read the chart; neither changes holdings.', {n: count(n)})}</p></div>`;
  }
  /* Set the window (null means all) and redraw what depends on it, in place. */
  function setWindow(a, b) {
    const n = Data.series().length;
    if (!n) return;
    const [wa, wb] = clampWindow(a, b, n);
    S.window = wa === 0 && wb === n - 1 ? null : [wa, wb];
    if (S.observation < wa || S.observation > wb) S.observation = Math.max(wa, Math.min(wb, S.observation));
    if (typeof Portfolio !== 'undefined') Portfolio.redrawChart();
  }
  const resetWindow = () => { S.window = null; };
  function syncObservation() {
    const rows = Data.series(), r = rows[S.observation];
    if (!r) return;
    const cursor = $('[data-navigator] [data-cursor]');
    if (cursor) cursor.style.left = ((S.observation / Math.max(1, rows.length - 1)) * 100).toFixed(3) + '%';
    const target = $('#performanceChart [data-chart]');
    if (target) {
      const [a, b] = chartWindow(rows.length), cross = target.closest('svg').querySelector('[data-cross]');
      if (S.observation < a || S.observation > b) cross.setAttribute('opacity', '0');
      else {
        const w = +target.dataset.width, l = +target.dataset.left, rr = +target.dataset.right;
        const x = l + ((S.observation - a) / Math.max(1, b - a)) * (w - l - rr);
        cross.setAttribute('x1', x);
        cross.setAttribute('x2', x);
        cross.setAttribute('opacity', '1');
      }
    }
    refreshLens();
  }
  function selectObservation(i) {
    S.observation = Math.max(0, Math.min(Data.series().length - 1, i));
    syncObservation();
  }
  /* ---- the proof lens: the read-only inspection beside the page (round 62); its bodies are
   * LiveViews.proofBody's; the research context left it for the workspace popover and the
   * Facts panel's first section (round 81) ---- */
  const PROOF_NAMES = {knowledge: 'Knowledge boundary', declaration: 'Declaration changes', observation: 'Observation source', coverage: 'Review coverage', annual: 'Return, with its scope', vol: 'Volatility, with its scope', drawdown: 'Drawdown, with its scope', sharpe: 'Sharpe, with its scope', sortino: 'Sortino, with its scope'};
  const proofBody = (key) => LiveViews.proofBody(key, S.observation);
  /* Round 92: the lens is gone -- a proof is the facts of one figure, so the proofs are sections of
   * the Facts tab on the pages that have them (the book's metrics, its knowledge boundary, the
   * observation under the pointer), and a tile's ⓘ opens Facts at that section. */
  const PROOF_KEYS = ['knowledge', 'declaration', 'observation', 'coverage', 'annual', 'vol', 'drawdown', 'sharpe', 'sortino'];
  const proofSections = () => app.page === 'portfolio' && Data.subject() ? PROOF_KEYS.filter((k) => k === 'knowledge' || k === 'observation' || k in (Data.metrics() || {}) || LiveViews.metricAbsence(k)).map((k) => ({key: k, title: t(PROOF_NAMES[k]), body: proofBody(k)})) : [];
  function openLens(key) {
    if (!PROOF_NAMES[key]) return false;
    closeDialog();
    const opened = openFacts();
    const section = $(`#inspector [data-facts="${key}"]`);
    if (section) { section.scrollIntoView({block: 'start', behavior: 'instant'}); section.classList.add('ui-arrived'); }
    return opened;
  }
  const closeLens = () => false; // kept for its callers; nothing to close (round 92)
  function refreshLens() {
    // the observation section follows the pointer without repainting the whole tab
    const host = $('#inspector [data-facts="observation"]');
    if (host && Window.inspectorMode() === 'facts') host.innerHTML = html`<h3>${t(PROOF_NAMES.observation)}</h3>${proofBody('observation')}`;
  }

  /* ---- focus mode ---- */
  function toggleFocus() {
    closeDialog();
    S.focus = !S.focus;
    document.body.classList.toggle('focus-mode', S.focus);
    $$('[data-action="focus-toggle"]').forEach((x) => {
      x.setAttribute('aria-pressed', String(S.focus));
      x.setAttribute('aria-label', S.focus ? t('Exit Focus mode') : t('Enter Focus mode'));
      x.dataset.tip = S.focus ? t('Exit Focus · Escape') : t('Focus mode');
    });
    $$('.inline-focus span').forEach((x) => (x.textContent = S.focus ? t('Exit Focus') : t('Focus')));
    // a menu row is menuRow's shape: its word, and its why as the row's tip -- in the page and in the page's menu
    // template too, from which the top row is rebuilt when Focus shows or hides it (else the row read the old word)
    for (const root of [document, ...$$('template.page-menu').map((tpl) => tpl.content)]) root.querySelectorAll('.menu-row[data-action="focus-toggle"]').forEach((x) => {
      const word = x.querySelector('.menu-word');
      if (word) word.textContent = S.focus ? t('Exit Focus') : t('Focus');
      x.dataset.tip = S.focus ? t('Exit Focus mode') : t('Enter Focus mode');
      x.setAttribute('aria-pressed', String(S.focus));
    });
    if (typeof Window !== 'undefined') Window.afterRender();
  }

  /* ---- Quick Open: local navigation, never an AI prompt ---- */
  /* The page's own actions (round 15): every labelled, enabled, non-mutating action button the
   * page or its chrome shows right now — the same admission the button has, read from the button.
   * A row's opener and the dialog's controls are not commands; a mutating action needs its own
   * confirmation and stays a button. */
  function pageActions() {
    const seen = new Set(), out = [];
    // the page's own actions first, then the dock's, then the chrome's
    for (const b of [...$$('#main button[data-action]'), ...$$('#top button[data-action]'), ...$$('#side button[data-action]')]) {
      const action = b.dataset.action, value = b.dataset.value || '';
      if (!action || ['close', 'quick-open', 'quick-pick', 'tools-menu', 'copy-block', 'workspace-switch', 'workspace'].includes(action)) continue;
      if (b.disabled || b.getAttribute('aria-disabled') === 'true' || b.dataset.mutating === 'true' || b.closest('.list-row, .row-actions, .card-note, dialog')) continue;
      if (!b.offsetParent && !b.closest('.rail-tools')) continue; // the folded tools stay reachable here
      // a menu item is a word and a reason (<strong> + <small>) or a word and its note (`.menu-word` + `.menu-note`, the note a chord): the word is the label, the chord its key (round 95: `Keyboard shortcuts?` and `Copy linkL` were the note glued on)
      const label = (b.getAttribute('aria-label') || b.querySelector('strong')?.textContent || b.querySelector('.menu-word')?.textContent || [...b.childNodes].filter((n) => !(n.getAttribute?.('aria-hidden') === 'true')).map((n) => n.textContent).join('') || '').replace(/\s+/g, ' ').trim();
      if (!label || label.length > 48) continue;
      const key = action + ':' + value;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(['!' + key, label, t('Action on this page'), 'arrow', action, value, (b.querySelector('.menu-note')?.textContent || '').trim()]);
    }
    return out;
  }
  /* ---- the command menu (round 54, Linear's Ctrl K): where you are, then what you can do ----
   * Sections: Recent (the last five choices, kept in the viewer's preferences), Actions (the
   * open object's first, then the page's, with their keys where round 51 bound one), Pages (the
   * routes with their chords) and Records (the saved records and the Tasks the browser already
   * holds). A query matches words in any order and id prefixes; Tab moves to the next section.
   * Nothing here runs a mutation directly: an action runs as its button would, confirmation and
   * all. */
  const KEY_OF_ACTION = {shortcuts: '?', 'focus-toggle': '', 'step-prev': '[', 'step-next': ']'};
  const RECORD_ICONS = {'portfolio.policy-development': 'portfolio', 'alpha.model-development': 'branch', 'factor.screening-development': 'lab', 'risk.covariance-development': 'evidence', CRO_REVIEW: 'review'};
  const recentIds = () => { const v = readPreference('commands.recent'); return Array.isArray(v) ? v : []; };
  function commands() {
    const out = [];
    const object = SAVED_VIEWS.has(app.page);
    const scoped = Boolean(S.commandScope) || !SAVED_VIEWS.has(app.page); // round 93: the chip gone, the object's own verbs leave the list and the search is the whole workspace
    for (const a of pageActions()) {
      const [id, label, , , action] = a;
      // the object's own verbs first, the page tools after them, the rest of the page last
      const button = $$('#main button[data-action], #top button[data-action]').find((b) => b.dataset.action === action && (b.dataset.value || '') === (a[5] || ''));
      const tool = Boolean(button?.closest('.rail-tools')), verb = Boolean(button?.closest('.object-actions, .top-actions')) && !tool;
      if (verb && object && !scoped) continue;
      out.push({section: 'Actions', id, icon: 'arrow', words: label, sub: verb && object ? t('On this object') : tool ? t('Page tools') : t('On this page'), key: KEY_OF_ACTION[action] || a[6] || '', rank: verb ? 0 : tool ? 1 : 2, run: () => window.AlphaLattice.action(action, a[5] || '')});
    }
    // the object's head links (Compare, Review evidence…) are its verbs too
    if (scoped) for (const link of $$('#main .object-actions a[href^="#page="], #top .top-actions a[href^="#page="]')) { const words = link.textContent.replace(/\s+/g, ' ').trim(); if (words) out.push({section: 'Actions', id: 'link:' + link.getAttribute('href'), icon: 'arrow', words, sub: t('On this object'), key: '', rank: 0, run: () => { location.href = link.getAttribute('href'); }}); }
    for (const [key, name, words, section] of LiveViews.quickEntries()) out.push({section: 'Pages', id: 'page:' + key, icon: section, words: t(name), sub: words, key: Controls.chordFor(key), rank: Controls.chordFor(key) ? 0 : 1, run: () => navigate(key)});
    for (const [mode, word] of NAV_MODES) out.push({section: 'Pages', id: 'nav:' + mode, icon: 'panel', words: t('Navigation: {mode}', {mode: t(word)}), sub: Window.navMode() === mode ? t('Current') : t('Where the navigation stands'), key: '', rank: 2, run: () => Window.setNavigation(mode)});
    for (const [side, word] of DOCK_SIDES) out.push({section: 'Pages', id: 'dock:' + side, icon: side === 'right' ? 'panel-right' : 'panel', words: t('Dock: {side}', {side: t(word)}), sub: Window.dockSide() === side ? t('Current') : t('Dock side'), key: '', rank: 2, run: () => Window.setDockSide(side)});
    out.push({section: 'Actions', id: 'zoom:+', icon: 'edit', words: t('Larger text'), sub: t('Text size {n} %', {n: Window.zoomLevel()}), key: 'Ctrl =', rank: 3, run: () => Window.stepZoom(1)},
      {section: 'Actions', id: 'zoom:-', icon: 'edit', words: t('Smaller text'), sub: t('Text size {n} %', {n: Window.zoomLevel()}), key: 'Ctrl −', rank: 3, run: () => Window.stepZoom(-1)},
      {section: 'Actions', id: 'zoom:0', icon: 'edit', words: t('Reset text size'), sub: '100 %', key: 'Ctrl 0', rank: 3, run: () => Window.setZoom(100)});
    for (const x of Data.history()) out.push({section: 'Records', id: 'record:' + x.id, icon: RECORD_ICONS[x.raw?.kind] || 'file', words: t(x.name) + (x.words ? ' · ' + x.words : ''), sub: [x.reference, x.recordedAt ? when(x.recordedAt) : ''].filter(Boolean).join(' · '), key: '', rank: 1, run: () => History.openItem(x.id)});
    for (const v of Data.tasks()) out.push({section: 'Records', id: 'task:' + v.task_id, icon: 'task', words: v.goal_summary ? t(v.goal_summary) : v.task_id, sub: t('Task') + ' ' + short(v.task_id || '', SHORT.id), key: '', rank: 2, run: () => { navigate('tasks'); LiveTasks.select(v.task_id); }});
    return out;
  }
  const SECTIONS = ['Recent', 'Actions', 'Pages', 'Records'];
  function commandResults(query) {
    const all = commands(), words = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const text = (c) => `${c.words} ${c.sub} ${c.id}`.toLowerCase();
    if (!words.length) {
      const recent = recentIds().map((id) => all.find((c) => c.id === id)).filter(Boolean).map((c) => ({...c, section: 'Recent'}));
      const take = (section, n) => all.filter((c) => c.section === section).sort((a, b) => a.rank - b.rank).slice(0, n);
      return [...recent, ...take('Actions', 6), ...take('Pages', 8), ...take('Records', 5)];
    }
    return all.filter((c) => words.every((w) => text(c).includes(w))).sort((a, b) => SECTIONS.indexOf(a.section) - SECTIONS.indexOf(b.section) || a.rank - b.rank).slice(0, 24);
  }
  function drawCommands() {
    const q = $('#quickInput')?.value || '';
    const rows = commandResults(q);
    S.commandRows = rows;
    S.commandIndex = Math.max(0, Math.min(S.commandIndex, rows.length - 1));
    let section = '';
    $('#quickResults').innerHTML = rows.length
      ? html`${rows.map((r, i) => { const head = r.section !== section ? html`<p class="menu-section">${t(r.section)}</p>` : ''; section = r.section; return html`${head}<button class="quick-result${i === S.commandIndex ? ' active' : ''}" data-action="quick-pick" data-value="${r.id}" id="quickOption${i}" role="option" aria-selected="${i === S.commandIndex}">${icon(r.icon)}<span class="quick-words"><strong>${r.words}</strong>${r.sub ? html`<small>${r.sub}</small>` : ''}</span>${r.key ? keycap(r.key) : ''}</button>`; })}`
      : html`<div class="quick-empty"><strong>${t('Nothing matches.')}</strong><p>${t('Try a page, an action or a record\'s words. No web search is sent.')}</p></div>`;
  }
  /* where you are: the open object (its name and id) or the page */
  function commandContext() {
    const object = SAVED_VIEWS.has(app.page) && $('#main .object-header h1');
    return {eyebrow: t(ROUTES[app.page]?.[0] || 'Workbench'), name: object ? object.textContent.replace(/\s+/g, ' ').trim() : t(ROUTES[app.page]?.[1] || app.page), id: object ? $('#main .object-id code')?.textContent || '' : ''};
  }
  function openQuick() {
    closeLens(false);
    const c = commandContext();
    S.commandScope = SAVED_VIEWS.has(app.page) ? c : null; // the open object: its verbs first while the chip stands
    const chip = S.commandScope ? html`<span class="quick-context" id="quickContext">${c.name}${c.id ? html` <code>${short(c.id, SHORT.id)}</code>` : ''}</span>` : '';
    openPalette(html`<div class="quick-search">${icon('search')}${chip}<label class="sr-only" for="quickInput">${t('Search pages, records and actions…')}</label><input id="quickInput" autocomplete="off" placeholder="${t('Search pages, records and actions…')}" aria-controls="quickResults" role="combobox" aria-expanded="true" aria-autocomplete="list"></div><div id="quickResults" role="listbox" aria-label="${t('Quick Open')}"></div>`, () => openQuick());
    S.commandIndex = 0;
    drawCommands();
    $('#quickInput').focus();
  }
  function dropCommandScope() { S.commandScope = null; $('#quickContext')?.remove(); drawCommands(); }
  function commandPick(id) {
    const row = (S.commandRows || []).find((c) => c.id === id);
    closeDialog();
    if (!row) return;
    if (/^(page|record|task|link):/.test(id) && typeof LiveActivity !== 'undefined') LiveActivity.pauseFollowing?.();
    savePreference('commands.recent', [id, ...recentIds().filter((x) => x !== id)].slice(0, 5));
    return row.run();
  }
  const quickOpenActive = () => $('#dialog').open && $('#dialog').classList.contains('quick-dialog');
  function quickKeydown(e) {
    if (!quickOpenActive() || e.target.id !== 'quickInput') return false;
    const rows = commandResults(e.target.value);
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      S.commandIndex = rows.length ? (S.commandIndex + (e.key === 'ArrowDown' ? 1 : -1) + rows.length) % rows.length : 0;
      drawCommands();
      $('#quickOption' + S.commandIndex)?.scrollIntoView({block: 'nearest'});
    }
    if (e.key === 'Backspace' && !e.target.value && S.commandScope) { e.preventDefault(); dropCommandScope(); return true; }
    if (e.key === 'Tab' && rows.length) {
      e.preventDefault();
      const here = rows[S.commandIndex]?.section;
      const next = rows.findIndex((r, i) => (e.shiftKey ? i < S.commandIndex : i > S.commandIndex) && r.section !== here);
      S.commandIndex = next >= 0 ? (e.shiftKey ? rows.findIndex((r) => r.section === rows[next].section) : next) : (e.shiftKey ? rows.length - 1 : 0);
      drawCommands();
      $('#quickOption' + S.commandIndex)?.scrollIntoView({block: 'nearest'});
    }
    if (e.key === 'Enter' && rows.length) {
      e.preventDefault();
      commandPick(rows[S.commandIndex].id);
    }
    return true;
  }

  /* ---- the peek (round 53, Linear's Space): a card beside the held row, from what the row
   * already renders — its id, its title, its properties, its note, its time; no second read.
   * At the row's right edge on a wide page, under the row on a narrow one; Space or Esc closes,
   * J / K move it with the held row, Enter opens the record. ---- */
  const peekEl = () => $('#objectPeek');
  const peekOpen = () => Boolean(peekEl() && !peekEl().hidden);
  let peekRow = null;
  const clean = (s) => String(s || '').replace(/\s+/g, ' ').trim();
  function peekCard(row) {
    const table = row.tagName === 'TR';
    const id = clean(row.querySelector('.mono')?.textContent) || (table ? '' : row.dataset.key || '');
    let title = '', props = [];
    if (table) {
      const heads = [...row.closest('table').querySelectorAll('thead th')].map((th) => clean(th.textContent));
      const cells = [...row.cells];
      title = clean(cells[0]?.querySelector('strong, .main-cell')?.textContent || cells[0]?.textContent);
      props = cells.slice(1).map((td, i) => { const v = clean(td.querySelector('button, a') ? '' : td.textContent); return v && heads[i + 1] ? `${heads[i + 1]} ${v}` : ''; }).filter(Boolean);
    } else {
      title = clean(row.querySelector('.list-row-main strong')?.textContent);
      props = [...row.querySelectorAll('.list-row-props > span')].map((s) => clean(s.textContent)).filter((s) => s && s !== id);
    }
    const why = clean(row.querySelector('.list-row-why, .sub-cell')?.textContent);
    const time = clean(row.querySelector('.list-row-time')?.textContent);
    // round 63: a Task row's card reads the Task itself, so the truth line (its goal words) is on the card, not in the row
    const task = row.classList.contains('tp-task') ? Data.tasks().find((v) => v.task_id === row.dataset.key) : null;
    const keys = html`<p class="peek-keys">${btnAttrs(html`${keycap('Enter')} <span>${t('Open')}</span>`, 'peek-open', '', 'text-btn compact')}${btnAttrs(html`${keycap('Esc')} <span>${t('Close')}</span>`, 'peek-close', '', 'text-btn compact')}</p>`;
    if (task) return html`${recordCard('task', task)}${keys}`;
    return html`${id ? html`<p class="peek-id mono">${id}</p>` : ''}<h3>${title}</h3>${props.length ? html`<p class="peek-props">${props.join(' · ')}</p>` : ''}${why && why !== title ? html`<p class="peek-note">${why}</p>` : ''}${time ? html`<p class="peek-time">${time}</p>` : ''}${keys}`;
  }
  function placePeek(row, below = false) {
    const el = peekEl(), r = layoutRect(row), vw = viewW(), vh = viewH(), wide = vw >= BREAKPOINTS.window.narrow && !below, edge = param('viewport-inset'), gap = param('popover-gap'), near = param('space-2'), w = Math.min(param('width-sm'), vw - 2 * edge);
    el.style.width = w + 'px';
    el.style.left = (wide ? Math.max(edge, Math.min(r.right - w - near, vw - w - edge)) : Math.max(edge, Math.min(r.left, vw - w - edge))) + 'px';
    el.style.top = '0px';
    const h = el.offsetHeight;
    el.style.top = (wide ? Math.min(Math.max(near, r.top), Math.max(near, vh - h - near)) : (r.bottom + gap + h > vh - near ? Math.max(near, r.top - h - gap) : r.bottom + gap)) + 'px';
  }
  /* Round 92: the peek is the hover card and nothing more -- Space on a held row shows the card
   * beside it, with `Open` as its one way (Enter on the row opens the object); it never becomes
   * the side column. */
  function peek(row) {
    if (!row) return false;
    if (peekMode === 'hover') closeHover();
    if (row.classList.contains('tp-task') && row.dataset.key) { LiveTasks.open(row.dataset.key); return true; } // round 66: a Task's peek is the Task
    const card = peekEl();
    if (!card) return false;
    peekRow = row; peekMode = 'peek'; hoverEl = null;
    card.innerHTML = peekCard(row);
    card.hidden = false;
    placePeek(row);
    requestAnimationFrame(() => card.classList.add('is-open'));
    return true;
  }
  /* the floating card (a hover card) closes on its own; the inspector's peek closes the region */
  function closeHover() {
    const el = peekEl();
    if (!el || el.hidden) return false;
    el.classList.remove('is-open'); el.hidden = true; hoverEl = null;
    return true;
  }
  function closePeek() {
    const hovered = closeHover();
    if (peekMode === 'peek') { const row = peekRow; peekRow = null; peekMode = 'hover'; row?.querySelector('.list-row-main, a[href], button[data-action]')?.focus?.({preventScroll: true}); }
    return hovered;
  }
  const peekRowIs = (row) => peekRow === row;
  /* the peek's Open: the row opens as Enter would open it */
  function openPeeked() { const main = peekRow?.querySelector('.list-row-main, a[href], button[data-action]'); closePeek(); main?.click(); }
  function peekFollow(row) { if (peekOpen() && peekMode === 'peek') peek(row); }

  /* ---- hover cards (round 58, Linear's): a link to a record the browser already holds shows
   * the record's card under the pointer after 500 ms, and it leaves with the pointer; a focused
   * link shows it on Space. The card reads what the page holds (no second fetch): a saved
   * record from the history, a Task from the Task list; a link whose object is not in the
   * browser has no card. A list row's own opener is not a link here (the row peeks). ---- */
  const HOVER_LINKS = '#main [data-action="history-open"]:not(.list-row-main), #main [data-action="task"]:not(.list-row-main), #main a[href*="study="]:not(.list-row-main), #main a[href*="book="]:not(.list-row-main), #main [data-cite], #inspector [data-cite]'; // round 80: a reading's cite pills carry the card too; round 93: a term's words are the tooltip's
  const NOT_A_LINK = '.view-rail, .object-actions, .tabs, nav'; // navigation carries the open object in its routes: not a link to it
  let peekMode = 'peek', hoverEl = null, hoverTimer = 0;
  function linkRecord(el) {
    const v = el.dataset.value || '';
    if (el.dataset.cite) return ['cite', LiveReview.spanOf(el.dataset.cite)]; // round 76: the span or finding a handle names
    if (el.dataset.action === 'history-open') return ['record', Data.history().find((r) => r.id === v)];
    if (el.dataset.action === 'task') return ['task', Data.tasks().find((r) => r.task_id === v)];
    // a study link names its object in `study`; a book link in `book` (the route carries the open
    // book beside every link, so `book` counts only on a Portfolio route)
    const href = el.getAttribute('href') || '', params = new URLSearchParams(href.replace(/^#/, ''));
    const id = params.get('study') || (/^(portfolio|compare)$/.test(params.get('page') || '') ? params.get('book') : '') || '';
    return ['record', id ? Data.history().find((r) => r.id === id || r.task_id === id || r.raw?.book?.result_hash === id) : null];
  }
  function recordCard(kind, x) {
    if (kind === 'cite') { const r = evidenceName(x.span_handle ? 'citation' : 'finding', x); return html`<p class="peek-id mono">${r.id}</p><h3>${x.title || x.document_handle || r.kind}</h3>${x.excerpt ? html`<p class="peek-note owner-text">${x.excerpt}</p>` : ''}<p class="peek-props">${[r.why, r.word].filter(Boolean).map((p, i) => html`${i ? ' · ' : ''}${p}`)}</p>`; }
    if (kind === 'task') return html`<p class="peek-id mono">${x.task_id}</p><h3>${LiveViews.nameOf(x).name || x.task_id}</h3>${x.goal_summary ? html`<p class="peek-note">${t(x.goal_summary)}</p>` : ''}<p class="peek-props">${[codeWords(x.task_kind), codeWords(x.lifecycle), html`${t('Verified stages')} · ${count(x.verified_stage_count)} / ${count(x.total_stage_count)}`].filter(Boolean).map(String).join(' · ')}</p>${x.last_activity_at ? html`<p class="peek-time">${when(x.last_activity_at)}</p>` : ''}`;
    const props = [x.reference, x.interval, x.input ? `${x.input}${x.inputCutoff ? ' · ' + x.inputCutoff : ''}` : '', x.holdingsSession ? `${t('holdings')} ${x.holdingsSession}` : ''].filter(Boolean);
    return html`<p class="peek-id mono">${x.id}</p><h3>${t(x.name)}${x.words || x.summary ? ' · ' + (x.words || x.summary) : ''}</h3>${props.length ? html`<p class="peek-props">${props.join(' · ')}</p>` : ''}${x.recordedAt ? html`<p class="peek-time">${when(x.recordedAt)}</p>` : ''}`;
  }
  let swallowSpace = false; // Space opened a card on a focused button: its keyup must not click the button
  function hoverCard(el, byKey = false) {
    const card = peekEl();
    if (!card || !el) return false;
    if (byKey) swallowSpace = true;
    if (peekMode === 'hover' && !card.hidden && hoverEl === el) { closePeek(); return true; } // Space again closes
    if ($('#dialog')?.open || $('.menu:not([hidden]):not([data-leaving])')) return false;
    if (el.closest(NOT_A_LINK)) return false;
    const [kind, x] = linkRecord(el);
    if (!x) return false;
    hoverEl = el; peekMode = 'hover'; peekRow = null;
    card.innerHTML = recordCard(kind, x);
    card.hidden = false;
    placePeek(el, true);
    requestAnimationFrame(() => card.classList.add('is-open'));
    return true;
  }
  function bindHover() {
    document.addEventListener('keyup', (e) => { if (e.key === ' ' && swallowSpace) { e.preventDefault(); swallowSpace = false; } });
    document.addEventListener('mouseover', (e) => {
      const el = e.target.closest?.(HOVER_LINKS);
      if (!el || el === hoverEl) return;
      clearTimeout(hoverTimer); hoverEl = el;
      hoverTimer = setTimeout(() => { if (hoverEl === el && el.isConnected && el.matches(':hover')) hoverCard(el); }, 500);
    });
    document.addEventListener('mouseout', (e) => {
      if (!hoverEl || hoverEl.contains(e.relatedTarget)) return;
      clearTimeout(hoverTimer);
      if (peekMode === 'hover' && !peekEl()?.hidden) closePeek();
      hoverEl = null;
    });
  }

  /* ---- stepping (round 53): a record opened from a list steps to its neighbours in that list's
   * shown order ([ and ], the ↑ ↓ pair beside the id); the list's context is `app.listContext`,
   * set by the list that opened the record and cleared by a navigation elsewhere. A step writes
   * the record's route in place, so Back still returns to the list. ---- */
  function stepping() {
    const c = app.listContext;
    // a list shows no stepper: the pair steps between objects, and the list is the whole
    // (2026-09-21, the Studies audit: `3 / 6` stood over a chooser)
    if (!c || !SAVED_VIEWS.has(app.page) || !routeObjectId() || c.keys.length < 2 || c.index < 0) return null; // N1: a list of one has no neighbour to step to
    return {index: c.index, total: c.keys.length, prev: c.index > 0, next: c.index < c.keys.length - 1};
  }
  function step(delta) {
    if (Window.inspectorMode() === 'task') { // round 66: the inspector's Task steps through the page's Task rows
      const keys = [...document.querySelectorAll('#main .tp-task-list [data-key]')].map((r) => r.dataset.key), i = keys.indexOf(hashParams().get('task') || ''), to = i + delta;
      if (i < 0 || to < 0 || to >= keys.length) return false;
      if (typeof LiveActivity !== 'undefined') LiveActivity.pauseFollowing?.();
      LiveTasks.open(keys[to]);
      return true;
    }
    const c = app.listContext, s = stepping();
    if (!s) return false;
    const to = c.index + delta;
    if (to < 0 || to >= c.keys.length) return false;
    if (typeof LiveActivity !== 'undefined') LiveActivity.pauseFollowing?.();
    c.index = to; c.from = routeObjectId(); c.at = ''; // the next object addressed is the one stepped to (keepListContext)
    Data.openEntry(c.keys[to], false);
    return true;
  }

  /* ---- the Facts mode (round 64): the object's facts — sources, limitations, the declaration —
   * read in the inspector as sections a page declares (`FACTS[page]`), from the `Facts` chip and
   * `···` → Facts; one fold's body (`factsRef`) opens the same way from its text link. ---- */
  const study = {has: () => LiveStudy.hasFacts(), sections: () => LiveStudy.factsSections()};
  const glossary = {has: () => true, sections: () => [glossarySection()]};
  const FACTS = {factor: study, alpha: study, risk: study, portfolio: {has: () => Boolean(Data.raw()?.declaration), sections: () => LiveViews.bookFactsSections()},
    evidence: {has: () => true, sections: () => [...LiveReview.overviewFacts(), glossarySection()]}, 'evidence-stream': {has: () => true, sections: () => [...LiveReview.sourcesFacts(), glossarySection()]}, 'evidence-reading': {has: () => true, sections: () => [...LiveReview.readingFacts(), glossarySection()]}, report: {has: () => true, sections: () => [...LiveReview.reportFacts(), glossarySection()]}, handoff: {has: () => true, sections: () => [...LiveReview.handoffFacts(), glossarySection()]}, 'team-evidence': glossary, ...Object.fromEntries(['goal', 'goal-conversation', 'goal-results'].map((page) => [page, {has: () => LiveGoals.hasFacts(), sections: () => [...LiveGoals.factsSections(), glossarySection()]}]))}; // U23: a goal's exact revision
  const hasFacts = () => Boolean(FACTS[app.page]?.has());
  /* The Record mode (round 67): what the owners recorded about the object — a Task's or a study's
   * observations in recorded order, a session's exchanges as a run log. */
  const studyTask = () => hashParams().get('study') || '';
  const RECORD = {
    tasks: {has: () => Boolean(hashParams().get('task')), body: () => LiveActivity.recordOf(hashParams().get('task'))},
    handoff: {has: () => LiveReview.boundTasks().length > 0, body: () => LiveReview.handoffRecord()}, // round 79: a case's and a handoff's record
    factor: {has: () => Boolean(studyTask()), body: () => LiveActivity.recordOf(studyTask())}, alpha: {has: () => Boolean(studyTask()), body: () => LiveActivity.recordOf(studyTask())}, risk: {has: () => Boolean(studyTask()), body: () => LiveActivity.recordOf(studyTask())},
    portfolio: {has: () => Boolean(Data.subject()?.task_id), body: () => LiveActivity.recordOf(Data.subject().task_id)},
    team: {has: () => Boolean(LiveTeam.summary?.().session), body: () => LiveTeam.recordOf(LiveTeam.summary().session)}, 'team-evidence': {has: () => Boolean(LiveTeam.summary?.().session), body: () => LiveTeam.recordOf(LiveTeam.summary().session)},
  };
  const hasRecord = () => Boolean(RECORD[app.page]?.has());
  /* The panel's head (round 92): the page's kind as the kind line, the object's name as the title. */
  const panelHead = () => { const route = ROUTES[app.page] || [], object = clean($('#main .object-header h1')?.textContent), name = t(route[1] || ''); return {kind: name, title: object || name, sameKind: !object || object === name}; };
  // one tab alone is not a row of tabs: its word joins the kind line instead
  const panelKind = (h, tabs) => { const group = typeof SIDEBAR_GROUPS !== 'undefined' && SIDEBAR_GROUPS[app.page] ? t(SIDEBAR_GROUPS[app.page].word) : t(ROUTES[app.page]?.[0] || ''); const kind = h.sameKind ? group : h.kind; return tabs.length > 1 ? kind : [kind, tabs[0]?.word].filter(Boolean).join(' · '); };
  const panelTabs = (on) => [hasFacts() ? {key: 'facts', word: t('Facts'), on: on === 'facts'} : null, hasRecord() ? {key: 'record', word: t('Record'), on: on === 'record'} : null].filter(Boolean);
  const panelHeader = (on) => { const h = panelHead(), tabs = panelTabs(on); return {title: h.title, kind: panelKind(h, tabs), tabs: tabs.length > 1 ? tabs : null}; };
  const openPanelTab = (key) => (key === 'record' ? openRecord() : openFacts());
  function rememberTab(key) { S.tabByPage = S.tabByPage || {}; S.tabByPage[app.page] = key; }
  /* Ctrl ] and the ⓘ chip: the tab the page last showed, else Facts, else Record. */
  function openPanel() {
    const want = S.tabByPage?.[app.page];
    if (want === 'record' && hasRecord()) return openRecord();
    if (want === 'facts' && hasFacts()) return openFacts();
    return hasFacts() ? openFacts() : hasRecord() ? openRecord() : false;
  }
  function openRecord(by = null) {
    if (!hasRecord()) return false;
    closeDialog();
    const h = panelHead();
    rememberTab('record');
    selectAddressMode('record');
    const tabs = panelTabs('record');
    return Window.openInspector({mode: 'record', readHeader: () => panelHeader('record'), title: h.title, kind: panelKind(h, tabs), tabs: tabs.length > 1 ? tabs : null, body: html`<section class="inspector-section">${RECORD[app.page].body()}</section>`, onClose: () => replaceHash({record: ''}), by: by || (Window.inspectorOpen() ? null : ['record', ''])});
  }
  function refreshRecord() {
    if (Window.inspectorMode() === 'record' && hasRecord()) Window.setInspectorBody(html`<section class="inspector-section">${RECORD[app.page].body()}</section>`);
  }
  /* V667 / PG2: the addressed reader owns restoration, including after connection.
   * Facts and Record wait for their object; neither falls through to a Task reader. */
  const addressModes = [
    {mode: 'facts', key: 'facts', value: '1', pages: Object.keys(FACTS), has: hasFacts, open: () => openFacts()},
    {mode: 'record', key: 'record', value: '1', pages: Object.keys(RECORD), has: hasRecord, open: () => openRecord()},
    {mode: 'reference', key: 'reference', pages: ['goal', 'goal-conversation', 'goal-results', 'goals'], has: () => LiveGoals.selected() && Data.workspaceStatus === 'ready', open: () => LiveGoals.openReference(hashParams().get('reference')), matches: () => LiveGoals.referenceMatchesAddress()},
    {mode: 'task', key: 'task', pages: null, has: () => Data.workspaceStatus === 'ready', open: () => LiveTasks.open(hashParams().get('task'))},
  ];
  function addressedMode() {
    const q = hashParams();
    return addressModes.find((x) => (!x.pages || x.pages.includes(app.page)) && (x.value ? q.get(x.key) === x.value : q.get(x.key)))?.mode || '';
  }
  function selectAddressMode(mode, extra = {}) {
    const x = addressModes.find((row) => row.mode === mode);
    if (!x) return false;
    const clear = Object.fromEntries(addressModes.filter((row) => row.mode !== 'task').map((row) => [row.key, '']));
    replaceHash({...clear, ...(x.value ? {[x.key]: x.value} : {}), ...extra});
    return true;
  }
  function reopenFromAddress() {
    const x = addressModes.find((row) => row.mode === addressedMode());
    const current = Window.inspectorMode();
    if (current === x?.mode && (!x.matches || x.matches())) return;
    if (current && !addressModes.some(row => row.mode === current)) return;
    if (current) Window.closeInspector(false, {nextMode: x?.mode});
    if (x?.has()) return x.open();
  }
  const sections = (list) => html`${list.map((s) => html`<section class="inspector-section"${s.key ? html` data-facts="${s.key}"` : ''}><h3>${s.title}</h3>${s.body}</section>`)}`;
  /* `by` is the press that opened it (law 149: the same press closes it); a tab or a repaint keeps the one before. */
  function openFacts(id = '', by = null) {
    closeDialog();
    const h = panelHead(), tabs = panelTabs('facts'), kind = panelKind(h, tabs), tabRow = tabs.length > 1 ? tabs : null;
    if (id) { // one fact section from its link (a fold's body): the Facts tab holds it alone until the tab is pressed
      const tpl = $(`template[data-facts-id="${id}"]`), opener = tpl?.previousElementSibling;
      if (!tpl) return false;
      S.factsId = id;
      return Window.openInspector({mode: 'facts', readHeader: () => panelHeader('facts'), title: h.title, kind, tabs: tabRow, body: html`<section class="inspector-section"><h3>${tpl.dataset.factsTitle || clean(opener?.textContent)}</h3>${raw(tpl.innerHTML)}</section>`, onClose: () => { S.factsId = ''; }, by: by || ['facts-open', id]}); // N6: a figure that opens facts names them on its template
    }
    const list = factsList();
    if (!list.length) return false;
    S.factsId = '';
    rememberTab('facts');
    selectAddressMode('facts');
    return Window.openInspector({mode: 'facts', readHeader: () => panelHeader('facts'), title: h.title, kind, tabs: tabRow, body: sections(list), onClose: () => { S.factsId = ''; replaceHash({facts: ''}); }, by: by || (Window.inspectorOpen() ? null : ['facts', ''])});
  }
  // the research context first (round 81: the page's workspace reading is a Facts section), then the page's own, then the proofs (round 92)
  const factsList = () => hasFacts() ? [{title: t('Research context'), body: LiveViews.proofBody('context', S.observation)}, ...FACTS[app.page].sections(), ...proofSections()] : [];
  function refreshFacts() {
    if (Window.inspectorMode() === 'facts' && !S.factsId && hasFacts()) Window.setInspectorBody(sections(factsList()));
  }

  /* ---- the shortcuts sheet (round 51): generated from the key map, searchable ---- */
  /* The keyboard sheet's body (round 62): one renderer for the ? sheet (with its search) and the
   * Settings page's Keyboard section (without). */
  function shortcutsBody(search = false) {
    const rows = Controls.keymap(), groups = [...new Set(rows.map((r) => r.group))];
    // a place's chord is named by the dock's word for it (PG4: one word per page)
    const words = (r) => r.words ? t(r.words) : t('Go to {place}', {place: t(SIDEBAR.find((x) => x.page === r.page)?.word || pageWord(r.page))});
    const row = (r) => html`<div class="key-row" data-words="${words(r).toLowerCase()}" data-keys="${r.keys.toLowerCase()}"><span>${words(r)}</span>${keycap(r.keys)}</div>`;
    const sections = groups.map((g) => html`<section class="key-group" data-group><h3>${t(g)}</h3>${rows.filter((r) => r.group === g).map(row)}</section>`);
    return html`${search ? html`<div class="key-search"><label class="sr-only" for="keySearch">${t('Search shortcuts')}</label><div class="input-with-icon">${icon('search')}<input id="keySearch" class="ui-field" type="search" data-key-filter placeholder="${t('Search shortcuts')}" autocomplete="off"></div></div>` : ''}${sections}`;
  }
  function openShortcuts() {
    const body = shortcutsBody(true);
    openDialog(t('Keyboard'), t('Keyboard shortcuts'), body, '', true, () => openShortcuts());
    $('#dialog').classList.add('key-sheet');
    $('#keySearch')?.focus({preventScroll: true});
  }

  /* ---- research map ---- */
  function researchMap() {
    const node = (page, name, sub, ic) => link(html`${icon(ic)}<span><strong>${t(name)}</strong><small>${t(sub)}</small></span>`, page, 'map-node');
    const prepare = html`${node('data', 'Data', 'Working copy', 'data')}${node('inputs', 'Feature', 'Versioned inputs', 'cube')}${node('factor', 'Factor', 'Saved decision', 'lab')}${node('foundation', 'Foundation', 'Immutable handoff', 'archive')}`;
    const models = html`${node('alpha', 'Alpha modeling', 'Candidate research', 'branch')}${node('risk', 'Risk modeling', 'Parallel diagnostic branch', 'evidence')}`;
    const outputs = html`${node('portfolio', 'Portfolio', 'Historical book', 'portfolio')}${node('evidence', 'Evidence', 'Book-specific', 'file')}${node('evidence', 'CRO Review', 'Independent review', 'review')}${node('report', 'Report', 'Limits included', 'file')}`;
    openDialog(t('Research map · capabilities'), t('From inputs to evidence.'), html`<div class="map-section"><div class="map-section-label"><b>${t('01 / PREPARE')}</b>${t('Research inputs')}</div><div class="map-nodes">${prepare}</div></div><div class="map-connector" aria-hidden="true">↓</div><div class="map-section"><div class="map-section-label"><b>${t('02 / MODEL')}</b>${t('Parallel branches')}</div><div class="map-nodes parallel">${models}</div></div><div class="map-connector" aria-hidden="true">↓</div><div class="map-section"><div class="map-section-label"><b>${t('03 / REVIEW')}</b>${t('Evidence and report')}</div><div class="map-nodes">${outputs}</div></div><p class="map-caption">${t('This is the product workflow, not the selected result’s lineage. Alpha and Risk are parallel upstream branches. A report-only Risk attachment is not a sizing input; exact participation is recorded in provenance.')}</p>`, '', false, () => researchMap());
    $('#dialog').classList.add('research-map-dialog');
  }

  return {
    get focus() {
      return S.focus;
    },
    get lens() {
      return S.lens;
    },
    get observation() {
      return S.observation;
    },
    baseline: S.baseline,
    diffValues,
    declarationDiff, recordIcon: (x) => RECORD_ICONS[x.raw?.kind] || 'file',
    observationDock, presetsFor,
    syncObservation,
    selectObservation,
    window: chartWindow,
    setWindow,
    resetWindow,
    presetWindow,
    presetActive,
    readingsMarkup,
    openLens,
    closeLens,
    refreshLens,
    openPanel,
    openPanelTab,
    openFacts,
    refreshFacts,
    hasFacts,
    openRecord,
    refreshRecord,
    hasRecord,
    addressModes,
    addressedMode,
    selectAddressMode,
    reopenFromAddress,
    toggleFocus,
    openQuick,
    openShortcuts,
    shortcutsBody,
    peek,
    closePeek,
    openPeeked,
    hoverCard,
    bindHover,
    HOVER_LINKS,
    peekOpen,
    peekRowIs,
    peekFollow,
    stepping,
    step,
    quickOpenActive,
    quickKeydown,
    drawCommands,
    commandPick,
    researchMap,
  };
})();
