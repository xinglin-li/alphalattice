/* Portfolio study: synthetic performance, chart with an observation cursor, holdings inspector,
 * diagnostics and comparison. */
const holdingSortHead = (key, label, type, unit = '') => {
  const raw = String(app.holdingsSort || 'listing-asc');
  const order = raw === 'source' ? 'listing-asc' : ['weight','target','execution'].includes(raw) ? 'weight-desc' : /^(target|execution)-(asc|desc)$/.test(raw) ? raw.replace(/^(target|execution)-/, 'weight-') : raw;
  const [active, direction] = order.split('-');
  const selected = active === key;
  const indicator = selected ? (direction === 'asc' ? '↑' : '↓') : '↕';
  return {label:btnAttrs(html`${label}<span class="table-sort-indicator" aria-hidden="true">${indicator}</span>`,'holding-sort',key,'table-sort',html`aria-label="${t('Sort by {label}',{label})}"`),unit,type,ariaSort:selected ? (direction === 'asc' ? 'ascending' : 'descending') : 'none'};
};
// Round 60: words first, figures last; every word here is short (a code, a sector), so the table
// spreads its slack evenly between the facts (`spread`) instead of leaving one stretch of blank.
// The index is its own column, the row's place in the shown order (FT6): the owner's holdings carry no index, and a
// ticker's digits are none (AES, WDC); the unit is in the cell.
const HOLDINGS_HEADERS = (weightLabel) => [{label:'#',type:'num',index:true}, holdingSortHead('listing',t('Listing'),'link'), {label:t('Sector'),type:'text'}, {label:t('Evidence mapping'),type:'status'}, holdingSortHead('weight',t(weightLabel),'num'), holdingSortHead('change',t('Change (pp)'),'num'), {label:'',type:'link'}];
const HOLDINGS_ORDERS = () => [['listing-asc',t('Listing ↑')],['weight-desc',t('Weight ↓')],['weight-asc',t('Weight ↑')]];
/* A holding is selected by its durable listing identity; the ticker is a label that may repeat. */
const holdingKey = (x) => x.listing_id ?? x.ticker;
/* Listings in the order of their labels, the identity breaking a tie (two listings may share a ticker). */
const byListing = (a, b) => String(a.ticker ?? '').localeCompare(String(b.ticker ?? '')) || String(holdingKey(a)).localeCompare(String(holdingKey(b)));

const Portfolio = (() => {
  let sessionAnchor = null;
  /* Sector and review mapping are per-holding facts. A column the owner supplied for no holding
   * of the shown book and date says nothing row by row; the absence is stated once under the
   * table instead. As soon as one holding has the fact, every row shows its own value again.
   * Unknown is never turned into a value. */
  const holdingsContext = () => {
    const forward = Data.performanceMode() === 'forward', raw = Data.raw() || {}, projection = forward ? raw.forward_holdings : null;
    const basis = forward ? projection?.basis : raw.holdings_basis?.basis;
    const weightLabel = basis === 'CONDITIONAL_ESTIMATE' ? 'Proposed weight' : basis === 'OBSERVED_RESEARCH_ENTRY' ? 'Research-entry weight' : basis === 'HISTORICAL_REPLAY' ? 'Replay weight' : 'Holdings weight';
    return {forward, projection, basis, weightLabel, rows: forward ? (projection?.available === true ? projection.holdings || [] : []) : Data.holdings()};
  };
  const ownerHoldings = () => holdingsContext().rows;
  const sectorsAbsent = () => ownerHoldings().length > 0 && ownerHoldings().every((x) => !x.sector);
  const mappingAbsent = () => ownerHoldings().length > 0 && ownerHoldings().every((x) => x.mapped == null);
  const uniformAbsence = () => sectorsAbsent() && mappingAbsent();
  // round 57: how the table is shown — grouping by sector and the two fact columns are offered
  // only where the owner supplied the fact for some holding; the ordering is the head glyphs' too
  const DISPLAY = () => ({
    groupings: sectorsAbsent() ? [] : [['none', t('No grouping')], ['sector', t('Sector')]],
    orderings: HOLDINGS_ORDERS(),
    properties: [...(sectorsAbsent() ? [] : [['sector', t('Sector')]]), ...(mappingAbsent() ? [] : [['mapping', t('Evidence mapping')]])],
    apply: (state) => { app.holdingsSort = state.order; app.holdingsPage = 0; refreshHoldings(); },
  });
  const display = () => displayState('holdings', DISPLAY());
  const sectorShown = () => !sectorsAbsent() && display().props.sector !== false;
  const mappingShown = () => !mappingAbsent() && display().props.mapping !== false;
  /* Exposure by sector: the owner-supplied current weights summed per sector, largest first. */
  function sectorExposure() {
    const totals = new Map();
    for (const h of ownerHoldings()) { if (!h.sector || h.current_weight == null || !Number.isFinite(Number(h.current_weight))) continue; totals.set(h.sector, (totals.get(h.sector) || 0) + Number(h.current_weight) * 100); }
    return [...totals.entries()].sort((a, b) => b[1] - a[1]);
  }
  function sectorSummary() {
    const context = holdingsContext();
    const rows = sectorExposure();
    if (!rows.length) return '';
    const s = (context.forward ? context.projection?.sectors : Data.raw()?.sectors) || {};
    const date = context.forward ? (context.basis === 'OBSERVED_RESEARCH_ENTRY' ? context.projection?.entry_session : context.projection?.formation_session) : Data.subject()?.session || app.session;
    // N6 (law 89, the book's one exposure figure): every sector a slice of the one meter, its legend the list; the source is the label's (i)
    // U52: the Sector map's time treatment is the Host's statement (Research timing), never this page's claim -- the
    // old words said "not the current classification", which the Panel's CURRENT_CLASSIFICATION_BACKFILLED contradicts
    const stated = ((Data.raw()?.timing?.temporal_scope || Data.raw()?.temporalScope)?.statements || []).length; // an authored book's timing, an installed book's reading context
    const source = s.status === 'BOOK_EXECUTION' ? t(stated ? 'Sectors are the classification the book\'s own execution used: the Sector map sealed with its Panel; Research timing states its time treatment.' : 'Sectors are the classification the book\'s own execution used: the Sector map sealed with its Panel.') : t('Sector source not resolved.');
    return figureBox(html`${t('Exposure by sector')} <span class="num">${count(rows.length)}</span>`,meter({kind:'composition',slices:rows,limit:rows.length > LEGEND.fold ? LEGEND.shown : rows.length,label:t('Sector composition')}),{info:infoMark(`${t('book weight at {date} · {n} sectors', {date, n: count(rows.length)})}. ${source}`),cls:'sector-exposure',attrs:html`aria-label="${t('Exposure by sector')}"`});
  }
  function holdingValues() {
    const q = app.holdingsQuery.toLowerCase();
    let values = ownerHoldings().filter((x) => !q || [x.ticker, x.name, x.listing_id].filter(Boolean).join(' ').toLowerCase().includes(q));
    const selected = app.holdingsSort === 'source' ? 'listing-asc' : ['weight','target','execution'].includes(app.holdingsSort) ? 'weight-desc' : app.holdingsSort;
    const [key, direction] = String(selected).split('-'), sign = direction === 'asc' ? 1 : -1;
    return [...values].sort((a,b) => {
      if (key === 'listing') return byListing(a,b);
      const field = key === 'change' ? 'weight_change' : 'current_weight', av = a[field] == null ? NaN : Number(a[field]), bv = b[field] == null ? NaN : Number(b[field]);
      return Number.isFinite(av) && Number.isFinite(bv) ? sign * (av - bv) || byListing(a,b) : byListing(a,b);
    });
  }
  function holdingsRows(values, start = 0) {
    const noSector = !sectorShown(), noMapping = !mappingShown();
    let place = start; // the row's place as the page shows it, its groups' rows in turn
    const row = (x) => html`<tr data-holding-row="${(holdingKey(x))}" data-key="${(holdingKey(x))}"><td class="num muted">${count(++place)}</td><td><button class="holding-name" data-action="holding" data-value="${(holdingKey(x))}" data-row-press aria-label="${t('Inspect {listing}', {listing: x.ticker})}"><span><strong>${(x.ticker)}</strong>${x.name ? html`<small>${x.name}</small>` : ''}</span></button></td>${noSector ? '' : html`<td>${x.sector ? x.sector : t('Not supplied')}</td>`}${noMapping ? '' : html`<td>${stateLine(x.mapped ? 'partial' : 'metadata', {word: x.mapped == null ? t('Not evaluated') : x.mapped ? t('Issuer reviewed') : t('Not mapped')})}</td>`}<td class="num">${x.current_weight == null || !Number.isFinite(Number(x.current_weight)) ? '' : num(Number(x.current_weight) * 100, 'percent')}</td><td class="num">${x.weight_change == null || !Number.isFinite(Number(x.weight_change)) ? '' : signed(Number(x.weight_change) * 100, 'pp')}</td><td>${btnAttrs(icon('chevron'), 'holding', holdingKey(x), 'icon-btn compact', html`aria-label="${t('Inspect {listing}', {listing: x.ticker})}"`)}</td></tr>`;
    if (display().group !== 'sector') return html`${values.map(row)}`;
    const groups = new Map(); // the page's rows under their sector, in the order they arrive
    for (const x of values) { const g = x.sector || t('Not supplied'); if (!groups.has(g)) groups.set(g, []); groups.get(g).push(x); }
    const span = 5 + (noSector ? 0 : 1) + (noMapping ? 0 : 1);
    return html`${[...groups].map(([g, xs]) => html`<tr class="table-group"><th colspan="${span}" scope="colgroup">${g}<b class="num">${xs.length}</b></th></tr>${xs.map(row)}`)}`;
  }
  function holdingsTable() {
    const hidden = [sectorShown() ? null : t('Sector'), mappingShown() ? null : t('Evidence mapping')].filter(Boolean);
    const {weightLabel} = holdingsContext();
    const headers = HOLDINGS_HEADERS(weightLabel).filter((h) => !hidden.includes(h?.label ?? h));
    const values = holdingValues(), {shown, start, page, pages} = pageOf(values, app.holdingsPage) /* the one table page (law 92) */;
    app.holdingsPage = page;
    const foot = pages > 1 ? pager({total: values.length, one: '{n} entry', many: '{n} entries', page, pages, prev: ['holding-page-prev', ''], next: ['holding-page-next', '']}) : ''; // N6: one page of holdings needs no foot (the tab says how many)
    const completeWeights = values.length > 0 && values.every((x) => x.current_weight != null && Number.isFinite(Number(x.current_weight)));
    const total = ['', t('Total'), ...(sectorShown() ? [''] : []), ...(mappingShown() ? [''] : []), completeWeights ? num(values.reduce((sum,x)=>sum+Number(x.current_weight),0)*100, 'percent') : '', '',''];
    return html`${table(headers, holdingsRows(shown, start), '', {classes:uniformAbsence() ? 'compact holdings-compact' : 'compact', report:true, spread:true, countLine:false, count:values.length, foot:total})}${foot}`;
  }
  function sortHoldings(key) {
    const current = app.holdingsSort === 'source' ? 'listing-asc' : ['weight','target','execution'].includes(app.holdingsSort) ? 'weight-desc' : app.holdingsSort;
    const [active,direction] = String(current).split('-');
    app.holdingsSort = `${key}-${active === key ? (direction === 'asc' ? 'desc' : 'asc') : key === 'listing' ? 'asc' : 'desc'}`;
    setDisplay('holdings', {order: app.holdingsSort}); // the head glyph and the popover are one choice
    app.holdingsPage = 0;
    refreshHoldings();
  }
  /* The shown session's own facts, apart from the whole-report metrics: what the policy decided
   * that day and what that one session cost and returned. Read from the verified display body;
   * an absent fact stays absent. */
  const UNIT_WORDS = {names: 'names held', 'portfolio weight': 'of portfolio weight', 'portfolio fraction': 'of the portfolio', fraction: 'a simple return'}; // the units the Host names (POSITION_UNITS), in words
  function sessionFacts() {
    if (holdingsContext().forward) return '';
    const r = Data.raw(), p = r?.position && {...r.position, units: r.metricUnits};
    if (!p) return '';
    // Session-level fractions are small: the shared rule keeps a tiny nonzero cost or return in words, not as zero.
    const pct = (v) => num(v, 'percent');
    const mode = {REBALANCE: 'rebalance', HOLD: 'hold'}[p.decisionMode] || (p.decisionMode ? String(p.decisionMode).toLowerCase() : '');
    // U58 (V343): each fact's unit as the book's readback names it by path, in words; a unit it does not name is not claimed
    const unit = (key) => { const u = p.units?.['position.' + key]; return u ? t(UNIT_WORDS[u] || u) : ''; };
    const daily = (key) => unit(key) ? t('{unit}, this session', {unit: unit(key)}) : t('this session');
    return html`<div class="stat-strip rail session-facts" aria-label="${t('Facts of the selected session')}">${stat(t('Decision at {date}', {date: p.session || app.session}), t(mode), t('what the policy did this session'))}${stat(t('Holdings'), p.holdingCount ?? '', unit('holding_count'))}${stat(t('One-way turnover'), num(p.turnover, 'percent'), daily('one_way_turnover'))}${stat(t('Cost'), num(p.costFraction, 'percent'), daily('cost_fraction'))}${stat(t('Net return'), html`<span class="num ${Number(p.netReturn) > 0 ? 'gain' : Number(p.netReturn) < 0 ? 'loss' : ''}">${signed(p.netReturn, 'percent')}</span>`, html`${daily('net_simple_return')}${p.benchmarkReturn === null || p.benchmarkReturn === undefined ? '' : html` · ${t('benchmark')} ${pct(p.benchmarkReturn)}`}`)}${stat(t('Cash'), num(p.cash, 'percent'), unit('cash'))}</div>`;
  }
  // The holdings date sits with the list it governs, beside the Holdings / Diagnostics tabs.
  const sessionSubject = () => html`<div class="session-selector"><label id="holdingsSessionLabel" for="holdingsSession">${t('Holdings session')}</label><div class="session-controls">${btn(icon('back'), 'session-prev', '', 'icon-btn')}${picker('holdingsSession', Data.sessions().map((x) => [x, x]), {selected: app.session, labelId: 'holdingsSessionLabel', kind: 'text'})}${btn(icon('arrow'), 'session-next', '', 'icon-btn')}</div></div>`;
  const forwardDates = () => {
    const metadata = holdingsContext().projection || {};
    const clauses = [];
    if (metadata.formation_session) clauses.push(`${t('Formation')} ${metadata.formation_session}`);
    if (metadata.entry_session) clauses.push(`${t('Entry')} ${metadata.entry_session}`);
    return clauses.length ? html`<p class="caption">${clauses.join(' · ')}</p>` : '';
  };
  const holdingsToolbar = () => holdingsContext().forward ? forwardDates() : sessionSubject();
  function tabs() {
    const context = holdingsContext(), raw = Data.raw() || {};
    const ownerCount = context.forward ? context.projection?.holding_count : raw.position?.holdingCount;
    const shownCount = ownerCount != null && Number.isInteger(Number(ownerCount)) ? count(Number(ownerCount)) : '';
    const items = [['holdings', 'Holdings', shownCount], ['diagnostics', 'Diagnostics', '']]; // N2: Compare is Portfolio study's tab (law 132); the replay union includes exited rows
    return tabStrip(t('Portfolio detail views'), items.map(([key, label, n]) => ({word: t(label), action: 'portfolio-tab', value: key, on: app.tab === key, count: n})), 'portfolio-tabs');
  }
  function holdingsBasisNote() {
    const context = holdingsContext(), metadata = context.forward ? context.projection : Data.raw()?.holdings_basis;
    if (context.basis === 'HISTORICAL_REPLAY') {
      const clauses = [];
      if (metadata?.formation_session) clauses.push(t('Historical replay weights at {date}.', {date: metadata.formation_session}));
      const comparison = metadata?.comparison_basis === 'PRECEDING_FORMATION' && metadata?.preceding_formation_session
        ? t('Change compares preceding formation {date}; replay drift is included, not actual trades.', {date: metadata.preceding_formation_session})
        : t({
        PRECEDING_FORMATION: 'Change compares the preceding formation; replay drift is included, not actual trades.',
        SEALED_CONTINUATION_BOUNDARY: 'Change compares the sealed continuation boundary; it is not an actual trade record.',
        PRECEDING_NOT_RECORDED: 'The preceding weight was not recorded; no change is shown.',
      }[metadata?.comparison_basis] || '');
      if (comparison) clauses.push(comparison);
      return clauses.join(' ');
    }
    if (context.basis === 'CONDITIONAL_ESTIMATE') {
      const clauses = [];
      if (metadata?.formation_session) clauses.push(t('Conditional estimated weights at {date}; they are not actual fills.', {date: metadata.formation_session}));
      const comparison = {FORMATION_CLOSE_ESTIMATE: 'Change compares formation-close estimates.', PRECEDING_NOT_RECORDED: 'The preceding weight was not recorded; no change is shown.'}[metadata?.comparison_basis];
      if (comparison) clauses.push(t(comparison));
      return clauses.join(' ');
    }
    if (context.basis === 'OBSERVED_RESEARCH_ENTRY') {
      const clauses = [t('Observed research-entry weights come from daily-bar QA, not actual fills.')];
      if (metadata?.entry_session) clauses.push(t('Research entry {date}.', {date: metadata.entry_session}));
      const comparison = {OBSERVED_PRETRADE_WEIGHTS: 'Change compares the observed pretrade weights.', PRECEDING_NOT_RECORDED: 'The preceding weight was not recorded; no change is shown.'}[metadata?.comparison_basis];
      if (comparison) clauses.push(t(comparison));
      return clauses.join(' ');
    }
    return '';
  }
  function details() {
    if (app.tab !== 'holdings') return LiveViews.portfolioDetails(app.tab);
    app.holdingsSort = display().order; // the viewer's ordering (round 57)
    // round 95 (the user's still): the search and the display options stand over the table they act on, under the session's facts and the sector exposure
    const context = holdingsContext(), basisNote = holdingsBasisNote();
    const noForwardRows = context.forward && context.projection && (context.projection.available !== true || !context.rows.length);
    const table = noForwardRows ? emptyState(t('No forward holdings are available.')) : holdingsTable();
    const absence = noForwardRows || context.forward ? '' : uniformAbsence() ? t('Sector and review mapping are not supplied for any holding of this book at this date; nothing is inferred.') : mappingAbsent() ? t('Review mapping is not evaluated for any holding of this book at this date; the sector is the book\'s own classification.') : t('Sector and review mapping are not inferred.');
    const note = [basisNote,absence].filter(Boolean).join(' ');
    return html`<section class="holdings-section" aria-label="${t(context.forward ? 'Forward holdings' : 'Historical holdings')}">${sessionFacts()}${sectorSummary()}${detailSplit(html`<div class="holdings-box" data-box="table">${searchBar('holdingsQuery', t('Find listing'), t('Find a listing…'), app.holdingsQuery, displayOptions('holdings', DISPLAY()))}<div class="holdings-table-area"><div id="holdingsTable">${table}</div>${note ? html`<p class="table-note">${note}</p>` : ''}</div></div>`, 'holding', {over: true})}</section>`;
  }
  /* A holding is a detail of the window (law 149): the window's column from the pane step up, the
   * pane's layer below it; the same press closes it. It follows the holdings date: a new session
   * shows the holding's weights there, and a book that does not hold it at that date closes it. */
  const holdingOf = (id) => ownerHoldings().find((x) => holdingKey(x) === id) || null;
  function inspectHolding(id) {
    const h = holdingOf(id);
    if (!h) return;
    const index = holdingValues().findIndex((x) => holdingKey(x) === id);
    if (index >= 0 && Math.floor(index / LIST_PAGE) !== app.holdingsPage) { app.holdingsPage = Math.floor(index / LIST_PAGE); refreshHoldings(); }
    app.holdingFocus = id;
    Window.openInspector({mode: 'holding', readHeader: () => ({title: h.ticker, kind: t('Holding · read-only')}), title: h.ticker, kind: t('Holding · read-only'), body: LiveViews.holdingDetail(h), by: ['holding', id], onClose: () => { app.holdingFocus = null; }});
  }
  function followHolding() {
    if (!app.holdingFocus || Window.inspectorMode() !== 'holding') return;
    const h = holdingOf(app.holdingFocus);
    if (h) Window.setInspectorBody(LiveViews.holdingDetail(h));
    else Window.closeInspector(false);
  }
  function refreshHoldings() {
    if ($('#holdingsTable')) $('#holdingsTable').innerHTML = holdingsTable();
  }
  function pageHoldings(delta) {
    const pages = Math.max(1, Math.ceil(holdingValues().length / LIST_PAGE));
    app.holdingsPage = Math.max(0, Math.min(pages - 1, app.holdingsPage + delta));
    refreshHoldings();
    $('#holdingsTable')?.scrollIntoView({block: 'nearest'});
  }
  function setTab(v) {
    if (!['holdings', 'diagnostics'].includes(v)) return;
    app.tab = v;
    const root = $('.portfolio-detail-surface');
    if (root) {
      const top = scrollY;
      root.innerHTML = html`${tabs()}<div id="portfolioDetail">${details()}</div>`;
      scrollTo({top, behavior: 'instant'});
    }
  }

  /* ---- holdings session ---- */
  function syncSessionButtons() {
    const session = Data.portfolioSession();
    $$('[data-action="session-prev"]').forEach((x) => {
      x.setAttribute('aria-label', t('Previous holdings session'));
      x.disabled = session === Data.sessions()[0];
    });
    $$('[data-action="session-next"]').forEach((x) => {
      x.setAttribute('aria-label', t('Next holdings session'));
      x.disabled = session === Data.sessions()[Data.sessions().length - 1];
    });
  }
  function setSession(date) {
    if (!Data.sessions().includes(date) || date === Data.portfolioSession()) return;
    const place = {...(sessionAnchor || {x: typeof scrollX === 'number' ? scrollX : 0, y: typeof scrollY === 'number' ? scrollY : 0}), restoreControl: true};
    sessionAnchor = null;
    return Data.openPortfolio(Data.subject().task_id, date, 'portfolio', app.compareOther || null, place, Data.performanceMode()); // the chosen pair survives a date change; the comparison is re-read at that date
  }
  function setPerformanceMode(mode) {
    if (!['historical', 'forward'].includes(mode) || mode === Data.performanceMode()) return;
    return Data.openPortfolio(Data.subject().task_id, Data.subject().session || null, 'portfolio', app.compareOther || null, null, mode);
  }
  function rememberSessionPlace(force = false) {
    if (sessionAnchor && !force) return;
    sessionAnchor = {x: typeof scrollX === 'number' ? scrollX : 0, y: typeof scrollY === 'number' ? scrollY : 0};
  }
  function stepSession(delta) {
    const i = Data.sessions().indexOf(Data.portfolioSession());
    if (i + delta >= 0 && i + delta < Data.sessions().length) return setSession(Data.sessions()[i + delta]);
  }
  /* A holdings-date read keeps the whole-report chart and the reader's place mounted. Only the
   * session surface is replaced after the owner answers; a refused read restores the old date. */
  function refreshSessionReading() {
    const surface = $('.portfolio-detail-surface');
    if (Data.portfolioReading()) surface?.setAttribute('aria-busy', 'true');
    else surface?.removeAttribute('aria-busy');
    if (typeof Picker !== 'undefined') Picker.select('holdingsSession', Data.subject()?.session || app.session);
    syncSessionButtons();
  }
  function refreshSessionSurface(place = null) {
    const surface = $('.portfolio-detail-surface');
    const top = place?.y ?? (typeof scrollY === 'number' ? scrollY : 0), left = place?.x ?? (typeof scrollX === 'number' ? scrollX : 0);
    const refocus = document.activeElement?.id === 'holdingsSession';
    if (surface) surface.innerHTML = html`${tabs()}<div class="holdings-toolbar">${holdingsToolbar()}</div><div id="portfolioDetail">${details()}</div>`;
    const timing = $('#portfolioTiming');
    if (timing) fillStackSlot(timing, LiveViews.researchTiming(Data.raw()?.timing));
    const standing = $('#portfolioStanding');
    if (standing) fillStackSlot(standing, standingPanel(Data.raw()?.standing)); // U59: the book's marks, beside its timing
    $$('[data-session-label]').forEach((x) => (x.textContent = app.session));
    const sessionControl = $('#holdingsSession');
    refreshSessionReading();
    if (place?.restoreControl && top === 0) sessionControl?.scrollIntoView({block: 'nearest', behavior: 'instant'});
    else if (typeof scrollTo === 'function') scrollTo({top, left, behavior: 'instant'});
    if (refocus && sessionControl) Geometry.focusQuietly(sessionControl);
    followHolding();
    if (Inspect.lens) Inspect.refreshLens();
    if (typeof Controls !== 'undefined') Controls.sync();
    if (typeof Geometry !== 'undefined') Geometry.schedule();
  }
  function refreshPerformanceSurface() {
    const surface = $('#portfolioPerformance');
    if (!surface) return;
    const top = typeof scrollY === 'number' ? scrollY : 0;
    const focus = document.activeElement?.dataset?.action === 'portfolio-performance';
    surface.outerHTML = performancePanel();
    bindCharts();
    if (focus) $(`[data-action="portfolio-performance"][data-value="${Data.performanceMode()}"]`)?.focus({preventScroll:true});
    if (typeof scrollTo === 'function') scrollTo({top, behavior:'instant'});
    if (typeof Geometry !== 'undefined') Geometry.schedule();
  }

  /* ---- chart ---- */
  function chartToolbar() {
    if (Data.performanceMode() === 'forward') return html`<div class="chart-toolbar"><div class="chart-heading"><h2>${t('Daily realized returns')}</h2><span class="chart-legend"><span><i class="legend-line"></i>${t('Realized strategy')}</span></span></div><p class="caption">${t('Daily bar QA returns; not verified venue execution.')}</p></div>`;
    const rolling = Boolean(Data.rollingPerformance());
    return html`<div class="chart-toolbar"><div class="chart-heading"><h2>${app.chart === 'daily' ? t('Daily returns') : t(rolling ? 'Rolling performance' : 'Performance')}</h2><span class="chart-legend"><span><i class="legend-line"></i>${t(rolling ? 'Recorded outcomes' : 'Study')}</span><span><i class="legend-line benchmark"></i>${t('Benchmark')}</span></span></div><div class="chart-controls">${rangeSegments()}<div class="segmented ui-segments" aria-label="${t('Chart measure')}">${segBtn(t('Indexed'), 'chart', 'indexed', app.chart === 'indexed')}${segBtn(t('Daily returns'), 'chart', 'daily', app.chart === 'daily')}</div></div></div>`;
  }
  /* The ranges (round 95, Stocks' row): beside Indexed / Daily when the series is long enough to
   * need them; a shorter series has the navigator alone. */
  function rangeSegments() {
    const rows = Data.series(), presets = Inspect.presetsFor(rows);
    if (!presets.length) return '';
    const [a, b] = Inspect.window(rows.length);
    return html`<div class="segmented ui-segments chart-ranges" aria-label="${t('Chart range')}">${presets.map(([key, label]) => segBtn(t(label), 'chart-range', key, Inspect.presetActive(key, rows, a, b)))}</div>`;
  }
  /* The hero as the readout (round 95, Apple Stocks): at rest the window's figure and its dates;
   * under the pointer or the keys the study's figure at that observation, the date and the
   * benchmark's figure on the line. */
  function heroTexts(i = null) {
    const rows = Data.series(), [a, b] = Inspect.window(rows.length), daily = Data.performanceMode() === 'forward' || app.chart === 'daily';
    // the indexed figure is the change from the window's base (the row before it, else 100), as the readings under the navigator read it
    const base = (key) => (a > 0 ? Number(rows[a - 1]?.[key]) : 100);
    const change = (r, key, dailyKey) => (daily ? signed(r?.[dailyKey], 'percent') : signed(Number.isFinite(Number(r?.[key])) && base(key) > 0 ? (Number(r[key]) / base(key) - 1) * 100 : null, 'percent'));
    const study = (r) => change(r, 'value', 'daily'), bench = (r) => change(r, 'benchmark', 'benchmarkDaily');
    if (i === null) return {value: study(rows[b]), span: dateRange(rows[a]?.date, rows[b]?.date)};
    const r = rows[i];
    return {value: study(r), span: html`${r?.date || ''} · ${t('Benchmark')} ${bench(r)}`};
  }
  function heroShow(i = null) {
    const head = $('.performance-surface .chart-headline'); if (!head) return;
    const {value, span} = heroTexts(i);
    head.querySelector('strong').innerHTML = html`${value}`;
    head.querySelector('small').innerHTML = html`${span}`;
    head.classList.toggle('is-reading', i !== null);
  }
  const chartUnder = () => {
    const s = Data.series();
    return t(Data.performanceMode() === 'forward' || app.chart === 'daily' ? 'Daily return (%) · {from} — {to}' : 'Indexed series · base 100 · {from} — {to}', {from: s[0]?.date || '', to: s[s.length - 1]?.date || ''});
  };
  function chartBlock() {
    const {value, span} = heroTexts();
    const forward = Data.performanceMode() === 'forward', rolling = Boolean(Data.rollingPerformance());
    return html`${chartHead(value, forward ? t('Realized strategy') : t(rolling ? 'Recorded outcomes' : 'Study'), span)}${chart(forward ? 'daily' : app.chart)}`;
  }
  function setChart(v) {
    app.chart = v;
    const surface = $('.performance-surface');
    if (!surface) return render();
    const pos = scrollY;
    $('#performanceChart').innerHTML = chartBlock();
    $$('[data-action="chart"]').forEach((x) => x.setAttribute('aria-pressed', String(x.dataset.value === v)));
    const h = $('.chart-heading h2');
    if (h) h.textContent = v === 'daily' ? t('Daily returns') : t('Performance');
    const under = $('.performance-surface .chart-under>span');
    if (under) under.textContent = chartUnder();
    bindCharts();
    scrollTo({top: pos, behavior: 'instant'});
  }
  /* The chart redrawn for the window in place: the SVG, its readers, the navigator's window,
   * the readings and the preset that matches. No page repaint; the scroll stays. */
  function redrawChart() {
    const box = $('#performanceChart');
    if (!box) return;
    box.innerHTML = chartBlock();
    bindCharts();
    navigatorLayout();
  }
  function navigatorLayout() {
    const nav = $('[data-navigator]');
    if (!nav) return;
    const plot = $('#performanceChart [data-chart]');
    const drawn = plot && CHART_ROWS.get(plot.dataset.chartId || 'portfolio');
    // A pending view read keeps the prior chart mounted. Its navigator measures the rows
    // and window that chart drew, as fitCharts does, until the accepted surface replaces it.
    const rows = drawn?.rows, [a, b] = drawn?.window || [];
    if (!Array.isArray(rows) || !rows.length || !Number.isInteger(a) || !Number.isInteger(b)
      || a < 0 || b < a || b >= rows.length || !rows[a]?.date || !rows[b]?.date) return;
    const n = rows.length;
    // The rail spans exactly the plot area as drawn (the SVG keeps its aspect and centres its
    // drawing), measured from the DOM, so a handle sits under the date it names.
    const rail = nav.querySelector('[data-rail]');
    if (plot) {
      const p = plot.getBoundingClientRect(), d = nav.getBoundingClientRect();
      if (p.width > 0 && d.width > 0) {
        rail.style.marginLeft = Math.max(0, p.left - d.left).toFixed(1) + 'px';
        rail.style.marginRight = Math.max(0, d.right - p.right).toFixed(1) + 'px';
      }
    }
    const at = (i) => ((i / Math.max(1, n - 1)) * 100).toFixed(3) + '%';
    const win = nav.querySelector('[data-window]');
    win.style.left = at(a);
    win.style.width = (((b - a) / Math.max(1, n - 1)) * 100).toFixed(3) + '%';
    nav.querySelector('[data-shade-start]').style.width = at(a);
    nav.querySelector('[data-shade-end]').style.width = (((n - 1 - b) / Math.max(1, n - 1)) * 100).toFixed(3) + '%';
    for (const handle of nav.querySelectorAll('[data-handle]')) {
      const i = handle.dataset.handle === 'start' ? a : b;
      handle.setAttribute('aria-valuenow', String(i));
      handle.setAttribute('aria-valuetext', rows[i].date);
    }
    nav.querySelector('[data-window-start]').textContent = rows[a].date;
    nav.querySelector('[data-window-end]').textContent = rows[b].date;
    nav.querySelector('[data-readings]').innerHTML = Inspect.readingsMarkup(rows, a, b);
    for (const button of $$('.performance-surface [data-action="chart-range"]')) button.setAttribute('aria-pressed', String(Inspect.presetActive(button.dataset.value, rows, a, b)));
    Inspect.syncObservation();
  }
  /* The navigator's pointer and keyboard: drag a handle, pan the window, press outside it to
   * bring the nearer handle there, arrows on a focused handle. Every move sets the window. */
  function bindNavigator() {
    const nav = $('[data-navigator]');
    if (!nav || nav.dataset.bound) return;
    nav.dataset.bound = 'true';
    const rail = nav.querySelector('[data-rail]'), win = nav.querySelector('[data-window]');
    const count = () => Data.series().length;
    const indexAt = (clientX) => { const box = rail.getBoundingClientRect(); return Math.round(Math.max(0, Math.min(1, (clientX - box.left) / Math.max(1, box.width))) * (count() - 1)); };
    let drag = null;
    rail.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || !count()) return;
      const [a, b] = Inspect.window(count()), i = indexAt(e.clientX), handle = e.target.closest('[data-handle]');
      if (handle) drag = {mode: handle.dataset.handle, a, b};
      else if (e.target.closest('[data-window]')) drag = {mode: 'pan', a, b, from: i};
      else { const mode = Math.abs(i - a) <= Math.abs(i - b) ? 'start' : 'end'; drag = {mode, a, b}; Inspect.setWindow(mode === 'start' ? i : a, mode === 'end' ? i : b); }
      rail.setPointerCapture(e.pointerId);
      win.classList.add('is-dragging');
      e.preventDefault();
    });
    rail.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const i = indexAt(e.clientX), n = count();
      if (drag.mode === 'start') Inspect.setWindow(Math.min(i, drag.b - 1), drag.b);
      else if (drag.mode === 'end') Inspect.setWindow(drag.a, Math.max(i, drag.a + 1));
      else { const span = drag.b - drag.a, a = Math.max(0, Math.min(n - 1 - span, drag.a + (i - drag.from))); Inspect.setWindow(a, a + span); }
    });
    const release = () => { drag = null; win.classList.remove('is-dragging'); };
    rail.addEventListener('pointerup', release);
    rail.addEventListener('pointercancel', release);
    rail.addEventListener('dblclick', () => Inspect.setWindow(0, count() - 1));
    for (const handle of nav.querySelectorAll('[data-handle]')) {
      handle.addEventListener('keydown', (e) => {
        const n = count(), [a, b] = Inspect.window(n), start = handle.dataset.handle === 'start';
        const step = e.shiftKey ? 21 : 1, page = 63;
        const moves = {ArrowLeft: -step, ArrowRight: step, ArrowDown: -step, ArrowUp: step, PageDown: -page, PageUp: page};
        let next = null;
        if (e.key in moves) next = (start ? a : b) + moves[e.key];
        else if (e.key === 'Home') next = start ? 0 : a + 1;
        else if (e.key === 'End') next = start ? b - 1 : n - 1;
        if (next === null) return;
        e.preventDefault();
        if (start) Inspect.setWindow(Math.min(next, b - 1), b); else Inspect.setWindow(a, Math.max(next, a + 1));
        handle.focus({preventScroll: true});
      });
    }
    if (!bindNavigator.resizing) { bindNavigator.resizing = true; let raf = 0; addEventListener('resize', () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(navigatorLayout); }, {passive: true}); } // round 94: once per frame
    navigatorLayout();
  }
  /* Pointer, keyboard and click interaction on every chart; the cursor is a reading position. */
  function bindCharts() {
    fitCharts();
    if (!bindCharts.resizing) {
      bindCharts.resizing = true;
      let settle = 0;
      addEventListener('resize', () => { clearTimeout(settle); settle = setTimeout(() => { if (fitCharts()) bindCharts(); }, 150); }, {passive: true});
    }
    for (const target of $$('[data-chart]')) {
      const id = target.dataset.chartId || 'portfolio', spec = CHART_ROWS.get(id);
      if (!spec) continue;
      const {rows, lines, window: [a, b]} = spec, n = b - a + 1, own = id === 'portfolio', valueOf = spec.value || fmt;
      const wrap = target.closest('[data-chart-wrap]');
      const tip = wrap.querySelector('[data-tooltip]');
      const cross = wrap.querySelector('[data-cross]');
      let index = own ? Math.max(a, Math.min(b, Inspect.observation)) : b;
      const position = (i) => {
        const w = Number(target.dataset.width), l = Number(target.dataset.left), r = Number(target.dataset.right);
        return l + ((i - a) / Math.max(1, n - 1)) * (w - l - r);
      };
      const marker = wrap.querySelector('[data-marker]');
      const yOf = (z) => {
        const top = Number(target.dataset.top), bottom = Number(target.dataset.bottom), h = Number(target.dataset.height);
        const lo = Number(target.dataset.min), hi = Number(target.dataset.max);
        return top + ((hi - z) / Math.max(1e-9, hi - lo)) * (h - top - bottom);
      };
      const crossAt = (i) => {
        if (i < a || i > b) { cross.setAttribute('opacity', '0'); marker?.setAttribute('opacity', '0'); return; }
        const x = position(i);
        cross.setAttribute('x1', x);
        cross.setAttribute('x2', x);
        cross.setAttribute('opacity', '1');
        // the marker rides the study line (the first drawn series) at the read observation
        const line = lines.find((l) => l.cls !== 'benchmark'), z = line ? rows[i]?.[target.dataset.chart === 'daily' ? line.daily : line.key] : null;
        if (marker && z !== null && z !== undefined && Number.isFinite(z) && target.dataset.chart !== 'daily') {
          marker.setAttribute('cx', x);
          marker.setAttribute('cy', yOf(z));
          marker.setAttribute('opacity', '1');
        } else marker?.setAttribute('opacity', '0');
      };
      const pointerIndex = (e) => {
        const box = target.getBoundingClientRect();
        return a + Math.max(0, Math.min(n - 1, Math.round(((e.clientX - box.left) / Math.max(1, box.width)) * (n - 1))));
      };
      const rest = () => {
        tip.hidden = true;
        if (own && app.page === 'portfolio') { crossAt(Inspect.observation); heroShow(null); }
        else cross.setAttribute('opacity', '0');
      };
      const show = (i) => {
        index = i;
        const r = rows[i];
        crossAt(i);
        if (own && app.page === 'portfolio') { heroShow(i); return; } // round 95: the hero is the readout; no tooltip over the featured chart
        const daily = target.dataset.chart === 'daily';
        tip.innerHTML = html`<span class="tip-date">${r.date}</span>${lines.map((line) => html`<span class="tip-row"><i class="tip-key ${line.cls}"></i><strong>${daily ? num(r[line.daily], 'percent') : valueOf(r[line.key])}</strong><span>${t(line.label)}</span></span>`)}<span class="muted">${spec.note || t(own ? 'Published returns · click to hold observation' : 'The owner\'s rows, indexed for reading')}</span>`;
        tip.hidden = false;
        const rect = target.closest('svg').getBoundingClientRect();
        const x = (position(i) / Number(target.dataset.width)) * rect.width;
        tip.style.left = Math.max(param('space-2'), Math.min(wrap.clientWidth - tip.offsetWidth - param('space-2'), x + param('chart-tick-gap'))) + 'px';
        tip.style.top = param('space-1') + 'px';
      };
      if (own && app.page === 'portfolio') crossAt(index);
      target.addEventListener('pointermove', (e) => show(pointerIndex(e)));
      target.addEventListener('pointerleave', rest);
      target.addEventListener('click', (e) => {
        if (!own) return;
        Inspect.selectObservation(pointerIndex(e));
        show(Inspect.observation);
      });
      target.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && own) { e.preventDefault(); rest(); return; } // lets go: the hero returns to the window
        if (['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(e.key)) {
          e.preventDefault();
          const next = e.key === 'Home' ? a : e.key === 'End' ? b : index + (e.key === 'ArrowRight' ? 1 : -1);
          index = Math.max(a, Math.min(b, next));
          if (own) Inspect.selectObservation(index);
          show(index);
        }
        if (e.key === 'Enter') {
          e.preventDefault();
          if (own && app.page === 'portfolio') Inspect.openLens('observation');
          else chartData();
        }
      });
      target.addEventListener('focus', () => show(own ? Inspect.observation : index));
      target.addEventListener('blur', rest);
    }
    if ($('#performanceChart')) Inspect.syncObservation();
    bindNavigator();
  }
  function chartData() {
    return LiveViews.chartData();
    openDialog(t('Chart · synthetic data'), t('Underlying values'), html`<p class="caption">${t('Generated design series, not historical market observations or product performance. The chart uses the same rows.')}</p>${table([t('Date'), t('Daily %'), t('Index'), t('Benchmark index')], Data.series().slice(-30).map((x) => tr([x.date, x.daily.toFixed(3), x.value.toFixed(2), x.benchmark.toFixed(2)])), t('Last 30 rows shown; export JSON for the complete 180-row fixture.'))}`, html`${btn(t('Export series JSON'), 'export-series', '', 'button primary')}`, true, () => chartData());
  }

  /* ---- pages ---- */
  const METRIC_STATS = [
    ['annual', 'Annualized return', (m) => num(m.annual, 'percent'), 'Synthetic series', 'annualized'],
    ['vol', 'Volatility', (m) => num(m.vol, 'percent'), 'Annualized · synthetic', 'annualized'],
    ['drawdown', 'Maximum drawdown', (m) => num(m.drawdown, 'percent'), 'Peak-to-trough · synthetic', 'peak to trough'],
    ['sharpe', 'Sharpe', (m) => fmt(m.sharpe), 'ratio'],
    ['sortino', 'Sortino', (m) => fmt(m.sortino), 'ratio · downside variation'],
  ];
  /* A whole-report value of the saved book: the owner's number over the report interval, net of
   * the declared cost; never recomputed here, and never the shown session's own figure. */
  // The basis every tile shares is said once under the strip; each tile keeps only its own word.
  const reportNote = () => {
    const s = Data.subject(), rows = Data.series();
    if (Data.performanceMode() === 'historical' && Data.rollingPerformance()) return t('The saved report and holdings remain pinned. Portfolio supplies the rolling outcomes and metrics; the UI does not recompute them.');
    return t('whole report {from} — {to} · net of {c} bps per side · the owner\'s values, not recomputed here', {from: rows[0]?.date || s?.support?.start || '', to: rows[rows.length - 1]?.date || s?.support?.end || '', c: count(s?.cost_per_side)});
  };
  // the strip's line: the facts (span, cost); the sentence about whose values they are is the (i)
  const reportSpan = () => {
    const s = Data.subject(), rows = Data.series();
    if (Data.performanceMode() === 'forward') {
      const f=Data.forwardPerformance(), p=f?.selected_window_metric_provenance || {}, parts=[];
      if (p.selected_start && p.selected_end && p.observation_count != null && f?.cost_bps_per_side != null) return t('Realized window {from} — {to} · {n} observations · {c} bps per side', {from:p.selected_start,to:p.selected_end,n:count(p.observation_count),c:count(f.cost_bps_per_side)});
      if (p.selected_start && p.selected_end) parts.push(t('Realized window {from} — {to}',{from:p.selected_start,to:p.selected_end}));
      if (p.observation_count != null) parts.push(t('{n} observations',{n:count(p.observation_count)}));
      if (f?.cost_bps_per_side != null) parts.push(t('Net of {c} bps per side',{c:count(f.cost_bps_per_side)}));
      return joinMarkup(parts,' · ');
    }
    const rolling = Data.rollingPerformance();
    if (rolling) {
      const range = rolling.formation_range || {}, parts = [];
      if (range.start && range.end) parts.push(t('Formation range {from} — {to}', {from: range.start, to: range.end}));
      if (rolling.outcomes_observed_through) parts.push(t('Outcomes through {date}', {date: rolling.outcomes_observed_through}));
      if (rolling.cost_bps_per_side !== undefined && rolling.cost_bps_per_side !== null) parts.push(t('Net of {c} bps per side', {c: count(rolling.cost_bps_per_side)}));
      return joinMarkup(parts);
    }
    return t('Whole report {from} — {to} · net of {c} bps per side', {from: rows[0]?.date || s?.support?.start || '', to: rows[rows.length - 1]?.date || s?.support?.end || '', c: count(s?.cost_per_side)});
  };
  function metricStat([key, label, value, word, liveWord = word]) {
    // V617 (U94): a value the saved report does not record says so in its place, the owner's reason on the label's hover
    const forward = Data.performanceMode() === 'forward', rolling=Data.rollingPerformance(), fp=Data.forwardPerformance();
    const absent = !Number.isFinite(Data.metrics()[key]) && (LiveViews.metricAbsence(key) || (forward ? {detail:fp?.status || 'No realized forward publication is available for this book and cost lane.'} : null));
    const inspect = html`<button class="metric-open" data-action="proof-open" data-value="${key}" aria-label="${t('Inspect ' + key + ' source and limitations')}"><span>${absent ? t('Not recorded') : value(Data.metrics())}</span>${icon('arrow')}</button>`;
    const note = forward ? 'Owner-reported realized forward metric' : rolling ? 'Owner-reported rolling outcome metric' : window.ALPHA_PRODUCT ? (Number.isFinite(Data.metrics()[key]) ? liveWord : 'Not supplied by this saved report') : word;
    const absence = forward ? 'Not recorded in the realized window' : rolling ? 'Not recorded in rolling outcomes' : 'Not recorded in the saved report';
    return stat(t(label), inspect, absent ? `${t(absence)} · ${absent.detail ? t(absent.detail) : ''}` : t(note), Boolean(absent));
  }
  const OPEN_POSITION_STATUS = {
    ENTRY_SETTLED_OUTCOME_PENDING: 'Entry settled; outcome pending',
    PROPOSAL_NOT_YET_SETTLED: 'Proposal not yet settled',
  };
  const openPositionFacts = (position) => {
    const status = OPEN_POSITION_STATUS[position.status];
    if (!status) return null;
    const dates = [
      position.formation_session ? `${t('Formation')} ${position.formation_session}` : '',
      position.entry_session ? `${t('Entry')} ${position.entry_session}` : '',
      position.holding_end_session ? `${t('Holding end')} ${position.holding_end_session}` : '',
    ].filter(Boolean);
    return [t(status), joinMarkup(dates)];
  };
  function rollingOpenPeriods() {
    const rolling = Data.performanceMode() === 'historical' ? Data.rollingPerformance() : null;
    const rows = (rolling?.open_positions || []).map(openPositionFacts).filter(Boolean);
    if (!rows.length) return '';
    return html`<section class="rolling-open-periods section-gap" aria-label="${t('Open periods')}">${sectionHead(t('Open periods'))}${kv(rows)}</section>`;
  }
  function performancePanel() {
    const forward=Data.performanceMode()==='forward', fp=Data.forwardPerformance(), proof=fp?.selected_window_metric_provenance || {};
    const controls=tabStrip(t('Performance source'),[
      {word:t('Historical report'),action:'portfolio-performance',value:'historical',on:!forward},
      {word:t('Forward realized'),action:'portfolio-performance',value:'forward',on:forward},
    ],'performance-tabs');
    const noForwardObservations=forward && Data.series().length===0;
    if (forward) {
      const claim = proof.claim === 'POST_OBSERVED_QA_NOT_TIMELY_ADVICE' ? codeWords(proof.claim) : '';
      const execution = proof.execution_basis === 'DAILY_BAR_QA_NOT_VERIFIED_VENUE_EXECUTION' ? t('Daily bar QA returns; not verified venue execution.') : '';
      const dates = proof.observed_through && proof.published_at
        ? t('Observed through {date}; published {published}', {date:proof.observed_through,published:proof.published_at})
        : '';
      const availableStatus = [claim,execution,dates].filter(Boolean).join(' · ');
      const absenceCode = proof.status || fp?.status;
      const absenceStatus = {
        INSUFFICIENT_REALIZED_OBSERVATIONS: t('Fewer than two settled outcomes were recorded'),
        NO_REALIZED_FORWARD_PUBLICATION: t('No realized forward publication is available for this book and cost lane.'),
      }[absenceCode] || (!fp?.status && fp?.available !== true ? t('No realized forward publication is available for this book and cost lane.') : '');
      if (noForwardObservations) {
        const scope = [reportSpan(), dates].filter(Boolean).join(' · ');
        const absence = absenceStatus || (absenceCode ? coded(absenceCode) : t('Not recorded'));
        return html`<div id="portfolioPerformance">${controls}${emptyState(absence, scope ? html`<p class="caption">${scope}</p>` : '', '', 'elsewhere')}</div>`;
      }
      const status = fp?.available === true ? (availableStatus ? html`<p class="caption">${availableStatus}</p>` : '') : (absenceStatus ? html`<p class="caption">${absenceStatus}</p>` : '');
      const basis = html`<div class="caption metric-basis">${icon('lock')}<span>${hint(reportSpan(), t('The exact installed book, publication and realized window supplied these values; historical research metrics remain separate.') )}</span></div>`;
      return html`<div id="portfolioPerformance">${controls}<div class="performance-surface" aria-label="${t('Forward realized performance')}">${basis}${status}${chartToolbar()}<div id="performanceChart">${chartBlock()}</div><div class="chart-under"><span>${chartUnder()}</span>${btn(t('Underlying values'), 'chart-data', '', 'text-btn')}</div></div></div>`;
    }
    const metrics=html`<div class="stat-strip rail five">${METRIC_STATS.map(metricStat)}</div>`;
    const rolling=Data.rollingPerformance(), facts=[];
    if (rolling) {
      const range=rolling.formation_range || {};
      if (range.start && range.end) facts.push([t('Formation range'), dateRange(range.start, range.end)]);
      if (rolling.outcomes_observed_through) facts.push([t('Outcomes through'), rolling.outcomes_observed_through]);
      if (rolling.cost_bps_per_side !== undefined && rolling.cost_bps_per_side !== null) facts.push([t('Cost'), `${count(rolling.cost_bps_per_side)} ${t('bps per side')}`]);
    }
    const basis=rolling
      ? html`<div class="stat-basis metric-basis">${icon('lock')}<div>${facts.length ? contextFacts(facts) : ''}${infoMark(reportNote())}</div></div>`
      : html`<div class="caption metric-basis">${icon('lock')}<span>${hint(reportSpan(), reportNote())}</span></div>`;
    return html`<div id="portfolioPerformance">${controls}<section class="panel performance-surface" aria-label="${t('Historical portfolio performance')}" data-box="figure">${metrics}${basis}${chartToolbar()}<div id="performanceChart">${chartBlock()}</div>${Inspect.observationDock()}<div class="chart-under"><span>${chartUnder()}</span>${btn(t('Underlying values'), 'chart-data', '', 'text-btn')}</div></section></div>`;
  }
  function page() {
    const subject=Data.subject(), sessions=Data.sessions(), recorded=Data.history().find((x)=>x.task_id===subject.task_id&&['portfolio.policy-development','INSTALLED_RESULT'].includes(x.raw.kind));
    if(typeof queueMicrotask==='function') queueMicrotask(()=>void LiveActivation.ensure(subject)); // U73: whether its strategy runs forward, read once a package
    const headFacts=[[t('Declared as-of'),subject.input_date],[t('Report'),dateRange(sessions[0],sessions.at(-1))],[t('trading|Sessions'),count(sessions.length)],[t('Recorded'),recorded?.recordedAt ? when(recorded.recordedAt) : ''],[t('Input'),subject.input_id || Data.raw()?.research_input_id || '']];
    const title=subject.source_kind==='INSTALLED_RESULT' ? LiveViews.studyFacts(null,{kind:'INSTALLED_RESULT',strategy_package_id:subject.title}).words : subject.title ? codeWords(subject.title) : LiveViews.bookWords(subject.policy) || t('Portfolio study');
    return (
      html`${objectHead(title, html`<p class="lede">${Data.notice()}</p>`, Data.raw()?.reviewSelector ? link(html`${t('Review evidence')}${icon('arrow')}`, 'evidence', 'button primary', {review_selector: JSON.stringify(Data.raw().reviewSelector)}) : '', badge('historical'), LiveViews.bookTools(subject), {object:true,id:subject.task_id,facts:headFacts})}
 ${performancePanel()}${rollingOpenPeriods()}
 ${stackSlot('bookActivation', LiveActivation.panel(subject))}${stackSlot('portfolioStanding', standingPanel(Data.raw()?.standing))}${stackSlot('portfolioTiming', LiveViews.researchTiming(Data.raw()?.timing))}
 <section class="panel portfolio-detail-surface section-gap"${Data.portfolioReading() ? ' aria-busy="true"' : ''}>${tabs()}<div class="holdings-toolbar">${holdingsToolbar()}</div><div id="portfolioDetail">${details()}</div></section>`
    );
  }
  function comparePage() {
    return LiveViews.comparePage();
  }
  return {page, comparePage, inspectHolding, refreshHoldings, refreshSessionReading, refreshSessionSurface, refreshPerformanceSurface, pageHoldings, sortHoldings, setTab, syncSessionButtons, rememberSessionPlace, setSession, stepSession, setChart, setPerformanceMode, redrawChart, bindCharts, chartData};
})();
PAGES.portfolio = Portfolio.page;
PAGES.compare = Portfolio.comparePage;

Object.assign(ACTIONS, {
  'portfolio-tab': (v) => Portfolio.setTab(v),
  'portfolio-performance': (v) => Portfolio.setPerformanceMode(v),
  holding: (v) => Portfolio.inspectHolding(v),
  'session-prev': () => Portfolio.stepSession(-1),
  'session-next': () => Portfolio.stepSession(1),
  'holding-page-prev': () => Portfolio.pageHoldings(-1),
  'holding-page-next': () => Portfolio.pageHoldings(1),
  'holding-sort': (key) => Portfolio.sortHoldings(key),
  chart: (v) => Portfolio.setChart(v),
  'chart-range': (key) => { const [a, b] = Inspect.presetWindow(key); Inspect.setWindow(a, b); },
  'chart-data': () => Portfolio.chartData(),
});
Object.assign(ON_INPUT, {
  holdingsQuery: (e) => {
    app.holdingsQuery = e.target.value;
    app.holdingsPage = 0;
    Portfolio.refreshHoldings();
  },
});
Object.assign(ON_CHANGE, {
  holdingsSession: (e) => Portfolio.setSession(e.target.value),
  compareSelect: (e) => Data.compare(e.target.value),
});
