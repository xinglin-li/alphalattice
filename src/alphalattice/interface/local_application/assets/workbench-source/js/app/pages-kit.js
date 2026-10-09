const Kit = (() => {
  const NOOP = 'kit-noop';
  const SAMPLE = 'Evidence review 2026 · 0123456789';
  const STYLES = ['large-title', 'title-1', 'title-2', 'title-3', 'headline', 'body', 'callout', 'footnote', 'caption', 'figure'];
  const INKS = ['text', 'muted', 'faint', 'ink-link', 'accent', 'disabled-ink', 'action-bg'];
  const TONES = ['good', 'warning', 'danger', 'review', 'neutral', 'cyan'];
  const GROUNDS = ['surface-page', 'surface-face', 'surface-box', 'surface-raised', 'surface-overlay', 'surface-well', 'subtle-surface', 'fill-selected', 'fill-highlight'];
  const LINES = ['line', 'edge-hair', 'edge-card', 'edge-strong'];

  const specimen = (name, body, o = {}) => html`<figure class="kit-specimen${o.wide ? ' kit-wide' : ''}" data-kit="${name}"${o.stackCaption ? ' data-stack-caption' : ''}><div class="kit-stage">${body}</div><figcaption><strong>${name}</strong>${o.note ? html`<span>${o.note}</span>` : ''}<span class="kit-measure" data-kit-measure="${o.measure || ''}"></span></figcaption></figure>`;
  const boxed = (x) => html`<section class="panel" data-box="table">${x}</section>`;
  const section = (id, title, items) => html`<section class="kit-section" id="kit-${id}" aria-labelledby="kit-${id}-title"><h2 id="kit-${id}-title">${title}</h2><div class="kit-grid">${items}</div></section>`;
  const swatch = (name, prop = 'background') => html`<span class="kit-swatch"><i data-ui-style="${prop}: var(--${name})"></i><span class="mono">--${name}</span></span>`;

  function typeSection() {
    return section('type', t('Text styles'), STYLES.map((s) => specimen(s, html`<span class="kit-type" data-ui-style="font-size: var(--style-${s}-size); line-height: var(--style-${s}-leading); letter-spacing: var(--style-${s}-tracking, 0)">${SAMPLE}</span>`, {measure: 'type'})));
  }
  function colourSection() {
    return section('colour', t('Inks, grounds and lines'), [
      specimen('inks', INKS.map((n) => swatch(n)), {wide: true}),
      specimen('tones · mark and ink', TONES.map((n) => html`${swatch(n)}${swatch(n + '-ink')}`), {wide: true}),
      specimen('grounds and fills', GROUNDS.map((n) => swatch(n)), {wide: true}),
      specimen('lines', LINES.map((n) => swatch(n)), {wide: true}),
    ]);
  }
  function controlSection() {
    const pick = [['a', t('Current')], ['b', t('Pinned')], ['c', t('Historical')]];
    return section('controls', t('Buttons, fields and choices'), [
      specimen('button · primary', btn(t('Prepare again'), NOOP, '', 'button primary')),
      specimen('button', btn(t('Export'), NOOP, '', 'button')),
      specimen('button · compact', btn(t('Export'), NOOP, '', 'button compact')),
      specimen('button · critical', btn(t('Cancel'), NOOP, '', 'button critical')),
      specimen('text button', btn(t('Show more'), NOOP, '', 'text-btn')),
      specimen('icon button', btnAttrs(icon('more'), NOOP, '', 'icon-btn', html`aria-label="${t('More')}"`)),
      specimen('icon button · compact', btnAttrs(icon('copy'), NOOP, '', 'icon-btn compact', html`aria-label="${t('Copy')}"`)),
      specimen('segmented control · a value', html`<div class="segmented ui-segments" aria-label="${t('Chart range')}">${segBtn('1M', NOOP, 'a', false)}${segBtn('3M', NOOP, 'b', true)}${segBtn('YTD', NOOP, 'c', false)}${segBtn(t('All'), NOOP, 'd', false)}</div>`),
      specimen('tabs · a view', tabStrip(t('Views'), [{word: t('Overview'), action: NOOP, value: 'a', on: true}, {word: t('Sources'), action: NOOP, value: 'b', count: '12'}, {word: t('Reading'), action: NOOP, value: 'c'}, {word: t('Review'), off: true}])),
      specimen('picker', picker('kitPicker', pick, {selected: 'a', label: t('Reading'), action: NOOP})),
      specimen('picker · text', picker('kitPickerText', pick, {selected: 'a', kind: 'text', label: t('Reading'), action: NOOP})),
      specimen('filter chip', chip(t('State'), '', {name: 'kit-state', options: [['running', t('Running')], ['failed', t('Failed')]]})),
      specimen('property chip', propertyChip(t('Session'), '2024-02-01', {note: t('Recorded by the owner; not editable here.')})),
      specimen('field', field(t('Name'), 'kitField', 'Synthetic single book'), {stackCaption: true}),
      specimen('field inline', html`<div class="flow">${field(t('Bytes'), 'kitInlineField', '21474836480', 'text', '', '', {inline: true})}${btn(t('Set'), NOOP, '', 'button compact')}${btn(t('Auto'), NOOP, '', 'button compact')}</div>`, {stackCaption: true}),
      specimen('search', html`<input type="search" class="search-input" placeholder="${t('Find an issuer…')}" aria-label="${t('Find an issuer…')}">`),
      specimen('stepper', stepper('3', NOOP, {label: t('Rows')})),
      specimen('keycap', keycap('Ctrl K')),
    ]);
  }
  function stateSection() {
    const states = ['running', 'succeeded', 'failed', 'cancelled', 'waiting', 'recorded', 'blocked'];
    return section('states', t('States and marks'), [
      specimen('badge', states.map((s) => badge(s)), {wide: true}),
      specimen('state line', states.map((s) => stateLine(s)), {wide: true}),
      specimen('status dot', states.map((s) => statusDot(s))),
      specimen('mark', [['lab', 'review'], ['branch', 'cyan'], ['evidence', 'warning'], ['portfolio', 'good'], ['data', 'neutral']].map(([ic, tone]) => tile(ic, tone))),
    ]);
  }
  function rowSection() {
    const rows = html`<div class="card-list lines slotted">${objectRow({state: 'succeeded', name: 'Alternative evidence document intelligence', why: '', to: {action: NOOP, value: '1'}}, {columns: ['duration', 'positions', 'session'], props: ['1 s', '', ''], time: 'Aug 13 04:28'})}${objectRow({lead: 'file', name: 'Chief risk officer portfolio review', why: 'Reviewed · 1 required action', to: {action: NOOP, value: '2'}}, {columns: ['state', 'duration', 'positions', 'session'], props: ['', '0 s', '', ''], time: 'Aug 13 04:44'})}${objectRow({lead: tile('portfolio', 'good'), name: 'Synthetic single book', to: null}, {columns: ['state', 'duration', 'positions', 'session'], props: ['', '', '35 positions', '2024-02-01']})}</div>`;
    return section('rows', t('Rows and lists'), [
      specimen('object rows', rows, {wide: true, measure: 'rows'}),
      specimen('group head', groupHead(t('Completed'), 3), {wide: true}),
      specimen('menu rows', html`<div class="kit-menu" role="menu">${menuRow({ic: 'link', word: t('Copy link'), key: 'L', action: NOOP})}${menuRow({ic: 'fork', word: t('Map'), action: NOOP})}${menuRow({word: t('Current'), checked: true, action: NOOP})}</div>`, {measure: 'rows'}),
    ]);
  }
  function formSection() {
    const rows = html`${formRow({title: t('Appearance'), line: t('Light or dark; following the host takes its choice.'), control: picker('kitTheme', [['light', t('Light')], ['dark', t('Dark')]], {selected: 'light', label: t('Appearance'), action: NOOP})})}${formRow({title: t('Workspace'), detail: 'qa-evidence-integrated', mono: true})}${formRow({title: t('Keyboard shortcuts'), line: t('Every key the page answers'), action: NOOP})}${switchRow('kitSwitch', true, t('Wide scrollbars'))}`;
    return section('forms', t('Forms'), [specimen('form group', formGroup(t('General'), rows), {wide: true, measure: 'rows', stackCaption: true})]);
  }
  function tableSection() {
    const heads = ['#', t('Issuer'), {label: t('Weight'), type: 'num'}, {label: t('Change (bps)'), type: 'num'}, t('Conclusion')];
    const data = [['1', 'AMZN', '17.8%', '+0.16', t('Concerns not adjudicated')], ['2', 'TSLA', '18.96%', '+0.07', t('Evidence gap')], ['3', 'AAPL', '21.86%', '-0.17', t('Concerns not adjudicated')]];
    return section('tables', t('Tables'), [
      specimen('table', boxed(table(heads, data.map(tr))), {wide: true, measure: 'rows', stackCaption: true}),
      specimen('table · compact', boxed(table(heads, data.map(tr), '', 'compact')), {wide: true, measure: 'rows', stackCaption: true}),
    ]);
  }
  function boxSection() {
    return section('boxes', t('Boxes and figures'), [
      specimen('panel', panel(t('Issuers'), t('Five issuers, one book'), html`<p>${SAMPLE}</p>`), {wide: true, stackCaption: true}),
      specimen('banner', banner(t('Request an evidence refresh'), t('Reviewed ending-weight coverage is below the minimum this route requires.')), {wide: true, stackCaption: true}),
      specimen('refusal', refusal({reason: t('The owner did not answer.'), next_action: 'overview'}, TONE.failure), {wide: true, stackCaption: true}),
      specimen('empty state', emptyState(t('No records yet.'))),
      specimen('figures', rail([stat(t('Issuers'), '5'), stat(t('Findings'), '7'), stat(t('review|Issues'), '6'), stat(t('Publications'), '2')]), {wide: true}),
      specimen('code cell', codeCell('chief_risk_officer.portfolio-evidence-review')),
      specimen('code editor', codeEditor('kitCode', '{\n  "assessment": "example"\n}', {lang: 'json', label: 'JSON'}), {wide: true}),
      specimen('properties', kv([[t('As-of'), '2026-08-13'], [t('Expires'), '2026-08-14'], [t('Analysis'), 'selected · 2026-08-13']])),
      specimen('meter', html`${meter({kind: 'share', segments: [{n: 2, label: t('Checked'), tone: 'good'}, {n: 3, label: t('Not checked'), tone: 'neutral'}], mark: 0.6})}${meter({kind: 'progress', now: 3, max: 5, label: t('Progress')})}`),
    ]);
  }
  function evidenceSection() {
    const runs = [
      {id: 'kit-run-1', name: 'Alternative evidence document intelligence', state: 'SUCCEEDED', kind: 'alternative_evidence', started: '2026-08-13T04:08:00Z', finished: '2026-08-13T04:09:00Z'},
      {id: 'kit-run-2', name: 'Chief risk officer portfolio review', state: 'SUCCEEDED', kind: 'chief_risk_officer_review', started: '2026-08-13T04:44:00Z', finished: '2026-08-13T04:44:00Z'},
      {id: 'kit-run-3', name: 'Installed strategy result', state: 'SUCCEEDED', kind: 'portfolio_replay', started: '2026-08-12T16:03:00Z', finished: '2026-08-12T16:04:00Z'},
    ];
    const finding = (title, sub, stance, cites) => tableRow('', [['col-text col-absorb', rowTitle({title, sub, action: NOOP, value: title})], ['col-text col-tight', stance], ['col-num col-tight', cites]]);
    return section('evidence', t('The Evidence library'), [
      specimen('status box', statusBox({tone: 'warning', title: t('Human review required'), caption: 'Published · Aug 13 04:44', checks: [
        {tone: 'good', name: t('Prepared'), time: 'Aug 13'},
        {tone: 'good', name: t('Analysed'), time: 'Aug 13'},
        {tone: 'warning', name: t('Reviewed'), why: t('human review required'), time: 'Aug 13'},
        {tone: 'neutral', name: t('Reported'), why: t('Not started')}], foot: [{tone: TONE.attention, word: t('Blocking'), why: t('Request an evidence refresh')}], label: t('Status')}), {wide: true, stackCaption: true}),
      specimen('measure strip', measureStrip([stat(t('Reviewed weight'), '39.65%', t('floor {m}', {m: '60%'})), stat(t('Mapping'), '100%'), stat(t('Issuers'), '5'), stat(t('Findings'), '7'), stat(t('review|Issues'), '6'), stat(t('Publications'), '2')], t('Measures')), {wide: true}),
      specimen('check list', checkList({title: t('Set up'), key: 'kit', rowCls: 'evidence-row setup-row', rows: [
        {id: 'authority', met: true, name: t('Evidence authority'), why: t('a recorded package is bound into this workspace')},
        {id: 'pack', met: false, name: t('Retrieval pack'), why: t('not installed'), detail: html`<div class="setup-card"><p>${SAMPLE}</p><pre class="code-block setup-command">scripts/install_retrieval_pack.py --status</pre></div>`}]}), {wide: true}),
      specimen('property list', propertyList(t('Properties'), [['calendar', t('As-of'), 'Aug 13'], ['clock', t('Expires'), 'Aug 14'], ['user', t('Reviewer'), 'CRO'], ['branch', t('Policy'), '']]), {stackCaption: true}),
      specimen('grouped table', boxed(groupedTable({columns: [[t('Finding'), 'col-text col-absorb'], [t('Stance'), 'col-text col-tight'], [t('Citations'), 'col-num col-tight']], cls: 'findings-table', rows: html`${tableGroup(3, 'AAPL', '2 findings · 1 material')}${finding(t('Legal & regulatory'), SAMPLE, t('Adverse'), '3')}${finding(t('Liquidity & going concern'), null, t('Mixed'), '2')}`})), {wide: true, measure: 'rows', stackCaption: true}),
      specimen('passage list', boxed(passageList([
        {handle: 'kit-span-1', document: 'DOC-AAPL-011', type: '10-Q', excerpt: SAMPLE, issuer: 'AAPL', day: 'Aug 1', state: stateLine('verified', {next: ''}), action: NOOP, value: 'kit-span-1', selected: true},
        {handle: 'kit-span-2', document: 'DOC-AMZN-016', type: '10-Q', excerpt: SAMPLE, issuer: 'AMZN', day: 'Aug 2', state: stateLine('verified', {next: ''}), action: NOOP, value: 'kit-span-2', selected: false}])), {wide: true, measure: 'rows', stackCaption: true}),
      specimen('activity feed', activityFeed({title: t('Activity'), list: runs, shown: 2, render: (rs) => html`<div class="card-list lines slotted">${rs.map((r) => runRow(r, {to: {action: NOOP, value: r.id}}))}</div>`, more: NOOP}), {wide: true, measure: 'rows', stackCaption: true}),
      specimen('pager', pager({total: 45, one: '{n} entry', many: '{n} entries', page: 1, pages: 3, prev: [NOOP, 'prev'], next: [NOOP, 'next']}), {wide: true}),
      specimen('search bar', searchBar('kitSearch', t('Find an issuer'), t('Find an issuer…'), '')),
    ]);
  }
  function parameterSection() {
    const groups = typeof PARAMETER_GROUPS !== 'undefined' ? PARAMETER_GROUPS : {};
    const rows = Object.entries(groups).flatMap(([group, names]) => names.map((name, i) => tr([i ? '' : html`<span class="mono">${group}</span>`, html`<span class="mono">--${name}</span>`, html`<span class="mono" data-kit-value="${name}"></span>`])));
    return section('parameters', t('Parameters'), [specimen('design/parameters.json', boxed(table([t('Group'), t('Parameter'), t('Value')], rows)), {wide: true, note: t('Each value as it resolves in this appearance and language'), stackCaption: true})]);
  }
  /* After the paint: each specimen's measure line and each parameter's value. */
  function measure() {
    const main = document.getElementById('main');
    if (!main || !main.querySelector('.kit-section')) return;
    const px = (v) => Math.round(parseFloat(v) || 0);
    for (const line of main.querySelectorAll('[data-kit-measure]')) {
      const stage = line.closest('.kit-specimen')?.querySelector('.kit-stage');
      const target = line.dataset.kitMeasure === 'rows' ? stage?.querySelector('.list-row, tbody tr, .menu-row, .form-row') : stage?.firstElementChild;
      if (!target) continue;
      const s = getComputedStyle(target), r = target.getBoundingClientRect();
      const zoom = rootZoom();
      const radius = px(s.borderTopLeftRadius);
      line.textContent = `h ${Math.round(r.height / zoom)}${radius ? ` · r ${radius}` : ''} · ${px(s.fontSize)}/${s.lineHeight === 'normal' ? 'n' : px(s.lineHeight)}/${s.fontWeight}`;
    }
    const root = getComputedStyle(document.body);
    for (const cell of main.querySelectorAll('[data-kit-value]')) cell.textContent = root.getPropertyValue('--' + cell.dataset.kitValue).trim() || '';
  }
  function page() {
    requestAnimationFrame(() => requestAnimationFrame(measure));
    return html`${objectHead(t('Component workshop'), html`<p class="lede">${t('Every component the pages compose, beside what it measures, and every parameter with its value.')}</p>`)}${controlSection()}${stateSection()}${rowSection()}${formSection()}${tableSection()}${boxSection()}${evidenceSection()}${typeSection()}${colourSection()}${parameterSection()}`;
  }
  return {page, measure};
})();
PAGES.kit = Kit.page;
