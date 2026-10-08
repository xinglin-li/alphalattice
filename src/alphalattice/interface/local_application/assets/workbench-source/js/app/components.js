/* Markup builders shared by every page. They return HTML strings; text passed in is already
 * localized by the caller (t()). Builders add the control-contract attributes (data-control,
 * data-size, data-ui-role, data-box) that the stylesheet keys on. */

/* ---- control contract ---- */
const CONTROL_KINDS = [
  ['icon-btn', 'icon'],
  ['text-btn', 'quiet'],
  ['link', 'quiet'],
  ['critical', 'critical'],
  ['primary', 'primary'],
];
/* Buttons and button-like links: one chassis, four jobs. Rows and top-bar tools stay unmarked. */
function controlAttrs(cls) {
  const classes = String(cls || '').split(/\s+/).filter(Boolean);
  const isControl = classes.some((c) => ['button', 'text-btn', 'icon-btn'].includes(c));
  if (!isControl || classes.some((c) => ['dialog-list-btn', 'quick-result', 'comparison-candidate'].includes(c))) return '';
  const kind = CONTROL_KINDS.find(([c]) => classes.includes(c))?.[1] || 'action';
  return ` data-control="${kind}" data-size="${classes.includes('compact') ? 'compact' : 'regular'}"`;
}
const segmentAttr = ' data-control="segment"';
const STATUS_ATTR = ' data-ui-role="status"';

const ACTION_LABELS = {context: 'Inspect research context', 'research-map': 'Open research map'};

/* Plain action button (no type attribute, as in the original shell markup). */
function btn(label, action, value = '', cls = 'button', mutate = false) {
  const aria = ACTION_LABELS[action] ? ` aria-label="${t(ACTION_LABELS[action])}"` : '';
  return html`<button${aria} class="${cls}" data-action="${action}" data-value="${(value)}"${mutate ? ' data-mutating="true"' : ''}${controlAttrs(cls)}>${label}</button>`;
}
/* Same button with explicit type and optional attributes (aria-label, aria-pressed, id …). */
function btnAttrs(label, action, value, cls, attrs = '') {
  return html`<button${attrs ? ' ' + attrs : ''} class="${cls}" data-action="${action}" data-value="${(value)}"${controlAttrs(cls)}>${label}</button>`;
}
/* Typed button used by the task, evidence and review surfaces. `disabled` carries the reason. */
function typedBtn(label, action, value = '', cls = 'button', disabled = '', extra = '', reasonAttr = 'title') {
  let off = '';
  if (disabled) {
    // data-reason keeps the English source; Controls.sync renders the localized note and title.
    if (reasonAttr === 'title') off = ` disabled data-tip="${esc(disabled)}"`;
    else off = ` disabled data-reason="${esc(disabled)}" data-tip="${esc(t(disabled))}"`;
  }
  return html`<button type="button" class="${cls}" data-action="${action}" data-value="${(value)}"${off}${extra ? ' ' + extra : ''}${controlAttrs(cls)}>${label}</button>`;
}
function link(label, page, cls = 'button', extra = {}) {
  return html`<a class="${cls}" href="${routeUrl(page, extra)}"${controlAttrs(cls)}>${label}</a>`;
}
/* The state vocabulary (round 71). `badge` is the tinted pill, kept only where a word alone
 * would mislead: a refusal, an attention item, a decision, Historical. Everything that runs or
 * waits reads as `stateLine`: the dot in the state's tone, the word from the table, the time or
 * the duration in the muted ink, and after a `·` the next action when the state is a person's
 * to move. `statusDot` is the dot alone, its word in the label: the lead of a list row. */
/* The root's scale (round 85): under `zoom`, a rect from getBoundingClientRect comes in the
 * viewport's px while every layout length (a menu's width, a fixed `left`) is scaled; anything
 * that places a surface reads rects and the viewport through these, in layout px. */
const rootZoom = () => Number(getComputedStyle(document.documentElement).zoom) || 1;
function layoutRect(el) { const r = el.getBoundingClientRect(), z = rootZoom(); return {left: r.left / z, top: r.top / z, right: r.right / z, bottom: r.bottom / z, width: r.width / z, height: r.height / z}; }
const viewW = () => innerWidth / rootZoom(), viewH = () => innerHeight / rootZoom();
function badge(status, label) {
  const a = stateOf(status);
  return html`<span class="badge ${a.tone}" data-status="${a.key}"${STATUS_ATTR}>${label || t(a.word)}</span>`;
}
/* The mark (round 85, Linear's rule: colour on a glyph, never on a word; the Team page's tile
 * made the product's): a 28 px circle tinted in its tone, the glyph in the tone's ink. A saved
 * object's mark is its kind's (one colour per kind everywhere, as Linear's labels); a row that
 * waits for a decision wears the warning mark, a live one the accent. */
const KIND_MARKS = {'factor.screening-development': ['lab', 'review'], 'alpha.model-development': ['branch', 'cyan'], 'risk.covariance-development': ['evidence', 'warning'], 'portfolio.policy-development': ['portfolio', 'good'], CRO_REVIEW: ['review', 'review'], INSTALLED_RESULT: ['archive', 'neutral'], CONTINUOUS_UPDATE: ['data', 'neutral'], workspace_data_update: ['data', 'neutral'], research_experiment: ['lab', 'review']};
function tile(ic, tone = 'neutral', label = '') { // `mark()` is the product's octagram (icons.js); the tile is the row's mark
  return html`<span class="mark ${tone}"${label ? html` role="img" aria-label="${label}" data-tip="${label}"` : html` aria-hidden="true"`}>${icon(ic)}</span>`;
}
const kindTile = (kind, label = '') => { const [ic, tone] = KIND_MARKS[kind] || ['file', 'neutral']; return tile(ic, tone, label); };
/* A state's dot. A moving state breathes (W, the user's reading 2026-09-23; law 16: motion only
 * while Task Control reports the Task moving -- the lists read its projections, kept current);
 * `o.live: false` keeps a recorded word still. */
function statusDot(status, label, o = {}) {
  const a = stateOf(status), word = label || t(a.word), live = o.live ?? Boolean(a.moving);
  return html`<span class="row-dot ${a.tone}" data-status="${a.key}"${live ? ' data-live="true"' : ''} role="img" aria-label="${word}" data-tip="${word}"><i></i></span>`;
}
/* `x` is a state key or an owner's projection ({lifecycle | state | status, running_since,
 * last_activity_at}); `o` overrides: word, at (an instant), duration (text), next, live (the
 * dot breathes: the owner's heartbeat is observed now), dot (false: the words alone), since (a
 * wait's start), until (the owner's retry instant: `retries at`). */
function stateLine(x, o = {}) {
  const code = typeof x === 'string' ? x : x?.lifecycle ?? x?.state ?? x?.status ?? '';
  const a = stateOf(code), word = o.word || t(a.word);
  const time = o.duration ?? (typeof x === 'object' && x ? durationOf(x) : '') ?? '';
  const at = !time && o.at ? when(o.at) : '';
  const next = o.next === undefined ? (a.next ? t(a.next) : '') : o.next;
  const wait = html`${o.since ? html`<span class="state-time">· ${t('since')} ${when(o.since)}</span>` : ''}${o.until ? html`<span class="state-time">· ${t('retries at')} ${when(o.until)}</span>` : ''}`;
  return html`<span class="state ${a.tone}" data-status="${a.key}"${o.live ? ' data-live="true"' : ''}${STATUS_ATTR}>${o.dot === false ? '' : html`<i aria-hidden="true"></i>`}<span class="state-word">${word}</span>${time || at ? html`<span class="state-time">· ${time || at}</span>` : ''}${wait}${next ? html`<span class="state-next">· ${next}</span>` : ''}</span>`;
}
/* The shape before the content (round 71): rows, a head or a body in the face's grey, shown
 * after the 400 ms hold, never a spinner and never a word; a slow shimmer where motion is allowed. */
function skeleton(shape = 'rows', n = 3) {
  const rows = () => Array.from({length: n}, () => html`<div class="skeleton-row" aria-hidden="true"><span class="ph ph-dot"></span><span class="ph ph-line"></span><span class="ph ph-short"></span></div>`);
  const body = shape === 'head' ? html`<span class="ph ph-title"></span><span class="ph ph-line"></span>` : shape === 'body' ? html`<span class="ph ph-line"></span><span class="ph ph-line"></span><span class="ph ph-short"></span>` : rows();
  return html`<div class="skeleton skeleton-${shape}" aria-busy="true" aria-label="${t('Reading')}">${body}</div>`;
}
/* A section (N3, law 121): a box named by a label -- 13 / 500 in the muted ink, with its count --
 * never a heading with a sentence under it; what the product would have said there is the label's
 * (i) (`sub`, read on hover and focus). */
function panel(title, sub, content, action = '', attrs = '') {
  return html`<section class="panel"${attrs ? ' ' + attrs : ''}>${sectionHead(title, sub, action)}<div class="panel-body">${content}</div></section>`;
}
/* A banner (round 75): attention that blocks the page's primary, in two tones; a refusal is
 * `refusal`, a note is `noteLine`, an unanswered owner is `notRead`. */
function banner(title, text, tone = 'warning', action = '', ic = 'info', decision = tone !== 'neutral') { // law 148: a notice with a tone is a decision; a neutral one is a line unless it blocks (`decision`)
  return html`<div class="banner ${tone}"${decision ? ' data-box="decision"' : ''} role="status">${icon(ic)}<div class="grow"><strong>${title}</strong><p>${text}</p></div>${action}</div>`;
}
/* One owner for "the owner did not answer" (round 64): the error as the exact token, the words
 * that say what stays unaffected, an optional way to read again. */
// Legacy Error.message retains a machine prefix; translate only its owner's sentence.
function refusalParts(error) {
  const body = error?.body, text = String(error?.message ?? error ?? '');
  const match = text.match(/^([a-z][a-z0-9_.]*(?::[A-Za-z0-9_.-]+)*): (.+)$/s);
  if (match) return {code: match[1], detail: match[2]};
  return {code: body?.failure_code || body?.refusal_code || body?.refused || body?.disposition || text, detail: body?.message || body?.detail || body?.explanation || body?.refusal_detail || ''};
}
function notRead(title, error, words = '', action = '') {
  // N6 (law 17, a refusal in words): what could not be read, in words, first; the owner's code small beneath it
  const failure = refusalParts(error), detail = words || said(failure.detail) || explainCode(failure.code);
  return banner(title, html`${detail ? html`${detail}<br>` : ''}<span class="mono refusal-code">${failure.code}</span>`, 'warning', action);
}
/* The one instrument rail (round 27): facts side by side over a hairline, wrapping 4 → 2 → 1 by
 * the lane; each item is `stat()`'s label / value / note with an optional tile. `cls` names the
 * page's rail for its own rules. */
function rail(items, cls = '', label = '') {
  return html`<div class="stat-strip rail ${cls}"${label ? html` aria-label="${label}"` : ''}>${items}</div>`;
}
/* A hairline between sections, with an optional label in the meta ink (round 27). */
function separator(label = '') {
  return label ? html`<div class="separator" role="separator"><span>${label}</span></div>` : html`<hr class="separator">`;
}
/* A fact on a strip; with `ic` a tinted tile leads it (round 20g). A value has no caption (N3, the
 * user: 数值下不跟说明): the `note` is the label's tip, the label dotted as a hint; `figure` (a
 * meter) is drawn under the value -- a figure, not words. */
function stat(label, value, note = '', word = false, ic = '', figure = '') {
  // the icon, when given, sits inline before the label (round 30): no tile on a rail
  const noted = note && String(note).trim();
  return html`<div class="stat"><div class="stat-label${noted ? ' hint' : ''}" data-tip="${noted ? note : '@overflow'}"${noted ? ' tabindex="0"' : ''}>${ic ? html`<span class="stat-icon" aria-hidden="true">${icon(ic)}</span>` : ''}${label}</div><div class="stat-value ${word ? 'word' : ''}">${value}</div>${figure ? html`<div class="stat-figure">${figure}</div>` : ''}</div>`;
}
/* A figure that opens its page (N6, the Data overview; Sentry's score cards, measured): the tile
 * is the way -- its label (what it counts is the label's tip), its figure; `to` is a page
 * ({page, extra}) or an action ({action, value}); without one it is a plain figure. */
function figureTile(label, value, to = null, note = '') {
  const noted = note && String(note).trim();
  const body = html`<span class="stat-label${noted ? ' hint' : ''}" data-tip="${noted ? note : '@overflow'}">${label}</span><span class="stat-value">${value}</span>`;
  if (!to) return html`<div class="stat">${body}</div>`;
  return to.page ? html`<a class="stat stat-link" href="${routeUrl(to.page, to.extra || {})}">${body}</a>` : btnAttrs(body, to.action, to.value || '', 'stat stat-link');
}
function propertyRow(row) {
  const value = Array.isArray(row) ? row : [row.label, row.value, row.subLabel, row.subValue];
  const [label, content, subLabel = '', subValue = ''] = value;
  return html`<div class="kv"><dt>${label}${subLabel ? html`<small>${subLabel}</small>` : ''}</dt><dd data-tip="@overflow">${content}${subValue ? html`<small>${subValue}</small>` : ''}</dd></div>`; // a value cut by the line reads whole on hover (law 88)
}
function kv(rows, cls = '') {
  return html`<dl class="facts-properties${cls ? ' ' + cls : ''}" data-fact-shape="properties">${rows.map(propertyRow)}</dl>`;
}
function feature(ic, name, desc, action) {
  return html`<div class="feature-row"><div class="feature-icon">${icon(ic)}</div><div class="grow"><h3>${name}</h3><p>${desc}</p></div>${action || ''}</div>`;
}
function tr(cells) {
  return html`<tr>${cells.map((x) => html`<td>${x == null ? '' : x}</td>`)}</tr>`;
}
const COLUMN_TYPES = new Set(['text', 'num', 'id', 'date', 'status', 'link']);
/* A head is one line (round 60, Stripe's table): a label written "Words (unit)" keeps the words on
 * the line and the unit as a sub-line under them; `unit` may also be given outright. */
function columnSpec(header) {
  const spec = typeof header === 'object' && header !== null && Object.hasOwn(header, 'label') ? {...header, type:COLUMN_TYPES.has(header.type) ? header.type : (header.cls === 'num' ? 'num' : 'text')} : {label:header, type:'text'};
  const m = typeof spec.label === 'string' && !spec.unit ? /^(.+?)\s*\(([^()]+)\)$/.exec(spec.label) : null;
  return m ? {...spec, label: m[1], unit: m[2]} : spec;
}
function typedTableRows(rows, columns) {
  let col = 0;
  const markup = renderValue(rows, false);
  return raw(markup.replace(/<tr\b[^>]*>|<td\b([^>]*)>/g, (match, attrs) => {
    if (match.startsWith('<tr')) { col = 0; return match; }
    const spec = columns[col++], type = spec?.type || 'text', cls = `col-${type}${spec?.cls ? ' ' + spec.cls.split(' ').filter((c) => c === 'col-index' || c === 'col-identity' || c === 'col-absorb' || c === 'col-share' || c === 'col-tight').join(' ') : ''}`.trim();
    if (/\bclass=/.test(attrs || '')) return '<td' + String(attrs).replace(/class="([^"]*)"/, `class="$1 ${cls}"`) + '>';
    return `<td${attrs || ''} class="${cls}">`;
  }));
}
function table(headers, rows, note = '', classes = '', options = {}) {
  if (note && typeof note === 'object' && !(note instanceof Markup)) { options = note; note = options.note || ''; classes = options.classes || ''; }
  else if (classes && typeof classes === 'object') { options = classes; classes = options.classes || ''; }
  const hasRows = Array.isArray(rows) ? rows.length > 0 : String(rows ?? '').length > 0; // rows: fragment list or fragment
  const columns = headers.map(columnSpec);
  // the column geometry (round 60): a '#' head is the index column, the first column after it the identity
  const indexAt = columns.findIndex((x) => x.index || String(x.label) === '#'); // a sortable '#' head is markup: it says `index` outright
  const actionAt = columns.length - 1 >= 0 && columns.at(-1).type === 'link' && !columns.at(-1).label ? columns.length - 1 : -1;
  // The slack. A list gives it to its substance column: the one the producer names (`absorb`), else
  // the last text-like column (the law: words lead, figures follow, right). A list whose words are
  // all short spreads it evenly between the facts instead (`spread`), as a grid does across all its
  // data columns: the sharing columns are marked and counted, and a word column right before a
  // figure column keeps to its content, so no seam doubles where a word meets a right-aligned figure.
  const data = (i) => i !== indexAt && i !== actionAt;
  let absorbAt = options.spread || options.grid ? -1 : columns.findIndex((x) => x.absorb);
  if (absorbAt < 0 && !options.spread && !options.grid) columns.forEach((x, i) => { if (data(i) && x.type !== 'num') absorbAt = i; });
  const tight = (i) => options.spread && columns[i].type !== 'num' && columns[i + 1]?.type === 'num' && data(i + 1);
  const shares = columns.map((x, i) => (options.spread || options.grid) && data(i) && !tight(i));
  columns.forEach((x, i) => { const cls = [x.cls || '', i === indexAt ? 'col-index' : '', i === indexAt + 1 && i !== absorbAt ? 'col-identity' : '', i === absorbAt ? 'col-absorb' : '', shares[i] ? 'col-share' : '', tight(i) ? 'col-tight' : ''].filter(Boolean).join(' '); if (cls) x.cls = cls; });
  const head = (x) => html`<th scope="col" class="col-${x.type}${x.cls ? ' ' + x.cls : ''}" data-column-type="${x.type}"${x.ariaSort ? html` aria-sort="${x.ariaSort}"` : ''}>${x.label}${x.unit ? html`<small>${x.unit}</small>` : ''}</th>`;
  const body = hasRows ? typedTableRows(rows, columns) : html`<tr><td colspan="${headers.length}">${emptyState(t('No records yet.'))}</td></tr>`;
  const foot = options.foot ? html`<tfoot>${typedTableRows(tr(options.foot), columns)}</tfoot>` : '';
  const countLine = options.report && options.countLine !== false ? html`<p class="report-count">${countText(options.count ?? (Array.isArray(rows) ? rows.length : 0), '{n} entry', '{n} entries')}</p>` : '';
  return html`${countLine}<div class="table-scroll${options.report ? ' report-scroll' : ''}" tabindex="0" role="region" aria-label="${t(options.report ? 'Data report' : 'Scrollable data table')}"><table class="data-table ${classes}${options.report ? ' report-table' : ''}${options.grid ? ' grid' : ''}${options.spread ? ' spread' : ''}" data-share="${shares.filter(Boolean).length}" data-data-surface="editorial"><thead><tr>${columns.map(head)}</tr></thead><tbody>${body}</tbody>${foot}</table></div>${note ? html`<p class="table-note">${note}</p>` : ''}`;
}
function field(label, id, value, type = 'text', hint = '', attrs = '', {inline = false} = {}) {
  return html`<div class="field${inline ? ' inline' : ''}"><label for="${id}">${label}</label><input id="${id}" class="ui-field" type="${type}" value="${(value)}"${attrs ? ' ' + attrs : ''}>${hint ? html`<small>${hint}</small>` : ''}</div>`;
}
/* One picker (round 91, the user's reading of four stills: several kinds of dropdown, no
 * standard, the native select's OS list). Every choice in the product is this component: a pill
 * (or a word, `kind: 'text'`; a chip, `kind: 'chip'`) that shows the value, and a list on the top
 * layer -- Radix's contract (the list at least the trigger's width with a floor, flips and shifts
 * at the viewport, re-measures on resize and on the trigger's own resize, type-ahead without a
 * search field, ↑↓ Home End Enter Esc), Apple's pop-up grammar (one-line rows, the ✓ on the
 * current one, never a full-width bar) and Linear's long list (a filter field over eight rows).
 * Below 640 px the list is a sheet from the foot. A row's `action` runs with its value; the
 * default `picker-choose` records the choice and tells the field's change handlers
 * (`Events.changed`), so a form field needs no action of its own -- `Picker.value(id)` reads it.
 * `multiple` keeps the list open and toggles rows; the trigger shows the count. */
const pickerChoice = (choice) => Array.isArray(choice) ? {value:choice[0], title:choice[1]} : choice;
/* A long list (round 94; measured: 1,264 sessions were 88 % of the Portfolio page's elements and
 * 130 ms of style on each open): a single-choice list longer than PICKER_WINDOW rows renders the
 * rows around the current one and keeps the rest here, by its popup id; its search draws the
 * first matches from here, and a foot line says how many more there are. */
const PICKER_WINDOW = 120;
const PICKER_ROWS = new Map();
/* A glyph and its word (the user, 2026-09-25: "字前面的图标感觉跟字对不齐"): one inline pair centred on
 * its line, never an inline svg on the text's baseline (`vertical-align: middle` sits on the x-height). */
const glyphWord = (glyph, words) => glyph ? html`<span class="glyph-word">${icon(glyph)}<span>${words}</span></span>` : words;
const pickerSearchText = (row) => [row.search, row.title, row.meta, row.value].filter(Boolean).map(String).join(' ').toLocaleLowerCase();
function pickerItems(rows, {action, multiple, chosen, selected}) {
  return rows.map((row) => {
    const on = multiple ? chosen.has(row.value) : row.value === selected;
    return btnAttrs(html`<span class="menu-words"><strong>${glyphWord(row.glyph, row.title)}</strong>${row.meta ? html`<small>${row.meta}</small>` : ''}</span>${row.theme ? '' : html`<span class="menu-check" aria-hidden="true">${icon('check')}</span>`}`, action, row.value, 'menu-row picker-option', html`role="option" aria-selected="${on}" data-picker-search="${pickerSearchText(row)}" data-picker-title="${row.word || row.trigger || row.title}"`);
  });
}
const pickerMore = (n) => n > 0 ? html`<p class="menu-more">${t('{n} more', {n: count(n)})}</p>` : ''; // the search field above says how to narrow
function picker(id, choices, {placeholder = '', disabled = false, action = 'picker-choose', selected = '', kind = 'long', labelId = '', label = '', searchThreshold = 8, attrs = '', multiple = false, face = null, context = ''} = {}) {
  const rows = choices.map(pickerChoice), chosen = multiple ? new Set(Array.isArray(selected) ? selected : []) : null, current = multiple ? null : rows.find((row) => row.value === selected), popupId = id + 'Popup', listId = id + 'List', searchable = rows.length > searchThreshold;
  // a picker with nothing to choose is held and says so: its face a word, no empty popup to open
  const empty = !multiple && !rows.length, held = disabled || empty;
  const count = multiple ? chosen.size : 0;
  const title = face ? face : multiple ? html`<span class="filter-chip-label">${placeholder}</span>${count ? html`<i></i><strong>${count}</strong>` : ''}`
    : kind === 'chip' ? html`<span class="filter-chip-label">${placeholder}</span>${current ? html`<i></i><strong>${current.title}</strong>` : ''}`
    : glyphWord(current?.glyph, current?.trigger || current?.title || placeholder || (empty ? t('None available') : '')); // round 84: the pill holds the value's glyph (Codex's `Cursor ⌄`)
  const windowed = !multiple && rows.length > PICKER_WINDOW;
  let shown = rows, more = 0;
  if (windowed) {
    const at = Math.max(0, rows.indexOf(current)), from = Math.max(0, Math.min(at - PICKER_WINDOW / 2, rows.length - PICKER_WINDOW));
    shown = rows.slice(from, from + PICKER_WINDOW); more = rows.length - shown.length;
    PICKER_ROWS.set(popupId, {rows, action, selected});
  } else PICKER_ROWS.delete(popupId);
  const items = html`${pickerItems(shown, {action, multiple, chosen, selected})}${pickerMore(more)}`;
  const labelled = labelId ? html` aria-labelledby="${labelId} ${id}"` : html` aria-label="${label}"`;
  const faceCls = kind === 'theme' ? 'ui-field picker-trigger theme-select' : kind === 'text' ? 'ui-field picker-trigger picker-text' : kind === 'chip' ? 'filter-chip-main picker-trigger picker-chip' : kind === 'switch' ? 'icon-btn compact picker-trigger picker-switch' : 'ui-field picker-trigger';
  const faceWords = kind === 'switch' ? icon('chevron') : html`<span class="picker-trigger-label">${title}</span><span class="picker-chevron" aria-hidden="true">${icon('chevron')}</span>`; // a switch is the chevron beside a name the head already says (E1)
  return html`<div class="picker-anchor" data-picker-id="${id}">${btnAttrs(faceWords, 'picker-menu', '', faceCls, html`id="${id}" aria-haspopup="dialog" aria-expanded="false" aria-controls="${popupId}"${labelled}${held ? ' disabled' : ''}${attrs ? html` ${attrs}` : ''}`)}<div class="menu menu-fixed picker-pop" id="${popupId}" role="dialog" aria-labelledby="${labelId || id}" data-picker-kind="${kind}"${multiple ? ' data-picker-multiple="true"' : ''} popover="manual" hidden>${context ? html`<p class="menu-context">${context}</p>` : ''}${searchable ? html`<div class="menu-search">${icon('search')}<input type="search" role="combobox" aria-expanded="true" aria-controls="${listId}" aria-autocomplete="list" placeholder="${t('Filter…')}" data-picker-search-input></div>` : ''}<div class="menu-list" id="${listId}" role="listbox"${multiple ? ' aria-multiselectable="true"' : ''}${labelId ? html` aria-labelledby="${labelId}"` : html` aria-label="${label}"`}>${items}<p class="menu-empty" hidden>${t('No matching choices')}</p></div></div></div>`;
}
/* A subject choice (round 93): a label and a picker that reads as a word, on the header's subject line. */
function subjectChoice(label, id, choices, {selected = '', disabled = false, action = 'picker-choose', attrs = '', kind = 'text'} = {}) {
  return html`<span class="subject-choice"><label id="${id}Label" for="${id}">${label}</label>${picker(id, choices, {selected, disabled, action, attrs, kind, labelId: id + 'Label'})}</span>`;
}
/* One segment of a segmented control (aria-pressed selection; the container carries `ui-segments`). */
function segBtn(label, action, value, pressed, extra = '', mutate = false) {
  return html`<button${extra ? ' ' + extra : ''} aria-pressed="${pressed}" class="" data-action="${action}" data-value="${(value)}"${mutate ? ' data-mutating="true"' : ''}${segmentAttr}>${label}</button>`;
}
/* Tabs (law 110; the user's reading of 2026-09-22: "the capsules are tacky"): the views of one
 * thing -- an object's pages, a panel's modes, an editor's two faces -- are words over the strip's
 * hairline, the current one in the ink over the indicator, the others muted (GitHub's
 * UnderlineNav, Vercel's and Sentry's tabs, measured). A choice of a value -- a range, a side, a
 * language -- stays a segmented control. An item: {word, action, value, on, count, off, cls,
 * attrs}; `off` is a view that cannot open yet, a word without an action. */
const tabAttr = ' data-control="tab"';
function tabStrip(label, items, cls = '') {
  const one = (x) => x.off
    ? html`<span class="tab-off${x.cls ? ' ' + x.cls : ''}" role="tab" aria-disabled="true" aria-selected="false"${tabAttr}>${x.word}</span>`
    : html`<button type="button" class="${x.cls || ''}" role="tab" aria-selected="${x.on ? 'true' : 'false'}" data-action="${x.action}" data-value="${x.value}"${x.attrs ? ' ' + x.attrs : ''}${tabAttr}>${x.word}${x.count ? html`<span class="tab-count num">${x.count}</span>` : ''}</button>`;
  // More (2026-09-22): the tabs a strip cannot hold -- from the end, never the current one -- as
  // rows of a menu beside the list; `fitTabs` measures the strip after each paint and on resize
  const key = 'tabs-' + textHash(label + '|' + cls);
  const rows = items.map((x, i) => (x.off ? '' : btnAttrs(html`<span class="menu-word">${x.word}</span>${x.count ? html`<span class="menu-note">${x.count}</span>` : ''}`, x.action, x.value, 'menu-row', html`role="menuitem" data-tab-index="${i}" hidden${x.attrs ? html` ${x.attrs}` : ''}`)));
  const more = html`<span class="tab-more-anchor" hidden>${btnAttrs(html`${t('More')}${icon('chevron')}`, 'tab-more', key, 'tab-more', html`aria-haspopup="menu" aria-expanded="false" data-control="tab"`)}<div class="menu" role="menu" aria-label="${t('More')}" hidden>${rows}</div></span>`;
  return html`<div class="tabs${cls ? ' ' + cls : ''}"><div class="tabs-list" role="tablist" aria-label="${label}">${items.map(one)}</div>${more}</div>`;
}
/* The strip that cannot hold its tabs shows More and moves the last tabs into it (the current
 * one stays); a strip that can shows every tab and no More. */
function fitTabs(root = document) {
  for (const strip of root.querySelectorAll('.tabs')) {
    const list = strip.querySelector(':scope > .tabs-list'), more = strip.querySelector(':scope > .tab-more-anchor');
    if (!list || !more) continue;
    const tabs = [...list.children];
    tabs.forEach((x) => { x.hidden = false; });
    more.hidden = true;
    if (list.scrollWidth > list.clientWidth + 1) {
      more.hidden = false;
      for (let i = tabs.length - 1; i >= 0 && list.scrollWidth > list.clientWidth + 1; i--) if (tabs[i].getAttribute('aria-selected') !== 'true') tabs[i].hidden = true;
    }
    for (const row of more.querySelectorAll('.menu-row[data-tab-index]')) row.hidden = !tabs[Number(row.dataset.tabIndex)]?.hidden;
  }
}
function tabMore(key) {
  const trigger = [...document.querySelectorAll('.tab-more')].find((b) => b.dataset.value === key), anchor = trigger?.closest('.tab-more-anchor');
  if (anchor) toggleMenu(anchor.querySelector('.menu'), trigger, {root: anchor});
}

/* ---- page chrome ---- */
/* The page tools (round 19, revised on the user's reading): the title row keeps one secondary and
 * one primary; everything else — Map, Focus and a page's own tools (`extra`) — lives in one ···
 * menu at the row's end (Apple's more button), an anchored popover of one-line rows (round 81).
 * The menu is rendered with the head, hidden, so a repaint compares it like any other markup;
 * `tools-menu` shows it. */
/* A group's head (round 52, Linear's "In Review · 9"): the label and its count on a band above
 * the rows; `tools` sits at its right (a Clear filters, a +). The foot line under a filtered
 * list says how many rows the filters hide and offers the way out. */
function groupHead(label, count, tools = '') {
  return html`<div class="group-head"><span>${label}</span><b class="num">${count == null ? '' : count}</b>${tools ? html`<span class="group-head-tools">${tools}</span>` : ''}</div>`;
}
/* ---- forms as inset grouped lists (round 83: macOS's form list, Codex's settings) ---- */
/* A group is a label on the ground and one box of rows, with an optional one-line note under
 * the box; `tools` is the label's one way to add a row (Codex's `Sources +`). */
function formGroup(label, rows, {note = '', tools = '', id = ''} = {}) { // law 148: a page of forms is a workspace -- every group its grouped-list box (Codex's settings), a read-only row or not
  return html`<section class="form-group"${id ? html` id="${id}"` : ''}><div class="form-label"><span>${label}</span>${tools ? html`<span class="group-head-tools">${tools}</span>` : ''}</div><div class="form-box" data-box="workspace">${rows}</div>${note ? html`<p class="form-note">${note}</p>` : ''}</section>`;
}
/* A row is a title, one line under it and one control at the right — the kinds of macOS's list:
 * a detail (a value in grey; `mono` for an id), a row that opens (`action` or `href`: the whole
 * row is the control and ends in ›), a pop-up (`picker`), a segment, a switch (`switchRow`), a
 * stepper (`stepper`), a pill (`btn`). Free text keeps the `.field` block. */
function formRow({title, line = '', control = '', detail = '', mono = false, action = '', value = '', href = '', cls = ''}) {
  const words = html`<span class="form-words"><span class="form-title">${title}</span>${line ? html`<small class="form-line">${line}</small>` : ''}</span>`;
  const right = html`${detail !== '' ? html`<span class="form-detail${mono ? ' mono' : ''}">${detail}</span>` : ''}${control}`;
  if (href) return html`<a class="form-row form-row-open${cls ? ' ' + cls : ''}" href="${href}" aria-label="${title}">${words}<span class="form-control">${right}<span class="form-tail" aria-hidden="true">›</span></span></a>`;
  if (action) return btnAttrs(html`${words}<span class="form-control">${right}<span class="form-tail" aria-hidden="true">›</span></span>`, action, value, `form-row form-row-open${cls ? ' ' + cls : ''}`, html`aria-label="${title}"`); // the row's name is its title
  return html`<div class="form-row${cls ? ' ' + cls : ''}">${words}<span class="form-control">${right}</span></div>`;
}
/* A boolean row: the native checkbox drawn as a switch (round 82), the row its label. */
function switchRow(id, on, title, line = '', {disabled = false} = {}) {
  const says = disabled && line; // a held switch's own line says why (CT7): it describes the switch
  return html`<label class="form-row" for="${id}"><span class="form-words"><span class="form-title">${title}</span>${line ? html`<small class="form-line"${says ? html` id="${id}-why"` : ''}>${line}</small>` : ''}</span><span class="form-control"><input id="${id}" class="switch" type="checkbox" role="switch"${on ? ' checked' : ''}${disabled ? ' disabled' : ''}${says ? html` aria-describedby="${id}-why"` : ''}></span></label>`;
}
/* A short question with two or three answers (round 82, macOS's alert): a bold title, one line of
 * body, equal pills; anything with a body of facts stays a dialog. */
function alertDialog(title, body, footer) {
  openDialog('', title, html`<p>${body}</p>`, footer, 'alert');
}
/* A stepper: `−  value  +`, and the way back to the default while away from it. */
function stepper(value, action, {label = '', reset = '', resetAction = '', resetValue = ''} = {}) {
  return html`<span class="stepper" role="group" aria-label="${label}">${btn('−', action, '-1', 'button compact')}<b class="num">${value}</b>${btn('+', action, '1', 'button compact')}${reset ? btn(reset, resetAction, resetValue, 'text-btn') : ''}</span>`;
}
function listFoot(hidden, clear) {
  return hidden > 0 ? html`<p class="list-foot">${t('{n} hidden by the filters', {n: count(hidden)})}${clear ? html` · ${clear}` : ''}</p>` : '';
}
/* A key as caps (round 51): "G then H" → two caps and the word, "Ctrl K" → two caps, "J / ↓" →
 * two caps and a slash; the sheet, the menus and the command menu all read it. */
function keycap(keys) {
  return html`<span class="keycap" aria-label="${keys}">${String(keys).split(' ').map((part) => part === 'then' || part === '/' ? html`<i>${part === 'then' ? t('then') : '/'}</i>` : html`<kbd>${part}</kbd>`)}</span>`;
}
function menuRow({ic = '', word = '', note = '', key = '', title = '', action = '', value = '', href = '', page = '', checked = null, disabled = false, pressed = null, cls = ''} = {}) {
  const inner = html`${checked !== null ? html`<span class="menu-check" aria-hidden="true">${icon('check')}</span>` : ''}${ic ? icon(ic) : ''}<span class="menu-word">${word}</span>${note || key ? html`<span class="menu-note">${note || key}</span>` : ''}`;
  const attrs = html`role="${checked !== null ? 'menuitemradio' : 'menuitem'}"${checked !== null ? html` aria-checked="${checked}"` : ''}${pressed !== null ? html` aria-pressed="${pressed}"` : ''}${disabled ? html` aria-disabled="true"` : ''}${title ? html` data-tip="${title}"` : ''}`;
  if (href) return html`<a class="menu-row ${cls}" href="${href}"${page ? html` data-page="${page}"` : ''} ${attrs}>${inner}</a>`;
  return btnAttrs(inner, action, value, 'menu-row ' + cls, attrs);
}
function railTools(extra = []) {
  const focus = typeof Inspect !== 'undefined' && Inspect.focus;
  // round 92: Facts and Record are the panel's tabs, opened from the ⓘ chip and Ctrl ]; a panel is a part of the view, not a command
  const items = [...extra, {ic: 'link', action: 'copy-link', word: t('Copy link'), why: t('The address of exactly this view'), key: 'L'},
    {ic: 'fork', action: 'research-map', word: t('Map'), why: t('Open research map')},
    {ic: 'eye', action: 'focus-toggle', word: focus ? t('Exit Focus') : t('Focus'), why: focus ? t('Exit Focus mode') : t('Enter Focus mode'), pressed: Boolean(focus)},
    {ic: 'keyboard', action: 'shortcuts', word: t('Keyboard shortcuts'), why: t('Every key the page answers'), key: '?'}];
  const item = (i) => menuRow({ic: i.ic, word: i.word, key: i.key || '', title: i.why || '', action: i.action, value: i.value || '', pressed: i.pressed === undefined ? null : Boolean(i.pressed)});
  return html`<span class="rail-tools">${btnAttrs(icon('more'), 'tools-menu', '', 'icon-btn rail-more', html`aria-label="${t('More')}" data-tip="${t('More · page tools')}" aria-haspopup="menu" aria-expanded="false"`)}<div class="menu" role="menu" aria-label="${t('Page tools')}" hidden>${items.map(item)}</div></span>`;
}
/* A reading pane (round 92; law 149): the lane's reading column -- the Lab's PLAN, the desk's item --
 * a detail beside the page's list from the pane step up and the pane's layer below it; never the
 * window's column. It takes focus as a region; `close` is the action that closes it. */
function readingPane(title, kind, body, close, cls = '', by = null) {
  return html`<aside class="reading-pane${cls ? ' ' + cls : ''}" data-layer="detail" tabindex="-1" aria-label="${title}"${by ? html` data-by-action="${by[0]}" data-by-value="${by[1] ?? ''}"` : ''}>${typeof Window !== 'undefined' && Window.backMarkup ? Window.backMarkup() : ''}<header class="reading-pane-head"><div>${kind ? html`<p class="caption">${kind}</p>` : ''}<h2>${title}</h2></div>${btnAttrs(icon('close'), close, '', 'icon-btn', html`aria-label="${t('Close the reading')}" data-tip="${t('Close the reading')}"`)}</header><div class="reading-pane-body">${body}</div></aside>`;
}
/* The detail's split (law 149; the Evidence reading's, the user's textbook): a list and the window's
 * detail it opened, side by side -- only that list gives way; the page's head, its figures and its
 * other sections keep the page's width. `modes` are the window's details the list hosts (a Task's
 * record, a factor's or a fold's evidence, a holding, a saved object). */
function detailSplit(content, modes, {over = false, pane: suppliedPane} = {}) {
  const pane = suppliedPane ?? (typeof Window !== 'undefined' && Window.detailPane ? Window.detailPane(modes) : '');
  // `over`: the detail floats over its list, which keeps its width (the Portfolio's holdings: the user, 2026-09-24)
  return html`<div class="lane-split${pane ? ' has-reading' : ''}" data-stack-box="section" data-detail-host="${[].concat(modes).join(' ')}"${over ? ' data-detail-over' : ''}><div class="lane-main">${content}</div>${pane}</div>`;
}
/* A leave (round 94; Apple's both ways, Linear's short exits): the element takes `data-leaving`
 * and its stylesheet plays the entrance in reverse over --dur-1; when the engine says that
 * animation finished, `done` hides it. At once when the page is hidden, under reduced motion or
 * where no rule animates the leave; a show in between (`cancelLeave`) drops the pending `done`. */
function leave(el, done) {
  if (!el || el.hidden || !el.isConnected || document.hidden || matchMedia('(prefers-reduced-motion: reduce)').matches) { if (el) delete el.dataset.leaving; return done(); }
  el.dataset.leaving = 'true';
  const playing = typeof el.getAnimations === 'function' ? el.getAnimations().filter((a) => a.playState !== 'finished') : [];
  if (!playing.length) { delete el.dataset.leaving; return done(); }
  Promise.all(playing.map((a) => a.finished)).then(() => { if (el.dataset.leaving) { delete el.dataset.leaving; done(); } }, () => {}); // a cancelled leave (a show in between) does nothing
}
function cancelLeave(el) { if (el) delete el.dataset.leaving; }
/* Show or hide the head's ··· menu; while it is open, a click elsewhere or Escape closes it and a
 * chosen item closes it before the item's own action runs. */
/* One menu opener (round 55): shows the menu beside its trigger (or at a pointer), closes it on
 * a choice, an outside click or Esc (one layer per Esc), and puts focus on its search field or
 * first row; any other open menu closes first. */
function toggleMenu(pop, trigger, {root = trigger?.parentElement, at = null, stay = false} = {}) { // `stay`: a choice inside keeps it open
  if (!pop || !trigger) return;
  const close = () => {
    trigger.setAttribute('aria-expanded', 'false');
    document.removeEventListener('click', away, true); document.removeEventListener('keydown', keys, true); document.removeEventListener('scroll', scrolled, true);
    leave(pop, () => { pop.hidden = true; pop.classList.remove('menu-at-pointer'); });
  };
  const away = (e) => { if (!root.contains(e.target) || (!stay && e.target.closest('.menu [data-action]:not([data-stay])'))) close(); };
  const keys = (e) => { if (e.key === 'Escape') { e.stopPropagation(); close(); trigger.focus(); } };
  const scrolled = (e) => { if (e?.target instanceof Node && pop.contains(e.target)) return; close(); }; // the page's scroll moves the trigger away; the menu's own does not
  if (!pop.hidden && !pop.dataset.leaving && !at) return close();
  for (const other of document.querySelectorAll('.menu:not(.picker-pop):not([hidden])')) { if (other === pop) continue; cancelLeave(other); other.hidden = true; other.classList.remove('menu-at-pointer'); other.parentElement?.querySelector('[aria-expanded="true"]')?.setAttribute('aria-expanded', 'false'); }
  cancelLeave(pop); pop.hidden = false; trigger.setAttribute('aria-expanded', 'true');
  // inside a clipping surface (a panel's rounded box) the menu would be cut at the surface's edge:
  // it is placed in the top layer at its trigger instead, and closes when the page scrolls
  let clipped = false;
  for (let el = pop.parentElement; el && el !== document.body && !at; el = el.parentElement) if (/hidden|clip|auto|scroll/.test(getComputedStyle(el).overflowY)) { clipped = true; break; }
  if (at || clipped) {
    const r = layoutRect(trigger), w = pop.offsetWidth, h = pop.offsetHeight, vw = viewW(), vh = viewH();
    const edge = param('viewport-inset'), gap = param('popover-gap');
    const x = at ? at.x : r.right - w, y = at ? at.y : r.bottom + gap;
    const up = y + h > vh - edge && (at ? at.y : r.top) - h - gap > edge;
    pop.classList.add('menu-at-pointer');
    pop.style.setProperty('--menu-x', Math.max(edge, Math.min(x, vw - w - edge)) + 'px');
    pop.style.setProperty('--menu-y', (up ? (at ? at.y : r.top) - h - gap : Math.max(edge, Math.min(y, vh - h - edge))) + 'px');
    document.addEventListener('scroll', scrolled, true);
  }
  else fitMenu(pop);
  setTimeout(() => { document.addEventListener('click', away, true); document.addEventListener('keydown', keys, true); }, 0);
  (pop.querySelector('[data-menu-filter], [data-picker-search-input]') || pop.querySelector('button, a'))?.focus({preventScroll: true});
}
function toolsMenu() {
  const tools = document.querySelector('#top .rail-tools') || document.querySelector('#main .rail-tools'); // the page's menu stands beside the path (N1)
  if (tools) toggleMenu(tools.querySelector('.menu'), tools.querySelector('.rail-more'), {root: tools});
}
/* The filter bar's grammar (round 95, Linear's): a stated filter is a clause chip -- the field,
 * the operator word (`is`, `is any of`), the values, its × -- and clauses combine with `and`;
 * `+ Filter` is the one picker of the fields not yet in a clause (a bar with one field shows that
 * field's chip in its place), and choosing a field opens its values at once; `Clear` removes them
 * all. A clause's chip is the field's own picker (round 91: multiple where the field allows),
 * so its values change in place; the choice reaches the owner through the change handlers
 * (`data-filter-name`), the field through `filter-field`, and the route carries the clauses. */
const chipValues = (value) => (value === '' || value === 'all' || value == null ? [] : String(value).split(',').filter(Boolean));
function chip(label, value, {name, options = [], multiple = false, pending = false} = {}) {
  const chosen = chipValues(value), active = chosen.length > 0, words = chosen.map((v) => { const o = options.map(pickerChoice).find((x) => String(x.value) === v); return o ? o.title : v; });
  const face = active ? html`<span class="filter-chip-label">${label}</span><span class="filter-op">${t(chosen.length > 1 ? 'is any of' : 'is')}</span><strong>${words.join(', ')}</strong>` : pending ? html`<span class="filter-chip-label">${label}</span><span class="filter-op">${t('is')}</span><strong>…</strong>` : null;
  const trigger = picker('filter-' + name, options, {kind: 'chip', multiple, selected: multiple ? chosen : chosen[0] || '', placeholder: label, label, face, attrs: html`data-filter-name="${name}"${pending ? ' data-filter-pending' : ''}`});
  return html`<span class="filter-chip${active ? ' is-active' : ''}${pending ? ' is-pending' : ''}">${trigger}${active || pending ? btnAttrs('×','filter-clear',name,'filter-chip-clear',html`aria-label="${t('Clear {label}', {label})}"`) : ''}</span>`;
}
function filterBar(fields, {pending = '', clear = '', clearValue = ''} = {}) {
  const clauses = fields.filter((f) => chipValues(f.value).length || f.name === pending), rest = fields.filter((f) => !clauses.includes(f));
  const stated = clauses.map((f, i) => html`${i ? html`<span class="filter-and">${t('and')}</span>` : ''}${chip(f.label, f.value, {name: f.name, options: f.options, multiple: f.multiple, pending: f.name === pending && !chipValues(f.value).length})}`);
  const add = !rest.length ? '' : fields.length === 1 ? chip(rest[0].label, rest[0].value, {name: rest[0].name, options: rest[0].options, multiple: rest[0].multiple})
    : html`<span class="filter-chip filter-add">${picker('filter-add-' + fields.map((f) => f.name).join('-'), rest.map((f) => [f.name, f.label]), {kind: 'chip', action: 'filter-field', placeholder: t('Filter'), label: t('Add a filter'), face: html`${icon('plus')}<span class="filter-chip-label">${t('Filter')}</span>`, searchThreshold: 99})}</span>`;
  const clearing = clauses.some((f) => chipValues(f.value).length) && clear ? btnAttrs(t('Clear'), clear, clearValue, 'text-btn filter-clear-all') : '';
  return html`<div class="filter-clauses">${stated}${add}${clearing}</div>`;
}
/* Round 56: a property is a chip you press (Linear's). The label in the muted ink, the value in
 * the text ink, its key as a cap; pressing it (or the key on its page) opens the menu with a
 * context line naming the object, a search field past eight rows and the current value checked.
 * A read-only property is the same chip: its rows are none, and its press says who recorded it. */
function propertyChip(label, value, {icon: ic = '', key = '', rows = [], context = '', action = '', current = '', note = '', id = '', press = ''} = {}) {
  const cap = key ? keycap(key.toUpperCase()) : '';
  const face = html`${ic ? icon(ic) : ''}<span class="filter-chip-label">${label}</span>${value ? html`<i></i><strong>${value}</strong>` : ''}${cap}`;
  if (press) return html`<span class="filter-chip property-chip is-active"${key ? html` data-property="${key}"` : ''}>${btnAttrs(face, press, '', 'filter-chip-main')}</span>`; // a chip that goes somewhere, no menu
  // round 91: a chip with choices is the one picker (the chip face, the top layer, the rules); a chip with a note alone keeps its note
  if (rows.length) return html`<span class="filter-chip property-chip${value ? ' is-active' : ''}"${id ? html` id="${id}"` : ''}${key ? html` data-property="${key}"` : ''}>${picker('prop-' + (id || key || textHash(String(label))), rows.map((r) => ({value: r.value, title: r.title, meta: r.meta || ''})), {kind: 'chip', action, selected: current, placeholder: label, label: t('Change {label}', {label}), face, context, searchThreshold: 7})}</span>`;
  // round 93: a read-only chip is not a menu; its note (and its context) is the tooltip's
  return html`<span class="filter-chip property-chip${value ? ' is-active' : ''} is-read"${id ? html` id="${id}"` : ''}${key ? html` data-property="${key}"` : ''}><span class="filter-chip-main" data-tip="${[context, note || t('Recorded by the owner; not editable here.')].filter(Boolean).join(' · ')}" tabindex="0">${face}</span></span>`;
}
/* Round 57: display options (Linear's Shift V). Everything about how a list is shown lives in one
 * popover per list: Grouping and Ordering as checked rows, Display properties as toggle chips. A
 * list declares its spec where it renders ({groupings, orderings, properties, apply}); the
 * viewer's choices are kept per list in the preference store (`display.<list>`), never in the
 * route — the route keeps the filters, so a shared link shows the same records in the viewer's
 * own display. A property the owner did not supply is not offered. C3 (law 142): a list may also
 * say what it shows (`shows`, and `members` for one member) -- the thread's Display; a choice the
 * route holds is passed as `current` and read from there. */
const DISPLAY_SPECS = {};
function displayState(name, spec = DISPLAY_SPECS[name]) {
  const saved = readPreference('display.' + name) || {};
  const pick = (rows, value) => rows?.some(([k]) => k === value) ? value : (rows?.[0]?.[0] || '');
  const props = Object.fromEntries((spec?.properties || []).map(([key, , on = true]) => [key, saved.props && key in saved.props ? Boolean(saved.props[key]) : on]));
  return {show: spec?.current?.show ?? pick(spec?.shows, saved.show), group: pick(spec?.groupings, saved.group), order: pick(spec?.orderings, saved.order), props, density: saved.density === 'compact' ? 'compact' : 'comfortable'};
}
function setDisplay(name, patch) {
  const state = displayState(name);
  const next = {...state, ...patch, props: {...state.props, ...(patch.props || {})}};
  savePreference('display.' + name, next);
  return next;
}
function displayMenuBody(name) {
  const spec = DISPLAY_SPECS[name], s = displayState(name, spec);
  const radio = (section, key, rows, current, least = 2) => rows?.length >= least ? html`<p class="menu-section">${t(section)}</p>${rows.map(([value, label]) => menuRow({word: label, action: 'display-set', value: `${name}:${key}:${value}`, checked: value === current}))}` : '';
  const props = spec.properties?.length ? html`<p class="menu-section">${t('Display properties')}</p><div class="display-props">${spec.properties.map(([key, label]) => btnAttrs(label, 'display-set', `${name}:prop:${key}`, 'filter-chip display-prop', html`aria-pressed="${s.props[key]}"`))}</div>` : '';
  const density = spec.density ? radio('Density', 'density', [['comfortable', t('Comfortable')], ['compact', t('Compact')]], s.density) : '';
  return html`${radio('Show', 'show', spec.shows, s.show)}${radio('One member', 'show', spec.members, s.show, 1)}${radio('Grouping', 'group', spec.groupings, s.group)}${radio('Ordering', 'order', spec.orderings, s.order)}${props}${density}`;
}
function displayOptions(name, spec) {
  DISPLAY_SPECS[name] = spec;
  return html`<span class="display-options" data-display="${name}">${btnAttrs(icon('sliders'), 'display-menu', name, 'icon-btn display-trigger', html`aria-label="${t('Display options')}" data-tip="${t('Display options')}" data-tip-key="Shift V" aria-haspopup="menu" aria-expanded="false"`)}<div class="menu display-menu" role="menu" aria-label="${t('Display options')}" hidden></div></span>`; // filled when opened, so a repaint sees the same markup and keeps it open
}
function toggleDisplayMenu(name = document.querySelector('#main .display-options')?.dataset.display) {
  const root = name && document.querySelector(`.display-options[data-display="${name}"]`);
  if (!root) return;
  const pop = root.querySelector('.menu');
  if (pop.hidden) pop.innerHTML = displayMenuBody(name);
  toggleMenu(pop, root.querySelector('.display-trigger'), {root, stay: true});
}
function chooseDisplay(value) {
  const [name, key, v] = value.split(':'), spec = DISPLAY_SPECS[name];
  if (!spec) return;
  const state = key === 'prop' ? setDisplay(name, {props: {[v]: !displayState(name).props[v]}}) : setDisplay(name, {[key]: v});
  spec.apply?.(state);
  // the open menu reads what now holds, after the list applied it: a choice the route keeps (the
  // thread's one member) is read back from the spec its repaint declared
  const pop = document.querySelector(`.display-options[data-display="${name}"] .menu`);
  if (pop && !pop.hidden) pop.innerHTML = displayMenuBody(name);
}
/* The key of a property (round 56): its chip takes focus and opens. */
function pressProperty(key) {
  const button = document.querySelector(`#main .property-chip[data-property="${key}"] .filter-chip-main`);
  if (!button) return false;
  button.focus({preventScroll: false});
  if (button.dataset.action === 'picker-menu') Picker.open(button); else if (button.dataset.tip) Tip.show(button); else button.click(); // a read-only chip shows its note
  return true;
}
/* Round 56: the composer (Linear's). One sheet for making a new thing: the context line above,
 * a title field and a body field with placeholders instead of labels, the properties as chips
 * along the bottom, more fields folded under a disclosure, one primary button; `Ctrl Enter`
 * submits, `Esc` closes. The composer is markup: the caller's save reads the same ids. */
function composer({eyebrow, context, title, body, chips = '', more = '', primary, note = ''}) {
  return html`<div class="composer">${context ? html`<p class="composer-context">${context}</p>` : ''}<label class="sr-only" for="${title.id}">${title.label}</label><input id="${title.id}" class="composer-title" placeholder="${title.label}" value="${title.value || ''}" autocomplete="off"><label class="sr-only" for="${body.id}">${body.label}</label><textarea id="${body.id}" class="composer-body" placeholder="${body.label}" rows="${body.rows || 4}">${body.value || ''}</textarea>${chips ? html`<div class="composer-chips">${chips}</div>` : ''}${more}${note ? html`<p class="composer-note">${note}</p>` : ''}</div>`;
}
/* A list row exposes one quiet ···, never a second trailing control. The menu reuses the
 * functional layer and closes on selection, Escape or an outside click. */
function rowMenu(items) {
  const peekItem = menuRow({ic: 'eye', word: t('Peek'), key: 'Space', action: 'row-peek'});
  return html`<span class="row-menu">${btnAttrs(icon('more'), 'row-menu', '', 'icon-btn row-more', html`aria-label="${t('Row actions')}" data-tip="${t('Row actions')}" aria-haspopup="menu" aria-expanded="false"`)}<span class="menu" role="menu" aria-label="${t('Row actions')}" hidden>${peekItem}${items}</span></span>`;
}
function toggleRowMenu(root, at = null) { // the action passes its (empty) value; the right-click passes the row's menu
  if (!(root instanceof Element)) root = document.activeElement?.closest?.('.row-menu');
  if (root) toggleMenu(root.querySelector('.menu'), root.querySelector('.row-more'), {root, at});
}
/* Popovers keep their anchor but choose the side with room. Width and scrolling remain CSS-owned;
 * these classes are the only geometry decision JS makes after measuring the open layer (a menu that
 * opens rightward from its trigger -- the path's ··· -- turns back at the window's right edge). */
function fitMenu(pop) {
  pop.classList.remove('menu-align-start', 'menu-align-end', 'menu-open-up');
  pop.style.removeProperty('--menu-limit');
  const edge = param('viewport-inset');
  let box = layoutRect(pop);
  if (box.left < edge) { pop.classList.add('menu-align-start'); box = layoutRect(pop); }
  else if (box.right > viewW() - edge) { pop.classList.add('menu-align-end'); box = layoutRect(pop); }
  const anchor = pop.parentElement ? layoutRect(pop.parentElement) : null, below = viewH() - (anchor?.bottom || box.top) - edge, above = (anchor?.top || box.top) - edge;
  const opensUp = box.height > below && above > below;
  if (opensUp) pop.classList.add('menu-open-up');
  pop.style.setProperty('--menu-limit', `${Math.max(edge, opensUp ? above : below)}px`);
}
const Picker = (() => {
  let active = null;
  const optionsOf = (pop) => [...pop.querySelectorAll('.picker-option:not([hidden])')];
  function position(resetSide = false) {
    if (!active || !active.trigger.isConnected || !active.pop.isConnected) return close(false);
    const {trigger, pop} = active, rect = layoutRect(trigger), edge = param('viewport-inset'), gap = param('popover-gap'), vw = viewW(), vh = viewH(), maxWidth = Math.max(0, vw - edge * 2);
    /* A picker is the open face of its trigger, not a second component with its own measure.
     * Long labels wrap inside that measure. This keeps Lab, Team and the theme chooser aligned
     * at every responsive width instead of widening or narrowing them by content kind. */
    // round 91: the list is at least the xs width and at most the md width (or the viewport), never a wall of wrapped words;
    // a pop-up in a form row (round 83) hangs from the pill's right edge; below 640 the list is a sheet from the foot
    const sheet = vw < BREAKPOINTS.window.phone;
    if (sheet) { pop.dataset.sheet = 'true'; pop.style.removeProperty('--picker-left'); pop.style.removeProperty('--picker-top'); pop.style.setProperty('--picker-width', `${Math.max(0, vw - edge * 2)}px`); pop.style.setProperty('--picker-max', `${Math.round(vh * 0.6)}px`); pop.dataset.side = 'sheet'; return; }
    delete pop.dataset.sheet;
    // the width: the fitted measure plus the classic scrollbar the list shows at its placed height (an overlay one takes none)
    const widthNow = () => { const w = Math.min(maxWidth, active.width ? Math.min(param('width-md'), active.width + listBar(pop)) : Math.max(rect.width, param('width-xs'))); pop.style.setProperty('--picker-width', `${w}px`); return w; };
    const compact = trigger.closest('.form-row') !== null;
    widthNow();
    // the natural height is the content's -- the box's own parts and the whole list -- never the box a previous placement clamped
    const list = pop.querySelector('.menu-list'), content = list ? pop.offsetHeight - list.clientHeight + list.scrollHeight : pop.scrollHeight;
    const viewportHeight = Math.max(param('control-xl'), vh - edge * 2), naturalHeight = Math.min(content, viewportHeight);
    const below = Math.max(0, vh - rect.bottom - gap - edge), above = Math.max(0, rect.top - gap - edge);
    if (resetSide) active.side = '';
    let up = active.side === 'top';
    if (!active.side) {
      // Stay below when roughly three rows fit; scrolling is calmer than a large surface jumping
      // above its control. Flip only near the viewport foot where a useful list cannot open.
      up = below < Math.min(naturalHeight, 3 * param('row-regular')) && above > below;
    } else if ((!up && below < param('control-xl') && above > below) || (up && above < param('control-xl') && below > above)) {
      up = !up;
    }
    active.side = up ? 'top' : 'bottom';
    const available = Math.max(param('control-xl'), up ? above : below), height = Math.min(naturalHeight, available);
    pop.style.setProperty('--picker-max', `${height}px`);
    const width = widthNow(); // at this height the list may now scroll: its bar is measured after the height is set
    const left = Math.max(edge, Math.min(compact ? rect.right - width : rect.left, vw - edge - width));
    const top = up ? Math.max(edge, rect.top - gap - height) : Math.min(vh - edge - height, rect.bottom + gap);
    pop.dataset.side = up ? 'top' : 'bottom';
    pop.style.setProperty('--picker-left', `${left}px`); pop.style.setProperty('--picker-top', `${top}px`);
  }
  /* The list's width (round 94, the user's rule: "the entry button and the menu must align"):
   * the trigger is the measure -- the list starts at the trigger's left edge and is the
   * trigger's width; only when the titles need more does it grow to its widest title (at most
   * 400 px), and a title within 6 px of the trigger snaps to it (the ✓'s box gives the pixels).
   * Measured on open and on each search from the rows themselves (`measuring`: the list at its
   * max-content with every row drawn and no metas); the metas then take what is left of the
   * row and give way whole before a title loses a letter; a classic scrollbar's width is added
   * once the list is placed. `position()` clamps the width to the viewport on every move. */
  const listBar = (pop) => { const list = pop.querySelector('.menu-list'); return list ? list.offsetWidth - list.clientWidth : 0; };
  function fit() {
    if (!active) return;
    const {trigger, pop} = active, list = pop.querySelector('.menu-list');
    if (!list || viewW() < BREAKPOINTS.window.phone) { active.width = 0; return; }
    // layout widths (`offsetWidth`), not client rects: the entrance plays at scale .96 while this runs;
    // the menu's own padding and border from its style (before its first placement the container is 10 px wide)
    const cs = getComputedStyle(pop), pad = ['paddingLeft', 'paddingRight', 'borderLeftWidth', 'borderRightWidth'].reduce((a, k) => a + (parseFloat(cs[k]) || 0), 0) + 2;
    pop.classList.add('measuring');
    const natural = list.offsetWidth + pad;
    pop.classList.remove('measuring');
    const measure = trigger.offsetWidth;
    active.width = natural <= measure + param('popover-gap') ? measure : Math.max(measure, Math.min(param('width-md'), natural));
  }
  function place() {
    if (!active) return;
    fit(); position();
  }
  // a resize reflows the page for a few frames after the event (fonts, wraps, the dock's fold): the
  // list follows its trigger frame by frame for half a second, then rests
  let settling = 0;
  const settle = () => { const until = performance.now() + 500; cancelAnimationFrame(settling); const tick = () => { if (!active) return; position(false); if (performance.now() < until) settling = requestAnimationFrame(tick); }; settling = requestAnimationFrame(tick); };
  // the page's scroll moves the trigger; the list's own scroll moves nothing
  const resize = () => { position(true); settle(); }, scroll = (e) => { if (active && e?.target instanceof Node && active.pop.contains(e.target)) return; position(false); };
  let observer = null; // the trigger's own resize (the dock's fold moves it without a window resize)
  function finish(pop) {
    for (const trigger of document.querySelectorAll('.picker-trigger')) if (trigger.getAttribute('aria-controls') === pop.id) trigger.setAttribute('aria-expanded', 'false');
    leave(pop, () => {
      try { if (pop.matches(':popover-open')) pop.hidePopover?.(); } catch (_) {}
      pop.hidden = true;
      pop.removeAttribute('data-side');
    });
  }
  function close(restore = false) {
    if (!active) return false;
    const {trigger, pop, dirty} = active; active = null;
    if ((dirty || trigger?.dataset.filterPending !== undefined) && trigger && typeof Events !== 'undefined' && Events.changed) Events.changed(trigger, value(trigger.id)); // a multi-select applies when it closes; a pending clause reports its emptiness too (round 95)
    trigger.setAttribute('aria-expanded', 'false'); document.removeEventListener('pointerdown', outside, true); window.removeEventListener('resize', resize); window.removeEventListener('scroll', scroll, true);
    if (observer) { observer.disconnect(); observer = null; }
    finish(pop);
    if (restore && trigger.isConnected) trigger.focus({preventScroll:true});
    return true;
  }
  function outside(event) { if (active && !active.pop.contains(event.target) && !active.trigger.contains(event.target)) close(false); }
  function open(trigger) {
    const pop = document.getElementById(trigger.getAttribute('aria-controls'));
    if (!pop) return;
    if (active?.trigger === trigger) return close(true);
    close(false);
    for (const openPop of document.querySelectorAll('.picker-pop:not([hidden])')) finish(openPop);
    active = {trigger, pop, side:''};
    cancelLeave(pop); pop.hidden = false; pop.showPopover?.(); trigger.setAttribute('aria-expanded', 'true'); place();
    setTimeout(() => document.addEventListener('pointerdown', outside, true), 0); window.addEventListener('resize', resize); window.addEventListener('scroll', scroll, true);
    if (typeof ResizeObserver !== 'undefined') { observer = new ResizeObserver(() => { if (active?.trigger === trigger) position(false); }); observer.observe(trigger); }
    (pop.querySelector('[data-picker-search-input]') || pop.querySelector('[aria-selected="true"]') || pop.querySelector('.picker-option'))?.focus({preventScroll:true});
    pop.querySelector('.picker-option[aria-selected="true"]')?.scrollIntoView({block: 'center'}); // the current choice in view (Apple's)
  }
  /* A choice (round 91): the row is marked, the trigger reads it, the list closes (a multi-select
   * list stays), and a field without an action of its own tells the change handlers. */
  function choose(option) {
    const pop = option.closest('.picker-pop'); if (!pop) return false;
    const id = pop.id.replace(/Popup$/, ''), trigger = document.getElementById(id), multiple = pop.dataset.pickerMultiple === 'true';
    if (multiple) {
      option.setAttribute('aria-selected', option.getAttribute('aria-selected') === 'true' ? 'false' : 'true');
      const n = pop.querySelectorAll('.picker-option[aria-selected="true"]').length, label = trigger?.querySelector('.picker-trigger-label');
      if (label) { const word = label.querySelector('.filter-chip-label')?.textContent || label.firstChild?.textContent || ''; label.innerHTML = html`<span class="filter-chip-label">${word}</span>${n ? html`<i></i><strong>${n}</strong>` : ''}`; }
      if (active) active.dirty = true;
      return true;
    }
    select(id, option.dataset.value); close(false);
    if (option.dataset.action === 'picker-choose' && trigger && typeof Events !== 'undefined' && Events.changed) Events.changed(trigger, option.dataset.value);
    return true;
  }
  const value = (id) => { const pop = document.getElementById(id + 'Popup'); if (!pop) return document.getElementById(id)?.value ?? ''; const on = [...pop.querySelectorAll('.picker-option[aria-selected="true"]')].map((o) => o.dataset.value); return pop.dataset.pickerMultiple === 'true' ? on : (on[0] ?? PICKER_ROWS.get(pop.id)?.selected ?? ''); };
  let typed = '', typedAt = 0; // type-ahead on a list without a search field (Radix's, Apple's)
  function toggle() { const trigger = document.activeElement?.closest?.('.picker-trigger'); if (trigger) open(trigger); }
  function filter(input) {
    const pop = input.closest('.picker-pop'), query = input.value.trim().toLocaleLowerCase(); let shown = 0;
    const held = PICKER_ROWS.get(pop.id);
    if (held) { // a windowed list: the first matches, drawn from the rows the page holds
      const hits = held.rows.filter((row) => !query || pickerSearchText(row).includes(query)), list = pop.querySelector('.menu-list'), empty = list.querySelector('.menu-empty');
      shown = Math.min(hits.length, PICKER_WINDOW);
      for (const el of list.querySelectorAll('.picker-option, .menu-more')) el.remove();
      empty.insertAdjacentHTML('beforebegin', html`${pickerItems(hits.slice(0, PICKER_WINDOW), {action: held.action, multiple: false, chosen: null, selected: value(pop.id.replace(/Popup$/, '')) || held.selected})}${pickerMore(hits.length - shown)}`);
      empty.hidden = shown > 0; list.scrollTop = 0; place(); return;
    }
    for (const option of pop.querySelectorAll('.picker-option')) { option.hidden = Boolean(query) && !option.dataset.pickerSearch.includes(query); if (!option.hidden) shown++; }
    pop.querySelector('.menu-empty').hidden = shown > 0; pop.querySelector('.menu-list').scrollTop = 0; place(); // the first match in view
  }
  function keydown(event) {
    const trigger = event.target.closest?.('.picker-trigger');
    if (!active && trigger && ['ArrowDown','ArrowUp'].includes(event.key)) { event.preventDefault(); open(trigger); return true; }
    if (!active) return false;
    if (event.key === 'Escape') { event.preventDefault(); close(true); return true; }
    if (event.key === 'Tab') { close(false); return false; }
    if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey && !active.pop.querySelector('[data-picker-search-input]')) {
      const now = Date.now(); typed = (now - typedAt < 600 ? typed : '') + event.key.toLocaleLowerCase(); typedAt = now;
      const hit = optionsOf(active.pop).find((o) => (o.dataset.pickerTitle || '').toLocaleLowerCase().startsWith(typed));
      if (hit) { hit.focus({preventScroll:true}); hit.scrollIntoView({block:'nearest'}); }
      event.preventDefault(); return true;
    }
    if (!['ArrowDown','ArrowUp','Home','End'].includes(event.key)) return false;
    const items = optionsOf(active.pop); if (!items.length) return false;
    const at = items.indexOf(document.activeElement); let next = at;
    if (event.key === 'Home') next = 0; else if (event.key === 'End') next = items.length - 1; else if (event.key === 'ArrowDown') next = Math.min(items.length - 1, at + 1); else next = at < 0 ? items.length - 1 : Math.max(0, at - 1);
    items[next].focus({preventScroll:true}); items[next].scrollIntoView({block:'nearest'}); event.preventDefault(); return true;
  }
  function select(id, value) {
    const trigger = document.getElementById(id), pop = document.getElementById(id + 'Popup'); if (!trigger || !pop) return;
    const options = [...pop.querySelectorAll('.picker-option')], chosen = options.find((option) => option.dataset.value === value);
    for (const option of options) option.setAttribute('aria-selected', String(option === chosen));
    const held = PICKER_ROWS.get(pop.id); if (held) held.selected = value;
    const title = chosen?.dataset.pickerTitle ?? held?.rows.find(row=>row.value===value)?.title;
    const label = trigger.querySelector('.picker-trigger-label'); // a `switch` face shows the chevron alone
    if (title !== undefined && label) label.textContent = title;
  }
  return {toggle, close, filter, keydown, select, choose, value, open};
})();
/* The rail is navigation only (round 17): a page's sub-tabs where it has them, nothing where it
 * has none — the title line names the page. In the product, an unprepared or preparing workspace
 * names its next step under the rail of every page (the scene itself excepted); a deep link is
 * never redirected. */
/* A grouped-list row inside a card (Apple's inset lists, Mercury's list cards): the whole row is
 * the link; it reads an icon, the thing to do or look at, why, and a chevron. `to` is a page
 * route ({page, extra}) or an action ({action, value}). */
/* One empty state (N6, law 130; the user: 这种情况统一用我们的logo): in the list's place the
 * product's own mark, one line of what would be here (what it means is the line's (i)), and the one
 * way to make it -- the same on every page. */
/* The one empty state (law 130; the user's reading 2026-09-23: 空页面好几种款式, 统一成几种 scenarios):
 * `create` a list the reader fills here (the mark, the line, the primary way); `elsewhere` a list
 * another owner fills (the mark, the line and its (i), at most a quiet way); `nomatch` a search or
 * a filter found nothing (the line and a quiet way to clear it, no mark). It draws no frame; its
 * line has no full stop. */
function emptyState(line, way = '', cls = '', kind = '') {
  const scenario = kind || (way ? 'create' : 'elsewhere');
  const said = typeof line === 'string' ? line.replace(/[.。]\s*$/, '') : line;
  // the mark is a page's own empty -- the whole page holds nothing; a section's or a box's empty is its line alone (the user, 2026-09-24: 这里你弄的logo到处都是太蠢了)
  const marked = scenario !== 'nomatch' && /(^|\s)page-empty(\s|$)/.test(cls);
  return html`<div class="section-empty${cls ? ' ' + cls : ''}" data-empty="${scenario}" role="status">${!marked ? '' : html`<span class="empty-mark" aria-hidden="true">${typeof mark === 'function' ? mark() : ''}</span>`}<p>${said}</p>${way}</div>`; // a reader without the product's glyphs (a harness) draws the line alone
}
/* A refusal or a failure, named once (round 75): the state's dot and word, the cause in the
 * owner's words (a code the table knows reads as words, the raw code stays on hover; an unknown
 * one is shown as one), the next lawful action as a line -- and the control beside it. `body` is
 * the owner's typed refusal ({failure_code | refused | disposition | code, reason | detail |
 * message | explanation, next_action | next}); `o.state` names the state (`refused` unless the
 * tone is danger: `failed`), `o.word` the word, `o.action` the control. */
/* U71 (V444): what a stopped stage's owner saw beside its code -- the raised type, its step, the unit in hand and its
 * sessions, its message -- so a person can tell an exhausted machine (a `MemoryError`, a paging file too small) from a
 * defect. As the owner kept it; nothing is inferred. `inline` for a sentence's paragraph, else its own caption line. */
function causeLine(cause, inline = false) {
  if (!cause?.exception_type) return '';
  const sessions = cause.first_session ? (cause.last_session && cause.last_session !== cause.first_session ? `${cause.first_session} — ${cause.last_session}` : cause.first_session) : '';
  const where = [cause.step ? codeWords(cause.step) : '', cause.unit || '', sessions].filter(Boolean).join(' · ');
  const words = html`${t('Raised')} <span class="mono">${cause.exception_type}</span>${where ? html` · ${where}` : ''}${cause.detail ? html` · <span class="owner-text">${cause.detail}</span>` : ''}`;
  return inline ? html`<br><span class="cause-line">${words}</span>` : html`<p class="caption cause-line">${words}</p>`;
}
function refusal(body, tone = 'warning', o = {}) {
  // `o.attrs` names the outcome for the sheet, `o.more` carries an owner's field list or a caption
  const b = body || {}, state = o.state || (tone === TONE.failure ? 'failed' : 'refused');
  const code = b.failure_code || b.refused || b.disposition || b.code || '';
  const said = b.reason || b.detail || b.message || b.explanation || '';
  const why = literalWords(o.catalog ? t(said || CODE_LINES[code] || '') : said);
  const next = b.next_action || b.next || '';
  const nextLine = next && ROUTES[String(next).toLowerCase()] ? link(html`${t(ROUTES[String(next).toLowerCase()][1])}${icon('arrow')}`, String(next).toLowerCase(), 'text-btn') : '';
  const known = code && (CODE_WORDS[code] || STATES[String(code).toLowerCase()]);
  // WD2: an owner's code is its sentence, the code on hover; a code without one reads as its words (the user,
  // 2026-09-26: `TASK_CANCELLED_BEFORE_START` beside the stop's own sentence)
  const cause = why ? html`${code && !known ? html`<span data-tip="${code}">${why}</span>` : why}${known ? html` · ${coded(code)}` : ''}` : code ? (known ? coded(code) : html`<span class="coded" data-tip="${code}">${codeWords(code)}</span>`) : '';
  const after = o.next !== undefined ? o.next : nextLine; // A state alone cannot name a route; the caller supplies its next press or command (ST6).
  return html`<div class="banner ${tone} refusal${o.cls ? ' ' + o.cls : ''}"${tone === 'neutral' ? '' : ' data-box="decision"'} role="alert"${o.attrs ? html` ${o.attrs}` : ''}><div class="grow">${stateLine(state, {word: o.word, next: ''})}${why || code ? html`<p>${cause}</p>` : ''}${after ? html`<p class="refusal-next">${t('Next step')}: ${after}</p>` : ''}${o.more || ''}</div>${o.action || ''}</div>`;
}
/* ---- run shapes (round 72): the steps, the log and its scroll owner; the harnesses load this block by its markers ---- */
/* A run's steps (round 72): one owner for a step list. `items`: [{id, name, mark, at, evidence,
 * note, under, current, shown, done}]. As the list a run's body reads: the mark's dot, the name,
 * the mark's word with the evidence count and the step's note, the instant at the right, the
 * current step's line under it. As the rail a work area inspects (`o.rail`): the round-44 nodes,
 * buttons that carry `o.action`; the mark is `data-state`. */
/* A reference (round 92): its kind worded, its short hash in mono, the whole on hover, a copy glyph
 * that copies the exact URI. A value never rolls sideways or breaks inside itself: what its column
 * cannot hold is its short form (WD3; the user, 2026-09-26: a scrollbar under every hash of a Task's
 * receipt). A hash the same way without the kind (`hashCell`), a locator by its ends (`locatorCell`). */
function refCell(uri) {
  // `scheme://host/<hash>` or `scheme://host/<kind>/<hash>` (an artifact's semantic locator:
  // `semantic://chief-risk-officer/cro_review_publication/<hash>`), with the query that narrows it
  // (`?spans=`, `?snapshot=`): the kind is the segment before the hash, else the host; a locator of
  // another shape is its ends
  const s = String(uri);
  const m = s.match(/^([a-z][a-z0-9+.-]*):\/\/([^/?#]+)((?:\/[^/?#]+)*?)\/([0-9a-f]{8,})(?:\.[a-z0-9]+)?\/?(?:\?[^#]*)?$/i); // an extension after the hash (`.json`) is the store's, not the identity's
  const segment = m ? (m[3] ? m[3].slice(1).split('/').pop() : m[2]) : '';
  const kind = m ? codeWords(segment.replace(/-/g, '_')) : t('Reference'), id = m ? short(m[4], SHORT.id) : ''; // never a local named `short`: it would shadow the library's own (2026-09-22: a Task's step list threw)
  return html`<span class="run-ref"><span class="run-ref-kind">${kind}</span>${m ? html`<span class="mono" data-tip="${s}">${id}</span>` : locatorWords(s)}${btnAttrs(icon('copy'), 'copy-text', s, 'icon-btn compact', html`aria-label="${t('Copy the reference')}" data-tip="${t('Copy the reference')}"`)}</span>`;
}
/* A hash as one cell: its leading `n` characters, the whole on hover, the copy glyph beside it. */
function hashCell(h, n = SHORT.hash) {
  if (!h) return '';
  const s = String(h);
  return html`<span class="run-ref"><span class="mono" data-tip="${s}">${s.length > n ? s.slice(0, n) + '…' : s}</span>${btnAttrs(icon('copy'), 'copy-text', s, 'icon-btn compact', html`aria-label="${t('Copy')}" data-tip="${t('Copy')}"`)}</span>`;
}
/* A locator or a path by its ends -- its scheme and host, then its last segment, the middle elided --
 * cut at its column's edge if even that is too long, whole on hover; `locatorCell` adds the copy glyph. */
const locatorWords = (uri) => {
  const s = String(uri), m = s.match(/^([a-z][a-z0-9+.-]*:\/\/[^/?#]+)((?:\/[^/?#]*)*)/i), parts = m ? m[2].split('/').filter(Boolean) : [];
  return html`<span class="locator line-cut" data-tip="${s}">${parts.length > 1 ? `${m[1]}/…/${parts.pop()}` : s}</span>`;
};
const locatorCell = (uri) => (uri ? html`<span class="run-ref">${locatorWords(uri)}${btnAttrs(icon('copy'), 'copy-text', String(uri), 'icon-btn compact', html`aria-label="${t('Copy')}" data-tip="${t('Copy')}"`)}</span>` : '');
/* An owner's named code -- a policy, a recipe, a rule set -- as one cell (WD2): its words, the exact
 * code on hover, the copy glyph beside it. A name is not an identity: it wraps as words and never
 * rolls (WD3; the user, 2026-09-25: a scrollbar under the review's policy was strange). */
function codeCell(code) {
  if (!code) return '';
  const s = String(code);
  return html`<span class="run-ref"><span data-tip="${s}">${codeWords(s)}</span>${btnAttrs(icon('copy'), 'copy-text', s, 'icon-btn compact', html`aria-label="${t('Copy')}" data-tip="${t('Copy')}"`)}</span>`;
}
function stepList(items, o = {}) {
  if (o.rail) return html`<nav class="fv-pipeline ${o.cls || ''}" aria-label="${o.label || ''}" style="--fv-stage-count:${items.length}">${items.map((s, i) => html`<button type="button" class="fv-step ${o.stepCls || ''}${s.current ? ' is-current' : ''}${s.shown && !s.current ? ' is-inspected' : ''}${s.done ? ' is-complete' : ''}${s.current && o.moving ? ' is-working' : ''}" data-action="${o.action}" data-value="${s.id}" data-state="${s.mark}" data-tone="${stateOf(s.mark).tone}"${s.current ? ' aria-current="step"' : ''} aria-pressed="${Boolean(s.shown)}" aria-label="${s.name} · ${s.words || codeWords(s.mark)}${s.current ? ' · ' + t('current stage') : ''}"><span class="fv-step-node">${s.done ? icon('check') : ['BLOCKED', 'CANCELLED', 'UNAVAILABLE', 'REFUSED'].includes(s.mark) ? icon('ban') : String(i + 1).padStart(2, '0')}</span><span class="fv-step-copy"><strong>${s.name}</strong><small>${s.words || codeWords(s.mark)}${s.evidence ? html` · ${t('{n} evidence', {n: s.evidence})}` : ''}</small></span></button>`)}</nav>`;
  // round 92 (GitHub Actions' shape): each step one row -- the number, the state's dot, the name,
  // the state's word with the evidence count, the instant at the right -- folded, and opened by its
  // chevron to its note, its evidence references (a kind, a short hash, a copy glyph: never a URI
  // cut) and its own lines of the log; the current step and a stopped one open by themselves.
  const ref = refCell;
  return html`<ol class="run-steps">${items.map((s, i) => {
    const open = s.open !== undefined ? s.open : s.current || ['BLOCKED', 'CANCELLED', 'FAILED'].includes(String(s.mark).toUpperCase());
    const refs = Array.isArray(s.refs) ? s.refs : [];
    const has = s.note || s.line || refs.length || (s.lines && s.lines.length) || s.under;
    return html`<li><details class="run-step${s.current ? ' is-current' : ''}" data-mark="${s.mark}"${open && has ? ' open' : ''}${has ? '' : ' data-empty="true"'}><summary class="run-step-head">${icon('chevron')}<span class="run-step-num">${String(i + 1).padStart(2, '0')}</span>${statusDot(s.mark)}<strong>${s.name}</strong><span class="run-step-word">${codeWords(s.mark)}${s.evidence ? html` · ${countText(s.evidence, '{n} evidence reference', '{n} evidence references')}` : ''}</span>${s.at ? html`<time class="run-step-at">${s.at}</time>` : ''}</summary>${has ? html`<div class="run-step-body">${s.line ? html`<p class="run-step-line">${s.line}</p>` : ''}${s.note ? html`<p class="run-step-note">${s.note}</p>` : ''}${refs.length ? html`<div class="run-step-refs">${refs.map(ref)}</div>` : ''}${s.lines && s.lines.length ? html`<div class="run-log-lines run-step-log" role="log" aria-label="${s.name}">${s.lines}</div>` : ''}${s.under || ''}</div>` : ''}</details></li>`;
  })}</ol>`;
}
/* One log (round 72): the shell every run reads its record in -- a head (the title, its line, the
 * controls, the filter `/` focuses), the viewport (`id` names it for the scroll owner), a status
 * line under it; `fold` makes the whole log a disclosure, open. `lines` are `logLine`s or a
 * scene's own rows. */
function runLog({id = 'runLog', title, caption = '', lines = [], empty = '', status = '', controls = '', search = false, rail = '', cls = '', linesCls = '', notes = '', attrs = '', fold = false, count = null, summary = '', open = true}) {
  const field = search ? html`<input type="search"  class="search-input log-search" data-filter=".log-line" data-filter-scope=".run-log" placeholder="${t('Filter the log')}" aria-label="${t('Filter the log')}">` : '';
  const titles = html`<div class="panel-label"><h3>${title}${count !== null ? html` <span>${count}</span>` : ''}</h3>${infoMark(caption)}</div>`; // N6 (law 121): the log's sentence is its label's (i)
  // round 83: a fold with a `summary` reads as Codex's one grey line (`Worked for 3 min · 4 ›`) until opened
  const folded = summary ? html`<summary class="run-log-head run-log-fold"><span class="run-log-summary">${summary}${count !== null ? html` · ${count}` : ''}</span></summary>` : html`<summary class="run-log-head">${titles}</summary>`;
  const head = fold ? html`${folded}${field || controls ? html`<div class="run-log-tools">${field}${controls}</div>` : ''}` : html`<header class="run-log-head">${titles}${field}${controls}</header>`;
  const body = html`${rail}${head}${notes}<div class="run-log-lines${linesCls ? ' ' + linesCls : ''}" id="${id}" role="log" aria-live="off" tabindex="0" aria-label="${title}">${lines.length ? lines : empty}</div>${status ? html`<div class="run-log-status">${status}</div>` : ''}`;
  return fold ? html`<details class="run-log reveal-details ${cls}"${open ? ' open' : ''}${attrs ? ' ' + attrs : ''}>${body}</details>` : html`<section class="run-log ${cls}"${attrs ? html` ${attrs}` : ''}>${body}</section>`;
}
/* One observation as a log line: who recorded it, the words, the telling reference -- an
 * operation, a Task Control transition, an artifact verification; `declared(payload)` words the
 * other kinds (the Team's declared events). */
const OBSERVED_WORDS = {
  ProductOperationObserved: (p) => [actorWords(p.caller, p.producer_id), html`${codeWords(p.operation)} · ${codeWords(p.phase)}${p.status ? html` · ${codeWords(p.status)}` : ''}${p.task_lifecycle ? html` · ${codeWords(p.task_lifecycle)}` : ''}`, p.operation_ref, 'activity'],
  TaskControlTransition: (p) => [t('Task Control'), html`${codeWords(p.task_lifecycle)}${p.stage_id ? html` · ${codeWords(p.stage_id)}` : ''}${p.total_units ? html` · ${p.verified_prefix_count} / ${p.total_units} ${t('verified')}` : ''}${p.disposition ? html` · ${codeWords(p.disposition)}` : ''}`, p.last_verified_result_ref || p.projection_hash, 'task'],
  ArtifactVerificationObserved: (p) => [t('artifact owner'), html`${t('Artifact verified')} · ${codeWords(p.artifact_kind)}${p.availability ? html` · ${codeWords(p.availability)}` : ''}`, p.artifact_hash, 'checkcircle'],
};
function observationLine(v, declared) {
  const p = v.payload || {};
  const [by, words, code, ic = 'team'] = (OBSERVED_WORDS[v.schema_kind] || declared)(p);
  return {at: when(v.occurred_at || v.observed_at), by, words: v.availability && v.availability !== 'AVAILABLE' ? html`${words} · ${codeWords(v.availability)}` : words, code: code || '', stage: v.stage_id || '', key: v.observation_id, ic};
}
/* A line of the log: a grey mark for the kind of thing that happened (Linear's activity, round 85), the instant, who, the words; a reference (a hash, a path) as code. */
function logLine({at = '', by = '', words = '', code = '', stage = '', key = '', cls = '', ic = 'activity'}) {
  return html`<div class="log-line${cls ? ' ' + cls : ''}"${stage ? html` data-stage="${stage}"` : ''}${key ? html` data-key="${key}"` : ''}>${tile(ic, 'neutral')}<time>${at}</time>${by ? html`<span class="log-by">${by}</span>` : ''}<span class="log-words">${words}${code ? html` <code class="log-code line-cut" data-tip="${code}">${code}</code>` : ''}</span></div>`;
}
/* The log's scroll owner (round 72, from the work area): a log follows its end until the reader
 * moves away, and resumes when they come back. `area` holds `follow` and `logTop`; `moved` is
 * told when the following changed (the work area saves it and rewords its control). */
function logMove(el, top, area) { el.scrollTop = top; area.logTop = el.scrollTop; }
function logHeld(el, area, moved) {
  // a log whose box changed (the window resized, the lane reflowed) was moved by its layout, not by the reader:
  // a following log keeps its end (the walk, 2026-09-26: a resize to 375 turned "Hold reading" into "Resume following")
  const box = `${el.clientWidth}x${el.clientHeight}`, reflowed = area.logBox !== undefined && area.logBox !== box;
  area.logBox = box;
  if (reflowed) { if (area.follow) logMove(el, el.scrollHeight, area); else area.logTop = el.scrollTop; return; }
  const restored = area.logTop !== null && el.scrollTop === Math.min(area.logTop, Math.max(0, el.scrollHeight - el.clientHeight));
  area.logTop = el.scrollTop;
  if (restored) return;
  const atEnd = el.scrollTop + el.clientHeight >= el.scrollHeight - 8;
  if (area.follow === atEnd) return;
  area.follow = atEnd; if (moved) moved();
}
/* The bounded ring a log keeps: the newest `n` rows stay. */
function logRetain(log, n) { while (log.rows.size > n) log.rows.delete(log.rows.keys().next().value); log.primed = true; }
/* A run's row (round 73): the state's dot, the name, the state's word with the duration, who
 * started it, its facts, the instant at the right -- one shape for a Task, a data update, a
 * session; `o.pinned` (the running one at the top of a list) adds the current step's line. */
/* The clock of an instant (HH:MM, 24-hour, in the reader's zone as `when` writes it; `seconds`
 * adds them, for a log), for a row under its day's head. */
function clockOf(iso, seconds = false) { // self-contained: the harnesses load this block without the reading helpers
  const s = String(iso || ''), m = s.match(/^\d{4}-\d{2}-\d{2}[T ](\d{2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?(Z|[+-]\d{2}:?\d{2})?$/);
  if (!m) return when(iso);
  const d = new Date(s.replace(' ', 'T') + (m[4] ? '' : 'Z')), two = (n) => String(n).padStart(2, '0'); // a zoneless instant is UTC (the owners' convention)
  const [hh, mm, ss] = !Number.isNaN(d.getTime()) ? [d.getHours(), d.getMinutes(), d.getSeconds()].map(two) : [m[1], m[2], m[3] || ''];
  return `${hh}:${mm}${seconds && ss ? ':' + ss : ''}`;
}
/* `o.inDay`: the row stands under its day's group head, which says the day -- its time is the clock
 * alone (W, 375 at 125 %: the day said twice left the title a letter). `o.dot`: the group says the
 * state (a lobby by state), the dot alone leads; `o.line: false`: one line, no way on under it. */
function runRow(run, o = {}) {
  const at = run.finished || run.started;
  // N6 (law 58): a held data update's way on is the Data page's; any other stop its code's or its state's
  const stop = typeof LiveWorkspace !== 'undefined' && LiveWorkspace.stopWords ? LiveWorkspace.stopWords(run.object) : null;
  const next = stop ? stop.next : wayOn(run.object?.latest_failure_code, run.state);
  const stage = o.pinned && run.current ? html`${t('stage')} ${typeof STAGES !== 'undefined' && STAGES[run.current] ? t(STAGES[run.current].word) : codeWords(run.current)}` : '';
  // LS2/ST6: descriptive words may fold; the owner's way on stays whole in its own fact disclosure.
  const after = o.line !== false && !o.why && !stage && next ? factsRef(t('Next step'), html`<p>${next}</p>`) : '';
  return objectRow({...(o.dot ? {lead: statusDot(run.state)} : {state: run}), name: run.markup || run.name, why: o.line === false ? '' : o.why || stage, to: o.to || {action: 'task', value: run.id}, cls: 'run-row' + (o.pinned ? ' is-pinned' : '') + (o.cls ? ' ' + o.cls : '')}, {key: run.id, selected: o.selected, until: o.until, word: o.word, columns: [...(o.dot ? ['state'] : []), 'starter', ...(o.columns || []), 'next'], props: [...(o.dot ? [''] : []), run.starter ? html`<span class="run-by">${run.starter}</span>` : '', ...(o.props || []), after], time: at ? (o.inDay ? clockOf(at) : when(at)) : '', actions: o.actions || '', attrs: o.attrs || ''});
}
/* ---- end of run shapes ---- */
/* ---- evidence objects (round 76): the object model written (the laws, "The evidence objects"),
 * one reading per type from the owners' own fields -- the kind, the subject, the state as a
 * STATES key with the owner's word, the one-line why, the facts. `evidenceRow` is the row every
 * evidence page renders; `citePill` the one shape of a handle, with the hover card (inspect.js).
 * Nothing here invents a field; a field the owner does not give is not shown. The harnesses
 * load this block by its markers. ---- */
const EVIDENCE_STATES = { // the owner's word (CODE_WORDS or the code) unless a second word is given
  SUPPORTED: ['verified'], SINGLE_SOURCE: ['partial'], CONTESTED: ['blocked'], UNSUPPORTED: ['blocked'],
  HIGH: ['blocked', 'High if true'], MODERATE: ['partial', 'Moderate if true'], LOW: ['metadata', 'Low if true'],
  NO_FINDING_IN_SCOPE: ['verified'], NO_ADVERSE_ISSUE: ['verified'], ISSUE_NOTED_WITHIN_LIMITS: ['partial'], CONCERNS_NOT_ADJUDICATED: ['metadata'], EVIDENCE_GAP: ['deferred'], NOT_EVALUATED: ['metadata'], HUMAN_REVIEW_REQUIRED: ['review_pending'], ISSUE_STANDS: ['blocked'],
  NO_MATERIAL_OBJECTION: ['verified'], ACCEPT_WITH_LIMITS: ['partial'], REQUEST_EVIDENCE_REFRESH: ['blocked'], MATERIAL_OBJECTION: ['blocked'], // a review's route (B1): its standing in its mark's colour
  ADVERSE: ['blocked'], MITIGATING: ['verified'], FAVOURABLE: ['verified'], MIXED: ['partial'], AMBIGUOUS: ['partial'], NONE: ['metadata'],
  VERIFIED_READBACK: ['verified'], VERIFIED_TASK_STATE: ['verified'], VERIFIED_REFUSAL: ['verified'], SAVED_NOT_VERIFIED: ['pending'], UNAVAILABLE: ['blocked'],
};
const evidenceState = (code, fallback = 'metadata') => { const e = EVIDENCE_STATES[code]; return {state: e ? e[0] : STATES[String(code ?? '').toLowerCase()] ? String(code).toLowerCase() : fallback, word: e && e[1] ? t(e[1]) : codeWords(code)}; };
const entityNames = (list) => (list || []).join(', ');
const weightWords = (w) => w == null || w === '' ? '' : typeof w === 'number' ? pctFraction(w) : pctText(w);
// the owner's signed change of the book: a string already in basis points, a fraction as bps to two decimals
const changeWords = (c) => c == null || c === '' ? '' : html`${typeof c === 'number' ? (c < 0 ? '−' : '+') + Math.abs(c * 10000).toFixed(2) : String(c).replace(/^-(0(?:\.0+)?)$/, '$1')} ${unit('bps')}`;
// COMPLETE ends the owner read, including empty and unavailable references; it is no scientific verdict.
function goalReferenceIntegrity(body) {
  const refs = body.references;
  if ((refs?.length ?? body.reference_count) === 0) return {state: 'pending', word: t('Nothing to check yet'), limit: ''};
  const limit = t('Reference integrity only; not scientific approval.');
  if (!refs || body.evidence_verification !== 'COMPLETE') return {state: 'pending', word: t('Reference integrity not checked'), limit};
  const checked = new Set(['VERIFIED_READBACK', 'VERIFIED_TASK_STATE', 'VERIFIED_REFUSAL']);
  const complete = refs.every((ref) => checked.has(ref.state));
  return {state: complete ? 'verified' : 'partial', word: t(complete ? 'Reference integrity checked' : 'Reference integrity not verified'), limit};
}
const EVIDENCE = {
  version: (x) => ({kind: 'Evidence version', subject: html`${t('as of')} ${dayOf(x.evidence_as_of)}`, why: html`${countText(x.issuer_count, '{n} issuer', '{n} issuers')} · ${t('expires')} ${dayOf(x.evidence_expires_at)}`, state: x.is_selected ? 'ready' : 'pending', word: t(x.is_selected ? 'Selected' : 'Eligible'), id: x.analysis_publication_hash}),
  issuer: (x) => ({kind: 'Issuer', subject: entityNames(x.tickers) || x.entity_id, why: x.findings_summary || (x.selection_reason ? codeWords(x.selection_reason) : ''), ...evidenceState(x.conclusion || x.exposure_band), columns: ['weight', 'change', 'band'], props: [weightWords(x.ending_weight), changeWords(x.signed_change), x.conclusion && x.exposure_band ? codeWords(x.exposure_band) : ''], id: x.entity_id}),
  filing: (x) => ({kind: 'Filing', subject: html`${x.document_type ? String(x.document_type) : x.document_handle}${x.revision || x.revision_label ? html` · ${x.revision || x.revision_label}` : ''}`, why: html`${x.entity_id || ''}${x.available_at ? html` · ${t('available')} ${when(x.available_at)}` : ''}`, ...(x.status ? evidenceState(x.status, 'recorded') : {state: 'recorded', word: t('Candidate')}), id: x.document_handle}),
  section: (x) => ({kind: 'Section', subject: x.title || '', why: html`${x.document_handle || ''}${x.start_line ? html` · ${t('lines {from}–{to}', {from: x.start_line, to: x.end_line})}` : ''}`, state: 'recorded', word: t('Recorded'), id: x.document_handle}),
  finding: (x) => ({kind: 'Finding', subject: x.topic ? codeWords(x.topic) : x.finding_handle, why: html`${entityNames(x.affected_entities)}${x.direction ? html` · ${codeWords(x.direction)}` : ''}${x.summary ? html` · ${x.summary}` : ''}`, ...evidenceState(x.structure || x.direction), columns: ['citations', 'severity', 'handle'], props: [x.supporting_document_count != null ? t('{a} supporting · {b} contradicting', {a: countText(x.supporting_document_count, '{n} document', '{n} documents'), b: count(x.contradicting_document_count)}) : x.cites != null ? countText(x.cites, '{n} citation', '{n} citations') : '', x.severity ? evidenceState(x.severity).word : '', x.topic ? ['', html`<span class="mono">${x.finding_handle}</span>`, 'drop'] : ''], id: x.finding_handle}), // item 6 (law 96): the topic leads, the handle follows small -- the first fact a narrow list drops
  citation: (x) => ({kind: 'Citation', subject: x.entity_id ? html`${x.entity_id}${x.document_type ? html` · ${String(x.document_type)}` : x.title ? html` · ${x.title}` : ''}` : html`<span class="mono">${x.span_handle}</span>`, why: x.entity_id && x.title && x.document_type && String(x.title).replace(/\s+/g, ' ').trim().toLowerCase() !== `${x.entity_id} ${String(x.document_type)}`.toLowerCase() ? x.title : '', state: x.excerpt ? 'verified' : 'recorded', word: t(x.excerpt ? 'Verified' : 'Recorded'), columns: ['span', 'document', 'available'], props: [x.entity_id ? html`<span class="mono">${x.span_handle}</span>` : '', x.document_handle ? html`<span class="mono">${x.document_handle}</span>` : '', x.available_at ? html`${t('available')} ${when(x.available_at)}` : ''], id: x.span_handle}),
  // item 6 (law 96): an issue leads with its issuers and the position's impact over the CRO's inference; its handle follows small
  issue: (x) => ({kind: 'review|Issue', subject: html`${entityNames(x.affected_entities) || ''}${x.position_impact_direction ? html` · ${codeWords(x.position_impact_direction)}` : ''}`, why: x.cro_inference || (x.rule_id ? (/^[A-Z][A-Z_]+$/.test(String(x.rule_id)) ? codeWords(x.rule_id) : x.rule_id) : ''), ...evidenceState(x.severity_if_true), columns: ['citations', 'handle'], props: [x.cited_finding_handles ? countText(x.cited_finding_handles.length, '{n} cited finding', '{n} cited findings') : '', ['', html`<span class="mono">${x.issue_handle}</span>`, 'drop']], id: x.issue_handle}),
  review: (x) => ({kind: 'Review', subject: x.review_publication_hash ? html`<span class="mono">${short(x.review_publication_hash, SHORT.hash)}</span>` : codeWords(x.state), why: x.disposition ? html`${codeWords(x.disposition)} · ${codeWords(x.review_state)}` : x.explanation || '', ...evidenceState(x.state), columns: ['asof', 'attribution'], props: [x.evidence_as_of ? html`${t('evidence as of')} ${dayOf(x.evidence_as_of)}` : '', attributionWords(x.review_attribution)], id: x.review_publication_hash}),
  case: (x) => ({kind: 'Research case', subject: x.document?.title || x.title || '', why: x.document?.question || x.question || '', ...goalReferenceIntegrity(x), columns: ['integrity', 'revision', 'purpose'], props: [goalReferenceIntegrity(x).limit, x.revision != null ? t('revision {n}', {n: x.revision}) : '', x.purpose || x.document?.purpose ? codeWords(x.purpose || x.document.purpose) : ''], id: x.case_hash}),
  reference: (x) => ({kind: 'Reference', subject: x.reference?.label || x.label || '', why: html`${codeWords((x.reference || x).stage)} · ${codeWords((x.reference || x).request?.operation)}${(x.reference || x).intent_relation ? html` · ${codeWords((x.reference || x).intent_relation)}` : ''}`, ...(x.failure_code ? {state: 'blocked', word: codeWords(x.state || 'UNAVAILABLE')} : evidenceState(x.state, 'recorded')), columns: ['failure'], props: [x.failure_code ? [t('Failure'), html`<span class="mono">${x.failure_code}</span>`, 'drop'] : ''], id: (x.reference || x).reference_id}),
  handoff: (x) => ({kind: x.dossier ? 'CRO dossier' : 'Analyst packet', subject: html`<span class="mono">${short(x.dossier?.dossier_hash || x.packet_hash || '', SHORT.hash)}</span>`, why: x.claim || '', state: x.status ? 'ready' : 'metadata', word: codeWords(x.status), columns: ['asof'], props: [x.dossier?.evidence_as_of || x.source_expires_at ? html`${t(x.dossier ? 'evidence as of' : 'source expires')} ${dayOf(x.dossier?.evidence_as_of || x.source_expires_at)}` : ''], id: x.dossier?.dossier_hash || x.packet_hash}),
  report: (x) => ({kind: 'Report', subject: html`<span class="mono">${short(x.report_hash || x.export_hash || '', SHORT.hash)}</span>`, why: html`${codeWords(x.review_status)}${x.review?.recommendation?.route ? html` · ${codeWords(x.review.recommendation.route)}` : ''}`, state: x.historical ? 'historical' : 'succeeded', word: t(x.historical ? 'Historical' : 'Published'), columns: ['verified'], props: [x.evidence?.verified_spans ? countText(x.evidence.verified_spans.length, '{n} verified span', '{n} verified spans') : ''], id: x.report_hash}),
};
/* The reading of one evidence object: `type` a key of `EVIDENCE` and `x` the owner's projection,
 * or a reading already made (the Team's observations read themselves). */
function evidenceName(type, x) {
  const r = typeof type === 'object' ? type : EVIDENCE[type] ? EVIDENCE[type](x || {}) : {kind: 'Evidence', subject: '', state: 'metadata', word: ''};
  const kind = t(r.kind || '');
  return {...r, kind, name: r.subject ? html`${kind} · ${r.subject}` : kind, type: typeof type === 'object' ? r.type || '' : type};
}
/* The row (round 76): the state's dot and word, the subject (with its kind when the list mixes
 * types or the row heads a reading: `o.named`), the why, the facts (`o.brief`: the state alone,
 * for a narrow list), the time; `o.to` the way on. */
function evidenceRow(type, x, o = {}) {
  const r = evidenceName(type, x);
  return objectRow({state: r.state, name: o.named ? r.name : r.subject || r.name, why: o.why ?? r.why, to: o.to || null, cls: 'evidence-row' + (o.cls ? ' ' + o.cls : '')}, {key: o.key || r.id, selected: o.selected, word: r.word, columns: [...(o.brief ? [] : r.columns || []), ...(o.columns || [])], props: [...(o.brief ? [] : r.props || []), ...(o.props || [])], time: o.time || '', actions: o.actions || '', attrs: html`data-kind="${r.type}"${o.attrs ? html` ${o.attrs}` : ''}`});
}
/* A handle as one pill: the citation's or the finding's; it opens the object where the page
 * lists it (`opens`), and is a mark inside a row or a button. The hover card reads the span. */
function citePill(handle, opens = true) {
  const pill = html`<span class="mono">${handle}</span>`;
  return opens ? btnAttrs(pill, 'review-live-item', handle, 'es-cite', html`data-cite="${handle}" aria-label="${t('Open citation {h}', {h: handle})}"`) : html`<span class="es-cite" data-cite="${handle}" tabindex="0">${pill}</span>`;
}
/* The words a page keeps behind the fold (round 76): each term once, one line, read in the
 * inspector's Facts as the Glossary; the pages name at most twelve objects above the fold. */
const GLOSSARY = [
  ['Evidence version', 'The sealed corpus one review reads: its issuers, documents and policy hashes, as of one instant, expiring at another.'],
  ['Issuer', 'A company the book holds, named by its tickers; the review concludes per issuer.'],
  ['Filing', 'One official document of an issuer at one revision, available from one date.'],
  ['Section', 'The titled part of a filing a passage sits in.'],
  ['Finding', 'A typed disclosure the Analyst read in the filings: its issuers, direction and the spans that support or contradict it.'],
  ['Citation', 'The span handle that carries a passage to a claim; verified when the owner replays its excerpt.'],
  ['review|Issue', 'A material concern the CRO raised against cited findings: its severity if true, interpretation and position impact.'],
  ['Review', 'The CRO\u2019s published judgement for one book and evidence version: route, completeness, issuer conclusions, required actions.'],
  ['Packet', 'The exact prepared evidence handed to an Analyst: admitted documents and spans, bound by its context hash.'],
  ['Dossier', 'The sealed compilation of the published analysis and the book handed to the CRO.'],
  ['Bundle', 'A packet or a dossier as one downloadable document with its submission template.'],
  ['Selector', 'The exact book a review page reads: an experiment Task, an update Task, a result or a handoff hash.'],
  ['Publication', 'One sealed review readback pinned by its hash; preparing newer evidence never rewrites it.'],
  ['Revision', 'One saved state of a case or a filing; the head is the current one.'],
  ['Coverage', 'How much of the book the review reached: positions, issuers and weight, against the required minimum.'],
  ['Mapping', 'Which holdings the admitted issuer registry names; an unmapped position has no issuer.'],
  ['Exposure band', 'The owner\u2019s ordinal band of a position from its weight and change: low, medium, high, critical.'],
  ['Evidence structure', 'How a finding is held up by its own citations: supported, single source, contested, unsupported.'],
  ['Disposition', 'The route a review recommends: no material objection, accept with limits, refresh, human review, material objection.'],
  ['Matters, families, lanes', 'The owner\u2019s grouping of typed disclosures: a matter is one named subject, a family its disclosure kind, a lane one issuer \u00d7 family.'],
  // the reading workbench's words (round E3)
  ['Passage', 'One excerpt of one document at one character range, delivered under a span handle; a row of the reading.'],
  ['Matter', 'A named unit of a filing -- a financing, a proceeding, a corporate event -- read through its window.'],
  ['Cell', 'One issuer and one topic in the ledger: the state of its sealed reading plan, never that the topic was checked.'],
  ['Residual search', 'The bounded search the plan runs after the routed windows, dealt by cell in rounds.'],
  ['Window', 'One bounded read of a document region the plan sealed.'],
  ['Session', 'One bounded reading of the sealed plan; a continuation adds one, under the declared limits.'],
  ['Candidate', 'A passage the residual search returned but the plan has not read: pending, covered by delivered spans, or overlapping one.'],
  ['Continuation', 'The request the owner returns when more of the sealed plan can be read; its cost is the owner\'s, its limits the reader\'s.'],
  ['Group', 'A wide book is prepared in groups of at most eight issuers; each group is its own packet, check and analysis.'],
  ['Working eligibility', 'The current reading: the owner\'s projection of the book as it stands now, against the current evidence version.'],
  ['Pinned readback', 'A historical review publication read exactly as it was sealed; preparing newer evidence never rewrites it.'],
  ['Reading budget', 'The allowance one reading session holds: its reads, windows and bytes; more is read only through a declared continuation.'],
];
const glossarySection = () => ({title: t('Glossary'), body: kv(GLOSSARY.map(([term, line]) => [t(term), t(line)]), 'glossary')});
/* A share of a whole (round 77) is one meter (law 89): the words first (the counts, read by
 * the agent), the bar under them with a segment per count in its tone and, where the owner
 * states one, a mark for the required minimum. */
const coverageBar = (segments, o = {}) => meter({kind: 'share', segments, ...o});
/* ---- end of evidence objects ---- */
/* The one row of every list (round 13, one shape since round 65): a lead (a glyph name, or a
 * status dot), the name with its second line, its facts on the same line, the time at the right,
 * the row's actions shown on hover or focus, and the way on — `to` is a route ({page, extra}), an
 * action ({action, value}) or null for a row that only carries facts. The title is the link and stretches over the whole row; the
 * actions sit above the stretch, so a row with actions is a div, never a link around a button. A short
 * reference (`ref`, law 136: what tells two alike apart) leads the facts; in a slotted list it is the
 * column before the name. */
function objectRow({lead: ic = '', state = null, name = '', why = '', ref = '', to = null, cls = ''} = {}, slots = null) {
  // a run's row (round 71): the dot leads, the state's word and duration are its first fact
  const lead = state ? statusDot(typeof state === 'string' ? state : state.lifecycle ?? state.state ?? state.status) : typeof ic === 'string' ? (ic ? html`<span class="feature-icon">${icon(ic)}</span>` : '') : ic;
  // round 94: the row says what leads it (a glyph, a dot, a mark) so the facts line's indent needs no :has()
  const leadKind = state ? 'dot' : typeof ic === 'string' ? (ic ? 'icon' : '') : /class="mark\b/.test(String(ic)) ? 'mark' : /class="row-dot\b/.test(String(ic)) ? 'dot' : /class="feature-icon\b/.test(String(ic)) ? 'icon' : '';
  if (state) slots = {...(slots || {}), props: [stateLine(state, {dot: false, next: '', until: slots?.until, word: slots?.word}), ...(slots?.props || [])], columns: slots?.columns ? ['state', ...slots.columns] : null}; // the next action is the row's second line, not the slot's
  const copy = html`<strong>${name}</strong>${why ? html`<span class="list-row-why">${why}</span>` : ''}`;
  // a dense row's title keeps one line (law 136) and reads whole on hover or focus where the line cuts it -- on the row's main part, which takes the pointer (the clip census, 2026-09-24: the hidden part was readable nowhere)
  const main = !to ? html`<span class="list-row-main" data-tip="@overflow">${copy}</span>` : to.page ? html`<a class="list-row-main" data-tip="@overflow" href="${routeUrl(to.page, to.extra || {})}">${copy}</a>` : html`<button type="button" class="list-row-main" data-tip="@overflow" data-action="${to.action}" data-value="${to.value}">${copy}</button>`;
  // round 90: every fact keeps its slot (an absent one an empty span, hidden unless the list sizes its slots), so the columns align across rows
  const facts = slots?.columns || slots?.props?.filter(Boolean).length ? (slots?.props || []).map((p, i) => {
    const column = slots?.columns ? html` data-column="${slots.columns[i]}"` : '';
    return !p ? html`<span data-empty${column}></span>` : Array.isArray(p) ? html`<span${column}${p[2] === 'drop' ? ' data-drop' : ''}>${p[0] ? icon(p[0]) : ''}${p[1]}</span>` : html`<span${column}>${p}</span>`;
  }) : [];
  const props = facts.length || ref ? html`<span class="list-row-props"${slots?.columns ? html` data-columns="${slots.columns.join(' ')}"` : ''}>${facts}${ref ? html`<span class="list-row-ref mono" data-column="ref">${ref}</span>` : ''}</span>` : '';
  const time = slots?.time ? html`<span class="list-row-time" data-column="time">${slots.time}</span>` : '';
  const actions = slots?.actions ? html`<span class="row-actions">${rowMenu(slots.actions)}</span>` : '';
  return html`<div class="list-row ${cls}" data-row${slots?.columns ? html` data-row-columns="${slots.columns.join(' ')}"` : ''}${leadKind ? html` data-lead="${leadKind}"` : ''}${slots?.selected ? ' aria-current="true"' : ''}${slots?.key ? html` data-key="${slots.key}"` : ''}${slots?.attrs ? html` ${slots.attrs}` : ''}>${lead}${main}${props}${time}${actions}${to && !slots?.actions ? icon('chevron') : slots?.actions ? '' : html`<span class="icon chevron-slot" aria-hidden="true"></span>`}${slots?.under || ''}</div>`;
}
/* The choice list (round 95; GitHub's label picker, Linear's multi-select): a multiple choice
 * in place as a well -- the chosen as chips above it (removable by their ×, the rest as a
 * count), a head with the count and All · None and the filter field, then one row per option in
 * the picker's shape (the name, a ✓ on the chosen); ↑ ↓ Home End walk the rows, Space or Enter
 * toggles one. The list carries the field's data attributes; `choiceValues` reads the chosen
 * values from it, and every change reaches the field's handlers through `Events.changed`. */
function choiceList(name, options, attrs = '', count = '') {
  const breaks = (s) => String(s).split(/(?<=[_./-])/).map((p, i) => html`${i ? html`<wbr>` : ''}${p}`); // W: a factor's name breaks at its separators
  const rows = pickerItems(options.map((o) => ({value: o.value, title: o.mono ? html`<span class="mono">${breaks(o.label)}</span>` : o.label, word: o.label, search: o.label})), {action: 'choice-toggle', multiple: true, chosen: new Set(options.filter((o) => o.checked).map((o) => o.value)), selected: ''});
  return html`<div class="choice-list" id="${name}" data-choice-list${attrs ? ' ' + attrs : ''}><div class="choice-head"><span class="choice-count">${count}</span>${btnAttrs(t('All'), 'choice-all', name, 'text-btn')}${btnAttrs(t('None'), 'choice-none', name, 'text-btn')}<label class="choice-search"><span class="sr-only">${t('Filter the list')}</span>${icon('search')}<input type="search" class="ui-field" placeholder="${t('Filter…')}" data-filter=".picker-option" data-filter-scope=".choice-list" autocomplete="off"></label></div><div class="menu-list choice-rows" role="listbox" aria-multiselectable="true" aria-label="${t('Choices')}">${rows}<p class="menu-empty" hidden>${t('No matching choices')}</p></div></div>`;
}
const choiceValues = (el) => { const list = el.closest?.('.choice-list') || el; return [...list.querySelectorAll('.picker-option[aria-selected="true"]')].map((o) => o.dataset.value); };
const Choice = (() => {
  const optionsOf = (list, shownOnly = false) => [...list.querySelectorAll('.picker-option' + (shownOnly ? ':not([hidden])' : ''))];
  function announce(list) {
    if (typeof Events !== 'undefined' && Events.changed) Events.changed(list, choiceValues(list));
  }
  function press(list, b) {
    if (b.classList.contains('picker-option')) b.setAttribute('aria-selected', b.getAttribute('aria-selected') === 'true' ? 'false' : 'true');
    else if (b.dataset.action === 'choice-all' || b.dataset.action === 'choice-none') { const on = b.dataset.action === 'choice-all'; for (const o of optionsOf(list, true)) o.setAttribute('aria-selected', String(on)); }
    else return false;
    announce(list);
    return true;
  }
  function keydown(event) {
    const option = event.target.closest?.('.choice-list .picker-option');
    if (!option || !['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return false;
    const items = optionsOf(option.closest('.choice-list'), true), at = items.indexOf(option);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : Math.max(0, Math.min(items.length - 1, at + (event.key === 'ArrowDown' ? 1 : -1)));
    items[next]?.focus({preventScroll: true}); items[next]?.scrollIntoView({block: 'nearest'}); event.preventDefault();
    return true;
  }
  return {press, keydown, values: choiceValues};
})();
/* A sentence that explains a surface lives behind an (i): the fact line stays, the prose opens on
 * demand (round 9). The sentence is still in the DOM for readers and probes. */
/* Round 64: a term explained by the hover card (Space on the term reads it too), instead of an
 * (i) fold beside it; the words are the card's, the term stays a word on the page. */
function hint(term, words) {
  return words ? html`<span class="hint" data-tip="${words}" tabindex="0">${term}</span>` : term;
}
/* A note is one line (round 64): an icon and a sentence, the successor of the informational
 * banner; a refusal or a call for attention keeps `banner`. */
const NOTE_ICONS = {neutral: 'info', '': 'info', warning: 'warning', ok: 'checkcircle', danger: 'warning'};
function noteLine(title, text = '', tone = 'info', action = '', ic = '') {
  return html`<p class="note-line" role="status">${icon(ic || NOTE_ICONS[tone] || tone)}<span>${title}${text ? html` · ${text}` : ''}</span>${action}</p>`;
}
/* A fact section read in the inspector (round 64), the successor of the page's fold: its opener
 * is a text link, its body travels inert in a template beside it and is shown in the Facts mode. */
const textHash = (s) => { let h = 5381; for (let i = 0; i < s.length; i++) h = (h * 33 + s.charCodeAt(i)) >>> 0; return h.toString(36); };
function factsRef(title, body, cls = '') {
  // round 92: a document is never a Facts body; a page that tries throws into the render boundary
  if (/<pre[\s>]/.test(String(body))) throw new Error('factsRef: a document (a <pre>) is a code dialog (codeRef), never a Facts body: ' + String(title).replace(/<[^>]+>/g, '').slice(0, 60));
  const id = 'facts-' + textHash(String(title) + String(body));
  return html`<p class="facts-ref${cls ? ' ' + cls : ''}">${btnAttrs(html`${title}${icon('arrow')}`, 'facts-open', id, 'text-btn')}<template data-facts-id="${id}">${body}</template></p>`;
}
/* A document's opener (round 92): the same text link as a fact section's, but the body is a
 * document -- JSON, YAML, a list of references -- and it opens in the code dialog, never in the
 * side column. `text` is the exact text the dialog shows and copies (a value is stringified as
 * JSON); it travels inert in a template beside the link. */
function codeRef(title, text, lang = 'json', cls = '') {
  const body = typeof text === 'string' ? text : JSON.stringify(text, null, 2);
  const id = 'code-' + textHash(String(title) + body);
  return html`<p class="facts-ref${cls ? ' ' + cls : ''}">${btnAttrs(html`${title}${icon('arrow')}`, 'code-open', id, 'text-btn')}<template data-code-id="${id}" data-code-lang="${lang}" data-code-title="${String(title).replace(/<[^>]+>/g, '')}">${body}</template></p>`;
}
/* A record's facts under its title (round 49) as read-only property chips (round 56): the same
 * chip as the declaration's, pressed it says who recorded the value. The last chip opens the
 * object's facts in the inspector when the page declares any (round 64). */
/* The head's facts (E1; law 110's head): one line of words -- a label in the muted ink, its value in
 * the text ink, a middle dot between -- never a row of boxes; the Facts panel opens from the
 * line's (i). An unknown fact is not shown (round 93, rule 1). */
function contextFacts(cells, label = '') {
  const known = (v) => v != null && String(v).trim() !== '' && String(v) !== '\u2014';
  const shown = cells.map((cell) => (Array.isArray(cell) ? cell : [cell.label, cell.value])).filter(([, v]) => known(v));
  const sep = html`<span class="context-sep" aria-hidden="true">\u00b7</span>`;
  // N1: a break may follow each dot, so the line wraps between facts at a phone's width; a fact is one
  // box that carries its dot, whole on its line unless it is longer than the line (W: at 375 and 125 %
  // a study's input name ran past the window)
  return html`<span class="context-facts"${label ? html` aria-label="${label}"` : ''}>${shown.map(([k, v], i) => html`<span class="context-fact"><span class="context-key">${k}</span> <b>${v}</b>${i < shown.length - 1 ? sep : ''}</span>${i < shown.length - 1 ? html`<wbr>` : ''}`)}</span>`;
}
/* The page's head (PG6; laws 119-120).
 * The top row's path is the page's name, so a collection page draws no head of its own: its
 * actions, its tools, its explanation and its state go to the top row -- templates the frame reads
 * after every paint and patch: `page-menu` (the ··· beside the path), `page-actions` (its verbs,
 * quiet, at the row's right: law 129), `page-info` (the (i)) and `page-state`. An object's page
 * (`details.object`: the workspace's Home, a book's overview, a study, a case) opens with the
 * object's own name, its state and its facts; its actions go to the top row too. `details.top`
 * is the page's chooser (the book's Reading): the content's first row on every view, never the top
 * row's -- a chooser is not a verb (law 129). */
function objectHead(name, meta, actions = '', state = '', tools = [], details = {}) {
  // N4-N6 (laws 132, 135 amended): the place's tabs -- an object's folders, a list's views, Data's pages -- under the head
  if (details.views === undefined && typeof Window !== 'undefined' && Window.pageTabs) details = {...details, views: Window.pageTabs()};
  const m = /^<p class="lede">([\s\S]*?)<\/p>/.exec(String(meta));
  const lede = m ? raw(m[1]) : typeof meta === 'string' ? meta : '';
  const top = html`<template class="page-menu">${railTools(tools)}</template><template class="page-actions">${actions}</template>${lede ? html`<template class="page-info">${lede}</template>` : ''}${state && !details.object ? html`<template class="page-state">${state}</template>` : ''}`;
  const chosen = details.object ? details.top : html`${details.top || ''}${details.subject || ''}`;
  const controls = String(chosen || '').trim() ? html`<div class="page-controls" role="group" aria-label="${t('What this page shows')}">${chosen}</div>` : '';
  // N2 (laws 122-123): the chosen object this page reads (`details.scope`: {name, href, self}) opens
  // the path; the frame reads the template after the paint. `self`: the page is the object's own.
  const scope = details.scope ? html`<template class="page-scope" data-href="${details.scope.href || ''}" data-self="${details.scope.self ? 'true' : 'false'}">${details.scope.name}</template>` : '';
  if (!details.object) return html`${top}${scope}${details.views || ''}${controls}`; // law 129: the page's tabs, then its choosers
  // a lede that is the whole meta leaves nothing: an empty remainder is no box (B1: it lowered two book tabs' tabs by the context's gap)
  const after = m ? String(meta).slice(m[0].length).trim() : '';
  const rest = m ? (after ? raw(after) : '') : typeof meta === 'string' ? '' : meta;
  const identity=details.id ? html`<span class="object-id"><code>${details.id}</code>${btnAttrs(icon('copy'),'copy-text',details.id,'icon-btn code-copy',html`aria-label="${t('Copy object ID')}" data-tip="${t('Copy object ID')}"`)}</span>` : '';
  // round 53: a record opened from a list steps through that list — its position and the pair
  const step = typeof Inspect !== 'undefined' && Inspect.stepping ? Inspect.stepping() : null;
  const stepper = step ? html`<span class="object-step"><b class="num">${count(step.index + 1)} / ${count(step.total)}</b>${btnAttrs(icon('chevron'),'step-prev','','icon-btn compact step-up',html`aria-label="${t('Previous record in the list')}" data-tip="${t('Previous record in the list')}" data-tip-key="["${step.prev ? '' : html` aria-disabled="true"`}`)}${btnAttrs(icon('chevron'),'step-next','','icon-btn compact',html`aria-label="${t('Next record in the list')}" data-tip="${t('Next record in the list')}" data-tip-key="]"${step.next ? '' : html` aria-disabled="true"`}`)}</span>` : '';
  const ident = stepper ? html`<span class="object-ident">${stepper}</span>` : '';
  // E1: the facts are words on the context line (no boxes), then the id, the choosers, the (i)
  const facts = details.facts?.length ? contextFacts(details.facts, t('Object facts')) : '';
  const opener = details.facts?.length && typeof Inspect !== 'undefined' && Inspect.hasFacts() ? btnAttrs(icon('info'), 'facts', '', 'icon-btn compact context-opener', html`aria-label="${t('Facts')}" data-tip="${t('Facts')}"`) : '';
  const subject = details.subject ? html`<div class="object-subject" role="group" aria-label="${t('What this page shows')}">${details.subject}</div>` : '';
  const context = rest || identity || facts || subject ? html`<div class="object-context">${rest ? html`<div class="object-meta">${rest}</div>` : ''}${facts}${identity}${subject}${opener}</div>` : '';
  return html`${top}${scope}${controls}<header class="object-header${details.cls ? ' ' + details.cls : ''}"><div class="object-heading"><div class="title-line"><h1${details.headingId ? html` id="${details.headingId}" tabindex="-1"` : ''} data-tip="@overflow">${name}</h1>${details.switcher || ''}${state}${ident}</div></div></header>${context}${details.views || ''}`;
}
/* Attributes a builder is handed (`attrs`) are written between the tag's attributes, where the
 * html tag keeps a plain string as structure (a nested html` ${attrs}` would escape its quotes:
 * the status box read data-overview="&quot;status&quot;" until 2026-09-22). */
/* The status box (E1; GitHub's merge box and Vercel's deployment summary, measured): the one box
 * at the top of an object's Overview (law 66) -- the verdict as a sentence with its mark, its
 * caption and reasons; the checks, each a mark, a name, a line and a day (a row opens its
 * reading); the requirement that blocks, if any. Colour is the marks' alone. `tone`: good,
 * warning, danger, accent (moving), neutral (not started). */
const STATUS_MARKS = {good: 'checkcircle', warning: 'partial', danger: 'ban', accent: 'activity', neutral: 'clock', review: 'review'};
const statusMark = (tone) => html`<span class="status-mark ${tone}" aria-hidden="true">${icon(STATUS_MARKS[tone] || 'clock')}</span>`;
function statusBox({tone = 'neutral', title = '', caption = '', reasons = [], checks = [], foot = [], label = '', attrs = ''}) {
  const inner = (c) => html`${statusMark(c.tone)}<b class="status-name">${c.name}</b><span class="status-why" data-tip="@overflow">${c.why || ''}</span><time>${c.time || ''}</time>`;
  const row = (c) => html`<li>${c.action ? btnAttrs(inner(c), c.action, c.value, 'status-row', html`${c.selected ? 'aria-pressed="true"' : ''}${c.tip ? html` data-tip="${c.tip}"` : ''}`) : html`<div class="status-row"${c.tip ? html` data-tip="${c.tip}"` : ''}>${inner(c)}</div>`}</li>`;
  const need = (f) => html`<div class="status-foot">${statusMark(f.tone)}<p><b class="${f.tone}">${f.word}</b> ${f.why}</p>${f.action ? btnAttrs(f.label || t('Open'), f.action, f.value, 'text-btn compact') : ''}</div>`;
  return html`<section class="status-box" data-box="decision"${label ? html` aria-label="${label}"` : ''}${attrs ? ' ' + attrs : ''}><div class="status-head">${statusMark(tone)}<div><h2>${title}</h2>${caption ? html`<p class="status-caption">${caption}</p>` : ''}${reasons.length ? html`<p class="status-reasons owner-text">${reasons.map((r, i) => html`${i ? ' ' : ''}${r}`)}</p>` : ''}</div></div>${checks.length ? html`<ol class="status-checks">${checks.map(row)}</ol>` : ''}${foot.map(need)}</section>`;
}
/* ---- the Evidence library (law 108):
 * the shapes the evidence views compose -- a page of a long list and its foot, the search above
 * it, a row's title over its line, the review table and its group rows, the passage list, the
 * checklist, the properties, the activity feed, the measures and the document with its contents.
 * Each is fed data and nothing else; the views keep what to say, these how it is drawn, and the
 * workshop (`pages-kit.js`) shows each as a specimen. ---- */
/* A section's head (N3, law 121): its label, the label's (i), its action at the right. */
function sectionHead(title, sub = '', action = '') {
  return html`<header class="panel-head"><div class="panel-label"><h2 data-tip="@overflow">${title}</h2>${infoMark(sub)}</div>${action}</header>`; // a label cut by a narrow box reads whole on hover (law 88)
}
/* A readback slot carries its section's stack relation without becoming another painted box.
   An absent section leaves neither a marker nor an extra step, including after a partial read. */
function stackSlot(id, content, kind = 'section') {
  const marked = String(content || '').trim();
  return html`<div id="${id}"${marked ? html` data-stack-box="${kind}"` : ''}>${content}</div>`;
}
function fillStackSlot(slot, content, kind = 'section') {
  slot.innerHTML = content;
  if (String(content || '').trim()) slot.dataset.stackBox = kind;
  else slot.removeAttribute('data-stack-box');
}
/* The (i) of a label (N3): the explanation a reader may need, on hover and focus, never on the
 * face; the words are its accessible name. The same glyph as a folded line's rest (round 90). */
function infoMark(words) {
  return words && String(words).trim() ? html`<button type="button" class="hint-more" data-tip="${words}" aria-label="${words}">${icon('info')}</button>` : '';
}
/* A page of a long list (law 92): the page asked for, held to the pages there are. */
/* Fragments joined by a separator, kept as markup (UX2-01): `Array.join` stringifies each fragment, and the string,
 * interpolated, is escaped -- its tags print. An empty list is '' so a fallback can stand after `||`. */
const joinMarkup = (list, separator = ' · ') => list.reduce((out, x, i) => html`${out}${i ? separator : ''}${x}`, '');
/* The operator's offline switch where an owner's sentence names it (V620, U93): the literal a person types, set as
 * code, never a raw code among the words; any other text comes back as it came. */
const OFFLINE_SWITCH = 'ALPHALATTICE_NETWORK_DISABLED=1';
const literalWords = (text) => (typeof text === 'string' && text.includes(OFFLINE_SWITCH) ? raw(esc(text).split(OFFLINE_SWITCH).join(`<code class="literal">${OFFLINE_SWITCH}</code>`)) : text);
function pageOf(list, page, size = LIST_PAGE) {
  const pages = Math.max(1, Math.ceil(list.length / size)), at = Math.min(Math.max(0, page || 0), pages - 1);
  return {shown: list.slice(at * size, at * size + size), start: at * size, page: at, pages};
}
/* The foot of a paged list: the count in words (`one` / `many`, when `total` is given), the page,
 * and Previous / Next -- each an [action, value] pair, held at its end. */
function pager({total = null, one = '', many = '', page, pages, prev, next}) {
  if (pages <= 1) return total === null ? '' : html`<div class="table-foot"><span>${countText(total, one, many)}</span></div>`; // N6: one page has no page to turn
  const turn = (label, [action, value], held, reason) => btnAttrs(t(label), action, value, 'button compact', held ? html`aria-disabled="true" data-tip="${t(reason)}"` : '');
  return html`<div class="table-foot"><span>${total === null ? '' : html`${countText(total, one, many)} · `}${t('Page {n} of {total}', {n: count(page + 1), total: count(pages)})}</span><span class="table-pages">${turn('Previous', prev, page === 0, 'First page')}${turn('Next', next, page >= pages - 1, 'Last page')}</span></div>`;
}
/* A long table's page turns at its head too (the user's phase 6 reading): the same turns as its foot, shown only when
 * there is more than one page; the count stays the foot's. */
function pagerTop({page, pages, prev, next}) {
  if (pages <= 1) return '';
  const turn = (label, [action, value], held, reason) => btnAttrs(t(label), action, value, 'button compact', held ? html`aria-disabled="true" data-tip="${t(reason)}"` : '');
  return html`<div class="table-head-pager"><span>${t('Page {n} of {total}', {n: count(page + 1), total: count(pages)})}</span><span class="table-pages">${turn('pager|Previous', prev, page === 0, 'First page')}${turn('pager|Next', next, page >= pages - 1, 'Last page')}</span></div>`;
}
/* The way to a long page's sections (the user's phase 6 reading): one row of their names at the head; a press brings a
 * section's head into view. `items` are [word, selector, present]; a section the page does not hold is not offered. */
function pageJumps(items) {
  const shown = items.filter(([, , present]) => present);
  return shown.length < 2 ? '' : html`<nav class="page-jumps" aria-label="${t('On this page')}">${shown.map(([word, selector]) => btn(word, 'page-jump', selector, 'text-btn'))}</nav>`;
}
function jumpTo(selector) {
  let place = null;
  try { place = document.querySelector('#main ' + selector); } catch { place = null; }
  if (place) place.scrollIntoView({block: 'start', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
}
/* The search above a long list: one field, named for the reader, the query as typed; `tools` (the
 * list's display options) sit after it on the same line. */
function searchBar(id, label, placeholder, value, tools = '') {
  return html`<div class="holdings-toolbar"><div class="holdings-filters"><label for="${id}" class="sr-only">${label}</label><div class="search-field">${icon('search')}<input id="${id}" class="search-input" value="${value}" placeholder="${placeholder}"></div>${tools}</div></div>`;
}

/* ---- the lobby (LS1; law 136) ----
 * One component for every list of objects, Linear's issues list: the rows under collapsible group
 * heads with their counts, grouped on the list's own axis -- its state where it has a lifecycle (the
 * active groups open, the ended folded), its time where it holds finished work (the reader's day,
 * This week, Earlier this month open; each month before folded), its family where the owner groups it so. A
 * group shows `LOBBY.shown` rows and `Show n more` expands it in place (`Show less`). The tool line:
 * the search, the Filter chips, Display (the axis, the facts, the density). The viewer's folds and
 * display are kept per list (`lobby.<list>`, `display.<list>`); the search and the filters are the
 * page's while it is open, or the route's where the list binds them (`bind`). A list declares {items, row, axes, words, filters, properties} and a
 * `foot` where its owner holds more than was read (law 17: the list says so). */
const Lobby = (() => {
  const S = new Map(), SPECS = new Map(), LISTED = new Map();
  const state = (name) => { if (!S.has(name)) S.set(name, {query: '', filters: {}, pending: '', more: new Set(), defaults: {}}); return S.get(name); };
  const prefs = (name) => readPreference('lobby.' + name) || {};
  const split = (v) => { const s = String(v), i = s.indexOf(':'); return [s.slice(0, i), s.slice(i + 1)]; };
  const repaint = () => { if (typeof patchMain === 'function') patchMain(); };
  const shut = (name, key) => { const f = prefs(name).folded || {}; return key in f ? Boolean(f[key]) : state(name).defaults[key] === false; };
  // A list that keeps its search and filters elsewhere (History: in the route, so a link shows the
  // same records) binds them -- `get()` reads {query, filters}, `set(patch)` writes; otherwise they
  // are the page's while it is open.
  const read = (name) => { const b = SPECS.get(name)?.bind, st = state(name); return b ? {query: '', filters: {}, ...b.get()} : {query: st.query, filters: st.filters}; };
  const write = (name, patch) => { const b = SPECS.get(name)?.bind, st = state(name); if (b) b.set(patch); else Object.assign(st, patch); repaint(); };
  function fold(value) {
    const [name, key] = split(value), f = {...(prefs(name).folded || {})};
    f[key] = !shut(name, key);
    savePreference('lobby.' + name, {...prefs(name), folded: f});
    repaint();
  }
  function more(value) { const [name, key] = split(value), m = state(name).more; if (m.has(key)) m.delete(key); else m.add(key); repaint(); }
  function query(name, value) { write(name, {query: String(value || '')}); }
  // a filter chip is named `lobby-<list>-<field>` (the list's name has no dash of its own)
  const chipOf = (name) => { const [, list, ...field] = String(name).split('-'); return [list, field.join('-')]; };
  function setFilter(spec) { const [name, value] = split(spec), [list, field] = chipOf(name), st = state(list); if (st.pending === field) st.pending = ''; write(list, {filters: {...read(list).filters, [field]: value}}); }
  function clearFilter(name) { const [list, field] = chipOf(name), st = state(list), filters = {...read(list).filters}; delete filters[field]; if (st.pending === field) st.pending = ''; write(list, {filters}); }
  function clearAll(list) { state(list).pending = ''; write(list, {query: '', filters: {}}); }
  function addClause(name) {
    const [list, field] = chipOf(name); state(list).pending = field; repaint();
    const trigger = document.getElementById('filter-' + name); if (trigger && typeof Picker !== 'undefined') Picker.open(trigger);
  }
  const displaySpec = (spec) => ({groupings: spec.axes.map((a) => [a.key, a.label]), orderings: spec.orders || [], properties: spec.properties || [], density: true, apply: (d) => { spec.apply?.(d); repaint(); }});
  /* The owner holds more than was read (law 17: a list never holds less than it says without
   * saying so): the foot says so and reads the next page. */
  const older = (words, action, value = '', busy = false, word = t('Read older')) => html`<p class="lobby-older">${words}${btnAttrs(busy ? t('Reading…') : word, action, value, 'text-btn', busy ? 'disabled aria-busy="true"' : '')}</p>`;
  function render(name, spec) {
    SPECS.set(name, spec);
    const st = state(name), d = displayState(name, displaySpec(spec)), now = read(name), q = now.query.trim().toLowerCase();
    const fields = (spec.filters || []).map((f) => ({...f, name: `lobby-${name}-${f.field}`, value: now.filters[f.field] || ''}));
    // a list that owns its filtering and its order (History's sorts) selects; any other is filtered here
    const items = spec.select ? spec.select(d) : spec.items.filter((x) => (!q || String(spec.words(x)).toLowerCase().includes(q)) && fields.every((f) => { const v = chipValues(f.value); return !v.length || v.some((one) => f.test(x, one)); }));
    const axis = spec.axes.find((a) => a.key === d.group) || spec.axes[0];
    const groups = new Map();
    for (const x of items) { const g = axis.group(x); if (!groups.has(g.key)) groups.set(g.key, {...g, items: []}); groups.get(g.key).items.push(x); }
    const ordered = [...groups.values()].sort((a, b) => (a.rank ?? 0) - (b.rank ?? 0));
    // A group's default is decided when the reader arrives at the list: while they stay, no group
    // folds or opens by itself (a Task that starts running does not fold the ended ones under the
    // reader's selection); a group that appears takes its own default.
    const arriving = !(typeof document !== 'undefined' && document.querySelector?.(`#main [data-lobby="${name}"]`));
    if (arriving) st.defaults = {};
    for (const g of ordered) if (!(g.key in st.defaults)) st.defaults[g.key] = g.open !== false;
    if (ordered.length && ordered.every((g) => shut(name, g.key))) st.defaults[ordered[0].key] = true; // nothing open (nothing at work, or the work just ended): the first group is not folded away -- unless the reader folded it
    LISTED.set(name, ordered.flatMap((g) => g.items)); // the rows in the order shown: J/K and a record's `n / m` step through them
    const block = (g) => {
      const folded = shut(name, g.key), all = st.more.has(g.key), n = g.items.length;
      const shown = folded ? [] : all ? g.items : g.items.slice(0, LOBBY.shown);
      // a group's note is its state's way on (a held Task's), in the owner's table's words
      const head = html`<div class="group-head lobby-head">${btnAttrs(html`${icon('chevron')}<span>${g.label}</span><b class="num">${count(n)}</b>`, 'lobby-fold', `${name}:${g.key}`, 'lobby-fold', html`aria-expanded="${!folded}"`)}${g.note ? html`<span class="lobby-note">${g.note}</span>` : ''}</div>`;
      const rest = !folded && n > LOBBY.shown ? btnAttrs(all ? t('Show less') : countText(n - LOBBY.shown, 'Show {n} more', 'Show {n} more'), 'lobby-more', `${name}:${g.key}`, 'text-btn lobby-more') : '';
      return html`${head}${shown.map((x) => spec.row(x, d))}${rest}`;
    };
    const search = spec.words ? html`<div class="search-field lobby-search">${icon('search')}<input id="lobby-q-${name}" class="search-input" type="search" data-lobby-query="${name}" value="${now.query}" placeholder="${spec.placeholder || t('Search')}" aria-label="${spec.placeholder || t('Search')}" autocomplete="off"></div>` : '';
    const tools = html`<div class="lobby-tools">${search}${fields.length ? filterBar(fields, {pending: st.pending ? `lobby-${name}-${st.pending}` : '', clear: 'lobby-clear', clearValue: name}) : ''}${displayOptions(name, displaySpec(spec))}</div>`;
    const hidden = spec.items.length - items.length;
    const body = items.length ? html`<div class="card-list lines lobby-list slotted${d.density === 'compact' ? ' density-compact' : ''}${spec.cls ? ' ' + spec.cls : ''}">${ordered.map(block)}</div>${hidden ? listFoot(hidden, btnAttrs(t('Clear filters'), 'lobby-clear', name, 'text-btn')) : ''}`
      : emptyState(t('No match for this search and these filters'), btnAttrs(t('Clear filters'), 'lobby-clear', name, 'text-btn'), '', 'nomatch');
    return html`${tools}<section class="panel lobby" data-lobby="${name}"><div class="panel-body">${body}${spec.foot || ''}</div></section>`;
  }
  return {render, fold, more, query, setFilter, clearFilter, clearAll, addClause, older, listed: (name) => LISTED.get(name) || [], owns: (name) => String(name).startsWith('lobby-')};
})();
/* The time axis (law 136): the reader's own day -- named by its date, as the reading grammar names
 * it (a development replay says nothing of the present) -- This week (from Monday), Earlier this
 * month -- open -- then each month before, folded; an item without a date is `Undated`, last. Keys
 * are stable across days. */
function timeGroup(iso) {
  const p = instantParts(iso);
  if (!p) return {key: 'undated', label: t('Undated'), rank: 9e9, open: false};
  const now = new Date(), day = new Date(p.year, p.month - 1, p.day), start = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const monday = new Date(start); monday.setDate(start.getDate() - ((start.getDay() + 6) % 7));
  if (day >= start) return {key: 'day', label: when(`${now.getFullYear()}-${pad2(now.getMonth() + 1)}-${pad2(now.getDate())}`), rank: 0, open: true};
  if (day >= monday) return {key: 'week', label: t('This week'), rank: 1, open: true};
  if (p.year === now.getFullYear() && p.month === now.getMonth() + 1) return {key: 'month', label: t('Earlier this month'), rank: 2, open: true};
  const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  const label = I18N.zh ? `${p.year === now.getFullYear() ? '' : p.year + '年'}${p.month}月` : `${t(months[p.month - 1])}${p.year === now.getFullYear() ? '' : ' ' + p.year}`;
  return {key: `m-${p.year}-${pad2(p.month)}`, label, rank: 3 + (now.getFullYear() * 12 + now.getMonth() - (p.year * 12 + p.month - 1)), open: false};
}
/* A row's title and its line under it, one opener (E2-E3): the title in ink, the line clamped to
 * two in the owner's words (`sub` null writes none); the row's reading opens from it. */
function rowTitle({title, sub = null, titleCls = '', action, value, cls = 'finding-open', attrs = ''}) {
  return btnAttrs(html`<span class="finding-title${titleCls ? ' ' + titleCls : ''}">${title}</span>${sub === null ? '' : html`<span class="finding-sub owner-text">${sub}</span>`}`, action, value, cls, attrs);
}
/* The review table (E2; Linear's grouped list, measured): the head's columns ([label, class]),
 * the rows, one word a cell; `cls` names the table's kind. A group is a row of its own
 * (`tableGroup`) across the columns, its title and its note. */
function groupedTable({columns, rows, cls = ''}) {
  return html`<div class="table-scroll"><table class="data-table review-table${cls ? ' ' + cls : ''}"><thead><tr>${columns.map(([label, c]) => html`<th scope="col" class="${c}">${label}</th>`)}</tr></thead><tbody>${rows}</tbody></table></div>`;
}
const tableGroup = (span, title, note) => html`<tr class="table-group"><th colspan="${span}" scope="rowgroup">${title}<span class="table-group-note">${note}</span></th></tr>`;
const tableRow = (attrs, cells) => html`<tr${attrs ? ' ' + attrs : ''}>${cells.map(([c, v]) => html`<td class="${c}">${v}</td>`)}</tr>`;
/* The passage list (E3; Mail's list, measured): a passage is its document over its words (two
 * lines; whole in the pane), then the issuer, the day and the state, one word each. */
function passageList(passages) {
  // B2 (item 4): where the passages carry the findings that cite them (`cited`), that column stands for the state's one word
  const cites = passages.some((p) => p.cited != null);
  const row = (p) => tableRow(html`data-span="${p.handle}"${p.selected ? html` aria-selected="true"` : ''}`, [
    ['col-text col-absorb', rowTitle({title: html`${p.type || t('Document')} <span class="mono muted">${p.document}</span>`, sub: p.excerpt, action: p.action, value: p.value, cls: 'finding-open passage-open', attrs: 'data-row-press'})],
    ['col-text col-tight', p.issuer], ['col-date col-tight', p.day], cites ? ['col-text col-tight passage-cited', p.cited ?? ''] : ['col-status col-tight', p.state]]);
  return groupedTable({columns: [[t('Passage'), 'col-text col-absorb'], [t('Issuer'), 'col-text col-tight'], [t('Available'), 'col-date col-tight'], cites ? [t('Cited by'), 'col-text col-tight'] : [t('State'), 'col-status col-tight']], rows: passages.map(row), cls: 'passages-table'});
}
/* A checklist (E3; GitHub's checks, measured): a fold whose summary counts what is met and what is
 * left -- open while anything is -- one row a check, verified or blocked, its line whole on hover,
 * and under an unmet check the way to meet it (`detail`). */
function checkList({title, rows, key, rowCls = '', cls = '', attrs = ''}) {
  const unmet = rows.filter((r) => !r.met);
  const list = html`<div class="card-list lines">${rows.map((r) => html`${objectRow({state: r.met ? 'verified' : 'blocked', name: r.name, why: r.why, cls: rowCls}, {key: key + ':' + r.id, attrs: html`data-tip="${String(r.why).replace(/<[^>]+>/g, '')}"`})}${!r.met && r.detail ? r.detail : ''}`)}</div>`;
  return html`<details class="reveal-details${cls ? ' ' + cls : ''}"${attrs ? ' ' + attrs : ''}${unmet.length ? ' open' : ''}><summary>${title} <span class="num">${count(rows.length - unmet.length)} / ${count(rows.length)}</span>${unmet.length ? html` · ${t('{n} to do', {n: count(unmet.length)})}` : ''}</summary>${list}</details>`;
}
/* An object's properties (G1; Linear's right column, measured): a glyph and a word a key, the value
 * in words; a property with nothing to say is not shown (round 93, rule 1). Rows: [icon, word, value]. */
const kvKey = (ic, word) => html`<span class="kv-key">${icon(ic)}${word}</span>`;
function propertyList(title, rows, attrs = '') {
  return panel(title, '', kv(rows.filter(([, , v]) => v !=='').map(([ic, word, v]) => [kvKey(ic, word), v]), 'kv-icons side-props'), '', attrs);
}
/* The activity feed (law 55; E1): the latest few, then every one of them; `render` draws a list of
 * runs (by day, paged), `more` is the action that opens all. */
function activityFeed({title, list, shown = 3, open = false, render, more, attrs = ''}) {
  if (!list.length) return '';
  const folded = list.length > shown && !open;
  return panel(title, '', folded ? html`${render(list.slice(0, shown))}${btnAttrs(html`${t('Show all {n} runs', {n: count(list.length)})}${icon('arrow')}`, more, '', 'text-btn compact activity-more')}` : render(list), '', attrs);
}
/* The measures (E1): a strip of figures in the box it measures itself by (six or three across). */
const measureStrip = (figures, label, cls = '') => html`<div class="es-measures-box">${rail(html`${figures}`, 'es-measures' + (cls ? ' ' + cls : ''), label)}</div>`;
/* An owner's count while its step runs: the number, its denominator in the owner's unit, and the
 * bar's two ends -- the one rendering of a count, for the data preparation's steps and the Evidence
 * preparation's stages (contract 10.10). */
const workCount = (n, total, unit) => ({count: html`<span class="tp-number">${Number(n).toLocaleString('en-US')}</span><span class="tp-denom"> / ${Number(total).toLocaleString('en-US')} ${unit}</span>`, bar: [Number(n), Number(total)]});
/* ---- end of the Evidence library ---- */
/* A record's properties (round 14): icon + label + value pairs in a two-column grid, the value a
 * link where the row names a route (`to`) or an action (`action`); `note` a second line under the
 * value. `opts`: eyebrow (with `about` behind an (i)), `body` (more inside the card: a list of
 * limitations, a disclosure), `fact` and `foot` (the card's foot: its own fact left, one opener
 * right). The one card a record page opens on. */
/* Sources as one kv (round 64): each value the way to its object; a `kv` row, not a block. */
function sourceRows(rows) {
  const value = (r) => r.to ? link(html`${r.value}${icon('arrow')}`, r.to.page, 'text-btn', r.to.extra || {}) : r.action ? btn(html`${r.value}${icon('arrow')}`, r.action.name, r.action.value, 'text-btn') : r.value;
  return rows.filter(Boolean).map((r) => [html`${r.icon ? icon(r.icon) : ''}${r.label}`, html`${value(r)}${r.note ? html`<span class="sub-cell">${r.note}</span>` : ''}`]);
}

/* ---- charts ---- */
/* The rows behind each drawn chart, by id, for the pointer, keyboard and navigator readers. The
 * Portfolio chart draws the opened book's series inside the reader's window; an overlay (the
 * comparison) registers its own rows and lines. Nothing is recomputed: the published index and
 * daily returns are drawn as returned, the window only chooses which rows are in view. */
const CHART_ROWS = new Map();
const chartGeometry = (compact = false) => ({w: param(innerWidth < BREAKPOINTS.window.phone ? 'chart-width-narrow' : 'chart-width'), h: param(compact ? 'chart-module' : 'chart-featured'), pad: {l: param('chart-pad-left'), r: param('chart-pad-right'), t: param('chart-pad-top'), b: param('chart-pad-bottom')}});
const PORTFOLIO_LINES = [{key: 'value', daily: 'daily', cls: 'series', label: 'Study'}, {key: 'benchmark', daily: 'benchmarkDaily', cls: 'benchmark', label: 'Benchmark'}];
const finiteValue = (z) => z !== null && z !== undefined && Number.isFinite(z);
/* A grid step of 1, 2, 2.5 or 5 at some power of ten, for about four intervals over a range. */
function niceStep(range) {
  const raw = Math.max(1e-9, range) / 4, mag = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw) || 10 * mag;
}
function chart(kind = 'indexed', compact = false, spec = null) {
  const isDaily = kind === 'daily';
  const {w, h, pad} = spec?.geometry || chartGeometry(compact); // a card may draw at its own size so its labels keep their scale
  const id = spec?.id || 'portfolio';
  const all = spec?.rows || Data.series();
  if (!all.length) return emptyState(t('No published observations'));
  const lines = spec?.lines || PORTFOLIO_LINES;
  const [a, b] = spec?.window || (id === 'portfolio' && typeof Inspect !== 'undefined' ? Inspect.window(all.length) : [0, all.length - 1]);
  CHART_ROWS.set(id, {rows: all, lines, window: [a, b], kind, tick: spec?.tick, axis: spec?.axis, value: spec?.value, note: spec?.note, title: spec?.title, label: spec?.label}); // N6: a chart may say its own ticks, values, reading note and what it is (law 17: a variance figure is never announced as returns)
  const n = b - a + 1;
  const values = lines.map((line) => all.slice(a, b + 1).map((r) => (isDaily ? r[line.daily] : r[line.key])));
  const flat = values.flat().filter(finiteValue);
  if (!flat.length) return emptyState(t('No finite observations'));
  const lo = Math.min(...flat, ...(isDaily ? [0] : [])), hi = Math.max(...flat, ...(isDaily ? [0] : []));
  const fixture = false;
  const step = fixture ? 1.5 : niceStep(hi - lo);
  const min = fixture ? -3 : Math.floor(lo / step) * step;
  const max = fixture ? 3 : Math.max(min + step, Math.ceil(hi / step) * step);
  const decimals = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
  // an axis in one power of ten (`axis: 'power'`; the user, 2026-09-24, on ticks that read 2.5e-5 … 1.0e-4): the ticks in the data's units, the power said once over them
  const top = Math.max(Math.abs(lo), Math.abs(hi)), power = spec?.axis === 'power' && top > 0 ? Math.floor(Math.log10(top)) : 0;
  const tickText = (g) => power ? String(Number((g / 10 ** power).toFixed(2))) : spec?.tick ? spec.tick(g) : g.toFixed(decimals);
  const x = (i) => pad.l + (i / Math.max(1, n - 1)) * (w - pad.l - pad.r);
  const y = (z) => pad.t + ((max - z) / (max - min)) * (h - pad.t - pad.b);
  const path = (zs) => zs.map((z, i) => (!finiteValue(z) ? '' : (i && finiteValue(zs[i - 1]) ? 'L' : 'M') + x(i).toFixed(2) + ' ' + y(z).toFixed(2))).join(' ');
  const grid = [];
  for (let g = min; g <= max + step / 1000; g += step) {
    grid.push(html`<line class="grid" x1="${pad.l}" x2="${w - pad.r}" y1="${y(g)}" y2="${y(g)}"/><text x="${w - pad.r + param('chart-tick-gap')}" y="${y(g) + param('chart-text-shift')}" text-anchor="start">${tickText(g)}${isDaily ? '%' : ''}</text>`);
  }
  if (power) grid.push(html`<text class="chart-unit" x="${w - pad.r + param('chart-tick-gap')}" y="${pad.t - param('chart-tick-gap') - param('chart-text-shift')}" text-anchor="start">×10${superscript(power)}</text>`);
  for (const i of [...new Set([0, n - 1])]) {
    grid.push(html`<text x="${x(i)}" y="${h - param('chart-tick-gap')}" text-anchor="${i === 0 ? 'start' : i === n - 1 ? 'end' : 'middle'}">${all[a + i].date}</text>`);
  }
  const barWidth = Math.max(0.6, Math.min(1.8, ((w - pad.l - pad.r) / n) * 0.7)).toFixed(2);
  const drawn = lines.map((line, k) => (line.cls === 'benchmark'
    ? html`<path class="benchmark" d="${path(values[k])}"/>`
    : isDaily
      ? values[k].map((z, i) => (!finiteValue(z) ? '' : html`<line x1="${x(i)}" x2="${x(i)}" y1="${y(0)}" y2="${y(z)}" class="bar ${line.cls}" stroke-width="${barWidth}" opacity=".8"/>`))
      : html`<path class="${line.cls}" d="${path(values[k])}"/>`));
  const label = spec?.label || t('Published historical returns; read-only chart.');
  const windowed = a > 0 || b < all.length - 1;
  // round 95 (Stocks' price label): the study's last value on the axis, in the series' ink, over a
  // ground patch so a tick under it is not read twice; the featured indexed chart only
  let lastLabel = '';
  if (!compact && !isDaily && id === 'portfolio') {
    const k = lines.findIndex((line) => line.cls !== 'benchmark'), zs = k >= 0 ? values[k] : [];
    let i = zs.length - 1; while (i >= 0 && !finiteValue(zs[i])) i--;
    if (i >= 0) { const ly = y(zs[i]), text = fmt(zs[i]); const chip = param('chart-tick-gap') - param('chart-chip-inset'), tall = param('chart-chip-height'); lastLabel = html`<rect class="chart-last-ground" x="${w - pad.r + chip}" y="${ly - tall / 2}" width="${pad.r - chip}" height="${tall}" rx="${param('chart-chip-inset')}"/><text class="chart-last" x="${w - pad.r + param('chart-tick-gap')}" y="${ly + param('chart-text-shift')}" text-anchor="start">${text}</text>`; }
  }
  return html`<div class="chart-wrap ${compact ? 'chart-module' : 'chart-featured'}" data-chart-wrap><svg class="chart-svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}"><title>${spec?.title || t('Published study and benchmark')}${windowed ? html` · ${t('{from} — {to}', {from: all[a].date, to: all[b].date})}` : ''}</title>${grid}${drawn}${lastLabel}<line class="chart-cross" data-cross x1="0" x2="0" y1="${pad.t}" y2="${h - pad.b}" opacity="0"/><circle class="chart-marker" data-marker r="${param('chart-marker')}" opacity="0"/><rect x="${pad.l}" y="${pad.t}" width="${w - pad.l - pad.r}" height="${h - pad.t - pad.b}" fill="transparent" data-chart="${kind}" data-chart-id="${id}" data-width="${w}" data-left="${pad.l}" data-right="${pad.r}" data-top="${pad.t}" data-bottom="${pad.b}" data-height="${h}" data-min="${min}" data-max="${max}" tabindex="0" role="button" aria-label="${t('Inspect published observations. Arrow keys select a date. Enter opens source details.')}"/></svg><div class="chart-tooltip" data-tooltip hidden></div></div>`;
}
/* A chart's viewBox takes its box's aspect: the box owns the height (`--chart-module`,
 * `--chart-featured`) and the page owns the width, so after a render (and on resize) each chart
 * is re-issued at its measured width and its labels keep their real size. Nothing is recomputed
 * but the geometry; the rows, lines and window are the ones the render registered. */
function fitCharts(root = document) {
  let fitted = 0;
  for (const wrap of root.querySelectorAll('[data-chart-wrap]')) {
    const svg = wrap.querySelector('svg.chart-svg'), target = wrap.querySelector('[data-chart]');
    if (!svg || !target) continue;
    const box = layoutRect(wrap); // layout px (round 95): under the root zoom a client rect is scaled, and a chart re-issued at the scaled width draws its ticks a size too small
    if (!box.width || !box.height) continue;
    const [, , w, h] = (svg.getAttribute('viewBox') || '').split(/\s+/).map(Number);
    const width = Math.round(box.width), height = Math.round(box.height);
    if (Math.abs(width - w) <= 8 && Math.abs(height - h) <= 8) continue;
    const id = target.dataset.chartId || 'portfolio', spec = CHART_ROWS.get(id);
    if (!spec) continue;
    const compact = wrap.classList.contains('chart-module');
    wrap.outerHTML = String(chart(spec.kind, compact, {id, rows: spec.rows, lines: spec.lines, window: spec.window, tick: spec.tick, axis: spec.axis, value: spec.value, note: spec.note, title: spec.title, label: spec.label, geometry: {w: width, h: height, pad: chartGeometry().pad}}));
    fitted += 1;
  }
  return fitted;
}
/* A scroll region records which of its edges still hides content (`data-edge`: left, right or
 * both), read after a render and on its own scroll and the window's resize; the sheet shows
 * the thin bar over its tinted track while there is something to roll (law 90) -- no edge dims
 * the content (law 112). */
/* A list page's empty state stands at one place on every page (the user's readings, 2026-09-23: 带 logo
 * 的空页面 logo 上下乱飘; logo 最顶端和 UI 最顶端距离一致): its mark's top edge `empty-top` below the top of
 * the UI, at any window height, in layout px (the text size scales it); moved down only, never above
 * where the page's own content leaves it. The place is one device pixel on every page: the shift
 * is not rounded in layout px (that kept each page's own fraction, at 125 % half a device pixel,
 * and the way's hard edge showed it -- the user, 2026-09-23: 黑色胶囊换页时会有轻微漂移); the hardest
 * edge, the way when there is one, else the mark, is snapped to the device grid (the text size
 * times the screen's scale), so the pill's edges are crisp and the mark is within half a pixel.
 * A place at exactly half a pixel rounds one way on every page: the measured parts carry float
 * noise (at 125 % the way stood at 358.5 and rounded to 358 on one page, 359 on the next). */
function placeEmpty(root = document) {
  const top = document.getElementById('top');
  if (!top || !root.querySelectorAll) return;
  const scale = rootZoom() * (globalThis.devicePixelRatio || 1);
  for (const e of root.querySelectorAll('.section-empty.page-empty')) {
    e.style.removeProperty('--empty-shift');
    const mark = e.querySelector('.empty-mark') || e;
    if (!mark.getBoundingClientRect().height) continue;
    const way = e.querySelector(':scope > :is(.button, .text-btn)');
    const under = way && way.getBoundingClientRect().height ? layoutRect(way).top - layoutRect(mark).top : 0;
    const target = Math.round((layoutRect(top).top + param('empty-top') + under) * scale + 1e-3) / scale - under;
    const shift = target - layoutRect(mark).top;
    if (shift > 0) e.style.setProperty('--empty-shift', Number(shift.toFixed(4)) + 'px');
  }
}
/* A clamped text says so (C2, law 140): `data-clamped` while the clamp hides words, read at the same
 * moments as the scroll edges; its way (`data-clamp-way`, the next element) stands only then, or
 * while the text is open (`data-clamp="open"`) to close it again. */
function markClamps(root = document) {
  if (!root.querySelectorAll) return;
  for (const el of root.querySelectorAll('[data-clamp]')) {
    const open = el.dataset.clamp === 'open', clamped = !open && el.scrollHeight > el.clientHeight + 1;
    el.toggleAttribute('data-clamped', clamped);
    const way = el.nextElementSibling;
    if (way?.hasAttribute('data-clamp-way')) way.hidden = !(clamped || open);
  }
}
/* A way to a place below the reader's view (C3, law 142: the new exchanges): `data-waypoint` (a
 * selector) stands while its place lies below the view and hides once the place is in view or
 * above it -- read at the same moments as the scroll edges, and on scroll. */
function markWaypoints(root = document) {
  if (!root.querySelectorAll) return;
  for (const el of root.querySelectorAll('[data-waypoint]')) {
    let place = null;
    try { place = document.querySelector(el.dataset.waypoint); } catch { place = null; }
    el.hidden = !place || layoutRect(place).top < viewH();
  }
}
/* The press on a waypoint: its place, brought into view -- smoothly unless the reader asked for less
 * motion -- and focused without a second scroll. */
function followWaypoint(way) {
  let place = null;
  try { place = way && document.querySelector(way.dataset.waypoint); } catch { place = null; }
  if (!place) return;
  place.scrollIntoView({block: 'center', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
  if (place.tabIndex >= 0) place.focus({preventScroll: true});
}
function markScrollEdges(root = document) {
  fitTabs(root); // a strip too long for its column shows More first (the same moments: a paint, a patch, a resize)
  placeEmpty(root);
  markClamps(root);
  markWaypoints(root);
  const mark = (el) => {
    const left = el.scrollLeft > 1, right = el.scrollLeft + el.clientWidth < el.scrollWidth - 1;
    const edge = [left ? 'left' : '', right ? 'right' : ''].filter(Boolean).join(' ');
    if (edge) el.dataset.edge = edge; else delete el.dataset.edge;
  };
  for (const el of root.querySelectorAll('.table-scroll')) { // the one thing that rolls sideways (FT5)
    mark(el);
    if (!el.dataset.edgeBound) { el.dataset.edgeBound = 'true'; el.addEventListener('scroll', () => mark(el), {passive: true}); }
  }
  if (!markScrollEdges.resizing) {
    markScrollEdges.resizing = true;
    let settle = 0, frame = 0;
    addEventListener('resize', () => { clearTimeout(settle); settle = setTimeout(() => markScrollEdges(), 150); }, {passive: true});
    // a waypoint follows the reader's scroll, once a frame (any scroller: the capture sees them all)
    addEventListener('scroll', () => { cancelAnimationFrame(frame); frame = requestAnimationFrame(() => markWaypoints()); }, {passive: true, capture: true});
  }
}
function chartHead(value, label, span) {
  return html`<div class="chart-headline"><span>${label}</span><strong>${value}</strong><small>${span}</small></div>`;
}
const SUPERSCRIPT = '⁰¹²³⁴⁵⁶⁷⁸⁹';
const superscript = (n) => String(n).replace(/-/g, '⁻').replace(/\d/g, (d) => SUPERSCRIPT[d]);
/* A figure (the user's rule, 2026-09-24: 以后图都需要带上背景box): every chart and every figure of a
 * page -- a drawn series, the coverage ruler, a composition -- stands in its own box, the raised
 * material (`data-box="figure"`, law 147; law 148 names a figure the first of a box's three purposes). Its
 * head is the figure's name, its (i) and its legend; its body the drawing and the words that read it. */
function figureBox(title, body, {info = '', legend = '', cls = '', attrs = ''} = {}) {
  return html`<section class="panel figure${cls ? ' ' + cls : ''}" data-box="figure"${attrs ? ' ' + attrs : ''}><div class="chart-toolbar"><div class="chart-heading"><div class="panel-label"><h2>${title}</h2>${info}</div>${legend}</div></div><div class="figure-body">${body}</div></section>`;
}
/* One meter (law 89, the user's rule, 2026-09-22): every bar in the product is this and nothing
 * is drawn by hand -- a share of one whole (`segments` in their tones, an optional `mark` for a
 * required minimum), a progress of one count toward another (`now` of `max`; an unknown extent
 * is a dashed track, never a travelling fill, law 16), or a composition (`slices` with a legend)
 * -- 6 px on the line's track, at most `--meter-max` (360 px) wide, left under its words. A
 * chart's bars are the chart's (law 18), not a meter. */
function meter(spec) {
  if (Array.isArray(spec)) spec = {kind: 'composition', slices: spec};
  const kind = spec.kind || 'share', cls = spec.cls ? ' ' + spec.cls : '', label = spec.label || '';
  if (kind === 'progress') {
    const now = Number(spec.now) || 0, max = Number(spec.max) || 0, fraction = max ? Math.min(1, now / max) : 0;
    return html`<div class="meter meter-progress${cls}"><span class="meter-bar${spec.indeterminate ? ' indeterminate' : ''}" role="progressbar" aria-label="${label}"${spec.indeterminate ? '' : html` aria-valuenow="${now}" aria-valuemin="0" aria-valuemax="${max}"`}><span${spec.id ? html` id="${spec.id}"` : ''} class="accent" data-fraction="${fraction.toFixed(4)}" style="width:${(fraction * 100).toFixed(2)}%"></span></span></div>`;
  }
  if (kind === 'share') {
    const segments = spec.segments || [], total = spec.total ?? segments.reduce((a, x) => a + (Number(x.n) || 0), 0);
    const width = (n) => total ? Math.max(0, Math.min(100, (Number(n) || 0) / total * 100)).toFixed(2) : '0';
    // B1: `markLabel` names the mark above it; `ends` says the whole's two ends under it (a gauge of one whole: 0 % and 100 %)
    const at = spec.mark != null ? Number(width(spec.mark * (spec.total ?? 1))) : null;
    const markRow = at !== null && spec.markLabel ? html`<span class="meter-mark-row"><span class="meter-mark-label" style="left:${at}%"${at > 85 ? ' data-edge="end"' : at < 15 ? ' data-edge="start"' : ''}>${spec.markLabel}</span></span>` : '';
    const ends = spec.ends ? html`<span class="meter-ends"><span>${spec.ends[0]}</span><span>${spec.ends[1]}</span></span>` : '';
    return html`<div class="meter meter-share${cls}">${spec.words === '' ? '' : html`<span class="meter-words">${spec.words || segments.map((x, i) => html`${i ? ' · ' : ''}${count(x.n)} ${x.label || ''}`)}</span>`}${markRow}<span class="meter-bar" role="img" aria-label="${label}">${segments.map((x) => html`<i class="${x.tone || 'neutral'}" style="width:${width(x.n)}%"></i>`)}${at !== null ? html`<b class="meter-mark" style="left:${at}%"></b>` : ''}</span>${ends}</div>`;
  }
  // N6: a composition of counts (the owner's) as well as of weights; `limit` slices before the rest is one (every slice when the legend lists them all)
  const slices = spec.slices || [], counted = spec.format === 'count', limit = spec.limit ?? 6;
  const total=slices.reduce((sum,[,value])=>sum+(Number(value)||0),0), shown=slices.slice(0,limit), tail=slices.slice(limit), rest=tail.reduce((sum,[,value])=>sum+(Number(value)||0),0);
  const other=t('Other {n} sectors',{n:tail.length}), values=rest>0 ? [...shown,[other,rest]] : shown;
  const figure=(value)=>counted ? count(value) : pctNumber(value), shownValue=(value)=>counted ? count(value) : num(value,'percent');
  // the slice's order is its step (`--slice`); its tone and fading are the meter's parameters, never numbers here
  const row=([name,value],i)=>html`<li><i data-ui-style="--slice:${i}"></i><span>${name}</span><b>${shownValue(value)}</b></li>`;
  // the rest is the bar's `Other` slice and the legend's one fold: opened, each of them at that slice's step
  const restFold=rest>0 ? html`<details class="meter-legend-rest"><summary><span>${other}</span><b>${shownValue(rest)}</b></summary><ul class="meter-legend">${tail.map((x)=>row(x,shown.length))}</ul></details>` : '';
  // one part is the whole: its legend line says it; a lone full bar reads as a progress stopped a third of the way (the user, 2026-09-24, Data maintenance)
  const bar=values.length>1 ? html`<span class="meter-bar composition" role="img" aria-label="${label || t('Sector composition')}">${values.map(([name,value],i)=>html`<i data-ui-style="flex-grow:${Math.max(0,Number(value)||0)};--slice:${i}" data-tip="${name} · ${figure(value)}"></i>`)}</span>` : '';
  return html`<div class="meter meter-composition${cls}" role="group" aria-label="${label || t('Sector composition')}">${bar}<ul class="meter-legend">${shown.map(row)}</ul>${restFold}<span class="sr-only">${figure(total)}</span></div>`;
}
/* ---- Code, coloured as GitHub colours it (the user's word, 2026-09-20): keys, strings, numbers
   and constants, comments; the punctuation in the text ink. YAML is read line by line -- a key
   before its colon, then the value, a comment after a space; a plain scalar is a string, as in
   GitHub's grammar. JSON is one expression. The result is escaped markup for a <pre> or the
   editor's paint; a text past the cap is escaped and left plain. */
const CODE_CAP = 200000;
const CODE_ESCAPES = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'};
const codeEsc = (s) => String(s).replace(/[&<>"]/g, (c) => CODE_ESCAPES[c]);
const codeSpan = (cls, s) => `<span class="${cls}">${codeEsc(s)}</span>`;
const YAML_WORD = [
  [/^(?:true|false|null|~|yes|no|on|off)$/i, 'c-const'],
  [/^(?:[-+]?(?:\d[\d_]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|0x[0-9a-fA-F_]+|0o[0-7_]+|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$/, 'c-num'],
  [/^[&*!]\S*$/, 'c-const'], // an anchor, an alias, a tag
];
const yamlWord = (body) => { for (const [re, cls] of YAML_WORD) if (re.test(body)) return codeSpan(cls, body); return codeSpan('c-str', body); };
const YAML_FLOW = /("(?:\\.|[^"\\])*"|'(?:[^']|'')*')(\s*:(?=[\s,}\]]|$))?|([^\s\[\]{},:"']+)(\s*:(?=[\s,}\]]|$))?|([\[\]{},:])/g;
function yamlFlow(text) { // a flow collection: [a, 1] or {k: v}
  let out = '', last = 0;
  for (const m of text.matchAll(YAML_FLOW)) {
    out += codeEsc(text.slice(last, m.index)); last = m.index + m[0].length;
    if (m[1] !== undefined) out += m[2] ? codeSpan('c-key', m[1]) + codeEsc(m[2]) : codeSpan('c-str', m[1]);
    else if (m[3] !== undefined) out += m[4] ? codeSpan('c-key', m[3]) + codeEsc(m[4]) : yamlWord(m[3]);
    else out += codeEsc(m[5]);
  }
  return out + codeEsc(text.slice(last));
}
function yamlValue(v) {
  const m = /^(\s*)([\s\S]*?)(\s*)$/.exec(v), body = m[2];
  if (!body) return codeEsc(v);
  const painted = /^[\[{]/.test(body) ? yamlFlow(body) : /^[|>][-+]?\d*$/.test(body) ? codeEsc(body) : yamlWord(body);
  return codeEsc(m[1]) + painted + codeEsc(m[3]);
}
const YAML_KEY = /^("(?:\\.|[^"\\])*"|'(?:[^']|'')*'|[^\s"'#][^:#]*?)(\s*:)(?=\s|$)/;
function yamlLine(line) {
  let cut = line.length, quote = '';
  for (let k = 0; k < line.length; k++) { // the comment: a # at the start or after a space, outside quotes
    const ch = line[k];
    if (quote) { if (ch === quote) quote = ''; }
    else if (ch === '"' || ch === "'") quote = ch;
    else if (ch === '#' && (k === 0 || /\s/.test(line[k - 1]))) { cut = k; break; }
  }
  const code = line.slice(0, cut), comment = line.slice(cut);
  const m = /^(\s*(?:-(?:\s+|$))*)([\s\S]*)$/.exec(code);
  let out = codeEsc(m[1]), rest = m[2];
  if (/^(?:---|\.\.\.)(?:\s|$)/.test(rest)) out += codeEsc(rest);
  else {
    const key = YAML_KEY.exec(rest);
    if (key) { out += codeSpan('c-key', key[1]) + codeEsc(key[2]); rest = rest.slice(key[0].length); }
    out += yamlValue(rest);
  }
  return out + (comment ? codeSpan('c-cmt', comment) : '');
}
const JSON_TOKEN = /"(?:\\.|[^"\\])*"(\s*:)?|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|\b(?:true|false|null)\b/g;
function jsonMarkup(text) {
  let out = '', last = 0;
  for (const m of text.matchAll(JSON_TOKEN)) {
    out += codeEsc(text.slice(last, m.index)); last = m.index + m[0].length;
    const tok = m[0];
    if (tok[0] === '"') out += m[1] ? codeSpan('c-key', tok.slice(0, -m[1].length)) + codeEsc(m[1]) : codeSpan('c-str', tok);
    else out += codeSpan(/^[-\d]/.test(tok) ? 'c-num' : 'c-const', tok);
  }
  return out + codeEsc(text.slice(last));
}
function codeMarkup(text, lang = 'yaml') {
  const s = String(text ?? '');
  if (s.length > CODE_CAP || lang === 'text') return codeEsc(s);
  return lang === 'json' ? jsonMarkup(s) : s.split('\n').map(yamlLine).join('\n');
}
const json = (value) => raw(codeMarkup(JSON.stringify(value, null, 2), 'json')); // a value as coloured JSON, for a <pre>
/* A code editor (PR2: one builder -- the Lab's declaration, a review's structured answer): the text a
 * person edits in a textarea laid over its painted copy (codeMarkup) and its line numbers. `attrs`
 * are written in tag position; `CodeEditor.input` repaints the copy and the numbers as the text is
 * typed, `CodeEditor.scroll` keeps them in step with the textarea's scroll (bound once, actions.js). */
function codeEditor(id, text, {lang = 'yaml', label = '', placeholder = '', attrs = ''} = {}) {
  const s = String(text ?? '');
  const lines = Array.from({length: Math.max(1, s.split('\n').length)}, (_, i) => html`<span>${i + 1}</span>`);
  return html`<div class="code-editor" data-code-lang="${lang}"><div class="editor-line-numbers" aria-hidden="true">${lines}</div><pre class="editor-paint" aria-hidden="true"><span>${raw(codeMarkup(s, lang))}${s.endsWith('\n') ? ' ' : ''}</span></pre><textarea id="${id}" class="yaml-editor" aria-label="${label}" spellcheck="false"${placeholder ? html` placeholder="${placeholder}"` : ''}${attrs ? ' ' + attrs : ''}>${s}</textarea></div>`;
}
const CodeEditor = {
  input(el) {
    const host = el.closest('.code-editor'); if (!host) return;
    const nums = host.querySelector(':scope > .editor-line-numbers'), count = Math.max(1, el.value.split('\n').length); // no empty numbered wall
    if (nums && nums.children.length !== count) nums.innerHTML = Array.from({length: count}, (_, i) => '<span>' + (i + 1) + '</span>').join('');
    // a <span> first defeats the parser's dropped leading newline; a trailing newline gets a space so the paint's last line exists as the textarea's does
    const paint = host.querySelector(':scope > .editor-paint');
    if (paint) paint.innerHTML = '<span>' + codeMarkup(el.value, host.dataset.codeLang || 'yaml') + (el.value.endsWith('\n') ? ' ' : '') + '</span>';
    CodeEditor.scroll(el);
  },
  scroll(el) {
    const host = el.closest('.code-editor'); if (!host) return;
    const nums = host.querySelector(':scope > .editor-line-numbers'), paint = host.querySelector(':scope > .editor-paint');
    if (nums) nums.style.transform = 'translateY(-' + el.scrollTop + 'px)';
    if (paint) { paint.scrollTop = el.scrollTop; paint.scrollLeft = el.scrollLeft; }
  },
};
const fmt = (v, n = 2) => v === null || v === undefined || !Number.isFinite(Number(v)) ? '' : Number(v).toFixed(n).replace(/^-/, '−');
/* ---- percentages for reading (user rule, 2026-09-16), one rule for every page: at most two
 * decimals, trailing zeroes trimmed (100.000% -> 100%, 9.90% -> 9.9%, 21.855% -> 21.86%),
 * rounded half-up on the decimal text; a tiny nonzero value keeps its own text rather than
 * reading as 0%; an owner-formatted percent that would round onto a threshold it does not reach
 * keeps its precision. Counts, ratios, bps, identifiers and parameters never pass through here;
 * owner values, thresholds and exports are untouched, and exact values stay inspectable. ---- */
// half-up rounding on the decimal text itself, so 21.855 rounds to 21.86 whatever its binary
// double says; returns null for text that is not a plain decimal
function roundDecimalText(text, places) {
  const m = String(text).match(/^(-?)(\d+)(?:\.(\d*))?$/); if (!m) return null;
  const sign = m[1], whole = m[2], frac = m[3] || '';
  let digits = whole + frac.slice(0, places).padEnd(places, '0');
  if (frac.length > places && frac[places] >= '5') { const arr = digits.split(''); let i = arr.length - 1; while (i >= 0) { if (arr[i] === '9') { arr[i] = '0'; i -= 1; } else { arr[i] = String(Number(arr[i]) + 1); break; } } if (i < 0) arr.unshift('1'); digits = arr.join(''); }
  const head = digits.slice(0, digits.length - places) || '0', tail = digits.slice(digits.length - places).replace(/0+$/, '');
  return (sign && /[1-9]/.test(digits) ? '-' : '') + head + (tail ? '.' + tail : '');
}
const decimalText = (n) => { const v = Number(n); if (Number.isInteger(v)) return String(v); const s = v.toPrecision(15); return s.includes('e') ? null : s.replace(/0+$/, '').replace(/\.$/, ''); };
// a number of percent (21.855 -> "21.86%", 9.9 -> "9.9%"); a tiny nonzero value that two decimals
// would read as 0% keeps two significant digits instead (0.000343… -> "0.00034%")
function pctNumber(n) {
  if (n == null || !Number.isFinite(Number(n))) return '';
  const v = Number(n), text = decimalText(v), rounded = text == null ? null : roundDecimalText(text, 2);
  if (rounded == null || (v !== 0 && Number(rounded) === 0)) { const places = Math.min(12, Math.max(3, 1 - Math.floor(Math.log10(Math.abs(v))))); return v.toFixed(places).replace(/0+$/, '').replace(/\.$/, '').replace(/^-/, '−') + '%'; }
  return rounded.replace(/^-/, '−') + '%';
}
/* One number reader for the financial surfaces: the existing helpers keep their exact-value
 * contracts; `num` only chooses the helper by kind and sets a percent's unit light. */
function num(value, kind = 'ratio', places = 2) {
  if (kind === 'percent') {
    const text = pctNumber(value);
    return text.endsWith('%') ? html`${text.slice(0, -1)}<span class="num-unit">%</span>` : text;
  }
  if (kind === 'count') return count(value);
  if (kind === 'bps') return value == null || !Number.isFinite(Number(value)) ? '' : html`${count(value)} ${unit('bps')}`;
  if (kind === 'delta') return signed(value, 'ratio', places);
  return fmt(value, places);
}
// an owner-formatted percent string ("100.000%"); `threshold` is another such string whose
// rounded form this value must not be mistaken for; any other text is shown as the owner wrote it
function pctText(s, threshold = null) {
  if (s == null || s === '') return '';
  const m = String(s).trim().match(/^(-?\d+(?:\.\d+)?)\s*%$/);
  if (!m) return String(s);
  const value = Number(m[1]), rounded = roundDecimalText(m[1], 2);
  if (rounded == null || (value !== 0 && Number(rounded) === 0)) return String(s);
  const shown = rounded.replace(/^-/, '−') + '%';
  if (threshold != null) { const tm = String(threshold).trim().match(/^(-?\d+(?:\.\d+)?)\s*%$/); if (tm && Number(tm[1]) !== value && roundDecimalText(tm[1], 2) + '%' === shown) return String(s); }
  return shown;
}
// a fraction of one (0.6 -> "60%")
/* ---- reading helpers: an owner's code in words (the code itself stays on hover and in the
   exports), a count with its noun, a session id told apart from its neighbours, an instant as a
   reader sees it. Nothing here changes a value; the owner's exact strings stay in the details. */
const CODE_WORDS = {
  SETTLE_EVIDENCE_CONTINUATION: 'Settle evidence continuation',
  SELECT_ANALYSIS: 'Choose the analysis',
  PREPARE_EVIDENCE: 'Prepare sources',
  ANALYZE_CONTINUED_EVIDENCE: 'Analyze continued evidence',
  READ_CRO_DOSSIER: 'Read the CRO dossier',
  SUBMIT_CRO_ASSESSMENT: 'Submit the CRO assessment',
  NOT_ALLOCATION_RISK: 'Separately linked Risk references',
  PREPARING_FEATURE_CLOSURE: 'Preparing features',
  noop: 'No work needed', completed: 'Completed',
  REFUSED_INVALID_COMMAND: 'Invalid command refused',
  'research_foundation.not_current': 'This Foundation is no longer current',
  COMPLETE_RECEIPT_AND_SCORE_SUPPORT_REQUIRED: 'Complete receipt and score support required',
  NO_COMMON_INTERSECTION_OR_ZERO_FILL: 'No common intersection or zero filling',
  NO_STATISTICAL_SIGNIFICANCE_OR_SELECTION_CLAIM: 'No statistical significance or selection claim',
  'unstructured document': 'Unstructured document',
  // Existing bilingual owner states, source metadata, structure and claim limits (U117).
  "ABANDONED": "Abandoned",
  "ACCEPTED": "Accepted",
  "ACTIVATED": "Activated",
  "ACTIVE": "Active",
  "AGENT_BUNDLE_READY": "Ready",
  "ALL_SLEEVES": "All sleeves",
  "BOUND": "Bound",
  "BOUND_SOURCE_METADATA": "Bound source metadata",
  "BUSY": "Busy",
  "COMMAND_RETURNED": "Command returned",
  "COMMITMENTS_RESTRUCTURING_PENSIONS": "Commitments restructuring pensions",
  "COMPLETED": "Completed",
  "COMPUTED": "Computed",
  "CONDITIONAL": "Conditional",
  "CONTROLS_GOVERNANCE_INSIDER": "Controls governance insider",
  "CORE": "Core",
  "CORRECT": "Returned for correction",
  "CPU_BUDGET": "CPU budget",
  "CURRENT": "Current",
  "DONE": "Done",
  "EXTERNAL_ASSESSMENT_IS_NOT_A_ROUTE_OR_PROTECTED_VALIDATION": "External assessment is not a route or protected validation",
  "EXTERNAL_FINDINGS_ARE_NOT_A_CRO_REVIEW_OR_TRADE_AUTHORITY": "External findings are not a CRO review or trade authority",
  "FINANCIAL_STATEMENTS": "Financial statements",
  "FINANCING_CAPITAL": "Financing capital",
  "FOLLOWING_COMMON_SESSION_OFFICIAL_OPEN": "Following common session official open",
  "FORMATION_OFFICIAL_CLOSE": "Formation official close",
  "FORWARD_RESEARCH_NOT_TRADING_ADVICE": "Forward research not trading advice",
  "FRONT_MATTER": "Front matter",
  "HISTORICAL_REVIEW": "Historical review",
  "INPUTS_READY_NOT_PORTFOLIO_EXECUTED_NO_CURRENT_ACTIVATION": "Inputs ready not portfolio executed no current activation",
  "INPUT_QUALIFICATION_ONLY_NOT_BINDING_OR_FORWARD_ADMISSION": "Input qualification only not binding or forward admission",
  "INTEGRATED_TOPIC_ROUTING": "Integrated topic routing",
  "ITEM": "Item",
  "LEGAL_REGULATORY_TAX": "Legal regulatory tax",
  "METADATA_DISCOVERY_NOT_EVIDENCE_VERIFICATION": "Metadata discovery not evidence verification",
  "NAVIGATION_NOT_AUTHORITY": "Navigation not authority",
  "NEXT_COMMON_SESSION_OFFICIAL_OPEN": "Next common session official open",
  "NOTE": "Note",
  "NOT_ACTIVE": "Not active",
  "NOT_REVIEWED": "Not reviewed",
  "NOT_SELECTED": "Not selected",
  "NO_CURRENT_STRATEGY_ACTIVATION": "No current strategy activation",
  "NO_PROSPECTIVE_VALIDATION": "No prospective validation",
  "OBSERVER_STATE_NOT_EXECUTION_STATE": "Observer state not execution state",
  "OPERATIONS_RESULTS_OUTLOOK": "Operations results outlook",
  "OTHER_UNMAPPED": "Other unmapped",
  "PART": "Part",
  "PINNED": "Pinned",
  "PRESERVE_DRIFTED_SLEEVE_NOTIONAL": "Preserve drifted sleeve notional",
  "PRODUCT_ARTIFACT": "Product artifact",
  "READ": "Read",
  "READ_ONLY_CLOCK_PROJECTION_NOT_EXECUTION_AUTHORITY": "Read only clock projection not execution authority",
  "RECORDED_INPUT_COLUMNS_AND_CURRENT_DEFINITIONS_NOT_ROW_AVAILABILITY": "Recorded input columns and current definitions not row availability",
  "RECORDED_INPUT_PREPARATION_ONLY_NO_MODEL_FITS": "Recorded input preparation only no model fits",
  "REGISTERED_ARITHMETIC_NOT_MATERIALIZED_OR_ADMITTED_TO_FACTOR_RESEARCH": "Registered arithmetic not materialized or admitted to factor research",
  "RESET": "Reset",
  "RISK_FACTORS": "Risk factors",
  "RISK_REPORT_LINKED": "Risk report linked",
  "SAVED_PREPARED_FEATURE_SOURCE_CURRENT_ADMISSION_CHECKED_AT_PLAN": "Saved prepared feature source current admission checked at plan",
  "SAVED_SUMMARY_NOT_FULL_GRAPH_VERIFICATION": "Saved summary not full graph verification",
  "STOPPED": "Stopped",
  "SUB": "Sub",
  "SUBSEQUENT_EVENTS_AMENDMENTS": "Subsequent events amendments",
  "SUB_UNCERTAIN": "Uncertain subheading",
  "TASK_CONTROL": "Task Control",
  "TRANSACTIONS": "Transactions",
  // Remaining closed comparison/report values and exact reference kinds (U117).
  "CHANGED_ALIGNED": "Changed and aligned",
  "ABSENT_LATER": "Absent in the later filing",
  "UNRESOLVED": "Unresolved",
  "NOTHING_FILED": "Nothing filed",
  "ANNOUNCED": "Announced",
  "DISPUTED": "Disputed",
  "FROZEN_CANDIDATE": "Frozen candidate",
  "VALIDATED_HANDOFF": "Validated handoff",
  "CONDITIONAL_RESEARCH_PROPOSAL": "Conditional research proposal",
  "OBSERVED_RESEARCH_ENTRY": "Observed research entry",
  "EXPLICIT_CURRENT_SELECTION": "Chosen current analysis",
  "EXPLICIT_OLDER_CURRENT_SELECTION": "Chosen older current analysis",
  "PROVIDED_FOR_THIS_DELIVERY_NOT_ORIGINAL_EXPERIMENT_INTENT": "Question supplied for this delivery; original experiment intent not recorded",
  "NOT_RECORDED": "Not recorded",
  "HISTORICAL_ARRAY_REPLAY": "Historical array replay",
  "CURRENT_MODEL_SCORING": "Current model scoring",
  "Reference": "Reference",
  "alternative_evidence_access_receipt": "Alternative evidence access receipt",
  "alternative_evidence_analysis_publication": "Alternative evidence analysis publication",
  "alternative_evidence_analyst_brief": "Alternative evidence analyst brief",
  "alternative_evidence_document_set": "Alternative evidence document set",
  "alternative_evidence_registry": "Alternative evidence registry",
  "alternative_evidence_request": "Alternative evidence request",
  "alternative_evidence_retrieval_generation": "Alternative evidence retrieval generation",
  "alternative_evidence_source_document_set": "Alternative evidence source document set",
  "alternative_evidence_unit_failure": "Alternative evidence unit failure",
  "decision_stages": "Decision stages",
  "lifecycle_input_preparations": "Lifecycle input preparations",
  "lifecycle_input_publications": "Lifecycle input publications",
  "lifecycle_research_authorities": "Lifecycle research authorities",
  "publications": "Publications",
  "research_evidence": "Research evidence",
  "research_feature_materializations": "Research feature materializations",
  "research_feature_preparations": "Research feature preparations",
  "results": "Saved results",
  // Operation words follow the declared operation table; new operations must declare theirs.
  ACTIVITY_LIST: "Activity feed",
  ACTIVITY_RECENT: "Recent requests",
  ACTIVITY_REFUSALS: "Activity refusals",
  AGENT_ANSWER_SUBMIT: "Submit bundle answer",
  AGENT_BUNDLE_PREPARE: "Prepare agent bundle",
  COMPARE: "Compare installed results",
  CONTROLS: "Portfolio controls",
  CPU_BUDGET_SET: "Set CPU budget",
  CPU_BUDGET_SHOW: "CPU budget",
  CRO_REVIEW: "Request CRO review",
  CRO_REVIEW_DOSSIER: "Read CRO dossier",
  CRO_REVIEW_FINDING: "Read CRO finding",
  CRO_REVIEW_SUBMIT: "Submit CRO assessment",
  DATA_CHANGE_CONFIRM: "Confirm data change",
  DATA_ISSUES: "Data issues",
  DATA_ISSUE_CONFIRM: "Confirm data decision",
  DATA_ISSUE_DELEGATE: "Grant data decision",
  DATA_ISSUE_PREVIEW: "Preview data decision",
  DATA_ISSUE_REVOKE: "Revoke data grant",
  DATA_UPDATE_READBACK: "Read data update",
  EVENT_DECLARE: "Declare event",
  EVIDENCE_CONTINUE: "Continue source reading",
  EVIDENCE_DOCUMENTS: "Retained documents",
  EVIDENCE_LEDGER: "Reading ledger",
  EXPERIMENTS: "Saved experiments",
  EXPERIMENT_ALPHA_COMPARE: "Compare Alpha candidates",
  EXPERIMENT_COMPARE: "Compare Portfolio studies",
  EXPERIMENT_CONTINUE: "Continue saved study",
  EXPERIMENT_CONTROLS: "Study controls",
  EXPERIMENT_CURATION: "Curation choices",
  EXPERIMENT_DRAFT: "Draft from saved study",
  EXPERIMENT_FOUNDATIONS: "Sealed Foundations",
  EXPERIMENT_FOUNDATION_DRAFT: "Draft Alpha from Foundation",
  EXPERIMENT_FOUNDATION_READBACK: "Read sealed Foundation",
  EXPERIMENT_HANDOFF_PREVIEW: "Preview Alpha handoff",
  EXPERIMENT_PREVIEW_READBACK: "Read saved PLAN",
  EXPERIMENT_PROMOTE: "Run on the whole universe",
  EXPERIMENT_READBACK: "Read saved study",
  EXPERIMENT_RISK_LINKS: "Linked Risk reports",
  EXPERIMENT_SUMMARY: "Saved study summary",
  EXPERIMENT_FOUNDATION_SUMMARY: "Foundation summary",
  EXPERIMENT_VERIFY_ALL: "Verify saved studies",
  EXPORT: "Export result",
  FEATURE_ACTIVATE: "Activate formula factor",
  FEATURE_CATALOG_BUILD: "Build feature values",
  FEATURE_CATALOG_BUILD_READBACK: "Read feature build",
  FEATURE_CATALOG_CONTROLS: "Feature controls",
  FEATURE_CATALOG_PLAN: "Plan feature change",
  FEATURE_CATALOG_READBACK: "Read feature PLAN",
  FEATURE_DEACTIVATE: "Deactivate formula factor",
  FEATURE_EXTENSIONS: "Formula factors",
  FEATURE_REVIEW: "Formula factor review",
  FEATURE_TRIAL: "Run feature trial",
  FEATURE_TRIALS: "Feature trials",
  FEATURE_TRIAL_READBACK: "Read feature trial",
  FINALIZATION: "Finalization",
  FREEZE: "Freeze candidate",
  GOAL_ABANDON: "Abandon Goal",
  GOAL_ATTACH: "Attach Goal reference",
  GOAL_REFERENCE: "Read Goal reference",
  GOAL_CONTINUE: "Continue Goal study",
  GOAL_EXPORT: "Export Goal",
  GOAL_LIST: "Goals",
  GOAL_NARRATIVE: "Goal narrative",
  GOAL_NOTE: "Record Goal note",
  GOAL_OPEN: "Open Goal",
  GOAL_REVISE: "Revise Goal",
  GOAL_SCHEMA: "Goal schemas",
  GOAL_SHOW: "Read Goal",
  GOAL_SUBMIT: "Submit Goal completion",
  GOAL_TAKE: "Take Goal",
  MODEL_ACTIVATE: "Activate Alpha model",
  MODEL_DEACTIVATE: "Deactivate Alpha model",
  MODEL_EXTENSIONS: "Alpha models",
  MODEL_TRAINING_INPUT_PLAN: "Plan training inputs",
  MODEL_TRAINING_INPUT_PREPARE: "Prepare training inputs",
  MODEL_TRAINING_INPUT_READBACK: "Read training inputs",
  NETWORK_ACCESS: "Network access",
  NETWORK_ACCESS_SET: "Set network access",
  OPERATION_LIST: "Operations",
  PENDING_DECISIONS: "Pending decisions",
  PORTFOLIO_UPDATE_PLAN: "Portfolio update PLAN",
  PORTFOLIO_UPDATE_READBACK: "Read Portfolio update",
  PORTFOLIO_UPDATE_RUN: "Portfolio update RUN",
  RESEARCH_HISTORY: "Research history",
  RESEARCH_INPUTS: "Research inputs",
  RESEARCH_INPUT_READBACK: "Read input publication",
  RESEARCH_STRATEGY_CONTROLS: "Strategy preparation controls",
  RESEARCH_STRATEGY_INSTALL: "Install research strategy",
  RESEARCH_STRATEGY_PLAN: "Strategy preparation PLAN",
  RESEARCH_STRATEGY_PREPARE: "Prepare research strategy",
  RESEARCH_STRATEGY_READBACK: "Read strategy preparation",
  RESEARCH_UPDATE_AUTOMATION_CONFIGURE: "Configure daily update",
  RESEARCH_UPDATE_AUTOMATION_READBACK: "Daily update automation",
  RESEARCH_UPDATE_PLAN: "Research update PLAN",
  RESEARCH_UPDATE_READBACK: "Read research update",
  RESEARCH_UPDATE_RUN: "Research update RUN",
  RESULTS: "Saved results",
  SESSION_USAGE_READ: "Read session usage",
  USAGE_READING: "Usage reading",
  USAGE_READING_SET: "Set usage reading",
  STATUS: "Task status",
  STORAGE_CONFIRM: "Confirm storage cleanup",
  STORAGE_EVIDENCE_REBUILD: "Rebuild evidence index",
  STORAGE_PIN: "Set retention pin",
  STORAGE_PLAN: "Storage cleanup PLAN",
  STORAGE_CAP_SHOW: "Storage cap readback", STORAGE_CAP_SET: "Set the storage cap",
  RAISE_THE_CAP_OR_PLAN_A_CLEANUP: "Raise the cap or plan a cleanup", SET_THE_STORAGE_CAP: "Set the storage cap",
  STORAGE_READBACK: "Storage readback",
  STRATEGY_ACTIVATE: "Run strategy forward",
  STRATEGY_CALIBRATION_PLAN: "Calibration PLAN",
  STRATEGY_CALIBRATION_READBACK: "Read calibration",
  STRATEGY_CALIBRATION_RUN: "Run calibration",
  STRATEGY_DEACTIVATE: "Stop strategy forward",
  STRATEGY_SCORE_PLAN: "Scoring PLAN",
  STRATEGY_SCORE_READBACK: "Read scoring",
  STRATEGY_SCORE_RUN: "Run scoring",
  TASKS: "Tasks",
  TASK_GUARDIAN: "Task supervision",
  TASK_INCIDENTS: "Task incidents",
  TASK_RECOVERY: "Task recovery",
  TASK_REMEDIATE: "Apply Task remedy",
  UPGRADE_ACKNOWLEDGE: "Acknowledge upgrade",
  UPGRADE_OVERVIEW: "Upgrade overview",
  WORKSPACE_BACKUP: "Back up workspace",
  WORKSPACE_BACKUPS: "Workspace backups",
  WORKSPACE_PREPARE_READBACK: "Read workspace preparation",
  WORKSPACE_SHOW: "Workspace overview",
  // U117: declared owner states, relations and kinds; never inferred from spelling.
  "VERIFIED_READBACK": "Owner readback checked",
  "VERIFIED_TASK_STATE": "Task state checked",
  "VERIFIED_REFUSAL": "Refusal checked",
  "SAVED_NOT_VERIFIED": "Saved; not checked",
  "QUESTION_RECORDED_BEFORE_TASK_ADMISSION": "Question recorded before task admission",
  "POST_HOC": "Post hoc",
  "NO_EXECUTION_TIME_PROOF": "No execution time proof",
  "MET": "Criterion met",
  "NOT_MET": "Criterion not met",
  "NOT_ASSESSED": "Criterion not assessed",
  "DECISION": "Decision",
  "CONCLUSION": "Conclusion",
  "PM_RESPONSE": "PM response",
  "SUPPORTS": "Supports",
  "DOES_NOT_SUPPORT": "Does not support",
  "INSUFFICIENT": "Insufficient",
  "OPEN": "state|Open",
  "REJECTED": "Rejected",
  "claude-code": "Claude Code",
  "codex": "Codex",
  "model": "Model",
  "assignment": "Assignment",
  "question": "Question",
  "pm_response": "PM response",
  "plan": "Plan",
  "decision": "Decision",
  "dead_end": "Dead end",
  "surprise": "Surprise",
  "answer": "Answer",
  "objection": "Objection",
  "CONFIRMED_PENDING_REVALIDATION": "Confirmed; revalidation pending",
  "alternative_evidence.document_intelligence": "Alternative evidence document intelligence",
  "chief_risk_officer.portfolio_review": "Chief risk officer portfolio review",
  "model_training_input_preparation": "Model training input preparation",
  "portfolio_calibration_preparation": "Portfolio calibration preparation",
  "portfolio_decision_update": "Portfolio decision update",
  "portfolio_public_development_replay": "Portfolio replay",
  "portfolio_public_protected_finalization": "Portfolio public protected finalization",
  "portfolio_public_watermark_advancement": "Portfolio public watermark advancement",
  "research_feature_materialization": "Research feature materialization",
  "research_input_capture": "Input publication",
  "research_strategy_preparation": "Research strategy preparation",
  "strategy_score_preparation": "Strategy score preparation",
  "evidence_preparation": "Evidence preparation",
  "evidence_refresh": "Evidence refresh",
  "STARTING": "Starting",
  "HEALTHY": "Healthy",
  "LIVENESS_STALE": "Liveness stale",
  "TERMINAL_SUCCEEDED": "Terminal succeeded",
  "TERMINAL_BLOCKED": "Terminal blocked",
  "TERMINAL_DEFERRED": "Terminal deferred",
  "NOT_RECOVERY_REQUIRED": "Recovery not required",
  "REFUSED_CHANGED_SINCE_ADMISSION": "Refused; changed since admission",
  "REPLAN_OFFERED": "Re-PLAN offered",
  "ANSWERED": "Answered",
  "COMPACTED": "Compacted",
  "OPERATIONAL_ASSERTION": "Operational assertion",
  "TASK_CONTROL_ASSERTION": "Task control assertion",
  "ARTIFACT_ASSERTION": "Artifact assertion",
  "GUARDIAN_ASSERTION": "Guardian assertion",
  "AGENT_PROPOSAL": "Agent proposal",
  "HOST_DECISION": "Host decision",
  "NOT_ADDRESSED": "Not addressed",
  "INVALID": "Invalid",
  "MISSING": "Missing",
  "ADMITTED_AS_TASK": "Admitted as Task",
  "PORTFOLIO_MARKET_PREPARATION_AND_BOOK_REPLAY": "Portfolio market preparation and book replay",
  "ALPHA_FIT_AND_SCORE": "Alpha fit and score",
  "MODEL_FIT_AND_SCORE": "Model fit and score",
  "RISK_MODEL": "Risk model",
  "FACTOR_SCREEN": "Factor screen",
  "ALPHA_FIT": "Alpha fit",
  "FEATURE_VALUES": "Feature values",
  "SOURCE_PRICES": "Source prices",
  "COMPARABLE": "Comparable",
  "INCOMPLETE": "Incomplete",
  "DESCRIPTIVE_ALPHA_COMPARISON_NO_SELECTION": "Descriptive Alpha comparison; nothing is selected",
  "EXPERIMENT_PUBLISHED": "Experiment published",
  DEVELOPMENT_FAILED: 'Development failed', RESTATED_LATER: 'Restated later', COMPARED_LATER: 'Compared later',
  LIMITED: 'Limited', STRONG: 'Strong',
  PRODUCT_OPERATION: 'Product operation', EXTERNAL_CLIENT: 'External client',
  ProductOperationObserved: 'Product operation observation', TaskControlTransition: 'Task Control transition',
  ArtifactVerificationObserved: 'Artifact verification observation', ExternalActivityObserved: 'External activity observation',
  "REPORT": "Report",
  'research_foundation.admission_artifact_unavailable': 'Foundation sources unavailable',
  ANALYST_PACKET_PREPARED: 'Analyst packet prepared', effort: 'Effort',
  RECORDED_SOURCE_POLICY: 'Recorded source policy',
  market_reference_derived: 'Market reference derived', provider_daily_bars: 'Provider daily bars',
  sector_aggregate_derived: 'Sector aggregate derived', verified_panel_child: 'Verified panel child',
  'task_control.ledger_rebuilt': 'Task control ledger rebuilt', WORKSPACE_DEFAULT: 'Workspace default',
  CLOSE_T: 'At the formation session\'s close', // the position clock's decision phase (portfolio_result_context)
  POST_OBSERVED_QA_NOT_TIMELY_ADVICE: 'Not timely advice', // a research update's claim (answers.json: post-observed QA, not timely advice)
  'alternative_evidence.answer_problems': 'Returned by the evidence owner',
  'chief_risk_officer.answer_problems': 'Returned by the review owner',
  'chief_risk_officer.portfolio-evidence-review': 'CRO portfolio evidence review', // the review route's policy (codeCell)
  // an installed book's Risk role, as its strategy package declares it (the formula kept as written)
  REPORT_ONLY_DFD_PRIME_PLUS_E_NEVER_USED_FOR_WEIGHTS: 'Report only (DFD′ + E); never used for weights',
  POLICY_CONSUMED_RISK_ALLOCATION_PLUS_REPORT_ONLY_DFD_PRIME_PLUS_E: 'Risk allocation consumed by the policy; DFD′ + E report only',
  // the owner's comparison (installed and authored): its disposition and its dimensions, in words
  DECLARED_PATH_COMPARISON_NO_SELECTION: 'Declared paths compared; nothing is selected', HOLDINGS: 'Holdings', CONCENTRATION: 'Concentration', COST: 'Cost', TURNOVER: 'Turnover', PERFORMANCE: 'Performance', LIMITATIONS: 'Limitations',
  DEVELOPMENT_EVIDENCE_ONLY: 'Development evidence only', DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED: 'Development evidence only · never published',
  NO_FOUNDATION_OR_STRATEGY_ACTIVATION: 'No Foundation or strategy is activated', NO_INDEPENDENT_SCIENTIFIC_VALIDATION: 'No independent scientific validation',
  RECORDED_CURATION_POLICY_NOT_CURRENT_ADMISSION: 'A recorded curation policy, not a current admission', POST_OBSERVED_DEVELOPMENT_NOT_INDEPENDENT_VALIDATION: 'Post-observed development, not independent validation',
  POST_OBSERVED_DEVELOPMENT: 'Post-observed development', CURRENT_UNIVERSE_NON_PIT: 'Current universe · membership is not point-in-time', CURRENT_SECTOR_SNAPSHOT_NON_PIT: 'Current sector snapshot · not point-in-time',
  CURRENT_UNIVERSE_RESEARCH_ONLY: 'Current-universe research only', CURRENT_ACTIVE_SURVIVORS: 'Current active survivors only', CURRENT_CLASSIFICATION_BACKFILLED: 'Current classification backfilled', NON_POINT_IN_TIME_RESEARCH: 'Not point-in-time research', SEALED_HOLDOUT_LOCKED: 'Sealed holdout locked', PREDICTED_RISK_AND_DECOMPOSITION_NOT_ADMITTED: 'Predicted risk and its decomposition are not admitted',
  NO_CAPACITY_OR_IMPACT_ESTIMATE: 'No capacity or impact estimate', NO_STRATEGY_ACTIVATION: 'No strategy is activated', NO_DATA_API_KEY: 'No data API key',
  DETERMINISTIC_LINEAR_METHOD_NOT_A_SEED_ENSEMBLE: 'Deterministic linear method · not a seed ensemble', DEVELOPMENT_EVALUATED: 'Development evaluated',
  POSITIVE_OOS_EVIDENCE: 'Positive out-of-sample evidence', MIXED_OOS_EVIDENCE: 'Mixed out-of-sample evidence', NO_DETECTABLE_EFFECT: 'No detectable effect', NEGATIVE_OOS_EVIDENCE: 'Negative out-of-sample evidence', INSUFFICIENT_EVIDENCE: 'Insufficient evidence',
  // a factor's out-of-sample reason (oos_evidence.py): BY is the Benjamini–Yekutieli family, never the word "by"
  DIRECTIONALLY_POSITIVE_BY_CONFIRMED: 'Positive in direction · confirmed under Benjamini–Yekutieli FDR', DIRECTIONALLY_POSITIVE_NOT_BY_CONFIRMED: 'Positive in direction · not confirmed under Benjamini–Yekutieli FDR',
  DIRECTIONALLY_NEGATIVE_BY_CONFIRMED: 'Negative in direction · confirmed under Benjamini–Yekutieli FDR', NONPOSITIVE_DIRECTION_NOT_BY_CONFIRMED: 'Not positive in direction · not confirmed under Benjamini–Yekutieli FDR',
  MIXED_OOS_DIRECTION: 'Mixed direction out of sample', NONPOSITIVE_DIRECTION_NOT_HOLM_CONFIRMED: 'Not positive in direction · not confirmed under Holm',
  NO_FINDING_IN_SCOPE: 'No finding in scope', CONCERNS_NOT_ADJUDICATED: 'Concerns not adjudicated', NO_ADVERSE_ISSUE: 'No adverse issue', ISSUE_NOTED_WITHIN_LIMITS: 'Issue noted within limits', EVIDENCE_GAP: 'Evidence gap', HUMAN_REVIEW_REQUIRED: 'Human review required', ISSUE_STANDS: 'Issue stands',
  NO_MATERIAL_OBJECTION: 'No material objection', ACCEPT_WITH_LIMITS: 'Accept with limits', REQUEST_EVIDENCE_REFRESH: 'Request an evidence refresh', MATERIAL_OBJECTION: 'Material objection', RECONSIDER_CANDIDATE: 'Reconsider the candidate',
  COMPLETE: 'complete', PARTIAL: 'partial', UNIQUE_CURRENT: 'the one current analysis', DEVELOPMENT_RESULT: 'Development result', INSTALLED_RESULT: 'Installed result',
  research_experiment: 'Research experiment', study_verification_sweep: 'Verify saved studies', INVERSE_VOLATILITY_FROM_THE_LINKED_RISK_STUDY: 'Inverse volatility from the linked Risk study', COVARIANCE_FROM_THE_LINKED_RISK_STUDY: 'Covariance from the linked Risk study', ALREADY_PROMOTION: 'Already on the whole universe', 'research_lane.sample_scheme_superseded': 'Sampled under an earlier scheme', 'alpha_research.lifecycle_program_scheme_superseded': 'Sealed under an earlier scheme', 'risk_research.program_scheme_superseded': 'Sealed under an earlier scheme', 'portfolio_research.program_scheme_superseded': 'Sealed under an earlier scheme', 'task_control.queue_full': 'The Task queue is full', REFUSED_QUEUE_FULL: 'The Task queue is full', 'study_verification_sweep.not_admitted': 'Not admitted', 'research_input.manifest_missing': 'Its manifest is missing', 'research_input.bundle_unreadable': 'Its files do not read', workspace_data_update: 'Data update', workspace_prepare: 'Workspace preparation', workspace_preparation: 'Workspace preparation', evidence_review: 'Evidence review', alternative_evidence: 'Alternative evidence',
  'research_workspace.manifest_from_newer_build': 'Written by a newer build', // R13 (B13)
  'research_authoring.section_key_unknown': 'A key its section does not name', // U18 (V249)
  // U56 (EX): a formula factor's packet -- what activation adds and does, a trial's outcome and steps, the Alpha study's change, the refusals
  DAILY_CATALOG_ENTRY: 'a daily catalog entry', DAILY_PANEL_REBUILT_AT_NEXT_DATA_UPDATE: 'the daily Panel is rebuilt at the next data update', FEATURE_NOT_ADMITTED_BY_SCREENING: 'Not admitted by screening', NOT_COMPARED: 'Not compared: no change is claimed',
  FEATURE_BUILD: 'Feature build', FACTOR_STUDY: 'Factor study', CURATION: 'Curation', ALPHA_STUDY: 'Alpha study', PORTFOLIO_STUDY: 'Portfolio study',
  mean_rank_ic: 'Mean rank IC', pooled_oos_r2: 'Pooled out-of-sample R²', mean_gross_decile_spread: 'Mean gross decile spread', fold_coverage_mean: 'Mean fold coverage',
  'feature_extension.preprocessing_recipe_absent': 'No preprocessing recipe named', 'feature_extension.sector_leaf_research_only': 'A Sector-level research formula', 'feature_extension.preprocessing_not_admitted_for_active_panel': 'A research recipe',
  'feature_extension.human_confirmation_required': 'A person activates formula factors', 'feature_extension.contract_failed': 'The contract fails', 'feature_extension.trial_required': 'A completed trial is needed',
  'feature_extension.not_active': 'Not active in this workspace', 'feature_extension.shipped': 'Shipped with the product', 'feature_trial.record_damaged': 'A trial record no longer reads',
  // U50 (EX): a model's refusals and its contract's findings, in words; a finding's subject follows its words
  'model_extension.sandbox_required': 'A sandbox trial is needed', 'model_extension.contract_failed': 'The contract fails', 'model_extension.human_confirmation_required': 'A person activates models',
  'model_extension.installed': 'Installed with the product', 'model_extension.not_active': 'Not active in this workspace', CAPABILITY: 'a capability',
  'model_contract.route_mismatch': 'Its route differs from its declaration', 'model_contract.numerical_binding_route_invalid': 'Its binding route is invalid', 'model_contract.axis_point_refused': 'A search axis point is refused',
  'model_contract.fit_protocol_differs': 'Its fit protocol differs from its declaration', 'model_contract.nested_fit_not_probed': 'A nested fit was not probed', 'model_contract.estimator_not_deterministic': 'Its fits are not deterministic',
  'model_contract.predictions_not_deterministic': 'Its predictions are not deterministic', 'model_contract.predictions_malformed': 'Its predictions are malformed', 'model_contract.prediction_reads_other_rows': 'A prediction reads other rows',
  'model_contract.state_route_mismatch': 'Its fitted state names another route', 'model_contract.linear_state_incomplete': 'Its linear state is incomplete', 'model_contract.tree_state_incomplete': 'Its tree state is incomplete',
  'model_contract.import_outside_lock': 'An import outside the lock', 'model_contract.import_not_installed': 'An import not installed',
  // U57 (A7): the evidence service's reuse counters, since it started
  excerpt_reuses: 'Excerpts reused', canonical_reuses: 'Canonical texts reused', canonical_revisions_damaged: 'Canonical texts found damaged', selection_reuses: 'Selections reused', generation_releases: 'Index generations released', generation_builds_avoided: 'Index builds avoided',
  // U54: the artifacts a Task's owner reads back, and why one cannot be
  PortfolioResearchResult: 'Portfolio result', ResearchExecutionEvidence: 'Experiment evidence',
  'tasks.artifact_readback_absent': 'Its owner found no artifact', 'tasks.artifact_readback_failed': 'Its owner could not read it back',
  // U48 (V209): a backup's refusals and its last automatic attempt's failure, in words
  'workspace_backup.refused': 'The backup was refused', 'workspace_backup.object_changed_while_copied': 'A file changed while it was copied', 'workspace_backup.root_inside_workspace': 'The backup root is inside the workspace', 'workspace_backup.workspace_id_invalid': 'The workspace identity is invalid', 'workspace_backup.generation_invalid': 'A kept generation is invalid', 'workspace_backup.generations_kept_invalid': 'The kept count is invalid', 'workspace_backup.table_not_held': 'A held table is missing', 'workspace_backup.object_invalid': 'A stored object is invalid',
  LOW: 'low exposure', MEDIUM: 'medium exposure', HIGH: 'high exposure', CRITICAL: 'critical exposure',
  OPENED_OR_INCREASED_BY_POSITIVE_CHANGE: 'opened or increased · positive change', HELD_BY_ENDING_WEIGHT: 'held · ending weight', REDUCED_OR_EXITED_BY_ABSOLUTE_CHANGE: 'reduced or exited · absolute change',
  RECORDED: 'recorded package · no network', LIVE_OFFICIAL: 'live official sources', RECORDED_LOCAL_READ: 'local read', EXPLICITLY_ADMITTED_OFFICIAL_ACQUISITION: 'admitted official acquisition', SEC_EDGAR_OFFICIAL: 'SEC EDGAR (official)', MATERIAL_SECTIONS: 'material sections', FULL_FILING: 'full filing',
  cro_review_dossier: 'CRO review dossier', cro_review_receipt: 'CRO review receipt', cro_review_publication: 'CRO review publication', // the CRO Task's kept artifacts, as their locators name them
  // the eight topics (round E2; `EvidenceTopic`) and the reasons a cell is not complete (`incomplete[]`)
  LIQUIDITY_GOING_CONCERN: 'Liquidity & going concern', CAPITAL_DILUTION: 'Capital & dilution', LEGAL_REGULATORY: 'Legal & regulatory', OPERATIONS_SUPPLY: 'Operations & supply', PRODUCT_SAFETY_CYBER: 'Product safety & cyber', GOVERNANCE_CONTROLS: 'Governance & controls', COMMERCIAL_COUNTERPARTY: 'Commercial & counterparty', CORPORATE_ACTION_LISTING: 'Corporate action & listing',
  UNIT_NEEDS_UNREAD: 'Unit needs unread', TABLES_NOT_DEALT: 'Tables not dealt a view', TABLES_DELIVERED_IN_PART: 'Tables delivered in part', TABLES_PROGRESS_UNKNOWN: 'Tables of unknown progress', TABLES_UNRENDERABLE: 'Tables that cannot be rendered', TABLES_WITHOUT_ORIGINAL: 'Tables without an original', CANDIDATES_UNREAD: 'Candidates unread', RESIDUAL_SCOPE_QUEUED: 'Residual search queued', ROUTE_GAP: 'Route gap', SOURCE_GAP: 'Source gap',
  QUEUED: 'Queued', COVERED: 'Covered', BROADER: 'Broader search', NOTHING_RESUMABLE: 'Nothing resumable', REPRESENTATION_GAP: 'Representation gap', NONE: 'None',
  // a passage's codes (round E3): the method, the placement in an interval, the delivery, the comparison, the statement, the families, the time's basis and precision
  UNIT_WINDOW: 'Matter window', RESIDUAL_SEARCH: 'Residual search', TYPED_RULE: 'Typed rule', TABLE_VIEW: 'Table view', STRUCTURAL_SCAN: 'Structural scan',
  IN: 'In the interval', BOUNDARY: 'On the boundary', OUTSIDE: 'Outside the interval', UNKNOWN: 'Unknown', HISTORICAL_CONTEXT: 'Historical context', EXACT_REPEAT: 'Exact repeat',
  DELIVERED: 'Delivered', SERVED_BY_LATER_READING: 'Served by a later reading', WHOLE_VIEW: 'Whole view',
  FIRST_OBSERVED: 'First observed', UNCHANGED: 'Unchanged', CHANGED: 'Changed', RESTATED: 'Restated',
  SOURCE_UNAVAILABLE: 'Source unavailable', ABSENT: 'The company states none', PRESENT: 'Stated', AMBIGUOUS: 'Ambiguous', MIXED: 'Mixed', MITIGATING: 'Mitigating', FAVOURABLE: 'Favourable', NO_CONCLUSION: 'No conclusion', SCOPE_UNRESOLVED: 'Scope unresolved', UNREAD_RELEVANT_STATEMENT: 'A relevant statement unread',
  FINANCING: 'Financing', LITIGATION: 'Litigation', CORPORATE_EVENT: 'Corporate event', CUSTOMER_CONCENTRATION: 'Customer concentration', GOING_CONCERN: 'Going concern', SUPPLY: 'Supply',
  ACCEPTANCE: 'acceptance time', PUBLICATION: 'publication time', DATE: 'date', INSTANT: 'instant', RECORDED_IMPORT_CAPTURE_ACCEPTANCE_UNKNOWN: 'recorded import; acceptance unknown', OFFICIAL_ACCEPTANCE: 'official acceptance',
  // the review's codes (round E4): a position's transition, an assessment's disposition of a finding, the dossier's delivery
  NOT_EVALUATED: 'Not evaluated', IN_FLIGHT_RECOVERY: 'In-flight recovery', ACTIVE_LEASE: 'Active lease', USER_PINNED: 'Pinned by you', PROVED: 'Proved', REVIEWS_NOT_COMPARABLE: 'Not comparable', COMPARED: 'Compared', SOURCE_OBJECTS: 'Source objects', VECTOR_OBJECTS: 'Vector objects', INDEXES: 'Indexes', EXACT_HISTORICAL_READBACK: 'Exact historical readback',
  ONGOING: 'Ongoing', RESOLVED: 'Resolved', SUPERSEDED: 'Superseded', SUPPORTED: 'Supported', SINGLE_SOURCE: 'Single source', CONTESTED: 'Contested', UNSUPPORTED: 'Unsupported', REFRESH_EVIDENCE: 'Refresh evidence', RESOLVE_ISSUER_MAPPING: 'Resolve the issuer mapping', HUMAN_REVIEW: 'Human review', DO_NOT_ACTIVATE: 'Do not activate', OPENED: 'Opened', INCREASED: 'Increased', HELD: 'Held', REDUCED: 'Reduced', EXITED: 'Exited', MATERIAL_ISSUE: 'Material issue', NOT_MATERIAL: 'Not material', RESOLVED_WITH_EVIDENCE: 'Resolved with evidence', DEFERRED: 'Deferred', WHOLE_DOSSIER: 'Whole dossier', SEC_EDGAR_OFFICIAL: 'SEC EDGAR (official)',
  UNKNOWN_PROGRESS: 'Unknown progress', SESSION_LIMIT: 'Session limit', WINDOW_LIMIT: 'Window limit', ADVERSE: 'adverse', MATTER: 'matter', STATE: 'state', STRUCTURE: 'structure', TYPED: 'typed',
  'alternative_evidence.minimum_entity_coverage_not_met': 'Minimum issuer coverage not met', REQUIRED_BASELINE_MISSING: 'Required baseline filing missing', 'alternative_evidence.sec_response_too_large': 'Body over the document cap', // the codes a unit or a resource fails by (round E1)
  EXACT_HISTORICAL_READBACK: 'Exact historical readback', REVIEW_PUBLISHED: 'Review published', CRO_DOSSIER_READY: 'Dossier ready', EVIDENCE_ANALYST_PACKET_READY: 'Packet ready', NOT_AUTHORIZED: 'not authorized', AUTHORIZED: 'authorized', HUMAN: 'human', OFFICIAL_CLOSE: 'official close',
  EXPERIMENT_PLAN: 'Experiment PLAN', EXPERIMENT_RUN: 'Experiment RUN', EXPERIMENT_REPLAY: 'Experiment replay', EXPERIMENT_CURATE: 'Curation decision saved', EXPERIMENT_FOUNDATION_PREVIEW: 'Foundation preview', EXPERIMENT_FOUNDATION_SEAL: 'Foundation sealed',
  EXPERIMENT_LINK_RISK: 'Risk report linked', EXPERIMENT_PORTFOLIO_DRAFT: 'Portfolio draft', EXPERIMENT_DELIVERY_EXPORT: 'Delivery export', EXPERIMENT_EXPORT: 'Experiment export', EXPERIMENT_RISK_EXPORT: 'Risk export', EXPERIMENT_FOUNDATION_EXPORT: 'Foundation export',
  EVIDENCE_PREVIEW: 'Evidence preview', EVIDENCE_PREPARE: 'Evidence preparation', EVIDENCE_PACKET: 'Evidence packet read', EVIDENCE_ANALYSIS_SUBMIT: 'Analyst answer submitted', EVIDENCE_CRO: 'CRO assessment', EVIDENCE_CRO_EXPORT: 'Review export', EVIDENCE_REFRESH: 'Evidence refresh (Provider)', EVIDENCE_SELECT: 'Analysis selected',
  RESEARCH_INPUT_PLAN: 'Input publication PLAN', RESEARCH_INPUT_CONFIRM: 'Input publication confirmed', WORKSPACE_PREPARE_PLAN: 'Preparation PLAN', WORKSPACE_PREPARE_CONFIRM: 'Preparation confirmed', DATA_UPDATE_PLAN: 'Data update PLAN', DATA_UPDATE_RUN: 'Data update RUN',
  CASE_SAVE: 'Research case saved', CANCEL: 'Cancel requested', RECOVER: 'Resume requested', PLAN: 'PLAN', RUN: 'RUN',
  RESEARCH_PREPARATION: 'Research preparation', DEVELOPMENT_REPLAY: 'Development replay', DISCOVERY_ONLY_SELECTED_OPERATIONS_REVALIDATE: 'Discovery only · selected operations re-validate',
  INITIALIZATION_REQUIRED: 'initialization required', FEATURE_BUILDING: 'Features being built', RESEARCH_READY: 'research-ready', MANIFEST_UPDATE_PENDING: 'membership transition pending', ONBOARDING_IN_PROGRESS: 'onboarding in progress',
  SOURCE_VERIFICATION_BLOCKED: 'source verification blocked', SOURCE_CHECK_REQUIRED: 'source check required', LOCAL_DATA_PRESENT: 'local data present · no research input',
  // a recorded fold's metrics (round 86): the words the Alpha tables use
  mae: 'MAE', mse: 'MSE', rank_ic: 'Rank IC', zero_relative_oos_r2: 'OOS R² (vs zero)', gross_decile_spread: 'Decile spread (gross, fraction)', formation_decile_turnover: 'Formation decile turnover',
  comparison_row_count: 'Comparison row count', scored_comparison_row_count: 'Scored comparison row count', coverage: 'Coverage', fold_index: 'Fold index', metrics_hash: 'Metrics hash',
  // a research case's budget and stages (round 86)
  maximum_tasks: 'Maximum Tasks', maximum_numerical_calls: 'Maximum numerical calls', maximum_model_calls: 'Maximum model calls',
  DATA_FEATURES: 'Data & Features', FACTOR_FOUNDATION: 'Factor foundation', ALPHA: 'Alpha', RISK: 'Risk', PORTFOLIO: 'Portfolio', EVIDENCE_CRO: 'Evidence & CRO',
};
/* Round 63: the failures and next actions the record carries, in words; a state's word is the
 * state table's (round 71). An undeclared word is named as absent; the raw code stays
 * in the exact observations and the tip, never turned into a guessed sentence. */
Object.assign(CODE_WORDS, {
  insufficient_research_history: 'Insufficient research history',
  provider_returned_non_session_dates: 'Provider returned non-session dates',
  EVIDENCE_PREREQUISITES_MISSING: 'Evidence prerequisites missing', 'activity.failure_detail_withheld': 'Detail withheld',
  'portfolio_application.pipeline_manifest_unreadable': 'Saved result index unavailable',
  'workspace_backup.generation_unreadable': 'Backup record unverified',
  'model_extension.review_unavailable': 'Model review unavailable',
  'task_control.projection_unavailable': 'Task projection unavailable',
  'task_runtime.liveness_stale': 'Task heartbeat not recent',
  'task_runtime.host_visibility_missing': 'Task runner heartbeat not observed',
  'task_runtime.work_stalled': 'Task work may be stalled',
  'task_runtime.recovery_required': 'Task awaiting recovery',
  'task_runtime.parked': 'Queued Task without a driving command',
  'alternative_evidence.artifact_missing': 'Evidence artifact missing',
  'alternative_evidence.artifact_tampered': 'Evidence artifact identity differs',
  'alternative_evidence.artifact_unavailable': 'Evidence artifact unavailable',
  'task_control.task_not_found': 'Task record missing',
  'storage.cap_setting_invalid': 'Storage cap invalid', 'storage.cap_setting_unreadable': 'Storage cap unreadable',
  // round 90: the limitations a curation decision acknowledges, said as the acknowledgement
  ALPHA_SCIENTIFIC_STOP_PRESERVED: 'The Alpha scientific stop is preserved', SEALED_HOLDOUT_UNREAD: 'The sealed holdout stays unread', RISK_RESEARCH_NOT_ADMITTED: 'Risk research is not admitted',
  ROLLING: 'rolling', EXPANDING: 'expanding', DAILY: 'daily',
  // N6 (law 80): the data owner's stops and a unit's refusals, said in words on every data state
  'data.truth_review_required': 'A data decision is needed', 'data.remediation_wait': 'Waiting, as decided', 'workspace_data_update.retry_not_due': 'The retry time has not come', 'data.fetch_failed': 'The fetch failed',
  UNEXPLAINED_RAW_MOVE: 'An unexplained price move', 'data.unexplained_raw_move': 'Unexplained move', // F2: a data issue's kind, its name's second part
});
/* The owners' refusal codes, each said in a sentence beside the code (the review desk's since
 * round 15; shared since round 77): `explainCode` finds the code inside an owner's message. */
const CODE_LINES = {
    'storage.cap_setting_invalid':'The storage cap must be auto or a positive whole byte count. Set it again (`storage set --cap-bytes auto`).',
    'storage.cap_setting_unreadable':'The workspace storage cap setting could not be read. Set it again through Settings or `storage set --cap-bytes auto`; retained records stay intact.',
    // U56 (U28): a damaged trial record, and why the daily Panel does not admit a formula -- the Host's words, its static form
    'feature_trial.record_damaged':'This trial\'s record in the workspace no longer reads: it was changed or cut short. Start the trial again with its feature plan and study; the damaged record is kept aside and every step that did not change is reused.',
    'feature_extension.preprocessing_not_admitted_for_active_panel':'The daily Panel preprocesses with ROBUST_SECTOR_NEUTRAL_Z alone; this recipe is a research recipe. Declare the factor again with that recipe to activate it.',
    'feature_extension.preprocessing_recipe_absent':'The research plan names no preprocessing recipe for this factor, so the daily Panel cannot admit it. Declare it again with a recipe to activate it.',
    'feature_extension.sector_leaf_research_only':'The daily build carries no Sector level yet, so a Sector-level formula stays a research factor: it can be tried, not activated.',
    'feature_extension.human_confirmation_required':'A person activates or deactivates a formula factor, in the Workbench; an agent declares, builds and tries it, and reads its review packet.',
    'local_web.operation_route_absent':'This Host names no route for the operation: nothing was sent. Reload the page; a Host of another version may not offer it.', // U13
    'tasks.cursor_moved_reload':'The Task list of this session moved since its first page was read: read it again from the first page.', // U54
    'local_web.service_unreachable':'The service did not answer: nothing was read or sent. It may be stopped or restarting; read again once it answers.',
    'local_web.service_answer_unreadable':'The service answered with something other than its JSON: nothing is inferred from it. Read again once it answers.',
    'product_host.evidence_review_not_configured':'This workspace admits no evidence review authority (issuer registry, listing authority, recorded source package). The book stays readable and can be delivered without a review section; admitting an authority is a workspace setup step, not a page action.',
    'REFUSED_NO_BOOK_TO_REVIEW':'No book is named, and this workspace has no default one: choose a sealed book on Books, or open one from its Portfolio page.', // U20: the Host's words (V242), true of a workspace with books too
    'product_host.evidence_review_experiment_selector_invalid':'The book selector is incomplete: an authored book is named by its Task, its receipt hash and its holdings date together. Open the book from its Portfolio page (Review evidence) or choose it above.',
    'REFUSED_NO_ADMITTED_EVIDENCE_AUTHORITY':'No issuer registry and listing authority are admitted for this workspace; coverage cannot be computed and nothing is invented.',
    // round E1: the refusals a preparation names, each with the recovery its owner names (the storage keys before the Task's, so a Task failure that carries one reads as that)
    'storage.managed_capacity_exceeded':'Managed storage exceeds the workspace cap. Raise the cap in Settings or plan a cleanup.',
    'storage.disk_space_insufficient':'The disk has too little free space for the preparation\'s writes, so nothing of it was placed. Free space on the disk or through the storage cleanup preview, then prepare again.',
    'alternative_evidence.preparation_superseded':'This packet was prepared under a contract this workspace has since superseded: what it read stays readable by its handles; its reading plan is not continued. Read the source preview again and prepare under the current contract.',
    'alternative_evidence.brief_source_stale':'The packet\'s cutoff window has expired, so more of its source cannot be read under it. Read the source preview again to prepare as of a new cutoff; the old packet stays readable.',
    'REFUSED_PREPARATION_EXPIRED':'The packet\'s cutoff window has expired. Read the source preview again to prepare as of a new cutoff; the old packet stays readable.',
    'REFUSED_PREPARATION_SUPERSEDED':'The packet was prepared under a contract this workspace has since superseded; what it read stays readable. Read the source preview again and prepare under the current contract.',
    'REFUSED_PREPARATION_BINDING_CHANGED':'The workspace\'s evidence authority changed under the captured intent, so nothing was prepared. Read the source preview again for the current binding.',
    'REFUSED_ACQUISITION_WINDOW_CLOSED':'The acquisition window of this request has closed, so nothing was fetched. Read the source preview again for a new window.',
    'alternative_evidence.minimum_entity_coverage_not_met':'Too few of the unit\'s issuers hold an admitted document for the admitted floor, so the unit was refused whole and is reported as not reviewed by that name; the other units are unaffected.',
    'REQUIRED_BASELINE_MISSING':'A required baseline filing is not held for this issuer under the admitted cap, so the issuer fails by that name; its siblings are unaffected. An oversize body is checked again after its recheck date or under a larger declared cap.',
    'alternative_evidence.sec_response_too_large':'The body exceeds the admitted document cap and was not transferred; it is deferred by name until its recheck date or a larger declared cap.',
    'alternative_evidence.task_failed':'The preparation Task failed and its owner recorded the code; Tasks holds the recovery actions.',
    'REFUSED_MODEL_AUTHORITY_NOT_ADMITTED':'The product runs no model of its own: an agent answers the Analyst\'s packet and the CRO\'s dossier through its bundle. Saved evidence and reviews stay readable.', // U15: no model of the product's own since AG2
    'REFUSED_NO_ADMITTED_EVIDENCE_RUNTIME':'No evidence runtime is admitted, so there is nothing to prepare sources with.',
    'alternative_evidence.external_analysis_binding_changed':'The answer was bound to a different packet: the prepared Task or its packet changed since the answer was written. Read the packet again and answer that one; nothing was recorded.',
    'unknown_span_reference':'A cited span handle is not among the packet\'s admitted passages. Only listed span handles may be cited; nothing was recorded.',
    'brief_submission_authority_invalid':'The structured answer does not satisfy the packet\'s citation and schema rules; the exact field is named in the code.',
    'alternative_evidence.prepared_book_scope_mismatch':'That prepared Task belongs to another book or scope; choose a Task prepared for this exact book.',
    'chief_risk_officer.external_review_binding_changed':'The assessment was bound to a different dossier, policy or schema than the current one; read the dossier again and assess that one.',
    'chief_risk_officer.submission_citation_invalid':'A cited finding handle is not in this dossier; the allowed handles are listed beside the field.',
    'alternative_evidence.submission_bytes_exceeded':'The Analyst answer is larger than its owner accepts; the code gives its size and the bound in bytes, counted as compact JSON. Nothing was recorded; shorten the answer and submit it again.',
    'chief_risk_officer.submission_bytes_exceeded':'The CRO assessment is larger than its owner accepts; the code gives its size and the bound in bytes, counted as compact JSON. Nothing was recorded; shorten the assessment and submit it again.',
    'local_web.body_too_large':'The request is larger than the service reads; the code gives the bound in bytes. Nothing was read or recorded.',
    'chief_risk_officer.submission_entity_invalid':'An affected entity is not among the dossier\'s issuers; the allowed issuers are listed beside the field.',
    'external_submission_entry_required':'Only a person or an external automation may submit this; the product never submits an answer to itself.',
    'local_web.session_renewed':'The request was sent and refused before dispatch: its session token was not accepted and nothing was admitted. The session was renewed; confirm again to send it.',
    'local_web.session_renewal_failed':'The request was sent and refused before dispatch: its session token was not accepted and nothing was admitted. Renewing the session failed; reload the page for a new session, then confirm again.',
    'feature.baseline_qualified_population_insufficient':'Too few listings passed the quality floor for a baseline Feature panel; the universe is below the population the owner requires, so nothing was built.', // round 93: the first-use fixture's stop, said in words beside the owner's detail
    'workspace_backup.root_inside_workspace':'The backup root lies inside the workspace, so no generation was made. Name a root outside it with ALPHALATTICE_BACKUP_ROOT, then back up again.',
    'workspace_backup.object_changed_while_copied':'A held file changed while it was copied, so no generation was kept. Back up again once the running work settles.',
    'research_workspace.manifest_from_newer_build':'A newer build of the product wrote this workspace: its manifest holds what this build does not know, named after the code. Nothing was changed.',
    'research_workspace.manifest_unreadable':'The workspace\'s manifest could not be read, so what this workspace holds is not known here. Nothing was changed; the service log names the file.'}; // N6: a refusal the Home, Research inputs and Storage show, in words
Object.assign(CODE_LINES, {
  "portfolio_application.pipeline_manifest_unreadable": "The saved result index could not be verified. Its Task and result contents are unknown. Read workspace show and backup list; restore a generation that holds this index into a new directory, then read the result there.",
  "workspace_backup.generation_unreadable": "A backup-root record could not be verified. Its sealed identity, workspace association and suitability for restore are unknown. Read workspace show and backup list; restore an appropriate verified full generation hash with alphalattice backup restore --dir <new directory> --generation <verified generation hash> --workspace-id <workspace id> --root <backup root>. Do not select an unverified filename prefix.",
  "model_extension.review_unavailable": "This model's review packet could not be read. Repair its declaration, module, or required runtime, then run the model check command shown here.",
  "task_control.projection_unavailable": "The named Task record could not be projected. Its lifecycle, progress and artifacts are unknown. Read workspace show and backup list; restore a generation that holds this Task into a new directory before reading it there.",
  "alternative_evidence.artifact_missing": "The named evidence artifact is missing from this workspace. Inspect storage and backup list; restore a generation that holds this artifact into a new directory before reading it there.",
  "alternative_evidence.artifact_tampered": "The named evidence artifact does not verify against its recorded identity. Inspect storage and backup list; restore a verified copy into a new directory before reading it there.",
  "alternative_evidence.artifact_unavailable": "The named evidence artifact could not be read. Its contents are unknown. Inspect storage and backup list; restore a generation that holds it into a new directory before reading it there.",
  "task_control.task_not_found": "The named Task is not held by Task Control in this workspace. Read workspace show and backup list; restore a generation that holds it into a new directory, or follow the offered owner action to plan new work.",
});
const explainCode = (message) => { const s = String(message || ''); const code = Object.keys(CODE_LINES).find((k) => s.includes(k)); return code ? t(CODE_LINES[code]) : ''; };
/* U72 (V451, NM1b): what the product installs is named by its title from the one label table (labels.json, embedded by
 * the build as LABELS) -- a strategy, component, recipe, policy, risk method, target, objective or weight rule -- its
 * Chinese title under Chinese; the id stays where a person copies one (a code cell's tip and copy). */
const labelOf = (id) => (typeof LABELS !== 'undefined' && id !== null && id !== undefined && Object.prototype.hasOwnProperty.call(LABELS, String(id)) ? LABELS[String(id)] : null);
const labelWords = (label) => (typeof I18N !== 'undefined' && I18N.locale === 'zh-CN' ? label.title_zh : label.title);
const declaredCodeWord = (code) => CODE_WORDS[code] || STATES[String(code ?? '').toLowerCase()]?.word || stageWord(code)?.word || '';
const codeWords = (code) => {
  if (code === null || code === undefined || code === '') return '';
  if (labelOf(code)) return labelWords(labelOf(code));
  const word = declaredCodeWord(code);
  return word ? t(word) : typeof code === 'string' && /\s/.test(code) ? said(code) : t('Word not declared');
};
/* A feature by its name (WD2; the user, 2026-09-26: `RELATIVE_STOCK_CROSS_SECTION::dist_52w_high::current` read as
 * code): an id `family::name::variant` reads as its name in mono, a variant other than the current one after it,
 * the whole id on hover; a list shows its first `n` and how many more. */
const featureName = (id) => { const s = String(id ?? ''), [family, name, variant] = s.includes('::') ? s.split('::') : ['', s, '']; return html`<span class="mono" data-tip="${s}">${name || family}${variant && variant !== 'current' ? ` · ${variant}` : ''}</span>`; };
const featureList = (ids, n = 3) => { const list = ids || []; return html`${list.slice(0, n).map((id, i) => html`${i ? ', ' : ''}${featureName(id)}`)}${list.length > n ? ` +${list.length - n}` : ''}`; };
/* The actors (round 63): who did a thing, by who they are -- the person at this window is "You";
 * the agent through the automation API is the agent; a hook's producer names the host session
 * (Codex, Claude Code); a subagent keeps its retained name. */
const ACTORS = {HOST_FALLBACK: 'Host fallback', HUMAN: 'You', EXTERNAL_AUTOMATION: 'Agent · API', INSTALLED_AGENT: 'Installed Agent', SERVICE_AUTOMATION: 'Service', EXTERNAL_CLIENT: 'External client', TASK_CONTROL: 'Task Control'};
const PRODUCERS = {codex: 'Codex', 'codex-cli': 'Codex', 'openai-codex': 'Codex', 'claude-code': 'Claude Code', claude_code: 'Claude Code', claude: 'Claude Code'};
/* Round 90: an owner's attribution sentence (`Assessed by HUMAN: local-web-human.`) reads as the role;
 * the id stays on hover, and beside the words where the reader asked for facts (`o.facts`). */
const ATTRIBUTION_ROLES = {HUMAN: 'a human reviewer', EXTERNAL_AUTOMATION: 'an external automation', INSTALLED_AGENT: 'the installed Agent', SERVICE_AUTOMATION: 'the service'};
const assessedBy = (line) => { const m = String(line).match(/^Assessed by ([A-Z_]+): ([^.]+)\.?$/); return m ? {who: t('Assessed by {who}', {who: t(ATTRIBUTION_ROLES[m[1]] || 'Unidentified')}), id: m[2]} : null; };
function attributionWords(list, o = {}) {
  const lines = (Array.isArray(list) ? list : [list]).filter(Boolean);
  if (o.plain) return lines.map((line) => assessedBy(line)?.who || '').filter(Boolean).join(' · '); // the actor's words alone, for a sentence (round F5)
  const parts = lines.map((line) => {
    const a = assessedBy(line);
    if (!a) return html`${said(line)}`;
    return o.facts ? html`${a.who} · <span class="mono">${a.id}</span>` : html`<span class="coded" data-tip="${a.id}">${a.who}</span>`;
  });
  return parts.length ? parts.reduce((acc, p, i) => html`${acc}${i ? ' · ' : ''}${p}`, html``) : '';
}
const actorWords = (caller, producer = '') => producer && PRODUCERS[String(producer).toLowerCase()] ? PRODUCERS[String(producer).toLowerCase()] : ACTORS[caller] ? t(ACTORS[caller]) : t('Unidentified');
/* A declared method, target or family in words; an identifier without words stays as declared. */
const METHOD_WORDS = { // what the label table names (U72: the risk methods, the weight rules) is its; these are the rest
  DIAGONAL_SHRUNK_CORRELATION: 'diagonally shrunk correlation', SECTOR_FACTOR_IDIOSYNCRATIC_EWMA_RESCALED: 'sector factor model · EWMA-rescaled idiosyncratic risk',
  CROSS_SECTIONAL_TOTAL_RETURN_STD_Z: 'cross-sectional total return · standardized z', CROSS_SECTIONAL_STD_Z: 'cross-sectional standardized z',
  SECTOR_RESIDUAL_CROSS_SECTIONAL_STD_Z: 'sector-residual return · standardized z', SECTOR_RESIDUAL_RANK_GAUSS: 'sector-residual return · rank-Gaussian', SECTOR_RESIDUAL_ROBUST_Z: 'sector-residual return · robust z',
  ridge: 'Ridge', lasso: 'Lasso', elastic_net: 'Elastic net',
  dynamic_panel_lightgbm: 'LightGBM', regularized_linear: 'Regularized linear model',
  BENJAMINI_YEKUTIELI_FDR: 'Benjamini–Yekutieli FDR', ABSOLUTE_CORRELATION_CLUSTER: 'absolute-correlation cluster',
  // a Portfolio study's weight rule and catalog policy (U41, U42): its book's name, its facts and the Lab's choices
  TOP_K_EQUAL_WEIGHT: 'top names, equal weight', TOP_K_MINIMUM_VARIANCE: 'top names, minimum variance', TOP_K_SCORE_RISK_COST: 'top names, score against risk and cost', SECTOR_DEVIATION_PENALTY: 'top names, sector deviation penalty',
  quarantine_listings: 'quarantine affected names', require_complete: 'require complete returns',
};
const methodWords = (token) => labelOf(token) ? labelWords(labelOf(token)) : METHOD_WORDS[token] ? t(METHOD_WORDS[token]) : String(token ?? '');
/* The words with the code on hover: a reader who needs the exact token still has it. */
// A code the catalog knows is said in words; a sentence the owner already wrote stays words; only a bare token is shown as one.
const coded = (code) => html`<span class="coded" data-tip="${code ?? ''}">${codeWords(code)}</span>`;
/* What a flow needs before it runs (U60, V367): the owner's sentence, the completed results it reads on its input (newest
 * first, at most five of each, each opening its page), what is missing, and the owner's next requests as the ways this
 * page already has. `input` is the input the block is for ({id, binding}), which a handoff's draft names. */
const PREREQ_RESULTS = {FACTOR_STUDY: 'Factor studies', ALPHA_STUDY: 'Alpha studies', RISK_STUDY: 'Risk studies', BOOK: 'Books'};
const PREREQ_MISSING = {CURATED_FACTOR_STUDY: 'a curated Factor study', ALPHA_STUDY: 'a completed Alpha study', BOOK: 'a completed book'};
const PREREQ_KINDS = {'factor.screening-development': 'Start a Factor study on this input', 'risk.covariance-development': 'Start a Risk study on this input'};
function prerequisiteWays(next, input = null) {
  const ways = Object.values(next || {}).map((r) => {
    if (r.operation === 'MODEL_EXTENSIONS') return link(t('Models'), 'models', 'button compact');
    if (r.operation === 'EXPERIMENT_CONTROLS' && PREREQ_KINDS[r.experiment_kind]) return link(t(PREREQ_KINDS[r.experiment_kind]), 'lab', 'button compact', {experiment_kind: r.experiment_kind, research_input: r.research_input_id, input_binding: r.input_binding_hash, origin: '', plan: '', draft_source: ''});
    if (r.operation === 'EXPERIMENT_CURATION') return btn(t('Curate the Factor study'), 'task-result', r.task_id, 'button compact');
    if (r.operation === 'EXPERIMENT_HANDOFF_PREVIEW') return r.curation_receipt_hash && input?.id && input?.binding
      ? link(t('Draft the Alpha study from its curation'), 'lab', 'button compact', {draft_source: JSON.stringify({kind: 'factor', task_id: r.task_id, curation_receipt_hash: r.curation_receipt_hash, research_input_id: input.id, input_binding_hash: input.binding}), origin: '', plan: ''})
      : btn(t('Open the curated Factor study'), 'task-result', r.task_id, 'button compact'); // several curations are a choice, made on the study
    if (r.operation === 'EXPERIMENT_PORTFOLIO_DRAFT') return btn(t('Open the Alpha study to choose a candidate'), 'task-result', r.task_id, 'button compact');
    if (r.operation === 'EXPERIMENT_READBACK') return btn(t('Open the book'), 'task-result', r.task_id, 'button compact');
    if (['TASK_STATUS','STATUS','TASK_RECOVERY'].includes(r.operation) && r.task_id) return btn(t('Open the Task'), 'task', r.task_id, 'button compact');
    if (r.operation === 'RESEARCH_INPUTS') return link(t('Research inputs'), 'inputs', 'button compact');
    if (r.operation === 'WORKSPACE_SHOW') return link(t('Workspace'), 'overview', 'button compact');
    if (['STORAGE_CAP_SET','STORAGE_CAP_SHOW'].includes(r.operation)) return link(t('Storage cap'), 'settings', 'button compact', {row: 'storageCap'});
    if (['WORKSPACE_BACKUPS','STORAGE_READBACK'].includes(r.operation)) return link(t('Storage and backups'), 'storage', 'button compact');
    return ''; // EXPERIMENT_PLAN is the page's own primary
  }).filter(Boolean);
  return ways.length ? joinMarkup(ways, '') : '';
}
function prerequisitesPanel(pr, input = null) {
  if (!pr || typeof pr !== 'object' || !pr.flow) return '';
  const open = (row) => html`${btn(html`<span class="mono">${short(row.task_id)}</span>`, 'task-result', row.task_id, 'text-btn')}${row.curation_receipt_hashes?.length ? html` ${t('(curated)')}` : ''}`;
  const held = Object.entries(pr.present || {}).map(([result, rows]) => [t(PREREQ_RESULTS[result] || result), rows.length ? html`<span class="flow">${rows.map((r, i) => html`${i ? html`<span aria-hidden="true">·</span>` : ''}<span>${open(r)}</span>`)}</span>` : t('none on this input')]); // a row that wraps with its gap (TY4: the walk at 375); the dots part the studies, so a mark reads as its study's (U62)
  const missing = (pr.missing || []).length ? [[t('Missing'), pr.missing.map((m) => t(PREREQ_MISSING[m] || m)).join(', ')]] : [];
  const ways = prerequisiteWays(pr.next_requests, input);
  return panel(t('Before it runs'), html`<span class="owner-text">${t(pr.detail || '')}</span>`, html`${kv([...held, ...missing], 'kv-columns')}${ways ? html`<div class="flow">${ways}</div>` : ''}`, '', 'data-prerequisites="' + pr.flow + '"');
}
/* A result's standing (U59, V368): what it can claim, one statement a mark in the owner's order (the comparison first),
 * each in the owner's words; a mark an owner's code holds says the code in words where the statement names it. */
const STANDING_MARKS = [['comparison', 'Baseline comparison'], ['execution', 'Run'], ['contract', 'Contract'], ['evidence', 'Evidence'], ['activation', 'Activation']];
function standingPanel(st) {
  if (!st || !Array.isArray(st.statements) || st.statements.length !== STANDING_MARKS.length) return '';
  const said = (field, i) => {
    const s = String(st.statements[i]), code = st.reasons?.[field];
    if (!code || !s.includes(code)) return t(s);
    const [a, b = ''] = t(s.replace(code, '{reason}')).split('{reason}'); // the owner's template, its code in words
    return html`${a}${codedSubject(code)}${b}`;
  };
  return panel(t('Standing'), t('What this result can claim, each mark from its owner.'), kv(STANDING_MARKS.map(([field, word], i) => [field === 'comparison' ? hint(t(word), t('Whether the owner compared this result with a baseline result, the study without the change. A chart\'s benchmark is the market reference, a different thing.')) : t(word), html`<span data-standing="${field}:${st[field]}">${said(field, i)}</span>`]), 'kv-columns'));
}
// a code with its subject (`model_contract.import_outside_lock:numpy`): the code in words, the subject as written (U50, U56)
const codedSubject = (code) => { const s = String(code || ''), at = s.indexOf(':'); return at < 0 ? coded(s) : html`${coded(s.slice(0, at))} <span class="mono">${s.slice(at + 1)}</span>`; };
/* One tooltip (round 93; Radix's contract in Apple's dress). `data-tip` on the element, `data-tip-key` a
 * chord. Shown after 400 ms under the pointer, at once from one tipped element to the next within
 * 300 ms, at once on keyboard focus; gone on leave, blur, Esc, any press or a scroll. The tip is the
 * footnote on the overlay surface in the top layer (a popover, so it stands over a dialog), at most
 * 320 px, below and centred, above near the foot -- in the dock, beside it over the lane, level with
 * its row -- inside the viewport; `aria-describedby` names it
 * while it shows. The browser's `title` is not used: it is late, unstyled and absent in the pane. */
const Tip = (() => {
  let el = null, host = null, timer = 0, lastClose = 0, px = 0, py = 0;
  const node = () => { if (!el) { el = document.createElement('div'); el.id = 'tip'; el.setAttribute('role', 'tooltip'); el.setAttribute('popover', 'manual'); document.body.appendChild(el); } return el; };
  function show(target) {
    if (!target.isConnected) target = document.elementFromPoint(px, py)?.closest('[data-tip]'); // a repaint replaced the element under the pointer (a page that re-templates each second)
    let words = target?.dataset.tip; if (!words) return;
    if (words === '@overflow') { // a line cut with an ellipsis reads whole on hover (a prop never wraps): the part the line cuts, itself or its first cut child (a row's title)
      const cut = [target, ...target.children].find((el) => el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1); // cut by the line or by a line clamp
      if (!cut) return;
      words = cut.textContent.replace(/\s+/g, ' ').trim();
    }
    const tip = node();
    tip.innerHTML = html`${words}${target.dataset.tipKey ? html` ${keycap(target.dataset.tipKey)}` : ''}`;
    try { if (!tip.matches(':popover-open')) tip.showPopover(); } catch (e) { /* the top layer refused (an older engine); the tip still positions */ }
    host = target; target.setAttribute('aria-describedby', 'tip');
    tip.style.left = '0px'; tip.style.top = '0px'; // measured at the origin (round 95, the user's rail still: placed near the right edge it wrapped `Evidence` in two)
    // layout coordinates (round 95, the user's 125 % text size: the tips flew): the tip lives in the zoomed
    // document, so the target's rect and the viewport are read through the root zoom, as every popover's are
    const r = layoutRect(target), w = tip.offsetWidth, h = tip.offsetHeight, gap = param('popover-gap'), vw = viewW(), vh = viewH();
    const near = param('space-2');
    let top = r.bottom + gap; if (top + h > vh - near) top = r.top - gap - h;
    let left = Math.max(near, Math.min(vw - w - near, r.left + r.width / 2 - w / 2));
    // a tip in the dock stands beside it, over the lane, level with its row: under a row it covered
    // the next (the user, 2026-09-25); where the lane has no room it falls back under the row
    const side = target.closest('#side');
    if (side) {
      const d = layoutRect(side), beside = document.documentElement.dataset.dock === 'right' ? d.left - gap - w : d.right + gap;
      if (beside >= near && beside + w <= vw - near) { left = beside; top = Math.max(near, Math.min(vh - h - near, r.top + r.height / 2 - h / 2)); }
    }
    tip.style.top = top + 'px'; tip.style.left = left + 'px';
  }
  function hide() {
    clearTimeout(timer); timer = 0;
    if (host) { host.removeAttribute('aria-describedby'); host = null; lastClose = Date.now(); }
    if (el && el.matches(':popover-open')) el.hidePopover();
  }
  const tipped = (e) => (e.target instanceof Element ? e.target.closest('[data-tip]') : null);
  if (typeof document !== 'undefined' && typeof document.addEventListener === 'function') { // the harnesses read this file without a document (or with a stub)
    document.addEventListener('pointerover', (e) => { const target = tipped(e); px = e.clientX; py = e.clientY; if (!target || target === host) return; hide(); clearTimeout(timer); timer = setTimeout(() => show(target), Date.now() - lastClose < 300 ? 0 : 400); });
    document.addEventListener('pointerout', (e) => { const target = tipped(e); if (target && !(e.relatedTarget instanceof Node && target.contains(e.relatedTarget))) hide(); });
    document.addEventListener('focusin', (e) => { const target = tipped(e); if (target && document.body.dataset.input !== 'pointer') { hide(); show(target); } });
    document.addEventListener('focusout', (e) => { if (tipped(e)) hide(); });
    document.addEventListener('keydown', () => hide(), true);
    document.addEventListener('pointerdown', () => hide(), true);
    addEventListener('scroll', () => hide(), true);
  }
  return {hide, show};
})();
/* One number grammar. Figures stay in the surrounding weight; units are always the lighter
 * companion. A delta alone owns a sign, and its minus is the mathematical U+2212. */
const count = (n) => n == null || !Number.isFinite(Number(n)) ? '' : Number(n).toLocaleString('en-US', {maximumFractionDigits: 0});
/* A byte count in words (round E1): binary units, one decimal under 100, none at or over; the owner's figure, never a promise about space. */
const bytesWords = (n) => { if (n == null || !Number.isFinite(Number(n))) return ''; const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']; let x = Number(n), i = 0; while (x >= 1024 && i < units.length - 1) { x /= 1024; i += 1; } return (i ? (x >= 100 ? x.toFixed(0) : x.toFixed(1)) : String(Math.round(x))) + ' ' + units[i]; };
const unit = (text) => html`<span class="num-unit">${text}</span>`;
/* ---- the library's constants (Q2; the node harnesses read this section as it is) ---- */
/* An identity's shown length (Q2): a task, a candidate or a reference by its first 8, a content
 * hash by 12 -- 8, 12 and 16 were mixed for the same hash across pages. */
const SHORT = {id: 8, hash: 12};
/* A long list's page (Q2; law 92): the one size every paged list and table uses -- `table-rows` in
 * design/parameters.json (the user, 2026-09-24: 50 rows, never a table's own number). */
const LIST_PAGE = PARAMETER_VALUES['table-rows'];
/* A composition's legend (N6, the user's reading: dozens of sectors): every slice listed up to
 * `fold`; past it, the first `shown` and the rest as one closing row. */
const LEGEND = {fold: 24, shown: 16};
/* A lobby's group (law 136, the user's word: 6 rows a group): its first rows, then `Show n more`. */
const LOBBY = {shown: PARAMETER_VALUES['lobby-rows']}; // a group's rows before `Show n more` (design/parameters.json `counts`)
/* An identity's leading characters (round 79: one owner). A cut never splits a word (the Team review,
 * 2026-09-24: an agent id `child-unknown-demo` read `child-un`): at a word's end it stays; inside a hex
 * run it is a handle and stays where it is; inside a word, a readable id up to three handles long is kept
 * whole, a longer one to the end of that word. */
const short = (v, n = SHORT.id) => {
  const s = String(v || ''), sep = /[-_:./\s]/;
  if (s.length <= n) return s;
  let a = n, b = n;
  while (a > 0 && !sep.test(s[a - 1])) a--;
  while (b < s.length && !sep.test(s[b])) b++;
  if (b === n && a < n) return s.slice(0, n); // the cut ends a word: nothing is split
  if (/^[0-9a-f]+$/i.test(s.slice(a === n ? n : a, b))) return s.slice(0, n); // the word at the cut (or the one it starts) is hex: a handle
  return s.length <= 3 * n ? s : s.slice(0, b);
};
/* A parameter the scripts read (Q2): its px value from design/parameters.json (`scripts`). */
const param = (name) => PARAMETER_VALUES[name];
/* ---- end of the library's constants ---- */
const mono = (v, n = SHORT.hash) => v ? html`<span class="mono">${short(v, n)}</span>` : ''; // the same as a mono span; a missing identity is a dash
function signed(value, kind = 'percent', places = 2) {
  if (value == null || !Number.isFinite(Number(value))) return '';
  const v = Number(value), sign = v > 0 ? '+' : v < 0 ? '−' : '';
  const magnitude = Math.abs(v);
  if (kind === 'percent') return html`${sign}${num(magnitude, 'percent')}`;
  if (kind === 'pp') return html`${sign}${fmt(magnitude, places)} ${unit('pp')}`;
  if (kind === 'bps') return html`${sign}${count(magnitude)} ${unit('bps')}`;
  return sign + fmt(magnitude, places);
}
const pluralText = (n, one, many, args = {}) => t(Number(n) === 1 ? one : many, args); // a noun by its count (round 90: no `(s)` anywhere); the count is in the args or said before
const countText = (n, one, many) => pluralText(n, one, many, {n: count(n)});
/* Record grammar: the owner's exact date or instant, in UTC -- for Facts alone (law 133). */
function whenText(iso) {
  const s = String(iso || ''), m = s.match(/^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:\.\d+)?(Z|[+-]\d{2}:?\d{2})?$/);
  if (!m) return s || '';
  const zone = !m[3] ? '' : m[3] === 'Z' || m[3] === '+00:00' ? ' UTC' : ' UTC' + m[3];
  return m[1] + ' ' + m[2] + zone;
}
/* One clock (N3, law 133). An instant reads in the reader's zone -- one the owner wrote without a
 * zone is UTC, the owners' convention; a date alone is a date (a market session is never an
 * instant). `instantParts` gives the fields; `dayOf` the reader's day. */
const pad2 = (n) => String(n).padStart(2, '0');
function instantParts(iso) {
  const s = String(iso || ''), m = s.match(/^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$/);
  if (!m) return null;
  if (m[4]) { // an instant without its zone is UTC, the owners' convention (the data owner's naive `sources_checked_at` is its `observed_at`)
    const d = new Date(s.replace(' ', 'T') + (m[7] ? '' : 'Z'));
    if (!Number.isNaN(d.getTime())) return {year: d.getFullYear(), month: d.getMonth() + 1, day: d.getDate(), hh: pad2(d.getHours()), mm: pad2(d.getMinutes()), ss: pad2(d.getSeconds()), timed: true};
  }
  return {year: Number(m[1]), month: Number(m[2]), day: Number(m[3]), hh: m[4] || '', mm: m[5] || '', ss: m[6] || '', timed: Boolean(m[4])};
}
const dayOf = (iso) => { const p = instantParts(iso); return p ? `${p.year}-${pad2(p.month)}-${pad2(p.day)}` : ''; };
/* Reading grammar: an instant on a face, 24-hour in both locales -- the time alone on the reader's
 * own day, the date before it otherwise, the year only when it is not this year. */
function when(iso) {
  const p = instantParts(iso);
  if (!p) return String(iso || '') || '';
  const now = new Date(), current = now.getFullYear();
  const time = p.timed ? `${p.hh}:${p.mm}` : '';
  if (time && p.year === current && p.month === now.getMonth() + 1 && p.day === now.getDate()) return time;
  const at = time ? ' ' + time : '';
  if (I18N.zh) return `${p.year === current ? '' : p.year + '年'}${p.month}月${p.day}日${at}`;
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${months[p.month - 1]} ${p.day}${p.year === current ? '' : ', ' + p.year}${at}`;
}
/* Absence is contextual: list cells stay blank, facts and reports carry a dash, stats say why. */
const blank = (container = 'list', word = 'none') => container === 'list' ? '' : container === 'stat' ? t(word) : '';
/* ---- end of reading helpers ---- */
const pctFraction = (fraction) => fraction == null || !Number.isFinite(Number(fraction)) ? '' : pctNumber(Number(fraction) * 100);
/* A span of dates reads as two dates and a dash (every interval on every page); a move from one
 * value to another -- a cutoff before and after an update -- as two and an arrow. A line may break
 * at the separator, never inside a date. */
function dateRange(a, b) {
  return html`<span class="nowrap">${a || ''}</span> — <span class="nowrap">${b || ''}</span>`;
}
function dateMove(a, b) {
  return html`<span class="nowrap">${a || ''}</span> → <span class="nowrap">${b || ''}</span>`;
}
