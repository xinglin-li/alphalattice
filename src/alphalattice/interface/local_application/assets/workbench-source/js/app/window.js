/* The window (phase 9, round 62; phase 12, round 81; round 87): one owner for the frame -- the
 * dock (the one SIDEBAR table as one column at the left or the right, the person's Dock side,
 * the right by default beside a Codex or Claude conversation; expanded as the sidebar of words
 * or folded as the 48 px rail of the groups' glyphs -- the Navigation setting: `follow` folds
 * it below 1180, `sidebar` and `rail` are the two states and the top row's toggle switches
 * them; `top` is no column, the drawer behind the toggle),
 * the top row (the side's toggle at the side's end, back within the product, where you are, the
 * session's recent objects, the search, the appearance), and the inspector (one region with
 * modes -- peek, lens, Task, Facts, Record -- a docked column at 1180 and wider, a panel over
 * the lane below, a layer over the whole lane below 900; it opens from an object, never from a
 * toggle, round 86). Focus mode hides the side and the top row. The person navigates here; the
 * agent hands links; both reach every page. The product's own scale (Text size) lives here too:
 * the host's pane may not zoom. Round 86 (Codex's dock): the docked side's edge drags to set its
 * width and closes past the fold; the drawer peeks out under a pointer at the left edge. */
/* The sidebar's tree, in the groups of ROUTES. A row is a page; a group collapses by its head
 * (the state kept per viewer). A page has one word (2026-09-21, the user's reading: the rail said
 * `Factor screening`, the page's head `Factor research`, and the crumb's way up was not
 * recognised): `ROUTES[page][1]` owns it, and the rail, the crumb, the head and the title read it. */
const SIDEBAR = [
  {page: 'overview', word: 'Home', icon: 'overview', attention: () => LiveViews.needs()}, // what needs you (Home's Needs a decision)
  {page: 'goals', word: 'Goals', icon: 'flag'}, // U23: a section of its own
  {page: 'tasks', word: 'Tasks', icon: 'task', count: () => openTaskCount()},
  {page: 'data', word: 'Data', icon: 'data', attention: () => LiveViews.dataNeeds()}, // N5 (the user, 2026-09-24: 并且可以把data移动到Studies上面): one row, its pages the place's tabs
  {group: 'studies', word: 'Studies', icon: 'lab', pages: ['factor', 'feature-research', 'foundation', 'alpha', 'alpha-compare', 'models', 'risk', 'portfolio', 'compare']}, // N6: the research chain's order -- the risk model before the portfolio it sizes (the user, 2026-09-24: risk应该在portfolio上面)
  {group: 'evidence', word: 'Evidence', icon: 'evidence', pages: ['books', 'evidence', 'evidence-stream', 'evidence-reading', 'handoff', 'report']},
  {group: 'team', word: 'Team', icon: 'team', pages: ['team-sessions', 'team', 'team-participants', 'team-outputs', 'team-evidence']},
  {page: 'history', word: 'History', icon: 'history'},
];
const SIDEBAR_GROUPS = {};
for (const item of SIDEBAR) if (item.group) for (const page of item.pages) SIDEBAR_GROUPS[page] = item;
/* The places whose pages are tabs (PG7; laws 132
 * and 135 amended, the user 2026-09-24: 改成页内标签, 改成列表页的标签). An object's folders -- a book's five, a
 * session's two -- are its tabs while it is chosen (`param`); a list's views (Compare) and Data's pages are
 * tabs of the list's own pages, hidden where the page addresses an object, but for a view that keeps them
 * (`keep`). The dock stops at the list's row; the tabs are one row under the page's head; the path keeps
 * the tab. `also`: a page of the place without a tab of its own (a data issue, read on Data maintenance). */
const PAGE_TABS = {books: {tabs: ['evidence', 'evidence-stream', 'evidence-reading', 'handoff', 'report'], param: 'review_selector'},
  'team-sessions': {tabs: ['team', 'team-participants', 'team-outputs', 'team-evidence'], param: 'team'}, // U54: 产出 after Participants; U51: Participants between the conversation and what the product observed
  goals: {tabs: ['goal', 'goal-conversation', 'goal-results'], param: 'goal'}, // U23: a goal's folders
  factor: {tabs: ['factor', 'feature-research'], words: {factor: 'Experiments'}, remember: true}, // U56: the formulas a Factor study screens are its second tab, as Models is Alpha modeling's
  alpha: {tabs: ['alpha', 'alpha-compare', 'models'], keep: ['alpha-compare'], words: {alpha: 'Experiments'}, remember: true}, // U50: the models Alpha studies fit are Alpha modeling's third tab, not a step of the chain (the user, 2026-09-30) // the user, 2026-09-24: 下划线标题应该叫Experiments
  portfolio: {tabs: ['portfolio', 'compare'], keep: ['compare'], words: {portfolio: 'Experiments'}, remember: true},
  data: {tabs: ['data', 'inputs', 'storage'], also: ['issues'], keep: ['data'], remember: true}}; // Data maintenance names its update Task, still the tab
const FOLDER_OF = {};
for (const [list, set] of Object.entries(PAGE_TABS)) for (const page of [...set.tabs, ...(set.also || [])]) if (page !== list) FOLDER_OF[page] = list;
const OBJECT_FOLDERS = new Set(Object.keys(PAGE_TABS).filter((list) => PAGE_TABS[list].param)); // an object's folders, as against a list's views
const pageWord = (page) => ROUTES[page]?.[1] || page;
/* The Navigation setting (round 81; round 87: the sidebar and the rail are the dock's two states,
 * expanded and folded, on either side). `follow` picks by width. */
const NAV_MODES = [['follow', 'Follow the width'], ['sidebar', 'Sidebar'], ['rail', 'Rail'], ['top', 'Top row only']];
/* The Dock side (round 87, the user's decision): the right by default -- the window's edge beside
 * a Codex or Claude conversation (their own file panels stand there); the left for a window of its own. */
const DOCK_SIDES = [['right', 'Right'], ['left', 'Left']];
/* Text size (round 81): the product's own scale, for the pane that cannot zoom. */
const ZOOM_STEPS = [80, 90, 100, 110, 125, 150, 175];

const Window = (() => {
  const DOCK_WIDTH = BREAKPOINTS.window.dock, PANE_WIDTH = BREAKPOINTS.window.pane; // the ladder's steps (design/parameters.json)
  // the docked side's width (round 86): dragged between min and max, closed under the fold
  const SIDE_WIDTH = {min: PARAMETER_VALUES['side-min'], max: PARAMETER_VALUES['side-max'], fold: PARAMETER_VALUES['side-fold'], default: PARAMETER_VALUES['side-default'], step: PARAMETER_VALUES['space-4']};
  const S = {sideOverlay: false, sidePeek: false, override: null, wide: null, inspector: null, inspectorReturn: null, groups: null, layout: '', sidePainted: ''};
  const win = () => $('#window');
  const rest = () => $('#rest');
  const cmdK = () => (/Mac|iPhone|iPad/.test(navigator.platform) ? '⌘ K' : 'Ctrl K');

  /* ---- the placement ---- */
  const navMode = () => { const m = readPreference('navigation'); return NAV_MODES.some(([k]) => k === m) ? m : 'follow'; };
  const dockSide = () => (readPreference('dockSide') === 'left' ? 'left' : 'right');
  const dockRight = () => dockSide() === 'right';
  const wide = () => viewW() >= DOCK_WIDTH;
  // docked: the dock as a column (expanded, the words; folded, the rail of glyphs) at the dock's
  // side; drawer: no column, the side is a panel over the lane behind the top row's toggle
  const layout = () => (navMode() === 'top' ? 'drawer' : 'docked');
  const docked = () => layout() === 'docked';
  const column = docked; // the side is a column, not an overlay
  const railed = () => docked() && !sideOpen(); // folded: the rail
  const pane = () => matchMedia(`(width < ${PANE_WIDTH}px)`).matches; // the pane: the detail is a layer over the lane -- when the sheet draws it as one (its media query reads the viewport, not the text size's scale: at 125 % of 1100 the scaled 880 inerted a lane the sheet drew beside a floating detail)
  function setNavigation(mode) {
    if (!NAV_MODES.some(([k]) => k === mode)) return;
    savePreference('navigation', mode);
    S.sideOverlay = false; S.override = null;
    render();
    notify(t('Navigation: {mode}', {mode: t(NAV_MODES.find(([k]) => k === mode)[1])}));
    repaintSettings();
  }
  function setDockSide(side) {
    if (!DOCK_SIDES.some(([k]) => k === side)) return;
    savePreference('dockSide', side);
    S.sideOverlay = false;
    render();
    notify(t('Dock: {side}', {side: t(DOCK_SIDES.find(([k]) => k === side)[1])}));
    repaintSettings();
  }

  /* ---- the side ---- */
  // expanded (the words) or folded (the rail): `sidebar` and `rail` say it; `follow` reads the
  // width, with the toggle's word for this session (`S.override`) until the width class changes
  function sideOpen() {
    if (!docked()) return S.sideOverlay;
    const m = navMode();
    if (m === 'sidebar') return true;
    if (m === 'rail') return false;
    return S.override ? S.override === 'open' : wide();
  }
  // Studies' `+`: the menu of what the group makes new (law 145)
  function toggleNew() {
    const wrap = $('#side .side-group-new'), pop = wrap?.querySelector('.menu'), trigger = wrap?.querySelector('[data-action="side-new"]');
    if (pop) toggleMenu(pop, trigger, {root: wrap});
  }
  function groupOpen(key) {
    if (!S.groups) S.groups = readPreference('sideGroups') || {};
    return S.groups[key] !== false;
  }
  function toggleGroup(key) {
    if (!S.groups) S.groups = readPreference('sideGroups') || {};
    S.groups[key] = !groupOpen(key);
    savePreference('sideGroups', S.groups);
    // round 94: the head's state moves in place so the rows can slide (the side is not rebuilt);
    // the memo follows, so the next repaint leaves the side alone
    const head = $(`#side .side-group[data-group="${key}"] > .side-group-head`);
    if (!head) return renderSide();
    head.setAttribute('aria-expanded', String(groupOpen(key)));
    S.sidePainted = String(sideMarkup());
  }
  // round 94: the drawer leaves the way it came (its slide reversed over --dur-1), then the frame is written
  function hideOverlay() {
    S.sideOverlay = false; S.sidePeek = false;
    leave($('#side'), syncFrame);
  }
  function toggleSide(open) {
    const next = open === undefined ? !sideOpen() : Boolean(open);
    if (!docked() && !next) { hideOverlay(); renderTop(); return; }
    if (!docked()) { S.sideOverlay = next; cancelLeave($('#side')); }
    else if (navMode() === 'follow') S.override = next === wide() ? null : (next ? 'open' : 'closed');
    else savePreference('navigation', next ? 'sidebar' : 'rail'); // the toggle is the Settings row's own choice (round 87)
    S.sidePeek = false; // a deliberate open stays until a click outside; a peek would go with the pointer
    if (docked()) renderSide(); // the words or the glyphs
    syncFrame();
    renderTop(); // the search icon moves between the side's head and the top row (round 86)
    if (docked()) repaintSettings();
    if (next && !docked()) $('#side .side-row[aria-current="page"], #side .side-row')?.focus({preventScroll: true});
  }
  function closeSideOverlay() {
    if (!column() && S.sideOverlay) hideOverlay();
  }
  /* The docked side's width (round 86, Codex's and Claude's dock): the person drags its edge; a
   * drag under the fold closes it (the next open is the default width); a double-click on the
   * edge is the default; the arrow keys on the focused edge step it. */
  const clampWidth = (w) => Math.max(SIDE_WIDTH.min, Math.min(SIDE_WIDTH.max, Math.round(w)));
  function sideWidth() { const w = Number(readPreference('sideWidth')); return w >= SIDE_WIDTH.min && w <= SIDE_WIDTH.max ? w : SIDE_WIDTH.default; }
  function setSideWidth(w) { savePreference('sideWidth', w === SIDE_WIDTH.default ? null : clampWidth(w)); syncFrame(); }
  function resizeHandle() {
    return html`<div class="side-resize" role="separator" aria-orientation="vertical" tabindex="0" aria-label="${t('Resize navigation')}" data-tip="${t('Drag to resize; double-click for the default width')}" aria-valuemin="${SIDE_WIDTH.min}" aria-valuemax="${SIDE_WIDTH.max}" aria-valuenow="${sideWidth()}"></div>`;
  }
  function startResize(e) {
    const handle = e.target.closest('.side-resize');
    if (!handle || e.button !== 0 || !docked()) return;
    e.preventDefault();
    const z = rootZoom(), x0 = e.clientX / z, w0 = sideWidth(), w = win(), dir = dockRight() ? -1 : 1; // a right dock widens leftwards
    let width = w0, folded = false;
    try { handle.setPointerCapture(e.pointerId); } catch { /* an old engine: the document's listeners carry the drag */ }
    document.documentElement.classList.add('is-resizing');
    handle.classList.add('is-dragging');
    const move = (ev) => {
      width = w0 + dir * (ev.clientX / z - x0);
      folded = width < SIDE_WIDTH.fold;
      w.dataset.fold = folded ? 'true' : 'false'; // past the fold the dock dims: it will fold to the rail on release
      w.style.setProperty('--side-width', clampWidth(width) + 'px');
    };
    const end = () => {
      document.removeEventListener('pointermove', move); document.removeEventListener('pointerup', end); document.removeEventListener('pointercancel', end);
      document.documentElement.classList.remove('is-resizing');
      handle.classList.remove('is-dragging');
      delete w.dataset.fold;
      if (folded) { savePreference('sideWidth', null); toggleSide(false); }
      else setSideWidth(width);
      renderTop();
    };
    document.addEventListener('pointermove', move); document.addEventListener('pointerup', end); document.addEventListener('pointercancel', end);
  }
  function keyResize(e) {
    if (!e.target.classList?.contains('side-resize')) return false;
    const dir = dockRight() ? -1 : 1; // the arrows move the edge: left widens a right dock
    const by = {ArrowLeft: -SIDE_WIDTH.step * dir, ArrowRight: SIDE_WIDTH.step * dir, Home: SIDE_WIDTH.min - sideWidth(), End: SIDE_WIDTH.max - sideWidth()}[e.key];
    if (by === undefined) return false;
    e.preventDefault();
    setSideWidth(clampWidth(sideWidth() + by));
    return true;
  }
  /* The drawer peeks out (round 86, Codex's and VS Code's): a mouse held at the left edge slides
   * the side over the lane without taking focus; it goes back when the pointer leaves it, unless
   * a menu of the side is open. The toggle and Ctrl \ still open it to stay. */
  let peekTimer = null;
  const drawerClosed = () => document.documentElement.dataset.nav === 'drawer' && win()?.dataset.side !== 'open' && !document.body.classList.contains('focus-mode'); // read from the frame: this runs on every pointer move
  function peekSide(e) {
    if (S.sidePeek) { // a peeked side goes back once the pointer is 24 px away from it (a right drawer opens under the scrollbar, where `pointerleave` never comes)
      const r = layoutRect($('#side')), z = rootZoom(), x = e.clientX / z, y = e.clientY / z;
      if (x < r.left - 24 || x > r.right + 24 || y < r.top || y > r.bottom) unpeekSide();
      return;
    }
    const right = dockRight(), near = right ? e.clientX >= innerWidth - 40 : e.clientX <= 6; // the cheap test first: this runs on every move
    const atEdge = e.pointerType === 'mouse' && near && drawerClosed() && (right ? e.clientX >= document.documentElement.clientWidth * rootZoom() - 3 * rootZoom() : e.clientX <= 3 * rootZoom());
    if (atEdge && !peekTimer) peekTimer = setTimeout(() => { peekTimer = null; if (drawerClosed()) { S.sideOverlay = true; S.sidePeek = true; cancelLeave($('#side')); syncFrame(); } }, 150);
    else if (!atEdge && peekTimer) { clearTimeout(peekTimer); peekTimer = null; }
  }
  function unpeekSide() {
    if (S.sidePeek && !$('#side .menu:not([hidden]):not([data-leaving])')) hideOverlay();
  }
  function row(page, word, ic, extra = '') {
    // an object opened from another list (`via`: a version from Storage) stands under that list's row, as its path does (law 135)
    const via = viaPage();
    const current = (app.page === page && !via) || via === page || FOLDER_OF[via || app.page] === page || (page === 'settings' && app.page === 'advanced'); // a place's tab keeps its list's row current (N4-N6)
    // no tip: the word is the row; its chord shows at its end under the pointer or the focus, as a
    // property chip's key does, and the word gives way (the user, 2026-09-25: a tip under the row covered
    // the next; a tip beside the dock repeated a word the chord had cut)
    const chord = Controls.chordFor(page);
    return html`<a class="side-row" href="${entryUrl(page)}" data-page="${page}"${current ? ' aria-current="page"' : ''}>${ic ? icon(ic) : ''}<span>${t(word)}</span>${chord ? html`<small class="side-key" aria-hidden="true">${keycap(chord.replace(' then ', ' '))}</small>` : ''}${extra}</a>`;
  }
  /* The product menu (round 81, Codex's `Codex ⌄`): the mark and the name; Map, the shortcuts
   * sheet, Text size, About. The person's own things (Settings) are the workspace popover's. */
  function productMenu(cls = 'side-anchor') {
    const zoom = zoomLevel();
    // the language row (round 86, the user's shape): the word, then EN / 中文, the current one lit; a press switches and the frame repaints in the other language
    const language = html`<div class="menu-row menu-static" role="presentation">${icon('globe')}<span class="menu-word">${t('Language')}</span><span class="ui-segments menu-segments" role="group" aria-label="${t('Language')}">${segBtn('EN', 'locale-set', 'en', I18N.locale !== 'zh-CN', html``)}${segBtn('中文', 'locale-set', 'zh-CN', I18N.locale === 'zh-CN', html``)}</span></div>`;
    const rows = html`${menuRow({ic: 'fork', action: 'research-map', word: t('Map'), title: t('Open research map')})}${menuRow({ic: 'keyboard', action: 'shortcuts', word: t('Keyboard shortcuts'), note: '?'})}${language}<div class="menu-row menu-static" role="presentation">${icon('edit')}<span class="menu-word">${t('Text size')}</span><span class="menu-note zoom-note">${zoom} %</span>${btnAttrs('−', 'zoom-step', '-1', 'icon-btn zoom-step', html`data-stay aria-label="${t('Smaller text')}" data-tip="${t('Smaller text')}" data-tip-key="Ctrl −"`)}${btnAttrs('+', 'zoom-step', '1', 'icon-btn zoom-step', html`data-stay aria-label="${t('Larger text')}" data-tip="${t('Larger text')}" data-tip-key="Ctrl ="`)}</div>${menuRow({ic: 'info', action: 'about', word: t('About')})}`;
    return html`<span class="side-anchor-wrap" data-anchor="product">${btnAttrs(html`<span class="side-mark">${mark()}</span><span class="side-anchor-name">AlphaLattice</span>${icon('chevron')}`, 'product-menu', '', cls, html`aria-haspopup="menu" aria-expanded="false" aria-label="AlphaLattice" `)}<div class="menu menu-align-start" role="menu" aria-label="AlphaLattice" hidden>${rows}</div></span>`;
  }
  /* The gear (round 84, the user's word): Settings in one press at the dock's foot -- a row with the
   * word in the sidebar (Codex's), the glyph alone in the rail; never a circle around it. */
  const gear = (cls) => btnAttrs(icon('gear'), 'go', 'settings', cls, html`aria-label="${t('Settings')}" data-tip="${t('Settings')}" data-tip-key="g ,"${app.page === 'settings' || app.page === 'advanced' ? html` aria-current="page"` : ''}`);
  /* The workspace row (round 81, Codex's account row): the folder, the name, the popover. */
  function workspaceRow(name, shown, cls = 'side-anchor') {
    const tip = name && (cls.includes('rail-item') || shown !== name); // the rail's glyph, or a name the row shortens
    return html`<span class="side-anchor-wrap" data-anchor="workspace">${btnAttrs(html`${icon('cube')}<span class="side-anchor-name">${app.empty ? t('No workspace') : shown}</span>${icon('chevron')}`, 'workspace', '', cls, html`aria-haspopup="menu" aria-expanded="false"${tip ? html` data-tip="${name}"` : ''} aria-label="${t('Workspace: {name}', {name})}"`)}<div class="menu menu-popover" role="menu" aria-label="${t('Workspace')}" hidden></div></span>`;
  }
  // a side menu asked for while the side is hidden (the drawer closed, the column folded) shows the side first
  function sideMenu(anchor, action) {
    if (!docked() && !sideOpen()) toggleSide(true);
    const wrap = $(`#side [data-anchor="${anchor}"]`), pop = wrap?.querySelector('.menu'), trigger = wrap?.querySelector(`[data-action="${action}"]`);
    return pop ? {wrap, pop, trigger} : null;
  }
  function toggleWorkspace() {
    const m = sideMenu('workspace', 'workspace');
    if (!m) return false;
    const {wrap, pop, trigger} = m;
    if (pop.hidden) pop.innerHTML = LiveViews.workspacePopover();
    toggleMenu(pop, trigger, {root: wrap});
    return true;
  }
  function toggleProduct() {
    const m = sideMenu('product', 'product-menu');
    if (!m) return false;
    const {wrap, pop, trigger} = m;
    const note = pop.querySelector('.zoom-note'); if (note) note.textContent = zoomLevel() + ' %';
    toggleMenu(pop, trigger, {root: wrap});
    return true;
  }
  function toggleRailGroup(key) {
    const wrap = $(`#side .rail-group[data-group="${key}"]`), pop = wrap?.querySelector('.menu'), trigger = wrap?.querySelector('[data-action="rail-group"]');
    if (pop) toggleMenu(pop, trigger, {root: wrap});
  }
  // a poll repaints the side for its counts (round 62); while a menu of the side is open, the
  // person is reading it: the counts follow in place and the menu stays
  function syncCounts() {
    for (const item of SIDEBAR) if (item.count || item.attention) {
      const a = $(`#side [data-page="${item.page}"]`);
      if (!a) continue;
      for (const b of a.querySelectorAll(':scope > .num')) b.remove();
      a.insertAdjacentHTML('beforeend', badges(item, railed()));
    }
  }
  /* A row's numbers: what needs you, the warning counter, before what merely runs (neutral); the rail
   * counts only what needs you (the user, 2026-09-25: 侧栏数字分开; the rail's restraint). */
  function badges(item, rail = false) {
    const a = item.attention ? item.attention() : 0, n = item.count ? item.count() : 0, cls = rail ? 'rail-badge' : 'side-badge';
    const tip = countText(a, '{n} needs a decision', '{n} need a decision');
    const need = a ? html`<b class="${cls} num is-attention" data-tip="${tip}" aria-label="${tip}">${a}</b>` : '';
    return rail ? need : html`${need}${n ? html`<b class="${cls} num">${n}</b>` : ''}`;
  }
  /* The side is rebuilt only when its markup changed (round 94): a poll's repaint, a theme
   * change and a page render that moved nothing leave its rows, hover, focus and scroll where
   * they are; a menu open keeps the side and follows the counts in place. */
  function renderSide() {
    const side = $('#side');
    if (!side) return;
    document.title = titleText(); // the window's title leads with what needs you, with the dock's counts (seen from another tab too)
    if (side.querySelector('.menu:not([hidden]):not([data-leaving])')) return syncCounts(); // a leaving menu (round 94's exit) is not an open one: the language switch pressed inside the product menu repaints the side at once
    side.classList.toggle('side-folded', railed());
    const markup = String(sideMarkup()); // the tag returns a Markup; the memo compares its text
    if (markup === S.sidePainted) return;
    S.sidePainted = markup;
    // the dock keeps its own place: a page change repaints its rows (the current one moves), never its
    // scroll -- the content and the dock scroll apart (the user, 2026-09-24: 换页时 dock 跟着滚到顶端)
    const kept = side.querySelector('.side-tree, .rail-tree')?.scrollTop || 0;
    side.innerHTML = markup;
    const tree = side.querySelector('.side-tree, .rail-tree');
    if (tree && kept) tree.scrollTop = kept;
  }
  /* New work starts from its group (law 145; the user, 2026-09-24: 有现存studies的时候, 反倒找不到去哪里创建
   * new experiments了 -- and, on the first answer, a row above Home read as a place and its pencil as edit):
   * Studies' head carries `+`, a menu of what the group makes new -- a research question, each experiment
   * kind in the chain's order -- each opening its composer, the Lab preset to its kind. Folded, the rail's
   * Studies menu opens with the same rows. The rows carry no `data-page`: nothing re-addresses them. */
  const NEW_KINDS = [['factor.screening-development', 'factor'], ['alpha.model-development', 'alpha'], ['risk.covariance-development', 'risk'], ['portfolio.policy-development', 'portfolio']];
  const newRows = () => html`<p class="menu-section">${t('New')}</p>${NEW_KINDS.map(([kind, page]) => menuRow({ic: KIND_MARKS[kind]?.[0] || 'lab', word: t(pageWord(page)), href: routeUrl('lab', {experiment_kind: kind})}))}`;
  const newMenu = () => html`<span class="side-group-new">${btnAttrs(icon('plus'), 'side-new', '', 'icon-btn side-new-btn', html`aria-haspopup="menu" aria-expanded="false" aria-label="${t('New in Studies')}" data-tip="${t('New in Studies')}"`)}<div class="menu" role="menu" aria-label="${t('New in Studies')}" hidden>${newRows()}</div></span>`;
  function sideMarkup() {
    const name = Data.workspace();
    const shown = name.length <= 24 ? name : name.slice(0, 13) + '…' + name.slice(-7);
    if (railed()) {
      // the rail (round 81): the groups' glyphs, a group's pages as a menu beside its glyph
      const items = SIDEBAR.map((item) => {
        if (item.page) {
          const chord = Controls.chordFor(item.page);
          return html`<a class="rail-item" href="${entryUrl(item.page)}" data-page="${item.page}"${app.page === item.page || FOLDER_OF[app.page] === item.page ? ' aria-current="page"' : ''} data-tip="${t(item.word)}"${chord ? html` data-tip-key="${chord}"` : ''} aria-label="${t(item.word)}">${icon(item.icon)}${badges(item, true)}</a>`;
        }
        const current = SIDEBAR_GROUPS[app.page] === item;
        return html`<span class="rail-group" data-group="${item.group}">${btnAttrs(icon(item.icon), 'rail-group', item.group, 'rail-item', html`aria-haspopup="menu" aria-expanded="false" data-tip="${t(item.word)}" aria-label="${t(item.word)}"${current ? html` aria-current="page"` : ''}`)}<div class="menu" role="menu" aria-label="${t(item.word)}" hidden>${item.group === 'studies' ? newRows() : ''}<p class="menu-section">${t(item.word)}</p>${item.pages.filter((page) => !FOLDER_OF[page]).map((page) => menuRow({word: t(pageWord(page)), href: railUrl(page), page, checked: app.page === page || FOLDER_OF[app.page] === page}))}</div></span>`;
      });
      return html`<div class="rail-head">${productMenu('rail-item side-anchor')}</div><nav class="rail-tree" aria-label="${t('Pages')}">${items}</nav><div class="rail-foot">${gear('rail-item side-gear')}${workspaceRow(name, shown, 'rail-item side-anchor')}</div>`;
    }
    // round 84 (Codex's dock): the page rows before the first group stay under the head; the groups scroll
    const pageRow = (item) => row(item.page, item.word, item.icon, badges(item));
    const firstGroup = SIDEBAR.findIndex((item) => item.group);
    const fixed = SIDEBAR.slice(0, firstGroup).map(pageRow);
    const tree = SIDEBAR.slice(firstGroup).map((item) => {
      if (item.page) return pageRow(item);
      const open = groupOpen(item.group);
      return html`<section class="side-group" data-group="${item.group}"><button type="button" class="side-group-head" data-action="side-group" data-value="${item.group}" aria-expanded="${open}"><span>${t(item.word)}</span>${icon('chevron')}</button><div class="side-group-fold"><div class="side-rows">${item.pages.filter((page) => !FOLDER_OF[page]).map((page) => row(page, pageWord(page), ''))}</div></div>${item.group === 'studies' ? newMenu() : ''}</section>`; // the dock stops at the lists (N4)
    });
    return html`${docked() ? resizeHandle() : ''}<div class="side-head">${productMenu()}${btnAttrs(icon('search'), 'quick-open', '', 'icon-btn side-find', html`aria-label="${t('Quick Open, Control or Command K')}" data-tip="${t('Search')}" data-tip-key="${cmdK()}"`)}</div><nav class="side-fixed" aria-label="${t('Pages')}">${fixed}</nav><nav class="side-tree" aria-label="${t('Sections')}">${tree}</nav><div class="side-foot">${row('settings', 'Settings', 'gear')}${workspaceRow(name, shown)}</div>`;
  }

  /* ---- the top row: back, where you are, the session's objects, the search, the panels ---- */
  /* The path is the working directory (F3, law 135): the group, the list, the object, its folder --
   * `Evidence / Books / <book> / Sources`, `Studies / Factor screening / <study>`, `Data / Research
   * inputs / <family> / <version>`; a page outside the groups is its own word. The list is the way
   * up (a link) and the object its home (a link, but where it is the place itself), its name cut at
   * `crumb-object` and whole on hover; the inspector's layer is the last segment. */
  // the list an object was opened from, where it is not its home's (a version opened from Storage & retention)
  const viaPage = () => { const v = routeObjectId() ? hashParams().get('via') || '' : ''; return v && v !== app.page && (SIDEBAR_GROUPS[v] || FOLDER_OF[v]) ? v : ''; };
  // a place that remembers its last tab (law 132: Data, and a list with its comparison) opens there from its row; an object's tabs open at its home (law 135)
  const entryUrl = (page) => { const set = PAGE_TABS[page]; if (!set?.remember) return railUrl(page); const last = readPreference('tab:' + page); return routeUrl(set.tabs.includes(last) ? last : page); };
  // a data issue is read on Data maintenance: that is its list in the path (N5)
  const LIST_HOME = {issues: 'data'};
  function pathSegments() {
    const page = app.page, group = SIDEBAR_GROUPS[page], via = viaPage(), tabOf = FOLDER_OF[page];
    // a tab of a place: an object's folder always, a list's view or Data's page while no object is addressed
    const parent = tabOf && (OBJECT_FOLDERS.has(tabOf) || !routeObjectId() || PAGE_TABS[tabOf].keep?.includes(page)) ? tabOf : '';
    // a place that is a row of its own (Data) opens the path as a group does (N5)
    const place = !group && tabOf && !OBJECT_FOLDERS.has(tabOf) ? SIDEBAR.find((x) => x.page === tabOf) : null;
    const segs = group ? [{kind: 'dir', word: t(group.word)}] : place ? [{kind: 'list', word: t(place.word), href: entryUrl(place.page)}] : [];
    if (parent) {
      const scope = OBJECT_FOLDERS.has(parent) ? $('#main template.page-scope') : null;
      if (!place) segs.push({kind: 'list', word: t(pageWord(parent)), href: listUrl(parent)});
      if (scope) segs.push({kind: 'object', word: scope.content.textContent.trim(), href: scope.dataset.self === 'true' ? '' : scope.dataset.href || ''});
      segs.push({kind: 'page', word: t(pageWord(page))});
    } else {
      const word = group ? pageWord(page) : (SIDEBAR.find((x) => x.page === page)?.word || pageWord(page));
      // an object the address names (a study, a case, an issue, a version) is named by its page's head, the one owner of its name
      const head = routeObjectId() ? $('#main .object-header h1')?.textContent?.trim() || '' : '';
      if (head && head !== t(word)) {
        // the path follows the list the object was opened from (law 135: one home, two paths)
        const home = LIST_HOME[page];
        if (tabOf && !place && !via && !home) segs.push({kind: 'list', word: t(pageWord(tabOf)), href: listUrl(tabOf)}); // an object on a place's tab: its place first (U50: Alpha modeling / Models / a model)
        segs.push(via ? {kind: 'list', word: t(pageWord(via)), href: listUrl(via)} : home ? {kind: 'list', word: t(pageWord(home)), href: listUrl(home)} : {kind: 'list', word: t(word), href: listUrl(page)});
        const family = via ? '' : $('#main template.page-directory')?.content.textContent.trim(); // a version's family: the directory between its list and it
        if (family) segs.push({kind: 'dir', word: family});
        segs.push({kind: 'object', word: head});
      } else segs.push({kind: 'page', word: t(word)});
    }
    const layer = pane() && S.inspector ? inspectorTitle() : ''; // the pushed layer is the last segment
    if (layer) segs.push({kind: 'page', word: layer});
    return segs;
  }
  function whereMarkup() {
    const segs = pathSegments(), last = segs.length - 1;
    const seg = (s, i) => {
      const cls = `top-seg ${s.href && i !== last ? 'top-group' : 'top-page'}${s.kind === 'object' ? ' top-object' : ''}`;
      const tip = s.kind === 'object' ? ' data-tip="@overflow"' : '', current = i === last ? ' aria-current="page"' : '';
      return s.href && i !== last ? html`<a href="${s.href}" class="${cls}"${tip}>${s.word}</a>` : html`<span class="${cls}"${tip}${current}>${s.word}</span>`;
    };
    // narrow, the upper levels fold from the left into `…` (its menu holds them), the place stays (law 135)
    const fold = last ? html`<span class="top-fold-wrap">${btnAttrs('…', 'path-fold', '', 'top-fold', html`aria-haspopup="menu" aria-expanded="false" aria-label="${t('The folders above')}" data-tip="${t('The folders above')}"`)}<span class="top-sep" aria-hidden="true">/</span><div class="menu" role="menu" aria-label="${t('The folders above')}" hidden></div></span>` : '';
    return html`${fold}${segs.map((s, i) => html`${i ? html`<span class="top-sep" aria-hidden="true">/</span>` : ''}${seg(s, i)}`)}`;
  }
  function togglePathFold() {
    const wrap = $('#top .top-fold-wrap'), pop = wrap?.querySelector('.menu'), trigger = wrap?.querySelector('.top-fold');
    if (!pop) return false;
    if (pop.hidden) pop.innerHTML = [...$$('#top .top-where > .top-seg[data-folded]')].map((s) => (s.tagName === 'A' ? menuRow({word: s.textContent.trim(), href: s.getAttribute('href')}) : html`<p class="menu-section">${s.textContent.trim()}</p>`)).join('');
    toggleMenu(pop, trigger, {root: wrap});
    return true;
  }
  /* Recent (round 81): the pane's tabs -- the objects this session touched (what the agent
   * linked, what the person opened), from the Home's own recent list. */
  function recentMenu() {
    return html`<span class="top-recent-wrap">${btnAttrs(icon('clock'), 'recent-menu', '', 'icon-btn top-recent', html`aria-haspopup="menu" aria-expanded="false" aria-label="${t('Recent')}" data-tip="${t('Recent · the objects this session touched')}"`)}<div class="menu" role="menu" aria-label="${t('Recent')}" hidden></div></span>`;
  }
  function toggleRecent() {
    const wrap = $('#top .top-recent-wrap'), pop = wrap?.querySelector('.menu'), trigger = wrap?.querySelector('.top-recent');
    if (!pop) return false;
    if (pop.hidden) {
      const rows = Data.recent(8);
      pop.innerHTML = rows.length ? html`<p class="menu-section">${t('Recent')}</p>${rows.map((x) => menuRow({ic: Inspect.recordIcon(x), action: 'history-open', value: x.id, word: LiveViews.nameOf(x).name, note: x.recordedAt ? when(x.recordedAt) : ''}))}` : html`<p class="menu-empty">${t('No saved research yet.')}</p>`;
    }
    toggleMenu(pop, trigger, {root: wrap});
    return true;
  }
  function renderTop() {
    const top = $('#top');
    if (!top) return;
    const dark = effectiveTheme() === 'dark';
    // the toggle stands at the dock's end (round 86, the user's word; round 87: the dock's side), its pane the dock's
    const rail = dockRight();
    const toggle = btnAttrs(icon(rail ? 'panel-right' : 'panel'), 'side-toggle', '', 'icon-btn top-side-toggle', html`aria-label="${t(sideOpen() ? 'Hide navigation' : 'Show navigation')}" aria-expanded="${sideOpen()}" aria-controls="side" data-tip="${t('Navigation')}" data-tip-key="Ctrl \\"`);
    const find = docked() && sideOpen() ? '' : btnAttrs(icon('search'), 'quick-open', '', 'icon-btn top-find', html`aria-label="${t('Quick Open, Control or Command K')}" data-tip="${t('Search')}" data-tip-key="${cmdK()}"`);
    const appearance = btnAttrs(icon(dark ? 'sun' : 'moon'), 'appearance', '', 'icon-btn top-appearance', html`aria-label="${t(dark ? 'Switch to Light' : 'Switch to Dark')}" data-tip="${t(dark ? 'Switch to Light' : 'Switch to Dark')}"`);
    top.innerHTML = html`${rail ? '' : toggle}<nav class="top-where" aria-label="${t('Where you are')}">${whereMarkup()}</nav><span class="top-actions"></span>${recentMenu()}${find}${appearance}${rail ? toggle : ''}`; // N1 (law 124): no Back -- the browser's is the history; the path and the dock are the ways around
    syncPageTop(); // N1: a redrawn row takes the page's marks, menu and verbs again (an inspector's open or close redraws it)
  }

  /* ---- the detail (law 149): one layer, beside the list it details ---- */
  /* The window's detail (round 92's side column): a Task's record, a factor's or a fold's evidence, a
   * holding, a saved object, the page's Facts and Record -- one head (the kind line, the name, the
   * close; the panel's tabs when it has two) over its body. A list that hosts it (`detailSplit(list,
   * modes)`) shows it beside itself, the Evidence reading's way: that list gives way, the page's head
   * and its other sections keep their width. A detail no list hosts (the page's Facts or Record, a
   * saved object named from elsewhere) floats over the lane under the top row, at its right
   * (`#inspector`), and moves nothing. `by` is the press that opens it, [action, value]: the same press
   * closes it and its row carries the mark; a body changed from inside (a tab, a link in the body)
   * keeps the one before. */
  const hostOf = (mode) => $(`#main [data-detail-host~="${mode}"]`);
  function detailHeadMarkup(x = S.inspector) {
    if (!x) return '';
    const tabRow = x.tabs && x.tabs.length ? tabStrip(t('Panel'), x.tabs.map((y) => ({word: y.word, action: 'inspector-tab', value: y.key, on: y.on, cls: 'inspector-tab'})), 'inspector-tabs') : '';
    // the body between two marks: a repaint of the list beside it that changes nothing but the body
    // (painted in place, `setInspectorBody`) replaces nothing (the router compares without it)
    return html`<header class="reading-pane-head detail-head"><div class="detail-head-row"><div>${x.kind ? html`<p class="caption detail-kind">${x.kind}</p>` : ''}<h2 id="inspectorTitle">${x.title}</h2></div>${btnAttrs(icon('close'), 'inspector-close', '', 'icon-btn', html`aria-label="${t('Close')}" data-tip="${t('Close')}"`)}</div>${tabRow}</header>`;
  }
  function detailMarkup(x = S.inspector) {
    return x ? html`${backMarkup()}${detailHeadMarkup(x)}<div class="reading-pane-body"><!--detail-body-->${x.body}<!--/detail-body--></div>` : '';
  }
  // What a list's split draws beside the list while the window's detail is one of the modes it hosts.
  function detailPane(modes) {
    const x = S.inspector;
    if (!x || ![].concat(modes).includes(x.mode)) return '';
    return html`<aside class="reading-pane window-detail" id="windowDetail" data-layer="detail" data-mode="${x.mode}" data-by="${x.by ? x.by.join(' ') : ''}" tabindex="-1" aria-labelledby="inspectorTitle">${detailMarkup(x)}</aside>`;
  }
  const detailEl = () => $('#main .window-detail') || (S.inspector && !$('#inspector')?.hidden ? $('#inspector') : null);
  // The detail drawn where it stands: beside its list (the page repaints with it) or over the lane.
  function paintDetail() {
    const shell = $('#inspector');
    if (!shell) return null;
    const hosted = Boolean(S.inspector) && Boolean(hostOf(S.inspector.mode));
    if (S.inspector) S.inspector.hosted = hosted;
    if (hosted || $('#main .window-detail')) { if (typeof patchMain === 'function') patchMain(); }
    if (hosted) { shell.hidden = true; delete shell.dataset.open; delete shell.dataset.mode; shell.innerHTML = ''; return $('#main .window-detail'); }
    cancelLeave(shell);
    shell.dataset.mode = S.inspector.mode;
    shell.innerHTML = detailMarkup();
    shell.setAttribute('aria-labelledby', 'inspectorTitle');
    shell.hidden = false;
    shell.dataset.open = 'true';
    return shell;
  }
  function openInspector({mode, title, kind = '', body, onClose = null, tabs = null, by = null, readHeader = null}) {
    if (!$('#inspector')) return false;
    const from = document.activeElement, shown = detailEl();
    if (!S.inspector || (from && from !== document.body && !shown?.contains(from))) S.inspectorReturn = from; // the latest opener, not the first
    if (S.inspector?.onClose && S.inspector.mode !== mode) S.inspector.onClose({nextMode: mode});
    S.inspector = {mode, title, kind, body, tabs, onClose, readHeader, locale: I18N.locale, by: by ? [String(by[0]), String(by[1] ?? '')] : S.inspector?.by || null};
    const el = paintDetail();
    syncFrame();
    renderTop(); // the toggle's pressed state; in the pane the layer is the crumb's last segment
    if (el) { markScrollEdges(el); el.focus({preventScroll: true}); } // a sideways scroller in the body shows its bar while there is something to roll (law 90)
    if (typeof Geometry !== 'undefined' && Geometry.fit) Geometry.fit(); // where it stands: under the page's head or the top row
    markDetail();
    return true;
  }
  // A language repaint reads only the open header's words, never reopens or rereads its owner.
  function refreshInspectorHeader() {
    const x = S.inspector;
    if (!x?.readHeader) return;
    const next = x.readHeader(), locale = I18N.locale;
    if (JSON.stringify([x.title, x.kind, x.tabs, x.locale]) === JSON.stringify([next.title, next.kind, next.tabs ?? null, locale])) return;
    Object.assign(x, next, {locale});
    const el = detailEl(), head = el?.querySelector('.detail-head');
    if (!head) return;
    const saved = preserveSurface(el), template = document.createElement('template');
    template.innerHTML = detailHeadMarkup(x);
    head.replaceWith(template.content);
    const back = el.querySelector('[data-action="detail-back"]');
    if (back) { template.innerHTML = backMarkup(); back.replaceWith(template.content); }
    restoreSurface(saved, {scroll: false});
  }
  function setInspectorBody(body, {keepScroll = true} = {}) {
    if (!S.inspector) return;
    S.inspector.body = body; // a repaint of the hosting list draws the body it holds now
    const el = detailEl(), host = el?.querySelector('.reading-pane-body');
    if (!host) return;
    const top = keepScroll ? host.scrollTop : 0; // the body scrolls; the glass card holds still (law 149)
    const lefts = [...host.querySelectorAll('.table-scroll')].map((x) => x.scrollLeft); // a repaint keeps each rolling table's place
    host.innerHTML = html`${body}`;
    host.scrollTop = top;
    [...host.querySelectorAll('.table-scroll')].forEach((x, i) => { if (lefts[i]) x.scrollLeft = lefts[i]; });
    markScrollEdges(host);
  }
  function closeInspector(restore = true, context = {}) {
    clearDetailTrail();
    if (!S.inspector) return false;
    const shell = $('#inspector'), was = S.inspector, old = S.inspectorReturn, hosted = Boolean($('#main .window-detail'));
    S.inspector = null; S.inspectorReturn = null;
    syncFrame();
    renderTop();
    if (was.onClose) was.onClose(context);
    if (hosted && typeof patchMain === 'function') patchMain(); // the list takes its width back
    markDetail();
    if (restore) returnFocus(old);
    // round 94: the layer over the lane slides back the way it came
    if (shell && !shell.hidden) leave(shell, () => {
      shell.hidden = true;
      delete shell.dataset.open;
      delete shell.dataset.mode;
      shell.innerHTML = '';
    });
    return true;
  }
  /* Ctrl ] (round 81; the top row's toggle retired in round 86 -- the inspector opens from an
   * object, and a memory of a closed body was a surprise): open → closed; closed → the page's
   * Facts, else its Record, else nothing. */
  function toggleInspector() {
    if (S.inspector) return closeInspector(true);
    return Inspect.openPanel(); // the remembered tab (round 92), else Facts, else Record
  }
  const inspectorOpen = () => Boolean(S.inspector);
  const inspectorMode = () => S.inspector?.mode || null;
  /* ---- the detail (law 149): one press opens it, the same press closes it ---- */
  // The press that opened the detail in view: the window's column, else the lane's reading.
  function detailBy() {
    if (S.inspector) return S.inspector.by;
    const pane = $('#main .reading-pane[data-by-action]');
    return pane ? [pane.dataset.byAction, pane.dataset.byValue || ''] : null;
  }
  const detailOpen = () => Boolean(S.inspector) || Boolean($('#main .reading-pane'));
  const openedBy = (action, value) => { const by = detailBy(); return Boolean(by) && by[0] === action && by[1] === String(value ?? ''); };
  /* ---- a detail's levels (LS5; the user, 2026-09-26: 这种多层的悬浮box, 需要有回退到上一层的按钮) ----
   * A press inside an open detail that opens another object in it goes one level down: the level it
   * left -- the press that opened it and its title -- is kept, and the head offers `‹ <that title>`,
   * which presses that level's own opener again. A detail opened from the page starts with no level
   * above it; a closed one keeps none. The dispatcher marks each press (`trailMark`) and settles the
   * levels once the press has painted (`trailAfter`). */
  const Trail = {levels: [], inside: false, back: false};
  function clearDetailTrail() { Trail.levels = []; Trail.inside = false; Trail.back = false; }
  const detailTitle = () => (detailEl() || $('#main .reading-pane'))?.querySelector('.reading-pane-head h2')?.textContent.replace(/\s+/g, ' ').trim() || '';
  const notePress = (el) => { Trail.inside = Boolean(el?.closest?.('[data-layer="detail"]')); };
  function trailMark() { const by = detailBy(), title = detailTitle(); const readHeader = S.inspector?.readHeader || (() => ({title: [...$$('#main .reading-pane[data-by-action]')].find(el => el.dataset.byAction === by?.[0] && el.dataset.byValue === by?.[1])?.querySelector('.reading-pane-head h2')?.textContent.trim() || title})); const mark = {by, title, readHeader, inside: Trail.inside}; Trail.inside = false; return mark; }
  function trailAfter(mark) {
    const now = detailBy(), before = Trail.levels.length;
    if (!now) Trail.levels = [];
    else if (Trail.back) Trail.back = false;
    else if (mark.by && mark.by.join('\n') === now.join('\n')) return; // the same level, repainted
    else if (mark.inside && mark.by) Trail.levels.push({by: mark.by, title: mark.title, readHeader: mark.readHeader});
    else Trail.levels = [];
    if (now && Trail.levels.length !== before) repaintDetail(); // the head painted before its level was known
  }
  const backTo = () => (detailOpen() ? Trail.levels.at(-1) || null : null);
  function back() { const level = Trail.levels.pop(); if (!level) return; Trail.back = true; return dispatchAction(level.by[0], level.by[1]); }
  // the way up as the card's hidden tab (readingPane, detailMarkup draw it first in the card): a bare glyph in the
  // card's left inset beside its head, out of sight until the card is under the pointer (the user, 2026-09-26: 就一个
  // <- 就够了; 回退, 能做成左侧隐藏的悬浮标); the level it goes to is its tip
  const backMarkup = () => { const level = backTo(); return level ? btnAttrs(icon('back'), 'detail-back', '', 'icon-btn detail-back detail-back-tab', html`aria-label="${t('Back to {name}', {name: level.readHeader?.().title || level.title})}" data-tip="${t('Back to {name}', {name: level.readHeader?.().title || level.title})}"`) : ''; };
  const repaintDetail = () => { if (S.inspector) paintDetail(); else if (typeof patchMain === 'function') patchMain(); };
  function closeDetail() {
    clearDetailTrail();
    if (S.inspector) return closeInspector(true);
    const close = $('#main .reading-pane .reading-pane-head [data-action]');
    if (close) dispatchAction(close.dataset.action, close.dataset.value);
    return Boolean(close);
  }
  // The row of the press that opened the detail carries the mark (`aria-current`) while it is open
  // and not after; an opener outside a row (a head's (i)) says it is expanded. A repaint marks it
  // again (the router calls this after every paint).
  function markDetail() {
    const by = detailBy(), key = by ? JSON.stringify(by) : '';
    for (const el of document.querySelectorAll('#main [data-detail-mark]')) if (el.dataset.detailMark !== key) { el.removeAttribute(el.matches('tr, .list-row') ? 'aria-current' : 'aria-expanded'); el.removeAttribute('data-detail-mark'); }
    if (S.focusBack && !$('#main .reading-pane')) { const el = openerOf(S.focusBack); S.focusBack = null; if (el) Geometry.focusQuietly(el); }
    if (typeof Geometry !== 'undefined' && Geometry.fit) Geometry.fit(); // the detail's height limit from the paint that drew it, not a frame later
    if (!by) return;
    for (const b of document.querySelectorAll('#main [data-action]')) {
      if (b.dataset.action !== by[0] || (b.dataset.value || '') !== by[1] || b.closest('[data-layer="detail"]')) continue;
      const row = b.closest('tr, .list-row');
      if (row && row.getAttribute('aria-current') !== 'true') { row.setAttribute('aria-current', 'true'); row.dataset.detailMark = key; }
      else if (!row && !b.hasAttribute('aria-expanded')) { b.setAttribute('aria-expanded', 'true'); b.dataset.detailMark = key; }
    }
  }
  // Closing returns the focus to the opener: the element itself, or its successor after a repaint;
  // a lane reading closes by a repaint (or a Back), so its owner names the press for the paint after.
  const openerOf = (by) => [...document.querySelectorAll('#main [data-action]')].find((x) => x.dataset.action === by[0] && (x.dataset.value || '') === (by[1] || '') && x.getClientRects().length && !x.closest('[data-layer="detail"]')) || null;
  function returnFocus(old) {
    const el = old?.isConnected ? old : old?.dataset?.action ? openerOf([old.dataset.action, old.dataset.value || '']) : null;
    if (el) Geometry.focusQuietly(el);
  }
  function focusAfterPaint(by) { S.focusBack = by; }
  const inspectorTitle = () => $('#inspectorTitle')?.textContent || '';

  /* ---- text size ---- */
  const zoomLevel = () => { const z = Number(readPreference('zoom')); return ZOOM_STEPS.includes(z) ? z : 100; };
  // the scale and its token: viewport lengths (`--vh`, `--vw`) divide by it, since `zoom` does not scale them (round 84)
  function applyZoom(pct) { const el = document.documentElement; el.style.zoom = pct === 100 ? '' : String(pct / 100); el.style.setProperty('--zoom', String(pct / 100)); }
  function setZoom(pct) {
    const next = ZOOM_STEPS.includes(Number(pct)) ? Number(pct) : 100;
    savePreference('zoom', next);
    applyZoom(next);
    $$('.zoom-note').forEach((el) => { el.textContent = next + ' %'; });
    notify('Text size {n} %', {n: next});
    syncLayout();
    repaintSettings();
    markScrollEdges(); // the text size moved every measured place (a strip's fit, a list's empty state, a table's roll); a zoom fires no resize
  }
  const stepZoom = (dir) => setZoom(ZOOM_STEPS[Math.max(0, Math.min(ZOOM_STEPS.length - 1, ZOOM_STEPS.indexOf(zoomLevel()) + Number(dir)))]);

  /* ---- the frame's attributes ---- */
  /* The page's place in the top row (N1; law 119): the actions, the explanation and the state its
   * head left as templates in `#main`, read after every paint and patch -- the actions in the
   * row's slot, the explanation an (i) and the state beside the path's last segment. */
  function syncPageTop() {
    const main = $('#main'), slot = $('#top .top-actions'), where = $('#top .top-where');
    if (!main || !slot || !where) return;
    const actions = main.querySelector('template.page-actions')?.innerHTML.trim() || '';
    if (slot.pageMarkup !== actions) { slot.innerHTML = actions; slot.pageMarkup = actions; where.pageMarks = null; }
    // the verbs annotate themselves (a disabled one writes its reason beside it) before the row is
    // measured: fitted first, a reason written later crushed the path (W, 900 px in Chinese)
    if (typeof Controls !== 'undefined') Controls.sync(slot);
    const state = main.querySelector('template.page-state')?.innerHTML.trim() || '';
    const info = main.querySelector('template.page-info')?.content.textContent.replace(/\s+/g, ' ').trim() || '';
    const menu = main.querySelector('template.page-menu')?.innerHTML.trim() || '';
    const path = String(whereMarkup()); // N2: the page's scope and object are known after its paint
    const marks = path + '|' + state + '|' + info + '|' + menu;
    if (where.pageMarks === marks) return fitTop();
    where.pageMarks = marks;
    where.innerHTML = path;
    if (state) where.insertAdjacentHTML('beforeend', html`<span class="top-state">${raw(state)}</span>`);
    if (info) where.insertAdjacentHTML('beforeend', html`<span class="top-info" tabindex="0" role="img" aria-label="${info}" data-tip="${info}">${icon('info')}</span>`);
    if (menu) where.insertAdjacentHTML('beforeend', html`<span class="top-menu">${raw(menu)}</span>`); // law 129: the page's ··· beside the path, as Linear's beside its breadcrumb
    // the page's verbs are also the menu's first rows, shown only while the row has folded them
    const list = where.querySelector('.top-menu .menu');
    if (list) list.insertAdjacentHTML('afterbegin', [...slot.querySelectorAll('button[data-action], a[href]')].map((v) => v.tagName === 'A'
      ? html`<a class="menu-row menu-verb" role="menuitem" href="${v.getAttribute('href')}"><span class="menu-word">${v.textContent.trim()}</span></a>`
      : btnAttrs(html`<span class="menu-word">${v.textContent.trim()}</span>`, v.dataset.action, v.dataset.value || '', 'menu-row menu-verb', html`role="menuitem"${v.disabled || v.getAttribute('aria-disabled') === 'true' ? html` aria-disabled="true" data-tip="${v.dataset.tip || ''}"` : ''}`)).join('')); // a held verb stays held in the menu, its reason its tip (a plain string here would be escaped: aria-disabled="&quot;true&quot;" pressed through the guard)
    fitTop();
  }
  /* N1 (law 129): the top row keeps the path's last word and never overflows. It folds in steps,
   * each only while the row is still tight: the page's verbs into its ··· (their first rows) as soon
   * as they would cut the path's last word; then the window's Recent and appearance (the product
   * menu and ⌘K keep both); then the path's upper levels, one at a time from the left, into `…`
   * (F3, law 135: the dock and the `…` menu keep them); then the state's word (its mark stays) --
   * each only while the row overflows or the last word keeps less than a third of the row.
   * Measured, not a breakpoint: the fold follows the words the page has, at any width and text
   * size; a row that fits unfolds. */
  function fitTop() {
    const top = $('#top');
    if (!top) return;
    delete top.dataset.fold;
    for (const el of top.querySelectorAll('.top-where > [data-folded]')) el.removeAttribute('data-folded');
    const current = top.querySelector('.top-where .top-page[aria-current]');
    const over = () => top.scrollWidth > top.clientWidth + 1;
    // crushed: narrower than its own width -- a name cut at its parameter (`crumb-object`) is not crushed
    const crushed = (el) => { if (!el) return false; const cap = parseFloat(getComputedStyle(el).maxWidth); return el.clientWidth + 1 < Math.min(el.scrollWidth, Number.isFinite(cap) ? cap : Infinity); };
    const cut = () => crushed(current);
    const tight = () => over() || (cut() && current.clientWidth < top.clientWidth / 3);
    const folds = [];
    const fold = (stage) => { if (!folds.includes(stage)) folds.push(stage); top.dataset.fold = folds.join(' '); };
    // a level above is whole or folded: cut to an ellipsis it names nothing (N6, 375)
    const segs = [...top.querySelectorAll('.top-where > .top-seg')], above = segs.slice(0, -1);
    const upCut = () => above.some((s) => !s.hasAttribute('data-folded') && crushed(s));
    if (top.querySelector('.top-actions')?.children.length && (over() || cut())) fold('verbs');
    if (tight() || upCut()) fold('tools'); // a crushed level folds the tools first, then itself
    for (const s of above) {
      if (!(tight() || upCut())) break;
      s.setAttribute('data-folded', ''); const sep = s.nextElementSibling; if (sep?.classList.contains('top-sep')) sep.setAttribute('data-folded', '');
      fold('path');
    }
    if (tight()) fold('state');
  }
  function syncFrame() {
    syncPageTop();
    const w = win(), r = rest(), l = layout();
    document.documentElement.dataset.nav = l;
    document.documentElement.dataset.dock = dockSide();
    if (w) { w.dataset.side = sideOpen() ? 'open' : 'closed'; w.style.setProperty('--side-width', sideWidth() + 'px'); }
    const edge = $('#side .side-resize'); if (edge) edge.setAttribute('aria-valuenow', String(sideWidth()));
    if (r) r.dataset.inspector = S.inspector ? 'open' : 'closed';
    const main = $('#main'); if (main) main.inert = Boolean(S.inspector) && pane() && !S.inspector.hosted; // the lane waits under the layer (a detail its list hosts is the layer inside it)
    const toggle = $('#top .top-side-toggle');
    if (toggle) { toggle.setAttribute('aria-expanded', String(sideOpen())); toggle.setAttribute('aria-label', t(sideOpen() ? 'Hide navigation' : 'Show navigation')); } // the toggle is always there (the user's word, 2026-09-20: "toggle按钮应该一直在"), as Codex's and Finder's are
  }
  // a width or a scale that moves the placement (sidebar / rail / drawer) redraws the frame
  function syncLayout() {
    const l = layout(), w = wide();
    if (l !== S.layout || w !== S.wide) { S.layout = l; S.wide = w; S.override = null; S.sideOverlay = false; renderSide(); renderTop(); }
    syncFrame();
  }
  /* The mounted entries follow the reader's current context: a page's in-place repaint replaces
   * sections of `#main` only, and a full render builds the side before the page records what it
   * shows, so the rows already on screen are re-addressed from the same `routeUrl` that made
   * them -- an href is what a click, a copy or a middle-click opens, and it must name the
   * reader's current session, participant and exchange. Attributes only: nothing is redrawn. */
  function syncRoutes() {
    for (const a of document.querySelectorAll('#side a[data-page]')) {
      const href = entryUrl(a.dataset.page); // a chooser page's row is its list; another page's row keeps its subject; Data opens on its last tab
      if (a.getAttribute('href') !== href) a.setAttribute('href', href);
    }
  }
  function render() {
    localizeChrome();
    S.layout = layout(); S.wide = wide();
    renderSide();
    renderTop();
    syncFrame();
  }
  function afterRender() {
    refreshInspectorHeader();
    document.body.classList.toggle('focus-mode', Inspect.focus);
    renderSide(); // a page that moved while it painted (Conversation without a session opens Sessions) moves the dock's current row with it; memoised, free when unchanged
    renderTop(); // the page has painted: its head names the object the top row says
    syncFrame();
    syncRoutes(); // the page has rendered and recorded what it shows; the side was built before it
    markScrollEdges();
  }
  /* A navigation closes the side's overlay and, when the page changed, the inspector (a body
   * belongs to the page it was opened on). */
  function onNavigate(pageChanged) {
    closeSideOverlay();
    if (pageChanged) closeInspector(false);
  }
  function about() {
    openDialog('AlphaLattice', t('About'), html`${kv([[t('Product'), 'AlphaLattice · Local Web'], [t('Version'), window.AlphaLattice?.version || ''], [t('Service'), html`<span class="mono">${location.origin}</span>`], [t('Workspace'), Data.workspace()], [t('Navigation'), t(NAV_MODES.find(([k]) => k === navMode())[1])], [t('Text size'), zoomLevel() + ' %']])}<p class="caption">${t('A person and the agent both run this system; the decisions are the person\'s. Nothing here runs a model or reaches the network.')}</p>`, '');
  }
  /* The window's own actions and keys (round 81) register here, after every owner loaded. */
  const OWN = {'product-menu': toggleProduct, 'rail-group': toggleRailGroup, 'recent-menu': toggleRecent, 'path-fold': togglePathFold, 'side-new': toggleNew, 'inspector-toggle': toggleInspector, 'nav-set': setNavigation, 'dock-set': setDockSide, 'zoom-step': stepZoom, 'zoom-set': setZoom, about};
  function hotkeys(e) {
    if (keyResize(e)) return true;
    if (!(e.ctrlKey || e.metaKey) || e.shiftKey || e.altKey || !['=', '+', '-', '_', '0', ']'].includes(e.key)) return false;
    e.preventDefault();
    if (e.key === ']') toggleInspector();
    else if (e.key === '0') setZoom(100);
    else stepZoom(e.key === '=' || e.key === '+' ? 1 : -1);
    return true;
  }
  // a setting changed from the Settings page repaints the page, so its rows say the new value (round 84)
  const repaintSettings = () => { if (S.repaint && (app.page === 'settings' || app.page === 'advanced')) S.repaint(); };
  function bind(repaint = null) {
    S.repaint = repaint;
    Object.assign(ACTIONS, OWN);
    for (const name of Object.keys(OWN)) READ_ACTIONS.add(name);
    applyZoom(zoomLevel());
    // the side's overlay closes on a click outside it (a docked side never does)
    document.addEventListener('click', (e) => {
      if (!column() && S.sideOverlay && !e.target.closest('#side, [data-action="side-toggle"]')) closeSideOverlay();
    }, true);
    let resizeTimer;
    addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(syncLayout, 60); });
    // the dock's edge (round 86): drag, double-click for the default; the drawer's peek at the left edge
    document.addEventListener('pointerdown', startResize);
    document.addEventListener('dblclick', (e) => { if (e.target.closest('.side-resize')) setSideWidth(SIDE_WIDTH.default); });
    document.addEventListener('pointermove', peekSide, {passive: true});
    $('#side')?.addEventListener('pointerleave', unpeekSide);
  }
  /* The static chrome is authored in English in shell.html; `data-i18n` names the text and
     attributes that follow the interface language. The English source is read once. */
  let chrome = null;
  function localizeChrome() {
    if (!chrome) {
      chrome = $$('[data-i18n]').map((el) => {
        const parts = el.dataset.i18n.split(' ');
        const texts = parts.includes('text') ? [...el.childNodes].filter((n) => n.nodeType === 3 && n.nodeValue.trim()).map((n) => [n, n.nodeValue]) : [];
        const attrs = parts.filter((p) => p !== 'text').map((p) => [p, el.getAttribute(p)]);
        return {el, texts, attrs};
      });
    }
    for (const {el, texts, attrs} of chrome) {
      for (const [node, en] of texts) node.nodeValue = t(en);
      for (const [attr, en] of attrs) el.setAttribute(attr, t(en));
    }
  }
  /* The place's tabs (N4-N6): one row of underline tabs under the page's head (objectHead's `views`) --
   * an object's folders while it is chosen, a list's views and Data's pages on the list's own pages; each
   * tab counts its own objects where the owner records them; Data remembers the last one (law 132). */
  const TAB_COUNT = {team: () => LiveTeam.counts?.().exchanges, 'team-participants': () => LiveTeam.counts?.().participants, 'team-evidence': () => LiveTeam.counts?.().observations, inputs: () => (Data.inputs() || []).length, alpha: () => (Data.experiments() || []).filter((v) => v.kind === 'alpha.model-development').length, portfolio: () => Data.portfolioEntries().length,
    'evidence-stream': () => LiveReview.counts?.().documents, 'evidence-reading': () => LiveReview.counts?.().citations, handoff: () => LiveReview.counts?.().findings}; // B1: a book's tabs count their objects where the owner records them
  function pageTabs() {
    const list = FOLDER_OF[app.page] || (PAGE_TABS[app.page] ? app.page : ''), set = PAGE_TABS[list];
    if (!set || !set.tabs.includes(app.page)) return '';
    if (set.param ? !hashParams().get(set.param) : routeObjectId() && !set.keep?.includes(app.page)) return '';
    if (set.remember && readPreference('tab:' + list) !== app.page) savePreference('tab:' + list, app.page);
    const place = SIDEBAR.find((x) => x.page === list)?.word || pageWord(list);
    const counted = (page) => { const n = TAB_COUNT[page]?.(); return typeof n === 'number' ? count(n) : ''; }; // a tab with nothing says 0 (a review, 2026-09-24)
    return tabStrip(t(place), set.tabs.map((page) => ({word: t(set.words?.[page] || pageWord(page)), on: page === app.page, action: 'go', value: page, count: counted(page)})), 'page-tabs');
  }
  return {render, afterRender, renderSide, renderTop, syncFrame, syncRoutes, pageTabs, toggleSide, toggleGroup, sideOpen, closeSideOverlay, openInspector, detailPane, inspectorTitle, setInspectorBody, refreshInspectorHeader, closeInspector, toggleInspector, inspectorOpen, inspectorMode, detailOpen, openedBy, closeDetail, clearDetailTrail, markDetail, notePress, trailMark, trailAfter, backTo, back, backMarkup, returnFocus, focusAfterPaint, onNavigate, bind, docked, railed, pane, layout, navMode, setNavigation, dockSide, setDockSide, zoomLevel, setZoom, stepZoom, toggleWorkspace, toggleProduct, toggleRailGroup, toggleRecent, about, hotkeys};
})();

/* Remember how the open dialog was produced so a language switch can rebuild it in place. */
const Dialog = (() => {
  let current = null;
  function remember(open) {
    current = open;
  }
  function rerender() {
    const d = $('#dialog');
    if (!d.open || !current) return;
    const values = new Map($$('input,select,textarea', d).filter((x) => x.id).map((x) => [x.id, x.type === 'checkbox' ? x.checked : x.value]));
    const scroll = d.querySelector('.dialog-body')?.scrollTop || 0;
    current();
    for (const [id, value] of values) {
      const el = document.getElementById(id);
      if (!el || !d.contains(el)) continue;
      if (el.type === 'checkbox') el.checked = value;
      else el.value = value;
    }
    const body = d.querySelector('.dialog-body');
    if (body) body.scrollTop = scroll;
    Controls.sync(d);
  }
  return {remember, rerender, clear: () => (current = null)};
})();
