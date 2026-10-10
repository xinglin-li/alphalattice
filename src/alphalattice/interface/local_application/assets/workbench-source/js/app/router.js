/* Hash routing, theme and language switching, and the single render pass. */
const PAGES = {}; // page id → () => html; filled by the page modules below.

/* Route identity and viewer preferences are intentionally separate. Old links cannot roll back a theme. */
/* The History view is its filters: they travel in the route (replaced while typing, never
 * pushed), so a copied link, a reload and Back/Forward all show the same list. Defaults are
 * omitted so an unfiltered link stays short. */
const HISTORY_ROUTE = {q: ['historyQuery', ''], kind: ['historyKind', 'all']}; // the ordering is a display choice (round 57), kept per viewer
const historyRoute = () => Object.fromEntries(Object.entries(HISTORY_ROUTE).filter(([, [field, fallback]]) => app[field] !== fallback).map(([key, [field]]) => [key, app[field]]));
/* The link grammar (round 67): `#page=<page>` names the page; the object travels as the page's
 * own key (`study`, `book`, `task`, `foundation`, `case`, `plan`, `team` / `actor` / `event`,
 * `review_publication` / `review_selector`, `preparation`, `update`); `input` and `session` name
 * the research input and the holdings date; History carries its filters (`q`, `kind`, `mode`);
 * `facts=1` and `record=1` reopen the inspector's Facts or Record on that object; `theme` and
 * `lang` never travel (a viewer's own). `copyLink()` is the one writer of a shareable address. */
function copyLink() {
  const q = hashParams();
  for (const k of ['theme', 'lang']) q.delete(k);
  return copyText(location.href.split('#')[0] + '#' + q.toString(), 'Link copied');
}
function routeUrl(page, extra = {}) {
  const {theme: ignoredTheme, lang: ignoredLang, ...context} = extra;
  for (const key of ['follow', 'follow_paused']) if (!Object.hasOwn(context, key) && hashParams().get(key)) context[key] = hashParams().get(key);
  if (page === 'history') Object.assign(context, {...historyRoute(), ...context});
  // An object's folder tabs keep the address the tab table requires, including a
  // Goal's exact revision. Explicit keys still clear or replace that object.
  if (typeof FOLDER_OF !== 'undefined') {
    const folder = FOLDER_OF[page], set = PAGE_TABS[folder];
    if (set?.param && set.tabs.includes(app.page)) Object.assign(context, {[set.param]: hashParams().get(set.param), ...context});
  }
  if (LiveReview.pages.has(page) && LiveReview.pages.has(app.page)) Object.assign(context,{...LiveReview.routeContext(),...context});
  if (LiveStudy.pages.has(page))Object.assign(context,{...LiveStudy.routeContext(page),...context});
  if (page==='lab')Object.assign(context,{...LiveResearch.routeContext(),...context});
  if (page==='features')Object.assign(context,{...LiveFeatures.routeContext(),...context});
  if (typeof LiveTeam!=='undefined' && LiveTeam.pages?.has(page) && LiveTeam.routeContext)Object.assign(context,{...LiveTeam.routeContext(),...context}); // the reader's retained session/actor/exchange; explicit keys win
  if (page==='compare')Object.assign(context,{compare: app.compareOther || '',...context}); // both members of a comparison travel together
  // A cleared value leaves the address rather than lingering as an empty key (the rule below,
  // for links as for rewrites): a copied address stays readable, and the way up to a page's
  // list is the page's own address. The prototype's `state=normal` had no reader and went.
  const entries = Object.entries({page, input: app.input, book: app.book, session: app.session, ...context}).filter(([, v]) => v !== '' && v !== null && v !== undefined);
  return '#' + new URLSearchParams(entries).toString();
}
const hashParams = () => new URLSearchParams(location.hash.slice(1));
/* Rewrite the current entry in place. A cleared value leaves the route rather than lingering
 * as an empty key: every reader treats absence and emptiness alike, and a copied address
 * stays readable. */
function replaceHash(update) {
  const q = hashParams();
  for (const [k, v] of Object.entries(update)) { if (v === '' || v === null || v === undefined) q.delete(k); else q.set(k, v); }
  history.replaceState(history.state, '', '#' + q);
}
const historyDepth = () => Number(history.state?.alpha) || 0; // round 81: the entries this product pushed and stamped
/* A pushed address (2026-09-22, the way back): a choice that changes what a page holds -- a packet, an
 * item opened in the pane, a page stepped to -- is one entry, stamped with its depth, so `<-` returns to
 * the level before it; the depth the next link navigation is stamped from follows it. */
function pushHash(update, detailReturn = history.state?.detailReturn) {
  const q = hashParams();
  const before = readingAddress(q);
  for (const [k, v] of Object.entries(update)) { if (v === '' || v === null || v === undefined) q.delete(k); else q.set(k, v); }
  const hash = '#' + q;
  if (location.hash !== hash) {
    if (readingAddress(q) !== before) Data.beginNavigation();
    history.pushState({alpha: historyDepth() + 1, ...(detailReturn ? {detailReturn} : {})}, '', hash);
  }
  entryDepth = historyDepth();
}
/* A routed detail keeps its opening page across all of its own route entries. Closing
 * traverses to that page without deleting the entries Back and Forward still read. */
function pushDetail(update) {
  pushHash(update, {depth: historyDepth(), hash: location.hash});
}
function closeDetailRoute(update) {
  const origin = history.state?.detailReturn;
  if (origin && Number.isInteger(origin.depth) && origin.depth >= 0 && origin.depth < historyDepth()) {
    history.go(origin.depth - historyDepth());
    return true;
  }
  replaceHash(update); // a cold detail link has no opening entry to traverse to
  return false;
}
function readRoute() {
  const q = hashParams();
  let page = q.get('page');
  // Preserve old exact Team bookmarks while separating the two working surfaces.
  if (page === 'tasks' && q.get('team')) {
    page = 'team'; q.set('page', page);
    history.replaceState(history.state, '', '#' + q);
  }
  if (page === 'advanced') page = 'settings'; // round 62: the Advanced page is a section of Settings
  if (page === 'welcome') page = 'overview'; // round 74: first use is the Home with one run; old links land there
  if (page === 'issues' && !q.get('issue')) page = 'data'; // N5 (2026-09-24): Data issues is Data maintenance's section; an issue keeps its page
  if (page === 'team-members') page = 'team'; // C4 (2026-09-24): Participants folded into the conversation's filter
  if (page && ROUTES[page]) app.page = page;
  const inp = q.get('input');
  if (app.inputs.some((x) => x.id === inp)) app.input = inp;
  const date = q.get('session');
  if (Data.sessions().includes(date)) app.session = date;
  // a Portfolio page shows what its address says: no `book` in it is the list, whatever was
  // read before (the way up; 2026-09-21, the user's reading), and the reader forgets the book
  if (DATA_PAGES.has(app.page)) { app.book = q.get('book') || ''; if (!app.book) app.compareOther = ''; }
  if (app.page === 'history') for (const [key, [field, fallback]] of Object.entries(HISTORY_ROUTE)) app[field] = q.get(key) || fallback;
}
/* One history entry per explicit navigation to another saved object: the click pushes the
 * place being left; the object's reader then writes its route in place (a load, a Back/Forward
 * re-open, a holdings-date or comparison change, a refresh add no entry). `key` names the object
 * about to open; opening the one the route already shows adds nothing. A follow-open makes its
 * one entry the same way, by the reader's earlier choice to follow. */
function objectEntry(key) {
  Data.beginNavigation();
  if (routeObject() === key) return false;
  history.pushState({alpha: historyDepth() + 1}, '', location.href);
  const opened = readPreference('objects.recent'); // round 65: the Home's recent objects, per viewer
  savePreference('objects.recent', [key, ...(Array.isArray(opened) ? opened : []).filter((k) => k !== key)].slice(0, 12));
  return true;
}
/* The object a page shows, by its address keys: a study page's `study`, the Foundation page's
 * `foundation`, a Portfolio page's `book` (a comparison's other member with it), a review page's
 * publication or selector, the Data page's `update`, a case on the Lab. A selector another page
 * left in the address (the study a reader came from, the book they read before) is not this
 * page's object and never counts as one. */
const REVIEW_OBJECT = {kind: 'review', ids: ['review_publication', 'review_selector'], clears: ['review_publication', 'review_selector']};
// the Reading page's own object is the packet it holds (round E4): its dock row and its way up are the chooser, the book stays
const READING_OBJECT = {kind: 'review', ids: ['review_publication', 'review_selector'], clears: ['prepared_task', 'prepared_unit', 'view_entity_id', 'view_topic', 'view_last_days', 'review_item'], chooser: true};
const STUDY_OBJECT = {kind: 'study', ids: ['study'], clears: ['study'], chooser: true};
const BOOK_OBJECT = {kind: 'book', ids: ['book'], clears: ['book', 'compare'], chooser: true};
const GOAL_OBJECT = {kind: 'goal', ids: ['goal', 'case'], clears: ['goal', 'case'], chooser: true};
const OBJECT_KEYS = {
  goals: GOAL_OBJECT, goal: GOAL_OBJECT, 'goal-conversation': GOAL_OBJECT, 'goal-results': GOAL_OBJECT, cases: GOAL_OBJECT,
  foundation: {kind: 'foundation', ids: ['foundation'], clears: ['foundation'], chooser: true},
  factor: STUDY_OBJECT, risk: STUDY_OBJECT, alpha: {kind: 'study', ids: ['study'], clears: ['study', 'alpha_left_task', 'alpha_left_candidate', 'alpha_right_task', 'alpha_right_candidate'], chooser: true},
  'alpha-compare': {kind: 'study', ids: ['alpha_left_task', 'study'], clears: ['study', 'alpha_left_task', 'alpha_left_candidate', 'alpha_right_task', 'alpha_right_candidate'], chooser: true},
  portfolio: BOOK_OBJECT, compare: BOOK_OBJECT,
  evidence: REVIEW_OBJECT, 'evidence-stream': REVIEW_OBJECT, 'evidence-reading': READING_OBJECT, report: REVIEW_OBJECT, handoff: REVIEW_OBJECT,
  data: {kind: 'update', ids: ['update'], clears: ['update']},
  issues: {kind: 'issue', ids: ['issue'], clears: ['issue'], chooser: true}, // F2: a data issue's page; the dock's row is the lobby
  inputs: {kind: 'version', ids: ['version'], clears: ['version', 'via'], chooser: true}, // F3: an input version's page (four levels); `via` the list it was opened from
  models: {kind: 'model', ids: ['model'], clears: ['model'], chooser: true}, // U50: a model's review packet; the dock's row is the list
  'feature-research': {kind: 'feature-review', ids: ['feature'], clears: ['feature'], chooser: true}, // U56: a formula factor's review packet, `<plan hash>:<factor id>`
  features: {kind: 'feature', ids: ['feature_plan', 'feature_build'], clears: ['feature_plan', 'feature_build']}, // a saved PLAN or a build's values; the way up is the composer over the same input
};
function routeObject(q = hashParams()) {
  const page = q.get('page') || app.page;
  const object = OBJECT_KEYS[page];
  if (!object) return 'page:' + page;
  const id = object.ids.map((k) => q.get(k)).find(Boolean) || '';
  return object.kind + ':' + id;
}
/* A reader's addressed subject, separate from Facts/Record and viewer preferences. Both
 * pushed choices and pending reads use this same grammar, including the object's context. */
function readingAddress(q = hashParams()) {
  const page = q.get('page') || app.page;
  const keys = ['input', 'session', 'task', 'reference', 'team', 'actor', 'event', 'committee', 'committee_point', ...(OBJECT_KEYS[page]?.clears || [])];
  if (page === 'lab') keys.push('plan', 'origin', 'draft_source', 'research_input', 'input_binding', 'experiment_kind');
  if (['evidence', 'evidence-stream', 'evidence-reading', 'report', 'handoff'].includes(page)) keys.push('book', 'prepared_task', 'prepared_unit', 'work');
  return JSON.stringify([page, routeObject(q), ...[...new Set(keys)].map(key => q.get(key) || '')]);
}
/* The id of the object the address shows, or '' on a page's list (or a page without objects). */
function routeObjectId() {
  const object = routeObject();
  return object.startsWith('page:') ? '' : object.slice(object.indexOf(':') + 1);
}
/* The page's list -- its address without the object it shows. The crumb's page word leads here
 * (the way up), and so does the dock's row of a page whose object is chosen from its own list
 * (`chooser`): the row is the page's root, as Linear's sidebar is. A page whose selection is a
 * subject that travels between its group's pages (the desk's book, the Team's session) keeps it
 * on its row. */
function listUrl(page) {
  const object = OBJECT_KEYS[page];
  return routeUrl(page, object ? Object.fromEntries(object.clears.map((k) => [k, ''])) : {});
}
const railUrl = (page) => (OBJECT_KEYS[page]?.chooser ? listUrl(page) : routeUrl(page));
/* An address reached by a link is the product's own navigation as much as a pushed one: the
 * entry the browser made for it is stamped with its depth, so `<-` offers the way back after a
 * rail row or a crumb, and never past the product's first entry. */
let entryDepth = 0;
function stampEntry() {
  if (history.state === null || history.state === undefined) history.replaceState({alpha: entryDepth + 1}, '', location.href);
  entryDepth = historyDepth();
}
function navigate(page, extra = {}, {replace = false} = {}) {
  Data.beginNavigation();
  if (typeof SAVED_VIEWS !== 'undefined' && !SAVED_VIEWS.has(page) && page !== 'history') app.listContext = null; // round 53: a navigation elsewhere forgets the list a record came from
  LiveWorkspace.dismissConfirmation();
  hideToast();
  closeDialog();
  if (extra.theme && THEMES.includes(THEME_ALIASES[extra.theme] || extra.theme)) setTheme(extra.theme); // the probes' way in; a link never carries it
  const hash = routeUrl(page, extra);
  if (location.hash !== hash) {
    if (replace) history.replaceState(history.state, '', hash);
    else history.pushState({alpha: historyDepth() + 1}, '', hash);
  }
  entryDepth = historyDepth(); // the next link navigation is stamped from this entry
  app.page = page;
  readRoute();
  render();
  // Instant: the page's smooth scrolling would animate this, and a repaint during the animation
  // would remember the half-way position as the reader's place and return the page to it.
  scrollTo({top: 0, behavior: 'instant'});
}

/* The appearance in effect (round 68): the viewer's choice, or the host's when following. */
const hostDark = () => typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: dark)').matches;
const effectiveTheme = () => app.theme === 'follow' ? (hostDark() ? 'dark' : 'light') : app.theme;
function applyTheme() {
  document.documentElement.dataset.theme = effectiveTheme();
  document.documentElement.style.colorScheme = effectiveTheme();
}
function setTheme(theme) {
  theme = THEME_ALIASES[theme] || theme;
  if (!THEMES.includes(theme)) return;
  app.theme = theme;
  savePreference('theme', theme);
  applyTheme();
  replaceHash({theme: '', page: app.page}); // a viewer's own; it never travels in a link
  if (typeof Window !== 'undefined') { Window.renderSide(); Window.renderTop(); } // the choice, wherever it was made
  if (app.page === 'settings') render();
}
/* The sun / moon: the other appearance, chosen (following the host ends with the click). */
function toggleTheme() { setTheme(effectiveTheme() === 'dark' ? 'light' : 'dark'); }
function setLocale(lang) {
  const normalized = ['zh-CN', 'cn', 'zh'].includes(lang) ? 'zh-CN' : 'en'; // the public API takes every spelling I18N.set takes (round E2)
  const apply = () => {
    I18N.set(normalized);
    savePreference('lang', normalized);
    replaceHash({lang: normalized, page: app.page});
    render();
    Dialog.rerender();
    refreshToast();
  };
  const dictionary = I18N.ensure(normalized); // the dictionary loads once, on the first switch (round 96)
  // a dictionary that did not load is named, never applied as an empty catalog: the page was
  // opened before the product changed (its content-hashed assets are gone), and a reload is the way
  if (dictionary) dictionary.then((ok) => { if (ok) apply(); else notify(t('The dictionary did not load: the product changed since this page was opened. Reload the page to switch the language.')); }); else apply();
}

function pageTitleText() {
  return 'AlphaLattice · ' + t(ROUTES[app.page]?.[1] || 'Research') + ' · Local Web';
}
/* The window's title leads with what needs you, as Home counts it -- seen from another tab or window
   (the user, 2026-09-25: 切走时也能知道). */
function titleText() {
  const n = typeof LiveViews !== 'undefined' && LiveViews.needs ? LiveViews.needs() : 0;
  return (n ? `(${n}) ` : '') + pageTitleText();
}

/* ---- surface memory across re-renders of the same page (scroll, focus, disclosure state) ---- */
/* The reading state inside one root (the page by default; a drawer region for its own repaint):
 * nested scroll positions, open disclosures, text-field scroll and the focused control with its
 * selection. Captured before markup is replaced and put back after, so a repaint that must
 * change a region does not lose where the reader was inside it. */
/* N5 (law 113): the Team's conversation flows with the page -- no scroller of its own. */
const SURFACE_SCROLLERS = '.table-scroll,details,.run-log-lines,.reading-pane-body'; // law 149: a detail beside its list keeps its place when the list repaints (its body scrolls)
/* A scroller or disclosure with an id is named by it (unique in the document, so a row arriving
 * above it moves nothing). Each other one is named by its nearest anchored ancestor (an id, an
 * activity row's key, a holding row, a team actor) and its place within it, so that a region
 * gaining or losing rows does not hand one row's open disclosure to another; without an
 * anchor, its place in the root. */
const SURFACE_ANCHORS = '[id],[data-activity-key],[data-holding-row],[data-team-actor]';
function surfaceNodes(root) {
  return [...root.querySelectorAll(SURFACE_SCROLLERS)].map((el) => {
    if (el.id) return {el, key: el.id};
    const anchor = el.parentElement?.closest(SURFACE_ANCHORS);
    const scope = anchor && anchor !== root && root.contains(anchor) ? anchor : root;
    const name = scope === root ? '' : scope.id || scope.dataset.activityKey || scope.dataset.holdingRow || scope.dataset.teamActor || '';
    return {el, key: name + '#' + [...scope.querySelectorAll(SURFACE_SCROLLERS)].indexOf(el)};
  });
}
function preserveSurface(root = $('#main')) {
  const active = document.activeElement, inside = Boolean(active) && root.contains(active);
  return {
    root,
    page: app.page,
    x: scrollX,
    y: scrollY,
    focus: inside ? active.id || null : null,
    action: inside ? active.dataset?.action : undefined,
    value: inside ? active.dataset?.value : undefined,
    start: inside ? active.selectionStart : undefined,
    end: inside ? active.selectionEnd : undefined,
    fields: [...root.querySelectorAll('textarea,input')].filter((x) => x.id).map((x) => ({id: x.id, top: x.scrollTop, left: x.scrollLeft})),
    scrolls: surfaceNodes(root).map(({el, key}) => ({key, open: el.open, top: el.scrollTop, left: el.scrollLeft})),
  };
}
/* `scroll: false` leaves the window where it is (a drawer region never moves the page). */
function restoreSurface(s, {scroll = true} = {}) {
  const root = s.root?.isConnected ? s.root : $('#main');
  s.fields.forEach((a) => {
    const el = document.getElementById(a.id);
    if (el) {
      el.scrollTop = a.top;
      el.scrollLeft = a.left;
    }
  });
  const nodes = new Map(surfaceNodes(root).map(({el, key}) => [key, el]));
  // Disclosures first, scroll positions after: an opened disclosure changes the layout above a
  // scrolled viewport, and the browser's scroll anchoring would move a position set before it.
  s.scrolls.forEach((a) => { const el = nodes.get(a.key); if (el && a.open !== undefined) el.open = a.open; });
  s.scrolls.forEach((a) => {
    const el = nodes.get(a.key);
    if (el) {
      el.scrollTop = a.top;
      el.scrollLeft = a.left;
    }
  });
  if (!$('#dialog').open) {
    let el = s.focus ? document.getElementById(s.focus) : null;
    if (!el && s.action) el = [...root.querySelectorAll('[data-action]')].find((x) => x.dataset.action === s.action && x.dataset.value === s.value);
    if (el && !el.disabled) {
      Geometry.focusQuietly(el);
      if (typeof s.start === 'number' && el.setSelectionRange) {
        try {
          el.setSelectionRange(s.start, s.end);
        } catch (e) {
          /* not a text control */
        }
      }
    }
  }
  if (scroll) scrollTo({left: s.x, top: s.y, behavior: 'instant'});
}

/* ---- the render pass ---- */
/* Pages that cannot render without study data show the data state instead of their content. */
const DATA_PAGES = new Set(['portfolio', 'compare']);
function dataStateView() {
  const title = ROUTES[app.page][1].toLowerCase();
  if (Data.status === 'loading') {
    return html`${skeleton('head')}${skeleton('rows')}`;
  }
  if (Data.status === 'error') {
    return notRead(t('The study could not be loaded'), Data.error, '', btn(t('Try again'), 'data-reload', '', 'button compact'));
  }
  return html`<section class="panel pad">${emptyState(t('No study data; open a workspace or a saved study.'), link(t('Open workspace'), 'overview', 'text-btn'), '', 'elsewhere')}</section>`;
}
/* No site footer: an instrument ends where its content ends (round 8). The original product's
 * entry stays reachable from Quick Open and the workspace dialog. */

/* The in-place repaint of a live page between polls: the same markup `render()` would
 * produce, applied section by section -- a top-level section whose markup is unchanged is
 * left alone (its text selection, scroll and disclosures with it), a changed one is replaced,
 * and focus inside a replaced section is returned to the same control by id or by
 * action/value. Nothing scrolls. A page whose section count changed falls back to `render()`.
 * This is the one live page's repaint, not a rendering framework. */
let patched = null; // the markup of each top-level section as it was last painted (a render resets it)
function patchMain() {
  try { patchPage(); } catch (error) { renderFailure(error, 'patch'); }
}
function patchPage() {
  const main = $('#main');
  if (lastRenderedPage !== app.page || !main.childElementCount) return render();
  const next = document.createElement('template');
  next.innerHTML = html`${LiveViews.page()}`;
  if (lastRenderedPage !== app.page) return render(); // the page moved while it painted (the shown session left the window: Sessions): drawn as the page it opened, its marks and its dock with it
  const fresh = [...next.content.children], current = [...main.children].filter((el) => !el.hasAttribute('data-render-failure'));
  if (fresh.length !== current.length) return render();
  // Compared against the markup that was rendered or last patched, never the live DOM: controls
  // annotate themselves, template styles are applied in place and a live status line is
  // written after each paint, none of which is a change to show. A render records its own
  // sections (`patched`), so the first idle patch after it replaces nothing either.
  const last = patched && patched.length === fresh.length ? patched : sectionMarkup(main.innerHTML);
  const saved = preserveSurface(main), overlay = preserveSurface($('#inspector'));
  let changed = 0;
  patched = fresh.map((el) => el.outerHTML);
  const stable = (m) => String(m || '').replace(/<!--detail-body-->[\s\S]*?<!--\/detail-body-->/g, ''); // law 149: a detail's body is painted in place
  current.forEach((el, i) => {
    if (patched[i] === last[i] || stable(patched[i]) === stable(last[i])) return;
    el.replaceWith(fresh[i]); changed += 1;
  });
  Window.syncRoutes(); // the page's route context may have moved even where no section changed (a selection the page wrote while painting)
  Window.renderSide(); // the dock's counts and the window's title follow what arrived: a Team objection reaches Home's Needs a decision by a patch (2026-09-25); memoised, free when unchanged
  // The live owners read the route after a patch as after a render (round 86): a deep-linked
  // study, review or case first paints while the workspace is loading, and the session's
  // arrival is a patch of the same page -- without this, the link never opened its object.
  if (LiveGoals.pages.has(app.page)) queueMicrotask(LiveGoals.ensure);
  if (LiveModels.pages.has(app.page)) queueMicrotask(LiveModels.ensure);
  if (LiveFeatureResearch.pages.has(app.page)) queueMicrotask(LiveFeatureResearch.ensure);
  if (LiveReview.pages.has(app.page)) queueMicrotask(LiveReview.ensure);
  if (LiveStudy.pages.has(app.page)) queueMicrotask(LiveStudy.ensure);
  if (!changed) return;
  Window.syncFrame(); // the frame follows what arrived: the Workroom's room comes by a patch after the first paint, and the console is decided by its columns (2026-09-22)
  markScrollEdges(main); // a replaced scroller says its hidden edges again (2026-09-22: a patch dropped them until the next render)
  Portfolio.bindCharts(); // a replaced chart keeps its readers and its navigator
  Controls.sync();
  Window.markDetail(); // law 149: the open detail's row carries the mark after every paint
  LiveTasks.refreshInspector(); // V649: local receipt words follow the locale before restoring its reading state
  Inspect.refreshFacts(); Inspect.refreshRecord(); LiveGoals.refreshReference(); Inspect.reopenFromAddress(); // round 67: the inspector follows the object
  restoreSurface(saved); restoreSurface(overlay, {scroll: false}); // restore after local detail bodies, as in a full render
  if (typeof LiveWorkspace !== 'undefined') LiveWorkspace.afterPaint();
  Geometry.schedule();
}
/* The top-level sections of one page markup, as strings, before anything annotates them. */
function sectionMarkup(markup) {
  const parsed = document.createElement('template');
  parsed.innerHTML = String(markup);
  return [...parsed.content.children].map((el) => el.outerHTML);
}

let lastRenderedPage = null;
/* The render boundary (round 96): a page whose template throws is not a blank page and not a
 * console line alone. The card in `#main` names the failure in the product's refusal shape --
 * the state, the error's words, `Reload` and a way to Home -- and says so on the live region.
 * Reload retries the same address; no in-memory count promises to survive it. `app.failNextRender` is the test
 * hook the DOM harness sets through `AlphaLattice.failNextRender()`. */
function render() {
  try {
    if (app.failNextRender) { app.failNextRender = false; throw new Error('a test asked this render to fail'); }
    renderPage();
  } catch (error) {
    renderFailure(error, 'render');
  }
}
function renderFailure(error, where) {
  console.error('The page could not be drawn (' + where + ')', error);
  const message = String((error && error.message) || error || '');
  const live = $('#tpLive'); if (live) live.textContent = t('The page could not be drawn') + ' · ' + message;
  const main = $('#main');
  if (main) main.innerHTML = failureCard(message);
  lastRenderedPage = app.page;
}
function failureCard(message) {
  return html`${refusal({reason: html`<span class="mono">${message}</span>`}, TONE.failure, {state: 'failed', word: t('The page could not be drawn'), next: t('Reload retries this page. You can also open Home.'), action: html`${btn(t('Reload'), 'page-reload', '', 'button')}${link(t('Home'), 'overview', 'button')}`, attrs: html`data-render-failure="${app.page}"`, cls: 'section-gap'})}`;
}
/* The page's marks: the window's title and the body's page, reading and chain marks the sheets
 * key a page's material on. */
/* The list a record was opened from holds while that record is shown: the first object addressed
 * after an open or a step is the record; another object addressed after it (a link, an address,
 * Back) forgets the list (2026-09-25: `]` on a study reached by a link stepped through History rows
 * it was never in, to a Portfolio book). */
function keepListContext() {
  const c = app.listContext, id = routeObjectId();
  if (!c || !id || id === c.from) return;
  if (!c.at) c.at = id; else if (c.at !== id) app.listContext = null;
}
/* The list a press opened a record from, as History's rows record theirs (`[` / `]` and the head's pair
 * step through it): a study's own list, its rows' keys as entries (2026-09-25: the studies left
 * History's list and kept none). Opened by an address or a link, a record has no list. */
function rememberList(id, entryOf = (key) => key) {
  const from = app.pressed?.isConnected ? app.pressed.closest('.card-list, [role=list]') : null;
  const keys = from ? [...from.querySelectorAll('[data-row][data-key]')].map((r) => r.dataset.key) : [];
  app.listContext = keys.includes(id) ? {list: app.page, keys: keys.map(entryOf), index: keys.indexOf(id), from: routeObjectId(), at: ''} : null;
}
function markPage() {
  keepListContext();
  document.title = titleText();
  const body = document.body;
  body.dataset.page = app.page;
  body.dataset.reading = readingRole();
  body.dataset.workFirst = app.page;
  body.dataset.tpMode = 'closed';
  body.classList.toggle('chain-v65', ['report', 'handoff'].includes(app.page));
  body.classList.toggle('evidence-stream-v3', app.page === 'evidence-stream');
}
function renderPage() {
  const saved = preserveSurface(), overlay = preserveSurface($('#inspector'));
  const pageChanged = lastRenderedPage !== app.page;
  if(pageChanged) {
    Data.leavePage(lastRenderedPage);Data.visitPage(app.page);
    if(LiveReview.pages.has(lastRenderedPage))LiveReview.leaveReads(lastRenderedPage);
    if(LiveGoals.pages.has(lastRenderedPage))LiveGoals.leaveReads();
    if(LiveStudy.pages.has(lastRenderedPage))LiveStudy.leaveReads(lastRenderedPage);
    if(lastRenderedPage==='lab')LiveResearch.leaveReads();
  }
  if (pageChanged && lastRenderedPage === 'features') LiveFeatures.leave(); // its late answers no longer take the page
  if (pageChanged && LiveModels.pages.has(lastRenderedPage)) LiveModels.leave(); // U50: the packets are read once a visit
  if (pageChanged && LiveFeatureResearch.pages.has(lastRenderedPage)) LiveFeatureResearch.leave(); // U56: as Models
  if (pageChanged && lastRenderedPage === 'team-outputs') LiveTeam.leaveOutputs(); // U54: the session's Tasks are read once a visit
  const previousKey = Places.key();
  Places.capture();
  if (Inspect.closePeek) Inspect.closePeek(); // a hover card belongs to the page it was read on (round 59)
  Window.onNavigate(pageChanged); // the side's overlay closes; a changed page closes the inspector (round 62)

  applyTheme();
  const painting = app.page;
  markPage();

  Window.render();
  const main = $('#main');
  const content = LiveViews.page();
  // a page that moved while it painted (Conversation without a session opens Sessions) wears the
  // page it opened -- its title and its material; the dock follows in `afterRender`
  if (app.page !== painting) markPage();
  const markup = html`${content}`;
  main.innerHTML = markup;
  patched = sectionMarkup(markup); // what a live page's in-place repaint compares against

  Portfolio.syncSessionButtons();
  Portfolio.bindCharts();
  // a page change is a cut (round 94; Linear's, Codex's, Finder's): the page is readable at its
  // first frame, and what arrives later settles by itself (the skeleton's reveal, a row's arrival)
  lastRenderedPage = app.page;

  Window.afterRender();
  if (typeof LiveWorkspace !== 'undefined') LiveWorkspace.afterPaint(pageChanged);
  Controls.sync();
  Window.markDetail(); // law 149: the open detail's row carries the mark after every paint
  LiveTasks.refreshInspector(); // V649: local receipt words follow the locale before restoring its reading state
  Inspect.refreshFacts(); Inspect.refreshRecord(); LiveGoals.refreshReference(); Inspect.reopenFromAddress(); // round 67: the inspector follows the object
  if (!pageChanged) { restoreSurface(saved); restoreSurface(overlay, {scroll: false}); }
  Geometry.schedule();
  Places.restoreAfterRender(previousKey, pageChanged);
  if (app.page==='lab') queueMicrotask(LiveResearch.ready);
  if (LiveGoals.pages.has(app.page)) queueMicrotask(LiveGoals.ensure);
  if (LiveModels.pages.has(app.page)) queueMicrotask(LiveModels.ensure);
  if (LiveFeatureResearch.pages.has(app.page)) queueMicrotask(LiveFeatureResearch.ensure);
  // the review reader learns of every navigation from the route it did not write itself
  if (typeof LiveReview.observeRoute==='function') LiveReview.observeRoute();
  if (LiveReview.pages.has(app.page)) queueMicrotask(LiveReview.ensure);
  if (LiveStudy.pages.has(app.page)) queueMicrotask(LiveStudy.ensure);
}
