// F30/U173: labelled public History/Portfolio/Review answers, actual consumers and words.
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const library = require('./workbench_library.cjs');
const [appDir, which] = process.argv.slice(2);
const read = name => fs.readFileSync(path.join(appDir, name + '.js'), 'utf8');
const finish = library.guard('installed book names ' + which);
const UNKNOWN = 'QA_UNDECLARED_STRATEGY_PACKAGE', PIN = 'b'.repeat(64);
const known = Object.entries(library(appDir).LABELS).find(([, label]) => label.kind === 'strategy');
assert.ok(known, 'the known specimen uses an installed strategy label');
const entries = [UNKNOWN, null, undefined, known[0]].map((packageId, i) => ({
  entry_id: 'result:' + String(i + 1).repeat(64), task_id: '00000000-0000-0000-0000-00000000000' + (i + 1),
  kind: 'INSTALLED_RESULT', status: 'SUCCEEDED', recorded_at: '2026-08-01T12:00:00Z',
  ...(packageId === undefined ? {} : {strategy_package_id: packageId}),
  book: {result_hash: String(i + 1).repeat(64)},
}));
const words = markup => String(markup).replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
const unescape = text => text.replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&');
const pickerChoices = markup => [...String(markup).matchAll(/<button\b([^>]*)>([\s\S]*?)<\/button>/g)]
  .filter(match => match[1].includes('role="option"'))
  .map(match => ({value: unescape(match[1].match(/data-value="([^"]*)"/)[1]),
    title: unescape(match[1].match(/data-picker-title="([^"]*)"/)[1]), text: words(match[2])}));

function fixture(viewsSource = read('live-views')) {
  const route = new URLSearchParams({page: 'history'}), requests = [], rows = [], heads = [], inspectors = [];
  const c = {console, URL, URLSearchParams, TextEncoder, setTimeout: () => 0, clearTimeout() {}, queueMicrotask() {},
    app: {page: 'history', historyQuery: '', historyKind: 'all', historySort: 'source', book: '', session: ''},
    window: {AlphaStaticMark: require(path.resolve(appDir, '../engines/static-mark.js'))},
    document: {documentElement: {}, body: {dataset: {}}, querySelector: () => null},
    location: {search: '', get hash() {return '#' + route;}, href: 'http://127.0.0.1:1/workbench.html'},
    render() {}, patchMain() {}, closeDialog() {}, clone: structuredClone, $: () => null, $$: () => [],
    readPreference: () => null, savePreference() {}, copyText() {}, routeObjectId: () => '',
    routeUrl: (page, extra = {}) => '#' + new URLSearchParams({page, ...extra}), listUrl: page => '#page=' + page,
    replaceHash: patch => {for (const [key, value] of Object.entries(patch)) {
      if (value === '' || value == null) route.delete(key); else route.set(key, String(value));
    }}, hashParams: () => new URLSearchParams(route), objectEntry() {},
    PAGES: {}, ACTIONS: {}, ROUTES: {history: ['Workspace', 'History'], books: ['Evidence', 'Books'], evidence: ['Evidence', 'Evidence']},
    Window: {render() {}, inspectorMode: () => '', openInspector: value => inspectors.push(value), focusAfterPaint() {}, clearDetailTrail() {}},
    Inspect: {resetWindow() {}, selectObservation() {}, reopenFromAddress() {}},
    LiveResearch: {ready() {}}, LiveStudy: {pages: new Set()}, LiveTasks: {open() {}, select() {}},
    LiveActivity: {logLines: () => []},
    html: () => '', t: key => key,
    fetch: async url => {
      requests.push(url); const q = new URL(url, 'http://127.0.0.1:1'); let value;
      if (q.pathname === '/api/session') value = {session_token: 'labelled-fixture', workspace_id: 'book-name-fixture', research_context: {tasks: {tasks: []}}};
      else if (q.pathname === '/api/research-history') value = {entries, blocked_entries: [], next_cursor: null};
      else if (q.pathname === '/api/decisions') value = {decisions: []};
      else if (q.pathname === '/api/experiments') value = {experiments: []};
      else if (q.pathname === '/api/workbench/portfolio') value = {subject: {task_id: q.searchParams.get('task_id'), session: '2026-08-01', source_kind: 'INSTALLED_RESULT'}, series: []};
      else if (q.pathname === '/api/evidence-cro') value = {state: 'EVIDENCE_AUTHORITY_NOT_ADMITTED', book: {result_hash: q.searchParams.get('result_hash')}, eligible_versions: [], issuer_rows: [], next_requests: {}};
      else throw Error('Unexpected fixture read: ' + url);
      return {ok: true, headers: {get: () => 'application/json'}, text: async () => JSON.stringify(value)};
    }};
  vm.createContext(library.into(c, appDir));
  for (const name of ['html', '../data/zh', 'i18n', 'icons', 'status', 'components']) vm.runInContext(read(name), c);
  // Chrome and grouping are presentation ports; each actual consumer and row/picker builder stays real.
  c.objectHead = (title, lede, action, state, tools, options = {}) => {
    heads.push({title, options}); return vm.runInContext('html', c)`<h1>${title}</h1>${options.switcher || ''}${options.top || ''}${options.subject || ''}`;
  };
  const lobby = vm.runInContext('Lobby', c);
  lobby.listed = () => c.D.history();
  lobby.render = (name, spec) => {
    rows.length = 0; const html = vm.runInContext('html', c);
    return html`${spec.items.map(item => {const markup = spec.row(item, {props: {}, group: 'time'}); rows.push({item, markup}); return markup;})}`;
  };
  vm.runInContext(viewsSource, c);
  for (const name of ['data', 'pages-history', 'live-review']) vm.runInContext(read(name), c);
  vm.runInContext('globalThis.D=Data;globalThis.V=LiveViews;globalThis.H=History;globalThis.R=LiveReview;globalThis.locale=I18N;', c);
  return {c, route, requests, rows, heads, inspectors};
}
function priorViews() {
  const current = read('live-views'), former = current.replace(
    "words.push(packageId && (labelOf(packageId) || declaredCodeWord(packageId)) ? codeWords(packageId) : t('Installed strategy result'));",
    "words.push(entry?.strategy_package_id ? codeWords(entry.strategy_package_id) : '');");
  assert.notEqual(former, current, 'the negative control restores the retired package-name reader');
  return former;
}
async function historyNames() {
  const f = fixture(); await f.c.D.connect(); await f.c.D.refreshExperiments();
  for (const lang of ['en', 'zh-CN']) {
    f.c.locale.set(lang); f.c.H.page();
    const fallback = lang === 'en' ? 'Installed strategy result' : f.c.locale.t('Installed strategy result');
    for (const {item, markup} of f.rows) {
      const title = words(f.c.V.nameOf(item).name);
      if (item.raw.strategy_package_id !== known[0]) assert.equal(title, fallback, 'absent or undeclared package uses the existing kind once');
      else assert.ok(title.endsWith(lang === 'en' ? known[1].title : known[1].title_zh), title);
      assert.ok(!words(markup).includes(lang === 'en' ? 'Word not declared' : f.c.locale.t('Word not declared')));
      assert.ok(String(markup).includes('data-value="' + item.id + '"'), 'History row retains its exact saved-object opener');
      f.c.V.existing(item.id); const card = f.inspectors.at(-1);
      assert.equal(words(card.title), title); assert.equal(words(card.readHeader().title), title);
      if (item.raw.strategy_package_id === UNKNOWN) {
        assert.ok(String(card.body).includes('data-tip="' + UNKNOWN + '"'), 'the real coded fact retains the exact undeclared package');
        assert.equal(item.summary, UNKNOWN); assert.equal(item.raw.strategy_package_id, UNKNOWN);
      }
    }
  }
  const target = entries[0]; await f.c.H.openItem(target.entry_id);
  const request = f.requests.find(url => url.startsWith('/api/workbench/portfolio?'));
  assert.equal(new URL(request, 'http://fixture').searchParams.get('task_id'), target.task_id);
  assert.equal(f.route.get('book'), target.task_id, 'History opens the exact owner Task despite its absent name');
  const old = fixture(priorViews()); await old.c.D.connect(); old.c.H.page();
  assert.ok(words(old.rows[0].markup).includes('Word not declared'), 'the actual History row reproduces the retired naming defect');
}
async function evidenceBookChoices() {
  const f = fixture(); await f.c.D.connect();
  for (const lang of ['en', 'zh-CN']) {
    f.c.locale.set(lang); f.c.app.page = 'books'; f.c.R.page();
    for (const {item, markup} of f.rows) {
      const entry = entries.find(value => value.book.result_hash === item.v.selector.result_hash);
      assert.ok(entry); assert.equal(words(item.v.name), words(f.c.D.history().find(row => row.id === entry.entry_id).words));
      const href = unescape(String(markup).match(/href="([^"]+)"/)[1]);
      assert.deepEqual(JSON.parse(new URLSearchParams(href.slice(1)).get('review_selector')), entry.book);
    }
    await f.c.R.open(entries[0].book, PIN, 'evidence'); const markup = String(f.c.R.page()), head = f.heads.at(-1);
    assert.equal(words(head.title), words(f.c.D.history()[0].words), 'Evidence reads the same installed name');
    assert.ok(!words(head.title).includes('Word not declared'));
    const choices = pickerChoices(head.options.switcher);
    for (const entry of entries) {
      const option = choices.find(choice => choice.value === JSON.stringify(entry.book));
      assert.ok(option, 'actual reviewBook picker retains every exact selector');
      assert.equal(option.title, f.c.D.history().find(row => row.id === entry.entry_id).words,
        'actual reviewBook picker reads the same fallback or installed label');
    }
    assert.ok(f.requests.some(url => {
      const q = new URL(url, 'http://fixture'); return q.pathname === '/api/evidence-cro' && q.searchParams.get('result_hash') === entries[0].book.result_hash && q.searchParams.get('review_publication_hash') === PIN;
    }), 'Evidence reads the exact result and pinned publication');
    await f.c.D.openPortfolio(entries[0].task_id, '', 'compare');
    const comparison = String(f.c.V.comparePage());
    const options = pickerChoices(comparison);
    for (const entry of entries.slice(1)) {
      const option = options.find(choice => choice.value === entry.task_id);
      assert.ok(option, 'actual Compare picker retains the exact Task id');
      const label = option.title, expected = words(f.c.V.nameOf(f.c.D.history().find(row => row.task_id === entry.task_id)).name);
      assert.ok(label.startsWith(expected), label);
      assert.equal(label.split(f.c.locale.t('Installed strategy result')).length - 1, 1,
        'Compare names the installed kind once, including absent package names');
    }
  }
  const old = fixture(priorViews()); await old.c.D.connect(); old.c.app.page = 'books'; old.c.R.page();
  assert.equal(words(old.rows[0].item.v.name), 'Word not declared', 'the actual Books producer reproduces the retired fallback');
  const current = read('live-views'), former = current.replace(
    'return `${nameOf(row).name} · ${row.reference}${row.holdingsSession',
    "return `${t(row.name)}${row.words || row.summary ? ' · ' + (row.words || row.summary) : ''} · ${row.reference}${row.holdingsSession");
  assert.notEqual(former, current, 'the negative control restores the independent retired Compare label');
  const duplicate = fixture(former); await duplicate.c.D.connect(); await duplicate.c.D.openPortfolio(entries[0].task_id, '', 'compare');
  assert.ok(words(duplicate.c.V.comparePage()).includes('Installed strategy result · Installed strategy result'),
    'the real Compare picker reproduces its former duplicated kind');
}
(async () => {
  assert.ok(['history-names', 'evidence-book-choices'].includes(which));
  await (which === 'history-names' ? historyNames() : evidenceBookChoices()); finish();
})().catch(error => {console.error(error); process.exitCode = 1;});
