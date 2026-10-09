// Controlled owner answers over the real transport and router: no service, browser or latency claim.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const library = require('./workbench_library.cjs');
const appDir = process.argv[2] || path.resolve(__dirname, '../../src/alphalattice/interface/local_application/assets/workbench-source/js/app');
const finish = library.guard('read lifetime');
const tick = () => new Promise(resolve => setImmediate(resolve));
const task = '00000000-0000-0000-0000-000000000001';
const day = '2024-08-12';
const book = (id = task) => ({subject: {task_id: id, session: day, result_hash: 'fixture-result', source_kind: 'INSTALLED_RESULT'}, series: [], sessions: [day]});
const entry = id => ({entry_id: id, kind: 'INSTALLED_RESULT', task_id: task, status: 'SUCCEEDED', book: {portfolio_session: day}});
const blocked = id => ({entry_id: id, status: 'REFUSED', failure_code: 'fixture.unreadable'});

function fixture(owners = {}, initial = '#page=history', search = '') {
  let hash = initial, state = {alpha: 0};
  const requests = [], holds = [], paints = [], opens = [];
  const defaults = {
    '/api/session': () => ({session_token: 'fixture-token', workspace_id: 'fixture', research_context: {tasks: {tasks: []}}}),
    '/api/research-history': () => ({entries: [], blocked_entries: [], next_cursor: null}),
    '/api/experiments': () => ({experiments: []}),
    '/api/decisions': () => ({decisions: []}),
    '/api/workbench/portfolio': q => book(q.get('task_id')),
  };
  const c = {
    console, URLSearchParams, TextEncoder, Date, Number, Math, Set, Map, Object, Array, String, Promise, JSON, Boolean, Error,
    setTimeout: () => 0, clearTimeout() {}, queueMicrotask,
    app: {page: new URLSearchParams(initial.slice(1)).get('page') || 'history', book: '', session: '', input: '', inputs: [], tasks: [], compareOther: ''},
    window: {addEventListener() {}},
    document: {hidden: false, addEventListener() {}, body: {dataset: {page: 'history'}}},
    location: {get hash() { return hash; }, set hash(value) { hash = value; }, search,
      get href() { return 'http://127.0.0.1:1/workbench.html' + search + hash; }},
    history: {get state() { return state; }, replaceState(next, title, url) { state = next; hash = url.startsWith('#') ? url : url.slice(url.indexOf('#')); },
      pushState(next, title, url) { state = next; hash = url.startsWith('#') ? url : url.slice(url.indexOf('#')); }},
    ROUTES: {portfolio: ['Portfolio', 'Portfolio study'], compare: ['Portfolio', 'Compare'], overview: ['Workspace', 'Home'], history: ['Workspace', 'History'], factor: ['Research', 'Factor'], risk: ['Research', 'Risk'], alpha: ['Research', 'Alpha'], lab: ['Research', 'Lab']},
    Window: {render() {}, inspectorMode: () => '', closeInspector() {}},
    Inspect: {resetWindow() {}, selectObservation() {}, reopenFromAddress() {}},
    LiveReview: {pages: new Set(), routeContext: () => ({})},
    LiveStudy: {pages: new Set(), routeContext: () => ({}), open: async (...args) => { opens.push(args); }},
    LiveResearch: {ready() {}, routeContext: () => ({})},
    LiveViews: {studyFacts: () => ({}), inputState: () => ({})},
    LiveWorkspace: {dismissConfirmation() {}},
    LiveFeatures: {routeContext: () => ({})},
    readPreference: () => null, savePreference() {}, closeDialog() {}, hideToast() {}, scrollTo() {},
    t: value => value, notify() {}, stateName: value => value, stateMoving: () => false, dayOf: value => String(value).slice(0, 10),
    alertDialog() {}, btn: () => '',
    html: (parts, ...values) => parts.reduce((out, part, i) => out + part + (values[i] ?? ''), ''),
    kv: () => '', icon: () => '', hashCell: () => '', codeWords: value => String(value ?? ''),
    stateLine: () => '', codeRef: () => '', refusal: () => '', explainCode: () => '',
    fetch: async (url, init = {}) => {
      const query = new URL(url, 'http://127.0.0.1:1');
      const request = {url, method: init.method || 'GET', textReads: 0}; requests.push(request);
      const hold = holds.find(h => !h.entered && h.matches(url));
      let reply;
      if (hold) { hold.entered = true; reply = await hold.promise; }
      else reply = {value: (owners[query.pathname] || defaults[query.pathname])?.(query.searchParams, init)};
      if (reply.error) throw reply.error;
      if (reply.value === undefined && reply.text === undefined) throw Error('Unexpected controlled route: ' + query.pathname);
      return {ok: reply.ok ?? true, status: reply.status ?? 200, headers: {get: () => 'application/json'},
        text: async () => { request.textReads++; return reply.text ?? JSON.stringify(reply.value); }};
    },
  };
  vm.createContext(library.into(c, appDir));
  for (const name of ['data.js', 'router.js']) vm.runInContext(fs.readFileSync(path.join(appDir, name), 'utf8'), c, {filename: name});
  vm.runInContext('render = () => globalThis.paints.push(app.page); patchMain = render; globalThis.D=Data; globalThis.go=navigate;', Object.assign(c, {paints}));
  function hold(matches) {
    let release;
    const h = {matches: typeof matches === 'function' ? matches : url => url === matches,
      entered: false, promise: new Promise(resolve => { release = resolve; }), release: value => release(value)};
    holds.push(h); return h;
  }
  return {c, D: c.D, requests, paints, opens, hold};
}
async function entered(hold) {
  for (let i = 0; i < 30 && !hold.entered; i++) await tick();
  assert.ok(hold.entered, 'the controlled owner request must start');
}
const paths = f => f.requests.map(r => r.url);
const prefixCount = (f, prefix) => paths(f).filter(p => p.startsWith(prefix)).length;

async function wire() {
  const f = fixture();
  const same = f.hold('/api/probe?selector=a'), other = f.hold('/api/probe?selector=b');
  const a = f.D.read('/api/probe?selector=a'), joined = f.D.read('/api/probe?selector=a'), b = f.D.read('/api/probe?selector=b');
  await entered(same); await entered(other);
  assert.equal(f.requests.length, 2, 'only identical in-flight GET keys join');
  same.release({value: {owner: 'a'}}); other.release({value: {owner: 'b'}});
  const values = await Promise.all([a, joined, b]);
  assert.equal(values[0].owner, 'a'); assert.equal(values[1].owner, 'a'); assert.equal(values[2].owner, 'b');
  assert.ok(f.requests.every(r => r.textReads === 1), 'the raw response is consumed once');
  const fresh = f.hold('/api/probe?selector=a'), again = f.D.read('/api/probe?selector=a');
  await entered(fresh); fresh.release({value: {owner: 'fresh'}}); assert.equal((await again).owner, 'fresh');
  assert.equal(f.requests.length, 3, 'a settled GET is read again');

  const unavailable = f.hold('/api/unavailable');
  const strict = f.D.read('/api/unavailable').then(() => null, error => error);
  const descriptive = f.D.read('/api/unavailable', true);
  await entered(unavailable); unavailable.release({value: {status: 'UNAVAILABLE', failure_code: 'fixture.missing'}});
  assert.equal((await strict).body.failure_code, 'fixture.missing');
  assert.equal((await descriptive).status, 'UNAVAILABLE', 'each caller applies its own refusal policy');
  assert.equal(prefixCount(f, '/api/unavailable'), 1);
  const named=f.hold('/api/refused'), refusal=f.D.read('/api/refused').catch(error=>error);
  await entered(named);named.release({value:{status:'REFUSED',refusal_code:'fixture.named',detail:'Exact owner detail'}});
  assert.equal((await refusal).message,'fixture.named: Exact owner detail','transport uses the same refusal fields as page readers');

  for (const reply of [{error: Error('fixture disconnected')}, {text: 'unreadable JSON'}]) {
    const held = f.hold('/api/retry');
    const first = f.D.read('/api/retry').then(() => null, error => error);
    const peer = f.D.read('/api/retry').then(() => null, error => error);
    await entered(held); held.release(reply);
    assert.ok(await first); assert.ok(await peer);
    const retry = f.hold('/api/retry'), next = f.D.read('/api/retry');
    await entered(retry); retry.release({value: {status: 'READ'}}); assert.equal((await next).status, 'READ');
  }
  await f.D.connect();
  const postA = f.hold('/api/mutate'), postB = f.hold('/api/mutate');
  const mutations = [f.D.post('/api/mutate', {choice: 'same'}), f.D.post('/api/mutate', {choice: 'same'})];
  await entered(postA); await entered(postB);
  postA.release({value: {status: 'FIRST'}}); postB.release({value: {status: 'SECOND'}});
  assert.deepEqual((await Promise.all(mutations)).map(v => v.status), ['FIRST', 'SECOND']);
  assert.equal(f.requests.filter(r => r.method === 'POST').length, 2, 'mutations never coalesce');
}

async function historyPages() {
  let first = {entries: [entry('result:first')], blocked_entries: [], next_cursor: 'c1'};
  const f = fixture({'/api/research-history': () => first}); await f.D.connect();
  const older = f.hold(url => url.includes('history_cursor=c1'));
  const one = f.D.refreshHistory(true), joined = f.D.refreshHistory(true);
  await entered(older);
  assert.equal(prefixCount(f, '/api/research-history?history_limit=50&history_cursor=c1'), 1);
  assert.equal(f.D.historyLoading, true);
  older.release({value: {entries: [entry('result:first'), entry('result:older'), entry('result:older')],
    blocked_entries: [blocked('blocked:a'), blocked('blocked:a')], next_cursor: 'c2'}});
  await Promise.all([one, joined]);
  assert.equal(f.D.history().length, 2); assert.equal(f.D.historyRefusals().length, 1); assert.equal(f.D.historyLoading, false);
  const next = f.hold(url => url.includes('history_cursor=c2')), paging = f.D.refreshHistory(true);
  await entered(next); next.release({value: {entries: [entry('result:older')], blocked_entries: [blocked('blocked:a')], next_cursor: 'c1'}});
  await paging; assert.equal(f.D.hasMoreHistory(), false, 'an already consumed cursor cannot restart paging');

  first = {entries: [entry('result:renewed')], blocked_entries: [], next_cursor: 'old'}; await f.D.refreshHistory();
  const stale = f.hold(url => url.includes('history_cursor=old')), late = f.D.refreshHistory(true);
  await entered(stale);
  first = {entries: [entry('result:new')], blocked_entries: [blocked('blocked:new')], next_cursor: 'new'};
  await f.D.refreshHistory();
  stale.release({value: {entries: [entry('result:stale')], blocked_entries: [blocked('blocked:stale')], next_cursor: 'rollback'}});
  await late;
  assert.equal(f.D.history().length, 1); assert.equal(f.D.history()[0].raw.entry_id, 'result:new');
  assert.equal(f.D.historyRefusals().length, 1); assert.equal(f.D.historyRefusals()[0].entry_id, 'blocked:new');
  const current = f.hold(url => url.includes('history_cursor=new')), currentPage = f.D.refreshHistory(true);
  await entered(current); current.release({value: {entries: [], blocked_entries: [], next_cursor: null}}); await currentPage;
  assert.ok(!paths(f).some(url => url.includes('history_cursor=rollback')), 'late pagination cannot roll back the refreshed cursor');
}

async function exactOpen() {
  for (const back of [false, true]) {
    const f = fixture();
    const held = f.hold(url => url.includes('history_entry_id=result%3Aexact'));
    const opening = f.D.openEntry('result:exact'); await entered(held);
    f.c.go('overview'); if (back) f.c.go('history');
    const address = f.c.location.hash;
    held.release({value: {entries: [entry('result:exact')]}}); await opening;
    assert.equal(f.c.app.page, back ? 'history' : 'overview'); assert.equal(f.c.location.hash, address);
    assert.equal(prefixCount(f, '/api/workbench/portfolio?'), 0, 'a late exact lookup cannot initiate the abandoned result reader');
  }
}

async function portfolioOpen() {
  const f = fixture({}, '#page=portfolio&book=' + task + '&session=' + day);
  const session = f.hold(url => url.startsWith('/api/session?')), connecting = f.D.connect();
  await entered(session);
  const portfolio = f.hold(url => url.startsWith('/api/workbench/portfolio?'));
  const routeOpen = f.D.openPortfolio(task, day), peer = f.D.openPortfolio(task, day);
  await entered(portfolio); assert.equal(prefixCount(f, '/api/workbench/portfolio?'), 1);
  session.release({value: {session_token: 'fixture-token', workspace_id: 'fixture', research_context: {tasks: {tasks: []}}}});
  await tick(); await tick();
  assert.equal(prefixCount(f, '/api/workbench/portfolio?'), 1, 'boot and route reopening share the pending Portfolio reader');
  portfolio.release({value: book()}); await Promise.all([connecting, routeOpen, peer]);
  assert.equal(f.D.status, 'ready'); assert.equal(f.c.app.page, 'portfolio');
  await f.D.openPortfolio(task, day); assert.equal(prefixCount(f, '/api/workbench/portfolio?'), 2, 'a later open revalidates the owner');

  const launch = fixture({}, '', '?history=result%3Acold');
  const discovery = launch.hold(url => url.includes('history_entry_id=result%3Acold'));
  const cold = launch.D.connect(); await entered(discovery);
  discovery.release({value: {entries: [entry('result:cold')]}}); await cold;
  assert.equal(launch.c.app.page, 'portfolio', 'cold exact launch owns its intentional navigation');
  assert.equal(prefixCount(launch, '/api/workbench/portfolio?'), 1);
}

async function taskStudyRead() {
  for (const [kind, page] of [['factor.screening-development', 'factor'], ['risk.covariance-development', 'risk']]) {
    for (const declared of [true, false]) {
      const body = {status: 'EXPERIMENT_PUBLISHED', task_id: task, program: {kind},
        research_input_id: 'fixture-input', input_binding_hash: 'fixture-binding', document: {}, result: {}};
      const f = fixture({
        '/api/experiments': () => ({experiments: declared ? [{task_id: task, kind}] : []}),
        '/api/experiments/readback': () => body,
        '/api/experiments/curation': () => ({choices: [], decisions: [], limitations: [], inputs: []}),
      }, '#page=history');
      library.into(f.c, appDir);
      for (const name of ['live-study.js', 'live-tasks.js']) vm.runInContext(fs.readFileSync(path.join(appDir, name), 'utf8'), f.c, {filename: name});
      vm.runInContext('globalThis.LT=LiveTasks;globalThis.LS=LiveStudy;', f.c);
      await f.D.connect(); f.D.setTasks([{task_id: task, task_kind: 'research_experiment', lifecycle: 'SUCCEEDED'}]);
      f.requests.length = 0;
      await f.c.LT.openResult(task);
      assert.equal(f.c.app.page, page);
      assert.equal(prefixCount(f, '/api/experiments/readback?'), 1,
        declared ? 'metadata picks the reader; that reader owns one strict read' : 'kind discovery passes its exact verified body to the reader');
      assert.equal(prefixCount(f, '/api/experiments/curation?'), page === 'factor' ? 1 : 0);
      assert.equal(f.c.LS.routeContext(page).study, task);
    }
  }
}

(async () => {
  await wire(); await historyPages(); await exactOpen(); await portfolioOpen(); await taskStudyRead();
  finish();
})().catch(error => { console.error(error); process.exitCode = 1; });
