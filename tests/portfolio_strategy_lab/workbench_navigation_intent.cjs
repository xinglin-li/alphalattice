// Controlled route boundaries using the checkout's Data, router, Task, Review,
// Research and Activity modules. Owner answers are held explicitly; there is no
// HTTP server, elapsed-time assertion, model call or browser dependency.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const project = path.resolve(__dirname, '..', '..');
const appDir = process.argv[2] || path.join(project, 'src/alphalattice/interface/local_application/assets/workbench-source/js/app');
const library = require('./workbench_library.cjs');
const finish = library.guard('workbench navigation intent boundaries');
const flush = () => new Promise(resolve => setImmediate(resolve));
const task = '00000000-0000-0000-0000-000000000001';
const other = '00000000-0000-0000-0000-000000000002';
const day = '2024-08-12';
const binding = 'b'.repeat(64);
const plan = 'a'.repeat(64);
const results = [];
const portfolio = (kind = 'RESEARCH_EXPERIMENT') => ({subject: {task_id: task, session: day, input_id: 'fixture-input', result_hash: 'saved-result', source_kind: kind}, series: [], sessions: [day]});
const controls = () => ({template: {experiment: {kind: 'factor.screening-development'}}, yaml: 'experiment:\n  kind: factor.screening-development\n', research_input_id: 'fixture-input', input_binding_hash: binding, controls: []});

function fixture(responses = {}, initial = '#page=evidence', inspector = '') {
  let hash = initial, state = {alpha: 0}, mode = inspector;
  const requests = [], gates = new Map(), scheduled = [], opens = [];
  const c = {
    console, URL, URLSearchParams, Date, Number, Math, Set, Map, Object, Array, String, Promise, JSON, Boolean, Error,
    setTimeout, clearTimeout, queueMicrotask: callback => scheduled.push(callback),
    app: {page: new URLSearchParams(initial.slice(1)).get('page') || 'evidence', input: '', book: '', session: '', inputs: [], tasks: [], compareOther: '', yaml: '', mode: 'yaml'},
    document: {body: {dataset: {}}, querySelector: () => null}, window: {},
    location: {get hash() {return hash;}, search: '', get href() {return 'http://127.0.0.1:1/workbench.html' + hash;}},
    history: {get state() {return state;}, replaceState(next, title, url) {state = next; hash = url.startsWith('#') ? url : url.slice(url.indexOf('#'));}, pushState(next, title, url) {state = next; hash = url.startsWith('#') ? url : url.slice(url.indexOf('#'));}},
    ROUTES: Object.fromEntries(['evidence', 'evidence-stream', 'evidence-reading', 'handoff', 'lab', 'data', 'overview', 'tasks', 'portfolio', 'compare', 'history', 'factor'].map(page => [page, ['Workspace', page]])),
    Window: {render() {}, inspectorMode: () => mode, closeInspector() {mode = ''; vm.runInContext("replaceHash({task:''});", c);}},
    Inspect: {resetWindow() {}, selectObservation() {}, reopenFromAddress() {}},
    LiveWorkspace: {dismissConfirmation() {}}, LiveFeatures: {routeContext: () => ({})},
    LiveStudy: {pages: new Set(), routeContext: () => ({}), open: async (id, page) => {opens.push({id, page}); c.app.page = page;}},
    LiveViews: {studyFacts: () => ({}), inputState: () => ({}), savedObjectLink: () => '', existing() {}},
    Lab: {markDraftChanged() {}},
    clone: value => JSON.parse(JSON.stringify(value)),
    readPreference: () => null, savePreference() {}, closeDialog() {}, hideToast() {}, scrollTo() {},
    t: (value, args = {}) => String(value).replace(/\{(\w+)\}/g, (_, key) => String(args[key] ?? '')),
    html: (parts, ...values) => parts.reduce((out, part, index) => out + part + (values[index] ?? ''), ''),
    notify() {}, alertDialog() {}, openDialog() {}, btn: () => '', dayOf: value => String(value).slice(0, 10),
    fetch: async (url, init = {}) => {
      assert.ok(!init.method || init.method === 'GET', 'navigation probes submit no mutation');
      const parsed = new URL(url, 'http://127.0.0.1:1');
      requests.push({path: parsed.pathname, query: parsed.search, page: c.app.page});
      const gate = gates.get(parsed.pathname);
      if (gate) {gates.delete(parsed.pathname); gate.entered = true; await gate.promise;}
      const body = responses[parsed.pathname]?.(parsed.searchParams);
      assert.notEqual(body, undefined, 'unmapped owner read: ' + url);
      return {ok: true, text: async () => JSON.stringify(body)};
    },
  };
  library.context(c, appDir);
  vm.runInContext(library.readingSource(appDir), c);
  vm.runInContext(library.explainCodeSource(appDir), c);
  for (const name of ['status', 'data', 'router', 'live-review', 'live-research', 'live-activity', 'live-tasks']) {
    vm.runInContext(fs.readFileSync(path.join(appDir, name + '.js'), 'utf8'), c, {filename: name + '.js'});
  }
  vm.runInContext('render=()=>{};patchMain=render;globalThis.D=Data;globalThis.Review=LiveReview;globalThis.R=LiveResearch;globalThis.T=LiveTasks;globalThis.go=navigate;globalThis.detail=pushDetail;', c);
  function gate(endpoint) {
    let release;
    const hold = {entered: false, promise: new Promise(resolve => {release = resolve;}), release: () => release()};
    gates.set(endpoint, hold);
    return hold;
  }
  return {c, requests, opens, scheduled, gate};
}
async function entered(hold) {
  for (let n = 0; n < 30 && !hold.entered; n++) await flush();
  assert.ok(hold.entered, 'the controlled owner read must be reached');
}
function result(name, f, extra = {}) {results.push({case: name, page: f.c.app.page, hash: f.c.location.hash, ...extra});}
function researchTask(f) {f.c.D.setTasks([{task_id: task, task_kind: 'research_experiment', lifecycle: 'SUCCEEDED'}]);}
const metadata = {'/api/experiments': () => ({experiments: [{task_id: task, kind: 'factor.screening-development', lifecycle: 'SUCCEEDED'}]})};
const replay = {
  '/api/results': () => ({results: [{task_id: task, result_hash: 'saved-result'}]}),
  '/api/report': () => ({result_hash: 'saved-result', used_by_task_ids: [task]}),
  '/api/research-history': () => ({entries: [{entry_id: 'result:saved-result', task_id: task, kind: 'INSTALLED_RESULT', status: 'SUCCEEDED', book: {portfolio_session: day}}], next_cursor: null}),
  '/api/workbench/portfolio': () => portfolio('INSTALLED_RESULT'),
};

(async () => {
  // Review uses its own pushed step rather than the generic navigate function.
  {
    const f = fixture(metadata); researchTask(f);
    const hold = f.gate('/api/experiments'), opening = f.c.T.openResult(task);
    await entered(hold); f.c.Review.step('evidence-reading'); hold.release();
    assert.equal(await opening, false);
    assert.equal(f.c.app.page, 'evidence-reading');
    assert.equal(new URLSearchParams(f.c.location.hash.slice(1)).get('page'), 'evidence-reading');
    assert.equal(f.opens.length, 0);
    result('held Task open, direct Review step', f);
  }
  // A Shared PLAN writes Lab directly and replaces its hash; the old Task open
  // still belongs to the earlier page and cannot adopt that newer route.
  {
    const f = fixture({...metadata, '/api/experiments/preview': () => ({plan_hash: plan, status: 'MISSING'})}); researchTask(f);
    const hold = f.gate('/api/experiments'), opening = f.c.T.openResult(task);
    await entered(hold); await f.c.R.inspectShared(plan); hold.release();
    assert.equal(await opening, false); assert.equal(f.c.app.page, 'lab');
    assert.equal(new URLSearchParams(f.c.location.hash.slice(1)).get('plan'), plan);
    assert.equal(f.opens.length, 0);
    result('held Task open, direct Shared PLAN', f);
  }
  // Facts and Record inspect the same addressed object, so their pushed entries
  // must leave an in-flight comparison eligible to settle.
  {
    const f = fixture({'/api/experiments/compare': () => ({left: {task_id: task}, right: {task_id: other}})}, '#page=compare&book=' + task + '&session=' + day);
    await f.c.D.load(portfolio());
    const hold = f.gate('/api/experiments/compare'), reading = f.c.D.compare(other);
    await entered(hold); const intent = f.c.D.navigationIntent();
    f.c.detail({facts: '1'});
    assert.equal(f.c.D.comparisonStatus, 'loading'); assert.equal(f.c.D.navigationCurrent(intent), true);
    f.c.detail({facts: '', record: '1'});
    assert.equal(f.c.D.comparisonStatus, 'loading'); assert.equal(f.c.D.navigationCurrent(intent), true);
    hold.release(); await reading;
    assert.equal(f.c.D.comparisonStatus, 'ready');
    assert.equal(f.requests.filter(r => r.path === '/api/experiments/compare').length, 1);
    result('Facts and Record preserve pending comparison', f, {comparison: f.c.D.comparisonStatus});
  }
  // useInput owns its synchronous Lab/input address commit, then protects only
  // the completion: normal answers finish that draft; late ones cannot restore Lab.
  {
    const f = fixture({'/api/experiments/controls': controls}, '#page=data');
    await f.c.R.useInput(JSON.stringify(['fixture-input', binding]));
    const q = new URLSearchParams(f.c.location.hash.slice(1));
    assert.equal(f.c.app.page, 'lab'); assert.equal(q.get('research_input'), 'fixture-input');
    assert.equal(q.get('input_binding'), binding); assert.equal(q.get('experiment_kind'), 'factor.screening-development');
    assert.ok(f.c.app.yaml.includes('factor.screening-development'));
    assert.equal(f.requests.filter(r => r.path === '/api/experiments/controls').length, 1);
    result('normal useInput finishes its owned Lab route', f);
  }
  {
    const f = fixture({'/api/experiments/controls': controls}, '#page=data');
    const hold = f.gate('/api/experiments/controls'), reading = f.c.R.useInput(JSON.stringify(['fixture-input', binding]));
    await entered(hold); f.c.go('overview'); const address = f.c.location.hash;
    hold.release(); await reading;
    assert.equal(f.c.app.page, 'overview'); assert.equal(f.c.location.hash, address);
    result('late useInput keeps the chosen Home route', f);
  }
  // Installed replay discovery dismisses the Task inspector before waiting for
  // History. That synchronous owned dismissal cannot invalidate its own open.
  for (const leave of [false, true]) {
    const f = fixture(replay, '#page=tasks&task=' + task, 'task');
    f.c.D.setTasks([{task_id: task, task_kind: 'portfolio_public_development_replay', lifecycle: 'SUCCEEDED'}]);
    const hold = f.gate('/api/research-history'), opening = f.c.T.openResult(task);
    await entered(hold);
    assert.equal(new URLSearchParams(f.c.location.hash.slice(1)).get('task'), null, 'owned close clears the Task address');
    if (leave) f.c.go('overview');
    const address = f.c.location.hash; hold.release(); const opened = await opening;
    if (leave) {
      assert.equal(opened, false); assert.equal(f.c.app.page, 'overview'); assert.equal(f.c.location.hash, address);
      assert.equal(f.requests.filter(r => r.path === '/api/workbench/portfolio').length, 0);
    } else {
      assert.equal(f.c.app.page, 'portfolio'); assert.equal(f.c.D.status, 'ready');
      assert.equal(new URLSearchParams(f.c.location.hash.slice(1)).get('book'), task);
      assert.equal(new URLSearchParams(f.c.location.hash.slice(1)).get('session'), day);
      assert.equal(f.requests.filter(r => r.path === '/api/workbench/portfolio').length, 1);
    }
    result(leave ? 'late installed replay after owned close' : 'normal installed replay after owned close', f);
  }
  // An exact History entry hands the owned numeric ticket to Portfolio: its own
  // page/book/session writes must not reject its continuation.
  {
    const f = fixture(replay, '#page=overview');
    await f.c.D.openEntry('result:saved-result');
    assert.equal(f.c.app.page, 'portfolio'); assert.equal(f.c.D.status, 'ready');
    assert.equal(new URLSearchParams(f.c.location.hash.slice(1)).get('session'), day);
    assert.equal(f.requests.filter(r => r.path === '/api/workbench/portfolio').length, 1);
    result('exact History handoff owns Portfolio route enrichment', f);
  }
  // A Review projection can synchronously supply its default book and one
  // prepared packet. Those owner enrichments are part of this read's commit;
  // they must preserve the automatic follow and cannot rewrite a later Home.
  for (const selector of [null, {result_hash: 'chosen-result'}]) for (const leave of [false, true]) {
    const f = fixture({
      '/api/evidence-cro': q => ({state: 'EVIDENCE_REFRESH_IN_PROGRESS', task_id: task,
        book: {result_hash: q.get('result_hash') || 'default-result'},
        next_requests: {packet: {operation: 'EVIDENCE_PACKET', task_id: other, evidence_unit_id: 'fixture-unit'}}}),
      '/api/tasks/recovery': () => ({task_id: task, task_kind: 'alternative_evidence.document_intelligence', lifecycle: 'RUNNING', stages: [], verified_stage_count: 0, total_stage_count: 1}),
    }, '#page=evidence-stream');
    const hold = f.gate('/api/evidence-cro'), reading = f.c.Review.open(selector, '', 'evidence-stream');
    await entered(hold); if (leave) f.c.go('overview');
    const address = f.c.location.hash; hold.release(); await reading; await flush();
    if (leave) {
      assert.equal(f.c.app.page, 'overview'); assert.equal(f.c.location.hash, address, 'a late packet/default selector cannot rewrite Home');
      assert.equal(f.c.Review.context().work, null);
      assert.equal(f.requests.filter(r => r.path === '/api/tasks/recovery').length, 0);
    } else {
      assert.equal(f.c.Review.context().work, task, 'the accepted projection follows its own running Task immediately');
      assert.equal(f.c.Review.context().prepared_task, other);
      const q = new URLSearchParams(f.c.location.hash.slice(1));
      assert.equal(JSON.parse(q.get('review_selector')).result_hash, selector?.result_hash || 'default-result');
      assert.equal(f.requests.filter(r => r.path === '/api/tasks/recovery').length, 1);
    }
    result((leave ? 'late' : 'normal') + ' Review ' + (selector ? 'explicit book' : 'default book') + ' and prepared packet', f);
  }
  console.log(JSON.stringify({cases: results.length, failures: 0, results}, null, 2));
  finish();
})().catch(error => {console.error(error); process.exitCode = 1;});

// alpha comparison route keeps the newer exact selection when an old read finishes.
{
const library=require('./workbench_library.cjs'),complete=library.guard("alpha_comparison_route_keeps_the_newer_exact_selection_when_an_old_read_finishes");
const _appDir=require('node:path').resolve(process.argv[2]),_project=require('node:path').resolve(__dirname,'../..');
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
let route=new URLSearchParams(
  'page=alpha&study=left&alpha_left_task=left&alpha_left_candidate=left-candidate&'+
  'alpha_right_task=old-right&alpha_right_candidate=old-candidate');
let finishOld;
const published=(candidate)=>({
  status:'EXPERIMENT_PUBLISHED',program:{kind:'alpha.model-development'},
  result:{candidates:[{candidate_id:candidate,status:'DEVELOPMENT_EVALUATED'}]}});
const c={URLSearchParams,app:{page:'alpha'},closeDialog(){},render(){},
  hashParams:()=>new URLSearchParams(route),
  replaceHash:(update)=>{for(const [k,v] of Object.entries(update)){
    if(v==='')route.delete(k);else route.set(k,v);}},
  Data:{read:async(url)=>{const task=new URL('http://local'+url).searchParams.get('task_id');
    if(task==='left')return published('left-candidate');
    if(task==='old-right')return new Promise(resolve=>{finishOld=resolve;});
    if(task==='new-right')return published('new-candidate');
    throw Error('unexpected task '+task);}}};
library.context(c);
vm.runInContext(fs.readFileSync(require('node:path').join(_appDir,'live-study.js'),'utf8')+';globalThis.live=LiveStudy;',c);
(async()=>{
  await c.live.open('left','alpha');
  await c.live.alphaComparisonTask('new-right','new-candidate');
  finishOld(published('old-candidate'));
  await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(JSON.parse(JSON.stringify(c.live.routeContext('alpha'))),{
    study:'left',alpha_left_task:'left',alpha_left_candidate:'left-candidate',
    alpha_right_task:'new-right',alpha_right_candidate:'new-candidate'});
complete();
})().catch(e=>{console.error(e);process.exitCode=1;});
}
