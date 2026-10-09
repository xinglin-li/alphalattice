// Collection regressions over actual Data/router/Goal/Team owners. Controlled
// wire answers; ordinary refresh, Older, leave and revisit actions only.
'use strict';
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm'), assert = require('node:assert/strict');
const project = path.resolve(__dirname, '..', '..');
const appDir = process.argv[2] || path.join(project, 'src/alphalattice/interface/local_application/assets/workbench-source/js/app');
const library = require('./workbench_library.cjs');
const finish = library.guard('collection lifetime additions');
const tick = () => new Promise(resolve => setImmediate(resolve));
const task = '00000000-0000-0000-0000-000000000001', session = 'fixture-session';
const entry = id => ({entry_id: id, kind: 'INSTALLED_RESULT', task_id: task, status: 'SUCCEEDED'});
const blocked = id => ({entry_id: id, status: 'REFUSED', failure_code: 'fixture.unreadable'});
const goal = id => ({goal_id: id, goal_hash: id.padEnd(64, '0'), title: id, objective: '', kind: 'RESEARCH', state: 'OPEN', revision: 1, reference_count: 0, recorded_at: '2026-10-04T12:00:00Z', sessions: []});
const goalRefusal = id => ({goal_id: id, status: 'REFUSED', failure_code: 'fixture.goal_unreadable'});
const teamTask = id => ({task_id: id, goal_summary: id, task_kind: 'research_experiment', lifecycle: 'SUCCEEDED', final_state: 'SUCCEEDED', submitted_by: {session, goal_id: null}, artifact: null, last_activity_at: '2026-10-04T12:00:00Z'});
const results = [];

function fixture(owners = {}, initial = '#page=history') {
  let hash = initial, state = {alpha: 0};
  const requests = [], holds = [], view = {lobby: null, tables: {}, refusals: []};
  const defaults = {
    '/api/session': () => ({session_token: 'fixture-token', workspace_id: 'fixture', research_context: {tasks: {tasks: []}}}),
    '/api/research-history': () => ({entries: [], blocked_entries: [], next_cursor: null}),
    '/api/experiments': () => ({experiments: []}), '/api/decisions': () => ({decisions: []}),
  };
  const c = {
    console, URLSearchParams, Date, Number, Math, Set, Map, Object, Array, String, Promise, JSON, Boolean, Error,
    setTimeout: () => 0, clearTimeout() {}, queueMicrotask: () => {},
    app: {page: new URLSearchParams(initial.slice(1)).get('page') || 'history', book: '', session: '', input: '', inputs: [], tasks: [], compareOther: ''},
    window: {addEventListener() {}}, document: {hidden: false, body: {dataset: {}}, addEventListener() {}, querySelector: () => null},
    location: {get hash() {return hash;}, search: '', get href() {return 'http://127.0.0.1:1/workbench.html' + hash;}},
    history: {get state() {return state;}, replaceState(next, title, url) {state = next; hash = url.startsWith('#') ? url : url.slice(url.indexOf('#'));}, pushState(next, title, url) {state = next; hash = url.startsWith('#') ? url : url.slice(url.indexOf('#'));}},
    ROUTES: Object.fromEntries(['overview', 'history', 'goals', 'goal', 'team', 'team-outputs', 'team-sessions', 'team-evidence'].map(page => [page, ['Workspace', page]])),
    Window: {render() {}, inspectorMode: () => '', closeInspector() {}}, Inspect: {resetWindow() {}, selectObservation() {}, reopenFromAddress() {}},
    LiveReview: {pages: new Set(), routeContext: () => ({})}, LiveStudy: {pages: new Set(), routeContext: () => ({})},
    LiveResearch: {ready() {}, routeContext: () => ({})}, LiveWorkspace: {dismissConfirmation() {}}, LiveFeatures: {routeContext: () => ({})},
    LiveViews: {studyFacts: () => ({}), inputState: () => ({})}, LiveTasks: {paintActivity() {}},
    readPreference: () => null, savePreference() {}, closeDialog() {}, hideToast() {}, scrollTo() {}, notify() {},
    html: (parts, ...values) => parts.reduce((out, part, i) => out + part + (Array.isArray(values[i]) ? values[i].join('') : values[i] ?? ''), ''),
    t: (value, vars = {}) => String(value).replace(/\{(\w+)\}/g, (_, key) => String(vars[key] ?? '')),
    alertDialog() {}, openDialog() {}, btn: (label, action, value) => `<${action}:${value}>${label}`,
    btnAttrs: (label, action, value) => `<${action}:${value}>${label}`, icon: () => '',
    dayOf: value => String(value).slice(0, 10), when: value => String(value ?? ''), count: value => String(value),
    objectHead: library.stubs.empty, panel: (title, sub, body) => '<panel>' + title + body + '</panel>',
    objectRow: (row, options) => String(options.key), link: label => String(label), hashCell: id => String(id),
    table: (headers, rows) => {view.tables[headers[0].label] = rows; return '<table>' + rows.join('') + '</table>';},
    tr: cells => '<tr>' + cells.join('|') + '</tr>', pageOf: (rows, page) => ({shown: rows, page: page || 0, pages: 1}), pager: () => '',
    stateLine: value => '[' + (value?.state || value?.lifecycle || value) + ']', skeleton: () => 'SKELETON', emptyState: label => 'EMPTY:' + label,
    banner: (title, body) => title + ':' + body, noteLine: (title, body) => title + ':' + body, notRead: (title, body) => title + ':' + body,
    prerequisiteWays: () => '', infoMark: () => '', hint: value => value,
    refusal: (body, tone, options = {}) => {view.refusals.push(body); return '<refusal>' + body.failure_code + (options.more || '') + '</refusal>';},
    Lobby: {older: (label, action, cursor, busy) => '<OLDER:' + action + ':' + cursor + ':' + busy + '>', render: (name, spec) => {view.lobby = spec; return spec.items.map(row => row.goal_id).join('|') + spec.foot;}},
    fetch: async (url, init = {}) => {
      const q = new URL(url, 'http://127.0.0.1:1'); requests.push(url);
      const held = holds.find(h => !h.entered && h.matches(url));
      const reply = held ? (held.entered = true, await held.promise) : {value: (owners[q.pathname] || defaults[q.pathname])?.(q.searchParams, init)};
      if (reply.error) throw reply.error;
      assert.notEqual(reply.value, undefined, 'unexpected controlled read: ' + url);
      return {ok: reply.ok ?? true, status: reply.status ?? 200, headers: {get: () => 'application/json'}, text: async () => JSON.stringify(reply.value)};
    },
  };
  library.context(c, appDir);
  vm.runInContext(library.readingSource(appDir), c);
  for (const name of ['status', 'data', 'router', 'live-goals', 'live-activity', 'live-team']) vm.runInContext(fs.readFileSync(path.join(appDir, name + '.js'), 'utf8'), c, {filename: name + '.js'});
  vm.runInContext('render=()=>{};patchMain=render;globalThis.D=Data;globalThis.G=LiveGoals;globalThis.A=LiveActivity;globalThis.T=LiveTeam;globalThis.go=navigate;', c);
  function hold(matches) {let release; const h = {matches: typeof matches === 'function' ? matches : url => url === matches, entered: false, promise: new Promise(resolve => {release = resolve;}), release: value => release(value)}; holds.push(h); return h;}
  function goalView() {view.refusals = []; const markup = c.G.page(); return {rows: view.lobby?.items || [], foot: view.lobby?.foot || '', refusals: view.refusals, markup};}
  function teamView() {view.refusals = []; const markup = c.T.section(); return {tables: view.tables, refusals: view.refusals, markup};}
  return {c, D: c.D, requests, hold, goalView, teamView};
}
async function entered(held) {for (let n = 0; n < 30 && !held.entered; n++) await tick(); assert.ok(held.entered, 'controlled owner request starts');}
const count = (f, match) => f.requests.filter(match).length;

async function collections() {
  for (const which of ['experiments', 'decisions']) {
    const endpoint = '/api/' + which;
    const f = fixture({[endpoint]: () => ({[which]: [{id: 'seed'}]})});
    const refresh = keep => which === 'experiments' ? f.D.refreshExperiments(keep) : f.D.refreshDecisions();
    const rows = () => which === 'experiments' ? f.D.experiments() : f.D.decisions();
    const error = () => which === 'experiments' ? f.D.experimentsError : f.D.decisionsError;
    await refresh(false);
    const success = f.hold(endpoint), older = refresh(false), newest = refresh(false);
    await entered(success); assert.equal(count(f, url => url === endpoint), 2, 'concurrent refreshes share one wire after the seeded read');
    success.release({value: {[which]: [{id: 'accepted'}]}}); await Promise.all([older, newest]);
    assert.equal(rows()[0].id, 'accepted'); assert.equal(error(), '');
    const failed = f.hold(endpoint), oldFailure = refresh(false), newFailure = refresh(true);
    await entered(failed); assert.equal(count(f, url => url === endpoint), 3);
    failed.release({error: Error('controlled disconnect')}); await Promise.all([oldFailure, newFailure]);
    assert.equal(rows()[0].id, 'accepted', 'an older non-keep failure cannot clear a newer keep listing');
    if (which === 'experiments') assert.equal(error(), '', 'newest keep failure retains the listing without installing the old error');
    else assert.ok(error().includes('controlled disconnect'), 'the newest decision failure is stated while its rows remain');
    const recovery = f.hold(endpoint), read = refresh(false); await entered(recovery);
    recovery.release({value: {[which]: [{id: 'recovered'}]}}); await read;
    assert.equal(rows()[0].id, 'recovered'); assert.equal(error(), '');
    results.push({case: which + ' concurrent refresh success/error/recovery', wires: count(f, url => url === endpoint)});
  }
}

async function historyResync() {
  let first = {entries: [entry('result:first')], blocked_entries: [blocked('blocked:first')], next_cursor: 'c1'};
  const f = fixture({'/api/research-history': () => first}); await f.D.connect();
  const older = f.hold(url => url.includes('history_cursor=c1')), paging = f.D.refreshHistory(true);
  await entered(older); older.release({value: {entries: [entry('result:first'), entry('result:older'), entry('result:older')], blocked_entries: [blocked('blocked:first'), blocked('blocked:older'), blocked('blocked:older')], next_cursor: null}}); await paging;
  assert.equal(f.D.hasMoreHistory(), false);
  f.D.setTasks([{task_id: task, task_kind: 'research_experiment', lifecycle: 'RUNNING'}]);
  const sync = f.hold('/api/research-history?history_limit=50');
  f.D.setTasks([{task_id: task, task_kind: 'research_experiment', lifecycle: 'SUCCEEDED'}]);
  await entered(sync);
  first = {entries: [entry('result:newest'), entry('result:first'), entry('result:newest')], blocked_entries: [blocked('blocked:older'), blocked('blocked:newest'), blocked('blocked:newest')], next_cursor: 'c1'};
  sync.release({value: first}); await tick(); await tick();
  assert.equal(f.D.hasMoreHistory(), false, 'resync cannot revive an exhausted Older cursor');
  assert.deepEqual(Array.from(f.D.history(), row => row.id), ['result:newest', 'result:first', 'result:older']);
  assert.deepEqual(Array.from(f.D.historyRefusals(), row => row.entry_id), ['blocked:older', 'blocked:newest', 'blocked:first']);
  const boundary = f.requests.length; await f.D.refreshHistory(true); assert.equal(f.requests.length, boundary, 'Older stays exhausted after resync');
  results.push({case: 'History Task resync retains unique rows/refusals and exhausted cursor', rows: f.D.history().length, refusals: f.D.historyRefusals().length});
}

async function goalPages() {
  const f = fixture({'/api/goals': () => ({goals: [goal('g-first')], refused: [goalRefusal('g-bad')], next_cursor: 'c1'})}, '#page=goals');
  await f.c.G.ensure();
  const held = f.hold(url => url.includes('history_cursor=c1')), first = f.c.G.more('c1'), peer = f.c.G.more('c1');
  await entered(held); assert.equal(count(f, url => url.includes('history_cursor=c1')), 1, 'overlapping Older presses have one accepted page');
  held.release({value: {goals: [goal('g-first'), goal('g-older'), goal('g-older')], refused: [goalRefusal('g-bad'), goalRefusal('g-bad2'), goalRefusal('g-bad2')], next_cursor: 'c2'}});
  await Promise.all([first, peer]); let shown = f.goalView();
  assert.deepEqual(Array.from(shown.rows, row => row.goal_id), ['g-first', 'g-older']); assert.equal(shown.refusals.length, 2); assert.ok(shown.foot.includes('c2'));
  const loop = f.hold(url => url.includes('history_cursor=c2')), reading = f.c.G.more('c2'); await entered(loop);
  loop.release({value: {goals: [goal('g-third')], refused: [], next_cursor: 'c1'}}); await reading;
  shown = f.goalView(); assert.equal(shown.rows.length, 3); assert.equal(shown.foot, '', 'a consumed cursor cannot loop the Goal list');
  results.push({case: 'Goal Older overlap, unique rows/refusals and cursor loop', rows: shown.rows.length});

  for (const fail of [false, true]) {
    let body = {goals: [goal('g-before')], refused: [], next_cursor: 'old'};
    const stale = fixture({'/api/goals': () => body}, '#page=goals'); await stale.c.G.ensure();
    const hold = stale.hold(url => url.includes('history_cursor=old')), late = stale.c.G.more('old'); await entered(hold);
    body = {goals: [goal('g-current')], refused: [], next_cursor: 'current'};
    stale.c.G.list(); await stale.c.G.ensure(); const address = stale.c.location.hash;
    hold.release(fail ? {error: Error('older page failed')} : {value: {goals: [goal('g-stale')], refused: [goalRefusal('g-stale-bad')], next_cursor: 'rollback'}}); await late;
    const view = stale.goalView(); assert.deepEqual(Array.from(view.rows, row => row.goal_id), ['g-current']);
    assert.ok(view.foot.includes('current')); assert.ok(!view.markup.includes('older page failed')); assert.equal(stale.c.location.hash, address);
    results.push({case: 'Goal reread supersedes older ' + (fail ? 'error' : 'answer')});
  }
}

function seedTeam(f) {
  f.c.A.absorbPage({disposition: 'TAIL', epoch: 'fixture-epoch', cursor: 'fixture-epoch:1', head: 1, more: false, tasks: {}, items: [{ordinal: 1, observation_id: 'message-1', schema_kind: 'ExternalActivityObserved', occurred_at: '2026-10-04T12:00:00Z', source_kind: 'EXTERNAL_CLIENT', source_id: 'fixture', source_sequence: 1, availability: 'AVAILABLE', authority: 'AGENT_PROPOSAL', payload: {event_kind: 'NATIVE_COORDINATION_MESSAGE', producer_id: 'fixture', producer_session: session, summary: 'Collection walk', subject: {native_session_id: session, native_agent_id: session, role: 'research_lead', message_kind: 'assignment', message_id: 'message-1', input_channel: 'ACTOR_DECLARED'}}}]});
}
async function teamPages() {
  let visit = 1;
  const f = fixture({
    '/api/tasks': () => ({tasks: [teamTask(visit === 1 ? 't-first' : 't-current')], refusals: [{task_id: 't-bad', status: 'REFUSED', failure_code: 'fixture.task_unreadable'}], next_cursor: 'c1'}),
    '/api/goals': () => ({goals: [goal('team-goal')], refused: [goalRefusal('team-bad')], next_cursor: 'g1'}),
  }, '#page=team-outputs&team=' + session); seedTeam(f);
  f.teamView(); await tick(); f.teamView();
  const tasks = f.hold(url => url.startsWith('/api/tasks?') && url.includes('history_cursor=c1'));
  const goals = f.hold(url => url.startsWith('/api/goals?') && url.includes('history_cursor=g1'));
  f.c.T.outputsOlder('tasks'); f.c.T.outputsOlder('tasks'); f.c.T.outputsOlder('goals'); f.c.T.outputsOlder('goals');
  await entered(tasks); await entered(goals);
  assert.equal(count(f, url => url.startsWith('/api/tasks?') && url.includes('history_cursor=c1')), 1);
  assert.equal(count(f, url => url.startsWith('/api/goals?') && url.includes('history_cursor=g1')), 1);
  tasks.release({value: {tasks: [teamTask('t-first'), teamTask('t-older'), teamTask('t-older')], refusals: [{task_id: 't-bad', status: 'REFUSED', failure_code: 'fixture.task_unreadable'}], next_cursor: 'c2'}});
  goals.release({value: {goals: [goal('team-goal'), goal('team-older'), goal('team-older')], refused: [goalRefusal('team-bad'), goalRefusal('team-bad')], next_cursor: 'g1'}});
  await tick(); let shown = f.teamView(); assert.equal(shown.tables.Task.length, 2); assert.equal(shown.tables.Goal.length, 2); assert.equal(shown.refusals.length, 2);
  assert.ok(!shown.markup.includes('<OLDER:team-outputs-older:goals'), 'Team Goal cursor self-loop stops');
  const late = f.hold(url => url.startsWith('/api/tasks?') && url.includes('history_cursor=c2')); f.c.T.outputsOlder('tasks'); await entered(late);
  f.c.go('overview'); f.c.T.leaveOutputs(); // the actual render boundary calls leaveOutputs when this page is left
  visit = 2; f.c.go('team-outputs', {team: session}); f.teamView(); await tick();
  late.release({value: {tasks: [teamTask('t-stale')], refusals: [], next_cursor: 'rollback'}}); await tick(); shown = f.teamView();
  assert.equal(shown.tables.Task.length, 1); assert.ok(shown.tables.Task[0].includes('t-current')); assert.ok(!shown.markup.includes('t-stale')); assert.ok(!shown.markup.includes('rollback'));
  const loop = f.hold(url => url.startsWith('/api/tasks?') && url.includes('history_cursor=c1')); f.c.T.outputsOlder('tasks'); await entered(loop);
  loop.release({value: {tasks: [teamTask('t-current'), teamTask('t-next'), teamTask('t-next')], refusals: [], next_cursor: 'c1'}}); await tick(); shown = f.teamView();
  assert.equal(shown.tables.Task.length, 2); assert.ok(!shown.markup.includes('<OLDER:team-outputs-older:tasks'), 'Team Task cursor self-loop stops');
  const boundary = f.requests.length; f.c.T.outputsOlder('tasks'); await tick(); assert.equal(f.requests.length, boundary);
  results.push({case: 'Team Outputs unique pages, overlap, loop and stale answer after leave/revisit', taskRows: shown.tables.Task.length});
}

(async () => {
  await collections(); await historyResync(); await goalPages(); await teamPages();
  console.log(JSON.stringify({cases: results.length, failures: 0, results,
    limit: 'Concurrent refreshes of the same collection URL share one raw answer. Independent old/new collection wire replies are therefore not a reachable staged race; no decoder or internal microtask hook is used.'}, null, 2));
  finish();
})().catch(error => {console.error(error); process.exitCode = 1;});

// workbench async selection never accepts an obsolete response.
{
const library=require('./workbench_library.cjs'),complete=library.guard("workbench_async_selection_never_accepts_an_obsolete_response");
const _appDir=require('node:path').resolve(process.argv[2]),_project=require('node:path').resolve(__dirname,'../..');
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const c = {window:{ALPHA_PRODUCT:true}, document:{body:{dataset:{}}}, app:{},
  Inspect:{selectObservation(){}}, URLSearchParams};
library.context(c);
vm.runInContext(fs.readFileSync(require('node:path').join(_appDir,'data.js'),'utf8')+';globalThis.data=Data;',c);
(async () => {
  let finishOld;
  const old = c.data.load(() => new Promise(r => { finishOld=r; }));
  await c.data.load({subject:{task_id:'new',session:'2024-01-03'},series:[]});
  finishOld({subject:{task_id:'old',session:'2024-01-02'},series:[]});
  await old;
  assert.equal(c.data.subject().task_id,'new');
  assert.equal(c.app.session,'2024-01-03');
  await c.data.load(() => Promise.reject(Error('artifact_tampered')));
  assert.equal(c.data.ready,false);
  assert.equal(c.data.error,'artifact_tampered');
complete();
})().catch(e=>{console.error(e);process.exitCode=1;});
}

// listed task state follows task control without a reload.
{
const library=require('./workbench_library.cjs'),complete=library.guard("listed_task_state_follows_task_control_without_a_reload");
const _appDir=require('node:path').resolve(process.argv[2]),_project=require('node:path').resolve(__dirname,'../..');
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
let state='RUNNING', reads={history:0,experiments:0}, repaints=0, hold=null;
const kind='alpha.model-development', one=(lifecycle)=>({task_id:'t1',lifecycle,task_kind:kind});
const answer=async(path)=>{
  if(path.startsWith('/api/session'))
    return {session_token:'s',workspace_id:'w',research_context:{tasks:{tasks:[one('RUNNING')]}}};
  if(path.startsWith('/api/research-history')){
    reads.history++;if(hold)await hold;
    return {entries:[{entry_id:'e1',task_id:'t1',kind,status:state}],next_cursor:null};}
  if(path.startsWith('/api/experiments')){
    reads.experiments++;return {experiments:[{task_id:'t1',kind,lifecycle:state}]};}
  throw Error('unexpected read '+path);};
const c={window:{ALPHA_PRODUCT:true},document:{body:{dataset:{page:'alpha'}}},app:{},
  URLSearchParams,location:{search:'',hash:'#page=alpha'},
  hashParams:()=>new URLSearchParams(c.location.hash.slice(1)),
  Inspect:{selectObservation(){},reopenFromAddress(){}},
  Window:{render(){}},LiveResearch:{ready(){}},replaceHash(){},render(){},
  LiveViews:{studyFacts:()=>({}),inputState:()=>({})},patchMain(){repaints++;},
  fetch:async(path)=>{const body=await answer(path);
    return {ok:true,status:200,text:async()=>JSON.stringify(body)};}};
library.context(c);
vm.runInContext(fs.readFileSync(require('node:path').join(_appDir,'data.js'),'utf8')+';globalThis.data=Data;',c);
const settle=()=>new Promise(r=>setImmediate(r));
(async()=>{
  await c.data.connect();
  assert.equal(c.data.history()[0].status,'running','the history lists the run as running');
  const before={...reads}, painted=repaints, once={history:before.history+1,
    experiments:before.experiments+1};
  // Task Control reports the run ended (the activity feed's projection): the lists say so at once
  state='SUCCEEDED';
  c.data.mergeTasks([one('SUCCEEDED')]);
  assert.equal(c.data.lifecycleOf('t1','RUNNING'),'SUCCEEDED','a snapshot gives way to the report');
  assert.equal(c.data.history()[0].status,'historical','the row reads the report before a re-read');
  await settle();
  assert.deepEqual(reads,once,'one re-read of both listings');
  assert.equal(c.data.experiments()[0].lifecycle,'SUCCEEDED','the experiments are read again');
  assert.ok(repaints>painted,'the page repaints');
  // the same report again moves nothing and reads nothing
  c.data.mergeTasks([one('SUCCEEDED')]);
  await settle();
  assert.deepEqual(reads,once,'an unchanged report reads nothing');
  // an ended word is final: a stale projection never turns it back
  assert.equal(c.data.lifecycleOf('t1','CANCELLED'),'CANCELLED');
  // moves reported during a re-read cost one more read, not one each
  let release;hold=new Promise(r=>{release=r;});
  c.data.mergeTasks([{task_id:'t2',lifecycle:'RUNNING',task_kind:kind}]);
  c.data.mergeTasks([{task_id:'t2',lifecycle:'SUCCEEDED',task_kind:kind}]);
  c.data.mergeTasks([{task_id:'t3',lifecycle:'RUNNING',task_kind:kind}]);
  hold=null;release();await settle();await settle();
  assert.equal(reads.history,before.history+3,'three moves during one read: it and one more');
complete();
})().catch(e=>{console.error(e);process.exitCode=1;});
}
