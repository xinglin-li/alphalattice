// ST3 F31/U174: public task_timing output over actual Data/Task consumers; no browser clock fact.
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const library = require('./workbench_library.cjs');
const [appDir, which, payload] = process.argv.slice(2), timing = JSON.parse(payload);
const read = name => fs.readFileSync(path.join(appDir, name + '.js'), 'utf8');
const finish = library.guard('Task owner duration ' + which);
const ID = '00000000-0000-0000-0000-000000000001', OTHER = '00000000-0000-0000-0000-000000000002';
const START = timing.RUNNING.started_at, END = timing.last_activity_at;
class BrowserDate extends Date {static now() {return Date.parse('2027-02-01T12:00:00Z');}}
const words = markup => String(markup).replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
const task = (extra = {}) => ({task_id: ID, task_kind: 'research_experiment', goal_summary: 'Labelled Task duration fixture', lifecycle: 'RUNNING', running_since: START, last_activity_at: END, current_stage: 'prepare_data', verified_stage_count: 0, total_stage_count: 0, ...extra});

function fixture({statusSource = read('status'), dataSource = read('data'), tasksSource = read('live-tasks'), statusId = ID, statusStage = 'prepare_data', span = timing.RUNNING.running_seconds, recordSpan} = {}) {
  const requests = [], rows = [], route = new URLSearchParams({page: 'history'});
  let inspector = null;
  const record = task(recordSpan === undefined ? {} : {running_seconds: recordSpan});
  const view = {task_id: ID, lifecycle: 'RUNNING', status: record, operation_running: false,
    liveness: {status: 'NOT_OBSERVED'}, stages: [], artifact_refs: [], actions: ['CANCEL', 'RECOVER', 'REPLAN'].map(action => ({action, available: false, scope: 'Fixture action unavailable', expected_effect: '', reason: 'Fixture reads only'})),
    next_requests: {}, health: {status: 'HEALTHY'}, verified_stage_count: 0, total_stage_count: 0};
  const c = {console, Date: BrowserDate, URL, URLSearchParams, TextEncoder, setTimeout: () => 0, clearTimeout() {}, queueMicrotask() {},
    app: {page: 'history', book: ''}, window: {}, document: {documentElement: {}, body: {dataset: {}}, querySelector: () => null},
    location: {search: '', get hash() {return '#' + route;}, href: 'http://127.0.0.1:1/workbench.html'},
    render() {}, patchMain() {}, closeDialog() {}, clone: structuredClone, $: () => null, $$: () => [],
    readPreference: () => null, savePreference() {}, routeUrl: (page, extra = {}) => '#' + new URLSearchParams({page, ...extra}), listUrl: page => '#page=' + page,
    replaceHash: patch => {for (const [key, value] of Object.entries(patch)) {if (value === '' || value == null) route.delete(key); else route.set(key, String(value));}}, hashParams: () => new URLSearchParams(route),
    Window: {render() {}, renderSide() {}, inspectorMode: () => inspector ? 'task' : '', openedBy: (kind, id) => kind === 'task' && inspector?.by[1] === id,
      openInspector: value => {inspector = value;}, setInspectorBody: body => {inspector.body = body;}},
    Inspect: {selectAddressMode: (kind, extra) => {for (const [key, value] of Object.entries(extra)) route.set(key, value);}, addressedMode: () => 'task', closeLens() {}},
    LiveReview: {taskFinished() {}}, LiveViews: {nameOf: () => ({name: 'Labelled Task duration fixture'}),
      // BADGE's successor and stop facts: this Task has neither.
      taskSuccessor: () => '', taskAttentionFacts: () => ''},
    LiveActivity: {starterOf: () => '', logLines: () => [], retained: () => [], refresh: async () => {}, markSeen() {}},
    ROUTES: {tasks: ['Workspace', 'Tasks']}, html: () => '', t: key => key,
    fetch: async url => {
      requests.push(url); const q = new URL(url, 'http://fixture'); let value;
      if (q.pathname === '/api/tasks') value = {tasks: [record]};
      else if (q.pathname === '/api/tasks/recovery') value = view;
      else if (q.pathname === '/api/status') value = {task_id: statusId, lifecycle: 'RUNNING', current_stage: statusStage, verified_stage_count: 0, timing: {...timing.RUNNING, running_seconds: span}, activity_timing: {stage: 'prepare_data', last_work_at: timing.RUNNING.sampled_at}};
      else if (q.pathname === '/api/tasks/incidents') value = {incidents: []};
      else throw Error('Unexpected duration fixture read: ' + url);
      return {ok: true, headers: {get: () => 'application/json'}, text: async () => JSON.stringify(value)};
    }};
  library.context(c, appDir);
  for (const name of ['html', '../data/zh', 'i18n', 'icons']) vm.runInContext(read(name), c);
  vm.runInContext(statusSource, c); vm.runInContext(read('components'), c);
  // The public Time grouping exposes the row's real stateLine duration; grouping chrome is a port.
  vm.runInContext('Lobby', c).render = (name, spec) => {rows.length = 0; const html = vm.runInContext('html', c);
    return html`${spec.items.map(item => {const markup = spec.row(item, {props: {}, group: 'time'}); rows.push({item, markup}); return markup;})}`;
  };
  c.objectHead = library.stubs.empty;
  vm.runInContext(dataSource, c); vm.runInContext(tasksSource, c);
  vm.runInContext('globalThis.D=Data;globalThis.T=LiveTasks;globalThis.line=stateLine;', c);
  return {c, requests, rows, body: () => String(inspector.body)};
}
const summary = f => words(f.body().match(/<p class="run-summary">([\s\S]*?)<\/p>/)[1]);
const logSummary = f => words(f.body().match(/class="run-log-summary">(Worked for[\s\S]*?)<\/span>/)?.[1] || '');
async function collectionOwnerSpans() {
  const f = fixture();
  const variants = [125, 0, null, undefined, -1, '125', {seconds: 125}, NaN, Infinity];
  f.c.D.setTasks(variants.map((value, i) => task({task_id: ID + ':' + i, ...(value === undefined ? {} : {running_seconds: value})})));
  const runs = f.c.D.runsOf('task');
  for (let i = 0; i < variants.length; i++) assert.equal(runs[i].running_seconds, variants[i], 'Data carries the owner span without conversion');
  f.c.T.page();
  for (let i = 0; i < variants.length; i++) {
    const markup = String(f.rows[i].markup), expected = i === 0 ? '2 min' : i === 1 ? 'under 1 min' : '';
    if (expected) assert.ok(words(markup).includes(expected), markup);
    else assert.ok(!markup.includes('class="state-time"'), 'absent, negative or malformed owner span leaves the duration empty');
  }
  for (const lifecycle of ['RUNNING', 'CANCEL_REQUESTED', 'QUEUED']) assert.ok(words(f.c.line(task({lifecycle, running_seconds: 125}))).includes('2 min'));
  const deferred = task({lifecycle:'DEFERRED', running_seconds:timing.DEFERRED.running_seconds});
  assert.equal(timing.DEFERRED.running_seconds, 45);
  assert.ok(words(f.c.line(deferred)).includes('45 s'));
  for (const extra of [{}, {running_since: ''}, {last_activity_at: ''}, {last_activity_at: 'invalid'}, {last_activity_at: '2026-07-01T00:00:00Z'}]) {
    const markup = String(f.c.line(task({lifecycle: 'SUCCEEDED', running_seconds: 9000, ...extra})));
    if (!Object.keys(extra).length) assert.ok(words(markup).includes('45 s'), 'ended duration uses the owner start and last activity');
    else assert.ok(!markup.includes('class="state-time"'), 'ended duration needs two valid ordered owner times');
  }
  assert.deepEqual(f.requests, [], 'reading already-owned collection spans requires no GET');
  const current = read('status'), former = current.replace(/function durationOf\(x\) \{[\s\S]*?\n\}/, `function durationOf(x) {
    const start = Date.parse(x?.running_since || x?.started || '');
    if (!Number.isFinite(start)) return '';
    const moving = stateMoving(x?.lifecycle || x?.state), end = moving ? Date.now() : Date.parse(x?.last_activity_at || x?.finished || '');
    return Number.isFinite(end) && end >= start ? durationText(end - start, moving) : '';
  }`);
  assert.notEqual(former, current);
  const old = fixture({statusSource: former}); old.c.D.setTasks([task({running_seconds: 125})]); old.c.T.page();
  assert.ok(!words(old.rows[0].markup).includes('2 min'), 'actual collection row reproduces the retired browser-clock span');
  const priorData = read('data').replace('running_seconds: v.running_seconds, ', ''); assert.notEqual(priorData, read('data'));
  const missing = fixture({dataSource: priorData}); missing.c.D.setTasks([task({running_seconds: 125})]); missing.c.T.page();
  assert.ok(!String(missing.rows[0].markup).includes('class="state-time"'), 'retiring collection passthrough removes the real row duration');
}
async function selectedTaskStatus() {
  const f = fixture(); await f.c.T.open(ID);
  assert.ok(summary(f).includes('2 min'), summary(f)); assert.ok(logSummary(f).includes('Worked for 2 min'), f.body());
  for (const key of ['Stage updated', 'Stage duration', 'Latest actual work']) assert.ok(words(f.body()).includes(key), key);
  const receiptValue = key => words(f.body().match(new RegExp('<dt>'+key+'</dt><dd[^>]*>([^]*?)</dd>'))[1]);
  assert.equal(receiptValue('Stage updated'), f.c.when(timing.RUNNING.stages[0].updated_at));
  assert.equal(receiptValue('Latest actual work'), f.c.when(timing.RUNNING.sampled_at));
  assert.notEqual(receiptValue('Latest actual work'), receiptValue('Stage updated'), 'separate owner clocks');
  const paths = f.requests.map(url => new URL(url, 'http://fixture'));
  assert.deepEqual(paths.map(q => q.pathname).sort(), ['/api/status', '/api/status', '/api/tasks', '/api/tasks/incidents', '/api/tasks/recovery'].sort(), 'duration reuses the existing STATUS and follow reads');
  assert.equal(paths.filter(q => q.pathname === '/api/status' && !q.searchParams.has('wait_seconds')).length, 1);
  assert.ok(paths.filter(q => ['/api/status', '/api/tasks/recovery'].includes(q.pathname)).every(q => q.searchParams.get('task_id') === ID));
  const other = fixture({statusId: OTHER, span: 840, recordSpan: 60}); await other.c.T.open(ID);
  assert.ok(summary(other).includes('1 min') && !summary(other).includes('14 min'), 'another Task STATUS cannot supply this Task summary');
  assert.ok(logSummary(other).includes('Worked for 1 min'), 'the log follows the same exact-id guard');
  assert.ok(!words(other.body()).includes('Latest actual work'), 'task binding');
  const moved = fixture({statusStage: 'prepare_features'}); await moved.c.T.open(ID);
  assert.ok(!words(moved.body()).includes('Stage updated') && !words(moved.body()).includes('Latest actual work'), 'stage binding');
  for (const span of [null, -1, '125']) {
    const absent = fixture({span}); await absent.c.T.open(ID);
    assert.ok(!absent.body().match(/<p class="run-summary">[\s\S]*?class="state-time"[\s\S]*?<\/p>/), 'invalid STATUS elapsed span leaves the summary empty');
    assert.equal(logSummary(absent), '', 'invalid STATUS span supplies no worked duration');
  }
  const current = read('live-tasks'), former = current.replace('...timedTask(v),lifecycle:', '...v,lifecycle:').replace('durationOf(timedTask(v))', 'durationOf(v)');
  assert.notEqual(former, current); const old = fixture({tasksSource: former}); await old.c.T.open(ID);
  assert.ok(!summary(old).includes('2 min') && !logSummary(old).includes('2 min'), 'actual selected readers lose the already-read owner span under their former source');
  const unguarded = current.replace('S.status?.task_id===v.task_id ?', 'S.status ?'); assert.notEqual(unguarded, current);
  const wrong = fixture({tasksSource: unguarded, statusId: OTHER, span: 840, recordSpan: 60}); await wrong.c.T.open(ID);
  assert.ok(summary(wrong).includes('14 min') && logSummary(wrong).includes('14 min'), 'removing the identity guard admits the unrelated owner span');
}
(async () => {
  assert.ok(['collection-owner-spans', 'selected-task-status'].includes(which));
  await (which === 'collection-owner-spans' ? collectionOwnerSpans() : selectedTaskStatus()); finish();
})().catch(error => {console.error(error); process.exitCode = 1;});
