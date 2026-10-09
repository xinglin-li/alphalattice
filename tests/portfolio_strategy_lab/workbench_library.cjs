// The component library's constants and the parameters the build hands the scripts, for a
// harness context: read from
// the source itself -- the marked section of components.js and design/parameters.json -- so a
// context carries the product's values, never a copy of them.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function library(appDir) {
  const source = JSON.parse(fs.readFileSync(path.join(appDir, '..', '..', 'design', 'parameters.json'), 'utf8'));
  const BREAKPOINTS = {};
  for (const [kind, body] of Object.entries(source.breakpoints || {})) {
    if (kind === 'about') continue;
    BREAKPOINTS[kind] = {};
    for (const [name, step] of Object.entries(body)) BREAKPOINTS[kind][name] = step.width ?? step.height;
  }
  const tokens = Object.assign({}, ...Object.values(source.groups).map((group) => group.tokens));
  const PARAMETER_VALUES = Object.fromEntries([...(source.scripts || []), ...(source.counts || [])].map((name) => [name, parseFloat(tokens[name].base)]).concat((source.tones || []).map((name) => [name, tokens[name].base]))); // a meaning's tone is a name
  const text = fs.readFileSync(path.join(appDir, 'components.js'), 'utf8');
  const from = text.indexOf("/* ---- the library's constants"), to = text.indexOf("/* ---- end of the library's constants ---- */");
  if (from < 0 || to < 0) throw Error("components.js: the library's constants are not marked");
  // the label table the build embeds (U72), read from its one source
  const table = JSON.parse(fs.readFileSync(path.join(appDir, '..', '..', '..', '..', 'labels.json'), 'utf8'));
  const LABELS = Object.fromEntries(table.labels.map((l) => [l.id, {kind: l.kind, title: l.title, title_zh: l.title_zh, summary: l.summary, summary_zh: l.summary_zh}]));
  const context = {String, BREAKPOINTS, PARAMETER_VALUES};
  vm.createContext(context);
  vm.runInContext(text.slice(from, to) + ';globalThis.out={SHORT,short,LIST_PAGE,LEGEND,LOBBY,param};', context);
  return {...context.out, BREAKPOINTS, PARAMETER_VALUES, LABELS};
}
// the Evidence library's builders (C1): the marked section of components.js evaluated inside the
// harness's own context, so each builder composes that harness's stubs (html, btnAttrs, panel ...)
// and a builder moved into the library needs no copy in any harness
function builders(c, appDir) {
  const text = fs.readFileSync(path.join(appDir, 'components.js'), 'utf8');
  const from = text.indexOf('/* ---- the Evidence library'), to = text.indexOf('/* ---- end of the Evidence library ---- */');
  if (from < 0 || to < 0) throw Error('components.js: the Evidence library is not marked');
  const section = text.slice(from, to);
  const names = [...section.matchAll(/^(?:function\s+(\w+)|const\s+(\w+)\s*=)/gm)].map((m) => m[1] || m[2]);
  vm.createContext(c);
  return vm.runInContext(`(() => {${section}; return {${names.join(',')}};})()`, c);
}
// The page chrome a lobby composes (law 136) from outside the library's sections: the first axis,
// every fact shown, no filter chips, nothing remembered; a harness that reads them keeps its own.
const chrome = {
  readPreference: () => null, savePreference() {}, displayOptions: () => '', filterBar: () => '',
  displayState: (name, spec) => ({show: spec?.current?.show ?? (spec?.shows?.[0]?.[0] || ''), group: spec?.groupings?.[0]?.[0] || '', order: spec?.orderings?.[0]?.[0] || '', props: {}, density: 'comfortable'}),
  setDisplay: (name, patch) => ({...patch}),
  chipValues: (v) => (v ? String(v).split(',').filter(Boolean) : []),
};
// The Host's routes (U13), from its own two tables in local_web_session.py: the session answer names them, and a
// page's `Data.route` turns an operation into its path by them -- a harness's Data stub gains the same
function hostRoutes() {
  const src = fs.readFileSync(path.join(__dirname, '..', '..', 'src', 'alphalattice', 'control', 'product_host', 'composition', 'local_web_session.py'), 'utf8');
  const routes = {};
  for (const name of ['OPERATION_ROUTES', 'HANDLED_OPERATION_ROUTES']) {
    const at = src.indexOf('\n' + name + ':'), end = src.indexOf('\n)\n', at);
    if (at < 0 || end < 0) throw Error('local_web_session.py: no ' + name + ' table');
    for (const m of src.slice(at, end).matchAll(/\(\s*"(GET|POST)",\s*"([^"]+)",\s*"([A-Z_]+)",?\s*\)/g)) routes[m[3]] = {method: m[1], path: m[2]};
  }
  return routes;
}
library.hostRoutes = hostRoutes;
// the context gains what it does not define itself (a harness may keep its own stub)
library.into = (c, appDir) => {
  if (!c.refusalParts) {
    const source = fs.readFileSync(path.join(appDir, 'components.js'), 'utf8');
    c.refusalParts = vm.runInNewContext(source.slice(source.indexOf('function refusalParts('), source.indexOf('function notRead(')) + ';refusalParts;');
  }
  // Presentation-only harnesses supply their own Data port. These defaults keep
  // their unrelated flows usable; lifetime tests load the actual Data module.
  if (c.Data && typeof c.Data === 'object') {
    if (!c.Data.navigationIntent) c.Data.navigationIntent = () => null;
    if (!c.Data.decisions) c.Data.decisions = () => [];
    if (!c.Data.navigationCurrent) c.Data.navigationCurrent = () => true;
    if (!c.Data.beginNavigation) c.Data.beginNavigation = () => null;
    if (!c.Data.uniqueRows) c.Data.uniqueRows = (rows, key) => {
      const seen = new Set();
      return rows.filter(row => { const id = key(row); if (seen.has(id)) return false; seen.add(id); return true; });
    };
  }
  // Data-only compositions have no inspector consumer; real address/lifecycle
  // contracts use actual Inspect in routes and follow_readers, never this adapter.
  if (c.Inspect && !c.Inspect.reopenFromAddress) c.Inspect.reopenFromAddress = () => {};
  if (!c.hashParams) c.hashParams = () => new URLSearchParams((c.location?.hash || '').slice(1));
  if (c.Data && typeof c.Data === 'object' && !c.Data.route) {
    const routes = hostRoutes();
    c.Data.offers = (op) => Boolean(routes[op]);
    c.Data.route = (op) => { if (!routes[op]) throw Error('local_web.operation_route_absent: ' + op); return routes[op].path; };
  }
  // the reading helpers (`when`) read the locale: a context reads English unless it says otherwise
  if (!('I18N' in c)) c.I18N = {zh: false};
  if (!('said' in c)) c.said = (value) => c.I18N.said ? c.I18N.said(value) : c.t(String(value ?? ''));
  for (const [name, value] of Object.entries(library(appDir))) if (!(name in c)) c[name] = value;
  for (const [name, value] of Object.entries(chrome)) if (!(name in c)) c[name] = value;
  for (const [name, value] of Object.entries(builders(c, appDir))) if (!(name in c)) c[name] = value;
  return c;
};
// The shared reading rules without the tooltip's page-global event listeners. Harnesses that
// need owner words and route helpers evaluate only the marked reading regions they consume.
library.readingSource = (appDir) => {
  const text = fs.readFileSync(path.join(appDir, 'components.js'), 'utf8');
  const from = text.indexOf('/* ---- percentages for reading');
  const tooltip = text.indexOf('/* One tooltip (round 93');
  const grammar = text.indexOf('/* One number grammar.', tooltip);
  const fraction = text.indexOf('const pctFraction', grammar);
  const end = text.indexOf('\n', fraction);
  if (from < 0 || tooltip < 0 || grammar < 0 || fraction < 0 || end < 0)
    throw Error('components.js: shared reading rules are not marked');
  return text.slice(from, tooltip) + text.slice(grammar, end + 1) +
    ';globalThis.pctNumber=pctNumber;globalThis.pctText=pctText;globalThis.pctFraction=pctFraction;' +
    'globalThis.short=short;globalThis.mono=mono;globalThis.roundDecimalText=roundDecimalText;' +
    'globalThis.CODE_WORDS=CODE_WORDS;globalThis.codeWords=codeWords;' +
    'globalThis.coded=coded;globalThis.methodWords=methodWords;globalThis.countText=countText;' +
    'globalThis.pluralText=pluralText;globalThis.whenText=whenText;globalThis.prerequisiteWays=prerequisiteWays;';
};
// A small isolated prerequisite reader for flow harnesses that deliberately keep their own word
// stubs. Its only UI dependencies are supplied by that harness; tooltip and reader globals stay out.
library.prerequisiteWaysSource = (appDir) => {
  const text = fs.readFileSync(path.join(appDir, 'components.js'), 'utf8');
  const from = text.indexOf('const PREREQ_RESULTS =');
  const to = text.indexOf('\nfunction prerequisitesPanel', from);
  if (from < 0 || to < 0) throw Error('components.js: prerequisite ways are not marked');
  return 'globalThis.prerequisiteWays = (() => {' + text.slice(from, to) +
    '\nconst joinMarkup = (items, separator = \'\') => items.join(separator); return prerequisiteWays; })();';
};
// The Task projection's refusal sentence has its own table. Isolate that table and function so the
// harness can use the real sentence while keeping its own codeWords/coded display stubs.
library.explainCodeSource = (appDir) => {
  const text = fs.readFileSync(path.join(appDir, 'components.js'), 'utf8');
  const from = text.indexOf('const CODE_WORDS = {');
  const explain = text.indexOf('const explainCode =', from);
  const end = text.indexOf('\n', explain);
  if (from < 0 || explain < 0 || end < 0) throw Error('components.js: explanation words are not marked');
  return 'globalThis.explainCode = (() => {' + text.slice(from, end) +
    '\nglobalThis.CODE_WORDS=CODE_WORDS; globalThis.CODE_LINES=CODE_LINES; return explainCode; })();';
};
// The refusal renderer lives outside the Evidence builders. A reader harness that uses the product
// renderer can load this one function after status.js and the reading helpers, without evaluating
// the whole presentation file or maintaining a second copy of its wording and route behavior.
library.refusal = (c, appDir) => {
  const text = fs.readFileSync(path.join(appDir, 'components.js'), 'utf8');
  const from = text.indexOf('function refusal(body, tone =');
  const to = text.indexOf('\n/* ---- run shapes', from);
  if (from < 0 || to < 0) throw Error('components.js: the refusal renderer is not marked');
  if (!vm.isContext(c)) vm.createContext(c);
  vm.runInContext(text.slice(from, to) + '\nglobalThis.refusal=refusal;', c);
  return c.refusal;
};
// A harness proves it finished. An async harness whose await never settles exits 0 with its later
// assertions unrun (handoffs: this hid 13 flows from round 21 to round 77), so a harness takes
// `finish` from here at its start and calls it at its end; an exit without it is a failure.
library.guard = (name) => {
  let complete = false;
  process.on('exit', (code) => {
    if (code === 0 && !process.exitCode && !complete) {
      console.error(name + ' did not complete: an await never settled');
      process.exitCode = 1;
    }
  });
  return () => { complete = true; console.log(name + ' complete'); };
};
// Real shared words in an isolated context: legacy layout stubs cannot translate a state.
library.words = (appDir, routeUrl = null) => {
  const c = {console, window: {}, document: {documentElement: {}}, app: {},
    html: (parts, ...values) => parts.reduce((out, part, i) => out + part + (values[i] ?? ''), ''),
    ...(routeUrl ? {routeUrl} : {}), ...library(appDir)};
  vm.createContext(c);
  for (const name of ['html.js', '../data/zh.js', 'i18n.js', 'status.js', 'icons.js', 'components.js'])
    vm.runInContext(fs.readFileSync(path.join(appDir, name), 'utf8'), c);
  return vm.runInContext('({I18N, t, said, html, coded, badge, countText, actorWords, ACTORS, codeWords, declaredCodeWord, stageOf, stageWord, stateLine, stateOf, typedBtn, refCell, hashCell, locatorCell, codeCell, notRead, refusalParts, causeLine, evidenceRow, goalReferenceIntegrity, hint, objectRow, stat, figureTile, glyphWord, measureStrip, TONE, EVIDENCE, CODE_WORDS, STATES, STAGES})', c);
};
// The handover's one-argument Node command also enumerates the public owners. Pytest
// supplies this JSON directly so its eight harnesses need no extra interpreter process.
library.codes = () => {
  if (process.argv[3]) return JSON.parse(process.argv[3]);
  const project = path.resolve(__dirname, '..', '..');
  const local = path.join(project, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  const python = fs.existsSync(local) ? local : (process.env.PYTHON || 'python');
  const result = require('node:child_process').spawnSync(python, ['-B', '-c',
    'import ast,json; from pathlib import Path; from alphalattice.control.observation_runtime.contracts import ObservationAuthority; from alphalattice.control.task_control.contracts import TaskLifecycle; tree=ast.parse(Path("src/alphalattice/control/product_host/composition/goals.py").read_text(encoding="utf-8")); operations=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="REFERENCE_OPERATIONS" for t in n.targets)); refusals=next(sorted(ast.literal_eval(n.value.args[0])) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="COMPARISON_REFUSALS" for t in n.targets)); print(json.dumps({"authorities": list(ObservationAuthority), "lifecycles": list(TaskLifecycle), "goal_reference_operations": operations, "goal_comparison_refusals": refusals}))'],
    {cwd: project, encoding: 'utf8', timeout: 10000, env: {...process.env, ALPHALATTICE_NETWORK_DISABLED: '1'}});
  if (result.status !== 0) throw new Error('owner enums unreadable: ' + (result.stderr || result.error));
  return JSON.parse(result.stdout);
};
module.exports = library;
