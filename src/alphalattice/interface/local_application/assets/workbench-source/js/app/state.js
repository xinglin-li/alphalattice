/* Product state and shared constants. One state object owns the interaction prototype; the task,
 * evidence and review engines keep their own stores (see engines/) and are read through views. */
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const clone = (value) => JSON.parse(JSON.stringify(value));

/* Interaction registries: data-action → handler, and element id → input/change handler. */
const ACTIONS = {};
const ON_INPUT = {};
const ON_CHANGE = {};

/* Round 68: two appearances and the host's choice. Old stored names map (silver → light, the rest → dark). */
const THEMES = ['follow', 'light', 'dark'];
const THEME_ALIASES = {silver: 'light', titanium: 'light', emerald: 'light', ruby: 'light', obsidian: 'dark'};

/* page id → [primary section label, page title, primary navigation group] */
const ROUTES = {
  overview: ['Home', 'Home', 'overview'], // law 134: the dock's word is the page's word everywhere (the title, the inspector)
  data: ['Data & Workspace', 'Data maintenance', 'data'],
  issues: ['Data & Workspace', 'Data issues', 'data'],
  inputs: ['Data & Workspace', 'Research inputs', 'data'],
  storage: ['Data & Workspace', 'Storage & retention', 'data'],
  lab: ['Research Lab', 'New experiment', 'lab'],
  factor: ['Research Lab', 'Factor screening', 'lab'],
  foundation: ['Research Lab', 'Research Foundation', 'lab'],
  alpha: ['Research Lab', 'Alpha modeling', 'lab'],
  risk: ['Research Lab', 'Risk modeling', 'lab'], // the user, 2026-09-24: 应该叫risk modeling (as Alpha modeling)
  portfolio: ['Portfolio', 'Portfolio study', 'portfolio'],
  compare: ['Portfolio', 'Compare', 'portfolio'], // N2 (law 132): Portfolio study's second tab; the path is `Portfolio study / Compare`
  'alpha-compare': ['Research Lab', 'Compare', 'lab'], // N2 (law 132): Alpha modeling's second tab
  models: ['Research Lab', 'Models', 'lab'], // U50: every Alpha model the workspace may fit; Alpha modeling's third tab
  'feature-research': ['Research Lab', 'Features', 'lab'], // U56: the formula factors agents declared, a person activates; Factor screening's second tab
  features: ['Research Lab', 'Research-local formulas', 'lab'], // the product's Local Feature (2026-09-25): reached from the Lab, a Task or its address
  goals: ['Goals', 'Goals', 'goals'], // U23: a section of its own (the user, 2026-09-28: 这个东西可以在UI里单独立一个section)
  goal: ['Goals', 'Timeline', 'goals'], // a goal's three folders, its tabs while it is chosen
  'goal-conversation': ['Goals', 'Conversation', 'goals'],
  'goal-results': ['Goals', 'Results', 'goals'],
  cases: ['Goals', 'Goals', 'goals'], // the research case page's old addresses open their goal
  books: ['Evidence & CRO', 'Books', 'evidence'], // N2 (law 123): the Evidence Home, where a book is chosen
  evidence: ['Evidence & CRO', 'Overview', 'evidence'], // round G2: the object's overview (law 87: the page's one word, Overview; the head names the book)
  'evidence-stream': ['Evidence & CRO', 'Sources', 'evidence'],
  'evidence-reading': ['Evidence & CRO', 'Reading', 'evidence'], // round E3: the reading workbench
  report: ['Evidence & CRO', 'Report & delivery', 'evidence'],
  handoff: ['Evidence & CRO', 'Review', 'evidence'], // round E4: the page's one word
  history: ['History', 'History', 'history'],
  team: ['Team', 'Conversation', 'team'], // N2: the session's object page (law 125)
  'team-participants': ['Team', 'Participants', 'team'], // U51: who took part, their models and tokens (the user, 2026-09-30)
  'team-outputs': ['Team', 'Outputs', 'team'], // U54: what the session produced (the user, 2026-09-30: 产出)
  'team-evidence': ['Team', 'Product record', 'team'], // U53: named for what it holds (the user, 2026-09-30) // N4 (the third review; law 134): the dock's Evidence names the books -- this is what the product observed
  'team-sessions': ['Team', 'Sessions', 'team'],
  tasks: ['Tasks', 'Tasks', 'history'],
  advanced: ['Advanced', 'Governance & validation', 'advanced'], // an alias of settings (round 62): old links open the Settings page
  settings: ['Settings', 'Settings', 'settings'],
  upgrade: ['Settings', 'Upgrade', 'settings'], // R1: what the installed code changed; reached from Home's decision and the workspace popover
  kit: ['Settings', 'Component workshop', 'kit'], // the workshop (law 106): opened by its address, never listed
};
/* Routes a reader never meets in the navigation or Quick Open. */
const HIDDEN_ROUTES = new Set(['kit', 'features', 'upgrade', 'cases']); // features: a composer needs its input, so it opens from the Lab, a Task or an address

/* Reading profile per page: interior typography and rhythm only; the outer frame is shared. */
const READING_ROLES = {
  overview: 'operations', data: 'operations', issues: 'operations', inputs: 'operations', storage: 'operations',
  lab: 'analysis', factor: 'analysis', foundation: 'analysis', alpha: 'analysis', risk: 'analysis', portfolio: 'analysis', compare: 'analysis',
  'alpha-compare': 'analysis', models: 'analysis', 'feature-research': 'analysis', goals: 'analysis', goal: 'analysis', 'goal-conversation': 'analysis', 'goal-results': 'analysis', cases: 'analysis', books: 'analysis', features: 'analysis',
  evidence: 'analysis', 'evidence-stream': 'analysis', 'evidence-reading': 'analysis', report: 'analysis', history: 'analysis', handoff: 'analysis',
  tasks: 'operations', advanced: 'operations', settings: 'operations', upgrade: 'operations',
};
const readingRole = () => READING_ROLES[app.page] || 'operations';

/* Status vocabulary: key → [label, icon, tone, meaning]. Labels are translated when rendered. */
/* The state table (STATES) and its readers live in status.js. */


/* Pages that read back a saved object; they keep their own input and book. */
const SAVED_VIEWS = new Set(['factor', 'foundation', 'alpha', 'risk', 'portfolio', 'compare', 'evidence', 'report']);

const app = {
  page: 'overview',
  theme: 'follow',
  workspace: Data.workspace(),
  empty: false,
  input: 'demo-input-0803',
  data: Data.clocks().data,
  feature: Data.clocks().feature,
  session: '2024-08-12',
  book: 'demo-book-01',
  analysis: 'demo-analysis-03',
  tab: 'holdings',
  mode: 'yaml',
  chart: 'indexed',
  plan: null,
  declaration: null,
  yaml: '',
  decisionSaved: false,
  foundationSaved: true,
  historyQuery: '',
  historyKind: 'all',
  historySort: 'source',
  holdingsQuery: '',
  holdingsPage: 0,
  compareOther: 'demo-book-02',
  issuesDecision: null,
  cleared: false,
  firstRun: {name: 'New Research', path: 'D:/AlphaLattice/new-research'},
  tasks: [],
  inputs: [
    {id: 'demo-input-0803', name: 'Factor development', date: '2026-08-03', verified: true, pinned: true},
    {id: 'demo-input-0727', name: 'Factor development', date: '2026-07-27', verified: true, pinned: false},
    {id: 'demo-input-0720', name: 'Factor development', date: '2026-07-20', verified: true, pinned: true},
  ],
  freeze: false,
  labTask: null,
  labTaskYaml: null,
  holdingFocus: null,
  listContext: null, // round 53: the list a record was opened from ({list, keys, index}) for stepping
  holdingsSort: 'listing-asc',
  reviewTab: 'scope',
  reviewItem: 'mapping',
};

app.tasks = [];
Object.assign(app, {input: '', session: '', book: '', analysis: '', compareOther: '', inputs: [], foundationSaved: false});
const factors = [
  ['MOM_21D', 'Momentum', 0.031, 0.043, 97.8, 'keep'],
  ['REV_5D', 'Reversal', 0.019, 0.078, 98.1, 'constrain'],
  ['VOL_20D', 'Volatility', -0.026, 0.061, 99.2, 'keep'],
  ['RANGE_10D', 'Range', 0.011, 0.196, 96.4, 'reject'],
  ['VOLUME_Z20', 'Liquidity', 0.017, 0.108, 97.7, 'constrain'],
  ['TREND_63D', 'Trend', 0.022, 0.059, 98.4, 'keep'],
].map((r, i) => ({id: i, name: r[0], family: r[1], ic: r[2], q: r[3], coverage: r[4], choice: r[5], rationale: i === 3 ? 'Weak support in this illustrative screen.' : 'Illustrative curation rationale; evaluate within the declared sample.'}));

/* ---- research declaration ---- */
function defaults() {
  return {kind: 'portfolio.policy-development', input: app.input, names: 40, sleeves: 3, exit: 70, cost: 5, seed: 20260816};
}
function dumpYaml(d) {
  const head = `experiment:\n  kind: ${d.kind}\n  research_input: ${d.input}\n  determinism:\n    seed: ${d.seed}\n    thread_limit: 1\n`;
  if (!d.kind.startsWith('portfolio')) return head + `  parameters:\n    sample_method: fixed_design_example\n# Method-specific product schemas were not supplied.\n`;
  return head + `  portfolio:\n    names_per_sleeve: ${d.names}\n    sleeve_count: ${d.sleeves}\n    exit_rank: ${d.exit}\n    cost_bps_per_side: ${d.cost}\n    unavailable_return_handling: quarantine_listings\n`;
}
app.declaration = defaults();
app.yaml = dumpYaml(app.declaration);

/* ---- context accessors ---- */
function input() {
  {
    if(LiveStudy.pages.has(app.page)){
      const source=LiveStudy.context();
      return Data.inputs().find(v=>v.id===source?.input_id&&v.binding_hash===source?.input_binding_hash)
        || {id:source?.input_id || '',name:source?.input_id || '',date: ''};
    }
    if(app.page==='lab') {
      const draft=LiveResearch.context();
      return Data.inputs().find(v=>v.id===draft.input_id && v.binding_hash===draft.input_binding_hash)
        || {id:draft.input_id || '',name:draft.input_id || '',date: ''};
    }
    const subject = Data.subject();
    return app.inputs.find((x) => x.id === (subject?.input_id || app.input) && (!subject || x.binding_hash === subject.input_hash))
      || {id: subject?.input_id || app.input || '', name: subject?.input_id || app.input || '', date: ''};
  }
}
const viewBook = () => Data.subject()?.task_id || '';
const viewSession = () => Data.subject()?.session || '';
const openTaskCount = () => Data.actionableTasks().length;

function currentContext() {
  if (LiveStudy.pages.has(app.page))return {page:app.page,workspace:Data.workspace(),study:LiveStudy.context(),claim:'SAVED_RESEARCH_NOT_CURRENT_AUTHORITY'};
  if (LiveReview.pages.has(app.page)) return {page:app.page,workspace:Data.workspace(),review:LiveReview.context(),claim:'EVIDENCE_AND_REVIEW_NOT_TRADE_AUTHORITY'};
  if (app.page==='lab') return {page:app.page,workspace:Data.workspace(),declaration:LiveResearch.context(),claim:'DECLARATION_AND_PREVIEW_NOT_EXECUTION'};
  return {page: app.page, workspace: Data.workspace(), subject: Data.subject(), claim: Data.notice()};
  return {prototype: true, page: app.page, workspace: app.workspace, inputId: input().id, inputCutoff: input().date, selectedInputForFutureResearch: app.input, workDataThrough: app.data, bookId: viewBook(), holdingsSession: viewSession(), reviewState: 'PARTIAL', route: 'REQUEST_EVIDENCE_REFRESH', blockingAction: 'RESOLVE_ISSUER_MAPPING'};
}

/* ---- viewer preferences (theme, language, comfort) ---- */
const PREF_KEY = 'alphalattice.workstation.viewer.v4';
function readPreference(key) {
  try {
    return JSON.parse(localStorage.getItem(PREF_KEY) || '{}')[key];
  } catch {
    return null;
  }
}
function savePreference(key, value) {
  try {
    const prefs = JSON.parse(localStorage.getItem(PREF_KEY) || '{}');
    prefs[key] = value;
    localStorage.setItem(PREF_KEY, JSON.stringify(prefs));
  } catch {
    /* Restricted file:// storage: keep this session's preference in memory only. */
  }
}
