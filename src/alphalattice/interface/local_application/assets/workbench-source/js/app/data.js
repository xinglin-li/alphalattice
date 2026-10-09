/* Shared transport and readback: the product's owners answer every read; nothing is fabricated here. */
const Data = (() => {
  let source = null;
  let portfolioPerformanceMode = 'historical';
  let status = source ? 'ready' : 'empty', error = '', generation = 0, loader = null;
  let workspaceStatus = 'loading', workspaceError = '', workspaceReopen = false;
  const REOPEN = 'REOPEN_FROM_LAUNCH_URL'; // HB: the Host issues a session only through the URL it printed at launch
  let historyError = null; // a failed collection read stays on History, never the workspace gateway
  let historyGeneration = 0, historyPaging = null, historyInitialized = false;
  let experimentsGeneration = 0, decisionsGeneration = 0, connectionGeneration = 0;
  const historyCursors = new Set();
  const uniqueRows = (rows, key) => { const seen = new Set(); return rows.filter(row => { const id = key(row); if (seen.has(id)) return false; seen.add(id); return true; }); };
  const refusalKey = row => row.entry_id || row.task_id || row.goal_id || JSON.stringify(row);
  let workspace = null, histories = [], historyRefusals = [], inputVersions = [], activeTasks = [], taskRefusals = [], historyCursor = null;
  let taskOwners = new Map();
  let experiments = null, experimentsError = '', experimentRefusals = []; // declared parameters and unreadable per-Task plans from one listing
  let preparation = null; // the preparation owner's readback: with the session, then as the scene re-reads it
  let decisions = null, decisionsError = ''; // U5: what waits on a person, the Host's one answer (PENDING_DECISIONS)
  // A comparison is identified by A, B and the holdings date it was requested at (`comparisonKey`);
  // a response is kept only while that identity is still the selection.
  let comparison = null, comparisonStatus = 'empty', comparisonError = '', comparisonGeneration = 0, comparisonText = '', comparisonKey = null;
  // The one open in flight and the last one asked for: book, date, page and comparison member.
  // Every step after an await checks the open is still current; a newer one supersedes all of it.
  let opening = null, opened = null;
  // Linked Risk reports of the opened book, discovered when the report area is shown; discarded
  // when another book is opened (a date change keeps them: links are per task, not per date).
  let riskLinks = {task: null, links: null, error: ''};
  let sessionToken = null; // Private to transport: never in state, exports or localStorage.
  // A navigation is the person's intent, even when Back returns to the same address.
  // Readers keep this ticket across awaits; an old answer cannot start another navigation.
  let navigationGeneration = 0;
  const beginNavigation = () => {
    if (comparisonStatus === 'loading') resetComparison();
    const ticket = ++navigationGeneration;
    if (typeof Portfolio !== 'undefined') Portfolio.refreshSessionReading?.();
    return ticket;
  };
  const readingObject = () => typeof readingAddress === 'function' ? readingAddress() : (typeof routeObject === 'function' ? routeObject() : null);
  const navigationIntent = () => ({generation: navigationGeneration, page: app.page, object: readingObject()});
  const navigationCurrent = (ticket) => typeof ticket === 'number' ? ticket === navigationGeneration
    : ticket.generation === navigationGeneration && ticket.page === app.page && ticket.object === readingObject();
  const portfolioReading = () => opening?.inPlace && opening.ticket === generation && navigationCurrent(opening.navigation)
    && app.page === 'portfolio' && app.book === opening.task ? opening : null;
  // Share only an unfinished wire read, not its accepted value or a mutation. Each caller
  // still checks its own generation and refusal policy before accepting the owner's answer.
  // A cancelled subscriber leaves immediately; the wire stops only after its last reader.
  const reads = new Map();
  function wireRead(path, signal) {
    if (signal?.aborted) return Promise.reject(cancelled());
    let wire = reads.get(path);
    if (!wire) {
      const controller = typeof AbortController === 'function' ? new AbortController() : null;
      wire = {controller, readers: 0, settled: false, pending: null};
      reads.set(path, wire);
      wire.pending = (async () => {
        const response = await fetch(path, {credentials: 'same-origin', headers: {Accept: 'application/json'}, ...(controller ? {signal: controller.signal} : {})});
        return {ok: response.ok, status: response.status, headers: response.headers, text: await response.text()};
      })().finally(() => {
        wire.settled = true;
        if (reads.get(path) === wire) reads.delete(path);
      });
    }
    wire.readers++;
    return new Promise((resolve, reject) => {
      let done = false;
      const finish = (complete, value) => {
        if (done) return;
        done = true;
        signal?.removeEventListener('abort', abort);
        wire.readers--;
        if (!wire.readers && !wire.settled) {
          // A revisit starts a fresh wire even if the abandoned transport settles late.
          if (reads.get(path) === wire) reads.delete(path);
          wire.controller?.abort();
        }
        complete(value);
      };
      const abort = () => finish(reject, cancelled());
      signal?.addEventListener('abort', abort, {once: true});
      wire.pending.then(value => finish(resolve, value), error => finish(reject, error));
      if (signal?.aborted) abort();
    });
  }
  const get = (key, fallback) => source?.[key] !== undefined ? source[key] : fallback;
  const showingForwardPerformance = () => app.page === 'portfolio' && portfolioPerformanceMode === 'forward';
  const showingRollingPerformance = () => app.page === 'portfolio' && portfolioPerformanceMode === 'historical' && Boolean(source?.rolling_performance);
  const forwardPerformance = () => source?.forward_performance || null;
  const rollingPerformance = () => source?.rolling_performance || null;
  const performanceMetricFields = {total: 'cumulative_return', annual: 'annualized_return', vol: 'annualized_volatility', drawdown: 'maximum_drawdown', sharpe: 'sharpe', sortino: 'sortino'};
  const forwardMetricFields = {annual: 'annualized_return', vol: 'annualized_volatility', drawdown: 'maximum_drawdown', sharpe: 'sharpe', sortino: 'sortino'};
  const percentageMetrics = new Set(['total', 'annual', 'vol', 'drawdown']);
  const metricAbsenceWords = {
    INSUFFICIENT_REALIZED_OBSERVATIONS: 'Fewer than two settled outcomes were recorded',
    ZERO_DOWNSIDE_DEVIATION: 'The recorded outcomes have zero downside deviation',
    NONFINITE_DERIVED_METRIC: 'The owner did not report a finite value',
  };
  const metricAbsences = (supplied, fields = performanceMetricFields) => Object.fromEntries(Object.entries(fields).filter(([, field]) => Object.hasOwn(supplied || {}, field)).map(([, field]) => {
    const value = supplied[field];
    return [field, typeof value === 'string' ? {reason: value, detail: metricAbsenceWords[value] || ''} : value];
  }));
  function performanceMetrics() {
    if (showingForwardPerformance()) {
      const forward = forwardPerformance();
      if (forward?.available !== true) return {};
      const supplied = forward.selected_window_metrics || {};
      // The owner publishes fractions; these are percentage-number display values, as in the
      // historical projection. Sharpe and Sortino remain ratios.
      return Object.fromEntries(Object.entries(forwardMetricFields).filter(([, field]) => Object.hasOwn(supplied, field)).map(([key, field]) => [key, percentageMetrics.has(key) && Number.isFinite(supplied[field]) ? supplied[field] * 100 : supplied[field]]));
    }
    if (!showingRollingPerformance()) return get('metrics', {});
    const supplied = rollingPerformance()?.metrics || {};
    // The owner publishes fractions; these are percentage-number display values, as in the
    // report projection. Ratios remain ratios.
    return Object.fromEntries(Object.entries(performanceMetricFields).filter(([, field]) => Object.hasOwn(supplied, field)).map(([key, field]) => [key, percentageMetrics.has(key) && Number.isFinite(supplied[field]) ? supplied[field] * 100 : supplied[field]]));
  }
  function performanceMetricAbsences() {
    if (showingForwardPerformance()) {
      const forward = forwardPerformance();
      return forward?.available === true ? metricAbsences(forward.selected_window_metric_absences || {}, forwardMetricFields) : {};
    }
    if (!showingRollingPerformance()) return get('metricAbsences', {});
    return metricAbsences(rollingPerformance()?.metric_absences || {});
  }
  function performanceSeries() {
    if (showingForwardPerformance()) {
      const forward = forwardPerformance();
      if (forward?.available !== true || !Array.isArray(forward.series)) return [];
      // These are display aliases for the chart owner; the source return remains the exact
      // per-session net_simple_return supplied by Portfolio. No indexed curve is derived here.
      return forward.series.map((row) => ({...row, date: row.holding_end_session, daily: Number.isFinite(row.net_simple_return) ? row.net_simple_return * 100 : null, benchmark: null, benchmarkDaily: null}));
    }
    if (!showingRollingPerformance()) return get('series', []);
    const rolling = rollingPerformance(), baseByDate = new Map((get('series', []) || []).map((row) => [row.date, row]));
    if (!Array.isArray(rolling?.curve)) return get('series', []);
    // The rolling owner supplies its curve and its cumulative value. The old benchmark is
    // carried only on exact formation dates present in the saved report; a new date stays empty.
    return rolling.curve.map((row) => {
      const date = row.formation_session, base = baseByDate.get(date);
      const benchmark = Object.hasOwn(row, 'benchmark') ? row.benchmark : base?.benchmark ?? null;
      const benchmarkDaily = Object.hasOwn(row, 'benchmarkDaily') ? row.benchmarkDaily : Number.isFinite(row.benchmark_simple_return) ? row.benchmark_simple_return * 100 : base?.benchmarkDaily ?? null;
      return {...row, date, daily: Number.isFinite(row.net_simple_return) ? row.net_simple_return * 100 : null, benchmark: benchmark ?? null, benchmarkDaily: benchmarkDaily ?? null};
    });
  }
  const workingInputs = () => {const v=workspace?.research_context?.data_update;return v?.after || v?.inputs || v || {};};
  // A repaint replaces only the sections that changed; a page not yet painted, or a page that
  // changed, is rendered whole (patchMain falls back to render for those).
  const refresh = () => { if (!document.body.dataset.page) return; (typeof patchMain === 'function' ? patchMain : render)(); };
  // V676: activation changes outside this page need the book owner's current Standing.
  // Read the exact selection; keep its sealed result, paging, comparison and route.
  async function refreshStanding() {
    const shown = source, ticket = generation;
    if (status !== 'ready' || shown?.subject?.source_kind !== 'INSTALLED_RESULT') return false;
    try {
      const value = await read('/api/workbench/portfolio?' + new URLSearchParams({task_id: shown.subject.task_id, portfolio_session: shown.subject.session}));
      if (ticket !== generation || source !== shown || value?.subject?.task_id !== shown.subject.task_id || value.subject.session !== shown.subject.session || value.subject.result_hash !== shown.subject.result_hash) return false;
      source = {...shown, standing: value.standing};
      refresh();
      return true;
    } catch (_) { return false; } // keep the last reading; the ordinary observation retries
  }
  // Page reads share one visit; application observations explicitly use readShared.
  // Cancelling a read never cancels an admitted Task or a mutation.
  let readVisit = null;
  function visitPage(page = app.page) {
    if (readVisit?.page === page && !readVisit.controller?.signal.aborted) return readVisit;
    readVisit?.controller?.abort();
    return readVisit = {page, controller: typeof AbortController === 'function' ? new AbortController() : null};
  }
  function leavePage(page) {
    if (readVisit?.page !== page) return;
    readVisit.controller?.abort(); readVisit = null;
    if(opening){generation++;opening=null;status=source?'ready':'empty';error='';if(source?.subject)app.session=source.subject.session;}
    if (typeof Portfolio !== 'undefined') Portfolio.refreshSessionReading?.();
    if(comparisonStatus==='loading')resetComparison();
    if(riskLinks.links===null)riskLinks={task:null,links:null,error:''};
  }
  const cancelled = () => Object.assign(new Error('Read cancelled'), {name: 'AbortError'});
  async function pageRead(path, allowUnavailable = false) {
    const visit = visitPage();
    const value = await read(path, allowUnavailable, visit.controller?.signal);
    if (visit !== readVisit || visit.controller?.signal.aborted) throw cancelled();
    return value;
  }
  async function pageDocument(path, representation = 'json') {
    const visit = visitPage();
    const value = await request(path, undefined, false, representation, visit.controller?.signal);
    if (visit !== readVisit || visit.controller?.signal.aborted) throw cancelled();
    return value;
  }
  async function request(path, payload, allowUnavailable = false, representation = 'json', signal = undefined) {
    if (payload !== undefined && !sessionToken) throw Error('local_web.session_required_refresh');
    const sentToken = sessionToken;
    let response;
    try {
      response = payload === undefined ? await wireRead(path, signal) : await fetch(path, {credentials: 'same-origin', method: 'POST',
        headers: {Accept: 'application/json', 'Content-Type': 'application/json', 'X-Alphalattice-Session': sessionToken}, body: JSON.stringify(payload)});
    } catch (e) {
      if (e.name === 'AbortError' || signal?.aborted) throw cancelled();
      // No answer at all (the service stopped or is restarting, or the path to it is gone):
      // nothing was read, and a mutation that never reached it admitted nothing. That is not
      // an owner's refusal and is named apart from one, with the browser's own words kept.
      const outage = new Error('local_web.service_unreachable: ' + (payload === undefined ? 'the service could not be reached, so nothing was read' : 'the service could not be reached, so the request was not delivered and nothing was admitted') + ' (' + e.message + '). Read again once the service answers.');
      outage.body = {refused: 'local_web.service_unreachable', transport: e.message};
      throw outage;
    }
    const text = payload === undefined ? response.text : await response.text();
    if (representation === 'html' && response.ok && response.headers?.get('content-type')?.startsWith('text/html')) return {value: null, text};
    let value;
    try { value = JSON.parse(text); } catch (e) {
      const unreadable = new Error('local_web.service_answer_unreadable: the service answered ' + response.status + ' with something other than JSON; nothing is inferred from it. Read again once the service answers.');
      unreadable.body = {refused: 'local_web.service_answer_unreadable', status: response.status};
      throw unreadable;
    }
    // a transport's refusal names its code (`refused`: a string); an answer may hold a list of that name (CONTROLS' refused controls, U73)
    if (!response.ok || String(value.status || value.disposition).startsWith('REFUSED') || (!allowUnavailable && value.failure_code && !value.task_id) || typeof value.refused === 'string') {
      // The mutation was sent and refused before dispatch: the service's preflight did not
      // accept the page's session token (its cookie or header), so nothing was read or
      // admitted. One renewal is tried and checked; the refused request is never resent by
      // itself -- on success the person confirms again, on failure they reload the page.
      if (payload !== undefined && ['local_web.session_token_invalid', 'local_web.session_token_absent'].includes(value.refused)) {
        const renewal = await renewSession(sentToken);
        const refused = value.refused + ': the request was sent and refused before dispatch; its session token was not accepted and nothing was admitted.';
        const outcome = new Error(renewal.renewed
          ? 'local_web.session_renewed: ' + refused + ' The session was renewed; confirm again to send it.'
          : renewal.reopen
            ? 'local_web.session_renewal_failed: ' + refused + ' The Host holds no session for this page (' + renewal.reason + '); open the Workbench again from the launch URL the Host printed, then confirm again.'
            : 'local_web.session_renewal_failed: ' + refused + ' Renewing the session failed (' + renewal.reason + '); reload the page for a new session, then confirm again.');
        outcome.body = value;
        throw outcome;
      }
      const {code, detail} = refusalParts({body:value, message:''});
      const refusal = new Error([code || 'Request refused', detail].filter(Boolean).join(': '));
      refusal.body = value; // the owner's typed refusal fields, for a caller that reads them
      throw refusal;
    }
    return {value, text};
  }
  /* One checked renewal (U1): the session read re-issues the token under the cookie the launch URL
   * gave; the document route issues none (HB). Each answer is checked and the token must be a
   * non-empty string other than the one just refused; anything else is a failed renewal with its
   * reason, and the refused token is left as it is (never presented as new). A refusal naming
   * `REOPEN_FROM_LAUNCH_URL` says the way: the launch URL the Host printed. No mutation is sent here. */
  async function renewSession(refusedToken) {
    let response;
    try {
      response = await wireRead('/api/session');
    } catch (e) {
      return {renewed: false, reason: e.message};
    }
    let session = null;
    try { session = JSON.parse(response.text); } catch (e) { return {renewed: false, reason: response.ok ? 'the session read is not JSON' : 'the session read answered ' + response.status}; }
    if (!response.ok) return {renewed: false, reopen: session?.next_action === REOPEN, reason: 'the session read answered ' + response.status};
    const token = session && typeof session === 'object' ? session.session_token : undefined;
    if (typeof token !== 'string' || !token) return {renewed: false, reason: 'the session read carries no token'};
    if (token === refusedToken) return {renewed: false, reason: 'the session read returned the token that was refused'};
    sessionToken = token;
    return {renewed: true};
  }
  const read = async (path, allowUnavailable = false, signal = undefined) => (await request(path, undefined, allowUnavailable, 'json', signal)).value;
  /* U13 (OW10): an operation's route as the session names it -- the Host's one table (`local_web_session`); a page keeps
   * no map of its own. An operation the session does not name is refused by name, never guessed. */
  const offers = (operation) => Boolean(workspace?.routes?.[operation]?.path);
  // V615: a request that writes (a preview's admission among them) is a POST in the session's routes; a read is not
  const posts = (operation) => workspace?.routes?.[operation]?.method === 'POST';
  function route(operation) {
    if (!offers(operation)) throw Error('local_web.operation_route_absent: ' + operation);
    return workspace.routes[operation].path;
  }
  const post = async (path, payload) => (await request(path, payload)).value;
  function setTasks(tasks, refusals = []) {
    const was = new Map(activeTasks.map((v) => [v.task_id, v.lifecycle]));
    activeTasks = tasks.map((v) => ({...v, status: v.lifecycle, kind: v.task_kind}));
    taskOwners = new Map(activeTasks.map(v => [v.task_id, v]));
    taskRefusals = refusals;
    app.tasks = activeTasks;
    settledSince(was);
    if (was.size && activeTasks.some((v) => was.get(v.task_id) !== v.lifecycle)) resync();
  }
  /* A Task that stopped moving since the last report is said once, by the activity reader (a toast; in
   * the background, a system notification the viewer turned on); the first read reports nothing, and
   * a report of the same move by the other path finds it already moved (the user, 2026-09-25). */
  function settledSince(was) {
    if (!was.size || typeof LiveActivity === 'undefined' || !LiveActivity.taskSettled) return;
    for (const v of activeTasks) {
      const before = was.get(v.task_id);
      if (before && before !== v.lifecycle && stateMoving(before) && !stateMoving(v.lifecycle)) LiveActivity.taskSettled(v);
    }
  }
  /* Projections that arrived with the activity feed: a Task admitted elsewhere joins the list
   * (or replaces its own older projection) so the dock count is truthful between task reads. */
  function mergeTasks(projections) {
    const was = new Map(activeTasks.map((v) => [v.task_id, v.lifecycle]));
    const byId = new Map(activeTasks.map((v) => [v.task_id, v]));
    const refusedById = new Map(taskRefusals.map((v) => [v.task_id, v]));
    let moved = false;
    for (const v of projections) {
      if (v.status === 'REFUSED') { byId.delete(v.task_id); refusedById.set(v.task_id, v); continue; }
      refusedById.delete(v.task_id);
      const previous = byId.get(v.task_id);
      if (previous?.lifecycle !== v.lifecycle) moved = true;
      // Activity reports the version without recovery attention; keep that owner's fact
      // only while it still describes the same canonical Task record.
      const carried = !Object.hasOwn(v, 'attention') && v.task_record_hash && v.task_record_hash === previous?.task_record_hash
        ? {attention: previous.attention} : {};
      byId.set(v.task_id, {...carried, ...v, status: v.lifecycle, kind: v.task_kind});
    }
    activeTasks = [...byId.values()];
    taskOwners = byId;
    taskRefusals = [...refusedById.values()];
    app.tasks = activeTasks;
    settledSince(was);
    if (moved) resync();
  }
  /* A Task's lifecycle as a listing recorded it (the research history, the experiments), overlaid
   * by Task Control's projection when one has arrived since -- the activity feed and the Task reads
   * keep those current: an ended word is final; any other gives way to the newer report (W, the
   * user's reading 2026-09-23: a finished run stayed blue in every list until a reload). */
  const ENDED = new Set(['SUCCEEDED', 'FAILED', 'CANCELLED', 'REFUSED']);
  const lifecycleOf = (taskId, recorded) => (ENDED.has(recorded) ? recorded : activeTasks.find((v) => v.task_id === taskId)?.lifecycle || recorded);
  const rowTaskId = (row) => row.task_id || row.object?.task_id || row.raw?.task_id || row.id;
  const rowTaskHash = (row) => row.task_record_hash || row.object?.task_record_hash || row.raw?.task_record_hash;
  const taskAttention = (row) => {
    const owner = row.attention ? row : row.object?.attention ? row.object : row.raw?.attention ? row.raw : taskOwners.get(rowTaskId(row));
    const fact = owner?.attention, current = taskOwners.get(rowTaskId(row));
    return fact && (!owner.task_record_hash || owner.task_record_hash === fact.task_record_hash) && (!current || current.task_record_hash === fact.task_record_hash) ? fact : null;
  };
  const taskSuccessor = (row) => {
    const fact = taskAttention(row);
    const current = fact && taskOwners.get(fact.successor_task_id);
    return fact?.unresolved === false && fact.resolution === 'SUCCESSOR_SUCCEEDED' && fact.successor_task_id && fact.successor_task_hash && fact.successor_lifecycle === 'SUCCEEDED' && (!current || current.task_record_hash === fact.successor_task_hash) ? fact : null;
  };
  /* A canonical successor is one item; its earlier stop remains an exact, disclosed row.
   * A successor outside this loaded collection has only its owner's exact Task reference. */
  function groupTaskSuccessors(rows) {
    const earlier = new Map(), folded = new Set(), references = new Map(), fronts = new Map(), byTask = new Map();
    for (const row of rows) {
      const id = rowTaskId(row), versions = byTask.get(id) || new Map(), hash = rowTaskHash(row);
      if (!versions.has(hash)) versions.set(hash, row);
      byTask.set(id, versions);
    }
    for (const row of rows) {
      const fact = taskSuccessor(row);
      const successor = fact && byTask.get(fact.successor_task_id)?.get(fact.successor_task_hash);
      if (!fact) continue;
      if (!successor) {
        let front = references.get(fact.successor_task_id);
        if (!front) {
          front = {id: fact.successor_task_id, task_id: fact.successor_task_id, kind: 'Task', name: 'Task', status: stateName(fact.successor_lifecycle), successorReference: fact, earlierStops: []};
          references.set(fact.successor_task_id, front); fronts.set(row, front);
        }
        front.earlierStops.push(row); folded.add(row); continue;
      }
      const stops = earlier.get(successor) || []; stops.push(row); earlier.set(successor, stops); folded.add(row);
    }
    return rows.flatMap(row => fronts.has(row) ? [fronts.get(row)] : folded.has(row) ? [] : [earlier.has(row) ? {...row, earlierStops: earlier.get(row)} : row]);
  }
  /* When Task Control reports a Task's lifecycle moving, the listings are read again -- the
   * history's newest page and the experiments, together -- and the page repaints: a run that ends
   * turns green where it is listed, a new one joins its lists. One read at a time; a move reported
   * meanwhile reads once more after it; a failed read keeps what is painted. */
  let resyncing = false, resyncAgain = false;
  /* V613/V621 (UX01, UX09): a page's read that a Task's move can change (the daily update's outcome and next run) is
   * registered here and read again with the lists, so every page showing it is current without a reload. */
  const taskReaders = new Set();
  const followTasks = (reader) => { taskReaders.add(reader); };
  async function resync() {
    if (workspaceStatus !== 'ready') return; // the first read lists them
    if (resyncing) { resyncAgain = true; return; }
    resyncing = true;
    const ticket = ++historyGeneration;
    try {
      const [value] = await Promise.all([readHistory(new URLSearchParams({history_limit: 50}), ticket), refreshExperiments(true), refreshDecisions(), ...[...taskReaders].map((reader) => reader())]);
      if (!value || ticket !== historyGeneration) return;
      const first = uniqueRows(value.entries || [], row => row.entry_id), ids = new Set(first.map((e) => e.entry_id));
      histories = [...first, ...histories.filter((e) => !ids.has(e.entry_id))]; // the pages read further down stay
      historyRefusals = uniqueRows([...(value.blocked_entries || []), ...historyRefusals], refusalKey);
      if (!historyInitialized || (!historyCursors.size && histories.length <= first.length && historyCursor)) historyCursor = historyCursors.has(value.next_cursor) ? null : value.next_cursor;
      historyInitialized = true;
    } catch (e) {
      // nothing is inferred from a failed read: the lists keep what they showed
    } finally {
      resyncing = false;
      refresh();
      if (resyncAgain) { resyncAgain = false; resync(); }
    }
  }
  function setInputs(body) {
    inputVersions=(body?.inputs || []).flatMap(item=>(item.versions || []).map(v=>({
      id:item.input_id,binding_hash:v.binding_hash,name:item.input_id,date:v.end || '',
      start:v.start || null,cutoff:v.end || null,available:v.available===true,lifecycle:v.lifecycle,pinned:v.configured_default===true})));
    app.inputs=inputVersions;
  }
  const load = (next) => settle(++generation, next);
  function acceptSource(value, keepReadingPlace = false) {
    source = value && typeof value === 'object' ? value : null;
    status = source ? 'ready' : 'empty';
    if (!source?.subject) return;
    app.book = source.subject.task_id;
    app.input = source.subject.input_id;
    app.session = source.subject.session;
    if (!keepReadingPlace) { app.holdingFocus = null; if (Window.inspectorMode() === 'holding') Window.closeInspector(false); } // another book: its holding closes; a new date of the same book keeps it (Portfolio.followHolding)
    app.holdingsPage = 0;
    if (!keepReadingPlace) {
      Inspect.resetWindow();
      Inspect.selectObservation(Math.max(0, (performanceSeries().length || 1) - 1));
    }
  }
  async function settle(ticket, next) {
    if (opening && opening.ticket !== ticket) opening = null; // a plain load supersedes an unfinished open
    if (typeof next === 'function') loader = next;
    status = 'loading'; error = ''; refresh();
    try {
      const value = await (typeof next === 'function' ? next() : next);
      if (ticket !== generation) return;
      acceptSource(value);
    } catch (reason) {
      if (ticket !== generation) return;
      status = 'error'; error = String(reason?.message || reason);
    }
    refresh();
  }
  const kindName = (kind) => ({
    'portfolio.policy-development': 'Portfolio study', 'alpha.model-development': 'Alpha modeling', // law 134: the dock's word for the kind's list
    'factor.screening-development': 'Factor screening', 'risk.covariance-development': 'Risk modeling',
    'CRO_REVIEW': 'CRO review', 'INSTALLED_RESULT': 'Installed strategy result', 'TASK_RECORD': 'Task record',
    'CONTINUOUS_UPDATE': 'Continuous update',
  })[kind] || kind;
  const stateName = (state) => ({
    SUCCEEDED: 'historical', BLOCKED: 'blocked', CANCELLED: 'cancelled',
    RECOVERY_REQUIRED: 'recovery_required', DEFERRED: 'deferred', RUNNING: 'running',
    QUEUED: 'queued', REVIEW_PENDING: 'review_pending',
  })[state] || 'metadata';
  // The input version an entry or study was bound to. Its recorded cutoff is absent (null) when the
  // owner recorded none; `available` is local source-file presence, never integrity verification.
  const inputVersion = (bindingHash) => (bindingHash ? inputVersions.find((v) => v.binding_hash === bindingHash) || null : null);
  const dateOf = (iso) => (iso ? dayOf(iso) || String(iso).slice(0, 10) : null); // N3 (law 133): the reader's day
  // Declared parameters come from the existing experiment listing, joined by task id: one
  // request for the whole collection, never a readback per row, and absent when it was refused.
  const declaredOf = (taskId) => (taskId && experiments ? experiments.find((v) => v.task_id === taskId) || null : null);
  const historyRows = () => histories.map((entry) => {
    const declared = declaredOf(entry.task_id); // a review shares its subject's task id; only same-kind rows join
    const facts = LiveViews.studyFacts(declared?.kind === entry.kind ? declared : {kind: entry.kind}, entry);
    return {
      id: entry.entry_id, kind: kindName(entry.kind), status: stateName(lifecycleOf(entry.task_id, entry.status)),
      name: kindName(entry.kind), summary: facts.summary, words: facts.words, wordsMarkup: facts.wordsMarkup, reference: facts.reference, ref: facts.ref,
      origin: facts.origin, interval: facts.interval, mode: 'readback',
      input: entry.input_id || '', ...LiveViews.inputState(entry.input_binding_hash),
      recordedAt: dateOf(entry.recorded_at), holdingsSession: entry.book?.portfolio_session || null,
      note: entry.failure_code || '', task_id: entry.task_id, raw: entry,
    };
  });
  const portfolioEntries = () => histories.filter((x) => ['portfolio.policy-development','INSTALLED_RESULT'].includes(x.kind) && x.status === 'SUCCEEDED');
  /* Runs (round 73): everything that ran or runs as one shape -- {id, kind, name, state, starter,
   * started, finished, running_seconds, current, verified, object} -- from the owners' projections: the Tasks
   * ('task'), the data updates and preparations among them ('update'), the Team's retained
   * sessions ('session', the Team's own reading). `groupByDay` orders runs newest first under
   * the date, by the owner's instant -- the date itself, never a relative day word (round 96:
   * a recorded instant is a fact; a word about the present is a claim the replay must not make). */
  const RUN_KINDS = {update: new Set(['workspace_data_update', 'workspace_preparation'])};
  function runsOf(kind = 'task') {
    if (kind === 'session') return typeof LiveTeam !== 'undefined' && LiveTeam.runs ? LiveTeam.runs() : [];
    const starter = (id) => (typeof LiveActivity !== 'undefined' && LiveActivity.starterOf ? LiveActivity.starterOf(id) : '');
    return activeTasks.filter((v) => kind === 'task' || RUN_KINDS[kind].has(v.task_kind)).map((v) => ({id: v.task_id, kind: v.task_kind, name: LiveViews.nameOf(v).name, markup: LiveViews.nameOf(v).markup || null, state: v.lifecycle, starter: starter(v.task_id), started: v.running_since || '', finished: stateMoving(v.lifecycle) ? '' : v.last_activity_at || '', running_seconds: v.running_seconds, current: v.current_stage || '', verified: [v.verified_stage_count, v.total_stage_count], object: v}));
  }
  const runAt = (r) => String(r.finished || r.started || '');
  function groupByDay(runs) {
    const dayKey = (r) => { const d = dayOf(runAt(r)); return !d ? t('Undated') : when(d); }; // N3 (law 133): the reader's day
    const groups = new Map();
    for (const r of [...runs].sort((a, b) => runAt(b).localeCompare(runAt(a)))) { const k = dayKey(r); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(r); }
    return groups;
  }
  async function refreshExperiments(keep = false) { // keep: a background re-read that fails keeps the listing it had
    const ticket = ++experimentsGeneration;
    try {
      const value = await read('/api/experiments');
      if (ticket !== experimentsGeneration) return;
      experiments = value.experiments || []; experimentRefusals = value.refusals || []; experimentsError = '';
    } catch (reason) {
      if (ticket !== experimentsGeneration) return;
      if (keep && experiments) return;
      experiments = []; experimentRefusals = []; experimentsError = String(reason?.message || reason); // rows fall back to history metadata
    }
  }
  /* What waits on a person (U5): the Host reads every owner's part at one moment and names each
   * decision's request; the page reads that answer whole and never assembles its own. A failed read
   * keeps the rows it had and says it was not read. */
  async function refreshDecisions() {
    const ticket = ++decisionsGeneration;
    try {
      const value = await read('/api/decisions');
      if (ticket !== decisionsGeneration) return;
      decisions = value.decisions || []; decisionsError = '';
    } catch (reason) {
      if (ticket !== decisionsGeneration) return;
      decisionsError = String(reason?.message || reason);
    }
  }
  async function readHistory(query, ticket) {
    try {
      const value = await read('/api/research-history?' + query);
      if (ticket === historyGeneration) historyError = null;
      return value;
    } catch (reason) {
      if (ticket === historyGeneration) historyError = reason?.body || {status: 'REFUSED', detail: 'History could not be read'};
      return null;
    }
  }
  async function refreshHistory(more = false) {
    if (more && !historyCursor) return;
    if (more && historyPaging?.generation === historyGeneration && historyPaging.cursor === null) return historyPaging.promise;
    if (more && historyPaging?.generation === historyGeneration && historyPaging.cursor === historyCursor) return historyPaging.promise;
    const ticket = more ? historyGeneration : ++historyGeneration, cursor = more ? historyCursor : null;
    if (!more) historyCursors.clear();
    const query = new URLSearchParams({history_limit: 50});
    if (cursor) query.set('history_cursor', cursor);
    const paging = {generation: ticket, cursor, promise: null};
    historyPaging = paging;
    paging.promise = (async () => {
      try {
        const [value] = await Promise.all([readHistory(query, ticket), more ? null : refreshExperiments()]);
        if (ticket !== historyGeneration || (more && cursor !== historyCursor)) return;
        if (!value) return;
        histories = uniqueRows([...(more ? histories : []), ...(value.entries || [])], row => row.entry_id);
        historyRefusals = uniqueRows([...(more ? historyRefusals : []), ...(value.blocked_entries || [])], refusalKey);
        if (cursor) historyCursors.add(cursor);
        historyCursor = value.next_cursor && !historyCursors.has(value.next_cursor) ? value.next_cursor : null;
        historyInitialized = true;
      } finally {
        if (historyPaging === paging) historyPaging = null;
        refresh();
      }
    })();
    refresh();
    return paging.promise;
  }
  function bareEntry(h = hashParams(), q = new URLSearchParams(location.search)) {
    return !h.get('page') && !q.has('review_experiment') && !q.has('review_update') && !h.get('book') && !q.get('task_id') && !h.get('study') && !h.get('foundation') && !h.get('task') && !q.get('task') && !h.get('plan') && !q.get('plan') && !q.get('history') && q.get('panel') !== 'workspace' && !h.get('feature_plan') && !q.get('feature_plan');
  }
  function entryPage(h = hashParams(), q = new URLSearchParams(location.search)) {
    if (!bareEntry(h, q)) return null;
    if (h.get('follow') || preparation && !preparation.inputs?.length) return 'overview';
    return decisions?.some(d => d.kind === 'UPGRADE') ? 'upgrade' : null;
  }
  async function connect() {
    const ticket = ++connectionGeneration;
    let navigation = navigationIntent();
    // The first open shows the loading box; a later read (Refresh) keeps the page painted and
    // repaints once the owners have answered, so nothing blinks or restarts.
    const again = workspaceStatus === 'ready';
    if (!again) { workspaceStatus = 'loading'; refresh(); }
    workspaceError = ''; workspaceReopen = false;
    try {
      // round 94 (the latency table): the history and experiments reads need nothing from the
      // session and start beside it; the object's own read waits for it (the page's admission)
      const [session] = await Promise.all([read('/api/session?context=1'), refreshHistory(), refreshDecisions()]);
      if (ticket !== connectionGeneration) return;
      const {session_token: token, ...safe} = session;
      sessionToken = token;
      workspace = safe;
      const context = safe.research_context || {};
      setInputs(context.inputs);
      setTasks(context.tasks?.tasks || [], context.tasks?.refusals || []);
      app.workspace = safe.workspace_id;
      preparation = context.preparation?.status ? context.preparation : null;
      app.data = workingInputs().market_through || workingInputs().data_through || '';
      app.feature = workingInputs().panel_through || '';
      app.tasks = activeTasks;
      workspaceStatus = 'ready';
      // the chrome names the workspace as soon as it is known: a page that patches its sections
      // (the Team console) never re-renders the shell by itself (round 31)
      if (typeof Window !== 'undefined') Window.render();
      // Connection populates the shared workspace, but its launch link belongs to the
      // initial navigation. A page chosen while these reads waited owns every way on.
      if (!navigationCurrent(navigation)) { LiveResearch.ready(); refresh(); return; }
      const q = new URLSearchParams(location.search);
      const h = new URLSearchParams(location.hash.slice(1));
      const selected = h.get('book') || q.get('task_id');
      const featurePlan = h.get('feature_plan') || q.get('feature_plan');
      const reviewAddress = q.has('review_experiment') || q.has('review_update');
      // A bare followed entry starts on Home until its owner has work to show. Without a
      // verified research input Home opens preparation; explicit destinations keep their route.
      const bare = bareEntry(h, q);
      const entrance = entryPage(h, q);
      if (entrance) { app.page = entrance; replaceHash({page: entrance}); }
      if (bare) navigation = navigationIntent();
      if (!h.get('page') && reviewAddress) {
        // These issued addresses name the owner's exact book, not a latest Portfolio result.
        // Keep missing/empty/mixed fields for its existing typed selector refusals.
        const fields = [['review_experiment', 'experiment_task_id'], ['r', 'experiment_receipt_hash'], ['d', 'portfolio_session'], ['review_update', 'update_task_id'], ['p', 'update_publication_hash'], ['b', 'position_basis']];
        const selector = Object.fromEntries(fields.filter(([key]) => q.has(key)).map(([key, field]) => [field, q.get(key)]));
        const pending = LiveReview.open(selector, '', 'handoff');
        navigation = navigationIntent();
        await pending;
      } else if (selected && !h.get('study') && !h.get('foundation') && /^[0-9a-f-]{36}$/i.test(selected)) {
        const known = experiments?.find(v=>v.task_id===selected);
        const studyPage = {'factor.screening-development':'factor','alpha.model-development':'alpha','risk.covariance-development':'risk'}[known?.kind];
        if (studyPage) {app.book='';replaceHash({book:'',session:''});await LiveStudy.open(selected,studyPage);}
        else {
          const pending = openPortfolio(selected, h.get('session') || q.get('portfolio_session'), h.get('page') || 'portfolio', h.get('compare') || null, null, h.get('performance') === 'forward' || q.get('performance') === 'forward' ? 'forward' : 'historical');
          navigation = navigationIntent();
          await pending;
        }
      } else if (!h.get('page') && q.get('history')) {
        navigation = beginNavigation();
        await openEntry(q.get('history'), false, q.get('portfolio_session'), navigation);
      } else {
        if (!h.get('page') && q.get('panel') === 'workspace') { app.page = 'data'; replaceHash({page: 'data'}); }
        refresh();
      }
      if (ticket !== connectionGeneration || !navigationCurrent(navigation)) return;
      // A shared PLAN link opens the editor with that PLAN beside the draft; never adopted by
      // itself, and never over a page the reader had moved on to (a reload keeps their page).
      const plan = h.get('page') && h.get('page') !== 'lab' ? null : h.get('plan') || q.get('plan');
      if (plan) { app.page = 'lab'; replaceHash({page: 'lab', plan}); refresh(); }
      LiveResearch.ready();
      const task = hashParams().get('task') || (!h.get('task') ? q.get('task') : '');
      if (task && !hashParams().get('task')) replaceHash({task});
      Inspect.reopenFromAddress();
      if (featurePlan && !h.get('page') && !task) { app.page = 'features'; replaceHash({page: 'features', feature_plan: featurePlan}); refresh(); } // the client's link to a saved PLAN opens its page
      const follow = h.get('follow');
      if (follow) LiveActivity.setFollowing(follow, h.get('follow_paused') === '1');
    } catch (reason) {
      if (ticket !== connectionGeneration) return;
      workspaceStatus = 'error'; workspaceError = String(reason?.message || reason);
      workspaceReopen = reason?.body?.next_action === REOPEN; // U1: no session the Host accepts; the launch URL is the way
      refresh();
    }
  }
  /* Open one saved book at one date. The projection is the only read: its verified body already
   * carries the policy and source references the page shows. The comparison member travels with
   * the open and is requested, on the compare page, only once this same open is still current. */
  function openPortfolio(task, session, page = 'portfolio', compareWith = null, place = null, performanceMode = 'historical', navigation = null) {
    performanceMode = performanceMode === 'forward' ? 'forward' : 'historical';
    if (opening && navigationCurrent(opening.navigation) && opening.task === task && opening.session === (session || null) && opening.page === page && opening.compareWith === (compareWith || '') && opening.performanceMode === performanceMode) return opening.promise;
    const pending = readPortfolio(task, session, page, compareWith, place, performanceMode, navigation);
    if (opening) opening.promise = pending;
    return pending;
  }
  async function readPortfolio(task, session, page, compareWith, place, performanceMode, navigation) {
    // the same book at another holdings date is a patch of the shown book; the book must be the
    // one shown (`app.book`), not only the one last read -- after the way up the list is shown
    // and the source is still held (2026-09-21, the user's reading: the row did nothing, twice)
    const sameBookSession = page === 'portfolio' && app.page === 'portfolio' && app.book === task && status === 'ready' && source?.subject?.task_id === task && Boolean(session);
    const sessionPlace = sameBookSession ? place || {x: typeof scrollX === 'number' ? scrollX : 0, y: typeof scrollY === 'number' ? scrollY : 0} : null;
    const previous = {source, opened, session: source?.subject?.session || app.session, performanceMode: portfolioPerformanceMode};
    const intent = {ticket: ++generation, navigation: navigation ?? beginNavigation(), task, session: session || null, page, compareWith: compareWith || '', performanceMode, inPlace: sameBookSession};
    const holdingsOnlyRead = sameBookSession
      && Boolean(session)
      && session !== previous.session
      && intent.performanceMode === 'historical'
      && previous.performanceMode === 'historical';
    opening = opened = intent;
    portfolioPerformanceMode = intent.performanceMode;
    resetComparison();
    if (riskLinks.task !== task) riskLinks = {task: null, links: null, error: ''};
    // This reader writes the book's route in place. The one history entry for the place being
    // left -- the exact Team scene (session, actor, event), a study, the History list -- is
    // made by the explicit navigation that called it (`objectEntry`), never here: a load, a
    // Back/Forward re-open, a date change and a reload pass through this same open.
    app.page = page; app.book = task; app.compareOther = intent.compareWith;
    visitPage(page); // establish the destination visit before its loading repaint closes the prior page
    const params = {task_id: task};
    if (session) params.portfolio_session = session;
    if (intent.performanceMode === 'forward') params.performance = 'latest';
    if (holdingsOnlyRead) params.scope = 'holdings';
    replaceHash({page, book: task, session: session || '', compare: intent.compareWith, performance: intent.performanceMode === 'forward' ? 'forward' : '', study: '', foundation: ''});
    if (sameBookSession) {
      // The selector and retained rows describe the accepted date until this read answers.
      // The open's requested date still owns rapid steps and supersedes an earlier choice.
      app.session = previous.session;
      if (typeof Portfolio !== 'undefined') Portfolio.refreshSessionReading?.();
      try {
        let value = await pageRead('/api/workbench/portfolio?' + new URLSearchParams(params));
        if (intent.ticket !== generation || !navigationCurrent(intent.navigation)) return;
        if (holdingsOnlyRead) {
          const prior = previous.source?.subject;
          const sameSavedResult = typeof prior?.result_hash === 'string'
            && prior.result_hash.length > 0
            && value?.subject?.task_id === task
            && value.subject.task_id === prior.task_id
            && value.subject.session === session
            && value.subject.result_hash === prior.result_hash;
          if (sameSavedResult) {
            const retained = {};
            for (const key of ['rolling_performance', 'forward_holdings']) {
              if (Object.hasOwn(previous.source, key)) retained[key] = previous.source[key];
            }
            value = {...value, ...retained};
          } else {
            // A scoped response may not carry new daily facts. If its saved result identity moved,
            // get the full owner read before accepting it; never attach the prior facts by task id alone.
            const fullParams = {...params};
            delete fullParams.scope;
            value = await pageRead('/api/workbench/portfolio?' + new URLSearchParams(fullParams));
            if (intent.ticket !== generation || !navigationCurrent(intent.navigation)) return;
          }
        }
        if (!value?.subject || value.subject.task_id !== task) throw new Error('portfolio_application.result_readback_mismatch');
        error = '';
        acceptSource(value, true);
        if (intent.performanceMode === 'forward' || previous.performanceMode !== intent.performanceMode) {
          Inspect.resetWindow();
          Inspect.selectObservation(Math.max(0, performanceSeries().length - 1));
        }
      } catch (reason) {
        if (intent.ticket !== generation || !navigationCurrent(intent.navigation)) return;
        source = previous.source; opened = previous.opened; opening = null; app.session = previous.session; portfolioPerformanceMode = previous.performanceMode;
        replaceHash({session: previous.session, performance: previous.performanceMode === 'forward' ? 'forward' : ''});
        if (typeof Portfolio !== 'undefined') Portfolio.refreshSessionSurface(sessionPlace);
        notify(t('Session unchanged') + ': ' + String(reason?.message || reason));
        return;
      }
      opening = null;
      if (!navigationCurrent(intent.navigation) || app.book !== task || app.page !== page) return;
      replaceHash({session: source.subject.session});
      if (typeof Portfolio !== 'undefined') Portfolio.refreshSessionSurface(sessionPlace);
      if (typeof Portfolio !== 'undefined') Portfolio.refreshPerformanceSurface();
      return;
    }
    await settle(intent.ticket, () => pageRead('/api/workbench/portfolio?' + new URLSearchParams(params)));
    if (intent.ticket !== generation) return; // a newer open (another book, or this book at another date) owns the page now
    opening = null;
    if (status !== 'ready' || source?.subject?.task_id !== task) return;
    // The route is written only while this book is still the view being shown: a reader who
    // moved to another page meanwhile keeps that page and route; the book stays opened.
    if (!navigationCurrent(intent.navigation) || app.book !== task || app.page !== page) return;
    replaceHash({session: source.subject.session});
    if (app.page === 'compare' && intent.compareWith) await compare(intent.compareWith); // elsewhere the member is kept and read when the compare page is shown
  }
  async function rereadPortfolioPerformance() {
    if (portfolioReading() || app.page !== 'portfolio' || document.visibilityState === 'hidden' || status !== 'ready' || source?.subject?.source_kind !== 'INSTALLED_RESULT' || !source?.subject?.task_id) return;
    return openPortfolio(source.subject.task_id, source.subject.session || null, 'portfolio', app.compareOther || null, null, portfolioPerformanceMode);
  }
  taskReaders.add(rereadPortfolioPerformance);
  function resetComparison() {
    ++comparisonGeneration;
    comparison = null; comparisonStatus = 'empty'; comparisonError = ''; comparisonText = ''; comparisonKey = null;
  }
  // Discovery for the opened book, started when the report area is shown; one request per book
  // unless explicitly refreshed. A response is kept only while that book is still open.
  async function discoverRiskLinks(task, force = false) {
    if (!task || (!force && riskLinks.task === task)) return false;
    if (source?.subject?.task_id === task && source.subject.source_kind === 'INSTALLED_RESULT') {
      riskLinks = {task, links: [], error: ''}; return true;
    }
    const requestState = {task, links: null, error: ''};
    riskLinks = requestState;
    const settled = (next) => { if (riskLinks === requestState && source?.subject?.task_id === task) riskLinks = next; };
    try {
      const value = await pageRead('/api/experiments/risk-links?' + new URLSearchParams({task_id: task}));
      settled({task, links: value.links || [], refusals: value.refused_links || [], error: ''});
    } catch (reason) {
      if(reason.name==='AbortError'){if(riskLinks===requestState)riskLinks={task:null,links:null,error:''};return false;}
      settled({task, links: null, error: String(reason?.message || reason)});
    }
    return true;
  }
  /* The saved result of one installed Portfolio replay Task, through the owners the Host's own
   * artifact resolver uses: the Portfolio index names each result with the Task that first produced
   * it, and the REPORT of that exact result names every Task that used it (`used_by_task_ids`, U10).
   * Exact Task metadata names its publishing or reusing manifest, then one REPORT verifies it.
   * `null` when no result names the Task; an answer for another object is a typed refusal. No cache:
   * every ask is a read. */
  async function installedResult(task) {
    const visit=visitPage();
    const selectedRead=async path=>{const value=await read(path,false,visit.controller?.signal);if(visit!==readVisit || visit.controller?.signal.aborted)throw cancelled();return value;};
    const collection=await selectedRead('/api/results?'+new URLSearchParams({task_id:task}));
    if(collection.refusals?.length){const refused=collection.refusals[0],error=new Error(refused.failure_code);error.body=refused;throw error;}
    const listed=(collection.results || [])[0];if(!listed)return null;
    if(listed.task_id!==task)throw Error('portfolio_application.result_readback_mismatch');
    const report=await selectedRead('/api/report?'+new URLSearchParams({result_hash:listed.result_hash}));
    if(report.result_hash!==listed.result_hash || !(report.used_by_task_ids || []).includes(task))throw Error('portfolio_application.result_readback_mismatch');
    return report;
  }

  /* Open one saved object the reader chose (History, a dossier card, an activity row, the
   * team scene, a follow). A reader that navigates -- a review, a study, a saved Portfolio --
   * first makes the one history entry for the place being left (`objectEntry`); the metadata
   * dialog of a record without a connected view changes no route and makes none. */
  // `push`: a reader's own choice makes one history entry (objectEntry); a step between the
  // records of a list (round 53) makes none, so Back returns to the list, not to the last record
  async function openEntry(id, push = true, portfolioSession = null, navigation = null) {
    if (navigation === null) { beginNavigation(); navigation = navigationIntent(); }
    const study = id.startsWith('experiment:') ? experiments?.find(v=>v.task_id===id.slice('experiment:'.length)) : null;
    let entry = histories.find((x) => x.entry_id === id) || (study ? {...study,status:study.lifecycle} : null);
    if (!entry && (id.startsWith('result:') || id.startsWith('task:'))) {
      let exact;
      try { exact = await pageRead('/api/research-history?' + new URLSearchParams({history_entry_id:id})); }
      catch (error) { if (navigationCurrent(navigation)) throw error; return; }
      if (!navigationCurrent(navigation)) return;
      entry = (exact.entries || []).find(x=>x.entry_id===id) || null;
      if (entry) histories.push(entry);
    }
    if (!entry) {alertDialog(t('Saved reference unavailable'),t('This exact reference was not discovered. No latest result was selected.'),btn(t('Close'),'close','','button'));return;} // an alert hides its X: its one answer closes it
    if (entry.kind === 'TASK_RECORD' && entry.task_id) return LiveTasks.open(entry.task_id);
    if(entry.review_publication_hash && entry.book) { if (push) objectEntry('review:' + entry.review_publication_hash); return LiveReview.open(entry.book, entry.review_publication_hash); }
    const studyPage={'factor.screening-development':'factor','alpha.model-development':'alpha','risk.covariance-development':'risk'}[entry.kind];
    if(studyPage && entry.task_id) { if (push) objectEntry('study:' + entry.task_id); return LiveStudy.open(entry.task_id,studyPage); }
    if (['portfolio.policy-development','INSTALLED_RESULT'].includes(entry.kind) && entry.status === 'SUCCEEDED') {
      if (push) objectEntry('book:' + entry.task_id);
      return openPortfolio(entry.task_id, portfolioSession || entry.book?.portfolio_session, app.page === 'compare' ? 'compare' : 'portfolio', null, null, 'historical', navigationGeneration); // Compare's list opens the study in Compare
    }
    LiveViews.existing(id);
  }
  // A, B and the holdings date travel in the route (`book`, `session`, `compare`). The owner reads
  // both books at that date through its existing portfolio_session field; changing any of the
  // three invalidates the shown comparison and its export, and a slow older response never
  // relabels a newer selection. While an open is still loading, the choice joins that open
  // instead of being compared against the book it is replacing.
  async function compare(task) {
    visitPage(app.page);
    app.compareOther = task || '';
    replaceHash({compare: app.compareOther});
    if (opening) { opening.compareWith = app.compareOther; resetComparison(); refresh(); return; }
    const key = selection();
    if (sameComparison(key) && comparisonStatus !== 'error') return; // already shown, or on its way
    resetComparison();
    if (!key.left || !key.right) { refresh(); return; }
    const ticket = comparisonGeneration;
    const navigation = navigationIntent();
    comparisonKey = key; comparisonStatus = 'loading'; refresh();
    try {
      let path;
      if (source?.subject?.source_kind === 'INSTALLED_RESULT') {
        const other = await installedResult(key.right);
        if (ticket !== comparisonGeneration || !navigationCurrent(navigation)) return;
        if (!other) throw new Error('portfolio_application.mixed_result_comparison_not_supported');
        path = '/api/compare?' + new URLSearchParams({left:source.subject.result_hash,right:other.result_hash});
      } else path = '/api/experiments/compare?' + new URLSearchParams({left_task_id: key.left, right_task_id: key.right, ...(key.session ? {portfolio_session: key.session} : {})});
      const {value, text} = await pageDocument(path);
      if (ticket !== comparisonGeneration || !navigationCurrent(navigation) || !sameComparison(selection())) return;
      comparison = value; comparisonText = text; comparisonStatus = 'ready';
    } catch (reason) {
      if (ticket !== comparisonGeneration || !navigationCurrent(navigation) || !sameComparison(selection())) return;
      comparisonStatus = 'error'; comparisonError = String(reason?.message || reason);
    }
    refresh();
  }
  const selection = () => ({left: source?.subject?.task_id || '', right: app.compareOther || '', session: source?.subject?.session || ''});
  const sameComparison = (key) => Boolean(comparisonKey) && comparisonKey.left === key.left && comparisonKey.right === key.right && comparisonKey.session === key.session;
  // The owner's exact response text, only while it still describes the selected pair and date.
  function exportComparison() {
    const key = selection();
    if (comparisonStatus !== 'ready' || !comparison || !sameComparison(key) || comparison.left?.task_id !== key.left || comparison.right?.task_id !== key.right) throw new Error('Comparison export unavailable for the selected pair');
    download(comparisonText, 'AlphaLattice-comparison-' + key.left + '-' + key.right + (key.session ? '-' + key.session : '') + '.json');
  }
  async function exportStudy(format = 'json') {
    const subject = source?.subject;
    if (!subject) return;
    if (subject.source_kind === 'INSTALLED_RESULT') {
      if (format === 'html') {
        const saved = await pageDocument('/report?' + new URLSearchParams({result_hash:subject.result_hash}), 'html');
        download(saved.text,'AlphaLattice-' + subject.task_id + '.html','text/html');
      } else {
        const value = await pageRead('/api/report?' + new URLSearchParams({result_hash:subject.result_hash,portfolio_session:subject.session}));
        download(JSON.stringify(value,null,2),'AlphaLattice-' + subject.task_id + '.json','application/json');
      }
      return;
    }
    const value = await pageRead('/api/experiments/export?' + new URLSearchParams({
      task_id: subject.task_id, portfolio_session: subject.session,
    }));
    if (typeof value[format] !== 'string') throw new Error('Export format unavailable');
    download(value[format], 'AlphaLattice-' + subject.task_id + '.' + format, format === 'html' ? 'text/html' : 'application/json');
  }
  return {
    read: pageRead, readShared: read, visitPage, leavePage, post, route, offers, posts, navigationIntent, navigationCurrent, beginNavigation, uniqueRows, readDocument:pageDocument, postDocument:(path,payload)=>request(path,payload),
    setTasks, mergeTasks, followTasks, setInputs, entryPage, connect, runsOf, kindName, groupByDay, openEntry, openPortfolio, installedResult, refreshHistory, refreshStanding, compare, exportComparison, exportStudy,
    riskLinks: (task) => (task && riskLinks.task === task ? riskLinks.links : null), riskLinkRefusals: (task) => (task && riskLinks.task === task ? riskLinks.refusals || [] : []), get riskLinksError() { return riskLinks.error; },
    discoverRiskLinks: async (task) => { if (await discoverRiskLinks(task)) refresh(); },
    refreshRiskLinks: async (task) => { await discoverRiskLinks(task, true); refresh(); },
    get status() { return status; }, get error() { return error; },
    get ready() { return status === 'ready'; },
    get workspaceStatus() { return workspaceStatus; }, get workspaceError() { return workspaceError; }, get workspaceReopen() { return workspaceReopen; },
    get comparisonStatus() { return comparisonStatus; }, get comparisonError() { return comparisonError; },
    // Reloading a book repeats the same open: book, shown date, page and comparison member.
    load, reload: () => workspaceStatus === 'error' ? connect() : opened ? openPortfolio(opened.task, source?.subject?.task_id === opened.task ? source.subject.session : opened.session, app.page, app.compareOther || null, null, opened.performanceMode || 'historical') : (loader ? load(loader) : undefined),
    workspace: () => workspace?.workspace_id || '',
    /* Round 65: the objects the viewer opened last (the router keeps their keys), then the newest
     * recorded ones — the Home's third group and the command menu's Recent. Local, not a record. */
    recent: (n = 8) => { const opened = readPreference('objects.recent'), ids = Array.isArray(opened) ? opened.map((k) => String(k).split(':').slice(1).join(':')) : []; const rows = historyRows(); const first = ids.map((id) => rows.find((r) => r.task_id === id || r.id.endsWith(':' + id))).filter(Boolean); return groupTaskSuccessors([...new Set([...first, ...rows])]).slice(0, n); },
    workspaceFacts: () => workspace, history: historyRows, historyRefusals: () => historyRefusals, historyError: () => historyError, portfolioEntries, hasMoreHistory: () => Boolean(historyCursor), get historyLoading() { return Boolean(historyPaging); },
    experiments: () => experiments, experimentRefusals: () => experimentRefusals, get experimentsError() { return experimentsError; }, refreshExperiments, declaredOf, inputVersion, lifecycleOf,
    decisions: () => decisions, get decisionsError() { return decisionsError; }, refreshDecisions,
    preparation: () => preparation, setPreparation: (body) => { preparation = body?.status ? body : preparation; },
    inputs: () => inputVersions, tasks: () => activeTasks, taskRefusals: () => taskRefusals, comparison: () => comparison,
    taskAttention, taskSuccessor, groupTaskSuccessors, taskOf: id => taskOwners.get(id) || null,
    actionableTasks: () => activeTasks.filter(task => taskAttention(task)?.unresolved === true && !(decisions || []).some(d => d.kind === 'STOPPED_TASK' && d.task_id === task.task_id && d.waits_on === 'AGENT' && taskAttention(d)?.task_record_hash === task.task_record_hash)),
    unrecoverableTasks: () => activeTasks.filter(task => { const fact = taskAttention(task); return fact?.unresolved === false && fact.resolution === 'UNRECOVERABLE'; }),
    comparisonKey: () => comparisonKey, comparisonShown: () => !app.compareOther || sameComparison(selection()),
    subject: () => get('subject', null), schema: () => get('schema', ''),
    notice: () => t(get('notice', '')), now: () => get('now', ''),
    clocks: () => ({data: workingInputs().market_through || workingInputs().data_through || '', feature: workingInputs().panel_through || '',
      input: inputVersions.find((v) => v.binding_hash === source?.subject?.input_hash)?.date || ''}),
    universe: () => get('universe', {eligible: null, total: null, quarantine: null, definition: ''}),
    series: performanceSeries, metrics: performanceMetrics, metricAbsences: performanceMetricAbsences, benchmarkMetrics: () => get('benchmarkMetrics', {}),
    performanceMode: () => portfolioPerformanceMode, forwardPerformance, rollingPerformance,
    holdings: () => get('holdings', []), sessions: () => get('sessions', []), cro: () => get('cro', null),
    portfolioReading: () => Boolean(portfolioReading()), portfolioSession: () => portfolioReading()?.session || source?.subject?.session || app.session,
    raw: () => source,
  };
})();
