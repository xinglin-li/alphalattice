/* Backend-connected content in the supplied presentation components. No execution owner. */
const LiveViews = (() => {
  const mapped = new Set(['overview', 'settings', 'advanced', 'history', 'portfolio', 'compare', 'alpha-compare', 'models', 'feature-research', 'goals', 'goal', 'goal-conversation', 'goal-results', 'cases', 'books', 'inputs', 'tasks', 'lab', 'evidence', 'evidence-stream', 'evidence-reading', 'report', 'handoff','factor','foundation','alpha','risk', 'advanced', 'kit', 'features', 'upgrade']);
  /* The two execution modes the service states, and its one verification policy, in words. */
  const MODES = {RESEARCH_PREPARATION: 'Research preparation: no strategy package is installed; inputs, Features and research are maintained without activation.', DEVELOPMENT_REPLAY: 'Development replay: an installed strategy package is replayed for development; nothing is activated or traded.'};
  const VERIFICATION = {DISCOVERY_ONLY_SELECTED_OPERATIONS_REVALIDATE: 'Discovery reads are not re-validated; the selected operations re-validate what they use before running.'};
  /* Governance & validation: the workspace's governing facts as the service states them, and
   * the bridge to the original product for what is not integrated here (freeze, finalization
   * and validation access). Read from the session envelope; nothing is fetched or changed. */
  /* The governance facts and the governed actions, as the Settings page's Advanced section (round 62). */
  function governanceBody() {
    const w = Data.workspaceFacts() || {}, rc = w.research_context || {}, strategies = w.installed_strategies || [];
    // round 85: the rows are form rows (a title, its line, the state at the right, one pill); no glyph
    const row = (ic, name, desc, state, action = '') => formRow({title: name, line: desc, detail: state, control: action});
    const held = (words) => html`${icon('lock')} ${words}`;
    const facts = html`
      ${row('advanced', t('Execution mode'), t(MODES[w.execution_mode] || 'The mode the service states for this workspace.'), html`<span data-tip="${w.execution_mode || ''}">${codeWords(w.execution_mode)}</span>`)}
      ${row('checkcircle', t('Verification policy'), t(VERIFICATION[rc.verification] || 'The verification policy the service states.'), html`<span data-tip="${rc.verification || ''}">${codeWords(rc.verification)}</span>`)}
      ${row('cube', t('Workspace manifest'), t('The verified manifest that binds this workspace\'s identity; every saved object names it.'), html`<span class="mono">${w.workspace_manifest_hash ? short(w.workspace_manifest_hash, SHORT.hash) : ''}</span>`, w.workspace_manifest_hash ? btn(t('Copy hash'), 'copy-text', w.workspace_manifest_hash, 'button compact') : '')}
      ${row('task', t('Service capacity'), t('Tasks the service runs and holds at once; anything beyond waits in order.'), w.capacity ? html`${w.capacity.active} ${t('active')} · ${w.capacity.queued} ${t('queued')}` : '')}
      ${row('archive', t('Installed strategies'), t('An installed strategy replays its research; a person activates a reviewed book to run it forward. Nothing is traded.'), strategies.length ? html`${strategies.map((v) => html`<span data-tip="${v.strategy_id}" tabindex="0">${codeWords(v.strategy_id)}</span> · ${(v.score_source_modes || []).map(codeWords).join(', ') || ''}<br>`)}` : held(t('none installed · research only')))}`;
    const governed = html`
      ${row('archive', t('Freeze candidate'), t('Records one exact installed development result as a candidate. No training, activation or change to current/default.'), held(t(strategies.length ? 'An operation of the Agent tool and the automation API' : 'Needs an installed strategy result')))}
      ${row('file', t('Finalization status'), t('Validation status for one frozen candidate; in a development workspace the ordinary answer is a wait, not a recommendation.'), held(t('No frozen candidate here')))}
      ${row('lock', t('Protected Validation'), t('Protected data access and an authorized identity are required; this workbench holds neither and infers no permit.'), held(t('Not admitted here')))}`;
    return {facts, governed};
  }

  /* A saved object's way in, inside the workbench (the original UI's entry was retired on
   * 2026-09-19): a Task opens in Task Center, an object History has discovered opens as itself,
   * anything else is looked for in History by its reference. */
  function savedObjectLink(label = t('Find it in History'), entry = '', cls = 'button compact') {
    if (entry.startsWith('task:')) return btn(html`${icon('task')}${label}${icon('arrow')}`, 'task', entry.slice(5), cls);
    const saved = entry && Data.history().find((r) => r.id === entry);
    if (saved) return btn(html`${icon(Inspect.recordIcon(saved))}${label}${icon('arrow')}`, 'history-open', entry, cls);
    const ref = entry.replace(/^(result|experiment|review):/, '');
    return link(html`${label}${icon('arrow')}`, 'history', cls, ref ? {q: short(ref, SHORT.hash)} : {});
  }
  function unwired() {
    return html`${objectHead(t(ROUTES[app.page][1]), t('Not part of this build'))}<section class="panel pad">${emptyState(t('This page is not part of this build.'), link(t('Home'), 'overview', 'text-btn'), '', 'elsewhere')}</section>`; // round 93 (rule 4): the honest state in the product's words
  }
  /* Reading labels from declared parameters only: method, target, size and interval. Never a
   * metric, ranking or winner. Machine identifiers stay untranslated; counts are localized.
   * `declared` is one row of the existing experiment listing (or the readback adapter below);
   * `entry` is the history entry, which also names the exact reference. */
  const shortRef = (v) => short(v, SHORT.id);
  const REFERENCE_WORDS = {experiment: 'Task', task: 'Task', review: 'Review', result: 'Result', update: 'Update'};
  function declaredFromReadback(b) {
    const d = b?.document || {};
    return {kind: b?.program?.kind, task_id: b?.task_id, sessions: d.experiment?.sessions, origin_task_id: b?.origin_task_id || null,
      target_recipe_id: d.alpha?.target_recipe_id, model_parameters: d.alpha?.model_parameters,
      model_adapter_id:b?.execution_preview?.model_adapter_id, component_recipe_id:d.alpha?.component_recipe_id,
      risk_capability_handle: d.risk?.estimator?.capability, risk_parameters: d.risk?.estimator?.parameters,
      factor_ids: d.factor?.factor_ids || d.alpha?.ordered_feature_ids || b?.execution_preview?.ordered_feature_ids || [], candidate_id: b?.portfolio_source?.candidate_id};
  }
  /* A Portfolio book's name (N3, law 134): the policy it declared -- names per sleeve, sleeves, the
   * weight rule in words -- the same in its list, on its page, in Evidence and in every picker; a
   * listing without the policy names the book by its Alpha candidate. */
  const weightWords = (rule) => methodWords(rule); // U41: the rules a study may declare, in the method table
  /* A catalog policy in words (U42, U45): its family, then its fields as declared; where a study declares one,
   * it runs in place of the tranche book, whose fields its document leaves out. */
  const POLICY_FIELDS = {risk_aversion: 'risk aversion', turnover_regularization: 'turnover regularization', sector_deviation_penalty: 'sector deviation penalty'};
  const policyWords = (cp) => [METHOD_WORDS[cp.family] ? methodWords(cp.family) : codeWords(cp.family), cp.top_k != null ? t('{k} names', {k: cp.top_k}) : '', cp.maximum_weight != null ? t('at most {w} a name', {w: pctFraction(Number(cp.maximum_weight))}) : '', ...Object.keys(POLICY_FIELDS).filter((k) => cp[k] != null).map((k) => `${t(POLICY_FIELDS[k])} ${cp[k]}`)].filter(Boolean).join(' · ');
  const bookWords = (p) => (!p ? '' : p.catalog_policy ? policyWords(p.catalog_policy) : t('{k} names per sleeve · {n} sleeves · {w}', {k: p.top_k ?? '', n: p.tranches ?? '', w: weightWords(p.weight_rule)}));
  function studyFacts(declared, entry = null) {
    const kind = declared?.kind || entry?.kind || '';
    const parts = [], brief = [], words = []; // brief: the short header form without long machine tokens; words: the same facts said for reading
    const named = (ids, n) => ids.slice(0, 3).join(', ') + (ids.length > 3 ? ' +' + (ids.length - 3) : '');
    let wordsMarkup = null; // round 90: the same words with an identifier in the mono (a row's title); `words` stays plain for search and sort
    if (kind === 'alpha.model-development') {
      const model=declared.model_parameters?.family || declared.model_adapter_id;
      parts.push(declared.component_recipe_id,model,declared.target_recipe_id);
      words.push(declared.component_recipe_id ? codeWords(declared.component_recipe_id) : '',methodWords(model),methodWords(declared.target_recipe_id));
      brief.push(declared.component_recipe_id || model);
      if (declared.factor_ids?.length) { const n = countText(declared.factor_ids.length, '{n} feature', '{n} features'); parts.push(n); brief.push(n); words.push(n); }
    } else if (kind === 'factor.screening-development') {
      if (declared.factor_ids?.length) { const n = countText(declared.factor_ids.length, '{n} factor', '{n} factors'); parts.push(n); brief.push(n); words.push(n + ' · ' + named(declared.factor_ids)); wordsMarkup = html`${n} · <span class="mono">${named(declared.factor_ids)}</span>`; }
    } else if (kind === 'risk.covariance-development') { parts.push(declared.risk_capability_handle); brief.push(declared.risk_capability_handle); words.push(methodWords(declared.risk_capability_handle)); }
    else if (kind === 'portfolio.policy-development') { parts.push(declared.candidate_id ? t('candidate') + ' ' + declared.candidate_id : ''); words.push(declared.portfolio_policy ? bookWords(declared.portfolio_policy) : declared.candidate_id ? t('from Alpha candidate {id}', {id: short(String(declared.candidate_id).replace(/^alpha-candidate-/, ''), SHORT.id)}) : ''); }
    else if (kind === 'INSTALLED_RESULT' || (kind === 'CRO_REVIEW' && entry?.strategy_package_id)) {
      const packageId = entry?.strategy_package_id;
      // A missing package label is a naming absence, not the book's name. The
      // exact owner code remains metadata and the saved object's coded fact.
      parts.push(packageId);
      words.push(packageId && (labelOf(packageId) || declaredCodeWord(packageId)) ? codeWords(packageId) : t('Installed strategy result'));
    }
    else if (kind === 'TASK_RECORD') {
      const taskKind = entry?.task_kind ? codeWords(entry.task_kind) : '';
      parts.push(taskKind); words.push(taskKind);
    }
    else { parts.push(entry?.strategy_package_id); words.push(entry?.strategy_package_id ? codeWords(entry.strategy_package_id) : ''); }
    const [type, value] = String(entry?.entry_id || (declared?.task_id ? 'task:' + declared.task_id : '')).split(':');
    const sessions = declared?.sessions;
    return {
      summary: parts.filter(Boolean).join(' · '),
      words: words.filter(Boolean).join(' · '),
      wordsMarkup,
      brief: brief.filter(Boolean).join(' · '),
      reference: value ? t(REFERENCE_WORDS[type] || 'Task') + ' ' + shortRef(value) : '',
      ref: value ? shortRef(value) : '', // the reference's own part: a lobby's reference column (law 136)
      interval: sessions?.start && sessions?.end ? `${sessions.start} — ${sessions.end}` : '',
      origin: declared?.origin_task_id || null,
    };
  }
  /* A history date is one of three different things; each is named, none stands in for another.
   * The input's recorded cutoff, whether its source files are present locally and whether a study
   * verified it are three separate facts; only the first two are known here. */
  const inputState = (bindingHash) => {
    const version = Data.inputVersion(bindingHash);
    return {inputCutoff: version?.cutoff || null, inputAvailable: version ? version.available === true : null};
  };
  function cutoffText(row) {
    const parts = [row.inputCutoff ? t('cutoff') + ' ' + row.inputCutoff : row.inputAvailable === null ? t('cutoff unknown') : t('cutoff not recorded')];
    if (row.inputAvailable === false) parts.push(t('source files unavailable'));
    return parts.join(' · ');
  }
  function existing(id) {
    const row = Data.history().find((x) => x.id === id);
    if (!row) return;
    // N6 (the plan: an object opens in the pane or as its page, never a dialog): its facts in words,
    // an absent fact no row, the exact reference behind its opener, its ways under them
    const words = (x) => /^[A-Z][A-Z0-9_]+$/.test(String(x || '')) ? codeWords(x) : x;
    const savedName = (current) => current.raw.kind === 'INSTALLED_RESULT' ? nameOf(current).name : current.summary ? `${t(current.name)} · ${words(current.summary)}` : t(current.name);
    const name = savedName(row);
    const rows = [[t('Kind'), t(row.kind)], [t('State'), codeWords(row.raw.status)], ...(row.raw.kind === 'INSTALLED_RESULT' && row.raw.strategy_package_id ? [[t('Strategy package'), coded(row.raw.strategy_package_id)]] : []), [t('Research input'), row.input ? html`${row.input} · ${cutoffText(row)}` : ''], [t('Recorded'), row.raw.recorded_at ? when(row.raw.recorded_at) : ''], [t('Holdings session'), row.holdingsSession || ''], [t('Declared interval'), row.interval || '']].filter(([, v]) => v);
    const note = row.raw.kind === 'INSTALLED_RESULT' ? t('An installed result is the replay of an installed strategy package through the Portfolio application, not an authored study: it has no declaration to continue and no candidate source. The original product reads it; the workbench keeps its reference.') : t('Metadata discovery is not artifact verification; the exact object is read through the Agent tool\'s readback.');
    const ways = html`<div class="flow">${row.raw.status === 'SUCCEEDED' && row.raw.kind.endsWith('-development') ? btn(t('Continue as a new draft'), 'research-continue', row.task_id, 'button primary') : ''}${btn(t('Copy reference'), 'history-copy', id, 'button compact')}</div>`;
    Window.openInspector({mode: 'object', readHeader: () => { const current = Data.history().find(x => x.id === id) || row; return {title: savedName(current), kind: t('Saved object')}; }, title: name, kind: t('Saved object'), by: ['history-open', id], body: html`<section class="inspector-section">${kv(rows, 'side-props')}<p class="caption">${note}</p>${ways}${codeRef(t('Exact reference'), id, 'text')}</section>`});
  }
  /* Exact owner Committee context and retained collaboration references; never a recency guess. */
  function collaborationRows(...refs) {
    if (typeof LiveTeam === 'undefined' || !LiveTeam.sessionsNaming) return [];
    const named = LiveTeam.sessionsNaming(...refs);
    if (!named.length) return [{icon: 'team', label: t('Team session'), value: t('none names this object')}];
    return named.map((s) => ({icon: 'team', label: t(s.committee ? 'Committee' : 'Team session'), value: html`<span><span class="mono">${LiveTeam.sessionLabel(s.id)}</span>${(s.roles || []).length ? html` · ${s.roles.map(LiveTeam.roleName).join(', ')}` : ''}${s.committee ? html` · ${codeWords(s.committee.verdict?.outcome || s.committee.stage)}` : html` · ${countText(s.entries, '{n} exchange', '{n} exchanges')}`}</span>`, ...(s.committee ? {to: {page: 'team-committee', extra: {team: s.id, committee: s.committee.update_task_id}}} : {action: {name: 'team-open', value: s.id}}), note: s.question ? html`<span class="owner-text line-cut" data-tip="@overflow">${s.question}</span>` : ''}));
  }
  /* The workspace chooser of the template: the workspace this service opened, and the two ways
   * to another one -- an empty folder the product is started on (its first page prepares it),
   * or an existing one, opened the same way. This service holds one workspace; nothing here
   * copies, migrates or switches one behind the reader's back. Facts come from the session. */
  const LAUNCH = 'python scripts/run_alphalattice.py --workspace <folder> serve --no-browser --stop-on-stdin';
  /* The workspace popover (round 81, Codex's Environment and account menu in one): the head,
   * the facts as rows -- short values only: the manifest by its hash; the installed strategies'
   * names are Settings' (the user, 2026-09-23: a name here overflowed the menu) -- the ways (create / open keep their dialogs: they take input), Settings.
   * One service holds one workspace; nothing here switches one behind the person's back. */
  function workspacePopover() {
    const w = Data.workspaceFacts() || {}, clocks = Data.clocks(), prep = Data.preparation(), unprepared = Boolean(prep) && !prep.inputs?.length;
    const fact = (word, value) => html`<div class="menu-row menu-fact" role="presentation"><span class="menu-word">${word}</span><span class="menu-note">${value}</span></div>`;
    return html`<div class="menu-head"><span class="menu-tile">${icon('cube')}</span><div><strong>${Data.workspace()}</strong><small>${t(unprepared ? 'local · not prepared yet' : 'local')}</small></div></div>${fact(t('Data through'), clocks.data)}${fact(t('Features through'), clocks.feature)}${fact(t('Input versions'), count(Data.inputs().length))}${fact(t('Execution mode'), codeWords(w.execution_mode))}${fact(t('Manifest'), mono(w.workspace_manifest_hash, SHORT.hash))}${fact(t('Tasks needing a decision'), count(openTaskCount()))}<hr class="menu-divider">${menuRow({ic: 'plus', action: 'workspace-how', value: 'create', word: t('Create a workspace'), note: '›'})}${menuRow({ic: 'archive', action: 'workspace-how', value: 'open', word: t('Open an existing workspace'), note: '›'})}${menuRow({ic: 'copy', action: 'copy-text', value: Data.workspace(), word: t('Copy the workspace id')})}<hr class="menu-divider">${menuRow({ic: 'refresh', action: 'go', value: 'upgrade', word: t('What the upgrade changed')})}${menuRow({ic: 'sliders', action: 'go', value: 'settings', word: t('Settings'), note: 'Ctrl ,'})}`;
  }
  function workspaceDialog(view = 'create') {
    if (view === 'choose') return Window.toggleWorkspace();
    const prep = Data.preparation();
    const unprepared = Boolean(prep) && !prep.inputs?.length;
    // round 83: one box of rows — the command (with its Copy), what happens, what to keep open
    const how = (kind) => formGroup(t('Start the service'), html`${formRow({title: t('Command'), line: LAUNCH, control: btn(t('Copy command'), 'copy-text', LAUNCH, 'button compact'), cls: 'form-row-mono'})}${formRow({title: t('What happens'), line: t(kind === 'create' ? 'Choose an empty folder and start the product on it. The service writes the workspace manifest there, and its first page offers the preparation: sources are captured, a Panel is built and one input version is published, each step after your confirmation.' : 'Start the product on the folder that holds the workspace. The service verifies its manifest and opens it; nothing is migrated or rewritten.')})}${formRow({title: t('Then'), line: t('Run it from the checkout root. The printed loopback URL opens that workspace, and /workbench.html on it opens this workbench. Keep the terminal open; enter stop to end the service.')})}${unprepared && kind === 'create' ? formRow({title: t('This workspace itself is not prepared yet; its preparation is one page away.'), control: link(t('Prepare this workspace'), 'overview', 'button compact')}) : ''}`);
    openDialog(t('Workspace · local'), t(view === 'create' ? 'Create a workspace' : 'Open an existing workspace'), how(view), '', false, () => workspaceDialog(view));
  }
  /* The object language (round 63): the one name of a thing wherever it appears -- its kind and
   * its subject -- and its truth (the owners' disclaimer) beside the name, never inside it. A
   * Task is named by the saved object it produced when the history has it, else by its kind; its
   * goal sentence is its truth. The evidence types read by the round-76 table (`EVIDENCE`). */
  function nameOf(x) {
    if (!x) return {kind: '', subject: '', name: '', truth: ''};
    if (x.type && EVIDENCE[x.type]) return {...evidenceName(x.type, x), truth: ''};
    if (x.task_kind !== undefined || (x.goal_summary !== undefined && x.task_id)) {
      const row = Data.history().find((r) => r.task_id === x.task_id);
      if (row) return {...nameOf(row), truth: x.goal_summary || ''};
      const kind = x.task_kind ? codeWords(x.task_kind) : t('Task');
      return {kind, subject: '', name: kind, truth: x.goal_summary || ''};
    }
    const kind = t(x.name || x.kind || ''), subject = x.words || x.summary || '';
    const typed = x.raw?.kind === 'INSTALLED_RESULT' && subject === kind;
    return {kind, subject, name: subject && !typed ? `${kind} · ${subject}` : kind, markup: subject && x.wordsMarkup ? html`${kind} · ${x.wordsMarkup}` : null, truth: ''}; // `markup`: the same name with an identifier in the mono (round 90), for a row's title
  }
  const truthOf = (x) => nameOf(x).truth;
  const taskSuccessor = (x) => {
    const fact = Data.taskSuccessor(x);
    return fact ? btn(html`${t('Superseded by')} ${t('Task')} <span class="mono">${shortRef(fact.successor_task_id)}</span>`, 'task', fact.successor_task_id, 'text-btn') : '';
  };
  const taskStopFacts = (x) => {
    const fact = Data.taskAttention(x);
    const task = x.object || Data.taskOf(x.task_id || x.raw?.task_id);
    if (fact?.resolution !== 'UNRECOVERABLE' && !x.raw?.failure_code && !task?.latest_failure_code) return '';
    const reason = x.raw?.detail || task?.detail, next = x.raw?.stop_next || task?.stop_next;
    return reason || next ? factsRef(t('Why it stopped'), html`${reason ? html`<p>${t(reason)}</p>` : ''}${next ? html`<p>${t(next)}</p>` : ''}`) : '';
  };
  const taskAttentionFacts = (x) => {
    const successor = taskSuccessor(x), stop = taskStopFacts(x);
    return successor || stop ? html`${successor} ${stop}` : '';
  };
  function recordRow(x, extra = {}) {
    if (x.earlierStops?.length) return earlierStopRows(x, row => recordRow({...row, earlierStops: undefined}, extra));
    const {name, markup} = nameOf(x);
    const on = (key) => extra.show ? extra.show[key] !== false : key !== 'reference'; // round 57: the viewer's display properties; the id is a property, off unless chosen (round 63)
    // a lobby's row carries its reference in its own column (`extra.ref`, law 136); what it continues is a fact
    const reference = extra.ref ? (on('origin') && x.origin ? html`${t('continued from')} <span class="mono">${shortRef(x.origin)}</span>` : '') : on('reference') ? html`<span class="mono">${x.reference}</span>${x.origin ? html` · ${t('continued from')} <span class="mono">${shortRef(x.origin)}</span>` : ''}` : '';
    const props = [reference, on('interval') ? x.interval || '' : '', on('input') && x.input ? ['', html`${x.input} · ${cutoffText(x)}`, 'drop'] : '', on('holdings') && x.holdingsSession ? html`${t('holdings')} ${x.holdingsSession}` : '', ...(extra.props || [])];
    const a = stateOf(x.status), status = a.tone === 'neutral' ? '' : stateLine(x.status, {dot: true, next: '', live: Boolean(a.moving)}); // round 85: the kind's mark leads; a state that is not the plain record is the first fact
    return objectRow({lead: kindTile(x.raw?.kind, name), name: markup || name, ref: extra.ref || '', to: {action: 'history-open', value: x.id}, cls: 'object-row'}, {key: x.id, props: [status, ...props, taskAttentionFacts(x)], columns: ['state', 'reference', 'interval', 'input', 'holdings', ...(extra.columns || []), 'recovery'], time: on('recorded') && (x.raw?.recorded_at || x.recordedAt) ? when(x.raw?.recorded_at || x.recordedAt) : '', actions: extra.actions || ''});
  }
  /* One row for every saved study (N6; F2, law 136): the reference, the label, the interval, the
   * input, the day -- the lists of Factor, Foundation, Alpha, Risk and Portfolio studies are one
   * list shape. A list of studies holds finished work: a finished study carries no state mark; one
   * that is not finished says its state. */
  function studyRow({state = 'SUCCEEDED', name, ref = '', interval = '', input = '', at = '', to, key, props = null}) {
    const on = (k) => !props || props[k] !== false, done = stateOf(state).key === 'succeeded';
    return objectRow({...(done ? {} : {state}), name, ref, to, cls: 'object-row study-row'}, {key, props: [...(done ? [''] : []), on('interval') && String(interval).trim() ? interval : '', on('input') && String(input).trim() ? ['', input, 'drop'] : ''], columns: [...(done ? ['state'] : []), 'interval', 'input'], time: at ? when(at) : ''});
  }
  /* A kind's studies as a lobby (F2, law 136): grouped by time -- the reader's day, this week,
   * earlier this month open, each month before folded; a study still moving is its own group on
   * top -- or by input version (Display; the newest open). A study's day is its History record's:
   * where History's older pages are not read, the day is not either, and the foot reads them.
   * `list`: [{key, state, name, words, ref, interval, input, inputKey, inputText, at, to}]. */
  function studyLobby(name, list, {axes = ['time', 'input'], placeholder = ''} = {}) {
    const cut = (h) => Data.inputVersion(h)?.cutoff || '';
    const versions = [...new Set(list.map((x) => x.inputKey).filter(Boolean))].sort((a, b) => String(cut(b)).localeCompare(String(cut(a))));
    const unread = Data.hasMoreHistory() && list.some((x) => !x.at);
    const AXES = {
      time: {key: 'time', label: t('Time'), group: (x) => stateMoving(x.state) ? {key: 'moving', label: t('In progress'), rank: -1, open: true} : x.at || !unread ? timeGroup(x.at) : {key: 'unread', label: t('Day not read yet'), rank: 9e9, open: false}},
      input: {key: 'input', label: t('Input version'), group: (x) => x.inputKey ? {key: x.inputKey, label: x.input, rank: versions.indexOf(x.inputKey), open: versions.indexOf(x.inputKey) === 0} : {key: 'none', label: t('No input version'), rank: 9e9, open: false}},
    };
    return Lobby.render(name, {items: list, row: (x, d) => studyRow({...x, props: d.props}), axes: axes.map((k) => AXES[k]),
      words: (x) => [x.words, x.ref, x.key, x.inputText, x.interval].join(' '), placeholder: placeholder || t('Name, factor or input'),
      properties: [['interval', t('Interval')], ['input', t('Input version')]],
      foot: unread ? Lobby.older(t('Older records are not read yet'), 'history-more', '', Data.historyLoading) : ''});
  }
  /* A Portfolio page without a book: the workspace is open; what is missing is the study. The
   * saved Portfolio studies are offered here to open, and where there is none, the way to one. */
  function bookless() {
    const entries = new Set(Data.portfolioEntries().map((x) => x.entry_id));
    const books = Data.history().filter((x) => entries.has(x.id));
    const list = books.map((x) => ({key: x.id, state: x.raw?.status, name: x.words || x.summary || t(x.name), words: x.words || x.summary || '', ref: x.ref, interval: x.interval, input: x.input ? html`${x.input} · ${cutoffText(x)}` : '', inputKey: x.raw?.input_binding_hash || '', inputText: x.input, at: x.raw?.recorded_at, to: {action: 'history-open', value: x.id}})); // N6: the kind is the page's, never the row's prefix
    // Compare's list is the same list with Compare's sentence: the row opens the first study here
    // and the second is chosen on the page (2026-09-21, the Studies audit: the row led to Portfolio)
    const lede = `${app.page === 'compare' ? t('No studies are open to compare. Choose the first below; the second is chosen on the page.') : t('No Portfolio study is open. Choose a saved study below; opening one reads it and runs nothing.')} ${t('A Portfolio study in the Lab publishes one; History lists every saved object.')}`;
    // the list's one verb (law 129; the user, 2026-09-24): a new Portfolio experiment -- Compare's list chooses, it composes nothing
    const verb = app.page === 'portfolio' && books.length ? link(html`${icon('plus')}${t('New experiment')}`, 'lab', 'button primary', {experiment_kind: 'portfolio.policy-development'}) : '';
    return html`${objectHead(t(ROUTES[app.page][1]), lede, verb)}${books.length
      ? studyLobby('portfolio', list)
      : emptyState(t('No experiment yet'), link(t('New experiment'), 'lab', 'button primary', {experiment_kind: 'portfolio.policy-development'}), 'page-empty')}`; // N3: the path names the list; the dock and ⌘K reach History
  }
  /* ---- Home (round 65): the workspace's name and state as the head, the data facts as chips,
   * and three groups of object rows for whoever operates — what needs a decision, what is
   * running, what was recently recorded. Nothing here runs, plans or selects; the chart of a
   * book is the book's page, the index of every object is History. ---- */
  /* Where and how a result was computed (R9): the Host's record of the settings and the machine its Task
   * started with, and an Alpha study's fits -- a record beside the result, never an identity (the same numbers
   * come from any setting); one line, what was not recorded said so. */
  function realizationWords(r) {
    if (!r) return '';
    if (!r.recorded) return t('Where and how it was computed was not recorded when it ran.');
    const where = [r.platform?.system, r.platform?.release].filter(Boolean).join(' ');
    return html`${[r.started_at ? when(r.started_at) : '', where, r.cores ? t('{cores} of {processors} processors', {cores: count(r.cores), processors: count(r.processors)}) : '', r.lightgbm_threads ? countText(r.lightgbm_threads, '{n} LightGBM thread', '{n} LightGBM threads') : ''].filter(Boolean).join(' · ')}${infoMark(t('A record, not an identity: the same numbers come from any setting.'))}`;
  }
  const KIND_OPENERS = {'portfolio.policy-development': 'Open the book', 'alpha.model-development': 'Open the study', 'factor.screening-development': 'Open the study', 'risk.covariance-development': 'Read the diagnostic', CRO_REVIEW: 'Read the review'};
  const latestCutoff = () => Data.inputs().map((v) => v.cutoff).filter(Boolean).sort().at(-1) || null;
  const actionableTasks = () => Data.actionableTasks();
  /* The head's state and its one primary, as facts: the preparation, a held Task, the latest
   * saved object, or the first study. */
  function resumeFacts(prep, unprepared) {
    if (unprepared) {
      const run = prep.task_id ? Data.tasks().find((v) => v.task_id === prep.task_id) : null;
      const state = run ? stateLine(run, {word: stateMoving(run.lifecycle) ? t('Preparing the workspace') : undefined, until: prep.progress?.retry_after_at || null}) : stateLine('draft', {word: t('Not prepared')});
      return {state, primary: '', secondary: ''}; // N6: the Needs-a-decision row is the way to prepare (one way)
    }
    const tasks = actionableTasks();
    const heldTask = tasks.find((v) => stateHeld(v.status)) || Data.tasks().find((v) => stateMoving(v.lifecycle));
    // the one primary starts work; a held Task and the latest object are rows below
    const primary = link(html`${t('New experiment')}${icon('arrow')}`, 'lab', 'button primary');
    if (heldTask) return {state: stateLine(heldTask, {duration: ''}), primary, secondary: ''};
    if (!Data.history()[0]) return {state: stateLine('ready', {word: t('inputs published')}), primary, secondary: link(t('Input versions'), 'inputs', 'button')};
    return {state: stateLine('ready', {word: t('Research-ready')}), primary, secondary: ''};
  }
  /* The data facts as chips: the working store's clock, the published input, the universe. */
  function homeChips() {
    const clocks = Data.clocks(), w = Data.workspaceFacts() || {}, du = w.research_context?.data_update || {};
    const cohort = du.membership?.bootstrap?.cohort_size, cutoff = latestCutoff(), versions = Data.inputs().length;
    return [[t('Data through'), clocks.data || '', t('Working store · Features through {d}', {d: clocks.feature})], [t('Input'), cutoff || '', versions ? countText(versions, '{n} published version · immutable', '{n} published versions · immutable') : t('none published yet')], cohort ? [t('Listings'), count(cohort), t('the research universe')] : null];
  }
  /* What waits on a person (U5): the Host's one answer (`/api/decisions`), in its order, each
   * decision a row whose way on is its object's. The page adds only what the reader's own
   * address holds (their PLAN previewed since the read) and the Team's own record (an objection
   * awaiting the Main PM, U25). A data issue's case is a token, so the cases are one row with
   * their count; a kind the page has no words for is named by its code, never dropped (ST7).
   * Each row: {lead, place, name, why, to, by, at, key, state}. */
  const DATA_TASKS = new Set(['workspace_data_update', 'workspace_preparation']); // the Tasks the Data place holds
  const STUDY_DECISIONS = {
    CURATION: ['Curate its factors', 'An Alpha study builds on a curation decision'],
    PROMOTION: ['Promote to the whole universe', 'It ran on a sample of names; a promotion runs the same declaration on every name'],
  };
  /* A published review's way: its History entry where the history read holds it, else `otherwise`. */
  const reviewWay = (hash, otherwise = {page: 'upgrade'}) => { const r = Data.history().find((x) => x.raw?.review_publication_hash === hash); return r ? {action: 'history-open', value: r.id} : otherwise; };
  function decisionRow(d, list) {
    switch (d.kind) {
      case 'FIRST_USE': // U70: it runs for the person, in Running -- not a decision that waits on them
        return null;
      case 'WORKSPACE_PREPARATION': // no preparation is offered while the owner cannot read the workspace: the first-use notice says why
        return LiveWorkspace.prepareRefused() ? null : {lead: 'cube', place: 'data', name: t('Prepare workspace'), why: t('No verified research input exists in this workspace; one explicit preparation makes it research-ready.'), to: {action: 'workspace-preview', value: 'prepare'}};
      case 'INPUT_VERSION':
        return {lead: 'cube', place: 'data', name: t('Publish an input version'), why: html`${t('New data is not yet a research input')} · ${t('Data through')} <span class="numeric">${d.data_through}</span> · ${t('research input through')} <span class="numeric">${d.input_through}</span>`, to: {page: 'inputs'}, at: d.data_through};
      case 'TASK_RECORD_UNREADABLE':
        return {lead: 'task', name: html`${t('Task')} ${hashCell(d.task_id)}`, refusal: d, key: d.task_id};
      case 'STOPPED_TASK': { // N6 (law 58): the stop said once, with its way on -- a data update's in the Data page's words
        // Only the current owner's unresolved fact is a decision; retained lifecycles stay historical.
        if (Data.taskAttention(d)?.unresolved !== true) return null;
        const v = Data.tasks().find((x) => x.task_id === d.task_id) || {task_id: d.task_id, task_kind: d.task_kind, kind: d.task_kind, lifecycle: d.lifecycle, status: d.lifecycle};
        const stop = LiveWorkspace.stopWords ? LiveWorkspace.stopWords(v) : null, next = v.stop_next ? t(v.stop_next) : stop ? stop.next : wayOn(v.latest_failure_code, v.status);
        return {state: v, place: DATA_TASKS.has(v.task_kind) ? 'data' : '', name: nameOf(v).name, why: v.detail ? t(v.detail) : stop ? stop.title : t(stateOf(v.status).line), next, to: {action: 'task', value: v.task_id}, key: v.task_id, at: v.last_activity_at || ''};
      }
      case 'DATA_ISSUE': { // the first case stands for them all
        if (list.find((x) => x.kind === 'DATA_ISSUE') !== d) return null;
        const n = list.filter((x) => x.kind === 'DATA_ISSUE').length;
        return {lead: 'data', place: 'data', name: countText(n, '{n} data issue waits for a choice', '{n} data issues wait for a choice'), why: t('Only a person may confirm its options; preview one to see what it would change'), to: {page: 'issues'}};
      }
      case 'UPGRADE':
        return {lead: 'refresh', name: t('Read what the upgrade changed'), why: t('The installed code changed since this workspace last acknowledged it'), to: {page: 'upgrade'}}; // the row's name says what the overview does (LS2, WD4: 2026-09-30, Home at 182 words with it)
      case 'PLAN_PREVIEW': {
        const mine = app.plan?.plan_hash === d.plan_hash;
        return {lead: 'lab', name: t(mine ? 'Your PLAN waits to be confirmed' : 'A PLAN waits to be confirmed'), why: html`${t(Data.kindName(d.study_kind))} · ${t('reading it grants nothing; confirming runs it')}`, to: mine ? {page: 'lab'} : {page: 'lab', extra: {plan: d.plan_hash}}, by: mine ? t('You') : actorWords(d.previewed_by), key: d.plan_hash, at: d.previewed_at || ''};
      }
      case 'CURATION': case 'PROMOTION': {
        const [name, why] = STUDY_DECISIONS[d.kind];
        return {lead: 'lab', name: t(name), why: html`${nameOf({task_id: d.task_id, task_kind: 'research_experiment'}).name} · ${t(why)}`, to: {action: 'task-result', value: d.task_id}, key: d.kind + ':' + d.task_id};
      }
      case 'REVIEW_CHANGED':
        return {lead: 'review', name: t('A review sealed under an earlier binding'), why: t('It reads back as recorded; its book can be reviewed again'), to: reviewWay(d.review_publication_hash), key: 'changed:' + d.review_publication_hash};
      case 'CRO_RECOMMENDATION': {
        const pa = d.person_action || {}, n = (pa.actions || []).length;
        return {lead: 'review', name: t('The CRO asks a person to act'), why: html`${pa.route ? codeWords(pa.route) : ''}${pa.route && n ? ' · ' : ''}${n ? countText(n, '{n} action', '{n} actions') : ''}`, to: reviewWay(d.review_publication_hash), key: 'cro:' + d.review_publication_hash};
      }
      default:
        return {lead: 'info', name: codeWords(d.kind), key: 'decision:' + d.kind};
    }
  }
  function waiting() {
    const all = Data.decisions() || [], list = all.filter(d => d.waits_on !== 'AGENT'), out = list.map((d) => decisionRow(d, list)).filter(Boolean);
    if (!Data.decisions() && Data.decisionsError) out.push({lead: 'warning', name: t('What waits on a person was not read'), why: coded(Data.decisionsError.split(':')[0]), to: {action: 'workspace-refresh'}});
    // the reader's own PLAN, previewed since the Host's answer was read
    if (app.plan?.plan_hash && !all.some((d) => d.kind === 'PLAN_PREVIEW' && d.plan_hash === app.plan.plan_hash)) out.push({lead: 'lab', name: t('Your PLAN waits to be confirmed'), why: t('Previewed and not run; the confirmation admits a Task.'), to: {page: 'lab'}, by: t('You'), key: app.plan.plan_hash});
    const team = typeof LiveTeam !== 'undefined' && LiveTeam.summary ? LiveTeam.summary() : null;
    if (team?.unresolved) out.push({lead: 'review', name: countText(team.unresolved, '{n} objection awaits the Main PM', '{n} objections await the Main PM'), why: html`<span class="coded" data-tip="${team.session}">${t('Retained session')}</span> · ${t('Team')}`, to: {page: 'team', extra: team.selected ? {} : {team: team.session, actor: '', event: ''}}, key: team.session, at: team.latest?.at || ''});
    return out;
  }
  /* The rows are one line (U9, LS2): the title whole, its words cut with an ellipsis and read whole
   * on hover; the facts ride the line. */
  // Identical owner sentences are said once over their stopped Tasks (ST6); each row's name
  // keeps the cause on hover. A cause or way only one row has stays on that row (WD4).
  const sharedWay = (all) => { const n = new Map(); for (const w of all) if (typeof w.next === 'string' && w.next) n.set(w.next, (n.get(w.next) || 0) + 1); let best = ''; for (const [k, c] of n) if (c > 1 && c > (n.get(best) || 0)) best = k; return best; };
  const sharedCause = (all) => { const n = new Map(); for (const w of all) if (w.state?.detail && w.state.task_id) { const ids = n.get(w.state.detail) || new Set(); ids.add(w.state.task_id); n.set(w.state.detail, ids); } let best = ''; for (const [k, ids] of n) if (ids.size > 1 && ids.size > (n.get(best)?.size || 0)) best = k; return {words: best, count: n.get(best)?.size || 0}; };
  function decisionRows() {
    const all = waiting(), shared = sharedWay(all), cause = sharedCause(all);
    const items = all.map((w) => {
      // ST6: structured recovery keeps the refusal's wrapping action area, never a clipped sentence.
      if (w.refusal) return refusal(w.refusal, 'warning', {catalog: true, more: html`<p>${w.name}</p>`, next: prerequisiteWays(w.refusal.next_requests), attrs: html`data-key="${w.key}"`});
      const common = w.state?.detail && w.state.detail === cause.words, why = common ? '' : w.why;
      return objectRow({lead: w.state ? w.lead : tile(w.lead, 'warning'), state: w.state, name: common ? html`<span class="hint" data-tip="${w.why}" tabindex="0">${w.name}</span>` : w.name, why, to: w.to}, {key: w.key || '', columns: [...(w.state ? [] : ['state']), 'by', 'next'], props: [...(w.state ? [] : ['']), w.by ? html`${t('by')} ${w.by}` : '', w.next && String(w.next) !== shared ? factsRef(t('Next step'), html`<p>${w.next}</p>`) : ''], time: w.at ? when(w.at) : ''});
    });
    return items;
  }
  /* What is running: the moving Tasks with their stage, and the newest retained team session. */
  function runningRows(tasksOnly=false) {
    const prep = Data.preparation();
    const rows = Data.runsOf('task').filter((r) => stateMoving(r.state)).slice(0,LOBBY.shown).map((r) => {
      const subject=LiveTasks.subjectContext?.(r.id);
      const name=I18N.zh && subject?.name_zh || (subject?.name ? t(subject.name) : bookWords(subject?.policy));
      return runRow(name ? {...r,name,markup:null} : r, {pinned:true,until:r.id===prep?.task_id ? prep.progress?.retry_after_at || null : null,word:r.id===prep?.task_id ? t('Preparing the workspace') : undefined,columns:['id','kind','date','verified'],props:['',codeWords(r.kind),subject?.date || '',html`${count(r.verified[0])} / ${count(r.verified[1])} ${t('verified')}`]});
    });
    if(tasksOnly)return rows;
    // U70 (V452, OP19): the person's first use, run by their agent -- the Host's FIRST_USE item while its delegation holds: the
    // steps it took as theirs, the hours left, its goal the way (where its one Stop is)
    for (const d of (Data.decisions() || []).filter((x) => x.kind === 'FIRST_USE')) {
      const steps = d.delegated_steps || [], hours = LiveGoals.hoursLeft({ends_at: d.ends_at, active: d.active !== false});
      rows.push(objectRow({lead: tile('flag', 'accent'), name: t('Your first use, run by your agent'), to: {action: 'goal-open', value: d.goal_hash}}, {key: 'first-use:' + d.goal_id, columns: ['steps', 'hours'], props: [steps.length ? steps.map(LiveGoals.stepWords).join(' · ') : t('No step taken for you yet.'), hours], time: steps.length ? when(steps.at(-1).recorded_at) : ''})); // one line (LS2): its facts in their slots, the sentence on its goal
    }
    const s = typeof LiveTeam !== 'undefined' && LiveTeam.summary ? LiveTeam.summary() : null;
    if (s?.session && s.latest) rows.push(objectRow({lead: tile('team', 'accent'), name: s.question?.text || t('Team'), to: {page: 'team', extra: s.selected ? {} : {team: s.session, actor: '', event: ''}}}, {key: s.session, columns: ['participants', 'latest', 'unread'], props: [countText(s.participants, '{n} participant', '{n} participants'), html`${s.latest.actor} · ${s.latest.kind}`, s.unread ? html`<span class="team-unread">${t('{n} unread', {n: s.unread})}</span>` : ''], time: s.latest.at ? when(s.latest.at) : ''}));
    return rows;
  }
  /* What needs you, as Home counts it (its "Needs a decision" group, the rows decisionRows draws): the
   * sidebar's attention badges and the window's title read this one count, never a list of their own
   * (the user, 2026-09-25: what waits on you apart from what merely runs); Data's share of it on Data. */
  const needs = () => { try { return Data.workspaceStatus === 'ready' ? waiting().length : 0; } catch { return 0; } };
  const dataNeeds = () => { try { return Data.workspaceStatus === 'ready' ? waiting().filter((w) => w.place === 'data').length : 0; } catch { return 0; } };
  const group = (title, rows, cls = 'card-list lines slotted', total = rows.length, more = '', note = '', shown = rows.length) => {
    const rest = rows.slice(shown);
    const reveal = rest.length ? html`<details class="reveal-details home-more"><summary>${countText(rest.length, 'Show {n} more', 'Show {n} more')}</summary><div class="${cls}">${rest}</div></details>` : '';
    return rows.length ? html`${groupHead(title, total)}${note}<div class="${cls}">${rows.slice(0, shown)}</div>${more}${reveal}` : ''; // N6 (law 81): an empty group is not drawn
  };
  const runningGroup=(tasksOnly=false)=>{
    const total=Data.tasks().filter(v=>stateMoving(v.lifecycle)).length, rows=runningRows(tasksOnly);
    return group(t('Running'),rows,'card-list lines slotted',total+rows.length-Math.min(total,LOBBY.shown),total>LOBBY.shown ? link(t('All Tasks'),'tasks','text-btn') : '');
  };
  /* U74: Home keeps WD4's first screen at any count -- the first decisions in the Host's order, the rest folded behind
   * one way that counts them (a person can pile up a dozen). */
  const HOME_DECISIONS = 5;
  const decisionGroup = (rows) => { const rest = rows.slice(HOME_DECISIONS), all = waiting(), shared = sharedWay(all), cause = sharedCause(all); return group(t('Needs a decision'), rows.slice(0, HOME_DECISIONS), 'card-list lines slotted', rows.length, rest.length ? html`<details class="reveal-details home-more"><summary>${countText(rest.length, '{n} more decision', '{n} more decisions')}</summary><div class="card-list lines slotted">${rest}</div></details>` : '', cause.words ? noteLine(t('{n} Tasks: {reason}', {n: cause.count, reason: t(cause.words)}), shared ? t('For each: {way}', {way: shared}) : '', 'warning') : shared ? html`<p class="caption group-note">${t('For each: {way}', {way: shared})}</p>` : ''); };
  const HOME_TOOLS = () => [{ic: 'data', action: 'go', value: 'data', word: t('Update data'), why: t('The working store; one explicit update at a time')}, {ic: 'evidence', action: 'go', value: 'evidence-stream', word: t('Prepare evidence'), why: t('Sources for the review desk')}, {ic: 'archive', action: 'go', value: 'storage', word: t('Storage cleanup'), why: t('Retention and reclaimable space')}];
  /* The Home's groups (N6): what needs a decision, what runs, what was recorded -- each only when
   * it holds something; a Home with nothing in any shows the one empty state and its way. */
  function homeGroups(unprepared) {
    const decide = decisionRows(), running = runningGroup(), forward = LiveActivation.forwardRows(), recent = Data.recent(8).map((r) => recordRow(r)), closure = LiveTasks.unrecoverableActions();
    if (!decide.length && !String(running).trim() && !forward.length && !recent.length && !closure) return html`${stackSlot('homeRunning','')}${emptyState(t('No saved research yet.'), unprepared ? '' : link(t('New experiment'), 'lab', 'button primary'), 'page-empty')}`;
    // V593 (U81): what runs forward, after what runs now (LiveActivation reads it from its owners)
    return html`${decisionGroup(decide)}${closure}${stackSlot('homeRunning',running)}${group(t('Running forward'), forward)}${group(t('Recently recorded'), recent, undefined, recent.length, '', '', LOBBY.shown)}`;
  }
  function overview() {
    const prep = Data.preparation(), unprepared = Boolean(prep) && !prep.inputs?.length;
    const f = resumeFacts(prep, unprepared);
    const meta = html`<span>${icon('task')}${countText(openTaskCount(), '{n} Task needs a decision', '{n} Tasks need a decision')}</span><span>${icon('lock')}${t('Local workspace')}</span>`;
    return html`${objectHead(Data.workspace(), meta, html`${f.secondary}${f.primary}`, f.state, HOME_TOOLS(), {object: true, facts: homeChips().filter(Boolean)})}${unprepared || LiveWorkspace.selectedTask('welcome') ? html`<section class="first-use" aria-label="${t('First use')}">${LiveWorkspace.firstUse()}</section>` : ''}<section class="home-groups" aria-label="${t('Home')}">${homeGroups(unprepared)}</section>`;
  }
  function tasks() {
    return LiveTasks.page();
  }

  function page() {
    if (!LiveTeam.pages.has(app.page) && !mapped.has(app.page) && !LiveWorkspace.pages.has(app.page)) return unwired();
    if (Data.workspaceStatus === 'loading') return html`${skeleton('head')}${skeleton('rows')}`;
    // U1: without the Host's session (opened outside its launch link, or a Host restarted since) a retry
    // cannot help; the way is the link the Host printed (ST6)
    if (Data.workspaceStatus === 'error' && Data.workspaceReopen) return banner(t('Open from the launch link'), t('The Host gives this page a session only through the link it printed when it started. Open the Workbench again from that link; a Host started again printed a new one.'), TONE.attention);
    if (Data.workspaceStatus === 'error') return banner(t('Workspace read refused'), Data.workspaceError, TONE.failure, btn(t('Try again'), 'workspace-refresh'));
    if (LiveGoals.pages.has(app.page)) return LiveGoals.page() || '';
    if (LiveModels.pages.has(app.page)) return LiveModels.page() || '';
    if (LiveFeatureResearch.pages.has(app.page)) return LiveFeatureResearch.page() || '';
    if (LiveTeam.pages.has(app.page)) return html`${LiveTeam.section()}`;
    if(LiveWorkspace.pages.has(app.page))return LiveWorkspace.page();
    if(LiveReview.pages.has(app.page))return LiveReview.page();
    if(LiveStudy.pages.has(app.page))return LiveStudy.page();
    if (DATA_PAGES.has(app.page) && (!Data.ready || !app.book)) return ['loading', 'error'].includes(Data.status) ? dataStateView() : bookless();
    return ({overview, tasks, advanced: Settings.page, settings: Settings.page, upgrade: Settings.upgrade, lab:LiveResearch.page}[app.page] || PAGES[app.page])();
  }
  /* A holding's detail (law 149): the window's column names it (kind, ticker, the close); its body is
   * the listing, the two weights, the facts of the date and the way to the book's provenance (the
   * page's Facts, where every study's provenance is read). */
  function holdingDetail(h) {
    const forward = Data.performanceMode?.() === 'forward', metadata = forward ? Data.raw()?.forward_holdings : Data.raw()?.holdings_basis;
    const weightLabel = metadata?.basis === 'CONDITIONAL_ESTIMATE' ? 'Proposed weight' : metadata?.basis === 'OBSERVED_RESEARCH_ENTRY' ? 'Research-entry weight' : metadata?.basis === 'HISTORICAL_REPLAY' ? 'Replay weight' : 'Holdings weight';
    const weightRows = [[t(weightLabel), h.current_weight == null || !Number.isFinite(Number(h.current_weight)) ? '' : num(Number(h.current_weight) * 100, 'percent')]];
    if (h.weight_change != null && Number.isFinite(Number(h.weight_change))) weightRows.push([t('Change (pp)'), signed(Number(h.weight_change) * 100, 'pp')]);
    if (h.target != null && h.current_weight != null && Number.isFinite(Number(h.target)) && Number.isFinite(Number(h.current_weight)) && Number(h.target) !== Number(h.current_weight) * 100) weightRows.push([t('Target'), num(h.target, 'percent')]);
    const dates = forward ? [[t('Formation session'), metadata?.formation_session], [t('Entry session'), metadata?.entry_session]] : [[t('trading|Session'), viewSession()], [t('Declared as-of'), Data.subject().input_date]];
    return html`<section class="inspector-section"><p class="caption">${t('Listing identity')} <span class="mono">${h.listing_id}</span></p>${kv(weightRows)}${kv([...dates.filter(([,value])=>value), [t('Sector'), h.sector ? html`${h.sector}${infoMark(t('the book\'s sealed Sector map'))}` : t('Not supplied')], [t('Review mapping'), t('Not evaluated')]])}<p class="small muted">${t('No review conclusion is inferred from a position or missing mapping.')}</p>${btn(t('Provenance'), 'facts', '', 'button compact')}</section>`;
  }
  /* A saved Portfolio named for reading: kind, declared candidate, short task and holdings date;
   * plain text for <option>, the same words elsewhere. Unknown members are named, never replaced. */
  function optionLabel(task) {
    const row = Data.history().find((r) => r.task_id === task && ['portfolio.policy-development','INSTALLED_RESULT'].includes(r.raw.kind));
    if (!row) return `${t('Task')} ${shortRef(task)} · ${t('not in discovered history')}`;
    return `${nameOf(row).name} · ${row.reference}${row.holdingsSession ? ' · ' + t('holdings') + ' ' + row.holdingsSession : ''}`;
  }
  /* The opened book in named facts beside its recorded sources. Policy, Alpha source, candidate and
   * Foundation come with the display projection of the same verified readback; the origin is the
   * plan's recorded origin from the same verified readback as the other source references.
   * Dates are not interchangeable and are each named. */
  /* The book's ··· (round 64): the comparison, the continuation and the two exports; one primary stays on the head. */
  const bookTools = (subject) => [{ic: 'grid', action: 'go', value: 'compare', word: t('Compare'), why: t('Set this book beside another saved one')}, {ic: 'branch', action: 'research-continue', value: subject.task_id, word: t('Continue as a new draft'), why: t('The same declaration, bound to the same source')}, {ic: 'file', action: 'export-html', word: t('Export HTML'), why: t('The owner-rendered report, exactly as saved')}, {ic: 'file', action: 'export-json', word: t('Export JSON'), why: t('The full readback')}].filter(v=>subject.source_kind!=='INSTALLED_RESULT'||v.action!=='research-continue');
  /* U52 (V347): what a result's window can claim about time, in the Host's statements -- generated from the Panel's
   * marks (T0, the initial cohort, survivorship, the Sector treatment, the price basis), shown as written, never
   * reworded: the first (the window against T0) beside the window, all of them in the Facts; none where not recorded. */
  const temporalSaid = (scope) => (scope?.statements || []).filter(Boolean).map((x) => t(x)); // U61: the owner's sentences, worded by i18n's templates
  const temporalRow = (scope) => { const said = temporalSaid(scope); return said.length ? [[t('Time and survivorship'), html`<span class="owner-text">${said[0]}</span>`]] : []; };
  const temporalAll = (scope) => { const said = temporalSaid(scope); return said.length ? html`<h4>${t('Time and survivorship')}</h4>${said.map((x) => html`<p class="owner-text">${x}</p>`)}` : ''; };
  function researchTiming(value) {
    if(!value) return '';
    const d=value.declared || {}, a=value.availability || {}, e=value.selected_event || {}, o=value.outcome || {}, r=value.evaluated || {}, tr=value.training || {}, life=tr.lifecycle || {}, split=tr.split_policy || {};
    const missing=t('Not recorded'), range=(s,z)=>s && z ? `${s} — ${z}` : missing;
    const cutoff=d.as_of && typeof d.as_of==='object' ? `${d.as_of.session} · ${codeWords(d.as_of.phase || 'UNSPECIFIED')}` : d.as_of ? `${d.as_of} · ${t('phase not recorded')}` : missing;
    const rows=[
      [t('Input observation range'),range(value.input?.start,value.input?.end)],
      [t('Requested research range'),range(d.start,d.end)],
      [t('Research cutoff, not execution time'),cutoff],
      [t('Evaluated formations'),html`${range(r.start,r.end)}${r.count!=null ? html` · ${r.count}` : ''}`],
      [t('Source availability evidence'),codeWords(a.status || 'NOT_RECORDED')],
      [t('Provider arrival timestamps'),t('Not measured; policy is not PIT proof')],
      [t('Decision information cutoff'),o.information_cutoff ? codeWords(o.information_cutoff) : missing],
      [t('Entry rule'),o.entry_timing ? codeWords(o.entry_timing) : missing],
      [t('Outcome end rule'),o.exit_timing ? codeWords(o.exit_timing) : missing],
      [t('Outcome maturity lag (sessions)'),o.maturity_lag_sessions ?? missing],
      [t('Target maturity lag (sessions)'),tr.maturity_lag_sessions ?? missing],
      [t('Training window / purge (sessions)'),`${life.training_window_sessions ?? split.train_sessions ?? missing} / ${life.purge_sessions ?? split.purge_sessions ?? missing}`],
      [t('Refit cadence'),life.month_interval ? t('Every {n} months; anchor month {m}',{n:life.month_interval,m:life.anchor_month}) : split.step_sessions!=null ? t('Fold step: {n} sessions',{n:split.step_sessions}) : missing],
    ];
    if(tr.selected_fold) rows.push([t('Selected score training dates'),range(tr.selected_fold.train_start,tr.selected_fold.train_end)],
      [t('Selected score validation dates'),range(tr.selected_fold.validation_start,tr.selected_fold.validation_end)]);
    if(value.rebalance) rows.push([t('Portfolio rebalance clock'),codeWords(value.rebalance.clock_id)],
      [t('Sleeves / hold-only sessions'),`${value.rebalance.tranches ?? missing} / ${value.rebalance.hold_session_count ?? missing}`]);
    if(o.status==='RISK_DIAGNOSTIC_NOT_EXECUTION') rows.push([t('Risk realization'),t('Next-session diagnostic, not a trade')],[t('Latest outcome session'),o.latest_outcome_session || missing]);
    if(e.status==='RECORDED_OUTCOME_SCHEDULE') rows.push(
      [t(e.purpose==='PLAN_EXAMPLE' ? 'Example formation from PLAN' : 'Selected Portfolio formation'),e.formation_session],
      [t('Decision at (recorded timestamp)'),e.formation_close_at],
      [t('Entry at (recorded timestamp)'),e.entry_open_at],
      [t('Outcome ends at (recorded timestamp)'),e.holding_end_open_at]);
    const primary=[...(e.status==='RECORDED_OUTCOME_SCHEDULE' ? [rows[2],...rows.slice(-4)] : [rows[0],rows[2],rows[3],rows[5]]),...temporalRow(value.temporal_scope)];
    const detail=html`${kv(rows)}${temporalAll(value.temporal_scope)}${(value.notices || []).map(v=>html`<p class="caption">${t(v)}</p>`)}${(a.sources || []).map(v=>html`<h4>${codeWords(v.policy_id)}</h4>${kv([[t('Available after'),codeWords(v.phase_name)],[t('Publication delay (sessions)'),v.publication_delay_sessions],[t('Revision risk'),v.revision_risk]])}`)}`;
    // a section on the ground (MA2): its label, the policy caveat as the label's (i), the primary clocks, the rest in Facts
    return panel(t('Research timing'),t('Source availability is declared policy, not measured provider arrival or PIT proof.'),html`${kv(primary,'kv-columns')}${factsRef(t('All research clocks and source policies'),detail)}`);
  }
  function bookFactsSections() {
    const s = Data.subject(), raw = Data.raw(), doc = raw.declaration || {}, src = raw.source || {};
    const committee=Data.performanceMode?.()==='forward' ? raw.forward_holdings?.committee_context : null, collaboration=collaborationRows(s.task_id,s.result_hash,committee?.update_task_id,committee?.review_selector?.update_publication_hash,committee);
    if (s.source_kind === 'INSTALLED_RESULT') {
      const clock=raw.positionClock || {}, policy=raw.bookPolicy || {};
      return [
        {title:t('Team'),body:kv(sourceRows(collaboration))},
        {title:t('Sources'),body:html`${kv([[t('Strategy'),s.title],[t('Exact result'),html`<span class="mono">${s.result_hash}</span>`],[t('Source input'),html`<span class="mono">${s.input_hash || ''}</span>`],[t('Source status'),codeWords(src.status || '')]])}${(src.alpha_tasks || []).map(v=>feature('branch',v.component_id,html`<span class="mono">${shortRef(v.task_id)}</span>`,link(t('Open Alpha study'),'alpha','button compact',{study:v.task_id})))}${src.reason ? html`<p class="caption">${src.reason}</p>` : ''}`},
        {title:t('Research timing'),body:html`${kv([[t('Input data through'),s.input_date || ''],[t('Formation date'),s.session],[t('Decision phase'),codeWords(clock.decision_phase || '')],[t('Execution session'),clock.entry_session || ''],[t('Outcome end session'),clock.holding_end_session || '']])}${temporalAll(raw.temporalScope)}<p class="caption">${t('The date selector is the formation date, not the execution date. Changing it reads saved weights and runs nothing.')}</p>`},
        {title:t('Frozen book policy'),body:html`${kv([[t('Weight rule'),weightWords(doc.weight_rule)],[t('Names per sleeve'),doc.top_k],[t('Sleeves'),doc.tranches],[t('Exit rank'),doc.exit_rank],[t('Sleeve notional'),codeWords(policy.sleeve_notional || '')],[t('Initial staging'),codeWords(policy.initial_staging || '')],[t('Review phase'),policy.review_phase ?? ''],[t('Risk role'),codeWords(raw.riskDisposition || '')]])}<p class="caption">${t('The saved HTML is the original whole report. The dated JSON contains exact weights and targets; legacy generic sleeve wording does not replace this frozen policy.')}</p>`},
        {title:t('Recorded limitations'),body:html`${(raw.limitations || []).map(v=>html`<p class="caption">${coded(v)}</p>`)}`},
      ];
    }
    const sessions = Data.sessions(), recorded = Data.history().find((r) => r.task_id === s.task_id && r.raw.kind === 'portfolio.policy-development');
    const method = html`${doc.catalog_policy ? policyWords(doc.catalog_policy) : t('{k} names per sleeve · {n} sleeves · exit rank {e} · {w}', {k: doc.top_k ?? '', n: doc.tranches ?? '', e: doc.exit_rank ?? '', w: weightWords(doc.weight_rule)})} <span class="sub-cell">${t('{c} bps per side', {c: doc.cost_bps_per_side ?? s.cost_per_side})} · ${methodWords(doc.unavailable_return_policy || '')}</span>`;
    const facts = html`<p class="caption">${t('Declared facts of the saved book; not a performance claim.')}</p>${kv([
      [t('Method'), method],
      [t('Source input'), html`${s.input_id} <span class="sub-cell">${cutoffText(inputState(s.input_hash))}</span>`],
      [t('Declared interval'), html`${s.support?.start || ''} — ${s.support?.end || ''} <span class="sub-cell">${t('research cutoff')} ${s.input_date}${s.support?.as_of?.phase ? html` · ${codeWords(s.support.as_of.phase)}` : ''}</span>`],
      [t('Report interval'), html`${sessions[0] || ''} — ${sessions.at(-1) || ''} <span class="sub-cell">${t('{n} published sessions', {n: count(sessions.length)})}</span>`],
      [t('Holdings date'), html`<span data-session-label>${s.session}</span>${infoMark(t('the date selected below; changing it never changes this book'))}`],
      [t('Recorded'), html`${recorded?.recordedAt || ''}${infoMark(t('Published development evidence; not current authority'))}`],
      [t('Sector map'), raw.sectors?.status === 'BOOK_EXECUTION' ? html`${t('sealed with the book\'s Panel')} <span class="sub-cell">${t('revision')} <span class="mono">${short(raw.sectors.sector_revision || '', SHORT.hash)}</span></span>` : t('not resolved')],
      [t('Recorded limitations'), (raw.limitations || []).length ? html`${(raw.limitations || []).map((v, i) => html`${i ? ' · ' : ''}${typeof v === 'string' ? coded(v) : html`<span class="mono">${JSON.stringify(v)}</span>`}`)}` : ''],
    ])}`;
    const risk = 'risk_task_id' in src; // U41 (V340): the Risk study the weights or covariance read; none for equal weight
    const sources = html`<p class="caption">${t('Recorded references only; nothing is looked up as the latest result.')}</p>${kv([
      [t('Alpha study'), src.alpha_task_id ? html`<span class="mono">${shortRef(src.alpha_task_id)}</span> ${link(html`${t('Open Alpha study')} ${icon('arrow')}`, 'alpha', 'button compact', {study: src.alpha_task_id})}` : ''],
      ...(risk ? [[t('Risk study'), src.risk_task_id ? html`<span class="mono">${shortRef(src.risk_task_id)}</span>${src.risk_surface_hash ? html` <span class="sub-cell">${t('surface')} <span class="mono" data-tip="${src.risk_surface_hash}">${short(src.risk_surface_hash, SHORT.hash)}</span></span>` : ''} ${link(html`${t('Open Risk study')} ${icon('arrow')}`, 'risk', 'button compact', {study: src.risk_task_id})}` : t('None · equal weight reads no Risk study')]] : []),
      [t('Candidate'), html`<span class="mono">${src.candidate_id || ''}</span> <span class="sub-cell" data-tip="${src.target_recipe_id || ''}">${src.target_recipe_id ? methodWords(src.target_recipe_id) : ''}</span>`],
      [t('Foundation admission'), src.foundation_admission_hash ? html`<span class="mono">${short(src.foundation_admission_hash, SHORT.hash)}</span> ${link(html`${t('Open Foundation')} ${icon('arrow')}`, 'foundation', 'button compact', {foundation: src.foundation_admission_hash})}` : t('None recorded · drafted from the development input, not a sealed Foundation')],
      [t('Exact input'), html`${s.input_id} · ${cutoffText(inputState(s.input_hash))} · <span class="mono">${short(s.input_hash || '', SHORT.hash)}</span>`],
      [t('Continued from'), src.origin_task_id ? html`<span class="mono">${shortRef(src.origin_task_id)}</span> ${link(html`${t('Open origin study')} ${icon('arrow')}`, 'portfolio', 'button compact', {book: src.origin_task_id, session: ''})}` : t('Not continued · authored directly')],
    ])}<p class="caption">${t('Compare another saved book, open the linked reports under Diagnostics, or continue this declaration as a new draft. Neither runs research.')}</p>`;
    // Round 64: the sources as one kv (each value the way to its object), the team sessions that
    // name the book, the recorded limitations, the declared facts — sections of the inspector's
    // Facts mode; the page leads with the chart.
    const rows = [
      {icon: 'branch', label: t('Alpha study'), value: src.alpha_task_id ? html`<span class="mono">${shortRef(src.alpha_task_id)}</span>` : '', to: src.alpha_task_id ? {page: 'alpha', extra: {study: src.alpha_task_id}} : null},
      ...(risk ? [{icon: 'branch', label: t('Risk study'), value: src.risk_task_id ? html`<span class="mono">${shortRef(src.risk_task_id)}</span>` : t('none · equal weight'), to: src.risk_task_id ? {page: 'risk', extra: {study: src.risk_task_id}} : null}] : []),
      {icon: 'cube', label: t('Candidate'), value: src.candidate_id ? html`<span class="mono" data-tip="${src.candidate_id}">${short(String(src.candidate_id).replace(/^alpha-candidate-/, ''), SHORT.id)}</span>` : '', note: src.target_recipe_id ? methodWords(src.target_recipe_id) : ''},
      {icon: 'archive', label: t('Foundation'), value: src.foundation_admission_hash ? html`<span class="mono">${short(src.foundation_admission_hash, SHORT.hash)}</span>` : t('none recorded'), to: src.foundation_admission_hash ? {page: 'foundation', extra: {foundation: src.foundation_admission_hash}} : null},
      {icon: 'cube', label: t('Exact input'), value: html`${s.input_id} · ${cutoffText(inputState(s.input_hash))}`},
      {icon: 'history', label: t('Continued from'), value: src.origin_task_id ? html`<span class="mono">${shortRef(src.origin_task_id)}</span>` : t('authored directly'), to: src.origin_task_id ? {page: 'portfolio', extra: {book: src.origin_task_id, session: ''}} : null},
      ...collaboration,
    ];
    const limits = (raw.limitations || []).length ? html`<div class="card-list">${(raw.limitations || []).map((v) => objectRow({lead: statusDot('metadata', t('Recorded limitation')), name: typeof v === 'string' ? coded(v) : html`<span class="mono">${JSON.stringify(v)}</span>`, cls: 'limit-row'}))}</div>` : '';
    return [{title: t('Sources'), body: kv(sourceRows(rows))}, limits ? {title: t('Recorded limitations'), body: limits} : null, {title: t('This study'), body: facts}, {title: t('Sources and continuation'), body: sources}, {title: t('Documents'), body: codeRef(t('Read the exact declaration (JSON)'), doc)}].filter(Boolean); // round 92: the book's declaration as a document, as the studies' Facts carry theirs
  }
  /* Reports already attached to, and reviews already published for, this exact book. Nothing is
   * attached, refreshed or reviewed from here; absence is stated, never filled in. */
  function reports() {
    const s = Data.subject(), selector = Data.raw()?.reviewSelector, links = Data.riskLinks(s.task_id);
    const refusedLinks=(Data.riskLinkRefusals?.(s.task_id) || []).map(r=>refusal(r,'warning',{catalog:true,more:html`<p>${t('Risk link')} ${hashCell(r.link_hash)}</p>`,next:prerequisiteWays(r.next_requests)}));
    const sameBook = (b) => b && (s.source_kind==='INSTALLED_RESULT' ? b.result_hash===s.result_hash : b.experiment_task_id === s.task_id && b.experiment_receipt_hash === s.receipt_hash);
    const reviews = Data.history().filter((r) => r.raw.review_publication_hash && sameBook(r.raw.book));
    const riskBody = s.source_kind==='INSTALLED_RESULT'
      ? html`<p>${codeWords(Data.raw()?.riskDisposition || 'NOT_RECORDED')}</p><p class="caption">${t('Risk attribution belongs to this saved path. It is not an authored-study report link or an instruction to change weights.')}</p>`
      : links === null
      ? (Data.riskLinksError ? notRead(t('Linked reports unavailable'), Data.riskLinksError) : skeleton('rows'))
      : links.length
        ? html`${links.map((v) => feature('evidence', html`${t('Risk study')} <span class="mono">${shortRef(v.risk_task_id)}</span>`, html`${v.risk_start} — ${v.risk_end} · ${t('{r} / {p} formations · risk / portfolio', {r: v.risk_formation_count, p: v.portfolio_formation_count})} · ${t('attached by')} ${v.attached_by}`, html`${link(html`${t('Open Risk study')} ${icon('arrow')}`, 'risk', 'button compact', {study: v.risk_task_id})}${btn(t('Export JSON'), 'portfolio-risk-export', v.link_hash, 'button compact')}`))}`
        : refusedLinks.length ? '' : emptyState(html`${t('No Risk report is linked to this Portfolio task.')}${infoMark(t('An association is created from a saved Risk study with explicit confirmation; nothing is attached automatically.'))}`);
    const reviewBody = html`${kv([[t('Book reference'), html`<span class="mono">${shortRef(s.task_id)}</span> · <span class="mono">${short(s.receipt_hash || '', SHORT.hash)}</span>`], [t('Holdings date'), selector?.portfolio_session || s.session]])}<div class="flow">${selector ? html`${link(html`${t('Working review')} ${icon('arrow')}`, 'evidence', 'button compact', {review_selector: JSON.stringify(selector), review_publication: ''})}${link(html`${t('Research report')} ${icon('arrow')}`, 'report', 'button compact', {review_selector: JSON.stringify(selector), review_publication: ''})}` : t('No review selector was published for this book.')}</div>${reviews.length ? html`<div>${reviews.map((r) => feature('review', html`${t('Saved review')} ${r.recordedAt || ''} · <span class="mono">${short(r.raw.review_publication_hash, SHORT.hash)}</span>`, html`${t('Reviewed holdings')} ${r.raw.book.portfolio_session || ''}`, link(html`${t('Open pinned review')} ${icon('arrow')}`, 'evidence', 'button compact', {review_selector: JSON.stringify(r.raw.book), review_publication: r.raw.review_publication_hash})))}</div><p class="caption">${t('A pinned review keeps its publication hash; the working review is the current eligibility state.')}</p>` : html`<p class="caption">${t('No saved CRO review for this book. Opening the working review runs no analysis, refresh or model work.')}</p>`}`;
    return html`${refusedLinks}<div class="grid-2 detail-section" data-stack-box="section">${panel(t('Linked Risk reports'), `${t('Discovered for this Portfolio task; not a sizing input.')} ${t('Report-only references; the covariance estimates did not size this book.')}`, riskBody, btn(t('Refresh'), 'portfolio-reports-refresh', '', 'text-btn'))}${panel(t('Reviews and reports'), t('Evidence/CRO for this exact book. Working eligibility and a pinned historical review stay distinct.'), reviewBody)}</div>`;
  }
  /* V617 (U94): a whole-report metric the saved report does not record says so, the owner's reason on hover
   * (`metricAbsences`, keyed by the report's own field); nothing is estimated here. */
  const METRIC_FIELDS = {total: 'cumulative_return', annual: 'annualized_return', vol: 'annualized_volatility', drawdown: 'maximum_drawdown', sharpe: 'sharpe', sortino: 'sortino', informationRatio: 'information_ratio', benchmarkRelative: 'benchmark_relative_return', beta: 'beta', trackingError: 'tracking_error', jensenAlpha: 'zero_cash_jensen_alpha'};
  const metricAbsence = (key) => (Data.metricAbsences ? Data.metricAbsences() : {})?.[METRIC_FIELDS[key]] || null;
  const absentMetric = (key) => { const a = metricAbsence(key); return a ? hint(t('Not recorded'), a.detail ? t(a.detail) : codeWords(a.reason || a.status || '')) : ''; };
  function portfolioDetails(tab) {
    const m = Data.metrics(), u = Data.universe(), s = Data.subject();
    Data.discoverRiskLinks(s.task_id); // the report area is the consumer: discovered when shown, once per opened book
    const has = (v) => v !== null && v !== undefined;
    if (Data.performanceMode?.() === 'forward') {
      const f=Data.forwardPerformance(), p=f?.selected_window_metric_provenance || {};
      const statusWords={INSUFFICIENT_REALIZED_OBSERVATIONS:t('Fewer than two settled outcomes were recorded'),NO_REALIZED_FORWARD_PUBLICATION:t('No realized forward publication is available for this book and cost lane.')};
      const claimWord=p.claim==='POST_OBSERVED_QA_NOT_TIMELY_ADVICE' ? codeWords(p.claim) : '';
      const executionWord=p.execution_basis==='DAILY_BAR_QA_NOT_VERIFIED_VENUE_EXECUTION' ? t('Daily bar QA returns; not verified venue execution.') : '';
      const facts=[...(statusWords[f?.status] ? [[t('Status'),statusWords[f.status]]] : []),[t('Source book task'),hashCell(p.source_book_task_id,Infinity)],[t('Publication'),hashCell(p.publication_hash)],[t('Selected window'),`${p.selected_start || ''} — ${p.selected_end || ''}`],[t('Observations'),count(p.observation_count || 0)],[t('Observed through'),p.observed_through || ''],[t('Published'),p.published_at || ''],[t('Window proof'),hashCell(p.window_hash)],[t('Cost lane'),`${p.cost_bps_per_side || f?.cost_bps_per_side || ''} bps per side`],...(claimWord ? [[t('Claim'),claimWord]] : []),...(executionWord ? [[t('Execution basis'),executionWord]] : [])];
      return html`<div class="grid-2 detail-section" data-stack-box="section">${panel(t('Realized forward source'),t('This read is separate from the historical research report above.'),kv(facts))}${panel(t('Historical report remains pinned'),t('The research report values and holdings are still available in Historical report mode; they are not mixed into realized returns.'),html`${kv([[t('Declared interval'),`${s.support?.start || ''} — ${s.support?.end || ''}`],[t('Research cutoff'),s.input_date],[t('Declared cost'),`${s.cost_per_side ?? ''} bps per side`]])}`)}</div>${reports()}`;
    }
    return html`<div class="grid-2 detail-section" data-stack-box="section">${panel(t('Research diagnostics'), t('The owner\'s whole-report values, net of the declared cost; not recomputed here.'), kv([[t('Information ratio'), html`${has(m.informationRatio) ? fmt(m.informationRatio) : absentMetric('informationRatio')}${infoMark(t('against the owner\'s benchmark series, whole report'))}`], [t('Return relative to benchmark'), html`${has(m.benchmarkRelative) ? num(m.benchmarkRelative, 'percent') : absentMetric('benchmarkRelative')}${infoMark(t('whole report, the owner\'s value'))}`], [t('Beta to benchmark'), has(m.beta) ? fmt(m.beta) : absentMetric('beta')], [t('Tracking error'), html`${has(m.trackingError) ? num(m.trackingError, 'percent') : absentMetric('trackingError')}${infoMark(t('annualized, whole report'))}`], [t('Jensen alpha (zero cash)'), has(m.jensenAlpha) ? num(m.jensenAlpha, 'percent') : absentMetric('jensenAlpha')], [t('One-way turnover at holdings date'), html`${num(m.turnover, 'percent')}${infoMark(t('of portfolio weight, at {date} only', {date: s.session}))}`], [t('Declared cost'), html`${num(s.cost_per_side, 'bps')}${infoMark(t('per side, the declared assumption'))}`], [t('Platform one-way cost'), html`${num(m.costBps, 'bps')}${infoMark(t('on platform one-way turnover, whole report'))}`], [t('Effective research universe'), `${u.eligible ?? ''} / ${u.total ?? ''}`], [t('Quarantined listings'), u.quarantine ?? '']]))}${panel(t('Scope'), u.definition || t('No data-quality notice recorded.'), html`${kv([[t('Support'), `${s.support?.start || ''} — ${s.support?.end || ''}`], [t('Research cutoff'), s.input_date], [t('Benchmark'), Data.series().some((r) => r.benchmark !== null) ? t('The owner\'s benchmark series published with this book; the relative return, beta and tracking error above are the owner\'s') : t('Not supplied')]])}<p class="caption">${t('A missing benchmark, sector or mapping is a stated absence, not zero. Whole-report values are not the selected session\'s; the holdings tab shows that session\'s own facts.')}</p>`)}</div>${reports()}`;
  }
  const cell = (v) => (typeof v === 'number' ? Number.isInteger(v) ? count(v) : fmt(v, 6) : v === null || v === undefined ? '' : Array.isArray(v) ? v.join(', ') : String(v));
  /* The owner's comparison, read as returned: its own position date, common support, both
   * declarations and limitations, then session-level rows apart from whole-report rows. */
  /* Both books' published rows on one chart: each indexed from 100 immediately before the first
   * common session (the same convention the display projection uses), the benchmark from A.
   * Rows are matched by session; nothing is aligned, filled or recomputed. */
  function overlayRows(left, right) {
    const other = new Map((right.series || []).map((r) => [r.session, r]));
    const finite = (z) => z !== null && z !== undefined && Number.isFinite(z);
    let a = 100, b = 100, bm = 100;
    const rows = [];
    for (const r of left.series || []) {
      const o = other.get(r.session);
      if (!o) continue;
      a = a === null || !finite(r.net_simple_return) ? null : a * (1 + r.net_simple_return);
      b = b === null || !finite(o.net_simple_return) ? null : b * (1 + o.net_simple_return);
      bm = bm === null || !finite(r.benchmark_simple_return) ? null : bm * (1 + r.benchmark_simple_return);
      rows.push({date: r.session, a, b, benchmark: bm});
    }
    return rows;
  }
  function overlayChart(left, right) {
    const rows = overlayRows(left, right);
    if (rows.length < 2) return '';
    const lines = [{key: 'a', cls: 'series', label: 'Study A'}, {key: 'b', cls: 'series series-b', label: 'Study B'}, {key: 'benchmark', cls: 'benchmark', label: 'Benchmark'}];
    const change=rows[0].a ? ((rows.at(-1).a/rows[0].a)-1)*100 : null;
    return figureBox(t('Two saved paths'),html`<div id="compareChart">${chartHead(signed(change,'percent'),t('Study A'),dateRange(rows[0].date,rows.at(-1).date))}${chart('indexed', true, {id: 'compare', rows, lines, title: t('Two saved paths'), label: t('Two saved books indexed from 100; read-only chart.')})}</div><p class="table-note">${t('Both books indexed from 100 before their first common session, {from} — {to}, from the owner\'s published rows; nothing is aligned or recomputed, and a higher line is not a recommendation.', {from: rows[0].date, to: rows.at(-1).date})}</p>`,{cls:'compare-overlay',legend:html`<span class="chart-legend"><span><i class="legend-line"></i>${t('A')} · <span class="mono">${shortRef(left.task_id)}</span></span><span><i class="legend-line series-b"></i>${t('B')} · <span class="mono">${shortRef(right.task_id)}</span></span><span><i class="legend-line benchmark"></i>${t('Benchmark')}</span></span>`});
  }
  function comparisonBody(v, s) {
    const dims = (list, title, sub) => (list.length ? html`<section class="panel section-gap" data-box="table"><div class="table-toolbar"><div class="panel-label"><h2>${title}</h2>${infoMark(sub)}</div></div>${list.map((d) => html`<h3>${codeWords(d.dimension)}</h3>${table([{label:t('Metric'),type:'text'},{label:t('Unit'),type:'text'},{label:t('Study A'),type:'num'},{label:t('Study B'),type:'num'}], d.metrics.map((m) => tr([m.label, m.unit || '', cell(m.left), cell(m.right)])))}`)}</section>` : '');
    // two installed results (the product, 3de3f325), read in the authored comparison's shape: the
    // owner's disposition in words, its context with short identities, the dimensions in one table box
    if (v.kind === 'INSTALLED_RESULT_COMPARISON') {
      const windowText=side=>`${side?.window?.selected_start || ''} — ${side?.window?.selected_end || ''}`;
      const context=panel(t('Returned comparison context'),t('Different periods remain different. No common-window alignment, isolated-effect claim or winner is inferred.'),kv([[t('Study A'),windowText(v.left)],[t('Study B'),windowText(v.right)],[t('Exact result A'),hashCell(v.left?.result_hash,SHORT.hash)],[t('Exact result B'),hashCell(v.right?.result_hash,SHORT.hash)]]));
      return html`${noteLine(t('Product comparison'),coded(v.disposition))}${context}${dims(v.dimensions || [],t('Whole report interval'),t('Whole saved reports; the holdings-date selector does not reslice either report.'))}`;
    }
    const support = v.support || [], left = v.left || {}, right = v.right || {};
    const requested = Data.comparisonKey()?.session || '', positionDate = left.position?.session, rightDate = right.position?.session;
    const returnedDates = positionDate === rightDate ? positionDate || '' : `A ${positionDate || ''} · B ${rightDate || ''}`;
    const cursorDiffers = Boolean(positionDate) && positionDate !== s.session;
    const policy = (side) => { const d = side.document?.portfolio || {}; return d.policy ? `${policyWords(d.policy)} · ${d.cost_bps_per_side ?? ''} ${t('bps per side')}` : `${d.top_k ?? ''} / ${d.tranches ?? ''} · ${t('exit rank')} ${d.exit_rank ?? ''} · ${weightWords(d.weight_rule)} · ${d.cost_bps_per_side ?? ''} ${t('bps per side')}`; };
    const context = panel(t('Returned comparison context'), t('What the owner actually compared.'), html`${kv([
      [t('Compared position date'), html`${returnedDates}${infoMark(requested ? t('Both books were requested at {date}, the holdings date shown for A.', {date: requested}) : t('No date was requested; each book was read at its own default date.'))}${cursorDiffers ? html` <span class="sub-cell">${t('The owner returned a different date; the Portfolio page shows {cursor}.', {cursor: s.session})} ${link(html`${t('Open A at that date')} ${icon('arrow')}`, 'portfolio', 'button compact', {book: left.task_id, session: positionDate})}</span>` : ''}`],
      [t('Common support'), html`${support[0] || ''} — ${support.at(-1) || ''} <span class="sub-cell">${t('{n} formation sessions', {n: count(support.length)})}</span>`],
      [t('A declaration'), html`<span class="mono">${shortRef(left.task_id)}</span> · ${policy(left)}`],
      [t('B declaration'), html`<span class="mono">${shortRef(right.task_id)}</span> · ${policy(right)}`],
      [t('A limitations'), html`<span class="mono">${(left.limitations || []).join(' · ') || ''}</span>`],
      [t('B limitations'), html`<span class="mono">${(right.limitations || []).join(' · ') || ''}</span>`],
      [t('Risk'), html`${codeWords(v.risk?.status)}${infoMark(v.risk?.detail || '')}`],
    ])}${factsRef(html`${t('Exact identity')}`, html`${kv([[t('A task'), html`<span class="mono">${left.task_id || ''}</span>`], [t('A receipt'), html`<span class="mono">${left.receipt?.receipt_hash || ''}</span>`], [t('B task'), html`<span class="mono">${right.task_id || ''}</span>`], [t('B receipt'), html`<span class="mono">${right.receipt?.receipt_hash || ''}</span>`], [t('Disposition'), v.disposition || '']])}`)}`);
    const sessionLevel = (d) => d.metrics.every((m) => Object.hasOwn(left.position || {}, m.label));
    const dimensions = v.dimensions || [];
    return html`${noteLine(t('Product comparison'), v.claim || v.disposition)}${overlayChart(left, right)}${context}${dims(dimensions.filter(sessionLevel), t('At the compared position date'), t('Session-level position, cost and return facts at {date}; not the whole report.', {date: positionDate || ''}))}${dims(dimensions.filter((d) => !sessionLevel(d)), t('Whole report interval'), t('Report-level performance over the common support; a different period from the position rows above.'))}`;
  }
  function comparePage() {
    const s = Data.subject(), value = Data.comparison(), other = app.compareOther || '';
    const candidates = Data.portfolioEntries().filter((x) => x.task_id !== s.task_id && (s.source_kind==='INSTALLED_RESULT' ? x.kind==='INSTALLED_RESULT' : x.kind==='portfolio.policy-development')).map((x) => [x.task_id, optionLabel(x.task_id)]);
    if (other && !candidates.some(([id]) => id === other)) candidates.push([other, optionLabel(other)]); // a routed member outside the discovered list stays visible, never replaced
    const a = panel(t('Study A'), t('The open book; the owner reports the exact comparison scope.'), html`<div class="doc-facts">${kv([[t('Study'), optionLabel(s.task_id)], [t('Task'), html`<span class="mono" data-tip="${s.task_id}">${shortRef(s.task_id)}</span>`], [t('Holdings date shown'), html`${s.session}${infoMark(t('change it on the Portfolio page; the pair is kept'))}`], [t('Declared interval'), `${s.support?.start || ''} — ${s.support?.end || ''}`]])}</div>`);
    const b = panel(t('Study B'), t('Choose explicitly; no nearest or best match is picked.'), other ? html`<div class="doc-facts">${kv([[t('Study'), optionLabel(other)], [t('Task'), html`<span class="mono" data-tip="${other}">${shortRef(other)}</span>`]])}</div>` : emptyState(t('No second study chosen.')));
    const subject = subjectChoice(t('Compare with'), 'compareSelect', [['', t('Choose a saved study')], ...candidates], {selected: other}); // round 93: the header's subject line
    const body = Data.comparisonStatus === 'loading' ? skeleton('body')
      : Data.comparisonStatus === 'error' ? refusal({code: Data.comparisonError, reason: t('The owner decides compatibility; the selection is kept and nothing is re-aligned, truncated or refitted.')}, TONE.attention, {word: t('Comparison not admitted')})
      : value ? comparisonBody(value, s) : noteLine(t('Select a second study'), t('No comparison or winner is inferred.'));
    // the head is the open study's (the object; the page's word is the crumb's way up), with
    // Compare's lede; the second study is the subject line (2026-09-21, the Studies audit)
    return html`${objectHead(t('Compare'), `${t('Compatibility and comparison values come from the product owner.')} ${t('What is compared: two saved books, both read at study A\'s holdings date -- the position facts of that one date and the whole-report performance over their common support, in the owner\'s units. The owner decides whether the pair is compatible and refuses otherwise with its reason; a larger number is not a recommendation, and no winner is inferred. The exact response is exportable as the owner returned it.')}`, value && Data.comparisonStatus === 'ready' ? btn(t('Export comparison JSON'), 'compare-export', '', 'button') : '', '', [], {subject})}<div class="grid-2" data-stack-box="section">${a}${b}</div>${body}`;
  }
  function proofBody(key, observation = 0) {
    if(LiveStudy.pages.has(app.page))return LiveStudy.proof();
    if(LiveReview.pages.has(app.page))return LiveReview.proof();
    if(app.page==='lab') {
      const c = LiveResearch.context(), handoff = c.handoff_source;
      return html`<p class="proof-intro">${t('The draft on this page: what it is bound to, and the PLAN or Task it has.')}</p>${kv([
        [t('Input'), c.input_id || ''], [t('Input version'), c.input_binding_hash ? html`<span class="mono">${c.input_binding_hash}</span>` : t('none selected')],
        [t('Continued from'), c.origin_task_id ? html`${t('Task')} <span class="mono">${c.origin_task_id}</span>` : t('not continued · authored directly')],
        [t('Handoff source'), handoff ? html`<span class="mono">${typeof handoff === 'string' ? handoff : JSON.stringify(handoff)}</span>` : ''],
        [t('PLAN'), c.plan_hash ? html`<span class="mono">${c.plan_hash}</span>` : t('none yet')], [t('Task on this page'), c.work_task_id ? html`<span class="mono">${c.work_task_id}</span>` : t('none')]])}<p class="caption">${t('Editing is not execution; create a PLAN to validate.')}</p>`;
    }
    // an observation's fields in words, never the row's keys (WD2; the closing zh census found `value` and `benchmarkDaily`)
    const OBSERVATION_WORDS = {date: 'Date', daily: 'Daily return', value: 'Study index', benchmark: 'Benchmark index', benchmarkDaily: 'Benchmark daily return'};
    const s = Data.subject(), row = Data.series()[observation];
    if (app.page==='portfolio' && Data.performanceMode?.()==='forward') {
      const f=Data.forwardPerformance(), p=f?.selected_window_metric_provenance || {};
      if(key==='observation') return row ? html`<p class="proof-intro">${t('The realized per-session net return as supplied by Portfolio.')}</p>${kv([[t('Formation session'),row.formation_session || ''],[t('Entry session'),row.entry_session || ''],[t('Holding end'),row.holding_end_session || ''],[t('Net simple return'),row.net_simple_return ?? '']])}` : t('No realized observation is recorded for this window.');
      if(key in Data.metrics() || metricAbsence(key)) return html`<p class="proof-intro">${t('Owner-reported metric for the selected realized window; never combined with historical research metrics.')}</p>${kv([[t('Published metric'),key],[t('Value'),Number.isFinite(Data.metrics()[key]) ? fmt(Data.metrics()[key],6) : absentMetric(key)],[t('Value as returned'),f?.selected_window_metrics?.[({annual:'annualized_return',vol:'annualized_volatility',drawdown:'maximum_drawdown',sharpe:'sharpe',sortino:'sortino'})[key]] ?? ''],[t('Unit'),['annual','vol','drawdown'].includes(key)?'%':'ratio'],[t('Owner return unit'),p.return_unit || 'FRACTION'],[t('Window'),`${p.selected_start || ''} — ${p.selected_end || ''}`],[t('Window proof'),hashCell(p.window_hash)],[t('Publication'),hashCell(p.publication_hash)],[t('Source book task'),hashCell(p.source_book_task_id,Infinity)],[t('Observed through'),p.observed_through || ''],[t('Published'),p.published_at || ''],[t('Annualization sessions'),p.annualization_sessions_per_year ?? ''],[t('Volatility degrees of freedom'),p.volatility_degrees_of_freedom ?? ''],[t('Sharpe cash return per session'),p.sharpe_cash_return_per_session ?? ''],[t('Sortino downside threshold'),p.sortino_downside_threshold ?? ''],[t('Claim'),p.claim ? t(p.claim) : ''],[t('Execution basis'),p.execution_basis ? t(p.execution_basis) : '']])}`;
      if(Data.performanceMode?.()==='forward' && key in {annual:1,vol:1,drawdown:1,sharpe:1,sortino:1}) return html`<p class="proof-intro">${t('No eligible realized forward publication was returned for this book and cost lane.')}</p>${kv([[t('Status'),f?.status || ''],[t('Requested exact book task'),hashCell(Data.subject()?.task_id,Infinity)]])}`;
      if(key==='context') return html`<p class="proof-intro">${t('Realized forward performance is an owner readback for the exact installed book and cost lane.')}</p>${kv([[t('Status'),f?.status || ''],[t('Strategy package'),hashCell(p.strategy_package_hash)],[t('Checkpoint'),hashCell(p.checkpoint_hash)],[t('Publication'),hashCell(p.publication_hash)],[t('History prefix'),hashCell(p.history_prefix_hash)],[t('Settlement path'),hashCell(p.settlement_path_hash)],[t('Source snapshot'),hashCell(p.source_snapshot_hash)],[t('Window proof'),hashCell(p.window_hash)]])}`;
    }
    if (app.page === 'portfolio' && Data.performanceMode?.() === 'historical' && Data.rollingPerformance?.()) {
      const rolling = Data.rollingPerformance(), metrics = rolling.metrics || {}, fields = {total: 'cumulative_return', annual: 'annualized_return', vol: 'annualized_volatility', drawdown: 'maximum_drawdown', sharpe: 'sharpe', sortino: 'sortino'};
      const claims = {DEVELOPMENT_EVIDENCE_WITH_POST_OBSERVED_QA_EXTENSION: 'Development evidence extended with post-observed QA outcomes'};
      const executionBases = {DAILY_BAR_QA_NOT_VERIFIED_VENUE_EXECUTION: 'Daily-bar QA; venue execution is not verified'};
      if (key in fields && (key in Data.metrics() || metricAbsence(key))) {
        const range = rolling.formation_range || {}, metric = fields[key], returned = metrics[metric], headStorage = {SEALED: 'Sealed', LEGACY_DERIVED_NOT_PERSISTED: 'Legacy derived; not persisted'};
        return html`<p class="proof-intro">${t('Owner-reported metric for rolling outcomes, separate from the pinned saved report.')}</p>${kv([
          [t('Published metric'), key],
          [t('Value'), Number.isFinite(Data.metrics()[key]) ? fmt(Data.metrics()[key], 6) : absentMetric(key)],
          [t('Value as returned'), Number.isFinite(returned) ? fmt(returned, 8) : ''],
          [t('Unit'), ['total', 'annual', 'vol', 'drawdown'].includes(key) ? '%' : 'ratio'],
          [t('Owner return unit'), 'FRACTION'],
          [t('Formation range'), `${range.start || ''} — ${range.end || ''}`],
          [t('Base report ends'), range.base_end || ''],
          ...(rolling.outcomes_observed_through ? [[t('Outcomes observed through'), rolling.outcomes_observed_through]] : []),
          ...(rolling.observation_count !== undefined && rolling.observation_count !== null ? [[t('Observations'), rolling.observation_count]] : []),
          [t('Cost lane'), rolling.cost_bps_per_side ? `${rolling.cost_bps_per_side} bps per side` : ''],
          [t('Rolling head'), hashCell(rolling.head_hash)],
          [t('Base report'), hashCell(rolling.base_report_hash)],
          [t('Base result'), hashCell(rolling.base_result_hash)],
          ...(headStorage[rolling.head_storage] ? [[t('Head storage'), t(headStorage[rolling.head_storage])]] : []),
          ...(claims[rolling.claim] ? [[t('Claim'), t(claims[rolling.claim])]] : []),
          ...(executionBases[rolling.execution_basis] ? [[t('Execution basis'), t(executionBases[rolling.execution_basis])]] : []),
        ])}`;
      }
    }
    // Pages that show no book describe the workspace they are in, never the last book opened.
    if(key === 'context' && !DATA_PAGES.has(app.page)) {
      const w = Data.workspaceFacts() || {}, clocks = Data.clocks();
      return html`<p class="proof-intro">${t('The workspace this page belongs to; nothing on this page is a saved book.')}</p>${kv([[t('Workspace'), Data.workspace()], [t('Manifest'), w.workspace_manifest_hash ? html`<span class="mono">${w.workspace_manifest_hash}</span>` : ''], [t('Execution mode'), coded(w.execution_mode)], [t('Market data through'), clocks.data], [t('Features through'), clocks.feature], [t('Published inputs'), countText(Data.inputs().length, '{n} input version', '{n} input versions')]])}<p class="proof-footnote">${t('Read-only · no authority or review is inferred.')}</p>`;
    }
    const provenance=Data.raw()?.metricProvenance || {};
    const provenanceRows=[['Status',provenance.status],['Report',provenance.report_hash],['Execution ledger',provenance.execution_ledger_hash],['Economic ledger',provenance.economic_ledger_hash],['Window proof',provenance.window_hash],['Selected window',provenance.selected_start || provenance.selected_end ? `${provenance.selected_start || ''} — ${provenance.selected_end || ''}` : null],['Observations',provenance.observation_count],['Return unit',provenance.return_unit],['Annualization sessions',provenance.annualization_sessions_per_year],['Volatility degrees of freedom',provenance.volatility_degrees_of_freedom],['Sharpe cash return per session',provenance.sharpe_cash_return_per_session],['Sortino downside threshold',provenance.sortino_downside_threshold]].filter(([,v])=>v!==undefined && v!==null && v!=='').map(([k,v])=>[t(k),/Report|ledger|proof/.test(k) ? hashCell(v) : v]);
    const rows = key === 'observation' && row ? Object.entries(row).map(([k, v]) => [OBSERVATION_WORDS[k] || k, v]) : key === 'knowledge' ? [[t('Holdings session'), viewSession()], [t('Declared as-of'), s?.input_date || ''], [t('Working data'), app.data], [t('Evidence knowledge date'), t('Not selected')]] : key in Data.metrics() || metricAbsence(key) ? [[t('Published metric'), key], [t('Value'), Number.isFinite(Data.metrics()[key]) ? fmt(Data.metrics()[key], 6) : absentMetric(key)], [t('Unit'), ['annual', 'vol', 'drawdown'].includes(key) ? '%' : 'ratio'],...provenanceRows] : [[t('Workspace'), Data.workspace()], [t('Task'), hashCell(s?.task_id, Infinity)], [t('Receipt'), hashCell(s?.receipt_hash)], [t('Input binding'), hashCell(s?.input_hash)], [t('Holdings session'), viewSession()]];
    // round 90: the book's notice and the read-only line are said once, in the context section, not in every proof section
    return html`${key === 'context' ? html`<p class="proof-intro">${Data.notice() || t('No study selected')}</p>` : ''}${kv(rows.map(([k, v]) => [t(k), v ?? '']))}${key === 'context' ? html`<p class="proof-footnote">${t('Read-only · no authority or review is inferred.')}</p>` : ''}`;
  }
  function chartData() {
    if (Data.performanceMode?.()==='forward') {
      const p=Data.forwardPerformance()?.selected_window_metric_provenance || {}, rows=Data.series();
      const content=rows.length ? html`<p class="caption">${t('Exact owner-supplied net simple returns for {from} — {to}; {n} observations.',{from:p.selected_start || '',to:p.selected_end || '',n:count(p.observation_count || 0)})} ${t('The chart shows published daily returns.')}</p>${table([t('Formation'),t('Entry'),t('Holding end'),t('Net simple return')],rows.slice(-30).map((r)=>tr([r.formation_session,r.entry_session,r.holding_end_session,r.net_simple_return==null ? '' : num(r.net_simple_return * 100,'percent')])),t('Last 30 observations shown; no cumulative performance is calculated here.'))}` : emptyState(t('No published observations'));
      openDialog(t('Realized forward returns'),t('Underlying values'),content,html``,true); return;
    }
    openDialog(t('Chart · published returns'), t('Underlying values'), html`<p class="caption">${Data.raw().seriesBasis}</p>${table([t('Date'), t('Daily %'), t('Index'), t('Benchmark index')], Data.series().slice(-30).map((r) => tr([r.date, fmt(r.daily, 6), fmt(r.value, 6), fmt(r.benchmark, 6)])), t('Last 30 observations shown; export includes all published returns.'))}`, html`${btn(t('Export JSON'), 'export-json')}`, true);
  }
  /* Quick Open names each page by what it holds, in the flyouts' own words where a flyout
   * lists it; the prototype-only routes are not offered. */
  // round 93 (rule 4): every route in ROUTES has its one line; a route without one says nothing, never a placeholder
  const QUICK_WORDS = {overview: 'The workspace, its facts and saved objects', history: 'Every saved object, exact readback', tasks: 'Recorded Task states and the activity feed',
    data: 'The working store: update, membership, receipts', issues: 'The decisions the data owner asks for', inputs: 'Immutable input versions: publish and select', storage: 'Managed roots, retention and cleanup',
    lab: 'A new declaration, or saved work continued', factor: 'Saved Factor studies and curation', foundation: 'Sealed Foundations and their previews', alpha: 'Saved Alpha studies and candidates', 'alpha-compare': 'Two saved Alpha candidates, compared', risk: 'Saved Risk studies', cases: 'Research questions kept as records',
    portfolio: 'A saved book: performance, holdings, sources', compare: 'Two saved books at one holdings date',
    books: 'The saved books whose evidence is read here', evidence: 'The review of one exact book', 'evidence-stream': 'Sources prepared for a book\'s issuers', 'evidence-reading': 'The prepared passages of one book, read as a view', report: 'The report of one book and its delivery', handoff: 'Analyst and CRO handoffs',
    team: 'The conversation and who is in it', 'team-outputs': 'What the session produced: its Tasks, their artifacts and its goals', 'team-evidence': 'The product\'s records about what the session cited', 'team-sessions': 'Retained conversations, read back exactly',
    advanced: 'Governing facts and the bridge to the original product', settings: 'Appearance, language, keys, workspace'};
  const quickWords = (key) => QUICK_WORDS[key] ? t(QUICK_WORDS[key]) : '';
  function quickEntries() {
    return Object.entries(ROUTES).filter(([key]) => !HIDDEN_ROUTES.has(key)).map(([key, v]) => [key, v[1], quickWords(key), v[2]]);
  }
  function bind() {
    // Styles authored by the template, not code from API responses. CSSOM keeps
    // the Host CSP intact without allowing inline scripts or arbitrary HTML.
    const apply = (root) => {
      if (!(root instanceof Element)) return;
      for (const el of [root, ...root.querySelectorAll('[data-ui-style]')]) {
        if (!el.hasAttribute('data-ui-style')) continue;
        const declarations = el.getAttribute('data-ui-style');
        const parser = document.createElement('span');
        parser.style.cssText = declarations;
        for (const name of parser.style) el.style.setProperty(name, parser.style.getPropertyValue(name), parser.style.getPropertyPriority(name));
        el.removeAttribute('data-ui-style');
      }
    };
    new MutationObserver((changes) => changes.forEach((change) => change.addedNodes.forEach(apply))).observe(document.body, {childList: true, subtree: true});
    apply(document.body);
  }
  return {temporalRow, temporalAll, page, existing, studyLobby, savedObjectLink, studyRow, workspaceDialog, workspacePopover, holdingDetail, bookTools, portfolioDetails, bookFactsSections, collaborationRows, comparePage, proofBody, chartData, quickEntries, bind, researchTiming,
    studyFacts, bookWords, metricAbsence, declaredFromReadback, cutoffText, inputState, shortRef, openerWords: (kind) => t(KIND_OPENERS[kind] || 'Open saved object'), recordRow, taskSuccessor, taskStopFacts, taskAttentionFacts, governanceBody, nameOf, truthOf, waiting, needs, dataNeeds, runningRows, runningGroup, decisionRow, decisionRows, decisionGroup, reviewWay, realizationWords};
})();
