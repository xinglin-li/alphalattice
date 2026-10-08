/* Settings (phase 9, round 62): the one page for what a person sets once -- the appearance, the
 * language, the reading comfort, the keyboard sheet, the workspace, the governance that used to
 * be the Advanced page. Every section is a kv or a choice of the existing shapes; each choice
 * calls the same owner it always did (setTheme, setLocale, the comfort handlers), so the sun /
 * moon (round 68) and the ? sheet are entries to the same functions, not copies. */
const Settings = (() => {
  /* Round 83: the page is four groups on the form shape (a label, a box of rows: a title, one
   * line, one control) — General (the appearance as a pop-up, the navigation as a segment, the
   * text size as a stepper, the language as a pop-up, the reading comfort as switches, the
   * motion as a detail), Keyboard (the ? sheet's own rows), Workspace, Advanced. */
  const THEME_WORDS = {follow: 'Follow the host', light: 'Light', dark: 'Dark'}, THEME_GLYPHS = {follow: 'auto', light: 'sun', dark: 'moon'};
  const NAV_METAS = {follow: 'The sidebar at 1180 and wider, the rail below', sidebar: 'Words; its edge drags to set the width', rail: 'Glyphs; the drawer peeks out under the pointer', top: 'The top row switches sidebar and rail'};
  function general() {
    const z = Window.zoomLevel();
    // round 94: the list is the pill's width, so the modes are four words; the chosen mode explains itself on the row's line
    const navigation = picker('navPick', NAV_MODES.map(([mode, word]) => ({value: mode, title: t(word)})), {action: 'nav-set', selected: Window.navMode(), label: t('Navigation')});
    const side = html`<div class="ui-segments" role="group" aria-label="${t('Dock side')}">${DOCK_SIDES.map(([key, word]) => segBtn(t(word), 'dock-set', key, Window.dockSide() === key))}</div>`;
    return formGroup(t('General'), html`${formRow({title: t('Appearance'), line: t('Light or dark; following the host takes its choice.'), control: picker('themePick', THEMES.map((theme) => ({value: theme, title: t(THEME_WORDS[theme]), word: t(THEME_WORDS[theme]), glyph: THEME_GLYPHS[theme]})), {action: 'theme-select', selected: app.theme, label: t('Appearance')})})}${formRow({title: t('Dock side'), line: t('Right stands at the window\'s edge beside a Codex or Claude conversation.'), control: side})}${formRow({title: t('Navigation'), line: t(NAV_METAS[Window.navMode()]), control: navigation})}${formRow({title: t('Text size'), line: t('The product scales itself; the host\'s pane may not.'), control: stepper(html`<span class="zoom-note">${z} %</span>`, 'zoom-step', {label: t('Text size'), reset: z === 100 ? '' : t('Reset'), resetAction: 'zoom-set', resetValue: '100'})})}${formRow({title: t('Language'), line: t('The interface language; machine identifiers, YAML and JSON stay as they are.'), control: picker('localePick', [['en', 'English'], ['zh-CN', '中文']], {action: 'locale-set', selected: I18N.locale, label: t('Language')})})}${switchRow('opaqueControls', document.body.classList.contains('opaque-controls'), t('Opaque controls'), t('Floating panels and warnings are solid, not frosted glass.'))}${switchRow('wideScrollbars', document.body.classList.contains('wide-scrollbars'), t('Wider scrollbars'), t('Easier to reach with a pointer.'))}${switchRow('reduceFocusEffects', document.body.classList.contains('reduced-focus'), t('Reduce focus effects'), t('No dimming or fog on the page behind a dialog.'))}${noticeRow()}${formRow({title: t('Reduced motion'), line: t("Your system's reduced-motion preference is respected."), detail: t('follows your system')})}`);
  }
  /* A Task that ends while this page is in the background (the user, 2026-09-25): the viewer's own
   * choice, off until turned on; the browser asks once, and its refusal is said here. */
  function noticeRow() {
    const s = LiveActivity.noticesState();
    const line = s === 'unsupported' ? t('This browser does not offer notifications.') : s === 'denied' ? t("The browser blocks this page's notifications; allow them in its site settings.") : t('In the background, a system notification says a Task ended; the browser asks once.');
    return switchRow('taskNotices', s === 'on', t('Notify when a Task ends'), line, {disabled: s === 'unsupported' || s === 'denied'});
  }
  function keyboard() {
    // the keys are the page's longest list and its least changed: last, closed until asked (the user's phase 6 reading)
    return formGroup(t('Keyboard'), html`<details class="reveal-details settings-keys"><summary>${t('Every key the product answers')}</summary>${Inspect.shortcutsBody()}</details>`, {note: t('The same sheet as ?.')});
  }
  /* The CPU budget (the Evidence line's F1; R12): how much of the machine a book's preparation, a Task's
   * reads and an Alpha study's fits use, never what they compute. Read from its owner when the page opens;
   * a choice writes it through the operation the CLI's `cpu-budget --set` sends, and the row says what the
   * owner answered: a refusal beside the budget last read, which stays. The Task queue (U36) is the same
   * owner's second setting, one a request: how many Tasks may wait behind the running one. */
  let budget = {value: null, refused: ''}, budgetRead = false;
  const repaint = () => (typeof patchMain === 'function' ? patchMain : render)();
  async function readBudget() {
    if (budgetRead) return; budgetRead = true;
    try { budget = {value: await Data.readShared('/api/workspace/cpu-budget', true), refused: ''}; }
    catch (e) { budget = {...budget, refused: String(e?.message || e)}; }
    finally { repaint(); }
  }
  async function setBudget(value, field = 'cpu_budget') {
    try { budget = {value: await Data.post('/api/workspace/cpu-budget', {[field]: value === 'auto' ? 'auto' : Number(value)}), refused: '', field: ''}; }
    catch (e) { budget = {...budget, refused: String(e?.message || e), field}; }
    repaint();
  }
  const setWaiting = (value) => setBudget(value, 'tasks_waiting');
  let storageCap = {value: null, refused: ''}, storageCapRead = false, storageCapDraft = null;
  let storageCapReading = null, storageCapBusy = false, storageCapTicket = 0;
  const editStorageCap = value => { storageCapDraft = value; };
  function readStorageCap(force = false) {
    if (storageCapReading) return storageCapReading;
    if (storageCapBusy || (storageCapRead && !force)) return;
    storageCapRead = true;
    const ticket = ++storageCapTicket;
    storageCapReading = (async () => {
      try {
        const value = await Data.readShared('/api/workspace/storage/cap', true);
        if (ticket === storageCapTicket) storageCap = {...storageCap, value, refused: storageCapDraft === null ? '' : storageCap.refused};
      } catch (e) { if (ticket === storageCapTicket) storageCap = {...storageCap, refused: String(e?.message || e)}; }
      finally { storageCapReading = null; if (app.page === 'settings') repaint(); }
    })();
    return storageCapReading;
  }
  async function setStorageCap(value) {
    if (storageCapBusy) return;
    const chosen = value === 'auto' ? 'auto' : String($('#storageCap')?.value || '').trim();
    const draft = storageCapDraft;
    storageCapBusy = true; ++storageCapTicket;
    try { storageCap = {value: await Data.post('/api/workspace/storage/cap', {storage_cap_bytes: chosen}), refused: ''}; if (storageCapDraft === draft) storageCapDraft = null; }
    catch (e) { storageCap = {...storageCap, refused: String(e?.message || e)}; }
    finally { storageCapBusy = false; repaint(); }
  }
  // V680: the same activity cadence reads changes made through another operator surface.
  const observeStorageCap = () => app.page === 'settings' ? readStorageCap(true) : undefined;
  function storageCapRow() {
    if (!storageCapRead && typeof queueMicrotask === 'function') queueMicrotask(() => void readStorageCap());
    const v = storageCap.value, c = v?.capacity;
    const words = t('Bounds managed writes; raising it preserves retained work.');
    const cause = storageCap.refused ? codeWords(storageCap.refused.split(':')[0]) : '';
    const about = [words, c?.estimate_limit ? t(c.estimate_limit) : '', c?.automatic_basis ? t(c.automatic_basis) : ''].filter(Boolean).join(' ');
    const control = html`<div class="flow">${field(t('Bytes'), 'storageCap', storageCapDraft ?? c?.setting?.cap_bytes ?? '', 'text', '', html`inputmode="numeric" aria-label="${t('Storage cap in bytes')}"`, {inline: true})}${btn(t('Set'), 'storage-cap-set', '', 'button compact')}${btn(t('Auto'), 'storage-cap-set', 'auto', 'button compact')}</div>`;
    const reading = c && c.cap_bytes < 1024 ? html`${count(c.cap_bytes)} ${t('Bytes')}` : v?.display?.cap?.split(' (')[0] || '';
    return formRow({title: hint(t('Storage cap'), about), line: cause ? html`${t('Not set')} · ${cause}` : '', detail: reading, control});
  }
  function budgetRow() {
    const about = t('How much of the machine a book\'s preparation, a Task\'s reads and an Alpha study\'s fits use; never what they compute.');
    if (!budgetRead && typeof queueMicrotask === 'function') queueMicrotask(() => void readBudget());
    const v = budget.value || {}, m = v.machine || {}, n = Number(m.processors) || 0;
    const cause = budget.refused && budget.field !== 'tasks_waiting' ? codeWords(budget.refused.split(':')[0]) : '';
    if (!n) return formRow({title: t('CPU budget'), line: cause ? html`${t('Not read')} · ${cause}` : about}); // reading: the value's slot stays empty (ST7)
    const sizes = [...new Set([1, 2, 4, 8, 12, 16, 24, 32, 48, 64, n])].filter((c) => c <= n).sort((a, b) => a - b);
    const choices = [{value: 'auto', title: t('Auto')}, ...sizes.map((c) => ({value: String(c), title: countText(c, '{n} core', '{n} cores')}))];
    // what the last runs took, each as its owner recorded it (R12); the owner's guidance on choosing is its own words
    const last = v.last_preparation, lastTask = v.last_task, lastFits = v.last_model_fits;
    const ran = [last ? t('The last book ran {u} units at once, {th} threads each.', {u: count(last.units_at_once), th: count(last.threads_per_session)}) : '', lastTask ? t('The last Task read with {n} threads.', {n: count(lastTask.reader_threads)}) : '', lastFits ? t('The last Alpha study fitted with {n} LightGBM threads.', {n: count(lastFits.lightgbm_threads)}) : ''].filter(Boolean).join(' ');
    const title = ran ? hint(t('CPU budget'), ran) : t('CPU budget');
    const load = m.busy_processors == null ? '' : t('{busy} / {n} in use', {busy: count(Math.round(m.busy_processors)), n: count(n)});
    const now = v.a_task_now?.cores ? ` ${t('A Task starting now gets {n}.', {n: countText(v.a_task_now.cores, '{n} core', '{n} cores')})}` : '';
    return formRow({title, line: cause ? html`${t('Not set')} · ${cause}` : html`${now ? now.trim() : about}${infoMark([now ? about : '', v.guidance ? t(v.guidance) : ''].filter(Boolean).join(' '))}`, /* the machine's figure in the line, what the budget is in its (i) (WD4: the first screen of Settings, the phase 6 reading) */ detail: load, control: picker('cpuBudgetPick', choices, {action: 'cpu-budget-set', selected: String(v.cpu_budget), label: t('CPU budget')})});
  }
  /* U36: the Task queue beside the budget -- its places (auto or a number, the owner's reason on the title), how
   * many wait now, and the choice; a run refused because every place is taken names this setting in its words. */
  const WAITING = [1, 2, 4, 8, 16, 32, 64]; // the choices offered; the owner holds the bound (MAXIMUM_TASKS_WAITING)
  function queueRow() {
    const q = budget.value?.task_queue, cause = budget.refused && budget.field === 'tasks_waiting' ? codeWords(budget.refused.split(':')[0]) : '';
    const line = t('How many Tasks may wait behind the running one; a request past the last place is refused before any work.');
    if (!q) return formRow({title: hint(t('Tasks that may wait'), line), line: cause ? html`${t('Not read')} · ${cause}` : ''}); // reading: the value's slot stays empty (ST7); what it is, on its title (WD4)
    const sizes = [...new Set([...WAITING, ...(q.tasks_waiting === 'auto' ? [] : [Number(q.tasks_waiting)])])].sort((a, b) => a - b);
    const choices = [{value: 'auto', title: t('Auto')}, ...sizes.map((c) => ({value: String(c), title: countText(c, '{n} Task', '{n} Tasks')}))];
    return formRow({title: hint(t('Tasks that may wait'), [line, q.reason ? t(q.reason) : ''].filter(Boolean).join(' ')), line: cause ? html`${t('Not set')} · ${cause}` : '', detail: t('{n} waiting of {places}', {n: count(q.waiting_now), places: count(q.places)}), control: picker('tasksWaitingPick', choices, {action: 'tasks-waiting-set', selected: String(q.tasks_waiting), label: t('Tasks that may wait')})});
  }
  /* U37: a whole verification of the saved studies, asked by a person (EXPERIMENT_VERIFY_ALL): one Task reads
   * every saved study's sealed evidence in full; what it finds is said where each study is read and on the
   * upgrade overview. The answer is said in place, with its Task. */
  let sweep = null;
  async function verifyAll() {
    if (sweep?.sending) return;
    sweep = {sending: true}; repaint();
    try { sweep = {answer: await Data.post('/api/experiments/verify-all', {})}; }
    catch (e) { sweep = {refused: String(e?.message || e)}; }
    repaint();
  }
  function sweepRow() {
    const a = sweep?.answer;
    const said = a?.task_id ? html`${t('Admitted as Task {id}', {id: short(a.task_id)})}${a.saved_studies == null ? '' : html` · ${countText(a.saved_studies, '{n} saved study to verify', '{n} saved studies to verify')}`}` : sweep?.refused ? html`${t('Not admitted')} · ${codeWords(sweep.refused.split(':')[0])}` : '';
    const control = a?.task_id ? btn(t('Follow the Task'), 'task', a.task_id, 'button compact') : btn(t(sweep?.sending ? 'Asking' : 'Verify all'), 'studies-verify-all', '', 'button compact');
    return formRow({title: hint(t('Verify saved studies'), t('One Task reads every saved study\'s sealed evidence in full; a study that does not verify says so where it is read.')), line: said || '', control}); // what it does, on its title (WD4: Settings' first screen)
  }
  /* The workspace's network control (CLI-15): whether its sources may be reached and what decides it, read
   * from its owner when the page opens; a person's switch writes it (NETWORK_ACCESS_SET, a person's only),
   * and the row says what the owner answered. The operator's offline switch holds it; its line says why. */
  let network = {value: null, refused: ''}, networkRead = false;
  async function readNetwork() {
    if (networkRead) return; networkRead = true;
    try { network = {value: await Data.readShared('/api/workspace/network', true), refused: ''}; }
    catch (e) { network = {...network, refused: String(e?.message || e)}; }
    finally { repaint(); }
  }
  async function setNetwork(on) {
    try { network = {value: await Data.post('/api/workspace/network', {network_enabled: Boolean(on)}), refused: ''}; }
    catch (e) { network = {...network, refused: String(e?.message || e)}; }
    repaint();
  }
  function networkRow() {
    if (!networkRead && typeof queueMicrotask === 'function') queueMicrotask(() => void readNetwork());
    const v = network.value, cause = network.refused ? codeWords(network.refused.split(':')[0]) : '';
    if (!v) return formRow({title: t('Network access'), line: cause ? html`${t('Not read')} · ${cause}` : t('Whether this workspace may reach its data sources.')}); // reading: no switch until the owner answers (ST7)
    return switchRow('networkAccess', v.network_allowed === true, t('Network access'), cause ? html`${t('Not set')} · ${cause}` : LiveWorkspace.networkWords(v), {disabled: v.decided_by === 'OPERATOR_OFFLINE_SWITCH'});
  }
  // FLOW-1: this person's workspace preference comes from its owner, never an assumed default.
  let usage = {value: null, error: null, read: false}, usageReading = null, usageTicket = 0, usageBusy = false;
  const repaintUsage = () => { if (app.page === 'settings') repaint(); };
  function readUsage(again = false) {
    if (usageReading) return usageReading;
    if (usageBusy || (usage.read && !again)) return Promise.resolve();
    usage.read = true;
    const ticket = ++usageTicket;
    usageReading = (async () => {
      try {
        const value = await Data.readShared('/api/workspace/usage-reading', true);
        if (ticket === usageTicket) usage = {...usage, value, error: null};
      } catch (error) { if (ticket === usageTicket) usage = {...usage, error}; }
      finally { usageReading = null; repaintUsage(); }
    })();
    return usageReading;
  }
  async function setUsage(on) {
    const request = usage.value?.next_requests?.set;
    if (usageBusy || !request || request.usage_reading_enabled !== on || !Data.offers(request.operation)) { repaintUsage(); return; }
    const {operation, ...payload} = request;
    usageBusy = true; ++usageTicket; repaintUsage();
    try { usage = {...usage, value: await Data.post(Data.route(operation), payload), error: null}; }
    catch (error) { usage = {...usage, error}; }
    finally { usageBusy = false; repaintUsage(); }
  }
  const rereadUsage = () => usageReading ? usageReading.then(() => readUsage(true)) : readUsage(true);
  const observeUsage = () => app.page === 'settings' && !usageReading && !usageBusy ? readUsage(true) : undefined;
  function usageRow() {
    if (!usage.read && typeof queueMicrotask === 'function') queueMicrotask(() => void readUsage());
    const title = t('Usage reading'), v = usage.value;
    const failed = usage.error ? notRead(title, usage.error, '', btn(t('Read again'), 'usage-reading-read', '', 'button compact')) : '';
    if (!['READ', 'OFF'].includes(v?.usage_reading)) return html`${failed}${formRow({title, line: t('Not read')})}`;
    return html`${failed}${switchRow('usageReading', v.usage_reading === 'READ', title, t(v.detail), {disabled: usageBusy || !Data.offers(v.next_requests?.set?.operation)})}`;
  }
  /* U73 (LS1, V459): the daily research update, for the strategies that run forward -- the Host lists them
   * (`runs_forward`) and offers turning it on for those alone (`next_requests.enable`) or off (`disable`); a person's
   * switch sends the request it named, and the row says what the owner answered. */
  const UPDATE_STATES = {DISABLED: 'Off', ENABLED_SERVICE_LIFETIME: 'On while this service runs', SETTINGS_REAPPROVAL_REQUIRED: 'Set for an earlier workspace configuration: turn it on again', AUTOMATION_CHECK_FAILED: 'Its last check failed'};
  let update = {value: null, refused: '', read: false};
  let updateReading = null, updateTicket = 0, updateBusy = false;
  function readUpdate(again = false) {
    if (updateReading) return updateReading;
    if (updateBusy || (update.read && !again)) return Promise.resolve();
    update.read = true;
    const ticket = ++updateTicket;
    updateReading = (async () => {
      try {
        const value = await Data.readShared('/api/research-update/automation', true);
        if (ticket === updateTicket) update = {...update, value, refused: '', error: null};
      } catch (e) { if (ticket === updateTicket) update = {...update, refused: String(e?.message || e), error: e}; }
      finally { updateReading = null; repaint(); }
    })();
    return updateReading;
  }
  async function setUpdate(on) {
    const request = update.value?.next_requests?.[on ? 'enable' : 'disable'];
    if (updateBusy || !request || !Data.offers(request.operation)) { repaint(); return; }
    const {operation, ...payload} = request;
    updateBusy = true; ++updateTicket; // an earlier GET cannot replace this person's answer
    try { update = {...update, value: await Data.post(Data.route(operation), payload), refused: '', error: null}; }
    catch (e) { update = {...update, refused: String(e?.body?.failure_code || e?.message || e), error: e}; }
    finally { updateBusy = false; repaint(); }
  }
  // V613/V621: read again after an answer that changes it (an activation, a stop) and whenever a Task moves -- the
  // last value stays shown until the new one arrives
  const rereadUpdate = () => updateReading ? updateReading.then(() => readUpdate(true)) : readUpdate(true);
  // V676: synchronous external activation/schedule changes do not move a Task. Home and
  // Settings share this owner read on the ordinary activity cadence, with no new timer.
  const observeUpdate = () => ['overview', 'settings'].includes(app.page) && !updateReading && !updateBusy ? readUpdate(true) : undefined;
  if (typeof Data !== 'undefined' && Data.followTasks) Data.followTasks(rereadUpdate);
  // the daily update's answer, read once for every page that shows it (Settings' switch, Home's Running forward)
  const dailyUpdate = () => { if (!update.read && typeof queueMicrotask === 'function') queueMicrotask(() => void readUpdate()); return update; };
  const updateState = (v, {brief = false} = {}) => html`${t(UPDATE_STATES[v.status] || '') || codeWords(v.status)}${!brief && v.status === 'ENABLED_SERVICE_LIFETIME' && v.next_due_at ? html` · ${t('next {t}', {t: whenText(v.next_due_at)})}` : ''}`;
  function updateRow() {
    dailyUpdate();
    const v = update.value;
    const title = hint(t('Daily research update'), t('Each day it runs the strategies that run forward, while this service runs; a person activates a reviewed book first.'));
    const failed = update.refused ? notRead(t('Daily research update'), update.error || update.refused, '', btn(t('Read again'), 'research-update-read', '', 'button compact')) : '';
    if (!v) return failed || formRow({title, line: ''});
    // on is running: set for this workspace's configuration (a check that failed is still on); a setting made for an earlier
    // configuration reads off and is turned on again. Only a strategy whose activation is ACTIVE runs forward.
    const forward = (v.runs_forward || []).filter((f) => f.status === 'ACTIVE'), on = ['ENABLED_SERVICE_LIFETIME', 'AUTOMATION_CHECK_FAILED'].includes(v.status);
    const way = on ? v.next_requests?.disable : forward.length ? v.next_requests?.enable : null;
    const state = updateState(v);
    const words = forward.length ? html`${forward.map((f) => codeWords(f.strategy_package_id)).join(' · ')} · ${state}` : t('No strategy runs forward: a person activates a reviewed book first.');
    // on, while a strategy it does not cover runs forward too: the Host offers `enable` for them all
    const covered = new Set(v.settings?.package_ids || []), added = on && v.next_requests?.enable ? forward.filter((f) => !covered.has(f.strategy_package_id)) : [];
    return html`${failed}${switchRow('researchUpdate', on, title, words, {disabled: !way})}${added.length ? formRow({title: t('Also running forward'), line: added.map((f) => codeWords(f.strategy_package_id)).join(' · '), control: btn(t('Include them'), 'research-update-enable', '', 'button compact')}) : ''}`;
  }
  /* A link to one row (`row=<control id>`, the preparation's refusal names the network switch): once the row
   * is drawn it is brought into view and focused, and the address lets the key go */
  let landing = '';
  function land() {
    const el = landing && document.getElementById(landing);
    if (!el) return;
    landing = '';
    el.closest('.form-row')?.scrollIntoView({block: 'center'});
    el.focus({preventScroll: true});
  }
  function workspace() {
    const w = Data.workspaceFacts() || {};
    return formGroup(t('Workspace'), html`${formRow({title: t('Workspace'), detail: Data.workspace()})}${formRow({title: t('Operated by'), detail: t('You and the agent alike')})}${formRow({title: t('Scope'), detail: t('Research only')})}${formRow({title: t('Execution mode'), detail: codeWords(w.execution_mode || '')})}${budgetRow()}${storageCapRow()}${queueRow()}${sweepRow()}${networkRow()}${usageRow()}${updateRow()}${formRow({title: t('Manifest'), detail: w.workspace_manifest_hash ? short(w.workspace_manifest_hash, SHORT.hash) : '', mono: true})}${formRow({title: t('Research input'), detail: app.input || ''})}${formRow({title: t('Where you are'), line: t('The workspace popover: its clocks, versions and ways.'), action: 'workspace'})}${formRow({title: t('Create or open a workspace'), action: 'workspace-how', value: 'create'})}`, {note: t('The local research space this window reads; switching never changes it.')});
  }
  function advanced() {
    const {facts, governed} = LiveViews.governanceBody();
    return html`${formGroup(t('Advanced'), facts, {note: t('Governance and validation: what governs this workspace, apart from everyday research; read, never inferred.')})}${formGroup(t('Governed actions'), governed)}`;
  }
  /* The upgrade overview (U49, R1): each saved study, published review and waiting Task as it stands
   * under the installed code -- the Host's one answer (UPGRADE_OVERVIEW) -- and a person's
   * acknowledgement of what they saw (UPGRADE_ACKNOWLEDGE, refused when the installed set moved since
   * the read). A bare start opens it while the Host says `show`; Home's decision and the workspace
   * popover reach it after. What changed is listed; what stands as it ran is not. */
  let upgrade = {value: null, refused: '', read: false, sending: false};
  async function readUpgrade(again = false) {
    if (upgrade.read && !again) return; upgrade.read = true;
    try { upgrade = {...upgrade, value: await Data.readShared('/api/upgrade'), refused: ''}; }
    catch (e) { upgrade = {...upgrade, refused: String(e?.message || e)}; }
    finally { repaint(); }
  }
  async function acknowledge() {
    const hash = upgrade.value?.next_requests?.acknowledge?.upgrade_set_hash;
    if (!hash || upgrade.sending) return;
    upgrade.sending = true; repaint();
    try {
      upgrade = {...upgrade, value: await Data.post('/api/upgrade/acknowledge', {upgrade_set_hash: hash}), refused: ''};
      notify('Upgrade acknowledged'); // notify says its English source through t()
      void Data.refreshDecisions().then(repaint); // Home's decision leaves with the Host's next answer
    } catch (e) { upgrade = {...upgrade, refused: String(e?.message || e)}; }
    finally { upgrade.sending = false; repaint(); }
  }
  const UPGRADE_MARKS = {CHANGED: ['historical', 'Changed'], INTEGRITY_FAILED: ['failed', 'Did not verify'], UNREADABLE: ['failed', 'Does not read'], CHANGED_SINCE_ADMISSION: ['blocked', 'Cannot resume'], RESUMABLE: ['ready', 'Resumes']};
  const upgradeMark = (state) => UPGRADE_MARKS[state] ? badge(UPGRADE_MARKS[state][0], t(UPGRADE_MARKS[state][1])) : badge('metadata', codeWords(state));
  function studyWhy(x) {
    if (x.state === 'INTEGRITY_FAILED') return html`${t('Its sealed evidence did not verify when the studies were last checked; it refuses when read.')} · ${coded(x.failure_code)}`;
    const calls = x.refresh_numerical_calls; // ST7: a count the plan did not declare is not said
    return html`${t('Its kind\'s implementation changed since it ran: it reads back as recorded, a replay refuses, and a continuation runs its declared work again.')}${calls == null ? '' : html` · ${countText(calls, '{n} numerical call declared', '{n} numerical calls declared')}`}`;
  }
  const REVIEW_WHY = {CHANGED: 'Sealed under an earlier Evidence binding or review policy; it reads back as recorded, and its book can be reviewed again.', UNREADABLE: 'Its analyses do not verify under any Evidence binding this build reads, so it does not read back here.'};
  const upgradeGroup = (title, rows) => rows.length ? html`${groupHead(title, rows.length)}<div class="card-list lines slotted">${rows}</div>` : ''; // what stands as it ran is not listed; an empty group is not drawn (law 81)
  function upgradePage() {
    if (!upgrade.read && typeof queueMicrotask === 'function') queueMicrotask(() => void readUpgrade());
    const v = upgrade.value, lede = t('Each saved study, published review and waiting Task as it stands under the installed code. Reading this runs nothing.');
    const refused = upgrade.refused ? notRead(t('Upgrade overview needs attention'), upgrade.refused, '', btn(t('Read again'), 'upgrade-read', '', 'button compact')) : '';
    if (!v) return html`${objectHead(t('Upgrade'), html`<p class="lede">${lede}</p>`)}${refused || skeleton('rows')}`;
    const ack = v.acknowledged, current = ack && ack.set_hash === v.installed?.set_hash;
    const state = current ? stateLine('verified', {word: html`${t('Acknowledged')} ${when(ack.acknowledged_at)}`}) : v.show ? stateLine('review_pending', {word: t('Not acknowledged')}) : '';
    const primary = v.show && v.next_requests?.acknowledge ? btn(t(upgrade.sending ? 'Acknowledging' : 'Acknowledge'), 'upgrade-acknowledge', '', 'button primary') : '';
    const studies = (v.studies || []).filter((x) => x.state !== 'CURRENT').map((x) => objectRow({lead: tile('lab', 'warning'), name: LiveViews.nameOf({task_id: x.task_id, task_kind: 'research_experiment'}).name, why: studyWhy(x), to: {action: 'task-result', value: x.task_id}}, {key: 'study:' + x.task_id, props: [upgradeMark(x.state)]}));
    const reviews = (v.reviews || []).filter((x) => x.state !== 'CURRENT').map((x) => objectRow({lead: tile('review', 'warning'), name: html`${t('Review')}${x.published_at ? html` · ${when(x.published_at)}` : ''}`, why: t(REVIEW_WHY[x.state] || ''), to: LiveViews.reviewWay(x.review_publication_hash, null)}, {key: 'review:' + x.review_publication_hash, props: [upgradeMark(x.state)]}));
    const tasks = (v.tasks || []).map((x) => { const task = Data.tasks().find((y) => y.task_id === x.task_id) || {task_id: x.task_id, task_kind: x.task_kind}, next = x.state === 'RESUMABLE' ? '' : factsRef(t('Next step'), html`<p>${t('The code installed now cannot resume this Task; cancel it and plan the same work again.')}</p>`); return objectRow({lead: tile('task', 'warning'), name: LiveViews.nameOf(task).name, why: x.state === 'RESUMABLE' ? t('It resumes under the installed code; its recovery view offers the way.') : x.resume_refusal ? coded(x.resume_refusal) : '', to: {action: 'task', value: x.task_id}}, {key: 'task:' + x.task_id, columns: ['state', 'next'], props: [upgradeMark(x.state), next]}); });
    const unread = v.acknowledgement_unreadable ? noteLine(t('The last acknowledgement does not read'), coded(v.acknowledgement_unreadable), 'warning') : '';
    const taskRefusals = (v.refusals || []).map(r=>refusal(r,'warning',{catalog:true,more:html`<p>${t('Task')} ${hashCell(r.task_id)}</p>`,next:prerequisiteWays(r.next_requests)}));
    const none = !studies.length && !reviews.length && !tasks.length;
    const body = none ? taskRefusals.length ? '' : emptyState(t('Every saved study, review and waiting Task stands as it did.'), '', 'page-empty') : html`${upgradeGroup(t('Studies'), studies)}${upgradeGroup(t('Reviews'), reviews)}${upgradeGroup(t('Waiting Tasks'), tasks)}`;
    return html`${objectHead(t('Upgrade'), html`<p class="lede">${lede}</p>`, primary, state)}${refused}${unread}${taskRefusals}<section class="home-groups" aria-label="${t('Upgrade')}">${body}</section>${codeRef(t('Installed identities (JSON)'), v.installed || {})}`;
  }
  function page() {
    const tools = typeof ACTIONS['design-tools'] === 'function' ? formGroup(t('Design tools'), formRow({title: t('Open Design tools'), line: t('Design mode only.'), action: 'design-tools'})) : '';
    const row = hashParams().get('row');
    if (row) { landing = row; replaceHash({row: ''}); }
    if (landing && typeof requestAnimationFrame === 'function') requestAnimationFrame(land);
    return html`${objectHead(t('Settings'), t('What you set once: the appearance, the language, the keys, the workspace.'))}${general()}${workspace()}${advanced()}${tools}${keyboard()}`;
  }
  return {page, setBudget, setWaiting, setStorageCap, editStorageCap, observeStorageCap, verifyAll, setNetwork, setUsage, rereadUsage, observeUsage, setUpdate, upgrade: upgradePage, readUpgrade, acknowledge, dailyUpdate, rereadUpdate, observeUpdate, updateState};
})();
