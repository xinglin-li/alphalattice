/* Global actions and the event delegation that routes clicks, edits and keys to handlers. */
/* The read-only actions every page shares (the product's own actions are PRODUCT_ACTIONS below). */
Object.assign(ACTIONS, {
  close: () => closeDialog(true),
  'data-reload': () => Data.reload(),
  'page-reload': () => location.reload(), // the render boundary's way on (round 96)
  'code-open': (id) => openCodeRef(id), // a document's link (round 92)
  'code-copy': () => copyText($('#dialog .code-document')?.textContent || ''),
  'quick-open': () => Inspect.openQuick(),
  'quick-pick': (v) => Inspect.commandPick(v),
  'focus-toggle': () => Inspect.toggleFocus(),
  'proof-open': (key) => Inspect.openLens(key),
  'proof-close': () => Inspect.closeLens(true),
  context: () => Window.toggleWorkspace(), // round 81: the popover (Window.bind adds the window's own)
  'research-map': () => Inspect.researchMap(),
  'tools-menu': toolsMenu,
  'tab-more': (key) => tabMore(key), // a tab strip's More (2026-09-22)
  shortcuts: () => Inspect.openShortcuts(),
  'peek-open': () => Inspect.openPeeked(),
  'peek-close': () => Inspect.closePeek(),
  'detail-back': () => Window.back(), // a detail's way one level up (LS5)
  'inspector-tab': (key) => Inspect.openPanelTab(key), // the panel's two tabs (round 92)
  'row-peek': (value) => { const row = (value && $$('#main [data-key]').find((r) => r.dataset.key === value)) || document.activeElement?.closest?.('[data-row], tr'); if (row) { $$('#main [data-held]').forEach((r) => r.removeAttribute('data-held')); row.setAttribute('data-held', ''); Inspect.peek(row); } },
  'step-prev': () => Inspect.step(-1),
  'step-next': () => Inspect.step(1),
  'row-menu': toggleRowMenu,
  'display-menu': (name) => toggleDisplayMenu(name),
  'display-set': (value) => chooseDisplay(value),
  'picker-menu': () => Picker.toggle(),
  'theme-select': (theme) => setTheme(theme),
  appearance: () => toggleTheme(),
  'filter-set': (v) => (String(v).startsWith('evidence-') ? LiveReview : Lobby).setFilter(v),
  'filter-clear': (v) => (String(v).startsWith('evidence-') ? LiveReview : Lobby).clearFilter(v),
  'filter-field': (v) => (String(v).startsWith('evidence-') ? LiveReview : Lobby).addClause(v), // round 95: `+ Filter` chose a field; its values open next
  'lobby-fold': (v) => Lobby.fold(v), // law 136: a group head folds and opens, kept per list
  'lobby-more': (v) => Lobby.more(v), // a group's rows past its first, in place
  'lobby-clear': (v) => Lobby.clearAll(v),
  'experience-settings': experienceSettings,
  'side-toggle': () => Window.toggleSide(),
  'side-group': (key) => Window.toggleGroup(key),
  'inspector-close': () => Window.closeInspector(true),
  facts: () => Inspect.openFacts('', ['facts', '']),
  record: () => Inspect.openRecord(['record', '']),
  go: (page) => navigate(page),
  'facts-open': (id) => openDialogFacts(id) || Inspect.openFacts(id, ['facts-open', id]),
  'locale-set': (lang) => setLocale(lang),
  'research-update-read': () => Settings.rereadUpdate(),
  'usage-reading-read': () => Settings.rereadUsage(),
});

/* Element-attribute handlers (change events on inputs named by a data attribute). */
// One dispatch seam for clicks and automation; no design simulation in product mode.
const READ_ACTIONS = new Set(['portfolio-performance', 'tab-more', 'detail-back', 'close', 'copy-block', 'data-reload', 'page-reload', 'code-open', 'code-copy', 'inspector-tab', 'lab-reader-close', 'review-item-close', 'quick-open', 'quick-pick', 'focus-toggle', 'shortcuts', 'side-toggle', 'side-group', 'inspector-close', 'facts', 'facts-open', 'record', 'copy-link', 'go', 'lab-reader', 'locale-set', 'row-peek', 'peek-open', 'peek-close', 'step-prev', 'step-next', 'proof-open', 'proof-close', 'context', 'research-map', 'tools-menu', 'row-menu', 'display-menu', 'display-set', 'picker-menu', 'theme-select', 'appearance', 'filter-set', 'filter-clear', 'filter-field', 'lobby-fold', 'lobby-more', 'lobby-clear', 'experience-settings', 'section-menu', 'holding', 'holding-page-prev', 'holding-page-next', 'holding-sort', 'portfolio-tab', 'session-prev', 'session-next', 'chart', 'chart-range', 'chart-data', 'history-open', 'history-copy', 'exact', 'research-update-read', 'usage-reading-read']);
const PRODUCT_ACTIONS = {
    'copy-text':(text)=>copyText(text),
    'copy-link':()=>copyLink(),
    'goal-list':()=>LiveGoals.list(),
    'goal-new':()=>LiveGoals.edit(true),
    'goal-edit':()=>LiveGoals.edit(),
    'goal-save':()=>LiveGoals.save(),
    'goal-open':hash=>LiveGoals.open(hash),
    'goal-verify':()=>LiveGoals.verify(),
    'goal-attach':()=>LiveGoals.attach(),
    'goal-attach-save':()=>LiveGoals.attachSave(),
    'goal-note':()=>LiveGoals.note(),
    'goal-note-save':()=>LiveGoals.noteSave(),
    'goal-abandon':()=>LiveGoals.abandon(),
    'goal-abandon-commit':()=>LiveGoals.abandonCommit(),
    'goal-first-use-stop':()=>LiveGoals.stopFirstUse(), // U70: the first use's one Stop
    'activation-confirm':v=>LiveActivation.confirm(v), // U73: a strategy run forward from its book, or stopped
    'activation-commit':()=>LiveActivation.commit(),
    'activation-read':()=>LiveActivation.read(),
    'research-update-enable':()=>Settings.setUpdate(true), // U73: the daily update for every strategy that runs forward
    'goal-first-use-stop-commit':()=>LiveGoals.stopFirstUseCommit(),
    'goal-export':format=>LiveGoals.exportGoal(format),
    'goal-evidence':i=>LiveGoals.evidence(i),
    'goal-reference-retry':id=>LiveGoals.openReference(id, {retry:true}),
    'goal-continue':i=>LiveGoals.continueFrom(i),
    'goal-more':cursor=>LiveGoals.more(cursor),
    'model-open':id=>LiveModels.open(id), // U50: a model's review packet
    'models-list':()=>LiveModels.list(),
    'models-refresh':()=>LiveModels.refresh(),
    'models-page':way=>LiveModels.turn(way),
    'model-confirm':value=>LiveModels.confirm(value), // a person's activation or deactivation, as the Host offers it
    'model-commit':()=>LiveModels.commit(),
    'feature-research-open':key=>LiveFeatureResearch.open(key), // U56: a formula factor's review packet
    'feature-research-list':()=>LiveFeatureResearch.list(),
    'feature-research-refresh':()=>LiveFeatureResearch.refresh(),
    'feature-research-page':v=>LiveFeatureResearch.turn(v), // the Host's page of the listing
    'feature-research-confirm':value=>LiveFeatureResearch.confirm(value), // a person's activation or deactivation, as the Host offers it
    'feature-research-commit':()=>LiveFeatureResearch.commit(),
    'workspace-preview':kind=>LiveWorkspace.preview(kind),
    'workspace-issue':key=>LiveWorkspace.preview('issue',key),
    'workspace-delegate':key=>LiveWorkspace.preview('delegate',key),
    'workspace-revoke-delegation':key=>LiveWorkspace.preview('revoke-delegation',key),
    'workspace-continue':key=>LiveWorkspace.preview('continue',key),
    'workspace-pin':key=>LiveWorkspace.preview('pin',key),
    'workspace-pin-index':key=>LiveWorkspace.preview('pin-index',key),
    'workspace-rebuild-index':key=>LiveWorkspace.preview('rebuild-index',key),
    'workspace-resume-cleanup':key=>LiveWorkspace.preview('resume-cleanup',key),
    'workspace-backups-page':way=>LiveWorkspace.turnBackups(way), // U48: the kept generations' pages
    'workspace-input':key=>LiveWorkspace.selectInput(key),
    'workspace-task':key=>LiveWorkspace.task(key),
    'workspace-more':key=>LiveWorkspace.more(key),
    'workspace-issue-open':token=>LiveWorkspace.openIssue(token), // F2: a data issue's own page
    'workspace-version-open':hash=>LiveWorkspace.openVersion(hash), // F3: an input version's own page, from Research inputs and from Storage
    'workspace-commit':()=>LiveWorkspace.commit(),
    'workspace-fold':()=>LiveWorkspace.fold(),
    'workspace-stage':stage=>LiveWorkspace.inspect(stage),
    'workspace-follow':()=>LiveWorkspace.follow(),
    'workspace-current':()=>LiveWorkspace.current(),
    'study-catalog':()=>LiveStudy.catalog(),
    'study-open':id=>{rememberList(id,(key)=>'experiment:'+key);objectEntry('study:'+id);return LiveStudy.open(id,app.page);},
    'alpha-compare-open':id=>navigate('alpha-compare',{study:id,alpha_left_task:id}), // N2 (law 132): the Alpha study's `Compare with…`
    'study-export':f=>LiveStudy.exportStudy(f),
    'study-factor-detail':id=>LiveStudy.factorDetail(id),
    'study-curate-preview':()=>LiveStudy.confirm('curate'),
    'study-foundation-preview':()=>LiveStudy.previewFoundation(),
    'study-foundation-confirm':()=>LiveStudy.confirm('seal'),
    'study-commit':()=>LiveStudy.commit(),
    'study-alpha-draft':()=>LiveStudy.alphaDraft(),
    'study-foundations':()=>LiveStudy.foundations(),
    'study-foundation-open':id=>{objectEntry('foundation:'+id);return LiveStudy.foundations(id);}, // an object opened from its list: one history entry, `<-` back to the list
    'study-foundation-draft':id=>LiveStudy.foundationDraft(id),
    'study-foundation-export':id=>LiveStudy.exportFoundation(id),
    'study-portfolio-draft':id=>LiveStudy.portfolioDraft(id),
    'study-alpha-comparison-run':()=>LiveStudy.alphaComparisonRun(),
    'study-alpha-comparison-export':()=>LiveStudy.exportAlphaComparison(),
    'study-fold':i=>LiveStudy.fold(i),
    'study-risk-preview':()=>LiveStudy.confirm('link'),
    'study-risk-window-preview':()=>LiveStudy.confirm('link-window'),
    'study-risk-links':id=>LiveStudy.links(id),
    'study-risk-export':id=>LiveStudy.riskExport(id),
    'study-risk-page':v=>LiveStudy.riskPage(v),
    'review-refresh':()=>LiveReview.refresh(),
    'review-current':()=>LiveReview.working(),
    'review-pin':v=>LiveReview.selectPublication(v), // a publication row pins it (round F5)
    'review-collection-page': v => LiveReview.collectionPage(v),
    'review-live-item':v=>LiveReview.setItem(v),
    'review-fold':v=>LiveReview.fold(v),
    'review-source':v=>LiveReview.source(v),
    'review-preview':()=>LiveReview.inspectSources(),
    'review-prepare':()=>LiveReview.confirm('prepare'),
    'review-packet':()=>LiveReview.getBundle('analyst'),
    'review-dossier':()=>LiveReview.getBundle('cro'),
    'review-select':v=>LiveReview.confirm('select',v),
    'review-provider':v=>LiveReview.confirm(v),
    'review-next':v=>LiveReview.next(v),
    'review-submit-preview':()=>LiveReview.confirm('submit'),
    'review-commit':()=>LiveReview.commit(),
    'review-bundle-export':v=>LiveReview.exportBundle(v),
    'review-step':v=>LiveReview.step(v),
    'review-watch':v=>LiveReview.watch(v),
    'review-use-task':v=>LiveReview.useTask(v),
    'review-role':v=>LiveReview.role(v),
    'review-refusal-dismiss':()=>LiveReview.dismissRefusal(),
    'review-reconcile':()=>LiveReview.reconcile(),
    'review-legacy':v=>LiveReview.settleLegacy(v),
    'review-unitless':v=>LiveReview.settleUnitless(v),
    'review-ledger-topic':v=>LiveReview.ledgerTopic(v),
    'review-ledger-group':v=>LiveReview.ledgerGroup(v),
    'review-ledger-page':v=>LiveReview.ledgerPage(v),
    'review-source-page':v=>LiveReview.sourcePage(v), // the Sources page's issuers table (2026-09-22)
    'page-jump':selector=>jumpTo(selector), // a long page's section, brought into view (the user's phase 6 reading)
    'review-answer-open':role=>LiveReview.openAnswer(role), // the answer's editor, opened when the material is read (phase 6 reading)
    'review-documents-page':v=>LiveReview.documentsPage(v), // U57 (A6): the retained documents, by the Host's page
    'review-issuer-page':v=>LiveReview.issuerPage(v), // the Evidence page's issuers table (round F1)
    'review-finding-page':v=>LiveReview.findingPage(v), // the Review page's findings table (E2)
    'review-finding-more':v=>LiveReview.findingMore(v),
    'review-name-risk':v=>LiveReview.nameRisk(v),
    'upgrade-acknowledge':()=>Settings.acknowledge(), 'upgrade-read':()=>Settings.readUpgrade(true), // R1: the upgrade overview, acknowledged by a person
    'storage-cap-set':(value)=>Settings.setStorageCap(value), 'cpu-budget-set':(value)=>Settings.setBudget(value), 'tasks-waiting-set':(value)=>Settings.setWaiting(value), 'studies-verify-all':()=>Settings.verifyAll(), // U36, U37 // the operator's CPU budget (Settings, Workspace) // the CRO's turn names a risk citing a finding's alias (contract 10.7)
    'review-run-page':v=>LiveReview.runPage(v), // the evidence pages' Runs lists (round H4)
    'review-runs-all':()=>LiveReview.runsAll(), // the Overview's Activity past its latest three (E1)
    'review-packet-page':v=>LiveReview.packetPage(v), // the Review page's packet groups (round H4, a wide book)
    'review-unit-page':v=>LiveReview.unitPage(v), // the Sources page's units table (law 92)
    'review-checks-page':v=>LiveReview.checksPage(v), // the Review's checks by issuer, fifty a page
    'report-cite-page':v=>LiveReview.citePage(v), // an issuer section's citations (round H4)
    'review-issuer-docs':v=>LiveReview.issuerDocs(v), // B3: an issuer's documents, opened in place on Sources
    'review-questions':()=>LiveReview.questions(), // item 8: a Report question opens the Review at the open questions
    'review-execution':()=>LiveReview.execution(), // Sources: the preparations and the runs, folded
    'review-issuer-state':(v)=>LiveReview.issuerState(v), // the Overview's coverage key: the Issuers table by what the review reached
    'review-storage':()=>LiveReview.storageAgain(), // Sources: the storage figures, read again
    'review-view-entity':v=>LiveReview.viewFilter('entity',v),
    'review-view-topic':v=>LiveReview.viewFilter('topic',v),
    'review-view-days':v=>LiveReview.viewFilter('days',v),
    'review-span-page':v=>LiveReview.spanPage(v),
    'review-read-part':()=>LiveReview.readPart(),
    'review-continue':()=>LiveReview.continueDialog(),
    'review-read-excerpts':()=>LiveReview.readExcerpts(),
    'review-use-packet':v=>LiveReview.usePacket(v),
    'review-read-cell':v=>LiveReview.readCell(v),
    'review-draft-clear':()=>LiveReview.clearDraft(),
    'review-draft-format':()=>LiveReview.formatDraft(),
    'review-file-open':()=>document.getElementById('reviewReplyFile')?.click(), // the one file field, hidden: its words are this verb's
    'review-work-refresh':()=>LiveReview.readWork(),
    'review-dismiss-work':()=>LiveReview.dismissWork(),
    'review-delivery-reopen':()=>LiveReview.reopenDelivery(),
    'task-review-scene':(id)=>{LiveTasks.close();return LiveReview.openTask(id);},
    'review-export':v=>LiveReview.exportReport(v),
    'review-delivery':()=>LiveReview.assembleDelivery(),
    'review-deliver':()=>LiveReview.deliveryDialog(),
    'review-risk-options':()=>LiveReview.deliveryOptions(),
    'review-delivery-export':v=>LiveReview.exportDelivery(v),
    // The one Refresh: the current page's owner is read again, the workspace facts otherwise.
    'workspace-refresh': () => {
      if (LiveWorkspace.pages.has(app.page)) return LiveWorkspace.refresh();
      if (typeof LiveTeam !== 'undefined' && LiveTeam.pages?.has(app.page)) return LiveTeam.refresh();
      if (LiveReview.pages.has(app.page)) return LiveReview.refresh();
      if (app.page === 'tasks') { LiveActivity.refresh(); return LiveTasks.refresh(); }
      if (LiveStudy.pages.has(app.page)) return app.page === 'foundation' ? LiveStudy.foundations() : LiveStudy.catalog();
      if (app.page === 'lab') LiveResearch.forget();
      return Data.connect();
    },
    'history-more': () => Data.refreshHistory(true),
    tasks: () => LiveTasks.toggle(),
    task: (id) => LiveTasks.open(id),
    'task-refresh': () => { LiveActivity.refresh(); return LiveTasks.refresh(); },
    'task-close': () => LiveTasks.close(),
    'task-page': () => { LiveTasks.close(); navigate('tasks'); },
    'task-scene': () => { LiveTasks.close(); navigate('overview'); },
    'task-update-scene': (id) => { LiveTasks.close(); navigate('data', id ? {update: id} : {}); },
    'task-cancel': (id) => LiveTasks.preview('cancel',id),
    'task-recovery': (id) => LiveTasks.preview('recover',id),
    'task-commit': () => LiveTasks.commit(),
    'task-remedy': (value) => LiveTasks.previewRemedy(value),
    'task-incident-page': (way) => LiveTasks.turnIncidents(way), // U38: a remedy the Host offers for an open incident
    'task-remedy-commit': () => LiveTasks.commitRemedy(),
    'task-replan': (id) => LiveTasks.replanPreview(id), // V615: a Task's re-PLAN as its owner binds it, previewed
    'task-replan-commit': (name) => LiveTasks.replanCommit(name), // then the admission it offers, confirmed
    'task-result': (id) => LiveTasks.openResult(id),
    'feature-definitions':binding=>LiveFeatures.open(binding || LiveResearch.context().input_binding_hash),
    'feature-open-plan':hash=>LiveFeatures.openPlan(hash),
    'feature-operation':value=>LiveFeatures.operation(value),
    'feature-plan':()=>LiveFeatures.plan(),
    'feature-build':()=>LiveFeatures.build(),
    'feature-export':()=>LiveFeatures.exportYaml(),
    'feature-revise':()=>LiveFeatures.revise(),
    'feature-edit-request':()=>LiveFeatures.editRequest(),
    'task-list': () => LiveTasks.close(),
    'activity-refresh': () => LiveActivity.refresh(),
    'activity-open': (key) => LiveActivity.open(key),
    'copy-yaml': () => LiveResearch.copy(),
    'mode': (mode) => LiveResearch.setMode(mode),
    'lab-reader': () => LiveResearch.openReader(),
    'lab-reader-close': () => LiveResearch.closeReader(),
    'review-item-close': () => LiveReview.setItem(''),
    'research-plan': () => LiveResearch.preview(),
    'research-confirm': () => LiveResearch.reviewRun(),
    'research-run': () => LiveResearch.run(),
    'research-continue': (id) => LiveResearch.continueFrom(id),
    'research-portfolio-redraft': (value) => { const [task_id, candidate_id] = JSON.parse(value); return LiveResearch.receiveHandoff({kind: 'alpha_candidate', task_id, candidate_id}); }, // U45: the Alpha study's Portfolio draft, as the refusal names it
    'study-promote': () => LiveStudy.confirm('promote'), // R4
    'research-adopt-shared': () => LiveResearch.adoptShared(),
    'research-adopt-confirm': (id) => LiveResearch.confirmAdoption(id),
    'research-replan-shared': () => LiveResearch.replanShared(),
    'research-shared-dismiss': () => LiveResearch.dismissShared(),
    'activity-follow': (key) => LiveActivity.follow(key),
    'activity-pin': () => LiveActivity.pin(),
    'team-open': (session) => LiveTeam.open(session),
    'team-select': (session) => LiveTeam.select(session),
    'team-actor': (actor) => LiveTeam.showActor(actor),
    'team-member-open': (actor) => LiveTeam.openMember(actor), // U51: a member's row opens its exchanges on the Conversation
    'team-participants-page': (way) => LiveTeam.turnParticipants(way),
    'team-outputs-page': (way) => LiveTeam.turnOutputs(way),
    'team-outputs-older': (which) => LiveTeam.outputsOlder(which),
    'team-outputs-read': () => LiveTeam.outputsRead(),
    'team-fact-kind': (key) => LiveTeam.setFactKind(key),
    'team-event': (event) => LiveTeam.showEvent(event),
    'team-reveal': (event) => LiveTeam.revealExchange(event), // the head's ways into the thread: its awaiting objection, a declared reference's first exchange
    'team-more': () => LiveTeam.more(),
    'team-older': () => LiveTeam.readOlder(),
    'team-fold': (key) => LiveTeam.toggleFold(key),
    'review-view-clear': () => LiveReview.viewClear(), // the reading's chips (round F3)
    'team-replies': (id) => LiveTeam.toggleReplies(id),
    // C2: a clamped exchange opens where it stands
    'team-words': (id) => LiveTeam.toggleWords(id),
    // C3: the new exchanges' pill goes to the newest, where it is
    'team-newest': () => followWaypoint(document.querySelector('#main .team-new-bar')),
    'team-seen': () => LiveTeam.markSeen(),
    'team-resolve': (reference) => LiveTeam.resolve(reference),
    'team-question': (reference) => LiveTeam.chooseQuestion(reference),
    'team-verify': (reference) => LiveTeam.verify(reference),
    'research-inspect-shared': (hash) => LiveResearch.inspectShared(hash),
    'research-start': (kind) => LiveResearch.selectKind(kind),
    'research-input': (key) => LiveResearch.selectInput(key), // the input chip (round 56)
    'research-set': (value) => LiveResearch.setField(...value.split(':').map(Number)), // a property chip's choice
    'research-dates': () => LiveResearch.focusDates(),
    'goal-kind': (value) => LiveGoals.setKind(value),
    'research-replace': (answer) => LiveResearch.answerReplace(answer),
    'research-guide-open': (which) => LiveResearch.guideOpen(which),
    'research-work-dismiss': () => LiveResearch.dismissWork(),
    'research-work-refresh': () => LiveResearch.readWork(),
    'research-guide-foundations': () => LiveResearch.guideRead(true),
    workspace: () => Window.toggleWorkspace(),
    'workspace-how': (view) => LiveViews.workspaceDialog(view === 'choose' ? 'choose' : view),
    'export-html': () => Data.exportStudy('html'),
    'export-json': () => Data.exportStudy('json'),
    'export-series': () => Data.exportStudy('json'),
    'compare-export': () => Data.exportComparison(),
    'portfolio-reports-refresh': () => Data.refreshRiskLinks(Data.subject()?.task_id),
    'portfolio-risk-export': (hash) => LiveStudy.riskExport(hash, Data.subject()?.task_id),
    codex: () => copyText(JSON.stringify(currentContext(), null, 2) + '\n' + location.href),
};
function dispatchAction(name, value) {
  try {
    if (Window.openedBy(name, value)) return Promise.resolve(Window.closeDetail()); // law 149: the press that opened the detail in view closes it
    const action = PRODUCT_ACTIONS[name] || (READ_ACTIONS.has(name) ? ACTIONS[name] : null);
    // a press inside an open detail that opens another object in it keeps the level it left (LS5; Window.trailAfter)
    const mark = name === 'detail-back' ? null : Window.trailMark();
    if (action) return Promise.resolve(action(value)).then((r) => { if (mark) Window.trailAfter(mark); return r; }).catch((error) => notify(error.message));
    return showInfo(t('This action is not offered here'), t('This operation is not part of the workbench; no simulated state was created.'));
  } catch (error) { notify(error.message); }
}

const Events = (() => {
  function onClick(e) {
    const b = e.target.closest('[data-action]');
    if (!b && !e.target.closest('a, button, input, select, textarea, label, summary, .menu')) { // the row opens the object wherever it is clicked (round 58)
      const main = e.target.closest('[data-row]')?.querySelector(':scope > .list-row-main:is(a, button)') || e.target.closest('tbody > tr')?.querySelector('[data-row-press]'); // law 149: a table row is one press too
      if (main) { main.focus({preventScroll: true}); return main.click(); }
      // an exchange of the thread is read where it is pressed -- the innermost, a reply its own -- but a press that ends a selection of its words reads nothing
      const read = e.target.closest('[data-read]');
      if (read && !String(getSelection?.() || '').trim()) { app.pressed = read; return dispatchAction('team-event', read.dataset.read); }
    }
    if (b && !b.disabled && b.getAttribute('aria-disabled') !== 'true') {
      const choice = b.closest('.choice-list');
      if (choice && (b.classList.contains('picker-option') || b.dataset.action?.startsWith('choice-'))) { e.preventDefault(); Choice.press(choice, b); return; } // round 95: the in-place multiple choice
      if (b.classList.contains('picker-option')) { Picker.choose(b); if (b.dataset.action === 'picker-choose') { e.preventDefault(); return; } }
      app.pressed = b; // the element an action came from (a record steps through the list that showed it)
      Window.notePress(b); // inside an open detail or not: a level down, or a new detail
      e.preventDefault(); dispatchAction(b.dataset.action, b.dataset.value); return;
    }
    if (e.target.closest('a[href^="#page="]')) closeDialog();
  }
  function onInput(e) {
    if(e.target.id==='storageCap')return Settings.editStorageCap(e.target.value);
    if(e.target.dataset.featureField)return LiveFeatures.edit(e.target.dataset.featureField,e.target.value);
    if (e.target.matches?.('[data-picker-search-input]')) return Picker.filter(e.target);
    if (e.target.dataset.lobbyQuery!==undefined)return Lobby.query(e.target.dataset.lobbyQuery, e.target.value); // a lobby's search (law 136)
    if (e.target.id==='studyFactorQuery')return LiveStudy.filter(e.target.value);
    if (e.target.dataset.studyRole!==undefined)return LiveStudy.edit('role',e.target.dataset.studyRole,e.target.value);
    if (e.target.dataset.studyReason!==undefined)return LiveStudy.edit('reason',e.target.dataset.studyReason,e.target.value);
    if (e.target.dataset.studyLimit!==undefined)return LiveStudy.edit('limit',e.target.dataset.studyLimit,e.target.checked);
    if (e.target.id==='reviewReply') { CodeEditor.input(e.target); return LiveReview.reply(e.target.value); }
    if (e.target.id==='reviewQuestion') return LiveReview.setQuestion(e.target.value);
    if (e.target.id==='ledgerQuery') return LiveReview.ledgerQuery(e.target.value); // the ledger's search field (round E2)
    if (e.target.id==='sourceQuery') return LiveReview.sourceQuery(e.target.value); // the issuers table's search field (2026-09-22)
    if (e.target.id?.startsWith('evidenceQuery-')) return LiveReview.collectionQuery(e.target.id.slice('evidenceQuery-'.length),e.target.value);
    if (e.target.id==='issuerQuery') return LiveReview.issuerQuery(e.target.value); // the Evidence page's issuers (round F1)
    if (e.target.id==='spanQuery') return LiveReview.spanQuery(e.target.value); // the passages' search field (round E3)
    if (e.target.dataset.continueLimit!==undefined) return LiveReview.continueLimit(e.target.dataset.continueLimit, e.target.value); // a declared limit of a continuation (round E3)
    if ((e.target.id==='yamlEditor' || e.target.dataset.researchField!==undefined)) return LiveResearch.edit(e);
    const handler = ['holdingsQuery', 'observationRange'].includes(e.target.id) ? ON_INPUT[e.target.id] : null;
    if (handler) handler(e);
    if (e.target.id === 'quickInput') Inspect.drawCommands();
  }
  /* A picker's choice reaches the same handlers a native field's change did (round 91): the
   * trigger stands in for the field -- its id, its data attributes -- with the chosen value. */
  function changed(trigger, value) {
    const target = {id: trigger.id, value, dataset: trigger.dataset, checked: false, files: null, type: 'picker'};
    const event = {target, currentTarget: target};
    onInput(event); onChange(event);
  }
  function onChange(e) {
    if(e.target.id==='featureSelection')return LiveFeatures.select(e.target.value);
    if (e.target.dataset?.filterName) { const name = e.target.dataset.filterName, v = Array.isArray(e.target.value) ? e.target.value.join(',') : e.target.value; return (name.startsWith('evidence-') ? LiveReview : Lobby).setFilter(name + ':' + v); }
    if (e.target.id==='workspaceInputFamily')return LiveWorkspace.changed('family',e.target.value);
    if (e.target.dataset.workspaceOption)return LiveWorkspace.changed(e.target.dataset.workspaceOption,e.target.value);
    const el = e.target;
    if (el.id==='studyDecision')return LiveStudy.chooseDecision(el.value);
    if (el.id==='studyHandoffInput')return LiveStudy.target(el.value);
    if (el.id==='studyAlphaComparisonLeft')return LiveStudy.alphaComparisonCandidate('leftCandidate',el.value);
    if (el.id==='studyAlphaComparisonTask')return LiveStudy.alphaComparisonTask(el.value);
    if (el.id==='studyAlphaComparisonCandidate')return LiveStudy.alphaComparisonCandidate('rightCandidate',el.value);
    if (el.id==='reviewBook')return LiveReview.choose(el.value);
    if (el.id==='reviewSavedPublication')return LiveReview.selectPublication(el.value);
    if (el.id==='reviewPreparedTask')return LiveReview.prepared(el.value);
    if (el.id==='reviewReplyFile') { const f=el.files[0]; el.value=''; return LiveReview.file(f).catch(e=>notify(e.message)); } // cleared: the same file chosen again is a choice again
    if (el.id==='reviewRisk')return LiveReview.selectRisk(el.value);
    if (el.id==='reviewComparison')return LiveReview.selectComparison(el.value);
    if (el.id==='expKind') return LiveResearch.selectKind(el.value);
    if (el.id==='expInput') return LiveResearch.selectInput(el.value);
    if (['guideStudy','guideDecision','guideInput','guideFoundation','guideAlpha','guideCandidate'].includes(el.id)) return LiveResearch.guideChange(el.id,el.value);
    const handler = ['holdingsSession', 'compareSelect'].includes(el.id) ? ON_CHANGE[el.id] : null;
    if (handler) handler(e);
    if (el.id === 'opaqueControls') {
      document.body.classList.toggle('opaque-controls', el.checked);
      savePreference('opaqueControls', el.checked);
    }
    if (el.id === 'focusSetting' && el.checked !== Inspect.focus) Inspect.toggleFocus();
    if (el.id === 'wideScrollbars') {
      document.body.classList.toggle('wide-scrollbars', el.checked);
      savePreference('wideScrollbars', el.checked);
    }
    if (el.id === 'reduceFocusEffects') {
      document.body.classList.toggle('reduced-focus', el.checked);
      savePreference('reduceFocusEffects', el.checked);
    }
    if (['opaqueControls', 'wideScrollbars', 'reduceFocusEffects'].includes(el.id)) patchMain();
    if (el.id === 'taskNotices') LiveActivity.setNotices(el.checked);
    if (el.id === 'networkAccess') Settings.setNetwork(el.checked); // the workspace's network control (CLI-15): the owner answers in the row
    if (el.id === 'usageReading') Settings.setUsage(el.checked);
    if (el.id === 'researchUpdate') Settings.setUpdate(el.checked); // U73: the daily research update, for the strategies that run forward
  }
  function onKeydown(e) {
    const dialog = $('#dialog');
    if (e.key === 'Enter' && !e.altKey && !e.ctrlKey && !e.metaKey && e.target.matches?.('[data-read]')) { e.preventDefault(); return dispatchAction('team-event', e.target.dataset.read); } // the focused exchange is read as a press reads it
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && dialog.open && dialog.classList.contains('composer-dialog')) { // round 56: the composer submits
      e.preventDefault();
      return dialog.querySelector('.dialog-foot :is([data-action="case-save"], [data-action="goal-save"])')?.click();
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      if (Inspect.quickOpenActive()) closeDialog();
      else Inspect.openQuick();
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key === '\\' && !dialog.open) { // round 62: the side
      e.preventDefault();
      Window.toggleSide();
      return;
    }
    if (!dialog.open && Window.hotkeys(e)) return; // round 81
    if (Inspect.quickKeydown(e)) return;
    if (Picker.keydown(e)) return;
    if (Choice.keydown(e)) return;
    if (Controls.layerKeys(e)) return;
    if (Controls.keys(e)) return; // round 51: the chords, `/`, `?`
    if (Controls.escape(e)) return; // round 51: one layer per Esc, the held row last
    if (Controls.walk(e)) return;
    if (e.key === 'Tab' && dialog.open) trapTab(e, dialog);
  }
  /* Keyboard focus stays inside the native dialog while it is open. */
  function trapTab(e, dialog) {
    const items = [...dialog.querySelectorAll('button:not([disabled]),a[href],input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]')].filter((x) => x.tabIndex >= 0 && x.getClientRects().length);
    const first = items[0];
    const last = items[items.length - 1];
    if (first && ((e.shiftKey && document.activeElement === first) || (!e.shiftKey && document.activeElement === last))) {
      e.preventDefault();
      (e.shiftKey ? last : first).focus();
    }
  }
  function bind() {
    document.addEventListener('pointerdown', (e) => { if (e.target?.id === 'holdingsSession') Portfolio.rememberSessionPlace(true); }, true);
    document.addEventListener('focusin', (e) => { if (e.target?.id === 'holdingsSession') Portfolio.rememberSessionPlace(); }, true);
    document.addEventListener('click', onClick);
    Inspect.bindHover(); // round 58: hover cards on object links
    // round 55: a row's right-click opens its ··· at the pointer; elsewhere the browser's own menu
    document.addEventListener('contextmenu', (e) => {
      const row = e.target.closest?.('#main [data-row], #main tbody > tr'), root = row?.querySelector('.row-menu');
      if (!root) return;
      e.preventDefault();
      $$('#main [data-held]').forEach((r) => r.removeAttribute('data-held')); row.setAttribute('data-held', '');
      toggleRowMenu(root, {x: e.clientX, y: e.clientY});
    });
    // a filter field (round 56's menu search, round 72's log filter): the rows `data-filter`
    // names inside the closest `data-filter-scope`, by their words; an empty line when none is left
    document.addEventListener('input', (e) => {
      const rows = e.target.dataset?.filter;
      if (!rows) return;
      const q = e.target.value.trim().toLowerCase(), scope = e.target.closest(e.target.dataset.filterScope || '.menu');
      let shown = 0;
      for (const row of scope.querySelectorAll(rows)) { const hit = !q || row.textContent.toLowerCase().includes(q); row.hidden = !hit; shown += hit ? 1 : 0; }
      const empty = scope.querySelector('.menu-empty'); if (empty) empty.hidden = shown > 0;
    });
    // the shortcuts sheet's search (round 51): rows by their words, a group with no row hides
    document.addEventListener('input', (e) => {
      if (!e.target.matches?.('[data-key-filter]')) return;
      const q = e.target.value.trim().toLowerCase(), sheet = e.target.closest('dialog');
      for (const group of sheet.querySelectorAll('[data-group]')) {
        let shown = 0;
        for (const row of group.querySelectorAll('.key-row')) { const hit = !q || row.dataset.words.includes(q) || row.dataset.keys.includes(q); row.hidden = !hit; shown += hit ? 1 : 0; }
        group.hidden = shown === 0;
      }
    });
    document.addEventListener('input', onInput);
    document.addEventListener('change', onChange);
    // a code editor's numbers and painted copy keep its scroll (the scroll does not bubble: captured once here)
    document.addEventListener('scroll', (e) => { if (e.target instanceof Element && e.target.matches('.code-editor > textarea')) CodeEditor.scroll(e.target); }, true);
    // a JSON file dropped on the answer's editor is the same import as its Import a file
    document.addEventListener('dragover', (e) => { if (e.target instanceof Element && e.target.closest('[data-drop-json]')) e.preventDefault(); });
    document.addEventListener('drop', (e) => { const zone = e.target instanceof Element && e.target.closest('[data-drop-json]'); const f = zone && e.dataTransfer?.files?.[0]; if (!f) return; e.preventDefault(); LiveReview.file(f).catch((error) => notify(error.message)); });
    document.addEventListener('keydown', onKeydown);
    // the actions own the state; unavailable controls are re-explained after each interaction
    for (const type of ['click', 'input', 'change']) document.addEventListener(type, () => queueMicrotask(() => Controls.sync()));
    ACTIONS['copy-block'] = (b) => {
      const block = b.closest('.code-block,.tp-json pre');
      copyText(block ? [...block.childNodes].filter((n) => n !== b).map((n) => n.textContent).join('').trim() : '');
    };
    // Buttons that are visually unavailable but not disabled must not fire.
    document.addEventListener('click', (e) => {
      const b = e.target.closest('button[aria-disabled="true"]');
      if (b) {
        e.preventDefault();
        e.stopImmediatePropagation();
      }
    }, true);
    window.addEventListener('hashchange', () => {
      Data.beginNavigation();
      stampEntry();
      hideToast();
      closeDialog();
      readRoute();
      LiveWorkspace.dismissConfirmation();
      if (app.page==='lab')LiveResearch.followRoute();
      const q = hashParams();
      if (['portfolio', 'compare'].includes(app.page) && q.get('book') && (q.get('book') !== Data.subject()?.task_id || (q.get('session') && q.get('session') !== Data.subject()?.session) || (app.page==='portfolio' && (q.get('performance')==='forward' ? 'forward' : 'historical') !== Data.performanceMode()))) {
        Data.openPortfolio(q.get('book'), q.get('session'), app.page, q.get('compare') || null, null, q.get('performance')==='forward' ? 'forward' : 'historical');
        return;
      }
      if (app.page === 'compare' && ((q.get('compare') || '') !== (app.compareOther || '') || !Data.comparisonShown())) {
        Data.compare(q.get('compare') || ''); // back/forward, a pasted link or a date change restores that exact pair at the shown date
        return;
      }
      render();
      // A place the reader has been (Back/Forward) is restored by the page's surface memory;
      // only a page reached for the first time starts at the top.
      if (!Places.get(Places.key())) scrollTo({top: 0, behavior: 'instant'});
    });
    const dialog = $('#dialog');
    dialog.addEventListener('cancel', (e) => { if (dialog.dataset.leaving) return; e.preventDefault(); closeDialog(true); }); // Escape leaves the way the close glyph does
    dialog.addEventListener('close', () => {
      if (!dialog.open) {
        document.body.classList.remove('modal-open');
        LiveResearch.dismissConfirmation(); LiveTasks.dismissConfirmation(); LiveWorkspace.dismissConfirmation();
        LiveReview.dismissConfirmation();
        LiveStudy.dismissConfirmation();
        Dialog.clear();
      }
    });
    // a press outside closes Quick Open and a sheet (panels that float over the page: law 150), never a
    // modal dialog (Escape or its buttons)
    dialog.addEventListener('click', (e) => {
      if (e.target === dialog && (dialog.classList.contains('quick-dialog') || dialog.classList.contains('sheet'))) {
        const r = dialog.getBoundingClientRect();
        if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) closeDialog();
      }
    });
    $('#toastClose').addEventListener('click', hideToast);
  }
  return {bind, changed};
})();
